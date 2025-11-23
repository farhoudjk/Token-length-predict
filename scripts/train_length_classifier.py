import os
import json
import joblib
import pandas as pd

from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# -----------------------
# Config
# -----------------------
DATA_PATH = "out/out_v2_with_new_features.csv"          # your input CSV
OUTPUT_DIR = "out/clf_results"        # where to save models + metrics
THRESHOLD = 500                       # <= 500 -> short (0), > 500 -> long (1)
OUTPUT_COL = "output_tokens"          # change if your column name differs

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def read_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df_ = pd.read_csv(path)
    print(f"Loaded {len(df_)} rows.")
    return df_


def parse_features_json_column(df_: pd.DataFrame) -> pd.DataFrame:
    # Try to find the correct feature JSON column
    if "feature_json" in df_.columns:
        feat_col = "feature_json"
    elif "features_json" in df_.columns:
        feat_col = "features_json"
    else:
        raise ValueError(
            "No 'feature_json' or 'features_json' column found in the CSV. "
            "Please adjust the script to match your column name."
        )

    print(f"Using feature JSON column: {feat_col}")

    def parse_features_json(s):
        try:
            return json.loads(s)
        except Exception:
            return {}

    feature_dicts = df_[feat_col].apply(parse_features_json)
    feat_df_ = pd.json_normalize(feature_dicts)
    return feat_df_


def get_prompt_series(df_: pd.DataFrame) -> pd.Series:
    # Try to locate a prompt column
    # Adjust if you use a different name (e.g. "prompt", "input_text", etc.)
    for cand in ["prompt_text", "prompt", "input_text"]:
        if cand in df_.columns:
            print(f"Using prompt column: {cand}")
            return df_[cand].astype(str)
    raise ValueError(
        "Could not find a prompt text column. "
        "Expected one of: 'prompt_text', 'prompt', 'input_text'. "
        "Please adjust the script."
    )


# -----------------------
# 1. Load data
# -----------------------
df = read_data(DATA_PATH)

if OUTPUT_COL not in df.columns:
    raise ValueError(
        f"Expected output length column '{OUTPUT_COL}' not found in CSV. "
        "Change OUTPUT_COL at the top of the script to match your file."
    )

# -----------------------
# 2. Parse feature_json into columns
# -----------------------
feat_df = parse_features_json_column(df)

# -----------------------
# 3. Add extra features from prompt text
# -----------------------
prompt_series = get_prompt_series(df)

feat_df["prompt_char_len"] = prompt_series.str.len()
feat_df["prompt_word_count"] = prompt_series.str.split().str.len()
feat_df["prompt_has_question_mark"] = prompt_series.str.contains(r"\?").astype(int)

# Keep only numeric columns
numeric_cols = feat_df.select_dtypes(include=["number"]).columns
X = feat_df[numeric_cols].fillna(0.0)

print(f"Number of numeric feature columns: {len(numeric_cols)}")

# -----------------------
# 4. Create binary target: short vs long
# -----------------------
y = (df[OUTPUT_COL] > THRESHOLD).astype(int)

print("\nLabel distribution (0=short, 1=long):")
print(y.value_counts())

# -----------------------
# 5. Train / test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")


# -----------------------
# Utility to train, evaluate, and save a model
# -----------------------
def train_and_evaluate_model(model, model_name: str):
    print(f"\n==================== {model_name} ====================")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")

    report = classification_report(y_test, y_pred, digits=4)
    print(f"\n{model_name} classification report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)

    # Save metrics
    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    metrics_path = os.path.join(OUTPUT_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {model_name} model to: {model_path}")
    print(f"Saved {model_name} metrics to: {metrics_path}")

    return acc, cm, report


# -----------------------
# 6. Train RandomForest
# -----------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

rf_acc, rf_cm, rf_report = train_and_evaluate_model(rf, "RandomForest")


# -----------------------
# 7. Train XGBoost
# -----------------------
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

xgb_acc, xgb_cm, xgb_report = train_and_evaluate_model(xgb, "XGBoost")


# -----------------------
# 8. Save feature column list once
# -----------------------
features_path = os.path.join(OUTPUT_DIR, "feature_columns.json")
with open(features_path, "w", encoding="utf-8") as f:
    json.dump(list(numeric_cols), f, indent=2)

print(f"\nSaved feature list to: {features_path}")

print("\nDone.")
