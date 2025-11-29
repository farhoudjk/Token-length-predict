import os
import json
import joblib
import pandas as pd

from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# -----------------------
# Config
# -----------------------
DATA_PATHS = [
    "out/out_v2_with_new_features.csv",
    "out/dolly_inference_results_llama2_awq_with_new_features.csv"
]
OUTPUT_DIR = "out/clf_3class_results"  # different directory to avoid overwriting
THRESHOLD_1 = 500    # <= 500 -> short (0)
THRESHOLD_2 = 1000   # 500 < x <= 1000 -> medium (1), > 1000 -> long (2)
OUTPUT_COL = "output_tokens"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def read_data(paths: List[str]) -> pd.DataFrame:
    """Load and combine multiple CSV files."""
    dfs = []
    for path in paths:
        print(f"Loading data from: {path}")
        df_temp = pd.read_csv(path)
        print(f"  Loaded {len(df_temp)} rows.")
        dfs.append(df_temp)

    df_ = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined total: {len(df_)} rows from {len(paths)} files.")
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
    for cand in ["prompt_text", "prompt", "input_text"]:
        if cand in df_.columns:
            print(f"Using prompt column: {cand}")
            return df_[cand].astype(str)
    raise ValueError(
        "Could not find a prompt text column. "
        "Expected one of: 'prompt_text', 'prompt', 'input_text'. "
        "Please adjust the script."
    )


def create_3class_labels(output_tokens: pd.Series, t1: int, t2: int) -> pd.Series:
    """
    Create 3-class labels:
      0 = short (tokens <= t1)
      1 = medium (t1 < tokens <= t2)
      2 = long (tokens > t2)
    """
    labels = pd.Series(index=output_tokens.index, dtype=int)
    labels[output_tokens <= t1] = 0
    labels[(output_tokens > t1) & (output_tokens <= t2)] = 1
    labels[output_tokens > t2] = 2
    return labels


# -----------------------
# 1. Load data
# -----------------------
df = read_data(DATA_PATHS)

if OUTPUT_COL not in df.columns:
    raise ValueError(
        f"Expected output length column '{OUTPUT_COL}' not found in CSV. "
        "Change OUTPUT_COL at the top of the script to match your file."
    )

# Filter out rows with less than 5 output tokens (likely errors or incomplete responses)
print(f"\nBefore filtering: {len(df)} rows")
rows_before = len(df)
df = df[df[OUTPUT_COL] >= 5].copy()
rows_after = len(df)
print(f"After filtering (output_tokens >= 5): {rows_after} rows")
print(f"Removed {rows_before - rows_after} rows with < 5 tokens\n")

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
# 4. Create 3-class target
# -----------------------
y = create_3class_labels(df[OUTPUT_COL], THRESHOLD_1, THRESHOLD_2)

print(f"\nLabel distribution (0=short ≤{THRESHOLD_1}, 1=medium {THRESHOLD_1}<x≤{THRESHOLD_2}, 2=long >{THRESHOLD_2}):")
print(y.value_counts().sort_index())

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
# Compute class weights for imbalanced classes
# -----------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"\nClass weights (to handle imbalance): {class_weight_dict}")


# -----------------------
# Utility to train, evaluate, and save a model
# -----------------------
def train_and_evaluate_model(model, model_name: str):
    print(f"\n==================== {model_name} ====================")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
        target_names=['short', 'medium', 'long']
    )
    print(f"\n{model_name} classification report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} confusion matrix (rows=true, cols=pred):")
    print("       short  medium  long")
    for i, row_label in enumerate(['short', 'medium', 'long']):
        print(f"{row_label:6s} {cm[i]}")

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
# 6. Train RandomForest (with class weights)
# -----------------------
rf = RandomForestClassifier(
    n_estimators=500,           # increased from 300
    max_depth=None,
    min_samples_split=5,        # tuned parameter
    min_samples_leaf=2,         # tuned parameter
    class_weight='balanced',    # handle class imbalance
    random_state=42,
    n_jobs=-1,
)

rf_acc, rf_cm, rf_report = train_and_evaluate_model(rf, "RandomForest")


# -----------------------
# 7. Train XGBoost (with class weights)
# -----------------------
xgb = XGBClassifier(
    n_estimators=500,            # increased from 400
    max_depth=10,                # increased from 8
    learning_rate=0.03,          # decreased for better convergence
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,                     # regularization
    min_child_weight=3,          # handle imbalance
    objective="multi:softprob",
    num_class=3,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=class_weights[1]/class_weights[0],  # handle imbalance
)

xgb_acc, xgb_cm, xgb_report = train_and_evaluate_model(xgb, "XGBoost")


# -----------------------
# 8. Train LightGBM (with class weights)
# -----------------------
lgbm = LGBMClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.03,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',    # handle class imbalance
    objective='multiclass',
    num_class=3,
    n_jobs=-1,
    random_state=42,
    verbose=-1,                 # suppress warnings
)

lgbm_acc, lgbm_cm, lgbm_report = train_and_evaluate_model(lgbm, "LightGBM")


# -----------------------
# 9. Train CatBoost (with class weights)
# -----------------------
catboost = CatBoostClassifier(
    iterations=500,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=3,              # regularization
    class_weights=list(class_weights),  # handle imbalance
    loss_function='MultiClass',
    random_seed=42,
    verbose=False,              # suppress output
    thread_count=-1,
)

catboost_acc, catboost_cm, catboost_report = train_and_evaluate_model(catboost, "CatBoost")


# -----------------------
# 10. Train Ensemble (Voting Classifier)
# -----------------------
print("\n==================== Ensemble (Voting) ====================")
print("Creating ensemble of RandomForest, XGBoost, LightGBM, and CatBoost...")

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('catboost', catboost),
    ],
    voting='soft',  # use probability voting
    n_jobs=-1,
)

ensemble_acc, ensemble_cm, ensemble_report = train_and_evaluate_model(ensemble, "Ensemble")


# -----------------------
# 11. Save feature column list once
# -----------------------
features_path = os.path.join(OUTPUT_DIR, "feature_columns.json")
with open(features_path, "w", encoding="utf-8") as f:
    json.dump(list(numeric_cols), f, indent=2)

print(f"\nSaved feature list to: {features_path}")

# -----------------------
# 12. Compare all models
# -----------------------
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
results = [
    ("RandomForest", rf_acc),
    ("XGBoost", xgb_acc),
    ("LightGBM", lgbm_acc),
    ("CatBoost", catboost_acc),
    ("Ensemble", ensemble_acc),
]

for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {acc:.4f} ({acc*100:.2f}%)")

best_model = max(results, key=lambda x: x[1])
print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]*100:.2f}% accuracy")

print("\nDone.")
