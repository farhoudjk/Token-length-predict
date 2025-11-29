import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from typing import List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from xgboost import XGBClassifier


# -----------------------
# Config
# -----------------------
DATA_PATHS = [
    "out/out_v2_with_enhanced_features.csv",
    "out/dolly_inference_results_llama2_awq_with_enhanced_features.csv"
]
OUTPUT_DIR = "out/enhanced_clf_results"
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
    if "feature_json" in df_.columns:
        feat_col = "feature_json"
    elif "features_json" in df_.columns:
        feat_col = "features_json"
    else:
        raise ValueError("No 'feature_json' or 'features_json' column found")

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
    for cand in ["prompt_text", "prompt", "input_text"]:
        if cand in df_.columns:
            print(f"Using prompt column: {cand}")
            return df_[cand].astype(str)
    raise ValueError("Could not find a prompt text column")


def create_3class_labels(output_tokens: pd.Series, t1: int, t2: int) -> pd.Series:
    """Create 3-class labels"""
    labels = pd.Series(index=output_tokens.index, dtype=int)
    labels[output_tokens <= t1] = 0
    labels[(output_tokens > t1) & (output_tokens <= t2)] = 1
    labels[output_tokens > t2] = 2
    return labels


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot and save feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f'{model_name}_feature_importance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  📊 Feature importance plot saved to: {plot_path}")

    # Save top features to JSON
    top_features = {
        feature_names[i]: float(importances[i])
        for i in indices
    }
    json_path = os.path.join(OUTPUT_DIR, f'{model_name}_top_features.json')
    with open(json_path, 'w') as f:
        json.dump(top_features, f, indent=2)

    # Print top 10
    print(f"\n  🔝 Top 10 Features for {model_name}:")
    for i, idx in enumerate(indices[:10], 1):
        print(f"     {i}. {feature_names[idx]}: {importances[idx]:.4f}")


# -----------------------
# 1. Load data
# -----------------------
df = read_data(DATA_PATHS)

if OUTPUT_COL not in df.columns:
    raise ValueError(f"Expected output length column '{OUTPUT_COL}' not found")

# Filter out rows with less than 5 output tokens
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

print(f"\n🎯 Total feature count: {len(numeric_cols)} features")
print(f"   (Original had 18 features, enhanced has {len(numeric_cols)} features)")
print(f"   Feature gain: +{len(numeric_cols) - 18} features\n")

# -----------------------
# 4. Create 3-class target
# -----------------------
y = create_3class_labels(df[OUTPUT_COL], THRESHOLD_1, THRESHOLD_2)

print(f"Label distribution (0=short ≤{THRESHOLD_1}, 1=medium {THRESHOLD_1}<x≤{THRESHOLD_2}, 2=long >{THRESHOLD_2}):")
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

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"\nClass weights: {class_weight_dict}")


# -----------------------
# Utility to train, evaluate, and save a model
# -----------------------
def train_and_evaluate_model(model, model_name: str):
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
        target_names=['short', 'medium', 'long']
    )
    print(f"\n{model_name} classification report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} confusion matrix:")
    print("       short  medium  long")
    for i, row_label in enumerate(['short', 'medium', 'long']):
        print(f"{row_label:6s} {cm[i]}")

    # Feature importance
    plot_feature_importance(model, list(numeric_cols), model_name)

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

    print(f"\n✅ Saved {model_name} model to: {model_path}")

    return acc, cm, report


# -----------------------
# 6. Train RandomForest
# -----------------------
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

rf_acc, rf_cm, rf_report = train_and_evaluate_model(rf, "RandomForest_Enhanced")


# -----------------------
# 7. Train XGBoost
# -----------------------
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=3,
    objective="multi:softprob",
    num_class=3,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

xgb_acc, xgb_cm, xgb_report = train_and_evaluate_model(xgb, "XGBoost_Enhanced")


# -----------------------
# 8. Save feature list
# -----------------------
features_path = os.path.join(OUTPUT_DIR, "feature_columns.json")
with open(features_path, "w", encoding="utf-8") as f:
    json.dump(list(numeric_cols), f, indent=2)

print(f"\n✅ Saved feature list to: {features_path}")

# -----------------------
# 9. Final comparison
# -----------------------
print("\n" + "="*60)
print("RESULTS WITH ENHANCED FEATURES")
print("="*60)
print(f"RandomForest_Enhanced: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print(f"XGBoost_Enhanced     : {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print("\nCompare with original features (XGBoost: 75.89%)")
improvement = (max(rf_acc, xgb_acc) - 0.7589) * 100
print(f"Improvement: {improvement:+.2f} percentage points")

if improvement > 0:
    print("\n🎉 Enhanced features improved performance!")
elif improvement > -1:
    print("\n⚠️  Similar performance - features may need more tuning")
else:
    print("\n⚠️  Worse performance - original features were better")

print("\nDone!")
