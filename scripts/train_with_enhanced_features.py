#!/usr/bin/env python3
import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import numpy as np
from xgboost import XGBClassifier

# -----------------------
# Config
# -----------------------
DATA_PATHS = [
    "data/vllm_mistral_7b_v0_2_dolly15k.csv",
    "data/vllm_mistral_7b_v0_2_mixed_prompts_v2.csv"
]
OUTPUT_DIR = "out/enhanced_clf_results_binary"
THRESHOLD_1 = 300
OUTPUT_COL = "output_tokens"
CLASS_NAMES = ['short', 'long']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def read_data(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            df_temp = pd.read_csv(path)
            print(f"  Loaded {len(df_temp)} rows.")
            dfs.append(df_temp)
    df_ = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined total: {len(df_)} rows.")
    return df_

def parse_features_json_column(df_: pd.DataFrame) -> pd.DataFrame:
    feat_col = "feature_json" if "feature_json" in df_.columns else "features_json"
    print(f"Using feature JSON column: {feat_col}")
    feature_dicts = df_[feat_col].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    return pd.json_normalize(feature_dicts)

def get_prompt_series(df_: pd.DataFrame) -> pd.Series:
    for cand in ["prompt_text", "prompt", "input_text"]:
        if cand in df_.columns:
            return df_[cand].astype(str)
    raise ValueError("Could not find a prompt text column")

def create_binary_labels(output_tokens: pd.Series, t1: int) -> pd.Series:
    return (output_tokens > t1).astype(int)

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    if not hasattr(model, 'feature_importances_'): return
    
    importances = model.feature_importances_
    
    # FIX: Ensure we don't try to plot more features than we actually have
    actual_n = min(len(importances), top_n)
    
    # Sort and get indices of top features
    indices = np.argsort(importances)[::-1][:actual_n]
    
    # Reverse order for horizontal plot (highest at top)
    indices = indices[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(f'Top {actual_n} Feature Importances - {model_name}')
    plt.barh(range(actual_n), importances[indices])
    plt.yticks(range(actual_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_importance.png'), dpi=150)
    plt.close()
    print(f"  📊 Feature importance plot saved for {model_name}")

# -----------------------
# Execution
# -----------------------
df = read_data(DATA_PATHS)
df = df[df[OUTPUT_COL] >= 5].copy()

feat_df = parse_features_json_column(df)
prompt_series = get_prompt_series(df)
feat_df["prompt_char_len"] = prompt_series.str.len()
feat_df["prompt_word_count"] = prompt_series.str.split().str.len()
feat_df["prompt_has_question_mark"] = prompt_series.str.contains(r"\?").astype(int)

numeric_cols = feat_df.select_dtypes(include=["number"]).columns
X = feat_df[numeric_cols].fillna(0.0)
y = create_binary_labels(df[OUTPUT_COL], THRESHOLD_1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def train_and_evaluate_model(model, model_name: str):
    print(f"\n{'='*40}\nTraining: {model_name}\n{'='*40}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print(f"\nReport:\n{classification_report(y_test, y_pred, digits=4, target_names=CLASS_NAMES)}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n       {CLASS_NAMES[0]}  {CLASS_NAMES[1]}")
    for i, label in enumerate(CLASS_NAMES):
        print(f"{label:6s} {cm[i]}")

    plot_feature_importance(model, list(numeric_cols), model_name)
    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{model_name}.pkl"))
    return acc

# 1. RandomForest
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
rf_acc = train_and_evaluate_model(rf, "RandomForest_Binary")

# 2. XGBoost
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1
)
xgb_acc = train_and_evaluate_model(xgb, "XGBoost_Binary")

print(f"\nFinal Comparison: RF: {rf_acc:.4f} | XGB: {xgb_acc:.4f}")