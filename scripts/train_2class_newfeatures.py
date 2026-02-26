#!/usr/bin/env python3
"""
Final training script for output token length prediction (2026 version).
Includes enhanced prompt features and diagnostics.
"""

import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

from typing import List

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import numpy as np

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

AVAILABLE_DATASETS = {
    "llama-3.1-8b-dolly":      "data/vllm_llama_3_1_8b_dolly15k.csv",
    "llama-3.1-8b-mixed":      "data/vllm_llama_3_1_8b_mixed_prompts_v2.csv",
    "mistral-7b-dolly":        "data/vllm_mistral_7b_v0_2_dolly15k.csv",
    "mistral-7b-mixed":        "data/vllm_mistral_7b_v0_2_mixed_prompts_v2.csv",
    "deepseek-8b-v2":          "data/vllm_deepseek_8b_mixed_prompts_v2.csv",
    "deepseek-8b-v234":        "data/vllm_deepseek_8b_mixed_prompts_v234.csv",
    "deepseek-r1-llama-8b":    "data/vllm_deepseek_r1_distill_llama_8b_dolly15k.csv",
    "deepseek-r1-qwen-14b":    "data/vllm_deepseek_r1_qwen_14b_mixed_prompts_v2.csv",
    "llama2-7b-awq":           "data/vllm_llama2_7b_awqmarlin_dolly15k.csv",
    "llama2-13b-dolly":        "data/vllm_llama2_13b_dolly15k.csv",
}

SELECTED_MODELS = [
   # "llama-3.1-8b-dolly",
   # "llama-3.1-8b-mixed",
    "mistral-7b-dolly",
    "mistral-7b-mixed",
    # Add more here if needed
]

DATA_PATHS = [AVAILABLE_DATASETS[k] for k in SELECTED_MODELS 
              if k in AVAILABLE_DATASETS and os.path.exists(AVAILABLE_DATASETS[k])]

if not DATA_PATHS:
    raise ValueError("No valid data paths selected or found.")

print(f"Training on {len(DATA_PATHS)} files:")
for p in DATA_PATHS:
    print(f"  - {p}")

OUTPUT_DIR = "out/enhanced_clf_results_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_COL = "output_tokens"          # ← Confirm this is correct!
MIN_OUTPUT_TOKENS = 5                 # Set to 0 to disable filtering
USE_CLASS_WEIGHT = True               # Toggle balanced weights
USE_3CLASS = False                     # False → 2-class (short vs long/medium)
VAL_SPLIT = 0.15
CV_FOLDS = 5
RANDOM_STATE = 42

if USE_3CLASS:
    THRESHOLD_1 = 500
    THRESHOLD_2 = 2000
    CLASS_NAMES = ['short', 'medium', 'long']
else:
    THRESHOLD_1 = 500
    THRESHOLD_2 = None
    CLASS_NAMES = ['short', 'long/medium']

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────

def read_and_combine_data(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        print(f"Loading: {path}")
        df_temp = pd.read_csv(path, low_memory=False)
        print(f"  → {len(df_temp):,} rows")
        dfs.append(df_temp)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined total: {len(df):,} rows")
    return df


def extract_features_from_json(df: pd.DataFrame) -> pd.DataFrame:
    json_col = next((col for col in ["feature_json", "features_json"] if col in df.columns), None)
    if json_col is None:
        raise ValueError("No JSON feature column found")
    print(f"Parsing JSON from: {json_col}")

    def safe_load(x):
        if pd.isna(x): return {}
        try: return json.loads(x)
        except: return {}

    return pd.json_normalize(df[json_col].apply(safe_load))


def add_prompt_derived_features(df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    prompt_col = next((c for c in ["prompt_text", "prompt", "input_text", "instruction"] if c in df.columns), None)
    if prompt_col is None:
        print("⚠️ No prompt text column found — skipping text-derived features")
        return feat_df

    prompts = df[prompt_col].astype(str).fillna("")
    lower = prompts.str.lower()

    feat_df["prompt_char_len"]         = prompts.str.len()
    feat_df["prompt_word_count"]       = prompts.str.split().str.len()
    feat_df["prompt_unique_ratio"]     = prompts.apply(lambda x: len(set(x.split())) / (len(x.split()) + 1e-6))
    feat_df["prompt_has_question"]     = prompts.str.contains(r"\?").astype(int)
    feat_df["prompt_has_exclam"]       = prompts.str.contains(r"\!").astype(int)
    feat_df["prompt_sentence_count"]   = prompts.apply(lambda x: max(1, len(re.findall(r'[.!?]', x))))
    feat_df["avg_word_length"]         = feat_df["prompt_char_len"] / (feat_df["prompt_word_count"] + 1e-6)

    # Instruction-style signals (very predictive!)
    feat_df["is_step_by_step"]         = lower.str.contains("step by step|think step by step|reason step by step").astype(int)
    feat_df["is_chain_of_thought"]     = lower.str.contains("let's think|chain of thought|reasoning").astype(int)
    feat_df["is_detailed"]             = lower.str.contains("detailed|comprehensive|in-depth|exhaustive|long answer").astype(int)
    feat_df["is_concise"]              = lower.str.contains("brief|short|concise|one sentence").astype(int)
    feat_df["has_role_play"]           = lower.str.contains("you are|act as|pretend|role").astype(int)
    feat_df["has_example"]             = lower.str.contains("example|for instance|such as").astype(int)

    print("Added prompt-derived features:", [c for c in feat_df.columns if c.startswith("prompt_") or c in ["is_","has_"]])

    return feat_df


def create_labels(series: pd.Series) -> pd.Series:
    if USE_3CLASS:
        labels = pd.Series(-1, index=series.index, dtype=int)
        labels[series <= THRESHOLD_1] = 0
        labels[(series > THRESHOLD_1) & (series <= THRESHOLD_2)] = 1
        labels[series > THRESHOLD_2] = 2
    else:
        labels = (series > THRESHOLD_1).astype(int)
    return labels


def plot_and_save_importance(model, feature_names: List[str], model_name: str, top_n=15):
    if not hasattr(model, 'feature_importances_'):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Features – {model_name}")
    plt.barh(range(top_n), imp[idx])
    plt.yticks(range(top_n), [feature_names[i] for i in idx])
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name}_importance.png")
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved importance plot: {path}")
    top_dict = {feature_names[i]: float(imp[i]) for i in idx}
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_top_features.json"), 'w') as f:
        json.dump(top_dict, f, indent=2)


def train_evaluate_save(model, name: str, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    print(f"\n{'═'*70}\n{name}\n{'═'*70}")

    # CV without early stopping
    cv_params = model.get_params()
    cv_params.pop('early_stopping_rounds', None)
    cv_model = model.__class__(**cv_params)

    cv = cross_validate(
        cv_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=['accuracy', 'f1_macro'],
        n_jobs=-1
    )
    print(f"CV Accuracy:  {cv['test_accuracy'].mean():.4f} ± {cv['test_accuracy'].std():.4f}")
    print(f"CV Macro-F1:  {cv['test_f1_macro'].mean():.4f} ± {cv['test_f1_macro'].std():.4f}")

    # Final fit
    fit_params = {}
    if name == "XGBoost" and X_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["verbose"] = False
    elif name == "CatBoost" and X_val is not None:
        fit_params["eval_set"] = (X_val, y_val)

    print("Training final model...")
    model.fit(X_train, y_train, **fit_params)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Macro-F1:  {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(" " * 8 + "  ".join(f"{n:>6}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:8}" + "  ".join(f"{v:6}" for v in row))

    plot_and_save_importance(model, X_train.columns.tolist(), name)

    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{name}.pkl"))
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES
    }
    with open(os.path.join(OUTPUT_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return acc, f1_macro


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

df = read_and_combine_data(DATA_PATHS)

print("\nDataFrame columns:", df.columns.tolist())
print("\nOutput column stats:\n", df[OUTPUT_COL].describe())
print("\nOutput tokens sample (first 10):\n", df[OUTPUT_COL].head(10).tolist())

df = df[df[OUTPUT_COL] >= MIN_OUTPUT_TOKENS].copy()
print(f"After ≥{MIN_OUTPUT_TOKENS} tokens filter: {len(df):,} rows")

feat_df = extract_features_from_json(df)
print("\nExtracted feature columns:", feat_df.columns.tolist())

feat_df = add_prompt_derived_features(df, feat_df)
print("\nFinal feature columns:", feat_df.columns.tolist())

X = feat_df.select_dtypes(include=np.number).fillna(0)
print(f"Numeric features used: {X.shape[1]} columns")
print("Any NaN remaining in X?", X.isna().sum().sum())

y = create_labels(df[OUTPUT_COL])
print("\nClass distribution (%):")
print(y.value_counts(normalize=True).sort_index() * 100)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ─── Models ─────────────────────────────────────

models = [
    (RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced' if USE_CLASS_WEIGHT else None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ), "RandomForest"),

    (XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        gamma=0.5,
        min_child_weight=3,
        objective="multi:softprob" if USE_3CLASS else "binary:logistic",
        num_class=3 if USE_3CLASS else 1,
        eval_metric="mlogloss" if USE_3CLASS else "logloss",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ), "XGBoost"),

    (CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.04,
        auto_class_weights='Balanced' if USE_CLASS_WEIGHT else None,
        verbose=100,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=40
    ), "CatBoost"),
]

results = {}
for model, name in models:
    acc, f1 = train_evaluate_save(
        model, name,
        X_train, y_train, X_test, y_test,
        X_val=X_val, y_val=y_val
    )
    results[name] = {"acc": acc, "f1_macro": f1}

print("\n" + "═"*70)
print("FINAL SUMMARY")
print("═"*70)
for name, res in results.items():
    print(f"{name:12} | Acc: {res['acc']:.4f} | Macro-F1: {res['f1_macro']:.4f}")

print(f"\nAll results saved in: {OUTPUT_DIR}")
print("Done!")