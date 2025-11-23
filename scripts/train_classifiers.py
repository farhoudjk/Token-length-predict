#!/usr/bin/env python3
"""Train classifiers on vLLM output features.

Usage example:
  python3 scripts/train_classifiers.py --csv out/out_v2.csv --models rf,xgb --test-size 0.3

The script:
 - Loads the CSV and detects a token-count column (e.g. `output_tokens`)
 - Maps token counts to 4 buckets (0..3)
 - Builds simple numeric features from available columns (numeric fields) and
   engineered features: prompt length and response length if present
 - Trains RandomForest and XGBoost (if installed) classifiers
 - Evaluates on test split (30% default) and prints MAE (on integer labels),
   accuracy, precision/recall/F1 and confusion matrix
 - Saves models and a CSV summary to `out/models_results/`
"""
import argparse
import csv
import io
import os
import sys
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_csv_strip_nuls(path):
    with open(path, 'rb') as fb:
        data = fb.read().replace(b'\x00', b' ')
    text = data.decode('utf-8', errors='replace')
    return pd.read_csv(io.StringIO(text))


def detect_token_column(df):
    candidates = [
        'output_tokens', 'output_token', 'output_token_count', 'output_tokens_count',
        'output_token_len', 'tokens_out', 'tokens'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any column with 'token' and 'out' or 'output'
    for c in df.columns:
        lc = c.lower()
        if 'token' in lc and ('out' in lc or 'output' in lc):
            return c
    return None


def map_bucket(tok):
    try:
        t = int(float(tok))
    except Exception:
        return None
    if 1 <= t < 9:
        return 0
    if 10 <= t <= 500:
        return 1
    if 501 <= t <= 2000:
        return 2
    if 2001 <= t <= 4096:
        return 3
    return None


def prepare_features(df, target_col):
    df2 = df.copy()
    # engineered features
    text_cols = [c for c in df2.columns if any(k in c.lower() for k in ('prompt', 'input', 'instruction'))]
    resp_cols = [c for c in df2.columns if any(k in c.lower() for k in ('output', 'response', 'reply'))]

    # create prompt_len and response_len if possible
    if text_cols:
        df2['prompt_len'] = df2[text_cols[0]].fillna('').astype(str).apply(len)
    else:
        df2['prompt_len'] = 0
    if resp_cols:
        df2['response_len'] = df2[resp_cols[0]].fillna('').astype(str).apply(len)
    else:
        df2['response_len'] = 0

    # numeric features: pick numeric columns except the target
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # also include any explicitly numeric-looking columns (integers stored as object)
    for c in df2.columns:
        if c in numeric_cols or c in (text_cols + resp_cols):
            continue
        try:
            _ = pd.to_numeric(df2[c].dropna())
            numeric_cols.append(c)
        except Exception:
            pass

    feature_cols = list(dict.fromkeys(numeric_cols + ['prompt_len', 'response_len']))

    X = df2[feature_cols].fillna(0)
    return X, feature_cols


def build_models(requested):
    models = {}
    if 'rf' in requested:
        models['rf'] = RandomForestClassifier(n_estimators=200, random_state=42)
    if 'xgb' in requested:
        try:
            from xgboost import XGBClassifier

            models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200)
        except Exception:
            print('xgboost not available, skipping xgb model')
    return models


def evaluate_and_print(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Model: {name}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  MAE (on integer labels): {mae:.4f}")
    print("  Classification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("  Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("")
    return dict(name=name, accuracy=acc, mae=mae)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', '-c', default='out/out_v2.csv')
    parser.add_argument('--models', default='rf,xgb', help='Comma-separated: rf,xgb')
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--out-dir', default='out/models_results')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print('CSV not found:', args.csv)
        sys.exit(2)

    print('Loading CSV...')
    df = read_csv_strip_nuls(args.csv)

    token_col = detect_token_column(df)
    if token_col is None:
        print('Could not detect token-count column. Please inspect CSV and provide a --token-col option (TODO).')
        sys.exit(2)

    print('Detected token column:', token_col)
    df['bucket'] = df[token_col].apply(map_bucket)
    before = len(df)
    df = df[df['bucket'].notnull()]
    after = len(df)
    print(f'Rows before filtering: {before}; after filtering invalid token rows: {after}')

    X, feature_cols = prepare_features(df, token_col)
    y = df['bucket'].astype(int)

    if X.shape[0] == 0:
        print('No data after filtering; exiting')
        sys.exit(1)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # standardize numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    requested = [m.strip() for m in args.models.split(',') if m.strip()]
    models = build_models(requested)

    results = []
    for name, mdl in models.items():
        print('Training', name)
        mdl.fit(X_train_scaled, y_train)
        y_pred = mdl.predict(X_test_scaled)
        res = evaluate_and_print(name, y_test, y_pred)
        results.append(res)
        # save model + scaler
        joblib.dump({'model': mdl, 'scaler': scaler, 'features': feature_cols}, os.path.join(args.out_dir, f'{name}_model.joblib'))

    # Save summary CSV
    out_csv = os.path.join(args.out_dir, 'summary.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print('Saved summary to', out_csv)


if __name__ == '__main__':
    main()
