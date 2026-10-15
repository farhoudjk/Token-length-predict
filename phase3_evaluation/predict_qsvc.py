"""
predict_qsvc.py — QSVC predictions for the ablation test pool
=============================================================
Run in .venv-quantum (the bundle refuses to load unless qiskit, scikit-learn,
numpy, pandas, joblib, threadpoolctl and scipy match its training versions).

Reads the features from export_qsvc_features.py and writes predictions in the
TAILOR convention (0 = SHORT, 1 = LONG; QSVC −1 → 0, +1 → 1) plus the raw
decision scores. The prompt-pool hash is copied alongside so ablation_study.py
can check the predictions belong to the prompts it is about to send.

Usage
-----
    .venv-quantum/bin/python phase3_evaluation/predict_qsvc.py \\
        --qsvc-dir /root/Quantum --features qsvc/features_deepseek_8b.csv \\
        --out qsvc/preds_deepseek_8b.npy
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="QSVC predictions for the ablation test pool")
    parser.add_argument("--qsvc-dir", type=str, default="/root/Quantum",
                        help="Directory holding qsvc_predictor.py + qsvc_model.joblib")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out",      type=str, required=True)
    parser.add_argument("--n-jobs",   type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    sys.path.insert(0, args.qsvc_dir)
    from qsvc_predictor import QSVCModel

    model = QSVCModel.load(os.path.join(args.qsvc_dir, "qsvc_model.joblib"), n_jobs=args.n_jobs)
    X = pd.read_csv(args.features)[model.feature_names_in_.tolist()]

    t0 = time.time()
    res = model.predict_with_scores(X)
    preds = (res["predicted_label"].to_numpy() == 1).astype(int)
    print(f"   {len(preds)} prompts in {time.time() - t0:.1f}s  "
          f"SHORT={int((preds == 0).sum())}  LONG={int((preds == 1).sum())}  "
          f"(label_median={model.label_median_})")

    np.save(args.out, preds)
    res.to_csv(os.path.splitext(args.out)[0] + "_scores.csv", index=False)
    shutil.copy(args.features + ".sha256", args.out + ".sha256")
    print(f"✅ {args.out}")


if __name__ == "__main__":
    main()
