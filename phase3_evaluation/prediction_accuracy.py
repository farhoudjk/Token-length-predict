"""
prediction_accuracy.py — classifier accuracy against real output lengths
========================================================================
The ablation runs cap each request at its predicted class budget
(512 SHORT / 2048 LONG), so they cannot tell whether a request predicted
SHORT would really have stayed short. This script runs one uncapped
reference pass over the same test pool (sampling as in Phase-1 corpus
creation: temperature 0.7, top_p 0.95, generation up to the context limit)
and scores each classifier against the observed output token counts.

Usage
-----
    python phase3_evaluation/prediction_accuracy.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --clf-dir deploy/deepseek_8b_t1000 --xgb-threshold 1000 \\
        --qsvc-preds qsvc/preds_deepseek_8b.npy --qsvc-threshold 930 \\
        --out qsvc/accuracy_deepseek_8b
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_study as a
import rq4_evaluation as m

SOURCES = [("GSM8K", 0, 170), ("HumanEval", 170, 334), ("WildChat", 334, 504)]


def reference_lengths(model_id, prompts, quantization=None):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, dtype="float16", max_model_len=m.MAX_MODEL_LEN,
              gpu_memory_utilization=0.9, enable_prefix_caching=False,
              quantization=quantization, seed=m.RANDOM_SEED)
    tok = llm.get_tokenizer()
    params = [SamplingParams(temperature=0.7, top_p=0.95, seed=m.RANDOM_SEED + i,
                             max_tokens=m.MAX_MODEL_LEN - len(tok.encode(p)) - 1)
              for i, p in enumerate(prompts)]
    outs = llm.generate(prompts, params)
    return (np.array([len(o.outputs[0].token_ids) for o in outs]),
            np.array([o.outputs[0].finish_reason == "length" for o in outs]))


def score(name, preds, truth):
    tn, fp, fn, tp = confusion_matrix(truth, preds, labels=[0, 1]).ravel()
    return {
        "Classifier":    name,
        "Accuracy":      accuracy_score(truth, preds),
        "Balanced_Acc":  balanced_accuracy_score(truth, preds),
        "F1_LONG":       f1_score(truth, preds, zero_division=0),
        "Recall_SHORT":  tn / max(1, tn + fp),
        "Recall_LONG":   tp / max(1, tp + fn),
        "TN_short_ok":   int(tn), "FP_short_as_long": int(fp),
        "FN_long_as_short": int(fn), "TP_long_ok": int(tp),
    }


def main():
    parser = argparse.ArgumentParser(description="Classifier accuracy against real output lengths")
    parser.add_argument("--model",          type=str, required=True)
    parser.add_argument("--clf-dir",        type=str, required=True)
    parser.add_argument("--xgb-threshold",  type=int, required=True)
    parser.add_argument("--qsvc-preds",     type=str, required=True)
    parser.add_argument("--qsvc-threshold", type=int, required=True)
    parser.add_argument("--quantization",   type=str, default=None)
    parser.add_argument("--out",            type=str, required=True,
                        help="Output prefix: <out>_lengths.csv and <out>_summary.csv")
    args = parser.parse_args()

    prompts = m.get_test_pool(args.model)
    sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    if open(args.qsvc_preds + ".sha256").read().strip() != sha:
        sys.exit("❌ QSVC predictions were made for a different prompt pool")

    xgb  = a.predict(prompts, os.path.join(args.clf_dir, "v13_vectorizer.pkl"),
                     os.path.join(args.clf_dir, "v13_xgb.json")).astype(int)
    qsvc = np.load(args.qsvc_preds).astype(int)

    lengths_csv = args.out + "_lengths.csv"
    if os.path.exists(lengths_csv):
        df = pd.read_csv(lengths_csv)
        print(f"Reusing reference lengths from {lengths_csv}")
    else:
        out_tokens, truncated = reference_lengths(args.model, prompts, args.quantization)
        source = np.empty(len(prompts), dtype=object)
        for name, lo, hi in SOURCES:
            source[lo:hi] = name
        df = pd.DataFrame({"source": source, "output_tokens": out_tokens,
                           "hit_context_limit": truncated,
                           "pred_xgb": xgb, "pred_qsvc": qsvc})
        os.makedirs(os.path.dirname(os.path.abspath(lengths_csv)), exist_ok=True)
        df.to_csv(lengths_csv, index=False)

    rows = []
    for subset, sdf in [("ALL", df)] + list(df.groupby("source", sort=False)):
        t_xgb  = (sdf["output_tokens"] > args.xgb_threshold).astype(int)
        t_qsvc = (sdf["output_tokens"] > args.qsvc_threshold).astype(int)
        for name, preds, truth, thr in [
            ("XGB",  sdf["pred_xgb"],  t_xgb,  args.xgb_threshold),
            ("QSVC", sdf["pred_qsvc"], t_qsvc, args.qsvc_threshold),
            # Same ground truth for both, so the threshold difference is not a factor
            ("XGB@QSVC_thr",  sdf["pred_xgb"],  t_qsvc, args.qsvc_threshold),
            ("QSVC@XGB_thr",  sdf["pred_qsvc"], t_xgb,  args.xgb_threshold),
        ]:
            rows.append({"Subset": subset, "Threshold": thr, "N": len(sdf),
                         "True_LONG%": truth.mean() * 100, **score(name, preds, truth)})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out + "_summary.csv", index=False)

    print(f"\nOutput tokens: median={df['output_tokens'].median():.0f}  "
          f"hit context limit={int(df['hit_context_limit'].sum())}/{len(df)}")
    pd.set_option("display.width", 200)
    print(summary[summary["Subset"] == "ALL"].round(3).to_string(index=False))
    print(f"\n✅ {lengths_csv}  {args.out}_summary.csv")


if __name__ == "__main__":
    main()
