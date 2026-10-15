"""
export_qsvc_features.py — QSVC input features for the ablation test pool
========================================================================
Builds the same test pool as ablation_study.py (rq4_evaluation.get_test_pool)
and writes, per prompt, the 15 raw features the QSVC bundle expects: the
Phase-1 prompt features (phase1_corpus/feature_extractor.py) plus
input_tokens = len(tokenizer.encode(prompt)), as in run_inference_vllm.py.

Run in the main .venv (needs transformers/datasets); the QSVC itself runs in
.venv-quantum via predict_qsvc.py, since its bundle pins different versions.

Usage
-----
    python phase3_evaluation/export_qsvc_features.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --out qsvc/features_deepseek_8b.csv
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase3_evaluation"))
sys.path.insert(0, str(ROOT / "phase1_corpus"))
import rq4_evaluation as m
from feature_extractor import extract_prompt_features


def prompts_sha(prompts) -> str:
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export QSVC features for the ablation test pool")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out",   type=str, required=True)
    args = parser.parse_args()

    prompts = m.get_test_pool(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = []
    for p in prompts:
        f = extract_prompt_features(p)
        f.pop("code_fences")
        f["input_tokens"] = len(tok.encode(p))
        rows.append(f)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    with open(args.out + ".sha256", "w") as fh:
        fh.write(prompts_sha(prompts))
    print(f"✅ {len(rows)} prompts → {args.out}  (sha {prompts_sha(prompts)[:12]})")


if __name__ == "__main__":
    main()
    # The WildChat streaming reader can leave a native thread behind that
    # intermittently aborts interpreter finalization (PyGILState_Release);
    # everything is written by now, so skip finalization.
    sys.stdout.flush()
    os._exit(0)
