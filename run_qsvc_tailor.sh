#!/usr/bin/env bash
# Classical (XGBoost) vs quantum (QSVC) classifier in full TAILOR mode
# (class routing + KV-aware spillover/overflow + SM work-stealing) on
# DeepSeek-R1-Distill-Llama-8B, with NO per-class token cap: every request
# gets max_tokens=4096, so each request's output length is its real length
# and prediction accuracy is measured from the same run.
#
#   XGB : deploy/deepseek_8b_t1000         (threshold 1000)
#   QSVC: $QSVC_DIR/qsvc_model.joblib      (threshold 930 = bundle label_median)
#
# Outputs
#   tailor_results_deepseek_8b_qsvc_uncapped.csv            one row per (rate, method)
#                                                           incl. Accuracy / Balanced_Acc
#   tailor_results_deepseek_8b_qsvc_uncapped_requests/      per-request tokens, pred, server
#
# Usage:  nohup ./run_qsvc_tailor.sh > logs/qsvc_tailor.log 2>&1 &
# Stop:   kill <this script's pid>; then
#         pkill -9 -f "rq4_evaluation.py --worker"; pkill -9 -f "VLLM::EngineCore"
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs qsvc

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
QSVC_DIR="${QSVC_DIR:-/root/Quantum}"
FEATS=qsvc/features_deepseek_8b.csv
PREDS=qsvc/preds_deepseek_8b.npy
OUT=tailor_results_deepseek_8b_qsvc_uncapped.csv
RATES=(--rate-start 8 --rate-step 16 --rate-end 72)

# Same budget for every request; prompts are truncated to 2000 tokens, so the
# context must hold 2000 + 4096.
export TAILOR_MAX_TOKENS=4096
export TAILOR_MAX_MODEL_LEN=6144

echo "[$(date +%F\ %T)] ── 1/3 QSVC features (.venv)"
.venv/bin/python phase3_evaluation/export_qsvc_features.py --model "$MODEL" --out "$FEATS"

echo "[$(date +%F\ %T)] ── 2/3 QSVC predictions (.venv-quantum)"
.venv-quantum/bin/python phase3_evaluation/predict_qsvc.py \
    --qsvc-dir "$QSVC_DIR" --features "$FEATS" --out "$PREDS"

echo "[$(date +%F\ %T)] ── 3/3 full TAILOR: TAILOR (XGB) vs TAILOR_QSVC, max_tokens=$TAILOR_MAX_TOKENS"
.venv/bin/python phase3_evaluation/ablation_study.py --model "$MODEL" "${RATES[@]}" \
    --clf-dir deploy/deepseek_8b_t1000 --instance-cfg instance_configs.json \
    --variants TAILOR,TAILOR_QSVC --qsvc-preds "$PREDS" --qsvc-threshold 930 --out "$OUT"

echo "[$(date +%F\ %T)] ALL DONE → $OUT"
