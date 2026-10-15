#!/usr/bin/env bash
# Classical (XGBoost) vs quantum (QSVC) classifier under pure class-based
# routing (ABL_NO_LOAD: SHORT → SHORT server, LONG → LONG server; no KV/load
# scores, spillover, overflow or SM stealing) on DeepSeek-R1-Distill-Llama-8B.
#
#   XGB : deploy/deepseek_8b_t1000         (threshold 1000)
#   QSVC: $QSVC_DIR/qsvc_model.joblib      (threshold 930 = bundle label_median)
#
# Same prompts, Poisson arrivals and M_Seq (instance_configs.json, 300/128 —
# the profile used for the earlier ablation_results_deepseek_8b.csv run).
#
# Usage:  nohup ./run_qsvc_ablation.sh > logs/qsvc_ablation.log 2>&1 &
# Stop:   kill <this script's pid>; then
#         pkill -9 -f "rq4_evaluation.py --worker"; pkill -9 -f "VLLM::EngineCore"
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs qsvc

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
QSVC_DIR="${QSVC_DIR:-/root/Quantum}"
FEATS=qsvc/features_deepseek_8b.csv
PREDS=qsvc/preds_deepseek_8b.npy
OUT=ablation_results_deepseek_8b_qsvc.csv
RATES=(--rate-start 8 --rate-step 16 --rate-end 72)

echo "[$(date +%F\ %T)] ── 1/3 QSVC features (.venv)"
.venv/bin/python phase3_evaluation/export_qsvc_features.py --model "$MODEL" --out "$FEATS"

echo "[$(date +%F\ %T)] ── 2/3 QSVC predictions (.venv-quantum)"
.venv-quantum/bin/python phase3_evaluation/predict_qsvc.py \
    --qsvc-dir "$QSVC_DIR" --features "$FEATS" --out "$PREDS"

echo "[$(date +%F\ %T)] ── 3/3 ablation: ABL_NO_LOAD (XGB) vs ABL_NO_LOAD_QSVC"
.venv/bin/python phase3_evaluation/ablation_study.py --model "$MODEL" "${RATES[@]}" \
    --clf-dir deploy/deepseek_8b_t1000 --instance-cfg instance_configs.json \
    --variants ABL_NO_LOAD,ABL_NO_LOAD_QSVC --qsvc-preds "$PREDS" --out "$OUT"

echo "[$(date +%F\ %T)] ALL DONE → $OUT"
