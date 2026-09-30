#!/usr/bin/env bash
# Per-model concurrency profiling (knee_new.py) followed by the ablation study
# with that model's own M_Seq configuration.
#
# Each model is profiled in profiles/<tag>/ so the shared instance_configs.json
# (Mistral-7B profile) is never overwritten. Ablation results go to
# ablation_results_<tag>_profiled.csv; the earlier runs with the Mistral-7B
# profile (300/128) stay in ablation_results_<tag>.csv.
#
# Usage:  nohup ./run_profiled_ablations.sh [wait_pid] > logs/profiled.log 2>&1 &
# Stop:   kill <this script's pid>; then
#         pkill -9 -f "rq4_evaluation.py --worker"; pkill -9 -f "VLLM::EngineCore"
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs profiles

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
    echo "[$(date +%F\ %T)] waiting for pid $WAIT_PID to finish"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

# Keep the Mistral-7B profile that all earlier runs used.
[[ -f instance_configs_mistral7b.json ]] || cp instance_configs.json instance_configs_mistral7b.json

RATES=(--rate-start 8 --rate-step 16 --rate-end 72)
SHORT_CFGS="128,180,220,256,300"   # same sweep grid as the Mistral-7B profile
LONG_CFGS="96,128,140,180,220"

# tag | HF model id | threshold | weight GiB | quantization | default clf dir | t500 clf dir
MODELS=(
  "nemo|mistralai/Mistral-Nemo-Instruct-2407|300|23|-|deploy/nemo_t300|deploy/ablation_t500_nemo"
  "llama2_7b|TheBloke/Llama-2-7B-Chat-AWQ|200|3.67|-|deploy/llama2_7b_t200|deploy/ablation_t500_llama2_7b"
  "llama2_13b|TheBloke/Llama-2-13B-chat-AWQ|200|6.82|-|deploy/llama2_13b_t200|deploy/ablation_t500_llama2_13b"
  "deepseek_8b|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|1000|15.0|-|deploy/deepseek_8b_t1000|deploy/ablation_t500_deepseek_8b"
  "deepseek_14b|deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|600|15.4|fp8|deploy/deepseek_14b_t600|deploy/ablation_t500_deepseek_14b"
)

cleanup_gpus() {
    pkill -9 -f "rq4_evaluation.py --worker" 2>/dev/null
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 5
}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag model thr wgib quant clf t500 <<< "$entry"
    cfg="instance_configs_${tag}.json"
    qargs=(); [[ "$quant" != "-" ]] && qargs=(--quantization "$quant")

    if [[ ! -f "$cfg" ]]; then
        echo "[$(date +%F\ %T)] ── $tag: profiling"
        mkdir -p "profiles/$tag"
        ( cd "profiles/$tag" && CUDA_VISIBLE_DEVICES=0 python ../../phase2_training/knee_new.py \
              --model "$model" --threshold "$thr" --gpu_gib 32 --weight_gib "$wgib" \
              --max_model_len 4096 --short_configs "$SHORT_CFGS" --long_configs "$LONG_CFGS" \
              "${qargs[@]}" ) > "logs/knee_${tag}.log" 2>&1
        cleanup_gpus
        if [[ -f "profiles/$tag/instance_configs.json" ]]; then
            cp "profiles/$tag/instance_configs.json" "$cfg"
            echo "[$(date +%F\ %T)]    $tag profile: $(python -c "import json;c=json.load(open('$cfg'));print(c['SHORT']['optimal_m_seq'],'/',c['LONG']['optimal_m_seq'])")"
        else
            echo "[$(date +%F\ %T)]    $tag profiling FAILED — see logs/knee_${tag}.log; skipping ablation"
            continue
        fi
    fi

    echo "[$(date +%F\ %T)] ── $tag: ablation"
    python phase3_evaluation/ablation_study.py --model "$model" "${qargs[@]}" "${RATES[@]}" \
        --clf-dir "$clf" --t500-dir "$t500" --instance-cfg "$cfg" \
        --out "ablation_results_${tag}_profiled.csv" > "logs/ablation_${tag}.log" 2>&1
    echo "[$(date +%F\ %T)]    $tag ablation exit=$? rows=$(($(wc -l < ablation_results_${tag}_profiled.csv 2>/dev/null || echo 1) - 1))"
    cleanup_gpus
done
echo "[$(date +%F\ %T)] ALL DONE"
