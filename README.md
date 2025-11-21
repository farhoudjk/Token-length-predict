
# PromptWork Dataset Starter

This starter gives you a minimal, GPU-ready pipeline to collect a cross-model workload dataset of (prompt → response) with token counts, latency, and prompt features.

## Contents
- `configs/schema.json` — canonical dataset schema.
- `scripts/feature_extractor.py` — admission-time feature extractor.
- `scripts/run_inference.py` — unified inference harness (Hugging Face local models + metrics).
- `data/sample_prompts.csv` — demo prompts.

## Quickstart (Local HF model)
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pandas pyarrow
python scripts/run_inference.py --prompts_csv data/sample_prompts.csv --out_csv data/run_llama3_demo.csv --model_name meta-llama/Meta-Llama-3-8B-Instruct --max_new_tokens 128
```

This writes a CSV with the schema in `configs/schema.json`. You can switch models (e.g., `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2-7B-Instruct`), change sampling parameters, and expand the prompts CSV.

## Next Steps
1. Add more domains (QA, reasoning, coding, math).
2. Add engines (vLLM, OpenAI-compatible servers).
3. Log TTFT/TPOT accurately for your serving stack.
4. Export to Parquet for scale: `df.to_parquet(...)`.
5. Train a length predictor (XGBoost/LightGBM) on the collected data.
