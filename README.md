
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

## Evaluate the classifier on public datasets
Use `scripts/evaluate_real_datasets.py` to sanity-check the trained length classifier on several real instruction datasets (Alpaca, Dolly 15k, and the multi-turn UltraChat SFT conversations). The script downloads each dataset, computes the prompt-only features, predicts whether the reference response exceeds 500 tokens, and reports accuracy/confusion matrices. Example:

```
python scripts/evaluate_real_datasets.py \
  --model-path out/clf_results/RandomForest_model.pkl \
  --feature-json out/clf_results/feature_columns.json \
  --datasets alpaca,dolly,ultrachat \
  --threshold 500
```

By default the script resolves the tokenizer/model id from `config/models.json` (matching `run_inference_vllm.py`), so it will use the quantized `TheBloke/Llama-2-7B-Chat-AWQ` setup out of the box. Override with `--tokenizer-model` if you want to count tokens using a different tokenizer (e.g., `--tokenizer-model meta-llama/Llama-3.1-8B-instruct`). Add `--generate-with-vllm` (and optionally adjust `--generation-*` flags) to run each prompt through the actual vLLM model for new completions before measuring token counts, so you can validate the classifier against live generations instead of static dataset answers. Results (per-dataset prediction CSVs + an aggregated `summary.json`) are written to `out/real_dataset_eval/`. Use `--samples-per-dataset` to adjust how many prompts you want to test per domain.

### Example commands

```bash
# RandomForest + live vLLM generations (4K max tokens)
python scripts/evaluate_real_datasets.py \
  --model-path out/clf_results/RandomForest_model.pkl \
  --feature-json out/clf_results/feature_columns.json \
  --datasets alpaca,dolly,ultrachat \
  --threshold 500 \
  --generate-with-vllm \
  --generation-max-new-tokens 4096 \
  --generation-batch-size 8 \
  --output-dir out/real_dataset_eval_rf_live4096

# XGBoost + live vLLM generations (requires free GPU memory)
python scripts/evaluate_real_datasets.py \
  --model-path out/clf_results/XGBoost_model.pkl \
  --feature-json out/clf_results/feature_columns.json \
  --datasets alpaca,dolly,ultrachat \
  --threshold 500 \
  --generate-with-vllm \
  --generation-max-new-tokens 4096 \
  --generation-batch-size 8 \
  --output-dir out/real_dataset_eval_xgb_live4096
```
