# Real Dataset Evaluation — 3-Class XGBoost (vLLM generations)

## Setup
- Command: `python scripts/evaluate_real_datasets.py --model-path out/clf_results/XGBoost_model.pkl --feature-json out/clf_results/feature_columns.json --datasets alpaca,dolly,ultrachat --thresholds 500 2000 --generate-with-vllm --generation-max-new-tokens 4096 --generation-batch-size 8 --output-dir out/real_dataset_eval_xgb_live4096`
- Tokenizer/generator: `TheBloke/Llama-2-7B-Chat-AWQ`
- Class bins: short ≤ 500 tokens, medium 501–2000, long > 2000
- Output directory: `out/real_dataset_eval_xgb_live4096`

## Dataset Summaries

### Alpaca (200 samples)
- Accuracy 68.5%, weighted F1 0.67
- Class metrics:
  - Short: precision 0.798, recall 0.832, F1 0.815 (161 true short)
  - Medium: precision 0.094, recall 0.103, F1 0.098 (29 true medium)
  - Long: precision/recall 0 (10 true long)
- Confusion matrix (rows=true, cols=pred):  
  `[[134, 26, 1], [23, 3, 3], [10, 0, 0]]`
- CSV: `out/real_dataset_eval_xgb_live4096/alpaca_predictions.csv`

### Dolly (200 samples)
- Accuracy 67.5%, weighted F1 0.60
- Class metrics:
  - Short: precision 0.721, recall 0.910, F1 0.805 (145 true short)
  - Medium: precision 0.177, recall 0.077, F1 0.107 (39 true medium)
  - Long: precision/recall 0 (16 true long)
- Confusion matrix:  
  `[[132, 12, 1], [31, 3, 5], [16, 0, 0]]`
- CSV: `out/real_dataset_eval_xgb_live4096/dolly_predictions.csv`

### UltraChat (150 samples)
- Accuracy 46.7%, weighted F1 0.35
- Class metrics:
  - Short: precision 0.443, recall 0.969, F1 0.608 (64 true short)
  - Medium: precision 0.800, recall 0.105, F1 0.186 (76 true medium)
  - Long: precision/recall 0 (10 true long)
- Confusion matrix:  
  `[[62, 2, 0], [68, 8, 0], [10, 0, 0]]`
- CSV: `out/real_dataset_eval_xgb_live4096/ultrachat_predictions.csv`

## Observations
1. The classifier handles short responses well but largely collapses medium/long examples into the short class across all datasets.
2. Long responses (>2000 tokens) are never predicted correctly; training data likely lacks enough long examples or needs different thresholds.
3. UltraChat’s multi-turn outputs emphasize the weakness: 78/86 non-short samples were predicted as short.

## Next Steps
1. Rebalance or augment training data for medium/long outputs (e.g., SMOTE, domain-specific sampling).
2. Experiment with calibrated probability thresholds or hierarchical classification (short vs non-short first, then split medium/long).
3. After retraining, rerun `scripts/evaluate_real_datasets.py` to refresh `summary.json` and CSVs for trend tracking.
