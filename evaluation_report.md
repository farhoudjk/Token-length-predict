## Length Classifier Evaluation (RandomForest vs. XGBoost)

This report compares the two trained classifiers on three public datasets. The RandomForest (RF) evaluation used **live ground-truth generations** from `TheBloke/Llama-2-7B-Chat-AWQ` via vLLM (`output_source: vllm`). The XGBoost (XGB) run used the **reference responses stored inside each dataset** (`output_source: dataset`) because GPU memory was insufficient to launch vLLM again. As a result, the two runs reflect different notions of “ground truth,” so XGB's higher accuracies largely stem from the fact that the original datasets contain very few long (>500 token) answers.

| Dataset    | Domain                 | Samples | RF Accuracy | RF Long Support | XGB Accuracy | XGB Long Support | Notes |
|-----------|------------------------|---------|-------------|-----------------|--------------|------------------|-------|
| Alpaca    | General instruction    | 200     | **0.77**    | 30 live long answers (4 predicted correctly) | **0.69**  | 50 live long answers (6 predicted correctly) | Both models now evaluated on live vLLM generations. XGB still struggles to capture long outputs despite the richer positive set. |
| Dolly 15k | Business conversation  | 200     | **0.765**   | 44 live long answers (1 correct) | **0.75**  | 47 live long answers (1 predicted correctly) | XGB accuracy dropped sharply once judged against the same generated completions as RF. |
| UltraChat | Multi-turn support     | 150     | **0.56**    | 73 live long answers (7 correct) | **0.52**  | 79 live long answers (7 predicted correctly) | UltraChat remains the hardest domain — both models predict long responses poorly even though half the outputs exceed 500 tokens. |

### Metrics (direct from summaries)

- **RandomForest (`out/real_dataset_eval/summary.json`)**
  - Alpaca: macro F1 0.508, weighted F1 0.759; confusion matrix `[[150, 20], [26, 4]]`.
  - Dolly: macro F1 0.454, weighted F1 0.684; confusion matrix `[[152, 4], [43, 1]]`.
  - UltraChat: macro F1 0.438, weighted F1 0.445; confusion matrix `[[77, 0], [66, 7]]`.
  - `output_source`: `vllm:TheBloke/Llama-2-7B-Chat-AWQ`.

- **XGBoost (`out/real_dataset_eval_xgb_live4096/summary.json`)**
  - Alpaca: macro F1 0.486, weighted F1 0.648; confusion matrix `[[132, 18], [44, 6]]`.
  - Dolly: macro F1 0.447, weighted F1 0.664; confusion matrix `[[149, 4], [46, 1]]`.
  - UltraChat: macro F1 0.413, weighted F1 0.400; confusion matrix `[[71, 0], [72, 7]]`.
  - `output_source`: `vllm:TheBloke/Llama-2-7B-Chat-AWQ`.

### Takeaways
1. **Ground truth choice matters**: Once both models were judged on the same live outputs, XGBoost’s apparent edge vanished (it now trails RF on every dataset). This underlines how misleading the dataset-reference evaluation was.
2. **Imbalanced class hurts macro metrics**: Even with live generations introducing many long completions, recall for the long class hovers below 10% on Dolly and UltraChat. Macro averages remain <0.5 across the board.
3. **Next steps**
   - Free GPU memory (or lower `VLLM_GPU_MEMORY_UTILIZATION`) and rerun XGB with `--generate-with-vllm` to compare models under the same live-ground-truth regime.
   - Revisit the threshold (500 tokens) or sample more long-form prompts so both models see a healthier positive class.
   - Consider calibration or asymmetric thresholds to reduce false positives on short prompts when the policy cost of misclassifying long responses is high.

All raw outputs and per-prompt predictions live in:
- `out/real_dataset_eval/*.csv`
- `out/real_dataset_eval_rf_live4096/*.csv`
- `out/real_dataset_eval_xgb_live4096/*.csv`

Re-run commands:
```bash
# RandomForest + live vLLM generations (as previously run)
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
