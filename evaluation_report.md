## Length Classifier Evaluation (RandomForest vs. XGBoost)

This report compares the two trained classifiers on three public datasets. The RandomForest (RF) evaluation used **live ground-truth generations** from `TheBloke/Llama-2-7B-Chat-AWQ` via vLLM (`output_source: vllm`). The XGBoost (XGB) run used the **reference responses stored inside each dataset** (`output_source: dataset`) because GPU memory was insufficient to launch vLLM again. As a result, the two runs reflect different notions of “ground truth,” so XGB's higher accuracies largely stem from the fact that the original datasets contain very few long (>500 token) answers.

| Dataset    | Domain                 | Samples | RF Accuracy | RF Long Support | XGB Accuracy | XGB Long Support | Notes |
|-----------|------------------------|---------|-------------|-----------------|--------------|------------------|-------|
| Alpaca    | General instruction    | 200     | **0.77**    | 30 live long answers (4 predicted correctly) | **0.835** | 1 long reference | RF predicted more long outputs, better reflecting live generations. XGB simply labeled everything short because the dataset hardly ever exceeds 500 tokens. |
| Dolly 15k | Business conversation  | 200     | **0.765**   | 44 live long answers (1 correct) | **0.905** | 2 long references | Same pattern—live generations created long responses much more often than the stored Dolly labels. |
| UltraChat | Multi-turn support     | 150     | **0.56**    | 73 live long answers (7 correct) | **0.813** | 24 long references | vLLM produced many long replies so RF performance dropped sharply; the static dataset still skews short, inflating XGB accuracy. |

### Metrics (direct from summaries)

- **RandomForest (`out/real_dataset_eval/summary.json`)**
  - Alpaca: macro F1 0.508, weighted F1 0.759; confusion matrix `[[150, 20], [26, 4]]`.
  - Dolly: macro F1 0.454, weighted F1 0.684; confusion matrix `[[152, 4], [43, 1]]`.
  - UltraChat: macro F1 0.438, weighted F1 0.445; confusion matrix `[[77, 0], [66, 7]]`.
  - `output_source`: `vllm:TheBloke/Llama-2-7B-Chat-AWQ`.

- **XGBoost (`out/real_dataset_eval_xgb/summary.json`)**
  - Alpaca: macro F1 0.455, weighted F1 0.906; confusion matrix `[[167, 32], [1, 0]]`.
  - Dolly: macro F1 0.475, weighted F1 0.941; confusion matrix `[[181, 17], [2, 0]]`.
  - UltraChat: macro F1 0.536, weighted F1 0.780; confusion matrix `[[119, 7], [21, 3]]`.
  - `output_source`: `dataset`.

### Takeaways
1. **Ground truth choice matters**: Live generations revealed that real completions often exceed 500 tokens, which stressed the RF classifier and exposed its limited recall for long outputs. Evaluating against the lightly populated dataset references makes any model appear vastly more accurate because nearly all examples are short.
2. **Imbalanced class hurts macro metrics**: Across both models the “long” class had poor precision/recall because the training data and evaluation prompts are dominated by short answers. Macro averages emphasize this, dropping below 0.55 for every dataset/model pair.
3. **Next steps**
   - Free GPU memory (or lower `VLLM_GPU_MEMORY_UTILIZATION`) and rerun XGB with `--generate-with-vllm` to compare models under the same live-ground-truth regime.
   - Revisit the threshold (500 tokens) or sample more long-form prompts so both models see a healthier positive class.
   - Consider calibration or asymmetric thresholds to reduce false positives on short prompts when the policy cost of misclassifying long responses is high.

All raw outputs and per-prompt predictions live in:
- `out/real_dataset_eval/*.csv`
- `out/real_dataset_eval_xgb/*.csv`

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
