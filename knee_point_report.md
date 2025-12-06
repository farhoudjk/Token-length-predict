# Batch Size Knee-Point Study (vLLM)

Command (latest run):  
`python scripts/analyze_batch_knee.py --model-name TheBloke/Llama-2-7B-Chat-AWQ --batch-sizes 8,12,16,24,30,36,40,50 --output-json out/knee_metrics.json`

- Dataset: `out/dolly_inference_results_llama2_awq (1).csv`
- 100 prompts per length class (short ≤500 tokens, medium 501–2000, long >2000).
- Metrics include throughput in tokens/sec (input+output), and latency p95/p99 (ms).  
- Score = throughput / (p95 + p99). Higher is better; knee point chosen by max score over the 8–50 sweep.

## Short Class (≤500 tokens)

| Batch size | Throughput (tok/s) | p95 latency (ms) | Score |
|-----------:|-------------------:|-----------------:|------:|
| 8          | 342.21             | 12,294           | 0.0139 |
| 12         | 456.61             | 14,125           | 0.0162 |
| 16         | 445.86             | 14,002           | 0.0159 |
| 24         | 618.66             | 19,964           | 0.0155 |
| 30         | 771.06             | 17,847           | 0.0216 |
| 36         | 852.24             | 19,552           | 0.0218 |
| 40         | 853.34             | 18,673           | 0.0228 |
| **50**     | **1,065.27**       | **21,504**       | **0.0248** |

**Knee:** batch size 50 (fastest throughput, best score despite higher latency).

## Medium Class (501–2000 tokens)

| Batch size | Throughput (tok/s) | p95 latency (ms) | Score |
|-----------:|-------------------:|-----------------:|------:|
| 8          | 435.52             | 14,764           | 0.0147 |
| 12         | 552.99             | 15,747           | 0.0176 |
| 16         | 657.66             | 17,406           | 0.0189 |
| 24         | 728.66             | 21,577           | 0.0169 |
| 30         | 847.05             | 23,134           | 0.0183 |
| **36**     | **950.12**         | **24,962**       | **0.0190** |
| 40         | 938.75             | 28,554           | 0.0164 |
| 50         | 1,069.16           | 32,394           | 0.0165 |

**Knee:** batch size 36 (best throughput/latency trade-off; larger batches add latency faster than throughput gains).

## Long Class (>2000 tokens)

| Batch size | Throughput (tok/s) | p95 latency (ms) | Score |
|-----------:|-------------------:|-----------------:|------:|
| 8          | 437.97             | 16,992           | 0.0129 |
| 12         | 549.04             | 21,170           | 0.0130 |
| **16**     | **637.70**         | **22,937**       | **0.0139** |
| 24         | 730.71             | 28,069           | 0.0130 |
| 30         | 799.99             | 31,116           | 0.0129 |
| 36         | 890.10             | 35,281           | 0.0126 |
| 40         | 900.99             | 37,724           | 0.0119 |
| 50         | 970.61             | 44,900           | 0.0108 |

**Knee:** batch size 16 (scores fall as latency explodes beyond ~600 tok/s).

## Takeaways
1. **Short prompts** benefit from the highest tested parallelism (batch 50) before p95/p99 blow up, maximizing throughput while tolerating ~21.5s batches.
2. **Medium prompts** saturate earlier; batch 36 offers the best balance. Larger batches bring marginal throughput gains but ~30% higher tail latency.
3. **Long prompts** prefer modest parallelism (batch 16). Beyond that, throughput gains are small compared to the jump in 95th/99th-percentile latency.
4. For mixed workloads, consider dynamic batching: steer short prompts to larger batch sizes, keep medium around 32–36, and cap long prompts near 16 to avoid runaway latency.
