# Batch Size Knee-Point Study (200 prompts / class)

Command (latest run):  
`python scripts/analyze_batch_knee.py --model-name TheBloke/Llama-2-7B-Chat-AWQ --samples-per-class 200 --batch-sizes 12,16,20,24,30,40,50,60,70 --output-json out/knee_metrics.json`

- Dataset: `out/dolly_inference_results_llama2_awq (1).csv`
- 200 prompts per length class (short ≤500, medium 501–2000, long >2000 tokens).
- Metrics capture combined input+output throughput (tokens/sec) and p95/p99 tail latency per prompt (ms).  
- Score = throughput / (p95 + p99); knee = max score for each class over the sweep.

## Short Class (≤500 tokens)

| Batch size | Throughput (tok/s) | p95 latency (ms) | Score |
|-----------:|-------------------:|-----------------:|------:|
| 12         | 444.0              | 14,224           | 0.0156 |
| 16         | 519.7              | 15,686           | 0.0166 |
| 20         | 594.7              | 17,075           | 0.0174 |
| 24         | 630.0              | 20,093           | 0.0157 |
| 30         | 774.5              | 18,982           | 0.0204 |
| 40         | 906.0              | 20,546           | 0.0220 |
| 50         | 1,027.2            | 22,759           | 0.0226 |
| 60         | 1,015.2            | 23,226           | 0.0219 |
| **70**     | **1,160.7**        | **25,398**       | **0.0228** |

**Knee:** batch size 70. Larger batches keep GPU utilization high while p95 stays manageable (~25 s).

## Medium Class (501–2000 tokens)

| Batch size | Throughput (tok/s) | p95 latency (ms) | Score |
|-----------:|-------------------:|-----------------:|------:|
| 12         | 553.2              | 18,115           | 0.0153 |
| **16**     | **661.6**          | **18,564**       | **0.0178** |
| 20         | 739.8              | 21,995           | 0.0168 |
| 24         | 791.3              | 24,290           | 0.0163 |
| 30         | 892.7              | 26,707           | 0.0167 |
| 40         | 985.2              | 35,035           | 0.0141 |
| 50         | 1,055.3            | 36,045           | 0.0146 |
| 60         | 1,050.2            | 44,273           | 0.0119 |
| 70         | 1,098.3            | 43,390           | 0.0127 |

**Knee:** batch size 16. Medium prompts saturate early; beyond ~20 parallel prompts, tail latency grows faster than throughput.

## Long Class (>2000 tokens)

| Batch size | Throughput (tok/s) | p95 latency (ms) | Score |
|-----------:|-------------------:|-----------------:|------:|
| 12         | 561.2              | 19,429           | 0.0144 |
| 16         | 655.4              | 22,261           | 0.0147 |
| **20**     | **748.2**          | **23,895**       | **0.0157** |
| 24         | 790.2              | 26,268           | 0.0150 |
| 30         | 836.5              | 30,792           | 0.0136 |
| 40         | 944.8              | 38,518           | 0.0123 |
| 50         | 965.2              | 50,501           | 0.0096 |
| 60         | 912.6              | 56,171           | 0.0081 |
| 70         | 928.7              | 70,492           | 0.0066 |

**Knee:** batch size 20. For very long generations, pushing parallelism higher balloons p95/p99 (up to 70s) with minimal throughput gain.

## Insights
1. Short prompts benefit from aggressive batching (50–70) thanks to narrow length variance.
2. Medium prompts prefer modest batches (~16). Above that, stragglers dominate latency.
3. Long prompts should stay near batches of 16–20 to avoid multi-minute tails.
4. For mixed workloads, consider routing by predicted length class to different batch caps (e.g., short=64+, medium≈16, long≈16–20).
