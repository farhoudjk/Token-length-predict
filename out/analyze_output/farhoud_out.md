🚀 Starting token length analysis per model...

✅ Loaded: data/vllm_llama_3_1_8b_dolly15k.csv  (12,563 rows)
✅ Loaded: data/vllm_llama_3_1_8b_mixed_prompts_v2.csv  (2,500 rows)
ℹ️  Using column 'output_tokens' for token counts.

============================================================
📊 MODEL: LLAMA-3.1-8B
============================================================
Total rows (combined): 15,063

1–8 tokens     (error)      :  2,763  ( 18.34%)
10–500 tokens  (short)      :  4,879  ( 32.39%)
501–2000 tokens (medium)    :  3,607  ( 23.95%)
2001–4096 tokens (exhaustive):  3,780  ( 25.09%)

Missing token count         :      0  (  0.00%)
Other / out-of-range        :     34  (  0.23%)

✅ Loaded: data/vllm_mistral_7b_v0_2_dolly15k.csv  (12,559 rows)
✅ Loaded: data/vllm_mistral_7b_v0_2_mixed_prompts_v2.csv  (2,500 rows)
ℹ️  Using column 'output_tokens' for token counts.

============================================================
📊 MODEL: MISTRAL-7B-V0.2
============================================================
Total rows (combined): 15,059

1–8 tokens     (error)      :     58  (  0.39%)
10–500 tokens  (short)      : 10,741  ( 71.33%)
501–2000 tokens (medium)    :  3,832  ( 25.45%)
2001–4096 tokens (exhaustive):    411  (  2.73%)

Missing token count         :      0  (  0.00%)
Other / out-of-range        :     17  (  0.11%)

✅ Loaded: data/vllm_deepseek_8b_mixed_prompts_v2.csv  (2,500 rows)
✅ Loaded: data/vllm_deepseek_8b_mixed_prompts_v234.csv  (2,500 rows)
ℹ️  Using column 'output_tokens' for token counts.

============================================================
📊 MODEL: DEEPSEEK-8B
============================================================
Total rows (combined): 5,000

1–8 tokens     (error)      :      0  (  0.00%)
10–500 tokens  (short)      :    510  ( 10.20%)
501–2000 tokens (medium)    :  3,016  ( 60.32%)
2001–4096 tokens (exhaustive):  1,474  ( 29.48%)

Missing token count         :      0  (  0.00%)
Other / out-of-range        :      0  (  0.00%)

✅ Loaded: data/vllm_deepseek_r1_distill_llama_8b_dolly15k.csv  (12,563 rows)
ℹ️  Using column 'output_tokens' for token counts.

============================================================
📊 MODEL: DEEPSEEK-R1-DISTILL-LLAMA-8B
============================================================
Total rows (combined): 12,563

1–8 tokens     (error)      :     14  (  0.11%)
10–500 tokens  (short)      :  2,477  ( 19.72%)
501–2000 tokens (medium)    :  8,437  ( 67.16%)
2001–4096 tokens (exhaustive):  1,635  ( 13.01%)

Missing token count         :      0  (  0.00%)
Other / out-of-range        :      0  (  0.00%)

✅ Loaded: data/vllm_deepseek_r1_qwen_14b_mixed_prompts_v2.csv  (2,500 rows)
ℹ️  Using column 'output_tokens' for token counts.

============================================================
📊 MODEL: DEEPSEEK-R1-QWEN-14B
============================================================
Total rows (combined): 2,500

1–8 tokens     (error)      :      0  (  0.00%)
10–500 tokens  (short)      :    534  ( 21.36%)
501–2000 tokens (medium)    :  1,346  ( 53.84%)
2001–4096 tokens (exhaustive):    620  ( 24.80%)

Missing token count         :      0  (  0.00%)
Other / out-of-range        :      0  (  0.00%)

✅ Loaded: data/vllm_llama2_7b_awqmarlin_dolly15k.csv  (12,558 rows)
ℹ️  Using column 'output_tokens' for token counts.

============================================================
📊 MODEL: LLAMA2-7B-AWQMARLIN
============================================================
Total rows (combined): 12,558

1–8 tokens     (error)      :    565  (  4.50%)
10–500 tokens  (short)      :  8,896  ( 70.84%)
501–2000 tokens (medium)    :  2,497  ( 19.88%)
2001–4096 tokens (exhaustive):    519  (  4.13%)

Missing token count         :      0  (  0.00%)
Other / out-of-range        :     81  (  0.65%)

✅ Loaded: data/vllm_llama2_13b_dolly15k.csv  (12,558 rows)
ℹ️  Using column 'output_tokens' for token counts.

============================================================
📊 MODEL: LLAMA2-13B
============================================================
Total rows (combined): 12,558

1–8 tokens     (error)      :    211  (  1.68%)
10–500 tokens  (short)      :  9,579  ( 76.28%)
501–2000 tokens (medium)    :  2,634  ( 20.97%)
2001–4096 tokens (exhaustive):    121  (  0.96%)

Missing token count         :      0  (  0.00%)
Other / out-of-range        :     13  (  0.10%)


✅ Analysis complete!