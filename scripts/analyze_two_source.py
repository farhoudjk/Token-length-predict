#!/usr/bin/env python3
import pandas as pd
import os
from collections import Counter

# ====================== MODEL GROUPS ======================
# Models with both sources → will be COMBINED
# Models with one source → analyzed as-is
model_groups = {
    "llama-3.1-8b": [
        "data/vllm_llama_3_1_8b_dolly15k.csv",
        "data/vllm_llama_3_1_8b_mixed_prompts_v2.csv"
    ],
    "mistral-7b-v0.2": [
        "data/vllm_mistral_7b_v0_2_dolly15k.csv",
        "data/vllm_mistral_7b_v0_2_mixed_prompts_v2.csv"
    ],
    "deepseek-8b": [                     # two mixed runs → combine
        "data/vllm_deepseek_8b_mixed_prompts_v2.csv",
        "data/vllm_deepseek_8b_mixed_prompts_v234.csv"
    ],
    "deepseek-r1-distill-llama-8b": [
        "data/vllm_deepseek_r1_distill_llama_8b_dolly15k.csv"
    ],
    "deepseek-r1-qwen-14b": [
        "data/vllm_deepseek_r1_qwen_14b_mixed_prompts_v2.csv"
    ],
    "llama2-7b-awqmarlin": [
        "data/vllm_llama2_7b_awqmarlin_dolly15k.csv"
    ],
    "llama2-13b": [
        "data/vllm_llama2_13b_dolly15k.csv"
    ],
}

def parse_token_count(v):
    if pd.isna(v):
        return None
    try:
        return int(float(str(v).strip()))
    except:
        return None

def analyze_df(df, model_name):
    total = len(df)
    if total == 0:
        print(f"❌ {model_name}: No rows found")
        return

    # Find the token column
    common_keys = [
        'output_tokens', 'output_token', 'output_token_count', 'output_tokens_count',
        'generated_tokens', 'num_output_tokens'  # Add more if needed from your CSVs
    ]
    tok_col = None
    for k in common_keys:
        if k in df.columns:
            tok_col = k
            break
    if tok_col is None:
        # Fallback: search for column containing 'output' and 'token'
        for col in df.columns:
            lc = col.lower()
            if 'output' in lc and 'token' in lc:
                tok_col = col
                break
    if tok_col is None:
        print(f"❌ {model_name}: No output token column found! Check CSV headers.")
        return

    print(f"ℹ️  Using column '{tok_col}' for token counts.")

    counts = Counter()
    for _, row in df.iterrows():
        tok = parse_token_count(row[tok_col])
        if tok is None:
            counts['missing'] += 1
            continue
        if 1 <= tok < 9:
            counts['error'] += 1
        elif 10 <= tok <= 500:
            counts['short'] += 1
        elif 501 <= tok <= 2000:
            counts['medium'] += 1
        elif 2001 <= tok <= 4096:
            counts['exhaustive'] += 1
        else:
            counts['other'] += 1

    def pct(n): 
        return 100.0 * n / total if total else 0

    print(f"\n{'='*60}")
    print(f"📊 MODEL: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Total rows (combined): {total:,}")
    print("")
    print(f"1–8 tokens     (error)      : {counts.get('error', 0):6,}  ({pct(counts.get('error', 0)):6.2f}%)")
    print(f"10–500 tokens  (short)      : {counts.get('short', 0):6,}  ({pct(counts.get('short', 0)):6.2f}%)")
    print(f"501–2000 tokens (medium)    : {counts.get('medium', 0):6,}  ({pct(counts.get('medium', 0)):6.2f}%)")
    print(f"2001–4096 tokens (exhaustive): {counts.get('exhaustive', 0):6,}  ({pct(counts.get('exhaustive', 0)):6.2f}%)")
    print("")
    print(f"Missing token count         : {counts.get('missing', 0):6,}  ({pct(counts.get('missing', 0)):6.2f}%)")
    print(f"Other / out-of-range        : {counts.get('other', 0):6,}  ({pct(counts.get('other', 0)):6.2f}%)")
    print("")

# ====================== RUN ANALYSIS ======================
print("🚀 Starting token length analysis per model...\n")

for model_name, file_list in model_groups.items():
    dfs = []
    for f in file_list:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
                print(f"✅ Loaded: {f}  ({len(df):,} rows)")
            except Exception as e:
                print(f"❌ Failed to read {f}: {e}")
        else:
            print(f"⚠️  File not found: {f}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        analyze_df(combined, model_name)
    else:
        print(f"❌ No files found for {model_name}")

print("\n✅ Analysis complete!")