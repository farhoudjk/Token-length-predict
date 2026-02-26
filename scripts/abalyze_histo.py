#!/usr/bin/env python3
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# ====================== MODEL GROUPS ======================
model_groups = {
    "llama-3.1-8b": [
        "data/vllm_llama_3_1_8b_dolly15k.csv",
        "data/vllm_llama_3_1_8b_mixed_prompts_v2.csv"
    ],
    "mistral-7b-v0.2": [
        "data/vllm_mistral_7b_v0_2_dolly15k.csv",
        "data/vllm_mistral_7b_v0_2_mixed_prompts_v2.csv"
    ],
    "deepseek-8b": [ 
        "data/vllm_deepseek_r1_distill_llama_8b_dolly15k.csv",
        "data/vllm_deepseek_8b_mixed_prompts_v234.csv"
    ],
    "deepseek-r1-qwen-14b": [
        "data/vllm_deepseek_r1_qwen_14b_mixed_prompts_v2.csv",
        "data/vllm_deepseek_r1_distill_qwen_14b_dolly15k.csv"
    ],
    "llama2-7b-awqmarlin": ["data/vllm_llama2_7b_awqmarlin_dolly15k.csv"],
    "llama2-13b": ["data/vllm_llama2_13b_dolly15k.csv"],
    "deepseek-v2-lite": ["data/vllm_deepseek_v2_lite_chat.csv"],
    "mistral_nemo_12b": ["data/vllm_mistral_nemo_12b.csv"],


}

def parse_token_count(v):
    if pd.isna(v): return None
    try: return int(float(str(v).strip()))
    except: return None

def analyze_and_plot(df, model_name):
    # 1. Find the correct column
    common_keys = ['output_tokens', 'output_token', 'output_token_count', 'generated_tokens']
    tok_col = next((k for k in common_keys if k in df.columns), None)
    if not tok_col:
        tok_col = next((c for c in df.columns if 'output' in c.lower() and 'token' in c.lower()), None)

    if tok_col is None:
        print(f"❌ {model_name}: No token column found.")
        return

    # 2. Process and Filter Data
    df['parsed_tokens'] = df[tok_col].apply(parse_token_count)
    valid_tokens = df['parsed_tokens'].dropna()
    # Filter for range 10 to 5000 as requested
    filtered = valid_tokens[(valid_tokens >= 10) & (valid_tokens <= 5000)]

    if filtered.empty:
        print(f"⚠️ {model_name}: No valid data in 10-5000 range.")
        return

    # 3. Calculate 50/50 Threshold (Median)
    median_val = filtered.median()

    # 4. Create the Plot
    plt.figure(figsize=(12, 6))
    
    # Bins: 10-100, then 100-200... up to 5000
    bins = [10] + list(range(100, 5100, 100))
    
    # Draw Histogram
    plt.hist(filtered, bins=bins, color='#5DADE2', edgecolor='white', alpha=0.7, label='Token Count')

    # Add Median Vertical Line (The 50/50 Threshold)
    plt.axvline(median_val, color='#E74C3C', linestyle='dashed', linewidth=2, 
                label=f'50/50 Threshold: {int(median_val)} tokens')

    # Formatting
    plt.title(f"Token Distribution & 50/50 Threshold: {model_name.upper()}", fontsize=14)
    plt.xlabel("Output Tokens", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(range(0, 5500, 500))
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    # Save
    save_path = f"analysis_{model_name.replace('-', '_')}.pdf"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"✅ {model_name.upper()}: Median = {int(median_val)} | Saved to {save_path}")

# ====================== EXECUTION ======================
print("🚀 Starting combined analysis...\n")

for model_name, file_list in model_groups.items():
    dfs = [pd.read_csv(f, low_memory=False) for f in file_list if os.path.exists(f)]
    if dfs:
        analyze_and_plot(pd.concat(dfs, ignore_index=True), model_name)
    else:
        print(f"❌ Missing files for {model_name}")

print("\n✨ All charts and thresholds generated!")