#!/usr/bin/env python3
import os, sys, csv, json, time, uuid, argparse
import multiprocessing as mp
import random # Added for dataset sampling

# Ensure spawn start method to avoid fork-related CUDA issues when running on GPU
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set; continue
    pass
from pathlib import Path
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset # Added for Hugging Face datasets

# local feature extractor (same as your TF harness)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from feature_extractor import extract_prompt_features  # noqa


def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def read_prompts_csv(path: str) -> List[Dict[str, str]]:
    """Reads prompts from a local CSV file."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_and_sample_dolly(dataset_name: str, num_prompts: int) -> List[Dict[str, str]]:
    """
    Loads a Hugging Face dataset (e.g., Dolly 15k), samples prompts evenly across 
    domains (categories), and converts them to the expected row format.
    """
    print(f"Loading dataset: {dataset_name}...")
    
    # Load the full dataset 'train' split
    ds: Dataset = load_dataset(dataset_name, split="train")

    # Get unique categories (domains)
    domains = ds.unique("category")
    num_domains = len(domains)
    
    # Calculate how many samples to take per domain to ensure even distribution
    samples_per_domain = num_prompts // num_domains
    remaining_samples = num_prompts % num_domains

    sampled_rows = []
    
    # Sample from each domain
    for domain_idx, domain in enumerate(domains):
        # Filter dataset for the current domain
        domain_ds = ds.filter(lambda x: x["category"] == domain)
        
        # Determine number of samples for this domain
        k = samples_per_domain
        if domain_idx < remaining_samples:
            k += 1 # Distribute remaining samples
            
        # Ensure k does not exceed the available size in the domain
        k = min(k, len(domain_ds))

        if k > 0:
            # Randomly sample k elements
            indices = random.sample(range(len(domain_ds)), k)
            batch = domain_ds.select(indices).to_list()
            
            for item in batch:
                # Dolly 15K columns: instruction, context, response, category
                
                # Combine instruction and context for the full prompt text
                prompt_text = item["instruction"]
                if item["context"]:
                    # Prepend context to instruction for a richer prompt
                    prompt_text = f"{item['context']}\n\n{item['instruction']}"
                    
                sampled_rows.append({
                    # Generate a unique ID for the prompt
                    "prompt_id": f"dolly15k_{domain}_{str(uuid.uuid4())[:8]}", 
                    "prompt_text": prompt_text,
                    "domain": item["category"],
                    "source": dataset_name,
                })

    print(f"Successfully sampled {len(sampled_rows)} prompts across {num_domains} domains.")
    return sampled_rows


def write_rows_csv_batch(
    path: str,
    rows: List[Dict[str, Any]],
    write_header: bool,
) -> None:
    """
    Write a batch of rows to CSV.

    - If write_header is True, overwrite file and write header + rows.
    - If write_header is False, append only rows (no header).
    """
    if not rows:
        return

    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        # Use an escapechar and explicit quoting to handle text containing
        # characters that require escaping (newlines, quotes, etc.).
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            escapechar="\\",
            quoting=csv.QUOTE_MINIMAL,
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def build_row(prompt_id: str, prompt: str, domain: str, source: str,
              gen: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    feats = extract_prompt_features(prompt)
    row = {
        "sample_id": str(uuid.uuid4()),
        "prompt_id": prompt_id,
        "prompt_text": prompt,
        "domain": domain or "unknown",
        "source": source or "synthetic",
        "temperature": cfg.get("temperature", 0.7),
        "top_p": cfg.get("top_p", 0.95),
        "max_new_tokens": 0,
        "natural_stop": bool(cfg.get("natural_stop", False)),
        "max_time": cfg.get("max_time", None),
        "ctx_cap": cfg.get("ctx_cap", 4096),
        "engine": "vllm",
        "model_name": cfg.get("model_name", "unknown"),
        "model_family": cfg.get("model_name", "unknown").split("/")[0],
        "features_json": json.dumps(feats, separators=(",", ":")),
        **gen,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    
    # Use a mutually exclusive group for input source selection
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompts_csv", default=None,
                       help="Path to the input CSV file containing prompts.")
    group.add_argument("--dataset_name", default=None,
                       help="Hugging Face dataset name (e.g., databricks/databricks-dolly-15k)")
                       
    ap.add_argument("--out_csv", required=True)
    
    ap.add_argument("--num_prompts", type=int, default=None,
                    help="Total number of prompts to sample from the dataset. Required if using --dataset_name.")
                    
    ap.add_argument("--model_name", default=None,
                    help="Model name or path (if omitted, read from config/models.json)")
    ap.add_argument("--batch_size", type=int, default=16, help="Parallel sequences per batch")
    
    # NOTE: Set default to 4096 to match Llama 2 native limit
    ap.add_argument("--ctx_cap", type=int, default=4096, help="Soft cap on max model length for speed")
    
    ap.add_argument("--natural_stop", action="store_true", help="Greedy decoding (temperature=0)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_time", type=float, default=None, help="(Not enforced per-request by vLLM)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # 1. Load Prompts (CSV or Dataset)
    # -------------------------------------------------------------------------
    
    rows_in: List[Dict[str, str]] = []
    
    if args.prompts_csv:
        print(f"Reading prompts from CSV: {args.prompts_csv}")
        if args.num_prompts is not None:
             print("Warning: --num_prompts is ignored when using --prompts_csv.")
        rows_in = read_prompts_csv(args.prompts_csv)
    
    elif args.dataset_name:
        if args.num_prompts is None:
            raise ValueError("--num_prompts is required when using a Hugging Face dataset.")
        
        # Dolly 15k is the canonical source
        if "dolly" in args.dataset_name.lower():
            dataset_full_name = "databricks/databricks-dolly-15k"
        else:
            dataset_full_name = args.dataset_name

        rows_in = load_and_sample_dolly(dataset_full_name, args.num_prompts)
    
    if not rows_in:
        print("Error: No prompts loaded. Exiting.")
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # 2. Model Configuration and VLLM Initialization
    # -------------------------------------------------------------------------

    # Try to read default model from config/models.json if model_name not provided
    chosen_model = args.model_name
    try:
        # SCRIPT_DIR is the scripts/ directory; config is at repo_root/config/models.json
        config_path = SCRIPT_DIR.parent / "config" / "models.json"
        if chosen_model is None and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as cf:
                cfg_json = json.load(cf)
                chosen_model = cfg_json.get("default_model")
    except Exception:
        chosen_model = chosen_model or None

    if chosen_model is None:
        # fallback to a sensible default if nothing found
        chosen_model = "meta-llama/Llama-3.1-8B-instruct"

    print(f"Using model: {chosen_model}")

    # vLLM init
    # NOTE: Added quantization="awq" to support AWQ models like Llama-2-7B-Chat-AWQ
    llm = LLM(
        model=chosen_model,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.ctx_cap,
        trust_remote_code=False,
        quantization="awq", 
    )
    tok = llm.get_tokenizer()

    cfg = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        natural_stop=args.natural_stop,
        max_time=args.max_time,
        ctx_cap=args.ctx_cap,
        model_name=chosen_model,
    )

    def input_len(p: str) -> int:
        return len(tok.encode(p))

    # Define the actual safe hard limit for Llama 2 models
    # This prevents the CUDA kernel index out-of-bounds error
    SAFE_MAX_LEN = 4096 
    
    # -------------------------------------------------------------------------
    # 3. Prompt Filtering
    # -------------------------------------------------------------------------
    
    # Filter out long prompts that cause CUDA kernel crash
    print(f"Filtering prompts that exceed the model's hard limit of {SAFE_MAX_LEN} tokens...")
    
    rows_filtered = []
    long_prompts_count = 0
    
    for r in rows_in:
        prompt_len = input_len(str(r["prompt_text"]))
        # Filter: keep only prompts that fit within the native context limit
        if prompt_len <= SAFE_MAX_LEN:
            rows_filtered.append(r)
        else:
            long_prompts_count += 1
            
    rows_in = rows_filtered
    print(f"Dropped {long_prompts_count} long prompts. Proceeding with {len(rows_in)} prompts.")
    
    if not rows_in:
        print("Error: All selected prompts were too long. Exiting.")
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # 4. Inference Loop
    # -------------------------------------------------------------------------
    
    def batch_max_tokens(prompts: List[str]) -> int:
        # remaining tokens headroom per prompt; take min across batch to be safe
        lens = [input_len(p) for p in prompts]
        # Calculate max tokens to generate, ensuring total length <= SAFE_MAX_LEN
        return max(1, min(SAFE_MAX_LEN - l - 1 for l in lens))

    total_rows = 0
    # Optional: if file already exists, you may want to remove it to avoid appending
    # to an old run. Comment out if you prefer appending.
    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    for batch_idx, batch in enumerate(chunked(rows_in, args.batch_size)):
        prompts = [str(r["prompt_text"]) for r in batch]
        max_tokens = batch_max_tokens(prompts)

        if args.natural_stop:
            sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
            )
        else:
            sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
                stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
            )

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        batch_rows_out: List[Dict[str, Any]] = []

        for r_in, out in zip(batch, outputs):
            text = out.outputs[0].text if out.outputs else ""
            # metrics may not always be present depending on vLLM version; compute fallbacks
            in_tokens = getattr(out.metrics, "prompt_tokens", None)
            out_tokens = getattr(out.metrics, "generated_tokens", None)
            if in_tokens is None:
                in_tokens = input_len(r_in["prompt_text"])
            if out_tokens is None:
                out_tokens = input_len(text)

            gen = {
                "response_text": text,
                "input_tokens": int(in_tokens),
                "output_tokens": int(out_tokens),
                "ttft_ms": None,  # not exposed in this batch API
                "tpot_ms": e2e_ms / max(1, int(out_tokens)),
                "e2e_latency_ms": float(e2e_ms),
            }
            batch_rows_out.append(build_row(
                prompt_id=str(r_in.get("prompt_id", str(uuid.uuid4()))),
                prompt=str(r_in["prompt_text"]),
                domain=str(r_in.get("domain", "unknown")),
                source=str(r_in.get("source", "synthetic")),
                gen=gen,
                cfg=cfg,
            ))

        # Write this batch immediately
        write_rows_csv_batch(
            args.out_csv,
            batch_rows_out,
            write_header=(batch_idx == 0),  # header only for first batch
        )

        total_rows += len(batch_rows_out)
        print(f"Processed batch {batch_idx + 1}, wrote {len(batch_rows_out)} rows (total {total_rows})")

    print(f"Done. Wrote {total_rows} rows to {args.out_csv}")


if __name__ == "__main__":
    main()