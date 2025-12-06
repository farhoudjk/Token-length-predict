#!/usr/bin/env python3
"""Measure vLLM throughput/latency knee points for prompt length classes.

We sample prompts from the Dolly inference CSV, bucket them into three output
token classes (<=500, 501-2000, >2000), and for each class replay the prompts
through vLLM while sweeping over different `batch_size` (= parallel sequences).
Throughput (tokens/sec) and tail latencies (p95/p99) are measured from the live
generation timings to surface the best-performing batch size for each class.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("vLLM must be installed to run this script.") from exc


LengthLabel = str
PROMPT_COL_CANDIDATES = ["prompt_text", "prompt", "input_text"]


@dataclass(frozen=True)
class ClassConfig:
    name: LengthLabel
    lower_bound: int
    upper_bound: int | None  # inclusive upper bound (None = infinity)


DEFAULT_CLASSES: Sequence[ClassConfig] = (
    ClassConfig(name="short", lower_bound=0, upper_bound=500),
    ClassConfig(name="medium", lower_bound=501, upper_bound=2000),
    ClassConfig(name="long", lower_bound=2001, upper_bound=None),
)


def chunked(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM with varying batch sizes to find throughput/latency knees per token-length class.",
    )
    parser.add_argument(
        "--input-csv",
        default="out/dolly_inference_results_llama2_awq (1).csv",
        help="Path to the Dolly inference CSV.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=200,
        help="Number of prompts to sample for each class (with replacement if needed).",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,12,16,20,24,32,40",
        help="Comma-separated candidate batch/parallel sequence sizes.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility.",
    )
    parser.add_argument(
        "--model-name",
        default="TheBloke/Llama-2-7B-Chat-AWQ",
        help="Model name or path to load with vLLM.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer/model id for counting tokens (defaults to --model-name).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per prompt (cap is also enforced by ctx window).",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument(
        "--ctx-cap",
        type=int,
        default=4096,
        help="Maximum context window to respect when batching prompts.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Model dtype hint passed to vLLM (e.g., float16, bfloat16).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models that require remote code.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to dump the per-class metrics as JSON.",
    )
    parser.add_argument(
        "--stop-on-eos",
        action="store_true",
        help="Stop generation when the tokenizer EOS token is produced.",
    )
    return parser.parse_args()


def classify_length(count: int, classes: Sequence[ClassConfig]) -> LengthLabel | None:
    for cfg in classes:
        if cfg.upper_bound is None:
            if count >= cfg.lower_bound:
                return cfg.name
        elif cfg.lower_bound <= count <= cfg.upper_bound:
            return cfg.name
    return None


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def detect_prompt_column(df: pd.DataFrame) -> str:
    for col in PROMPT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Unable to locate a prompt text column. Expected one of: {PROMPT_COL_CANDIDATES}")


def sample_class_prompts(
    df: pd.DataFrame, label: LengthLabel, n_samples: int, seed: int
) -> pd.DataFrame:
    subset = df[df["length_class"] == label]
    if subset.empty:
        raise ValueError(f"No rows available for length class '{label}'.")
    if len(subset) < n_samples:
        sampled = subset.sample(n=n_samples, replace=True, random_state=seed)
    else:
        sampled = subset.sample(n=n_samples, replace=False, random_state=seed)
    return sampled.reset_index(drop=True)


def chunk_max_new_tokens(
    prompt_token_counts: Sequence[int],
    ctx_cap: int,
    requested_max_new: int,
) -> int:
    headroom = min(ctx_cap - count - 1 for count in prompt_token_counts)
    headroom = max(1, headroom)
    return max(1, min(headroom, requested_max_new))


def run_vllm_batches(
    llm: LLM,
    tokenizer,
    prompts_df: pd.DataFrame,
    batch_size: int,
    ctx_cap: int,
    requested_max_new: int,
    sampling_base: Dict[str, Optional[float]],
    stop_token_ids: Optional[List[int]],
) -> Dict[str, float]:
    prompts = prompts_df["prompt_text"].tolist()
    prompt_token_counts = prompts_df["prompt_token_len"].tolist()
    if not prompts:
        raise ValueError("No prompts provided to run through vLLM.")

    latencies_ms: List[float] = []
    total_time_s = 0.0
    output_token_counts: List[int] = []
    input_token_total = 0

    for chunk in chunked(list(zip(prompts, prompt_token_counts)), batch_size):
        chunk_prompts = [c[0] for c in chunk]
        chunk_token_counts = [c[1] for c in chunk]
        max_new = chunk_max_new_tokens(chunk_token_counts, ctx_cap, requested_max_new)
        sampling = SamplingParams(
            temperature=sampling_base["temperature"],
            top_p=sampling_base["top_p"],
            max_tokens=max_new,
            stop_token_ids=stop_token_ids,
        )

        start = time.perf_counter()
        outputs = llm.generate(chunk_prompts, sampling)
        elapsed_s = time.perf_counter() - start
        total_time_s += elapsed_s
        chunk_latency_ms = elapsed_s * 1000.0
        latencies_ms.extend([chunk_latency_ms] * len(chunk_prompts))

        for prompt_tokens, out in zip(chunk_token_counts, outputs):
            text = out.outputs[0].text if out.outputs else ""
            gen_tokens = getattr(out.metrics, "generated_tokens", None)
            if gen_tokens is None:
                gen_tokens = count_tokens(text, tokenizer)
            output_token_counts.append(int(gen_tokens))
            input_token_total += int(prompt_tokens)

    total_generated = sum(output_token_counts)
    total_tokens = total_generated + input_token_total
    throughput = float(total_tokens / total_time_s) if total_time_s > 0 else float("inf")
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    score = throughput / (p95 + p99) if (p95 + p99) > 0 else float("inf")

    return {
        "batch_size": batch_size,
        "samples": len(prompts),
        "avg_output_tokens": total_generated / max(1, len(prompts)),
        "throughput_tokens_per_s": throughput,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "score": score,
        "total_time_s": total_time_s,
    }


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")

    tokenizer_id = args.tokenizer_model or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    prompt_col = None

    df = pd.read_csv(args.input_csv)
    if "output_tokens" not in df.columns:
        raise ValueError("CSV must contain an 'output_tokens' column.")
    prompt_col = detect_prompt_column(df)

    df = df.dropna(subset=[prompt_col, "output_tokens"]).copy()
    df["prompt_text"] = df[prompt_col].astype(str)
    df["output_tokens"] = df["output_tokens"].astype(int)
    print(f"Loaded {len(df)} rows from {args.input_csv} (prompt column='{prompt_col}').")

    df["prompt_token_len"] = df["prompt_text"].apply(lambda txt: count_tokens(txt, tokenizer))
    before_filter = len(df)
    df = df[df["prompt_token_len"] < args.ctx_cap - 1].copy()
    filtered = before_filter - len(df)
    if filtered:
        print(f"Dropped {filtered} rows that exceeded ctx cap {args.ctx_cap}.")
    if df.empty:
        raise RuntimeError("No prompts remain after filtering by context window.")

    df["length_class"] = df["output_tokens"].apply(lambda x: classify_length(x, DEFAULT_CLASSES))
    df = df.dropna(subset=["length_class"])
    if df.empty:
        raise RuntimeError("No prompts were assigned to any length class.")

    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",") if bs.strip()]
    if not batch_sizes:
        raise ValueError("No batch sizes provided.")

    print(f"Evaluating batch candidates: {batch_sizes}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.ctx_cap,
    )
    stop_token_ids = None
    if args.stop_on_eos and tokenizer.eos_token_id is not None:
        stop_token_ids = [tokenizer.eos_token_id]

    sampling_base = {"temperature": args.temperature, "top_p": args.top_p}

    results: Dict[str, Dict[str, object]] = {}

    for cfg in DEFAULT_CLASSES:
        label = cfg.name
        sampled = sample_class_prompts(df, label, args.samples_per_class, args.random_seed)
        if sampled.empty:
            continue
        print(f"\n=== {label.upper()} CLASS : {len(sampled)} prompts ===")

        class_metrics: List[Dict[str, float]] = []
        for batch_size in batch_sizes:
            print(f"Running batch_size={batch_size} ({label})...")
            metrics = run_vllm_batches(
                llm=llm,
                tokenizer=tokenizer,
                prompts_df=sampled,
                batch_size=batch_size,
                ctx_cap=args.ctx_cap,
                requested_max_new=args.max_new_tokens,
                sampling_base=sampling_base,
                stop_token_ids=stop_token_ids,
            )
            class_metrics.append(metrics)

        best = max(class_metrics, key=lambda m: m["score"])
        print("batch_size\tthroughput(tok/s)\tp95_ms\tp99_ms\tscore")
        for m in class_metrics:
            marker = "*" if m["batch_size"] == best["batch_size"] else " "
            print(
                f"{m['batch_size']:>10}{marker}\t"
                f"{m['throughput_tokens_per_s']:>18.2f}\t"
                f"{m['p95_latency_ms']:>7.1f}\t"
                f"{m['p99_latency_ms']:>7.1f}\t"
                f"{m['score']:>8.4f}"
            )
        print(f"-> Suggested knee batch size for {label}: {best['batch_size']} (score={best['score']:.4f})")
        results[label] = {"samples": len(sampled), "metrics": class_metrics, "best_batch_size": best["batch_size"]}

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
