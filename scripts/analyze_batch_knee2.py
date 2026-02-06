#!/usr/bin/env python3
"""
Measure vLLM throughput/latency knee points for prompt length classes.

Key fixes vs. original:
- Sweeps vLLM scheduler concurrency by setting max_num_seqs = batch_size
  (closest equivalent to ASTL NP), instead of only chunking inputs.
- Optionally fixes max_num_batched_tokens across sweeps to avoid confounding.
- Computes per-request latency using RequestOutput.metrics timestamps if present;
  otherwise falls back to batch wall-time (and marks it).
- Knee selection uses a derivative rule (plateau in throughput + rising p95 slope),
  closer to ASTL's knee definition than throughput/(p95+p99).

Assumes CSV contains:
- output_tokens (int) : realized output token count from prior runs
- a prompt column among: prompt_text, prompt, input_text
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover
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
    p = argparse.ArgumentParser(
        description="Run vLLM with varying max_num_seqs (NP-equivalent) to find throughput/latency knees per token-length class.",
    )
    p.add_argument("--input-csv", default="out/dolly_inference_results.csv")
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--batch-sizes", default="1,2,4,8,12,16,20,24,32")
    p.add_argument("--random-seed", type=int, default=42)

    p.add_argument("--model-name", default="TheBloke/Llama-2-7B-Chat-AWQ")
    p.add_argument("--tokenizer-model", default=None)

    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)

    p.add_argument("--ctx-cap", type=int, default=4096)

    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--trust-remote-code", action="store_true")

    # Important: vLLM batching knobs
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="If set, fixes vLLM token budget per iteration across sweeps (recommended).",
    )

    p.add_argument("--stop-on-eos", action="store_true")
    p.add_argument("--warmup-iters", type=int, default=1, help="Warmup generate() calls per setting (not recorded).")

    # Knee detection thresholds (derivative-based)
    p.add_argument(
        "--eps-throughput-gain",
        type=float,
        default=0.01,
        help="Throughput plateau threshold as relative gain: (T_k - T_{k-1})/T_{k-1} < eps.",
    )
    p.add_argument(
        "--tau-p95-slope-ms",
        type=float,
        default=2.0,
        help="Tail-slope threshold in ms per step: p95_k - p95_{k-1} > tau.",
    )

    p.add_argument("--output-json", default=None)
    return p.parse_args()


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


def sample_class_prompts(df: pd.DataFrame, label: LengthLabel, n_samples: int, seed: int) -> pd.DataFrame:
    subset = df[df["length_class"] == label]
    if subset.empty:
        raise ValueError(f"No rows available for length class '{label}'.")
    replace = len(subset) < n_samples
    sampled = subset.sample(n=n_samples, replace=replace, random_state=seed)
    return sampled.reset_index(drop=True)


def chunk_max_new_tokens(prompt_token_counts: Sequence[int], ctx_cap: int, requested_max_new: int) -> int:
    # Keep enough headroom for the largest prompt in this chunk
    headroom = min(ctx_cap - c - 1 for c in prompt_token_counts)
    headroom = max(1, headroom)
    return max(1, min(headroom, requested_max_new))


def build_llm(
    model_name: str,
    ctx_cap: int,
    tensor_parallel_size: int,
    dtype: str,
    trust_remote_code: bool,
    max_num_seqs: int,
    max_num_batched_tokens: Optional[int],
) -> LLM:
    # NOTE: LLM() args vary a bit by vLLM version. These are common engine args.
    kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        max_model_len=ctx_cap,
        max_num_seqs=max_num_seqs,  # NP-equivalent cap
    )
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    return LLM(**kwargs)


def extract_request_latency_ms(out) -> Optional[float]:
    """
    Try to compute per-request latency from vLLM metrics timestamps.
    Different vLLM versions expose different fields. We probe safely.
    """
    m = getattr(out, "metrics", None)
    if m is None:
        return None

    # Common naming patterns seen in some versions:
    # - arrival_time / first_token_time / finished_time (seconds)
    # - prompt_start_time / finish_time
    arrival = getattr(m, "arrival_time", None)
    finished = getattr(m, "finished_time", None)

    if arrival is not None and finished is not None:
        try:
            return float(finished - arrival) * 1000.0
        except Exception:
            return None

    # Fallback variants
    start = getattr(m, "start_time", None) or getattr(m, "prompt_start_time", None)
    end = getattr(m, "end_time", None) or getattr(m, "finish_time", None)
    if start is not None and end is not None:
        try:
            return float(end - start) * 1000.0
        except Exception:
            return None

    return None


def run_setting(
    llm: LLM,
    tokenizer,
    prompts_df: pd.DataFrame,
    batch_size: int,
    ctx_cap: int,
    requested_max_new: int,
    sampling_base: Dict[str, float],
    stop_token_ids: Optional[List[int]],
    warmup_iters: int,
) -> Dict[str, float]:
    prompts = prompts_df["prompt_text"].tolist()
    prompt_token_counts = prompts_df["prompt_token_len"].tolist()
    if not prompts:
        raise ValueError("No prompts provided to run through vLLM.")

    # Warmup (not recorded)
    for _ in range(max(0, warmup_iters)):
        chunk = list(zip(prompts[: min(len(prompts), batch_size)], prompt_token_counts[: min(len(prompts), batch_size)]))
        if not chunk:
            break
        chunk_prompts = [c[0] for c in chunk]
        chunk_token_counts = [c[1] for c in chunk]
        max_new = chunk_max_new_tokens(chunk_token_counts, ctx_cap, requested_max_new)
        sampling = SamplingParams(
            temperature=sampling_base["temperature"],
            top_p=sampling_base["top_p"],
            max_tokens=max_new,
            stop_token_ids=stop_token_ids,
        )
        _ = llm.generate(chunk_prompts, sampling)

    latencies_ms: List[float] = []
    used_batch_fallback = 0

    total_time_s = 0.0
    output_token_counts: List[int] = []
    input_token_total = 0

    # We still chunk submissions to keep memory stable, but NP-equivalent is max_num_seqs (engine cap).
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

        # Per-request latency if metrics timestamps exist; else fallback to batch wall-time.
        for out in outputs:
            per_req = extract_request_latency_ms(out)
            if per_req is None:
                per_req = elapsed_s * 1000.0
                used_batch_fallback += 1
            latencies_ms.append(float(per_req))

            text = out.outputs[0].text if out.outputs else ""
            gen_tokens = getattr(getattr(out, "metrics", None), "generated_tokens", None)
            if gen_tokens is None:
                gen_tokens = count_tokens(text, tokenizer)
            output_token_counts.append(int(gen_tokens))

        for pt in chunk_token_counts:
            input_token_total += int(pt)

    total_generated = int(sum(output_token_counts))
    total_tokens = int(total_generated + input_token_total)

    total_tok_s = float(total_tokens / total_time_s) if total_time_s > 0 else float("inf")
    out_tok_s = float(total_generated / total_time_s) if total_time_s > 0 else float("inf")

    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))

    return {
        "batch_size": int(batch_size),
        "samples": int(len(prompts)),
        "avg_output_tokens": float(total_generated / max(1, len(prompts))),
        "throughput_total_tok_s": total_tok_s,
        "throughput_output_tok_s": out_tok_s,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "total_time_s": float(total_time_s),
        "used_batch_fallback_count": int(used_batch_fallback),
    }


def pick_knee_derivative(
    rows: List[Dict[str, float]],
    eps_throughput_gain: float,
    tau_p95_slope_ms: float,
    throughput_key: str = "throughput_output_tok_s",
) -> Tuple[int, Dict[str, float]]:
    """
    Knee = smallest batch_size where:
      relative throughput gain < eps  AND  p95 increase > tau
    If no such point, fallback to argmax throughput.
    """
    rows_sorted = sorted(rows, key=lambda r: r["batch_size"])
    if len(rows_sorted) < 2:
        return int(rows_sorted[0]["batch_size"]), rows_sorted[0]

    for i in range(1, len(rows_sorted)):
        prev = rows_sorted[i - 1]
        cur = rows_sorted[i]

        T_prev = float(prev[throughput_key])
        T_cur = float(cur[throughput_key])
        p_prev = float(prev["p95_latency_ms"])
        p_cur = float(cur["p95_latency_ms"])

        rel_gain = (T_cur - T_prev) / max(1e-9, T_prev)
        dp95 = p_cur - p_prev

        if rel_gain < eps_throughput_gain and dp95 > tau_p95_slope_ms:
            return int(cur["batch_size"]), cur

    best = max(rows_sorted, key=lambda r: float(r[throughput_key]))
    return int(best["batch_size"]), best


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")

    tokenizer_id = args.tokenizer_model or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)

    df = pd.read_csv(args.input_csv)
    if "output_tokens" not in df.columns:
        raise ValueError("CSV must contain an 'output_tokens' column.")
    prompt_col = detect_prompt_column(df)

    df = df.dropna(subset=[prompt_col, "output_tokens"]).copy()
    df["prompt_text"] = df[prompt_col].astype(str)
    df["output_tokens"] = df["output_tokens"].astype(int)

    df["prompt_token_len"] = df["prompt_text"].apply(lambda txt: count_tokens(txt, tokenizer))
    before = len(df)
    df = df[df["prompt_token_len"] < args.ctx_cap - 1].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows that exceeded ctx cap {args.ctx_cap}.")
    if df.empty:
        raise RuntimeError("No prompts remain after filtering by context window.")

    df["length_class"] = df["output_tokens"].apply(lambda x: classify_length(x, DEFAULT_CLASSES))
    df = df.dropna(subset=["length_class"])
    if df.empty:
        raise RuntimeError("No prompts were assigned to any length class.")

    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",") if bs.strip()]
    if not batch_sizes:
        raise ValueError("No batch sizes provided.")

    print(f"Evaluating max_num_seqs candidates: {batch_sizes}")

    stop_token_ids = None
    if args.stop_on_eos and tokenizer.eos_token_id is not None:
        stop_token_ids = [tokenizer.eos_token_id]

    sampling_base = {"temperature": float(args.temperature), "top_p": float(args.top_p)}

    results: Dict[str, Dict[str, object]] = {}

    for cfg in DEFAULT_CLASSES:
        label = cfg.name
        sampled = sample_class_prompts(df, label, args.samples_per_class, args.random_seed)
        print(f"\n=== {label.upper()} CLASS : {len(sampled)} prompts ===")

        class_metrics: List[Dict[str, float]] = []

        for bs in batch_sizes:
            print(f"Running max_num_seqs={bs} ({label}) ...")

            # IMPORTANT: max_num_seqs is set at engine construction time.
            llm = build_llm(
                model_name=args.model_name,
                ctx_cap=args.ctx_cap,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                max_num_seqs=bs,
                max_num_batched_tokens=args.max_num_batched_tokens,
            )

            metrics = run_setting(
                llm=llm,
                tokenizer=tokenizer,
                prompts_df=sampled,
                batch_size=bs,  # also used for submission chunking
                ctx_cap=args.ctx_cap,
                requested_max_new=args.max_new_tokens,
                sampling_base=sampling_base,
                stop_token_ids=stop_token_ids,
                warmup_iters=args.warmup_iters,
            )
            class_metrics.append(metrics)

        knee_bs, knee_row = pick_knee_derivative(
            class_metrics,
            eps_throughput_gain=args.eps_throughput_gain,
            tau_p95_slope_ms=args.tau_p95_slope_ms,
            throughput_key="throughput_output_tok_s",
        )

        print("max_num_seqs\tout_tok/s\ttotal_tok/s\tp95_ms\tp99_ms\tfallback_lat")
        for m in sorted(class_metrics, key=lambda r: r["batch_size"]):
            marker = "*" if int(m["batch_size"]) == knee_bs else " "
            print(
                f"{int(m['batch_size']):>11}{marker}\t"
                f"{m['throughput_output_tok_s']:>8.2f}\t"
                f"{m['throughput_total_tok_s']:>11.2f}\t"
                f"{m['p95_latency_ms']:>7.1f}\t"
                f"{m['p99_latency_ms']:>7.1f}\t"
                f"{int(m['used_batch_fallback_count']):>12}"
            )

        print(f"-> Suggested knee for {label}: max_num_seqs={knee_bs}")

        results[label] = {
            "samples": len(sampled),
            "metrics": class_metrics,
            "knee_max_num_seqs": knee_bs,
            "knee_row": knee_row,
            "note": (
                "Per-request latency uses vLLM metrics timestamps when available; "
                "fallback_lat indicates how many requests used batch wall-time fallback."
            ),
        }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
