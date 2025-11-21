#!/usr/bin/env python3
import os, sys, csv, json, time, uuid, argparse
from pathlib import Path
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

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
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    ap.add_argument("--prompts_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-instruct")
    ap.add_argument("--batch_size", type=int, default=16, help="Parallel sequences per batch")
    ap.add_argument("--ctx_cap", type=int, default=4096, help="Soft cap on max model length for speed")
    ap.add_argument("--natural_stop", action="store_true", help="Greedy decoding (temperature=0)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_time", type=float, default=None, help="(Not enforced per-request by vLLM)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    rows_in = read_prompts_csv(args.prompts_csv)

    # vLLM init
    llm = LLM(
        model=args.model_name,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.ctx_cap,
        trust_remote_code=False,
    )
    tok = llm.get_tokenizer()

    cfg = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        natural_stop=args.natural_stop,
        max_time=args.max_time,
        ctx_cap=args.ctx_cap,
        model_name=args.model_name,
    )

    def input_len(p: str) -> int:
        return len(tok.encode(p))

    def batch_max_tokens(prompts: List[str]) -> int:
        # remaining tokens headroom per prompt; take min across batch to be safe
        lens = [input_len(p) for p in prompts]
        return max(1, min(args.ctx_cap - l - 1 for l in lens))

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
