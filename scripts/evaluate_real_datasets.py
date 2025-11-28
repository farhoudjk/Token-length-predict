#!/usr/bin/env python3
"""Evaluate the length classifier on several public datasets.

The script loads a trained classifier (RF or XGBoost), computes the same
feature set that was used during training, and compares the predicted
length class (<500 vs. >500 tokens) against the actual token count of
ground-truth responses contained in real datasets.

Example:
    python scripts/evaluate_real_datasets.py \
        --model-path out/clf_results/RandomForest_model.pkl \
        --feature-json out/clf_results/feature_columns.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing as mp
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None

try:
    from scripts.feature_extractor import extract_prompt_features
except ImportError:
    # Allow running as `python scripts/evaluate_real_datasets.py` from repo root.
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir))
    from feature_extractor import extract_prompt_features


FeatureDict = Dict[str, float]


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_alpaca_prompt(row: Dict[str, Any]) -> str:
    instruction = (row.get("instruction") or "").strip()
    input_text = (row.get("input") or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def build_alpaca_output(row: Dict[str, Any]) -> str:
    return (row.get("output") or "").strip()


def build_dolly_prompt(row: Dict[str, Any]) -> str:
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def build_dolly_output(row: Dict[str, Any]) -> str:
    return (row.get("response") or "").strip()


def build_eli5_prompt(row: Dict[str, Any]) -> str:
    title = (row.get("title") or "").strip()
    body = (row.get("selftext") or "").strip()
    if body:
        return f"{title}\n\n{body}"
    return title


def build_eli5_output(row: Dict[str, Any]) -> str:
    answers = row.get("answers") or {}
    texts = answers.get("text") or []
    scores = answers.get("score") or []
    if isinstance(texts, list) and texts:
        if isinstance(scores, list) and len(scores) == len(texts):
            best_idx = int(np.argmax(scores))
        else:
            best_idx = 0
        return (texts[best_idx] or "").strip()
    return ""


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a dataset + how to build prompts/outputs."""

    name: str
    domain: str
    prompt_fn: Callable[[Dict[str, Any]], str]
    output_fn: Callable[[Dict[str, Any]], str]
    split: str = "train"
    hf_id: Optional[str] = None
    config: Optional[str] = None
    data_files: Optional[Dict[str, Any]] = None
    loader_name: Optional[str] = None
    sample_size: int = 200


class VLLMGenerator:
    """Thin wrapper that batches prompts through vLLM."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        ctx_cap: int,
        tensor_parallel_size: int,
        batch_size: int,
    ) -> None:
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vLLM is not installed. Install it or run without --generate-with-vllm.")

        self.model_id = model_id
        self.batch_size = max(1, batch_size)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        self.llm = LLM(
            model=model_id,
            dtype="float16",
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=ctx_cap,
            trust_remote_code=False,
        )
        self.source_name = f"vllm:{model_id}"

    def generate(self, prompts: List[str]) -> List[str]:
        outputs: List[str] = []
        for chunk in chunked(prompts, self.batch_size):
            request_outputs = self.llm.generate(chunk, self.sampling_params)
            for req in request_outputs:
                if req.outputs:
                    text = req.outputs[0].text
                else:
                    text = ""
                outputs.append((text or "").strip())
        return outputs


def build_ultrachat_prompt(row: Dict[str, Any]) -> str:
    messages = row.get("messages") or []
    convo_parts = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        if role == "assistant":
            break
        content = (msg.get("content") or "").strip()
        if content:
            convo_parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(convo_parts)


def build_ultrachat_output(row: Dict[str, Any]) -> str:
    messages = row.get("messages") or []
    for msg in reversed(messages):
        if (msg.get("role") or "").lower() == "assistant":
            return (msg.get("content") or "").strip()
    return ""


EVAL_DATASETS: Dict[str, DatasetSpec] = {
    "alpaca": DatasetSpec(
        name="alpaca",
        hf_id="tatsu-lab/alpaca",
        split="train",
        domain="general_instruction",
        prompt_fn=build_alpaca_prompt,
        output_fn=build_alpaca_output,
    ),
    "dolly": DatasetSpec(
        name="dolly",
        hf_id="databricks/databricks-dolly-15k",
        split="train",
        domain="business_conversation",
        prompt_fn=build_dolly_prompt,
        output_fn=build_dolly_output,
    ),
    "ultrachat": DatasetSpec(
        name="ultrachat",
        domain="multi_turn_support",
        loader_name="parquet",
        data_files={
            "train": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/train_sft-00000-of-00003-a3ecf92756993583.parquet"
        },
        split="train",
        prompt_fn=build_ultrachat_prompt,
        output_fn=build_ultrachat_output,
        sample_size=150,
    ),
}


def load_model_bundle(model_path: str, feature_json: str) -> tuple[Any, Optional[Any], List[str]]:
    """Load classifier, optional scaler, and feature ordering."""
    bundle = joblib.load(model_path)
    scaler = None
    features: Optional[List[str]] = None
    model = bundle

    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        scaler = bundle.get("scaler")
        features = bundle.get("features")

    if features is None:
        if not os.path.exists(feature_json):
            raise FileNotFoundError(
                f"Feature list not found. Provide --feature-json or bundle with feature metadata (got: {model_path})."
            )
        with open(feature_json, "r", encoding="utf-8") as fh:
            features = json.load(fh)

    return model, scaler, features


def prepare_features(prompts: Iterable[str]) -> List[FeatureDict]:
    """Compute admission-time features for each prompt."""
    feats: List[FeatureDict] = []
    for prompt in prompts:
        data = extract_prompt_features(prompt)
        data["prompt_char_len"] = len(prompt)
        data["prompt_word_count"] = len(prompt.split())
        data["prompt_has_question_mark"] = int("?" in prompt)
        feats.append(data)
    return feats


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def materialize_rows(ds, spec: DatasetSpec, sample_size: int) -> List[Dict[str, Any]]:
    """Shuffle dataset and keep rows that have both prompt + output."""
    rows: List[Dict[str, Any]] = []
    shuffled = ds.shuffle(seed=42)
    for row in shuffled:
        prompt = spec.prompt_fn(row).strip()
        output = spec.output_fn(row).strip()
        if not prompt or not output:
            continue
        rows.append({"prompt": prompt, "reference_output": output})
        if len(rows) >= sample_size:
            break
    return rows


def evaluate_dataset(
    spec: DatasetSpec,
    model,
    scaler,
    feature_cols: List[str],
    tokenizer,
    threshold: int,
    output_dir: str,
    text_generator: Optional[VLLMGenerator] = None,
) -> Dict[str, Any]:
    """Evaluate classifier on one dataset and persist row-level results."""
    if spec.data_files:
        loader_name = spec.loader_name or "json"
        ds = load_dataset(loader_name, data_files=spec.data_files, split=spec.split)
    else:
        if not spec.hf_id:
            raise ValueError(f"Dataset '{spec.name}' is missing hf_id or data_files.")
        load_kwargs = {"split": spec.split}
        if spec.config:
            ds = load_dataset(spec.hf_id, spec.config, **load_kwargs)
        else:
            ds = load_dataset(spec.hf_id, **load_kwargs)
    rows = materialize_rows(ds, spec, spec.sample_size)
    if not rows:
        raise RuntimeError(f"No valid rows collected for dataset {spec.name}")

    prompts = [r["prompt"] for r in rows]
    reference_outputs = [r["reference_output"] for r in rows]

    actual_outputs = reference_outputs
    output_source = "dataset"
    if text_generator is not None:
        actual_outputs = text_generator.generate(prompts)
        if len(actual_outputs) != len(prompts):
            raise RuntimeError(
                f"Generator returned {len(actual_outputs)} outputs for {len(prompts)} prompts on dataset {spec.name}"
            )
        output_source = text_generator.source_name

    feats = prepare_features(prompts)
    feat_df = pd.DataFrame(feats)
    feat_df = feat_df.reindex(columns=feature_cols, fill_value=0.0)
    X = feat_df
    if scaler is not None:
        scaled = scaler.transform(X)
        X = pd.DataFrame(scaled, columns=feature_cols)

    predictions = model.predict(X)
    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X)[:, 1]
        except Exception:
            probas = None

    true_counts = [count_tokens(txt, tokenizer) for txt in actual_outputs]
    y_true = np.array([int(cnt > threshold) for cnt in true_counts])
    y_pred = np.array(predictions).astype(int)

    acc = accuracy_score(y_true, y_pred)
    report_txt = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred).tolist()

    os.makedirs(output_dir, exist_ok=True)
    df_out = pd.DataFrame(
        {
            "dataset": spec.name,
            "domain": spec.domain,
            "prompt": prompts,
            "reference_output": reference_outputs,
            "evaluated_output": actual_outputs,
            "output_source": output_source,
            "true_output_tokens": true_counts,
            "true_label_long": y_true,
            "predicted_label_long": y_pred,
        }
    )
    if probas is not None:
        df_out["predicted_prob_long"] = probas
    df_path = os.path.join(output_dir, f"{spec.name}_predictions.csv")
    df_out.to_csv(df_path, index=False)

    return {
        "dataset": spec.name,
        "domain": spec.domain,
        "samples": len(rows),
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "report": report_txt,
        "predictions_csv": df_path,
        "output_source": output_source,
    }


def load_default_model_id() -> str:
    """Mirror the default model selection from run_inference_vllm."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "config" / "models.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
                model_id = cfg.get("default_model")
                if model_id:
                    return model_id
        except Exception:
            pass
    return "TheBloke/Llama-2-7B-Chat-AWQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the length classifier on multiple datasets.")
    parser.add_argument(
        "--model-path",
        default="out/clf_results/RandomForest_model.pkl",
        help="Path to the trained classifier bundle (joblib).",
    )
    parser.add_argument(
        "--feature-json",
        default="out/clf_results/feature_columns.json",
        help="JSON file containing the ordered list of feature columns.",
    )
    parser.add_argument(
        "--datasets",
        default="alpaca,dolly,ultrachat",
        help=f"Comma-separated subset of datasets to run. Available: {','.join(EVAL_DATASETS)}",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=0,
        help="Override the default sample count for each dataset (<=0 keeps the dataset-specific default).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=500,
        help="Token threshold that defines the long-output label.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer/model id to use for token counting. Defaults to config/models.json when absent.",
    )
    parser.add_argument(
        "--generate-with-vllm",
        action="store_true",
        help="If set, ignore dataset answers and instead run prompts through vLLM to obtain ground-truth outputs.",
    )
    parser.add_argument(
        "--generation-model",
        default=None,
        help="Model id to load with vLLM when --generate-with-vllm is set. Defaults to config/models.json.",
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=512, help="Max new tokens for vLLM runs.")
    parser.add_argument("--generation-temperature", type=float, default=0.7, help="Sampling temperature for vLLM.")
    parser.add_argument("--generation-top-p", type=float, default=0.95, help="Top-p for vLLM sampling.")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM.generate when --generate-with-vllm is set.",
    )
    parser.add_argument(
        "--generation-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size passed to vLLM (set >1 for multi-GPU runs).",
    )
    parser.add_argument(
        "--generation-ctx-cap",
        type=int,
        default=4096,
        help="Soft context window passed to vLLM for --generate-with-vllm.",
    )
    parser.add_argument(
        "--output-dir",
        default="out/real_dataset_eval",
        help="Directory where per-dataset CSV predictions and summary JSON are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, scaler, features = load_model_bundle(args.model_path, args.feature_json)

    tokenizer_model = args.tokenizer_model or load_default_model_id()
    print(f"Using tokenizer/model for token counting: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)

    text_generator: Optional[VLLMGenerator] = None
    if args.generate_with_vllm:
        generation_model = args.generation_model or load_default_model_id()
        print(f"Generating ground-truth outputs with vLLM model: {generation_model}")
        text_generator = VLLMGenerator(
            model_id=generation_model,
            max_new_tokens=args.generation_max_new_tokens,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            ctx_cap=args.generation_ctx_cap,
            tensor_parallel_size=args.generation_tensor_parallel_size,
            batch_size=args.generation_batch_size,
        )

    requested = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    summaries: List[Dict[str, Any]] = []

    for name in requested:
        if name not in EVAL_DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(EVAL_DATASETS)}")
        spec = EVAL_DATASETS[name]
        if args.samples_per_dataset > 0:
            spec = replace(spec, sample_size=args.samples_per_dataset)
        print(f"Evaluating dataset '{spec.name}' with {spec.sample_size} samples...")
        summary = evaluate_dataset(
            spec=spec,
            model=model,
            scaler=scaler,
            feature_cols=features,
            tokenizer=tokenizer,
            threshold=args.threshold,
            output_dir=args.output_dir,
            text_generator=text_generator,
        )
        print(summary["report"])
        summaries.append(summary)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
