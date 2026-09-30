"""
ablation_study.py — TAILOR component ablations
==============================================
Runs full TAILOR (ADAPTIVE_CFG1) next to three ablated variants on the same
prompts, arrivals and M_Seq configuration, so each row differs from TAILOR
in exactly one component.

Variants
--------
TAILOR        Full system: per-model-threshold classifier + class routing
              + KV-aware spillover/overflow + SM work-stealing.
ABL_T500      Classifier trained with a fixed 500-token threshold instead of
              the per-model balanced threshold. Routing unchanged.
ABL_NO_LOAD   Class routing only: SHORT → SHORT server, LONG → LONG server.
              No KV/load scores, no spillover, no overflow, no SM stealing.
ABL_NO_CLASS  No class-based routing: preferred server alternates
              round-robin, then KV-aware spillover/overflow + SM stealing.

Token budgets (max_tokens 512 SHORT / 2048 LONG) always follow the classifier
used by the variant. ABL_NO_CLASS keeps the default classifier's budgets so
that only the routing decision changes.

Prerequisite for ABL_T500 — train the 500-token classifier on the same
datasets as the default one:
    python phase2_training/train_2class_advancedfeatures.py \\
        --datasets mistral-7b-dolly,mistral-7b-mixed \\
        --threshold 500 --classifiers xgb --deploy_dir deploy/ablation_t500

Usage
-----
    python phase3_evaluation/ablation_study.py --model mistralai/Mistral-7B-Instruct-v0.2 --rate 56
    python phase3_evaluation/ablation_study.py --model ... --rate-start 8 --rate-step 16 --rate-end 72
    python phase3_evaluation/ablation_study.py --model ... --rate 56 --variants TAILOR,ABL_NO_LOAD
    python phase3_evaluation/ablation_study.py --model mistralai/Mistral-Nemo-Instruct-2407 --rate 56 \
        --clf-dir deploy/per_model/Mistral-Nemo-12B --t500-dir deploy/ablation_t500_nemo \
        --instance-cfg instance_configs_nemo.json --out ablation_results_nemo.csv

Output
------
    ablation_results.csv   — one row per (rate, variant), saved after every run.
                             Rerunning with the same --out resumes: finished
                             (rate, variant) pairs are skipped. Delete the file
                             to start over.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq4_evaluation as m

VARIANTS = {
    "TAILOR":       {"classifier": "default", "route_by_class": True,  "load_aware": True},
    "ABL_T500":     {"classifier": "t500",    "route_by_class": True,  "load_aware": True},
    "ABL_NO_LOAD":  {"classifier": "default", "route_by_class": True,  "load_aware": False},
    "ABL_NO_CLASS": {"classifier": "default", "route_by_class": False, "load_aware": True},
}


def routing_label(v: dict) -> str:
    route = "Class" if v["route_by_class"] else "RoundRobin"
    load  = "+RealTimeKV+SM" if v["load_aware"] else ""
    return route + load


def predict(prompts, vec_path=None, xgb_path=None) -> np.ndarray:
    xgb_model, vectorizer = m.load_classifier(vec_path, xgb_path)
    features, _ = m.extract_v13_features(prompts, vectorizer)
    preds = xgb_model.predict(features.values).flatten()
    print(f"   SHORT={int((preds == 0).sum())}  LONG={int((preds == 1).sum())}")
    return preds


def clean_ipc_dirs():
    for d in [m.BARRIER_DIR, m.RESULT_DIR, m.PROMPTS_DIR, m.KV_STAT_DIR]:
        if os.path.exists(d):
            for fname in os.listdir(d):
                try: os.remove(os.path.join(d, fname))
                except Exception: pass


def run_variant(name, variant, rate, prompts, preds, short_m_seq, long_m_seq, debug_kv):
    print("\n" + "=" * 60)
    print(f"  {name}  routing={routing_label(variant)}  "
          f"SHORT_M={short_m_seq}  LONG_M={long_m_seq}  (rate={rate})")
    print("=" * 60)

    clean_ipc_dirs()
    rs, rl = m.launch_adaptive_realtime(
        f"{name.lower()}_r{int(rate)}", short_m_seq, long_m_seq,
        prompts, preds, rate, m.UTIL_PER_GPU, debug_kv,
        route_by_class=variant["route_by_class"],
        load_aware=variant["load_aware"],
    )
    s = m.compute_stats_parallel(rs, rl, "SHORT", "LONG")
    return {
        "Rate":           rate,
        "Method":         name,
        "Routing":        routing_label(variant),
        "Classifier":     variant["classifier"],
        "M_Seq":          f"{short_m_seq}/{long_m_seq}",
        "Pred_SHORT":     int((preds == 0).sum()),
        "Pred_LONG":      int((preds == 1).sum()),
        "N_SHORT":        len(rs.get("ttfts", [])),
        "N_LONG":         len(rl.get("ttfts", [])),
        "TPS":            s["tps"],
        "TTFT_P50_ALL":   s["ttft_p50_all"],
        "TTFT_P95_ALL":   s["ttft_p95_all"],
        "TTFT_P99_ALL":   s["ttft_p99_all"],
        "TTFT_P95_SHORT": s["ttft_p95_SHORT"],
        "TTFT_P95_LONG":  s["ttft_p95_LONG"],
        "E2E_P95_ALL":    s["e2e_p95_all"],
        "KV_Avg":         s["kv_avg_pct"],
        "KV_Peak":        s["kv_peak_pct"],
        "Preemptions":    s["preemptions"],
        "Preempt_Rate%":  s["preemption_rate"],
        "Total_S":        s["wall_clock"],
    }


def print_summary(df: pd.DataFrame):
    print(f"\n  {'Rate':>6}  {'Method':<14} {'TTFT_P95':>10} {'Δ_TTFT':>8}  "
          f"{'E2E_P95':>9} {'Δ_E2E':>7}  {'TPS':>8}")
    for rate, rdf in df.groupby("Rate"):
        base = rdf[rdf["Method"] == "TAILOR"]
        bt = base["TTFT_P95_ALL"].iloc[0] if not base.empty else None
        be = base["E2E_P95_ALL"].iloc[0]  if not base.empty else None
        for _, r in rdf.iterrows():
            dt = f"{(r['TTFT_P95_ALL'] - bt) / bt * 100:+7.1f}%" if bt else f"{'—':>8}"
            de = f"{(r['E2E_P95_ALL'] - be) / be * 100:+6.1f}%"  if be else f"{'—':>7}"
            print(f"  {rate:>6.1f}  {r['Method']:<14} {r['TTFT_P95_ALL']:>10.2f} {dt:>8}  "
                  f"{r['E2E_P95_ALL']:>9.2f} {de:>7}  {r['TPS']:>8.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="TAILOR ablation study")
    parser.add_argument("--model",      type=str,   required=True)
    parser.add_argument("--rate",       type=float, default=None)
    parser.add_argument("--rate-start", type=float, default=8.0)
    parser.add_argument("--rate-step",  type=float, default=16.0)
    parser.add_argument("--rate-end",   type=float, default=72.0)
    parser.add_argument("--variants",   type=str,   default=",".join(VARIANTS),
                        help=f"Comma-separated subset of: {', '.join(VARIANTS)}")
    parser.add_argument("--clf-dir",    type=str,   default=f"{m.PROJECT_ROOT}/deploy",
                        help="Directory holding the default (per-model threshold) v13_vectorizer.pkl + v13_xgb.json")
    parser.add_argument("--instance-cfg", type=str, default=m.INSTANCE_CFG,
                        help="Profiled M_Seq configs from knee_new.py for this model")
    parser.add_argument("--t500-dir",   type=str,   default=f"{m.PROJECT_ROOT}/deploy/ablation_t500",
                        help="Directory holding v13_vectorizer.pkl + v13_xgb.json trained with threshold 500")
    parser.add_argument("--quantization", type=str, default=None,
                        help="vLLM weight quantization for workers, e.g. fp8 (default: none)")
    parser.add_argument("--out",        type=str,   default="ablation_results.csv")
    parser.add_argument("--debug-kv",   action="store_true")
    args = parser.parse_args()

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(names) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")

    clf_paths = {
        "default": (os.path.join(args.clf_dir,  "v13_vectorizer.pkl"), os.path.join(args.clf_dir,  "v13_xgb.json")),
        "t500":    (os.path.join(args.t500_dir, "v13_vectorizer.pkl"), os.path.join(args.t500_dir, "v13_xgb.json")),
    }
    needed = {VARIANTS[n]["classifier"] for n in names}
    for clf in needed:
        missing = [p for p in clf_paths[clf] if not os.path.exists(p)]
        if missing:
            parser.error(f"{clf} classifier missing: {missing} — see the training command in this file's docstring")

    # Workers are launched with --model m.MODEL_ID, so set the module globals.
    m.MODEL_ID        = args.model
    m.INSTANCE_CFG    = args.instance_cfg
    m.QUANTIZATION    = args.quantization
    m.TOKEN_THRESHOLD = m.MODEL_THRESHOLDS.get(args.model, m.TOKEN_THRESHOLD)

    rates = ([args.rate] if args.rate is not None
             else list(np.arange(args.rate_start, args.rate_end + 1e-9, args.rate_step)))

    cfg         = m.load_instance_configs()
    short_m_seq = cfg["SHORT"]["optimal_m_seq"]
    long_m_seq  = cfg["LONG"]["optimal_m_seq"]
    prompts     = m.get_test_pool(m.MODEL_ID)

    preds_by_clf = {}
    for clf in sorted(needed):
        print(f"\n[Classifier] {clf}")
        preds_by_clf[clf] = predict(prompts, *clf_paths[clf])

    print(f"\nModel={m.MODEL_ID}  Prompts={len(prompts)}  Rates={[round(r, 1) for r in rates]}")
    print(f"Variants={names}")

    # Resume: keep rows already in --out and skip (rate, variant) pairs that are done.
    rows = pd.read_csv(args.out).to_dict("records") if os.path.exists(args.out) else []
    done = {(float(r["Rate"]), r["Method"]) for r in rows}
    if done:
        print(f"Resuming: {len(done)} rows already in {args.out}")

    failed = []
    for rate in rates:
        for name in names:
            if (float(rate), name) in done:
                print(f"  ↷ {name} @ rate={rate} already done — skipping")
                continue
            variant = VARIANTS[name]
            try:
                rows.append(run_variant(
                    name, variant, rate, prompts, preds_by_clf[variant["classifier"]],
                    short_m_seq, long_m_seq, args.debug_kv,
                ))
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print(f"\n  💾 {len(rows)} rows → {args.out}")
            except Exception as e:
                print(f"\n  ❌ {name} @ rate={rate} FAILED: {e}")
                failed.append((name, rate))

    if rows:
        print_summary(pd.DataFrame(rows))
    if failed:
        print(f"  Failed: {failed}")
    print(f"✅ {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
