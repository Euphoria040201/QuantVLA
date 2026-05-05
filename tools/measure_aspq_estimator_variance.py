#!/usr/bin/env python
"""Measure repeatability of the ASPQ Hutchinson estimator.

This keeps the FP policy loaded once, then repeats the ASPQ Jacobian collection
multiple times with different RNG seeds. We report per-layer mean/std/CV for
`tr(M)` and the derived ASPQ score so we can see how noisy the `g = J^T u`
estimator is in practice.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    DEFAULT_EXCLUDE_REGEX,
    DEFAULT_INCLUDE_REGEX,
    ensure_libero_runtime,
    load_libero_samples,
    load_policy,
)
from tools.aspq_jacobian_sanity import (  # noqa: E402
    LayerAccum,
    collect_one_sample,
    family_of,
    load_data_config,
    resolve_target_layers,
    short_layer_name,
    spearman,
    unwrap_no_grad_methods,
    restore_no_grad_methods,
)

DEFAULT_LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/work/mingze/LIBERO")
DEFAULT_LIBERO_DATASET = f"{DEFAULT_LIBERO_ROOT}/datasets/lerobot_libero_10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--data-source", default="libero", choices=["libero"])
    p.add_argument("--dataset-path", default=DEFAULT_LIBERO_DATASET)
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task", choices=["sequential", "one_per_task"])
    p.add_argument("--libero-resolution", type=int, default=256)

    p.add_argument("--include-regex", default=DEFAULT_INCLUDE_REGEX)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)

    p.add_argument("--random-dirs", type=int, default=8)
    p.add_argument("--token-cap", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5, help="For rank-stability summaries.")
    return p.parse_args()


def summarize_once(layer_names: list[str], accums: dict[str, LayerAccum]) -> list[dict]:
    rows = []
    for name in layer_names:
        ac = accums[name]
        if ac.n_tokens == 0:
            continue
        tr_M_norm = ac.tr_M / max(ac.n_tokens, 1)
        tr_H_norm = ac.tr_H / max(ac.n_tokens, 1)
        score = tr_M_norm * tr_H_norm / max(ac.d_out, 1)
        rows.append(
            {
                "layer_name": name,
                "family": family_of(name),
                "d_in": ac.d_in,
                "d_out": ac.d_out,
                "n_calib_tokens": ac.n_tokens,
                "tr_M": tr_M_norm,
                "tr_H": tr_H_norm,
                "score": score,
            }
        )
    return rows


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _std(xs: list[float]) -> float:
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=1)) if len(xs) > 1 else 0.0


def _cv(mean: float, std: float) -> float:
    return std / abs(mean) if abs(mean) > 1e-12 else float("nan")


def aggregate_runs(run_rows: list[list[dict]], top_k: int) -> tuple[list[dict], dict]:
    by_layer: dict[str, dict[str, list[float]]] = {}
    top_sets = []
    score_vectors = []
    ordered_names = [row["layer_name"] for row in run_rows[0]]

    for rows in run_rows:
        sorted_rows = sorted(rows, key=lambda r: -r["score"])
        top_sets.append(set(r["layer_name"] for r in sorted_rows[:top_k]))
        score_vectors.append([next(r["score"] for r in rows if r["layer_name"] == name) for name in ordered_names])
        for row in rows:
            layer = by_layer.setdefault(
                row["layer_name"],
                {
                    "family": row["family"],
                    "d_in": row["d_in"],
                    "d_out": row["d_out"],
                    "n_calib_tokens": row["n_calib_tokens"],
                    "tr_M": [],
                    "tr_H": [],
                    "score": [],
                },
            )
            layer["tr_M"].append(float(row["tr_M"]))
            layer["tr_H"].append(float(row["tr_H"]))
            layer["score"].append(float(row["score"]))

    summary_rows = []
    for name in ordered_names:
        layer = by_layer[name]
        tr_M_mean = _mean(layer["tr_M"])
        tr_M_std = _std(layer["tr_M"])
        score_mean = _mean(layer["score"])
        score_std = _std(layer["score"])
        tr_H_mean = _mean(layer["tr_H"])
        tr_H_std = _std(layer["tr_H"])
        summary_rows.append(
            {
                "layer_name": name,
                "short_name": short_layer_name(name),
                "family": layer["family"],
                "d_in": layer["d_in"],
                "d_out": layer["d_out"],
                "n_calib_tokens": layer["n_calib_tokens"],
                "tr_M_mean": tr_M_mean,
                "tr_M_std": tr_M_std,
                "tr_M_cv": _cv(tr_M_mean, tr_M_std),
                "tr_H_mean": tr_H_mean,
                "tr_H_std": tr_H_std,
                "tr_H_cv": _cv(tr_H_mean, tr_H_std),
                "score_mean": score_mean,
                "score_std": score_std,
                "score_cv": _cv(score_mean, score_std),
                "score_ci95_halfwidth": 1.96 * score_std / np.sqrt(max(len(layer["score"]), 1)),
            }
        )

    summary_rows.sort(key=lambda r: -r["score_mean"])

    pairwise_rhos = []
    pairwise_jaccards = []
    for i in range(len(score_vectors)):
        for j in range(i + 1, len(score_vectors)):
            rho = spearman(score_vectors[i], score_vectors[j])
            pairwise_rhos.append(float(rho))
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            pairwise_jaccards.append(inter / union if union else 1.0)

    meta = {
        "top_k": top_k,
        "pairwise_score_spearman_mean": _mean(pairwise_rhos) if pairwise_rhos else float("nan"),
        "pairwise_score_spearman_min": min(pairwise_rhos) if pairwise_rhos else float("nan"),
        "pairwise_topk_jaccard_mean": _mean(pairwise_jaccards) if pairwise_jaccards else float("nan"),
        "pairwise_topk_jaccard_min": min(pairwise_jaccards) if pairwise_jaccards else float("nan"),
        "most_unstable_by_score_cv": [row["layer_name"] for row in sorted(summary_rows, key=lambda r: -r["score_cv"])[:3]],
        "most_stable_by_score_cv": [row["layer_name"] for row in sorted(summary_rows, key=lambda r: r["score_cv"])[:3]],
    }
    return summary_rows, meta


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ensure_libero_runtime()
    print(f"[ASPQ-EST] loading FP policy once ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    layer_names = resolve_target_layers(policy, args)
    print(f"[ASPQ-EST] target layers: {len(layer_names)}")

    print(f"[ASPQ-EST] loading {args.num_samples} LIBERO observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    print(f"[ASPQ-EST] got {len(samples)} samples")

    for p in policy.model.parameters():
        p.requires_grad_(True)
    policy.model.eval()
    patched = unwrap_no_grad_methods(policy.model)

    run_rows: list[list[dict]] = []
    try:
        for rep in range(args.repeats):
            rep_seed = args.seed + rep
            rng = np.random.default_rng(rep_seed)
            accums = {n: LayerAccum() for n in layer_names}
            print(f"[ASPQ-EST] repeat {rep + 1}/{args.repeats} seed={rep_seed}")
            for sample_idx, sample in enumerate(samples, start=1):
                print(f"[ASPQ-EST]   sample {sample_idx}/{len(samples)}")
                collect_one_sample(
                    policy,
                    sample,
                    layer_names,
                    accums,
                    full_M_layers=set(),
                    n_random_dirs=args.random_dirs,
                    token_cap=args.token_cap,
                    rng=rng,
                )
                torch.cuda.empty_cache()
            rows = summarize_once(layer_names, accums)
            for row in rows:
                row["repeat"] = rep
                row["seed"] = rep_seed
            run_rows.append(rows)

        summary_rows, meta = aggregate_runs(run_rows, args.top_k)

        raw_csv = out / "per_run_scores.csv"
        with open(raw_csv, "w", newline="") as f:
            fieldnames = list(run_rows[0][0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rows in run_rows:
                for row in rows:
                    writer.writerow(row)

        summary_csv = out / "estimator_variance.csv"
        with open(summary_csv, "w", newline="") as f:
            fieldnames = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        payload = {
            "checkpoint": args.checkpoint,
            "num_samples": args.num_samples,
            "random_dirs": args.random_dirs,
            "token_cap": args.token_cap,
            "repeats": args.repeats,
            "seed": args.seed,
            "summary": meta,
        }
        with open(out / "estimator_variance_summary.json", "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[ASPQ-EST] wrote {raw_csv}")
        print(f"[ASPQ-EST] wrote {summary_csv}")
        print(f"[ASPQ-EST] wrote {out / 'estimator_variance_summary.json'}")
        print(
            "[ASPQ-EST] pairwise score Spearman "
            f"mean={meta['pairwise_score_spearman_mean']:.4f} "
            f"min={meta['pairwise_score_spearman_min']:.4f}"
        )
        print(
            "[ASPQ-EST] pairwise top-k Jaccard "
            f"mean={meta['pairwise_topk_jaccard_mean']:.4f} "
            f"min={meta['pairwise_topk_jaccard_min']:.4f}"
        )
    finally:
        restore_no_grad_methods(patched)


if __name__ == "__main__":
    main()
