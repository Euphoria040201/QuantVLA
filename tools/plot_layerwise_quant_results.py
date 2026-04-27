#!/usr/bin/env python
"""
Generate summary plots for layer-wise quantization drift analysis results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": ":",
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

COLORS = {
    "llm": "#0f766e",
    "dit": "#b45309",
    "other": "#475569",
    "individual": "#1d4ed8",
    "cumulative": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot layer-wise quant analysis results.")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing scenario_summary.json and per_layer_metrics.jsonl.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k layers to show in ranking plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def scenario_index_map(scenario_rows: list[dict[str, Any]]) -> dict[str, int]:
    non_baseline = [row for row in scenario_rows if row.get("mode") != "baseline"]
    return {row["scenario"]: idx for idx, row in enumerate(non_baseline)}


def add_mode_trend_panel(ax, rows: list[dict[str, Any]], mode: str, metric_key: str, ylabel: str) -> None:
    mode_rows = [row for row in rows if row.get("mode") == mode]
    mode_rows.sort(key=lambda row: row["scenario"])
    if not mode_rows:
        ax.set_visible(False)
        return

    x = np.arange(len(mode_rows))
    y = np.array([float(row.get(metric_key, np.nan)) for row in mode_rows], dtype=float)
    ax.plot(x, y, color=COLORS.get(mode, "#334155"), linewidth=2)
    ax.scatter(x, y, color=COLORS.get(mode, "#334155"), s=18)
    ax.set_title(mode.capitalize())
    ax.set_xlabel("Scenario Index")
    ax.set_ylabel(ylabel)


def plot_scenario_trends(out_dir: Path, scenario_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    add_mode_trend_panel(axes[0, 0], scenario_rows, "individual", "action_fp_rmse", "Action FP RMSE")
    add_mode_trend_panel(axes[0, 1], scenario_rows, "cumulative", "action_fp_rmse", "Action FP RMSE")
    add_mode_trend_panel(
        axes[1, 0], scenario_rows, "individual", "delta_gt_mse_vs_fp", "Delta GT MSE vs FP"
    )
    add_mode_trend_panel(
        axes[1, 1], scenario_rows, "cumulative", "delta_gt_mse_vs_fp", "Delta GT MSE vs FP"
    )
    fig.suptitle("Quantization Impact by Scenario", fontsize=12)
    save_fig(fig, out_dir, "scenario_trends")


def plot_top_layer_bars(out_dir: Path, individual_focus_rows: list[dict[str, Any]], top_k: int) -> None:
    if not individual_focus_rows:
        return

    sorted_rows = sorted(
        individual_focus_rows,
        key=lambda row: float(row.get("action_fp_rmse", 0.0)),
        reverse=True,
    )[:top_k]

    labels = [row["layer_name"].split("language_model.")[-1].replace("action_head.model.", "dit.") for row in sorted_rows]
    values = [float(row.get("action_fp_rmse", 0.0)) for row in sorted_rows]
    colors = [COLORS.get(row.get("family", "other"), COLORS["other"]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(sorted_rows) + 1)))
    y = np.arange(len(sorted_rows))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Action FP RMSE")
    ax.set_title(f"Top {len(sorted_rows)} Most Sensitive Individual Layers")
    save_fig(fig, out_dir, "top_individual_layers_action_fp_rmse")


def plot_weight_wx_bars(out_dir: Path, individual_focus_rows: list[dict[str, Any]], top_k: int) -> None:
    if not individual_focus_rows:
        return

    rows = sorted(
        individual_focus_rows,
        key=lambda row: float(row.get("wx_rel_l2", 0.0)),
        reverse=True,
    )[:top_k]

    labels = [row["layer_name"].split("language_model.")[-1].replace("action_head.model.", "dit.") for row in rows]
    wx_vals = np.array([float(row.get("wx_rel_l2", 0.0)) for row in rows], dtype=float)
    w_vals = np.array([float(row.get("w_rel_l2", 0.0)) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(rows) + 1)))
    y = np.arange(len(rows))
    ax.barh(y - 0.18, wx_vals, height=0.35, color="#2563eb", label="WX rel L2")
    ax.barh(y + 0.18, w_vals, height=0.35, color="#ea580c", label="W rel L2")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative L2 Drift")
    ax.set_title(f"Top {len(rows)} Individual Layers by WX Drift")
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "top_individual_layers_w_wx")


def plot_wx_vs_action_scatter(out_dir: Path, individual_focus_rows: list[dict[str, Any]]) -> None:
    if not individual_focus_rows:
        return

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in individual_focus_rows:
        grouped[row.get("family", "other")].append(row)

    for family, rows in grouped.items():
        x = [float(row.get("wx_rel_l2", 0.0)) for row in rows]
        y = [float(row.get("action_fp_rmse", 0.0)) for row in rows]
        ax.scatter(x, y, s=28, alpha=0.8, label=family.upper(), color=COLORS.get(family, COLORS["other"]))

    ax.set_xlabel("Per-layer WX rel L2")
    ax.set_ylabel("Scenario Action FP RMSE")
    ax.set_title("Output Drift vs End-to-End Action Drift")
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "wx_vs_action_scatter")


def summarize_tables(
    out_dir: Path,
    scenario_rows: list[dict[str, Any]],
    individual_focus_rows: list[dict[str, Any]],
    top_k: int,
) -> None:
    baseline = next((row for row in scenario_rows if row.get("mode") == "baseline"), None)
    top_action = sorted(
        individual_focus_rows,
        key=lambda row: float(row.get("action_fp_rmse", 0.0)),
        reverse=True,
    )[:top_k]
    top_wx = sorted(
        individual_focus_rows,
        key=lambda row: float(row.get("wx_rel_l2", 0.0)),
        reverse=True,
    )[:top_k]

    summary = {
        "baseline": baseline,
        "top_action_sensitive_layers": [
            {
                "rank": idx + 1,
                "layer_name": row["layer_name"],
                "family": row.get("family", "other"),
                "action_fp_rmse": row.get("action_fp_rmse"),
                "delta_gt_mse_vs_fp": row.get("delta_gt_mse_vs_fp"),
                "wx_rel_l2": row.get("wx_rel_l2"),
                "w_rel_l2": row.get("w_rel_l2"),
            }
            for idx, row in enumerate(top_action)
        ],
        "top_wx_layers": [
            {
                "rank": idx + 1,
                "layer_name": row["layer_name"],
                "family": row.get("family", "other"),
                "wx_rel_l2": row.get("wx_rel_l2"),
                "action_fp_rmse": row.get("action_fp_rmse"),
                "w_rel_l2": row.get("w_rel_l2"),
            }
            for idx, row in enumerate(top_wx)
        ],
    }
    with open(out_dir / "plot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "plot_summary.md", "w") as f:
        f.write("# Layerwise Quant Plot Summary\n\n")
        if baseline:
            f.write("## Baseline\n\n")
            f.write(
                f"- FP GT MSE: {baseline.get('fp_gt_mse', 'n/a')}\n"
                f"- FP GT RMSE: {baseline.get('fp_gt_rmse', 'n/a')}\n\n"
            )
        f.write("## Top Action-Sensitive Layers\n\n")
        for idx, row in enumerate(top_action, start=1):
            f.write(
                f"{idx}. `{row['layer_name']}` "
                f"(family={row.get('family')}, action_fp_rmse={float(row.get('action_fp_rmse', 0.0)):.6f}, "
                f"wx_rel_l2={float(row.get('wx_rel_l2', 0.0)):.6f}, "
                f"w_rel_l2={float(row.get('w_rel_l2', 0.0)):.6f})\n"
            )


def attach_action_metrics(
    scenario_rows: list[dict[str, Any]],
    per_layer_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_scenario = {row["scenario"]: row for row in scenario_rows}
    merged = []
    for row in per_layer_rows:
        combined = dict(row)
        scenario = by_scenario.get(row["scenario"])
        if scenario:
            for key in ("action_fp_rmse", "action_fp_rel_l2", "delta_gt_mse_vs_fp", "quant_gt_rmse"):
                if key in scenario:
                    combined[key] = scenario[key]
        merged.append(combined)
    return merged


def plot_family_breakdown(out_dir: Path, individual_focus_rows: list[dict[str, Any]]) -> None:
    if not individual_focus_rows:
        return

    family_to_vals: dict[str, list[float]] = defaultdict(list)
    for row in individual_focus_rows:
        family_to_vals[row.get("family", "other")].append(float(row.get("action_fp_rmse", 0.0)))

    families = sorted(family_to_vals.keys())
    means = [float(np.mean(family_to_vals[family])) for family in families]
    stds = [float(np.std(family_to_vals[family])) for family in families]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = np.arange(len(families))
    ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS.get(f, COLORS["other"]) for f in families])
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in families])
    ax.set_ylabel("Mean Action FP RMSE")
    ax.set_title("Family-Level Sensitivity")
    save_fig(fig, out_dir, "family_sensitivity")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    scenario_rows = load_json(results_dir / "scenario_summary.json")
    per_layer_rows = load_jsonl(results_dir / "per_layer_metrics.jsonl")
    merged_rows = attach_action_metrics(scenario_rows, per_layer_rows)

    scenario_map = scenario_index_map(scenario_rows)
    for row in merged_rows:
        row["scenario_index"] = scenario_map.get(row["scenario"], -1)

    individual_focus_rows = [
        row for row in merged_rows if row.get("mode") == "individual" and row.get("focus_layer")
    ]
    individual_focus_rows.sort(key=lambda row: row.get("scenario_index", 10**9))

    plot_scenario_trends(plots_dir, scenario_rows)
    plot_top_layer_bars(plots_dir, individual_focus_rows, args.top_k)
    plot_weight_wx_bars(plots_dir, individual_focus_rows, args.top_k)
    plot_wx_vs_action_scatter(plots_dir, individual_focus_rows)
    plot_family_breakdown(plots_dir, individual_focus_rows)
    summarize_tables(plots_dir, scenario_rows, individual_focus_rows, args.top_k)

    manifest = {
        "plots_dir": str(plots_dir),
        "generated_files": sorted(path.name for path in plots_dir.iterdir() if path.is_file()),
    }
    with open(plots_dir / "plot_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Plot] Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
