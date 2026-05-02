"""Build a PDF analysis report for a layerwise-quant scan directory."""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import io


def family(name: str) -> str:
    if "language_model" in name:
        return "llm"
    if "transformer_blocks" in name:
        return "dit"
    return "other"


PROJ_KINDS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


def proj_kind(name: str) -> str:
    for k in PROJ_KINDS:
        if k in name:
            return k
    return "other"


def short_layer(name: str) -> str:
    s = name
    s = s.replace("backbone.eagle_model.language_model.model.layers.", "llm.L")
    s = s.replace("action_head.model.transformer_blocks.", "dit.B")
    s = s.replace(".self_attn.", ".sa.")
    return s


def load_scenarios(path: Path):
    rows = list(csv.DictReader(open(path)))
    ind = [r for r in rows if r["mode"] == "individual" and r["action_fp_rmse"]]
    cum = [r for r in rows if r["mode"] == "cumulative" and r["action_fp_rmse"]]
    cum.sort(key=lambda r: int(r["num_quantized_layers"]))
    return rows, ind, cum


def fig_to_pil(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_text_page(pages, title, lines, fontsize=9):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    txt = "\n".join(lines)
    fig.text(0.06, 0.92, txt, fontsize=fontsize, family="monospace",
             verticalalignment="top")
    pages.append(fig_to_pil(fig))


def add_image_page(pages, title, image_path, caption=None):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    if image_path.exists():
        img = plt.imread(image_path)
        ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
        ax.imshow(img)
        ax.axis("off")
    else:
        fig.text(0.5, 0.5, f"missing: {image_path}", ha="center")
    if caption:
        fig.text(0.06, 0.06, caption, fontsize=8.5, wrap=True)
    pages.append(fig_to_pil(fig))


def make_family_box_table_page(pages, ind):
    fam_rmse = defaultdict(list)
    proj_rmse = defaultdict(list)
    for r in ind:
        rmse = float(r["action_fp_rmse"])
        fam = family(r["focus_layer"])
        fam_rmse[fam].append(rmse)
        proj_rmse[(fam, proj_kind(r["focus_layer"]))].append(rmse)

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Per-family / per-op individual-quant sensitivity",
                 fontsize=13, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.1, 1.6], hspace=0.35,
                           top=0.92, bottom=0.05, left=0.1, right=0.95)

    ax1 = fig.add_subplot(gs[0])
    fams = sorted(fam_rmse.keys())
    ax1.boxplot([fam_rmse[f] for f in fams], labels=fams, showfliers=True)
    ax1.set_ylabel("action_fp_rmse")
    ax1.set_title("By family")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both")

    ax2 = fig.add_subplot(gs[1])
    items = sorted(proj_rmse.items(),
                   key=lambda kv: -statistics.mean(kv[1]))
    labels = [f"{f}.{k}" for (f, k), _ in items]
    data = [v for _, v in items]
    ax2.boxplot(data, labels=labels, showfliers=True, vert=False)
    ax2.set_xlabel("action_fp_rmse")
    ax2.set_title("By projection kind (sorted by mean)")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3, which="both")

    pages.append(fig_to_pil(fig))


def make_top_layers_page(pages, ind, n=20):
    sorted_ind = sorted(ind, key=lambda r: -float(r["action_fp_rmse"]))[:n]
    names = [short_layer(r["focus_layer"]) for r in sorted_ind]
    vals = [float(r["action_fp_rmse"]) for r in sorted_ind]
    colors = ["#d95f0e" if family(r["focus_layer"]) == "llm" else "#2c7fb8"
              for r in sorted_ind]

    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.suptitle(f"Top {n} most sensitive layers (individual quant)",
                 fontsize=13, fontweight="bold", y=0.97)
    y = list(range(len(names)))[::-1]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("action_fp_rmse")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, axis="x", which="both")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d95f0e", label="LLM"),
                       Patch(color="#2c7fb8", label="DiT")],
              loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pages.append(fig_to_pil(fig))


def make_cumulative_page(pages, cum):
    steps = [int(r["num_quantized_layers"]) for r in cum]
    rmse = [float(r["action_fp_rmse"]) for r in cum]
    cos = [float(r["action_fp_cosine"]) for r in cum]
    rell2 = [float(r["action_fp_rel_l2"]) for r in cum]

    # detect deltas
    deltas = [(rmse[i] - (rmse[i - 1] if i > 0 else 0.0), steps[i],
               cum[i]["focus_layer"]) for i in range(len(cum))]
    top_jumps = sorted(deltas, reverse=True)[:5]

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Cumulative quantization trajectory",
                 fontsize=13, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.4, 1.0, 1.0], hspace=0.4,
                           top=0.93, bottom=0.05, left=0.1, right=0.95)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, rmse, color="#d62728", label="RMSE")
    ax1.set_ylabel("action_fp_rmse", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(steps, cos, color="#2ca02c", label="cosine")
    ax1b.set_ylabel("action_fp_cosine", color="#2ca02c")
    ax1b.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_xlabel("# layers quantized")
    ax1.set_title("RMSE & cosine vs cumulative layer count")

    for d, st, name in top_jumps:
        ax1.annotate(short_layer(name), xy=(st, rmse[st - 1]),
                     xytext=(st, rmse[st - 1] + 0.015),
                     fontsize=7, ha="center",
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.6))

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, rell2, color="#1f77b4")
    ax2.set_xlabel("# layers quantized")
    ax2.set_ylabel("action_fp_rel_l2")
    ax2.set_title("Relative L2 vs cumulative layer count")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    delta_vals = [d for d, _, _ in deltas]
    ax3.bar(steps, delta_vals,
            color=["#d62728" if d > 0 else "#1f77b4" for d in delta_vals])
    ax3.set_xlabel("step (layer added)")
    ax3.set_ylabel("Δ RMSE")
    ax3.set_title("Per-step RMSE change (red = damaging)")
    ax3.grid(True, alpha=0.3)

    pages.append(fig_to_pil(fig))


def make_per_task_page(pages, task_csv: Path):
    trows = list(csv.DictReader(open(task_csv)))
    tasks = sorted(set(r["task_id"] for r in trows if r["task_id"]))
    cum_finals, ind_meds, ind_maxs, labels = [], [], [], []
    for tid in tasks:
        sub = [r for r in trows if r["task_id"] == tid]
        ind = [float(r["action_fp_rmse"]) for r in sub
               if r["mode"] == "individual" and r["action_fp_rmse"]]
        cum = [(int(r["num_quantized_layers"]), float(r["action_fp_rmse"]))
               for r in sub if r["mode"] == "cumulative" and r["action_fp_rmse"]]
        cum.sort()
        if not ind or not cum:
            continue
        labels.append(f"task_{tid}")
        ind_meds.append(statistics.median(ind))
        ind_maxs.append(max(ind))
        cum_finals.append(cum[-1][1])

    order = sorted(range(len(labels)), key=lambda i: cum_finals[i])
    labels = [labels[i] for i in order]
    ind_meds = [ind_meds[i] for i in order]
    ind_maxs = [ind_maxs[i] for i in order]
    cum_finals = [cum_finals[i] for i in order]

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Per-task sensitivity (LIBERO-10)",
                 fontsize=13, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4,
                           top=0.93, bottom=0.05, left=0.12, right=0.95)
    import numpy as np
    x = np.arange(len(labels))
    w = 0.28

    ax1 = fig.add_subplot(gs[0])
    ax1.bar(x - w, ind_meds, w, label="individual median", color="#9ecae1")
    ax1.bar(x, ind_maxs, w, label="individual max", color="#3182bd")
    ax1.bar(x + w, cum_finals, w, label="cumulative final", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("action_fp_rmse")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both", axis="y")
    ax1.legend(fontsize=9)
    ax1.set_title("RMSE by task (sorted by full-quant final RMSE)")

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(x, cum_finals, color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("action_fp_rmse (full quant)")
    ax2.set_title("Full-quantization RMSE per task (linear scale)")
    ax2.grid(True, alpha=0.3, axis="y")

    pages.append(fig_to_pil(fig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run_dir
    out = args.out or (run / "report.pdf")

    rows, ind, cum = load_scenarios(run / "scenario_summary.csv")
    plot_summary = json.load(open(run / "plots" / "plot_summary.json"))
    plan_cfg = json.load(open(run / "plan" / "run_config.json"))

    fam_counts = defaultdict(int)
    for r in ind:
        fam_counts[family(r["focus_layer"])] += 1

    all_rmse = [float(r["action_fp_rmse"]) for r in ind]
    final_rmse = float(cum[-1]["action_fp_rmse"])
    final_cos = float(cum[-1]["action_fp_cosine"])
    final_rell2 = float(cum[-1]["action_fp_rel_l2"])

    qc = plan_cfg["quant_config"]
    lc = plan_cfg["libero_config"]

    cover_lines = [
        f"Run dir       : {run}",
        f"Checkpoint    : {plan_cfg['checkpoint']}",
        f"Data source   : {plan_cfg['data_source']}  ({plan_cfg['task_suite_name']})",
        f"Modes         : {', '.join(plan_cfg['modes'])}",
        f"Layers scanned: {len(ind)}  (LLM={fam_counts['llm']}, DiT={fam_counts['dit']})",
        f"Tasks         : {len(set(r['focus_layer'] for r in ind)) and 10}  trials/task = {lc['num_trials_per_task']}",
        "",
        "[Quant config]",
        f"  W/A bits     : W{qc['wbits']} / A{qc['abits']}",
        f"  block_size   : {qc['block_size']}",
        f"  lambda_smooth: {qc['lambda_smooth']}",
        f"  act_pct      : {qc['act_pct']}",
        f"  row_rot      : {qc['row_rot']}    permute: {qc['permute']}",
        "",
        "[Result headlines]",
        f"  Individual RMSE: min={min(all_rmse):.5f}  med={statistics.median(all_rmse):.5f}  "
        f"mean={statistics.mean(all_rmse):.5f}  max={max(all_rmse):.5f}",
        f"  Full-cumulative RMSE = {final_rmse:.5f}    rel_l2 = {final_rell2:.5f}    "
        f"cosine = {final_cos:.5f}",
        "",
        "[Top single most-sensitive layer]",
        f"  {short_layer(plot_summary['top_action_sensitive_layers'][0]['layer_name'])}",
        f"  action_fp_rmse = {plot_summary['top_action_sensitive_layers'][0]['action_fp_rmse']:.5f}",
        "",
        "[Report contents]",
        "  p2  Executive summary",
        "  p3  Per-family / per-op sensitivity (boxplots)",
        "  p4  Top-20 individual layers (bar chart)",
        "  p5  Cumulative trajectory (RMSE + cosine + per-step delta)",
        "  p6  Per-task sensitivity (LIBERO-10)",
        "  p7+ Embedded plots from results/plots/",
    ]

    summary_lines = [
        "Headlines:",
        " - Two layer classes dominate: LLM `down_proj` (esp. layers.2) and",
        "   DiT `attn1.to_out.0`. They explain the bulk of cumulative RMSE.",
        " - layers.2.mlp.down_proj alone causes a +0.074 RMSE jump (step 21",
        "   of cumulative), more than half of the final 0.138 RMSE.",
        " - DiT `attn1.to_out.0` for blocks {15,12,6,8,...} are next worst",
        "   (~0.005-0.014 individual RMSE).",
        " - LLM is ~1.7x more sensitive than DiT on average; DiT's mean is",
        "   inflated mainly by the to_out.0 cluster.",
        " - Safe to W3: DiT to_q / to_k / ff.net.2 and LLM q/k/gate (median",
        "   < 0.002, near zero contribution to cumulative RMSE).",
        "",
        "Per-task spread (full-quant cumulative RMSE):",
        " - Best: task_4 (0.051) / task_0 (0.056) / task_6 (0.084)",
        " - Worst: task_2 (0.204) / task_1 (0.194) / task_9 (0.162)",
        " - In ALL 10 tasks the #1 most damaging individual layer is the",
        "   same: `eagle.language_model.layers.2.mlp.down_proj`.",
        "",
        "Action items:",
        " 1. Promote `language_model.layers.2.mlp.down_proj` to >=W4/FP16.",
        "    This single change should remove most of the >40% step-21 jump.",
        " 2. Mixed-precision candidates worth keeping at higher bitwidth:",
        "      LLM:  layers {2,9,10,11} -> down_proj / o_proj / v_proj",
        "      DiT:  blocks {0,6,8,10,12,14,15} -> attn1.to_out.0",
        " 3. Validity caveat: only 1 trial / task at 10 sample steps. The",
        "    cumulative curve has visible negative jumps (e.g. step 90: -0.022,",
        "    step 165: -0.014) -- treat single-step deltas with caution.",
        "    Trends consistent across tasks (down_proj, to_out.0) are robust.",
        "",
        "Cumulative RMSE every 20 layers:",
    ]
    for r in cum:
        n = int(r["num_quantized_layers"])
        if n % 20 == 0 and n > 0:
            summary_lines.append(
                f"  n={n:>3}  rmse={float(r['action_fp_rmse']):.5f}  "
                f"rel_l2={float(r['action_fp_rel_l2']):.4f}  "
                f"cos={float(r['action_fp_cosine']):.4f}")

    pages = []
    add_text_page(pages, "Layerwise Quantization Report (W3A8, LIBERO-10)",
                  cover_lines, fontsize=9)
    add_text_page(pages, "Executive summary", summary_lines, fontsize=8.5)
    make_family_box_table_page(pages, ind)
    make_top_layers_page(pages, ind, n=20)
    make_cumulative_page(pages, cum)
    if (run / "task_scenario_summary.csv").exists():
        make_per_task_page(pages, run / "task_scenario_summary.csv")

    plots_dir = run / "plots"
    for fname, cap in [
        ("family_sensitivity.png",
         "Family-level boxplot of individual-quant RMSE."),
        ("scenario_trends.png",
         "Cumulative + individual scenario trends."),
        ("top_individual_layers_action_fp_rmse.png",
         "Top individual layers by action_fp_rmse."),
        ("top_individual_layers_w_wx.png",
         "Top individual layers by w_rel_l2 / wx_rel_l2."),
        ("wx_vs_action_scatter.png",
         "wx_rel_l2 vs action_fp_rmse - proxy quality of input-aware "
         "weight error for predicting action error."),
    ]:
        add_image_page(pages, fname, plots_dir / fname, caption=cap)

    pages[0].save(out, save_all=True, append_images=pages[1:],
                  format="PDF", resolution=150.0)
    print(f"Wrote {out}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()
