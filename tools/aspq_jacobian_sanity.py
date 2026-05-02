"""ASPQ sanity check.

Goal: verify the Action-Subspace-Projected Quantization motivation on a FP
GR00T policy WITHOUT touching the quant pipeline yet.

For each target Linear layer l we form
    M_l = E_s[ J_l(s)^T J_l(s) ]   # output-side action metric  (d_out x d_out)
    H_l = E_s[ X_l(s)^T X_l(s) ]   # input  Gram                (d_in  x d_in)

where J_l(s) = d a(s) / d h_l(s)  and h_l = X_l W_l^T (post-Linear output).

Cheap stochastic estimator (no need to materialise full J):
    g(s, u) = autograd.grad( <u, a(s)>, h_l(s) )      with u ~ N(0, I_{d_a})
    => E_{s,u}[ g g^T ] = M_l    (unbiased)

We accumulate per-layer:
    tr_M_l, diag(M_l)                       (always; cheap)
    full M_l                                (only for a small set of layers we
                                             want to inspect spectrum on)
    tr_H_l, diag(H_l)                       (always; cheap)

Scoring
-------
Under a layer-uniform quant noise model  E[Δ_l Δ_l^T] = sigma^2 I,
    E[ || J_l Δ_l X_l ||_F^2 ] = sigma^2 * tr(M_l) * tr(H_l) * (1/d_out)
so the per-layer ASPQ score (rank-equivalent to action error contribution
under uniform Δ) is

    score_l := tr(M_l) * tr(H_l) / d_out

We compare the ranking of score_l against the **observed** individual-quant
action RMSE in scenario_summary.csv, and we measure the effective rank of
M_l for a sub-set of layers (= 'how low-dimensional is the action subspace
seen at layer l').

Outputs (under --output-dir):
    aspq_per_layer.csv
    aspq_subset_spectrum.json     (top-K selected layers' eigenvalues)
    plots/score_vs_rmse.png
    plots/top20_score.png
    plots/effective_rank.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.autograd as autograd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from the existing scan script.
from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    DEFAULT_EXCLUDE_REGEX,
    DEFAULT_INCLUDE_REGEX,
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    seed_everything,
)
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE, unsqueeze_dict_values  # noqa: E402
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-source", default="libero", choices=["libero"])
    p.add_argument("--dataset-path", default="/data/ziyu/LIBERO/datasets/lerobot_libero_10")
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10,
                   help="LIBERO observations to use (one per task is typical).")
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task",
                   choices=["sequential", "one_per_task"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--include-regex", default=DEFAULT_INCLUDE_REGEX)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)

    p.add_argument("--random-dirs", type=int, default=8,
                   help="Random Gaussian probes for stochastic Jacobian estimator.")
    p.add_argument("--token-cap", type=int, default=512,
                   help="Cap tokens kept per layer per sample (random subsample).")
    p.add_argument("--full-spectrum-top-k", type=int, default=20,
                   help="How many layers (top by RMSE) to keep full M_l for spectrum.")
    p.add_argument("--full-spectrum-extra", type=int, default=10,
                   help="Random extra layers for spectrum baseline.")

    p.add_argument("--scan-dir",
                   default="/data/ziyu/QuantVLA/results/layerwise_quant_2gpu_taskwise_s10_w3a8_gpu67",
                   help="Existing layerwise scan directory (for RMSE join).")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hooked policy forward (with grad)
# ---------------------------------------------------------------------------

def resolve_target_layers(policy, args) -> list[str]:
    targets = select_targets(
        policy.model,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        scope_prefix=args.scope or None,
        whitelist=None,
        blacklist=None,
    )
    names = [n for n, _ in targets]
    names = names[args.start_layer:]
    if args.max_layers > 0:
        names = names[: args.max_layers]
    return names


def family_of(name: str) -> str:
    if "language_model" in name:
        return "llm"
    if "transformer_blocks" in name:
        return "dit"
    return "other"


@dataclass
class LayerAccum:
    d_in: int = 0
    d_out: int = 0
    n_tokens: int = 0
    tr_M: float = 0.0           # sum of column-wise squared grads
    tr_H: float = 0.0           # sum of column-wise squared activations
    diag_M: torch.Tensor | None = None   # CPU float32 [d_out]
    diag_H: torch.Tensor | None = None   # CPU float32 [d_in]
    full_M: torch.Tensor | None = None   # CPU float64 [d_out, d_out]; None if not selected


def unwrap_no_grad_methods(model) -> list[tuple[Any, str, Any]]:
    """Strip @torch.no_grad decorators from inference paths so autograd can
    flow back from the predicted action. Returns a list of (cls, attr, orig)
    tuples for restoration."""
    from gr00t.model.action_head.flow_matching_action_head import (
        FlowmatchingActionHead,
    )
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import (
        Eagle2_5_VLForConditionalGeneration,
    )
    patched = []
    for cls in (FlowmatchingActionHead, Eagle2_5_VLForConditionalGeneration):
        for attr in ("get_action", "forward", "generate"):
            if not hasattr(cls, attr):
                continue
            fn = getattr(cls, attr)
            wrapped = getattr(fn, "__wrapped__", None)
            if wrapped is None:
                continue
            patched.append((cls, attr, fn))
            setattr(cls, attr, wrapped)
            print(f"[ASPQ] unwrapped no_grad on {cls.__name__}.{attr}")
    return patched


def restore_no_grad_methods(patched):
    for cls, attr, fn in patched:
        setattr(cls, attr, fn)


def normalized_input_no_inference(policy, sample_obs):
    """Mimic policy.get_action's preprocessing without inference_mode."""
    obs_copy = sample_obs.copy()
    obs_copy = unsqueeze_dict_values(obs_copy)
    for k, v in obs_copy.items():
        if not isinstance(v, np.ndarray):
            obs_copy[k] = np.array(v)
    normalized_input = policy.apply_transforms(obs_copy)
    return normalized_input


def collect_one_sample(
    policy,
    sample,
    layer_names: Sequence[str],
    accums: dict[str, LayerAccum],
    full_M_layers: set[str],
    n_random_dirs: int,
    token_cap: int,
    rng: np.random.Generator,
) -> int:
    """Forward+backward one sample, accumulate per-layer M, H. Returns d_a."""
    modules = {n: get_named_module(policy.model, n) for n in layer_names}
    in_cache: dict[str, torch.Tensor] = {}
    out_cache: dict[str, torch.Tensor] = {}

    def make_pre_hook(name):
        def hook(_m, inputs):
            in_cache[name] = inputs[0]
        return hook

    def make_post_hook(name):
        def hook(_m, _inputs, output):
            out_cache[name] = output
        return hook

    handles = []
    for n, m in modules.items():
        handles.append(m.register_forward_pre_hook(make_pre_hook(n)))
        handles.append(m.register_forward_hook(make_post_hook(n)))

    try:
        seed_everything(sample["seed"])
        normalized = normalized_input_no_inference(policy, sample["obs"])

        with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = policy.model.get_action(normalized)
        a_pred = model_pred["action_pred"].float()
        a_flat = a_pred.reshape(-1)
        d_a = int(a_flat.numel())

        # ----- accumulate H (no grad needed) ---------------------------------
        for n in layer_names:
            x = in_cache[n].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            # token subsample (deterministic per sample for repeatability)
            if x.shape[0] > token_cap:
                idx = torch.from_numpy(
                    rng.choice(x.shape[0], size=token_cap, replace=False)
                ).to(x.device)
                x = x.index_select(0, idx)
                in_cache[n] = (in_cache[n].reshape(-1, in_cache[n].shape[-1])
                               .index_select(0, idx))
            ac = accums[n]
            if ac.d_in == 0:
                ac.d_in = int(x.shape[1])
                ac.diag_H = torch.zeros(ac.d_in, dtype=torch.float64)
            ac.n_tokens += int(x.shape[0])
            ac.tr_H += float((x * x).sum().item())
            ac.diag_H += (x * x).sum(dim=0).double().cpu()

        # ----- accumulate M via stochastic Jacobian estimator ---------------
        outs = [out_cache[n] for n in layer_names]
        for r in range(n_random_dirs):
            u = torch.randn_like(a_flat)
            scalar = (u * a_flat).sum()
            grads = autograd.grad(
                scalar, outs,
                retain_graph=(r < n_random_dirs - 1),
                allow_unused=True,
            )
            for n, g in zip(layer_names, grads):
                if g is None:
                    continue
                gflat = g.detach()
                gflat = gflat.reshape(-1, gflat.shape[-1]).float()
                # token subsample to match H bookkeeping
                if gflat.shape[0] > token_cap:
                    # Re-use the same subsample by recomputing from in_cache rows
                    # (already replaced above), but the post-Linear output may
                    # have a different sequence length than input only when
                    # the layer changes shape; for nn.Linear this never happens.
                    idx = torch.from_numpy(
                        rng.choice(gflat.shape[0], size=token_cap, replace=False)
                    ).to(gflat.device)
                    gflat = gflat.index_select(0, idx)

                ac = accums[n]
                if ac.d_out == 0:
                    ac.d_out = int(gflat.shape[1])
                    ac.diag_M = torch.zeros(ac.d_out, dtype=torch.float64)
                    if n in full_M_layers:
                        ac.full_M = torch.zeros(
                            (ac.d_out, ac.d_out), dtype=torch.float64
                        )
                w = 1.0 / n_random_dirs
                ac.tr_M += float((gflat * gflat).sum().item()) * w
                ac.diag_M += (gflat * gflat).sum(dim=0).double().cpu() * w
                if ac.full_M is not None:
                    # outer product accumulation on GPU then move
                    contrib = (gflat.T @ gflat).double().cpu()
                    ac.full_M += contrib * w

    finally:
        for h in handles:
            h.remove()

    return d_a


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_vs_rmse(rows, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys, fams = [], [], []
    for r in rows:
        if r["action_fp_rmse_individual"] is None:
            continue
        xs.append(r["score"]); ys.append(r["action_fp_rmse_individual"])
        fams.append(r["family"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for f, c in [("llm", "#d95f0e"), ("dit", "#2c7fb8")]:
        sx = [x for x, fm in zip(xs, fams) if fm == f]
        sy = [y for y, fm in zip(ys, fams) if fm == f]
        ax.scatter(sx, sy, s=14, color=c, alpha=0.7, label=f)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ASPQ score: tr(M) * tr(H) / d_out")
    ax.set_ylabel("Observed individual W3A8 action_fp_rmse")
    ax.set_title("ASPQ score vs measured per-layer RMSE")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_top20_score(rows, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: -r["score"])[:20]
    names = [r["layer_name"]
             .replace("backbone.eagle_model.language_model.model.layers.", "llm.L")
             .replace("action_head.model.transformer_blocks.", "dit.B")
             for r in rows]
    vals = [r["score"] for r in rows]
    cols = ["#d95f0e" if r["family"] == "llm" else "#2c7fb8" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 8))
    y = list(range(len(names)))[::-1]
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ASPQ score (log)")
    ax.set_title("Top-20 layers by ASPQ score (FP-only diagnostic)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_eff_rank(spectrum: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = list(spectrum.keys())
    eff_ratios = [spectrum[l]["eff_rank"] / spectrum[l]["d_out"] for l in layers]
    fams = [family_of(l) for l in layers]
    fig, ax = plt.subplots(figsize=(8, 5))
    for f, c in [("llm", "#d95f0e"), ("dit", "#2c7fb8")]:
        v = [e for e, fm in zip(eff_ratios, fams) if fm == f]
        ax.hist(v, bins=20, alpha=0.55, color=c, label=f"{f} (n={len(v)})")
    ax.set_xlabel("effective_rank(M_l) / d_out")
    ax.set_ylabel("count of layers")
    ax.set_title("Action-subspace dimension is small relative to layer width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[ASPQ] loading FP policy ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    layer_names = resolve_target_layers(policy, args)
    print(f"[ASPQ] target layers: {len(layer_names)}")

    # Load existing scan RMSE table for join
    rmse_lookup: dict[str, float] = {}
    scan_csv = Path(args.scan_dir) / "scenario_summary.csv"
    if scan_csv.exists():
        for row in csv.DictReader(open(scan_csv)):
            if row["mode"] == "individual" and row["action_fp_rmse"]:
                rmse_lookup[row["focus_layer"]] = float(row["action_fp_rmse"])
        print(f"[ASPQ] joined {len(rmse_lookup)} per-layer individual RMSEs from scan")

    # Pick which layers get full M_l
    full_M_layers: set[str] = set()
    if rmse_lookup:
        ranked = sorted(rmse_lookup.items(), key=lambda kv: -kv[1])
        top = [n for n, _ in ranked[: args.full_spectrum_top_k]]
        rest = [n for n in layer_names if n not in set(top)]
        extras = list(rng.choice(rest, size=min(args.full_spectrum_extra, len(rest)),
                                 replace=False)) if rest else []
        full_M_layers = set(top) | set(extras)
    print(f"[ASPQ] storing full M for {len(full_M_layers)} layers")

    # Load samples
    print(f"[ASPQ] loading {args.num_samples} LIBERO observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    print(f"[ASPQ] got {len(samples)} samples")

    # Make sure model parameters require_grad so the autograd graph forms
    for p in policy.model.parameters():
        p.requires_grad_(False)
    # We just need outputs to require grad. Activate grad on the embedding /
    # input only. Easier: enable grad on all params; we'll never call .grad on them.
    for p in policy.model.parameters():
        p.requires_grad_(True)
    policy.model.eval()  # disables dropout etc; eval is fine with grad enabled
    patched = unwrap_no_grad_methods(policy.model)

    accums: dict[str, LayerAccum] = {n: LayerAccum() for n in layer_names}

    d_a_seen = None
    for i, s in enumerate(samples):
        print(f"[ASPQ] sample {i+1}/{len(samples)}")
        d_a = collect_one_sample(
            policy, s, layer_names, accums, full_M_layers,
            args.random_dirs, args.token_cap, rng,
        )
        if d_a_seen is None:
            print(f"[ASPQ] action dim d_a = {d_a}")
            d_a_seen = d_a
        torch.cuda.empty_cache()

    # ----- summarize ---------------------------------------------------------
    rows = []
    spectrum: dict[str, dict] = {}
    for n in layer_names:
        ac = accums[n]
        if ac.n_tokens == 0:
            continue
        tr_M_norm = ac.tr_M / max(ac.n_tokens, 1)
        tr_H_norm = ac.tr_H / max(ac.n_tokens, 1)
        score = tr_M_norm * tr_H_norm / max(ac.d_out, 1)
        eff_rank = None; eff_rank_ratio = None
        if ac.full_M is not None:
            M = ac.full_M / max(ac.n_tokens, 1)
            try:
                eigs = torch.linalg.eigvalsh(M).numpy()
            except Exception:
                eigs = np.linalg.eigvalsh(M.numpy())
            eigs = np.clip(eigs, 0.0, None)
            tr1 = float(eigs.sum())
            tr2 = float((eigs ** 2).sum())
            eff_rank = (tr1 ** 2) / max(tr2, 1e-30)
            eff_rank_ratio = eff_rank / ac.d_out
            spectrum[n] = {
                "d_out": ac.d_out,
                "eff_rank": eff_rank,
                "eff_rank_ratio": eff_rank_ratio,
                "top10_eigs": [float(x) for x in eigs[::-1][:10]],
                "tr_M": tr1,
            }
        rows.append({
            "layer_name": n,
            "family": family_of(n),
            "d_in": ac.d_in,
            "d_out": ac.d_out,
            "n_calib_tokens": ac.n_tokens,
            "tr_M": tr_M_norm,
            "tr_H": tr_H_norm,
            "score": score,
            "eff_rank": eff_rank,
            "eff_rank_ratio": eff_rank_ratio,
            "action_fp_rmse_individual": rmse_lookup.get(n),
        })

    # CSV
    csv_path = out / "aspq_per_layer.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ASPQ] wrote {csv_path}")

    with open(out / "aspq_subset_spectrum.json", "w") as f:
        json.dump(spectrum, f, indent=2)

    # Spearman correlation
    pairs = [(r["score"], r["action_fp_rmse_individual"])
             for r in rows if r["action_fp_rmse_individual"] is not None]
    if pairs:
        sx = np.array([p[0] for p in pairs])
        sy = np.array([p[1] for p in pairs])
        rx = np.argsort(np.argsort(sx))
        ry = np.argsort(np.argsort(sy))
        rho = np.corrcoef(rx, ry)[0, 1]
        print(f"[ASPQ] Spearman(score, observed RMSE) = {rho:.3f}  over {len(pairs)} layers")

    plot_score_vs_rmse(rows, out / "plots" / "score_vs_rmse.png")
    plot_top20_score(rows, out / "plots" / "top20_score.png")
    if spectrum:
        plot_eff_rank(spectrum, out / "plots" / "effective_rank.png")
    restore_no_grad_methods(patched)
    print(f"[ASPQ] done. results in {out}")


if __name__ == "__main__":
    main()
