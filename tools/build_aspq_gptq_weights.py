#!/usr/bin/env python
"""Build offline ASPQ-GPTQ quantized weights for GR00T.

This script keeps the existing DuQuant-based ASPQ flow untouched. It builds a
parallel artifact containing dequantized GPTQ weights per layer, optionally
refined inside the ASPQ action subspace.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    DEFAULT_EXCLUDE_REGEX,
    DEFAULT_INCLUDE_REGEX,
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    seed_everything,
)
from tools.aspq_jacobian_sanity import normalized_input_no_inference  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402
from gr00t.quantization.aspq_gptq import (  # noqa: E402
    load_aspq_basis_for_layer,
    solve_aspq_gptq_weight,
)
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402

DEFAULT_LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/work/mingze/LIBERO")
DEFAULT_LIBERO_DATASET = f"{DEFAULT_LIBERO_ROOT}/datasets/lerobot_libero_10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metric-path", required=True, help="ASPQ metric .pt from collect_aspq_jacobian.py")
    p.add_argument("--output-path", required=True, help="Where to save the ASPQ-GPTQ quantized weights (.pt)")
    p.add_argument("--weight-bits", type=int, default=3)
    p.add_argument("--metric-top-k", type=int, default=0, help="Further truncate the loaded ASPQ basis; 0 keeps all stored vectors.")
    p.add_argument("--min-eig", type=float, default=1e-12)
    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--missing-metric", choices=["error", "fallback"], default="fallback")
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")

    p.add_argument("--data-source", default="libero", choices=["libero"])
    p.add_argument("--dataset-path", default=DEFAULT_LIBERO_DATASET)
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task", choices=["sequential", "one_per_task"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--include-regex", default=DEFAULT_INCLUDE_REGEX)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--cache-layer-inputs", action="store_true",
                   help="Capture sampled layer inputs for all target layers in one pass and reuse them during GPTQ build.")
    return p.parse_args()


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


def _clear_quant_env() -> None:
    for prefix in ("GR00T_DUQUANT_", "GR00T_ASPQ_GPTQ"):
        for key in list(os.environ.keys()):
            if key == prefix or key.startswith(prefix):
                os.environ.pop(key, None)


def _autocast_context(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE)
    return nullcontext()


def collect_layer_gram(
    policy,
    normalized_samples,
    layer_name: str,
    token_cap: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, int]:
    module = get_named_module(policy.model, layer_name)
    gram: Optional[torch.Tensor] = None
    n_tokens = 0
    gram_device = module.weight.device if hasattr(module, "weight") else torch.device(device)

    def hook(_m, inputs):
        nonlocal gram, n_tokens
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        if token_cap > 0 and x.shape[0] > token_cap:
            idx = torch.from_numpy(rng.choice(x.shape[0], size=token_cap, replace=False)).to(x.device)
            x = x.index_select(0, idx)
        if gram is None:
            gram = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32, device=gram_device)
        x = x.to(device=gram_device, dtype=torch.float32, non_blocking=True)
        gram += x.t() @ x
        n_tokens += int(x.shape[0])

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(), _autocast_context(device):
            for sample in normalized_samples:
                seed_everything(sample["seed"])
                policy.model.get_action(sample["normalized"])
    finally:
        handle.remove()

    if gram is None or n_tokens == 0:
        raise RuntimeError(f"Failed to collect activations for layer '{layer_name}'")
    return gram / float(n_tokens), n_tokens


def collect_cached_layer_inputs(
    policy,
    normalized_samples,
    layer_names: list[str],
    token_cap: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    modules = {name: get_named_module(policy.model, name) for name in layer_names}
    cache_devices = {
        name: (module.weight.device if hasattr(module, "weight") else torch.device(device))
        for name, module in modules.items()
    }
    cached: dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}
    n_tokens: dict[str, int] = {name: 0 for name in layer_names}

    def make_hook(name: str):
        def hook(_m, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            if token_cap > 0 and x.shape[0] > token_cap:
                idx = torch.from_numpy(rng.choice(x.shape[0], size=token_cap, replace=False)).to(x.device)
                x = x.index_select(0, idx)
            x = x.to(device=cache_devices[name], dtype=torch.float16, non_blocking=True)
            cached[name].append(x.contiguous())
            n_tokens[name] += int(x.shape[0])
        return hook

    handles = [module.register_forward_pre_hook(make_hook(name)) for name, module in modules.items()]
    try:
        with torch.no_grad(), _autocast_context(device):
            for sample in normalized_samples:
                seed_everything(sample["seed"])
                policy.model.get_action(sample["normalized"])
    finally:
        for handle in handles:
            handle.remove()

    merged: dict[str, torch.Tensor] = {}
    for name in layer_names:
        if not cached[name] or n_tokens[name] == 0:
            raise RuntimeError(f"Failed to cache activations for layer '{name}'")
        merged[name] = torch.cat(cached[name], dim=0)
    return merged, n_tokens


def main() -> None:
    args = parse_args()
    ensure_libero_runtime()
    _clear_quant_env()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"[ASPQ-GPTQ] loading FP policy ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()

    layer_names = resolve_target_layers(policy, args)
    print(f"[ASPQ-GPTQ] target layers: {len(layer_names)}")

    print(f"[ASPQ-GPTQ] loading {args.num_samples} LIBERO observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(
            {
                "seed": sample["seed"],
                "normalized": normalized_input_no_inference(policy, sample["obs"]),
            }
        )
    print(f"[ASPQ-GPTQ] prepared {len(normalized_samples)} normalized observations")

    cached_inputs = None
    cached_tokens = None
    if args.cache_layer_inputs:
        print("[ASPQ-GPTQ] caching sampled inputs for all target layers in one forward sweep")
        cached_inputs, cached_tokens = collect_cached_layer_inputs(
            policy,
            normalized_samples,
            layer_names,
            args.token_cap,
            rng,
            str(args.device),
        )
        total_mb = sum(t.numel() * t.element_size() for t in cached_inputs.values()) / (1024 ** 2)
        print(f"[ASPQ-GPTQ] cached {len(cached_inputs)} layers of sampled inputs ({total_mb:.1f} MiB on-device)")

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    records: dict[str, dict] = {}

    for idx, layer_name in enumerate(layer_names, start=1):
        module = get_named_module(policy.model, layer_name)
        if not isinstance(module, torch.nn.Linear):
            continue
        print(f"[ASPQ-GPTQ] layer {idx}/{len(layer_names)}: {layer_name}")
        solve_device = module.weight.device
        if cached_inputs is not None and cached_tokens is not None:
            x_cached = cached_inputs[layer_name].to(device=solve_device, dtype=torch.float32, non_blocking=True)
            n_tokens = int(cached_tokens[layer_name])
            H = (x_cached.t() @ x_cached) / float(n_tokens)
        else:
            H, n_tokens = collect_layer_gram(
                policy,
                normalized_samples,
                layer_name,
                args.token_cap,
                rng,
                str(args.device),
            )
        basis = load_aspq_basis_for_layer(
            layer_name,
            args.metric_path,
            out_features=module.out_features,
            topk=args.metric_top_k,
            min_eig=args.min_eig,
        )
        if basis is None and args.missing_metric == "error":
            raise FileNotFoundError(f"No usable ASPQ metric found for layer '{layer_name}' in {args.metric_path}")
        U = basis[0] if basis is not None else None
        eigvals = basis[1] if basis is not None else None
        W = module.weight.detach().to(dtype=torch.float32, device=solve_device)
        W_q = solve_aspq_gptq_weight(
            W,
            H.to(dtype=torch.float32, device=solve_device),
            bits=args.weight_bits,
            U=U.to(dtype=torch.float32, device=solve_device) if U is not None else None,
            eigvals=eigvals.to(dtype=torch.float32, device=solve_device) if eigvals is not None else None,
            block_size=args.gptq_block_size,
            damp_percent=args.gptq_damp_percent,
            min_eig=args.min_eig,
        )
        aspq_rank = int(U.shape[1]) if U is not None else 0
        records[layer_name] = {
            "weight_q": W_q.to(dtype=save_dtype).cpu().contiguous(),
            "weight_bits": int(args.weight_bits),
            "n_calib_tokens": int(n_tokens),
            "aspq_rank": aspq_rank,
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
        }
        print(
            f"[ASPQ-GPTQ] saved {layer_name} "
            f"(shape={tuple(W_q.shape)} calib_tokens={n_tokens} aspq_rank={aspq_rank})"
        )

    payload = {
        "__meta__": {
            "checkpoint": args.checkpoint,
            "metric_path": args.metric_path,
            "weight_bits": int(args.weight_bits),
            "num_samples": int(args.num_samples),
            "token_cap": int(args.token_cap),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
            "missing_metric": args.missing_metric,
            "save_dtype": args.save_dtype,
        }
    }
    payload.update(records)
    torch.save(payload, out_path)
    print(f"[ASPQ-GPTQ] wrote {out_path}")


if __name__ == "__main__":
    main()
