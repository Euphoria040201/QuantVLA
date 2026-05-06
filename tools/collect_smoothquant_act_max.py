"""Collect per-input-channel act-max for SmoothQuant.

For each target Linear layer, hook its forward and accumulate
    act_max[j] = max over (calibration tokens) of |x_t,j|

Saves a dict {layer_name -> {act_max: float32 [d_in], weight_max: float32 [d_in]}} as .pt.
DuQuantLinear consumes this when GR00T_DUQUANT_SMOOTHQUANT_PATH is set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from tools.aspq_jacobian_sanity import (  # type: ignore
    load_policy,
    load_libero_samples,
    load_data_config,
    resolve_target_layers,
    unwrap_no_grad_methods,
    restore_no_grad_methods,
    normalized_input_no_inference,
)
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--include-regex", default=(
        r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        r"|action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))).*"
    ))
    p.add_argument("--exclude-regex", default=(
        r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|"
        r"action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)"
    ))
    p.add_argument("--output", required=True)
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scan-dir", default="")
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SQ] loading FP policy ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    layer_names = resolve_target_layers(policy, args)
    print(f"[SQ] target layers: {len(layer_names)}")
    name_set = set(layer_names)

    # Per-layer running stats
    act_max: dict[str, torch.Tensor] = {}
    weight_max: dict[str, torch.Tensor] = {}

    # capture weight column-wise max (constant)
    for n, m in policy.model.named_modules():
        if n in name_set and hasattr(m, "weight"):
            w = m.weight.detach().float().abs()
            # columns = input dim
            w_max_col = w.max(dim=0).values  # [d_in]
            weight_max[n] = w_max_col.cpu().clone()

    handles = []

    def make_hook(name):
        def hook(_mod, inputs, _out):
            x = inputs[0]
            if x is None:
                return
            xa = x.detach().abs()
            # x shape: (..., d_in) — collapse all batch/token dims
            xa = xa.reshape(-1, xa.shape[-1])
            cur_max = xa.max(dim=0).values.float()  # [d_in]
            if name in act_max:
                act_max[name] = torch.maximum(act_max[name].to(cur_max.device), cur_max).cpu()
            else:
                act_max[name] = cur_max.cpu()
        return hook

    for n, m in policy.model.named_modules():
        if n in name_set:
            handles.append(m.register_forward_hook(make_hook(n)))

    # Load samples
    print(f"[SQ] loading {args.num_samples} LIBERO observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    print(f"[SQ] got {len(samples)} samples")

    # Disable grads for speed
    for p in policy.model.parameters():
        p.requires_grad_(False)
    policy.model.eval()
    patched = unwrap_no_grad_methods(policy.model)

    try:
        for i, obs in enumerate(samples, 1):
            print(f"[SQ] sample {i}/{len(samples)}", flush=True)
            with torch.no_grad():
                _ = policy.get_action(obs)
    finally:
        for h in handles:
            h.remove()
        restore_no_grad_methods(patched)

    print(f"[SQ] saving to {out_path}")
    out: dict[str, dict] = {}
    for n in layer_names:
        rec = {}
        if n in act_max:
            rec["act_max"] = act_max[n].cpu()
        if n in weight_max:
            rec["weight_max"] = weight_max[n].cpu()
        if rec:
            out[n] = rec
    torch.save(out, out_path)
    print(f"[SQ] done. {len(out)} layers saved.")


if __name__ == "__main__":
    main()
