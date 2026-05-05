"""ASPQ-GPTQ runtime wrapper and solver.

This module intentionally lives alongside the existing DuQuant-based ASPQ path.
The old path stays intact; this one follows the ASPQ-GPTQ algorithm family:

1. Quantize the full weight with a standard GPTQ-style solver.
2. Quantize the ASPQ action subspace in the metric eigenbasis.
3. Replace only the action-sensitive subspace of the baseline solution.

When the ASPQ metric provides the full output eigenspace, this reduces to the
paper-style rotate -> GPTQ -> rotate-back recipe. When only a top-k subspace is
available, we keep the baseline GPTQ solution in the orthogonal complement and
apply the ASPQ correction inside span(U).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from .duquant_layers import (
    _aspq_basis_from_record,
    _load_aspq_record,
    _parse_per_layer_wbits,
    select_targets,
)
from .duquant_preprocess import (
    PercentileCalibrator,
    compute_mse_scales,
    fake_quantize_sym,
    qmax,
    sanitize_name,
)


_ASPQ_GPTQ_CACHE: Dict[str, Any] = {}


@dataclass
class AspqGptqConfig:
    enabled: Optional[bool] = None
    path: Optional[str] = None
    act_bits: Optional[int] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    weight_bits: Optional[int] = None
    missing: Optional[str] = None

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = os.environ.get("GR00T_ASPQ_GPTQ", "0") not in ("0", "false", "False")
        if self.path is None:
            self.path = os.environ.get("GR00T_ASPQ_GPTQ_PATH")
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("GR00T_ASPQ_GPTQ_ABITS", 8))
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("GR00T_ASPQ_GPTQ_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("GR00T_ASPQ_GPTQ_CALIB_STEPS", 32))
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("GR00T_ASPQ_GPTQ_WBITS_DEFAULT", 4))
        if self.missing is None:
            self.missing = os.environ.get("GR00T_ASPQ_GPTQ_MISSING", "error").lower()
        if self.enabled and not self.path:
            raise ValueError("GR00T_ASPQ_GPTQ=1 requires GR00T_ASPQ_GPTQ_PATH")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_quantized_container(path: Path) -> Any:
    cache_key = str(path.resolve())
    if cache_key in _ASPQ_GPTQ_CACHE:
        return _ASPQ_GPTQ_CACHE[cache_key]
    if path.suffix not in (".pt", ".pth"):
        raise ValueError(f"Unsupported ASPQ-GPTQ weight file: {path}")
    value = _torch_load(path)
    _ASPQ_GPTQ_CACHE[cache_key] = value
    return value


def _record_get(record: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    if isinstance(record, dict):
        for key in keys:
            if key in record:
                return record[key]
    return None


def _load_quantized_record(layer_name: str, weights_path: str) -> Optional[Any]:
    path = Path(weights_path)
    if path.is_dir():
        safe = sanitize_name(layer_name)
        for suffix in (".pt", ".pth"):
            candidate = path / f"{safe}{suffix}"
            if candidate.exists():
                return _load_quantized_container(candidate)
        return None

    container = _load_quantized_container(path)
    if not isinstance(container, dict):
        return container
    if any(key in container for key in ("weight_q", "W_q", "quant_weight")):
        return container
    for key in (layer_name, sanitize_name(layer_name)):
        if key in container:
            return container[key]
    return None


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def _quantize_vector_sym(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    max_q = qmax(bits)
    return torch.clamp(torch.round(x / scale), -max_q - 1, max_q) * scale


def _prepare_hessian(H: torch.Tensor, damp_percent: float) -> Tuple[torch.Tensor, torch.Tensor]:
    H = 0.5 * (H + H.t())
    diag = torch.diag(H).clone()
    dead = diag <= 0
    if dead.any():
        H = H.clone()
        H[dead, dead] = 1.0
    mean_diag = float(diag[~dead].mean().item()) if (~dead).any() else 1.0
    damp = max(float(damp_percent), 0.0) * max(mean_diag, 1e-8)
    if damp > 0:
        idx = torch.arange(H.shape[0], device=H.device)
        H = H.clone()
        H[idx, idx] += damp
    return H, dead


def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    block_size: int = 128,
    damp_percent: float = 0.01,
) -> torch.Tensor:
    """A practical GPTQ-style solver on the input Gram matrix H.

    We quantize each input block independently. This is exact when block_size
    covers the full input dimension and otherwise acts as a standard blockwise
    GPTQ approximation, which is much cheaper for large robot-policy layers.
    """
    if bits <= 0:
        return W.detach().clone()

    W = W.detach().to(dtype=torch.float32)
    h_dtype = torch.float32 if W.is_cuda else torch.float64
    H = H.detach().to(device=W.device, dtype=h_dtype)
    out_features, in_features = W.shape
    if H.shape != (in_features, in_features):
        raise ValueError(f"H has shape {tuple(H.shape)}, expected {(in_features, in_features)}")

    if block_size <= 0 or block_size >= in_features:
        block_size = in_features

    Q = W.clone()
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        W_block = W[:, start:end].clone()
        H_block = H[start:end, start:end].clone()
        H_block, dead = _prepare_hessian(H_block, damp_percent)
        if dead.any():
            W_block[:, dead] = 0

        try:
            chol = torch.linalg.cholesky(H_block)
            Hinv = torch.cholesky_inverse(chol).to(dtype=torch.float32)
        except RuntimeError:
            Hinv = torch.linalg.pinv(H_block).to(dtype=torch.float32)

        block_q = torch.zeros_like(W_block)
        row_scales = compute_mse_scales(W_block, bits).to(dtype=W_block.dtype, device=W_block.device)

        for col in range(W_block.shape[1]):
            denom = float(Hinv[col, col].item())
            if not torch.isfinite(Hinv[col, col]) or abs(denom) < 1e-12:
                denom = 1.0
            w_col = W_block[:, col]
            q_col = _quantize_vector_sym(w_col, row_scales, bits)
            block_q[:, col] = q_col
            err = (w_col - q_col) / denom
            if col + 1 < W_block.shape[1]:
                W_block[:, col + 1:] -= err[:, None] * Hinv[col, col + 1:][None, :]

        Q[:, start:end] = block_q
    return Q


def solve_aspq_gptq_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    U: Optional[torch.Tensor],
    eigvals: Optional[torch.Tensor],
    block_size: int = 128,
    damp_percent: float = 0.01,
    min_eig: float = 1e-12,
) -> torch.Tensor:
    """Quantize W with GPTQ, then apply an ASPQ low-rank correction.

    If U spans the full output space this matches the paper's rotate -> GPTQ ->
    rotate-back view. For top-k metrics we keep the baseline GPTQ solution in
    the complement space and correct only the ASPQ-sensitive subspace.
    """
    baseline_q = gptq_quantize_weight(
        W,
        H,
        bits=bits,
        block_size=block_size,
        damp_percent=damp_percent,
    )

    if U is None or eigvals is None or U.numel() == 0 or eigvals.numel() == 0:
        return baseline_q

    U = U.detach().to(dtype=torch.float32, device=W.device)
    eigvals = eigvals.detach().flatten().to(dtype=torch.float32, device=W.device)
    keep = torch.isfinite(eigvals) & (eigvals > float(min_eig))
    if keep.sum().item() == 0:
        return baseline_q

    U = U[:, keep]
    eigvals = eigvals[keep]
    row_weights = torch.sqrt(torch.clamp(eigvals, min=float(min_eig)))

    W_action = U.t() @ W.to(dtype=torch.float32)
    baseline_action = U.t() @ baseline_q
    weighted_action = W_action * row_weights[:, None]
    weighted_action_q = gptq_quantize_weight(
        weighted_action,
        H,
        bits=bits,
        block_size=block_size,
        damp_percent=damp_percent,
    )
    action_q = weighted_action_q / row_weights[:, None].clamp_min(float(min_eig))
    return baseline_q + U @ (action_q - baseline_action)


class AspqGptqLinear(nn.Module):
    """Runtime wrapper for offline-built ASPQ-GPTQ quantized weights."""

    def __init__(self, base: nn.Linear, name: str, cfg: AspqGptqConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.cfg = cfg
        self.weight_bits = int(weight_bits if weight_bits is not None else (cfg.weight_bits or 0))
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.register_buffer("_weight_fp", base.weight.detach().clone(), persistent=False)
        self.register_buffer("_weight_q", base.weight.detach().clone(), persistent=False)
        self._debug_enabled = os.environ.get("GR00T_ASPQ_GPTQ_DEBUG", "0") not in ("0", "false", "False")

        self._act_scale: Optional[torch.Tensor] = None
        self._act_scale_initialized = False
        self.calibrator = PercentileCalibrator(
            percentile=float(cfg.act_percentile or 99.9),
            max_batches=int(cfg.calib_batches or 32),
        )

        record = _load_quantized_record(name, cfg.path or "")
        if record is None:
            if cfg.missing == "error":
                raise FileNotFoundError(f"No ASPQ-GPTQ record found for layer '{name}' in {cfg.path}")
            self._quant_available = False
        else:
            weight_q = _record_get(record, ("weight_q", "W_q", "quant_weight", "weight"))
            if weight_q is None:
                raise ValueError(f"ASPQ-GPTQ record for '{name}' is missing quantized weight data")
            weight_q_t = _to_tensor(weight_q).to(dtype=base.weight.dtype)
            if tuple(weight_q_t.shape) != tuple(base.weight.shape):
                raise ValueError(
                    f"ASPQ-GPTQ weight for '{name}' has shape {tuple(weight_q_t.shape)}, "
                    f"expected {tuple(base.weight.shape)}"
                )
            self._weight_q.copy_(weight_q_t)
            self._quant_available = True

    @property
    def weight(self) -> torch.Tensor:
        return self._weight_q if self._quant_available else self._weight_fp

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        if self._act_scale_initialized and self._act_scale is not None:
            return self._act_scale

        with torch.no_grad():
            if self.calibrator is not None and not self.calibrator.is_full():
                self.calibrator.observe(x)
                if self.calibrator.is_full():
                    p_vec = self.calibrator.finalize()
                    max_q = qmax(int(self.cfg.act_bits or 8))
                    scale = torch.clamp(p_vec / max_q, min=1e-6)
                    scale = scale.to(dtype=x.dtype, device=x.device).clone()
                    self._act_scale = scale
                    self._act_scale_initialized = True

            if not self._act_scale_initialized:
                x_abs = torch.abs(x.detach().to(torch.float32))
                x2d = x_abs.reshape(-1, x_abs.shape[-1])
                p_vec = torch.quantile(x2d, float(self.cfg.act_percentile or 99.9) / 100.0, dim=0)
                max_q = qmax(int(self.cfg.act_bits or 8))
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                scale = scale.to(dtype=x.dtype, device=x.device).clone()
                self._act_scale = scale
                self._act_scale_initialized = True

        return self._act_scale if self._act_scale is not None else torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = x
        if int(self.cfg.act_bits or 0) > 0:
            s_a = self._get_act_scale(x)
            x_q = fake_quantize_sym(x, s_a, int(self.cfg.act_bits or 0), label="aspq_gptq_activation")
        weight = self._weight_q if self._quant_available else self._weight_fp
        y = torch.nn.functional.linear(x_q, weight.to(dtype=x_q.dtype, device=x_q.device), None)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype, device=y.device)
        if self._debug_enabled and not hasattr(self, "_debug_forward_logged"):
            print(
                f"[GR00T-ASPQ-GPTQ][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y.shape)} "
                f"W{self.weight_bits} A{self.cfg.act_bits} quant={int(self._quant_available)}",
                flush=True,
            )
            self._debug_forward_logged = True
        return y


def wrap_aspq_gptq(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: AspqGptqConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
) -> int:
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    listed = 0
    for name in layer_names:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        if dry_run:
            print(
                f"[GR00T-ASPQ-GPTQ][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits}"
            )
            listed += 1
            continue
        wrapped = AspqGptqLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, wrapped)
        print(
            f"[GR00T-ASPQ-GPTQ][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) "
            f"-> AspqGptqLinear W{wbits} A{cfg.act_bits} quant={int(wrapped._quant_available)}"
        )
        replaced += 1
    if dry_run:
        print(f"[GR00T-ASPQ-GPTQ] Dry-run total layers listed: {listed}")
        return listed
    print(f"[GR00T-ASPQ-GPTQ] Total layers replaced: {replaced}")
    return replaced


def load_aspq_basis_for_layer(
    layer_name: str,
    metric_path: str,
    *,
    out_features: int,
    topk: int = 0,
    min_eig: float = 1e-12,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    record = _load_aspq_record(layer_name, metric_path)
    return _aspq_basis_from_record(
        record,
        out_features=out_features,
        topk=topk,
        min_eig=min_eig,
    )


def enable_aspq_gptq_if_configured(model: nn.Module) -> bool:
    env = os.environ
    activate = env.get("GR00T_ASPQ_GPTQ", "0") not in ("0", "false", "False")
    if not activate:
        return False

    scope = env.get("GR00T_ASPQ_GPTQ_SCOPE", "")
    whitelist = env.get("GR00T_ASPQ_GPTQ_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None
    inc = env.get(
        "GR00T_ASPQ_GPTQ_INCLUDE",
        (
            r".*(?:"
            r"backbone\.eagle_model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"|"
            r"action_head\.model\..*(?:attn1\.to_(?:q|k|v)|attn1\.to_out\.0|ff\.net\.(?:0\.proj|2))"
            r").*"
        ),
    )
    exc = env.get(
        "GR00T_ASPQ_GPTQ_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)"
            r"(?:\.|$)"
        ),
    )
    per_layer_wbits = _parse_per_layer_wbits(env.get("GR00T_ASPQ_GPTQ_WBITS"))
    dry_run = env.get("GR00T_ASPQ_GPTQ_DRYRUN", "0") not in ("0", "false", "False")

    cfg = AspqGptqConfig()
    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[GR00T-ASPQ-GPTQ] SCOPE filter: '{scope}'")
    print(f"[GR00T-ASPQ-GPTQ] Matched Linear layers: {len(layer_names)}")
    wrap_aspq_gptq(model, layer_names, cfg, per_layer_wbits, dry_run=dry_run)
    return True
