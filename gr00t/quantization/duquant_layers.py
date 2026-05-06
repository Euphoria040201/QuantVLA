"""
GR00T DuQuant W4A8 Fake Quantization Layers

Adapted from OpenPI duquant implementation for GR00T model quantization.
Supports quantization of LLM (Eagle VLM) and DiT (action transformer) layers.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import numpy as np
from torch import nn

from .duquant_preprocess import (
    PackResult,
    PercentileCalibrator,
    apply_input_transform,
    apply_output_restore,
    apply_bias_row_rot,
    compute_mse_scales,
    fake_quantize_sym,
    load_pack,
    pack_weight,
    qmax,
    sanitize_name,
    save_pack,
    transform_weight_for_forward,
)


_ASPQ_CACHE: Dict[str, Any] = {}


@dataclass
class DuQuantConfig:
    """DuQuant configuration matching OpenPI parameters.

    NOTE: Default values are set to None and resolved in __post_init__ to ensure
    environment variables are read at instantiation time, not at module import time.
    """
    weight_bits: Optional[int] = None
    act_bits: Optional[int] = None
    block_size: Optional[int] = None
    lambda_smooth: Optional[float] = None
    enable_permute: Optional[bool] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    pack_dir: Optional[str] = None
    row_rot_mode: Optional[str] = None
    block_out_size: Optional[int] = None
    aspq_enabled: Optional[bool] = None
    aspq_path: Optional[str] = None
    aspq_topk: Optional[int] = None
    aspq_min_eig: Optional[float] = None
    aspq_missing: Optional[str] = None

    def __post_init__(self):
        """Read environment variables at instantiation time."""
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("GR00T_DUQUANT_WBITS_DEFAULT", 4))
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("GR00T_DUQUANT_ABITS", 8))
        if self.block_size is None:
            self.block_size = int(os.environ.get("GR00T_DUQUANT_BLOCK", 16))
        if self.lambda_smooth is None:
            self.lambda_smooth = float(os.environ.get("GR00T_DUQUANT_LS", 0.15))
        if self.enable_permute is None:
            self.enable_permute = os.environ.get("GR00T_DUQUANT_PERMUTE", "1") not in ("0", "false", "False")
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("GR00T_DUQUANT_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("GR00T_DUQUANT_CALIB_STEPS", 32))
        if self.pack_dir is None:
            self.pack_dir = os.environ.get("GR00T_DUQUANT_PACKDIR", None)
        if self.row_rot_mode is None:
            self.row_rot_mode = os.environ.get("GR00T_DUQUANT_ROW_ROT", "restore")
        if self.block_out_size is None:
            self.block_out_size = int(os.environ.get("GR00T_DUQUANT_BLOCK_OUT", os.environ.get("GR00T_DUQUANT_BLOCK", 16)))
        if self.aspq_enabled is None:
            self.aspq_enabled = os.environ.get("GR00T_DUQUANT_ASPQ", "0") not in ("0", "false", "False")
        if self.aspq_path is None:
            self.aspq_path = (
                os.environ.get("GR00T_DUQUANT_ASPQ_PATH")
                or os.environ.get("GR00T_DUQUANT_ASPQ_DIR")
            )
        if self.aspq_topk is None:
            self.aspq_topk = int(os.environ.get("GR00T_DUQUANT_ASPQ_TOPK", 0))
        if self.aspq_min_eig is None:
            self.aspq_min_eig = float(os.environ.get("GR00T_DUQUANT_ASPQ_MIN_EIG", 1e-12))
        if self.aspq_missing is None:
            self.aspq_missing = os.environ.get("GR00T_DUQUANT_ASPQ_MISSING", "error").lower()
        if self.aspq_enabled and not self.aspq_path:
            raise ValueError("GR00T_DUQUANT_ASPQ=1 requires GR00T_DUQUANT_ASPQ_PATH or GR00T_DUQUANT_ASPQ_DIR")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_aspq_container(path: Path) -> Any:
    cache_key = str(path.resolve())
    if cache_key in _ASPQ_CACHE:
        return _ASPQ_CACHE[cache_key]
    if path.suffix == ".pt" or path.suffix == ".pth":
        value = _torch_load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:  # type: ignore[name-defined]
            value = {key: f[key] for key in f.files}
    else:
        raise ValueError(f"Unsupported ASPQ metric file: {path}")
    _ASPQ_CACHE[cache_key] = value
    return value


def _load_aspq_record(layer_name: str, aspq_path: str) -> Optional[Any]:
    path = Path(aspq_path)
    if path.is_dir():
        safe = sanitize_name(layer_name)
        for suffix in (".pt", ".pth", ".npz"):
            candidate = path / f"{safe}{suffix}"
            if candidate.exists():
                return _load_aspq_container(candidate)
        return None

    container = _load_aspq_container(path)
    if not isinstance(container, dict):
        return container
    if any(key in container for key in ("M", "U", "eigvecs", "eigenvectors")):
        return container
    for key in (layer_name, sanitize_name(layer_name)):
        if key in container:
            return container[key]
    return None


def _record_get(record: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    if isinstance(record, dict):
        for key in keys:
            if key in record:
                return record[key]
    return None


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(value, dtype=torch.float32)


def _aspq_basis_from_record(
    record: Any,
    *,
    out_features: int,
    topk: int,
    min_eig: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return U, eigvals with U shaped [out_features, k]."""
    if record is None:
        return None

    U_value = _record_get(record, ("U", "eigvecs", "eigenvectors", "u"))
    eig_value = _record_get(record, ("eigvals", "eigenvalues", "lambda", "lambdas", "Lambda", "S"))
    M_value = _record_get(record, ("M", "metric", "action_metric"))

    if U_value is None and M_value is None and isinstance(record, torch.Tensor):
        M_value = record

    if U_value is not None and eig_value is not None:
        U = _to_tensor(U_value)
        eigvals = _to_tensor(eig_value).flatten()
        if U.ndim != 2:
            raise ValueError(f"ASPQ U must be rank-2, got shape {tuple(U.shape)}")
        if U.shape[0] != out_features and U.shape[1] == out_features:
            U = U.t().contiguous()
        if U.shape[0] != out_features:
            raise ValueError(f"ASPQ U has incompatible shape {tuple(U.shape)} for out_features={out_features}")
        if eigvals.numel() != U.shape[1]:
            raise ValueError(f"ASPQ eigvals length {eigvals.numel()} does not match U columns {U.shape[1]}")
    elif M_value is not None:
        M = _to_tensor(M_value)
        if M.shape != (out_features, out_features):
            raise ValueError(f"ASPQ M has shape {tuple(M.shape)}, expected {(out_features, out_features)}")
        M = 0.5 * (M + M.t())
        eigvals, U = torch.linalg.eigh(M)
    else:
        raise ValueError("ASPQ metric record must contain either M or both U and eigvals")

    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    U = U[:, order]
    keep = torch.isfinite(eigvals) & (eigvals > float(min_eig))
    eigvals = eigvals[keep]
    U = U[:, keep]
    if topk and topk > 0:
        eigvals = eigvals[:topk]
        U = U[:, :topk]
    if eigvals.numel() == 0:
        return None
    return U.contiguous(), eigvals.contiguous()


def _parse_per_layer_wbits(env_val: Optional[str]) -> Dict[str, int]:
    """Parse per-layer weight bits from environment variable."""
    if not env_val:
        return {}
    result: Dict[str, int] = {}
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            pass
    return result


class DuQuantLinear(nn.Module):
    """DuQuant quantized linear layer with W4A8 fake quantization."""

    def __init__(self, base: nn.Linear, name: str, cfg: DuQuantConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        # SmoothQuant absorption: pre-divide x by s, pre-multiply W column-wise by s
        # We modify the weight HERE before pack so DuQuant rotation is computed on the rescaled W.
        _w_for_pack = base.weight.detach().clone()
        self._sq_s_inv: Optional[torch.Tensor] = None
        sq_path = os.environ.get("GR00T_DUQUANT_SMOOTHQUANT_PATH", "")
        if sq_path and os.path.exists(sq_path):
            try:
                _sq_alpha = float(os.environ.get("GR00T_DUQUANT_SMOOTHQUANT_ALPHA", "0.5"))
                _sq_clip = float(os.environ.get("GR00T_DUQUANT_SMOOTHQUANT_CLIP", "1e3"))
                _sq_min = 1.0 / _sq_clip
                _sq_data = torch.load(sq_path, map_location="cpu", weights_only=False)
                _rec = _sq_data.get(name, None)
                if _rec is not None and "act_max" in _rec and "weight_max" in _rec:
                    a = _rec["act_max"].float().clamp(min=1e-5)
                    w_col = _rec["weight_max"].float().clamp(min=1e-5)
                    s = (a.pow(_sq_alpha) / w_col.pow(1.0 - _sq_alpha)).clamp(min=_sq_min, max=_sq_clip)
                    # scale weight columns (input dim) by s
                    s_dev = s.to(dtype=_w_for_pack.dtype)
                    _w_for_pack = _w_for_pack * s_dev[None, :]
                    self.register_buffer("_sq_s_inv", (1.0 / s_dev).clone(), persistent=False)
                    if os.environ.get("GR00T_DUQUANT_DEBUG", "0") not in ("0", "false", "False"):
                        print(f"[GR00T-SQ] {name} alpha={_sq_alpha} s_min={float(s.min()):.3e} s_max={float(s.max()):.3e} a_max={float(a.max()):.3f} w_max={float(w_col.max()):.3f}", flush=True)
            except Exception as _e:
                print(f"[GR00T-SQ][WARN] {name}: {_e}", flush=True)
        self.register_buffer("_weight", _w_for_pack)

        # Config
        self.cfg = cfg
        self.weight_bits = cfg.weight_bits if weight_bits is None else int(weight_bits)

        # Load or compute packing
        pack = load_pack(self.name, cfg.pack_dir)
        if pack is None:
            pack = pack_weight(
                self._weight,
                block_size=cfg.block_size,
                block_out_size=cfg.block_out_size,
                enable_permute=cfg.enable_permute,
                lambda_smooth=cfg.lambda_smooth,
            )
            save_pack(self.name, pack, cfg.pack_dir)
        self.pack: PackResult = pack

        # Cache rotation matrices as torch tensors
        if pack.perm is not None:
            self.register_buffer("_perm_cache", torch.from_numpy(pack.perm).long())
        else:
            self._perm_cache = None

        # Cache input rotation matrices
        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                buffer_name = f"_R_in_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_in_block_indices.append(b)

        # Cache output rotation matrices
        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                buffer_name = f"_R_out_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_out_block_indices.append(b)

        # Store metadata
        self._block_size = int(pack.meta.get("block_size", 16))
        self._block_out_size = int(pack.meta.get("block_out_size", self._block_size))

        # Pre-build a single stacked rotation buffer per side. The hot path of every
        # forward used to do `torch.stack([R_in_cache[b] for b in range(n_blocks)])`,
        # which dominated CPU time when there are O(1k) Linear layers per step.
        # Building it once here removes that overhead entirely.
        # PyTorch >= 2.5 rejects register_buffer if the name is already a regular
        # attribute, so we register the buffer up-front (with None) before the
        # conditional branch.
        self.register_buffer("_R_in_stack", None, persistent=False)
        if pack.R_in_blocks and self.in_features % self._block_size == 0:
            n_in_blocks = self.in_features // self._block_size
            if all(b in pack.R_in_blocks for b in range(n_in_blocks)) and all(
                pack.R_in_blocks[b].shape == (self._block_size, self._block_size)
                for b in range(n_in_blocks)
            ):
                stack_in = torch.stack(
                    [torch.from_numpy(pack.R_in_blocks[b]) for b in range(n_in_blocks)],
                    dim=0,
                ).to(dtype=self._weight.dtype).contiguous()
                self._R_in_stack = stack_in

        self.register_buffer("_R_out_stack", None, persistent=False)
        if pack.R_out_blocks and self.out_features % self._block_out_size == 0:
            n_out_blocks = self.out_features // self._block_out_size
            if all(b in pack.R_out_blocks for b in range(n_out_blocks)) and all(
                pack.R_out_blocks[b].shape == (self._block_out_size, self._block_out_size)
                for b in range(n_out_blocks)
            ):
                stack_out = torch.stack(
                    [torch.from_numpy(pack.R_out_blocks[b]) for b in range(n_out_blocks)],
                    dim=0,
                ).to(dtype=self._weight.dtype).contiguous()
                self._R_out_stack = stack_out

        # Calibrator for activation
        self.calibrator = PercentileCalibrator(
            percentile=cfg.act_percentile, max_batches=cfg.calib_batches
        ) if self.cfg.act_bits > 0 else None
        self.register_buffer("_act_scale", None)
        self._act_scale_initialized = False

        # Cache transformed weight
        self._cached_weight_key: Optional[Tuple[Any, ...]] = None
        self.register_buffer("_W_t", torch.zeros_like(self._weight))
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))

        # Pre-cache quantized weights
        self._precache_weight = os.environ.get("GR00T_DUQUANT_PRECACHE_WEIGHTS", "1") not in (
            "0", "false", "False",
        )
        if self._precache_weight:
            self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight))
        else:
            self._W_t_quantized = None
        self._weight_quantized_cached = False

        self._bias_rot: Optional[torch.Tensor] = None
        self._debug_enabled = os.environ.get("GR00T_DUQUANT_DEBUG", "0") not in ("0", "false", "False")
        self._debug_forward_logged = False
        # Resolve hot-path env flags ONCE at construction. Reading os.environ on every
        # forward is itself a measurable cost when called O(1k * denoising_steps) times.
        self._layer_stats_enabled = os.environ.get("GR00T_DEBUG_LAYER_STATS", "0") not in (
            "0", "false", "False",
        )
        try:
            self._layer_stats_every = int(os.environ.get("GR00T_DEBUG_LAYER_STATS_EVERY", "200"))
        except ValueError:
            self._layer_stats_every = 200
        self._aspq_enabled = bool(cfg.aspq_enabled)
        self._aspq_available = False
        if self._aspq_enabled:
            basis = None
            try:
                record = _load_aspq_record(self.name, cfg.aspq_path or "")
                basis = _aspq_basis_from_record(
                    record,
                    out_features=self.out_features,
                    topk=int(cfg.aspq_topk or 0),
                    min_eig=float(cfg.aspq_min_eig or 0.0),
                )
            except Exception as exc:
                if cfg.aspq_missing == "error":
                    raise
                if self._debug_enabled:
                    import logging
                    logging.warning(f"[GR00T-DUQUANT][ASPQ] {self.name}: failed to load ASPQ metric: {exc}")
            if basis is None:
                if cfg.aspq_missing == "error":
                    raise FileNotFoundError(
                        f"No usable ASPQ metric found for layer '{self.name}' in {cfg.aspq_path}"
                    )
                if self._debug_enabled:
                    import logging
                    logging.warning(f"[GR00T-DUQUANT][ASPQ] {self.name}: metric missing; using baseline quantization")
            else:
                U, eigvals = basis
                self.register_buffer("_aspq_U_orig", U, persistent=False)
                self.register_buffer("_aspq_eigvals", eigvals, persistent=False)
                self._aspq_available = True

    def _get_R_in_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_in rotation matrices on the correct device."""
        if not hasattr(self, '_R_in_cache_dict'):
            self._R_in_cache_dict = {}
        for b in self._R_in_block_indices:
            self._R_in_cache_dict[b] = getattr(self, f"_R_in_{b}")
        return self._R_in_cache_dict

    def _get_R_out_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_out rotation matrices on the correct device."""
        if not hasattr(self, '_R_out_cache_dict'):
            self._R_out_cache_dict = {}
        for b in self._R_out_block_indices:
            self._R_out_cache_dict[b] = getattr(self, f"_R_out_{b}")
        return self._R_out_cache_dict

    def _aspq_basis_for_weight(self, W_t: torch.Tensor, apply_row_rot: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        U = self._aspq_U_orig.to(dtype=W_t.dtype, device=W_t.device)
        eigvals = self._aspq_eigvals.to(dtype=W_t.dtype, device=W_t.device)

        R_out_cache = self._get_R_out_cache()
        if apply_row_rot and R_out_cache:
            U = U.clone()
            out_features = U.shape[0]
            n_row_blocks = (out_features + self._block_out_size - 1) // self._block_out_size
            for b in range(n_row_blocks):
                if b not in R_out_cache:
                    continue
                rs = b * self._block_out_size
                re = min((b + 1) * self._block_out_size, out_features)
                Rb = R_out_cache[b][: (re - rs), : (re - rs)].to(dtype=U.dtype, device=U.device)
                U[rs:re, :] = Rb @ U[rs:re, :]
        return U, eigvals

    def _quantize_weight(self, W_t: torch.Tensor, scales: torch.Tensor, apply_row_rot: bool, *, label: str) -> torch.Tensor:
        if self.weight_bits <= 0:
            return W_t
        if not (self._aspq_enabled and self._aspq_available):
            return fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label=label)

        U, eigvals = self._aspq_basis_for_weight(W_t, apply_row_rot=apply_row_rot)
        if U.numel() == 0:
            return fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label=label)

        baseline_q = fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label=f"{label}_baseline")
        W_action = U.t() @ W_t
        baseline_action = U.t() @ baseline_q

        row_weights = torch.sqrt(torch.clamp(eigvals, min=float(self.cfg.aspq_min_eig or 1e-12)))
        W_weighted = W_action * row_weights[:, None]
        action_scales = compute_mse_scales(W_weighted, self.weight_bits)
        action_q = fake_quantize_sym(
            W_weighted,
            action_scales[:, None],
            self.weight_bits,
            label=f"{label}_aspq_action",
        ) / row_weights[:, None].clamp_min(1e-12)

        return baseline_q + U @ (action_q - baseline_action)

    @property
    def weight(self) -> torch.Tensor:
        """Expose packed weight buffer for compatibility."""
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self._weight.copy_(value)

    def _maybe_update_weight_cache(self) -> None:
        apply_row = (self.cfg.row_rot_mode != "0")
        if self._aspq_enabled:
            key = (
                str(self._weight.device),
                self._weight.dtype,
                int(self.weight_bits),
                int(apply_row),
                "aspq",
                int(self._aspq_available),
            )
        else:
            key = (str(self._weight.device), self._weight.dtype, int(self.weight_bits), int(apply_row))
        if self._cached_weight_key == key:
            return

        from .duquant_preprocess import transform_weight_for_forward_optimized

        W_t, scales = transform_weight_for_forward_optimized(
            self._weight,
            self.pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row,
            perm_cache=self._perm_cache,
            R_in_cache=self._get_R_in_cache(),
            R_out_cache=self._get_R_out_cache(),
            block_size=self._block_size,
            block_out_size=self._block_out_size,
        )
        self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)

        # Pre-quantize weights if enabled
        if self._precache_weight and self.weight_bits > 0:
            with torch.no_grad():
                self._W_t_quantized.copy_(self._quantize_weight(W_t, scales, apply_row, label="weight_prequant"))
            self._weight_quantized_cached = True
        else:
            self._weight_quantized_cached = False

        self._cached_weight_key = key
        if self.bias is not None:
            if self.cfg.row_rot_mode == "propagate" and self.pack.R_out_blocks is not None:
                with torch.no_grad():
                    from .duquant_preprocess import apply_bias_row_rot_optimized
                    self._bias_rot = apply_bias_row_rot_optimized(
                        self.bias.detach(), self.pack, self._get_R_out_cache(), self._block_out_size
                    )
            else:
                self._bias_rot = None
        if self._debug_enabled:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][CACHE] {self.name} device={self._weight.device} dtype={self._weight.dtype} "
                f"Wbits={self.weight_bits} Abits={self.cfg.act_bits} block_in={self.cfg.block_size} "
                f"permute={self.pack.perm is not None} row_rot={self.cfg.row_rot_mode} "
                f"aspq={self._aspq_enabled and self._aspq_available}"
            )
            if self._weight_quantized_cached:
                logging.info(f"[GR00T-DUQUANT][CACHE] {self.name} pre-quantized weights cached")

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        if self._act_scale_initialized:
            return self._act_scale

        with torch.no_grad():
            if self.calibrator is not None and not self.calibrator.is_full():
                self.calibrator.observe(x)
                if self.calibrator.is_full():
                    p_vec = self.calibrator.finalize()
                    max_q = qmax(self.cfg.act_bits)
                    scale = torch.clamp(p_vec / max_q, min=1e-6)
                    scale = scale.to(dtype=x.dtype, device=x.device).clone()
                    if self._act_scale is None:
                        self._act_scale = scale
                    else:
                        self._act_scale.copy_(scale)
                    self._act_scale_initialized = True

            if not self._act_scale_initialized:
                x_abs = torch.abs(x.detach().to(torch.float32))
                C = x_abs.shape[-1]
                x2d = x_abs.reshape(-1, C)
                p_vec = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                scale = scale.to(dtype=x.dtype, device=x.device).clone()
                if self._act_scale is None:
                    self._act_scale = scale
                else:
                    self._act_scale.copy_(scale)
                self._act_scale_initialized = True

        return self._act_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SmoothQuant: pre-divide x by per-channel s (broadcast on input dim)
        if self._sq_s_inv is not None:
            x = x * self._sq_s_inv.to(dtype=x.dtype, device=x.device)

        # Per-block input rotation. Fast path uses the prebuilt stacked tensor so we
        # avoid Python list-comp / torch.stack overhead per layer per step.
        if self._perm_cache is not None:
            x = x.index_select(dim=-1, index=self._perm_cache)
        if self._R_in_stack is not None:
            orig_shape = x.shape
            n_blocks = self._R_in_stack.shape[0]
            x_t = torch.einsum(
                "rnb,nbc->rnc",
                x.reshape(-1, n_blocks, self._block_size),
                self._R_in_stack,
            ).reshape(orig_shape)
        elif self._R_in_block_indices:
            from .duquant_preprocess import apply_input_transform_optimized
            x_t = apply_input_transform_optimized(
                x, self.pack, None, self._get_R_in_cache(), self._block_size
            )
        else:
            x_t = x

        # Fake-quantize activations if enabled
        if self.cfg.act_bits > 0:
            s_a = self._get_act_scale(x_t)
            x_t = fake_quantize_sym(x_t, s_a, self.cfg.act_bits, label="activation_forward")

        # Transform and fake-quantize weights
        self._maybe_update_weight_cache()

        # Use pre-quantized weights
        if self._weight_quantized_cached:
            y_lin = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
        elif self.weight_bits > 0:
            y_lin = torch.nn.functional.linear(
                x_t,
                self._quantize_weight(
                    self._W_t,
                    self._w_scales,
                    self.cfg.row_rot_mode != "0",
                    label="weight_fallback",
                ),
                None
            )
        else:
            y_lin = torch.nn.functional.linear(x_t, self._W_t, None)

        # Apply row restore if requested
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            if self._R_out_stack is not None:
                orig_shape = y_lin.shape
                n_out = self._R_out_stack.shape[0]
                y_lin = torch.einsum(
                    "rnb,nbc->rnc",
                    y_lin.reshape(-1, n_out, self._block_out_size),
                    self._R_out_stack,
                ).reshape(orig_shape)
            else:
                from .duquant_preprocess import apply_output_restore_optimized
                y_lin = apply_output_restore_optimized(
                    y_lin, self.pack, self._get_R_out_cache(), self._block_out_size
                )
            if self.bias is not None:
                y_lin = y_lin + self.bias
        else:
            if self.bias is not None:
                bias_to_add = (
                    self._bias_rot
                    if self.cfg.row_rot_mode == "propagate" and self._bias_rot is not None
                    else self.bias
                )
                y_lin = y_lin + bias_to_add
        if self._debug_enabled and not self._debug_forward_logged:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y_lin.shape)} "
                f"weight_bits={self.weight_bits} act_bits={self.cfg.act_bits}"
            )
            self._debug_forward_logged = True
        # Optional per-layer stat logging (env-gated, resolved once at init)
        if self._layer_stats_enabled:
            try:
                _every = self._layer_stats_every
                _cnt = getattr(self, "_dbg_call_cnt", 0) + 1
                self._dbg_call_cnt = _cnt
                if _cnt == 1 or (_cnt % _every) == 0:
                    with torch.no_grad():
                        xa = x.detach().abs()
                        ya = y_lin.detach()
                        nan_cnt = int(torch.isnan(ya).sum().item())
                        inf_cnt = int(torch.isinf(ya).sum().item())
                        x_max = float(xa.max().item()) if xa.numel() else 0.0
                        y_amax = float(ya.abs().max().item()) if ya.numel() else 0.0
                        y_norm = float(ya.float().norm().item()) if ya.numel() else 0.0
                        # crude clip rate: fraction of |x| beyond p99.9 of own batch (proxy)
                        if xa.numel() > 1024:
                            thr = float(torch.quantile(xa.flatten()[: min(xa.numel(), 1<<20)].float(), 0.999).item())
                            clip_rate = float((xa > thr).float().mean().item())
                        else:
                            thr, clip_rate = 0.0, 0.0
                    print(
                        f"[LAYER] call#{_cnt} {self.name} "
                        f"xmax={x_max:.3f} ymax={y_amax:.3f} ynorm={y_norm:.2f} "
                        f"nan={nan_cnt} inf={inf_cnt} clip_rate@p99.9={clip_rate:.3e} "
                        f"aspq={int(getattr(self, '_aspq_available', False))}",
                        flush=True,
                    )
            except Exception as _e:
                pass
        return y_lin


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def select_targets(
    model: nn.Module,
    *,
    include_regex: str = r".*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|up_proj|down_proj|gate_proj).*",
    exclude_regex: str = r"(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)",
    scope_prefix: Optional[str] = None,
    whitelist: Optional[Iterable[str]] = None,
    blacklist: Optional[Iterable[str]] = None,
) -> List[Tuple[str, nn.Linear]]:
    """Select linear layers to quantize based on regex patterns."""
    inc = re.compile(include_regex)
    exc = re.compile(exclude_regex)
    wl = set(whitelist or [])
    bl = set(blacklist or [])
    results: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if scope_prefix is not None and not name.startswith(scope_prefix):
            continue
        if name in bl:
            continue
        if wl and name not in wl:
            continue
        if not wl and (not inc.search(name) or exc.search(name)):
            continue
        results.append((name, mod))
    return results


def wrap_duquant(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: DuQuantConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
) -> None:
    """Wrap selected layers with DuQuant quantization."""
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    listed = 0
    for name in layer_names:
        # Skip action head by default unless explicitly requested
        if os.environ.get("GR00T_DUQUANT_INCLUDE_ACTION_HEAD", "0") in ("0", "false", "False"):
            is_action_head = "action_head" in name and not name.startswith("action_head.model.")
            if (
                name.endswith("action_out_proj")
                or ".action_out_proj" in name
                or is_action_head
            ):
                continue
        parent, attr = _get_parent_module_and_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        if dry_run:
            msg = (
                f"[GR00T-DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} "
                f"block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode} "
                f"aspq={cfg.aspq_enabled}"
            )
            print(msg)
            listed += 1
            continue
        dq = DuQuantLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, dq)
        # Use actual block sizes from pack (not cfg defaults)
        actual_block_in = dq._block_size
        actual_block_out = dq._block_out_size
        print(
            f"[GR00T-DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> DuQuantLinear "
            f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} block_in={actual_block_in} "
            f"block_out={actual_block_out} row_rot={cfg.row_rot_mode} aspq={dq._aspq_enabled and dq._aspq_available}"
        )
        replaced += 1
    if dry_run:
        print(f"[GR00T-DUQUANT] Dry-run total layers listed: {listed}")
    else:
        print(f"[GR00T-DUQUANT] Total layers replaced: {replaced}")


def enable_duquant_if_configured(model: nn.Module) -> None:
    """
    Entry point to enable DuQuant based on environment variables.

    Activation conditions:
    - If GR00T_DUQUANT_DRYRUN is set => dry-run listing only
    - Or if any GR00T_DUQUANT_* variable (other than PACKDIR) is set => perform replacement
    - Otherwise do nothing
    """
    env = os.environ
    keys = [k for k in env.keys() if k.startswith("GR00T_DUQUANT_")]
    activate = any(k not in ("GR00T_DUQUANT_PACKDIR",) for k in keys)
    if not activate:
        return

    # Scope defaults to empty (search entire model)
    scope = env.get("GR00T_DUQUANT_SCOPE", "")
    whitelist = env.get("GR00T_DUQUANT_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None

    # Default: quantize LLM + DiT MLP layers (matching OpenPI pattern)
    # Include LLM attention+MLP and DiT MLP projections
    inc = env.get(
        "GR00T_DUQUANT_INCLUDE",
        (
            r".*(?:"
            r"backbone\.eagle_model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"|"
            r"action_head\.model\..*(?:attn1\.to_(?:q|k|v)|attn1\.to_out\.0|ff\.net\.(?:0\.proj|2))"
            r").*"
        ),
    )
    # Exclude vision encoder, embeddings, auxiliary projectors
    exc = env.get(
        "GR00T_DUQUANT_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)"
            r"(?:\.|$)"
        ),
    )

    per_layer_wbits = _parse_per_layer_wbits(env.get("GR00T_DUQUANT_WBITS"))
    dry_run = env.get("GR00T_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

    cfg = DuQuantConfig()

    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[GR00T-DUQUANT] SCOPE filter: '{scope}'")
    print(f"[GR00T-DUQUANT] Matched Linear layers: {len(layer_names)}")

    if len(layer_names) == 0 and scope:
        # Debug: print some layer names to help diagnose
        all_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        print(f"[GR00T-DUQUANT] DEBUG: Total Linear layers in model: {len(all_linears)}")
        print(f"[GR00T-DUQUANT] DEBUG: First 10 Linear layer names:")
        for name, _ in all_linears[:10]:
            print(f"[GR00T-DUQUANT] DEBUG:   {name}")
        if scope:
            matching_prefix = [n for n, _ in all_linears if n.startswith(scope.rstrip('.'))]
            print(f"[GR00T-DUQUANT] DEBUG: Layers matching prefix '{scope.rstrip('.')}': {len(matching_prefix)}")
            if matching_prefix:
                for name in matching_prefix[:5]:
                    print(f"[GR00T-DUQUANT] DEBUG:   {name}")

    if dry_run:
        wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=True)
        return
    wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=False)
