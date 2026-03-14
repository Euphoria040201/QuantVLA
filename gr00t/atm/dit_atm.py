import bisect
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

try:
    from diffusers.models.attention import Attention
    from diffusers.models.attention_processor import AttnProcessor2_0
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "diffusers is required for GR00T ATM support. "
        "Please ensure the gr00t environment is activated."
    ) from exc

ATM_ENABLE_ENV = "GR00T_ATM_ENABLE"
ATM_ALPHA_ENV = "GR00T_ATM_ALPHA_PATH"
ATM_SCOPE_ENV = "GR00T_ATM_SCOPE"
OHB_ENABLE_ENV = "GR00T_OHB_ENABLE"
OHB_SCOPE_ENV = "GR00T_OHB_SCOPE"
OHB_ONLY_DIT_ENV = "GR00T_OHB_ONLY_DIT"
OHB_FALLBACK_ENV = "GR00T_OHB_FALLBACK"

# New envs for learnable ATM/OHB
ATM_TRAINABLE_ENV = "GR00T_ATM_TRAINABLE"
OHB_TRAINABLE_ENV = "GR00T_OHB_TRAINABLE"
ATM_LOG_CLIP_ENV = "GR00T_ATM_LOG_CLIP"
TRAINABLE_INIT_FROM_JSON_ENV = "GR00T_ATM_TRAINABLE_INIT_FROM_JSON"

_ATM_PATCH_FLAG = "_gr00t_atm_processor_patched"


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value not in ("0", "false", "False", "")



def _log_clip() -> float:
    return float(os.environ.get(ATM_LOG_CLIP_ENV, "0.3"))



def _is_dit_attention(name: str, module: Attention, scope: str = "dit") -> bool:
    if not isinstance(module, Attention):
        return False
    if scope == "dit":
        return "action_head.model.transformer_blocks" in name
    if scope == "all":
        return True
    return scope in name



def _is_2d_list(x) -> bool:
    """True if x looks like list[list[...]]."""
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)



def step_to_group(step_idx: int, group_edges: Optional[list[int]]) -> int:
    """
    Map denoising step index -> group id using group_edges (len = G+1).
    Example: edges [0,2,4,6,8] => groups {0,1},{2,3},{4,5},{6,7}.
    """
    if not group_edges or len(group_edges) < 2:
        return 0
    g = bisect.bisect_right(group_edges, int(step_idx)) - 1
    return max(0, min(g, len(group_edges) - 2))



def set_atm_ohb_group(model: torch.nn.Module, group_id: int, scope: str = "dit") -> None:
    """
    Switch active ATM/OHB parameters to the given group_id.
    This expects enable_dit_atm_if_configured() has populated *_table buffers.
    """
    gid = int(group_id)
    for name, module in model.named_modules():
        if not _is_dit_attention(name, module, scope=scope):
            continue

        # ATM alpha per-head table: [G, heads]
        alpha_tab = getattr(module, "_atm_alpha_table", None)
        if alpha_tab is not None:
            g = max(0, min(gid, alpha_tab.shape[0] - 1))
            setattr(module, "_atm_alpha_all", alpha_tab[g])

        # OHB per-head table: [G, heads]
        beta_ph_tab = getattr(module, "_ohb_beta_perhead_table", None)
        if beta_ph_tab is not None:
            g = max(0, min(gid, beta_ph_tab.shape[0] - 1))
            setattr(module, "_ohb_beta_perhead", beta_ph_tab[g])

        # OHB scalar table: [G]
        beta_s_tab = getattr(module, "_ohb_beta_scalar_table", None)
        if beta_s_tab is not None:
            g = max(0, min(gid, beta_s_tab.shape[0] - 1))
            setattr(module, "_ohb_beta_scalar", float(beta_s_tab[g].item()))



def _safe_log_tensor(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values.detach().to(torch.float32).clamp_min(1e-6))



def _resolve_json_entry(alpha_data: dict, layer_name: str):
    return alpha_data.get(layer_name) or alpha_data.get(layer_name.replace("model.", "model", 1))


class _ATMProcessor(AttnProcessor2_0):
    """Attention processor with per-head statistics capture and optional scaling."""

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Capture logits statistics before ATM scaling
        capture_cb = getattr(attn, "_atm_capture_callback", None)
        logits_capture_cb = getattr(attn, "_atm_logits_capture_callback", None)
        if capture_cb is not None or logits_capture_cb is not None:
            std, logits_tensor = _compute_logits_std(
                query, key, attention_mask, head_dim, return_logits=True
            )
            if capture_cb is not None:
                capture_cb(attn, std)
            if logits_capture_cb is not None:
                logits_capture_cb(attn, logits_tensor)

        # Apply ATM scaling, preferring learnable log-alpha over static JSON alpha.
        alpha = None
        trainable_alpha = getattr(attn, "atm_log_alpha", None)
        if trainable_alpha is not None:
            alpha = torch.exp(torch.clamp(trainable_alpha, -_log_clip(), _log_clip()))
        else:
            alpha = getattr(attn, "_atm_alpha_all", None)
        if alpha is not None:
            alpha = alpha.to(dtype=query.dtype, device=query.device).view(1, -1, 1, 1)
            query = query * alpha

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Capture per-head RMS for per-head OHB calibration (BEFORE reshape)
        ohb_perhead_capture_cb = getattr(attn, "_atm_ohb_perhead_capture_callback", None)
        if ohb_perhead_capture_cb is not None:
            ohb_perhead_capture_cb(attn, _compute_rms_per_head(hidden_states))

        # Apply per-head OHB beta scaling (BEFORE reshape)
        beta_perhead = getattr(attn, "_ohb_beta_perhead", None)
        if beta_perhead is not None:
            beta_perhead = beta_perhead.to(dtype=hidden_states.dtype, device=hidden_states.device)
            beta_perhead = beta_perhead.view(1, -1, 1, 1)
            hidden_states = hidden_states * beta_perhead

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Capture/output scale for OHB calibration
        ohb_capture_cb = getattr(attn, "_atm_ohb_capture_callback", None)
        if ohb_capture_cb is not None:
            ohb_capture_cb(attn, _compute_rms(hidden_states))

        # Apply OHB scaling, preferring learnable scalar log-beta over static JSON beta.
        beta = None
        trainable_beta = getattr(attn, "ohb_log_beta", None)
        if trainable_beta is not None:
            beta = torch.exp(torch.clamp(trainable_beta, -_log_clip(), _log_clip()))
        else:
            beta = getattr(attn, "_ohb_beta_scalar", None)
        if beta is not None:
            if torch.is_tensor(beta):
                beta = beta.to(dtype=hidden_states.dtype, device=hidden_states.device)
                hidden_states = hidden_states * beta
            elif beta != 1.0:
                hidden_states = hidden_states * beta

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



def _compute_logits_std(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_dim: int,
    *,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    scale = 1.0 / math.sqrt(max(float(head_dim), 1.0))
    logits = torch.matmul(query.to(dtype), key.to(dtype).transpose(-1, -2)) * scale
    if attention_mask is not None:
        valid = attention_mask >= -1e4
    else:
        valid = torch.ones_like(logits, dtype=torch.bool)
    valid = valid.to(dtype)
    count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
    mean = (logits * valid).sum(dim=(-1, -2)) / count
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    var = ((logits - mean) ** 2 * valid).sum(dim=(-1, -2)) / count
    std = torch.sqrt(var.clamp_min(1e-12))
    std = std.detach()
    if return_logits:
        return std, logits.detach()
    return std



def _compute_rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.detach().to(torch.float32) ** 2) + 1e-12)



def _compute_rms_per_head(tensor: torch.Tensor) -> torch.Tensor:
    """Compute RMS per head for per-head OHB.

    Args:
        tensor: Shape (batch, heads, seq, head_dim) - attention output BEFORE reshape.
    Returns:
        Shape (heads,) - RMS value per head averaged over batch, seq, head_dim.
    """
    t = tensor.detach().to(torch.float32)
    rms_per_head = torch.sqrt(torch.mean(t ** 2, dim=(0, 2, 3)) + 1e-12)  # (heads,)
    return rms_per_head



def ensure_dit_attention_patch(model: torch.nn.Module, scope: str = "dit") -> None:
    """Replace attention processors for DiT attention layers with ATM-enabled processor."""
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            if getattr(module, _ATM_PATCH_FLAG, False):
                continue
            module.set_processor(_ATMProcessor())
            setattr(module, _ATM_PATCH_FLAG, True)



def register_atm_capture(
    model: torch.nn.Module,
    callback: Callable[[Attention, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_capture_callback", lambda attn, std, layer=name: callback(layer, std))
            setattr(module, "_atm_capture_name", name)



def register_atm_logits_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_logits_capture_callback",
                lambda attn, logits, layer=name: callback(layer, logits),
            )
            setattr(module, "_atm_logits_capture_name", name)



def register_ohb_capture(
    model: torch.nn.Module,
    callback: Callable[[Attention, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_ohb_capture_callback", lambda attn, rms, layer=name: callback(layer, rms))
            setattr(module, "_atm_ohb_capture_name", name)



def register_ohb_perhead_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    """Register per-head OHB capture callback."""
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_ohb_perhead_capture_callback",
                lambda attn, rms, layer=name: callback(layer, rms),
            )
            setattr(module, "_atm_ohb_perhead_capture_name", name)



def clear_atm_capture(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, Attention):
            for attr in (
                "_atm_capture_callback",
                "_atm_capture_name",
                "_atm_logits_capture_callback",
                "_atm_logits_capture_name",
                "_atm_ohb_capture_callback",
                "_atm_ohb_capture_name",
                "_atm_ohb_perhead_capture_callback",
                "_atm_ohb_perhead_capture_name",
            ):
                if hasattr(module, attr):
                    delattr(module, attr)


@dataclass
class _AlphaSummary:
    matched_layers: int = 0
    total_heads: int = 0



def collect_learnable_atm_ohb_params(model: torch.nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in model.named_modules():
        if hasattr(module, "atm_log_alpha"):
            params.append(module.atm_log_alpha)
        if hasattr(module, "ohb_log_beta"):
            params.append(module.ohb_log_beta)
    return params



def enable_trainable_dit_atm_ohb(
    model: torch.nn.Module,
    *,
    init_from_json_path: Optional[str] = None,
    scope: str = "dit",
    train_atm: bool = True,
    train_ohb: bool = True,
) -> None:
    """Attach learnable ATM/OHB parameters to target attention modules.

    This is the first-step training path: per-head ATM alpha and per-layer scalar OHB beta.
    It preserves the original static JSON path by only attaching parameters when explicitly called.
    """
    ensure_dit_attention_patch(model, scope=scope)

    alpha_data = None
    if init_from_json_path and os.path.exists(init_from_json_path):
        with open(init_from_json_path, "r", encoding="utf-8") as f:
            alpha_data = json.load(f)

    added_atm = 0
    added_ohb = 0
    for name, module in model.named_modules():
        if not _is_dit_attention(name, module, scope=scope):
            continue

        entry = _resolve_json_entry(alpha_data, name) if alpha_data is not None else None

        if train_atm and not hasattr(module, "atm_log_alpha"):
            init_alpha = None
            if entry is not None:
                alpha_values = entry.get("all") or entry.get("alpha")
                if alpha_values is not None:
                    if _is_2d_list(alpha_values):
                        alpha_values = alpha_values[0]
                    init_alpha = _safe_log_tensor(torch.tensor(alpha_values, dtype=torch.float32))
            if init_alpha is None:
                init_alpha = torch.zeros(module.heads, dtype=torch.float32)
            module.register_parameter("atm_log_alpha", nn.Parameter(init_alpha))
            added_atm += 1

        if train_ohb and not hasattr(module, "ohb_log_beta"):
            init_beta = None
            if entry is not None:
                beta_value = entry.get("beta")
                if isinstance(beta_value, list) and len(beta_value) > 0:
                    beta_value = beta_value[0]
                if beta_value is not None:
                    init_beta = _safe_log_tensor(torch.tensor(float(beta_value), dtype=torch.float32))
            if init_beta is None:
                init_beta = torch.zeros((), dtype=torch.float32)
            module.register_parameter("ohb_log_beta", nn.Parameter(init_beta))
            added_ohb += 1

    print(
        f"[GR00T-ATM] Learnable ATM/OHB attached: atm_layers={added_atm}, ohb_layers={added_ohb}, scope={scope}"
    )



def maybe_enable_trainable_dit_atm_ohb(model: torch.nn.Module) -> None:
    train_atm = _env_flag(ATM_TRAINABLE_ENV)
    train_ohb = _env_flag(OHB_TRAINABLE_ENV)
    if not train_atm and not train_ohb:
        return
    scope = os.environ.get(ATM_SCOPE_ENV, "dit")
    init_path = os.environ.get(ATM_ALPHA_ENV) if _env_flag(TRAINABLE_INIT_FROM_JSON_ENV, "1") else None
    enable_trainable_dit_atm_ohb(
        model,
        init_from_json_path=init_path,
        scope=scope,
        train_atm=train_atm,
        train_ohb=train_ohb,
    )



def enable_dit_atm_if_configured(model: torch.nn.Module) -> None:
    atm_flag = os.environ.get(ATM_ENABLE_ENV, "0")
    ohb_flag = os.environ.get(OHB_ENABLE_ENV, "0")
    atm_enabled = atm_flag not in ("0", "false", "False", "")
    ohb_enabled = ohb_flag not in ("0", "false", "False", "")
    if not atm_enabled and not ohb_enabled:
        return

    alpha_path = os.environ.get(ATM_ALPHA_ENV)
    if not alpha_path:
        print("[GR00T-ATM] Scaling requested but GR00T_ATM_ALPHA_PATH not set; skipping.")
        return
    if not os.path.exists(alpha_path):
        print(f"[GR00T-ATM] Alpha JSON not found at {alpha_path}; skipping ATM.")
        return

    with open(alpha_path, "r", encoding="utf-8") as f:
        alpha_data = json.load(f)

    # Optional grouped config
    group_edges = alpha_data.get("group_edges", None)
    if group_edges is not None:
        setattr(model, "_atm_group_edges", group_edges)

    scope = os.environ.get(ATM_SCOPE_ENV, "dit")
    ohb_scope = os.environ.get(OHB_SCOPE_ENV, None)
    if ohb_scope is None:
        ohb_only_dit = os.environ.get(OHB_ONLY_DIT_ENV, "1") not in ("0", "false", "False")
        ohb_scope = "dit" if ohb_only_dit else scope

    summary = _AlphaSummary()
    ohb_layers = 0
    ohb_fallback = float(os.environ.get(OHB_FALLBACK_ENV, "1.0"))

    ensure_dit_attention_patch(model, scope=scope)

    for name, module in model.named_modules():
        if not _is_dit_attention(name, module, scope=scope):
            continue

        alpha_entry = _resolve_json_entry(alpha_data, name)

        if not alpha_entry:
            alpha_values = None
            beta_value = None
            beta_perhead_values = None
        else:
            alpha_values = alpha_entry.get("all") or alpha_entry.get("alpha")
            beta_value = alpha_entry.get("beta")
            beta_perhead_values = alpha_entry.get("beta_perhead")

        # ---------- ATM alpha ----------
        if atm_enabled and alpha_values:
            if _is_2d_list(alpha_values):
                # [G, heads]
                alpha_table = torch.tensor(alpha_values, dtype=torch.float32)
                setattr(module, "_atm_alpha_table", alpha_table)
                setattr(module, "_atm_alpha_all", alpha_table[0])  # default group0
                summary.matched_layers += 1
                summary.total_heads += int(alpha_table.shape[1])
            else:
                # [heads]
                alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)
                setattr(module, "_atm_alpha_all", alpha_tensor)
                summary.matched_layers += 1
                summary.total_heads += len(alpha_values)

        # ---------- OHB beta ----------
        if ohb_enabled and _is_dit_attention(name, module, scope=ohb_scope):
            # per-head beta preferred
            if beta_perhead_values is not None:
                if _is_2d_list(beta_perhead_values):
                    beta_ph_table = torch.tensor(beta_perhead_values, dtype=torch.float32)  # [G, heads]
                    setattr(module, "_ohb_beta_perhead_table", beta_ph_table)
                    setattr(module, "_ohb_beta_perhead", beta_ph_table[0])
                else:
                    beta_tensor = torch.tensor(beta_perhead_values, dtype=torch.float32)  # [heads]
                    setattr(module, "_ohb_beta_perhead", beta_tensor)
                ohb_layers += 1
            else:
                # per-layer beta: supports [G] or scalar
                if isinstance(beta_value, list) and len(beta_value) > 0:
                    beta_s_table = torch.tensor(beta_value, dtype=torch.float32)  # [G]
                    setattr(module, "_ohb_beta_scalar_table", beta_s_table)
                    setattr(module, "_ohb_beta_scalar", float(beta_s_table[0].item()))
                else:
                    beta = float(beta_value) if beta_value is not None else ohb_fallback
                    setattr(module, "_ohb_beta_scalar", beta)
                ohb_layers += 1

    if summary.matched_layers == 0 and atm_enabled:
        print(f"[GR00T-ATM] No attention layers matched alpha JSON ({alpha_path}).")
    elif atm_enabled:
        print(
            f"[GR00T-ATM] ATM enabled for {summary.matched_layers} layers "
            f"({summary.total_heads} heads) using {alpha_path}"
        )
        if group_edges is not None:
            print(f"[GR00T-ATM] Grouped ATM detected: group_edges={group_edges}")

    if ohb_enabled:
        if ohb_layers == 0:
            print(
                f"[GR00T-ATM] OHB requested but no layers found (scope={ohb_scope}); "
                f"fallback beta={ohb_fallback}"
            )
        else:
            print(f"[GR00T-ATM] OHB enabled for {ohb_layers} layers using {alpha_path}")
            if group_edges is not None:
                print(f"[GR00T-ATM] Grouped OHB detected: group_edges={group_edges}")
