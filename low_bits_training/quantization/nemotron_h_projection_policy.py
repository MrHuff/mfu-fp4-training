"""Selective FP4 projection routing for exact Nemotron-H models.

The dense MLP blocks in Nemotron-H can use the existing fused FFN wrappers, but
attention and Mamba blocks expose different projection names from Llama. This
module centralizes the small amount of plumbing needed to route those
projections through the same lower-level FP4 linear wrappers.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.tools.logging import logger


_HEAD_KEYWORDS = ("output", "lm_head")
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def nemotron_h_fp4_output_head_enabled(default: bool = False) -> bool:
    return _env_bool("LBT_NEMOTRON_H_FP4_OUTPUT_HEAD", default)


def nemotron_h_fp4_mamba_out_enabled(default: bool = False) -> bool:
    return _env_bool("LBT_NEMOTRON_H_FP4_MAMBA_OUT_PROJ", default)


def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr_name):
        return getattr(obj, attr_name, *args)

    return functools.reduce(_getattr, attr.split("."), obj)


def _rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _layer_index_from_name(name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(name)
    return int(match.group(1)) if match else None


def _is_output_head_name(name: str) -> bool:
    return any(keyword in name for keyword in _HEAD_KEYWORDS)


def _has_nemotron_h_blocks(model: nn.Module) -> bool:
    return any(
        hasattr(module, "mixer") and hasattr(module, "norm") and hasattr(module, "block_type")
        for module in model.modules()
    )


def _is_plain_linear(module: object) -> bool:
    return isinstance(module, nn.Linear)


def is_nemotron_h_attention_block(module: nn.Module) -> bool:
    mixer = getattr(module, "mixer", None)
    return (
        getattr(module, "block_type", None) == "attention"
        and mixer is not None
        and isinstance(getattr(mixer, "q_proj", None), nn.Linear)
        and isinstance(getattr(mixer, "k_proj", None), nn.Linear)
        and isinstance(getattr(mixer, "v_proj", None), nn.Linear)
        and isinstance(getattr(mixer, "o_proj", None), nn.Linear)
    )


def use_nemotron_h_fused_attention(default: bool = True) -> bool:
    return _env_bool("LBT_NEMOTRON_H_FP4_FUSED_ATTENTION", default)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NemotronHFusedAttentionWrapper(nn.Module):
    """Nemotron-H attention shell around a fused FP4 QKV/WO implementation."""

    def __init__(
        self,
        orig_attention: nn.Module,
        fused_attention: nn.Module,
        *,
        use_direct_wo_layout: Callable[[], bool] | None = None,
    ):
        super().__init__()
        self.fused = fused_attention
        self.num_heads = orig_attention.num_heads
        self.num_key_value_heads = orig_attention.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = orig_attention.head_dim
        self.attention_dropout = orig_attention.attention_dropout
        self.layer_idx = getattr(orig_attention, "layer_idx", None)
        self._use_direct_wo_layout = use_direct_wo_layout or (lambda: False)
        forward_wo_parameters = inspect.signature(
            fused_attention.forward_wo
        ).parameters.values()
        self._forward_wo_accepts_cde_emit = any(
            parameter.name == "cde_emit"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in forward_wo_parameters
        )

    def _forward_wo(
        self,
        attn_output: torch.Tensor,
        *,
        residual: torch.Tensor | None,
        cde_emit: bool,
    ):
        kwargs = {"residual": residual}
        if cde_emit:
            if not self._forward_wo_accepts_cde_emit:
                raise RuntimeError(
                    f"{type(self.fused).__name__}.forward_wo does not support "
                    "Nemotron inter-layer CDE emission"
                )
            kwargs["cde_emit"] = True
        return self.fused.forward_wo(attn_output, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        cde_row_rms_partial: torch.Tensor | None = None,
        cde_emit: bool = False,
        **kwargs,
    ):
        del position_ids, output_attentions, use_cache, cache_position, kwargs
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.fused.forward_qkv(
            hidden_states,
            freqs_cis=None,
            cde_row_rms_partial=cde_row_rms_partial,
        )
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        use_native_gqa = (
            os.environ.get("USE_FP4_TK_NATIVE_GQA", "1").strip().lower()
            in {"1", "true", "yes", "on"}
            and self.num_key_value_groups > 1
        )
        if not use_native_gqa:
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = causal_mask is None and q_len > 1
        dropout_p = self.attention_dropout if self.training else 0.0
        try:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=use_native_gqa,
            )
        except TypeError:
            if use_native_gqa:
                key_states = _repeat_kv(key_states, self.num_key_value_groups)
                value_states = _repeat_kv(value_states, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        if self._use_direct_wo_layout():
            out = self._forward_wo(
                attn_output,
                residual=residual,
                cde_emit=cde_emit,
            )
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
            out = self._forward_wo(
                attn_output,
                residual=residual,
                cde_emit=cde_emit,
            )

        return out, None, past_key_value

    def invalidate_weight_cache(self):
        invalidate = getattr(self.fused, "invalidate_weight_cache", None)
        if callable(invalidate):
            invalidate()

    def init_weights(self, init_std: float = 0.02):
        init_weights = getattr(self.fused, "init_weights", None)
        if callable(init_weights):
            init_weights(init_std)


def replace_nemotron_h_projection_linears(
    model: nn.Module,
    *,
    make_linear: Callable[[nn.Linear, str, str], nn.Module],
    backend_for_layer: Callable[[int | None], str],
    backend_for_projection: Callable[[int | None, str, str], str] | None = None,
    tail_bf16_names: set[str] | None = None,
    final_bf16_layer_indices: set[int] | None = None,
    label: str,
) -> dict[str, int]:
    """Replace exact Nemotron-H attention/Mamba projection linears.

    Environment controls:
      LBT_NEMOTRON_H_FP4_ATTENTION_PROJ  default on
      LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ   default on
      LBT_NEMOTRON_H_FP4_MAMBA_OUT_PROJ  default off, keeps native fused scan+outproj
      LBT_NEMOTRON_H_FP4_OUTPUT_HEAD     default off
    """

    counts = {
        "attention": 0,
        "mamba_in": 0,
        "mamba_out": 0,
        "head": 0,
    }
    if not _has_nemotron_h_blocks(model):
        return counts

    tail_bf16_names = tail_bf16_names or set()
    final_bf16_layer_indices = final_bf16_layer_indices or set()
    use_attention = _env_bool("LBT_NEMOTRON_H_FP4_ATTENTION_PROJ", True)
    use_mamba_in = _env_bool("LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ", True)
    use_mamba_out = nemotron_h_fp4_mamba_out_enabled(False)
    use_head = nemotron_h_fp4_output_head_enabled(False)

    def projection_backend(layer_idx: int | None, kind: str, name: str) -> str:
        if backend_for_projection is not None:
            return backend_for_projection(layer_idx, kind, name)
        return backend_for_layer(layer_idx)

    replacements: list[tuple[str, nn.Linear, str, str]] = []

    for block_name, block in model.named_modules():
        block_type = getattr(block, "block_type", None)
        mixer = getattr(block, "mixer", None)
        if mixer is None:
            continue
        layer_idx = _layer_index_from_name(block_name)
        if layer_idx in final_bf16_layer_indices:
            logger.info(
                "  %s KEEP BF16 NEMOTRON-H PROJECTIONS: %s final-layer ablation",
                label,
                block_name,
            )
            continue
        if block_type == "attention" and use_attention:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                child = getattr(mixer, attr, None)
                name = f"{block_name}.mixer.{attr}"
                if not _is_plain_linear(child):
                    continue
                if name in tail_bf16_names:
                    logger.info("  %s KEEP BF16 NEMOTRON-H ATTN PROJ: %s tail", label, name)
                    continue
                replacements.append(
                    (name, child, "attention", projection_backend(layer_idx, "attention", name))
                )
        elif block_type == "mamba":
            if use_mamba_in:
                child = getattr(mixer, "in_proj", None)
                name = f"{block_name}.mixer.in_proj"
                if _is_plain_linear(child) and name not in tail_bf16_names:
                    replacements.append(
                        (name, child, "mamba_in", projection_backend(layer_idx, "mamba_in", name))
                    )
                elif name in tail_bf16_names:
                    logger.info("  %s KEEP BF16 NEMOTRON-H MAMBA IN: %s tail", label, name)
            if use_mamba_out:
                child = getattr(mixer, "out_proj", None)
                name = f"{block_name}.mixer.out_proj"
                if _is_plain_linear(child) and name not in tail_bf16_names:
                    replacements.append(
                        (name, child, "mamba_out", projection_backend(layer_idx, "mamba_out", name))
                    )
                elif name in tail_bf16_names:
                    logger.info("  %s KEEP BF16 NEMOTRON-H MAMBA OUT: %s tail", label, name)

    if use_head:
        for name, module in model.named_modules():
            if _is_output_head_name(name) and _is_plain_linear(module):
                if name in tail_bf16_names:
                    logger.info("  %s KEEP BF16 NEMOTRON-H HEAD: %s tail", label, name)
                    continue
                replacements.append((name, module, "head", projection_backend(None, "head", name)))

    for name, module, kind, backend in replacements:
        new_module = make_linear(module, name, backend)
        setattr(new_module, "_lbt_debug_name", name)
        _rsetattr(model, name, new_module)
        counts[kind] += 1
        logger.info("  %s NEMOTRON-H %s: %s -> %s", label, kind, name, backend)

    if counts["mamba_out"]:
        logger.info(
            "  %s NEMOTRON-H MAMBA OUT_PROJ FP4 enabled: gated RMS producer "
            "dispatch is controlled by "
            "LBT_NEMOTRON_H_FUSED_MAMBA_GATED_OUT_PROJ.",
            label,
        )

    if any(counts.values()):
        logger.info(
            "%s Nemotron-H projection conversion: attention=%d mamba_in=%d "
            "mamba_out=%d head=%d",
            label,
            counts["attention"],
            counts["mamba_in"],
            counts["mamba_out"],
            counts["head"],
        )
    return counts
