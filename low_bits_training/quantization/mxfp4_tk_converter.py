"""Benchmark-only MXFP4 TK converters.

Two modes:
- backend: replace individual linear layers with MXFP4LinearTK
- fused: replace attention+norm and FFN+norm with dedicated MXFP4 fused wrappers
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import sys
import time
import types
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb as te_apply_rotary_pos_emb
except ImportError:
    te_apply_rotary_pos_emb = None

from .float32_linear import Float32Linear
from .nemotron_h_projection_policy import (
    NemotronHFusedAttentionWrapper,
    is_nemotron_h_attention_block,
    nemotron_h_fp4_output_head_enabled,
    replace_nemotron_h_projection_linears,
    use_nemotron_h_fused_attention,
)
from .mxfp4_fused_linear import (
    FusedDeepSeekMLAProjMXFP4_TK,
    FusedDeepSeekMLAMXFP4_TK,
    ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK,
    FusedAttentionMXFP4_TK,
    FusedFeedForwardMXFP4_TK,
    FusedFeedForwardNoNormMXFP4_TK,
    FusedSquaredReLUFeedForwardMXFP4_TK,
    MXFP4GroupedExpertsTK,
    MXFP4LinearTK,
    MXFP4RMSNormLinearTK,
    _as_contiguous_bf16,
    _mxfp4_bool_env,
    _mxfp4_data_sr_for_role,
    _mxfp4_int_env,
    _mxfp4_scale_sr_for_role,
    _mxfp4_attn_ffn_residual_overlap_safe,
    _mxfp4_qkv_bwd_state_slots,
    _mxfp4_rht_axes,
    _mxfp4_rht_block_size,
    _mxfp4_rht_for_role,
    _mxfp4_rht_random_sign_mask,
    _mxfp4_stage_begin,
    _mxfp4_stage_end,
    _validate_mxfp4_localcta_dgrad_contract,
    mxfp4_dgrad_route_identity,
    mxfp4_split3_qkv_onepass_config_idx,
    use_mxfp4_bwd_state_cache,
    use_mxfp4_bwd_wgrad_overlap,
    use_mxfp4_deepseek_mla_rope_epilogue,
    use_mxfp4_deepseek_mla_padded_wq_wkva_param,
    use_mxfp4_fused_sqrelu_deriv_quant,
    use_mxfp4_fused_sqrelu_quant,
    use_mxfp4_fused_silu_ffn_quant,
    use_mxfp4_fused_silu_ffn_quant_data_sr,
    use_mxfp4_fused_silu_ffn_quant_rht,
    use_mxfp4_fused_silu_ffn_quant_scale_sr,
    use_mxfp4_ffn_wgrad_overlap,
    use_mxfp4_ffn_w13_wgrad_overlap,
    use_mxfp4_ffn_w2_wgrad_overlap,
    use_mxfp4_linear_residual_config,
    use_mxfp4_localcta_dgrad,
    use_mxfp4_qkv_bf16_wgrad,
    use_mxfp4_qkv_fwd_weight_quant_overlap,
    use_mxfp4_qkv_wgrad_overlap,
    use_mxfp4_qkv_wgrad_wait_before_rmsnorm,
    use_mxfp4_qkv_wgrad_wait_before_rmsnorm_dgamma,
    use_mxfp4_qkv_direct_outputs,
    use_mxfp4_generic_qkv_rope_epilogue,
    use_mxfp4_qkv_rope_epilogue,
    use_mxfp4_residual_fusion,
    use_mxfp4_residual_fusion_attn,
    use_mxfp4_residual_fusion_ffn,
    use_mxfp4_simple_sqrelu_fused_w2,
    use_mxfp4_sqrelu_fused_rms_w1,
    use_mxfp4_sqrelu_deriv_gemm_epilogue,
    use_mxfp4_sqrelu_w2_wgrad_after_dgrad_overlap,
    use_mxfp4_sqrelu_w2_wgrad_overlap,
    use_mxfp4_split2_ffn_onepass_dgrad,
    use_mxfp4_split2_persistent_grad_sr,
    use_mxfp4_split2_ffn_producer_split,
    use_mxfp4_split2_ffn_row_overlap,
    use_mxfp4_split3_qkv_onepass_dgrad,
    use_mxfp4_stage_timing,
    use_mxfp4_stage_timing_sync,
    use_mxfp4_wo_attn_layout,
    use_mxfp4_wo_nhsd_quant,
)
from .mxfp4_backend import (
    mxfp4_backend_version,
    mxfp4_fused_rmsnorm_to_bf16,
    mxfp4_rope_live_head_dim_available,
)


_MXFP4_USE_TE_FUSED_ROPE = os.environ.get("MXFP4_USE_TE_FUSED_ROPE", "1") == "1"
_MXFP4_TE_ROPE_CACHE = {}
_MXFP4_TE_ROPE_FALLBACK_WARNED = False
_ATTN_ENABLE_GQA_SUPPORT_CACHE: dict[type, bool] = {}


def _log_mxfp4_rht_route_once() -> None:
    if getattr(_log_mxfp4_rht_route_once, "_logged", False):
        return
    setattr(_log_mxfp4_rht_route_once, "_logged", True)
    if not (
        _mxfp4_rht_for_role("activation")
        or _mxfp4_rht_for_role("grad")
        or _mxfp4_rht_for_role("weight")
    ):
        return
    logger.info(
        "  MXFP4 RHT ROUTE: te_style=%s axes=%s block=%s random_sign=%s act=%s grad=%s weight=%s",
        _mxfp4_bool_env("MXFP4_RHT_TE_STYLE", False),
        _mxfp4_rht_axes(),
        _mxfp4_rht_block_size(),
        _mxfp4_rht_random_sign_mask(),
        _mxfp4_rht_for_role("activation"),
        _mxfp4_rht_for_role("grad"),
        _mxfp4_rht_for_role("weight"),
    )


def _log_mxfp4_highwater_route_once() -> None:
    if getattr(_log_mxfp4_highwater_route_once, "_logged", False):
        return
    setattr(_log_mxfp4_highwater_route_once, "_logged", True)
    split2_row_overlap_rht_gate = _mxfp4_bool_env("MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT", True)
    split2_row_overlap_rht_effective = (
        use_mxfp4_split2_ffn_row_overlap()
        and split2_row_overlap_rht_gate
        and _mxfp4_rht_for_role("grad")
        and not _mxfp4_data_sr_for_role("grad")
        and not _mxfp4_scale_sr_for_role("grad")
    )
    logger.info(
        "  MXFP4 HIGHWATER ROUTE: "
        "rht(act=%s grad=%s weight=%s axes=%s block=%s sign=%s) "
        "sr(act=%s grad=%s weight=%s scale_act=%s scale_grad=%s scale_weight=%s) "
        "rht_fused(rmsnorm=%s split2_row_overlap_gate=%s split2_row_overlap_effective=%s "
        "fused_silu_deriv=%s fused_silu_ffn=%s fused_silu_rht=%s fused_silu_data_sr=%s fused_silu_scale_sr=%s) "
        "residual(requested=%s attn=%s ffn=%s overlap_safe=%s fallback=%s) "
        "qkv(rope_epilogue=%s generic_rope=%s packed_rope_h64=%s packed_rope_h128=%s direct_outputs=%s rms_quant=%s onepass_dgrad=%s onepass_cfg=%s slots=%s bf16_wgrad=%s fwd_wq_overlap=%s wgrad_overlap=%s wait_rms=%s wait_dgamma=%s combined_bwd_env=%s stage_copy_env=%s stage_copy_mask=%s) "
        "wo(attn_layout=%s nhsd_quant=%s) "
        "ffn(onepass_dgrad=%s row_overlap=%s persistent_grad_sr=%s producer_split=%s wgrad_overlap=%s w2_wgrad_overlap=%s w13_wgrad_overlap=%s global_wgrad_overlap=%s "
        "bwd_state_cache=%s linear_residual_config=%s sqrelu_fused_rms_w1=%s sqrelu_fused_w2=%s sqrelu_w2_wgrad_overlap=%s sqrelu_w2_wgrad_after_dgrad_overlap=%s sqrelu_deriv_gemm_epilogue=%s sqrelu_quant=%s sqrelu_tma=%s sqrelu_deriv_quant=%s stable_wgrad_config=%s) "
        "env(CUDA_DEVICE_MAX_CONNECTIONS=%s FP4_MATMUL_ROOT=%s FP4_MXFP4_ROOT=%s FP4_MATMUL_GEMM_ROOT=%s)",
        _mxfp4_rht_for_role("activation"),
        _mxfp4_rht_for_role("grad"),
        _mxfp4_rht_for_role("weight"),
        _mxfp4_rht_axes(),
        _mxfp4_rht_block_size(),
        _mxfp4_rht_random_sign_mask(),
        _mxfp4_data_sr_for_role("activation"),
        _mxfp4_data_sr_for_role("grad"),
        _mxfp4_data_sr_for_role("weight"),
        _mxfp4_scale_sr_for_role("activation"),
        _mxfp4_scale_sr_for_role("grad"),
        _mxfp4_scale_sr_for_role("weight"),
        _mxfp4_bool_env("MXFP4_USE_FUSED_RMSNORM_QUANT_RHT", True),
        split2_row_overlap_rht_gate,
        split2_row_overlap_rht_effective,
        os.environ.get("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT", "0") == "1",
        use_mxfp4_fused_silu_ffn_quant(),
        use_mxfp4_fused_silu_ffn_quant_rht(),
        use_mxfp4_fused_silu_ffn_quant_data_sr(),
        use_mxfp4_fused_silu_ffn_quant_scale_sr(),
        use_mxfp4_residual_fusion(),
        use_mxfp4_residual_fusion_attn(),
        use_mxfp4_residual_fusion_ffn(),
        _mxfp4_attn_ffn_residual_overlap_safe(),
        os.environ.get("MXFP4_UNSAFE_RESIDUAL_FALLBACK", "prefer_ffn"),
        use_mxfp4_qkv_rope_epilogue(),
        use_mxfp4_generic_qkv_rope_epilogue(),
        mxfp4_rope_live_head_dim_available(64),
        mxfp4_rope_live_head_dim_available(128),
        use_mxfp4_qkv_direct_outputs(),
        _mxfp4_bool_env("MXFP4_USE_QKV_RMSNORM_QUANT_FUSION", True),
        use_mxfp4_split3_qkv_onepass_dgrad(),
        mxfp4_split3_qkv_onepass_config_idx(),
        _mxfp4_qkv_bwd_state_slots(),
        use_mxfp4_qkv_bf16_wgrad(),
        use_mxfp4_qkv_fwd_weight_quant_overlap(),
        use_mxfp4_qkv_wgrad_overlap(),
        use_mxfp4_qkv_wgrad_wait_before_rmsnorm(),
        use_mxfp4_qkv_wgrad_wait_before_rmsnorm_dgamma(),
        os.environ.get("MXFP4_USE_QKV_COMBINED_BWD", "<auto>"),
        os.environ.get("MXFP4_USE_SPLIT3_QKV_STAGE_COPY", "<auto>"),
        os.environ.get("MXFP4_SPLIT3_QKV_STAGE_COPY_MASK", "qkv"),
        use_mxfp4_wo_attn_layout(),
        use_mxfp4_wo_nhsd_quant(),
        use_mxfp4_split2_ffn_onepass_dgrad(),
        use_mxfp4_split2_ffn_row_overlap(),
        use_mxfp4_split2_persistent_grad_sr(),
        use_mxfp4_split2_ffn_producer_split(),
        use_mxfp4_ffn_wgrad_overlap(),
        use_mxfp4_ffn_w2_wgrad_overlap(),
        use_mxfp4_ffn_w13_wgrad_overlap(),
        use_mxfp4_bwd_wgrad_overlap(),
        use_mxfp4_bwd_state_cache(),
        use_mxfp4_linear_residual_config(),
        use_mxfp4_sqrelu_fused_rms_w1(),
        use_mxfp4_simple_sqrelu_fused_w2(),
        use_mxfp4_sqrelu_w2_wgrad_overlap(),
        use_mxfp4_sqrelu_w2_wgrad_after_dgrad_overlap(),
        use_mxfp4_sqrelu_deriv_gemm_epilogue(),
        use_mxfp4_fused_sqrelu_quant(),
        os.environ.get("MXFP4_USE_TMA_SQRELU_QUANT", "0") == "1",
        use_mxfp4_fused_sqrelu_deriv_quant(),
        18,
        os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "<unset>"),
        os.environ.get("FP4_MATMUL_ROOT", "<unset>"),
        os.environ.get("FP4_MXFP4_ROOT", os.environ.get("FP4_MATMUL_QUANT_ROOT", "<unset>")),
        os.environ.get("FP4_MATMUL_GEMM_ROOT", "<unset>"),
    )


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr_name):
        return getattr(obj, attr_name, *args)
    return functools.reduce(_getattr, attr.split('.'), obj)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def _get_te_rope_freqs(freqs_cis: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(freqs_cis):
        raise RuntimeError(f"expected complex freqs_cis, got {freqs_cis.dtype}")
    key = (
        freqs_cis.data_ptr(),
        tuple(freqs_cis.shape),
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _MXFP4_TE_ROPE_CACHE.get(key)
    if cached is None:
        if len(_MXFP4_TE_ROPE_CACHE) >= 8:
            _MXFP4_TE_ROPE_CACHE.clear()
        angles = torch.angle(freqs_cis)
        cached = (
            torch.stack((angles, angles), dim=-1)
            .flatten(-2)
            .view(freqs_cis.shape[0], 1, 1, -1)
            .contiguous()
        )
        _MXFP4_TE_ROPE_CACHE[key] = cached
    return cached


def _apply_rotary_emb_fast(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _MXFP4_TE_ROPE_FALLBACK_WARNED
    from torchtitan.models.llama3.model.model import apply_rotary_emb

    if (
        not _MXFP4_USE_TE_FUSED_ROPE
        or te_apply_rotary_pos_emb is None
        or not xq.is_cuda
        or not xk.is_cuda
        or not freqs_cis.is_cuda
    ):
        return apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    try:
        rope_freqs = _get_te_rope_freqs(freqs_cis)
        return (
            te_apply_rotary_pos_emb(
                xq,
                rope_freqs,
                tensor_format="bshd",
                fused=True,
                interleaved=True,
            ),
            te_apply_rotary_pos_emb(
                xk,
                rope_freqs,
                tensor_format="bshd",
                fused=True,
                interleaved=True,
            ),
        )
    except RuntimeError as exc:
        if not _MXFP4_TE_ROPE_FALLBACK_WARNED:
            print(
                f"[MXFP4] fused TE RoPE unavailable, falling back to torchtitan RoPE: {exc}",
                file=sys.stderr,
                flush=True,
            )
            _MXFP4_TE_ROPE_FALLBACK_WARNED = True
        return apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)


def _attention_accepts_enable_gqa(module: nn.Module) -> bool:
    """Handle older Torchtitan attention wrappers during benchmark profiling."""
    cls = type(module)
    cached = _ATTN_ENABLE_GQA_SUPPORT_CACHE.get(cls)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(module.forward)
    except (TypeError, ValueError):
        cached = False
    else:
        cached = "enable_gqa" in sig.parameters
    _ATTN_ENABLE_GQA_SUPPORT_CACHE[cls] = cached
    return cached


def _env_truthy_default(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _use_mxfp4_native_gqa() -> bool:
    value = os.environ.get("USE_MXFP4_TK_NATIVE_GQA")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return _env_truthy_default("USE_FP4_TK_NATIVE_GQA", True)


def _sdpa_attention(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    enable_gqa: bool,
) -> torch.Tensor:
    backends = getattr(module, "sdpa_backends", None)
    kwargs = {
        "dropout_p": 0.0,
        "is_causal": True,
        "enable_gqa": enable_gqa,
    }
    if backends:
        with sdpa_kernel(backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


class _NormIdentity(nn.Module):
    def __init__(self, dim, dtype=torch.bfloat16, trainable: bool = True):
        super().__init__()
        weight = torch.ones(dim, dtype=dtype)
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight, persistent=False)

    def forward(self, x):
        return x

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class _FusedAttentionWrapper(nn.Module):
    def __init__(self, orig_attention, fused_attn_fp4):
        super().__init__()
        self.fused = fused_attn_fp4
        self.n_heads = orig_attention.n_heads
        self.n_kv_heads = getattr(orig_attention, 'n_kv_heads', self.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = orig_attention.head_dim
        self.use_flex_attn = orig_attention.use_flex_attn
        self.inner_attention = orig_attention.inner_attention

    @staticmethod
    def _attach_attention_backward_timing(
        output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        debug_name: str,
    ) -> None:
        if not use_mxfp4_stage_timing():
            return
        active_step = os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip()
        step_filter = (
            os.environ.get("MXFP4_STAGE_TRACE_STEP", "").strip()
            or os.environ.get("TK_STAGE_TRACE_STEP", "").strip()
        )
        if step_filter and active_step != step_filter:
            return
        stage_filter = os.environ.get("MXFP4_STAGE_TRACE_STAGE_FILTER", "").strip()
        if stage_filter and stage_filter not in "attn_inner_bwd":
            return
        name_filter = os.environ.get("MXFP4_STAGE_TRACE_FILTER", "").strip()
        if name_filter and name_filter not in debug_name:
            return
        if not output.requires_grad:
            return
        inputs = [t for t in (q, k, v) if isinstance(t, torch.Tensor) and t.requires_grad]
        if not inputs:
            return

        state = {"start": None, "pending": len(inputs)}

        def _start_hook(grad):
            if state["start"] is None:
                if use_mxfp4_stage_timing_sync():
                    torch.cuda.synchronize()
                state["start"] = time.perf_counter()
                prefix = f"[MXFP4 TRACE step={active_step}]" if active_step else "[MXFP4 TRACE]"
                print(
                    f"{prefix} attn_inner_bwd start {debug_name}",
                    file=sys.stderr,
                    flush=True,
                )
            return grad

        def _end_hook(grad):
            pending = state["pending"] - 1
            state["pending"] = pending
            if pending == 0 and state["start"] is not None:
                if use_mxfp4_stage_timing_sync():
                    torch.cuda.synchronize()
                prefix = f"[MXFP4 TRACE step={active_step}]" if active_step else "[MXFP4 TRACE]"
                elapsed_ms = (time.perf_counter() - state["start"]) * 1000.0
                print(
                    f"{prefix} attn_inner_bwd end {debug_name} elapsed_ms={elapsed_ms:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
            return grad

        output.register_hook(_start_hook)
        for tensor in inputs:
            tensor.register_hook(_end_hook)

    def _forward_impl(
        self,
        x,
        freqs_cis,
        attention_masks,
        residual=None,
        h_gamma=None,
        input_h_carrier=None,
        input_cde_partial=None,
    ):
        from torchtitan.models.llama3.model.model import repeat_kv
        from torch.nn.attention.flex_attention import BlockMask

        bs, seqlen, _ = x.shape
        debug_name = getattr(self.fused, "_lbt_debug_name", self.__class__.__name__)
        wrap_start = _mxfp4_stage_begin("attn_wrap_fwd", debug_name)
        xq, xk, xv = self.fused.forward_qkv(
            x,
            freqs_cis=freqs_cis,
            h_carrier=input_h_carrier,
            cde_row_rms_partial=input_cde_partial,
        )
        qkv_ready = _mxfp4_stage_begin("attn_rope_repeat_fwd", debug_name)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        if not getattr(self.fused, "_last_qkv_rope_applied", False):
            xq, xk = _apply_rotary_emb_fast(xq, xk, freqs_cis=freqs_cis)
        accepts_enable_gqa = (not self.use_flex_attn) and _attention_accepts_enable_gqa(self.inner_attention)
        use_sdpa_gqa = (not self.use_flex_attn) and _use_mxfp4_native_gqa() and self.n_rep > 1
        xq = xq.transpose(1, 2)
        if use_sdpa_gqa:
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
        else:
            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)
            xk = keys.transpose(1, 2)
            xv = values.transpose(1, 2)
        _mxfp4_stage_end("attn_rope_repeat_fwd", debug_name, qkv_ready)

        assert isinstance(attention_masks, BlockMask) or attention_masks is None, attention_masks
        inner_start = _mxfp4_stage_begin("attn_inner_fwd", debug_name)
        if self.use_flex_attn:
            assert isinstance(attention_masks, BlockMask), attention_masks
            output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
        else:
            assert attention_masks is None
            if accepts_enable_gqa:
                output = self.inner_attention(xq, xk, xv, enable_gqa=use_sdpa_gqa)
            elif use_sdpa_gqa:
                output = _sdpa_attention(self.inner_attention, xq, xk, xv, enable_gqa=True)
            else:
                output = self.inner_attention(xq, xk, xv)
        _mxfp4_stage_end("attn_inner_fwd", debug_name, inner_start)
        self._attach_attention_backward_timing(output, xq, xk, xv, debug_name)

        out_start = _mxfp4_stage_begin("attn_output_fwd", debug_name)
        if use_mxfp4_wo_attn_layout():
            result = self.fused.forward_wo(output, residual=residual, h_gamma=h_gamma)
        else:
            output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
            result = self.fused.forward_wo(output, residual=residual, h_gamma=h_gamma)
        _mxfp4_stage_end("attn_output_fwd", debug_name, out_start)
        _mxfp4_stage_end("attn_wrap_fwd", debug_name, wrap_start)
        return result

    def forward(self, x, freqs_cis, attention_masks):
        return self._forward_impl(x, freqs_cis, attention_masks, residual=None)

    def forward_with_residual(
        self,
        x,
        freqs_cis,
        attention_masks,
        residual,
        h_gamma=None,
        input_h_carrier=None,
    ):
        return self._forward_impl(
            x,
            freqs_cis,
            attention_masks,
            residual=residual,
            h_gamma=h_gamma,
            input_h_carrier=input_h_carrier,
        )

    def forward_with_cde_partial(
        self,
        x,
        freqs_cis,
        attention_masks,
        input_cde_partial=None,
    ):
        return self._forward_impl(
            x,
            freqs_cis,
            attention_masks,
            input_cde_partial=input_cde_partial,
        )

    def init_weights(self, init_std: float):
        self.fused.init_weights(init_std)


def _residual_fused_block_forward(
    self,
    x,
    freqs_cis: torch.Tensor,
    attention_masks,
    next_attention_gamma=None,
):
    use_h = os.environ.get("USE_FP4_CODA_H_TILE_RMS", "0") == "1"
    use_exact_cde = os.environ.get("USE_FP4_CODA_EXACT_CDE", "0") == "1"
    if use_h and use_exact_cde:
        raise RuntimeError("exact C/D/E and MX H tile carriers are mutually exclusive")
    if use_exact_cde:
        if not (
            hasattr(self.attention, "forward_with_cde_partial")
            and hasattr(self.feed_forward, "forward_with_residual")
        ):
            raise RuntimeError("MX exact C/D/E requires fused attention and FFN owners")
        if isinstance(x, tuple):
            if len(x) != 2:
                raise RuntimeError(
                    "MX exact C/D/E expected a (residual, row_rms_partial) carrier"
                )
            residual, input_cde_partial = x
        else:
            residual = x
            input_cde_partial = None
        h = residual + self.attention.forward_with_cde_partial(
            self.attention_norm(residual),
            freqs_cis,
            attention_masks,
            input_cde_partial=input_cde_partial,
        )
        return self.feed_forward.forward_with_residual(
            self.ffn_norm(h),
            residual=h,
            cde_emit=bool(getattr(self, "_fp4_cde_has_next", False)),
        )
    if (
        use_h
        and hasattr(self.attention, "forward_with_residual")
        and hasattr(self.feed_forward, "forward_with_h_carrier")
    ):
        if next_attention_gamma is None:
            owner_ref = getattr(self, "_fp4_coda_h_next_attention_owner", None)
            owner = owner_ref() if owner_ref is not None else None
            next_attention_gamma = getattr(owner, "norm_weight", None)
        input_carrier = x if isinstance(x, tuple) else None
        residual = input_carrier[0] if input_carrier is not None else x
        attention_input = residual if input_carrier is not None else self.attention_norm(x)
        carrier = self.attention.forward_with_residual(
            attention_input,
            freqs_cis,
            attention_masks,
            residual=residual,
            h_gamma=self.feed_forward.norm_weight,
            input_h_carrier=input_carrier,
        )
        return self.feed_forward.forward_with_h_carrier(
            carrier, next_attention_gamma=next_attention_gamma
        )
    if use_mxfp4_residual_fusion_attn() and hasattr(self.attention, "forward_with_residual"):
        h = self.attention.forward_with_residual(
            self.attention_norm(x),
            freqs_cis,
            attention_masks,
            residual=x,
        )
    else:
        h = x + self.attention(self.attention_norm(x), freqs_cis, attention_masks)

    if use_mxfp4_residual_fusion_ffn() and hasattr(self.feed_forward, "forward_with_residual"):
        return self.feed_forward.forward_with_residual(self.ffn_norm(h), residual=h)
    return h + self.feed_forward(self.ffn_norm(h))


def _nemotron_h_mlp_residual_fused_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    residual = hidden_states
    hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
    if getattr(self, "residual_in_fp32", False):
        residual = residual.to(torch.float32)
    return self.mixer.forward_with_residual(hidden_states, residual=residual)


_HEAD_KEYWORDS = ["output", "lm_head"]


def _is_output_head_name(name: str) -> bool:
    return any(k in name for k in _HEAD_KEYWORDS)


def _use_mxfp4_tk_convert_output_head() -> bool:
    """Return whether the MXFP4-TK route may replace the output head.

    Keep this policy aligned with ``fp4_converter._use_fp4_convert_output_head``:
    the safe/default route leaves the output head as an ordinary BF16
    ``nn.Linear``.  ``USE_FP4_CONVERT_OUTPUT_HEAD=1`` is the explicit legacy
    opt-in for the Float32Linear replacement.
    """

    return os.environ.get("USE_FP4_CONVERT_OUTPUT_HEAD", "0") == "1"


def _require_mxfp4_tk_llama_bf16_output_head() -> bool:
    """Enable the production Llama-specific, exactly-one-head assertion."""

    return os.environ.get("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "0") == "1"


def _keep_mxfp4_tk_output_head_bf16(name: str, module: nn.Module) -> None:
    """Fail closed unless *module* is an ordinary BF16 ``nn.Linear``."""

    if type(module) is not nn.Linear:
        raise RuntimeError(
            "MXFP4-TK ordinary BF16 output-head contract requires exact "
            f"torch.nn.Linear at {name!r}, got {type(module).__qualname__}"
        )
    module.to(dtype=torch.bfloat16)
    if module.weight.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"MXFP4-TK output head {name!r} did not remain BF16: "
            f"weight dtype={module.weight.dtype}"
        )
    if module.bias is not None and module.bias.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"MXFP4-TK output head {name!r} did not retain a BF16 bias: "
            f"bias dtype={module.bias.dtype}"
        )
    logger.info("  MXFP4 KEEP ORDINARY BF16 HEAD: %s (exact nn.Linear)", name)


def _assert_mxfp4_tk_output_head_contract(model: nn.Module) -> None:
    """Recheck the safe head route after the converter has finished."""

    require_llama_head = _require_mxfp4_tk_llama_bf16_output_head()
    legacy_head_enabled = _use_mxfp4_tk_convert_output_head()
    nemotron_fp4_head_enabled = nemotron_h_fp4_output_head_enabled(False)
    if legacy_head_enabled or nemotron_fp4_head_enabled:
        if require_llama_head:
            raise RuntimeError(
                "MXFP4-TK production Llama BF16 head contract conflicts with "
                "an explicitly enabled low-precision/legacy output-head route"
            )
        return
    heads = []
    for name, module in model.named_modules():
        if not _is_output_head_name(name):
            continue
        if isinstance(getattr(module, "weight", None), nn.Parameter):
            heads.append((name, module))
            _keep_mxfp4_tk_output_head_bf16(name, module)
    if require_llama_head:
        names = [name for name, _ in heads]
        if names != ["output"]:
            raise RuntimeError(
                "MXFP4-TK production Llama head contract requires exactly one "
                f"root output module, found {names}"
            )
        logger.info(
            "MXFP4-TK PRODUCTION HEAD CONTRACT: exactly one root output; "
            "exact nn.Linear; converter-pre-master dtype=BF16"
        )


def _tail_bf16_linear_names(model: nn.Module) -> set[str]:
    count = int(os.environ.get("FP4_KEEP_TAIL_BF16_LINEAR_COUNT", "0") or 0)
    if count <= 0:
        return set()
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        or (_is_output_head_name(name) and isinstance(getattr(module, "weight", None), nn.Parameter))
    ]
    tail = set(names[-count:])
    logger.info("MXFP4 tail BF16 ablation: keeping last %d linear modules BF16: %s", count, sorted(tail))
    return tail


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def _layer_index_from_name(name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(name)
    return int(match.group(1)) if match else None


def _is_regular_transformer_block(module: nn.Module) -> bool:
    return hasattr(module, "attention") and hasattr(module, "feed_forward")


def _is_nemotron_h_block(module: nn.Module) -> bool:
    return hasattr(module, "mixer") and hasattr(module, "norm") and hasattr(module, "block_type")


def _is_nemotron_h_mlp_block(module: nn.Module) -> bool:
    mixer = getattr(module, "mixer", None)
    return (
        getattr(module, "block_type", None) == "mlp"
        and mixer is not None
        and hasattr(mixer, "up_proj")
        and isinstance(mixer.up_proj, nn.Linear)
        and hasattr(mixer, "down_proj")
        and isinstance(mixer.down_proj, nn.Linear)
    )


def _is_transformer_layer_module(module: nn.Module) -> bool:
    return _is_regular_transformer_block(module) or _is_nemotron_h_block(module)


def _is_ffn_layer_module(module: nn.Module) -> bool:
    return (
        (hasattr(module, "feed_forward") and hasattr(module, "ffn_norm"))
        or _is_nemotron_h_mlp_block(module)
    )


def _last_bf16_layer_indices(model: nn.Module) -> set[int]:
    count = int(os.environ.get("FP4_KEEP_LAST_N_LAYERS_BF16", "0") or 0)
    if count <= 0:
        return set()
    indices = []
    for name, module in model.named_modules():
        if not _is_transformer_layer_module(module):
            continue
        idx = _layer_index_from_name(name)
        if idx is not None:
            indices.append(idx)
    if not indices:
        logger.warning("FP4_KEEP_LAST_N_LAYERS_BF16=%d set, but no transformer layer indices were found", count)
        return set()
    last = max(indices)
    selected = set(range(max(0, last - count + 1), last + 1))
    logger.info("MXFP4 final-layer BF16 ablation: keeping transformer layer indices BF16: %s", sorted(selected))
    return selected


def _last_bf16_ffn_layer_indices(model: nn.Module) -> set[int]:
    count = int(os.environ.get("FP4_KEEP_LAST_N_FFNS_BF16", "0") or 0)
    if count <= 0:
        return set()
    indices = []
    for name, module in model.named_modules():
        if not _is_ffn_layer_module(module):
            continue
        idx = _layer_index_from_name(name)
        if idx is not None:
            indices.append(idx)
    if not indices:
        logger.warning("FP4_KEEP_LAST_N_FFNS_BF16=%d set, but no FFN layer indices were found", count)
        return set()
    last = max(indices)
    selected = set(range(max(0, last - count + 1), last + 1))
    logger.info("MXFP4 final-FFN BF16 ablation: keeping FFN layer indices BF16: %s", sorted(selected))
    return selected


def _is_router_gate_name(name: str) -> bool:
    return name.endswith("router.gate") or ".router.gate" in name


def _is_llama_stacked_qkv_attention(attn: nn.Module) -> bool:
    return (
        hasattr(attn, "wq")
        and hasattr(attn, "wk")
        and hasattr(attn, "wv")
        and hasattr(attn, "wo")
        and isinstance(attn.wq, nn.Linear)
        and isinstance(attn.wk, nn.Linear)
        and isinstance(attn.wv, nn.Linear)
        and isinstance(attn.wo, nn.Linear)
        and hasattr(attn, "head_dim")
    )


def _is_deepseek_mla_attention(attn: nn.Module) -> bool:
    return (
        hasattr(attn, "wkv_a")
        and hasattr(attn, "wkv_b")
        and hasattr(attn, "kv_norm")
        and hasattr(attn, "wo")
    )


def _use_deepseek_mxfp4_grouped_experts() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_EXPERTS", "1") != "0"


def _deepseek_mxfp4_fused_mla_mode() -> str:
    mode = os.environ.get("MXFP4_DEEPSEEK_FUSED_MLA", "proj").strip().lower()
    if mode in ("0", "false", "off", "no", "none", ""):
        return "0"
    if mode in ("proj", "linear", "wq_wkv_a", "qkv"):
        return "proj"
    if mode in ("1", "true", "on", "yes", "rms", "rmsnorm"):
        return "rms"
    logger.warning("Unknown MXFP4_DEEPSEEK_FUSED_MLA=%r; disabling MLA fusion", mode)
    return "0"


def _use_deepseek_mxfp4_fused_mla() -> bool:
    return _deepseek_mxfp4_fused_mla_mode() != "0"


def _use_deepseek_mxfp4_rms_fused_mla() -> bool:
    return _deepseek_mxfp4_fused_mla_mode() == "rms"


def _deepseek_mxfp4_mla_bf16_norms() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_MLA_BF16_NORMS", "1") != "0"


def _deepseek_mxfp4_bf16_norms() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_BF16_NORMS", "0") == "1"


def _deepseek_mxfp4_attention_bf16() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_ATTENTION_BF16", "0") == "1"


def _deepseek_mxfp4_precast_bf16() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_PRECAST_BF16", "1") != "0"


def _deepseek_mxfp4_cast_model_bf16(model: nn.Module) -> None:
    # Mirror the explicit bfloat16 converter without touching complex RoPE buffers.
    for attr in ("layers", "tok_embeddings", "norm", "output"):
        module = getattr(model, attr, None)
        if module is not None:
            module.to(dtype=torch.bfloat16)


def _use_deepseek_mxfp4_shared_fused_moe() -> bool:
    default = "1" if mxfp4_backend_version().strip().lower() == "v4" else "0"
    return os.environ.get("MXFP4_DEEPSEEK_SHARED_FUSED_MOE", default) == "1"


def _deepseek_mxfp4_fused_moe_rmsnorm() -> bool:
    default = "1" if _use_deepseek_mxfp4_shared_fused_moe() else "0"
    return os.environ.get("MXFP4_DEEPSEEK_FUSED_MOE_RMSNORM", default) == "1"


def _deepseek_mxfp4_shared_experts_bf16() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_SHARED_EXPERTS_BF16", "0") == "1"


def _deepseek_mxfp4_fused_shared_experts() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_FUSED_SHARED_EXPERTS", "1") == "1"


def _deepseek_mxfp4_dense_ffn_bf16() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_DENSE_FFN_BF16", "0") == "1"


def _deepseek_mxfp4_head_bf16() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_HEAD_BF16", "0") == "1"


def _deepseek_mxfp4_head_mxfp4() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_HEAD_MXFP4", "0") == "1"


def _deepseek_mxfp4_router_mxfp4() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_ROUTER_MXFP4", "0") == "1"


def _is_deepseek_shared_expert_linear_name(name: str) -> bool:
    return ".moe.shared_experts." in name


def _is_deepseek_dense_ffn_linear_name(name: str) -> bool:
    return ".feed_forward." in name


def _is_deepseek_attention_linear_name(name: str) -> bool:
    return (
        ".attention." in name
        and any(
            token in name
            for token in (
                ".attention.wq",
                ".attention.wkv_a",
                ".attention.wkv_b",
                ".attention.wo",
            )
        )
    )


def _deepseek_mxfp4_rmsnorm_to_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp4_fused_rmsnorm_to_bf16(
        _as_contiguous_bf16(x),
        _as_contiguous_bf16(weight),
        float(eps),
    )


class MXFP4TKBackendConverter(ModelConverter):
    """Replace standard linears with the benchmark-only MXFP4 backend."""

    def __init__(self, job_config: JobConfig | None, parallel_dims: ParallelDims | None):
        self.job_config = job_config

    def convert(self, model: nn.Module):
        _log_mxfp4_highwater_route_once()
        _log_mxfp4_rht_route_once()

        modules_to_replace = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, module, _is_output_head_name(name)))

        for name, module, is_head in modules_to_replace:
            if is_head:
                if not _use_mxfp4_tk_convert_output_head():
                    _keep_mxfp4_tk_output_head_bf16(name, module)
                    continue
                new_layer = Float32Linear(module.in_features, module.out_features, bias=module.bias is not None)
                new_layer = new_layer.to(module.weight.device).to(module.weight.dtype)
                if module.weight.device.type != 'meta':
                    with torch.no_grad():
                        new_layer.weight.copy_(module.weight)
                        if module.bias is not None:
                            new_layer.bias.copy_(module.bias)
            else:
                new_layer = MXFP4LinearTK.from_linear(module)
            rsetattr(model, name, new_layer)

        logger.info(
            "MXFP4TKBackendConverter: body=MXFP4LinearTK output_head=%s",
            (
                "Float32Linear (explicit legacy opt-in)"
                if _use_mxfp4_tk_convert_output_head()
                else "ordinary BF16 nn.Linear"
            ),
        )
        _assert_mxfp4_tk_output_head_contract(model)

    def post_optimizer_hook(self, model):
        pass


class MXFP4TKFusedConverter(ModelConverter):
    """Replace attention/FFN blocks with benchmark-only MXFP4 fused wrappers."""

    def __init__(self, job_config: JobConfig | None, parallel_dims: ParallelDims | None):
        self.job_config = job_config

    def convert(self, model: nn.Module):
        _log_mxfp4_highwater_route_once()
        _log_mxfp4_rht_route_once()
        if use_mxfp4_localcta_dgrad():
            _validate_mxfp4_localcta_dgrad_contract()
            logger.info(
                "  MXFP4 DGRAD ROUTE: %s; forward/wgrad remain MXFP4+col-RHT",
                mxfp4_dgrad_route_identity(),
            )
        tail_bf16_names = _tail_bf16_linear_names(model)
        final_bf16_layer_indices = _last_bf16_layer_indices(model)
        final_bf16_ffn_layer_indices = _last_bf16_ffn_layer_indices(model)

        head_replacements = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _is_output_head_name(name):
                if nemotron_h_fp4_output_head_enabled(False):
                    continue
                if not _use_mxfp4_tk_convert_output_head() or name in tail_bf16_names:
                    _keep_mxfp4_tk_output_head_bf16(name, module)
                    continue
                head_replacements.append((name, module))
        for name, module in head_replacements:
            new_layer = Float32Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_layer = new_layer.to(module.weight.device).to(module.weight.dtype)
            if module.weight.device.type != 'meta':
                with torch.no_grad():
                    new_layer.weight.copy_(module.weight)
                    if module.bias is not None:
                        new_layer.bias.copy_(module.bias)
            rsetattr(model, name, new_layer)

        blocks_to_fuse_attn = []
        for block_name, block in model.named_modules():
            if hasattr(block, 'attention') and hasattr(block, 'attention_norm'):
                attn = block.attention
                norm = block.attention_norm
                if _is_llama_stacked_qkv_attention(attn):
                    layer_idx = _layer_index_from_name(block_name)
                    if layer_idx in final_bf16_layer_indices:
                        logger.info("  MXFP4 KEEP BF16 ATTN: %s.attention final-layer ablation", block_name)
                        continue
                    qkv_tail = {
                        f"{block_name}.attention.wq",
                        f"{block_name}.attention.wk",
                        f"{block_name}.attention.wv",
                    } & tail_bf16_names
                    if qkv_tail:
                        logger.info("  MXFP4 KEEP BF16 ATTN: %s.attention tail=%s", block_name, sorted(qkv_tail))
                        continue
                    wo_bf16 = f"{block_name}.attention.wo" in tail_bf16_names
                    blocks_to_fuse_attn.append((block_name, block, attn, norm, wo_bf16))
                elif _is_deepseek_mla_attention(attn):
                    logger.info(
                        "  MXFP4 SKIP MLA ATTN: %s.attention needs DeepSeek-specific wq/wkv_a/wkv_b fusion",
                        block_name,
                    )
        for block_name, block, attn, norm, wo_bf16 in blocks_to_fuse_attn:
            fused_attn = FusedAttentionMXFP4_TK.from_attention(attn, norm)
            if wo_bf16:
                fused_attn._force_wo_bf16 = True
            fused_attn._lbt_debug_name = f"{block_name}.attention"
            block.attention = _FusedAttentionWrapper(attn, fused_attn)
            dim = norm.weight.shape[0] if hasattr(norm, 'weight') else 0
            norm_dtype = norm.weight.dtype if hasattr(norm, 'weight') else torch.bfloat16
            block.attention_norm = _NormIdentity(dim, dtype=norm_dtype, trainable=False)
            logger.info(f"  MXFP4 FUSED ATTN: {block_name}.attention{' (BF16 wo)' if wo_bf16 else ''}")

        if use_nemotron_h_fused_attention():
            blocks_to_fuse_nemotron_attn = []
            for block_name, block in model.named_modules():
                if not is_nemotron_h_attention_block(block):
                    continue
                layer_idx = _layer_index_from_name(block_name)
                if layer_idx in final_bf16_layer_indices:
                    logger.info("  MXFP4 KEEP BF16 NEMOTRON-H ATTN: %s.mixer final-layer ablation", block_name)
                    continue
                qkv_tail = {
                    f"{block_name}.mixer.q_proj",
                    f"{block_name}.mixer.k_proj",
                    f"{block_name}.mixer.v_proj",
                } & tail_bf16_names
                if qkv_tail:
                    logger.info("  MXFP4 KEEP BF16 NEMOTRON-H ATTN: %s.mixer tail=%s", block_name, sorted(qkv_tail))
                    continue
                wo_bf16 = f"{block_name}.mixer.o_proj" in tail_bf16_names
                blocks_to_fuse_nemotron_attn.append((block_name, block, block.mixer, block.norm, wo_bf16))

            for block_name, block, attn, norm, wo_bf16 in blocks_to_fuse_nemotron_attn:
                fused_attn = FusedAttentionMXFP4_TK.from_attention(attn, norm)
                if wo_bf16:
                    fused_attn._force_wo_bf16 = True
                fused_attn._lbt_debug_name = f"{block_name}.mixer"
                block.mixer = NemotronHFusedAttentionWrapper(
                    attn,
                    fused_attn,
                    use_direct_wo_layout=use_mxfp4_wo_attn_layout,
                )
                dim = norm.weight.shape[0] if hasattr(norm, 'weight') else 0
                norm_dtype = norm.weight.dtype if hasattr(norm, 'weight') else torch.bfloat16
                block.norm = _NormIdentity(dim, dtype=norm_dtype, trainable=False)
                logger.info(
                    f"  MXFP4 FUSED NEMOTRON-H ATTN: {block_name}.mixer"
                    f"{' (BF16 o_proj)' if wo_bf16 else ''}"
                )

        blocks_to_fuse_ffn = []
        for block_name, block in model.named_modules():
            if hasattr(block, 'feed_forward') and hasattr(block, 'ffn_norm'):
                ffn = block.feed_forward
                norm = block.ffn_norm
                if hasattr(ffn, 'w1') and isinstance(ffn.w1, nn.Linear):
                    layer_idx = _layer_index_from_name(block_name)
                    if layer_idx in final_bf16_layer_indices:
                        logger.info("  MXFP4 KEEP BF16 FFN: %s.feed_forward final-layer ablation", block_name)
                        continue
                    if layer_idx in final_bf16_ffn_layer_indices:
                        logger.info("  MXFP4 KEEP BF16 FFN: %s.feed_forward final-FFN ablation", block_name)
                        continue
                    ffn_tail = {
                        f"{block_name}.feed_forward.w1",
                        f"{block_name}.feed_forward.w2",
                        f"{block_name}.feed_forward.w3",
                    } & tail_bf16_names
                    if ffn_tail:
                        logger.info("  MXFP4 KEEP BF16 FFN: %s.feed_forward tail=%s", block_name, sorted(ffn_tail))
                        continue
                    blocks_to_fuse_ffn.append(
                        (block_name, block, ffn, norm, "feed_forward", "ffn_norm", f"{block_name}.feed_forward")
                    )
        for block_name, block in model.named_modules():
            if not _is_nemotron_h_mlp_block(block):
                continue
            layer_idx = _layer_index_from_name(block_name)
            if layer_idx in final_bf16_layer_indices:
                logger.info("  MXFP4 KEEP BF16 NEMOTRON-H MLP: %s.mixer final-layer ablation", block_name)
                continue
            if layer_idx in final_bf16_ffn_layer_indices:
                logger.info("  MXFP4 KEEP BF16 NEMOTRON-H MLP: %s.mixer final-FFN ablation", block_name)
                continue
            ffn_tail = {
                f"{block_name}.mixer.up_proj",
                f"{block_name}.mixer.down_proj",
            } & tail_bf16_names
            if ffn_tail:
                logger.info("  MXFP4 KEEP BF16 NEMOTRON-H MLP: %s.mixer tail=%s", block_name, sorted(ffn_tail))
                continue
            blocks_to_fuse_ffn.append((block_name, block, block.mixer, block.norm, "mixer", "norm", f"{block_name}.mixer"))
        skip_fused_ffn = _mxfp4_bool_env("MXFP4_SKIP_FUSED_FFN", False)
        for block_name, block, ffn, norm, ffn_attr, norm_attr, debug_name in blocks_to_fuse_ffn:
            if skip_fused_ffn:
                for child_name in ("w1", "w2", "w3"):
                    child = getattr(ffn, child_name, None)
                    if isinstance(child, nn.Linear):
                        setattr(ffn, child_name, MXFP4LinearTK.from_linear(child))
                for child_name in ("up_proj", "down_proj"):
                    child = getattr(ffn, child_name, None)
                    if isinstance(child, nn.Linear):
                        setattr(ffn, child_name, MXFP4LinearTK.from_linear(child))
                logger.info("  MXFP4 UNFUSED FFN LINEARS: %s", debug_name)
                continue
            if hasattr(ffn, "w3"):
                ffn_cls = FusedFeedForwardMXFP4_TK
            elif os.environ.get("MXFP4_USE_EXPERIMENTAL_SQRELU_FFN", "0") == "1":
                ffn_cls = ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK
            else:
                ffn_cls = FusedSquaredReLUFeedForwardMXFP4_TK
            fused_ffn = ffn_cls.from_unfused(ffn, norm)
            fused_ffn._lbt_debug_name = debug_name
            setattr(block, ffn_attr, fused_ffn)
            dim = norm.weight.shape[0] if hasattr(norm, 'weight') else 0
            norm_dtype = norm.weight.dtype if hasattr(norm, 'weight') else torch.bfloat16
            setattr(block, norm_attr, _NormIdentity(dim, dtype=norm_dtype, trainable=False))
            logger.info(f"  MXFP4 FUSED FFN: {debug_name} -> {ffn_cls.__name__}")

        fused_mamba_norm_projections = []
        if (
            _mxfp4_bool_env("LBT_NEMOTRON_H_FUSED_MAMBA_RMS_IN_PROJ", True)
            and os.environ.get("LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ", "1") != "0"
        ):
            for block_name, block in model.named_modules():
                if getattr(block, "block_type", None) != "mamba":
                    continue
                layer_idx = _layer_index_from_name(block_name)
                if layer_idx in final_bf16_layer_indices:
                    continue
                in_proj = getattr(getattr(block, "mixer", None), "in_proj", None)
                norm = getattr(block, "norm", None)
                projection_name = f"{block_name}.mixer.in_proj"
                if (
                    not isinstance(in_proj, nn.Linear)
                    or in_proj.bias is not None
                    or in_proj.in_features != 4096
                    or in_proj.out_features != 18560
                    or projection_name in tail_bf16_names
                    or not isinstance(getattr(norm, "weight", None), nn.Parameter)
                ):
                    continue
                fused_mamba_norm_projections.append(
                    (block_name, block, in_proj, norm, projection_name)
                )

        for (
            block_name,
            block,
            in_proj,
            norm,
            projection_name,
        ) in fused_mamba_norm_projections:
            fused = MXFP4RMSNormLinearTK(
                in_proj.in_features,
                in_proj.out_features,
                eps=float(
                    getattr(
                        norm,
                        "variance_epsilon",
                        getattr(norm, "eps", 1e-5),
                    )
                ),
                device=in_proj.weight.device,
                dtype=in_proj.weight.dtype,
            )
            if in_proj.weight.device.type != "meta":
                with torch.no_grad():
                    fused.weight.copy_(in_proj.weight)
                    fused.norm_weight.copy_(norm.weight)
            fused._lbt_debug_name = projection_name
            block.mixer.in_proj = fused
            block.norm = _NormIdentity(
                in_proj.in_features,
                dtype=norm.weight.dtype,
                trainable=False,
            )
            logger.info(
                "  MXFP4 FUSED NEMOTRON-H MAMBA RMS+IN: %s",
                projection_name,
            )

        replace_nemotron_h_projection_linears(
            model,
            make_linear=lambda linear, name, backend: MXFP4LinearTK.from_linear(linear),
            backend_for_layer=lambda layer_idx: "mxfp4",
            tail_bf16_names=tail_bf16_names,
            final_bf16_layer_indices=final_bf16_layer_indices,
            label="MXFP4",
        )

        residual_attn_enabled = use_mxfp4_residual_fusion_attn()
        residual_ffn_enabled = use_mxfp4_residual_fusion_ffn()
        h_tile_enabled = os.environ.get("USE_FP4_CODA_H_TILE_RMS", "0") == "1"
        exact_cde_requested = os.environ.get("USE_FP4_CODA_EXACT_CDE", "0") == "1"
        if exact_cde_requested:
            if h_tile_enabled:
                raise RuntimeError("MX exact C/D/E and H tile carriers are mutually exclusive")
            if not residual_ffn_enabled:
                raise RuntimeError("MX exact C/D/E requires the W2 residual epilogue")
            if (
                _mxfp4_rht_for_role("activation")
                or _mxfp4_data_sr_for_role("activation")
                or _mxfp4_scale_sr_for_role("activation")
            ):
                raise RuntimeError("MX exact C/D/E does not support activation RHT/SR")
        if (
            residual_attn_enabled
            or residual_ffn_enabled
            or h_tile_enabled
            or exact_cde_requested
        ):
            logger.info(
                "  MXFP4 RESIDUAL FUSION ROUTE: attn=%s ffn=%s h_tile=%s exact_cde=%s",
                residual_attn_enabled,
                residual_ffn_enabled,
                h_tile_enabled,
                exact_cde_requested,
            )
            for block_name, block in model.named_modules():
                if (
                    hasattr(block, "attention")
                    and hasattr(block, "feed_forward")
                    and hasattr(block.attention, "forward_with_residual")
                    and hasattr(block.feed_forward, "forward_with_residual")
                ):
                    block.forward = types.MethodType(_residual_fused_block_forward, block)
                    logger.info(f"  MXFP4 RESIDUAL FUSED BLOCK: {block_name}")
                elif (
                    getattr(block, "block_type", None) == "mlp"
                    and hasattr(block, "mixer")
                    and hasattr(block.mixer, "forward_with_residual")
                ):
                    block.forward = types.MethodType(_nemotron_h_mlp_residual_fused_block_forward, block)
                    logger.info(f"  MXFP4 NEMOTRON-H MLP RESIDUAL FUSED BLOCK: {block_name}")
            if h_tile_enabled:
                layers = getattr(model, "layers", None)
                if layers is None:
                    raise RuntimeError("MX H requires a model.layers container")
                h_layers = list(layers.values()) if hasattr(layers, "values") else list(layers)
                if not h_layers:
                    raise RuntimeError("MX H found no model layers to wire")
                for index, layer in enumerate(h_layers):
                    if not (
                        hasattr(getattr(layer, "attention", None), "forward_with_residual")
                        and hasattr(getattr(layer, "feed_forward", None), "forward_with_h_carrier")
                    ):
                        raise RuntimeError(f"MX H layer {index} lacks a fused attention/FFN carrier owner")
                    owner_ref = None
                    if (
                        hasattr(getattr(layer, "feed_forward", None), "forward_with_h_carrier")
                        and index + 1 < len(h_layers)
                    ):
                        next_attention = getattr(h_layers[index + 1], "attention", None)
                        next_owner = getattr(next_attention, "fused", None)
                        if next_owner is None:
                            raise RuntimeError("MX H next-layer fused attention owner is unavailable")
                        owner_ref = weakref.ref(next_owner)
                    object.__setattr__(
                        layer, "_fp4_coda_h_next_attention_owner", owner_ref
                    )
                    object.__setattr__(
                        layer, "_fsdp_preserve_forward_input_dtypes", True
                    )
                logger.info("  MXFP4 H TILE CARRIER WIRED: %d layers", len(h_layers))
            if exact_cde_requested:
                layers = getattr(model, "layers", None)
                if layers is None:
                    raise RuntimeError("MX exact C/D/E requires a model.layers container")
                cde_layers = list(layers.values()) if hasattr(layers, "values") else list(layers)
                if not cde_layers:
                    raise RuntimeError("MX exact C/D/E found no model layers to wire")
                for index, layer in enumerate(cde_layers):
                    attention = getattr(layer, "attention", None)
                    feed_forward = getattr(layer, "feed_forward", None)
                    if not (
                        hasattr(attention, "forward_with_cde_partial")
                        and hasattr(feed_forward, "forward_with_residual")
                    ):
                        raise RuntimeError(
                            f"MX exact C/D/E layer {index} lacks fused attention/FFN owners"
                        )
                    if (
                        getattr(feed_forward, "dim", None) != 4096
                        or getattr(feed_forward, "hidden_dim", None) != 14336
                    ):
                        raise RuntimeError(
                            f"MX exact C/D/E layer {index} is not the production 4096x14336 FFN"
                        )
                    if index + 1 < len(cde_layers):
                        next_attention = getattr(cde_layers[index + 1], "attention", None)
                        next_owner = getattr(next_attention, "fused", None)
                        if next_owner is None:
                            raise RuntimeError(
                                "MX exact C/D/E next-layer fused attention owner is unavailable"
                            )
                        producer_eps = float(getattr(feed_forward, "epsilon"))
                        consumer_eps = float(getattr(next_owner, "epsilon"))
                        if producer_eps != consumer_eps:
                            raise RuntimeError(
                                "MX exact C/D/E requires matching producer/consumer epsilon, "
                                f"got {producer_eps} and {consumer_eps} at layer {index}"
                            )
                    object.__setattr__(
                        layer, "_fp4_cde_has_next", index + 1 < len(cde_layers)
                    )
                    object.__setattr__(
                        layer, "_fsdp_preserve_forward_input_dtypes", True
                    )
                logger.info(
                    "  MXFP4 EXACT C/D/E CARRIER WIRED: %d layers", len(cde_layers)
                )

        _assert_mxfp4_tk_output_head_contract(model)

    def post_optimizer_hook(self, model):
        pass


class MXFP4TKConverter(MXFP4TKFusedConverter):
    """Experimental registered converter for the custom MXFP4 fused path."""


class MXFP4TKDeepSeekConverter(ModelConverter):
    """DeepSeek-V3-safe MXFP4 route.

    This conservative route keeps router gates and output heads out of MXFP4,
    quantizes MLA/shared-expert linears individually, and reuses the existing
    fused MXFP4 FFN only for dense pre-MoE blocks. Grouped expert parameters
    need a dedicated MXFP4 grouped-MoE GEMM and are logged as TODO.
    """

    def __init__(self, job_config: JobConfig | None, parallel_dims: ParallelDims | None):
        self.job_config = job_config
        self.parallel_dims = parallel_dims

    def convert(self, model: nn.Module):
        _log_mxfp4_highwater_route_once()
        _log_mxfp4_rht_route_once()
        tp_enabled = bool(getattr(self.parallel_dims, "tp_enabled", False))
        if tp_enabled and "MXFP4_DEEPSEEK_GROUPED_BULK_COL_SLICE" not in os.environ:
            os.environ["MXFP4_DEEPSEEK_GROUPED_BULK_COL_SLICE"] = "0"
        if _deepseek_mxfp4_precast_bf16():
            _deepseek_mxfp4_cast_model_bf16(model)
            logger.info("  MXFP4 DEEPSEEK PRECAST BF16: layers/tok_embeddings/norm/output")

        dense_ffn_count = 0
        dense_ffn_bf16_count = 0
        linear_count = 0
        head_count = 0
        head_bf16_count = 0
        router_count = 0
        router_mxfp4_count = 0
        shared_expert_bf16_count = 0
        fused_shared_expert_count = 0
        mla_count = 0
        grouped_expert_count = 0
        grouped_expert_mxfp4_count = 0
        shared_fused_moe_count = 0

        fused_mla_mode = (
            "0"
            if tp_enabled or _deepseek_mxfp4_attention_bf16()
            else _deepseek_mxfp4_fused_mla_mode()
        )
        if _deepseek_mxfp4_attention_bf16():
            logger.info(
                "  MXFP4 DEEPSEEK KEEP ATTENTION BF16: MLA fusion and attention linear replacement disabled"
            )
        if tp_enabled:
            logger.info(
                "  MXFP4 DEEPSEEK TP COMPAT: leaving MLA/FFN/shared-MoE structure unfused for Tensor Parallel; grouped bulk col slicing=%s",
                os.environ.get("MXFP4_DEEPSEEK_GROUPED_BULK_COL_SLICE", "1"),
            )
        if fused_mla_mode != "0":
            blocks_to_fuse_mla = []
            for block_name, block in model.named_modules():
                if not (hasattr(block, "attention") and hasattr(block, "attention_norm")):
                    continue
                attn = block.attention
                if (
                    _is_deepseek_mla_attention(attn)
                    and getattr(attn, "q_lora_rank", 0) == 0
                    and hasattr(attn, "wq")
                    and isinstance(attn.wq, nn.Linear)
                    and isinstance(attn.wkv_a, nn.Linear)
                    and isinstance(attn.wkv_b, nn.Linear)
                    and isinstance(attn.wo, nn.Linear)
                ):
                    blocks_to_fuse_mla.append((block_name, block, attn, block.attention_norm))
            for block_name, block, attn, norm in blocks_to_fuse_mla:
                if fused_mla_mode == "rms":
                    fused_mla = FusedDeepSeekMLAMXFP4_TK.from_attention(attn, norm)
                else:
                    fused_mla = FusedDeepSeekMLAProjMXFP4_TK.from_attention(
                        attn,
                        force_bf16_norms=_deepseek_mxfp4_mla_bf16_norms(),
                    )
                fused_mla._lbt_debug_name = f"{block_name}.attention"
                block.attention = fused_mla
                if fused_mla_mode == "rms":
                    dim = norm.weight.shape[0] if hasattr(norm, "weight") else 0
                    norm_dtype = norm.weight.dtype if hasattr(norm, "weight") else torch.bfloat16
                    block.attention_norm = _NormIdentity(dim, dtype=norm_dtype, trainable=False)
                elif _deepseek_mxfp4_mla_bf16_norms():
                    block.attention_norm = norm.to(dtype=attn.wq.weight.dtype)
                mla_count += 1
                logger.info(
                    "  MXFP4 DEEPSEEK FUSED MLA[%s bf16_norms=%s rope_epilogue=%s padded_wq_wkva=%s]: %s.attention",
                    fused_mla_mode,
                    _deepseek_mxfp4_mla_bf16_norms() if fused_mla_mode == "proj" else False,
                    use_mxfp4_deepseek_mla_rope_epilogue() if fused_mla_mode == "proj" else False,
                    use_mxfp4_deepseek_mla_padded_wq_wkva_param(),
                    block_name,
                )

        if _deepseek_mxfp4_bf16_norms():
            norm_count = 0
            for module in model.modules():
                if isinstance(module, nn.RMSNorm):
                    module.to(dtype=torch.bfloat16)
                    norm_count += 1
            logger.info("  MXFP4 DEEPSEEK BF16 NORMS: %d RMSNorm modules", norm_count)

        blocks_to_fuse_ffn = []
        if not tp_enabled:
            for block_name, block in model.named_modules():
                if hasattr(block, "feed_forward") and hasattr(block, "ffn_norm"):
                    ffn = block.feed_forward
                    norm = block.ffn_norm
                    if hasattr(ffn, "w1") and isinstance(ffn.w1, nn.Linear):
                        if _deepseek_mxfp4_dense_ffn_bf16():
                            ffn.to(dtype=torch.bfloat16)
                            norm.to(dtype=torch.bfloat16)
                            dense_ffn_bf16_count += 1
                            logger.info("  MXFP4 DEEPSEEK KEEP DENSE FFN BF16: %s.feed_forward", block_name)
                        else:
                            blocks_to_fuse_ffn.append((block_name, block, ffn, norm))

        for block_name, block, ffn, norm in blocks_to_fuse_ffn:
            fused_ffn = FusedFeedForwardMXFP4_TK.from_unfused(ffn, norm)
            fused_ffn._lbt_debug_name = f"{block_name}.feed_forward"
            block.feed_forward = fused_ffn
            dim = norm.weight.shape[0] if hasattr(norm, "weight") else 0
            norm_dtype = norm.weight.dtype if hasattr(norm, "weight") else torch.bfloat16
            block.ffn_norm = _NormIdentity(dim, dtype=norm_dtype, trainable=False)
            dense_ffn_count += 1
            logger.info("  MXFP4 DEEPSEEK DENSE FFN: %s.feed_forward", block_name)

        if _deepseek_mxfp4_fused_shared_experts() and not tp_enabled:
            for name, module in model.named_modules():
                if module.__class__.__name__ != "MoE":
                    continue
                shared = getattr(module, "shared_experts", None)
                if shared is None:
                    continue
                if not (
                    hasattr(shared, "w1")
                    and hasattr(shared, "w2")
                    and hasattr(shared, "w3")
                    and isinstance(shared.w1, nn.Linear)
                    and isinstance(shared.w2, nn.Linear)
                    and isinstance(shared.w3, nn.Linear)
                ):
                    continue
                fused_shared = FusedFeedForwardNoNormMXFP4_TK.from_unfused(shared)
                fused_shared._lbt_debug_name = f"{name}.shared_experts"
                module.shared_experts = fused_shared
                fused_shared_expert_count += 1
                logger.info("  MXFP4 DEEPSEEK FUSED SHARED EXPERT: %s.shared_experts", name)

        modules_to_replace = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if _is_output_head_name(name):
                if _deepseek_mxfp4_head_mxfp4():
                    modules_to_replace.append((name, module, "mxfp4_head"))
                elif _deepseek_mxfp4_head_bf16():
                    module.to(dtype=torch.bfloat16)
                    head_bf16_count += 1
                    logger.info("  MXFP4 DEEPSEEK KEEP HEAD BF16: %s", name)
                else:
                    modules_to_replace.append((name, module, "head"))
            elif _is_router_gate_name(name):
                if _deepseek_mxfp4_router_mxfp4():
                    modules_to_replace.append((name, module, "mxfp4_router"))
                else:
                    router_count += 1
                    logger.info("  MXFP4 DEEPSEEK KEEP ROUTER BF16: %s", name)
            elif _deepseek_mxfp4_shared_experts_bf16() and _is_deepseek_shared_expert_linear_name(name):
                module.to(dtype=torch.bfloat16)
                shared_expert_bf16_count += 1
                logger.info("  MXFP4 DEEPSEEK KEEP SHARED EXPERT BF16: %s", name)
            elif _deepseek_mxfp4_dense_ffn_bf16() and _is_deepseek_dense_ffn_linear_name(name):
                module.to(dtype=torch.bfloat16)
                logger.info("  MXFP4 DEEPSEEK KEEP DENSE FFN LINEAR BF16: %s", name)
            elif _deepseek_mxfp4_attention_bf16() and _is_deepseek_attention_linear_name(name):
                module.to(dtype=torch.bfloat16)
                logger.info("  MXFP4 DEEPSEEK KEEP ATTENTION LINEAR BF16: %s", name)
            else:
                modules_to_replace.append((name, module, "mxfp4"))

        for name, module, kind in modules_to_replace:
            if kind == "head":
                new_layer = Float32Linear(module.in_features, module.out_features, bias=module.bias is not None)
                new_layer = new_layer.to(module.weight.device).to(module.weight.dtype)
                if module.weight.device.type != "meta":
                    with torch.no_grad():
                        new_layer.weight.copy_(module.weight)
                        if module.bias is not None:
                            new_layer.bias.copy_(module.bias)
                head_count += 1
                logger.info("  MXFP4 DEEPSEEK HEAD BF16/FLOAT32: %s", name)
            else:
                new_layer = MXFP4LinearTK.from_linear(module)
                linear_count += 1
                if kind == "mxfp4_head":
                    head_count += 1
                    logger.info("  MXFP4 DEEPSEEK HEAD MXFP4: %s", name)
                elif kind == "mxfp4_router":
                    router_mxfp4_count += 1
                    logger.info("  MXFP4 DEEPSEEK ROUTER MXFP4: %s", name)
                else:
                    logger.info("  MXFP4 DEEPSEEK LINEAR: %s", name)
            rsetattr(model, name, new_layer)

        grouped_experts_to_replace = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == "GroupedExperts":
                grouped_expert_count += 1
                if _use_deepseek_mxfp4_grouped_experts():
                    grouped_experts_to_replace.append((name, module))
                else:
                    logger.info(
                        "  MXFP4 DEEPSEEK KEEP GROUPED EXPERTS BF16: %s",
                        name,
                    )

        for name, module in grouped_experts_to_replace:
            rsetattr(
                model,
                name,
                MXFP4GroupedExpertsTK.from_grouped_experts(
                    module,
                    packed_w13_param=False if tp_enabled else None,
                ),
            )
            grouped_expert_mxfp4_count += 1
            logger.info("  MXFP4 DEEPSEEK GROUPED EXPERTS BASELINE: %s", name)

        if _use_deepseek_mxfp4_shared_fused_moe() and not tp_enabled:
            for name, block in model.named_modules():
                module = getattr(block, "moe", None)
                if module is None or module.__class__.__name__ != "MoE":
                    continue
                if getattr(module, "shared_experts", None) is None:
                    continue
                if getattr(module.experts, "forward_moe_combine", None) is None:
                    continue
                module._lbt_debug_name = f"{name}.moe"
                if hasattr(module, "router"):
                    module.router._lbt_debug_name = f"{name}.moe.router"
                if _deepseek_mxfp4_fused_moe_rmsnorm() and hasattr(block, "ffn_norm"):
                    norm = block.ffn_norm
                    module._mxfp4_ffn_norm = norm
                    module._mxfp4_rmsnorm_to_bf16 = _deepseek_mxfp4_rmsnorm_to_bf16
                    dim = norm.weight.shape[0] if hasattr(norm, "weight") else 0
                    norm_dtype = norm.weight.dtype if hasattr(norm, "weight") else torch.bfloat16
                    block.ffn_norm = _NormIdentity(dim, dtype=norm_dtype, trainable=False)
                shared_fused_moe_count += 1
                logger.info(
                    "  MXFP4 DEEPSEEK SHARED MOE ROUTE[rmsnorm=%s]: %s.moe",
                    bool(getattr(module, "_mxfp4_ffn_norm", None) is not None),
                    name,
                )

        logger.info(
            "MXFP4TKDeepSeekConverter: fused_mla=%d dense_ffn=%d dense_ffn_bf16=%d fused_shared_experts=%d linears=%d heads=%d head_bf16=%d routers_kept=%d routers_mxfp4=%d shared_expert_bf16=%d grouped_experts=%d grouped_experts_mxfp4=%d shared_fused_moe=%d",
            mla_count,
            dense_ffn_count,
            dense_ffn_bf16_count,
            fused_shared_expert_count,
            linear_count,
            head_count,
            head_bf16_count,
            router_count,
            router_mxfp4_count,
            shared_expert_bf16_count,
            grouped_expert_count,
            grouped_expert_mxfp4_count,
            shared_fused_moe_count,
        )

    def post_optimizer_hook(self, model):
        pass


register_model_converter(MXFP4TKConverter, "mxfp4_tk")
register_model_converter(MXFP4TKDeepSeekConverter, "mxfp4_tk_deepseek")
