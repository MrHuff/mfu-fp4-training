import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice
from typing import Optional, Any, Tuple
from dataclasses import dataclass
try:
    from transformer_engine.pytorch.experimental.quantization_custom_triton import (
        FORMAT_E2M1,
        FORMAT_E4M3,
        FORMAT_E5M2,
        FORMAT_E8M0,
        FORMAT_E5M3,
        _round_float_kernel_impl,
    )
except ImportError:
    FORMAT_E2M1 = None
    FORMAT_E4M3 = None
    FORMAT_E5M2 = None
    FORMAT_E8M0 = None
    FORMAT_E5M3 = None
    _round_float_kernel_impl = None

if _round_float_kernel_impl is None:
    from .fused_quant_triton import (
        _round_float_kernel_impl as _fallback_round_float_kernel_impl,
    )

    _round_float_kernel_impl = _fallback_round_float_kernel_impl

torch.set_float32_matmul_precision("high")


# ============================================================================
# CONSTANTS & HELPERS
# ============================================================================
@dataclass
class TritonFormatInfo:
    """Lightweight format info for Triton kernels."""

    max_val: float
    min_val: float
    precision: int
    bias: int
    has_subnormals: bool
    is_signed: bool
    k: int  # total bits
    has_nz: bool
    has_infs: bool
    num_nans: bool


# Define a static, lightweight container for quantization metadata
class QuantizationMetadata:
    __slots__ = ["global_amax_row", "global_amax_col"]

    def __init__(self, row, col):
        self.global_amax_row = row
        self.global_amax_col = col


if FORMAT_E2M1 is None:
    FORMAT_E2M1 = TritonFormatInfo(
        max_val=6.0,
        min_val=-6.0,
        precision=2,
        bias=1,
        has_subnormals=True,
        is_signed=True,
        k=4,
        has_nz=True,
        has_infs=False,
        num_nans=False,
    )
    FORMAT_E5M3 = TritonFormatInfo(
        max_val=61440.0,
        min_val=0.0,
        precision=4,
        bias=15,
        has_subnormals=True,
        is_signed=False,
        k=8,
        has_nz=True,
        has_infs=True,
        num_nans=True,
    )
    FORMAT_E4M3 = TritonFormatInfo(
        max_val=448.0,
        min_val=-448.0,
        precision=4,
        bias=7,
        has_subnormals=True,
        is_signed=True,
        k=8,
        has_nz=True,
        has_infs=False,
        num_nans=False,
    )
    FORMAT_E8M0 = TritonFormatInfo(
        max_val=float(2**127),
        min_val=float(2**-127),
        precision=1,
        bias=127,
        has_subnormals=False,
        is_signed=False,
        k=8,
        has_nz=True,
        has_infs=True,
        num_nans=True,
    )
    FORMAT_E5M2 = TritonFormatInfo(
        max_val=57344.0,
        min_val=6.103515625e-05,
        precision=3,
        bias=15,
        has_subnormals=True,
        is_signed=True,
        k=8,
        has_nz=True,
        has_infs=True,
        num_nans=True,
    )


FORMAT_MAP = {
    "E2M1": FORMAT_E2M1,
    "E4M3": FORMAT_E4M3,
    "E5M2": FORMAT_E5M2,
    "E8M0": FORMAT_E8M0,
    "E5M3": FORMAT_E5M3,
}


def get_format_info(name):
    if name in FORMAT_MAP:
        return FORMAT_MAP[name]
    # Default fallback or error
    raise ValueError(f"Unknown format: {name}")


# Constants for Python-side reference (keep these global for the wrappers)
RM_TIES_TO_EVEN = 0
RM_TOWARD_ZERO = 1
RM_TOWARD_POSITIVE = 2
RM_TOWARD_NEGATIVE = 3
RM_TIES_TO_AWAY = 4
RM_STOCHASTIC = 5
RM_STOCHASTIC_ODD = 6
RM_STOCHASTIC_FAST = 7
RM_STOCHASTIC_FASTEST = 8


def get_round_mode_constant(name):
    if name == "TiesToEven":
        return RM_TIES_TO_EVEN
    if name == "TowardZero":
        return RM_TOWARD_ZERO
    if name == "TowardPositive":
        return RM_TOWARD_POSITIVE
    if name == "TowardNegative":
        return RM_TOWARD_NEGATIVE
    if name == "TiesToAway":
        return RM_TIES_TO_AWAY
    if name == "Stochastic":
        return RM_STOCHASTIC
    if name == "StochasticOdd":
        return RM_STOCHASTIC_ODD
    if name == "StochasticFast":
        return RM_STOCHASTIC_FAST
    if name == "StochasticFastest":
        return RM_STOCHASTIC_FASTEST
    return RM_TIES_TO_EVEN


@triton.jit
def triton_amax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    val = tl.abs(x)
    block_max = tl.max(val)
    tl.atomic_max(out_ptr, block_max)


# Triton FWHT kernel for external RHT application
# Applies FWHT on last dimension (BLOCK_K=16) with signs and 0.25 scaling
@triton.jit
def triton_fwht_rht_kernel(
    x_ptr,
    out_ptr,
    signs_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_om,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Use 64-bit arithmetic for large offsets
    offs_m_64 = offs_m[:, None].to(tl.int64)
    offs_k_64 = offs_k[None, :].to(tl.int64)

    x_ptrs = x_ptr + offs_m_64 * stride_xm + offs_k_64 * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    scale = 1.0 / (BLOCK_K**0.5)

    # Stages: log2(K)
    # e.g. K=16. stages=4. stride=8, 4, 2, 1.
    # K=32. stages=5. stride=16, 8, 4, 2, 1.

    # Unroll logic needs Python loop over Log2 K?
    # Simple unroll loop trying to detect max possible K is messy if BLOCK_K varies.
    # But BLOCK_K is constexpr! So we can calculate range.

    # However, triton.jit doesn't support complex python math flow easily unless I know Log2 K.
    # Since we support 16 and 32 specifically:

    # FWHT butterfly
    # Generalized for any power of 2 BLOCK_K
    idx_mask = tl.arange(0, 2).reshape(1, 1, 2, 1)
    merge_idx = tl.arange(0, 2).reshape(2, 1, 1, 1)

    if BLOCK_K == 32:
        # Stride 16
        x_r = x.reshape(BLOCK_M, 1, 2, 16)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 1, 1, 16), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 1, 1, 16), [2, 0, 1, 3])
        x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
            BLOCK_M, 32
        )

        # Then flow into Stride 8 (but now with groups=2)
        # Stride 8
        x_r = x.reshape(BLOCK_M, 2, 2, 8)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 2, 1, 8), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 2, 1, 8), [2, 0, 1, 3])
        x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
            BLOCK_M, 32
        )

        # Stride 4
        x_r = x.reshape(BLOCK_M, 4, 2, 4)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 4, 1, 4), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 4, 1, 4), [2, 0, 1, 3])
        x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
            BLOCK_M, 32
        )

        # Stride 2
        x_r = x.reshape(BLOCK_M, 8, 2, 2)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 8, 1, 2), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 8, 1, 2), [2, 0, 1, 3])
        x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
            BLOCK_M, 32
        )

        # Stride 1
        x_r = x.reshape(BLOCK_M, 16, 2, 1)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 16, 1, 1), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 16, 1, 1), [2, 0, 1, 3])
        x_out = (
            tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
                BLOCK_M, 32
            )
            * scale
        )

    else:
        # Default BLOCK_K=16
        # Stride 8
        x_r = x.reshape(BLOCK_M, 1, 2, 8)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 1, 1, 8), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 1, 1, 8), [2, 0, 1, 3])
        x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
            BLOCK_M, 16
        )

        # Stride 4
        x_r = x.reshape(BLOCK_M, 2, 2, 4)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 2, 1, 4), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 2, 1, 4), [2, 0, 1, 3])
        x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
            BLOCK_M, 16
        )

        # Stride 2
        x_r = x.reshape(BLOCK_M, 4, 2, 2)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 4, 1, 2), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 4, 1, 2), [2, 0, 1, 3])
        x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
            BLOCK_M, 16
        )

        # Stride 1
        x_r = x.reshape(BLOCK_M, 8, 2, 1)
        a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
        b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
        sp = tl.permute((a + b).reshape(BLOCK_M, 8, 1, 1), [2, 0, 1, 3])
        dp = tl.permute((a - b).reshape(BLOCK_M, 8, 1, 1), [2, 0, 1, 3])
        x_out = (
            tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(
                BLOCK_M, 16
            )
            * scale
        )

    if signs_ptr is not None:
        signs = tl.load(signs_ptr + tl.arange(0, BLOCK_K))
        x_out = x_out * signs[None, :]

    out_ptrs = out_ptr + offs_m_64 * stride_om + offs_k_64 * stride_ok
    tl.store(out_ptrs, x_out, mask=mask)


@triton.jit
def fake_quant_simultaneous_kernel(
    # Pointers
    a_ptr,
    a_out_ptr,
    b_ptr,
    b_out_ptr,
    global_amax_a_ptr,
    global_amax_b_ptr,
    srbits_a_ptr,
    srbits_b_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides A
    stride_am,
    stride_ak,
    stride_out_am,
    stride_out_ak,
    # Strides B
    stride_bn,
    stride_bk,
    stride_out_bn,
    stride_out_bk,
    # Params
    scale_max_a,
    scale_max_b,
    # A Format
    a_prec: tl.constexpr,
    a_bias: tl.constexpr,
    a_has_sub: tl.constexpr,
    a_max: tl.constexpr,
    a_min: tl.constexpr,
    a_signed: tl.constexpr,
    a_nz: tl.constexpr,
    a_inf: tl.constexpr,
    a_nan: tl.constexpr,
    # B Format
    b_prec: tl.constexpr,
    b_bias: tl.constexpr,
    b_has_sub: tl.constexpr,
    b_max: tl.constexpr,
    b_min: tl.constexpr,
    b_signed: tl.constexpr,
    b_nz: tl.constexpr,
    b_inf: tl.constexpr,
    b_nan: tl.constexpr,
    # Scale A Format
    sa_prec: tl.constexpr,
    sa_bias: tl.constexpr,
    sa_has_sub: tl.constexpr,
    sa_max: tl.constexpr,
    sa_min: tl.constexpr,
    sa_signed: tl.constexpr,
    sa_nz: tl.constexpr,
    sa_inf: tl.constexpr,
    sa_nan: tl.constexpr,
    # Scale B Format
    sb_prec: tl.constexpr,
    sb_bias: tl.constexpr,
    sb_has_sub: tl.constexpr,
    sb_max: tl.constexpr,
    sb_min: tl.constexpr,
    sb_signed: tl.constexpr,
    sb_nz: tl.constexpr,
    sb_inf: tl.constexpr,
    sb_nan: tl.constexpr,
    # Rounding Modes
    data_round_mode_a: tl.constexpr,
    scale_round_mode_a: tl.constexpr,
    data_round_mode_b: tl.constexpr,
    scale_round_mode_b: tl.constexpr,
    # Options
    use_global_scale: tl.constexpr,
    use_2d_b: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    encode_centric: tl.constexpr = False,
    return_encoded: tl.constexpr = False,
):
    pid_row = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Global Scale logic (Identical to fused kernel)
    g_enc_a = 1.0
    g_dec_a = 1.0
    g_enc_b = 1.0
    g_dec_b = 1.0

    if use_global_scale:
        max_f32_val = 3.4028235e38
        one_val = 1.0

        # A
        g_amax_a = tl.load(global_amax_a_ptr)
        factor_a = scale_max_a * a_max
        g_amax_a_f32 = g_amax_a.to(tl.float32)
        factor_a_f32 = tl.full(g_amax_a.shape, factor_a, tl.float32)

        # Use simple div (reference style) - wait, ref uses div_rn(factor, g_amax).
        # We need to protect against div-by-zero during computation if we want to avoid NaN/Inf early?
        # Ref source: global_encode_scale = div_rn(factor_f32, global_amax)
        # Assuming global_amax not modified yet.
        # But wait, division by zero is UB/Inf.
        # Let's check Ref Step 1470 again.
        # Line 640: global_encode_scale = tl.extra.cuda.libdevice.div_rn(factor_f32, global_amax)
        # It calculates it RAW.

        ges_a = tl.extra.cuda.libdevice.div_rn(factor_a_f32, g_amax_a_f32)

        max_f32_t = tl.full(ges_a.shape, max_f32_val, tl.float32)
        ges_a = tl.minimum(ges_a, max_f32_t)
        one_f32_t = tl.full(ges_a.shape, one_val, tl.float32)

        ges_a = tl.where(g_amax_a_f32 == 0.0, one_f32_t, ges_a)

        # GDS: div_rn(1.0, ges). No epsilon clamp on GES!
        gds_a_computed = tl.extra.cuda.libdevice.div_rn(one_f32_t, ges_a)
        g_dec_a = tl.where(ges_a == 0.0, one_f32_t, gds_a_computed)
        g_enc_a = ges_a

        # B
        g_amax_b = tl.load(global_amax_b_ptr)
        factor_b = scale_max_b * b_max
        g_amax_b_f32 = g_amax_b.to(tl.float32)
        factor_b_f32 = tl.full(g_amax_b.shape, factor_b, tl.float32)

        ges_b = tl.extra.cuda.libdevice.div_rn(factor_b_f32, g_amax_b_f32)

        max_f32_b_t = tl.full(ges_b.shape, max_f32_val, tl.float32)
        ges_b = tl.minimum(ges_b, max_f32_b_t)
        one_f32_b_t = tl.full(ges_b.shape, one_val, tl.float32)

        ges_b = tl.where(g_amax_b_f32 == 0.0, one_f32_b_t, ges_b)

        gds_b_computed = tl.extra.cuda.libdevice.div_rn(one_f32_b_t, ges_b)
        g_dec_b = tl.where(ges_b == 0.0, one_f32_b_t, gds_b_computed)
        g_enc_b = ges_b

    # ==========================
    # Process A: (M, K)
    # ==========================
    if pid_row < num_pid_m:
        pid_m = pid_row
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)

        srbits_a = tl.full(a.shape, 0, dtype=tl.int32)
        if srbits_a_ptr is not None:
            srbits_a_ptrs = srbits_a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            srbits_a = tl.load(srbits_a_ptrs, mask=mask_a, other=0)

        # A Quantization (Same as fused kernels A section)
        a_max_val = tl.max(tl.abs(a), axis=1)
        data_max_a_f32 = tl.full(a_max_val.shape, a_max, tl.float32)
        eps_f32 = tl.full(a_max_val.shape, 1e-9, tl.float32)
        is_zero_block_a = a_max_val <= eps_f32

        if encode_centric:
            # 1. Compute Multiplier (Encode Centric View)
            # Reference: M = FP4_MAX / (AMAX * GES)
            # ----------------------------------------------------------------
            # Compute Raw Multiplier (6.0 / Amax)

            if use_global_scale:
                # Divide by GES to get (6.0 / (AMAX * GES))
                tmp_a = tl.extra.cuda.libdevice.mul_rn(a_max_val, g_enc_a)
                s_a = tl.extra.cuda.libdevice.div_rn(data_max_a_f32, tmp_a)

            else:
                s_a = tl.extra.cuda.libdevice.div_rn(data_max_a_f32, a_max_val)

            # Zero Handling: If zero, Multiplier is Max (448.0)
            s_a = tl.where(is_zero_block_a, scale_max_a, s_a)

            # Clamp Multiplier
            scale_max_t = tl.full(s_a.shape, scale_max_a, tl.float32)
            s_a = tl.minimum(s_a, scale_max_t)

            s_a = s_a[:, None]

            # 2. Round Multiplier to FP8
            # ----------------------------------------------------------------
            srbits_sa = tl.full((BLOCK_M, 1), 0, tl.int32)
            if srbits_a_ptr is not None:
                srbits_sa_ptrs = srbits_a_ptr + (offs_am * stride_am)
                srbits_sa_val = tl.load(srbits_sa_ptrs, mask=offs_am < M, other=0)
                srbits_sa = srbits_sa_val[:, None]

            s_a_rounded = _round_float_kernel_impl(
                s_a,
                sa_prec,
                sa_bias,
                sa_has_sub,
                sa_max,
                sa_min,
                sa_signed,
                sa_nz,
                sa_inf,
                sa_nan,
                scale_round_mode_a,
                srbits_sa,
                8,
            )

            # 3. Store Reciprocal (Scale)
            # ----------------------------------------------------------------
            one_f32_post = tl.full(s_a_rounded.shape, 1.0, tl.float32)

            # SAFEGUARD: Avoid Division by Zero
            # If s_a_rounded is 0, the reciprocal is Inf.
            # We map this to sa_max (e.g. 448.0) to avoid NaN in subsequent steps.
            # NOTE: We must ensure we don't actually execute 1/0 if it traps,
            # though on GPU it usually returns Inf.

            recip = tl.extra.cuda.libdevice.div_rn(one_f32_post, s_a_rounded)
            max_f32_sa = tl.full(recip.shape, sa_max, tl.float32)

            # If s_a_rounded == 0.0, use sa_max, else use computed reciprocal
            s_store = tl.where(s_a_rounded == 0.0, max_f32_sa, recip)

            # Additional Clamp: Ensure s_store fits in target FP8 range
            s_store = tl.minimum(s_store, max_f32_sa)

            s_store_rounded = _round_float_kernel_impl(
                s_store,
                sa_prec,
                sa_bias,
                sa_has_sub,
                sa_max,
                sa_min,
                sa_signed,
                sa_nz,
                sa_inf,
                sa_nan,
                scale_round_mode_a,
                srbits_sa,
                8,
            )

            # 4. Data Scaling (Use Multiplier directly)
            # ----------------------------------------------------------------
            # s_a_rounded is the Multiplier (M).
            # We want to scale data by: M * GES.

            max_f32_t_a = tl.full(s_a_rounded.shape, 3.4028235e38, tl.float32)

            if use_global_scale:
                es_a = tl.extra.cuda.libdevice.mul_rn(s_a_rounded, g_enc_a)
            else:
                es_a = s_a_rounded

            es_a = tl.minimum(es_a, max_f32_t_a)

            es_a_b = tl.broadcast_to(es_a, a.shape)
            a_scaled = tl.extra.cuda.libdevice.mul_rn(a, es_a_b)

            # 6. Quantize & Output
            # ----------------------------------------------------------------
            a_q = _round_float_kernel_impl(
                a_scaled,
                a_prec,
                a_bias,
                a_has_sub,
                a_max,
                a_min,
                a_signed,
                a_nz,
                a_inf,
                a_nan,
                data_round_mode_a,
                srbits_a,
                8,
            )

            # Dequantize for Fake Quant Output:
            # We stored Scale (1/M).
            # Output = Quantized * Scale = (Input * M) * (1/M) ~= Input. Correct.
            a_dq = a_q * s_store_rounded

        else:
            # DECODE CENTRIC
            tmp_a = tl.extra.cuda.libdevice.div_rn(a_max_val, data_max_a_f32)
            s_a = tmp_a
            if use_global_scale:
                s_a = tl.extra.cuda.libdevice.mul_rn(tmp_a, g_enc_a)
            s_a = tl.where(is_zero_block_a, tl.full(s_a.shape, 0.0, tl.float32), s_a)
            s_a = s_a[:, None]

            srbits_sa = tl.full((BLOCK_M, 1), 0, tl.int32)
            if srbits_a_ptr is not None:
                srbits_sa_ptrs = srbits_a_ptr + (offs_am * stride_am)
                srbits_sa_val = tl.load(srbits_sa_ptrs, mask=offs_am < M, other=0)
                srbits_sa = srbits_sa_val[:, None]

            s_a = _round_float_kernel_impl(
                s_a,
                sa_prec,
                sa_bias,
                sa_has_sub,
                sa_max,
                sa_min,
                sa_signed,
                sa_nz,
                sa_inf,
                sa_nan,
                scale_round_mode_a,
                srbits_sa,
                8,
            )

            # MATCH REFERENCE BEHAVIOR (Noise Amplification for Zero Blocks)
            max_f32_val = 3.4028235e38
            if use_global_scale:
                denom_a = tl.extra.cuda.libdevice.mul_rn(s_a, g_dec_a)
                # Denom allowed to be 0 -> es_a becomes Inf -> Clamped to MaxFloat
                one_f32_bcast = tl.full(denom_a.shape, 1.0, tl.float32)
                es_a = tl.extra.cuda.libdevice.div_rn(one_f32_bcast, denom_a)

                max_f32_bcast = tl.full(es_a.shape, max_f32_val, tl.float32)
                es_a = tl.minimum(es_a, max_f32_bcast)
            else:
                one_f32_bcast = tl.full(s_a.shape, 1.0, tl.float32)
                es_a = tl.extra.cuda.libdevice.div_rn(one_f32_bcast, s_a)

                max_f32_bcast = tl.full(es_a.shape, max_f32_val, tl.float32)
                es_a = tl.minimum(es_a, max_f32_bcast)

            es_a_b = tl.broadcast_to(es_a, a.shape)
            a_scaled = tl.extra.cuda.libdevice.mul_rn(a, es_a_b)
            a_q = _round_float_kernel_impl(
                a_scaled,
                a_prec,
                a_bias,
                a_has_sub,
                a_max,
                a_min,
                a_signed,
                a_nz,
                a_inf,
                a_nan,
                data_round_mode_a,
                srbits_a,
                8,
            )
            a_dq = a_q * s_a

        # Unscale global quantization factor if needed for fake quantization output
        if use_global_scale:
            if not return_encoded:
                a_dq = tl.extra.cuda.libdevice.mul_rn(a_dq, g_dec_a)

        # Store A_dq
        out_a_ptrs = a_out_ptr + (
            offs_am[:, None] * stride_out_am + offs_k[None, :] * stride_out_ak
        )
        tl.store(out_a_ptrs, a_dq, mask=mask_a)

    # ==========================
    # Process B: (K, N)
    # ==========================
    if pid_row < num_pid_n:
        pid_n = pid_row
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_b = (offs_bn[None, :] < N) & (offs_k[:, None] < K)
        b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)

        srbits_b = tl.full(b.shape, 0, dtype=tl.int32)
        if srbits_b_ptr is not None:
            srbits_b_ptrs = srbits_b_ptr + (
                offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk
            )
            srbits_b = tl.load(srbits_b_ptrs, mask=mask_b, other=0)

        if use_2d_b:
            # Reshape to (K_chunks, BLOCK_K, N_chunks, BLOCK_K) to support variable tiling
            # defined by BLOCK_K (e.g. 16 or 32).
            # Assumes BLOCK_K and BLOCK_N are compatible (BLOCK_N % BLOCK_K == 0).

            # 1. Reshape to 4D to isolate BLOCK_K x BLOCK_K tiles
            b_4d = tl.reshape(
                b, (BLOCK_K // BLOCK_K, BLOCK_K, BLOCK_N // BLOCK_K, BLOCK_K)
            )

            # 2. Compute Max within tiles (reduce dimensions 1 and 3)
            # Max over rows (axis 1)
            m1 = tl.max(tl.abs(b_4d), axis=1)  # -> (1, BLOCK_N//BLOCK_K, BLOCK_K)
            # Max over cols (axis 2 of result, which was axis 3 of original)
            b_max_val_tiled = tl.max(m1, axis=2)  # -> (1, BLOCK_N//BLOCK_K)

            # 3. Broadcast scale back to original shape
            m_expanded_1 = b_max_val_tiled[:, None, :, None]  # (1, 1, N/K, 1)
            one_4d = tl.full((1, BLOCK_K, 1, BLOCK_K), 1.0, tl.float32)
            b_max_val_4d = m_expanded_1 * one_4d
            b_max_val = tl.reshape(b_max_val_4d, (BLOCK_K, BLOCK_N))

        else:
            b_max_val = tl.max(tl.abs(b), axis=0)
            # For 1D, shape is (BLOCK_N,).
            # broadcast_to later handles (BLOCK_N,) -> (BLOCK_K, BLOCK_N).
            # My 2D logic produces (BLOCK_K, BLOCK_N).
            # broadcast_to will handle (K, N) -> (K, N) fine.

        data_max_b_f32 = tl.full(b_max_val.shape, b_max, tl.float32)
        eps_f32_b = tl.full(b_max_val.shape, 1e-9, tl.float32)
        is_zero_block_b = b_max_val <= eps_f32_b

        if encode_centric:
            # 1. Compute Base Scale (Multiplier for Encode Centric)
            # Reference: M = FP4_MAX / (AMAX * GES)
            # ----------------------------------------------------------------
            # Compute Raw Multiplier: M = 6.0 / AMAX

            if use_global_scale:
                # Divide by GES to get (6.0 / (AMAX * GES))
                tmp_b = tl.extra.cuda.libdevice.mul_rn(b_max_val, g_enc_b)
                s_b = tl.extra.cuda.libdevice.div_rn(data_max_b_f32, tmp_b)

            else:
                s_b = tl.extra.cuda.libdevice.div_rn(data_max_b_f32, b_max_val)

            # Zero handling: If block is zero, Multiplier = Max
            s_b = tl.where(is_zero_block_b, scale_max_b, s_b)

            # Clamp Multiplier
            scale_max_b_t = tl.full(s_b.shape, scale_max_b, tl.float32)
            s_b = tl.minimum(s_b, scale_max_b_t)

            # Handle 1D broadcast if necessary
            if not use_2d_b:
                s_b = s_b[None, :]

            # 2. Round Multiplier to FP8
            # ----------------------------------------------------------------
            srbits_sb = tl.full((1, BLOCK_N), 0, tl.int32)
            if srbits_b_ptr is not None:
                srbits_sb_ptrs = srbits_b_ptr + (offs_bn * stride_bn)
                srbits_sb_val = tl.load(srbits_sb_ptrs, mask=offs_bn < N, other=0)
                # Broadcast 1D bits to 2D tiles
                srbits_sb = srbits_sb_val[None, :]

            s_b_rounded = _round_float_kernel_impl(
                s_b,
                sb_prec,
                sb_bias,
                sb_has_sub,
                sb_max,
                sb_min,
                sb_signed,
                sb_nz,
                sb_inf,
                sb_nan,
                scale_round_mode_b,
                srbits_sb,
                8,
            )

            one_f32_post_b = tl.full(s_b_rounded.shape, 1.0, tl.float32)

            # SAFEGUARD: Avoid Division by Zero
            recip_b = tl.extra.cuda.libdevice.div_rn(one_f32_post_b, s_b_rounded)
            max_f32_sb = tl.full(recip_b.shape, sb_max, tl.float32)

            s_store_b = tl.where(s_b_rounded == 0.0, max_f32_sb, recip_b)

            # Additional Clamp
            s_store_b = tl.minimum(s_store_b, max_f32_sb)

            # MATCH REFERENCE: Reciprocal is stored as FP8
            s_store_b = _round_float_kernel_impl(
                s_store_b,
                sb_prec,
                sb_bias,
                sb_has_sub,
                sb_max,
                sb_min,
                sb_signed,
                sb_nz,
                sb_inf,
                sb_nan,
                scale_round_mode_b,
                srbits_sb,
                8,
            )

            # 4. Data Scaling (Use Multiplier directly)
            # ----------------------------------------------------------------
            # s_b_rounded is the Multiplier (M).
            # We want to scale data by: M * GES.

            max_f32_val_b_t = tl.full(s_b_rounded.shape, 3.4028235e38, tl.float32)

            if use_global_scale:
                es_b = tl.extra.cuda.libdevice.mul_rn(s_b_rounded, g_enc_b)
            else:
                es_b = s_b_rounded

            es_b = tl.minimum(es_b, max_f32_val_b_t)

            es_b_b = tl.broadcast_to(es_b, b.shape)
            b_scaled = tl.extra.cuda.libdevice.mul_rn(b, es_b_b)

            # 6. Quantize & Store
            # ----------------------------------------------------------------
            b_q = _round_float_kernel_impl(
                b_scaled,
                b_prec,
                b_bias,
                b_has_sub,
                b_max,
                b_min,
                b_signed,
                b_nz,
                b_inf,
                b_nan,
                data_round_mode_b,
                srbits_b,
                8,
            )

            b_dq = b_q * s_store_b

        else:
            tmp_b = tl.extra.cuda.libdevice.div_rn(b_max_val, data_max_b_f32)
            s_b = tmp_b
            if use_global_scale:
                s_b = tl.extra.cuda.libdevice.mul_rn(tmp_b, g_enc_b)
            s_b = tl.where(is_zero_block_b, tl.full(s_b.shape, 0.0, tl.float32), s_b)

            if not use_2d_b:
                s_b = s_b[None, :]

            srbits_sb = tl.full((1, BLOCK_N), 0, tl.int32)
            if srbits_b_ptr is not None:
                # For 2D case, we ideally want (BLOCK_K, BLOCK_N) bits.
                # But for now, let's keep (1, N) broadcast if K dim is not needed implies using row 0 bits for all rows.
                # If we reuse srbits_b (Data bits), we get (K, N).
                if use_2d_b:
                    # Reuse data bits for scale (simplest fix to match shape)
                    # Or broadcast (1, N) to (K, N) implicitly in sub?
                    # round kernel handles broadcast.
                    # But (1, N) vs (K, N) scale works.
                    pass

                # Logic for 1D extraction
                srbits_sb_ptrs = srbits_b_ptr + (offs_bn * stride_bn)
                srbits_sb_val = tl.load(srbits_sb_ptrs, mask=offs_bn < N, other=0)

                # If 1D, unsqeeze. If 2D, we might want to broadcast explicitly?
                # Actually, srbits_sb_val is (BLOCK_N,).
                if not use_2d_b:
                    srbits_sb = srbits_sb_val[None, :]
                else:
                    srbits_sb = srbits_sb_val[None, :]  # Broadcast 1D bits to 2D tiles

            s_b = _round_float_kernel_impl(
                s_b,
                sb_prec,
                sb_bias,
                sb_has_sub,
                sb_max,
                sb_min,
                sb_signed,
                sb_nz,
                sb_inf,
                sb_nan,
                scale_round_mode_b,
                srbits_sb,
                8,
            )

            # MATCH REFERENCE BEHAVIOR (Noise Amplification for Zero Blocks)
            max_f32_val_b = 3.4028235e38
            if use_global_scale:
                denom_b = tl.extra.cuda.libdevice.mul_rn(s_b, g_dec_b)
                one_f32_bcast_b = tl.full(denom_b.shape, 1.0, tl.float32)
                es_b = tl.extra.cuda.libdevice.div_rn(one_f32_bcast_b, denom_b)

                max_f32_bcast_b = tl.full(es_b.shape, max_f32_val_b, tl.float32)
                es_b = tl.minimum(es_b, max_f32_bcast_b)
            else:
                one_f32_bcast_b = tl.full(s_b.shape, 1.0, tl.float32)
                es_b = tl.extra.cuda.libdevice.div_rn(one_f32_bcast_b, s_b)

                max_f32_bcast_b = tl.full(es_b.shape, max_f32_val_b, tl.float32)
                es_b = tl.minimum(es_b, max_f32_bcast_b)

            es_b_b = tl.broadcast_to(es_b, b.shape)
            b_scaled = tl.extra.cuda.libdevice.mul_rn(b, es_b_b)
            b_q = _round_float_kernel_impl(
                b_scaled,
                b_prec,
                b_bias,
                b_has_sub,
                b_max,
                b_min,
                b_signed,
                b_nz,
                b_inf,
                b_nan,
                data_round_mode_b,
                srbits_b,
                8,
            )
            b_dq = b_q * s_b

        # Unscale global quantization factor if needed for fake quantization output
        if use_global_scale:
            if not return_encoded:
                b_dq = tl.extra.cuda.libdevice.mul_rn(b_dq, g_dec_b)

        # Store B
        out_b_ptrs = b_out_ptr + (
            offs_bn[None, :] * stride_out_bn + offs_k[:, None] * stride_out_bk
        )
        tl.store(out_b_ptrs, b_dq, mask=mask_b)


def fake_quant_simultaneous(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_max_a: float = 448.0,
    scale_max_b: float = 448.0,
    use_global_scale: bool = True,
    ga_a: Optional[torch.Tensor] = None,
    ga_b: Optional[torch.Tensor] = None,
    scale_type: str = "E4M3",
    data_dtype: torch.dtype = torch.float32,
    # Strides
    stride_am=None,
    stride_ak=None,
    stride_bn=None,
    stride_bk=None,
    stride_out_am=None,
    stride_out_ak=None,
    stride_out_bn=None,
    stride_out_bk=None,
    # Rounding
    round_mode_a="TiesToEven",
    scale_round_mode_a="TiesToEven",
    round_mode_b="TiesToEven",
    scale_round_mode_b="TiesToEven",
    # Advanced
    srbits_a=None,
    srbits_b=None,
    use_2d_b: bool = False,
    encode_centric: bool = False,
    block_size: int = 16,
    return_encoded: bool = False,
    **kwargs,
):
    """
    Fake quantize A and B simultaneously along specified axes (K).
    Assumes A is (M, K) and B is (K, N) logically.
    K is the quantization axis.
    """
    M, K = a.shape
    if b.ndim == 2:
        Kb, N = b.shape
        # We don't enforce K==Kb strict equality if user wants custom control via strides?
        # But for simultaneous launch with shared K block grid, K should be similar.
    else:
        # B might be specialized
        N = 1

    # Defaults
    if stride_am is None:
        stride_am = a.stride(0)
    if stride_ak is None:
        stride_ak = a.stride(1)
    if stride_bn is None:
        stride_bn = b.stride(1)  # B=(K,N), stride_n is axis 1
    if stride_bk is None:
        stride_bk = b.stride(0)  # B=(K,N), stride_k is axis 0

    out_a = torch.empty_like(a, dtype=data_dtype)
    out_b = torch.empty_like(b, dtype=data_dtype)

    if stride_out_am is None:
        stride_out_am = out_a.stride(0)
    if stride_out_ak is None:
        stride_out_ak = out_a.stride(1)
    if stride_out_bn is None:
        stride_out_bn = out_b.stride(1)
    if stride_out_bk is None:
        stride_out_bk = out_b.stride(0)

    # Compute Global Amax if needed
    # Internal calculation removed per instruction

    # Format Utils
    format_a = FORMAT_E2M1  # Placeholder, maybe use args later
    format_b = FORMAT_E2M1

    format_sa = get_format_info(scale_type)
    format_sb = get_format_info(scale_type)

    # Kernel Launch
    def grid_fn(META):
        num_m = triton.cdiv(M, META["BLOCK_M"])
        num_n = triton.cdiv(N, META["BLOCK_N"])
        return (max(num_m, num_n), triton.cdiv(K, META["BLOCK_K"]))

    # Round Modes
    drm_a = get_round_mode_constant(round_mode_a)
    srm_a = get_round_mode_constant(scale_round_mode_a)

    drm_b = get_round_mode_constant(round_mode_b)
    srm_b = get_round_mode_constant(scale_round_mode_b)

    BLOCK_N_VAL = kwargs.get("BLOCK_N", 64)

    fake_quant_simultaneous_kernel[grid_fn](
        a,
        out_a,
        b,
        out_b,
        ga_a,
        ga_b,
        srbits_a,
        srbits_b,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_out_am,
        stride_out_ak,
        stride_bn,
        stride_bk,
        stride_out_bn,
        stride_out_bk,
        scale_max_a,
        scale_max_b,
        # A
        format_a.precision,
        format_a.bias,
        format_a.has_subnormals,
        format_a.max_val,
        format_a.min_val,
        format_a.is_signed,
        format_a.has_nz,
        format_a.has_infs,
        format_a.num_nans,
        # B
        format_b.precision,
        format_b.bias,
        format_b.has_subnormals,
        format_b.max_val,
        format_b.min_val,
        format_b.is_signed,
        format_b.has_nz,
        format_b.has_infs,
        format_b.num_nans,
        # SA
        format_sa.precision,
        format_sa.bias,
        format_sa.has_subnormals,
        format_sa.max_val,
        format_sa.min_val,
        format_sa.is_signed,
        format_sa.has_nz,
        format_sa.has_infs,
        format_sa.num_nans,
        # SB
        format_sb.precision,
        format_sb.bias,
        format_sb.has_subnormals,
        format_sb.max_val,
        format_sb.min_val,
        format_sb.is_signed,
        format_sb.has_nz,
        format_sb.has_infs,
        format_sb.num_nans,
        drm_a,
        srm_a,
        drm_b,
        srm_b,
        use_global_scale,
        use_2d_b,
        BLOCK_M=64,  # Default tuning
        BLOCK_N=BLOCK_N_VAL,
        BLOCK_K=block_size,
        encode_centric=encode_centric,
        return_encoded=return_encoded,
    )

    return out_a, out_b


# ============================================================================
# FUNCTION
# ============================================================================


class TritonFusedQuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_flat, weight, bias, params):
        params_saved = params.copy()

        scale_max = params["scale_max"]
        use_global_scale = params["use_global_scale"]
        block_size = params["block_size"]
        scale_type = params.get("scale_type", "E4M3")

        # Rounding params
        srbits_a = params.get("srbits_a", None)
        srbits_b = params.get("srbits_b", None)

        # RHT / 2D params
        use_2d_weights = params.get("use_2d_weights", False)
        encode_centric = params.get("encode_centric", False)

        ctx.saved_params = params_saved

        # Handle >= 2D inputs

        M, K = input_flat.shape
        data_dtype = params.get("data_dtype", torch.float32)

        # Compute AMAX here
        # Compute AMAX here
        input_f32 = input_flat.to(torch.float32).contiguous()
        weight_f32 = weight.to(torch.float32).contiguous()
        if use_global_scale:
            # For X
            ga_x = (
                torch.max(input_f32.abs().max(), input_f32.t().abs().max())
                .view(1)
                .clamp(min=1e-9)
                .to(torch.float32)
            )
            # For W
            ga_w = (
                torch.max(weight_f32.abs().max(), weight_f32.t().abs().max())
                .view(1)
                .clamp(min=1e-9)
                .to(torch.float32)
            )
        else:
            ga_x = torch.tensor([1.0], device=input_flat.device, dtype=torch.float32)
            ga_w = torch.tensor([1.0], device=weight.device, dtype=torch.float32)

        # 1. Fake Quantize Input (A) and Weight (B) simultaneously
        # Ensure BLOCK_N >= block_size for 2D reshaping logic
        blk_n = max(64, block_size)

        # Use Unscaled (Encoded) values for Parity with Reference
        a_dq, b_dq = fake_quant_simultaneous(
            input_f32,
            weight_f32.t(),
            scale_max_a=scale_max,
            scale_max_b=scale_max,
            use_global_scale=use_global_scale,
            scale_type=scale_type,
            data_dtype=data_dtype,
            # Strides inferred from transposed input
            stride_am=input_f32.shape[1],
            stride_ak=1,
            stride_bn=weight_f32.shape[
                1
            ],  # weight.t() stride(1) -> weight.stride(0) -> K
            stride_bk=1,  # weight.t() stride(0) -> weight.stride(1) -> 1
            stride_out_am=input_f32.shape[1],
            stride_out_ak=1,
            stride_out_bn=weight_f32.shape[1],
            stride_out_bk=1,
            # Rounding
            round_mode_a=params.get("roundMode", "TiesToEven"),
            scale_round_mode_a=params.get("scale_round_mode", "TiesToEven"),
            round_mode_b=params.get("roundMode", "TiesToEven"),
            scale_round_mode_b=params.get("scale_round_mode", "TiesToEven"),
            srbits_a=srbits_a,
            srbits_b=srbits_b,
            use_2d_b=use_2d_weights,
            encode_centric=encode_centric,
            block_size=block_size,
            BLOCK_N=blk_n,  # Pass explicit BLOCK_N
            return_encoded=True,
            ga_a=ga_x,
            ga_b=ga_w,
        )

        if use_global_scale:
            ctx.save_for_backward(input_flat, weight, bias, ga_x, ga_w)
        else:
            ctx.save_for_backward(input_flat, weight, bias, None, None)

        # 2. Matmul
        use_fp32_matmul = params.get("use_fp32_matmul", True)
        if use_fp32_matmul:
            a_dq = a_dq.to(torch.float32)
            b_dq = b_dq.to(torch.float32)
        else:
            a_dq = a_dq.to(torch.bfloat16)
            b_dq = b_dq.to(torch.bfloat16)

        y = torch.mm(a_dq, b_dq)

        # 3. Apply Alpha Scaling (if global scale is used)
        if use_global_scale:
            # Alpha = (ga_a * ga_b) / (factor ^ 2)
            # factor = SCALE_MAX * DATA_MAX
            # Note: We need correct DATA_MAX (usually 6.0 for FP4/E2M1)
            # We assume E2M1 data format as per TritonCustomQuantizer default
            DATA_MAX = 6.0  # FORMAT_E2M1.max_val
            factor = (scale_max * DATA_MAX) ** 2
            # Ensure pure float32 accumulation before division
            term = (ga_x * ga_w).to(torch.float32)
            alpha = term / factor
            y = y * alpha

        if bias is not None:
            y = y + bias

        return y.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, ga_x, ga_w = ctx.saved_tensors
        params = ctx.saved_params
        # Retrieve saved AMAX
        # Retrieve saved AMAX
        # ga_x, ga_w unpacked from saved_tensors

        block_size = params["block_size"]
        scale_max = params["scale_max"]
        use_global_scale = params["use_global_scale"]
        scale_type = params.get("scale_type", "E4M3")
        use_2d_weights = params.get("use_2d_weights", False)  # For W
        data_dtype = params.get("data_dtype", torch.float32)
        encode_centric = params.get("encode_centric", False)

        # RHT params
        with_rht = params.get("with_rht", False)
        with_random_sign_mask = params.get("with_random_sign_mask", True)

        dY = grad_output.to(torch.float32).contiguous()
        X = input.to(torch.float32).contiguous()
        W = weight.to(torch.float32).contiguous()

        # Flatten input/grad if needed
        dY_shape = dY.shape
        if dY.dim() > 2:
            dY_flat = dY.reshape(-1, dY_shape[-1])
        else:
            dY_flat = dY

        X_shape = X.shape
        if X.dim() > 2:
            X_flat = X.reshape(-1, X_shape[-1])
        else:
            X_flat = X

        M, N = dY_flat.shape
        _, K = W.shape

        # Ensure BLOCK_N >= block_size for 2D reshaping logic
        blk_n = max(64, block_size)

        # Compute dY AMAX
        if use_global_scale:
            ga_dy = (
                torch.max(dY_flat.abs().max(), dY_flat.t().abs().max())
                .view(1)
                .clamp(min=1e-9)
                .to(torch.float32)
            )
        else:
            ga_dy = torch.tensor([1.0], device=dY_flat.device, dtype=torch.float32)

        # 1. dX = dY @ W
        dy_dq, w_dq = fake_quant_simultaneous(
            dY_flat,
            W,
            scale_max_a=scale_max,
            scale_max_b=scale_max,
            use_global_scale=use_global_scale,
            scale_type=scale_type,
            data_dtype=data_dtype,
            stride_bn=W.stride(1),
            stride_bk=W.stride(0),
            stride_out_bn=W.stride(1),
            stride_out_bk=W.stride(0),
            use_2d_b=use_2d_weights,  # W usually uses 2D
            encode_centric=encode_centric,
            block_size=block_size,
            BLOCK_N=blk_n,  # Pass explicit BLOCK_N
            return_encoded=True,
            ga_a=ga_dy,
            ga_b=ga_w,
        )

        use_fp32_matmul = params.get("use_fp32_matmul", True)
        if use_fp32_matmul:
            dy_dq = dy_dq.to(torch.float32)
            w_dq = w_dq.to(torch.float32)
        else:
            dy_dq = dy_dq.to(torch.bfloat16)
            w_dq = w_dq.to(torch.bfloat16)

        grad_input = torch.mm(dy_dq, w_dq)

        if use_global_scale:
            DATA_MAX = 6.0
            factor = (scale_max * DATA_MAX) ** 2
            # Ensure pure float32 accumulation before division
            term = (ga_dy * ga_w).to(torch.float32)
            alpha_dx = term / factor
            grad_input = grad_input * alpha_dx

        grad_input = grad_input.to(input.dtype)

        # Reshape grad_input if needed
        if dY.dim() > 2:
            grad_input = grad_input.view(*dY_shape[:-1], K)

        # 2. dW = dY.T @ X
        if with_rht:
            if with_random_sign_mask:
                if block_size == 16:
                    WGRAD_SIGNS = torch.tensor(
                        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
                        dtype=torch.float32,
                        device=dY.device,
                    )
                else:
                    # Generate signs for other block sizes
                    WGRAD_SIGNS = (
                        torch.randint(
                            0, 2, (block_size,), device=dY.device, dtype=torch.float32
                        )
                        * 2
                        - 1
                    )
            else:
                WGRAD_SIGNS = torch.ones(
                    block_size, dtype=torch.float32, device=dY.device
                )

            rht_algo = params.get("rht_algo", "fwht")

            if rht_algo == "matmul_torch" or rht_algo == "matmul":
                # Matmul implementation
                def get_hadamard_matrix(n, device, dtype):
                    # Recursive construction
                    if n == 1:
                        return torch.tensor([[1.0]], device=device, dtype=dtype)
                    h = get_hadamard_matrix(n // 2, device, dtype)
                    return torch.cat(
                        [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
                    )

                # Cache mechanism could be added but for now just generate (small size)
                H = get_hadamard_matrix(block_size, dY.device, torch.float32)
                H = H / (block_size**0.5)  # Normalize

                # Apply Signs and Transform dY_t
                dY_t = dY_flat.t().contiguous()  # (N, M)
                dY_t_fwht = dY_t * WGRAD_SIGNS.view(1, -1).repeat(
                    1, M // block_size
                )  # This assumes WGRAD_SIGNS repeats? NO.
                # WGRAD_SIGNS is size `block_size`.
                # We apply it to every block.
                # dY_t shape (N, M). M is divisible by block_size.
                # Reshape to (N, M/B, B).

                # Careful with WGRAD_SIGNS application:
                # Kernel does: x = x * signs[None, :]
                # signs is applied to the LAST dimension (which is BLOCK_K).
                # So we reshape to (..., B). Multiply by signs (broadcast). Then Matmul.

                dY_reshaped = dY_t.view(-1, block_size)
                dY_reshaped = dY_reshaped * WGRAD_SIGNS  # (..., B) * (B)
                if use_fp32_matmul:
                    dY_t_fwht = torch.matmul(dY_reshaped, H).view(dY_t.shape).contiguous()
                else:
                    dY_t_fwht = (
                        torch.matmul(dY_reshaped.bfloat16(), H.bfloat16())
                        .view(dY_t.shape)
                        .contiguous()
                    )

                # Apply Signs and Transform X_t
                X_t = X_flat.t().contiguous()
                X_reshaped = X_t.view(-1, block_size)
                X_reshaped = X_reshaped * WGRAD_SIGNS
                if use_fp32_matmul:
                    X_t_fwht = torch.matmul(X_reshaped, H).view(X_t.shape).contiguous()
                else:
                    X_t_fwht = (
                        torch.matmul(X_reshaped.bfloat16(), H.bfloat16())
                        .view(X_t.shape)
                        .contiguous()
                    )

            else:
                # Triton FWHT algorithm
                BLOCK_M_RHT = 32

                dY_t = dY_flat.t().contiguous()  # (N, M)
                dY_t_fwht = torch.empty_like(dY_t, dtype=torch.float32)
                grid_dy = (triton.cdiv(N, BLOCK_M_RHT), triton.cdiv(M, block_size))
                triton_fwht_rht_kernel[grid_dy](
                    dY_t,
                    dY_t_fwht,
                    WGRAD_SIGNS,
                    N,
                    M,
                    dY_t.stride(0),
                    dY_t.stride(1),
                    dY_t_fwht.stride(0),
                    dY_t_fwht.stride(1),
                    BLOCK_M=BLOCK_M_RHT,
                    BLOCK_K=block_size,
                )

                X_t = X_flat.t().contiguous()  # (K, M)
                X_t_fwht = torch.empty_like(X_t, dtype=torch.float32)
                grid_x = (triton.cdiv(K, BLOCK_M_RHT), triton.cdiv(M, block_size))
                triton_fwht_rht_kernel[grid_x](
                    X_t,
                    X_t_fwht,
                    WGRAD_SIGNS,
                    K,
                    M,
                    X_t.stride(0),
                    X_t.stride(1),
                    X_t_fwht.stride(0),
                    X_t_fwht.stride(1),
                    BLOCK_M=BLOCK_M_RHT,
                    BLOCK_K=block_size,
                )

            dY_target = dY_t_fwht  # (N, M)
            X_target = X_t_fwht  # (K, M)

            if use_global_scale:
                ga_dy_target = (
                    torch.max(dY_target.abs().max(), dY_target.t().abs().max())
                    .view(1)
                    .clamp(min=1e-9)
                    .to(torch.float32)
                )
                ga_x_target = (
                    torch.max(X_target.abs().max(), X_target.t().abs().max())
                    .view(1)
                    .clamp(min=1e-9)
                    .to(torch.float32)
                )
            else:
                ga_dy_target = torch.tensor(
                    [1.0], device=dY_target.device, dtype=torch.float32
                )
                ga_x_target = torch.tensor(
                    [1.0], device=X_target.device, dtype=torch.float32
                )

            dy_dq, x_dq = fake_quant_simultaneous(
                dY_target,
                X_target.t(),
                scale_max_a=scale_max,
                scale_max_b=scale_max,
                use_global_scale=use_global_scale,
                scale_type=scale_type,
                stride_am=dY_target.shape[1],
                stride_ak=1,
                stride_bn=X_target.shape[
                    1
                ],  # X_target (K,M). X_target.t() (M,K). stride(1)=M.
                stride_bk=1,  # X_target.t() stride(0)=1.
                stride_out_am=dY_target.shape[1],
                stride_out_ak=1,
                stride_out_bn=X_target.shape[1],
                stride_out_bk=1,
                data_dtype=data_dtype,
                # stride_bn/stride_bk inferred from transposed input
                use_2d_b=False,
                encode_centric=encode_centric,
                block_size=block_size,
                BLOCK_N=blk_n,  # Pass explicit BLOCK_N
                return_encoded=True,
                ga_a=ga_dy_target,
                ga_b=ga_x_target,
            )

            use_fp32_matmul = params.get("use_fp32_matmul", True)
            if use_fp32_matmul:
                dy_dq = dy_dq.to(torch.float32)
                x_dq = x_dq.to(torch.float32)
            else:
                dy_dq = dy_dq.to(torch.bfloat16)
                x_dq = x_dq.to(torch.bfloat16)

            grad_weight = torch.mm(dy_dq, x_dq)

            if use_global_scale:
                DATA_MAX = 6.0
                factor = (scale_max * DATA_MAX) ** 2
                # Ensure pure float32 accumulation before division
                term = (ga_dy_target * ga_x_target).to(torch.float32)
                alpha_dw = term / factor
                grad_weight = grad_weight * alpha_dw

        else:
            # WGRAD: dW = dY.T @ X

            # Standard Mode (Efficient FP8 WGRAD)
            # Quantize dY.T (N, M) -> Row-wise (Scale s_n)
            # Quantize X (M, K) -> Col-wise (Scale s_k)
            # (Matches A_row @ B_col standard)
            dY_t = dY_flat.t().contiguous()

            dy_dq, x_dq = fake_quant_simultaneous(
                dY_t,
                X_flat,
                scale_max_a=scale_max,
                scale_max_b=scale_max,
                use_global_scale=use_global_scale,
                scale_type=scale_type,
                data_dtype=data_dtype,
                stride_am=dY_t.shape[1],
                stride_ak=1,
                stride_bn=1,  # X_flat (M, K). stride(1)=1.
                stride_bk=X_flat.shape[1],  # X_flat (M, K). stride(0)=K.
                stride_out_am=dY_t.shape[1],
                stride_out_ak=1,
                stride_out_bn=1,
                stride_out_bk=X_flat.shape[1],
                use_2d_b=False,
                encode_centric=encode_centric,
                block_size=block_size,
                BLOCK_N=blk_n,  # Pass explicit BLOCK_N
                return_encoded=True,
                ga_a=ga_dy,
                ga_b=ga_x,
            )
            use_fp32_matmul = params.get("use_fp32_matmul", True)
            if use_fp32_matmul:
                dy_dq = dy_dq.to(torch.float32)
                x_dq = x_dq.to(torch.float32)

            grad_weight = torch.mm(dy_dq, x_dq)

            if use_global_scale:
                DATA_MAX = 6.0
                factor = (scale_max * DATA_MAX) ** 2
                term = (ga_dy * ga_x).to(torch.float32)
                alpha_dw = term / factor
                grad_weight = grad_weight * alpha_dw

        grad_bias = None
        if bias is not None:
            grad_bias = dY_flat.sum(dim=0).to(torch.bfloat16)  # Sum over M (batch dims)

        return (
            grad_input.to(torch.bfloat16),
            grad_weight.to(torch.bfloat16),
            grad_bias,
            None,
        )


class TritonFusedQuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        scale_max: float = 448.0,
        use_global_scale: bool = True,
        block_size: int = 16,
        scale_dtype: str = "E4M3",
        data_dtype: torch.dtype = torch.bfloat16,
        mx_config=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Default Config
        self.scale_max = scale_max
        self.use_global_scale = use_global_scale
        self.block_size = block_size
        self.scale_dtype = scale_dtype
        self.data_dtype = data_dtype

        # Extended Config
        self.scale_type = scale_dtype  # Default
        self.scale_round_mode = "TiesToEven"
        self.round_mode = "TiesToEven"
        self.with_rht = False
        self.rht_algo = "fwht"
        self.eps = 0.0
        self.with_random_sign_mask = True
        self.use_2d_weights = False
        self.encode_centric = False

        if mx_config:

            def get_cfg(name, default):
                return getattr(mx_config, name) if hasattr(mx_config, name) else default

            self.block_size = get_cfg("block_size", self.block_size)
            self.scale_max = get_cfg("scale_max", self.scale_max)
            self.use_global_scale = get_cfg("use_global_scale", self.use_global_scale)

            self.scale_type = get_cfg("scale_type", "E4M3")
            self.scale_round_mode = get_cfg("scale_round_mode", "TiesToEven")
            self.round_mode = get_cfg("roundMode", "TiesToEven")
            self.with_rht = get_cfg("use_rht", False)
            self.rht_algo = get_cfg("rht_algo", "fwht")  # Load Algo
            self.eps = get_cfg("eps", 0.0)
            self.with_random_sign_mask = get_cfg("with_random_sign_mask", True)
            self.use_2d_weights = get_cfg("use_2d_weights", False)
            self.encode_centric = get_cfg("encode_centric", False)
            self.use_fp32_matmul = get_cfg("use_fp32_matmul", False)

            if get_cfg("use_bf16_data", True):
                self.data_dtype = torch.bfloat16
            else:
                self.data_dtype = torch.float32

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.023)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # RHT Buffers
        if self.with_rht:
            if self.in_features % self.block_size != 0:
                raise ValueError("in_features must be divisible by block_size for RHT")

    @classmethod
    def from_float(cls, mod, config):
        # Determine if we should propagate use_dequant_gemm if it exists on mod?
        # Usually from_float acts on standard Linear, which doesn't have it.
        # But we can check config for it maybe?
        # Priority: explicit arg > config object > default

        new_mod = cls(
            mod.in_features,
            mod.out_features,
            data_dtype=torch.float32,
            bias=mod.bias is not None,
            mx_config=config,
        )
        new_mod = new_mod.to(mod.weight.device).to(mod.weight.dtype)
        with torch.no_grad():
            new_mod.weight.copy_(mod.weight)
            if mod.bias is not None:
                new_mod.bias.copy_(mod.bias)
        return new_mod

    def forward(self, input):
        x = input
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x = x.view(B * S, H)

        # Pack params
        params = {
            "scale_max": self.scale_max,
            "use_global_scale": self.use_global_scale,
            "block_size": self.block_size,
            "scale_type": self.scale_type,
            "data_round_mode_val": get_round_mode_constant(self.round_mode),
            "scale_round_mode_val": get_round_mode_constant(self.scale_round_mode),
            "data_dtype": self.data_dtype,
            # RHT
            "with_rht": self.with_rht,
            "rht_algo": self.rht_algo,  # Pass Algo
            "with_random_sign_mask": self.with_random_sign_mask,
            "use_2d_weights": self.use_2d_weights,  # For W
            "encode_centric": self.encode_centric,
            "roundMode": self.round_mode,
            "scale_round_mode": self.scale_round_mode,
            "use_fp32_matmul": self.use_fp32_matmul,
        }

        y = TritonFusedQuantLinearFunction.apply(x, self.weight, self.bias, params)
        if is_3d:
            y = y.view(B, S, self.out_features)
        return y
