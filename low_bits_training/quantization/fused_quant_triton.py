"""
Triton-based Fused Quantization-Matmul Linear Layer.
Re-implemented for exact bit parity with Reference quantization_custom_triton.py.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice
from typing import Optional, Any, Tuple
from dataclasses import dataclass


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


# Pre-defined format infos
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
    max_val=61440.0,  # 1.875 * 2^16 = 122880.0
    min_val=0.0,
    precision=4,  # 3 explicit + 1 implicit
    bias=15,
    has_subnormals=True,
    is_signed=False,  # Unsigned per OCP
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
    min_val=6.103515625e-05,  # 2^-14
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


# Define RoundMode constants to match gfloat/standard enums
# You should ensure these integers match the Enum values passed to the kernel


@triton.jit
def _ldexp(v, s):
    # Split shift strategy for stability
    offset = 24.0
    s_f = s.to(tl.float32)
    pow_s_minus_off = tl.exp2(s_f - offset)
    v_scaled_pos = v * 16777216.0  # 2^24
    vlo = v_scaled_pos * pow_s_minus_off
    pow_s_plus_off = tl.exp2(s_f + offset)
    v_scaled_neg = v * 5.9604645e-8  # 2^-24
    vhi = v_scaled_neg * pow_s_plus_off
    return tl.where(tl.abs(v) < 1.0, vlo, vhi)


@triton.jit
def _isodd_int(v):
    return (v & 1) != 0


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
def _round_float_kernel_impl(
    x,
    fi_precision: tl.constexpr,
    fi_bias: tl.constexpr,
    fi_has_subnormals: tl.constexpr,
    fi_max: tl.constexpr,
    fi_min: tl.constexpr,
    fi_is_signed: tl.constexpr,
    fi_has_nz: tl.constexpr,
    fi_has_infs: tl.constexpr,
    fi_num_nans: tl.constexpr,
    round_mode: tl.constexpr,
    srbits,
    srnumbits: tl.constexpr,
):
    """
    Implementation of rounding logic.
    NOTE: round_mode comparisons use integer literals to avoid Triton JIT
    global variable visibility errors.
    0=TiesToEven, 1=TowardZero, 2=TowardPositive, 3=TowardNegative,
    4=TiesToAway, 5=Stochastic, 6=StochasticOdd, 7=StochasticFast, 8=StochasticFastest
    """

    # 1. Setup constants and masks
    p = fi_precision

    # Detect special values
    is_nan = x != x
    is_inf = tl.abs(x) == float("inf")
    is_zero = x == 0.0
    finite_nonzero = ~(is_nan | is_inf | is_zero)

    # 2. Extract Sign and Absolute Value
    is_negative_val = x < 0.0
    if fi_is_signed:
        is_negative = is_negative_val
    else:
        is_negative = 0  # False

    absv = tl.abs(x)
    absv = tl.where(is_negative, -x, x)  # Ensure positive magnitude

    # Mask absv for log2 calculation to avoid log2(0)
    absv_masked = tl.where(finite_nonzero, absv, 1.0)

    # 3. Calculate Exponent (expval)
    log2_val = tl.log2(absv_masked)
    expval = tl.floor(log2_val).to(tl.int32)

    if fi_has_subnormals:
        min_exp = 1 - fi_bias
        expval = tl.maximum(expval, min_exp)

    expval = expval - p + 1

    # 4. Extract Significand
    fsignificand = _ldexp(absv_masked, -expval)

    floorfsignificand = tl.floor(fsignificand)
    isignificand = floorfsignificand.to(tl.int64)
    delta = fsignificand - floorfsignificand

    # 5. Odd Check (for TiesToEven)
    if fi_precision > 1:
        code_is_odd = _isodd_int(isignificand)
    else:
        exp_bias_sum = expval + fi_bias
        code_is_odd = (isignificand != 0) & _isodd_int(exp_bias_sum)

    # 6. Determine Rounding Direction (should_round_away)
    should_round_away = 0  # Default False

    # RM_TOWARD_ZERO = 1
    if round_mode == 1:
        should_round_away = 0

    # RM_TOWARD_POSITIVE = 2
    elif round_mode == 2:
        # ~is_negative & (delta > 0)
        should_round_away = (~is_negative) & (delta > 0.0)

    # RM_TOWARD_NEGATIVE = 3
    elif round_mode == 3:
        # is_negative & (delta > 0)
        should_round_away = is_negative & (delta > 0.0)

    # RM_TIES_TO_AWAY = 4
    elif round_mode == 4:
        should_round_away = delta >= 0.5

    # RM_TIES_TO_EVEN = 0
    elif round_mode == 0:
        # (delta > 0.5) | ((delta == 0.5) & code_is_odd)
        should_round_away = (delta > 0.5) | ((delta == 0.5) & code_is_odd)

    # RM_STOCHASTIC = 5
    elif round_mode == 5:
        # d = delta * 2**srnumbits
        scale = tl.exp2(float(srnumbits))
        d = delta * scale
        floord = tl.floor(d)
        dd = d - floord
        floord_int = floord.to(tl.int64)

        # should_round_away_tne = (dd > 0.5) | ((dd == 0.5) & _isodd(floord))
        tne_cond = (dd > 0.5) | ((dd == 0.5) & _isodd_int(floord_int))

        drnd = floord_int + tne_cond.to(tl.int64)

        # should_round_away = drnd + srbits >= 2**srnumbits
        limit = 1 << srnumbits
        should_round_away = (drnd + srbits) >= limit

    # RM_STOCHASTIC_ODD = 6
    elif round_mode == 6:
        scale = tl.exp2(float(srnumbits))
        d = delta * scale
        floord = tl.floor(d)
        dd = d - floord
        floord_int = floord.to(tl.int64)

        # TNO: (dd > 0.5) | ((dd == 0.5) & ~_isodd(floord))
        tno_cond = (dd > 0.5) | ((dd == 0.5) & (~_isodd_int(floord_int)))

        drnd = floord_int + tno_cond.to(tl.int64)

        limit = 1 << srnumbits
        should_round_away = (drnd + srbits) >= limit

    # RM_STOCHASTIC_FAST = 7
    elif round_mode == 7:
        # delta + (2*srbits + 1) * 2**-(1+srnumbits) >= 1.0
        term = (2 * srbits + 1).to(tl.float32)
        pow_term = tl.exp2(-1.0 - float(srnumbits))
        should_round_away = (delta + term * pow_term) >= 1.0

    # RM_STOCHASTIC_FASTEST = 8
    elif round_mode == 8:
        # delta + srbits * 2**-srnumbits >= 1.0
        term = srbits.to(tl.float32)
        pow_term = tl.exp2(-float(srnumbits))
        should_round_away = (delta + term * pow_term) >= 1.0

    # Apply rounding
    isignificand = isignificand + should_round_away.to(tl.int64)

    # 7. Reconstruct Result
    fresult = _ldexp(isignificand.to(tl.float32), expval)

    # Restore non-finite values (masked logic)
    result = tl.where(finite_nonzero, fresult, absv)

    # 8. Saturation / Overflow Handling
    # amax determination
    amax_val = tl.where(is_negative, -fi_min, fi_max)

    # For formats without infs and without NaNs, we must saturate to amax
    # For all rounding modes, results > amax should be clamped
    is_overflow = result > amax_val

    # Decide what to do with overflow based on format capabilities
    if fi_has_infs:
        # Format supports infinity - convert overflow to inf
        result = tl.where(is_overflow, float("inf"), result)
    elif fi_num_nans > 0:
        # Format supports NaN but not inf - convert overflow to NaN
        result = tl.where(is_overflow, float("nan"), result)
    else:
        # Format doesn't support inf or NaN - saturate to amax
        result = tl.where(is_overflow, amax_val, result)

    # 9. Final Sign and Zero Handling
    result = tl.where(is_negative, -result, result)

    # Negative Zero Handling
    if fi_has_nz:
        # Logic handled by sign bit preservation above
        pass
    else:
        result = tl.where(result == 0.0, 0.0, result)

    return result


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
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    if signs_ptr is not None:
        signs = tl.load(signs_ptr + tl.arange(0, BLOCK_K))
        x = x * signs[None, :]

    # FWHT butterfly (16)
    idx_mask = tl.arange(0, 2).reshape(1, 1, 2, 1)
    merge_idx = tl.arange(0, 2).reshape(2, 1, 1, 1)
    scale = 0.25

    # Stride 8
    x_r = x.reshape(BLOCK_M, 1, 2, 8)
    a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
    b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
    sp = tl.permute((a + b).reshape(BLOCK_M, 1, 1, 8), [2, 0, 1, 3])
    dp = tl.permute((a - b).reshape(BLOCK_M, 1, 1, 8), [2, 0, 1, 3])
    x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(BLOCK_M, 16)

    # Stride 4
    x_r = x.reshape(BLOCK_M, 2, 2, 4)
    a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
    b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
    sp = tl.permute((a + b).reshape(BLOCK_M, 2, 1, 4), [2, 0, 1, 3])
    dp = tl.permute((a - b).reshape(BLOCK_M, 2, 1, 4), [2, 0, 1, 3])
    x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(BLOCK_M, 16)

    # Stride 2
    x_r = x.reshape(BLOCK_M, 4, 2, 2)
    a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
    b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
    sp = tl.permute((a + b).reshape(BLOCK_M, 4, 1, 2), [2, 0, 1, 3])
    dp = tl.permute((a - b).reshape(BLOCK_M, 4, 1, 2), [2, 0, 1, 3])
    x = tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(BLOCK_M, 16)

    # Stride 1
    x_r = x.reshape(BLOCK_M, 8, 2, 1)
    a = tl.sum(tl.where(idx_mask == 0, x_r, 0.0), axis=2)
    b = tl.sum(tl.where(idx_mask == 1, x_r, 0.0), axis=2)
    sp = tl.permute((a + b).reshape(BLOCK_M, 8, 1, 1), [2, 0, 1, 3])
    dp = tl.permute((a - b).reshape(BLOCK_M, 8, 1, 1), [2, 0, 1, 3])
    x_out = (
        tl.permute(tl.where(merge_idx == 0, sp, dp), [1, 2, 0, 3]).reshape(BLOCK_M, 16)
        * scale
    )

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, x_out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_quant_matmul_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    global_amax_a_ptr,
    global_amax_b_ptr,
    srbits_a_ptr,
    srbits_b_ptr,
    # RHT Pointers
    signs_a_ptr,
    signs_b_ptr,  # Removed h_ptr
    # Dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Quant Params
    scale_max_a,
    scale_max_b,
    # A Format (Data)
    a_prec: tl.constexpr,
    a_bias: tl.constexpr,
    a_has_sub: tl.constexpr,
    a_max: tl.constexpr,
    a_min: tl.constexpr,
    a_signed: tl.constexpr,
    a_nz: tl.constexpr,
    a_inf: tl.constexpr,
    a_nan: tl.constexpr,
    # B Format (Data)
    b_prec: tl.constexpr,
    b_bias: tl.constexpr,
    b_has_sub: tl.constexpr,
    b_max: tl.constexpr,
    b_min: tl.constexpr,
    b_signed: tl.constexpr,
    b_nz: tl.constexpr,
    b_inf: tl.constexpr,
    b_nan: tl.constexpr,
    # Scale Format parameter (A)
    sa_prec: tl.constexpr,
    sa_bias: tl.constexpr,
    sa_has_sub: tl.constexpr,
    sa_max: tl.constexpr,
    sa_min: tl.constexpr,
    sa_signed: tl.constexpr,
    sa_nz: tl.constexpr,
    sa_inf: tl.constexpr,
    sa_nan: tl.constexpr,
    # Scale Format parameter (B)
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
    ROTATION_SIZE: tl.constexpr,  # Added
    rht_on_features: tl.constexpr = False,  # Default False
    encode_centric: tl.constexpr = False,  # Encode-centric (inverse) quantization
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    srbits_a_ptrs = None
    if srbits_a_ptr is not None:
        srbits_a_ptrs = srbits_a_ptr + (
            offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        )

    srbits_b_ptrs = None
    if srbits_b_ptr is not None:
        srbits_b_ptrs = srbits_b_ptr + (
            offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk
        )

    # Global Scale logic matching Reference
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
        # Safe division for ges_a
        g_amax_a_safe = tl.where(tl.abs(g_amax_a_f32) < 1e-9, 1.0, g_amax_a_f32)
        ges_a = tl.extra.cuda.libdevice.div_rn(factor_a_f32, g_amax_a_safe)

        max_f32_t = tl.full(ges_a.shape, max_f32_val, tl.float32)
        ges_a = tl.minimum(ges_a, max_f32_t)
        one_f32_t = tl.full(ges_a.shape, one_val, tl.float32)

        # If amax was effectively 0, default scale to 1.0
        ges_a = tl.where(tl.abs(g_amax_a_f32) < 1e-9, one_f32_t, ges_a)

        # Safe division for gds_a (inverse)
        ges_a_safe = tl.where(tl.abs(ges_a) < 1e-9, 1.0, ges_a)
        gds_a = tl.extra.cuda.libdevice.div_rn(one_f32_t, ges_a_safe)

        g_dec_a = tl.where(tl.abs(ges_a) < 1e-9, one_f32_t, gds_a)
        g_enc_a = ges_a

        # B
        g_amax_b = tl.load(global_amax_b_ptr)
        factor_b = scale_max_b * b_max
        g_amax_b_f32 = g_amax_b.to(tl.float32)
        factor_b_f32 = tl.full(g_amax_b.shape, factor_b, tl.float32)
        # Safe division for ges_b
        g_amax_b_safe = tl.where(tl.abs(g_amax_b_f32) < 1e-9, 1.0, g_amax_b_f32)
        ges_b = tl.extra.cuda.libdevice.div_rn(factor_b_f32, g_amax_b_safe)

        max_f32_b_t = tl.full(ges_b.shape, max_f32_val, tl.float32)
        ges_b = tl.minimum(ges_b, max_f32_b_t)
        one_f32_b_t = tl.full(ges_b.shape, one_val, tl.float32)

        # If amax was effectively 0, default scale to 1.0
        ges_b = tl.where(tl.abs(g_amax_b_f32) < 1e-9, one_f32_b_t, ges_b)

        # Safe division for gds_b (inverse)
        ges_b_safe = tl.where(tl.abs(ges_b) < 1e-9, 1.0, ges_b)
        gds_b = tl.extra.cuda.libdevice.div_rn(one_f32_b_t, ges_b_safe)

        g_dec_b = tl.where(tl.abs(ges_b) < 1e-9, one_f32_b_t, gds_b)
        g_enc_b = ges_b

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_base = k * BLOCK_K

        # Load mask
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K - k_base)
        mask_b = (offs_bn[None, :] < N) & (offs_k[:, None] < K - k_base)

        # Load Data + Cast to FP32
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)

        srbits_a = tl.full(a.shape, 0, dtype=tl.int32)
        srbits_b = tl.full(b.shape, 0, dtype=tl.int32)

        if srbits_a_ptrs is not None:
            srbits_a = tl.load(srbits_a_ptrs, mask=mask_a, other=0)

        if srbits_b_ptrs is not None:
            srbits_b = tl.load(srbits_b_ptrs, mask=mask_b, other=0)

        # --- A ---
        a_max_val = tl.max(tl.abs(a), axis=1)  # (BLOCK_M,)
        data_max_a_f32 = tl.full(a_max_val.shape, a_max, tl.float32)
        eps_f32 = tl.full(a_max_val.shape, 1e-9, tl.float32)
        is_zero_block_a = a_max_val <= eps_f32

        if encode_centric:
            # ENCODE-CENTRIC: Match CUDA nvfp4_transpose.cuh exactly
            # 1. Compute MULTIPLIER: M = data_max / (amax * S_enc)
            #    CUDA: compute_encoding_scaling_factor_nv(block_amax, S_enc)
            #    Returns: fp4_max / (block_amax * S_enc), or scale_max for zeros

            if use_global_scale:
                # denom = amax * S_enc (multiply first, then divide)
                denom_a = tl.extra.cuda.libdevice.mul_rn(a_max_val, g_enc_a)
                denom_a_safe = tl.where(tl.abs(denom_a) < 1e-9, 1.0, denom_a)
                s_mult_a = tl.extra.cuda.libdevice.div_rn(data_max_a_f32, denom_a_safe)
            else:
                a_max_safe = tl.where(tl.abs(a_max_val) < 1e-9, 1.0, a_max_val)
                s_mult_a = tl.extra.cuda.libdevice.div_rn(data_max_a_f32, a_max_safe)

            # Zero blocks get scale_max (E4M3 max = 448)
            s_mult_a = tl.where(
                is_zero_block_a,
                tl.full(s_mult_a.shape, scale_max_a, tl.float32),
                s_mult_a,
            )

            # Clamp to float_max
            max_f32 = tl.full(s_mult_a.shape, 3.4028235e38, tl.float32)
            s_mult_a = tl.minimum(s_mult_a, max_f32)
            s_mult_a = s_mult_a[:, None]

            # 2. Round multiplier to scale format (E4M3/E8M0)
            srbits_sa = tl.full((BLOCK_M, 1), 0, tl.int32)
            if srbits_a_ptrs is not None:
                srbits_sa_ptrs = srbits_a_ptr + (offs_am * stride_am)
                srbits_sa_val = tl.load(srbits_sa_ptrs, mask=offs_am < M, other=0)
                srbits_sa = srbits_sa_val[:, None]

            s_mult_a_rounded = _round_float_kernel_impl(
                s_mult_a,
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

            # 3. block_scale_inverse = M_rounded * S_enc (for quantization)
            #    CUDA: block_scale_inverse = static_cast<float>(S_mult_fp8) * S_enc_colwise;
            if use_global_scale:
                es_a = tl.extra.cuda.libdevice.mul_rn(s_mult_a_rounded, g_enc_a)
            else:
                es_a = s_mult_a_rounded

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

            # 4. Store reciprocal for dequant: S_b = 1/M_rounded
            #    CUDA: S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
            one_f32 = tl.full(s_mult_a_rounded.shape, 1.0, tl.float32)
            s_mult_safe = tl.where(tl.abs(s_mult_a_rounded) < 1e-9, 1.0, s_mult_a_rounded)
            s_a = tl.extra.cuda.libdevice.div_rn(one_f32, s_mult_safe)
            a_dq = a_q * s_a

        else:
            # DECODE-CENTRIC (Default): Match CUDA compute_decoding_scaling_factor
            # 1. Compute DIVISOR: D = amax / data_max * S_enc
            tmp_a = tl.extra.cuda.libdevice.div_rn(a_max_val, data_max_a_f32)

            s_a = tmp_a
            if use_global_scale:
                s_a = tl.extra.cuda.libdevice.mul_rn(tmp_a, g_enc_a)

            # Zero blocks get 0
            s_a = tl.where(is_zero_block_a, tl.full(s_a.shape, 0.0, tl.float32), s_a)
            s_a = s_a[:, None]

            # 2. Round divisor to scale format
            srbits_sa = tl.full((BLOCK_M, 1), 0, tl.int32)
            if srbits_a_ptrs is not None:
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

            # 3. block_scale_inverse = 1 / (D_rounded * S_dec)
            if use_global_scale:
                denom_a = tl.extra.cuda.libdevice.mul_rn(s_a, g_dec_a)
                denom_a_safe = tl.where(tl.abs(denom_a) < 1e-9, 1.0, denom_a)
                one_f32_bcast = tl.full(denom_a.shape, 1.0, tl.float32)
                es_a = tl.extra.cuda.libdevice.div_rn(one_f32_bcast, denom_a_safe)
            else:
                s_a_safe = tl.where(tl.abs(s_a) < 1e-9, 1.0, s_a)
                one_f32_bcast = tl.full(s_a.shape, 1.0, tl.float32)
                es_a = tl.extra.cuda.libdevice.div_rn(one_f32_bcast, s_a_safe)

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

            # 4. Dequant: dq = q * D_rounded
            a_dq = a_q * s_a

        # --- B ---
        # 2D Weight Quantization uses Scalar Block Max if enabled
        if use_2d_b:
            b_scalar_max = tl.max(tl.abs(b))  # Scalar
            b_reshaped = tl.reshape(b, (BLOCK_K, BLOCK_N // 16, 16))

            m1 = tl.max(tl.abs(b_reshaped), axis=0)  # (num_chunks, 16)
            m2 = tl.max(m1, axis=1)  # (num_chunks,)
            m2_exp = m2[:, None]
            one_16 = tl.full((1, 16), 1.0, tl.float32)
            vals = m2_exp * one_16  # (num_chunks, 16)
            b_max_val = tl.reshape(vals, (BLOCK_N,))

        else:
            b_max_val = tl.max(tl.abs(b), axis=0)  # (BLOCK_N,)

        data_max_b_f32 = tl.full(b_max_val.shape, b_max, tl.float32)
        eps_f32_b = tl.full(b_max_val.shape, 1e-9, tl.float32)
        is_zero_block_b = b_max_val <= eps_f32_b

        if encode_centric:
            # ENCODE-CENTRIC: Match CUDA nvfp4_transpose.cuh exactly
            # 1. Compute MULTIPLIER: M = data_max / (amax * S_enc)

            if use_global_scale:
                denom_b = tl.extra.cuda.libdevice.mul_rn(b_max_val, g_enc_b)
                denom_b_safe = tl.where(tl.abs(denom_b) < 1e-9, 1.0, denom_b)
                s_mult_b = tl.extra.cuda.libdevice.div_rn(data_max_b_f32, denom_b_safe)
            else:
                b_max_safe = tl.where(tl.abs(b_max_val) < 1e-9, 1.0, b_max_val)
                s_mult_b = tl.extra.cuda.libdevice.div_rn(data_max_b_f32, b_max_safe)

            # Zero blocks get scale_max
            s_mult_b = tl.where(
                is_zero_block_b,
                tl.full(s_mult_b.shape, scale_max_b, tl.float32),
                s_mult_b,
            )

            # Clamp to float_max
            max_f32_b = tl.full(s_mult_b.shape, 3.4028235e38, tl.float32)
            s_mult_b = tl.minimum(s_mult_b, max_f32_b)
            s_mult_b = s_mult_b[None, :]

            # 2. Round multiplier to scale format
            srbits_sb = tl.full((1, BLOCK_N), 0, tl.int32)
            if srbits_b_ptrs is not None:
                srbits_sb_ptrs = srbits_b_ptr + (offs_bn * stride_bn)
                srbits_sb_val = tl.load(srbits_sb_ptrs, mask=offs_bn < N, other=0)
                srbits_sb = srbits_sb_val[None, :]

            s_mult_b_rounded = _round_float_kernel_impl(
                s_mult_b,
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

            # 3. block_scale_inverse = M_rounded * S_enc
            if use_global_scale:
                es_b = tl.extra.cuda.libdevice.mul_rn(s_mult_b_rounded, g_enc_b)
            else:
                es_b = s_mult_b_rounded

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

            # 4. Store reciprocal for dequant: S_b = 1/M_rounded
            one_f32_b = tl.full(s_mult_b_rounded.shape, 1.0, tl.float32)
            s_mult_b_safe = tl.where(
                tl.abs(s_mult_b_rounded) < 1e-9, 1.0, s_mult_b_rounded
            )
            s_b = tl.extra.cuda.libdevice.div_rn(one_f32_b, s_mult_b_safe)
            b_dq = b_q * s_b

        else:
            # DECODE-CENTRIC (Default)
            tmp_b = tl.extra.cuda.libdevice.div_rn(b_max_val, data_max_b_f32)

            s_b = tmp_b
            if use_global_scale:
                s_b = tl.extra.cuda.libdevice.mul_rn(tmp_b, g_enc_b)

            # Zero blocks get 0
            s_b = tl.where(is_zero_block_b, tl.full(s_b.shape, 0.0, tl.float32), s_b)
            s_b = s_b[None, :]

            srbits_sb = tl.full((1, BLOCK_N), 0, tl.int32)
            if srbits_b_ptrs is not None:
                srbits_sb_ptrs = srbits_b_ptr + (offs_bn * stride_bn)
                srbits_sb_val = tl.load(srbits_sb_ptrs, mask=offs_bn < N, other=0)
                srbits_sb = srbits_sb_val[None, :]

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

            if use_global_scale:
                denom_b = tl.extra.cuda.libdevice.mul_rn(s_b, g_dec_b)
                denom_b_safe = tl.where(tl.abs(denom_b) < 1e-9, 1.0, denom_b)
                one_f32_bcast_b = tl.full(denom_b.shape, 1.0, tl.float32)
                es_b = tl.extra.cuda.libdevice.div_rn(one_f32_bcast_b, denom_b_safe)
            else:
                s_b_safe = tl.where(tl.abs(s_b) < 1e-9, 1.0, s_b)
                one_f32_bcast_b = tl.full(s_b.shape, 1.0, tl.float32)
                es_b = tl.extra.cuda.libdevice.div_rn(one_f32_bcast_b, s_b_safe)

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

            # Dequant: dq = q * D_rounded
            b_dq = b_q * s_b

        # Matmul
        acc += tl.dot(a_dq, b_dq)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    if use_global_scale:
        alpha = g_dec_a * g_dec_b
        acc = acc * alpha

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


# ============================================================================
# FUNCTION
# ============================================================================


# ============================================================================
# FUNCTION
# ============================================================================


class TritonFusedQuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, params):
        params_saved = params.copy()

        scale_max = params["scale_max"]
        use_global_scale = params["use_global_scale"]
        block_size = params["block_size"]
        scale_type = params.get("scale_type", "E4M3")

        # Rounding params
        data_rm_val = params.get("data_round_mode_val", 0)
        scale_rm_val = params.get("scale_round_mode_val", 0)
        srbits_a = params.get("srbits_a", None)
        srbits_b = params.get("srbits_b", None)

        # RHT / 2D params
        h_matrix = params.get("h_matrix", None)  # Ignored if FWHT used
        signs_a = None
        signs_b = None
        use_2d_weights = params.get("use_2d_weights", False)
        encode_centric = params.get("encode_centric", False)

        ctx.saved_params = params_saved
        ctx.save_for_backward(input, weight, bias)

        M, K = input.shape
        N = weight.shape[0]
        data_dtype = params.get("data_dtype", torch.float32)
        y = torch.empty((M, N), device=input.device, dtype=data_dtype)

        if use_global_scale:
            BLOCK_SIZE = 1024
            ga_a = torch.zeros(1, device=input.device, dtype=torch.float32)
            triton_amax_kernel[(triton.cdiv(input.numel(), BLOCK_SIZE),)](
                input, ga_a, input.numel(), BLOCK_SIZE=BLOCK_SIZE
            )
            ga_b = torch.zeros(1, device=input.device, dtype=torch.float32)
            triton_amax_kernel[(triton.cdiv(weight.numel(), BLOCK_SIZE),)](
                weight, ga_b, weight.numel(), BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            ga_a = torch.empty(1, device=input.device)
            ga_b = torch.empty(1, device=input.device)

        format_a = FORMAT_E2M1
        format_b = FORMAT_E2M1
        format_sa = get_format_info(scale_type)
        format_sb = get_format_info(scale_type)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )

        rot_size = block_size

        fused_quant_matmul_kernel[grid](
            input,
            weight,
            y,
            ga_a,
            ga_b,
            srbits_a,
            srbits_b,
            signs_a,
            signs_b,  # RHT: Removed h_matrix
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(1),
            weight.stride(0),
            y.stride(0),
            y.stride(1),
            scale_max,
            scale_max,
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
            # Scale A
            format_sa.precision,
            format_sa.bias,
            format_sa.has_subnormals,
            format_sa.max_val,
            format_sa.min_val,
            format_sa.is_signed,
            format_sa.has_nz,
            format_sa.has_infs,
            format_sa.num_nans,
            # Scale B
            format_sb.precision,
            format_sb.bias,
            format_sb.has_subnormals,
            format_sb.max_val,
            format_sb.min_val,
            format_sb.is_signed,
            format_sb.has_nz,
            format_sb.has_infs,
            format_sb.num_nans,
            # Round Modes
            data_rm_val,
            scale_rm_val,  # A
            data_rm_val,
            scale_rm_val,  # B
            use_global_scale,
            use_2d_weights,  # use_2d_b
            BLOCK_K=block_size,
            ROTATION_SIZE=rot_size,  # Added
            rht_on_features=False,  # Forward uses Reduction RHT (K)
            encode_centric=encode_centric,
        )

        if bias is not None:
            y = y + bias
        return y.to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        params = ctx.saved_params
        block_size = params["block_size"]
        scale_max = params["scale_max"]
        use_global_scale = params["use_global_scale"]
        scale_type = params.get("scale_type", "E4M3")
        data_rm_val = params.get("data_round_mode_val", 0)
        scale_rm_val = params.get("scale_round_mode_val", 0)
        use_2d_weights = params.get("use_2d_weights", False)
        data_dtype = params.get("data_dtype", torch.float32)

        # RHT params from Forward
        signs_a = params.get("signs_a", None)
        signs_b = params.get("signs_b", None)
        with_rht = params.get("with_rht", False)
        with_random_sign_mask = params.get("with_random_sign_mask", True)
        encode_centric = params.get("encode_centric", False)

        # Generation of Backward SRBits
        is_stoch = data_rm_val >= 5 or scale_rm_val >= 5
        srbits_dy = None
        srbits_dw = None
        srbits_dx = None

        if is_stoch:
            srbits_dy = torch.randint(
                0, 256, grad_output.shape, device=grad_output.device, dtype=torch.int32
            )
            srbits_dw = torch.randint(
                0, 256, weight.shape, device=weight.device, dtype=torch.int32
            )
            srbits_dx = torch.randint(
                0, 256, input.shape, device=input.device, dtype=torch.int32
            )

        dY = grad_output.contiguous()
        X = input.contiguous()
        W = weight.contiguous()

        M, N = dY.shape
        _, K = W.shape

        dX = torch.empty((M, K), device=dY.device, dtype=data_dtype)
        dW = torch.empty((N, K), device=dY.device, dtype=data_dtype)

        # Global Amax Helper
        def compute_amax(t):
            BLOCK_SIZE = 1024
            ga = torch.zeros(1, device=t.device, dtype=torch.float32)
            triton_amax_kernel[(triton.cdiv(t.numel(), BLOCK_SIZE),)](
                t, ga, t.numel(), BLOCK_SIZE=BLOCK_SIZE
            )
            return ga

        if use_global_scale:
            ga_dy = compute_amax(dY)
            ga_w = compute_amax(W)
            ga_x = compute_amax(X)
        else:
            ga_dy = torch.empty(1, device=dY.device)
            ga_w = torch.empty(1, device=dY.device)
            ga_x = torch.empty(1, device=dY.device)

        format_a = FORMAT_E2M1
        format_b = FORMAT_E2M1
        format_sa = get_format_info(scale_type)
        format_sb = get_format_info(scale_type)

        rot_size = block_size

        # dX (dgrad) = dY @ W
        # User confirmed: Hadamard is NOT applied to dY (dgrad).
        # So disable RHT for dX.

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_N"]),
        )
        fused_quant_matmul_kernel[grid](
            dY,
            W,
            dX,
            ga_dy,
            ga_w,
            srbits_dy,
            srbits_dw,
            None,
            None,  # RHT Disabled for dX
            M,
            K,
            N,
            dY.stride(0),
            dY.stride(1),
            W.stride(0),
            W.stride(1),
            dX.stride(0),
            dX.stride(1),
            scale_max,
            scale_max,
            format_a.precision,
            format_a.bias,
            format_a.has_subnormals,
            format_a.max_val,
            format_a.min_val,
            format_a.is_signed,
            format_a.has_nz,
            format_a.has_infs,
            format_a.num_nans,
            format_b.precision,
            format_b.bias,
            format_b.has_subnormals,
            format_b.max_val,
            format_b.min_val,
            format_b.is_signed,
            format_b.has_nz,
            format_b.has_infs,
            format_b.num_nans,
            format_sa.precision,
            format_sa.bias,
            format_sa.has_subnormals,
            format_sa.max_val,
            format_sa.min_val,
            format_sa.is_signed,
            format_sa.has_nz,
            format_sa.has_infs,
            format_sa.num_nans,
            format_sb.precision,
            format_sb.bias,
            format_sb.has_subnormals,
            format_sb.max_val,
            format_sb.min_val,
            format_sb.is_signed,
            format_sb.has_nz,
            format_sb.has_infs,
            format_sb.num_nans,
            data_rm_val,
            scale_rm_val,
            data_rm_val,
            scale_rm_val,
            use_global_scale,
            use_2d_weights,  # use_2d_b for W
            BLOCK_K=block_size,
            ROTATION_SIZE=rot_size,
            rht_on_features=False,
            encode_centric=encode_centric,
        )
        grad_input = dX.to(input.dtype)

        # dW (wgrad) = dY.T @ X = (N, M) @ (M, K) = (N, K)
        # TE's asymmetric RHT flow (from te_parity_linear_tex.py use_dequant_gemm path):
        # - dY uses _rowwise_data (NO RHT)
        # - X uses _columnwise_data (WITH RHT applied to X.T)

        if with_rht:
            if with_random_sign_mask:
                WGRAD_SIGNS = torch.tensor(
                    [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
                    dtype=torch.float32,
                    device=dY.device,
                )
            else:
                WGRAD_SIGNS = torch.ones(
                    block_size, dtype=torch.float32, device=dY.device
                )

            BLOCK_M_RHT = 64

            # Correct wgrad RHT: (H @ dY).T @ (H @ X) = (N, K)
            # H @ dY: Apply FWHT to COLUMNS of dY (M, N) -> (M, N)
            # To apply H to columns: transpose -> FWHT on last dim -> transpose back
            dY_t = dY.t().contiguous()  # (N, M)
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

            # H @ X: Apply FWHT to COLUMNS of X (M, K) -> (M, K)
            X_t = X.t().contiguous()  # (K, M)
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
            X_rht = X_t_fwht.t().contiguous()  # (M, K) = H @ X

            # Compute amax on RHT'd data
            ga_dy_wgrad = compute_amax(dY_t_fwht)
            ga_x_wgrad = compute_amax(X_rht)
            grid = lambda META: (
                triton.cdiv(N, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_N"]),
            )
            fused_quant_matmul_kernel[grid](
                dY_t_fwht,
                X_rht,
                dW,
                ga_dy_wgrad,
                ga_x_wgrad,
                srbits_dy,
                srbits_dx,
                None,
                None,  # No kernel RHT, already applied
                N,
                K,
                M,
                dY_t_fwht.stride(0),
                dY_t_fwht.stride(1),  # dY_rht.T via swapped strides
                X_rht.stride(0),
                X_rht.stride(1),  # X_rht normal strides
                dW.stride(0),
                dW.stride(1),
                scale_max,
                scale_max,
                format_a.precision,
                format_a.bias,
                format_a.has_subnormals,
                format_a.max_val,
                format_a.min_val,
                format_a.is_signed,
                format_a.has_nz,
                format_a.has_infs,
                format_a.num_nans,
                format_b.precision,
                format_b.bias,
                format_b.has_subnormals,
                format_b.max_val,
                format_b.min_val,
                format_b.is_signed,
                format_b.has_nz,
                format_b.has_infs,
                format_b.num_nans,
                format_sa.precision,
                format_sa.bias,
                format_sa.has_subnormals,
                format_sa.max_val,
                format_sa.min_val,
                format_sa.is_signed,
                format_sa.has_nz,
                format_sa.has_infs,
                format_sa.num_nans,
                format_sb.precision,
                format_sb.bias,
                format_sb.has_subnormals,
                format_sb.max_val,
                format_sb.min_val,
                format_sb.is_signed,
                format_sb.has_nz,
                format_sb.has_infs,
                format_sb.num_nans,
                data_rm_val,
                scale_rm_val,
                data_rm_val,
                scale_rm_val,
                use_global_scale,
                False,  # use_2d_b for X
                BLOCK_K=block_size,
                ROTATION_SIZE=rot_size,
                rht_on_features=False,  # RHT on reduction dim for cancellation
                encode_centric=encode_centric,
            )
            grad_weight = dW.to(weight.dtype)
        else:
            dY_rht = dY.float()
            X_rht = X.float()
            ga_dy_wgrad = ga_dy
            ga_x_wgrad = ga_x

            # Matmul: (H @ dY).T @ (H @ X) = dY_rht.T @ X_rht = (N, M) @ (M, K) = (N, K)
            # Use ORIGINAL stride pattern: dY_rht with swapped strides (transpose), X_rht normal
            grid = lambda META: (
                triton.cdiv(N, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_N"]),
            )
            fused_quant_matmul_kernel[grid](
                dY_rht,
                X_rht,
                dW,
                ga_dy_wgrad,
                ga_x_wgrad,
                srbits_dy,
                srbits_dx,
                None,
                None,  # No kernel RHT, already applied
                N,
                K,
                M,
                dY_rht.stride(1),
                dY_rht.stride(0),  # dY_rht.T via swapped strides
                X_rht.stride(0),
                X_rht.stride(1),  # X_rht normal strides
                dW.stride(0),
                dW.stride(1),
                scale_max,
                scale_max,
                format_a.precision,
                format_a.bias,
                format_a.has_subnormals,
                format_a.max_val,
                format_a.min_val,
                format_a.is_signed,
                format_a.has_nz,
                format_a.has_infs,
                format_a.num_nans,
                format_b.precision,
                format_b.bias,
                format_b.has_subnormals,
                format_b.max_val,
                format_b.min_val,
                format_b.is_signed,
                format_b.has_nz,
                format_b.has_infs,
                format_b.num_nans,
                format_sa.precision,
                format_sa.bias,
                format_sa.has_subnormals,
                format_sa.max_val,
                format_sa.min_val,
                format_sa.is_signed,
                format_sa.has_nz,
                format_sa.has_infs,
                format_sa.num_nans,
                format_sb.precision,
                format_sb.bias,
                format_sb.has_subnormals,
                format_sb.max_val,
                format_sb.min_val,
                format_sb.is_signed,
                format_sb.has_nz,
                format_sb.has_infs,
                format_sb.num_nans,
                data_rm_val,
                scale_rm_val,
                data_rm_val,
                scale_rm_val,
                use_global_scale,
                False,  # use_2d_b for X
                BLOCK_K=block_size,
                ROTATION_SIZE=rot_size,
                rht_on_features=False,  # RHT on reduction dim for cancellation
                encode_centric=encode_centric,
            )
            grad_weight = dW.to(weight.dtype)

        if bias is not None:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None


# ============================================================================
# FUNCTION
# ============================================================================


class TritonFusedQuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        scale_max: float = 448.0,
        use_global_scale: bool = True,
        block_size: int = 16,
        scale_dtype: torch.dtype = torch.float32,
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
        self.scale_type = "E4M3"  # Default
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
            self.rht_algo = get_cfg("rht_algo", "fwht")
            self.eps = get_cfg("eps", 0.0)
            self.with_random_sign_mask = get_cfg("with_random_sign_mask", True)
            self.use_2d_weights = get_cfg("use_2d_weights", False)
            self.encode_centric = get_cfg("encode_centric", False)

            # Type selection
            if get_cfg("use_bf16_data", True):
                self.data_dtype = torch.bfloat16
            else:
                self.data_dtype = torch.float32  # Or user provided input dtype?

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.023)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Constants
        self.data_rm_val = get_round_mode_constant(self.round_mode)
        self.scale_rm_val = get_round_mode_constant(self.scale_round_mode)

        # RHT Precomputation
        if self.with_rht:
            if self.in_features % self.block_size != 0:
                raise ValueError("in_features must be divisible by block_size for RHT")

            # We assume FWHT inside kernel, no need for dense H matrix
            self.register_buffer("H_matrix", None)
        else:
            self.register_buffer("H_matrix", None)

    @classmethod
    def from_float(cls, mod, config):
        # Determine if we should propagate use_dequant_gemm if it exists on mod?
        # Usually from_float acts on standard Linear, which doesn't have it.
        # But we can check config for it maybe?
        # Priority: explicit arg > config object > default

        new_mod = cls(
            mod.in_features,
            mod.out_features,
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

        w = self.weight

        # RHT Signs        # RHT Logic
        # User requirement: RHT disabled in Forward.
        signs_a = None
        signs_b = None

        if self.with_rht:
            # Generate signs for RHT
            if self.with_random_sign_mask:
                signs_a = (
                    torch.randint(
                        0, 2, (self.block_size,), device=x.device, dtype=torch.float32
                    )
                    * 2
                    - 1
                )
            else:
                signs_a = torch.ones(
                    (self.block_size,), device=x.device, dtype=torch.float32
                )

        # Generate SR Bits if needed
        srbits_a = None
        srbits_b = None

        is_stoch = self.data_rm_val >= 5 or self.scale_rm_val >= 5
        if is_stoch:
            srbits_a = torch.randint(0, 256, x.shape, device=x.device, dtype=torch.int32)
            srbits_b = torch.randint(0, 256, w.shape, device=x.device, dtype=torch.int32)

        params = {
            "scale_max": self.scale_max,
            "use_global_scale": self.use_global_scale,
            "block_size": self.block_size,
            "scale_type": self.scale_type,
            "scale_round_mode": self.scale_round_mode,
            "round_mode": self.round_mode,
            "with_rht": self.with_rht,
            "scale_dtype": self.scale_dtype,
            "data_dtype": self.data_dtype,
            "data_round_mode_val": self.data_rm_val,
            "scale_round_mode_val": self.scale_rm_val,
            "srbits_a": srbits_a,
            "srbits_b": srbits_b,
            "signs_a": signs_a,
            "signs_b": signs_b,
            "use_2d_weights": self.use_2d_weights,  # Passed to Function
            "encode_centric": self.encode_centric,
            "with_random_sign_mask": self.with_random_sign_mask,
        }

        y = TritonFusedQuantLinearFunction.apply(x, w, self.bias, params)

        if is_3d:
            y = y.view(B, S, self.out_features)

        return y.to(input.dtype)
