import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice
from typing import Optional, Any
from dataclasses import dataclass
from transformer_engine.pytorch.experimental.quantization_custom_triton import (
    FORMAT_E2M1,
    FORMAT_E4M3,
    FORMAT_E5M2,
    FORMAT_E8M0,
    FORMAT_E5M3,
    _round_float_kernel_impl,
)

# Constants
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


@dataclass
class TritonFormatInfo:
    max_val: float
    min_val: float
    precision: int
    bias: int
    has_subnormals: bool
    is_signed: bool
    k: int
    has_nz: bool
    has_infs: bool
    num_nans: bool


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
    raise ValueError(f"Unknown format: {name}")


@triton.jit
def debug_fake_quant_simultaneous_kernel(
    # Pointers
    a_ptr,
    a_out_ptr,
    b_ptr,
    b_out_ptr,
    global_amax_a_ptr,
    global_amax_b_ptr,
    srbits_a_ptr,
    srbits_b_ptr,
    # DEBUG POINTERS
    a_q_ptr,
    b_q_ptr,
    sa_ptr,
    sb_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides A
    stride_am,
    stride_ak,
    stride_out_am,
    stride_out_ak,
    stride_aq_m,
    stride_aq_k,
    stride_sa_m,
    stride_sa_k,
    # Strides B
    stride_bn,
    stride_bk,
    stride_out_bn,
    stride_out_bk,
    stride_bq_n,
    stride_bq_k,
    stride_sb_n,
    stride_sb_k,
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
    return_encoded: tl.constexpr = True,
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

        ges_a = tl.extra.cuda.libdevice.div_rn(factor_a_f32, g_amax_a_f32)

        max_f32_t = tl.full(ges_a.shape, max_f32_val, tl.float32)
        ges_a = tl.minimum(ges_a, max_f32_t)
        one_f32_t = tl.full(ges_a.shape, one_val, tl.float32)

        ges_a = tl.where(g_amax_a_f32 == 0.0, one_f32_t, ges_a)

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
            # TE Reference Parity Logic
            # 1. Local Scale: decode_scale = vec_max / a_max
            scale_max_e2m1 = tl.full(a_max_val.shape, a_max, tl.float32)
            decode_scale = a_max_val / scale_max_e2m1

            # 2. Upscale by Global Encode Scale
            if use_global_scale:
                decode_scale = decode_scale * g_enc_a

            # 3. Clamp to Max Range (scale_max_a, e.g. 448.0)
            decode_scale = tl.clamp(decode_scale, -scale_max_a, scale_max_a)

            # 4. Zero Handling
            # If vec_max <= 1e-9, decode_scale = 1.0 / scale_max_a
            # scale_for_zeros needs to match Ref: (1.0 / FLOAT8_E4M3_MAX)
            # =========================================================
            # IN: fake_quant_simultaneous_kernel
            # LOC: ~ line 538
            # =========================================================

            # --- OLD (Fast Version) ---
            # scale_for_zeros = tl.full(decode_scale.shape, 1.0 / scale_max_a, tl.float32)
            # decode_scale = tl.where(is_zero_block_a, scale_for_zeros, decode_scale)

            # --- NEW (Fixed Version) ---
            if encode_centric:
                scale_for_zeros = tl.full(
                    decode_scale.shape, 1.0 / scale_max_a, tl.float32
                )
            else:
                scale_for_zeros = tl.full(decode_scale.shape, 0.0, tl.float32)

            decode_scale = tl.where(is_zero_block_a, scale_for_zeros, decode_scale)

            # 5. Round Scale to Format
            srbits_sa = tl.full((BLOCK_M, 1), 0, tl.int32)
            if srbits_a_ptr is not None:
                srbits_sa_ptrs = srbits_a_ptr + (offs_am * stride_am)
                srbits_sa_val = tl.load(srbits_sa_ptrs, mask=offs_am < M, other=0)
                srbits_sa = srbits_sa_val[:, None]

            decode_scale_rounded = _round_float_kernel_impl(
                decode_scale,
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

            # 6. Calculate Encode Scale (Multiplier)
            # Normal case: encode_scale = 1 / (decode_scale_rounded * global_decode_scale)
            if use_global_scale:
                # Use mul_rn for strict IEEE parity
                denom = tl.extra.cuda.libdevice.mul_rn(decode_scale_rounded, g_dec_a)
            else:
                denom = decode_scale_rounded

            denom = tl.where(tl.abs(denom) < 1e-9, 1.0, denom)
            one_f32 = tl.full(denom.shape, 1.0, tl.float32)

            # Use div_rn for strict IEEE parity
            encode_scale_normal = tl.extra.cuda.libdevice.div_rn(one_f32, denom)

            # Zero Case: scale_max_a * global_encode_scale
            # Cast scale_max_a to float32 just in case
            scale_max_val_t = tl.full(g_enc_a.shape, scale_max_a, tl.float32)
            if use_global_scale:
                encode_scale_zeros = tl.extra.cuda.libdevice.mul_rn(
                    g_enc_a, scale_max_val_t
                )
            else:
                encode_scale_zeros = scale_max_val_t

            # Select
            s_mult_a = tl.where(is_zero_block_a, encode_scale_zeros, encode_scale_normal)

            # Clamp Max Float
            max_f32_clamp = tl.full(s_mult_a.shape, 3.4028235e38, tl.float32)
            s_mult_a = tl.minimum(s_mult_a, max_f32_clamp)
            s_mult_a = s_mult_a[:, None]

            # Store Logic (Save Decode Scale)
            s_a = decode_scale_rounded[:, None]

            # Apply Scaling to Data
            es_a_b = tl.broadcast_to(s_mult_a, a.shape)
            a_scaled = a * es_a_b

            # Quantize Data
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

            # Reconstruct (Local DQ)
            a_dq = a_q * s_a

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

        # DEBUG STORE A
        if a_q_ptr is not None:
            aq_ptrs = a_q_ptr + (
                offs_am[:, None] * stride_aq_m + offs_k[None, :] * stride_aq_k
            )
            tl.store(aq_ptrs, a_q, mask=mask_a)

        if sa_ptr is not None:
            # s_a is (BLOCK_M, 1)
            # Store at (offs_am, pid_k)
            # sa_ptr + offs_am * stride_sa_m + pid_k * stride_sa_k
            sa_ptrs = sa_ptr + (offs_am * stride_sa_m + pid_k * stride_sa_k)
            val_to_store = tl.reshape(s_a, (BLOCK_M,))
            tl.store(sa_ptrs, val_to_store, mask=offs_am < M)

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
            # Reshape to (K_chunks, 16, N_chunks, 16) to support standard 16x16 tiling
            # regardless of BLOCK_K size.
            # Assumes BLOCK_K % 16 == 0 and BLOCK_N % 16 == 0

            # 1. Reshape to 4D to isolate 16x16 tiles
            b_4d = tl.reshape(b, (BLOCK_K // 16, 16, BLOCK_N // 16, 16))

            # 2. Compute Max within tiles (reduce dimensions 1 and 3)
            # Max over rows (axis 1)
            m1 = tl.max(tl.abs(b_4d), axis=1)  # -> (BLOCK_K//16, BLOCK_N//16, 16)
            # Max over cols (axis 2 of result, which was axis 3 of original)
            b_max_val_tiled = tl.max(m1, axis=2)  # -> (BLOCK_K//16, BLOCK_N//16)

            # 3. Broadcast scale back to original shape for "b_max_val" usage?
            # Wait, "b_max_val" variable implies a single vector for 1D?
            # Or is it used as a full tensor later?
            # "data_max_b_f32 = tl.full(b_max_val.shape, b_max, tl.float32)"
            # "b_max_val <= eps_f32_b"
            # "div_rn(b_max_val, ...)"
            # If "b_max_val" is 2D (BLOCK_K//16, BLOCK_N//16), subsequent logic must handle it.
            # But "is_zero_block_b" will be 2D.
            # "decode_scale" will be 2D.
            # "decode_scale_rounded" 2D.
            # "denom" 2D.
            # "es_b" 2D.
            # "es_b_b = tl.broadcast_to(es_b, b.shape)".
            # If es_b is (K/16, N/16), we need to expand it to (K, N).
            # We can do "es_b_expanded = es_b[:, None, :, None] + zeros(4D)" ?
            # Then reshape.

            # Current V2 logic used "b_max_val" as (BLOCK_N,) ???
            # "b_max_val = tl.reshape(vals, (BLOCK_N,))" in old code (Lines 667).
            # That implies OLD code was doing 1D Scaling dependent on columns?
            # Ref C++ computes "out_colwise_scales_sh".
            # If use_2d_b is True, we want 2D scales.
            # My V2 code logic for "es_b_b" broadcast assumes "es_b" broadcasts to "b".
            # If "es_b" is 2D, we need proper expansion.

            # Let's perform the expansion HERE to make "b_max_val" full size (BLOCK_K, BLOCK_N).
            # This simplifies downstream logic (it just works element-wise).
            m_expanded_1 = b_max_val_tiled[:, None, :, None]  # (K/16, 1, N/16, 1)
            one_4d = tl.full((1, 16, 1, 16), 1.0, tl.float32)
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
            # TE Reference Parity Logic for B
            # 1. Local Scale: decode_scale = vec_max / b_max
            scale_max_e2m1 = tl.full(b_max_val.shape, b_max, tl.float32)
            decode_scale = tl.extra.cuda.libdevice.div_rn(b_max_val, scale_max_e2m1)

            # 2. Upscale by Global Encode Scale
            if use_global_scale:
                decode_scale = tl.extra.cuda.libdevice.mul_rn(decode_scale, g_enc_b)

            # 3. Clamp to Max Range (scale_max_b, e.g. 448.0)
            decode_scale = tl.clamp(decode_scale, -scale_max_b, scale_max_b)

            # 4. Zero Handling
            # =========================================================
            # IN: fake_quant_simultaneous_kernel
            # LOC: ~ line 538
            # =========================================================

            # --- OLD (Fast Version) ---
            # scale_for_zeros = tl.full(decode_scale.shape, 1.0 / scale_max_a, tl.float32)
            # decode_scale = tl.where(is_zero_block_a, scale_for_zeros, decode_scale)

            # --- NEW (Fixed Version) ---
            if encode_centric:
                scale_for_zeros = tl.full(
                    decode_scale.shape, 1.0 / scale_max_a, tl.float32
                )
            else:
                scale_for_zeros = tl.full(decode_scale.shape, 0.0, tl.float32)

            decode_scale = tl.where(is_zero_block_b, scale_for_zeros, decode_scale)

            # 5. Round Scale to FP8
            srbits_sb = tl.full((1, BLOCK_N), 0, tl.int32)
            if srbits_b_ptr is not None:
                srbits_sb_ptrs = srbits_b_ptr + (offs_bn * stride_bn)
                srbits_sb_val = tl.load(srbits_sb_ptrs, mask=offs_bn < N, other=0)
                srbits_sb = srbits_sb_val[None, :]

            decode_scale_rounded = _round_float_kernel_impl(
                decode_scale,
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

            # 6. Calculate Encode Scale (Multiplier)
            if use_global_scale:
                denom = decode_scale_rounded * g_dec_b
            else:
                denom = decode_scale_rounded

            denom = tl.where(tl.abs(denom) < 1e-9, 1.0, denom)
            one_f32_b = tl.full(denom.shape, 1.0, tl.float32)
            encode_scale_normal = tl.extra.cuda.libdevice.div_rn(one_f32_b, denom)

            # Zero Case: scale_max_b * global_encode_scale
            scale_max_val_b = tl.full(g_enc_b.shape, scale_max_b, tl.float32)
            if use_global_scale:
                encode_scale_zeros = tl.extra.cuda.libdevice.mul_rn(
                    g_enc_b, scale_max_val_b
                )
            else:
                encode_scale_zeros = scale_max_val_b

            # Select
            s_mult_b = tl.where(is_zero_block_b, encode_scale_zeros, encode_scale_normal)

            # Clamp Max Float
            max_f32_clamp_b = tl.full(s_mult_b.shape, 3.4028235e38, tl.float32)
            s_mult_b = tl.minimum(s_mult_b, max_f32_clamp_b)
            s_mult_b = s_mult_b[None, :]

            # Store Logic (Save Decode Scale)
            s_b = decode_scale_rounded[None, :]

            # Apply Scaling to Data
            es_b_b = tl.broadcast_to(s_mult_b, b.shape)
            b_scaled = b * es_b_b

            # Quantize Data
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

            # Reconstruct
            b_dq = b_q * s_b

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
                # Logic for 1D extraction
                srbits_sb_ptrs = srbits_b_ptr + (offs_bn * stride_bn)
                srbits_sb_val = tl.load(srbits_sb_ptrs, mask=offs_bn < N, other=0)

                # If 1D, unsqeeze. If 2D, we might want to broadcast explicitly?
                # Actually, srbits_sb_val is (BLOCK_N,).
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

        # DEBUG STORE B
        if b_q_ptr is not None:
            bq_ptrs = b_q_ptr + (
                offs_bn[None, :] * stride_bq_n + offs_k[:, None] * stride_bq_k
            )
            tl.store(bq_ptrs, b_q, mask=mask_b)

        if sb_ptr is not None:
            if use_2d_b:
                # s_b is (BLOCK_K, BLOCK_N) where values change every 16 rows.
                # We want to store (BLOCK_K // 16, BLOCK_N) unique scales.
                # Reshape to isolate the 16-row blocks
                s_b_4d = tl.reshape(s_b, (BLOCK_K // 16, 16, BLOCK_N))
                # Reduce over axis 1 (size 16) - since values are identical, max preserves the value
                val_to_store = tl.max(s_b_4d, axis=1)  # (BLOCK_K // 16, BLOCK_N)

                # Pointers
                # Each pid_k handles BLOCK_K rows, which is (BLOCK_K // 16) scale-blocks.
                global_block_start = pid_k * (BLOCK_K // 16)
                offs_scale_k = global_block_start + tl.arange(0, BLOCK_K // 16)

                sb_ptrs = sb_ptr + (
                    offs_scale_k[:, None] * stride_sb_k + offs_bn[None, :] * stride_sb_n
                )

                # Store
                tl.store(sb_ptrs, val_to_store, mask=offs_bn[None, :] < N)
            else:
                sb_ptrs = sb_ptr + (pid_k * stride_sb_k + offs_bn * stride_sb_n)
                # s_b is (1, BLOCK_N) or (BLOCK_N,) depending on squeeze
                val_to_store = tl.reshape(s_b, (BLOCK_N,))
                tl.store(sb_ptrs, val_to_store, mask=offs_bn < N)


def debug_fake_quant_simultaneous(
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
    return_encoded: bool = True,
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
    else:
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

    # Debug buffers
    a_q = torch.empty_like(a, dtype=torch.float32)
    b_q = torch.empty_like(b, dtype=torch.float32)

    # UPDATED SIZES
    # A Scale: (M, K/block_size)
    num_blocks_k = triton.cdiv(K, block_size)
    s_a = torch.empty((M, num_blocks_k), dtype=torch.float32, device=a.device)

    # B Scale: (num_blocks_k, N)
    if use_2d_b:
        num_blocks_k_sb = triton.cdiv(K, 16)
    else:
        num_blocks_k_sb = triton.cdiv(K, block_size)

    s_b = torch.empty((num_blocks_k_sb, N), dtype=torch.float32, device=b.device)

    # Format Utils
    format_a = FORMAT_E2M1
    format_b = FORMAT_E2M1
    format_sa = get_format_info(scale_type)
    format_sb = get_format_info(scale_type)

    if ga_a is None:
        ga_a = torch.tensor([1.0], device=a.device)
    if ga_b is None:
        ga_b = torch.tensor([1.0], device=b.device)

    # Round Modes
    drm_a = get_round_mode_constant(round_mode_a)
    srm_a = get_round_mode_constant(scale_round_mode_a)
    drm_b = get_round_mode_constant(round_mode_b)
    srm_b = get_round_mode_constant(scale_round_mode_b)

    BLOCK_N_VAL = kwargs.get("BLOCK_N", 64)

    def grid_fn(META):
        num_m = triton.cdiv(M, META["BLOCK_M"])
        num_n = triton.cdiv(N, META["BLOCK_N"])
        return (max(num_m, num_n), triton.cdiv(K, META["BLOCK_K"]))

    debug_fake_quant_simultaneous_kernel[grid_fn](
        a,
        out_a,
        b,
        out_b,
        ga_a,
        ga_b,
        srbits_a,
        srbits_b,
        a_q,
        b_q,
        s_a,
        s_b,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_out_am,
        stride_out_ak,
        a_q.stride(0),
        a_q.stride(1),
        s_a.stride(0),
        s_a.stride(1),
        stride_bn,
        stride_bk,
        stride_out_bn,
        stride_out_bk,
        b_q.stride(1),
        b_q.stride(0),
        s_b.stride(1),
        s_b.stride(0),
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
        BLOCK_M=64,
        BLOCK_N=BLOCK_N_VAL,
        BLOCK_K=block_size,
        encode_centric=encode_centric,
        return_encoded=return_encoded,
    )

    return out_a, out_b, a_q, b_q, s_a, s_b
