"""
Triton-based Fused Quantization-Matmul Linear Layer (V3).
Refactored to use split kernels for A and B to avoid thread divergence and improve occupancy.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice
from typing import Optional, Any, Tuple
from dataclasses import dataclass
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
    # Fallback/Dummy if experimental TE is not available
    print("[-] Warning: transformer_engine.pytorch.experimental.quantization_custom_triton not found.")
    FORMAT_E2M1 = None
    FORMAT_E4M3 = None
    FORMAT_E5M2 = None
    FORMAT_E8M0 = None
    FORMAT_E5M3 = None
    _round_float_kernel_impl = None

from .fused_quant_triton_v2 import (
    get_format_info,
    get_round_mode_constant,
    _round_float_kernel_impl as _fallback_round_float_kernel_impl,
    triton_amax_kernel,
    triton_fwht_rht_kernel,
)

if _round_float_kernel_impl is None:
    _round_float_kernel_impl = _fallback_round_float_kernel_impl

if FORMAT_E2M1 is None:
    FORMAT_E2M1 = get_format_info("E2M1")
    FORMAT_E4M3 = get_format_info("E4M3")
    FORMAT_E5M2 = get_format_info("E5M2")
    FORMAT_E8M0 = get_format_info("E8M0")
    FORMAT_E5M3 = get_format_info("E5M3")

torch.set_float32_matmul_precision("high")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "num_warps": 8}, num_stages=3),
        triton.Config({"BLOCK_M": 128, "num_warps": 4}, num_stages=3),
        triton.Config({"BLOCK_M": 64, "num_warps": 4}, num_stages=3),
        triton.Config({"BLOCK_M": 32, "num_warps": 4}, num_stages=3),
    ],
    key=["M", "K"],
)
@triton.jit
def fake_quant_kernel(
    # Pointers
    ptr,
    out_ptr,
    global_amax_ptr,
    srbits_ptr,
    # Dimensions
    M,
    K,
    stride_m,
    stride_k,
    stride_out_m,
    stride_out_k,
    # Params
    scale_max,
    # Format
    prec: tl.constexpr,
    bias: tl.constexpr,
    has_sub: tl.constexpr,
    max_val: tl.constexpr,
    min_val: tl.constexpr,
    is_signed: tl.constexpr,
    has_nz: tl.constexpr,
    has_infs: tl.constexpr,
    num_nans: tl.constexpr,
    # Scale Format
    s_prec: tl.constexpr,
    s_bias: tl.constexpr,
    s_has_sub: tl.constexpr,
    s_max: tl.constexpr,
    s_min: tl.constexpr,
    s_signed: tl.constexpr,
    s_nz: tl.constexpr,
    s_inf: tl.constexpr,
    s_nan: tl.constexpr,
    # Rounding Modes
    data_round_mode: tl.constexpr,
    scale_round_mode: tl.constexpr,
    # Options
    use_global_scale: tl.constexpr,
    use_2d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    encode_centric: tl.constexpr = False,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    # Grid logic: Kernel launched with grid (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))?
    # Actually, let's keep it simple: 1D grid for M, loop/grid for K?
    # K is usually small-ish (block_size=16/32) but can be large if we don't tile K.
    # But wait, quantization is block-wise along K.
    # The existing v2 logic had pid_k.
    # Let's use 2D grid: (M blocks, K blocks).

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Check bounds
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Pointers
    ptrs = ptr + (offs_m[:, None] * stride_m + offs_k[None, :] * stride_k)
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    # SRBits
    srbits = tl.full(x.shape, 0, dtype=tl.int32)
    if srbits_ptr is not None:
        srbits_ptrs = srbits_ptr + (
            offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
        )
        srbits = tl.load(srbits_ptrs, mask=mask, other=0)

    # Global Scale
    g_enc = 1.0
    g_dec = 1.0

    if use_global_scale:
        max_f32_val = 3.4028235e38
        one_val = 1.0

        g_amax = tl.load(global_amax_ptr)
        factor = scale_max * max_val
        g_amax_f32 = g_amax.to(tl.float32)
        factor_f32 = tl.full(g_amax.shape, factor, tl.float32)

        g_amax_safe = tl.where(tl.abs(g_amax_f32) < 1e-9, 1.0, g_amax_f32)
        ges = tl.extra.cuda.libdevice.div_rn(factor_f32, g_amax_safe)

        max_f32_t = tl.full(ges.shape, max_f32_val, tl.float32)
        ges = tl.minimum(ges, max_f32_t)
        one_f32_t = tl.full(ges.shape, one_val, tl.float32)

        ges = tl.where(tl.abs(g_amax_f32) < 1e-9, one_f32_t, ges)
        ges_safe = tl.where(tl.abs(ges) < 1e-9, 1.0, ges)
        gds = tl.extra.cuda.libdevice.div_rn(one_f32_t, ges_safe)

        g_dec = tl.where(tl.abs(ges) < 1e-9, one_f32_t, gds)
        g_enc = ges

    # 2D Logic (for weights usually)
    # V3 Logic for 'use_2d':
    # If use_2d is True, we process input x as (BLOCK_M, BLOCK_K).
    # But effectively we want to compute max over (BLOCK_K) chunks spread across M.
    # Wait, B is (N, K). Row-wise quantization means scaling per-row (per-N).
    # use_2d means we tile (N, K) into (BLOCK_N/BLOCK_K, BLOCK_K)??
    # From V2:
    # b_reshaped = tl.reshape(b, (BLOCK_K, BLOCK_N // BLOCK_K, BLOCK_K))
    # m1 = max(abs, axis=0) -> (N/K, K).
    # m2 = max(m1, axis=1) -> (N/K).
    # vals = m2 * 1.
    # This implies for every BLOCK_N input rows (N dim), we produce BLOCK_N scales?
    # No, we produce scale per... ?
    # V2 `b_max_val` shape is `(BLOCK_N,)`. So it is per-row (N).
    # But calculated using a reshaped max.
    # The V2 logic is:
    # 1. Take (BLOCK_N, BLOCK_K) tile of B.
    # 2. Reshape to (BLOCK_K, BLOCK_N/BLOCK_K, BLOCK_K).
    # 3. Max over dim 0 (reduction of BLOCK_K size).
    # 4. Max over dim 2 (reduction of BLOCK_K size).
    # 5. Result: (BLOCK_N/BLOCK_K,).
    # 6. Broadcast back to (BLOCK_N,).
    # This means each 'stripe' of rows of height BLOCK_K shares a scale?

    if use_2d:
        # Assumes BLOCK_M % BLOCK_K == 0
        # x is (BLOCK_M, BLOCK_K) inside kernel.
        # However, due to grid mapping, this corresponds to B block (N_chunk, K_chunk).
        # We need to transpose to (BLOCK_K, BLOCK_M) to match V2 tiling logic which expects (K, N).
        x_t = tl.trans(x)  # (BLOCK_K, BLOCK_M)

        # V2 Tiling Logic on (K, N) block
        x_reshaped = tl.reshape(x_t, (BLOCK_K, BLOCK_M // BLOCK_K, BLOCK_K))
        m1 = tl.max(tl.abs(x_reshaped), axis=0)  # (BLOCK_M // BLOCK_K, BLOCK_K)
        m2 = tl.max(m1, axis=1)  # (BLOCK_M // BLOCK_K,)
        m2_exp = m2[:, None]  # (BLOCK_M // BLOCK_K, 1)
        one_tile = tl.full((1, BLOCK_K), 1.0, tl.float32)
        vals = m2_exp * one_tile  # (BLOCK_M // BLOCK_K, BLOCK_K)
        x_max = tl.reshape(vals, (BLOCK_M,))
    else:
        x_max = tl.max(tl.abs(x), axis=1)  # (BLOCK_M,)

    data_max_f32 = tl.full(x_max.shape, max_val, tl.float32)
    eps_f32 = tl.full(x_max.shape, 1e-9, tl.float32)
    is_zero_block = x_max <= eps_f32

    if encode_centric:
        # Encode Centric Logic
        # 1. Local Scale
        scale_max_e2m1 = tl.full(x_max.shape, max_val, tl.float32)
        decode_scale = x_max / scale_max_e2m1

        if use_global_scale:
            decode_scale = decode_scale * g_enc

        decode_scale = tl.clamp(decode_scale, -scale_max, scale_max)

        scale_for_zeros = tl.full(decode_scale.shape, 1.0 / scale_max, tl.float32)
        decode_scale = tl.where(is_zero_block, scale_for_zeros, decode_scale)

        # Round Scale
        srbits_s = tl.full((BLOCK_M, 1), 0, tl.int32)
        if srbits_ptr is not None:
            # Re-read srbits for scale if needed? v2 used same srbits_ptr but strided differently?
            # v2: srbits_sa_ptrs = srbits_a_ptr + (offs_am * stride_am)
            srbits_s_ptrs = srbits_ptr + (offs_m * stride_m)
            srbits_s_val = tl.load(srbits_s_ptrs, mask=offs_m < M, other=0)
            srbits_s = srbits_s_val[:, None]

        decode_scale_rounded = _round_float_kernel_impl(
            decode_scale,
            s_prec,
            s_bias,
            s_has_sub,
            s_max,
            s_min,
            s_signed,
            s_nz,
            s_inf,
            s_nan,
            scale_round_mode,
            srbits_s,
            8,
        )

        # Calculate Encode Scale
        if use_global_scale:
            denom = decode_scale_rounded * g_dec
        else:
            denom = decode_scale_rounded

        denom = tl.where(tl.abs(denom) < 1e-9, 1.0, denom)
        one_f32 = tl.full(denom.shape, 1.0, tl.float32)
        encode_scale_normal = one_f32 / denom

        scale_max_val_t = tl.full(g_enc.shape, scale_max, tl.float32)
        if use_global_scale:
            encode_scale_zeros = g_enc * scale_max_val_t
        else:
            encode_scale_zeros = scale_max_val_t

        s_mult = tl.where(is_zero_block, encode_scale_zeros, encode_scale_normal)

        max_f32_clamp = tl.full(s_mult.shape, 3.4028235e38, tl.float32)
        s_mult = tl.minimum(s_mult, max_f32_clamp)
        s_mult = s_mult[:, None]

        s_x = decode_scale_rounded[:, None]

        es_b = tl.broadcast_to(s_mult, x.shape)
        x_scaled = x * es_b

        x_q = _round_float_kernel_impl(
            x_scaled,
            prec,
            bias,
            has_sub,
            max_val,
            min_val,
            is_signed,
            has_nz,
            has_infs,
            num_nans,
            data_round_mode,
            srbits,
            8,
        )

        x_dq = x_q * s_x

    else:
        # Decode Centric Logic (Default)
        tmp = tl.extra.cuda.libdevice.div_rn(x_max, data_max_f32)
        s_x = tmp
        if use_global_scale:
            s_x = tl.extra.cuda.libdevice.mul_rn(tmp, g_enc)

        s_x = tl.where(is_zero_block, tl.full(s_x.shape, 0.0, tl.float32), s_x)
        s_x = s_x[:, None]

        srbits_s = tl.full((BLOCK_M, 1), 0, tl.int32)
        if srbits_ptr is not None:
            srbits_s_ptrs = srbits_ptr + (offs_m * stride_m)
            srbits_s_val = tl.load(srbits_s_ptrs, mask=offs_m < M, other=0)
            srbits_s = srbits_s_val[:, None]

        s_x = _round_float_kernel_impl(
            s_x,
            s_prec,
            s_bias,
            s_has_sub,
            s_max,
            s_min,
            s_signed,
            s_nz,
            s_inf,
            s_nan,
            scale_round_mode,
            srbits_s,
            8,
        )

        if use_global_scale:
            denom = tl.extra.cuda.libdevice.mul_rn(s_x, g_dec)
            denom_safe = tl.where(tl.abs(denom) < 1e-9, 1.0, denom)
            one_f32_bcast = tl.full(denom.shape, 1.0, tl.float32)
            es = tl.extra.cuda.libdevice.div_rn(one_f32_bcast, denom_safe)
        else:
            s_safe = tl.where(tl.abs(s_x) < 1e-9, 1.0, s_x)
            one_f32_bcast = tl.full(s_x.shape, 1.0, tl.float32)
            es = tl.extra.cuda.libdevice.div_rn(one_f32_bcast, s_safe)

        es_b = tl.broadcast_to(es, x.shape)
        x_scaled = tl.extra.cuda.libdevice.mul_rn(x, es_b)

        x_q = _round_float_kernel_impl(
            x_scaled,
            prec,
            bias,
            has_sub,
            max_val,
            min_val,
            is_signed,
            has_nz,
            has_infs,
            num_nans,
            data_round_mode,
            srbits,
            8,
        )
        x_dq = x_q * s_x

    # Unscale global if needed
    if use_global_scale:
        x_dq = tl.extra.cuda.libdevice.mul_rn(x_dq, g_dec)

    # Store
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_k[None, :] * stride_out_k)
    tl.store(out_ptrs, x_dq, mask=mask)


def fake_quant_simultaneous(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_max_a: float = 448.0,
    scale_max_b: float = 448.0,
    use_global_scale: bool = True,
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
    **kwargs,
):
    M, K = a.shape
    # B is (K, N) usually, but logic assumes it's (Kb, N).
    # We want to quantize B along K (columns).
    # Since our kernel is Row-Wise (along stride_k), we should treat B transposed: (N, K).
    # Then each 'row' is a column of B.

    # Ensure Contiguous
    a = a.contiguous()
    b = b.contiguous()

    if stride_am is None:
        stride_am = a.shape[1]
    if stride_ak is None:
        stride_ak = 1

    # B strides: B is (K, N). stride_bn is axis 1, stride_bk is axis 0.
    if stride_bn is None:
        stride_bn = 1  # Axis 1 (N) is inner dimension for RowMajor?
        # Wait: "B is (K, N) usually".
        # If contiguous RowMajor: stride(0) = N, stride(1) = 1.
        # stride_bn is axis 1 (N). So 1.
        # stride_bk is axis 0 (K). So N.
        stride_bn = 1
    if stride_bk is None:
        stride_bk = b.shape[1]  # N

    out_a = torch.empty_like(a, dtype=data_dtype)
    out_b = torch.empty_like(b, dtype=data_dtype)

    if stride_out_am is None:
        stride_out_am = out_a.shape[1]
    if stride_out_ak is None:
        stride_out_ak = 1
    if stride_out_bn is None:
        stride_out_bn = 1
    if stride_out_bk is None:
        stride_out_bk = out_b.shape[1]

    # Compute Global Amax
    if use_global_scale:
        BLOCK_SIZE = 1024
        ga_a = torch.zeros(1, device=a.device, dtype=torch.float32)
        triton_amax_kernel[(triton.cdiv(a.numel(), BLOCK_SIZE),)](
            a, ga_a, a.numel(), BLOCK_SIZE=BLOCK_SIZE
        )
        ga_b = torch.zeros(1, device=b.device, dtype=torch.float32)
        triton_amax_kernel[(triton.cdiv(b.numel(), BLOCK_SIZE),)](
            b, ga_b, b.numel(), BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        ga_a = torch.empty(1, device=a.device)
        ga_b = torch.empty(1, device=a.device)

    format_data = FORMAT_E2M1  # Placeholder
    format_scale = get_format_info(scale_type)

    drm_a = get_round_mode_constant(round_mode_a)
    srm_a = get_round_mode_constant(scale_round_mode_a)
    drm_b = get_round_mode_constant(round_mode_b)
    srm_b = get_round_mode_constant(scale_round_mode_b)

    # Kernel A
    # Grid: (M, K) blocks
    def grid_a(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(K, META["BLOCK_K"]),
        )

    fake_quant_kernel[grid_a](
        a,
        out_a,
        ga_a,
        srbits_a,
        M,
        K,
        stride_am,
        stride_ak,
        stride_out_am,
        stride_out_ak,
        scale_max_a,
        format_data.precision,
        format_data.bias,
        format_data.has_subnormals,
        format_data.max_val,
        format_data.min_val,
        format_data.is_signed,
        format_data.has_nz,
        format_data.has_infs,
        format_data.num_nans,
        format_scale.precision,
        format_scale.bias,
        format_scale.has_subnormals,
        format_scale.max_val,
        format_scale.min_val,
        format_scale.is_signed,
        format_scale.has_nz,
        format_scale.has_infs,
        format_scale.num_nans,
        drm_a,
        srm_a,
        use_global_scale,
        False,  # use_2d not implemented for A
        BLOCK_K=block_size,
        encode_centric=encode_centric,
    )

    # Kernel B
    # B is (K, N). We want to quantize along K.
    # We treat it as (N, K).
    # Strides for (N, K) view:
    # Stride_M (dim 0 aka N) -> stride_bn
    # Stride_K (dim 1 aka K) -> stride_bk
    N = b.shape[1]

    def grid_b(META):
        return (
            triton.cdiv(N, META["BLOCK_M"]),
            triton.cdiv(K, META["BLOCK_K"]),
        )

    fake_quant_kernel[grid_b](
        b,
        out_b,
        ga_b,
        srbits_b,
        N,
        K,
        stride_bn,
        stride_bk,
        stride_out_bn,
        stride_out_bk,
        scale_max_b,
        format_data.precision,
        format_data.bias,
        format_data.has_subnormals,
        format_data.max_val,
        format_data.min_val,
        format_data.is_signed,
        format_data.has_nz,
        format_data.has_infs,
        format_data.num_nans,
        format_scale.precision,
        format_scale.bias,
        format_scale.has_subnormals,
        format_scale.max_val,
        format_scale.min_val,
        format_scale.is_signed,
        format_scale.has_nz,
        format_scale.has_infs,
        format_scale.num_nans,
        drm_b,
        srm_b,
        use_global_scale,
        use_2d_b,
        BLOCK_K=block_size,
        encode_centric=encode_centric,
    )

    return out_a, out_b


class TritonFusedQuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, params):
        params_saved = params.copy()

        scale_max = params["scale_max"]
        use_global_scale = params["use_global_scale"]
        block_size = params["block_size"]
        scale_type = params.get("scale_type", "E4M3")

        # RHT / 2D params saving
        use_2d_weights = params.get("use_2d_weights", False)
        encode_centric = params.get("encode_centric", False)
        # We need these for backward even if forward doesn't use RHT on inputs traditionally

        ctx.saved_params = params_saved
        ctx.save_for_backward(input, weight, bias)

        input_shape = input.shape
        if input.dim() > 2:
            input_flat = input.reshape(-1, input_shape[-1])
        else:
            input_flat = input

        M, K = input_flat.shape
        N = weight.shape[0]
        data_dtype = params.get("data_dtype", torch.float32)

        a_dq, b_dq = fake_quant_simultaneous(
            input_flat,
            weight.t(),
            scale_max_a=scale_max,
            scale_max_b=scale_max,
            use_global_scale=use_global_scale,
            scale_type=scale_type,
            data_dtype=data_dtype,
            round_mode_a=params.get("roundMode", "TiesToEven"),
            scale_round_mode_a=params.get("scale_round_mode", "TiesToEven"),
            round_mode_b=params.get("roundMode", "TiesToEven"),
            scale_round_mode_b=params.get("scale_round_mode", "TiesToEven"),
            srbits_a=params.get("srbits_a", None),
            srbits_b=params.get("srbits_b", None),
            use_2d_b=use_2d_weights,  # Correctly pass use_2d for Weights
            encode_centric=encode_centric,
            block_size=block_size,
        )

        y = torch.mm(a_dq, b_dq)
        if bias is not None:
            y = y + bias

        if input.dim() > 2:
            y = y.view(*input_shape[:-1], N)

        return y.to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        params = ctx.saved_params
        block_size = params["block_size"]
        scale_max = params["scale_max"]
        use_global_scale = params["use_global_scale"]
        scale_type = params.get("scale_type", "E4M3")
        data_dtype = params.get("data_dtype", torch.float32)
        encode_centric = params.get("encode_centric", False)

        # RHT Logic
        with_rht = params.get("with_rht", False)
        with_random_sign_mask = params.get("with_random_sign_mask", True)

        dY = grad_output.contiguous()
        X = input.contiguous()
        W = weight.contiguous()

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

        # 1. dX = dY @ W
        # Standard Quantization for dX
        dy_dq, w_dq = fake_quant_simultaneous(
            dY_flat,
            W,
            scale_max_a=scale_max,
            scale_max_b=scale_max,
            use_global_scale=use_global_scale,
            scale_type=scale_type,
            data_dtype=data_dtype,
            use_2d_b=params.get("use_2d_weights", False),
            encode_centric=encode_centric,
            block_size=block_size,
        )

        grad_input = torch.mm(dy_dq, w_dq).to(input.dtype)
        if dY.dim() > 2:
            grad_input = grad_input.view(*dY_shape[:-1], K)

        # 2. dW = dY.T @ X
        if with_rht:
            # Generate Signs
            if with_random_sign_mask:
                if block_size == 16:
                    WGRAD_SIGNS = torch.tensor(
                        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
                        dtype=torch.float32,
                        device=dY.device,
                    )
                else:
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

            # Apply RHT to dY.T and X.T (effectively transforming rows of dY.T and X)
            # triton_fwht_rht_kernel transforms along last dim (stride_k).
            # dY.T is (N, M). X is (M, K).
            # We want dY.T @ X.
            # We transform N-dim of dY.T? No, M-dim.
            # dW = (dY.T * H) @ (H * X)  <-- H cancel?
            # Standard RHT: dW ~ (dY @ R) @ (R.T @ X) ??
            # TE Logic: transform along the contraction dimension (M)?
            # Yes, "Rotated Accumulation".
            # We transform M dimension of dY^T and X.
            # dY_t is (N, M). Transform M.

            BLOCK_M_RHT = 32

            dY_t = dY_flat.t().contiguous()  # (N, M)
            dY_t_fwht = torch.empty_like(dY_t, dtype=torch.float32)

            # Grid M (program_id(0)) maps to dim 0 of input (N).
            # Grid K (program_id(1)) maps to dim 1 of input (M).
            # Kernel expects: (M, K) logical.
            # We pass N as M-param, M as K-param.
            # Stride XM -> stride 0. Stride XK -> stride 1.

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

            # Quantize and Matmul
            # dY (N, M) and X (K, M).
            # We want (N, M) @ (M, K) -> (N, K).
            # So X_target should be (K, M) and used as B (transposed logic works out).
            # Basically dW = dY_target @ X_target.T

            dy_dq, x_dq = fake_quant_simultaneous(
                dY_target,
                X_target.t(),  # (M, K)
                scale_max_a=scale_max,
                scale_max_b=scale_max,
                use_global_scale=use_global_scale,
                scale_type=scale_type,
                data_dtype=data_dtype,
                use_2d_b=False,
                encode_centric=encode_centric,
                block_size=block_size,
            )
            grad_weight = torch.mm(dy_dq, x_dq)

        else:
            # Standard WGRAD
            dY_t = dY_flat.t().contiguous()

            dy_dq, x_dq = fake_quant_simultaneous(
                dY_t,
                X_flat,
                scale_max_a=scale_max,
                scale_max_b=scale_max,
                use_global_scale=use_global_scale,
                scale_type=scale_type,
                data_dtype=data_dtype,
                use_2d_b=False,
                encode_centric=encode_centric,
                block_size=block_size,
            )
            grad_weight = torch.mm(dy_dq, x_dq)

        grad_bias = None
        if bias is not None:
            grad_bias = dY_flat.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None


class TritonFusedQuantLinearV3(nn.Module):
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
            data_dtype=torch.bfloat16,
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
        params = {
            "scale_max": self.scale_max,
            "use_global_scale": self.use_global_scale,
            "block_size": self.block_size,
            "scale_type": self.scale_type,
            "data_dtype": self.data_dtype,
        }
        if hasattr(self, "mx_config") and self.mx_config:
            params.update(
                {
                    "roundMode": self.mx_config.roundMode,
                    "scale_round_mode": self.mx_config.scale_round_mode,
                    "use_2d_weights": self.mx_config.use_2d_weights,
                    "encode_centric": self.mx_config.encode_centric,
                    "with_rht": getattr(self.mx_config, "with_rht", False),
                    "with_random_sign_mask": getattr(
                        self.mx_config, "with_random_sign_mask", True
                    ),
                }
            )

        return TritonFusedQuantLinearFunction.apply(input, self.weight, self.bias, params)
