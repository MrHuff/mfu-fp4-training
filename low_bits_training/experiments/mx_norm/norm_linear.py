#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
import functools
from enum import Enum, auto
from typing import Any, Union, Dict, List
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    SUPPORTED_ELEM_DTYPES,
    F32_MIN_NORMAL,
    MBITS_F32,
    F8E4M3_MAX_POW2,
    MBITS_F8_E4M3,
    F8E5M2_MAX_POW2,
    MBITS_F8_E5M2,
    DTYPE_FP6_E2M3,
    F6_E2M3_MAX_POW2,
    MBITS_F6_E2M3,
    DTYPE_FP6_E3M2,
    F6_E3M2_MAX_POW2,
    MBITS_F6_E3M2,
    DTYPE_FP4,
    F4_E2M1_MAX_POW2,
    MBITS_F4_E2M1,
    E8M0_EXPONENT_BIAS,
    F8E4M3_MAX,
    EBITS_F32,
    SBITS,
    E8M0_EXPONENT_NAN_VAL,
    F8E5M2_MAX,
    F6_E2M3_MAX,
    F4_E2M1_MAX,
    F6_E3M2_MAX,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
    f32_to_f4_unpacked,
    pack_uint4,
    pack_uint6,
)
from torchao.prototype.mx_formats.mx_linear import MXGemmKernelChoice
from torchao.prototype.mx_formats.mx_tensor import to_mx as to_mx_orig  # noqa: F401

from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

import torch.nn as nn

from low_bits_training.quantization.mxfp import (
    replace_with_custom_fn_if_matches_filter,
    to_mxfp_dtype,
    _is_linear,
    swap_linear_with_quantization_linear,
    MXLinearGeneral,
    MXLinearConfig,
)
from low_bits_training.experiments.mx_norm.mxfp_patching import ScaleCalculationMode

from low_bits_training.models.fuse_linear import (
    swap_unfused_with_fused,
    AttentionWithFusedLinear,
    FeedForwardWithFusedLinear,
)
from low_bits_training.config.job_config import JobConfig
from low_bits_training.models import get_model_config
from low_bits_training.experiments.mx_norm import lut_and_lerp
from low_bits_training.experiments.mx_norm.mx_norm import _MX_ROUNDING_MODES

from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.model.model import (
    Attention,
    FeedForward,
)

_dtype_mantissa_bits = {
    torch.float8_e4m3fn: MBITS_F8_E4M3,
    torch.float8_e5m2: MBITS_F8_E5M2,
    DTYPE_FP6_E2M3: MBITS_F6_E2M3,
    DTYPE_FP6_E3M2: MBITS_F6_E3M2,
    DTYPE_FP4: MBITS_F4_E2M1,
}


class Reduction(Enum):
    MEAN = auto()
    RMS = auto()


class NormMode(Enum):
    """
    Enum representing methods for normalising MX-Tensors.
    PRE:  This method estimates the RMS from absmax and divides scales by the estimate
          before rounding to e8m0
    POST: This method estimates the RMS from scales and divides data_lp by the estimate
          after rounding scales to e8m0
    """

    PRE = auto()
    POST = auto()


def scale_round_nearest(max_abs, dtype):
    mbits = _dtype_mantissa_bits[dtype]
    # Pre-computed constants
    val_to_add = 1 << (MBITS_F32 - mbits - 1)
    # Fix due to E4M3 NaN encoding instead of 480 value.
    if dtype == torch.float8_e4m3fn:
        val_to_add = 2 * val_to_add + val_to_add
    mask = ((1 << (EBITS_F32 + SBITS)) - 1) << MBITS_F32

    nan_mask = torch.isnan(max_abs)
    max_abs = max_abs.to(torch.float32).view(torch.int32)
    max_abs = (max_abs + val_to_add) & mask
    max_abs = max_abs.view(torch.float32)
    max_abs = torch.where(nan_mask, float("nan"), max_abs)
    return max_abs


def scale_round_cublas_ceil(max_abs, dtype):
    mbits = _dtype_mantissa_bits[dtype]
    # Pre-computed constants
    val_to_add = 1 << (MBITS_F32 - mbits)
    # Fix due to E4M3 NaN encoding instead of 480 value.
    if dtype == torch.float8_e4m3fn:
        val_to_add = 2 * val_to_add
    val_to_add -= 1
    mask = ((1 << (EBITS_F32 + SBITS)) - 1) << MBITS_F32

    nan_mask = torch.isnan(max_abs)
    max_abs = max_abs.to(torch.float32).view(torch.int32)
    max_abs = (max_abs + val_to_add) & mask
    max_abs = max_abs.view(torch.float32)
    max_abs = torch.where(nan_mask, float("nan"), max_abs)
    return max_abs


def rms(x, dim=None, keepdim=False):
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()


def pre_round_rms_estimator_fn(
    max_abs: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    reduction: Reduction = Reduction.MEAN,
):
    # MEAN
    # block_size = 16 -> scale = 0.4814
    # block_size = 32 -> scale = 0.4261
    # block_size = 64 -> scale = 0.3852

    # RMS
    # block_size = 16 -> scale = 0.4688
    # block_size = 32 -> scale = 0.4185
    # block_size = 64 -> scale = 0.3803

    if reduction == Reduction.MEAN:
        fn = torch.mean
    elif reduction == Reduction.RMS:
        fn = rms
    else:
        raise NotImplementedError(f"No function for {reduction}")

    return fn(max_abs, dim=-1, keepdim=True) * scale + eps


def post_round_rms_estimator_fn(
    max_abs: torch.Tensor, lut: torch.Tensor, eps: float = 1e-6
):
    mean_max_abs = max_abs.mean(dim=-1).flatten()
    lut_n_entries = lut.numel() - 1
    log2 = mean_max_abs.log2()
    log2_floor = log2.floor()
    log2_frac = log2 - log2_floor

    lut_index_as_float = log2_frac * lut_n_entries

    lut_smaller_index = lut_index_as_float.floor()
    lut_lerp_coefficient = lut_index_as_float - lut_smaller_index

    lut_smaller_index = lut_smaller_index.long()
    lut_larger_index = lut_smaller_index + 1

    rms_estimate = (
        lut[lut_smaller_index] * (1 - lut_lerp_coefficient)
        + lut[lut_larger_index] * lut_lerp_coefficient
    )
    rms_estimate *= log2_floor.exp2()
    rms_estimate = rms_estimate.reshape(max_abs.shape[:-1])[..., None]
    return rms_estimate + eps


def get_rms_estimator_fn(norm_mode: NormMode):
    func_dict = {
        NormMode.PRE: pre_round_rms_estimator_fn,
        NormMode.POST: post_round_rms_estimator_fn,
    }
    return func_dict[norm_mode]


def to_mx_with_norm(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    norm_mode: NormMode = NormMode.POST,
    reduction: Reduction = Reduction.MEAN,
    norm_kwargs: Dict[str, Any] = {},
    pack_fp6: bool = False,
):
    """
    Takes a high precision tensor and converts to MX scale and raw data, in
    naive layout (scale and raw data are separate tensors).
    """

    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    # TODO(future PR): consider supporting padding
    assert data_hp.numel() % block_size == 0, "unsupported"
    assert data_hp.is_contiguous(), "unsupported"
    assert elem_dtype in SUPPORTED_ELEM_DTYPES, "unsupported"

    rms_estimator_fn = get_rms_estimator_fn(norm_mode)

    # calculate the scale in e8m0 format
    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, orig_shape[-1] // block_size, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(data_hp), dim=-1)

    if norm_mode == NormMode.PRE:
        rms_estimate = rms_estimator_fn(max_abs, reduction=reduction, **norm_kwargs)
        max_abs = max_abs / rms_estimate

    data_hp = data_hp.reshape(-1, block_size)
    max_abs = max_abs.reshape(-1)

    # Add an epsilon to prevent the log2 function call for returning -inf
    # where the values are zero.
    eps = F32_MIN_NORMAL * (max_abs == 0).type(max_abs.dtype)

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type, and get the mbits at the same time
    if elem_dtype == torch.float8_e4m3fn:
        target_max_pow2 = F8E4M3_MAX_POW2
        mbits = MBITS_F8_E4M3
    elif elem_dtype == torch.float8_e5m2:
        target_max_pow2 = F8E5M2_MAX_POW2
        mbits = MBITS_F8_E5M2
    elif elem_dtype == DTYPE_FP6_E2M3:
        target_max_pow2 = F6_E2M3_MAX_POW2
        mbits = MBITS_F6_E2M3
    elif elem_dtype == DTYPE_FP6_E3M2:
        target_max_pow2 = F6_E3M2_MAX_POW2
        mbits = MBITS_F6_E3M2
    elif elem_dtype == DTYPE_FP4:
        target_max_pow2 = F4_E2M1_MAX_POW2
        mbits = MBITS_F4_E2M1
    else:
        raise AssertionError("unsupported element dtype")

    # rounding before calculating the largest power of 2
    # X = 2^(floor(log2(rounding(max_abs(v)))-max_exp))
    if scaling_mode == ScaleCalculationMode.EVEN:
        nan_mask = torch.isnan(max_abs)
        max_abs = max_abs.to(torch.float32).view(torch.int32)
        val_to_add = 1 << (MBITS_F32 - mbits - 1)
        mask = ((1 << (EBITS_F32 + SBITS)) - 1) << MBITS_F32
        max_abs = (max_abs + val_to_add) & mask
        max_abs = max_abs.view(torch.float32)
        max_abs = torch.where(nan_mask, float("nan"), max_abs)
        # max_abs[nan_mask] = torch.tensor(float("nan"), device=max_abs.device)

    if scaling_mode == ScaleCalculationMode.NEAREST_TIES_UP:
        max_abs = scale_round_nearest(max_abs, elem_dtype)
    if scaling_mode == ScaleCalculationMode.CUBLAS_CEIL:
        max_abs = scale_round_cublas_ceil(max_abs, elem_dtype)

    # Calculate the scale for different modes
    if scaling_mode in (
        ScaleCalculationMode.FLOOR,
        ScaleCalculationMode.EVEN,
        ScaleCalculationMode.NEAREST_TIES_UP,
        ScaleCalculationMode.CUBLAS_CEIL,
    ):
        scale_e8m0_unbiased = torch.floor(torch.log2(max_abs + eps)) - target_max_pow2
    elif scaling_mode == ScaleCalculationMode.CEIL:
        scale_e8m0_unbiased = torch.ceil(torch.log2(max_abs + eps)) - target_max_pow2
    else:
        raise AssertionError("unsupported scaling calculation mode")

    # Clamp to exponents that can be represented in e8m0
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS
    )

    # Create the biased e8m0 representation and cast it to 8 bits
    scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS
    scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)

    # Conversion to torch.uint8 sets NaN values to 0, fix this by
    # explicitly setting known NaN values to 255
    scale_e8m0_biased = torch.where(
        torch.isnan(scale_e8m0_unbiased),
        E8M0_EXPONENT_NAN_VAL,
        scale_e8m0_biased,
    )

    # For now, calculate the scale in floating point.
    scale_fp32 = (scale_e8m0_biased.to(torch.int32) << MBITS_F32).view(torch.float32)

    # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
    # float32 denormal range. For now, manually adjust the fp scale. This is
    # relevant if all of the incoming block values are zeroes.
    # See https://github.com/pytorch/pytorch/issues/125557 for details.
    # Note: it would be more correct to set the minimum to 2**-127, but this
    # does not work in triton either as it looks like subnormal value handling
    # has some gaps.  So, for now just set to the minimum normal value.
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    # scale and saturated cast the data elements to max of target dtype
    if elem_dtype == torch.float8_e4m3fn:
        max_pos = F8E4M3_MAX
    elif elem_dtype == torch.float8_e5m2:
        max_pos = F8E5M2_MAX
    elif elem_dtype == DTYPE_FP6_E2M3:
        max_pos = F6_E2M3_MAX
    elif elem_dtype == DTYPE_FP6_E3M2:
        max_pos = F6_E3M2_MAX
    elif elem_dtype == DTYPE_FP4:
        max_pos = F4_E2M1_MAX
    else:
        raise AssertionError("unsupported")

    scale_fp32 = scale_fp32.reshape(-1, orig_shape[-1] // block_size)
    if norm_mode == NormMode.POST:
        rms_estimate = rms_estimator_fn(scale_fp32 * 2**target_max_pow2, **norm_kwargs)
        rms_estimate = rms_estimate.to(data_hp.dtype)
        scale_bias = torch.floor(torch.log2(rms_estimate))
        scale_fp32 = scale_fp32 / 2**scale_bias
        scale_e8m0_biased = scale_e8m0_biased.reshape(-1, orig_shape[-1] // block_size)
        scale_e8m0_biased = scale_e8m0_biased - scale_bias.to(torch.uint8)
        scale_e8m0_biased = scale_e8m0_biased.reshape(-1)

    scale_fp32 = scale_fp32 * rms_estimate
    scale_fp32 = scale_fp32.reshape(-1)

    data_lp = torch.clamp(
        data_hp / scale_fp32.unsqueeze(1), min=-1 * max_pos, max=max_pos
    )
    data_lp = data_lp.reshape(orig_shape)

    # cast to target dtype
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_lp = data_lp.to(elem_dtype)
    elif elem_dtype == DTYPE_FP6_E2M3:
        data_lp = f32_to_f6_e2m3_unpacked(data_lp)
        if pack_fp6:
            orig_shape = [*orig_shape[:-1], 3 * orig_shape[-1] // 4]
            data_lp = pack_uint6(data_lp)
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == DTYPE_FP6_E3M2:
        data_lp = f32_to_f6_e3m2_unpacked(data_lp)
        if pack_fp6:
            orig_shape = [*orig_shape[:-1], 3 * orig_shape[-1] // 4]
            data_lp = pack_uint6(data_lp)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif elem_dtype == DTYPE_FP4:
        data_lp = f32_to_f4_unpacked(data_lp)
        data_lp = pack_uint4(data_lp)
    else:
        raise AssertionError("unsupported")

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    return scale_e8m0_biased, data_lp, rms_estimate


@torch._dynamo.allow_in_graph
class mx_mm(torch.autograd.Function):
    # There are three gemms in a forward + backward of a Linear layer:
    #
    # 1.       input @ weight_t    = output     (forward pass)
    # 2. grad_output @ weight      = grad_input (backward pass)
    # 3.     input_t @ grad_output = grad_weight (backward pass)
    #
    # input, weight and grad_output can have each their own MX element dtype.

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        input_mx_r_dim0: MXTensor,
        weight_hp: torch.Tensor,
        norm_weight_hp: torch.Tensor,
        in_elem_dtype: Any,
        w_elem_dtype: Any,
        grad_elem_dtype: Any,
        block_size: int,
        gemm_kernel_choice: MXGemmKernelChoice,
        scale_rounding: ScaleCalculationMode,
    ):
        ctx.in_elem_dtype = in_elem_dtype
        ctx.w_elem_dtype = w_elem_dtype
        ctx.grad_elem_dtype = grad_elem_dtype
        ctx.block_size = block_size
        ctx.gemm_kernel_choice = gemm_kernel_choice
        ctx.scale_rounding = scale_rounding

        # OCP MX scale calculation (default)
        # scaling_mode = ScaleCalculationMode.FLOOR,
        # Improved rounding.
        w_scaling_mode = scale_rounding
        # act_scaling_mode = ScaleCalculationMode.EVEN
        # w_scaling_mode = ScaleCalculationMode.FLOOR

        # input @ weight_t = output
        ctx.save_for_backward(input_hp, weight_hp, norm_weight_hp)

        input_orig_shape = input_hp.shape

        # Incorporate norm weight into linear weight
        fused_weight_hp = weight_hp * norm_weight_hp

        weight_mx_dim0 = MXTensor.to_mx(
            fused_weight_hp,
            w_elem_dtype,
            block_size,
            scaling_mode=w_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        output = torch.mm(input_mx_r_dim0, weight_mx_dim0.t())
        output = output.reshape(*input_orig_shape[:-1], output.shape[-1])

        return output

    @staticmethod
    def backward(ctx, grad_output_hp: torch.Tensor):
        input_hp, weight_hp, norm_weight_hp = ctx.saved_tensors
        weight_hp_t_c = weight_hp.t().contiguous()
        in_elem_dtype = ctx.in_elem_dtype
        w_elem_dtype = ctx.w_elem_dtype
        grad_elem_dtype = ctx.grad_elem_dtype
        block_size = ctx.block_size
        gemm_kernel_choice = ctx.gemm_kernel_choice
        scale_rounding = ctx.scale_rounding

        grad_output_orig_shape = grad_output_hp.shape
        grad_output_hp_r = grad_output_hp.reshape(-1, grad_output_orig_shape[-1])

        input_hp_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_hp_orig_shape[-1])

        # OCP MX scale calculation (default)
        # scaling_mode = ScaleCalculationMode.FLOOR,
        # Improved rounding.
        act_scaling_mode = scale_rounding
        w_scaling_mode = scale_rounding
        grad_scaling_mode = scale_rounding

        # grad_output @ weight = grad_input
        grad_output_mx_dim0 = MXTensor.to_mx(
            grad_output_hp_r,
            grad_elem_dtype,
            block_size,
            scaling_mode=grad_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )  # mx, [B, out]
        weight_mx_dim1 = MXTensor.to_mx(
            weight_hp_t_c,
            w_elem_dtype,
            block_size,
            scaling_mode=w_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )  # mx, [in, out]
        grad_mm_input = torch.mm(
            grad_output_mx_dim0, weight_mx_dim1.t()
        )  # high_prec, [B, in]

        # input_t @ grad_output = grad_weight
        grad_output_mx_dim1 = MXTensor.to_mx(
            grad_output_hp_r.t().contiguous(),
            grad_elem_dtype,
            block_size,
            scaling_mode=grad_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )  # [out, B]

        input_t_mx_dim0_tmp = MXTensor.to_mx(
            input_hp_r.t().contiguous(),
            in_elem_dtype,
            block_size,
            scaling_mode=act_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        input_t_mx_dim0 = input_t_mx_dim0_tmp.t()

        grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0) * norm_weight_hp
        grad_norm_weight = torch.sum(input_hp_r * grad_mm_input, dim=0)

        grad_input = grad_mm_input * norm_weight_hp

        grad_input = grad_input.reshape(input_hp_orig_shape)

        return (
            grad_input,
            None,  # input_mx - pass grads via high-precious path instead
            grad_weight,
            grad_norm_weight,
            None,  # in_elem_dtype
            None,  # w_elem_dtype
            None,  # grad_elem_dtype
            None,  # block_size
            None,  # gemm_kernel_choice
            None,  # scale_rounding
        )


@functools.cache
def get_max_abs_pre_round_norm_scaling(block_size, reduction):
    if reduction == Reduction.MEAN:
        fn = torch.mean
    elif reduction == Reduction.RMS:
        fn = rms
    else:
        raise NotImplementedError(f"No function for {reduction}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for _ in range(16):
        samples = 2**24
        x = torch.randn((samples, block_size), device=device)
        mean_absmaxes = fn(x.abs().amax(dim=-1), dim=-1)
        results.append(mean_absmaxes.reciprocal().mean())
    return torch.stack(results).mean().cpu().item()


@torch._dynamo.allow_in_graph
class mx_norm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        in_elem_dtype: Any,
        block_size: int,
        scaling_mode: ScaleCalculationMode,
        norm_mode: NormMode,
        reduction: Reduction,
        clamp_val: float,
        norm_kwargs: Dict[str, Any],
        gemm_kernel_choice: MXGemmKernelChoice,
    ):
        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])

        normed_input_mx_r_dim0_scale, normed_input_mx_r_dim0_data, rms_estimate = (
            to_mx_with_norm(
                input_hp_r,
                in_elem_dtype,
                block_size,
                scaling_mode=scaling_mode,
                norm_mode=norm_mode,
                reduction=reduction,
                norm_kwargs=norm_kwargs,
            )
        )

        ctx.save_for_backward(input_hp, rms_estimate)

        normed_input_mx_r_dim0 = MXTensor(
            normed_input_mx_r_dim0_data,
            normed_input_mx_r_dim0_scale,
            in_elem_dtype,
            block_size,
            input_hp_r.dtype,
            gemm_kernel_choice=gemm_kernel_choice,
            pack_fp6=False,
            act_quant_kwargs=None,
        )

        normed_input_r = input_hp_r / rms_estimate
        normed_input = normed_input_r.reshape(input_orig_shape)

        if clamp_val:
            # Initial hack to test whether clamping works
            normed_input_r = torch.clamp(normed_input_r, min=-clamp_val, max=clamp_val)
            normed_input_mx_r_dim0 = MXTensor.to_mx(
                normed_input_r,
                elem_dtype=in_elem_dtype,
                block_size=block_size,
                scaling_mode=scaling_mode,
            )

            normed_input = normed_input_r.reshape(input_orig_shape)

        return normed_input, normed_input_mx_r_dim0

    @staticmethod
    def backward(ctx, grad_out_hp: torch.Tensor, grad_out_mx=None):
        input_hp, rms_estimate = ctx.saved_tensors

        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])

        grad_out_hp_r = grad_out_hp.reshape(-1, input_orig_shape[-1])

        inv_rms = rms_estimate.reciprocal()
        delta = torch.einsum("...d,...d->...", grad_out_hp_r, input_hp_r).unsqueeze(-1)
        grad_input = (-1 / input_hp_r.shape[-1]) * inv_rms.pow(
            3
        ) * input_hp_r * delta + inv_rms * grad_out_hp_r
        grad_input = grad_input
        grad_input = grad_input.reshape(input_orig_shape)
        return grad_input, None, None, None, None, None, None, None, None


class MXNorm(torch.nn.Module):
    """
    Approximate RMS Normalisation for MX-Tensors
    """

    def __init__(
        self,
        eps: float = 1e-6,
        mode: NormMode = NormMode.PRE,
        reduction: Reduction = Reduction.MEAN,
        clamp_val: float | None = None,
        mx_kwargs: Dict[str, Any] = {},
        n_lut_entries: int = 256,
    ):
        super().__init__()
        self.eps = eps
        self.mode = mode
        self.reduction = reduction
        self.clamp_val = clamp_val
        self.scale_rounding = _MX_ROUNDING_MODES[mx_kwargs["scale_rounding"]]
        self.mx_elem_dtype = mx_kwargs["adtype"]
        self.block_size = mx_kwargs["block_size"]
        if mode == NormMode.POST:
            self._info = torch.tensor(
                lut_and_lerp.create_lut(
                    n_lut_entries=n_lut_entries,
                    scale_dtype=torch.float8_e8m0fnu,
                    data_dtype=mx_kwargs["adtype"],
                    scale_rounding_fn=mx_kwargs["scale_rounding"],
                    block_size=mx_kwargs["block_size"],
                ),
                device="cpu",
            )
        elif self.mode == NormMode.PRE:
            self._info = torch.tensor(
                get_max_abs_pre_round_norm_scaling(mx_kwargs["block_size"], reduction),
                device="cpu",
            )
        self.register_buffer("info", self._info)

    def forward(self, x):
        norm_kwargs = {"eps": self.eps}
        if self.mode == NormMode.POST:
            norm_kwargs["lut"] = self.info
        elif self.mode == NormMode.PRE:
            norm_kwargs["scale"] = self.info
        normed_x_hp, normed_x_mx_dim0 = mx_norm.apply(
            x,
            self.mx_elem_dtype,
            self.block_size,
            self.scale_rounding,
            self.mode,
            self.reduction,
            self.clamp_val,
            norm_kwargs,
            MXGemmKernelChoice.EMULATED,
        )
        return normed_x_hp, normed_x_mx_dim0

    def reset_parameters(self):
        self.info.data.copy_(self._info)


class MXNormLinear(torch.nn.Linear):
    """
    Normalised linear layer with the compute happening in MX.

    Normalisation estimates the RMS from block absmax and scales input during
    the cast to MX.
    """

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        adtype,
        wdtype,
        gdtype,
        block_size,
        scale_rounding,
    ):
        mod.__class__ = MXNormLinear
        mod.adtype = adtype
        mod.wdtype = wdtype
        mod.gdtype = gdtype
        mod.block_size = block_size
        mod.scale_rounding = _MX_ROUNDING_MODES[scale_rounding]

        device = mod.weight.device
        mod.norm_weight = torch.nn.Parameter(
            torch.ones(mod.weight.shape[-1], dtype=mod.weight.dtype, device=device)
        )
        return mod

    def forward(self, x_tup):
        x_hp, x_mx_dim0 = x_tup

        y = mx_mm.apply(
            x_hp,
            x_mx_dim0,
            self.weight,
            self.norm_weight,
            self.adtype,
            self.wdtype,
            self.gdtype,
            self.block_size,
            MXGemmKernelChoice.EMULATED,
            self.scale_rounding,
        )

        if self.bias is not None:
            y = y + self.bias
        return y

    def init_weights(self):
        """
        Reset the parameters of the linear layer.
        """
        super().init_weights()
        self.norm_weight.data.fill_(1.0)
        self.norm_info.data.copy_(self._norm_info)


def _is_norm(mod, fqn):
    return isinstance(mod, torch.nn.RMSNorm)


def swap_norm_linear_with_mx_norm_linear(
    model: nn.Module, norm_linear_cls: Any, filter_fn: Any = None, **kwargs
):
    """Swap Torch Linear by custom quantization linear layer.

    Args:
        model
        norm_linear_cls: Norm linear class.
        filter_fn: Additional filter function.
        **kwargs: Additional factory arguments.
    """
    if filter_fn is None:
        linear_filter_fn = _is_linear
        norm_filter_fn = _is_norm
    else:

        def __fn_linear(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        def __fn_norm(mod, fqn):
            return _is_norm(mod, fqn) and filter_fn(mod, fqn)

        linear_filter_fn = __fn_linear
        norm_filter_fn = __fn_norm

    mx_norm_kwargs = {
        "eps": kwargs["norm_kwargs"]["eps"],
        "mode": kwargs["norm_mode"],
        "mx_kwargs": {
            "adtype": kwargs["adtype"],
            "block_size": kwargs["block_size"],
            "scale_rounding": kwargs["scale_rounding"],
        },
        "n_lut_entries": kwargs["norm_kwargs"]["n_lut_entries"],
        "reduction": kwargs["reduction"],
        "clamp_val": kwargs["clamp_val"],
    }

    mx_linear_kwargs = {
        "adtype": kwargs["adtype"],
        "wdtype": kwargs["wdtype"],
        "gdtype": kwargs["gdtype"],
        "block_size": kwargs["block_size"],
        "scale_rounding": kwargs["scale_rounding"],
    }

    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXNorm(**mx_norm_kwargs),  # replace with identity function
        norm_filter_fn,
    )

    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: norm_linear_cls.from_float(mod, **mx_linear_kwargs),
        linear_filter_fn,
    )


def config_to_enum(key, enum_cls):
    try:
        name = key.upper()
        return enum_cls[name]
    except KeyError:
        valid = [e.name for e in enum_cls]
        raise ValueError(f"No value named {name} for {enum_cls} (valid options: {valid})")


class MXNormLinearConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self._block_size: int = int(job_config.mxfp.block_size)
        self._scale_rounding = job_config.mxfp.scale_rounding_fn
        # Activations, weights, gradients dtypes.
        self._adtype = to_mxfp_dtype(job_config.mxfp.activation_dtype)
        self._wdtype = to_mxfp_dtype(job_config.mxfp.weight_dtype)
        self._gdtype = to_mxfp_dtype(job_config.mxfp.gradient_dtype)
        self._norm_mode = config_to_enum(job_config.mx_norm_linear.norm_mode, NormMode)
        self._reduction = config_to_enum(job_config.mx_norm_linear.reduction, Reduction)
        self._clamp_val = job_config.mx_norm_linear.clamp_val

        self._uses_fused_linear = "fused_linear" in job_config.model.converters

        model_config = get_model_config(job_config.model.name, job_config.model.flavor)
        self._norm_kwargs = {
            "eps": float(model_config.norm_eps),
            "n_lut_entries": int(job_config.mx_norm_linear.n_lut_entries),
        }

    def convert(
        self,
        model: nn.Module,
    ):
        if not self._uses_fused_linear:
            # Could throw an error here instead?
            logger.info("Converting model to use fused linear layers")
            swap_unfused_with_fused(model, Attention, AttentionWithFusedLinear)
            swap_unfused_with_fused(model, FeedForward, FeedForwardWithFusedLinear)

        logger.info(
            f"MXNormLinear converter ({self._adtype}, {self._wdtype}, {self._gdtype})"
            f"with block size {self._block_size}, scale_rounding {self._scale_rounding}, "
            f"norm mode {self._norm_mode} and eps {self._norm_kwargs['eps']}"
        )

        def norm_linear_filter_fn(fqn):
            result = fqn not in ["output", "norm"]
            result = result and "attention.wo" not in fqn
            result = result and "feed_forward.w_out" not in fqn
            return result

        swap_norm_linear_with_mx_norm_linear(
            model=model,
            norm_linear_cls=MXNormLinear,
            filter_fn=lambda mod, fqn: norm_linear_filter_fn(fqn),
            adtype=self._adtype,
            wdtype=self._wdtype,
            gdtype=self._gdtype,
            block_size=self._block_size,
            scale_rounding=self._scale_rounding,
            norm_mode=self._norm_mode,
            reduction=self._reduction,
            clamp_val=self._clamp_val,
            norm_kwargs=self._norm_kwargs,
        )

        def mx_linear_filter_fn(fqn):
            result = fqn != "output"
            result = result and "attention.wqkv" not in fqn
            result = result and "feed_forward.w_in" not in fqn
            return result

        mx_config = MXLinearConfig(
            adtype=self._adtype,
            wdtype=self._wdtype,
            gdtype=self._gdtype,
            block_size=self._block_size,
            scale_rounding_fn=_MX_ROUNDING_MODES[self._scale_rounding],
        )

        swap_linear_with_quantization_linear(
            model=model,
            linear_cls=MXLinearGeneral,
            filter_fn=lambda mod, fqn: mx_linear_filter_fn(fqn),
            mx_config=mx_config,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        # Nothing to do: MX scaling is computed on the fly in Linear layer.
        pass


# Register the MXFP converter in TorchTitan
register_model_converter(MXNormLinearConverter, "mx_norm_linear")
