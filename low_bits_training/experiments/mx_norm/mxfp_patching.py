#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
from enum import Enum, auto
from typing import Any, Union
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
import torchao

import low_bits_training
from low_bits_training.quantization.mxfp import MXLinearConfig
from low_bits_training.quantization.mxfp import to_mx as to_mx_orig  # noqa: F401

_dtype_mantissa_bits = {
    torch.float8_e4m3fn: MBITS_F8_E4M3,
    torch.float8_e5m2: MBITS_F8_E5M2,
    DTYPE_FP6_E2M3: MBITS_F6_E2M3,
    DTYPE_FP6_E3M2: MBITS_F6_E3M2,
    DTYPE_FP4: MBITS_F4_E2M1,
}


class ScaleCalculationMode(Enum):
    """
    Enum representing the different methods for calculating MX block scaling.
    There are three methods available:
    FLOOR: This method is recommended by the OCP MX Spec 1.0 and uses X = 2^floor(log2(max_abs(v))-max_exp).
           It result in overflow issues for large values and bad for gradient quantization.
    CEIL: This method avoids overflow issues, but small values may shift to 0 due to a large scaling factor.
           It uses X = 2^ceil(log2(max_abs(v))-max_exp).
    EVEN: This method is a trade-off between Option 1 and Option 2. It uses X = 2^(floor(log2(rounding(max_abs(v)))-max_exp)).
           It provides better accuracy for MX4 training compared to FLOOR and CEIL.
    By default, we use the EVEN method for better accuracy.

    NEAREST_TIES_UP: E4M3 464 rounding boundary.
    CUBLAS_CEIL: E4M3 448 rounding bounding.
    """

    FLOOR = auto()
    CEIL = auto()
    EVEN = auto()
    NEAREST_TIES_UP = auto()
    CUBLAS_CEIL = auto()


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


def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
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

    # calculate the scale in e8m0 format

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(data_hp), 1)

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
        raise AssertionError(f"unsupported scaling calculation mode: {scaling_mode}")

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
    return scale_e8m0_biased, data_lp


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
        weight_hp: torch.Tensor,
        in_elem_dtype: Any,
        w_elem_dtype: Any,
        grad_elem_dtype: Any,
        block_size: int,
        gemm_kernel_choice: MXGemmKernelChoice,
        scale_rounding: ScaleCalculationMode,
    ):
        ctx.save_for_backward(input_hp, weight_hp)
        ctx.in_elem_dtype = in_elem_dtype
        ctx.w_elem_dtype = w_elem_dtype
        ctx.grad_elem_dtype = grad_elem_dtype
        ctx.block_size = block_size
        ctx.gemm_kernel_choice = gemm_kernel_choice
        ctx.scale_rounding = scale_rounding

        # OCP MX scale calculation (default)
        # scaling_mode = ScaleCalculationMode.FLOOR,
        # Improved rounding.
        act_scaling_mode = scale_rounding
        w_scaling_mode = scale_rounding
        # act_scaling_mode = ScaleCalculationMode.EVEN
        # w_scaling_mode = ScaleCalculationMode.FLOOR

        # input @ weight_t = output
        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])

        input_mx_r_dim0 = MXTensor.to_mx(
            input_hp_r,
            in_elem_dtype,
            block_size,
            scaling_mode=act_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        weight_mx_dim0 = MXTensor.to_mx(
            weight_hp,
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
        input_hp, weight_hp = ctx.saved_tensors
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
        )
        weight_mx_dim1 = MXTensor.to_mx(
            weight_hp_t_c,
            w_elem_dtype,
            block_size,
            scaling_mode=w_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        grad_input = torch.mm(grad_output_mx_dim0, weight_mx_dim1.t())
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        # input_t @ grad_output = grad_weight
        grad_output_mx_dim1 = MXTensor.to_mx(
            grad_output_hp_r.t().contiguous(),
            grad_elem_dtype,
            block_size,
            scaling_mode=grad_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        input_t_mx_dim0_tmp = MXTensor.to_mx(
            input_hp_r.t().contiguous(),
            in_elem_dtype,
            block_size,
            scaling_mode=act_scaling_mode,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
        grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)

        return grad_input, grad_weight, None, None, None, None, None, None


def mx_linear(x, w, bias, mx_config: MXLinearConfig):
    """Custom MX linear function, passing additional arguments:

    scale_rounding_fn: Scale rounding used during MX cast.
    """
    y = mx_mm.apply(
        x,
        w,
        mx_config.adtype,
        mx_config.wdtype,
        mx_config.gdtype,
        mx_config.block_size,
        mx_config.gemm_kernel_choice,
        # use_fp8_dim1_cast_triton_kernel,
        mx_config.scale_rounding_fn,
    )
    if bias is not None:
        y = y + bias
    return y


# `to_mx` torchao patching.
torchao.prototype.mx_formats.mx_tensor.to_mx = to_mx
# `mx_linear` + `ScaleCalculationMode` core patching.
low_bits_training.quantization.mxfp.mx_linear = mx_linear
low_bits_training.quantization.mxfp.ScaleCalculationMode = ScaleCalculationMode
