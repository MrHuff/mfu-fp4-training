#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Union, Any, Optional

import torch
import re
from quantization.MXFPconfig import MXGemmKernelChoice
from quantization.ExMy import ExMy_new
from dataclasses import dataclass
import gfloat

from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    F4_E2M1_MAX,
    F4_E2M1_MAX_POW2,
    F6_E2M3_MAX,
    F6_E2M3_MAX_POW2,
    F6_E3M2_MAX,
    F6_E3M2_MAX_POW2,
    F8E4M3_MAX,
    F8E4M3_MAX_POW2,
    F8E5M2_MAX,
    F8E5M2_MAX_POW2,
    SUPPORTED_ELEM_DTYPES,
)

# TODO(later): read from somewhere else?
SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2


@dataclass
class DimensionMXTensor:
    """
    A dataclass to represent a micro-exponent (MX) tensor.
    It holds the quantized data and all necessary metadata for dequantization.
    """

    _data: torch.Tensor
    _scale_fp: torch.Tensor
    _scale_lp: torch.Tensor
    _elem_dtype: Union[torch.dtype, str]
    _block_size: int
    _block_dim: Optional[int]
    _orig_dtype: torch.dtype
    _use_fp4_custom_triton_dequant_kernel: bool
    _gemm_kernel_choice: Any
    _max_abs: torch.Tensor
    _max_abs_mask: Optional[torch.Tensor]
    _sm: Optional[torch.Tensor]
    _g: torch.Tensor
    _global_abs_mask: Optional[torch.Tensor]

    def to_dtype(self, target_dtype: torch.dtype) -> torch.Tensor:
        """Dequantizes the MXTensor back to a standard torch.Tensor."""
        return to_dtype_dim(
            self._data,
            self._scale_lp,
            self._elem_dtype,
            self._block_size,
            self._block_dim,
            target_dtype,
            self._g,
        )

    def __repr__(self):
        """Custom string representation for the MXTensor."""
        return (
            f"MXTensor(data={self._data.shape}, "
            f"scale_shape={self._scale_lp.shape}, "
            f"elem_dtype={self._elem_dtype})"
        )


def blockify(data_hp: torch.Tensor, block_dim, block_size):
    orig_shape = data_hp.shape
    if block_dim is not None:
        data_hp = data_hp.reshape(-1, orig_shape[block_dim])
        block_size = data_hp.shape
    else:
        assert data_hp.numel() % block_size == 0, "unsupported"
        data_hp = data_hp.reshape(-1, block_size)
    return data_hp, orig_shape


class MXFPscalingModule(torch.nn.Module):
    def __init__(
        self,
        elem_dtype: Union[torch.dtype, str],
        block_size: int,
        block_dim: int = None,
        scale_type: str = "E8M0",  # ExMy
        roundMode=gfloat.RoundMode.TowardPositive,
        use_approx: dict = {"smooth": False},
        fp_scale_factor: bool = False,
    ):
        super().__init__()
        self.elem_dtype = elem_dtype
        self.block_size = block_size
        self.block_dim = block_dim
        self.scale_type = scale_type
        self.roundMode = roundMode
        self.use_approx = use_approx or {"smooth": False}
        self.fp_scale_factor = fp_scale_factor

    def forward(self, data_hp):
        """
        Takes a high precision tensor and converts to MX scale and raw data, in
        naive layout (scale and raw data are separate tensors).
        """
        E, M = extract_e_m(scale_type=self.scale_type)
        assert self.elem_dtype in SUPPORTED_ELEM_DTYPES, "unsupported"

        # calculate the scale in e8m0 format

        data_hp, data_hp_orig_shape = blockify(data_hp, self.block_dim, self.block_size)

        scale_lp_exmy, max_abs, max_abs_mask, scale_fp, sm, g, glob_max_abs_mask = (
            compute_exmy_scaling(
                data_hp, self.elem_dtype, E, M, self.use_approx, self.roundMode
            )
        )

        # E4M3 allows for division with 0!

        # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
        # float32 denormal range. For now, manually adjust the fp scale. This is
        # relevant if all of the incoming block values are zeroes.
        # See https://github.com/pytorch/pytorch/issues/125557 for details.
        # Note: it would be more correct to set the minimum to 2**-127, but this
        # does not work in triton either as it looks like subnormal value handling
        # has some gaps.  So, for now just set to the minimum normal value.
        # scale and saturated cast the data elements to max of target dtype
        if self.elem_dtype == torch.float8_e4m3fn:
            max_pos = F8E4M3_MAX
        elif self.elem_dtype == torch.float8_e5m2:
            max_pos = F8E5M2_MAX
        elif self.elem_dtype == DTYPE_FP6_E2M3:
            max_pos = F6_E2M3_MAX
        elif self.elem_dtype == DTYPE_FP6_E3M2:
            max_pos = F6_E3M2_MAX
        elif self.elem_dtype == DTYPE_FP4:
            max_pos = F4_E2M1_MAX
        else:
            raise AssertionError("unsupported")
        """
        Beware of clamp, it's a gradient goblin - write custom clamp function.
        """

        data_hp = data_hp / g

        if not self.fp_scale_factor:
            data_lp = torch.clamp(data_hp * scale_lp_exmy, -max_pos, max_pos)
        else:
            data_lp = torch.clamp(data_hp * scale_fp, -max_pos, max_pos)
        # what's going on here! - in some cases the quantisation error is so big when you quantise the scale it kills the content of the matrix.
        data_lp = data_lp.reshape(data_hp_orig_shape)
        return (
            scale_fp,
            scale_lp_exmy,
            data_lp,
            max_abs,
            max_abs_mask,
            sm,
            g,
            glob_max_abs_mask,
        )


def extract_e_m(scale_type: str):
    match = re.match(r"E(\d+)M(\d+)", scale_type)
    if not match:
        raise ValueError(f"Invalid format: {scale_type}")

    E = int(match.group(1))  # Exponent bits
    M = int(match.group(2))  # Mantissa bits
    return E, M


def compute_exmy_scaling(
    data_hp, elem_dtype, E, M, use_approx: dict, round_mode: gfloat.RoundMode
):
    quantizer = ExMy_new(E, M, roundMode=round_mode)
    abs_data = torch.abs(data_hp)
    if use_approx["smooth"] in ["absmax", "STE"]:
        max_abs = abs_data.amax(dim=1, keepdim=True)
        max_abs = torch.where(max_abs == 0, 1.0, max_abs)
        max_abs_mask = abs_data != max_abs
        glob_max = max_abs.amax()
        glob_max_abs_mask = abs_data == glob_max
        sm = None
    elif use_approx["smooth"] == "softsoftmax":
        alpha_abs = use_approx["alpha"] * abs_data
        max_abs = (1 / use_approx["alpha"]) * torch.logsumexp(
            alpha_abs, dim=1, keepdim=True
        )
        max_abs = torch.where(max_abs == 0, 1.0, max_abs)
        sm = torch.softmax(alpha_abs, dim=1)
        max_abs_mask = None
        glob_max = max_abs.amax()
        glob_max_abs_mask = 1
    elif use_approx["smooth"] == "hardsoftmax":
        max_abs = abs_data.amax(dim=1, keepdim=True)
        max_abs = torch.where(max_abs == 0, 1.0, max_abs)
        alpha_abs = use_approx["alpha"] * abs_data
        sm = torch.softmax(alpha_abs, dim=1)
        max_abs_mask = None
        glob_max = max_abs.amax()
        glob_max_abs_mask = 1

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
    g = glob_max if use_approx["use_tensor_scaling"] else 1.0
    # Step 4: Compute the correct floating-point scale
    scale_exmy, non_quantised_scale = quantizer.scale_by_format_max(
        max_abs, target_max_pow2, mbits, g
    )
    return (
        scale_exmy,
        max_abs,
        max_abs_mask,
        non_quantised_scale,
        sm,
        g,
        glob_max_abs_mask,
    )


def to_dtype_dim(data_lp, scale_lp, elem_dtype, block_size, block_dim, target_dtype, g):
    orig_shape = data_lp.shape
    is_transposed = not data_lp.is_contiguous()
    # if the underlying data is transposed, convert to row major before
    # unpacking and unscaling
    if is_transposed:
        data_lp = data_lp.t()
        assert data_lp.is_contiguous()
        orig_shape = (orig_shape[1], orig_shape[0])

    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_hp = data_lp.to(target_dtype)

    elif elem_dtype == DTYPE_FP4:
        # TODO(future PR): add cast directly to bf16
        data_hp = data_lp.to(target_dtype)
        # manually adjust shape to account for the unpacking
        # TODO(future PR): clean up the shape code and remove the hack
        # below
        orig_shape = (*orig_shape[:-1], orig_shape[-1])
    else:
        raise AssertionError("unsupported")

    if block_dim is not None:
        data_hp = data_hp.reshape(-1, orig_shape[block_dim])
    else:
        data_hp = data_hp.reshape(-1, block_size)

    s_fp = scale_lp.reshape(-1, 1).to(target_dtype)
    data_hp = data_hp / s_fp
    data_hp = data_hp.reshape(orig_shape) * g

    # if we converted to row-major before unscaling convert back
    if is_transposed:
        data_hp = data_hp.t()

    return data_hp


def new_to_mx(
    tensor: torch.Tensor,
    scalingModule: MXFPscalingModule,
    gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
    fp4_quantiser: torch.nn.Module = None,
):
    """
    Converts a torch.Tensor to an MXTensor using the specified scaling module.
    This version uses a dataclass and avoids custom autograd functions.
    """
    tensor_orig_dtype = tensor.dtype
    tensor_orig_shape = tensor.shape
    tensor_hp_r = tensor.reshape(-1, tensor_orig_shape[-1])

    # 1. Calculate scales and get the pre-quantized data from the scaling module.
    (
        scale_fp,
        scale_lp,
        tensor_pre_quant,
        max_abs,
        max_abs_mask,
        sm,
        g,
        glob_max_abs_mask,
    ) = scalingModule.forward(tensor_hp_r)

    # 2. Apply the final quantization step if a quantizer is provided.
    quantized_data = fp4_quantiser(tensor_pre_quant)

    # 3. Instantiate the MXTensor dataclass with all computed components.
    tensor_mx = DimensionMXTensor(
        _data=quantized_data,
        _scale_fp=scale_fp,
        _scale_lp=scale_lp,
        _elem_dtype=scalingModule.elem_dtype,
        _block_size=scalingModule.block_size,
        _block_dim=scalingModule.block_dim,
        _orig_dtype=tensor_orig_dtype,
        # This parameter is kept for compatibility but is not actively used without custom kernels
        _use_fp4_custom_triton_dequant_kernel=False,
        _gemm_kernel_choice=gemm_kernel_choice,
        _max_abs=max_abs,
        _max_abs_mask=max_abs_mask,
        _sm=sm,
        _g=g,
        _global_abs_mask=glob_max_abs_mask,
    )

    # Return the MXTensor, original shape, and the pre-quantized tensor to maintain original functionality.
    return tensor_mx, tensor_orig_shape, tensor_pre_quant
