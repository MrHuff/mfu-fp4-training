#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from dataclasses import dataclass
import torch
from typing import Any, Optional
from enum import Enum

import gfloat

DTYPE_FP4 = "fp4_e2m1"
DTYPE_FP6_E3M2 = "fp6_e3m2"
DTYPE_FP6_E2M3 = "fp6_e2m3"

SUPPORTED_ELEM_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    DTYPE_FP4,
]

# TODO(later): read from somewhere else?
SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2


class MXGemmKernelChoice(Enum):
    # always available - MX operands are dequantized and a high precision
    # gemm is run
    EMULATED = "emulated"

    # available only when CUDA capability is greater than or equal to 10.0
    CUTLASS = "cutlass"

    # TODO(future PR): add cuBLAS here once we land pytorch/pytorch support


@dataclass
class MXLinearDimConfig:
    # block size for scaling, default is 32 to match
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
    # section 5.2
    block_size: int = 32

    # element dtype, used for activations, weights and gradients
    elem_dtype: Any = torch.float8_e4m3fn

    # overrides for element dtype for weights and gradients
    # TODO(future PR): refactor to make this cleaner
    elem_dtype_weight_override: Optional[Any] = None
    elem_dtype_grad_output_override: Optional[Any] = None

    # defines the gemm kernel choice, if the chosen kernel is not supported
    # on the given hardware an exception will be thrown
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.EMULATED

    # If True, uses a custom triton kernel for fp4 dequantize
    use_fp4_custom_triton_dequant_kernel: bool = False

    scale_type: str = "E8M0"

    block_dim: int = (None,)
    use_approx: dict = None
    dtype: torch.dtype = torch.bfloat16
    roundMode: gfloat.RoundMode = gfloat.RoundMode.TowardPositive
    fp_scale_factor: bool = False

    def __post_init__(self):
        # validate elem_dtype and its overrides
        assert (
            self.elem_dtype in SUPPORTED_ELEM_DTYPES
        ), f"elem_dtype: expected one of {SUPPORTED_ELEM_DTYPES}, got {self.elem_dtype}"
        if self.elem_dtype_weight_override is not None:
            assert (
                self.elem_dtype_weight_override in SUPPORTED_ELEM_DTYPES
            ), f"elem_dtype_weight_override: expected one of {SUPPORTED_ELEM_DTYPES}, got {self.elem_dtype}"
        if self.elem_dtype_grad_output_override is not None:
            assert (
                self.elem_dtype_grad_output_override in SUPPORTED_ELEM_DTYPES
            ), f"elem_dtype_grad_output_override: expected one of {SUPPORTED_ELEM_DTYPES}, got {self.elem_dtype}"

        # validate that block size and elem_dtype matches kernel choice
        if self.gemm_kernel_choice == MXGemmKernelChoice.CUTLASS:
            assert (
                self.block_size == 32
            ), f"block_size must be 32 to use the CUTLASS MX gemm kernels, got {self.block_size}"
            valid_dtypes = [torch.float8_e4m3fn, DTYPE_FP4]
            assert (
                self.elem_dtype in valid_dtypes
            ), f"elem_dtype must be one of {valid_dtypes} to use the CUTLASS MX gemm kernels, got {self.elem_dtype}"
            assert (
                self.elem_dtype_weight_override is None
            ), "elem_dtype_weight_override not supported for CUTLASS MX gemm kernels"
            assert (
                self.elem_dtype_grad_output_override is None
            ), "elem_dtype_grad_output_override not supported for CUTLASS MX gemm kernels"

        # def to_mx_dim(
        # data_hp: torch.Tensor,
        # elem_dtype: Union[torch.dtype, str],
        # block_size: int,
        # block_dim: int = None,
        # scale_type: str = 'E8M0', #ExMy
        # scaling_mode: ScaleCalculationMode = ScaleCalculationMode.CEIL,
        # use_approx: dict = {'smooth': False}
