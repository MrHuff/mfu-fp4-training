#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
from typing import Any, Union, List, Callable, Optional, Dict

import torch
import torch.nn as nn

from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger
from torchao.prototype.mx_formats.mx_linear import mx_mm, MXFP8Dim1CastKernelChoice

try:
    from torchao.prototype.mx_formats.mx_linear import MXGemmKernelChoice
except ImportError:
    # TorchAO 0.15 replaced this enum with the shared KernelPreference API.
    # This converter currently selects only EMULATED, which is common to both.
    from torchao.quantization.quantize_.common.kernel_preference import (
        KernelPreference as MXGemmKernelChoice,
    )
from torchao.prototype.mx_formats.mx_tensor import (
    ScaleCalculationMode,
    to_mx,
)  # noqa: F401

from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
)
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from dataclasses import dataclass

_mxfp_dtypes_map = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
    "e2m3": DTYPE_FP6_E2M3,
    "e3m2": DTYPE_FP6_E3M2,
    "e2m1": torch.float4_e2m1fn_x2,
}

SwapFilterFn = Callable[[torch.nn.Module, str], bool]
"""Swap filter function: (module, name) -> swap module?
"""
swap_filter_fn_collection: Dict[str, SwapFilterFn] = {
    "all_but_output": lambda mod, fqn: fqn != "output",
}
"""Collection of swap filter functions.
"""


def to_mxfp_dtype(dtype: str) -> Any:
    """Get the MXFP dtype from string encoding."""
    assert dtype in _mxfp_dtypes_map, f"Unknown MX dtype {dtype}"
    return _mxfp_dtypes_map[dtype]


def mxfp_dtypes() -> List[str]:
    """Get the list of support MX dtypes."""
    return list(_mxfp_dtypes_map.keys())


def _is_linear(mod, fqn):
    return isinstance(mod, torch.nn.Linear)


@dataclass(frozen=True)
class MXLinearConfig:
    """MX linear layer configuration.

    Args:
        adtype: MX activation dtype.
        wdtype: MX weight dtype.
        gdtype: MX gradient dtype.
        block_size: MX block size.
        scale_rounding_fn: Scale rounding mode.
        swap_filter_fn: Swap filter function (e.g. excluding `output` layer by default).
        gemm_kernel_choice: GEMM kernel (emulated, cublas, ...)
    """

    adtype: Any
    wdtype: Any
    gdtype: Any
    block_size: int = 32
    scale_rounding_fn: ScaleCalculationMode = ScaleCalculationMode.FLOOR
    swap_filter_fn: Optional[str] = None
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.EMULATED

    @classmethod
    def from_job_config(cls, job_config: JobConfig) -> "MXLinearConfig":
        """Build from Job config instance."""
        scale_rounding_fn = ScaleCalculationMode[
            job_config.mxfp.scale_rounding_fn.upper()
        ]
        return MXLinearConfig(
            adtype=to_mxfp_dtype(job_config.mxfp.activation_dtype),
            wdtype=to_mxfp_dtype(job_config.mxfp.weight_dtype),
            gdtype=to_mxfp_dtype(job_config.mxfp.gradient_dtype),
            block_size=int(job_config.mxfp.block_size),
            scale_rounding_fn=scale_rounding_fn,
            swap_filter_fn=str(job_config.mxfp.swap_filter_fn),
            gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
        )


def mx_linear(x, w, bias, mx_config: MXLinearConfig):
    """MX linear (pure function).

    A simple pure `mx_linear` function is making it easier for
    users to patch MX logic, and implement custom additional custom logic.
    """
    mxfp8_cast_kernel_choice = MXFP8Dim1CastKernelChoice.TORCH
    y = mx_mm.apply(
        x,
        w,
        mx_config.adtype,
        mx_config.wdtype,
        mx_config.gdtype,
        mx_config.block_size,
        mx_config.gemm_kernel_choice,
        mxfp8_cast_kernel_choice,
        mx_config.scale_rounding_fn,
    )
    if bias is not None:
        y = y + bias
    return y


class MXLinearGeneral(torch.nn.Linear):
    """
    Linear layer with the compute happening in emulate MX. Currently the MX
    matmul is emulated since there is no hardware support yet. Activations,
    weights and grads are casted to MX and back to high precision for each
    matmul.
    """

    @classmethod
    @torch.no_grad()
    def from_float(cls, mod: nn.Module, mx_config: MXLinearConfig):
        mod.__class__ = MXLinearGeneral
        mod.mx_config = mx_config
        return mod

    def forward(self, x):
        return mx_linear(x, self.weight, self.bias, self.mx_config)


def replace_with_custom_fn_if_matches_filter(
    model: nn.Module,
    replacement_fn: Callable[[nn.Module], nn.Module],
    filter_fn: Callable[[nn.Module, str], bool],
    cur_fqn: str = "",
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`

    Inspired by TorchAO implementation.
    """
    for name, child in model.named_children():
        if cur_fqn == "":
            new_fqn = name
        else:
            new_fqn = f"{cur_fqn}.{name}"
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn
            )


def swap_linear_with_quantization_linear(
    model: nn.Module, linear_cls: Any, filter_fn: Any = None, **kwargs
):
    """Swap Torch Linear by custom quantization linear layer.

    Args:
        model
        linear_cls: Quantization linear class.
        filter_fn: Additional filter function.
        **kwargs: Additional factory arguments.
    """
    if filter_fn is None:
        combined_filter_fn = _is_linear
    else:

        def __fn(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        combined_filter_fn = __fn
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: linear_cls.from_float(mod, **kwargs),
        combined_filter_fn,
    )


class MXFPConverter(ModelConverter):
    """MXFP format TorchTitan training integration.

    We use a more general MXGeneralLinear layer than TorchAO, allowing
    parametrization of activation, weight and gradient datatypes separately.

    Using TorchAO MX experimental integration:
        https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        # Extract MX linear config.
        self._mx_config = MXLinearConfig.from_job_config(job_config)

    def convert(self, model: nn.Module):
        mx_config = self._mx_config
        logger.info(
            f"MXFP converter ({mx_config.adtype},{mx_config.wdtype},{mx_config.gdtype},{mx_config.block_size},{mx_config.scale_rounding_fn},{mx_config.swap_filter_fn})."
        )
        # Like FP8, not converting output/projection layer by default
        swap_filter_fn = self._mx_config.swap_filter_fn or "all_but_output"
        swap_linear_with_quantization_linear(
            model=model,
            linear_cls=MXLinearGeneral,
            filter_fn=swap_filter_fn_collection[swap_filter_fn],
            mx_config=self._mx_config,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        # Nothing to do: MX scaling is computed on the fly in Linear layer.
        pass


# Register the MXFP converter in TorchTitan
register_model_converter(MXFPConverter, "mxfp")
