#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Union, List
import torch
from .dimensionQuantisationClass import (
    DimensionMXTensor,
)  # Assuming your dataclass is here

# Note: The MXTensor/DimensionMXTensor dataclass and its to_dtype() method
# are required for these functions to work.


def mx_mm(a: DimensionMXTensor, b: DimensionMXTensor) -> torch.Tensor:
    """
    Performs matrix multiplication on two MXTensors.

    This operation dequantizes both input tensors to their original precision
    and then performs a standard torch.mm. It returns a high-precision tensor.
    """
    assert isinstance(a, DimensionMXTensor) and isinstance(
        b, DimensionMXTensor
    ), "Inputs must be DimensionMXTensor"
    assert (
        a._gemm_kernel_choice == b._gemm_kernel_choice
    ), "GEMM kernel choice must match between tensors"

    # Dequantize tensors to their original data type for the multiplication
    a_hp = a.to_dtype(a._orig_dtype)
    b_hp = b.to_dtype(b._orig_dtype)

    return torch.mm(a_hp, b_hp)


def mx_matmul(a: DimensionMXTensor, b: DimensionMXTensor) -> torch.Tensor:
    """
    Performs matrix multiplication on two MXTensors.

    This operation dequantizes both input tensors to their original precision
    and then performs a standard torch.matmul. It returns a high-precision tensor.
    """
    assert isinstance(a, DimensionMXTensor) and isinstance(
        b, DimensionMXTensor
    ), "Inputs must be DimensionMXTensor"
    assert (
        a._gemm_kernel_choice == b._gemm_kernel_choice
    ), "GEMM kernel choice must match between tensors"

    # Dequantize tensors to their original data type for the multiplication
    a_hp = a.to_dtype(a._orig_dtype)
    b_hp = b.to_dtype(b._orig_dtype)

    return torch.matmul(a_hp, b_hp)


def mx_t(tensor: DimensionMXTensor) -> DimensionMXTensor:
    """
    Transposes an MXTensor.

    This operation transposes the underlying quantized data tensor (_data)
    and returns a new MXTensor with the same metadata.
    """
    assert isinstance(tensor, DimensionMXTensor), "Input must be a DimensionMXTensor"

    # Create a new MXTensor with transposed data but identical metadata
    return DimensionMXTensor(
        _data=tensor._data.t(),
        _scale_fp=tensor._scale_fp,
        _scale_lp=tensor._scale_lp,
        _elem_dtype=tensor._elem_dtype,
        _block_size=tensor._block_size,
        _block_dim=tensor._block_dim,
        _orig_dtype=tensor._orig_dtype,
        _use_fp4_custom_triton_dequant_kernel=tensor._use_fp4_custom_triton_dequant_kernel,
        _gemm_kernel_choice=tensor._gemm_kernel_choice,
        _max_abs=tensor._max_abs,
        _max_abs_mask=tensor._max_abs_mask,
        _sm=tensor._sm,
        _g=tensor._g,
        _global_abs_mask=tensor._global_abs_mask,
    )


def mx_view(
    tensor: DimensionMXTensor, new_size: Union[torch.Size, List[int]]
) -> DimensionMXTensor:
    """
    Changes the view of an MXTensor.

    This operation calls .view() on the underlying quantized data tensor (_data)
    and returns a new MXTensor with the same metadata.
    """
    assert isinstance(tensor, DimensionMXTensor), "Input must be a DimensionMXTensor"

    # Create a new MXTensor with the new view but identical metadata
    return DimensionMXTensor(
        _data=tensor._data.view(new_size),
        _scale_fp=tensor._scale_fp,
        _scale_lp=tensor._scale_lp,
        _elem_dtype=tensor._elem_dtype,
        _block_size=tensor._block_size,
        _block_dim=tensor._block_dim,
        _orig_dtype=tensor._orig_dtype,
        _use_fp4_custom_triton_dequant_kernel=tensor._use_fp4_custom_triton_dequant_kernel,
        _gemm_kernel_choice=tensor._gemm_kernel_choice,
        _max_abs=tensor._max_abs,
        _max_abs_mask=tensor._max_abs_mask,
        _sm=tensor._sm,
        _g=tensor._g,
        _global_abs_mask=tensor._global_abs_mask,
    )


def mx_detach(tensor: DimensionMXTensor) -> DimensionMXTensor:
    """
    Detaches an MXTensor from the computation graph.

    This operation calls .detach() on the underlying quantized data tensor (_data)
    and returns a new MXTensor with the same metadata.
    """
    assert isinstance(tensor, DimensionMXTensor), "Input must be a DimensionMXTensor"

    # Create a new MXTensor with detached data but identical metadata
    return DimensionMXTensor(
        _data=tensor._data.detach(),
        _scale_fp=tensor._scale_fp,
        _scale_lp=tensor._scale_lp,
        _elem_dtype=tensor._elem_dtype,
        _block_size=tensor._block_size,
        _block_dim=tensor._block_dim,
        _orig_dtype=tensor._orig_dtype,
        _use_fp4_custom_triton_dequant_kernel=tensor._use_fp4_custom_triton_dequant_kernel,
        _gemm_kernel_choice=tensor._gemm_kernel_choice,
        _max_abs=tensor._max_abs,
        _max_abs_mask=tensor._max_abs_mask,
        _sm=tensor._sm,
        _g=tensor._g,
        _global_abs_mask=tensor._global_abs_mask,
    )


def mx_sum(tensor: DimensionMXTensor, *args, **kwargs) -> torch.Tensor:
    """
    Performs a sum operation on an MXTensor.

    This is a fallback operation that dequantizes the input tensor to its
    original precision and then performs a standard torch.sum.
    """
    assert isinstance(tensor, DimensionMXTensor), "Input must be a DimensionMXTensor"

    # Dequantize the tensor and then perform the operation
    tensor_hp = tensor.to_dtype(tensor._orig_dtype)
    return torch.sum(tensor_hp, *args, **kwargs)


def mx_to(tensor: DimensionMXTensor, dtype: torch.dtype) -> DimensionMXTensor:
    """
    Handles autocast conversion for an MXTensor.

    This mimics the `_to_copy` behavior by changing the target dequantization
    dtype (_orig_dtype) without altering the quantized data. It's useful
    when working inside a `torch.autocast` block.
    """
    assert isinstance(tensor, DimensionMXTensor), "Input must be a DimensionMXTensor"
    assert dtype in {
        torch.float16,
        torch.bfloat16,
    }, "Only support float16/bfloat16 conversion for MXTensor autocast"

    # Return a new MXTensor with the `_orig_dtype` metadata updated
    return DimensionMXTensor(
        _data=tensor._data,
        _scale_fp=tensor._scale_fp,
        _scale_lp=tensor._scale_lp,
        _elem_dtype=tensor._elem_dtype,
        _block_size=tensor._block_size,
        _block_dim=tensor._block_dim,
        _orig_dtype=dtype,  # Update the target dtype
        _use_fp4_custom_triton_dequant_kernel=tensor._use_fp4_custom_triton_dequant_kernel,
        _gemm_kernel_choice=tensor._gemm_kernel_choice,
        _max_abs=tensor._max_abs,
        _max_abs_mask=tensor._max_abs_mask,
        _sm=tensor._sm,
        _g=tensor._g,
        _global_abs_mask=tensor._global_abs_mask,
    )
