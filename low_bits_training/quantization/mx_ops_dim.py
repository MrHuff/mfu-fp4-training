from typing import Union, List
import torch
from .dimensionQuantisationClass import DimensionMXTensor # Assuming your dataclass is here

# Note: The MXTensor/DimensionMXTensor dataclass and its to_dtype() method 
# are required for these functions to work.

def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix) -> torch.Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        use_triton_kernel: Whether to use a triton implementation instead of relying on
            torch.compile

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    #this is fishy as fuck, work backwards and figure out scaled matmul API and requirments
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten(1)


def mx_mm(a: DimensionMXTensor, b: DimensionMXTensor) -> torch.Tensor:
    """
    Performs matrix multiplication on two MXTensors.

    This operation dequantizes both input tensors to FP32 precision,
    performs the mm in FP32 for accurate accumulation (matching TE),
    then converts back to original dtype.
    """
    assert isinstance(a, DimensionMXTensor) and isinstance(b, DimensionMXTensor), "Inputs must be DimensionMXTensor"
    assert a._gemm_kernel_choice == b._gemm_kernel_choice, "GEMM kernel choice must match between tensors"

    # Dequantize tensors to FP32 first
    a_hp = a.to_dtype(torch.float32)
    b_hp = b.to_dtype(torch.float32)

    # Convert to BF16 for Matmul to match TE Reference behavior (BF16 Inputs)
    # TE Ref dequantizes to BF16, then runs Matmul (effectively BF16 inputs).
    # Matmul accumulation is usually FP32 on Ampere+ for BF16.
    a_bf16 = a_hp.to(torch.bfloat16)
    b_bf16 = b_hp.to(torch.bfloat16)

    return torch.mm(a_bf16, b_bf16).to(a._orig_dtype)

def mx_matmul(a: DimensionMXTensor, b: DimensionMXTensor) -> torch.Tensor:
    """
    Performs matrix multiplication on two MXTensors.

    This operation dequantizes both input tensors to FP32 precision,
    performs the matmul in FP32 for accurate accumulation (matching TE),
    then converts back to original dtype.
    """
    
    # Dequantize tensors to FP32 for accurate accumulation
    a_hp = a.to_dtype(torch.float32)
    b_hp = b.to_dtype(torch.float32)

    return torch.matmul(a_hp, b_hp).to(a._orig_dtype)

def mx_matmul_right_transpose(a: DimensionMXTensor, b: DimensionMXTensor) -> torch.Tensor:
    """
    Performs block-wise matrix multiplication matching TE's NVFP4 GEMM.
    
    This implements the TE algorithm:
    - Loop over K dimension in blocks of block_size
    - For each block: y += outer(sx_block, sw_block) * gemm(qx_block, qw_block)
    - Apply global scale (gA * gB / factor) at the end
    """
    a_hp = a.to_dtype(a._orig_dtype) #note swapping to float32 fucks a lot of shit up.
    b_hp = b.to_dtype(b._orig_dtype)
    
    # print(f"DEBUG mx_mm_rt: a_hp mean={a_hp.mean().item():.6f}, b_hp mean={b_hp.mean().item():.6f}")

    return torch.matmul(a_hp, b_hp.T)

def mx_matmul_fp8(a: DimensionMXTensor, b: DimensionMXTensor) -> torch.Tensor:
    """
    Performs matrix multiplication accelerated by hardware FP8.
    It takes MX-tensors containing FP4-quantized data, casts them to FP8 for
    the GEMM operation, and then dequantizes the result.
    """

    #Test torchao later and figure out the shapes!

    return torch.matmul(a, b)
    
    

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
        _global_abs_mask=tensor._global_abs_mask
    )

def mx_view(tensor: DimensionMXTensor, new_size: Union[torch.Size, List[int]]) -> DimensionMXTensor:
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
        _global_abs_mask=tensor._global_abs_mask
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
        _global_abs_mask=tensor._global_abs_mask
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
    assert dtype in {torch.float16, torch.bfloat16}, "Only support float16/bfloat16 conversion for MXTensor autocast"

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
        _global_abs_mask=tensor._global_abs_mask
    )