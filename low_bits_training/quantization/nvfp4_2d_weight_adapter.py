"""Layout conversion for TE 2D NVFP4 weights consumed by TK GEMMs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _swizzle_2d_nvfp4_scales_for_tk_kernel(
    row_source,
    col_source,
    row_output,
    col_output,
    n_elements: tl.constexpr,
    row_ntk: tl.constexpr,
    col_ntk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    within_tile = offsets % 512
    quartet = within_tile // 4
    source_row = (quartet % 4) * 32 + quartet // 4
    source_k = within_tile % 4
    tile = offsets // 512

    row_tm = tile // row_ntk
    row_tk = tile % row_ntk
    row_source_offsets = (
        ((row_tm * 128 + source_row) * row_ntk + row_tk) * 4 + source_k
    )
    col_tm = tile // col_ntk
    col_tk = tile % col_ntk
    col_source_offsets = (
        ((col_tm * 128 + source_row) * col_ntk + col_tk) * 4 + source_k
    )

    mask = offsets < n_elements
    row_values = tl.load(row_source + row_source_offsets, mask=mask)
    col_values = tl.load(col_source + col_source_offsets, mask=mask)
    tl.store(row_output + offsets, row_values, mask=mask)
    tl.store(col_output + offsets, col_values, mask=mask)


def swizzle_2d_nvfp4_scales_for_tk(
    row_scales: torch.Tensor,
    col_scales: torch.Tensor,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert both TE scale orientations directly into TK's tiled layout."""
    if not row_scales.is_cuda or not col_scales.is_cuda:
        raise RuntimeError("2D NVFP4 scale conversion requires CUDA tensors")
    if row_scales.device != col_scales.device:
        raise RuntimeError("row and column 2D NVFP4 scales must share a device")
    if row_scales.dtype != torch.uint8 or col_scales.dtype != torch.uint8:
        raise TypeError("2D NVFP4 scales must use TE's uint8 E4M3 storage")
    if not row_scales.is_contiguous() or not col_scales.is_contiguous():
        raise RuntimeError("2D NVFP4 scales must be contiguous")
    if rows % 128 != 0 or cols % 128 != 0:
        raise RuntimeError(
            f"2D NVFP4 scale conversion requires 128-aligned dimensions, got {(rows, cols)}"
        )
    expected_row_shape = (rows, cols // 16)
    expected_col_shape = (cols, rows // 16)
    if tuple(row_scales.shape) != expected_row_shape:
        raise RuntimeError(
            f"unexpected row scale shape {tuple(row_scales.shape)}; "
            f"expected {expected_row_shape}"
        )
    if tuple(col_scales.shape) != expected_col_shape:
        raise RuntimeError(
            f"unexpected column scale shape {tuple(col_scales.shape)}; "
            f"expected {expected_col_shape}"
        )
    if row_scales.numel() != col_scales.numel():
        raise RuntimeError("row and column 2D NVFP4 scale payloads must be equal-sized")

    row_output = torch.empty(
        (rows // 128, cols // 64, 512),
        dtype=torch.uint8,
        device=row_scales.device,
    )
    col_output = torch.empty(
        (cols // 128, rows // 64, 512),
        dtype=torch.uint8,
        device=col_scales.device,
    )
    n_elements = row_scales.numel()
    block_size = 4096
    _swizzle_2d_nvfp4_scales_for_tk_kernel[
        (triton.cdiv(n_elements, block_size),)
    ](
        row_scales,
        col_scales,
        row_output,
        col_output,
        n_elements=n_elements,
        row_ntk=cols // 64,
        col_ntk=rows // 64,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return row_output, col_output
