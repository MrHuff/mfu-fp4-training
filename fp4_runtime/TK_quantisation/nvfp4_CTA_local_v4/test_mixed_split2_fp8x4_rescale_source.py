#!/usr/bin/env python3
"""Fail-closed source contracts for the mixed split2 FP8x4 rescaler."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "tk_quantize.cu").read_text()


def _slice(start: str, end: str) -> str:
    begin = SOURCE.index(start)
    finish = SOURCE.index(end, begin)
    return SOURCE[begin:finish]


def test_split2_rescaler_is_grouped_fp8x4_warp_code() -> None:
    kernel = _slice(
        "__global__ void rescale_row_sc_strided_split2_fp8x4_warp_kernel(",
        "__global__ void fold_row_sc_chunk_sg_strided_kernel(",
    )
    assert "template <int COLS_PER_BLOCK = 8, int BLOCK_SIZE = 256>" in SOURCE
    assert "constexpr int kWarpsPerBlock = BLOCK_SIZE / 32" in kernel
    assert "const int split = blockIdx.z" in kernel
    assert "const int sc_col_start = blockIdx.y * COLS_PER_BLOCK" in kernel
    assert "for (int scale_task = warp;" in kernel
    assert "ratio = __shfl_sync(kFullWarpMask, ratio, 0)" in kernel
    assert "for (int packed_group = 0; packed_group < 4; ++packed_group)" in kernel
    assert "rescale_fp8x4_inplace(row_sc + base + packed_i * 4, ratio)" in kernel
    assert "static_cast<__nv_fp8_e4m3>(value * ratio)" not in kernel


def test_split2_rescaler_preserves_per_arm_addresses_and_denominators() -> None:
    kernel = _slice(
        "__global__ void rescale_row_sc_strided_split2_fp8x4_warp_kernel(",
        "__global__ void fold_row_sc_chunk_sg_strided_kernel(",
    )
    required = (
        "split == 0 ? row_sc_0 : row_sc_1",
        "split == 0 ? row_sg_chunk_0 : row_sg_chunk_1",
        "split == 0 ? row_sg_0 : row_sg_1",
        "split == 0 ? row_sc_stride0_0 : row_sc_stride0_1",
        "split == 0 ? row_sc_stride1_0 : row_sc_stride1_1",
        "split == 0 ? row_sg_chunk_stride0_0 : row_sg_chunk_stride0_1",
        "split == 0 ? row_sg_chunk_stride1_0 : row_sg_chunk_stride1_1",
        "const float denom = row_sg[tile]",
        "static_cast<int64_t>(row) * row_sg_chunk_stride0",
        "static_cast<int64_t>(sc_col / 2) * row_sg_chunk_stride1",
        "static_cast<int64_t>(row) * row_sc_stride0",
        "static_cast<int64_t>(sc_col) * row_sc_stride1",
    )
    for marker in required:
        assert marker in kernel
    assert "thread_max" not in kernel
    assert "row_sg[tile] =" not in kernel


def test_split2_finalizer_keeps_exactly_one_reduce_and_one_rescale_launch() -> None:
    finalizer = _slice(
        "static void finalize_row_quant_contract_v3_strided_split2(",
        "static void finalize_quant_contract_v3_split2(",
    )
    assert finalizer.count("<<<") == 2
    assert finalizer.count("reduce_row_sg_tiles_strided_split2_kernel<<<") == 1
    assert (
        finalizer.count(
            "rescale_row_sc_strided_split2_fp8x4_warp_kernel<\n"
            "        kSplit2RescaleColsPerBlock><<<"
        )
        == 1
    )
    assert "constexpr int kSplit2RescaleColsPerBlock = 8" in finalizer
    assert "max_sc_col_blocks" in finalizer
    assert "static_cast<unsigned int>(max_sc_col_blocks)" in finalizer
    assert "2u" in finalizer
    assert "rescale_row_sc_strided_split2_kernel<<<" not in finalizer


def test_split2_fp8x4_preconditions_fail_closed() -> None:
    finalizer = _slice(
        "static void finalize_row_quant_contract_v3_strided_split2(",
        "static void finalize_quant_contract_v3_split2(",
    )
    required = (
        "row_sc_0.scalar_type() == torch::kFloat8_e4m3fn",
        "row_sc_0.dim() == 3",
        "row_sc_0.size(2) == 512",
        "row_sc_0.stride(2) == 1",
        "row_sg_chunk_0.scalar_type() == torch::kFloat32",
        "row_sg_chunk_0.dim() == 2",
        "row_sc_0.size(1) == 2 * row_sg_chunk_0.size(1)",
        "row_sc_0.stride(1) % 4 == 0",
        "reinterpret_cast<uintptr_t>(row_sc_0.data_ptr()) & 0x3u",
        "row_sg_0.numel() == outer_tiles",
    )
    for marker in required:
        assert marker in finalizer


def test_split2_fp8x4_contract_is_advertised() -> None:
    capabilities = _slice(
        "py::dict tk_mixed_mx_localcta_capabilities()",
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)",
    )
    assert 'result["split2_row_rescale"] = "grouped_fp8x4_warp"' in capabilities
    assert 'result["split2_row_rescale_cols_per_block"] = 8' in capabilities
    assert 'result["split2_row_rescale_launches"] = 1' in capabilities


def test_grouped_grid_covers_each_arm_scale_tile_once() -> None:
    cols_per_block = 8
    warps_per_block = 8
    for row_chunks in (1, 2, 3, 256):
        outer_tiles = (row_chunks + 1) // 2
        for sc_cols in (1, 7, 8, 9, 224):
            col_blocks = (sc_cols + cols_per_block - 1) // cols_per_block
            visited: list[tuple[int, int, int]] = []
            for split in range(2):
                for tile in range(outer_tiles):
                    row0 = tile * 2
                    rows_in_tile = min(2, row_chunks - row0)
                    for col_block in range(col_blocks):
                        col0 = col_block * cols_per_block
                        cols_this_block = min(cols_per_block, sc_cols - col0)
                        scale_tasks = rows_in_tile * cols_this_block
                        for warp in range(warps_per_block):
                            for task in range(warp, scale_tasks, warps_per_block):
                                row = row0 + task // cols_this_block
                                col = col0 + task % cols_this_block
                                visited.append((split, row, col))
            expected = [
                (split, row, col)
                for split in range(2)
                for row in range(row_chunks)
                for col in range(sc_cols)
            ]
            assert sorted(visited) == expected


def test_fp8x4_lane_mapping_covers_one_scale_tile_once() -> None:
    offsets = [
        (lane + packed_group * 32) * 4 + component
        for packed_group in range(4)
        for lane in range(32)
        for component in range(4)
    ]
    assert sorted(offsets) == list(range(512))
