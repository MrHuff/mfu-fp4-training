#!/usr/bin/env python3
"""Fail-closed source/ABI contracts for the heterogeneous carriers."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
HEADER = (HERE / "mixed_mxfp4_localcta.cuh").read_text()
BINDING = (HERE / "tk_quantize.cu").read_text()
MX_HEADER = (HERE.parent / "mxfp4_v4" / "mxfp4_v3_quantize.cuh").read_text()


def test_grad_is_localcta_row_sr_and_fixed_signed_h32_mx_col() -> None:
    assert "rowwise_scaling_opt<\n            true, false, true, false, false, false, false" in HEADER
    assert "MXMode::ENCODE, false, false, true, 32, true, false" in HEADER
    assert "mixed_grad_localcta_row_mx_col_kernel" in HEADER


def test_activation_and_grad_share_literal_fixed_sign_helper() -> None:
    """The generic X carrier and mixed dY carrier must use the same basis."""
    start = MX_HEADER.index("uint32_t make_rht_sign_bits(")
    end = MX_HEADER.index("__device__ __forceinline__ void fwht16_unnormalized", start)
    helper = MX_HEADER[start:end]
    assert "return 0x00002817u;" in helper
    assert "return next_rbits(" not in helper
    assert "MXMode::ENCODE, false, false, true, 32, true, false" in HEADER


def test_weight_is_exact_two_dimensional_pair() -> None:
    assert "tk_localcta::weight_2d_scaling<true>" in HEADER
    assert "mxfp4_v3::mx_weight_2d_quantize<false>" in HEADER
    assert "mixed_weight_mx_row_localcta_col_kernel" in HEADER


def test_weight_omits_dead_mx_column_payload_and_scale_scratch() -> None:
    """The mixed ABI consumes MX row + localCTA col and nothing else."""
    weight_start = HEADER.index("mixed_weight_mx_row_localcta_col_kernel")
    weight = HEADER[weight_start:]
    assert "kMixedWeightPackedBytes" in weight
    assert "mx_col_sc_scratch" not in weight
    assert "mxfp4_v3::mx_weight_2d_quantize<false>" in weight
    assert "nullptr" in weight


def test_each_mixed_kernel_has_one_bf16_tile_load() -> None:
    grad_start = HEADER.index("mixed_grad_localcta_row_mx_col_kernel")
    weight_start = HEADER.index("mixed_weight_mx_row_localcta_col_kernel")
    grad = HEADER[grad_start:weight_start]
    weight = HEADER[weight_start:]
    assert grad.count("load_mixed_chunk(") == 1
    assert weight.count("load_mixed_chunk(") == 1
    mixed_binding = BINDING[
        BINDING.index("tk_mixed_grad_localcta_row_mx_col_alloc") :
        BINDING.index("PYBIND11_MODULE")
    ]
    assert "torch::cat" not in mixed_binding


def test_grad_uses_double_buffered_grouped_tma_stores() -> None:
    grad_start = HEADER.index("mixed_grad_localcta_row_mx_col_kernel")
    grad_end = HEADER.index("mixed_weight_mx_row_localcta_col_kernel")
    grad = HEADER[grad_start:grad_end]
    loop_start = grad.index("#pragma unroll\n    for (int t = 0;")
    loop_end = grad.index("tk_localcta::swizzle_scales_row_inplace", loop_start)
    loop = grad[loop_start:loop_end]
    assert "const int local_buffer = t & 1" in loop
    assert "const int mx_buffer = t & 1" in loop
    assert loop.count("cp_async_bulk_commit_group()") == 1
    assert loop.count("cp_async_bulk_wait_group_read<1>()") == 1
    assert loop.count("cp_async_bulk_wait_group_read<0>()") == 1


def test_rng_coordinate_is_explicit_and_not_advanced() -> None:
    grad_start = HEADER.index("mixed_grad_localcta_row_mx_col_kernel")
    grad_end = HEADER.index("mixed_weight_mx_row_localcta_col_kernel")
    grad = HEADER[grad_start:grad_end]
    assert "rng_seed" in grad
    assert "rng_subsequence_base" in grad
    assert "atomicAdd" not in grad
    assert "invocation_offset" not in grad
    assert 'result["runtime_advances_rng"] = false' in BINDING
    assert 'result["grad_coordinate_mode"] = "explicit_seed_subsequence"' in BINDING


def test_abi_is_feature_gated_and_inplace() -> None:
    required = (
        "tk_mixed_mx_localcta_capabilities",
        "tk_mixed_grad_localcta_row_mx_col_alloc",
        "tk_mixed_grad_localcta_row_mx_col_launch_inplace",
        "tk_mixed_split2_grad_localcta_row_mx_col_alloc",
        "tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace",
        "tk_mixed_weight_mx_row_localcta_col_alloc",
        "tk_mixed_weight_mx_row_localcta_col_launch_inplace",
    )
    for symbol in required:
        assert BINDING.count(f'"{symbol}"') == 1
    assert 'result["abi_version"] = 1' in BINDING
    assert 'result["grad_mx_col_rht"] = "block32_fixed_0x2817"' in BINDING
    assert 'result["mxfp4_rht_block_size"] = 32' in BINDING
    assert (
        'result["mxfp4_rht_sign_contract"] = "fixed_0x2817_per_h16_half"'
        in BINDING
    )
    assert 'result["weight_mx_2d"] = true' in BINDING
    assert 'result["weight_localcta_2d"] = true' in BINDING
    assert 'result["localcta_encode_mode"] = "encode_centric"' in BINDING
    assert 'result["localcta_sg_contract"] = "outer"' in BINDING
    assert 'result["split2_grad_one_coordinate"] = true' in BINDING
    assert 'result["split2_dgrad_onepass_outer_sg"] = true' in BINDING
    assert 'result["split2_row_outer_sg"] = "per_arm"' in BINDING
    assert 'result["split2_row_rescale"] = "grouped_fp8x4_warp"' in BINDING
    assert 'result["split2_row_rescale_cols_per_block"] = 8' in BINDING
    assert 'result["split2_row_rescale_launches"] = 1' in BINDING


def test_split2_uses_two_inputs_but_one_logical_coordinate() -> None:
    start = BINDING.index("tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace")
    end = BINDING.index("tk_mixed_weight_mx_row_localcta_col_alloc", start)
    body = BINDING[start:end]
    assert "tmap_in0" in body and "tmap_in1" in body
    assert "static_cast<int>(H / 128)" in body
    assert "rng_seed" in body and "rng_subsequence_base" in body
    assert "torch::cat" not in body
    assert "copy_" not in body
    assert (
        '"logical_dim1_concat_per_arm_outer_no_bf16_materialization"'
        in BINDING
    )
    assert body.count("finalize_row_quant_contract_v3_strided_split2(") == 1
    assert body.count("finalize_row_quant_contract_v3_strided(") == 0
    assert "reduce_row_sg_tiles_strided_split2_kernel" in BINDING
    assert "rescale_row_sc_strided_split2_fp8x4_warp_kernel" in BINDING


def test_shapes_and_resources_fail_closed() -> None:
    assert BINDING.count("% 256 == 0") >= 4
    assert BINDING.count("V3ContractMode::TileGrid256") >= 4
    assert HEADER.count("__launch_bounds__(kThreads, 2)") == 2
    assert "mixed_grad_shmem_size" in HEADER
    assert "mixed_weight_shmem_size" in HEADER
