from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).with_name("tk_quantize.cu").read_text()
HEADER = Path(__file__).with_name("fused_localcta_quantize.cuh").read_text()


def _between(start: str, end: str) -> str:
    begin = SOURCE.index(start)
    return SOURCE[begin : SOURCE.index(end, begin)]


def _compact(value: str) -> str:
    return "".join(value.split())


def test_atomic_paired_col_rht_is_a_default_off_direct_producer_extension() -> None:
    producer = _between(
        "void quantize_into_outputs_v3_atomic_final_sg(",
        "void quantize_rmsnorm_into_outputs_opt(",
    )

    assert "bool paired_fixed_sign_col_rht = false" in producer
    assert "paired column-RHT atomic producer requires return_transpose=True" in producer
    assert "M == input_rows && K == input_cols" in producer

    paired = producer[
        producer.index("if (paired_fixed_sign_col_rht) {", producer.index("#undef")) :
        producer.index("} else if (encode_centric)")
    ]
    compact = _compact(paired)
    assert (
        "launch_localcta_quant_opt<"
        "true,true,false,false,false,false,true,true,"
        "false,false,false,false,true>"
    ) in compact
    assert (
        "launch_localcta_quant_opt<"
        "true,false,false,false,false,false,true,true,"
        "false,false,false,false,true>"
    ) in compact

    # The established direct row path remains available verbatim when the new
    # flag is false; this is the byte-exact baseline for the CUDA regression.
    old_launch_tree = producer[producer.index("} else if (encode_centric)") :]
    assert (
        "launch_localcta_quant_opt<"
        "true,true,false,false,false,false,false,false,"
        "false,false,false,false,true>"
    ) in _compact(old_launch_tree)


def test_atomic_paired_col_rht_public_api_is_narrow_and_bound() -> None:
    wrapper = _between(
        "tk_localcta_quantize_for_gemm_atomic_paired_col_rht(",
        "tk_localcta_quantize_for_gemm_opt(",
    )
    compact = _compact(wrapper)

    assert "pairedcolumn-RHTatomicproducerrequiresreturn_transpose=True" in compact
    assert (
        "quantize_into_outputs_v3_atomic_final_sg("
        "input,return_transpose,encode_centric,"
        "row_fp4,row_sc,col_fp4,col_sc,row_sg,col_sg,-1,-1,true);"
    ) in compact
    assert SOURCE.count(
        'm.def("tk_localcta_quantize_for_gemm_atomic_paired_col_rht"'
    ) == 1


def test_final_sg_paired_col_rht_public_api_is_route_matched_and_bound() -> None:
    wrapper = _between(
        "tk_localcta_quantize_for_gemm_final_sg_paired_col_rht(",
        "tk_localcta_quantize_for_gemm_opt(",
    )
    compact = _compact(wrapper)

    assert "pairedcolumn-RHTfinal-SGproducerrequiresreturn_transpose=True" in compact
    assert (
        "quantize_into_outputs_v3_final_sg("
        "input,false,encode_centric,"
        "row_fp4,row_sc,col_fp4,col_sc,row_sg,col_sg);"
    ) in compact
    assert (
        "quantize_col_into_outputs_v3_opt_fixed_sign_rht("
        "input,encode_centric,col_fp4,col_sc,col_sg);"
    ) in compact
    assert SOURCE.count(
        'm.def("tk_localcta_quantize_for_gemm_final_sg_paired_col_rht"'
    ) == 1


def test_final_sg_paired_col_rht_suppresses_legacy_col_and_row_payload_emission() -> None:
    producer = _between(
        "static void quantize_col_into_outputs_v3_opt_fixed_sign_rht(",
        "tk_localcta_quantize_for_gemm_final_sg_paired_col_rht(",
    )
    compact = _compact(producer)

    assert (
        "launch_localcta_quant_opt<"
        "true,true,false,false,false,false,true,true,"
        "false,false,false,false,false,false,false>"
    ) in compact
    assert (
        "launch_localcta_quant_opt<"
        "true,false,false,false,false,false,true,true,"
        "false,false,false,false,false,false,false>"
    ) in compact
    assert "row_sg_chunk_scratch" in producer
    assert "use_localcta_v4_split_finalize_single()" in producer
    assert "finalize_col_quant_contract_v3(col_sc, col_sg_chunk, col_sg)" in producer


def test_final_sg_opt_random_sign_routes_remain_fail_closed() -> None:
    scan = _between(
        "static void launch_scan_single_sg_opt_impl(",
        "static void launch_scan_single_sg_opt(",
    )
    api = _between(
        "tk_localcta_quantize_for_gemm_final_sg_opt(",
        "tk_localcta_quantize_col_for_gemm_final_sg_opt(",
    )

    assert (
        'TORCH_CHECK(!random_sign, "final-SG opt scan does not support random RHT signs yet")'
        in scan
    )
    assert (
        'TORCH_CHECK(!with_random_sign_mask,\n'
        '                "final-SG opt producer does not support random RHT signs yet")'
        in api
    )


def test_callfree_te_silu_math_is_paired_rht_only() -> None:
    kernel = HEADER[
        HEADER.index("fused_localcta_silu_quantize_raw_kernel(") :
        HEADER.index("fused_localcta_quantize_split2_prepared_kernel(")
    ]

    assert 'asm("ex2.approx.ftz.f32 %0, %1;"' in HEADER
    assert "if constexpr (CALL_FREE_TE_MATH)" in HEADER
    assert "__uint_as_float(0x83949c56u)" in HEADER
    assert "__uint_as_float(0x83354df1u)" in HEADER
    assert "__uint_as_float(0x82dd2eb1u)" in HEADER
    assert "SERIAL_PAIRS" not in HEADER
    assert HEADER.count(
        "template <int TOTAL_THREADS, bool CALL_FREE_TE_MATH = false>"
    ) == 3
    assert HEADER.count("localcta_silu_value<CALL_FREE_TE_MATH>") == 4
    assert kernel.count("SILU_RAW_THREADS, COL_WITH_RHT>") == 2


def test_callfree_scale_math_is_paired_rht_production_only() -> None:
    kernel = HEADER[
        HEADER.index("fused_localcta_silu_quantize_raw_kernel(") :
        HEADER.index("fused_localcta_quantize_split2_prepared_kernel(")
    ]
    helper = HEADER[
        HEADER.index("localcta_scale_divide_callfree(") :
        HEADER.index("template <bool CALL_FREE_SCALE_MATH = false>")
    ]

    assert 'asm("rcp.approx.ftz.f64 %0, %1;"' in helper
    assert helper.count("reciprocal = fma(reciprocal_error, reciprocal, reciprocal)") == 2
    assert "quotient = fma(quotient_error, reciprocal, quotient)" in helper
    assert "const float quotient_f32 = static_cast<float>(quotient)" in helper
    assert "numerator_abs_bits > kInf || denominator_abs_bits > kInf" in helper
    assert "denominator_abs_bits == 0u" in helper
    assert "numerator_abs_bits == kInf" in helper
    assert "denominator_abs_bits == kInf || numerator_abs_bits == 0u" in helper
    assert "CALL_FREE_SCALE_MATH = false" in HEADER

    scale_and_coeff = HEADER[
        HEADER.index("localcta_compute_scale_and_coeff(") :
        HEADER.index("localcta_compute_four_over_six_candidates(")
    ]
    decode_callfree = (
        "const float S_dec_b = localcta_scale_divide<true>(\n"
        "                    block_amax, fp4_max) * S_enc;"
    )
    assert decode_callfree in scale_and_coeff
    assert (
        "S_b_fp8 = static_cast<nvfp4_scale_t>(fminf(\n"
        "                    S_dec_b,\n"
        "                    transformer_engine::detail::TypeExtrema<float>::max));"
    ) in scale_and_coeff
    assert "} else {\n                S_b_fp8 = compute_decoding_scaling_factor(" in scale_and_coeff
    assert scale_and_coeff.count("compute_decoding_scaling_factor(") == 1

    # Only the sealed paired-RHT W2 specialization forwards COL_WITH_RHT to
    # the call-free scale-math template parameters.  Defaults preserve every
    # pre-existing localCTA caller.
    assert kernel.count(
        "compute_localcta_encode_scaling_factor_FP4<\n"
        "                COL_WITH_RHT>"
    ) == 4
    assert kernel.count("localcta_scale_divide<COL_WITH_RHT>(") == 2
    assert kernel.count(
        "THREADS, ENCODE_CENTRIC, COL_WITH_RHT>"
    ) == 2
    assert kernel.count(
        "true, COL_WITH_RANDOM_SIGN_MASK, false, false,\n"
        "                            COL_WITH_RHT>"
    ) == 1
    assert kernel.count(
        "true, COL_WITH_RANDOM_SIGN_MASK, false, false,\n"
        "                                COL_WITH_RHT>"
    ) == 1
    assert HEADER.count("compute_localcta_encode_scaling_factor_FP4<") == 4
    assert SOURCE.count('m.def("tk_localcta_test_scale_divide_callfree"') == 1


def test_parallel_silu_quant_loop_limits_address_liveness() -> None:
    kernel = HEADER[
        HEADER.index("fused_localcta_silu_quantize_raw_kernel(") :
        HEADER.index("fused_localcta_quantize_split2_prepared_kernel(")
    ]
    compact = _compact(kernel)

    # CUDA 13.2 hoists all four tiles' row/column scale addresses when this
    # loop is fully unrolled.  Only the parallel producer is rolled; every
    # structurally different specialization preserves the four-way unroll.
    assert kernel.count("constexpr int quant_loop_unroll =") == 1
    assert (
        "constexprintquant_loop_unroll="
        "(PARALLEL_ROW_COL&&RETURN_TRANSPOSE)?1:NUM_TILES;"
        "#pragmaunrollquant_loop_unroll"
        "for(intt=0;t<NUM_TILES;++t)"
    ) in compact
