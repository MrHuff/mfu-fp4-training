from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest


def _load_tk_gemm(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    monkeypatch.delenv(
        "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", raising=False
    )
    from low_bits_training.quantization import tk_gemm

    return tk_gemm


def test_qkv_direct_wgrad_layout_is_explicit_only(monkeypatch) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)

    assert not tk_gemm.use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", "1"
    )
    assert tk_gemm.use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", "0"
    )
    assert not tk_gemm.use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()


def test_qkv_direct_wgrad_layout_requires_localcta_backend(monkeypatch) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", "1"
    )
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")

    assert not tk_gemm.use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()


def test_highwater_profile_enables_qkv_direct_wgrad_layout(monkeypatch) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("LBT_LOCALCTA_V4_PROFILE", "highwater")
    monkeypatch.setenv("USE_TK_FFN_DISABLE_WGRAD_STREAM", "1")
    monkeypatch.delenv("USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO", raising=False)
    from low_bits_training.quantization import fp4_converter

    assert fp4_converter.apply_localcta_v4_profile_defaults() == "highwater"
    assert tk_gemm.use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
    assert os.environ["USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER"] == "1"
    assert os.environ["USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD"] == "0"
    assert os.environ["USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD"] == "0"
    assert os.environ["USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO"] == "0"


def test_highwater_profile_preserves_explicit_qkv_direct_override(
    monkeypatch,
) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("LBT_LOCALCTA_V4_PROFILE", "highwater")
    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", "0"
    )
    monkeypatch.setenv("USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER", "0")
    from low_bits_training.quantization import fp4_converter

    assert fp4_converter.apply_localcta_v4_profile_defaults() == "highwater"
    assert not tk_gemm.use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
    assert os.environ["USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER"] == "0"


def test_highwater_profile_preserves_explicit_split2_clear_skip(
    monkeypatch,
) -> None:
    _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("LBT_LOCALCTA_V4_PROFILE", "highwater")
    monkeypatch.setenv("USE_TK_FFN_DISABLE_WGRAD_STREAM", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO", "1")
    from low_bits_training.quantization import fp4_converter

    assert fp4_converter.apply_localcta_v4_profile_defaults() == "highwater"
    assert os.environ["USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO"] == "1"


def test_direct_qkv_wgrad_return_is_allocated_before_side_stream(
    monkeypatch,
) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    source = inspect.getsource(tk_gemm.tk_fused_qkv_backward)
    policy = source.index("use_localcta_direct_wgrad =")
    allocation = source.index("grad_w_qkv = torch.empty(", policy)
    side_stream = source.index("with torch.cuda.stream(wgrad_stream):", policy)

    assert allocation < side_stream


def test_localcta_v4_grouped_qkv_uses_row_gradient_sr(monkeypatch) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("NVFP4_RNG_SEED", "42")
    monkeypatch.setenv("NVFP4_RNG_SUBSEQUENCE_BASE", "17")
    calls = []

    def quantize(*args):
        calls.append(args)
        return "quantized"

    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_group_quantize_dim1_split3_for_gemm=quantize,
        )
    )
    inputs = (object(), object(), object())

    assert adapter.tk_group_quantize_dim1_split3_for_gemm(*inputs) == "quantized"
    assert calls == [(*inputs, True, 42, 17, "row")]


def test_localcta_v4_opt_quantizer_forwards_optional_checkpointed_sr_state(
    monkeypatch,
) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    calls = []
    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_quantize_for_gemm_opt=(
                lambda *args: calls.append(args) or "quantized"
            ),
        )
    )
    args = (object(), True, True, True, False, "none", False, 42, 17, "row")

    assert adapter.tk_quantize_for_gemm_opt(*args) == "quantized"
    assert calls == [args]

    calls.clear()
    state = object()
    assert adapter.tk_quantize_for_gemm_opt(*args, state) == "quantized"
    assert calls == [(*args, state)]


def test_localcta_v4_grouped_qkv_forwards_checkpointed_sr_state(
    monkeypatch,
) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    calls = []

    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_group_quantize_dim1_split3_for_gemm=(
                lambda *args: calls.append(args) or "quantized"
            ),
        )
    )
    inputs = (object(), object(), object())
    state = object()

    assert (
        adapter.tk_group_quantize_dim1_split3_for_gemm(*inputs, state)
        == "quantized"
    )
    assert calls[0][-1] is state


def test_localcta_v4_inverse_rope_qkv_uses_row_gradient_sr(
    monkeypatch,
) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("NVFP4_RNG_SEED", "42")
    monkeypatch.setenv("NVFP4_RNG_SUBSEQUENCE_BASE", "17")
    calls = []

    def quantize(*args):
        calls.append(args)
        return "quantized"

    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64=(
                quantize
            ),
        )
    )
    inputs = (object(), object(), object())
    rope = object()

    assert (
        adapter.tk_group_quantize_dim1_split3_for_gemm_inverse_rope_live64(
            *inputs, rope, 8192
        )
        == "quantized"
    )
    assert calls == [
        (*inputs, rope, 8192, True, False, "none", False, 42, 17, "row")
    ]


def test_localcta_v4_inverse_rope_forwards_checkpointed_sr_state(
    monkeypatch,
) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    calls = []
    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64=(
                lambda *args: calls.append(args) or "quantized"
            ),
        )
    )
    inputs = (object(), object(), object())
    rope = object()
    state = object()

    assert (
        adapter.tk_group_quantize_dim1_split3_for_gemm_inverse_rope_live64(
            *inputs, rope, 8192, state
        )
        == "quantized"
    )
    assert calls[0][-1] is state


def test_qkv_paired_col_rht_adapter_matches_selected_producer(monkeypatch) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    final_calls = []
    atomic_calls = []
    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_quantize_for_gemm_final_sg=lambda *args: None,
            tk_localcta_quantize_for_gemm_final_sg_paired_col_rht=(
                lambda *args: final_calls.append(args) or "paired-final"
            ),
            tk_localcta_quantize_for_gemm_atomic_paired_col_rht=(
                lambda *args: atomic_calls.append(args) or "paired-atomic"
            ),
        )
    )
    activation = object()

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_DIRECT_CONTRACT", "0")
    assert adapter.supports_paired_col_rht_direct_forward()
    assert (
        adapter.tk_quantize_for_gemm_direct_forward_paired_col_rht(
            activation, True, False
        )
        == "paired-final"
    )
    assert final_calls == [(activation, True, False)]
    assert atomic_calls == []

    with pytest.raises(RuntimeError, match="return_transpose=True"):
        adapter.tk_quantize_for_gemm_direct_forward_paired_col_rht(
            activation, False, True
        )

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER", "0")
    assert adapter.supports_paired_col_rht_direct_forward()
    assert (
        adapter.tk_quantize_for_gemm_direct_forward_paired_col_rht(activation)
        == "paired-atomic"
    )
    assert atomic_calls == [(activation, True, True)]

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_DIRECT_CONTRACT", "1")
    assert not adapter.supports_paired_col_rht_direct_forward()


def test_qkv_paired_col_rht_requires_route_matched_extension_symbol(monkeypatch) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_DIRECT_CONTRACT", "0")

    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_quantize_for_gemm_final_sg=lambda *args: None,
            tk_localcta_quantize_for_gemm_atomic_paired_col_rht=lambda *args: None,
        )
    )

    assert not adapter.supports_paired_col_rht_direct_forward()

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER", "0")
    assert adapter.supports_paired_col_rht_direct_forward()

    for contract in ("tilegrid256", "tilegrid", "2d"):
        monkeypatch.setenv("USE_TK_LOCALCTA_V3_CONTRACT", contract)
        assert not adapter.supports_paired_col_rht_direct_forward()



def test_qkv_forward_keeps_two_pass_paired_col_rht_fallback(monkeypatch) -> None:
    _load_tk_gemm(monkeypatch)
    from low_bits_training.quantization import fused_te_linear

    source = inspect.getsource(fused_te_linear._FusedQKVFunction_TK.forward)
    route = source[
        source.index('qkv_activation_path = "localCTA QKV activation producer"') :
        source.index("_tk_stage_trace('qkv_fwd_sub', 'act_quant_done'")
    ]
    native_branch = route[
        route.index("if native_paired_col_rht:") : route.index("else:")
    ]
    fallback_branch = route[route.index("else:") :]

    assert "supports_paired_col_rht_direct_forward" in route
    assert "tk_quantize_for_gemm_direct_forward_paired_col_rht" in native_branch
    assert "_localcta_quantized_from_result" in native_branch
    assert "native_route_matched_paired" in native_branch
    assert "tk_quantize_for_gemm_direct_forward(" in fallback_branch
    assert "_localcta_replace_col_with_paired_rht(" in fallback_branch
    assert "python_two_pass_fallback" in fallback_branch
    assert "path=qkv_activation_path" in fallback_branch


def test_w2_native_paired_col_rht_adapter_is_fail_closed(monkeypatch) -> None:
    tk_gemm = _load_tk_gemm(monkeypatch)
    calls = []
    adapter = tk_gemm._LocalCTAQuantAdapter(
        SimpleNamespace(
            tk_localcta_silu_supports_paired_col_rht=lambda: True,
            tk_localcta_silu_quantize_split_for_gemm_paired_col_rht=(
                lambda *args: calls.append(args) or "paired-w2"
            ),
        )
    )
    h1_raw = object()
    h3 = object()

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_SILU_ATOMIC_FINAL_SG_PRODUCER", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE", "0")
    monkeypatch.setenv("NVTE_NVFP4_ENCODE_CENTRIC", "0")
    assert adapter.supports_silu_paired_col_rht()
    assert (
        adapter.tk_silu_quantize_split_for_gemm_paired_col_rht(h1_raw, h3)
        == "paired-w2"
    )
    assert calls == [(h1_raw, h3)]

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE", "1")
    assert not adapter.supports_silu_paired_col_rht()
    with pytest.raises(RuntimeError, match="virtual rescale off"):
        adapter.tk_silu_quantize_split_for_gemm_paired_col_rht(h1_raw, h3)

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE", "0")
    monkeypatch.setenv("NVTE_NVFP4_ENCODE_CENTRIC", "1")
    assert not adapter.supports_silu_paired_col_rht()

    monkeypatch.setenv("NVTE_NVFP4_ENCODE_CENTRIC", "0")
    monkeypatch.setenv("USE_TK_LOCALCTA_V4_FUSED_SILU_RAW", "0")
    assert not adapter.supports_silu_paired_col_rht()

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_FUSED_SILU_RAW", "1")
    for contract in ("tilegrid256", "tilegrid", "2d"):
        monkeypatch.setenv("USE_TK_LOCALCTA_V3_CONTRACT", contract)
        assert not adapter.supports_silu_paired_col_rht()

    monkeypatch.setenv("USE_TK_LOCALCTA_V3_CONTRACT", "outer")
    aligned = SimpleNamespace(shape=(256, 512))
    unaligned = SimpleNamespace(shape=(128, 512))
    assert adapter.supports_silu_paired_col_rht(aligned, aligned)
    assert not adapter.supports_silu_paired_col_rht(unaligned, unaligned)


def test_w2_forward_keeps_two_pass_paired_col_rht_fallback(monkeypatch) -> None:
    _load_tk_gemm(monkeypatch)
    from low_bits_training.quantization import fused_te_linear

    source = inspect.getsource(
        fused_te_linear._localcta_silu_quantize_split_for_gemm
    )
    native_end = source.index(
        "result = tk_q.tk_silu_quantize_split_for_gemm(h1_raw, h3)"
    )
    native_branch = source[:native_end]
    fallback_branch = source[native_end:]

    assert "supports_silu_paired_col_rht" in native_branch
    assert "tk_silu_quantize_split_for_gemm_paired_col_rht" in native_branch
    assert "_localcta_quantized_from_result" in native_branch
    assert "native_fused_paired" in native_branch
    assert "fused_silu_mul_bf16_out_no_amax" in fallback_branch
    assert "_localcta_replace_col_with_paired_rht" in fallback_branch
    assert "python_two_pass_fallback" in fallback_branch
