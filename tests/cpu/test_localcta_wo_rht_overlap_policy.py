from __future__ import annotations

import inspect


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    from low_bits_training.quantization import fused_te_linear

    return fused_te_linear


def test_localcta_wo_rht_weight_overlap_is_explicit_and_default_off(
    monkeypatch,
) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.delenv(
        "USE_TK_LOCALCTA_V4_WO_RHT_WEIGHT_QUANT_OVERLAP", raising=False
    )
    assert not fused_te_linear.use_tk_localcta_v4_wo_rht_weight_quant_overlap()

    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_WO_RHT_WEIGHT_QUANT_OVERLAP", "1"
    )
    assert fused_te_linear.use_tk_localcta_v4_wo_rht_weight_quant_overlap()


def test_localcta_wo_overlap_route_is_rht_only_and_pointer_safe(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    source = inspect.getsource(fused_te_linear._WoFunction_TK.forward)

    selection = source[
        source.index("overlap_localcta_v4_wo_weight_quant = (") :
        source.index("if use_attn_layout_view:")
    ]
    assert "use_localcta_v4" in selection
    assert 'use_nvfp4_rht_for_role("activation")' in selection
    assert "use_tk_localcta_v4_wo_rht_weight_quant_overlap()" in selection
    assert '_nvfp4_quantizer_extras_enabled("weight")' in selection
    assert "use_cuda_graph()" in selection

    branch = source[
        source.index("elif overlap_localcta_v4_wo_weight_quant:") :
        source.index("elif use_tk_ms():")
    ]
    launch_wait = branch.index("s1.wait_stream(s0)")
    source_record = branch.index("_record_tensors_on_stream((wo_weight,), s1)")
    weight_quant = branch.index("w_nvfp4 = _fast_quantize(")
    activation_quant = branch.index("x_nvfp4 = _fast_quantize(")
    consumer_wait = branch.index("s0.wait_stream(s1)")
    result_record = branch.index("_record_tensors_on_stream(", source_record + 1)

    assert launch_wait < source_record < weight_quant < consumer_wait
    assert source_record < activation_quant < consumer_wait
    assert consumer_wait < result_record
    assert "w_nvfp4._tk_row" in branch[result_record:]
    assert "w_nvfp4._tk_col" in branch[result_record:]
    assert "w_nvfp4._keepalive" in branch[result_record:]
