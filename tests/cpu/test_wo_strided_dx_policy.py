from __future__ import annotations


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.delenv(
        "USE_TK_LOCALCTA_V4_WO_ATTN_LAYOUT_STRIDED_DX", raising=False
    )
    from low_bits_training.quantization import fused_te_linear

    return fused_te_linear


def test_wo_strided_dx_is_enabled_by_default(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    assert fused_te_linear.use_tk_localcta_v4_wo_attn_layout_strided_dx()


def test_wo_strided_dx_honors_explicit_override(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.setenv("USE_TK_LOCALCTA_V4_WO_ATTN_LAYOUT_STRIDED_DX", "0")
    assert not fused_te_linear.use_tk_localcta_v4_wo_attn_layout_strided_dx()
    monkeypatch.setenv("USE_TK_LOCALCTA_V4_WO_ATTN_LAYOUT_STRIDED_DX", "1")
    assert fused_te_linear.use_tk_localcta_v4_wo_attn_layout_strided_dx()
