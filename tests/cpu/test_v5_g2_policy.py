from __future__ import annotations


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_GEMM", "1")
    monkeypatch.setenv("USE_TK_QUANT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.delenv("USE_TK_FFN_BWD_SAFE_PRODUCER", raising=False)
    monkeypatch.delenv("USE_TK_FFN_V5_DELAYED_SILU_DERIV", raising=False)
    from low_bits_training.quantization import fused_te_linear

    return fused_te_linear


def test_v5_g2_fused_default_is_production_shape_scoped(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    assert not fused_te_linear.use_tk_ffn_bwd_safe_producer(
        32768, 4096, 14336
    )
    assert fused_te_linear.use_tk_ffn_bwd_safe_producer(1024, 4096, 14336)
    assert fused_te_linear.use_tk_ffn_bwd_safe_producer(32768, 4096, 5632)


def test_v5_g2_delayed_route_keeps_its_producer(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_FFN_V5_DELAYED_SILU_DERIV", "1")

    assert fused_te_linear.use_tk_ffn_bwd_safe_producer(
        32768, 4096, 14336
    )


def test_v5_g2_explicit_override_precedes_shape_policy(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.setenv("USE_TK_FFN_BWD_SAFE_PRODUCER", "1")
    assert fused_te_linear.use_tk_ffn_bwd_safe_producer(
        32768, 4096, 14336
    )
    monkeypatch.setenv("USE_TK_FFN_BWD_SAFE_PRODUCER", "0")
    assert not fused_te_linear.use_tk_ffn_bwd_safe_producer(
        1024, 4096, 14336
    )


def test_localcta_does_not_inherit_regular_v5_policy(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")

    assert not fused_te_linear.use_tk_ffn_bwd_safe_producer(
        32768, 4096, 14336
    )
