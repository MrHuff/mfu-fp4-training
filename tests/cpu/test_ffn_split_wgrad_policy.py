from __future__ import annotations


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.delenv("USE_TK_FFN_SPLIT_WGRAD_EAGER", raising=False)
    monkeypatch.delenv("USE_TK_FFN_V5_DELAYED_DIRECT_SPLIT", raising=False)
    from low_bits_training.quantization import tk_gemm

    return tk_gemm


def test_regular_v5_defaults_to_eager_split_wgrad(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)

    assert tk_gemm.use_tk_ffn_split_wgrad_eager()


def test_delayed_v5_defaults_to_batched_split_wgrad(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_FFN_V5_DELAYED_DIRECT_SPLIT", "1")

    assert not tk_gemm.use_tk_ffn_split_wgrad_eager()


def test_explicit_split_wgrad_override_precedes_route_policy(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_FFN_V5_DELAYED_DIRECT_SPLIT", "1")

    monkeypatch.setenv("USE_TK_FFN_SPLIT_WGRAD_EAGER", "1")
    assert tk_gemm.use_tk_ffn_split_wgrad_eager()
    monkeypatch.setenv("USE_TK_FFN_SPLIT_WGRAD_EAGER", "0")
    assert not tk_gemm.use_tk_ffn_split_wgrad_eager()


def test_localcta_stays_outside_regular_v5_policy(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")

    assert not tk_gemm.use_tk_ffn_split_wgrad_eager()
