from __future__ import annotations

import pytest


class _FakeV5Quantizer:
    def __init__(self) -> None:
        self.calls = []

    def tk_quantize_for_gemm(self, *args):
        return ("raw", args)

    def tk_quantize_for_gemm_opt(self, *args):
        self.calls.append(args)
        return ("calibrated", args)


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    from low_bits_training.quantization import tk_gemm

    return tk_gemm


def test_v5_ffn_scale_target_wraps_generic_quantizer_only_in_scope(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    quantizer = _FakeV5Quantizer()
    monkeypatch.setenv("USE_TK_V5_FFN_SCALE_TARGET", "457")

    assert tk_gemm._maybe_wrap_v5_ffn_quantizer(quantizer) is quantizer
    with tk_gemm.v5_ffn_quant_scope():
        wrapped = tk_gemm._maybe_wrap_v5_ffn_quantizer(quantizer)
        result = wrapped.tk_quantize_for_gemm("input", False, False)

    assert result[0] == "calibrated"
    assert quantizer.calls == [
        ("input", False, False, False, False, "none", False, 42, 0, 457.0)
    ]


@pytest.mark.parametrize("value", ["0", "-1", "513", "nan", "inf", "invalid"])
def test_v5_ffn_scale_target_rejects_invalid_values(
    monkeypatch, value: str
) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_V5_FFN_SCALE_TARGET", value)

    with pytest.raises(ValueError, match=r"finite and in \(0, 512\]"):
        with tk_gemm.v5_ffn_quant_scope():
            tk_gemm._maybe_wrap_v5_ffn_quantizer(_FakeV5Quantizer())


def test_v5_default_scale_keeps_fast_quantizer(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    quantizer = _FakeV5Quantizer()
    monkeypatch.setenv("USE_TK_V5_FFN_SCALE_TARGET", "448")

    with tk_gemm.v5_ffn_quant_scope():
        assert tk_gemm._maybe_wrap_v5_ffn_quantizer(quantizer) is quantizer

    assert quantizer.tk_quantize_for_gemm("input")[0] == "raw"


def test_v5_ffn_scope_does_not_mutate_shared_quantizer(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    quantizer = _FakeV5Quantizer()
    monkeypatch.setenv("USE_TK_V5_FFN_SCALE_TARGET", "457")

    with tk_gemm.v5_ffn_quant_scope():
        wrapped = tk_gemm._maybe_wrap_v5_ffn_quantizer(quantizer)
        assert wrapped.tk_quantize_for_gemm("input")[0] == "calibrated"

    assert quantizer.tk_quantize_for_gemm("input")[0] == "raw"
