from __future__ import annotations

import pytest


class _FakeLocalCTAQuantizer:
    def __init__(self) -> None:
        self.global_scale_num = None

    def tk_localcta_set_global_scale_num(self, value: float) -> None:
        self.global_scale_num = value


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    from low_bits_training.quantization import tk_gemm

    return tk_gemm


def test_localcta_global_scale_is_applied(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    quantizer = _FakeLocalCTAQuantizer()
    monkeypatch.setenv("USE_TK_LOCALCTA_SCALE_NUM", "774")

    tk_gemm._maybe_apply_localcta_quant_tuning(quantizer)

    assert quantizer.global_scale_num == 774.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_localcta_global_scale_rejects_invalid_values(monkeypatch, value: str) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA_SCALE_NUM", value)

    with pytest.raises(ValueError, match="finite positive number"):
        tk_gemm._maybe_apply_localcta_quant_tuning(_FakeLocalCTAQuantizer())


def test_localcta_global_scale_requires_runtime_setter(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA_SCALE_NUM", "774")

    with pytest.raises(RuntimeError, match="does not expose global-scale control"):
        tk_gemm._maybe_apply_localcta_quant_tuning(object())
