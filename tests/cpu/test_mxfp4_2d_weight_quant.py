from __future__ import annotations

import pytest
import torch


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    from low_bits_training.quantization import mxfp4_fused_linear

    return mxfp4_fused_linear


def test_mxfp4_2d_weight_quant_is_explicitly_opt_in(monkeypatch) -> None:
    mxfp4 = _load(monkeypatch)

    monkeypatch.delenv("MXFP4_USE_2D_WEIGHT_QUANT", raising=False)
    assert not mxfp4.use_mxfp4_2d_weight_quant()

    monkeypatch.setenv("MXFP4_USE_2D_WEIGHT_QUANT", "1")
    assert mxfp4.use_mxfp4_2d_weight_quant()

    monkeypatch.setenv("MXFP4_USE_2D_WEIGHT_QUANT", "0")
    assert not mxfp4.use_mxfp4_2d_weight_quant()


def test_mxfp4_weight_quant_routes_to_native_2d_producer(monkeypatch) -> None:
    mxfp4 = _load(monkeypatch)
    monkeypatch.setenv("MXFP4_USE_2D_WEIGHT_QUANT", "1")
    monkeypatch.setenv("MXFP4_USE_WEIGHT_QUANT_CACHE", "0")
    weight = torch.empty((256, 128), dtype=torch.bfloat16)
    outputs = (
        torch.empty((256, 64), dtype=torch.uint8),
        torch.empty((2, 1, 32, 16), dtype=torch.uint8),
        torch.empty((128, 128), dtype=torch.uint8),
        torch.empty((1, 2, 32, 16), dtype=torch.uint8),
    )
    calls = []

    def quantize(value):
        calls.append(value)
        return outputs

    monkeypatch.setattr(mxfp4, "mxfp4_quantize_weight_2d", quantize)

    result = mxfp4._quantize_weight_row_col_bf16(weight, mode=1)

    assert calls == [weight]
    assert result.row_fp4 is outputs[0]
    assert result.row_sc is outputs[1]
    assert result.col_fp4 is outputs[2]
    assert result.col_sc is outputs[3]


def test_mxfp4_2d_weight_quant_rejects_non_encode_mode(monkeypatch) -> None:
    mxfp4 = _load(monkeypatch)
    monkeypatch.setenv("MXFP4_USE_2D_WEIGHT_QUANT", "1")
    monkeypatch.setenv("MXFP4_USE_WEIGHT_QUANT_CACHE", "0")

    with pytest.raises(RuntimeError, match="requires encode-centric mode"):
        mxfp4._quantize_weight_row_col_bf16(
            torch.empty((128, 128), dtype=torch.bfloat16),
            mode=0,
        )
