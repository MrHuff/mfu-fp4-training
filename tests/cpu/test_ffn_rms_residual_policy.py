from __future__ import annotations

import pytest
import torch


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_GEMM", "1")
    from low_bits_training.quantization import tk_gemm

    return tk_gemm


def test_rms_residual_default_is_production_shape_scoped(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.delenv("USE_TK_FFN_RMS_RESIDUAL_BWD", raising=False)
    monkeypatch.delenv("USE_TK_FFN_LOCALCTA_DELAYED_SPLIT", raising=False)
    monkeypatch.delenv("USE_TK_FFN_H13_TILE_DELAYED_AMAX", raising=False)

    assert tk_gemm.use_tk_ffn_rms_residual_bwd_for_shape(
        32768, 4096, 14336
    )
    assert tk_gemm.use_tk_ffn_rms_residual_bwd_for_shape(
        32768, 4096, 14336, use_localcta=True
    )
    assert not tk_gemm.use_tk_ffn_rms_residual_bwd_for_shape(
        1024, 4096, 14336
    )
    assert not tk_gemm.use_tk_ffn_rms_residual_bwd_for_shape(
        32768, 4096, 5632
    )


@pytest.mark.parametrize(
    "delayed_flag",
    ["USE_TK_FFN_LOCALCTA_DELAYED_SPLIT", "USE_TK_FFN_H13_TILE_DELAYED_AMAX"],
)
def test_localcta_delayed_routes_keep_prior_rms_path(
    monkeypatch,
    delayed_flag: str,
) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_FFN_RMS_RESIDUAL_BWD", "1")
    monkeypatch.delenv("USE_TK_FFN_LOCALCTA_DELAYED_SPLIT", raising=False)
    monkeypatch.delenv("USE_TK_FFN_H13_TILE_DELAYED_AMAX", raising=False)
    monkeypatch.setenv(delayed_flag, "1")

    assert not tk_gemm.use_tk_ffn_rms_residual_bwd_for_shape(
        32768, 4096, 14336, use_localcta=True
    )
    assert tk_gemm.use_tk_ffn_rms_residual_bwd_for_shape(
        32768, 4096, 14336, use_localcta=False
    )


def test_rms_residual_can_be_disabled(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_FFN_RMS_RESIDUAL_BWD", "0")

    assert not tk_gemm.use_tk_ffn_rms_residual_bwd_for_shape(
        32768, 4096, 14336
    )


def test_rms_residual_requires_same_differentiable_alias(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    input_tensor = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)

    assert tk_gemm._ffn_rms_residual_aliases_input(
        input_tensor, input_tensor
    )
    assert tk_gemm._ffn_rms_residual_aliases_input(
        input_tensor, input_tensor.view_as(input_tensor)
    )
    assert not tk_gemm._ffn_rms_residual_aliases_input(
        input_tensor, input_tensor.clone()
    )
    assert not tk_gemm._ffn_rms_residual_aliases_input(
        input_tensor, input_tensor.detach()
    )
    assert not tk_gemm._ffn_rms_residual_aliases_input(input_tensor, None)
