from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_GEMM", "1")
    from low_bits_training.quantization import tk_gemm

    return tk_gemm


def test_v5_residual_default_is_shape_scoped(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.delenv("MXFP4_BACKEND_VERSION", raising=False)
    monkeypatch.delenv("USE_TK_V5_FFN_RESIDUAL_EPILOGUE", raising=False)

    assert tk_gemm.use_tk_v5_ffn_residual_epilogue_for_shape(
        32768, 4096, 14336
    )
    assert not tk_gemm.use_tk_v5_ffn_residual_epilogue_for_shape(
        1024, 4096, 14336
    )
    assert not tk_gemm.use_tk_v5_ffn_residual_epilogue_for_shape(
        32768, 4096, 5632
    )


def test_v5_residual_is_disabled_for_other_backends(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_V5_FFN_RESIDUAL_EPILOGUE", "1")

    monkeypatch.setenv("USE_TK_LOCALCTA", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    assert not tk_gemm.use_tk_v5_ffn_residual_epilogue()

    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.setenv("MXFP4_BACKEND_VERSION", "v4")
    assert not tk_gemm.use_tk_v5_ffn_residual_epilogue()


def test_selected_v5_residual_missing_symbol_fails_before_write(
    monkeypatch,
) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.delenv("MXFP4_BACKEND_VERSION", raising=False)
    monkeypatch.setenv("USE_TK_V5_FFN_RESIDUAL_EPILOGUE", "1")
    monkeypatch.setattr(
        tk_gemm,
        "_get_tk_plain",
        lambda: SimpleNamespace(nvfp4_gemm=lambda *args: None),
    )
    x = SimpleNamespace(
        shape=(32768, 14336),
        _tk_row=(SimpleNamespace(shape=(32768, 7168)), object(), object()),
    )
    w = SimpleNamespace(
        shape=(4096, 14336),
        _tk_row=(SimpleNamespace(shape=(4096, 7168)), object(), object()),
    )
    residual = torch.full((2, 2), 3.0, dtype=torch.bfloat16)
    output = torch.full((2, 2), 7.0, dtype=torch.bfloat16)
    before = output.clone()

    with pytest.raises(RuntimeError, match="requires nvfp4_gemm_residual"):
        tk_gemm.tk_forward_gemm_residual(
            x, w, residual, out=output, use_localcta=False
        )

    assert torch.equal(output, before)


@pytest.mark.parametrize(
    ("enabled", "x_shape", "w_shape"),
    [
        (False, (32768, 7168), (4096, 7168)),
        (True, (1024, 7168), (4096, 7168)),
    ],
)
def test_opt_out_and_unsupported_shape_keep_split_route(
    monkeypatch,
    enabled: bool,
    x_shape: tuple[int, int],
    w_shape: tuple[int, int],
) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.delenv("MXFP4_BACKEND_VERSION", raising=False)
    monkeypatch.setenv(
        "USE_TK_V5_FFN_RESIDUAL_EPILOGUE", "1" if enabled else "0"
    )
    calls = {"split": 0}

    def split(*args):
        calls["split"] += 1
        args[-1].fill_(2.0)

    monkeypatch.setattr(
        tk_gemm,
        "_get_tk_plain",
        lambda: SimpleNamespace(nvfp4_gemm=split),
    )
    x = SimpleNamespace(
        shape=(x_shape[0], x_shape[1] * 2),
        _tk_row=(SimpleNamespace(shape=x_shape), object(), object()),
    )
    w = SimpleNamespace(
        shape=(w_shape[0], w_shape[1] * 2),
        _tk_row=(SimpleNamespace(shape=w_shape), object(), object()),
    )
    residual = torch.full((2, 2), 3.0, dtype=torch.bfloat16)
    output = torch.full((2, 2), -1.0, dtype=torch.bfloat16)

    result = tk_gemm.tk_forward_gemm_residual(
        x, w, residual, out=output, use_localcta=False
    )

    assert result is output
    assert calls == {"split": 1}
    assert torch.equal(output, torch.full_like(output, 5.0))
