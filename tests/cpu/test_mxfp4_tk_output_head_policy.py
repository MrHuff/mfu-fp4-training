from __future__ import annotations

import pytest
import torch
from torch import nn

from low_bits_training.quantization.float32_linear import Float32Linear
from low_bits_training.quantization import mxfp4_tk_converter as converter
from low_bits_training.quantization import mixed_fp4_converter as mixed_converter


class _HeadOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Start in FP32 so the safe converter route itself must enforce BF16.
        self.output = nn.Linear(8, 4, bias=True, dtype=torch.float32)


@pytest.fixture(autouse=True)
def _isolate_head_policy_from_native_route_logging(monkeypatch) -> None:
    """Run each CPU policy test from the sealed production head environment."""

    monkeypatch.setattr(converter, "_log_mxfp4_highwater_route_once", lambda: None)
    monkeypatch.setattr(converter, "_log_mxfp4_rht_route_once", lambda: None)
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "0")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "1")
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_OUTPUT_HEAD", "0")


def _converter_classes():
    return (converter.MXFP4TKBackendConverter, converter.MXFP4TKConverter)


def test_mxfp4_tk_output_head_env_is_fail_safe(monkeypatch) -> None:
    monkeypatch.delenv("USE_FP4_CONVERT_OUTPUT_HEAD", raising=False)
    assert not converter._use_mxfp4_tk_convert_output_head()

    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "0")
    assert not converter._use_mxfp4_tk_convert_output_head()

    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "unexpected")
    assert not converter._use_mxfp4_tk_convert_output_head()

    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "1")
    assert converter._use_mxfp4_tk_convert_output_head()


def test_mxfp4_tk_backend_and_fused_keep_exact_bf16_linear(monkeypatch) -> None:
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "0")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "1")

    for converter_class in _converter_classes():
        model = _HeadOnlyModel()
        original_head = model.output

        converter_class(None, None).convert(model)

        assert model.output is original_head
        assert type(model.output) is nn.Linear
        assert model.output.weight.dtype is torch.bfloat16
        assert model.output.bias is not None
        assert model.output.bias.dtype is torch.bfloat16
        logits = model.output(torch.ones(2, 8, dtype=torch.bfloat16))
        assert logits.dtype is torch.bfloat16


def test_mxfp4_tk_required_llama_head_cannot_pass_vacuously(monkeypatch) -> None:
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "0")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "1")

    for converter_class in _converter_classes():
        with pytest.raises(RuntimeError, match="requires exactly one root output module"):
            converter_class(None, None).convert(nn.Identity())


def test_mxfp4_tk_required_llama_head_rejects_legacy_route(monkeypatch) -> None:
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "1")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "1")

    for converter_class in _converter_classes():
        with pytest.raises(RuntimeError, match="conflicts with an explicitly enabled"):
            converter_class(None, None).convert(_HeadOnlyModel())


def test_mxfp4_tk_backend_and_fused_require_explicit_legacy_head_opt_in(
    monkeypatch,
) -> None:
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "1")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "0")
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_OUTPUT_HEAD", "0")

    for converter_class in _converter_classes():
        model = _HeadOnlyModel()

        converter_class(None, None).convert(model)

        assert type(model.output) is Float32Linear
        logits = model.output(torch.ones(2, 8, dtype=torch.bfloat16))
        assert logits.dtype is torch.float32


def test_mixed_localcta_mxfp4_keeps_exact_bf16_linear_by_default(monkeypatch) -> None:
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "0")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "1")
    model = _HeadOnlyModel()
    original_head = model.output

    replacements = mixed_converter._replace_output_heads(model, set())

    assert replacements == 0
    assert model.output is original_head
    assert type(model.output) is nn.Linear
    assert model.output.weight.dtype is torch.bfloat16
    assert model.output(torch.ones(2, 8, dtype=torch.bfloat16)).dtype is torch.bfloat16


def test_mixed_localcta_mxfp4_legacy_head_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "1")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "0")
    model = _HeadOnlyModel()

    replacements = mixed_converter._replace_output_heads(model, set())

    assert replacements == 1
    assert type(model.output) is Float32Linear


def test_mixed_localcta_mxfp4_rejects_legacy_head_in_production(monkeypatch) -> None:
    monkeypatch.setenv("USE_FP4_CONVERT_OUTPUT_HEAD", "1")
    monkeypatch.setenv("MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD", "1")

    with pytest.raises(RuntimeError, match="conflicts with an explicitly enabled"):
        mixed_converter._replace_output_heads(_HeadOnlyModel(), set())
