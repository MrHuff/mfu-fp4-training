from __future__ import annotations

import torch

from low_bits_training.cce import backend as cce_backend


class _LowPrecisionBackend:
    name = "nvfp4"
    implementation = "v4"
    ignore_index = -100
    filter_eps = 0.0

    def __init__(self) -> None:
        self.hidden_requires_grad: bool | None = None

    def training_loss_prequantized_x(self, hidden, x_q, x_col_q, weight, labels):
        del x_q, x_col_q, labels
        self.hidden_requires_grad = hidden.requires_grad
        return (hidden @ weight.transpose(0, 1)).sum()


class _Norm(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16)
        )
        self.eps = 1e-5

    def forward(self, x):
        return x * self.weight


def _enable_selective_candidate(monkeypatch) -> None:
    for name in (
        "LBT_FP4_CCE_BF16_DHIDDEN_ONLY",
        "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER",
        "FP4_CCE_V4_NVFP4_MXFP8_FORWARD",
        "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE",
        "FP4_CCE_V4_MIXED_DW_MXFP8_COLS",
        "FP4_CCE_V4_NVFP4_G_TARGET_SPLIT",
        "FP4_CCE_V4_NVFP4_FUSED_G_CACHE",
        "FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW",
        "FP4_CCE_V4_EXACT_TARGET_TOPK_LOGITS",
        "FP4_CCE_V4_MX_COMPACT_DW_REPAIR",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_QUANT_BACKEND", "localcta_v4")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_FUSED_G_CACHE_IMPL", "tiled")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_G_TOPK_SPLIT", "16")
    monkeypatch.setenv("FP4_CCE_V4_EXACT_SELECTED_TOPK", "16")


def test_selective_backward_preserves_lowp_loss_and_dweight(
    monkeypatch,
) -> None:
    _enable_selective_candidate(monkeypatch)
    expected_dhidden_value = 7.0
    helper_calls = []

    def fake_bf16_dhidden(
        hidden,
        weight,
        labels,
        ignore_index,
        filter_eps,
        grad_output,
    ):
        helper_calls.append(
            (hidden.detach().clone(), weight.detach().clone(), labels.detach().clone())
        )
        assert ignore_index == -100
        assert filter_eps == 0.0
        return torch.full_like(hidden, expected_dhidden_value) * grad_output

    monkeypatch.setattr(
        cce_backend,
        "_bf16_cce_dhidden_from_saved",
        fake_bf16_dhidden,
    )

    def fake_producer(pre_norm, norm_weight, epsilon, backend):
        del epsilon, backend
        return pre_norm * norm_weight, object(), object()

    monkeypatch.setattr(
        cce_backend,
        "_produce_final_norm_x_with_quant",
        fake_producer,
    )

    original = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    backend = _LowPrecisionBackend()
    head = cce_backend.TitanCCEHead(original, backend)
    norm = _Norm(3)
    x = torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3).requires_grad_(True)
    labels = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    expected_hidden = x.detach().reshape(-1, 3) * norm.weight.detach()
    expected_loss = (expected_hidden @ head.weight.detach().T).sum()
    loss = head.forward_from_pre_norm(x, norm, labels=labels)
    torch.testing.assert_close(loss.detach(), expected_loss)
    loss.backward()

    assert backend.hidden_requires_grad is False
    assert len(helper_calls) == 1
    torch.testing.assert_close(
        x.grad,
        torch.full_like(x, expected_dhidden_value),
    )
    expected_dweight_row = expected_hidden.sum(dim=0)
    torch.testing.assert_close(
        head.weight.grad,
        expected_dweight_row.expand_as(head.weight),
    )


def test_selective_backward_rejects_ordinary_forward(monkeypatch) -> None:
    monkeypatch.setenv("LBT_FP4_CCE_BF16_DHIDDEN_ONLY", "1")
    original = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    backend = _LowPrecisionBackend()
    head = cce_backend.TitanCCEHead(original, backend)
    x = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    try:
        head(x, labels=labels)
    except RuntimeError as exc:
        assert "requires the fused final-norm prequantized-X path" in str(exc)
    else:  # pragma: no cover - defensive failure message
        raise AssertionError("selective backward accepted the ordinary CCE path")


def test_selective_backward_rejects_unmeasured_cache_format(monkeypatch) -> None:
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    backend = _LowPrecisionBackend()

    try:
        cce_backend._validate_bf16_dhidden_only_backend(
            backend,
            prequantized_x=True,
        )
    except RuntimeError as exc:
        assert "missing:" in str(exc)
        assert "FP4_CCE_V4_MIXED_DW_MXFP8_COLS" in str(exc)
    else:  # pragma: no cover - defensive failure message
        raise AssertionError("selective backward accepted an unmeasured cache format")


def _enable_internal_bf16_dhidden_candidate(monkeypatch) -> None:
    _enable_selective_candidate(monkeypatch)
    monkeypatch.setenv("LBT_FP4_CCE_BF16_DHIDDEN_ONLY", "0")
    monkeypatch.setenv("FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN", "1")
    monkeypatch.setenv("FP4_CCE_V4_SPARSE_REPAIR_OVERLAP", "0")
    monkeypatch.setenv("FP4_CCE_V4_MX_BACKWARD_GEMM_OVERLAP", "0")


def test_internal_bf16_dhidden_keeps_lowp_autograd_connected(monkeypatch) -> None:
    _enable_internal_bf16_dhidden_candidate(monkeypatch)

    def fake_producer(pre_norm, norm_weight, epsilon, backend):
        del epsilon, backend
        return pre_norm * norm_weight, object(), object()

    monkeypatch.setattr(
        cce_backend,
        "_produce_final_norm_x_with_quant",
        fake_producer,
    )
    original = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    backend = _LowPrecisionBackend()
    head = cce_backend.TitanCCEHead(original, backend)
    norm = _Norm(3)
    x = torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3).requires_grad_(True)
    labels = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    loss = head.forward_from_pre_norm(x, norm, labels=labels)
    loss.backward()

    assert backend.hidden_requires_grad is True
    assert x.grad is not None
    assert head.weight.grad is not None
    expected_hidden = x.detach().reshape(-1, 3) * norm.weight.detach()
    expected_dweight_row = expected_hidden.sum(dim=0)
    torch.testing.assert_close(
        head.weight.grad,
        expected_dweight_row.expand_as(head.weight),
    )


def test_internal_bf16_dhidden_rejects_ordinary_forward(monkeypatch) -> None:
    monkeypatch.setenv("FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN", "1")
    original = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    backend = _LowPrecisionBackend()
    head = cce_backend.TitanCCEHead(original, backend)
    x = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    try:
        head(x, labels=labels)
    except RuntimeError as exc:
        assert "requires the fused final-norm prequantized-X path" in str(exc)
    else:  # pragma: no cover - defensive failure message
        raise AssertionError("internal BF16 dHidden accepted ordinary CCE")


def test_internal_and_outer_bf16_dhidden_flags_conflict(monkeypatch) -> None:
    _enable_selective_candidate(monkeypatch)
    monkeypatch.setenv("FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN", "1")
    backend = _LowPrecisionBackend()

    try:
        cce_backend._validate_lowp_logits_bf16_dhidden_backend(
            backend,
            prequantized_x=True,
        )
    except RuntimeError as exc:
        assert "conflicts with LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1" in str(exc)
    else:  # pragma: no cover - defensive failure message
        raise AssertionError("two BF16 dHidden implementations were enabled")
