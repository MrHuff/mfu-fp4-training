from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from low_bits_training import metrics
from low_bits_training.cce import backend as cce_backend
from low_bits_training import trainer as trainer_module


class _RecordingMetrics:
    def __init__(self):
        self.records = []

    def delayed_log(self, values):
        self.records.append(values)


def test_common_eval_logs_exact_backend_and_bf16_pair(monkeypatch):
    recorder = _RecordingMetrics()
    localized = []

    class FakeBF16Backend:
        def __init__(self, **kwargs):
            assert kwargs["forward_precision"] == "bf16"
            assert kwargs["backward_precision"] == "bf16"

        def training_loss(self, hidden, weight, labels):
            assert not hidden.requires_grad
            assert not weight.requires_grad
            assert hidden.is_contiguous()
            assert weight.is_contiguous()
            assert labels.is_contiguous()
            assert hidden.dtype == torch.bfloat16
            assert weight.dtype == torch.bfloat16
            assert labels.dtype == torch.int64
            return torch.tensor(2.5)

    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL", "1")
    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL_EVERY", "1")
    monkeypatch.setattr(cce_backend, "_COMMON_EVAL_COUNTER", 0)
    monkeypatch.setattr(
        cce_backend,
        "_NativeMXFP4PrecisionBackend",
        FakeBF16Backend,
    )
    monkeypatch.setattr(
        cce_backend,
        "_local_tensor_for_cce",
        lambda tensor: localized.append(tensor) or tensor,
    )
    monkeypatch.setattr(metrics, "get_metrics_processor", lambda: recorder)

    cce_backend._queue_common_eval_metric(
        torch.randn(3, 4).t(),
        torch.randn(3, 5).t(),
        torch.tensor([0, 99, 1, 99, 2, 99, 3, 99])[::2],
        -100,
        torch.tensor(2.0),
    )

    assert recorder.records == [
        {
            "eval_backend/loss": 2.0,
            "eval_bf16/loss": 2.5,
            "eval_gap/bf16_minus_backend": 0.5,
            "eval_gap/abs_bf16_minus_backend": 0.5,
            "eval_gap/relative_abs_bf16": 0.2,
            "eval_gap/bf16_over_backend": 1.25,
        }
    ]
    assert len(localized) == 3


def test_common_eval_can_use_memory_bounded_triton_bf16(monkeypatch):
    recorder = _RecordingMetrics()

    class FakeTritonBF16Backend:
        def __init__(self, **kwargs):
            assert kwargs == {"ignore_index": -100, "filter_eps": 0.0}

        def training_loss(self, hidden, weight, labels):
            assert hidden.dtype == torch.bfloat16
            assert weight.dtype == torch.bfloat16
            return torch.tensor(2.25)

    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL", "1")
    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL_EVERY", "1")
    monkeypatch.setenv(
        "LBT_FP4_CCE_COMMON_EVAL_BACKEND", "triton_bf16"
    )
    monkeypatch.setattr(cce_backend, "_COMMON_EVAL_COUNTER", 0)
    monkeypatch.setattr(
        cce_backend,
        "_TritonBF16Backend",
        FakeTritonBF16Backend,
    )
    monkeypatch.setattr(metrics, "get_metrics_processor", lambda: recorder)

    cce_backend._queue_common_eval_metric(
        torch.randn(4, 3),
        torch.randn(5, 3),
        torch.tensor([0, 1, 2, 3]),
        -100,
        torch.tensor(2.0),
    )

    assert recorder.records[0]["eval_bf16/loss"] == pytest.approx(2.25)


@pytest.mark.parametrize(
    ("filter_eps", "expected"),
    [(0.0, None), ("none", None), ("auto", torch.finfo(torch.bfloat16).eps / 32)],
)
def test_triton_bf16_passes_explicit_filter_setting(
    monkeypatch, filter_eps, expected
):
    recorded = {}

    def fake_linear_cross_entropy(hidden, weight, labels, **kwargs):
        recorded.update(kwargs)
        return hidden.sum() + weight.sum() + labels.sum() * 0

    monkeypatch.setattr(
        cce_backend, "linear_cross_entropy", fake_linear_cross_entropy
    )
    backend = cce_backend._TritonBF16Backend(
        ignore_index=-100, filter_eps=filter_eps
    )
    backend.training_loss(
        torch.randn(4, 3),
        torch.randn(5, 3),
        torch.tensor([0, 1, 2, 3]),
    )

    assert recorded["filter_eps"] == expected


def test_saved_bf16_backward_explicitly_disables_filtering(monkeypatch):
    recorded = {}

    def fake_linear_cross_entropy(hidden, weight, labels, **kwargs):
        recorded.update(kwargs)
        return hidden.sum() + weight.sum() + labels.sum() * 0

    monkeypatch.setattr(
        cce_backend,
        "_get_triton_linear_cross_entropy",
        lambda: fake_linear_cross_entropy,
    )
    cce_backend._bf16_cce_backward_from_saved(
        torch.randn(4, 3),
        torch.randn(5, 3),
        torch.tensor([0, 1, 2, 3]),
        -100,
        0.0,
        torch.tensor(1.0),
    )

    assert recorded["filter_eps"] is None


def test_common_eval_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL_BACKEND", "unknown")

    with pytest.raises(
        ValueError, match="LBT_FP4_CCE_COMMON_EVAL_BACKEND"
    ):
        cce_backend._common_eval_backend()


def test_common_eval_rejects_excessive_relative_loss_gap(monkeypatch):
    recorder = _RecordingMetrics()

    class FakeBF16Backend:
        def __init__(self, **_kwargs):
            pass

        def training_loss(self, hidden, weight, labels):
            return torch.tensor(2.5)

    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL", "1")
    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL_EVERY", "1")
    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL_MAX_REL_GAP", "0.1")
    monkeypatch.setattr(cce_backend, "_COMMON_EVAL_COUNTER", 0)
    monkeypatch.setattr(cce_backend, "_NativeMXFP4PrecisionBackend", FakeBF16Backend)
    monkeypatch.setattr(metrics, "get_metrics_processor", lambda: recorder)

    with pytest.raises(RuntimeError, match="paired CCE relative loss gap"):
        cce_backend._queue_common_eval_metric(
            torch.randn(4, 3),
            torch.randn(5, 3),
            torch.tensor([0, 1, 2, 3]),
            -100,
            torch.tensor(2.0),
        )


def test_fused_nvfp4_x_producer_preserves_distinct_row_and_col_scales(
    monkeypatch,
):
    row_sg = torch.tensor([[2.0], [3.0]])
    col_sg = torch.tensor([[5.0, 7.0]])

    def fake_quantize(pre_norm, norm_weight, epsilon, encode_centric):
        assert epsilon == pytest.approx(1e-5)
        assert encode_centric
        normed = pre_norm * norm_weight
        row_q = SimpleNamespace(
            fp4=torch.zeros(2, 2, dtype=torch.uint8),
            sc=torch.ones(1, 1, 1),
            sg=row_sg,
        )
        col_q = SimpleNamespace(
            fp4=torch.zeros(2, 2, dtype=torch.uint8),
            sc=torch.ones(1, 1, 1),
            sg=col_sg,
        )
        return normed, row_q, col_q, torch.ones(2), torch.tensor(1.0)

    monkeypatch.setattr(
        cce_backend,
        "_load_fp4_cce_tk_v4",
        lambda: SimpleNamespace(
            quantize_nvfp4_norm_row_and_col_with_output_tk=fake_quantize
        ),
    )

    normed, x_q, x_col_q = cce_backend._produce_final_norm_x_with_quant(
        torch.ones(2, 2),
        torch.ones(2),
        1e-5,
        SimpleNamespace(name="nvfp4", quant_mode="enc"),
    )

    assert torch.equal(normed, torch.ones(2, 2))
    assert torch.equal(x_q.sg, row_sg)
    assert torch.equal(x_col_q.sg, col_sg)


def test_common_eval_guard_does_not_depend_on_metrics_processor(monkeypatch):
    class FakeBF16Backend:
        def __init__(self, **_kwargs):
            pass

        def training_loss(self, hidden, weight, labels):
            return torch.tensor(2.5)

    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL", "1")
    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL_EVERY", "1")
    monkeypatch.setenv("LBT_FP4_CCE_COMMON_EVAL_MAX_REL_GAP", "0.1")
    monkeypatch.setattr(cce_backend, "_COMMON_EVAL_COUNTER", 0)
    monkeypatch.setattr(cce_backend, "_NativeMXFP4PrecisionBackend", FakeBF16Backend)
    monkeypatch.setattr(metrics, "get_metrics_processor", lambda: None)

    with pytest.raises(RuntimeError, match="paired CCE relative loss gap"):
        cce_backend._queue_common_eval_metric(
            torch.randn(4, 3),
            torch.randn(5, 3),
            torch.tensor([0, 1, 2, 3]),
            -100,
            torch.tensor(2.0),
        )


def test_cce_head_passes_optimized_loss_to_paired_evaluation(monkeypatch):
    observed = {}

    class FakeBackend:
        ignore_index = -100

        def training_loss(self, hidden, weight, labels):
            return hidden.sum() * 0.0 + 1.75

    def capture(*args):
        observed["backend_loss"] = args[-1]

    monkeypatch.setattr(cce_backend, "_queue_common_eval_metric", capture)
    head = cce_backend.TitanCCEHead(nn.Linear(3, 5, bias=False), FakeBackend())
    loss = head(torch.randn(2, 3), labels=torch.tensor([1, 2]))

    assert observed["backend_loss"] is loss
    assert loss.item() == pytest.approx(1.75)


def test_cce_trainer_returns_microbatch_scaled_loss(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.loss = torch.tensor(12.0, requires_grad=True)

        def __call__(self, input_ids, labels=None):
            return self.loss

    model = FakeModel()
    trainer = SimpleNamespace(
        job_config=object(),
        model_parts=[model],
        gradient_accumulation_steps=3,
        step=1,
    )
    monkeypatch.setattr(trainer_module, "cce_path_handles_loss", lambda config: True)

    reported = trainer_module.Trainer.forward_backward_step(
        trainer,
        {"input": torch.ones(1, dtype=torch.long)},
        torch.ones(1, dtype=torch.long),
    )

    assert reported.item() == pytest.approx(4.0)
    assert model.loss.grad.item() == pytest.approx(1.0 / 3.0)
