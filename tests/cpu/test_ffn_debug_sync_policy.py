from __future__ import annotations

import pytest


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    from low_bits_training.quantization import fused_te_linear

    return fused_te_linear


def test_ffn_debug_sync_is_disabled_by_default(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.delenv("USE_TK_FFN_DEBUG_SYNC_CHECK", raising=False)
    monkeypatch.setattr(
        fused_te_linear.torch.cuda,
        "synchronize",
        lambda: pytest.fail("unexpected CUDA synchronization"),
    )

    fused_te_linear._tk_ffn_debug_sync_checkpoint("ffn_w2_dgrad")


def test_ffn_debug_sync_honors_label_filter(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    calls = []
    monkeypatch.setenv("USE_TK_FFN_DEBUG_SYNC_CHECK", "1")
    monkeypatch.setenv(
        "USE_TK_FFN_DEBUG_SYNC_LABELS",
        "ffn_w2_dgrad, ffn_split_producer",
    )
    monkeypatch.setattr(
        fused_te_linear.torch.cuda,
        "synchronize",
        lambda: calls.append("sync"),
    )

    fused_te_linear._tk_ffn_debug_sync_checkpoint("ffn_dy_quant")
    fused_te_linear._tk_ffn_debug_sync_checkpoint("ffn_w2_dgrad")
    fused_te_linear._tk_ffn_debug_sync_checkpoint("ffn_split_producer")

    assert calls == ["sync", "sync"]


def test_ffn_debug_sync_reports_failing_stage(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_FFN_DEBUG_SYNC_CHECK", "1")
    monkeypatch.delenv("USE_TK_FFN_DEBUG_SYNC_LABELS", raising=False)

    def _raise():
        raise RuntimeError("latent launch failure")

    monkeypatch.setattr(fused_te_linear.torch.cuda, "synchronize", _raise)

    with pytest.raises(
        RuntimeError,
        match="TK FFN debug sync failed after ffn_split_dgrad",
    ):
        fused_te_linear._tk_ffn_debug_sync_checkpoint("ffn_split_dgrad")
