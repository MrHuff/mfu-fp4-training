from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _trainer(lr: float = 3e-4, step: int = 11000):
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scheduler.last_epoch = step
    scheduler._last_lr = [lr]
    optimizer.param_groups[0]["lr"] = lr
    return SimpleNamespace(
        step=step,
        optimizers=SimpleNamespace(optimizers=[optimizer]),
        lr_schedulers=SimpleNamespace(schedulers=[scheduler]),
    )


def _contract(monkeypatch, override: str = "0.00015") -> None:
    monkeypatch.setenv("LBT_RESUME_PEAK_LR_OVERRIDE", override)
    monkeypatch.setenv("LBT_RESUME_PEAK_LR_EXPECTED_STEP", "11000")
    monkeypatch.setenv("LBT_RESUME_PEAK_LR_EXPECTED_CURRENT", "0.0003")


def test_resume_peak_lr_override_updates_optimizer_and_scheduler(monkeypatch) -> None:
    from low_bits_training.trainer import _apply_resume_peak_lr_override

    _contract(monkeypatch)
    trainer = _trainer()
    _apply_resume_peak_lr_override(trainer)

    optimizer = trainer.optimizers.optimizers[0]
    scheduler = trainer.lr_schedulers.schedulers[0]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-4)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(1.5e-4)
    assert scheduler.base_lrs == pytest.approx([1.5e-4])
    assert scheduler.get_last_lr() == pytest.approx([1.5e-4])

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-4)


def test_resume_peak_lr_override_rejects_wrong_step(monkeypatch) -> None:
    from low_bits_training.trainer import _apply_resume_peak_lr_override

    _contract(monkeypatch)
    with pytest.raises(RuntimeError, match="wrong checkpoint step"):
        _apply_resume_peak_lr_override(_trainer(step=10999))


def test_resume_peak_lr_override_rejects_unexpected_checkpoint_lr(monkeypatch) -> None:
    from low_bits_training.trainer import _apply_resume_peak_lr_override

    _contract(monkeypatch)
    with pytest.raises(RuntimeError, match="expected .*lr"):
        _apply_resume_peak_lr_override(_trainer(lr=2e-4))


def test_resume_peak_lr_override_is_dormant_when_unset(monkeypatch) -> None:
    from low_bits_training.trainer import _apply_resume_peak_lr_override

    monkeypatch.delenv("LBT_RESUME_PEAK_LR_OVERRIDE", raising=False)
    trainer = _trainer()
    _apply_resume_peak_lr_override(trainer)
    assert trainer.optimizers.optimizers[0].param_groups[0]["lr"] == pytest.approx(3e-4)
