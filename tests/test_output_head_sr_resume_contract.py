from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_gpu_proof_contract(monkeypatch):
    fake_package = types.ModuleType("fp4_cce_TK")
    fake_package.v4_common = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "fp4_cce_TK", fake_package)

    path = Path(__file__).with_name("run_output_head_sr_resume_gpu.py")
    spec = importlib.util.spec_from_file_location(
        "_output_head_sr_resume_gpu_contract",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(loss: torch.Tensor, payload: int = 7) -> tuple[torch.Tensor, ...]:
    return (
        loss,
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([2.0], dtype=torch.float32),
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor([payload], dtype=torch.uint8),
        torch.tensor([3], dtype=torch.uint8),
        torch.empty(0, dtype=torch.float32),
        torch.tensor([4.0], dtype=torch.float32),
    )


def test_resume_contract_allows_atomic_loss_jitter_but_not_payload_drift(
    monkeypatch,
):
    proof = _load_gpu_proof_contract(monkeypatch)
    original_loss = torch.tensor(4.0, dtype=torch.float32)
    resumed_loss = original_loss.clone()
    toward = torch.tensor(float("inf"), dtype=torch.float32)
    for _ in range(4):
        resumed_loss = torch.nextafter(resumed_loss, toward)

    uninterrupted = _result(original_loss)
    resumed = _result(resumed_loss)
    assert not torch.equal(
        uninterrupted[0].reshape(-1).view(torch.uint8),
        resumed[0].reshape(-1).view(torch.uint8),
    )
    proof._assert_resume_equivalent(uninterrupted, resumed)

    payload_drift = _result(resumed_loss, payload=8)
    with pytest.raises(RuntimeError, match=r"exact fields.*\[5\]"):
        proof._assert_resume_equivalent(uninterrupted, payload_drift)


def test_resume_contract_rejects_material_scalar_loss_drift(monkeypatch):
    proof = _load_gpu_proof_contract(monkeypatch)
    uninterrupted = _result(torch.tensor(4.0, dtype=torch.float32))
    drifted = _result(torch.tensor(4.01, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="floating-reduction tolerance"):
        proof._assert_resume_equivalent(uninterrupted, drifted)
