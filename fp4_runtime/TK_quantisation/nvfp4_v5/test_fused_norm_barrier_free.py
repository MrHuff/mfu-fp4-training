"""Numerical contract for the barrier-free fused RMSNorm producer."""

import os
from pathlib import Path
import sys

import pytest
import torch


if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_v5 as tkq


def test_barrier_free_fused_norm_uses_bf16_amax_contract(monkeypatch):
    monkeypatch.setenv("USE_TK_NORM_QUANT_TWO_PASS", "1")
    torch.manual_seed(123)
    x = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((512,), device="cuda", dtype=torch.bfloat16)

    result = tkq.tk_fused_norm_quantize(x, gamma, 1e-5, False, True)
    reference_inv_rms = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    reference_norm = (
        x.float() * reference_inv_rms[:, None] * gamma.float()
    ).to(torch.bfloat16)
    torch.cuda.synchronize()

    assert torch.allclose(result[5], reference_inv_rms, atol=2e-6, rtol=2e-6)
    assert result[6].item() == reference_norm.abs().max().item()
    assert result[4].item() == pytest.approx(result[6].item() / 2688.0)
    assert result[0].view(torch.uint8).numel() == x.numel() // 2
    assert result[2].view(torch.uint8).numel() == x.numel() // 2
    assert all(
        torch.isfinite(result[index].float()).all()
        for index in (1, 3, 4, 5, 6)
    )


def test_barrier_free_fused_norm_is_default(monkeypatch):
    monkeypatch.delenv("USE_TK_NORM_QUANT_TWO_PASS", raising=False)
    torch.manual_seed(456)
    x = torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones((512,), device="cuda", dtype=torch.bfloat16)

    result = tkq.tk_fused_norm_quantize(x, gamma, 1e-5, False, True)
    reference_inv_rms = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    reference_norm = (x.float() * reference_inv_rms[:, None]).to(torch.bfloat16)
    torch.cuda.synchronize()

    assert result[6].item() == reference_norm.abs().max().item()
