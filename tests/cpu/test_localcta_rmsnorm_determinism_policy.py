from __future__ import annotations

import os

import pytest
import torch


def _load_policy(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    from low_bits_training.quantization import fp4_converter, tk_gemm

    return fp4_converter, tk_gemm


def test_localcta_highwater_defaults_to_fixed_order_tiled_dgamma(
    monkeypatch,
) -> None:
    fp4_converter, tk_gemm = _load_policy(monkeypatch)
    monkeypatch.setenv("LBT_LOCALCTA_V4_PROFILE", "highwater")
    monkeypatch.delenv("USE_TK_RMSNORM_BWD_SINGLE_OUT", raising=False)

    assert fp4_converter.apply_localcta_v4_profile_defaults() == "highwater"
    assert os.environ["USE_TK_RMSNORM_BWD_SINGLE_OUT"] == "0"
    assert not tk_gemm.use_tk_rmsnorm_bwd_single_out()


def test_localcta_profile_rejects_explicit_atomic_dgamma(monkeypatch) -> None:
    fp4_converter, _ = _load_policy(monkeypatch)
    monkeypatch.setenv("LBT_LOCALCTA_V4_PROFILE", "highwater")
    monkeypatch.setenv("USE_TK_RMSNORM_BWD_SINGLE_OUT", "1")

    with pytest.raises(RuntimeError, match="schedule-dependent atomic dgamma"):
        fp4_converter.apply_localcta_v4_profile_defaults()


def test_localcta_runtime_selector_rejects_converter_bypass(monkeypatch) -> None:
    _, tk_gemm = _load_policy(monkeypatch)
    monkeypatch.setenv("USE_TK_RMSNORM_BWD_SINGLE_OUT", "1")

    with pytest.raises(RuntimeError, match="fixed-order tiled dgamma"):
        tk_gemm.use_tk_rmsnorm_bwd_single_out()


def test_single_output_remains_available_to_non_localcta_backends(
    monkeypatch,
) -> None:
    _, tk_gemm = _load_policy(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    monkeypatch.setenv("USE_TK_RMSNORM_BWD_SINGLE_OUT", "1")

    assert tk_gemm.use_tk_rmsnorm_bwd_single_out()


def test_single_output_rejection_is_scoped_to_localcta_v4(monkeypatch) -> None:
    _, tk_gemm = _load_policy(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v3")
    monkeypatch.setenv("USE_TK_RMSNORM_BWD_SINGLE_OUT", "1")

    assert tk_gemm.use_tk_rmsnorm_bwd_single_out()


@pytest.mark.skipif(
    os.environ.get("LBT_RUN_LOCALCTA_RMS_DETERMINISM_GPU", "0") != "1",
    reason="opt-in production-shape GB200 replay",
)
def test_production_shape_atomic_dgamma_drifts_but_tiled_is_exact() -> None:
    """Causal GPU gate for the native helper rejected by the policy above."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from low_bits_training.quantization.fused_te_linear import _get_te_fused

    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    rows, cols = 32768, 4096
    x = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    weight = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
    inv_rms = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    grad_input = torch.empty_like(x)
    atomic_fp32 = torch.empty((cols,), device="cuda", dtype=torch.float32)
    atomic_bf16 = torch.empty((cols,), device="cuda", dtype=torch.bfloat16)
    partials = torch.empty(
        ((rows + 255) // 256, cols), device="cuda", dtype=torch.float32
    )
    tiled_bf16 = torch.empty((cols,), device="cuda", dtype=torch.bfloat16)
    extension = _get_te_fused()

    def atomic_once() -> torch.Tensor:
        extension.fused_rmsnorm_backward_out(
            dy, x, weight, inv_rms, grad_input, atomic_fp32
        )
        atomic_bf16.copy_(atomic_fp32)
        torch.cuda.synchronize()
        return atomic_bf16.clone()

    def tiled_once() -> torch.Tensor:
        extension.fused_rmsnorm_backward_dx_only_out(
            dy, x, weight, inv_rms, grad_input
        )
        extension.fused_rmsnorm_backward_dgamma_tiled_bf16_out(
            dy, x, inv_rms, partials, tiled_bf16
        )
        torch.cuda.synchronize()
        return tiled_bf16.clone()

    atomic_reference = atomic_once()
    atomic_replays = [atomic_once() for _ in range(7)]
    assert any(
        not torch.equal(replay, atomic_reference) for replay in atomic_replays
    ), "the rejected atomic helper unexpectedly replayed byte-exactly"

    tiled_reference = tiled_once()
    for _ in range(4):
        assert torch.equal(tiled_once(), tiled_reference)
