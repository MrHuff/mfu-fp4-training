import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not os.environ.get("LBT_NEMOTRON_H_CUDA_SITE_PACKAGES")
    or not os.environ.get("FP4_MATMUL_ROOT"),
    reason="requires built Nemotron CUDA extensions and their explicit roots",
)


def test_gated_group_rmsnorm_forward_and_backward():
    from low_bits_training.models.nemotron_h_hf.mamba_cuda import gated_rmsnorm_cuda

    torch.manual_seed(17)
    batch, length, hidden = 3, 2, 8192
    x_storage = torch.randn(
        batch, length, hidden + 128, device="cuda", dtype=torch.bfloat16
    )
    gate_storage = torch.randn_like(x_storage)
    x = x_storage[..., :hidden].detach().requires_grad_()
    gate = gate_storage[..., :hidden].detach().requires_grad_()
    assert not x.is_contiguous()
    assert not gate.is_contiguous()
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = gated_rmsnorm_cuda(x, gate, weight, 1e-5, 1024)

    x_ref = x.float().detach().requires_grad_()
    gate_ref = gate.float().detach().requires_grad_()
    weight_ref = weight.float().detach().requires_grad_()
    u = x_ref * torch.nn.functional.silu(gate_ref)
    grouped = u.reshape(batch, length, hidden // 1024, 1024)
    ref = (
        grouped
        * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-5)
    ).reshape_as(u) * weight_ref
    torch.testing.assert_close(out.float(), ref, rtol=0.015, atol=0.04)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    for actual, expected in (
        (x.grad.float(), x_ref.grad),
        (gate.grad.float(), gate_ref.grad),
        (weight.grad.float(), weight_ref.grad),
    ):
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        )
        assert cosine.item() > 0.9999


def test_gated_group_rmsnorm_materializes_unsupported_inner_stride():
    from low_bits_training.models.nemotron_h_hf.mamba_cuda import gated_rmsnorm_cuda

    torch.manual_seed(23)
    rows, hidden = 2, 8192
    x = torch.randn(
        hidden, rows, device="cuda", dtype=torch.bfloat16
    ).transpose(0, 1)
    gate = torch.randn_like(x)
    assert x.stride(-1) != 1
    assert gate.stride(-1) != 1
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    out = gated_rmsnorm_cuda(x, gate, weight, 1e-5, 1024)
    u = x.float() * torch.nn.functional.silu(gate.float())
    grouped = u.reshape(rows, hidden // 1024, 1024)
    ref = (
        grouped
        * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-5)
    ).reshape_as(u) * weight.float()
    torch.testing.assert_close(out.float(), ref, rtol=0.015, atol=0.04)


def test_grouped_selective_scan_forward_and_backward():
    from low_bits_training.models.nemotron_h_hf.mamba_cuda import selective_scan_mamba2

    torch.manual_seed(31)
    batch, length, heads, head_dim, state_size, groups = 1, 16, 4, 8, 16, 2
    x = torch.randn(
        batch, length, heads, head_dim,
        device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    dt = torch.randn(batch, length, heads, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    A = (-torch.exp(torch.randn(heads, device="cuda"))).requires_grad_()
    B = torch.randn(batch, length, groups, state_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    C = torch.randn_like(B, requires_grad=True)
    D = torch.randn(heads, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn_like(D, requires_grad=True)
    out = selective_scan_mamba2(x, dt, A, B, C, D, bias)

    recurrent = torch.zeros(batch, heads, head_dim, state_size, device="cuda")
    reference = []
    for index in range(length):
        step = torch.nn.functional.softplus(dt[:, index].float() + bias.float())
        step_B = B[:, index].float().repeat_interleave(heads // groups, dim=1)
        step_C = C[:, index].float().repeat_interleave(heads // groups, dim=1)
        recurrent = (
            torch.exp(step[:, :, None, None] * A[None, :, None, None]) * recurrent
            + step[:, :, None, None]
            * step_B[:, :, None, :]
            * x[:, index].float()[:, :, :, None]
        )
        reference.append(
            (recurrent * step_C[:, :, None, :]).sum(-1)
            + D.float()[None, :, None] * x[:, index].float()
        )
    ref = torch.stack(reference, dim=1)
    torch.testing.assert_close(out.float(), ref, rtol=0.03, atol=0.08)

    grad = torch.randn_like(out)
    out.backward(grad, retain_graph=True)
    actual_grads = [value.grad.float().clone() for value in (x, dt, A, B, C, D, bias)]
    for value in (x, dt, A, B, C, D, bias):
        value.grad = None
    ref.backward(grad.float())
    for actual, value in zip(actual_grads, (x, dt, A, B, C, D, bias)):
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), value.grad.float().flatten(), dim=0
        )
        assert cosine.item() > 0.999
