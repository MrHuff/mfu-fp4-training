"""Validate batched production Mamba2 SSD against independent CUDA calls."""

from __future__ import annotations

import torch

from low_bits_training.models.nemotron_h_hf import mamba_cuda


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        difference = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{name} differs: max={difference.max().item():.6g}, "
            f"mean={difference.mean().item():.6g}"
        )


def main() -> None:
    torch.manual_seed(20260723)
    torch.cuda.set_device(0)
    batch, length, heads, head_dim = 2, 8192, 128, 64
    groups, state = 8, 128
    device = torch.device("cuda")

    x = (
        torch.randn(
            batch,
            length,
            heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.2
    ).contiguous()
    dt = (
        torch.randn(
            batch,
            length,
            heads,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.5
        - 2.0
    ).contiguous()
    A = (
        -torch.exp(
            torch.linspace(-1.0, 0.5, heads, device=device, dtype=torch.float32)
        )
    ).contiguous()
    B = (
        torch.randn(
            batch,
            length,
            groups,
            state,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.2
    ).contiguous()
    C = (torch.randn_like(B) * 0.2).contiguous()
    D = torch.randn(heads, device=device, dtype=torch.bfloat16).contiguous()
    bias = (
        torch.randn(heads, device=device, dtype=torch.bfloat16) * 0.25 - 1.0
    ).contiguous()
    dout = (torch.randn_like(x) * 0.2).contiguous()

    mamba_cuda._load_native_extensions()
    extension = mamba_cuda._gated_rmsnorm_cuda
    output = extension.mamba2_ssd_fwd_grouped(x, dt, A, B, C, D, bias)
    for index in range(batch):
        expected = extension.mamba2_ssd_fwd_grouped(
            x[index : index + 1],
            dt[index : index + 1],
            A,
            B[index : index + 1],
            C[index : index + 1],
            D,
            bias,
        )
        _assert_equal(f"forward[{index}]", output[index : index + 1], expected)
        del expected

    gradients = extension.mamba2_grouped_bwd_adjacent(
        x, dt, A, B, C, D, bias, dout
    )
    expected_A = torch.zeros_like(A)
    expected_D = torch.zeros_like(D, dtype=torch.float32)
    expected_bias = torch.zeros_like(bias, dtype=torch.float32)
    for index in range(batch):
        expected = extension.mamba2_grouped_bwd_adjacent(
            x[index : index + 1],
            dt[index : index + 1],
            A,
            B[index : index + 1],
            C[index : index + 1],
            D,
            bias,
            dout[index : index + 1],
        )
        for name, actual_index, expected_index in (
            ("x", 0, 0),
            ("dt", 1, 1),
            ("B", 3, 3),
            ("C", 4, 4),
        ):
            _assert_equal(
                f"{name}[{index}]",
                gradients[actual_index][index : index + 1],
                expected[expected_index],
            )
        expected_A += expected[2]
        expected_D += expected[5].float()
        expected_bias += expected[6].float()
        del expected

    torch.testing.assert_close(gradients[2], expected_A, rtol=0.0, atol=1e-5)
    _assert_equal("D", gradients[5], expected_D.to(torch.bfloat16))
    _assert_equal("dt_bias", gradients[6], expected_bias.to(torch.bfloat16))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    grad_x, _, _, grad_B, grad_C, _, _ = gradients
    assert grad_x.untyped_storage().data_ptr() == grad_B.untyped_storage().data_ptr()
    assert grad_x.untyped_storage().data_ptr() == grad_C.untyped_storage().data_ptr()
    assert grad_x.stride(1) == grad_B.stride(1) == grad_C.stride(1) == 10240
    assert grad_B.storage_offset() == grad_x.storage_offset() + heads * head_dim
    assert grad_C.storage_offset() == grad_B.storage_offset() + groups * state
    torch.cuda.synchronize()
    print("batched production SSD: forward/backward exact for batch=2")


if __name__ == "__main__":
    main()
