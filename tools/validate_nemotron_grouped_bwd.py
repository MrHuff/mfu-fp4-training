"""Validate the production grouped Mamba2 backward against the trusted CUDA path."""

from __future__ import annotations

import gc
import os
import statistics
import time

import torch

from low_bits_training.models.nemotron_h_hf import mamba_cuda


def synchronize_time(function, iterations: int) -> list[float]:
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1_000.0)
        del result
    return samples


def main() -> None:
    torch.manual_seed(20260723)
    torch.cuda.set_device(0)
    batch, length, heads, head_dim = 1, 8192, 128, 64
    groups, state = 8, 128
    device = torch.device("cuda")

    x = (torch.randn(
        batch, length, heads, head_dim, device=device, dtype=torch.bfloat16
    ) * 0.2).contiguous()
    dt = (torch.randn(
        batch, length, heads, device=device, dtype=torch.bfloat16
    ) * 0.5 - 2.0).contiguous()
    A = (-torch.exp(torch.linspace(
        -1.0, 0.5, heads, device=device, dtype=torch.float32
    ))).contiguous()
    B = (torch.randn(
        batch, length, groups, state, device=device, dtype=torch.bfloat16
    ) * 0.2).contiguous()
    C = (torch.randn_like(B) * 0.2).contiguous()
    D = torch.randn(heads, device=device, dtype=torch.bfloat16).contiguous()
    bias = (torch.randn(
        heads, device=device, dtype=torch.bfloat16
    ) * 0.25 - 1.0).contiguous()
    dout = (torch.randn_like(x) * 0.2).contiguous()

    mamba_cuda._load_native_extensions()
    extension = mamba_cuda._gated_rmsnorm_cuda
    trusted_output, checkpoints = extension.mamba2_ssd_fwd(
        x, dt, A, B, C, D, bias
    )
    grouped_output = extension.mamba2_ssd_fwd_grouped(
        x, dt, A, B, C, D, bias
    )
    output_cosine = torch.nn.functional.cosine_similarity(
        grouped_output.float().flatten(),
        trusted_output.float().flatten(),
        dim=0,
    ).item()
    output_max = (
        grouped_output.float() - trusted_output.float()
    ).abs().max().item()
    print(f"forward: cos={output_cosine:.9f} max={output_max:.6g}")
    assert output_cosine > 0.999999

    conv_storage = torch.empty(
        batch, length, heads * head_dim + 2 * groups * state,
        device=device,
        dtype=torch.bfloat16,
    )
    strided_x, strided_B, strided_C = conv_storage.split(
        [heads * head_dim, groups * state, groups * state], dim=-1
    )
    strided_x = strided_x.view_as(x)
    strided_B = strided_B.view_as(B)
    strided_C = strided_C.view_as(C)
    strided_x.copy_(x)
    strided_B.copy_(B)
    strided_C.copy_(C)
    projection_storage = torch.empty(
        batch, length, 18688, device=device, dtype=torch.bfloat16
    )
    strided_dt = projection_storage[..., 18560:18688]
    strided_dt.copy_(dt)
    strided_output = extension.mamba2_ssd_fwd_grouped(
        strided_x, strided_dt, A, strided_B, strided_C, D, bias
    )
    torch.cuda.synchronize()
    print(
        "strided forward:",
        f"equal={torch.equal(strided_output, grouped_output)}",
        f"x_stride={strided_x.stride(1)}",
        f"dt_stride={strided_dt.stride(1)}",
    )
    assert torch.equal(strided_output, grouped_output)

    def trusted_forward():
        return extension.mamba2_ssd_fwd(x, dt, A, B, C, D, bias)

    def grouped_forward():
        return extension.mamba2_ssd_fwd_grouped(x, dt, A, B, C, D, bias)

    def trusted_backward():
        u, delta, scan_A, scan_B, scan_C, scan_D, scan_bias = (
            mamba_cuda._expanded_selective_scan_inputs(x, dt, A, B, C, D, bias)
        )
        gradients = extension.mamba2_scan_bwd_256x8(
            u,
            delta,
            scan_A,
            scan_B,
            scan_C,
            scan_D,
            scan_bias,
            dout.permute(0, 2, 3, 1).reshape_as(u).contiguous(),
            checkpoints,
        )
        return extension.mamba2_scan_bwd_collapse(
            *gradients, D, bias
        )

    def grouped_backward():
        return extension.mamba2_grouped_bwd(x, dt, A, B, C, D, bias, dout)

    trusted = trusted_backward()
    grouped = grouped_backward()
    adjacent_grouped = extension.mamba2_grouped_bwd_adjacent(
        x, dt, A, B, C, D, bias, dout
    )
    strided_grouped = extension.mamba2_grouped_bwd(
        strided_x, strided_dt, A, strided_B, strided_C, D, bias, dout
    )
    torch.cuda.synchronize()
    names = ("x", "dt", "A", "B", "C", "D", "bias")
    for name, actual, expected in zip(names, grouped, trusted):
        actual_float = actual.float()
        expected_float = expected.float()
        difference = actual_float - expected_float
        cosine = torch.nn.functional.cosine_similarity(
            actual_float.flatten(), expected_float.flatten(), dim=0
        ).item()
        relative_l2 = (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
        ).item()
        print(
            f"{name:>4}: finite={bool(torch.isfinite(actual_float).all())} "
            f"cos={cosine:.9f} rel_l2={relative_l2:.6g} "
            f"max={difference.abs().max().item():.6g}"
        )
        assert torch.isfinite(actual_float).all()
        assert cosine > 0.999
        del actual_float, expected_float, difference

    for name, actual, expected in zip(names, adjacent_grouped, grouped):
        relative_l2 = (
            torch.linalg.vector_norm(actual.float() - expected.float())
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item()
        print(f"adjacent {name:>4}: rel_l2={relative_l2:.6g}")
        assert relative_l2 < 1e-5
    grad_x, _, _, grad_B, grad_C, _, _ = adjacent_grouped
    assert grad_x.untyped_storage().data_ptr() == grad_B.untyped_storage().data_ptr()
    assert grad_x.untyped_storage().data_ptr() == grad_C.untyped_storage().data_ptr()
    assert grad_x.stride(1) == grad_B.stride(1) == grad_C.stride(1) == 10240
    assert grad_B.storage_offset() == grad_x.storage_offset() + heads * head_dim
    assert grad_C.storage_offset() == grad_B.storage_offset() + groups * state

    for name, actual, expected in zip(names, strided_grouped, grouped):
        actual_float = actual.float()
        expected_float = expected.float()
        relative_l2 = (
            torch.linalg.vector_norm(actual_float - expected_float)
            / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
        ).item()
        print(
            f"strided {name:>4}: equal={torch.equal(actual, expected)} "
            f"rel_l2={relative_l2:.6g}"
        )
        assert relative_l2 < 1e-5

    os.environ["LBT_NEMOTRON_H_GROUPED_SSD_BWD"] = "1"
    wrapper_inputs = [
        value.detach().clone().requires_grad_()
        for value in (x, dt, A, B, C, D, bias)
    ]
    wrapper_output = mamba_cuda.selective_scan_mamba2_cutlass(*wrapper_inputs)
    wrapper_output.backward(dout)
    for name, value, expected in zip(names, wrapper_inputs, grouped):
        cosine = torch.nn.functional.cosine_similarity(
            value.grad.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        print(f"autograd {name:>4}: cos={cosine:.9f}")
        assert cosine > 0.999999
    del wrapper_output, wrapper_inputs

    del trusted, grouped, adjacent_grouped, strided_grouped, strided_output
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(2):
        result = grouped_backward()
        del result
    grouped_times = synchronize_time(grouped_backward, 7)
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(2):
        result = trusted_backward()
        del result
    trusted_times = synchronize_time(trusted_backward, 7)
    grouped_forward_times = synchronize_time(grouped_forward, 7)
    trusted_forward_times = synchronize_time(trusted_forward, 7)
    print(
        "grouped ms:",
        ", ".join(f"{value:.3f}" for value in grouped_times),
        f"median={statistics.median(grouped_times):.3f}",
    )
    print(
        "trusted ms:",
        ", ".join(f"{value:.3f}" for value in trusted_times),
        f"median={statistics.median(trusted_times):.3f}",
    )
    print(
        "grouped forward ms:",
        ", ".join(f"{value:.3f}" for value in grouped_forward_times),
        f"median={statistics.median(grouped_forward_times):.3f}",
    )
    print(
        "trusted forward ms:",
        ", ".join(f"{value:.3f}" for value in trusted_forward_times),
        f"median={statistics.median(trusted_forward_times):.3f}",
    )


if __name__ == "__main__":
    main()
