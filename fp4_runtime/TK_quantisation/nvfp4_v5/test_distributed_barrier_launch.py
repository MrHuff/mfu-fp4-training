#!/usr/bin/env python3
"""Exercise v5 software-barrier kernels while NCCL collectives are active."""

import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_v5 as tkq


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    x = torch.randn(
        (32768, 4096), device="cuda", dtype=torch.bfloat16
    )
    collective_buffers = [
        torch.ones(
            (16 * 1024 * 1024,), device="cuda", dtype=torch.bfloat16
        )
        for _ in range(4)
    ]

    tkq.tk_quantize_for_gemm_opt(
        x, True, True, False, False, "none", False, 123, 456
    )
    split3_inputs = (
        torch.randn((32768, 2048), device="cuda", dtype=torch.bfloat16),
        torch.randn((32768, 512), device="cuda", dtype=torch.bfloat16),
        torch.randn((32768, 512), device="cuda", dtype=torch.bfloat16),
    )
    gamma = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
    tkq.tk_fused_norm_quantize(x, gamma, 1e-5, False, True)
    tkq.tk_group_quantize_dim1_split3_for_gemm(
        *split3_inputs, True, 123, 456
    )
    dist.all_reduce(collective_buffers[0])
    torch.cuda.synchronize()
    dist.barrier()

    # The v5 persistent kernels use device-wide software barriers. Exercise the
    # production QKV split3 and generic FFN quantizers on independent streams:
    # without host-side sequence guards these launches can overlap and hang.
    norm_stream = torch.cuda.Stream()
    split3_stream = torch.cuda.Stream()
    generic_stream = torch.cuda.Stream()
    concurrent_norm_result = None
    concurrent_generic_result = None
    concurrent_split3_result = None
    start = time.perf_counter()
    for index in range(24):
        with torch.cuda.stream(norm_stream):
            concurrent_norm_result = tkq.tk_fused_norm_quantize(
                x, gamma, 1e-5, False, True
            )
        with torch.cuda.stream(split3_stream):
            concurrent_split3_result = (
                tkq.tk_group_quantize_dim1_split3_for_gemm(
                    *split3_inputs, True, 123, 456
                )
            )
        with torch.cuda.stream(generic_stream):
            concurrent_generic_result = tkq.tk_quantize_for_gemm_opt(
                x, True, True, False, False, "none", False, 123, 456
            )
        if (index + 1) % 8 == 0:
            norm_stream.synchronize()
            split3_stream.synchronize()
            generic_stream.synchronize()
    concurrent_elapsed = time.perf_counter() - start

    assert concurrent_norm_result is not None
    assert concurrent_generic_result is not None
    assert concurrent_split3_result is not None
    assert torch.isfinite(concurrent_norm_result[1].float()).all()
    assert torch.isfinite(concurrent_norm_result[4].float()).all()
    assert torch.isfinite(concurrent_generic_result[1].float()).all()
    assert torch.isfinite(concurrent_generic_result[4].float()).all()
    assert torch.isfinite(concurrent_split3_result[2]).all()
    dist.barrier()

    start = time.perf_counter()
    works = []
    result = None
    norm_result = None
    for index in range(32):
        works.append(
            dist.all_reduce(
                collective_buffers[index % len(collective_buffers)],
                async_op=True,
            )
        )
        norm_result = tkq.tk_fused_norm_quantize(
            x, gamma, 1e-5, False, True
        )
        result = tkq.tk_quantize_for_gemm_opt(
            x, True, True, False, False, "none", False, 123, 456
        )

    for work in works:
        work.wait()
    torch.cuda.synchronize()
    generic_elapsed = time.perf_counter() - start

    assert norm_result is not None
    assert result is not None
    assert torch.isfinite(norm_result[1].float()).all()
    assert torch.isfinite(norm_result[4].float()).all()
    assert torch.isfinite(result[1].float()).all()
    assert torch.isfinite(result[4].float()).all()

    dist.barrier()
    start = time.perf_counter()
    works = []
    split3_result = None
    for index in range(256):
        works.append(
            dist.all_reduce(
                collective_buffers[index % len(collective_buffers)],
                async_op=True,
            )
        )
        split3_result = tkq.tk_group_quantize_dim1_split3_for_gemm(
            *split3_inputs, True, 123, 456
        )

    for work in works:
        work.wait()
    torch.cuda.synchronize()
    split3_elapsed = time.perf_counter() - start

    assert split3_result is not None
    assert len(split3_result) == 15
    assert torch.isfinite(split3_result[2]).all()
    success = torch.ones((), device="cuda", dtype=torch.int32)
    dist.all_reduce(success)
    assert success.item() == dist.get_world_size()

    if dist.get_rank() == 0:
        print(
            "distributed NCCL/barrier overlap: PASS "
            f"world_size={dist.get_world_size()} "
            f"concurrent_elapsed={concurrent_elapsed:.3f}s "
            f"generic_elapsed={generic_elapsed:.3f}s "
            f"split3_sr_elapsed={split3_elapsed:.3f}s"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
