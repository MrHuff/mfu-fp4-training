#!/usr/bin/env python3
"""Exercise the production v5 QKV weight quantizer alongside NCCL."""

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

    weight = torch.randn((6144, 4096), device="cuda", dtype=torch.bfloat16)
    splits = [4096, 1024, 1024]
    collective_buffers = [
        torch.ones((16 * 1024 * 1024,), device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]

    result = tkq.tk_group_quantize_for_gemm(weight, splits)
    torch.cuda.synchronize()
    dist.barrier()

    works = []
    start = time.perf_counter()
    for index in range(64):
        works.append(
            dist.all_reduce(
                collective_buffers[index % len(collective_buffers)],
                async_op=True,
            )
        )
        result = tkq.tk_group_quantize_for_gemm(weight, splits)

    for work in works:
        work.wait()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    assert torch.isfinite(result[1].float()).all()
    assert torch.isfinite(result[2]).all()
    assert all(torch.isfinite(scale.float()).all() for scale in result[4])
    success = torch.ones((), device="cuda", dtype=torch.int32)
    dist.all_reduce(success)
    assert success.item() == dist.get_world_size()

    if dist.get_rank() == 0:
        print(
            "grouped QKV weight quantization with NCCL: PASS "
            f"world_size={dist.get_world_size()} elapsed={elapsed:.3f}s"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
