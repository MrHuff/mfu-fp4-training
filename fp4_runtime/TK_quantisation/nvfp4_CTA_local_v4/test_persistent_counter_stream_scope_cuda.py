#!/usr/bin/env python3
"""CUDA contract test for persistent localCTA work-counter ownership.

The extension keeps each work counter alive in a C++ static cache.  This test
warms the default stream, preallocates every ABI input/output, and then measures
the live PyTorch allocation delta after first and repeated launches on two side
streams.  A stream-scoped implementation retains one new one-element int32
tensor per new stream; the pre-patch device-global implementation retains none
after the default-stream warmup.  The expected live-allocation increment is
measured rather than assumed because PyTorch reports allocator-rounded bytes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any


_FORCED_ENV = {
    "USE_TK_LOCALCTA_V3_CONTRACT": "outer",
    "USE_TK_LOCALCTA_V4_DIRECT_STRICT_SPLIT2": "0",
    "USE_TK_LOCALCTA_V4_TUNED_STRICT_SPLIT2": "0",
    "USE_TK_LOCALCTA_V4_SPLIT2_PRECOMPUTE_AMAX": "1",
    "USE_TK_LOCALCTA_V4_SPLIT2_PREFINALIZE_OUTER_SG": "1",
}
os.environ.update(_FORCED_ENV)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing _tk_quant_localcta_v4",
    )
    parser.add_argument(
        "--expect",
        choices=("stream-scoped", "device-global"),
        required=True,
    )
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--h", type=int, default=2048)
    args = parser.parse_args()
    if args.m <= 0 or args.h <= 0 or args.m % 128 or args.h % 128:
        parser.error("--m and --h must be positive multiples of 128")
    return args


def main() -> None:
    args = _parse_args()
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _tk_quant_localcta_v4 as tkq

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("localCTA v4 requires a Blackwell GPU")

    device = torch.device("cuda")
    torch.manual_seed(20260821)

    allocation_baseline = torch.cuda.memory_allocated()
    allocation_probe = torch.empty((1,), device=device, dtype=torch.int32)
    counter_allocation_bytes = torch.cuda.memory_allocated() - allocation_baseline
    del allocation_probe
    if torch.cuda.memory_allocated() != allocation_baseline:
        raise AssertionError("temporary allocation probe did not release cleanly")

    dh = torch.randn((args.m, args.h), device=device, dtype=torch.bfloat16)
    h3 = torch.randn_like(dh)
    h1 = torch.randn_like(dh)

    def make_state() -> dict[str, Any]:
        return {
            "dh1": torch.empty_like(dh),
            "dh3": torch.empty_like(dh),
            "buffers": tuple(
                tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
                    args.m, args.h, device
                )
            ),
        }

    states = [make_state() for _ in range(3)]
    streams = (torch.cuda.Stream(), torch.cuda.Stream())

    def launch(state: dict[str, Any]) -> None:
        returned = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
            dh,
            h3,
            h1,
            state["dh1"],
            state["dh3"],
            *state["buffers"],
            False,
            False,
            False,
            0,
            0,
            "none",
        )
        if len(returned) != 12:
            raise AssertionError(f"split2 ABI returned {len(returned)} tensors")

    # Warm every route-level cache on the default stream.  All later input and
    # output tensors already exist, so only a new persistent counter may remain
    # allocated after a side-stream launch.
    launch(states[0])
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    with torch.cuda.stream(streams[0]):
        launch(states[1])
    streams[0].synchronize()
    after_stream_0 = torch.cuda.memory_allocated()

    with torch.cuda.stream(streams[0]):
        launch(states[1])
    streams[0].synchronize()
    after_stream_0_repeat = torch.cuda.memory_allocated()

    with torch.cuda.stream(streams[1]):
        launch(states[2])
    streams[1].synchronize()
    after_stream_1 = torch.cuda.memory_allocated()

    deltas = (
        after_stream_0 - baseline,
        after_stream_0_repeat - after_stream_0,
        after_stream_1 - after_stream_0_repeat,
    )
    expected = (
        (counter_allocation_bytes, 0, counter_allocation_bytes)
        if args.expect == "stream-scoped"
        else (0, 0, 0)
    )
    if deltas != expected:
        raise AssertionError(
            f"counter allocation deltas {deltas} do not match {args.expect} "
            f"contract {expected}"
        )

    print(
        "persistent-counter CUDA ownership: PASS "
        f"expect={args.expect} shape={args.m}x{args.h} "
        f"counter_allocation_bytes={counter_allocation_bytes} "
        f"allocation_deltas={deltas}"
    )


if __name__ == "__main__":
    main()
