#!/usr/bin/env python3
"""Byte-exact two-stream stress for the persistent split2 localCTA producer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any


_TEST_DIR = Path(__file__).resolve().parent
_HELPER_DIR = Path(os.environ.get("LOCALCTA_EXTENSION_DIR", _TEST_DIR)).resolve()
sys.path.insert(0, str(_HELPER_DIR))
from test_active_output_reuse_stress import (  # noqa: E402
    Case,
    _patterned,
    _serialized_reference,
    _standard_triplets,
    _stress_multistream,
    tkq,
    torch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--h", type=int, default=5632)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.m <= 0 or args.h <= 0 or args.m % 128 or args.h % 128:
        parser.error("--m and --h must be positive multiples of 128")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("localCTA v4 requires a Blackwell GPU")

    dh = _patterned((args.m, args.h), 1.31)
    h3 = _patterned((args.m, args.h), 1.61)
    h1 = _patterned((args.m, args.h), 1.97)

    def make_state() -> dict[str, Any]:
        return {
            "buffers": tuple(
                tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
                    args.m, args.h, dh.device
                )
            ),
            "dh1": torch.empty_like(dh),
            "dh3": torch.empty_like(dh),
        }

    def execute(state: dict[str, Any] | None) -> dict[str, Any]:
        if state is None:
            raise AssertionError("split2 in-place ABI requires state")
        buffers = state["buffers"]
        returned = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
            dh,
            h3,
            h1,
            state["dh1"],
            state["dh3"],
            *buffers,
            True,
            False,
            False,
            0,
            0,
            "none",
        )
        if len(returned) != 12:
            raise AssertionError(f"split2 ABI returned {len(returned)} tensors")
        return {
            "_lifetime": buffers[12:16],
            "quant": tuple(returned),
            "dh1": state["dh1"],
            "dh3": state["dh3"],
        }

    def triplets(output: dict[str, Any]):
        buffers = output["quant"]
        return _standard_triplets("dh1", buffers[0:6]) + _standard_triplets(
            "dh3", buffers[6:12]
        )

    case = Case(
        name=f"split2_raw_inplace_{args.m}x{args.h}",
        execute=execute,
        make_state=make_state,
        reconstruction_triplets=triplets,
    )
    references = _serialized_reference(case)
    _stress_multistream(case, references, args.iterations)
    print(
        "persistent split2 multistream stress: PASS "
        f"shape={args.m}x{args.h} paired_launches={args.iterations}"
    )


if __name__ == "__main__":
    main()
