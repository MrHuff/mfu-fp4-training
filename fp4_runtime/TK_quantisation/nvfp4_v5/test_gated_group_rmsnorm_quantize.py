#!/usr/bin/env python3
"""Parity and timing for the native Nemotron gated-RMS -> v5 producer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch


HIDDEN = 8192
GATE_ROW_STRIDE = 18688
DEFAULT_SHAPES = (8192, 16384, 24576, 32768)


def _extension_path(directory: Path, stem: str) -> Path:
    matches = sorted(directory.glob(f"{stem}*.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {stem} extension in {directory}, found {matches}"
        )
    return matches[0].resolve()


def _load_extension(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bytewise_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))


def _assert_parity(
    candidate: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
    reference_inv_rms: torch.Tensor,
    rows: int,
) -> None:
    failures = []
    names = (
        "row_fp4",
        "row_scales",
        "col_fp4",
        "col_scales",
        "row_global_scale",
        "col_global_scale",
    )
    for index, name in enumerate(names):
        if not _bytewise_equal(candidate[index], reference[index]):
            mismatches = torch.count_nonzero(
                candidate[index].view(torch.uint8)
                != reference[index].view(torch.uint8)
            ).item()
            failures.append(f"{name}: {mismatches} differing bytes")

    if not _bytewise_equal(candidate[6], reference[6]):
        failures.append(
            "amax/global-scale keepalive differs: "
            f"candidate={candidate[6].cpu().tolist()}, "
            f"reference={reference[6].cpu().tolist()}"
        )
    expected_inv_rms = reference_inv_rms.view(rows, HIDDEN // 1024)
    if not torch.equal(candidate[8], expected_inv_rms):
        max_error = torch.max(torch.abs(candidate[8] - expected_inv_rms)).item()
        mismatches = torch.count_nonzero(
            candidate[8] != expected_inv_rms
        ).item()
        failures.append(
            f"inv_rms: {mismatches} differences; "
            f"max abs error={max_error}"
        )
    if failures:
        raise AssertionError("; ".join(failures))


def _time_cuda(
    function: Callable[[], tuple[torch.Tensor, ...]],
    warmup: int,
    iterations: int,
) -> tuple[float, float, tuple[torch.Tensor, ...]]:
    output = function()
    for _ in range(warmup - 1):
        output = function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        output = function()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iterations
    return start.elapsed_time(end) / iterations, wall_ms, output


def _run_shape(
    v5: ModuleType,
    mamba: ModuleType,
    rows: int,
    epsilon: float,
    warmup: int,
    iterations: int,
    seed: int,
    encode_centric: bool,
) -> dict[str, float | int | bool]:
    torch.manual_seed(seed + rows)
    scan = torch.randn(
        (rows, HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    gate_storage = torch.randn(
        (rows, GATE_ROW_STRIDE), device="cuda", dtype=torch.bfloat16
    )
    gate = gate_storage[:, :HIDDEN]
    gamma = (
        1.0
        + 0.1
        * torch.randn((HIDDEN,), device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)

    normalized, reference_inv_rms = mamba.gated_rmsnorm_fwd(
        scan, gate, gamma, epsilon
    )
    reference = v5.tk_quantize_for_gemm(
        normalized, True, encode_centric
    )
    candidate = v5.tk_gated_group_rmsnorm_quantize_for_gemm(
        scan, gate, gamma, epsilon, encode_centric
    )
    torch.cuda.synchronize()
    _assert_parity(candidate, reference, reference_inv_rms, rows)

    del normalized, reference_inv_rms, reference, candidate
    torch.cuda.empty_cache()

    def reference_path() -> tuple[torch.Tensor, ...]:
        normalized_output, inv_rms = mamba.gated_rmsnorm_fwd(
            scan, gate, gamma, epsilon
        )
        quantized = v5.tk_quantize_for_gemm(
            normalized_output, True, encode_centric
        )
        return (*quantized, inv_rms)

    def candidate_path() -> tuple[torch.Tensor, ...]:
        return v5.tk_gated_group_rmsnorm_quantize_for_gemm(
            scan, gate, gamma, epsilon, encode_centric
        )

    reference_ms, reference_wall_ms, reference_output = _time_cuda(
        reference_path, warmup, iterations
    )
    candidate_ms, candidate_wall_ms, candidate_output = _time_cuda(
        candidate_path, warmup, iterations
    )
    del reference_output, candidate_output

    result: dict[str, float | int | bool] = {
        "rows": rows,
        "encode_centric": encode_centric,
        "bitwise_parity": True,
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "speedup": reference_ms / candidate_ms,
        "reference_allocation_inclusive_ms": reference_wall_ms,
        "candidate_allocation_inclusive_ms": candidate_wall_ms,
        "allocation_inclusive_speedup": (
            reference_wall_ms / candidate_wall_ms
        ),
        "reference_mrows_per_second": rows / reference_ms / 1000.0,
        "candidate_mrows_per_second": rows / candidate_ms / 1000.0,
    }
    del scan, gate, gate_storage, gamma
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--v5-extension",
        type=Path,
        default=_extension_path(script_dir, "_tk_quant_v5"),
    )
    parser.add_argument(
        "--mamba-extension",
        type=Path,
        default=None,
        help="Path to _nemotron_mamba_cuda*.so",
    )
    parser.add_argument(
        "--shapes",
        default=",".join(str(shape) for shape in DEFAULT_SHAPES),
    )
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--encode-centric",
        choices=("both", "true", "false"),
        default="both",
        help="v5 block-scale mode to validate (default: both)",
    )
    args = parser.parse_args()

    if args.mamba_extension is None:
        args.mamba_extension = _extension_path(
            script_dir.parent / "mamba_cuda", "_nemotron_mamba_cuda"
        )
    shapes = tuple(int(shape) for shape in args.shapes.split(","))
    invalid_shapes = set(shapes) - set(DEFAULT_SHAPES)
    if invalid_shapes:
        raise ValueError(f"unsupported production shapes: {invalid_shapes}")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must both be positive")
    modes = {
        "both": (False, True),
        "true": (True,),
        "false": (False,),
    }[args.encode_centric]

    v5 = _load_extension("_tk_quant_v5", args.v5_extension.resolve())
    mamba = _load_extension(
        "_nemotron_mamba_cuda", args.mamba_extension.resolve()
    )
    if not hasattr(v5, "tk_gated_group_rmsnorm_quantize_for_gemm"):
        raise RuntimeError("v5 extension does not expose the native producer")

    results = []
    for encode_centric in modes:
        for rows in shapes:
            result = _run_shape(
                v5,
                mamba,
                rows,
                args.epsilon,
                args.warmup,
                args.iterations,
                args.seed,
                encode_centric,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "device": torch.cuda.get_device_name(),
        "epsilon": args.epsilon,
        "iterations": args.iterations,
        "modes": list(modes),
        "warmup": args.warmup,
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
