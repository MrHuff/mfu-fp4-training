#!/usr/bin/env python3
"""Sweep native TK NVFP4 configs on firing Llama/Nemotron production shapes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


EXPECTED_FP4_COMMIT = "0e9ab834519287a6c96cd723109146fd691c85cf"
EXPECTED_TK_COMMIT = "eb04ee2771b7b34f9c1ccadc466058d09ad53378"
DEFAULT_FP4_ROOT = Path("/tmp/fp4_matmul_nemotron_h_8b_pure_tk_20260720")
CONFIG_IDS = tuple(range(47))


@dataclass(frozen=True)
class ProductionShape:
    name: str
    model: str
    m: int
    n: int
    k: int
    owners: str
    v5_route: str
    localcta_route: str
    sweepable: bool = True


# M, N, K describe D[M,N] = A[M,K] @ B[N,K].T at the native ABI.
# Duplicate owners sharing an exact key are intentionally coalesced.
PRODUCTION_SHAPES = (
    ProductionShape(
        "llama_wo_fwd_dgrad",
        "llama8b",
        32768,
        4096,
        4096,
        "attention Wo forward and dgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "llama_wo_wgrad",
        "llama8b",
        4096,
        4096,
        32768,
        "attention Wo wgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "llama_w2_dgrad",
        "llama8b",
        32768,
        14336,
        4096,
        "SwiGLU W2 dgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "llama_ffn_wgrad",
        "llama8b",
        4096,
        14336,
        32768,
        "W2 wgrad and two eager W1/W3 wgrads",
        "single PDL",
        "direct/grouped outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_wo_fwd_dgrad",
        "nemotron8b",
        24576,
        4096,
        4096,
        "attention Wo forward and dgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_wo_wgrad",
        "nemotron8b",
        4096,
        4096,
        24576,
        "attention Wo wgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_sqrelu_w1_fwd_w2_dgrad",
        "nemotron8b",
        24576,
        21504,
        4096,
        "square-ReLU W1 forward and W2 dgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_sqrelu_w2_wgrad",
        "nemotron8b",
        4096,
        21504,
        24576,
        "square-ReLU W2 wgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_sqrelu_w1_dgrad",
        "nemotron8b",
        24576,
        4096,
        21504,
        "square-ReLU W1 dgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_sqrelu_w1_wgrad",
        "nemotron8b",
        21504,
        4096,
        24576,
        "square-ReLU W1 wgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_mamba_in_fwd",
        "nemotron8b",
        24576,
        18688,
        4096,
        "padded Mamba input projection forward",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_mamba_in_dgrad",
        "nemotron8b",
        24576,
        4096,
        18688,
        "padded Mamba input projection dgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_mamba_in_wgrad",
        "nemotron8b",
        18688,
        4096,
        24576,
        "padded Mamba input projection wgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_mamba_out_fwd",
        "nemotron8b",
        24576,
        4096,
        8192,
        "Mamba output projection forward",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_mamba_out_dgrad",
        "nemotron8b",
        24576,
        8192,
        4096,
        "Mamba output projection dgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "nemotron_mamba_out_wgrad",
        "nemotron8b",
        4096,
        8192,
        24576,
        "Mamba output projection wgrad",
        "single PDL",
        "fast outer-SG, hard-configured",
    ),
    ProductionShape(
        "llama_qkv_wgrad",
        "llama8b",
        4096,
        6144,
        32768,
        "grouped QKV wgrad",
        "grouped PDL, no per-call config API",
        "direct grouped outer-SG, hard-configured",
        False,
    ),
    ProductionShape(
        "nemotron_qkv_wgrad",
        "nemotron8b",
        4096,
        5120,
        24576,
        "grouped QKV wgrad",
        "grouped PDL, no per-call config API",
        "direct grouped outer-SG, hard-configured",
        False,
    ),
)


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _validate_roots(fp4_root: Path) -> None:
    actual_fp4 = _git_head(fp4_root)
    actual_tk = _git_head(fp4_root / "ThunderKittens")
    if actual_fp4 != EXPECTED_FP4_COMMIT or actual_tk != EXPECTED_TK_COMMIT:
        raise SystemExit(
            "Pinned native root mismatch: "
            f"fp4={actual_fp4}, TK={actual_tk}; expected "
            f"{EXPECTED_FP4_COMMIT}, {EXPECTED_TK_COMMIT}"
        )


def _load_extension(path: Path, module_name: str):
    old = sys.modules.pop(module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load extension spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old is not None:
            sys.modules[module_name] = old
        else:
            sys.modules.pop(module_name, None)


def _load_native(fp4_root: Path):
    gemm = _load_extension(
        fp4_root
        / "ThunderKittens/kernels/gemm/nvfp4_b200"
        / "_C.cpython-312-aarch64-linux-gnu.so",
        "_C",
    )
    quant = _load_extension(
        fp4_root
        / "TK_quantisation/nvfp4_v5"
        / "_tk_quant_v5.cpython-312-aarch64-linux-gnu.so",
        "_tk_quant_v5",
    )
    required = (
        "nvfp4_gemm",
        "nvfp4_gemm_config",
        "nvfp4_gemm_nopdl",
        "nvfp4_gemm_config_nopdl",
    )
    missing = [name for name in required if not hasattr(gemm, name)]
    if missing:
        raise RuntimeError(f"native GEMM extension lacks {missing}")
    return gemm, quant


def _quantize_row(quant, rows: int, cols: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    source = torch.empty((rows, cols), dtype=torch.bfloat16, device="cuda").normal_(
        generator=generator
    )
    result = quant.tk_quantize_for_gemm(source, True)
    payload = (result[0], result[1], result[4])
    del source, result
    return payload


def _error_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, object]:
    exact = bool(torch.equal(reference, candidate))
    finite = bool(torch.isfinite(candidate).all().item())
    if exact:
        return {
            "bitwise": True,
            "finite": finite,
            "max_abs": 0.0,
            "rel_l2": 0.0,
        }

    diff_sq = 0.0
    ref_sq = 0.0
    max_abs = 0.0
    for start in range(0, reference.shape[0], 512):
        ref = reference[start : start + 512].float()
        got = candidate[start : start + 512].float()
        diff = got - ref
        diff_sq += float(torch.sum(diff * diff).item())
        ref_sq += float(torch.sum(ref * ref).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(diff)).item()))
    return {
        "bitwise": False,
        "finite": finite,
        "max_abs": max_abs,
        "rel_l2": (diff_sq / max(ref_sq, 1.0e-30)) ** 0.5,
    }


def _event_ms(fn: Callable[[], None], iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iters


def _paired_timing(
    baseline: Callable[[], None],
    candidate: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    rounds: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    baseline_ms: list[float] = []
    candidate_ms: list[float] = []
    for round_idx in range(rounds):
        if round_idx % 2:
            candidate_ms.append(_event_ms(candidate, iters))
            baseline_ms.append(_event_ms(baseline, iters))
        else:
            baseline_ms.append(_event_ms(baseline, iters))
            candidate_ms.append(_event_ms(candidate, iters))
    return baseline_ms, candidate_ms


def _select_shapes(names: list[str], models: list[str]) -> list[ProductionShape]:
    wanted_names = set(names)
    wanted_models = set(models)
    selected = [
        shape
        for shape in PRODUCTION_SHAPES
        if (not wanted_names or shape.name in wanted_names)
        and (not wanted_models or shape.model in wanted_models)
    ]
    missing = wanted_names - {shape.name for shape in selected}
    if missing:
        raise SystemExit(f"Unknown or filtered shape names: {sorted(missing)}")
    return selected


def _print_census(shapes: list[ProductionShape]) -> None:
    print("production-shape census:")
    for shape in shapes:
        status = "sweep" if shape.sweepable else "audit-only"
        print(
            f"  {shape.name}: M={shape.m} N={shape.n} K={shape.k}; "
            f"{shape.owners}; v5={shape.v5_route}; "
            f"localCTA={shape.localcta_route}; {status}"
        )


def _parse_csv_ints(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(CONFIG_IDS)
    result = [int(item) for item in value.split(",") if item.strip()]
    invalid = sorted(set(result) - set(CONFIG_IDS))
    if invalid:
        raise SystemExit(f"Invalid config IDs {invalid}; valid IDs are 0-46")
    return result


def _parse_candidates(values: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    known = {shape.name for shape in PRODUCTION_SHAPES}
    for value in values:
        if "=" not in value:
            raise SystemExit("--candidate must use SHAPE=CONFIGS")
        name, raw_configs = value.split("=", 1)
        if name not in known:
            raise SystemExit(f"Unknown candidate shape {name!r}")
        result[name] = _parse_csv_ints(raw_configs)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp4-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument(
        "--model", action="append", choices=("llama8b", "nemotron8b"), default=[]
    )
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--configs", default="all")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Restrict one shape to comma-separated configs: SHAPE=CONFIGS",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--nopdl", action="store_true")
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_roots(args.fp4_root)
    shapes = _select_shapes(args.shape, args.model)
    _print_census(shapes)
    if args.census_only:
        return 0
    configs = _parse_csv_ints(args.configs)
    candidates = _parse_candidates(args.candidate)
    if candidates:
        shapes = [shape for shape in shapes if shape.name in candidates]
    sweep_shapes = [shape for shape in shapes if shape.sweepable]
    if not sweep_shapes:
        print("No selected production route has a configurable native entrypoint.")
        return 0

    torch.cuda.init()
    gemm, quant = _load_native(args.fp4_root)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(
        f"device={props.name}; sm={props.major}.{props.minor}; "
        f"mode={'nopdl' if args.nopdl else 'pdl'}"
    )
    baseline_entry = gemm.nvfp4_gemm_nopdl if args.nopdl else gemm.nvfp4_gemm
    config_entry = (
        gemm.nvfp4_gemm_config_nopdl if args.nopdl else gemm.nvfp4_gemm_config
    )

    rows: list[dict[str, object]] = []
    for shape_idx, shape in enumerate(sweep_shapes):
        print(
            f"shape={shape.name} M={shape.m} N={shape.n} K={shape.k}",
            flush=True,
        )
        a, a_sc, a_sg = _quantize_row(quant, shape.m, shape.k, 1000 + shape_idx * 2)
        b, b_sc, b_sg = _quantize_row(quant, shape.n, shape.k, 1001 + shape_idx * 2)
        reference = torch.empty((shape.m, shape.n), dtype=torch.bfloat16, device="cuda")
        candidate_out = torch.empty_like(reference)
        baseline = lambda: baseline_entry(  # noqa: E731
            a, a_sc, a_sg, b, b_sc, b_sg, reference
        )
        baseline()
        torch.cuda.synchronize()

        shape_configs = candidates.get(shape.name, configs)
        for config_id in shape_configs:
            candidate = lambda config_id=config_id: config_entry(  # noqa: E731
                a, a_sc, a_sg, b, b_sc, b_sg, candidate_out, config_id
            )
            try:
                candidate()
                torch.cuda.synchronize()
                parity = _error_metrics(reference, candidate_out)
                if not parity["finite"] or not parity["bitwise"]:
                    row = {
                        **asdict(shape),
                        "config_id": config_id,
                        "mode": "nopdl" if args.nopdl else "pdl",
                        "status": "parity_reject",
                        **parity,
                    }
                else:
                    baseline_ms, candidate_ms = _paired_timing(
                        baseline,
                        candidate,
                        warmup=args.warmup,
                        iters=args.iters,
                        rounds=args.rounds,
                    )
                    base_median = statistics.median(baseline_ms)
                    candidate_median = statistics.median(candidate_ms)
                    paired_speedups = [
                        base / candidate
                        for base, candidate in zip(baseline_ms, candidate_ms)
                    ]
                    row = {
                        **asdict(shape),
                        "config_id": config_id,
                        "mode": "nopdl" if args.nopdl else "pdl",
                        "status": "ok",
                        **parity,
                        "baseline_ms": base_median,
                        "candidate_ms": candidate_median,
                        "speedup": base_median / candidate_median,
                        "paired_speedup_median": statistics.median(paired_speedups),
                        "baseline_round_ms": baseline_ms,
                        "candidate_round_ms": candidate_ms,
                    }
            except Exception as exc:  # noqa: BLE001
                torch.cuda.synchronize()
                row = {
                    **asdict(shape),
                    "config_id": config_id,
                    "mode": "nopdl" if args.nopdl else "pdl",
                    "status": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"  config={config_id:02d} "
                    f"base={row['baseline_ms']:.6f}ms "
                    f"candidate={row['candidate_ms']:.6f}ms "
                    f"speedup={row['paired_speedup_median']:.5f}x",
                    flush=True,
                )
            else:
                print(
                    f"  config={config_id:02d} status={row['status']}",
                    flush=True,
                )
        del a, a_sc, a_sg, b, b_sc, b_sg, reference, candidate_out
        torch.cuda.empty_cache()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
