#!/usr/bin/env python3
"""Sweep exact production MXFP4 GEMM selectors on native TK CUDA kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_FP4_COMMIT = "0e9ab834519287a6c96cd723109146fd691c85cf"
PINNED_GEMM_EXTENSION_SHA256 = (
    "5308af6a7c559c95c61794bc234a9ca1c28e978d8b6b1f25ceb588fbd0793616"
)
PINNED_QUANT_EXTENSION_SHA256 = (
    "48742f8bf31595eabdeec2fa33acf53f7c011f3ec2b6e38911b024b9f14f506e"
)
DEFAULT_FP4_ROOT = Path("/tmp/fp4_matmul_volt_runtime_sm100_0e9ab834")
GEMM_EXTENSION_RELATIVE_PATH = Path(
    "ThunderKittens/kernels/gemm/mxfp4_gb200/" "_C_mx.cpython-312-aarch64-linux-gnu.so"
)
QUANT_EXTENSION_RELATIVE_PATH = Path(
    "TK_quantisation/mxfp4_v3/" "mxfp4_quant_v3.cpython-312-aarch64-linux-gnu.so"
)


@dataclass(frozen=True)
class SweepCase:
    name: str
    model: str
    kernel: str
    orientation: str
    m: int
    n: int
    k: int
    num_batches: int = 1
    split_k: tuple[int, ...] = ()

    @property
    def flops(self) -> float:
        multiplier = self.num_batches if self.kernel == "batched" else 1
        return float(2 * self.m * self.n * self.k * multiplier)


@dataclass
class PreparedCase:
    reference_outputs: list[torch.Tensor]
    candidate_outputs: list[torch.Tensor]
    launch_native: Callable[[list[torch.Tensor]], None]
    launch_config: Callable[[int, list[torch.Tensor]], None]
    allocations: tuple[object, ...]


PRODUCTION_CASES = (
    SweepCase(
        "nemotron_mamba_out_forward",
        "nemotron_h_8b",
        "dense",
        "forward",
        32768,
        4096,
        8192,
    ),
    SweepCase(
        "nemotron_mamba_out_dgrad",
        "nemotron_h_8b",
        "dense",
        "dgrad",
        32768,
        8192,
        4096,
    ),
    SweepCase(
        "nemotron_mamba_out_wgrad",
        "nemotron_h_8b",
        "dense",
        "wgrad",
        4096,
        8192,
        32768,
    ),
    SweepCase(
        "nemotron_mlp_w13_forward",
        "nemotron_h_8b",
        "dense",
        "forward",
        32768,
        21504,
        4096,
    ),
    SweepCase(
        "nemotron_mlp_w13_forward_batched",
        "nemotron_h_8b",
        "batched",
        "forward",
        32768,
        21504,
        4096,
        num_batches=2,
    ),
    SweepCase(
        "nemotron_mlp_w2_forward_residual",
        "nemotron_h_8b",
        "residual",
        "forward",
        32768,
        4096,
        21504,
    ),
    SweepCase(
        "nemotron_mlp_w2_dgrad",
        "nemotron_h_8b",
        "dense",
        "dgrad",
        32768,
        21504,
        4096,
    ),
    SweepCase(
        "nemotron_mlp_w13_dgrad_onepass",
        "nemotron_h_8b",
        "split2_onepass",
        "dgrad",
        32768,
        4096,
        43008,
        split_k=(21504, 21504),
    ),
    SweepCase(
        "nemotron_mlp_w13_wgrad_batched",
        "nemotron_h_8b",
        "batched",
        "wgrad",
        21504,
        4096,
        32768,
        num_batches=2,
    ),
    SweepCase(
        "nemotron_mlp_w2_wgrad",
        "nemotron_h_8b",
        "dense",
        "wgrad",
        4096,
        21504,
        32768,
    ),
    SweepCase(
        "nemotron_qkv_dgrad",
        "nemotron_h_8b",
        "dense",
        "dgrad",
        32768,
        4096,
        5120,
    ),
    SweepCase(
        "nemotron_qkv_dgrad_onepass",
        "nemotron_h_8b",
        "split3_onepass",
        "dgrad",
        32768,
        4096,
        5120,
        split_k=(4096, 512, 512),
    ),
    SweepCase(
        "llama_w13_forward",
        "llama3_8b",
        "dense",
        "forward",
        32768,
        14336,
        4096,
    ),
    SweepCase(
        "llama_w13_forward_batched",
        "llama3_8b",
        "batched",
        "forward",
        32768,
        14336,
        4096,
        num_batches=2,
    ),
    SweepCase(
        "llama_w13_dgrad_onepass",
        "llama3_8b",
        "split2_onepass",
        "dgrad",
        32768,
        4096,
        28672,
        split_k=(14336, 14336),
    ),
    SweepCase(
        "llama_qkv_dgrad",
        "llama3_8b",
        "dense",
        "dgrad",
        32768,
        4096,
        6144,
    ),
    SweepCase(
        "llama_qkv_dgrad_onepass",
        "llama3_8b",
        "split3_onepass",
        "dgrad",
        32768,
        4096,
        6144,
        split_k=(4096, 1024, 1024),
    ),
)

CONFIGS_BY_KERNEL = {
    "dense": tuple(range(11)),
    "residual": tuple(range(11)),
    "batched": tuple(range(10)),
    "split2_onepass": (1, 3, 5),
    "split3_onepass": (1, 3, 5),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", default=[], help="Case name; repeatable."
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--configs", help="Comma-separated override for every selected case."
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--stabilize-ms", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--min-speedup", type=float, default=1.005)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--parity-chunk-rows", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--fp4-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument("--gemm-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument("--expected-fp4-commit", default=PINNED_FP4_COMMIT)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--no-isolate-configs", action="store_true")
    parser.add_argument("--worker-timeout", type=float, default=180.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _setup_backend(args: argparse.Namespace):
    os.environ.setdefault("LBT_LIGHT_IMPORT", "1")
    os.environ.setdefault("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    os.environ["FP4_MXFP4_ROOT"] = str(args.fp4_root.resolve())
    os.environ["FP4_MATMUL_GEMM_ROOT"] = str(args.gemm_root.resolve())
    sys.path.insert(0, str(REPO_ROOT))

    from low_bits_training.quantization import mxfp4_backend

    return mxfp4_backend


def _root_provenance(root: Path, expected_commit: str) -> dict[str, object]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"PROVENANCE_ERROR: FP4 root does not exist: {resolved}")

    git_commit = _command_output(["git", "rev-parse", "HEAD"], resolved)
    if git_commit != "unavailable":
        status = _command_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            resolved,
        )
        if git_commit != expected_commit:
            raise RuntimeError(
                "PROVENANCE_ERROR: "
                f"FP4 root {resolved} is at {git_commit}, expected {expected_commit}"
            )
        if status:
            raise RuntimeError(
                "PROVENANCE_ERROR: "
                f"FP4 root {resolved} is dirty and cannot be benchmarked:\n{status}"
            )
        return {
            "path": str(resolved),
            "kind": "clean_git_checkout",
            "commit": git_commit,
            "clean": True,
        }

    marker = resolved / ".lbt_fp4_matmul_commit"
    marker_commit = marker.read_text().strip() if marker.is_file() else ""
    if marker_commit != expected_commit:
        raise RuntimeError(
            "PROVENANCE_ERROR: "
            f"Non-Git FP4 root {resolved} lacks an exact {expected_commit} "
            "provenance marker"
        )
    return {
        "path": str(resolved),
        "kind": "immutable_runtime",
        "commit": marker_commit,
        "commit_marker": str(marker),
        "clean": True,
    }


def _validate_roots(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    expected = args.expected_fp4_commit.strip()
    if len(expected) != 40:
        raise ValueError("--expected-fp4-commit must be a full 40-character SHA")
    return {
        "quant": _root_provenance(args.fp4_root, expected),
        "gemm": _root_provenance(args.gemm_root, expected),
    }


def _validate_pinned_artifacts(args: argparse.Namespace) -> None:
    artifacts = (
        (
            "GEMM",
            args.gemm_root.resolve() / GEMM_EXTENSION_RELATIVE_PATH,
            PINNED_GEMM_EXTENSION_SHA256,
        ),
        (
            "quant",
            args.fp4_root.resolve() / QUANT_EXTENSION_RELATIVE_PATH,
            PINNED_QUANT_EXTENSION_SHA256,
        ),
    )
    for label, path, expected_sha256 in artifacts:
        if not path.is_file():
            raise RuntimeError(
                f"PROVENANCE_ERROR: pinned {label} extension is missing: {path}"
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"PROVENANCE_ERROR: pinned {label} extension {path} has SHA256 "
                f"{actual_sha256}, expected {expected_sha256}"
            )


def _ensure_native_only() -> None:
    loaded = sorted(
        name for name in sys.modules if name == "triton" or name.startswith("triton.")
    )
    if loaded:
        raise RuntimeError(
            f"Triton modules were loaded during a native TK sweep: {loaded}"
        )


def _physical_gpu_index(args: argparse.Namespace) -> int:
    if args.physical_gpu is not None:
        return args.physical_gpu
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", maxsplit=1)[0]
    if visible.strip().isdigit():
        return int(visible)
    if args.device.startswith("cuda:") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return int(args.device.split(":", maxsplit=1)[1])
    raise RuntimeError(
        "Pass --physical-gpu when CUDA_VISIBLE_DEVICES does not begin with "
        "a physical numeric GPU index"
    )


def _pmon_snapshot(physical_gpu: int) -> dict[str, object]:
    raw = _command_output(["nvidia-smi", "pmon", "-i", str(physical_gpu), "-c", "1"])
    processes: list[dict[str, object]] = []
    if raw != "unavailable":
        for line in raw.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3 or fields[1] == "-":
                continue
            processes.append(
                {
                    "gpu": int(fields[0]),
                    "pid": int(fields[1]),
                    "type": fields[2],
                    "raw": line,
                }
            )
    return {
        "captured_utc": datetime.now(UTC).isoformat(),
        "raw": raw,
        "processes": processes,
    }


def _assert_no_foreign_gpu_process(
    physical_gpu: int,
    *,
    allowed_pids: set[int],
) -> dict[str, object]:
    snapshot = _pmon_snapshot(physical_gpu)
    foreign = [
        process
        for process in snapshot["processes"]
        if int(process["pid"]) not in allowed_pids
    ]
    if foreign:
        raise RuntimeError(
            "FOREIGN_GPU_PROCESS: physical GPU "
            f"{physical_gpu} has unexpected pmon entries {foreign}"
        )
    return snapshot


def _clock_sample(physical_gpu: int, phase: str) -> dict[str, object]:
    fields = (
        "timestamp,index,uuid,pstate,utilization.gpu,clocks.sm,clocks.mem,"
        "power.draw,temperature.gpu"
    )
    raw = _command_output(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    names = fields.split(",")
    values = [value.strip() for value in raw.split(",")]
    parsed = dict(zip(names, values, strict=False))
    return {
        "captured_utc": datetime.now(UTC).isoformat(),
        "phase": phase,
        "raw": raw,
        **parsed,
    }


def _quantize_matrix(
    backend,
    rows: int,
    cols: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randn(
        rows,
        cols,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    source.mul_(cols**-0.25)
    fp4, scales = backend.mxfp4_quantize_for_gemm(source)
    del source
    return fp4, scales


def _prepare_case(
    case: SweepCase,
    backend,
    module,
    generator: torch.Generator,
    device: torch.device,
) -> PreparedCase:
    if case.kernel in {"dense", "residual"}:
        a, a_sc = _quantize_matrix(backend, case.m, case.k, generator, device)
        b, b_sc = _quantize_matrix(backend, case.n, case.k, generator, device)
        reference = torch.empty(case.m, case.n, dtype=torch.bfloat16, device=device)
        candidate = torch.empty_like(reference)
        if case.kernel == "dense":
            launch_native = lambda outputs: module.mxfp4_gemm(  # noqa: E731
                a, a_sc, b, b_sc, outputs[0]
            )
            launch_config = (
                lambda config, outputs: module.mxfp4_gemm_config(  # noqa: E731
                    a, a_sc, b, b_sc, outputs[0], config
                )
            )
            extra: tuple[object, ...] = ()
        else:
            residual = torch.randn(
                case.m,
                case.n,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            launch_native = lambda outputs: module.mxfp4_gemm_residual(  # noqa: E731
                a, a_sc, b, b_sc, residual, outputs[0]
            )
            launch_config = (  # noqa: E731
                lambda config, outputs: module.mxfp4_gemm_residual_config(
                    a, a_sc, b, b_sc, residual, outputs[0], config
                )
            )
            extra = (residual,)
        return PreparedCase(
            [reference],
            [candidate],
            launch_native,
            launch_config,
            (a, a_sc, b, b_sc, reference, candidate, *extra),
        )

    if case.kernel == "batched":
        a, a_sc = _quantize_matrix(backend, case.m, case.k, generator, device)
        a_list = [a] * case.num_batches
        a_sc_list = [a_sc] * case.num_batches
        b_list: list[torch.Tensor] = []
        b_sc_list: list[torch.Tensor] = []
        for _ in range(case.num_batches):
            b, b_sc = _quantize_matrix(backend, case.n, case.k, generator, device)
            b_list.append(b)
            b_sc_list.append(b_sc)
        reference = [
            torch.empty(case.m, case.n, dtype=torch.bfloat16, device=device)
            for _ in range(case.num_batches)
        ]
        candidate = [torch.empty_like(output) for output in reference]
        launch_native = lambda outputs: module.mxfp4_batched_gemm(  # noqa: E731
            a_list, a_sc_list, b_list, b_sc_list, outputs
        )
        launch_config = (
            lambda config, outputs: module.mxfp4_batched_gemm_config(  # noqa: E731
                a_list, a_sc_list, b_list, b_sc_list, outputs, config
            )
        )
        return PreparedCase(
            reference,
            candidate,
            launch_native,
            launch_config,
            (a, a_sc, b_list, b_sc_list, reference, candidate),
        )

    if case.kernel not in {"split2_onepass", "split3_onepass"}:
        raise ValueError(f"Unsupported sweep kernel {case.kernel!r}")

    a, a_sc = _quantize_matrix(backend, case.m, case.k, generator, device)
    offsets: list[int] = []
    packed_widths: list[int] = []
    a_sc_list: list[torch.Tensor] = []
    b_list: list[torch.Tensor] = []
    b_sc_list: list[torch.Tensor] = []
    unpacked_offset = 0
    for width in case.split_k:
        offsets.append(unpacked_offset // 2)
        packed_widths.append(width // 2)
        a_sc_list.append(a_sc.narrow(1, unpacked_offset // 128, width // 128))
        b, b_sc = _quantize_matrix(backend, case.n, width, generator, device)
        b_list.append(b)
        b_sc_list.append(b_sc)
        unpacked_offset += width

    reference = torch.empty(case.m, case.n, dtype=torch.bfloat16, device=device)
    candidate = torch.empty_like(reference)
    function = getattr(
        module,
        (
            "mxfp4_split2_dgrad_strided_onepass_gemm"
            if case.kernel == "split2_onepass"
            else "mxfp4_split3_dgrad_strided_onepass_gemm"
        ),
    )

    def launch(config: int, outputs: list[torch.Tensor]) -> None:
        function(
            a,
            a_sc_list,
            offsets,
            packed_widths,
            b_list,
            b_sc_list,
            outputs[0],
            config,
        )

    return PreparedCase(
        [reference],
        [candidate],
        lambda outputs: launch(-1, outputs),
        launch,
        (
            a,
            a_sc,
            a_sc_list,
            b_list,
            b_sc_list,
            reference,
            candidate,
        ),
    )


def _parity(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    *,
    atol: float,
    rtol: float,
    chunk_rows: int,
) -> dict[str, object]:
    exact = True
    violations = 0
    elements = 0
    max_abs = 0.0
    max_rel = 0.0
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        for start in range(0, actual_tensor.size(0), chunk_rows):
            stop = min(start + chunk_rows, actual_tensor.size(0))
            lhs = actual_tensor[start:stop].float()
            rhs = expected_tensor[start:stop].float()
            diff = (lhs - rhs).abs()
            tolerance = atol + rtol * rhs.abs()
            exact = exact and torch.equal(lhs, rhs)
            violations += int((diff > tolerance).sum().item())
            elements += diff.numel()
            max_abs = max(max_abs, float(diff.max().item()))
            denominator = rhs.abs().clamp_min(torch.finfo(torch.float32).tiny)
            max_rel = max(max_rel, float((diff / denominator).max().item()))
    return {
        "passed": violations == 0,
        "bitwise_exact": exact,
        "violations": violations,
        "elements": elements,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
    }


def _timings(
    prepared: PreparedCase,
    valid_configs: list[int],
    *,
    warmup: int,
    iters: int,
    trials: int,
    seed: int,
    stabilize_ms: float,
    physical_gpu: int,
) -> tuple[
    dict[str, list[float]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    labels = ["native", *(str(config) for config in valid_configs)]

    def run(label: str) -> None:
        if label == "native":
            prepared.launch_native(prepared.candidate_outputs)
        else:
            prepared.launch_config(int(label), prepared.candidate_outputs)

    stabilize_deadline = time.monotonic() + stabilize_ms / 1000.0
    while time.monotonic() < stabilize_deadline:
        for label in labels:
            for _ in range(iters):
                run(label)
        torch.cuda.synchronize()

    clock_samples = [_clock_sample(physical_gpu, "post_stabilize")]
    pmon_samples = [
        _assert_no_foreign_gpu_process(
            physical_gpu,
            allowed_pids={os.getpid()},
        )
    ]
    for label in labels:
        for _ in range(warmup):
            run(label)
    torch.cuda.synchronize()

    rng = random.Random(seed)
    samples = {label: [] for label in labels}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for trial in range(trials):
        order = labels.copy()
        rng.shuffle(order)
        trial_samples = {label: [] for label in labels}
        for trial_order in (order, list(reversed(order))):
            for label in trial_order:
                start.record()
                for _ in range(iters):
                    run(label)
                end.record()
                end.synchronize()
                trial_samples[label].append(float(start.elapsed_time(end) / iters))
        for label in labels:
            samples[label].append(statistics.mean(trial_samples[label]))
        clock_samples.append(_clock_sample(physical_gpu, f"trial_{trial}"))
        pmon_samples.append(
            _assert_no_foreign_gpu_process(
                physical_gpu,
                allowed_pids={os.getpid()},
            )
        )
        if trial + 1 < trials:
            for label in labels:
                for _ in range(warmup):
                    run(label)
            torch.cuda.synchronize()
    return samples, clock_samples, pmon_samples


def _command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(
    args: argparse.Namespace,
    backend,
    roots: dict[str, dict[str, object]],
) -> dict[str, object]:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    physical_gpu = _physical_gpu_index(args)
    physical_sample = _clock_sample(physical_gpu, "metadata")
    physical_uuid = str(physical_sample.get("uuid", "")).removeprefix("GPU-")
    if physical_uuid != str(props.uuid):
        raise RuntimeError(
            f"Logical CUDA device UUID {props.uuid} does not match physical GPU "
            f"{physical_gpu} UUID {physical_sample.get('uuid')}"
        )
    gemm_path = Path(backend._candidate_gemm_paths()[0]).resolve()
    quant_path = Path(backend._candidate_quant_paths()[0]).resolve()
    quant_root = Path(str(roots["quant"]["path"]))
    gemm_root = Path(str(roots["gemm"]["path"]))
    if not quant_path.is_relative_to(quant_root):
        raise RuntimeError(
            f"Loaded quantizer {quant_path} is outside pinned root {quant_root}"
        )
    if not gemm_path.is_relative_to(gemm_root):
        raise RuntimeError(
            f"Loaded GEMM extension {gemm_path} is outside pinned root {gemm_root}"
        )
    gemm_sha256 = _sha256(gemm_path)
    quant_sha256 = _sha256(quant_path)
    if gemm_sha256 != PINNED_GEMM_EXTENSION_SHA256:
        raise RuntimeError(
            "PROVENANCE_ERROR: loaded GEMM extension SHA256 changed to "
            f"{gemm_sha256}; expected {PINNED_GEMM_EXTENSION_SHA256}"
        )
    if quant_sha256 != PINNED_QUANT_EXTENSION_SHA256:
        raise RuntimeError(
            "PROVENANCE_ERROR: loaded quant extension SHA256 changed to "
            f"{quant_sha256}; expected {PINNED_QUANT_EXTENSION_SHA256}"
        )
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_commit": _command_output(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "repo_status": _command_output(["git", "status", "--short"], REPO_ROOT),
        "fp4_roots": roots,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": {
            "name": props.name,
            "uuid": str(props.uuid),
            "capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "logical_index": device,
            "physical_index": physical_gpu,
        },
        "physical_gpu_metadata_sample": physical_sample,
        "gemm_extension": str(gemm_path),
        "gemm_extension_sha256": gemm_sha256,
        "gemm_extension_size_bytes": gemm_path.stat().st_size,
        "quant_extension": str(quant_path),
        "quant_extension_sha256": quant_sha256,
        "quant_extension_size_bytes": quant_path.stat().st_size,
        "quant_backend_version": backend.mxfp4_backend_version(),
        "timing": {
            "warmup": args.warmup,
            "iters": args.iters,
            "trials": args.trials,
            "stabilize_ms": args.stabilize_ms,
            "min_speedup": args.min_speedup,
            "seed": args.seed,
            "trial_order": "randomized ABBA; mean of two blocks per label",
        },
        "parity": {
            "reference": "native unconfigured TK entrypoint",
            "atol": args.atol,
            "rtol": args.rtol,
            "chunk_rows": args.parity_chunk_rows,
        },
    }


def _run_case(
    case: SweepCase,
    args: argparse.Namespace,
    backend,
    module,
    device: torch.device,
) -> dict[str, object]:
    print(
        f"\n[{case.name}] kernel={case.kernel} orientation={case.orientation} "
        f"M={case.m} N={case.n} K={case.k} B={case.num_batches} "
        f"split_k={case.split_k or '-'}",
        flush=True,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    prepared = _prepare_case(case, backend, module, generator, device)
    prepared.launch_native(prepared.reference_outputs)
    torch.cuda.synchronize()

    configs = (
        [int(value) for value in args.configs.split(",") if value.strip()]
        if args.configs
        else list(CONFIGS_BY_KERNEL[case.kernel])
    )
    parity_rows: dict[int, dict[str, object]] = {}
    valid_configs: list[int] = []
    for config in configs:
        try:
            prepared.launch_config(config, prepared.candidate_outputs)
            torch.cuda.synchronize()
            parity = _parity(
                prepared.candidate_outputs,
                prepared.reference_outputs,
                atol=args.atol,
                rtol=args.rtol,
                chunk_rows=args.parity_chunk_rows,
            )
            parity_rows[config] = parity
            if parity["passed"]:
                valid_configs.append(config)
            print(
                f"  config={config:>2} parity={'PASS' if parity['passed'] else 'FAIL'} "
                f"exact={parity['bitwise_exact']} max_abs={parity['max_abs']:.8g} "
                f"violations={parity['violations']}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            parity_rows[config] = {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  config={config:>2} ERROR {error}", flush=True)

    samples, clock_samples, pmon_samples = _timings(
        prepared,
        valid_configs,
        warmup=args.warmup,
        iters=args.iters,
        trials=args.trials,
        seed=args.seed + sum(ord(character) for character in case.name),
        stabilize_ms=args.stabilize_ms,
        physical_gpu=_physical_gpu_index(args),
    )
    native_median = statistics.median(samples["native"])
    timing_rows: dict[str, dict[str, object]] = {}
    for label, trial_samples in samples.items():
        median_ms = statistics.median(trial_samples)
        speedup = native_median / median_ms
        timing_rows[label] = {
            "trials_ms": trial_samples,
            "median_ms": median_ms,
            "min_ms": min(trial_samples),
            "max_ms": max(trial_samples),
            "tflops": case.flops / (median_ms * 1e-3) * 1e-12,
            "speedup_vs_native": speedup,
        }
        print(
            f"  timing={label:>6} median_ms={median_ms:.6f} "
            f"tflops={timing_rows[label]['tflops']:.2f} speedup={speedup:.5f}",
            flush=True,
        )

    required_trial_wins = math.ceil(args.trials * 2 / 3)
    eligible: list[tuple[float, int]] = []
    for config in valid_configs:
        row = timing_rows[str(config)]
        paired_wins = sum(
            candidate < native
            for candidate, native in zip(
                row["trials_ms"],
                timing_rows["native"]["trials_ms"],
                strict=True,
            )
        )
        row["paired_trial_wins"] = paired_wins
        row["eligible"] = (
            row["speedup_vs_native"] >= args.min_speedup
            and paired_wins >= required_trial_wins
        )
        if row["eligible"]:
            eligible.append((row["median_ms"], config))
    winner = min(eligible)[1] if eligible else None
    print(f"  winner={winner if winner is not None else 'native'}", flush=True)

    result = {
        "selector": asdict(case),
        "native": timing_rows["native"],
        "candidates": {
            str(config): {
                "parity": parity_rows[config],
                "timing": timing_rows.get(str(config)),
            }
            for config in configs
        },
        "winner": winner,
        "gpu_samples": {
            "clock": clock_samples,
            "pmon": pmon_samples,
        },
    }
    del prepared
    torch.cuda.empty_cache()
    return result


def _write_document(document: dict[str, object], output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        print(f"\nresults={output.resolve()}")
    else:
        print(json.dumps(document, indent=2, sort_keys=True))


def _run_worker_process(
    command: list[str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        )
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )


def _isolated_worker_command(
    args: argparse.Namespace,
    case: SweepCase,
    config: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        case.name,
        "--configs",
        str(config),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--trials",
        str(args.trials),
        "--stabilize-ms",
        str(args.stabilize_ms),
        "--seed",
        str(args.seed),
        "--min-speedup",
        str(args.min_speedup),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--parity-chunk-rows",
        str(args.parity_chunk_rows),
        "--device",
        args.device,
        "--physical-gpu",
        str(_physical_gpu_index(args)),
        "--fp4-root",
        str(args.fp4_root),
        "--gemm-root",
        str(args.gemm_root),
        "--expected-fp4-commit",
        args.expected_fp4_commit,
        "--output",
        str(output),
    ]


def _run_isolated(
    args: argparse.Namespace,
    cases: list[SweepCase],
) -> int:
    started = time.monotonic()
    physical_gpu = _physical_gpu_index(args)
    parent_pmon_samples = [
        _assert_no_foreign_gpu_process(physical_gpu, allowed_pids=set())
    ]
    parent_clock_samples = [_clock_sample(physical_gpu, "parent_preflight")]
    metadata: dict[str, object] | None = None
    combined_cases: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="mxfp4-selector-workers-") as temp_dir:
        temp_root = Path(temp_dir)
        for case in cases:
            configs = (
                [int(value) for value in args.configs.split(",") if value.strip()]
                if args.configs
                else list(CONFIGS_BY_KERNEL[case.kernel])
            )
            combined: dict[str, object] = {
                "selector": asdict(case),
                "native": {"paired_by_config": {}},
                "candidates": {},
                "winner": None,
            }
            eligible: list[tuple[float, float, int]] = []
            native_medians: list[float] = []
            print(
                f"\n[{case.name}] isolated_configs={','.join(map(str, configs))}",
                flush=True,
            )
            for config in configs:
                _validate_roots(args)
                _validate_pinned_artifacts(args)
                parent_pmon_samples.append(
                    _assert_no_foreign_gpu_process(
                        physical_gpu,
                        allowed_pids=set(),
                    )
                )
                parent_clock_samples.append(
                    _clock_sample(
                        physical_gpu,
                        f"before_{case.name}_config_{config}",
                    )
                )
                worker_output = temp_root / f"{case.name}-config-{config}.json"
                command = _isolated_worker_command(
                    args,
                    case,
                    config,
                    worker_output,
                )
                try:
                    process = _run_worker_process(
                        command,
                        args.worker_timeout,
                    )
                except subprocess.TimeoutExpired as error:
                    combined["candidates"][str(config)] = {
                        "parity": {
                            "passed": False,
                            "error": (
                                f"worker timeout after {args.worker_timeout:.1f}s"
                            ),
                        },
                        "timing": None,
                    }
                    print(
                        f"  config={config:>2} REJECT worker timeout "
                        f"after {args.worker_timeout:.1f}s",
                        flush=True,
                    )
                    if error.stdout:
                        print(str(error.stdout)[-2000:], flush=True)
                    continue

                if process.returncode != 0 or not worker_output.exists():
                    error_tail = (process.stderr or process.stdout)[-4000:]
                    if "FOREIGN_GPU_PROCESS" in error_tail:
                        raise RuntimeError(
                            "Authoritative sweep aborted after foreign GPU "
                            f"activity was detected:\n{error_tail}"
                        )
                    if "PROVENANCE_ERROR" in error_tail:
                        raise RuntimeError(
                            "Authoritative sweep aborted after pinned extension "
                            f"provenance changed:\n{error_tail}"
                        )
                    combined["candidates"][str(config)] = {
                        "parity": {
                            "passed": False,
                            "error": (
                                f"worker exited {process.returncode}: {error_tail}"
                            ),
                        },
                        "timing": None,
                    }
                    print(
                        f"  config={config:>2} REJECT worker_exit={process.returncode}",
                        flush=True,
                    )
                    if error_tail:
                        print(error_tail, flush=True)
                    continue

                worker_document = json.loads(worker_output.read_text())
                worker_case = worker_document["cases"][0]
                if metadata is None:
                    metadata = worker_document["metadata"]
                candidate = worker_case["candidates"][str(config)]
                candidate["gpu_samples"] = worker_case["gpu_samples"]
                native = worker_case["native"]
                candidate["paired_native"] = native
                combined["candidates"][str(config)] = candidate
                combined["native"]["paired_by_config"][str(config)] = native
                native_medians.append(float(native["median_ms"]))
                timing = candidate["timing"]
                parity = candidate["parity"]
                if timing is not None and timing.get("eligible", False):
                    eligible.append(
                        (
                            -float(timing["speedup_vs_native"]),
                            float(timing["median_ms"]),
                            config,
                        )
                    )
                print(
                    f"  config={config:>2} "
                    f"parity={'PASS' if parity['passed'] else 'FAIL'} "
                    f"exact={parity.get('bitwise_exact', False)} "
                    f"native_ms={native['median_ms']:.6f} "
                    f"candidate_ms="
                    f"{timing['median_ms'] if timing is not None else 'n/a'} "
                    f"speedup="
                    f"{timing['speedup_vs_native'] if timing is not None else 'n/a'} "
                    f"eligible={timing.get('eligible', False) if timing else False}",
                    flush=True,
                )

            if native_medians:
                combined["native"]["median_of_paired_medians_ms"] = statistics.median(
                    native_medians
                )
                combined["native"]["paired_medians_ms"] = native_medians
            if eligible:
                combined["winner"] = min(eligible)[2]
            print(
                f"  winner="
                f"{combined['winner'] if combined['winner'] is not None else 'native'}",
                flush=True,
            )
            combined_cases.append(combined)

    if metadata is None:
        metadata = {
            "created_utc": datetime.now(UTC).isoformat(),
            "repo_commit": _command_output(["git", "rev-parse", "HEAD"], REPO_ROOT),
            "error": "No isolated worker completed successfully",
        }
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["isolation"] = {
        "mode": "one subprocess per case/config",
        "worker_timeout_seconds": args.worker_timeout,
    }
    _validate_roots(args)
    _validate_pinned_artifacts(args)
    parent_pmon_samples.append(
        _assert_no_foreign_gpu_process(physical_gpu, allowed_pids=set())
    )
    parent_clock_samples.append(_clock_sample(physical_gpu, "parent_postflight"))
    metadata["parent_gpu_samples"] = {
        "physical_gpu": physical_gpu,
        "clock": parent_clock_samples,
        "pmon": parent_pmon_samples,
    }
    document = {
        "schema_version": 3,
        "metadata": metadata,
        "cases": combined_cases,
    }
    _write_document(document, args.output)
    return 0


def main() -> int:
    args = _parse_args()
    if args.list_cases:
        for case in PRODUCTION_CASES:
            print(case.name)
        return 0
    if (
        min(
            args.warmup,
            args.iters,
            args.trials,
            args.parity_chunk_rows,
            args.stabilize_ms,
        )
        <= 0
    ):
        raise ValueError(
            "warmup, iters, trials, parity chunk rows, and stabilize-ms "
            "must be positive"
        )

    wanted = set(args.case)
    cases = [case for case in PRODUCTION_CASES if not wanted or case.name in wanted]
    unknown = wanted.difference(case.name for case in PRODUCTION_CASES)
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")

    if not args.worker and not args.no_isolate_configs:
        _validate_roots(args)
        _validate_pinned_artifacts(args)
        return _run_isolated(args, cases)

    roots = _validate_roots(args)
    _validate_pinned_artifacts(args)
    backend = _setup_backend(args)
    _ensure_native_only()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    module = backend._load_gemm_module()
    _ensure_native_only()

    document = {
        "schema_version": 1,
        "metadata": _metadata(args, backend, roots),
        "cases": [],
    }
    started = time.monotonic()
    for case in cases:
        document["cases"].append(_run_case(case, args, backend, module, device))
    document["metadata"]["elapsed_seconds"] = time.monotonic() - started
    _ensure_native_only()
    document["metadata"]["triton_loaded"] = False
    _write_document(document, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
