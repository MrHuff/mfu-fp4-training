"""Shared contracts for the mixed MXFP4/localCTA correctness and speed gate.

The CUDA-facing command lives in :mod:`tools.check_mixed_mxfp4_localcta_dgrad`.
This module deliberately contains no backend imports so its receipt, tensor,
storage, timing, and logical-GEMM accounting rules can be tested on CPU before
the optional fused extension exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch


SCHEMA_VERSION = 1
METHOD = "mixed-mxfp4-localcta-dgrad-correctness-speed-gate-v2"
ROUTE_ID = "mxfp4_fixed_h32_col_localcta_row_sr_dgrad_v2"
SUBSEQUENCE_STRIDE = 1 << 32

EXACT_RECIPE = {
    "forward": "mxfp4_v4",
    "dweight": "mxfp4_v4_col_rht",
    "dhidden": "localcta_v4_row_sr",
    "mxfp4_rht_axes": "col",
    "mxfp4_rht_block_size": 32,
    "mxfp4_rht_signs": "fixed_0x2817",
    "rht_activation": True,
    "rht_grad": True,
    "rht_weight": False,
    "data_sr_activation": False,
    "data_sr_grad": True,
    "data_sr_grad_axes": "row",
    "data_sr_weight": False,
    "scale_sr": False,
    "weight_quantization": "2d",
    "sr_stream": "existing_ranked_mxfp4_logical_producer",
    "sr_reservations_per_backward": 1,
}


class GateFailure(RuntimeError):
    """A correctness, lifetime, recipe, or performance gate failed."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the one canonical byte representation used by receipt seals."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a SHA-256 seal without permitting a pre-existing seal."""

    if "seal" in payload:
        raise ValueError("receipt payload must not already contain a seal")
    result = dict(payload)
    result["seal"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": sha256(canonical_json_bytes(result)).hexdigest(),
    }
    return result


def validate_receipt_seal(receipt: Mapping[str, Any]) -> None:
    seal = receipt.get("seal")
    if not isinstance(seal, Mapping) or seal.get("algorithm") != "sha256":
        raise GateFailure("receipt has no supported SHA-256 seal")
    payload = dict(receipt)
    payload.pop("seal")
    actual = sha256(canonical_json_bytes(payload)).hexdigest()
    if actual != seal.get("canonical_payload_sha256"):
        raise GateFailure("receipt SHA-256 seal mismatch")


def write_receipt_exclusive(path: str | Path, receipt: Mapping[str, Any]) -> None:
    """Write one sealed receipt and refuse to replace existing evidence."""

    validate_receipt_seal(receipt)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(receipt))


def _byte_view(value: torch.Tensor) -> torch.Tensor:
    if not value.is_contiguous():
        raise GateFailure(
            f"exact comparison requires contiguous tensors, got stride={value.stride()}"
        )
    return value.detach().view(torch.uint8).reshape(-1)


def exact_tensor_report(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> dict[str, Any]:
    """Compare tensor metadata and every stored byte, including packed FP4."""

    metadata_match = (
        candidate.shape == reference.shape
        and candidate.dtype == reference.dtype
        and candidate.stride() == reference.stride()
    )
    if not metadata_match:
        return {
            "name": name,
            "pass": False,
            "candidate": {
                "shape": list(candidate.shape),
                "dtype": str(candidate.dtype),
                "stride": list(candidate.stride()),
            },
            "reference": {
                "shape": list(reference.shape),
                "dtype": str(reference.dtype),
                "stride": list(reference.stride()),
            },
            "reason": "metadata_mismatch",
        }
    candidate_bytes = _byte_view(candidate)
    reference_bytes = _byte_view(reference)
    mismatch = candidate_bytes != reference_bytes
    mismatched_bytes = int(mismatch.sum().item())
    return {
        "name": name,
        "pass": mismatched_bytes == 0,
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "num_bytes": candidate_bytes.numel(),
        "mismatched_bytes": mismatched_bytes,
        "candidate_sha256": sha256(candidate_bytes.cpu().numpy().tobytes()).hexdigest(),
        "reference_sha256": sha256(reference_bytes.cpu().numpy().tobytes()).hexdigest(),
    }


def require_exact_tensors(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if set(candidate) != set(reference):
        raise GateFailure(
            "exact tensor maps differ: "
            f"candidate={sorted(candidate)}, reference={sorted(reference)}"
        )
    reports = {
        name: exact_tensor_report(candidate[name], reference[name], name=name)
        for name in sorted(candidate)
    }
    failures = [name for name, report in reports.items() if not report["pass"]]
    if failures:
        raise GateFailure(f"bit-exact tensor gate failed: {failures}")
    return {"pass": True, "zero_tolerance": True, "tensors": reports}


def strict_close_report(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Strict numeric comparison with explicit, receipt-visible tolerances."""

    if candidate.shape != reference.shape:
        raise GateFailure(
            f"{name} shape mismatch: {tuple(candidate.shape)} != {tuple(reference.shape)}"
        )
    candidate_f = candidate.detach().float()
    reference_f = reference.detach().float()
    finite = bool(
        torch.isfinite(candidate_f).all().item()
        and torch.isfinite(reference_f).all().item()
    )
    if not finite:
        raise GateFailure(f"{name} contains non-finite values")
    difference = (candidate_f - reference_f).abs()
    tolerance = atol + rtol * reference_f.abs()
    mismatched = difference > tolerance
    report = {
        "name": name,
        "pass": not bool(mismatched.any().item()),
        "shape": list(candidate.shape),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
        "atol": float(atol),
        "rtol": float(rtol),
        "mismatched_elements": int(mismatched.sum().item()),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
    }
    if not report["pass"]:
        raise GateFailure(
            f"{name} exceeded atol={atol} rtol={rtol}: "
            f"max_abs={report['max_abs']} mismatches={report['mismatched_elements']}"
        )
    return report


def require_finite(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    failures: list[str] = []
    for name, value in sorted(tensors.items()):
        finite = bool(torch.isfinite(value.detach().float()).all().item())
        result[name] = {
            "finite": finite,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if not finite:
            failures.append(name)
    if failures:
        raise GateFailure(f"non-finite tensors: {failures}")
    return {"pass": True, "tensors": result}


@dataclass(frozen=True)
class StorageInterval:
    name: str
    device: str
    start: int
    stop: int


def storage_interval(name: str, value: torch.Tensor) -> StorageInterval:
    if not value.is_contiguous():
        raise GateFailure(f"payload {name} must be contiguous")
    element_bytes = value.element_size()
    start = value.untyped_storage().data_ptr() + value.storage_offset() * element_bytes
    stop = start + value.numel() * element_bytes
    return StorageInterval(name=name, device=str(value.device), start=start, stop=stop)


def require_disjoint_payload_storage(
    tensors: Mapping[str, torch.Tensor],
    *,
    allowed_aliases: Iterable[frozenset[str]] = (),
) -> dict[str, Any]:
    """Reject accidental overlap between logically independent payloads."""

    allowed = set(allowed_aliases)
    intervals = [storage_interval(name, value) for name, value in sorted(tensors.items())]
    overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            if left.device != right.device:
                continue
            if max(left.start, right.start) < min(left.stop, right.stop):
                pair = frozenset((left.name, right.name))
                if pair not in allowed:
                    overlaps.append(
                        {
                            "left": left.name,
                            "right": right.name,
                            "bytes": min(left.stop, right.stop)
                            - max(left.start, right.start),
                        }
                    )
    if overlaps:
        raise GateFailure(f"payload storage unexpectedly aliases: {overlaps}")
    return {
        "pass": True,
        "intervals": [interval.__dict__ for interval in intervals],
        "unexpected_overlaps": [],
    }


def require_distinct_scale_carriers(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_name: str,
    right_name: str,
    require_distinct_values: bool,
) -> dict[str, Any]:
    """Prove two independently finalized outer-SG carriers were not shared."""

    if left is right:
        raise GateFailure(f"{left_name} and {right_name} are the same tensor object")
    storage = require_disjoint_payload_storage(
        {left_name: left, right_name: right}
    )
    left_bytes = _byte_view(left)
    right_bytes = _byte_view(right)
    byte_equal = bool(
        left_bytes.numel() == right_bytes.numel()
        and torch.equal(left_bytes, right_bytes)
    )
    if require_distinct_values and byte_equal:
        raise GateFailure(
            f"{left_name} and {right_name} are byte-identical even though the "
            "test arms have deliberately different amax ranges"
        )
    return {
        "pass": True,
        "same_object": False,
        "storage": storage,
        "byte_equal": byte_equal,
        "distinct_values_required": bool(require_distinct_values),
    }


def require_one_sr_advance(
    before: tuple[int, int],
    after: tuple[int, int],
    *,
    stride: int = SUBSEQUENCE_STRIDE,
) -> dict[str, Any]:
    """Require one format-agnostic logical-producer SR reservation."""

    expected = (int(before[0]), int(before[1]) + int(stride))
    passed = tuple(map(int, after)) == expected
    report = {
        "pass": passed,
        "before": [int(before[0]), int(before[1])],
        "after": [int(after[0]), int(after[1])],
        "expected_after": list(expected),
        "stride": int(stride),
        "stream_semantics": "format_agnostic_existing_mxfp4_logical_producer",
    }
    if not passed:
        raise GateFailure(f"SR counter must advance exactly once: {report}")
    return report


class LogicalGemmAudit:
    """Count semantic GEMMs while allowing implementation kernels to vary."""

    def __init__(self, expected: Mapping[str, int]) -> None:
        self.expected = Counter({str(key): int(value) for key, value in expected.items()})
        self.observed: Counter[str] = Counter()

    @contextmanager
    def record(self, label: str):
        self.observed[str(label)] += 1
        yield

    def wrap(self, label: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args, **kwargs):
            with self.record(label):
                return function(*args, **kwargs)

        return wrapped

    def report(self) -> dict[str, Any]:
        passed = self.observed == self.expected
        report = {
            "pass": passed,
            "expected": dict(sorted(self.expected.items())),
            "observed": dict(sorted(self.observed.items())),
            "expected_total": sum(self.expected.values()),
            "observed_total": sum(self.observed.values()),
        }
        if not passed:
            raise GateFailure(f"logical GEMM count mismatch: {report}")
        return report


def summarize_ms(samples_ms: Sequence[float]) -> dict[str, Any]:
    if not samples_ms:
        raise ValueError("timing sample list must be non-empty")
    values = [float(value) for value in samples_ms]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise GateFailure(f"invalid timing samples: {values}")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "samples": len(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": min(values),
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "max_ms": max(values),
    }


def cuda_event_samples(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    before_each: Callable[[], None] | None = None,
) -> list[float]:
    """Measure one callable with CUDA events and an explicit scored window."""

    if not torch.cuda.is_available():
        raise GateFailure("CUDA timing requested but CUDA is unavailable")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        if before_each is not None:
            before_each()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples


def timing_comparison(
    baseline_ms: Sequence[float],
    candidate_ms: Sequence[float],
) -> dict[str, Any]:
    baseline = summarize_ms(baseline_ms)
    candidate = summarize_ms(candidate_ms)
    denominator = max(float(baseline["median_ms"]), torch.finfo(torch.float64).tiny)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "median_ratio": float(candidate["median_ms"]) / denominator,
        "median_overhead_pct": (float(candidate["median_ms"]) / denominator - 1.0)
        * 100.0,
    }
