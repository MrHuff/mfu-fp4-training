#!/usr/bin/env python3
"""Estimate constant seed-42 loss offsets from public control histories."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics


EXPECTED_STEPS = (1, *range(50, 2_001, 50))
HUBER_K = 1.345


@dataclass(frozen=True)
class HuberEstimate:
    location: float
    mad_scale: float
    iterations: int


def huber_location(values: list[float], k: float = HUBER_K) -> HuberEstimate:
    """Return a deterministic Huber M-location with a MAD-fixed scale."""

    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError("Huber input must be finite and nonempty")
    location = statistics.median(values)
    mad = statistics.median(abs(value - location) for value in values)
    scale = 1.4826 * mad
    if scale == 0.0:
        return HuberEstimate(location=location, mad_scale=0.0, iterations=0)
    for iteration in range(1, 101):
        weights = []
        for value in values:
            residual = abs((value - location) / scale)
            weights.append(1.0 if residual <= k else k / residual)
        updated = sum(
            weight * value for weight, value in zip(weights, values, strict=True)
        ) / sum(weights)
        if abs(updated - location) <= 1.0e-12:
            return HuberEstimate(updated, scale, iteration)
        location = updated
    raise RuntimeError("Huber location did not converge")


def load_historical(snapshot: Path) -> dict[str, dict[int, float]]:
    result = {"bf16": {}, "te_native": {}}
    with snapshot.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lineage = row["lineage"]
            if lineage not in result:
                continue
            step = int(row["step"])
            if step in EXPECTED_STEPS:
                if step in result[lineage]:
                    raise RuntimeError(f"duplicate historical step: {lineage} {step}")
                result[lineage][step] = float(row["loss"])
    for lineage, values in result.items():
        if tuple(sorted(values)) != EXPECTED_STEPS:
            raise RuntimeError(f"historical overlap is incomplete for {lineage}")
    return result


def load_seed_history(lineage: str, path: Path) -> dict[int, float]:
    """Load a public two-column ``step,loss`` seed-control history."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("step", "loss"):
            raise RuntimeError(f"invalid seed-control columns for {lineage}")
        rows = list(reader)
    losses = {int(row["step"]): float(row["loss"]) for row in rows}
    if tuple(sorted(losses)) != EXPECTED_STEPS:
        raise RuntimeError(f"seed-control step inventory is not exact for {lineage}")
    if not all(math.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"seed-control contains non-finite loss for {lineage}")
    return losses


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    historical = load_historical(args.historical_snapshot)
    histories = {
        "bf16": args.bf16_seed_history,
        "te_native": args.te_seed_history,
    }
    overlap_rows = []
    adjustment_rows = []
    for lineage in ("bf16", "te_native"):
        seed = load_seed_history(lineage, histories[lineage])
        differences = []
        for step in EXPECTED_STEPS:
            difference = seed[step] - historical[lineage][step]
            differences.append(difference)
            overlap_rows.append(
                {
                    "lineage": lineage,
                    "step": step,
                    "historical_loss": f"{historical[lineage][step]:.12f}",
                    "seed42_loss": f"{seed[step]:.12f}",
                    "seed42_minus_historical": f"{difference:.12f}",
                }
            )
        estimate = huber_location(differences)
        adjustment_rows.append(
            {
                "lineage": lineage,
                "matched_points": len(differences),
                "first_step": EXPECTED_STEPS[0],
                "last_step": EXPECTED_STEPS[-1],
                "estimator": "Huber location of exact-step loss differences",
                "huber_k": HUBER_K,
                "loss_offset_seed42_minus_historical": f"{estimate.location:.12f}",
                "raw_mean_difference": f"{statistics.fmean(differences):.12f}",
                "median_difference": f"{statistics.median(differences):.12f}",
                "sample_std_difference": f"{statistics.stdev(differences):.12f}",
                "mad_scale": f"{estimate.mad_scale:.12f}",
                "huber_iterations": estimate.iterations,
            }
        )
    write_csv(
        args.overlap_out,
        overlap_rows,
        [
            "lineage",
            "step",
            "historical_loss",
            "seed42_loss",
            "seed42_minus_historical",
        ],
    )
    write_csv(
        args.adjustment_out,
        adjustment_rows,
        [
            "lineage",
            "matched_points",
            "first_step",
            "last_step",
            "estimator",
            "huber_k",
            "loss_offset_seed42_minus_historical",
            "raw_mean_difference",
            "median_difference",
            "sample_std_difference",
            "mad_scale",
            "huber_iterations",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-snapshot", type=Path, required=True)
    parser.add_argument("--bf16-seed-history", type=Path, required=True)
    parser.add_argument("--te-seed-history", type=Path, required=True)
    parser.add_argument("--overlap-out", type=Path, required=True)
    parser.add_argument("--adjustment-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
