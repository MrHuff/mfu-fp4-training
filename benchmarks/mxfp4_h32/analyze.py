"""Create strict, topology-parameterized MXFP4 performance receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+0-9.eEinfnaINFNA]+"
METRIC = re.compile(
    rf"step:\s*(?P<step>\d+)\s+"
    rf"loss:\s*(?P<loss>{NUMBER})\s+"
    rf"grad_norm:\s*(?P<grad>{NUMBER})\s+"
    rf"memory:\s*(?P<memory>{NUMBER})GiB\([^)]*\)\s+"
    rf"tps:\s*(?P<tps>[0-9,]+)\s+"
    rf"tflops:\s*(?P<tflops>[-+0-9.,eEinfnaINFNA]+)\s+"
    rf"mfu:\s*(?P<mfu>{NUMBER})%"
)
FIELDS = (
    "loss",
    "grad_norm",
    "memory_gib",
    "tps_per_gpu",
    "tflops_per_gpu",
    "mfu_percent",
)
NODE_SCHEMA = "mfu-route-probe-node-v3"
AGGREGATE_SCHEMA = "mfu-route-probe-aggregate-v3"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def window_summary(
    rows: dict[int, dict[str, float]], first: int, last: int
) -> dict[str, Any]:
    selected = [rows[step] for step in range(first, last + 1)]
    result: dict[str, Any] = {
        "first_step": first,
        "last_step": last,
        "updates": len(selected),
    }
    for field in FIELDS:
        values = [row[field] for row in selected]
        result[field] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p10": percentile(values, 0.10),
            "p90": percentile(values, 0.90),
            "min": min(values),
            "max": max(values),
        }
    return result


def parse_node_log(
    path: Path, final_step: int, local_processes: int
) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
    by_step: dict[int, list[dict[str, float]]] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC.search(ANSI.sub("", raw))
        if not match:
            continue
        step = int(match.group("step"))
        if not 1 <= step <= final_step:
            continue
        values = {
            "loss": float(match.group("loss")),
            "grad_norm": float(match.group("grad")),
            "memory_gib": float(match.group("memory")),
            "tps_per_gpu": float(match.group("tps").replace(",", "")),
            "tflops_per_gpu": float(match.group("tflops").replace(",", "")),
            "mfu_percent": float(match.group("mfu")),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite metric at step {step} in {path}")
        by_step[step].append(values)

    required = set(range(1, final_step + 1))
    missing = sorted(required - by_step.keys())
    if missing:
        raise ValueError(f"{path} missing metric steps; first={missing[:8]}")
    counts = {step: len(by_step[step]) for step in sorted(required)}
    invalid = {step: count for step, count in counts.items() if count != local_processes}
    if invalid:
        raise ValueError(
            f"{path} requires {local_processes} records per step; "
            f"first={dict(list(invalid.items())[:8])}"
        )
    reduced = {
        step: {
            field: statistics.median(record[field] for record in by_step[step])
            for field in FIELDS
        }
        for step in sorted(required)
    }
    return reduced, counts


def build_node_summary(
    log: Path,
    *,
    run_id: str,
    route: str,
    node_rank: int,
    node_count: int,
    local_processes: int,
    final_step: int,
    steady_start: int,
    steady_end: int,
    source_sha256: str,
    route_contract_sha256: str,
    world_size: int,
    local_batch: int,
    gradient_accumulation: int,
    global_batch: int,
) -> dict[str, Any]:
    if node_rank not in range(node_count):
        raise ValueError("node_rank is outside node_count")
    if node_count * local_processes != world_size:
        raise ValueError("node topology does not produce world_size")
    if world_size * local_batch * gradient_accumulation != global_batch:
        raise ValueError("batch geometry does not produce global_batch")
    if not 1 <= steady_start <= steady_end <= final_step:
        raise ValueError("invalid measurement window")
    for name, value in (
        ("source_sha256", source_sha256),
        ("route_contract_sha256", route_contract_sha256),
    ):
        if HEX_SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    if not run_id or not route:
        raise ValueError("run_id and route must be non-empty")
    rows, counts = parse_node_log(log, final_step, local_processes)
    return {
        "schema": NODE_SCHEMA,
        "run_id": run_id,
        "route": route,
        "node_rank": node_rank,
        "node_count": node_count,
        "local_processes": local_processes,
        "final_step": final_step,
        "source_log_sha256": digest(log),
        "source_sha256": source_sha256,
        "route_contract_sha256": route_contract_sha256,
        "world_size": world_size,
        "local_batch": local_batch,
        "gradient_accumulation": gradient_accumulation,
        "global_batch": global_batch,
        "record_counts": {str(step): counts[step] for step in counts},
        "per_step": {str(step): row for step, row in rows.items()},
        "final_metric": rows[final_step],
        "steady_state": window_summary(rows, steady_start, steady_end),
    }


def _load_node(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != NODE_SCHEMA:
        raise ValueError(f"{path} is not a node-v3 receipt")
    return document, sha256(raw).hexdigest()


def aggregate_node_summaries(paths: list[Path]) -> dict[str, Any]:
    loaded = [_load_node(path) for path in paths]
    if not loaded:
        raise ValueError("at least one node summary is required")
    nodes = [item[0] for item in loaded]
    node_count = int(nodes[0]["node_count"])
    if len(nodes) != node_count:
        raise ValueError(f"expected {node_count} node summaries, got {len(nodes)}")
    ranks = [int(node["node_rank"]) for node in nodes]
    if sorted(ranks) != list(range(node_count)) or len(set(ranks)) != node_count:
        raise ValueError("node ranks are incomplete or duplicated")
    invariant_keys = (
        "run_id",
        "route",
        "node_count",
        "local_processes",
        "final_step",
        "source_sha256",
        "route_contract_sha256",
        "world_size",
        "local_batch",
        "gradient_accumulation",
        "global_batch",
    )
    for key in invariant_keys:
        values = {json.dumps(node[key], sort_keys=True) for node in nodes}
        if len(values) != 1:
            raise ValueError(f"cross-node invariant mismatch for {key}")
    final_step = int(nodes[0]["final_step"])
    required_steps = {str(step) for step in range(1, final_step + 1)}
    local_processes = int(nodes[0]["local_processes"])
    for node in nodes:
        if set(node["per_step"]) != required_steps:
            raise ValueError(f"node {node['node_rank']} step manifest is incomplete")
        if set(node["record_counts"]) != required_steps or any(
            int(count) != local_processes for count in node["record_counts"].values()
        ):
            raise ValueError(f"node {node['node_rank']} record proof is incomplete")
    rows = {
        step: {
            field: statistics.median(
                float(node["per_step"][str(step)][field]) for node in nodes
            )
            for field in FIELDS
        }
        for step in range(1, final_step + 1)
    }
    first = int(nodes[0]["steady_state"]["first_step"])
    last = int(nodes[0]["steady_state"]["last_step"])
    if any(
        int(node["steady_state"]["first_step"]) != first
        or int(node["steady_state"]["last_step"]) != last
        for node in nodes
    ):
        raise ValueError("cross-node measurement windows differ")
    return {
        "schema": AGGREGATE_SCHEMA,
        "run_id": nodes[0]["run_id"],
        "route": nodes[0]["route"],
        "source_sha256": nodes[0]["source_sha256"],
        "route_contract_sha256": nodes[0]["route_contract_sha256"],
        "node_count": node_count,
        "local_processes": local_processes,
        "world_size": nodes[0]["world_size"],
        "local_batch": nodes[0]["local_batch"],
        "gradient_accumulation": nodes[0]["gradient_accumulation"],
        "global_batch": nodes[0]["global_batch"],
        "final_step": final_step,
        "node_receipt_sha256": {
            str(node["node_rank"]): receipt_sha
            for node, (_, receipt_sha) in zip(nodes, loaded, strict=True)
        },
        "world_metric_records_per_step": node_count * local_processes,
        "per_step": {str(step): row for step, row in rows.items()},
        "final_metric": rows[final_step],
        "steady_state": window_summary(rows, first, last),
        "complete": True,
    }


def _write(document: dict[str, Any], output: Path) -> None:
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    node = commands.add_parser("node")
    node.add_argument("--log", type=Path, required=True)
    node.add_argument("--run-id", required=True)
    node.add_argument("--route", required=True)
    node.add_argument("--node-rank", type=int, required=True)
    node.add_argument("--node-count", type=int, required=True)
    node.add_argument("--local-processes", type=int, required=True)
    node.add_argument("--final-step", type=int, required=True)
    node.add_argument("--steady-start", type=int, required=True)
    node.add_argument("--steady-end", type=int, required=True)
    node.add_argument("--source-sha256", required=True)
    node.add_argument("--route-contract-sha256", required=True)
    node.add_argument("--world-size", type=int, required=True)
    node.add_argument("--local-batch", type=int, required=True)
    node.add_argument("--gradient-accumulation", type=int, required=True)
    node.add_argument("--global-batch", type=int, required=True)
    node.add_argument("--output", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--node-summary", action="append", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "node":
        document = build_node_summary(
            args.log,
            run_id=args.run_id,
            route=args.route,
            node_rank=args.node_rank,
            node_count=args.node_count,
            local_processes=args.local_processes,
            final_step=args.final_step,
            steady_start=args.steady_start,
            steady_end=args.steady_end,
            source_sha256=args.source_sha256,
            route_contract_sha256=args.route_contract_sha256,
            world_size=args.world_size,
            local_batch=args.local_batch,
            gradient_accumulation=args.gradient_accumulation,
            global_batch=args.global_batch,
        )
    else:
        document = aggregate_node_summaries(args.node_summary)
    _write(document, args.output)


if __name__ == "__main__":
    main()
