#!/usr/bin/env python3
"""Run held-out validation endpoint jobs for the NVIDIA 1.2B matrix.

Periodic in-loop validation can leave some FP4 kernels in a bad post-validation
state. This helper builds validation curves by running independent jobs that
validate on the final step, so each point uses held-out validation without
resuming training afterward.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools" / "run_nvpaper_1p2b_numerics_500.py"
DEFAULT_TRAIN_DATA = Path("/tmp/lbt_packed/c4_train_67m_tokens_20260529.bin")
DEFAULT_VALIDATION_DATA = Path("/tmp/lbt_packed/c4_validation_heldout_16m_20260603")


@dataclass(frozen=True)
class Endpoint:
    case: str
    step: int
    out_dir: Path
    gpu: str = ""


def _same_resolved_path(lhs: Path, rhs: Path) -> bool:
    try:
        return lhs.expanduser().resolve() == rhs.expanduser().resolve()
    except OSError:
        return lhs.expanduser().absolute() == rhs.expanduser().absolute()


def _parse_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_endpoint(
    endpoint: Endpoint,
    args: argparse.Namespace,
) -> dict[str, object]:
    endpoint.out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = endpoint.out_dir / "runner.log"
    env = os.environ.copy()
    env["LBT_VALIDATION_KEEP_TRAIN_MODE"] = "1"
    env["LBT_VALIDATION_SKIP_STEP1"] = "1"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--gpu",
        endpoint.gpu,
        "--steps",
        str(endpoint.step),
        "--log-freq",
        str(args.log_freq),
        "--steady-from",
        str(min(args.steady_from, endpoint.step)),
        "--cases",
        endpoint.case,
        "--out-base",
        str(endpoint.out_dir),
        "--dataset",
        "packed-bin",
        "--dataset-path",
        str(args.train_data),
        "--load-dataset-kwargs",
        args.load_dataset_kwargs,
        "--validation-enable",
        "--validation-dataset",
        "packed-bin",
        "--validation-dataset-path",
        str(args.validation_data),
        "--validation-local-batch-size",
        str(args.validation_local_batch_size),
        "--validation-seq-len",
        str(args.validation_seq_len),
        "--validation-freq",
        str(endpoint.step),
        "--validation-steps",
        str(args.validation_steps),
        "--validation-load-dataset-kwargs",
        args.validation_load_dataset_kwargs,
    ]
    if args.seed is None:
        cmd.append("--no-debug-seed")
    else:
        cmd.extend(["--seed", str(args.seed)])
    with stdout_path.open("w", encoding="utf-8") as stdout:
        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    summary_rows = _parse_csv(endpoint.out_dir / "summary.csv")
    validation_rows = _parse_csv(endpoint.out_dir / endpoint.case / "validation.csv")
    final_validation = next(
        (row for row in validation_rows if int(row["step"]) == endpoint.step),
        validation_rows[-1] if validation_rows else None,
    )
    summary = summary_rows[0] if summary_rows else {}
    return {
        "case": endpoint.case,
        "step": endpoint.step,
        "gpu": endpoint.gpu,
        "returncode": proc.returncode,
        "completed": summary.get("completed", "False"),
        "train_loss": summary.get("last_loss", ""),
        "validation_loss": final_validation.get("loss", "") if final_validation else "",
        "validation_tps": final_validation.get("tps", "") if final_validation else "",
        "peak_mfu": summary.get("peak_mfu", ""),
        "steady_mfu": summary.get("steady_mfu", ""),
        "wall_s": f"{time.time() - start:.3f}",
        "train_log": str(endpoint.out_dir / endpoint.case / "train.log"),
        "runner_log": str(stdout_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["mxfp4_highwater", "tk_v5_noextras", "localcta_v4_noextras"],
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--out-base", type=Path, default=None)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--validation-data", type=Path, default=DEFAULT_VALIDATION_DATA)
    parser.add_argument("--validation-local-batch-size", type=int, default=4)
    parser.add_argument("--validation-seq-len", type=int, default=8192)
    parser.add_argument("--validation-steps", type=int, default=2)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--steady-from", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--load-dataset-kwargs",
        default='{"num_workers":8,"prefetch_factor":4,"pin_memory":false,"repeat":false,"require_full_run":true}',
    )
    parser.add_argument(
        "--validation-load-dataset-kwargs",
        default='{"num_workers":0,"pin_memory":false,"repeat":false}',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.train_data.exists():
        raise SystemExit(f"Training packed dataset not found: {args.train_data}")
    if not args.validation_data.exists():
        raise SystemExit(f"Validation packed dataset not found: {args.validation_data}")
    if _same_resolved_path(args.train_data, args.validation_data):
        raise SystemExit(
            "Validation dataset must be held out: --validation-data matches --train-data."
        )
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id")
    max_parallel = args.max_parallel or len(gpus)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = args.out_base or Path(f"/tmp/lbt_1p2b_validation_endpoints_{stamp}")
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    pending = [
        Endpoint(
            case=case,
            step=step,
            out_dir=out_base / case / f"step_{step:04d}",
        )
        for case in args.cases
        for step in args.steps
    ]
    rows: list[dict[str, object]] = []
    active: dict[Future[dict[str, object]], Endpoint] = {}
    active_gpus: dict[Future[dict[str, object]], str] = {}
    free_gpus = gpus.copy()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        while pending or active:
            while pending and free_gpus and len(active) < max_parallel:
                pending_endpoint = pending.pop(0)
                gpu = free_gpus.pop(0)
                endpoint = Endpoint(
                    case=pending_endpoint.case,
                    step=pending_endpoint.step,
                    gpu=gpu,
                    out_dir=pending_endpoint.out_dir,
                )
                future = pool.submit(_run_endpoint, endpoint, args)
                active[future] = endpoint
                active_gpus[future] = gpu
                print(f"start {endpoint.case} step={endpoint.step} gpu={gpu}", flush=True)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                endpoint = active.pop(future)
                free_gpus.append(active_gpus.pop(future))
                row = future.result()
                rows.append(row)
                print(
                    f"done {endpoint.case} step={endpoint.step} rc={row['returncode']} "
                    f"val={row['validation_loss']}",
                    flush=True,
                )
                _write_summary(out_base, rows)
                if int(row["returncode"]) != 0:
                    raise SystemExit(int(row["returncode"]))
    _write_summary(out_base, rows)
    print(f"summary: {out_base / 'validation_endpoints.csv'}")
    return 0


def _write_summary(out_base: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "case",
        "step",
        "gpu",
        "returncode",
        "completed",
        "train_loss",
        "validation_loss",
        "validation_tps",
        "peak_mfu",
        "steady_mfu",
        "wall_s",
        "train_log",
        "runner_log",
    ]
    with (out_base / "validation_endpoints.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (str(item["case"]), int(item["step"]))):
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
