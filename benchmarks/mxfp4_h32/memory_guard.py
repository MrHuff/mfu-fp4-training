"""Monitor local GPU memory and optionally stop one explicit process group."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import time


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def query_used_mib(expected_local_gpus: int) -> list[tuple[int, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        index, used = (item.strip() for item in line.split(",", 1))
        rows.append((int(index), int(used)))
    expected = set(range(expected_local_gpus))
    if len(rows) != expected_local_gpus or {index for index, _ in rows} != expected:
        raise RuntimeError(f"expected local GPUs {sorted(expected)}, got {rows}")
    return rows


def terminate_process_group(pgid: int) -> None:
    if pgid <= 1:
        raise ValueError("refusing to signal process group <= 1")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def monitor(
    *,
    pgid: int,
    expected_local_gpus: int,
    cap_mib: int,
    mode: str,
    stop_file: Path,
    violation_file: Path,
    telemetry_file: Path,
    poll_seconds: float,
    grace_seconds: float,
) -> None:
    if expected_local_gpus < 1 or cap_mib < 1:
        raise ValueError("GPU count and memory cap must be positive")
    if mode not in {"enforce", "monitor"}:
        raise ValueError("mode must be enforce or monitor")
    if poll_seconds <= 0 or grace_seconds < 0:
        raise ValueError("invalid polling interval or grace period")
    if stop_file == violation_file or telemetry_file in {stop_file, violation_file}:
        raise ValueError("guard output paths must be distinct")

    telemetry_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    consecutive_errors = 0
    notice_emitted = False
    with telemetry_file.open("x", encoding="utf-8") as output:
        output.write("timestamp_utc\tgpu_index\tused_mib\tcap_mib\tmode\n")
        output.flush()
        while not stop_file.exists():
            try:
                rows = query_used_mib(expected_local_gpus)
                consecutive_errors = 0
            except Exception as error:
                consecutive_errors += 1
                output.write(
                    f"{utc_now()}\tquery_error\t{type(error).__name__}\t"
                    f"{cap_mib}\t{mode}\n"
                )
                output.flush()
                if consecutive_errors >= 3 and mode == "enforce":
                    violation_file.write_text(
                        f"reason=nvidia_smi_failed timestamp_utc={utc_now()}\n",
                        encoding="utf-8",
                    )
                    terminate_process_group(pgid)
                    raise RuntimeError("three consecutive nvidia-smi failures")
                time.sleep(poll_seconds)
                continue

            for index, used in rows:
                output.write(f"{utc_now()}\t{index}\t{used}\t{cap_mib}\t{mode}\n")
            output.flush()
            offenders = [(index, used) for index, used in rows if used > cap_mib]
            if offenders and time.monotonic() - started >= grace_seconds:
                detail = ",".join(f"gpu{index}={used}MiB" for index, used in offenders)
                if mode == "monitor":
                    if not notice_emitted:
                        print(f"memory cap exceeded (monitor only): {detail}", flush=True)
                        notice_emitted = True
                else:
                    violation_file.write_text(
                        f"reason=memory_cap_exceeded detail={detail} "
                        f"timestamp_utc={utc_now()}\n",
                        encoding="utf-8",
                    )
                    terminate_process_group(pgid)
                    raise RuntimeError(f"memory cap exceeded: {detail}")
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgid", type=int, required=True)
    parser.add_argument("--expected-local-gpus", type=int, required=True)
    parser.add_argument("--cap-mib", type=int, required=True)
    parser.add_argument("--mode", choices=("enforce", "monitor"), default="monitor")
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--violation-file", type=Path, required=True)
    parser.add_argument("--telemetry-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--grace-seconds", type=float, default=180.0)
    args = parser.parse_args()
    monitor(
        pgid=args.pgid,
        expected_local_gpus=args.expected_local_gpus,
        cap_mib=args.cap_mib,
        mode=args.mode,
        stop_file=args.stop_file,
        violation_file=args.violation_file,
        telemetry_file=args.telemetry_file,
        poll_seconds=args.poll_seconds,
        grace_seconds=args.grace_seconds,
    )


if __name__ == "__main__":
    main()
