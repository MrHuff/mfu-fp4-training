#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_fp4_training_matrix import Case, _config, run_track, write_json


# These cases are the remaining 1B_legacy follow-up items from
# docs/fp4_mxfp4_matrix_status_2026_04_11.md.
#
# The env additions here are intentionally narrow:
# - plain TK attention uses the recovered TE-quant / TK-GEMM contract
# - localCTA attention uses the prepared-activation contract
#
# They keep the TK/localCTA GEMM paths active while avoiding the stale QKV
# activation payloads that the runtime comments still call out as numerics gaps.


def _env(*, tk: bool, localcta: bool, localcta_fused: bool, qkv_recovery: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "USE_TK_GEMM": "1" if tk else "0",
        "USE_TK_LOCALCTA": "1" if localcta else "0",
        "USE_TK_LOCALCTA_FUSED": "1" if localcta_fused else "0",
        "USE_MXFP4_TK_FUSED": "0",
        "USE_MXFP4_TK_BACKEND": "0",
    }
    if qkv_recovery:
        env.update(qkv_recovery)
    return env


TK_QKV_RECOVERY_ENV = {
    "USE_TK_QKV_TE_ACT_QUANT": "1",
}

LOCALCTA_QKV_RECOVERY_ENV = {
    "USE_TK_QKV_LOCALCTA_TK_PREPARED_ACT": "1",
}


MFU_LEGACY_FOLLOWUP_CASES = [
    Case(
        name="fp4_fused_ref_legacy",
        family="mfu_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(tk=False, localcta=False, localcta_fused=False),
        overrides=[],
    ),
    Case(
        name="fp4_tk_legacy",
        family="mfu_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(
            tk=True,
            localcta=False,
            localcta_fused=False,
            qkv_recovery=TK_QKV_RECOVERY_ENV,
        ),
        overrides=[],
    ),
    Case(
        name="localcta_legacy",
        family="mfu_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(
            tk=True,
            localcta=True,
            localcta_fused=False,
            qkv_recovery=LOCALCTA_QKV_RECOVERY_ENV,
        ),
        overrides=[],
    ),
    Case(
        name="localcta_fused_legacy",
        family="mfu_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(
            tk=True,
            localcta=True,
            localcta_fused=True,
            qkv_recovery=LOCALCTA_QKV_RECOVERY_ENV,
        ),
        overrides=[],
    ),
]


CONVERGENCE_LEGACY_FOLLOWUP_CASES = [
    Case(
        name="te_fp4_legacy",
        family="convergence_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_te_fp4_matrix.toml"),
        env=_env(tk=False, localcta=False, localcta_fused=False),
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="fp4_fused_ref_legacy",
        family="convergence_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(tk=False, localcta=False, localcta_fused=False),
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="fp4_tk_legacy",
        family="convergence_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(
            tk=True,
            localcta=False,
            localcta_fused=False,
            qkv_recovery=TK_QKV_RECOVERY_ENV,
        ),
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="localcta_legacy",
        family="convergence_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(
            tk=True,
            localcta=True,
            localcta_fused=False,
            qkv_recovery=LOCALCTA_QKV_RECOVERY_ENV,
        ),
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="localcta_fused_legacy",
        family="convergence_legacy_followup",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env=_env(
            tk=True,
            localcta=True,
            localcta_fused=True,
            qkv_recovery=LOCALCTA_QKV_RECOVERY_ENV,
        ),
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
]


TRACKS = {
    "mfu_legacy_followup": MFU_LEGACY_FOLLOWUP_CASES,
    "convergence_legacy_followup": CONVERGENCE_LEGACY_FOLLOWUP_CASES,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["mfu_legacy_followup", "convergence_legacy_followup"],
        choices=list(TRACKS.keys()),
    )
    parser.add_argument(
        "--out-root",
        default=f"/tmp/fp4_tk_localcta_followup_gpu3_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_root": str(REPO_ROOT),
        "out_root": str(out_root),
        "tracks": args.tracks,
        "notes": {
            "source_status_doc": "docs/fp4_mxfp4_matrix_status_2026_04_11.md",
            "tk_qkv_recovery_env": TK_QKV_RECOVERY_ENV,
            "localcta_qkv_recovery_env": LOCALCTA_QKV_RECOVERY_ENV,
        },
    }
    write_json(out_root / "manifest.json", manifest)

    all_results: dict[str, object] = {}
    for track in args.tracks:
        all_results[track] = run_track(track, TRACKS[track], out_root)
    write_json(out_root / "all_results.json", all_results)
    print(f"Results written to {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
