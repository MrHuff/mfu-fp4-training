#!/usr/bin/env python3
"""Compare 8B MXFP4 runner flags against the 1.2B high-water route."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


INTENTIONAL_8B_SWIGLU_DELTAS = {
    "FP4_CCE_ASSUME_NONEMPTY_LABELS",
    "MXFP4_USE_FUSED_SILU_FFN_QUANT",
    "MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN",
    "MXFP4_USE_SIMPLE_SQRELU_FUSED_W2",
    "MXFP4_USE_FUSED_SQRELU_QUANT",
    "MXFP4_USE_TMA_SQRELU_QUANT",
    "MXFP4_USE_FUSED_SQRELU_DERIV_QUANT",
    "MXFP4_USE_SQRELU_FUSED_RMS_W1",
    "MXFP4_USE_SQRELU_SPLIT_COL_OVERLAP",
    "MXFP4_USE_SQRELU_SPLIT_COL_QUANT",
    "MXFP4_USE_SQRELU_SPLIT_COL_WAIT_FORWARD",
    "MXFP4_USE_SQRELU_W2_WGRAD_OVERLAP",
    "MXFP4_USE_SQRELU_W2_WGRAD_AFTER_DGRAD_OVERLAP",
    "MXFP4_USE_SQRELU_DERIV_GEMM_EPILOGUE",
}

EIGHT_B_EXTRA_HIGHWATER_FIXES = {
    "MXFP4_USE_QKV_COMBINED_BWD",
    "MXFP4_USE_SPLIT3_QKV_STAGE_COPY",
}


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    nvpaper = _load_module("tools/run_nvpaper_1p2b_numerics_500.py", "nvpaper_env_audit")
    nvblog = _load_module("tools/run_nvblog_llama3_8b_matrix.py", "nvblog_env_audit")
    ref = nvpaper._mxfp4_highwater_env()
    cand = nvblog._mxfp4_swiglu_env()

    keys = sorted(
        k
        for k in (set(ref) | set(cand))
        if k.startswith("MXFP4") or k.startswith("FP4_CCE") or k.startswith("FP4_M")
    )
    unexpected: list[str] = []
    print("MXFP4 8B vs 1.2B high-water env audit")
    for key in keys:
        if key in EIGHT_B_EXTRA_HIGHWATER_FIXES and key in cand:
            print(f"extra-8b-fix {key}={cand[key]}")
            continue
        if key in INTENTIONAL_8B_SWIGLU_DELTAS:
            if key in ref and key in cand and str(ref[key]) != str(cand[key]):
                print(f"intentional-swiglu {key}: 1.2B={ref[key]} 8B={cand[key]}")
            elif key in ref and key not in cand:
                print(f"intentional-swiglu absent-8b {key}=1.2B:{ref[key]}")
            elif key in cand and key not in ref:
                print(f"intentional-swiglu extra-8b {key}={cand[key]}")
            continue
        if key not in cand:
            unexpected.append(f"missing-8b {key}=1.2B:{ref[key]}")
        elif key not in ref:
            unexpected.append(f"extra-8b {key}={cand[key]}")
        elif str(ref[key]) != str(cand[key]):
            unexpected.append(f"different {key}: 1.2B={ref[key]} 8B={cand[key]}")

    if unexpected:
        print("\nUnexpected deltas:")
        for item in unexpected:
            print(f"  {item}")
        return 1
    print("\nNo unexpected MXFP4 high-water env deltas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
