#!/usr/bin/env python3
"""Check production MXFP4 and NVFP4 GEMMs against a BF16 reference.

This is intentionally a small, deterministic single-GPU release gate.  It
loads the six-extension runtime by exact path through the ABI checker, then
checks the two global-scaling GEMM families used by the supported routes.  It
does not claim to replace a distributed training replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import sysconfig
from typing import Any

import torch


SHAPE = 512
SEED = 20260903
MAX_REL_L2 = 0.25
MIN_COSINE = 0.96
MIN_NORM_RATIO = 0.80
MAX_NORM_RATIO = 1.20


def _load_extension(name: str, path: Path):
    if not path.is_file():
        raise RuntimeError(f"required extension is absent: {path.name}")
    previous = sys.modules.pop(name, None)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot construct extension loader for {path.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous
        raise


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    reference_norm = torch.linalg.vector_norm(reference_f)
    values = {
        "rel_l2": float(
            torch.linalg.vector_norm(actual_f - reference_f) / reference_norm
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f.flatten(), reference_f.flatten(), dim=0
            )
        ),
        "norm_ratio": float(torch.linalg.vector_norm(actual_f) / reference_norm),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise AssertionError(f"non-finite numerical metric: {values}")
    return values


def _assert_metrics(label: str, values: dict[str, float]) -> None:
    if values["rel_l2"] >= MAX_REL_L2:
        raise AssertionError(
            f"{label}: rel_l2={values['rel_l2']:.8f} is not below {MAX_REL_L2}"
        )
    if values["cosine"] <= MIN_COSINE:
        raise AssertionError(
            f"{label}: cosine={values['cosine']:.8f} is not above {MIN_COSINE}"
        )
    if not MIN_NORM_RATIO <= values["norm_ratio"] <= MAX_NORM_RATIO:
        raise AssertionError(
            f"{label}: norm_ratio={values['norm_ratio']:.8f} is outside "
            f"[{MIN_NORM_RATIO}, {MAX_NORM_RATIO}]"
        )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    # Keep the BF16 reference near unit output variance without making the
    # quantizers depend on a specially selected low-dynamic-range tensor.
    scale = SHAPE ** -0.25
    lhs = (
        torch.randn(
            (SHAPE, SHAPE), device="cuda", dtype=torch.float32, generator=generator
        )
        * scale
    ).to(torch.bfloat16).contiguous()
    rhs = (
        torch.randn(
            (SHAPE, SHAPE), device="cuda", dtype=torch.float32, generator=generator
        )
        * scale
    ).to(torch.bfloat16).contiguous()
    return lhs, rhs


def _mxfp4_gate(root: Path, suffix: str) -> dict[str, float]:
    quantizer = _load_extension(
        "mxfp4_quant_v4",
        root / f"TK_quantisation/mxfp4_v4/mxfp4_quant_v4{suffix}",
    )
    gemm = _load_extension(
        "_C_mx",
        root / f"ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx{suffix}",
    )
    lhs, rhs = _inputs()
    reference = lhs @ rhs.t()
    # This is the supported signed-H32 MXFP4-v4 path.  Stochastic rounding is
    # disabled here so the release gate is exactly repeatable; its advancing
    # state is covered by the adjacent correlated-SR behavioral gate.
    lhs_q = quantizer.mxfp4_quantize_for_gemm_opt_rht(
        lhs, 1, False, False, 32, True, 1234, 0
    )
    rhs_q = quantizer.mxfp4_quantize_for_gemm_opt_rht(
        rhs, 1, False, False, 32, True, 1234, 0
    )
    actual = torch.empty_like(reference)
    gemm.mxfp4_gemm(lhs_q[0], lhs_q[1], rhs_q[0], rhs_q[1], actual)
    torch.cuda.synchronize()
    values = _metrics(actual, reference)
    _assert_metrics("MXFP4-v4 signed-H32", values)
    return values


def _nvfp4_gate(root: Path, suffix: str) -> dict[str, float]:
    quantizer = _load_extension(
        "_tk_quant_v5",
        root / f"TK_quantisation/nvfp4_v5/_tk_quant_v5{suffix}",
    )
    gemm = _load_extension(
        "_C",
        root / f"ThunderKittens/kernels/gemm/nvfp4_b200/_C{suffix}",
    )
    lhs, rhs = _inputs()
    reference = lhs @ rhs.t()
    lhs_q = quantizer.tk_quantize_for_gemm(lhs, True)
    rhs_q = quantizer.tk_quantize_for_gemm(rhs, True)
    actual = torch.empty_like(reference)
    gemm.nvfp4_gemm(
        lhs_q[0], lhs_q[1], lhs_q[4], rhs_q[0], rhs_q[1], rhs_q[4], actual
    )
    torch.cuda.synchronize()
    values = _metrics(actual, reference)
    _assert_metrics("NVFP4-v5", values)
    return values


def _result(mx: dict[str, float], nv: dict[str, float]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass",
        "device_contract": "sm_100a",
        "shape": [SHAPE, SHAPE, SHAPE],
        "seed": SEED,
        "thresholds": {
            "rel_l2_strict_upper_bound": MAX_REL_L2,
            "cosine_strict_lower_bound": MIN_COSINE,
            "norm_ratio_closed_interval": [MIN_NORM_RATIO, MAX_NORM_RATIO],
        },
        "routes": {
            "mxfp4_v4_signed_h32": mx,
            "nvfp4_v5": nv,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--json", action="store_true", help="emit one machine-readable JSON object"
    )
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"expected CUDA 13.0, found {torch.version.cuda}")
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 0):
        raise RuntimeError(
            f"expected GB200/B200 compute capability (10, 0), found {capability}"
        )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension suffix is unavailable")

    mx = _mxfp4_gate(root, suffix)
    nv = _nvfp4_gate(root, suffix)
    result = _result(mx, nv)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for route, values in result["routes"].items():
            print(
                f"{route}: rel_l2={values['rel_l2']:.8f} "
                f"cosine={values['cosine']:.8f} "
                f"norm_ratio={values['norm_ratio']:.8f}"
            )
        print("MXFP4-v4/NVFP4-v5 BF16-reference numerical gates: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
