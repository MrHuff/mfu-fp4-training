#!/usr/bin/env python3
"""Gate the fused MXFP4+RHT/localCTA-dgrad carrier on one local GPU.

The producer-only gate can run as soon as the localCTA-v4 extension containing
the mixed ABI has been built.  It compares every fused payload byte against an
independent production quantizer:

* localCTA row-SR at the exact explicit Philox coordinate;
* MXFP4 column-RHT (H32 with the fixed 0x2817 sign motif, no data/scale SR);
* MXFP4 exact 2-D weight row; and
* localCTA exact 2-D weight column and outer scale.

The FFN split2 gate separately rejects the obsolete shared-outer-scale ABI. It
requires two independently finalized localCTA row outer scales, compares both
arms byte-for-byte to a zero-padded ordinary localCTA oracle at the fused
global coordinates, proves one logical SR reservation, and executes the
established one-pass dgrad consumer.

When the source integration is present, the full gate additionally executes a
real MXFP4 Wo forward/backward with the feature off and on.  Forward and
dWeight are zero-tolerance bit-exact requirements.  dHidden is compared to the
separate localCTA oracle using the explicit CLI tolerance (zero by default).
Logical GEMM accounting requires exactly one forward, one dHidden, and one
dWeight GEMM in both arms.

No cluster interaction occurs.  The output is an exclusively-created compact
JSON receipt sealed over its canonical payload with SHA-256.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

# Keep package import policy inert until the exact recipe environment is set.
os.environ.setdefault("LBT_LIGHT_IMPORT", "1")
os.environ.setdefault("LBT_QUANTIZATION_LIGHT_IMPORT", "1")

import torch

# The workstation may have an older LBT checkout on PYTHONPATH.  This gate is
# evidence for the checkout that owns the script, so resolve it first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_GATE_PATH = REPO_ROOT / "low_bits_training" / "analysis" / "mixed_carrier_gate.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "_lbt_mixed_carrier_gate", _GATE_PATH
)
if _GATE_SPEC is None or _GATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load gate contracts from {_GATE_PATH}")
gate = importlib.util.module_from_spec(_GATE_SPEC)
sys.modules[_GATE_SPEC.name] = gate
_GATE_SPEC.loader.exec_module(gate)

REQUIRED_SOURCE_COMMIT = "5011942c018b26a187395b15d199271adb209f5f"
REQUIRED_RUNTIME_COMMIT = "301ab63d354a4f8c24b7c0da499736e3f14b7400"


REQUIRED_RUNTIME_CAPABILITIES = {
    "abi_version": 1,
    "grad_coordinate_mode": "explicit_seed_subsequence",
    "grad_mx_col_rht": "block32_fixed_0x2817",
    "mxfp4_rht_block_size": 32,
    "mxfp4_rht_sign_contract": "fixed_0x2817_per_h16_half",
    "grad_localcta_row_sr": True,
    "grad_scale_sr": False,
    "weight_mx_2d": True,
    "weight_localcta_2d": True,
    "prepared_outer_sg": True,
    "localcta_encode_mode": "encode_centric",
    "localcta_sg_contract": "outer",
    "min_alignment": 256,
    "single_bf16_tile_load": True,
    "runtime_advances_rng": False,
    "split2_grad_one_coordinate": True,
    "split2_dgrad_onepass_outer_sg": True,
    "split2_row_outer_sg": "per_arm",
    "split2_layout": (
        "logical_dim1_concat_per_arm_outer_no_bf16_materialization"
    ),
}


class FeatureUnavailable(gate.GateFailure):
    """The post-fix source or runtime ABI has not been installed yet."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        required=True,
        help="fp4_matmul tree containing the newly built mixed localCTA-v4 ABI",
    )
    parser.add_argument("--output", required=True, help="new sealed receipt path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsequence", type=int, default=17)
    parser.add_argument("--tensor-seed", type=int, default=20260831)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--producer-only",
        action="store_true",
        help="stop after payload correctness, SR, lifetime, and producer timing",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help=(
            "measure producer speed at large production shapes without rerunning "
            "the correctness oracles; requires --producer-only and is never a "
            "correctness receipt"
        ),
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="run correctness only (receipt explicitly records this)",
    )
    parser.add_argument("--dhidden-atol", type=float, default=0.0)
    parser.add_argument("--dhidden-rtol", type=float, default=0.0)
    parser.add_argument(
        "--max-full-overhead-pct",
        type=float,
        default=3.0,
        help="fail when candidate median full Wo forward+backward overhead exceeds this",
    )
    parser.add_argument("--localcta-scale-num", type=float, default=448.0)
    return parser.parse_args()


def _set_exact_policy(args: argparse.Namespace) -> None:
    runtime_root = str(Path(args.runtime_root).resolve())
    values = {
        "FP4_MATMUL_ROOT": runtime_root,
        "FP4_MATMUL_GEMM_ROOT": runtime_root,
        "FP4_MXFP4_ROOT": runtime_root,
        "FP4_CCE_TK_ROOT": runtime_root,
        "MXFP4_BACKEND_VERSION": "v4",
        "USE_TK_GEMM": "1",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_VARIANT": "v4",
        "MXFP4_USE_LOCALCTA_DGRAD": "1",
        "MXFP4_USE_2D_WEIGHT_QUANT": "1",
        "MXFP4_USE_RHT": "1",
        "MXFP4_RHT_TE_STYLE": "1",
        "MXFP4_RHT_ACTIVATION": "1",
        "MXFP4_RHT_GRAD": "1",
        "MXFP4_RHT_WEIGHT": "0",
        "MXFP4_RHT_AXES": "col",
        "MXFP4_RHT_BLOCK_SIZE": "32",
        "MXFP4_RHT_RANDOM_SIGN_MASK": "1",
        "MXFP4_USE_STOCHASTIC_ROUNDING": "1",
        "MXFP4_SR_ACTIVATION": "0",
        "MXFP4_SR_GRAD": "1",
        "MXFP4_GRAD_SR_AXES": "row",
        "MXFP4_SR_WEIGHT": "0",
        "MXFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "MXFP4_SCALE_SR_ACTIVATION": "0",
        "MXFP4_SCALE_SR_GRAD": "0",
        "MXFP4_SCALE_SR_WEIGHT": "0",
        "MXFP4_SR_SEED": str(args.seed),
        "MXFP4_SR_SUBSEQUENCE": str(args.subsequence),
        "MXFP4_USE_WEIGHT_QUANT_CACHE": "0",
        "NVTE_NVFP4_ENCODE_CENTRIC": "1",
    }
    os.environ.update(values)


def _validate_args(args: argparse.Namespace) -> None:
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("M, N, and K must be positive")
    if any(value % 256 for value in (args.m, args.n, args.k)):
        raise ValueError("mixed producer gate requires M, N, and K divisible by 256")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    if args.dhidden_atol < 0 or args.dhidden_rtol < 0:
        raise ValueError("dHidden tolerances must be non-negative")
    if args.timing_only and not args.producer_only:
        raise ValueError("--timing-only requires --producer-only")
    if args.timing_only and args.skip_timing:
        raise ValueError("--timing-only cannot be combined with --skip-timing")


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_is_ancestor(path: Path, ancestor: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "merge-base", "--is-ancestor", ancestor, "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_signed_int64(value: int) -> int:
    value = int(value) & ((1 << 64) - 1)
    return value if value < (1 << 63) else value - (1 << 64)


def _as_uint64(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def _load_runtime_module(args: argparse.Namespace):
    runtime_root = Path(args.runtime_root).resolve()
    extension_dir = runtime_root / "TK_quantisation" / "nvfp4_CTA_local_v4"
    if not extension_dir.is_dir():
        raise FeatureUnavailable(f"localCTA-v4 extension directory missing: {extension_dir}")
    candidates = sorted(extension_dir.glob("_tk_quant_localcta_v4*.so"))
    if len(candidates) != 1:
        raise FeatureUnavailable(
            "expected exactly one built mixed runtime extension under the explicit "
            f"--runtime-root, found {[str(path) for path in candidates]}"
        )
    module_path = candidates[0].resolve()
    module_name = "_tk_quant_localcta_v4"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if loaded_path != module_path:
            raise gate.GateFailure(
                "mixed runtime module was already loaded from a different root: "
                f"loaded={loaded_path}, required={module_path}"
            )
        module = loaded
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise FeatureUnavailable(f"cannot load mixed runtime extension {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    query = getattr(module, "tk_mixed_mx_localcta_capabilities", None)
    if query is None:
        raise FeatureUnavailable("runtime lacks tk_mixed_mx_localcta_capabilities")
    capabilities = query()
    if not isinstance(capabilities, dict):
        raise FeatureUnavailable("mixed runtime capability query did not return a dict")
    mismatch = {
        name: {"actual": capabilities.get(name), "expected": expected}
        for name, expected in REQUIRED_RUNTIME_CAPABILITIES.items()
        if capabilities.get(name) != expected
    }
    if mismatch:
        raise FeatureUnavailable(f"mixed runtime capability mismatch: {mismatch}")
    setter = getattr(module, "tk_localcta_set_global_scale_num", None)
    if setter is not None:
        setter(float(args.localcta_scale_num))
    return module, module_path, capabilities


def _require_six_tensor_allocator(
    module,
    *,
    name: str,
    rows: int,
    cols: int,
    device: torch.device,
) -> dict[str, Any]:
    allocator = getattr(module, name, None)
    if allocator is None:
        raise FeatureUnavailable(f"runtime lacks {name}")
    result = allocator(int(rows), int(cols), device)
    if not isinstance(result, (tuple, list)) or len(result) != 6:
        raise gate.GateFailure(f"{name} must return exactly six tensors")
    if not all(torch.is_tensor(value) for value in result):
        raise gate.GateFailure(f"{name} returned a non-tensor payload")
    report = {
        "symbol": name,
        "count": len(result),
        "tensors": [
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "contiguous": bool(value.is_contiguous()),
            }
            for value in result
        ],
    }
    del result
    return report


def _require_seven_tensor_split2_allocator(
    module,
    *,
    rows: int,
    arm_width: int,
    device: torch.device,
) -> dict[str, Any]:
    """Seal the corrected split2 ABI, including two independent outer SGs."""

    name = "tk_mixed_split2_grad_localcta_row_mx_col_alloc"
    allocator = getattr(module, name, None)
    if allocator is None:
        raise FeatureUnavailable(f"runtime lacks {name}")
    result = tuple(allocator(int(rows), int(arm_width), device))
    if len(result) != 7 or not all(torch.is_tensor(value) for value in result):
        raise gate.GateFailure(f"{name} must return exactly seven tensors")
    expected = (
        ((rows, arm_width), torch.float4_e2m1fn_x2),
        ((rows // 128, arm_width // 32, 512), torch.float8_e4m3fn),
        ((rows // 256, 1), torch.float32),
        ((rows // 256, 1), torch.float32),
        ((2 * arm_width, rows // 2), torch.float4_e2m1fn_x2),
        ((arm_width // 64, rows // 128, 32, 16), torch.uint8),
        ((rows // 128, arm_width // 64), torch.float32),
    )
    mismatches = []
    for index, (value, (shape, dtype)) in enumerate(zip(result, expected)):
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            mismatches.append(
                {
                    "index": index,
                    "actual_shape": list(value.shape),
                    "expected_shape": list(shape),
                    "actual_dtype": str(value.dtype),
                    "expected_dtype": str(dtype),
                    "contiguous": bool(value.is_contiguous()),
                }
            )
    if mismatches:
        raise gate.GateFailure(f"{name} shape/dtype ABI mismatch: {mismatches}")
    distinct = gate.require_distinct_scale_carriers(
        result[2],
        result[3],
        left_name="split2_alloc_sg0",
        right_name="split2_alloc_sg1",
        require_distinct_values=False,
    )
    report = {
        "symbol": name,
        "count": 7,
        "row_outer_sg": "per_arm",
        "outer_sg_storage": distinct,
        "tensors": [
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "contiguous": bool(value.is_contiguous()),
            }
            for value in result
        ],
    }
    del result
    return report


def _candidate_grad(
    module,
    value: torch.Tensor,
    coordinate: tuple[int, int],
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    allocator = getattr(module, "tk_mixed_grad_localcta_row_mx_col_alloc", None)
    launch = getattr(
        module, "tk_mixed_grad_localcta_row_mx_col_launch_inplace", None
    )
    if allocator is None or launch is None:
        raise FeatureUnavailable("mixed gradient alloc/launch ABI is incomplete")
    buffers = tuple(allocator(int(value.shape[0]), int(value.shape[1]), value.device))
    if len(buffers) != 6 or not all(torch.is_tensor(item) for item in buffers):
        raise gate.GateFailure("mixed gradient allocator violated six-tensor ABI")
    launch(value, *buffers, int(coordinate[0]), int(coordinate[1]))
    return {
        "local_row_fp4": buffers[0],
        "local_row_sc": buffers[1],
        "local_row_sg": buffers[2],
        "mx_col_fp4": buffers[3],
        "mx_col_sc": buffers[4],
        "keepalive": tuple(buffers[5:]),
    }


def _candidate_split2_grad(
    module,
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    coordinate: tuple[int, int],
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    allocator = getattr(
        module, "tk_mixed_split2_grad_localcta_row_mx_col_alloc", None
    )
    launch = getattr(
        module, "tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace", None
    )
    if allocator is None or launch is None:
        raise FeatureUnavailable("mixed split2 gradient alloc/launch ABI is incomplete")
    if grad0.shape != grad1.shape:
        raise gate.GateFailure("split2 diagnostic arms must have identical shapes")
    buffers = tuple(
        allocator(int(grad0.shape[0]), int(grad0.shape[1]), grad0.device)
    )
    if len(buffers) != 7 or not all(torch.is_tensor(item) for item in buffers):
        raise gate.GateFailure("mixed split2 allocator violated seven-tensor ABI")
    launch(grad0, grad1, *buffers, int(coordinate[0]), int(coordinate[1]))
    return {
        "local_row_fp4": buffers[0],
        "local_row_sc": buffers[1],
        "local_row_sg0": buffers[2],
        "local_row_sg1": buffers[3],
        "mx_col_fp4": buffers[4],
        "mx_col_sc": buffers[5],
        "keepalive": tuple(buffers[6:]),
    }


def _split2_local_arms(
    payload: Mapping[str, Any],
    *,
    arm_width: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Materialize comparison-only arm views without invoking FP4 ``copy_``."""

    packed = payload["local_row_fp4"]
    scales = payload["local_row_sc"]
    if packed.dtype != torch.float4_e2m1fn_x2:
        raise gate.GateFailure(f"split2 packed carrier has wrong dtype: {packed.dtype}")
    packed_arm = arm_width // 2
    scale_arm = arm_width // 64
    if packed.shape[1] != arm_width or scales.shape[1] != 2 * scale_arm:
        raise gate.GateFailure(
            "split2 combined localCTA layout mismatch: "
            f"packed={tuple(packed.shape)} scales={tuple(scales.shape)} "
            f"arm_width={arm_width}"
        )

    def packed_slice(offset: int) -> torch.Tensor:
        return (
            packed.view(torch.uint8)
            .narrow(1, offset, packed_arm)
            .contiguous()
            .view(torch.float4_e2m1fn_x2)
        )

    arm0 = {
        "local_row_fp4": packed_slice(0),
        "local_row_sc": scales.narrow(1, 0, scale_arm).contiguous(),
        "local_row_sg": payload["local_row_sg0"],
    }
    arm1 = {
        "local_row_fp4": packed_slice(packed_arm),
        "local_row_sc": scales.narrow(1, scale_arm, scale_arm).contiguous(),
        "local_row_sg": payload["local_row_sg1"],
    }
    return arm0, arm1


def _zero_padded_localcta_split2_oracle(
    module,
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    coordinate: tuple[int, int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Use ordinary one-input no-RHT row-SR at true split2 coordinates.

    Each arm is placed in its true half of a zero-padded logical
    ``[grad0|grad1]`` matrix.  The padding preserves the fused global chunk/RNG
    coordinates while making the outer-scale reduction depend on only that
    arm.  This diagnostic materialization calls neither a split2 producer nor
    the invalid shared-SG column-RHT derivative path.
    """

    if grad0.shape != grad1.shape:
        raise gate.GateFailure("split2 oracle arms must have identical shapes")
    rows, arm_width = map(int, grad0.shape)
    arms: list[dict[str, torch.Tensor]] = []
    sr_reports: list[dict[str, Any]] = []
    for arm_index, value in enumerate((grad0, grad1)):
        logical = torch.zeros(
            (rows, 2 * arm_width), device=value.device, dtype=value.dtype
        )
        logical.narrow(1, arm_index * arm_width, arm_width).copy_(value)
        full, sr_report = _localcta_grad_oracle(module, logical, coordinate)
        packed_offset = arm_index * (arm_width // 2)
        scale_offset = arm_index * (arm_width // 64)
        arms.append(
            {
                "local_row_fp4": (
                    full["local_row_fp4"]
                    .view(torch.uint8)
                    .narrow(1, packed_offset, arm_width // 2)
                    .contiguous()
                    .view(torch.float4_e2m1fn_x2)
                ),
                "local_row_sc": (
                    full["local_row_sc"]
                    .narrow(1, scale_offset, arm_width // 64)
                    .contiguous()
                ),
                "local_row_sg": full["local_row_sg"],
            }
        )
        sr_reports.append(sr_report)
    torch.cuda.synchronize(grad0.device)
    return (
        arms[0],
        arms[1],
        {
            "pass": True,
            "oracle": "ordinary_localcta_row_sr_zero_padded_logical_coordinate",
            "logical_layout": "dim1_concat_arm0_then_arm1",
            "diagnostic_bf16_materialization": True,
            "production_bf16_materialization": False,
            "rht_axes": "none",
            "data_sr_axes": "row",
            "scale_sr": False,
            "oracle_state_advances": sr_reports,
            "mutates_ranked_production_state": False,
        },
    )


def _combine_split2_local_oracle(
    arm0: Mapping[str, torch.Tensor],
    arm1: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_bytes = torch.cat(
        (
            arm0["local_row_fp4"].view(torch.uint8),
            arm1["local_row_fp4"].view(torch.uint8),
        ),
        dim=1,
    ).contiguous()
    packed = packed_bytes.view(torch.float4_e2m1fn_x2)
    scales = torch.cat(
        (arm0["local_row_sc"], arm1["local_row_sc"]), dim=1
    ).contiguous()
    return packed, scales


def _candidate_weight(
    module,
    value: torch.Tensor,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    allocator = getattr(module, "tk_mixed_weight_mx_row_localcta_col_alloc", None)
    launch = getattr(
        module, "tk_mixed_weight_mx_row_localcta_col_launch_inplace", None
    )
    if allocator is None or launch is None:
        raise FeatureUnavailable("mixed weight alloc/launch ABI is incomplete")
    buffers = tuple(allocator(int(value.shape[0]), int(value.shape[1]), value.device))
    if len(buffers) != 6 or not all(torch.is_tensor(item) for item in buffers):
        raise gate.GateFailure("mixed weight allocator violated six-tensor ABI")
    launch(value, *buffers)
    return {
        "mx_row_fp4": buffers[0],
        "mx_row_sc": buffers[1],
        "local_col_fp4": buffers[2],
        "local_col_sc": buffers[3],
        "local_col_sg": buffers[4],
        "keepalive": tuple(buffers[5:]),
    }


def _localcta_grad_oracle(
    module,
    value: torch.Tensor,
    coordinate: tuple[int, int],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    quantize = getattr(module, "tk_localcta_quantize_for_gemm_opt", None)
    if quantize is None:
        raise FeatureUnavailable("runtime lacks localCTA opt oracle")
    state = torch.tensor(
        [_as_signed_int64(coordinate[0]), _as_signed_int64(coordinate[1])],
        dtype=torch.int64,
        device=value.device,
    )
    result = quantize(
        value,
        True,
        True,
        True,
        False,
        "none",
        False,
        int(coordinate[0]),
        int(coordinate[1]),
        "row",
        state,
    )
    if not isinstance(result, (tuple, list)) or len(result) < 6:
        raise gate.GateFailure("localCTA opt oracle returned an incomplete carrier")
    torch.cuda.synchronize(value.device)
    advanced = tuple(_as_uint64(item) for item in state.cpu().tolist())
    sr_report = gate.require_one_sr_advance(coordinate, advanced)
    return (
        {
            "local_row_fp4": result[0],
            "local_row_sc": result[1],
            "local_row_sg": result[4],
        },
        sr_report,
    )


def _localcta_weight_oracle(
    module,
    value: torch.Tensor,
) -> dict[str, torch.Tensor]:
    quantize = getattr(module, "tk_localcta_quantize_weight_2d", None)
    if quantize is None:
        raise FeatureUnavailable("runtime lacks localCTA exact-2D weight oracle")
    result = quantize(value)
    if not isinstance(result, (tuple, list)) or len(result) < 6:
        raise gate.GateFailure("localCTA weight oracle returned an incomplete carrier")
    return {
        "local_col_fp4": result[2],
        "local_col_sc": result[3],
        "local_col_sg": result[5],
    }


def _load_mx_backend(runtime_root: Path):
    from low_bits_training.quantization import mxfp4_backend as backend

    required = (
        "mxfp4_quantize_for_gemm_opt",
        "mxfp4_quantize_col_only_opt_rht",
        "mxfp4_quantize_weight_2d",
        "mxfp4_quantize_split2_row_only_opt_launch_inplace",
        "mxfp4_quantize_split2_col_only_opt_launch_inplace",
    )
    missing = [name for name in required if not hasattr(backend, name)]
    if missing:
        raise FeatureUnavailable(f"MXFP4 backend lacks oracle functions: {missing}")
    quant_module = backend._load_quant_module("v4")
    module_path = Path(getattr(quant_module, "__file__", "")).resolve()
    runtime_root = runtime_root.resolve()
    if runtime_root not in module_path.parents:
        raise gate.GateFailure(
            "MXFP4 oracle module resolved outside explicit --runtime-root: "
            f"module={module_path}, root={runtime_root}"
        )
    return backend, module_path


def _mx_grad_oracle(
    backend,
    value: torch.Tensor,
    coordinate: tuple[int, int],
) -> dict[str, torch.Tensor]:
    row_fp4, row_sc = backend.mxfp4_quantize_for_gemm_opt(
        value,
        1,
        data_stochastic_rounding=True,
        scale_stochastic_rounding=False,
        rng_seed=int(coordinate[0]),
        rng_subsequence=int(coordinate[1]),
    )
    col_fp4, col_sc = backend.mxfp4_quantize_col_only_opt_rht(
        value,
        1,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        rht_axes="col",
        rht_block_size=16,
        with_random_sign_mask=False,
        rng_seed=int(coordinate[0]),
        rng_subsequence=int(coordinate[1]),
    )
    return {
        "mx_row_fp4": row_fp4,
        "mx_row_sc": row_sc,
        "mx_col_fp4": col_fp4,
        "mx_col_sc": col_sc,
    }


def _mx_weight_oracle(backend, value: torch.Tensor) -> dict[str, torch.Tensor]:
    result = backend.mxfp4_quantize_weight_2d(value)
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise gate.GateFailure("MXFP4 2-D weight oracle must return four tensors")
    return {
        "mx_row_fp4": result[0],
        "mx_row_sc": result[1],
        "mx_col_fp4": result[2],
        "mx_col_sc": result[3],
    }


def _mx_split2_grad_oracle(
    backend,
    value0: torch.Tensor,
    value1: torch.Tensor,
    coordinate: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Run the exact production-oriented native MX split-2 producers."""
    if value0.shape != value1.shape or value0.dim() != 2:
        raise gate.GateFailure("native MX split2 oracle requires equal 2D arms")
    rows, arm_width = map(int, value0.shape)
    logical_width = 2 * arm_width
    row_fp4 = torch.empty(
        (rows, logical_width // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=value0.device,
    )
    row_sc = torch.empty(
        (rows // 128, logical_width // 128, 32, 16),
        dtype=torch.uint8,
        device=value0.device,
    )
    col_fp4 = torch.empty(
        (logical_width, rows // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=value0.device,
    )
    col_sc = torch.empty(
        (logical_width // 128, rows // 128, 32, 16),
        dtype=torch.uint8,
        device=value0.device,
    )
    # The production recipe is oriented: stochastic rounding belongs only to
    # row data; deterministic fixed-sign column RHT belongs only to column.
    backend.mxfp4_quantize_split2_row_only_opt_launch_inplace(
        value0,
        value1,
        row_fp4,
        row_sc,
        1,
        data_stochastic_rounding=True,
        scale_stochastic_rounding=False,
        use_rht=False,
        rht_block_size=16,
        with_random_sign_mask=False,
        rng_seed=int(coordinate[0]),
        rng_subsequence=int(coordinate[1]),
    )
    backend.mxfp4_quantize_split2_col_only_opt_launch_inplace(
        value0,
        value1,
        col_fp4,
        col_sc,
        1,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        use_rht=True,
        rht_block_size=16,
        with_random_sign_mask=False,
        rng_seed=int(coordinate[0]),
        rng_subsequence=int(coordinate[1]),
    )
    return {
        "mx_row_fp4": row_fp4,
        "mx_row_sc": row_sc,
        "mx_col_fp4": col_fp4,
        "mx_col_sc": col_sc,
    }


def _tensor_fields(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in payload.items() if torch.is_tensor(value)}


def _payload_storage_fields(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    fields = _tensor_fields(payload)
    for index, value in enumerate(payload.get("keepalive", ())):
        if not torch.is_tensor(value):
            raise gate.GateFailure("producer keepalive contains a non-tensor")
        fields[f"keepalive_{index}"] = value
    return fields


def _clone_stored_bytes(value: torch.Tensor) -> torch.Tensor:
    """Clone packed FP4 without invoking unsupported FP4 ``copy_``."""

    if value.dtype == torch.float4_e2m1fn_x2:
        return value.view(torch.uint8).clone().view(torch.float4_e2m1fn_x2)
    return value.detach().clone()


def _new_sr_state(source, sr, key: str, args: argparse.Namespace, device):
    # The state owns the logical producer coordinate independently of format.
    return sr.MXFP4SRState(
        (key,),
        device=device,
        user_seed=args.seed,
        user_subsequence_base=args.subsequence,
        training_steps=args.warmup + args.iterations + 32,
        gradient_accumulation_steps=1,
        reservation_margin=64,
    )


def _require_common_broadcast_sg(value: torch.Tensor, *, name: str) -> dict[str, Any]:
    if value.dtype != torch.float32 or not value.is_contiguous() or value.numel() == 0:
        raise gate.GateFailure(
            f"{name} is not a non-empty contiguous FP32 common-SG grid"
        )
    flat = value.view(-1)
    finite = bool(torch.isfinite(flat).all().item())
    byte_equal = bool(
        torch.equal(
            flat.view(torch.uint8),
            flat.narrow(0, 0, 1).expand_as(flat).contiguous().view(torch.uint8),
        )
    )
    if not finite or not byte_equal:
        raise gate.GateFailure(
            f"{name} violates the mixed common-broadcast SG contract"
        )
    return {
        "pass": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "all_values_byte_equal": byte_equal,
    }


def _producer_correctness(
    args: argparse.Namespace,
    module,
    backend,
    dy: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coordinate = (int(args.seed), int(args.subsequence))
    candidate_grad = _candidate_grad(module, dy, coordinate)
    candidate_weight = _candidate_weight(module, weight)
    local_grad, local_sr = _localcta_grad_oracle(module, dy, coordinate)
    local_weight = _localcta_weight_oracle(module, weight)
    mx_grad = _mx_grad_oracle(backend, dy, coordinate)
    mx_weight = _mx_weight_oracle(backend, weight)
    torch.cuda.synchronize(dy.device)

    payload_exactness = {
        "grad_local_row": gate.require_exact_tensors(
            {
                name: candidate_grad[name]
                for name in ("local_row_fp4", "local_row_sc", "local_row_sg")
            },
            local_grad,
        ),
        "grad_mx_col_rht": gate.require_exact_tensors(
            {
                name: candidate_grad[name]
                for name in ("mx_col_fp4", "mx_col_sc")
            },
            {
                name: mx_grad[name]
                for name in ("mx_col_fp4", "mx_col_sc")
            },
        ),
        "weight_mx_2d_row": gate.require_exact_tensors(
            {
                name: candidate_weight[name]
                for name in ("mx_row_fp4", "mx_row_sc")
            },
            {
                name: mx_weight[name]
                for name in ("mx_row_fp4", "mx_row_sc")
            },
        ),
        "weight_localcta_2d_col": gate.require_exact_tensors(
            {
                name: candidate_weight[name]
                for name in ("local_col_fp4", "local_col_sc", "local_col_sg")
            },
            local_weight,
        ),
    }

    # A repeated explicit coordinate must be bit-exact, while the source-side
    # state reservation is audited separately when its integration is present.
    repeated_grad = _candidate_grad(module, dy, coordinate)
    repeated_weight = _candidate_weight(module, weight)
    reproducibility = {
        "grad": gate.require_exact_tensors(
            _tensor_fields(candidate_grad), _tensor_fields(repeated_grad)
        ),
        "weight": gate.require_exact_tensors(
            _tensor_fields(candidate_weight), _tensor_fields(repeated_weight)
        ),
        "coordinate": [coordinate[0], coordinate[1]],
    }

    storage = {
        "grad": gate.require_disjoint_payload_storage(_tensor_fields(candidate_grad)),
        "weight": gate.require_disjoint_payload_storage(_tensor_fields(candidate_weight)),
    }
    keepalive = {
        "grad_count": len(candidate_grad["keepalive"]),
        "weight_count": len(candidate_weight["keepalive"]),
        "grad_all_tensors": all(
            torch.is_tensor(value) for value in candidate_grad["keepalive"]
        ),
        "weight_all_tensors": all(
            torch.is_tensor(value) for value in candidate_weight["keepalive"]
        ),
    }
    if (
        keepalive["grad_count"] < 1
        or keepalive["weight_count"] < 1
        or not keepalive["grad_all_tensors"]
        or not keepalive["weight_all_tensors"]
    ):
        raise gate.GateFailure(f"mixed producer keepalive contract failed: {keepalive}")

    # Retain only candidate carriers, churn the allocator, and then re-check
    # bytes.  This catches output views that accidentally depended on a dead
    # allocator workspace.
    stable_grad = {
        name: _clone_stored_bytes(value)
        for name, value in _tensor_fields(candidate_grad).items()
    }
    stable_weight = {
        name: _clone_stored_bytes(value)
        for name, value in _tensor_fields(candidate_weight).items()
    }
    del repeated_grad, repeated_weight
    gc.collect()
    churn = [
        torch.empty((1024, 1024), device=dy.device, dtype=torch.bfloat16)
        for _ in range(8)
    ]
    for index, value in enumerate(churn):
        value.fill_(float(index))
    torch.cuda.synchronize(dy.device)
    lifetime = {
        "grad": gate.require_exact_tensors(
            _tensor_fields(candidate_grad), stable_grad
        ),
        "weight": gate.require_exact_tensors(
            _tensor_fields(candidate_weight), stable_weight
        ),
        "allocator_churn_bytes": sum(
            value.numel() * value.element_size() for value in churn
        ),
    }
    del churn

    report = {
        "pass": True,
        "payload_exactness": payload_exactness,
        "explicit_localcta_oracle_sr": local_sr,
        "explicit_coordinate_reproducibility": reproducibility,
        "payload_storage": storage,
        "keepalive": keepalive,
        "post_allocator_churn_lifetime": lifetime,
        "mixed_weight_common_broadcast_sg": {
            "candidate": _require_common_broadcast_sg(
                candidate_weight["local_col_sg"], name="candidate_local_col_sg"
            ),
            "independent_oracle": _require_common_broadcast_sg(
                local_weight["local_col_sg"], name="oracle_local_col_sg"
            ),
        },
        "comparison_semantics": {
            "carrier_payloads": "bit_exact_zero_tolerance",
            "gemm_outputs": "separate_gate_with_explicit_tolerance",
        },
    }
    oracles = {
        "candidate_grad": candidate_grad,
        "candidate_weight": candidate_weight,
        "local_grad": local_grad,
        "local_weight": local_weight,
        "mx_grad": mx_grad,
        "mx_weight": mx_weight,
    }
    return report, oracles


def _split2_correctness(
    args: argparse.Namespace,
    module,
    backend,
    integration,
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    weight0: torch.Tensor,
    weight1: torch.Tensor,
) -> dict[str, Any]:
    """Gate the FFN split2 path against independent per-arm localCTA semantics."""

    coordinate = (int(args.seed), int(args.subsequence))
    candidate = _candidate_split2_grad(module, grad0, grad1, coordinate)
    candidate_arm0, candidate_arm1 = _split2_local_arms(
        candidate, arm_width=int(grad0.shape[1])
    )
    oracle_arm0, oracle_arm1, oracle_contract = (
        _zero_padded_localcta_split2_oracle(
            module, grad0, grad1, coordinate
        )
    )
    logical = torch.cat((grad0, grad1), dim=1)
    mx = _mx_grad_oracle(backend, logical, coordinate)
    native_split2 = _mx_split2_grad_oracle(
        backend, grad0, grad1, coordinate
    )
    torch.cuda.synchronize(grad0.device)

    carrier_exactness = {
        "localcta_arm0": gate.require_exact_tensors(candidate_arm0, oracle_arm0),
        "localcta_arm1": gate.require_exact_tensors(candidate_arm1, oracle_arm1),
        "mx_col_rht_logical_concat": gate.require_exact_tensors(
            {
                "mx_col_fp4": candidate["mx_col_fp4"],
                "mx_col_sc": candidate["mx_col_sc"],
            },
            {
                "mx_col_fp4": mx["mx_col_fp4"],
                "mx_col_sc": mx["mx_col_sc"],
            },
        ),
        "mx_col_rht_native_split2": gate.require_exact_tensors(
            {
                "mx_col_fp4": candidate["mx_col_fp4"],
                "mx_col_sc": candidate["mx_col_sc"],
            },
            {
                "mx_col_fp4": native_split2["mx_col_fp4"],
                "mx_col_sc": native_split2["mx_col_sc"],
            },
        ),
    }
    independent_outer_sg = {
        "candidate": gate.require_distinct_scale_carriers(
            candidate["local_row_sg0"],
            candidate["local_row_sg1"],
            left_name="candidate_sg0",
            right_name="candidate_sg1",
            require_distinct_values=True,
        ),
        "independent_oracle": gate.require_distinct_scale_carriers(
            oracle_arm0["local_row_sg"],
            oracle_arm1["local_row_sg"],
            left_name="oracle_sg0",
            right_name="oracle_sg1",
            require_distinct_values=True,
        ),
    }

    repeated = _candidate_split2_grad(module, grad0, grad1, coordinate)
    reproducibility = gate.require_exact_tensors(
        _tensor_fields(candidate), _tensor_fields(repeated)
    )
    storage = gate.require_disjoint_payload_storage(
        _payload_storage_fields(candidate)
    )
    stable = {
        name: _clone_stored_bytes(value)
        for name, value in _tensor_fields(candidate).items()
    }
    del repeated
    gc.collect()
    churn = [
        torch.empty((1024, 1024), device=grad0.device, dtype=torch.bfloat16)
        for _ in range(8)
    ]
    for index, value in enumerate(churn):
        value.fill_(float(index))
    torch.cuda.synchronize(grad0.device)
    lifetime = gate.require_exact_tensors(_tensor_fields(candidate), stable)
    churn_bytes = sum(value.numel() * value.element_size() for value in churn)
    del churn

    source, sr = integration
    source._validate_mxfp4_localcta_dgrad_contract()
    key = sr.ffn_deriv_grad_key("mixed_carrier_gate.ffn")
    state = _new_sr_state(source, sr, key, args, grad0.device)
    sr.set_active_mxfp4_sr_state(state)
    try:
        source_before = tuple(map(int, state.peek(key)))
        source_carrier = source._quantize_mixed_split2_grad_bf16(
            grad0, grad1, producer_key=key
        )
        source_after = tuple(map(int, state.peek(key)))
        source_sr = gate.require_one_sr_advance(source_before, source_after)
        direct_at_source = _candidate_split2_grad(
            module, grad0, grad1, source_before
        )
        source_payload = {
            "local_row_fp4": source_carrier.local_row_fp4,
            "local_row_sc": source_carrier.local_row_sc,
            "local_row_sg0": source_carrier.local_row_sg0,
            "local_row_sg1": source_carrier.local_row_sg1,
            "mx_col_fp4": source_carrier.mx_col_fp4,
            "mx_col_sc": source_carrier.mx_col_sc,
        }
        source_dispatch_exact = gate.require_exact_tensors(
            source_payload, _tensor_fields(direct_at_source)
        )
        source_sg = gate.require_distinct_scale_carriers(
            source_carrier.local_row_sg0,
            source_carrier.local_row_sg1,
            left_name="source_sg0",
            right_name="source_sg1",
            require_distinct_values=True,
        )

        source_weight0 = source._quantize_mixed_weight_bf16(weight0)
        source_weight1 = source._quantize_mixed_weight_bf16(weight1)
        local_weight0 = _localcta_weight_oracle(module, weight0)
        local_weight1 = _localcta_weight_oracle(module, weight1)
        source_oracle_arm0, source_oracle_arm1, source_oracle_contract = (
            _zero_padded_localcta_split2_oracle(
                module, grad0, grad1, source_before
            )
        )
        from low_bits_training.quantization import tk_gemm

        candidate_audit = gate.LogicalGemmAudit({"dhidden_split2_onepass": 1})
        with candidate_audit.record("dhidden_split2_onepass"):
            candidate_dhidden = source._mixed_localcta_split2_dgrad(
                source_carrier, source_weight0, source_weight1
            )
        reference_audit = gate.LogicalGemmAudit(
            {"dhidden_arm0": 1, "dhidden_arm1": 1}
        )
        with reference_audit.record("dhidden_arm0"):
            reference_arm0 = tk_gemm.tk_mixed_localcta_dgrad(
                source_oracle_arm0["local_row_fp4"],
                source_oracle_arm0["local_row_sc"],
                source_oracle_arm0["local_row_sg"],
                local_weight0["local_col_fp4"],
                local_weight0["local_col_sc"],
                local_weight0["local_col_sg"],
            )
        with reference_audit.record("dhidden_arm1"):
            reference_arm1 = tk_gemm.tk_mixed_localcta_dgrad(
                source_oracle_arm1["local_row_fp4"],
                source_oracle_arm1["local_row_sc"],
                source_oracle_arm1["local_row_sg"],
                local_weight1["local_col_fp4"],
                local_weight1["local_col_sc"],
                local_weight1["local_col_sg"],
            )
        reference_dhidden = reference_arm0 + reference_arm1
        torch.cuda.synchronize(grad0.device)
        onepass = gate.strict_close_report(
            candidate_dhidden,
            reference_dhidden,
            name="split2_dhidden_zero_padded_per_arm_oracle",
            atol=args.dhidden_atol,
            rtol=args.dhidden_rtol,
        )
        finite = gate.require_finite(
            {
                "candidate_split2_dhidden": candidate_dhidden,
                "reference_split2_dhidden": reference_dhidden,
            }
        )
    finally:
        sr.set_active_mxfp4_sr_state(None)

    return {
        "pass": True,
        "carrier_payload_exactness": carrier_exactness,
        "independent_zero_padded_oracle": oracle_contract,
        "per_arm_outer_sg": independent_outer_sg,
        "explicit_coordinate_reproducibility": reproducibility,
        "payload_storage": storage,
        "post_allocator_churn_lifetime": {
            "pass": True,
            "payload": lifetime,
            "allocator_churn_bytes": churn_bytes,
        },
        "source_dispatch": {
            "pass": True,
            "coordinate": [int(source_before[0]), int(source_before[1])],
            "one_logical_sr_advance": source_sr,
            "independent_oracle": source_oracle_contract,
            "exact_runtime_payload": source_dispatch_exact,
            "per_arm_outer_sg": source_sg,
        },
        "onepass_dgrad": {
            "pass": True,
            "comparison": onepass,
            "finite": finite,
            "candidate_logical_gemms": candidate_audit.report(),
            "reference_logical_gemms": reference_audit.report(),
            "config_idx": -1,
            "reference": (
                "sum_of_two_independent_one_input_localcta_dgrad_gemms_"
                "using_zero_padded_per_arm_carriers"
            ),
        },
        "rejected_contract": "shared_or_byte_identical_outer_sg",
    }


def _source_integration():
    from low_bits_training.quantization import mxfp4_fused_linear as source

    required = (
        "_WoFunction_MXFP4_TK",
        "_quantize_mixed_grad_dy_bf16",
        "_quantize_mixed_split2_grad_bf16",
        "_quantize_mixed_weight_bf16",
        "_mixed_localcta_dgrad",
        "_mixed_localcta_split2_dgrad",
        "_validate_mxfp4_localcta_dgrad_contract",
    )
    if any(not hasattr(source, name) for name in required):
        return None
    from low_bits_training.quantization import mxfp4_sr_state as sr

    return source, sr


@contextmanager
def _logical_gemm_wrappers(source, candidate: bool):
    if candidate:
        mapping = {
            "_mxfp4_gemm_qkv": "forward",
            "_mixed_localcta_dgrad": "dhidden",
            "_mxfp4_gemm_wgrad": "dweight",
        }
    else:
        mapping = {
            "_mxfp4_gemm_qkv": "forward",
            "_mxfp4_gemm_wo_dgrad": "dhidden",
            "_mxfp4_gemm_wgrad": "dweight",
        }
    audit = gate.LogicalGemmAudit({"forward": 1, "dhidden": 1, "dweight": 1})
    originals = {}
    try:
        for attribute, label in mapping.items():
            originals[attribute] = getattr(source, attribute)
            setattr(source, attribute, audit.wrap(label, originals[attribute]))
        yield audit
    finally:
        for attribute, original in originals.items():
            setattr(source, attribute, original)


def _run_wo_once(
    source,
    sr,
    args: argparse.Namespace,
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    *,
    candidate: bool,
    audit_gemms: bool,
) -> tuple[dict[str, torch.Tensor], tuple[int, int], tuple[int, int], dict[str, Any] | None]:
    debug_name = "mixed_carrier_gate.wo"
    key = sr.wo_grad_key(debug_name)
    state = _new_sr_state(source, sr, key, args, x.device)
    sr.set_active_mxfp4_sr_state(state)
    os.environ["MXFP4_USE_LOCALCTA_DGRAD"] = "1" if candidate else "0"
    before = tuple(map(int, state.peek(key)))
    input_leaf = x.detach().clone().requires_grad_(True)
    weight_leaf = weight.detach().clone().requires_grad_(True)

    if audit_gemms:
        context = _logical_gemm_wrappers(source, candidate)
    else:
        context = nullcontext(None)
    with context as audit:
        output = source._WoFunction_MXFP4_TK.apply(
            input_leaf, weight_leaf, debug_name, None, None
        )
        output.backward(dy)
    torch.cuda.synchronize(x.device)
    after = tuple(map(int, state.peek(key)))
    sr_report = gate.require_one_sr_advance(before, after)
    gemm_report = audit.report() if audit is not None else None
    result = {
        "forward": output.detach().clone(),
        "dhidden": input_leaf.grad.detach().clone(),
        "dweight": weight_leaf.grad.detach().clone(),
    }
    sr.set_active_mxfp4_sr_state(None)
    return result, before, after, {"sr": sr_report, "gemms": gemm_report}


def _full_correctness(
    args: argparse.Namespace,
    integration,
    module,
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    oracles: Mapping[str, Any],
) -> dict[str, Any]:
    source, sr = integration
    os.environ["MXFP4_USE_LOCALCTA_DGRAD"] = "1"
    source._validate_mxfp4_localcta_dgrad_contract()
    baseline, baseline_before, _, baseline_audit = _run_wo_once(
        source, sr, args, x, weight, dy, candidate=False, audit_gemms=True
    )
    candidate, candidate_before, _, candidate_audit = _run_wo_once(
        source, sr, args, x, weight, dy, candidate=True, audit_gemms=True
    )
    if baseline_before != candidate_before:
        raise gate.GateFailure(
            "baseline/candidate SR logical coordinates differ: "
            f"{baseline_before} != {candidate_before}"
        )

    from low_bits_training.quantization import tk_gemm

    # The ranked state namespaces the user seed by logical producer and rank,
    # so this full-linear oracle must use the actual candidate coordinate, not
    # the un-namespaced producer-only diagnostic coordinate.
    local_grad, full_local_sr = _localcta_grad_oracle(
        module, dy, candidate_before
    )
    local_weight = _localcta_weight_oracle(module, weight)
    oracle_dhidden = tk_gemm.tk_mixed_localcta_dgrad(
        local_grad["local_row_fp4"],
        local_grad["local_row_sc"],
        local_grad["local_row_sg"],
        local_weight["local_col_fp4"],
        local_weight["local_col_sc"],
        local_weight["local_col_sg"],
    )
    torch.cuda.synchronize(x.device)
    exact_mx = gate.require_exact_tensors(
        {"forward": candidate["forward"], "dweight": candidate["dweight"]},
        {"forward": baseline["forward"], "dweight": baseline["dweight"]},
    )
    dhidden = gate.strict_close_report(
        candidate["dhidden"],
        oracle_dhidden,
        name="dhidden_localcta_oracle",
        atol=args.dhidden_atol,
        rtol=args.dhidden_rtol,
    )
    finite = gate.require_finite(
        {
            **{f"baseline_{name}": value for name, value in baseline.items()},
            **{f"candidate_{name}": value for name, value in candidate.items()},
            "oracle_dhidden": oracle_dhidden,
        }
    )

    # Revert the already-advanced candidate stream to native MX row-SR and
    # compare its next coordinate to an explicit native oracle.
    debug_name = "mixed_carrier_gate.wo"
    key = sr.wo_grad_key(debug_name)
    state = _new_sr_state(source, sr, key, args, x.device)
    sr.set_active_mxfp4_sr_state(state)
    os.environ["MXFP4_USE_LOCALCTA_DGRAD"] = "1"
    mixed = source._quantize_mixed_grad_dy_bf16(dy, producer_key=key)
    after_mixed = tuple(map(int, state.peek(key)))
    os.environ["MXFP4_USE_LOCALCTA_DGRAD"] = "0"
    reverted = source._quantize_row_col_bf16(dy, role="grad", producer_key=key)
    after_revert = tuple(map(int, state.peek(key)))
    revert_advance = gate.require_one_sr_advance(after_mixed, after_revert)
    backend, _ = _load_mx_backend(Path(args.runtime_root))
    explicit_revert = _mx_grad_oracle(backend, dy, after_mixed)
    revert_exact = gate.require_exact_tensors(
        {
            "mx_row_fp4": reverted.row_fp4,
            "mx_row_sc": reverted.row_sc,
            "mx_col_fp4": reverted.col_fp4,
            "mx_col_sc": reverted.col_sc,
        },
        explicit_revert,
    )
    # Keep this reference alive through the native revert, then prove the old
    # mixed payload remains readable.
    mixed_lifetime = gate.require_exact_tensors(
        {
            "local_row_fp4": mixed.local_row_fp4,
            "local_row_sc": mixed.local_row_sc,
            "local_row_sg": mixed.local_row_sg,
            "mx_col_fp4": mixed.mx_col_fp4,
            "mx_col_sc": mixed.mx_col_sc,
        },
        {
            "local_row_fp4": _clone_stored_bytes(mixed.local_row_fp4),
            "local_row_sc": _clone_stored_bytes(mixed.local_row_sc),
            "local_row_sg": _clone_stored_bytes(mixed.local_row_sg),
            "mx_col_fp4": _clone_stored_bytes(mixed.mx_col_fp4),
            "mx_col_sc": _clone_stored_bytes(mixed.mx_col_sc),
        },
    )
    sr.set_active_mxfp4_sr_state(None)
    os.environ["MXFP4_USE_LOCALCTA_DGRAD"] = "1"
    return {
        "pass": True,
        "mx_forward_dweight_bit_exact": exact_mx,
        "localcta_dhidden": dhidden,
        "finite_outputs": finite,
        "logical_gemm_audit": {
            "baseline": baseline_audit,
            "candidate": candidate_audit,
        },
        "sr_coordinates": {
            "baseline_initial": list(baseline_before),
            "candidate_initial": list(candidate_before),
            "explicit_localcta_oracle": full_local_sr,
        },
        "revert_to_native_mx": {
            "pass": True,
            "counter_continuity": revert_advance,
            "native_payload_exact_at_next_coordinate": revert_exact,
            "prior_mixed_payload_lifetime": mixed_lifetime,
        },
    }


def _producer_timing(
    args: argparse.Namespace,
    module,
    backend,
    dy: torch.Tensor,
    dy1: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, Any]:
    counters = {
        "candidate_grad": int(args.subsequence),
        "baseline_grad": int(args.subsequence),
        "candidate_split2_grad": int(args.subsequence),
        "baseline_split2_grad": int(args.subsequence),
        "baseline_native_split2_grad": int(args.subsequence),
        "baseline_split2_grad_with_concat": int(args.subsequence),
    }

    def candidate_grad() -> None:
        coordinate = (int(args.seed), counters["candidate_grad"])
        _candidate_grad(module, dy, coordinate)
        counters["candidate_grad"] += gate.SUBSEQUENCE_STRIDE

    def baseline_grad() -> None:
        coordinate = (int(args.seed), counters["baseline_grad"])
        _mx_grad_oracle(backend, dy, coordinate)
        counters["baseline_grad"] += gate.SUBSEQUENCE_STRIDE

    logical_split2 = torch.cat((dy, dy1), dim=1).contiguous()

    def candidate_split2_grad() -> None:
        coordinate = (int(args.seed), counters["candidate_split2_grad"])
        _candidate_split2_grad(module, dy, dy1, coordinate)
        counters["candidate_split2_grad"] += gate.SUBSEQUENCE_STRIDE

    def baseline_split2_grad() -> None:
        coordinate = (int(args.seed), counters["baseline_split2_grad"])
        _mx_grad_oracle(backend, logical_split2, coordinate)
        counters["baseline_split2_grad"] += gate.SUBSEQUENCE_STRIDE

    def baseline_native_split2_grad() -> None:
        coordinate = (
            int(args.seed),
            counters["baseline_native_split2_grad"],
        )
        _mx_split2_grad_oracle(backend, dy, dy1, coordinate)
        counters["baseline_native_split2_grad"] += gate.SUBSEQUENCE_STRIDE

    def baseline_split2_grad_with_concat() -> None:
        coordinate = (
            int(args.seed),
            counters["baseline_split2_grad_with_concat"],
        )
        logical = torch.cat((dy, dy1), dim=1).contiguous()
        _mx_grad_oracle(backend, logical, coordinate)
        counters["baseline_split2_grad_with_concat"] += gate.SUBSEQUENCE_STRIDE

    baseline_grad_ms = gate.cuda_event_samples(
        baseline_grad, warmup=args.warmup, iterations=args.iterations
    )
    candidate_grad_ms = gate.cuda_event_samples(
        candidate_grad, warmup=args.warmup, iterations=args.iterations
    )
    baseline_split2_grad_ms = gate.cuda_event_samples(
        baseline_split2_grad, warmup=args.warmup, iterations=args.iterations
    )
    baseline_native_split2_grad_ms = gate.cuda_event_samples(
        baseline_native_split2_grad,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    candidate_split2_grad_ms = gate.cuda_event_samples(
        candidate_split2_grad, warmup=args.warmup, iterations=args.iterations
    )
    bf16_concat_ms = gate.cuda_event_samples(
        lambda: torch.cat((dy, dy1), dim=1).contiguous(),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    baseline_split2_grad_with_concat_ms = gate.cuda_event_samples(
        baseline_split2_grad_with_concat,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    baseline_weight_ms = gate.cuda_event_samples(
        lambda: _mx_weight_oracle(backend, weight),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    candidate_weight_ms = gate.cuda_event_samples(
        lambda: _candidate_weight(module, weight),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    return {
        "pass": True,
        "window": {"warmup": args.warmup, "scored": args.iterations},
        "grad_producer": gate.timing_comparison(
            baseline_grad_ms, candidate_grad_ms
        ),
        "split2_grad_producer": gate.timing_comparison(
            baseline_native_split2_grad_ms, candidate_split2_grad_ms
        ),
        "split2_grad_producer_prebuilt_concat_diagnostic": gate.timing_comparison(
            baseline_split2_grad_ms, candidate_split2_grad_ms
        ),
        "split2_grad_producer_with_bf16_concat": gate.timing_comparison(
            baseline_split2_grad_with_concat_ms,
            candidate_split2_grad_ms,
        ),
        "bf16_concat": {
            **gate.summarize_ms(bf16_concat_ms),
            "transient_bytes": int(dy.numel() + dy1.numel())
            * int(dy.element_size()),
        },
        "weight_producer": gate.timing_comparison(
            baseline_weight_ms, candidate_weight_ms
        ),
        "baseline_semantics": {
            "grad": "MX row-SR plus deterministic MX col-RHT",
            "split2_grad": (
                "production-oriented native split2: row data SR plus "
                "deterministic fixed-sign column RHT, no BF16 concat"
            ),
            "split2_grad_prebuilt_concat_diagnostic": (
                "one-input native MX row-SR plus deterministic MX col-RHT "
                "on a pre-materialized logical concatenation"
            ),
            "split2_grad_with_bf16_concat": (
                "BF16 dim-1 concatenation followed by native MX row-SR plus "
                "deterministic MX col-RHT"
            ),
            "weight": "MX exact-2D row+col",
        },
        "candidate_semantics": {
            "grad": "fused localCTA row-SR plus deterministic MX col-RHT",
            "split2_grad": (
                "one fused no-BF16-concat producer with per-arm localCTA outer SG"
            ),
            "weight": "fused MX exact-2D row plus localCTA exact-2D col",
        },
    }


def _full_timing(
    args: argparse.Namespace,
    integration,
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
) -> dict[str, Any]:
    source, sr = integration
    debug_name = "mixed_carrier_gate.timing.wo"
    key = sr.wo_grad_key(debug_name)

    def arm(candidate: bool):
        state = _new_sr_state(source, sr, key, args, x.device)
        sr.set_active_mxfp4_sr_state(state)
        os.environ["MXFP4_USE_LOCALCTA_DGRAD"] = "1" if candidate else "0"
        input_leaf = x.detach().clone().requires_grad_(True)
        weight_leaf = weight.detach().clone().requires_grad_(True)

        def before_each() -> None:
            input_leaf.grad = None
            weight_leaf.grad = None

        def run() -> None:
            output = source._WoFunction_MXFP4_TK.apply(
                input_leaf, weight_leaf, debug_name, None, None
            )
            output.backward(dy)

        samples = gate.cuda_event_samples(
            run,
            warmup=args.warmup,
            iterations=args.iterations,
            before_each=before_each,
        )
        return samples

    baseline_ms = arm(False)
    candidate_ms = arm(True)
    sr.set_active_mxfp4_sr_state(None)
    os.environ["MXFP4_USE_LOCALCTA_DGRAD"] = "1"
    comparison = gate.timing_comparison(baseline_ms, candidate_ms)
    passed = comparison["median_overhead_pct"] <= args.max_full_overhead_pct
    comparison.update(
        {
            "pass": passed,
            "max_overhead_pct": args.max_full_overhead_pct,
            "window": {"warmup": args.warmup, "scored": args.iterations},
            "scope": "Wo forward plus backward; exactly three logical GEMMs",
        }
    )
    if not passed:
        raise gate.GateFailure(
            "full mixed Wo overhead exceeded gate: "
            f"{comparison['median_overhead_pct']:.3f}% > "
            f"{args.max_full_overhead_pct:.3f}%"
        )
    return comparison


def _base_receipt(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = Path(args.runtime_root).resolve()
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "method": gate.METHOD,
        "route_id": gate.ROUTE_ID,
        "recipe": gate.EXACT_RECIPE,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "rng": {
            "user_seed": args.seed,
            "user_subsequence": args.subsequence,
            "tensor_seed": args.tensor_seed,
            "subsequence_stride": gate.SUBSEQUENCE_STRIDE,
        },
        "tolerances": {
            "carrier_bytes": "zero",
            "mx_forward": "zero",
            "mx_dweight": "zero",
            "localcta_dhidden_atol": args.dhidden_atol,
            "localcta_dhidden_rtol": args.dhidden_rtol,
        },
        "source": {
            "root": str(repo_root),
            "head": _git_head(repo_root),
            "required_ancestor": REQUIRED_SOURCE_COMMIT,
        },
        "runtime": {
            "root": str(runtime_root),
            "head": _git_head(runtime_root),
            "required_ancestor": REQUIRED_RUNTIME_COMMIT,
        },
        "command_contract": {
            "producer_only": bool(args.producer_only),
            "timing_only": bool(args.timing_only),
            "timing_requested": not args.skip_timing,
            "warmup": args.warmup,
            "scored_iterations": args.iterations,
            "max_full_overhead_pct": args.max_full_overhead_pct,
        },
    }


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    _set_exact_policy(args)
    receipt = _base_receipt(args)
    started = time.time()
    status_code = 1
    try:
        repo_root = Path(__file__).resolve().parents[1]
        runtime_root = Path(args.runtime_root).resolve()
        if not _git_is_ancestor(repo_root, REQUIRED_SOURCE_COMMIT):
            raise FeatureUnavailable(
                f"source must contain required commit {REQUIRED_SOURCE_COMMIT}"
            )
        if not _git_is_ancestor(runtime_root, REQUIRED_RUNTIME_COMMIT):
            raise FeatureUnavailable(
                f"runtime must contain required commit {REQUIRED_RUNTIME_COMMIT}"
            )
        if not torch.cuda.is_available():
            raise FeatureUnavailable("CUDA is unavailable")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        module, module_path, capabilities = _load_runtime_module(args)
        backend, mx_module_path = _load_mx_backend(Path(args.runtime_root))
        receipt["runtime"]["mx_extension"] = str(mx_module_path)
        receipt["runtime"]["mx_extension_sha256"] = _sha256_file(mx_module_path)
        gemm_candidates = [
            Path(path).resolve() for path in backend._candidate_gemm_paths()
        ]
        if gemm_candidates:
            gemm_path = gemm_candidates[0]
            if runtime_root not in gemm_path.parents:
                raise gate.GateFailure(
                    "MXFP4 GEMM resolved outside explicit --runtime-root: "
                    f"module={gemm_path}, root={runtime_root}"
                )
            receipt["runtime"]["mx_gemm_extension"] = str(gemm_path)
            receipt["runtime"]["mx_gemm_extension_sha256"] = _sha256_file(
                gemm_path
            )
        receipt["runtime"].update(
            {
                "extension": str(module_path),
                "extension_sha256": _sha256_file(module_path),
                "capabilities": capabilities,
            }
        )
        receipt["allocator_abi"] = {
            "grad": _require_six_tensor_allocator(
                module,
                name="tk_mixed_grad_localcta_row_mx_col_alloc",
                rows=args.m,
                cols=args.n,
                device=device,
            ),
            "weight": _require_six_tensor_allocator(
                module,
                name="tk_mixed_weight_mx_row_localcta_col_alloc",
                rows=args.n,
                cols=args.k,
                device=device,
            ),
            "split2_grad": _require_seven_tensor_split2_allocator(
                module,
                rows=args.m,
                arm_width=args.n,
                device=device,
            ),
        }

        generator = torch.Generator(device=device)
        generator.manual_seed(args.tensor_seed)
        weight = torch.randn(
            (args.n, args.k), generator=generator, device=device, dtype=torch.bfloat16
        ) * 0.02
        dy = torch.randn(
            (args.m, args.n), generator=generator, device=device, dtype=torch.bfloat16
        ) * 0.01
        # Deliberately separate the two arm ranges so a shared outer-SG carrier
        # cannot accidentally pass merely because both arms quantize similarly.
        dy1 = torch.randn(
            (args.m, args.n), generator=generator, device=device, dtype=torch.bfloat16
        ) * (0.01 / 128.0)
        if args.timing_only:
            x = None
            weight1 = None
        else:
            x = torch.randn(
                (args.m, args.k),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            weight1 = torch.randn(
                (args.n, args.k),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ) * 0.005

        integration = _source_integration()
        receipt["source_integration_available"] = integration is not None
        if args.timing_only:
            receipt["correctness"] = {
                "status": "NOT_REQUESTED",
                "reason": (
                    "shape-faithful producer timing only; correctness must be "
                    "established by a separate sealed receipt with identical "
                    "source, runtime, and extension hashes"
                ),
            }
            oracles = None
        else:
            if integration is None:
                raise FeatureUnavailable(
                    "source integration at commit 5011942c0 or a descendant is required "
                    "for the split2 one-pass dgrad gate"
                )
            producer_correctness, oracles = _producer_correctness(
                args, module, backend, dy, weight
            )
            receipt["producer_correctness"] = producer_correctness
            assert weight1 is not None
            receipt["split2_correctness"] = _split2_correctness(
                args,
                module,
                backend,
                integration,
                dy,
                dy1,
                weight,
                weight1,
            )
        if not args.producer_only:
            assert integration is not None
            assert x is not None
            assert oracles is not None
            receipt["full_linear_correctness"] = _full_correctness(
                args, integration, module, x, weight, dy, oracles
            )

        if args.skip_timing:
            receipt["timing"] = {"status": "NOT_REQUESTED"}
        else:
            receipt["producer_timing"] = _producer_timing(
                args, module, backend, dy, dy1, weight
            )
            if not args.producer_only:
                assert integration is not None
                assert x is not None
                receipt["full_linear_timing"] = _full_timing(
                    args, integration, x, weight, dy
                )
        receipt["status"] = "PASS"
        status_code = 0
    except FeatureUnavailable as error:
        receipt["status"] = "UNAVAILABLE"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        status_code = 2
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        status_code = 1
    finally:
        receipt["elapsed_seconds"] = time.time() - started
        sealed = gate.seal_receipt(receipt)
        gate.write_receipt_exclusive(args.output, sealed)
        print(json.dumps(sealed, indent=2, sort_keys=True))
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
