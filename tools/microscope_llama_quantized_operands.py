#!/usr/bin/env python3
"""Measure production FP4 operands from a sealed, row-aligned Llama capture.

This is an operand microscope, not a model replay.  It calls the production
quantizer for one route in a fresh process, reconstructs the actual row and
column operands, and records zeroing, range, saturation, and error statistics.
The initial scope is the Llama FFN down projection at layers 12, 16, and 31:

* the fused SwiGLU activation consumed by W2;
* W2 dY (the FFN output gradient); and
* the production 2D-quantized W2 weight.

RHT acts across adjacent source rows.  Consequently, this program rejects a
capture unless every sampled row belongs to a complete, globally aligned RHT
block.  Arbitrarily sampled rows must never be presented as exact RHT input.

Only compact JSON metrics and provenance hashes are written.  Quantized or
BF16 operand payloads are never persisted by this tool.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

# Keep the heavyweight package initializer inert.  Native policy is owned by
# this standalone process and is not inherited from an installed LBT package.
os.environ.setdefault("LBT_LIGHT_IMPORT", "1")

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CAPTURE_HELPER_PATH = (
    REPO_ROOT / "low_bits_training" / "analysis" / "layerwise_replay_capture.py"
)
_CAPTURE_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_lbt_operand_microscope_capture_helpers", _CAPTURE_HELPER_PATH
)
if _CAPTURE_HELPER_SPEC is None or _CAPTURE_HELPER_SPEC.loader is None:
    raise RuntimeError(f"cannot load capture helpers from {_CAPTURE_HELPER_PATH}")
capture_helpers = importlib.util.module_from_spec(_CAPTURE_HELPER_SPEC)
sys.modules[_CAPTURE_HELPER_SPEC.name] = capture_helpers
_CAPTURE_HELPER_SPEC.loader.exec_module(capture_helpers)

CAPTURE_METHOD = capture_helpers.CAPTURE_METHOD
CAPTURE_SCHEMA_VERSION = capture_helpers.SCHEMA_VERSION
canonical_json_bytes = capture_helpers.canonical_json_bytes
load_json_object = capture_helpers.load_json_object
seal_receipt = capture_helpers.seal_receipt
sha256_file = capture_helpers.sha256_file
staged_output_directory = capture_helpers.staged_output_directory
tensor_sha256 = capture_helpers.tensor_sha256
validate_converted_model = capture_helpers.validate_converted_model
validate_receipt_seal = capture_helpers.validate_receipt_seal


METHOD = "direct-production-fp4-operand-microscope-v1"
SCHEMA_VERSION = 1
ROUTES = (
    "localcta-sr-only",
    "localcta-rht",
    "mxfp4-sr-only",
    "mxfp4-rht",
)
SITES = ("w2_activation", "w2_dy", "down_weight")
DEFAULT_LAYERS = "12,16,31"
DEFAULT_RHT_BLOCK_SIZE = 16
FIXED_RHT_POSITIVE_MASK = 0x2817
FIXED_RHT_SIGNS = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)
E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
TINY = torch.finfo(torch.float64).tiny

LOCALCTA_EXTENSION_DIRECTORY = Path("TK_quantisation/nvfp4_CTA_local_v4")
MXFP4_EXTENSION_DIRECTORY = Path("TK_quantisation/mxfp4_v4")
LOCALCTA_SOURCE_FILE = LOCALCTA_EXTENSION_DIRECTORY / "tk_quantize.cu"
MXFP4_SOURCE_FILE = MXFP4_EXTENSION_DIRECTORY / "tk_quantize.cu"


def _parse_layer_spec(value: str, num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if value.strip().lower() == "all":
        return list(range(num_layers))
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"descending layer range: {item!r}")
            result.update(range(start, stop + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 0 or max(result) >= num_layers:
        raise ValueError(
            f"layers must be zero-based values in [0, {num_layers}), got {value!r}"
        )
    return sorted(result)


def _parse_sites(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return SITES
    result = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",")))
    if not result or any(item not in SITES for item in result):
        raise ValueError(f"sites must be all or a subset of {SITES}")
    return result


def validate_aligned_row_blocks(
    row_indices: Sequence[int],
    *,
    total_rows: int,
    block_size: int,
) -> dict[str, Any]:
    """Validate that sampled rows are a union of complete source RHT blocks."""

    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    rows = [int(value) for value in row_indices]
    if not rows:
        raise RuntimeError("capture selected no rows")
    if rows != sorted(rows) or len(rows) != len(set(rows)):
        raise RuntimeError("capture row indices must be strictly increasing")
    if rows[0] < 0 or rows[-1] >= total_rows:
        raise RuntimeError("capture row index is outside the source geometry")
    if len(rows) % block_size:
        raise RuntimeError(
            f"capture has {len(rows)} rows, not a multiple of RHT block {block_size}"
        )
    block_starts: list[int] = []
    for offset in range(0, len(rows), block_size):
        group = rows[offset : offset + block_size]
        start = group[0]
        if start % block_size or group != list(range(start, start + block_size)):
            raise RuntimeError(
                "capture rows are not complete globally aligned RHT blocks; "
                f"first invalid group begins with sampled row {start}"
            )
        block_starts.append(start)
    return {
        "block_size": block_size,
        "sample_rows": len(rows),
        "source_rows": total_rows,
        "complete_blocks": len(block_starts),
        "block_starts": block_starts,
        "covers_all_source_rows": rows == list(range(total_rows)),
    }


class AlignedCaptureReader:
    """Strict reader for a sealed capture with complete aligned RHT blocks."""

    def __init__(self, root: Path, *, rht_block_size: int) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "capture_manifest.json"
        self.manifest = load_json_object(manifest_path)
        validate_receipt_seal(self.manifest)
        if self.manifest.get("schema_version") != CAPTURE_SCHEMA_VERSION:
            raise RuntimeError("capture schema version mismatch")
        if self.manifest.get("method") != CAPTURE_METHOD:
            raise RuntimeError("capture method mismatch")
        geometry = self.manifest.get("geometry")
        if not isinstance(geometry, dict):
            raise RuntimeError("capture manifest has no geometry")
        self.num_layers = int(geometry["num_layers"])
        self.batch_size = int(geometry["batch_size"])
        self.seq_len = int(geometry["seq_len"])
        self.row_indices = [int(value) for value in geometry["sample_row_indices"]]
        if int(geometry["sample_rows"]) != len(self.row_indices):
            raise RuntimeError("capture sample-row count does not match row ledger")
        self.alignment = validate_aligned_row_blocks(
            self.row_indices,
            total_rows=self.batch_size * self.seq_len,
            block_size=rht_block_size,
        )
        self.manifest_sha256 = sha256_file(manifest_path)

    def load_layer(
        self, layer: int, required_names: Iterable[str]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        required = set(required_names)
        tensors: dict[str, torch.Tensor] = {}
        evidence: dict[str, Any] = {}
        files = self.manifest.get("files")
        if not isinstance(files, dict):
            raise RuntimeError("capture manifest has no file ledger")
        for phase in ("forward", "backward"):
            filename = f"layer_{layer:02d}_{phase}.pt"
            record = files.get(filename)
            if not isinstance(record, dict):
                raise RuntimeError(f"capture manifest is missing {filename}")
            path = self.root / filename
            digest = sha256_file(path)
            if digest != record.get("sha256") or path.stat().st_size != record.get(
                "bytes"
            ):
                raise RuntimeError(f"capture file ledger mismatch: {filename}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"capture payload is not a dict: {filename}")
            expected_header = {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "method": CAPTURE_METHOD,
                "layer": layer,
                "phase": phase,
                "row_indices": self.row_indices,
            }
            for key, expected in expected_header.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"capture {filename} has invalid {key}")
            phase_tensors = payload.get("tensors")
            metadata = payload.get("tensor_metadata")
            if not isinstance(phase_tensors, dict) or not isinstance(metadata, dict):
                raise RuntimeError(f"capture {filename} has invalid tensor inventory")
            if set(phase_tensors) != set(metadata) or set(metadata) != set(
                record.get("tensors", {})
            ):
                raise RuntimeError(f"capture tensor inventory mismatch: {filename}")
            for name, tensor in phase_tensors.items():
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(f"capture tensor is invalid: {filename}:{name}")
                tensor_record = metadata[name]
                if (
                    list(tensor.shape) != tensor_record.get("sample_shape")
                    or str(tensor.dtype) != tensor_record.get("dtype")
                    or tensor_sha256(tensor) != tensor_record.get("sha256")
                    or tensor_record != record["tensors"][name]
                ):
                    raise RuntimeError(
                        f"capture tensor metadata mismatch: {filename}:{name}"
                    )
                if name in required:
                    if name in tensors:
                        raise RuntimeError(f"duplicate capture tensor: {name}")
                    tensors[name] = tensor
            evidence[phase] = {
                "filename": filename,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        missing = sorted(required - set(tensors))
        if missing:
            raise RuntimeError(f"capture layer {layer} is missing tensors: {missing}")
        return tensors, evidence


class ConvertedDownWeightStore:
    """Load and verify only one Llama layer's BF16 down-projection weight."""

    def __init__(self, root: Path, *, verify_shards: bool) -> None:
        self.root = root.resolve()
        self.summary = validate_converted_model(self.root, verify_weight_files=False)
        index = load_json_object(self.root / "model.safetensors.index.json")
        self.weight_map = index.get("weight_map")
        if not isinstance(self.weight_map, dict) or not self.weight_map:
            raise RuntimeError("converted model index has no weight map")
        conversion = load_json_object(self.root / "conversion_manifest.json")
        self.file_hashes = conversion.get("weight_files")
        if not isinstance(self.file_hashes, dict):
            raise RuntimeError("conversion manifest has no weight-file hashes")
        self.verify_shards = verify_shards
        self._verified: dict[str, str] = {}

    def load(self, layer: int) -> tuple[torch.Tensor, dict[str, Any]]:
        from safetensors import safe_open

        name = f"model.layers.{layer}.mlp.down_proj.weight"
        filename = self.weight_map.get(name)
        if not isinstance(filename, str):
            raise RuntimeError(f"converted model has no tensor {name}")
        path = self.root / filename
        expected_digest = self.file_hashes.get(filename)
        if not isinstance(expected_digest, str):
            raise RuntimeError(f"conversion has no hash for {filename}")
        verified_now = False
        if self.verify_shards and filename not in self._verified:
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest:
                raise RuntimeError(f"converted weight shard SHA mismatch: {filename}")
            self._verified[filename] = actual_digest
            verified_now = True
        with safe_open(path, framework="pt", device="cpu") as handle:
            if name not in set(handle.keys()):
                raise RuntimeError(f"weight shard does not contain indexed tensor {name}")
            weight = handle.get_tensor(name).contiguous()
        if weight.dtype is not torch.bfloat16 or weight.ndim != 2:
            raise RuntimeError(
                f"down weight is not a BF16 matrix: {name}={weight.dtype}/{weight.shape}"
            )
        return weight, {
            "tensor_name": name,
            "tensor_sha256": tensor_sha256(weight),
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "shard": filename,
            "shard_bytes": path.stat().st_size,
            "shard_sha256": expected_digest,
            "shard_sha256_verified_now": verified_now,
        }


def _check_capture_conversion_binding(
    capture_manifest: Mapping[str, Any], converted_summary: Mapping[str, Any]
) -> None:
    serialized = canonical_json_bytes(capture_manifest.get("bindings", {})).decode()
    required = (
        converted_summary["conversion_manifest_sha256"],
        converted_summary["checkpoint_metadata_sha256"],
        converted_summary["source_uri_sha256"],
    )
    missing = [value for value in required if value not in serialized]
    if missing:
        raise RuntimeError(
            "capture does not bind the selected conversion/checkpoint identity: "
            f"{missing}"
        )


def _required_source_symbols(route: str) -> tuple[Path, tuple[str, ...]]:
    if route.startswith("localcta"):
        symbols = [
            "tk_localcta_quantize_for_gemm_opt",
            "tk_localcta_quantize_weight_2d",
            "tk_localcta_reconstruct_row",
            "tk_localcta_reconstruct_col",
            "tk_localcta_set_global_scale_num",
            "tk_localcta_silu_quantize_split_for_gemm",
        ]
        if route == "localcta-rht":
            symbols.extend(
                (
                    "tk_localcta_silu_supports_paired_col_rht",
                    "tk_localcta_silu_quantize_split_for_gemm_paired_col_rht",
                )
            )
        return LOCALCTA_SOURCE_FILE, tuple(symbols)
    if route.startswith("mxfp4"):
        symbols = [
            "mxfp4_quantize_for_gemm_opt",
            "mxfp4_quantize_col_only",
            "mxfp4_quantize_weight_2d",
            "mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace",
        ]
        if route == "mxfp4-rht":
            symbols.extend(
                (
                    "mxfp4_quantize_row_and_col_opt_rht",
                    "mxfp4_quantize_col_only_opt_rht",
                )
            )
        return MXFP4_SOURCE_FILE, tuple(symbols)
    raise ValueError(f"unsupported route: {route}")


def audit_runtime_source_contract(root: Path, route: str) -> dict[str, Any]:
    """Hash the selected runtime source and fail if its pybind ABI drifted."""

    relative, symbols = _required_source_symbols(route)
    path = root.resolve() / relative
    if not path.is_file():
        raise RuntimeError(f"runtime source contract file is absent: {path}")
    source = path.read_text()
    missing = [symbol for symbol in symbols if f'm.def("{symbol}"' not in source]
    if missing:
        raise RuntimeError(f"runtime source is missing required pybind symbols: {missing}")
    result = {
        "root": str(root.resolve()),
        "source_file": relative.as_posix(),
        "source_file_bytes": path.stat().st_size,
        "source_file_sha256": sha256_file(path),
        "required_pybind_symbols": list(symbols),
    }
    if route == "localcta-rht":
        header_relative = LOCALCTA_EXTENSION_DIRECTORY / "fused_localcta_quantize.cuh"
        header = root.resolve() / header_relative
        if not header.is_file():
            raise RuntimeError(f"localCTA fixed-sign source header is absent: {header}")
        header_text = header.read_text()
        if "return 0x00002817u;" not in header_text:
            raise RuntimeError("localCTA source does not contain sealed fixed mask 0x2817")
        result["fixed_sign_source"] = {
            "file": header_relative.as_posix(),
            "bytes": header.stat().st_size,
            "sha256": sha256_file(header),
            "positive_mask": "0x2817",
        }
    return result


def _git_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()

    def command(*args: str) -> str:
        return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()

    return {
        "root": str(root),
        "head": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_short": command(
            "status", "--short", "--untracked-files=all"
        ).splitlines(),
    }


def _find_quant_extension(runtime_root: Path, route: str) -> tuple[Path, str]:
    runtime_root = runtime_root.resolve()
    if route.startswith("localcta"):
        directory = runtime_root / LOCALCTA_EXTENSION_DIRECTORY
        module_name = "_tk_quant_localcta_v4"
    else:
        directory = runtime_root / MXFP4_EXTENSION_DIRECTORY
        module_name = "mxfp4_quant_v4"
    candidates = sorted(directory.glob("*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one quant extension in {directory}, found {candidates}"
        )
    return candidates[0].resolve(), module_name


def _load_extension(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_route_environment(route: str, *, localcta_scale_num: float) -> dict[str, str]:
    """Seal native producer switches before loading either extension."""

    if route.startswith("localcta"):
        use_rht = route == "localcta-rht"
        policy = {
            "USE_TK_LOCALCTA_VARIANT": "v4",
            "USE_TK_LOCALCTA_SCALE_NUM": str(localcta_scale_num),
            "USE_TK_LOCALCTA_V3_CONTRACT": "outer",
            "USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER": "1",
            "USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER": "1",
            "USE_TK_LOCALCTA_V4_FAST_DATA_SR": "1",
            "USE_TK_LOCALCTA_V4_SILU_FINAL_SG_PRODUCER": "0",
            "USE_TK_LOCALCTA_V4_SILU_ATOMIC_FINAL_SG_PRODUCER": "1",
            "USE_TK_LOCALCTA_V4_FUSED_SILU_RAW": "1",
            "USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE": "0",
            "USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE_FORCE_RAW": "0",
            "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW": "1" if use_rht else "0",
            "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_RAW_MULTIPLIER": "2.0",
        }
    elif route.startswith("mxfp4"):
        # MX producer policy is passed through the explicit pybind ABI.  These
        # values seal the corresponding high-level recipe for provenance and
        # prevent an inherited shell from claiming a different route.
        use_rht = route == "mxfp4-rht"
        policy = {
            "MXFP4_BACKEND_VERSION": "v4",
            "MXFP4_USE_2D_WEIGHT_QUANT": "1",
            "MXFP4_USE_RHT": "1" if use_rht else "0",
            "MXFP4_RHT_ACTIVATION": "1" if use_rht else "0",
            "MXFP4_RHT_GRAD": "1" if use_rht else "0",
            "MXFP4_RHT_WEIGHT": "0",
            "MXFP4_RHT_TE_STYLE": "1",
            "MXFP4_RHT_AXES": "col",
            "MXFP4_GRAD_SR_AXES": "row",
        }
    else:
        raise ValueError(route)
    os.environ.update(policy)
    return policy


def tensor_payload_sha256(tensor: torch.Tensor) -> str:
    """Hash a native tensor after viewing shell dtypes as bytes on-device."""

    value = tensor.detach().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    )
    raw = value.view(torch.uint8).cpu().contiguous().numpy().tobytes(order="C")
    return sha256(header + b"\0" + raw).hexdigest()


def unpack_e2m1_codes(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Unpack low nibble then high nibble into a logical E2M1 code matrix."""

    if rows <= 0 or cols <= 0 or cols % 2:
        raise ValueError("packed E2M1 geometry requires positive rows and even cols")
    raw = packed.detach().contiguous().view(torch.uint8)
    if raw.numel() != rows * cols // 2:
        raise RuntimeError(
            f"packed E2M1 payload has {raw.numel()} bytes, expected {rows * cols // 2}"
        )
    raw = raw.reshape(rows, cols // 2)
    return torch.stack((raw & 0x0F, raw >> 4), dim=-1).reshape(rows, cols)


def unswizzle_mxfp4_scales(
    scales: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """Invert the production [mb,kb,32,16] MXFP4 scale swizzle."""

    if rows <= 0 or cols <= 0 or rows % 128 or cols % 128:
        raise ValueError("MXFP4 scale geometry requires 128-aligned rows and cols")
    mb, kb = rows // 128, cols // 128
    raw = scales.detach().contiguous().view(torch.uint8)
    if raw.numel() != rows * (cols // 32):
        raise RuntimeError(
            f"MXFP4 scale payload has {raw.numel()} bytes, "
            f"expected {rows * (cols // 32)}"
        )
    return (
        raw.reshape(mb, kb, 32, 4, 4)
        .transpose(-2, -3)
        .reshape(mb, kb, 128, 4)
        .transpose(1, 2)
        .reshape(rows, kb * 4)
        .contiguous()
    )


def swizzle_mxfp4_scales_for_test(
    logical: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """Inverse of :func:`unswizzle_mxfp4_scales`, used by CPU oracle tests."""

    if rows % 128 or cols % 128:
        raise ValueError("MXFP4 scale geometry requires 128-aligned rows and cols")
    mb, kb = rows // 128, cols // 128
    if list(logical.shape) != [rows, cols // 32]:
        raise RuntimeError("logical MXFP4 scale matrix has the wrong shape")
    return (
        logical.contiguous()
        .reshape(mb, 128, kb, 4)
        .transpose(1, 2)
        .reshape(mb, kb, 4, 32, 4)
        .transpose(-2, -3)
        .reshape(mb, kb, 32, 16)
        .contiguous()
    )


@dataclass(frozen=True)
class MXFP4Decoded:
    values: torch.Tensor
    codes: torch.Tensor
    scale_codes: torch.Tensor
    maximum: torch.Tensor
    minimum_positive: torch.Tensor


def decode_mxfp4(
    packed: torch.Tensor, scales: torch.Tensor, rows: int, cols: int
) -> MXFP4Decoded:
    """Decode production MXFP4 payloads into original BF16 operand units.

    The quantizer multiplies each value by ``6 / 2**(e-127)``.  Consequently,
    the inverse includes ``/ 6``; the paired GEMM applies ``1 / 36``.
    """

    codes = unpack_e2m1_codes(packed, rows, cols)
    scale_codes = unswizzle_mxfp4_scales(scales, rows, cols)
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=codes.device)
    values = lut[codes.long()]
    block_maximum = torch.exp2(scale_codes.float() - 127.0)
    maximum = block_maximum.repeat_interleave(32, dim=1)
    decoded = values * (maximum / 6.0)
    return MXFP4Decoded(
        values=decoded,
        codes=codes,
        scale_codes=scale_codes,
        maximum=maximum,
        minimum_positive=maximum / 12.0,
    )


def hadamard_matrix(
    block_size: int, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    result = torch.ones((1, 1), dtype=torch.float32, device=device)
    while result.shape[0] < block_size:
        result = torch.cat(
            (
                torch.cat((result, result), dim=1),
                torch.cat((result, -result), dim=1),
            ),
            dim=0,
        )
    return result / math.sqrt(block_size)


def decode_positive_sign_mask(mask: int, block_size: int) -> tuple[int, ...]:
    if mask < 0 or mask >= 1 << block_size:
        raise ValueError("RHT sign mask is outside the selected block size")
    return tuple(1 if (mask >> index) & 1 else -1 for index in range(block_size))


def block_rht(
    value: torch.Tensor,
    *,
    block_size: int,
    sign_mode: str,
) -> torch.Tensor:
    """Apply QX, Q=H diag(sign), over adjacent row blocks."""

    if value.ndim != 2 or value.shape[0] % block_size:
        raise RuntimeError("RHT input must be a matrix with block-aligned rows")
    if sign_mode == "fixed-0x2817":
        if block_size != 16:
            raise RuntimeError("fixed 0x2817 signs are defined only for block 16")
        signs = FIXED_RHT_SIGNS
    elif sign_mode == "plain":
        signs = (1,) * block_size
    else:
        raise ValueError(f"unsupported RHT sign mode: {sign_mode}")
    matrix = hadamard_matrix(block_size, device=value.device)
    sign_tensor = torch.tensor(signs, dtype=torch.float32, device=value.device)
    q = matrix * sign_tensor.unsqueeze(0)
    source = value.float().reshape(-1, block_size, value.shape[1])
    transformed = torch.matmul(q.unsqueeze(0), source)
    # The kernels load BF16 inputs, but execute the butterfly and quantization
    # scale arithmetic in FP32.  Keep this oracle in FP32; rounding it back to
    # BF16 would incorrectly charge carrier-rounding error to the FP4 payload.
    return transformed.reshape_as(value)


def run_cpu_oracle_gates() -> dict[str, Any]:
    """Run decoder/swizzle and paired-RHT math gates without CUDA."""

    if decode_positive_sign_mask(FIXED_RHT_POSITIVE_MASK, 16) != FIXED_RHT_SIGNS:
        raise RuntimeError("fixed RHT sign-mask convention drifted")

    rows = cols = 128
    logical_scales = (
        torch.arange(rows * (cols // 32), dtype=torch.int64).reshape(
            rows, cols // 32
        )
        % 40
        + 100
    ).to(torch.uint8)
    swizzled = swizzle_mxfp4_scales_for_test(logical_scales, rows, cols)
    restored = unswizzle_mxfp4_scales(swizzled, rows, cols)
    if not torch.equal(restored, logical_scales):
        raise RuntimeError("MXFP4 scale swizzle roundtrip failed")

    codes = torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols) & 0x0F
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    decoded = decode_mxfp4(packed, swizzled, rows, cols)
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32)
    expected_maximum = torch.exp2(logical_scales.float() - 127.0).repeat_interleave(
        32, dim=1
    )
    expected = lut[codes.long()] * expected_maximum / 6.0
    decode_max_error = float((decoded.values - expected).abs().max())
    if decode_max_error != 0.0:
        raise RuntimeError(f"MXFP4 synthetic decode gate failed: {decode_max_error}")

    generator = torch.Generator().manual_seed(42)
    x = torch.randn(32, 19, generator=generator)
    dy = torch.randn(32, 11, generator=generator)
    x_rht = block_rht(x, block_size=16, sign_mode="fixed-0x2817").float()
    dy_rht = block_rht(dy, block_size=16, sign_mode="fixed-0x2817").float()
    contraction_error = float((dy.T @ x - dy_rht.T @ x_rht).abs().max())
    contraction_scale = max(float((dy.T @ x).abs().max()), 1.0)
    if contraction_error / contraction_scale > 1.0e-5:
        raise RuntimeError(
            f"paired BF16 RHT contraction gate failed: {contraction_error}"
        )
    q = hadamard_matrix(16)
    orthogonality_error = float((q @ q.T - torch.eye(16)).abs().max())
    if orthogonality_error > 1.0e-6:
        raise RuntimeError(f"Hadamard orthogonality gate failed: {orthogonality_error}")
    return {
        "method": "cpu-decoder-and-rht-oracle-gates-v1",
        "fixed_positive_mask": f"0x{FIXED_RHT_POSITIVE_MASK:04x}",
        "fixed_signs": list(FIXED_RHT_SIGNS),
        "mxfp4_scale_swizzle_roundtrip": True,
        "mxfp4_synthetic_decode_max_abs_error": decode_max_error,
        "hadamard_orthogonality_max_abs_error": orthogonality_error,
        "paired_fp32_transform_contraction_max_abs_error": contraction_error,
        "paired_fp32_transform_contraction_relative_max": contraction_error
        / contraction_scale,
    }


def _json_float(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite diagnostic scalar: {result}")
    return result


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate_f = candidate.float()
    reference_f = reference.float()
    error_energy = torch.sum((candidate_f - reference_f).square(), dtype=torch.float64)
    reference_energy = torch.sum(reference_f.square(), dtype=torch.float64)
    return math.sqrt(_json_float(error_energy) / max(_json_float(reference_energy), TINY))


def _even_sample_tensors(
    values: Sequence[torch.Tensor], max_samples: int
) -> tuple[torch.Tensor, ...]:
    if not values or max_samples <= 0:
        raise ValueError("sample inputs and max_samples must be non-empty/positive")
    numel = values[0].numel()
    if any(value.numel() != numel for value in values):
        raise RuntimeError("sampled tensors have different element counts")
    count = min(numel, max_samples)
    if count == numel:
        return tuple(value.detach().reshape(-1).float().cpu() for value in values)
    positions = torch.arange(count, device=values[0].device, dtype=torch.int64)
    positions = torch.div(positions * numel, count, rounding_mode="floor")
    return tuple(
        value.detach().reshape(-1).index_select(0, positions).float().cpu()
        for value in values
    )


def _quantile_dict(value: torch.Tensor) -> dict[str, float]:
    if value.numel() == 0:
        return {name: 0.0 for name in ("p0", "p50", "p90", "p99", "p999", "max")}
    probabilities = torch.tensor((0.0, 0.5, 0.9, 0.99, 0.999, 1.0), dtype=torch.float64)
    result = torch.quantile(value.double(), probabilities)
    return dict(zip(("p0", "p50", "p90", "p99", "p999", "max"), map(float, result)))


def _raw_byte_histogram(value: torch.Tensor) -> dict[str, int]:
    raw = value.detach().contiguous().view(torch.uint8).reshape(-1)
    counts = torch.bincount(raw.long(), minlength=256).cpu()
    return {str(index): int(count) for index, count in enumerate(counts) if int(count)}


def _outer_scale_summary(values: Sequence[torch.Tensor]) -> list[dict[str, Any]]:
    result = []
    for value in values:
        sample = value.detach().float().reshape(-1).cpu()
        if not bool(torch.isfinite(sample).all()):
            raise RuntimeError("localCTA outer scale contains non-finite values")
        result.append(
            {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_payload_sha256(value),
                "quantiles": _quantile_dict(sample.abs()),
            }
        )
    return result


def direct_operand_metrics(
    reference: torch.Tensor,
    decoded: torch.Tensor,
    maximum: torch.Tensor,
    minimum_positive: torch.Tensor,
    codes: torch.Tensor,
    *,
    scale_payload: torch.Tensor,
    scale_kind: str,
    profile_samples: int,
) -> dict[str, Any]:
    """Measure exact whole-tensor zero/range counts plus sampled distributions."""

    expected_shape = tuple(reference.shape)
    if any(tuple(value.shape) != expected_shape for value in (decoded, maximum, minimum_positive, codes)):
        raise RuntimeError("operand metric tensors have different logical shapes")
    reference_f = reference.detach().float()
    decoded_f = decoded.detach().float()
    maximum_f = maximum.detach().float().abs()
    minimum_f = minimum_positive.detach().float().abs()
    if not bool(
        torch.isfinite(reference_f).all()
        and torch.isfinite(decoded_f).all()
        and torch.isfinite(maximum_f).all()
        and torch.isfinite(minimum_f).all()
    ):
        raise RuntimeError("operand metrics received a non-finite tensor")
    if bool((maximum_f <= 0).any()) or bool((minimum_f <= 0).any()):
        raise RuntimeError("decoded representable range is not positive")

    reference_energy = torch.sum(reference_f.square(), dtype=torch.float64)
    error = decoded_f - reference_f
    error_energy = torch.sum(error.square(), dtype=torch.float64)
    decoded_energy = torch.sum(decoded_f.square(), dtype=torch.float64)
    cross = torch.sum(decoded_f * reference_f, dtype=torch.float64)
    nonzero = reference_f != 0
    decoded_zero = decoded_f == 0
    zeroed = nonzero & decoded_zero
    below_min = nonzero & (reference_f.abs() < minimum_f)
    clipped = reference_f.abs() > maximum_f
    saturated_output = decoded_f.abs() >= maximum_f
    sign_flip = nonzero & ~decoded_zero & (torch.signbit(reference_f) != torch.signbit(decoded_f))
    code_magnitude = codes.to(torch.uint8) & 0x07
    max_code = code_magnitude == 0x07

    nonzero_count = int(nonzero.sum())
    zeroed_count = int(zeroed.sum())
    clipped_count = int(clipped.sum())
    total_energy = max(_json_float(reference_energy), TINY)
    decoded_norm = math.sqrt(max(_json_float(decoded_energy), 0.0))
    reference_norm = math.sqrt(max(_json_float(reference_energy), 0.0))

    reference_sample, decoded_sample, maximum_sample, minimum_sample = _even_sample_tensors(
        (reference_f, decoded_f, maximum_f, minimum_f), profile_samples
    )
    sample_nonzero = reference_sample != 0
    headroom = reference_sample.abs() / maximum_sample.clamp_min(1.0e-38)
    nonzero_magnitude = reference_sample.abs()[sample_nonzero]
    bin_records: list[dict[str, Any]] = []
    if nonzero_magnitude.numel():
        cuts = torch.quantile(
            nonzero_magnitude.double(),
            torch.tensor((0.0, 0.1, 0.5, 0.9, 0.99, 1.0), dtype=torch.float64),
        ).float()
        names = ("q0_q10", "q10_q50", "q50_q90", "q90_q99", "q99_q100")
        for index, name in enumerate(names):
            lower, upper = cuts[index], cuts[index + 1]
            mask = sample_nonzero & (reference_sample.abs() >= lower)
            mask &= (
                reference_sample.abs() <= upper
                if index == len(names) - 1
                else reference_sample.abs() < upper
            )
            count = int(mask.sum())
            bin_records.append(
                {
                    "name": name,
                    "abs_lower": float(lower),
                    "abs_upper": float(upper),
                    "count": count,
                    "decoded_zero_fraction": (
                        float((decoded_sample[mask] == 0).double().mean()) if count else 0.0
                    ),
                    "relative_l2": (
                        _relative_l2(decoded_sample[mask], reference_sample[mask])
                        if count
                        else 0.0
                    ),
                }
            )

    return {
        "finite": True,
        "shape": list(reference.shape),
        "numel": reference.numel(),
        "reference_dtype": str(reference.dtype),
        "decoded_dtype": str(decoded.dtype),
        "relative_l2": math.sqrt(_json_float(error_energy) / total_energy),
        "cosine": _json_float(cross)
        / max(reference_norm * decoded_norm, TINY),
        "rms_ratio": decoded_norm / max(reference_norm, TINY),
        "mean_error": _json_float(error.mean()),
        "reference_zero_fraction": _json_float((~nonzero).double().mean()),
        "decoded_zero_fraction": _json_float(decoded_zero.double().mean()),
        "reference_nonzero_to_decoded_zero": {
            "count": zeroed_count,
            "fraction_of_reference_nonzero": zeroed_count / max(nonzero_count, 1),
            "lost_reference_energy_fraction": _json_float(
                torch.sum(reference_f[zeroed].square(), dtype=torch.float64)
            )
            / total_energy,
        },
        "reference_below_minimum_positive": {
            "count": int(below_min.sum()),
            "fraction_of_reference_nonzero": int(below_min.sum())
            / max(nonzero_count, 1),
        },
        "true_source_range_exceedance": {
            "count": clipped_count,
            "fraction": clipped_count / reference.numel(),
            "reference_energy_fraction": _json_float(
                torch.sum(reference_f[clipped].square(), dtype=torch.float64)
            )
            / total_energy,
        },
        "max_code_occupancy": {
            "count": int(max_code.sum()),
            "fraction": _json_float(max_code.double().mean()),
            "note": "max-code occupancy is not itself proof of clipping",
        },
        "decoded_at_range_boundary_fraction": _json_float(
            saturated_output.double().mean()
        ),
        "sign_flip_excluding_decoded_zero_fraction": int(sign_flip.sum())
        / max(nonzero_count - zeroed_count, 1),
        "sampled_profiles": {
            "sampling": "even-index-v1",
            "sample_numel": reference_sample.numel(),
            "reference_abs": _quantile_dict(reference_sample.abs()),
            "representable_maximum": _quantile_dict(maximum_sample),
            "minimum_positive": _quantile_dict(minimum_sample),
            "source_to_maximum_headroom": _quantile_dict(headroom),
            "reference_magnitude_bins": bin_records,
        },
        "payload": {
            "fp4_code_histogram": {
                str(index): int(count)
                for index, count in enumerate(
                    torch.bincount(codes.reshape(-1).long(), minlength=16).cpu()
                )
                if int(count)
            },
            "scale_kind": scale_kind,
            "scale_shape": list(scale_payload.shape),
            "scale_dtype": str(scale_payload.dtype),
            "scale_sha256": tensor_payload_sha256(scale_payload),
            "scale_raw_byte_histogram": _raw_byte_histogram(scale_payload),
        },
    }


def rht_oracle_gate(
    decoded: torch.Tensor,
    raw_reference: torch.Tensor,
    *,
    expected_sign_mode: str,
    block_size: int,
    minimum_margin: float,
) -> dict[str, Any]:
    """Prove which fixed RHT geometry the reconstructed column follows."""

    candidates = {
        "raw": raw_reference,
        "plain": block_rht(raw_reference, block_size=block_size, sign_mode="plain"),
    }
    if block_size == 16:
        candidates["fixed-0x2817"] = block_rht(
            raw_reference, block_size=block_size, sign_mode="fixed-0x2817"
        )
    expected_name = expected_sign_mode
    if expected_name not in candidates:
        raise RuntimeError(f"RHT oracle has no expected candidate {expected_name}")
    errors = {name: _relative_l2(decoded, value) for name, value in candidates.items()}
    ordered = sorted(errors.items(), key=lambda item: item[1])
    winner, best = ordered[0]
    second = ordered[1][1]
    margin = second / max(best, TINY)
    passed = winner == expected_name and margin >= minimum_margin
    result = {
        "expected": expected_name,
        "winner": winner,
        "relative_l2_by_candidate": errors,
        "second_to_best_error_ratio": margin,
        "minimum_margin": minimum_margin,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"RHT geometry oracle failed: {result}")
    return result


@dataclass
class QuantizedPair:
    row_fp4: torch.Tensor
    row_sc: torch.Tensor
    col_fp4: torch.Tensor
    col_sc: torch.Tensor
    row_outer: tuple[torch.Tensor, ...]
    col_outer: tuple[torch.Tensor, ...]
    keepalive: tuple[torch.Tensor, ...]
    producer: dict[str, Any]


def _localcta_pair(result: Sequence[Any], producer: Mapping[str, Any]) -> QuantizedPair:
    if len(result) < 6 or any(not isinstance(result[index], torch.Tensor) for index in range(6)):
        raise RuntimeError("localCTA producer returned an invalid row/column payload")
    return QuantizedPair(
        row_fp4=result[0],
        row_sc=result[1],
        col_fp4=result[2],
        col_sc=result[3],
        row_outer=(result[4],),
        col_outer=(result[5],),
        keepalive=tuple(
            value for value in result[6:] if isinstance(value, torch.Tensor)
        ),
        producer=dict(producer),
    )


def _allocate_mxfp4_pair(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if value.ndim != 2 or value.shape[0] % 128 or value.shape[1] % 128:
        raise RuntimeError("MXFP4 producer input must be a 128-aligned matrix")
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError("this Torch build does not expose float4_e2m1fn_x2")
    rows, cols = value.shape
    return (
        torch.empty((rows, cols // 2), device=value.device, dtype=fp4_dtype),
        torch.empty(
            (rows // 128, cols // 128, 32, 16),
            device=value.device,
            dtype=torch.uint8,
        ),
        torch.empty((cols, rows // 2), device=value.device, dtype=fp4_dtype),
        torch.empty(
            (cols // 128, rows // 128, 32, 16),
            device=value.device,
            dtype=torch.uint8,
        ),
    )


def _mxfp4_pair(result: Sequence[Any], producer: Mapping[str, Any]) -> QuantizedPair:
    if len(result) != 4 or any(not isinstance(value, torch.Tensor) for value in result):
        raise RuntimeError("MXFP4 producer returned an invalid row/column payload")
    return QuantizedPair(
        row_fp4=result[0],
        row_sc=result[1],
        col_fp4=result[2],
        col_sc=result[3],
        row_outer=(),
        col_outer=(),
        keepalive=(),
        producer=dict(producer),
    )


def _site_coordinate(args: argparse.Namespace, layer: int, site: str) -> tuple[int, int, str]:
    key = f"layer_{layer}.{site}"
    if args.rng_coordinate_ledger is None:
        return (
            args.rng_seed,
            args.rng_subsequence_base + layer * args.rng_layer_stride,
            "diagnostic-cli-coordinate-v1",
        )
    ledger = load_json_object(args.rng_coordinate_ledger)
    coordinates = ledger.get("coordinates")
    if ledger.get("schema_version") != 1 or not isinstance(coordinates, dict):
        raise RuntimeError("RNG coordinate ledger has an invalid schema")
    record = coordinates.get(key)
    if not isinstance(record, dict):
        raise RuntimeError(f"RNG coordinate ledger has no entry {key}")
    seed, subsequence = record.get("seed"), record.get("subsequence")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(subsequence, bool)
        or not isinstance(subsequence, int)
        or seed < 0
        or subsequence < 0
    ):
        raise RuntimeError(f"RNG coordinate ledger entry is invalid: {key}")
    return seed, subsequence, "sealed-external-coordinate-ledger-v1"


def _localcta_quantize_site(
    module: Any,
    route: str,
    site: str,
    source: Mapping[str, torch.Tensor],
    *,
    seed: int,
    subsequence: int,
) -> QuantizedPair:
    use_rht = route == "localcta-rht"
    if site == "w2_activation":
        if use_rht:
            if not bool(module.tk_localcta_silu_supports_paired_col_rht()):
                raise RuntimeError("localCTA extension rejects paired W2 column RHT")
            symbol = "tk_localcta_silu_quantize_split_for_gemm_paired_col_rht"
        else:
            symbol = "tk_localcta_silu_quantize_split_for_gemm"
        result = getattr(module, symbol)(source["gate"], source["up"])
        return _localcta_pair(
            result,
            {
                "symbol": symbol,
                "role": "activation",
                "encode_centric": True,
                "data_sr": False,
                "scale_sr": False,
                "column_rht": use_rht,
                "weight_rht": False,
            },
        )
    if site == "w2_dy":
        symbol = "tk_localcta_quantize_for_gemm_opt"
        result = getattr(module, symbol)(
            source["value"],
            True,
            False,
            True,
            False,
            "col" if use_rht else "none",
            use_rht,
            seed,
            subsequence,
            "row",
        )
        return _localcta_pair(
            result,
            {
                "symbol": symbol,
                "role": "grad",
                "encode_centric": False,
                "data_sr": True,
                "data_sr_axes": "row",
                "scale_sr": False,
                "rht_axes": "col" if use_rht else "none",
                "with_fixed_sign_mask": use_rht,
                "rng_seed": seed,
                "rng_subsequence": subsequence,
            },
        )
    if site == "down_weight":
        symbol = "tk_localcta_quantize_weight_2d"
        result = getattr(module, symbol)(source["value"])
        return _localcta_pair(
            result,
            {
                "symbol": symbol,
                "role": "weight",
                "two_dimensional_weight_quantization": True,
                "data_sr": False,
                "scale_sr": False,
                "weight_rht": False,
            },
        )
    raise ValueError(site)


def _mxfp4_quantize_site(
    module: Any,
    route: str,
    site: str,
    source: Mapping[str, torch.Tensor],
    *,
    seed: int,
    subsequence: int,
    rht_block_size: int,
    rht_sign_mode: str,
) -> QuantizedPair:
    use_rht = route == "mxfp4-rht"
    sign_flag = use_rht and rht_sign_mode == "fixed-0x2817"
    if site == "w2_activation":
        symbol = "mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace"
        outputs = _allocate_mxfp4_pair(source["value"])
        getattr(module, symbol)(
            source["gate"],
            source["up"],
            *outputs,
            1,
            False,
            False,
            use_rht,
            rht_block_size,
            sign_flag,
            seed,
            subsequence,
            False,
        )
        return _mxfp4_pair(
            outputs,
            {
                "symbol": symbol,
                "role": "activation",
                "mode": 1,
                "data_sr": False,
                "scale_sr": False,
                "column_rht": use_rht,
                "row_rht": False,
                "producer_sign_flag": sign_flag,
                "rng_seed": seed,
                "rng_subsequence": subsequence,
            },
        )
    if site == "w2_dy":
        row_symbol = "mxfp4_quantize_for_gemm_opt"
        row_fp4, row_sc = getattr(module, row_symbol)(
            source["value"], 1, True, False, seed, subsequence
        )
        if use_rht:
            col_symbol = "mxfp4_quantize_col_only_opt_rht"
            col_fp4, col_sc = getattr(module, col_symbol)(
                source["value"],
                1,
                False,
                False,
                rht_block_size,
                sign_flag,
                seed,
                subsequence,
            )
        else:
            col_symbol = "mxfp4_quantize_col_only"
            col_fp4, col_sc = getattr(module, col_symbol)(source["value"], 1)
        return _mxfp4_pair(
            (row_fp4, row_sc, col_fp4, col_sc),
            {
                "row_symbol": row_symbol,
                "col_symbol": col_symbol,
                "role": "grad",
                "mode": 1,
                "row_data_sr": True,
                "column_data_sr": False,
                "scale_sr": False,
                "column_rht": use_rht,
                "row_rht": False,
                "producer_sign_flag": sign_flag,
                "rng_seed": seed,
                "rng_subsequence": subsequence,
            },
        )
    if site == "down_weight":
        symbol = "mxfp4_quantize_weight_2d"
        result = getattr(module, symbol)(source["value"])
        return _mxfp4_pair(
            result,
            {
                "symbol": symbol,
                "role": "weight",
                "mode": 1,
                "two_dimensional_weight_quantization": True,
                "data_sr": False,
                "scale_sr": False,
                "weight_rht": False,
            },
        )
    raise ValueError(site)


def _localcta_decode(
    module: Any, pair: QuantizedPair, orientation: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if orientation == "row":
        symbol = "tk_localcta_reconstruct_row"
        fp4, sc, outer = pair.row_fp4, pair.row_sc, pair.row_outer[0]
    elif orientation == "col":
        symbol = "tk_localcta_reconstruct_col"
        fp4, sc, outer = pair.col_fp4, pair.col_sc, pair.col_outer[0]
    else:
        raise ValueError(orientation)
    decoder = getattr(module, symbol)
    decoded = decoder(fp4, sc, outer)
    max_payload = torch.empty_like(fp4)
    max_payload.view(torch.uint8).fill_(0x77)
    minimum_payload = torch.empty_like(fp4)
    minimum_payload.view(torch.uint8).fill_(0x11)
    maximum = decoder(max_payload, sc, outer).abs()
    minimum = decoder(minimum_payload, sc, outer).abs()
    codes = unpack_e2m1_codes(fp4, decoded.shape[0], decoded.shape[1])
    if orientation == "col":
        decoded = decoded.T.contiguous()
        maximum = maximum.T.contiguous()
        minimum = minimum.T.contiguous()
        codes = codes.T.contiguous()
    if bool((decoded.abs() > maximum * 1.0001).any()):
        raise RuntimeError("localCTA reconstruction exceeds its all-max-code ceiling")
    return decoded, maximum, minimum, codes


def _mxfp4_decode_pair(
    pair: QuantizedPair, orientation: str, logical_shape: Sequence[int]
) -> MXFP4Decoded:
    rows, cols = map(int, logical_shape)
    if orientation == "row":
        return decode_mxfp4(pair.row_fp4, pair.row_sc, rows, cols)
    if orientation != "col":
        raise ValueError(orientation)
    physical = decode_mxfp4(pair.col_fp4, pair.col_sc, cols, rows)
    return MXFP4Decoded(
        values=physical.values.T.contiguous(),
        codes=physical.codes.T.contiguous(),
        scale_codes=physical.scale_codes,
        maximum=physical.maximum.T.contiguous(),
        minimum_positive=physical.minimum_positive.T.contiguous(),
    )


def _capture_names_for_sites(sites: Sequence[str]) -> set[str]:
    names: set[str] = set()
    if "w2_activation" in sites:
        names.update(("ffn.gate_linear_ref", "ffn.up_linear_ref", "ffn.w2_input_ref"))
    if "w2_dy" in sites:
        names.add("ffn.output_ref_grad")
    return names


def _source_for_site(
    site: str,
    capture: Mapping[str, torch.Tensor],
    weight: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if site == "w2_activation":
        gate = capture["ffn.gate_linear_ref"].to(device=device).contiguous()
        up = capture["ffn.up_linear_ref"].to(device=device).contiguous()
        value = capture["ffn.w2_input_ref"].to(device=device).contiguous()
        recomputed = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        parity = {
            "python_bf16_swiglu_relative_l2_to_capture": _relative_l2(
                recomputed, value
            ),
            "python_bf16_swiglu_exact_equal_to_capture": bool(
                torch.equal(recomputed, value)
            ),
        }
        return {"gate": gate, "up": up, "value": value}, parity
    if site == "w2_dy":
        value = capture["ffn.output_ref_grad"].to(device=device).contiguous()
        return {"value": value}, {}
    if site == "down_weight":
        return {"value": weight.to(device=device).contiguous()}, {}
    raise ValueError(site)


def _payload_hashes(pair: QuantizedPair) -> dict[str, Any]:
    tensors = {
        "row_fp4": pair.row_fp4,
        "row_sc": pair.row_sc,
        "col_fp4": pair.col_fp4,
        "col_sc": pair.col_sc,
        **{f"row_outer_{index}": value for index, value in enumerate(pair.row_outer)},
        **{f"col_outer_{index}": value for index, value in enumerate(pair.col_outer)},
        **{f"keepalive_{index}": value for index, value in enumerate(pair.keepalive)},
    }
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": tensor_payload_sha256(value),
        }
        for name, value in tensors.items()
    }


def _run_layer(
    args: argparse.Namespace,
    module: Any,
    layer: int,
    capture: Mapping[str, torch.Tensor],
    capture_evidence: Mapping[str, Any],
    weight: torch.Tensor,
    weight_evidence: Mapping[str, Any],
    sites: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "route": args.route,
        "layer": layer,
        "capture_files": dict(capture_evidence),
        "weight": dict(weight_evidence),
        "sites": {},
    }
    for site in sites:
        source, source_parity = _source_for_site(site, capture, weight, device)
        reference = source["value"]
        seed, subsequence, coordinate_method = _site_coordinate(args, layer, site)
        if args.route.startswith("localcta"):
            pair = _localcta_quantize_site(
                module,
                args.route,
                site,
                source,
                seed=seed,
                subsequence=subsequence,
            )
        else:
            pair = _mxfp4_quantize_site(
                module,
                args.route,
                site,
                source,
                seed=seed,
                subsequence=subsequence,
                rht_block_size=args.rht_block_size,
                rht_sign_mode=args.rht_sign_mode,
            )

        site_result: dict[str, Any] = {
            "source": {
                "name": {
                    "w2_activation": "ffn.w2_input_ref",
                    "w2_dy": "ffn.output_ref_grad",
                    "down_weight": weight_evidence["tensor_name"],
                }[site],
                "dtype": str(reference.dtype),
                "shape": list(reference.shape),
                "sha256": tensor_sha256(reference),
                "parity": source_parity,
            },
            "rng_coordinate": {
                "method": coordinate_method,
                "seed": seed,
                "subsequence": subsequence,
            },
            "producer": pair.producer,
            "payload_hashes": _payload_hashes(pair),
            "orientations": {},
        }
        use_column_rht = args.route.endswith("-rht") and site != "down_weight"
        for orientation in ("row", "col"):
            if args.route.startswith("localcta"):
                decoded, maximum, minimum, codes = _localcta_decode(
                    module, pair, orientation
                )
                scale_payload = pair.row_sc if orientation == "row" else pair.col_sc
                scale_kind = "localcta-e4m3-local-scale"
                outer_scales = pair.row_outer if orientation == "row" else pair.col_outer
                scale_extra = {"outer_scales": _outer_scale_summary(outer_scales)}
            else:
                decoded_mx = _mxfp4_decode_pair(pair, orientation, reference.shape)
                decoded = decoded_mx.values
                maximum = decoded_mx.maximum
                minimum = decoded_mx.minimum_positive
                codes = decoded_mx.codes
                scale_payload = pair.row_sc if orientation == "row" else pair.col_sc
                scale_kind = "mxfp4-e8m0"
                scale_extra = {
                    "logical_scale_code_shape": list(decoded_mx.scale_codes.shape),
                    "logical_scale_code_sha256": tensor_sha256(decoded_mx.scale_codes),
                }
            transformed = orientation == "col" and use_column_rht
            target = (
                block_rht(
                    reference,
                    block_size=args.rht_block_size,
                    sign_mode=args.rht_sign_mode,
                )
                if transformed
                else reference
            )
            metrics = direct_operand_metrics(
                target,
                decoded,
                maximum,
                minimum,
                codes,
                scale_payload=scale_payload,
                scale_kind=scale_kind,
                profile_samples=args.profile_samples,
            )
            metrics["scale_details"] = scale_extra
            expected_oracle = args.rht_sign_mode if transformed else "raw"
            metrics["geometry_oracle"] = rht_oracle_gate(
                decoded,
                reference,
                expected_sign_mode=expected_oracle,
                block_size=args.rht_block_size,
                minimum_margin=args.oracle_margin_min,
            )
            metrics["logical_contract"] = {
                "orientation": orientation,
                "column_rht": transformed,
                "target": (
                    f"block-{args.rht_block_size}-{args.rht_sign_mode}-QX"
                    if transformed
                    else "raw-bf16-operand"
                ),
            }
            site_result["orientations"][orientation] = metrics
        result["sites"][site] = site_result
        del pair, source
    torch.cuda.synchronize(device)
    return result


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    if args.rht_block_size != 16:
        raise RuntimeError("v1 microscope is sealed to the production block-16 RHT")
    cpu_gates = run_cpu_oracle_gates()
    route_environment = configure_route_environment(
        args.route, localcta_scale_num=args.localcta_scale_num
    )
    reader = AlignedCaptureReader(
        args.capture_dir, rht_block_size=args.rht_block_size
    )
    layers = _parse_layer_spec(args.layers, reader.num_layers)
    sites = _parse_sites(args.sites)
    store = ConvertedDownWeightStore(
        args.converted, verify_shards=args.verify_loaded_shards
    )
    _check_capture_conversion_binding(reader.manifest, store.summary)
    source_contract = audit_runtime_source_contract(
        args.runtime_source_root, args.route
    )
    extension_path, module_name = _find_quant_extension(
        args.runtime_root, args.route
    )
    extension_digest = sha256_file(extension_path)
    if extension_digest != args.quant_extension_sha256:
        raise RuntimeError(
            "quant extension SHA mismatch: "
            f"expected={args.quant_extension_sha256} actual={extension_digest}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production operand microscope requires a CUDA device")
    torch.cuda.set_device(device)
    module = _load_extension(extension_path, module_name)
    _, symbols = _required_source_symbols(args.route)
    missing_binary_symbols = [symbol for symbol in symbols if not hasattr(module, symbol)]
    if missing_binary_symbols:
        raise RuntimeError(
            f"loaded quant extension is missing symbols: {missing_binary_symbols}"
        )
    if args.route.startswith("localcta"):
        module.tk_localcta_set_global_scale_num(args.localcta_scale_num)

    required_capture_names = _capture_names_for_sites(sites)
    with staged_output_directory(args.output) as staging:
        result_files: dict[str, Any] = {}
        for layer in layers:
            capture, capture_evidence = reader.load_layer(
                layer, required_capture_names
            )
            weight, weight_evidence = store.load(layer)
            layer_result = _run_layer(
                args,
                module,
                layer,
                capture,
                capture_evidence,
                weight,
                weight_evidence,
                sites,
                device,
            )
            filename = f"layer_{layer:02d}.json"
            path = staging / filename
            path.write_bytes(canonical_json_bytes(layer_result) + b"\n")
            result_files[filename] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            del capture, weight, layer_result
            torch.cuda.empty_cache()

        rng_ledger = None
        if args.rng_coordinate_ledger is not None:
            rng_ledger = {
                "path": str(args.rng_coordinate_ledger.resolve()),
                "sha256": sha256_file(args.rng_coordinate_ledger),
            }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "route": args.route,
            "layers": layers,
            "sites": list(sites),
            "capture": {
                "root": str(args.capture_dir.resolve()),
                "manifest_sha256": reader.manifest_sha256,
                "receipt_sha256": reader.manifest["receipt_sha256"],
                "alignment": reader.alignment,
            },
            "converted": {"root": str(args.converted.resolve()), **store.summary},
            "source": _git_identity(REPO_ROOT),
            "runtime_source": {
                **source_contract,
                "git": _git_identity(args.runtime_source_root),
            },
            "runtime_binary": {
                "root": str(args.runtime_root.resolve()),
                "extension": str(extension_path),
                "extension_bytes": extension_path.stat().st_size,
                "extension_sha256": extension_digest,
                "module_name": module_name,
                "required_symbols_present": list(symbols),
            },
            "recipe": {
                "rht_block_size": args.rht_block_size,
                "rht_sign_mode": args.rht_sign_mode,
                "fixed_positive_mask": f"0x{FIXED_RHT_POSITIVE_MASK:04x}",
                "localcta_scale_num": (
                    args.localcta_scale_num
                    if args.route.startswith("localcta")
                    else None
                ),
                "rng_seed_default": args.rng_seed,
                "rng_subsequence_base": args.rng_subsequence_base,
                "rng_layer_stride": args.rng_layer_stride,
                "rng_coordinate_ledger": rng_ledger,
                "profile_samples": args.profile_samples,
                "oracle_margin_min": args.oracle_margin_min,
            },
            "route_environment": route_environment,
            "cpu_oracle_gates": cpu_gates,
            "hardware": {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "result_files": result_files,
            "result_file_ledger_sha256": sha256(
                canonical_json_bytes(result_files)
            ).hexdigest(),
            "scientific_scope": {
                "direct_production_quantized_operands": True,
                "full_training_resume": False,
                "optimizer_state_used": False,
                "aligned_complete_rht_blocks_required": True,
                "payloads_persisted": False,
                "row_and_column_orientations_reported_separately": True,
                "max_code_occupancy_is_not_labeled_clipping": True,
            },
        }
        manifest = seal_receipt(payload)
        (staging / "result_manifest.json").write_bytes(
            canonical_json_bytes(manifest) + b"\n"
        )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "route": args.route,
                "layers": layers,
                "sites": list(sites),
                "receipt_sha256": manifest["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--quant-extension-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--sites", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rht-block-size", type=int, default=DEFAULT_RHT_BLOCK_SIZE)
    parser.add_argument(
        "--rht-sign-mode", choices=("fixed-0x2817", "plain"), default="fixed-0x2817"
    )
    parser.add_argument("--localcta-scale-num", type=float, default=448.0)
    parser.add_argument("--rng-seed", type=int, default=1 << 40)
    parser.add_argument("--rng-subsequence-base", type=int, default=0)
    parser.add_argument("--rng-layer-stride", type=int, default=1 << 40)
    parser.add_argument("--rng-coordinate-ledger", type=Path)
    parser.add_argument("--profile-samples", type=int, default=262144)
    parser.add_argument("--oracle-margin-min", type=float, default=2.0)
    parser.add_argument(
        "--verify-loaded-shards", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.profile_samples <= 0:
        parser.error("--profile-samples must be positive")
    if args.oracle_margin_min <= 1.0:
        parser.error("--oracle-margin-min must exceed 1")
    if args.rng_seed < 0 or args.rng_subsequence_base < 0 or args.rng_layer_stride <= 0:
        parser.error("RNG seed/subsequence values must be nonnegative and stride positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
