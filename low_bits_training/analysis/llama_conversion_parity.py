# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Receipt and logit checks for route-aware Llama checkpoint conversion.

This module contains the small, dependency-light portion of the parity gate so
that receipt sealing and validation can be tested on CPU.  The executable gate
which constructs the pinned TorchTitan and Transformers models lives in
``scripts/evaluation/validate_llama8b_conversion_parity.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import torch

from low_bits_training.analysis.llama_checkpoint_routes import (
    BF16_UNFUSED,
    LOCALCTA_FUSED,
    MXFP4_FUSED,
    PURE_V5_FUSED,
    TE_NATIVE_NVFP4_UNFUSED,
)


PARITY_RECEIPT_SCHEMA_VERSION = 2
PARITY_METHOD = "pinned-torchtitan-native-plus-exact-state-r12"
PARITY_POLICY = "llama8b-canonical-10-token-logits-r12"
PINNED_TORCHTITAN_COMMIT = "20b3de7585696c327bd5aa9f9627f0300abdbf9d"
PINNED_TRANSFORMERS_VERSION = "4.48.2"
SUPPORTED_ROUTES = {
    PURE_V5_FUSED,
    LOCALCTA_FUSED,
    MXFP4_FUSED,
    BF16_UNFUSED,
    TE_NATIVE_NVFP4_UNFUSED,
}
SOURCE_TENSORS_BY_ROUTE = {
    PURE_V5_FUSED: 227,
    LOCALCTA_FUSED: 227,
    MXFP4_FUSED: 227,
    BF16_UNFUSED: 291,
    TE_NATIVE_NVFP4_UNFUSED: 291,
}
FROZEN_ALIASES_BY_ROUTE = {
    PURE_V5_FUSED: 64,
    LOCALCTA_FUSED: 64,
    MXFP4_FUSED: 0,
    BF16_UNFUSED: 0,
    TE_NATIVE_NVFP4_UNFUSED: 0,
}
LLAMA8B_NATIVE_PARAMETERS = 291
LLAMA8B_NATIVE_ATTENTION_MODULES = 32
LLAMA8B_HF_TENSORS = 291
LLAMA8B_HF_ELEMENTS = 8_030_261_248
LLAMA8B_VOCAB_SIZE = 128_256
CANONICAL_FIXED_TOKEN_IDS = (
    128000,
    791,
    1489,
    374,
    264,
    1296,
    315,
    872,
    1344,
    13,
)
CANONICAL_FIXED_TOKEN_IDS_SHA256 = (
    "7efecfa934a69fc22e9cba559b9547061cc3a0f58a7bbaba256d6df41a335909"
)
CANONICAL_LOGITS_SHAPE = (1, len(CANONICAL_FIXED_TOKEN_IDS), LLAMA8B_VOCAB_SIZE)
PARITY_ENVIRONMENT_FIELDS = frozenset(
    {
        "python",
        "platform",
        "torch",
        "transformers",
        "safetensors",
        "cuda_runtime",
        "cudnn",
        "device",
        "torchtitan_commit",
        "project_git_commit",
        "project_tracked_dirty",
        "native_attention",
        "converted_attention",
        "attention_backend",
        "compute_dtype",
        "device_name",
        "compute_capability",
        "canonical_semantic_rope",
        "canonical_semantic_rmsnorm",
        "stock_hf_rope",
        "stock_hf_rmsnorm",
    }
)
PARITY_CODE_FILE_KEYS = {
    "parity_tool",
    "parity_receipt_module",
    "checkpoint_routes",
    "checkpoint_streamer",
    "torchtitan_llama_model",
    "torchtitan_llama_args",
    "torchtitan_llama_adapter",
    "torchtitan_attention",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_JOB_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")


@dataclass(frozen=True)
class ParityTolerances:
    """Frozen r12 numerical acceptance thresholds."""

    logit_atol: float = 0.125
    logit_rtol: float = 0.02
    max_close_failure_count: int = 1
    max_abs_error: float = 0.5
    max_mean_abs_error: float = 0.03125
    max_rms_error: float = 0.046875
    top_k: int = 10
    max_top_1_mismatch_count: int = 0
    min_top_k_intersection_count_per_position: int = 9
    min_top_k_intersection_count_total: int = 99

    def __post_init__(self) -> None:
        for name in (
            "logit_atol",
            "logit_rtol",
            "max_abs_error",
            "max_mean_abs_error",
            "max_rms_error",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")
        integer_fields = (
            "max_close_failure_count",
            "max_top_1_mismatch_count",
            "min_top_k_intersection_count_per_position",
            "min_top_k_intersection_count_total",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.min_top_k_intersection_count_per_position > self.top_k:
            raise ValueError("min_top_k_intersection_count_per_position exceeds top_k")
        position_count = CANONICAL_LOGITS_SHAPE[0] * CANONICAL_LOGITS_SHAPE[1]
        if self.min_top_k_intersection_count_total > position_count * self.top_k:
            raise ValueError("min_top_k_intersection_count_total is impossible")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticParityTolerances:
    """Exact acceptance thresholds for the TorchTitan-semantics HF path."""

    max_mismatched_elements: int = 0
    max_abs_error: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_mismatched_elements, bool)
            or not isinstance(self.max_mismatched_elements, int)
            or self.max_mismatched_elements < 0
        ):
            raise ValueError("max_mismatched_elements must be a nonnegative integer")
        if (
            not isinstance(self.max_abs_error, float)
            or not math.isfinite(self.max_abs_error)
            or self.max_abs_error < 0
        ):
            raise ValueError("max_abs_error must be a finite nonnegative float")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


CANONICAL_PARITY_TOLERANCES = ParityTolerances()
CANONICAL_SEMANTIC_TOLERANCES = SemanticParityTolerances()
PARITY_LOGIT_MEASUREMENT_FIELDS = frozenset(
    {
        "passed",
        "logits_shape",
        "logit_element_count",
        "strict_allclose",
        "close_success_count",
        "close_failure_count",
        "close_success_ratio",
        "close_failure_ratio",
        "max_abs_error",
        "mean_abs_error",
        "rms_error",
        "max_relative_error",
        "position_count",
        "top_1_match_count",
        "top_1_mismatch_count",
        "top_1_agreement_ratio",
        "top_k",
        "top_k_membership_count",
        "top_k_intersection_counts",
        "top_k_intersection_ratios",
        "top_k_intersection_count_min",
        "top_k_intersection_count_total",
        "top_k_intersection_ratio_min",
        "top_k_intersection_ratio_total",
        "reference_logits_sha256",
        "converted_logits_sha256",
    }
)
PARITY_STATE_MEASUREMENT_FIELDS = frozenset(
    {
        "source_tensors_streamed",
        "native_parameters_loaded",
        "converted_tensors_exact",
        "converted_elements_exact",
        "frozen_aliases_checked",
        "native_math_sdpa_modules",
    }
)
PARITY_SEMANTIC_MEASUREMENT_FIELDS = frozenset(
    {
        "passed",
        "logits_shape",
        "logit_element_count",
        "strict_bit_exact",
        "exact_match_count",
        "mismatch_count",
        "exact_match_ratio",
        "mismatch_ratio",
        "max_abs_error",
        "mean_abs_error",
        "rms_error",
        "reference_logits_sha256",
        "canonical_logits_sha256",
    }
)
PARITY_TOLERANCE_FIELDS = frozenset({"canonical_semantic", "stock_hf_evaluator_drift"})
PARITY_MEASUREMENT_FIELDS = (
    frozenset({"passed", "canonical_semantic", "stock_hf_evaluator_drift"})
    | PARITY_STATE_MEASUREMENT_FIELDS
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def token_ids_sha256(token_ids: list[int]) -> str:
    return sha256_bytes(canonical_json_bytes(token_ids))


if (
    token_ids_sha256(list(CANONICAL_FIXED_TOKEN_IDS))
    != CANONICAL_FIXED_TOKEN_IDS_SHA256
):
    raise RuntimeError("canonical fixed-token policy hash is inconsistent")


def tensor_sha256(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(torch.float32).contiguous().cpu().numpy()
    return sha256_bytes(canonical.tobytes(order="C"))


def compare_logits(
    reference: torch.Tensor,
    converted: torch.Tensor,
    tolerances: ParityTolerances,
) -> dict[str, Any]:
    """Apply the frozen r12 stock-HF bounded-drift policy."""

    if tuple(reference.shape) != tuple(converted.shape):
        raise RuntimeError(
            "logit shape mismatch: "
            f"reference={tuple(reference.shape)} converted={tuple(converted.shape)}"
        )
    if tuple(reference.shape) != CANONICAL_LOGITS_SHAPE:
        raise RuntimeError(
            "logit shape is not the canonical r12 shape: "
            f"{tuple(reference.shape)} != {CANONICAL_LOGITS_SHAPE}"
        )
    if tolerances != CANONICAL_PARITY_TOLERANCES:
        raise RuntimeError("parity tolerances are not the canonical r12 policy")

    reference_f32 = reference.detach().to(torch.float32).cpu()
    converted_f32 = converted.detach().to(torch.float32).cpu()
    if not bool(torch.isfinite(reference_f32).all()):
        raise RuntimeError("TorchTitan reference logits contain nonfinite values")
    if not bool(torch.isfinite(converted_f32).all()):
        raise RuntimeError("Transformers logits contain nonfinite values")

    absolute = (reference_f32 - converted_f32).abs()
    denominator = reference_f32.abs().clamp_min(torch.finfo(torch.float32).eps)
    relative = absolute / denominator
    close = torch.isclose(
        reference_f32,
        converted_f32,
        atol=tolerances.logit_atol,
        rtol=tolerances.logit_rtol,
    )

    reference_top = reference_f32.topk(tolerances.top_k, dim=-1).indices
    converted_top = converted_f32.topk(tolerances.top_k, dim=-1).indices
    intersection_counts_tensor = (
        (reference_top.unsqueeze(-1) == converted_top.unsqueeze(-2))
        .any(dim=-1)
        .sum(dim=-1, dtype=torch.int64)
    )
    top_1_matches = reference_f32.argmax(dim=-1) == converted_f32.argmax(dim=-1)

    logits_shape = list(reference_f32.shape)
    logit_element_count = reference_f32.numel()
    close_success_count = int(close.sum(dtype=torch.int64).item())
    close_failure_count = logit_element_count - close_success_count
    position_count = reference_f32.shape[0] * reference_f32.shape[1]
    top_1_match_count = int(top_1_matches.sum(dtype=torch.int64).item())
    top_1_mismatch_count = position_count - top_1_match_count
    top_k_intersection_counts = [
        int(value) for value in intersection_counts_tensor.flatten().tolist()
    ]
    top_k_intersection_count_min = min(top_k_intersection_counts)
    top_k_intersection_count_total = sum(top_k_intersection_counts)
    top_k_membership_count = position_count * tolerances.top_k
    max_abs = float(absolute.max().item())
    mean_abs = float(absolute.mean().item())
    rms = float(torch.sqrt(torch.mean(absolute.square())).item())
    passed = (
        close_failure_count <= tolerances.max_close_failure_count
        and max_abs <= tolerances.max_abs_error
        and mean_abs <= tolerances.max_mean_abs_error
        and rms <= tolerances.max_rms_error
        and top_1_mismatch_count <= tolerances.max_top_1_mismatch_count
        and top_k_intersection_count_min
        >= tolerances.min_top_k_intersection_count_per_position
        and top_k_intersection_count_total
        >= tolerances.min_top_k_intersection_count_total
    )
    return {
        "passed": passed,
        "logits_shape": logits_shape,
        "logit_element_count": logit_element_count,
        "strict_allclose": close_failure_count == 0,
        "close_success_count": close_success_count,
        "close_failure_count": close_failure_count,
        "close_success_ratio": close_success_count / logit_element_count,
        "close_failure_ratio": close_failure_count / logit_element_count,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rms_error": rms,
        "max_relative_error": float(relative.max().item()),
        "position_count": position_count,
        "top_1_match_count": top_1_match_count,
        "top_1_mismatch_count": top_1_mismatch_count,
        "top_1_agreement_ratio": top_1_match_count / position_count,
        "top_k": tolerances.top_k,
        "top_k_membership_count": top_k_membership_count,
        "top_k_intersection_counts": top_k_intersection_counts,
        "top_k_intersection_ratios": [
            count / tolerances.top_k for count in top_k_intersection_counts
        ],
        "top_k_intersection_count_min": top_k_intersection_count_min,
        "top_k_intersection_count_total": top_k_intersection_count_total,
        "top_k_intersection_ratio_min": (
            top_k_intersection_count_min / tolerances.top_k
        ),
        "top_k_intersection_ratio_total": (
            top_k_intersection_count_total / top_k_membership_count
        ),
        "reference_logits_sha256": tensor_sha256(reference_f32),
        "converted_logits_sha256": tensor_sha256(converted_f32),
    }


def compare_semantic_logits(
    reference: torch.Tensor,
    canonical: torch.Tensor,
    tolerances: SemanticParityTolerances,
) -> dict[str, Any]:
    """Require the HF model with TorchTitan-equivalent RoPE to be exact."""

    if tuple(reference.shape) != tuple(canonical.shape):
        raise RuntimeError(
            "semantic logit shape mismatch: "
            f"reference={tuple(reference.shape)} canonical={tuple(canonical.shape)}"
        )
    if tuple(reference.shape) != CANONICAL_LOGITS_SHAPE:
        raise RuntimeError(
            "semantic logit shape is not the canonical r12 shape: "
            f"{tuple(reference.shape)} != {CANONICAL_LOGITS_SHAPE}"
        )
    if tolerances != CANONICAL_SEMANTIC_TOLERANCES:
        raise RuntimeError("semantic tolerances are not the canonical exact policy")

    reference_f32 = reference.detach().to(torch.float32).contiguous().cpu()
    canonical_f32 = canonical.detach().to(torch.float32).contiguous().cpu()
    if not bool(torch.isfinite(reference_f32).all()):
        raise RuntimeError("TorchTitan reference logits contain nonfinite values")
    if not bool(torch.isfinite(canonical_f32).all()):
        raise RuntimeError("canonical-semantics HF logits contain nonfinite values")

    exact = torch.eq(reference_f32.view(torch.int32), canonical_f32.view(torch.int32))
    absolute = (reference_f32 - canonical_f32).abs()
    logit_element_count = reference_f32.numel()
    exact_match_count = int(exact.sum(dtype=torch.int64).item())
    mismatch_count = logit_element_count - exact_match_count
    max_abs = float(absolute.max().item())
    mean_abs = float(absolute.mean().item())
    rms = float(torch.sqrt(torch.mean(absolute.square())).item())
    passed = (
        mismatch_count <= tolerances.max_mismatched_elements
        and max_abs <= tolerances.max_abs_error
    )
    return {
        "passed": passed,
        "logits_shape": list(reference_f32.shape),
        "logit_element_count": logit_element_count,
        "strict_bit_exact": mismatch_count == 0,
        "exact_match_count": exact_match_count,
        "mismatch_count": mismatch_count,
        "exact_match_ratio": exact_match_count / logit_element_count,
        "mismatch_ratio": mismatch_count / logit_element_count,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rms_error": rms,
        "reference_logits_sha256": tensor_sha256(reference_f32),
        "canonical_logits_sha256": tensor_sha256(canonical_f32),
    }


def seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add a hash over every other receipt field."""

    if "receipt_sha256" in payload:
        raise ValueError("unsealed payload must not contain receipt_sha256")
    sealed = dict(payload)
    sealed["receipt_sha256"] = sha256_bytes(canonical_json_bytes(sealed))
    return sealed


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"receipt {field} is not a lowercase SHA-256")


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"receipt {field} must be a nonnegative integer")
    return value


def _require_nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0:
        raise RuntimeError(f"receipt {field} must be a finite nonnegative float")
    return value


def _require_exact_ratio(
    value: Any, numerator: int, denominator: int, field: str
) -> None:
    observed = _require_nonnegative_float(value, field)
    if denominator <= 0 or observed != numerator / denominator:
        raise RuntimeError(f"receipt {field} is inconsistent with its exact counts")


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed unless a receipt is a passing, intact, expected receipt."""

    required = {
        "schema_version",
        "method",
        "policy",
        "passed",
        "created_at_utc",
        "conversion_manifest_sha256",
        "route",
        "step",
        "ntokens_seen",
        "checkpoint_metadata_sha256",
        "source_job_id",
        "source_uri_sha256",
        "fixed_token_ids",
        "fixed_token_ids_sha256",
        "expected_logits_shape",
        "tool_sha256",
        "code_bundle_sha256",
        "code_files_sha256",
        "environment",
        "tolerances",
        "measurements",
        "limitations",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise RuntimeError(
            "parity receipt field inventory is not exact: "
            f"missing={sorted(required - set(receipt))} "
            f"extra={sorted(set(receipt) - required)}"
        )
    if receipt["schema_version"] != PARITY_RECEIPT_SCHEMA_VERSION:
        raise RuntimeError("unsupported parity receipt schema")
    if receipt["method"] != PARITY_METHOD:
        raise RuntimeError("unexpected parity method")
    if receipt["policy"] != PARITY_POLICY:
        raise RuntimeError("unexpected parity policy")
    if not isinstance(receipt["passed"], bool):
        raise RuntimeError("parity receipt verdict is not a boolean")
    route = receipt["route"]
    if route not in SUPPORTED_ROUTES:
        raise RuntimeError(f"unsupported parity receipt route: {route!r}")
    source_job_id = receipt["source_job_id"]
    if (
        not isinstance(source_job_id, str)
        or SOURCE_JOB_PATTERN.fullmatch(source_job_id) is None
    ):
        raise RuntimeError("receipt source_job_id is malformed")
    created_at = receipt["created_at_utc"]
    if not isinstance(created_at, str):
        raise RuntimeError("receipt creation timestamp is absent")
    try:
        parsed_at = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise RuntimeError("receipt creation timestamp is malformed") from error
    if parsed_at.tzinfo is None:
        raise RuntimeError("receipt creation timestamp is not timezone-aware")
    for field in (
        "conversion_manifest_sha256",
        "checkpoint_metadata_sha256",
        "source_uri_sha256",
        "fixed_token_ids_sha256",
        "tool_sha256",
        "code_bundle_sha256",
        "receipt_sha256",
    ):
        _require_sha256(receipt[field], field)
    for field in ("step", "ntokens_seen"):
        _require_nonnegative_int(receipt[field], field)

    expected_hash = receipt["receipt_sha256"]
    unsealed = dict(receipt)
    del unsealed["receipt_sha256"]
    if sha256_bytes(canonical_json_bytes(unsealed)) != expected_hash:
        raise RuntimeError("parity receipt payload hash mismatch")

    token_ids = receipt["fixed_token_ids"]
    if (
        not isinstance(token_ids, list)
        or token_ids != list(CANONICAL_FIXED_TOKEN_IDS)
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= LLAMA8B_VOCAB_SIZE
            for token in token_ids
        )
    ):
        raise RuntimeError("receipt fixed_token_ids are not the canonical r12 tokens")
    if (
        receipt["fixed_token_ids_sha256"] != CANONICAL_FIXED_TOKEN_IDS_SHA256
        or token_ids_sha256(token_ids) != CANONICAL_FIXED_TOKEN_IDS_SHA256
    ):
        raise RuntimeError("receipt canonical fixed-token binding is invalid")
    expected_logits_shape = receipt["expected_logits_shape"]
    if (
        not isinstance(expected_logits_shape, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in expected_logits_shape
        )
        or expected_logits_shape != list(CANONICAL_LOGITS_SHAPE)
    ):
        raise RuntimeError("receipt expected logits shape is not canonical r12")
    code_files = receipt["code_files_sha256"]
    if not isinstance(code_files, Mapping) or set(code_files) != PARITY_CODE_FILE_KEYS:
        raise RuntimeError("receipt code-file hash inventory is not exact")
    for name, digest in code_files.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("receipt code-file hash has an invalid name")
        _require_sha256(digest, f"code_files_sha256.{name}")
    if (
        sha256_bytes(canonical_json_bytes(dict(code_files)))
        != receipt["code_bundle_sha256"]
    ):
        raise RuntimeError("receipt code-bundle binding is invalid")
    if receipt["tool_sha256"] != code_files["parity_tool"]:
        raise RuntimeError("receipt parity-tool binding is invalid")

    environment = receipt["environment"]
    if (
        not isinstance(environment, Mapping)
        or set(environment) != PARITY_ENVIRONMENT_FIELDS
    ):
        raise RuntimeError("receipt environment field inventory is not exact")
    expected_environment = {
        "torchtitan_commit": PINNED_TORCHTITAN_COMMIT,
        "transformers": PINNED_TRANSFORMERS_VERSION,
        "compute_dtype": "torch.bfloat16",
        "attention_backend": "SDPBackend.MATH",
        "native_attention": "TorchTitan scaled_dot_product_attention causal math",
        "converted_attention": "Transformers SDPA causal math",
        "canonical_semantic_rope": (
            "TorchTitan interleaved complex64 RoPE in converted HF model"
        ),
        "canonical_semantic_rmsnorm": (
            "TorchTitan torch.nn.functional.rms_norm in converted HF model"
        ),
        "stock_hf_rope": "Transformers half-split BF16 RoPE",
        "stock_hf_rmsnorm": "Transformers LlamaRMSNorm FP32-normalize BF16-scale",
    }
    for field, expected in expected_environment.items():
        if environment.get(field) != expected:
            raise RuntimeError(
                f"receipt environment mismatch for {field}: "
                f"{environment.get(field)!r} != {expected!r}"
            )
    for field in ("python", "torch", "device"):
        if not isinstance(environment.get(field), str) or not environment[field]:
            raise RuntimeError(f"receipt environment field is absent: {field}")
    if not environment["device"].startswith("cuda:"):
        raise RuntimeError("receipt parity device is not CUDA")
    if (
        not isinstance(environment.get("device_name"), str)
        or "B200" not in environment["device_name"].upper()
    ):
        raise RuntimeError("receipt parity device is not an NVIDIA B200")
    compute_capability = environment.get("compute_capability")
    if (
        not isinstance(compute_capability, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in compute_capability
        )
        or compute_capability != [10, 0]
    ):
        raise RuntimeError("receipt parity device compute capability is not B200")
    if receipt["limitations"] != []:
        raise RuntimeError("passing native parity receipt must have no limitations")

    tolerance_data = receipt["tolerances"]
    if (
        not isinstance(tolerance_data, Mapping)
        or set(tolerance_data) != PARITY_TOLERANCE_FIELDS
    ):
        raise RuntimeError("receipt tolerance field inventory is not exact")
    semantic_tolerance_data = tolerance_data["canonical_semantic"]
    stock_tolerance_data = tolerance_data["stock_hf_evaluator_drift"]
    if not isinstance(semantic_tolerance_data, Mapping) or set(
        semantic_tolerance_data
    ) != set(SemanticParityTolerances.__dataclass_fields__):
        raise RuntimeError("receipt semantic tolerance inventory is not exact")
    if not isinstance(stock_tolerance_data, Mapping) or set(
        stock_tolerance_data
    ) != set(ParityTolerances.__dataclass_fields__):
        raise RuntimeError("receipt stock-HF tolerance inventory is not exact")
    try:
        semantic_tolerances = SemanticParityTolerances(**dict(semantic_tolerance_data))
        tolerances = ParityTolerances(**dict(stock_tolerance_data))
    except (TypeError, ValueError) as error:
        raise RuntimeError("receipt tolerances are malformed") from error
    if (
        semantic_tolerances != CANONICAL_SEMANTIC_TOLERANCES
        or dict(semantic_tolerance_data) != CANONICAL_SEMANTIC_TOLERANCES.to_dict()
    ):
        raise RuntimeError("receipt semantic tolerances are not canonical")
    if (
        tolerances != CANONICAL_PARITY_TOLERANCES
        or dict(stock_tolerance_data) != CANONICAL_PARITY_TOLERANCES.to_dict()
    ):
        raise RuntimeError("receipt stock-HF tolerances are not canonical r12")

    outer_measurements = receipt["measurements"]
    if (
        not isinstance(outer_measurements, Mapping)
        or set(outer_measurements) != PARITY_MEASUREMENT_FIELDS
    ):
        raise RuntimeError("receipt measurement field inventory is not exact")
    if not isinstance(outer_measurements["passed"], bool):
        raise RuntimeError("receipt combined measurement verdict is not a boolean")

    semantic_measurements = outer_measurements["canonical_semantic"]
    if (
        not isinstance(semantic_measurements, Mapping)
        or set(semantic_measurements) != PARITY_SEMANTIC_MEASUREMENT_FIELDS
    ):
        raise RuntimeError("receipt semantic measurement inventory is not exact")
    if not isinstance(semantic_measurements["passed"], bool):
        raise RuntimeError("receipt semantic verdict is not a boolean")
    if not isinstance(semantic_measurements["strict_bit_exact"], bool):
        raise RuntimeError("receipt strict_bit_exact is not a boolean")
    semantic_shape = semantic_measurements["logits_shape"]
    if (
        not isinstance(semantic_shape, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in semantic_shape
        )
        or semantic_shape != list(CANONICAL_LOGITS_SHAPE)
    ):
        raise RuntimeError("receipt semantic logits shape is not canonical r12")
    for field in ("logit_element_count", "exact_match_count", "mismatch_count"):
        _require_nonnegative_int(
            semantic_measurements[field], f"measurements.canonical_semantic.{field}"
        )
    for field in (
        "exact_match_ratio",
        "mismatch_ratio",
        "max_abs_error",
        "mean_abs_error",
        "rms_error",
    ):
        _require_nonnegative_float(
            semantic_measurements[field], f"measurements.canonical_semantic.{field}"
        )
    if semantic_measurements["mean_abs_error"] > semantic_measurements["rms_error"]:
        raise RuntimeError("receipt semantic mean-absolute error exceeds RMS error")
    if semantic_measurements["rms_error"] > semantic_measurements["max_abs_error"]:
        raise RuntimeError("receipt semantic RMS error exceeds max-absolute error")
    for field in ("reference_logits_sha256", "canonical_logits_sha256"):
        _require_sha256(
            semantic_measurements[field], f"measurements.canonical_semantic.{field}"
        )

    logit_element_count = math.prod(CANONICAL_LOGITS_SHAPE)
    if semantic_measurements["logit_element_count"] != logit_element_count:
        raise RuntimeError("receipt semantic logit element count is inconsistent")
    exact_match_count = semantic_measurements["exact_match_count"]
    mismatch_count = semantic_measurements["mismatch_count"]
    if exact_match_count + mismatch_count != logit_element_count:
        raise RuntimeError("receipt semantic counts do not cover every logit")
    if semantic_measurements["strict_bit_exact"] is not (mismatch_count == 0):
        raise RuntimeError("receipt strict_bit_exact is inconsistent with counts")
    _require_exact_ratio(
        semantic_measurements["exact_match_ratio"],
        exact_match_count,
        logit_element_count,
        "measurements.canonical_semantic.exact_match_ratio",
    )
    _require_exact_ratio(
        semantic_measurements["mismatch_ratio"],
        mismatch_count,
        logit_element_count,
        "measurements.canonical_semantic.mismatch_ratio",
    )
    if (
        mismatch_count == 0
        and semantic_measurements["reference_logits_sha256"]
        != semantic_measurements["canonical_logits_sha256"]
    ):
        raise RuntimeError("receipt exact semantic logits hashes disagree")
    semantic_hashes_are_identical = (
        semantic_measurements["reference_logits_sha256"]
        == semantic_measurements["canonical_logits_sha256"]
    )
    if semantic_hashes_are_identical and not (
        semantic_measurements["strict_bit_exact"]
        and exact_match_count == logit_element_count
        and mismatch_count == 0
        and semantic_measurements["exact_match_ratio"] == 1.0
        and semantic_measurements["mismatch_ratio"] == 0.0
        and semantic_measurements["max_abs_error"] == 0.0
        and semantic_measurements["mean_abs_error"] == 0.0
        and semantic_measurements["rms_error"] == 0.0
    ):
        raise RuntimeError(
            "receipt equal semantic logits hashes contradict non-exact measurements"
        )
    semantic_recomputed_passed = (
        mismatch_count <= semantic_tolerances.max_mismatched_elements
        and semantic_measurements["max_abs_error"] <= semantic_tolerances.max_abs_error
    )
    if semantic_measurements["passed"] is not semantic_recomputed_passed:
        raise RuntimeError("receipt semantic verdict does not recompute")

    measurements = outer_measurements["stock_hf_evaluator_drift"]
    if (
        not isinstance(measurements, Mapping)
        or set(measurements) != PARITY_LOGIT_MEASUREMENT_FIELDS
    ):
        raise RuntimeError("receipt stock-HF measurement inventory is not exact")
    if not isinstance(measurements["passed"], bool):
        raise RuntimeError("receipt stock-HF verdict is not a boolean")
    if not isinstance(measurements["strict_allclose"], bool):
        raise RuntimeError("receipt strict_allclose is not a boolean")
    measured_logits_shape = measurements["logits_shape"]
    if (
        not isinstance(measured_logits_shape, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in measured_logits_shape
        )
        or measured_logits_shape != list(CANONICAL_LOGITS_SHAPE)
    ):
        raise RuntimeError("receipt stock-HF logits shape is not canonical r12")

    integer_measurements = (
        "logit_element_count",
        "close_success_count",
        "close_failure_count",
        "position_count",
        "top_1_match_count",
        "top_1_mismatch_count",
        "top_k",
        "top_k_membership_count",
        "top_k_intersection_count_min",
        "top_k_intersection_count_total",
    )
    for field in integer_measurements:
        _require_nonnegative_int(
            measurements[field], f"measurements.stock_hf_evaluator_drift.{field}"
        )
    float_measurements = (
        "close_success_ratio",
        "close_failure_ratio",
        "max_abs_error",
        "mean_abs_error",
        "rms_error",
        "max_relative_error",
        "top_1_agreement_ratio",
        "top_k_intersection_ratio_min",
        "top_k_intersection_ratio_total",
    )
    for field in float_measurements:
        _require_nonnegative_float(
            measurements[field], f"measurements.stock_hf_evaluator_drift.{field}"
        )
    if measurements["mean_abs_error"] > measurements["max_abs_error"]:
        raise RuntimeError("receipt stock-HF mean-absolute error exceeds max error")
    if measurements["mean_abs_error"] > measurements["rms_error"]:
        raise RuntimeError("receipt stock-HF mean-absolute error exceeds RMS error")
    if measurements["rms_error"] > measurements["max_abs_error"]:
        raise RuntimeError("receipt stock-HF RMS error exceeds max-absolute error")
    for field in ("reference_logits_sha256", "converted_logits_sha256"):
        _require_sha256(
            measurements[field], f"measurements.stock_hf_evaluator_drift.{field}"
        )
    if (
        semantic_measurements["reference_logits_sha256"]
        != measurements["reference_logits_sha256"]
    ):
        raise RuntimeError("receipt subgates do not bind the same reference logits")

    if measurements["logit_element_count"] != logit_element_count:
        raise RuntimeError("receipt stock-HF logit element count is inconsistent")
    close_success_count = measurements["close_success_count"]
    close_failure_count = measurements["close_failure_count"]
    if close_success_count + close_failure_count != logit_element_count:
        raise RuntimeError("receipt stock-HF close counts do not cover every logit")
    if measurements["strict_allclose"] is not (close_failure_count == 0):
        raise RuntimeError("receipt strict_allclose is inconsistent with close counts")
    _require_exact_ratio(
        measurements["close_success_ratio"],
        close_success_count,
        logit_element_count,
        "measurements.stock_hf_evaluator_drift.close_success_ratio",
    )
    _require_exact_ratio(
        measurements["close_failure_ratio"],
        close_failure_count,
        logit_element_count,
        "measurements.stock_hf_evaluator_drift.close_failure_ratio",
    )

    position_count = CANONICAL_LOGITS_SHAPE[0] * CANONICAL_LOGITS_SHAPE[1]
    if measurements["position_count"] != position_count:
        raise RuntimeError("receipt stock-HF position count is inconsistent")
    top_1_match_count = measurements["top_1_match_count"]
    top_1_mismatch_count = measurements["top_1_mismatch_count"]
    if top_1_match_count + top_1_mismatch_count != position_count:
        raise RuntimeError("receipt top-1 counts do not cover every position")
    _require_exact_ratio(
        measurements["top_1_agreement_ratio"],
        top_1_match_count,
        position_count,
        "measurements.stock_hf_evaluator_drift.top_1_agreement_ratio",
    )

    top_k = tolerances.top_k
    top_k_membership_count = position_count * top_k
    if (
        measurements["top_k"] != top_k
        or measurements["top_k_membership_count"] != top_k_membership_count
    ):
        raise RuntimeError("receipt top-k policy/count is inconsistent")
    intersection_counts = measurements["top_k_intersection_counts"]
    if (
        not isinstance(intersection_counts, list)
        or len(intersection_counts) != position_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= top_k
            for value in intersection_counts
        )
    ):
        raise RuntimeError("receipt per-position top-k intersection counts are invalid")
    intersection_ratios = measurements["top_k_intersection_ratios"]
    if (
        not isinstance(intersection_ratios, list)
        or len(intersection_ratios) != position_count
    ):
        raise RuntimeError("receipt per-position top-k intersection ratios are invalid")
    for index, (ratio, count) in enumerate(
        zip(intersection_ratios, intersection_counts, strict=True)
    ):
        _require_exact_ratio(
            ratio,
            count,
            top_k,
            f"measurements.stock_hf_evaluator_drift."
            f"top_k_intersection_ratios[{index}]",
        )
    intersection_min = min(intersection_counts)
    intersection_total = sum(intersection_counts)
    if (
        measurements["top_k_intersection_count_min"] != intersection_min
        or measurements["top_k_intersection_count_total"] != intersection_total
    ):
        raise RuntimeError("receipt top-k intersection summaries are inconsistent")
    _require_exact_ratio(
        measurements["top_k_intersection_ratio_min"],
        intersection_min,
        top_k,
        "measurements.stock_hf_evaluator_drift.top_k_intersection_ratio_min",
    )
    _require_exact_ratio(
        measurements["top_k_intersection_ratio_total"],
        intersection_total,
        top_k_membership_count,
        "measurements.stock_hf_evaluator_drift.top_k_intersection_ratio_total",
    )
    stock_logits_hashes_are_identical = (
        measurements["reference_logits_sha256"]
        == measurements["converted_logits_sha256"]
    )
    if stock_logits_hashes_are_identical and not (
        measurements["strict_allclose"]
        and close_success_count == logit_element_count
        and close_failure_count == 0
        and measurements["max_abs_error"] == 0.0
        and measurements["mean_abs_error"] == 0.0
        and measurements["rms_error"] == 0.0
        and measurements["max_relative_error"] == 0.0
        and top_1_match_count == position_count
        and top_1_mismatch_count == 0
        and intersection_counts == [top_k] * position_count
        and intersection_min == top_k
        and intersection_total == top_k_membership_count
    ):
        raise RuntimeError(
            "receipt equal stock logits hashes contradict non-exact measurements"
        )

    exact_counts = {
        "source_tensors_streamed": SOURCE_TENSORS_BY_ROUTE[route],
        "native_parameters_loaded": LLAMA8B_NATIVE_PARAMETERS,
        "converted_tensors_exact": LLAMA8B_HF_TENSORS,
        "converted_elements_exact": LLAMA8B_HF_ELEMENTS,
        "frozen_aliases_checked": FROZEN_ALIASES_BY_ROUTE[route],
        "native_math_sdpa_modules": LLAMA8B_NATIVE_ATTENTION_MODULES,
    }
    for field, expected in exact_counts.items():
        value = outer_measurements[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RuntimeError(
                f"receipt exact-state measurement mismatch for {field}: "
                f"{value!r} != {expected}"
            )

    stock_recomputed_passed = (
        close_failure_count <= tolerances.max_close_failure_count
        and measurements["max_abs_error"] <= tolerances.max_abs_error
        and measurements["mean_abs_error"] <= tolerances.max_mean_abs_error
        and measurements["rms_error"] <= tolerances.max_rms_error
        and top_1_mismatch_count <= tolerances.max_top_1_mismatch_count
        and intersection_min >= tolerances.min_top_k_intersection_count_per_position
        and intersection_total >= tolerances.min_top_k_intersection_count_total
    )
    if measurements["passed"] is not stock_recomputed_passed:
        raise RuntimeError("receipt stock-HF verdict does not recompute")
    recomputed_passed = semantic_recomputed_passed and stock_recomputed_passed
    if outer_measurements["passed"] is not recomputed_passed:
        raise RuntimeError("receipt combined measurement verdict does not recompute")
    if receipt["passed"] is not recomputed_passed:
        raise RuntimeError("receipt top-level verdict does not recompute")
    if not recomputed_passed:
        raise RuntimeError("parity receipt is not a passing receipt")

    for field, expected in (expected_bindings or {}).items():
        if field not in receipt or receipt[field] != expected:
            raise RuntimeError(
                f"parity receipt binding mismatch for {field}: "
                f"{receipt.get(field)!r} != {expected!r}"
            )


def write_atomic_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Publish a new receipt atomically and never overwrite prior evidence."""

    path = path.resolve()
    if path.exists():
        raise RuntimeError(f"refusing to overwrite parity receipt: {path}")
    if not path.parent.is_dir():
        raise RuntimeError(f"parity receipt parent is absent: {path.parent}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.incomplete.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"refusing to overwrite parity receipt: {path}"
            ) from error
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
