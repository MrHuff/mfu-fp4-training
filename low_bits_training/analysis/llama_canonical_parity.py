# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Sealed receipt contract for canonical-only Llama checkpoint parity.

This is intentionally separate from the stock-Hugging-Face bounded-drift gate.
A passing receipt proves exact state conversion and bit-exact fixed-token logits
under TorchTitan-equivalent RoPE/RMSNorm with MATH SDPA.  It contains no stock
HF measurement and does not provide a mechanism for relaxing tolerances.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Mapping

from .llama_checkpoint_routes import (
    LLAMA3_8B,
    SUPPORTED_ROUTES,
    route_alias_keys,
    route_trainable_shapes,
)
from .llama_conversion_parity import (
    CANONICAL_FIXED_TOKEN_IDS,
    CANONICAL_FIXED_TOKEN_IDS_SHA256,
    CANONICAL_LOGITS_SHAPE,
    CANONICAL_SEMANTIC_TOLERANCES,
    PINNED_TORCHTITAN_COMMIT,
    PINNED_TRANSFORMERS_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    token_ids_sha256,
)


CANONICAL_PARITY_SCHEMA_VERSION = 1
CANONICAL_PARITY_METHOD = (
    "pinned-torchtitan-native-plus-exact-state-canonical-only-v1"
)
CANONICAL_PARITY_POLICY = "fully-canonical-hf-rope-rmsnorm-math-sdpa-v1"
CANONICAL_PARITY_CODE_KEYS = {
    "canonical_parity_tool",
    "canonical_receipt_module",
    "base_parity_tool",
    "parity_measurement_module",
    "checkpoint_routes",
    "checkpoint_streamer",
    "canonical_eval_wrapper",
    "torchtitan_llama_model",
    "torchtitan_llama_args",
    "torchtitan_llama_adapter",
    "torchtitan_attention",
}
CANONICAL_SEMANTIC_FIELDS = {
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
CANONICAL_STATE_FIELDS = {
    "source_tensors_streamed",
    "native_parameters_loaded",
    "converted_tensors_exact",
    "converted_elements_exact",
    "frozen_aliases_checked",
    "native_math_sdpa_modules",
}
SHA256_CHARS = frozenset("0123456789abcdef")


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_CHARS for character in value)
    ):
        raise RuntimeError(f"canonical parity {field} is not a lowercase SHA-256")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"canonical parity {field} must be a nonnegative integer"
        )
    return value


def _nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0:
        raise RuntimeError(f"canonical parity {field} must be nonnegative finite")
    return value


def seal_canonical_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise ValueError("unsealed canonical payload contains receipt_sha256")
    receipt = dict(payload)
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def validate_canonical_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed unless this is an intact, passing canonical-only receipt."""

    required = {
        "schema_version",
        "method",
        "policy",
        "passed",
        "can_authorize_downstream_evaluation",
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
        raise RuntimeError("canonical parity receipt field inventory is not exact")
    if (
        receipt["schema_version"] != CANONICAL_PARITY_SCHEMA_VERSION
        or receipt["method"] != CANONICAL_PARITY_METHOD
        or receipt["policy"] != CANONICAL_PARITY_POLICY
        or receipt["passed"] is not True
        or receipt["can_authorize_downstream_evaluation"] is not True
        or receipt["route"] not in SUPPORTED_ROUTES
        or receipt["limitations"] != []
    ):
        raise RuntimeError("canonical parity receipt policy or verdict drift")
    created = receipt["created_at_utc"]
    if not isinstance(created, str):
        raise RuntimeError("canonical parity receipt has no timestamp")
    try:
        parsed = datetime.fromisoformat(created)
    except ValueError as error:
        raise RuntimeError("canonical parity timestamp is malformed") from error
    if parsed.tzinfo is None:
        raise RuntimeError("canonical parity timestamp is not timezone-aware")
    if not isinstance(receipt["source_job_id"], str) or not receipt["source_job_id"]:
        raise RuntimeError("canonical parity source job is malformed")
    for field in ("step", "ntokens_seen"):
        _nonnegative_int(receipt[field], field)
    for field in (
        "conversion_manifest_sha256",
        "checkpoint_metadata_sha256",
        "source_uri_sha256",
        "fixed_token_ids_sha256",
        "tool_sha256",
        "code_bundle_sha256",
        "receipt_sha256",
    ):
        _sha256(receipt[field], field)
    unsealed = dict(receipt)
    seal = unsealed.pop("receipt_sha256")
    if sha256_bytes(canonical_json_bytes(unsealed)) != seal:
        raise RuntimeError("canonical parity receipt seal mismatch")

    tokens = receipt["fixed_token_ids"]
    if (
        tokens != list(CANONICAL_FIXED_TOKEN_IDS)
        or token_ids_sha256(tokens) != CANONICAL_FIXED_TOKEN_IDS_SHA256
        or receipt["fixed_token_ids_sha256"] != CANONICAL_FIXED_TOKEN_IDS_SHA256
        or receipt["expected_logits_shape"] != list(CANONICAL_LOGITS_SHAPE)
    ):
        raise RuntimeError("canonical parity fixed-token contract drift")

    code_files = receipt["code_files_sha256"]
    if not isinstance(code_files, Mapping) or set(code_files) != CANONICAL_PARITY_CODE_KEYS:
        raise RuntimeError("canonical parity code-file inventory is not exact")
    for name, digest in code_files.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("canonical parity code-file name is malformed")
        _sha256(digest, f"code_files_sha256.{name}")
    if (
        sha256_bytes(canonical_json_bytes(dict(code_files)))
        != receipt["code_bundle_sha256"]
        or receipt["tool_sha256"] != code_files["canonical_parity_tool"]
    ):
        raise RuntimeError("canonical parity code-bundle binding is invalid")

    environment = receipt["environment"]
    if not isinstance(environment, Mapping):
        raise RuntimeError("canonical parity environment is malformed")
    expected_environment = {
        "torchtitan_commit": PINNED_TORCHTITAN_COMMIT,
        "transformers": PINNED_TRANSFORMERS_VERSION,
        "compute_dtype": "torch.bfloat16",
        "attention_backend": "SDPBackend.MATH",
        "canonical_semantic_rope": (
            "TorchTitan interleaved complex64 RoPE in converted HF model"
        ),
        "canonical_semantic_rmsnorm": (
            "TorchTitan torch.nn.functional.rms_norm in converted HF model"
        ),
        "stock_hf_computed": False,
    }
    for field, wanted in expected_environment.items():
        if environment.get(field) != wanted:
            raise RuntimeError(f"canonical parity environment drift for {field}")

    if receipt["tolerances"] != CANONICAL_SEMANTIC_TOLERANCES.to_dict():
        raise RuntimeError("canonical parity exact tolerance drift")
    measurements = receipt["measurements"]
    if (
        not isinstance(measurements, Mapping)
        or set(measurements) != {"passed", "canonical_semantic"} | CANONICAL_STATE_FIELDS
        or measurements["passed"] is not True
    ):
        raise RuntimeError("canonical parity measurement inventory or verdict drift")
    semantic = measurements["canonical_semantic"]
    if not isinstance(semantic, Mapping) or set(semantic) != CANONICAL_SEMANTIC_FIELDS:
        raise RuntimeError("canonical semantic measurement inventory is not exact")
    element_count = math.prod(CANONICAL_LOGITS_SHAPE)
    if (
        semantic["passed"] is not True
        or semantic["strict_bit_exact"] is not True
        or semantic["logits_shape"] != list(CANONICAL_LOGITS_SHAPE)
        or semantic["logit_element_count"] != element_count
        or semantic["exact_match_count"] != element_count
        or semantic["mismatch_count"] != 0
        or semantic["exact_match_ratio"] != 1.0
        or semantic["mismatch_ratio"] != 0.0
    ):
        raise RuntimeError("canonical semantic logits are not bit-exact")
    for field in ("max_abs_error", "mean_abs_error", "rms_error"):
        if _nonnegative_float(semantic[field], field) != 0.0:
            raise RuntimeError("canonical semantic error summary is nonzero")
    reference_hash = _sha256(
        semantic["reference_logits_sha256"], "reference_logits_sha256"
    )
    if (
        _sha256(semantic["canonical_logits_sha256"], "canonical_logits_sha256")
        != reference_hash
    ):
        raise RuntimeError("canonical semantic logits hashes differ")

    route = receipt["route"]
    expected_state = {
        "source_tensors_streamed": len(route_trainable_shapes(route, LLAMA3_8B)),
        "native_parameters_loaded": 291,
        "converted_tensors_exact": 291,
        "frozen_aliases_checked": len(route_alias_keys(route, LLAMA3_8B)),
        "native_math_sdpa_modules": 32,
    }
    for field, wanted in expected_state.items():
        if measurements[field] != wanted:
            raise RuntimeError(f"canonical parity state drift for {field}")
    if _nonnegative_int(
        measurements["converted_elements_exact"], "converted_elements_exact"
    ) <= 0:
        raise RuntimeError("canonical parity compared no converted elements")

    if expected_bindings is not None:
        allowed = {
            "conversion_manifest_sha256",
            "route",
            "step",
            "ntokens_seen",
            "checkpoint_metadata_sha256",
            "source_job_id",
            "source_uri_sha256",
        }
        if set(expected_bindings) != allowed:
            raise RuntimeError("canonical parity expected-binding inventory drift")
        for field, wanted in expected_bindings.items():
            if receipt[field] != wanted:
                raise RuntimeError(f"canonical parity binding mismatch for {field}")
