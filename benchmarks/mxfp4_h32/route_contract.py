"""Fail-closed contract for the pure MXFP4-v4 fixed-sign H32 probe."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = ROOT / "benchmark.json"
SCIENTIFIC_PREFIXES = (
    "MXFP4_",
    "NVFP4_",
    "NVTE_",
    "USE_TK_",
    "USE_FP4_",
    "USE_LBT_",
    "USE_MXFP4_",
    "FP4_KEEP_",
    "FP4_ATTN_",
    "FP4_FFN_",
    "FP4_CUDA_",
    "FP4_ENABLE_",
    "FP4_GPU_",
    "LBT_",
    "PAIR_",
    "TORCHTITAN_FSDP_",
)
FORBIDDEN_DEBUG_ENVIRONMENT = ("CUDA_LAUNCH_BLOCKING", "TORCH_USE_CUDA_DSA")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_spec(document)
    return document


def validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported benchmark schema")
    if spec.get("route") != "mxfp4-v4-row-sr-h32-rht":
        raise ValueError("unexpected route")
    topology = spec.get("topology", {})
    batch = spec.get("batch", {})
    measurement = spec.get("measurement", {})
    model = spec.get("model", {})
    nodes = _positive_int(topology.get("nodes"), "topology.nodes")
    processes = _positive_int(
        topology.get("processes_per_node"), "topology.processes_per_node"
    )
    world = _positive_int(topology.get("world_size"), "topology.world_size")
    if nodes * processes != world:
        raise ValueError("topology does not produce world_size")
    local = _positive_int(batch.get("local_sequences"), "batch.local_sequences")
    accumulation = _positive_int(
        batch.get("gradient_accumulation"), "batch.gradient_accumulation"
    )
    global_batch = _positive_int(
        batch.get("global_sequences"), "batch.global_sequences"
    )
    if world * local * accumulation != global_batch:
        raise ValueError("batch geometry does not produce global_sequences")
    blocks = _positive_int(model.get("blocks"), "model.blocks")
    if blocks != 32:
        raise ValueError("this benchmark is sealed to the 32-block Llama route")
    updates = _positive_int(measurement.get("updates"), "measurement.updates")
    first = _positive_int(
        measurement.get("steady_state_first_update"),
        "measurement.steady_state_first_update",
    )
    if first > updates:
        raise ValueError("steady-state window starts after the final update")
    environment = spec.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("environment contract is empty")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("environment contract must contain string pairs")
    required = {
        "MXFP4_BACKEND_VERSION": "v4",
        "MXFP4_USE_2D_WEIGHT_QUANT": "1",
        "MXFP4_USE_STOCHASTIC_ROUNDING": "1",
        "MXFP4_SR_GRAD": "1",
        "MXFP4_GRAD_SR_AXES": "row",
        "MXFP4_SR_SEED": "1234",
        "MXFP4_SR_EXPECTED_PRODUCERS": "128",
        "MXFP4_USE_RHT": "1",
        "MXFP4_RHT_AXES": "col",
        "MXFP4_RHT_BLOCK_SIZE": "32",
        "MXFP4_RHT_RANDOM_SIGN_MASK": "1",
        "MXFP4_RHT_WEIGHT": "0",
        "USE_FP4_CONVERT_OUTPUT_HEAD": "0",
        "MXFP4_TK_REQUIRE_LLAMA_BF16_OUTPUT_HEAD": "1",
    }
    drift = {
        key: (environment.get(key), wanted)
        for key, wanted in required.items()
        if environment.get(key) != wanted
    }
    if drift:
        raise ValueError(f"scientific route contract drifted: {drift}")
    _validate_no_remote_values(spec)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_no_remote_values(value: Any, path: str = "spec") -> None:
    if isinstance(value, str):
        if "://" in value:
            raise ValueError(f"remote URI is forbidden in {path}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _validate_no_remote_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_remote_values(item, f"{path}[{index}]")


def expected_logical_keys(blocks: int = 32) -> tuple[str, ...]:
    keys: list[str] = []
    for layer in range(blocks):
        attention = f"layers.{layer}.attention"
        feed_forward = f"layers.{layer}.feed_forward"
        keys.extend(
            (
                f"{attention}:qkv:sr:qkv_grad",
                f"{attention}:wo:sr:wo_grad",
                f"{feed_forward}:sr:ffn_w2_grad",
                f"{feed_forward}:sr:ffn_deriv_grad",
            )
        )
    result = tuple(sorted(keys))
    if len(result) != blocks * 4 or len(set(result)) != len(result):
        raise RuntimeError("internal producer manifest is malformed")
    return result


def scrub_scientific_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return a copy with inherited low-precision selectors removed."""

    return {
        key: value
        for key, value in environment.items()
        if not key.startswith(SCIENTIFIC_PREFIXES)
    }


def validate_environment(
    spec: Mapping[str, Any], environment: Mapping[str, str] | None = None
) -> None:
    validate_spec(spec)
    environment = os.environ if environment is None else environment
    expected = spec["environment"]
    drift = {
        key: (environment.get(key), wanted)
        for key, wanted in expected.items()
        if environment.get(key) != wanted
    }
    extras = sorted(
        key
        for key in environment
        if key.startswith(SCIENTIFIC_PREFIXES) and key not in expected
    )
    debug = sorted(key for key in FORBIDDEN_DEBUG_ENVIRONMENT if key in environment)
    if drift or extras or debug:
        raise RuntimeError(
            "MXFP4 H32 environment drifted: "
            f"values={drift}, extras={extras}, debug={debug}"
        )


def _unique_modules(model_parts: Iterable[object]) -> tuple[object, ...]:
    modules: dict[int, object] = {}
    for model in model_parts:
        for module in model.modules():
            modules.setdefault(id(module), module)
    return tuple(modules.values())


def validate_trainer_route(trainer: object, spec: Mapping[str, Any]) -> str:
    """Validate the live converted model and checkpointed SR namespace."""

    from low_bits_training.quantization.mxfp4_sr_state import (
        STATE_VERSION,
        active_mxfp4_sr_state,
    )

    validate_environment(spec)
    state = getattr(trainer, "mxfp4_sr_state", None)
    if state is None or active_mxfp4_sr_state() is not state:
        raise RuntimeError("MXFP4 row-SR state is absent or inactive")
    states = getattr(getattr(trainer, "checkpointer", None), "states", {})
    if states.get("mxfp4_sr_state") is not state:
        raise RuntimeError("MXFP4 row-SR state is not registered with DCP")
    if STATE_VERSION != 1:
        raise RuntimeError("MXFP4 row-SR ABI is not v1")
    world = spec["topology"]["world_size"]
    if state.rank not in range(world) or state.world_size != world:
        raise RuntimeError("MXFP4 row-SR rank/world namespace is not exact")
    expected = expected_logical_keys(spec["model"]["blocks"])
    if tuple(state.logical_keys) != expected:
        raise RuntimeError("MXFP4 row-SR producer manifest is not exact")
    if (
        int(state.user_seed) != 1234
        or int(state._active_user_seed) != 1234
        or int(state.user_subsequence_base) != 0
    ):
        raise RuntimeError("MXFP4 row-SR seed/subsequence namespace drifted")

    model_parts = getattr(trainer, "model_parts", None)
    if model_parts is None:
        raise RuntimeError("trainer has no converted model_parts")
    counts = Counter(module.__class__.__name__ for module in _unique_modules(model_parts))
    blocks = spec["model"]["blocks"]
    for class_name in ("FusedAttentionMXFP4_TK", "FusedFeedForwardMXFP4_TK"):
        if counts[class_name] != blocks:
            raise RuntimeError(
                f"pure MXFP4 route requires {blocks} {class_name}, "
                f"found {counts[class_name]}"
            )
    if any("LocalCTA" in name and count for name, count in counts.items()):
        raise RuntimeError("pure MXFP4 route contains a localCTA module")
    return (
        "[MXFP4 H32 ROUTE PASS] llama_attention=32 llama_ffn=32 "
        f"row_sr=on wgrad_col_h32=on weight_rht=off rank={state.rank}/{world}"
    )
