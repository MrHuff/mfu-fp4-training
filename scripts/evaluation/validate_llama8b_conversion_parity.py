#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Fail-closed fixed-token parity gate for a converted Llama-8B checkpoint.

The gate independently defuses the source DCP into an ordinary TorchTitan
Llama model, uses TorchTitan's own state-dict adapter to compare every BF16
destination tensor exactly, and then compares native TorchTitan and
Transformers logits for fixed token IDs.  A receipt is published atomically;
only a passing receipt is suitable for authorizing downstream evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors import safe_open

# Avoid importing training-only Transformer Engine and custom-kernel modules.
os.environ["LBT_LIGHT_IMPORT"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_bits_training.analysis.llama_checkpoint_routes import (  # noqa: E402
    FUSED_ROUTES,
    LLAMA3_8B,
    LlamaSpec,
    SUPPORTED_ROUTES,
    UNFUSED_ROUTES,
    bf16_unfused_shapes,
    expected_hf_shapes,
    hf_config,
    route_alias_keys,
    route_extra_state_shapes,
    route_model_shapes,
    route_trainable_shapes,
    validate_route_metadata,
)
from low_bits_training.analysis.llama_conversion_parity import (  # noqa: E402
    CANONICAL_FIXED_TOKEN_IDS,
    CANONICAL_LOGITS_SHAPE,
    CANONICAL_PARITY_TOLERANCES,
    CANONICAL_SEMANTIC_TOLERANCES,
    PARITY_METHOD,
    PARITY_POLICY,
    PARITY_RECEIPT_SCHEMA_VERSION,
    PINNED_TORCHTITAN_COMMIT,
    PINNED_TRANSFORMERS_VERSION,
    canonical_json_bytes,
    compare_logits,
    compare_semantic_logits,
    seal_receipt,
    sha256_bytes,
    sha256_file,
    token_ids_sha256,
    validate_receipt,
    write_atomic_receipt,
)
from low_bits_training.analysis.stream_checkpoints import (  # noqa: E402
    get_metadata,
    stream_checkpoint_reader,
)
from low_bits_training.analysis.torchtitan_gitlink import (  # noqa: E402
    validate_torchtitan_gitlink_marker,
)


SUPPORTED_TORCHTITAN_COMMIT = PINNED_TORCHTITAN_COMMIT
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_JOB_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
SAFE_FILENAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required JSON file is absent or empty: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _expected_dcp_shard_names(expected_shards: int) -> frozenset[str]:
    if (
        isinstance(expected_shards, bool)
        or not isinstance(expected_shards, int)
        or expected_shards <= 0
    ):
        raise RuntimeError("expected DCP shard count must be a positive integer")
    return frozenset(
        f"__{rank}_0.distcp" for rank in range(expected_shards)
    )


def _canonical_dcp_shard_name(
    value: Any, label: str, expected_shards: int
) -> str:
    if (
        not isinstance(value, str)
        or value not in _expected_dcp_shard_names(expected_shards)
    ):
        raise RuntimeError(f"noncanonical {label}: {value!r}")
    return value


def _safe_filename(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or SAFE_FILENAME_PATTERN.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise RuntimeError(f"unsafe {label}: {value!r}")
    return value


def _read_scalar(
    checkpoint: Path, metadata: Any, fqn: str, expected_shards: int
) -> int:
    storage = getattr(metadata, "storage_data", None)
    if not isinstance(storage, Mapping):
        raise RuntimeError("checkpoint metadata has no storage-data mapping")
    matches = [
        item for index, item in storage.items() if getattr(index, "fqn", None) == fqn
    ]
    if len(matches) != 1:
        raise RuntimeError(f"checkpoint scalar storage is not unique: {fqn}")
    item = matches[0]
    relative_path = _canonical_dcp_shard_name(
        getattr(item, "relative_path", None), "shard name", expected_shards
    )
    offset = getattr(item, "offset", None)
    length = getattr(item, "length", None)
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
    ):
        raise RuntimeError(f"checkpoint scalar storage is malformed: {fqn}")
    with (checkpoint / relative_path).open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(length)
    if len(payload) != length:
        raise RuntimeError(f"checkpoint scalar storage is truncated: {fqn}")
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"checkpoint scalar is not an integer: {fqn}")
    return value


def _validate_source_inventory(
    checkpoint: Path,
    metadata: Any,
    manifest: Mapping[str, Any],
    expected_shards: int,
) -> None:
    expected = _expected_dcp_shard_names(expected_shards)
    storage = getattr(metadata, "storage_data", None)
    if not isinstance(storage, Mapping) or not storage:
        raise RuntimeError("checkpoint metadata has no storage-data mapping")
    referenced = {
        _canonical_dcp_shard_name(
            getattr(item, "relative_path", None), "source shard", expected_shards
        )
        for item in storage.values()
        if isinstance(getattr(item, "relative_path", None), str)
        and getattr(item, "relative_path").endswith(".distcp")
    }
    manifest_shards = manifest.get("source_shards")
    if not isinstance(manifest_shards, list) or not manifest_shards:
        raise RuntimeError("conversion manifest has no source-shard inventory")
    manifest_set = {
        _canonical_dcp_shard_name(
            name, "manifest source shard", expected_shards
        )
        for name in manifest_shards
    }
    if len(manifest_set) != len(manifest_shards):
        raise RuntimeError("conversion manifest contains duplicate source shards")
    if manifest_set != expected:
        raise RuntimeError(
            "conversion manifest does not contain the exact contiguous "
            f"{expected_shards}-shard inventory"
        )
    actual = {path.name for path in checkpoint.glob("*.distcp") if path.is_file()}
    if referenced != expected or actual != expected:
        raise RuntimeError(
            "source shard inventory mismatch between metadata, manifest, and disk"
        )
    if any((checkpoint / name).stat().st_size <= 0 for name in manifest_set):
        raise RuntimeError("source checkpoint contains an empty shard")


def _validate_converted_inventory(
    converted: Path,
    manifest: Mapping[str, Any],
    spec: LlamaSpec,
) -> dict[str, str]:
    index_path = converted / "model.safetensors.index.json"
    config_path = converted / "config.json"
    if sha256_file(index_path) != _require_sha256(
        manifest.get("hf_index_sha256"), "manifest HF-index SHA-256"
    ):
        raise RuntimeError("converted HF index hash differs from conversion manifest")
    if sha256_file(config_path) != _require_sha256(
        manifest.get("hf_config_sha256"), "manifest HF-config SHA-256"
    ):
        raise RuntimeError("converted HF config hash differs from conversion manifest")
    if _load_json(config_path) != hf_config(spec, dtype="bfloat16"):
        raise RuntimeError(
            "converted HF model configuration differs from route contract"
        )

    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("converted HF index has no weight map")
    expected_shapes = expected_hf_shapes(spec)
    if set(weight_map) != set(expected_shapes):
        raise RuntimeError("converted HF tensor-key inventory is not exact")
    normalized_map = {
        key: _safe_filename(filename, "HF weight filename")
        for key, filename in weight_map.items()
    }

    weight_hashes = manifest.get("weight_files")
    if not isinstance(weight_hashes, dict) or not weight_hashes:
        raise RuntimeError("conversion manifest has no weight-file hashes")
    normalized_hashes = {
        _safe_filename(name, "manifest weight filename"): _require_sha256(
            digest, f"weight hash for {name}"
        )
        for name, digest in weight_hashes.items()
    }
    indexed_files = set(normalized_map.values())
    actual_files = {
        path.name for path in converted.glob("*.safetensors") if path.is_file()
    }
    if indexed_files != set(normalized_hashes) or actual_files != set(
        normalized_hashes
    ):
        raise RuntimeError("converted weight-file inventory is not exact")

    for name, wanted_hash in sorted(normalized_hashes.items()):
        path = converted / name
        if sha256_file(path) != wanted_hash:
            raise RuntimeError(f"converted weight hash mismatch: {name}")
        expected_members = {
            key for key, filename in normalized_map.items() if filename == name
        }
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_members:
                raise RuntimeError(f"converted weight member mismatch: {name}")
            for key in expected_members:
                tensor_slice = handle.get_slice(key)
                if (
                    tuple(tensor_slice.get_shape()) != expected_shapes[key]
                    or tensor_slice.get_dtype() != "BF16"
                ):
                    raise RuntimeError(
                        f"converted tensor shape or dtype mismatch: {key}"
                    )

    tokenizer_hashes = manifest.get("tokenizer_sha256")
    if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
        raise RuntimeError("conversion manifest has no tokenizer hashes")
    for name, wanted_hash in sorted(tokenizer_hashes.items()):
        safe_name = _safe_filename(name, "tokenizer filename")
        if sha256_file(converted / safe_name) != _require_sha256(
            wanted_hash, f"tokenizer hash for {safe_name}"
        ):
            raise RuntimeError(f"converted tokenizer hash mismatch: {safe_name}")
    return normalized_map


def validate_inputs(
    args: argparse.Namespace, spec: LlamaSpec = LLAMA3_8B
) -> tuple[dict[str, Any], Any, dict[str, str], str]:
    checkpoint = args.checkpoint.resolve()
    converted = args.converted.resolve()
    if not checkpoint.is_dir() or not converted.is_dir():
        raise RuntimeError("checkpoint and converted model must both be directories")
    manifest_path = converted / "conversion_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported conversion-manifest schema")
    if manifest.get("route") != args.expected_route:
        raise RuntimeError("conversion-manifest route differs from expected route")
    if manifest.get("source_job_id") != args.expected_source_job_id:
        raise RuntimeError("conversion-manifest source job differs from expected job")
    if SOURCE_JOB_PATTERN.fullmatch(args.expected_source_job_id) is None:
        raise RuntimeError("expected source job ID is malformed")
    expected_uri_hash = _require_sha256(
        args.expected_source_uri_sha256, "expected source URI SHA-256"
    )
    if manifest.get("source_uri_sha256") != expected_uri_hash:
        raise RuntimeError(
            "conversion-manifest source URI binding differs from expected"
        )
    expected_metadata_hash = _require_sha256(
        args.expected_metadata_sha256, "expected checkpoint metadata SHA-256"
    )
    metadata_path = checkpoint / ".metadata"
    if sha256_file(metadata_path) != expected_metadata_hash:
        raise RuntimeError("source checkpoint metadata differs from expected hash")
    if manifest.get("checkpoint_metadata_sha256") != expected_metadata_hash:
        raise RuntimeError("conversion manifest is bound to different source metadata")
    if manifest.get("source_dtype") != "torch.float32":
        raise RuntimeError("parity gate requires an FP32-master source checkpoint")
    if manifest.get("output_dtype") != "torch.bfloat16":
        raise RuntimeError("parity gate requires a BF16 converted checkpoint")
    if manifest.get("transformers_version") != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError("conversion-manifest Transformers version drift")
    if version("transformers") != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError("runtime Transformers version drift")

    metadata = get_metadata(checkpoint)
    route_validation = validate_route_metadata(
        metadata,
        args.expected_route,
        spec=spec,
        expected_dtype=torch.float32,
        require_optimizer=True,
    )
    expected_model_tensors = len(route_model_shapes(args.expected_route, spec))
    expected_optimizer_parameters = len(
        route_trainable_shapes(args.expected_route, spec)
    )
    expected_frozen_aliases = len(route_alias_keys(args.expected_route, spec))
    expected_extra_state_tensors = len(
        route_extra_state_shapes(args.expected_route, spec)
    )
    expected_manifest_fields = {
        "model_tensors": expected_model_tensors,
        "optimizer_parameters": expected_optimizer_parameters,
        "frozen_aliases": expected_frozen_aliases,
        "aliases_value_checked": expected_frozen_aliases,
        "hf_tensors": 291,
        "hf_tensor_bytes": 16_060_522_496,
    }
    if expected_extra_state_tensors or "extra_state_tensors" in manifest:
        expected_manifest_fields["extra_state_tensors"] = expected_extra_state_tensors
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"conversion-manifest field mismatch for {field}: "
                f"{manifest.get(field)!r} != {expected}"
            )
    if (
        route_validation.model_tensors != manifest["model_tensors"]
        or route_validation.extra_state_tensors != expected_extra_state_tensors
        or route_validation.optimizer_parameters != manifest["optimizer_parameters"]
        or route_validation.frozen_aliases != manifest["frozen_aliases"]
    ):
        raise RuntimeError("conversion manifest differs from source route validation")
    _validate_source_inventory(
        checkpoint, metadata, manifest, args.expected_shards
    )

    step = _read_scalar(
        checkpoint, metadata, "train_state.step", args.expected_shards
    )
    ntokens_seen = _read_scalar(
        checkpoint, metadata, "train_state.ntokens_seen", args.expected_shards
    )
    directory_match = re.fullmatch(r"step-([0-9]+)", checkpoint.name)
    if directory_match is None or int(directory_match.group(1)) != step:
        raise RuntimeError(
            "source checkpoint directory does not match train-state step"
        )
    if step != args.expected_step or ntokens_seen != args.expected_ntokens_seen:
        raise RuntimeError(
            f"source train-state mismatch: step={step} ntokens_seen={ntokens_seen}"
        )
    if manifest.get("train_state") != {"step": step, "ntokens_seen": ntokens_seen}:
        raise RuntimeError("conversion manifest is bound to different train state")

    weight_map = _validate_converted_inventory(converted, manifest, spec)
    return manifest, metadata, weight_map, manifest_sha


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_pinned_torchtitan(root: Path):
    root = root.resolve()
    package = root / "torchtitan" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"initialized TorchTitan package is absent: {package}")
    commit = validate_torchtitan_gitlink_marker(root)
    if commit != SUPPORTED_TORCHTITAN_COMMIT:
        raise RuntimeError(
            f"TorchTitan commit mismatch: {commit} != {SUPPORTED_TORCHTITAN_COMMIT}"
        )

    root_text = str(root)
    sys.path.insert(0, root_text)
    from torchtitan.models.llama3 import (  # type: ignore[import-not-found]
        Llama3StateDictAdapter,
        Transformer,
        TransformerModelArgs,
    )
    from torchtitan.models.attention import (  # type: ignore[import-not-found]
        ScaledDotProductAttentionWrapper,
    )

    resolved = Path(sys.modules[Transformer.__module__].__file__).resolve()
    if root not in resolved.parents:
        raise RuntimeError(
            f"TorchTitan import resolved outside pinned root: {resolved}"
        )
    return (
        TransformerModelArgs,
        Transformer,
        Llama3StateDictAdapter,
        ScaledDotProductAttentionWrapper,
        commit,
    )


def _independent_unfused_tensors(
    key: str,
    tensor: torch.Tensor,
    route: str,
    spec: LlamaSpec,
) -> dict[str, torch.Tensor]:
    """Defuse without calling the conversion helper under test."""

    if route in UNFUSED_ROUTES:
        wanted = bf16_unfused_shapes(spec)
        if key not in wanted or tuple(tensor.shape) != wanted.get(key):
            raise RuntimeError(f"invalid source {route} tensor: {key}")
        return {key: tensor}
    if route not in FUSED_ROUTES:
        raise RuntimeError(f"unsupported source route: {route}")

    wanted = route_model_shapes(route, spec)
    if key not in wanted or tuple(tensor.shape) != wanted.get(key):
        raise RuntimeError(f"invalid source {route} tensor: {key}")
    if key in route_alias_keys(route, spec):
        raise RuntimeError(f"frozen alias unexpectedly entered trainable stream: {key}")
    if key in {"tok_embeddings.weight", "norm.weight", "output.weight"}:
        return {key: tensor}

    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit():
        raise RuntimeError(f"malformed layer tensor name: {key}")
    prefix = f"layers.{parts[1]}."
    suffix = parts[2]
    if suffix == "attention.fused.norm_weight":
        return {prefix + "attention_norm.weight": tensor}
    if suffix == "attention.fused.w_qkv":
        q, k, v = tensor.split((spec.q_rows, spec.kv_rows, spec.kv_rows), dim=0)
        return {
            prefix + "attention.wq.weight": q,
            prefix + "attention.wk.weight": k,
            prefix + "attention.wv.weight": v,
        }
    if suffix == "attention.fused.wo_weight":
        return {prefix + "attention.wo.weight": tensor}
    if suffix == "feed_forward.norm_weight":
        return {prefix + "ffn_norm.weight": tensor}
    feed_forward = {
        "feed_forward.w1_weight": "feed_forward.w1.weight",
        "feed_forward.w2_weight": "feed_forward.w2.weight",
        "feed_forward.w3_weight": "feed_forward.w3.weight",
    }
    if suffix in feed_forward:
        return {prefix + feed_forward[suffix]: tensor}
    raise RuntimeError(f"unhandled {route} source tensor: {key}")


def _set_parameter(model: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        child = module._modules.get(part)
        if child is None:
            raise RuntimeError(f"TorchTitan model has no module for parameter: {name}")
        module = child
    leaf = parts[-1]
    if leaf not in module._parameters:
        raise RuntimeError(f"TorchTitan model has no parameter slot: {name}")
    setattr(module, leaf, torch.nn.Parameter(value, requires_grad=False))


def _converted_tensor(
    converted: Path, weight_map: Mapping[str, str], name: str
) -> torch.Tensor:
    filename = weight_map.get(name)
    if filename is None:
        raise RuntimeError(f"converted HF tensor is absent from index: {name}")
    with safe_open(converted / filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _assert_frozen_aliases(checkpoint: Path, route: str, spec: LlamaSpec) -> int:
    wanted = route_alias_keys(route, spec)
    if not wanted:
        return 0
    seen: set[str] = set()
    for key, tensor in stream_checkpoint_reader(
        checkpoint,
        batch_tensors=1,
        tensors_to_load=sorted(wanted),
        progress=False,
    ):
        if tensor.dtype is not torch.float32 or not torch.equal(
            tensor, torch.ones_like(tensor)
        ):
            raise RuntimeError(f"{route} frozen alias is not exact FP32 ones: {key}")
        seen.add(key)
    if seen != wanted:
        raise RuntimeError(f"not every {route} frozen alias was read")
    return len(seen)


def _all_finite(tensor: torch.Tensor, chunk_elements: int = 16 * 1024 * 1024) -> bool:
    flat = tensor.reshape(-1)
    return all(
        bool(torch.isfinite(flat[start : start + chunk_elements]).all())
        for start in range(0, flat.numel(), chunk_elements)
    )


def build_native_reference(
    checkpoint: Path,
    converted: Path,
    route: str,
    weight_map: Mapping[str, str],
    device: torch.device,
    token_count: int,
    torchtitan_types: tuple[Any, Any, Any, Any, str],
    spec: LlamaSpec = LLAMA3_8B,
) -> tuple[torch.nn.Module, dict[str, int]]:
    (
        TransformerModelArgs,
        Transformer,
        Llama3StateDictAdapter,
        ScaledDotProductAttentionWrapper,
        _,
    ) = torchtitan_types
    model_args = TransformerModelArgs(
        dim=spec.dim,
        n_layers=spec.layers,
        n_heads=spec.heads,
        n_kv_heads=spec.kv_heads,
        vocab_size=spec.vocab_size,
        multiple_of=1024,
        ffn_dim_multiplier=1.3,
        norm_eps=spec.norm_eps,
        rope_theta=spec.rope_theta,
        rope_scaling_args=None,
        max_seq_len=max(8192, token_count),
        use_flex_attn=False,
        attn_mask_type="causal",
    )
    with torch.device("meta"):
        model = Transformer(model_args)
    attention_modules = [
        module
        for module in model.modules()
        if isinstance(module, ScaledDotProductAttentionWrapper)
    ]
    if len(attention_modules) != spec.layers:
        raise RuntimeError(
            "native reference does not contain one SDPA module per layer"
        )
    for module in attention_modules:
        module.sdpa_backends = [SDPBackend.MATH]
    adapter = Llama3StateDictAdapter(model_args, None)
    expected_native = {
        name: tuple(value.shape) for name, value in model.named_parameters()
    }
    expected_hf = expected_hf_shapes(spec)
    loaded_native: set[str] = set()
    compared_hf: set[str] = set()
    compared_elements = 0

    frozen_aliases_checked = _assert_frozen_aliases(checkpoint, route, spec)
    source_keys = sorted(route_trainable_shapes(route, spec))
    for source_index, (source_key, source_tensor) in enumerate(
        stream_checkpoint_reader(
            checkpoint,
            batch_tensors=1,
            tensors_to_load=source_keys,
            progress=False,
        ),
        start=1,
    ):
        if source_tensor.dtype is not torch.float32 or not _all_finite(source_tensor):
            raise RuntimeError(f"source tensor is not finite FP32: {source_key}")
        native_tensors = _independent_unfused_tensors(
            source_key, source_tensor, route, spec
        )
        for native_name, native_tensor in native_tensors.items():
            if native_name in loaded_native:
                raise RuntimeError(
                    f"native parameter was produced twice: {native_name}"
                )
            if tuple(native_tensor.shape) != expected_native.get(native_name):
                raise RuntimeError(f"native parameter shape mismatch: {native_name}")

            adapted = adapter.to_hf({native_name: native_tensor})
            if len(adapted) != 1:
                raise RuntimeError(
                    f"TorchTitan adapter output is not unique: {native_name}"
                )
            hf_name, expected_tensor = next(iter(adapted.items()))
            if hf_name in compared_hf or tuple(
                expected_tensor.shape
            ) != expected_hf.get(hf_name):
                raise RuntimeError(f"invalid TorchTitan adapter destination: {hf_name}")
            actual_tensor = _converted_tensor(converted, weight_map, hf_name)
            expected_bf16 = expected_tensor.to(torch.bfloat16).contiguous()
            if actual_tensor.dtype is not torch.bfloat16 or not torch.equal(
                actual_tensor, expected_bf16
            ):
                raise RuntimeError(
                    f"converted tensor differs from TorchTitan adapter output: {hf_name}"
                )
            compared_elements += actual_tensor.numel()
            compared_hf.add(hf_name)

            _set_parameter(
                model,
                native_name,
                native_tensor.to(device=device, dtype=torch.bfloat16),
            )
            loaded_native.add(native_name)
            del actual_tensor, expected_bf16
        if (
            source_index == 1
            or source_index % 16 == 0
            or source_index == len(source_keys)
        ):
            print(
                "[LLAMA PARITY STATE] "
                f"source={source_index}/{len(source_keys)} "
                f"native={len(loaded_native)}/291 converted={len(compared_hf)}/291",
                flush=True,
            )

    if loaded_native != set(expected_native):
        raise RuntimeError("source DCP did not populate the exact TorchTitan model")
    if compared_hf != set(expected_hf):
        raise RuntimeError("TorchTitan adapter did not cover the exact converted model")
    for name, parameter in model.named_parameters():
        if (
            parameter.is_meta
            or parameter.device != device
            or parameter.dtype is not torch.bfloat16
        ):
            raise RuntimeError(f"native parameter was not materialized as BF16: {name}")

    model.freqs_cis = model._precompute_freqs_cis().to(device)  # noqa: SLF001
    model.eval()
    return model, {
        "source_tensors_streamed": len(source_keys),
        "native_parameters_loaded": len(loaded_native),
        "converted_tensors_exact": len(compared_hf),
        "converted_elements_exact": compared_elements,
        "frozen_aliases_checked": frozen_aliases_checked,
        "native_math_sdpa_modules": len(attention_modules),
    }


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("parity gate requires an NVIDIA B200 CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_device(index)
    resolved = torch.device("cuda", index)
    if "B200" not in torch.cuda.get_device_name(resolved).upper() or list(
        torch.cuda.get_device_capability(resolved)
    ) != [10, 0]:
        raise RuntimeError("parity gate requires an NVIDIA B200 CUDA device")
    return resolved


def _tt_equivalent_hf_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
    *,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply TorchTitan's interleaved complex RoPE to HF-permuted Q/K.

    TorchTitan's state-dict adapter permutes each Q/K head from adjacent
    real/imaginary pairs to Transformers' half-split RoPE layout.  That mapping
    is mathematically exact, but stock Transformers then performs different
    BF16 multiply/add operations and presents a permuted reduction dimension to
    SDPA.  Re-interleaving before TorchTitan's complex64 multiply isolates
    semantic conversion parity from that expected evaluator implementation
    drift.  The result intentionally stays interleaved for SDPA.
    """

    if query.ndim != 4 or key.ndim != 4:
        raise RuntimeError("canonical RoPE requires [batch, heads, sequence, dim]")
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16:
        raise RuntimeError("canonical RoPE requires BF16 query and key tensors")
    if query.device != key.device or query.device != freqs_cis.device:
        raise RuntimeError("canonical RoPE tensors must share one device")
    if freqs_cis.dtype is not torch.complex64 or freqs_cis.ndim != 2:
        raise RuntimeError("canonical RoPE frequencies must be rank-2 complex64")
    if unsqueeze_dim != 1:
        raise RuntimeError("canonical RoPE requires the pinned HF head dimension")
    if query.shape[0] != key.shape[0] or query.shape[-2:] != key.shape[-2:]:
        raise RuntimeError("canonical RoPE query/key batch, sequence, or dim drift")

    batch, _, sequence, head_dim = query.shape
    if head_dim <= 0 or head_dim % 2 or freqs_cis.shape[1] * 2 != head_dim:
        raise RuntimeError("canonical RoPE head dimension is inconsistent")
    if sequence <= 0 or sequence > freqs_cis.shape[0]:
        raise RuntimeError("canonical RoPE sequence exceeds pinned frequencies")
    if position_ids is not None:
        if position_ids.shape[0] not in {1, batch}:
            raise RuntimeError(
                "canonical RoPE position batch must be one or the query batch"
            )
        expected_positions = torch.arange(sequence, device=query.device).expand(
            position_ids.shape[0], sequence
        )
        if position_ids.shape != expected_positions.shape or not torch.equal(
            position_ids, expected_positions
        ):
            raise RuntimeError("canonical RoPE requires contiguous positions from zero")

    active_freqs = freqs_cis[:sequence]
    for name, actual in (("cos", cos), ("sin", sin)):
        if (
            actual.ndim != 3
            or actual.shape[0] not in {1, batch}
            or actual.shape[1:] != (sequence, head_dim)
            or actual.device != query.device
            or actual.dtype is not torch.bfloat16
            or not torch.isfinite(actual).all()
            or not torch.equal(
                actual[..., : head_dim // 2], actual[..., head_dim // 2 :]
            )
            or not torch.equal(actual, actual[:1].expand_as(actual))
        ):
            raise RuntimeError(
                f"canonical RoPE {name} carrier structure is invalid"
            )
    # HF computes these carriers before calling apply_rotary_pos_emb.  They are
    # intentionally not used by the canonical path: Torch/PyTorch versions can
    # round long-sequence sin/cos differently even with the same pinned config.
    # Validate that HF supplied a well-formed phase carrier, then apply the
    # independently pinned TorchTitan complex64 frequencies below.
    magnitude = cos[..., : head_dim // 2].float().square() + sin[
        ..., : head_dim // 2
    ].float().square()
    if not torch.allclose(
        magnitude,
        torch.ones_like(magnitude),
        rtol=0.0,
        atol=0.015625,
    ):
        raise RuntimeError("canonical RoPE carrier is not on the unit circle")

    def rotate(value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        interleaved = torch.stack(
            (value[..., :half], value[..., half:]), dim=-1
        ).flatten(-2)
        complex_value = torch.view_as_complex(
            interleaved.float().reshape(*interleaved.shape[:-1], -1, 2)
        )
        frequencies = active_freqs.view(1, 1, sequence, -1)
        return (
            torch.view_as_real(complex_value * frequencies).flatten(-2).to(value.dtype)
        )

    return rotate(query), rotate(key)


@contextmanager
def _canonical_hf_rope(freqs_cis: torch.Tensor):
    """Temporarily replace only the pinned HF Llama RoPE implementation."""

    import transformers.models.llama.modeling_llama as modeling_llama

    original = modeling_llama.apply_rotary_pos_emb

    def canonical(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _tt_equivalent_hf_rope(
            query,
            key,
            cos,
            sin,
            position_ids,
            unsqueeze_dim,
            freqs_cis=freqs_cis,
        )

    modeling_llama.apply_rotary_pos_emb = canonical
    try:
        yield
    finally:
        unchanged = modeling_llama.apply_rotary_pos_emb is canonical
        modeling_llama.apply_rotary_pos_emb = original
        if not unchanged:
            raise RuntimeError(
                "canonical HF RoPE binding changed during parity forward"
            )


@contextmanager
def _canonical_hf_rmsnorm():
    """Use TorchTitan RMSNorm arithmetic only for the canonical HF forward."""

    import transformers.models.llama.modeling_llama as modeling_llama

    original = modeling_llama.LlamaRMSNorm.forward

    def canonical(module, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dtype is not torch.bfloat16:
            raise RuntimeError("canonical HF RMSNorm requires BF16 activations")
        weight = module.weight
        if (
            weight.dtype is not torch.bfloat16
            or weight.device != hidden_states.device
            or weight.ndim != 1
            or weight.shape[0] != hidden_states.shape[-1]
        ):
            raise RuntimeError("canonical HF RMSNorm weight contract drift")
        return torch.nn.functional.rms_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weight,
            module.variance_epsilon,
        )

    modeling_llama.LlamaRMSNorm.forward = canonical
    try:
        yield
    finally:
        unchanged = modeling_llama.LlamaRMSNorm.forward is canonical
        modeling_llama.LlamaRMSNorm.forward = original
        if not unchanged:
            raise RuntimeError(
                "canonical HF RMSNorm binding changed during parity forward"
            )


def _run_transformers(
    converted: Path,
    token_ids: torch.Tensor,
    device: torch.device,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        converted,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    import transformers.models.llama.modeling_llama as modeling_llama

    rmsnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, modeling_llama.LlamaRMSNorm)
    ]
    if len(rmsnorm_modules) != 65:
        raise RuntimeError(
            "canonical HF model does not contain the exact 65 Llama RMSNorm modules"
        )
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH], set_priority=True):
        stock_logits = model(input_ids=token_ids, use_cache=False).logits.float().cpu()
    with (
        _canonical_hf_rope(freqs_cis),
        _canonical_hf_rmsnorm(),
        torch.inference_mode(),
        sdpa_kernel([SDPBackend.MATH], set_priority=True),
    ):
        canonical_logits = (
            model(input_ids=token_ids, use_cache=False).logits.float().cpu()
        )
    del model
    return stock_logits, canonical_logits


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _code_bundle(
    torchtitan_root: Path, project_root: Path
) -> tuple[dict[str, str], str]:
    files = {
        "parity_tool": Path(__file__).resolve(),
        "parity_receipt_module": project_root
        / "low_bits_training/analysis/llama_conversion_parity.py",
        "checkpoint_routes": project_root
        / "low_bits_training/analysis/llama_checkpoint_routes.py",
        "checkpoint_streamer": project_root
        / "low_bits_training/analysis/stream_checkpoints.py",
        "torchtitan_llama_model": torchtitan_root
        / "torchtitan/models/llama3/model/model.py",
        "torchtitan_llama_args": torchtitan_root
        / "torchtitan/models/llama3/model/args.py",
        "torchtitan_llama_adapter": torchtitan_root
        / "torchtitan/models/llama3/model/state_dict_adapter.py",
        "torchtitan_attention": torchtitan_root / "torchtitan/models/attention.py",
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    return hashes, sha256_bytes(canonical_json_bytes(hashes))


def _environment(
    device: torch.device,
    torchtitan_commit: str,
    project_root: Path,
) -> dict[str, Any]:
    try:
        project_commit = _git(project_root, "rev-parse", "HEAD")
        project_dirty = bool(
            _git(project_root, "status", "--porcelain", "--untracked-files=no")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        project_commit = None
        project_dirty = None
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "safetensors": _package_version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "torchtitan_commit": torchtitan_commit,
        "project_git_commit": project_commit,
        "project_tracked_dirty": project_dirty,
        "native_attention": "TorchTitan scaled_dot_product_attention causal math",
        "converted_attention": "Transformers SDPA causal math",
        "attention_backend": "SDPBackend.MATH",
        "compute_dtype": "torch.bfloat16",
        "canonical_semantic_rope": (
            "TorchTitan interleaved complex64 RoPE in converted HF model"
        ),
        "canonical_semantic_rmsnorm": (
            "TorchTitan torch.nn.functional.rms_norm in converted HF model"
        ),
        "stock_hf_rope": "Transformers half-split BF16 RoPE",
        "stock_hf_rmsnorm": "Transformers LlamaRMSNorm FP32-normalize BF16-scale",
    }
    if device.type == "cuda":
        environment.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return environment


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.receipt.exists():
        raise RuntimeError(f"refusing to overwrite parity receipt: {args.receipt}")
    tolerances = CANONICAL_PARITY_TOLERANCES
    semantic_tolerances = CANONICAL_SEMANTIC_TOLERANCES
    token_ids = list(CANONICAL_FIXED_TOKEN_IDS)

    manifest, _, weight_map, manifest_sha = validate_inputs(args)
    device = _device(args.device)
    torchtitan_root = args.torchtitan_root.resolve()
    torchtitan_types = load_pinned_torchtitan(torchtitan_root)
    torchtitan_commit = torchtitan_types[-1]
    project_root = Path(__file__).resolve().parents[2]
    code_files, code_bundle_sha = _code_bundle(torchtitan_root, project_root)

    native_model, state_measurements = build_native_reference(
        args.checkpoint.resolve(),
        args.converted.resolve(),
        args.expected_route,
        weight_map,
        device,
        len(token_ids),
        torchtitan_types,
    )
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    torch.manual_seed(42)
    with torch.inference_mode():
        native_logits = native_model(tokens, attention_masks=None).float().cpu()
    native_freqs_cis = native_model.freqs_cis.detach()
    del native_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stock_logits, canonical_logits = _run_transformers(
        args.converted.resolve(), tokens, device, native_freqs_cis
    )
    stock_measurements = compare_logits(native_logits, stock_logits, tolerances)
    semantic_measurements = compare_semantic_logits(
        native_logits, canonical_logits, semantic_tolerances
    )
    measurements = {
        "passed": (semantic_measurements["passed"] and stock_measurements["passed"]),
        "canonical_semantic": semantic_measurements,
        "stock_hf_evaluator_drift": stock_measurements,
        **state_measurements,
    }
    print(
        "[LLAMA PARITY MEASUREMENTS] "
        + json.dumps(measurements, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    payload = {
        "schema_version": PARITY_RECEIPT_SCHEMA_VERSION,
        "method": PARITY_METHOD,
        "policy": PARITY_POLICY,
        "passed": measurements["passed"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_manifest_sha256": manifest_sha,
        "route": args.expected_route,
        "step": args.expected_step,
        "ntokens_seen": args.expected_ntokens_seen,
        "checkpoint_metadata_sha256": args.expected_metadata_sha256,
        "source_job_id": EXAMPLE,
        "source_uri_sha256": args.expected_source_uri_sha256,
        "fixed_token_ids": token_ids,
        "fixed_token_ids_sha256": token_ids_sha256(token_ids),
        "expected_logits_shape": list(CANONICAL_LOGITS_SHAPE),
        "tool_sha256": code_files["parity_tool"],
        "code_bundle_sha256": code_bundle_sha,
        "code_files_sha256": code_files,
        "environment": _environment(device, torchtitan_commit, project_root),
        "tolerances": {
            "canonical_semantic": semantic_tolerances.to_dict(),
            "stock_hf_evaluator_drift": tolerances.to_dict(),
        },
        "measurements": measurements,
        "limitations": [],
    }
    receipt = seal_receipt(payload)
    if receipt["passed"]:
        validate_receipt(receipt)
    write_atomic_receipt(args.receipt, receipt)
    if not receipt["passed"]:
        raise RuntimeError(
            "fixed-token TorchTitan/Transformers parity failed; "
            f"failure receipt written to {args.receipt}"
        )
    print(
        "[LLAMA CONVERSION PARITY PASS] "
        f"route={args.expected_route} step={args.expected_step} "
        f"semantic_mismatches={semantic_measurements['mismatch_count']} "
        f"stock_max_abs={stock_measurements['max_abs_error']:.8g} "
        f"stock_close_failures={stock_measurements['close_failure_count']} "
        f"stock_top10_min={stock_measurements['top_k_intersection_count_min']} "
        f"stock_top10_total={stock_measurements['top_k_intersection_count_total']} "
        f"receipt={args.receipt}",
        flush=True,
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--expected-route", choices=SUPPORTED_ROUTES, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-ntokens-seen", type=int, required=True)
    parser.add_argument("--expected-source-job-id", required=True)
    parser.add_argument("--expected-source-uri-sha256", required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.expected_step < 0
        or args.expected_ntokens_seen < 0
        or args.expected_shards <= 0
    ):
        raise SystemExit(
            "expected step and token count must be nonnegative and expected "
            "shards must be positive"
        )
    try:
        run(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"[LLAMA CONVERSION PARITY FAIL] {error}") from error


if __name__ == "__main__":
    main()
