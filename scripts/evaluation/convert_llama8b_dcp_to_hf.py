#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Stream an exact Llama-8B DCP route into a common BF16 HF checkpoint.

This converter is intentionally local-filesystem only.  Downloading a remote
checkpoint and obtaining credentials are separate operational steps.  The
source checkpoint is opened read-only, its exact route and optimizer schema are
validated, and the output is published with an atomic directory rename.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import torch
from safetensors.torch import save_file

# This conversion path needs only CPU DCP/analysis utilities.  Avoid importing
# training converters and their Transformer Engine/custom-kernel dependencies.
os.environ["LBT_LIGHT_IMPORT"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_bits_training.analysis.llama_checkpoint_routes import (
    LLAMA3_8B,
    LlamaSpec,
    SUPPORTED_ROUTES,
    detect_route,
    expected_hf_shapes,
    hf_config,
    route_alias_keys,
    route_trainable_shapes,
    tt_to_hf_tensors,
    validate_route_metadata,
)
from low_bits_training.analysis.stream_checkpoints import (
    get_metadata,
    stream_checkpoint_reader,
)


PINNED_TOKENIZER_SHA256 = {
    "tokenizer.json": "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
    "tokenizer_config.json": "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a",
    "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
}

SOURCE_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_source_identity(source_job_id: str, source_uri_sha256: str) -> None:
    """Validate non-secret identity fields carried into durable eval receipts."""

    if SOURCE_JOB_ID_PATTERN.fullmatch(source_job_id) is None:
        raise RuntimeError(f"invalid source job id: {source_job_id!r}")
    if SHA256_PATTERN.fullmatch(source_uri_sha256) is None:
        raise RuntimeError("source URI SHA-256 must be 64 lowercase hex characters")


def _dtype(value: str) -> torch.dtype:
    choices = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    try:
        return choices[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            f"dtype must be one of {sorted(choices)}, got {value!r}"
        ) from error


def validate_tokenizer_assets(
    path: Path, pinned_hashes: dict[str, str] = PINNED_TOKENIZER_SHA256
) -> dict[str, str]:
    if not path.is_dir():
        raise RuntimeError(f"tokenizer asset directory is absent: {path}")
    observed = {}
    for name, wanted in pinned_hashes.items():
        asset = path / name
        if not asset.is_file() or asset.stat().st_size <= 0:
            raise RuntimeError(f"pinned tokenizer asset is absent or empty: {asset}")
        observed[name] = sha256(asset)
        if observed[name] != wanted:
            raise RuntimeError(
                f"tokenizer asset hash mismatch for {name}: {observed[name]} != {wanted}"
            )

    config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    decoder = config.get("added_tokens_decoder", {})
    if (
        config.get("bos_token") != "<|begin_of_text|>"
        or config.get("eos_token") != "<|end_of_text|>"
        or decoder.get("128000", {}).get("content") != "<|begin_of_text|>"
        or decoder.get("128001", {}).get("content") != "<|end_of_text|>"
    ):
        raise RuntimeError("tokenizer special-token contract does not match Llama 3.1")
    return observed


def validate_local_inventory(
    checkpoint: Path, metadata: Any, expected_shards: int
) -> list[str]:
    if not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint directory is absent: {checkpoint}")
    metadata_file = checkpoint / ".metadata"
    if not metadata_file.is_file() or metadata_file.stat().st_size <= 0:
        raise RuntimeError("checkpoint .metadata is absent or empty")
    storage = getattr(metadata, "storage_data", None)
    if not isinstance(storage, dict) or not storage:
        raise RuntimeError("checkpoint metadata has no storage_data mapping")
    referenced = sorted(
        {
            item.relative_path
            for item in storage.values()
            if isinstance(getattr(item, "relative_path", None), str)
            and item.relative_path.endswith(".distcp")
        }
    )
    actual = sorted(path.name for path in checkpoint.glob("*.distcp") if path.is_file())
    if len(referenced) != expected_shards:
        raise RuntimeError(
            f"checkpoint metadata references {len(referenced)} shards, expected {expected_shards}"
        )
    if actual != referenced:
        raise RuntimeError(
            "checkpoint shard inventory differs from metadata: "
            f"missing={sorted(set(referenced) - set(actual))} "
            f"extra={sorted(set(actual) - set(referenced))}"
        )
    empty = [name for name in referenced if (checkpoint / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"checkpoint contains empty shards: {empty}")
    return referenced


def _read_scalar(checkpoint: Path, metadata: Any, fqn: str) -> int:
    storage = getattr(metadata, "storage_data", None)
    if not isinstance(storage, dict):
        raise RuntimeError("checkpoint metadata has no storage_data mapping")
    matches = [
        item for index, item in storage.items() if getattr(index, "fqn", None) == fqn
    ]
    if len(matches) != 1:
        raise RuntimeError(f"checkpoint scalar storage is not unique: {fqn}")
    item = matches[0]
    relative_path = getattr(item, "relative_path", None)
    offset = getattr(item, "offset", None)
    length = getattr(item, "length", None)
    if (
        not isinstance(relative_path, str)
        or not isinstance(offset, int)
        or not isinstance(length, int)
        or offset < 0
        or length <= 0
    ):
        raise RuntimeError(f"checkpoint scalar storage is malformed: {fqn}")
    shard = checkpoint / relative_path
    with shard.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(length)
    if len(payload) != length:
        raise RuntimeError(f"checkpoint scalar storage is truncated: {fqn}")
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"checkpoint scalar is not an integer: {fqn}")
    return value


def validate_train_state(
    checkpoint: Path,
    metadata: Any,
    expected_ntokens_seen: int | None = None,
) -> dict[str, int]:
    match = re.fullmatch(r"step-([0-9]+)", checkpoint.name)
    if match is None:
        raise RuntimeError("checkpoint directory must have the exact name step-N")
    expected_step = int(match.group(1))
    step = _read_scalar(checkpoint, metadata, "train_state.step")
    ntokens_seen = _read_scalar(checkpoint, metadata, "train_state.ntokens_seen")
    if step != expected_step:
        raise RuntimeError(
            f"checkpoint train_state.step does not match directory: {step} != {expected_step}"
        )
    if ntokens_seen < 0:
        raise RuntimeError("checkpoint train_state.ntokens_seen is negative")
    if expected_ntokens_seen is not None and ntokens_seen != expected_ntokens_seen:
        raise RuntimeError(
            "checkpoint train_state.ntokens_seen mismatch: "
            f"{ntokens_seen} != {expected_ntokens_seen}"
        )
    return {"step": step, "ntokens_seen": ntokens_seen}


def _required_output_bytes(dtype: torch.dtype, spec: LlamaSpec = LLAMA3_8B) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return sum(
        int(torch.tensor(shape).prod()) * element_size
        for shape in expected_hf_shapes(spec).values()
    )


def check_output_location(
    checkpoint: Path,
    output: Path,
    dtype: torch.dtype,
    spec: LlamaSpec = LLAMA3_8B,
) -> int:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if output == checkpoint or checkpoint in output.parents:
        raise RuntimeError("output must not be inside the source checkpoint")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise RuntimeError(f"output parent must already exist: {output.parent}")

    required = _required_output_bytes(dtype, spec)
    # Leave 10% plus 2 GiB for safetensors headers, the largest in-flight
    # tensor, evaluation logs, and filesystem accounting variance.
    floor = int(required * 1.10) + 2 * 1024**3
    free = shutil.disk_usage(output.parent).free
    if free < floor:
        raise RuntimeError(
            f"insufficient output space: free={free} required_floor={floor}"
        )
    return required


def _assert_frozen_aliases(
    checkpoint: Path,
    route: str,
    source_dtype: torch.dtype,
    spec: LlamaSpec = LLAMA3_8B,
) -> int:
    aliases = sorted(route_alias_keys(route, spec))
    if not aliases:
        return 0
    seen = set()
    for key, tensor in stream_checkpoint_reader(
        checkpoint, batch_tensors=1, tensors_to_load=aliases, progress=False
    ):
        if tensor.dtype != source_dtype or not torch.equal(
            tensor, torch.ones_like(tensor)
        ):
            raise RuntimeError(
                f"{route} frozen alias is not exact all-ones {source_dtype}: {key}"
            )
        seen.add(key)
    if seen != set(aliases):
        raise RuntimeError(
            f"failed to read every frozen alias: missing={sorted(set(aliases) - seen)}"
        )
    return len(seen)


def _copy_tokenizer_assets(
    source: Path, target: Path, pinned_hashes: dict[str, str]
) -> None:
    for name in pinned_hashes:
        shutil.copyfile(source / name, target / name)


def _all_finite(tensor: torch.Tensor, chunk_elements: int = 16 * 1024 * 1024) -> bool:
    flat = tensor.reshape(-1)
    return all(
        bool(torch.isfinite(flat[start : start + chunk_elements]).all())
        for start in range(0, flat.numel(), chunk_elements)
    )


def convert(
    checkpoint: Path,
    output: Path,
    tokenizer_assets: Path,
    source_job_id: str,
    source_uri_sha256: str,
    *,
    route: str = "auto",
    source_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.bfloat16,
    expected_shards: int = 32,
    expected_ntokens_seen: int | None = None,
    spec: LlamaSpec = LLAMA3_8B,
    pinned_tokenizer_hashes: dict[str, str] = PINNED_TOKENIZER_SHA256,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    tokenizer_assets = tokenizer_assets.resolve()
    validate_source_identity(source_job_id, source_uri_sha256)
    expected_output_bytes = check_output_location(checkpoint, output, output_dtype, spec)
    tokenizer_hashes = validate_tokenizer_assets(
        tokenizer_assets, pinned_tokenizer_hashes
    )

    metadata = get_metadata(checkpoint)
    selected_route = detect_route(metadata, spec) if route == "auto" else route
    validation = validate_route_metadata(
        metadata,
        selected_route,
        spec=spec,
        expected_dtype=source_dtype,
        require_optimizer=True,
    )
    referenced_shards = validate_local_inventory(checkpoint, metadata, expected_shards)
    train_state = validate_train_state(checkpoint, metadata, expected_ntokens_seen)
    aliases_checked = _assert_frozen_aliases(
        checkpoint, selected_route, source_dtype, spec
    )

    source_keys = sorted(route_trainable_shapes(selected_route, spec))
    wanted_hf = expected_hf_shapes(spec)
    suffix = "BF16" if output_dtype == torch.bfloat16 else "F32"
    temp_raw = tempfile.mkdtemp(prefix=f".{output.name}.incomplete.", dir=output.parent)
    temp = Path(temp_raw)
    try:
        weight_map: dict[str, str] = {}
        file_hashes: dict[str, str] = {}
        observed_source = set()
        total_files = len(source_keys)
        iterator = stream_checkpoint_reader(
            checkpoint,
            batch_tensors=1,
            tensors_to_load=source_keys,
            progress=False,
        )
        for index, (source_key, tensor) in enumerate(iterator, start=1):
            if source_key in observed_source:
                raise RuntimeError(f"source tensor was streamed twice: {source_key}")
            observed_source.add(source_key)
            tensors = tt_to_hf_tensors(
                source_key,
                tensor,
                selected_route,
                spec=spec,
                output_dtype=output_dtype,
            )
            overlap = set(tensors) & set(weight_map)
            if overlap:
                raise RuntimeError(
                    f"HF destination tensors were produced twice: {sorted(overlap)}"
                )
            for name, value in tensors.items():
                if name not in wanted_hf or tuple(value.shape) != wanted_hf[name]:
                    raise RuntimeError(
                        f"unexpected converted tensor {name}: shape={tuple(value.shape)}"
                    )
                if value.dtype != output_dtype or not _all_finite(value):
                    raise RuntimeError(
                        f"converted tensor is nonfinite or wrong dtype: {name}"
                    )

            filename = f"model-{index:05d}-of-{total_files:05d}.safetensors"
            file_path = temp / filename
            save_file(tensors, file_path, metadata={"format": "pt", "dtype": suffix})
            file_hashes[filename] = sha256(file_path)
            for name in tensors:
                weight_map[name] = filename
            print(
                f"[{index}/{total_files}] {source_key} -> {','.join(sorted(tensors))}",
                flush=True,
            )

        if observed_source != set(source_keys):
            raise RuntimeError(
                f"not all source tensors were read: missing={sorted(set(source_keys) - observed_source)}"
            )
        if set(weight_map) != set(wanted_hf):
            raise RuntimeError(
                "converted HF tensor manifest mismatch: "
                f"missing={sorted(set(wanted_hf) - set(weight_map))} "
                f"extra={sorted(set(weight_map) - set(wanted_hf))}"
            )

        actual_output_bytes = sum(
            int(torch.tensor(shape).prod())
            * torch.empty((), dtype=output_dtype).element_size()
            for shape in wanted_hf.values()
        )
        if actual_output_bytes != expected_output_bytes:
            raise RuntimeError("internal converted byte-count contract changed")
        index_document = {
            "metadata": {"total_size": actual_output_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (temp / "model.safetensors.index.json").write_text(
            json.dumps(index_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_dtype = "bfloat16" if output_dtype == torch.bfloat16 else "float32"
        (temp / "config.json").write_text(
            json.dumps(hf_config(spec, dtype=config_dtype), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _copy_tokenizer_assets(tokenizer_assets, temp, pinned_tokenizer_hashes)

        manifest = {
            "schema_version": 1,
            "converter_sha256": sha256(Path(__file__).resolve()),
            "source_job_id": source_job_id,
            "source_uri_sha256": source_uri_sha256,
            "checkpoint_metadata_sha256": sha256(checkpoint / ".metadata"),
            "route": validation.route,
            "model_tensors": validation.model_tensors,
            "extra_state_tensors": validation.extra_state_tensors,
            "optimizer_parameters": validation.optimizer_parameters,
            "frozen_aliases": validation.frozen_aliases,
            "aliases_value_checked": aliases_checked,
            "source_dtype": str(source_dtype),
            "output_dtype": str(output_dtype),
            "source_shards": referenced_shards,
            "train_state": train_state,
            "hf_tensors": len(weight_map),
            "hf_tensor_bytes": actual_output_bytes,
            "weight_files": file_hashes,
            "hf_index_sha256": sha256(temp / "model.safetensors.index.json"),
            "hf_config_sha256": sha256(temp / "config.json"),
            "tokenizer_sha256": tokenizer_hashes,
            "transformers_version": "4.48.2",
            "lm_eval_version": "0.4.12",
        }
        (temp / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print(
        f"[LLAMA DCP TO HF PASS] route={selected_route} output={output} "
        f"tensors={len(wanted_hf)} dtype={output_dtype}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-assets",
        type=Path,
        required=True,
        help="Pinned Llama-3.1-8B tokenizer.json/config/special-token directory",
    )
    parser.add_argument(
        "--source-job-id",
        required=True,
        help="Non-secret durable training job identifier",
    )
    parser.add_argument(
        "--source-uri-sha256",
        required=True,
        help="SHA-256 of the redacted canonical source checkpoint URI",
    )
    parser.add_argument("--route", choices=("auto", *SUPPORTED_ROUTES), default="auto")
    parser.add_argument("--source-dtype", type=_dtype, default=torch.float32)
    parser.add_argument("--output-dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument(
        "--expected-ntokens-seen",
        type=int,
        default=None,
        help="Optional exact rank-local train_state.ntokens_seen contract",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_shards <= 0:
        raise SystemExit("--expected-shards must be positive")
    convert(
        args.checkpoint,
        args.output,
        args.tokenizer_assets,
        args.source_job_id,
        args.source_uri_sha256,
        route=args.route,
        source_dtype=args.source_dtype,
        output_dtype=args.output_dtype,
        expected_shards=args.expected_shards,
        expected_ntokens_seen=args.expected_ntokens_seen,
    )


if __name__ == "__main__":
    main()
