# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Deterministic inputs and bounded-memory reference capture for Llama replay.

The helpers in this module deliberately separate two pieces of evidence:

* a calibration bundle made from hash-bound Arrow files and a local tokenizer;
* sampled forward/backward tensors from an exact converted BF16 checkpoint.

Captures are written per layer and per phase.  Forward samples are released
before the next layer runs and backward samples are released as soon as a
layer's input gradient is produced.  The capture therefore never keeps full
activation copies, nor all sampled layers, in host memory at once.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import torch


SCHEMA_VERSION = 1
CALIBRATION_METHOD = "round-robin-arrow-document-token-quanta-v1"
CAPTURE_METHOD = "canonical-hf-bf16-sampled-layer-replay-v1"
SHA256_HEX = frozenset("0123456789abcdef")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for receipt and ledger hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return value


def load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise ValueError("payload is already sealed")
    result = dict(payload)
    result["receipt_sha256"] = sha256(canonical_json_bytes(result)).hexdigest()
    return result


def validate_receipt_seal(receipt: Mapping[str, Any]) -> None:
    actual = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    unsealed = dict(receipt)
    unsealed.pop("receipt_sha256")
    if sha256(canonical_json_bytes(unsealed)).hexdigest() != actual:
        raise RuntimeError("receipt SHA-256 mismatch")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and exact contiguous CPU bytes."""

    value = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    )
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    return sha256(header + b"\0" + raw).hexdigest()


def tensor_magnitude_summary(
    tensor: torch.Tensor,
    *,
    max_distribution_samples: int = 1 << 20,
) -> dict[str, Any]:
    """Summarize sampled magnitudes for range-versus-precision attribution."""

    if max_distribution_samples <= 0:
        raise ValueError("max_distribution_samples must be positive")
    values = tensor.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        raise RuntimeError("cannot summarize an empty tensor")
    finite = torch.isfinite(values)
    if not bool(finite.all()):
        raise RuntimeError("captured tensor contains a non-finite value")
    magnitudes = values.abs()
    distribution_count = min(values.numel(), max_distribution_samples)
    if distribution_count == values.numel():
        distribution_magnitudes = magnitudes
        distribution_sampling = "exact"
    else:
        # Select the midpoint of each equal-width interval. This is bounded,
        # deterministic, covers the full flattened tensor, and avoids the
        # large-tensor limit in torch.quantile.
        positions = torch.arange(distribution_count, dtype=torch.int64)
        positions = torch.div(
            (2 * positions + 1) * values.numel(),
            2 * distribution_count,
            rounding_mode="floor",
        )
        distribution_magnitudes = magnitudes.index_select(0, positions)
        distribution_sampling = "deterministic-stratified-midpoint-v1"
    quantile_points = torch.tensor(
        [0.5, 0.9, 0.99, 0.999], dtype=torch.float32
    )
    quantiles = torch.quantile(distribution_magnitudes, quantile_points)
    energy = values.square()
    total_energy = energy.sum()
    distribution_energy = distribution_magnitudes.square()
    distribution_total_energy = distribution_energy.sum()
    top_count = max(1, (distribution_count + 99) // 100)
    top_energy = torch.topk(
        distribution_energy, top_count, sorted=False
    ).values.sum()
    top_energy_fraction = (
        0.0
        if float(distribution_total_energy) == 0.0
        else float(top_energy / distribution_total_energy)
    )
    return {
        "elements": values.numel(),
        "distribution_sample_elements": distribution_count,
        "distribution_sampling": distribution_sampling,
        "abs_p50": float(quantiles[0]),
        "abs_p90": float(quantiles[1]),
        "abs_p99": float(quantiles[2]),
        "abs_p999": float(quantiles[3]),
        "abs_max": float(magnitudes.max()),
        "rms": float(torch.sqrt(energy.mean())),
        "zero_fraction": float((values == 0).float().mean()),
        "top_1pct_energy_fraction": top_energy_fraction,
        "top_1pct_energy_fraction_is_sampled": distribution_count != values.numel(),
    }


def directory_file_ledger(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"not a directory: {root}")
    ledger: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        ledger[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not ledger:
        raise RuntimeError(f"directory contains no files: {root}")
    return ledger


def safetensors_payload_bytes(path: Path) -> int:
    """Return tensor-data bytes after validating a safetensors offset ledger."""

    size = path.stat().st_size
    if size < 8:
        raise RuntimeError(f"safetensors file is shorter than its header: {path}")
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size <= 0 or 8 + header_size > size:
            raise RuntimeError(f"invalid safetensors header size: {path}")
        try:
            header = json.loads(handle.read(header_size))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(f"invalid safetensors header JSON: {path}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"safetensors header is not an object: {path}")

    intervals: list[tuple[int, int]] = []
    for name, record in header.items():
        if name == "__metadata__":
            if record is not None and not isinstance(record, dict):
                raise RuntimeError(f"invalid safetensors metadata record: {path}")
            continue
        if not isinstance(name, str) or not isinstance(record, dict):
            raise RuntimeError(f"invalid safetensors tensor record: {path}")
        offsets = record.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
        ):
            raise RuntimeError(f"invalid safetensors data offsets for {name}: {path}")
        intervals.append((offsets[0], offsets[1]))
    if not intervals:
        raise RuntimeError(f"safetensors file contains no tensors: {path}")
    intervals.sort()
    cursor = 0
    for start, stop in intervals:
        if start != cursor:
            raise RuntimeError(f"non-contiguous safetensors data offsets: {path}")
        cursor = stop
    if cursor != size - 8 - header_size:
        raise RuntimeError(f"safetensors payload size differs from its offsets: {path}")
    return cursor


def deterministic_row_indices(total_rows: int, sample_rows: int, seed: int) -> list[int]:
    """Select stable rows using SHA-256 ranking, independent of Torch RNG state."""

    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive")
    count = min(total_rows, sample_rows)
    prefix = f"layerwise-replay-row-v1\0{seed}\0".encode()
    selected = heapq.nsmallest(
        count,
        range(total_rows),
        key=lambda index: sha256(prefix + str(index).encode()).digest(),
    )
    return sorted(selected)


def _iter_arrow_text_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow as pa

    row_index = 0
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        names = set(reader.schema.names)
        if "text" not in names:
            raise RuntimeError(f"Arrow source has no text column: {path}")
        for batch in reader:
            text_column = batch.column(reader.schema.get_field_index("text"))
            id_column = (
                batch.column(reader.schema.get_field_index("id"))
                if "id" in names
                else None
            )
            source_column = (
                batch.column(reader.schema.get_field_index("source_ds"))
                if "source_ds" in names
                else None
            )
            for offset in range(batch.num_rows):
                text = text_column[offset].as_py()
                if not isinstance(text, str):
                    raise RuntimeError(
                        f"Arrow text is not a string at {path}:{row_index}"
                    )
                yield {
                    "row": row_index,
                    "text": text,
                    "id": None if id_column is None else id_column[offset].as_py(),
                    "source_ds": (
                        None
                        if source_column is None
                        else source_column[offset].as_py()
                    ),
                }
                row_index += 1


def build_token_batches(
    arrow_paths: Sequence[Path],
    tokenizer: Any,
    *,
    seq_len: int,
    batch_size: int,
    num_batches: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Pack two Arrow sources in deterministic document round-robin order."""

    if len(arrow_paths) != 2:
        raise ValueError("exactly two Arrow files are required")
    if seq_len <= 1 or batch_size <= 0 or num_batches <= 0:
        raise ValueError("seq_len > 1 and positive batch counts are required")
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(bos, int) or not isinstance(eos, int):
        raise RuntimeError("sealed tokenizer must define integer BOS and EOS IDs")

    ordered_paths = sorted((path.resolve() for path in arrow_paths), key=str)
    iterators = [_iter_arrow_text_rows(path) for path in ordered_paths]
    required = num_batches * batch_size * seq_len + 1
    token_quantum = max(1, min(256, (required - 1) // len(iterators)))
    tokens = [bos]
    documents: list[dict[str, Any]] = []
    active = [True, True]
    current_tokens: list[list[int]] = [[], []]
    current_offsets = [0, 0]
    current_document_indices: list[int | None] = [None, None]
    while len(tokens) < required:
        made_progress = False
        for source_index, iterator in enumerate(iterators):
            if not active[source_index] or len(tokens) >= required:
                continue
            if current_offsets[source_index] >= len(current_tokens[source_index]):
                try:
                    row = next(iterator)
                except StopIteration:
                    active[source_index] = False
                    continue
                encoded = tokenizer.encode(row["text"], add_special_tokens=False)
                if not isinstance(encoded, list) or any(
                    isinstance(token, bool) or not isinstance(token, int)
                    for token in encoded
                ):
                    raise RuntimeError(
                        "tokenizer returned a non-integer token sequence"
                    )
                current_tokens[source_index] = [*encoded, eos]
                current_offsets[source_index] = 0
                current_document_indices[source_index] = len(documents)
                documents.append(
                    {
                        "source_index": source_index,
                        "source_file": ordered_paths[source_index].name,
                        "row": row["row"],
                        "id": row["id"],
                        "source_ds": row["source_ds"],
                        "text_sha256": sha256(
                            row["text"].encode("utf-8")
                        ).hexdigest(),
                        "encoded_tokens": len(encoded),
                        "selected_tokens_with_eos": 0,
                        "stream_ranges": [],
                    }
                )
            available = len(current_tokens[source_index]) - current_offsets[source_index]
            take = min(token_quantum, available, required - len(tokens))
            start = len(tokens)
            token_start = current_offsets[source_index]
            tokens.extend(
                current_tokens[source_index][token_start : token_start + take]
            )
            current_offsets[source_index] += take
            document_index = current_document_indices[source_index]
            if document_index is None:
                raise AssertionError("active Arrow source has no current document")
            documents[document_index]["selected_tokens_with_eos"] += take
            documents[document_index]["stream_ranges"].append(
                {
                    "document_token_start": token_start,
                    "document_token_stop": token_start + take,
                    "stream_start": start,
                    "stream_stop": len(tokens),
                }
            )
            made_progress = True
        if not made_progress:
            raise RuntimeError(
                f"Arrow sources exhausted after {len(tokens)} of {required} tokens"
            )

    stream = torch.tensor(tokens[:required], dtype=torch.int64)
    sequence_count = num_batches * batch_size
    inputs = torch.empty((sequence_count, seq_len), dtype=torch.int64)
    targets = torch.empty_like(inputs)
    for sequence in range(sequence_count):
        start = sequence * seq_len
        inputs[sequence] = stream[start : start + seq_len]
        targets[sequence] = stream[start + 1 : start + seq_len + 1]
    payload = {
        "input_ids": inputs.view(num_batches, batch_size, seq_len),
        "target_ids": targets.view(num_batches, batch_size, seq_len),
    }
    return payload, documents


def calibration_tensor_ledger(payload: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if set(payload) != {"input_ids", "target_ids"}:
        raise RuntimeError("calibration tensor inventory is not exact")
    inputs = payload["input_ids"]
    targets = payload["target_ids"]
    if (
        inputs.dtype is not torch.int64
        or targets.dtype is not torch.int64
        or inputs.shape != targets.shape
        or inputs.ndim != 3
        or not torch.equal(inputs.reshape(-1)[1:], targets.reshape(-1)[:-1])
    ):
        raise RuntimeError("calibration tensors violate contiguous next-token packing")
    return {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(payload.items())
    }


def validate_converted_model(
    converted: Path,
    *,
    verify_weight_files: bool = True,
) -> dict[str, Any]:
    """Validate and summarize a converted HF directory against its manifest."""

    converted = converted.resolve()
    manifest_path = converted / "conversion_manifest.json"
    manifest = load_json_object(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    for filename, field in (
        ("config.json", "hf_config_sha256"),
        ("model.safetensors.index.json", "hf_index_sha256"),
    ):
        expected = _require_sha256(manifest.get(field), field)
        if sha256_file(converted / filename) != expected:
            raise RuntimeError(f"converted {filename} SHA-256 mismatch")

    weight_files = manifest.get("weight_files")
    if not isinstance(weight_files, dict) or not weight_files:
        raise RuntimeError("conversion manifest contains no weight-file ledger")
    actual_names = {
        path.name for path in converted.glob("*.safetensors") if path.is_file()
    }
    if actual_names != set(weight_files):
        raise RuntimeError("converted weight-file inventory mismatch")
    total_file_bytes = 0
    total_tensor_bytes = 0
    for filename, expected_value in sorted(weight_files.items()):
        expected = _require_sha256(expected_value, f"weight_files.{filename}")
        path = converted / filename
        total_file_bytes += path.stat().st_size
        total_tensor_bytes += safetensors_payload_bytes(path)
        if verify_weight_files and sha256_file(path) != expected:
            raise RuntimeError(f"converted weight SHA-256 mismatch: {filename}")
    if manifest.get("hf_tensor_bytes") != total_tensor_bytes:
        raise RuntimeError("converted aggregate weight byte count mismatch")

    return {
        "conversion_manifest_sha256": manifest_sha,
        "conversion_route": manifest.get("route"),
        "checkpoint_metadata_sha256": _require_sha256(
            manifest.get("checkpoint_metadata_sha256"),
            "checkpoint_metadata_sha256",
        ),
        "source_uri_sha256": _require_sha256(
            manifest.get("source_uri_sha256"), "source_uri_sha256"
        ),
        "train_state": manifest.get("train_state"),
        "hf_config_sha256": manifest["hf_config_sha256"],
        "hf_index_sha256": manifest["hf_index_sha256"],
        "weight_file_count": len(weight_files),
        "weight_file_bytes": total_file_bytes,
        "weight_tensor_bytes": total_tensor_bytes,
        "weight_ledger_sha256": sha256(canonical_json_bytes(weight_files)).hexdigest(),
        "weight_files_verified": verify_weight_files,
        "transformers_version": manifest.get("transformers_version"),
    }


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
    """Apply the bit-exact canonical RoPE used by the sealed evaluation panel."""

    if query.ndim != 4 or key.ndim != 4:
        raise RuntimeError("canonical RoPE requires [batch, heads, sequence, dim]")
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16:
        raise RuntimeError("canonical RoPE requires BF16 query and key")
    if query.device != key.device or query.device != freqs_cis.device:
        raise RuntimeError("canonical RoPE tensors must share one device")
    if freqs_cis.dtype is not torch.complex64 or freqs_cis.ndim != 2:
        raise RuntimeError("canonical RoPE frequencies must be rank-2 complex64")
    if unsqueeze_dim != 1:
        raise RuntimeError("canonical RoPE requires HF heads at dimension one")
    if query.shape[0] != key.shape[0] or query.shape[-2:] != key.shape[-2:]:
        raise RuntimeError("canonical RoPE query/key geometry mismatch")

    batch, _, sequence, head_dim = query.shape
    if head_dim <= 0 or head_dim % 2 or freqs_cis.shape[1] * 2 != head_dim:
        raise RuntimeError("canonical RoPE head dimension mismatch")
    if sequence <= 0 or sequence > freqs_cis.shape[0]:
        raise RuntimeError("canonical RoPE sequence exceeds frequencies")
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
            raise RuntimeError(f"canonical RoPE {name} carrier structure is invalid")
    # HF computes these carriers before this hook. Long-sequence BF16 carriers
    # can round differently across Torch versions even with one pinned config;
    # the canonical path validates only their structure/unit-circle contract and
    # applies the independently pinned TorchTitan complex64 frequencies below.
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
        interleaved = torch.stack((value[..., :half], value[..., half:]), dim=-1)
        complex_value = torch.view_as_complex(
            interleaved.flatten(-2).float().reshape(*value.shape[:-1], -1, 2)
        )
        frequencies = active_freqs.view(1, 1, sequence, -1)
        return torch.view_as_real(complex_value * frequencies).flatten(-2).to(
            value.dtype
        )

    return rotate(query), rotate(key)


@contextmanager
def canonical_hf_rmsnorm():
    """Temporarily install TorchTitan functional RMSNorm in HF Llama."""

    import transformers.models.llama.modeling_llama as modeling_llama

    original = modeling_llama.LlamaRMSNorm.forward

    def canonical(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
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
            raise RuntimeError("canonical HF RMSNorm binding changed during capture")


@contextmanager
def canonical_hf_rope(
    freqs_cis: torch.Tensor,
    *,
    writer: "LayerCaptureWriter | None" = None,
    expected_layers: int | None = None,
):
    """Install canonical RoPE and optionally sample rotated Q/K and gradients."""

    import transformers.models.llama.modeling_llama as modeling_llama

    original = modeling_llama.apply_rotary_pos_emb
    state = {"calls": 0}

    def canonical(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = state["calls"]
        state["calls"] += 1
        query_out, key_out = _tt_equivalent_hf_rope(
            query,
            key,
            cos,
            sin,
            position_ids,
            unsqueeze_dim,
            freqs_cis=freqs_cis,
        )
        if writer is not None:
            writer.capture_forward(layer, "qkv.q_rope_ref", query_out, "bhsd")
            writer.capture_forward(layer, "qkv.k_rope_ref", key_out, "bhsd")
            writer.register_backward(
                layer, "qkv.q_rope_grad_ref", query_out, "bhsd"
            )
            writer.register_backward(
                layer, "qkv.k_rope_grad_ref", key_out, "bhsd"
            )
        return query_out, key_out

    modeling_llama.apply_rotary_pos_emb = canonical
    try:
        yield state
    finally:
        unchanged = modeling_llama.apply_rotary_pos_emb is canonical
        modeling_llama.apply_rotary_pos_emb = original
        if not unchanged:
            raise RuntimeError("canonical HF RoPE binding changed during capture")
        if expected_layers is not None and state["calls"] != expected_layers:
            raise RuntimeError(
                f"canonical RoPE ran {state['calls']} times, expected {expected_layers}"
            )


def _primary_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise RuntimeError(f"hook did not receive a primary tensor: {type(value).__name__}")


@dataclass(frozen=True)
class CaptureFile:
    filename: str
    bytes: int
    sha256: str
    tensors: Mapping[str, Mapping[str, Any]]


class LayerCaptureWriter:
    """Stream sampled tensors to one forward and one backward file per layer."""

    def __init__(
        self,
        output_dir: Path,
        *,
        batch_size: int,
        seq_len: int,
        sample_rows: int,
        sample_seed: int,
        num_layers: int,
        bindings: Mapping[str, Any],
    ) -> None:
        self.output_dir = output_dir
        if self.output_dir.exists():
            if any(self.output_dir.iterdir()):
                raise RuntimeError(
                    f"capture staging directory is not empty: {self.output_dir}"
                )
        else:
            self.output_dir.mkdir(parents=True)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.row_indices = deterministic_row_indices(
            batch_size * seq_len, sample_rows, sample_seed
        )
        self.num_layers = num_layers
        self.bindings = dict(bindings)
        self._forward: dict[int, dict[str, torch.Tensor]] = {}
        self._backward: dict[int, dict[str, torch.Tensor]] = {}
        self._source_shapes: dict[tuple[int, str, str], list[int]] = {}
        self._files: dict[str, CaptureFile] = {}

    def _sample(self, tensor: torch.Tensor, layout: str) -> torch.Tensor:
        if layout == "bsh":
            if tensor.ndim != 3 or tensor.shape[:2] != (
                self.batch_size,
                self.seq_len,
            ):
                raise RuntimeError(
                    f"expected BSH tensor, found {tuple(tensor.shape)}"
                )
            rows = tensor.reshape(self.batch_size * self.seq_len, -1)
        elif layout == "bhsd":
            if (
                tensor.ndim != 4
                or tensor.shape[0] != self.batch_size
                or tensor.shape[2] != self.seq_len
            ):
                raise RuntimeError(
                    f"expected BHSD tensor, found {tuple(tensor.shape)}"
                )
            rows = tensor.transpose(1, 2).reshape(
                self.batch_size * self.seq_len, -1
            )
        else:
            raise ValueError(f"unsupported tensor layout: {layout}")
        index = torch.tensor(self.row_indices, dtype=torch.long, device=rows.device)
        return rows.index_select(0, index).detach().to("cpu").contiguous()

    def _capture(
        self,
        phase: str,
        layer: int,
        name: str,
        tensor: torch.Tensor,
        layout: str,
    ) -> None:
        if layer < 0 or layer >= self.num_layers:
            raise RuntimeError(f"capture layer is out of range: {layer}")
        target = self._forward if phase == "forward" else self._backward
        records = target.setdefault(layer, {})
        if name in records:
            raise RuntimeError(f"duplicate {phase} capture L{layer}:{name}")
        records[name] = self._sample(tensor, layout)
        self._source_shapes[(layer, phase, name)] = list(tensor.shape)

    def capture_forward(
        self, layer: int, name: str, tensor: torch.Tensor, layout: str = "bsh"
    ) -> None:
        self._capture("forward", layer, name, tensor, layout)

    def register_backward(
        self,
        layer: int,
        name: str,
        tensor: torch.Tensor,
        layout: str = "bsh",
        *,
        flush_layer: bool = False,
    ) -> None:
        if not tensor.requires_grad:
            raise RuntimeError(f"capture tensor does not require grad: L{layer}:{name}")

        def capture(gradient: torch.Tensor) -> torch.Tensor:
            self._capture("backward", layer, name, gradient, layout)
            if flush_layer:
                self.flush_backward(layer)
            return gradient

        tensor.register_hook(capture)

    def _flush(self, phase: str, layer: int) -> None:
        target = self._forward if phase == "forward" else self._backward
        records = target.pop(layer, None)
        if not records:
            raise RuntimeError(f"no {phase} captures to flush for layer {layer}")
        filename = f"layer_{layer:02d}_{phase}.pt"
        path = self.output_dir / filename
        if path.exists():
            raise RuntimeError(f"refusing to overwrite capture file: {path}")
        tensor_meta = {
            name: {
                "dtype": str(tensor.dtype),
                "sample_shape": list(tensor.shape),
                "source_shape": self._source_shapes.pop((layer, phase, name)),
                "sha256": tensor_sha256(tensor),
                "magnitude": tensor_magnitude_summary(tensor),
            }
            for name, tensor in sorted(records.items())
        }
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "method": CAPTURE_METHOD,
                "layer": layer,
                "phase": phase,
                "row_indices": self.row_indices,
                "tensors": records,
                "tensor_metadata": tensor_meta,
            },
            path,
        )
        file_record = CaptureFile(
            filename=filename,
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
            tensors=tensor_meta,
        )
        self._files[filename] = file_record

    def flush_forward(self, layer: int) -> None:
        self._flush("forward", layer)

    def flush_backward(self, layer: int) -> None:
        self._flush("backward", layer)

    def finalize(self, *, loss: float, extra: Mapping[str, Any]) -> dict[str, Any]:
        if self._forward or self._backward or self._source_shapes:
            raise RuntimeError("capture finalized with unflushed tensors")
        expected = {
            f"layer_{layer:02d}_{phase}.pt"
            for layer in range(self.num_layers)
            for phase in ("forward", "backward")
        }
        if set(self._files) != expected:
            missing = sorted(expected - set(self._files))
            extra_files = sorted(set(self._files) - expected)
            raise RuntimeError(
                f"capture file inventory mismatch missing={missing} extra={extra_files}"
            )
        files = {
            name: {
                "bytes": record.bytes,
                "sha256": record.sha256,
                "tensors": record.tensors,
            }
            for name, record in sorted(self._files.items())
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": CAPTURE_METHOD,
            "bindings": self.bindings,
            "geometry": {
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "num_layers": self.num_layers,
                "sample_rows": len(self.row_indices),
                "sample_row_indices": self.row_indices,
            },
            "loss": loss,
            "files": files,
            "file_ledger_sha256": sha256(canonical_json_bytes(files)).hexdigest(),
            "extra": dict(extra),
        }
        receipt = seal_receipt(payload)
        (self.output_dir / "capture_manifest.json").write_bytes(
            canonical_json_bytes(receipt) + b"\n"
        )
        return receipt


def install_hf_layer_capture_hooks(
    model: Any, writer: LayerCaptureWriter
) -> list[Any]:
    """Install boundary hooks for HF Llama QKV, WO, and SwiGLU FFN replay."""

    layers = model.model.layers
    if len(layers) != writer.num_layers:
        raise RuntimeError(
            f"HF layer count {len(layers)} != capture count {writer.num_layers}"
        )
    handles: list[Any] = []

    def forward_tensor_hook(layer: int, name: str, layout: str = "bsh"):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = _primary_tensor(output)
            writer.capture_forward(layer, name, tensor, layout)
            writer.register_backward(layer, f"{name}_grad", tensor, layout)

        return hook

    def pre_tensor_hook(layer: int, name: str, layout: str = "bsh"):
        def hook(_module: Any, inputs: Any) -> None:
            tensor = _primary_tensor(inputs)
            writer.capture_forward(layer, name, tensor, layout)
            writer.register_backward(layer, f"{name}_grad", tensor, layout)

        return hook

    for layer_index, layer in enumerate(layers):
        def block_pre(_module: Any, inputs: Any, index: int = layer_index) -> None:
            tensor = _primary_tensor(inputs)
            writer.capture_forward(index, "block.pre_norm_input", tensor)
            writer.register_backward(
                index,
                "block.pre_norm_input_grad_ref",
                tensor,
                flush_layer=True,
            )

        def block_post(
            _module: Any, _inputs: Any, output: Any, index: int = layer_index
        ) -> None:
            tensor = _primary_tensor(output)
            writer.capture_forward(index, "block.output_ref", tensor)
            writer.register_backward(index, "block.output_grad_ref", tensor)
            writer.flush_forward(index)

        handles.append(layer.register_forward_pre_hook(block_pre))
        handles.append(layer.register_forward_hook(block_post))
        handles.append(
            layer.input_layernorm.register_forward_hook(
                forward_tensor_hook(layer_index, "qkv.normed_input_ref")
            )
        )
        for projection_name, reference_name in (
            ("q_proj", "qkv.q_linear_ref"),
            ("k_proj", "qkv.k_linear_ref"),
            ("v_proj", "qkv.v_linear_ref"),
        ):
            handles.append(
                getattr(layer.self_attn, projection_name).register_forward_hook(
                    forward_tensor_hook(layer_index, reference_name)
                )
            )
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                pre_tensor_hook(layer_index, "wo.input_ref")
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_hook(
                forward_tensor_hook(layer_index, "wo.output_ref")
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                pre_tensor_hook(layer_index, "ffn.pre_norm_input")
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                forward_tensor_hook(layer_index, "ffn.normed_input_ref")
            )
        )
        handles.append(
            layer.mlp.gate_proj.register_forward_hook(
                forward_tensor_hook(layer_index, "ffn.gate_linear_ref")
            )
        )
        handles.append(
            layer.mlp.up_proj.register_forward_hook(
                forward_tensor_hook(layer_index, "ffn.up_linear_ref")
            )
        )
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                pre_tensor_hook(layer_index, "ffn.w2_input_ref")
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                forward_tensor_hook(layer_index, "ffn.output_ref")
            )
        )
    return handles


def chunked_causal_ce_hidden_gradient(
    hidden_states: torch.Tensor,
    lm_head: Any,
    targets: torch.Tensor,
    *,
    chunk_tokens: int,
) -> tuple[float, torch.Tensor]:
    """Compute regular CE and dHidden without retaining full-vocabulary logits."""

    if hidden_states.ndim != 3 or targets.shape != hidden_states.shape[:2]:
        raise RuntimeError("hidden/target geometry mismatch")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = targets.reshape(-1)
    gradient = torch.empty_like(flat_hidden)
    total = flat_targets.numel()
    loss_value = 0.0
    for start in range(0, total, chunk_tokens):
        stop = min(total, start + chunk_tokens)
        chunk = flat_hidden[start:stop].detach().requires_grad_(True)
        logits = lm_head(chunk).float()
        loss = torch.nn.functional.cross_entropy(
            logits,
            flat_targets[start:stop],
            reduction="sum",
        ) / total
        chunk_gradient = torch.autograd.grad(loss, chunk, only_inputs=True)[0]
        gradient[start:stop].copy_(chunk_gradient.detach())
        loss_value += float(loss.detach())
        del logits, loss, chunk, chunk_gradient
    return loss_value, gradient.view_as(hidden_states)


@contextmanager
def staged_output_directory(destination: Path) -> Iterator[Path]:
    """Build a new output directory atomically and never overwrite evidence."""

    destination = destination.resolve()
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        yield temporary
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
