#!/usr/bin/env python3
"""Materialize a deterministic, globally mixed Llama validation stream.

This stream is deliberately named ``fixed independent`` rather than held out:
the exact Arrow-to-Mosaic training-sample mapping is unavailable, so absence
from every training trajectory cannot yet be proved.  It draws separate
contiguous windows from the shuffled DCLM and OLMo-without-DCLM Mosaic streams,
retokenizes every document with the exact Llama-3.1 tokenizer, enforces an
82/18 stored-token split, hash-shuffles documents globally, and only then packs
the fixed 768 x 8193 token panel.

The output is immutable and self-auditing.  It contains only local relative
paths and content hashes; storage locations and access credentials are never
written into the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import torch
from safetensors.torch import load_file, save_file
from streaming import Stream, StreamingDataset
from tokenizers import Tokenizer


SCHEMA = "mfu_llama_fixed_independent_validation_manifest_v1"
STREAM_ID = "llama31-fixed-independent-82dclm18olmo-r15-20260902"
SEED = 42
WORLD_SIZE = 32
SEQUENCES_PER_RANK = 24
SEQUENCE_LENGTH = 8192
STORED_WIDTH = SEQUENCE_LENGTH + 1
SEQUENCES = WORLD_SIZE * SEQUENCES_PER_RANK
STORED_TOKENS = SEQUENCES * STORED_WIDTH
VALIDATION_TOKENS = SEQUENCES * SEQUENCE_LENGTH
DCLM_TOKENS = round(STORED_TOKENS * 0.82)
OLMO_TOKENS = STORED_TOKENS - DCLM_TOKENS
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
EOS_TOKEN_ID = 128001

TOKENIZER_HASHES = {
    "special_tokens_map.json": (
        "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc"
    ),
    "tokenizer.json": (
        "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa"
    ),
    "tokenizer_config.json": (
        "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a"
    ),
}

SOURCE_SPECS = {
    "dclm": {
        "samples": 590_178_671,
        "token_quota": DCLM_TOKENS,
    },
    "olmo-no-dclm": {
        "samples": 133_343_623,
        "token_quota": OLMO_TOKENS,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_document(document: dict[str, Any], field: str) -> dict[str, Any]:
    if field in document:
        raise ValueError(f"{field} already exists")
    result = dict(document)
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def tokenizer_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in TOKENIZER_HASHES}


def deterministic_start(source: str, samples: int, seed: int) -> int:
    """Choose a reproducible start in the 90--95% physical sample band."""

    lower = samples * 90 // 100
    width = max(1, samples * 5 // 100)
    payload = f"{STREAM_ID}\0{seed}\0{source}\0start".encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % width
    return lower + offset


def shuffle_key(source: str, sample_id: int, text_sha256: str, seed: int) -> str:
    payload = (
        f"{STREAM_ID}\0{seed}\0{source}\0{sample_id}\0{text_sha256}"
    ).encode()
    return sha256_bytes(payload)


@dataclass
class SelectedDocument:
    source: str
    sample_id: int
    hf_global_id: int
    text_sha256: str
    meta_sha256: str
    text_bytes: int
    full_token_count: int
    selected_tokens: list[int]
    source_ordinal: int
    key: str

    def ledger_record(self, shuffled_ordinal: int) -> dict[str, Any]:
        return {
            "shuffled_ordinal": shuffled_ordinal,
            "shuffle_key": self.key,
            "source": self.source,
            "source_ordinal": self.source_ordinal,
            "mosaic_sample_id": self.sample_id,
            "hf_global_id": self.hf_global_id,
            "text_sha256": self.text_sha256,
            "meta_sha256": self.meta_sha256,
            "text_bytes": self.text_bytes,
            "full_token_count": self.full_token_count,
            "selected_token_count": len(self.selected_tokens),
            "terminal_prefix_truncated": len(self.selected_tokens) != self.full_token_count,
            "selected_tokens_sha256": sha256_bytes(
                torch.tensor(self.selected_tokens, dtype=torch.int32).numpy().tobytes()
            ),
        }


def build_tokenizer(root: Path) -> Tokenizer:
    observed = tokenizer_hashes(root)
    if observed != TOKENIZER_HASHES:
        raise RuntimeError(f"tokenizer hash drift: {observed}")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if (
        tokenizer.get_vocab_size() != VOCAB_SIZE
        or tokenizer.token_to_id("<|begin_of_text|>") != BOS_TOKEN_ID
        or tokenizer.token_to_id("<|end_of_text|>") != EOS_TOKEN_ID
    ):
        raise RuntimeError("Llama tokenizer identity drift")
    empty = tokenizer.encode("").ids
    if empty != [BOS_TOKEN_ID]:
        raise RuntimeError(f"unexpected tokenizer BOS behavior: {empty}")
    return tokenizer


def encode_document(tokenizer: Tokenizer, text: str) -> list[int]:
    tokens = tokenizer.encode(text).ids
    if not tokens or tokens[0] != BOS_TOKEN_ID:
        raise RuntimeError("document does not begin with the configured BOS token")
    if tokens[-1] != EOS_TOKEN_ID:
        tokens.append(EOS_TOKEN_ID)
    return tokens


def select_source_documents(
    *,
    source: str,
    spec: dict[str, Any],
    tokenizer: Tokenizer,
    cache_dir: Path,
    seed: int,
    seen_text_hashes: set[str],
) -> tuple[list[SelectedDocument], dict[str, Any]]:
    os.environ.update(
        {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"}
    )
    dataset = StreamingDataset(
        streams=[
            Stream(
                remote=str(spec["source"]),
                local=str(cache_dir),
                split="train",
            )
        ],
        batch_size=1,
        shuffle=False,
    )
    if len(dataset) != int(spec["samples"]):
        raise RuntimeError(f"{source} sample-count drift: {len(dataset)}")
    start = deterministic_start(source, len(dataset), seed)
    quota = int(spec["token_quota"])
    selected: list[SelectedDocument] = []
    selected_count = 0
    duplicates_skipped = 0
    empty_skipped = 0
    sample_id = start
    while selected_count < quota:
        if sample_id >= len(dataset):
            raise RuntimeError(f"{source} deterministic window exhausted")
        sample = dataset[sample_id]
        text = sample.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{source} sample {sample_id} text is not str")
        text_payload = text.encode("utf-8")
        text_sha = sha256_bytes(text_payload)
        if text_sha in seen_text_hashes:
            duplicates_skipped += 1
            sample_id += 1
            continue
        tokens = encode_document(tokenizer, text)
        if len(tokens) <= 2:
            empty_skipped += 1
            sample_id += 1
            continue
        remaining = quota - selected_count
        chosen = tokens[:remaining]
        hf_global_id = int(sample.get("hf_global_id", sample_id))
        if hf_global_id != sample_id:
            raise RuntimeError(
                f"{source} physical/global sample ID mismatch: {sample_id}/{hf_global_id}"
            )
        meta = sample.get("meta")
        meta_payload = (
            meta.encode("utf-8")
            if isinstance(meta, str)
            else canonical_json_bytes(meta)
        )
        selected.append(
            SelectedDocument(
                source=source,
                sample_id=sample_id,
                hf_global_id=hf_global_id,
                text_sha256=text_sha,
                meta_sha256=sha256_bytes(meta_payload),
                text_bytes=len(text_payload),
                full_token_count=len(tokens),
                selected_tokens=chosen,
                source_ordinal=len(selected),
                key=shuffle_key(source, sample_id, text_sha, seed),
            )
        )
        seen_text_hashes.add(text_sha)
        selected_count += len(chosen)
        sample_id += 1
        if len(selected) % 500 == 0:
            print(
                f"[SELECT] source={source} documents={len(selected)} "
                f"tokens={selected_count}/{quota}",
                flush=True,
            )
    return selected, {
        "source": source,
        "total_samples": len(dataset),
        "selection_start_sample_id": start,
        "selection_stop_sample_id_exclusive": sample_id,
        "documents": len(selected),
        "duplicates_skipped": duplicates_skipped,
        "empty_documents_skipped": empty_skipped,
        "selected_tokens": selected_count,
        "token_quota": quota,
        "terminal_prefix_truncations": sum(
            len(item.selected_tokens) != item.full_token_count for item in selected
        ),
    }


def build_packed_tensor(
    documents: list[SelectedDocument],
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(documents, key=lambda item: item.key)
    if len({item.key for item in ordered}) != len(ordered):
        raise RuntimeError("document shuffle-key collision")
    total = sum(len(item.selected_tokens) for item in ordered)
    if total != STORED_TOKENS:
        raise RuntimeError(f"stored-token total drift: {total} != {STORED_TOKENS}")
    flat = torch.empty(total, dtype=torch.int32)
    document_ledger: list[dict[str, Any]] = []
    cursor = 0
    for ordinal, document in enumerate(ordered):
        values = torch.tensor(document.selected_tokens, dtype=torch.int32)
        stop = cursor + values.numel()
        flat[cursor:stop].copy_(values)
        record = document.ledger_record(ordinal)
        record.update({"packed_token_start": cursor, "packed_token_stop": stop})
        document_ledger.append(record)
        cursor = stop
    packed = flat.view(SEQUENCES, STORED_WIDTH).contiguous()
    if int(packed.min()) < 0 or int(packed.max()) >= VOCAB_SIZE:
        raise RuntimeError("packed token IDs exceed Llama vocabulary")

    packing_ledger: list[dict[str, Any]] = []
    doc_index = 0
    for row in range(SEQUENCES):
        start = row * STORED_WIDTH
        stop = start + STORED_WIDTH
        while document_ledger[doc_index]["packed_token_stop"] <= start:
            doc_index += 1
        cursor_index = doc_index
        segments = []
        while (
            cursor_index < len(document_ledger)
            and document_ledger[cursor_index]["packed_token_start"] < stop
        ):
            document = document_ledger[cursor_index]
            overlap_start = max(start, int(document["packed_token_start"]))
            overlap_stop = min(stop, int(document["packed_token_stop"]))
            segments.append(
                {
                    "shuffled_document_ordinal": int(document["shuffled_ordinal"]),
                    "source": document["source"],
                    "mosaic_sample_id": int(document["mosaic_sample_id"]),
                    "row_token_start": overlap_start - start,
                    "row_token_stop": overlap_stop - start,
                    "document_token_start": overlap_start
                    - int(document["packed_token_start"]),
                    "document_token_stop": overlap_stop
                    - int(document["packed_token_start"]),
                }
            )
            cursor_index += 1
        packing_ledger.append(
            {
                "row": row,
                "rank": row // SEQUENCES_PER_RANK,
                "rank_row": row % SEQUENCES_PER_RANK,
                "segments": segments,
            }
        )
    return packed, document_ledger, packing_ledger


def jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def write_outputs(
    *,
    output: Path,
    packed: torch.Tensor,
    document_ledger: list[dict[str, Any]],
    packing_ledger: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    for rank in range(WORLD_SIZE):
        path = output / f"rank-{rank:02d}.safetensors"
        start = rank * SEQUENCES_PER_RANK
        stop = start + SEQUENCES_PER_RANK
        save_file({"tokens": packed[start:stop].contiguous()}, path)
        reopened = load_file(path, device="cpu")["tokens"]
        if reopened.dtype != torch.int32 or not torch.equal(
            reopened, packed[start:stop]
        ):
            raise RuntimeError(f"rank {rank} safetensors round-trip failed")
        shards.append(
            {
                "rank": rank,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(reopened.shape),
                "sequences": SEQUENCES_PER_RANK,
                "validation_tokens": SEQUENCES_PER_RANK * SEQUENCE_LENGTH,
                "token_id_min": int(reopened.min()),
                "token_id_max": int(reopened.max()),
            }
        )
    ledgers = {
        "source-ledger.jsonl": jsonl_bytes(document_ledger),
        "packing-ledger.jsonl": jsonl_bytes(packing_ledger),
    }
    for name, payload in ledgers.items():
        (output / name).write_bytes(payload)
    return shards, ledgers


def validate_manifest(manifest: dict[str, Any]) -> None:
    unsigned = dict(manifest)
    observed = unsigned.pop("manifest_sha256", None)
    if observed != sha256_bytes(canonical_json_bytes(unsigned)):
        raise RuntimeError("manifest seal drift")
    if manifest.get("schema") != SCHEMA or manifest.get("stream_id") != STREAM_ID:
        raise RuntimeError("manifest identity drift")
    if manifest.get("claim") != "fixed-independent-not-proven-held-out":
        raise RuntimeError("validation-set claim drift")
    geometry = manifest.get("geometry")
    expected_geometry = {
        "world_size": WORLD_SIZE,
        "sequences_per_rank": SEQUENCES_PER_RANK,
        "sequences": SEQUENCES,
        "stored_tokens_per_sequence": STORED_WIDTH,
        "scored_tokens_per_sequence": SEQUENCE_LENGTH,
        "stored_tokens": STORED_TOKENS,
        "validation_tokens": VALIDATION_TOKENS,
        "padding": False,
    }
    if geometry != expected_geometry:
        raise RuntimeError("manifest geometry drift")
    if manifest.get("stratification", {}).get("selected_tokens") != {
        "dclm": DCLM_TOKENS,
        "olmo-no-dclm": OLMO_TOKENS,
    }:
        raise RuntimeError("manifest stratum token totals drift")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or [item.get("rank") for item in shards] != list(
        range(WORLD_SIZE)
    ):
        raise RuntimeError("manifest shard inventory drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--dclm-stream", type=Path, required=True)
    parser.add_argument("--olmo-stream", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error(f"the sealed r15 contract requires --seed={SEED}")
    if args.output.exists():
        raise FileExistsError(args.output)

    tokenizer = build_tokenizer(args.tokenizer_path)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    source_specs = {
        "dclm": {**SOURCE_SPECS["dclm"], "source": args.dclm_stream},
        "olmo-no-dclm": {
            **SOURCE_SPECS["olmo-no-dclm"],
            "source": args.olmo_stream,
        },
    }
    dataset_indices = {
        source: {
            "name": "train/index.json",
            "sha256": sha256_file(Path(spec["source"]) / "train" / "index.json"),
        }
        for source, spec in source_specs.items()
    }
    selected_documents: list[SelectedDocument] = []
    source_summaries = []
    seen_text_hashes: set[str] = set()
    for source, spec in source_specs.items():
        documents, summary = select_source_documents(
            source=source,
            spec=spec,
            tokenizer=tokenizer,
            cache_dir=args.cache_dir / source,
            seed=args.seed,
            seen_text_hashes=seen_text_hashes,
        )
        selected_documents.extend(documents)
        source_summaries.append(summary)

    packed, document_ledger, packing_ledger = build_packed_tensor(selected_documents)
    shards, ledgers = write_outputs(
        output=args.output,
        packed=packed,
        document_ledger=document_ledger,
        packing_ledger=packing_ledger,
    )
    source_counts = {
        source: sum(
            int(item["selected_token_count"])
            for item in document_ledger
            if item["source"] == source
        )
        for source in SOURCE_SPECS
    }
    if source_counts != {"dclm": DCLM_TOKENS, "olmo-no-dclm": OLMO_TOKENS}:
        raise RuntimeError(f"post-pack stratum drift: {source_counts}")

    script_path = Path(__file__).resolve()
    manifest = seal_document(
        {
            "schema": SCHEMA,
            "stream_id": STREAM_ID,
            "claim": "fixed-independent-not-proven-held-out",
            "claim_note": (
                "Exact exclusion from every Llama training example is not claimed because "
                "the Arrow-to-Mosaic sample mapping is unavailable."
            ),
            "dataset_description": (
                "Deterministic 82/18 DCLM/OLMo-no-DCLM sample, Llama-3.1 "
                "retokenized, globally hash-shuffled by document before packing"
            ),
            "selection": {
                "seed": args.seed,
                "physical_sample_band": "90%-95%",
                "within_source_order": "ascending contiguous physical sample ID",
                "cross_source_order": "SHA256(stream_id,seed,source,sample_id,text_sha256)",
                "duplicate_policy": "skip repeated raw-text SHA256 globally",
                "source_summaries": source_summaries,
            },
            "stratification": {
                "unit": "stored tokens before sequence packing",
                "target_fraction": {"dclm": 0.82, "olmo-no-dclm": 0.18},
                "selected_tokens": source_counts,
                "terminal_prefix_truncation": "at most one document per stratum",
            },
            "dataset_indices": dataset_indices,
            "tokenizer": {
                "identity": "Meta-Llama-3.1-8B-exact-assets",
                "files_sha256": tokenizer_hashes(args.tokenizer_path),
                "vocab_size": VOCAB_SIZE,
                "bos_token_id": BOS_TOKEN_ID,
                "eos_token_id": EOS_TOKEN_ID,
                "document_policy": "tokenizer-native BOS plus explicit EOS",
            },
            "geometry": {
                "world_size": WORLD_SIZE,
                "sequences_per_rank": SEQUENCES_PER_RANK,
                "sequences": SEQUENCES,
                "stored_tokens_per_sequence": STORED_WIDTH,
                "scored_tokens_per_sequence": SEQUENCE_LENGTH,
                "stored_tokens": STORED_TOKENS,
                "validation_tokens": VALIDATION_TOKENS,
                "padding": False,
            },
            "materializer": {
                "sha256": sha256_file(script_path),
                "streaming_version": getattr(__import__("streaming"), "__version__", None),
                "tokenizers_version": getattr(__import__("tokenizers"), "__version__", None),
                "torch_version": torch.__version__,
            },
            "provenance": {
                "source_documents": len(document_ledger),
                "unique_raw_text_sha256": len(
                    {item["text_sha256"] for item in document_ledger}
                ),
                "source_ledger": {
                    "path": "source-ledger.jsonl",
                    "bytes": len(ledgers["source-ledger.jsonl"]),
                    "sha256": sha256_bytes(ledgers["source-ledger.jsonl"]),
                },
                "packing_ledger": {
                    "path": "packing-ledger.jsonl",
                    "bytes": len(ledgers["packing-ledger.jsonl"]),
                    "sha256": sha256_bytes(ledgers["packing-ledger.jsonl"]),
                },
            },
            "shards": shards,
        },
        "manifest_sha256",
    )
    validate_manifest(manifest)
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    (args.output / "manifest.json").write_bytes(manifest_payload)
    print(
        "MFU_LLAMA_FIXED_INDEPENDENT_VALIDATION_PASS "
        f"stream_id={STREAM_ID} manifest_file_sha256={sha256_bytes(manifest_payload)} "
        f"manifest_seal={manifest['manifest_sha256']} documents={len(document_ledger)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
