#!/usr/bin/env python
#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
"""Build token-packed binary datasets for local training screens."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torchtitan_submodule"))
sys.path.insert(0, str(ROOT))

from torchtitan.components.tokenizer import HuggingFaceTokenizer  # noqa: E402


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_tokenizer(tokenizer_path: str):
    if tokenizer_path.endswith(".model"):
        spec = importlib.util.spec_from_file_location(
            "lbt_tiktoken_tokenizer",
            ROOT / "low_bits_training" / "components" / "tiktoken_tokenizer.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load low_bits_training tiktoken tokenizer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TikTokenizer = module.TikTokenizer
        return TikTokenizer(tokenizer_path)
    return HuggingFaceTokenizer(tokenizer_path)


def _sample_text(sample: dict) -> str:
    text = sample.get("text", "")
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return text


def _encode_texts(
    tokenizer,
    texts: list[str],
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[list[int]]:
    hf_tokenizer = getattr(tokenizer, "tokenizer", None)
    encode_batch = getattr(hf_tokenizer, "encode_batch", None)
    if callable(encode_batch):
        encoded = [enc.ids for enc in encode_batch(texts)]
        if add_bos and not getattr(tokenizer, "hf_adds_bos", False):
            bos_id = getattr(tokenizer, "bos_id", None)
            if bos_id is not None:
                encoded = [[bos_id, *ids] for ids in encoded]
        if add_eos and not getattr(tokenizer, "hf_adds_eos", False):
            eos_id = getattr(tokenizer, "eos_id", None)
            if eos_id is not None:
                encoded = [[*ids, eos_id] for ids in encoded]
        return encoded
    return [
        tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        for text in texts
    ]


def _has_aws_credentials() -> bool:
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
    )


def _force_unsigned_s3_downloads() -> None:
    import streaming.base.storage.download as download

    if getattr(download.S3Downloader._create_s3_client, "_lbt_unsigned_s3", False):
        return
    original = download.S3Downloader._create_s3_client

    def _create_unsigned(self, unsigned: bool = False, timeout=download.DEFAULT_TIMEOUT):
        return original(self, unsigned=True, timeout=timeout)

    _create_unsigned._lbt_unsigned_s3 = True
    download.S3Downloader._create_s3_client = _create_unsigned


def _mosaic_uses_unsigned_s3(args: argparse.Namespace) -> bool:
    return bool(
        args.unsigned_s3
        or (
            args.unsigned_s3_auto
            and args.dataset_path.startswith("s3://")
            and not _has_aws_credentials()
        )
    )


def _prepare_mosaic_downloads(args: argparse.Namespace) -> None:
    if _mosaic_uses_unsigned_s3(args):
        _force_unsigned_s3_downloads()


def _mosaic_stream(args: argparse.Namespace):
    from streaming import Stream

    if not args.dataset_path:
        raise ValueError("--dataset-path is required for --source mosaic")
    _prepare_mosaic_downloads(args)
    stream = Stream(remote=args.dataset_path, local=args.cache_dir, split=args.split)
    # streaming==0.11 exposes these as underscored fields, but its private
    # _download_file helper still reads the public names.
    if not hasattr(stream, "download_retry"):
        stream.download_retry = getattr(stream, "_download_retry", None) or 2
    if not hasattr(stream, "download_timeout"):
        stream.download_timeout = getattr(stream, "_download_timeout", None) or 60
    return stream


def _iter_mosaic(args: argparse.Namespace) -> Iterable[dict]:
    from streaming import StreamingDataset

    stream = _mosaic_stream(args)
    kwargs = {}
    if args.cache_limit:
        kwargs["cache_limit"] = args.cache_limit
    dataset = StreamingDataset(
        streams=[stream],
        batch_size=1,
        shuffle=args.shuffle,
        predownload=args.predownload,
        **kwargs,
    )
    return iter(dataset)


def _mosaic_local_split_dir(args: argparse.Namespace) -> Path:
    return Path(args.cache_dir).expanduser() / args.split


def _ensure_mosaic_file(
    stream,
    local_split_dir: Path,
    basename: str,
    expected_bytes: int | None = None,
) -> Path:
    path = local_split_dir / basename
    if path.exists():
        if expected_bytes is None or path.stat().st_size == expected_bytes:
            return path
        path.unlink()
    local_split_dir.mkdir(parents=True, exist_ok=True)
    stream._download_file(basename)
    if not path.exists():
        raise FileNotFoundError(f"Mosaic download did not create {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        path.unlink()
        raise IOError(f"Mosaic download for {path} has the wrong size")
    return path


def _load_mosaic_index(args: argparse.Namespace, stream) -> dict:
    local_split_dir = _mosaic_local_split_dir(args)
    index_path = local_split_dir / "index.json"
    if not index_path.exists():
        local_split_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(stream._download_file("index.json", "index.json.tmp"))
        os.replace(tmp_path, index_path)
    with index_path.open() as handle:
        obj = json.load(handle)
    if obj.get("version") != 2:
        raise ValueError(f"Unsupported Mosaic dataset version: {obj.get('version')}")
    return obj


def _is_fast_mds_text_shard(info: dict) -> bool:
    return (
        info.get("format") == "mds"
        and info.get("compression") is None
        and info.get("zip_data") is None
        and info.get("column_names") == ["text"]
        and info.get("column_encodings") == ["str"]
        and info.get("column_sizes") == [None]
        and isinstance(info.get("raw_data"), dict)
        and bool(info["raw_data"].get("basename"))
    )


def _iter_mds_text_shard(path: Path, expected_samples: int) -> Iterable[dict]:
    data = path.read_bytes()
    if len(data) < 4:
        raise ValueError(f"{path} is too small to be a valid MDS shard")
    samples = struct.unpack_from("<I", data, 0)[0]
    if samples != expected_samples:
        raise ValueError(
            f"{path} records {samples} samples, expected {expected_samples}"
        )
    offset_count = samples + 1
    offset_bytes = 4 * offset_count
    if len(data) < 4 + offset_bytes:
        raise ValueError(f"{path} has a truncated MDS offset table")
    offsets = np.frombuffer(data, dtype="<u4", count=offset_count, offset=4)
    for begin_u32, end_u32 in zip(offsets[:-1], offsets[1:]):
        begin = int(begin_u32)
        end = int(end_u32)
        if begin < 0 or end < begin or end > len(data) or end - begin < 4:
            raise ValueError(f"{path} contains an invalid MDS sample offset")
        text_size = struct.unpack_from("<I", data, begin)[0]
        text_begin = begin + 4
        text_end = text_begin + text_size
        if text_end > end:
            raise ValueError(f"{path} contains a truncated text column")
        yield {"text": data[text_begin:text_end].decode("utf-8", errors="ignore")}


def _iter_mosaic_fast_local(args: argparse.Namespace) -> Iterable[dict]:
    if args.shuffle:
        print("Mosaic fast-local path disabled because --shuffle was requested.", flush=True)
        return _iter_mosaic(args)

    stream = _mosaic_stream(args)
    obj = _load_mosaic_index(args, stream)
    shards = obj["shards"]
    if not all(_is_fast_mds_text_shard(info) for info in shards):
        print(
            "Mosaic fast-local path disabled because at least one shard has "
            "an unsupported format.",
            flush=True,
        )
        return _iter_mosaic(args)

    local_split_dir = _mosaic_local_split_dir(args)

    def _iterator() -> Iterable[dict]:
        for info in shards:
            basename = info["raw_data"]["basename"]
            expected_bytes = int(info["raw_data"]["bytes"])
            shard_path = _ensure_mosaic_file(
                stream, local_split_dir, basename, expected_bytes
            )
            yield from _iter_mds_text_shard(shard_path, int(info["samples"]))

    return _iterator()


def _iter_hf(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    path = args.hf_dataset_path or args.dataset
    kwargs = {
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.hf_name:
        kwargs["name"] = args.hf_name
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    return iter(load_dataset(path, **kwargs))


def _out_paths(output_prefix: str) -> tuple[Path, Path]:
    prefix = Path(output_prefix).expanduser()
    if prefix.suffix == ".bin":
        return prefix, prefix.with_suffix(".json")
    return prefix.with_suffix(".bin"), prefix.with_suffix(".json")


def _manifest_path(output_dir: str, manifest_name: str) -> Path:
    return Path(output_dir).expanduser() / manifest_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["mosaic", "hf"], default="mosaic")
    parser.add_argument(
        "--dataset",
        default="mosaic/cerebras___slim_pajama-627_b",
        help="Dataset name recorded in metadata, or HF dataset path for --source hf.",
    )
    parser.add_argument(
        "--dataset-path",
        default="OBJECT_STORE_URI",
        help="Mosaic remote/local dataset path.",
    )
    parser.add_argument("--hf-dataset-path", default=None)
    parser.add_argument("--hf-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--cache-dir",
        default="/tmp/lbt_mosaic_cache/cerebras_slim_pajama_627b_small_shards",
    )
    parser.add_argument("--cache-limit", default=None)
    parser.add_argument("--predownload", type=int, default=16)
    parser.add_argument(
        "--unsigned-s3",
        action="store_true",
        help="Force unsigned S3 downloads for public Mosaic datasets.",
    )
    parser.add_argument(
        "--unsigned-s3-auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use unsigned S3 automatically for s3:// Mosaic paths when AWS credentials are absent.",
    )
    parser.add_argument(
        "--mosaic-fast-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For compatible Mosaic MDS text shards, read local shard files "
            "sequentially instead of using per-sample StreamingDataset access."
        ),
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tokenizer-path",
        default="./torchtitan_submodule/tests/assets/tokenizer",
    )
    parser.add_argument(
        "--output-prefix",
        default="/tmp/lbt_packed/slimpajama_64m_tokens",
        help="Single-file output prefix. Ignored when --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write a sharded packed dataset directory with a manifest.json.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename for --output-dir.",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=1_073_741_824,
        help="Tokens per .bin shard when --output-dir is set. Default is 4 GiB of uint32 tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=67_108_864,
        help="Stop after this many tokens. Default is enough for about 1K 8x8192 steps.",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=512,
        help="Number of documents to batch into each tokenizer call.",
    )
    parser.add_argument("--dtype", choices=["uint32"], default="uint32")
    parser.add_argument("--env-file", default="/tmp/lbt_bench_env")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_file(Path(args.env_file))

    sharded = args.output_dir is not None
    if sharded:
        out_dir = Path(args.output_dir).expanduser()
        manifest_path = _manifest_path(args.output_dir, args.manifest_name)
        if manifest_path.exists() and not args.overwrite:
            raise FileExistsError(f"{manifest_path} exists; pass --overwrite to replace it")
        if args.tokens_per_shard <= 0:
            raise ValueError("--tokens-per-shard must be >0 when --output-dir is set")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        bin_path, meta_path = _out_paths(args.output_prefix)
        if bin_path.exists() and not args.overwrite:
            raise FileExistsError(f"{bin_path} exists; pass --overwrite to replace it")
        bin_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = _build_tokenizer(args.tokenizer_path)
    if args.source == "mosaic":
        samples = (
            _iter_mosaic_fast_local(args)
            if args.mosaic_fast_local
            else _iter_mosaic(args)
        )
    else:
        samples = _iter_hf(args)

    token_count = 0
    doc_count = 0
    shards: list[dict[str, object]] = []
    shard_handle = None
    shard_token_count = 0
    shard_doc_count = 0
    shard_idx = 0
    started = time.perf_counter()
    next_log = 10_000
    text_batch: list[str] = []

    def _close_current_shard():
        nonlocal shard_handle, shard_token_count, shard_doc_count, shard_idx
        if shard_handle is not None:
            shard_handle.flush()
            os.fsync(shard_handle.fileno())
            shard_handle.close()
            if shard_token_count > 0:
                shards.append(
                    {
                        "path": f"tokens_{shard_idx:06d}.bin",
                        "token_count": shard_token_count,
                        "doc_count": shard_doc_count,
                    }
                )
                shard_idx += 1
            shard_handle = None
            shard_token_count = 0
            shard_doc_count = 0

    def _open_next_shard():
        nonlocal shard_handle, shard_token_count, shard_doc_count
        _close_current_shard()
        shard_path = out_dir / f"tokens_{shard_idx:06d}.bin"
        if shard_path.exists() and not args.overwrite:
            raise FileExistsError(f"{shard_path} exists; pass --overwrite to replace it")
        shard_handle = shard_path.open("wb")
        shard_token_count = 0
        shard_doc_count = 0

    def _write_ids(ids: list[int], count_doc: bool) -> None:
        nonlocal token_count, shard_token_count, shard_doc_count
        start = 0
        while start < len(ids) and token_count < args.max_tokens:
            if sharded and (shard_handle is None or shard_token_count >= args.tokens_per_shard):
                _open_next_shard()
            remaining_total = args.max_tokens - token_count
            remaining_shard = (
                args.tokens_per_shard - shard_token_count
                if sharded
                else len(ids) - start
            )
            take = min(len(ids) - start, remaining_total, remaining_shard)
            if take <= 0:
                break
            target = shard_handle if sharded else handle
            np.asarray(ids[start : start + take], dtype=np.uint32).tofile(target)
            if count_doc:
                shard_doc_count += 1
                count_doc = False
            token_count += take
            shard_token_count += take
            start += take

    def _flush_text_batch() -> None:
        nonlocal doc_count, next_log, text_batch
        if not text_batch:
            return
        encoded_docs = _encode_texts(tokenizer, text_batch, add_bos=True, add_eos=True)
        text_batch = []
        for ids in encoded_docs:
            if not ids or token_count >= args.max_tokens:
                break
            remaining = args.max_tokens - token_count
            if len(ids) > remaining:
                ids = ids[:remaining]
            _write_ids(ids, count_doc=True)
            doc_count += 1
            if doc_count >= next_log:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"docs={doc_count:,} tokens={token_count:,} "
                    f"tok_s={token_count / elapsed:,.0f}",
                    flush=True,
                )
                next_log += 10_000
            if args.max_docs is not None and doc_count >= args.max_docs:
                break

    handle = None
    try:
        if sharded:
            _open_next_shard()
        else:
            handle = bin_path.open("wb")
        for sample in samples:
            text = _sample_text(sample)
            if not text:
                continue
            text_batch.append(text)
            if len(text_batch) < args.encode_batch_size:
                continue
            _flush_text_batch()
            if token_count >= args.max_tokens:
                break
            if args.max_docs is not None and doc_count >= args.max_docs:
                break
        _flush_text_batch()
    finally:
        if sharded:
            _close_current_shard()
        elif handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    elapsed = time.perf_counter() - started
    metadata = {
        "format": "lbt_packed_tokens_manifest_v1" if sharded else "lbt_packed_tokens_v1",
        "dtype": args.dtype,
        "token_count": token_count,
        "doc_count": doc_count,
        "source": args.source,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path if args.source == "mosaic" else args.hf_dataset_path,
        "split": args.split,
        "cache_dir": args.cache_dir,
        "tokenizer_path": args.tokenizer_path,
        "elapsed_s": elapsed,
    }
    if sharded:
        metadata["tokens_per_shard"] = args.tokens_per_shard
        metadata["shards"] = shards
        manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"wrote {manifest_path}: shards={len(shards):,} docs={doc_count:,} "
            f"tokens={token_count:,} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"wrote {bin_path} and {meta_path}: docs={doc_count:,} "
            f"tokens={token_count:,} elapsed_s={elapsed:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
