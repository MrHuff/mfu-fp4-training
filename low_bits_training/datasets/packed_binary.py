#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
"""Pretokenized token-packed binary dataset support."""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class _PackedShard:
    path: Path
    token_count: int
    sample_count: int
    doc_count: int | None = None


def _str_to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_dataset_path(dataset_path: str | None, load_kwargs: dict[str, Any]) -> Path:
    path_str = load_kwargs.get("path") or load_kwargs.get("bin_path") or dataset_path
    if not path_str:
        raise ValueError(
            "Packed binary dataset requires --training.dataset-path or "
            '"path" in --training.load-dataset-kwargs.'
        )
    return Path(path_str).expanduser()


def _metadata_path(path: Path) -> Path | None:
    candidates = [
        path.with_suffix(".json"),
        path.parent / "metadata.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _single_shard(bin_path: Path, seq_len: int, dtype: np.dtype) -> _PackedShard:
    if not bin_path.exists():
        raise FileNotFoundError(f"Packed binary token file not found: {bin_path}")
    token_count = bin_path.stat().st_size // dtype.itemsize
    sample_count = max(0, (token_count - 1) // seq_len)
    return _PackedShard(
        path=bin_path,
        token_count=token_count,
        sample_count=sample_count,
    )


def _manifest_candidates(path: Path) -> list[Path]:
    if path.is_dir():
        return [
            path / "manifest.json",
            path / "packed_tokens_manifest.json",
            path / "metadata.json",
        ]
    if path.suffix == ".json":
        return [path]
    return []


def _resolve_packed_shards(
    dataset_path: str | None,
    load_kwargs: dict[str, Any],
    seq_len: int,
) -> tuple[Path, list[_PackedShard], str]:
    explicit_manifest = load_kwargs.get("manifest") or load_kwargs.get("manifest_path")
    source_path = (
        Path(explicit_manifest).expanduser()
        if explicit_manifest
        else _resolve_dataset_path(dataset_path, load_kwargs)
    )
    dtype_name = load_kwargs.get("dtype")

    for manifest_path in _manifest_candidates(source_path):
        manifest = _load_json_if_exists(manifest_path)
        if not manifest or manifest.get("format") != "lbt_packed_tokens_manifest_v1":
            continue
        dtype_name = dtype_name or manifest.get("dtype") or "uint32"
        dtype = np.dtype(dtype_name)
        shards = []
        for entry in manifest.get("shards", []):
            shard_path = Path(entry["path"])
            if not shard_path.is_absolute():
                shard_path = manifest_path.parent / shard_path
            token_count = int(
                entry.get("token_count", shard_path.stat().st_size // dtype.itemsize)
            )
            shards.append(
                _PackedShard(
                    path=shard_path,
                    token_count=token_count,
                    sample_count=max(0, (token_count - 1) // seq_len),
                    doc_count=entry.get("doc_count"),
                )
            )
        if not shards:
            raise ValueError(f"Packed binary manifest has no shards: {manifest_path}")
        return manifest_path, shards, dtype_name

    if source_path.is_dir():
        candidate = source_path / "tokens.bin"
        bin_paths = [candidate] if candidate.exists() else sorted(source_path.glob("*.bin"))
        if not bin_paths:
            raise FileNotFoundError(
                f"Packed binary directory has no manifest.json and no .bin shards: {source_path}"
            )
        metadata = _load_json_if_exists(source_path / "metadata.json") or {}
        dtype_name = dtype_name or metadata.get("dtype") or "uint32"
        dtype = np.dtype(dtype_name)
        return source_path, [_single_shard(path, seq_len, dtype) for path in bin_paths], dtype_name

    if source_path.suffix == ".json":
        metadata = _load_json_if_exists(source_path) or {}
        bin_path = source_path.with_suffix(".bin")
        dtype_name = dtype_name or metadata.get("dtype") or "uint32"
        return source_path, [_single_shard(bin_path, seq_len, np.dtype(dtype_name))], dtype_name

    metadata_path = _metadata_path(source_path)
    metadata = _load_json_if_exists(metadata_path) if metadata_path is not None else {}
    metadata = metadata or {}
    dtype_name = dtype_name or metadata.get("dtype") or "uint32"
    return source_path, [_single_shard(source_path, seq_len, np.dtype(dtype_name))], dtype_name


def _same_resolved_path(lhs: Path, rhs: Path) -> bool:
    try:
        return lhs.resolve() == rhs.resolve()
    except OSError:
        return lhs.expanduser().absolute() == rhs.expanduser().absolute()


def _close_memmap(memmap: np.memmap) -> None:
    mmap_obj = getattr(memmap, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


class PackedBinaryDataset(IterableDataset, Stateful):
    """Read packed token shards as fixed-length next-token sequences."""

    def __init__(
        self,
        source_path: Path,
        shards: Sequence[_PackedShard],
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        dtype: str = "uint32",
        infinite: bool = True,
        sample_count: int | None = None,
        max_open_shards: int = 8,
    ) -> None:
        self.source_path = Path(source_path)
        self.shards = list(shards)
        self.seq_len = int(seq_len)
        self.dp_rank = int(dp_rank)
        self.dp_world_size = int(dp_world_size)
        self.dtype = np.dtype(dtype)
        self.infinite = bool(infinite)
        self.max_open_shards = max(1, int(max_open_shards))
        self._cursor = 0
        self._open_tokens: OrderedDict[int, np.memmap] = OrderedDict()

        if not self.shards:
            raise ValueError(f"No packed binary shards found for {self.source_path}")
        sample_count = int(sample_count) if sample_count is not None else None
        total_samples = sum(shard.sample_count for shard in self.shards)
        self.num_samples = min(total_samples, sample_count) if sample_count else total_samples
        self.token_count = sum(shard.token_count for shard in self.shards)
        self._cumulative_samples: list[int] = []
        running = 0
        for shard in self.shards:
            if not shard.path.exists():
                raise FileNotFoundError(f"Packed binary shard not found: {shard.path}")
            running += shard.sample_count
            self._cumulative_samples.append(running)

        if self.num_samples <= 0:
            raise ValueError(
                f"Packed binary dataset {self.source_path} does not contain enough "
                f"tokens for seq_len={self.seq_len}."
            )

    def _tokens_for_shard(self, shard_idx: int) -> np.memmap:
        tokens = self._open_tokens.get(shard_idx)
        if tokens is not None:
            self._open_tokens.move_to_end(shard_idx)
            return tokens
        shard = self.shards[shard_idx]
        tokens = np.memmap(shard.path, mode="r", dtype=self.dtype)
        self._open_tokens[shard_idx] = tokens
        while len(self._open_tokens) > self.max_open_shards:
            _, old_tokens = self._open_tokens.popitem(last=False)
            _close_memmap(old_tokens)
        return tokens

    def _sample_window(self, sample_idx: int) -> np.ndarray:
        shard_idx = bisect_right(self._cumulative_samples, sample_idx)
        previous = 0 if shard_idx == 0 else self._cumulative_samples[shard_idx - 1]
        local_sample_idx = sample_idx - previous
        token_offset = local_sample_idx * self.seq_len
        tokens = self._tokens_for_shard(shard_idx)
        return np.asarray(
            tokens[token_offset : token_offset + self.seq_len + 1],
            dtype=np.int64,
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        stride = self.dp_world_size * num_workers
        first_sample = self.dp_rank * num_workers + worker_id
        if first_sample >= self.num_samples:
            raise ValueError(
                f"Packed binary dataset {self.source_path} has {self.num_samples} "
                f"samples, fewer than rank/worker start index {first_sample}."
            )
        cursor = self._cursor

        while True:
            sample_idx = first_sample + cursor * stride
            while sample_idx < self.num_samples:
                window = self._sample_window(sample_idx)
                x = torch.from_numpy(window[:-1])
                y = torch.from_numpy(window[1:])
                # The checkpoint cursor names the next sample, including while
                # this generator is paused at yield.
                cursor += 1
                self._cursor = cursor
                yield {"input": x}, y
                sample_idx = first_sample + cursor * stride

            if not self.infinite:
                return
            cursor = 0
            self._cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self._cursor}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cursor = int(state_dict.get("cursor", 0))

    def close(self) -> None:
        while self._open_tokens:
            _, tokens = self._open_tokens.popitem()
            _close_memmap(tokens)


def _effective_training_steps(job_config: JobConfig) -> int:
    steps = int(getattr(job_config.training, "steps", 0) or 0)
    job_steps = int(getattr(job_config.job, "steps", -1) or -1)
    if job_steps > 0:
        return min(steps, job_steps) if steps > 0 else job_steps
    return max(steps, 0)


def _effective_global_batch_size(job_config: JobConfig, dp_world_size: int) -> int:
    global_batch_size = int(getattr(job_config.training, "global_batch_size", -1) or -1)
    if global_batch_size > 0:
        return global_batch_size
    return int(job_config.training.local_batch_size) * int(dp_world_size)


class PackedBinaryDataloader(StatefulDataLoader, BaseDataLoader):
    """Multiprocess/prefetching dataloader for packed token binaries."""

    def __init__(
        self,
        dataset: PackedBinaryDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        pin_memory: bool = False,
    ) -> None:
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        self._rank_id = f"dp_rank_{dp_rank}"
        logger.info(
            "Packed binary dataloader with batch size %s, %s workers, "
            "prefetch_factor=%s, pin_memory=%s.",
            batch_size,
            num_workers,
            prefetch_factor,
            pin_memory,
        )
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            self._rank_id: super().state_dict(),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict or self._rank_id not in state_dict:
            return
        assert self.dp_world_size == state_dict["world_size"]
        super().load_state_dict(state_dict[self._rank_id])


def build_packed_binary_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> BaseDataLoader:
    del tokenizer
    load_kwargs = json.loads(job_config.training.load_dataset_kwargs or "{}")
    source_path, shards, dtype = _resolve_packed_shards(
        job_config.training.dataset_path,
        load_kwargs,
        job_config.training.seq_len,
    )
    repeat = _str_to_bool(
        load_kwargs.get("repeat", load_kwargs.get("infinite")),
        default=infinite,
    )
    require_full_run = _str_to_bool(
        load_kwargs.get("require_full_run"),
        default=not repeat,
    )
    capacity_margin = float(load_kwargs.get("capacity_margin", 1.0))

    dataset = PackedBinaryDataset(
        source_path=source_path,
        shards=shards,
        seq_len=job_config.training.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        dtype=dtype,
        infinite=repeat,
        sample_count=load_kwargs.get("sample_count"),
        max_open_shards=int(load_kwargs.get("max_open_shards", 8)),
    )
    global_batch_size = _effective_global_batch_size(job_config, dp_world_size)
    effective_steps = _effective_training_steps(job_config)
    capacity_steps = dataset.num_samples // global_batch_size
    required_samples = int(effective_steps * global_batch_size * capacity_margin)
    if require_full_run and effective_steps > 0 and dataset.num_samples < required_samples:
        required_tokens = required_samples * int(job_config.training.seq_len)
        raise ValueError(
            "Packed binary dataset is too small for a no-repeat run: "
            f"{dataset.num_samples:,} samples available from {dataset.token_count:,} tokens, "
            f"but {required_samples:,} samples are required for {effective_steps:,} steps, "
            f"global_batch_size={global_batch_size:,}, seq_len={job_config.training.seq_len:,}, "
            f"capacity_margin={capacity_margin}. Build at least {required_tokens:,} packed tokens "
            "or set repeat=true for a benchmark-only run."
        )
    logger.info(
        "Using packed binary token dataset %s: %s shards, %s tokens, %s samples, "
        "seq_len=%s, dtype=%s, repeat=%s, capacity_steps=%s at global_batch_size=%s.",
        source_path,
        len(shards),
        dataset.token_count,
        dataset.num_samples,
        job_config.training.seq_len,
        dtype,
        repeat,
        capacity_steps,
        global_batch_size,
    )
    return PackedBinaryDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
        num_workers=int(load_kwargs.get("num_workers", 8)),
        prefetch_factor=int(load_kwargs.get("prefetch_factor", 4)),
        pin_memory=_str_to_bool(load_kwargs.get("pin_memory"), default=False),
    )


def build_packed_binary_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> BaseDataLoader:
    """Build a held-out packed-token validation dataloader.

    Validation has no native ``load_dataset_kwargs`` config field in the
    Torchtitan dataclass, so optional dataloader knobs come from
    ``LBT_VALIDATION_LOAD_DATASET_KWARGS``.  If unset, packed validation uses
    conservative no-repeat defaults.
    """

    del tokenizer
    load_kwargs = json.loads(os.environ.get("LBT_VALIDATION_LOAD_DATASET_KWARGS", "{}"))
    source_path, shards, dtype = _resolve_packed_shards(
        job_config.validation.dataset_path,
        load_kwargs,
        job_config.validation.seq_len,
    )
    training_dataset = str(getattr(job_config.training, "dataset", "")).lower()
    if training_dataset in {"packed-bin", "packed_binary", "token-packed"}:
        training_kwargs = json.loads(job_config.training.load_dataset_kwargs or "{}")
        training_source_path, _, _ = _resolve_packed_shards(
            job_config.training.dataset_path,
            training_kwargs,
            job_config.training.seq_len,
        )
        if _same_resolved_path(source_path, training_source_path):
            raise ValueError(
                "Packed validation dataset must be held out from training: "
                f"validation source {source_path} matches training source "
                f"{training_source_path}."
            )
    repeat = _str_to_bool(
        load_kwargs.get("repeat", load_kwargs.get("infinite")),
        default=False,
    )
    dataset = PackedBinaryDataset(
        source_path=source_path,
        shards=shards,
        seq_len=job_config.validation.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        dtype=dtype,
        infinite=repeat,
        sample_count=load_kwargs.get("sample_count"),
        max_open_shards=int(load_kwargs.get("max_open_shards", 8)),
    )
    validation_steps = int(job_config.validation.steps)
    if validation_steps > 0 and not repeat:
        required_samples = (
            validation_steps
            * int(job_config.validation.local_batch_size)
            * int(dp_world_size)
        )
        if dataset.num_samples < required_samples:
            required_tokens = required_samples * int(job_config.validation.seq_len)
            raise ValueError(
                "Packed validation dataset is too small for a no-repeat validation pass: "
                f"{dataset.num_samples:,} samples available from {dataset.token_count:,} tokens, "
                f"but {required_samples:,} samples are required for validation.steps={validation_steps}, "
                f"local_batch_size={job_config.validation.local_batch_size}, "
                f"dp_world_size={dp_world_size}. Build at least {required_tokens:,} packed tokens "
                "or set repeat=true in LBT_VALIDATION_LOAD_DATASET_KWARGS."
            )
    logger.info(
        "Using held-out packed binary validation dataset %s: %s shards, %s tokens, "
        "%s samples, seq_len=%s, dtype=%s, repeat=%s.",
        source_path,
        len(shards),
        dataset.token_count,
        dataset.num_samples,
        job_config.validation.seq_len,
        dtype,
        repeat,
    )
    return PackedBinaryDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.validation.local_batch_size,
        num_workers=int(load_kwargs.get("num_workers", 2)),
        prefetch_factor=int(load_kwargs.get("prefetch_factor", 2)),
        pin_memory=_str_to_bool(load_kwargs.get("pin_memory"), default=False),
    )
