#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.dataloader import ParallelAwareDataloader

import low_bits_training.batch_resume as batch_resume_module
from low_bits_training.batch_resume import (
    CheckpointAlignedDataloader,
    fingerprint_batch,
    install_checkpoint_aligned_dataloader,
)
from low_bits_training.datasets.packed_binary import (
    PackedBinaryDataloader,
    PackedBinaryDataset,
    _PackedShard,
)
import low_bits_training.trainer as trainer_module


class _NumberedDataset(Dataset):
    def __init__(self, size: int):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        value = torch.tensor(index, dtype=torch.int64)
        return {"input": value}, value + 1


class _CursorDataloader:
    def __init__(self):
        self.cursor = 0

    def __iter__(self) -> _CursorDataloader:
        return self

    def __next__(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        value = self.cursor
        self.cursor += 1
        labels = torch.tensor([value + 1, value + 2], dtype=torch.int64)
        return {"input": labels - 1}, labels

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.cursor = state_dict["cursor"]


class _FakeCudaStream:
    def wait_event(self, event: object) -> None:
        del event


class _FakeCudaEvent:
    def record(self, stream: object) -> None:
        del stream


def _build_dataloader() -> StatefulDataLoader:
    return StatefulDataLoader(
        _NumberedDataset(64),
        batch_size=1,
        num_workers=1,
        prefetch_factor=4,
        persistent_workers=True,
        snapshot_every_n_steps=1,
    )


def _build_parallel_aware_dataloader() -> ParallelAwareDataloader:
    return ParallelAwareDataloader(
        _NumberedDataset(64),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        num_workers=1,
        prefetch_factor=4,
        persistent_workers=True,
    )


def _close_dataloader(dataloader: StatefulDataLoader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def _lookahead_values(
    adapter: CheckpointAlignedDataloader,
) -> Iterator[int]:
    iterator = iter(adapter.dataloader)
    pending = adapter.next_for_prefetch(iterator)
    while True:
        adapter.mark_prefetched_batch_current()
        current = pending
        try:
            pending = adapter.next_for_prefetch(iterator)
        except StopIteration:
            yield int(current[0]["input"].item())
            return
        yield int(current[0]["input"].item())


def _take_ga2_steps(iterator: Iterator[int], steps: int) -> list[list[int]]:
    return [[next(iterator), next(iterator)] for _ in range(steps)]


def _build_position_unique_packed_loader(
    token_path: Path,
) -> tuple[PackedBinaryDataset, PackedBinaryDataloader]:
    seq_len = 8
    sample_count = 128
    tokens = np.arange(sample_count * seq_len + 1, dtype=np.uint32)
    tokens.tofile(token_path)
    shard = _PackedShard(
        path=token_path,
        token_count=int(tokens.size),
        sample_count=sample_count,
    )
    dataset = PackedBinaryDataset(
        source_path=token_path,
        shards=[shard],
        seq_len=seq_len,
        dp_rank=0,
        dp_world_size=2,
        dtype="uint32",
        infinite=False,
    )
    dataloader = PackedBinaryDataloader(
        dataset,
        dp_rank=0,
        dp_world_size=2,
        batch_size=3,
        num_workers=0,
        pin_memory=False,
    )
    return dataset, dataloader


def _assert_batch_byte_exact(
    actual: tuple[dict[str, torch.Tensor], torch.Tensor],
    expected: tuple[dict[str, torch.Tensor], torch.Tensor],
) -> None:
    assert torch.equal(actual[0]["input"], expected[0]["input"])
    assert torch.equal(actual[1], expected[1])
    assert fingerprint_batch(*actual) == fingerprint_batch(*expected)


def test_cuda_lookahead_checkpoint_replays_pending_batch_with_ga2() -> None:
    """num_workers=1/prefetch=4 resumes the exact continuous GA2 samples."""
    continuous_loader = _build_dataloader()
    resumed_loader = _build_dataloader()
    old_semantics_loader = _build_dataloader()
    try:
        continuous_adapter = CheckpointAlignedDataloader(continuous_loader)
        continuous = _lookahead_values(continuous_adapter)

        assert _take_ga2_steps(continuous, 1) == [[0, 1]]
        aligned_state = continuous_adapter.state_dict()
        old_semantics_state = copy.deepcopy(continuous_loader.state_dict())
        expected_after_checkpoint = _take_ga2_steps(continuous, 2)

        resumed_adapter = CheckpointAlignedDataloader(resumed_loader)
        resumed_adapter.load_state_dict(aligned_state)
        resumed = _lookahead_values(resumed_adapter)
        assert _take_ga2_steps(resumed, 2) == expected_after_checkpoint
        assert expected_after_checkpoint == [[2, 3], [4, 5]]

        # This is the prior bug: the raw dataloader state is already after the
        # trainer-owned pending batch and resumes at sample 3 instead of 2.
        old_semantics_loader.load_state_dict(old_semantics_state)
        old_semantics = iter(old_semantics_loader)
        assert int(next(old_semantics)[0]["input"].item()) == 3
    finally:
        _close_dataloader(continuous_loader)
        _close_dataloader(resumed_loader)
        _close_dataloader(old_semantics_loader)


def test_position_unique_packed_state_advances_before_yield(tmp_path: Path) -> None:
    dataset, _ = _build_position_unique_packed_loader(tmp_path / "direct.bin")
    try:
        first = next(iter(dataset))
        assert first[0]["input"][0].item() == 0
        assert dataset.state_dict() == {"cursor": 1}
    finally:
        dataset.close()


def test_position_unique_packed_no_worker_resume_replays_pending_batches(
    tmp_path: Path,
) -> None:
    """Packed iterable state advances before yield and resumes byte-exactly."""
    continuous_dataset, continuous_loader = _build_position_unique_packed_loader(
        tmp_path / "continuous.bin"
    )
    resumed_dataset, resumed_loader = _build_position_unique_packed_loader(
        tmp_path / "resumed.bin"
    )
    try:
        continuous_adapter = CheckpointAlignedDataloader(continuous_loader)
        continuous_iterator = iter(continuous_loader)
        pending = continuous_adapter.next_for_prefetch(continuous_iterator)

        for _ in range(6):
            continuous_adapter.mark_prefetched_batch_current()
            pending = continuous_adapter.next_for_prefetch(continuous_iterator)

        checkpoint_state = copy.deepcopy(continuous_adapter.state_dict())
        rank_state = checkpoint_state["dp_rank_0"]
        assert rank_state["dataset_state"]["cursor"] == 18
        assert rank_state["_num_yielded"] == 6
        assert rank_state["_sampler_iter_state"]["samples_yielded"] == 18

        expected_batches = []
        for _ in range(3):
            continuous_adapter.mark_prefetched_batch_current()
            expected_batches.append(pending)
            pending = continuous_adapter.next_for_prefetch(continuous_iterator)
        assert len({fingerprint_batch(*batch).combined_sha256 for batch in expected_batches}) == 3

        resumed_adapter = CheckpointAlignedDataloader(resumed_loader)
        resumed_adapter.load_state_dict(checkpoint_state)
        resumed_iterator = iter(resumed_loader)
        resumed_pending = resumed_adapter.next_for_prefetch(resumed_iterator)
        for expected in expected_batches:
            resumed_adapter.mark_prefetched_batch_current()
            _assert_batch_byte_exact(resumed_pending, expected)
            resumed_pending = resumed_adapter.next_for_prefetch(resumed_iterator)
    finally:
        continuous_dataset.close()
        resumed_dataset.close()


def test_parallel_aware_capture_defers_pickle_until_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production wrapper retains its schema without hot-path pickle work."""
    continuous_loader = _build_parallel_aware_dataloader()
    resumed_loader = _build_parallel_aware_dataloader()
    try:
        adapter = CheckpointAlignedDataloader(continuous_loader)
        iterator = iter(continuous_loader)

        real_pickle_dumps = batch_resume_module.pickle.dumps
        pickle_calls = 0

        def counted_pickle_dumps(*args, **kwargs):
            nonlocal pickle_calls
            pickle_calls += 1
            return real_pickle_dumps(*args, **kwargs)

        monkeypatch.setattr(batch_resume_module.pickle, "dumps", counted_pickle_dumps)

        first = adapter.next_for_prefetch(iterator)
        assert int(first[0]["input"].item()) == 0
        adapter.mark_prefetched_batch_current()
        second = adapter.next_for_prefetch(iterator)
        assert int(second[0]["input"].item()) == 1
        assert pickle_calls == 0

        aligned_state = adapter.state_dict()
        assert pickle_calls == 1
        assert set(aligned_state) == {"dp_rank_0", "world_size"}
        assert isinstance(aligned_state["dp_rank_0"], bytes)
        assert aligned_state["world_size"] == 1

        resumed_loader.load_state_dict(aligned_state)
        resumed = iter(resumed_loader)
        assert int(next(resumed)[0]["input"].item()) == 1
    finally:
        _close_dataloader(continuous_loader)
        _close_dataloader(resumed_loader)


def test_generic_loader_uses_public_state_dict_capture() -> None:
    dataloader = _CursorDataloader()
    adapter = CheckpointAlignedDataloader(dataloader)
    state_dict_calls = 0
    original_state_dict = dataloader.state_dict

    def counted_state_dict() -> dict[str, int]:
        nonlocal state_dict_calls
        state_dict_calls += 1
        return original_state_dict()

    dataloader.state_dict = counted_state_dict  # type: ignore[method-assign]
    iterator = iter(dataloader)
    pending = adapter.next_for_prefetch(iterator)

    assert pending[0]["input"].tolist() == [0, 1]
    assert state_dict_calls == 1
    assert adapter.state_dict() == {"cursor": 0}


def test_trainer_cuda_lookahead_accounts_and_hashes_only_actual_yield(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataloader = _CursorDataloader()
    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    trainer.device = torch.device("cuda")
    trainer.step = 42
    trainer.ntokens_seen = 0
    trainer.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        data_loading_times=[],
    )
    trainer._prefetch_checkpoint_dataloader = CheckpointAlignedDataloader(dataloader)

    fake_stream = _FakeCudaStream()
    monkeypatch.setattr(trainer_module, "_pin_memory_tree", lambda value: value)
    monkeypatch.setattr(trainer_module, "_to_device_tree", lambda value, device: value)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: fake_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda **kwargs: fake_stream)
    monkeypatch.setenv("USE_LBT_DEBUG_BATCH_HASHES", "1")
    monkeypatch.setenv("RANK", "7")

    batches = trainer.batch_generator(dataloader)
    assert trainer.ntokens_seen == 0
    first_inputs, first_labels = next(batches)

    assert first_inputs["input"].tolist() == [0, 1]
    assert first_labels.tolist() == [1, 2]
    assert dataloader.cursor == 2  # Batch 1 is the device-copy lookahead.
    assert trainer.ntokens_seen == 2  # Only batch 0 has actually been yielded.
    assert trainer.metrics_processor.ntokens_since_last_log == 2
    assert trainer._prefetch_checkpoint_dataloader.state_dict() == {"cursor": 1}

    second_inputs, second_labels = next(batches)
    assert second_inputs["input"].tolist() == [1, 2]
    assert second_labels.tolist() == [2, 3]
    assert trainer.ntokens_seen == 4

    stderr = capsys.readouterr().err
    assert "rank=7 step=42 microbatch=0" in stderr
    assert "rank=7 step=42 microbatch=1" in stderr
    assert stderr.count("input_sha256=") == 2
    assert stderr.count("labels_sha256=") == 2


def test_install_replaces_persistent_and_fault_tolerance_states() -> None:
    dataloader = object()
    checkpointer = SimpleNamespace(
        states={"dataloader": dataloader, "train_state": object()},
        ft_states={"dataloader": dataloader},
    )

    adapter = install_checkpoint_aligned_dataloader(checkpointer, dataloader)

    assert checkpointer.states["dataloader"] is adapter
    assert checkpointer.ft_states["dataloader"] is adapter


def test_install_rejects_a_mismatched_checkpointer_dataloader() -> None:
    checkpointer = SimpleNamespace(states={"dataloader": object()})

    with pytest.raises(RuntimeError, match="does not reference"):
        install_checkpoint_aligned_dataloader(checkpointer, object())


def test_batch_fingerprint_separates_inputs_and_labels() -> None:
    input_dict = {
        "input": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        "metadata": (torch.tensor([5], dtype=torch.int32),),
    }
    labels = torch.tensor([[2, 3], [4, 5]], dtype=torch.int64)

    first = fingerprint_batch(input_dict, labels)
    repeated = fingerprint_batch(
        dict(reversed(list(input_dict.items()))), labels.clone()
    )
    changed_labels = fingerprint_batch(input_dict, labels + 1)

    assert first == repeated
    assert first.input_sha256 == changed_labels.input_sha256
    assert first.labels_sha256 != changed_labels.labels_sha256
    assert first.combined_sha256 != changed_labels.combined_sha256
    assert first.input_numel == 5
    assert first.labels_numel == 4
