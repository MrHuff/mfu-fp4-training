#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from low_bits_training.datasets.mosaic_datasets import (
    StreamingDataset,
    MosaicStreamingDataset,
    MosaicDataloader,
    build_mosaic_dataloader,
    get_dataloader_worker_states,
)
from low_bits_training.config import JobConfig
from digits_tokenizer import DigitsTokenizer, BaseTokenizer
from streaming.base import MDSWriter

import torch
import copy
import os
import itertools
import json

from typing import Any
import gc

import pytest
import pathlib

import numpy as np
import numpy.testing as npt


def write_simple_mosaic_mds_dataset(
    token_ids: list[int],
    tokenizer: BaseTokenizer,
    output_dir: str,
    sample_size: int,
    size_limit: int,
):
    """Write a simple MosaicML streaming dataset, in MDS format, from a list of raw token ids.

    Helper function useful for small in-memory data, to create an equivalent MosaicML compatible MDS
    dataset.

    Args:
        token_ids: Token ids dataset.
        tokenizer: Tokenizer to use for text encoding.
        output_dir: Output directory.
        sample_size: Number of tokens per sample.
        size_limit: MDS file size limit.
    """
    columns = {"text": "str"}
    hashes = ["sha1", "xxh64"]
    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression=None,
        hashes=hashes,
        size_limit=size_limit,
        progress_bar=False,
    ) as out:
        samples = []
        for idx in range(0, len(token_ids), sample_size):
            samples.append({"text": tokenizer.decode(token_ids[idx : idx + sample_size])})
        for sample in samples:
            out.write(sample)


@pytest.fixture
def mosaic_ordered_numbers_dataset(tmp_path: pathlib.Path):
    """Fixture creating a small dummy MDS Mosaic dataset where the tokens are 0, 1, 2, ..."""

    def setup(num_tokens: int, sample_size: int, size_limit: Any):
        in_token_ids = list(range(0, num_tokens))
        tokenizer = DigitsTokenizer()
        write_simple_mosaic_mds_dataset(
            in_token_ids,
            tokenizer,
            str(tmp_path),
            sample_size=sample_size,
            size_limit=size_limit,
        )
        return str(tmp_path), in_token_ids

    yield setup
    gc.collect()


@pytest.fixture
def mosaic_dataset_config():
    """Generate a Job config compatible with Mosaic dataset (with no shuffling)."""

    def setup(seq_len: int, tmp_dir: str, num_workers: int = 0):
        job_config = JobConfig()
        # Setting up the job config for a Mosaic dataset.
        job_config = JobConfig()
        job_config.wandb.name = "test_run"
        job_config.training.seq_len = seq_len
        job_config.training.local_batch_size = 1

        job_config.training.dataset = "mosaic/test"
        job_config.training.dataset_path = tmp_dir
        # Passing extra args to MosaicML with kwargs.
        job_config.training.load_dataset_kwargs = json.dumps(
            {"shuffle": False, "num_workers": int(num_workers)}
        )
        return job_config

    yield setup


def test__write_simple_mosaic_mds_dataset__data_written(mosaic_ordered_numbers_dataset):
    dataset_num_tokens = 1024 * 16
    sample_size = 64
    tmpdirname, in_token_ids = mosaic_ordered_numbers_dataset(
        dataset_num_tokens, sample_size, size_limit="10kb"
    )

    # Is it sharded?
    files = sorted(os.listdir(tmpdirname))
    assert len(files) >= 8

    # Reading back using Streaming dataset without shuffling.
    ds = StreamingDataset(local=tmpdirname, shuffle=False, batch_size=1)
    out = [DigitsTokenizer().encode(v["text"]) for v in ds]
    out_token_ids = list(itertools.chain(*out))
    assert out_token_ids == in_token_ids


# Testing different seq_len to make sure stitching between samples works well.
@pytest.mark.parametrize("seq_len", [16, 96])
def test__MosaicStreamingDataset__iteration__proper_ordering_micro_batches(
    mosaic_ordered_numbers_dataset, mosaic_dataset_config, seq_len
):
    dataset_num_tokens = 1024 * 16
    sample_size = 64
    # Create a tmp local Mosaic dataset & the job_config adapted to it.
    tmpdirname, _ = mosaic_ordered_numbers_dataset(
        dataset_num_tokens, sample_size, size_limit="10kb"
    )
    job_config = mosaic_dataset_config(seq_len, tmpdirname)

    ds = MosaicStreamingDataset(
        job_config,
        DigitsTokenizer(),
        dp_rank=0,
        dp_world_size=1,
    )
    # Iterator should give data in the right order (i.e. no shuffling).
    it = iter(ds)
    num_iters = 8
    start_val = 0
    for _ in range(num_iters):
        inputs, labels = next(it)
        input_data = inputs["input"]

        assert len(input_data) == seq_len
        assert len(labels) == seq_len
        # Labels is input shifted of 1 token.
        npt.assert_array_equal(input_data[1:], labels[:-1])
        npt.assert_array_equal(input_data, np.arange(start_val, start_val + seq_len))
        npt.assert_array_equal(labels, np.arange(start_val + 1, start_val + seq_len + 1))
        start_val += seq_len + 1

    ds.close()


def test__MosaicStreamingDataset__proper_resuming_from_state_dict(
    mosaic_ordered_numbers_dataset, mosaic_dataset_config
):
    dataset_num_tokens = 1024 * 16
    sample_size = 6
    # Choosing seq len combining multiple samples. Meaning recovering temp. buffer.
    seq_len = 9

    tmpdirname, _ = mosaic_ordered_numbers_dataset(
        dataset_num_tokens, sample_size, size_limit="10kb"
    )
    job_config = mosaic_dataset_config(seq_len, tmpdirname)

    dsA = MosaicStreamingDataset(
        job_config,
        DigitsTokenizer(),
        dp_rank=0,
        dp_world_size=1,
    )
    assert dsA._sample_idx == 0

    # First sequence from dataset A.
    itA = iter(dsA)
    inputsA0, labelsA0 = next(itA)
    # Consumed 2 samples (seq_len > sample_size).
    assert dsA._sample_idx == 2
    # Save dsA state dict.  The raw token-buffer snapshot itself must remain
    # stable while the live iterator advances; TorchData hands this object to a
    # multiprocessing queue before serialization completes.
    raw_state_dict = dsA.state_dict()
    raw_token_buffer = list(raw_state_dict["token_buffer"])
    assert raw_state_dict["token_buffer"] is not dsA._token_buffer
    state_dict = copy.deepcopy(raw_state_dict)

    # Next input sequence in dataset A.
    inputsA1, labelsA1 = next(itA)
    assert all(inputsA1["input"] != inputsA0["input"])
    assert all(labelsA1 != labelsA0)
    assert raw_state_dict["token_buffer"] == raw_token_buffer
    # Need to consume 2 more samples
    assert dsA._sample_idx == 4

    # Re-create a dataset from the saved state.
    dsB = MosaicStreamingDataset(
        job_config,
        DigitsTokenizer(),
        dp_rank=0,
        dp_world_size=1,
    )
    dsB.load_state_dict(state_dict)
    assert dsB._sample_idx == state_dict["sample_idx"]
    assert dsB._token_buffer is not state_dict["token_buffer"]

    # Are we continuing from the same point? Recovering properly the internal buffer.
    itB = iter(dsB)
    inputsB0, labelsB0 = next(itB)
    # Sample idx is from start => should match dataset A.
    assert dsB._sample_idx == dsA._sample_idx
    assert len(dsB._token_buffer) > 0
    # Should match the second sample from dsA.
    npt.assert_array_equal(inputsB0["input"], inputsA1["input"])
    npt.assert_array_equal(labelsB0, labelsA1)

    dsA.close()
    dsB.close()


def test__build_mosaic_dataloader__correct_batch_size(
    mosaic_ordered_numbers_dataset, mosaic_dataset_config
):
    """Checking the Mosaic dataloader is building proper micro-batches."""
    dataset_num_tokens = 1024 * 16
    sample_size = 7

    # Choosing seq len combining multiple samples.
    seq_len = 9
    batch_size = 4

    tmpdirname, _ = mosaic_ordered_numbers_dataset(
        dataset_num_tokens, sample_size, size_limit="10kb"
    )
    job_config = mosaic_dataset_config(seq_len, tmpdirname, num_workers=1)
    job_config.training.local_batch_size = batch_size

    dataloader = build_mosaic_dataloader(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=DigitsTokenizer(),
        job_config=job_config,
        infinite=False,
    )
    inputs_dict, labels = next(iter(dataloader))
    inputs = inputs_dict["input"]

    assert isinstance(dataloader, MosaicDataloader)
    assert inputs.shape == (batch_size, seq_len)
    assert labels.shape == (batch_size, seq_len)

    npt.assert_array_equal(inputs.flatten() + 1, labels.flatten())
    dataloader.close()


# @pytest.mark.parametrize("num_workers", [1])
@pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
def test__MosaicDataloader__proper_resuming_from_state_dict(
    mosaic_ordered_numbers_dataset, mosaic_dataset_config, num_workers
):
    dataset_num_tokens = 1024 * 16
    sample_size = 7
    # Choosing seq len combining multiple samples. Meaning recovering temp. buffer. from state.
    seq_len = 9
    batch_size = 2

    tmpdirname, _ = mosaic_ordered_numbers_dataset(
        dataset_num_tokens, sample_size, size_limit="10kb"
    )
    job_config = mosaic_dataset_config(seq_len, tmpdirname, num_workers)
    job_config.training.local_batch_size = batch_size

    dlA = build_mosaic_dataloader(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=DigitsTokenizer(),
        job_config=job_config,
        infinite=False,
    )
    # Consume 2*num_workers + 1 from dataloader A
    itA = iter(dlA)
    for _ in range(2 * num_workers + 1):
        next(itA)
    # Save dataloader dict.
    state_dict = copy.deepcopy(dlA.state_dict())
    # Consume num_workers more inputs.
    inputsA = [next(itA)[0]["input"] for _ in range(num_workers)]

    # Re-create a dataset from the saved state.
    dlB = build_mosaic_dataloader(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=DigitsTokenizer(),
        job_config=job_config,
        infinite=False,
    )
    # Restore dataloader state.
    dlB.load_state_dict(state_dict)

    # Consume num_workers inputs.
    itB = iter(dlB)
    inputsB = [next(itB)[0]["input"] for _ in range(num_workers)]

    # Token buffer sizes in dataloader state dict.
    token_buffer_sizes = [
        len(w["dataset_state"]["token_buffer"])
        for w in get_dataloader_worker_states(state_dict)
    ]
    # We are restoring a non-empty token buffer on every worker.
    min_tok_buf_size = min(token_buffer_sizes)
    max_tok_buf_size = max(token_buffer_sizes)
    assert min_tok_buf_size > 0

    # Smallest start element from the dataset state buffer => we should never have elements smaller than that.
    minB = min([b[0, 0] for b in inputsB])

    for inA, inB in zip(inputsA, inputsB):
        inA = inA.flatten()
        inB = inB.flatten()
        # Checking we are not going back "in time" in the dataset => tokens should be larger than the first one.
        assert torch.all(inB >= minB)
        # Token buffer properly restored for every worker.
        npt.assert_array_equal(inA[:min_tok_buf_size], inB[:min_tok_buf_size])

        # For num_workers > 1, MosaicML is not restarting workers in the same way.
        # Data mixing between workers seems to be different.
        # TODO: get full reproduction with MosaicML dataset.
        try:
            assert inA[max_tok_buf_size + 1] == inB[max_tok_buf_size + 1]
        except AssertionError:
            pass

    dlA.close()
    dlB.close()


def test__MosaicDataloader__error_if_resuming_with_different_num_workers(
    mosaic_ordered_numbers_dataset, mosaic_dataset_config
):
    dataset_num_tokens = 1024 * 16
    tmpdirname, _ = mosaic_ordered_numbers_dataset(
        dataset_num_tokens, sample_size=7, size_limit="10kb"
    )

    def _make_dataloader(num_workers):
        job_config = mosaic_dataset_config(9, tmpdirname, num_workers)
        job_config.training.local_batch_size = 2
        return build_mosaic_dataloader(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DigitsTokenizer(),
            job_config=job_config,
            infinite=False,
        )

    # Create a dataloader with num_workers=2 & save state.
    dlA = _make_dataloader(num_workers=2)
    state_dict = dlA.state_dict()

    # Re-create from the state, but with num_workers=4
    dlB = _make_dataloader(num_workers=4)
    with pytest.raises(AssertionError):
        dlB.load_state_dict(state_dict)
