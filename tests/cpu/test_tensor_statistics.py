# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
from typing import Tuple
import pytest
from pathlib import Path
import json
import torch

from low_bits_training.analysis import tensor_statistics
from low_bits_training.components import dtypes
from low_bits_training import JobConfig
import numpy as np


@pytest.fixture
def checkpoint_to_stream(config_and_checkpoint: Tuple[JobConfig, Path], tmp_path: Path):
    """Symlinks the checkpoints for processing"""
    checkpoint_path = config_and_checkpoint[1]
    tmp_checkpoint = tmp_path / "checkpoint" / "step-1"
    assert config_and_checkpoint[0].job.config_file
    config_path = Path(config_and_checkpoint[0].job.config_file)
    (tmp_path / config_path.name).symlink_to(config_path)
    for file in checkpoint_path.rglob("*"):
        if file.is_file():
            new_file = tmp_checkpoint / file.relative_to(checkpoint_path)
            new_file.parent.mkdir(exist_ok=True, parents=True)
            new_file.symlink_to(file)
    return tmp_checkpoint


def test_dump_checkpoint_statistics(checkpoint_to_stream: Path):
    """Integration test for the calculation of tensor statistics directly from a checkpoint."""
    tensor_statistics.dump_checkpoint_statistics(checkpoint_to_stream)

    # Check that the output file is generated and has valid jsonl format
    output_file = checkpoint_to_stream / "tensor_statistics.jsonl"

    assert output_file.exists()
    content = [
        json.loads(line) for line in output_file.read_text().splitlines() if line.strip()
    ]
    assert len(content) in [85, 86]
    print(content)
    content_has_neither_bins_nor_fqn = [
        row for row in content if "fqn" not in row and "bins" not in row
    ]
    assert not content_has_neither_bins_nor_fqn
    # Check that the histogram has as many values as there are values in the tensor
    expected_count_and_hist_count = [
        (np.prod(row["size"]), np.sum(row["hist"]))
        for row in content
        if "hist" in row and "size" in row
    ]
    assert all(
        expected == hist_count for expected, hist_count in expected_count_and_hist_count
    )


class TestStatisticsCalculations:
    def test_no_scale_1_value_per_bin(self):
        """Test that each bin has 1 value when the tensor"""
        bins = tensor_statistics.get_standard_bins().to(torch.float64)
        mid_bins = bins[1:] / 2 + bins[:-1] / 2  # divide first to avoid overflow

        stats = tensor_statistics.tensor_statistics(mid_bins, bins)

        assert all(
            val == 1.0 for val in stats["histogram"]
        ), "All bins should have one value"
        assert "mean" in stats
        assert "scale_mean" not in stats

    def test_scale_1_value_per_bin(self):
        """Test that histograms have the expected number of values"""
        value_dtype = dtypes.get_dtype_map()[torch.float8_e4m3fn]
        bins = tensor_statistics.get_standard_bins().to(torch.float64)
        mid_bins = bins[1:] / 2 + bins[:-1] / 2  # divide first to avoid overflow
        mid_bins = mid_bins.to(torch.float32)
        bins = bins.to(torch.float32)

        stats = tensor_statistics.tensor_statistics(
            mid_bins, bins, data_dtype=value_dtype, block_size=1
        )

        assert sum(stats["histogram"]) == stats["numel"]
        assert sum(stats["scale_histogram"]) == stats["scale_numel"]
        assert "mean" in stats
        assert "scale_mean" in stats


class TestBlockScaling:
    @pytest.mark.parametrize("ml_dtype", dtypes.get_dtype_map().values())
    def test_block_scale_calculation(self, ml_dtype):
        """We test the block scale calculation by checking that the scales are 1 when the maximum
        value in each block is the maximum value of the data type of the tensor."""
        # Setup
        max_val = float(tensor_statistics.finfo(ml_dtype).max)
        test_tensor = torch.tensor([max_val, 0, max_val, 0])
        block_size = 2

        expected_scales = torch.tensor([1.0, 1.0], dtype=torch.float32)

        data, scale = tensor_statistics.get_block_data_and_scale(
            test_tensor, ml_dtype, block_size
        )

        torch.testing.assert_close(data, test_tensor, atol=0, rtol=0)
        torch.testing.assert_close(scale, expected_scales, atol=0, rtol=0)

    def test_block_scale_calculation_precision(self):
        """At a block size of 1 we test that we can go from scaled dat back
        to the original tensor and that the data has been scaled to the max
        dtype value."""
        value_dtype = dtypes.get_dtype_map()[torch.float8_e4m3fn]
        bins = tensor_statistics.get_standard_bins().to(torch.float64)
        mid_bins = bins[1:] / 2 + bins[:-1] / 2  # divide first to avoid overflow
        mid_bins = mid_bins.to(torch.float32)
        bins = bins.to(torch.float32)
        value_finfo = tensor_statistics.finfo(value_dtype)

        data, scale = tensor_statistics.get_block_data_and_scale(mid_bins, value_dtype, 1)

        # Check we can get the correct value by rescaling the tensors
        torch.testing.assert_close(data.mul(scale), mid_bins)
        assert torch.all(data.sign() == mid_bins.sign())
        # Assert that the values returned by block scaling are the maximum of the dtype, this
        # is true for a block size of 1
        scaled_value_is_approximately_max = (
            data.abs() - float(value_finfo.max)
        ).abs() < 0.1
        expected = torch.where(
            mid_bins.abs() == 0,  # at 0 the scale is 1
            (mid_bins.abs() == 0) & (scale == 1) & (data == 0),
            scaled_value_is_approximately_max,
        )
        assert torch.all(expected), "Values were expected to be the maximum of the dtype"

    def test_block_scaling_errors_on_incompatible_block_size_for_tensor(self):
        value_dtype = dtypes.get_dtype_map()[torch.float8_e4m3fn]

        with pytest.raises(RuntimeError):
            tensor_statistics.get_block_data_and_scale(
                torch.ones([3], dtype=torch.float32), value_dtype, 2
            )
