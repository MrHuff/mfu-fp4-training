# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
from typing import Generator, Tuple, TypedDict, Callable, NamedTuple, Any
from pathlib import Path
import time

import torch
from torch import Tensor
import ml_dtypes
import numpy as np

from .. import utils
from .. import quantization
from . import stream_checkpoints
from ..components.dtypes import MLDtype, TorchDType, finfo, from_torch_to_ml_dtypes


from torchtitan.tools.logging import init_logger, logger


def block_scale_hist_bins():
    """Block scale histogram bins: max, smallest, smallest subnormal
    of different formats:

        E4M3, E5M3, BF16
    """
    dtypes = [
        ml_dtypes.float8_e4m3fn,
        ml_dtypes.bfloat16,
        "float8_e5m3",
        ml_dtypes.float8_e8m0fnu,
    ]
    bins = []
    for d in dtypes:
        fi = finfo(d)
        bins += [fi.max, fi.smallest_normal, fi.smallest_subnormal]
    bins = np.array(bins, dtype=np.float32)
    bins = np.sort(bins)
    return bins


def get_block_data_and_scale(t, data_dtype, bs: int = 32) -> Tuple[Tensor, Tensor]:
    """(data, scale) block estimates, without quantization."""
    import torch

    max_val = float(finfo(data_dtype).max)
    data_hp = t.reshape((-1, bs)).to(torch.float32)
    # scale & data, without casting.
    scale = torch.amax(torch.abs(data_hp), dim=-1).unsqueeze(-1) / max_val
    # Avoid division by 0 on underflow - might not be needed on real tensors
    # but is needed to pass tests.
    scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    data = (data_hp / scale).reshape(t.shape)
    scale = scale.squeeze(dim=-1)
    return (data, scale)


def get_standard_bins() -> Tensor:
    """Returns the bins ready to be passed to torch.hist

    The "standard" bins have been design to support numerical analysis of the values
    in the tensor without reloading the tensor from a checkpoint. Key properties of the bins:
    - symmetric around 0
    - distributed logarithmically
    - additional bin boundaries are placed at e5m3, e4m3, e8m0 and bf16 min and max values
    """
    log2_bins = torch.cat(
        [torch.tensor(block_scale_hist_bins()).log2(), torch.arange(-17, 18)],
    )
    bins = torch.cat([-(2**log2_bins), (2**log2_bins)]).sort().values.unique()
    return bins


def tensor_statistics(
    tensor: Tensor,
    bins: None | Tensor = None,
    data_dtype: None | MLDtype = None,
    block_size: int = 32,
):
    """Calculates statistics and histograms for a tensor

    Accounts for block scale formats by estimating the scale for each block
    and rescaling the values according to those scales. Then calculates the
    statistics for the scale and the values.

    Calculates min, max, mean, std for signed and absolute value tensors, and an histogram.
    """
    if bins is None:
        bins = get_standard_bins()

    scale = None
    if data_dtype is not None:
        tensor, scale = get_block_data_and_scale(tensor, data_dtype, block_size)

    output_stats = dict()
    output_stats["data_dtype"] = str(data_dtype)
    output_stats.update(basic_statistics(tensor, bins))
    if scale is not None:
        output_stats.update(
            {f"scale_{k}": v for k, v in basic_statistics(scale, bins).items()}
        )
    return output_stats


def basic_statistics(tensor: Tensor, bins: Tensor):
    """Calculates mean, max, min, std and histograms for a tensor."""
    t_abs = tensor.abs()
    return dict(
        size=list(tensor.size()),
        numel=tensor.numel(),
        min=tensor.min().item(),
        max=tensor.max().item(),
        mean=tensor.mean().item(),
        std=tensor.std().item(),
        abs_min=t_abs.min().item(),
        abs_max=t_abs.max().item(),
        abs_mean=t_abs.mean().item(),
        abs_std=t_abs.std().item(),
        histogram=tensor.histogram(bins=bins).hist.tolist(),
    )


def plot_histogram(histogram, bins=None):
    """Given a histogram and some bins plot them"""
    import matplotlib.pyplot as plt

    if bins is None:
        bins = get_standard_bins()
    fig, ax = plt.subplots(1, 1)
    line_x = (
        torch.cat([bins[:-1].reshape(1, -1), bins[1:].reshape(1, -1)], dim=0)
        .transpose(0, 1)
        .flatten()
    )
    line_y = (
        torch.cat(
            [
                torch.tensor(histogram).reshape(1, -1),
                torch.tensor(histogram).reshape(1, -1),
            ],
            dim=0,
        )
        .transpose(0, 1)
        .flatten()
    )
    line_y[line_y == 0] = 1e-6
    ax.plot(line_x, line_y)

    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Tensor Histogram")
    min_nonzero_bin = bins[:-1][(torch.tensor(histogram) > 0)].min() * 2
    max_nonzero_bin = bins[1:][(torch.tensor(histogram) > 0)].max() * 2
    logger.info(f"{min_nonzero_bin=}, {max_nonzero_bin=}")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlim(min_nonzero_bin, max_nonzero_bin)
    ax.set_yscale("log")
    ax.set_ylim(1e-1, max(line_y) * 2)
    return fig, ax


def find_config_from_checkpoint(ckpt_path: Path) -> utils.JobConfig:
    run = Path(ckpt_path).parents[1]
    config_files = list(run.glob("config-*"))
    assert len(config_files) >= 1, f"No config found in {run}"
    config_file = config_files[0]
    logger.info(f"Using config file {config_file}")
    return utils.load_config(config_file, allow_upgrade=True, validate=False)


class ScaledFloatDTypes(TypedDict, total=False):
    """A class which represents a model's scaled floating points dtypes"""

    block_size: int
    weight: TorchDType | None
    activation: TorchDType | None
    gradient: TorchDType | None
    fqn_filter: Callable[[str], bool]


def get_tensor_dtypes(config: utils.JobConfig) -> ScaledFloatDTypes:
    """Interpret the Job config to understand what the dtype for each type of layer"""

    def all_but_output(fqn: str):
        return "output" not in fqn

    swap_filter_functions = {
        "all_but_output": all_but_output,
    }
    if "mxfp" in config.model.converters and "mx_rmsnorm" in config.model.converters:
        assert config.mxfp.block_size == config.mx_rmsnorm.block_size
        assert config.mxfp.weight_dtype == config.mx_rmsnorm.data_dtype
    if "mxfp" in config.model.converters or "mx_norm_linear" in config.model.converters:
        return ScaledFloatDTypes(
            block_size=config.mxfp.block_size,
            weight=quantization.mxfp.to_mxfp_dtype(config.mxfp.weight_dtype),
            activation=quantization.mxfp.to_mxfp_dtype(config.mxfp.activation_dtype),
            gradient=quantization.mxfp.to_mxfp_dtype(config.mxfp.gradient_dtype),
            fqn_filter=swap_filter_functions[config.mxfp.swap_filter_fn],
        )
    if "mx_rmsnorm" in config.model.converters:
        return ScaledFloatDTypes(
            block_size=config.mx_rmsnorm.block_size,
            weight=quantization.mxfp.to_mxfp_dtype(config.mx_rmsnorm.data_dtype),
            activation=None,
            gradient=None,
            fqn_filter=lambda fqn: "norm" in fqn,
        )
    return {}


############################################################################
# Analyse checkpoint file statistics
############################################################################


class _TensorStatisticsCalculationOutput(NamedTuple):
    fqn: str
    stats: dict
    load_time: float
    process_time: float
    size: tuple[int, ...]


def stream_checkpoint_statistics(
    ckpt: Path, bins: None | Tensor = None, **kwargs
) -> Generator[_TensorStatisticsCalculationOutput, Any, None]:
    """Iterate over a checkpoint streaming the statistics of the tensor.

    Statistics are calculated with the ``tensor_statistics`` function.
    """
    config = find_config_from_checkpoint(ckpt)
    dtypes = get_tensor_dtypes(config)
    if bins is None:
        bins = get_standard_bins()
    logger.info("Using dtypes: %s", dtypes)
    load_start = time.time()
    for fqn, tensor in stream_checkpoints.stream_checkpoint_reader(ckpt, **kwargs):
        process_start = time.time()
        load_time = process_start - load_start
        block_size = dtypes.get("block_size", 32)
        data_dtype = None
        if dtypes.get("fqn_filter", lambda fqn: False)(fqn):
            data_dtype = dtypes.get("weight", None)
        if data_dtype is not None:
            data_dtype = from_torch_to_ml_dtypes(data_dtype)
        try:
            stats = tensor_statistics(
                tensor, bins=bins, data_dtype=data_dtype, block_size=block_size
            )
        except Exception as e:
            logger.info(
                f"Error processing tensor {fqn} of size {tensor.size()} with dtype {tensor.dtype}: {e}"
            )
            stats = {"error": str(e)}
        process_time = time.time() - process_start
        yield _TensorStatisticsCalculationOutput(
            fqn, stats, load_time, process_time, tensor.size()
        )
        load_start = time.time()


def dump_checkpoint_statistics(ckpt: Path):
    """Collects all checkpoint statistics and dumps them to a jsonl file in the checkpoint directory."""
    import json

    ckpt = Path(ckpt)
    out_file = ckpt / "tensor_statistics.jsonl"
    assert (
        not out_file.exists()
    ), f"Output file already exists {out_file}, please delete it first"
    with open(out_file, "w") as f:
        count = 0
        total_mb = 0
        total_processing_time = 0.0
        total_load_time = 0.0
        bins = get_standard_bins()
        json_line = json.dumps({"bins": bins.tolist()})
        f.write(json_line + "\n")
        for (
            fqn,
            stats,
            load_time,
            process_time,
            tensor_size,
        ) in stream_checkpoint_statistics(ckpt, bins=bins, batch_tensors=100):
            stats["fqn"] = fqn
            json_line = json.dumps(stats)
            f.write(json_line + "\n")
            count += 1
            mb = (
                np.prod(tensor_size) * 4 / 1024**2
            )  # 4 is for the number of bytes per element
            total_mb += mb
            total_processing_time += process_time
            total_load_time += load_time
            if count % 100 == 0:
                logger.info(
                    f"Processed {count} tensors, {total_mb:.2f}MB, load time {total_load_time:.2f}s, processing time {total_processing_time:.2f}s"
                )
                logger.info(
                    f"Average load rate {total_mb / total_load_time:.2f}MB/s, processing rate {total_mb / total_processing_time:.2f}MB/s"
                )

    logger.info(f"Processing complete for checkpoint {ckpt}")
    logger.info(f"Statistics dumped to {out_file}")
    logger.info(
        f"Processed {count} tensors, {total_mb:.2f}MB, load time {total_load_time:.2f}s, processing time {total_processing_time:.2f}s"
    )
    logger.info(
        f"Average load rate {total_mb / total_load_time:.2f}MB/s, processing rate {total_mb / total_processing_time:.2f}MB/s"
    )


def main():
    import argparse

    init_logger()

    parser = argparse.ArgumentParser(
        description="Dump checkpoint statistics to jsonl file"
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint).resolve()
    if (checkpoint_dir / "checkpoint").exists():
        for checkpoint_root in checkpoint_dir.glob("*checkpoint"):
            if not checkpoint_root.is_dir():
                continue
            for checkpoint_dir in checkpoint_root.glob("step-*"):
                try:
                    logger.info(f"Start processing Checkpoint {checkpoint_dir}")
                    dump_checkpoint_statistics(checkpoint_dir)
                except Exception as e:
                    logger.error(f"[ERROR] for {checkpoint_dir} : {e}")
    else:
        dump_checkpoint_statistics(checkpoint_dir)


if __name__ == "__main__":
    main()
