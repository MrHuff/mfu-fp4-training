#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Dict, List, Union, Tuple
import fnmatch

from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.distributed import ParallelDims

from low_bits_training.config import JobConfig
from collections import defaultdict

from low_bits_training.metrics import _logger_model_cache


def resolve_layer_patterns(model: nn.Module, patterns: List[str]) -> List[str]:
    """
    Resolve glob-style patterns to actual layer names in the model.

    Supports standard glob patterns:
    - '*' matches any sequence of characters
    - '?' matches any single character
    - '[seq]' matches any character in seq
    - '[!seq]' matches any character not in seq

    Args:
        model: PyTorch model
        patterns: List of glob-style patterns (e.g., "transformer.*.mlp", "*.attention")

    Returns:
        Sorted list of unique matching layer names

    Examples:
        >>> patterns = ["transformer.layers.*.mlp", "*.attention.output"]
        >>> layer_names = resolve_layer_patterns(model, patterns)
    """

    all_layer_names = [name for name, _ in model.named_modules() if name]

    if "all" in patterns:
        return all_layer_names

    if not patterns:
        return []

    matched_layers = set()
    for pattern in patterns:
        for layer_name in all_layer_names:
            if fnmatch.fnmatch(layer_name, pattern):
                matched_layers.add(layer_name)

    return [name for name in all_layer_names if name in matched_layers]


class LayerStatsGatherer:
    """Manages statistics gathering for neural network layers."""

    def __init__(
        self,
        gather_forward: bool = True,
        gather_backward: bool = False,
        gather_interval: int = 1,
    ):
        """
        Initialize the statistics gatherer.

        Args:
            gather_forward: Whether to collect stats during forward pass
            gather_backward: Whether to collect stats during backward pass (gradients)
            gather_interval: Interval of steps at which to gather statistics
        """
        self.gather_forward = gather_forward
        self.gather_backward = gather_backward
        self.gather_interval = gather_interval
        self.registered_layers = defaultdict(dict)

    def create_hook(self, layer_name: str, direction: str = "forward"):
        """Create a hook for gathering statistics."""

        def hook(module, input, output):
            if not hasattr(module, "_step_count"):
                module._step_count = 0

            if _should_gather(module._step_count, self.gather_interval):
                # Handle different output types
                stats = None
                if isinstance(output, torch.Tensor):
                    stats = compute_stats(output)
                elif isinstance(output, (tuple, list)):
                    # For modules that return multiple outputs, use the first tensor
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            stats = compute_stats(out)
                            break

                if stats is not None:
                    if not hasattr(module, "_stats_buffer"):
                        module._stats_buffer = {}

                    # Store stats in buffer (on GPU)
                    module._stats_buffer[f"{layer_name}_{direction}"] = stats

            module._step_count += 1

        return hook

    def get_buffered_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Retrieve buffered statistics and move to CPU.
        This should be called after loss computation to minimize synchronization.

        Returns:
            Dictionary of statistics with CPU values
        """
        cpu_stats = {}

        for layer, layer_name in self.registered_layers.items():
            if hasattr(layer, "_stats_buffer"):
                for layer_name, stats in layer._stats_buffer.items():
                    cpu_stats[layer_name] = {}
                    aggregated_stats = aggregate_distributed_stats(stats)

                    for stat_name, value in aggregated_stats.items():
                        cpu_stats[layer_name][stat_name] = value.item()

                # Clear buffer after retrieval
                layer._stats_buffer.clear()

        return cpu_stats

    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """
        Register hooks for specified layers in the model.

        Args:
            model: PyTorch model
            layer_names: List of layer names to monitor
        """
        handles = []

        for name, module in model.named_modules():
            if name in layer_names:
                if self.gather_forward:
                    handle = module.register_forward_hook(
                        self.create_hook(name, "forward")
                    )
                    handles.append(handle)

                if self.gather_backward:
                    handle = module.register_backward_hook(
                        self.create_hook(name, "backward")
                    )
                    handles.append(handle)

                self.registered_layers[module] = name

        return handles


def _should_gather(step_count, gather_interval) -> bool:
    """Check if statistics should be gathered at current step."""
    return step_count % gather_interval == 0


def compute_stats(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute statistics for a tensor on the appropriate device.

    Args:
        tensor: Input tensor

    Returns:
        Dictionary containing computed statistics
    """
    tensor = tensor.detach()

    # Flatten tensor for statistics computation
    flat_tensor = tensor.reshape(-1)

    stats = {
        "mean": flat_tensor.mean(),
        "std": flat_tensor.std(unbiased=False),
        "absmax": flat_tensor.abs().max(),
        "norm": flat_tensor.norm(),
    }

    return stats


def aggregate_distributed_stats(
    stats: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Aggregate statistics across distributed processes.

    Args:
        stats: Dictionary of statistics tensors

    Returns:
        Aggregated statistics
    """
    if not dist.is_initialized():
        return stats

    world_size = dist.get_world_size()
    aggregated_stats = {}

    for key, value in stats.items():
        if key == "mean":
            # For mean, we need to average across processes
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            aggregated_stats[key] = value / world_size
        elif key == "absmax":
            # For absmax, take the maximum across processes
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            aggregated_stats[key] = value
        elif key == "norm":
            # For norm, we need to combine L2 norms: sqrt(sum of squares)
            value_squared = value**2
            dist.all_reduce(value_squared, op=dist.ReduceOp.SUM)
            aggregated_stats[key] = value_squared.sqrt()
        elif key == "std":
            # For std, this is approximate but efficient
            # More accurate would require two-pass algorithm
            value_squared = value**2
            dist.all_reduce(value_squared, op=dist.ReduceOp.SUM)
            aggregated_stats[key] = (value_squared / world_size).sqrt()
        else:
            aggregated_stats[key] = value

    return aggregated_stats


# Store stats gatherer as a model attribute
def attach_stats_gatherer(model: nn.Module, stats_gatherer: LayerStatsGatherer):
    """Attach stats gatherer to model as an attribute."""
    model._stats_gatherer = stats_gatherer


def convert_model_with_stats(
    model: nn.Module,
    layer_names: List[str],
    gather_forward: bool = True,
    gather_backward: bool = False,
    gather_interval: int = 1,
) -> Tuple[nn.Module, LayerStatsGatherer]:
    """
    Convert a model to gather statistics for specified layers.

    Args:
        model: PyTorch model to instrument
        layer_names: List of layer names to monitor
        gather_forward: Whether to collect stats during forward pass
        gather_backward: Whether to collect stats during backward pass
        gather_interval: Interval of steps at which to gather statistics
    Returns:
        Tuple of (model, stats_gatherer)
    """
    stats_gatherer = LayerStatsGatherer(
        gather_forward=gather_forward,
        gather_backward=gather_backward,
        gather_interval=gather_interval,
    )

    stats_gatherer.register_hooks(model, layer_names)
    attach_stats_gatherer(model, stats_gatherer)
    _logger_model_cache["model"] = model
    return model, stats_gatherer


class LayerStatsConverter(ModelConverter):
    """
    Enable gathering of layer output statistics
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        assert not job_config.compile.enable, "torch.compile with the LayerStatsConverter is not currently supported. Set --compile.no-enable"

        self.model_config = job_config.model
        self.stats_config = job_config.stats
        self.layer_patterns = self.stats_config.layer_patterns
        self.gather_forward = self.stats_config.gather_forward
        self.gather_backward = self.stats_config.gather_backward
        self.gather_interval = job_config.metrics.log_freq

    def convert(self, model: nn.Module):
        layer_names = resolve_layer_patterns(model, self.layer_patterns)

        convert_model_with_stats(
            model,
            layer_names,
            gather_forward=self.gather_forward,
            gather_backward=self.gather_backward,
            gather_interval=self.gather_interval,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass


register_model_converter(LayerStatsConverter, "stats")
