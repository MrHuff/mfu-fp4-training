#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import os
from typing import Any, Dict, Optional, Union
from queue import Queue, Empty

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

import torchtitan.components.metrics
from torchtitan.config import JobConfig
import wandb
from torchtitan.distributed import ParallelDims
import torchtitan.distributed

import torchtitan.protocols.model_converter  # noqa:F401
from torchtitan.tools.logging import logger

from torchtitan.components.metrics import MetricsProcessor as TTMetricsProcessor
from torchtitan.components.metrics import BaseLogger

# Import to keep track of original method + trigger error in future TT upgrade
from torchtitan.components.metrics import _build_metric_logger as tt_build_metric_logger  # noqa:F401

_logger_model_cache: Dict[str, Any] = {
    "model": None,
}
"""Logger model (global) cache for extracting additional metrics."""


_metrics_processor_instance: TTMetricsProcessor | None = None
"""Global reference to the metrics processor
"""


def _use_async_scalar_metrics() -> bool:
    return os.environ.get("USE_LBT_ASYNC_SCALAR_METRICS", "1") == "1"


def get_metrics_processor() -> Optional[TTMetricsProcessor]:
    """Get the MetricsProcessor instance.

    Returns:
        The MetricsProcessor instance or None if not initialized
    """
    return _metrics_processor_instance


def _get_stats_gatherer(model: nn.Module):
    """Get stats gatherer from model if it exists."""
    return getattr(model, "_stats_gatherer", None)


def append_model_metrics(
    metrics: Dict[str, Union[float, int]], model: nn.Module
) -> Dict[str, Union[float, int]]:
    """
    Append layer statistics to metrics dictionary.

    Args:
        metrics: Dictionary of global metrics
        model: PyTorch model instance

    Returns:
        Updated metrics dictionary
    """

    stats_gatherer = _get_stats_gatherer(model)

    if stats_gatherer is None:
        return metrics

    # Check if this model has registered layers
    has_stats = any(
        hasattr(module, "_stats_buffer") and module._stats_buffer
        for module in model.modules()
    )

    if not has_stats:
        return metrics

    # Get buffered statistics (this moves to CPU)
    layer_stats = stats_gatherer.get_buffered_stats()

    # Append to metrics with appropriate naming
    for layer_name, stats in layer_stats.items():
        for stat_name, value in stats.items():
            key = f"model_metrics/{layer_name}.{stat_name}"
            metrics[key] = value
    return metrics


def append_optimizer_metrics(
    metrics: Dict[str, Any], optimiser: Optional[Any]
) -> Dict[str, Any]:
    """Additional optimizer metrics."""
    if optimiser is None:
        return metrics
    return metrics


def append_lr_scheduler_metrics(
    metrics: Dict[str, Any], lr_scheduler: Optional[Any]
) -> Dict[str, Any]:
    """Additional LR scheduler metrics."""
    if lr_scheduler is None:
        return metrics
    # NOTE: multiple LR schedulers when using pipelining. Last one should be reference.
    metrics["LR"] = lr_scheduler.schedulers[-1].get_last_lr()[0]
    return metrics


def dist_sum(x: Union[int, float], mesh: DeviceMesh) -> float:
    # funcol.all_reduce only supporting 1D mesh
    if mesh.size() == 1:
        return x
    tensor = torch.tensor(x).to(mesh.device_type)
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.SUM.name, group=mesh).item()


def append_total_wps_metrics(
    metrics: Dict[str, Any],
    mesh: DeviceMesh,
    *,
    mesh_dim_names: tuple[str, ...] | None = None,
):
    metric_name = "throughput(tps)"
    if metric_name in metrics:
        # Summing across the full mesh to get "raw" wps
        total_wps = metrics[metric_name]
        dimensions = mesh_dim_names or tuple(mesh.mesh_dim_names or ())
        if dimensions:
            for mesh_dim in dimensions:
                total_wps = dist_sum(total_wps, mesh[mesh_dim])
        metrics[f"total_{metric_name}"] = total_wps
    return metrics


def _is_cuda_scalar_tensor(value: Any) -> bool:
    return torch.is_tensor(value) and value.numel() == 1 and value.device.type == "cuda"


def _to_host_scalar(value: Any) -> float | int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        value = value.detach()
        if value.device.type != "cpu":
            value = value.cpu()
        return value.item()
    return value


class WBMetricLogger(BaseLogger):
    """Weight & Biases metric logger, following the same interface as
    TorchTitan Tensorboard `MetricLogger`.
    """

    def __init__(
        self,
        parallel_dims: ParallelDims,
        tag: Optional[str] = None,
    ):
        self.parallel_dims = parallel_dims
        self.tag = tag

    def log(self, metrics: Dict[str, Any], step: int):
        # Direct log into W&B.
        # Need to keep this step here, as requiring the existing metrics.
        dimensions = None
        if os.environ.get(
            "LBT_HSDP_HIERARCHICAL_SCALAR_METRICS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}:
            # TorchTitan applies HSDP on this exact pair.  In particular,
            # dp_shard_cp is a distinct flattened process group even when
            # context parallelism is one.
            dimensions = ("dp_shard_cp", "dp_replicate")
        metrics = append_total_wps_metrics(
            metrics,
            self.parallel_dims.world_mesh,
            mesh_dim_names=dimensions,
        )
        run = getattr(wandb, "run", None)
        if run is None:
            return
        wandb.log(metrics, step=step)

    def close(self):
        # Finish run?
        # if self.wandb.run is not None:
        #     self.wandb.finish()
        pass


class MetricsProcessor(TTMetricsProcessor):
    """Extending TorchTitan metrics processor."""

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: str | None = None,
    ):
        super().__init__(job_config, parallel_dims, tag)
        # Thread-safe queue for delayed metrics
        self._delayed_metrics: Queue = Queue()
        # Keep track of world mesh
        self._world_mesh: DeviceMesh = None
        self._scalar_copy_stream: torch.cuda.Stream | None = None
        self._pending_async_scalars: dict[str, Any] | None = None
        self._last_logged_scalars: dict[str, float] | None = None

    def _drain_delayed_metrics(self, extra_metrics: dict[str, Any]) -> dict[str, Any]:
        while not self._delayed_metrics.empty():
            try:
                delayed_metrics = self._delayed_metrics.get_nowait()
                extra_metrics.update(delayed_metrics)
            except Empty:
                break
        return extra_metrics

    def _get_scalar_copy_stream(self) -> torch.cuda.Stream | None:
        if not torch.cuda.is_available():
            return None
        if self._scalar_copy_stream is None:
            self._scalar_copy_stream = torch.cuda.Stream()
        return self._scalar_copy_stream

    def _refresh_async_scalar_cache(self) -> None:
        pending = self._pending_async_scalars
        if pending is None:
            return
        event = pending["event"]
        if not event.query():
            return
        self._last_logged_scalars = {
            key: _to_host_scalar(value)
            for key, value in pending["values"].items()
        }
        self._pending_async_scalars = None

    def _schedule_async_scalar_copy(
        self,
        global_avg_loss: Any,
        global_max_loss: Any,
        grad_norm: Any,
    ) -> None:
        copy_stream = self._get_scalar_copy_stream()
        if copy_stream is None:
            return
        src_stream = torch.cuda.current_stream()
        copy_stream.wait_stream(src_stream)

        values: dict[str, Any] = {}

        def _enqueue(name: str, value: Any) -> None:
            if _is_cuda_scalar_tensor(value):
                host_value = torch.empty(
                    (), dtype=value.dtype, device="cpu", pin_memory=True
                )
                value_det = value.detach().reshape(())
                with torch.cuda.stream(copy_stream):
                    value_det.record_stream(copy_stream)
                    host_value.copy_(value_det, non_blocking=True)
                values[name] = host_value
            else:
                values[name] = _to_host_scalar(value)

        _enqueue("global_avg_loss", global_avg_loss)
        _enqueue("global_max_loss", global_max_loss)
        _enqueue("grad_norm", grad_norm)
        event = torch.cuda.Event()
        event.record(copy_stream)
        self._pending_async_scalars = {"event": event, "values": values}

    def log(
        self,
        step: int,
        global_avg_loss: Any,
        global_max_loss: Any,
        grad_norm: Any,
        extra_metrics: dict[str, Any] | None = None,
    ):
        """Log, adding extra metrics on top of TorchTitan ones."""
        extra_metrics = dict(extra_metrics or {})
        # Additional custom metrics.
        extra_metrics = append_model_metrics(
            extra_metrics, _logger_model_cache.get("model")
        )
        extra_metrics = append_optimizer_metrics(extra_metrics, self.optimizers)
        extra_metrics = append_lr_scheduler_metrics(extra_metrics, self.lr_schedulers)
        extra_metrics = self._drain_delayed_metrics(extra_metrics)

        use_async_scalars = _use_async_scalar_metrics() and any(
            _is_cuda_scalar_tensor(value)
            for value in (global_avg_loss, global_max_loss, grad_norm)
        )
        if use_async_scalars:
            self._refresh_async_scalar_cache()
            if self._last_logged_scalars is None:
                # Seed the cache once. Subsequent log steps use delayed host copies.
                self._last_logged_scalars = {
                    "global_avg_loss": _to_host_scalar(global_avg_loss),
                    "global_max_loss": _to_host_scalar(global_max_loss),
                    "grad_norm": _to_host_scalar(grad_norm),
                }
            current_scalars = self._last_logged_scalars
            self._schedule_async_scalar_copy(global_avg_loss, global_max_loss, grad_norm)
            global_avg_loss = current_scalars["global_avg_loss"]
            global_max_loss = current_scalars["global_max_loss"]
            grad_norm = current_scalars["grad_norm"]
        else:
            global_avg_loss = _to_host_scalar(global_avg_loss)
            global_max_loss = _to_host_scalar(global_max_loss)
            grad_norm = _to_host_scalar(grad_norm)
        return super().log(
            step, global_avg_loss, global_max_loss, grad_norm, extra_metrics
        )

    def delayed_log(self, metrics: Dict[str, Any]):
        """Queue metrics to be logged on the next call to log()."""
        if metrics:
            self._delayed_metrics.put(metrics)

    def close(self):
        global _metrics_processor_instance
        self._refresh_async_scalar_cache()
        # Clear any remaining delayed metrics
        while not self._delayed_metrics.empty():
            try:
                self._delayed_metrics.get_nowait()
            except Empty:
                break
        # Clear the cache to avoid holding references during teardown
        _metrics_processor_instance = None
        _logger_model_cache.clear()
        # TorchTitan processor cleaning.
        super().close()


def build_metric_logger(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: Optional[str] = None
) -> WBMetricLogger:
    """Build W&B metric logger, replacing TorchTitan's default metric logger."""
    logger.info(f"Building WandDB logger with config: {job_config.wandb}.")
    return WBMetricLogger(parallel_dims, tag)


def build_metrics_processor(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: str | None = None
) -> MetricsProcessor:
    """Create a metrics processor.

    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        tag (Optional[str]): Tag to use for TensorBoard or WandB. Defaults to None.
    Returns:
        MetricsProcessor: A metrics processor.
    """
    global _metrics_processor_instance

    mp = MetricsProcessor(job_config, parallel_dims, tag)
    _metrics_processor_instance = mp
    return mp


# Monkey-patching original TorchTitan factory method.
torchtitan.components.metrics._build_metric_logger = build_metric_logger
