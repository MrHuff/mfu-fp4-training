"""Benchmark-only MXFP4 TK backend and fused wrappers.

This mirrors the NVFP4 localCTA decomposition at a high level:
- absorbed RMSNorm for QKV / FFN
- one activation quant for the normed input
- stacked QKV GEMM
- stacked W1/W3 GEMM
- row-oriented dgrad and col-oriented wgrad

Unlike NVFP4 localCTA, there is no amax machinery anywhere in this path.
The implementation stays eager-only and benchmark-oriented for now.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

_te_apply_rotary_pos_emb = None
_MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE: dict[
    int, tuple[object, ...]
] = {}
_MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE: dict[
    int, tuple[object, ...]
] = {}
_MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE: dict[
    int, tuple[object, ...]
] = {}
_MXFP4_FSDP_FORWARD_LAYER_INDICES: frozenset[int] | None = None
_MXFP4_FSDP_BACKWARD_LAYER_INDICES: frozenset[int] | None = None
_MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE: dict[int, object] = {}
_MXFP4_FSDP_REDUCE_SCATTER_EVENT_BY_STREAM: dict[
    tuple[int, int], object
] = {}
_MXFP4_FSDP_COMM_CONTEXT_LOGGED_BY_DEVICE: set[int] = set()
_MXFP4_FSDP_REDUCE_SCATTER_WAIT_LOGGED_BY_DEVICE: set[int] = set()
_MXFP4_FSDP_QKV_OVERLAP_STATES_LOGGED: set[
    tuple[int, str, bool, bool, bool, bool, bool]
] = set()
_MXFP4_CUDNN_SDPA_AUTOGRAD_WARMUP = threading.local()
_MXFP4_QKV_FORWARD_KEEPALIVE_BY_DEVICE: dict[
    int, deque[tuple[torch.cuda.Event, tuple[object, ...]]]
] = {}
_MXFP4_FLIGHT_RECORDER_LOCK = threading.Lock()
_MXFP4_FLIGHT_RECORDER_BY_DEVICE: dict[
    int, deque[tuple[int, str, str, torch.cuda.Event]]
] = {}
_MXFP4_FLIGHT_RECORDER_SEQ_BY_DEVICE: dict[int, int] = {}
_MXFP4_FLIGHT_RECORDER_LAST_COMPLETED_BY_DEVICE: dict[
    int, tuple[int, str, str]
] = {}
_MXFP4_FLIGHT_RECORDER_ENABLED = (
    os.environ.get("MXFP4_FLIGHT_RECORDER", "0") == "1"
)
_MXFP4_FLIGHT_RECORDER_STAGES_RAW = os.environ.get(
    "MXFP4_FLIGHT_RECORDER_STAGES", ""
).strip()
_MXFP4_FLIGHT_RECORDER_STAGES = (
    frozenset(
        stage.strip()
        for stage in _MXFP4_FLIGHT_RECORDER_STAGES_RAW.split(",")
        if stage.strip()
    )
    if _MXFP4_FLIGHT_RECORDER_STAGES_RAW
    else None
)
try:
    _MXFP4_FLIGHT_RECORDER_LAG = max(
        1, int(os.environ.get("MXFP4_FLIGHT_RECORDER_LAG", "64"))
    )
except ValueError:
    _MXFP4_FLIGHT_RECORDER_LAG = 64


class _DelayedMXFP4FSDPForwardPrefetchSentinel:
    _fsdp_param_group = None


_DELAYED_MXFP4_FSDP_FORWARD_PREFETCH_SENTINEL = (
    _DelayedMXFP4FSDPForwardPrefetchSentinel()
)


class _DelayedMXFP4FSDPBackwardPrefetchSentinel:
    _fsdp_param_group = None


_DELAYED_MXFP4_FSDP_BACKWARD_PREFETCH_SENTINEL = (
    _DelayedMXFP4FSDPBackwardPrefetchSentinel()
)


def _first_cuda_device(value) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device if value.is_cuda else None
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        return None
    for item in values:
        if (device := _first_cuda_device(item)) is not None:
            return device
    return None


def _parse_mxfp4_fsdp_layer_ranges(ranges: str) -> frozenset[int]:
    """Parse a 1-based FSDP layer-range string into zero-based indices."""
    indices: set[int] = set()
    for item in ranges.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
        else:
            start = end = int(item)
        if start <= 0 or end < start:
            raise ValueError(
                "TORCHTITAN_FSDP_MXFP4_LAYER_RANGES must contain positive "
                f"1-based ranges, got {item!r}"
            )
        indices.update(range(start - 1, end))
    return frozenset(indices)


def _register_mixed_mxfp4_fsdp_layer_indices(
    *,
    forward: set[int] | frozenset[int],
    backward: set[int] | frozenset[int],
) -> None:
    """Register the actual mixed-converter producers for delayed prefetch.

    Forward prefetch is released by a fused MXFP4 QKV producer, while backward
    prefetch is released by a fused MXFP4 FFN producer. Keeping the scopes
    separate also makes component overrides fail safe.
    """
    forward_scope = frozenset(forward)
    backward_scope = frozenset(backward)
    if any(index < 0 for index in forward_scope | backward_scope):
        raise ValueError(
            "MXFP4 FSDP layer indices must be zero-based and non-negative"
        )

    configured_ranges = os.environ.get(
        "TORCHTITAN_FSDP_MXFP4_LAYER_RANGES",
        "",
    ).strip()
    if configured_ranges:
        configured_scope = _parse_mxfp4_fsdp_layer_ranges(configured_ranges)
        if (
            configured_scope != forward_scope
            or configured_scope != backward_scope
        ):
            raise RuntimeError(
                "TORCHTITAN_FSDP_MXFP4_LAYER_RANGES does not match the actual "
                "mixed MXFP4 QKV/FFN layer scopes"
            )

    global _MXFP4_FSDP_FORWARD_LAYER_INDICES
    global _MXFP4_FSDP_BACKWARD_LAYER_INDICES
    _MXFP4_FSDP_FORWARD_LAYER_INDICES = forward_scope
    _MXFP4_FSDP_BACKWARD_LAYER_INDICES = backward_scope


def _fsdp_module_uses_mxfp4(
    module_fqn: str,
    registered_scope: frozenset[int] | None = None,
) -> bool:
    """Return whether an FSDP layer is routed through the MXFP4 wrappers."""
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_fqn or "")
    if match is None:
        return False

    if registered_scope is not None:
        return int(match.group(1)) in registered_scope

    ranges = os.environ.get(
        "TORCHTITAN_FSDP_MXFP4_LAYER_RANGES",
        "",
    ).strip()
    if not ranges:
        return True
    return int(match.group(1)) in _parse_mxfp4_fsdp_layer_ranges(ranges)


def _install_mxfp4_fsdp_all_gather_stream_priority() -> None:
    raw_all_gather_priority = os.environ.get(
        "TORCHTITAN_FSDP_MXFP4_ALL_GATHER_STREAM_PRIORITY",
    )
    raw_reduce_scatter_priority = os.environ.get(
        "TORCHTITAN_FSDP_MXFP4_REDUCE_SCATTER_STREAM_PRIORITY",
    )
    if raw_all_gather_priority is None and raw_reduce_scatter_priority is None:
        return

    def _parse_priority(raw_value: str | None, env_name: str) -> int | None:
        if raw_value is None:
            return None
        if raw_value.strip().lower() == "native":
            return None
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer or 'native'") from exc

    all_gather_priority = _parse_priority(
        raw_all_gather_priority,
        "TORCHTITAN_FSDP_MXFP4_ALL_GATHER_STREAM_PRIORITY",
    )
    reduce_scatter_priority = _parse_priority(
        raw_reduce_scatter_priority,
        "TORCHTITAN_FSDP_MXFP4_REDUCE_SCATTER_STREAM_PRIORITY",
    )
    if all_gather_priority is None and reduce_scatter_priority is None:
        return

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPCommContext,
    )

    if getattr(
        FSDPCommContext,
        "_lbt_mxfp4_all_gather_stream_priority_installed",
        False,
    ):
        return

    original_lazy_init = FSDPCommContext.lazy_init

    def _lazy_init_with_mxfp4_all_gather_priority(self, device):
        original_lazy_init(self, device)
        # No work has been submitted when lazy_init returns, so replacing these
        # streams cannot orphan an event. Matching the compute-stream priority
        # prevents a high-priority collective from preempting one CTA in a
        # two-CTA Blackwell tensor-core cluster. Reduce-scatter is controlled
        # separately because it can remain live into the next accumulated
        # microbatch after all-gather has completed.
        if all_gather_priority is not None:
            self.all_gather_copy_in_stream = self.device_handle.Stream(
                priority=all_gather_priority
            )
            self.all_gather_stream = self.device_handle.Stream(
                priority=all_gather_priority
            )
        if reduce_scatter_priority is not None:
            self.reduce_scatter_stream = self.device_handle.Stream(
                priority=reduce_scatter_priority
            )

    FSDPCommContext.lazy_init = _lazy_init_with_mxfp4_all_gather_priority
    FSDPCommContext._lbt_mxfp4_all_gather_stream_priority_installed = True


_install_mxfp4_fsdp_all_gather_stream_priority()


def _register_mxfp4_fsdp_comm_context(
    device: torch.device,
    comm_ctx: object,
) -> None:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE[device_index] = comm_ctx
    if device_index not in _MXFP4_FSDP_COMM_CONTEXT_LOGGED_BY_DEVICE:
        print(
            "[MXFP4 FSDP] registered communication context for fused RoPE "
            f"QKV ordering on cuda:{device_index}",
            file=sys.stderr,
            flush=True,
        )
        _MXFP4_FSDP_COMM_CONTEXT_LOGGED_BY_DEVICE.add(device_index)


def _install_mxfp4_fsdp_reduce_scatter_qkv_ordering() -> None:
    if os.environ.get(
        "TORCHTITAN_FSDP_ORDER_REDUCE_SCATTER_BEFORE_MX_ROPE_QKV",
        "0",
    ) != "1":
        return

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPCommContext,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    if getattr(
        FSDPCommContext,
        "_lbt_mxfp4_reduce_scatter_qkv_ordering_installed",
        False,
    ):
        return

    original_lazy_init = FSDPCommContext.lazy_init

    def _lazy_init_with_mxfp4_qkv_ordering(self, device):
        original_lazy_init(self, device)
        device = torch.device(device)
        if device.type != "cuda":
            return
        _register_mxfp4_fsdp_comm_context(device, self)

    original_pre_forward = FSDPState._pre_forward

    def _pre_forward_with_mxfp4_qkv_ordering(self, module, args, kwargs):
        device = _first_cuda_device((args, kwargs))
        if device is not None:
            _register_mxfp4_fsdp_comm_context(device, self._comm_ctx)
        return original_pre_forward(self, module, args, kwargs)

    FSDPCommContext.lazy_init = _lazy_init_with_mxfp4_qkv_ordering
    FSDPCommContext._lbt_mxfp4_reduce_scatter_qkv_ordering_installed = True
    FSDPState._pre_forward = _pre_forward_with_mxfp4_qkv_ordering
    FSDPState._lbt_mxfp4_reduce_scatter_qkv_ordering_installed = True


def _order_mxfp4_rope_qkv_after_fsdp_reduce_scatter(
    device: torch.device,
) -> bool:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    comm_ctx = _MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE.get(device_index)
    reduce_scatter_state = (
        getattr(comm_ctx, "reduce_scatter_state", None)
        if comm_ctx is not None
        else None
    )
    event = getattr(reduce_scatter_state, "event", None)
    if event is None:
        return False

    stream = torch.cuda.current_stream(device)
    stream_id = int(getattr(stream, "cuda_stream", id(stream)))
    stream_key = (device_index, stream_id)
    if _MXFP4_FSDP_REDUCE_SCATTER_EVENT_BY_STREAM.get(stream_key) is event:
        return False

    stream.wait_event(event)
    _MXFP4_FSDP_REDUCE_SCATTER_EVENT_BY_STREAM[stream_key] = event
    if device_index not in _MXFP4_FSDP_REDUCE_SCATTER_WAIT_LOGGED_BY_DEVICE:
        print(
            "[MXFP4 FSDP] ordering fused RoPE QKV after outstanding "
            f"reduce-scatter on cuda:{device_index}",
            file=sys.stderr,
            flush=True,
        )
        _MXFP4_FSDP_REDUCE_SCATTER_WAIT_LOGGED_BY_DEVICE.add(device_index)
    return True


def _mxfp4_current_graph_task_phase() -> str:
    current_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)
    if current_graph_task_id is None:
        return "unknown"
    try:
        return "backward" if int(current_graph_task_id()) >= 0 else "forward"
    except Exception:
        return "unknown"


def _log_mxfp4_fsdp_overlap_before_rope_qkv(device: torch.device) -> None:
    if os.environ.get(
        "TORCHTITAN_FSDP_LOG_MX_ROPE_QKV_OVERLAP",
        "0",
    ) != "1":
        return

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    comm_ctx = _MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE.get(device_index)
    all_gather_state = (
        getattr(comm_ctx, "all_gather_state", None)
        if comm_ctx is not None
        else None
    )
    reduce_scatter_state = (
        getattr(comm_ctx, "reduce_scatter_state", None)
        if comm_ctx is not None
        else None
    )
    state = (
        device_index,
        _mxfp4_current_graph_task_phase(),
        device_index in _MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE,
        device_index in _MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE,
        device_index in _MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE,
        getattr(all_gather_state, "event", None) is not None,
        getattr(reduce_scatter_state, "event", None) is not None,
    )
    if state in _MXFP4_FSDP_QKV_OVERLAP_STATES_LOGGED:
        return
    _MXFP4_FSDP_QKV_OVERLAP_STATES_LOGGED.add(state)
    (
        _,
        phase,
        delayed_forward,
        delayed_backward,
        active_backward,
        all_gather_event,
        reduce_scatter_event,
    ) = state
    print(
        "[MXFP4 FSDP] fused RoPE QKV overlap state "
        f"cuda:{device_index} phase={phase} "
        f"delayed_forward={int(delayed_forward)} "
        f"delayed_backward={int(delayed_backward)} "
        f"active_backward={int(active_backward)} "
        f"all_gather_event={int(all_gather_event)} "
        f"reduce_scatter_event={int(reduce_scatter_event)}",
        file=sys.stderr,
        flush=True,
    )


_install_mxfp4_fsdp_reduce_scatter_qkv_ordering()


def _install_delayed_mxfp4_fsdp_forward_prefetch() -> None:
    if os.environ.get(
        "TORCHTITAN_FSDP_DELAY_FORWARD_PREFETCH_UNTIL_MX_QKV",
        "0",
    ) != "1":
        return

    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    if getattr(
        FSDPState,
        "_lbt_delayed_mxfp4_forward_prefetch_installed",
        False,
    ):
        return

    original_pre_forward = FSDPState._pre_forward

    def _pre_forward_after_mxfp4_qkv(self, module, args, kwargs):
        if self._training_state == TrainingState.PRE_BACKWARD:
            return original_pre_forward(self, module, args, kwargs)

        target_states = tuple(self._states_to_forward_prefetch)
        param_group = self._fsdp_param_group
        module_fqn = getattr(param_group, "_module_fqn", "") if param_group else ""
        device = _first_cuda_device((args, kwargs))
        delay_prefetch = (
            bool(target_states)
            and _fsdp_module_uses_mxfp4(
                module_fqn,
                _MXFP4_FSDP_FORWARD_LAYER_INDICES,
            )
            and device is not None
        )
        if not delay_prefetch:
            return original_pre_forward(self, module, args, kwargs)

        device_index = device.index
        if device_index is None:
            raise RuntimeError("delayed MXFP4 forward prefetch requires a CUDA device index")
        if device_index in _MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE:
            raise RuntimeError(
                "an earlier delayed MXFP4 forward prefetch was not released"
            )

        self._states_to_forward_prefetch = [
            _DELAYED_MXFP4_FSDP_FORWARD_PREFETCH_SENTINEL
        ]
        try:
            result = original_pre_forward(self, module, args, kwargs)
        finally:
            self._states_to_forward_prefetch = list(target_states)
        _MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE[device_index] = (
            target_states
        )
        return result

    FSDPState._pre_forward = _pre_forward_after_mxfp4_qkv
    FSDPState._lbt_delayed_mxfp4_forward_prefetch_installed = True


def _release_delayed_mxfp4_fsdp_forward_prefetch(
    device: torch.device,
) -> bool:
    device_index = device.index
    if device_index is None:
        return False
    target_states = _MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE.pop(
        device_index,
        None,
    )
    if target_states is None:
        return False

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
    )

    qkv_done = torch.cuda.Event()
    qkv_done.record(torch.cuda.current_stream(device))
    for target_state in target_states:
        target_param_group = getattr(target_state, "_fsdp_param_group", None)
        if target_param_group is not None:
            target_param_group._wait_all_gather_streams_on_event(qkv_done)
            FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
    return True


_install_delayed_mxfp4_fsdp_forward_prefetch()


def _install_delayed_mxfp4_fsdp_backward_prefetch() -> None:
    delay_enabled = os.environ.get(
        "TORCHTITAN_FSDP_DELAY_BACKWARD_PREFETCH_UNTIL_MX_FFN_PRODUCER",
        "0",
    ) == "1"
    legacy_drain_enabled = os.environ.get(
        "TORCHTITAN_FSDP_DRAIN_BACKWARD_PREFETCH_BEFORE_MX_QKV",
        "0",
    ) == "1"
    if not (delay_enabled or legacy_drain_enabled):
        return

    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    if getattr(
        FSDPState,
        "_lbt_delayed_mxfp4_backward_prefetch_installed",
        False,
    ):
        return

    original_pre_backward = FSDPState._pre_backward

    def _pre_backward_after_mxfp4_ffn_producer(self, grad):
        target_states = tuple(self._states_to_backward_prefetch)
        param_group = self._fsdp_param_group
        module_fqn = getattr(param_group, "_module_fqn", "") if param_group else ""
        device = _first_cuda_device(grad)
        delay_prefetch = (
            bool(target_states)
            and _fsdp_module_uses_mxfp4(
                module_fqn,
                _MXFP4_FSDP_BACKWARD_LAYER_INDICES,
            )
            and device is not None
        )
        if not delay_prefetch:
            return original_pre_backward(self, grad)

        device_index = device.index
        if device_index is None:
            raise RuntimeError(
                "delayed MXFP4 backward prefetch requires a CUDA device index"
            )
        if device_index in _MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE:
            raise RuntimeError(
                "an earlier delayed MXFP4 backward prefetch was not released"
            )

        self._states_to_backward_prefetch = [
            _DELAYED_MXFP4_FSDP_BACKWARD_PREFETCH_SENTINEL
        ]
        try:
            result = original_pre_backward(self, grad)
        finally:
            self._states_to_backward_prefetch = list(target_states)
        _MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE[device_index] = (
            target_states
        )
        return result

    FSDPState._pre_backward = _pre_backward_after_mxfp4_ffn_producer
    FSDPState._lbt_delayed_mxfp4_backward_prefetch_installed = True


def _release_delayed_mxfp4_fsdp_backward_prefetch(
    device: torch.device,
    producer_streams: tuple[torch.cuda.Stream, ...] = (),
) -> bool:
    device_index = device.index
    if device_index is None:
        return False
    if device_index in _MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE:
        raise RuntimeError(
            "an earlier MXFP4 backward prefetch was not drained"
        )
    target_states = _MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE.pop(
        device_index,
        None,
    )
    if target_states is None:
        return False

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
    )

    producer_events = [torch.cuda.Event()]
    producer_events[0].record(torch.cuda.current_stream(device))
    for producer_stream in producer_streams:
        producer_done = torch.cuda.Event()
        producer_done.record(producer_stream)
        producer_events.append(producer_done)
    for target_state in target_states:
        target_param_group = getattr(target_state, "_fsdp_param_group", None)
        if target_param_group is not None:
            for producer_done in producer_events:
                target_param_group._wait_all_gather_streams_on_event(
                    producer_done
                )
            FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
    _MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE[device_index] = (
        target_states
    )
    return True


def _drain_mxfp4_fsdp_backward_prefetch(device: torch.device) -> bool:
    device_index = device.index
    if device_index is None:
        return False
    target_states = _MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE.pop(
        device_index,
        None,
    )
    if target_states is None:
        return False

    if _MXFP4_FLIGHT_RECORDER_ENABLED:
        _mxfp4_record_stage_completion(
            "fsdp_bwd_prefetch_drain_entry",
            None,
        )
    try:
        for target_state in target_states:
            target_param_group = getattr(target_state, "_fsdp_param_group", None)
            if target_param_group is not None:
                target_param_group.wait_for_unshard()
    except Exception:
        _mxfp4_report_flight_failure(
            device_index,
            "fsdp_bwd_prefetch_drain",
        )
        raise
    return True


_install_delayed_mxfp4_fsdp_backward_prefetch()


def _get_te_apply_rotary_pos_emb():
    global _te_apply_rotary_pos_emb
    if _te_apply_rotary_pos_emb is None:
        try:
            from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
        except ImportError:
            _te_apply_rotary_pos_emb = False
        else:
            _te_apply_rotary_pos_emb = apply_rotary_pos_emb
    return None if _te_apply_rotary_pos_emb is False else _te_apply_rotary_pos_emb

from .fused_te_linear import _as_contiguous_bf16, _get_te_fused, _safe_trunc_normal_
from .sqrelu import sqrelu, sqrelu_bwd, sqrelu_fwd
from .tk_gemm import (
    _get_tk_mixed_mx_localcta_quant,
    _launch_rmsnorm_bwd_out_async,
    tk_mixed_localcta_dgrad,
    tk_mixed_localcta_split2_dgrad,
    tk_mixed_mx_localcta_quant_capabilities,
)
from .mxfp4_backend import (
    mxfp4_batched_gemm,
    mxfp4_batched_gemm_rope,
    mxfp4_batched_gemm_rope_live64,
    mxfp4_backend_version,
    mxfp4_batched_kv_gemm_rope_live64,
    mxfp4_batched_qkv_gemm_rope_live64,
    mxfp4_copy_col_slices,
    mxfp4_dot_and_pack_indexed_scaled_rows_bf16_variable,
    mxfp4_fused_rmsnorm_to_bf16,
    mxfp4_fused_rmsnorm_quantize_row_and_col,
    mxfp4_fused_rmsnorm_quantize_row_and_col_from_row_rms_partial,
    mxfp4_fused_rmsnorm_quantize_row_and_col_opt,
    mxfp4_fused_silu_mul_quantize_row_and_col,
    mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace,
    mxfp4_fused_silu_mul_sigmoid_quantize_row_and_col_launch_inplace,
    mxfp4_fused_silu_mul_quantize_row_and_col_strided,
    mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace,
    mxfp4_fused_sqrelu_quantize_row_and_col,
    mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace,
    mxfp4_fused_sqrelu_deriv_quantize_row_and_col,
    mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace,
    mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace,
    mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_bf16_launch_inplace,
    mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace,
    mxfp4_fused_silu_deriv_quantize_split2_row_and_col,
    mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace,
    mxfp4_fused_silu_deriv_quantize_split2_row_and_col_opt_launch_inplace,
    mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols,
    mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace,
    mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_and_col_splitcols_launch_inplace,
    mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined,
    mxfp4_dense_gemm_config_for_shape,
    mxfp4_gemm,
    mxfp4_gemm_residual,
    mxfp4_gemm_residual_rms,
    mxfp4_h_residual_carrier,
    mxfp4_h_tile_backward,
    mxfp4_gemm_residual_config,
    mxfp4_gemm_rope,
    mxfp4_gemm_rope_live64,
    mxfp4_gemm_rope_live64_config,
    mxfp4_gemm_rope_config,
    mxfp4_gemm_config,
    mxfp4_gemm_sqrelu_deriv_config,
    mxfp4_gemm_silu_dgrad_quant,
    mxfp4_gemm_silu_dgrad_from_sigmoid_quant,
    mxfp4_gemm_silu_dgrad_from_sigmoid_row_bf16_quant,
    mxfp4_grouped_gemm_strided,
    mxfp4_moe_build_route_inverse,
    mxfp4_moe_build_route_inverse_padded,
    mxfp4_moe_gather_scores,
    mxfp4_moe_indexed_dot_rows_padded_bf16,
    mxfp4_moe_indexed_dot_rows_bf16,
    mxfp4_moe_route_combine_bf16,
    mxfp4_moe_route_combine_padded_index_bf16,
    mxfp4_moe_route_combine_padded_bf16,
    mxfp4_moe_route_scatter_gradx_bf16,
    mxfp4_moe_route_scatter_gradx_padded_index_bf16,
    mxfp4_moe_scale_scatter_add_bf16,
    mxfp4_moe_scatter_scores,
    mxfp4_moe_scatter_add_bf16,
    mxfp4_pack_grouped_rows_bf16,
    mxfp4_pack_grouped_rows_quantize_row_and_col,
    mxfp4_pack_indexed_rows_bf16,
    mxfp4_pack_indexed_rmsnorm_rows_bf16_variable,
    mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col,
    mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col_variable,
    mxfp4_pack_indexed_scaled_rows_bf16_variable,
    mxfp4_pack_indexed_scaled_rows_quantize_row_and_col,
    mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable,
    mxfp4_pack_shared_routed_rows_bf16,
    mxfp4_pack_w13_bf16,
    mxfp4_batched_gemm_config,
    mxfp4_quantize_col_only,
    mxfp4_quantize_col_only_opt,
    mxfp4_quantize_col_only_opt_rht,
    mxfp4_quantize_for_gemm,
    mxfp4_quantize_for_gemm_opt,
    mxfp4_quantize_for_gemm_opt_rht,
    mxfp4_quantize_nhsd_wo_row_and_col,
    mxfp4_quantize_row_and_col,
    mxfp4_quantize_weight_2d,
    mxfp4_quantize_row_and_col_launch_inplace,
    mxfp4_quantize_row_and_col_opt,
    mxfp4_quantize_row_and_col_opt_rht,
    mxfp4_quantize_split2_col_only_launch_inplace,
    mxfp4_quantize_split2_col_only_opt_launch_inplace,
    mxfp4_quantize_split2_row_only_launch_inplace,
    mxfp4_quantize_split2_row_only_opt_launch_inplace,
    mxfp4_scatter_grouped_rows_bf16,
    mxfp4_split_w13_bf16,
    mxfp4_split2_dgrad_strided_onepass_gemm,
    mxfp4_split2_dgrad_strided_onepass_h_gemm,
    mxfp4_split3_dgrad_strided_onepass_gemm,
    mxfp4_quantize_split2_row_and_col,
    mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace,
    mxfp4_quantize_split2_row_and_col_launch_inplace,
    mxfp4_quantize_split2_row_and_col_opt_launch_inplace,
    mxfp4_quantize_split3_row_and_col,
    mxfp4_quantize_split3_row_and_col_inverse_rope_live64,
    mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace,
    mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace,
    mxfp4_quantize_split3_row_and_col_launch_inplace,
    mxfp4_quantize_split3_row_and_col_opt_launch_inplace,
    mxfp4_rope_live_head_dim_available,
)


def _mxfp4_v4_quant_extension() -> bool:
    return mxfp4_backend_version().strip().lower() == "v4"


def use_mxfp4_tk_fused() -> bool:
    return os.environ.get("USE_MXFP4_TK_FUSED", "0") == "1"


def use_mxfp4_fused_rmsnorm_quant(kind: str) -> bool:
    scoped = os.environ.get(f"MXFP4_USE_FUSED_RMSNORM_QUANT_{kind.upper()}")
    if scoped is not None:
        return scoped == "1"
    if kind in {"qkv", "ffn"}:
        return os.environ.get(
            "MXFP4_USE_FUSED_RMSNORM_QUANT",
            "1" if _mxfp4_v4_quant_extension() else "0",
        ) == "1"
    return os.environ.get("MXFP4_USE_FUSED_RMSNORM_QUANT", "0") == "1"


def use_mxfp4_split3_qkv_quant() -> bool:
    return os.environ.get(
        "MXFP4_USE_SPLIT3_QKV_QUANT",
        "1" if _mxfp4_v4_quant_extension() else "0",
    ) == "1"


def use_mxfp4_split3_qkv_stage_copy(m: int | None = None) -> bool:
    value = os.environ.get("MXFP4_USE_SPLIT3_QKV_STAGE_COPY")
    if value is not None:
        return value == "1"
    # The v4 split3 QKV grad quantizer builds host-side descriptors for its
    # BF16 inputs. At the 8B single-GPU M=32768 shape, autograd can hand us
    # producer-stream grad buffers whose lifetime is not stable enough for the
    # async quantize/GEMM chain unless we stage them onto current-stream scratch.
    return m is not None and m <= 32768 and _mxfp4_v4_quant_extension()


def _mxfp4_split3_qkv_stage_copy_mask(m: int | None = None) -> set[str]:
    if not use_mxfp4_split3_qkv_stage_copy(m):
        return set()
    raw = os.environ.get("MXFP4_SPLIT3_QKV_STAGE_COPY_MASK", "qkv").strip().lower()
    if raw in {"", "all", "1", "true", "yes"}:
        raw = "qkv"
    if raw in {"0", "false", "none", "off"}:
        return set()
    mask = {ch for ch in raw if ch in {"q", "k", "v"}}
    if not mask:
        raise ValueError(
            "MXFP4_SPLIT3_QKV_STAGE_COPY_MASK must contain q, k, and/or v; "
            f"got {raw!r}"
        )
    return mask


def use_mxfp4_split3_qkv_onepass_dgrad(
    m: int | None = None,
    q_dim: int | None = None,
    k_dim: int | None = None,
    v_dim: int | None = None,
) -> bool:
    env_value = os.environ.get("MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD")
    if env_value is not None:
        return env_value == "1"
    # End-to-end legacy MFU tuning favors the split/overlap dgrad path here.
    return False


def use_mxfp4_split3_qkv_inplace_quant() -> bool:
    return os.environ.get("MXFP4_USE_SPLIT3_QKV_INPLACE_QUANT", "1") == "1"


def use_mxfp4_qkv_direct_outputs() -> bool:
    return os.environ.get("MXFP4_USE_QKV_DIRECT_OUTPUTS", "1") == "1"


def use_mxfp4_qkv_forward_sync() -> bool:
    return os.environ.get("MXFP4_USE_QKV_FORWARD_SYNC", "0") == "1"


def use_mxfp4_qkv_rope_epilogue() -> bool:
    return os.environ.get("MXFP4_USE_QKV_ROPE_EPILOGUE", "1") == "1"


def use_mxfp4_deepseek_mla_rope_epilogue() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_MLA_ROPE_EPILOGUE", "0") == "1"


def use_mxfp4_deepseek_mla_fused_kv_b() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_MLA_FUSED_KV_B", "0") == "1"


def use_mxfp4_deepseek_mla_fused_attn_wo() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_MLA_FUSED_ATTN_WO", "0") == "1"


def use_mxfp4_deepseek_mla_padded_wq_wkva_param() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_MLA_PADDED_WQ_WKVA_PARAM", "1") == "1"


def use_low_bits_tk_b300_mla_attention() -> bool:
    return os.environ.get("LOW_BITS_TK_B300_MLA_ATTENTION", "0") == "1"


def use_mxfp4_generic_qkv_rope_epilogue() -> bool:
    return os.environ.get("MXFP4_USE_GENERIC_QKV_ROPE_EPILOGUE", "0") == "1"


def use_mxfp4_split2_persistent_grad_sr() -> bool:
    return os.environ.get("MXFP4_USE_SPLIT2_FFN_PERSISTENT_GRAD_SR", "0") == "1"


def _mxfp4_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mxfp4_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def use_mxfp4_qkv_bf16_wgrad() -> bool:
    return _mxfp4_bool_env("MXFP4_USE_QKV_BF16_WGRAD", False)


def use_mxfp4_qkv_bf16_dgrad() -> bool:
    return _mxfp4_bool_env("MXFP4_USE_QKV_BF16_DGRAD", False)


def use_mxfp4_qkv_bf16_q_forward() -> bool:
    return _mxfp4_bool_env("MXFP4_USE_QKV_BF16_Q_FORWARD", False)


def use_mxfp4_qkv_bf16_kv_forward() -> bool:
    return _mxfp4_bool_env("MXFP4_USE_QKV_BF16_KV_FORWARD", False)


def use_mxfp4_force_wo_bf16() -> bool:
    return _mxfp4_bool_env("MXFP4_FORCE_WO_BF16", False)


_MXFP4_LOCALCTA_DGRAD_ROUTE_ID = (
    "mxfp4_fixed_h32_col_localcta_row_sr_dgrad_v2"
)


def use_mxfp4_localcta_dgrad() -> bool:
    """Use localCTA precision only for dgrad in the MXFP4+RHT route.

    This switch is deliberately off by default.  Enabling it must not select
    the process-wide localCTA backend: MXFP4 remains the owner of forward and
    wgrad, while a fused producer emits the localCTA row/weight-column payload
    consumed by one localCTA dgrad GEMM.
    """
    return _mxfp4_bool_env("MXFP4_USE_LOCALCTA_DGRAD", False)


def mxfp4_dgrad_route_identity() -> str:
    return (
        _MXFP4_LOCALCTA_DGRAD_ROUTE_ID
        if use_mxfp4_localcta_dgrad()
        else "mxfp4_native_dgrad"
    )


def _require_mixed_localcta_supported_path(
    path: str,
    supported: bool,
) -> None:
    """Reject native/BF16 fallbacks when the mixed route was requested."""
    if use_mxfp4_localcta_dgrad() and not supported:
        raise RuntimeError(
            f"{_MXFP4_LOCALCTA_DGRAD_ROUTE_ID} has no {path} fallback; "
            "the route must run its sealed fused producer/consumer path"
        )


def _validate_mxfp4_localcta_dgrad_contract(
    *,
    require_runtime: bool = True,
) -> None:
    """Fail closed unless the proven MXFP4+RHT recipe is unchanged.

    The localCTA slice is intentionally narrow: gradient row data-SR and the
    dgrad weight column only.  The deterministic MX column-RHT payload feeding
    wgrad, the exact MX 2D weight row feeding forward, and their RNG behavior
    must remain the successful long-run recipe byte-for-byte.
    """
    if not use_mxfp4_localcta_dgrad():
        return
    errors: list[str] = []
    if mxfp4_backend_version().strip().lower() != "v4":
        errors.append("MXFP4_BACKEND_VERSION must be v4")
    fp4_runtime_root = os.environ.get("FP4_MATMUL_ROOT", "").strip()
    if not fp4_runtime_root or not os.path.isabs(fp4_runtime_root):
        errors.append("FP4_MATMUL_ROOT must be an explicit absolute runtime pin")
    if _mxfp4_bool_env("USE_TK_LOCALCTA", False):
        errors.append("USE_TK_LOCALCTA must remain disabled")
    if os.environ.get("USE_TK_LOCALCTA_VARIANT", "v4").strip().lower() != "v4":
        errors.append("USE_TK_LOCALCTA_VARIANT must be v4")
    localcta_sg_contract = os.environ.get(
        "USE_TK_LOCALCTA_V3_CONTRACT", "outerscale"
    ).strip().lower()
    if localcta_sg_contract not in {"outer", "outerscale"}:
        errors.append(
            "USE_TK_LOCALCTA_V3_CONTRACT must be outer/outerscale"
        )
    if not use_mxfp4_2d_weight_quant():
        errors.append("MXFP4_USE_2D_WEIGHT_QUANT must be enabled")
    if use_mxfp4_qkv_bf16_dgrad() or use_mxfp4_qkv_bf16_wgrad():
        errors.append("QKV BF16 dgrad/wgrad overrides must be disabled")
    if use_mxfp4_qkv_bf16_q_forward() or use_mxfp4_qkv_bf16_kv_forward():
        errors.append("QKV BF16 forward overrides must be disabled")
    if use_mxfp4_force_wo_bf16():
        errors.append("the Wo BF16 override must be disabled")
    if use_mxfp4_wo_attn_layout() or use_mxfp4_wo_nhsd_quant():
        errors.append("the unsupported NHSD/direct Wo route must be disabled")
    if _mxfp4_bool_env("MXFP4_SKIP_FUSED_FFN", False):
        errors.append("MXFP4_SKIP_FUSED_FFN must be disabled")
    if _mxfp4_bool_env("USE_FP4_CONVERT_OUTPUT_HEAD", False):
        errors.append("the output head must remain ordinary BF16")
    for name in (
        "FP4_KEEP_TAIL_BF16_LINEAR_COUNT",
        "FP4_KEEP_LAST_N_LAYERS_BF16",
        "FP4_KEEP_LAST_N_FFNS_BF16",
    ):
        if _mxfp4_int_env(name, 0) != 0:
            errors.append(f"{name} must be zero for the sealed all-layer route")
    if _mxfp4_rht_axes() != "col":
        errors.append("MXFP4_RHT_AXES must be col")
    if _mxfp4_rht_block_size() != 32:
        errors.append("MXFP4_RHT_BLOCK_SIZE must be 32")
    if not _mxfp4_rht_random_sign_mask():
        errors.append(
            "MXFP4_RHT_RANDOM_SIGN_MASK must be enabled; the pinned runtime "
            "implements it as the deterministic 0x2817 sign diagonal"
        )
    for role in ("activation", "grad"):
        if not _mxfp4_rht_for_role(role):
            errors.append(f"MXFP4_RHT_{role.upper()} must be enabled")
        if not _mxfp4_rht_has_col(role) or _mxfp4_rht_has_row(role):
            errors.append(f"{role} RHT must be column-only")
    if _mxfp4_rht_for_role("weight"):
        errors.append("MXFP4_RHT_WEIGHT must be disabled")
    if not _mxfp4_data_sr_for_role("grad") or _mxfp4_grad_sr_axes() != "row":
        errors.append("gradient data SR must be row-only")
    for role in ("activation", "weight"):
        if _mxfp4_data_sr_for_role(role):
            errors.append(f"MXFP4_SR_{role.upper()} must be disabled")
    for role in ("activation", "grad", "weight"):
        if _mxfp4_scale_sr_for_role(role):
            errors.append(f"MXFP4 scale SR must be disabled for {role}")
    if errors:
        raise RuntimeError(
            f"{_MXFP4_LOCALCTA_DGRAD_ROUTE_ID} contract violation: "
            + "; ".join(errors)
        )
    if not require_runtime:
        return
    capabilities = tk_mixed_mx_localcta_quant_capabilities()
    required = {
        "abi_version": 1,
        "grad_coordinate_mode": "explicit_seed_subsequence",
        "grad_mx_col_rht": "block32_fixed_0x2817",
        "mxfp4_rht_block_size": 32,
        "mxfp4_rht_sign_contract": "fixed_0x2817_per_h16_half",
        "grad_localcta_row_sr": True,
        "grad_scale_sr": False,
        "localcta_encode_mode": "encode_centric",
        "weight_mx_2d": True,
        "weight_localcta_2d": True,
        "prepared_outer_sg": True,
        "localcta_sg_contract": "outer",
        "min_alignment": 256,
        "single_bf16_tile_load": True,
        "runtime_advances_rng": False,
        "split2_grad_one_coordinate": True,
        "split2_dgrad_onepass_outer_sg": True,
        "split2_row_outer_sg": "per_arm",
        "split2_layout": (
            "logical_dim1_concat_per_arm_outer_no_bf16_materialization"
        ),
    }
    if not isinstance(capabilities, dict):
        raise RuntimeError(
            f"{_MXFP4_LOCALCTA_DGRAD_ROUTE_ID} fused producer ABI is unavailable"
        )
    mismatched = {
        key: (capabilities.get(key), expected)
        for key, expected in required.items()
        if capabilities.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(
            f"{_MXFP4_LOCALCTA_DGRAD_ROUTE_ID} capability mismatch: {mismatched}"
        )


def use_mxfp4_qkv_combined_bwd(m: int, total_out: int) -> bool:
    value = os.environ.get("MXFP4_USE_QKV_COMBINED_BWD")
    if value is not None:
        return value == "1"
    if m >= 65536:
        return True
    # Keep smaller shapes as explicit recipe opt-ins. The 8B blog-shape runner
    # enables this after a same-GPU 20-step confirm, but other entry points may
    # still have different activation-checkpointing or CCE lifetimes.
    return False


def _use_mxfp4_qkv_split3_grad_fast_for_route(
    mixed_localcta_dgrad: bool | None = None,
) -> bool:
    """Keep the legacy split3 producer unreachable from the mixed route."""
    mixed = (
        use_mxfp4_localcta_dgrad()
        if mixed_localcta_dgrad is None
        else bool(mixed_localcta_dgrad)
    )
    return (
        not mixed
        and use_mxfp4_split3_qkv_quant()
        and _mxfp4_oriented_grad_data_sr("grad") is None
    )


def _mxfp4_data_sr_for_role_raw(role: str) -> bool:
    scoped = os.environ.get(f"MXFP4_SR_{role.upper()}")
    if scoped is not None:
        return scoped == "1"
    if not _mxfp4_bool_env("MXFP4_USE_STOCHASTIC_ROUNDING", False):
        return False
    return role != "grad"


def _mxfp4_scale_sr_for_role_raw(role: str) -> bool:
    scoped = os.environ.get(f"MXFP4_SCALE_SR_{role.upper()}")
    if scoped is not None:
        return scoped == "1"
    if not _mxfp4_bool_env("MXFP4_USE_SCALE_STOCHASTIC_ROUNDING", False):
        return False
    return role != "grad"


def _mxfp4_data_sr_for_role(role: str) -> bool:
    enabled = _mxfp4_data_sr_for_role_raw(role)
    if (
        enabled
        and role == "grad"
        and _mxfp4_grad_random_sign_rht()
        and not _mxfp4_bool_env("MXFP4_ALLOW_UNSAFE_GRAD_RHT_SR", False)
    ):
        # Grad data-SR plus random-sign RHT currently trips an async CUDA launch
        # failure in the grad producer path. Keep activation/weight SR enabled.
        return False
    if enabled and _mxfp4_scale_sr_for_role_raw(role):
        raise RuntimeError(
            f"MXFP4 data SR and scale SR are mutually exclusive for now; both are enabled for role={role}. "
            f"Unset MXFP4_USE_STOCHASTIC_ROUNDING/MXFP4_USE_SCALE_STOCHASTIC_ROUNDING or use "
            f"MXFP4_SR_{role.upper()} / MXFP4_SCALE_SR_{role.upper()}."
        )
    return enabled


def _mxfp4_grad_sr_axes() -> str:
    """Select the gradient orientation receiving data SR.

    This mirrors localCTA's NVFP4_GRAD_SR_AXES contract: row feeds dgrad and
    column feeds wgrad.  MXFP4's legacy combined kernels round both views, so a
    one-axis policy is dispatched through separate row/column producers.
    """

    value = os.environ.get("MXFP4_GRAD_SR_AXES", "both")
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "all": "both",
        "row_col": "both",
        "rowcol": "both",
        "dgrad": "row",
        "wgrad": "col",
        "column": "col",
        "columns": "col",
        "off": "none",
        "0": "none",
    }
    value = aliases.get(value, value)
    if value not in {"none", "row", "col", "both"}:
        raise ValueError(
            f"Unsupported MXFP4_GRAD_SR_AXES={value!r}; "
            "expected none, row, col, or both"
        )
    return value


def _mxfp4_oriented_grad_data_sr(role: str) -> str | None:
    if role != "grad" or not _mxfp4_data_sr_for_role(role):
        return None
    axes = _mxfp4_grad_sr_axes()
    return axes if axes in {"row", "col"} else None


def _mxfp4_scale_sr_for_role(role: str) -> bool:
    enabled = _mxfp4_scale_sr_for_role_raw(role)
    if enabled and _mxfp4_data_sr_for_role_raw(role):
        raise RuntimeError(
            f"MXFP4 data SR and scale SR are mutually exclusive for now; both are enabled for role={role}. "
            f"Unset MXFP4_USE_STOCHASTIC_ROUNDING/MXFP4_USE_SCALE_STOCHASTIC_ROUNDING or use "
            f"MXFP4_SR_{role.upper()} / MXFP4_SCALE_SR_{role.upper()}."
        )
    return enabled


def _mxfp4_rht_for_role(role: str) -> bool:
    scoped = os.environ.get(f"MXFP4_RHT_{role.upper()}")
    if scoped is not None:
        return scoped == "1"
    if not _mxfp4_bool_env("MXFP4_USE_RHT", False):
        return False
    if _mxfp4_bool_env("MXFP4_RHT_TE_STYLE", False) and role == "weight":
        return False
    return role in {"activation", "weight"}


def _mxfp4_rht_axes() -> str:
    default_axes = "col" if _mxfp4_bool_env("MXFP4_RHT_TE_STYLE", False) else "row"
    axes = os.environ.get("MXFP4_RHT_AXES", default_axes).strip().lower().replace("-", "_")
    aliases = {
        "rowcol": "both",
        "row_col": "both",
        "all": "both",
        "cols": "col",
        "columns": "col",
        "rows": "row",
    }
    axes = aliases.get(axes, axes)
    if axes not in {"row", "col", "both"}:
        raise ValueError(f"Unsupported MXFP4_RHT_AXES={axes!r}; expected row, col, or both")
    return axes


def _mxfp4_rht_block_size() -> int:
    try:
        block_size = int(os.environ.get("MXFP4_RHT_BLOCK_SIZE", "32"))
    except ValueError as exc:
        raise ValueError("MXFP4_RHT_BLOCK_SIZE must be an integer") from exc
    if block_size not in {16, 32}:
        raise ValueError("MXFP4_RHT_BLOCK_SIZE must be 16 or 32")
    return block_size


def _mxfp4_rht_random_sign_mask() -> bool:
    # Random signs are coordinate-stable in the backend now, but keep them opt-in
    # while the grad producer overlap path is still being hardened.
    return _mxfp4_bool_env("MXFP4_RHT_RANDOM_SIGN_MASK", False)


def _mxfp4_needs_opt_quant(role: str) -> bool:
    return (
        _mxfp4_rht_for_role(role)
        or _mxfp4_data_sr_for_role(role)
        or _mxfp4_scale_sr_for_role(role)
    )


def _mxfp4_rht_has_row(role: str) -> bool:
    return _mxfp4_rht_for_role(role) and _mxfp4_rht_axes() in {"row", "both"}


def _mxfp4_rht_has_col(role: str) -> bool:
    return _mxfp4_rht_for_role(role) and _mxfp4_rht_axes() in {"col", "both"}


def _mxfp4_grad_random_sign_rht() -> bool:
    return _mxfp4_rht_has_row("grad") and _mxfp4_rht_random_sign_mask()


def _mxfp4_split2_grad_random_sign_safe() -> bool:
    if not _mxfp4_grad_random_sign_rht():
        return True
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        return True
    # The split2 grad-RHT random-sign producer is stable by itself, but racing
    # its outputs against overlapped wgrad/dgrad can fault asynchronously.
    return _mxfp4_bool_env("MXFP4_ALLOW_UNSAFE_RHT_SIGN_OVERLAP", False)


def _mxfp4_rng_seed(*, data_sr: bool, scale_sr: bool) -> int:
    # Activation/weight experimental SR retains its explicit fixed coordinate.
    # Production gradient SR is handled by _mxfp4_opt_kwargs through the
    # checkpointed per-producer allocator below.
    return int(os.environ.get("MXFP4_SR_SEED", "1234"))


def _mxfp4_rng_subsequence(*, data_sr: bool, scale_sr: bool) -> int:
    return int(os.environ.get("MXFP4_SR_SUBSEQUENCE", "0"))


def _mxfp4_opt_kwargs(
    role: str,
    producer_key: str | None = None,
) -> dict[str, object]:
    data_sr = _mxfp4_data_sr_for_role(role)
    scale_sr = _mxfp4_scale_sr_for_role(role)
    if role == "grad" and _mxfp4_oriented_grad_data_sr(role) == "row":
        from .mxfp4_sr_state import reserve_mxfp4_sr

        rng_seed, rng_subsequence = reserve_mxfp4_sr(producer_key)
    else:
        rng_seed = _mxfp4_rng_seed(data_sr=data_sr, scale_sr=scale_sr)
        rng_subsequence = _mxfp4_rng_subsequence(
            data_sr=data_sr,
            scale_sr=scale_sr,
        )
    return {
        "data_stochastic_rounding": data_sr,
        "scale_stochastic_rounding": scale_sr,
        "rng_seed": rng_seed,
        "rng_subsequence": rng_subsequence,
    }


def _mxfp4_grad_producer_key(
    debug_name: str | None,
    operation: str,
) -> str | None:
    if _mxfp4_oriented_grad_data_sr("grad") != "row":
        return None
    from .mxfp4_sr_state import (
        ffn_deriv_grad_key,
        ffn_w2_grad_key,
        linear_grad_key,
        qkv_grad_key,
        wo_grad_key,
    )

    builders = {
        "qkv": qkv_grad_key,
        "wo": wo_grad_key,
        "ffn_w2": ffn_w2_grad_key,
        "ffn_deriv": ffn_deriv_grad_key,
        "linear": linear_grad_key,
    }
    try:
        builder = builders[operation]
    except KeyError as exc:
        raise RuntimeError(f"unknown MXFP4 SR producer operation {operation!r}") from exc
    return builder(debug_name)


def _mxfp4_rht_kwargs(
    role: str,
    producer_key: str | None = None,
) -> dict[str, object]:
    kwargs = _mxfp4_opt_kwargs(role, producer_key)
    kwargs.update(_mxfp4_rht_settings())
    return kwargs


def _mxfp4_rht_settings() -> dict[str, object]:
    return {
        "rht_axes": _mxfp4_rht_axes(),
        "rht_block_size": _mxfp4_rht_block_size(),
        "with_random_sign_mask": _mxfp4_rht_random_sign_mask(),
    }


def _mxfp4_no_sr_opt_kwargs() -> dict[str, object]:
    """Return deterministic opt-producer arguments without reserving SR state.

    A one-axis gradient SR policy deliberately gives the other orientation no
    stochastic rounding.  In particular, TE-style paired column RHT feeds
    wgrad while row SR feeds dgrad.  Building the column arguments through
    ``_mxfp4_opt_kwargs('grad')`` would both enable SR on the wrong view and
    consume a checkpointed row-SR coordinate that no row producer used.
    """

    return {
        "data_stochastic_rounding": False,
        "scale_stochastic_rounding": False,
        "rng_seed": _mxfp4_rng_seed(data_sr=False, scale_sr=False),
        "rng_subsequence": _mxfp4_rng_subsequence(
            data_sr=False,
            scale_sr=False,
        ),
    }


def _mxfp4_no_sr_rht_kwargs(role: str) -> dict[str, object]:
    kwargs = _mxfp4_no_sr_opt_kwargs()
    kwargs.update(_mxfp4_rht_settings())
    return kwargs


def _mxfp4_split_opt_kwargs(
    role: str,
    producer_key: str | None = None,
) -> dict[str, object]:
    kwargs = _mxfp4_opt_kwargs(role, producer_key)
    kwargs.update(
        {
            "use_rht": _mxfp4_rht_has_col(role),
            "rht_block_size": _mxfp4_rht_block_size(),
            "with_random_sign_mask": _mxfp4_rht_random_sign_mask(),
        }
    )
    if _mxfp4_rht_has_row(role):
        kwargs["row_with_rht"] = True
    return kwargs


def _mxfp4_split_row_only_opt_kwargs(
    role: str,
    producer_key: str | None = None,
) -> dict[str, object]:
    kwargs = _mxfp4_opt_kwargs(role, producer_key)
    kwargs.update(
        {
            "use_rht": _mxfp4_rht_has_row(role),
            "rht_block_size": _mxfp4_rht_block_size(),
            "with_random_sign_mask": _mxfp4_rht_random_sign_mask(),
        }
    )
    return kwargs


def _mxfp4_split_col_only_opt_kwargs(
    role: str,
    producer_key: str | None = None,
) -> dict[str, object]:
    kwargs = _mxfp4_opt_kwargs(role, producer_key)
    kwargs.update(
        {
            "use_rht": _mxfp4_rht_has_col(role),
            "rht_block_size": _mxfp4_rht_block_size(),
            "with_random_sign_mask": _mxfp4_rht_random_sign_mask(),
        }
    )
    return kwargs


def _mxfp4_split_axis_only_no_sr_rht_kwargs(
    role: str,
    axis: str,
) -> dict[str, object]:
    if axis not in {"row", "col"}:
        raise ValueError(f"unsupported MXFP4 split orientation {axis!r}")
    kwargs = _mxfp4_no_sr_opt_kwargs()
    kwargs.update(
        {
            "use_rht": (
                _mxfp4_rht_has_row(role)
                if axis == "row"
                else _mxfp4_rht_has_col(role)
            ),
            "rht_block_size": _mxfp4_rht_block_size(),
            "with_random_sign_mask": _mxfp4_rht_random_sign_mask(),
        }
    )
    return kwargs


def use_mxfp4_split2_ffn_quant() -> bool:
    return os.environ.get(
        "MXFP4_USE_SPLIT2_FFN_QUANT",
        "1" if _mxfp4_v4_quant_extension() else "0",
    ) == "1"


def use_mxfp4_fused_silu_deriv_split2_ffn() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN", "0") == "1"


def use_mxfp4_fused_row_producer_split2_ffn() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_ROW_PRODUCER_SPLIT2_FFN", "0") == "1"


def use_mxfp4_fused_row_producer_tile_split2_ffn() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_ROW_PRODUCER_TILE_SPLIT2_FFN", "0") == "1"


def use_mxfp4_fused_silu_ffn_quant() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_SILU_FFN_QUANT", "0") == "1"


def use_mxfp4_saved_sigmoid_ffn() -> bool:
    return os.environ.get("MXFP4_USE_SAVED_SIGMOID_FFN", "1") == "1"


def use_mxfp4_saved_sigmoid_fwd_inplace_quant() -> bool:
    return os.environ.get("MXFP4_USE_SAVED_SIGMOID_FWD_INPLACE_QUANT", "0") == "1"


def use_mxfp4_saved_sigmoid_fused_fwd_quant() -> bool:
    return os.environ.get("MXFP4_USE_SAVED_SIGMOID_FUSED_FWD_QUANT", "0") == "1"


def use_mxfp4_saved_sigmoid_split2_row_overlap() -> bool:
    return os.environ.get("MXFP4_USE_SAVED_SIGMOID_SPLIT2_ROW_OVERLAP", "0") == "1"


def use_mxfp4_saved_sigmoid_row_producer_split2_ffn() -> bool:
    return os.environ.get("MXFP4_USE_SAVED_SIGMOID_ROW_PRODUCER_SPLIT2_FFN", "0") == "1"


def use_mxfp4_saved_sigmoid_fused_split2_ffn() -> bool:
    return os.environ.get("MXFP4_USE_SAVED_SIGMOID_FUSED_SPLIT2_FFN", "1") == "1"


def use_mxfp4_recompute_ffn_w13() -> bool:
    return os.environ.get("MXFP4_RECOMPUTE_FFN_W13", "0") == "1"


def use_mxfp4_packed_w13_ffn() -> bool:
    # Experimental: saves a pair of BF16 W13 allocations and enables strided
    # SiLU/deriv quant producers, but current 8B A/Bs do not show an MFU win.
    return os.environ.get("MXFP4_USE_PACKED_W13_FFN", "0") == "1"


def use_mxfp4_fused_silu_ffn_quant_rht() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_SILU_FFN_QUANT_RHT", "0") == "1"


def use_mxfp4_fused_silu_ffn_quant_data_sr() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_SILU_FFN_QUANT_DATA_SR", "0") == "1"


def use_mxfp4_fused_silu_ffn_quant_scale_sr() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_SILU_FFN_QUANT_SCALE_SR", "0") == "1"


def use_mxfp4_deepseek_grouped_fused_silu_quant() -> bool:
    return os.environ.get(
        "MXFP4_DEEPSEEK_GROUPED_FUSED_SILU_QUANT",
        "1" if _mxfp4_v4_quant_extension() else "0",
    ) == "1"


def use_mxfp4_fused_sqrelu_quant() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_SQRELU_QUANT", "0") == "1"


def use_mxfp4_fused_sqrelu_deriv_quant() -> bool:
    return os.environ.get("MXFP4_USE_FUSED_SQRELU_DERIV_QUANT", "0") == "1"


def use_mxfp4_simple_sqrelu_fused_w2() -> bool:
    return os.environ.get("MXFP4_USE_SIMPLE_SQRELU_FUSED_W2", "1") == "1"


def use_mxfp4_sqrelu_fused_rms_w1() -> bool:
    return os.environ.get("MXFP4_USE_SQRELU_FUSED_RMS_W1", "0") == "1"


def use_mxfp4_sqrelu_deriv_gemm_epilogue() -> bool:
    return os.environ.get("MXFP4_USE_SQRELU_DERIV_GEMM_EPILOGUE", "0") == "1"


def use_mxfp4_sqrelu_deriv_rht_sr() -> bool:
    raw = os.environ.get("MXFP4_USE_SQRELU_DERIV_RHT_SR")
    if raw is not None:
        return _mxfp4_bool_env("MXFP4_USE_SQRELU_DERIV_RHT_SR", True)
    return _mxfp4_bool_env("MXFP4_USE_UNSAFE_SQRELU_DERIV_RHT_SR", False)


def use_mxfp4_sqrelu_split_col_overlap() -> bool:
    return os.environ.get(
        "MXFP4_USE_SQRELU_SPLIT_COL_OVERLAP",
        "0",
    ) == "1"


def use_mxfp4_sqrelu_w2_wgrad_overlap() -> bool:
    return os.environ.get("MXFP4_USE_SQRELU_W2_WGRAD_OVERLAP", "0") == "1"


def use_mxfp4_sqrelu_w2_wgrad_after_dgrad_overlap() -> bool:
    return os.environ.get("MXFP4_USE_SQRELU_W2_WGRAD_AFTER_DGRAD_OVERLAP", "0") == "1"


def use_mxfp4_residual_fusion() -> bool:
    return os.environ.get("MXFP4_USE_RESIDUAL_FUSION", "0") == "1"


def _mxfp4_attn_residual_requested() -> bool:
    scoped = os.environ.get("MXFP4_USE_RESIDUAL_FUSION_ATTN")
    if scoped is not None:
        return scoped == "1"
    return use_mxfp4_residual_fusion()


def _mxfp4_ffn_residual_requested() -> bool:
    scoped = os.environ.get("MXFP4_USE_RESIDUAL_FUSION_FFN")
    if scoped is not None:
        return scoped == "1"
    return use_mxfp4_residual_fusion()


def _mxfp4_unsafe_residual_fallback() -> str:
    # The combined attn+FFN residual epilogue can race with backward overlap
    # under the default multi-connection scheduler. Prefer the FFN residual
    # epilogue because it is the stronger stable end-to-end path.
    return os.environ.get("MXFP4_UNSAFE_RESIDUAL_FALLBACK", "prefer_ffn").strip().lower()


def _mxfp4_attn_ffn_residual_overlap_safe() -> bool:
    if os.environ.get("MXFP4_ALLOW_UNSAFE_ATTN_FFN_RESIDUAL_OVERLAP", "0") == "1":
        return True
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        return True
    if not use_mxfp4_bwd_wgrad_overlap():
        return True
    return False


def use_mxfp4_residual_fusion_attn() -> bool:
    if not _mxfp4_attn_residual_requested():
        return False
    if _mxfp4_ffn_residual_requested() and not _mxfp4_attn_ffn_residual_overlap_safe():
        return _mxfp4_unsafe_residual_fallback() == "prefer_attn"
    return True


def use_mxfp4_residual_fusion_ffn() -> bool:
    if not _mxfp4_ffn_residual_requested():
        return False
    if _mxfp4_attn_residual_requested() and not _mxfp4_attn_ffn_residual_overlap_safe():
        return _mxfp4_unsafe_residual_fallback() == "prefer_ffn"
    return True


def use_mxfp4_split2_ffn_onepass_dgrad() -> bool:
    return os.environ.get("MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD", "1") == "1"


def use_mxfp4_split2_ffn_inplace_quant() -> bool:
    return os.environ.get("MXFP4_USE_SPLIT2_FFN_INPLACE_QUANT", "1") == "1"


def use_mxfp4_split2_ffn_row_overlap() -> bool:
    return os.environ.get("MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP", "0") == "1"


def use_mxfp4_split2_ffn_producer_split() -> bool:
    return os.environ.get("MXFP4_USE_SPLIT2_FFN_PRODUCER_SPLIT", "0") == "1"


def use_mxfp4_bwd_wgrad_overlap() -> bool:
    override = os.environ.get("MXFP4_USE_BWD_WGRAD_OVERLAP")
    if override is not None:
        return override == "1"
    # The global overlap path is useful as an explicit experiment, but it
    # collapses the 1.2B high-water route to ~60 MFU by disabling the tuned
    # residual/GEMM schedule and contending with the FFN/QKV producer streams.
    return False


def use_mxfp4_qkv_wgrad_overlap() -> bool:
    scoped = os.environ.get("MXFP4_USE_QKV_WGRAD_OVERLAP")
    if scoped is not None:
        return scoped == "1"
    return use_mxfp4_bwd_wgrad_overlap()


def use_mxfp4_qkv_wgrad_wait_before_rmsnorm() -> bool:
    return os.environ.get("MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM", "0") == "1"


def use_mxfp4_qkv_wgrad_wait_before_rmsnorm_dgamma() -> bool:
    return os.environ.get("MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM_DGAMMA", "0") == "1"


def use_mxfp4_qkv_fwd_weight_quant_overlap() -> bool:
    return os.environ.get("MXFP4_USE_QKV_FWD_WEIGHT_QUANT_OVERLAP", "0") == "1"


def use_mxfp4_ffn_fwd_w2_weight_quant_overlap() -> bool:
    return os.environ.get("MXFP4_USE_FFN_FWD_W2_WEIGHT_QUANT_OVERLAP", "0") == "1"


def use_mxfp4_ffn_fwd_w13_weight_quant_overlap() -> bool:
    return os.environ.get("MXFP4_USE_FFN_FWD_W13_WEIGHT_QUANT_OVERLAP", "0") == "1"


def use_mxfp4_ffn_wgrad_overlap() -> bool:
    scoped = os.environ.get("MXFP4_USE_FFN_WGRAD_OVERLAP")
    if scoped is not None:
        return scoped == "1"
    return use_mxfp4_bwd_wgrad_overlap()


def use_mxfp4_ffn_w2_wgrad_overlap() -> bool:
    scoped = os.environ.get("MXFP4_USE_FFN_W2_WGRAD_OVERLAP")
    if scoped is not None:
        return scoped == "1"
    return use_mxfp4_ffn_wgrad_overlap()


def use_mxfp4_ffn_w13_wgrad_overlap() -> bool:
    scoped = os.environ.get("MXFP4_USE_FFN_W13_WGRAD_OVERLAP")
    if scoped is not None:
        return scoped == "1"
    return use_mxfp4_ffn_wgrad_overlap()


def use_mxfp4_async_rmsnorm_bwd() -> bool:
    return os.environ.get("MXFP4_USE_ASYNC_RMSNORM_BWD", "1") == "1"


def use_mxfp4_async_rmsnorm_bwd_single_out() -> bool:
    return os.environ.get("MXFP4_USE_ASYNC_RMSNORM_BWD_SINGLE_OUT", "1") == "1"


def use_mxfp4_qkv_async_rmsnorm_bwd() -> bool:
    return os.environ.get("MXFP4_USE_QKV_ASYNC_RMSNORM_BWD", "1") == "1"


def use_mxfp4_rms_bwd_split_dgamma() -> bool:
    return os.environ.get("MXFP4_USE_RMS_BWD_SPLIT_DGAMMA", "1") == "1"


def mxfp4_ffn_wgrad_overlap_min_m() -> int:
    try:
        return int(os.environ.get("MXFP4_FFN_WGRAD_OVERLAP_MIN_M", "65536"))
    except ValueError:
        return 65536


def mxfp4_split2_ffn_onepass_config_idx() -> int:
    try:
        idx = int(os.environ.get("MXFP4_SPLIT2_FFN_ONEPASS_CONFIG_IDX", "-1"))
    except ValueError:
        return -1
    if idx not in (-1, 1, 3, 5):
        raise ValueError(
            "MXFP4_SPLIT2_FFN_ONEPASS_CONFIG_IDX must be one of -1, 1, 3, or 5; "
            f"got {idx}"
        )
    return idx


def mxfp4_split3_qkv_onepass_config_idx() -> int:
    try:
        return int(os.environ.get("MXFP4_SPLIT3_QKV_ONEPASS_CONFIG_IDX", "-1"))
    except ValueError:
        return -1


def use_mxfp4_bwd_state_cache() -> bool:
    return os.environ.get("MXFP4_USE_BWD_STATE_CACHE", "0") == "1"


def use_mxfp4_lazy_ffn_bwd_state() -> bool:
    return os.environ.get("MXFP4_USE_LAZY_FFN_BWD_STATE", "0") == "1"


def use_mxfp4_stage_timing() -> bool:
    return os.environ.get("USE_MXFP4_STAGE_TIMING", "0") == "1"


def use_mxfp4_stage_timing_sync() -> bool:
    return os.environ.get("USE_MXFP4_STAGE_TIMING_SYNC", "0") == "1"


def use_mxfp4_stage_timing_quiet() -> bool:
    return os.environ.get("MXFP4_STAGE_TRACE_QUIET", "0") == "1"


def _mxfp4_record_stage_completion(stage: str, name: str | None) -> None:
    """Keep a lagged event trail so asynchronous launch failures retain context."""
    if not _MXFP4_FLIGHT_RECORDER_ENABLED:
        return
    if (
        _MXFP4_FLIGHT_RECORDER_STAGES is not None
        and stage not in _MXFP4_FLIGHT_RECORDER_STAGES
    ):
        return

    device_index = torch.cuda.current_device()
    label = name or stage
    try:
        stream = torch.cuda.current_stream(device_index)
        event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
    except Exception:
        _mxfp4_report_flight_failure(
            device_index,
            "event_record",
            enqueue=(stage, label),
        )
        raise

    with _MXFP4_FLIGHT_RECORDER_LOCK:
        sequence = _MXFP4_FLIGHT_RECORDER_SEQ_BY_DEVICE.get(device_index, 0) + 1
        _MXFP4_FLIGHT_RECORDER_SEQ_BY_DEVICE[device_index] = sequence
        pending = _MXFP4_FLIGHT_RECORDER_BY_DEVICE.setdefault(
            device_index,
            deque(),
        )
        pending.append((sequence, stage, label, event))
        try:
            while (
                len(pending) > _MXFP4_FLIGHT_RECORDER_LAG
                and pending[0][3].query()
            ):
                completed_sequence, completed_stage, completed_label, _ = (
                    pending.popleft()
                )
                _MXFP4_FLIGHT_RECORDER_LAST_COMPLETED_BY_DEVICE[device_index] = (
                    completed_sequence,
                    completed_stage,
                    completed_label,
                )
        except Exception:
            _mxfp4_report_flight_failure_locked(
                device_index,
                "event_query",
                enqueue=(sequence, stage, label),
            )
            raise


def _mxfp4_report_flight_failure_locked(
    device_index: int,
    context: str,
    *,
    enqueue: tuple[object, ...] | None = None,
) -> None:
    pending = _MXFP4_FLIGHT_RECORDER_BY_DEVICE.get(device_index)
    pending_context = list(pending)[-16:] if pending else []
    recent = [entry[:3] for entry in pending_context]
    oldest = pending[0][:3] if pending else None
    last_completed = _MXFP4_FLIGHT_RECORDER_LAST_COMPLETED_BY_DEVICE.get(
        device_index
    )
    print(
        "[MXFP4 FLIGHT FAILURE] "
        f"context={context} device={device_index} enqueue={enqueue} "
        f"last_completed={last_completed} oldest_pending={oldest} "
        f"pending_events={len(pending) if pending else 0} recent={recent}",
        file=sys.stderr,
        flush=True,
    )


def _mxfp4_report_flight_failure(
    device_index: int,
    context: str,
    *,
    enqueue: tuple[object, ...] | None = None,
) -> None:
    if not _MXFP4_FLIGHT_RECORDER_ENABLED:
        return
    with _MXFP4_FLIGHT_RECORDER_LOCK:
        _mxfp4_report_flight_failure_locked(
            device_index,
            context,
            enqueue=enqueue,
        )


def _prewarm_mxfp4_cudnn_sdpa_autograd_handle(device: torch.device) -> None:
    """Create the thread-local cuDNN handle before checkpoint recomputation."""
    if os.environ.get("MXFP4_PREWARM_CUDNN_SDPA_AUTOGRAD", "0") != "1":
        return

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    warmed_devices = getattr(
        _MXFP4_CUDNN_SDPA_AUTOGRAD_WARMUP,
        "devices",
        None,
    )
    if warmed_devices is not None and device_index in warmed_devices:
        return

    from torch.nn.attention import SDPBackend, sdpa_kernel

    start = time.perf_counter()
    with torch.cuda.device(device_index):
        # Non-reentrant checkpoint recomputation runs on the CUDA autograd
        # worker. cuDNN handles are thread-local, so the handle created during
        # the original forward is not visible here. Initializing it while
        # earlier work is still in flight can wedge the GB200 CUDA 13 driver.
        torch.cuda.synchronize(device_index)
        with torch.inference_mode():
            q = torch.empty(
                (1, 1, 128, 128),
                dtype=torch.bfloat16,
                device=device_index,
            )
            with sdpa_kernel(
                [SDPBackend.CUDNN_ATTENTION],
                set_priority=True,
            ):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize(device_index)

    if warmed_devices is None:
        warmed_devices = set()
        _MXFP4_CUDNN_SDPA_AUTOGRAD_WARMUP.devices = warmed_devices
    warmed_devices.add(device_index)
    print(
        "[MXFP4] prewarmed cuDNN SDPA on the CUDA autograd worker "
        f"device={device_index} elapsed_ms="
        f"{(time.perf_counter() - start) * 1000.0:.3f}",
        file=sys.stderr,
        flush=True,
    )


def use_mxfp4_w2_dgrad_silu_producer() -> bool:
    # The native producer launches on isolated 8B-shape smokes, but still
    # fails inside real training backward. Require an explicit unsafe gate so
    # old benchmark envs cannot silently take the crashing path.
    return (
        os.environ.get("MXFP4_USE_W2_DGRAD_SILU_PRODUCER", "0") == "1"
        and os.environ.get("MXFP4_ALLOW_UNSAFE_W2_DGRAD_SILU_PRODUCER", "0") == "1"
    )


def use_mxfp4_w2_dgrad_saved_sigmoid_producer() -> bool:
    # Same producer family as MXFP4_USE_W2_DGRAD_SILU_PRODUCER: it is useful
    # for isolated diagnostics, but has produced launch failures in full
    # Megatron Bridge backward. Keep it behind an explicit unsafe gate.
    return (
        os.environ.get("MXFP4_USE_W2_DGRAD_SAVED_SIGMOID_PRODUCER", "0") == "1"
        and os.environ.get("MXFP4_ALLOW_UNSAFE_W2_DGRAD_SAVED_SIGMOID_PRODUCER", "0") == "1"
    )


def use_mxfp4_w2_dgrad_saved_sigmoid_row_bf16_producer() -> bool:
    return (
        os.environ.get("MXFP4_USE_W2_DGRAD_SAVED_SIGMOID_ROW_BF16_PRODUCER", "0") == "1"
        and os.environ.get("MXFP4_ALLOW_UNSAFE_W2_DGRAD_SAVED_SIGMOID_ROW_BF16_PRODUCER", "0") == "1"
    )


def mxfp4_w2_dgrad_silu_producer_config_id() -> int:
    return _mxfp4_int_env("MXFP4_W2_DGRAD_SILU_PRODUCER_CONFIG_ID", 4)


def mxfp4_w2_dgrad_saved_sigmoid_producer_config_id() -> int:
    return _mxfp4_int_env("MXFP4_W2_DGRAD_SAVED_SIGMOID_PRODUCER_CONFIG_ID", 44)


def mxfp4_w2_dgrad_saved_sigmoid_row_bf16_producer_config_id() -> int:
    return _mxfp4_int_env("MXFP4_W2_DGRAD_SAVED_SIGMOID_ROW_BF16_PRODUCER_CONFIG_ID", 44)


def use_mxfp4_linear_gemm_configs() -> bool:
    return os.environ.get("MXFP4_USE_LINEAR_GEMM_CONFIGS", "1") == "1"


def use_mxfp4_wgrad_gemm_configs() -> bool:
    return os.environ.get("MXFP4_USE_WGRAD_GEMM_CONFIGS", "1") == "1"


def _mxfp4_gemm_config_override(prefix: str, m: int, n: int, k: int, default: int | None) -> int | None:
    default_value = default if default is not None else -1
    for key in (
        f"{prefix}_M{m}_N{n}_K{k}",
        f"{prefix}_M{m}_N{n}",
        f"{prefix}_ID",
    ):
        if key in os.environ:
            value = _mxfp4_int_env(key, default_value)
            return None if value < 0 else value
    return default


def _mxfp4_batched_gemm_config_override(
    prefix: str,
    out_list: list[torch.Tensor],
    a_list: list[torch.Tensor],
    default: int | None,
) -> int | None:
    if not out_list or not a_list:
        return default
    M = int(out_list[0].size(0))
    N = int(out_list[0].size(1))
    K = int(a_list[0].size(1)) * 2
    return _mxfp4_gemm_config_override(prefix, M, N, K, default)


def _mxfp4_batched_gemm_configured(
    A_list: list[torch.Tensor],
    A_sc_list: list[torch.Tensor],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    D_out_list: list[torch.Tensor],
    *,
    prefix: str,
    default: int | None = None,
) -> None:
    config_id = _mxfp4_batched_gemm_config_override(prefix, D_out_list, A_list, default)
    if config_id is None:
        mxfp4_batched_gemm(A_list, A_sc_list, B_list, B_sc_list, D_out_list)
        return
    mxfp4_batched_gemm_config(A_list, A_sc_list, B_list, B_sc_list, D_out_list, config_id)


def use_mxfp4_linear_residual_config() -> bool:
    return os.environ.get("MXFP4_USE_LINEAR_RESIDUAL_CONFIG", "1") == "1"


def use_mxfp4_wo_attn_layout() -> bool:
    return _mxfp4_bool_env("MXFP4_USE_WO_ATTN_LAYOUT", False)


def use_mxfp4_wo_nhsd_quant() -> bool:
    value = os.environ.get("MXFP4_USE_WO_NHSD_QUANT")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return use_mxfp4_wo_attn_layout()


def _nhsd_attention_output_matrix_view(
    input: torch.Tensor,
    B: int,
    H: int,
    S: int,
    D: int,
) -> torch.Tensor | None:
    if input.storage_offset() != 0:
        return None
    expected = (H * S * D, D, H * D, 1)
    if tuple(input.stride()) != expected:
        return None
    return input.as_strided((B * S, H * D), (H * D, 1))


def _mxfp4_stage_begin(stage: str, name: str | None) -> float | None:
    if not use_mxfp4_stage_timing():
        return None
    active_step = os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip()
    step_filter = (
        os.environ.get("MXFP4_STAGE_TRACE_STEP", "").strip()
        or os.environ.get("TK_STAGE_TRACE_STEP", "").strip()
    )
    if step_filter and active_step != step_filter:
        return None
    stage_filter = os.environ.get("MXFP4_STAGE_TRACE_STAGE_FILTER", "").strip()
    if stage_filter and stage_filter not in stage:
        return None
    prefix = f"[MXFP4 TRACE step={active_step}]" if active_step else "[MXFP4 TRACE]"
    label = name or stage
    name_filter = os.environ.get("MXFP4_STAGE_TRACE_FILTER", "").strip()
    if name_filter and name_filter not in label:
        return None
    if use_mxfp4_stage_timing_sync():
        torch.cuda.synchronize()
    start = time.perf_counter()
    if not use_mxfp4_stage_timing_quiet():
        print(f"{prefix} {stage} start {label}", file=sys.stderr, flush=True)
    return start


def _mxfp4_stage_end(stage: str, name: str | None, start: float | None) -> None:
    _mxfp4_record_stage_completion(stage, name)
    if start is None:
        return
    if use_mxfp4_stage_timing_sync():
        torch.cuda.synchronize()
    active_step = os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip()
    prefix = f"[MXFP4 TRACE step={active_step}]" if active_step else "[MXFP4 TRACE]"
    label = name or stage
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not use_mxfp4_stage_timing_quiet():
        print(
            f"{prefix} {stage} end {label} elapsed_ms={elapsed_ms:.3f}",
            file=sys.stderr,
            flush=True,
        )


@dataclass
class _MXFP4RowCol:
    row_fp4: torch.Tensor
    row_sc: torch.Tensor
    col_fp4: torch.Tensor
    col_sc: torch.Tensor


@dataclass
class _MixedMXLocalCTAGradCarrier:
    """One fused dY producer: localCTA row plus deterministic MX RHT column."""

    local_row_fp4: torch.Tensor
    local_row_sc: torch.Tensor
    local_row_sg: torch.Tensor
    mx_col_fp4: torch.Tensor
    mx_col_sc: torch.Tensor
    shape: tuple[int, int]
    keepalive: tuple[torch.Tensor, ...] = ()

    # Column aliases let the established MX wgrad code consume exactly the
    # same payload without route-specific branches.
    @property
    def col_fp4(self) -> torch.Tensor:
        return self.mx_col_fp4

    @property
    def col_sc(self) -> torch.Tensor:
        return self.mx_col_sc


@dataclass
class _MixedMXLocalCTASplit2GradCarrier:
    """One logical split2 producer with independent localCTA arm scales."""

    local_row_fp4: torch.Tensor
    local_row_sc: torch.Tensor
    local_row_sg0: torch.Tensor
    local_row_sg1: torch.Tensor
    mx_col_fp4: torch.Tensor
    mx_col_sc: torch.Tensor
    shape: tuple[int, int]
    keepalive: tuple[torch.Tensor, ...] = ()

    @property
    def col_fp4(self) -> torch.Tensor:
        return self.mx_col_fp4

    @property
    def col_sc(self) -> torch.Tensor:
        return self.mx_col_sc


@dataclass
class _MixedMXLocalCTAWeightCarrier:
    """One fused weight producer: MX 2D row plus localCTA 2D column."""

    mx_row_fp4: torch.Tensor
    mx_row_sc: torch.Tensor
    local_col_fp4: torch.Tensor
    local_col_sc: torch.Tensor
    local_col_sg: torch.Tensor
    shape: tuple[int, int]
    keepalive: tuple[torch.Tensor, ...] = ()

    # Row aliases preserve all existing MX forward call sites.
    @property
    def row_fp4(self) -> torch.Tensor:
        return self.mx_row_fp4

    @property
    def row_sc(self) -> torch.Tensor:
        return self.mx_row_sc


def _mxfp4_weight_backward_col(
    weight,
    *,
    mixed_localcta_dgrad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a backward weight column without hiding its format."""
    if mixed_localcta_dgrad:
        if not isinstance(weight, _MixedMXLocalCTAWeightCarrier):
            raise RuntimeError("mixed route received a non-mixed weight carrier")
        return weight.local_col_fp4, weight.local_col_sc
    if isinstance(weight, _MixedMXLocalCTAWeightCarrier):
        raise RuntimeError("native MX route received a mixed weight carrier")
    return weight.col_fp4, weight.col_sc


def _mixed_mx_localcta_alloc_result(
    result,
    *,
    path: str,
    expected_tensors: int = 6,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(result, (tuple, list)) or len(result) != expected_tensors:
        raise RuntimeError(
            f"{path} allocator must return exactly {expected_tensors} tensors"
        )
    tensors = tuple(result)
    if not all(torch.is_tensor(value) for value in tensors):
        raise RuntimeError(f"{path} allocator returned a non-tensor payload")
    return tensors


def _quantize_mixed_grad_dy_bf16(
    tensor: torch.Tensor,
    *,
    producer_key: str | None,
) -> _MixedMXLocalCTAGradCarrier:
    """Fuse localCTA row-SR with MX column-RHT for one gradient matrix.

    The existing checkpointed MX logical producer owns the stochastic
    coordinate.  We reserve it exactly once and hand that coordinate to the
    localCTA row producer; the deterministic MX column consumes no coordinate.
    This keeps resume and a later revert to native MX row-SR continuous.
    """
    _validate_mxfp4_localcta_dgrad_contract()
    value = _as_contiguous_bf16(tensor)
    if (
        value.dim() != 2
        or value.shape[0] % 256
        or value.shape[1] % 256
    ):
        raise RuntimeError(
            "mixed MX/localCTA gradient producer requires an aligned 2D BF16 "
            f"matrix, got shape={tuple(value.shape)} dtype={value.dtype}"
        )
    module = _get_tk_mixed_mx_localcta_quant()
    if module is None:
        raise RuntimeError("mixed MX/localCTA quantizer module is unavailable")
    alloc = getattr(
        module,
        "tk_mixed_grad_localcta_row_mx_col_alloc",
        None,
    )
    launch = getattr(
        module,
        "tk_mixed_grad_localcta_row_mx_col_launch_inplace",
        None,
    )
    if alloc is None or launch is None:
        raise RuntimeError("mixed gradient fused-producer ABI is incomplete")
    buffers = _mixed_mx_localcta_alloc_result(
        alloc(int(value.shape[0]), int(value.shape[1]), value.device),
        path="mixed gradient",
    )
    # This is the only SR reservation in the mixed producer.  Do not create a
    # LocalCTASRState and do not reserve a shadow MX coordinate.
    opt = _mxfp4_opt_kwargs("grad", producer_key)
    launch(
        value,
        *buffers,
        int(opt["rng_seed"]),
        int(opt["rng_subsequence"]),
    )
    return _MixedMXLocalCTAGradCarrier(
        local_row_fp4=buffers[0],
        local_row_sc=buffers[1],
        local_row_sg=buffers[2],
        mx_col_fp4=buffers[3],
        mx_col_sc=buffers[4],
        shape=(int(value.shape[0]), int(value.shape[1])),
        keepalive=tuple(buffers[5:]),
    )


def _quantize_mixed_split2_grad_bf16(
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    *,
    producer_key: str | None,
) -> _MixedMXLocalCTASplit2GradCarrier:
    """Fuse the two FFN derivative arms under one logical SR coordinate.

    The runtime reads ``grad0`` and ``grad1`` independently, but lays them out
    as one logical ``[grad0|grad1]`` matrix.  In particular, it must not build
    a BF16 concatenation and it must not advance a private RNG counter.
    """
    _validate_mxfp4_localcta_dgrad_contract()
    value0 = _as_contiguous_bf16(grad0)
    value1 = _as_contiguous_bf16(grad1)
    if value0.shape != value1.shape:
        raise RuntimeError(
            "mixed split2 gradient arms must have identical shapes, got "
            f"{tuple(value0.shape)} and {tuple(value1.shape)}"
        )
    if (
        value0.dim() != 2
        or value0.shape[0] % 256
        or value0.shape[1] % 256
    ):
        raise RuntimeError(
            "mixed split2 gradient producer requires two aligned 2D BF16 "
            f"matrices, got shape={tuple(value0.shape)} dtype={value0.dtype}"
        )
    module = _get_tk_mixed_mx_localcta_quant()
    if module is None:
        raise RuntimeError("mixed MX/localCTA quantizer module is unavailable")
    alloc = getattr(
        module,
        "tk_mixed_split2_grad_localcta_row_mx_col_alloc",
        None,
    )
    launch = getattr(
        module,
        "tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace",
        None,
    )
    if alloc is None or launch is None:
        raise RuntimeError("mixed split2 gradient fused-producer ABI is incomplete")
    m, h = (int(value0.shape[0]), int(value0.shape[1]))
    buffers = _mixed_mx_localcta_alloc_result(
        alloc(m, h, value0.device),
        path="mixed split2 gradient",
        expected_tensors=7,
    )
    # One logical [grad0|grad1] producer owns one checkpointed coordinate.
    opt = _mxfp4_opt_kwargs("grad", producer_key)
    launch(
        value0,
        value1,
        *buffers,
        int(opt["rng_seed"]),
        int(opt["rng_subsequence"]),
    )
    return _MixedMXLocalCTASplit2GradCarrier(
        local_row_fp4=buffers[0],
        local_row_sc=buffers[1],
        local_row_sg0=buffers[2],
        local_row_sg1=buffers[3],
        mx_col_fp4=buffers[4],
        mx_col_sc=buffers[5],
        shape=(m, 2 * h),
        keepalive=tuple(buffers[6:]),
    )


def _quantize_mixed_weight_bf16(
    tensor: torch.Tensor,
) -> _MixedMXLocalCTAWeightCarrier:
    """Fuse the exact MX 2D forward row with the localCTA 2D dgrad column."""
    _validate_mxfp4_localcta_dgrad_contract()
    value = _as_contiguous_bf16(tensor)
    if (
        value.dim() != 2
        or value.shape[0] % 256
        or value.shape[1] % 256
    ):
        raise RuntimeError(
            "mixed MX/localCTA weight producer requires an aligned 2D BF16 "
            f"matrix, got shape={tuple(value.shape)} dtype={value.dtype}"
        )
    module = _get_tk_mixed_mx_localcta_quant()
    if module is None:
        raise RuntimeError("mixed MX/localCTA quantizer module is unavailable")
    alloc = getattr(module, "tk_mixed_weight_mx_row_localcta_col_alloc", None)
    launch = getattr(
        module,
        "tk_mixed_weight_mx_row_localcta_col_launch_inplace",
        None,
    )
    if alloc is None or launch is None:
        raise RuntimeError("mixed weight fused-producer ABI is incomplete")
    buffers = _mixed_mx_localcta_alloc_result(
        alloc(int(value.shape[0]), int(value.shape[1]), value.device),
        path="mixed weight",
    )
    launch(value, *buffers)
    return _MixedMXLocalCTAWeightCarrier(
        mx_row_fp4=buffers[0],
        mx_row_sc=buffers[1],
        local_col_fp4=buffers[2],
        local_col_sc=buffers[3],
        local_col_sg=buffers[4],
        shape=(int(value.shape[0]), int(value.shape[1])),
        keepalive=tuple(buffers[5:]),
    )


def _mixed_localcta_dgrad(
    grad: _MixedMXLocalCTAGradCarrier,
    weight: _MixedMXLocalCTAWeightCarrier,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if grad.shape[1] != weight.shape[0]:
        raise RuntimeError(
            "mixed localCTA dgrad contraction mismatch: "
            f"grad={grad.shape}, weight={weight.shape}"
        )
    return tk_mixed_localcta_dgrad(
        grad.local_row_fp4,
        grad.local_row_sc,
        grad.local_row_sg,
        weight.local_col_fp4,
        weight.local_col_sc,
        weight.local_col_sg,
        out,
    )


def _mixed_localcta_split2_dgrad(
    grad: _MixedMXLocalCTASplit2GradCarrier,
    weight0: _MixedMXLocalCTAWeightCarrier,
    weight1: _MixedMXLocalCTAWeightCarrier,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if weight0.shape != weight1.shape:
        raise RuntimeError(
            "mixed localCTA split2 weights must have identical shapes, got "
            f"{weight0.shape} and {weight1.shape}"
        )
    if grad.shape[1] != 2 * weight0.shape[0]:
        raise RuntimeError(
            "mixed localCTA split2 contraction mismatch: "
            f"grad={grad.shape}, weights={weight0.shape}"
        )
    return tk_mixed_localcta_split2_dgrad(
        grad.local_row_fp4,
        grad.local_row_sc,
        grad.local_row_sg0,
        grad.local_row_sg1,
        weight0.local_col_fp4,
        weight0.local_col_sc,
        weight0.local_col_sg,
        weight1.local_col_fp4,
        weight1.local_col_sc,
        weight1.local_col_sg,
        out,
    )


@dataclass
class _MXFP4SharedRoutedXQuant:
    shared: _MXFP4RowCol
    routed: _MXFP4RowCol
    shared_rows_padded: int
    routed_rows_padded: int
    cols_padded: int


def _empty_mxfp4_row_col(M: int, H: int, device: torch.device) -> _MXFP4RowCol:
    return _MXFP4RowCol(
        row_fp4=torch.empty(M, H // 2, dtype=torch.float4_e2m1fn_x2, device=device),
        row_sc=torch.empty(M // 128, H // 128, 32, 16, dtype=torch.uint8, device=device),
        col_fp4=torch.empty(H, M // 2, dtype=torch.float4_e2m1fn_x2, device=device),
        col_sc=torch.empty(H // 128, M // 128, 32, 16, dtype=torch.uint8, device=device),
    )


_MXFP4_QKV_ROPE_TABLE_CACHE: dict[tuple[int, int, torch.dtype, str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
_MXFP4_QKV_TE_ROPE_CACHE: dict[tuple[int, int, torch.dtype, str, int | None], torch.Tensor] = {}
_MXFP4_QKV_LIVE64_ROPE_CACHE: dict[tuple[int, int, torch.dtype, str, int | None], torch.Tensor] = {}
_MXFP4_WEIGHT_QUANT_CACHE: dict[tuple[int, str], tuple[tuple[object, ...], _MXFP4RowCol]] = {}
_MXFP4_EMPTY_TENSOR_CACHE: dict[tuple[torch.dtype, str, int | None], torch.Tensor] = {}


def _mxfp4_supported(M: int, K: int) -> bool:
    return (M % 128 == 0) and (K % 128 == 0)


def _mxfp4_empty_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, device.type, device.index)
    tensor = _MXFP4_EMPTY_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty(0, dtype=dtype, device=device)
        _MXFP4_EMPTY_TENSOR_CACHE[key] = tensor
    return tensor


def _mxfp4_round_up_128(value: int) -> int:
    return _mxfp4_round_up(value, 128)


def _mxfp4_round_up_256(value: int) -> int:
    return _mxfp4_round_up(value, 256)


def _mxfp4_round_up_512(value: int) -> int:
    return _mxfp4_round_up(value, 512)


def _mxfp4_round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _mxfp4_batched_gemm_dim(value: int) -> int:
    # The current batched TK GEMM launcher rejects narrow/odd N tile shapes
    # such as 256 and 768. Use the same conservative 512-granularity contract
    # for DeepSeek grouped experts until there is a dedicated MoE kernel.
    return max(512, _mxfp4_round_up_512(value))


def _mxfp4_linear_padding_enabled() -> bool:
    return os.environ.get("MXFP4_LINEAR_PAD_UNSUPPORTED", "1") != "0"


def _mxfp4_native_nemotron_padding(
    M: int,
    K: int,
    N: int,
    Np: int,
) -> bool:
    batched_enabled = (
        os.environ.get("MXFP4_USE_BATCHED_NEMOTRON_PADDING", "1") != "0"
    )
    return (
        M % 256 == 0
        and (M == 8192 or batched_enabled)
        and K == 4096
        and N == 18560
        and Np == 18688
        and not _mxfp4_needs_opt_quant("weight")
        and not _mxfp4_rht_for_role("weight")
        and not _mxfp4_needs_opt_quant("grad")
        and not _mxfp4_rht_for_role("grad")
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _mxfp4_qkv_rope_supported(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
) -> bool:
    total_out = q_dim + k_dim + v_dim
    return (
        _mxfp4_supported(M, K)
        and _mxfp4_supported(total_out, K)
        and head_dim > 0
        and (head_dim % 2) == 0
        and (q_dim % head_dim) == 0
        and (k_dim % head_dim) == 0
        and (v_dim % 128) == 0
        and (q_dim % 128) == 0
        and (k_dim % 128) == 0
    )


def _mxfp4_qkv_rope_live64_supported(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    seq_len: int,
) -> bool:
    return (
        use_mxfp4_qkv_direct_outputs()
        and _mxfp4_qkv_rope_supported(M, K, q_dim, k_dim, v_dim, head_dim)
        and head_dim in (64, 128)
        and _is_power_of_two(seq_len)
        and (q_dim % head_dim) == 0
        and (k_dim % head_dim) == 0
    )


def _mxfp4_qkv_rope_route(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    seq_len: int,
) -> str | None:
    if _mxfp4_qkv_rope_live64_supported(
        M, K, q_dim, k_dim, v_dim, head_dim, seq_len
    ) and mxfp4_rope_live_head_dim_available(head_dim):
        return "packed"
    if (
        use_mxfp4_qkv_direct_outputs()
        and use_mxfp4_generic_qkv_rope_epilogue()
        and _mxfp4_qkv_rope_supported(
            M, K, q_dim, k_dim, v_dim, head_dim
        )
    ):
        return "generic"
    return None


def _get_mxfp4_rope_tables(
    freqs_cis: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        freqs_cis.data_ptr(),
        seq_len,
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _MXFP4_QKV_ROPE_TABLE_CACHE.get(key)
    if cached is None:
        if len(_MXFP4_QKV_ROPE_TABLE_CACHE) >= 8:
            _MXFP4_QKV_ROPE_TABLE_CACHE.clear()
        freqs_slice = freqs_cis[:seq_len]
        cached = (freqs_slice.real.contiguous(), freqs_slice.imag.contiguous())
        _MXFP4_QKV_ROPE_TABLE_CACHE[key] = cached
    return cached


def _get_mxfp4_live64_rope_cs(
    freqs_cis: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    key = (
        freqs_cis.data_ptr(),
        seq_len,
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _MXFP4_QKV_LIVE64_ROPE_CACHE.get(key)
    if cached is None:
        if len(_MXFP4_QKV_LIVE64_ROPE_CACHE) >= 8:
            _MXFP4_QKV_LIVE64_ROPE_CACHE.clear()
        freqs_slice = freqs_cis[:seq_len]
        pair_dim = int(freqs_slice.size(-1))
        if pair_dim not in (32, 64):
            raise ValueError(
                f"optimized MXFP4 RoPE requires head dimension 64 or 128, got {2 * pair_dim}"
            )
        cached = torch.stack((freqs_slice.real, freqs_slice.imag), dim=-1).contiguous()
        _MXFP4_QKV_LIVE64_ROPE_CACHE[key] = cached
    return cached


def _get_mxfp4_te_rope_freqs(freqs_cis: torch.Tensor, seq_len: int) -> torch.Tensor:
    key = (
        freqs_cis.data_ptr(),
        seq_len,
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _MXFP4_QKV_TE_ROPE_CACHE.get(key)
    if cached is None:
        if len(_MXFP4_QKV_TE_ROPE_CACHE) >= 8:
            _MXFP4_QKV_TE_ROPE_CACHE.clear()
        angles = torch.angle(freqs_cis[:seq_len])
        cached = (
            torch.stack((angles, angles), dim=-1)
            .flatten(-2)
            .view(seq_len, 1, 1, -1)
            .contiguous()
        )
        _MXFP4_QKV_TE_ROPE_CACHE[key] = cached
    return cached


def _quantize_row_col_bf16(
    tensor: torch.Tensor,
    mode: int = 1,
    role: str = "activation",
    producer_key: str | None = None,
) -> _MXFP4RowCol:
    tensor = _as_contiguous_bf16(tensor)
    if (oriented_sr := _mxfp4_oriented_grad_data_sr(role)) is not None:
        if _mxfp4_scale_sr_for_role(role):
            raise RuntimeError("oriented MXFP4 gradient data SR requires scale SR off")
        if oriented_sr == "row":
            opt_kwargs = _mxfp4_opt_kwargs(role, producer_key)
            if _mxfp4_rht_has_row(role):
                opt_kwargs.update(_mxfp4_rht_settings())
                row_fp4, row_sc = mxfp4_quantize_for_gemm_opt_rht(
                    tensor, mode, **opt_kwargs
                )
            else:
                row_fp4, row_sc = mxfp4_quantize_for_gemm_opt(
                    tensor, mode, **opt_kwargs
                )
            if _mxfp4_rht_has_col(role):
                col_fp4, col_sc = mxfp4_quantize_col_only_opt_rht(
                    tensor,
                    mode,
                    **_mxfp4_no_sr_rht_kwargs(role),
                )
            else:
                col_fp4, col_sc = mxfp4_quantize_col_only(tensor, mode)
        else:
            if _mxfp4_rht_has_row(role):
                row_fp4, row_sc = mxfp4_quantize_for_gemm_opt_rht(
                    tensor,
                    mode,
                    **_mxfp4_no_sr_rht_kwargs(role),
                )
            else:
                row_fp4, row_sc = mxfp4_quantize_for_gemm(tensor, mode)
            if _mxfp4_rht_has_col(role):
                opt_kwargs = _mxfp4_opt_kwargs(role, producer_key)
                opt_kwargs.update(_mxfp4_rht_settings())
                col_fp4, col_sc = mxfp4_quantize_col_only_opt_rht(
                    tensor, mode, **opt_kwargs
                )
            else:
                opt_kwargs = _mxfp4_opt_kwargs(role, producer_key)
                col_fp4, col_sc = mxfp4_quantize_col_only_opt(
                    tensor, mode, **opt_kwargs
                )
    elif _mxfp4_rht_for_role(role):
        row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_row_and_col_opt_rht(
            tensor,
            mode,
            **_mxfp4_rht_kwargs(role, producer_key),
        )
    elif _mxfp4_needs_opt_quant(role):
        row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_row_and_col_opt(
            tensor,
            mode,
            **_mxfp4_opt_kwargs(role, producer_key),
        )
    else:
        row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_row_and_col(tensor, mode)
    return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)


def _quantize_row_col_bf16_padded(
    tensor: torch.Tensor,
    padded_rows: int,
    padded_cols: int,
    mode: int = 1,
) -> _MXFP4RowCol:
    tensor = _as_contiguous_bf16(tensor)
    row_fp4, row_sc, col_fp4, col_sc = (
        mxfp4_pack_grouped_rows_quantize_row_and_col(
            tensor,
            1,
            tensor.shape[0],
            padded_rows,
            padded_cols,
            mode,
        )
    )
    return _MXFP4RowCol(
        row_fp4=row_fp4,
        row_sc=row_sc,
        col_fp4=col_fp4,
        col_sc=col_sc,
    )


def use_mxfp4_weight_quant_cache() -> bool:
    return os.environ.get("MXFP4_USE_WEIGHT_QUANT_CACHE", "0") == "1"


def use_mxfp4_2d_weight_quant() -> bool:
    """Use one orientation-consistent 32x32 encoding per MXFP4 weight."""
    return os.environ.get("MXFP4_USE_2D_WEIGHT_QUANT", "0") == "1"


def clear_mxfp4_weight_quant_cache() -> None:
    _MXFP4_WEIGHT_QUANT_CACHE.clear()


def _use_weight_quant_cache_for_mode(mode: int) -> bool:
    if (
        not use_mxfp4_weight_quant_cache()
        or mode != 1
        or _mxfp4_rht_for_role("weight")
        or _mxfp4_needs_opt_quant("weight")
    ):
        return False
    return True


def _weight_quant_cache_key_signature(tensor: torch.Tensor) -> tuple[tuple[int, str], tuple[object, ...]]:
    quant_contract = "row_col_2d" if use_mxfp4_2d_weight_quant() else "row_col"
    key = (int(tensor.data_ptr()), quant_contract)
    version = int(getattr(tensor, "_version", 0))
    signature = (
        version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )
    return key, signature


def _lookup_weight_row_col_bf16(tensor: torch.Tensor, mode: int = 1) -> _MXFP4RowCol | None:
    tensor = _as_contiguous_bf16(tensor)
    if not _use_weight_quant_cache_for_mode(mode):
        return None
    key, signature = _weight_quant_cache_key_signature(tensor)
    cached = _MXFP4_WEIGHT_QUANT_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    return None


def _quantize_weight_row_col_bf16(tensor: torch.Tensor, mode: int = 1) -> _MXFP4RowCol:
    tensor = _as_contiguous_bf16(tensor)
    use_cache = _use_weight_quant_cache_for_mode(mode)
    if use_cache:
        cached = _lookup_weight_row_col_bf16(tensor, mode)
        if cached is not None:
            return cached
    if use_mxfp4_2d_weight_quant():
        if mode != 1:
            raise RuntimeError("MXFP4 2D weight quantization requires encode-centric mode")
        row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_weight_2d(tensor)
        value = _MXFP4RowCol(
            row_fp4=row_fp4,
            row_sc=row_sc,
            col_fp4=col_fp4,
            col_sc=col_sc,
        )
    else:
        value = _quantize_row_col_bf16(tensor, mode, role="weight")
    if use_cache:
        key, signature = _weight_quant_cache_key_signature(tensor)
        _MXFP4_WEIGHT_QUANT_CACHE[key] = (signature, value)
    return value


def _use_grad_split_col_overlap() -> bool:
    return (
        use_mxfp4_sqrelu_split_col_overlap()
        and _mxfp4_needs_opt_quant("grad")
        and _mxfp4_rht_has_col("grad")
        and not _mxfp4_rht_has_row("grad")
    )


def _quantize_nhsd_wo_row_col_bf16(tensor: torch.Tensor, mode: int = 1, role: str = "activation") -> _MXFP4RowCol:
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.dim() != 4:
        raise RuntimeError(f"MXFP4 NHSD WO quantizer expects [B,H,S,D], got {tuple(tensor.shape)}")
    if _mxfp4_needs_opt_quant(role) or _mxfp4_rht_for_role(role):
        B, H, S, D = tensor.shape
        return _quantize_row_col_bf16(
            tensor.transpose(1, 2).contiguous().view(B * S, H * D),
            mode=mode,
            role=role,
        )
    row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_nhsd_wo_row_and_col(tensor, mode)
    return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)


def _sqrelu_quantize_row_col_bf16(tensor: torch.Tensor, mode: int = 1, role: str = "activation") -> _MXFP4RowCol:
    tensor = _as_contiguous_bf16(tensor)
    if use_mxfp4_fused_sqrelu_quant():
        if _mxfp4_needs_opt_quant(role):
            opt_supported = (
                mode == 1
                and not _mxfp4_rht_random_sign_mask()
                and _mxfp4_rht_block_size() == 32
                and not _mxfp4_scale_sr_for_role(role)
                and (
                    (
                        _mxfp4_rht_has_col(role)
                        and not _mxfp4_rht_has_row(role)
                        and not _mxfp4_data_sr_for_role(role)
                    )
                    or (
                        not _mxfp4_rht_has_col(role)
                        and not _mxfp4_rht_has_row(role)
                        and _mxfp4_data_sr_for_role(role)
                    )
                )
            )
            if opt_supported:
                try:
                    q = _empty_mxfp4_row_col(tensor.shape[0], tensor.shape[1], tensor.device)
                    opt_kwargs = _mxfp4_opt_kwargs(role)
                    mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace(
                        tensor,
                        q.row_fp4,
                        q.row_sc,
                        q.col_fp4,
                        q.col_sc,
                        mode,
                        use_rht=_mxfp4_rht_has_col(role),
                        rht_block_size=_mxfp4_rht_block_size(),
                        with_random_sign_mask=False,
                        row_with_rht=_mxfp4_rht_has_row(role),
                        **opt_kwargs,
                    )
                    return q
                except AttributeError:
                    pass
        else:
            try:
                row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_sqrelu_quantize_row_and_col(tensor, mode)
                return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
            except AttributeError:
                pass
    return _quantize_row_col_bf16(sqrelu_fwd(tensor), mode=mode, role=role)


def _sqrelu_deriv_quantize_row_col_bf16(
    dh: torch.Tensor,
    h1_raw: torch.Tensor,
    mode: int = 1,
    role: str = "grad",
    producer_key: str | None = None,
) -> _MXFP4RowCol:
    dh = _as_contiguous_bf16(dh)
    h1_raw = _as_contiguous_bf16(h1_raw)
    if use_mxfp4_fused_sqrelu_deriv_quant():
        if _mxfp4_needs_opt_quant(role):
            opt_supported = (
                mode == 1
                and not _mxfp4_rht_random_sign_mask()
                and _mxfp4_rht_block_size() == 32
                and not _mxfp4_rht_has_row(role)
                and not _mxfp4_scale_sr_for_role(role)
                and (
                    (
                        _mxfp4_data_sr_for_role(role)
                        and not _mxfp4_rht_has_col(role)
                    )
                    or _mxfp4_rht_has_col(role)
                )
                and (
                    not (_mxfp4_data_sr_for_role(role) and _mxfp4_rht_has_col(role))
                    or use_mxfp4_sqrelu_deriv_rht_sr()
                )
            )
            if opt_supported:
                try:
                    q = _empty_mxfp4_row_col(dh.shape[0], dh.shape[1], dh.device)
                    opt_kwargs = _mxfp4_opt_kwargs(role, producer_key)
                    mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace(
                        dh,
                        h1_raw,
                        q.row_fp4,
                        q.row_sc,
                        q.col_fp4,
                        q.col_sc,
                        mode,
                        use_rht=_mxfp4_rht_has_col(role),
                        rht_block_size=_mxfp4_rht_block_size(),
                        with_random_sign_mask=False,
                        row_with_rht=_mxfp4_rht_has_row(role),
                        **opt_kwargs,
                    )
                    return q
                except AttributeError:
                    pass
        else:
            try:
                row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_sqrelu_deriv_quantize_row_and_col(dh, h1_raw, mode)
                return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
            except AttributeError:
                pass
    return _quantize_row_col_bf16(
        sqrelu_bwd(dh, h1_raw),
        mode=mode,
        role=role,
        producer_key=producer_key,
    )


def _rmsnorm_quantize_row_col_bf16(
    te_fused,
    tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
    kind: str,
    mode: int = 1,
) -> tuple[_MXFP4RowCol, torch.Tensor]:
    role = "activation"
    # DATA_SR support exists in the fused backend, but the first trainer screen
    # was neutral/slightly slower than the old materialized route. Keep it
    # opt-in until it beats the end-to-end RHT+SR path.
    fused_opt_supported = (
        not _mxfp4_data_sr_for_role(role)
        or _mxfp4_bool_env("MXFP4_USE_FUSED_RMSNORM_QUANT_DATA_SR", False)
    )
    fused_rht_supported = (
        not _mxfp4_rht_for_role(role)
        or (
            _mxfp4_bool_env("MXFP4_USE_FUSED_RMSNORM_QUANT_RHT", True)
            and not _mxfp4_rht_has_row(role)
        )
    )
    if (
        use_mxfp4_fused_rmsnorm_quant(kind)
        and fused_rht_supported
        and (not _mxfp4_needs_opt_quant(role) or fused_opt_supported)
    ):
        if _mxfp4_needs_opt_quant(role):
            opt_kwargs = _mxfp4_opt_kwargs(role)
            if _mxfp4_rht_has_row(role):
                opt_kwargs["row_with_rht"] = True
            row_fp4, row_sc, col_fp4, col_sc, inv_rms = mxfp4_fused_rmsnorm_quantize_row_and_col_opt(
                _as_contiguous_bf16(tensor),
                _as_contiguous_bf16(norm_weight),
                float(epsilon),
                mode,
                use_rht=_mxfp4_rht_has_col(role),
                rht_block_size=_mxfp4_rht_block_size(),
                with_random_sign_mask=_mxfp4_rht_random_sign_mask(),
                **opt_kwargs,
            )
            return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc), inv_rms
        row_fp4, row_sc, col_fp4, col_sc, inv_rms = mxfp4_fused_rmsnorm_quantize_row_and_col(
            _as_contiguous_bf16(tensor),
            _as_contiguous_bf16(norm_weight),
            float(epsilon),
            mode,
        )
        return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc), inv_rms

    normed, inv_rms = te_fused.fused_rmsnorm_only(
        _as_contiguous_bf16(tensor),
        _as_contiguous_bf16(norm_weight),
        float(epsilon),
    )
    return _quantize_row_col_bf16(normed, mode, role=role), inv_rms


def _rmsnorm_to_bf16(
    te_fused,
    tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "qkv" and use_mxfp4_fused_rmsnorm_quant(kind):
        return mxfp4_fused_rmsnorm_to_bf16(
            _as_contiguous_bf16(tensor),
            _as_contiguous_bf16(norm_weight),
            float(epsilon),
        )
    return te_fused.fused_rmsnorm_only(
        _as_contiguous_bf16(tensor),
        _as_contiguous_bf16(norm_weight),
        float(epsilon),
    )


def _mxfp4_gemm_qkv(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """QKV/WO-specific GEMM config picks for large aligned MXFP4 shapes."""
    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))

    config_id = _mxfp4_gemm_config_override("MXFP4_QKV_GEMM_CONFIG", M, N, K, config_id)
    if config_id is None:
        return mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
    return mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, config_id=config_id)


def _mxfp4_gemm_wo_dgrad(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """WO dgrad GEMM with a selector independent from fused QKV/RoPE."""
    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))
    config_id = _mxfp4_gemm_config_override(
        "MXFP4_WO_DGRAD_GEMM_CONFIG", M, N, K, config_id
    )
    if config_id is None:
        return mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
    return mxfp4_gemm_config(
        A_fp4, A_sc, B_fp4, B_sc, out, config_id=config_id
    )


def _mxfp4_gemm_linear(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shape picks for the paper square-ReLU FFN's plain MXFP4 linear path."""
    if not use_mxfp4_linear_gemm_configs():
        return mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)

    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))

    if config_id is None and M >= 32768:
        if K == 2048 and N == 6144:
            config_id = 10
        elif K == 6144 and N == 2048:
            config_id = 10

    config_id = _mxfp4_gemm_config_override("MXFP4_LINEAR_GEMM_CONFIG", M, N, K, config_id)
    if config_id is None:
        return mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
    return mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, config_id=config_id)


def _mxfp4_gemm_linear_sqrelu_deriv(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (use_mxfp4_linear_gemm_configs() and use_mxfp4_sqrelu_deriv_gemm_epilogue()):
        hidden = _mxfp4_gemm_linear(A_fp4, A_sc, B_fp4, B_sc, out)
        return sqrelu_bwd(hidden, x)

    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))

    if config_id is None and M >= 32768:
        if K == 2048 and N == 6144:
            config_id = 10
        elif K == 6144 and N == 2048:
            config_id = 10

    shape_config_env = f"MXFP4_SQRELU_DERIV_GEMM_CONFIG_M{M}_N{N}"
    if shape_config_env in os.environ:
        config_id = _mxfp4_int_env(
            shape_config_env,
            config_id if config_id is not None else -1,
        )
    elif os.environ.get("MXFP4_SQRELU_DERIV_GEMM_CONFIG_ID") is not None:
        config_id = _mxfp4_int_env(
            "MXFP4_SQRELU_DERIV_GEMM_CONFIG_ID",
            config_id if config_id is not None else -1,
        )
    if config_id is not None and config_id < 0:
        config_id = None

    if config_id is None:
        hidden = mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
        return sqrelu_bwd(hidden, x)
    return mxfp4_gemm_sqrelu_deriv_config(
        A_fp4, A_sc, B_fp4, B_sc, x, out, config_id=config_id
    )


def _mxfp4_gemm_ffn_dh(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))
    config_id = _mxfp4_gemm_config_override("MXFP4_FFN_DH_GEMM_CONFIG", M, N, K, config_id)
    if config_id is None:
        return mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
    return mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, config_id=config_id)


def _mxfp4_gemm_linear_residual(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (use_mxfp4_linear_gemm_configs() and use_mxfp4_linear_residual_config()):
        return mxfp4_gemm_residual(A_fp4, A_sc, B_fp4, B_sc, residual, out)

    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))

    if config_id is None and M >= 32768:
        if K == 2048 and N == 6144:
            config_id = 10
        elif K == 6144 and N == 2048:
            config_id = 10

    config_id = _mxfp4_gemm_config_override("MXFP4_LINEAR_RESIDUAL_GEMM_CONFIG", M, N, K, config_id)
    if config_id is None:
        return mxfp4_gemm_residual(A_fp4, A_sc, B_fp4, B_sc, residual, out)
    return mxfp4_gemm_residual_config(
        A_fp4, A_sc, B_fp4, B_sc, residual, out, config_id=config_id
    )


_NEMOTRON_PROJECTION_WGRAD_CONFIGS = {
    (18688, 4096, 8192): 7,
    (18688, 4096, 16384): 10,
    (18688, 4096, 24576): 0,
    (18688, 4096, 32768): 0,
}


def _mxfp4_wgrad_config_for_shape(M: int, N: int, K: int) -> int | None:
    nemotron_wgrad_config = _NEMOTRON_PROJECTION_WGRAD_CONFIGS.get((M, N, K))
    if nemotron_wgrad_config is not None:
        return nemotron_wgrad_config
    if K < 32768:
        return None
    if M == 4096 and N in {4096, 14336}:
        return 10
    if M == 14336 and N == 4096:
        return 10
    if M == 6144 and N == 4096:
        return 10
    if (M, N) in {(4096, 2048), (2048, 2048), (2048, 6144)}:
        # A step-dependent config-5/7 or config-3/10 handoff can leave
        # clustered launches from two schedules resident together. Config 18
        # is bitwise-equivalent, faster for all three shapes, and remains
        # fixed for the lifetime of the process.
        return 18
    if M == 1024 and N == 2048:
        return 10
    if M == 6144 and N == 2048:
        return 4
    return None


def _mxfp4_gemm_wgrad(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not use_mxfp4_wgrad_gemm_configs():
        return mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)

    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = _mxfp4_wgrad_config_for_shape(M, N, K)

    config_id = _mxfp4_gemm_config_override("MXFP4_WGRAD_GEMM_CONFIG", M, N, K, config_id)
    shape_config_env = f"MXFP4_WGRAD_GEMM_CONFIG_M{M}_N{N}"
    if config_id is None and shape_config_env in os.environ:
        config_id = _mxfp4_int_env(
            shape_config_env,
            config_id if config_id is not None else -1,
        )
    elif config_id is None and os.environ.get("MXFP4_WGRAD_GEMM_CONFIG_ID") is not None:
        config_id = _mxfp4_int_env("MXFP4_WGRAD_GEMM_CONFIG_ID", config_id if config_id is not None else -1)
    if config_id is not None and config_id < 0:
        config_id = None

    if config_id is None:
        return mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
    return mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, config_id=config_id)


def _mxfp4_gemm_qkv_rope(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_seq_len: int,
    rope_head_dim: int,
    rope_rotary_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))

    config_id = _mxfp4_gemm_config_override("MXFP4_QKV_GEMM_CONFIG", M, N, K, config_id)
    if config_id is None:
        return mxfp4_gemm_rope(
            A_fp4,
            A_sc,
            B_fp4,
            B_sc,
            rope_cos,
            rope_sin,
            rope_seq_len,
            rope_head_dim,
            rope_rotary_dim,
            out,
        )
    return mxfp4_gemm_rope_config(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        rope_cos,
        rope_sin,
        rope_seq_len,
        rope_head_dim,
        rope_rotary_dim,
        out,
        config_id=config_id,
    )


def _mxfp4_gemm_qkv_rope_live64(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M = A_fp4.size(0)
    N = B_fp4.size(0)
    K = A_fp4.size(1) * 2
    config_id = mxfp4_dense_gemm_config_for_shape(int(M), int(N), int(K))

    config_id = _mxfp4_gemm_config_override("MXFP4_QKV_GEMM_CONFIG", M, N, K, config_id)
    if int(rope_cs.size(1)) == 64 and config_id is None:
        # The native untuned launcher faults for the smaller Llama head-128 Q
        # shape. Preserve selector-backed launchers such as the proven
        # M=32768 config-10 production route, and use the one-problem batched
        # launcher only when no configured route is available.
        outputs = mxfp4_batched_gemm_rope_live64(
            [A_fp4],
            [A_sc],
            [B_fp4],
            [B_sc],
            [rope_cs],
            [rope_seq_len],
            None if out is None else [out],
        )
        return outputs[0]

    if config_id is None:
        return mxfp4_gemm_rope_live64(
            A_fp4,
            A_sc,
            B_fp4,
            B_sc,
            rope_cs,
            rope_seq_len,
            out,
        )
    return mxfp4_gemm_rope_live64_config(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        rope_cs,
        rope_seq_len,
        out,
        config_id=config_id,
    )


def _apply_inverse_rotary_qk(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    freqs_cis: torch.Tensor,
    batch_size: int,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gq_4d = grad_q.view(batch_size, seq_len, -1, head_dim)
    gk_4d = grad_k.view(batch_size, seq_len, -1, head_dim)

    te_apply_rotary_pos_emb = _get_te_apply_rotary_pos_emb()
    if te_apply_rotary_pos_emb is not None and grad_q.is_cuda and grad_k.is_cuda and freqs_cis.is_cuda:
        rope_freqs = _get_mxfp4_te_rope_freqs(freqs_cis, seq_len)
        gq_4d = te_apply_rotary_pos_emb(
            gq_4d,
            -rope_freqs,
            tensor_format="bshd",
            fused=True,
            interleaved=True,
        )
        gk_4d = te_apply_rotary_pos_emb(
            gk_4d,
            -rope_freqs,
            tensor_format="bshd",
            fused=True,
            interleaved=True,
        )
        return gq_4d.reshape_as(grad_q), gk_4d.reshape_as(grad_k)

    from torchtitan.models.llama3.model.model import apply_rotary_emb

    gq_4d, gk_4d = apply_rotary_emb(gq_4d, gk_4d, freqs_cis=freqs_cis[:seq_len].conj())
    return gq_4d.reshape_as(grad_q), gk_4d.reshape_as(grad_k)


def _apply_forward_rotary_tensor(
    value: torch.Tensor,
    freqs_cis: torch.Tensor,
    batch_size: int,
    seq_len: int,
    head_dim: int,
) -> torch.Tensor:
    value_4d = value.view(batch_size, seq_len, -1, head_dim)
    te_apply_rotary_pos_emb = _get_te_apply_rotary_pos_emb()
    if te_apply_rotary_pos_emb is not None and value.is_cuda and freqs_cis.is_cuda:
        rope_freqs = _get_mxfp4_te_rope_freqs(freqs_cis, seq_len)
        value_4d = te_apply_rotary_pos_emb(
            value_4d,
            rope_freqs,
            tensor_format="bshd",
            fused=True,
            interleaved=True,
        )
        return value_4d.reshape_as(value)

    from torchtitan.models.llama3.model.model import apply_rotary_emb

    value_4d, _ = apply_rotary_emb(
        value_4d,
        value_4d,
        freqs_cis=freqs_cis[:seq_len],
    )
    return value_4d.reshape_as(value)


_MXFP4_RMS_BWD_CACHE: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
_MXFP4_FFN_BWD_CACHE: dict[tuple[int, int, int, int], dict[str, torch.Tensor]] = {}
_MXFP4_QKV_BWD_CACHE: dict[tuple[int, int, int, int, int, int], dict[str, object]] = {}


class _LazyMXFP4FFNBwdState(dict):
    def __init__(self, M: int, K: int, H: int, device: torch.device):
        super().__init__()
        self.M = int(M)
        self.K = int(K)
        self.H = int(H)
        self.device = device

    def __missing__(self, key: str) -> torch.Tensor:
        M, K, H, device = self.M, self.K, self.H, self.device
        if key == "dh":
            value = torch.empty(M, H, dtype=torch.bfloat16, device=device)
        elif key == "dh1":
            value = torch.empty(M, H, dtype=torch.bfloat16, device=device)
        elif key == "dh3":
            value = torch.empty(M, H, dtype=torch.bfloat16, device=device)
        elif key == "split2_row_fp4":
            value = torch.empty(M, H, dtype=torch.float4_e2m1fn_x2, device=device)
        elif key == "split2_row_sc":
            value = torch.empty(M // 128, (2 * H) // 128, 32, 16, dtype=torch.uint8, device=device)
        elif key == "split2_col_fp4":
            value = torch.empty(2 * H, M // 2, dtype=torch.float4_e2m1fn_x2, device=device)
        elif key == "split2_col_sc":
            value = torch.empty((2 * H) // 128, M // 128, 32, 16, dtype=torch.uint8, device=device)
        elif key == "fused_split2_row_fp4":
            value = torch.empty(2, M, H // 2, dtype=torch.float4_e2m1fn_x2, device=device)
        elif key == "fused_split2_row_sc":
            value = torch.empty(2, M // 128, H // 128, 32, 16, dtype=torch.uint8, device=device)
        elif key == "fused_split2_col_fp4":
            value = torch.empty(2, H, M // 2, dtype=torch.float4_e2m1fn_x2, device=device)
        elif key == "fused_split2_col_sc":
            value = torch.empty(2, H // 128, M // 128, 32, 16, dtype=torch.uint8, device=device)
        elif key == "grad_w2":
            value = torch.empty(K, H, dtype=torch.bfloat16, device=device)
        elif key == "dx0":
            value = torch.empty(M, K, dtype=torch.bfloat16, device=device)
        elif key == "dx1":
            value = torch.empty(M, K, dtype=torch.bfloat16, device=device)
        elif key == "grad_w1":
            value = torch.empty(H, K, dtype=torch.bfloat16, device=device)
        elif key == "grad_w3":
            value = torch.empty(H, K, dtype=torch.bfloat16, device=device)
        else:
            raise KeyError(key)
        self[key] = value
        return value


def _get_mxfp4_rms_bwd_state(M: int, K: int, device: torch.device) -> dict[str, torch.Tensor]:
    grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=device)
    grad_norm = torch.empty(K, dtype=torch.float32, device=device)
    row_tiles = (M + 255) // 256
    if not use_mxfp4_bwd_state_cache():
        return {
            "grad_input": grad_input,
            "grad_norm": grad_norm,
            "grad_norm_partials": torch.empty(row_tiles, K, dtype=torch.float32, device=device),
        }
    key = (M, K, int(device.index))
    state = _MXFP4_RMS_BWD_CACHE.get(key)
    if state is None:
        state = {
            "grad_norm_partials": torch.empty(row_tiles, K, dtype=torch.float32, device=device),
        }
        _MXFP4_RMS_BWD_CACHE[key] = state
    return {
        "grad_input": grad_input,
        "grad_norm": grad_norm,
        "grad_norm_partials": state["grad_norm_partials"],
    }


def _get_mxfp4_ffn_bwd_state(
    M: int,
    K: int,
    H: int,
    device: torch.device,
    *,
    force_lazy: bool = False,
) -> dict[str, torch.Tensor]:
    # The mixed dgrad route creates its own localCTA-row/MX-column split-2
    # carrier.  Eagerly allocating the native and fused split-2 carrier
    # families as well retains another 1,996,488,704 bytes (1,904 MiB) at the
    # production M=32768, H=14336 shape without ever consuming them.  Keep the
    # native policy unchanged, but let that route explicitly request the
    # existing lazy state so only buffers it actually indexes are materialized.
    if force_lazy or use_mxfp4_lazy_ffn_bwd_state():
        return _LazyMXFP4FFNBwdState(M, K, H, device)
    if not use_mxfp4_bwd_state_cache():
        return {
            "dh": torch.empty(M, H, dtype=torch.bfloat16, device=device),
            "dh1": torch.empty(M, H, dtype=torch.bfloat16, device=device),
            "dh3": torch.empty(M, H, dtype=torch.bfloat16, device=device),
            "split2_row_fp4": torch.empty(M, H, dtype=torch.float4_e2m1fn_x2, device=device),
            "split2_row_sc": torch.empty(M // 128, (2 * H) // 128, 32, 16, dtype=torch.uint8, device=device),
            "split2_col_fp4": torch.empty(2 * H, M // 2, dtype=torch.float4_e2m1fn_x2, device=device),
            "split2_col_sc": torch.empty((2 * H) // 128, M // 128, 32, 16, dtype=torch.uint8, device=device),
            "fused_split2_row_fp4": torch.empty(2, M, H // 2, dtype=torch.float4_e2m1fn_x2, device=device),
            "fused_split2_row_sc": torch.empty(2, M // 128, H // 128, 32, 16, dtype=torch.uint8, device=device),
            "fused_split2_col_fp4": torch.empty(2, H, M // 2, dtype=torch.float4_e2m1fn_x2, device=device),
            "fused_split2_col_sc": torch.empty(2, H // 128, M // 128, 32, 16, dtype=torch.uint8, device=device),
            "grad_w2": torch.empty(K, H, dtype=torch.bfloat16, device=device),
            "dx0": torch.empty(M, K, dtype=torch.bfloat16, device=device),
            "dx1": torch.empty(M, K, dtype=torch.bfloat16, device=device),
            "grad_w1": torch.empty(H, K, dtype=torch.bfloat16, device=device),
            "grad_w3": torch.empty(H, K, dtype=torch.bfloat16, device=device),
        }
    key = (M, K, H, int(device.index))
    state = _MXFP4_FFN_BWD_CACHE.get(key)
    if state is None:
        state = {
            "dh": torch.empty(M, H, dtype=torch.bfloat16, device=device),
            "dh1": torch.empty(M, H, dtype=torch.bfloat16, device=device),
            "dh3": torch.empty(M, H, dtype=torch.bfloat16, device=device),
            "split2_row_fp4": torch.empty(M, H, dtype=torch.float4_e2m1fn_x2, device=device),
            "split2_row_sc": torch.empty(M // 128, (2 * H) // 128, 32, 16, dtype=torch.uint8, device=device),
            "split2_col_fp4": torch.empty(2 * H, M // 2, dtype=torch.float4_e2m1fn_x2, device=device),
            "split2_col_sc": torch.empty((2 * H) // 128, M // 128, 32, 16, dtype=torch.uint8, device=device),
            "fused_split2_row_fp4": torch.empty(2, M, H // 2, dtype=torch.float4_e2m1fn_x2, device=device),
            "fused_split2_row_sc": torch.empty(2, M // 128, H // 128, 32, 16, dtype=torch.uint8, device=device),
            "fused_split2_col_fp4": torch.empty(2, H, M // 2, dtype=torch.float4_e2m1fn_x2, device=device),
            "fused_split2_col_sc": torch.empty(2, H // 128, M // 128, 32, 16, dtype=torch.uint8, device=device),
            "grad_w2": torch.empty(K, H, dtype=torch.bfloat16, device=device),
            "dx0": torch.empty(M, K, dtype=torch.bfloat16, device=device),
            "dx1": torch.empty(M, K, dtype=torch.bfloat16, device=device),
            "grad_w1": torch.empty(H, K, dtype=torch.bfloat16, device=device),
            "grad_w3": torch.empty(H, K, dtype=torch.bfloat16, device=device),
        }
        _MXFP4_FFN_BWD_CACHE[key] = state
    state = dict(state)
    state["grad_w2"] = torch.empty(K, H, dtype=torch.bfloat16, device=device)
    state["grad_w1"] = torch.empty(H, K, dtype=torch.bfloat16, device=device)
    state["grad_w3"] = torch.empty(H, K, dtype=torch.bfloat16, device=device)
    return state


@dataclass
class _MXFP4StateHandle:
    state: dict[str, torch.Tensor]
    entry: dict[str, object] | None = None
    slot_idx: int | None = None


def _alloc_mxfp4_qkv_bwd_state(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state = {
        "split3_row_fp4": torch.empty(M, (q_dim + k_dim + v_dim) // 2, dtype=torch.float4_e2m1fn_x2, device=device),
        "split3_row_sc": torch.empty(M // 128, (q_dim + k_dim + v_dim) // 128, 32, 16, dtype=torch.uint8, device=device),
        "split3_col_fp4": torch.empty(q_dim + k_dim + v_dim, M // 2, dtype=torch.float4_e2m1fn_x2, device=device),
        "split3_col_sc": torch.empty((q_dim + k_dim + v_dim) // 128, M // 128, 32, 16, dtype=torch.uint8, device=device),
        "dx": torch.empty(M, K, dtype=torch.bfloat16, device=device),
    }
    stage_mask = _mxfp4_split3_qkv_stage_copy_mask(M)
    if "q" in stage_mask:
        state["gq"] = torch.empty(M, q_dim, dtype=torch.bfloat16, device=device)
    if "k" in stage_mask:
        state["gk"] = torch.empty(M, k_dim, dtype=torch.bfloat16, device=device)
    if "v" in stage_mask:
        state["gv"] = torch.empty(M, v_dim, dtype=torch.bfloat16, device=device)
    return state


def _mxfp4_qkv_bwd_state_slots() -> int:
    default_slots = "2" if use_mxfp4_split3_qkv_stage_copy() else "4"
    try:
        return max(2, int(os.environ.get("MXFP4_QKV_BWD_STATE_SLOTS", default_slots)))
    except ValueError:
        return int(default_slots)


def _get_mxfp4_qkv_bwd_state(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    device: torch.device,
    protect_async_reuse: bool = False,
) -> _MXFP4StateHandle:
    if not use_mxfp4_bwd_state_cache():
        return _MXFP4StateHandle(
            state=_alloc_mxfp4_qkv_bwd_state(M, K, q_dim, k_dim, v_dim, device)
        )
    key = (M, K, q_dim, k_dim, v_dim, int(device.index))
    entry = _MXFP4_QKV_BWD_CACHE.get(key)
    required_slots = _mxfp4_qkv_bwd_state_slots() if protect_async_reuse else 1
    if entry is None:
        entry = {
            "slots": [
                _alloc_mxfp4_qkv_bwd_state(M, K, q_dim, k_dim, v_dim, device)
                for _ in range(required_slots)
            ],
            "events": [torch.cuda.Event() for _ in range(required_slots)],
            "armed": [False] * required_slots,
            "next": 0,
        }
        _MXFP4_QKV_BWD_CACHE[key] = entry
    elif required_slots > len(entry["slots"]):
        extra = required_slots - len(entry["slots"])
        entry["slots"].extend(
            _alloc_mxfp4_qkv_bwd_state(M, K, q_dim, k_dim, v_dim, device)
            for _ in range(extra)
        )
        entry["events"].extend(torch.cuda.Event() for _ in range(extra))
        entry["armed"].extend([False] * extra)

    slot_idx = int(entry["next"])
    entry["next"] = (slot_idx + 1) % len(entry["slots"])
    if entry["armed"][slot_idx]:
        torch.cuda.current_stream().wait_event(entry["events"][slot_idx])
        entry["armed"][slot_idx] = False
    return _MXFP4StateHandle(
        state=entry["slots"][slot_idx],
        entry=entry,
        slot_idx=slot_idx,
    )


def _stage_split3_qkv_grads(
    state: dict[str, torch.Tensor],
    gq: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not any(key in state for key in ("gq", "gk", "gv")):
        current_stream = torch.cuda.current_stream()
        # The split3 quantizer is a custom extension, so make the allocator keep
        # the autograd-produced grad buffers alive on the consuming stream even
        # when we skip the explicit current-stream scratch copies.
        _record_stream_tree((gq, gk, gv), current_stream)
        return gq, gk, gv
    # The split3 quantizer builds host-side TMA descriptors for its BF16 inputs.
    # In autograd these grad tensors can arrive from a producer stream that is not
    # immediately visible to that host-side encode step, so copy them into cached
    # current-stream scratch first.
    staged_q = state["gq"] if "gq" in state else gq
    staged_k = state["gk"] if "gk" in state else gk
    staged_v = state["gv"] if "gv" in state else gv
    if "gq" in state:
        staged_q.copy_(gq)
    if "gk" in state:
        staged_k.copy_(gk)
    if "gv" in state:
        staged_v.copy_(gv)
    current_stream = torch.cuda.current_stream()
    _record_stream_tree(
        tuple(t for key, t in (("gq", gq), ("gk", gk), ("gv", gv)) if key not in state),
        current_stream,
    )
    return staged_q, staged_k, staged_v


def _quantize_split3_qkv_grads(
    state: dict[str, torch.Tensor],
    gq: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    rope_cs: torch.Tensor | None = None,
    rope_seq_len: int = 0,
) -> _MXFP4RowCol:
    split3_gq, split3_gk, split3_gv = _stage_split3_qkv_grads(state, gq, gk, gv)
    try:
        if not use_mxfp4_split3_qkv_inplace_quant():
            raise AttributeError
        if rope_cs is None:
            if _mxfp4_needs_opt_quant("grad"):
                mxfp4_quantize_split3_row_and_col_opt_launch_inplace(
                    split3_gq,
                    split3_gk,
                    split3_gv,
                    state["split3_row_fp4"],
                    state["split3_row_sc"],
                    state["split3_col_fp4"],
                    state["split3_col_sc"],
                    **_mxfp4_split_opt_kwargs("grad"),
                )
            else:
                mxfp4_quantize_split3_row_and_col_launch_inplace(
                    split3_gq,
                    split3_gk,
                    split3_gv,
                    state["split3_row_fp4"],
                    state["split3_row_sc"],
                    state["split3_col_fp4"],
                    state["split3_col_sc"],
                )
        else:
            if _mxfp4_needs_opt_quant("grad"):
                mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace(
                    split3_gq,
                    split3_gk,
                    split3_gv,
                    rope_cs,
                    rope_seq_len,
                    state["split3_row_fp4"],
                    state["split3_row_sc"],
                    state["split3_col_fp4"],
                    state["split3_col_sc"],
                    **_mxfp4_split_opt_kwargs("grad"),
                )
            else:
                mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace(
                    split3_gq,
                    split3_gk,
                    split3_gv,
                    rope_cs,
                    rope_seq_len,
                    state["split3_row_fp4"],
                    state["split3_row_sc"],
                    state["split3_col_fp4"],
                    state["split3_col_sc"],
                )
        return _MXFP4RowCol(
            row_fp4=state["split3_row_fp4"],
            row_sc=state["split3_row_sc"],
            col_fp4=state["split3_col_fp4"],
            col_sc=state["split3_col_sc"],
        )
    except AttributeError:
        if _mxfp4_needs_opt_quant("grad"):
            raise
        if rope_cs is None:
            row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_split3_row_and_col(
                split3_gq, split3_gk, split3_gv
            )
        else:
            row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_split3_row_and_col_inverse_rope_live64(
                split3_gq, split3_gk, split3_gv, rope_cs, rope_seq_len
            )
        return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)


def _split_qkv_rowcol(
    qkv_q: _MXFP4RowCol,
    q_dim: int,
    k_dim: int,
    v_dim: int,
) -> tuple[_MXFP4RowCol, _MXFP4RowCol, _MXFP4RowCol]:
    dims = (q_dim, k_dim, v_dim)
    row_fp4_u8 = qkv_q.row_fp4.view(torch.uint8)
    col_fp4_u8 = qkv_q.col_fp4.view(torch.uint8)
    packed_offsets = [0]
    sc_offsets = [0]
    for dim in dims[:-1]:
        packed_offsets.append(packed_offsets[-1] + dim // 2)
        sc_offsets.append(sc_offsets[-1] + dim // 128)
    splits = []
    for dim, packed_off, sc_off in zip(dims, packed_offsets, sc_offsets, strict=True):
        packed = dim // 2
        sc_cols = dim // 128
        splits.append(
            _MXFP4RowCol(
                row_fp4=row_fp4_u8.narrow(1, packed_off, packed).contiguous().view(torch.float4_e2m1fn_x2),
                row_sc=qkv_q.row_sc.narrow(1, sc_off, sc_cols).contiguous(),
                col_fp4=col_fp4_u8.narrow(0, packed_off * 2, dim).contiguous().view(torch.float4_e2m1fn_x2),
                col_sc=qkv_q.col_sc.narrow(0, sc_off, sc_cols).contiguous(),
            )
        )
    return tuple(splits)


def _quantize_split2_row_and_col_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    role: str = "grad",
    producer_key: str | None = None,
) -> None:
    if _mxfp4_needs_opt_quant(role):
        oriented_sr = _mxfp4_oriented_grad_data_sr(role)
        if oriented_sr is not None:
            if _mxfp4_scale_sr_for_role(role):
                raise RuntimeError(
                    "oriented MXFP4 gradient data SR requires scale SR off"
                )
            if oriented_sr == "row":
                mxfp4_quantize_split2_row_only_opt_launch_inplace(
                    input0,
                    input1,
                    row_fp4,
                    row_sc,
                    mode=1,
                    **_mxfp4_split_row_only_opt_kwargs(role, producer_key),
                )
                if _mxfp4_rht_has_col(role):
                    mxfp4_quantize_split2_col_only_opt_launch_inplace(
                        input0,
                        input1,
                        col_fp4,
                        col_sc,
                        mode=1,
                        **_mxfp4_split_axis_only_no_sr_rht_kwargs(role, "col"),
                    )
                else:
                    mxfp4_quantize_split2_col_only_launch_inplace(
                        input0, input1, col_fp4, col_sc, mode=1
                    )
            else:
                if _mxfp4_rht_has_row(role):
                    mxfp4_quantize_split2_row_only_opt_launch_inplace(
                        input0,
                        input1,
                        row_fp4,
                        row_sc,
                        mode=1,
                        **_mxfp4_split_axis_only_no_sr_rht_kwargs(role, "row"),
                    )
                else:
                    mxfp4_quantize_split2_row_only_launch_inplace(
                        input0, input1, row_fp4, row_sc, mode=1
                    )
                mxfp4_quantize_split2_col_only_opt_launch_inplace(
                    input0,
                    input1,
                    col_fp4,
                    col_sc,
                    mode=1,
                    **_mxfp4_split_col_only_opt_kwargs(role, producer_key),
                )
            return
        if (
            role == "grad"
            and use_mxfp4_split2_persistent_grad_sr()
            and _mxfp4_data_sr_for_role(role)
            and not _mxfp4_scale_sr_for_role(role)
            and not _mxfp4_rht_has_row(role)
            and not _mxfp4_rht_has_col(role)
        ):
            opt_kwargs = _mxfp4_opt_kwargs(role, producer_key)
            mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace(
                input0,
                input1,
                row_fp4,
                row_sc,
                col_fp4,
                col_sc,
                mode=1,
                rng_seed=int(opt_kwargs["rng_seed"]),
                rng_subsequence=int(opt_kwargs["rng_subsequence"]),
            )
            return
        mxfp4_quantize_split2_row_and_col_opt_launch_inplace(
            input0,
            input1,
            row_fp4,
            row_sc,
            col_fp4,
            col_sc,
            **_mxfp4_split_opt_kwargs(role),
        )
    else:
        mxfp4_quantize_split2_row_and_col_launch_inplace(
            input0,
            input1,
            row_fp4,
            row_sc,
            col_fp4,
            col_sc,
        )


def _mxfp4_rmsnorm_backward(
    te_fused,
    d_normed: torch.Tensor,
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    inv_rms: torch.Tensor,
    wait_stream_before_dgamma: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = d_normed.shape
    state = _get_mxfp4_rms_bwd_state(M, K, d_normed.device)
    if (
        use_mxfp4_rms_bwd_split_dgamma()
        and hasattr(te_fused, "fused_rmsnorm_backward_dx_only_out")
        and hasattr(te_fused, "fused_rmsnorm_backward_dgamma_tiled_out")
    ):
        te_fused.fused_rmsnorm_backward_dx_only_out(
            d_normed, input_tensor, norm_weight, inv_rms, state["grad_input"]
        )
        if wait_stream_before_dgamma is not None:
            torch.cuda.current_stream().wait_stream(wait_stream_before_dgamma)
        te_fused.fused_rmsnorm_backward_dgamma_tiled_out(
            d_normed, input_tensor, inv_rms,
            state["grad_norm_partials"], state["grad_norm"]
        )
        return state["grad_input"], state["grad_norm"]

    if hasattr(te_fused, "fused_rmsnorm_backward_out"):
        if wait_stream_before_dgamma is not None:
            torch.cuda.current_stream().wait_stream(wait_stream_before_dgamma)
        te_fused.fused_rmsnorm_backward_out(
            d_normed, input_tensor, norm_weight, inv_rms,
            state["grad_input"], state["grad_norm"]
        )
        return state["grad_input"], state["grad_norm"]

    return te_fused.fused_rmsnorm_backward(d_normed, input_tensor, norm_weight, inv_rms)


_MXFP4_BWD_SIDE_STREAM = None
_MXFP4_FWD_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}
_MXFP4_FWD_W13_STREAM = None


def _get_mxfp4_bwd_side_stream():
    global _MXFP4_BWD_SIDE_STREAM
    if _MXFP4_BWD_SIDE_STREAM is None:
        _MXFP4_BWD_SIDE_STREAM = torch.cuda.Stream()
    return _MXFP4_BWD_SIDE_STREAM


def _get_mxfp4_fwd_side_stream():
    device_index = torch.cuda.current_device()
    stream = _MXFP4_FWD_SIDE_STREAMS.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _MXFP4_FWD_SIDE_STREAMS[device_index] = stream
    return stream


def _get_mxfp4_fwd_w13_stream():
    global _MXFP4_FWD_W13_STREAM
    if _MXFP4_FWD_W13_STREAM is None:
        _MXFP4_FWD_W13_STREAM = torch.cuda.Stream()
    return _MXFP4_FWD_W13_STREAM


def _record_stream_tree(obj, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj.record_stream(stream)
        return
    if isinstance(obj, _MXFP4RowCol):
        _record_stream_tree(obj.row_fp4, stream)
        _record_stream_tree(obj.row_sc, stream)
        _record_stream_tree(obj.col_fp4, stream)
        _record_stream_tree(obj.col_sc, stream)
        return
    if isinstance(obj, _MixedMXLocalCTAGradCarrier):
        _record_stream_tree(
            (
                obj.local_row_fp4,
                obj.local_row_sc,
                obj.local_row_sg,
                obj.mx_col_fp4,
                obj.mx_col_sc,
                obj.keepalive,
            ),
            stream,
        )
        return
    if isinstance(obj, _MixedMXLocalCTASplit2GradCarrier):
        _record_stream_tree(
            (
                obj.local_row_fp4,
                obj.local_row_sc,
                obj.local_row_sg0,
                obj.local_row_sg1,
                obj.mx_col_fp4,
                obj.mx_col_sc,
                obj.keepalive,
            ),
            stream,
        )
        return
    if isinstance(obj, _MixedMXLocalCTAWeightCarrier):
        _record_stream_tree(
            (
                obj.mx_row_fp4,
                obj.mx_row_sc,
                obj.local_col_fp4,
                obj.local_col_sc,
                obj.local_col_sg,
                obj.keepalive,
            ),
            stream,
        )
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _record_stream_tree(value, stream)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _record_stream_tree(value, stream)


def _record_mxfp4_qkv_forward_rows(
    x_q: _MXFP4RowCol,
    w_qkv_q: _MXFP4RowCol,
) -> None:
    """Keep forward-only quantized rows alive until their GEMMs complete."""
    stream = torch.cuda.current_stream(x_q.row_fp4.device)
    _record_stream_tree(
        (x_q.row_fp4, x_q.row_sc, w_qkv_q.row_fp4, w_qkv_q.row_sc),
        stream,
    )


def _retain_mxfp4_qkv_forward_launch(
    device: torch.device,
    *refs: object,
) -> None:
    """Retain launch arguments until their current-stream work completes."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    pending = _MXFP4_QKV_FORWARD_KEEPALIVE_BY_DEVICE.setdefault(
        device_index,
        deque(),
    )
    while pending and pending[0][0].query():
        pending.popleft()

    stream = torch.cuda.current_stream(device)
    done = torch.cuda.Event()
    done.record(stream)
    pending.append((done, refs))


class _MXFP4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        debug_name: str | None = None,
    ):
        trace = use_mxfp4_stage_timing()
        stage_start = _mxfp4_stage_begin("linear_fwd", debug_name) if trace else None
        inp = _as_contiguous_bf16(input)
        w = _as_contiguous_bf16(weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        ctx._lbt_debug_name = debug_name
        # Small routed MoE groups padded only to M=128 hit invalid TK cluster
        # configs for D=H=256. Use 256-row granularity for generic linears.
        Mp = _mxfp4_round_up_256(M)
        Kp = _mxfp4_round_up_128(K)
        # The dgrad GEMM contracts over the padded output dimension. Some TK
        # configs reject 128-only shapes such as DeepSeek wkv_a's 640, so use
        # the safer 256 granularity for padded output features.
        Np = _mxfp4_round_up_256(N)
        needs_padding = (Mp != M) or (Kp != K) or (Np != N)

        if (
            M == 0
            or not _mxfp4_supported(Mp, Kp)
            or not _mxfp4_supported(Np, Kp)
            or (needs_padding and not _mxfp4_linear_padding_enabled())
        ):
            ctx.fast_path = False
            ctx.save_for_backward(inp, weight, bias if bias is not None else torch.empty(0, device=inp.device, dtype=inp.dtype))
            y = F.linear(inp, weight, bias)
            if trace:
                _mxfp4_stage_end("linear_fwd", debug_name, stage_start)
            return y

        ctx.fast_path = True
        ctx.orig_m = M
        ctx.orig_k = K
        ctx.orig_n = N
        ctx.padded_m = Mp
        ctx.padded_k = Kp
        ctx.padded_n = Np
        native_nemotron_padding = (
            K == Kp
            and M == Mp
            and _mxfp4_native_nemotron_padding(M, K, N, Np)
        )
        ctx.native_nemotron_padding = native_nemotron_padding
        if native_nemotron_padding:
            inp_q_src = inp
            w_q_src = w
        elif needs_padding:
            inp_q_src = F.pad(inp, (0, Kp - K, 0, Mp - M)) if (Kp != K or Mp != M) else inp
            w_q_src = F.pad(w, (0, Kp - K, 0, Np - N)) if (Kp != K or Np != N) else w
        else:
            inp_q_src = inp
            w_q_src = w
        quant_start = _mxfp4_stage_begin("linear_fwd_quant", debug_name) if trace else None
        x_q = _quantize_row_col_bf16(inp_q_src, role="activation")
        if native_nemotron_padding:
            w_q = _quantize_row_col_bf16_padded(w_q_src, Np, Kp)
        else:
            w_q = _quantize_weight_row_col_bf16(w_q_src)
        if trace:
            _mxfp4_stage_end("linear_fwd_quant", debug_name, quant_start)
        gemm_start = _mxfp4_stage_begin("linear_fwd_gemm", debug_name) if trace else None
        y = _mxfp4_gemm_linear(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc)
        if Mp != M or Np != N:
            y = y[:M, :N] if native_nemotron_padding else y[:M, :N].contiguous()
        if trace:
            _mxfp4_stage_end("linear_fwd_gemm", debug_name, gemm_start)
        if bias is not None:
            y = y + bias

        ctx.save_for_backward(
            x_q.col_fp4,
            x_q.col_sc,
            w_q.col_fp4,
            w_q.col_sc,
            torch.empty(0, device=inp.device, dtype=inp.dtype) if bias is None else bias,
        )
        if trace:
            _mxfp4_stage_end("linear_fwd", debug_name, stage_start)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        debug_name = getattr(ctx, "_lbt_debug_name", None)
        trace = use_mxfp4_stage_timing()
        stage_start = _mxfp4_stage_begin("linear_bwd", debug_name) if trace else None
        dY = _as_contiguous_bf16(grad_output)

        if not ctx.fast_path:
            inp, weight, bias = ctx.saved_tensors
            grad_input = dY.matmul(weight)
            grad_weight = dY.transpose(0, 1).matmul(inp)
            grad_bias = dY.sum(0) if bias.numel() > 0 else None
            if trace:
                _mxfp4_stage_end("linear_bwd", debug_name, stage_start)
            return grad_input, grad_weight, grad_bias, None

        x_col_fp4, x_col_sc, w_col_fp4, w_col_sc, bias = ctx.saved_tensors
        M = ctx.orig_m
        N = ctx.orig_n
        K = ctx.orig_k
        Mp = ctx.padded_m
        Np = ctx.padded_n
        Kp = ctx.padded_k
        native_nemotron_padding = ctx.native_nemotron_padding
        if native_nemotron_padding:
            dY_q_src = dY
        elif Np != N or Mp != M:
            dY_q_src = F.pad(dY, (0, Np - N, 0, Mp - M))
        else:
            dY_q_src = dY
        quant_start = _mxfp4_stage_begin("linear_bwd_dy_quant", debug_name) if trace else None
        if native_nemotron_padding:
            dY_q = _quantize_row_col_bf16_padded(dY_q_src, Mp, Np)
        else:
            dY_q = _quantize_row_col_bf16(dY_q_src, role="grad")
        if trace:
            _mxfp4_stage_end("linear_bwd_dy_quant", debug_name, quant_start)
        dgrad_start = _mxfp4_stage_begin("linear_bwd_dgrad", debug_name) if trace else None
        grad_input = _mxfp4_gemm_linear(dY_q.row_fp4, dY_q.row_sc, w_col_fp4, w_col_sc)
        if Mp != M or Kp != K:
            grad_input = grad_input[:M, :K].contiguous()
        if trace:
            _mxfp4_stage_end("linear_bwd_dgrad", debug_name, dgrad_start)
        wgrad_start = _mxfp4_stage_begin("linear_bwd_wgrad", debug_name) if trace else None
        grad_weight = _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)
        if Np != N or Kp != K:
            grad_weight = grad_weight[:N, :K].contiguous()
        if trace:
            _mxfp4_stage_end("linear_bwd_wgrad", debug_name, wgrad_start)
        grad_bias = dY.sum(0) if bias.numel() > 0 else None
        if trace:
            _mxfp4_stage_end("linear_bwd", debug_name, stage_start)
        return grad_input, grad_weight, grad_bias, None


class _MXFP4LinearResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        residual: torch.Tensor,
    ):
        inp = _as_contiguous_bf16(input)
        res = _as_contiguous_bf16(residual)
        w = _as_contiguous_bf16(weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        Mp = _mxfp4_round_up_256(M)
        Kp = _mxfp4_round_up_128(K)
        Np = _mxfp4_round_up_256(N)
        needs_padding = (Mp != M) or (Kp != K) or (Np != N)
        if res.shape != (M, N):
            raise RuntimeError(
                f"MXFP4 linear residual shape {tuple(res.shape)} does not match output {(M, N)}"
            )

        if (
            M == 0
            or not _mxfp4_supported(Mp, Kp)
            or not _mxfp4_supported(Np, Kp)
            or (needs_padding and not _mxfp4_linear_padding_enabled())
        ):
            ctx.fast_path = False
            ctx.save_for_backward(
                inp,
                weight,
                torch.empty(0, device=inp.device, dtype=inp.dtype) if bias is None else bias,
            )
            return F.linear(inp, weight, bias) + res

        ctx.fast_path = True
        ctx.orig_m = M
        ctx.orig_k = K
        ctx.orig_n = N
        ctx.padded_m = Mp
        ctx.padded_k = Kp
        ctx.padded_n = Np
        if needs_padding:
            inp_q_src = F.pad(inp, (0, Kp - K, 0, Mp - M)) if (Kp != K or Mp != M) else inp
            w_q_src = F.pad(w, (0, Kp - K, 0, Np - N)) if (Kp != K or Np != N) else w
        else:
            inp_q_src = inp
            w_q_src = w

        x_q = _quantize_row_col_bf16(inp_q_src, role="activation")
        w_q = _quantize_weight_row_col_bf16(w_q_src)
        if needs_padding:
            y = _mxfp4_gemm_linear(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc)
        else:
            y = _mxfp4_gemm_linear_residual(
                x_q.row_fp4,
                x_q.row_sc,
                w_q.row_fp4,
                w_q.row_sc,
                res,
            )
        if needs_padding and (Mp != M or Np != N):
            y = y[:M, :N].contiguous()
        if needs_padding:
            y = y + res
        if bias is not None:
            y = y + bias

        ctx.save_for_backward(
            x_q.col_fp4,
            x_q.col_sc,
            w_q.col_fp4,
            w_q.col_sc,
            torch.empty(0, device=inp.device, dtype=inp.dtype) if bias is None else bias,
        )
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dY = _as_contiguous_bf16(grad_output)

        if not ctx.fast_path:
            inp, weight, bias = ctx.saved_tensors
            grad_input = dY.matmul(weight)
            grad_weight = dY.transpose(0, 1).matmul(inp)
            grad_bias = dY.sum(0) if bias.numel() > 0 else None
            return grad_input, grad_weight, grad_bias, dY

        x_col_fp4, x_col_sc, w_col_fp4, w_col_sc, bias = ctx.saved_tensors
        M = ctx.orig_m
        N = ctx.orig_n
        K = ctx.orig_k
        Mp = ctx.padded_m
        Np = ctx.padded_n
        Kp = ctx.padded_k
        dY_q_src = F.pad(dY, (0, Np - N, 0, Mp - M)) if (Np != N or Mp != M) else dY
        dY_q = _quantize_row_col_bf16(dY_q_src, role="grad")
        grad_input = _mxfp4_gemm_linear(dY_q.row_fp4, dY_q.row_sc, w_col_fp4, w_col_sc)
        if Mp != M or Kp != K:
            grad_input = grad_input[:M, :K].contiguous()
        grad_weight = _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)
        if Np != N or Kp != K:
            grad_weight = grad_weight[:N, :K].contiguous()
        grad_bias = dY.sum(0) if bias.numel() > 0 else None
        return grad_input, grad_weight, grad_bias, dY


class _MXFP4SqReLULinearResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        residual: torch.Tensor | None,
        debug_name: str | None = None,
    ):
        trace = use_mxfp4_stage_timing()
        stage_start = _mxfp4_stage_begin("sqrelu_linear_fwd", debug_name) if trace else None
        inp = _as_contiguous_bf16(input)
        res = _as_contiguous_bf16(residual) if residual is not None else None
        w = _as_contiguous_bf16(weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        ctx.has_residual = res is not None
        ctx._lbt_debug_name = debug_name
        if res is not None and res.shape != (M, N):
            raise RuntimeError(
                f"MXFP4 square-ReLU linear residual shape {tuple(res.shape)} does not match output {(M, N)}"
            )

        if not (_mxfp4_supported(M, K) and _mxfp4_supported(N, K)):
            ctx.fast_path = False
            ctx.save_for_backward(
                inp,
                weight,
                torch.empty(0, device=inp.device, dtype=inp.dtype) if bias is None else bias,
            )
            hidden = sqrelu_fwd(inp)
            y = F.linear(hidden, weight, bias)
            if res is not None:
                y = y + res
            if trace:
                _mxfp4_stage_end("sqrelu_linear_fwd", debug_name, stage_start)
            return y

        ctx.fast_path = True
        quant_start = _mxfp4_stage_begin("sqrelu_linear_fwd_quant", debug_name) if trace else None
        x_q = _sqrelu_quantize_row_col_bf16(inp, role="activation")
        w_q = _quantize_weight_row_col_bf16(w)
        if trace:
            _mxfp4_stage_end("sqrelu_linear_fwd_quant", debug_name, quant_start)
        gemm_start = _mxfp4_stage_begin("sqrelu_linear_fwd_gemm", debug_name) if trace else None
        if (
            res is not None
            and use_mxfp4_residual_fusion_ffn()
            and use_mxfp4_linear_residual_config()
        ):
            y = _mxfp4_gemm_linear_residual(
                x_q.row_fp4,
                x_q.row_sc,
                w_q.row_fp4,
                w_q.row_sc,
                res,
            )
        else:
            y = _mxfp4_gemm_linear(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc)
            if res is not None:
                y = y + res
        if trace:
            _mxfp4_stage_end("sqrelu_linear_fwd_gemm", debug_name, gemm_start)
        if bias is not None:
            y = y + bias

        ctx.save_for_backward(
            inp,
            x_q.col_fp4,
            x_q.col_sc,
            w_q.col_fp4,
            w_q.col_sc,
            torch.empty(0, device=inp.device, dtype=inp.dtype) if bias is None else bias,
        )
        if trace:
            _mxfp4_stage_end("sqrelu_linear_fwd", debug_name, stage_start)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        debug_name = getattr(ctx, "_lbt_debug_name", None)
        trace = use_mxfp4_stage_timing()
        stage_start = _mxfp4_stage_begin("sqrelu_linear_bwd", debug_name) if trace else None
        dY = _as_contiguous_bf16(grad_output)

        if not ctx.fast_path:
            inp, weight, bias = ctx.saved_tensors
            grad_hidden = dY.matmul(weight)
            grad_input = sqrelu_bwd(grad_hidden, inp)
            grad_weight = dY.transpose(0, 1).matmul(sqrelu_fwd(inp))
            grad_bias = dY.sum(0) if bias.numel() > 0 else None
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd", debug_name, stage_start)
            grad_residual = dY if ctx.has_residual else None
            return grad_input, grad_weight, grad_bias, grad_residual, None

        inp, x_col_fp4, x_col_sc, w_col_fp4, w_col_sc, bias = ctx.saved_tensors
        quant_start = _mxfp4_stage_begin("sqrelu_linear_bwd_dy_quant", debug_name) if trace else None
        dY_q = _quantize_row_col_bf16(dY, role="grad")
        if trace:
            _mxfp4_stage_end("sqrelu_linear_bwd_dy_quant", debug_name, quant_start)

        use_wgrad_overlap = (
            dY.size(0) >= 65536
            and use_mxfp4_ffn_wgrad_overlap()
            and use_mxfp4_sqrelu_w2_wgrad_overlap()
            and dY_q.col_fp4.is_cuda
        )
        use_wgrad_after_dgrad_overlap = (
            dY.size(0) >= 65536
            and use_mxfp4_ffn_wgrad_overlap()
            and use_mxfp4_sqrelu_w2_wgrad_after_dgrad_overlap()
            and dY_q.col_fp4.is_cuda
            and not use_wgrad_overlap
        )
        if use_wgrad_overlap:
            wgrad_start = _mxfp4_stage_begin("sqrelu_linear_bwd_wgrad_launch", debug_name) if trace else None
            grad_weight = torch.empty(
                dY_q.col_fp4.size(0),
                x_col_fp4.size(0),
                dtype=torch.bfloat16,
                device=dY.device,
            )
            wgrad_stream = _get_mxfp4_bwd_side_stream()
            wgrad_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(dY_q.col_fp4, wgrad_stream)
            _record_stream_tree(dY_q.col_sc, wgrad_stream)
            _record_stream_tree(x_col_fp4, wgrad_stream)
            _record_stream_tree(x_col_sc, wgrad_stream)
            _record_stream_tree(grad_weight, wgrad_stream)
            with torch.cuda.stream(wgrad_stream):
                _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc, grad_weight)
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd_wgrad_launch", debug_name, wgrad_start)
        elif not use_wgrad_after_dgrad_overlap:
            wgrad_start = _mxfp4_stage_begin("sqrelu_linear_bwd_wgrad", debug_name) if trace else None
            grad_weight = _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd_wgrad", debug_name, wgrad_start)

        if use_wgrad_after_dgrad_overlap:
            dgrad_start = _mxfp4_stage_begin("sqrelu_linear_bwd_dgrad_gemm", debug_name) if trace else None
            grad_hidden = _mxfp4_gemm_linear(
                dY_q.row_fp4,
                dY_q.row_sc,
                w_col_fp4,
                w_col_sc,
            )
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd_dgrad_gemm", debug_name, dgrad_start)

            wgrad_start = _mxfp4_stage_begin("sqrelu_linear_bwd_wgrad_launch_after_dgrad", debug_name) if trace else None
            grad_weight = torch.empty(
                dY_q.col_fp4.size(0),
                x_col_fp4.size(0),
                dtype=torch.bfloat16,
                device=dY.device,
            )
            wgrad_stream = _get_mxfp4_bwd_side_stream()
            wgrad_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(dY_q.col_fp4, wgrad_stream)
            _record_stream_tree(dY_q.col_sc, wgrad_stream)
            _record_stream_tree(x_col_fp4, wgrad_stream)
            _record_stream_tree(x_col_sc, wgrad_stream)
            _record_stream_tree(grad_weight, wgrad_stream)
            with torch.cuda.stream(wgrad_stream):
                _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc, grad_weight)
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd_wgrad_launch_after_dgrad", debug_name, wgrad_start)

            deriv_start = _mxfp4_stage_begin("sqrelu_linear_bwd_deriv", debug_name) if trace else None
            grad_input = sqrelu_bwd(grad_hidden, inp)
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd_deriv", debug_name, deriv_start)
        else:
            dgrad_start = _mxfp4_stage_begin("sqrelu_linear_bwd_dgrad", debug_name) if trace else None
            grad_input = _mxfp4_gemm_linear_sqrelu_deriv(
                dY_q.row_fp4,
                dY_q.row_sc,
                w_col_fp4,
                w_col_sc,
                inp,
            )
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd_dgrad", debug_name, dgrad_start)

        if use_wgrad_overlap or use_wgrad_after_dgrad_overlap:
            wgrad_wait_start = _mxfp4_stage_begin("sqrelu_linear_bwd_wgrad_wait", debug_name) if trace else None
            torch.cuda.current_stream().wait_stream(wgrad_stream)
            if trace:
                _mxfp4_stage_end("sqrelu_linear_bwd_wgrad_wait", debug_name, wgrad_wait_start)
        grad_bias = dY.sum(0) if bias.numel() > 0 else None
        if trace:
            _mxfp4_stage_end("sqrelu_linear_bwd", debug_name, stage_start)
        grad_residual = dY if ctx.has_residual else None
        return grad_input, grad_weight, grad_bias, grad_residual, None


class _MXFP4PrequantInputLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        x_row_fp4: torch.Tensor,
        x_row_sc: torch.Tensor,
        x_col_fp4: torch.Tensor,
        x_col_sc: torch.Tensor,
        padded_m: int,
        padded_k: int,
    ):
        inp = _as_contiguous_bf16(input)
        w = _as_contiguous_bf16(weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        Mp = _mxfp4_round_up_256(M)
        Kp = _mxfp4_round_up_128(K)
        Np = _mxfp4_round_up_256(N)
        needs_padding = (Mp != M) or (Kp != K) or (Np != N)

        if (
            int(padded_m) != Mp
            or int(padded_k) != Kp
            or M == 0
            or not _mxfp4_supported(Mp, Kp)
            or not _mxfp4_supported(Np, Kp)
            or (needs_padding and not _mxfp4_linear_padding_enabled())
        ):
            ctx.fast_path = False
            ctx.save_for_backward(inp, weight, bias if bias is not None else torch.empty(0, device=inp.device, dtype=inp.dtype))
            return F.linear(inp, weight, bias)

        if (
            not x_row_fp4.is_contiguous()
            or not x_row_sc.is_contiguous()
            or not x_col_fp4.is_contiguous()
            or not x_col_sc.is_contiguous()
        ):
            ctx.fast_path = False
            ctx.save_for_backward(inp, weight, bias if bias is not None else torch.empty(0, device=inp.device, dtype=inp.dtype))
            return F.linear(inp, weight, bias)

        ctx.fast_path = True
        ctx.orig_m = M
        ctx.orig_k = K
        ctx.orig_n = N
        ctx.padded_m = Mp
        ctx.padded_k = Kp
        ctx.padded_n = Np
        w_q_src = F.pad(w, (0, Kp - K, 0, Np - N)) if (Kp != K or Np != N) else w
        w_q = _quantize_weight_row_col_bf16(w_q_src)
        y = mxfp4_gemm(x_row_fp4, x_row_sc, w_q.row_fp4, w_q.row_sc)
        if Mp != M or Np != N:
            y = y[:M, :N].contiguous()
        if bias is not None:
            y = y + bias

        ctx.save_for_backward(
            x_col_fp4,
            x_col_sc,
            w_q.col_fp4,
            w_q.col_sc,
            torch.empty(0, device=inp.device, dtype=inp.dtype) if bias is None else bias,
        )
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dY = _as_contiguous_bf16(grad_output)

        if not ctx.fast_path:
            inp, weight, bias = ctx.saved_tensors
            grad_input = dY.matmul(weight)
            grad_weight = dY.transpose(0, 1).matmul(inp)
            grad_bias = dY.sum(0) if bias.numel() > 0 else None
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

        x_col_fp4, x_col_sc, w_col_fp4, w_col_sc, bias = ctx.saved_tensors
        M = ctx.orig_m
        N = ctx.orig_n
        K = ctx.orig_k
        Mp = ctx.padded_m
        Np = ctx.padded_n
        Kp = ctx.padded_k
        dY_q_src = F.pad(dY, (0, Np - N, 0, Mp - M)) if (Np != N or Mp != M) else dY
        dY_q = _quantize_row_col_bf16(dY_q_src, role="grad")
        grad_input = mxfp4_gemm(dY_q.row_fp4, dY_q.row_sc, w_col_fp4, w_col_sc)
        if Mp != M or Kp != K:
            grad_input = grad_input[:M, :K].contiguous()
        grad_weight = mxfp4_gemm(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)
        if Np != N or Kp != K:
            grad_weight = grad_weight[:N, :K].contiguous()
        grad_bias = dY.sum(0) if bias.numel() > 0 else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class _MXFP4RMSNormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
        debug_name: str | None = None,
    ):
        inp = _as_contiguous_bf16(input)
        w = _as_contiguous_bf16(weight.detach())
        nw = _as_contiguous_bf16(norm_weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        Np = _mxfp4_round_up_256(N)
        needs_output_padding = Np != N
        te_fused = _get_te_fused()

        if (
            M == 0
            or not _mxfp4_supported(M, K)
            or not _mxfp4_supported(Np, K)
            or (needs_output_padding and not _mxfp4_linear_padding_enabled())
        ):
            ctx.fast_path = False
            normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
            ctx.save_for_backward(inp, nw, inv_rms, weight)
            ctx.epsilon = epsilon
            ctx._mxfp4_debug_name = debug_name
            return normed.matmul(weight.t())

        ctx.fast_path = True
        ctx.orig_n = N
        ctx.padded_n = Np
        native_nemotron_padding = _mxfp4_native_nemotron_padding(
            M, K, N, Np
        )
        ctx.native_nemotron_padding = native_nemotron_padding
        if native_nemotron_padding:
            w_q_src = w
        elif needs_output_padding:
            w_q_src = F.pad(w, (0, 0, 0, Np - N))
        else:
            w_q_src = w
        x_q, inv_rms = _rmsnorm_quantize_row_col_bf16(
            te_fused,
            inp,
            nw,
            float(epsilon),
            kind="qkv",
        )
        w_q = (
            _quantize_row_col_bf16_padded(w_q_src, Np, K)
            if native_nemotron_padding
            else _quantize_weight_row_col_bf16(w_q_src)
        )
        y = (
            _mxfp4_gemm_linear(
                x_q.row_fp4,
                x_q.row_sc,
                w_q.row_fp4,
                w_q.row_sc,
            )
            if native_nemotron_padding
            else _mxfp4_gemm_qkv(
                x_q.row_fp4,
                x_q.row_sc,
                w_q.row_fp4,
                w_q.row_sc,
            )
        )
        if needs_output_padding:
            y = y[:, :N] if native_nemotron_padding else y[:, :N].contiguous()
        ctx.save_for_backward(
            inp,
            nw,
            inv_rms,
            x_q.col_fp4,
            x_q.col_sc,
            w_q.col_fp4,
            w_q.col_sc,
        )
        ctx.epsilon = epsilon
        ctx._mxfp4_debug_name = debug_name
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dY = _as_contiguous_bf16(grad_output)
        te_fused = _get_te_fused()
        if not ctx.fast_path:
            inp, nw, inv_rms, weight = ctx.saved_tensors
            normed, _ = te_fused.fused_rmsnorm_only(inp, nw, float(ctx.epsilon))
            dx_normed = dY.matmul(weight)
            grad_w = dY.transpose(0, 1).matmul(normed)
            grad_input, grad_norm = _mxfp4_rmsnorm_backward(
                te_fused,
                dx_normed,
                inp,
                nw,
                inv_rms,
            )
            return grad_input, grad_w, grad_norm, None, None

        inp, nw, inv_rms, x_col_fp4, x_col_sc, w_col_fp4, w_col_sc = ctx.saved_tensors
        N = ctx.orig_n
        Np = ctx.padded_n
        native_nemotron_padding = ctx.native_nemotron_padding
        if native_nemotron_padding or Np == N:
            dY_q_src = dY
        else:
            dY_q_src = F.pad(dY, (0, Np - N))
        dY_q = (
            _quantize_row_col_bf16_padded(dY_q_src, dY.size(0), Np)
            if native_nemotron_padding
            else _quantize_row_col_bf16(dY_q_src, role="grad")
        )
        dx_normed = (
            _mxfp4_gemm_linear(
                dY_q.row_fp4,
                dY_q.row_sc,
                w_col_fp4,
                w_col_sc,
            )
            if native_nemotron_padding
            else _mxfp4_gemm_qkv(
                dY_q.row_fp4,
                dY_q.row_sc,
                w_col_fp4,
                w_col_sc,
            )
        )
        grad_w = (
            _mxfp4_gemm_wgrad(
                dY_q.col_fp4,
                dY_q.col_sc,
                x_col_fp4,
                x_col_sc,
            )
            if native_nemotron_padding
            else mxfp4_gemm(
                dY_q.col_fp4,
                dY_q.col_sc,
                x_col_fp4,
                x_col_sc,
            )
        )
        if Np != N:
            grad_w = grad_w[:N].contiguous()
        grad_input, grad_norm = _mxfp4_rmsnorm_backward(
            te_fused,
            dx_normed,
            inp,
            nw,
            inv_rms,
        )
        return grad_input, grad_w, grad_norm, None, None


class MXFP4LinearTK(nn.Linear):
    """Benchmark-only MXFP4 linear autograd built on the existing MXFP4 backend."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        _safe_trunc_normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_is_dtensor = isinstance(x, DTensor)
        weight_is_dtensor = isinstance(self.weight, DTensor)
        bias_is_dtensor = isinstance(self.bias, DTensor)
        input_dtensor = x if input_is_dtensor else None
        weight_dtensor = self.weight if weight_is_dtensor else None

        if input_is_dtensor:
            x = x.to_local()
        weight = self.weight.to_local() if weight_is_dtensor else self.weight
        bias = self.bias.to_local() if bias_is_dtensor else self.bias

        is_nd = x.dim() > 2
        if is_nd:
            orig_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        y = _MXFP4LinearFunction.apply(
            x, weight, bias, getattr(self, "_lbt_debug_name", None)
        )
        if is_nd:
            y = y.reshape(*orig_shape, y.shape[-1])

        if input_is_dtensor or weight_is_dtensor:
            if weight_dtensor is not None:
                device_mesh = weight_dtensor.device_mesh
                placements = weight_dtensor.placements
            elif input_dtensor is not None:
                device_mesh = input_dtensor.device_mesh
                placements = input_dtensor.placements
            else:
                return y

            if any(isinstance(p, Shard) and p.dim == 0 for p in placements):
                output_placements = (Shard(-1),)
            elif any(isinstance(p, Shard) and p.dim in (1, -1) for p in placements):
                output_placements = (Partial(),)
            else:
                output_placements = (Replicate(),)
            return DTensor.from_local(
                y,
                device_mesh,
                output_placements,
                run_check=False,
            )
        return y

    def invalidate_weight_cache(self):
        clear_mxfp4_weight_quant_cache()

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "MXFP4LinearTK":
        out = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        if linear.weight.device.type != "meta":
            with torch.no_grad():
                out.weight.copy_(linear.weight)
                if linear.bias is not None:
                    out.bias.copy_(linear.bias)
        return out


class MXFP4RMSNormLinearTK(nn.Module):
    """MXFP4 linear with a fused RMSNorm-to-quant producer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.norm_weight = nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.norm_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_nd = x.dim() > 2
        if is_nd:
            orig_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        y = _MXFP4RMSNormLinearFunction.apply(
            x,
            self.weight,
            self.norm_weight,
            float(self.eps),
            getattr(self, "_lbt_debug_name", None),
        )
        if is_nd:
            y = y.reshape(*orig_shape, self.out_features)
        return y


def _deepseek_mla_reorder_wq_pe_first(
    wq: torch.Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    head_dim = qk_nope_head_dim + qk_rope_head_dim
    wq_h = wq.view(n_heads, head_dim, wq.shape[-1])
    return torch.cat(
        (wq_h[:, qk_nope_head_dim:, :], wq_h[:, :qk_nope_head_dim, :]),
        dim=1,
    ).reshape_as(wq)


def _deepseek_mla_apply_inverse_rope_first(
    grad_q: torch.Tensor,
    freqs_cis: torch.Tensor,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == head_dim:
        from torchtitan.models.deepseek_v3.model.model import apply_rotary_emb

        q = grad_q.view(batch_size, seq_len, n_heads, head_dim)
        q = apply_rotary_emb(q, freqs_cis[:seq_len].conj())
        return q.reshape_as(grad_q)

    from torchtitan.models.deepseek_v3.model.model import apply_rotary_emb

    q = grad_q.view(batch_size, seq_len, n_heads, head_dim)
    q_rope = apply_rotary_emb(q[..., :rope_dim], freqs_cis[:seq_len].conj())
    return torch.cat((q_rope, q[..., rope_dim:]), dim=-1).reshape_as(grad_q)


def _deepseek_mla_apply_inverse_rope_kpe(
    grad_kpe: torch.Tensor,
    freqs_cis: torch.Tensor,
    batch_size: int,
    seq_len: int,
    rope_dim: int,
) -> torch.Tensor:
    from torchtitan.models.deepseek_v3.model.model import apply_rotary_emb

    kpe = grad_kpe.view(batch_size, seq_len, 1, rope_dim)
    kpe = apply_rotary_emb(kpe, freqs_cis[:seq_len].conj())
    return kpe.reshape_as(grad_kpe)


class _DeepSeekMLAAttentionWoFunction_MXFP4_TK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        wo_weight: torch.Tensor,
        softmax_scale: float,
        debug_name: str | None = None,
    ):
        from low_bits_training.models import _load_tk_b300_mla_fwd

        b300_mha_fwd = _load_tk_b300_mla_fwd()
        out, lse = b300_mha_fwd(
            q,
            k,
            v,
            causal=True,
            softmax_scale=float(softmax_scale),
            return_lse=True,
        )

        B, S, H, Dv = out.shape
        flat = _as_contiguous_bf16(out.reshape(B * S, H * Dv))
        w = _as_contiguous_bf16(wo_weight.detach())
        x_q = _quantize_row_col_bf16(flat, role="activation")
        w_q = _quantize_weight_row_col_bf16(w)
        y = _mxfp4_gemm_qkv(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc)
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            x_q.col_fp4,
            x_q.col_sc,
            w_q.col_fp4,
            w_q.col_sc,
        )
        ctx.softmax_scale = float(softmax_scale)
        ctx._mxfp4_debug_name = debug_name
        ctx.shape_info = (B, S, H, Dv)
        return y.view(B, S, w.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, out, lse, x_col_fp4, x_col_sc, w_col_fp4, w_col_sc = ctx.saved_tensors
        B, S, H, Dv = ctx.shape_info
        dY = _as_contiguous_bf16(grad_output.reshape(B * S, -1))
        dY_q = _quantize_row_col_bf16(dY, role="grad")
        grad_attn = _mxfp4_gemm_qkv(dY_q.row_fp4, dY_q.row_sc, w_col_fp4, w_col_sc)
        grad_w = mxfp4_gemm(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)

        dout = grad_attn.view(B, S, H, Dv)
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()
        out_bhsd = out.transpose(1, 2).contiguous()
        dout_bhsd = dout.transpose(1, 2).contiguous()
        lse_bhs = lse.permute(0, 2, 1).contiguous()
        pad = q.size(-1) - v.size(-1)
        v_pad = F.pad(v_bhsd, (0, pad)).contiguous()
        out_pad = F.pad(out_bhsd, (0, pad)).contiguous()
        dout_pad = F.pad(dout_bhsd, (0, pad)).contiguous()
        philox_seed = torch.empty(0, dtype=torch.uint64, device=q.device)
        philox_offset = torch.empty(0, dtype=torch.uint64, device=q.device)
        dq, dk, dv_pad = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dout_pad,
            q_bhsd,
            k_bhsd,
            v_pad,
            out_pad,
            lse_bhs,
            None,
            None,
            q.size(1),
            k.size(1),
            0.0,
            True,
            philox_seed,
            philox_offset,
            scale=ctx.softmax_scale,
        )
        return (
            dq.transpose(1, 2).contiguous(),
            dk.transpose(1, 2).contiguous(),
            dv_pad[..., :Dv].transpose(1, 2).contiguous(),
            grad_w,
            None,
            None,
        )


def _maybe_tk_b300_mla_attention_wo_bshd(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    wo_weight: torch.Tensor,
    softmax_scale: float,
    use_flex_attn: bool,
    attention_masks,
) -> torch.Tensor | None:
    """Fuse the TK B300 MLA forward wrapper with immediate MXFP4 WO quant/GEMM."""
    if (
        not use_mxfp4_deepseek_mla_fused_attn_wo()
        or not use_low_bits_tk_b300_mla_attention()
        or use_flex_attn
        or attention_masks is not None
        or q.device.type != "cuda"
        or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or wo_weight.dtype != torch.bfloat16
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:-1] != k.shape[:-1]
        or q.shape[:-1] != v.shape[:-1]
        or q.size(-1) != 192
        or k.size(-1) != 192
        or v.size(-1) != 128
        or q.size(1) < 2048
    ):
        return None
    B, S, H, Dv = v.shape
    K = H * Dv
    M = B * S
    N = wo_weight.shape[0]
    if wo_weight.shape[1] != K or not (_mxfp4_supported(M, K) and _mxfp4_supported(N, K)):
        return None
    try:
        return _DeepSeekMLAAttentionWoFunction_MXFP4_TK.apply(
            q,
            k,
            v,
            wo_weight,
            float(softmax_scale),
            f"{getattr(module, '_lbt_debug_name', module.__class__.__name__)}:attn_wo",
        )
    except Exception as exc:
        if not getattr(module, "_warned_tk_b300_mla_wo_fallback", False):
            device = torch.cuda.current_device() if torch.cuda.is_available() else -1
            sys.stdout.write(
                f"[Rank{device}] WARNING: fused TK B300 MLA+WO failed. "
                f"Fallback to split attention/WO. Error: {exc}\n"
            )
            module._warned_tk_b300_mla_wo_fallback = True
        return None


def _maybe_tk_b300_mla_attention_bshd(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    use_flex_attn: bool,
    attention_masks,
) -> torch.Tensor | None:
    """Route DeepSeek MLA's native BSHD layout to the TK B300 forward path."""
    if (
        not use_low_bits_tk_b300_mla_attention()
        or use_flex_attn
        or attention_masks is not None
        or q.device.type != "cuda"
        or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:-1] != k.shape[:-1]
        or q.shape[:-1] != v.shape[:-1]
        or q.size(-1) != 192
        or k.size(-1) != 192
        or v.size(-1) != 128
        or q.size(1) < 2048
    ):
        return None
    try:
        from low_bits_training.models import tk_b300_mla_attention_bshd

        return tk_b300_mla_attention_bshd(q, k, v, softmax_scale)
    except Exception as exc:
        if not getattr(module, "_warned_tk_b300_mla_bshd_fallback", False):
            device = torch.cuda.current_device() if torch.cuda.is_available() else -1
            sys.stdout.write(
                f"[Rank{device}] WARNING: direct BSHD TK B300 MLA failed. "
                f"Fallback to module attention. Error: {exc}\n"
            )
            module._warned_tk_b300_mla_bshd_fallback = True
        return None


class _MXFP4DeepSeekMLAInputProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        freqs_cis: torch.Tensor,
        batch_size: int,
        seq_len: int,
        n_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        debug_name: str | None = None,
    ):
        inp = _as_contiguous_bf16(input)
        w = _as_contiguous_bf16(weight.detach())
        M, K = inp.shape
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        q_dim = n_heads * qk_head_dim
        kpe_dim = qk_rope_head_dim
        kv_pad_dim = _mxfp4_batched_gemm_dim(kv_lora_rank)
        kpe_pad_dim = _mxfp4_batched_gemm_dim(kpe_dim)
        padded_n = q_dim + kv_pad_dim + kpe_pad_dim
        needs_padding = padded_n != w.shape[0]

        if (
            M == 0
            or not _mxfp4_supported(M, K)
            or not _mxfp4_supported(q_dim, K)
            or not _mxfp4_supported(kv_pad_dim, K)
            or not _mxfp4_supported(kpe_pad_dim, K)
            or (needs_padding and not _mxfp4_linear_padding_enabled())
        ):
            ctx.fast_path = False
            y = F.linear(inp, w)
            q = y[:, :q_dim].contiguous()
            kv = y[:, q_dim:q_dim + kv_lora_rank].contiguous()
            kpe = y[:, q_dim + kv_lora_rank:].contiguous()
            from torchtitan.models.deepseek_v3.model.model import apply_rotary_emb

            q4 = q.view(batch_size, seq_len, n_heads, qk_head_dim)
            q_rope = apply_rotary_emb(q4[..., :qk_rope_head_dim], freqs_cis[:seq_len])
            q = torch.cat((q_rope, q4[..., qk_rope_head_dim:]), dim=-1).reshape_as(q)
            kpe = apply_rotary_emb(kpe.view(batch_size, seq_len, 1, kpe_dim), freqs_cis[:seq_len]).reshape_as(kpe)
            ctx.save_for_backward(inp, weight, freqs_cis)
            ctx.batch_size = batch_size
            ctx.seq_len = seq_len
            ctx.n_heads = n_heads
            ctx.qk_nope_head_dim = qk_nope_head_dim
            ctx.qk_rope_head_dim = qk_rope_head_dim
            ctx.kv_lora_rank = kv_lora_rank
            ctx._mxfp4_debug_name = debug_name
            return q, kv, kpe

        ctx.fast_path = True
        w_q_src = F.pad(w, (0, 0, 0, padded_n - w.shape[0])) if needs_padding else w
        x_q = _quantize_row_col_bf16(inp, role="activation")
        w_q = _quantize_weight_row_col_bf16(w_q_src)

        q = torch.empty(M, q_dim, dtype=torch.bfloat16, device=inp.device)
        kv = torch.empty(M, kv_pad_dim, dtype=torch.bfloat16, device=inp.device)
        kpe = torch.empty(M, kpe_pad_dim, dtype=torch.bfloat16, device=inp.device)
        q_blocks = q_dim // 128
        kv_blocks = kv_pad_dim // 128
        rope_cos, rope_sin = _get_mxfp4_rope_tables(freqs_cis, seq_len)
        rope_empty = torch.empty(0, dtype=torch.float32, device=inp.device)
        row_fp4 = w_q.row_fp4
        row_sc = w_q.row_sc
        mxfp4_batched_gemm_rope(
            [x_q.row_fp4, x_q.row_fp4, x_q.row_fp4],
            [x_q.row_sc, x_q.row_sc, x_q.row_sc],
            [
                row_fp4[:q_dim],
                row_fp4[q_dim:q_dim + kv_pad_dim],
                row_fp4[q_dim + kv_pad_dim:q_dim + kv_pad_dim + kpe_pad_dim],
            ],
            [
                row_sc[:q_blocks],
                row_sc[q_blocks:q_blocks + kv_blocks],
                row_sc[q_blocks + kv_blocks:q_blocks + kv_blocks + (kpe_pad_dim // 128)],
            ],
            [rope_cos, rope_empty, rope_cos],
            [rope_sin, rope_empty, rope_sin],
            [seq_len, 0, seq_len],
            [qk_head_dim, 0, kpe_dim],
            [kpe_dim, 0, kpe_dim],
            [q, kv, kpe],
        )
        ctx.save_for_backward(
            x_q.col_fp4,
            x_q.col_sc,
            w_q.col_fp4,
            w_q.col_sc,
            freqs_cis,
        )
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.n_heads = n_heads
        ctx.qk_nope_head_dim = qk_nope_head_dim
        ctx.qk_rope_head_dim = qk_rope_head_dim
        ctx.kv_lora_rank = kv_lora_rank
        ctx.kv_pad_dim = kv_pad_dim
        ctx.kpe_pad_dim = kpe_pad_dim
        ctx.padded_n = padded_n
        ctx._mxfp4_debug_name = debug_name
        return q, kv[:, :kv_lora_rank].contiguous(), kpe[:, :kpe_dim].contiguous()

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_kv: torch.Tensor, grad_kpe: torch.Tensor):
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        n_heads = ctx.n_heads
        qk_nope_head_dim = ctx.qk_nope_head_dim
        qk_rope_head_dim = ctx.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_lora_rank = ctx.kv_lora_rank
        q_dim = n_heads * qk_head_dim

        if not ctx.fast_path:
            inp, weight, freqs_cis = ctx.saved_tensors
            gq = _deepseek_mla_apply_inverse_rope_first(
                _as_contiguous_bf16(grad_q),
                freqs_cis,
                batch_size,
                seq_len,
                n_heads,
                qk_head_dim,
                qk_rope_head_dim,
            )
            gkpe = _deepseek_mla_apply_inverse_rope_kpe(
                _as_contiguous_bf16(grad_kpe),
                freqs_cis,
                batch_size,
                seq_len,
                qk_rope_head_dim,
            )
            dY = torch.cat([gq, _as_contiguous_bf16(grad_kv), gkpe], dim=1)
            grad_input = dY.matmul(weight)
            grad_w = dY.transpose(0, 1).matmul(inp)
            return grad_input, grad_w, None, None, None, None, None, None, None, None

        x_col_fp4, x_col_sc, w_col_fp4, w_col_sc, freqs_cis = ctx.saved_tensors
        kv_pad_dim = ctx.kv_pad_dim
        kpe_pad_dim = ctx.kpe_pad_dim
        padded_n = ctx.padded_n
        gq = _deepseek_mla_apply_inverse_rope_first(
            _as_contiguous_bf16(grad_q),
            freqs_cis,
            batch_size,
            seq_len,
            n_heads,
            qk_head_dim,
            qk_rope_head_dim,
        )
        gkv = _as_contiguous_bf16(grad_kv)
        gkpe = _deepseek_mla_apply_inverse_rope_kpe(
            _as_contiguous_bf16(grad_kpe),
            freqs_cis,
            batch_size,
            seq_len,
            qk_rope_head_dim,
        )
        if kv_pad_dim != kv_lora_rank:
            gkv = F.pad(gkv, (0, kv_pad_dim - kv_lora_rank))
        if kpe_pad_dim != qk_rope_head_dim:
            gkpe = F.pad(gkpe, (0, kpe_pad_dim - qk_rope_head_dim))
        dY = torch.cat([gq, gkv, gkpe], dim=1)
        dY_q = _quantize_row_col_bf16(dY, role="grad")
        grad_input = _mxfp4_gemm_qkv(dY_q.row_fp4, dY_q.row_sc, w_col_fp4, w_col_sc)
        grad_w = mxfp4_gemm(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)
        grad_w = torch.cat(
            [
                grad_w[:q_dim],
                grad_w[q_dim:q_dim + kv_lora_rank],
                grad_w[q_dim + kv_pad_dim:q_dim + kv_pad_dim + qk_rope_head_dim],
            ],
            dim=0,
        ).contiguous()
        return grad_input, grad_w, None, None, None, None, None, None, None, None


class MXFP4DeepSeekMLAInputProjTK(nn.Module):
    """DeepSeek MLA input projection with Q/K RoPE in MXFP4 GEMM epilogues.

    Stored Q rows are permuted from the model layout `[nope, pe]` to
    `[pe, nope]` per head so the existing RoPE epilogue can rotate the first
    `qk_rope_head_dim` columns of each head block.  The attention dot product
    remains equivalent because K uses the same `[pe, nope]` layout.
    """

    def __init__(
        self,
        in_features: int,
        n_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_dim = n_heads * self.qk_head_dim
        self.out_features = self.q_dim + kv_lora_rank + qk_rope_head_dim
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, batch_size: int, seq_len: int):
        is_nd = x.dim() > 2
        if is_nd:
            x = x.reshape(-1, x.shape[-1])
        q, kv, kpe = _MXFP4DeepSeekMLAInputProjFunction.apply(
            x,
            self.weight,
            freqs_cis,
            int(batch_size),
            int(seq_len),
            int(self.n_heads),
            int(self.qk_nope_head_dim),
            int(self.qk_rope_head_dim),
            int(self.kv_lora_rank),
            getattr(self, "_lbt_debug_name", None),
        )
        if is_nd:
            q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
            kv = kv.view(batch_size, seq_len, self.kv_lora_rank)
            kpe = kpe.view(batch_size, seq_len, 1, self.qk_rope_head_dim)
        return q, kv, kpe

    @classmethod
    def from_attention(cls, attention) -> "MXFP4DeepSeekMLAInputProjTK":
        out = cls(
            attention.wq.in_features,
            attention.n_heads,
            attention.qk_nope_head_dim,
            attention.qk_rope_head_dim,
            attention.kv_lora_rank,
            device=attention.wq.weight.device,
            dtype=attention.wq.weight.dtype,
        )
        if attention.wq.weight.device.type != "meta":
            with torch.no_grad():
                out.weight[:out.q_dim].copy_(
                    _deepseek_mla_reorder_wq_pe_first(
                        attention.wq.weight,
                        attention.n_heads,
                        attention.qk_nope_head_dim,
                        attention.qk_rope_head_dim,
                    )
                )
                out.weight[out.q_dim:].copy_(attention.wkv_a.weight)
        return out


def _mxfp4_deepseek_grouped_batched_enabled() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_BATCHED", "1") != "0"


def _mxfp4_deepseek_grouped_debug_nan() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_DEBUG_NAN", "0") == "1"


def _mxfp4_deepseek_grouped_min_k_padding() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_MIN_K_PAD", "1") != "0"


def _mxfp4_deepseek_grouped_pad_batched_outputs() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_PAD_BATCHED_OUTPUTS", "0") != "0"


def _mxfp4_deepseek_grouped_bulk_weight_quant() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_BULK_WEIGHT_QUANT", "1") != "0"


def _mxfp4_deepseek_grouped_bulk_activation_row_quant(num_active_experts: int) -> bool:
    raw = os.environ.get("MXFP4_DEEPSEEK_GROUPED_BULK_ACT_ROW_QUANT", "auto").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    # The fallback is per-expert F.pad + row/col quantization launch fan-out.
    # For DeepSeek MoE it is better to keep the pack+quant producer fused even
    # when routing happens to leave only a subset of experts active.
    return num_active_experts > 0


def _mxfp4_deepseek_grouped_bulk_col_slice() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_BULK_COL_SLICE", "1") != "0"


def _mxfp4_deepseek_grouped_bulk_silu_quant() -> bool:
    return os.environ.get(
        "MXFP4_DEEPSEEK_GROUPED_BULK_SILU_QUANT",
        "1" if _mxfp4_v4_quant_extension() else "0",
    ) != "0"


def _mxfp4_deepseek_grouped_bulk_silu_deriv_quant() -> bool:
    return os.environ.get(
        "MXFP4_DEEPSEEK_GROUPED_BULK_SILU_DERIV_QUANT",
        "1" if _mxfp4_v4_quant_extension() else "0",
    ) != "0"


def _mxfp4_deepseek_grouped_strided_gemm() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_STRIDED_GEMM", "1") != "0"


def _mxfp4_deepseek_grouped_pack_kernel() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_PACK_KERNEL", "1") != "0"


def _mxfp4_deepseek_grouped_fused_pack_rowcol_quant() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_FUSED_PACK_ROWCOL_QUANT", "0") != "0"


def _mxfp4_deepseek_fused_moe_combine() -> bool:
    return os.environ.get(
        "MXFP4_DEEPSEEK_FUSED_MOE_COMBINE",
        "1" if _mxfp4_v4_quant_extension() else "0",
    ) != "0"


def _mxfp4_deepseek_shared_routed_combined_x_quant() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_SHARED_ROUTED_COMBINED_X_QUANT", "0") != "0"


def _mxfp4_deepseek_shared_routed_backend_pack() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_SHARED_ROUTED_BACKEND_PACK", "0") != "0"


def _mxfp4_deepseek_indexed_scaled_dy_quant() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_INDEXED_SCALED_DY_QUANT", "1") != "0"


def _mxfp4_deepseek_indexed_x_quant() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_INDEXED_X_QUANT", "0") != "0"


def _mxfp4_deepseek_indexed_x_pack() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_INDEXED_X_PACK", "1") != "0"


def _mxfp4_deepseek_indexed_x_rmsnorm_quant() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_INDEXED_X_RMSNORM_QUANT", "1") != "0"


def _mxfp4_deepseek_variable_indexed_producer() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_VARIABLE_INDEXED_PRODUCER", "0") != "0"


def _mxfp4_deepseek_variable_indexed_pack() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_VARIABLE_INDEXED_PACK", "1") != "0"


def _mxfp4_deepseek_fused_indexed_dy_dot_pack() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_FUSED_INDEXED_DY_DOT_PACK", "0") != "0"


def _mxfp4_deepseek_unsorted_score_gather() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_UNSORTED_SCORE_GATHER", "1") != "0"


def _mxfp4_deepseek_padded_moe_combine() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_PADDED_MOE_COMBINE", "1") != "0"


def _mxfp4_deepseek_fused_combine_kernels() -> bool:
    return os.environ.get(
        "MXFP4_DEEPSEEK_FUSED_COMBINE_KERNELS",
        "1" if _mxfp4_v4_quant_extension() else "0",
    ) != "0"


def _mxfp4_deepseek_fused_gradx_scatter() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_FUSED_GRADX_SCATTER", "1") != "0"


def _mxfp4_deepseek_route_inverse_combine() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_ROUTE_INVERSE_COMBINE", "1") != "0"


def _mxfp4_deepseek_route_inverse_cache() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_ROUTE_INVERSE_CACHE", "0") != "0"


def _mxfp4_materialize_indexed_scaled_dy(
    grad_output: torch.Tensor,
    token_indices: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    gathered = grad_output.index_select(0, token_indices)
    return (gathered.float() * scores.reshape(-1, 1)).to(torch.bfloat16).contiguous()


def _mxfp4_narrow_features_contiguous_bf16(tensor: torch.Tensor, cols: int) -> torch.Tensor:
    if tensor.shape[1] == cols:
        return _as_contiguous_bf16(tensor)
    return _as_contiguous_bf16(tensor[:, :cols].contiguous())


def _mxfp4_deepseek_grouped_packed_w13_param() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_PACKED_W13_PARAM", "1") != "0"


def _mxfp4_deepseek_grouped_split_w13_wgrad() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_SPLIT_W13_WGRAD", "1") != "0"


def _mxfp4_deepseek_grouped_split_w13_grad_kernel() -> bool:
    return os.environ.get("MXFP4_DEEPSEEK_GROUPED_SPLIT_W13_GRAD_KERNEL", "1") != "0"


def _mxfp4_deepseek_grouped_output_dim(value: int) -> int:
    if _mxfp4_deepseek_grouped_pad_batched_outputs():
        return _mxfp4_batched_gemm_dim(value)
    return _mxfp4_round_up_128(value)


def _mxfp4_deepseek_grouped_m_granularity() -> int:
    raw = os.environ.get("MXFP4_DEEPSEEK_GROUPED_M_GRANULARITY", "256")
    try:
        granularity = int(raw)
    except ValueError:
        granularity = 256
    if granularity not in (256, 512, 1024):
        granularity = 256
    return granularity


def _mxfp4_deepseek_grouped_max_batched_gemm(bucket_size: int) -> int:
    raw = os.environ.get("MXFP4_DEEPSEEK_GROUPED_MAX_BATCHED_GEMM", "auto").strip().lower()
    if raw == "auto":
        return 32 if bucket_size > 16 else 8
    try:
        value = int(raw)
    except ValueError:
        value = 32 if bucket_size > 16 else 8
    return max(1, min(value, 32))


def _mxfp4_deepseek_grouped_batched_min_n() -> int:
    raw = os.environ.get("MXFP4_DEEPSEEK_GROUPED_BATCHED_MIN_N", "512").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 512
    if value <= 128:
        return 128
    if value <= 256:
        return 256
    return 512


def _mxfp4_deepseek_grouped_m_pad(tokens: int) -> int:
    return _mxfp4_round_up(tokens, _mxfp4_deepseek_grouped_m_granularity())


def _mxfp4_check_finite(name: str, tensor: torch.Tensor) -> None:
    if not _mxfp4_deepseek_grouped_debug_nan():
        return
    torch.cuda.synchronize()
    if bool(torch.isfinite(tensor).all().item()):
        return
    t = tensor.detach().float()
    nan_count = int(torch.isnan(t).sum().item())
    posinf_count = int(torch.isposinf(t).sum().item())
    neginf_count = int(torch.isneginf(t).sum().item())
    finite = t[torch.isfinite(t)]
    max_abs = float(finite.abs().max().item()) if finite.numel() else float("nan")
    raise RuntimeError(
        f"Non-finite MXFP4 grouped tensor {name}: shape={tuple(tensor.shape)} "
        f"nan={nan_count} +inf={posinf_count} -inf={neginf_count} finite_max_abs={max_abs}"
    )


def _pad_2d_bf16(tensor: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if tensor.shape == (rows, cols):
        return _as_contiguous_bf16(tensor)
    return F.pad(_as_contiguous_bf16(tensor), (0, cols - tensor.shape[1], 0, rows - tensor.shape[0]))


def _mxfp4_slice_row_quant_by_rows(
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    row_start: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_block_start = row_start // 128
    row_blocks = rows // 128
    return (
        row_fp4[row_start:row_start + rows],
        row_sc[row_block_start:row_block_start + row_blocks],
    )


def _mxfp4_slice_col_quant_by_rows(
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    row_start: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_start = row_start // 2
    packed_cols = rows // 2
    row_block_start = row_start // 128
    row_blocks = rows // 128
    return (
        col_fp4[:, packed_start:packed_start + packed_cols],
        col_sc[:, row_block_start:row_block_start + row_blocks],
    )


def _mxfp4_slice_col_quant_lists_by_rows(
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    row_starts: list[int],
    rows: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    fp4_list = []
    sc_list = []
    for row_start, row_count in zip(row_starts, rows, strict=True):
        fp4, sc = _mxfp4_slice_col_quant_by_rows(col_fp4, col_sc, row_start, row_count)
        fp4_list.append(fp4)
        sc_list.append(sc)
    return fp4_list, sc_list


def _mxfp4_build_shared_routed_x_quant(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    counts: torch.Tensor,
) -> _MXFP4SharedRoutedXQuant | None:
    if (
        not _mxfp4_deepseek_shared_routed_combined_x_quant()
        or not _mxfp4_deepseek_grouped_bulk_col_slice()
        or _mxfp4_needs_opt_quant("activation")
        or _mxfp4_rht_for_role("activation")
        or token_indices.dim() != 1
        or x.dim() != 2
        or not x.is_cuda
    ):
        return None

    x_bf16 = _as_contiguous_bf16(x)
    M, D = x_bf16.shape
    if M <= 0 or D <= 0:
        return None

    counts_list = counts.to(dtype=torch.int64).tolist()
    active = [(idx, int(c)) for idx, c in enumerate(counts_list) if int(c) > 0]
    if not active:
        return None

    Kp_shared = _mxfp4_round_up_128(D)
    Dn = _mxfp4_deepseek_grouped_output_dim(D)
    Dk = _mxfp4_round_up_128(D) if _mxfp4_deepseek_grouped_min_k_padding() else Dn
    if Dk != Kp_shared or not _mxfp4_supported(_mxfp4_round_up_256(M), Kp_shared):
        return None

    starts_plan = []
    live_rows = []
    padded_starts_plan = []
    m_padded_plan = []
    scan_offset = 0
    padded_scan_offset = 0
    for _, count in active:
        starts_plan.append(scan_offset)
        live_rows.append(count)
        padded_starts_plan.append(padded_scan_offset)
        m_padded_plan.append(_mxfp4_deepseek_grouped_m_pad(count))
        scan_offset += count
        padded_scan_offset += m_padded_plan[-1]
    if scan_offset != int(token_indices.numel()):
        return None

    shared_rows_padded = _mxfp4_round_up_256(M)
    routed_rows_padded = sum(m_padded_plan)
    token_indices = token_indices.to(dtype=torch.int64).contiguous()
    combined = None
    if _mxfp4_deepseek_shared_routed_backend_pack():
        try:
            combined = mxfp4_pack_shared_routed_rows_bf16(
                x_bf16,
                token_indices,
                starts_plan,
                live_rows,
                padded_starts_plan,
                m_padded_plan,
                shared_rows_padded,
                Dk,
            )
        except (AttributeError, FileNotFoundError, ImportError):
            combined = None
    if combined is None:
        combined = x_bf16.new_zeros((shared_rows_padded + routed_rows_padded, Dk))
        combined[:M, :D].copy_(x_bf16)
        token_indices_expanded = token_indices.reshape(-1, 1).expand(-1, D)
        routed_live = torch.gather(x_bf16, dim=0, index=token_indices_expanded)
        routed_packed = _mxfp4_pack_grouped_rows(
            routed_live,
            starts_plan,
            live_rows,
            m_padded_plan,
            Dk,
        )
        combined[shared_rows_padded:].copy_(routed_packed)
    q = _quantize_row_col_bf16(combined, role="activation")
    shared_row_fp4, shared_row_sc = _mxfp4_slice_row_quant_by_rows(
        q.row_fp4,
        q.row_sc,
        0,
        shared_rows_padded,
    )
    try:
        shared_col_fp4_list, shared_col_sc_list = mxfp4_copy_col_slices(
            q.col_fp4,
            q.col_sc,
            [0],
            [shared_rows_padded],
        )
        shared_col_fp4, shared_col_sc = shared_col_fp4_list[0], shared_col_sc_list[0]
    except (AttributeError, FileNotFoundError, ImportError):
        shared_col_fp4, shared_col_sc = _mxfp4_slice_col_quant_by_rows(
            q.col_fp4,
            q.col_sc,
            0,
            shared_rows_padded,
        )
    routed_row_fp4, routed_row_sc = _mxfp4_slice_row_quant_by_rows(
        q.row_fp4,
        q.row_sc,
        shared_rows_padded,
        routed_rows_padded,
    )
    routed_col_fp4, routed_col_sc = _mxfp4_slice_col_quant_by_rows(
        q.col_fp4,
        q.col_sc,
        shared_rows_padded,
        routed_rows_padded,
    )
    return _MXFP4SharedRoutedXQuant(
        shared=_MXFP4RowCol(
            row_fp4=shared_row_fp4,
            row_sc=shared_row_sc,
            col_fp4=shared_col_fp4,
            col_sc=shared_col_sc,
        ),
        routed=_MXFP4RowCol(
            row_fp4=routed_row_fp4,
            row_sc=routed_row_sc,
            col_fp4=routed_col_fp4,
            col_sc=routed_col_sc,
        ),
        shared_rows_padded=shared_rows_padded,
        routed_rows_padded=routed_rows_padded,
        cols_padded=Dk,
    )


def _quantize_row_bf16(
    tensor: torch.Tensor,
    mode: int = 1,
    role: str = "activation",
    producer_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = _as_contiguous_bf16(tensor)
    oriented_sr = _mxfp4_oriented_grad_data_sr(role)
    if oriented_sr == "col":
        if _mxfp4_rht_has_row(role):
            return mxfp4_quantize_for_gemm_opt_rht(
                tensor,
                mode,
                **_mxfp4_no_sr_rht_kwargs(role),
            )
        return mxfp4_quantize_for_gemm(tensor, mode)
    if _mxfp4_rht_has_row(role):
        return mxfp4_quantize_for_gemm_opt_rht(
            tensor,
            mode,
            **_mxfp4_rht_kwargs(role, producer_key),
        )
    if _mxfp4_needs_opt_quant(role):
        return mxfp4_quantize_for_gemm_opt(
            tensor,
            mode,
            **_mxfp4_opt_kwargs(role, producer_key),
        )
    return mxfp4_quantize_for_gemm(tensor, mode)


def _quantize_col_bf16(
    tensor: torch.Tensor,
    mode: int = 1,
    role: str = "activation",
    producer_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = _as_contiguous_bf16(tensor)
    oriented_sr = _mxfp4_oriented_grad_data_sr(role)
    if oriented_sr == "row":
        if _mxfp4_rht_has_col(role):
            return mxfp4_quantize_col_only_opt_rht(
                tensor,
                mode,
                **_mxfp4_no_sr_rht_kwargs(role),
            )
        return mxfp4_quantize_col_only(tensor, mode)
    if _mxfp4_rht_has_col(role):
        return mxfp4_quantize_col_only_opt_rht(
            tensor,
            mode,
            **_mxfp4_rht_kwargs(role, producer_key),
        )
    if _mxfp4_needs_opt_quant(role):
        return mxfp4_quantize_col_only_opt(
            tensor,
            mode,
            **_mxfp4_opt_kwargs(role, producer_key),
        )
    return mxfp4_quantize_col_only(tensor, mode)


def _mxfp4_pack_w13_for_grouped(w1: torch.Tensor, w3: torch.Tensor, H13n: int, Dk: int) -> torch.Tensor:
    E, H, D = w1.shape
    H13 = 2 * H
    if _mxfp4_deepseek_grouped_pack_kernel() and w1.is_cuda:
        try:
            return mxfp4_pack_w13_bf16(w1, w3, H13n, Dk)
        except AttributeError:
            pass
    w13 = torch.cat((w1, w3), dim=1)
    if H13n == H13 and Dk == D:
        return w13.reshape(E * H13, D)
    packed = w13.new_zeros((E, H13n, Dk))
    packed[:, :H13, :D] = w13
    return packed.reshape(E * H13n, Dk)


def _mxfp4_pack_w2_for_grouped(w2: torch.Tensor, Dn: int, Hk: int) -> torch.Tensor:
    E, D, H = w2.shape
    if Dn == D and Hk == H:
        return w2.reshape(E * D, H)
    packed = w2.new_zeros((E, Dn, Hk))
    packed[:, :D, :H] = w2
    return packed.reshape(E * Dn, Hk)


def _mxfp4_pack_grouped_rows(
    tensor: torch.Tensor,
    starts: list[int],
    live_rows: list[int],
    padded_rows: list[int],
    cols: int,
) -> torch.Tensor:
    total_padded = sum(padded_rows)
    total_live = sum(live_rows)
    if total_padded == total_live and tensor.shape[1] == cols:
        return _as_contiguous_bf16(tensor[:total_live])
    if _mxfp4_deepseek_grouped_pack_kernel() and tensor.is_cuda:
        try:
            return mxfp4_pack_grouped_rows_bf16(
                _as_contiguous_bf16(tensor),
                starts,
                live_rows,
                padded_rows,
                cols,
            )
        except AttributeError:
            pass
    packed = tensor.new_zeros((total_padded, cols))
    dst = 0
    for src_start, rows, rows_padded in zip(starts, live_rows, padded_rows, strict=True):
        if rows > 0:
            packed[dst:dst + rows, :tensor.shape[1]] = tensor[src_start:src_start + rows, :cols]
        dst += rows_padded
    return packed


def _mxfp4_pack_grouped_rows_quantize_uniform(
    tensor: torch.Tensor,
    starts: list[int],
    live_rows: list[int],
    padded_rows: list[int],
    cols: int,
    mode: int = 1,
) -> _MXFP4RowCol | None:
    if (
        not _mxfp4_deepseek_grouped_fused_pack_rowcol_quant()
        or not _mxfp4_deepseek_grouped_pack_kernel()
        or not tensor.is_cuda
        or not live_rows
    ):
        return None
    live0 = int(live_rows[0])
    padded0 = int(padded_rows[0])
    if (
        live0 <= 0
        or padded0 < live0
        or cols < tensor.shape[1]
        or tensor.shape[1] % 128 != 0
        or cols % 128 != 0
        or padded0 % 128 != 0
        or any(int(x) != live0 for x in live_rows)
        or any(int(x) != padded0 for x in padded_rows)
        or any(int(start) != idx * live0 for idx, start in enumerate(starts))
    ):
        return None
    try:
        row_fp4, row_sc, col_fp4, col_sc = mxfp4_pack_grouped_rows_quantize_row_and_col(
            _as_contiguous_bf16(tensor),
            len(live_rows),
            live0,
            padded0,
            cols,
            mode,
        )
    except AttributeError:
        return None
    return _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)


def _mxfp4_grouped_gemm_bucketed(
    a_fp4: list[torch.Tensor],
    a_sc: list[torch.Tensor],
    b_fp4: list[torch.Tensor],
    b_sc: list[torch.Tensor],
    out: list[torch.Tensor],
) -> None:
    """Run the variable-M MXFP4 batched launcher for routed experts."""

    max_batched = _mxfp4_deepseek_grouped_max_batched_gemm(len(out))
    for start in range(0, len(out), max_batched):
        chunk = list(range(start, min(start + max_batched, len(out))))
        has_tma_view = any(
            not a_fp4[i].is_contiguous()
            or not a_sc[i].is_contiguous()
            or not b_fp4[i].is_contiguous()
            or not b_sc[i].is_contiguous()
            for i in chunk
        )
        batched_supported = all(
            out[i].shape[0] % 256 == 0
            and out[i].shape[1] % 128 == 0
            # TP2 shards DeepSeek's 1408 expert hidden width to 704, padded to
            # 768 here. The variable batched GEMM can produce non-finite
            # W2-backward hidden grads for that width; the single GEMM path is
            # finite, so keep this shape off the batched route until the TK
            # launcher is fixed.
            and out[i].shape[1] != 768
            and (out[i].shape[1] >= _mxfp4_deepseek_grouped_batched_min_n() or has_tma_view)
            for i in chunk
        )
        if (len(chunk) == 1 and not has_tma_view) or not batched_supported:
            for i in chunk:
                out[i].copy_(
                    mxfp4_gemm(
                        a_fp4[i],
                        a_sc[i].contiguous(),
                        b_fp4[i],
                        b_sc[i].contiguous(),
                    )
                )
            continue
        mxfp4_batched_gemm(
            [a_fp4[i] for i in chunk],
            [a_sc[i] for i in chunk],
            [b_fp4[i] for i in chunk],
            [b_sc[i] for i in chunk],
            [out[i] for i in chunk],
        )


class _MXFP4GroupedExpertsBatchedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, counts: torch.Tensor):
        x = _as_contiguous_bf16(x)
        w1_bf16 = _as_contiguous_bf16(w1.detach())
        w2_bf16 = _as_contiguous_bf16(w2.detach())
        packed_w13_param = w3.numel() == 0 and w1_bf16.ndim == 3 and w1_bf16.shape[1] % 2 == 0
        if packed_w13_param:
            w13_param_bf16 = w1_bf16
            E, H13, D = w13_param_bf16.shape
            H = H13 // 2
            w1_bf16 = w13_param_bf16[:, :H, :]
            w3_bf16 = w13_param_bf16[:, H:H13, :]
        else:
            w13_param_bf16 = None
            w3_bf16 = _as_contiguous_bf16(w3.detach())
        counts_list = counts.to(dtype=torch.int64).tolist()
        active = [(idx, int(c)) for idx, c in enumerate(counts_list) if int(c) > 0]
        if not active:
            ctx.active = []
            ctx.packed_w13_param = packed_w13_param
            ctx.save_for_backward(counts)
            return torch.zeros_like(x)

        if not packed_w13_param:
            E, H, D = w1_bf16.shape
            H13 = 2 * H
        # Keep GEMM contraction dimensions as narrow as the MXFP4 block
        # contract allows. The batched kernel needs conservative output widths
        # for small DeepSeek debug shapes, but padding K to those widths changes
        # the quantization problem and can destabilize trainer numerics.
        Dn = _mxfp4_deepseek_grouped_output_dim(D)
        Hn = _mxfp4_deepseek_grouped_output_dim(H)
        H13n = _mxfp4_deepseek_grouped_output_dim(H13)
        # TP can shard DeepSeek's 2816-wide packed W1/W3 projection to 1408
        # rows. The MXFP4 batched GEMM wgrad path tiles output rows by 256,
        # so keep the internal W13 workspace on that boundary and slice the
        # live H/H13 rows when returning gradients.
        H13n = _mxfp4_round_up_256(H13n)
        if _mxfp4_deepseek_grouped_min_k_padding():
            Dk = _mxfp4_round_up_128(D)
            Hk = _mxfp4_round_up_128(H)
        else:
            Dk = Dn
            Hk = Hn

        starts_plan = []
        m_padded_plan = []
        padded_starts_plan = []
        scan_offset = 0
        padded_scan_offset = 0
        for _, c in active:
            starts_plan.append(scan_offset)
            padded_starts_plan.append(padded_scan_offset)
            padded_rows = _mxfp4_deepseek_grouped_m_pad(c)
            m_padded_plan.append(padded_rows)
            scan_offset += c
            padded_scan_offset += padded_rows
        uniform_active = (
            len(active) == E
            and all(expert_idx == idx for idx, (expert_idx, _) in enumerate(active))
            and len(set(m_padded_plan)) == 1
        )
        skip_bulk_col_lists = (
            _mxfp4_deepseek_grouped_strided_gemm()
            and _mxfp4_deepseek_grouped_bulk_activation_row_quant(len(active))
            and _mxfp4_deepseek_grouped_bulk_weight_quant()
            and _mxfp4_deepseek_grouped_bulk_col_slice()
            and _mxfp4_deepseek_grouped_bulk_silu_quant()
            and use_mxfp4_deepseek_grouped_fused_silu_quant()
            and uniform_active
            and Hk == Hn == H
            and Dk == Dn
            and Dn >= 512
            and Hn >= 512
            and H13n >= 512
            and not _mxfp4_needs_opt_quant("activation")
            and not _mxfp4_needs_opt_quant("weight")
            and not _mxfp4_rht_for_role("weight")
        )

        w13_bulk_src = None
        w2_bulk_src = None
        w13_bulk_row_fp4 = None
        w13_bulk_row_sc = None
        w13_bulk_col_fp4 = None
        w13_bulk_col_sc = None
        w13_bulk_col_fp4_list = None
        w13_bulk_col_sc_list = None
        w2_bulk_row_fp4 = None
        w2_bulk_row_sc = None
        w2_bulk_col_fp4 = None
        w2_bulk_col_sc = None
        w2_bulk_col_fp4_list = None
        w2_bulk_col_sc_list = None
        if (
            _mxfp4_deepseek_grouped_bulk_weight_quant()
            and not _mxfp4_needs_opt_quant("weight")
            and not _mxfp4_rht_for_role("weight")
            and Dk == Dn
            and Hk == Hn
            and _mxfp4_deepseek_grouped_bulk_col_slice()
        ):
            w2_bulk_src = _mxfp4_pack_w2_for_grouped(w2_bf16, Dn, Hk)
            if packed_w13_param and H13n == H13 and Dk == D:
                w13_bulk_src = w13_param_bf16.reshape(E * H13, D)
            else:
                w13_bulk_src = _mxfp4_pack_w13_for_grouped(w1_bf16, w3_bf16, H13n, Dk)
            w13_bulk_q = _quantize_weight_row_col_bf16(w13_bulk_src)
            w2_bulk_q = _quantize_weight_row_col_bf16(w2_bulk_src)
            w13_bulk_row_fp4, w13_bulk_row_sc = w13_bulk_q.row_fp4, w13_bulk_q.row_sc
            w13_bulk_col_fp4, w13_bulk_col_sc = w13_bulk_q.col_fp4, w13_bulk_q.col_sc
            w2_bulk_row_fp4, w2_bulk_row_sc = w2_bulk_q.row_fp4, w2_bulk_q.row_sc
            w2_bulk_col_fp4, w2_bulk_col_sc = w2_bulk_q.col_fp4, w2_bulk_q.col_sc
            if not skip_bulk_col_lists:
                w13_bulk_col_fp4_list, w13_bulk_col_sc_list = _mxfp4_slice_col_quant_lists_by_rows(
                    w13_bulk_col_fp4,
                    w13_bulk_col_sc,
                    [expert_idx * H13n for expert_idx, _ in active],
                    [H13n for _ in active],
                )
                w2_bulk_col_fp4_list, w2_bulk_col_sc_list = _mxfp4_slice_col_quant_lists_by_rows(
                    w2_bulk_col_fp4,
                    w2_bulk_col_sc,
                    [expert_idx * Dn for expert_idx, _ in active],
                    [Dn for _ in active],
                )

        x_bulk_src = None
        x_bulk_row_fp4 = None
        x_bulk_row_sc = None
        x_bulk_col_fp4 = None
        x_bulk_col_sc = None
        x_bulk_col_fp4_list = None
        x_bulk_col_sc_list = None
        indexed_x_base = getattr(ctx, "_mxfp4_indexed_x_base", None)
        indexed_x_tokens = getattr(ctx, "_mxfp4_indexed_x_tokens", None)
        indexed_x_scores = getattr(ctx, "_mxfp4_indexed_x_scores", None)
        indexed_x_rms_base = getattr(ctx, "_mxfp4_indexed_x_rms_base", None)
        indexed_x_rms_weight = getattr(ctx, "_mxfp4_indexed_x_rms_weight", None)
        indexed_x_rms_inv = getattr(ctx, "_mxfp4_indexed_x_rms_inv", None)
        indexed_x_dummy = bool(getattr(ctx, "_mxfp4_indexed_x_dummy", False))
        prequant_x_bulk_q = getattr(ctx, "_mxfp4_prequant_x_bulk_q", None)
        if (
            _mxfp4_deepseek_grouped_bulk_activation_row_quant(len(active))
            and not _mxfp4_needs_opt_quant("activation")
            and not _mxfp4_rht_for_role("activation")
            and Dk == Dn
            and _mxfp4_deepseek_grouped_bulk_col_slice()
        ):
            x_live_rows = [c for _, c in active]
            x_bulk_q = None
            if prequant_x_bulk_q is not None:
                x_bulk_q = prequant_x_bulk_q
            if (
                indexed_x_base is not None
                and indexed_x_tokens is not None
                and indexed_x_scores is not None
                and _mxfp4_deepseek_indexed_x_quant()
                and x_bulk_q is None
            ):
                try:
                    if (
                        uniform_active
                        and len(set(x_live_rows)) == 1
                        and len(set(m_padded_plan)) == 1
                        and starts_plan == [i * x_live_rows[0] for i in range(len(active))]
                    ):
                        row_fp4, row_sc, col_fp4, col_sc = mxfp4_pack_indexed_scaled_rows_quantize_row_and_col(
                            _as_contiguous_bf16(indexed_x_base),
                            indexed_x_tokens.contiguous(),
                            indexed_x_scores.contiguous(),
                            len(active),
                            x_live_rows[0],
                            m_padded_plan[0],
                            Dk,
                            1,
                        )
                    elif _mxfp4_deepseek_variable_indexed_producer():
                        row_fp4, row_sc, col_fp4, col_sc = (
                            mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable(
                                _as_contiguous_bf16(indexed_x_base),
                                indexed_x_tokens.contiguous(),
                                indexed_x_scores.contiguous(),
                                starts_plan,
                                x_live_rows,
                                padded_starts_plan,
                                m_padded_plan,
                                Dk,
                                1,
                            )
                        )
                    else:
                        raise AttributeError("variable indexed producer disabled")
                    x_bulk_q = _MXFP4RowCol(
                        row_fp4=row_fp4,
                        row_sc=row_sc,
                        col_fp4=col_fp4,
                        col_sc=col_sc,
                    )
                except (AttributeError, FileNotFoundError, ImportError):
                    x_bulk_q = None
            if (
                x_bulk_q is None
                and indexed_x_rms_base is not None
                and indexed_x_rms_weight is not None
                and indexed_x_rms_inv is not None
                and indexed_x_tokens is not None
                and indexed_x_dummy
                and _mxfp4_deepseek_indexed_x_rmsnorm_quant()
            ):
                try:
                    if (
                        uniform_active
                        and len(set(x_live_rows)) == 1
                        and len(set(m_padded_plan)) == 1
                        and starts_plan == [i * x_live_rows[0] for i in range(len(active))]
                    ):
                        row_fp4, row_sc, col_fp4, col_sc = mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col(
                            _as_contiguous_bf16(indexed_x_rms_base),
                            _as_contiguous_bf16(indexed_x_rms_weight),
                            indexed_x_rms_inv.contiguous(),
                            indexed_x_tokens.contiguous(),
                            len(active),
                            x_live_rows[0],
                            m_padded_plan[0],
                            Dk,
                            1,
                        )
                    elif _mxfp4_deepseek_variable_indexed_producer():
                        row_fp4, row_sc, col_fp4, col_sc = (
                            mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col_variable(
                                _as_contiguous_bf16(indexed_x_rms_base),
                                _as_contiguous_bf16(indexed_x_rms_weight),
                                indexed_x_rms_inv.contiguous(),
                                indexed_x_tokens.contiguous(),
                                starts_plan,
                                x_live_rows,
                                padded_starts_plan,
                                m_padded_plan,
                                Dk,
                                1,
                            )
                        )
                    else:
                        raise AttributeError("variable indexed producer disabled")
                    x_bulk_q = _MXFP4RowCol(
                        row_fp4=row_fp4,
                        row_sc=row_sc,
                        col_fp4=col_fp4,
                        col_sc=col_sc,
                    )
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    x_bulk_q = None
            if (
                x_bulk_q is None
                and indexed_x_base is not None
                and indexed_x_tokens is not None
                and indexed_x_dummy
                and _mxfp4_deepseek_indexed_x_pack()
                and uniform_active
                and len(set(x_live_rows)) == 1
                and len(set(m_padded_plan)) == 1
                and starts_plan == [i * x_live_rows[0] for i in range(len(active))]
            ):
                try:
                    x_bulk_src = mxfp4_pack_indexed_rows_bf16(
                        _as_contiguous_bf16(indexed_x_base),
                        indexed_x_tokens.contiguous(),
                        len(active),
                        x_live_rows[0],
                        m_padded_plan[0],
                        Dk,
                    )
                    x_bulk_q = _quantize_row_col_bf16(x_bulk_src, role="activation")
                except (AttributeError, FileNotFoundError, ImportError):
                    x_bulk_q = None
            if (
                x_bulk_q is None
                and indexed_x_base is not None
                and indexed_x_tokens is not None
                and indexed_x_scores is not None
                and indexed_x_dummy
                and _mxfp4_deepseek_indexed_x_pack()
                and _mxfp4_deepseek_variable_indexed_pack()
            ):
                try:
                    x_bulk_src = mxfp4_pack_indexed_scaled_rows_bf16_variable(
                        _as_contiguous_bf16(indexed_x_base),
                        indexed_x_tokens.contiguous(),
                        indexed_x_scores.contiguous(),
                        starts_plan,
                        x_live_rows,
                        padded_starts_plan,
                        m_padded_plan,
                        Dk,
                    )
                    x_bulk_q = _quantize_row_col_bf16(x_bulk_src, role="activation")
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    x_bulk_q = None
            if (
                x_bulk_q is None
                and indexed_x_rms_base is not None
                and indexed_x_rms_weight is not None
                and indexed_x_rms_inv is not None
                and indexed_x_tokens is not None
                and indexed_x_dummy
                and _mxfp4_deepseek_indexed_x_pack()
                and _mxfp4_deepseek_variable_indexed_pack()
            ):
                try:
                    x_bulk_src = mxfp4_pack_indexed_rmsnorm_rows_bf16_variable(
                        _as_contiguous_bf16(indexed_x_rms_base),
                        _as_contiguous_bf16(indexed_x_rms_weight),
                        indexed_x_rms_inv.contiguous(),
                        indexed_x_tokens.contiguous(),
                        starts_plan,
                        x_live_rows,
                        padded_starts_plan,
                        m_padded_plan,
                        Dk,
                    )
                    x_bulk_q = _quantize_row_col_bf16(x_bulk_src, role="activation")
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    x_bulk_q = None
            if x_bulk_q is None and indexed_x_dummy:
                token_indices_expanded = indexed_x_tokens.reshape(-1, 1).expand(-1, indexed_x_base.shape[1])
                x = torch.gather(_as_contiguous_bf16(indexed_x_base), dim=0, index=token_indices_expanded)
            if x_bulk_q is None:
                x_bulk_q = _mxfp4_pack_grouped_rows_quantize_uniform(
                    x,
                    starts_plan,
                    x_live_rows,
                    m_padded_plan,
                    Dk,
                )
            if x_bulk_q is None:
                x_bulk_src = _mxfp4_pack_grouped_rows(
                    x,
                    starts_plan,
                    x_live_rows,
                    m_padded_plan,
                    Dk,
                )
                x_bulk_q = _quantize_row_col_bf16(x_bulk_src, role="activation")
            x_bulk_row_fp4, x_bulk_row_sc = x_bulk_q.row_fp4, x_bulk_q.row_sc
            x_bulk_col_fp4, x_bulk_col_sc = x_bulk_q.col_fp4, x_bulk_q.col_sc
            if not skip_bulk_col_lists:
                x_bulk_col_fp4_list, x_bulk_col_sc_list = _mxfp4_slice_col_quant_lists_by_rows(
                    x_bulk_col_fp4,
                    x_bulk_col_sc,
                    padded_starts_plan,
                    m_padded_plan,
                )
        if indexed_x_dummy and x_bulk_row_fp4 is None:
            token_indices_expanded = indexed_x_tokens.reshape(-1, 1).expand(-1, indexed_x_base.shape[1])
            x = torch.gather(_as_contiguous_bf16(indexed_x_base), dim=0, index=token_indices_expanded)

        x_row_q = []
        x_row_sc = []
        w13_row_q = []
        w13_row_sc = []
        h13_outs = []
        h1_list = []
        h3_list = []
        h_col_q = []
        h_col_sc = []
        w2_row_q = []
        w2_row_sc = []
        y_outs = []
        saved = []
        starts = []
        use_bulk_silu_quant = (
            _mxfp4_deepseek_grouped_bulk_silu_quant()
            and use_mxfp4_deepseek_grouped_fused_silu_quant()
            and Hk == Hn == H
            and not _mxfp4_needs_opt_quant("activation")
        )
        h13_bulk = (
            torch.empty(sum(m_padded_plan), H13n, dtype=torch.bfloat16, device=x.device)
            if use_bulk_silu_quant
            else None
        )
        use_strided_w13_gemm = (
            _mxfp4_deepseek_grouped_strided_gemm()
            and uniform_active
            and h13_bulk is not None
            and x_bulk_row_fp4 is not None
            and w13_bulk_row_fp4 is not None
        )
        use_fast_strided_bulk = (
            use_strided_w13_gemm
            and w2_bulk_row_fp4 is not None
            and w2_bulk_col_fp4 is not None
            and w2_bulk_col_sc is not None
            and x_bulk_col_fp4 is not None
            and x_bulk_col_sc is not None
            and w13_bulk_col_fp4 is not None
            and w13_bulk_col_sc is not None
            and Hk == Hn == H
            and H13n == 2 * H
            and not _mxfp4_needs_opt_quant("grad")
            and not _mxfp4_rht_for_role("grad")
        )
        if use_fast_strided_bulk:
            Mp0 = m_padded_plan[0]
            mxfp4_grouped_gemm_strided(
                x_bulk_row_fp4,
                x_bulk_row_sc,
                w13_bulk_row_fp4,
                w13_bulk_row_sc,
                h13_bulk,
                len(active),
                Mp0,
                H13n,
                Dk,
                Mp0,
                0,
                H13n,
                0,
                Mp0,
            )
            row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_mul_quantize_row_and_col_strided(
                h13_bulk,
                H,
                H,
            )
            y_bulk = torch.empty(sum(m_padded_plan), Dn, dtype=torch.bfloat16, device=x.device)
            mxfp4_grouped_gemm_strided(
                row_fp4,
                row_sc,
                w2_bulk_row_fp4,
                w2_bulk_row_sc,
                y_bulk,
                len(active),
                Mp0,
                Dn,
                Hn,
                Mp0,
                0,
                Dn,
                0,
                Mp0,
            )
            skip_y_cat_scatter = (
                bool(getattr(ctx, "_mxfp4_moe_skip_y_cat_scatter", False))
                and scan_offset == x.shape[0]
                and uniform_active
                and len(active) > 0
            )
            y_cat = (
                torch.empty(x.shape[0], D, dtype=torch.bfloat16, device=x.device)
                if scan_offset == x.shape[0]
                else torch.zeros(x.shape[0], D, dtype=torch.bfloat16, device=x.device)
            )
            if skip_y_cat_scatter:
                ctx._mxfp4_grouped_y_bulk = y_bulk
                ctx._mxfp4_grouped_live_rows_per_batch = active[0][1]
                ctx._mxfp4_grouped_padded_rows_per_batch = Mp0
            else:
                ctx._mxfp4_grouped_y_bulk = None
                ctx._mxfp4_grouped_live_rows_per_batch = 0
                ctx._mxfp4_grouped_padded_rows_per_batch = 0
                mxfp4_scatter_grouped_rows_bf16(
                    y_bulk,
                    y_cat,
                    starts_plan,
                    [M for _, M in active],
                    m_padded_plan,
                )
            ctx.save_for_backward(
                counts,
                h13_bulk,
                x_bulk_col_fp4,
                x_bulk_col_sc,
                w13_bulk_col_fp4,
                w13_bulk_col_sc,
                w2_bulk_col_fp4,
                w2_bulk_col_sc,
                col_fp4,
                col_sc,
            )
            ctx.has_h13_bulk = True
            ctx.saved_h13_splits = False
            ctx.bulk_col_saved = True
            ctx.fast_strided_bulk = True
            ctx.packed_w13_param = packed_w13_param
            ctx.active = active
            ctx.starts = starts_plan
            ctx.m_padded = m_padded_plan
            ctx.dims = (E, H, D, Dk, Hk, Dn, Hn, H13, H13n)
            return y_cat

        offset = 0
        padded_offset = 0
        for out_idx, (expert_idx, c) in enumerate(active):
            starts.append(offset)
            M = c
            Mp = m_padded_plan[out_idx]
            expert_x = x[offset:offset + M]
            if x_bulk_row_fp4 is not None:
                row_fp4, row_sc = _mxfp4_slice_row_quant_by_rows(x_bulk_row_fp4, x_bulk_row_sc, padded_offset, Mp)
                col_fp4, col_sc = x_bulk_col_fp4_list[out_idx], x_bulk_col_sc_list[out_idx]
                x_row_quant = _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
                x_col_quant = x_row_quant
            else:
                x_row_src = _pad_2d_bf16(expert_x, Mp, Dk)
                x_col_src = _pad_2d_bf16(expert_x, Mp, Dn)
                x_row_quant = _quantize_row_col_bf16(x_row_src, role="activation")
                x_col_quant = x_row_quant if Dk == Dn else _quantize_row_col_bf16(x_col_src, role="activation")
            if w13_bulk_row_fp4 is not None:
                row_fp4, row_sc = _mxfp4_slice_row_quant_by_rows(
                    w13_bulk_row_fp4,
                    w13_bulk_row_sc,
                    expert_idx * H13n,
                    H13n,
                )
                col_fp4, col_sc = w13_bulk_col_fp4_list[out_idx], w13_bulk_col_sc_list[out_idx]
                w13_row_quant = _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
                w13_col_quant = w13_row_quant
            else:
                w13 = torch.cat((w1_bf16[expert_idx], w3_bf16[expert_idx]), dim=0)
                w13_row_src = _pad_2d_bf16(w13, H13n, Dk)
                w13_col_src = _pad_2d_bf16(w13, H13n, Dn)
                w13_row_quant = _quantize_weight_row_col_bf16(w13_row_src)
                w13_col_quant = w13_row_quant if Dk == Dn else _quantize_weight_row_col_bf16(w13_col_src)
            h13 = (
                h13_bulk[padded_offset:padded_offset + Mp]
                if h13_bulk is not None
                else torch.empty(Mp, H13n, dtype=torch.bfloat16, device=x.device)
            )

            x_row_q.append(x_row_quant.row_fp4)
            x_row_sc.append(x_row_quant.row_sc)
            w13_row_q.append(w13_row_quant.row_fp4)
            w13_row_sc.append(w13_row_quant.row_sc)
            h13_outs.append(h13)
            saved.append((
                M, Mp, expert_idx,
                x_col_quant.col_fp4, x_col_quant.col_sc,
                w13_col_quant.col_fp4, w13_col_quant.col_sc,
            ))
            offset += M
            padded_offset += Mp

        if use_strided_w13_gemm:
            mxfp4_grouped_gemm_strided(
                x_bulk_row_fp4,
                x_bulk_row_sc,
                w13_bulk_row_fp4,
                w13_bulk_row_sc,
                h13_bulk,
                len(active),
                m_padded_plan[0],
                H13n,
                Dk,
                m_padded_plan[0],
                0,
                H13n,
                0,
                m_padded_plan[0],
            )
        else:
            _mxfp4_grouped_gemm_bucketed(x_row_q, x_row_sc, w13_row_q, w13_row_sc, h13_outs)
        h_bulk_quant = None
        if h13_bulk is not None:
            row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_mul_quantize_row_and_col_strided(
                h13_bulk,
                H,
                H,
            )
            h_bulk_quant = _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)

        h_row_q = []
        h_row_sc = []
        h_padded_offset = 0
        use_strided_w2_gemm = (
            _mxfp4_deepseek_grouped_strided_gemm()
            and uniform_active
            and h_bulk_quant is not None
            and w2_bulk_row_fp4 is not None
        )
        y_bulk = (
            torch.empty(sum(m_padded_plan), Dn, dtype=torch.bfloat16, device=x.device)
            if use_strided_w2_gemm
            else None
        )
        save_h13_splits = not (
            h13_bulk is not None
            and _mxfp4_deepseek_grouped_bulk_silu_deriv_quant()
            and Hn == H
            and H13n == 2 * H
            and not _mxfp4_needs_opt_quant("grad")
            and not _mxfp4_rht_for_role("grad")
        )
        for out_idx, (expert_idx, c) in enumerate(active):
            M, Mp, _, x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc = saved[out_idx]
            h13_padded = h13_outs[out_idx][:Mp, :H13]
            h13_live = h13_padded[:M]
            _mxfp4_check_finite(f"grouped_fwd.h13_live[{out_idx}]", h13_live)
            h1 = None
            h3 = None
            if h_bulk_quant is not None:
                row_fp4, row_sc = _mxfp4_slice_row_quant_by_rows(
                    h_bulk_quant.row_fp4,
                    h_bulk_quant.row_sc,
                    h_padded_offset,
                    Mp,
                )
                col_fp4, col_sc = _mxfp4_slice_col_quant_by_rows(
                    h_bulk_quant.col_fp4,
                    h_bulk_quant.col_sc,
                    h_padded_offset,
                    Mp,
                )
                h_row_quant = _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
                h_col_quant = h_row_quant
            else:
                h1_pad, h3_pad = h13_padded.chunk(2, dim=-1)
                h1 = h1_pad[:M]
                h3 = h3_pad[:M]
                fused_silu_opt_supported = (
                _mxfp4_needs_opt_quant("activation")
                and (
                    not _mxfp4_rht_for_role("activation")
                    or use_mxfp4_fused_silu_ffn_quant_rht()
                )
                and (
                    not _mxfp4_data_sr_for_role("activation")
                    or use_mxfp4_fused_silu_ffn_quant_data_sr()
                )
                and (
                    not _mxfp4_scale_sr_for_role("activation")
                    or use_mxfp4_fused_silu_ffn_quant_scale_sr()
                )
                )
                use_grouped_fused_silu = (
                    use_mxfp4_deepseek_grouped_fused_silu_quant()
                    and Hk == Hn == H
                    and (not _mxfp4_needs_opt_quant("activation") or fused_silu_opt_supported)
                )
                if use_grouped_fused_silu:
                    try:
                        h1_fused = h1_pad.contiguous()
                        h3_fused = h3_pad.contiguous()
                        if fused_silu_opt_supported:
                            h_row_quant = _empty_mxfp4_row_col(Mp, H, x.device)
                            opt_kwargs = _mxfp4_opt_kwargs("activation")
                            if _mxfp4_rht_has_row("activation"):
                                opt_kwargs["row_with_rht"] = True
                            mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace(
                                h1_fused,
                                h3_fused,
                                h_row_quant.row_fp4,
                                h_row_quant.row_sc,
                                h_row_quant.col_fp4,
                                h_row_quant.col_sc,
                                1,
                                use_rht=_mxfp4_rht_has_col("activation"),
                                rht_block_size=_mxfp4_rht_block_size(),
                                with_random_sign_mask=_mxfp4_rht_random_sign_mask(),
                                **opt_kwargs,
                            )
                        else:
                            row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_mul_quantize_row_and_col(h1_fused, h3_fused)
                            h_row_quant = _MXFP4RowCol(
                                row_fp4=row_fp4,
                                row_sc=row_sc,
                                col_fp4=col_fp4,
                                col_sc=col_sc,
                            )
                        h_col_quant = h_row_quant
                    except AttributeError:
                        h = (F.silu(h1.float()).to(torch.bfloat16) * h3).contiguous()
                        _mxfp4_check_finite(f"grouped_fwd.h_live[{out_idx}]", h)
                        h_row_src = _pad_2d_bf16(h, Mp, Hk)
                        h_col_src = _pad_2d_bf16(h, Mp, Hn)
                        h_row_quant = _quantize_row_col_bf16(h_row_src, role="activation")
                        h_col_quant = h_row_quant if Hk == Hn else _quantize_row_col_bf16(h_col_src, role="activation")
                else:
                    h = (F.silu(h1.float()).to(torch.bfloat16) * h3).contiguous()
                    _mxfp4_check_finite(f"grouped_fwd.h_live[{out_idx}]", h)
                    h_row_src = _pad_2d_bf16(h, Mp, Hk)
                    h_col_src = _pad_2d_bf16(h, Mp, Hn)
                    h_row_quant = _quantize_row_col_bf16(h_row_src, role="activation")
                    h_col_quant = h_row_quant if Hk == Hn else _quantize_row_col_bf16(h_col_src, role="activation")
            if save_h13_splits and h1 is None:
                h1_pad, h3_pad = h13_padded.chunk(2, dim=-1)
                h1 = h1_pad[:M]
                h3 = h3_pad[:M]
            if w2_bulk_row_fp4 is not None:
                row_fp4, row_sc = _mxfp4_slice_row_quant_by_rows(
                    w2_bulk_row_fp4,
                    w2_bulk_row_sc,
                    expert_idx * Dn,
                    Dn,
                )
                col_fp4, col_sc = w2_bulk_col_fp4_list[out_idx], w2_bulk_col_sc_list[out_idx]
                w2_row_quant = _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
                w2_col_quant = w2_row_quant
            else:
                w2_row_src = _pad_2d_bf16(w2_bf16[expert_idx], Dn, Hk)
                w2_col_src = _pad_2d_bf16(w2_bf16[expert_idx], Dn, Hn)
                w2_row_quant = _quantize_weight_row_col_bf16(w2_row_src)
                w2_col_quant = w2_row_quant if Hk == Hn else _quantize_weight_row_col_bf16(w2_col_src)
            y = (
                y_bulk[h_padded_offset:h_padded_offset + Mp]
                if y_bulk is not None
                else torch.empty(Mp, Dn, dtype=torch.bfloat16, device=x.device)
            )

            if save_h13_splits:
                h1_list.append(h1)
                h3_list.append(h3)
            h_row_q.append(h_row_quant.row_fp4)
            h_row_sc.append(h_row_quant.row_sc)
            h_col_q.append(h_col_quant.col_fp4)
            h_col_sc.append(h_col_quant.col_sc)
            w2_row_q.append(w2_row_quant.row_fp4)
            w2_row_sc.append(w2_row_quant.row_sc)
            y_outs.append(y)
            saved[out_idx] = saved[out_idx] + (
                h_col_quant.col_fp4, h_col_quant.col_sc,
                w2_col_quant.col_fp4, w2_col_quant.col_sc,
            )
            h_padded_offset += Mp

        if use_strided_w2_gemm:
            mxfp4_grouped_gemm_strided(
                h_bulk_quant.row_fp4,
                h_bulk_quant.row_sc,
                w2_bulk_row_fp4,
                w2_bulk_row_sc,
                y_bulk,
                len(active),
                m_padded_plan[0],
                Dn,
                Hn,
                m_padded_plan[0],
                0,
                Dn,
                0,
                m_padded_plan[0],
            )
            if _mxfp4_deepseek_grouped_debug_nan():
                y_offset = 0
                for i, (_, rows) in enumerate(active):
                    y_part = y_bulk[y_offset:y_offset + rows, :D]
                    _mxfp4_check_finite(f"grouped_fwd.y_live[{i}]", y_part)
                    y_offset += m_padded_plan[i]
            y_cat = (
                torch.empty(x.shape[0], D, dtype=torch.bfloat16, device=x.device)
                if scan_offset == x.shape[0]
                else torch.zeros(x.shape[0], D, dtype=torch.bfloat16, device=x.device)
            )
            mxfp4_scatter_grouped_rows_bf16(
                y_bulk,
                y_cat,
                starts,
                [M for _, M in active],
                m_padded_plan,
            )
        else:
            _mxfp4_grouped_gemm_bucketed(h_row_q, h_row_sc, w2_row_q, w2_row_sc, y_outs)
            y_live = [y_outs[i][:active[i][1], :D] for i in range(len(active))]
            if _mxfp4_deepseek_grouped_debug_nan():
                for i, y_part in enumerate(y_live):
                    _mxfp4_check_finite(f"grouped_fwd.y_live[{i}]", y_part)
            y_cat = torch.cat(y_live, dim=0)
            if y_cat.shape[0] < x.shape[0]:
                y_cat = torch.vstack((y_cat, y_cat.new_zeros((x.shape[0] - y_cat.shape[0], D))))

        flat_saved = []
        for idx, item in enumerate(saved):
            M, Mp, expert_idx, x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc, h_col_fp4, h_col_sc, w2_col_fp4, w2_col_sc = item
            flat_saved.extend([x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc])
            if save_h13_splits:
                flat_saved.extend([h1_list[idx], h3_list[idx]])
            flat_saved.extend([h_col_fp4, h_col_sc, w2_col_fp4, w2_col_sc])
        bulk_col_saved = (
            _mxfp4_deepseek_grouped_strided_gemm()
            and uniform_active
            and x_bulk_col_fp4 is not None
            and x_bulk_col_sc is not None
            and w13_bulk_col_fp4 is not None
            and w13_bulk_col_sc is not None
            and w2_bulk_col_fp4 is not None
            and w2_bulk_col_sc is not None
            and h_bulk_quant is not None
        )
        if bulk_col_saved:
            flat_saved.extend([
                x_bulk_col_fp4,
                x_bulk_col_sc,
                w13_bulk_col_fp4,
                w13_bulk_col_sc,
                w2_bulk_col_fp4,
                w2_bulk_col_sc,
                h_bulk_quant.col_fp4,
                h_bulk_quant.col_sc,
            ])
        save_tensors = [counts]
        if h13_bulk is not None:
            save_tensors.append(h13_bulk)
        save_tensors.extend(flat_saved)
        ctx.save_for_backward(*save_tensors)
        ctx.has_h13_bulk = h13_bulk is not None
        ctx.saved_h13_splits = save_h13_splits
        ctx.bulk_col_saved = bulk_col_saved
        ctx.packed_w13_param = packed_w13_param
        ctx.active = active
        ctx.starts = starts
        ctx.m_padded = [item[1] for item in saved]
        ctx.dims = (E, H, D, Dk, Hk, Dn, Hn, H13, H13n)
        return y_cat

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        counts_and_saved = ctx.saved_tensors
        counts = counts_and_saved[0]
        saved_offset = 1
        h13_bulk = None
        if getattr(ctx, "has_h13_bulk", False):
            h13_bulk = counts_and_saved[1]
            saved_offset = 2
        tensors = counts_and_saved[saved_offset:]
        active = ctx.active
        if not active:
            return torch.zeros_like(grad_output), None, None, None, None

        E, H, D, Dk, Hk, Dn, Hn, H13, H13n = ctx.dims
        indexed_dy_base = getattr(ctx, "_mxfp4_indexed_scaled_dy_base", None)
        indexed_dy_tokens = getattr(ctx, "_mxfp4_indexed_scaled_dy_tokens", None)
        indexed_dy_scores = getattr(ctx, "_mxfp4_indexed_scaled_dy_scores", None)
        has_indexed_dy = (
            indexed_dy_base is not None
            and indexed_dy_tokens is not None
            and indexed_dy_scores is not None
        )
        if has_indexed_dy:
            dY = torch.empty(
                (int(indexed_dy_tokens.numel()), D),
                dtype=torch.bfloat16,
                device=indexed_dy_base.device,
            )
        else:
            dY = _as_contiguous_bf16(grad_output)
        active_rows = sum(M for _, M in active)
        all_experts_active = len(active) == E and all(expert_idx == i for i, (expert_idx, _) in enumerate(active))

        dY_row_q = []
        dY_row_sc = []
        dY_col_q = []
        dY_col_sc = []
        w2_col_q = []
        w2_col_sc_list = []
        h_col_q = []
        h_col_sc_list = []
        grad_h_outs = []
        grad_w2_outs = []
        per_expert = []
        saved_h13_splits = getattr(ctx, "saved_h13_splits", True)
        use_bulk_silu_deriv_quant = (
            h13_bulk is not None
            and (not saved_h13_splits or _mxfp4_deepseek_grouped_bulk_silu_deriv_quant())
            and Hn == H
            and H13n == 2 * H
            and not _mxfp4_needs_opt_quant("grad")
            and not _mxfp4_rht_for_role("grad")
        )
        grad_h_bulk = (
            torch.empty(sum(ctx.m_padded), Hn, dtype=torch.bfloat16, device=dY.device)
            if use_bulk_silu_deriv_quant
            else None
        )

        dY_bulk_src = None
        dY_bulk_row_fp4 = None
        dY_bulk_row_sc = None
        dY_bulk_col_fp4 = None
        dY_bulk_col_sc = None
        dY_bulk_col_fp4_list = None
        dY_bulk_col_sc_list = None
        prepacked_indexed_dy_bulk_src = getattr(ctx, "_mxfp4_prepacked_indexed_scaled_dy_bulk_src", None)
        if (
            _mxfp4_deepseek_grouped_bulk_activation_row_quant(len(active))
            and not _mxfp4_needs_opt_quant("grad")
            and not _mxfp4_rht_for_role("grad")
            and _mxfp4_deepseek_grouped_bulk_col_slice()
        ):
            dy_padded_starts = []
            dy_padded_offset = 0
            for rows_padded in ctx.m_padded:
                dy_padded_starts.append(dy_padded_offset)
                dy_padded_offset += rows_padded
            dy_live_rows = [M for _, M in active]
            dY_bulk_q = None
            if (
                has_indexed_dy
                and _mxfp4_deepseek_indexed_scaled_dy_quant()
            ):
                try:
                    indexed_scores = indexed_dy_scores
                    if indexed_scores.dtype != torch.float32:
                        indexed_scores = indexed_scores.float()
                    if (
                        all_experts_active
                        and len(set(dy_live_rows)) == 1
                        and len(set(ctx.m_padded)) == 1
                        and ctx.starts == [i * dy_live_rows[0] for i in range(len(active))]
                    ):
                        row_fp4, row_sc, col_fp4, col_sc = mxfp4_pack_indexed_scaled_rows_quantize_row_and_col(
                            _as_contiguous_bf16(indexed_dy_base),
                            indexed_dy_tokens.contiguous(),
                            indexed_scores.contiguous(),
                            len(active),
                            dy_live_rows[0],
                            ctx.m_padded[0],
                            Dn,
                            1,
                        )
                    elif _mxfp4_deepseek_variable_indexed_producer():
                        row_fp4, row_sc, col_fp4, col_sc = (
                            mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable(
                                _as_contiguous_bf16(indexed_dy_base),
                                indexed_dy_tokens.contiguous(),
                                indexed_scores.contiguous(),
                                ctx.starts,
                                dy_live_rows,
                                dy_padded_starts,
                                ctx.m_padded,
                                Dn,
                                1,
                            )
                        )
                    else:
                        raise AttributeError("variable indexed producer disabled")
                    dY_bulk_q = _MXFP4RowCol(
                        row_fp4=row_fp4,
                        row_sc=row_sc,
                        col_fp4=col_fp4,
                        col_sc=col_sc,
                    )
                except AttributeError:
                    dY_bulk_q = None
            if (
                dY_bulk_q is None
                and prepacked_indexed_dy_bulk_src is not None
                and has_indexed_dy
                and _mxfp4_deepseek_variable_indexed_pack()
            ):
                dY_bulk_src = prepacked_indexed_dy_bulk_src
                dY_bulk_q = _quantize_row_col_bf16(dY_bulk_src, role="grad")
            if (
                dY_bulk_q is None
                and has_indexed_dy
                and _mxfp4_deepseek_indexed_scaled_dy_quant()
                and _mxfp4_deepseek_variable_indexed_pack()
            ):
                try:
                    indexed_scores = indexed_dy_scores
                    if indexed_scores.dtype != torch.float32:
                        indexed_scores = indexed_scores.float()
                    dY_bulk_src = mxfp4_pack_indexed_scaled_rows_bf16_variable(
                        _as_contiguous_bf16(indexed_dy_base),
                        indexed_dy_tokens.contiguous(),
                        indexed_scores.contiguous(),
                        ctx.starts,
                        dy_live_rows,
                        dy_padded_starts,
                        ctx.m_padded,
                        Dn,
                    )
                    dY_bulk_q = _quantize_row_col_bf16(dY_bulk_src, role="grad")
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    dY_bulk_q = None
            if dY_bulk_q is None:
                if has_indexed_dy:
                    dY = _mxfp4_materialize_indexed_scaled_dy(
                        _as_contiguous_bf16(indexed_dy_base),
                        indexed_dy_tokens.contiguous(),
                        indexed_dy_scores.contiguous(),
                    )
                    has_indexed_dy = False
                dY_bulk_q = _mxfp4_pack_grouped_rows_quantize_uniform(
                    dY,
                    ctx.starts,
                    dy_live_rows,
                    ctx.m_padded,
                    Dn,
                )
            if dY_bulk_q is None:
                dY_bulk_src = _mxfp4_pack_grouped_rows(
                    dY,
                    ctx.starts,
                    dy_live_rows,
                    ctx.m_padded,
                    Dn,
                )
                dY_bulk_q = _quantize_row_col_bf16(dY_bulk_src, role="grad")
            dY_bulk_row_fp4, dY_bulk_row_sc = dY_bulk_q.row_fp4, dY_bulk_q.row_sc
            dY_bulk_col_fp4, dY_bulk_col_sc = dY_bulk_q.col_fp4, dY_bulk_q.col_sc
            dY_bulk_col_fp4_list, dY_bulk_col_sc_list = _mxfp4_slice_col_quant_lists_by_rows(
                dY_bulk_col_fp4,
                dY_bulk_col_sc,
                dy_padded_starts,
                ctx.m_padded,
            )
        if has_indexed_dy and dY_bulk_row_fp4 is None:
            dY = _mxfp4_materialize_indexed_scaled_dy(
                _as_contiguous_bf16(indexed_dy_base),
                indexed_dy_tokens.contiguous(),
                indexed_dy_scores.contiguous(),
            )
            has_indexed_dy = False

        grad_x = (
            torch.empty(dY.shape[0], D, dtype=torch.bfloat16, device=dY.device)
            if active_rows == dY.shape[0]
            else torch.zeros(dY.shape[0], D, dtype=torch.bfloat16, device=dY.device)
        )
        grad_w2_full = (
            torch.empty(E, Dn, Hn, dtype=torch.bfloat16, device=dY.device)
            if all_experts_active
            else torch.zeros(E, Dn, Hn, dtype=torch.bfloat16, device=dY.device)
        )
        use_split_w13_wgrad = (
            _mxfp4_deepseek_grouped_split_w13_wgrad()
            and not getattr(ctx, "packed_w13_param", False)
            and Dn == D
            and H13n == 2 * H
            and H % 256 == 0
        )
        if use_split_w13_wgrad:
            grad_w1_full = (
                torch.empty(E, H, D, dtype=torch.bfloat16, device=dY.device)
                if all_experts_active
                else torch.zeros(E, H, D, dtype=torch.bfloat16, device=dY.device)
            )
            grad_w3_full = (
                torch.empty(E, H, D, dtype=torch.bfloat16, device=dY.device)
                if all_experts_active
                else torch.zeros(E, H, D, dtype=torch.bfloat16, device=dY.device)
            )
            grad_w13_full = None
        else:
            grad_w1_full = None
            grad_w3_full = None
            grad_w13_full = (
                torch.empty(E, H13n, Dn, dtype=torch.bfloat16, device=dY.device)
                if all_experts_active
                else torch.zeros(E, H13n, Dn, dtype=torch.bfloat16, device=dY.device)
            )

        if getattr(ctx, "fast_strided_bulk", False):
            if dY_bulk_row_fp4 is None or dY_bulk_col_fp4 is None or h13_bulk is None or not all_experts_active:
                raise RuntimeError("fast MXFP4 grouped strided backward lost its required bulk tensors")
            x_bulk_col_fp4, x_bulk_col_sc, w13_bulk_col_fp4, w13_bulk_col_sc, w2_bulk_col_fp4, w2_bulk_col_sc, h_bulk_col_fp4, h_bulk_col_sc = tensors[:8]
            Mp0 = ctx.m_padded[0]
            grad_h_bulk = torch.empty(sum(ctx.m_padded), Hn, dtype=torch.bfloat16, device=dY.device)
            mxfp4_grouped_gemm_strided(
                dY_bulk_row_fp4,
                dY_bulk_row_sc,
                w2_bulk_col_fp4,
                w2_bulk_col_sc,
                grad_h_bulk,
                len(active),
                Mp0,
                Hn,
                Dn,
                Mp0,
                0,
                0,
                Dn,
                Mp0,
            )
            mxfp4_grouped_gemm_strided(
                dY_bulk_col_fp4,
                dY_bulk_col_sc,
                h_bulk_col_fp4,
                h_bulk_col_sc,
                grad_w2_full.reshape(E * Dn, Hn),
                len(active),
                Dn,
                Hn,
                Mp0,
                0,
                Mp0,
                0,
                Mp0,
                Dn,
            )
            row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined(
                grad_h_bulk,
                h13_bulk,
                H,
                H,
            )
            grad_x = (
                torch.empty(dY.shape[0], D, dtype=torch.bfloat16, device=dY.device)
                if active_rows == dY.shape[0]
                else torch.zeros(dY.shape[0], D, dtype=torch.bfloat16, device=dY.device)
            )
            grad_x_bulk = torch.empty(sum(ctx.m_padded), Dn, dtype=torch.bfloat16, device=dY.device)
            mxfp4_grouped_gemm_strided(
                row_fp4,
                row_sc,
                w13_bulk_col_fp4,
                w13_bulk_col_sc,
                grad_x_bulk,
                len(active),
                Mp0,
                Dn,
                H13n,
                Mp0,
                0,
                0,
                H13n,
                Mp0,
            )
            mxfp4_grouped_gemm_strided(
                col_fp4,
                col_sc,
                x_bulk_col_fp4,
                x_bulk_col_sc,
                grad_w13_full.reshape(E * H13n, Dn),
                len(active),
                H13n,
                Dn,
                Mp0,
                0,
                Mp0,
                0,
                Mp0,
                H13n,
            )
            skip_grad_x_scatter = (
                bool(getattr(ctx, "_mxfp4_moe_skip_grad_x_scatter", False))
                and len(active) > 0
                and len(set(ctx.m_padded)) == 1
            )
            if skip_grad_x_scatter:
                ctx._mxfp4_grouped_grad_x_bulk = grad_x_bulk
                ctx._mxfp4_grouped_grad_x_live_rows_per_batch = active[0][1]
                ctx._mxfp4_grouped_grad_x_padded_rows_per_batch = ctx.m_padded[0]
            else:
                mxfp4_scatter_grouped_rows_bf16(
                    grad_x_bulk,
                    grad_x,
                    ctx.starts,
                    [M for _, M in active],
                    ctx.m_padded,
                )
            if getattr(ctx, "packed_w13_param", False):
                grad_w13_ret = grad_w13_full[:, :H13, :D]
                return (
                    grad_x,
                    grad_w13_ret,
                    grad_w2_full[:, :D, :H],
                    None,
                    None,
                )
            if _mxfp4_deepseek_grouped_split_w13_grad_kernel() and Dn == D and H13n == 2 * H:
                try:
                    grad_w1_ret, grad_w3_ret = mxfp4_split_w13_bf16(grad_w13_full, H, D)
                except AttributeError:
                    grad_w1_ret = grad_w13_full[:, :H, :D]
                    grad_w3_ret = grad_w13_full[:, H:H13, :D]
            else:
                grad_w1_ret = grad_w13_full[:, :H, :D]
                grad_w3_ret = grad_w13_full[:, H:H13, :D]
            return (
                grad_x,
                grad_w1_ret,
                grad_w2_full[:, :D, :H],
                grad_w3_ret,
                None,
            )

        padded_offset = 0
        saved_stride = 10 if saved_h13_splits else 8
        bulk_col_tensors = None
        if getattr(ctx, "bulk_col_saved", False):
            bulk_base = saved_stride * len(active)
            bulk_col_tensors = tensors[bulk_base:bulk_base + 8]
        for i, (expert_idx, M) in enumerate(active):
            base = saved_stride * i
            if saved_h13_splits:
                x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc, h1, h3, h_col_fp4, h_col_sc_tensor, w2_col_fp4, w2_col_sc_tensor = tensors[base:base + 10]
            else:
                x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc, h_col_fp4, h_col_sc_tensor, w2_col_fp4, w2_col_sc_tensor = tensors[base:base + 8]
                h1 = None
                h3 = None
            Mp = ctx.m_padded[i]
            start = ctx.starts[i]
            if dY_bulk_row_fp4 is not None:
                row_fp4, row_sc = _mxfp4_slice_row_quant_by_rows(dY_bulk_row_fp4, dY_bulk_row_sc, padded_offset, Mp)
                col_fp4, col_sc = dY_bulk_col_fp4_list[i], dY_bulk_col_sc_list[i]
                dy_q = _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
            else:
                dy_pad = _pad_2d_bf16(dY[start:start + M, :D], Mp, Dn)
                dy_q = _quantize_row_col_bf16(dy_pad, role="grad")
            gh = (
                grad_h_bulk[padded_offset:padded_offset + Mp]
                if grad_h_bulk is not None
                else torch.empty(Mp, Hn, dtype=torch.bfloat16, device=dY.device)
            )
            gw2 = grad_w2_full[expert_idx]

            dY_row_q.append(dy_q.row_fp4)
            dY_row_sc.append(dy_q.row_sc)
            dY_col_q.append(dy_q.col_fp4)
            dY_col_sc.append(dy_q.col_sc)
            w2_col_q.append(w2_col_fp4)
            w2_col_sc_list.append(w2_col_sc_tensor)
            h_col_q.append(h_col_fp4)
            h_col_sc_list.append(h_col_sc_tensor)
            grad_h_outs.append(gh)
            grad_w2_outs.append(gw2)
            per_expert.append((expert_idx, M, Mp, x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc, h1, h3))
            padded_offset += Mp

        use_strided_bwd = (
            _mxfp4_deepseek_grouped_strided_gemm()
            and bulk_col_tensors is not None
            and dY_bulk_row_fp4 is not None
            and dY_bulk_col_fp4 is not None
            and grad_h_bulk is not None
            and all_experts_active
            and len(set(ctx.m_padded)) == 1
        )
        if use_strided_bwd:
            x_bulk_col_fp4, x_bulk_col_sc, w13_bulk_col_fp4, w13_bulk_col_sc, w2_bulk_col_fp4, w2_bulk_col_sc, h_bulk_col_fp4, h_bulk_col_sc = bulk_col_tensors
            Mp0 = ctx.m_padded[0]
            mxfp4_grouped_gemm_strided(
                dY_bulk_row_fp4,
                dY_bulk_row_sc,
                w2_bulk_col_fp4,
                w2_bulk_col_sc,
                grad_h_bulk,
                len(active),
                Mp0,
                Hn,
                Dn,
                Mp0,
                0,
                0,
                Dn,
                Mp0,
            )
            mxfp4_grouped_gemm_strided(
                dY_bulk_col_fp4,
                dY_bulk_col_sc,
                h_bulk_col_fp4,
                h_bulk_col_sc,
                grad_w2_full.reshape(E * Dn, Hn),
                len(active),
                Dn,
                Hn,
                Mp0,
                0,
                Mp0,
                0,
                Mp0,
                Dn,
            )
        else:
            _mxfp4_grouped_gemm_bucketed(dY_row_q, dY_row_sc, w2_col_q, w2_col_sc_list, grad_h_outs)
            _mxfp4_grouped_gemm_bucketed(dY_col_q, dY_col_sc, h_col_q, h_col_sc_list, grad_w2_outs)
        if _mxfp4_deepseek_grouped_debug_nan():
            for i, (grad_h_part, grad_w2_part) in enumerate(zip(grad_h_outs, grad_w2_outs)):
                M = active[i][1]
                _mxfp4_check_finite(f"grouped_bwd.grad_h_live[{i}]", grad_h_part[:M, :H])
                _mxfp4_check_finite(f"grouped_bwd.grad_w2_live[{i}]", grad_w2_part[:D, :H])

        dh13_row_q = []
        dh13_row_sc = []
        dh13_col_q = []
        dh13_col_sc = []
        w13_col_q = []
        w13_col_sc_list = []
        x_col_q = []
        x_col_sc_list = []
        grad_x_outs = []
        grad_w13_outs = []
        dh13_col_q_w1 = []
        dh13_col_sc_w1 = []
        dh13_col_q_w3 = []
        dh13_col_sc_w3 = []
        grad_w1_outs = []
        grad_w3_outs = []
        use_grad_x_bulk_scatter = _mxfp4_deepseek_grouped_pack_kernel()
        grad_x_bulk = (
            torch.empty(sum(ctx.m_padded), Dn, dtype=torch.bfloat16, device=dY.device)
            if use_grad_x_bulk_scatter
            else None
        )

        if use_bulk_silu_deriv_quant and grad_h_bulk is not None and h13_bulk is not None:
            row_fp4, row_sc, col_fp4, col_sc = (
                mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined(
                    grad_h_bulk,
                    h13_bulk,
                    H,
                    H,
                )
            )
            padded_offset = 0
            for item in per_expert:
                expert_idx, M, Mp, x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc, h1, h3 = item
                row_slice = _mxfp4_slice_row_quant_by_rows(row_fp4, row_sc, padded_offset, Mp)
                col_slice = _mxfp4_slice_col_quant_by_rows(col_fp4, col_sc, padded_offset, Mp)
                gx = (
                    grad_x_bulk[padded_offset:padded_offset + Mp]
                    if grad_x_bulk is not None
                    else torch.empty(Mp, Dn, dtype=torch.bfloat16, device=dY.device)
                )

                dh13_row_q.append(row_slice[0])
                dh13_row_sc.append(row_slice[1])
                w13_col_q.append(w13_col_fp4)
                w13_col_sc_list.append(w13_col_sc)
                x_col_q.append(x_col_fp4)
                x_col_sc_list.append(x_col_sc)
                grad_x_outs.append(gx)
                if use_split_w13_wgrad:
                    dh13_col_q_w1.append(col_slice[0][:H])
                    dh13_col_sc_w1.append(col_slice[1][:H // 128])
                    dh13_col_q_w3.append(col_slice[0][H:H13])
                    dh13_col_sc_w3.append(col_slice[1][H // 128:H13 // 128])
                    grad_w1_outs.append(grad_w1_full[expert_idx])
                    grad_w3_outs.append(grad_w3_full[expert_idx])
                else:
                    dh13_col_q.append(col_slice[0])
                    dh13_col_sc.append(col_slice[1])
                    grad_w13_outs.append(grad_w13_full[expert_idx])
                padded_offset += Mp
        else:
            padded_offset = 0
            for i, item in enumerate(per_expert):
                expert_idx, M, Mp, x_col_fp4, x_col_sc, w13_col_fp4, w13_col_sc, h1, h3 = item
                gh = grad_h_outs[i][:M, :H].float()
                h1_f = h1.float()
                _mxfp4_check_finite(f"grouped_bwd.gh[{i}]", gh)
                _mxfp4_check_finite(f"grouped_bwd.h1[{i}]", h1)
                _mxfp4_check_finite(f"grouped_bwd.h3[{i}]", h3)
                sig = torch.sigmoid(h1_f)
                silu = h1_f * sig
                silu_deriv = sig * (1.0 + h1_f * (1.0 - sig))
                _mxfp4_check_finite(f"grouped_bwd.silu_deriv[{i}]", silu_deriv)
                dh1 = (gh * h3.float() * silu_deriv).to(torch.bfloat16)
                dh3 = (gh * silu).to(torch.bfloat16)
                _mxfp4_check_finite(f"grouped_bwd.dh1[{i}]", dh1)
                _mxfp4_check_finite(f"grouped_bwd.dh3[{i}]", dh3)
                dh13 = torch.cat((dh1, dh3), dim=-1)
                _mxfp4_check_finite(f"grouped_bwd.dh13[{i}]", dh13)
                dh13_pad = _pad_2d_bf16(dh13, Mp, H13n)
                dh13_q = _quantize_row_col_bf16(dh13_pad, role="grad")
                gx = (
                    grad_x_bulk[padded_offset:padded_offset + Mp]
                    if grad_x_bulk is not None
                    else torch.empty(Mp, Dn, dtype=torch.bfloat16, device=dY.device)
                )

                dh13_row_q.append(dh13_q.row_fp4)
                dh13_row_sc.append(dh13_q.row_sc)
                w13_col_q.append(w13_col_fp4)
                w13_col_sc_list.append(w13_col_sc)
                x_col_q.append(x_col_fp4)
                x_col_sc_list.append(x_col_sc)
                grad_x_outs.append(gx)
                if use_split_w13_wgrad:
                    dh13_col_q_w1.append(dh13_q.col_fp4[:H])
                    dh13_col_sc_w1.append(dh13_q.col_sc[:H // 128])
                    dh13_col_q_w3.append(dh13_q.col_fp4[H:H13])
                    dh13_col_sc_w3.append(dh13_q.col_sc[H // 128:H13 // 128])
                    grad_w1_outs.append(grad_w1_full[expert_idx])
                    grad_w3_outs.append(grad_w3_full[expert_idx])
                else:
                    dh13_col_q.append(dh13_q.col_fp4)
                    dh13_col_sc.append(dh13_q.col_sc)
                    grad_w13_outs.append(grad_w13_full[expert_idx])
                padded_offset += Mp

        use_strided_dh13_bwd = (
            use_strided_bwd
            and use_bulk_silu_deriv_quant
            and grad_x_bulk is not None
            and grad_w13_full is not None
            and not use_split_w13_wgrad
        )
        if use_strided_dh13_bwd:
            Mp0 = ctx.m_padded[0]
            mxfp4_grouped_gemm_strided(
                row_fp4,
                row_sc,
                w13_bulk_col_fp4,
                w13_bulk_col_sc,
                grad_x_bulk,
                len(active),
                Mp0,
                Dn,
                H13n,
                Mp0,
                0,
                0,
                H13n,
                Mp0,
            )
            mxfp4_grouped_gemm_strided(
                col_fp4,
                col_sc,
                x_bulk_col_fp4,
                x_bulk_col_sc,
                grad_w13_full.reshape(E * H13n, Dn),
                len(active),
                H13n,
                Dn,
                Mp0,
                0,
                Mp0,
                0,
                Mp0,
                H13n,
            )
        else:
            _mxfp4_grouped_gemm_bucketed(dh13_row_q, dh13_row_sc, w13_col_q, w13_col_sc_list, grad_x_outs)
        if use_split_w13_wgrad:
            _mxfp4_grouped_gemm_bucketed(dh13_col_q_w1, dh13_col_sc_w1, x_col_q, x_col_sc_list, grad_w1_outs)
            _mxfp4_grouped_gemm_bucketed(dh13_col_q_w3, dh13_col_sc_w3, x_col_q, x_col_sc_list, grad_w3_outs)
        elif not use_strided_dh13_bwd:
            _mxfp4_grouped_gemm_bucketed(dh13_col_q, dh13_col_sc, x_col_q, x_col_sc_list, grad_w13_outs)
        if _mxfp4_deepseek_grouped_debug_nan():
            grad_w13_check = (
                [torch.cat((grad_w1_outs[i], grad_w3_outs[i]), dim=0) for i in range(len(grad_w1_outs))]
                if use_split_w13_wgrad
                else grad_w13_outs
            )
            for i, (grad_x_part, grad_w13) in enumerate(zip(grad_x_outs, grad_w13_check)):
                M = active[i][1]
                _mxfp4_check_finite(f"grouped_bwd.grad_x_live[{i}]", grad_x_part[:M, :D])
                _mxfp4_check_finite(f"grouped_bwd.grad_w13_live[{i}]", grad_w13[:H13, :D])

        skip_grad_x_scatter = (
            grad_x_bulk is not None
            and bool(getattr(ctx, "_mxfp4_moe_skip_grad_x_scatter", False))
            and len(active) > 0
            and len(set(ctx.m_padded)) == 1
        )
        if skip_grad_x_scatter:
            ctx._mxfp4_grouped_grad_x_bulk = grad_x_bulk
            ctx._mxfp4_grouped_grad_x_live_rows_per_batch = active[0][1]
            ctx._mxfp4_grouped_grad_x_padded_rows_per_batch = ctx.m_padded[0]
        elif grad_x_bulk is not None:
            try:
                mxfp4_scatter_grouped_rows_bf16(
                    grad_x_bulk,
                    grad_x,
                    ctx.starts,
                    [M for _, M in active],
                    ctx.m_padded,
                )
            except AttributeError:
                for i, (expert_idx, M) in enumerate(active):
                    start = ctx.starts[i]
                    grad_x[start:start + M] = grad_x_outs[i][:M, :D]
        else:
            for i, (expert_idx, M) in enumerate(active):
                start = ctx.starts[i]
                grad_x[start:start + M] = grad_x_outs[i][:M, :D]

        if getattr(ctx, "packed_w13_param", False):
            grad_w1_ret = grad_w13_full[:, :H13, :D]
            grad_w3_ret = None
        elif use_split_w13_wgrad:
            grad_w1_ret = grad_w1_full
            grad_w3_ret = grad_w3_full
        elif (
            _mxfp4_deepseek_grouped_split_w13_grad_kernel()
            and grad_w13_full is not None
            and Dn == D
            and H13n == 2 * H
        ):
            try:
                grad_w1_ret, grad_w3_ret = mxfp4_split_w13_bf16(grad_w13_full, H, D)
            except AttributeError:
                grad_w1_ret = grad_w13_full[:, :H, :D]
                grad_w3_ret = grad_w13_full[:, H:H13, :D]
        else:
            grad_w1_ret = grad_w13_full[:, :H, :D]
            grad_w3_ret = grad_w13_full[:, H:H13, :D]

        return (
            grad_x,
            grad_w1_ret,
            grad_w2_full[:, :D, :H],
            grad_w3_ret,
            None,
        )


class _MXFP4MoEScoreGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, route_positions: torch.Tensor):
        route_positions = route_positions.to(dtype=torch.int64).contiguous()
        flat_scores = scores.reshape(-1).to(dtype=torch.float32).contiguous()
        ctx.orig_shape = tuple(scores.shape)
        ctx.num_scores = int(flat_scores.numel())
        ctx.save_for_backward(route_positions)
        try:
            return mxfp4_moe_gather_scores(flat_scores, route_positions)
        except (AttributeError, FileNotFoundError, ImportError):
            return flat_scores[route_positions]

    @staticmethod
    def backward(ctx, grad_sorted: torch.Tensor):
        (route_positions,) = ctx.saved_tensors
        grad_sorted = grad_sorted.to(dtype=torch.float32).contiguous()
        try:
            grad_flat = mxfp4_moe_scatter_scores(grad_sorted, route_positions, ctx.num_scores)
        except (AttributeError, FileNotFoundError, ImportError):
            grad_flat = torch.empty(
                (ctx.num_scores,),
                device=grad_sorted.device,
                dtype=grad_sorted.dtype,
            )
            grad_flat[route_positions] = grad_sorted
        return grad_flat.reshape(ctx.orig_shape), None


class _MXFP4GroupedMoECombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        token_indices: torch.Tensor,
        counts: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        route_positions: torch.Tensor | None = None,
        route_inverse: torch.Tensor | None = None,
        route_inverse_padded: torch.Tensor | None = None,
        indexed_x_scores: torch.Tensor | None = None,
        indexed_x_rms_base: torch.Tensor | None = None,
        indexed_x_rms_weight: torch.Tensor | None = None,
        indexed_x_rms_inv: torch.Tensor | None = None,
        prequant_x_row_fp4: torch.Tensor | None = None,
        prequant_x_row_sc: torch.Tensor | None = None,
        prequant_x_col_fp4: torch.Tensor | None = None,
        prequant_x_col_sc: torch.Tensor | None = None,
    ):
        x_bf16 = _as_contiguous_bf16(x)
        token_indices = token_indices.to(dtype=torch.int64).contiguous()
        top_scores = top_scores.to(dtype=torch.float32).contiguous()
        indexed_x_scores = (
            indexed_x_scores.to(dtype=torch.float32).contiguous()
            if indexed_x_scores is not None
            else None
        )
        route_positions = (
            route_positions.to(dtype=torch.int64).contiguous()
            if route_positions is not None
            else None
        )
        route_inverse = (
            route_inverse.to(dtype=torch.int64).contiguous()
            if route_inverse is not None
            else None
        )
        route_inverse_padded = (
            route_inverse_padded.to(dtype=torch.int64).contiguous()
            if route_inverse_padded is not None
            else None
        )
        use_indexed_x = (
            indexed_x_scores is not None
            and (_mxfp4_deepseek_indexed_x_quant() or _mxfp4_deepseek_indexed_x_pack())
            and token_indices.numel() > 0
        )
        ctx._mxfp4_moe_skip_y_cat_scatter = (
            _mxfp4_deepseek_padded_moe_combine()
            and _mxfp4_deepseek_fused_combine_kernels()
            and _mxfp4_deepseek_route_inverse_combine()
        )
        if use_indexed_x:
            routed_input = torch.empty(
                (token_indices.numel(), x_bf16.shape[1]),
                dtype=torch.bfloat16,
                device=x_bf16.device,
            )
            ctx._mxfp4_indexed_x_base = x_bf16
            ctx._mxfp4_indexed_x_tokens = token_indices
            ctx._mxfp4_indexed_x_scores = indexed_x_scores
            ctx._mxfp4_indexed_x_rms_base = (
                _as_contiguous_bf16(indexed_x_rms_base)
                if indexed_x_rms_base is not None
                else None
            )
            ctx._mxfp4_indexed_x_rms_weight = (
                _as_contiguous_bf16(indexed_x_rms_weight)
                if indexed_x_rms_weight is not None
                else None
            )
            ctx._mxfp4_indexed_x_rms_inv = (
                indexed_x_rms_inv.contiguous()
                if indexed_x_rms_inv is not None
                else None
            )
            ctx._mxfp4_indexed_x_dummy = True
        else:
            token_indices_expanded = token_indices.reshape(-1, 1).expand(-1, x_bf16.shape[1])
            routed_input = torch.gather(x_bf16, dim=0, index=token_indices_expanded)
            ctx._mxfp4_indexed_x_dummy = False
        ctx._mxfp4_prequant_x_bulk_q = (
            _MXFP4RowCol(
                row_fp4=prequant_x_row_fp4,
                row_sc=prequant_x_row_sc,
                col_fp4=prequant_x_col_fp4,
                col_sc=prequant_x_col_sc,
            )
            if prequant_x_row_fp4 is not None
            and prequant_x_row_sc is not None
            and prequant_x_col_fp4 is not None
            and prequant_x_col_sc is not None
            else None
        )

        expert_out = _MXFP4GroupedExpertsBatchedFunction.forward(
            ctx,
            routed_input,
            w1,
            w2,
            w3,
            counts,
        )
        ctx._mxfp4_indexed_x_base = None
        ctx._mxfp4_indexed_x_tokens = None
        ctx._mxfp4_indexed_x_scores = None
        ctx._mxfp4_indexed_x_rms_base = None
        ctx._mxfp4_indexed_x_rms_weight = None
        ctx._mxfp4_indexed_x_rms_inv = None
        ctx._mxfp4_indexed_x_dummy = False
        ctx._mxfp4_prequant_x_bulk_q = None
        y_bulk = getattr(ctx, "_mxfp4_grouped_y_bulk", None)
        y_bulk_live_rows = int(getattr(ctx, "_mxfp4_grouped_live_rows_per_batch", 0))
        y_bulk_padded_rows = int(getattr(ctx, "_mxfp4_grouped_padded_rows_per_batch", 0))
        top_k = int(token_indices.numel() // x_bf16.shape[0]) if x_bf16.shape[0] > 0 else 0
        can_route_inverse_combine = (
            _mxfp4_deepseek_fused_combine_kernels()
            and _mxfp4_deepseek_route_inverse_combine()
            and (route_positions is not None or route_inverse is not None)
            and top_k > 0
            and token_indices.numel() == x_bf16.shape[0] * top_k
        )
        out = torch.empty_like(x_bf16) if can_route_inverse_combine else torch.zeros_like(x_bf16)
        used_fused_scatter = False
        if can_route_inverse_combine:
            try:
                if route_inverse is None:
                    route_inverse = mxfp4_moe_build_route_inverse(route_positions)
                if y_bulk is not None and y_bulk_live_rows > 0 and y_bulk_padded_rows >= y_bulk_live_rows:
                    if route_inverse_padded is not None:
                        mxfp4_moe_route_combine_padded_index_bf16(
                            _as_contiguous_bf16(y_bulk),
                            top_scores,
                            route_inverse,
                            route_inverse_padded,
                            out,
                            top_k,
                        )
                    else:
                        mxfp4_moe_route_combine_padded_bf16(
                            _as_contiguous_bf16(y_bulk),
                            top_scores,
                            route_inverse,
                            out,
                            top_k,
                            y_bulk_live_rows,
                            y_bulk_padded_rows,
                        )
                else:
                    mxfp4_moe_route_combine_bf16(
                        _as_contiguous_bf16(expert_out),
                        top_scores,
                        route_inverse,
                        out,
                        top_k,
                    )
                used_fused_scatter = True
            except (AttributeError, FileNotFoundError, ImportError):
                if y_bulk is not None:
                    mxfp4_scatter_grouped_rows_bf16(
                        y_bulk,
                        expert_out,
                        ctx.starts,
                        [M for _, M in ctx.active],
                        ctx.m_padded,
                    )
                    y_bulk = None
                route_inverse = None
                used_fused_scatter = False
                out.zero_()
        if y_bulk is not None and not used_fused_scatter:
            mxfp4_scatter_grouped_rows_bf16(
                y_bulk,
                expert_out,
                ctx.starts,
                [M for _, M in ctx.active],
                ctx.m_padded,
            )
            y_bulk = None
        if _mxfp4_deepseek_fused_combine_kernels() and not used_fused_scatter:
            try:
                mxfp4_moe_scale_scatter_add_bf16(
                    _as_contiguous_bf16(expert_out),
                    top_scores,
                    token_indices,
                    out,
                )
                used_fused_scatter = True
            except (AttributeError, FileNotFoundError, ImportError):
                used_fused_scatter = False
                if can_route_inverse_combine:
                    out.zero_()
        if not used_fused_scatter:
            token_indices_expanded = token_indices.reshape(-1, 1).expand(-1, x_bf16.shape[1])
            scaled = (expert_out.float() * top_scores.reshape(-1, 1)).to(torch.bfloat16)
            out.scatter_add_(dim=0, index=token_indices_expanded, src=scaled)

        ctx._mxfp4_moe_token_indices = token_indices
        ctx._mxfp4_moe_top_scores = top_scores
        ctx._mxfp4_moe_expert_out = expert_out
        ctx._mxfp4_moe_expert_out_padded = y_bulk if used_fused_scatter else None
        ctx._mxfp4_moe_y_bulk_live_rows = y_bulk_live_rows if y_bulk is not None else 0
        ctx._mxfp4_moe_y_bulk_padded_rows = y_bulk_padded_rows if y_bulk is not None else 0
        ctx._mxfp4_moe_route_inverse = route_inverse
        ctx._mxfp4_moe_route_inverse_padded = route_inverse_padded
        ctx._mxfp4_moe_top_k = top_k
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_out = _as_contiguous_bf16(grad_output)
        token_indices = ctx._mxfp4_moe_token_indices
        top_scores = ctx._mxfp4_moe_top_scores
        expert_out = ctx._mxfp4_moe_expert_out
        expert_out_padded = ctx._mxfp4_moe_expert_out_padded
        y_bulk_live_rows = ctx._mxfp4_moe_y_bulk_live_rows
        y_bulk_padded_rows = ctx._mxfp4_moe_y_bulk_padded_rows
        route_inverse = ctx._mxfp4_moe_route_inverse
        route_inverse_padded = ctx._mxfp4_moe_route_inverse_padded
        top_k = ctx._mxfp4_moe_top_k

        grad_scores = None
        fused_dy_bulk_src = None
        if (
            _mxfp4_deepseek_fused_indexed_dy_dot_pack()
            and expert_out_padded is None
            and _mxfp4_deepseek_fused_combine_kernels()
            and _mxfp4_deepseek_variable_indexed_pack()
            and token_indices.numel() > 0
            and grad_out.shape[1] % 128 == 0
        ):
            try:
                dy_padded_starts = []
                dy_padded_offset = 0
                for rows_padded in ctx.m_padded:
                    dy_padded_starts.append(dy_padded_offset)
                    dy_padded_offset += rows_padded
                fused_dy_bulk_src, grad_scores = mxfp4_dot_and_pack_indexed_scaled_rows_bf16_variable(
                    grad_out,
                    token_indices,
                    top_scores,
                    _as_contiguous_bf16(expert_out),
                    ctx.starts,
                    [M for _, M in ctx.active],
                    dy_padded_starts,
                    ctx.m_padded,
                    _mxfp4_deepseek_grouped_output_dim(grad_out.shape[1]),
                )
            except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                fused_dy_bulk_src = None
                grad_scores = None
        if _mxfp4_deepseek_fused_combine_kernels():
            try:
                if grad_scores is not None:
                    pass
                elif expert_out_padded is not None and y_bulk_live_rows > 0:
                    grad_scores = mxfp4_moe_indexed_dot_rows_padded_bf16(
                        grad_out,
                        token_indices,
                        _as_contiguous_bf16(expert_out_padded),
                        y_bulk_live_rows,
                        y_bulk_padded_rows,
                    )
                else:
                    grad_scores = mxfp4_moe_indexed_dot_rows_bf16(
                        grad_out,
                        token_indices,
                        _as_contiguous_bf16(expert_out),
                    )
            except (AttributeError, FileNotFoundError, ImportError):
                if expert_out_padded is not None:
                    mxfp4_scatter_grouped_rows_bf16(
                        expert_out_padded,
                        expert_out,
                        ctx.starts,
                        [M for _, M in ctx.active],
                        ctx.m_padded,
                    )
                    expert_out_padded = None
                grad_scores = None
        if grad_scores is None:
            token_indices_expanded = token_indices.reshape(-1, 1).expand(-1, grad_out.shape[1])
            gathered_grad = torch.gather(grad_out, dim=0, index=token_indices_expanded)
            grad_scores = (gathered_grad.float() * expert_out.float()).sum(dim=1)

        ctx._mxfp4_indexed_scaled_dy_base = grad_out
        ctx._mxfp4_indexed_scaled_dy_tokens = token_indices
        ctx._mxfp4_indexed_scaled_dy_scores = top_scores
        ctx._mxfp4_prepacked_indexed_scaled_dy_bulk_src = fused_dy_bulk_src
        ctx._mxfp4_moe_skip_grad_x_scatter = (
            _mxfp4_deepseek_padded_moe_combine()
            and _mxfp4_deepseek_fused_combine_kernels()
            and _mxfp4_deepseek_route_inverse_combine()
            and _mxfp4_deepseek_fused_gradx_scatter()
            and route_inverse_padded is not None
        )
        dummy_expert_grad = torch.empty_like(expert_out)
        grad_routed_x, grad_w1, grad_w2, grad_w3, _ = _MXFP4GroupedExpertsBatchedFunction.backward(
            ctx,
            dummy_expert_grad,
        )
        ctx._mxfp4_prepacked_indexed_scaled_dy_bulk_src = None
        grad_routed_x_padded = getattr(ctx, "_mxfp4_grouped_grad_x_bulk", None)

        can_route_inverse_gradx = (
            _mxfp4_deepseek_fused_combine_kernels()
            and _mxfp4_deepseek_route_inverse_combine()
            and _mxfp4_deepseek_fused_gradx_scatter()
            and route_inverse is not None
            and top_k > 0
        )
        grad_x = torch.empty_like(grad_out) if can_route_inverse_gradx else torch.zeros_like(grad_out)
        used_fused_scatter = False
        if can_route_inverse_gradx:
            try:
                if grad_routed_x_padded is not None and route_inverse_padded is not None:
                    mxfp4_moe_route_scatter_gradx_padded_index_bf16(
                        _mxfp4_narrow_features_contiguous_bf16(grad_routed_x_padded, grad_out.shape[1]),
                        route_inverse_padded,
                        grad_x,
                        top_k,
                    )
                else:
                    mxfp4_moe_route_scatter_gradx_bf16(
                        _mxfp4_narrow_features_contiguous_bf16(grad_routed_x, grad_out.shape[1]),
                        route_inverse,
                        grad_x,
                        top_k,
                    )
                used_fused_scatter = True
            except (AttributeError, FileNotFoundError, ImportError):
                if grad_routed_x_padded is not None:
                    mxfp4_scatter_grouped_rows_bf16(
                        grad_routed_x_padded,
                        grad_routed_x,
                        ctx.starts,
                        [M for _, M in ctx.active],
                        ctx.m_padded,
                    )
                    grad_routed_x_padded = None
                used_fused_scatter = False
                grad_x.zero_()
        if grad_routed_x_padded is not None and not used_fused_scatter:
            mxfp4_scatter_grouped_rows_bf16(
                grad_routed_x_padded,
                grad_routed_x,
                ctx.starts,
                [M for _, M in ctx.active],
                ctx.m_padded,
            )
            grad_routed_x_padded = None
        if (
            _mxfp4_deepseek_fused_combine_kernels()
            and _mxfp4_deepseek_fused_gradx_scatter()
            and not used_fused_scatter
        ):
            try:
                mxfp4_moe_scatter_add_bf16(
                    _mxfp4_narrow_features_contiguous_bf16(grad_routed_x, grad_out.shape[1]),
                    token_indices,
                    grad_x,
                )
                used_fused_scatter = True
            except (AttributeError, FileNotFoundError, ImportError):
                used_fused_scatter = False
                if can_route_inverse_gradx:
                    grad_x.zero_()
        if not used_fused_scatter:
            token_indices_expanded = token_indices.reshape(-1, 1).expand(-1, grad_out.shape[1])
            grad_x.scatter_add_(dim=0, index=token_indices_expanded, src=grad_routed_x[:, :grad_out.shape[1]])
        return (
            grad_x,
            grad_scores,
            None,
            None,
            grad_w1,
            grad_w2,
            grad_w3,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MXFP4GroupedExpertsTK(nn.Module):
    """DeepSeek grouped experts using MXFP4 routed expert GEMMs.

    This preserves the TorchTitan GroupedExperts parameter layout while making
    the routed expert matmuls use the same MXFP4 quantization contract as MLA
    and shared experts. The batched path uses TK's variable-M batched launcher
    for uneven MoE token counts.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        device=None,
        dtype=torch.bfloat16,
        packed_w13_param: bool | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.packed_w13_param = (
            _mxfp4_deepseek_grouped_packed_w13_param()
            if packed_w13_param is None
            else bool(packed_w13_param)
        )
        if self.packed_w13_param:
            self.w13 = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, dim, device=device, dtype=dtype))
        else:
            self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim, device=device, dtype=dtype))
            self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim, device=device, dtype=dtype))
        self._moe_route_inverse_cache = {}
        self._moe_route_inverse_padded_cache = {}
        self._moe_indexed_x_ones_cache = {}

    @staticmethod
    def _local_param(param: torch.Tensor) -> torch.Tensor:
        local = param
        to_local = getattr(local, "to_local", None)
        if to_local is not None:
            try:
                local = to_local()
            except Exception:
                local = param
        return local

    @staticmethod
    def _local_expert_dim(param: torch.Tensor) -> int:
        local = MXFP4GroupedExpertsTK._local_param(param)
        return int(local.shape[0]) if local.ndim > 0 else 0

    def _num_local_experts(self) -> int:
        return self._local_expert_dim(self.w13 if self.packed_w13_param else self.w1)

    def _counts_are_local(self, num_tokens_per_expert: torch.Tensor) -> bool:
        # ExpertParallel sharding passes global expert counts before its forward
        # hooks run. The fused combine entrypoints are direct method calls, so
        # they must only run once counts already match this rank's local shard.
        return (
            num_tokens_per_expert.dim() == 1
            and int(num_tokens_per_expert.numel()) == self._num_local_experts()
        )

    def can_fuse_moe_combine(self, num_experts: int) -> bool:
        return int(num_experts) == self._num_local_experts()

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if _mxfp4_deepseek_grouped_batched_enabled():
            if self.packed_w13_param:
                w13 = self._local_param(self.w13)
                w2 = self._local_param(self.w2)
                empty_w3 = w13.new_empty(0)
                return _MXFP4GroupedExpertsBatchedFunction.apply(
                    x, w13, w2, empty_w3, num_tokens_per_expert
                )
            w1 = self._local_param(self.w1)
            w2 = self._local_param(self.w2)
            w3 = self._local_param(self.w3)
            return _MXFP4GroupedExpertsBatchedFunction.apply(
                x, w1, w2, w3, num_tokens_per_expert
            )
        if x.numel() == 0:
            return x

        counts = num_tokens_per_expert.to(dtype=torch.int64).tolist()
        active_tokens = int(sum(counts))
        padded_tokens = x.shape[0] - active_tokens
        if active_tokens <= 0:
            return torch.zeros_like(x)

        expert_inputs = torch.split(x[:active_tokens], counts, dim=0)
        outputs = []
        for expert_idx, expert_x in enumerate(expert_inputs):
            if expert_x.shape[0] == 0:
                continue
            if self.packed_w13_param:
                w13 = self._local_param(self.w13)[expert_idx]
            else:
                w13 = torch.cat(
                    (
                        self._local_param(self.w1)[expert_idx],
                        self._local_param(self.w3)[expert_idx],
                    ),
                    dim=0,
                )
            h13 = _MXFP4LinearFunction.apply(expert_x, w13, None, None)
            h1, h3 = h13.chunk(2, dim=-1)
            h = F.silu(h1) * h3
            outputs.append(_MXFP4LinearFunction.apply(h, self._local_param(self.w2)[expert_idx], None, None))

        if outputs:
            out = torch.cat(outputs, dim=0)
        else:
            out = x.new_zeros((0, x.shape[-1]))
        if padded_tokens > 0:
            out = torch.vstack((out, out.new_zeros((padded_tokens, out.shape[-1]))))
        return out

    def forward_moe_combine_unsorted_scores(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        token_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        route_positions: torch.Tensor | None = None,
        route_inverse: torch.Tensor | None = None,
        route_inverse_padded: torch.Tensor | None = None,
        indexed_x_rms_base: torch.Tensor | None = None,
        indexed_x_rms_weight: torch.Tensor | None = None,
        indexed_x_rms_inv: torch.Tensor | None = None,
        prequant_x_bulk_q: _MXFP4RowCol | None = None,
    ) -> torch.Tensor | None:
        if not _mxfp4_deepseek_unsorted_score_gather() or route_positions is None:
            return None
        if not self._counts_are_local(num_tokens_per_expert):
            return None
        top_scores_sorted = _MXFP4MoEScoreGatherFunction.apply(top_scores, route_positions)
        return self.forward_moe_combine(
            x,
            top_scores_sorted,
            token_indices,
            num_tokens_per_expert,
            route_positions,
            route_inverse,
            route_inverse_padded,
            indexed_x_rms_base=indexed_x_rms_base,
            indexed_x_rms_weight=indexed_x_rms_weight,
            indexed_x_rms_inv=indexed_x_rms_inv,
            prequant_x_bulk_q=prequant_x_bulk_q,
        )

    def forward_moe_combine(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        token_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        route_positions: torch.Tensor | None = None,
        route_inverse: torch.Tensor | None = None,
        route_inverse_padded: torch.Tensor | None = None,
        indexed_x_rms_base: torch.Tensor | None = None,
        indexed_x_rms_weight: torch.Tensor | None = None,
        indexed_x_rms_inv: torch.Tensor | None = None,
        prequant_x_bulk_q: _MXFP4RowCol | None = None,
    ) -> torch.Tensor | None:
        if not _mxfp4_deepseek_fused_moe_combine():
            return None
        if not _mxfp4_deepseek_grouped_batched_enabled():
            return None
        if token_indices.dim() != 1 or top_scores.dim() != 1:
            return None
        local_num_experts = self._num_local_experts()
        if num_tokens_per_expert.dim() != 1 or int(num_tokens_per_expert.numel()) != local_num_experts:
            return None
        indexed_x_scores = None
        if _mxfp4_deepseek_indexed_x_quant() or _mxfp4_deepseek_indexed_x_pack():
            cache_key = (
                int(token_indices.numel()),
                token_indices.device.type,
                token_indices.device.index,
            )
            indexed_x_scores = self._moe_indexed_x_ones_cache.get(cache_key)
            if indexed_x_scores is None:
                indexed_x_scores = torch.ones(
                    (token_indices.numel(),),
                    dtype=torch.float32,
                    device=token_indices.device,
                )
                self._moe_indexed_x_ones_cache[cache_key] = indexed_x_scores
        if (
            _mxfp4_deepseek_fused_combine_kernels()
            and _mxfp4_deepseek_route_inverse_combine()
            and route_positions is not None
        ):
            route_positions = route_positions.to(dtype=torch.int64).contiguous()
            cache_key = (
                int(route_positions.numel()),
                route_positions.device.type,
                route_positions.device.index,
                int(token_indices.numel() // max(1, x.shape[0])),
            )
            use_inverse_cache = _mxfp4_deepseek_route_inverse_cache()
            if route_inverse is None:
                route_inverse = self._moe_route_inverse_cache.get(cache_key) if use_inverse_cache else None
            if route_inverse is None:
                route_inverse = mxfp4_moe_build_route_inverse(route_positions)
                if use_inverse_cache:
                    self._moe_route_inverse_cache[cache_key] = route_inverse
            routes = int(token_indices.numel())
            if local_num_experts > 0 and routes % local_num_experts == 0:
                live_rows = routes // local_num_experts
                padded_rows = _mxfp4_deepseek_grouped_m_pad(live_rows)
                padded_cache_key = cache_key + (local_num_experts, live_rows, padded_rows)
                if route_inverse_padded is None:
                    route_inverse_padded = (
                        self._moe_route_inverse_padded_cache.get(padded_cache_key)
                        if use_inverse_cache
                        else None
                    )
                if route_inverse_padded is None:
                    route_inverse_padded = mxfp4_moe_build_route_inverse_padded(
                        route_positions,
                        live_rows,
                        padded_rows,
                    )
                    if use_inverse_cache:
                        self._moe_route_inverse_padded_cache[padded_cache_key] = route_inverse_padded
        if self.packed_w13_param:
            w13 = self._local_param(self.w13)
            w2 = self._local_param(self.w2)
            empty_w3 = w13.new_empty(0)
            return _MXFP4GroupedMoECombineFunction.apply(
                x,
                top_scores,
                token_indices,
                num_tokens_per_expert,
                w13,
                w2,
                empty_w3,
                route_positions,
                route_inverse,
                route_inverse_padded,
                indexed_x_scores,
                indexed_x_rms_base,
                indexed_x_rms_weight,
                indexed_x_rms_inv,
                prequant_x_bulk_q.row_fp4 if prequant_x_bulk_q is not None else None,
                prequant_x_bulk_q.row_sc if prequant_x_bulk_q is not None else None,
                prequant_x_bulk_q.col_fp4 if prequant_x_bulk_q is not None else None,
                prequant_x_bulk_q.col_sc if prequant_x_bulk_q is not None else None,
            )
        w1 = self._local_param(self.w1)
        w2 = self._local_param(self.w2)
        w3 = self._local_param(self.w3)
        return _MXFP4GroupedMoECombineFunction.apply(
            x,
            top_scores,
            token_indices,
            num_tokens_per_expert,
            w1,
            w2,
            w3,
            route_positions,
            route_inverse,
            route_inverse_padded,
            indexed_x_scores,
            indexed_x_rms_base,
            indexed_x_rms_weight,
            indexed_x_rms_inv,
            prequant_x_bulk_q.row_fp4 if prequant_x_bulk_q is not None else None,
            prequant_x_bulk_q.row_sc if prequant_x_bulk_q is not None else None,
            prequant_x_bulk_q.col_fp4 if prequant_x_bulk_q is not None else None,
            prequant_x_bulk_q.col_sc if prequant_x_bulk_q is not None else None,
        )

    def forward_moe_combine_with_shared(
        self,
        x: torch.Tensor,
        shared_experts: nn.Module,
        top_scores: torch.Tensor,
        token_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        route_positions: torch.Tensor | None = None,
        route_inverse: torch.Tensor | None = None,
        route_inverse_padded: torch.Tensor | None = None,
        indexed_x_rms_base: torch.Tensor | None = None,
        indexed_x_rms_weight: torch.Tensor | None = None,
        indexed_x_rms_inv: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._counts_are_local(num_tokens_per_expert):
            return None
        if (
            not _mxfp4_deepseek_shared_routed_combined_x_quant()
            or not hasattr(shared_experts, "forward_prequant_w13")
        ):
            return None
        combined_q = _mxfp4_build_shared_routed_x_quant(
            x,
            token_indices,
            num_tokens_per_expert,
        )
        if combined_q is None:
            return None
        routed = self.forward_moe_combine(
            x,
            top_scores,
            token_indices,
            num_tokens_per_expert,
            route_positions,
            route_inverse,
            route_inverse_padded,
            indexed_x_rms_base=indexed_x_rms_base,
            indexed_x_rms_weight=indexed_x_rms_weight,
            indexed_x_rms_inv=indexed_x_rms_inv,
            prequant_x_bulk_q=combined_q.routed,
        )
        if routed is None:
            return None
        shared = shared_experts.forward_prequant_w13(
            x,
            combined_q.shared,
            combined_q.shared_rows_padded,
            combined_q.cols_padded,
        )
        return routed + shared

    def init_weights(self, init_std: float):
        if self.packed_w13_param:
            nn.init.trunc_normal_(self.w13[:, :self.hidden_dim], mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.w13[:, self.hidden_dim:], mean=0.0, std=init_std)
        else:
            nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)

    @classmethod
    def from_grouped_experts(
        cls,
        experts: nn.Module,
        packed_w13_param: bool | None = None,
    ) -> "MXFP4GroupedExpertsTK":
        out = cls(
            dim=experts.w1.shape[-1],
            hidden_dim=experts.w1.shape[-2],
            num_experts=experts.w1.shape[0],
            device=experts.w1.device,
            dtype=experts.w1.dtype,
            packed_w13_param=packed_w13_param,
        )
        if experts.w1.device.type != "meta":
            with torch.no_grad():
                if out.packed_w13_param:
                    out.w13[:, :out.hidden_dim].copy_(experts.w1)
                    out.w13[:, out.hidden_dim:].copy_(experts.w3)
                else:
                    out.w1.copy_(experts.w1)
                    out.w3.copy_(experts.w3)
                out.w2.copy_(experts.w2)
        return out


class _FusedQKVFunction_MXFP4_TK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        w_qkv: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        rope_freqs_cis: torch.Tensor | None,
        rope_batch_size: int,
        rope_seq_len: int,
        rope_head_dim: int,
        debug_name: str | None = None,
        h_row_fp4: torch.Tensor | None = None,
        h_row_sc: torch.Tensor | None = None,
        h_col_fp4: torch.Tensor | None = None,
        h_col_sc: torch.Tensor | None = None,
        h_r_tile: torch.Tensor | None = None,
        cde_row_rms_partial: torch.Tensor | None = None,
    ):
        stage_start = _mxfp4_stage_begin("qkv_fwd", debug_name)
        inp = _as_contiguous_bf16(input)
        nw = _as_contiguous_bf16(norm_weight.detach())
        M, K = inp.shape
        total_out = q_dim + k_dim + v_dim
        rope_applied = False
        ctx.h_tile = h_row_fp4 is not None and h_row_fp4.numel() != 0
        ctx.cde_exact = (
            cde_row_rms_partial is not None
            and cde_row_rms_partial.numel() != 0
        )
        if ctx.h_tile and ctx.cde_exact:
            raise RuntimeError("exact C/D/E and MX H QKV carriers are mutually exclusive")

        te_fused = _get_te_fused()
        supported = _mxfp4_supported(M, K) and _mxfp4_supported(total_out, K)
        _require_mixed_localcta_supported_path("QKV", supported)
        if not supported:
            if ctx.h_tile or ctx.cde_exact:
                raise RuntimeError("MX QKV carrier requires the native supported shape")
            ctx.fast_path = False
            normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
            ctx.save_for_backward(inp, nw, inv_rms, w_qkv)
            ctx.q_dim = q_dim
            ctx.k_dim = k_dim
            ctx.v_dim = v_dim
            ctx.epsilon = epsilon
            ctx.rope_applied = False
            ctx._mxfp4_debug_name = debug_name
            y = normed.matmul(w_qkv.t())
            _mxfp4_stage_end("qkv_fwd", debug_name, stage_start)
            return (
                y[:, :q_dim].contiguous(),
                y[:, q_dim:q_dim + k_dim].contiguous(),
                y[:, q_dim + k_dim:].contiguous(),
            )

        ctx.fast_path = True
        ctx.mixed_localcta_dgrad = use_mxfp4_localcta_dgrad()
        w_qkv_bf16 = _as_contiguous_bf16(w_qkv.detach())
        w_qkv_q = None
        weight_quant_stream = None
        if use_mxfp4_qkv_fwd_weight_quant_overlap():
            weight_quant_stream = _get_mxfp4_fwd_side_stream()
            weight_quant_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(w_qkv_bf16, weight_quant_stream)
            with torch.cuda.stream(weight_quant_stream):
                w_qkv_q = (
                    _quantize_mixed_weight_bf16(w_qkv_bf16)
                    if ctx.mixed_localcta_dgrad
                    else _quantize_weight_row_col_bf16(w_qkv_bf16)
                )

        if ctx.h_tile:
            x_q = _MXFP4RowCol(h_row_fp4, h_row_sc, h_col_fp4, h_col_sc)
            inv_rms = h_r_tile
        elif ctx.cde_exact:
            if (
                M % 256
                or K != 4096
                or cde_row_rms_partial.dtype != torch.float32
                or not cde_row_rms_partial.is_cuda
                or not cde_row_rms_partial.is_contiguous()
                or tuple(cde_row_rms_partial.shape) != (M, K // 256)
            ):
                partial_stride = (
                    tuple(cde_row_rms_partial.stride())
                    if hasattr(cde_row_rms_partial, "stride")
                    else None
                )
                raise RuntimeError(
                    "MX exact C/D/E QKV requires contiguous CUDA float32 "
                    f"row partial {(M, K // 256)} for [M,4096]; "
                    f"got type={type(cde_row_rms_partial).__name__} "
                    f"shape={tuple(cde_row_rms_partial.shape)} "
                    f"dtype={cde_row_rms_partial.dtype} "
                    f"device={cde_row_rms_partial.device} "
                    f"is_cuda={cde_row_rms_partial.is_cuda} "
                    f"contiguous={cde_row_rms_partial.is_contiguous()} "
                    f"stride={partial_stride} layer={debug_name}"
                )
            if _mxfp4_needs_opt_quant("activation"):
                raise RuntimeError("MX exact C/D/E does not support activation RHT/SR")
            row_fp4, row_sc, col_fp4, col_sc, inv_rms = (
                mxfp4_fused_rmsnorm_quantize_row_and_col_from_row_rms_partial(
                    inp, nw, cde_row_rms_partial, float(epsilon), 1
                )
            )
            x_q = _MXFP4RowCol(row_fp4, row_sc, col_fp4, col_sc)
        elif _mxfp4_bool_env("MXFP4_USE_QKV_RMSNORM_QUANT_FUSION", True):
            qkv_substage = _mxfp4_stage_begin("qkv_fwd_rms_quant_x", debug_name)
            x_q, inv_rms = _rmsnorm_quantize_row_col_bf16(te_fused, inp, nw, float(epsilon), kind="qkv")
            _mxfp4_stage_end("qkv_fwd_rms_quant_x", debug_name, qkv_substage)
        else:
            qkv_substage = _mxfp4_stage_begin("qkv_fwd_rms", debug_name)
            normed, inv_rms = _rmsnorm_to_bf16(te_fused, inp, nw, float(epsilon), kind="qkv")
            _mxfp4_stage_end("qkv_fwd_rms", debug_name, qkv_substage)

            qkv_substage = _mxfp4_stage_begin("qkv_fwd_quant_x", debug_name)
            x_q = _quantize_row_col_bf16(normed, role="activation")
            _mxfp4_stage_end("qkv_fwd_quant_x", debug_name, qkv_substage)

        qkv_substage = _mxfp4_stage_begin("qkv_fwd_quant_w", debug_name)
        if w_qkv_q is None:
            w_qkv_q = (
                _quantize_mixed_weight_bf16(w_qkv_bf16)
                if ctx.mixed_localcta_dgrad
                else _quantize_weight_row_col_bf16(w_qkv_bf16)
            )
        else:
            torch.cuda.current_stream().wait_stream(weight_quant_stream)
            _record_stream_tree(w_qkv_q, torch.cuda.current_stream())
        _mxfp4_stage_end("qkv_fwd_quant_w", debug_name, qkv_substage)
        q_row_blocks = q_dim // 128
        k_row_blocks = k_dim // 128
        rope_live64 = False
        if use_mxfp4_qkv_direct_outputs():
            bf16_q_forward = use_mxfp4_qkv_bf16_q_forward()
            bf16_kv_forward = use_mxfp4_qkv_bf16_kv_forward()
            normed_bf16_forward = None
            if bf16_q_forward or bf16_kv_forward:
                normed_bf16_forward, _ = _rmsnorm_to_bf16(
                    te_fused,
                    inp,
                    nw,
                    float(epsilon),
                    kind="qkv",
                )
            q = (
                None
                if bf16_q_forward
                else torch.empty(M, q_dim, dtype=torch.bfloat16, device=inp.device)
            )
            k = (
                None
                if bf16_kv_forward
                else torch.empty(M, k_dim, dtype=torch.bfloat16, device=inp.device)
            )
            v = (
                None
                if bf16_kv_forward
                else torch.empty(M, v_dim, dtype=torch.bfloat16, device=inp.device)
            )
            q_fp4 = w_qkv_q.row_fp4[:q_dim]
            q_sc = w_qkv_q.row_sc[:q_row_blocks]
            k_fp4 = w_qkv_q.row_fp4[q_dim:q_dim + k_dim]
            k_sc = w_qkv_q.row_sc[q_row_blocks:q_row_blocks + k_row_blocks]
            v_fp4 = w_qkv_q.row_fp4[q_dim + k_dim:]
            v_sc = w_qkv_q.row_sc[q_row_blocks + k_row_blocks:]
            rope_cs = rope_cos = rope_sin = rope_empty = None
            if rope_freqs_cis is not None:
                rope_applied = True
                rope_live64 = (
                    _mxfp4_qkv_rope_live64_supported(
                        M, K, q_dim, k_dim, v_dim, rope_head_dim, rope_seq_len
                    )
                    and mxfp4_rope_live_head_dim_available(rope_head_dim)
                )
                if rope_live64:
                    rope_cs = _get_mxfp4_live64_rope_cs(rope_freqs_cis, rope_seq_len)
                else:
                    rope_cos, rope_sin = _get_mxfp4_rope_tables(rope_freqs_cis, rope_seq_len)
                    rope_empty = torch.empty(0, dtype=torch.float32, device=inp.device)
            if q_dim == k_dim == v_dim and not (bf16_q_forward or bf16_kv_forward):
                qkv_substage = _mxfp4_stage_begin("qkv_fwd_gemm_batched", debug_name)
                if rope_live64:
                    mxfp4_batched_qkv_gemm_rope_live64(
                        x_q.row_fp4,
                        x_q.row_sc,
                        q_fp4,
                        q_sc,
                        k_fp4,
                        k_sc,
                        v_fp4,
                        v_sc,
                        rope_cs,
                        rope_seq_len,
                        q,
                        k,
                        v,
                    )
                elif rope_applied:
                    mxfp4_batched_gemm_rope(
                        [x_q.row_fp4, x_q.row_fp4, x_q.row_fp4],
                        [x_q.row_sc, x_q.row_sc, x_q.row_sc],
                        [q_fp4, k_fp4, v_fp4],
                        [q_sc, k_sc, v_sc],
                        [rope_cos, rope_cos, rope_empty],
                        [rope_sin, rope_sin, rope_empty],
                        [rope_seq_len, rope_seq_len, 0],
                        [rope_head_dim, rope_head_dim, 0],
                        [rope_head_dim, rope_head_dim, 0],
                        [q, k, v],
                    )
                else:
                    mxfp4_batched_gemm(
                        [x_q.row_fp4, x_q.row_fp4, x_q.row_fp4],
                        [x_q.row_sc, x_q.row_sc, x_q.row_sc],
                        [q_fp4, k_fp4, v_fp4],
                        [q_sc, k_sc, v_sc],
                        [q, k, v],
                    )
                _mxfp4_stage_end("qkv_fwd_gemm_batched", debug_name, qkv_substage)
            else:
                qkv_substage = _mxfp4_stage_begin("qkv_fwd_gemm_q", debug_name)
                if bf16_q_forward:
                    q = F.linear(normed_bf16_forward, w_qkv_bf16[:q_dim])
                    if rope_applied:
                        q = _apply_forward_rotary_tensor(
                            q,
                            rope_freqs_cis,
                            rope_batch_size,
                            rope_seq_len,
                            rope_head_dim,
                        )
                elif rope_live64:
                    _mxfp4_gemm_qkv_rope_live64(
                        x_q.row_fp4,
                        x_q.row_sc,
                        q_fp4,
                        q_sc,
                        rope_cs,
                        rope_seq_len,
                        q,
                    )
                elif rope_applied:
                    _mxfp4_gemm_qkv_rope(
                        x_q.row_fp4,
                        x_q.row_sc,
                        q_fp4,
                        q_sc,
                        rope_cos,
                        rope_sin,
                        rope_seq_len,
                        rope_head_dim,
                        rope_head_dim,
                        q,
                    )
                else:
                    _mxfp4_gemm_qkv(x_q.row_fp4, x_q.row_sc, q_fp4, q_sc, q)
                _mxfp4_stage_end("qkv_fwd_gemm_q", debug_name, qkv_substage)
                qkv_substage = _mxfp4_stage_begin("qkv_fwd_gemm_kv", debug_name)
                if bf16_kv_forward:
                    kv_bf16 = F.linear(normed_bf16_forward, w_qkv_bf16[q_dim:])
                    k = kv_bf16[:, :k_dim].contiguous()
                    v = kv_bf16[:, k_dim:].contiguous()
                    if rope_applied:
                        k = _apply_forward_rotary_tensor(
                            k,
                            rope_freqs_cis,
                            rope_batch_size,
                            rope_seq_len,
                            rope_head_dim,
                        )
                elif rope_live64:
                    mxfp4_batched_kv_gemm_rope_live64(
                        x_q.row_fp4,
                        x_q.row_sc,
                        k_fp4,
                        k_sc,
                        v_fp4,
                        v_sc,
                        rope_cs,
                        rope_seq_len,
                        k,
                        v,
                    )
                elif rope_applied:
                    mxfp4_batched_gemm_rope(
                        [x_q.row_fp4, x_q.row_fp4],
                        [x_q.row_sc, x_q.row_sc],
                        [k_fp4, v_fp4],
                        [k_sc, v_sc],
                        [rope_cos, rope_empty],
                        [rope_sin, rope_empty],
                        [rope_seq_len, 0],
                        [rope_head_dim, 0],
                        [rope_head_dim, 0],
                        [k, v],
                    )
                else:
                    mxfp4_batched_gemm(
                        [x_q.row_fp4, x_q.row_fp4],
                        [x_q.row_sc, x_q.row_sc],
                        [k_fp4, v_fp4],
                        [k_sc, v_sc],
                        [k, v],
                    )
                if use_mxfp4_qkv_forward_sync():
                    torch.cuda.current_stream().synchronize()
                _mxfp4_stage_end("qkv_fwd_gemm_kv", debug_name, qkv_substage)
        else:
            rope_applied = False
            qkv_substage = _mxfp4_stage_begin("qkv_fwd_gemm_q", debug_name)
            q = _mxfp4_gemm_qkv(
                x_q.row_fp4,
                x_q.row_sc,
                w_qkv_q.row_fp4[:q_dim],
                w_qkv_q.row_sc[:q_row_blocks],
            )
            _mxfp4_stage_end("qkv_fwd_gemm_q", debug_name, qkv_substage)
            qkv_substage = _mxfp4_stage_begin("qkv_fwd_gemm_kv", debug_name)
            kv = _mxfp4_gemm_qkv(
                x_q.row_fp4,
                x_q.row_sc,
                w_qkv_q.row_fp4[q_dim:],
                w_qkv_q.row_sc[q_row_blocks:],
            )
            _mxfp4_stage_end("qkv_fwd_gemm_kv", debug_name, qkv_substage)
            k = kv[:, :k_dim].contiguous()
            v = kv[:, k_dim:].contiguous()

        # Only the column layouts are saved for backward. The custom GEMMs may
        # still be consuming the row layouts when this autograd forward returns,
        # so register that use before the local row references are released.
        _record_mxfp4_qkv_forward_rows(x_q, w_qkv_q)
        if rope_live64:
            # The packed-RoPE pybind launch also borrows its Python argument
            # objects. Keep the complete launch set alive without synchronizing
            # the stream; completed entries are reclaimed on later QKV calls.
            _retain_mxfp4_qkv_forward_launch(
                inp.device,
                x_q,
                w_qkv_q,
                q_fp4,
                q_sc,
                k_fp4,
                k_sc,
                v_fp4,
                v_sc,
                rope_cs,
                q,
                k,
                v,
            )

        # The mixed producer emits one localCTA row and one MX RHT column for
        # the complete inverse-RoPE QKV gradient.  Splitting that matrix would
        # consume the logical SR coordinate more than once and break resume.
        use_combined_bwd = (
            True
            if ctx.mixed_localcta_dgrad
            else use_mxfp4_qkv_combined_bwd(M, total_out)
        )

        if ctx.mixed_localcta_dgrad:
            saved_tensors = [
                inp,
                nw,
                inv_rms,
                x_q.col_fp4,
                x_q.col_sc,
                w_qkv_q.local_col_fp4,
                w_qkv_q.local_col_sc,
                w_qkv_q.local_col_sg,
            ]
            ctx._mixed_weight_keepalive = w_qkv_q.keepalive
        else:
            saved_tensors = [
                inp,
                nw,
                inv_rms,
                x_q.col_fp4,
                x_q.col_sc,
                w_qkv_q.col_fp4,
                w_qkv_q.col_sc,
            ]
        ctx.use_bf16_dgrad = use_mxfp4_qkv_bf16_dgrad()
        if ctx.use_bf16_dgrad:
            saved_tensors.append(w_qkv_bf16)
        ctx.save_for_backward(*saved_tensors)
        ctx.q_dim = q_dim
        ctx.k_dim = k_dim
        ctx.v_dim = v_dim
        ctx.epsilon = epsilon
        ctx.use_combined_bwd = use_combined_bwd
        ctx.rope_applied = rope_applied
        ctx.rope_live64 = rope_live64
        ctx.rope_freqs_cis = rope_freqs_cis
        ctx.rope_batch_size = rope_batch_size
        ctx.rope_seq_len = rope_seq_len
        ctx.rope_head_dim = rope_head_dim
        ctx._mxfp4_debug_name = debug_name
        _mxfp4_stage_end("qkv_fwd", debug_name, stage_start)
        return q.contiguous(), k, v

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor, grad_v: torch.Tensor):
        debug_name = getattr(ctx, "_mxfp4_debug_name", None)
        qkv_sr_key = _mxfp4_grad_producer_key(debug_name, "qkv")
        stage_start = _mxfp4_stage_begin("qkv_bwd", debug_name)
        te_fused = _get_te_fused()
        q_dim = ctx.q_dim
        k_dim = ctx.k_dim
        inp, nw, inv_rms, *rest = ctx.saved_tensors
        _drain_mxfp4_fsdp_backward_prefetch(inp.device)

        if not ctx.fast_path:
            w_qkv = rest[0]
            dY = torch.cat([
                _as_contiguous_bf16(grad_q),
                _as_contiguous_bf16(grad_k),
                _as_contiguous_bf16(grad_v),
            ], dim=1)
            normed, _ = te_fused.fused_rmsnorm_only(inp, nw, float(ctx.epsilon))
            dx_normed = dY.matmul(w_qkv)
            grad_w = dY.transpose(0, 1).matmul(normed)
            grad_input, grad_norm = _mxfp4_rmsnorm_backward(te_fused, dx_normed, inp, nw, inv_rms)
            _mxfp4_stage_end("qkv_bwd", debug_name, stage_start)
            return (
                grad_input, grad_w, grad_norm, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None,
                None,
            )

        if getattr(ctx, "mixed_localcta_dgrad", False):
            (
                x_col_fp4,
                x_col_sc,
                w_qkv_col_fp4,
                w_qkv_col_sc,
                w_qkv_col_sg,
            ) = rest[:5]
            w_qkv_bf16 = (
                rest[5] if getattr(ctx, "use_bf16_dgrad", False) else None
            )
        else:
            x_col_fp4, x_col_sc, w_qkv_col_fp4, w_qkv_col_sc = rest[:4]
            w_qkv_bf16 = (
                rest[4] if getattr(ctx, "use_bf16_dgrad", False) else None
            )
        gq = _as_contiguous_bf16(grad_q)
        gk = _as_contiguous_bf16(grad_k)
        gv = _as_contiguous_bf16(grad_v)
        rope_live64_cs = None
        # The fused split3 producer has a legacy one-boolean SR ABI and would
        # round both row and column payloads.  Preserve localCTA-equivalent
        # row-only semantics by taking the BF16 inverse-RoPE + decomposed
        # orientation path when a one-axis policy is requested.
        use_split3_grad_fast = _use_mxfp4_qkv_split3_grad_fast_for_route(
            getattr(ctx, "mixed_localcta_dgrad", False)
        )
        use_bf16_wgrad = use_mxfp4_qkv_bf16_wgrad()
        use_bf16_dgrad = getattr(ctx, "use_bf16_dgrad", False)
        if getattr(ctx, "rope_applied", False):
            if (
                getattr(ctx, "rope_live64", False)
                and ctx.rope_head_dim == 64
                and use_split3_grad_fast
            ):
                rope_live64_cs = _get_mxfp4_live64_rope_cs(
                    ctx.rope_freqs_cis, ctx.rope_seq_len
                )
            else:
                gq, gk = _apply_inverse_rotary_qk(
                    gq,
                    gk,
                    ctx.rope_freqs_cis,
                    ctx.rope_batch_size,
                    ctx.rope_seq_len,
                    ctx.rope_head_dim,
                )
        gq_bf16 = gq
        gk_bf16 = gk
        if rope_live64_cs is not None and (use_bf16_wgrad or use_bf16_dgrad):
            gq_bf16, gk_bf16 = _apply_inverse_rotary_qk(
                gq,
                gk,
                ctx.rope_freqs_cis,
                ctx.rope_batch_size,
                ctx.rope_seq_len,
                ctx.rope_head_dim,
            )
        if use_bf16_dgrad and use_bf16_wgrad:
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_bf16", debug_name)
            qkv_grad_bf16 = torch.cat([gq_bf16, gk_bf16, gv], dim=1)
            normed_wgrad, _ = _rmsnorm_to_bf16(
                te_fused,
                inp,
                nw,
                float(ctx.epsilon),
                kind="qkv",
            )
            dx_normed = qkv_grad_bf16.matmul(w_qkv_bf16)
            grad_w = qkv_grad_bf16.transpose(0, 1).matmul(normed_wgrad)
            grad_input, grad_norm = _mxfp4_rmsnorm_backward(
                te_fused,
                dx_normed,
                inp,
                nw,
                inv_rms,
            )
            _mxfp4_stage_end("qkv_bwd_bf16", debug_name, qkv_substage)
            _mxfp4_stage_end("qkv_bwd", debug_name, stage_start)
            return (
                grad_input, grad_w, grad_norm, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None,
                None,
            )
        use_wgrad_overlap = (
            inp.size(0) >= 65536
            and use_mxfp4_qkv_wgrad_overlap()
            and not use_bf16_wgrad
        )
        defer_qkv_wgrad_for_async_rms = (
            use_mxfp4_qkv_async_rmsnorm_bwd()
            and not ctx.h_tile
            and not use_bf16_wgrad
            and not use_wgrad_overlap
        )
        qkv_state_handle = _get_mxfp4_qkv_bwd_state(
            inp.size(0),
            inp.size(1),
            q_dim,
            k_dim,
            ctx.v_dim,
            inp.device,
            protect_async_reuse=use_wgrad_overlap,
        )
        qkv_state = qkv_state_handle.state
        grad_w = torch.empty(q_dim + k_dim + ctx.v_dim, inp.size(1), dtype=torch.bfloat16, device=inp.device)
        if getattr(ctx, "use_combined_bwd", False):
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_quant", debug_name)
            if getattr(ctx, "mixed_localcta_dgrad", False):
                gall_q = _quantize_mixed_grad_dy_bf16(
                    torch.cat([gq, gk, gv], dim=1),
                    producer_key=qkv_sr_key,
                )
            elif use_split3_grad_fast:
                gall_q = _quantize_split3_qkv_grads(
                    qkv_state,
                    gq,
                    gk,
                    gv,
                    rope_cs=rope_live64_cs,
                    rope_seq_len=ctx.rope_seq_len if rope_live64_cs is not None else 0,
                )
            else:
                gall_q = _quantize_row_col_bf16(
                    torch.cat([gq, gk, gv], dim=1),
                    role="grad",
                    producer_key=qkv_sr_key,
                )
            _mxfp4_stage_end("qkv_bwd_quant", debug_name, qkv_substage)
            if use_wgrad_overlap:
                qkv_substage = _mxfp4_stage_begin("qkv_bwd_wgrad_launch", debug_name)
                wgrad_stream = _get_mxfp4_bwd_side_stream()
                wgrad_stream.wait_stream(torch.cuda.current_stream())
                _record_stream_tree(gall_q.col_fp4, wgrad_stream)
                _record_stream_tree(gall_q.col_sc, wgrad_stream)
                _record_stream_tree(x_col_fp4, wgrad_stream)
                _record_stream_tree(x_col_sc, wgrad_stream)
                _record_stream_tree(grad_w, wgrad_stream)
                with torch.cuda.stream(wgrad_stream):
                    _mxfp4_gemm_wgrad(gall_q.col_fp4, gall_q.col_sc, x_col_fp4, x_col_sc, grad_w)
                    if qkv_state_handle.entry is not None and qkv_state_handle.slot_idx is not None:
                        qkv_state_handle.entry["events"][qkv_state_handle.slot_idx].record(wgrad_stream)
                        qkv_state_handle.entry["armed"][qkv_state_handle.slot_idx] = True
                _mxfp4_stage_end("qkv_bwd_wgrad_launch", debug_name, qkv_substage)
            use_onepass_dgrad = use_split3_grad_fast and use_mxfp4_split3_qkv_onepass_dgrad(
                inp.size(0),
                q_dim,
                k_dim,
                ctx.v_dim,
            )
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_dgrad", debug_name)
            if use_bf16_dgrad:
                qkv_grad_bf16 = torch.cat([gq_bf16, gk_bf16, gv], dim=1)
                dx_normed = qkv_grad_bf16.matmul(w_qkv_bf16)
            elif getattr(ctx, "mixed_localcta_dgrad", False):
                weight_q = _MixedMXLocalCTAWeightCarrier(
                    mx_row_fp4=_mxfp4_empty_tensor(
                        torch.float4_e2m1fn_x2, w_qkv_col_fp4.device
                    ),
                    mx_row_sc=_mxfp4_empty_tensor(
                        torch.uint8, w_qkv_col_fp4.device
                    ),
                    local_col_fp4=w_qkv_col_fp4,
                    local_col_sc=w_qkv_col_sc,
                    local_col_sg=w_qkv_col_sg,
                    shape=(q_dim + k_dim + ctx.v_dim, inp.size(1)),
                    keepalive=getattr(ctx, "_mixed_weight_keepalive", ()),
                )
                dx_normed = _mixed_localcta_dgrad(
                    gall_q,
                    weight_q,
                    qkv_state["dx"],
                )
            elif use_onepass_dgrad:
                q_packed = q_dim // 2
                k_packed = k_dim // 2
                v_packed = ctx.v_dim // 2
                q_sc_cols = q_dim // 128
                k_sc_cols = k_dim // 128
                v_sc_cols = ctx.v_dim // 128
                w_qkv_col_fp4_u8 = w_qkv_col_fp4.view(torch.uint8)
                w_qkv_col_sc_u8 = w_qkv_col_sc.view(torch.uint8)
                try:
                    dx_normed = qkv_state["dx"]
                    mxfp4_split3_dgrad_strided_onepass_gemm(
                        gall_q.row_fp4,
                        [
                            gall_q.row_sc.narrow(1, 0, q_sc_cols),
                            gall_q.row_sc.narrow(1, q_sc_cols, k_sc_cols),
                            gall_q.row_sc.narrow(1, q_sc_cols + k_sc_cols, v_sc_cols),
                        ],
                        [0, q_packed, q_packed + k_packed],
                        [q_packed, k_packed, v_packed],
                        [
                            w_qkv_col_fp4_u8.narrow(1, 0, q_packed).contiguous().view(torch.float4_e2m1fn_x2),
                            w_qkv_col_fp4_u8.narrow(1, q_packed, k_packed).contiguous().view(torch.float4_e2m1fn_x2),
                            w_qkv_col_fp4_u8.narrow(1, q_packed + k_packed, v_packed).contiguous().view(torch.float4_e2m1fn_x2),
                        ],
                        [
                            w_qkv_col_sc_u8.narrow(1, 0, q_sc_cols).contiguous(),
                            w_qkv_col_sc_u8.narrow(1, q_sc_cols, k_sc_cols).contiguous(),
                            w_qkv_col_sc_u8.narrow(1, q_sc_cols + k_sc_cols, v_sc_cols).contiguous(),
                        ],
                        dx_normed,
                        config_idx=mxfp4_split3_qkv_onepass_config_idx(),
                    )
                except AttributeError:
                    dx_normed = _mxfp4_gemm_qkv(gall_q.row_fp4, gall_q.row_sc, w_qkv_col_fp4, w_qkv_col_sc)
            else:
                dx_normed = _mxfp4_gemm_qkv(gall_q.row_fp4, gall_q.row_sc, w_qkv_col_fp4, w_qkv_col_sc)
            _mxfp4_stage_end("qkv_bwd_dgrad", debug_name, qkv_substage)
            if use_bf16_wgrad:
                qkv_substage = _mxfp4_stage_begin("qkv_bwd_wgrad_bf16", debug_name)
                normed_wgrad, _ = _rmsnorm_to_bf16(te_fused, inp, nw, float(ctx.epsilon), kind="qkv")
                grad_w = torch.cat(
                    [gq_bf16, gk_bf16, gv], dim=1
                ).transpose(0, 1).matmul(normed_wgrad)
                _mxfp4_stage_end("qkv_bwd_wgrad_bf16", debug_name, qkv_substage)
            elif not use_wgrad_overlap and not defer_qkv_wgrad_for_async_rms:
                qkv_substage = _mxfp4_stage_begin("qkv_bwd_wgrad", debug_name)
                _mxfp4_gemm_wgrad(gall_q.col_fp4, gall_q.col_sc, x_col_fp4, x_col_sc, grad_w)
                _mxfp4_stage_end("qkv_bwd_wgrad", debug_name, qkv_substage)
        else:
            gall_q = None
            use_split3_fallback = use_split3_grad_fast
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_quant", debug_name)
            if use_split3_fallback:
                gall_q = _quantize_split3_qkv_grads(
                    qkv_state,
                    gq,
                    gk,
                    gv,
                    rope_cs=rope_live64_cs,
                    rope_seq_len=ctx.rope_seq_len if rope_live64_cs is not None else 0,
                )
                gq_q, gk_q, gv_q = _split_qkv_rowcol(gall_q, q_dim, k_dim, ctx.v_dim)
            else:
                gq_q = _quantize_row_col_bf16(
                    gq, role="grad", producer_key=qkv_sr_key
                )
                gk_q = _quantize_row_col_bf16(
                    gk, role="grad", producer_key=qkv_sr_key
                )
                gv_q = _quantize_row_col_bf16(
                    gv, role="grad", producer_key=qkv_sr_key
                )
            _mxfp4_stage_end("qkv_bwd_quant", debug_name, qkv_substage)
            q_col_blocks = q_dim // 128
            k_col_blocks = k_dim // 128
            v_col_blocks = ctx.v_dim // 128
            w_qkv_col_fp4_u8 = w_qkv_col_fp4.view(torch.uint8)
            q_col_fp4 = w_qkv_col_fp4_u8.narrow(1, 0, q_dim // 2).contiguous().view(torch.float4_e2m1fn_x2)
            k_col_fp4 = w_qkv_col_fp4_u8.narrow(1, q_dim // 2, k_dim // 2).contiguous().view(torch.float4_e2m1fn_x2)
            v_col_fp4 = w_qkv_col_fp4_u8.narrow(1, (q_dim + k_dim) // 2, ctx.v_dim // 2).contiguous().view(torch.float4_e2m1fn_x2)
            q_col_sc = w_qkv_col_sc[:, :q_col_blocks].contiguous()
            k_col_sc = w_qkv_col_sc[:, q_col_blocks : q_col_blocks + k_col_blocks].contiguous()
            v_col_sc = w_qkv_col_sc[:, q_col_blocks + k_col_blocks : q_col_blocks + k_col_blocks + v_col_blocks].contiguous()
            if use_wgrad_overlap:
                qkv_substage = _mxfp4_stage_begin("qkv_bwd_wgrad_launch", debug_name)
                wgrad_stream = _get_mxfp4_bwd_side_stream()
                wgrad_stream.wait_stream(torch.cuda.current_stream())
                if gall_q is not None:
                    _record_stream_tree(gall_q.col_fp4, wgrad_stream)
                    _record_stream_tree(gall_q.col_sc, wgrad_stream)
                else:
                    _record_stream_tree(gq_q.col_fp4, wgrad_stream)
                    _record_stream_tree(gq_q.col_sc, wgrad_stream)
                    _record_stream_tree(gk_q.col_fp4, wgrad_stream)
                    _record_stream_tree(gk_q.col_sc, wgrad_stream)
                    _record_stream_tree(gv_q.col_fp4, wgrad_stream)
                    _record_stream_tree(gv_q.col_sc, wgrad_stream)
                _record_stream_tree(x_col_fp4, wgrad_stream)
                _record_stream_tree(x_col_sc, wgrad_stream)
                _record_stream_tree(grad_w, wgrad_stream)
                with torch.cuda.stream(wgrad_stream):
                    _mxfp4_gemm_wgrad(gq_q.col_fp4, gq_q.col_sc, x_col_fp4, x_col_sc, grad_w[:q_dim])
                    _mxfp4_gemm_wgrad(gk_q.col_fp4, gk_q.col_sc, x_col_fp4, x_col_sc, grad_w[q_dim:q_dim + k_dim])
                    _mxfp4_gemm_wgrad(gv_q.col_fp4, gv_q.col_sc, x_col_fp4, x_col_sc, grad_w[q_dim + k_dim:])
                    if gall_q is not None and qkv_state_handle.entry is not None and qkv_state_handle.slot_idx is not None:
                        qkv_state_handle.entry["events"][qkv_state_handle.slot_idx].record(wgrad_stream)
                        qkv_state_handle.entry["armed"][qkv_state_handle.slot_idx] = True
                _mxfp4_stage_end("qkv_bwd_wgrad_launch", debug_name, qkv_substage)
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_dgrad", debug_name)
            if use_bf16_dgrad:
                qkv_grad_bf16 = torch.cat([gq_bf16, gk_bf16, gv], dim=1)
                dx_normed = qkv_grad_bf16.matmul(w_qkv_bf16)
            else:
                dx_normed = _mxfp4_gemm_qkv(gq_q.row_fp4, gq_q.row_sc, q_col_fp4, q_col_sc)
                dx_normed.add_(_mxfp4_gemm_qkv(gk_q.row_fp4, gk_q.row_sc, k_col_fp4, k_col_sc))
                dx_normed.add_(_mxfp4_gemm_qkv(gv_q.row_fp4, gv_q.row_sc, v_col_fp4, v_col_sc))
            _mxfp4_stage_end("qkv_bwd_dgrad", debug_name, qkv_substage)
            if use_bf16_wgrad:
                qkv_substage = _mxfp4_stage_begin("qkv_bwd_wgrad_bf16", debug_name)
                normed_wgrad, _ = _rmsnorm_to_bf16(te_fused, inp, nw, float(ctx.epsilon), kind="qkv")
                grad_w = torch.cat(
                    [gq_bf16, gk_bf16, gv], dim=1
                ).transpose(0, 1).matmul(normed_wgrad)
                _mxfp4_stage_end("qkv_bwd_wgrad_bf16", debug_name, qkv_substage)
            elif not use_wgrad_overlap and not defer_qkv_wgrad_for_async_rms:
                qkv_substage = _mxfp4_stage_begin("qkv_bwd_wgrad", debug_name)
                _mxfp4_gemm_wgrad(gq_q.col_fp4, gq_q.col_sc, x_col_fp4, x_col_sc, grad_w[:q_dim])
                _mxfp4_gemm_wgrad(gk_q.col_fp4, gk_q.col_sc, x_col_fp4, x_col_sc, grad_w[q_dim:q_dim + k_dim])
                _mxfp4_gemm_wgrad(gv_q.col_fp4, gv_q.col_sc, x_col_fp4, x_col_sc, grad_w[q_dim + k_dim:])
                _mxfp4_stage_end("qkv_bwd_wgrad", debug_name, qkv_substage)
        qkv_wgrad_waited_before_rmsnorm = False
        if use_wgrad_overlap and use_mxfp4_qkv_wgrad_wait_before_rmsnorm():
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_wait_before_rmsnorm", debug_name)
            torch.cuda.current_stream().wait_stream(wgrad_stream)
            _mxfp4_stage_end("qkv_bwd_wait_before_rmsnorm", debug_name, qkv_substage)
            qkv_wgrad_waited_before_rmsnorm = True
        qkv_wait_stream_before_dgamma = None
        if (
            use_wgrad_overlap
            and not qkv_wgrad_waited_before_rmsnorm
            and use_mxfp4_qkv_wgrad_wait_before_rmsnorm_dgamma()
        ):
            qkv_wait_stream_before_dgamma = wgrad_stream
            qkv_wgrad_waited_before_rmsnorm = True
        if ctx.h_tile:
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_h_tile", debug_name)
            grad_input, grad_norm = mxfp4_h_tile_backward(
                dx_normed, inp, nw, inv_rms
            )
            _mxfp4_stage_end("qkv_bwd_h_tile", debug_name, qkv_substage)
        elif defer_qkv_wgrad_for_async_rms:
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_rmsnorm_launch", debug_name)
            # Keep QKV RMSNorm and wgrad ordered on the compute stream. The
            # side-stream overlap can leave clustered MXFP4 work live across
            # the next FSDP collective and has produced delayed launch failures
            # in production-shape training. This ordering retained the same
            # throughput in the 477-step control while completing cleanly.
            qkv_rms_state, qkv_rms_stream = _launch_rmsnorm_bwd_out_async(
                dx_normed,
                inp,
                nw,
                inv_rms,
                te_fused,
                owner_key=debug_name,
                tag="mxfp4_qkv",
                force_current_stream=True,
                force_single_out=False,
                force_fp32_dgamma=True,
            )
            _mxfp4_stage_end("qkv_bwd_rmsnorm_launch", debug_name, qkv_substage)
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_wgrad", debug_name)
            if getattr(ctx, "use_combined_bwd", False):
                _mxfp4_gemm_wgrad(gall_q.col_fp4, gall_q.col_sc, x_col_fp4, x_col_sc, grad_w)
            else:
                _mxfp4_gemm_wgrad(gq_q.col_fp4, gq_q.col_sc, x_col_fp4, x_col_sc, grad_w[:q_dim])
                _mxfp4_gemm_wgrad(gk_q.col_fp4, gk_q.col_sc, x_col_fp4, x_col_sc, grad_w[q_dim:q_dim + k_dim])
                _mxfp4_gemm_wgrad(gv_q.col_fp4, gv_q.col_sc, x_col_fp4, x_col_sc, grad_w[q_dim + k_dim:])
            _mxfp4_stage_end("qkv_bwd_wgrad", debug_name, qkv_substage)
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_wait_after_rmsnorm", debug_name)
            torch.cuda.current_stream().wait_stream(qkv_rms_stream)
            _mxfp4_stage_end("qkv_bwd_wait_after_rmsnorm", debug_name, qkv_substage)
            grad_input = qkv_rms_state["grad_input"]
            grad_norm = qkv_rms_state["dgamma"]
        else:
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_rmsnorm", debug_name)
            grad_input, grad_norm = _mxfp4_rmsnorm_backward(
                te_fused,
                dx_normed,
                inp,
                nw,
                inv_rms,
                wait_stream_before_dgamma=qkv_wait_stream_before_dgamma,
            )
            _mxfp4_stage_end("qkv_bwd_rmsnorm", debug_name, qkv_substage)
        if use_wgrad_overlap and not qkv_wgrad_waited_before_rmsnorm:
            qkv_substage = _mxfp4_stage_begin("qkv_bwd_wait_after_rmsnorm", debug_name)
            torch.cuda.current_stream().wait_stream(wgrad_stream)
            _mxfp4_stage_end("qkv_bwd_wait_after_rmsnorm", debug_name, qkv_substage)
        _mxfp4_stage_end("qkv_bwd", debug_name, stage_start)
        return (
            grad_input, grad_w, grad_norm, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
            None,
        )


class _WoFunction_MXFP4_TK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        wo_weight: torch.Tensor,
        debug_name: str | None = None,
        residual: torch.Tensor | None = None,
        h_gamma: torch.Tensor | None = None,
    ):
        stage_start = _mxfp4_stage_begin("wo_fwd", debug_name)
        inp = _as_contiguous_bf16(input)
        res = _as_contiguous_bf16(residual) if residual is not None else None
        w = _as_contiguous_bf16(wo_weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        if res is not None and res.shape != (M, N):
            raise RuntimeError(f"MXFP4 Wo residual shape {tuple(res.shape)} does not match output {(M, N)}")
        ctx.has_residual = res is not None
        ctx.h_carrier = h_gamma is not None
        if ctx.h_carrier and res is None:
            raise RuntimeError("MX H Wo carrier requires the residual stream")

        supported = _mxfp4_supported(M, K) and _mxfp4_supported(N, K)
        _require_mixed_localcta_supported_path("Wo", supported)
        if not supported:
            ctx.fast_path = False
            ctx.save_for_backward(inp, wo_weight)
            ctx._mxfp4_debug_name = debug_name
            _mxfp4_stage_end("wo_fwd", debug_name, stage_start)
            if ctx.h_carrier:
                raise RuntimeError("MX H Wo carrier requires the native supported shape")
            y = inp.matmul(wo_weight.t())
            if res is not None:
                y = y + res
            return y

        ctx.fast_path = True
        x_q = _quantize_row_col_bf16(inp, role="activation")
        ctx.mixed_localcta_dgrad = use_mxfp4_localcta_dgrad()
        w_q = (
            _quantize_mixed_weight_bf16(w)
            if ctx.mixed_localcta_dgrad
            else _quantize_weight_row_col_bf16(w)
        )
        if ctx.h_carrier:
            carrier = mxfp4_h_residual_carrier(
                x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc,
                res, _as_contiguous_bf16(h_gamma), 1.0e-5,
            )
            y = carrier.z_out
        elif res is not None and use_mxfp4_residual_fusion_attn():
            y = mxfp4_gemm_residual(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc, res)
        else:
            y = _mxfp4_gemm_qkv(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc)
            if res is not None:
                y = y + res
        if ctx.mixed_localcta_dgrad:
            ctx.save_for_backward(
                x_q.col_fp4,
                x_q.col_sc,
                w_q.local_col_fp4,
                w_q.local_col_sc,
                w_q.local_col_sg,
            )
            ctx._mixed_weight_keepalive = w_q.keepalive
        else:
            ctx.save_for_backward(
                x_q.col_fp4,
                x_q.col_sc,
                w_q.col_fp4,
                w_q.col_sc,
            )
        ctx._mxfp4_debug_name = debug_name
        _mxfp4_stage_end("wo_fwd", debug_name, stage_start)
        if ctx.h_carrier:
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(
                carrier.row_fp4, carrier.row_sc, carrier.col_fp4,
                carrier.col_sc, carrier.r_tile,
            )
            return (
                y, carrier.row_fp4, carrier.row_sc, carrier.col_fp4,
                carrier.col_sc, carrier.r_tile,
            )
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *carrier_grads):
        debug_name = getattr(ctx, "_mxfp4_debug_name", None)
        wo_sr_key = _mxfp4_grad_producer_key(debug_name, "wo")
        stage_start = _mxfp4_stage_begin("wo_bwd", debug_name)
        dY = _as_contiguous_bf16(grad_output)
        if not ctx.fast_path:
            inp, wo_weight = ctx.saved_tensors
            grad_input = dY.matmul(wo_weight)
            grad_w = dY.transpose(0, 1).matmul(inp)
            _mxfp4_stage_end("wo_bwd", debug_name, stage_start)
            grad_residual = dY if ctx.has_residual else None
            return grad_input, grad_w, None, grad_residual, None

        if getattr(ctx, "mixed_localcta_dgrad", False):
            (
                x_col_fp4,
                x_col_sc,
                w_col_fp4,
                w_col_sc,
                w_col_sg,
            ) = ctx.saved_tensors
        else:
            x_col_fp4, x_col_sc, w_col_fp4, w_col_sc = ctx.saved_tensors
        wo_substage = _mxfp4_stage_begin("wo_bwd_quant_dy", debug_name)
        dY_q = (
            _quantize_mixed_grad_dy_bf16(dY, producer_key=wo_sr_key)
            if getattr(ctx, "mixed_localcta_dgrad", False)
            else _quantize_row_col_bf16(
                dY, role="grad", producer_key=wo_sr_key
            )
        )
        _mxfp4_stage_end("wo_bwd_quant_dy", debug_name, wo_substage)
        wo_substage = _mxfp4_stage_begin("wo_bwd_dgrad_gemm", debug_name)
        if getattr(ctx, "mixed_localcta_dgrad", False):
            weight_q = _MixedMXLocalCTAWeightCarrier(
                mx_row_fp4=_mxfp4_empty_tensor(
                    torch.float4_e2m1fn_x2, w_col_fp4.device
                ),
                mx_row_sc=_mxfp4_empty_tensor(torch.uint8, w_col_fp4.device),
                local_col_fp4=w_col_fp4,
                local_col_sc=w_col_sc,
                local_col_sg=w_col_sg,
                shape=(dY_q.shape[1], x_col_fp4.shape[0]),
                keepalive=getattr(ctx, "_mixed_weight_keepalive", ()),
            )
            grad_input = _mixed_localcta_dgrad(dY_q, weight_q)
        else:
            grad_input = _mxfp4_gemm_wo_dgrad(
                dY_q.row_fp4, dY_q.row_sc, w_col_fp4, w_col_sc
            )
        _mxfp4_stage_end("wo_bwd_dgrad_gemm", debug_name, wo_substage)
        wo_substage = _mxfp4_stage_begin("wo_bwd_wgrad_gemm", debug_name)
        grad_w = _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)
        _mxfp4_stage_end("wo_bwd_wgrad_gemm", debug_name, wo_substage)
        _mxfp4_stage_end("wo_bwd", debug_name, stage_start)
        grad_residual = dY if ctx.has_residual else None
        return grad_input, grad_w, None, grad_residual, None


class _WoNHSDQuantFunction_MXFP4_TK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        wo_weight: torch.Tensor,
        debug_name: str | None = None,
        residual: torch.Tensor | None = None,
    ):
        stage_start = _mxfp4_stage_begin("wo_nhsd_fwd", debug_name)
        inp = input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        w = _as_contiguous_bf16(wo_weight.detach())
        B, H, S, D = inp.shape
        M = B * S
        K = H * D
        N = w.shape[0]
        res = _as_contiguous_bf16(residual) if residual is not None else None
        if res is not None and res.shape != (M, N):
            raise RuntimeError(f"MXFP4 Wo residual shape {tuple(res.shape)} does not match output {(M, N)}")
        ctx.has_residual = res is not None
        ctx.input_shape = (B, H, S, D)
        ctx._mxfp4_debug_name = debug_name

        if not (_mxfp4_supported(M, K) and _mxfp4_supported(N, K) and D % 64 == 0 and S % 128 == 0):
            ctx.fast_path = False
            matrix = inp.transpose(1, 2).contiguous().view(M, K)
            ctx.save_for_backward(matrix, wo_weight)
            y = matrix.matmul(wo_weight.t())
            if res is not None:
                y = y + res
            _mxfp4_stage_end("wo_nhsd_fwd", debug_name, stage_start)
            return y

        ctx.fast_path = True
        x_q = _quantize_nhsd_wo_row_col_bf16(inp, role="activation")
        w_q = _quantize_weight_row_col_bf16(w)
        if res is not None and use_mxfp4_residual_fusion_attn():
            y = mxfp4_gemm_residual(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc, res)
        else:
            y = _mxfp4_gemm_qkv(x_q.row_fp4, x_q.row_sc, w_q.row_fp4, w_q.row_sc)
            if res is not None:
                y = y + res
        ctx.save_for_backward(x_q.col_fp4, x_q.col_sc, w_q.col_fp4, w_q.col_sc)
        _mxfp4_stage_end("wo_nhsd_fwd", debug_name, stage_start)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        debug_name = getattr(ctx, "_mxfp4_debug_name", None)
        wo_sr_key = _mxfp4_grad_producer_key(debug_name, "wo")
        stage_start = _mxfp4_stage_begin("wo_nhsd_bwd", debug_name)
        dY = _as_contiguous_bf16(grad_output)
        B, H, S, D = ctx.input_shape
        M = B * S
        K = H * D

        if not ctx.fast_path:
            inp_matrix, wo_weight = ctx.saved_tensors
            grad_matrix = dY.matmul(wo_weight)
            grad_w = dY.transpose(0, 1).matmul(inp_matrix)
            _mxfp4_stage_end("wo_nhsd_bwd", debug_name, stage_start)
            grad_input = grad_matrix.view(B, S, H, D).transpose(1, 2)
            grad_residual = dY if ctx.has_residual else None
            return grad_input, grad_w, None, grad_residual

        x_col_fp4, x_col_sc, w_col_fp4, w_col_sc = ctx.saved_tensors
        dY_q = _quantize_row_col_bf16(
            dY, role="grad", producer_key=wo_sr_key
        )
        grad_matrix = _mxfp4_gemm_qkv(dY_q.row_fp4, dY_q.row_sc, w_col_fp4, w_col_sc)
        grad_w = _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, x_col_fp4, x_col_sc)
        _mxfp4_stage_end("wo_nhsd_bwd", debug_name, stage_start)
        grad_input = grad_matrix.view(B, S, H, D).transpose(1, 2)
        grad_residual = dY if ctx.has_residual else None
        return grad_input, grad_w, None, grad_residual


class _FusedFFNFunctionV2_MXFP4_TK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        w1_weight: torch.Tensor,
        w3_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
        debug_name: str | None = None,
        residual: torch.Tensor | None = None,
        h_row_fp4: torch.Tensor | None = None,
        h_row_sc: torch.Tensor | None = None,
        h_col_fp4: torch.Tensor | None = None,
        h_col_sc: torch.Tensor | None = None,
        h_r_tile: torch.Tensor | None = None,
        h_next_gamma: torch.Tensor | None = None,
        cde_emit: bool = False,
    ):
        stage_start = _mxfp4_stage_begin("ffn_fwd", debug_name)
        inp = _as_contiguous_bf16(input)
        res = _as_contiguous_bf16(residual) if residual is not None else None
        nw = _as_contiguous_bf16(norm_weight.detach())
        M, K = inp.shape
        H = w1_weight.shape[0]
        N = w2_weight.shape[0]
        if res is not None and res.shape != (M, N):
            raise RuntimeError(f"MXFP4 FFN residual shape {tuple(res.shape)} does not match output {(M, N)}")
        ctx.has_residual = res is not None
        ctx.h_tile = h_row_fp4 is not None and h_row_fp4.numel() != 0
        ctx.h_output = h_next_gamma is not None
        ctx.cde_output = bool(cde_emit)
        if (ctx.h_tile or ctx.h_output) and ctx.cde_output:
            raise RuntimeError("exact C/D/E and MX H FFN carriers are mutually exclusive")
        if ctx.cde_output and res is None:
            raise RuntimeError("MX exact C/D/E W2 producer requires the residual stream")
        te_fused = _get_te_fused()

        supported = (
            _mxfp4_supported(M, K)
            and _mxfp4_supported(2 * H, K)
            and _mxfp4_supported(N, H)
        )
        _require_mixed_localcta_supported_path("FFN", supported)

        if not supported:
            if ctx.h_tile or ctx.h_output or ctx.cde_output:
                raise RuntimeError("MX FFN carrier requires the native supported shape")
            ctx.fast_path = False
            normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
            h1 = normed.matmul(w1_weight.t())
            h3 = normed.matmul(w3_weight.t())
            h = F.silu(h1.float()).to(torch.bfloat16) * h3
            y = h.matmul(w2_weight.t())
            if res is not None:
                y = y + res
            ctx.save_for_backward(inp, nw, inv_rms, normed, h1, h3, h, w1_weight, w3_weight, w2_weight)
            ctx.epsilon = epsilon
            ctx._mxfp4_debug_name = debug_name
            _mxfp4_stage_end("ffn_fwd", debug_name, stage_start)
            return y

        ctx.fast_path = True
        ctx.mixed_localcta_dgrad = use_mxfp4_localcta_dgrad()
        w1_q = None
        w3_q = None
        w13_weight_quant_stream = None
        if use_mxfp4_ffn_fwd_w13_weight_quant_overlap():
            w1_bf16 = _as_contiguous_bf16(w1_weight.detach())
            w3_bf16 = _as_contiguous_bf16(w3_weight.detach())
            w13_weight_quant_stream = _get_mxfp4_fwd_w13_stream()
            w13_weight_quant_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(w1_bf16, w13_weight_quant_stream)
            _record_stream_tree(w3_bf16, w13_weight_quant_stream)
            with torch.cuda.stream(w13_weight_quant_stream):
                if ctx.mixed_localcta_dgrad:
                    w1_q = _quantize_mixed_weight_bf16(w1_bf16)
                    w3_q = _quantize_mixed_weight_bf16(w3_bf16)
                else:
                    w1_q = _quantize_weight_row_col_bf16(w1_bf16)
                    w3_q = _quantize_weight_row_col_bf16(w3_bf16)
        substage = _mxfp4_stage_begin("ffn_fwd_rms_quant_x", debug_name)
        if ctx.h_tile:
            x_q = _MXFP4RowCol(h_row_fp4, h_row_sc, h_col_fp4, h_col_sc)
            inv_rms = h_r_tile
        else:
            x_q, inv_rms = _rmsnorm_quantize_row_col_bf16(
                te_fused, inp, nw, float(epsilon), kind="ffn"
            )
        _mxfp4_stage_end("ffn_fwd_rms_quant_x", debug_name, substage)
        w2_bf16 = _as_contiguous_bf16(w2_weight.detach())
        w2_q = None
        w2_weight_quant_stream = None
        w2_weight_quant_done = None
        if (
            use_mxfp4_ffn_fwd_w2_weight_quant_overlap()
            and not ctx.mixed_localcta_dgrad
        ):
            w2_q = _lookup_weight_row_col_bf16(w2_bf16)
        substage = _mxfp4_stage_begin("ffn_fwd_quant_w13", debug_name)
        if w1_q is None:
            if ctx.mixed_localcta_dgrad:
                w1_q = _quantize_mixed_weight_bf16(
                    _as_contiguous_bf16(w1_weight.detach())
                )
                w3_q = _quantize_mixed_weight_bf16(
                    _as_contiguous_bf16(w3_weight.detach())
                )
            else:
                w1_q = _quantize_weight_row_col_bf16(
                    _as_contiguous_bf16(w1_weight.detach())
                )
                w3_q = _quantize_weight_row_col_bf16(
                    _as_contiguous_bf16(w3_weight.detach())
                )
        else:
            torch.cuda.current_stream().wait_stream(w13_weight_quant_stream)
            _record_stream_tree(w1_q, torch.cuda.current_stream())
            _record_stream_tree(w3_q, torch.cuda.current_stream())
        _mxfp4_stage_end("ffn_fwd_quant_w13", debug_name, substage)
        if use_mxfp4_ffn_fwd_w2_weight_quant_overlap() and w2_q is None:
            # Keep independent quantizer instances from racing while retaining
            # the useful W2-quant/W13-GEMM overlap window.
            w2_weight_quant_stream = _get_mxfp4_fwd_side_stream()
            w2_weight_quant_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(w2_bf16, w2_weight_quant_stream)
            with torch.cuda.stream(w2_weight_quant_stream):
                w2_q = (
                    _quantize_mixed_weight_bf16(w2_bf16)
                    if ctx.mixed_localcta_dgrad
                    else _quantize_weight_row_col_bf16(w2_bf16)
                )
                w2_weight_quant_done = torch.cuda.Event()
                w2_weight_quant_done.record()
        use_packed_w13_ffn = (
            use_mxfp4_packed_w13_ffn()
            and use_mxfp4_fused_silu_ffn_quant()
            and use_mxfp4_fused_silu_deriv_split2_ffn()
            and use_mxfp4_split2_ffn_onepass_dgrad()
            and not _mxfp4_needs_opt_quant("activation")
            and not _mxfp4_needs_opt_quant("grad")
        )
        h13_packed = (
            torch.empty(M, 2 * H, dtype=torch.bfloat16, device=inp.device)
            if use_packed_w13_ffn
            else None
        )
        if h13_packed is not None:
            h1_raw = h13_packed[:, :H]
            h3 = h13_packed[:, H:]
        else:
            h1_raw = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
            h3 = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
        empty_bf16 = _mxfp4_empty_tensor(torch.bfloat16, inp.device)
        sig_h1_save = empty_bf16
        substage = _mxfp4_stage_begin("ffn_fwd_w13_gemm", debug_name)
        _mxfp4_batched_gemm_configured(
            [x_q.row_fp4, x_q.row_fp4],
            [x_q.row_sc, x_q.row_sc],
            [w1_q.row_fp4, w3_q.row_fp4],
            [w1_q.row_sc, w3_q.row_sc],
            [h1_raw, h3],
            prefix="MXFP4_FFN_W13_FWD_BATCHED_GEMM_CONFIG",
        )
        _mxfp4_stage_end("ffn_fwd_w13_gemm", debug_name, substage)
        fused_silu_opt_supported = (
            _mxfp4_needs_opt_quant("activation")
            and (
                not _mxfp4_rht_for_role("activation")
                or use_mxfp4_fused_silu_ffn_quant_rht()
            )
            and (
                not _mxfp4_data_sr_for_role("activation")
                or use_mxfp4_fused_silu_ffn_quant_data_sr()
            )
            and (
                not _mxfp4_scale_sr_for_role("activation")
                or use_mxfp4_fused_silu_ffn_quant_scale_sr()
            )
        )
        saved_sigmoid_supported = (
            use_mxfp4_saved_sigmoid_ffn()
            and h13_packed is None
            and not _mxfp4_needs_opt_quant("activation")
            and hasattr(te_fused, "fused_silu_mul_and_sigmoid_bf16_out_no_amax")
        )
        substage = _mxfp4_stage_begin("ffn_fwd_silu_quant", debug_name)
        if saved_sigmoid_supported:
            sig_h1_save = torch.empty_like(h1_raw)
            fused_saved_fwd = False
            if use_mxfp4_saved_sigmoid_fused_fwd_quant():
                try:
                    h_q = _empty_mxfp4_row_col(M, H, inp.device)
                    mxfp4_fused_silu_mul_sigmoid_quantize_row_and_col_launch_inplace(
                        h1_raw,
                        h3,
                        sig_h1_save,
                        h_q.row_fp4,
                        h_q.row_sc,
                        h_q.col_fp4,
                        h_q.col_sc,
                    )
                    fused_saved_fwd = True
                except AttributeError:
                    fused_saved_fwd = False
            if not fused_saved_fwd:
                h = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
                te_fused.fused_silu_mul_and_sigmoid_bf16_out_no_amax(h1_raw, h3, h, sig_h1_save)
                if use_mxfp4_saved_sigmoid_fwd_inplace_quant():
                    h_q = _empty_mxfp4_row_col(M, H, inp.device)
                    mxfp4_quantize_row_and_col_launch_inplace(
                        h,
                        h_q.row_fp4,
                        h_q.row_sc,
                        h_q.col_fp4,
                        h_q.col_sc,
                        1,
                    )
                else:
                    h_q = _quantize_row_col_bf16(h, role="activation")
        elif use_mxfp4_fused_silu_ffn_quant() and (
            not _mxfp4_needs_opt_quant("activation") or fused_silu_opt_supported
        ):
            try:
                if h13_packed is not None:
                    row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_mul_quantize_row_and_col_strided(
                        h13_packed,
                        H,
                        H,
                    )
                    h_q = _MXFP4RowCol(row_fp4=row_fp4, row_sc=row_sc, col_fp4=col_fp4, col_sc=col_sc)
                elif fused_silu_opt_supported:
                    h_q = _empty_mxfp4_row_col(M, H, inp.device)
                    opt_kwargs = _mxfp4_opt_kwargs("activation")
                    if _mxfp4_rht_has_row("activation"):
                        opt_kwargs["row_with_rht"] = True
                    mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace(
                        h1_raw,
                        h3,
                        h_q.row_fp4,
                        h_q.row_sc,
                        h_q.col_fp4,
                        h_q.col_sc,
                        1,
                        use_rht=_mxfp4_rht_has_col("activation"),
                        rht_block_size=_mxfp4_rht_block_size(),
                        with_random_sign_mask=_mxfp4_rht_random_sign_mask(),
                        **opt_kwargs,
                    )
                else:
                    h_q = _empty_mxfp4_row_col(M, H, inp.device)
                    mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace(
                        h1_raw,
                        h3,
                        h_q.row_fp4,
                        h_q.row_sc,
                        h_q.col_fp4,
                        h_q.col_sc,
                    )
            except AttributeError:
                h = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
                if hasattr(te_fused, "fused_silu_mul_bf16_out_no_amax"):
                    te_fused.fused_silu_mul_bf16_out_no_amax(h1_raw, h3, h)
                elif hasattr(te_fused, "fused_silu_mul_bf16_out"):
                    amax = torch.empty(1, dtype=torch.float32, device=inp.device)
                    te_fused.fused_silu_mul_bf16_out(h1_raw, h3, h, amax)
                else:
                    h = te_fused.fused_silu_mul_bf16(h1_raw, h3)[0]
                h_q = _quantize_row_col_bf16(h, role="activation")
        else:
            h = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
            if hasattr(te_fused, "fused_silu_mul_bf16_out_no_amax"):
                te_fused.fused_silu_mul_bf16_out_no_amax(h1_raw, h3, h)
            elif hasattr(te_fused, "fused_silu_mul_bf16_out"):
                amax = torch.empty(1, dtype=torch.float32, device=inp.device)
                te_fused.fused_silu_mul_bf16_out(h1_raw, h3, h, amax)
            else:
                h = te_fused.fused_silu_mul_bf16(h1_raw, h3)[0]
            h_q = _quantize_row_col_bf16(h, role="activation")
        _mxfp4_stage_end("ffn_fwd_silu_quant", debug_name, substage)
        substage = _mxfp4_stage_begin("ffn_fwd_quant_w2", debug_name)
        if w2_q is None:
            w2_q = (
                _quantize_mixed_weight_bf16(w2_bf16)
                if ctx.mixed_localcta_dgrad
                else _quantize_weight_row_col_bf16(w2_bf16)
            )
        else:
            if w2_weight_quant_stream is not None:
                torch.cuda.current_stream().wait_event(w2_weight_quant_done)
                _record_stream_tree(w2_q, torch.cuda.current_stream())
        _mxfp4_stage_end("ffn_fwd_quant_w2", debug_name, substage)
        recompute_w13 = (
            not ctx.mixed_localcta_dgrad
            and
            use_mxfp4_recompute_ffn_w13()
            and h13_packed is None
            and use_mxfp4_fused_silu_deriv_split2_ffn()
            and use_mxfp4_split2_ffn_onepass_dgrad()
            and use_mxfp4_split2_ffn_inplace_quant()
            and not use_mxfp4_w2_dgrad_silu_producer()
            and not use_mxfp4_w2_dgrad_saved_sigmoid_producer()
            and not use_mxfp4_w2_dgrad_saved_sigmoid_row_bf16_producer()
            and not _mxfp4_needs_opt_quant("grad")
        )
        empty_fp4 = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, inp.device)
        empty_sc = _mxfp4_empty_tensor(torch.uint8, inp.device)
        if recompute_w13:
            h1_save = empty_bf16
            h3_save = empty_bf16
            sig_h1_save = empty_bf16
            h13_save = empty_bf16
            x_row_fp4_save, x_row_sc_save = x_q.row_fp4, x_q.row_sc
            w1_row_fp4_save, w1_row_sc_save = w1_q.row_fp4, w1_q.row_sc
            w3_row_fp4_save, w3_row_sc_save = w3_q.row_fp4, w3_q.row_sc
            h1_raw = None
            h3 = None
            h13_packed = None
        else:
            h1_save = h1_raw
            h3_save = h3
            h13_save = h13_packed if h13_packed is not None else empty_bf16
            x_row_fp4_save, x_row_sc_save = empty_fp4, empty_sc
            w1_row_fp4_save, w1_row_sc_save = empty_fp4, empty_sc
            w3_row_fp4_save, w3_row_sc_save = empty_fp4, empty_sc
        substage = _mxfp4_stage_begin("ffn_fwd_w2_gemm", debug_name)
        cde_row_rms_partial = None
        if ctx.cde_output:
            if M % 256 or K != 4096 or H != 14336 or N != 4096:
                raise RuntimeError(
                    "MX exact C/D/E W2 producer requires [M,4096,14336] "
                    "with M divisible by 256"
                )
            y, cde_row_rms_partial = mxfp4_gemm_residual_rms(
                h_q.row_fp4, h_q.row_sc, w2_q.row_fp4, w2_q.row_sc, res
            )
        elif ctx.h_output:
            if res is None:
                raise RuntimeError("MX H W2 carrier requires the residual stream")
            out_carrier = mxfp4_h_residual_carrier(
                h_q.row_fp4, h_q.row_sc, w2_q.row_fp4, w2_q.row_sc,
                res, _as_contiguous_bf16(h_next_gamma), 1.0e-5,
            )
            y = out_carrier.z_out
        elif res is not None and use_mxfp4_residual_fusion_ffn():
            y = mxfp4_gemm_residual(h_q.row_fp4, h_q.row_sc, w2_q.row_fp4, w2_q.row_sc, res)
        else:
            y = mxfp4_gemm(h_q.row_fp4, h_q.row_sc, w2_q.row_fp4, w2_q.row_sc)
            if res is not None:
                y = y + res
        _mxfp4_stage_end("ffn_fwd_w2_gemm", debug_name, substage)

        w1_col_fp4_save, w1_col_sc_save = _mxfp4_weight_backward_col(
            w1_q,
            mixed_localcta_dgrad=ctx.mixed_localcta_dgrad,
        )
        w3_col_fp4_save, w3_col_sc_save = _mxfp4_weight_backward_col(
            w3_q,
            mixed_localcta_dgrad=ctx.mixed_localcta_dgrad,
        )
        w2_col_fp4_save, w2_col_sc_save = _mxfp4_weight_backward_col(
            w2_q,
            mixed_localcta_dgrad=ctx.mixed_localcta_dgrad,
        )
        saved_tensors = [
            inp,
            nw,
            inv_rms,
            x_q.col_fp4,
            x_q.col_sc,
            h1_save,
            h3_save,
            h13_save,
            sig_h1_save,
            h_q.col_fp4,
            h_q.col_sc,
            w1_col_fp4_save,
            w1_col_sc_save,
            w3_col_fp4_save,
            w3_col_sc_save,
            w2_col_fp4_save,
            w2_col_sc_save,
            x_row_fp4_save,
            x_row_sc_save,
            w1_row_fp4_save,
            w1_row_sc_save,
            w3_row_fp4_save,
            w3_row_sc_save,
        ]
        if ctx.mixed_localcta_dgrad:
            saved_tensors.extend(
                [
                    w1_q.local_col_sg,
                    w3_q.local_col_sg,
                    w2_q.local_col_sg,
                ]
            )
            ctx._mixed_weight_keepalive = (
                *w1_q.keepalive,
                *w3_q.keepalive,
                *w2_q.keepalive,
            )
        ctx.save_for_backward(*saved_tensors)
        ctx.packed_w13_ffn = h13_save.numel() != 0
        ctx.recompute_w13_ffn = recompute_w13
        ctx.ffn_hidden_dim = H
        ctx._mxfp4_debug_name = debug_name
        _mxfp4_stage_end("ffn_fwd", debug_name, stage_start)
        if ctx.h_output:
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(
                out_carrier.row_fp4, out_carrier.row_sc,
                out_carrier.col_fp4, out_carrier.col_sc,
                out_carrier.r_tile,
            )
            return (
                y, out_carrier.row_fp4, out_carrier.row_sc,
                out_carrier.col_fp4, out_carrier.col_sc, out_carrier.r_tile,
            )
        if ctx.cde_output:
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(cde_row_rms_partial)
            return y, cde_row_rms_partial
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *carrier_grads):
        debug_name = getattr(ctx, "_mxfp4_debug_name", None)
        ffn_w2_sr_key = _mxfp4_grad_producer_key(debug_name, "ffn_w2")
        ffn_deriv_sr_key = _mxfp4_grad_producer_key(debug_name, "ffn_deriv")
        stage_start = _mxfp4_stage_begin("ffn_bwd", debug_name)
        te_fused = _get_te_fused()
        dY = _as_contiguous_bf16(grad_output)
        _prewarm_mxfp4_cudnn_sdpa_autograd_handle(dY.device)

        if not ctx.fast_path:
            inp, nw, inv_rms, normed, h1, h3, h, w1_weight, w3_weight, w2_weight = ctx.saved_tensors
            dh = dY.matmul(w2_weight)
            grad_w2 = dY.transpose(0, 1).matmul(h)
            sig = torch.sigmoid(h1.float())
            silu = (h1.float() * sig).to(torch.bfloat16)
            dh3 = dh * silu
            dsilu = (sig * (1.0 + h1.float() * (1.0 - sig))).to(torch.bfloat16)
            dh1 = dh * h3 * dsilu
            dx_normed = dh1.matmul(w1_weight) + dh3.matmul(w3_weight)
            grad_w1 = dh1.transpose(0, 1).matmul(normed)
            grad_w3 = dh3.transpose(0, 1).matmul(normed)
            _release_delayed_mxfp4_fsdp_backward_prefetch(inp.device)
            grad_input, grad_norm = _mxfp4_rmsnorm_backward(te_fused, dx_normed, inp, nw, inv_rms)
            _mxfp4_stage_end("ffn_bwd", debug_name, stage_start)
            grad_residual = dY if ctx.has_residual else None
            return (
                grad_input, grad_w1, grad_w3, grad_w2, grad_norm,
                None, None, grad_residual, None, None, None, None, None, None,
                None,
            )

        saved_tensors = ctx.saved_tensors
        (
            inp,
            nw,
            inv_rms,
            x_col_fp4,
            x_col_sc,
            h1_raw,
            h3,
            h13_packed,
            sig_h1,
            h_col_fp4,
            h_col_sc,
            w1_col_fp4,
            w1_col_sc,
            w3_col_fp4,
            w3_col_sc,
            w2_col_fp4,
            w2_col_sc,
            x_row_fp4,
            x_row_sc,
            w1_row_fp4,
            w1_row_sc,
            w3_row_fp4,
            w3_row_sc,
        ) = saved_tensors[:23]
        if getattr(ctx, "mixed_localcta_dgrad", False):
            w1_col_sg, w3_col_sg, w2_col_sg = saved_tensors[23:26]

        H_attr = getattr(ctx, "ffn_hidden_dim", None)
        H = int(H_attr) if H_attr is not None else h1_raw.size(1)
        ffn_state = _get_mxfp4_ffn_bwd_state(
            inp.size(0),
            inp.size(1),
            H,
            inp.device,
            force_lazy=getattr(ctx, "mixed_localcta_dgrad", False),
        )
        if getattr(ctx, "recompute_w13_ffn", False):
            substage = _mxfp4_stage_begin("ffn_bwd_recompute_w13", debug_name)
            h1_raw = ffn_state["dh1"]
            h3 = ffn_state["dh3"]
            mxfp4_batched_gemm(
                [x_row_fp4, x_row_fp4],
                [x_row_sc, x_row_sc],
                [w1_row_fp4, w3_row_fp4],
                [w1_row_sc, w3_row_sc],
                [h1_raw, h3],
            )
            h13_packed = _mxfp4_empty_tensor(torch.bfloat16, inp.device)
            _mxfp4_stage_end("ffn_bwd_recompute_w13", debug_name, substage)
        if getattr(ctx, "mixed_localcta_dgrad", False):
            # The mixed route keeps the successful MXFP4+RHT forward and
            # wgrad payloads.  Only dgrad consumes the localCTA row/weight-col
            # slice.  There are exactly two logical gradient producers in an
            # FFN backward: W2 dY and the joint [dh1|dh3] derivative.
            mixed_keepalive = getattr(ctx, "_mixed_weight_keepalive", ())
            empty_fp4 = _mxfp4_empty_tensor(
                torch.float4_e2m1fn_x2, inp.device
            )
            empty_sc = _mxfp4_empty_tensor(torch.uint8, inp.device)
            w1_mixed = _MixedMXLocalCTAWeightCarrier(
                mx_row_fp4=empty_fp4,
                mx_row_sc=empty_sc,
                local_col_fp4=w1_col_fp4,
                local_col_sc=w1_col_sc,
                local_col_sg=w1_col_sg,
                shape=(H, inp.size(1)),
                keepalive=mixed_keepalive,
            )
            w3_mixed = _MixedMXLocalCTAWeightCarrier(
                mx_row_fp4=empty_fp4,
                mx_row_sc=empty_sc,
                local_col_fp4=w3_col_fp4,
                local_col_sc=w3_col_sc,
                local_col_sg=w3_col_sg,
                shape=(H, inp.size(1)),
                keepalive=mixed_keepalive,
            )
            w2_mixed = _MixedMXLocalCTAWeightCarrier(
                mx_row_fp4=empty_fp4,
                mx_row_sc=empty_sc,
                local_col_fp4=w2_col_fp4,
                local_col_sc=w2_col_sc,
                local_col_sg=w2_col_sg,
                shape=(inp.size(1), H),
                keepalive=mixed_keepalive,
            )

            substage = _mxfp4_stage_begin("ffn_bwd_quant_dy", debug_name)
            dY_q = _quantize_mixed_grad_dy_bf16(
                dY,
                producer_key=ffn_w2_sr_key,
            )
            _mxfp4_stage_end("ffn_bwd_quant_dy", debug_name, substage)

            substage = _mxfp4_stage_begin("ffn_bwd_dh_gemm", debug_name)
            dh = _mixed_localcta_dgrad(dY_q, w2_mixed, ffn_state["dh"])
            _mxfp4_stage_end("ffn_bwd_dh_gemm", debug_name, substage)

            use_async_rmsnorm_bwd = (
                use_mxfp4_async_rmsnorm_bwd() and not ctx.h_tile
            )
            use_w2_wgrad_overlap = (
                inp.size(0) >= mxfp4_ffn_wgrad_overlap_min_m()
                and use_mxfp4_ffn_w2_wgrad_overlap()
            )
            use_w13_wgrad_overlap = (
                inp.size(0) >= mxfp4_ffn_wgrad_overlap_min_m()
                and use_mxfp4_ffn_w13_wgrad_overlap()
                and not use_async_rmsnorm_bwd
            )
            wgrad_stream = (
                _get_mxfp4_bwd_side_stream()
                if use_w2_wgrad_overlap or use_w13_wgrad_overlap
                else None
            )
            grad_w2 = ffn_state["grad_w2"]
            substage = _mxfp4_stage_begin("ffn_bwd_w2_wgrad", debug_name)
            if use_w2_wgrad_overlap:
                wgrad_stream.wait_stream(torch.cuda.current_stream())
                _record_stream_tree(dY_q, wgrad_stream)
                _record_stream_tree(h_col_fp4, wgrad_stream)
                _record_stream_tree(h_col_sc, wgrad_stream)
                _record_stream_tree(grad_w2, wgrad_stream)
                with torch.cuda.stream(wgrad_stream):
                    mxfp4_gemm(
                        dY_q.mx_col_fp4,
                        dY_q.mx_col_sc,
                        h_col_fp4,
                        h_col_sc,
                        grad_w2,
                    )
            else:
                mxfp4_gemm(
                    dY_q.mx_col_fp4,
                    dY_q.mx_col_sc,
                    h_col_fp4,
                    h_col_sc,
                    grad_w2,
                )
            _mxfp4_stage_end("ffn_bwd_w2_wgrad", debug_name, substage)

            substage = _mxfp4_stage_begin(
                "ffn_bwd_silu_deriv_quant", debug_name
            )
            dh1 = ffn_state["dh1"]
            dh3 = ffn_state["dh3"]
            if (
                sig_h1.numel() != 0
                and use_mxfp4_saved_sigmoid_ffn()
                and hasattr(
                    te_fused,
                    "fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax",
                )
            ):
                te_fused.fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax(
                    dh,
                    h3,
                    h1_raw,
                    sig_h1,
                    dh1,
                    dh3,
                )
            elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
                te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                    dh, h3, h1_raw, dh1, dh3
                )
            elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out"):
                amax1 = torch.empty(1, dtype=torch.float32, device=dh.device)
                amax2 = torch.empty(1, dtype=torch.float32, device=dh.device)
                te_fused.fused_silu_deriv_dual_mul_bf16_out(
                    dh, h3, h1_raw, dh1, dh3, amax1, amax2
                )
            else:
                dh1, dh3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(
                    dh, h3, h1_raw
                )
            d13_q = _quantize_mixed_split2_grad_bf16(
                dh1,
                dh3,
                producer_key=ffn_deriv_sr_key,
            )
            _mxfp4_stage_end(
                "ffn_bwd_silu_deriv_quant", debug_name, substage
            )

            grad_w_parts = [ffn_state["grad_w1"], ffn_state["grad_w3"]]
            h_sc = H // 128
            d13_col_fp4 = d13_q.mx_col_fp4
            d13_col_sc = d13_q.mx_col_sc

            def launch_mixed_w13_wgrad() -> None:
                _mxfp4_batched_gemm_configured(
                    [
                        d13_col_fp4.narrow(0, 0, H),
                        d13_col_fp4.narrow(0, H, H),
                    ],
                    [
                        d13_col_sc.narrow(0, 0, h_sc),
                        d13_col_sc.narrow(0, h_sc, h_sc),
                    ],
                    [x_col_fp4, x_col_fp4],
                    [x_col_sc, x_col_sc],
                    grad_w_parts,
                    prefix="MXFP4_FFN_W13_WGRAD_BATCHED_GEMM_CONFIG",
                )

            substage = _mxfp4_stage_begin("ffn_bwd_w13_wgrad", debug_name)
            if use_w13_wgrad_overlap:
                wgrad_stream.wait_stream(torch.cuda.current_stream())
                _record_stream_tree(d13_q, wgrad_stream)
                _record_stream_tree(x_col_fp4, wgrad_stream)
                _record_stream_tree(x_col_sc, wgrad_stream)
                _record_stream_tree(grad_w_parts, wgrad_stream)
                with torch.cuda.stream(wgrad_stream):
                    launch_mixed_w13_wgrad()
            else:
                launch_mixed_w13_wgrad()
            _mxfp4_stage_end("ffn_bwd_w13_wgrad", debug_name, substage)

            substage = _mxfp4_stage_begin("ffn_bwd_dgrad_gemm", debug_name)
            dx_normed = _mixed_localcta_split2_dgrad(
                d13_q,
                w1_mixed,
                w3_mixed,
                ffn_state["dx0"],
            )
            _mxfp4_stage_end("ffn_bwd_dgrad_gemm", debug_name, substage)

            substage = _mxfp4_stage_begin("ffn_bwd_rmsnorm", debug_name)
            if ctx.h_tile:
                rms_state = None
                rms_stream = None
            elif use_async_rmsnorm_bwd:
                rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                    dx_normed,
                    inp,
                    nw,
                    inv_rms,
                    te_fused,
                    owner_key=debug_name,
                    tag="mxfp4_ffn_mixed_localcta_dgrad",
                    force_single_out=use_mxfp4_async_rmsnorm_bwd_single_out(),
                    force_norm_weight_copy=True,
                )
            else:
                rms_state = None
                rms_stream = None
            _mxfp4_stage_end("ffn_bwd_rmsnorm", debug_name, substage)

            producer_streams = ()
            if wgrad_stream is not None:
                producer_streams += (wgrad_stream,)
            if rms_stream is not None:
                producer_streams += (rms_stream,)
            _release_delayed_mxfp4_fsdp_backward_prefetch(
                inp.device,
                producer_streams=producer_streams,
            )

            substage = _mxfp4_stage_begin(
                "ffn_bwd_rmsnorm_compute", debug_name
            )
            if ctx.h_tile:
                grad_input, grad_norm = mxfp4_h_tile_backward(
                    dx_normed, inp, nw, inv_rms
                )
            elif use_async_rmsnorm_bwd:
                grad_input = rms_state["grad_input"]
                grad_norm = rms_state["dgamma"]
            else:
                grad_input, grad_norm = _mxfp4_rmsnorm_backward(
                    te_fused, dx_normed, inp, nw, inv_rms
                )
            _mxfp4_stage_end(
                "ffn_bwd_rmsnorm_compute", debug_name, substage
            )

            substage = _mxfp4_stage_begin(
                "ffn_bwd_wait_side_streams", debug_name
            )
            if wgrad_stream is not None:
                torch.cuda.current_stream().wait_stream(wgrad_stream)
            if rms_stream is not None:
                torch.cuda.current_stream().wait_stream(rms_stream)
            _mxfp4_stage_end(
                "ffn_bwd_wait_side_streams", debug_name, substage
            )
            _mxfp4_stage_end("ffn_bwd", debug_name, stage_start)
            grad_w1, grad_w3 = grad_w_parts
            grad_residual = dY if ctx.has_residual else None
            return (
                grad_input, grad_w1, grad_w3, grad_w2, grad_norm,
                None, None, grad_residual, None, None, None, None, None, None,
                None,
            )
        substage = _mxfp4_stage_begin("ffn_bwd_quant_dy", debug_name)
        dY_q = _quantize_row_col_bf16(
            dY, role="grad", producer_key=ffn_w2_sr_key
        )
        _mxfp4_stage_end("ffn_bwd_quant_dy", debug_name, substage)
        dh = ffn_state["dh"]
        native_silu_quant = False
        native_silu_common = (
            use_mxfp4_fused_silu_deriv_split2_ffn()
            and use_mxfp4_split2_ffn_onepass_dgrad()
            and use_mxfp4_split2_ffn_inplace_quant()
            and not _mxfp4_needs_opt_quant("grad")
            and h1_raw.size(0) % 256 == 0
            and h1_raw.size(1) % 256 == 0
        )
        native_silu_from_sigmoid = (
            native_silu_common
            and use_mxfp4_w2_dgrad_saved_sigmoid_producer()
            and sig_h1.numel() != 0
            and use_mxfp4_saved_sigmoid_ffn()
        )
        native_silu_from_h1 = (
            native_silu_common
            and use_mxfp4_w2_dgrad_silu_producer()
            and not native_silu_from_sigmoid
        )
        native_silu_row_bf16 = (
            native_silu_common
            and use_mxfp4_w2_dgrad_saved_sigmoid_row_bf16_producer()
            and sig_h1.numel() != 0
            and use_mxfp4_saved_sigmoid_ffn()
        )
        native_silu_use_onepass = (native_silu_from_sigmoid or native_silu_from_h1) and not native_silu_row_bf16
        native_silu_row_bf16_quant = False
        if native_silu_use_onepass:
            stage_name = (
                "ffn_bwd_dh_silu_sigmoid_quant_gemm"
                if native_silu_from_sigmoid
                else "ffn_bwd_dh_silu_quant_gemm"
            )
            substage = _mxfp4_stage_begin(stage_name, debug_name)
            if native_silu_from_sigmoid:
                native_silu_quant = mxfp4_gemm_silu_dgrad_from_sigmoid_quant(
                    dY_q.row_fp4,
                    dY_q.row_sc,
                    w2_col_fp4,
                    w2_col_sc,
                    h3,
                    h1_raw,
                    sig_h1,
                    ffn_state["split2_row_fp4"],
                    ffn_state["split2_row_sc"],
                    ffn_state["fused_split2_col_fp4"][0],
                    ffn_state["fused_split2_col_sc"][0],
                    ffn_state["fused_split2_col_fp4"][1],
                    ffn_state["fused_split2_col_sc"][1],
                    config_id=mxfp4_w2_dgrad_saved_sigmoid_producer_config_id(),
                    mode=1,
                )
            else:
                native_silu_quant = mxfp4_gemm_silu_dgrad_quant(
                    dY_q.row_fp4,
                    dY_q.row_sc,
                    w2_col_fp4,
                    w2_col_sc,
                    h3,
                    h1_raw,
                    ffn_state["split2_row_fp4"],
                    ffn_state["split2_row_sc"],
                    ffn_state["fused_split2_col_fp4"][0],
                    ffn_state["fused_split2_col_sc"][0],
                    ffn_state["fused_split2_col_fp4"][1],
                    ffn_state["fused_split2_col_sc"][1],
                    config_id=mxfp4_w2_dgrad_silu_producer_config_id(),
                    mode=1,
                )
            _mxfp4_stage_end(stage_name, debug_name, substage)
        elif native_silu_row_bf16:
            substage = _mxfp4_stage_begin("ffn_bwd_dh_silu_sigmoid_row_bf16_quant_gemm", debug_name)
            dh1 = ffn_state["dh1"]
            dh3 = ffn_state["dh3"]
            native_silu_row_bf16_quant = mxfp4_gemm_silu_dgrad_from_sigmoid_row_bf16_quant(
                dY_q.row_fp4,
                dY_q.row_sc,
                w2_col_fp4,
                w2_col_sc,
                h3,
                h1_raw,
                sig_h1,
                dh1,
                dh3,
                ffn_state["split2_row_fp4"],
                ffn_state["split2_row_sc"],
                config_id=mxfp4_w2_dgrad_saved_sigmoid_row_bf16_producer_config_id(),
                mode=1,
            )
            _mxfp4_stage_end("ffn_bwd_dh_silu_sigmoid_row_bf16_quant_gemm", debug_name, substage)
        if not native_silu_quant and not native_silu_row_bf16_quant:
            substage = _mxfp4_stage_begin("ffn_bwd_dh_gemm", debug_name)
            _mxfp4_gemm_ffn_dh(dY_q.row_fp4, dY_q.row_sc, w2_col_fp4, w2_col_sc, dh)
            _mxfp4_stage_end("ffn_bwd_dh_gemm", debug_name, substage)
        split2_row_ready_event = None
        use_w2_wgrad_overlap = (
            inp.size(0) >= mxfp4_ffn_wgrad_overlap_min_m()
            and use_mxfp4_ffn_w2_wgrad_overlap()
        )
        wgrad_stream = None
        grad_w2 = ffn_state["grad_w2"]
        if use_w2_wgrad_overlap:
            substage = _mxfp4_stage_begin("ffn_bwd_w2_wgrad", debug_name)
            wgrad_stream = _get_mxfp4_bwd_side_stream()
            wgrad_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(dY_q.col_fp4, wgrad_stream)
            _record_stream_tree(dY_q.col_sc, wgrad_stream)
            _record_stream_tree(h_col_fp4, wgrad_stream)
            _record_stream_tree(h_col_sc, wgrad_stream)
            _record_stream_tree(grad_w2, wgrad_stream)
            with torch.cuda.stream(wgrad_stream):
                mxfp4_gemm(dY_q.col_fp4, dY_q.col_sc, h_col_fp4, h_col_sc, grad_w2)
            _mxfp4_stage_end("ffn_bwd_w2_wgrad", debug_name, substage)
        else:
            substage = _mxfp4_stage_begin("ffn_bwd_w2_wgrad", debug_name)
            mxfp4_gemm(dY_q.col_fp4, dY_q.col_sc, h_col_fp4, h_col_sc, grad_w2)
            _mxfp4_stage_end("ffn_bwd_w2_wgrad", debug_name, substage)
        use_onepass_dgrad = False
        use_split2_row_overlap = False
        substage = _mxfp4_stage_begin("ffn_bwd_silu_deriv_quant", debug_name)
        if native_silu_row_bf16_quant:
            H = h1_raw.size(1)
            H_packed = H // 2
            H_sc = H // 128
            use_onepass_dgrad = True
            use_split2_row_overlap = True
            row_fp4 = ffn_state["split2_row_fp4"]
            row_sc = ffn_state["split2_row_sc"]
            col_fp4 = ffn_state["split2_col_fp4"]
            col_sc = ffn_state["split2_col_sc"]
            split2_row_ready_event = torch.cuda.Event()
            split2_row_ready_event.record(torch.cuda.current_stream())
            dh1_q = None
            dh3_q = None
        elif native_silu_quant:
            H = h1_raw.size(1)
            H_packed = H // 2
            H_sc = H // 128
            use_onepass_dgrad = True
            row_fp4 = ffn_state["split2_row_fp4"]
            row_sc = ffn_state["split2_row_sc"]
            empty_row = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device)
            empty_sc = _mxfp4_empty_tensor(row_sc.dtype, dh.device)
            dh1_q = _MXFP4RowCol(
                row_fp4=empty_row,
                row_sc=empty_sc,
                col_fp4=ffn_state["fused_split2_col_fp4"][0],
                col_sc=ffn_state["fused_split2_col_sc"][0],
            )
            dh3_q = _MXFP4RowCol(
                row_fp4=empty_row,
                row_sc=empty_sc,
                col_fp4=ffn_state["fused_split2_col_fp4"][1],
                col_sc=ffn_state["fused_split2_col_sc"][1],
            )
        elif (
            sig_h1.numel() != 0
            and use_mxfp4_saved_sigmoid_ffn()
            and use_mxfp4_split2_ffn_onepass_dgrad()
            and use_mxfp4_split2_ffn_inplace_quant()
            and not _mxfp4_needs_opt_quant("grad")
            and hasattr(te_fused, "fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax")
        ):
            H = dh.size(1)
            H_sc = H // 128
            H_packed = H // 2
            use_onepass_dgrad = True
            dh1 = ffn_state["dh1"]
            dh3 = ffn_state["dh3"]
            row_fp4 = ffn_state["split2_row_fp4"]
            row_sc = ffn_state["split2_row_sc"]
            col_fp4 = ffn_state["split2_col_fp4"]
            col_sc = ffn_state["split2_col_sc"]
            fused_saved_split2 = False
            row_produced_saved_split2 = False
            if use_mxfp4_saved_sigmoid_row_producer_split2_ffn():
                try:
                    mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_bf16_launch_inplace(
                        dh,
                        h3,
                        h1_raw,
                        sig_h1,
                        dh1,
                        dh3,
                        row_fp4,
                        row_sc,
                    )
                    row_produced_saved_split2 = True
                except AttributeError:
                    row_produced_saved_split2 = False
            if (
                not row_produced_saved_split2
                and (
                use_mxfp4_saved_sigmoid_fused_split2_ffn()
                and not use_mxfp4_saved_sigmoid_split2_row_overlap()
                )
            ):
                try:
                    mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_and_col_splitcols_launch_inplace(
                        dh,
                        h3,
                        h1_raw,
                        sig_h1,
                        row_fp4,
                        row_sc,
                        ffn_state["fused_split2_col_fp4"][0],
                        ffn_state["fused_split2_col_sc"][0],
                        ffn_state["fused_split2_col_fp4"][1],
                        ffn_state["fused_split2_col_sc"][1],
                    )
                    fused_saved_split2 = True
                except AttributeError:
                    fused_saved_split2 = False
            if row_produced_saved_split2:
                use_split2_row_overlap = True
                split2_row_ready_event = torch.cuda.Event()
                split2_row_ready_event.record(torch.cuda.current_stream())
                dh1_q = None
                dh3_q = None
            elif fused_saved_split2:
                empty_row = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device)
                empty_sc = _mxfp4_empty_tensor(row_sc.dtype, dh.device)
                dh1_q = _MXFP4RowCol(
                    row_fp4=empty_row,
                    row_sc=empty_sc,
                    col_fp4=ffn_state["fused_split2_col_fp4"][0],
                    col_sc=ffn_state["fused_split2_col_sc"][0],
                )
                dh3_q = _MXFP4RowCol(
                    row_fp4=empty_row,
                    row_sc=empty_sc,
                    col_fp4=ffn_state["fused_split2_col_fp4"][1],
                    col_sc=ffn_state["fused_split2_col_sc"][1],
                )
            else:
                te_fused.fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax(
                    dh,
                    h3,
                    h1_raw,
                    sig_h1,
                    dh1,
                    dh3,
                )
                if (
                    use_mxfp4_saved_sigmoid_split2_row_overlap()
                    and use_mxfp4_split2_ffn_row_overlap()
                    and use_mxfp4_ffn_w13_wgrad_overlap()
                    and inp.size(0) >= mxfp4_ffn_wgrad_overlap_min_m()
                ):
                    mxfp4_quantize_split2_row_only_launch_inplace(dh1, dh3, row_fp4, row_sc)
                    use_split2_row_overlap = True
                    split2_row_ready_event = torch.cuda.Event()
                    split2_row_ready_event.record(torch.cuda.current_stream())
                    dh1_q = None
                    dh3_q = None
                else:
                    _quantize_split2_row_and_col_inplace(
                        dh1,
                        dh3,
                        row_fp4,
                        row_sc,
                        col_fp4,
                        col_sc,
                        role="grad",
                    )
                    col_fp4_u8 = col_fp4.view(torch.uint8)
                    empty_row = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device)
                    empty_sc = _mxfp4_empty_tensor(row_sc.dtype, dh.device)
                    dh1_q = _MXFP4RowCol(
                        row_fp4=empty_row,
                        row_sc=empty_sc,
                        col_fp4=col_fp4_u8[:H].contiguous().view(torch.float4_e2m1fn_x2),
                        col_sc=col_sc[:H_sc].contiguous(),
                    )
                    dh3_q = _MXFP4RowCol(
                        row_fp4=empty_row,
                        row_sc=empty_sc,
                        col_fp4=col_fp4_u8[H:].contiguous().view(torch.float4_e2m1fn_x2),
                        col_sc=col_sc[H_sc:].contiguous(),
                    )
        elif use_mxfp4_fused_silu_deriv_split2_ffn() and not _mxfp4_needs_opt_quant("grad"):
            try:
                use_packed_split2 = (
                    bool(getattr(ctx, "packed_w13_ffn", False))
                    and h13_packed.numel() != 0
                    and use_mxfp4_split2_ffn_onepass_dgrad()
                )
                use_onepass_dgrad = use_mxfp4_split2_ffn_onepass_dgrad()
                if use_packed_split2:
                    H = dh.size(1)
                    H_packed = H // 2
                    H_sc = H // 128
                    row_fp4, row_sc, col_fp4, col_sc = (
                        mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined(
                            dh,
                            h13_packed,
                            H,
                            H,
                        )
                    )
                    empty_row = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device)
                    empty_sc = _mxfp4_empty_tensor(row_sc.dtype, dh.device)
                    col_fp4_u8 = col_fp4.view(torch.uint8)
                    dh1_q = _MXFP4RowCol(
                        row_fp4=empty_row,
                        row_sc=empty_sc,
                        col_fp4=col_fp4_u8[:H].contiguous().view(torch.float4_e2m1fn_x2),
                        col_sc=col_sc[:H_sc].contiguous(),
                    )
                    dh3_q = _MXFP4RowCol(
                        row_fp4=empty_row,
                        row_sc=empty_sc,
                        col_fp4=col_fp4_u8[H:].contiguous().view(torch.float4_e2m1fn_x2),
                        col_sc=col_sc[H_sc:].contiguous(),
                    )
                elif use_onepass_dgrad:
                    H = dh.size(1)
                    H_packed = H // 2
                    H_sc = H // 128
                    if not use_mxfp4_split2_ffn_inplace_quant():
                        raise AttributeError
                    mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace(
                        dh,
                        h3,
                        h1_raw,
                        ffn_state["split2_row_fp4"],
                        ffn_state["split2_row_sc"],
                        ffn_state["fused_split2_col_fp4"][0],
                        ffn_state["fused_split2_col_sc"][0],
                        ffn_state["fused_split2_col_fp4"][1],
                        ffn_state["fused_split2_col_sc"][1],
                    )
                    row_fp4 = ffn_state["split2_row_fp4"]
                    row_sc = ffn_state["split2_row_sc"]
                    empty_row = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device)
                    empty_sc = _mxfp4_empty_tensor(row_sc.dtype, dh.device)
                    dh1_q = _MXFP4RowCol(
                        row_fp4=empty_row,
                        row_sc=empty_sc,
                        col_fp4=ffn_state["fused_split2_col_fp4"][0],
                        col_sc=ffn_state["fused_split2_col_sc"][0],
                    )
                    dh3_q = _MXFP4RowCol(
                        row_fp4=empty_row,
                        row_sc=empty_sc,
                        col_fp4=ffn_state["fused_split2_col_fp4"][1],
                        col_sc=ffn_state["fused_split2_col_sc"][1],
                    )
                else:
                    if not use_mxfp4_split2_ffn_inplace_quant():
                        raise AttributeError
                    mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace(
                        dh,
                        h3,
                        h1_raw,
                        ffn_state["fused_split2_row_fp4"],
                        ffn_state["fused_split2_row_sc"],
                        ffn_state["fused_split2_col_fp4"],
                        ffn_state["fused_split2_col_sc"],
                    )
                    row_fp4 = ffn_state["fused_split2_row_fp4"]
                    row_sc = ffn_state["fused_split2_row_sc"]
                    col_fp4 = ffn_state["fused_split2_col_fp4"]
                    col_sc = ffn_state["fused_split2_col_sc"]
            except AttributeError:
                if use_onepass_dgrad:
                    row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc = (
                        mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols(dh, h3, h1_raw)
                    )
                    H = dh.size(1)
                    H_packed = H // 2
                    H_sc = H // 128
                    dh1_q = _MXFP4RowCol(
                        row_fp4=_mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device),
                        row_sc=_mxfp4_empty_tensor(row_sc.dtype, dh.device),
                        col_fp4=col0_fp4,
                        col_sc=col0_sc,
                    )
                    dh3_q = _MXFP4RowCol(
                        row_fp4=_mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device),
                        row_sc=_mxfp4_empty_tensor(row_sc.dtype, dh.device),
                        col_fp4=col1_fp4,
                        col_sc=col1_sc,
                    )
                else:
                    row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_deriv_quantize_split2_row_and_col(
                        dh, h3, h1_raw
                    )
                    dh1_q = _MXFP4RowCol(
                        row_fp4=row_fp4[0],
                        row_sc=row_sc[0],
                        col_fp4=col_fp4[0],
                        col_sc=col_sc[0],
                    )
                    dh3_q = _MXFP4RowCol(
                        row_fp4=row_fp4[1],
                        row_sc=row_sc[1],
                        col_fp4=col_fp4[1],
                        col_sc=col_sc[1],
                    )
            if not use_onepass_dgrad:
                dh1_q = _MXFP4RowCol(
                    row_fp4=row_fp4[0],
                    row_sc=row_sc[0],
                    col_fp4=col_fp4[0],
                    col_sc=col_sc[0],
                )
                dh3_q = _MXFP4RowCol(
                    row_fp4=row_fp4[1],
                    row_sc=row_sc[1],
                    col_fp4=col_fp4[1],
                    col_sc=col_sc[1],
                )
        else:
            dh1 = ffn_state["dh1"]
            dh3 = ffn_state["dh3"]
            split2_fastpath_supported = (
                dh.size(0) % 256 == 0
                and inp.size(1) % 256 == 0
                and dh.size(1) % 256 == 0
                and _mxfp4_split2_grad_random_sign_safe()
            )
            if use_mxfp4_split2_ffn_quant() and split2_fastpath_supported:
                H = dh.size(1)
                H_packed = H // 2
                H_sc = H // 128
                use_onepass_dgrad = use_mxfp4_split2_ffn_onepass_dgrad()
                grad_split2_overlap_opt = (
                    _mxfp4_bool_env("MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT", True)
                    and _mxfp4_rht_for_role("grad")
                    and not _mxfp4_data_sr_for_role("grad")
                    and not _mxfp4_scale_sr_for_role("grad")
                ) or (
                    _mxfp4_bool_env("MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_GRAD_SR", False)
                    and _mxfp4_data_sr_for_role("grad")
                    and not _mxfp4_scale_sr_for_role("grad")
                    and not _mxfp4_rht_for_role("grad")
                )
                use_fused_row_producer = (
                    use_onepass_dgrad
                    and use_mxfp4_ffn_w13_wgrad_overlap()
                    and use_mxfp4_fused_row_producer_split2_ffn()
                    and not _mxfp4_needs_opt_quant("grad")
                )
                use_split2_row_overlap = (
                    use_onepass_dgrad
                    and use_mxfp4_ffn_w13_wgrad_overlap()
                    and use_mxfp4_split2_ffn_row_overlap()
                    and (not _mxfp4_needs_opt_quant("grad") or grad_split2_overlap_opt)
                )
                row_fp4 = ffn_state["split2_row_fp4"]
                row_sc = ffn_state["split2_row_sc"]
                col_fp4 = ffn_state["split2_col_fp4"]
                col_sc = ffn_state["split2_col_sc"]
                produced_split2_quant = False
                if (
                    use_mxfp4_fused_silu_deriv_split2_ffn()
                    and os.environ.get("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT", "0") == "1"
                    and _mxfp4_needs_opt_quant("grad")
                    and not _mxfp4_data_sr_for_role("grad")
                    and not _mxfp4_scale_sr_for_role("grad")
                    and use_mxfp4_split2_ffn_inplace_quant()
                ):
                    try:
                        mxfp4_fused_silu_deriv_quantize_split2_row_and_col_opt_launch_inplace(
                            dh,
                            h3,
                            h1_raw,
                            row_fp4,
                            row_sc,
                            col_fp4,
                            col_sc,
                            1,
                            **_mxfp4_split_opt_kwargs("grad"),
                        )
                        produced_split2_quant = True
                        use_split2_row_overlap = False
                    except AttributeError:
                        produced_split2_quant = False
                if not produced_split2_quant and use_fused_row_producer:
                    try:
                        row_producer = (
                            mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace
                            if use_mxfp4_fused_row_producer_tile_split2_ffn()
                            else mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace
                        )
                        row_producer(
                            dh,
                            h3,
                            h1_raw,
                            dh1,
                            dh3,
                            row_fp4,
                            row_sc,
                        )
                        use_split2_row_overlap = True
                        split2_row_ready_event = torch.cuda.Event()
                        split2_row_ready_event.record(torch.cuda.current_stream())
                    except AttributeError:
                        use_fused_row_producer = False
                if not produced_split2_quant and not use_fused_row_producer:
                    if hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
                        te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(dh, h3, h1_raw, dh1, dh3)
                    elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out"):
                        amax1 = torch.empty(1, dtype=torch.float32, device=dh.device)
                        amax2 = torch.empty(1, dtype=torch.float32, device=dh.device)
                        te_fused.fused_silu_deriv_dual_mul_bf16_out(dh, h3, h1_raw, dh1, dh3, amax1, amax2)
                    else:
                        dh1, dh3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1_raw)
                    if use_split2_row_overlap:
                        try:
                            if not use_mxfp4_split2_ffn_inplace_quant():
                                raise AttributeError
                            if _mxfp4_needs_opt_quant("grad"):
                                mxfp4_quantize_split2_row_only_opt_launch_inplace(
                                    dh1,
                                    dh3,
                                    row_fp4,
                                    row_sc,
                                    1,
                                    **_mxfp4_split_row_only_opt_kwargs(
                                        "grad", ffn_deriv_sr_key
                                    ),
                                )
                            else:
                                mxfp4_quantize_split2_row_only_launch_inplace(
                                    dh1,
                                    dh3,
                                    row_fp4,
                                    row_sc,
                                )
                            split2_row_ready_event = torch.cuda.Event()
                            split2_row_ready_event.record(torch.cuda.current_stream())
                        except AttributeError:
                            use_split2_row_overlap = False
                    try:
                        if not use_split2_row_overlap:
                            if not use_mxfp4_split2_ffn_inplace_quant():
                                raise AttributeError
                            _quantize_split2_row_and_col_inplace(
                                dh1,
                                dh3,
                                row_fp4,
                                row_sc,
                                col_fp4,
                                col_sc,
                                role="grad",
                                producer_key=ffn_deriv_sr_key,
                            )
                    except AttributeError:
                        if _mxfp4_needs_opt_quant("grad"):
                            raise
                        use_split2_row_overlap = False
                        row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_split2_row_and_col(dh1, dh3)
                if use_split2_row_overlap:
                    dh1_q = None
                    dh3_q = None
                else:
                    col_fp4_u8 = col_fp4.view(torch.uint8)
                    if use_onepass_dgrad:
                        empty_row = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, dh.device)
                        empty_sc = _mxfp4_empty_tensor(row_sc.dtype, dh.device)
                        dh1_q = _MXFP4RowCol(
                            row_fp4=empty_row,
                            row_sc=empty_sc,
                            col_fp4=col_fp4_u8[:H].contiguous().view(torch.float4_e2m1fn_x2),
                            col_sc=col_sc[:H_sc].contiguous(),
                        )
                        dh3_q = _MXFP4RowCol(
                            row_fp4=empty_row,
                            row_sc=empty_sc,
                            col_fp4=col_fp4_u8[H:].contiguous().view(torch.float4_e2m1fn_x2),
                            col_sc=col_sc[H_sc:].contiguous(),
                        )
                    else:
                        row_fp4_u8 = row_fp4.view(torch.uint8)
                        dh1_q = _MXFP4RowCol(
                            row_fp4=row_fp4_u8[:, :H_packed].contiguous().view(torch.float4_e2m1fn_x2),
                            row_sc=row_sc[:, :H_sc].contiguous(),
                            col_fp4=col_fp4_u8[:H].contiguous().view(torch.float4_e2m1fn_x2),
                            col_sc=col_sc[:H_sc].contiguous(),
                        )
                        dh3_q = _MXFP4RowCol(
                            row_fp4=row_fp4_u8[:, H_packed:].contiguous().view(torch.float4_e2m1fn_x2),
                            row_sc=row_sc[:, H_sc:].contiguous(),
                            col_fp4=col_fp4_u8[H:].contiguous().view(torch.float4_e2m1fn_x2),
                            col_sc=col_sc[H_sc:].contiguous(),
                        )
            else:
                if hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
                    te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(dh, h3, h1_raw, dh1, dh3)
                elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out"):
                    amax1 = torch.empty(1, dtype=torch.float32, device=dh.device)
                    amax2 = torch.empty(1, dtype=torch.float32, device=dh.device)
                    te_fused.fused_silu_deriv_dual_mul_bf16_out(dh, h3, h1_raw, dh1, dh3, amax1, amax2)
                else:
                    dh1, dh3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1_raw)
                dh1_q = _quantize_row_col_bf16(
                    dh1, role="grad", producer_key=ffn_deriv_sr_key
                )
                dh3_q = _quantize_row_col_bf16(
                    dh3, role="grad", producer_key=ffn_deriv_sr_key
                )

        _mxfp4_stage_end("ffn_bwd_silu_deriv_quant", debug_name, substage)
        grad_w_parts = [ffn_state["grad_w1"], ffn_state["grad_w3"]]

        def launch_w13_wgrad_overlap() -> None:
            dh_quant_ready = torch.cuda.Event()
            dh_quant_ready.record(torch.cuda.current_stream())
            wgrad_stream.wait_event(dh_quant_ready)
            _record_stream_tree(dh1_q.col_fp4, wgrad_stream)
            _record_stream_tree(dh1_q.col_sc, wgrad_stream)
            _record_stream_tree(dh3_q.col_fp4, wgrad_stream)
            _record_stream_tree(dh3_q.col_sc, wgrad_stream)
            _record_stream_tree(x_col_fp4, wgrad_stream)
            _record_stream_tree(x_col_sc, wgrad_stream)
            _record_stream_tree(grad_w_parts, wgrad_stream)
            with torch.cuda.stream(wgrad_stream):
                _mxfp4_batched_gemm_configured(
                    [dh1_q.col_fp4, dh3_q.col_fp4],
                    [dh1_q.col_sc, dh3_q.col_sc],
                    [x_col_fp4, x_col_fp4],
                    [x_col_sc, x_col_sc],
                    grad_w_parts,
                    prefix="MXFP4_FFN_W13_WGRAD_BATCHED_GEMM_CONFIG",
                )

        def launch_split2_col_wgrad_overlap(wait_event: torch.cuda.Event) -> None:
            wgrad_stream.wait_event(wait_event)
            _record_stream_tree(dh1, wgrad_stream)
            _record_stream_tree(dh3, wgrad_stream)
            _record_stream_tree(col_fp4, wgrad_stream)
            _record_stream_tree(col_sc, wgrad_stream)
            _record_stream_tree(x_col_fp4, wgrad_stream)
            _record_stream_tree(x_col_sc, wgrad_stream)
            _record_stream_tree(grad_w_parts, wgrad_stream)
            with torch.cuda.stream(wgrad_stream):
                if _mxfp4_oriented_grad_data_sr("grad") == "row":
                    # Row carries dgrad and is the only stochastic view in the
                    # proven policy.  The overlapped column producer feeds
                    # wgrad and may receive deterministic paired RHT, but it
                    # must never consume row-SR state.
                    if _mxfp4_rht_has_col("grad"):
                        mxfp4_quantize_split2_col_only_opt_launch_inplace(
                            dh1,
                            dh3,
                            col_fp4,
                            col_sc,
                            1,
                            **_mxfp4_split_axis_only_no_sr_rht_kwargs(
                                "grad", "col"
                            ),
                        )
                    else:
                        mxfp4_quantize_split2_col_only_launch_inplace(
                            dh1,
                            dh3,
                            col_fp4,
                            col_sc,
                            1,
                        )
                elif _mxfp4_needs_opt_quant("grad"):
                    mxfp4_quantize_split2_col_only_opt_launch_inplace(
                        dh1,
                        dh3,
                        col_fp4,
                        col_sc,
                        1,
                        **_mxfp4_split_col_only_opt_kwargs(
                            "grad", ffn_deriv_sr_key
                        ),
                    )
                else:
                    mxfp4_quantize_split2_col_only_launch_inplace(
                        dh1,
                        dh3,
                        col_fp4,
                        col_sc,
                    )
                mxfp4_batched_gemm(
                    [col_fp4.narrow(0, 0, H), col_fp4.narrow(0, H, H)],
                    [col_sc.narrow(0, 0, H_sc), col_sc.narrow(0, H_sc, H_sc)],
                    [x_col_fp4, x_col_fp4],
                    [x_col_sc, x_col_sc],
                    grad_w_parts,
                )

        launched_w13_wgrad = False
        if (
            use_split2_row_overlap
            and use_mxfp4_split2_ffn_producer_split()
            and split2_row_ready_event is not None
        ):
            if wgrad_stream is None:
                wgrad_stream = _get_mxfp4_bwd_side_stream()
            launch_split2_col_wgrad_overlap(split2_row_ready_event)
            launched_w13_wgrad = True
        elif use_onepass_dgrad and use_mxfp4_ffn_w13_wgrad_overlap() and not use_split2_row_overlap:
            if wgrad_stream is None:
                wgrad_stream = _get_mxfp4_bwd_side_stream()
            launch_w13_wgrad_overlap()
            launched_w13_wgrad = True

        h_fused_grad_norm = None
        if use_onepass_dgrad:
            substage = _mxfp4_stage_begin("ffn_bwd_dgrad_gemm", debug_name)
            dgrad_args = (
                row_fp4,
                [row_sc.narrow(1, 0, H_sc), row_sc.narrow(1, H_sc, H_sc)],
                [0, H_packed],
                [H_packed, H_packed],
                [w1_col_fp4, w3_col_fp4],
                [w1_col_sc, w3_col_sc],
            )
            if ctx.h_tile:
                _, h_fused_grad_norm = mxfp4_split2_dgrad_strided_onepass_h_gemm(
                    *dgrad_args,
                    inp,
                    nw,
                    inv_rms,
                    ffn_state["dx0"],
                    config_idx=mxfp4_split2_ffn_onepass_config_idx(),
                )
            else:
                mxfp4_split2_dgrad_strided_onepass_gemm(
                    *dgrad_args,
                    ffn_state["dx0"],
                    config_idx=mxfp4_split2_ffn_onepass_config_idx(),
                )
            dx_normed = ffn_state["dx0"]
            _mxfp4_stage_end("ffn_bwd_dgrad_gemm", debug_name, substage)
        else:
            substage = _mxfp4_stage_begin("ffn_bwd_dgrad_gemm", debug_name)
            dx_parts = [ffn_state["dx0"], ffn_state["dx1"]]
            mxfp4_batched_gemm(
                [dh1_q.row_fp4, dh3_q.row_fp4],
                [dh1_q.row_sc, dh3_q.row_sc],
                [w1_col_fp4, w3_col_fp4],
                [w1_col_sc, w3_col_sc],
                dx_parts,
            )
            dx_parts[0].add_(dx_parts[1])
            dx_normed = dx_parts[0]
            _mxfp4_stage_end("ffn_bwd_dgrad_gemm", debug_name, substage)

        substage = _mxfp4_stage_begin("ffn_bwd_rmsnorm", debug_name)
        use_async_rmsnorm_bwd = use_mxfp4_async_rmsnorm_bwd() and not ctx.h_tile
        use_w13_wgrad_overlap = (
            inp.size(0) >= mxfp4_ffn_wgrad_overlap_min_m()
            and use_mxfp4_ffn_w13_wgrad_overlap()
            and not use_async_rmsnorm_bwd
        )
        if use_w13_wgrad_overlap and wgrad_stream is None:
            wgrad_stream = _get_mxfp4_bwd_side_stream()
        if ctx.h_tile:
            rms_state = None
            rms_stream = None
        elif use_async_rmsnorm_bwd:
            rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                dx_normed,
                inp,
                nw,
                inv_rms,
                te_fused,
                owner_key=debug_name,
                tag="mxfp4_ffn",
                force_single_out=use_mxfp4_async_rmsnorm_bwd_single_out(),
                force_norm_weight_copy=True,
            )
        else:
            rms_state = None
            rms_stream = None
        _mxfp4_stage_end("ffn_bwd_rmsnorm", debug_name, substage)

        substage = _mxfp4_stage_begin("ffn_bwd_w13_wgrad", debug_name)
        if use_split2_row_overlap and not launched_w13_wgrad:
            if wgrad_stream is None:
                wgrad_stream = _get_mxfp4_bwd_side_stream()
            dh_quant_ready = torch.cuda.Event()
            dh_quant_ready.record(torch.cuda.current_stream())
            launch_split2_col_wgrad_overlap(dh_quant_ready)
        elif use_w13_wgrad_overlap:
            if not launched_w13_wgrad:
                launch_w13_wgrad_overlap()
        elif not launched_w13_wgrad:
            _mxfp4_batched_gemm_configured(
                [dh1_q.col_fp4, dh3_q.col_fp4],
                [dh1_q.col_sc, dh3_q.col_sc],
                [x_col_fp4, x_col_fp4],
                [x_col_sc, x_col_sc],
                grad_w_parts,
                prefix="MXFP4_FFN_W13_WGRAD_BATCHED_GEMM_CONFIG",
            )
        _mxfp4_stage_end("ffn_bwd_w13_wgrad", debug_name, substage)
        prefetch_producer_streams = ()
        if wgrad_stream is not None and (
            launched_w13_wgrad
            or use_w13_wgrad_overlap
            or use_split2_row_overlap
            or use_w2_wgrad_overlap
        ):
            prefetch_producer_streams = (wgrad_stream,)
        if rms_stream is not None:
            prefetch_producer_streams += (rms_stream,)
        _release_delayed_mxfp4_fsdp_backward_prefetch(
            inp.device,
            producer_streams=prefetch_producer_streams,
        )
        grad_w1, grad_w3 = grad_w_parts
        substage = _mxfp4_stage_begin("ffn_bwd_rmsnorm_compute", debug_name)
        if h_fused_grad_norm is not None:
            grad_input = dx_normed
            grad_norm = h_fused_grad_norm
        elif ctx.h_tile:
            grad_input, grad_norm = mxfp4_h_tile_backward(
                dx_normed, inp, nw, inv_rms
            )
        elif use_async_rmsnorm_bwd:
            grad_input = rms_state["grad_input"]
            grad_norm = rms_state["dgamma"]
        else:
            grad_input, grad_norm = _mxfp4_rmsnorm_backward(te_fused, dx_normed, inp, nw, inv_rms)
        _mxfp4_stage_end("ffn_bwd_rmsnorm_compute", debug_name, substage)
        substage = _mxfp4_stage_begin("ffn_bwd_wait_side_streams", debug_name)
        if use_w13_wgrad_overlap or use_split2_row_overlap or use_w2_wgrad_overlap:
            torch.cuda.current_stream().wait_stream(wgrad_stream)
        if rms_stream is not None:
            torch.cuda.current_stream().wait_stream(rms_stream)
        _mxfp4_stage_end("ffn_bwd_wait_side_streams", debug_name, substage)
        _mxfp4_stage_end("ffn_bwd", debug_name, stage_start)
        grad_residual = dY if ctx.has_residual else None
        return (
            grad_input, grad_w1, grad_w3, grad_w2, grad_norm,
            None, None, grad_residual, None, None, None, None, None, None,
            None,
        )


class FusedAttentionMXFP4_TK(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, norm_eps: float = 1e-5,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.epsilon = norm_eps

        self.q_dim = n_heads * head_dim
        self.k_dim = n_kv_heads * head_dim
        self.v_dim = n_kv_heads * head_dim
        self.total_out = self.q_dim + self.k_dim + self.v_dim

        self.norm_weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.w_qkv = nn.Parameter(torch.empty(self.total_out, dim, device=device, dtype=dtype))
        self.wo_weight = nn.Parameter(torch.empty(dim, self.q_dim, device=device, dtype=dtype))
        self._workspace = None
        self._workspace_device = None
        self.init_weights()

    def _forward_wo_bf16(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_nhsd = attn_output.dim() == 4
        is_3d = attn_output.dim() == 3
        if is_nhsd:
            B, H, S, D = attn_output.shape
            out_2d = attn_output.transpose(1, 2).contiguous().view(B * S, H * D)
        elif is_3d:
            B, S, D = attn_output.shape
            out_2d = attn_output.reshape(B * S, D)
        else:
            out_2d = attn_output

        y = F.linear(out_2d.to(self.wo_weight.dtype), self.wo_weight)
        if residual is not None:
            y = y + residual.reshape(y.shape).to(y.dtype)
        if is_nhsd:
            return y.view(B, S, self.dim)
        if is_3d:
            return y.view(B, S, self.dim)
        return y

    def _ensure_workspace(self, device):
        if self._workspace_device != device:
            self._workspace = torch.empty(1, dtype=torch.uint8, device=device)
            self._workspace_device = device

    def forward_qkv(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        h_carrier=None,
        cde_row_rms_partial: torch.Tensor | None = None,
    ):
        if h_carrier is not None and cde_row_rms_partial is not None:
            raise RuntimeError("exact C/D/E and MX H QKV carriers are mutually exclusive")
        if h_carrier is not None:
            x, h_row_fp4, h_row_sc, h_col_fp4, h_col_sc, h_r_tile = h_carrier
        else:
            h_row_fp4 = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, x.device)
            h_row_sc = _mxfp4_empty_tensor(torch.uint8, x.device)
            h_col_fp4 = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, x.device)
            h_col_sc = _mxfp4_empty_tensor(torch.uint8, x.device)
            h_r_tile = _mxfp4_empty_tensor(torch.float32, x.device)
        if cde_row_rms_partial is None:
            cde_row_rms_partial = _mxfp4_empty_tensor(torch.float32, x.device)
        debug_name = f"{getattr(self, '_lbt_debug_name', self.__class__.__name__)}:qkv"
        is_3d = x.dim() == 3
        rope_batch_size = 0
        rope_seq_len = 0
        rope_freqs = None
        rope_enabled = False
        rope_route = "separate"
        if is_3d:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
            effective_rope_route = None
            if (
                freqs_cis is not None
                and use_mxfp4_qkv_rope_epilogue()
                and freqs_cis.is_cuda
                and x.is_cuda
            ):
                effective_rope_route = _mxfp4_qkv_rope_route(
                    B * S,
                    D,
                    self.q_dim,
                    self.k_dim,
                    self.v_dim,
                    self.head_dim,
                    S,
                )
            if effective_rope_route is not None:
                rope_enabled = True
                rope_route = effective_rope_route
                rope_batch_size = B
                rope_seq_len = S
                rope_freqs = freqs_cis
        self._ensure_workspace(x.device)
        if rope_enabled:
            _log_mxfp4_fsdp_overlap_before_rope_qkv(x.device)
            _order_mxfp4_rope_qkv_after_fsdp_reduce_scatter(x.device)
        xq, xk, xv = _FusedQKVFunction_MXFP4_TK.apply(
            x, self.w_qkv, self.norm_weight, self.epsilon,
            self.q_dim, self.k_dim, self.v_dim,
            rope_freqs, rope_batch_size, rope_seq_len, self.head_dim,
            debug_name,
            h_row_fp4, h_row_sc, h_col_fp4, h_col_sc, h_r_tile,
            cde_row_rms_partial,
        )
        _release_delayed_mxfp4_fsdp_forward_prefetch(x.device)
        self._last_qkv_rope_applied = rope_enabled
        self._last_qkv_rope_route = rope_route
        if is_3d:
            xq = xq.view(B, S, -1)
            xk = xk.view(B, S, -1)
            xv = xv.view(B, S, -1)
        return xq, xk, xv

    def forward_wo(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor | None = None,
        h_gamma: torch.Tensor | None = None,
    ):
        if getattr(self, "_force_wo_bf16", False) or use_mxfp4_force_wo_bf16():
            return self._forward_wo_bf16(attn_output, residual=residual)

        debug_name = f"{getattr(self, '_lbt_debug_name', self.__class__.__name__)}:wo"
        is_nhsd = attn_output.dim() == 4
        is_3d = attn_output.dim() == 3
        if is_nhsd:
            B, H, S, D = attn_output.shape
            if h_gamma is None and (
                use_mxfp4_wo_nhsd_quant()
                and attn_output.is_cuda
                and attn_output.is_contiguous()
                and D % 64 == 0
                and S % 128 == 0
            ):
                residual_2d = residual.reshape(B * S, self.dim) if residual is not None else None
                y = _WoNHSDQuantFunction_MXFP4_TK.apply(
                    attn_output,
                    self.wo_weight,
                    debug_name,
                    residual_2d,
                )
                return y.view(B, S, self.dim)
            view_2d = _nhsd_attention_output_matrix_view(attn_output, B, H, S, D)
            if view_2d is None:
                view_2d = attn_output.transpose(1, 2).contiguous().view(B * S, H * D)
            attn_output = view_2d
            if residual is not None:
                residual = residual.reshape(B * S, self.dim)
        elif is_3d:
            B, S, D = attn_output.shape
            attn_output = attn_output.reshape(B * S, D)
            if residual is not None:
                residual = residual.reshape(B * S, self.dim)
        y = _WoFunction_MXFP4_TK.apply(
            attn_output, self.wo_weight, debug_name, residual, h_gamma
        )
        if h_gamma is not None:
            z, row_fp4, row_sc, col_fp4, col_sc, r_tile = y
            if is_3d or is_nhsd:
                z = z.view(B, S, self.dim)
            return z, row_fp4, row_sc, col_fp4, col_sc, r_tile
        if is_nhsd:
            return y.view(B, S, self.dim)
        if is_3d:
            y = y.view(B, S, self.dim)
        return y

    def init_weights(self, init_std: float = 0.02):
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self.w_qkv, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.wo_weight, mean=0.0, std=init_std)

    @classmethod
    def from_attention(cls, attention, norm, model_args=None):
        q_proj = getattr(attention, "wq", getattr(attention, "q_proj", None))
        k_proj = getattr(attention, "wk", getattr(attention, "k_proj", None))
        v_proj = getattr(attention, "wv", getattr(attention, "v_proj", None))
        o_proj = getattr(attention, "wo", getattr(attention, "o_proj", None))
        if not all(isinstance(p, nn.Linear) for p in (q_proj, k_proj, v_proj, o_proj)):
            raise AttributeError("FusedAttentionMXFP4_TK expects wq/wk/wv/wo or q_proj/k_proj/v_proj/o_proj linears")
        n_heads = getattr(attention, "n_heads", None)
        if n_heads is None:
            n_heads = attention.num_heads
        n_kv_heads = getattr(attention, "n_kv_heads", None)
        if n_kv_heads is None:
            n_kv_heads = getattr(attention, "num_key_value_heads", n_heads)
        fused = cls(
            dim=q_proj.in_features,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=attention.head_dim,
            norm_eps=getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-5)),
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )
        if q_proj.weight.device.type != "meta":
            with torch.no_grad():
                fused.w_qkv.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
                fused.wo_weight.copy_(o_proj.weight)
                if getattr(norm, "weight", None) is not None:
                    fused.norm_weight.copy_(norm.weight)
        return fused


class FusedFeedForwardNoNormMXFP4_TK(nn.Module):
    """No-norm FFN wrapper for inputs that were already normalized by the block.

    DeepSeek shared experts receive `ffn_norm(x)` from the parent block. This
    wrapper packs W1/W3 into one MXFP4 linear so the shared path quantizes the
    normalized input once instead of once for W1 and once for W3.
    """

    def __init__(self, dim, hidden_dim, bias=False, device=None, dtype=torch.bfloat16, recipe=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w13_weight = nn.Parameter(torch.empty(2 * hidden_dim, dim, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(dim, hidden_dim, device=device, dtype=dtype))
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x = x.reshape(B * S, H)
        h13 = _MXFP4LinearFunction.apply(x, self.w13_weight, None, None)
        h1, h3 = h13.split(self.hidden_dim, dim=-1)
        h = F.silu(h1) * h3
        y = _MXFP4LinearFunction.apply(h, self.w2_weight, None, None)
        if is_3d:
            y = y.view(B, S, self.dim)
        return y

    def forward_prequant_w13(
        self,
        x: torch.Tensor,
        x_q: _MXFP4RowCol,
        padded_m: int,
        padded_k: int,
    ) -> torch.Tensor:
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x = x.reshape(B * S, H)
        h13 = _MXFP4PrequantInputLinearFunction.apply(
            x,
            self.w13_weight,
            None,
            x_q.row_fp4,
            x_q.row_sc,
            x_q.col_fp4,
            x_q.col_sc,
            int(padded_m),
            int(padded_k),
        )
        h1, h3 = h13.split(self.hidden_dim, dim=-1)
        h = F.silu(h1) * h3
        y = _MXFP4LinearFunction.apply(h, self.w2_weight, None, None)
        if is_3d:
            y = y.view(B, S, self.dim)
        return y

    def invalidate_weight_cache(self):
        clear_mxfp4_weight_quant_cache()

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w13_weight[: self.hidden_dim], mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w13_weight[self.hidden_dim :], mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w2_weight, mean=0.0, std=init_std)

    @classmethod
    def from_unfused(cls, ffn, recipe=None):
        fused = cls(
            dim=ffn.w1.in_features,
            hidden_dim=ffn.w1.out_features,
            bias=False,
            device=ffn.w1.weight.device,
            dtype=ffn.w1.weight.dtype,
            recipe=recipe,
        )
        if ffn.w1.weight.device.type != "meta":
            with torch.no_grad():
                fused.w13_weight[: fused.hidden_dim].copy_(ffn.w1.weight)
                fused.w13_weight[fused.hidden_dim :].copy_(ffn.w3.weight)
                fused.w2_weight.copy_(ffn.w2.weight)
        return fused


class FusedDeepSeekMLAMXFP4_TK(nn.Module):
    """DeepSeek MLA wrapper with fused MXFP4 projection producers.

    This handles the common q_lora_rank == 0 path:
    - attention_norm + [wq; wkv_a] share one RMSNorm+activation quantization
    - kv_norm + wkv_b uses fused RMSNorm+activation quantization
    - wo remains an MXFP4 linear
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        softmax_scale: float,
        use_flex_attn: bool,
        inner_attention: nn.Module,
        input_norm_eps: float = 1e-5,
        kv_norm_eps: float = 1e-5,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = 0
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.wq_out_dim = n_heads * self.qk_head_dim
        self.wkv_a_out_dim = kv_lora_rank + qk_rope_head_dim
        self.wq_wkv_a_out_dim = self.wq_out_dim + self.wkv_a_out_dim
        self.wq_wkv_a_storage_out_dim = (
            _mxfp4_round_up_256(self.wq_wkv_a_out_dim)
            if use_mxfp4_deepseek_mla_padded_wq_wkva_param()
            else self.wq_wkv_a_out_dim
        )
        self.softmax_scale = softmax_scale
        self.use_flex_attn = use_flex_attn
        self.inner_attention = inner_attention

        self.wq_wkv_a = MXFP4RMSNormLinearTK(
            dim,
            self.wq_wkv_a_storage_out_dim,
            eps=input_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.wkv_b = MXFP4RMSNormLinearTK(
            kv_lora_rank,
            n_heads * (qk_nope_head_dim + v_head_dim),
            eps=kv_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.wo = MXFP4LinearTK(
            n_heads * v_head_dim,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks,
    ) -> torch.Tensor:
        from torchtitan.models.deepseek_v3.model.model import apply_rotary_emb

        bsz, seqlen, _ = x.size()

        q_kva = self.wq_wkv_a(x)
        q = q_kva[..., :self.wq_out_dim]
        kv = q_kva[..., self.wq_out_dim:self.wq_wkv_a_out_dim]
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        kv = self.wkv_b(kv)
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        fused_attn_wo = _maybe_tk_b300_mla_attention_wo_bshd(
            self,
            q,
            k,
            v,
            self.wo.weight,
            self.softmax_scale,
            self.use_flex_attn,
            attention_masks,
        )
        if fused_attn_wo is not None:
            return fused_attn_wo

        output = _maybe_tk_b300_mla_attention_bshd(
            self,
            q,
            k,
            v,
            self.softmax_scale,
            self.use_flex_attn,
            attention_masks,
        )
        if output is None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.use_flex_attn:
                output = self.inner_attention(
                    q,
                    k,
                    v,
                    block_mask=attention_masks,
                    scale=self.softmax_scale,
                )
            else:
                assert attention_masks is None
                output = self.inner_attention(q, k, v, scale=self.softmax_scale)

            output = output.transpose(1, 2).contiguous()
        output = output.reshape(bsz, seqlen, -1)
        return self.wo(output)

    def init_weights(self, init_std: float = 0.02):
        nn.init.ones_(self.wq_wkv_a.norm_weight)
        nn.init.trunc_normal_(self.wq_wkv_a.weight, mean=0.0, std=0.02)
        if self.wq_wkv_a_storage_out_dim > self.wq_wkv_a_out_dim:
            with torch.no_grad():
                self.wq_wkv_a.weight[self.wq_wkv_a_out_dim:].zero_()
        nn.init.ones_(self.wkv_b.norm_weight)
        nn.init.trunc_normal_(self.wkv_b.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.wo.bias is not None:
            nn.init.zeros_(self.wo.bias)

    @classmethod
    def from_attention(cls, attention, input_norm):
        if getattr(attention, "q_lora_rank", 0) != 0:
            raise ValueError("FusedDeepSeekMLAMXFP4_TK currently supports q_lora_rank == 0 only")
        fused = cls(
            dim=attention.dim,
            n_heads=attention.n_heads,
            kv_lora_rank=attention.kv_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            v_head_dim=attention.v_head_dim,
            softmax_scale=attention.softmax_scale,
            use_flex_attn=attention.use_flex_attn,
            inner_attention=attention.inner_attention,
            input_norm_eps=getattr(input_norm, "eps", 1e-5),
            kv_norm_eps=getattr(attention.kv_norm, "eps", 1e-5),
            device=attention.wq.weight.device,
            dtype=attention.wq.weight.dtype,
        )
        if attention.wq.weight.device.type != "meta":
            with torch.no_grad():
                packed_wq_wkva = torch.cat([attention.wq.weight, attention.wkv_a.weight], dim=0)
                fused.wq_wkv_a.weight[:packed_wq_wkva.shape[0]].copy_(packed_wq_wkva)
                if fused.wq_wkv_a_storage_out_dim > packed_wq_wkva.shape[0]:
                    fused.wq_wkv_a.weight[packed_wq_wkva.shape[0]:].zero_()
                if getattr(input_norm, "weight", None) is not None:
                    fused.wq_wkv_a.norm_weight.copy_(input_norm.weight)
                fused.wkv_b.weight.copy_(attention.wkv_b.weight)
                if getattr(attention.kv_norm, "weight", None) is not None:
                    fused.wkv_b.norm_weight.copy_(attention.kv_norm.weight)
                fused.wo.weight.copy_(attention.wo.weight)
        fused.wq_wkv_a._lbt_debug_name = "deepseek_mla:wq_wkv_a"
        fused.wkv_b._lbt_debug_name = "deepseek_mla:wkv_b"
        return fused


class FusedDeepSeekMLAProjMXFP4_TK(nn.Module):
    """DeepSeek MLA wrapper that only fuses the shared input projection.

    The block-level attention_norm remains outside this module.  This keeps the
    existing RMSNorm implementation while avoiding two separate activation
    quantizations/GEMMs for wq and wkv_a.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        softmax_scale: float,
        use_flex_attn: bool,
        inner_attention: nn.Module,
        kv_norm_eps: float = 1e-5,
        device=None,
        dtype=torch.bfloat16,
        norm_dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = 0
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.wq_out_dim = n_heads * self.qk_head_dim
        self.wkv_a_out_dim = kv_lora_rank + qk_rope_head_dim
        self.wq_wkv_a_out_dim = self.wq_out_dim + self.wkv_a_out_dim
        self.wq_wkv_a_storage_out_dim = (
            _mxfp4_round_up_256(self.wq_wkv_a_out_dim)
            if use_mxfp4_deepseek_mla_padded_wq_wkva_param() and not use_mxfp4_deepseek_mla_rope_epilogue()
            else self.wq_wkv_a_out_dim
        )
        self.softmax_scale = softmax_scale
        self.use_flex_attn = use_flex_attn
        self.inner_attention = inner_attention
        self.use_rope_epilogue = use_mxfp4_deepseek_mla_rope_epilogue()
        self.use_fused_kv_b = use_mxfp4_deepseek_mla_fused_kv_b()

        if self.use_rope_epilogue:
            self.wq_wkv_a = MXFP4DeepSeekMLAInputProjTK(
                dim,
                n_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                kv_lora_rank,
                device=device,
                dtype=dtype,
            )
        else:
            self.wq_wkv_a = MXFP4LinearTK(
                dim,
                self.wq_wkv_a_storage_out_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
        if self.use_fused_kv_b:
            self.kv_norm = nn.Identity()
            self.wkv_b = MXFP4RMSNormLinearTK(
                kv_lora_rank,
                n_heads * (qk_nope_head_dim + v_head_dim),
                eps=kv_norm_eps,
                device=device,
                dtype=dtype if norm_dtype is None else norm_dtype,
            )
        else:
            self.kv_norm = nn.RMSNorm(
                kv_lora_rank,
                eps=kv_norm_eps,
                device=device,
                dtype=dtype if norm_dtype is None else norm_dtype,
            )
            self.wkv_b = MXFP4LinearTK(
                kv_lora_rank,
                n_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
                device=device,
                dtype=dtype,
            )
        self.wo = MXFP4LinearTK(
            n_heads * v_head_dim,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks,
    ) -> torch.Tensor:
        from torchtitan.models.deepseek_v3.model.model import apply_rotary_emb

        bsz, seqlen, _ = x.size()

        if self.use_rope_epilogue:
            q, kv, k_pe = self.wq_wkv_a(x, freqs_cis, bsz, seqlen)
            k_pe_first = True
        else:
            q_kva = self.wq_wkv_a(x)
            q = q_kva[..., :self.wq_out_dim]
            kv = q_kva[..., self.wq_out_dim:self.wq_wkv_a_out_dim]
            q = q.view(bsz, seqlen, -1, self.qk_head_dim)
            q_nope, q_pe = torch.split(
                q,
                [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1,
            )
            q_pe = apply_rotary_emb(q_pe, freqs_cis)
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
            k_pe_first = False

        if self.use_fused_kv_b:
            kv = self.wkv_b(kv)
        else:
            kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if k_pe_first:
            k = torch.cat([k_pe.expand(-1, -1, self.n_heads, -1), k_nope], dim=-1)
        else:
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        fused_attn_wo = _maybe_tk_b300_mla_attention_wo_bshd(
            self,
            q,
            k,
            v,
            self.wo.weight,
            self.softmax_scale,
            self.use_flex_attn,
            attention_masks,
        )
        if fused_attn_wo is not None:
            return fused_attn_wo

        output = _maybe_tk_b300_mla_attention_bshd(
            self,
            q,
            k,
            v,
            self.softmax_scale,
            self.use_flex_attn,
            attention_masks,
        )
        if output is None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.use_flex_attn:
                output = self.inner_attention(
                    q,
                    k,
                    v,
                    block_mask=attention_masks,
                    scale=self.softmax_scale,
                )
            else:
                assert attention_masks is None
                output = self.inner_attention(q, k, v, scale=self.softmax_scale)

            output = output.transpose(1, 2).contiguous()
        output = output.reshape(bsz, seqlen, -1)
        return self.wo(output)

    def init_weights(self, init_std: float = 0.02):
        self.wq_wkv_a.reset_parameters()
        if (
            not self.use_rope_epilogue
            and self.wq_wkv_a_storage_out_dim > self.wq_wkv_a_out_dim
        ):
            with torch.no_grad():
                self.wq_wkv_a.weight[self.wq_wkv_a_out_dim:].zero_()
        if hasattr(self.kv_norm, "reset_parameters"):
            self.kv_norm.reset_parameters()
        nn.init.trunc_normal_(self.wkv_b.weight, mean=0.0, std=0.02)
        if self.use_fused_kv_b:
            nn.init.ones_(self.wkv_b.norm_weight)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.wo.bias is not None:
            nn.init.zeros_(self.wo.bias)

    @classmethod
    def from_attention(cls, attention, force_bf16_norms: bool = False):
        if getattr(attention, "q_lora_rank", 0) != 0:
            raise ValueError("FusedDeepSeekMLAProjMXFP4_TK currently supports q_lora_rank == 0 only")
        kv_norm_weight = getattr(attention.kv_norm, "weight", None)
        norm_dtype = attention.wq.weight.dtype if force_bf16_norms else (
            kv_norm_weight.dtype if kv_norm_weight is not None else attention.wq.weight.dtype
        )
        fused = cls(
            dim=attention.dim,
            n_heads=attention.n_heads,
            kv_lora_rank=attention.kv_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            v_head_dim=attention.v_head_dim,
            softmax_scale=attention.softmax_scale,
            use_flex_attn=attention.use_flex_attn,
            inner_attention=attention.inner_attention,
            kv_norm_eps=getattr(attention.kv_norm, "eps", 1e-5),
            device=attention.wq.weight.device,
            dtype=attention.wq.weight.dtype,
            norm_dtype=norm_dtype,
        )
        if attention.wq.weight.device.type != "meta":
            with torch.no_grad():
                if fused.use_rope_epilogue:
                    fused.wq_wkv_a.weight[:fused.wq_wkv_a.q_dim].copy_(
                        _deepseek_mla_reorder_wq_pe_first(
                            attention.wq.weight,
                            attention.n_heads,
                            attention.qk_nope_head_dim,
                            attention.qk_rope_head_dim,
                        )
                    )
                    fused.wq_wkv_a.weight[fused.wq_wkv_a.q_dim:].copy_(attention.wkv_a.weight)
                else:
                    packed_wq_wkva = torch.cat([attention.wq.weight, attention.wkv_a.weight], dim=0)
                    fused.wq_wkv_a.weight[:packed_wq_wkva.shape[0]].copy_(packed_wq_wkva)
                    if fused.wq_wkv_a_storage_out_dim > packed_wq_wkva.shape[0]:
                        fused.wq_wkv_a.weight[packed_wq_wkva.shape[0]:].zero_()
                if fused.use_fused_kv_b:
                    if kv_norm_weight is not None:
                        fused.wkv_b.norm_weight.copy_(kv_norm_weight)
                elif kv_norm_weight is not None:
                    fused.kv_norm.weight.copy_(kv_norm_weight)
                fused.wkv_b.weight.copy_(attention.wkv_b.weight)
                fused.wo.weight.copy_(attention.wo.weight)
        fused.wq_wkv_a._lbt_debug_name = "deepseek_mla_proj:wq_wkv_a"
        return fused


class FusedFeedForwardMXFP4_TK(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        norm_eps=1e-5,
        bias=False,
        device=None,
        dtype=torch.bfloat16,
        recipe=None,
        packed_w13: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epsilon = norm_eps
        self.packed_w13 = packed_w13

        self.norm_weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        if packed_w13:
            self.w13_weight = nn.Parameter(
                torch.empty(2 * hidden_dim, dim, device=device, dtype=dtype)
            )
        else:
            self.w1_weight = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=dtype))
            self.w3_weight = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(dim, hidden_dim, device=device, dtype=dtype))
        self._workspace = None
        self._workspace_device = None
        self.init_weights()

    def _w1_weight_view(self) -> torch.Tensor:
        if self.packed_w13:
            return self.w13_weight[: self.hidden_dim]
        return self.w1_weight

    def _w3_weight_view(self) -> torch.Tensor:
        if self.packed_w13:
            return self.w13_weight[self.hidden_dim :]
        return self.w3_weight

    def _ensure_workspace(self, device):
        if self._workspace_device != device:
            self._workspace = torch.empty(1, dtype=torch.uint8, device=device)
            self._workspace_device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_residual(x, residual=None)

    def forward_with_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        cde_emit: bool = False,
    ):
        debug_name = getattr(self, "_lbt_debug_name", self.__class__.__name__)
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x = x.reshape(B * S, H)
            if residual is not None:
                residual = residual.reshape(B * S, self.dim)
        self._ensure_workspace(x.device)
        empty_fp4 = _mxfp4_empty_tensor(torch.float4_e2m1fn_x2, x.device)
        empty_sc = _mxfp4_empty_tensor(torch.uint8, x.device)
        empty_r = _mxfp4_empty_tensor(torch.float32, x.device)
        y = _FusedFFNFunctionV2_MXFP4_TK.apply(
            x,
            self._w1_weight_view(),
            self._w3_weight_view(),
            self.w2_weight,
            self.norm_weight,
            self.epsilon,
            debug_name,
            residual,
            empty_fp4,
            empty_sc,
            empty_fp4,
            empty_sc,
            empty_r,
            None,
            cde_emit,
        )
        if cde_emit:
            y, row_rms_partial = y
            if is_3d:
                y = y.view(B, S, self.dim)
            return y, row_rms_partial
        if is_3d:
            y = y.view(B, S, self.dim)
        return y

    def forward_with_h_carrier(self, carrier, next_attention_gamma=None):
        z, row_fp4, row_sc, col_fp4, col_sc, r_tile = carrier
        is_3d = z.dim() == 3
        if is_3d:
            B, S, H = z.shape
            z = z.reshape(B * S, H)
        debug_name = getattr(self, "_lbt_debug_name", self.__class__.__name__)
        self._ensure_workspace(z.device)
        y = _FusedFFNFunctionV2_MXFP4_TK.apply(
            z,
            self._w1_weight_view(),
            self._w3_weight_view(),
            self.w2_weight,
            self.norm_weight,
            self.epsilon,
            debug_name,
            z,
            row_fp4,
            row_sc,
            col_fp4,
            col_sc,
            r_tile,
            next_attention_gamma,
            False,
        )
        if next_attention_gamma is not None:
            z_next, row_fp4, row_sc, col_fp4, col_sc, r_next = y
            if is_3d:
                z_next = z_next.view(B, S, self.dim)
            return z_next, row_fp4, row_sc, col_fp4, col_sc, r_next
        return y.view(B, S, self.dim) if is_3d else y

    def invalidate_weight_cache(self):
        clear_mxfp4_weight_quant_cache()

    def init_weights(self, init_std: float = 0.02):
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self._w1_weight_view(), mean=0.0, std=0.02)
        _safe_trunc_normal_(self.w2_weight, mean=0.0, std=init_std)
        _safe_trunc_normal_(self._w3_weight_view(), mean=0.0, std=init_std)

    @classmethod
    def from_unfused(cls, ffn, norm, recipe=None):
        fused = cls(
            dim=ffn.w1.in_features,
            hidden_dim=ffn.w1.out_features,
            norm_eps=getattr(norm, "eps", 1e-5),
            bias=False,
            device=ffn.w1.weight.device,
            dtype=ffn.w1.weight.dtype,
            recipe=recipe,
        )
        if ffn.w1.weight.device.type != "meta":
            with torch.no_grad():
                fused.w1_weight.copy_(ffn.w1.weight)
                fused.w3_weight.copy_(ffn.w3.weight)
                fused.w2_weight.copy_(ffn.w2.weight)
                if getattr(norm, "weight", None) is not None:
                    fused.norm_weight.copy_(norm.weight)
        return fused


class _FusedSquaredReLUFFNFunctionV2_MXFP4_TK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
        debug_name: str | None = None,
        residual: torch.Tensor | None = None,
    ):
        stage_start = _mxfp4_stage_begin("ffn_sqrelu_fwd", debug_name)
        inp = _as_contiguous_bf16(input)
        res = _as_contiguous_bf16(residual) if residual is not None else None
        nw = _as_contiguous_bf16(norm_weight.detach())
        M, K = inp.shape
        H = w1_weight.shape[0]
        N = w2_weight.shape[0]
        if res is not None and res.shape != (M, N):
            raise RuntimeError(f"MXFP4 square-ReLU FFN residual shape {tuple(res.shape)} does not match output {(M, N)}")
        ctx.has_residual = res is not None
        te_fused = _get_te_fused()

        supported = (
            _mxfp4_supported(M, K)
            and _mxfp4_supported(H, K)
            and _mxfp4_supported(N, H)
        )

        if not supported:
            ctx.fast_path = False
            normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
            h1 = normed.matmul(w1_weight.t())
            h = sqrelu_fwd(h1)
            y = h.matmul(w2_weight.t())
            if res is not None:
                y = y + res
            ctx.save_for_backward(inp, nw, inv_rms, normed, h1, h, w1_weight, w2_weight)
            ctx.epsilon = epsilon
            ctx._mxfp4_debug_name = debug_name
            _mxfp4_stage_end("ffn_sqrelu_fwd", debug_name, stage_start)
            return y

        ctx.fast_path = True
        x_q, inv_rms = _rmsnorm_quantize_row_col_bf16(te_fused, inp, nw, float(epsilon), kind="ffn")
        w1_q = _quantize_weight_row_col_bf16(_as_contiguous_bf16(w1_weight.detach()))
        h1_raw = _mxfp4_gemm_linear(x_q.row_fp4, x_q.row_sc, w1_q.row_fp4, w1_q.row_sc)
        h_q = _sqrelu_quantize_row_col_bf16(h1_raw, role="activation")
        w2_q = _quantize_weight_row_col_bf16(_as_contiguous_bf16(w2_weight.detach()))
        if (
            res is not None
            and use_mxfp4_residual_fusion_ffn()
            and use_mxfp4_linear_residual_config()
        ):
            y = _mxfp4_gemm_linear_residual(
                h_q.row_fp4,
                h_q.row_sc,
                w2_q.row_fp4,
                w2_q.row_sc,
                res,
            )
        else:
            y = _mxfp4_gemm_linear(h_q.row_fp4, h_q.row_sc, w2_q.row_fp4, w2_q.row_sc)
            if res is not None:
                y = y + res

        ctx.save_for_backward(
            inp,
            nw,
            inv_rms,
            x_q.col_fp4,
            x_q.col_sc,
            h1_raw,
            h_q.col_fp4,
            h_q.col_sc,
            w1_q.col_fp4,
            w1_q.col_sc,
            w2_q.col_fp4,
            w2_q.col_sc,
        )
        ctx._mxfp4_debug_name = debug_name
        _mxfp4_stage_end("ffn_sqrelu_fwd", debug_name, stage_start)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        debug_name = getattr(ctx, "_mxfp4_debug_name", None)
        ffn_w2_sr_key = _mxfp4_grad_producer_key(debug_name, "ffn_w2")
        ffn_deriv_sr_key = _mxfp4_grad_producer_key(debug_name, "ffn_deriv")
        stage_start = _mxfp4_stage_begin("ffn_sqrelu_bwd", debug_name)
        te_fused = _get_te_fused()
        dY = _as_contiguous_bf16(grad_output)

        if not ctx.fast_path:
            inp, nw, inv_rms, normed, h1, h, w1_weight, w2_weight = ctx.saved_tensors
            dh = dY.matmul(w2_weight)
            grad_w2 = dY.transpose(0, 1).matmul(h)
            dh1 = sqrelu_bwd(dh, h1)
            dx_normed = dh1.matmul(w1_weight)
            grad_w1 = dh1.transpose(0, 1).matmul(normed)
            grad_input, grad_norm = _mxfp4_rmsnorm_backward(te_fused, dx_normed, inp, nw, inv_rms)
            _mxfp4_stage_end("ffn_sqrelu_bwd", debug_name, stage_start)
            grad_residual = dY if ctx.has_residual else None
            return grad_input, grad_w1, grad_w2, grad_norm, None, None, grad_residual

        (
            inp,
            nw,
            inv_rms,
            x_col_fp4,
            x_col_sc,
            h1_raw,
            h_col_fp4,
            h_col_sc,
            w1_col_fp4,
            w1_col_sc,
            w2_col_fp4,
            w2_col_sc,
        ) = ctx.saved_tensors

        M, K = inp.shape
        H = h1_raw.shape[1]
        N = dY.shape[1]

        substage = _mxfp4_stage_begin("ffn_sqrelu_bwd_quant_dy", debug_name)
        dY_col_ready_event = None
        if _use_grad_split_col_overlap():
            dY_row_fp4, dY_row_sc = _quantize_row_bf16(
                dY, role="grad", producer_key=ffn_w2_sr_key
            )
            dY_col_stream = _get_mxfp4_bwd_side_stream()
            dY_col_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(dY, dY_col_stream)
            with torch.cuda.stream(dY_col_stream):
                dY_col_fp4, dY_col_sc = _quantize_col_bf16(
                    dY, role="grad", producer_key=ffn_w2_sr_key
                )
                dY_col_ready_event = torch.cuda.Event()
                dY_col_ready_event.record(dY_col_stream)
            dY_q = _MXFP4RowCol(
                row_fp4=dY_row_fp4,
                row_sc=dY_row_sc,
                col_fp4=dY_col_fp4,
                col_sc=dY_col_sc,
            )
        else:
            dY_q = _quantize_row_col_bf16(
                dY, role="grad", producer_key=ffn_w2_sr_key
            )
        _mxfp4_stage_end("ffn_sqrelu_bwd_quant_dy", debug_name, substage)

        substage = _mxfp4_stage_begin("ffn_sqrelu_bwd_dh_gemm", debug_name)
        dh = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
        _mxfp4_gemm_linear(dY_q.row_fp4, dY_q.row_sc, w2_col_fp4, w2_col_sc, dh)
        _mxfp4_stage_end("ffn_sqrelu_bwd_dh_gemm", debug_name, substage)

        grad_w2 = torch.empty(N, H, dtype=torch.bfloat16, device=inp.device)
        wgrad_stream = None
        use_w2_wgrad_overlap = (
            M >= mxfp4_ffn_wgrad_overlap_min_m()
            and use_mxfp4_ffn_w2_wgrad_overlap()
        )
        substage = _mxfp4_stage_begin("ffn_sqrelu_bwd_w2_wgrad", debug_name)
        if dY_col_ready_event is not None:
            torch.cuda.current_stream().wait_event(dY_col_ready_event)
        if use_w2_wgrad_overlap:
            wgrad_stream = _get_mxfp4_bwd_side_stream()
            wgrad_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(dY_q.col_fp4, wgrad_stream)
            _record_stream_tree(dY_q.col_sc, wgrad_stream)
            _record_stream_tree(h_col_fp4, wgrad_stream)
            _record_stream_tree(h_col_sc, wgrad_stream)
            _record_stream_tree(grad_w2, wgrad_stream)
            with torch.cuda.stream(wgrad_stream):
                _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, h_col_fp4, h_col_sc, grad_w2)
        else:
            _mxfp4_gemm_wgrad(dY_q.col_fp4, dY_q.col_sc, h_col_fp4, h_col_sc, grad_w2)
        _mxfp4_stage_end("ffn_sqrelu_bwd_w2_wgrad", debug_name, substage)

        substage = _mxfp4_stage_begin("ffn_sqrelu_bwd_deriv_quant", debug_name)
        dh1_col_ready_event = None
        if _use_grad_split_col_overlap():
            dh1 = sqrelu_bwd(dh, h1_raw)
            dh1_row_fp4, dh1_row_sc = _quantize_row_bf16(
                dh1, role="grad", producer_key=ffn_deriv_sr_key
            )
            dh1_col_stream = _get_mxfp4_bwd_side_stream()
            dh1_col_stream.wait_stream(torch.cuda.current_stream())
            _record_stream_tree(dh1, dh1_col_stream)
            with torch.cuda.stream(dh1_col_stream):
                dh1_col_fp4, dh1_col_sc = _quantize_col_bf16(
                    dh1, role="grad", producer_key=ffn_deriv_sr_key
                )
                dh1_col_ready_event = torch.cuda.Event()
                dh1_col_ready_event.record(dh1_col_stream)
            dh1_q = _MXFP4RowCol(
                row_fp4=dh1_row_fp4,
                row_sc=dh1_row_sc,
                col_fp4=dh1_col_fp4,
                col_sc=dh1_col_sc,
            )
        else:
            dh1_q = _sqrelu_deriv_quantize_row_col_bf16(
                dh,
                h1_raw,
                role="grad",
                producer_key=ffn_deriv_sr_key,
            )
        _mxfp4_stage_end("ffn_sqrelu_bwd_deriv_quant", debug_name, substage)

        substage = _mxfp4_stage_begin("ffn_sqrelu_bwd_dgrad_gemm", debug_name)
        dx_normed = torch.empty(M, K, dtype=torch.bfloat16, device=inp.device)
        _mxfp4_gemm_linear(dh1_q.row_fp4, dh1_q.row_sc, w1_col_fp4, w1_col_sc, dx_normed)
        _mxfp4_stage_end("ffn_sqrelu_bwd_dgrad_gemm", debug_name, substage)

        substage = _mxfp4_stage_begin("ffn_sqrelu_bwd_w1_wgrad", debug_name)
        grad_w1 = torch.empty(H, K, dtype=torch.bfloat16, device=inp.device)
        if dh1_col_ready_event is not None:
            torch.cuda.current_stream().wait_event(dh1_col_ready_event)
        _mxfp4_gemm_wgrad(dh1_q.col_fp4, dh1_q.col_sc, x_col_fp4, x_col_sc, grad_w1)
        _mxfp4_stage_end("ffn_sqrelu_bwd_w1_wgrad", debug_name, substage)

        substage = _mxfp4_stage_begin("ffn_sqrelu_bwd_rmsnorm", debug_name)
        grad_input, grad_norm = _mxfp4_rmsnorm_backward(te_fused, dx_normed, inp, nw, inv_rms)
        _mxfp4_stage_end("ffn_sqrelu_bwd_rmsnorm", debug_name, substage)

        if wgrad_stream is not None:
            torch.cuda.current_stream().wait_stream(wgrad_stream)

        _mxfp4_stage_end("ffn_sqrelu_bwd", debug_name, stage_start)
        grad_residual = dY if ctx.has_residual else None
        return grad_input, grad_w1, grad_w2, grad_norm, None, None, grad_residual


def _sqrelu_ffn_projection_pair(ffn):
    if hasattr(ffn, "w1") and hasattr(ffn, "w2"):
        return ffn.w1, ffn.w2
    if hasattr(ffn, "up_proj") and hasattr(ffn, "down_proj"):
        return ffn.up_proj, ffn.down_proj
    raise AttributeError(
        "square-ReLU MXFP4 fusion expects w1/w2 or up_proj/down_proj linear projections"
    )


class ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK(nn.Module):
    """Paper 1.2B FFN path: w2(relu(w1(rms_norm(x))) ** 2).

    This keeps the two-projection paper architecture on the same absorbed-norm
    MXFP4 FFN route used by the SwiGLU high-water path.
    """

    def __init__(self, dim, hidden_dim, norm_eps=1e-5, bias=False, device=None, dtype=torch.bfloat16, recipe=None):
        super().__init__()
        if bias:
            raise NotImplementedError("ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK does not support bias")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epsilon = norm_eps
        self.norm_weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.w1_weight = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(dim, hidden_dim, device=device, dtype=dtype))
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_residual(x, residual=None)

    def forward_with_residual(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x_2d = x.reshape(B * S, H)
            residual_2d = residual.reshape(B * S, self.dim) if residual is not None else None
        else:
            x_2d = x
            residual_2d = residual

        out = _FusedSquaredReLUFFNFunctionV2_MXFP4_TK.apply(
            x_2d,
            self.w1_weight,
            self.w2_weight,
            self.norm_weight,
            self.epsilon,
            getattr(self, "_lbt_debug_name", None),
            residual_2d,
        )
        if is_3d:
            out = out.view(B, S, self.dim)
        return out

    def invalidate_weight_cache(self):
        clear_mxfp4_weight_quant_cache()

    def init_weights(self, init_std: float = 0.02):
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self.w1_weight, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.w2_weight, mean=0.0, std=init_std)

    @classmethod
    def from_unfused(cls, ffn, norm, recipe=None):
        up_proj, down_proj = _sqrelu_ffn_projection_pair(ffn)
        fused = cls(
            dim=up_proj.in_features,
            hidden_dim=up_proj.out_features,
            norm_eps=getattr(norm, "eps", 1e-5),
            bias=getattr(up_proj, "bias", None) is not None,
            device=up_proj.weight.device,
            dtype=up_proj.weight.dtype,
            recipe=recipe,
        )
        if up_proj.weight.device.type != "meta":
            with torch.no_grad():
                fused.w1_weight.copy_(up_proj.weight)
                fused.w2_weight.copy_(down_proj.weight)
                if getattr(norm, "weight", None) is not None:
                    fused.norm_weight.copy_(norm.weight)
        return fused


class FusedSquaredReLUFeedForwardMXFP4_TK(nn.Module):
    """Paper 1.2B FFN path using the fastest validated MXFP4 square-ReLU route."""

    def __init__(self, dim, hidden_dim, norm_eps=1e-5, bias=False, device=None, dtype=torch.bfloat16, recipe=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epsilon = norm_eps
        self.norm_weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.w1 = MXFP4LinearTK(dim, hidden_dim, bias=bias, device=device, dtype=dtype)
        self.w2 = MXFP4LinearTK(hidden_dim, dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_residual(x, residual=None)

    def forward_with_residual(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x_2d = x.reshape(B * S, H)
            residual_2d = residual.reshape(B * S, self.dim) if residual is not None else None
        else:
            x_2d = x
            residual_2d = residual

        debug_name = getattr(self, "_lbt_debug_name", self.__class__.__name__)
        trace = use_mxfp4_stage_timing()
        if trace:
            self.w1._lbt_debug_name = f"{debug_name}:w1"
            self.w2._lbt_debug_name = f"{debug_name}:w2"
        stage_start = _mxfp4_stage_begin("ffn_sqrelu_simple_fwd", debug_name) if trace else None
        if use_mxfp4_sqrelu_fused_rms_w1():
            rms_w1_start = _mxfp4_stage_begin("ffn_sqrelu_simple_rms_w1", debug_name) if trace else None
            hidden = _MXFP4RMSNormLinearFunction.apply(
                x_2d,
                self.w1.weight,
                self.norm_weight,
                float(self.epsilon),
                f"{debug_name}:w1_rms",
            )
            if self.w1.bias is not None:
                hidden = hidden + self.w1.bias
            if trace:
                _mxfp4_stage_end("ffn_sqrelu_simple_rms_w1", debug_name, rms_w1_start)
        else:
            rms_start = _mxfp4_stage_begin("ffn_sqrelu_simple_rmsnorm", debug_name) if trace else None
            normed, _ = _get_te_fused().fused_rmsnorm_only(
                _as_contiguous_bf16(x_2d),
                _as_contiguous_bf16(self.norm_weight),
                float(self.epsilon),
            )
            if trace:
                _mxfp4_stage_end("ffn_sqrelu_simple_rmsnorm", debug_name, rms_start)
            hidden = self.w1(normed)
        if use_mxfp4_simple_sqrelu_fused_w2():
            out = _MXFP4SqReLULinearResidualFunction.apply(
                hidden,
                self.w2.weight,
                self.w2.bias,
                residual_2d,
                f"{debug_name}:w2_sqrelu",
            )
        else:
            sqrelu_start = _mxfp4_stage_begin("ffn_sqrelu_simple_sqrelu", debug_name) if trace else None
            hidden = sqrelu(hidden)
            if trace:
                _mxfp4_stage_end("ffn_sqrelu_simple_sqrelu", debug_name, sqrelu_start)
            if (
                residual_2d is not None
                and use_mxfp4_residual_fusion_ffn()
                and use_mxfp4_linear_residual_config()
            ):
                out = _MXFP4LinearResidualFunction.apply(
                    hidden,
                    self.w2.weight,
                    self.w2.bias,
                    residual_2d,
                )
            else:
                out = self.w2(hidden)
                if residual_2d is not None:
                    residual_start = _mxfp4_stage_begin("ffn_sqrelu_simple_residual", debug_name) if trace else None
                    out = out + residual_2d
                    if trace:
                        _mxfp4_stage_end("ffn_sqrelu_simple_residual", debug_name, residual_start)
        if is_3d:
            out = out.view(B, S, self.dim)
        if trace:
            _mxfp4_stage_end("ffn_sqrelu_simple_fwd", debug_name, stage_start)
        return out

    def invalidate_weight_cache(self):
        self.w1.invalidate_weight_cache()
        self.w2.invalidate_weight_cache()

    def init_weights(self, init_std: float = 0.02):
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.w2.weight, mean=0.0, std=init_std)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    @classmethod
    def from_unfused(cls, ffn, norm, recipe=None):
        up_proj, down_proj = _sqrelu_ffn_projection_pair(ffn)
        fused = cls(
            dim=up_proj.in_features,
            hidden_dim=up_proj.out_features,
            norm_eps=getattr(norm, "eps", 1e-5),
            bias=getattr(up_proj, "bias", None) is not None,
            device=up_proj.weight.device,
            dtype=up_proj.weight.dtype,
            recipe=recipe,
        )
        if up_proj.weight.device.type != "meta":
            with torch.no_grad():
                fused.w1.weight.copy_(up_proj.weight)
                fused.w2.weight.copy_(down_proj.weight)
                if up_proj.bias is not None and fused.w1.bias is not None:
                    fused.w1.bias.copy_(up_proj.bias)
                if down_proj.bias is not None and fused.w2.bias is not None:
                    fused.w2.bias.copy_(down_proj.bias)
                if getattr(norm, "weight", None) is not None:
                    fused.norm_weight.copy_(norm.weight)
        return fused
