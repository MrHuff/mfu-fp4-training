# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Opt-in CUDA allocator cache release immediately before DCP saves."""

from __future__ import annotations

import functools
import gc
import json
import os
from typing import Any

import torch


_ENABLE_ENV = "LBT_DCP_RELEASE_CUDA_CACHE"
_TELEMETRY_PREFIX = "LBT_DCP_CUDA_CACHE_MEMORY "
_WRAPPER_MARKER = "_lbt_dcp_release_cuda_cache"


def _enabled() -> bool:
    raw = os.environ.get(_ENABLE_ENV, "0")
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{_ENABLE_ENV} must be exactly 0 or 1, got {raw!r}")
    return raw == "1"


def _rank() -> int:
    distributed = torch.distributed
    if distributed.is_available() and distributed.is_initialized():
        return int(distributed.get_rank())
    return -1


def _memory_record(stage: str, checkpoint_id: Any) -> None:
    device = int(torch.cuda.current_device())
    stats = torch.cuda.memory_stats(device)
    active = int(stats.get("active_bytes.all.current", 0))
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    free, total = (int(value) for value in torch.cuda.mem_get_info(device))
    gib = 2**30
    record = {
        "active_bytes": active,
        "active_gib": active / gib,
        "allocated_bytes": allocated,
        "allocated_gib": allocated / gib,
        "checkpoint_id": str(checkpoint_id),
        "device": device,
        "event": "lbt_dcp_cuda_cache_memory",
        "free_bytes": free,
        "free_gib": free / gib,
        "rank": _rank(),
        "reserved_bytes": reserved,
        "reserved_gib": reserved / gib,
        "schema_version": 1,
        "stage": stage,
        "total_bytes": total,
        "total_gib": total / gib,
    }
    print(
        _TELEMETRY_PREFIX
        + json.dumps(record, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def install_checkpoint_cuda_cache_release() -> bool:
    """Wrap ``CheckpointManager.dcp_save`` when explicitly enabled.

    Returns ``True`` only when this call installs the wrapper.  With the gate
    unset or set to ``0`` it does not import or mutate ``CheckpointManager``.
    Repeated enabled calls are idempotent.
    """

    if not _enabled():
        return False

    from torchtitan.components.checkpoint import CheckpointManager

    original = CheckpointManager.dcp_save
    if getattr(original, _WRAPPER_MARKER, False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if "checkpoint_id" in kwargs:
            checkpoint_id = kwargs["checkpoint_id"]
        elif len(args) >= 2:
            checkpoint_id = args[1]
        else:
            raise TypeError("CheckpointManager.dcp_save call has no checkpoint_id")

        _memory_record("before_release", checkpoint_id)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        _memory_record("after_release", checkpoint_id)
        return original(self, *args, **kwargs)

    setattr(wrapped, _WRAPPER_MARKER, True)
    CheckpointManager.dcp_save = wrapped
    return True


__all__ = ["install_checkpoint_cuda_cache_release"]
