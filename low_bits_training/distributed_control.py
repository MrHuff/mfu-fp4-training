# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Small-object distributed control plane for LBT startup and checkpointing.

Python-object collectives are control traffic, not model traffic.  On some
multi-node deployments the default NCCL process group can remain live for
tensor collectives while hanging indefinitely in ``all_gather_object``.  This
module provides an explicitly opt-in, process-wide Gloo group for those small
control messages and for DCP planning.  Numerical tensor collectives remain on
the default process group.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import torch
import torch.distributed as dist


_CONTROL_PLANE_ENV = "LBT_USE_GLOO_CONTROL_PLANE"
_NCCL_PREWARM_ENV = "LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO"
_HSDP_NCCL_PREWARM_ENV = "LBT_PREWARM_HSDP_NCCL_BEFORE_GLOO"
_CONTROL_PROCESS_GROUP: Any | None = None
_LOGGER = logging.getLogger(__name__)

# This path exists only for the per-producer Philox ``[seed, subsequence]``
# checkpoint table.  Keep the accepted payload deliberately small so a future
# caller cannot accidentally move model or optimizer tensors onto Gloo.  The
# bound allows 8,192 producers (far above the 128-producer H16 production
# route) while limiting each local row to 128 KiB.
_MAX_CHECKPOINT_CONTROL_NUMEL = 16_384


def _enabled() -> bool:
    return os.getenv(_CONTROL_PLANE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _nccl_prewarm_enabled() -> bool:
    return os.getenv(_NCCL_PREWARM_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _hsdp_nccl_prewarm_enabled() -> bool:
    return os.getenv(_HSDP_NCCL_PREWARM_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def prewarm_default_nccl_process_group(device: torch.device | str) -> str | None:
    """Prove the default model-traffic NCCL group before creating Gloo.

    This is deliberately opt-in and performs one exact one-element all-reduce.
    It catches a bad Lambda network-plugin selection before checkpoint restore,
    training, or creation of the separate Gloo control group.
    """

    if not _nccl_prewarm_enabled():
        return None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{_NCCL_PREWARM_ENV}=1 requires an initialized default process group"
        )
    backend = str(dist.get_backend()).lower()
    if backend != "nccl":
        raise RuntimeError(
            "default process-group prewarm requires backend=nccl, "
            f"observed backend={backend!r}"
        )
    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError(
            "default NCCL prewarm requires a CUDA device, "
            f"observed device={target}"
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    probe = torch.tensor([rank + 1], dtype=torch.int64, device=target)
    dist.all_reduce(probe)
    torch.cuda.synchronize(target)
    observed = int(probe.item())
    expected = world_size * (world_size + 1) // 2
    if observed != expected:
        raise RuntimeError(
            "default NCCL prewarm returned an invalid all-rank sum: "
            f"rank={rank}, world={world_size}, observed={observed}, expected={expected}"
        )
    return f"backend=nccl rank={rank}/{world_size} sum={observed}"


def prewarm_hsdp_nccl_process_groups(
    device: torch.device | str,
    parallel_dims: Any,
) -> tuple[str, str] | None:
    """Prove the two NCCL groups that carry HSDP model traffic.

    TorchTitan applies HSDP on the ``dp_replicate`` and flattened
    ``dp_shard_cp`` dimensions.  The latter is a distinct process group from
    ``dp_shard`` even when context parallelism is one, so proving the unflattened
    group would not prove the communicator that FSDP actually uses.  Prewarming
    the default all-rank process group would create a scientifically unused
    world-sized NCCL communicator and can fail on clusters where the smaller
    production groups are healthy.  This opt-in gate therefore performs one
    exact one-element all-reduce on each real model group and never touches the
    default group.
    """

    if not _hsdp_nccl_prewarm_enabled():
        return None
    if _nccl_prewarm_enabled():
        raise RuntimeError(
            f"{_HSDP_NCCL_PREWARM_ENV}=1 and {_NCCL_PREWARM_ENV}=1 are "
            "mutually exclusive"
        )
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{_HSDP_NCCL_PREWARM_ENV}=1 requires an initialized default "
            "process group"
        )

    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError(
            "HSDP NCCL prewarm requires a CUDA device, "
            f"observed device={target}"
        )

    dp_shard = int(getattr(parallel_dims, "dp_shard", 0))
    dp_replicate = int(getattr(parallel_dims, "dp_replicate", 0))
    cp = int(getattr(parallel_dims, "cp", 1))
    dp_shard_cp = dp_shard * cp
    configured_world = int(getattr(parallel_dims, "world_size", 0))
    observed_world = dist.get_world_size()
    if (
        dp_shard <= 1
        or dp_replicate <= 1
        or cp <= 0
        or configured_world != observed_world
        or dp_shard_cp * dp_replicate != observed_world
    ):
        raise RuntimeError(
            "HSDP NCCL prewarm requires a pure HSDP/CP mesh: "
            f"dp_shard={dp_shard}, cp={cp}, dp_replicate={dp_replicate}, "
            f"configured_world={configured_world}, observed_world={observed_world}"
        )

    world_mesh = parallel_dims.world_mesh
    mesh_dim_names = tuple(getattr(world_mesh, "mesh_dim_names", ()))
    required_dims = ("dp_shard", "dp_replicate") + (("cp",) if cp > 1 else ())
    missing = [name for name in required_dims if name not in mesh_dim_names]
    if missing:
        raise RuntimeError(
            "HSDP NCCL prewarm could not find required model mesh dimensions: "
            f"missing={missing}, observed={mesh_dim_names}"
        )

    global_rank = dist.get_rank()
    receipts: list[str] = []
    for dimension, expected_size in (
        ("dp_shard_cp", dp_shard_cp),
        ("dp_replicate", dp_replicate),
    ):
        group = world_mesh[dimension].get_group()
        backend = str(dist.get_backend(group)).lower()
        group_rank = dist.get_rank(group)
        group_size = dist.get_world_size(group)
        if backend != "nccl" or group_size != expected_size:
            raise RuntimeError(
                "HSDP model process-group contract drifted: "
                f"dimension={dimension}, backend={backend!r}, "
                f"group_size={group_size}, expected_size={expected_size}"
            )

        probe = torch.tensor(
            [group_rank + 1], dtype=torch.int64, device=target
        )
        dist.all_reduce(probe, group=group)
        torch.cuda.synchronize(target)
        observed = int(probe.item())
        expected = group_size * (group_size + 1) // 2
        if observed != expected:
            raise RuntimeError(
                "HSDP NCCL prewarm returned an invalid group sum: "
                f"dimension={dimension}, global_rank={global_rank}, "
                f"group_rank={group_rank}, group_size={group_size}, "
                f"observed={observed}, expected={expected}"
            )
        receipts.append(
            f"dimension={dimension} backend=nccl global_rank={global_rank}/"
            f"{observed_world} group_rank={group_rank}/{group_size} sum={observed}"
        )

    return receipts[0], receipts[1]


def initialize_control_process_group() -> Any | None:
    """Create the all-rank Gloo control group once when explicitly enabled."""

    global _CONTROL_PROCESS_GROUP
    if not _enabled():
        return None
    if _CONTROL_PROCESS_GROUP is not None:
        return _CONTROL_PROCESS_GROUP
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{_CONTROL_PLANE_ENV}=1 requires an initialized default process group"
        )

    group = dist.new_group(backend="gloo")
    backend = str(dist.get_backend(group)).lower()
    expected_world = dist.get_world_size()
    control_world = dist.get_world_size(group)
    if backend != "gloo":
        raise RuntimeError(
            f"LBT control process group must use Gloo, observed backend={backend!r}"
        )
    if control_world != expected_world:
        raise RuntimeError(
            "LBT control process group must contain every training rank: "
            f"control_world={control_world}, default_world={expected_world}"
        )
    _CONTROL_PROCESS_GROUP = group
    return group


def get_control_process_group() -> Any | None:
    """Return the initialized control group, or ``None`` when it is disabled."""

    if not _enabled():
        return None
    if _CONTROL_PROCESS_GROUP is None:
        raise RuntimeError(
            f"{_CONTROL_PLANE_ENV}=1 but the Gloo control process group is uninitialized"
        )
    return _CONTROL_PROCESS_GROUP


def validate_control_process_group() -> str | None:
    """Run a tiny all-rank object collective and return its receipt digest."""

    group = get_control_process_group()
    if group is None:
        return None
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    local_receipt = ("lbt-gloo-control-v1", rank, world_size)
    receipts: list[object] = [None] * world_size
    dist.all_gather_object(receipts, local_receipt, group=group)
    expected = [
        ("lbt-gloo-control-v1", expected_rank, world_size)
        for expected_rank in range(world_size)
    ]
    if receipts != expected:
        raise RuntimeError(
            "LBT Gloo control-plane preflight returned an invalid all-rank receipt: "
            f"rank={rank}, observed={receipts!r}"
        )
    payload = json.dumps(receipts, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def gather_checkpoint_tensor(
    local_tensor: torch.Tensor,
    *,
    expected_rank: int,
    expected_world_size: int,
) -> torch.Tensor:
    """Gather a tiny replicated checkpoint table on the control plane.

    With the opt-in disabled this preserves the historical same-device/default
    process-group behavior exactly.  With it enabled, checkpoint-control state
    is copied to CPU and gathered by Gloo; model and training tensor collectives
    remain untouched on their normal backend.
    """

    if not torch.is_tensor(local_tensor):
        raise TypeError(
            "LBT checkpoint control payload must be a torch.Tensor"
        )
    if local_tensor.layout != torch.strided:
        raise RuntimeError(
            "LBT checkpoint control payload must be a dense strided tensor"
        )
    if local_tensor.dtype != torch.int64:
        raise RuntimeError(
            "LBT checkpoint control payload must have dtype=torch.int64, "
            f"got {local_tensor.dtype}"
        )
    if local_tensor.ndim != 2 or local_tensor.shape[-1] != 2:
        raise RuntimeError(
            "LBT checkpoint control payload must have shape [producers, 2], "
            f"got {tuple(local_tensor.shape)}"
        )
    if not 0 < local_tensor.numel() <= _MAX_CHECKPOINT_CONTROL_NUMEL:
        raise RuntimeError(
            "LBT checkpoint control payload is empty or exceeds the bounded "
            f"tiny-table contract: numel={local_tensor.numel()}, "
            f"max={_MAX_CHECKPOINT_CONTROL_NUMEL}"
        )

    group = get_control_process_group()
    if group is None:
        gathered = [
            torch.empty_like(local_tensor) for _ in range(expected_world_size)
        ]
        dist.all_gather(gathered, local_tensor)
        return torch.stack(gathered).cpu()

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if (rank, world_size) != (expected_rank, expected_world_size):
        raise RuntimeError(
            "LBT checkpoint control rank/world drifted: "
            f"expected=({expected_rank},{expected_world_size}), "
            f"observed=({rank},{world_size})"
        )
    local_cpu = local_tensor.detach().to(device="cpu", copy=True).contiguous()
    gathered_cpu = [
        torch.empty_like(local_cpu) for _ in range(expected_world_size)
    ]
    dist.all_gather(gathered_cpu, local_cpu, group=group)
    _LOGGER.info(
        "LBT checkpoint tensor control receipt: backend=gloo rank=%d/%d "
        "shape=%s dtype=%s",
        rank,
        world_size,
        tuple(local_cpu.shape),
        local_cpu.dtype,
    )
    return torch.stack(gathered_cpu)


def reset_control_process_group_for_testing() -> None:
    """Reset module state without destroying a process group (tests only)."""

    global _CONTROL_PROCESS_GROUP
    _CONTROL_PROCESS_GROUP = None
