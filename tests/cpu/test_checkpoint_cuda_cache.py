from __future__ import annotations

import json

import pytest


ENABLE_ENV = "LBT_DCP_RELEASE_CUDA_CACHE"
PREFIX = "LBT_DCP_CUDA_CACHE_MEMORY "


def _fake_dcp_save(events):
    def original(self, state_dict, checkpoint_id, async_mode, *args, **kwargs):
        events.append("original")
        return (state_dict, checkpoint_id, async_mode, args, kwargs)

    return original


def test_gate_off_leaves_dcp_save_untouched(monkeypatch) -> None:
    from torchtitan.components.checkpoint import CheckpointManager

    from low_bits_training.checkpoint_cuda_cache import (
        install_checkpoint_cuda_cache_release,
    )

    events = []
    original = _fake_dcp_save(events)
    monkeypatch.setattr(CheckpointManager, "dcp_save", original)
    monkeypatch.delenv(ENABLE_ENV, raising=False)

    assert install_checkpoint_cuda_cache_release() is False
    assert CheckpointManager.dcp_save is original
    result = CheckpointManager.dcp_save(object(), {"model": 1}, "step-6", "sync")
    assert result == ({"model": 1}, "step-6", "sync", (), {})
    assert events == ["original"]


def test_enabled_release_order_telemetry_and_forwarding(monkeypatch, capsys) -> None:
    import low_bits_training.checkpoint_cuda_cache as cache_release
    from torchtitan.components.checkpoint import CheckpointManager

    events = []
    original = _fake_dcp_save(events)
    monkeypatch.setattr(CheckpointManager, "dcp_save", original)
    monkeypatch.setenv(ENABLE_ENV, "1")

    snapshots = iter(
        [
            (101, 102, 103, 104, 105),
            (201, 202, 203, 204, 205),
        ]
    )
    current = {}

    def memory_stats(device):
        events.append("telemetry")
        active, allocated, reserved, free, total = next(snapshots)
        current.update(
            active=active,
            allocated=allocated,
            reserved=reserved,
            free=free,
            total=total,
        )
        return {"active_bytes.all.current": active}

    monkeypatch.setattr(cache_release.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(cache_release.torch.cuda, "memory_stats", memory_stats)
    monkeypatch.setattr(
        cache_release.torch.cuda,
        "memory_allocated",
        lambda device: current["allocated"],
    )
    monkeypatch.setattr(
        cache_release.torch.cuda,
        "memory_reserved",
        lambda device: current["reserved"],
    )
    monkeypatch.setattr(
        cache_release.torch.cuda,
        "mem_get_info",
        lambda device: (current["free"], current["total"]),
    )
    monkeypatch.setattr(
        cache_release.torch.cuda,
        "synchronize",
        lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(
        cache_release.gc, "collect", lambda: events.append("gc_collect") or 0
    )
    monkeypatch.setattr(
        cache_release.torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(cache_release.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(cache_release.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(cache_release.torch.distributed, "get_rank", lambda: 3)

    assert cache_release.install_checkpoint_cuda_cache_release() is True
    assert cache_release.install_checkpoint_cuda_cache_release() is False
    result = CheckpointManager.dcp_save(
        object(),
        {"model": 1},
        "/checkpoints/step-6",
        "sync",
        enable_garbage_collection=True,
    )

    assert result == (
        {"model": 1},
        "/checkpoints/step-6",
        "sync",
        (),
        {"enable_garbage_collection": True},
    )
    assert events == [
        "telemetry",
        "gc_collect",
        "synchronize",
        "empty_cache",
        "synchronize",
        "telemetry",
        "original",
    ]

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    records = [json.loads(line.removeprefix(PREFIX)) for line in lines]
    assert all(line.startswith(PREFIX) for line in lines)
    assert [record["stage"] for record in records] == [
        "before_release",
        "after_release",
    ]
    assert [record["active_bytes"] for record in records] == [101, 201]
    assert [record["allocated_bytes"] for record in records] == [102, 202]
    assert [record["reserved_bytes"] for record in records] == [103, 203]
    assert [record["free_bytes"] for record in records] == [104, 204]
    assert [record["total_bytes"] for record in records] == [105, 205]
    assert all(record["checkpoint_id"] == "/checkpoints/step-6" for record in records)
    assert all(record["device"] == 2 for record in records)
    assert all(record["rank"] == 3 for record in records)
    assert all(record["schema_version"] == 1 for record in records)


def test_checkpoint_id_keyword_and_non_distributed_rank(monkeypatch, capsys) -> None:
    import low_bits_training.checkpoint_cuda_cache as cache_release
    from torchtitan.components.checkpoint import CheckpointManager

    events = []
    monkeypatch.setattr(CheckpointManager, "dcp_save", _fake_dcp_save(events))
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setattr(cache_release.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cache_release.torch.cuda,
        "memory_stats",
        lambda device: {"active_bytes.all.current": 1},
    )
    monkeypatch.setattr(cache_release.torch.cuda, "memory_allocated", lambda device: 2)
    monkeypatch.setattr(cache_release.torch.cuda, "memory_reserved", lambda device: 3)
    monkeypatch.setattr(cache_release.torch.cuda, "mem_get_info", lambda device: (4, 5))
    monkeypatch.setattr(cache_release.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(cache_release.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cache_release.gc, "collect", lambda: 0)
    monkeypatch.setattr(cache_release.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(cache_release.torch.distributed, "is_initialized", lambda: False)

    assert cache_release.install_checkpoint_cuda_cache_release() is True
    CheckpointManager.dcp_save(
        object(),
        state_dict={},
        checkpoint_id="keyword-step",
        async_mode="sync",
    )

    records = [
        json.loads(line.removeprefix(PREFIX))
        for line in capsys.readouterr().out.splitlines()
    ]
    assert len(records) == 2
    assert all(record["checkpoint_id"] == "keyword-step" for record in records)
    assert all(record["rank"] == -1 for record in records)


def test_invalid_gate_is_rejected(monkeypatch) -> None:
    from low_bits_training.checkpoint_cuda_cache import (
        install_checkpoint_cuda_cache_release,
    )

    monkeypatch.setenv(ENABLE_ENV, "true")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        install_checkpoint_cuda_cache_release()
