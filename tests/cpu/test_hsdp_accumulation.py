import os
from types import SimpleNamespace

import pytest
import torch

from torch.distributed.fsdp import FSDPModule

from low_bits_training import metrics as metrics_module
from low_bits_training import trainer as trainer_module


def test_hsdp_reduce_scatter_accumulation_disabled_is_noop(monkeypatch):
    original = FSDPModule.set_requires_gradient_sync
    monkeypatch.delenv(
        trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV, raising=False
    )

    with trainer_module._hsdp_reduce_scatter_accumulation(False):
        assert FSDPModule.set_requires_gradient_sync is original
        assert trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV not in os.environ

    assert FSDPModule.set_requires_gradient_sync is original


def test_hsdp_reduce_scatter_accumulation_maps_only_all_reduce(monkeypatch):
    original = FSDPModule.set_requires_gradient_sync
    monkeypatch.setenv(trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV, "0")

    with trainer_module._hsdp_reduce_scatter_accumulation(True):
        assert (
            FSDPModule.set_requires_gradient_sync
            is FSDPModule.set_requires_all_reduce
        )
        assert (
            os.environ[trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV]
            == "1"
        )

    assert FSDPModule.set_requires_gradient_sync is original
    assert (
        os.environ[trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV]
        == "0"
    )


def test_hsdp_reduce_scatter_accumulation_restores_after_exception(
    monkeypatch,
):
    original = FSDPModule.set_requires_gradient_sync
    monkeypatch.delenv(
        trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV, raising=False
    )

    with pytest.raises(RuntimeError, match="test failure"):
        with trainer_module._hsdp_reduce_scatter_accumulation(True):
            raise RuntimeError("test failure")

    assert FSDPModule.set_requires_gradient_sync is original
    assert (
        trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV
        not in os.environ
    )


def test_hsdp_reduce_scatter_accumulation_rejects_broad_no_sync(
    monkeypatch,
):
    monkeypatch.setenv(trainer_module._FSDP_NO_SYNC_ACCUMULATION_ENV, "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        with trainer_module._hsdp_reduce_scatter_accumulation(True):
            pass


class _MeshDimension:
    def __init__(self, group):
        self._group = group

    def get_group(self):
        return self._group


class _WorldMesh:
    mesh_dim_names = ("dp_replicate", "dp_shard")

    def __init__(self, legacy_shard_group, model_shard_group, replicate_group):
        self._groups = {
            "dp_shard": legacy_shard_group,
            "dp_shard_cp": model_shard_group,
            "dp_replicate": replicate_group,
        }

    def __getitem__(self, dimension):
        return _MeshDimension(self._groups[dimension])


class _FlatMesh:
    mesh_dim_names = ("dp_cp",)

    def size(self):
        return 64


def _parallel_dims():
    legacy_shard_group = object()
    model_shard_group = object()
    replicate_group = object()
    return SimpleNamespace(
        dp_shard=8,
        dp_replicate=8,
        cp=1,
        world_size=64,
        world_mesh=_WorldMesh(
            legacy_shard_group, model_shard_group, replicate_group
        ),
    ), legacy_shard_group, model_shard_group, replicate_group


def test_hsdp_scalar_metrics_use_only_two_model_groups(monkeypatch):
    (
        parallel_dims,
        legacy_shard_group,
        model_shard_group,
        replicate_group,
    ) = _parallel_dims()
    monkeypatch.setenv(
        trainer_module._HSDP_HIERARCHICAL_SCALAR_METRICS_ENV, "1"
    )
    monkeypatch.setenv("LBT_PREWARM_HSDP_NCCL_BEFORE_GLOO", "1")
    monkeypatch.delenv(
        "LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO", raising=False
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group: 8
    )
    monkeypatch.setattr(
        torch.distributed, "get_backend", lambda group: "nccl"
    )
    calls = []

    def all_reduce(value, *, op, group):
        calls.append((op, group))
        if op == torch.distributed.ReduceOp.SUM:
            value.mul_(8)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    flat_mesh = _FlatMesh()

    with trainer_module._hsdp_hierarchical_scalar_metrics(parallel_dims):
        assert trainer_module.dist_utils.dist_sum(
            torch.tensor(2.0), flat_mesh
        ) == 128.0
        assert trainer_module.dist_utils.dist_mean(
            torch.tensor(2.0), flat_mesh
        ) == 2.0
        assert trainer_module.dist_utils.dist_max(
            torch.tensor(2.0), flat_mesh
        ) == 2.0

    assert [group for _, group in calls] == [
        model_shard_group,
        replicate_group,
        model_shard_group,
        replicate_group,
        model_shard_group,
        replicate_group,
    ]
    assert legacy_shard_group not in [group for _, group in calls]


def test_hsdp_total_throughput_uses_only_model_groups(monkeypatch):
    (
        parallel_dims,
        legacy_shard_group,
        model_shard_group,
        replicate_group,
    ) = _parallel_dims()
    monkeypatch.setenv(
        trainer_module._HSDP_HIERARCHICAL_SCALAR_METRICS_ENV, "1"
    )
    calls = []

    def dist_sum(value, mesh):
        calls.append(mesh.get_group())
        return value * 8

    monkeypatch.setattr(metrics_module, "dist_sum", dist_sum)
    logger = metrics_module.WBMetricLogger(parallel_dims)
    monkeypatch.setattr(metrics_module.wandb, "run", None)
    values = {"throughput(tps)": 2.0}

    logger.log(values, step=37010)

    assert values["total_throughput(tps)"] == 128.0
    assert calls == [model_shard_group, replicate_group]
    assert legacy_shard_group not in calls


def test_hsdp_scalar_metrics_restore_torchtitan_functions(monkeypatch):
    parallel_dims, _, _, _ = _parallel_dims()
    monkeypatch.delenv(
        trainer_module._HSDP_HIERARCHICAL_SCALAR_METRICS_ENV,
        raising=False,
    )
    original_sum = trainer_module.dist_utils.dist_sum
    original_mean = trainer_module.dist_utils.dist_mean
    original_max = trainer_module.dist_utils.dist_max

    with trainer_module._hsdp_hierarchical_scalar_metrics(parallel_dims):
        assert trainer_module.dist_utils.dist_sum is original_sum
        assert trainer_module.dist_utils.dist_mean is original_mean
        assert trainer_module.dist_utils.dist_max is original_max

    assert trainer_module.dist_utils.dist_sum is original_sum
    assert trainer_module.dist_utils.dist_mean is original_mean
    assert trainer_module.dist_utils.dist_max is original_max


def test_hsdp_scalar_metrics_reject_default_world_prewarm(monkeypatch):
    parallel_dims, _, _, _ = _parallel_dims()
    monkeypatch.setenv(
        trainer_module._HSDP_HIERARCHICAL_SCALAR_METRICS_ENV, "1"
    )
    monkeypatch.setenv("LBT_PREWARM_HSDP_NCCL_BEFORE_GLOO", "1")
    monkeypatch.setenv("LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO", "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        with trainer_module._hsdp_hierarchical_scalar_metrics(parallel_dims):
            pass
