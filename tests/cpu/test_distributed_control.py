from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _module(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    from low_bits_training import distributed_control as control

    control.reset_control_process_group_for_testing()
    return control


def test_control_group_is_disabled_by_default(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.delenv("LBT_USE_GLOO_CONTROL_PLANE", raising=False)

    assert control.initialize_control_process_group() is None
    assert control.get_control_process_group() is None


def test_default_nccl_prewarm_is_disabled_by_default(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.delenv("LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO", raising=False)
    monkeypatch.setattr(
        control.dist,
        "all_reduce",
        lambda _tensor: pytest.fail("disabled prewarm reached NCCL"),
    )

    assert control.prewarm_default_nccl_process_group("cuda:0") is None


def test_default_nccl_prewarm_proves_exact_all_rank_sum(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO", "1")
    monkeypatch.setattr(control.dist, "is_available", lambda: True)
    monkeypatch.setattr(control.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(control.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(control.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(control.dist, "get_world_size", lambda: 4)
    real_tensor = torch.tensor
    monkeypatch.setattr(
        control.torch,
        "tensor",
        lambda data, *, dtype, device: real_tensor(data, dtype=dtype),
    )
    monkeypatch.setattr(control.torch.cuda, "synchronize", lambda device: None)

    def all_reduce(probe):
        probe.fill_(10)

    monkeypatch.setattr(control.dist, "all_reduce", all_reduce)

    assert control.prewarm_default_nccl_process_group("cuda:2") == (
        "backend=nccl rank=2/4 sum=10"
    )


def test_hsdp_nccl_prewarm_is_disabled_by_default(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.delenv("LBT_PREWARM_HSDP_NCCL_BEFORE_GLOO", raising=False)
    monkeypatch.setattr(
        control.dist,
        "all_reduce",
        lambda _tensor, **_kwargs: pytest.fail("disabled HSDP prewarm reached NCCL"),
    )

    assert (
        control.prewarm_hsdp_nccl_process_groups(
            "cuda:0", SimpleNamespace()
        )
        is None
    )


def test_hsdp_nccl_prewarm_proves_only_two_model_groups(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_PREWARM_HSDP_NCCL_BEFORE_GLOO", "1")
    monkeypatch.delenv("LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO", raising=False)
    monkeypatch.setattr(control.dist, "is_available", lambda: True)
    monkeypatch.setattr(control.dist, "is_initialized", lambda: True)

    legacy_shard_group = object()
    model_shard_group = object()
    replicate_group = object()

    class _MeshDimension:
        def __init__(self, group):
            self._group = group

        def get_group(self):
            return self._group

    class _WorldMesh:
        mesh_dim_names = ("dp_replicate", "dp_shard")

        def __getitem__(self, dimension):
            return _MeshDimension(
                {
                    "dp_shard": legacy_shard_group,
                    "dp_shard_cp": model_shard_group,
                    "dp_replicate": replicate_group,
                }[dimension]
            )

    parallel_dims = SimpleNamespace(
        dp_shard=8,
        dp_replicate=8,
        cp=1,
        world_size=64,
        world_mesh=_WorldMesh(),
    )
    monkeypatch.setattr(
        control.dist,
        "get_world_size",
        lambda group=None: 64 if group is None else 8,
    )
    monkeypatch.setattr(
        control.dist,
        "get_rank",
        lambda group=None: 19 if group is None else (
            3 if group is model_shard_group else 2
        ),
    )
    monkeypatch.setattr(control.dist, "get_backend", lambda _group: "nccl")
    real_tensor = torch.tensor
    monkeypatch.setattr(
        control.torch,
        "tensor",
        lambda data, *, dtype, device: real_tensor(data, dtype=dtype),
    )
    monkeypatch.setattr(control.torch.cuda, "synchronize", lambda _device: None)
    observed_groups = []

    def all_reduce(probe, *, group):
        observed_groups.append(group)
        probe.fill_(36)

    monkeypatch.setattr(control.dist, "all_reduce", all_reduce)

    assert control.prewarm_hsdp_nccl_process_groups(
        "cuda:3", parallel_dims
    ) == (
        "dimension=dp_shard_cp backend=nccl global_rank=19/64 "
        "group_rank=3/8 sum=36",
        "dimension=dp_replicate backend=nccl global_rank=19/64 "
        "group_rank=2/8 sum=36",
    )
    assert observed_groups == [model_shard_group, replicate_group]
    assert legacy_shard_group not in observed_groups


def test_hsdp_nccl_prewarm_rejects_default_world_prewarm(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_PREWARM_HSDP_NCCL_BEFORE_GLOO", "1")
    monkeypatch.setenv("LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO", "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        control.prewarm_hsdp_nccl_process_groups(
            "cuda:0", SimpleNamespace()
        )


def test_hsdp_nccl_prewarm_rejects_non_hsdp_geometry(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_PREWARM_HSDP_NCCL_BEFORE_GLOO", "1")
    monkeypatch.delenv("LBT_PREWARM_DEFAULT_NCCL_BEFORE_GLOO", raising=False)
    monkeypatch.setattr(control.dist, "is_available", lambda: True)
    monkeypatch.setattr(control.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(control.dist, "get_world_size", lambda: 64)

    with pytest.raises(RuntimeError, match="pure HSDP/CP mesh"):
        control.prewarm_hsdp_nccl_process_groups(
            "cuda:0",
            SimpleNamespace(
                dp_shard=64,
                dp_replicate=1,
                world_size=64,
            ),
        )


def test_control_group_is_all_rank_gloo_and_reused(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_USE_GLOO_CONTROL_PLANE", "1")
    group = object()
    calls = []
    monkeypatch.setattr(control.dist, "is_available", lambda: True)
    monkeypatch.setattr(control.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        control.dist,
        "new_group",
        lambda *, backend: calls.append(backend) or group,
    )
    monkeypatch.setattr(control.dist, "get_backend", lambda candidate: "gloo")
    monkeypatch.setattr(control.dist, "get_world_size", lambda candidate=None: 64)

    assert control.initialize_control_process_group() is group
    assert control.initialize_control_process_group() is group
    assert control.get_control_process_group() is group
    assert calls == ["gloo"]


def test_control_group_preflight_is_all_rank_and_deterministic(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_USE_GLOO_CONTROL_PLANE", "1")
    group = object()
    monkeypatch.setattr(control, "_CONTROL_PROCESS_GROUP", group)
    monkeypatch.setattr(control.dist, "get_rank", lambda candidate: 0)
    monkeypatch.setattr(control.dist, "get_world_size", lambda candidate: 3)

    def gather(outputs, value, *, group):
        assert group is control._CONTROL_PROCESS_GROUP
        assert value == ("lbt-gloo-control-v1", 0, 3)
        outputs[:] = [
            ("lbt-gloo-control-v1", rank, 3)
            for rank in range(3)
        ]

    monkeypatch.setattr(control.dist, "all_gather_object", gather)

    assert control.validate_control_process_group() == (
        "738c6b336cb1c93704c8d0692beadca89dd116ddddb488338da58aeabb83db4b"
    )


def test_enabled_control_group_requires_initialized_distributed(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_USE_GLOO_CONTROL_PLANE", "1")
    monkeypatch.setattr(control.dist, "is_available", lambda: True)
    monkeypatch.setattr(control.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="initialized default process group"):
        control.initialize_control_process_group()


def test_checkpoint_tensor_gather_preserves_disabled_default_path(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.delenv("LBT_USE_GLOO_CONTROL_PLANE", raising=False)
    observed = []

    def gather(outputs, value, group=None):
        observed.append((value.clone(), group))
        outputs[0].copy_(value)
        outputs[1].copy_(value + 10)

    monkeypatch.setattr(control.dist, "all_gather", gather)
    local = torch.tensor([[1, 2]], dtype=torch.int64)
    result = control.gather_checkpoint_tensor(
        local,
        expected_rank=0,
        expected_world_size=2,
    )

    assert observed[0][1] is None
    assert torch.equal(result, torch.tensor([[[1, 2]], [[11, 12]]]))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (torch.tensor([[1.0, 2.0]]), "dtype=torch.int64"),
        (torch.tensor([1, 2], dtype=torch.int64), "shape \\[producers, 2\\]"),
        (torch.ones((1, 3), dtype=torch.int64), "shape \\[producers, 2\\]"),
        (torch.empty((0, 2), dtype=torch.int64), "empty or exceeds"),
        (torch.ones((8193, 2), dtype=torch.int64), "empty or exceeds"),
    ],
)
def test_checkpoint_tensor_gather_rejects_non_control_payloads(
    monkeypatch,
    payload: torch.Tensor,
    message: str,
) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_USE_GLOO_CONTROL_PLANE", "1")
    monkeypatch.setattr(control, "_CONTROL_PROCESS_GROUP", object())

    with pytest.raises(RuntimeError, match=message):
        control.gather_checkpoint_tensor(
            payload,
            expected_rank=0,
            expected_world_size=2,
        )


def test_checkpoint_tensor_gather_rejects_non_tensor(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_USE_GLOO_CONTROL_PLANE", "1")
    monkeypatch.setattr(control, "_CONTROL_PROCESS_GROUP", object())

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        control.gather_checkpoint_tensor(  # type: ignore[arg-type]
            [[1, 2]],
            expected_rank=0,
            expected_world_size=2,
        )


def test_mxfp4_manifest_uses_control_group(monkeypatch) -> None:
    control = _module(monkeypatch)
    monkeypatch.setenv("LBT_USE_GLOO_CONTROL_PLANE", "1")
    group = object()
    monkeypatch.setattr(control, "_CONTROL_PROCESS_GROUP", group)

    from low_bits_training.quantization import mxfp4_sr_state as sr

    monkeypatch.setattr(sr.dist, "is_available", lambda: True)
    monkeypatch.setattr(sr.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sr.dist, "get_backend", lambda candidate=None: "gloo")
    monkeypatch.setattr(sr.dist, "get_rank", lambda candidate=None: 0)
    monkeypatch.setattr(sr.dist, "get_world_size", lambda candidate=None: 2)
    observed = []

    def gather(outputs, value, *, group):
        observed.append(group)
        outputs[:] = [value, value]

    monkeypatch.setattr(sr.dist, "all_gather_object", gather)
    state = sr.MXFP4SRState(
        ("layer.0:qkv", "layer.0:w2"),
        device="cpu",
        user_seed=1234,
        user_subsequence_base=0,
        training_steps=38_147,
        gradient_accumulation_steps=2,
        rank=0,
        world_size=2,
    )

    assert state.logical_keys == ("layer.0:qkv", "layer.0:w2")
    assert observed == [group]


def test_checkpoint_override_forwards_same_control_group(monkeypatch) -> None:
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    from low_bits_training import ema_checkpoint

    group = object()
    observed = []
    monkeypatch.setattr(
        ema_checkpoint,
        "dcp_load_helper",
        lambda state_dict, checkpoint_id, *, process_group=None: observed.append(
            (state_dict, checkpoint_id, process_group)
        ),
    )
    manager = SimpleNamespace(
        _lbt_control_process_group=group,
        states={},
    )
    state = {"train_state.step": object()}

    ema_checkpoint.CheckpointManager_dcp_load_override(
        manager,
        state,
        "/checkpoint/step-37000",
        False,
        False,
    )

    assert observed == [(state, "/checkpoint/step-37000", group)]


def test_checkpoint_save_planning_forwards_same_control_group(monkeypatch) -> None:
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    from low_bits_training import ema_checkpoint

    group = object()
    observed = []
    collections = []
    monkeypatch.setattr(ema_checkpoint.dist, "get_backend", lambda candidate: "gloo")
    monkeypatch.setattr(ema_checkpoint.dist, "get_rank", lambda candidate: 0)
    monkeypatch.setattr(ema_checkpoint.dist, "get_world_size", lambda candidate: 64)
    monkeypatch.setattr(
        ema_checkpoint.dcp,
        "save",
        lambda *args, **kwargs: observed.append((args, kwargs)) or "saved",
    )
    monkeypatch.setattr(
        ema_checkpoint.GarbageCollection,
        "collect",
        lambda message: collections.append(message),
    )
    manager = SimpleNamespace(_lbt_control_process_group=group)
    state = {"train_state.step": object()}

    result = ema_checkpoint.CheckpointManager_dcp_save_override(
        manager,
        state,
        "/checkpoint/step-38000",
        ema_checkpoint.checkpoint.AsyncMode.DISABLED,
        True,
    )

    assert result == "saved"
    assert observed == [
        (
            (state,),
            {
                "checkpoint_id": "/checkpoint/step-38000",
                "process_group": group,
            },
        )
    ]
    assert collections == ["GC collection invoked by checkpointer."]


def test_checkpoint_save_disabled_control_delegates_exactly(monkeypatch) -> None:
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    from low_bits_training import ema_checkpoint

    observed = []
    sentinel = object()

    def original(*args, **kwargs):
        observed.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        ema_checkpoint,
        "_original_CheckpointManager_dcp_save",
        original,
    )
    state = {"train_state.step": object()}
    managers = [
        SimpleNamespace(_lbt_control_process_group=None),
        SimpleNamespace(),
    ]
    for manager in managers:
        result = ema_checkpoint.CheckpointManager_dcp_save_override(
            manager,
            state,
            "/checkpoint/step-38000",
            ema_checkpoint.checkpoint.AsyncMode.DISABLED,
            True,
            False,
        )
        assert result is sentinel

    assert observed == [
        (
            (
                manager,
                state,
                "/checkpoint/step-38000",
                ema_checkpoint.checkpoint.AsyncMode.DISABLED,
                True,
                False,
            ),
            {},
        )
        for manager in managers
    ]
