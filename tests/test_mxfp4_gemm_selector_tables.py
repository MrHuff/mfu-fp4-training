import os
from types import SimpleNamespace

import pytest
import torch


os.environ.setdefault("LBT_LIGHT_IMPORT", "1")
os.environ.setdefault("LBT_QUANTIZATION_LIGHT_IMPORT", "1")

from low_bits_training.quantization import mxfp4_backend as backend  # noqa: E402
from low_bits_training.quantization import mxfp4_fused_linear as fused  # noqa: E402


EXPECTED_SELECTORS = {
    backend.MXFP4GemmSelectorKey("forward", 32768, 4096, 8192): 10,
    backend.MXFP4GemmSelectorKey("dgrad", 32768, 8192, 4096): 10,
    backend.MXFP4GemmSelectorKey("forward", 32768, 21504, 4096): 10,
    backend.MXFP4GemmSelectorKey("dgrad", 32768, 21504, 4096): 10,
    backend.MXFP4GemmSelectorKey("dgrad", 32768, 4096, 5120): 10,
    backend.MXFP4GemmSelectorKey("dgrad", 32768, 4096, 6144): 10,
}


@pytest.mark.parametrize(
    ("name", "reader"),
    [
        ("MXFP4_USE_QKV_BF16_DGRAD", fused.use_mxfp4_qkv_bf16_dgrad),
        ("MXFP4_USE_QKV_BF16_WGRAD", fused.use_mxfp4_qkv_bf16_wgrad),
        ("MXFP4_USE_QKV_BF16_Q_FORWARD", fused.use_mxfp4_qkv_bf16_q_forward),
        ("MXFP4_USE_QKV_BF16_KV_FORWARD", fused.use_mxfp4_qkv_bf16_kv_forward),
        ("MXFP4_FORCE_WO_BF16", fused.use_mxfp4_force_wo_bf16),
    ],
)
def test_mxfp4_attention_diagnostic_switches_are_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    reader,
) -> None:
    monkeypatch.delenv(name, raising=False)
    assert reader() is False

    monkeypatch.setenv(name, "1")
    assert reader() is True


@pytest.mark.parametrize(
    ("module_fqn", "ranges", "expected"),
    [
        ("layers.0", "", True),
        ("model.layers.27", "28-32", True),
        ("model.layers.31.attention", "28-32", True),
        ("model.layers.26", "28-32", False),
        ("model.layers.4", "2,5,9-12", True),
        ("model.norm", "28-32", False),
    ],
)
def test_mxfp4_fsdp_layer_scope(
    monkeypatch: pytest.MonkeyPatch,
    module_fqn: str,
    ranges: str,
    expected: bool,
) -> None:
    if ranges:
        monkeypatch.setenv("TORCHTITAN_FSDP_MXFP4_LAYER_RANGES", ranges)
    else:
        monkeypatch.delenv(
            "TORCHTITAN_FSDP_MXFP4_LAYER_RANGES",
            raising=False,
        )

    assert fused._fsdp_module_uses_mxfp4(module_fqn) is expected


def test_mxfp4_fsdp_layer_scope_rejects_invalid_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHTITAN_FSDP_MXFP4_LAYER_RANGES", "32-28")

    with pytest.raises(ValueError, match="positive 1-based ranges"):
        fused._fsdp_module_uses_mxfp4("layers.30")


def test_mixed_mxfp4_fsdp_registration_scopes_prefetch_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHTITAN_FSDP_MXFP4_LAYER_RANGES", raising=False)
    monkeypatch.setattr(fused, "_MXFP4_FSDP_FORWARD_LAYER_INDICES", None)
    monkeypatch.setattr(fused, "_MXFP4_FSDP_BACKWARD_LAYER_INDICES", None)

    fused._register_mixed_mxfp4_fsdp_layer_indices(
        forward=set(range(27, 32)),
        backward=set(range(27, 32)),
    )

    assert not fused._fsdp_module_uses_mxfp4(
        "model.layers.26",
        fused._MXFP4_FSDP_FORWARD_LAYER_INDICES,
    )
    assert fused._fsdp_module_uses_mxfp4(
        "model.layers.27",
        fused._MXFP4_FSDP_FORWARD_LAYER_INDICES,
    )


def test_mixed_mxfp4_fsdp_registration_rejects_stale_explicit_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHTITAN_FSDP_MXFP4_LAYER_RANGES", "28-32")

    with pytest.raises(RuntimeError, match="does not match the actual"):
        fused._register_mixed_mxfp4_fsdp_layer_indices(
            forward=set(range(27, 32)),
            backward=set(range(28, 32)),
        )


def test_mxfp4_registers_fsdp_comm_context_for_qkv_ordering(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPCommContext,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    calls: list[tuple[object, torch.device]] = []

    def fake_lazy_init(self, device) -> None:
        calls.append((self, device))

    def fake_pre_forward(self, _module, args, kwargs):
        return args, kwargs

    contexts: dict[int, object] = {}
    monkeypatch.setenv(
        "TORCHTITAN_FSDP_ORDER_REDUCE_SCATTER_BEFORE_MX_ROPE_QKV",
        "1",
    )
    monkeypatch.setattr(FSDPCommContext, "lazy_init", fake_lazy_init)
    monkeypatch.setattr(FSDPState, "_pre_forward", fake_pre_forward)
    monkeypatch.setattr(
        FSDPCommContext,
        "_lbt_mxfp4_reduce_scatter_qkv_ordering_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        FSDPState,
        "_lbt_mxfp4_reduce_scatter_qkv_ordering_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(fused, "_MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE", contexts)
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_COMM_CONTEXT_LOGGED_BY_DEVICE",
        set(),
    )

    context = SimpleNamespace()
    device = torch.device("cuda", 3)
    fused._install_mxfp4_fsdp_reduce_scatter_qkv_ordering()
    FSDPCommContext.lazy_init(context, device)

    assert calls == [(context, device)]
    assert contexts == {3: context}
    assert capsys.readouterr().err.count("registered communication context") == 1


def test_mxfp4_registers_fsdp_comm_context_from_pre_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPCommContext,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    def fake_lazy_init(_self, _device) -> None:
        return None

    def fake_pre_forward(_self, _module, args, kwargs):
        return args, kwargs

    contexts: dict[int, object] = {}
    monkeypatch.setenv(
        "TORCHTITAN_FSDP_ORDER_REDUCE_SCATTER_BEFORE_MX_ROPE_QKV",
        "1",
    )
    monkeypatch.setattr(FSDPCommContext, "lazy_init", fake_lazy_init)
    monkeypatch.setattr(FSDPState, "_pre_forward", fake_pre_forward)
    monkeypatch.setattr(
        FSDPCommContext,
        "_lbt_mxfp4_reduce_scatter_qkv_ordering_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        FSDPState,
        "_lbt_mxfp4_reduce_scatter_qkv_ordering_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(fused, "_MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE", contexts)
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_COMM_CONTEXT_LOGGED_BY_DEVICE",
        set(),
    )
    monkeypatch.setattr(
        fused,
        "_first_cuda_device",
        lambda _value: torch.device("cuda", 4),
    )

    fused._install_mxfp4_fsdp_reduce_scatter_qkv_ordering()
    context = object()
    state = SimpleNamespace(_comm_ctx=context)
    args, kwargs = FSDPState._pre_forward(state, object(), (object(),), {})

    assert len(args) == 1
    assert kwargs == {}
    assert contexts == {4: context}


def test_mxfp4_rope_qkv_waits_once_per_reduce_scatter_event_and_stream(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[object] = []

    class FakeStream:
        cuda_stream = 123

        def wait_event(self, event) -> None:
            calls.append(event)

    first_event = object()
    second_event = object()
    stream = FakeStream()
    context = SimpleNamespace(
        reduce_scatter_state=SimpleNamespace(event=first_event)
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE",
        {2: context},
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_REDUCE_SCATTER_EVENT_BY_STREAM",
        {},
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_REDUCE_SCATTER_WAIT_LOGGED_BY_DEVICE",
        set(),
    )
    monkeypatch.setattr(fused.torch.cuda, "current_stream", lambda _device: stream)

    device = torch.device("cuda", 2)
    assert fused._order_mxfp4_rope_qkv_after_fsdp_reduce_scatter(device)
    assert not fused._order_mxfp4_rope_qkv_after_fsdp_reduce_scatter(device)
    context.reduce_scatter_state = SimpleNamespace(event=second_event)
    assert fused._order_mxfp4_rope_qkv_after_fsdp_reduce_scatter(device)

    assert calls == [first_event, second_event]
    assert capsys.readouterr().err.count("ordering fused RoPE QKV") == 1


def test_mxfp4_rope_qkv_does_not_wait_without_pending_reduce_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE",
        {1: SimpleNamespace(reduce_scatter_state=None)},
    )

    assert not fused._order_mxfp4_rope_qkv_after_fsdp_reduce_scatter(
        torch.device("cuda", 1)
    )


def test_mxfp4_qkv_orders_only_fused_rope_route_after_reduce_scatter() -> None:
    source = fused.FusedAttentionMXFP4_TK.forward_qkv.__code__

    assert "rope_enabled" in source.co_varnames
    assert (
        "_order_mxfp4_rope_qkv_after_fsdp_reduce_scatter"
        in source.co_names
    )


def test_mxfp4_qkv_delays_only_layer_forward_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    observed_prefetch_states: list[tuple[object, ...]] = []

    def fake_pre_forward(self, _module, args, kwargs):
        observed_prefetch_states.append(tuple(self._states_to_forward_prefetch))
        return args, kwargs

    target_state = SimpleNamespace(_fsdp_param_group=object())
    state = SimpleNamespace(
        _training_state=TrainingState.IDLE,
        _states_to_forward_prefetch=[target_state],
        _fsdp_param_group=SimpleNamespace(_module_fqn="layers.0"),
    )
    delayed: dict[int, tuple[object, ...]] = {}
    monkeypatch.setenv(
        "TORCHTITAN_FSDP_DELAY_FORWARD_PREFETCH_UNTIL_MX_QKV",
        "1",
    )
    monkeypatch.setattr(FSDPState, "_pre_forward", fake_pre_forward)
    monkeypatch.setattr(
        FSDPState,
        "_lbt_delayed_mxfp4_forward_prefetch_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE",
        delayed,
    )
    monkeypatch.setattr(
        fused,
        "_first_cuda_device",
        lambda _value: torch.device("cuda", 3),
    )

    input_value = object()
    fused._install_delayed_mxfp4_fsdp_forward_prefetch()
    args, kwargs = FSDPState._pre_forward(
        state,
        object(),
        (input_value,),
        {},
    )

    assert args == (input_value,)
    assert kwargs == {}
    assert observed_prefetch_states == [
        (fused._DELAYED_MXFP4_FSDP_FORWARD_PREFETCH_SENTINEL,)
    ]
    assert state._states_to_forward_prefetch == [target_state]
    assert delayed == {3: (target_state,)}


def test_mixed_localcta_layer_cannot_arm_mxfp4_forward_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    observed_prefetch_states: list[tuple[object, ...]] = []

    def fake_pre_forward(self, _module, args, kwargs):
        observed_prefetch_states.append(tuple(self._states_to_forward_prefetch))
        return args, kwargs

    target_state = SimpleNamespace(_fsdp_param_group=object())
    localcta_state = SimpleNamespace(
        _training_state=TrainingState.IDLE,
        _states_to_forward_prefetch=[target_state],
        _fsdp_param_group=SimpleNamespace(_module_fqn="layers.26"),
    )
    mxfp4_state = SimpleNamespace(
        _training_state=TrainingState.IDLE,
        _states_to_forward_prefetch=[target_state],
        _fsdp_param_group=SimpleNamespace(_module_fqn="layers.27"),
    )
    delayed: dict[int, tuple[object, ...]] = {}
    monkeypatch.setenv(
        "TORCHTITAN_FSDP_DELAY_FORWARD_PREFETCH_UNTIL_MX_QKV",
        "1",
    )
    monkeypatch.setattr(FSDPState, "_pre_forward", fake_pre_forward)
    monkeypatch.setattr(
        FSDPState,
        "_lbt_delayed_mxfp4_forward_prefetch_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE",
        delayed,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_FORWARD_LAYER_INDICES",
        frozenset(range(27, 32)),
    )
    monkeypatch.setattr(
        fused,
        "_first_cuda_device",
        lambda _value: torch.device("cuda", 3),
    )

    fused._install_delayed_mxfp4_fsdp_forward_prefetch()
    FSDPState._pre_forward(localcta_state, object(), (object(),), {})

    assert observed_prefetch_states == [(target_state,)]
    assert delayed == {}

    FSDPState._pre_forward(mxfp4_state, object(), (object(),), {})
    assert observed_prefetch_states[-1] == (
        fused._DELAYED_MXFP4_FSDP_FORWARD_PREFETCH_SENTINEL,
    )
    assert delayed == {3: (target_state,)}


def test_mxfp4_qkv_releases_prefetch_behind_completion_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
    )

    calls: list[tuple[str, object]] = []

    class FakeEvent:
        def record(self, stream) -> None:
            calls.append(("record", stream))

    class FakeParamGroup:
        def _wait_all_gather_streams_on_event(self, event) -> None:
            calls.append(("wait", event))

    event = FakeEvent()
    stream = object()
    param_group = FakeParamGroup()
    target_state = SimpleNamespace(_fsdp_param_group=param_group)
    delayed = {3: (target_state,)}
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE",
        delayed,
    )
    monkeypatch.setattr(fused.torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(
        fused.torch.cuda,
        "current_stream",
        lambda _device: stream,
    )
    monkeypatch.setattr(
        FSDPParamGroup,
        "_prefetch_unshard",
        staticmethod(
            lambda group, pass_type: calls.append(
                (f"prefetch:{pass_type}", group)
            )
        ),
    )

    assert fused._release_delayed_mxfp4_fsdp_forward_prefetch(
        torch.device("cuda", 3)
    )
    assert delayed == {}
    assert calls == [
        ("record", stream),
        ("wait", event),
        ("prefetch:forward", param_group),
    ]


def test_mxfp4_delays_only_layer_backward_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    observed_prefetch_states: list[tuple[object, ...]] = []

    def fake_pre_backward(self, grad):
        observed_prefetch_states.append(tuple(self._states_to_backward_prefetch))
        return grad

    target_state = SimpleNamespace(_fsdp_param_group=object())
    state = SimpleNamespace(
        _states_to_backward_prefetch=[target_state],
        _fsdp_param_group=SimpleNamespace(_module_fqn="layers.0"),
    )
    active: dict[int, tuple[object, ...]] = {}
    monkeypatch.setenv(
        "TORCHTITAN_FSDP_DELAY_BACKWARD_PREFETCH_UNTIL_MX_FFN_PRODUCER",
        "1",
    )
    monkeypatch.setattr(FSDPState, "_pre_backward", fake_pre_backward)
    monkeypatch.setattr(
        FSDPState,
        "_lbt_delayed_mxfp4_backward_prefetch_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE",
        active,
    )
    monkeypatch.setattr(
        fused,
        "_first_cuda_device",
        lambda _value: torch.device("cuda", 2),
    )

    grad = object()
    fused._install_delayed_mxfp4_fsdp_backward_prefetch()
    assert FSDPState._pre_backward(state, grad) is grad
    assert observed_prefetch_states == [
        (fused._DELAYED_MXFP4_FSDP_BACKWARD_PREFETCH_SENTINEL,)
    ]
    assert state._states_to_backward_prefetch == [target_state]
    assert active == {2: (target_state,)}


def test_mixed_localcta_layer_cannot_arm_mxfp4_backward_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    observed_prefetch_states: list[tuple[object, ...]] = []

    def fake_pre_backward(self, grad):
        observed_prefetch_states.append(tuple(self._states_to_backward_prefetch))
        return grad

    target_state = SimpleNamespace(_fsdp_param_group=object())
    localcta_state = SimpleNamespace(
        _states_to_backward_prefetch=[target_state],
        _fsdp_param_group=SimpleNamespace(_module_fqn="layers.26"),
    )
    mxfp4_state = SimpleNamespace(
        _states_to_backward_prefetch=[target_state],
        _fsdp_param_group=SimpleNamespace(_module_fqn="layers.27"),
    )
    delayed: dict[int, tuple[object, ...]] = {}
    monkeypatch.setenv(
        "TORCHTITAN_FSDP_DELAY_BACKWARD_PREFETCH_UNTIL_MX_FFN_PRODUCER",
        "1",
    )
    monkeypatch.setattr(FSDPState, "_pre_backward", fake_pre_backward)
    monkeypatch.setattr(
        FSDPState,
        "_lbt_delayed_mxfp4_backward_prefetch_installed",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE",
        delayed,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_BACKWARD_LAYER_INDICES",
        frozenset(range(27, 32)),
    )
    monkeypatch.setattr(
        fused,
        "_first_cuda_device",
        lambda _value: torch.device("cuda", 2),
    )

    grad = object()
    fused._install_delayed_mxfp4_fsdp_backward_prefetch()
    assert FSDPState._pre_backward(localcta_state, grad) is grad
    assert observed_prefetch_states == [(target_state,)]
    assert delayed == {}

    assert FSDPState._pre_backward(mxfp4_state, grad) is grad
    assert observed_prefetch_states[-1] == (
        fused._DELAYED_MXFP4_FSDP_BACKWARD_PREFETCH_SENTINEL,
    )
    assert delayed == {2: (target_state,)}


def test_mxfp4_releases_backward_prefetch_behind_ffn_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
    )

    calls: list[tuple[str, object]] = []

    class FakeEvent:
        def record(self, stream) -> None:
            calls.append(("record", stream))

    class FakeParamGroup:
        def _wait_all_gather_streams_on_event(self, event) -> None:
            calls.append(("wait", event))

        def wait_for_unshard(self) -> None:
            calls.append(("drain", self))

    current_event = FakeEvent()
    side_event = FakeEvent()
    events = iter((current_event, side_event))
    stream = object()
    side_stream = object()
    param_group = FakeParamGroup()
    target_state = SimpleNamespace(_fsdp_param_group=param_group)
    delayed = {2: (target_state,)}
    active: dict[int, tuple[object, ...]] = {}
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE",
        delayed,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE",
        active,
    )
    monkeypatch.setattr(fused.torch.cuda, "Event", lambda: next(events))
    monkeypatch.setattr(
        fused.torch.cuda,
        "current_stream",
        lambda _device: stream,
    )
    monkeypatch.setattr(
        FSDPParamGroup,
        "_prefetch_unshard",
        staticmethod(
            lambda group, pass_type: calls.append(
                (f"prefetch:{pass_type}", group)
            )
        ),
    )

    assert fused._release_delayed_mxfp4_fsdp_backward_prefetch(
        torch.device("cuda", 2),
        producer_streams=(side_stream,),
    )
    assert delayed == {}
    assert active == {2: (target_state,)}
    assert calls == [
        ("record", stream),
        ("record", side_stream),
        ("wait", current_event),
        ("wait", side_event),
        ("prefetch:backward", param_group),
    ]

    assert fused._drain_mxfp4_fsdp_backward_prefetch(
        torch.device("cuda", 2)
    )
    assert active == {}
    assert calls[-1] == ("drain", param_group)


def test_mxfp4_prefetch_drain_reports_flight_state_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeParamGroup:
        def wait_for_unshard(self) -> None:
            raise RuntimeError("injected drain failure")

    target_state = SimpleNamespace(_fsdp_param_group=FakeParamGroup())
    monkeypatch.setattr(fused, "_MXFP4_FLIGHT_RECORDER_ENABLED", True)
    monkeypatch.setattr(
        fused,
        "_MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE",
        {2: (target_state,)},
    )
    monkeypatch.setattr(
        fused,
        "_mxfp4_record_stage_completion",
        lambda _stage, _name: None,
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_FLIGHT_RECORDER_LAST_COMPLETED_BY_DEVICE",
        {2: (41, "ffn_bwd", "layers.17.feed_forward")},
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_FLIGHT_RECORDER_BY_DEVICE",
        {2: fused.deque()},
    )

    with pytest.raises(RuntimeError, match="injected drain failure"):
        fused._drain_mxfp4_fsdp_backward_prefetch(torch.device("cuda", 2))

    stderr = capsys.readouterr().err
    assert "[MXFP4 FLIGHT FAILURE]" in stderr
    assert "context=fsdp_bwd_prefetch_drain" in stderr
    assert "layers.17.feed_forward" in stderr


def test_mxfp4_logs_distinct_fsdp_overlap_states_before_rope_qkv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    device = torch.device("cuda", 2)
    all_gather_event = object()
    comm_ctx = SimpleNamespace(
        all_gather_state=SimpleNamespace(event=all_gather_event),
        reduce_scatter_state=None,
    )
    monkeypatch.setenv("TORCHTITAN_FSDP_LOG_MX_ROPE_QKV_OVERLAP", "1")
    monkeypatch.setattr(fused, "_mxfp4_current_graph_task_phase", lambda: "forward")
    monkeypatch.setattr(
        fused,
        "_MXFP4_FSDP_COMM_CONTEXT_BY_DEVICE",
        {2: comm_ctx},
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_FORWARD_PREFETCH_BY_DEVICE",
        {2: (object(),)},
    )
    monkeypatch.setattr(
        fused,
        "_MXFP4_DELAYED_FSDP_BACKWARD_PREFETCH_BY_DEVICE",
        {},
    )
    active_backward: dict[int, tuple[object, ...]] = {}
    monkeypatch.setattr(
        fused,
        "_MXFP4_ACTIVE_FSDP_BACKWARD_PREFETCH_BY_DEVICE",
        active_backward,
    )
    monkeypatch.setattr(fused, "_MXFP4_FSDP_QKV_OVERLAP_STATES_LOGGED", set())

    fused._log_mxfp4_fsdp_overlap_before_rope_qkv(device)
    fused._log_mxfp4_fsdp_overlap_before_rope_qkv(device)
    first = capsys.readouterr().err
    assert first.count("fused RoPE QKV overlap state") == 1
    assert "phase=forward" in first
    assert "delayed_forward=1" in first
    assert "active_backward=0" in first
    assert "all_gather_event=1" in first
    assert "reduce_scatter_event=0" in first

    active_backward[2] = (object(),)
    fused._log_mxfp4_fsdp_overlap_before_rope_qkv(device)
    second = capsys.readouterr().err
    assert second.count("fused RoPE QKV overlap state") == 1
    assert "active_backward=1" in second


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((32768, 18688, 4096), 10),
        ((32768, 4096, 18688), 0),
        ((32768, 6144, 4096), 10),
        ((32768, 4096, 14336), 10),
        ((6144, 4096, 32768), 10),
        ((2048, 5632, 32768), 4),
        ((1024, 1024, 1024), None),
    ],
)
def test_existing_dense_shape_defaults_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int],
    expected: int | None,
) -> None:
    monkeypatch.delenv("MXFP4_DENSE_GEMM_CONFIG_ID", raising=False)
    monkeypatch.setattr(backend, "_MXFP4_EXACT_GEMM_CONFIGS", {})

    assert backend.mxfp4_dense_gemm_config_for_shape(*shape) == expected


def test_committed_selector_table_is_exact() -> None:
    assert backend.mxfp4_gemm_selector_table() == EXPECTED_SELECTORS


@pytest.mark.parametrize("key", EXPECTED_SELECTORS)
def test_each_retained_selector_resolves_by_orientation(
    key: backend.MXFP4GemmSelectorKey,
) -> None:
    assert (
        backend.mxfp4_gemm_config_for_selector(
            key.M,
            key.N,
            key.K,
            orientation=key.orientation,
        )
        == 10
    )


def test_orientationless_lookup_fires_for_existing_call_sites() -> None:
    physical_shapes = {(key.M, key.N, key.K) for key in EXPECTED_SELECTORS}

    for shape in physical_shapes:
        assert backend.mxfp4_gemm_config_for_selector(*shape) == 10
        assert backend.mxfp4_dense_gemm_config_for_shape(*shape) == 10


def test_orientationless_lookup_rejects_ambiguous_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {
        backend.MXFP4GemmSelectorKey("forward", 32768, 4096, 8192): 3,
        backend.MXFP4GemmSelectorKey("dgrad", 32768, 4096, 8192): 5,
    }
    monkeypatch.setattr(backend, "_MXFP4_EXACT_GEMM_CONFIGS", table)

    assert backend.mxfp4_gemm_config_for_selector(32768, 4096, 8192) is None
    assert (
        backend.mxfp4_gemm_config_for_selector(
            32768,
            4096,
            8192,
            orientation="forward",
        )
        == 3
    )


def test_selector_requires_an_exact_shape() -> None:
    assert (
        backend.mxfp4_gemm_config_for_selector(
            32767,
            4096,
            8192,
            orientation="forward",
        )
        is None
    )


def test_selector_validation() -> None:
    with pytest.raises(ValueError, match="orientation"):
        backend.mxfp4_gemm_config_for_selector(
            32768,
            4096,
            8192,
            orientation="sideways",
        )
    with pytest.raises(ValueError, match="positive"):
        backend.mxfp4_gemm_config_for_selector(0, 4096, 8192)


def test_selector_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MXFP4_USE_EXACT_GEMM_SELECTORS", "0")

    assert (
        backend.mxfp4_gemm_config_for_selector(
            32768,
            4096,
            8192,
            orientation="forward",
        )
        is None
    )


def test_dense_negative_override_requests_native_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MXFP4_DENSE_GEMM_CONFIG_ID", "-1")

    assert backend.mxfp4_dense_gemm_config_for_shape(32768, 6144, 4096) is None


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((4096, 2048, 32768), 18),
        ((2048, 2048, 32768), 18),
        ((2048, 6144, 32768), 18),
        ((4096, 4096, 32768), 10),
        ((1024, 1024, 32768), None),
    ],
)
def test_wgrad_selector_is_stateless(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int],
    expected: int | None,
) -> None:
    monkeypatch.setenv("MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP", "199")
    for step in ("0", "199", "200", "10000"):
        monkeypatch.setenv("LBT_TRACE_ACTIVE_STEP", step)
        assert fused._mxfp4_wgrad_config_for_shape(*shape) == expected


class _FakeGemmModule:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def mxfp4_gemm(self, *_args) -> None:
        self.calls.append(("native", None))

    def mxfp4_gemm_config(self, *_args) -> None:
        self.calls.append(("config", int(_args[-1])))

    def mxfp4_gemm_residual(self, *_args) -> None:
        self.calls.append(("residual_native", None))

    def mxfp4_gemm_residual_config(self, *_args) -> None:
        self.calls.append(("residual_config", int(_args[-1])))

    def mxfp4_batched_gemm_rope_live64(self, *_args) -> None:
        self.calls.append(("qkv_native", None))

    def mxfp4_batched_gemm_rope_live64_config(self, *_args) -> None:
        self.calls.append(("qkv_config", int(_args[-1])))

    def mxfp4_gemm_rope_live64(self, *_args) -> None:
        self.calls.append(("rope_live64", None))

    def mxfp4_gemm_rope_live64_config(self, *_args) -> None:
        self.calls.append(("rope_live64_config", int(_args[-1])))

    def mxfp4_gemm_rope_live(self, *_args) -> None:
        self.calls.append(("rope_live", None))

    def mxfp4_gemm_rope_live_config(self, *_args) -> None:
        self.calls.append(("rope_live_config", int(_args[-1])))

    def mxfp4_batched_gemm_rope_live(self, *_args) -> None:
        self.calls.append(("qkv_general_native", None))

    def mxfp4_batched_gemm_rope_live_config(self, *_args) -> None:
        self.calls.append(("qkv_general_config", int(_args[-1])))


def test_dense_wrapper_dispatches_an_unambiguous_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeGemmModule()
    monkeypatch.setattr(backend, "_load_gemm_module", lambda: module)
    monkeypatch.setattr(
        backend,
        "_MXFP4_EXACT_GEMM_CONFIGS",
        {backend.MXFP4GemmSelectorKey("dgrad", 8, 4, 256): 10},
    )
    a = torch.empty(8, 128)
    b = torch.empty(4, 128)
    scales = torch.empty(1)

    backend.mxfp4_gemm(a, scales, b, scales, torch.empty(8, 4))

    assert module.calls == [("config", 10)]


def test_residual_wrapper_does_not_consume_dense_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeGemmModule()
    monkeypatch.setattr(backend, "_load_gemm_module", lambda: module)
    monkeypatch.setattr(
        backend,
        "_MXFP4_EXACT_GEMM_CONFIGS",
        {backend.MXFP4GemmSelectorKey("forward", 8, 4, 256): 10},
    )
    a = torch.empty(8, 128)
    b = torch.empty(4, 128)
    scales = torch.empty(1)
    out = torch.empty(8, 4)

    backend.mxfp4_gemm_residual(a, scales, b, scales, out, out)

    assert module.calls == [("residual_native", None)]


def test_exact_qkv_negative_override_does_not_reenter_dense_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeGemmModule()
    monkeypatch.setattr(backend, "_load_gemm_module", lambda: module)
    monkeypatch.setenv("MXFP4_QKV_GEMM_CONFIG_M8_N4_K6", "-1")
    monkeypatch.setattr(
        backend,
        "mxfp4_dense_gemm_config_for_shape",
        lambda *_args, **_kwargs: 7,
    )
    a = torch.empty(8, 3)
    b = torch.empty(4, 3)
    scales = torch.empty(1)

    backend.mxfp4_gemm(a, scales, b, scales, torch.empty(8, 4))

    assert module.calls == [("native", None)]


def test_wo_dgrad_override_is_independent_from_qkv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_gemm_config(*args, config_id: int):
        calls.append(config_id)
        return args[4]

    monkeypatch.setenv("MXFP4_QKV_GEMM_CONFIG_M8_N4_K6", "4")
    monkeypatch.setenv("MXFP4_WO_DGRAD_GEMM_CONFIG_M8_N4_K6", "18")
    monkeypatch.setattr(fused, "mxfp4_gemm_config", fake_gemm_config)
    a = torch.empty(8, 3)
    b = torch.empty(4, 3)
    scales = torch.empty(1)
    out = torch.empty(8, 4)

    result = fused._mxfp4_gemm_wo_dgrad(a, scales, b, scales, out)

    assert result is out
    assert calls == [18]


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("4", ("qkv_config", 4)),
        ("-1", ("qkv_native", None)),
    ],
)
def test_batched_qkv_override_uses_combined_logical_width(
    monkeypatch: pytest.MonkeyPatch,
    override: str,
    expected: tuple[str, int | None],
) -> None:
    module = _FakeGemmModule()
    monkeypatch.setattr(backend, "_load_gemm_module", lambda: module)
    monkeypatch.setenv("MXFP4_QKV_GEMM_CONFIG_M8_N8_K6", override)
    a = torch.empty(8, 3)
    q = torch.empty(4, 3)
    k = torch.empty(2, 3)
    v = torch.empty(2, 3)
    scales = torch.empty(1)
    rope = torch.empty(1)

    backend.mxfp4_batched_qkv_gemm_rope_live64(
        a,
        scales,
        q,
        scales,
        k,
        scales,
        v,
        scales,
        rope,
        8,
        torch.empty(8, 4),
        torch.empty(8, 2),
        torch.empty(8, 2),
    )

    assert module.calls == [expected]


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("6", ("qkv_general_config", 6)),
        ("-1", ("qkv_general_native", None)),
    ],
)
def test_batched_rope_shape_override_is_exact_and_can_request_native(
    monkeypatch: pytest.MonkeyPatch,
    override: str,
    expected: tuple[str, int | None],
) -> None:
    module = _FakeGemmModule()
    monkeypatch.setattr(backend, "_load_gemm_module", lambda: module)
    monkeypatch.setenv(
        "MXFP4_BATCHED_GEMM_CONFIG_M8_N4_K128_B2",
        override,
    )
    a = torch.empty(8, 64)
    b = torch.empty(4, 64)
    scales = torch.empty(1)
    rope = torch.empty(8, 64, 2)
    empty_rope = torch.empty(0)
    outputs = [torch.empty(8, 4), torch.empty(8, 4)]

    result = backend.mxfp4_batched_gemm_rope_live64(
        [a, a],
        [scales, scales],
        [b, b],
        [scales, scales],
        [rope, empty_rope],
        [8, 0],
        outputs,
    )

    assert result is outputs
    assert module.calls == [expected]


@pytest.mark.parametrize("head_dim", [64, 128])
def test_qkv_optimized_rope_supports_tuned_head_dims(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
) -> None:
    monkeypatch.setenv("MXFP4_USE_QKV_DIRECT_OUTPUTS", "1")

    assert fused._mxfp4_qkv_rope_live64_supported(
        M=32768,
        K=4096,
        q_dim=4096,
        k_dim=1024,
        v_dim=1024,
        head_dim=head_dim,
        seq_len=8192,
    )


def test_qkv_optimized_rope_rejects_non_power_of_two_head_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MXFP4_USE_QKV_DIRECT_OUTPUTS", "1")

    assert not fused._mxfp4_qkv_rope_live64_supported(
        M=32768,
        K=4096,
        q_dim=4224,
        k_dim=1152,
        v_dim=1024,
        head_dim=96,
        seq_len=8192,
    )


def test_qkv_optimized_rope_table_preserves_all_head128_pairs() -> None:
    fused._MXFP4_QKV_LIVE64_ROPE_CACHE.clear()
    angles = torch.arange(8 * 64, dtype=torch.float32).view(8, 64) / 100.0
    freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)

    rope_cs = fused._get_mxfp4_live64_rope_cs(freqs, seq_len=8)

    assert rope_cs.shape == (8, 64, 2)
    torch.testing.assert_close(rope_cs[..., 0], freqs.real)
    torch.testing.assert_close(rope_cs[..., 1], freqs.imag)


def test_qkv_rope_route_does_not_silently_enable_generic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MXFP4_USE_QKV_DIRECT_OUTPUTS", "1")
    monkeypatch.setenv("MXFP4_USE_GENERIC_QKV_ROPE_EPILOGUE", "0")
    monkeypatch.setattr(
        fused,
        "mxfp4_rope_live_head_dim_available",
        lambda _head_dim: False,
    )

    route = fused._mxfp4_qkv_rope_route(
        M=32768,
        K=4096,
        q_dim=4096,
        k_dim=1024,
        v_dim=1024,
        head_dim=128,
        seq_len=8192,
    )

    assert route is None


@pytest.mark.parametrize(
    ("packed_available", "generic_enabled", "expected"),
    [(True, False, "packed"), (False, True, "generic")],
)
def test_qkv_rope_route_reports_effective_epilogue(
    monkeypatch: pytest.MonkeyPatch,
    packed_available: bool,
    generic_enabled: bool,
    expected: str,
) -> None:
    monkeypatch.setenv("MXFP4_USE_QKV_DIRECT_OUTPUTS", "1")
    monkeypatch.setenv(
        "MXFP4_USE_GENERIC_QKV_ROPE_EPILOGUE",
        "1" if generic_enabled else "0",
    )
    monkeypatch.setattr(
        fused,
        "mxfp4_rope_live_head_dim_available",
        lambda _head_dim: packed_available,
    )

    route = fused._mxfp4_qkv_rope_route(
        M=32768,
        K=4096,
        q_dim=4096,
        k_dim=1024,
        v_dim=1024,
        head_dim=128,
        seq_len=8192,
    )

    assert route == expected


@pytest.mark.parametrize(
    ("pair_dim", "expected"),
    [(32, ("rope_live64", None)), (64, ("rope_live", None))],
)
def test_packed_rope_dispatches_by_head_dimension(
    monkeypatch: pytest.MonkeyPatch,
    pair_dim: int,
    expected: tuple[str, int | None],
) -> None:
    module = _FakeGemmModule()
    monkeypatch.setattr(backend, "_load_gemm_module", lambda: module)
    a = torch.empty(8, 64)
    b = torch.empty(4, 64)
    scales = torch.empty(1)
    rope_cs = torch.empty(8, pair_dim, 2)

    backend.mxfp4_gemm_rope_live64(
        a,
        scales,
        b,
        scales,
        rope_cs,
        8,
        torch.empty(8, 4),
    )

    assert module.calls == [expected]


def test_head128_q_rope_uses_one_problem_batched_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []

    def fake_batched(
        a_list,
        _a_sc_list,
        b_list,
        _b_sc_list,
        rope_list,
        _seq_lens,
        out_list,
    ):
        calls.append((len(a_list), int(b_list[0].size(0)), int(rope_list[0].size(1))))
        return out_list

    monkeypatch.setattr(fused, "mxfp4_batched_gemm_rope_live64", fake_batched)
    monkeypatch.setattr(
        fused,
        "mxfp4_gemm_rope_live64",
        lambda *_args, **_kwargs: pytest.fail("unsafe single-GEMM launcher used"),
    )
    a = torch.empty(8, 64)
    b = torch.empty(4, 64)
    scales = torch.empty(1)
    rope_cs = torch.empty(8, 64, 2)
    out = torch.empty(8, 4)

    result = fused._mxfp4_gemm_qkv_rope_live64(
        a,
        scales,
        b,
        scales,
        rope_cs,
        8,
        out,
    )

    assert result is out
    assert calls == [(1, 4, 64)]


def test_head128_q_rope_retains_tuned_configured_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int, int]] = []

    monkeypatch.setattr(
        fused,
        "mxfp4_dense_gemm_config_for_shape",
        lambda *_args, **_kwargs: 10,
    )
    monkeypatch.setattr(
        fused,
        "mxfp4_batched_gemm_rope_live64",
        lambda *_args, **_kwargs: pytest.fail("tuned route used batched fallback"),
    )

    def fake_configured(
        a,
        _a_sc,
        b,
        _b_sc,
        rope,
        _seq_len,
        out,
        *,
        config_id,
    ):
        calls.append((int(a.size(0)), int(b.size(0)), int(rope.size(1)), config_id))
        return out

    monkeypatch.setattr(fused, "mxfp4_gemm_rope_live64_config", fake_configured)
    a = torch.empty(8, 64)
    b = torch.empty(4, 64)
    scales = torch.empty(1)
    rope_cs = torch.empty(8, 64, 2)
    out = torch.empty(8, 4)

    result = fused._mxfp4_gemm_qkv_rope_live64(
        a,
        scales,
        b,
        scales,
        rope_cs,
        8,
        out,
    )

    assert result is out
    assert calls == [(8, 4, 64, 10)]


def test_qkv_forward_records_ephemeral_rows_on_consumer_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = object()
    observed: list[tuple[object, object]] = []
    x_q = SimpleNamespace(row_fp4=torch.empty(1), row_sc=torch.empty(1))
    w_qkv_q = SimpleNamespace(row_fp4=torch.empty(1), row_sc=torch.empty(1))
    monkeypatch.setattr(fused.torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(
        fused,
        "_record_stream_tree",
        lambda value, consumer: observed.append((value, consumer)),
    )

    fused._record_mxfp4_qkv_forward_rows(x_q, w_qkv_q)

    assert observed == [
        (
            (x_q.row_fp4, x_q.row_sc, w_qkv_q.row_fp4, w_qkv_q.row_sc),
            stream,
        )
    ]


def test_qkv_forward_keepalive_reclaims_only_completed_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = object()
    events = []

    class FakeEvent:
        def __init__(self):
            self.complete = False
            self.recorded_stream = None
            events.append(self)

        def query(self):
            return self.complete

        def record(self, recorded_stream):
            self.recorded_stream = recorded_stream

    monkeypatch.setattr(fused, "_MXFP4_QKV_FORWARD_KEEPALIVE_BY_DEVICE", {})
    monkeypatch.setattr(fused.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(fused.torch.cuda, "current_stream", lambda _device: stream)
    first_ref = object()
    second_ref = object()
    device = torch.device("cuda", 3)

    fused._retain_mxfp4_qkv_forward_launch(device, first_ref)
    pending = fused._MXFP4_QKV_FORWARD_KEEPALIVE_BY_DEVICE[3]
    assert len(pending) == 1
    assert pending[0][1] == (first_ref,)
    assert events[0].recorded_stream is stream

    events[0].complete = True
    fused._retain_mxfp4_qkv_forward_launch(device, second_ref)
    assert len(pending) == 1
    assert pending[0][1] == (second_ref,)


def test_old_runtime_does_not_advertise_head128_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyRopeModule:
        def mxfp4_gemm_rope_live64(self, *_args) -> None:
            pass

        def mxfp4_gemm_rope_live64_config(self, *_args) -> None:
            pass

        def mxfp4_batched_gemm_rope_live64(self, *_args) -> None:
            pass

    monkeypatch.setattr(backend, "_load_gemm_module", _LegacyRopeModule)

    assert backend.mxfp4_rope_live_head_dim_available(64)
    assert not backend.mxfp4_rope_live_head_dim_available(128)
