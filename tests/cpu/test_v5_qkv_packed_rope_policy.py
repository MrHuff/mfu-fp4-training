from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


class _FakeFreqs:
    is_cuda = True

    def __init__(self, seq_len: int = 8192, rotary_pairs: int = 64) -> None:
        self._shape = (seq_len, rotary_pairs)

    def dim(self) -> int:
        return 2

    def size(self, dim: int) -> int:
        return self._shape[dim]


def _load(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_GEMM", "1")
    from low_bits_training.quantization import fused_te_linear, tk_gemm

    monkeypatch.setattr(fused_te_linear.torch, "is_complex", lambda _: True)
    return fused_te_linear, tk_gemm


def _supported(fused_te_linear, freqs=None, **overrides) -> bool:
    args = {
        "M": 32768,
        "K": 4096,
        "q_dim": 4096,
        "k_dim": 1024,
        "v_dim": 1024,
        "head_dim": 128,
        "seq_len": 8192,
        "freqs_cis": freqs or _FakeFreqs(),
    }
    args.update(overrides)
    return fused_te_linear._tk_qkv_rope_packed_supported(**args)


def test_packed_rope_defaults_on_only_for_locked_v5_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_te_linear, _ = _load(monkeypatch)
    monkeypatch.setattr(fused_te_linear, "use_tk_localcta", lambda: False)

    monkeypatch.delenv("USE_TK_QKV_PACKED_ROPE_EPILOGUE", raising=False)
    assert _supported(fused_te_linear)
    assert not _supported(fused_te_linear, M=65536)
    monkeypatch.setenv("USE_TK_QKV_PACKED_ROPE_EPILOGUE", "0")
    assert not _supported(fused_te_linear)
    monkeypatch.setenv("USE_TK_QKV_PACKED_ROPE_EPILOGUE", "1")
    assert _supported(fused_te_linear)
    assert _supported(fused_te_linear, M=65536)


def test_packed_rope_is_structurally_disabled_for_localcta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_te_linear, _ = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_QKV_PACKED_ROPE_EPILOGUE", "1")
    monkeypatch.setattr(fused_te_linear, "use_tk_localcta", lambda: True)

    assert not _supported(fused_te_linear)
    assert not fused_te_linear._tk_qkv_rope_packed_backend_available()


@pytest.mark.parametrize(
    "overrides",
    (
        {"M": 32512},
        {"K": 3968},
        {"q_dim": 3968},
        {"k_dim": 128},
        {"v_dim": 128},
        {"head_dim": 64},
        {"seq_len": 6000},
    ),
)
def test_packed_rope_rejects_unmeasured_or_unsafe_contracts(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int],
) -> None:
    fused_te_linear, _ = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_QKV_PACKED_ROPE_EPILOGUE", "1")
    monkeypatch.setattr(fused_te_linear, "use_tk_localcta", lambda: False)

    assert not _supported(fused_te_linear, **overrides)


def test_packed_rope_requires_cuda_complex_frequency_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_te_linear, _ = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_QKV_PACKED_ROPE_EPILOGUE", "1")
    monkeypatch.setattr(fused_te_linear, "use_tk_localcta", lambda: False)
    freqs = _FakeFreqs()
    freqs.is_cuda = False

    assert not _supported(fused_te_linear, freqs=freqs)


def test_packed_rope_backend_symbol_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_te_linear, tk_gemm = _load(monkeypatch)
    monkeypatch.setattr(fused_te_linear, "use_tk_localcta", lambda: False)

    monkeypatch.setattr(tk_gemm, "_get_tk_plain", lambda: SimpleNamespace())
    assert not fused_te_linear._tk_qkv_rope_packed_backend_available()

    monkeypatch.setattr(
        tk_gemm,
        "_get_tk_plain",
        lambda: SimpleNamespace(nvfp4_grouped_gemm_rope_packed_split=object()),
    )
    assert fused_te_linear._tk_qkv_rope_packed_backend_available()


def test_graph_capture_retains_and_explicitly_releases_forward_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_te_linear, tk_gemm = _load(monkeypatch)
    keepalive = fused_te_linear._TK_QKV_FORWARD_GRAPH_KEEPALIVE
    keepalive.clear()

    monkeypatch.setattr(
        fused_te_linear.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    fused_te_linear._retain_tk_qkv_forward_graph_state("eager")
    assert keepalive == []

    monkeypatch.setattr(
        fused_te_linear.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    fused_te_linear._retain_tk_qkv_forward_graph_state("normed", "quantized")
    assert keepalive == [("normed", "quantized")]

    monkeypatch.setattr(
        fused_te_linear.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(tk_gemm, "clear_tk_qkv_split3_graph_cache", lambda: None)
    monkeypatch.setattr(
        fused_te_linear,
        "clear_tk_qkv_persistent_weight_quant_state",
        lambda: None,
    )
    fused_te_linear.clear_tk_qkv_packed_graph_caches()
    assert keepalive == []


def test_qkv_forward_capture_keeps_native_inputs_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_te_linear, _ = _load(monkeypatch)
    source = inspect.getsource(fused_te_linear._FusedQKVFunction_TK.forward)
    retain_call = source[source.index("_retain_tk_qkv_forward_graph_state(") :]
    retain_call = retain_call[: retain_call.index("\n        )")]

    assert "locals().get('normed')" in retain_call
    assert "x_nvfp4" in retain_call
    assert "qkv_weight_quant_keepalive" in retain_call


def test_qkv_rms_overlap_is_eager_only_and_waits_owned_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tk_gemm = _load(monkeypatch)
    source = inspect.getsource(tk_gemm.tk_fused_qkv_backward)

    assert "and not torch.cuda.is_current_stream_capturing()" in source
    assert "force_current_stream=not plain_rms_async" in source
    assert "wait_event(rms_state['done_event'])" in source
    assert source.count("owner_key=debug_name") >= 2


def test_qkv_rms_state_owner_is_qualified_by_caller_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tk_gemm = _load(monkeypatch)
    owner = object()
    stream_a = SimpleNamespace(cuda_stream=101)
    stream_b = SimpleNamespace(cuda_stream=202)

    key_a = tk_gemm._rmsnorm_bwd_stream_owner_key("qkv", owner, stream_a)
    key_b = tk_gemm._rmsnorm_bwd_stream_owner_key("qkv", owner, stream_b)

    assert key_a != key_b
    assert key_a[1] == 101
    assert key_b[1] == 202
    assert tk_gemm._rmsnorm_bwd_stream_owner_key(
        "other", owner, stream_a
    ) is owner


def test_rmsnorm_return_outputs_are_not_retained_in_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tk_gemm = _load(monkeypatch)
    scratch_source = inspect.getsource(tk_gemm._get_rmsnorm_bwd_state)
    return_source = inspect.getsource(tk_gemm._prepare_rmsnorm_bwd_return_state)

    assert "USE_TK_TRANSIENT_RMSNORM_RETURNS" in scratch_source
    assert "'grad_input': torch.empty" in return_source
    assert "state = dict(scratch)" in return_source


def test_native_sum3_rms_uses_state_owned_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tk_gemm = _load(monkeypatch)
    state_source = inspect.getsource(tk_gemm._get_rmsnorm_bwd_state)
    launch_source = inspect.getsource(
        tk_gemm._launch_native_sum3_rmsnorm_bwd_out_async
    )

    assert "'ready_event': torch.cuda.Event()" in state_source
    assert "'done_event': torch.cuda.Event()" in state_source
    assert "state['ready_event'].record(caller_stream)" in launch_source
    assert "rms_stream.wait_event(state['ready_event'])" in launch_source
    assert "state['done_event'].record(rms_stream)" in launch_source
