from __future__ import annotations

import torch


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    from low_bits_training.quantization import tk_gemm

    return tk_gemm


class _Split3Recorder:
    def __init__(self):
        self.args = None

    def tk_group_quantize_dim1_split3_for_gemm(self, *args):
        self.args = args
        return args


def test_qkv_split3_uses_global_sr_by_default(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_USE_STOCHASTIC_ROUNDING", "1")
    monkeypatch.delenv("NVFP4_SR_GRAD", raising=False)
    monkeypatch.setenv("NVFP4_RNG_SEED", "17")
    monkeypatch.setenv("NVFP4_RNG_SUBSEQUENCE_BASE", "29")
    recorder = _Split3Recorder()

    tk_gemm._plain_qkv_split3_quantize_eager(recorder, "q", "k", "v")

    assert recorder.args == ("q", "k", "v", True, 17, 29, "both")


def test_qkv_split3_honors_explicit_gradient_sr_override(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_USE_STOCHASTIC_ROUNDING", "1")
    monkeypatch.setenv("NVFP4_SR_GRAD", "0")
    recorder = _Split3Recorder()

    tk_gemm._plain_qkv_split3_quantize_eager(recorder, "q", "k", "v")

    assert recorder.args[3:] == (False, 0, 0, "none")


def test_qkv_split3_honors_gradient_sr_axes(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "dgrad")
    recorder = _Split3Recorder()

    tk_gemm._plain_qkv_split3_quantize_eager(recorder, "q", "k", "v")

    assert recorder.args[3:] == (True, 0, 0, "row")


def test_qkv_split3_graph_key_includes_sr_policy(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)

    class _Device:
        index = 0

    class _Tensor:
        device = _Device()
        shape = (128, 128)
        dtype = "bf16"

        @staticmethod
        def data_ptr():
            return 123

        @staticmethod
        def stride():
            return (128, 1)

    monkeypatch.setattr(
        tk_gemm.torch.cuda,
        "current_stream",
        lambda _device: type("_Stream", (), {"cuda_stream": 7})(),
    )
    tensors = (_Tensor(), _Tensor(), _Tensor())
    monkeypatch.setenv("NVFP4_SR_GRAD", "0")
    deterministic = tk_gemm._plain_qkv_split3_graph_key(tensors, "owner")
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    stochastic = tk_gemm._plain_qkv_split3_graph_key(tensors, "owner")

    assert deterministic != stochastic


def test_plain_qkv_production_path_skips_fallback_only_scratch(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setattr(tk_gemm, "_use_plain_tk_small_m_qkv_dgrad_eager", lambda *_: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_bf16_dgrad", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_bf16_wgrad", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_plain_batched_accum_dgrad", lambda: True)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_cached_return_transpose", lambda: True)
    monkeypatch.setattr(tk_gemm, "_debug_qkv_capture_path", lambda: "")
    monkeypatch.delenv("USE_TK_DEBUG_QKV_DGRAD_REF", raising=False)
    monkeypatch.delenv("USE_TK_DEBUG_QKV_WGRAD_REF", raising=False)

    policy = tk_gemm._qkv_fused_bwd_aux_buffer_policy(
        32768,
        3,
        use_localcta_runtime=False,
        has_batched_accum=True,
        has_bf16_transpose=True,
    )

    assert policy == {
        "D_list": False,
        "dW_T": True,
        "grad_w_materialized": True,
        "gw_list": False,
        "dy_cat": False,
        "normed": False,
        "inv_rms_bf16": False,
    }


def test_qkv_scale_backoff_reuses_logical_subsequence_without_advancing_primary(
    monkeypatch,
) -> None:
    tk_gemm = _load(monkeypatch)

    class _Quantizer:
        scale = 1493.0

        def tk_get_global_scale_num(self):
            return self.scale

        def tk_set_global_scale_num(self, value):
            self.scale = value

    seen_subsequences = []

    def fake_package(*_args, persistent_rng_state=None, **_kwargs):
        seen_subsequences.append(int(persistent_rng_state[1].item()))
        persistent_rng_state[1] += 1 << 32
        return {"underflow": len(seen_subsequences) == 1}

    monkeypatch.setattr(
        tk_gemm, "get_tk_qkv_localcta_scale_backoff_values", lambda: (1000.0, 500.0)
    )
    monkeypatch.setattr(tk_gemm, "_localcta_grouped_k_dgrad_package", fake_package)
    monkeypatch.setattr(
        tk_gemm,
        "_localcta_qkv_package_underflow_details",
        lambda package: package,
    )
    primary = torch.tensor([7, 17 + (1 << 32)], dtype=torch.int64)

    package, info = tk_gemm._try_localcta_qkv_scale_backoff_package(
        _Quantizer(),
        (object(), object(), object()),
        (1, 1, 1),
        persistent_rng_state=primary,
    )

    assert package == {"underflow": False}
    assert info["taken"]
    assert seen_subsequences == [17, 17]
    assert primary.tolist() == [7, 17 + (1 << 32)]


def test_bf16_dgrad_keeps_only_its_required_aux_scratch(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setattr(tk_gemm, "_use_plain_tk_small_m_qkv_dgrad_eager", lambda *_: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_bf16_dgrad", lambda: True)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_bf16_wgrad", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_plain_batched_accum_dgrad", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_cached_return_transpose", lambda: False)
    monkeypatch.setattr(tk_gemm, "_debug_qkv_capture_path", lambda: "")
    monkeypatch.delenv("USE_TK_DEBUG_QKV_DGRAD_REF", raising=False)
    monkeypatch.delenv("USE_TK_DEBUG_QKV_WGRAD_REF", raising=False)

    bf16_dgrad = tk_gemm._qkv_fused_bwd_aux_buffer_policy(
        32768,
        3,
        use_localcta_runtime=False,
    )
    assert bf16_dgrad == {
        "D_list": False,
        "dW_T": True,
        "grad_w_materialized": False,
        "gw_list": False,
        "dy_cat": True,
        "normed": False,
        "inv_rms_bf16": False,
    }


def test_localcta_direct_path_skips_all_fallback_aux_scratch(monkeypatch) -> None:
    tk_gemm = _load(monkeypatch)
    monkeypatch.setattr(tk_gemm, "_use_plain_tk_small_m_qkv_dgrad_eager", lambda *_: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_bf16_dgrad", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_bf16_wgrad", lambda: False)
    monkeypatch.setattr(tk_gemm, "get_tk_localcta_variant", lambda: "v4")
    monkeypatch.setattr(tk_gemm, "use_tk_localcta_v4_sg_direct_consumers", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_localcta_v4_fast_qkv_split_wgrad", lambda: False)
    monkeypatch.setattr(
        tk_gemm,
        "use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout",
        lambda: True,
    )
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_localcta_consistent_nofold_operands", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_bf16_underflow_rescue", lambda: False)
    monkeypatch.setattr(tk_gemm, "use_tk_qkv_cached_return_transpose", lambda: True)
    monkeypatch.setattr(tk_gemm, "_debug_qkv_capture_path", lambda: "")
    monkeypatch.delenv("USE_TK_DEBUG_QKV_DGRAD_REF", raising=False)
    monkeypatch.delenv("USE_TK_DEBUG_QKV_WGRAD_REF", raising=False)

    localcta = tk_gemm._qkv_fused_bwd_aux_buffer_policy(
        32768,
        3,
        use_localcta_runtime=True,
        has_batched_accum=True,
        has_bf16_transpose=True,
    )

    assert not any(localcta.values())
