import types

import pytest
import torch

from low_bits_training.cce import backend as cce_backend


def _backend():
    return cce_backend._NVFP4Backend(
        ignore_index=-100,
        implementation="v4",
        quant_mode="enc",
    )


def test_mixed_head_routes_mxfp8_row_and_mxfp4_column(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    x_row = object()
    x_col = object()
    calls = {}

    def quantize_x(value):
        calls["producer"] = value
        return x_row, x_col

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        calls["consumer"] = (
            value,
            value_q,
            value_col_q,
            head_weight,
            targets,
            kwargs,
        )
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        MXFP8Quantized=object,
        quantize_mxfp8_row_mxfp4_col=quantize_x,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    loss = _backend().training_loss(hidden, weight, labels)

    assert calls["producer"].data_ptr() == hidden.data_ptr()
    assert calls["producer"].requires_grad is False
    consumed = calls["consumer"]
    assert consumed[0].data_ptr() == hidden.data_ptr()
    assert consumed[1] is x_row
    assert consumed[2] is x_col
    assert consumed[3].data_ptr() == weight.data_ptr()
    assert consumed[4].data_ptr() == labels.data_ptr()
    assert consumed[5] == {"ignore_index": -100, "encode_centric": True}
    assert loss.item() == 0.0


def test_mixed_head_routes_mxfp8_row_and_native_nvfp4_column(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    x_row = object()
    x_col = object()
    calls = {}

    def quantize_x(value, *, encode_centric, four_over_six_mae):
        calls["producer"] = (value, encode_centric, four_over_six_mae)
        return x_row, x_col

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        calls["consumer"] = (
            value,
            value_q,
            value_col_q,
            head_weight,
            targets,
            kwargs,
        )
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        MXFP8Quantized=object,
        quantize_mxfp8_row_nvfp4_col_localcta_v4=quantize_x,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    loss = _backend().training_loss(hidden, weight, labels)

    produced, encode_centric, four_over_six_mae = calls["producer"]
    assert produced.data_ptr() == hidden.data_ptr()
    assert produced.requires_grad is False
    assert encode_centric is True
    assert four_over_six_mae is True
    consumed = calls["consumer"]
    assert consumed[0].data_ptr() == hidden.data_ptr()
    assert consumed[1] is x_row
    assert consumed[2] is x_col
    assert consumed[3].data_ptr() == weight.data_ptr()
    assert consumed[4].data_ptr() == labels.data_ptr()
    assert consumed[5] == {"ignore_index": -100, "encode_centric": True}
    assert loss.item() == 0.0


def test_direct_fp8_head_routes_fp8_row_and_mxfp4_column(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    x_row = object()
    x_col = object()
    calls = {}

    def quantize_x(value, *, role):
        calls["producer"] = (value, role)
        return x_row, x_col

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        calls["consumer"] = (
            value,
            value_q,
            value_col_q,
            head_weight,
            targets,
            kwargs,
        )
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        DirectFP8Quantized=object,
        quantize_direct_fp8_row_mxfp4_col=quantize_x,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    loss = _backend().training_loss(hidden, weight, labels)

    produced, role = calls["producer"]
    assert produced.data_ptr() == hidden.data_ptr()
    assert produced.requires_grad is False
    assert role == "X"
    consumed = calls["consumer"]
    assert consumed[0].data_ptr() == hidden.data_ptr()
    assert consumed[1] is x_row
    assert consumed[2] is x_col
    assert consumed[3].data_ptr() == weight.data_ptr()
    assert consumed[4].data_ptr() == labels.data_ptr()
    assert consumed[5] == {"ignore_index": -100, "encode_centric": True}
    assert loss.item() == 0.0


def test_direct_fp8_weight_quantization_is_step_scoped(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    calls = {"X": 0, "W": 0, "cached": []}

    def quantize(value, *, role):
        calls[role] += 1
        return (f"{role}-row-{calls[role]}", f"{role}-col-{calls[role]}")

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        calls["cached"].append(kwargs["weight_quantized"])
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        DirectFP8Quantized=object,
        quantize_direct_fp8_row_mxfp4_col=quantize,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "1")
    monkeypatch.setenv("FP4_CCE_V4_WEIGHT_QUANT_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)
    backend = _backend()

    backend.training_loss(hidden, weight, labels)
    backend.training_loss(hidden, weight, labels)
    assert calls["X"] == 2
    assert calls["W"] == 1
    assert calls["cached"][0] is calls["cached"][1]

    with torch.no_grad():
        weight.add_(1.0)
    backend.training_loss(hidden, weight, labels)
    assert calls["W"] == 2
    assert calls["cached"][2] != calls["cached"][1]

    backend.invalidate_weight_cache()
    backend.training_loss(hidden, weight, labels)
    assert calls["W"] == 3


def test_direct_fp8_weight_cache_refreshes_when_training_step_changes(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    calls = {"X": 0, "W": 0}

    def quantize(value, *, role):
        calls[role] += 1
        return (f"{role}-row-{calls[role]}", f"{role}-col-{calls[role]}")

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        DirectFP8Quantized=object,
        quantize_direct_fp8_row_mxfp4_col=quantize,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "1")
    monkeypatch.setenv("FP4_CCE_V4_WEIGHT_QUANT_CACHE", "1")
    monkeypatch.setenv("LBT_TRACE_ACTIVE_STEP", "7")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)
    backend = _backend()

    backend.training_loss(hidden, weight, labels)
    backend.training_loss(hidden, weight, labels)
    assert calls["W"] == 1

    # Model the FSDP case where the optimizer updates storage without changing
    # the tensor object's version counter.
    monkeypatch.setenv("LBT_TRACE_ACTIVE_STEP", "8")
    backend.training_loss(hidden, weight, labels)
    assert calls["W"] == 2


def test_direct_fp8_prequantized_x_routes_cached_weight(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    cached_weight = (object(), object())
    calls = {}

    def quantize(value, *, role):
        assert role == "W"
        return cached_weight

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        calls.update(kwargs)
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        quantize_direct_fp8_row_mxfp4_col=quantize,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_WEIGHT_QUANT_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    _backend().training_loss_prequantized_x(
        hidden, object(), object(), weight, labels
    )

    assert calls["weight_quantized"] is cached_weight


def test_mxfp8_weight_quantization_is_reused(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    produced = []
    consumed = []

    def quantize(value):
        result = (object(), object())
        produced.append((value, result))
        return result

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        consumed.append(kwargs["weight_quantized"])
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        MXFP8Quantized=object,
        quantize_mxfp8_row_mxfp4_col=quantize,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "1")
    monkeypatch.setenv("FP4_CCE_V4_WEIGHT_QUANT_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)
    backend = _backend()

    backend.training_loss(hidden, weight, labels)
    backend.training_loss(hidden, weight, labels)

    assert len(produced) == 3  # X twice, W once.
    assert produced[0][0].data_ptr() == hidden.data_ptr()
    assert produced[1][0].data_ptr() == weight.data_ptr()
    assert produced[2][0].data_ptr() == hidden.data_ptr()
    assert consumed[0] is consumed[1]


def test_mxfp8_native_nvfp4_weight_quantization_is_reused(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    calls = []
    consumed = []

    def quantize(value, *, encode_centric, four_over_six_mae):
        result = (object(), object())
        calls.append((value, encode_centric, four_over_six_mae, result))
        return result

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        consumed.append(kwargs["weight_quantized"])
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        MXFP8Quantized=object,
        quantize_mxfp8_row_nvfp4_col_localcta_v4=quantize,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE", "1")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_W_FOUROVERSIX_MAE", "0")
    monkeypatch.setenv("FP4_CCE_V4_WEIGHT_QUANT_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)
    backend = _backend()

    backend.training_loss(hidden, weight, labels)
    backend.training_loss(hidden, weight, labels)

    assert len(calls) == 3
    assert calls[0][0].data_ptr() == hidden.data_ptr()
    assert calls[0][1:3] == (True, True)
    assert calls[1][0].data_ptr() == weight.data_ptr()
    assert calls[1][1:3] == (True, False)
    assert calls[2][0].data_ptr() == hidden.data_ptr()
    assert consumed[0] is consumed[1]


def test_mxfp8_mixed_gcache_caches_mxfp8_weight_column(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    x_calls = []
    weight_calls = []
    consumed = []

    def quantize_x(value, *, encode_centric, four_over_six_mae):
        result = (object(), object())
        x_calls.append((value, encode_centric, four_over_six_mae, result))
        return result

    def quantize_weight(value):
        result = (object(), object())
        weight_calls.append((value, result))
        return result

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        consumed.append(kwargs["weight_quantized"])
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        MXFP8Quantized=object,
        quantize_mxfp8_row_nvfp4_col_localcta_v4=quantize_x,
        quantize_mxfp8_row_and_col_fused=quantize_weight,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE", "1")
    monkeypatch.setenv("FP4_CCE_V4_WEIGHT_QUANT_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)
    backend = _backend()

    backend.training_loss(hidden, weight, labels)
    backend.training_loss(hidden, weight, labels)

    assert len(x_calls) == 2
    assert len(weight_calls) == 1
    assert weight_calls[0][0].data_ptr() == weight.data_ptr()
    assert consumed[0] is consumed[1]
    assert consumed[0] is weight_calls[0][1]


@pytest.mark.parametrize(
    ("forward_env", "quantized_type"),
    [
        ("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "MXFP8Quantized"),
        ("FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD", "DirectFP8Quantized"),
    ],
)
def test_fused_final_norm_producer_reconstructs_fp8_and_mxfp4_types(
    monkeypatch,
    forward_env,
    quantized_type,
):
    class RowQuantized:
        def __init__(self, fp8, sc):
            self.fp8 = fp8
            self.sc = sc

    class ColQuantized:
        def __init__(self, fp4, sc):
            self.fp4 = fp4
            self.sc = sc

    pre_norm = torch.randn(4, 8, dtype=torch.bfloat16)
    gamma = torch.randn(8, dtype=torch.bfloat16)
    row_data = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
    row_sc = torch.ones(4, 1, dtype=torch.uint8)
    col_data = torch.zeros(4, 4, dtype=torch.uint8)
    col_sc = torch.ones(1, 8, dtype=torch.uint8)
    calls = {}

    def quantize(value, weight, epsilon, **kwargs):
        calls["producer"] = (value, weight, epsilon, kwargs)
        normed = value * weight
        row_q = types.SimpleNamespace(fp8=row_data, sc=row_sc)
        col_q = types.SimpleNamespace(fp4=col_data, sc=col_sc)
        inv_rms = torch.ones(value.shape[0], dtype=torch.float32)
        scratch = torch.empty(0)
        return normed, row_q, col_q, inv_rms, scratch

    runtime = types.SimpleNamespace(
        MXFP8Quantized=RowQuantized,
        DirectFP8Quantized=RowQuantized,
        MXFP4Quantized=ColQuantized,
        quantize_mxfp8_norm_row_mxfp4_col_with_output_localcta_v4=quantize,
        quantize_direct_fp8_norm_row_mxfp4_col_with_output_localcta_v4=quantize,
    )
    monkeypatch.setenv(forward_env, "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    normed, x_q, x_col_q = cce_backend._produce_final_norm_x_with_quant(
        pre_norm,
        gamma,
        1e-5,
        _backend(),
    )

    assert torch.equal(normed, pre_norm * gamma)
    assert isinstance(x_q, getattr(runtime, quantized_type))
    assert x_q.fp8 is row_data
    assert x_q.sc is row_sc
    assert isinstance(x_col_q, ColQuantized)
    assert x_col_q.fp4 is col_data
    assert x_col_q.sc is col_sc
    expected_kwargs = (
        {"role": "X"}
        if forward_env == "FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD"
        else {}
    )
    assert calls["producer"][2] == pytest.approx(1e-5)
    assert calls["producer"][3] == expected_kwargs


def test_fused_final_norm_producer_preserves_native_nvfp4_column(monkeypatch):
    class RowQuantized:
        def __init__(self, fp8, sc):
            self.fp8 = fp8
            self.sc = sc

    class ColQuantized:
        def __init__(self, fp4, sc, sg, *, layout=None):
            self.fp4 = fp4
            self.sc = sc
            self.sg = sg
            self.layout = layout

    pre_norm = torch.randn(4, 8, dtype=torch.bfloat16)
    gamma = torch.randn(8, dtype=torch.bfloat16)
    row_data = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
    row_sc = torch.ones(4, 1, dtype=torch.uint8)
    col_data = torch.zeros(4, 4, dtype=torch.uint8)
    col_sc = torch.ones(1, 8, dtype=torch.uint8)
    col_sg = torch.ones(1, 8, dtype=torch.float32)
    calls = {}

    def quantize(value, weight, epsilon, **kwargs):
        calls["producer"] = (value, weight, epsilon, kwargs)
        normed = value * weight
        row_q = RowQuantized(row_data, row_sc)
        col_q = ColQuantized(col_data, col_sc, col_sg, layout="localcta")
        inv_rms = torch.ones(value.shape[0], dtype=torch.float32)
        return normed, row_q, col_q, inv_rms, torch.empty(0)

    runtime = types.SimpleNamespace(
        NVFP4Quantized=ColQuantized,
        MXFP8Quantized=RowQuantized,
        quantize_mxfp8_norm_row_nvfp4_col_with_output_localcta_v4=quantize,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    normed, x_q, x_col_q = cce_backend._produce_final_norm_x_with_quant(
        pre_norm,
        gamma,
        1e-5,
        _backend(),
    )

    assert torch.equal(normed, pre_norm * gamma)
    assert isinstance(x_q, RowQuantized)
    assert x_q.fp8 is row_data
    assert isinstance(x_col_q, ColQuantized)
    assert x_col_q.fp4 is col_data
    assert x_col_q.sc is col_sc
    assert x_col_q.sg is col_sg
    assert x_col_q.layout == "localcta"
    assert calls["producer"][3] == {
        "encode_centric": True,
        "four_over_six_mae": True,
    }


def test_fused_final_norm_producer_carries_mxfp8_dweight_column(monkeypatch):
    class MXFP8Quantized:
        def __init__(self, fp8, sc):
            self.fp8 = fp8
            self.sc = sc

    pre_norm = torch.randn(4, 8, dtype=torch.bfloat16)
    gamma = torch.randn(8, dtype=torch.bfloat16)
    row_data = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
    row_sc = torch.ones(4, 1, dtype=torch.uint8)
    col_data = torch.zeros(8, 4, dtype=torch.float8_e4m3fn)
    col_sc = torch.ones(8, 1, dtype=torch.uint8)
    calls = {}

    def quantize(value, weight, epsilon, **kwargs):
        calls["producer"] = (value, weight, epsilon, kwargs)
        normed = value * weight
        row_q = MXFP8Quantized(row_data, row_sc)
        col_q = MXFP8Quantized(col_data, col_sc)
        inv_rms = torch.ones(value.shape[0], dtype=torch.float32)
        return normed, row_q, col_q, inv_rms, torch.empty(0)

    runtime = types.SimpleNamespace(
        NVFP4Quantized=None,
        MXFP8Quantized=MXFP8Quantized,
        quantize_mxfp8_norm_row_nvfp4_col_with_output_localcta_v4=quantize,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MIXED_DW_MXFP8_COLS", "1")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE", "1")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    normed, x_q, x_col_q = cce_backend._produce_final_norm_x_with_quant(
        pre_norm,
        gamma,
        1e-5,
        _backend(),
    )

    assert torch.equal(normed, pre_norm * gamma)
    assert isinstance(x_q, MXFP8Quantized)
    assert x_q.fp8 is row_data
    assert x_q.sc is row_sc
    assert isinstance(x_col_q, MXFP8Quantized)
    assert x_col_q.fp8 is col_data
    assert x_col_q.sc is col_sc
    assert calls["producer"][3] == {
        "encode_centric": True,
        "four_over_six_mae": True,
    }


def test_mxfp4_forward_routes_native_v5_column(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    x_row = object()
    x_col = object()
    calls = {}

    def quantize_x(value, *, encode_centric, role):
        calls["producer"] = (value, encode_centric, role)
        return x_row, x_col

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        calls["consumer"] = (
            value,
            value_q,
            value_col_q,
            head_weight,
            targets,
            kwargs,
        )
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        MXFP4Quantized=object,
        quantize_mxfp4_row_nvfp4_col_v5=quantize_x,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP4_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "0")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_G_CACHE", "0")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    loss = _backend().training_loss(hidden, weight, labels)

    produced, encode_centric, role = calls["producer"]
    assert produced.data_ptr() == hidden.data_ptr()
    assert produced.requires_grad is False
    assert encode_centric is True
    assert role == "X"
    consumed = calls["consumer"]
    assert consumed[0].data_ptr() == hidden.data_ptr()
    assert consumed[1] is x_row
    assert consumed[2] is x_col
    assert consumed[3].data_ptr() == weight.data_ptr()
    assert consumed[4].data_ptr() == labels.data_ptr()
    assert consumed[5] == {"ignore_index": -100, "encode_centric": True}
    assert loss.item() == 0.0


def test_mxfp4_forward_routes_mxfp4_g_cache_column(monkeypatch):
    hidden = torch.randn(256, 128, dtype=torch.bfloat16)
    weight = torch.randn(512, 128, dtype=torch.bfloat16)
    labels = torch.randint(512, (256,), dtype=torch.int64)
    x_row = object()
    x_col = object()
    calls = {}

    def quantize_x(value, *, mode, role):
        calls["producer"] = (value, mode, role)
        return x_row, x_col

    def training_loss(value, value_q, value_col_q, head_weight, targets, **kwargs):
        calls["consumer"] = (
            value,
            value_q,
            value_col_q,
            head_weight,
            targets,
            kwargs,
        )
        return value.float().sum() * 0.0

    runtime = types.SimpleNamespace(
        MXFP4Quantized=object,
        quantize_mxfp4_row_and_col_tk=quantize_x,
        nvfp4_cce_tk_v4_pcache_prequantized_x=training_loss,
    )
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP4_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP4_G_CACHE", "1")
    monkeypatch.setenv("FP4_CCE_V4_MXFP8_G_CACHE", "0")
    monkeypatch.setattr(cce_backend, "_load_fp4_cce_tk_v4", lambda: runtime)

    loss = _backend().training_loss(hidden, weight, labels)

    produced, mode, role = calls["producer"]
    assert produced.data_ptr() == hidden.data_ptr()
    assert produced.requires_grad is False
    assert mode == 1
    assert role == "X"
    consumed = calls["consumer"]
    assert consumed[0].data_ptr() == hidden.data_ptr()
    assert consumed[1] is x_row
    assert consumed[2] is x_col
    assert consumed[3].data_ptr() == weight.data_ptr()
    assert consumed[4].data_ptr() == labels.data_ptr()
    assert consumed[5] == {"ignore_index": -100, "encode_centric": True}
    assert loss.item() == 0.0


def test_mixed_forward_formats_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP8_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_MXFP4_FORWARD", "1")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD", "1")
    monkeypatch.setattr(
        cce_backend,
        "_load_fp4_cce_tk_v4",
        lambda: types.SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _backend().training_loss(
            torch.randn(256, 128, dtype=torch.bfloat16),
            torch.randn(512, 128, dtype=torch.bfloat16),
            torch.randint(512, (256,), dtype=torch.int64),
        )
