from __future__ import annotations

import gc
import weakref

import pytest
import torch


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_GEMM", "1")
    monkeypatch.setenv("USE_TK_QUANT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    from low_bits_training.quantization import fused_te_linear

    return fused_te_linear


class _FakeRegularQuantizer:
    def __init__(self) -> None:
        self.row_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.col_fp4_ptrs: list[int] = []
        self.col_scale_ptrs: list[int] = []
        self.aux_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def tk_quantize_for_gemm(self, _weight, _return_transpose):
        row_fp4 = torch.zeros((256, 128), dtype=torch.uint8)
        row_scale = torch.zeros((256, 16), dtype=torch.uint8)
        col_fp4 = torch.zeros((256, 128), dtype=torch.uint8)
        col_scale = torch.zeros((256, 16), dtype=torch.uint8)
        row_sg = torch.ones(1, dtype=torch.float32)
        col_sg = torch.ones(1, dtype=torch.float32)
        auxiliary = torch.ones(1, dtype=torch.float32)
        self.row_refs.extend((weakref.ref(row_fp4), weakref.ref(row_scale)))
        self.col_fp4_ptrs.append(col_fp4.data_ptr())
        self.col_scale_ptrs.append(col_scale.data_ptr())
        self.aux_refs.append(weakref.ref(auxiliary))
        return (
            row_fp4,
            row_scale,
            col_fp4,
            col_scale,
            row_sg,
            col_sg,
            auxiliary,
        )


def test_decomposed_ffn_quant_drops_copied_row_payloads(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    quantizer = _FakeRegularQuantizer()
    weight = torch.empty((256, 256), dtype=torch.bfloat16)

    result = fused_te_linear._regular_tk_group_quantize_ffn_weights_decomposed(
        quantizer, weight, weight
    )
    gc.collect()

    assert len(result[-1]) == 2
    assert all(ref() is None for ref in quantizer.row_refs)
    assert [tensor.data_ptr() for tensor in result[3]] == quantizer.col_fp4_ptrs
    assert [tensor.data_ptr() for tensor in result[4]] == quantizer.col_scale_ptrs
    assert all(ref() is not None for ref in quantizer.aux_refs)


def test_release_tk_row_storage_preserves_col_payloads(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    quantized = object.__new__(fused_te_linear._TKQuantized)
    row = tuple(torch.ones(8, dtype=torch.uint8) for _ in range(3))
    col = tuple(torch.ones(8, dtype=torch.uint8) for _ in range(3))
    row_chunk_sg = torch.ones(1)
    auxiliary = torch.ones(1)
    row_refs = [weakref.ref(tensor) for tensor in (*row, row_chunk_sg)]
    auxiliary_ref = weakref.ref(auxiliary)
    col_ptrs = [tensor.data_ptr() for tensor in col]
    quantized._tk_row = row
    quantized._tk_col = col
    quantized._tk_row_chunk_sg = row_chunk_sg
    quantized._tk_col_chunk_sg = None
    quantized._keepalive = (auxiliary,)

    del row, row_chunk_sg, auxiliary
    fused_te_linear._release_tk_row_storage(quantized)
    gc.collect()

    assert all(ref() is None for ref in row_refs)
    assert all(tensor.numel() == 0 for tensor in quantized._tk_row)
    assert [tensor.data_ptr() for tensor in quantized._tk_col] == col_ptrs
    assert auxiliary_ref() is None
    assert quantized._keepalive == ()


def test_ffn_h13_operand_requant_is_explicitly_opt_in(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv("USE_TK_FFN_REQUANT_H13_OPERANDS", raising=False)
    assert not fused_te_linear.use_tk_ffn_requant_h13_operands()

    monkeypatch.setenv("USE_TK_FFN_REQUANT_H13_OPERANDS", "1")
    assert fused_te_linear.use_tk_ffn_requant_h13_operands()

    monkeypatch.setenv("USE_TK_FFN_REQUANT_H13_OPERANDS", "false")
    assert not fused_te_linear.use_tk_ffn_requant_h13_operands()


def test_v5_2d_weight_quant_is_explicitly_opt_in(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv("USE_TK_V5_2D_WEIGHT_QUANT", raising=False)
    monkeypatch.delenv("USE_TK_DEBUG_TE_2D_WEIGHT_QUANT", raising=False)
    assert not fused_te_linear.use_tk_v5_2d_weight_quant()

    monkeypatch.setenv("USE_TK_V5_2D_WEIGHT_QUANT", "1")
    assert fused_te_linear.use_tk_v5_2d_weight_quant()

    monkeypatch.setenv("USE_TK_V5_2D_WEIGHT_QUANT", "false")
    monkeypatch.setenv("USE_TK_DEBUG_TE_2D_WEIGHT_QUANT", "1")
    assert not fused_te_linear.use_tk_v5_2d_weight_quant()


def test_v5_2d_weight_quant_accepts_debug_compatibility_alias(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv("USE_TK_V5_2D_WEIGHT_QUANT", raising=False)
    monkeypatch.setenv("USE_TK_DEBUG_TE_2D_WEIGHT_QUANT", "1")

    assert fused_te_linear.use_tk_v5_2d_weight_quant()


def test_native_v5_2d_weight_quant_preserves_auxiliary_storage(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    auxiliary = torch.ones(1)

    class FakeQuantizer:
        @staticmethod
        def tk_quantize_weight_2d(weight):
            rows, cols = weight.shape
            return (
                torch.zeros((rows, cols // 2), dtype=torch.uint8),
                torch.zeros((rows // 128, cols // 64, 512), dtype=torch.uint8),
                torch.zeros((cols, rows // 2), dtype=torch.uint8),
                torch.zeros((cols // 128, rows // 64, 512), dtype=torch.uint8),
                torch.ones(1),
                torch.ones(1),
                auxiliary,
            )

    from low_bits_training.quantization import tk_gemm

    monkeypatch.setattr(tk_gemm, "_get_tk_quant_for_gemm", lambda: FakeQuantizer())
    quantized = fused_te_linear._fast_quantize_v5_2d_weight_swizzled(
        torch.empty((128, 128), dtype=torch.bfloat16)
    )
    result = fused_te_linear._tk_quantized_as_result_tuple(quantized)

    assert result[6] is auxiliary
    assert quantized._keepalive == (auxiliary,)


def test_native_v5_2d_weight_quant_requires_runtime_symbol(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    from low_bits_training.quantization import tk_gemm

    monkeypatch.setattr(tk_gemm, "_get_tk_quant_for_gemm", lambda: object())

    with pytest.raises(RuntimeError, match="tk_quantize_weight_2d"):
        fused_te_linear._fast_quantize_v5_2d_weight_swizzled(
            torch.empty((128, 128), dtype=torch.bfloat16)
        )


def test_localcta_2d_weight_quant_is_explicitly_opt_in(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv("USE_TK_LOCALCTA_2D_WEIGHT_QUANT", raising=False)
    assert not fused_te_linear.use_tk_localcta_2d_weight_quant()

    monkeypatch.setenv("USE_TK_LOCALCTA_2D_WEIGHT_QUANT", "1")
    assert fused_te_linear.use_tk_localcta_2d_weight_quant()

    monkeypatch.setenv("USE_TK_LOCALCTA_2D_WEIGHT_QUANT", "false")
    assert not fused_te_linear.use_tk_localcta_2d_weight_quant()


def test_localcta_2d_weight_quant_compacts_folded_outer_sgs(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    row_sg_workspace = torch.ones((4, 2), dtype=torch.float32)
    col_sg_workspace = torch.ones((2, 4), dtype=torch.float32)

    class FakeQuantizer:
        @staticmethod
        def tk_quantize_weight_2d(weight):
            rows, cols = weight.shape
            return (
                torch.zeros((rows, cols // 2), dtype=torch.uint8),
                torch.zeros((rows // 128, cols // 64, 512), dtype=torch.uint8),
                torch.zeros((cols, rows // 2), dtype=torch.uint8),
                torch.zeros((cols // 128, rows // 64, 512), dtype=torch.uint8),
                row_sg_workspace,
                col_sg_workspace,
            )

    from low_bits_training.quantization import tk_gemm

    monkeypatch.setattr(tk_gemm, "_get_tk_quant_for_gemm", lambda: FakeQuantizer())
    quantized = fused_te_linear._fast_quantize_localcta_2d_weight_swizzled(
        torch.empty((512, 256), dtype=torch.bfloat16)
    )

    assert tuple(quantized._tk_row[2].shape) == (2,)
    assert tuple(quantized._tk_col[2].shape) == (1,)
    assert torch.equal(quantized._tk_row[2], torch.ones(2))
    assert torch.equal(quantized._tk_col[2], torch.ones(1))


def test_localcta_2d_group_contract_expands_orientation_sgs(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    calls = 0

    def fake_quantize(weight):
        nonlocal calls
        calls += 1
        rows, cols = weight.shape
        row_fp4 = torch.zeros((rows, cols // 2), dtype=torch.uint8)
        row_sc = torch.zeros((rows // 128, cols // 64, 512), dtype=torch.uint8)
        col_fp4 = torch.zeros((cols, rows // 2), dtype=torch.uint8)
        col_sc = torch.zeros((cols // 128, rows // 64, 512), dtype=torch.uint8)
        return fused_te_linear._TKQuantized(
            row_fp4,
            row_sc,
            torch.tensor([float(calls)]),
            col_fp4,
            col_sc,
            torch.tensor([float(10 + calls)]),
        )

    monkeypatch.setattr(
        fused_te_linear,
        "_fast_quantize_localcta_2d_weight_swizzled",
        fake_quantize,
    )
    weight = torch.empty((768, 256), dtype=torch.bfloat16)

    result = fused_te_linear._localcta_group_quantize_weights_2d(
        weight,
        [256, 512],
    )

    assert len(result) == 8
    assert tuple(result[0].shape) == (768, 128)
    assert tuple(result[1].shape) == (6, 4, 512)
    assert torch.equal(result[2], torch.tensor([1.0, 2.0, 2.0]))
    assert [tuple(value.shape) for value in result[3]] == [(256, 128), (256, 256)]
    assert [tuple(value.shape) for value in result[4]] == [(2, 4, 512), (2, 8, 512)]
    assert torch.equal(result[5], torch.tensor([11.0, 12.0]))
    assert [value.tolist() for value in result[6]] == [[1.0], [2.0, 2.0]]
    assert [value.tolist() for value in result[7]] == [[11.0], [12.0]]


def test_ffn_h13_activation_requant_is_explicitly_opt_in(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv("USE_TK_FFN_REQUANT_H13_ACTIVATION", raising=False)
    assert not fused_te_linear.use_tk_ffn_requant_h13_activation()

    monkeypatch.setenv("USE_TK_FFN_REQUANT_H13_ACTIVATION", "1")
    assert fused_te_linear.use_tk_ffn_requant_h13_activation()

    monkeypatch.setenv("USE_TK_FFN_REQUANT_H13_ACTIVATION", "false")
    assert not fused_te_linear.use_tk_ffn_requant_h13_activation()


def test_ffn_h_recompute_for_w2_wgrad_is_explicitly_opt_in(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv("USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD", raising=False)
    assert not fused_te_linear.use_tk_ffn_recompute_h_for_w2_wgrad()

    monkeypatch.setenv("USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD", "1")
    assert fused_te_linear.use_tk_ffn_recompute_h_for_w2_wgrad()

    monkeypatch.setenv("USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD", "false")
    assert not fused_te_linear.use_tk_ffn_recompute_h_for_w2_wgrad()


def test_localcta_deriv_w2_wgrad_overlap_is_explicitly_opt_in(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv(
        "USE_TK_LOCALCTA_V4_FFN_DERIV_W2_WGRAD_OVERLAP",
        raising=False,
    )
    assert not fused_te_linear.use_tk_localcta_v4_ffn_deriv_w2_wgrad_overlap()

    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_FFN_DERIV_W2_WGRAD_OVERLAP",
        "1",
    )
    assert fused_te_linear.use_tk_localcta_v4_ffn_deriv_w2_wgrad_overlap()

    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_FFN_DERIV_W2_WGRAD_OVERLAP",
        "false",
    )
    assert not fused_te_linear.use_tk_localcta_v4_ffn_deriv_w2_wgrad_overlap()


def test_localcta_persistent_step_scratch_is_explicitly_opt_in(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    monkeypatch.delenv("USE_TK_LOCALCTA_PERSISTENT_STEP_SCRATCH", raising=False)
    assert not fused_te_linear.use_tk_localcta_persistent_step_scratch()

    monkeypatch.setenv("USE_TK_LOCALCTA_PERSISTENT_STEP_SCRATCH", "1")
    assert fused_te_linear.use_tk_localcta_persistent_step_scratch()

    monkeypatch.setenv("USE_TK_LOCALCTA_PERSISTENT_STEP_SCRATCH", "false")
    assert not fused_te_linear.use_tk_localcta_persistent_step_scratch()


def test_step_cache_clear_can_preserve_localcta_scratch(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    fused_te_linear._ffn_bwd_graph_cache["graph"] = object()
    fused_te_linear._ffn_sb_cache["shared"] = object()
    fused_te_linear._ffn_localcta_fwd_cache["forward"] = object()
    fused_te_linear._ffn_localcta_bwd_cache["backward"] = object()
    fused_te_linear._qkv_full_graph_cache["qkv"] = object()
    fused_te_linear._qkv_bwd_graph_cache["qkv_bwd"] = object()

    monkeypatch.setenv("USE_TK_LOCALCTA_PERSISTENT_STEP_SCRATCH", "1")
    fused_te_linear.clear_fused_fp4_step_caches()

    assert not fused_te_linear._ffn_bwd_graph_cache
    assert not fused_te_linear._ffn_sb_cache
    assert "forward" in fused_te_linear._ffn_localcta_fwd_cache
    assert "backward" in fused_te_linear._ffn_localcta_bwd_cache
    assert not fused_te_linear._qkv_full_graph_cache
    assert not fused_te_linear._qkv_bwd_graph_cache

    monkeypatch.setenv("USE_TK_LOCALCTA_PERSISTENT_STEP_SCRATCH", "0")
    fused_te_linear.clear_fused_fp4_step_caches()
    assert not fused_te_linear._ffn_localcta_fwd_cache
    assert not fused_te_linear._ffn_localcta_bwd_cache


def test_localcta_ffn_scratch_tensors_are_materialized_on_access(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    state = fused_te_linear._LazyFFNScratch(
        torch.device("cpu"),
        {
            "empty": ((2, 3), torch.bfloat16, "empty"),
            "ones": ((2,), torch.float32, "ones"),
            "zeros": ((3,), torch.float32, "zeros"),
        },
    )

    assert not state
    assert state["empty"].shape == (2, 3)
    assert set(state) == {"empty"}
    assert torch.equal(state["ones"], torch.ones(2))
    assert torch.equal(state["zeros"], torch.zeros(3))
    assert state["empty"] is state["empty"]


def test_fused_w2_producer_does_not_materialize_dh_scratch(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    state = fused_te_linear._LazyFFNScratch(
        torch.device("cpu"),
        {"dh": ((2, 3), torch.bfloat16, "empty")},
    )

    assert (
        fused_te_linear._get_ffn_localcta_dh_scratch(
            state,
            fused_producer=True,
        )
        is None
    )
    assert "dh" not in state

    dh = fused_te_linear._get_ffn_localcta_dh_scratch(
        state,
        fused_producer=False,
    )
    assert dh.shape == (2, 3)
    assert state["dh"] is dh


def test_localcta_h13_recompute_buffers_alias_derivative_scratch(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    state = {
        "dh1": torch.empty((2, 3), dtype=torch.bfloat16),
        "dh3": torch.empty((2, 3), dtype=torch.bfloat16),
    }
    monkeypatch.setattr(
        fused_te_linear,
        "_get_ffn_localcta_bwd_state",
        lambda *_args: state,
    )
    monkeypatch.setenv("USE_TK_FFN_RECOMPUTE_H13", "1")
    monkeypatch.setenv("USE_TK_FFN_LOCALCTA_INPLACE_H13_DERIV", "1")

    first = fused_te_linear._get_ffn_localcta_h13_recompute_buffers(
        2, 4, 3, torch.device("cpu")
    )
    second = fused_te_linear._get_ffn_localcta_h13_recompute_buffers(
        2, 4, 3, torch.device("cpu")
    )

    assert first[0] is second[0]
    assert first[1] is second[1]


def test_localcta_inplace_h13_deriv_requires_recompute(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("USE_TK_FFN_LOCALCTA_INPLACE_H13_DERIV", "1")
    monkeypatch.delenv("USE_TK_FFN_RECOMPUTE_H13", raising=False)
    monkeypatch.setattr(
        fused_te_linear,
        "_get_ffn_localcta_bwd_state",
        lambda *_args: {},
    )

    with pytest.raises(RuntimeError, match="requires USE_TK_FFN_RECOMPUTE_H13=1"):
        fused_te_linear._get_ffn_localcta_h13_recompute_buffers(
            2, 4, 3, torch.device("cpu")
        )
