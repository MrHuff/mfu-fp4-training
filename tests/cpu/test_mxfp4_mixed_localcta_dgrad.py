from __future__ import annotations

from types import SimpleNamespace
import inspect

import pytest
import torch

from low_bits_training.quantization import mxfp4_fused_linear as mixed
from low_bits_training.quantization import tk_gemm


def _set_exact_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "MXFP4_USE_LOCALCTA_DGRAD": "1",
        "FP4_MATMUL_ROOT": "/tmp/pinned-fp4-matmul",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_VARIANT": "v4",
        "USE_TK_LOCALCTA_V3_CONTRACT": "outerscale",
        "MXFP4_BACKEND_VERSION": "v4",
        "MXFP4_USE_2D_WEIGHT_QUANT": "1",
        "MXFP4_SKIP_FUSED_FFN": "0",
        "MXFP4_USE_WO_ATTN_LAYOUT": "0",
        "MXFP4_USE_WO_NHSD_QUANT": "0",
        "USE_FP4_CONVERT_OUTPUT_HEAD": "0",
        "FP4_KEEP_TAIL_BF16_LINEAR_COUNT": "0",
        "FP4_KEEP_LAST_N_LAYERS_BF16": "0",
        "FP4_KEEP_LAST_N_FFNS_BF16": "0",
        "MXFP4_USE_RHT": "1",
        "MXFP4_RHT_TE_STYLE": "1",
        "MXFP4_RHT_ACTIVATION": "1",
        "MXFP4_RHT_GRAD": "1",
        "MXFP4_RHT_WEIGHT": "0",
        "MXFP4_RHT_AXES": "col",
        "MXFP4_RHT_BLOCK_SIZE": "32",
        "MXFP4_RHT_RANDOM_SIGN_MASK": "1",
        "MXFP4_USE_STOCHASTIC_ROUNDING": "0",
        "MXFP4_SR_ACTIVATION": "0",
        "MXFP4_SR_GRAD": "1",
        "MXFP4_SR_WEIGHT": "0",
        "MXFP4_GRAD_SR_AXES": "row",
        "MXFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "MXFP4_SCALE_SR_ACTIVATION": "0",
        "MXFP4_SCALE_SR_GRAD": "0",
        "MXFP4_SCALE_SR_WEIGHT": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _capabilities() -> dict[str, object]:
    return {
        "abi_version": 1,
        "grad_coordinate_mode": "explicit_seed_subsequence",
        "grad_mx_col_rht": "block32_fixed_0x2817",
        "mxfp4_rht_block_size": 32,
        "mxfp4_rht_sign_contract": "fixed_0x2817_per_h16_half",
        "grad_localcta_row_sr": True,
        "grad_scale_sr": False,
        "localcta_encode_mode": "encode_centric",
        "weight_mx_2d": True,
        "weight_localcta_2d": True,
        "prepared_outer_sg": True,
        "localcta_sg_contract": "outer",
        "min_alignment": 256,
        "single_bf16_tile_load": True,
        "runtime_advances_rng": False,
        "split2_grad_one_coordinate": True,
        "split2_dgrad_onepass_outer_sg": True,
        "split2_row_outer_sg": "per_arm",
        "split2_layout": (
            "logical_dim1_concat_per_arm_outer_no_bf16_materialization"
        ),
    }


def test_mixed_route_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MXFP4_USE_LOCALCTA_DGRAD", raising=False)
    assert not mixed.use_mxfp4_localcta_dgrad()
    assert mixed.mxfp4_dgrad_route_identity() == "mxfp4_native_dgrad"


def test_exact_recipe_contract_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exact_recipe(monkeypatch)
    mixed._validate_mxfp4_localcta_dgrad_contract(require_runtime=False)
    assert mixed._mxfp4_data_sr_for_role("grad")
    assert mixed._mxfp4_oriented_grad_data_sr("grad") == "row"
    assert (
        mixed.mxfp4_dgrad_route_identity()
        == "mxfp4_fixed_h32_col_localcta_row_sr_dgrad_v2"
    )


def test_contract_rejects_unsigned_rht_or_weight_rht(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    monkeypatch.setenv("MXFP4_RHT_RANDOM_SIGN_MASK", "0")
    monkeypatch.setenv("MXFP4_RHT_WEIGHT", "1")
    with pytest.raises(RuntimeError, match="deterministic 0x2817") as excinfo:
        mixed._validate_mxfp4_localcta_dgrad_contract(require_runtime=False)
    assert "MXFP4_RHT_WEIGHT must be disabled" in str(excinfo.value)


def test_contract_rejects_tilegrid_localcta_scale_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    monkeypatch.setenv("USE_TK_LOCALCTA_V3_CONTRACT", "tilegrid256")
    with pytest.raises(RuntimeError, match="outer/outerscale"):
        mixed._validate_mxfp4_localcta_dgrad_contract(require_runtime=False)


def test_contract_rejects_route_bypasses(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exact_recipe(monkeypatch)
    monkeypatch.setenv("MXFP4_SKIP_FUSED_FFN", "1")
    monkeypatch.setenv("MXFP4_USE_WO_NHSD_QUANT", "1")
    monkeypatch.setenv("FP4_KEEP_LAST_N_LAYERS_BF16", "4")
    with pytest.raises(RuntimeError) as excinfo:
        mixed._validate_mxfp4_localcta_dgrad_contract(require_runtime=False)
    message = str(excinfo.value)
    assert "MXFP4_SKIP_FUSED_FFN must be disabled" in message
    assert "unsupported NHSD/direct Wo route" in message
    assert "FP4_KEEP_LAST_N_LAYERS_BF16 must be zero" in message


def test_mixed_route_rejects_native_or_bf16_shape_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    with pytest.raises(RuntimeError, match="has no FFN fallback"):
        mixed._require_mixed_localcta_supported_path("FFN", False)
    mixed._require_mixed_localcta_supported_path("FFN", True)


def test_mixed_ffn_state_forces_lazy_without_changing_global_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MXFP4_USE_LAZY_FFN_BWD_STATE", raising=False)
    monkeypatch.delenv("MXFP4_USE_BWD_STATE_CACHE", raising=False)
    allocations: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_empty(*shape: object, **kwargs: object) -> object:
        allocations.append((shape, kwargs))
        return object()

    monkeypatch.setattr(mixed.torch, "empty", record_empty)
    state = mixed._get_mxfp4_ffn_bwd_state(
        32768,
        4096,
        14336,
        torch.device("cpu"),
        force_lazy=True,
    )
    assert isinstance(state, mixed._LazyMXFP4FFNBwdState)
    assert allocations == []

    # These are the only reusable BF16 outputs indexed by the mixed branch.
    mixed_keys = {
        "dh",
        "dh1",
        "dh3",
        "grad_w2",
        "dx0",
        "grad_w1",
        "grad_w3",
    }
    for key in mixed_keys:
        state[key]
    assert set(state) == mixed_keys
    assert len(allocations) == len(mixed_keys)
    assert not any("split2" in key for key in state)
    assert "dx1" not in state


def test_native_ffn_state_keeps_eager_default_when_lazy_env_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MXFP4_USE_LAZY_FFN_BWD_STATE", raising=False)
    monkeypatch.delenv("MXFP4_USE_BWD_STATE_CACHE", raising=False)
    allocations: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_empty(*shape: object, **kwargs: object) -> object:
        allocations.append((shape, kwargs))
        return object()

    monkeypatch.setattr(mixed.torch, "empty", record_empty)
    state = mixed._get_mxfp4_ffn_bwd_state(
        256,
        128,
        256,
        torch.device("cpu"),
    )
    assert not isinstance(state, mixed._LazyMXFP4FFNBwdState)
    assert set(state) == {
        "dh",
        "dh1",
        "dh3",
        "split2_row_fp4",
        "split2_row_sc",
        "split2_col_fp4",
        "split2_col_sc",
        "fused_split2_row_fp4",
        "fused_split2_row_sc",
        "fused_split2_col_fp4",
        "fused_split2_col_sc",
        "grad_w2",
        "dx0",
        "dx1",
        "grad_w1",
        "grad_w3",
    }
    assert len(allocations) == len(state) == 16


def test_production_split2_duplicate_memory_contract_is_exact() -> None:
    m, h = 32768, 14336
    split2_family_bytes = sum(
        (
            m * h,
            (m // 128) * ((2 * h) // 128) * 32 * 16,
            (2 * h) * (m // 2),
            ((2 * h) // 128) * (m // 128) * 32 * 16,
            2 * m * (h // 2),
            2 * (m // 128) * (h // 128) * 32 * 16,
            2 * h * (m // 2),
            2 * (h // 128) * (m // 128) * 32 * 16,
        )
    )
    assert split2_family_bytes == 1_996_488_704
    assert split2_family_bytes // (1024 * 1024) == 1904


def test_contract_rejects_missing_runtime_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    incomplete = _capabilities()
    incomplete["weight_localcta_2d"] = False
    monkeypatch.setattr(
        mixed,
        "tk_mixed_mx_localcta_quant_capabilities",
        lambda: incomplete,
    )
    with pytest.raises(RuntimeError, match="capability mismatch"):
        mixed._validate_mxfp4_localcta_dgrad_contract()


def test_grad_fused_producer_reuses_one_mx_logical_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    monkeypatch.setattr(
        mixed,
        "tk_mixed_mx_localcta_quant_capabilities",
        _capabilities,
    )
    calls: list[tuple[object, ...]] = []

    def alloc(m: int, n: int, device: torch.device):
        assert (m, n, device.type) == (256, 256, "cpu")
        return tuple(torch.empty(1) for _ in range(6))

    def launch(*args):
        assert len(args) == 9  # input + six buffers + seed + subsequence
        calls.append(args)

    module = SimpleNamespace(
        tk_mixed_grad_localcta_row_mx_col_alloc=alloc,
        tk_mixed_grad_localcta_row_mx_col_launch_inplace=launch,
    )
    monkeypatch.setattr(mixed, "_get_tk_mixed_mx_localcta_quant", lambda: module)
    reservations: list[tuple[str, str | None]] = []

    def reserve(role: str, producer_key: str | None = None):
        reservations.append((role, producer_key))
        return {
            "data_stochastic_rounding": True,
            "scale_stochastic_rounding": False,
            "rng_seed": 91,
            "rng_subsequence": 123 << 32,
        }

    monkeypatch.setattr(mixed, "_mxfp4_opt_kwargs", reserve)
    carrier = mixed._quantize_mixed_grad_dy_bf16(
        torch.zeros((256, 256), dtype=torch.bfloat16),
        producer_key="layers.31.attention:wo:sr:wo_grad",
    )
    assert reservations == [
        ("grad", "layers.31.attention:wo:sr:wo_grad")
    ]
    assert len(calls) == 1
    assert calls[0][-2:] == (91, 123 << 32)
    assert carrier.col_fp4 is carrier.mx_col_fp4
    assert carrier.col_sc is carrier.mx_col_sc
    assert len(carrier.keepalive) == 1


def test_weight_fused_producer_and_localcta_dgrad_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    monkeypatch.setattr(
        mixed,
        "tk_mixed_mx_localcta_quant_capabilities",
        _capabilities,
    )
    launched: list[tuple[object, ...]] = []

    def alloc(n: int, k: int, device: torch.device):
        assert (n, k, device.type) == (256, 256, "cpu")
        return tuple(torch.empty(1) for _ in range(6))

    def launch(*args):
        assert len(args) == 7  # input + six buffers
        launched.append(args)

    module = SimpleNamespace(
        tk_mixed_weight_mx_row_localcta_col_alloc=alloc,
        tk_mixed_weight_mx_row_localcta_col_launch_inplace=launch,
    )
    monkeypatch.setattr(mixed, "_get_tk_mixed_mx_localcta_quant", lambda: module)
    weight = mixed._quantize_mixed_weight_bf16(
        torch.zeros((256, 256), dtype=torch.bfloat16)
    )
    assert len(launched) == 1
    assert weight.row_fp4 is weight.mx_row_fp4
    assert weight.row_sc is weight.mx_row_sc
    assert not hasattr(weight, "col_fp4")
    assert not hasattr(weight, "col_sc")
    assert len(weight.keepalive) == 1

    grad = mixed._MixedMXLocalCTAGradCarrier(
        *(torch.empty(1) for _ in range(5)),
        shape=(256, 256),
    )
    dispatch: list[tuple[object, ...]] = []
    sentinel = torch.empty(1)
    monkeypatch.setattr(
        mixed,
        "tk_mixed_localcta_dgrad",
        lambda *args: dispatch.append(args) or sentinel,
    )
    assert mixed._mixed_localcta_dgrad(grad, weight) is sentinel
    assert len(dispatch) == 1
    assert dispatch[0][0] is grad.local_row_fp4
    assert dispatch[0][3] is weight.local_col_fp4


def test_forward_save_selects_explicit_weight_column_format() -> None:
    values = [torch.empty(1) for _ in range(5)]
    weight = mixed._MixedMXLocalCTAWeightCarrier(
        *values,
        shape=(256, 256),
    )
    fp4, sc = mixed._mxfp4_weight_backward_col(
        weight,
        mixed_localcta_dgrad=True,
    )
    assert fp4 is weight.local_col_fp4
    assert sc is weight.local_col_sc

    native = SimpleNamespace(col_fp4=torch.empty(1), col_sc=torch.empty(1))
    fp4, sc = mixed._mxfp4_weight_backward_col(
        native,
        mixed_localcta_dgrad=False,
    )
    assert fp4 is native.col_fp4
    assert sc is native.col_sc
    with pytest.raises(RuntimeError, match="native MX route"):
        mixed._mxfp4_weight_backward_col(
            weight,
            mixed_localcta_dgrad=False,
        )


def test_split2_fused_producer_reuses_one_logical_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    monkeypatch.setattr(
        mixed,
        "tk_mixed_mx_localcta_quant_capabilities",
        _capabilities,
    )
    calls: list[tuple[object, ...]] = []

    def alloc(m: int, h: int, device: torch.device):
        assert (m, h, device.type) == (256, 256, "cpu")
        return tuple(torch.empty(1) for _ in range(7))

    def launch(*args):
        assert len(args) == 11  # two inputs + seven buffers + one coordinate
        calls.append(args)

    module = SimpleNamespace(
        tk_mixed_split2_grad_localcta_row_mx_col_alloc=alloc,
        tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace=launch,
    )
    monkeypatch.setattr(mixed, "_get_tk_mixed_mx_localcta_quant", lambda: module)
    reservations: list[tuple[str, str | None]] = []

    def reserve(role: str, producer_key: str | None = None):
        reservations.append((role, producer_key))
        return {
            "data_stochastic_rounding": True,
            "scale_stochastic_rounding": False,
            "rng_seed": 17,
            "rng_subsequence": 29 << 32,
        }

    monkeypatch.setattr(mixed, "_mxfp4_opt_kwargs", reserve)
    grad = torch.zeros((256, 256), dtype=torch.bfloat16)
    carrier = mixed._quantize_mixed_split2_grad_bf16(
        grad,
        grad.clone(),
        producer_key="layers.31.feed_forward:sr:ffn_deriv",
    )
    assert reservations == [
        ("grad", "layers.31.feed_forward:sr:ffn_deriv")
    ]
    assert len(calls) == 1
    assert calls[0][-2:] == (17, 29 << 32)
    assert carrier.shape == (256, 512)
    assert carrier.local_row_sg0 is calls[0][4]
    assert carrier.local_row_sg1 is calls[0][5]
    assert carrier.local_row_sg0 is not carrier.local_row_sg1


def test_split2_localcta_dgrad_dispatches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grad = mixed._MixedMXLocalCTASplit2GradCarrier(
        *(torch.empty(1) for _ in range(6)),
        shape=(256, 512),
    )
    weight0 = mixed._MixedMXLocalCTAWeightCarrier(
        *(torch.empty(1) for _ in range(5)),
        shape=(256, 256),
    )
    weight1 = mixed._MixedMXLocalCTAWeightCarrier(
        *(torch.empty(1) for _ in range(5)),
        shape=(256, 256),
    )
    calls: list[tuple[object, ...]] = []
    sentinel = torch.empty(1)
    monkeypatch.setattr(
        mixed,
        "tk_mixed_localcta_split2_dgrad",
        lambda *args: calls.append(args) or sentinel,
    )
    assert (
        mixed._mixed_localcta_split2_dgrad(grad, weight0, weight1)
        is sentinel
    )
    assert len(calls) == 1
    assert calls[0][0] is grad.local_row_fp4
    assert calls[0][2] is grad.local_row_sg0
    assert calls[0][3] is grad.local_row_sg1
    assert calls[0][2] is not calls[0][3]
    assert calls[0][4] is weight0.local_col_fp4
    assert calls[0][7] is weight1.local_col_fp4


def test_direct_split2_consumer_preserves_distinct_outer_sg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    direct = SimpleNamespace(
        nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg=(
            lambda *args: calls.append(args)
        )
    )
    monkeypatch.setattr(
        tk_gemm,
        "_get_tk_mixed_localcta_direct",
        lambda: direct,
    )
    a_fp4 = torch.empty((256, 256), dtype=torch.float4_e2m1fn_x2)
    a_sc = torch.empty((2, 8, 1), dtype=torch.uint8)
    sg0 = torch.empty((1, 1))
    sg1 = torch.empty((1, 1))
    b0_fp4 = torch.empty((256, 128), dtype=torch.float4_e2m1fn_x2)
    b1_fp4 = torch.empty((256, 128), dtype=torch.float4_e2m1fn_x2)
    b0_sc = torch.empty(1)
    b1_sc = torch.empty(1)
    b0_sg = torch.full((2, 2), 0.25, dtype=torch.float32)
    b1_sg = torch.full((2, 2), 0.5, dtype=torch.float32)
    prepare_calls: list[tuple[object, ...]] = []
    original_prepare = (
        tk_gemm._prepare_mixed_localcta_common_weight_sg_for_split2_direct
    )
    monkeypatch.setattr(
        tk_gemm,
        "_prepare_mixed_localcta_common_weight_sg_for_split2_direct",
        lambda *args: prepare_calls.append(args) or original_prepare(*args),
    )
    out = torch.empty((256, 256), dtype=torch.bfloat16)
    assert (
        tk_gemm.tk_mixed_localcta_split2_dgrad(
            a_fp4,
            a_sc,
            sg0,
            sg1,
            b0_fp4,
            b0_sc,
            b0_sg,
            b1_fp4,
            b1_sc,
            b1_sg,
            out,
        )
        is out
    )
    assert len(calls) == 1
    assert calls[0][2][0] is sg0
    assert calls[0][2][1] is sg1
    assert calls[0][2][0] is not calls[0][2][1]
    assert prepare_calls == [
        (b0_sg, b0_fp4),
        (b1_sg, b1_fp4),
    ]
    assert torch.equal(calls[0][7][0], torch.tensor([[0.25]]))
    assert torch.equal(calls[0][7][1], torch.tensor([[0.5]]))
    assert (
        calls[0][7][0].untyped_storage().data_ptr()
        == b0_sg.untyped_storage().data_ptr()
    )
    assert (
        calls[0][7][1].untyped_storage().data_ptr()
        == b1_sg.untyped_storage().data_ptr()
    )


def test_mixed_split2_weight_sg_zero_copy_contract_rejects_other_carriers() -> None:
    packed = torch.empty((256, 128), dtype=torch.float4_e2m1fn_x2)
    with pytest.raises(RuntimeError, match="common-broadcast"):
        tk_gemm._prepare_mixed_localcta_common_weight_sg_for_split2_direct(
            torch.empty((1, 1), dtype=torch.float32), packed
        )
    with pytest.raises(RuntimeError, match="common-broadcast"):
        tk_gemm._prepare_mixed_localcta_common_weight_sg_for_split2_direct(
            torch.empty((2, 2), dtype=torch.bfloat16), packed
        )


def test_mixed_direct_consumer_rejects_unpinned_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = SimpleNamespace(
        _extension_path="/tmp/stale-runtime/nvfp4_localcta_gemm.so"
    )
    monkeypatch.setattr(tk_gemm, "_get_tk_localcta_direct", lambda: direct)
    monkeypatch.setattr(
        tk_gemm,
        "_fp4_matmul_root",
        lambda: "/tmp/pinned-fp4-matmul",
    )
    with pytest.raises(RuntimeError, match="outside the pinned"):
        tk_gemm._get_tk_mixed_localcta_direct()


def test_production_qkv_uses_one_combined_mixed_producer() -> None:
    source = inspect.getsource(mixed._FusedQKVFunction_MXFP4_TK)
    assert "_quantize_mixed_weight_bf16(w_qkv_bf16)" in source
    assert "_quantize_mixed_grad_dy_bf16" in source
    assert "torch.cat([gq, gk, gv], dim=1)" in source
    assert "dx_normed = _mixed_localcta_dgrad" in source
    assert "True\n            if ctx.mixed_localcta_dgrad" in source
    assert source.index("if getattr(ctx, \"mixed_localcta_dgrad\", False):\n                gall_q") < source.index(
        "elif use_split3_grad_fast:"
    )


def test_mixed_qkv_preempts_enabled_legacy_split3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_recipe(monkeypatch)
    monkeypatch.setenv("MXFP4_USE_SPLIT3_QKV_QUANT", "1")
    monkeypatch.setattr(mixed, "_mxfp4_oriented_grad_data_sr", lambda role: None)
    assert not mixed._use_mxfp4_qkv_split3_grad_fast_for_route(True)


def test_production_ffn_uses_two_mixed_coordinates_and_onepass_dgrad() -> None:
    source = inspect.getsource(mixed._FusedFFNFunctionV2_MXFP4_TK)
    assert 'force_lazy=getattr(ctx, "mixed_localcta_dgrad", False)' in source
    assert "_quantize_mixed_grad_dy_bf16" in source
    assert "producer_key=ffn_w2_sr_key" in source
    assert "_quantize_mixed_split2_grad_bf16" in source
    assert "producer_key=ffn_deriv_sr_key" in source
    assert "dx_normed = _mixed_localcta_split2_dgrad" in source
