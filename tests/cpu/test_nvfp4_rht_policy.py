from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    from low_bits_training.quantization import fused_te_linear

    return fused_te_linear


def test_unpaired_gradient_rht_is_rejected(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_RHT_GRAD", "1")

    with pytest.raises(RuntimeError, match="rotates dY"):
        fused_te_linear._validate_nvfp4_rht_contract("grad")


def test_sr_only_gradient_policy_is_allowed(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_RHT_GRAD", "0")
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")

    fused_te_linear._validate_nvfp4_rht_contract("grad")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("both", "both"),
        ("row_col", "both"),
        ("dgrad", "row"),
        ("wgrad", "col"),
        ("off", "none"),
    ],
)
def test_gradient_sr_axis_policy(monkeypatch, value: str, expected: str) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", value)

    assert fused_te_linear._nvfp4_grad_sr_axes() == expected


def test_gradient_sr_axis_policy_rejects_unknown_value(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "diagonal")

    with pytest.raises(ValueError, match="NVFP4_GRAD_SR_AXES"):
        fused_te_linear._nvfp4_grad_sr_axes()


def test_activation_rht_contract_is_unchanged(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_RHT_ACTIVATION", "1")

    fused_te_linear._validate_nvfp4_rht_contract("activation")


@pytest.mark.parametrize(
    ("dim", "start", "length"),
    [(1, 2, 4), (0, 1, 2)],
)
def test_packed_fp4_slice_copies_exact_bytes(
    monkeypatch, dim: int, start: int, length: int
) -> None:
    fused_te_linear = _load(monkeypatch)
    source_bytes = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
    source = source_bytes.view(torch.float4_e2m1fn_x2)

    result = fused_te_linear._narrow_packed_fp4_contiguous(
        source, dim, start, length
    )

    expected = source_bytes.narrow(dim, start, length).contiguous()
    assert result.dtype == torch.float4_e2m1fn_x2
    assert result.is_contiguous()
    assert torch.equal(result.view(torch.uint8), expected)
    assert result.stride() == expected.stride()


def test_noncontiguous_packed_fp4_materializes_exact_bytes(monkeypatch) -> None:
    _load(monkeypatch)
    from low_bits_training.quantization import tk_gemm

    source_bytes = torch.arange(48, dtype=torch.uint8).reshape(6, 8)
    source = source_bytes.view(torch.float4_e2m1fn_x2)[:, 1:7]
    assert not source.is_contiguous()

    result = tk_gemm._packed_fp4_contiguous(source)

    expected = source_bytes[:, 1:7].contiguous()
    assert result.dtype == torch.float4_e2m1fn_x2
    assert result.shape == source.shape
    assert result.stride() == expected.stride()
    assert result.is_contiguous()
    assert torch.equal(result.view(torch.uint8), expected)


def test_packed_fp4_slice_rejects_nonpacked_dtype(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    with pytest.raises(RuntimeError, match="float4_e2m1fn_x2"):
        fused_te_linear._narrow_packed_fp4_contiguous(
            torch.zeros((4, 8), dtype=torch.uint8), 1, 2, 4
        )


class _Split3QuantRecorder:
    def __init__(self, row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg):
        self.payload = row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg
        self.args = None

    def tk_quantize_for_gemm_opt(self, *args):
        self.args = args
        return self.payload


def test_localcta_paired_rht_split3_copies_exact_packed_bytes(monkeypatch) -> None:
    _load(monkeypatch)
    from low_bits_training.quantization import tk_gemm

    monkeypatch.setattr(tk_gemm, "get_tk_localcta_variant", lambda: "v4")
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", "0")
    monkeypatch.setenv("NVFP4_SCALE_SR_GRAD", "0")
    monkeypatch.setenv("NVFP4_RHT_GRAD", "1")
    monkeypatch.setenv("NVFP4_RHT_ACTIVATION", "1")
    monkeypatch.setenv("NVFP4_RHT_WEIGHT", "0")
    monkeypatch.setenv("NVFP4_RHT_AXES", "col")
    monkeypatch.setenv("NVFP4_RHT_RANDOM_SIGNS", "1")

    M = 256
    widths = [256, 512, 256]
    total = sum(widths)
    row_bytes = (
        torch.arange(M * (total // 2), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(M, total // 2)
    )
    col_bytes = (
        torch.arange(total * (M // 2), dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(total, M // 2)
    )
    row_fp4 = row_bytes.view(torch.float4_e2m1fn_x2)
    col_fp4 = col_bytes.view(torch.float4_e2m1fn_x2)
    row_sc = torch.zeros((M, total // 64), dtype=torch.uint8)
    col_sc = torch.zeros((total // 128, M), dtype=torch.uint8)
    row_sg = torch.ones((M // 256, 1), dtype=torch.float32)
    col_sg = torch.ones((1, total // 256), dtype=torch.float32)
    recorder = _Split3QuantRecorder(
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg
    )
    grads = [torch.zeros((M, width), dtype=torch.bfloat16) for width in widths]

    package = tk_gemm._localcta_paired_rht_split3_package(
        recorder,
        *grads,
        widths,
        persistent_rng_state=torch.zeros(4, dtype=torch.int64),
    )

    row_offset = col_offset = 0
    for width, row_part, col_part in zip(
        widths, package["fp4_row_list"], package["fp4_col_list"], strict=True
    ):
        expected_row = row_bytes[:, row_offset : row_offset + width // 2].contiguous()
        expected_col = col_bytes[col_offset : col_offset + width].contiguous()
        assert row_part.dtype == col_part.dtype == torch.float4_e2m1fn_x2
        assert row_part.is_contiguous() and col_part.is_contiguous()
        assert row_part.stride() == expected_row.stride()
        assert col_part.stride() == expected_col.stride()
        assert torch.equal(row_part.view(torch.uint8), expected_row)
        assert torch.equal(col_part.view(torch.uint8), expected_col)
        row_offset += width // 2
        col_offset += width


class _NativeSplit3QuantRecorder:
    def __init__(self, result):
        self.result = result
        self.kwargs = None
        self.inputs = None

    def tk_localcta_split3_supports_paired_rht(self):
        return True

    def tk_group_quantize_dim1_split3_for_gemm_paired_rht(
        self, *inputs, **kwargs
    ):
        self.inputs = inputs
        self.kwargs = kwargs
        return self.result


def test_localcta_paired_rht_split3_native_route_has_no_cat_or_split_copy(
    monkeypatch,
) -> None:
    _load(monkeypatch)
    from low_bits_training.quantization import tk_gemm

    monkeypatch.setattr(tk_gemm, "get_tk_localcta_variant", lambda: "v4")
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", "0")
    monkeypatch.setenv("NVFP4_SCALE_SR_GRAD", "0")
    monkeypatch.setenv("NVFP4_RHT_GRAD", "1")
    monkeypatch.setenv("NVFP4_RHT_ACTIVATION", "1")
    monkeypatch.setenv("NVFP4_RHT_WEIGHT", "0")
    monkeypatch.setenv("NVFP4_RHT_AXES", "col")
    monkeypatch.setenv("NVFP4_RHT_RANDOM_SIGNS", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_NATIVE_PAIRED_RHT_SPLIT3", "1")

    M = 256
    widths = [256, 512, 256]
    total = sum(widths)
    row_fp4_full = torch.zeros(
        (M, total // 2), dtype=torch.uint8
    ).view(torch.float4_e2m1fn_x2)
    row_sc_full = torch.zeros((M // 128, total // 64, 512), dtype=torch.uint8)
    col_fp4_full = torch.zeros(
        (total, M // 2), dtype=torch.uint8
    ).view(torch.float4_e2m1fn_x2)
    col_sc_full = torch.zeros((total // 128, M // 64, 512), dtype=torch.uint8)
    shared_row_sg = torch.ones((M // 256, 1), dtype=torch.float32)
    col_sg_full = torch.ones((1, total // 256), dtype=torch.float32)

    row_fp4_list = []
    row_sc_list = []
    col_fp4_list = []
    col_sc_list = []
    col_sg_list = []
    fp4_off = row_sc_off = col_off = col_sc_off = col_sg_off = 0
    for width in widths:
        row_fp4_list.append(row_fp4_full.narrow(1, fp4_off, width // 2))
        row_sc_list.append(row_sc_full.narrow(1, row_sc_off, width // 64))
        col_fp4_list.append(col_fp4_full.narrow(0, col_off, width))
        col_sc_list.append(col_sc_full.narrow(0, col_sc_off, width // 128))
        col_sg_list.append(col_sg_full.narrow(1, col_sg_off, width // 256))
        fp4_off += width // 2
        row_sc_off += width // 64
        col_off += width
        col_sc_off += width // 128
        col_sg_off += width // 256

    result = (
        row_fp4_list,
        row_sc_list,
        [shared_row_sg, shared_row_sg, shared_row_sg],
        col_fp4_list,
        col_sc_list,
        col_sg_list,
        row_fp4_full,
        row_sc_full,
        shared_row_sg,
        col_fp4_full,
        col_sc_full,
        col_sg_full,
    )
    recorder = _NativeSplit3QuantRecorder(result)
    grads = [torch.zeros((M, width), dtype=torch.bfloat16) for width in widths]
    state = torch.tensor([42, 17], dtype=torch.int64)

    def _forbid_cat(*_args, **_kwargs):
        raise AssertionError("native paired split3 must not concatenate")

    monkeypatch.setattr(tk_gemm.torch, "cat", _forbid_cat)
    package = tk_gemm._localcta_paired_rht_split3_package(
        recorder,
        *grads,
        widths,
        persistent_rng_state=state,
    )

    assert recorder.inputs == tuple(grads)
    assert recorder.kwargs == {
        "data_stochastic_rounding": True,
        "rng_seed": 0,
        "rng_subsequence_base": 0,
        "data_sr_axes": "row",
        "persistent_rng_state": state,
        "encode_centric": False,
    }
    assert package["a_fp4_full"].data_ptr() == row_fp4_full.data_ptr()
    assert package["a_sc_full"].data_ptr() == row_sc_full.data_ptr()
    assert package["a_sg_full"].data_ptr() == shared_row_sg.data_ptr()
    assert package["fp4_col_full"].data_ptr() == col_fp4_full.data_ptr()
    assert package["a_sc_outer_folded"] is False
    assert package["paired_rht_backend"] == "native_split3"
    assert all(sg.data_ptr() == shared_row_sg.data_ptr() for sg in package["a_sg_list"])
    assert all(
        part.untyped_storage().data_ptr() == row_fp4_full.untyped_storage().data_ptr()
        for part in package["fp4_row_list"]
    )
    assert all(
        part.untyped_storage().data_ptr() == col_fp4_full.untyped_storage().data_ptr()
        for part in package["fp4_col_list"]
    )


def _packed_root_name(node: ast.AST) -> str | None:
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
            continue
        return None


def test_integrated_sources_forbid_raw_packed_fp4_contiguous() -> None:
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for relative in (
        "low_bits_training/quantization/fused_te_linear.py",
        "low_bits_training/quantization/tk_gemm.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "contiguous"
            ):
                continue
            root_name = (_packed_root_name(node.func.value) or "").lower()
            if "fp4" not in root_name:
                continue
            expression = ast.unparse(node)
            byte_backed = (
                "view(torch.uint8).contiguous" in expression
                or root_name.endswith("_u8")
                or root_name.endswith("bytes")
            )
            if not byte_backed:
                offenders.append(f"{relative}:{node.lineno}:{expression}")
    assert offenders == []


def test_paired_rht_functions_forbid_raw_packed_copy_or_conversion() -> None:
    root = Path(__file__).resolve().parents[2]
    targets = {
        "_fast_quantize_localcta_v4_split2_paired_rht_carrier",
        "_localcta_paired_rht_split3_package",
    }
    offenders = []
    for relative in (
        "low_bits_training/quantization/fused_te_linear.py",
        "low_bits_training/quantization/tk_gemm.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef) or function.name not in targets:
                continue
            for node in ast.walk(function):
                if (
                    not isinstance(node, ast.Call)
                    or not isinstance(node.func, ast.Attribute)
                    or node.func.attr not in {"contiguous", "copy_", "to"}
                ):
                    continue
                root_name = (_packed_root_name(node.func.value) or "").lower()
                expression = ast.unparse(node)
                if "fp4" in root_name and "view(torch.uint8)" not in expression:
                    offenders.append(
                        f"{relative}:{node.lineno}:{function.name}:{expression}"
                    )
    assert offenders == []


class _QuantRecorder:
    def __init__(self) -> None:
        self.args = None
        self.kwargs = None

    def tk_quantize_for_gemm_opt(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        payload = torch.empty((1, 1), dtype=torch.uint8)
        scale = torch.empty((1, 1), dtype=torch.uint8)
        sg = torch.empty((1,), dtype=torch.float32)
        return payload, scale, payload, scale, sg, sg


class _Split2LaunchRecorder:
    def __init__(self) -> None:
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return "split2-result"


def test_split2_native_rht_capability_requires_explicit_marker(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    class OldExtension:
        @staticmethod
        def tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace():
            pass

    assert not fused_te_linear._tk_localcta_silu_deriv_split2_supports_rht(
        OldExtension()
    )


def test_split2_native_rht_capability_accepts_consistent_extension(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)

    class NewExtension:
        @staticmethod
        def tk_localcta_silu_deriv_split2_supports_rht():
            return True

        @staticmethod
        def tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace():
            pass

    assert fused_te_linear._tk_localcta_silu_deriv_split2_supports_rht(
        NewExtension()
    )


def test_split2_native_rht_capability_rejects_inconsistent_extension(
    monkeypatch,
) -> None:
    fused_te_linear = _load(monkeypatch)

    class BrokenExtension:
        @staticmethod
        def tk_localcta_silu_deriv_split2_supports_rht():
            return True

    with pytest.raises(RuntimeError, match="lacks its launch API"):
        fused_te_linear._tk_localcta_silu_deriv_split2_supports_rht(
            BrokenExtension()
        )


def test_sr_only_paired_control_retains_generic_carrier(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_RHT_GRAD", "0")

    class Extension:
        @staticmethod
        def tk_localcta_silu_deriv_split2_supports_rht():
            raise AssertionError("SR-only control must not query native RHT support")

        @staticmethod
        def tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace():
            pass

    assert not fused_te_linear._use_tk_localcta_native_paired_rht_split2(
        Extension(),
        paired_rht_carrier=True,
    )


def test_rht_treatment_selects_advertised_native_carrier(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_RHT_GRAD", "1")

    class Extension:
        @staticmethod
        def tk_localcta_silu_deriv_split2_supports_rht():
            return True

        @staticmethod
        def tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace():
            pass

    assert fused_te_linear._use_tk_localcta_native_paired_rht_split2(
        Extension(),
        paired_rht_carrier=True,
    )


def test_native_split2_rht_can_be_disabled_for_same_binary_ab(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_RHT_GRAD", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_NATIVE_PAIRED_RHT_SPLIT2", "0")

    class Extension:
        @staticmethod
        def tk_localcta_silu_deriv_split2_supports_rht():
            raise AssertionError("disabled native carrier must not query capability")

    assert not fused_te_linear._use_tk_localcta_native_paired_rht_split2(
        Extension(),
        paired_rht_carrier=True,
    )


def test_native_split2_rht_call_forwards_row_sr_rht_and_checkpoint_state(
    monkeypatch,
) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", "0")
    monkeypatch.setenv("NVFP4_SCALE_SR_GRAD", "0")
    monkeypatch.setenv("NVFP4_RHT_GRAD", "1")
    monkeypatch.setenv("NVFP4_RHT_ACTIVATION", "1")
    monkeypatch.setenv("NVFP4_RHT_WEIGHT", "0")
    monkeypatch.setenv("NVFP4_RHT_AXES", "col")
    monkeypatch.setenv("NVFP4_RHT_RANDOM_SIGNS", "1")
    state = torch.zeros(2, dtype=torch.int64)
    recorder = _Split2LaunchRecorder()

    result = fused_te_linear._call_localcta_silu_deriv_split2(
        recorder,
        "existing-positional-abi",
        persistent_rng_state=state,
        native_paired_rht=True,
    )

    assert result == "split2-result"
    assert recorder.args == ("existing-positional-abi",)
    assert recorder.kwargs == {
        "rht_axes": "col",
        "with_random_sign_mask": True,
        "derivatives_precomputed": True,
        "encode_centric": False,
        "persistent_rng_state": state,
    }


def test_old_split2_abi_keeps_checkpoint_state_positional(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    state = torch.zeros(2, dtype=torch.int64)
    recorder = _Split2LaunchRecorder()

    fused_te_linear._call_localcta_silu_deriv_split2(
        recorder,
        "existing-positional-abi",
        persistent_rng_state=state,
        native_paired_rht=False,
    )

    assert recorder.args == ("existing-positional-abi", state)
    assert recorder.kwargs == {}


def test_native_split2_uses_established_te_bf16_derivative_producer(
    monkeypatch,
) -> None:
    fused_te_linear = _load(monkeypatch)

    class TERecorder:
        def __init__(self) -> None:
            self.calls = []

        def fused_silu_deriv_dual_mul_bf16_out_no_amax(
            self, dh, h3, h1_raw, dh1, dh3_out
        ) -> None:
            self.calls.append((dh, h3, h1_raw, dh1, dh3_out))
            dh1.fill_(3)
            dh3_out.fill_(5)

        def fused_silu_deriv_dual_mul_bf16_out(self, *args) -> None:
            raise AssertionError("lower-priority TE producer was selected")

    te_fused = TERecorder()
    dh = torch.zeros((2, 2), dtype=torch.bfloat16)
    h3 = torch.zeros_like(dh)
    h1_raw = torch.zeros_like(dh)
    dh1 = torch.empty_like(dh)
    dh3_out = torch.empty_like(dh)

    fused_te_linear._produce_ffn_localcta_derivatives_with_te(
        te_fused,
        dh,
        h3,
        h1_raw,
        dh1,
        dh3_out,
        torch.zeros(1),
        torch.zeros(1),
    )

    assert len(te_fused.calls) == 1
    assert all(
        observed is expected
        for observed, expected in zip(
            te_fused.calls[0], (dh, h3, h1_raw, dh1, dh3_out), strict=True
        )
    )
    assert torch.equal(dh1, torch.full_like(dh1, 3))
    assert torch.equal(dh3_out, torch.full_like(dh3_out, 5))


def test_ffn_paired_rht_native_route_bypasses_only_split2_fallback() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "low_bits_training/quantization/fused_te_linear.py"
    ).read_text(encoding="utf-8")

    assert "if paired_rht_carrier and not native_paired_rht_split2:" in source
    assert "native_paired_rht=native_paired_rht_split2" in source
    assert "derivatives_precomputed=True" in source
    assert source.count("_produce_ffn_localcta_derivatives_with_te(") == 4
    assert (
        "use_tk_localcta_v4_ffn_deriv_w2_wgrad_overlap()\n"
        "                and not paired_rht_carrier"
    ) in source
    # The global paired flag remains available to QKV/WO; only the FFN split2
    # fallback branch is marker-gated.
    assert source.count("paired_rht_carrier = (") == 2


def test_localcta_gradient_quantizer_forwards_row_sr(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    from low_bits_training.quantization import tk_gemm

    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("NVFP4_RHT_GRAD", "0")
    recorder = _QuantRecorder()
    monkeypatch.setattr(tk_gemm, "_get_tk_quant_for_gemm", lambda: recorder)
    monkeypatch.setattr(tk_gemm, "get_tk_localcta_variant", lambda: "v4")

    fused_te_linear._fast_quantize_localcta_v4_opt(
        torch.zeros((128, 128), dtype=torch.bfloat16),
        nvfp4_role="grad",
    )

    assert recorder.args[3] is True
    assert recorder.args[-1] == "row"


def test_v5_gradient_quantizer_forwards_row_sr(monkeypatch) -> None:
    fused_te_linear = _load(monkeypatch)
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("NVFP4_RHT_GRAD", "0")
    recorder = _QuantRecorder()
    monkeypatch.setattr(fused_te_linear, "_get_tk_quant", lambda: recorder)

    fused_te_linear._fast_quantize_tk_regular_opt(
        torch.zeros((128, 128), dtype=torch.bfloat16),
        nvfp4_role="grad",
    )

    assert recorder.args[3] is True
    assert recorder.kwargs == {"data_sr_axes": "row"}
