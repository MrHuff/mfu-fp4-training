import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "microscope_llama_quantized_operands.py"
)
SPEC = importlib.util.spec_from_file_location(
    "microscope_llama_quantized_operands", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
microscope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = microscope
SPEC.loader.exec_module(microscope)


def test_aligned_capture_requires_complete_global_rht_blocks():
    receipt = microscope.validate_aligned_row_blocks(
        [*range(0, 16), *range(48, 64)], total_rows=128, block_size=16
    )
    assert receipt == {
        "block_size": 16,
        "sample_rows": 32,
        "source_rows": 128,
        "complete_blocks": 2,
        "block_starts": [0, 48],
        "covers_all_source_rows": False,
    }
    full = microscope.validate_aligned_row_blocks(
        list(range(128)), total_rows=128, block_size=16
    )
    assert full["covers_all_source_rows"] is True
    with pytest.raises(RuntimeError, match="not a multiple"):
        microscope.validate_aligned_row_blocks(
            list(range(17)), total_rows=128, block_size=16
        )
    with pytest.raises(RuntimeError, match="not complete globally aligned"):
        microscope.validate_aligned_row_blocks(
            list(range(1, 17)), total_rows=128, block_size=16
        )
    with pytest.raises(RuntimeError, match="strictly increasing"):
        microscope.validate_aligned_row_blocks(
            list(range(16)) + list(range(16)), total_rows=128, block_size=16
        )


def test_mxfp4_scale_swizzle_and_decoder_are_exact_on_synthetic_payload():
    rows = cols = 128
    logical_scales = (
        torch.arange(rows * (cols // 32), dtype=torch.int64).reshape(
            rows, cols // 32
        )
        % 21
        + 117
    ).to(torch.uint8)
    swizzled = microscope.swizzle_mxfp4_scales_for_test(
        logical_scales, rows, cols
    )
    assert torch.equal(
        microscope.unswizzle_mxfp4_scales(swizzled, rows, cols), logical_scales
    )

    codes = torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols) & 0x0F
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    decoded = microscope.decode_mxfp4(packed, swizzled, rows, cols)
    lut = torch.tensor(microscope.E2M1_LUT)
    maximum = torch.exp2(logical_scales.float() - 127.0).repeat_interleave(
        32, dim=1
    )
    expected = lut[codes.long()] * maximum / 6.0
    assert torch.equal(decoded.codes, codes)
    assert torch.equal(decoded.scale_codes, logical_scales)
    assert torch.equal(decoded.values, expected)
    assert torch.equal(decoded.maximum, maximum)
    assert torch.equal(decoded.minimum_positive, maximum / 12.0)


def test_mxfp4_unpack_is_low_nibble_then_high_nibble():
    packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8)
    assert microscope.unpack_e2m1_codes(packed, 1, 4).tolist() == [[1, 2, 8, 15]]


def test_fixed_rht_sign_mask_and_paired_contraction_contract():
    assert microscope.decode_positive_sign_mask(0x2817, 16) == (
        1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
    )
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(32, 7, generator=generator)
    dy = torch.randn(32, 5, generator=generator)
    x_rht = microscope.block_rht(
        x, block_size=16, sign_mode="fixed-0x2817"
    ).float()
    dy_rht = microscope.block_rht(
        dy, block_size=16, sign_mode="fixed-0x2817"
    ).float()
    reference = dy.T @ x
    transformed = dy_rht.T @ x_rht
    assert float((reference - transformed).abs().max()) / float(
        reference.abs().max()
    ) < 0.01
    one_sided = dy_rht.T @ x
    assert float((reference - one_sided).abs().max()) > 0.1


def test_cpu_oracle_gate_runs_without_cuda():
    receipt = microscope.run_cpu_oracle_gates()
    assert receipt["mxfp4_scale_swizzle_roundtrip"] is True
    assert receipt["mxfp4_synthetic_decode_max_abs_error"] == 0.0
    assert receipt["fixed_positive_mask"] == "0x2817"


def test_direct_metrics_separate_true_clipping_from_max_code_occupancy():
    reference = torch.tensor([[0.0, 0.01, 0.50, 2.0, -3.0, 0.25]])
    decoded = torch.tensor([[0.0, 0.0, 0.50, 1.0, -1.0, -0.25]])
    maximum = torch.ones_like(reference)
    minimum = torch.full_like(reference, 0.05)
    # Mark the in-range 0.5 value as max-code too: occupancy is not clipping.
    codes = torch.tensor([[0, 0, 7, 7, 15, 9]], dtype=torch.uint8)
    metrics = microscope.direct_operand_metrics(
        reference,
        decoded,
        maximum,
        minimum,
        codes,
        scale_payload=torch.tensor([127], dtype=torch.uint8),
        scale_kind="fixture",
        profile_samples=reference.numel(),
    )
    assert metrics["reference_nonzero_to_decoded_zero"]["count"] == 1
    assert metrics["reference_below_minimum_positive"]["count"] == 1
    assert metrics["true_source_range_exceedance"]["count"] == 2
    assert metrics["max_code_occupancy"]["count"] == 3
    assert metrics["sign_flip_excluding_decoded_zero_fraction"] > 0


def test_rht_geometry_oracle_accepts_fixed_and_rejects_wrong_expectation():
    generator = torch.Generator().manual_seed(99)
    raw = torch.randn(32, 128, generator=generator).to(torch.bfloat16)
    fixed = microscope.block_rht(
        raw, block_size=16, sign_mode="fixed-0x2817"
    )
    result = microscope.rht_oracle_gate(
        fixed,
        raw,
        expected_sign_mode="fixed-0x2817",
        block_size=16,
        minimum_margin=2.0,
    )
    assert result["passed"] is True
    assert result["winner"] == "fixed-0x2817"
    with pytest.raises(RuntimeError, match="geometry oracle failed"):
        microscope.rht_oracle_gate(
            fixed,
            raw,
            expected_sign_mode="plain",
            block_size=16,
            minimum_margin=2.0,
        )


def _write_source_contract(root: Path, route: str, *, omit: str | None = None):
    relative, symbols = microscope._required_source_symbols(route)
    path = root / relative
    path.parent.mkdir(parents=True)
    lines = [
        f'm.def("{symbol}", &{symbol});'
        for symbol in symbols
        if symbol != omit
    ]
    path.write_text("\n".join(lines) + "\n")
    if route == "localcta-rht":
        header = (
            root
            / microscope.LOCALCTA_EXTENSION_DIRECTORY
            / "fused_localcta_quantize.cuh"
        )
        header.write_text("unsigned make_sign_bits() { return 0x00002817u; }\n")


@pytest.mark.parametrize("route", microscope.ROUTES)
def test_runtime_source_contract_seals_exact_required_pybind_symbols(
    tmp_path, route
):
    _write_source_contract(tmp_path, route)
    receipt = microscope.audit_runtime_source_contract(tmp_path, route)
    _, required = microscope._required_source_symbols(route)
    assert receipt["required_pybind_symbols"] == list(required)
    assert len(receipt["source_file_sha256"]) == 64


def test_runtime_source_contract_fails_closed_on_missing_symbol(tmp_path):
    route = "localcta-rht"
    missing = "tk_localcta_reconstruct_col"
    _write_source_contract(tmp_path, route, omit=missing)
    with pytest.raises(RuntimeError, match=missing):
        microscope.audit_runtime_source_contract(tmp_path, route)


def test_runtime_source_contract_fails_closed_on_localcta_sign_mask_drift(tmp_path):
    route = "localcta-rht"
    _write_source_contract(tmp_path, route)
    header = (
        tmp_path
        / microscope.LOCALCTA_EXTENSION_DIRECTORY
        / "fused_localcta_quantize.cuh"
    )
    header.write_text("unsigned make_sign_bits() { return 0x0000ffffu; }\n")
    with pytest.raises(RuntimeError, match="fixed mask 0x2817"):
        microscope.audit_runtime_source_contract(tmp_path, route)


def test_route_environment_seals_production_localcta_switches(monkeypatch):
    for key in (
        "USE_TK_LOCALCTA_V4_FAST_DATA_SR",
        "USE_TK_LOCALCTA_V4_SILU_ATOMIC_FINAL_SG_PRODUCER",
        "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW",
    ):
        monkeypatch.delenv(key, raising=False)
    policy = microscope.configure_route_environment(
        "localcta-rht", localcta_scale_num=448.0
    )
    assert policy["USE_TK_LOCALCTA_V3_CONTRACT"] == "outer"
    assert policy["USE_TK_LOCALCTA_V4_FAST_DATA_SR"] == "1"
    assert policy["USE_TK_LOCALCTA_V4_SILU_ATOMIC_FINAL_SG_PRODUCER"] == "1"
    assert policy["USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW"] == "1"
    assert policy["USE_TK_LOCALCTA_V4_COL_RHT_AMAX_RAW_MULTIPLIER"] == "2.0"


def test_default_scope_is_requested_three_layers_and_w2_sites():
    assert microscope.DEFAULT_LAYERS == "12,16,31"
    assert microscope.SITES == ("w2_activation", "w2_dy", "down_weight")
    assert microscope._parse_layer_spec(microscope.DEFAULT_LAYERS, 32) == [12, 16, 31]
    assert microscope._parse_sites("all") == microscope.SITES
