# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("LBT_LIGHT_IMPORT", "1")
import low_bits_training.analysis as _analysis_package

_ANALYSIS_PATH = Path(__file__).resolve().parents[2] / "low_bits_training/analysis"
if str(_ANALYSIS_PATH) not in _analysis_package.__path__:
    _analysis_package.__path__.insert(0, str(_ANALYSIS_PATH))

from low_bits_training.analysis.llama_checkpoint_routes import (
    BF16_UNFUSED,
    FUSED_ROUTES,
    LOCALCTA_FUSED,
    LlamaSpec,
    MXFP4_FUSED,
    PURE_V5_FUSED,
    TE_NATIVE_NVFP4_UNFUSED,
)
from low_bits_training.analysis.llama_conversion_parity import (
    CANONICAL_FIXED_TOKEN_IDS,
    CANONICAL_FIXED_TOKEN_IDS_SHA256,
    CANONICAL_LOGITS_SHAPE,
    CANONICAL_PARITY_TOLERANCES,
    CANONICAL_SEMANTIC_TOLERANCES,
    FROZEN_ALIASES_BY_ROUTE,
    PARITY_METHOD,
    PARITY_ENVIRONMENT_FIELDS,
    PARITY_LOGIT_MEASUREMENT_FIELDS,
    PARITY_MEASUREMENT_FIELDS,
    PARITY_POLICY,
    PARITY_RECEIPT_SCHEMA_VERSION,
    PARITY_CODE_FILE_KEYS,
    PARITY_SEMANTIC_MEASUREMENT_FIELDS,
    PARITY_TOLERANCE_FIELDS,
    PINNED_TORCHTITAN_COMMIT,
    SOURCE_TENSORS_BY_ROUTE,
    SUPPORTED_ROUTES,
    canonical_json_bytes,
    compare_logits,
    compare_semantic_logits,
    seal_receipt,
    sha256_bytes,
    token_ids_sha256,
    validate_receipt,
    write_atomic_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_TOOL_PATH = (
    REPO_ROOT / "scripts/evaluation/validate_llama8b_conversion_parity.py"
)


def _load_parity_tool():
    spec = importlib.util.spec_from_file_location(
        "validate_llama8b_conversion_parity", PARITY_TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_spec() -> LlamaSpec:
    return LlamaSpec(
        layers=2,
        dim=8,
        hidden_dim=12,
        vocab_size=16,
        heads=2,
        kv_heads=1,
        head_dim=4,
        max_position_embeddings=32,
    )


def _canonical_reference() -> torch.Tensor:
    logits = torch.full(CANONICAL_LOGITS_SHAPE, -10.0, dtype=torch.float32)
    top_candidates = torch.tensor(
        [
            2.0,
            1.99,
            1.7,
            1.6,
            1.5,
            1.4,
            1.3,
            1.2,
            0.02,
            0.01,
            0.0,
            -0.01,
        ],
        dtype=torch.float32,
    )
    logits[:, :, : len(top_candidates)] = top_candidates
    return logits


def _replace_tenth_candidate(logits: torch.Tensor, position: int) -> None:
    logits[0, position, 9] = -0.01
    logits[0, position, 10] = 0.02


def _passing_measurements() -> dict:
    reference = _canonical_reference()
    measurements = compare_logits(
        reference,
        reference.clone(),
        CANONICAL_PARITY_TOLERANCES,
    )
    assert measurements["passed"] is True
    return measurements


def _passing_payload(route: str = PURE_V5_FUSED) -> dict:
    tokens = list(CANONICAL_FIXED_TOKEN_IDS)
    reference = _canonical_reference()
    stock_measurements = _passing_measurements()
    semantic_measurements = compare_semantic_logits(
        reference,
        reference.clone(),
        CANONICAL_SEMANTIC_TOLERANCES,
    )
    measurements = {
        "passed": True,
        "canonical_semantic": semantic_measurements,
        "stock_hf_evaluator_drift": stock_measurements,
        "source_tensors_streamed": SOURCE_TENSORS_BY_ROUTE[route],
        "native_parameters_loaded": 291,
        "converted_tensors_exact": 291,
        "converted_elements_exact": 8_030_261_248,
        "frozen_aliases_checked": FROZEN_ALIASES_BY_ROUTE[route],
        "native_math_sdpa_modules": 32,
    }
    code_files = {
        name: sha256_bytes(name.encode("utf-8"))
        for name in sorted(PARITY_CODE_FILE_KEYS)
    }
    return {
        "schema_version": PARITY_RECEIPT_SCHEMA_VERSION,
        "method": PARITY_METHOD,
        "policy": PARITY_POLICY,
        "passed": True,
        "created_at_utc": "2026-08-27T21:00:00+00:00",
        "conversion_manifest_sha256": "a" * 64,
        "route": route,
        "step": 1,
        "ntokens_seen": 4,
        "checkpoint_metadata_sha256": "b" * 64,
        "source_job_id": "EXAMPLE",
        "source_uri_sha256": "c" * 64,
        "fixed_token_ids": tokens,
        "fixed_token_ids_sha256": token_ids_sha256(tokens),
        "expected_logits_shape": list(CANONICAL_LOGITS_SHAPE),
        "tool_sha256": code_files["parity_tool"],
        "code_bundle_sha256": sha256_bytes(canonical_json_bytes(code_files)),
        "code_files_sha256": code_files,
        "environment": {
            "python": "3.12.0",
            "platform": "Linux-6.8-test",
            "torch": torch.__version__,
            "transformers": "4.48.2",
            "safetensors": "0.5.3",
            "cuda_runtime": "13.0",
            "cudnn": 99999,
            "device": "cuda:0",
            "device_name": "NVIDIA B200",
            "compute_capability": [10, 0],
            "torchtitan_commit": PINNED_TORCHTITAN_COMMIT,
            "project_git_commit": "d" * 40,
            "project_tracked_dirty": False,
            "native_attention": ("TorchTitan scaled_dot_product_attention causal math"),
            "converted_attention": "Transformers SDPA causal math",
            "compute_dtype": "torch.bfloat16",
            "attention_backend": "SDPBackend.MATH",
            "canonical_semantic_rope": (
                "TorchTitan interleaved complex64 RoPE in converted HF model"
            ),
            "canonical_semantic_rmsnorm": (
                "TorchTitan torch.nn.functional.rms_norm in converted HF model"
            ),
            "stock_hf_rope": "Transformers half-split BF16 RoPE",
            "stock_hf_rmsnorm": ("Transformers LlamaRMSNorm FP32-normalize BF16-scale"),
        },
        "tolerances": {
            "canonical_semantic": CANONICAL_SEMANTIC_TOLERANCES.to_dict(),
            "stock_hf_evaluator_drift": CANONICAL_PARITY_TOLERANCES.to_dict(),
        },
        "measurements": measurements,
        "limitations": [],
    }


def test_compare_logits_accepts_one_close_miss_and_rejects_two():
    reference = _canonical_reference()
    converted = reference.clone()
    converted[0, 0, 100] += 0.4
    one_miss = compare_logits(reference, converted, CANONICAL_PARITY_TOLERANCES)
    assert one_miss["passed"] is True
    assert one_miss["strict_allclose"] is False
    assert one_miss["close_failure_count"] == 1
    assert one_miss["close_success_count"] == one_miss["logit_element_count"] - 1
    assert one_miss["close_failure_ratio"] == 1 / one_miss["logit_element_count"]

    converted[0, 1, 100] += 0.4
    two_misses = compare_logits(reference, converted, CANONICAL_PARITY_TOLERANCES)
    assert two_misses["close_failure_count"] == 2
    assert two_misses["passed"] is False


def test_compare_logits_accepts_99_of_100_top10_memberships_not_98():
    reference = _canonical_reference()
    converted = reference.clone()
    _replace_tenth_candidate(converted, 0)
    ninety_nine = compare_logits(reference, converted, CANONICAL_PARITY_TOLERANCES)
    assert ninety_nine["strict_allclose"] is True
    assert ninety_nine["top_k_intersection_counts"] == [9] + [10] * 9
    assert ninety_nine["top_k_intersection_count_min"] == 9
    assert ninety_nine["top_k_intersection_count_total"] == 99
    assert ninety_nine["top_k_intersection_ratio_min"] == 0.9
    assert ninety_nine["top_k_intersection_ratio_total"] == 0.99
    assert torch.tensor(9 / 10, dtype=torch.float32).item() < 0.9
    assert ninety_nine["passed"] is True

    _replace_tenth_candidate(converted, 1)
    ninety_eight = compare_logits(reference, converted, CANONICAL_PARITY_TOLERANCES)
    assert ninety_eight["top_k_intersection_count_min"] == 9
    assert ninety_eight["top_k_intersection_count_total"] == 98
    assert ninety_eight["passed"] is False


def test_compare_logits_rejects_eight_of_ten_at_any_position():
    reference = _canonical_reference()
    converted = reference.clone()
    converted[0, 0, 8] = -0.02
    converted[0, 0, 9] = -0.03
    converted[0, 0, 10] = 0.03
    converted[0, 0, 11] = 0.04
    result = compare_logits(reference, converted, CANONICAL_PARITY_TOLERANCES)
    assert result["strict_allclose"] is True
    assert result["top_k_intersection_count_min"] == 8
    assert result["top_k_intersection_count_total"] == 98
    assert result["passed"] is False


def test_compare_logits_rejects_top1_mismatch_even_when_top10_is_exact():
    reference = _canonical_reference()
    converted = reference.clone()
    converted[0, 0, 0], converted[0, 0, 1] = (
        converted[0, 0, 1].clone(),
        converted[0, 0, 0].clone(),
    )
    result = compare_logits(reference, converted, CANONICAL_PARITY_TOLERANCES)
    assert result["close_failure_count"] == 0
    assert result["top_1_mismatch_count"] == 1
    assert result["top_k_intersection_count_total"] == 100
    assert result["passed"] is False


def test_compare_logits_rejects_nonfinite_and_shape_drift():
    with pytest.raises(RuntimeError, match="shape mismatch"):
        compare_logits(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 2, 2),
            CANONICAL_PARITY_TOLERANCES,
        )
    with pytest.raises(RuntimeError, match="canonical r12 shape"):
        compare_logits(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            CANONICAL_PARITY_TOLERANCES,
        )
    reference = _canonical_reference()
    reference[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        compare_logits(
            reference,
            _canonical_reference(),
            CANONICAL_PARITY_TOLERANCES,
        )


def test_compare_logits_rejects_noncanonical_policy():
    reference = _canonical_reference()
    with pytest.raises(RuntimeError, match="canonical r12 policy"):
        compare_logits(
            reference,
            reference.clone(),
            replace(CANONICAL_PARITY_TOLERANCES, logit_atol=0.126),
        )


def test_compare_semantic_logits_requires_bit_exact_canonical_logits():
    reference = _canonical_reference()
    exact = compare_semantic_logits(
        reference, reference.clone(), CANONICAL_SEMANTIC_TOLERANCES
    )
    assert exact["passed"] is True
    assert exact["strict_bit_exact"] is True
    assert exact["exact_match_count"] == math.prod(CANONICAL_LOGITS_SHAPE)
    assert exact["mismatch_count"] == 0
    assert exact["exact_match_ratio"] == 1.0
    assert exact["mismatch_ratio"] == 0.0
    assert exact["max_abs_error"] == 0.0
    assert exact["reference_logits_sha256"] == exact["canonical_logits_sha256"]

    changed = reference.clone()
    changed[0, 0, 100] += 0.125
    mismatch = compare_semantic_logits(
        reference, changed, CANONICAL_SEMANTIC_TOLERANCES
    )
    assert mismatch["passed"] is False
    assert mismatch["strict_bit_exact"] is False
    assert mismatch["mismatch_count"] == 1
    assert mismatch["max_abs_error"] == 0.125

    signed_zero = reference.clone()
    assert reference[0, 0, 10].item() == 0.0
    signed_zero[0, 0, 10] = -0.0
    bit_mismatch = compare_semantic_logits(
        reference, signed_zero, CANONICAL_SEMANTIC_TOLERANCES
    )
    assert bit_mismatch["passed"] is False
    assert bit_mismatch["mismatch_count"] == 1
    assert bit_mismatch["max_abs_error"] == 0.0
    assert (
        bit_mismatch["reference_logits_sha256"]
        != bit_mismatch["canonical_logits_sha256"]
    )


def test_compare_semantic_logits_rejects_shape_policy_and_nonfinite_drift():
    reference = _canonical_reference()
    with pytest.raises(RuntimeError, match="canonical exact policy"):
        compare_semantic_logits(
            reference,
            reference.clone(),
            replace(CANONICAL_SEMANTIC_TOLERANCES, max_abs_error=0.125),
        )
    with pytest.raises(RuntimeError, match="semantic logit shape mismatch"):
        compare_semantic_logits(
            reference,
            reference[:, :-1],
            CANONICAL_SEMANTIC_TOLERANCES,
        )
    changed = reference.clone()
    changed[0, 0, 0] = float("inf")
    with pytest.raises(RuntimeError, match="nonfinite"):
        compare_semantic_logits(reference, changed, CANONICAL_SEMANTIC_TOLERANCES)


def test_tt_equivalent_hf_rope_restores_interleaved_complex_semantics():
    parity_tool = _load_parity_tool()
    interleaved_q = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
        dtype=torch.bfloat16,
    )
    interleaved_k = interleaved_q + 1

    def to_hf_half_split(value):
        return torch.cat((value[..., 0::2], value[..., 1::2]), dim=-1)

    q = to_hf_half_split(interleaved_q)
    k = to_hf_half_split(interleaved_k)
    angles = torch.tensor([[0.0, 0.0], [0.25, 0.5]], dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    cos = (
        torch.cat((freqs_cis.real, freqs_cis.real), dim=-1)
        .to(torch.bfloat16)
        .unsqueeze(0)
    )
    sin = (
        torch.cat((freqs_cis.imag, freqs_cis.imag), dim=-1)
        .to(torch.bfloat16)
        .unsqueeze(0)
    )

    observed_q, observed_k = parity_tool._tt_equivalent_hf_rope(
        q, k, cos, sin, freqs_cis=freqs_cis
    )

    def native(value):
        complex_value = torch.view_as_complex(
            value.float().reshape(*value.shape[:-1], -1, 2)
        )
        frequencies = freqs_cis.view(1, 1, 2, 2)
        return (
            torch.view_as_real(complex_value * frequencies)
            .flatten(-2)
            .to(torch.bfloat16)
        )

    assert torch.equal(observed_q, native(interleaved_q))
    assert torch.equal(observed_k, native(interleaved_k))
    assert not torch.equal(observed_q, to_hf_half_split(native(interleaved_q)))


def test_tt_equivalent_hf_rope_fails_closed_on_malformed_phase_carrier():
    parity_tool = _load_parity_tool()
    q = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
    freqs_cis = torch.ones((2, 2), dtype=torch.complex64)
    cos = torch.ones((1, 2, 4), dtype=torch.bfloat16)
    sin = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
    cos[..., 0] = 0
    cos[..., 2] = 0
    with pytest.raises(RuntimeError, match="not on the unit circle"):
        parity_tool._tt_equivalent_hf_rope(q, q, cos, sin, freqs_cis=freqs_cis)


def test_tt_equivalent_hf_rope_ignores_valid_long_sequence_carrier_rounding():
    parity_tool = _load_parity_tool()
    sequence = 8192
    head_dim = 128
    interleaved = torch.zeros(
        (1, 1, sequence, head_dim), dtype=torch.bfloat16
    )
    interleaved[..., 0] = 1
    half_split = torch.cat(
        (interleaved[..., 0::2], interleaved[..., 1::2]), dim=-1
    )
    inv_freq = 1.0 / (
        500000.0
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(torch.arange(sequence, dtype=torch.float32), inv_freq)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16)[None]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(torch.bfloat16)[None]
    # Reproduce a harmless one-ULP carrier discrepancy observed during
    # lm-eval auto-batch probing.  Canonical arithmetic must remain bound to
    # freqs_cis rather than to these HF-generated carrier values.
    rounded = torch.nextafter(
        sin[0, -1, 0], torch.tensor(1.0, dtype=torch.bfloat16)
    )
    sin[0, -1, 0] = rounded
    sin[0, -1, head_dim // 2] = rounded

    observed, _ = parity_tool._tt_equivalent_hf_rope(
        half_split, half_split, cos, sin, freqs_cis=freqs_cis
    )
    native = torch.view_as_real(
        torch.view_as_complex(interleaved.float().reshape(1, 1, sequence, -1, 2))
        * freqs_cis.view(1, 1, sequence, -1)
    ).flatten(-2).to(torch.bfloat16)
    assert torch.equal(observed, native)


def test_tt_equivalent_hf_rope_broadcasts_one_position_row_across_batch():
    parity_tool = _load_parity_tool()
    torch.manual_seed(0)
    interleaved = torch.randn((3, 2, 4, 8), dtype=torch.bfloat16)
    half_split = torch.cat(
        (interleaved[..., 0::2], interleaved[..., 1::2]), dim=-1
    )
    angles = torch.randn((4, 4), dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    cos = torch.cat((freqs_cis.real, freqs_cis.real), dim=-1).to(
        torch.bfloat16
    )[None]
    sin = torch.cat((freqs_cis.imag, freqs_cis.imag), dim=-1).to(
        torch.bfloat16
    )[None]

    observed, _ = parity_tool._tt_equivalent_hf_rope(
        half_split, half_split, cos, sin, freqs_cis=freqs_cis
    )
    complex_value = torch.view_as_complex(
        interleaved.float().reshape(*interleaved.shape[:-1], -1, 2)
    )
    expected = (
        torch.view_as_real(complex_value * freqs_cis.view(1, 1, 4, 4))
        .flatten(-2)
        .to(torch.bfloat16)
    )
    assert torch.equal(observed, expected)


def test_canonical_hf_rope_restores_global_binding_after_failure():
    modeling_llama = pytest.importorskip("transformers.models.llama.modeling_llama")

    parity_tool = _load_parity_tool()
    original = modeling_llama.apply_rotary_pos_emb
    freqs_cis = torch.ones((2, 2), dtype=torch.complex64)
    with pytest.raises(ValueError, match="sentinel"):
        with parity_tool._canonical_hf_rope(freqs_cis):
            assert modeling_llama.apply_rotary_pos_emb is not original
            raise ValueError("sentinel")
    assert modeling_llama.apply_rotary_pos_emb is original


def test_canonical_hf_rmsnorm_matches_torchtitan_arithmetic_only_in_context():
    modeling_llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    parity_tool = _load_parity_tool()
    torch.manual_seed(0)
    hidden = torch.randn(2, 4096, dtype=torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(4096)).to(torch.bfloat16)
    norm = modeling_llama.LlamaRMSNorm(4096, eps=1e-5).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    native = torch.nn.functional.rms_norm(hidden, (hidden.shape[-1],), weight, 1e-5)
    original = modeling_llama.LlamaRMSNorm.forward
    stock = norm(hidden)
    assert not torch.equal(stock, native)

    with parity_tool._canonical_hf_rmsnorm():
        assert torch.equal(norm(hidden), native)
    assert modeling_llama.LlamaRMSNorm.forward is original
    assert torch.equal(norm(hidden), stock)


def test_canonical_hf_rmsnorm_restores_global_binding_after_failure():
    modeling_llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    parity_tool = _load_parity_tool()
    original = modeling_llama.LlamaRMSNorm.forward
    with pytest.raises(ValueError, match="sentinel"):
        with parity_tool._canonical_hf_rmsnorm():
            raise ValueError("sentinel")
    assert modeling_llama.LlamaRMSNorm.forward is original


@pytest.mark.parametrize("expected_shards", [32, 64])
def test_parity_tool_accepts_the_exact_sealed_dcp_shard_set(
    expected_shards: int,
):
    parity_tool = _load_parity_tool()
    expected = {f"__{rank}_0.distcp" for rank in range(expected_shards)}
    assert parity_tool._expected_dcp_shard_names(expected_shards) == expected
    assert {
        parity_tool._canonical_dcp_shard_name(
            name, "test shard", expected_shards
        )
        for name in expected
    } == expected


def _source_inventory_fixture(
    checkpoint: Path, expected_shards: int
) -> tuple[SimpleNamespace, dict[str, object]]:
    names = [f"__{rank}_0.distcp" for rank in range(expected_shards)]
    storage_data = {}
    for rank, name in enumerate(names):
        (checkpoint / name).write_bytes(b"sealed")
        storage_data[rank] = SimpleNamespace(relative_path=name)
    return (
        SimpleNamespace(storage_data=storage_data),
        {"source_shards": names},
    )


@pytest.mark.parametrize("expected_shards", [32, 64])
def test_source_inventory_accepts_exact_contiguous_geometry(
    tmp_path: Path, expected_shards: int
):
    parity_tool = _load_parity_tool()
    metadata, manifest = _source_inventory_fixture(tmp_path, expected_shards)
    parity_tool._validate_source_inventory(
        tmp_path, metadata, manifest, expected_shards
    )


@pytest.mark.parametrize("kind", ["gap", "extra", "duplicate"])
def test_source_inventory_rejects_gap_extra_or_duplicate(
    tmp_path: Path, kind: str
):
    parity_tool = _load_parity_tool()
    metadata, manifest = _source_inventory_fixture(tmp_path, 64)
    if kind == "gap":
        missing = "__31_0.distcp"
        manifest["source_shards"].remove(missing)
        metadata.storage_data.pop(31)
        (tmp_path / missing).unlink()
        match = "exact contiguous"
    elif kind == "extra":
        extra = "__64_0.distcp"
        manifest["source_shards"].append(extra)
        metadata.storage_data[64] = SimpleNamespace(relative_path=extra)
        (tmp_path / extra).write_bytes(b"sealed")
        match = "noncanonical"
    else:
        manifest["source_shards"].append("__0_0.distcp")
        match = "duplicate"
    with pytest.raises(RuntimeError, match=match):
        parity_tool._validate_source_inventory(
            tmp_path, metadata, manifest, 64
        )


@pytest.mark.parametrize(
    "name",
    [
        "../__0_0.distcp",
        "/tmp/__0_0.distcp",
        "__0_0.distcp/extra",
        "__00_0.distcp",
        "__32_0.distcp",
        "__0_1.distcp",
        "0_0.distcp",
        ".metadata",
    ],
)
def test_parity_tool_rejects_noncanonical_dcp_shard_names(name):
    parity_tool = _load_parity_tool()
    with pytest.raises(RuntimeError, match="noncanonical test shard"):
        parity_tool._canonical_dcp_shard_name(name, "test shard", 32)


def test_parity_tool_keeps_generic_hf_filename_policy_separate_from_dcp():
    parity_tool = _load_parity_tool()
    assert (
        parity_tool._safe_filename(
            "model-00001-of-00004.safetensors", "HF weight filename"
        )
        == "model-00001-of-00004.safetensors"
    )
    with pytest.raises(RuntimeError, match="unsafe HF weight filename"):
        parity_tool._safe_filename("../model.safetensors", "HF weight filename")


def test_parity_tool_pins_math_sdpa_for_both_forward_paths():
    parity_tool = _load_parity_tool()
    source = PARITY_TOOL_PATH.read_text(encoding="utf-8")
    assert "module.sdpa_backends = [SDPBackend.MATH]" in source
    assert "sdpa_kernel(" in source
    assert "[SDPBackend.MATH], set_priority=True" in source
    assert '"attention_backend": "SDPBackend.MATH"' in source
    assert "_tt_equivalent_hf_rope" in source
    assert "modeling_llama.apply_rotary_pos_emb = canonical" in source
    assert "modeling_llama.apply_rotary_pos_emb = original" in source
    assert "modeling_llama.LlamaRMSNorm.forward = canonical" in source
    assert "modeling_llama.LlamaRMSNorm.forward = original" in source
    assert "len(rmsnorm_modules) != 65" in source
    assert 'if receipt["passed"]:\n        validate_receipt(receipt)' in source


def test_r12_policy_constants_thresholds_and_cli_are_frozen():
    parity_tool = _load_parity_tool()
    assert PARITY_RECEIPT_SCHEMA_VERSION == 2
    assert PARITY_METHOD == "pinned-torchtitan-native-plus-exact-state-r12"
    assert PARITY_POLICY == "llama8b-canonical-10-token-logits-r12"
    assert list(CANONICAL_FIXED_TOKEN_IDS) == [
        128000,
        791,
        1489,
        374,
        264,
        1296,
        315,
        872,
        1344,
        13,
    ]
    assert CANONICAL_FIXED_TOKEN_IDS_SHA256 == (
        "7efecfa934a69fc22e9cba559b9547061cc3a0f58a7bbaba256d6df41a335909"
    )
    assert token_ids_sha256(list(CANONICAL_FIXED_TOKEN_IDS)) == (
        CANONICAL_FIXED_TOKEN_IDS_SHA256
    )
    assert CANONICAL_LOGITS_SHAPE == (1, 10, 128256)
    assert CANONICAL_PARITY_TOLERANCES.to_dict() == {
        "logit_atol": 0.125,
        "logit_rtol": 0.02,
        "max_close_failure_count": 1,
        "max_abs_error": 0.5,
        "max_mean_abs_error": 0.03125,
        "max_rms_error": 0.046875,
        "top_k": 10,
        "max_top_1_mismatch_count": 0,
        "min_top_k_intersection_count_per_position": 9,
        "min_top_k_intersection_count_total": 99,
    }
    assert CANONICAL_SEMANTIC_TOLERANCES.to_dict() == {
        "max_mismatched_elements": 0,
        "max_abs_error": 0.0,
    }
    parser = parity_tool.build_parser()
    option_strings = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert "--expected-shards" in option_strings
    expected_shards_action = next(
        action
        for action in parser._actions  # noqa: SLF001
        if "--expected-shards" in action.option_strings
    )
    assert expected_shards_action.required is True
    assert (
        not {
            "--token-ids",
            "--logit-atol",
            "--logit-rtol",
            "--max-abs-error",
            "--top-k",
            "--min-top-k-overlap",
        }
        & option_strings
    )


@pytest.mark.parametrize(
    "route",
    [
        PURE_V5_FUSED,
        LOCALCTA_FUSED,
        MXFP4_FUSED,
        BF16_UNFUSED,
        TE_NATIVE_NVFP4_UNFUSED,
    ],
)
def test_sealed_receipt_validates_all_supported_route_counts(route):
    receipt = seal_receipt(_passing_payload(route))
    validate_receipt(
        receipt,
        expected_bindings={
            "route": route,
            "step": 1,
            "source_job_id": "EXAMPLE",
            "conversion_manifest_sha256": "a" * 64,
        },
    )


def test_parity_route_count_contracts_are_exact():
    assert SUPPORTED_ROUTES == {
        PURE_V5_FUSED,
        LOCALCTA_FUSED,
        MXFP4_FUSED,
        BF16_UNFUSED,
        TE_NATIVE_NVFP4_UNFUSED,
    }
    assert SOURCE_TENSORS_BY_ROUTE == {
        PURE_V5_FUSED: 227,
        LOCALCTA_FUSED: 227,
        MXFP4_FUSED: 227,
        BF16_UNFUSED: 291,
        TE_NATIVE_NVFP4_UNFUSED: 291,
    }
    assert FROZEN_ALIASES_BY_ROUTE == {
        PURE_V5_FUSED: 64,
        LOCALCTA_FUSED: 64,
        MXFP4_FUSED: 0,
        BF16_UNFUSED: 0,
        TE_NATIVE_NVFP4_UNFUSED: 0,
    }


@pytest.mark.parametrize("route", FUSED_ROUTES)
def test_native_parity_independently_defuses_each_fused_route(route):
    parity_tool = _load_parity_tool()
    spec = _tiny_spec()
    qkv = torch.arange(spec.qkv_rows * spec.dim, dtype=torch.float32).reshape(
        spec.qkv_rows, spec.dim
    )
    observed = parity_tool._independent_unfused_tensors(
        "layers.1.attention.fused.w_qkv", qkv, route, spec
    )
    assert list(observed) == [
        "layers.1.attention.wq.weight",
        "layers.1.attention.wk.weight",
        "layers.1.attention.wv.weight",
    ]
    assert torch.equal(observed["layers.1.attention.wq.weight"], qkv[: spec.q_rows])


@pytest.mark.parametrize("route", [PURE_V5_FUSED, LOCALCTA_FUSED])
def test_native_parity_rejects_frozen_alias_in_trainable_stream(route):
    parity_tool = _load_parity_tool()
    spec = _tiny_spec()
    with pytest.raises(RuntimeError, match="frozen alias"):
        parity_tool._independent_unfused_tensors(
            "layers.1.attention_norm.weight",
            torch.ones(spec.dim),
            route,
            spec,
        )


def test_native_parity_treats_te_native_weights_as_standard_unfused_parameters():
    parity_tool = _load_parity_tool()
    spec = _tiny_spec()
    weight = torch.arange(spec.q_rows * spec.dim, dtype=torch.float32).reshape(
        spec.q_rows, spec.dim
    )
    key = "layers.1.attention.wq.weight"
    observed = parity_tool._independent_unfused_tensors(
        key, weight, TE_NATIVE_NVFP4_UNFUSED, spec
    )
    assert list(observed) == [key]
    assert torch.equal(observed[key], weight)


def test_receipt_validation_rejects_tampering_and_failure():
    receipt = seal_receipt(_passing_payload())
    tampered = dict(receipt)
    tampered["step"] = 0
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        validate_receipt(tampered)

    reference = _canonical_reference()
    converted = reference.clone()
    converted[0, 0, 100] += 0.4
    converted[0, 1, 100] += 0.4
    failed_measurements = compare_logits(
        reference, converted, CANONICAL_PARITY_TOLERANCES
    )
    failed_payload = _passing_payload()
    failed_payload["passed"] = False
    failed_payload["measurements"] = dict(failed_payload["measurements"])
    failed_payload["measurements"]["passed"] = False
    failed_payload["measurements"]["stock_hf_evaluator_drift"] = failed_measurements
    with pytest.raises(RuntimeError, match="not a passing receipt"):
        validate_receipt(seal_receipt(failed_payload))


def test_receipt_policy_token_shape_and_expected_binding_fail_closed():
    receipt = seal_receipt(_passing_payload())
    with pytest.raises(RuntimeError, match="binding mismatch"):
        validate_receipt(receipt, expected_bindings={"route": BF16_UNFUSED})

    wrong_policy = _passing_payload()
    wrong_policy["policy"] = "different-policy"
    with pytest.raises(RuntimeError, match="unexpected parity policy"):
        validate_receipt(seal_receipt(wrong_policy))

    wrong_tokens = _passing_payload()
    wrong_tokens["fixed_token_ids"] = list(CANONICAL_FIXED_TOKEN_IDS)
    wrong_tokens["fixed_token_ids"][1] += 1
    wrong_tokens["fixed_token_ids_sha256"] = token_ids_sha256(
        wrong_tokens["fixed_token_ids"]
    )
    with pytest.raises(RuntimeError, match="canonical r12 tokens"):
        validate_receipt(seal_receipt(wrong_tokens))

    wrong_expected_shape = _passing_payload()
    wrong_expected_shape["expected_logits_shape"] = [1, 9, 128256]
    with pytest.raises(RuntimeError, match="expected logits shape"):
        validate_receipt(seal_receipt(wrong_expected_shape))

    wrong_measured_shape = _passing_payload()
    wrong_measured_shape["measurements"] = dict(wrong_measured_shape["measurements"])
    wrong_measured_shape["measurements"]["stock_hf_evaluator_drift"] = dict(
        wrong_measured_shape["measurements"]["stock_hf_evaluator_drift"]
    )
    wrong_measured_shape["measurements"]["stock_hf_evaluator_drift"]["logits_shape"] = [
        1,
        9,
        128256,
    ]
    with pytest.raises(RuntimeError, match="stock-HF logits shape"):
        validate_receipt(seal_receipt(wrong_measured_shape))

    wrong_tolerances = _passing_payload()
    wrong_tolerances["tolerances"] = dict(wrong_tolerances["tolerances"])
    wrong_tolerances["tolerances"]["stock_hf_evaluator_drift"] = dict(
        wrong_tolerances["tolerances"]["stock_hf_evaluator_drift"]
    )
    wrong_tolerances["tolerances"]["stock_hf_evaluator_drift"]["logit_atol"] = 0.126
    with pytest.raises(RuntimeError, match="stock-HF tolerances"):
        validate_receipt(seal_receipt(wrong_tolerances))


@pytest.mark.parametrize(
    "container", ["top", "environment", "tolerances", "measurements"]
)
def test_receipt_rejects_extra_fields_in_every_exact_inventory(container):
    payload = _passing_payload()
    if container == "top":
        payload["unexpected"] = True
        message = "receipt field inventory"
    else:
        payload[container] = dict(payload[container])
        payload[container]["unexpected"] = True
        message = f"{container.rstrip('s')}.*field inventory"
    with pytest.raises(RuntimeError, match=message):
        validate_receipt(seal_receipt(payload))


def test_receipt_exact_environment_and_measurement_inventories_are_frozen():
    payload = _passing_payload()
    assert set(payload["environment"]) == PARITY_ENVIRONMENT_FIELDS
    assert set(payload["tolerances"]) == PARITY_TOLERANCE_FIELDS
    assert set(payload["measurements"]) == PARITY_MEASUREMENT_FIELDS
    assert (
        set(payload["measurements"]["canonical_semantic"])
        == PARITY_SEMANTIC_MEASUREMENT_FIELDS
    )
    assert (
        set(payload["measurements"]["stock_hf_evaluator_drift"])
        == PARITY_LOGIT_MEASUREMENT_FIELDS
    )


@pytest.mark.parametrize(
    ("outer", "inner", "message"),
    [
        ("tolerances", "canonical_semantic", "semantic tolerance inventory"),
        (
            "tolerances",
            "stock_hf_evaluator_drift",
            "stock-HF tolerance inventory",
        ),
        (
            "measurements",
            "canonical_semantic",
            "semantic measurement inventory",
        ),
        (
            "measurements",
            "stock_hf_evaluator_drift",
            "stock-HF measurement inventory",
        ),
    ],
)
def test_receipt_rejects_extra_fields_in_nested_inventories(outer, inner, message):
    payload = _passing_payload()
    payload[outer] = dict(payload[outer])
    payload[outer][inner] = dict(payload[outer][inner])
    payload[outer][inner]["unexpected"] = True
    with pytest.raises(RuntimeError, match=message):
        validate_receipt(seal_receipt(payload))


def test_receipt_recomputes_semantic_and_combined_verdicts():
    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["canonical_semantic"] = dict(
        payload["measurements"]["canonical_semantic"]
    )
    payload["measurements"]["canonical_semantic"]["passed"] = False
    with pytest.raises(RuntimeError, match="semantic verdict does not recompute"):
        validate_receipt(seal_receipt(payload))

    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["passed"] = False
    with pytest.raises(RuntimeError, match="combined measurement verdict"):
        validate_receipt(seal_receipt(payload))


def test_receipt_rejects_semantic_tolerance_hash_and_environment_drift():
    payload = _passing_payload()
    payload["tolerances"] = dict(payload["tolerances"])
    payload["tolerances"]["canonical_semantic"] = dict(
        payload["tolerances"]["canonical_semantic"]
    )
    payload["tolerances"]["canonical_semantic"]["max_abs_error"] = 0.125
    with pytest.raises(RuntimeError, match="semantic tolerances are not canonical"):
        validate_receipt(seal_receipt(payload))

    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["canonical_semantic"] = dict(
        payload["measurements"]["canonical_semantic"]
    )
    payload["measurements"]["canonical_semantic"]["canonical_logits_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="exact semantic logits hashes disagree"):
        validate_receipt(seal_receipt(payload))

    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["canonical_semantic"] = dict(
        payload["measurements"]["canonical_semantic"]
    )
    semantic = payload["measurements"]["canonical_semantic"]
    element_count = semantic["logit_element_count"]
    semantic.update(
        {
            "passed": False,
            "strict_bit_exact": False,
            "exact_match_count": element_count - 1,
            "mismatch_count": 1,
            "exact_match_ratio": (element_count - 1) / element_count,
            "mismatch_ratio": 1 / element_count,
            "max_abs_error": 0.125,
            "mean_abs_error": 0.0001,
            "rms_error": 0.001,
        }
    )
    payload["measurements"]["passed"] = False
    payload["passed"] = False
    assert semantic["reference_logits_sha256"] == semantic["canonical_logits_sha256"]
    with pytest.raises(RuntimeError, match="equal semantic logits hashes contradict"):
        validate_receipt(seal_receipt(payload))

    payload = _passing_payload()
    payload["environment"] = dict(payload["environment"])
    payload["environment"]["stock_hf_rope"] = "drifted"
    with pytest.raises(RuntimeError, match="environment mismatch for stock_hf_rope"):
        validate_receipt(seal_receipt(payload))


def test_receipt_recomputes_exact_count_and_ratio_summaries():
    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["stock_hf_evaluator_drift"] = dict(
        payload["measurements"]["stock_hf_evaluator_drift"]
    )
    measurements = payload["measurements"]["stock_hf_evaluator_drift"]
    measurements["close_failure_count"] = 1
    measurements["close_success_count"] -= 1
    measurements["strict_allclose"] = False
    with pytest.raises(RuntimeError, match="inconsistent with its exact counts"):
        validate_receipt(seal_receipt(payload))

    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["stock_hf_evaluator_drift"] = dict(
        payload["measurements"]["stock_hf_evaluator_drift"]
    )
    stock = payload["measurements"]["stock_hf_evaluator_drift"]
    stock["top_k_intersection_counts"] = [9] + [10] * 9
    stock["top_k_intersection_ratios"] = [0.9] + [1.0] * 9
    with pytest.raises(RuntimeError, match="intersection summaries"):
        validate_receipt(seal_receipt(payload))

    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["stock_hf_evaluator_drift"] = dict(
        payload["measurements"]["stock_hf_evaluator_drift"]
    )
    payload["measurements"]["stock_hf_evaluator_drift"]["passed"] = False
    with pytest.raises(RuntimeError, match="stock-HF verdict does not recompute"):
        validate_receipt(seal_receipt(payload))


@pytest.mark.parametrize(
    "contradiction",
    ["nonzero_error", "top_1_mismatch", "top_k_mismatch"],
)
def test_receipt_rejects_equal_stock_hashes_with_nonexact_drift(contradiction):
    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["stock_hf_evaluator_drift"] = dict(
        payload["measurements"]["stock_hf_evaluator_drift"]
    )
    stock = payload["measurements"]["stock_hf_evaluator_drift"]
    if contradiction == "nonzero_error":
        element_count = stock["logit_element_count"]
        stock.update(
            {
                "strict_allclose": False,
                "close_success_count": element_count - 1,
                "close_failure_count": 1,
                "close_success_ratio": (element_count - 1) / element_count,
                "close_failure_ratio": 1 / element_count,
                "max_abs_error": 0.4,
                "mean_abs_error": 0.001,
                "rms_error": 0.01,
                "max_relative_error": 0.4,
            }
        )
    elif contradiction == "top_1_mismatch":
        position_count = stock["position_count"]
        stock.update(
            {
                "top_1_match_count": position_count - 1,
                "top_1_mismatch_count": 1,
                "top_1_agreement_ratio": (position_count - 1) / position_count,
            }
        )
    else:
        top_k = stock["top_k"]
        position_count = stock["position_count"]
        intersection_counts = [top_k - 1] + [top_k] * (position_count - 1)
        intersection_total = sum(intersection_counts)
        stock.update(
            {
                "top_k_intersection_counts": intersection_counts,
                "top_k_intersection_ratios": [
                    count / top_k for count in intersection_counts
                ],
                "top_k_intersection_count_min": top_k - 1,
                "top_k_intersection_count_total": intersection_total,
                "top_k_intersection_ratio_min": (top_k - 1) / top_k,
                "top_k_intersection_ratio_total": (
                    intersection_total / stock["top_k_membership_count"]
                ),
            }
        )
    assert stock["reference_logits_sha256"] == stock["converted_logits_sha256"]
    with pytest.raises(RuntimeError, match="equal stock logits hashes contradict"):
        validate_receipt(seal_receipt(payload))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"max_abs_error": float("nan")}, "finite nonnegative float"),
        ({"max_abs_error": 0.500001}, "verdict does not recompute"),
        (
            {
                "max_abs_error": 0.5,
                "mean_abs_error": 0.031251,
                "rms_error": 0.031251,
            },
            "verdict does not recompute",
        ),
        (
            {"max_abs_error": 0.5, "rms_error": 0.046876},
            "verdict does not recompute",
        ),
        ({"native_parameters_loaded": 290}, "exact-state measurement mismatch"),
    ],
)
def test_receipt_recomputes_measurement_acceptance(updates, message):
    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["stock_hf_evaluator_drift"] = dict(
        payload["measurements"]["stock_hf_evaluator_drift"]
    )
    stock_updated = False
    for field, value in updates.items():
        target = (
            payload["measurements"]["stock_hf_evaluator_drift"]
            if field in PARITY_LOGIT_MEASUREMENT_FIELDS
            else payload["measurements"]
        )
        target[field] = value
        stock_updated = stock_updated or field in PARITY_LOGIT_MEASUREMENT_FIELDS
    if stock_updated:
        payload["measurements"]["stock_hf_evaluator_drift"][
            "converted_logits_sha256"
        ] = ("e" * 64)
    with pytest.raises(RuntimeError, match=message):
        validate_receipt(seal_receipt(payload))


@pytest.mark.parametrize(
    ("field", "cap", "companion_updates"),
    [
        ("max_abs_error", 0.5, {}),
        (
            "mean_abs_error",
            0.03125,
            {"max_abs_error": 0.5, "rms_error": 0.03125},
        ),
        ("rms_error", 0.046875, {"max_abs_error": 0.5}),
    ],
)
def test_receipt_numeric_caps_accept_boundary_and_reject_nextafter(
    field, cap, companion_updates
):
    payload = _passing_payload()
    payload["measurements"] = dict(payload["measurements"])
    payload["measurements"]["stock_hf_evaluator_drift"] = dict(
        payload["measurements"]["stock_hf_evaluator_drift"]
    )
    stock = payload["measurements"]["stock_hf_evaluator_drift"]
    stock["converted_logits_sha256"] = "e" * 64
    stock.update(companion_updates)
    stock[field] = cap
    validate_receipt(seal_receipt(payload))

    stock[field] = math.nextafter(cap, math.inf)
    if field == "mean_abs_error":
        stock["rms_error"] = stock[field]
    with pytest.raises(RuntimeError, match="verdict does not recompute"):
        validate_receipt(seal_receipt(payload))


def test_receipt_rejects_old_schema_missing_field_and_boolean_count():
    old_schema = _passing_payload()
    old_schema["schema_version"] = 1
    with pytest.raises(RuntimeError, match="unsupported parity receipt schema"):
        validate_receipt(seal_receipt(old_schema))

    missing = _passing_payload()
    del missing["policy"]
    with pytest.raises(RuntimeError, match="field inventory is not exact"):
        validate_receipt(seal_receipt(missing))

    boolean_count = _passing_payload()
    boolean_count["measurements"] = dict(boolean_count["measurements"])
    boolean_count["measurements"]["stock_hf_evaluator_drift"] = dict(
        boolean_count["measurements"]["stock_hf_evaluator_drift"]
    )
    boolean_count["measurements"]["stock_hf_evaluator_drift"][
        "close_failure_count"
    ] = True
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        validate_receipt(seal_receipt(boolean_count))


def test_receipt_recomputes_code_bundle_binding():
    payload = _passing_payload()
    payload["code_files_sha256"] = dict(payload["code_files_sha256"])
    payload["code_files_sha256"]["parity_tool"] = "0" * 64
    with pytest.raises(RuntimeError, match="code-bundle binding"):
        validate_receipt(seal_receipt(payload))


def test_receipt_recomputes_parity_tool_binding():
    payload = _passing_payload()
    payload["tool_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="parity-tool binding"):
        validate_receipt(seal_receipt(payload))


def test_atomic_receipt_publication_never_overwrites(tmp_path):
    path = tmp_path / "parity.json"
    receipt = seal_receipt(_passing_payload())
    write_atomic_receipt(path, receipt)
    assert json.loads(path.read_text(encoding="utf-8")) == receipt
    assert not list(tmp_path.glob(".parity.json.incomplete.*"))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_atomic_receipt(path, receipt)
