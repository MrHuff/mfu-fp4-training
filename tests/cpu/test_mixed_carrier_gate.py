from __future__ import annotations

import json

import pytest
import torch

from low_bits_training.analysis import mixed_carrier_gate as gate


def test_receipt_seal_is_canonical_and_tamper_evident(tmp_path) -> None:
    left = gate.seal_receipt({"z": 1, "a": [2, 3]})
    right = gate.seal_receipt({"a": [2, 3], "z": 1})
    assert left == right
    gate.validate_receipt_seal(left)

    output = tmp_path / "receipt.json"
    gate.write_receipt_exclusive(output, left)
    assert json.loads(output.read_text()) == left
    with pytest.raises(FileExistsError):
        gate.write_receipt_exclusive(output, left)

    tampered = dict(left)
    tampered["z"] = 2
    with pytest.raises(gate.GateFailure, match="seal mismatch"):
        gate.validate_receipt_seal(tampered)


def test_packed_payload_exactness_uses_stored_bytes() -> None:
    reference = torch.arange(32, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    candidate = reference.view(torch.uint8).clone().view(torch.float4_e2m1fn_x2)
    report = gate.require_exact_tensors({"fp4": candidate}, {"fp4": reference})
    assert report["pass"]
    candidate.view(torch.uint8)[7] ^= 1
    with pytest.raises(gate.GateFailure, match="bit-exact"):
        gate.require_exact_tensors({"fp4": candidate}, {"fp4": reference})


def test_strict_close_has_explicit_zero_and_nonzero_tolerance() -> None:
    reference = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    candidate = reference.clone()
    assert gate.strict_close_report(
        candidate, reference, name="dhidden", atol=0.0, rtol=0.0
    )["pass"]
    candidate[0] = torch.tensor(1.015625, dtype=torch.bfloat16)
    with pytest.raises(gate.GateFailure, match="exceeded"):
        gate.strict_close_report(
            candidate, reference, name="dhidden", atol=0.0, rtol=0.0
        )
    assert gate.strict_close_report(
        candidate, reference, name="dhidden", atol=0.02, rtol=0.0
    )["pass"]


def test_storage_audit_rejects_accidental_alias() -> None:
    base = torch.arange(64, dtype=torch.uint8)
    with pytest.raises(gate.GateFailure, match="aliases"):
        gate.require_disjoint_payload_storage(
            {"row": base[:32], "col": base[16:48]}
        )
    assert gate.require_disjoint_payload_storage(
        {"row": base[:32], "col": base[32:]}
    )["pass"]


def test_split2_outer_scales_must_be_independent_and_range_specific() -> None:
    left = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    right = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    report = gate.require_distinct_scale_carriers(
        left,
        right,
        left_name="row_sg0",
        right_name="row_sg1",
        require_distinct_values=True,
    )
    assert report["pass"]
    assert not report["byte_equal"]

    with pytest.raises(gate.GateFailure, match="same tensor object"):
        gate.require_distinct_scale_carriers(
            left,
            left,
            left_name="row_sg0",
            right_name="row_sg1",
            require_distinct_values=False,
        )
    with pytest.raises(gate.GateFailure, match="byte-identical"):
        gate.require_distinct_scale_carriers(
            left,
            left.clone(),
            left_name="row_sg0",
            right_name="row_sg1",
            require_distinct_values=True,
        )


def test_sr_counter_requires_exactly_one_format_agnostic_reservation() -> None:
    assert gate.require_one_sr_advance((9, 17), (9, 17 + (1 << 32)))["pass"]
    with pytest.raises(gate.GateFailure, match="exactly once"):
        gate.require_one_sr_advance((9, 17), (9, 17))
    with pytest.raises(gate.GateFailure, match="exactly once"):
        gate.require_one_sr_advance((9, 17), (9, 17 + 2 * (1 << 32)))


def test_logical_gemm_audit_counts_semantics() -> None:
    audit = gate.LogicalGemmAudit({"forward": 1, "dhidden": 1, "dweight": 1})

    def identity(value):
        return value

    for label in ("forward", "dhidden", "dweight"):
        assert audit.wrap(label, identity)(3) == 3
    assert audit.report()["observed_total"] == 3

    extra = gate.LogicalGemmAudit({"forward": 1})
    wrapped = extra.wrap("forward", identity)
    wrapped(1)
    wrapped(1)
    with pytest.raises(gate.GateFailure, match="count mismatch"):
        extra.report()


def test_timing_summary_and_comparison() -> None:
    summary = gate.summarize_ms([3.0, 1.0, 2.0])
    assert summary["median_ms"] == 2.0
    comparison = gate.timing_comparison([1.0, 1.0], [1.02, 1.02])
    assert comparison["median_ratio"] == pytest.approx(1.02)
    assert comparison["median_overhead_pct"] == pytest.approx(2.0)
