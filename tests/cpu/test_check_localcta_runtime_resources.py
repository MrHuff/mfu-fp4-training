import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "nvl72"))

import check_localcta_runtime_resources as resources  # noqa: E402


def _dump(registers: int, stack_bytes: int) -> str:
    return f"""
Fatbin elf code:
================
arch = sm_100a
 Function _ZN11tk_localcta{resources.HOT_SILU_KERNEL_MARKER}EEv:
  REG:{registers} STACK:{stack_bytes} SHARED:1104 LOCAL:0 CONSTANT[0]:1736
"""


def test_accepts_the_validated_high_occupancy_cubin():
    parsed = resources.validate_hot_kernel_resources(_dump(54, 0))

    assert len(parsed) == 1
    assert parsed[0].registers == 54
    assert parsed[0].stack_bytes == 0
    assert parsed[0].local_bytes == 0


def test_rejects_the_one_cta_register_regression():
    with pytest.raises(ValueError, match="194 registers exceeds the limit of 64"):
        resources.validate_hot_kernel_resources(_dump(194, 0))


def test_rejects_a_register_cap_that_spills_to_thread_local_stack():
    with pytest.raises(ValueError, match="432 stack bytes exceeds the limit of 16"):
        resources.validate_hot_kernel_resources(_dump(64, 432))


def test_accepts_the_stack_free_two_cta_resource_envelope():
    parsed = resources.validate_hot_kernel_resources(
        _dump(96, 0), max_registers=128, max_stack_bytes=0
    )

    assert len(parsed) == 1
    assert parsed[0].registers == 96
    assert parsed[0].stack_bytes == 0


def test_two_cta_resource_envelope_rejects_any_stack_frame():
    with pytest.raises(ValueError, match="1 stack bytes exceeds the limit of 0"):
        resources.validate_hot_kernel_resources(
            _dump(96, 1), max_registers=128, max_stack_bytes=0
        )


def test_two_cta_resource_envelope_rejects_register_drift():
    with pytest.raises(ValueError, match="129 registers exceeds the limit of 128"):
        resources.validate_hot_kernel_resources(
            _dump(129, 0), max_registers=128, max_stack_bytes=0
        )


def test_accepts_exact_explicit_sm100_stack_ceiling_without_relaxing_registers():
    parsed = resources.validate_hot_kernel_resources(
        _dump(64, 432), max_stack_bytes=432
    )

    assert len(parsed) == 1
    assert parsed[0].registers == 64
    assert parsed[0].stack_bytes == 432


def test_exact_explicit_sm100_stack_ceiling_still_rejects_drift():
    with pytest.raises(ValueError, match="433 stack bytes exceeds the limit of 432"):
        resources.validate_hot_kernel_resources(
            _dump(64, 433), max_stack_bytes=432
        )


def test_rejects_a_dump_without_the_production_silu_kernel():
    with pytest.raises(ValueError, match="hot localCTA SiLU quantizer was not found"):
        resources.validate_hot_kernel_resources(
            " Function unrelated_kernel:\n  REG:32 STACK:0 SHARED:0\n"
        )


def test_rejects_thread_local_memory_even_with_zero_stack():
    with pytest.raises(ValueError, match="8 local bytes exceeds the limit of 0"):
        resources.validate_hot_kernel_resources(
            _dump(54, 0).replace("LOCAL:0", "LOCAL:8")
        )


def test_rejects_duplicate_exact_specializations():
    with pytest.raises(
        ValueError, match="expected exactly one hot localCTA SiLU quantizer; found 2"
    ):
        resources.validate_hot_kernel_resources(_dump(54, 0) + _dump(54, 0))


def test_selects_only_the_paired_rht_specialization_from_a_mixed_dump():
    legacy_marker = (
        "39fused_localcta_silu_quantize_raw_kernel"
        "ILb1ELb1ELi1ELb0ELb1ELb1ELb0ELb0ELb1E"
    )
    mixed = f"""
 Function _ZN11tk_localcta{legacy_marker}Ev:
  REG:54 STACK:0 SHARED:1104 LOCAL:0 CONSTANT[0]:1736
{_dump(64, 432)}
"""

    parsed = resources.validate_hot_kernel_resources(
        mixed, max_stack_bytes=432
    )

    assert len(parsed) == 1
    assert parsed[0].registers == 64
    assert parsed[0].stack_bytes == 432
    assert parsed[0].local_bytes == 0
