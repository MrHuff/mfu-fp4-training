#!/usr/bin/env python3
"""Exact CUDA oracles for the heterogeneous MXFP4/localCTA carriers."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parent
MX_ROOT = Path(os.environ.get("MIXED_MX_MODULE_DIR", HERE.parent / "mxfp4_v4"))
os.environ["USE_TK_LOCALCTA_V3_CONTRACT"] = "outer"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MX_ROOT))

import _tk_quant_localcta_v4 as mixed
import mxfp4_quant_v4 as mx


SEED = 20260831
SUBSEQUENCE = 0x13579


def _bytes(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.uint8)


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: shape/dtype {actual.shape}/{actual.dtype} != "
            f"{expected.shape}/{expected.dtype}"
        )
    actual_bytes = _bytes(actual)
    expected_bytes = _bytes(expected)
    mismatches = int(torch.count_nonzero(actual_bytes != expected_bytes))
    if mismatches:
        raise AssertionError(
            f"{name}: {mismatches}/{actual_bytes.numel()} bytes differ"
        )


def _input(rows: int, cols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    value = torch.randn(
        (rows, cols), device="cuda", dtype=torch.float32,
        generator=generator,
    )
    value.mul_(0.125)
    value[0, 0] = 64.0
    value[-1, -1] = -32.0
    return value.to(torch.bfloat16).contiguous()


def _local_row_reference(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    state = torch.tensor(
        [SEED, SUBSEQUENCE], device=value.device, dtype=torch.int64
    )
    return tuple(mixed.tk_localcta_quantize_for_gemm_opt(
        value,
        True,
        True,
        True,
        False,
        "none",
        False,
        SEED,
        SUBSEQUENCE,
        "row",
        state,
    ))


def _mx_col_reference(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(mx.mxfp4_quantize_col_only_opt_rht(
        value,
        1,
        False,
        False,
        32,
        True,
        0,
        0,
    ))


def _logical_localcta_split2_reference(
    grad0: torch.Tensor, grad1: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Independent no-RHT row-SR oracle with split2's logical coordinates.

    Each arm is placed at its true position in a zero-padded logical
    ``[grad0 | grad1]`` matrix and quantized by the ordinary one-input localCTA
    producer.  The padding preserves the global chunk/RNG coordinates while
    making the outer-scale reduction depend on that arm alone.  This is a
    test-only materialization; no split2 or column-RHT implementation is used.
    """
    m, h = grad0.shape
    result = []
    for arm, grad in enumerate((grad0, grad1)):
        logical = torch.zeros((m, 2 * h), device=grad.device, dtype=grad.dtype)
        logical.narrow(1, arm * h, h).copy_(grad)
        state = torch.tensor(
            [SEED, SUBSEQUENCE], device=grad.device, dtype=torch.int64
        )
        full = tuple(mixed.tk_localcta_quantize_for_gemm_opt(
            logical,
            True,
            True,
            True,
            False,
            "none",
            False,
            SEED,
            SUBSEQUENCE,
            "row",
            state,
        ))
        result.append((
            full[0].view(torch.uint8).narrow(1, arm * (h // 2), h // 2),
            full[1].narrow(1, arm * (h // 64), h // 64),
            full[4],
        ))
    return tuple(result)


def test_capability_contract() -> None:
    capability = dict(mixed.tk_mixed_mx_localcta_capabilities())
    expected = {
        "abi_version": 1,
        "grad_coordinate_mode": "explicit_seed_subsequence",
        "grad_mx_col_rht": "block32_fixed_0x2817",
        "mxfp4_rht_block_size": 32,
        "mxfp4_rht_sign_contract": "fixed_0x2817_per_h16_half",
        "grad_localcta_row_sr": True,
        "grad_scale_sr": False,
        "weight_mx_2d": True,
        "weight_localcta_2d": True,
        "localcta_encode_mode": "encode_centric",
        "localcta_sg_contract": "outer",
        "single_bf16_tile_load": True,
        "runtime_advances_rng": False,
        "split2_grad_one_coordinate": True,
        "split2_dgrad_onepass_outer_sg": True,
        "split2_row_outer_sg": "per_arm",
    }
    for key, value in expected.items():
        assert capability.get(key) == value, (key, capability.get(key), value)


def test_grad_carrier_exact() -> None:
    value = _input(256, 512, 101)
    buffers = mixed.tk_mixed_grad_localcta_row_mx_col_alloc(
        *value.shape, value.device
    )
    actual = tuple(mixed.tk_mixed_grad_localcta_row_mx_col_launch_inplace(
        value, *buffers, SEED, SUBSEQUENCE
    ))
    local = _local_row_reference(value)
    mx_col = _mx_col_reference(value)
    for name, got, want in (
        ("local.row.fp4", actual[0], local[0]),
        ("local.row.scale", actual[1], local[1]),
        ("local.row.outer", actual[2], local[4]),
        ("mx.col.fp4", actual[3], mx_col[0]),
        ("mx.col.scale", actual[4], mx_col[1]),
    ):
        _assert_exact(name, got, want)


@pytest.mark.parametrize(
    ("m", "h", "arm_scale"),
    (
        (256, 256, 64.0),
        (256, 512, 32.0),
        (512, 256, 96.0),
        (512, 512, 128.0),
    ),
)
def test_split2_grad_carrier_exact_without_bf16_concat(
    m: int, h: int, arm_scale: float
) -> None:
    grad0 = _input(m, h, 201 + m + h)
    # Deliberately give the second arm a very different range.  Equal arm
    # extrema can hide the distinction between split2's per-arm row outer
    # scales and the shared-row-SG contract used by the column-RHT producer.
    grad1 = (_input(m, h, 203 + m + 2 * h).float() / arm_scale).to(
        torch.bfloat16
    ).contiguous()
    buffers = mixed.tk_mixed_split2_grad_localcta_row_mx_col_alloc(
        grad0.shape[0], grad0.shape[1], grad0.device
    )
    actual = tuple(
        mixed.tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace(
            grad0, grad1, *buffers, SEED, SUBSEQUENCE
        )
    )
    logical = torch.cat((grad0, grad1), dim=1)
    local_arms = _logical_localcta_split2_reference(grad0, grad1)
    mx_col = _mx_col_reference(logical)
    for name, got, want in (
        ("split2.local.row.fp4.arm0",
         actual[0].view(torch.uint8).narrow(1, 0, h // 2),
         local_arms[0][0].view(torch.uint8)),
        ("split2.local.row.fp4.arm1",
         actual[0].view(torch.uint8).narrow(1, h // 2, h // 2),
         local_arms[1][0].view(torch.uint8)),
        ("split2.local.row.scale.arm0",
         actual[1].narrow(1, 0, h // 64), local_arms[0][1]),
        ("split2.local.row.scale.arm1",
         actual[1].narrow(1, h // 64, h // 64), local_arms[1][1]),
        ("split2.local.row.outer.arm0", actual[2], local_arms[0][2]),
        ("split2.local.row.outer.arm1", actual[3], local_arms[1][2]),
        ("split2.mx.col.fp4", actual[4], mx_col[0]),
        ("split2.mx.col.scale", actual[5], mx_col[1]),
    ):
        _assert_exact(name, got, want)


def test_weight_carrier_exact() -> None:
    weight = _input(512, 256, 301)
    buffers = mixed.tk_mixed_weight_mx_row_localcta_col_alloc(
        *weight.shape, weight.device
    )
    actual = tuple(mixed.tk_mixed_weight_mx_row_localcta_col_launch_inplace(
        weight, *buffers
    ))
    mx_weight = tuple(mx.mxfp4_quantize_weight_2d(weight))
    local_weight = tuple(mixed.tk_localcta_quantize_weight_2d(weight))
    for name, got, want in (
        ("mx.weight.row.fp4", actual[0], mx_weight[0]),
        ("mx.weight.row.scale", actual[1], mx_weight[1]),
        ("local.weight.col.fp4", actual[2], local_weight[2]),
        ("local.weight.col.scale", actual[3], local_weight[3]),
        ("local.weight.col.outer", actual[4], local_weight[5]),
    ):
        _assert_exact(name, got, want)
