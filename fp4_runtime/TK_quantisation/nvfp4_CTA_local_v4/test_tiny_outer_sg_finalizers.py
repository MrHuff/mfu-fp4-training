#!/usr/bin/env python3
"""Regress exact-zero-safe localCTA outer-scale finalization."""

import os
from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_localcta_v4 as tkq


_CONTRACT_ENV = "USE_TK_LOCALCTA_V3_CONTRACT"
_SCALE_NUM = 448.0
_SIZE = 512
_TINY_RMS = 1.0e-10
_LOG_MAGNITUDES = (1.0e-12, 1.0e-10, 5.0e-10, 1.0e-8)
_GAIN_RANGE = (0.90, 1.10)
_MAX_REL_L2 = 0.15
_MIN_COSINE = 0.99

_TEST_ENV = {
    "USE_TK_LOCALCTA_V4_FUSED_ATOMIC_INIT": "1",
    "USE_TK_LOCALCTA_V4_FUSED_ATOMIC_INIT_THREADS": "64",
    "USE_TK_LOCALCTA_V4_REUSE_ATOMIC_SCRATCH": "1",
}
_RESCALE_ENV = (
    "USE_TK_LOCALCTA_V4_FINAL_SG_RESCALE_COLS_PER_BLOCK",
    "USE_TK_LOCALCTA_V4_FINAL_SG_RESCALE_WARP",
)


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _set_atomic_rescaler(mode: str) -> None:
    if mode == "scalar":
        os.environ[_RESCALE_ENV[0]] = "0"
        os.environ[_RESCALE_ENV[1]] = "1"
    elif mode == "vector":
        os.environ[_RESCALE_ENV[0]] = "16"
        os.environ[_RESCALE_ENV[1]] = "0"
    elif mode == "warp":
        os.environ[_RESCALE_ENV[0]] = "16"
        os.environ[_RESCALE_ENV[1]] = "1"
    else:
        raise AssertionError(f"unknown atomic rescaler mode: {mode}")


def _quantize_atomic(input_: torch.Tensor, mode: str):
    _set_atomic_rescaler(mode)
    return tkq.tk_localcta_quantize_for_gemm_atomic_final_sg(
        input_, True, True
    )


def _quantize_final_sg_opt(input_: torch.Tensor):
    return tkq.tk_localcta_quantize_for_gemm_final_sg_opt(
        input_, True, True
    )


def _producer_cases(input_: torch.Tensor):
    for mode in ("scalar", "vector", "warp"):
        yield f"atomic-{mode}", _quantize_atomic(input_, mode)
    yield "final-sg-opt", _quantize_final_sg_opt(input_)


def _decoded_axes(input_: torch.Tensor, outputs):
    row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg = outputs
    for tensor in (row_sc, col_sc, row_sg, col_sg):
        assert torch.isfinite(tensor.float()).all()
    yield (
        "row",
        input_.float(),
        tkq.tk_localcta_reconstruct_row(row_fp4, row_sc, row_sg).float(),
        row_sg,
    )
    yield (
        "col",
        input_.t().float(),
        tkq.tk_localcta_reconstruct_col(col_fp4, col_sc, col_sg).float(),
        col_sg,
    )


def _assert_gain(name: str, reference: torch.Tensor, decoded: torch.Tensor) -> None:
    reference64 = reference.double().flatten()
    decoded64 = decoded.double().flatten()
    reference_norm = torch.linalg.vector_norm(reference64)
    decoded_norm = torch.linalg.vector_norm(decoded64)
    gain = torch.dot(reference64, decoded64) / torch.dot(reference64, reference64)
    rel_l2 = torch.linalg.vector_norm(decoded64 - reference64) / reference_norm
    cosine = torch.dot(reference64, decoded64) / (reference_norm * decoded_norm)
    assert _GAIN_RANGE[0] <= gain <= _GAIN_RANGE[1], (name, gain)
    assert rel_l2 <= _MAX_REL_L2, (name, rel_l2)
    assert cosine >= _MIN_COSINE, (name, cosine)


def _assert_log_sweep() -> None:
    for magnitude in _LOG_MAGNITUDES:
        input_ = torch.full(
            (_SIZE, _SIZE), magnitude, device="cuda", dtype=torch.bfloat16
        )
        exact_amax = input_.float().abs().amax()
        for producer, outputs in _producer_cases(input_):
            for axis, _, decoded, _ in _decoded_axes(input_, outputs):
                ratio_min = decoded.abs().amin() / exact_amax
                ratio_max = decoded.abs().amax() / exact_amax
                label = f"{producer}/{axis}/magnitude={magnitude:.1e}"
                assert torch.isfinite(decoded).all(), label
                assert _GAIN_RANGE[0] <= ratio_min, (label, ratio_min)
                assert ratio_max <= _GAIN_RANGE[1], (label, ratio_max)


def _assert_signed_normal() -> None:
    input_ = (
        torch.randn((_SIZE, _SIZE), device="cuda", dtype=torch.float32)
        .mul_(_TINY_RMS)
        .to(torch.bfloat16)
    )
    for producer, outputs in _producer_cases(input_):
        for axis, reference, decoded, _ in _decoded_axes(input_, outputs):
            assert torch.isfinite(decoded).all(), (producer, axis)
            _assert_gain(f"{producer}/{axis}/signed-normal", reference, decoded)


def _assert_exact_zero() -> None:
    input_ = torch.zeros(
        (_SIZE, _SIZE), device="cuda", dtype=torch.bfloat16
    )
    for producer, outputs in _producer_cases(input_):
        for axis, _, decoded, outer_sg in _decoded_axes(input_, outputs):
            label = f"{producer}/{axis}/zero"
            assert torch.isfinite(decoded).all(), label
            assert torch.count_nonzero(decoded) == 0, label
            assert torch.count_nonzero(outer_sg) == 0, label


def main() -> None:
    torch.manual_seed(7)
    env_names = {_CONTRACT_ENV, *_TEST_ENV, *_RESCALE_ENV}
    previous = {name: os.environ.get(name) for name in env_names}
    try:
        os.environ.pop(_CONTRACT_ENV, None)
        os.environ.update(_TEST_ENV)
        tkq.tk_localcta_set_global_scale_num(_SCALE_NUM)
        _assert_log_sweep()
        _assert_signed_normal()
        _assert_exact_zero()
    finally:
        tkq.tk_localcta_reset_global_scale_num()
        _restore_env(previous)

    print(
        "localCTA tiny outer-SG finalizers: "
        "atomic-scalar=PASS atomic-vector=PASS atomic-warp=PASS "
        "final-sg-opt=PASS log-sweep=PASS signed-normal=PASS zero=PASS"
    )


if __name__ == "__main__":
    main()
