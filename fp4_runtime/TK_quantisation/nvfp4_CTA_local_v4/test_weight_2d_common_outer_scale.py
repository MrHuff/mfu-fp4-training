#!/usr/bin/env python3
"""Numerical gate for the localCTA 2D-weight common-outer-scale contract.

The legacy producer folded each 128x128 chunk's absolute FP32 decode scale
into E4M3 and returned unit outer scales. Weight blocks below E4M3's absolute
minimum nonzero value were therefore inflated. These cases intentionally sit
in that regime and exercise both physical orientations through reconstruction
and the production outer-SG GEMM consumer.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
GEMM_ROOT = Path(
    os.environ.get(
        "LOCALCTA_GEMM_MODULE_DIR",
        REPO_ROOT
        / "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue",
    )
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GEMM_ROOT))

import _tk_quant_localcta_v4 as tkq

GEMM_MODULE_NAME = os.environ.get(
    "LOCALCTA_GEMM_MODULE_NAME", "_C_nv_localcta_gemm"
)
try:
    local_gemm = importlib.import_module(GEMM_MODULE_NAME)
except ModuleNotFoundError as error:
    raise RuntimeError(
        "build the ThunderKittens localCTA GEMM module or set "
        "LOCALCTA_GEMM_MODULE_DIR and LOCALCTA_GEMM_MODULE_NAME"
    ) from error


def _outer_sg_gemm(*args: torch.Tensor) -> None:
    fast = getattr(local_gemm, "nvfp4_localcta_fast_gemm_outer_sg", None)
    if fast is not None:
        fast(*args)
        return
    production = getattr(local_gemm, "nvfp4_localcta_gemm", None)
    if production is None:
        raise RuntimeError(
            f"{GEMM_MODULE_NAME} exposes neither the legacy fast outer-SG "
            "entrypoint nor the production contract-dispatch GEMM"
        )
    production(*args)


GLOBAL_SCALE_NUM = 448.0
SIZE = 256
LAYER = 31
DEPTH_INIT_STD = 0.02 / (2.0 * (LAYER + 1)) ** 0.5


def _patterned(
    rows: int,
    cols: int,
    phase: float,
    amplitude: float,
) -> torch.Tensor:
    index = torch.arange(rows * cols, device="cuda", dtype=torch.float32)
    values = (
        torch.sin(index * 0.017 + phase)
        + 0.31 * torch.cos(index * 0.041 - phase * 0.7)
        + 0.13 * torch.sin(index * 0.0031 + phase * 1.9)
    )
    values = values / values.abs().amax()
    return (values.reshape(rows, cols) * amplitude).to(torch.bfloat16).contiguous()


def _random_normal(
    rows: int,
    cols: int,
    std: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return (
        torch.randn(
            rows,
            cols,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * std
    ).to(torch.bfloat16).contiguous()


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    reference_norm = torch.linalg.vector_norm(reference_f)
    return {
        "rel_l2": float(
            torch.linalg.vector_norm(actual_f - reference_f) / reference_norm
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f.flatten(), reference_f.flatten(), dim=0
            )
        ),
        "norm_ratio": float(
            torch.linalg.vector_norm(actual_f) / reference_norm
        ),
        "max_abs": float(actual_f.abs().amax()),
    }


def _assert_numerics(
    label: str,
    values: dict[str, float],
    *,
    max_rel_l2: float,
    min_cosine: float,
    min_norm_ratio: float,
    max_norm_ratio: float,
) -> None:
    print(
        f"{label}: rel_l2={values['rel_l2']:.8f} "
        f"cosine={values['cosine']:.8f} "
        f"norm_ratio={values['norm_ratio']:.8f} "
        f"max_abs={values['max_abs']:.8e}"
    )
    if values["rel_l2"] > max_rel_l2:
        raise AssertionError(
            f"{label}: relative L2 {values['rel_l2']} exceeds {max_rel_l2}"
        )
    if values["cosine"] < min_cosine:
        raise AssertionError(
            f"{label}: cosine {values['cosine']} is below {min_cosine}"
        )
    if not min_norm_ratio <= values["norm_ratio"] <= max_norm_ratio:
        raise AssertionError(
            f"{label}: norm ratio {values['norm_ratio']} is outside "
            f"[{min_norm_ratio}, {max_norm_ratio}]"
        )


def _compact_common_outer_sg(sg_grid: torch.Tensor, rows: int) -> torch.Tensor:
    compact = sg_grid.reshape(-1)[: rows // 256].contiguous()
    if compact.numel() != rows // 256:
        raise AssertionError("2D-weight SG grid cannot supply the compact outer ABI")
    return compact


def _check_common_outer_contract(
    label: str,
    weight: torch.Tensor,
    quantized: tuple[torch.Tensor, ...],
) -> None:
    row_sg = quantized[4]
    col_sg = quantized[5]
    common = row_sg.reshape(-1)[0]
    if not bool(torch.all(row_sg == common)):
        raise AssertionError(f"{label}: row SG grid is not one common scale")
    if not bool(torch.all(col_sg == common)):
        raise AssertionError(f"{label}: column SG grid differs from row common scale")
    if float(common) <= 0.0:
        raise AssertionError(f"{label}: nonzero weight produced a non-positive outer scale")

    expected = weight.float().abs().amax() / GLOBAL_SCALE_NUM
    if not bool(torch.isclose(common, expected, rtol=2e-6, atol=0.0)):
        raise AssertionError(
            f"{label}: common outer scale {float(common)} does not match "
            f"amax/{GLOBAL_SCALE_NUM:g}={float(expected)}"
        )
    print(
        f"{label}.outer_sg: common={float(common):.8e} "
        f"grid={tuple(row_sg.shape)} col_grid={tuple(col_sg.shape)}"
    )


def _run_case(label: str, weight: torch.Tensor) -> None:
    n, k = weight.shape
    m = SIZE
    weight_q = tuple(tkq.tk_localcta_quantize_weight_2d(weight))
    _check_common_outer_contract(label, weight, weight_q)

    row_reconstruction = tkq.tk_localcta_reconstruct_row(
        weight_q[0], weight_q[1], weight_q[4]
    )
    col_reconstruction_t = tkq.tk_localcta_reconstruct_col(
        weight_q[2], weight_q[3], weight_q[5]
    ).t().contiguous()
    if not torch.equal(row_reconstruction, col_reconstruction_t):
        mismatch = int(torch.count_nonzero(
            row_reconstruction != col_reconstruction_t
        ))
        raise AssertionError(
            f"{label}: shared 2D row/column reconstructions differ at "
            f"{mismatch} elements"
        )

    for orientation, reconstruction in (
        ("row", row_reconstruction),
        ("col_transposed", col_reconstruction_t),
    ):
        _assert_numerics(
            f"{label}.reconstruction.{orientation}",
            _metrics(reconstruction, weight),
            max_rel_l2=0.20,
            min_cosine=0.975,
            min_norm_ratio=0.90,
            max_norm_ratio=1.10,
        )

    # Independent random consumers avoid a cancellation-heavy sinusoidal
    # product where ordinary FP4 elementwise error dominates a tiny BF16 sum.
    activation = _random_normal(m, k, 0.20, 2026082101)
    output_grad = _random_normal(m, n, 0.20, 2026082103)
    activation_q = tuple(tkq.tk_localcta_quantize_weight_2d(activation))
    output_grad_q = tuple(tkq.tk_localcta_quantize_weight_2d(output_grad))

    forward = torch.empty(
        (m, n), device="cuda", dtype=torch.bfloat16
    )
    _outer_sg_gemm(
        activation_q[0],
        activation_q[1],
        _compact_common_outer_sg(activation_q[4], m),
        weight_q[0],
        weight_q[1],
        _compact_common_outer_sg(weight_q[4], n),
        forward,
    )
    forward_reference = activation @ weight.t()
    _assert_numerics(
        f"{label}.gemm.forward",
        _metrics(forward, forward_reference),
        max_rel_l2=0.30,
        min_cosine=0.95,
        min_norm_ratio=0.80,
        max_norm_ratio=1.20,
    )

    dgrad = torch.empty(
        (m, k), device="cuda", dtype=torch.bfloat16
    )
    _outer_sg_gemm(
        output_grad_q[0],
        output_grad_q[1],
        _compact_common_outer_sg(output_grad_q[4], m),
        weight_q[2],
        weight_q[3],
        _compact_common_outer_sg(weight_q[5], k),
        dgrad,
    )
    dgrad_reference = output_grad @ weight
    _assert_numerics(
        f"{label}.gemm.dgrad",
        _metrics(dgrad, dgrad_reference),
        max_rel_l2=0.30,
        min_cosine=0.95,
        min_norm_ratio=0.80,
        max_norm_ratio=1.20,
    )


def _check_exact_zero() -> None:
    zero = torch.zeros(
        (SIZE, SIZE), device="cuda", dtype=torch.bfloat16
    )
    quantized = tuple(tkq.tk_localcta_quantize_weight_2d(zero))
    if not bool(torch.all(quantized[4] == 0)) or not bool(
        torch.all(quantized[5] == 0)
    ):
        raise AssertionError("zero weight must retain an exact-zero common outer scale")
    reconstructed = tkq.tk_localcta_reconstruct_row(
        quantized[0], quantized[1], quantized[4]
    )
    if bool(torch.any(reconstructed != 0)):
        raise AssertionError("zero weight reconstruction must remain exactly zero")
    print("exact_zero: PASS")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"localCTA 2D-weight gate requires Blackwell, got {major}.{minor}"
        )

    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    previous_scale_num = float(tkq.tk_localcta_get_global_scale_num())
    tkq.tk_localcta_set_global_scale_num(GLOBAL_SCALE_NUM)
    try:
        below_threshold = _patterned(SIZE, SIZE, 0.61, 0.005)
        layer31 = (
            torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float32)
            * DEPTH_INIT_STD
        ).to(torch.bfloat16).contiguous()
        rectangular_layer31 = (
            torch.randn(512, 256, device="cuda", dtype=torch.float32)
            * DEPTH_INIT_STD
        ).to(torch.bfloat16).contiguous()
        _run_case("below_e4m3_absolute_floor", below_threshold)
        _run_case("depth_init_layer31", layer31)
        _run_case("rectangular_depth_init_layer31_512x256", rectangular_layer31)
        _check_exact_zero()
        torch.cuda.synchronize()
    finally:
        tkq.tk_localcta_set_global_scale_num(previous_scale_num)

    print(
        "localCTA 2D-weight common-outer-scale reconstruction/GEMM gate: PASS"
    )


if __name__ == "__main__":
    main()
