#!/usr/bin/env python3
"""Bitwise and lifetime checks for the native localCTA split2 RHT carrier.

The production fallback forms one logical [M, 2H] BF16 carrier, applies
row-only stochastic rounding plus fixed-sign column RHT, and then slices the
quantized result.  The native split2 producer must preserve that exact
contract while writing the two final layouts directly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch


os.environ["USE_TK_LOCALCTA_V4_FAST_DATA_SR"] = "1"
os.environ["USE_TK_LOCALCTA_V3_CONTRACT"] = "outer"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_localcta_v4 as tkq


_SEED = 42
_SUBSEQUENCE = 17
_STATE_STRIDE = 1 << 32
_ENCODE_CENTRIC = False
_PRODUCTION_SCALE_NUM = 448.0


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    if "float4" in str(tensor.dtype):
        return _bytes(tensor).clone()
    return tensor.clone()


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype in (torch.float32, torch.float64):
        equal = torch.equal(actual, expected)
    else:
        equal = torch.equal(_bytes(actual), _bytes(expected))
    if equal:
        return
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    byte_mismatches = int((_bytes(actual) != _bytes(expected)).sum().item())
    raise AssertionError(
        f"{name}: {byte_mismatches} byte mismatches over "
        f"{_bytes(actual).numel()} bytes"
    )


def _make_inputs(kind: str, m: int, h: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    if kind == "random":
        values = tuple(
            torch.randn((m, h), device="cuda", dtype=torch.bfloat16,
                        generator=generator)
            for _ in range(3)
        )
    elif kind == "zeros":
        values = tuple(
            torch.zeros((m, h), device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
    elif kind == "tiny":
        dh = torch.full((m, h), 1.0e-10, device="cuda", dtype=torch.bfloat16)
        h3 = torch.full_like(dh, -1.0e-7)
        h1 = torch.full_like(dh, 1.0e-4)
        values = (dh, h3, h1)
    elif kind == "outlier":
        values_list = [
            torch.randn((m, h), device="cuda", dtype=torch.bfloat16,
                        generator=generator)
            for _ in range(3)
        ]
        values_list[0][0, 0] = 1.0e4
        values_list[0][-1, -1] = -1.0e4
        values_list[1][m // 2, h // 2] = 2.0e3
        values_list[2][m // 3, h // 3] = -2.0e3
        values = tuple(values_list)
    elif kind == "signed":
        row = torch.arange(m, device="cuda", dtype=torch.float32)[:, None]
        col = torch.arange(h, device="cuda", dtype=torch.float32)[None, :]
        checker = torch.where(((row + col).to(torch.int64) & 1) == 0, 1.0, -1.0)
        values = (
            checker.to(torch.bfloat16),
            (checker * 0.5).to(torch.bfloat16),
            (checker * (1.0 + col / max(h, 1))).to(torch.bfloat16),
        )
    else:
        raise ValueError(kind)
    return tuple(value.contiguous() for value in values)


def _generic_reference(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    dh1 = torch.empty_like(dh)
    dh3 = torch.empty_like(dh)
    tkq.tk_localcta_silu_deriv_split_bf16_launch_inplace(
        dh, h3, h1, dh1, dh3
    )
    combined = torch.cat((dh1, dh3), dim=1)
    full = tkq.tk_localcta_quantize_for_gemm_opt(
        combined,
        True,
        _ENCODE_CENTRIC,
        True,
        False,
        "col",
        True,
        _SEED,
        _SUBSEQUENCE,
        "row",
        state,
    )
    width = dh.shape[1]
    row_fp4 = tuple(
        _bytes(full[0]).narrow(
            1, split * (width // 2), width // 2
        ).contiguous()
        for split in range(2)
    )
    row_sc = tuple(
        full[1].narrow(1, split * (width // 64), width // 64).contiguous()
        for split in range(2)
    )
    row_sg = (full[4], full[4])
    col_fp4 = tuple(
        _bytes(full[2]).narrow(0, split * width, width).contiguous()
        for split in range(2)
    )
    col_sc = tuple(
        full[3].narrow(0, split * (width // 128), width // 128).contiguous()
        for split in range(2)
    )
    col_sg = tuple(
        full[5].narrow(1, split * (width // 256), width // 256).contiguous()
        for split in range(2)
    )
    return (
        dh1,
        dh3,
        row_fp4[0], row_sc[0], row_sg[0], col_fp4[0], col_sc[0], col_sg[0],
        row_fp4[1], row_sc[1], row_sg[1], col_fp4[1], col_sc[1], col_sg[1],
    )


def _native(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1: torch.Tensor,
    state: torch.Tensor,
    *,
    cat_alloc: bool = True,
) -> tuple[torch.Tensor, ...]:
    m, h = dh.shape
    dh1 = torch.empty_like(dh)
    dh3 = torch.empty_like(dh)
    if cat_alloc:
        allocation = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc(
            m, h, dh.device
        )
        buffers = allocation[:16]
    else:
        allocation = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
            m, h, dh.device
        )
        buffers = allocation
    # The promoted production route intentionally retains its existing TE
    # BF16 producer and asks TK only to quantize the two completed derivative
    # buffers.  TK's producer is used here solely to create a deterministic
    # direct-quantization fixture; the LBT integration gate separately uses
    # the actual TE producer.
    tkq.tk_localcta_silu_deriv_split_bf16_launch_inplace(
        dh, h3, h1, dh1, dh3
    )
    output = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
        dh,
        h3,
        h1,
        dh1,
        dh3,
        *buffers,
        True,
        True,
        False,
        _SEED,
        _SUBSEQUENCE,
        "row",
        rht_axes="col",
        with_random_sign_mask=True,
        derivatives_precomputed=True,
        encode_centric=_ENCODE_CENTRIC,
        persistent_rng_state=state,
    )
    result = (
        dh1,
        dh3,
        output[0], output[1], output[4], output[2], output[3], output[5],
        output[6], output[7], output[10], output[8], output[9], output[11],
    )
    # In production the concatenated column owners are cached separately, but
    # each returned view must retain valid storage even after this local owner
    # tuple dies.  Returning only the views exercises that lifetime property.
    del allocation
    return result


_NAMES = (
    "dh1", "dh3",
    "row_fp4_0", "row_sc_0", "row_sg_0",
    "col_fp4_0", "col_sc_0", "col_sg_0",
    "row_fp4_1", "row_sc_1", "row_sg_1",
    "col_fp4_1", "col_sc_1", "col_sg_1",
)


def _check_case(kind: str, m: int, h: int, churn: int) -> None:
    inputs = _make_inputs(kind, m, h)
    initial = torch.tensor(
        [_SEED, _SUBSEQUENCE], device="cuda", dtype=torch.int64
    )
    reference_state = initial.clone()
    reference = tuple(_snapshot(t) for t in _generic_reference(*inputs, reference_state))
    assert int(reference_state[1].item()) == _SUBSEQUENCE + _STATE_STRIDE

    for iteration in range(churn + 1):
        if iteration:
            junk = [
                torch.empty((m, h), device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            del junk
        native_state = initial.clone()
        for cat_alloc in (True, False):
            native_state.copy_(initial)
            actual = _native(*inputs, native_state, cat_alloc=cat_alloc)
            torch.cuda.synchronize()
            _assert_exact("persistent_rng_state", native_state, reference_state)
            for name, actual_tensor, expected_tensor in zip(_NAMES, actual, reference):
                _assert_exact(name, actual_tensor, expected_tensor)
            for index, tensor in enumerate(actual):
                if index >= 2 and not tensor.is_contiguous():
                    raise AssertionError(
                        f"{_NAMES[index]} is not contiguous "
                        f"(cat_alloc={cat_alloc})"
                    )


def _benchmark(m: int, h: int, iterations: int) -> None:
    inputs = _make_inputs("random", m, h)
    state = torch.tensor([_SEED, _SUBSEQUENCE], device="cuda", dtype=torch.int64)
    for _ in range(5):
        _native(*inputs, state)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _native(*inputs, state)
    end.record()
    end.synchronize()
    native_ms = start.elapsed_time(end) / iterations

    start.record()
    for _ in range(iterations):
        _generic_reference(*inputs, state)
    end.record()
    end.synchronize()
    generic_ms = start.elapsed_time(end) / iterations
    print(
        f"benchmark M={m} H={h}: native={native_ms:.4f} ms "
        f"generic_cat={generic_ms:.4f} ms speedup={generic_ms/native_ms:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-extension",
        type=Path,
        required=True,
        help="Exact _tk_quant_localcta_v4 shared object this gate must import",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-m", type=int, default=32768)
    parser.add_argument("--benchmark-h", type=int, default=14336)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    args = parser.parse_args()

    loaded_extension = Path(tkq.__file__).resolve()
    expected_extension = args.expected_extension.resolve()
    if loaded_extension != expected_extension:
        raise RuntimeError(
            "stale/wrong localCTA extension loaded: "
            f"{loaded_extension}; expected {expected_extension}"
        )
    print(f"split2 gate extension: {loaded_extension}")

    marker = getattr(tkq, "tk_localcta_silu_deriv_split2_supports_rht", None)
    if marker is None or not marker():
        raise RuntimeError("extension does not advertise native split2 RHT support")
    tkq.tk_localcta_set_global_scale_num(_PRODUCTION_SCALE_NUM)
    if float(tkq.tk_localcta_get_global_scale_num()) != _PRODUCTION_SCALE_NUM:
        raise RuntimeError("failed to apply the production localCTA scale numerator")

    started = time.monotonic()
    for m, h in ((256, 256), (512, 512), (256, 512)):
        for kind in ("random", "zeros", "tiny", "outlier", "signed"):
            _check_case(kind, m, h, churn=2)
            print(f"PASS kind={kind} M={m} H={h}")
    print(f"bitwise split2 RHT equivalence PASS in {time.monotonic() - started:.2f}s")
    if args.benchmark:
        _benchmark(args.benchmark_m, args.benchmark_h, args.benchmark_iters)


if __name__ == "__main__":
    main()
