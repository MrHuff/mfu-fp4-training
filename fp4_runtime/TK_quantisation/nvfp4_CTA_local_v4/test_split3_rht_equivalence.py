#!/usr/bin/env python3
"""Bitwise, RNG, alias, and speed gate for native paired QKV split3 RHT."""

from __future__ import annotations

import argparse
import importlib.util
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
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    return _bytes(tensor).clone() if "float4" in str(tensor.dtype) else tensor.clone()


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    actual_bytes = _bytes(actual)
    expected_bytes = _bytes(expected)
    if torch.equal(actual_bytes, expected_bytes):
        return
    mismatches = int((actual_bytes != expected_bytes).sum().item())
    raise AssertionError(
        f"{name}: {mismatches} byte mismatches over {actual_bytes.numel()} bytes"
    )


def _make_inputs(kind: str, m: int, widths: tuple[int, int, int]):
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    if kind == "random":
        values = [
            torch.randn((m, width), device="cuda", dtype=torch.bfloat16,
                        generator=generator)
            for width in widths
        ]
    elif kind == "zeros":
        values = [
            torch.zeros((m, width), device="cuda", dtype=torch.bfloat16)
            for width in widths
        ]
    elif kind == "tiny":
        values = [
            torch.full((m, width), value, device="cuda", dtype=torch.bfloat16)
            for width, value in zip(widths, (1.0e-10, -1.0e-7, 1.0e-4), strict=True)
        ]
    elif kind == "outlier":
        values = [
            torch.randn((m, width), device="cuda", dtype=torch.bfloat16,
                        generator=generator)
            for width in widths
        ]
        for index, value in enumerate((1.0e4, -2.0e3, 3.0e3)):
            values[index][index, index] = value
            values[index][-1 - index, -1 - index] = -value
    elif kind == "signed":
        values = []
        row = torch.arange(m, device="cuda", dtype=torch.float32)[:, None]
        for split, width in enumerate(widths):
            col = torch.arange(width, device="cuda", dtype=torch.float32)[None, :]
            sign = torch.where(
                ((row + col + split).to(torch.int64) & 1) == 0, 1.0, -1.0
            )
            values.append((sign * (split + 1) / 3.0).to(torch.bfloat16))
    else:
        raise ValueError(kind)
    return tuple(value.contiguous() for value in values)


def _generic(inputs, state):
    combined = torch.cat(inputs, dim=1)
    return tkq.tk_localcta_quantize_for_gemm_opt(
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


def _native(inputs, state):
    return tkq.tk_localcta_group_quantize_dim1_split3_for_gemm(
        *inputs,
        True,
        _SEED,
        _SUBSEQUENCE,
        "row",
        state,
        "col",
        True,
        _ENCODE_CENTRIC,
    )


def _assert_aliases(result, widths):
    row_fp4s, row_scs, row_sgs, col_fp4s, col_scs, col_sgs = result[:6]
    row_fp4_full, row_sc_full = result[6], result[7]
    col_fp4_full, col_sc_full, col_sg_full = result[9:12]

    row_fp4_storage = row_fp4_full.untyped_storage().data_ptr()
    row_sc_storage = row_sc_full.untyped_storage().data_ptr()
    col_fp4_storage = col_fp4_full.untyped_storage().data_ptr()
    col_sc_storage = col_sc_full.untyped_storage().data_ptr()
    col_sg_storage = col_sg_full.untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr() != row_fp4_storage for t in row_fp4s):
        raise AssertionError("row FP4 splits do not alias the full carrier")
    if any(t.untyped_storage().data_ptr() != row_sc_storage for t in row_scs):
        raise AssertionError("row scale splits do not alias the full carrier")
    if any(t.untyped_storage().data_ptr() != col_fp4_storage for t in col_fp4s):
        raise AssertionError("column FP4 splits do not alias the full carrier")
    if any(t.untyped_storage().data_ptr() != col_sc_storage for t in col_scs):
        raise AssertionError("column scale splits do not alias the full carrier")
    if any(t.untyped_storage().data_ptr() != col_sg_storage for t in col_sgs):
        raise AssertionError("column SG splits do not alias the full carrier")
    if len({t.data_ptr() for t in row_sgs}) != 1:
        raise AssertionError("Q/K/V row outer SG tensors must share one pointer")

    fp4_offsets = [0, widths[0] // 2, (widths[0] + widths[1]) // 2]
    col_offsets = [0, widths[0], widths[0] + widths[1]]
    for index in range(3):
        if row_fp4s[index].storage_offset() != fp4_offsets[index]:
            raise AssertionError(f"row FP4 split {index} has wrong storage offset")
        if col_fp4s[index].storage_offset() != col_offsets[index] * (result[9].shape[1]):
            raise AssertionError(f"column FP4 split {index} has wrong storage offset")


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_tensors(child)


def _pointer_signature(value):
    return tuple(
        (
            tensor.data_ptr(),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
        )
        for tensor in _iter_tensors(value)
    )


def _load_localcta_gemm(gemm_so: str | None):
    if gemm_so is None:
        gemm_so = os.environ.get("TK_LOCALCTA_GEMM_SO")
    if gemm_so is not None:
        candidates = [Path(gemm_so).resolve()]
    else:
        gemm_dir = (
            _REPO_ROOT
            / "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3"
        )
        candidates = sorted(
            gemm_dir.glob("_C_nv_localcta_gemm_v3*.so"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    if not candidates or not candidates[0].is_file():
        raise RuntimeError(
            "consumer gate requires the v4 production GEMM extension; build "
            "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3 "
            "or pass --gemm-so/TK_LOCALCTA_GEMM_SO"
        )
    path = candidates[0]
    module_name = path.name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load GEMM extension spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "nvfp4_localcta_fast_split3_dgrad_strided_gemm_sg",
        "nvfp4_localcta_grouped_gemm",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"GEMM extension {path} lacks {missing}")
    print(f"consumer gate GEMM extension: {path}")
    return module


def _allocator_churn(rounds: int) -> None:
    # Large enough to force real allocator activity without dominating the
    # production-shape carriers.  The live payloads must remain pinned by the
    # explicit keepalive below and must never be recycled into this storage.
    for iteration in range(rounds):
        junk = [
            torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
            for _ in range(2)
        ]
        for index, tensor in enumerate(junk):
            tensor.fill_((iteration + index) & 0xFF)
        del junk


def _make_consumer_payloads(inputs, widths):
    initial = torch.tensor([_SEED, _SUBSEQUENCE], device="cuda", dtype=torch.int64)
    generic_state = initial.clone()
    combined = torch.cat(inputs, dim=1)
    generic_result = tkq.tk_localcta_quantize_for_gemm_opt(
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
        generic_state,
    )
    native_state = initial.clone()
    native_result = _native(inputs, native_state)
    torch.cuda.synchronize()
    _assert_exact("consumer persistent RNG state", native_state, generic_state)
    if int(native_state[1].item()) != _SUBSEQUENCE + _STATE_STRIDE:
        raise AssertionError("consumer payload did not reserve exactly one RNG stride")
    for name, native_tensor, generic_tensor in (
        ("row_fp4_full", native_result[6], generic_result[0]),
        ("row_sc_full", native_result[7], generic_result[1]),
        ("col_fp4_full", native_result[9], generic_result[2]),
        ("col_sc_full", native_result[10], generic_result[3]),
        ("col_sg_full", native_result[11], generic_result[5]),
    ):
        _assert_exact(f"consumer {name}", native_tensor, generic_tensor)
    for split, row_sg in enumerate(native_result[2]):
        _assert_exact(f"consumer row_sg_{split}", row_sg, generic_result[4])

    generic_row_sc = []
    sc_offset = 0
    for width in widths:
        # This is intentionally the real concat fallback layout: each row
        # scale split is copied contiguous before the strict strided consumer.
        generic_row_sc.append(
            generic_result[1].narrow(1, sc_offset, width // 64).contiguous()
        )
        sc_offset += width // 64
    generic = {
        "row_fp4": generic_result[0],
        "row_sc": generic_row_sc,
        "row_sg": [generic_result[4]] * 3,
        "col_fp4": generic_result[2],
        "col_sc": generic_result[3],
        "col_sg": generic_result[5],
    }
    native = {
        "row_fp4": native_result[6],
        "row_sc": list(native_result[1]),
        "row_sg": list(native_result[2]),
        "col_fp4": native_result[9],
        "col_sc": native_result[10],
        "col_sg": native_result[11],
    }
    keepalive = (
        inputs,
        combined,
        generic_result,
        generic_row_sc,
        native_result,
        generic_state,
        native_state,
    )
    return generic, native, keepalive


def _quantize_consumer_operand(
    operand: torch.Tensor,
    *,
    paired_rht: bool,
    state_subsequence: int,
):
    if paired_rht:
        state = torch.tensor(
            [_SEED + 1, state_subsequence], device="cuda", dtype=torch.int64
        )
        result = tkq.tk_localcta_quantize_for_gemm_opt(
            operand,
            True,
            _ENCODE_CENTRIC,
            True,
            False,
            "col",
            True,
            _SEED + 1,
            state_subsequence,
            "row",
            state,
        )
        return result, state
    result = tkq.tk_localcta_quantize_for_gemm_opt(
        operand,
        True,
        _ENCODE_CENTRIC,
        False,
        False,
        "none",
        False,
        _SEED + 2,
        state_subsequence,
        "none",
    )
    return result, None


def _consumer_equivalence_gate(
    m: int,
    widths: tuple[int, int, int],
    k: int,
    *,
    gemm_so: str | None,
    churn: int,
) -> None:
    if m % 256 or k % 256 or any(width % 256 for width in widths):
        raise ValueError("consumer gate dimensions must all be 256-aligned")
    gemm = _load_localcta_gemm(gemm_so)
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    grads = tuple(
        torch.randn(
            (m, width), device="cuda", dtype=torch.bfloat16, generator=generator
        ).mul_(0.01)
        for width in widths
    )
    activation = torch.randn(
        (m, k), device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(0.01)
    weight = torch.randn(
        (sum(widths), k), device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(0.01)
    input_snapshots = tuple(tensor.clone() for tensor in (*grads, activation, weight))

    generic, native, qkv_keepalive = _make_consumer_payloads(grads, widths)
    activation_quant, activation_state = _quantize_consumer_operand(
        activation, paired_rht=True, state_subsequence=73
    )
    weight_quant, _ = _quantize_consumer_operand(
        weight, paired_rht=False, state_subsequence=101
    )
    torch.cuda.synchronize()
    if int(activation_state[1].item()) != 73 + _STATE_STRIDE:
        raise AssertionError("activation carrier did not reserve exactly one RNG stride")

    dgrad_generic = torch.full(
        (m, k), -7.5, device="cuda", dtype=torch.bfloat16
    )
    dgrad_native = torch.full_like(dgrad_generic, -7.5)
    wgrad_default_generic = torch.full(
        (k, sum(widths)), -7.5, device="cuda", dtype=torch.bfloat16
    )
    wgrad_default_native = torch.full_like(wgrad_default_generic, -7.5)
    wgrad_direct_generic = torch.full(
        (sum(widths), k), -7.5, device="cuda", dtype=torch.bfloat16
    )
    wgrad_direct_native = torch.full_like(wgrad_direct_generic, -7.5)
    a_col_offsets = [0, widths[0] // 2, (widths[0] + widths[1]) // 2]
    a_col_widths = [width // 2 for width in widths]
    weight_fp4_bytes = weight_quant[2].view(torch.uint8)
    weight_sc_bytes = weight_quant[3].view(torch.uint8)
    weight_fp4_splits = []
    weight_sc_splits = []
    fp4_offset = 0
    sc_offset = 0
    for width in widths:
        # Match `_split_weight_col_tensors`: strict strided-sum consumes
        # contiguous split-B payloads, while A remains one full carrier.
        weight_fp4_splits.append(
            weight_fp4_bytes[:, fp4_offset:fp4_offset + width // 2]
            .contiguous()
            .view(torch.float4_e2m1fn_x2)
        )
        weight_sc_splits.append(
            weight_sc_bytes[:, sc_offset:sc_offset + width // 64]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        fp4_offset += width // 2
        sc_offset += width // 64
    weight_sg_splits = [weight_quant[5].contiguous()] * 3

    # These objects deliberately outlive both the caller-stream DGRAD and the
    # side-stream WGRAD, matching the production package's keepalive contract.
    keepalive = (
        qkv_keepalive,
        generic,
        native,
        activation,
        activation_quant,
        activation_state,
        weight,
        weight_quant,
        weight_fp4_splits,
        weight_sc_splits,
        weight_sg_splits,
        dgrad_generic,
        dgrad_native,
        wgrad_default_generic,
        wgrad_default_native,
        wgrad_direct_generic,
        wgrad_direct_native,
    )
    pointer_signature = _pointer_signature(keepalive)
    previous_dgrad = None
    previous_wgrad_default = None
    previous_wgrad_direct = None
    caller_stream = torch.cuda.current_stream()
    wgrad_stream = torch.cuda.Stream()

    for iteration in range(churn + 1):
        for output in (
            dgrad_generic,
            dgrad_native,
            wgrad_default_generic,
            wgrad_default_native,
            wgrad_direct_generic,
            wgrad_direct_native,
        ):
            output.fill_(-7.5)
        _allocator_churn(1)
        if _pointer_signature(keepalive) != pointer_signature:
            raise AssertionError("live consumer carrier pointer/layout changed under churn")

        wgrad_stream.wait_stream(caller_stream)
        wgrad_tensors = (
            activation_quant[2], activation_quant[3], activation_quant[5],
            generic["col_fp4"], generic["col_sc"], generic["col_sg"],
            native["col_fp4"], native["col_sc"], native["col_sg"],
            wgrad_default_generic, wgrad_default_native,
            wgrad_direct_generic, wgrad_direct_native,
        )
        for tensor in wgrad_tensors:
            tensor.record_stream(wgrad_stream)
        with torch.cuda.stream(wgrad_stream):
            # This x-first `(K, N_total)` layout is the actual default fused
            # QKV route when no direct-layout override is present.
            gemm.nvfp4_localcta_grouped_gemm(
                activation_quant[2], activation_quant[3], activation_quant[5],
                generic["col_fp4"], generic["col_sc"], generic["col_sg"],
                wgrad_default_generic,
            )
            gemm.nvfp4_localcta_grouped_gemm(
                activation_quant[2], activation_quant[3], activation_quant[5],
                native["col_fp4"], native["col_sc"], native["col_sg"],
                wgrad_default_native,
            )
            # Also gate the explicit direct-layout route cheaply; it reverses
            # the operands and writes `(N_total, K)` without a transpose.
            gemm.nvfp4_localcta_grouped_gemm(
                generic["col_fp4"], generic["col_sc"], generic["col_sg"],
                activation_quant[2], activation_quant[3], activation_quant[5],
                wgrad_direct_generic,
            )
            gemm.nvfp4_localcta_grouped_gemm(
                native["col_fp4"], native["col_sc"], native["col_sg"],
                activation_quant[2], activation_quant[3], activation_quant[5],
                wgrad_direct_native,
            )

        gemm.nvfp4_localcta_fast_split3_dgrad_strided_gemm_sg(
            generic["row_fp4"], generic["row_sc"], generic["row_sg"],
            a_col_offsets, a_col_widths,
            weight_fp4_splits, weight_sc_splits, weight_sg_splits,
            dgrad_generic,
        )
        gemm.nvfp4_localcta_fast_split3_dgrad_strided_gemm_sg(
            native["row_fp4"], native["row_sc"], native["row_sg"],
            a_col_offsets, a_col_widths,
            weight_fp4_splits, weight_sc_splits, weight_sg_splits,
            dgrad_native,
        )
        # Stress the allocator while the production-style side-stream WGRAD
        # can still be in flight; all of its inputs remain live and recorded.
        _allocator_churn(1)
        caller_stream.wait_stream(wgrad_stream)
        torch.cuda.synchronize()

        _assert_exact(f"strict DGRAD iteration {iteration}", dgrad_native, dgrad_generic)
        _assert_exact(
            f"grouped WGRAD default iteration {iteration}",
            wgrad_default_native,
            wgrad_default_generic,
        )
        _assert_exact(
            f"grouped WGRAD direct iteration {iteration}",
            wgrad_direct_native,
            wgrad_direct_generic,
        )
        if previous_dgrad is not None:
            _assert_exact(
                f"strict DGRAD deterministic iteration {iteration}",
                dgrad_generic,
                previous_dgrad,
            )
            _assert_exact(
                f"grouped WGRAD default deterministic iteration {iteration}",
                wgrad_default_generic,
                previous_wgrad_default,
            )
            _assert_exact(
                f"grouped WGRAD direct deterministic iteration {iteration}",
                wgrad_direct_generic,
                previous_wgrad_direct,
            )
        previous_dgrad = dgrad_generic.clone()
        previous_wgrad_default = wgrad_default_generic.clone()
        previous_wgrad_direct = wgrad_direct_generic.clone()
        if _pointer_signature(keepalive) != pointer_signature:
            raise AssertionError("consumer carrier pointer/layout changed after launch")

    for index, (current, snapshot) in enumerate(
        zip((*grads, activation, weight), input_snapshots, strict=True)
    ):
        _assert_exact(f"consumer input {index}", current, snapshot)
    print(
        "production consumer equivalence PASS: "
        f"M={m} K={k} widths={widths} strict_strided_sum=byte-exact "
        "grouped_wgrad_default_x_first=byte-exact "
        "grouped_wgrad_optional_direct=byte-exact side_stream_keepalive=PASS"
    )


def _check_case(kind: str, m: int, widths: tuple[int, int, int], churn: int) -> None:
    inputs = _make_inputs(kind, m, widths)
    input_snapshots = tuple(t.clone() for t in inputs)
    initial = torch.tensor([_SEED, _SUBSEQUENCE], device="cuda", dtype=torch.int64)
    reference_state = initial.clone()
    reference = tuple(_snapshot(t) for t in _generic(inputs, reference_state))
    if int(reference_state[1].item()) != _SUBSEQUENCE + _STATE_STRIDE:
        raise AssertionError("generic carrier did not reserve exactly one RNG stride")

    for iteration in range(churn + 1):
        if iteration:
            junk = [
                torch.empty((m, sum(widths)), device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            del junk
        state = initial.clone()
        actual = _native(inputs, state)
        torch.cuda.synchronize()
        _assert_exact("persistent_rng_state", state, reference_state)
        _assert_exact("row_fp4_full", actual[6], reference[0])
        _assert_exact("row_sc_full", actual[7], reference[1])
        _assert_exact("col_fp4_full", actual[9], reference[2])
        _assert_exact("col_sc_full", actual[10], reference[3])
        for split, row_sg in enumerate(actual[2]):
            _assert_exact(f"row_sg_{split}", row_sg, reference[4])
        _assert_exact("col_sg_full", actual[11], reference[5])
        _assert_aliases(actual, widths)
        for index, (current, snapshot) in enumerate(zip(inputs, input_snapshots, strict=True)):
            _assert_exact(f"input_{index}", current, snapshot)


def _benchmark(m: int, widths: tuple[int, int, int], iterations: int) -> None:
    inputs = _make_inputs("random", m, widths)
    state = torch.tensor([_SEED, _SUBSEQUENCE], device="cuda", dtype=torch.int64)
    for _ in range(5):
        _native(inputs, state)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _native(inputs, state)
    end.record()
    end.synchronize()
    native_ms = start.elapsed_time(end) / iterations

    start.record()
    for _ in range(iterations):
        _generic(inputs, state)
    end.record()
    end.synchronize()
    generic_ms = start.elapsed_time(end) / iterations
    print(
        f"benchmark M={m} widths={widths}: native={native_ms:.4f} ms "
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
    parser.add_argument("--benchmark-widths", type=int, nargs=3, default=(4096, 1024, 1024))
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--consumer-gate", action="store_true")
    parser.add_argument("--consumer-k", type=int, default=4096)
    parser.add_argument("--consumer-churn", type=int, default=2)
    parser.add_argument("--gemm-so")
    args = parser.parse_args()

    loaded_extension = Path(tkq.__file__).resolve()
    expected_extension = args.expected_extension.resolve()
    if loaded_extension != expected_extension:
        raise RuntimeError(
            "stale/wrong localCTA extension loaded: "
            f"{loaded_extension}; expected {expected_extension}"
        )
    print(f"split3 gate extension: {loaded_extension}")

    marker = getattr(tkq, "tk_localcta_split3_supports_paired_rht", None)
    if marker is None or not marker():
        raise RuntimeError("extension does not advertise native paired split3 RHT")
    tkq.tk_localcta_set_global_scale_num(_PRODUCTION_SCALE_NUM)
    if float(tkq.tk_localcta_get_global_scale_num()) != _PRODUCTION_SCALE_NUM:
        raise RuntimeError("failed to apply the production localCTA scale numerator")

    started = time.monotonic()
    for m, widths in (
        (256, (256, 256, 256)),
        (512, (512, 256, 256)),
        (256, (256, 512, 256)),
    ):
        for kind in ("random", "zeros", "tiny", "outlier", "signed"):
            _check_case(kind, m, widths, churn=2)
            print(f"PASS kind={kind} M={m} widths={widths}")
    print(f"bitwise split3 RHT equivalence PASS in {time.monotonic() - started:.2f}s")
    if args.benchmark:
        benchmark_widths = tuple(args.benchmark_widths)
        _check_case("random", args.benchmark_m, benchmark_widths, churn=1)
        print(
            "production-shape bitwise split3 RHT equivalence PASS "
            f"M={args.benchmark_m} widths={benchmark_widths}"
        )
        _benchmark(args.benchmark_m, benchmark_widths, args.benchmark_iters)
    if args.consumer_gate:
        _consumer_equivalence_gate(
            args.benchmark_m,
            tuple(args.benchmark_widths),
            args.consumer_k,
            gemm_so=args.gemm_so,
            churn=args.consumer_churn,
        )


if __name__ == "__main__":
    main()
