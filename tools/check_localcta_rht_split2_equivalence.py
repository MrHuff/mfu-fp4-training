#!/usr/bin/env python3
"""Production equivalence and pointer-safety gates for native split2 RHT.

This intentionally uses the established TE BF16 SiLU-derivative producer on
both arms.  The reference then concatenates the derivatives and invokes the
generic localCTA carrier; the treatment writes the two quantized layouts
directly.  A checkpoint continuation is safe only if every observable tensor
and the persistent SR state match bit-for-bit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time


os.environ["USE_TK_LOCALCTA_V4_FAST_DATA_SR"] = "1"
os.environ["USE_TK_LOCALCTA_V3_CONTRACT"] = "outer"
os.environ["NVTE_NVFP4_ENCODE_CENTRIC"] = "0"
os.environ["NVFP4_SR_GRAD"] = "1"
os.environ["NVFP4_GRAD_SR_AXES"] = "row"
os.environ["NVFP4_USE_SCALE_STOCHASTIC_ROUNDING"] = "0"
os.environ["NVFP4_SCALE_SR_GRAD"] = "0"
os.environ["NVFP4_RHT_GRAD"] = "1"
os.environ["NVFP4_RHT_ACTIVATION"] = "1"
os.environ["NVFP4_RHT_WEIGHT"] = "0"
os.environ["NVFP4_RHT_AXES"] = "col"
os.environ["NVFP4_RHT_RANDOM_SIGNS"] = "1"

import torch

from low_bits_training.quantization import fused_te_linear as fte
import _tk_quant_localcta_v4 as tkq


SEED = 42
SUBSEQUENCE = 17
STATE_STRIDE = 1 << 32
ENCODE_CENTRIC = False
PRODUCTION_SCALE_NUM = 448.0


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    # PyTorch deliberately has no copy_/clone kernel for packed e2m1x2.  Its
    # storage is byte-addressable, so snapshot the packed payload directly.
    if "float4" in str(tensor.dtype):
        return _bytes(tensor).clone()
    return tensor.clone()


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    lhs = actual if actual.dtype in (torch.float32, torch.float64) else _bytes(actual)
    rhs = expected if expected.dtype in (torch.float32, torch.float64) else _bytes(expected)
    if torch.equal(lhs, rhs):
        return
    mismatch = int((_bytes(actual) != _bytes(expected)).sum().item())
    raise AssertionError(
        f"{name}: {mismatch} mismatched bytes of {_bytes(actual).numel()}"
    )


def _state(seed: int = SEED, subsequence: int = SUBSEQUENCE) -> torch.Tensor:
    return torch.tensor([seed, subsequence], device="cuda", dtype=torch.int64)


def _inputs(m: int, h: int, kind: str = "random") -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    if kind == "random":
        tensors = tuple(
            torch.randn((m, h), device="cuda", dtype=torch.bfloat16,
                        generator=generator)
            for _ in range(3)
        )
    elif kind == "zero":
        tensors = tuple(
            torch.zeros((m, h), device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
    elif kind == "tiny":
        dh = torch.full((m, h), 1.0e-10, device="cuda", dtype=torch.bfloat16)
        tensors = (dh, torch.full_like(dh, -1.0e-7), torch.full_like(dh, 1.0e-4))
    elif kind == "outlier":
        values = [
            torch.randn((m, h), device="cuda", dtype=torch.bfloat16,
                        generator=generator)
            for _ in range(3)
        ]
        values[0][0, 0] = 1.0e4
        values[0][-1, -1] = -1.0e4
        values[1][m // 2, h // 2] = 2.0e3
        values[2][m // 3, h // 3] = -2.0e3
        tensors = tuple(values)
    else:
        raise ValueError(kind)
    return tuple(t.contiguous() for t in tensors)


def _produce_te(
    te_fused,
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1: torch.Tensor,
    dh1: torch.Tensor,
    dh3: torch.Tensor,
) -> None:
    amax1 = torch.empty(1, device=dh.device, dtype=torch.float32)
    amax2 = torch.empty_like(amax1)
    fte._produce_ffn_localcta_derivatives_with_te(
        te_fused, dh, h3, h1, dh1, dh3, amax1, amax2
    )


def _split_generic(full: tuple[torch.Tensor, ...], width: int) -> tuple[torch.Tensor, ...]:
    row_fp4 = tuple(
        _bytes(full[0]).narrow(1, i * (width // 2), width // 2).contiguous()
        for i in range(2)
    )
    row_sc = tuple(
        full[1].narrow(1, i * (width // 64), width // 64).contiguous()
        for i in range(2)
    )
    col_fp4 = tuple(
        _bytes(full[2]).narrow(0, i * width, width).contiguous()
        for i in range(2)
    )
    col_sc = tuple(
        full[3].narrow(0, i * (width // 128), width // 128).contiguous()
        for i in range(2)
    )
    col_sg = tuple(
        full[5].narrow(1, i * (width // 256), width // 256).contiguous()
        for i in range(2)
    )
    split = (
        row_fp4[0], row_sc[0], full[4], col_fp4[0], col_sc[0], col_sg[0],
        row_fp4[1], row_sc[1], full[4], col_fp4[1], col_sc[1], col_sg[1],
    )
    # Grouped WGRAD consumes these full owners, not merely the split views.
    return (*split, full[2], full[3], full[5])


def _reference(
    dh1: torch.Tensor,
    dh3: torch.Tensor,
    state: torch.Tensor,
    *,
    encode_centric: bool = ENCODE_CENTRIC,
) -> tuple[torch.Tensor, ...]:
    combined = torch.cat((dh1, dh3), dim=1)
    full = tkq.tk_localcta_quantize_for_gemm_opt(
        combined,
        True,
        encode_centric,
        True,
        False,
        "col",
        True,
        SEED,
        SUBSEQUENCE,
        "row",
        state,
    )
    return _split_generic(full, dh1.shape[1])


def _alloc_native(m: int, h: int) -> tuple[torch.Tensor, ...]:
    allocation = tuple(
        tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc(
            m, h, torch.device("cuda")
        )
    )
    _assert_full_owner_views(allocation, h)
    return allocation


def _assert_alias(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    for attr in ("shape", "dtype", "device", "stride", "storage_offset"):
        actual_value = getattr(actual, attr)
        expected_value = getattr(expected, attr)
        actual_value = actual_value() if callable(actual_value) else actual_value
        expected_value = expected_value() if callable(expected_value) else expected_value
        if actual_value != expected_value:
            raise AssertionError(
                f"{name}: {attr} {actual_value!r} != {expected_value!r}"
            )
    if actual.untyped_storage().data_ptr() != expected.untyped_storage().data_ptr():
        raise AssertionError(f"{name}: does not share the owner's storage")
    if actual.data_ptr() != expected.data_ptr():
        raise AssertionError(f"{name}: data pointer does not match the expected view")


def _assert_full_owner_views(allocation: tuple[torch.Tensor, ...], h: int) -> None:
    if len(allocation) != 19:
        raise AssertionError(f"native cat allocation has {len(allocation)} tensors, not 19")
    col_fp4_owner, col_sc_owner, col_sg_owner = allocation[16:19]
    expected_views = (
        ("col_fp4_0", allocation[2], col_fp4_owner.narrow(0, 0, h)),
        ("col_fp4_1", allocation[8], col_fp4_owner.narrow(0, h, h)),
        ("col_sc_0", allocation[3], col_sc_owner.narrow(0, 0, h // 128)),
        ("col_sc_1", allocation[9], col_sc_owner.narrow(0, h // 128, h // 128)),
        ("col_sg_0", allocation[5], col_sg_owner.narrow(1, 0, h // 256)),
        ("col_sg_1", allocation[11], col_sg_owner.narrow(1, h // 256, h // 256)),
    )
    for name, actual, expected in expected_views:
        _assert_alias(name, actual, expected)


def _allocation_fingerprint(allocation: tuple[torch.Tensor, ...]) -> tuple[tuple, ...]:
    return tuple(
        (
            tensor.untyped_storage().data_ptr(),
            tensor.data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )
        for tensor in allocation
    )


def _assert_native_aliases(
    actual: tuple[torch.Tensor, ...], allocation: tuple[torch.Tensor, ...]
) -> None:
    expected_slots = (
        allocation[0], allocation[1], allocation[4],
        allocation[2], allocation[3], allocation[5],
        allocation[6], allocation[7], allocation[10],
        allocation[8], allocation[9], allocation[11],
        allocation[16], allocation[17], allocation[18],
    )
    if len(actual) != len(expected_slots):
        raise AssertionError(
            f"native launch returned {len(actual)} tensors, expected {len(expected_slots)}"
        )
    for name, tensor, expected in zip(NAMES, actual, expected_slots):
        _assert_alias(f"launch_alias.{name}", tensor, expected)


def _native(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1: torch.Tensor,
    dh1: torch.Tensor,
    dh3: torch.Tensor,
    allocation: tuple[torch.Tensor, ...],
    state: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    output = fte._call_localcta_silu_deriv_split2(
        tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace,
        dh,
        h3,
        h1,
        dh1,
        dh3,
        *allocation[:16],
        True,
        True,
        False,
        SEED,
        SUBSEQUENCE,
        "row",
        persistent_rng_state=state,
        native_paired_rht=True,
    )
    result = (
        output[0], output[1], output[4], output[2], output[3], output[5],
        output[6], output[7], output[10], output[8], output[9], output[11],
        allocation[16], allocation[17], allocation[18],
    )
    _assert_native_aliases(result, allocation)
    return result


NAMES = (
    "row_fp4_0", "row_sc_0", "row_sg_0", "col_fp4_0", "col_sc_0", "col_sg_0",
    "row_fp4_1", "row_sc_1", "row_sg_1", "col_fp4_1", "col_sc_1", "col_sg_1",
    "col_fp4_full", "col_sc_full", "col_sg_full",
)

NATIVE_DTYPES = (
    torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float32,
    torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float32,
    torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float32,
    torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float32,
    torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float32,
)


def _compare_outputs(actual, expected) -> None:
    if not (len(actual) == len(expected) == len(NAMES) == len(NATIVE_DTYPES)):
        raise AssertionError(
            f"output arity actual={len(actual)} expected={len(expected)} "
            f"names={len(NAMES)} dtypes={len(NATIVE_DTYPES)}"
        )
    for name, dtype, lhs, rhs in zip(NAMES, NATIVE_DTYPES, actual, expected):
        if lhs.dtype != dtype or lhs.device.type != "cuda":
            raise AssertionError(
                f"{name}: native dtype/device {lhs.dtype}/{lhs.device} != {dtype}/cuda"
            )
        _assert_exact(name, lhs, rhs)
        if not lhs.is_contiguous():
            raise AssertionError(f"{name}: native output is not contiguous")


def check_exact(te_fused, m: int, h: int, kind: str, inplace: bool) -> None:
    dh, h3_initial, h1_initial = _inputs(m, h, kind)
    h3_ref = h3_initial.clone()
    h1_ref = h1_initial.clone()
    h3_native = h3_initial.clone()
    h1_native = h1_initial.clone()
    if inplace:
        dh1_ref, dh3_ref = h1_ref, h3_ref
        dh1_native, dh3_native = h1_native, h3_native
    else:
        dh1_ref, dh3_ref = torch.empty_like(dh), torch.empty_like(dh)
        dh1_native, dh3_native = torch.empty_like(dh), torch.empty_like(dh)
    _produce_te(te_fused, dh, h3_ref, h1_ref, dh1_ref, dh3_ref)
    _produce_te(te_fused, dh, h3_native, h1_native, dh1_native, dh3_native)
    torch.cuda.synchronize()
    _assert_exact("te_dh1_repeat", dh1_native, dh1_ref)
    _assert_exact("te_dh3_repeat", dh3_native, dh3_ref)
    dh1_before_native = dh1_native.clone()
    dh3_before_native = dh3_native.clone()

    reference_state = _state()
    native_state = _state()
    expected = tuple(_snapshot(t) for t in _reference(dh1_ref, dh3_ref, reference_state))
    allocation = _alloc_native(m, h)
    actual = _native(
        dh, h3_native, h1_native, dh1_native, dh3_native, allocation, native_state
    )
    torch.cuda.synchronize()
    _assert_exact("precomputed_dh1_unchanged", dh1_native, dh1_before_native)
    _assert_exact("precomputed_dh3_unchanged", dh3_native, dh3_before_native)
    _compare_outputs(actual, expected)
    _assert_exact("persistent_rng_state", native_state, reference_state)
    assert int(native_state[1].item()) == SUBSEQUENCE + STATE_STRIDE


def check_reuse_stream_and_restart(te_fused, m: int, h: int) -> None:
    dh, h3_initial, h1_initial = _inputs(m, h)
    allocation = _alloc_native(m, h)
    allocation_fingerprint = _allocation_fingerprint(allocation)
    native_state = _state()
    launch_stream = torch.cuda.Stream()
    for iteration in range(3):
        h3 = h3_initial.clone()
        h1 = h1_initial.clone()
        dh1 = torch.empty_like(dh)
        dh3 = torch.empty_like(dh)
        _produce_te(te_fused, dh, h3, h1, dh1, dh3)
        torch.cuda.synchronize()
        dh1_before_native = dh1.clone()
        dh3_before_native = dh3.clone()
        state_before = native_state.clone()
        reference_state = state_before.clone()
        expected = tuple(_snapshot(t) for t in _reference(dh1, dh3, reference_state))
        launch_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(launch_stream):
            actual = _native(dh, h3, h1, dh1, dh3, allocation, native_state)
        # Churn the default-stream allocator while the native launch is live.
        junk = [torch.empty_like(dh) for _ in range(3)]
        del junk
        torch.cuda.current_stream().wait_stream(launch_stream)
        torch.cuda.synchronize()
        _assert_exact("reuse_precomputed_dh1_unchanged", dh1, dh1_before_native)
        _assert_exact("reuse_precomputed_dh3_unchanged", dh3, dh3_before_native)
        if _allocation_fingerprint(allocation) != allocation_fingerprint:
            raise AssertionError("cached allocation pointers or metadata changed")
        _compare_outputs(actual, expected)
        _assert_exact("stream_rng_state", native_state, reference_state)
        if iteration == 0:
            checkpoint = native_state.clone()
            expected_second = tuple(_snapshot(t) for t in expected)
        elif iteration == 1:
            replay_target = tuple(_snapshot(t) for t in actual)
            native_state.copy_(checkpoint)
            torch.cuda.synchronize()
        else:
            _compare_outputs(actual, replay_target)


def check_encode_centric_true(te_fused) -> None:
    previous = os.environ.get("NVTE_NVFP4_ENCODE_CENTRIC")
    os.environ["NVTE_NVFP4_ENCODE_CENTRIC"] = "1"
    try:
        m, h = 512, 512
        dh, h3, h1 = _inputs(m, h)
        dh1, dh3 = torch.empty_like(dh), torch.empty_like(dh)
        _produce_te(te_fused, dh, h3, h1, dh1, dh3)
        reference_state = _state()
        native_state = _state()
        expected = tuple(
            _snapshot(t)
            for t in _reference(
                dh1, dh3, reference_state, encode_centric=True
            )
        )
        allocation = _alloc_native(m, h)
        actual = _native(dh, h3, h1, dh1, dh3, allocation, native_state)
        torch.cuda.synchronize()
        _compare_outputs(actual, expected)
        _assert_exact("encode_true_rng_state", native_state, reference_state)
    finally:
        if previous is None:
            os.environ.pop("NVTE_NVFP4_ENCODE_CENTRIC", None)
        else:
            os.environ["NVTE_NVFP4_ENCODE_CENTRIC"] = previous


def check_legacy_positional_state_abi() -> None:
    m, h = 256, 256
    dh, h3, h1 = _inputs(m, h)
    dh1, dh3 = torch.empty_like(dh), torch.empty_like(dh)
    allocation = _alloc_native(m, h)
    state = _state()
    returned = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
        dh, h3, h1, dh1, dh3,
        *allocation[:16],
        True, True, False, SEED, SUBSEQUENCE, "row", state,
    )
    torch.cuda.synchronize()
    if len(returned) != 12:
        raise AssertionError(f"legacy positional ABI returned {len(returned)} tensors")
    if int(state[1].item()) != SUBSEQUENCE + STATE_STRIDE:
        raise AssertionError("legacy positional ABI did not advance persistent state")
    remapped = (
        returned[0], returned[1], returned[4], returned[2], returned[3], returned[5],
        returned[6], returned[7], returned[10], returned[8], returned[9], returned[11],
        allocation[16], allocation[17], allocation[18],
    )
    _assert_native_aliases(remapped, allocation)
    reference_dh1, reference_dh3 = torch.empty_like(dh), torch.empty_like(dh)
    reference_allocation = _alloc_native(m, h)
    reference_state = _state()
    reference_returned = (
        tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
            dh, h3, h1, reference_dh1, reference_dh3,
            *reference_allocation[:16],
            True, True, False, SEED, SUBSEQUENCE, "row",
            rht_axes="none",
            with_random_sign_mask=False,
            derivatives_precomputed=False,
            encode_centric=True,
            persistent_rng_state=reference_state,
        )
    )
    expected = (
        reference_returned[0], reference_returned[1], reference_returned[4],
        reference_returned[2], reference_returned[3], reference_returned[5],
        reference_returned[6], reference_returned[7], reference_returned[10],
        reference_returned[8], reference_returned[9], reference_returned[11],
        reference_allocation[16], reference_allocation[17], reference_allocation[18],
    )
    expected = tuple(_snapshot(t) for t in expected)
    _compare_outputs(remapped, expected)
    _assert_exact("legacy_dh1", dh1, reference_dh1)
    _assert_exact("legacy_dh3", dh3, reference_dh3)
    _assert_exact("legacy_rng_state", state, reference_state)


def check_signed_checkpoint_state(te_fused) -> None:
    m, h = 512, 512
    dh, h3, h1 = _inputs(m, h, "outlier")
    dh1, dh3 = torch.empty_like(dh), torch.empty_like(dh)
    _produce_te(te_fused, dh, h3, h1, dh1, dh3)
    seed = -(1 << 63) + 42
    subsequence = -17
    reference_state = _state(seed, subsequence)
    native_state = _state(seed, subsequence)
    expected = tuple(
        _snapshot(t) for t in _reference(dh1, dh3, reference_state)
    )
    allocation = _alloc_native(m, h)
    actual = _native(dh, h3, h1, dh1, dh3, allocation, native_state)
    torch.cuda.synchronize()
    _compare_outputs(actual, expected)
    _assert_exact("signed_checkpoint_rng_state", native_state, reference_state)
    if int(native_state[1].item()) != subsequence + STATE_STRIDE:
        raise AssertionError("signed checkpoint state did not advance by 2^32")


def check_cuda_graph(te_fused, m: int, h: int) -> None:
    dh, h3, h1 = _inputs(m, h)
    dh1, dh3 = torch.empty_like(dh), torch.empty_like(dh)
    allocation = _alloc_native(m, h)
    graph_state = _state()
    initial_state = graph_state.clone()

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        _produce_te(te_fused, dh, h3, h1, dh1, dh3)
        _native(dh, h3, h1, dh1, dh3, allocation, graph_state)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph_state.copy_(initial_state)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _produce_te(te_fused, dh, h3, h1, dh1, dh3)
        graph_output = _native(dh, h3, h1, dh1, dh3, allocation, graph_state)
    torch.cuda.synchronize()
    graph_state.copy_(initial_state)
    torch.cuda.synchronize()

    for replay in range(2):
        reference_state = graph_state.clone()
        ref_dh1, ref_dh3 = torch.empty_like(dh), torch.empty_like(dh)
        _produce_te(te_fused, dh, h3, h1, ref_dh1, ref_dh3)
        expected = tuple(_snapshot(t) for t in _reference(ref_dh1, ref_dh3, reference_state))
        graph.replay()
        torch.cuda.synchronize()
        _compare_outputs(graph_output, expected)
        _assert_exact(f"graph_rng_state_{replay}", graph_state, reference_state)


def benchmark(
    te_fused,
    m: int,
    h: int,
    iterations: int,
    min_speedup: float,
) -> None:
    dh, h3, h1 = _inputs(m, h)
    dh1, dh3 = torch.empty_like(dh), torch.empty_like(dh)
    allocation = _alloc_native(m, h)
    native_state = _state()
    fallback_state = _state()

    for _ in range(3):
        _produce_te(te_fused, dh, h3, h1, dh1, dh3)
        _native(dh, h3, h1, dh1, dh3, allocation, native_state)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _produce_te(te_fused, dh, h3, h1, dh1, dh3)
        _native(dh, h3, h1, dh1, dh3, allocation, native_state)
    end.record()
    end.synchronize()
    native_ms = start.elapsed_time(end) / iterations

    start.record()
    for _ in range(iterations):
        _produce_te(te_fused, dh, h3, h1, dh1, dh3)
        _reference(dh1, dh3, fallback_state)
    end.record()
    end.synchronize()
    fallback_ms = start.elapsed_time(end) / iterations
    speedup = fallback_ms / native_ms
    print(
        f"BENCH M={m} H={h}: native={native_ms:.4f}ms "
        f"fallback={fallback_ms:.4f}ms speedup={speedup:.3f}x"
    )
    if speedup < min_speedup:
        raise AssertionError(
            f"native carrier speedup {speedup:.3f}x is below required "
            f"{min_speedup:.3f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-extension",
        required=True,
        help="Absolute path of the freshly built _tk_quant_localcta_v4 module",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip production shape and benchmark; never a promotion gate",
    )
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--min-speedup", type=float, default=1.20)
    parser.add_argument(
        "--scale-num",
        type=float,
        required=True,
        help="Production localCTA global-scale numerator (must be 448)",
    )
    args = parser.parse_args()

    if args.scale_num != PRODUCTION_SCALE_NUM:
        raise RuntimeError(
            f"promotion gate requires production scale numerator "
            f"{PRODUCTION_SCALE_NUM:g}, got {args.scale_num:g}"
        )
    os.environ["USE_TK_LOCALCTA_SCALE_NUM"] = f"{args.scale_num:g}"

    actual_extension = Path(tkq.__file__).resolve()
    expected_extension = Path(args.expected_extension).resolve()
    if actual_extension != expected_extension:
        raise RuntimeError(
            f"stale/wrong TK extension loaded: {actual_extension}; "
            f"expected {expected_extension}"
        )
    expected_lbt = (
        Path(__file__).resolve().parents[1]
        / "low_bits_training/quantization/fused_te_linear.py"
    ).resolve()
    if Path(fte.__file__).resolve() != expected_lbt:
        raise RuntimeError(
            f"wrong LBT worktree loaded: {Path(fte.__file__).resolve()}; "
            f"expected {expected_lbt}"
        )
    marker = getattr(tkq, "tk_localcta_silu_deriv_split2_supports_rht", None)
    if marker is None or not marker():
        raise RuntimeError("loaded TK extension lacks native split2 RHT support")
    if not hasattr(tkq, "tk_localcta_set_global_scale_num"):
        raise RuntimeError("loaded TK extension lacks global-scale control")
    tkq.tk_localcta_set_global_scale_num(args.scale_num)
    applied_scale_num = float(tkq.tk_localcta_get_global_scale_num())
    if applied_scale_num != args.scale_num:
        raise RuntimeError(
            f"extension applied scale numerator {applied_scale_num:g}, "
            f"expected {args.scale_num:g}"
        )
    if not fte._use_tk_localcta_native_paired_rht_split2(
        tkq, paired_rht_carrier=True
    ):
        raise RuntimeError("LBT production policy did not select native split2 RHT")
    te_fused = fte._get_te_fused()
    started = time.monotonic()

    for m, h in ((256, 256), (512, 512), (256, 14336), (32768, 256)):
        for kind in (("random", "zero", "tiny", "outlier") if h <= 512 else ("random",)):
            for inplace in (False, True):
                check_exact(te_fused, m, h, kind, inplace)
                print(f"PASS exact M={m} H={h} kind={kind} inplace={inplace}")

    check_reuse_stream_and_restart(te_fused, 8192, 14336)
    print("PASS cached reuse + separate stream + allocator churn + state restart")
    check_cuda_graph(te_fused, 512, 512)
    print("PASS CUDA graph replay + persistent state")
    check_encode_centric_true(te_fused)
    print("PASS encode-centric=True exact equivalence")
    check_legacy_positional_state_abi()
    print("PASS legacy positional persistent-state ABI + exact payload")
    check_signed_checkpoint_state(te_fused)
    print("PASS signed/high-bit checkpoint state + exact payload")

    if not args.quick:
        check_exact(te_fused, 32768, 14336, "random", False)
        check_exact(te_fused, 32768, 14336, "random", True)
        print("PASS full production M=32768 H=14336 distinct+inplace")
        benchmark(
            te_fused,
            32768,
            14336,
            args.benchmark_iters,
            args.min_speedup,
        )
    else:
        print("QUICK diagnostic only: production shape/performance not gated")

    label = "QUICK diagnostics" if args.quick else "ALL split2 RHT production gates"
    print(f"{label} PASS in {time.monotonic()-started:.2f}s")


if __name__ == "__main__":
    main()
