#!/usr/bin/env python3
"""Stress the production-active localCTA v4 output-buffer reuse paths.

This is deliberately a deterministic test: data/scale stochastic rounding and
RHT are disabled so every payload, block scale, and outer/chunk scale can be
compared byte-for-byte with a synchronized serialized reference.  The forced
environment below matches the relevant production routing choices from the
v4 high-water profile while keeping the test independent of the caller's
shell environment.

The default is a long soak (10,000 iterations per stress phase and producer
family).  Use ``--quick`` for a small gate suitable for a freshly built
extension.  The CUDA-graph phase treats capture as recording and validates
only completed replay outputs.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import torch


# Set routing before importing the extension; several C++ dispatch helpers read
# getenv on every call and the atomic scratch cache is initialized lazily.
_FORCED_ENV = {
    "USE_TK_LOCALCTA_V3_CONTRACT": "outer",
    "USE_TK_LOCALCTA_V3_MULTIINPUT_QUANT": "onecall",
    "USE_TK_LOCALCTA_V3_SPLIT3_MULTIINPUT_QUANT": "onecall",
    "USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER": "1",
    "USE_TK_LOCALCTA_V4_FUSED_ATOMIC_INIT": "1",
    "USE_TK_LOCALCTA_V4_REUSE_ATOMIC_SCRATCH": "1",
    "USE_TK_LOCALCTA_V4_FAST_DATA_SR": "1",
    "USE_TK_LOCALCTA_V4_FINAL_SG_OPT_DIRECT_FINAL_SCAN": "1",
    "USE_TK_LOCALCTA_V4_SPLIT3_TWO_PHASE": "1",
    "USE_TK_LOCALCTA_V4_SPLIT3_DIRECT_FINAL_SG_SCAN": "1",
    "USE_TK_LOCALCTA_V4_SPLIT3_FUSED_SG_REDUCE": "1",
    "USE_TK_LOCALCTA_V4_NHSD_REDUCED_WARP_FINALIZE": "1",
    "USE_TK_LOCALCTA_V4_DIRECT_STRICT_SPLIT2": "0",
    "USE_TK_LOCALCTA_V4_TUNED_STRICT_SPLIT2": "0",
    "USE_TK_LOCALCTA_V4_SPLIT2_PRECOMPUTE_AMAX": "1",
    "USE_TK_LOCALCTA_V4_SPLIT2_PREFINALIZE_OUTER_SG": "1",
}
os.environ.update(_FORCED_ENV)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_localcta_v4 as tkq


Tree = Any
State = Any
Triplet = tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class TensorReference:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    image: torch.Tensor


@dataclass(frozen=True)
class Case:
    name: str
    execute: Callable[[State | None], Tree]
    make_state: Callable[[], State] | None
    reconstruction_triplets: Callable[[Tree], Sequence[Triplet]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="iterations per eager, churn, graph, and multistream phase (default: 10000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="cap every stress phase at 8 iterations",
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.quick:
        args.iterations = min(args.iterations, 8)
    return args


def _flatten(tree: Tree, prefix: str = "output") -> list[tuple[str, torch.Tensor]]:
    if isinstance(tree, torch.Tensor):
        return [(prefix, tree)]
    if isinstance(tree, Mapping):
        leaves: list[tuple[str, torch.Tensor]] = []
        for key in sorted(tree, key=str):
            # Reserved for tensors whose lifetime must extend past an async
            # launch but whose contents are not defined ABI outputs.
            if key == "_lifetime":
                continue
            leaves.extend(_flatten(tree[key], f"{prefix}.{key}"))
        return leaves
    if isinstance(tree, Sequence) and not isinstance(tree, (str, bytes, bytearray)):
        leaves = []
        for index, value in enumerate(tree):
            leaves.extend(_flatten(value, f"{prefix}[{index}]"))
        return leaves
    raise TypeError(f"{prefix}: expected a tensor/container, got {type(tree).__name__}")


def _byte_image(tensor: torch.Tensor) -> torch.Tensor:
    # Split3 returns strided views into concatenated FP4 storage.  PyTorch has
    # no Float4 copy_ kernel for materializing such a view, but its byte view is
    # fully supported and preserves the logical payload ordering we compare.
    return tensor.detach().view(torch.uint8).contiguous()


def _materialize_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a logical tensor through bytes, including strided FP4 views."""
    tensor = tensor.detach()
    if tensor.is_contiguous():
        return tensor
    contiguous = torch.empty(
        tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype
    )
    contiguous.view(torch.uint8).copy_(tensor.view(torch.uint8))
    return contiguous


def _snapshot(tree: Tree) -> dict[str, TensorReference]:
    references: dict[str, TensorReference] = {}
    for path, tensor in _flatten(tree):
        references[path] = TensorReference(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            image=_byte_image(tensor).clone(),
        )
    return references


def _accumulate_mismatches(
    counter: torch.Tensor,
    tree: Tree,
    references: Mapping[str, TensorReference],
) -> None:
    leaves = dict(_flatten(tree))
    if leaves.keys() != references.keys():
        missing = sorted(references.keys() - leaves.keys())
        extra = sorted(leaves.keys() - references.keys())
        raise AssertionError(f"output tree changed: missing={missing}, extra={extra}")

    for path, reference in references.items():
        tensor = leaves[path]
        metadata = (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        expected = (reference.shape, reference.stride, reference.dtype)
        if metadata != expected:
            raise AssertionError(
                f"{path}: metadata changed: actual={metadata}, expected={expected}"
            )
        counter.add_(torch.count_nonzero(_byte_image(tensor) != reference.image))


def _assert_equal_now(
    case: Case,
    phase: str,
    tree: Tree,
    references: Mapping[str, TensorReference],
) -> None:
    counter = torch.zeros((), device="cuda", dtype=torch.int64)
    _accumulate_mismatches(counter, tree, references)
    mismatches = int(counter.item())
    if mismatches:
        raise AssertionError(
            f"{case.name}/{phase}: {mismatches} output bytes differ from serialized reference"
        )


def _assert_counter_zero(case: Case, phase: str, counter: torch.Tensor) -> None:
    mismatches = int(counter.item())
    if mismatches:
        raise AssertionError(
            f"{case.name}/{phase}: {mismatches} output bytes differ from serialized reference"
        )


def _poison_allocator(
    references: Mapping[str, TensorReference], poison_byte: int
) -> None:
    """Prime matching CUDA allocator bins, then release them for the next ABI call."""
    junk = [
        torch.empty(reference.image.numel(), device="cuda", dtype=torch.uint8)
        for reference in references.values()
        if reference.image.numel()
    ]
    for tensor in junk:
        tensor.fill_(poison_byte)
    junk.clear()


def _poison_state(state: State | None, poison_byte: int) -> None:
    if state is None:
        return
    for path, tensor in _flatten(state, "state"):
        if not tensor.is_contiguous():
            raise AssertionError(f"{path}: in-place stress state must be contiguous")
        tensor.view(torch.uint8).fill_(poison_byte)


def _new_state(case: Case, poison_byte: int | None = None) -> State | None:
    state = case.make_state() if case.make_state is not None else None
    if poison_byte is not None:
        _poison_state(state, poison_byte)
    return state


def _address_signature(tree: Tree) -> tuple[tuple[str, int], ...]:
    return tuple((path, int(tensor.data_ptr())) for path, tensor in _flatten(tree))


def _standard_triplets(prefix: str, outputs: Sequence[torch.Tensor]) -> list[Triplet]:
    if len(outputs) != 6:
        raise AssertionError(f"{prefix}: expected six quantization outputs, got {len(outputs)}")
    row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg = outputs
    return [
        (f"{prefix}.row", "row", row_fp4, row_sc, row_sg),
        (f"{prefix}.col", "col", col_fp4, col_sc, col_sg),
    ]


def _assert_finite_reconstruction(case: Case, output: Tree) -> None:
    triplets = list(case.reconstruction_triplets(output))
    if not triplets:
        raise AssertionError(f"{case.name}: no reconstruction triplets registered")
    for label, orientation, fp4, block_scales, super_scales in triplets:
        # Split3 exposes row pieces as strided views into concatenated storage;
        # the diagnostic decoder intentionally accepts only contiguous layouts.
        fp4 = _materialize_contiguous(fp4)
        block_scales = _materialize_contiguous(block_scales)
        super_scales = _materialize_contiguous(super_scales)
        if orientation == "row":
            reconstructed = tkq.tk_localcta_reconstruct_row(
                fp4, block_scales, super_scales
            )
        elif orientation == "col":
            reconstructed = tkq.tk_localcta_reconstruct_col(
                fp4, block_scales, super_scales
            )
        else:
            raise AssertionError(f"{label}: unknown reconstruction orientation {orientation}")

        values = reconstructed.float()
        if not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{case.name}/{label}: non-finite reconstruction")
        if float(values.abs().amax()) == 0.0:
            raise AssertionError(f"{case.name}/{label}: reconstruction is identically zero")

        # A small dense consumer catches finite inputs that nevertheless produce
        # NaN/Inf during ordinary accumulation.  This is a proxy GEMM over the
        # decoded contract, not a replacement for a production FP4 GEMM gate.
        probe = torch.ones(
            (values.size(1), 4), device=values.device, dtype=torch.float32
        )
        proxy_gemm = values @ probe
        if not bool(torch.isfinite(proxy_gemm).all()):
            raise AssertionError(f"{case.name}/{label}: non-finite proxy GEMM")


def _patterned(shape: tuple[int, ...], phase: float) -> torch.Tensor:
    count = 1
    for extent in shape:
        count *= extent
    index = torch.arange(count, device="cuda", dtype=torch.float32)
    values = (
        0.61 * torch.sin(index * 0.013 + phase)
        + 0.29 * torch.cos(index * 0.031 - phase * 0.7)
        + 0.07 * torch.sin(index * 0.0017 + phase * 1.9)
    )
    return values.reshape(shape).to(torch.bfloat16).contiguous()


def _build_cases(size: int) -> list[Case]:
    matrix = _patterned((size, size), 0.11)
    weight = _patterned((size, size), 0.37)
    split3_inputs = tuple(
        _patterned((size, size), phase) for phase in (0.53, 0.79, 1.07)
    )
    dh = _patterned((size, size), 1.31)
    h3 = _patterned((size, size), 1.61)
    h1 = _patterned((size, size), 1.97)
    nhsd = _patterned((1, 4, size, 64), 2.23)

    def execute_weight(_state: State | None) -> Tree:
        return tkq.tk_localcta_quantize_weight_2d(weight)

    def execute_opt_atomic(_state: State | None) -> Tree:
        # The generic ABI is forced through the production atomic-final route;
        # the opt ABI exercises the same patched output stage before serialized
        # outer-scale finalization.
        return {
            "atomic_final": tkq.tk_localcta_quantize_for_gemm(
                matrix, True, False
            ),
            "opt": tkq.tk_localcta_quantize_for_gemm_opt(
                matrix,
                True,
                False,
                False,
                False,
                "none",
                False,
                0,
                0,
                "none",
            ),
        }

    def opt_atomic_triplets(output: Tree) -> list[Triplet]:
        return (
            _standard_triplets("atomic_final", output["atomic_final"])
            + _standard_triplets("opt", output["opt"])
        )

    def execute_split3(_state: State | None) -> Tree:
        return tkq.tk_localcta_group_quantize_dim1_split3_for_gemm(
            *split3_inputs, False, 0, 0, "none"
        )

    def split3_triplets(output: Tree) -> list[Triplet]:
        triplets: list[Triplet] = []
        for split in range(3):
            triplets.append(
                (
                    f"split{split}.row",
                    "row",
                    output[0][split],
                    output[1][split],
                    output[2][split],
                )
            )
            triplets.append(
                (
                    f"split{split}.col",
                    "col",
                    output[3][split],
                    output[4][split],
                    output[5][split],
                )
            )
        return triplets

    def make_split2_state() -> State:
        return {
            "buffers": tuple(
                tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
                    size, size, dh.device
                )
            ),
            "dh1": torch.empty_like(dh),
            "dh3": torch.empty_like(dh),
        }

    def execute_split2(state: State | None) -> Tree:
        if state is None:
            raise AssertionError("split2 in-place ABI requires preallocated state")
        buffers = state["buffers"]
        returned = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
            dh,
            h3,
            h1,
            state["dh1"],
            state["dh3"],
            *buffers,
            True,
            False,
            False,
            0,
            0,
            "none",
        )
        if len(returned) != 12:
            raise AssertionError(f"split2 ABI returned {len(returned)} tensors, expected 12")
        for index, tensor in enumerate(returned):
            if tensor.data_ptr() != buffers[index].data_ptr():
                raise AssertionError(f"split2 returned output {index} is not its in-place buffer")
        # The production prefinalized-outer-SG route deliberately leaves the four
        # chunk-SG scratch buffers undefined.  They are poisoned above (so any
        # accidental consumption affects this test), but only the twelve defined
        # payload/scale/SG returns and two derivative outputs are compared.
        return {
            "_lifetime": buffers[12:16],
            "quant": tuple(returned),
            "dh1": state["dh1"],
            "dh3": state["dh3"],
        }

    def split2_triplets(output: Tree) -> list[Triplet]:
        buffers = output["quant"]
        return (
            _standard_triplets("dh1", buffers[0:6])
            + _standard_triplets("dh3", buffers[6:12])
        )

    def execute_nhsd(_state: State | None) -> Tree:
        return tkq.tk_localcta_quantize_nhsd_wo_for_gemm(nhsd, False)

    return [
        Case(
            name="weight_2d",
            execute=execute_weight,
            make_state=None,
            reconstruction_triplets=lambda output: _standard_triplets(
                "weight_2d", output
            ),
        ),
        Case(
            name="opt_atomic_final",
            execute=execute_opt_atomic,
            make_state=None,
            reconstruction_triplets=opt_atomic_triplets,
        ),
        Case(
            name="split3_final_sg",
            execute=execute_split3,
            make_state=None,
            reconstruction_triplets=split3_triplets,
        ),
        Case(
            name="split2_raw_inplace",
            execute=execute_split2,
            make_state=make_split2_state,
            reconstruction_triplets=split2_triplets,
        ),
        Case(
            name="nhsd_wo",
            execute=execute_nhsd,
            make_state=None,
            reconstruction_triplets=lambda output: _standard_triplets(
                "nhsd", output
            ),
        ),
    ]


def _serialized_reference(case: Case) -> dict[str, TensorReference]:
    # Preserve the actual first call, then compare it with a separately launched,
    # fully synchronized serialized reference.  This catches lazy-init-only output
    # corruption rather than warming it away before validation.
    first_state = _new_state(case, 0xA5)
    first = case.execute(first_state)
    torch.cuda.synchronize()

    reference_state = _new_state(case, 0x3C)
    reference_output = case.execute(reference_state)
    torch.cuda.synchronize()
    references = _snapshot(reference_output)
    _assert_equal_now(case, "first-call", first, references)
    _assert_finite_reconstruction(case, reference_output)
    print(
        f"[{case.name}] first-call/serialized: PASS "
        f"({len(references)} tensor leaves)"
    )
    return references


def _stress_eager(
    case: Case,
    references: Mapping[str, TensorReference],
    iterations: int,
) -> None:
    counter = torch.zeros((), device="cuda", dtype=torch.int64)
    for _ in range(iterations):
        output = case.execute(_new_state(case))
        _accumulate_mismatches(counter, output, references)
    _assert_counter_zero(case, "eager", counter)
    print(f"[{case.name}] eager: PASS ({iterations} iterations)")


def _stress_poisoned_churn(
    case: Case,
    references: Mapping[str, TensorReference],
    iterations: int,
) -> None:
    counter = torch.zeros((), device="cuda", dtype=torch.int64)
    keepalive: deque[tuple[State | None, Tree]] = deque(maxlen=4)
    signatures: set[tuple[tuple[str, int], ...]] = set()
    for iteration in range(iterations):
        poison_byte = (0x5A + 131 * iteration) & 0xFF
        _poison_allocator(references, poison_byte)
        state = _new_state(case, poison_byte)
        output = case.execute(state)
        _accumulate_mismatches(counter, output, references)
        signatures.add(_address_signature(output))
        keepalive.append((state, output))
    _assert_counter_zero(case, "poisoned-churn", counter)
    if iterations > 1 and len(signatures) < 2:
        raise AssertionError(
            f"{case.name}/poisoned-churn: allocator produced no output-address churn"
        )
    print(
        f"[{case.name}] poisoned/reallocated: PASS "
        f"({iterations} iterations, {len(signatures)} address signatures)"
    )


def _stress_cuda_graph(
    case: Case,
    references: Mapping[str, TensorReference],
    iterations: int,
) -> None:
    current = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current)
    warmups: list[tuple[State | None, Tree]] = []
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            state = _new_state(case)
            warmups.append((state, case.execute(state)))
    current.wait_stream(warmup_stream)
    torch.cuda.synchronize()
    warmups.clear()

    graph_state = _new_state(case, 0xC3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = case.execute(graph_state)

    # Capture records the graph but is not a replay result.  In particular,
    # producer-local initialization may be represented by captured nodes whose
    # replay ordering is the contract under test.  Validate exactly the requested
    # number of completed replays, beginning with the first replay.
    counter = torch.zeros((), device="cuda", dtype=torch.int64)
    for _ in range(iterations):
        graph.replay()
        _accumulate_mismatches(counter, graph_output, references)
    _assert_counter_zero(case, "cuda-graph", counter)
    print(
        f"[{case.name}] CUDA graph: PASS "
        f"({iterations} replays; capture recording excluded)"
    )


def _stress_multistream(
    case: Case,
    references: Mapping[str, TensorReference],
    iterations: int,
) -> None:
    current = torch.cuda.current_stream()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    counters = (
        torch.zeros((), device="cuda", dtype=torch.int64),
        torch.zeros((), device="cuda", dtype=torch.int64),
    )
    live: list[tuple[State | None, Tree] | None] = [None, None]
    for stream in streams:
        stream.wait_stream(current)

    for _ in range(iterations):
        for stream_index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                state = _new_state(case)
                output = case.execute(state)
                _accumulate_mismatches(
                    counters[stream_index], output, references
                )
                # Release a stream's preceding allocation only while that same
                # stream is current; this keeps allocator lifetime ordering clear.
                live[stream_index] = (state, output)

    for stream in streams:
        stream.synchronize()
    for counter in counters:
        _assert_counter_zero(case, "multistream", counter)
    live.clear()
    print(
        f"[{case.name}] independent multistream: PASS "
        f"({iterations} paired launches)"
    )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"localCTA v4 stress requires a Blackwell GPU, got capability {major}.{minor}"
        )

    torch.manual_seed(20260820)
    torch.cuda.manual_seed_all(20260820)
    size = 256
    cases = _build_cases(size)
    torch.cuda.synchronize()

    print(
        "localCTA active output-reuse stress: "
        f"device={torch.cuda.get_device_name()} size={size} "
        f"iterations={args.iterations} deterministic=1"
    )
    for case in cases:
        references = _serialized_reference(case)
        _stress_eager(case, references, args.iterations)
        _stress_poisoned_churn(case, references, args.iterations)
        _stress_cuda_graph(case, references, args.iterations)
        _stress_multistream(case, references, args.iterations)

    print(
        "localCTA active output-reuse stress: PASS "
        f"({len(cases)} producer families, {args.iterations} iterations/phase)"
    )


if __name__ == "__main__":
    main()
