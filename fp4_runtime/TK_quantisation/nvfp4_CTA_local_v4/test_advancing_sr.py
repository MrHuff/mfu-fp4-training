#!/usr/bin/env python3
"""Smoke-test eager and CUDA-graph localCTA SR state advancement."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_localcta_v4 as tkq


_STRIDE = 1 << 32


def _quantize(
    x: torch.Tensor,
    stochastic: bool,
    persistent_rng_state: torch.Tensor | None = None,
) -> torch.Tensor:
    args = (
        x,
        True,
        True,
        stochastic,
        False,
        "none",
        False,
        123,
        456,
        "both",
    )
    result = tkq.tk_localcta_quantize_for_gemm_opt(
        *args,
        *(() if persistent_rng_state is None else (persistent_rng_state,)),
    )
    return result[0].view(torch.uint8)


def _new_state(seed: int, subsequence: int) -> torch.Tensor:
    return torch.tensor(
        [seed, subsequence], dtype=torch.int64, device="cuda"
    )


def _assert_explicit_state_restart_and_stream_identity(x: torch.Tensor) -> None:
    state = _new_state(1001, 456)
    first = _quantize(x, True, state).clone()
    after_first = state.clone()
    assert int(state[1].item()) == 456 + _STRIDE

    second = _quantize(x, True, state).clone()
    assert int(state[1].item()) == 456 + 2 * _STRIDE
    state.copy_(after_first)
    replayed_second = _quantize(x, True, state).clone()
    torch.cuda.synchronize()
    assert torch.equal(second, replayed_second)
    assert not torch.equal(first, second)

    # Reversing host launch order across streams must not reassign either
    # logical producer's subsequence.
    state_a = _new_state(2001, 900)
    state_b = _new_state(2002, 900)
    initial_a = state_a.clone()
    initial_b = state_b.clone()
    expected_a = _quantize(x, True, state_a).clone()
    expected_b = _quantize(x, True, state_b).clone()
    torch.cuda.synchronize()
    state_a.copy_(initial_a)
    state_b.copy_(initial_b)
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a.wait_stream(torch.cuda.current_stream())
    stream_b.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream_b):
        actual_b = _quantize(x, True, state_b).clone()
    with torch.cuda.stream(stream_a):
        actual_a = _quantize(x, True, state_a).clone()
    torch.cuda.current_stream().wait_stream(stream_a)
    torch.cuda.current_stream().wait_stream(stream_b)
    torch.cuda.synchronize()
    assert torch.equal(expected_a, actual_a)
    assert torch.equal(expected_b, actual_b)


def _assert_explicit_state_graph_replay(x: torch.Tensor) -> None:
    state = _new_state(3001, 1200)
    before_capture = state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _quantize(x, True, state)
    torch.cuda.synchronize()

    # Training integration restores this snapshot after synthetic graph
    # warmup/capture.  The captured kernel retains the live state pointer.
    state.copy_(before_capture)
    torch.cuda.synchronize()
    graph.replay()
    first = graph_output.clone()
    torch.cuda.synchronize()
    assert int(state[1].item()) == 1200 + _STRIDE
    graph.replay()
    second = graph_output.clone()
    torch.cuda.synchronize()
    assert int(state[1].item()) == 1200 + 2 * _STRIDE
    assert not torch.equal(first, second)

    state.copy_(before_capture)
    torch.cuda.synchronize()
    graph.replay()
    restarted_first = graph_output.clone()
    torch.cuda.synchronize()
    assert torch.equal(first, restarted_first)


def _assert_explicit_state_validation(x: torch.Tensor) -> None:
    invalid_states = (
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda"),
        torch.empty(4, dtype=torch.int64, device="cuda")[::2],
        torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda"),
    )
    for state in invalid_states:
        try:
            _quantize(x, True, state)
        except RuntimeError as error:
            assert "localCTA explicit RNG state" in str(error), error
        else:
            raise AssertionError(f"invalid explicit RNG state was accepted: {state}")


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)

    deterministic_a = _quantize(x, False).clone()
    deterministic_b = _quantize(x, False).clone()
    assert torch.equal(deterministic_a, deterministic_b)

    stochastic_a = _quantize(x, True).clone()
    stochastic_b = _quantize(x, True).clone()
    assert not torch.equal(stochastic_a, stochastic_b)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _quantize(x, True)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _quantize(x, True)
    graph.replay()
    graph_a = graph_output.clone()
    graph.replay()
    graph_b = graph_output.clone()
    torch.cuda.synchronize()
    assert not torch.equal(graph_a, graph_b)

    _assert_explicit_state_restart_and_stream_identity(x)
    _assert_explicit_state_graph_replay(x)
    _assert_explicit_state_validation(x)

    print("localCTA advancing stochastic rounding: "
          "eager=PASS graph=PASS deterministic=PASS explicit_restart=PASS "
          "stream_identity=PASS validation=PASS")


if __name__ == "__main__":
    main()
