#!/usr/bin/env python3
"""Smoke-test eager and CUDA-graph stochastic-rounding state advancement."""

import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_v5 as tkq


def _generic_quantize(x: torch.Tensor, stochastic: bool) -> torch.Tensor:
    return tkq.tk_quantize_for_gemm_opt(
        x,
        True,
        True,
        stochastic,
        False,
        "none",
        False,
        123,
        456,
    )[0].view(torch.uint8)


def _assert_generic_axis_isolation(x: torch.Tensor) -> None:
    outputs = {}
    for axes in ("none", "row", "col", "both"):
        outputs[axes] = tkq.tk_quantize_for_gemm_opt(
            x,
            True,
            True,
            axes != "none",
            False,
            "none",
            False,
            123,
            456,
            448.0,
            axes,
        )
    torch.cuda.synchronize()

    def same(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        return torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))

    assert not same(outputs["row"][0], outputs["none"][0])
    assert same(outputs["row"][2], outputs["none"][2])
    assert same(outputs["col"][0], outputs["none"][0])
    assert not same(outputs["col"][2], outputs["none"][2])
    assert not same(outputs["both"][0], outputs["none"][0])
    assert not same(outputs["both"][2], outputs["none"][2])


def _quantize_with_scale_target(
    x: torch.Tensor,
    global_scale_target: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = tkq.tk_quantize_for_gemm_opt(
        x,
        True,
        True,
        True,
        False,
        "none",
        False,
        123,
        456,
        global_scale_target,
    )
    return result[0], result[1], result[4]


def _sqrelu_deriv_quantize(
    dh: torch.Tensor,
    h1_raw: torch.Tensor,
    stochastic: bool,
) -> torch.Tensor:
    return tkq.tk_sqrelu_deriv_quantize_for_gemm_opt(
        dh,
        h1_raw,
        True,
        True,
        stochastic,
        False,
        "row",
        False,
        123,
        456,
    )[0].view(torch.uint8)


def _row_only_quantize(x: torch.Tensor) -> torch.Tensor:
    return tkq.tk_quantize_row_for_gemm_sr(x, True)[0].view(torch.uint8)


def _split3_quantize(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    stochastic: bool,
) -> torch.Tensor:
    result = tkq.tk_group_quantize_dim1_split3_for_gemm(
        *inputs,
        stochastic,
        123,
        456,
    )
    return torch.cat(
        (
            result[5].view(torch.uint8).flatten(),
            result[7].view(torch.uint8).flatten(),
        )
    )


def _assert_split3_axis_isolation(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    outputs = {}
    for axes in ("none", "row", "col", "both"):
        outputs[axes] = tkq.tk_group_quantize_dim1_split3_for_gemm(
            *inputs,
            axes != "none",
            123,
            456,
            axes,
        )
    torch.cuda.synchronize()

    def same(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        return torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))

    assert not same(outputs["row"][5], outputs["none"][5])
    assert same(outputs["row"][7], outputs["none"][7])
    assert same(outputs["col"][5], outputs["none"][5])
    assert not same(outputs["col"][7], outputs["none"][7])
    assert not same(outputs["both"][5], outputs["none"][5])
    assert not same(outputs["both"][7], outputs["none"][7])


def _assert_split3_graph_advances(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager = tkq.tk_group_quantize_dim1_split3_for_gemm(
            *inputs, True, 123, 456
        )
        state = tkq.tk_group_quantize_dim1_split3_capture_alloc(
            *inputs,
            eager[5].view(torch.uint8),
            eager[7].view(torch.uint8),
            eager[2],
            [tensor.view(torch.uint8) for tensor in eager[1]],
            [tensor.view(torch.uint8) for tensor in eager[3]],
            [tensor.view(torch.uint8) for tensor in eager[4]],
            eager[6].view(torch.uint8),
            eager[8].view(torch.uint8),
            eager[9],
            eager[10],
            True,
            123,
            456,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_output = tkq.tk_group_quantize_dim1_split3_launch(
                *inputs, state
            )
    graph.replay()
    stream.synchronize()
    first = torch.cat(
        (
            graph_output[5].view(torch.uint8).flatten(),
            graph_output[7].view(torch.uint8).flatten(),
        )
    ).clone()
    graph.replay()
    stream.synchronize()
    second = torch.cat(
        (
            graph_output[5].view(torch.uint8).flatten(),
            graph_output[7].view(torch.uint8).flatten(),
        )
    ).clone()
    assert not torch.equal(first, second)


def _assert_split3_eager_scratch_is_retained(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for index in range(256):
        result = tkq.tk_group_quantize_dim1_split3_for_gemm(
            *inputs, True, 123, 456
        )
        assert len(result) == 15
        # Callers are allowed to drop every returned scratch tensor.
        result = None
        if index % 8 == 7:
            torch.empty(
                (8 * 1024 * 1024,), device="cuda", dtype=torch.uint8
            )
        if index % 32 == 31:
            torch.cuda.synchronize()
    torch.cuda.synchronize()


def _assert_split3_two_pass_scale_parity(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    old_policy = os.environ.get("USE_TK_SPLIT3_TWO_PASS")
    try:
        os.environ["USE_TK_SPLIT3_TWO_PASS"] = "0"
        legacy = tkq.tk_group_quantize_dim1_split3_for_gemm(
            *inputs, True, 123, 456
        )
        os.environ["USE_TK_SPLIT3_TWO_PASS"] = "1"
        two_pass = tkq.tk_group_quantize_dim1_split3_for_gemm(
            *inputs, True, 123, 456
        )
        torch.cuda.synchronize()
    finally:
        if old_policy is None:
            os.environ.pop("USE_TK_SPLIT3_TWO_PASS", None)
        else:
            os.environ["USE_TK_SPLIT3_TWO_PASS"] = old_policy

    assert torch.equal(legacy[2], two_pass[2])
    for index in (1, 4):
        for legacy_scale, two_pass_scale in zip(legacy[index], two_pass[index]):
            assert torch.equal(
                legacy_scale.view(torch.uint8),
                two_pass_scale.view(torch.uint8),
            )
    assert torch.equal(
        torch.cat(
            [tensor.contiguous().view(torch.uint8) for tensor in two_pass[0]],
            dim=1,
        ),
        two_pass[5].view(torch.uint8),
    )
    assert torch.equal(
        torch.cat(
            [tensor.contiguous().view(torch.uint8) for tensor in two_pass[3]],
            dim=0,
        ),
        two_pass[7].view(torch.uint8),
    )


def _fused_norm_quantize(
    x: torch.Tensor,
    gamma: torch.Tensor,
    stochastic: bool,
) -> torch.Tensor:
    return tkq.tk_fused_norm_quantize_opt(
        x,
        gamma,
        1e-5,
        True,
        True,
        stochastic,
        False,
        "row",
        False,
        123,
        456,
    )[0].view(torch.uint8)


def _silu_deriv_launch(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    stochastic: bool,
) -> torch.Tensor:
    return tkq.tk_silu_deriv_quantize_split_for_gemm_opt_launch(
        dh,
        h3,
        h1_raw,
        True,
        True,
        stochastic,
        False,
        "row",
        False,
        123,
        456,
        *outputs,
    )[0].view(torch.uint8)


def _assert_eager_advances(fn) -> None:
    deterministic_a = fn(False).clone()
    deterministic_b = fn(False).clone()
    assert torch.equal(deterministic_a, deterministic_b)

    stochastic_a = fn(True).clone()
    stochastic_b = fn(True).clone()
    assert not torch.equal(stochastic_a, stochastic_b)


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    aux = torch.randn_like(x)
    h1_raw = -aux
    gamma = torch.randn((x.shape[1],), device="cuda", dtype=torch.bfloat16)

    _assert_generic_axis_isolation(x)
    _assert_eager_advances(lambda stochastic: _generic_quantize(x, stochastic))
    default = _quantize_with_scale_target(x, 448.0)
    candidate = _quantize_with_scale_target(x, 256.0)
    expected_scale_shape = (x.shape[0] // 128, x.shape[1] // 64, 512)
    assert default[1].shape == expected_scale_shape
    assert candidate[1].shape == expected_scale_shape
    assert all(
        torch.isfinite(t.float()).all()
        for t in (default[1], default[2], candidate[1], candidate[2])
    )
    assert not torch.equal(
        default[1].view(torch.uint8), candidate[1].view(torch.uint8)
    )
    assert not torch.equal(default[2], candidate[2])
    try:
        _quantize_with_scale_target(x, 1024.0)
    except RuntimeError as error:
        assert "global_scale_target" in str(error)
    else:
        raise AssertionError("unsafe global scale target was not rejected")
    _assert_eager_advances(
        lambda stochastic: _sqrelu_deriv_quantize(x, aux, stochastic)
    )
    row_only_a = _row_only_quantize(x).clone()
    row_only_b = _row_only_quantize(x).clone()
    assert not torch.equal(row_only_a, row_only_b)
    _assert_eager_advances(
        lambda stochastic: _fused_norm_quantize(x, gamma, stochastic)
    )
    split3_inputs = (
        torch.randn((512, 512), device="cuda", dtype=torch.bfloat16),
        torch.randn((512, 256), device="cuda", dtype=torch.bfloat16),
        torch.randn((512, 256), device="cuda", dtype=torch.bfloat16),
    )
    _assert_split3_axis_isolation(split3_inputs)
    _assert_eager_advances(
        lambda stochastic: _split3_quantize(split3_inputs, stochastic)
    )
    _assert_split3_two_pass_scale_parity(split3_inputs)
    persistent_split3_inputs = (
        torch.randn((32768, 512), device="cuda", dtype=torch.bfloat16),
        torch.randn((32768, 256), device="cuda", dtype=torch.bfloat16),
        torch.randn((32768, 256), device="cuda", dtype=torch.bfloat16),
    )
    _assert_split3_eager_scratch_is_retained(persistent_split3_inputs)
    old_graph_policy = os.environ.get("USE_CUDA_GRAPH")
    os.environ["USE_CUDA_GRAPH"] = "1"
    try:
        _assert_split3_graph_advances(split3_inputs)
    finally:
        if old_graph_policy is None:
            os.environ.pop("USE_CUDA_GRAPH", None)
        else:
            os.environ["USE_CUDA_GRAPH"] = old_graph_policy

    silu_outputs = tkq.tk_silu_deriv_quantize_split_for_gemm_opt_alloc(
        x.shape[0], x.shape[1], x.device
    )
    _assert_eager_advances(
        lambda stochastic: _silu_deriv_launch(
            x, aux, h1_raw, silu_outputs, stochastic
        )
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _silu_deriv_launch(x, aux, h1_raw, silu_outputs, True)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _silu_deriv_launch(x, aux, h1_raw, silu_outputs, True)
    graph.replay()
    graph_a = graph_output.clone()
    graph.replay()
    graph_b = graph_output.clone()
    torch.cuda.synchronize()
    assert not torch.equal(graph_a, graph_b)

    print(
        "advancing stochastic rounding: generic=PASS row_only=PASS "
        "fused_norm=PASS split3=PASS split3_lifetime=PASS split3_graph=PASS "
        "sqrelu=PASS silu_graph=PASS "
        "deterministic=PASS"
    )


if __name__ == "__main__":
    main()
