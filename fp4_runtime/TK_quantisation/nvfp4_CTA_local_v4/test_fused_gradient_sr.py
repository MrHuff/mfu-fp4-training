#!/usr/bin/env python3
"""Validate advancing, axis-selective SR in fused localCTA gradient producers."""

import os
from pathlib import Path
import sys

import torch


os.environ["USE_TK_LOCALCTA_V4_FAST_DATA_SR"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_localcta_v4 as tkq


_CONTRACT_ENV = "USE_TK_LOCALCTA_V3_CONTRACT"


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _same(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return torch.equal(_bytes(lhs), _bytes(rhs))


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _split2_once(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1: torch.Tensor,
    dh1: torch.Tensor,
    dh3: torch.Tensor,
    buffers,
    axes: str,
    persistent_rng_state: torch.Tensor | None = None,
) -> None:
    args = (
        dh,
        h3,
        h1,
        dh1,
        dh3,
        *buffers,
        True,
        axes != "none",
        False,
        123,
        456,
        axes,
    )
    tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
        *args,
        *(() if persistent_rng_state is None else (persistent_rng_state,)),
    )


def _assert_split2_advances() -> None:
    size = 512
    dh = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
    h3 = torch.randn_like(dh)
    h1 = torch.randn_like(dh)
    dh1 = torch.empty_like(dh)
    dh3 = torch.empty_like(dh)
    buffers = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
        size, size, dh.device
    )

    _split2_once(dh, h3, h1, dh1, dh3, buffers, "none")
    deterministic_row = buffers[0].clone()
    deterministic_col = buffers[2].clone()
    _split2_once(dh, h3, h1, dh1, dh3, buffers, "none")
    assert _same(deterministic_row, buffers[0])
    assert _same(deterministic_col, buffers[2])

    _split2_once(dh, h3, h1, dh1, dh3, buffers, "row")
    stochastic_row = buffers[0].clone()
    stochastic_col = buffers[2].clone()
    _split2_once(dh, h3, h1, dh1, dh3, buffers, "row")
    assert not _same(stochastic_row, buffers[0])
    assert _same(stochastic_col, buffers[2])

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            _split2_once(dh, h3, h1, dh1, dh3, buffers, "row")
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _split2_once(dh, h3, h1, dh1, dh3, buffers, "row")
    graph.replay()
    graph_row_a = buffers[0].clone()
    graph_col_a = buffers[2].clone()
    graph.replay()
    graph_row_b = buffers[0].clone()
    graph_col_b = buffers[2].clone()
    torch.cuda.synchronize()
    assert not _same(graph_row_a, graph_row_b)
    assert _same(graph_col_a, graph_col_b)

    explicit_state = torch.tensor(
        [991, 456], dtype=torch.int64, device=dh.device
    )
    _split2_once(
        dh, h3, h1, dh1, dh3, buffers, "row", explicit_state
    )
    assert int(explicit_state[1].item()) == 456 + (1 << 32)


def _split3_once(inputs, axes: str, persistent_rng_state=None):
    args = (
        *inputs,
        axes != "none",
        123,
        456,
        axes,
    )
    return tkq.tk_localcta_group_quantize_dim1_split3_for_gemm(
        *args,
        *(() if persistent_rng_state is None else (persistent_rng_state,)),
    )


def _assert_split3_axes_advance() -> None:
    size = 512
    inputs = tuple(
        torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )

    deterministic_a = _split3_once(inputs, "none")
    deterministic_b = _split3_once(inputs, "none")
    assert _same(deterministic_a[6], deterministic_b[6])
    assert _same(deterministic_a[9], deterministic_b[9])

    row_a = _split3_once(inputs, "row")
    row_b = _split3_once(inputs, "row")
    assert not _same(row_a[6], row_b[6])
    assert _same(row_a[9], row_b[9])

    col_a = _split3_once(inputs, "col")
    col_b = _split3_once(inputs, "col")
    assert _same(col_a[6], col_b[6])
    assert not _same(col_a[9], col_b[9])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _split3_once(inputs, "row")
    graph.replay()
    graph_row_a = graph_output[6].clone()
    graph_col_a = graph_output[9].clone()
    graph.replay()
    graph_row_b = graph_output[6].clone()
    graph_col_b = graph_output[9].clone()
    torch.cuda.synchronize()
    assert not _same(graph_row_a, graph_row_b)
    assert _same(graph_col_a, graph_col_b)

    explicit_state = torch.tensor(
        [992, 456], dtype=torch.int64, device=inputs[0].device
    )
    _split3_once(inputs, "row", explicit_state)
    assert int(explicit_state[1].item()) == 456 + (1 << 32)


def _assert_inverse_rope_split3_advances() -> None:
    size = 512
    inputs = tuple(
        torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    angles = torch.randn((size, 32), device="cuda", dtype=torch.float32)
    rope = torch.stack((angles.cos(), angles.sin()), dim=-1).contiguous()

    def quantize():
        return tkq.tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64(
            *inputs,
            rope,
            size,
            True,
            False,
            "none",
            False,
            123,
            456,
            "row",
        )

    first = quantize()
    second = quantize()
    assert not _same(first[6], second[6])
    assert _same(first[9], second[9])


def _assert_tiny_nonzero_tiles_keep_relative_scale() -> None:
    size = 256
    previous_contract = os.environ.pop(_CONTRACT_ENV, None)
    try:
        tkq.tk_localcta_set_global_scale_num(774.0)
        dh = torch.full(
            (size, size), 1.0e-10, device="cuda", dtype=torch.bfloat16
        )
        h1 = torch.ones_like(dh)
        h3 = torch.zeros_like(dh)
        dh1 = torch.empty_like(dh)
        dh3 = torch.empty_like(dh)
        buffers = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
            size, size, dh.device
        )
        outputs = tkq.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
            dh,
            h3,
            h1,
            dh1,
            dh3,
            *buffers,
            True,
            True,
            False,
            123,
            456,
            "row",
        )
        row_fp4, row_sc, _, _, row_sg, _ = outputs[6:12]
        max_payload = torch.empty_like(row_fp4)
        max_payload.view(torch.uint8).fill_(0x77)
        decoded_ceiling = tkq.tk_localcta_reconstruct_row(
            max_payload, row_sc, row_sg
        ).float().abs()
        exact_amax = dh3.float().abs().amax()
        min_relative_ceiling = decoded_ceiling.amin() / exact_amax
        assert torch.isfinite(decoded_ceiling).all()
        assert min_relative_ceiling > 0.9, min_relative_ceiling
    finally:
        tkq.tk_localcta_reset_global_scale_num()
        _restore_env(_CONTRACT_ENV, previous_contract)


def _assert_reconstruct_scale_geometries() -> None:
    """Use fully initialized tensors so this decoder gate cannot sample stale memory."""
    def make_quant_tensors(rows: int, cols: int):
        fp4 = torch.empty(
            (rows, cols // 2), device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        fp4.view(torch.uint8).fill_(0x77)  # Both packed E2M1 values are +6.
        sc = torch.ones(
            (rows // 128, cols // 64, 512),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        return fp4, sc

    def expected_outer(rows: int, cols: int, values: torch.Tensor):
        return (
            values.flatten()
            .repeat_interleave(256)
            .reshape(rows, 1)
            .expand(rows, cols)
            .mul(6.0)
            .to(torch.bfloat16)
        )

    def expected_grid(values: torch.Tensor):
        return (
            values.repeat_interleave(256, dim=0)
            .repeat_interleave(256, dim=1)
            .mul(6.0)
            .to(torch.bfloat16)
        )

    # Rectangular row layout: [M, K] = [512, 768].
    row_rows, row_cols = 512, 768
    row_fp4, row_sc = make_quant_tensors(row_rows, row_cols)
    row_outer = torch.tensor([[1.0], [2.0]], device="cuda")
    decoded_row_outer = tkq.tk_localcta_reconstruct_row(
        row_fp4, row_sc, row_outer
    )
    assert torch.equal(
        decoded_row_outer, expected_outer(row_rows, row_cols, row_outer)
    )
    row_grid = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda"
    )
    decoded_row_grid = tkq.tk_localcta_reconstruct_row(row_fp4, row_sc, row_grid)
    assert torch.equal(decoded_row_grid, expected_grid(row_grid))

    # Public col wrapper sees the transposed [K, M] = [768, 512] layout and
    # its OuterScale vector has the opposite rank-2 orientation [1, K / 256].
    col_rows, col_cols = row_cols, row_rows
    col_fp4, col_sc = make_quant_tensors(col_rows, col_cols)
    col_outer = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    decoded_col_outer = tkq.tk_localcta_reconstruct_col(
        col_fp4, col_sc, col_outer
    )
    assert torch.equal(
        decoded_col_outer, expected_outer(col_rows, col_cols, col_outer)
    )
    col_grid = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda"
    )
    decoded_col_grid = tkq.tk_localcta_reconstruct_col(col_fp4, col_sc, col_grid)
    assert torch.equal(decoded_col_grid, expected_grid(col_grid))

    malformed_sg = torch.ones((3, 3), device="cuda", dtype=torch.float32)
    try:
        tkq.tk_localcta_reconstruct_row(row_fp4, row_sc, malformed_sg)
    except RuntimeError as error:
        assert "unsupported SG shape" in str(error), error
    else:
        raise AssertionError("malformed SG geometry was not rejected")


def main() -> None:
    torch.manual_seed(7)
    _assert_split2_advances()
    _assert_split3_axes_advance()
    _assert_inverse_rope_split3_advances()
    _assert_reconstruct_scale_geometries()
    _assert_tiny_nonzero_tiles_keep_relative_scale()
    print(
        "localCTA fused gradient SR: "
        "split2=PASS split3=PASS inverse_rope=PASS graph=PASS axes=PASS "
        "reconstruct_geometry=PASS tiny_tiles=PASS"
    )


if __name__ == "__main__":
    main()
