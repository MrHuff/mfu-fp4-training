#!/usr/bin/env python3
import argparse
import os

import torch
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


def time_ms(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def bench_shape(
    rows: int,
    cols: int,
    iters: int,
    warmup: int,
    random_sign: bool,
    rowwise: bool,
    columnwise: bool,
) -> None:
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    x = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    q = NVFP4Quantizer(
        rowwise=rowwise,
        columnwise=columnwise,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=False,
        with_random_sign_mask=random_sign,
    )
    out = q.make_empty(x.shape, dtype=x.dtype, device=x.device)

    for _ in range(warmup):
        q.update_quantized(x, out)
    torch.cuda.synchronize()

    ms = time_ms(lambda: q.update_quantized(x, out), iters)
    row_checksum = -1
    col_checksum = -1
    if out._rowwise_data is not None:
        row_checksum = int(out._rowwise_data[: min(rows, 4), : min(cols // 2, 64)].sum().item())
    if out._columnwise_data is not None:
        col_checksum = int(out._columnwise_data[: min(cols, 4), : min(rows // 2, 64)].sum().item())
    row_shape = None if out._rowwise_data is None else tuple(out._rowwise_data.shape)
    col_shape = None if out._columnwise_data is None else tuple(out._columnwise_data.shape)
    row_scale_shape = None if out._rowwise_scale_inv is None else tuple(out._rowwise_scale_inv.shape)
    col_scale_shape = None if out._columnwise_scale_inv is None else tuple(out._columnwise_scale_inv.shape)
    mode = ("row" if rowwise else "") + ("col" if columnwise else "")
    print(f"==== TE native NVFP4 RHT {mode} shape {rows} x {cols} ====")
    print(f"rowwise_data={row_shape} columnwise_data={col_shape}")
    print(f"rowwise_scale={row_scale_shape} columnwise_scale={col_scale_shape}")
    print(f"checksum_te_native_row={row_checksum} col={col_checksum}")
    print(f"te_native_nvfp4_rht_{mode}_update_quantized: {ms:.6f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--iters", type=int, default=int(os.getenv("TE_RHT_ITERS", "160")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("TE_RHT_WARMUP", "20")))
    parser.add_argument("--no-random-sign", action="store_true")
    parser.add_argument("--rowwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--columnwise", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    shapes = args.shape or ["2048x65536", "3072x65536", "8192x2048"]
    for shape in shapes:
        rows_s, cols_s = shape.lower().replace(",", "x").split("x")
        bench_shape(
            int(rows_s),
            int(cols_s),
            args.iters,
            args.warmup,
            not args.no_random_sign,
            args.rowwise,
            args.columnwise,
        )


if __name__ == "__main__":
    main()
