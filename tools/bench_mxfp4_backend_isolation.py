#!/usr/bin/env python3
"""Benchmark the existing MXFP4 backend surfaces before fused routing."""

from __future__ import annotations

import argparse
import json
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCHTITAN_ROOT = os.path.join(REPO_ROOT, "torchtitan_submodule")
FALLBACK_TORCHTITAN_ROOT = "/opt/mfu/EXTERNAL_PATH"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, TORCHTITAN_ROOT)
if os.path.isdir(FALLBACK_TORCHTITAN_ROOT) and FALLBACK_TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, FALLBACK_TORCHTITAN_ROOT)


PROBLEMS = {
    "qkv": {"M": 64 * 1024, "K": 2048, "N": 3 * 2048},
    "ffn": {"M": 64 * 1024, "K": 2048, "N": 5632},
}


def configure_env(backend_version: str):
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("NVTE_NVFP4_DISABLE_RHT", "1")
    os.environ.setdefault("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "1")
    os.environ.setdefault("NVTE_NVFP4_ENCODE_CENTRIC", "0")
    os.environ.setdefault("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "1")
    os.environ["MXFP4_BACKEND_VERSION"] = backend_version


def bench(fn, warmup=5, iters=20):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cosine(a, b):
    import torch

    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def run_problem(problem: str, warmup: int, iters: int, device_index: int, backend_version: str):
    import torch
    import low_bits_training  # noqa: F401
    from low_bits_training.quantization.mxfp4_backend import (
        mxfp4_backend_capabilities,
        mxfp4_gemm,
        mxfp4_group_quantize_dim0,
        mxfp4_quantize_for_gemm,
        mxfp4_quantize_row_and_col,
    )
    from low_bits_training.quantization.mxfp_custom_te_fp4 import (
        BoundRecipeLinear,
        MXFP4BlockScaling,
    )

    cfg = PROBLEMS[problem]
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"

    caps = mxfp4_backend_capabilities()
    if not caps["backend_available"]:
        raise RuntimeError(f"MXFP4 backend unavailable: {caps}")

    torch.manual_seed(1234)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K ** 0.25)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) / (K ** 0.25)
    stacked = torch.cat([x, w], dim=0).contiguous()

    x_fp4, x_sc = mxfp4_quantize_for_gemm(x, 1)
    w_fp4, w_sc = mxfp4_quantize_for_gemm(w, 1)
    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    quant_single_ms = bench(lambda: mxfp4_quantize_for_gemm(x, 1), warmup=warmup, iters=iters)
    quant_row_col_ms = bench(lambda: mxfp4_quantize_row_and_col(x, 1), warmup=warmup, iters=iters)
    quant_group_ms = bench(
        lambda: mxfp4_group_quantize_dim0(stacked, [M, N]), warmup=warmup, iters=iters
    )
    gemm_ms = bench(lambda: mxfp4_gemm(x_fp4, x_sc, w_fp4, w_sc, out), warmup=warmup, iters=iters)

    def backend_chain():
        qx_fp4, qx_sc = mxfp4_quantize_for_gemm(x, 1)
        qw_fp4, qw_sc = mxfp4_quantize_for_gemm(w, 1)
        tmp = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        mxfp4_gemm(qx_fp4, qx_sc, qw_fp4, qw_sc, tmp)
        return tmp

    backend_chain_ms = bench(backend_chain, warmup=warmup, iters=iters)
    backend_out = backend_chain()

    bf16_out = torch.matmul(x.float(), w.float().T).to(torch.bfloat16)

    recipe = MXFP4BlockScaling(encode=True)
    te_linear = BoundRecipeLinear(
        in_features=K,
        out_features=N,
        bias=False,
        params_dtype=torch.bfloat16,
        recipe=recipe,
        device=device,
    )
    with torch.no_grad():
        te_linear.weight.copy_(w)

    te_chain_ms = bench(lambda: te_linear(x), warmup=warmup, iters=iters)
    te_out = te_linear(x)

    return {
        "problem": problem,
        "device_index": device_index,
        "backend_version": backend_version,
        "M": M,
        "K": K,
        "N": N,
        "backend_quant_single_ms": quant_single_ms,
        "backend_quant_row_col_ms": quant_row_col_ms,
        "backend_quant_group_ms": quant_group_ms,
        "backend_gemm_ms": gemm_ms,
        "backend_chain_ms": backend_chain_ms,
        "te_native_chain_ms": te_chain_ms,
        "backend_cosine_to_bf16": cosine(backend_out, bf16_out),
        "te_native_cosine_to_bf16": cosine(te_out, bf16_out),
        "backend_vs_te_cosine": cosine(backend_out, te_out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["qkv", "ffn", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--backend-version", choices=["v3", "v4"], default="v4")
    args = parser.parse_args()

    configure_env(args.backend_version)

    problems = [args.problem] if args.problem != "both" else ["qkv", "ffn"]
    results = [run_problem(problem, args.warmup, args.iters, args.device_index, args.backend_version) for problem in problems]
    print(json.dumps(results if len(results) > 1 else results[0], sort_keys=True))


if __name__ == "__main__":
    main()
