"""
Test script for NVFP4 Fused Quantize+GEMM kernel.

Compares 3 modes:
1. Separate quantize→GEMM pipeline (global amax, 2-pass)
2. Fused GEMM with constant SCALE_MAX (no pre-scan)
3. Fused GEMM with CTA-level amax (pre-scan)
All compared against PyTorch bf16 matmul reference.

Usage:
  python test_fused_gemm.py [M] [N] [K]
  python test_fused_gemm.py --sweep
"""
import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import (nvfp4_fused_gemm, nvfp4_fused_gemm_cta_amax,
                nvfp4_gemm, nvfp4_quantize)  # type: ignore


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor) -> None:
    A = A.to(torch.float32)
    A_ref = A_ref.to(torch.float32)
    print(f"  [DEBUG] A has NaN? {torch.isnan(A).any().item()}, A_ref has NaN? {torch.isnan(A_ref).any().item()}")
    max_diff = (A - A_ref).abs().max().item()
    mean_diff = (A - A_ref).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        A.flatten().unsqueeze(0), A_ref.flatten().unsqueeze(0)
    ).item()
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Cos sim:   {cos_sim:.10f}")


def fused_backend_label(N: int) -> str:
    if N % 256 == 0 and N <= 2048:
        return "dual-column reuse backend"
    if N % 256 == 0:
        return "single-column wide-tile backend"
    return "single-column fallback"


def run_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  M={M}, N={N}, K={K}")
    print(f"  Fused backend: {fused_backend_label(N)}")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    # ── Reference: PyTorch bf16 matmul ──
    D_ref = torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16)

    # ── Pre-quantize B (weights) — used by all modes ──
    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    # ── Mode 0: separate quantize → GEMM (baseline) ──
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
    D_separate = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()
    print(f"\n[0] Separate quantize→GEMM vs bf16 ref:")
    check_diff("sep vs ref", D_separate, D_ref)

    # ── Mode 1: fused, constant SCALE_MAX ──
    D_fused_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    print(f"  [DEBUG] D_fused_const has NaN BEFORE? {torch.isnan(D_fused_const).any().item()}")
    nvfp4_fused_gemm(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_const)
    torch.cuda.synchronize()
    print(f"  [DEBUG] D_fused_const has NaN AFTER? {torch.isnan(D_fused_const).any().item()}")
    print(f"\n[1] Fused (constant SCALE_MAX) vs bf16 ref:")
    check_diff("fused_const vs ref", D_fused_const, D_ref)
    print(f"\n[1] Fused (constant SCALE_MAX) vs separate:")
    check_diff("fused_const vs sep", D_fused_const, D_separate)

    # ── Mode 2: fused, CTA-level amax ──
    D_fused_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm_cta_amax(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_cta)
    torch.cuda.synchronize()
    print(f"\n[2] Fused (CTA amax) vs bf16 ref:")
    check_diff("fused_cta vs ref", D_fused_cta, D_ref)
    print(f"\n[2] Fused (CTA amax) vs separate:")
    check_diff("fused_cta vs sep", D_fused_cta, D_separate)

    # ── Benchmark ──
    NUM_WARMUPS = 5
    NUM_ITERS = 10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def bench(fn, label):
        for _ in range(NUM_WARMUPS): fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(NUM_ITERS): fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / NUM_ITERS
        flops = 2.0 * M * N * K
        tflops = flops * 1e-12
        print(f"  {label}: {ms:.3f} ms  ({tflops/ms*1e3:.2f} TFLOPs)")

    print(f"\n{'='*72}")
    print("Benchmarks:")
    bench(lambda: nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False),
          "Separate quantize only  ")
    bench(lambda: nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate),
          "Separate GEMM only      ")
    bench(lambda: [nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False),
                   nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)],
          "Separate (quantize+GEMM)  ")
    bench(lambda: nvfp4_fused_gemm(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_const),
          "Fused (constant SCALE_MAX)")
    bench(lambda: nvfp4_fused_gemm_cta_amax(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_cta),
          "Fused (CTA amax)          ")
    print(f"{'='*72}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device unavailable. Run test_fused_gemm.py on a GPU-enabled B200 host."
        )

    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        for size in (256, 1024, 2048, 4096):
            run_case(size, size, size)
            print()
    else:
        M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
        run_case(M, N, K)
