"""
Persistent quantize→GEMM benchmark at realistic QKV fwd/bwd sizes.
Matches bench_qkv.py: dim=2048, N_total=6144, M=2048..65536
"""
import sys
import torch
torch.random.manual_seed(42)

from _C import nvfp4_gemm, nvfp4_quantize, nvfp4_persistent_gemm  # type: ignore


def bench(fn, num_warmups=10, num_iters=20):
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / num_iters


def run_case(M, N, K):
    print(f"\n{'='*72}")
    print(f"  M={M}, N={N}, K={K}")
    fp4_bytes = (M*K + N*K) // 2
    print(f"  Quantized data: {fp4_bytes/1024:.0f} KB")
    print(f"{'='*72}")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    # Separate baseline
    A_fp4 = torch.empty(M, K//2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sg = torch.empty(1, dtype=torch.float32, device="cuda")
    B_fp4 = torch.empty(N, K//2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sg = torch.empty(1, dtype=torch.float32, device="cuda")
    D_sep = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_pers = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Benchmark quantize only
    ms_q = bench(lambda: [nvfp4_quantize(A, A_fp4, A_sc, A_sg, False),
                           nvfp4_quantize(B, B_fp4, B_sc, B_sg, False)])

    # Benchmark GEMM only
    nvfp4_quantize(A, A_fp4, A_sc, A_sg, False)
    nvfp4_quantize(B, B_fp4, B_sc, B_sg, False)
    torch.cuda.synchronize()
    ms_g = bench(lambda: nvfp4_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D_sep))

    # Benchmark separate pipeline
    ms_sep = bench(lambda: [nvfp4_quantize(A, A_fp4, A_sc, A_sg, False),
                             nvfp4_quantize(B, B_fp4, B_sc, B_sg, False),
                             nvfp4_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D_sep)])

    # Benchmark persistent
    ms_pers = bench(lambda: nvfp4_persistent_gemm(A, B, D_pers))

    # Correctness
    nvfp4_quantize(A, A_fp4, A_sc, A_sg, False)
    nvfp4_quantize(B, B_fp4, B_sc, B_sg, False)
    nvfp4_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D_sep)
    nvfp4_persistent_gemm(A, B, D_pers)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        D_pers.flatten().float().unsqueeze(0),
        D_sep.flatten().float().unsqueeze(0)).item()

    flops = 2.0 * M * N * K
    sep_tflops = flops / (ms_sep * 1e-3) / 1e12
    pers_tflops = flops / (ms_pers * 1e-3) / 1e12

    print(f"    Quant A+B:    {ms_q:.4f} ms")
    print(f"    GEMM only:    {ms_g:.4f} ms")
    print(f"    Separate:     {ms_sep:.4f} ms  ({sep_tflops:.0f} TFLOPs)")
    print(f"    Persistent:   {ms_pers:.4f} ms  ({pers_tflops:.0f} TFLOPs)")
    delta = ms_sep - ms_pers
    if delta > 0:
        print(f"    → Persistent WINS by {delta:.4f} ms ({delta/ms_sep*100:.1f}%)")
    else:
        print(f"    → Separate WINS by {-delta:.4f} ms ({-delta/ms_pers*100:.1f}%)")
    print(f"    Cosine vs sep: {cos:.8f}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    # QKV forward sizes: M×6144×2048
    dim = 2048
    N_total = 6144  # q+k+v
    print("\n*** QKV Forward shapes (M × 6144 × 2048) ***")
    for M in [2048, 4096, 8192, 16384, 32768, 65536]:
        try:
            run_case(M, N_total, dim)
        except Exception as e:
            print(f"  M={M}: FAILED: {e}")

    # QKV dgrad shapes: M×2048×6144
    print("\n\n*** QKV Dgrad shapes (M × 2048 × 6144) ***")
    for M in [2048, 4096, 8192, 16384, 32768]:
        try:
            run_case(M, dim, N_total)
        except Exception as e:
            print(f"  M={M}: FAILED: {e}")

    # Square shapes for reference
    print("\n\n*** Square shapes ***")
    for s in [4096, 8192]:
        try:
            run_case(s, s, s)
        except Exception as e:
            print(f"  {s}³: FAILED: {e}")
