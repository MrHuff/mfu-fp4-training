"""
Deeper analysis: can we overlap quantize and GEMM with streams?
If so, that's cheaper than writing a persistent kernel.

Also tests: what if we just pre-quantize (cached in L2) and
replay GEMM immediately? Does L2 caching of quantized data help?
"""
import sys
import torch
torch.random.manual_seed(42)

from _C import nvfp4_gemm, nvfp4_quantize  # type: ignore


def bench(fn, label, num_warmups=10, num_iters=30):
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / num_iters
    return ms


def run_case(M, N, K):
    print(f"\n{'='*72}")
    print(f"  M={M}, N={N}, K={K}")
    fp4_bytes = M * K // 2 + N * K // 2
    scale_bytes = (M * K // 16 + N * K // 16)
    total_quant_bytes = fp4_bytes + scale_bytes
    print(f"  Total quantized data: {total_quant_bytes / 1024:.1f} KB")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    # Pre-quantize B
    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    # A quantize outputs
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # --- Baseline: separate ---
    def separate():
        nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
        nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D)

    ms_separate = bench(separate, "Separate")

    # --- Quantize only, GEMM only ---
    ms_quant = bench(lambda: nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False), "Quant")
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
    torch.cuda.synchronize()
    ms_gemm = bench(lambda: nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D), "GEMM")

    # --- GEMM immediately after quantize (L2 warm) ---
    # Run quantize right before each GEMM iteration to keep L2 warm
    def quant_then_gemm_l2_test():
        nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
        nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D)

    # Time just the GEMM after a quantize (to see if L2 warmth helps GEMM)
    # We'll do this by timing quant+gemm and subtracting quant
    ms_qg = bench(quant_then_gemm_l2_test, "Q+G")
    ms_gemm_after_quant = ms_qg - ms_quant  # rough estimate of GEMM with warm L2

    # --- CUDA graph ---
    ms_graph = None
    try:
        for _ in range(3):
            separate()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
            nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D)
        ms_graph = bench(lambda: g.replay(), "Graph")
    except Exception as e:
        print(f"  Graph failed: {e}")

    # --- Double-quantize B test: quantize both A and B, then GEMM ---
    # This is the "pre-quantize everything first" scenario
    A_fp4x2_2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc_2 = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global_2 = torch.empty(1, dtype=torch.float32, device="cuda")
    B_fp4x2_2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc_2 = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global_2 = torch.empty(1, dtype=torch.float32, device="cuda")
    D_2 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    def quant_both_then_gemm():
        nvfp4_quantize(A_bf16, A_fp4x2_2, A_sc_2, A_sc_global_2, False)
        nvfp4_quantize(B_bf16, B_fp4x2_2, B_sc_2, B_sc_global_2, False)
        nvfp4_gemm(A_fp4x2_2, A_sc_2, A_sc_global_2, B_fp4x2_2, B_sc_2, B_sc_global_2, D_2)

    ms_both = bench(quant_both_then_gemm, "Both quant+GEMM")

    # CUDA graph of quant-both + GEMM
    ms_both_graph = None
    try:
        for _ in range(3):
            quant_both_then_gemm()
        torch.cuda.synchronize()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            nvfp4_quantize(A_bf16, A_fp4x2_2, A_sc_2, A_sc_global_2, False)
            nvfp4_quantize(B_bf16, B_fp4x2_2, B_sc_2, B_sc_global_2, False)
            nvfp4_gemm(A_fp4x2_2, A_sc_2, A_sc_global_2, B_fp4x2_2, B_sc_2, B_sc_global_2, D_2)
        ms_both_graph = bench(lambda: g2.replay(), "Both+G graph")
    except Exception as e:
        print(f"  Both graph failed: {e}")

    # --- Results ---
    print(f"\n  Individual:")
    print(f"    Quantize A:            {ms_quant:.4f} ms")
    print(f"    GEMM only:             {ms_gemm:.4f} ms")
    print(f"    GEMM after quant (est):{ms_gemm_after_quant:.4f} ms")

    print(f"\n  Pipelines (A pre-quantized B):")
    print(f"    Separate (quantA+GEMM):{ms_separate:.4f} ms")
    if ms_graph:
        print(f"    Graph (quantA+GEMM):   {ms_graph:.4f} ms  (saves {ms_separate - ms_graph:.4f} ms)")

    print(f"\n  Pipelines (both bf16):")
    print(f"    Separate (both+GEMM):  {ms_both:.4f} ms")
    if ms_both_graph:
        print(f"    Graph (both+GEMM):     {ms_both_graph:.4f} ms  (saves {ms_both - ms_both_graph:.4f} ms)")

    print(f"\n  Key metric — persistent kernel headroom:")
    if ms_graph:
        theoretical = max(ms_quant, ms_gemm)
        print(f"    max(quant, gemm):      {theoretical:.4f} ms")
        print(f"    Graph measured:        {ms_graph:.4f} ms")
        print(f"    Persistent potential:  {ms_graph - theoretical:.4f} ms  ({(ms_graph - theoretical)/ms_graph*100:.1f}%)")
        print(f"    vs separate:           {ms_separate - theoretical:.4f} ms  ({(ms_separate - theoretical)/ms_separate*100:.1f}%)")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")

    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        for size in (256, 512, 1024, 2048, 4096):
            run_case(size, size, size)
    else:
        M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
        run_case(M, N, K)
