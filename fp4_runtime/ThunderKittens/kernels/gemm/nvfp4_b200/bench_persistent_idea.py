"""
Benchmark to test whether a persistent quantize→GEMM kernel could win.

Tests three things:
1. Separate quantize + GEMM (current baseline)
2. CUDA graph of quantize + GEMM (captures launch gap savings)
3. Quantize-only and GEMM-only (to see overlap potential)

If (2) is much faster than (1), there's a real launch-gap to capture with a
persistent kernel. If they're similar, PDL is already hiding the gap.
"""
import sys
import torch
torch.random.manual_seed(42)

from _C import nvfp4_gemm, nvfp4_quantize  # type: ignore


def bench(fn, label, num_warmups=10, num_iters=20):
    """Benchmark a function and return ms per call."""
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
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    # Pre-quantize B (weights) — used by all modes
    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    # Allocate A quantize outputs
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")

    # Allocate output
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # --- Benchmark 1: Separate quantize + GEMM ---
    def separate():
        nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
        nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D)

    ms_separate = bench(separate, "Separate quantize+GEMM")

    # --- Benchmark 2: Quantize only ---
    def quant_only():
        nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)

    ms_quant = bench(quant_only, "Quantize only")

    # --- Benchmark 3: GEMM only ---
    # First quantize so we have valid inputs
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
    torch.cuda.synchronize()

    def gemm_only():
        nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D)

    ms_gemm = bench(gemm_only, "GEMM only")

    # --- Benchmark 4: CUDA graph of quantize + GEMM ---
    ms_graph = None
    try:
        # Warmup for graph capture
        for _ in range(3):
            separate()
        torch.cuda.synchronize()

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
            nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D)

        def graph_replay():
            g.replay()

        ms_graph = bench(graph_replay, "CUDA graph quantize+GEMM")
    except Exception as e:
        print(f"  CUDA graph capture failed: {e}")
        ms_graph = None

    # --- Results ---
    print(f"\n  Results:")
    print(f"    Quantize only:         {ms_quant:.4f} ms")
    print(f"    GEMM only:             {ms_gemm:.4f} ms")
    print(f"    Sum (quant+gemm):      {ms_quant + ms_gemm:.4f} ms")
    print(f"    max(quant, gemm):      {max(ms_quant, ms_gemm):.4f} ms")
    print(f"    Separate (measured):   {ms_separate:.4f} ms")
    if ms_graph is not None:
        print(f"    CUDA graph (measured): {ms_graph:.4f} ms")
        gap = ms_separate - ms_graph
        print(f"    Launch gap captured:   {gap:.4f} ms ({gap/ms_separate*100:.1f}%)")
    else:
        print(f"    CUDA graph:            FAILED")

    print(f"\n  Analysis:")
    measured_overlap = (ms_quant + ms_gemm) - ms_separate
    print(f"    PDL overlap:           {measured_overlap:.4f} ms ({measured_overlap/(ms_quant+ms_gemm)*100:.1f}% of sum)")
    theoretical_min = max(ms_quant, ms_gemm)
    print(f"    Theoretical min:       {theoretical_min:.4f} ms")
    headroom = ms_separate - theoretical_min
    print(f"    Headroom (sep - min):  {headroom:.4f} ms ({headroom/ms_separate*100:.1f}%)")
    if ms_graph is not None:
        graph_headroom = ms_graph - theoretical_min
        print(f"    Headroom (graph-min):  {graph_headroom:.4f} ms ({graph_headroom/ms_graph*100:.1f}%)")


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
