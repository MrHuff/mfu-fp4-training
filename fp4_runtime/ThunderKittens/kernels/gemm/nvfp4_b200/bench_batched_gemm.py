#!/usr/bin/env python3
"""
Fair benchmark: batched GEMM vs CUDA-graphed sequential loop.

The sequential loop is captured into a CUDA graph so that CPU-side
launch overhead is eliminated — making this a true GPU-vs-GPU comparison.
"""
import sys, torch
sys.path.insert(0, '.')
from _C import nvfp4_quantize, nvfp4_gemm, nvfp4_batched_gemm

torch.random.manual_seed(42)

def quantize(A):
    M, Kh = A.shape; K = Kh * 2
    fp4 = torch.empty(M, Kh, dtype=torch.float4_e2m1fn_x2, device='cuda')
    sc  = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device='cuda')
    sg  = torch.empty(1, dtype=torch.float32, device='cuda')
    nvfp4_quantize(A, fp4, sc, sg, False)
    return fp4, sc, sg

def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms

def make_cuda_graph(fn, warmup=3):
    """Capture fn into a CUDA graph and return a replay callable."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    # Warmup in side stream
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    def replay():
        g.replay()
    return replay

SHAPES = [
    (16384, 2048, 2048),
    (16384, 4096, 4096),
    (16384, 8192, 8192),
]

NUM_BATCHES = [1, 2, 3, 4]

print("=" * 140)
print(f"{'Shape':>22s} | {'Method':>18s} | {'Time(ms)':>10s} | {'TFLOPs':>8s} | {'vs_graph':>10s} | {'vs_eager':>10s} | {'Note':>25s}")
print("=" * 140)

for M, K, N in SHAPES:
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / (K ** 0.5)
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / (K ** 0.5)
    Aq = quantize(A); Bq = quantize(B)
    flops_per_gemm = 2.0 * M * N * K

    # ── 1. Standard GEMM (single, eager) ──
    D_std = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    def run_std():
        nvfp4_gemm(Aq[0], Aq[1], Aq[2], Bq[0], Bq[1], Bq[2], D_std)
    std_ms = bench(run_std)
    std_tflops = flops_per_gemm / (std_ms * 1e-3) / 1e12
    print(f"{M}x{K}x{N:>5d} | {'1x std (eager)':>18s} | {std_ms:>10.4f} | {std_tflops:>8.1f} | {'':>10s} | {'':>10s} | {'reference':>25s}")

    # ── 2. Standard GEMM (single, CUDA graphed) ──
    std_graph = make_cuda_graph(run_std)
    std_graph_ms = bench(std_graph)
    std_graph_tflops = flops_per_gemm / (std_graph_ms * 1e-3) / 1e12
    print(f"{' ':>22s} | {'1x std (graphed)':>18s} | {std_graph_ms:>10.4f} | {std_graph_tflops:>8.1f} | {'':>10s} | {'':>10s} | {'graph baseline':>25s}")

    for nb in NUM_BATCHES:
        try:
            # Pre-allocate everything
            A_fp4_list = [Aq[0]] * nb
            A_sc_list  = [Aq[1]] * nb
            A_sg_list  = [Aq[2]] * nb
            B_fp4_list = [Bq[0]] * nb
            B_sc_list  = [Bq[1]] * nb
            B_sg_list  = [Bq[2]] * nb
            D_list = [torch.zeros(M, N, dtype=torch.bfloat16, device='cuda') for _ in range(nb)]
            D_loop = [torch.zeros(M, N, dtype=torch.bfloat16, device='cuda') for _ in range(nb)]

            total_flops = flops_per_gemm * nb

            # ── 3. Batched GEMM (single kernel launch) ──
            def run_batched():
                nvfp4_batched_gemm(A_fp4_list, A_sc_list, A_sg_list,
                                   B_fp4_list, B_sc_list, B_sg_list, D_list)
            b_ms = bench(run_batched)
            b_tflops = total_flops / (b_ms * 1e-3) / 1e12

            # ── 4. Sequential loop (eager, N launches from Python) ──
            def run_loop_eager():
                for d in D_loop:
                    nvfp4_gemm(Aq[0], Aq[1], Aq[2], Bq[0], Bq[1], Bq[2], d)
            loop_eager_ms = bench(run_loop_eager)

            # ── 5. Sequential loop (CUDA graphed, eliminates CPU launch overhead) ──
            loop_graph = make_cuda_graph(run_loop_eager)
            loop_graph_ms = bench(loop_graph)

            vs_graph = loop_graph_ms / b_ms
            vs_eager = loop_eager_ms / b_ms

            print(f"{' ':>22s} | {'batched_x' + str(nb):>18s} | {b_ms:>10.4f} | {b_tflops:>8.1f} | {vs_graph:>9.2f}x | {vs_eager:>9.2f}x | {'':>25s}")
            print(f"{' ':>22s} | {str(nb) + 'x loop(eager)':>18s} | {loop_eager_ms:>10.4f} | {'':>8s} | {'':>10s} | {'':>10s} | {'':>25s}")
            print(f"{' ':>22s} | {str(nb) + 'x loop(graph)':>18s} | {loop_graph_ms:>10.4f} | {'':>8s} | {'':>10s} | {'':>10s} | {'':>25s}")
        except Exception as e:
            print(f"{' ':>22s} | {'batched_x' + str(nb):>18s} | {'ERR':>10s} | {'':>8s} | {'':>10s} | {'':>10s} | {str(e)[:25]:>25s}")
            torch.cuda.synchronize()

    print("-" * 140)

print("DONE")
