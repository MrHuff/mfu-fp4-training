"""
Test persistent quantize→GEMM kernel.
Compares against separate quantize+GEMM pipeline.
"""
import sys
import torch
torch.random.manual_seed(42)

from _C import nvfp4_gemm, nvfp4_quantize, nvfp4_persistent_gemm  # type: ignore


def bench(fn, label, num_warmups=10, num_iters=20):
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
    print(f"{'='*72}")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    ref = torch.matmul(A, B.T).to(torch.bfloat16)

    # --- Separate baseline ---
    A_fp4 = torch.empty(M, K//2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sg = torch.empty(1, dtype=torch.float32, device="cuda")
    B_fp4 = torch.empty(N, K//2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sg = torch.empty(1, dtype=torch.float32, device="cuda")
    D_sep = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    nvfp4_quantize(A, A_fp4, A_sc, A_sg, False)
    nvfp4_quantize(B, B_fp4, B_sc, B_sg, False)
    nvfp4_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D_sep)
    torch.cuda.synchronize()

    cos_sep = torch.nn.functional.cosine_similarity(
        D_sep.flatten().float().unsqueeze(0),
        ref.flatten().float().unsqueeze(0)).item()
    print(f"\n  [Separate] vs bf16 ref: cosine={cos_sep:.10f}")

    # --- Persistent kernel ---
    D_pers = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_persistent_gemm(A, B, D_pers)
    torch.cuda.synchronize()

    cos_pers_ref = torch.nn.functional.cosine_similarity(
        D_pers.flatten().float().unsqueeze(0),
        ref.flatten().float().unsqueeze(0)).item()
    cos_pers_sep = torch.nn.functional.cosine_similarity(
        D_pers.flatten().float().unsqueeze(0),
        D_sep.flatten().float().unsqueeze(0)).item()
    print(f"  [Persistent] vs bf16 ref: cosine={cos_pers_ref:.10f}")
    print(f"  [Persistent] vs separate: cosine={cos_pers_sep:.10f}")
    print(f"  [Persistent] max diff vs separate: {(D_pers.float() - D_sep.float()).abs().max().item():.6f}")

    finite = D_pers.isfinite().sum().item()
    total = D_pers.numel()
    print(f"  [Persistent] finite: {finite}/{total}")

    # --- Benchmarks ---
    print(f"\n  Benchmarks:")
    ms_q = bench(lambda: [nvfp4_quantize(A, A_fp4, A_sc, A_sg, False),
                           nvfp4_quantize(B, B_fp4, B_sc, B_sg, False)],
                 "Quantize A+B")
    ms_g = bench(lambda: nvfp4_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D_sep),
                 "GEMM only")
    ms_sep = bench(lambda: [nvfp4_quantize(A, A_fp4, A_sc, A_sg, False),
                             nvfp4_quantize(B, B_fp4, B_sc, B_sg, False),
                             nvfp4_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D_sep)],
                   "Separate")
    ms_pers = bench(lambda: nvfp4_persistent_gemm(A, B, D_pers), "Persistent")

    print(f"    Quantize A+B:   {ms_q:.4f} ms")
    print(f"    GEMM only:      {ms_g:.4f} ms")
    print(f"    Separate total: {ms_sep:.4f} ms")
    print(f"    Persistent:     {ms_pers:.4f} ms")
    if ms_pers < ms_sep:
        print(f"    \u2192 Persistent WINS by {ms_sep - ms_pers:.4f} ms ({(ms_sep-ms_pers)/ms_sep*100:.1f}%)")
    else:
        print(f"    \u2192 Separate WINS by {ms_pers - ms_sep:.4f} ms ({(ms_pers-ms_sep)/ms_pers*100:.1f}%)")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")

    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        for size in (256, 512, 1024, 2048, 4096):
            run_case(size, size, size)
    else:
        M = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
        run_case(M, N, K)
