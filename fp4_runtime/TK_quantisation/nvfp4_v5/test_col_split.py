"""Benchmark: row+col vs row-only + col-only split (with constant scale)."""
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_v5 as tkq

torch.manual_seed(42)

def bench(fn, n_warmup=20, n_iters=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000

# Unit-bound sg for col-only.
sg_const = torch.full((1,), 1.0 / 2688.0, dtype=torch.float32, device='cuda')

print("=== Single-call timings ===")
print(f"{'M':>6} {'K':>5}  {'row+col_dyn':>12} {'row+col_cs':>11} {'row_only_cs':>12} {'col_only':>9} {'row+col_split':>14} {'split_save':>11}")

for M in [2048, 4096, 8192, 16384, 32768, 65536]:
    K = 2048
    x = torch.tanh(torch.randn(M, K, dtype=torch.bfloat16, device='cuda'))

    # A: dynamic amax, row+col
    t_dyn = bench(lambda: tkq.tk_quantize_for_gemm(x, True))

    # B: constant scale, row+col
    t_cs_both = bench(lambda: tkq.tk_quantize_for_gemm_constant_scale(x, True))

    # C: constant scale, row only
    t_row = bench(lambda: tkq.tk_quantize_for_gemm_constant_scale(x, False))

    # D: col-only (using the matching unit-bound global scale)
    t_col = bench(lambda: tkq.tk_quantize_col_only(x, sg_const))

    # Split total = row-only + col-only
    t_split = t_row + t_col
    save = t_cs_both - t_split

    print(f"{M:6d} {K:5d}  {t_dyn:12.3f} {t_cs_both:11.3f} {t_row:12.3f} {t_col:9.3f} {t_split:14.3f} {save:+11.3f}")

print("\n=== 3× calls (QKV backward simulation) ===")
print(f"{'M':>6}  {'3×row+col_dyn':>14} {'3×row+col_cs':>13} {'3×row+3×col':>12} {'3×split_save':>13} {'vs_dyn_save':>12}")

for M in [2048, 4096, 8192, 16384, 32768, 65536]:
    K = 2048
    xs = [
        torch.tanh(torch.randn(M, K, dtype=torch.bfloat16, device='cuda'))
        for _ in range(3)
    ]

    # 3× dynamic row+col
    def dyn_3(): 
        for x in xs: tkq.tk_quantize_for_gemm(x, True)
    t3_dyn = bench(dyn_3)

    # 3× constant scale row+col
    def cs_both_3():
        for x in xs: tkq.tk_quantize_for_gemm_constant_scale(x, True)
    t3_cs = bench(cs_both_3)

    # 3× row-only + 3× col-only (split)
    def split_3():
        for x in xs: tkq.tk_quantize_for_gemm_constant_scale(x, False)
        for x in xs: tkq.tk_quantize_col_only(x, sg_const)
    t3_split = bench(split_3)

    save_vs_cs = t3_cs - t3_split
    save_vs_dyn = t3_dyn - t3_split

    print(f"{M:6d}  {t3_dyn:14.3f} {t3_cs:13.3f} {t3_split:12.3f} {save_vs_cs:+13.3f} {save_vs_dyn:+12.3f}")

print("\n=== Critical path analysis (row-only → GEMM can start, col runs after) ===")
print("Row-only latency = dgrad critical path contribution")
print("Col-only runs after GEMM (not on critical path if GEMM is slower)")
print()
for M in [2048, 4096, 8192, 16384, 32768, 65536]:
    K = 2048
    x = torch.tanh(torch.randn(M, K, dtype=torch.bfloat16, device='cuda'))
    t_row = bench(lambda: tkq.tk_quantize_for_gemm_constant_scale(x, False))
    t_both = bench(lambda: tkq.tk_quantize_for_gemm_constant_scale(x, True))
    crit_save = t_both - t_row
    print(f"  M={M:5d}: row_only={t_row:.3f}ms  row+col={t_both:.3f}ms  "
          f"critical_path_saved={crit_save:.3f}ms  (×3 = {3*crit_save:.3f}ms)")

print("\nDone!")
