"""Quick test: compare constant-scale quant vs dynamic-amax quant."""
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_v5 as tkq

torch.manual_seed(42)

def test_correctness(M, K, return_transpose=True):
    """Compare constant-scale vs dynamic-amax output."""
    x = torch.tanh(torch.randn(M, K, dtype=torch.bfloat16, device='cuda'))
    
    # Dynamic amax (original)
    r_dyn = tkq.tk_quantize_for_gemm(x, return_transpose)
    row_fp4_dyn, row_sc_dyn, col_fp4_dyn, col_sc_dyn, sg_dyn, _ = r_dyn
    
    # Constant scale (new)
    r_cs = tkq.tk_quantize_for_gemm_constant_scale(x, return_transpose)
    row_fp4_cs, row_sc_cs, col_fp4_cs, col_sc_cs, sg_cs, _ = r_cs
    
    # Check shapes match
    assert row_fp4_dyn.shape == row_fp4_cs.shape, f"row fp4 shape mismatch: {row_fp4_dyn.shape} vs {row_fp4_cs.shape}"
    assert row_sc_dyn.shape == row_sc_cs.shape, f"row sc shape mismatch: {row_sc_dyn.shape} vs {row_sc_cs.shape}"
    if return_transpose:
        assert col_fp4_dyn.shape == col_fp4_cs.shape, f"col fp4 shape mismatch"
        assert col_sc_dyn.shape == col_sc_cs.shape, f"col sc shape mismatch"
    
    # Unit-bounded inputs use fixed amax=1 and sg=1/2688.
    sg_val = sg_cs.item()
    expected_sg = 1.0 / 2688.0
    assert abs(sg_val - expected_sg) < 1e-7, f"sg should be {expected_sg}, got {sg_val}"
    
    # FP4 values may differ (different scaling) — check they're non-zero
    row_bytes_cs = row_fp4_cs.view(torch.uint8)
    nz = (row_bytes_cs != 0).sum().item()
    total = row_bytes_cs.numel()
    nz_pct = nz / total * 100
    
    print(f"  M={M:5d} K={K:4d}: sg_dyn={sg_dyn.item():.4f} sg_cs={sg_val:.4f} "
          f"nonzero={nz_pct:.1f}% ✓")

def test_perf(M, K, return_transpose=True, n_warmup=20, n_iters=100):
    """Benchmark constant-scale vs dynamic-amax."""
    x = torch.tanh(torch.randn(M, K, dtype=torch.bfloat16, device='cuda'))
    
    # Warmup dynamic
    for _ in range(n_warmup):
        tkq.tk_quantize_for_gemm(x, return_transpose)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        tkq.tk_quantize_for_gemm(x, return_transpose)
    torch.cuda.synchronize()
    t_dyn = (time.perf_counter() - t0) / n_iters * 1000  # ms
    
    # Warmup constant
    for _ in range(n_warmup):
        tkq.tk_quantize_for_gemm_constant_scale(x, return_transpose)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        tkq.tk_quantize_for_gemm_constant_scale(x, return_transpose)
    torch.cuda.synchronize()
    t_cs = (time.perf_counter() - t0) / n_iters * 1000  # ms
    
    speedup = t_dyn / t_cs if t_cs > 0 else float('inf')
    saved = t_dyn - t_cs
    print(f"  M={M:5d} K={K:4d}: dyn={t_dyn:.3f}ms  const={t_cs:.3f}ms  "
          f"saved={saved:.3f}ms  speedup={speedup:.2f}x")

print("=== Correctness Tests ===")
for M in [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    test_correctness(M, 2048)

print("\n=== Performance Tests (return_transpose=True) ===")
for M in [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    test_perf(M, 2048)

print("\n=== Performance Tests (return_transpose=False — row only) ===")
for M in [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    test_perf(M, 2048, return_transpose=False)

print("\nDone!")
