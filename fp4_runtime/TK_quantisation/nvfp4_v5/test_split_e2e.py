"""Benchmark: All strategies with batched_gemm+sum3 (not accum_gemm).

Focuses on the quant savings, keeping GEMM+sum3 the same.
"""
import torch
import sys
import time
from pathlib import Path

NVFP4_V5_DIR = Path(__file__).resolve().parent
REPO_ROOT = NVFP4_V5_DIR.parents[1]
sys.path.insert(0, str(NVFP4_V5_DIR))
import _tk_quant_v5 as tkq

import importlib.util
_gemm_so = str(
    REPO_ROOT
    / "ThunderKittens/kernels/gemm/nvfp4_b200/_C.cpython-312-aarch64-linux-gnu.so"
)
_old_c = sys.modules.pop('_C', None)
spec = importlib.util.spec_from_file_location('_C', _gemm_so)
tk_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tk_mod)
if _old_c is not None:
    sys.modules['_C'] = _old_c
elif '_C' in sys.modules:
    del sys.modules['_C']

torch.manual_seed(42)

def make_weight_splits(K_weight, N_dims, device='cuda'):
    B_fp4_list, B_sc_list, B_sg_list = [], [], []
    for n_i in N_dims:
        W_i = torch.randn(K_weight, n_i, dtype=torch.bfloat16, device=device)
        r = tkq.tk_quantize_for_gemm(W_i, True)
        B_fp4_list.append(r[2])
        B_sc_list.append(r[3])
        B_sg_list.append(r[4].to(torch.float32))
    return B_fp4_list, B_sc_list, B_sg_list

def bench(fn, n_warmup=20, n_iters=100):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000

def run_bench(M, K_weight, N_dims):
    n_groups = len(N_dims)
    device = 'cuda'
    grad_splits = [
        torch.tanh(torch.randn(M, n, dtype=torch.bfloat16, device=device))
        for n in N_dims
    ]
    B_fp4_list, B_sc_list, B_sg_list = make_weight_splits(K_weight, N_dims, device)
    sg_const = torch.full((1,), 1.0 / 2688.0, dtype=torch.float32, device=device)
    D_list = [torch.empty(M, K_weight, dtype=torch.bfloat16, device=device) for _ in range(n_groups)]
    D_sum = torch.empty(M, K_weight, dtype=torch.bfloat16, device=device)

    def do_gemm_sum(A_fp4, A_sc, A_sg):
        tk_mod.nvfp4_batched_gemm(A_fp4, A_sc, A_sg,
                                   B_fp4_list, B_sc_list, B_sg_list, D_list)
        tk_mod.sum3_bf16(D_list[0], D_list[1], D_list[2], D_sum)

    # (c) Baseline: dynamic amax row+col → gemm → sum3
    def strat_c():
        qr = [tkq.tk_quantize_for_gemm(g, True) for g in grad_splits]
        do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
    t_c = bench(strat_c)

    # (b) Constant-scale row+col → gemm → sum3
    def strat_b():
        qr = [tkq.tk_quantize_for_gemm_constant_scale(g, True) for g in grad_splits]
        do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
    t_b = bench(strat_b)

    # (a) Split: row-only → gemm → sum3 → col-only (total wall)
    def strat_a_total():
        rr = [tkq.tk_quantize_for_gemm_constant_scale(g, False) for g in grad_splits]
        do_gemm_sum([r[0] for r in rr], [r[1] for r in rr], [r[4] for r in rr])
        for g in grad_splits:
            tkq.tk_quantize_col_only(g, sg_const)
    t_a = bench(strat_a_total)

    # (a') Critical path: row-only → gemm → sum3 (no col)
    def strat_a_crit():
        rr = [tkq.tk_quantize_for_gemm_constant_scale(g, False) for g in grad_splits]
        do_gemm_sum([r[0] for r in rr], [r[1] for r in rr], [r[4] for r in rr])
    t_a_crit = bench(strat_a_crit)

    save_b_vs_c = t_c - t_b
    save_a_vs_c = t_c - t_a_crit
    
    print(f"  M={M:5d}: "
          f"(c)baseline={t_c:.3f}  "
          f"(b)const_sc={t_b:.3f} [{save_b_vs_c:+.3f}]  "
          f"(a)split_all={t_a:.3f}  "
          f"(a')crit={t_a_crit:.3f} [{save_a_vs_c:+.3f}]")

print("=== Full Pipeline: quant + batched_gemm + sum3 (ms) ===")
print("  (c)  = dynamic amax row+col [BASELINE]")
print("  (b)  = constant-scale row+col [Opp 1]")
print("  (a)  = split: row-only + gemm/sum3 + col-only [total wall]")
print("  (a') = split: row-only + gemm/sum3 [critical path to D_sum]\n")
for M in [2048, 4096, 8192, 16384, 32768, 65536]:
    run_bench(M, 2048, [2048, 2048, 2048])

print("\nDone!")
