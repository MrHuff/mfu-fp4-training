"""Benchmark: row/col split with dynamic amax (no constant scale)."""
import torch, sys, time
sys.path.insert(0, '/opt/mfu/EXTERNAL_PATH')
import _tk_quant_v5 as tkq

import importlib.util
_gemm_so = '/opt/mfu/EXTERNAL_PATH'
_old_c = sys.modules.pop('_C', None)
spec = importlib.util.spec_from_file_location('_C', _gemm_so)
tk_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tk_mod)
if _old_c is not None: sys.modules['_C'] = _old_c
elif '_C' in sys.modules: del sys.modules['_C']

torch.manual_seed(42)

def make_weights(K, N_dims, device='cuda'):
    B_fp4, B_sc, B_sg = [], [], []
    for n_i in N_dims:
        W = torch.randn(K, n_i, dtype=torch.bfloat16, device=device)
        r = tkq.tk_quantize_for_gemm(W, True)
        B_fp4.append(r[2]); B_sc.append(r[3]); B_sg.append(r[4].to(torch.float32))
    return B_fp4, B_sc, B_sg

def bench(fn, nw=20, ni=100):
    for _ in range(nw): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ni): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ni * 1000

K = 2048
N_dims = [2048, 2048, 2048]

print("=== Row/Col Split with Dynamic Amax (ms) ===")
print("  baseline  = row+col quant → gemm → sum3")
print("  crit_path = row-only quant → gemm → sum3  (col-only deferred)")
print("  total     = row-only quant → gemm → sum3 → col-only quant\n")

for M in [2048, 4096, 8192, 16384, 32768, 65536]:
    gs = [torch.randn(M, n, dtype=torch.bfloat16, device='cuda') for n in N_dims]
    Bf, Bs, Bg = make_weights(K, N_dims)
    D_list = [torch.empty(M, K, dtype=torch.bfloat16, device='cuda') for _ in range(3)]
    D_sum = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')

    def do_gemm_sum(Af, As, Asg):
        tk_mod.nvfp4_batched_gemm(Af, As, Asg, Bf, Bs, Bg, D_list)
        tk_mod.sum3_bf16(D_list[0], D_list[1], D_list[2], D_sum)

    # Baseline: row+col together
    def baseline():
        qr = [tkq.tk_quantize_for_gemm(g, True) for g in gs]
        do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
    t_base = bench(baseline)

    # Split: row-only → gemm → sum3 (critical path)
    def split_crit():
        qr = [tkq.tk_quantize_for_gemm(g, False) for g in gs]
        do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
    t_crit = bench(split_crit)

    # Split: row-only → gemm → sum3 → col-only (total wall)
    def split_total():
        qr = [tkq.tk_quantize_for_gemm(g, False) for g in gs]
        do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
        sgs = [r[4] for r in qr]
        for i, g in enumerate(gs):
            tkq.tk_quantize_col_only(g, sgs[i])
    t_total = bench(split_total)

    save = t_base - t_crit
    print(f"  M={M:5d}: baseline={t_base:.3f}  crit_path={t_crit:.3f} [{save:+.3f}]  total={t_total:.3f}")

print("\nDone!")
