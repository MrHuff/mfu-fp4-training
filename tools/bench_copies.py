#!/usr/bin/env python3
"""Measure: CUDA graph replay copy cost vs Python overhead."""
import torch
torch.manual_seed(42)

dim = 2048; q_dim = 2048; k_dim = 2048; v_dim = 2048
N_total = q_dim + k_dim + v_dim; K = dim

def bench(fn, w=10, n=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

print(f"{'M':>7s}  {'3×split→dy':>11s}  {'all copies':>11s}  {'3×clone out':>11s}  {'total graph':>11s}")
print("-" * 60)

for M in [4096, 16384, 32768, 65536]:
    dy_s = torch.empty(M, N_total, dtype=torch.bfloat16, device='cuda')
    gq = torch.randn(M, q_dim, dtype=torch.bfloat16, device='cuda')
    gk = torch.randn(M, k_dim, dtype=torch.bfloat16, device='cuda')
    gv = torch.randn(M, v_dim, dtype=torch.bfloat16, device='cuda')

    # Small static buffers for weights/x
    cf = torch.empty(K, N_total//2, dtype=torch.uint8, device='cuda')
    cs = torch.empty(K, N_total//64, dtype=torch.uint8, device='cuda')
    ws = torch.empty(3, dtype=torch.float32, device='cuda')
    xf = torch.empty(K, M//2, dtype=torch.uint8, device='cuda')
    xs = torch.empty(K, M//64, dtype=torch.uint8, device='cuda')
    xsg = torch.empty(1, dtype=torch.float32, device='cuda')
    inp = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')
    nw = torch.empty(K, dtype=torch.bfloat16, device='cuda')
    irms = torch.empty(M, 1, dtype=torch.float32, device='cuda')
    # Sources
    cf2=cf.clone();cs2=cs.clone();ws2=ws.clone()
    xf2=xf.clone();xs2=xs.clone();xsg2=xsg.clone()
    inp2=inp.clone();nw2=nw.clone();irms2=irms.clone()

    def split_copy():
        dy_s[:, :q_dim].copy_(gq)
        dy_s[:, q_dim:q_dim+k_dim].copy_(gk)
        dy_s[:, q_dim+k_dim:].copy_(gv)
    t_dy = bench(split_copy)

    def all_copy():
        split_copy()
        cf.copy_(cf2);cs.copy_(cs2);ws.copy_(ws2)
        xf.copy_(xf2);xs.copy_(xs2);xsg.copy_(xsg2)
        inp.copy_(inp2);nw.copy_(nw2);irms.copy_(irms2)
    t_all = bench(all_copy)

    gi = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')
    gw = torch.empty(N_total, K, dtype=torch.bfloat16, device='cuda')
    gnw = torch.empty(K, dtype=torch.float32, device='cuda')
    def clone_out():
        gi.clone(); gw.clone(); gnw.clone()
    t_clone = bench(clone_out)

    t_tot = t_all + t_clone
    print(f"{M:7d}  {t_dy:10.3f}ms  {t_all:10.3f}ms  {t_clone:10.3f}ms  {t_tot:10.3f}ms")
    torch.cuda.empty_cache()

print(f"\nPython overhead (non-graph) at M=65536 ≈ 0.537ms")
print(f"If total_graph_copies < 0.537ms → extending graph to M=65536 wins")
