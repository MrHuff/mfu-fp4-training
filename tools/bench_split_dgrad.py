#!/usr/bin/env python3
"""E2E benchmark: row/col quant split for QKV backward dgrad.

Direct kernel-level benchmark — constructs properly-formatted per-group
weight tensors (matching real training layout) and benchmarks the full
quant → GEMM → sum3 pipeline with split ON vs OFF.
"""
import os, sys, time
os.environ['USE_TK_GEMM'] = '1'
os.environ['USE_TK_QUANT'] = '1'

import torch

sys.path.insert(0, '/opt/mfu/EXTERNAL_PATH')
import _tk_quant_v5 as tkq

import importlib.util
_so = '/opt/mfu/EXTERNAL_PATH'
_old = sys.modules.pop('_C', None)
spec = importlib.util.spec_from_file_location('_C', _so)
tk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tk)
if _old: sys.modules['_C'] = _old
elif '_C' in sys.modules: del sys.modules['_C']

torch.manual_seed(42)

def bench_ms(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def make_weight_col(K, N_dims):
    """Create per-group col-quantized weight tensors."""
    Bf, Bs, Bg = [], [], []
    for n in N_dims:
        W = torch.randn(K, n, dtype=torch.bfloat16, device='cuda')
        r = tkq.tk_quantize_for_gemm(W, True)
        Bf.append(r[2]); Bs.append(r[3]); Bg.append(r[4].to(torch.float32))
    return Bf, Bs, Bg

# ── Configs ──
CONFIGS = [
    (2048, [2048, 2048, 2048], 'Llama-1B TP1 (K=2048, Q=K=V=2048)'),
    (4096, [4096, 1024, 1024], 'Llama-8B TP1 (K=4096, Q=4096,K=V=1024)'),
    (2048, [2048, 512, 512],   'Llama-8B TP2 (K=2048, Q=2048,K=V=512)'),
    (8192, [1024, 1024, 1024], 'Llama-70B TP8 (K=8192, Q=K=V=1024)'),
]

M_VALUES = [2048, 4096, 8192, 16384, 32768, 65536]

print("=" * 90)
print("QKV Backward dgrad: Row/Col Split Benchmark")
print("=" * 90)

for K, N_dims, desc in CONFIGS:
    n = len(N_dims)
    Bf, Bs, Bg = make_weight_col(K, N_dims)

    print(f"\n{'─' * 90}")
    print(f"  {desc}")
    print(f"{'─' * 90}")
    print(f"  {'M':>6s}  {'OFF (ms)':>9s}  {'ON (ms)':>9s}  {'Δ (ms)':>8s}  {'speedup':>8s}  {'note':s}")
    print(f"  {'-' * 70}")

    for M in M_VALUES:
        # Check all dims are multiples of 128
        if any(d % 128 != 0 for d in [M, K] + N_dims):
            print(f"  {M:6d}  SKIP (alignment)")
            continue

        gs = [torch.randn(M, ni, dtype=torch.bfloat16, device='cuda') for ni in N_dims]
        Dl = [torch.empty(M, K, dtype=torch.bfloat16, device='cuda') for _ in range(n)]
        Ds = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')

        def do_gemm_sum(Af, As, Asg):
            tk.nvfp4_batched_gemm(Af, As, Asg, Bf, Bs, Bg, Dl)
            tk.sum3_bf16(Dl[0], Dl[1], Dl[2], Ds) if n == 3 else None

        # ── Split OFF: row+col quant → GEMM → sum3 ──
        def pipeline_off():
            qr = [tkq.tk_quantize_for_gemm(g, True) for g in gs]
            do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
        try:
            t_off = bench_ms(pipeline_off)
        except Exception as e:
            print(f"  {M:6d}  OFF FAILED: {str(e)[:50]}")
            continue

        # ── Split ON: row-only → GEMM → sum3 → col-only ──
        def pipeline_on():
            qr = [tkq.tk_quantize_for_gemm(g, False) for g in gs]
            do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
            # Col-only quant (off critical path for dgrad)
            for i, g in enumerate(gs):
                tkq.tk_quantize_col_only(g, qr[i][4])
        try:
            t_on = bench_ms(pipeline_on)
        except Exception as e:
            print(f"  {M:6d}  ON FAILED: {str(e)[:50]}")
            continue

        # ── Split ON critical path only: row-only → GEMM → sum3 ──
        def pipeline_crit():
            qr = [tkq.tk_quantize_for_gemm(g, False) for g in gs]
            do_gemm_sum([r[0] for r in qr], [r[1] for r in qr], [r[4] for r in qr])
        t_crit = bench_ms(pipeline_crit)

        delta = t_off - t_on
        delta_crit = t_off - t_crit
        spd = t_off / t_crit
        print(f"  {M:6d}  {t_off:8.3f}ms  {t_on:8.3f}ms  {delta_crit:+7.3f}  {spd:7.2f}x  "
              f"crit={t_crit:.3f}ms")

        del gs, Dl, Ds

    del Bf, Bs, Bg
    torch.cuda.empty_cache()

print(f"\n{'=' * 90}")
print("Δ and speedup measured against CRITICAL PATH (row-only → GEMM → sum3, no col)")
print("ON (ms) includes total wall time with col-only (for comparison)")
print(f"{'=' * 90}")
