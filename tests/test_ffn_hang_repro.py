#!/usr/bin/env python3
import os, sys
import signal
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)
import torch

os.environ.setdefault('NVTE_CUSTOM_QUANT', '1')
os.environ.setdefault('USE_TK_QUANT', '1')
os.environ.setdefault('USE_TK_GEMM', '1')

from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4_TK, FusedFeedForwardFP4_TE

def cs(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()

def bench(fn, warmup=10, steps=30, timeout_s=30):
    class _Timeout(Exception): pass
    def _handler(signum, frame): raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    try:
        for _ in range(warmup):
            signal.alarm(timeout_s)
            fn()
            torch.cuda.synchronize()
        times = []
        for _ in range(steps):
            signal.alarm(timeout_s)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        signal.alarm(0)
        times.sort()
        return times[len(times) // 2]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def main():
    torch.manual_seed(42)
    K = 2048
    H = 5632
    te_ffn = FusedFeedForwardFP4_TE(K, H).cuda()
    tk_ffn = FusedFeedForwardFP4_TK(K, H).cuda()

    with torch.no_grad():
        tk_ffn.norm_weight.copy_(te_ffn.norm_weight)
        tk_ffn.w1_weight.copy_(te_ffn.w1_weight)
        tk_ffn.w3_weight.copy_(te_ffn.w3_weight)
        tk_ffn.w2_weight.copy_(te_ffn.w2_weight)

    x = torch.randn(1024, K, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    dY = torch.randn(1024, K, dtype=torch.bfloat16, device='cuda')
    y_te = te_ffn(x)
    x_tk = x.detach().clone().requires_grad_(True)
    y_tk = tk_ffn(x_tk)
    y_te.backward(dY)
    y_tk.backward(dY)
    del y_te, y_tk
    torch.cuda.empty_cache()

    M_VALUES = [256, 512, 1024, 2048, 4096]
    for M in M_VALUES:
        print(f"\n--- Testing M={M} ---", flush=True)
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        dY = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        print("Running TE FWD...", flush=True)
        te_fwd = bench(lambda: te_ffn(x))
        print("Running TE TOT...", flush=True)
        def te_fb():
            xr = x.detach().requires_grad_(True)
            te_ffn(xr).backward(dY)
        te_tot = bench(te_fb)
        
        print("Running TK FWD...", flush=True)
        tk_fwd = bench(lambda: tk_ffn(x))
        print("Running TK TOT...", flush=True)
        def tk_fb():
            xr = x.detach().requires_grad_(True)
            tk_ffn(xr).backward(dY)
        tk_tot = bench(tk_fb)
        print(f"M={M} SUCCESS!", flush=True)
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
