#!/usr/bin/env python3
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)
import torch
import torch.nn.functional as F

os.environ.setdefault('NVTE_CUSTOM_QUANT', '1')
os.environ.setdefault('USE_TK_QUANT', '1')
os.environ.setdefault('USE_TK_GEMM', '1')

from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4_TK

def main():
    torch.manual_seed(42)
    K = 2048
    H = 5632
    tk_ffn = FusedFeedForwardFP4_TK(K, H).cuda()

    M = 4096
    print(f"Testing M={M} forward...", flush=True)
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    print("Calling tk_ffn...", flush=True)
    out = tk_ffn(x)
    torch.cuda.synchronize()
    print("tk_ffn forward success!", flush=True)
    
    print("Testing backward...", flush=True)
    dY = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    out.backward(dY)
    torch.cuda.synchronize()
    print("backward success!", flush=True)
    
    # Also loop it a bit to mimic bench warmup
    print("Looping 5 times...", flush=True)
    for i in range(5):
        print(f"Iter {i}...", flush=True)
        out = tk_ffn(x)
        out.backward(dY)
        torch.cuda.synchronize()
    print("All done!")

if __name__ == '__main__':
    main()
