#!/usr/bin/env python3
"""NCU profiling script for batched GEMM — runs a single invocation."""
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

M, K, N = 16384, 2048, 2048
NB = 3

A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / (K ** 0.5)
B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / (K ** 0.5)
Aq = quantize(A); Bq = quantize(B)

# Warmup
for _ in range(2):
    D_list = [torch.zeros(M, N, dtype=torch.bfloat16, device='cuda') for _ in range(NB)]
    nvfp4_batched_gemm(
        [Aq[0]] * NB, [Aq[1]] * NB, [Aq[2]] * NB,
        [Bq[0]] * NB, [Bq[1]] * NB, [Bq[2]] * NB, D_list)
torch.cuda.synchronize()

# Also run standard GEMM for comparison
D_ref = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
nvfp4_gemm(Aq[0], Aq[1], Aq[2], Bq[0], Bq[1], Bq[2], D_ref)
torch.cuda.synchronize()

# Profiled run
D_list = [torch.zeros(M, N, dtype=torch.bfloat16, device='cuda') for _ in range(NB)]
nvfp4_batched_gemm(
    [Aq[0]] * NB, [Aq[1]] * NB, [Aq[2]] * NB,
    [Bq[0]] * NB, [Bq[1]] * NB, [Bq[2]] * NB, D_list)
torch.cuda.synchronize()

print("Profile run complete")
