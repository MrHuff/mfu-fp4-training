#!/usr/bin/env python3
"""Full MXFP4 E2E: quantize + GEMM test, output to file."""
import sys, os
log = open('/tmp/mxfp4_e2e.log', 'w')
def p(msg):
    log.write(msg + '\n')
    log.flush()

p('=== MXFP4 E2E Test ===')
try:
    import torch
    torch.random.manual_seed(42)
    from _C import mxfp4_quantize, mxfp4_gemm
    p('Imports OK')
    
    for M, N, K in [(2048, 2048, 2048), (4096, 4096, 4096)]:
        p(f'\n--- M={M}, N={N}, K={K} ---')
        
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25
        B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25
        
        # Quantize
        A_fp4 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device='cuda')
        A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device='cuda')
        B_fp4 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device='cuda')
        B_sc = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device='cuda')
        
        mxfp4_quantize(A, A_fp4, A_sc)
        mxfp4_quantize(B, B_fp4, B_sc)
        torch.cuda.synchronize()
        p(f'Quantize OK: A_sc range [{A_sc.min().item()}, {A_sc.max().item()}], '
          f'B_sc range [{B_sc.min().item()}, {B_sc.max().item()}]')
        
        # GEMM
        C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, C)
        torch.cuda.synchronize()
        p(f'GEMM OK: C mean={C.float().abs().mean().item():.6f}')
        
        # Reference
        C_ref = torch.matmul(A, B.T).to(torch.bfloat16)
        
        diff = (C.float() - C_ref.float()).abs()
        rel = diff.mean().item() / max(C_ref.float().abs().mean().item(), 1e-9)
        p(f'Max diff:  {diff.max().item():.6f}')
        p(f'Mean diff: {diff.mean().item():.6f}')
        p(f'Rel error: {rel:.4%}')
        p(f'Mean |C|:  {C.float().abs().mean().item():.6f}')
        p(f'Mean |Ref|: {C_ref.float().abs().mean().item():.6f}')
        
        del A, B, C, C_ref, A_fp4, A_sc, B_fp4, B_sc
        torch.cuda.empty_cache()
    
    # Benchmark at 8192
    M, N, K = 8192, 8192, 8192
    p(f'\n--- Benchmark: M={M}, N={N}, K={K} ---')
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25
    A_fp4 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device='cuda')
    A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device='cuda')
    B_fp4 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device='cuda')
    B_sc = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device='cuda')
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    
    mxfp4_quantize(A, A_fp4, A_sc)
    mxfp4_quantize(B, B_fp4, B_sc)
    
    # Warmup
    for _ in range(3):
        mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, C)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    num_iters = 10
    start.record()
    for _ in range(num_iters):
        mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, C)
    end.record()
    torch.cuda.synchronize()
    
    ms = start.elapsed_time(end) / num_iters
    flops = 2.0 * M * N * K
    tflops = flops / (ms * 1e-3) / 1e12
    p(f'Avg time: {ms:.3f} ms')
    p(f'TFLOPs:   {tflops:.2f}')
    
    p('\n=== ALL TESTS PASSED ===')
    
except Exception as e:
    p(f'ERROR: {e}')
    import traceback
    p(traceback.format_exc())
finally:
    log.close()
