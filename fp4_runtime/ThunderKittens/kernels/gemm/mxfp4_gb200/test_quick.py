#!/usr/bin/env python3
"""Quick MXFP4 TK sanity test writing to file."""
import sys, os
log = open('/tmp/mxfp4_test.log', 'w')
def p(msg):
    log.write(msg + '\n')
    log.flush()

p('1: start')
try:
    import torch
    p('2: torch imported')
    sys.path.insert(0, os.path.dirname(__file__))
    from _C import mxfp4_quantize, mxfp4_gemm
    p('3: _C imported')
    
    # Basic CUDA
    a = torch.randn(100, 100, device='cuda')
    b = torch.matmul(a, a.T)
    p(f'4: basic matmul OK {b.shape}')
    
    M, K = 2048, 2048
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25
    p(f'5: A created')
    
    A_fp4 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device='cuda')
    A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device='cuda')
    p(f'6: outputs created')
    
    p('7: calling quantize...')
    mxfp4_quantize(A, A_fp4, A_sc)
    p('8: quantize returned, synchronizing...')
    torch.cuda.synchronize()
    p(f'9: sync OK! sc min={A_sc.min().item()} max={A_sc.max().item()}')
    
except Exception as e:
    p(f'ERROR: {e}')
    import traceback
    p(traceback.format_exc())
finally:
    log.close()
