#!/usr/bin/env python3
"""
Focused TK MXFP4 GEMM diagnostic: uses known quantized values to isolate the magnitude bug.

Strategy:
  1. Create controlled FP4 data and E8M0 scales
  2. Compute expected result analytically
  3. Run TK GEMM and compare
  4. Also run TK quantize on known BF16 input and verify scales/data
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import mxfp4_quantize, mxfp4_gemm

LOG = open("/tmp/tk_diag.log", "w")
def p(msg=""):
    LOG.write(msg + "\n")
    LOG.flush()

# TE ref imports
_te_root = "/opt/mfu/EXTERNAL_PATH"
if _te_root not in sys.path:
    sys.path.insert(0, _te_root)
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import (
    cast_from_fp4x2, e8m0_to_scale
)


def tk_unswizzle_scales(tk_scales, M, K):
    """Convert TK's swizzled scales to flat (M, K//32)."""
    n_tile_rows = M // 128
    n_tile_cols = K // 128
    out = torch.zeros(M, K // 32, dtype=torch.uint8, device=tk_scales.device)
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            for lr in range(128):
                gr = tr * 128 + lr
                for kb in range(4):
                    gk = tc * 4 + kb
                    r32 = lr % 32
                    rc = lr // 32
                    sc = rc * 4 + kb
                    if gr < M and gk < (K // 32):
                        out[gr, gk] = tk_scales[tr, tc, r32, sc]
    return out


def ref_gemm_fp4_e8m0(qA, sA, qB, sB, M, N, K):
    """Reference matmul: C[m,n] = sum_k A_fp4[m,k]*sA[m,k//32] * B_fp4[n,k]*sB[n,k//32]"""
    A_vals = cast_from_fp4x2(qA, torch.float32)  # (M, K)
    B_vals = cast_from_fp4x2(qB, torch.float32)  # (N, K)
    sA_float = e8m0_to_scale(sA)  # (M, K//32)
    sB_float = e8m0_to_scale(sB)  # (N, K//32)

    result = torch.zeros(M, N, dtype=torch.float32, device=qA.device)
    for k in range(K // 32):
        a_blk = A_vals[:, k*32:(k+1)*32]  # (M, 32)
        b_blk = B_vals[:, k*32:(k+1)*32]  # (N, 32)
        partial = torch.mm(a_blk, b_blk.T)  # (M, N)
        result += torch.outer(sA_float[:, k], sB_float[:, k]) * partial
    return result.to(torch.bfloat16)


p("=" * 70)
p("  TK MXFP4 GEMM & Quantize Diagnostic")
p("=" * 70)

try:
    # ======================================================
    # Test 1: Quantize diagnostic — verify scale computation
    # ======================================================
    p("\n--- Test 1: Quantize Scale Verification ---")
    M, K = 256, 256
    # Use uniform values so we know the expected scale
    # If all values = 3.0, amax_block = 3.0, scale = ceil(log2(3.0/6.0)) = ceil(-1.0) = -1
    # E8M0 = -1 + 127 = 126
    val = 3.0
    A = torch.full((M, K), val, dtype=torch.bfloat16, device="cuda")
    
    A_fp4 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    mxfp4_quantize(A, A_fp4, A_sc)
    torch.cuda.synchronize()

    sc_flat = tk_unswizzle_scales(A_sc.cpu(), M, K)
    
    expected_e8m0 = 126  # ceil(log2(3.0/6.0)) + 127 = ceil(-1) + 127 = 126
    unique_sc = sc_flat.unique()
    p(f"  Input: all {val}, expected E8M0={expected_e8m0}")
    p(f"  TK scales unique: {unique_sc.tolist()}")
    
    # Check FP4 values
    fp4_bytes = A_fp4.view(torch.uint8).cpu()
    # With scale = 2^(-1) = 0.5, value 3.0 is quantized as: 3.0 / 0.5 = 6.0
    # FP4 E2M1: 6.0 → code 7 (max positive). Two of these: 0x77 = 119
    p(f"  FP4 bytes unique: {fp4_bytes.unique().tolist()}")
    p(f"  Expected: 3.0/0.5=6.0 → FP4 code 7, packed 0x77=119")
    
    # ======================================================
    # Test 2: GEMM with uniform data
    # ======================================================
    p("\n--- Test 2: GEMM with Uniform Data ---")
    M, N, K = 256, 256, 256
    
    # All ones (BF16) → quantize → GEMM
    A_bf16 = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
    B_bf16 = torch.ones(N, K, dtype=torch.bfloat16, device="cuda")
    
    # Quantize
    Aq = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    As = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    Bq = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    Bs = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    mxfp4_quantize(A_bf16, Aq, As)
    mxfp4_quantize(B_bf16, Bq, Bs)
    torch.cuda.synchronize()

    As_flat = tk_unswizzle_scales(As.cpu(), M, K)
    Bs_flat = tk_unswizzle_scales(Bs.cpu(), N, K)

    p(f"  A scale unique: {As_flat.unique().tolist()}")
    p(f"  A FP4 unique: {Aq.view(torch.uint8).cpu().unique().tolist()}")
    
    # Scale: ceil(log2(1.0/6.0)) + 127 = ceil(-2.585) + 127 = -2 + 127 = 125
    # value_float = 1.0 / 2^(-2) = 1.0 * 4 = 4.0
    # FP4: 4.0 → code 6 (E2M1: 100.0 = 4.0). Packed: 0x66 = 102
    p(f"  Expected: scale E8M0=125, FP4 code 6 (4.0), packed 0x66=102")

    # Reference GEMM
    C_ref = ref_gemm_fp4_e8m0(Aq.view(torch.uint8).cpu(), As_flat, Bq.view(torch.uint8).cpu(), Bs_flat,
                               M, N, K)
    
    # TK GEMM
    C_tk = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    mxfp4_gemm(Aq, As, Bq, Bs, C_tk)
    torch.cuda.synchronize()

    # BF16 ref
    C_bf16 = torch.mm(A_bf16.float(), B_bf16.float().T).to(torch.bfloat16)

    p(f"  BF16 ref:     mean={C_bf16.float().mean():.4f}, [0,0]={C_bf16[0,0].item():.4f}")  # Should be 256
    p(f"  Ref GEMM:     mean={C_ref.float().mean():.4f}, [0,0]={C_ref[0,0].item():.4f}")
    p(f"  TK GEMM:      mean={C_tk.float().mean():.4f}, [0,0]={C_tk[0,0].item():.4f}")

    # ======================================================
    # Test 3: GEMM with random data at multiple sizes
    # ======================================================
    p("\n--- Test 3: GEMM at Multiple Sizes ---")
    for M, N, K in [(256, 256, 256), (256, 256, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        torch.manual_seed(42)
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        Aq = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        As = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
        Bq = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        Bs = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
        mxfp4_quantize(A_bf16, Aq, As)
        mxfp4_quantize(B_bf16, Bq, Bs)
        torch.cuda.synchronize()

        As_flat = tk_unswizzle_scales(As.cpu(), M, K)
        Bs_flat = tk_unswizzle_scales(Bs.cpu(), N, K)

        C_ref = ref_gemm_fp4_e8m0(Aq.view(torch.uint8).cpu(), As_flat, Bq.view(torch.uint8).cpu(), Bs_flat,
                                    M, N, K)
        C_tk = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        mxfp4_gemm(Aq, As, Bq, Bs, C_tk)
        torch.cuda.synchronize()
        C_bf16 = torch.mm(A_bf16.float(), B_bf16.float().T).to(torch.bfloat16)

        ref_mean = C_ref.float().abs().mean().item()
        tk_mean = C_tk.float().abs().mean().item()
        bf16_mean = C_bf16.float().abs().mean().item()
        ratio_tk_ref = tk_mean / ref_mean if ref_mean > 0 else float('inf')

        p(f"  {M:5d}x{N:5d}x{K:5d}: "
          f"BF16={bf16_mean:10.4f}  Ref={ref_mean:10.4f}  TK={tk_mean:10.4f}  "
          f"TK/Ref={ratio_tk_ref:.4f}")

    p("\n=== DONE ===")

except Exception as e:
    p(f"ERROR: {e}")
    import traceback
    p(traceback.format_exc())
finally:
    LOG.close()
