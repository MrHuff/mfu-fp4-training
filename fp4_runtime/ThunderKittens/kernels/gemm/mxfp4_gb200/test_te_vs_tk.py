#!/usr/bin/env python3
"""
Numerical comparison: TE MXFP4 Quantization + TK GEMM vs TK-only vs BF16 reference.

Key insight about scale conventions:
  - TK: scale = cvt_e8m0(amax/6.0, ceil). GEMM result = sum(fp4*sx * fp4*sw) — correct magnitude.
  - TE (use_global_scale=False): scale = round(log2(vec_max))+127. GEMM needs alpha=1/36.
    Because TE computes effective_val = vec_max, then round(log2(vec_max)).
    The quantized value is x * (6 / (2^exponent)), and dequant is fp4 * 2^exponent.
    Since scale tracks vec_max not vec_max/6, the GEMM result is inflated by 6^2 = 36.

Paths tested:
  1. TK quantize → TK GEMM (current working path)
  2. TE ref quantize (no global scale) → TE ref GEMM (with alpha=1/36)
  3. TE ref quantize → TK GEMM (cross-path — requires NO alpha since TK scales already account for /6)
  4. BF16 reference matmul
"""

import sys
import os
import torch

# TK kernel imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import mxfp4_quantize, mxfp4_gemm

# TE MXFP4 reference imports
_te_root = "/opt/mfu/EXTERNAL_PATH"
if _te_root not in sys.path:
    sys.path.insert(0, _te_root)
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import (
    MXFP4QuantizerRef, cast_from_fp4x2, e8m0_to_scale, cast_to_fp4x2
)

LOG = open("/tmp/te_vs_tk_mxfp4.log", "w")
def p(msg=""):
    LOG.write(msg + "\n")
    LOG.flush()


# ===== Helper: Convert flat E8M0 scales to TK's MMA swizzle layout =====
def flat_scales_to_tk_swizzle(scales_flat: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Convert TE's flat E8M0 scales (M, K//32) to TK's MMA-swizzled layout (M//128, K//128, 32, 16)."""
    n_tile_rows = M // 128
    n_tile_cols = K // 128

    out = torch.zeros(n_tile_rows, n_tile_cols, 32, 16, dtype=torch.uint8, device=scales_flat.device)

    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            for local_row in range(128):
                global_row = tr * 128 + local_row
                for k_blk in range(4):
                    global_k_blk = tc * 4 + k_blk
                    if global_row < scales_flat.shape[0] and global_k_blk < scales_flat.shape[1]:
                        row_in_32 = local_row % 32
                        row_chunk = local_row // 32
                        swizzle_col = row_chunk * 4 + k_blk
                        out[tr, tc, row_in_32, swizzle_col] = scales_flat[global_row, global_k_blk]
    return out


def tk_unswizzle_scales(tk_scales: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Convert TK's swizzled scales back to flat (M, K//32) for comparison."""
    n_tile_rows = M // 128
    n_tile_cols = K // 128
    n_scale_cols = K // 32
    out = torch.zeros(M, n_scale_cols, dtype=torch.uint8, device=tk_scales.device)

    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            for local_row in range(128):
                global_row = tr * 128 + local_row
                for k_blk in range(4):
                    global_k_blk = tc * 4 + k_blk
                    row_in_32 = local_row % 32
                    row_chunk = local_row // 32
                    swizzle_col = row_chunk * 4 + k_blk
                    if global_row < M and global_k_blk < n_scale_cols:
                        out[global_row, global_k_blk] = tk_scales[tr, tc, row_in_32, swizzle_col]
    return out


# ===== TE reference tiled GEMM with proper alpha =====
def te_ref_gemm(qx, sx, qw, sw, alpha=1.0/36.0, out_dtype=torch.bfloat16):
    """Reference tiled GEMM for MXFP4 with E8M0 scales and alpha factor."""
    x_vals = cast_from_fp4x2(qx, torch.float32)
    w_vals = cast_from_fp4x2(qw, torch.float32)
    sx_float = e8m0_to_scale(sx)
    sw_float = e8m0_to_scale(sw)

    M, K = x_vals.shape
    N = w_vals.shape[0]
    block_size = 32
    n_blocks = K // block_size

    y = torch.zeros(M, N, dtype=torch.float32, device=qx.device)
    for k in range(n_blocks):
        x_blk = x_vals[:, k*block_size:(k+1)*block_size]
        w_blk = w_vals[:, k*block_size:(k+1)*block_size]
        gemm_blk = torch.mm(x_blk, w_blk.T)
        y += torch.outer(sx_float[:, k], sw_float[:, k]) * gemm_blk

    return (alpha * y).to(out_dtype)


def tk_quantize(x_bf16):
    """Run TK quantize."""
    M, K = x_bf16.shape
    A_fp4 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    mxfp4_quantize(x_bf16, A_fp4, A_sc)
    torch.cuda.synchronize()
    return A_fp4, A_sc


def te_quantize(x_bf16):
    """TE MXFP4 reference quantize: block-32, E8M0, decode-centric, no global scale."""
    quantizer = MXFP4QuantizerRef(
        encode_centric=False,
        quant_tile_shape=(1, 32),
        use_global_scale=False,
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_random_sign_mask=False,
    )
    result = quantizer.quantize(x_bf16.float())
    return result.data, result.scale


def compare(name_a, out_a, name_b, out_b):
    """Print comparison metrics."""
    diff = (out_a.float() - out_b.float()).abs()
    mean_a = out_a.float().abs().mean().item()
    mean_b = out_b.float().abs().mean().item()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel = mean_diff / max(mean_b, 1e-9) * 100

    p(f"  {name_a:20s} vs {name_b:15s}: "
      f"mean|A|={mean_a:12.4f}  mean|B|={mean_b:12.4f}  "
      f"max_diff={max_diff:10.4f}  mean_diff={mean_diff:10.4f}  rel={rel:8.2f}%")
    return rel


def analyze_scales(te_scales, tk_scales_flat, label=""):
    """Analyze scale differences."""
    M = te_scales.shape[0]
    K32 = te_scales.shape[1]

    te_f = te_scales.float()
    tk_f = tk_scales_flat.float()

    diff = (te_f - tk_f).int()
    exact_match = (te_scales == tk_scales_flat).sum().item()
    total = te_scales.numel()
    off_by_1 = ((diff.abs() == 1).sum().item())
    off_by_2 = ((diff.abs() == 2).sum().item())
    mean_diff = diff.float().mean().item()

    p(f"  {label} Scale analysis ({total} values):")
    p(f"    Exact match:    {exact_match:7d} ({100*exact_match/total:.1f}%)")
    p(f"    Off by 1:       {off_by_1:7d} ({100*off_by_1/total:.1f}%)")
    p(f"    Off by 2:       {off_by_2:7d} ({100*off_by_2/total:.1f}%)")
    p(f"    Mean TE-TK:     {mean_diff:.3f}")
    p(f"    TE range:       [{te_scales.min().item()}, {te_scales.max().item()}]")
    p(f"    TK range:       [{tk_scales_flat.min().item()}, {tk_scales_flat.max().item()}]")


def run_test(M, N, K, seed=42):
    """Run full comparison."""
    p(f"\n{'='*80}")
    p(f"  M={M}, N={N}, K={K}")
    p(f"{'='*80}")

    torch.manual_seed(seed)
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # === BF16 reference ===
    C_bf16 = torch.mm(A_bf16.float(), B_bf16.float().T).to(torch.bfloat16)

    # === TK: quantize → GEMM ===
    tk_qA, tk_sA = tk_quantize(A_bf16)
    tk_qB, tk_sB = tk_quantize(B_bf16)
    C_tk = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    mxfp4_gemm(tk_qA, tk_sA, tk_qB, tk_sB, C_tk)
    torch.cuda.synchronize()

    # === TE: quantize → ref GEMM (alpha=1/36) ===
    te_qA, te_sA = te_quantize(A_bf16)
    te_qB, te_sB = te_quantize(B_bf16)
    C_te_36 = te_ref_gemm(te_qA, te_sA, te_qB, te_sB, alpha=1.0/36.0)

    # === TE: quantize → ref GEMM (alpha=1.0 — no correction) ===
    C_te_1 = te_ref_gemm(te_qA, te_sA, te_qB, te_sB, alpha=1.0)

    # === Cross-path: TE quantize → TK GEMM ===
    # Need to swizzle TE scales to TK layout
    te_sA_sw = flat_scales_to_tk_swizzle(te_sA.cuda(), M, K)
    te_sB_sw = flat_scales_to_tk_swizzle(te_sB.cuda(), N, K)
    te_qA_tk = te_qA.view(torch.float4_e2m1fn_x2).cuda()
    te_qB_tk = te_qB.view(torch.float4_e2m1fn_x2).cuda()
    C_cross = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    mxfp4_gemm(te_qA_tk, te_sA_sw, te_qB_tk, te_sB_sw, C_cross)
    torch.cuda.synchronize()

    # === TK data + TE ref GEMM ===
    # Un-swizzle TK scales to flat, then run TE ref GEMM to compare quantization only
    tk_sA_flat = tk_unswizzle_scales(tk_sA.cpu(), M, K)
    tk_sB_flat = tk_unswizzle_scales(tk_sB.cpu(), N, K)
    tk_qA_bytes = tk_qA.view(torch.uint8).cpu()
    tk_qB_bytes = tk_qB.view(torch.uint8).cpu()

    # TK uses scale = ceil(log2(amax/6))+127, so the GEMM already has /6^2 = /36 baked in
    # → alpha=1.0 for the ref GEMM
    C_tk_ref = te_ref_gemm(tk_qA_bytes.cuda(), tk_sA_flat.cuda(),
                            tk_qB_bytes.cuda(), tk_sB_flat.cuda(), alpha=1.0)

    # === Cross-path with scale correction ===
    # TE scales track vec_max: E8M0 = round(log2(vec_max)) + 127
    # TK scales track vec_max/6: E8M0 = ceil(log2(vec_max/6)) + 127
    # Difference ≈ round(log2(6)) ≈ 3 (log2(6) = 2.585)
    # To feed TE quantized data into TK GEMM, we need to adjust scales:
    #   TE_to_TK_scale = TE_scale - round(log2(6)) ≈ TE_scale - 3
    # But this is approximate. Better to also apply alpha=1/36 after TK GEMM.
    # For now, just divide the TK GEMM result by 36 to match TE convention.
    C_cross_corrected = (C_cross.float() / 36.0).to(torch.bfloat16)

    # === Comparisons ===
    p(f"\n  --- GEMM Output vs BF16 ref ---")
    compare("TK (native)",        C_tk,               "BF16", C_bf16)
    compare("TE ref (a=1/36)",    C_te_36,             "BF16", C_bf16)
    compare("Cross (TE→TK raw)",  C_cross,             "BF16", C_bf16)
    compare("Cross (TE→TK /36)",  C_cross_corrected,   "BF16", C_bf16)
    compare("TK data→ref",        C_tk_ref,            "BF16", C_bf16)

    p(f"\n  --- Quantized path agreement ---")
    compare("TK native",          C_tk,               "TK data→ref", C_tk_ref)
    compare("Cross /36",          C_cross_corrected,   "TE ref(1/36)", C_te_36)
    compare("TE ref (a=1/36)",    C_te_36,             "TK data→ref", C_tk_ref)

    # === Scale analysis ===
    p(f"\n  --- Scale analysis ---")
    analyze_scales(te_sA.cpu(), tk_sA_flat.cpu(), label="A")

    # === FP4 data comparison ===
    te_bytes = te_qA.view(torch.uint8).cpu()
    tk_bytes = tk_qA_bytes.cpu()
    if te_bytes.shape == tk_bytes.shape:
        total = te_bytes.numel()
        match = (te_bytes == tk_bytes).sum().item()
        p(f"  FP4 data byte match: {match}/{total} ({100*match/total:.2f}%)")
    
    # === Dequantized comparison (quantize → dequantize → compare vs original) ===
    p(f"\n  --- Dequant quality (quantize → dequant → compare vs BF16 input) ---")
    # TE decode: fp4_val * 2^(scale-127). Since TE scale = round(log2(amax))+127,
    # the dequantized range is fp4*amax, but original was fp4*amax/6. So alpha=1/6.
    # Wait - more precisely: TE quantizes as x*(6/2^exp) then stores 2^exp.
    # So dequant is fp4 * 2^(scale-127). But the quantized fp4 was x/(2^exp/6) = 6x/2^exp.
    # So dequant = fp4 * 2^(scale-127) = 6x/2^exp * 2^exp = 6x. Need alpha=1/6.
    te_deq = dequant_ref(te_qA.cpu(), te_sA.cpu(), M, K, alpha=1.0/6.0)
    tk_deq = dequant_ref(tk_qA_bytes.cpu(), tk_sA_flat.cpu(), M, K, alpha=1.0)
    
    te_err = (te_deq.cuda() - A_bf16.float()).abs().mean().item()
    tk_err = (tk_deq.cuda() - A_bf16.float()).abs().mean().item()
    orig_rms = A_bf16.float().abs().mean().item()
    p(f"  TE dequant error:  {te_err:.6f} ({100*te_err/orig_rms:.2f}% of mean|A|)")
    p(f"  TK dequant error:  {tk_err:.6f} ({100*tk_err/orig_rms:.2f}% of mean|A|)")

    return


def dequant_ref(fp4_packed, scales_flat, M, K, alpha=1.0):
    """Dequantize FP4 data with E8M0 scales."""
    fp4_vals = cast_from_fp4x2(fp4_packed, torch.float32)
    scale_float = e8m0_to_scale(scales_flat)
    block_size = 32
    n_blocks = K // block_size
    result = torch.zeros(M, K, dtype=torch.float32)
    for k in range(n_blocks):
        result[:, k*block_size:(k+1)*block_size] = (
            fp4_vals[:, k*block_size:(k+1)*block_size] * scale_float[:, k:k+1] * alpha
        )
    return result


# ===== Main =====
p("=" * 80)
p("  TE MXFP4 vs TK MXFP4: Quantization + GEMM Numerical Comparison")
p("=" * 80)

try:
    p(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    p(f"GPU: {torch.cuda.get_device_name(0)}")
    p()
    p("Scale conventions:")
    p("  TE (use_global_scale=False): E8M0 = round(log2(vec_max)) + 127")
    p("    GEMM: alpha = 1/36 to compensate for scale tracking vec_max not vec_max/6")
    p("  TK: E8M0 = ceil(log2(vec_max/6)) + 127")
    p("    GEMM: no alpha needed, scale already accounts for /6")

    for M, N, K in [
        (256, 256, 256),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]:
        run_test(M, N, K)

    p(f"\n{'='*80}")
    p("  DONE")
    p(f"{'='*80}")
except Exception as e:
    p(f"ERROR: {e}")
    import traceback
    p(traceback.format_exc())
finally:
    LOG.close()
