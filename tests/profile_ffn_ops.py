#!/usr/bin/env python3
"""Per-operation FFN profiling: times each TK vs TE op individually.

Reports a breakdown of every forward and backward operation so we can
identify exactly which operations cause the TK slowdown at large M.

Usage:
    setsid python3 tests/profile_ffn_ops.py
"""

import os, sys, signal

os.environ['CYPARI_NO_SIGNALS'] = '1'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault('NVTE_NVFP4_DISABLE_RHT', '1')
os.environ.setdefault('NVTE_NVFP4_DISABLE_2D_QUANTIZATION', '1')
os.environ.setdefault('NVTE_NVFP4_ENCODE_CENTRIC', '0')
os.environ.setdefault('NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING', '1')
os.environ.setdefault('NVTE_CUSTOM_QUANT', '1')
os.environ.setdefault('USE_TK_QUANT', '1')
os.environ.setdefault('USE_TK_GEMM', '1')
os.environ.setdefault('FUSED_TE_QUANT', '0')
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
os.environ.setdefault('NVTE_CUDA_GRAPHS', '0')
os.environ.setdefault('NVTE_FP4_MULTI_STREAM', '0')
os.environ.setdefault('NVTE_MS', '0')
os.environ.setdefault('NVTE_MULTI_STREAM', '0')

signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])

import torch
import torch.nn.functional as F

from low_bits_training.quantization.fused_te_linear import (
    FusedFeedForwardFP4_TE,
    FusedFeedForwardFP4_TK,
    _fast_quantize,
    _get_te_fused,
    _TKQuantized,
)

signal.signal(signal.SIGINT, signal.default_int_handler)

# ─── Bench helper ──────────────────────────────────────────────────────
def bench(fn, warmup=5, steps=20):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]

# ─── Main ──────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    K = 2048
    H = 5632

    import transformer_engine_torch as tex
    from transformer_engine.pytorch.constants import TE_DType
    from low_bits_training.quantization.tk_gemm import (
        _get_tk, _get_tk_quant_for_gemm, tk_forward_gemm,
        tk_dgrad_gemm, tk_wgrad_gemm, tk_grouped_wgrad_gemm,
        _get_sg_tile_indices, _get_wgrad_buf,
    )

    tk = _get_tk()
    te_fused = _get_te_fused()

    # Get TK quantizer
    from low_bits_training.quantization.fused_te_linear import _get_tk_quant
    tk_q = _get_tk_quant()

    ws = torch.empty(32*1024*1024, dtype=torch.uint8, device='cuda')

    M_VALUES = [256, 1024, 4096, 8192, 16384, 32768, 65536]

    print("=" * 120, flush=True)
    print(f"  FFN Per-Op Profile: TK vs TE  |  K={K}, H={H}", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name()}", flush=True)
    print("=" * 120, flush=True)

    # ── Forward Op-by-Op ──
    print("\n  ── Forward Per-Op Timing (ms) ──", flush=True)
    hdr = (f"  {'M':>7} | {'Op':>25} | {'TE':>9} | {'TK':>9} | {'Δms':>8} | {'sp':>6}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    for M in M_VALUES:
        try:
            x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01
            nw = torch.ones(K, dtype=torch.bfloat16, device='cuda')
            w1_bf16 = torch.randn(H, K, dtype=torch.bfloat16, device='cuda') * 0.01
            w3_bf16 = torch.randn(H, K, dtype=torch.bfloat16, device='cuda') * 0.01
            w2_bf16 = torch.randn(K, H, dtype=torch.bfloat16, device='cuda') * 0.01
            h13_bf16 = torch.randn(M, 2*H, dtype=torch.bfloat16, device='cuda') * 0.01

            ops = []

            # --- Op 1: Quantize x ---
            te_q1 = bench(lambda: _fast_quantize(x_bf16, tk_swizzle=False))
            tk_q1 = bench(lambda: _fast_quantize(x_bf16, tk_swizzle=True))
            ops.append(("quantize(x)", te_q1, tk_q1))

            # --- Op 2: Grouped GEMM W1,W3 ---
            x_te = _fast_quantize(x_bf16, tk_swizzle=False)
            x_tk = _fast_quantize(x_bf16, tk_swizzle=True)
            w1_te = _fast_quantize(w1_bf16, tk_swizzle=False)
            w3_te = _fast_quantize(w3_bf16, tk_swizzle=False)
            w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=0)
            N_dims = [H, H]

            # TE grouped GEMM: 2 separate calls
            def te_gemm_w13():
                out1 = torch.empty(M, H, dtype=torch.bfloat16, device='cuda')
                tex.generic_gemm(w1_te, True, x_te, False, out1, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
                out3 = torch.empty(M, H, dtype=torch.bfloat16, device='cuda')
                tex.generic_gemm(w3_te, True, x_te, False, out3, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
            te_g2 = bench(te_gemm_w13)

            # TK grouped GEMM: single call
            # Quantize W13 for grouped GEMM
            wc_fp4_row, wc_sc_row, fwd_b_sg, wc_fp4_cols, wc_sc_cols, _, sg_cat, _ = \
                tk_q.tk_group_quantize_for_gemm(w13_bf16, N_dims)
            x_fp4, x_sc, x_sg = x_tk._tk_row
            h13_out = torch.empty(M, 2*H, dtype=torch.bfloat16, device='cuda')
            def tk_gemm_w13():
                tk.nvfp4_grouped_gemm(x_fp4, x_sc, x_sg, wc_fp4_row, wc_sc_row, fwd_b_sg, h13_out)
            tk_g2 = bench(tk_gemm_w13)
            ops.append(("GEMM([W1,W3] grouped)", te_g2, tk_g2))

            # --- Op 3: SiLU + Quantize ---
            h_bf16 = torch.randn(M, H, dtype=torch.bfloat16, device='cuda') * 0.01
            def te_silu_q():
                h, _ = te_fused.fused_silu_mul_strided_bf16(h13_bf16, H)
                _fast_quantize(h, tk_swizzle=False)
            te_sq = bench(te_silu_q)

            if hasattr(tk_q, 'tk_silu_quantize_for_gemm'):
                def tk_silu_q():
                    tk_q.tk_silu_quantize_for_gemm(h13_bf16, H)
                tk_sq = bench(tk_silu_q)
            else:
                def tk_silu_q():
                    h, _ = te_fused.fused_silu_mul_strided_bf16(h13_bf16, H)
                    _fast_quantize(h, tk_swizzle=True)
                tk_sq = bench(tk_silu_q)
            ops.append(("fused_silu+quantize", te_sq, tk_sq))

            # --- Op 4: GEMM W2 (single) ---
            h_tk = _fast_quantize(h_bf16, tk_swizzle=True)
            w2_te = _fast_quantize(w2_bf16, tk_swizzle=False)
            w2_tk = _fast_quantize(w2_bf16, tk_swizzle=True)
            out_te = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')
            out_tk = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')

            def te_gemm_w2():
                tex.generic_gemm(w2_te, True, _fast_quantize(h_bf16, tk_swizzle=False), False,
                                 out_te, None, TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
            te_g4 = bench(te_gemm_w2)

            def tk_gemm_w2():
                tk_forward_gemm(h_tk, w2_tk, out_tk)
            tk_g4 = bench(tk_gemm_w2)
            ops.append(("GEMM(W2 single)", te_g4, tk_g4))

            # --- Print forward summary ---
            total_te = sum(o[1] for o in ops)
            total_tk = sum(o[2] for o in ops)
            for name, te_t, tk_t in ops:
                delta = te_t - tk_t  # positive means TK is faster
                sp = te_t / tk_t if tk_t > 0 else 0
                print(f"  {M:>7} | {name:>25} | {te_t:>7.3f}ms | {tk_t:>7.3f}ms | {delta:>+7.3f} | {sp:>5.2f}x", flush=True)
            sp_tot = total_te / total_tk if total_tk > 0 else 0
            delta_tot = total_te - total_tk
            print(f"  {M:>7} | {'TOTAL FWD':>25} | {total_te:>7.3f}ms | {total_tk:>7.3f}ms | {delta_tot:>+7.3f} | {sp_tot:>5.2f}x", flush=True)
            print("  " + "-" * (len(hdr) - 2), flush=True)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {M:>7} | FAILED: {str(e)[:60]}", flush=True)

    # ── Backward Op-by-Op ──
    print("\n  ── Backward Per-Op Timing (ms) ──", flush=True)
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    for M in M_VALUES:
        try:
            dY_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01
            h13_bf16 = torch.randn(M, 2*H, dtype=torch.bfloat16, device='cuda') * 0.01
            dh_bf16 = torch.randn(M, H, dtype=torch.bfloat16, device='cuda') * 0.01
            w2_bf16 = torch.randn(K, H, dtype=torch.bfloat16, device='cuda') * 0.01
            h_bf16 = torch.randn(M, H, dtype=torch.bfloat16, device='cuda') * 0.01
            x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01
            w1_bf16 = torch.randn(H, K, dtype=torch.bfloat16, device='cuda') * 0.01
            w3_bf16 = torch.randn(H, K, dtype=torch.bfloat16, device='cuda') * 0.01
            dh1_bf16 = torch.randn(M, H, dtype=torch.bfloat16, device='cuda') * 0.01
            dh3_bf16 = torch.randn(M, H, dtype=torch.bfloat16, device='cuda') * 0.01
            input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01

            ops = []

            # --- Op B1: quantize(dY) ---
            te_bq = bench(lambda: _fast_quantize(dY_bf16, tk_swizzle=False))
            tk_bq = bench(lambda: _fast_quantize(dY_bf16, tk_swizzle=True))
            ops.append(("quantize(dY)", te_bq, tk_bq))

            # --- Op B2: W2 dgrad ---
            dY_te = _fast_quantize(dY_bf16, tk_swizzle=False)
            dY_tk = _fast_quantize(dY_bf16, tk_swizzle=True)
            w2_te = _fast_quantize(w2_bf16, tk_swizzle=False)
            w2_tk = _fast_quantize(w2_bf16, tk_swizzle=True)

            def te_dgrad_w2():
                tex.generic_gemm(w2_te, False, dY_te, False, None, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
            te_dg = bench(te_dgrad_w2)
            tk_dg = bench(lambda: tk_dgrad_gemm(dY_tk, w2_tk))
            ops.append(("GEMM dgrad(dY,W2)", te_dg, tk_dg))

            # --- Op B3: W2 wgrad ---
            h_te = _fast_quantize(h_bf16, tk_swizzle=False)
            h_tk = _fast_quantize(h_bf16, tk_swizzle=True)

            def te_wgrad_w2():
                tex.generic_gemm(h_te, False, dY_te, True, None, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
            te_wg = bench(te_wgrad_w2)
            tk_wg = bench(lambda: tk_wgrad_gemm(h_tk, dY_tk))
            ops.append(("GEMM wgrad(h,dY)", te_wg, tk_wg))

            # --- Op B4: silu_deriv + quantize ---
            def te_silu_deriv_q():
                dh1_raw, dh3_, _, _ = te_fused.fused_silu_deriv_dual_mul_strided_bf16(
                    dh_bf16.contiguous(), h13_bf16)
                _fast_quantize(dh1_raw, tk_swizzle=False)
                _fast_quantize(dh3_, tk_swizzle=False)
            te_sdq = bench(te_silu_deriv_q)

            _use_fused = hasattr(tk_q, 'tk_silu_deriv_quantize_for_gemm')
            if _use_fused:
                def tk_silu_deriv_q():
                    tk_q.tk_silu_deriv_quantize_for_gemm(dh_bf16.contiguous(), h13_bf16, H)
                tk_sdq = bench(tk_silu_deriv_q)
            else:
                def tk_silu_deriv_q():
                    dh1_raw, dh3_, _, _ = te_fused.fused_silu_deriv_dual_mul_strided_bf16(
                        dh_bf16.contiguous(), h13_bf16)
                    _fast_quantize(dh1_raw, tk_swizzle=True)
                    _fast_quantize(dh3_, tk_swizzle=True)
                tk_sdq = bench(tk_silu_deriv_q)
            label = "fused_silu_deriv+quant" if _use_fused else "silu_deriv+2×quant"
            ops.append((label, te_sdq, tk_sdq))

            # --- Op B5: [W1,W3] batched dgrad ---
            dh1_tk = _fast_quantize(dh1_bf16, tk_swizzle=True)
            dh3_tk = _fast_quantize(dh3_bf16, tk_swizzle=True)
            w1_tk = _fast_quantize(w1_bf16, tk_swizzle=True)
            w3_tk = _fast_quantize(w3_bf16, tk_swizzle=True)

            dh1_te = _fast_quantize(dh1_bf16, tk_swizzle=False)
            dh3_te = _fast_quantize(dh3_bf16, tk_swizzle=False)
            w1_te = _fast_quantize(w1_bf16, tk_swizzle=False)
            w3_te = _fast_quantize(w3_bf16, tk_swizzle=False)

            def te_batched_dgrad():
                d1 = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')
                tex.generic_gemm(w1_te, False, dh1_te, False, d1, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
                d3 = torch.empty(M, K, dtype=torch.bfloat16, device='cuda')
                tex.generic_gemm(w3_te, False, dh3_te, False, d3, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
            te_bd = bench(te_batched_dgrad)

            dh1_fp4, dh1_sc, dh1_sg = dh1_tk._tk_row
            dh3_fp4, dh3_sc, dh3_sg = dh3_tk._tk_row
            w1_fp4_c, w1_sc_c, w1_sg_c = w1_tk._tk_col
            w3_fp4_c, w3_sc_c, w3_sg_c = w3_tk._tk_col
            D_list = [torch.empty(M, K, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
            def tk_batched_dgrad():
                tk.nvfp4_batched_gemm(
                    [dh1_fp4, dh3_fp4], [dh1_sc, dh3_sc],
                    [dh1_sg.to(torch.float32), dh3_sg.to(torch.float32)],
                    [w1_fp4_c.view(torch.float4_e2m1fn_x2), w3_fp4_c.view(torch.float4_e2m1fn_x2)],
                    [w1_sc_c.view(torch.float8_e4m3fn), w3_sc_c.view(torch.float8_e4m3fn)],
                    [w1_sg_c.to(torch.float32), w3_sg_c.to(torch.float32)],
                    D_list)
            tk_bd = bench(tk_batched_dgrad)
            ops.append(("GEMM batched dgrad", te_bd, tk_bd))

            # --- Op B6: [W1,W3] grouped wgrad ---
            x_tk = _fast_quantize(x_bf16, tk_swizzle=True)
            x_te = _fast_quantize(x_bf16, tk_swizzle=False)
            N_dims = [H, H]

            dh1_fp4c, dh1_scc, dh1_sg_c2 = dh1_tk._tk_col
            dh3_fp4c, dh3_scc, dh3_sg_c2 = dh3_tk._tk_col
            dy_col_quant = ([dh1_fp4c, dh3_fp4c], [dh1_scc, dh3_scc],
                            torch.stack([dh1_sg_c2, dh3_sg_c2]))
            def tk_grouped_wg():
                tk_grouped_wgrad_gemm(dy_col_quant, x_tk, N_dims)

            def te_grouped_wg():
                tex.generic_gemm(dh1_te, False, x_te, True, None, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
                tex.generic_gemm(dh3_te, False, x_te, True, None, None,
                                 TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                                 False, None, False, ws, ws.shape[0], False, False)
            te_gw = bench(te_grouped_wg)
            tk_gw = bench(tk_grouped_wg)
            ops.append(("GEMM grouped wgrad", te_gw, tk_gw))

            # --- Print backward summary ---
            total_te = sum(o[1] for o in ops)
            total_tk = sum(o[2] for o in ops)
            for name, te_t, tk_t in ops:
                delta = te_t - tk_t
                sp = te_t / tk_t if tk_t > 0 else 0
                print(f"  {M:>7} | {name:>25} | {te_t:>7.3f}ms | {tk_t:>7.3f}ms | {delta:>+7.3f} | {sp:>5.2f}x", flush=True)
            sp_tot = total_te / total_tk if total_tk > 0 else 0
            delta_tot = total_te - total_tk
            print(f"  {M:>7} | {'TOTAL BWD':>25} | {total_te:>7.3f}ms | {total_tk:>7.3f}ms | {delta_tot:>+7.3f} | {sp_tot:>5.2f}x", flush=True)
            print("  " + "-" * (len(hdr) - 2), flush=True)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  {M:>7} | FAILED: {str(e)[:60]}", flush=True)
            traceback.print_exc()

    print("\n✅ Per-op FFN profile complete!", flush=True)


if __name__ == "__main__":
    main()
