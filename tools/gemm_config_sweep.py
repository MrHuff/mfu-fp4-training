#!/usr/bin/env python3
"""
Reusable GEMM config sweeper for ThunderKittens NVFP4 kernels.

Usage:
    # Sweep a specific shape:
    python gemm_config_sweep.py --M 32768 --K 4096 --N 6144

    # Sweep a model preset:
    python gemm_config_sweep.py --preset llama-8b-tp1
    python gemm_config_sweep.py --preset llama-70b-tp8

    # Sweep all layers for a model:
    python gemm_config_sweep.py --preset llama-8b-tp1 --all-layers

    # Custom M values:
    python gemm_config_sweep.py --preset llama-8b-tp1 --M 8192,32768,65536
"""
import os, argparse
os.environ['USE_TK_GEMM'] = '1'
os.environ['USE_TK_QUANT'] = '1'
os.environ['NVTE_NVFP4_DISABLE_RHT'] = '1'
os.environ['NVTE_NVFP4_DISABLE_2D_QUANTIZATION'] = '1'
os.environ['NVTE_NVFP4_ENCODE_CENTRIC'] = '0'
os.environ['NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING'] = '1'

import torch
import transformer_engine.pytorch
from low_bits_training.quantization.tk_gemm import _get_tk
from low_bits_training.quantization.fused_te_linear import _fast_quantize

tk = _get_tk()

# ── Model Presets ──────────────────────────────────────────────
# Each preset defines the per-GPU GEMM shapes for all linear layers.
# Format: {layer_name: (K, N, description)}
#
# TP splits:
#   Column-parallel (QKV, Gate/Up): K=H (full), N=N_total/TP
#   Row-parallel (Wo, Down):        K=K_in/TP,  N=H (full)

def make_preset(H, n_heads, n_kv_heads, head_dim, ffn, tp):
    """Generate per-GPU GEMM shapes for all layers."""
    N_q = n_heads * head_dim
    N_k = n_kv_heads * head_dim
    N_v = N_k
    return {
        'qkv': (H, (N_q + N_k + N_v) // tp,
                f'QKV col-par: K={H}, N=({N_q}+{N_k}+{N_v})/{tp}'),
        'wo':  (N_q // tp, H,
                f'Wo  row-par: K={N_q}/{tp}, N={H}'),
        'gate_up': (H, 2 * ffn // tp,
                f'W1W3 col-par: K={H}, N=2×{ffn}/{tp}'),
        'down': (ffn // tp, H,
                f'W2  row-par: K={ffn}/{tp}, N={H}'),
    }

PRESETS = {
    # ── Small models (TP=1) ──
    'llama-1b-tp1':   make_preset(2048, 16, 16, 128, 5632, 1),
    'llama-3b-tp1':   make_preset(3072, 24, 8,  128, 8192, 1),
    'llama-8b-tp1':   make_preset(4096, 32, 8,  128, 14336, 1),

    # ── Medium models (TP=2) ──
    'llama-8b-tp2':   make_preset(4096, 32, 8,  128, 14336, 2),

    # ── Large models (TP=4-8) ──
    'llama-70b-tp4':  make_preset(8192, 64, 8, 128, 28672, 4),
    'llama-70b-tp8':  make_preset(8192, 64, 8, 128, 28672, 8),
    'llama-405b-tp8': make_preset(16384, 128, 8, 128, 53248, 8),
    'llama-405b-tp16': make_preset(16384, 128, 8, 128, 53248, 16),

    # ── Our benchmark ──
    'bench-qkv':      {'qkv': (2048, 6144, 'Current bench: K=2048, N=6144')},
}

CONFIGS = {
    0:  "<256, 4, 16,  1, 2, false>  Nb=1024 SG=1",
    1:  "<256, 4, 16,  4, 2, false>  Nb=1024 SG=4",
    2:  "<256, 4, 16, 12, 2, false>  Nb=1024 SG=12",
    3:  "<256, 5,  8,  4, 2, true >  Nb=512  SG=4  ovlp",
    4:  "<256, 5,  8, 12, 2, true >  Nb=512  SG=12 ovlp",
    5:  "<256, 5,  8,  4, 2, false>  Nb=512  SG=4",
    6:  "<256, 4,  8, 12, 2, false>  Nb=512  SG=12",
    7:  "<128, 5,  4, 12, 2, true >  Nb=256  SG=12 ovlp",
    8:  "<128, 4,  4, 12, 2, false>  Nb=256  SG=12",
    9:  "<128, 5,  4,  4, 2, true >  Nb=256  SG=4  ovlp",
    10: "<256, 5, 16,  4, 2, true >  Nb=1024 SG=4  ovlp",
    11: "<256, 5, 16, 12, 2, true >  Nb=1024 SG=12 ovlp",
}

def bench_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def sweep_shape(M, K, N, label=""):
    """Sweep all 12 configs for D(M,N) = A(M,K/2) × B(N,K/2)^T."""
    # Check N is multiple of required tile sizes
    # Nb=256→N%256==0, Nb=512→N%512==0, Nb=1024→N%1024==0
    # M must be multiple of 256 (Mb=256)
    if M % 256 != 0:
        print(f"  SKIP: M={M} not multiple of 256")
        return {}
    if K % 256 != 0 or K < 256:
        print(f"  SKIP: K={K} not multiple of 256")
        return {}

    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    a_q = _fast_quantize(A, tk_swizzle=True)
    b_q = _fast_quantize(B, tk_swizzle=True)
    a_fp4, a_sc, a_sg = a_q._tk_row
    b_fp4, b_sc, b_sg = b_q._tk_row
    D = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    results = {}
    for cid in range(12):
        # Check N alignment for this config's Nb
        nb_map = {0:1024,1:1024,2:1024,3:512,4:512,5:512,6:512,
                  7:256,8:256,9:256,10:1024,11:1024}
        nb = nb_map[cid]
        if N % nb != 0:
            results[cid] = "N%Nb≠0"
            continue
        try:
            tk.nvfp4_gemm_config(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, D, cid)
            torch.cuda.synchronize()
            t = bench_ms(lambda cid=cid: tk.nvfp4_gemm_config(
                a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, D, cid))
            results[cid] = t
        except Exception as e:
            results[cid] = f"ERR:{str(e)[:30]}"

    # Auto-detect current config
    if K <= 2048 and N <= 4096:
        current = 5  # config<256,5,8,4,2,false>
    elif K <= 2048:
        current = 5  # was 1, now 5 after sweep
    else:
        current = 4  # was 6, now 4 after sweep

    valid = [v for v in results.values() if isinstance(v, float)]
    if not valid:
        print(f"  ALL FAILED for M={M}, K={K}, N={N}")
        return results

    best = min(valid)
    print(f"  {'ID':>4s}  {'Time':>8s}  {'vs Best':>8s}  Description")
    print(f"  {'-'*65}")
    for cid in sorted(results.keys()):
        t = results[cid]
        if isinstance(t, float):
            marker = " <<<" if abs(t - best) < 0.001 else ""
            curr = " *" if cid == current else ""
            print(f"  {cid:4d}  {t:7.3f}ms  {t/best:7.2f}×  {CONFIGS[cid]}{curr}{marker}")
        else:
            print(f"  {cid:4d}  {t:40s}  {CONFIGS[cid]}")

    del A, B, D, a_q, b_q
    torch.cuda.empty_cache()
    return results

def main():
    parser = argparse.ArgumentParser(description='GEMM Config Sweeper')
    parser.add_argument('--M', type=str, default='4096,16384,65536',
                       help='Comma-separated M values (batch×seqlen)')
    parser.add_argument('--K', type=int, help='Reduction dim (hidden size)')
    parser.add_argument('--N', type=int, help='Output dim')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                       help='Model preset')
    parser.add_argument('--all-layers', action='store_true',
                       help='Sweep all layers (default: QKV only)')
    parser.add_argument('--list-presets', action='store_true')
    args = parser.parse_args()

    if args.list_presets:
        for name, layers in PRESETS.items():
            print(f"\n{name}:")
            for layer, (k, n, desc) in layers.items():
                print(f"  {layer:10s}: {desc}")
        return

    M_vals = [int(x) for x in args.M.split(',')]

    if args.preset:
        layers = PRESETS[args.preset]
        if not args.all_layers:
            layers = {k: v for k, v in layers.items() if k == 'qkv'}
    elif args.K and args.N:
        layers = {'custom': (args.K, args.N, f'Custom K={args.K}, N={args.N}')}
    else:
        parser.error('Specify --preset or both --K and --N')

    print(f"\n{'='*70}")
    print(f"GEMM Config Sweep")
    print(f"{'='*70}")

    for layer_name, (K, N, desc) in layers.items():
        for M in M_vals:
            print(f"\n{'─'*70}")
            print(f"  {layer_name.upper()}: {desc}")
            print(f"  M={M} (e.g. B={M//2048}×S=2048)")
            print(f"  GEMM: D({M}, {N}) = A({M}, {K}) × B({N}, {K})^T")
            print(f"{'─'*70}")
            sweep_shape(M, K, N, label=layer_name)

    print("\nDONE")

if __name__ == '__main__':
    main()
