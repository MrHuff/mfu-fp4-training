#!/usr/bin/env python3
"""Simple gap analysis: where does time go in TK backward vs TE?"""
import os, sys
os.environ['CYPARI_NO_SIGNALS'] = '1'
os.environ.setdefault('NVTE_NVFP4_DISABLE_RHT', '1')
os.environ.setdefault('NVTE_NVFP4_DISABLE_2D_QUANTIZATION', '1')
os.environ.setdefault('NVTE_NVFP4_ENCODE_CENTRIC', '0')
os.environ.setdefault('NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING', '1')
os.environ.setdefault('NVTE_CUSTOM_QUANT', '0')
os.environ.setdefault('USE_TK_QUANT', '1')
os.environ.setdefault('USE_TK_GEMM', '1')
os.environ.setdefault('FUSED_TE_QUANT', '0')
os.environ['TK_SPLIT_ROW_COL_BWD'] = '0'

import torch, signal
signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
import ctypes, importlib.util
for dep in ['/usr/local/cuda/lib64/libnvrtc.so', '/usr/local/cuda/lib64/libcudart.so']:
    if os.path.exists(dep): ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
import transformer_engine as _te
_so_path = os.environ.get(
    'TE_FUSED_RMSNORM_EXTENSION',
    os.path.join(
        os.environ.get(
            'TORCH_EXTENSIONS_DIR',
            os.path.join(os.path.expanduser('~'), '.cache', 'torch_extensions'),
        ),
        'py312_cu130',
        'te_fused_rmsnorm_ext_linear',
        'te_fused_rmsnorm_ext_linear.so',
    ),
)
if os.path.exists(_so_path):
    spec = importlib.util.spec_from_file_location('te_fused_rmsnorm_ext_linear', _so_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import low_bits_training.quantization.fused_te_linear as _fte
if os.path.exists(_so_path): _fte._te_fused_ext = mod
from low_bits_training.quantization.fused_te_linear import (
    FusedAttentionFP4_TE, FusedAttentionFP4_TK, _fast_quantize, _get_te_fused,
)
from low_bits_training.quantization.tk_gemm import tk_split_dgrad, tk_grouped_wgrad_gemm
signal.signal(signal.SIGINT, signal.default_int_handler)

torch.manual_seed(42)
dim = 2048; n_heads = 32; n_kv_heads = 32; head_dim = 64
q_dim = n_heads * head_dim; k_dim = n_kv_heads * head_dim; v_dim = k_dim
N_dims = [q_dim, k_dim, v_dim]

te_attn = FusedAttentionFP4_TE(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                                head_dim=head_dim, device='cuda', dtype=torch.bfloat16)
te_attn.init_weights()
tk_attn = FusedAttentionFP4_TK(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                                head_dim=head_dim, device='cuda', dtype=torch.bfloat16)
tk_attn.init_weights()
with torch.no_grad():
    tk_attn.w_qkv.copy_(te_attn.w_qkv); tk_attn.norm_weight.copy_(te_attn.norm_weight)

def bench(fn, w=8, n=20):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

te_fused = _get_te_fused()
from low_bits_training.quantization.fused_te_linear import _get_tk_quant
tk_q = _get_tk_quant()

print("=" * 90)
print("Gap Analysis: TK backward vs TE")
print("=" * 90)

for M in [4096, 16384, 32768, 65536]:
    x = torch.randn(M, dim, dtype=torch.bfloat16, device='cuda')
    gq = torch.ones(M, q_dim, dtype=torch.bfloat16, device='cuda')
    gk = torch.ones(M, k_dim, dtype=torch.bfloat16, device='cuda')
    gv = torch.ones(M, v_dim, dtype=torch.bfloat16, device='cuda')

    # E2E
    t_te_fwd = bench(lambda: te_attn.forward_qkv(x))
    t_tk_fwd = bench(lambda: tk_attn.forward_qkv(x))
    def te_fb():
        xi = x.detach().requires_grad_(True)
        q, k, v = te_attn.forward_qkv(xi)
        torch.autograd.backward([q,k,v], [gq[:,:q_dim], gk[:,:k_dim], gv[:,:v_dim]])
    def tk_fb():
        xi = x.detach().requires_grad_(True)
        q, k, v = tk_attn.forward_qkv(xi)
        torch.autograd.backward([q,k,v], [gq[:,:q_dim], gk[:,:k_dim], gv[:,:v_dim]])
    t_te_tot = bench(te_fb); t_tk_tot = bench(tk_fb)
    t_te_bwd = t_te_tot - t_te_fwd; t_tk_bwd = t_tk_tot - t_tk_fwd

    # Build w_col same way as backward
    w_bf16 = tk_attn.w_qkv.data
    r = tk_q.tk_group_quantize_for_gemm(w_bf16, N_dims)
    wc_fp4_cols = r[3]; wc_sc_cols = r[4]; sg_cat = r[6]
    col_fp4 = torch.cat([f.contiguous().view(torch.uint8) for f in wc_fp4_cols], dim=1).view(torch.float4_e2m1fn_x2)
    col_sc = torch.cat([s.contiguous().view(torch.uint8) for s in wc_sc_cols], dim=1).view(torch.float8_e4m3fn)
    class _C:
        __slots__ = ('_tk_col',)
        def __init__(s, c): s._tk_col = c
    w_col = _C((col_fp4, col_sc, sg_cat.float()))
    x_nvfp4 = _fast_quantize(x, tk_swizzle=True)
    inv_rms = torch.ones(M, 1, dtype=torch.float32, device='cuda')
    nw = tk_attn.norm_weight.to(torch.bfloat16)

    # Isolated components
    t_dgrad = bench(lambda: tk_split_dgrad((gq, gk, gv), w_col, N_dims))
    _, dyc = tk_split_dgrad((gq, gk, gv), w_col, N_dims)
    t_wgrad = bench(lambda: tk_grouped_wgrad_gemm(dyc, x_nvfp4, N_dims))
    dx, _ = tk_split_dgrad((gq, gk, gv), w_col, N_dims)
    dxc = dx.contiguous()
    t_rms = bench(lambda: te_fused.fused_rmsnorm_backward(dxc, x.data.contiguous(), nw, inv_rms))

    kern = t_dgrad + t_wgrad + t_rms
    gap = t_tk_bwd - kern

    print(f"\n  M={M}:")
    print(f"    TE bwd={t_te_bwd:.3f}ms  TK bwd={t_tk_bwd:.3f}ms  ({t_te_bwd/t_tk_bwd:.2f}x)")
    print(f"    ┌─ split_dgrad (3×quant+gemm+sum3)  {t_dgrad:.3f}ms")
    print(f"    ├─ wgrad_gemm                       {t_wgrad:.3f}ms")
    print(f"    ├─ rmsnorm_bwd                      {t_rms:.3f}ms")
    print(f"    ├─ kernel sum                       {kern:.3f}ms")
    print(f"    └─ OVERHEAD (autograd+python)        {gap:.3f}ms  ({gap/t_tk_bwd*100:.0f}%)")
    torch.cuda.empty_cache()

print(f"\n{'=' * 90}")
