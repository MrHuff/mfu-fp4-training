#!/usr/bin/env python3
import os, sys, traceback
os.environ['CYPARI_NO_SIGNALS']='1'
os.environ['NVTE_NVFP4_DISABLE_RHT']='1'
os.environ['NVTE_NVFP4_DISABLE_2D_QUANTIZATION']='1'
os.environ['NVTE_NVFP4_ENCODE_CENTRIC']='0'
os.environ['NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING']='1'
os.environ['USE_TK_QUANT']='1'
os.environ['USE_TK_GEMM']='1'
import torch, signal
signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
import ctypes
for d in ['/usr/local/cuda/lib64/libnvrtc.so','/usr/local/cuda/lib64/libcudart.so']:
    if os.path.exists(d): ctypes.CDLL(d, mode=ctypes.RTLD_GLOBAL)
import transformer_engine, importlib.util
so=os.environ.get(
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
if os.path.exists(so):
    spec=importlib.util.spec_from_file_location('te_fused_rmsnorm_ext_linear',so);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import low_bits_training.quantization.fused_te_linear as f
if os.path.exists(so): f._te_fused_ext=m
from low_bits_training.quantization.fused_te_linear import FusedAttentionFP4_TK
signal.signal(signal.SIGINT, signal.default_int_handler)
torch.manual_seed(42)
attn = FusedAttentionFP4_TK(dim=2048, n_heads=32, n_kv_heads=32, head_dim=64, device='cuda', dtype=torch.bfloat16)
attn.init_weights()
print('Module created', flush=True)
for M in [2048, 32768, 65536]:
    try:
        x = torch.randn(M, 2048, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        q, k, v = attn.forward_qkv(x)
        loss = q.sum() + k.sum() + v.sum()
        loss.backward()
        print(f'M={M}: grad_x={x.grad.norm().item():.4f} grad_w={attn.w_qkv.grad.norm().item():.4f}', flush=True)
        x.grad = None; attn.w_qkv.grad = None; attn.norm_weight.grad = None
        torch.cuda.empty_cache()
    except Exception as e:
        traceback.print_exc()
        print(f'M={M}: FAILED {e}', flush=True)
        sys.exit(1)
print('ALL PASSED', flush=True)
