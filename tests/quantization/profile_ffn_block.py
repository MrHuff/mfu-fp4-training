"""
Profile a single fused FP4 transformer block (FFN + Attention QKV+WO).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u tests/quantization/profile_ffn_block.py
"""

import os, sys, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4, FusedAttentionFP4

# ---------- Config ----------
DIM = 2048
FFN_DIM = 5632
N_HEADS = 32
HEAD_DIM = DIM // N_HEADS
BATCH, SEQ = 4, 1024
M = BATCH * SEQ

WARMUP = 5
PROFILE_ITERS = 5
TRACE_DIR = "/tmp/profile_fp4_block"

device = torch.device("cuda:0")
torch.cuda.set_device(device)

# ---------- Build layers ----------
print("Building layers...")
ffn = FusedFeedForwardFP4(dim=DIM, hidden_dim=FFN_DIM, norm_eps=1e-5).to(device)
attn = FusedAttentionFP4(dim=DIM, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, norm_eps=1e-5).to(device)

def run_one_iter():
    # FFN fwd+bwd
    x = torch.randn(M, DIM, dtype=torch.bfloat16, device=device, requires_grad=True)
    y = ffn(x)
    y.sum().backward()

    # QKV fwd+bwd
    x2 = torch.randn(M, DIM, dtype=torch.bfloat16, device=device, requires_grad=True)
    xq, xk, xv = attn.forward_qkv(x2)
    (xq.sum() + xk.sum() + xv.sum()).backward()

    # WO fwd only (no autograd wrapper)
    ao = torch.randn(M, DIM, dtype=torch.bfloat16, device=device)
    attn.forward_wo(ao)

# ---------- Warmup ----------
print(f"Warming up ({WARMUP} iters)...")
for _ in range(WARMUP):
    run_one_iter()
torch.cuda.synchronize()
print("Warmup done.")

# ---------- Profile ----------
print(f"Profiling ({PROFILE_ITERS} iters)...")
os.makedirs(TRACE_DIR, exist_ok=True)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(PROFILE_ITERS):
        run_one_iter()
        torch.cuda.synchronize()

# ---------- Print summary ----------
print("\n" + "=" * 120)
print("CUDA Kernel Summary (sorted by self CUDA time)")
print("=" * 120)
print(prof.key_averages().table(
    sort_by="self_device_time_total",
    row_limit=50,
    top_level_events_only=False,
))

print(f"\nTrace saved to: {TRACE_DIR}")
