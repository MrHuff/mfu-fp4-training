"""
FP4 Linear & Converter Tests
Uses torchrun (needs NCCL + 1 GPU): torchrun --nproc_per_node=1 tests/quantization/test_fp4_linear.py
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
DEVICE = torch.device("cuda")

print("=" * 60)
print("FP4 Linear & Converter Tests (TE-fused)")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# 1. TELinearFP4: forward/backward parity with nn.Linear
# ═══════════════════════════════════════════════════════════════════════
def test_te_linear_fp4():
    from low_bits_training.quantization.fused_te_linear import TELinearFP4

    K, N, M = 512, 256, 64
    ref = nn.Linear(K, N, bias=False, device=DEVICE, dtype=torch.bfloat16)
    fp4 = TELinearFP4(K, N, bias=False, device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        fp4.weight.copy_(ref.weight)

    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y_fp4 = fp4(x)
    y_ref = ref(x_ref)

    fwd_err = (y_fp4 - y_ref).abs().mean() / y_ref.abs().mean()
    print(f"TELinearFP4 fwd relative error: {fwd_err:.4f}")
    assert fwd_err < 0.5, f"Forward error too large: {fwd_err}"

    y_fp4.sum().backward()
    y_ref.sum().backward()
    grad_err = (x.grad - x_ref.grad).abs().mean() / x_ref.grad.abs().mean()
    print(f"TELinearFP4 grad relative error: {grad_err:.4f}")
    assert grad_err < 0.5, f"Backward error too large: {grad_err}"

    print("✓ TELinearFP4 PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# 2. NormTELinearFP4: fused norm+silu+quant → GEMM (forward + backward)
# ═══════════════════════════════════════════════════════════════════════
def test_norm_te_linear_fp4():
    from low_bits_training.quantization.fused_te_linear import NormTELinearFP4

    M, K, N = 64, 512, 256
    layer = NormTELinearFP4(
        K, N, bias=False, norm_eps=1e-5,
        use_silu=True, device=DEVICE, dtype=torch.bfloat16,
    )
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)
    print(f"NormTELinearFP4 output shape: {y.shape}, mean: {y.mean():.4f}, std: {y.std():.4f}")

    assert y.shape == (M, N), f"Wrong output shape: {y.shape}"
    assert not y.isnan().any(), "Output has NaN"

    y.sum().backward()
    print(f"NormTELinearFP4 input grad norm: {x.grad.norm():.4f}")
    print(f"NormTELinearFP4 weight grad norm: {layer.weight.grad.norm():.4f}")
    assert not x.grad.isnan().any(), "Input grad has NaN"
    assert not layer.weight.grad.isnan().any(), "Weight grad has NaN"

    print("✓ NormTELinearFP4 PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# 3. NormTELinearFP4: 3D input (B, S, K)
# ═══════════════════════════════════════════════════════════════════════
def test_norm_te_linear_3d():
    from low_bits_training.quantization.fused_te_linear import NormTELinearFP4

    B, S, K, N = 2, 32, 512, 256
    layer = NormTELinearFP4(
        K, N, bias=False, norm_eps=1e-5,
        use_silu=True, device=DEVICE, dtype=torch.bfloat16,
    )
    x = torch.randn(B, S, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)

    assert y.shape == (B, S, N), f"Wrong 3D output shape: {y.shape}"
    assert not y.isnan().any(), "3D output has NaN"
    y.sum().backward()
    assert not x.grad.isnan().any(), "3D input grad has NaN"

    print(f"NormTELinearFP4 3D output shape: {y.shape}")
    print("✓ NormTELinearFP4 3D PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# 4. NormTELinearFP4: identity activation (no SiLU) — for w3
# ═══════════════════════════════════════════════════════════════════════
def test_norm_te_linear_identity():
    from low_bits_training.quantization.fused_te_linear import NormTELinearFP4

    M, K, N = 64, 512, 256
    layer = NormTELinearFP4(
        K, N, bias=False, norm_eps=1e-5,
        use_silu=False, device=DEVICE, dtype=torch.bfloat16,
    )
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)

    assert y.shape == (M, N), f"Wrong output shape: {y.shape}"
    assert not y.isnan().any(), "Identity output has NaN"
    y.sum().backward()
    assert not x.grad.isnan().any(), "Identity input grad has NaN"
    assert not layer.weight.grad.isnan().any(), "Identity weight grad has NaN"

    print(f"NormTELinearFP4 identity output shape: {y.shape}")
    print("✓ NormTELinearFP4 identity PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# 5. FusedFeedForwardFP4: forward + backward
# ═══════════════════════════════════════════════════════════════════════
def test_fused_feed_forward_fp4():
    from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4

    B, S = 4, 32
    dim, hidden = 512, 1024
    ffn = FusedFeedForwardFP4(
        dim, hidden, norm_eps=1e-5, bias=False,
        device=DEVICE, dtype=torch.bfloat16,
    )
    x = torch.randn(B, S, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    y = ffn(x)

    assert y.shape == (B, S, dim), f"Wrong FFN shape: {y.shape}"
    assert not y.isnan().any(), "FFN output has NaN"

    y.sum().backward()
    assert not x.grad.isnan().any(), "FFN input grad has NaN"

    print(f"FusedFeedForwardFP4 output shape: {y.shape}, norm: {y.norm():.2f}")
    print(f"FusedFeedForwardFP4 grad norm: {x.grad.norm():.2f}")
    print("✓ FusedFeedForwardFP4 PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# 6. FusedFeedForwardFP4.from_unfused
# ═══════════════════════════════════════════════════════════════════════
def test_fused_feed_forward_from_unfused():
    from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4

    dim, hidden = 512, 1024

    # Simulate unfused layers
    class FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden, bias=False, device=DEVICE, dtype=torch.bfloat16)
            self.w2 = nn.Linear(hidden, dim, bias=False, device=DEVICE, dtype=torch.bfloat16)
            self.w3 = nn.Linear(dim, hidden, bias=False, device=DEVICE, dtype=torch.bfloat16)

    norm = nn.RMSNorm(dim, device=DEVICE, dtype=torch.bfloat16)
    ffn = FF()

    fused = FusedFeedForwardFP4.from_unfused(ffn, norm)
    assert isinstance(fused, FusedFeedForwardFP4)

    # Verify weight copying
    with torch.no_grad():
        assert torch.equal(fused.w1_weight, ffn.w1.weight)
        assert torch.equal(fused.w3_weight, ffn.w3.weight)
        assert torch.equal(fused.w2_weight, ffn.w2.weight)
        assert torch.equal(fused.norm_weight, norm.weight)

    print("✓ FusedFeedForwardFP4.from_unfused PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# 7. Converter wiring: correct layer types after conversion
# ═══════════════════════════════════════════════════════════════════════
def test_converter():
    from low_bits_training.quantization.fused_te_linear import (
        TELinearFP4, FusedFeedForwardFP4
    )
    from low_bits_training.quantization.mxfp_custom_te_fp4 import BoundRecipeLinear
    from low_bits_training.quantization.fp4_converter import FP4Converter, _NormIdentity

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)
            self.wo = nn.Linear(d, d, bias=False)
        def forward(self, x): return x

    class FF(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.w1 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(h, d, bias=False)
            self.w3 = nn.Linear(d, h, bias=False)
        def forward(self, x): return x

    class Blk(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.attention = Attn(d)
            self.feed_forward = FF(d, h)
            self.attention_norm = nn.RMSNorm(d)
            self.ffn_norm = nn.RMSNorm(d)
        def forward(self, x): return x

    model = nn.ModuleDict({
        'layers': nn.ModuleDict({'0': Blk(512, 1024)})
    }).to(DEVICE, torch.bfloat16)

    class DC: pass
    FP4Converter(DC(), None).convert(model)
    block = model.layers['0']

    # Type checks
    assert isinstance(block.attention.wq, BoundRecipeLinear), \
        f"wq: {type(block.attention.wq).__name__}"
    assert isinstance(block.attention.wk, BoundRecipeLinear), \
        f"wk: {type(block.attention.wk).__name__}"
    assert isinstance(block.attention.wv, BoundRecipeLinear), \
        f"wv: {type(block.attention.wv).__name__}"
    assert isinstance(block.attention.wo, BoundRecipeLinear), \
        f"wo: {type(block.attention.wo).__name__}"
    assert isinstance(block.feed_forward, FusedFeedForwardFP4), \
        f"ffn: {type(block.feed_forward).__name__}"
    assert isinstance(block.ffn_norm, _NormIdentity), \
        f"ffn_norm: {type(block.ffn_norm).__name__}"
    assert isinstance(block.attention_norm, nn.RMSNorm), \
        f"Expected nn.RMSNorm for attention_norm, got {type(block.attention_norm).__name__}"

    # Count layers
    attn_only = sum(1 for m in model.modules()
                    if isinstance(m, BoundRecipeLinear) and not isinstance(m, TELinearFP4))
    te_fp4_count = sum(1 for m in model.modules() if isinstance(m, TELinearFP4))
    ffn_count = sum(1 for m in model.modules() if isinstance(m, FusedFeedForwardFP4))
    print(f"Converter: {attn_only} BoundRecipeLinear (attn), {te_fp4_count} TELinearFP4, {ffn_count} FusedFeedForwardFP4")
    assert attn_only == 4, f"Expected 4 BoundRecipeLinear (attn), got {attn_only}"
    assert te_fp4_count == 0, f"Expected 0 TELinearFP4 (FFN now uses raw params), got {te_fp4_count}"
    assert ffn_count == 1, f"Expected 1 FusedFeedForwardFP4, got {ffn_count}"
    print("✓ Converter PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# 8. Converter forward/backward: full block passes without errors
# ═══════════════════════════════════════════════════════════════════════
def test_converter_forward_backward():
    from low_bits_training.quantization.fused_te_linear import (
        TELinearFP4, FusedFeedForwardFP4
    )
    from low_bits_training.quantization.mxfp_custom_te_fp4 import BoundRecipeLinear
    from low_bits_training.quantization.fp4_converter import FP4Converter

    dim, hidden_dim = 512, 1024

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)
            self.wo = nn.Linear(d, d, bias=False)
        def forward(self, x):
            return self.wo(self.wv(self.wk(self.wq(x))))

    class FF(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.w1 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(h, d, bias=False)
            self.w3 = nn.Linear(d, h, bias=False)
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class Blk(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.attention = Attn(d)
            self.feed_forward = FF(d, h)
            self.attention_norm = nn.RMSNorm(d)
            self.ffn_norm = nn.RMSNorm(d)
        def forward(self, x):
            h = self.attention(self.attention_norm(x)) + x
            out = self.feed_forward(self.ffn_norm(h)) + h
            return out

    model = nn.ModuleDict({
        'layers': nn.ModuleDict({'0': Blk(dim, hidden_dim)})
    }).to(DEVICE, torch.bfloat16)

    class DC: pass
    FP4Converter(DC(), None).convert(model)

    block = model.layers['0']
    x = torch.randn(4, 32, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    # Forward
    y = block(x)
    assert not y.isnan().any(), "Forward pass has NaN"

    # Backward
    y.sum().backward()
    assert not x.grad.isnan().any(), "Backward pass has NaN grad"

    print(f"Converter fwd/bwd: output norm={y.norm():.2f}, grad norm={x.grad.norm():.2f}")
    print("✓ Converter forward/backward PASSED\n")


# ═══════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════
ALL_TESTS = [
    ("TELinearFP4", test_te_linear_fp4),
    ("NormTELinearFP4", test_norm_te_linear_fp4),
    ("NormTELinearFP4 3D", test_norm_te_linear_3d),
    ("NormTELinearFP4 identity", test_norm_te_linear_identity),
    ("FusedFeedForwardFP4", test_fused_feed_forward_fp4),
    ("FusedFeedForwardFP4.from_unfused", test_fused_feed_forward_from_unfused),
    ("Converter", test_converter),
    ("Converter fwd/bwd", test_converter_forward_backward),
]

failed = []
for name, fn in ALL_TESTS:
    try:
        fn()
    except Exception as e:
        failed.append(name)
        print(f"\n✗ FAILED: {name}")
        import traceback
        traceback.print_exc()
        print()

dist.destroy_process_group()

print("=" * 60)
if failed:
    print(f"FAILURES: {', '.join(failed)}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
