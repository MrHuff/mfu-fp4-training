"""
Compare _NormLinearFP4Function backward against TE standard approach:
  RMSNorm (autograd) → SiLU (autograd) → TEParityLinearTexFunction (TE FP4 GEMM)

This uses the same TE quantizers for everything, so the ONLY difference
is how the backward is computed:
  - TE: PyTorch autograd handles RMSNorm+SiLU backward, TE handles GEMM backward
  - Ours: fused_silu_rmsnorm_backward kernel handles RMSNorm+SiLU, TE handles GEMM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/opt/mfu/EXTERNAL_PATH')

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.constants import TE_DType
from low_bits_training.quantization.fused_te_linear import (
    NormTELinearFP4, _NormLinearFP4Function, _get_te_fused,
)
from low_bits_training.quantization.te_parity_linear_tex import (
    TEParityLinearTexFunction,
)


def make_quantizers(device='cuda'):
    te_dtype = tex.DType.kFloat4E2M1
    kwargs = dict(
        fp4_dtype=te_dtype, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )
    return (NVFP4Quantizer(**kwargs), NVFP4Quantizer(**kwargs), NVFP4Quantizer(**kwargs))


def te_reference_forward_backward(x, weight, norm_weight, epsilon, use_silu):
    """
    TE standard: separate RMSNorm → SiLU → FP4 Linear (each with autograd).
    This is what TE's converter gives you.
    """
    x = x.clone().detach().requires_grad_(True)
    weight = weight.clone().detach().requires_grad_(True)
    norm_weight = norm_weight.clone().detach().requires_grad_(True)
    device = x.device

    # RMSNorm manually (matching torch.nn.RMSNorm autograd behavior)
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    normed = (x.float() * inv_rms * norm_weight.float()).to(torch.bfloat16)

    # SiLU
    if use_silu:
        h = F.silu(normed)
    else:
        h = normed

    # FP4 GEMM via TE (same as TEParityLinearTex)
    inp_q, wt_q, go_q = make_quantizers(device)
    workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=device)

    y = TEParityLinearTexFunction.apply(
        h, weight, None,  # no bias
        inp_q, wt_q, go_q, workspace,
        False,  # use_dequant_gemm
        "E8M0",
    )

    loss = y.sum()
    loss.backward()

    return y, x.grad, weight.grad, norm_weight.grad


def our_fused_forward_backward(x, weight, norm_weight, epsilon, use_silu):
    """
    Our fused: NormLinearFP4Function.
    """
    x = x.clone().detach().requires_grad_(True)
    weight = weight.clone().detach().requires_grad_(True)
    norm_weight = norm_weight.clone().detach().requires_grad_(True)
    device = x.device

    inp_q, wt_q, go_q = make_quantizers(device)
    workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=device)

    y = _NormLinearFP4Function.apply(
        x, weight, norm_weight, None,  # no bias
        epsilon, use_silu,
        inp_q, wt_q, go_q, workspace,
    )

    loss = y.sum()
    loss.backward()

    return y, x.grad, weight.grad, norm_weight.grad


def compare(name, ours, ref):
    rel = (ours.float() - ref.float()).norm() / (ref.float().norm() + 1e-8)
    cos = F.cosine_similarity(ours.float().flatten(), ref.float().flatten(), dim=0)
    max_diff = (ours.float() - ref.float()).abs().max()
    ratio = ours.float().norm() / (ref.float().norm() + 1e-8)
    print(f"  {name:25s}: rel_err={rel:.4f}  cos={cos:.6f}  max_diff={max_diff:.4f}  "
          f"norm_ratio={ratio:.4f}  |ours|={ours.norm():.2f}  |ref|={ref.norm():.2f}")
    return rel.item()


def test_parity(M=128, K=512, N=256, use_silu=True, label=""):
    torch.manual_seed(42)
    device = 'cuda'

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02
    norm_weight = torch.ones(K, dtype=torch.bfloat16, device=device)
    epsilon = 1e-5

    print(f"\n{'='*70}")
    print(f"Test: {label} (M={M}, K={K}, N={N}, silu={use_silu})")
    print(f"{'='*70}")

    y_ref, gx_ref, gw_ref, gnw_ref = te_reference_forward_backward(
        x, weight, norm_weight, epsilon, use_silu
    )
    y_ours, gx_ours, gw_ours, gnw_ours = our_fused_forward_backward(
        x, weight, norm_weight, epsilon, use_silu
    )

    compare("forward output", y_ours, y_ref)
    gx_e = compare("grad_input (dx)", gx_ours, gx_ref)
    gw_e = compare("grad_weight (dW)", gw_ours, gw_ref)
    gnw_e = compare("grad_norm_weight (dgamma)", gnw_ours, gnw_ref)

    # Check if norms are wildly different (indicates scaling bug)
    gx_ratio = gx_ours.norm() / gx_ref.norm()
    gw_ratio = gw_ours.norm() / gw_ref.norm()
    gnw_ratio = gnw_ours.norm() / gnw_ref.norm()

    print(f"\n  ⚠️  grad norms — ours/ref ratios:")
    print(f"       dx:     {gx_ratio:.4f}")
    print(f"       dW:     {gw_ratio:.4f}")
    print(f"       dgamma: {gnw_ratio:.4f}")

    if abs(gnw_ratio - 1.0) > 0.1:
        print(f"\n  🚨 dgamma norm ratio is {gnw_ratio:.4f} — OFF BY {abs(gnw_ratio-1)*100:.1f}%!")

    return gx_e, gw_e, gnw_e


if __name__ == '__main__':
    # Test with realistic transformer dimensions
    for silu in [True, False]:
        test_parity(64, 2048, 2048, use_silu=silu, label=f"2048x2048 silu={silu}")
        test_parity(128, 2048, 5632, use_silu=silu, label=f"FFN silu={silu}")

    print("\n✓ All tests complete")
