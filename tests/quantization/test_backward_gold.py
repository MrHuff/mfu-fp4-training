"""
Definitive backward parity test: compare our _NormLinearFP4Function backward
against a fully autograd-traced path using BF16 matmul (gold standard).

Tests grad_input, grad_weight, grad_norm_weight element-by-element.
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


def gold_standard_backward(x, weight, norm_weight, bias, epsilon, use_silu):
    """
    Pure BF16 autograd backward through RMSNorm+SiLU+Linear.
    No quantization — this is the "ideal" gradient.
    """
    x = x.clone().detach().requires_grad_(True)
    weight = weight.clone().detach().requires_grad_(True)
    norm_weight = norm_weight.clone().detach().requires_grad_(True)

    # RMSNorm
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    normed = (x.float() * inv_rms * norm_weight.float()).to(torch.bfloat16)

    # SiLU if requested
    if use_silu:
        h = F.silu(normed)
    else:
        h = normed

    # Linear
    y = h @ weight.t()
    if bias is not None:
        y = y + bias

    # Backward
    loss = y.sum()
    loss.backward()

    return y, x.grad, weight.grad, norm_weight.grad


def quantized_backward(x, weight, norm_weight, bias, epsilon, use_silu):
    """
    Our _NormLinearFP4Function forward + backward.
    """
    x = x.clone().detach().requires_grad_(True)
    weight = weight.clone().detach().requires_grad_(True)
    norm_weight = norm_weight.clone().detach().requires_grad_(True)

    te_dtype = tex.DType.kFloat4E2M1
    inp_q = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )
    wt_q = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )
    go_q = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )
    workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=x.device)

    y = _NormLinearFP4Function.apply(
        x, weight, norm_weight, bias,
        epsilon, use_silu,
        inp_q, wt_q, go_q, workspace,
    )

    loss = y.sum()
    loss.backward()

    return y, x.grad, weight.grad, norm_weight.grad


def test_backward_parity(M=128, K=512, N=256, use_silu=True, label=""):
    torch.manual_seed(42)
    device = 'cuda'

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02
    norm_weight = torch.ones(K, dtype=torch.bfloat16, device=device)
    bias = None
    epsilon = 1e-5

    print(f"\n{'='*60}")
    print(f"Test: {label} (M={M}, K={K}, N={N}, silu={use_silu})")
    print(f"{'='*60}")

    y_gold, gx_gold, gw_gold, gnw_gold = gold_standard_backward(
        x, weight, norm_weight, bias, epsilon, use_silu
    )
    y_ours, gx_ours, gw_ours, gnw_ours = quantized_backward(
        x, weight, norm_weight, bias, epsilon, use_silu
    )

    def rel_err(a, b, name):
        err = (a.float() - b.float()).norm() / (b.float().norm() + 1e-8)
        cos = F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
        max_abs = (a.float() - b.float()).abs().max()
        print(f"  {name:20s}: rel_err={err:.4f}  cos={cos:.6f}  max_diff={max_abs:.4f}  "
              f"norm_ours={a.norm():.2f}  norm_gold={b.norm():.2f}")
        return err.item()

    rel_err(y_ours, y_gold, "forward output")
    gx_err = rel_err(gx_ours, gx_gold, "grad_input")
    gw_err = rel_err(gw_ours, gw_gold, "grad_weight")
    gnw_err = rel_err(gnw_ours, gnw_gold, "grad_norm_weight")

    # Check if any gradient is zero
    print(f"\n  grad_input all-zero: {(gx_ours == 0).all().item()}")
    print(f"  grad_weight all-zero: {(gw_ours == 0).all().item()}")
    print(f"  grad_norm_weight all-zero: {(gnw_ours == 0).all().item()}")

    # Check specific patterns
    print(f"\n  grad_input sign match: {((gx_ours > 0) == (gx_gold > 0)).float().mean():.4f}")
    print(f"  grad_weight sign match: {((gw_ours > 0) == (gw_gold > 0)).float().mean():.4f}")

    return gx_err, gw_err, gnw_err


if __name__ == '__main__':
    for silu in [True, False]:
        test_backward_parity(128, 512, 256, use_silu=silu, label=f"silu={silu}")
        test_backward_parity(128, 2048, 1024, use_silu=silu, label=f"Large silu={silu}")

    print("\n✓ All tests complete")
