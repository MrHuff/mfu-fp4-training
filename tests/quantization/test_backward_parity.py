"""
Diagnostic: compare _NormLinearFP4Function backward vs reference (PyTorch autograd).

Tests:
  1. Forward parity (our fused vs manual RMSNorm+SiLU+quant+GEMM)
  2. Backward: check grad_weight specifically (this uses cached x_nvfp4)
  3. Backward: check grad_input (uses fused backward kernel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/opt/mfu/EXTERNAL_PATH')

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor
from transformer_engine.pytorch.constants import TE_DType

from low_bits_training.quantization.fused_te_linear import (
    NormTELinearFP4, _NormLinearFP4Function, _get_te_fused,
)


def test_backward_parity():
    """Compare our fused backward vs a reference backward that re-quantizes."""
    torch.manual_seed(42)
    device = 'cuda'
    M, K, N = 128, 512, 256
    
    # Create layer
    layer = NormTELinearFP4(K, N, bias=False, use_silu=True, device=device)
    
    # Clone weight/norm_weight for reference path
    w_ref = layer.weight.data.clone()
    nw_ref = layer.norm_weight.data.clone()
    
    # Input
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    x_ref = x.data.clone().requires_grad_(True)
    
    # ---- Our fused forward ----
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    our_gw = layer.weight.grad.clone()
    our_gnw = layer.norm_weight.grad.clone()
    our_gx = x.grad.clone()
    
    print(f"Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"Our grad_weight norm: {our_gw.norm().item():.2f}")
    print(f"Our grad_norm_weight norm: {our_gnw.norm().item():.4f}")
    print(f"Our grad_input norm: {our_gx.norm().item():.2f}")
    
    # ---- Reference: re-quant everything in backward ----
    # Do forward manually: norm → SiLU → quant → GEMM
    inp_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )
    wt_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )
    go_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )
    
    # RMSNorm + SiLU manually
    inv_rms = torch.rsqrt(x_ref.float().pow(2).mean(dim=-1) + layer.epsilon)
    normed = (x_ref.float() * inv_rms.unsqueeze(-1) * nw_ref.float()).to(torch.bfloat16)
    activated = F.silu(normed)
    
    # Quantize with TE quantizer (produces both row+col)
    x_nvfp4_ref = inp_q.quantize(activated)
    w_nvfp4_ref = wt_q.quantize(w_ref)
    
    workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=device)
    y_ref = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    tex.generic_gemm(
        w_nvfp4_ref, True, x_nvfp4_ref, False,
        y_ref, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        workspace, workspace.shape[0], False, False,
    )
    
    print(f"\nRef forward output: mean={y_ref.mean().item():.4f}, std={y_ref.std().item():.4f}")
    print(f"Forward parity: relative error = {(y - y_ref).norm() / y_ref.norm():.4f}")
    
    # Now do backward manually with re-quantization
    grad_output = torch.ones_like(y_ref)  # match loss.sum()
    
    dY_nvfp4 = go_q.quantize(grad_output)
    
    # Re-quantize weight for dgrad (same as before)
    w_nvfp4_bwd = wt_q.quantize(w_ref)
    
    # dgrad = dY @ W
    dx_proj = tex.generic_gemm(
        w_nvfp4_bwd, False, dY_nvfp4, False,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        workspace, workspace.shape[0], False, False,
    )[0]
    
    # Re-quantize input for wgrad
    x_nvfp4_bwd = inp_q.quantize(activated)
    
    # wgrad = dY.T @ X
    gw_ref = tex.generic_gemm(
        x_nvfp4_bwd, False, dY_nvfp4, True,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        workspace, workspace.shape[0], False, False,
    )[0]
    
    print(f"\nRef grad_weight norm: {gw_ref.norm().item():.2f}")
    print(f"Grad_weight parity: relative error = {(our_gw - gw_ref).norm() / gw_ref.norm():.4f}")
    
    # Check the x_nvfp4 we built from kernel vs the TE quantizer output
    print(f"\n--- NVFP4Tensor comparison ---")
    
    # Our x_nvfp4 from the fused kernel
    te_fused = _get_te_fused()
    fp4_data, scale_inv, fp4_data_t, scale_inv_t, inv_rms_k, amax, amax_t = \
        te_fused.fused_te_quantize_rmsnorm_silu_2pass_full(
            x.data.detach(), nw_ref.detach(), float(layer.epsilon), False)
    
    x_nvfp4_kernel = NVFP4Tensor(
        shape=(M, K),
        dtype=torch.bfloat16,
        rowwise_data=fp4_data,
        rowwise_scale_inv=scale_inv,
        columnwise_data=fp4_data_t,
        columnwise_scale_inv=scale_inv_t,
        amax_rowwise=amax,
        amax_columnwise=amax_t,
        fp4_dtype=tex.DType.kFloat4E2M1,
        quantizer=inp_q,
        requires_grad=False,
    )
    
    print(f"Kernel rowwise data shape: {fp4_data.shape}, ref: {x_nvfp4_ref._rowwise_data.shape}")
    print(f"Kernel scale_inv shape: {scale_inv.shape}, ref: {x_nvfp4_ref._rowwise_scale_inv.shape}")
    print(f"Kernel amax: {amax.item():.4f}, ref: {x_nvfp4_ref._amax_rowwise.item():.4f}")
    
    if x_nvfp4_ref._columnwise_data is not None:
        print(f"Kernel colwise data shape: {fp4_data_t.shape}, ref: {x_nvfp4_ref._columnwise_data.shape}")
        print(f"Kernel colwise scale shape: {scale_inv_t.shape}, ref: {x_nvfp4_ref._columnwise_scale_inv.shape}")
        print(f"Kernel colwise amax: {amax_t.item():.4f}, ref: {x_nvfp4_ref._amax_columnwise.item():.4f}")
    else:
        print("Ref has NO columnwise data!")
    
    # Compare rowwise FP4 data
    if fp4_data.shape == x_nvfp4_ref._rowwise_data.shape:
        match_pct = (fp4_data == x_nvfp4_ref._rowwise_data).float().mean().item() * 100
        print(f"Rowwise data match: {match_pct:.1f}%")
    
    # Compute wgrad with kernel-produced x_nvfp4
    gw_kernel = tex.generic_gemm(
        x_nvfp4_kernel, False, dY_nvfp4, True,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        workspace, workspace.shape[0], False, False,
    )[0]
    
    print(f"\nKernel-based grad_weight norm: {gw_kernel.norm().item():.2f}")
    print(f"Kernel vs ref wgrad error: {(gw_kernel - gw_ref).norm() / gw_ref.norm():.4f}")
    print(f"Kernel vs our wgrad error: {(gw_kernel - our_gw).norm() / our_gw.norm():.4f}")


if __name__ == '__main__':
    test_backward_parity()
    print("\n✓ Diagnostic complete")
