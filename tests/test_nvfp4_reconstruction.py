
import torch
import torch.nn as nn
from low_bits_training.quantization.te_parity_linear_tex import TEParityLinearTex
import transformer_engine.pytorch as te
import pytest

import torch.distributed as dist
import os


def test_nvfp4_reconstruction_parity():
    # Setup
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32 # Use Float32 for parity check
    
    M, N, K = 1024, 2048, 2048
    
    # Create inputs
    x = torch.randn(M, K, device=device, dtype=dtype).requires_grad_(True)
    dy = torch.randn(M, N, device=device, dtype=dtype)
    
    # 1. Run Native NVFP4 Kernel
    print("\n--- Running Native Kernel ---")
    model_native = TEParityLinearTex(K, N, bias=False, use_dequant_gemm=False).to(device).to(dtype)
    # Ensure weights are identical
    weight_ref = model_native.weight.clone()
    # bias_ref = model_native.bias.clone()
    
    y_native = model_native(x)
    y_native.backward(dy)
    grad_x_native = x.grad.clone()
    grad_w_native = model_native.weight.grad.clone()
    x.grad = None
    
    # 2. Run PyTorch Reconstruction
    print("\n--- Running PyTorch Reconstruction ---")
    model_reconst = TEParityLinearTex(K, N, bias=False, use_dequant_gemm=True).to(device).to(dtype)
    with torch.no_grad():
        model_reconst.weight.copy_(weight_ref)
        # model_reconst.bias.copy_(bias_ref)
        
    y_reconst = model_reconst(x)
    y_reconst.backward(dy)
    grad_x_reconst = x.grad.clone()
    grad_w_reconst = model_reconst.weight.grad.clone()
    
    # 3. Compare
    print("\n--- Comparison ---")
    
    # Allow some tolerance because CuBLASLt generic vs Torch MM might have minor rounding diffs
    # but theoretically they should be very close if dequantization is identical.
    # Note: TE kernel does accumulate in float32 for scaling factors?
    
    def compare(name, a, b, tol=1e-2): # Relaxed tolerance for BF16/FP4 noise
        mag = a.abs().mean().item()
        print(f"{name}: Mean Mag = {mag:.4f}")
        diff = (a - b).abs()
        max_diff = diff.max().item()
        print(f"{name}: Max Diff = {max_diff:.6f}")
        
        if max_diff > 0:
             uniques = diff.unique().sort()[0]
             print(f"  Smallest 5 diffs: {uniques[:5].tolist()}")
             print(f"  Largest 5 diffs: {uniques[-5:].tolist()}")
             
             # Print values at max diff
             max_idx = diff.argmax()
             # Convert linear index to subscripts if needed, or just print
             print(f"  At Max Diff Index {max_idx}: Native={a.flatten()[max_idx].item():.6f}, Reconst={b.flatten()[max_idx].item():.6f}")

             if max_diff > tol:
                  print(f"  FAIL: {name} mismatch > {tol}")
                  return False
        return True
        
    correct = True
    correct &= compare("Forward Output", y_native, y_reconst)
    correct &= compare("Grad Input", grad_x_native, grad_x_reconst)
    correct &= compare("Grad Weight", grad_w_native, grad_w_reconst)
    
    assert correct, "Reconstruction failed to match Native Kernel"
    print("\nSUCCESS: PyTorch reconstruction matches TE Native Kernel!")

if __name__ == "__main__":
    test_nvfp4_reconstruction_parity()
