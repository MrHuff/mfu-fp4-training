"""
NVFP4 Reference Linear Layer for bit parity testing.

This module provides a linear layer that uses the NVFP4QuantizerRef reference
implementation to establish a baseline for bit parity with native TE QGEMM.
"""
import torch
import torch.nn as nn
from typing import Optional, Any

from transformer_engine.pytorch.experimental.quantization_nvfp4 import (
    NVFP4QuantizerRef,
    NVFP4TensorRef,
    cast_from_fp4x2,
    high_precision_gemm_ref,
)
from transformer_engine.pytorch.experimental import utils
from transformer_engine.pytorch.experimental.quantization import GEMMType


class NVFPReferenceLinearFunction(torch.autograd.Function):
    """
    Autograd function for NVFP4 Reference linear layer.
    
    Uses the NVFP4QuantizerRef's qgemm method which mimics the native
    tensor core FP4 GEMM by iterating block-by-block.
    """
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_quantizer: NVFP4QuantizerRef,
        weight_quantizer: NVFP4QuantizerRef,
        grad_output_quantizer: NVFP4QuantizerRef,
    ):
        # Quantize input and weight
        q_input = input_quantizer.quantize(input)
        q_weight = weight_quantizer.quantize(weight)
        
        # Save for backward
        ctx.save_for_backward(input, weight, bias)
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
        ctx.grad_output_quantizer = grad_output_quantizer
        
        # Perform QGEMM using reference implementation
        output = input_quantizer.qgemm(
            qx=q_input.data,
            qw=q_weight.data,
            m_params=None,
            out_dtype=input.dtype,
            sx=q_input.scale,
            sw=q_weight.scale,
            bias=bias,
            out=None,
            accumulate=False,
            gemm_type=GEMMType.FPROP,
            qresult_x=q_input,
            qresult_w=q_weight,
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        input_quantizer = ctx.input_quantizer
        weight_quantizer = ctx.weight_quantizer
        grad_output_quantizer = ctx.grad_output_quantizer
        
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        # 1. Quantize grad_output (dY)
        q_grad_output = grad_output_quantizer.quantize(grad_output)
        
        # 2. Gradient w.r.t Input: dX = dY @ W
        # Re-quantize weight for dgrad (use transposed for correct dimensions)
        q_weight_T = weight_quantizer.quantize(weight.t().contiguous())
        
        grad_input = grad_output_quantizer.qgemm(
            qx=q_grad_output.data,
            qw=q_weight_T.data,
            m_params=None,
            out_dtype=input.dtype,
            sx=q_grad_output.scale,
            sw=q_weight_T.scale,
            gemm_type=GEMMType.FPROP,  # DGRAD uses same path
            qresult_x=q_grad_output,
            qresult_w=q_weight_T,
        )
        
        # 3. Gradient w.r.t Weight: dW = dY^T @ X
        # Transpose inputs for wgrad
        q_input_T = input_quantizer.quantize(input.t().contiguous())
        q_grad_output_T = grad_output_quantizer.quantize(grad_output.t().contiguous())
        
        grad_weight_T = grad_output_quantizer.qgemm(
            qx=q_input_T.data,
            qw=q_grad_output_T.data,
            m_params=None,
            out_dtype=weight.dtype,
            sx=q_input_T.scale,
            sw=q_grad_output_T.scale,
            gemm_type=GEMMType.WGRAD,
            qresult_x=q_input_T,
            qresult_w=q_grad_output_T,
        )
        
        grad_weight = grad_weight_T.t()
        
        # 4. Gradient w.r.t Bias
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)
        
        return grad_input, grad_weight, grad_bias, None, None, None


class NVFPReferenceLinear(nn.Module):
    """
    Linear layer using NVFP4QuantizerRef for bit-exact parity testing.
    
    This layer uses the reference Python implementation of NVFP4 quantization
    and QGEMM, which is known to match the native TE QGEMM output to within
    the tolerance specified in the TE tests (atol=8e-3, rtol=8e-3).
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include a bias term
        mx_config: Configuration object (optional, for compatibility)
    """
    
    def __init__(self, in_features, out_features, bias=True, mx_config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights in bfloat16 (matching native TE)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter('bias', None)
        
        # Create NVFP4 Reference Quantizers
        # These match the default NVFP4 settings used by the native TE
        self.input_quantizer = NVFP4QuantizerRef(
            dtype=utils.Fp4Formats.E2M1,
            rowwise=True,
            columnwise=True,
            pow_2_scales=False,  # NVFP4 uses E4M3 scales
            use_global_scale=False,  # Standard NVFP4
            eps=0.0,
            quant_tile_shape=(1, 16),  # NVFP4 uses 16-element blocks
            with_rht=False,
            with_random_sign_mask=True,
            encode_centric=False,
        )
        
        self.weight_quantizer = NVFP4QuantizerRef(
            dtype=utils.Fp4Formats.E2M1,
            rowwise=True,
            columnwise=True,
            pow_2_scales=False,
            use_global_scale=False,
            eps=0.0,
            quant_tile_shape=(1, 16),
            with_rht=False,
            with_random_sign_mask=True,
            encode_centric=False,
        )
        
        self.grad_output_quantizer = NVFP4QuantizerRef(
            dtype=utils.Fp4Formats.E2M1,
            rowwise=True,
            columnwise=True,
            pow_2_scales=False,
            use_global_scale=False,
            eps=0.0,
            quant_tile_shape=(1, 16),
            with_rht=False,
            with_random_sign_mask=True,
            encode_centric=False,
        )
    
    @classmethod
    def from_float(cls, mod, config=None):
        """Create from a regular nn.Linear module."""
        new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, mx_config=config)
        new_mod = new_mod.to(mod.weight.device).to(mod.weight.dtype)
        with torch.no_grad():
            new_mod.weight.copy_(mod.weight)
            if mod.bias is not None:
                new_mod.bias.copy_(mod.bias)
        return new_mod
    
    def forward(self, input):
        # Handle 3D inputs (Batch, Seq, Hidden)
        is_3d = input.dim() == 3
        if is_3d:
            B, S, H = input.shape
            input_2d = input.view(B * S, H)
        else:
            input_2d = input
        
        output_2d = NVFPReferenceLinearFunction.apply(
            input_2d,
            self.weight,
            self.bias,
            self.input_quantizer,
            self.weight_quantizer,
            self.grad_output_quantizer,
        )
        
        if is_3d:
            return output_2d.view(B, S, self.out_features)
        else:
            return output_2d
