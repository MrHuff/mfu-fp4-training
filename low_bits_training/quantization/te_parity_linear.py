import torch
import torch.nn as nn
import sys
from typing import Optional, Any
import os
import importlib.util
from transformer_engine.pytorch.experimental.quantization_custom_format import (
    cast_from_fp4x2,
    high_precision_gemm_ref,
    LOOKUP,
    get_custom_quantizer_factory,
    CustomTensorRef
)

from transformer_engine.pytorch.experimental.quantization import GEMMType

class MockTensorRef:
    """Helper mock object to pass global amax values to qgemm interface"""
    def __init__(self, global_amax_row, global_amax_col):
        self.global_amax_row = global_amax_row
        self.global_amax_col = global_amax_col

class TEParityLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor],
        q_input_data: torch.Tensor,
        q_input_scale: torch.Tensor,
        q_input_amax_row: torch.Tensor,
        q_input_amax_col: torch.Tensor,
        q_weight_data: torch.Tensor,
        q_weight_scale: torch.Tensor,
        q_weight_amax_row: torch.Tensor,
        q_weight_amax_col: torch.Tensor,
        input_quantizer: Any,
        weight_quantizer: Any,
        grad_output_quantizer: Any,
        use_dequant_gemm: bool = False
    ):
        # Safety check moved to static graph construction where possible
        if input.device != weight.device:
             raise RuntimeError(f"Device mismatch! Input: {input.device}, Weight: {weight.device}")
        
        ctx.save_for_backward(input, weight, bias)
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
        ctx.grad_output_quantizer = grad_output_quantizer
        ctx.use_dequant_gemm = use_dequant_gemm

        if use_dequant_gemm:
             # Dequantize Inputs
             # Note: q_input is just data/scale tensors here, need to reconstruct or use quantizer helper if available
             # But we passed quantizers.
             # We need to construct temporary objects to call dequantize, OR implement dequantize helper that takes tensors.
             # CustomQuantizerRef has dequantize(self, dtype). CustomTensorRef has dequantize.
             # The q_input_data passed here are just tensors.
             # We should probably use the objects created in Module.forward if possible, but they are not passed directly?
             # Actually, we can reconstruct a lightweight container.
             
             # Reconstruct Input wrapper
             
             # Helper to make wrapper
             def make_ref(data, scale, gA_row, gA_col, quantizer, shape, dtype):
                 ref = CustomTensorRef(
                     data=data, scale=scale, 
                     global_amax_row=gA_row, global_amax_col=gA_col,
                     dtype=dtype, device=data.device, _quantizer=quantizer, original_shape=shape
                 )
                 return ref

             # Input Wrapper
             # We don't have original shape easily here unless passed, but for 2D GEMM typically (M, K).
             # Assuming 2D for simplicity or inferring.
             # q_input_data is packed or not.
             
             # For CustomTensorRef, dequantize() relies on self.data, self.scale etc.
             
             # Input
             q_in_ref = make_ref(q_input_data, q_input_scale, q_input_amax_row, q_input_amax_col, input_quantizer, input.shape, input.dtype)
             x_dq = q_in_ref.dequantize(dtype=input.dtype)
             
             # Weight
             q_w_ref = make_ref(q_weight_data, q_weight_scale, q_weight_amax_row, q_weight_amax_col, weight_quantizer, weight.shape, weight.dtype)
             w_dq = q_w_ref.dequantize(dtype=input.dtype) # Use input dtype for GEMM computation typically
             
             output = torch.mm(x_dq, w_dq.t())
             
             if bias is not None:
                 output += bias
                 
             return output

        # Wrap amax values in mock objects for qgemm interface
        qresult_x = MockTensorRef(q_input_amax_row, q_input_amax_col)
        qresult_w = MockTensorRef(q_weight_amax_row, q_weight_amax_col)

        # 2. Perform GEMM (Forward) using hoisted quantized inputs
        # The quantization now happens in the Module.forward, exposed to torch.compile
        output = input_quantizer.qgemm(
            qx=q_input_data,
            qw=q_weight_data,
            m_params=None, 
            out_dtype=input.dtype,
            sx=q_input_scale,
            sw=q_weight_scale,
            bias=bias,
            out=None,
            accumulate=False,
            qresult_x=qresult_x,
            qresult_w=qresult_w
        )
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_dequant_gemm = ctx.use_dequant_gemm
        
        # Retrieve the specific quantizers we saved
        input_quantizer = ctx.input_quantizer
        weight_quantizer = ctx.weight_quantizer
        grad_output_quantizer = ctx.grad_output_quantizer
        
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        if use_dequant_gemm:
             # Dequantize paths
             
             # 1. dY
             q_grad_output = grad_output_quantizer.quantize(grad_output)
             dY_dq = q_grad_output.dequantize(dtype=input.dtype)
             
             # 2. W (Dequantized)
             # We need to re-quantize weight to get same error profile? or just use original weight?
             # Task says "Quantize -> Dequantize". So we should quantize then dequantize.
             q_weight = weight_quantizer.quantize(weight)
             w_dq = q_weight.dequantize(dtype=input.dtype)
             
             # dX = dY @ W
             grad_input = torch.mm(dY_dq, w_dq)
             
             # 3. X (Dequantized)
             q_input = input_quantizer.quantize(input)
             x_dq = q_input.dequantize(dtype=input.dtype)
             
             # dW = dY.T @ X
             grad_weight = torch.mm(dY_dq.t(), x_dq)
             
             if bias is not None:
                 grad_bias = grad_output.sum(dim=0)
                 
             return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None

        
        # 1. Quantize Grad Output
        q_grad_output = grad_output_quantizer.quantize(grad_output)
        # Cache scale for debug inspection
        grad_output_quantizer.last_scale = q_grad_output.scale
        # Mock history for debug compatibility
        grad_output_quantizer.amax_history = torch.zeros(1, device=grad_output.device) # placeholder
        
        # --- Gradient w.r.t Input ---
        # dX = dY @ W^T
        
        # Quantize Weights Transposed
        q_weight_T = weight_quantizer.quantize(weight.t().contiguous())
        
        grad_input = grad_output_quantizer.qgemm(
            qx=q_grad_output.data,
            qw=q_weight_T.data,
            m_params=None,
            out_dtype=input.dtype,
            sx=q_grad_output.scale,
            sw=q_weight_T.scale,
            gemm_type=None, # DGRAD
            qresult_x=q_grad_output,
            qresult_w=q_weight_T
        )
        
        # --- Gradient w.r.t Weight ---
        # dW = dY^T @ X (Computed as dW^T = X^T @ dY)
        
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
            qresult_w=q_grad_output_T
        )
        
        grad_weight = grad_weight_T.t()
        
        # Bias gradient
        if bias is not None:
             grad_bias = grad_output.sum(dim=0)
             
        # Return None for the new arguments (q_data/scale/amax args and quantizer args)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None


class TEParityLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mx_config=None, use_dequant_gemm=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_dequant_gemm = use_dequant_gemm
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # --- Config Setup ---
        self.factory_kwargs = {}
        if mx_config:
            def get_cfg(name, default):
                return getattr(mx_config, name) if hasattr(mx_config, name) else default
            
            strategy = get_cfg('strategy', None)
            if strategy:
                encode_centric = (strategy == 'encode')
            else:
                encode_centric = get_cfg('encode_centric', False)

            self.factory_kwargs = {
                "scale_format": get_cfg('scale_type', 'E8M0'),
                "block_size": get_cfg('block_size', 32),
                "use_global_scale": get_cfg('use_global_scale', True),
                "encode_centric": encode_centric,
                "with_rht": get_cfg('use_rht', False),
                "scale_round_mode": get_cfg('scale_round_mode', "TiesToEven"),
                "round_mode": get_cfg('roundMode', "TiesToEven"),
                "with_2d_weights": get_cfg('use_2d_weights', False),
                "eps": get_cfg('eps', 0.0),
                "with_random_sign_mask":get_cfg('with_random_sign_mask', True),
            }
        
        # --- Quantizer Initialization ---
        # We perform the factory lookups ONCE here, so we don't do python lookups during forward/compile.
        factory = get_custom_quantizer_factory(**self.factory_kwargs)
        
        self.input_quantizer = factory("linear_input")
        self.weight_quantizer = factory("linear_weight")
        self.grad_output_quantizer = factory("linear_grad_output")

    @classmethod
    def from_float(cls, mod, config, use_dequant_gemm=False):
        new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, mx_config=config, use_dequant_gemm=use_dequant_gemm)
        new_mod = new_mod.to(mod.weight.device).to(mod.weight.dtype)
        with torch.no_grad():
            new_mod.weight.copy_(mod.weight)
            if mod.bias is not None:
                new_mod.bias.copy_(mod.bias)
        return new_mod

    def forward(self, input):
        # Handle 3D inputs (Batch, Seq, Hidden) by flattening
        is_3d = input.dim() == 3
        if is_3d:
            B, S, H = input.shape
            input_2d = input.view(B * S, H)
        else:
            input_2d = input

        # 1. OPTIMIZATION: HOIST QUANTIZATION OUT OF AUTOGRAD FUNCTION
        # This makes quantization logic visible to torch.compile for fusion
        q_input = self.input_quantizer.quantize(input_2d)
        q_weight = self.weight_quantizer.quantize(self.weight)
        # 2. Pass Hoisted Quantized Tensors to Autograd Function
        output_2d = TEParityLinearFunction.apply(
            input_2d, 
            self.weight, 
            self.bias, 
            q_input.data,
            q_input.scale,
            q_input.global_amax_row,
            q_input.global_amax_col,
            q_weight.data,
            q_weight.scale,
            q_weight.global_amax_row,
            q_weight.global_amax_col,
            self.input_quantizer, 
            self.weight_quantizer, 
            self.grad_output_quantizer,
            self.use_dequant_gemm
        )
        
        if is_3d:
            return output_2d.view(B, S, self.out_features)
        else:
            return output_2d