"""
Triton-enabled version of TEParityLinear for testing.

This module provides a drop-in replacement for te_parity_linear.py that uses
Triton kernels for quantization and dequantization.
"""
import torch
import torch.nn as nn
from typing import Optional, Any

# Import the Triton quantizer
import sys
from transformer_engine.pytorch.experimental.quantization_custom_triton import (
    get_triton_quantizer_factory,
    QuantizationMetadata,
    TritonQuantizedTensor,
    TritonQuantizedTensor2D,
)
from transformer_engine.pytorch.experimental.quantization import GEMMType
from transformer_engine.pytorch.experimental.quantization_custom_triton import TritonQuantizedTensor
torch.set_float32_matmul_precision("high")

class TritonTEParityLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor],
        input_quantizer: Any,
        weight_quantizer: Any,
        grad_output_quantizer: Any,
        use_dequant_gemm: bool = False,
        dequant_impl: str = "torch",
        use_fp32_matmul: bool = False,
        scale_dtype: torch.dtype = torch.bfloat16,
        data_dtype: torch.dtype = torch.bfloat16,
        use_caching: bool = True,
        quant_impl: str = "v2",
    ):
        if input.device != weight.device:
            raise RuntimeError(f"Device mismatch! Input: {input.device}, Weight: {weight.device}")
        
        # Determine matmul dtype
        matmul_dtype = torch.float32 if use_fp32_matmul else torch.bfloat16
        
        # ===================================================================
        # OPTIMIZATION: Quantize input and weight with fused row+col ONCE
        # Uses quantize_rowcol_v2 with single Triton kernel via concat+split
        # ===================================================================
        
        # Quantize input with row+col (for forward and dW in backward)
        if quant_impl == "v2":
            q_input = input_quantizer.quantize_rowcol_v2(
                input, scale_dtype=scale_dtype, data_dtype=data_dtype
            )
            q_weight = weight_quantizer.quantize_rowcol_v2(
                weight, scale_dtype=scale_dtype, data_dtype=data_dtype
            )
        else:
            # v1 (Sequential/Original)
            q_input = input_quantizer.quantize_rowcol(input)
            q_weight = weight_quantizer.quantize_rowcol(weight)
        
        # Save to ctx for backward reuse
        ctx.save_for_backward(input, weight, bias)
        
        ctx.use_caching = use_caching
        if use_caching:
            ctx.q_input = q_input
            ctx.q_weight = q_weight
        
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
            
        ctx.grad_output_quantizer = grad_output_quantizer
        ctx.use_dequant_gemm = use_dequant_gemm
        ctx.use_fp32_matmul = use_fp32_matmul
        ctx.scale_dtype = scale_dtype
        ctx.data_dtype = data_dtype
        ctx.matmul_dtype = matmul_dtype
        ctx.quant_impl = quant_impl
        ctx.dequant_impl = dequant_impl

        if use_dequant_gemm:
            # ===== DEQUANT+MM PATH =====
            if dequant_impl == "triton":
                x_dq = q_input.dequantize_row_triton(dtype=matmul_dtype)
                w_dq = q_weight.dequantize_row_triton(dtype=matmul_dtype)
            else:
                x_dq = q_input.dequantize_row(dtype=matmul_dtype)
                w_dq = q_weight.dequantize_row(dtype=matmul_dtype)
            
            # GEMM
            output = torch.mm(x_dq, w_dq.t())
            
            # Alpha scaling
            if q_weight.use_global_scale:
                SCALE_MAX = float(q_weight.scale_max)
                DATA_MAX = float(q_weight.data_max)
                factor = (DATA_MAX * SCALE_MAX) ** 2
                alpha = (q_input.global_amax * q_weight.global_amax / factor).squeeze(-1).to(matmul_dtype)
                output = output * alpha
            
            if bias is not None:
                output = output + bias.to(output.dtype)
            
            return output.to(input.dtype)
        
        # ===== QGEMM PATH =====
        meta_x = QuantizationMetadata(q_input.global_amax, q_input.global_amax)
        meta_w = QuantizationMetadata(q_weight.global_amax, q_weight.global_amax)
        
        output = input_quantizer.qgemm(
            qx=q_input.data_row,
            qw=q_weight.data_row,
            m_params=None, 
            out_dtype=input.dtype,
            sx=q_input.scale_row,
            sw=q_weight.scale_row,
            bias=bias,
            out=None,
            accumulate=False,
            qresult_x=meta_x,
            qresult_w=meta_w,
            accumulate_in_fp32=use_fp32_matmul,
        )
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_dequant_gemm = ctx.use_dequant_gemm
        use_fp32_matmul = ctx.use_fp32_matmul
        
        # Get params from ctx
        grad_output_quantizer = ctx.grad_output_quantizer
        scale_dtype = getattr(ctx, 'scale_dtype', torch.float32)
        data_dtype = getattr(ctx, 'data_dtype', torch.float32)
        matmul_dtype = getattr(ctx, 'matmul_dtype', torch.float32)
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        input_quantizer = ctx.input_quantizer
        weight_quantizer = ctx.weight_quantizer

        if use_dequant_gemm:
            # ==================================================================
            # OPTIMIZATION: Reuse quantized tensors from forward pass
            # Only quantize grad_output using fused v2, reuse q_input/q_weight
            # ==================================================================
            
            # Quantize grad_output with fused v2
            q_grad_output = grad_output_quantizer.quantize_rowcol_v2(
                grad_output, scale_dtype=scale_dtype, data_dtype=data_dtype
            )
            
            # REUSE from forward pass (no re-quantization!) or RECOMPUTE
            if ctx.use_caching:
                q_weight = ctx.q_weight
                q_input = ctx.q_input
            else:
                 # Re-quantize!
                 # Note: This simulates a "memory efficient" but slower path
                 if ctx.quant_impl == "v2":
                     q_input = input_quantizer.quantize_rowcol_v2(
                        input, scale_dtype=scale_dtype, data_dtype=data_dtype
                     )
                     q_weight = weight_quantizer.quantize_rowcol_v2(
                        weight, scale_dtype=scale_dtype, data_dtype=data_dtype
                     )
                 else:
                     q_input = input_quantizer.quantize_rowcol(input)
                     q_weight = weight_quantizer.quantize_rowcol(weight)
            
            # ==================================================================
            # dX = dY @ W
            # ==================================================================
            # ==================================================================
            # dX = dY @ W
            # ==================================================================
            if getattr(ctx, 'dequant_impl', 'torch') == "triton":
                w_dq_col_T = q_weight.dequantize_col_as_transpose_triton(dtype=matmul_dtype)  # (K, N)
                w_dq = w_dq_col_T.t()  # (N, K)
                dY_dq = q_grad_output.dequantize_row_triton(dtype=matmul_dtype)  # (M, N)
            else:
                w_dq_col_T = q_weight.dequantize_col_as_transpose(dtype=matmul_dtype)  # (K, N)
                w_dq = w_dq_col_T.t()  # (N, K)
                dY_dq = q_grad_output.dequantize_row(dtype=matmul_dtype)  # (M, N)
            
            grad_input = torch.mm(dY_dq, w_dq)
            
            # Alpha scaling
            if q_weight.use_global_scale:
                SCALE_MAX = float(q_weight.scale_max)
                DATA_MAX = float(q_weight.data_max)
                factor = (DATA_MAX * SCALE_MAX) ** 2
                alpha = (q_grad_output.global_amax * q_weight.global_amax / factor).squeeze(-1).to(matmul_dtype)
                grad_input = grad_input * alpha
            
            grad_input = grad_input.to(input.dtype)
            
            # ==================================================================
            # dW = dY.T @ X
            # ==================================================================
            # ==================================================================
            # dW = dY.T @ X
            # ==================================================================
            if getattr(ctx, 'dequant_impl', 'torch') == "triton":
                dY_T_dq = q_grad_output.dequantize_col_as_transpose_triton(dtype=matmul_dtype)  # (N, M)
                x_dq_col_T = q_input.dequantize_col_as_transpose_triton(dtype=matmul_dtype)  # (K, M)
            else:
                dY_T_dq = q_grad_output.dequantize_col_as_transpose(dtype=matmul_dtype)  # (N, M)
                x_dq_col_T = q_input.dequantize_col_as_transpose(dtype=matmul_dtype)  # (K, M)
            x_dq = x_dq_col_T.t()  # (M, K)
            
            grad_weight = torch.mm(dY_T_dq, x_dq)
            
            # Alpha scaling
            if q_input.use_global_scale:
                SCALE_MAX = float(q_input.scale_max)
                DATA_MAX = float(q_input.data_max)
                factor = (DATA_MAX * SCALE_MAX) ** 2
                alpha = (q_grad_output.global_amax * q_input.global_amax / factor).squeeze(-1).to(matmul_dtype)
                grad_weight = grad_weight * alpha
            
            grad_weight = grad_weight.to(weight.dtype)
            
            if bias is not None:
                grad_bias = grad_output.sum(dim=0)
                
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None

        
        # ==================================================================
        # NON-DEQUANT_GEMM PATH (QGEMM): Use same logic as dequant_gemm path
        # Quantize grad_output with fused v2, reuse q_input/q_weight from forward
        # ==================================================================
        
        # Quantize grad_output with fused v2 (same as use_dequant_gemm=True path)
        if ctx.quant_impl == "v2":
            q_grad_output = grad_output_quantizer.quantize_rowcol_v2(
                grad_output, scale_dtype=scale_dtype, data_dtype=data_dtype
            )
        else:
            q_grad_output = grad_output_quantizer.quantize_rowcol(grad_output)
        
        # REUSE from forward pass (no re-quantization!) or RECOMPUTE
        if ctx.use_caching:
            q_weight = ctx.q_weight
            q_input = ctx.q_input
        else:
            # Re-quantize!
            # Note: This simulates a "memory efficient" but slower path
            if ctx.quant_impl == "v2":
                q_input = input_quantizer.quantize_rowcol_v2(
                    input, scale_dtype=scale_dtype, data_dtype=data_dtype
                )
                q_weight = weight_quantizer.quantize_rowcol_v2(
                    weight, scale_dtype=scale_dtype, data_dtype=data_dtype
                )
            else:
                q_input = input_quantizer.quantize_rowcol(input)
                q_weight = weight_quantizer.quantize_rowcol(weight)

        # 2. --- dX = dY @ W ---
        # Use rowwise dY and columnwise W (transposed as row)
        meta_dY = QuantizationMetadata(q_grad_output.global_amax, q_grad_output.global_amax)
        meta_w = QuantizationMetadata(q_weight.global_amax, q_weight.global_amax)
        
        grad_input = grad_output_quantizer.qgemm(
            qx=q_grad_output.data_row,
            qw=q_weight.data_col,  # Use columnwise quantization (acts as W^T row)
            m_params=None,
            out_dtype=input.dtype,
            sx=q_grad_output.scale_row,
            sw=q_weight.scale_col,
            gemm_type=None,
            qresult_x=meta_dY,
            qresult_w=meta_w,
            accumulate_in_fp32=use_fp32_matmul,
        )
        
        # 3. --- dW = dY^T @ X ---
        # Use columnwise dY (transposed as row) and columnwise X (transposed as row)
        meta_x = QuantizationMetadata(q_input.global_amax, q_input.global_amax)
        
        grad_weight = grad_output_quantizer.qgemm(
            qx=q_grad_output.data_col,  # dY^T as row
            qw=q_input.data_col,        # X^T as row
            m_params=None,
            out_dtype=weight.dtype,
            sx=q_grad_output.scale_col,
            sw=q_input.scale_col,
            gemm_type=GEMMType.WGRAD,
            qresult_x=meta_dY,
            qresult_w=meta_x,
            accumulate_in_fp32=use_fp32_matmul,
        )
        
        if bias is not None:
            # Efficient bias grad
            grad_bias = grad_output.sum(dim=0)
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None


class TritonTEParityLinear(nn.Module):
    """
    Triton-enabled version of TEParityLinear.
    
    Uses Triton kernels for quantization instead of pure PyTorch.
    """
    def __init__(self, in_features, out_features, bias=True, mx_config=None, use_dequant_gemm=False, dequant_impl="torch", use_caching=True, quant_impl="v2"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_dequant_gemm = use_dequant_gemm
        self.dequant_impl = dequant_impl
        self.use_caching = use_caching
        self.quant_impl = quant_impl
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Config setup
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
                "with_random_sign_mask": get_cfg('with_random_sign_mask', True),
                "scale_max": get_cfg('scale_max', 448.0),
                "rht_algo": get_cfg('rht_algo', "fwht"),
            }
            
            self.use_fp32_matmul = get_cfg('use_fp32_matmul', False)
            self.scale_dtype = torch.bfloat16 if get_cfg('use_bf16_scales', False) else torch.float32
            self.data_dtype = torch.bfloat16 if get_cfg('use_bf16_data', False) else torch.float32
        else:
            self.use_fp32_matmul = False
            self.scale_dtype = torch.float32
            self.data_dtype = torch.float32
        
        # Use Triton quantizer factory
        factory = get_triton_quantizer_factory(**self.factory_kwargs)
        
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
        # Handle 3D inputs
        is_3d = input.dim() == 3
        if is_3d:
            B, S, H = input.shape
            input_2d = input.view(B * S, H)
        else:
            input_2d = input

        # Call autograd function
        output_2d = TritonTEParityLinearFunction.apply(
            input_2d, 
            self.weight, 
            self.bias, 
            self.input_quantizer, 
            self.weight_quantizer, 
            self.grad_output_quantizer,
            self.use_dequant_gemm,
            self.dequant_impl,
            self.use_fp32_matmul,
            self.scale_dtype,
            self.data_dtype,
            self.use_caching,
            self.quant_impl,
        )
        
        # Ensure output is in model precision (bf16) for FlashAttention compatibility
        output_2d = output_2d.to(self.weight.dtype)
        
        if is_3d:
            return output_2d.view(B, S, self.out_features)
        else:
            return output_2d
