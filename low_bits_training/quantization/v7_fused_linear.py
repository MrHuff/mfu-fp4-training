"""
V7FusedLinear: TE-compatible linear layer with V7 fused RMSNorm+Act+Quant.

Forward pass:
  1. V7 kernel fuses rmsnorm(x, w) → activation → FP4 quant (single kernel)
  2. V7 writes FP4 data + scales directly into pre-allocated NVFP4Tensor buffers
  3. Weight is quantized via TE's NVFP4Quantizer
  4. tex.generic_gemm performs the FP4 GEMM

Backward pass:
  Identical to TEParityLinearTex (no norm/act fusion needed for gradients)

Key optimization: all intermediate tensors (NVFP4Tensor, workspace, scratch) are
pre-allocated in __init__ and reused across forward calls to eliminate Python overhead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor
from transformer_engine.pytorch.constants import TE_DType, NVFP4_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.utils import round_up_to_nearest_multiple

# Lazy-load V7 CUDA extension
_v7_ext = None
def get_v7():
    global _v7_ext
    if _v7_ext is None:
        from torch.utils.cpp_extension import load
        CSRC = '/opt/mfu/EXTERNAL_PATH'
        FL = ['-std=c++20', '-O3', '--expt-relaxed-constexpr',
              '-gencode=arch=compute_100a,code=sm_100a']
        _v7_ext = load(name='fused_te_quant_v7_linear',
            sources=[CSRC+'/fused_te_quant_v7_torch.cpp', CSRC+'/fused_te_quant_v7.cu'],
            extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)
    return _v7_ext


class V7FusedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,          # (M, K) bf16 — pre-norm input
        weight: torch.Tensor,          # (N, K) bf16
        norm_weight: torch.Tensor,     # (K,) bf16 — RMSNorm weight
        bias: Optional[torch.Tensor],
        epsilon: float,
        norm_mode: int,
        act_mode: int,
        # Pre-allocated buffers (no per-call allocation!)
        fp4_data_buf: torch.Tensor,    # (M, K/2) uint8
        scale_inv_buf: torch.Tensor,   # (padded_M, padded_K/16) uint8
        amax_buf: torch.Tensor,        # (1,) float32
        input_quantizer: NVFP4Quantizer,
        weight_quantizer: NVFP4Quantizer,
        grad_output_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
    ):
        M, K = input.shape
        N = weight.shape[0]
        v7 = get_v7()

        # ---- V7 fused: rmsnorm + act + FP4 quant ----
        # V7 writes into pre-allocated fp4_data_buf (M, K/2) and returns scales + global_scale
        fp4_data, scales, global_scale, inv_rms = v7.forward_full(
            input, norm_weight, epsilon,
            norm_mode, act_mode, 0  # decode-centric scaling
        )

        # Copy V7 output into pre-allocated padded scale buffer (one small copy)
        src_cols = K // NVFP4_BLOCK_SCALING_SIZE
        scale_inv_buf[:M, :src_cols] = scales[:M, :src_cols]

        # Update amax: amax = global_scale * (6.0 * 448.0)
        amax_buf.fill_(global_scale.item() * 6.0 * 448.0)

        # Construct NVFP4Tensor wrapping pre-allocated buffers (very cheap - just Python object init)
        x_nvfp4 = NVFP4Tensor(
            shape=(M, K),
            dtype=torch.bfloat16,
            rowwise_data=fp4_data,
            rowwise_scale_inv=scale_inv_buf,
            columnwise_data=None,
            columnwise_scale_inv=None,
            amax_rowwise=amax_buf,
            amax_columnwise=None,
            fp4_dtype=tex.DType.kFloat4E2M1,
            quantizer=input_quantizer,
            requires_grad=False,
        )

        # ---- Weight: standard TE quantization ----
        w_nvfp4 = weight_quantizer.quantize(weight)

        # ---- GEMM: tex.generic_gemm ----
        y = torch.empty((M, N), dtype=input.dtype, device=input.device)
        tex.generic_gemm(
            w_nvfp4,    # A
            True,       # transa
            x_nvfp4,    # B
            False,      # transb
            y,          # D (output)
            None,       # D_quantizer
            TE_DType[input.dtype],
            bias,
            TE_DType[bias.dtype] if bias is not None else TE_DType[input.dtype],
            False,      # use_gelu
            None,       # gelu_input
            False,      # use_grad
            workspace,
            workspace.shape[0],
            False,      # accumulate
            False,      # use_split_accumulator
        )

        # Save for backward
        ctx.save_for_backward(input, weight, norm_weight)
        ctx.epsilon = epsilon
        ctx.norm_mode = norm_mode
        ctx.act_mode = act_mode
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
        ctx.grad_output_quantizer = grad_output_quantizer
        ctx.workspace = workspace

        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, norm_weight = ctx.saved_tensors
        workspace = ctx.workspace

        # Backward doesn't need norm/act fusion — use standard TE path
        # 1. Quantize grad_output
        dY_nvfp4 = ctx.grad_output_quantizer.quantize(grad_output.to(torch.bfloat16))

        # 2. Re-quantize weight
        w_nvfp4 = ctx.weight_quantizer.quantize(weight)

        # 3. dgrad = dY @ W (backward input gradient)
        grad_input = tex.generic_gemm(
            w_nvfp4, False, dY_nvfp4, False,
            None, None, TE_DType[input.dtype],
            None, TE_DType[input.dtype],
            False, None, False,
            workspace, workspace.shape[0], False, False,
        )[0]

        # 4. Re-quantize input (recompute norm+act, then quantize)
        h = F.silu(F.rms_norm(input, (input.shape[-1],), norm_weight, ctx.epsilon))
        x_nvfp4 = ctx.input_quantizer.quantize(h)

        # 5. wgrad = dY.T @ X
        grad_weight = tex.generic_gemm(
            x_nvfp4, False, dY_nvfp4, True,
            None, None, TE_DType[weight.dtype],
            None, TE_DType[weight.dtype],
            False, None, False,
            workspace, workspace.shape[0], False, False,
        )[0]

        # 6. dbias
        grad_bias = None
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=0)

        grad_norm = None

        # Return grads for all forward inputs (14 inputs)
        return grad_input, grad_weight, grad_norm, grad_bias, None, None, None, None, None, None, None, None, None, None


class V7FusedLinear(nn.Module):
    """
    Linear layer with V7 fused RMSNorm+Activation+NVFP4 quantization.

    Drop-in replacement for TEParityLinearTex. The key difference:
    - Forward input quantization uses V7 (fuses norm+act+quant in 1 CUDA kernel)
    - Everything else (weight quant, GEMM, backward) uses standard TE ops
    - All intermediate buffers are pre-allocated to eliminate Python overhead
    """

    def __init__(self, in_features, out_features, bias=True, mx_config=None,
                 norm_mode=0, act_mode=0, epsilon=1e-5, max_batch_tokens=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon
        self.norm_mode = norm_mode  # 0=RMS, 1=AbsMax, 2=MXNorm
        self.act_mode = act_mode    # 0=SiLU, 1=GeLU, 2=Identity

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.023
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        # RMSNorm weight
        self.norm_weight = nn.Parameter(torch.ones(in_features, dtype=torch.bfloat16))

        def get_cfg(name, default):
            if mx_config is None:
                return default
            return getattr(mx_config, name, default)

        te_dtype = tex.DType.kFloat4E2M1
        use_rht = get_cfg("use_rht", False)
        use_2d = get_cfg("use_2d_weights", False)
        with_random_sign_mask = get_cfg("with_random_sign_mask", True)
        encode_centric = get_cfg("encode_centric", False)

        round_mode = get_cfg("roundMode", "TiesToEven")
        stochastic_rounding = (round_mode == "Stochastic") or get_cfg("stochastic_rounding", False)

        # Input quantizer (used for backward re-quant, not forward — forward uses V7)
        self.input_quantizer = NVFP4Quantizer(
            fp4_dtype=te_dtype, rowwise=True, columnwise=True,
            with_amax_reduction=False, amax_reduction_group=None,
            with_rht=use_rht, with_post_rht_amax=use_rht,
            with_2d_quantization=False, stochastic_rounding=False,
            with_random_sign_mask=with_random_sign_mask,
            encode_centric=encode_centric,
        )
        self.weight_quantizer = NVFP4Quantizer(
            fp4_dtype=te_dtype, rowwise=True, columnwise=True,
            with_amax_reduction=False, amax_reduction_group=None,
            with_rht=False, with_post_rht_amax=False,
            with_2d_quantization=use_2d, stochastic_rounding=False,
            with_random_sign_mask=with_random_sign_mask,
            encode_centric=encode_centric,
        )
        self.grad_output_quantizer = NVFP4Quantizer(
            fp4_dtype=te_dtype, rowwise=True, columnwise=True,
            with_amax_reduction=False, amax_reduction_group=None,
            with_rht=use_rht, with_post_rht_amax=use_rht,
            with_2d_quantization=False, stochastic_rounding=stochastic_rounding,
            with_random_sign_mask=with_random_sign_mask,
            encode_centric=encode_centric,
        )

        # Pre-allocate buffers (lazy init on first forward)
        self._max_M = max_batch_tokens or 0
        self._bufs_device = None
        self._fp4_data_buf = None
        self._scale_inv_buf = None
        self._amax_buf = None
        self._workspace = None

    def _ensure_buffers(self, M, K, device):
        """Pre-allocate or resize buffers if needed. Only re-allocates when M grows."""
        if self._bufs_device == device and M <= self._max_M:
            return  # Already allocated and big enough

        self._max_M = max(M, self._max_M)
        padded_M = round_up_to_nearest_multiple(self._max_M, 128)
        padded_scale_cols = round_up_to_nearest_multiple(
            math.ceil(K / NVFP4_BLOCK_SCALING_SIZE), 4)

        self._fp4_data_buf = torch.empty(self._max_M, K // 2, dtype=torch.uint8, device=device)
        self._scale_inv_buf = torch.zeros(padded_M, padded_scale_cols, dtype=torch.uint8, device=device)
        self._amax_buf = torch.zeros(1, dtype=torch.float32, device=device)
        self._workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._bufs_device = device

    @classmethod
    def from_float(cls, mod, config=None):
        """Create V7FusedLinear from nn.Linear, similar to TEParityLinearTex.from_float"""
        new_mod = cls(
            mod.in_features, mod.out_features,
            bias=mod.bias is not None,
            mx_config=config,
        )
        new_mod = new_mod.to(mod.weight.device).to(mod.weight.dtype)
        with torch.no_grad():
            new_mod.weight.copy_(mod.weight)
            if mod.bias is not None:
                new_mod.bias.copy_(mod.bias)
        return new_mod

    def forward(self, input):
        is_3d = input.dim() == 3
        if is_3d:
            B, S, H = input.shape
            input_2d = input.view(B * S, H)
        else:
            input_2d = input

        M, K = input_2d.shape

        # Ensure pre-allocated buffers exist and are large enough
        self._ensure_buffers(M, K, input.device)

        # Zero just the scale region we'll write to (cheaper than full zero)
        src_cols = K // NVFP4_BLOCK_SCALING_SIZE
        self._scale_inv_buf[:M, :src_cols].zero_()

        out = V7FusedLinearFunction.apply(
            input_2d, self.weight, self.norm_weight, self.bias,
            self.epsilon, self.norm_mode, self.act_mode,
            self._fp4_data_buf, self._scale_inv_buf, self._amax_buf,
            self.input_quantizer, self.weight_quantizer,
            self.grad_output_quantizer, self._workspace,
        )

        if is_3d:
            return out.view(B, S, self.out_features)
        return out
