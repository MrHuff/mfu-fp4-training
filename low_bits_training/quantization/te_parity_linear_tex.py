"""
TE version of TEParityLinear for testing against Triton implementation.

This module provides a linear layer that uses Transformer Engine's native
kernels (tex.generic_gemm) via NVFP4Quantizer.

It also includes a "Reconstruction Mode" (`use_dequant_gemm=True`) that performs
the GEMM using pure PyTorch operations on dequantized data. This verifies the
algebraic correctness of the NVFP4 GEMM kernel.

Algebraic Expression (Forward Pass):
------------------------------------
Let:
  X: Input tensor (M, K)
  W: Weight tensor (N, K)
  S_X_blk: Input Block Scales (M, K/16) [Level 1]
  S_X_glob: Input Global Scale (scalar) [Level 2]
  S_W_blk: Weight Block Scales (N, K/16) [Level 1]
  S_W_glob: Weight Global Scale (scalar) [Level 2]

tex.gemm(X_quant, W_quant) computes:
  Y = (X_quant.dequantize() @ W_quant.dequantize().T)

Where .dequantize() effectively reconstructs:
  X_dq[i, k] = X_raw[i, k] * S_X_blk[i, k//16] * S_X_glob
  W_dq[j, k] = W_raw[j, k] * S_W_blk[j, k//16] * S_W_glob

The GEMM operation is then:
  Y[i, j] = Sum_k ( X_dq[i, k] * W_dq[j, k] )
"""

import torch
import torch.nn as nn
from typing import Optional, Any
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.constants import TE_DType

# TODO ablate on gemm, dequantize BF16 matmul test convergence... if different double check black_magic Nvidia Gemm...
# TODO We know naive linear layer is popping!


class TEParityLinearTexFunction(torch.autograd.Function):
    @staticmethod
    def _fp4_e2m1_vals(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Values representable in FP4 E2M1 format"""
        return torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _decode_scale(
        scale_inv: torch.Tensor,
        target_shape: tuple,
        scale_format: str = "E4M3",
        expand: bool = False,
    ) -> torch.Tensor:
        """
        Decodes the scale tensor (scale_inv) into a float32 scale tensor expanded to target_shape.
        If expand is False, returns the compressed scale tensor (R, K/BlockSize).
        """
        R, C = target_shape

        # 2. Block Scales (E8M0 / E4M3FN)
        # Handle potential padding in scale_inv
        dim1_blocks = C // 16

        if scale_format == "E8M0":
            # UE8M0 decoding: 2^(uint8 - 127)
            scale_u8 = scale_inv.view(torch.uint8)
            # Use actual columns
            scale_f32 = 2.0 ** (scale_u8.float() - 127.0)

        elif scale_format == "E4M3":
            scale_f32 = scale_inv.view(torch.float8_e4m3fn).to(torch.float32)
        else:
            raise ValueError(
                f"Unsupported scale format for manual dequant: {scale_format}"
            )

        # Determine actual block size available in scale
        # scale_f32 is (R_padded, C_scales_padded)
        scale_f32 = scale_f32[:R, :]  # Slice Rows

        if not expand:
            return scale_f32

        num_scale_cols = scale_f32.shape[1]

        # If we have 4 cols for 128 elements, block size is 32.
        # If we have 4 cols for 64 elements, block size is 16.
        # Check against C (target cols)

        if num_scale_cols > 0:
            block_size = C // num_scale_cols
            # Handle padding if C is not divisible? Usually aligned.
        else:
            block_size = 32  # Default fallback

        scale_expanded = torch.repeat_interleave(scale_f32, block_size, dim=1)

        # Clip to C if expanded is larger
        scale_expanded = scale_expanded[:, :C]

        return scale_expanded

    @staticmethod
    def _manual_dequantize(
        data_packed: torch.Tensor,
        scale_inv: torch.Tensor,
        amax: torch.Tensor,
        target_shape: tuple,
        scale_format: str = "E4M3",
        label: str = "Unknown",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Manually dequantize NVFP4 packed data.
        Returns:
            tensor: Block-scaled (but not globally scaled) float tensor.
            global_scale: Scalar float tensor.
        """
        import math

        R, C = target_shape
        device = data_packed.device

        # 1. Unpack Data (E2M1)
        # CAST TO INT32 to avoid boolean mask indexing error!
        data_u8 = data_packed.view(torch.uint8).to(torch.int32)
        # stack: [low, high]
        unpacked_indices = torch.stack((data_u8 & 0x0F, data_u8 >> 4), dim=-1).reshape(
            R, C
        )
        # unpacked_indices is now int32, safe for indexing

        # Lookup values
        lut = TEParityLinearTexFunction._fp4_e2m1_vals(device, torch.float32)
        data_f32 = lut[unpacked_indices.to(torch.long)]  # Explicitly use Long for safety

        # 2. Get Expanded Scale
        scale_expanded = TEParityLinearTexFunction._decode_scale(
            scale_inv, target_shape, scale_format
        )

        # 3. Global Scale
        global_scale = amax / (6.0 * 448.0)

        # 4. Combine (Block Scale only)
        result = data_f32 * scale_expanded

        return result, global_scale

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_quantizer: NVFP4Quantizer,
        weight_quantizer: NVFP4Quantizer,
        grad_output_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
        use_dequant_gemm: bool = False,
        scale_format: str = "E8M0",
    ):
        input_shape = input.shape
        weight_shape = weight.shape

        x_nvfp4 = input_quantizer.quantize(input)
        w_nvfp4 = weight_quantizer.quantize(weight)

        if use_dequant_gemm:
            # Dequantize first (Deferred Scaling)
            x_dq_blk, x_scale = TEParityLinearTexFunction._manual_dequantize(
                x_nvfp4._rowwise_data,
                x_nvfp4._rowwise_scale_inv,
                x_nvfp4._amax_rowwise,
                x_nvfp4.shape,
                scale_format=scale_format,
                label="Forward Input",
            )
            w_dq_blk, w_scale = TEParityLinearTexFunction._manual_dequantize(
                w_nvfp4._rowwise_data,
                w_nvfp4._rowwise_scale_inv,
                w_nvfp4._amax_rowwise,
                w_nvfp4.shape,
                scale_format=scale_format,
                label="Forward Weight",
            )

            # GEMM: Input @ Weight.T
            # Accumulate high precision, then scale
            y = torch.mm(x_dq_blk, w_dq_blk.t()).to(input.dtype)

            # Derived Alpha Scaling:
            # alpha = (amax_x * amax_w) / (6 * 448)^2
            nvfp4_scale_const = 6.0 * 448.0
            alpha = (x_nvfp4._amax_rowwise * w_nvfp4._amax_rowwise) / (
                nvfp4_scale_const * nvfp4_scale_const
            )
            y = y * alpha

            if bias is not None:
                y = y + bias
        else:
            # Explicitly allocate output to enforce device
            y = torch.empty(
                (input_shape[0], weight_shape[0]), dtype=input.dtype, device=input.device
            )

            tex.generic_gemm(
                w_nvfp4,  # A
                True,  # transa
                x_nvfp4,  # B
                False,  # transb
                y,  # D (output)
                None,  # D_quantizer
                TE_DType[input.dtype],  # D_dtype
                bias,  # bias
                TE_DType[bias.dtype]
                if bias is not None
                else TE_DType[input.dtype],  # bias_dtype
                False,  # use_gelu
                None,  # gelu_input
                False,  # use_grad
                workspace,  # workspace
                workspace.shape[0],  # workspaceSize
                False,  # accumulate
                False,  # use_split_accumulator
            )

        # Save for backward
        ctx.save_for_backward(input, weight)
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
        ctx.grad_output_quantizer = grad_output_quantizer
        ctx.workspace = workspace
        ctx.use_dequant_gemm = use_dequant_gemm
        ctx.scale_format = scale_format

        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_quantizer = ctx.input_quantizer
        weight_quantizer = ctx.weight_quantizer
        grad_output_quantizer = ctx.grad_output_quantizer
        workspace = ctx.workspace
        use_dequant_gemm = ctx.use_dequant_gemm
        scale_format = getattr(ctx, "scale_format", "E8M0")

        grad_input = None
        grad_weight = None
        grad_bias = None

        # 1. Quantize grad_output (dY)
        # 1. Quantize grad_output (dY)
        # Force BF16 for RHT compatibility
        dY_nvfp4 = grad_output_quantizer.quantize(grad_output.to(torch.bfloat16))

        # Re-quantize Weight (N, K)
        w_nvfp4 = weight_quantizer.quantize(weight)

        if use_dequant_gemm:
            # Backward Input: dY @ W

            w_dq_blk = None
            w_scale = None

            if w_nvfp4._columnwise_data is not None:
                w_T_dq_blk, w_scale = TEParityLinearTexFunction._manual_dequantize(
                    w_nvfp4._columnwise_data,
                    w_nvfp4._columnwise_scale_inv,
                    w_nvfp4._amax_columnwise,
                    (w_nvfp4.shape[1], w_nvfp4.shape[0]),  # (K, N)
                    scale_format=scale_format,
                    label="Backward Weight (Col)",
                )
                w_dq_blk = w_T_dq_blk.t()
            else:
                w_dq_blk, w_scale = TEParityLinearTexFunction._manual_dequantize(
                    w_nvfp4._rowwise_data,
                    w_nvfp4._rowwise_scale_inv,
                    w_nvfp4._amax_rowwise,
                    w_nvfp4.shape,
                    scale_format=scale_format,
                    label="Backward Weight (Row)",
                )

            # Need dY (M, N).
            dY_dq_blk, _ = TEParityLinearTexFunction._manual_dequantize(
                dY_nvfp4._rowwise_data,
                dY_nvfp4._rowwise_scale_inv,
                dY_nvfp4._amax_rowwise,
                dY_nvfp4.shape,
                scale_format=scale_format,
                label="Backward dY",
            )

            # dgrad = dY @ W
            grad_input = torch.mm(dY_dq_blk, w_dq_blk).to(input.dtype)

            # Alpha Scaling
            nvfp4_scale_const = 6.0 * 448.0
            w_amax = (
                w_nvfp4._amax_columnwise
                if w_nvfp4._columnwise_data is not None
                else w_nvfp4._amax_rowwise
            )
            alpha = (dY_nvfp4._amax_rowwise * w_amax) / (
                nvfp4_scale_const * nvfp4_scale_const
            )
            grad_input = grad_input * alpha
        else:
            grad_input = tex.generic_gemm(
                w_nvfp4,  # A
                False,  # transa
                dY_nvfp4,  # B
                False,  # transb
                None,  # D
                None,
                TE_DType[input.dtype],
                None,  # bias
                TE_DType[input.dtype],  # bias_dtype
                False,
                None,
                False,
                workspace,
                workspace.shape[0],
                False,
                False,
            )[0]

        # Re-quantize Input (M, K)
        x_nvfp4 = input_quantizer.quantize(input)

        if use_dequant_gemm:
            # dWeight = dY.T @ X

            dY_T_ready = None
            dY_scale = None

            # 1. Prepare dY transpose for matmul.
            if dY_nvfp4._columnwise_data is not None:
                dY_T_dq_blk, dY_scale = TEParityLinearTexFunction._manual_dequantize(
                    dY_nvfp4._columnwise_data,
                    dY_nvfp4._columnwise_scale_inv,
                    dY_nvfp4._amax_columnwise,
                    (dY_nvfp4.shape[1], dY_nvfp4.shape[0]),  # (N, M)
                    scale_format=scale_format,
                    label="Backward dY (Col)",
                )
                dY_T_ready = dY_T_dq_blk
            else:
                dY_dq_blk, dY_scale = TEParityLinearTexFunction._manual_dequantize(
                    dY_nvfp4._rowwise_data,
                    dY_nvfp4._rowwise_scale_inv,
                    dY_nvfp4._amax_rowwise,
                    dY_nvfp4.shape,
                    scale_format=scale_format,
                    label="Backward dY (Row)",
                )
                dY_T_ready = dY_dq_blk.t()

            x_ready = None
            x_scale = None

            # 2. Prepare X for matmul.
            if x_nvfp4._columnwise_data is not None:
                x_T_dq_blk, x_scale = TEParityLinearTexFunction._manual_dequantize(
                    x_nvfp4._columnwise_data,
                    x_nvfp4._columnwise_scale_inv,
                    x_nvfp4._amax_columnwise,
                    (x_nvfp4.shape[1], x_nvfp4.shape[0]),  # (K, M)
                    scale_format=scale_format,
                    label="Backward Input (Col)",
                )
                x_ready = x_T_dq_blk.t()
            else:
                x_dq_blk, x_scale = TEParityLinearTexFunction._manual_dequantize(
                    x_nvfp4._rowwise_data,
                    x_nvfp4._rowwise_scale_inv,
                    x_nvfp4._amax_rowwise,
                    x_nvfp4.shape,
                    scale_format=scale_format,
                    label="Backward Input (Row)",
                )
                x_ready = x_dq_blk

            # wgrad = dY.T @ X
            grad_weight = torch.mm(dY_T_ready, x_ready).to(weight.dtype)

            # Alpha Scaling
            nvfp4_scale_const = 6.0 * 448.0
            dY_amax = (
                dY_nvfp4._amax_columnwise
                if dY_nvfp4._columnwise_data is not None
                else dY_nvfp4._amax_rowwise
            )
            x_amax = (
                x_nvfp4._amax_columnwise
                if x_nvfp4._columnwise_data is not None
                else x_nvfp4._amax_rowwise
            )

            alpha = (dY_amax * x_amax) / (nvfp4_scale_const * nvfp4_scale_const)
            grad_weight = grad_weight * alpha
        else:
            grad_weight = tex.generic_gemm(
                x_nvfp4,  # A
                False,  # transa
                dY_nvfp4,  # B
                True,  # transb
                None,  # D
                None,
                TE_DType[weight.dtype],
                None,  # bias
                TE_DType[weight.dtype],
                False,
                None,
                False,
                workspace,
                workspace.shape[0],
                False,
                False,
            )[0]

        # 4. dbias = sum(dY, dim=0)
        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class TEParityLinearTex(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, mx_config=None, use_dequant_gemm=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_dequant_gemm = use_dequant_gemm

        # --- FIX: Explicitly use bfloat16 (or pass in a dtype arg) ---
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.023
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        # Config setup
        self.mx_config = mx_config

        def get_cfg(name, default):
            if mx_config is None:
                return default
            return getattr(mx_config, name, default)

        # TE DType for FP4
        self.te_dtype = tex.DType.kFloat4E2M1

        # Extract config
        use_rht = get_cfg("use_rht", False)
        use_2d = get_cfg("use_2d_weights", False)
        with_random_sign_mask = get_cfg("with_random_sign_mask", True)
        encode_centric = get_cfg("encode_centric", False)

        self.scale_format = get_cfg(
            "scale_type", "E4M3"
        )  # Default to E4M3 per user preference

        round_mode = get_cfg("roundMode", "TiesToEven")
        # Check if stochastic rounding is requested (standard TE specific)
        stochastic_rounding = (round_mode == "Stochastic") or get_cfg(
            "stochastic_rounding", False
        )
        # Initialize Quantizers
        self.input_quantizer = NVFP4Quantizer(
            fp4_dtype=self.te_dtype,
            rowwise=True,
            columnwise=True,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=use_rht,
            with_post_rht_amax=use_rht,
            with_2d_quantization=False,  # Input usually doesn't use 2D quantization
            stochastic_rounding=False,
            with_random_sign_mask=with_random_sign_mask,
            encode_centric=encode_centric,
        )
        self.weight_quantizer = NVFP4Quantizer(
            fp4_dtype=self.te_dtype,
            rowwise=True,
            columnwise=True,  # Disable columnwise to avoid shared memory OOM and match transa=True
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=use_2d,  # Weight can use 2D
            stochastic_rounding=False,
            with_random_sign_mask=with_random_sign_mask,
            encode_centric=encode_centric,
        )
        self.grad_output_quantizer = NVFP4Quantizer(
            fp4_dtype=self.te_dtype,
            rowwise=True,
            columnwise=True,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=use_rht,
            with_post_rht_amax=use_rht,
            with_2d_quantization=False,
            stochastic_rounding=stochastic_rounding,  # Gradients support SR
            with_random_sign_mask=with_random_sign_mask,
            encode_centric=encode_centric,
        )

    @classmethod
    def from_float(cls, mod, config, use_dequant_gemm=False):
        # Determine if we should propagate use_dequant_gemm if it exists on mod?
        # Usually from_float acts on standard Linear, which doesn't have it.
        # But we can check config for it maybe?
        # Priority: explicit arg > config object > default
        if not use_dequant_gemm:
            use_dequant_gemm = getattr(config, "use_dequant_gemm", False)

        new_mod = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            mx_config=config,
            use_dequant_gemm=use_dequant_gemm,
        )
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

        # Workspace for cuBLAS (Allocate once per forward, or reuse buffer)
        # TE seems to use 4 bytes or similar small size for workspace in test?
        # "Allocates internal workspace for cublasLt"
        # In test_nvfp4_gemm_exact.py: workspace = torch.empty(4, dtype=torch.uint8, device=device)
        # But real usage might need proper sizing. For now let's stick to test usage.
        # Actually `workspaceSize` arg is passed. If too small, it might fail or use internal?
        # Let's allocate a decent size just in case, e.g. 32MB? or simple 4 bytes as in test.
        # Test sets workspace.shape[0] which is 4.
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=input.device)
        # if not hasattr(self, '_printed_cfg'):
        #      print(f"[TEParityLinearTex] Custom config: rht={self.input_quantizer.with_rht} 2d={self.weight_quantizer.with_2d_quantization} col_in={self.input_quantizer.columnwise_usage} col_w={self.weight_quantizer.columnwise_usage}")
        #      self._printed_cfg = True

        out = TEParityLinearTexFunction.apply(
            input_2d,
            self.weight,
            self.bias,
            self.input_quantizer,
            self.weight_quantizer,
            self.grad_output_quantizer,
            workspace,
            self.use_dequant_gemm,
            self.scale_format,
        )

        if out.device != input.device:
            print(
                f"DEBUG ERROR: Output device {out.device} != Input device {input.device}"
            )

        if is_3d:
            return out.view(B, S, self.out_features)

        return out
