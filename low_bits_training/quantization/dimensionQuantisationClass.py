from typing import Dict, Union, Any, Optional

import torch
import re
# Import torch._dynamo to adjust configs if needed, though the structural fix usually suffices
import torch._dynamo 
import torch._inductor.config

# --- OPTIONAL: Increase limit globally just in case ---
torch._dynamo.config.cache_size_limit = 128 
torch._inductor.config.triton.cudagraphs = True 

# ENABLE COORDINATE DESCENT TUNING:
# Helps finding better Triton config parameters
torch._inductor.config.coordinate_descent_tuning = True

from low_bits_training.quantization.MXFPconfig import MXGemmKernelChoice
from low_bits_training.quantization.ExMy import ExMy_new
from low_bits_training.quantization.quantization_autograd_functions import *
from scipy.linalg import hadamard
from dataclasses import dataclass

from torchao.prototype.mx_formats.constants import (
    BLOCK_SIZE_DEFAULT, DTYPE_FP4, DTYPE_FP6_E2M3, DTYPE_FP6_E3M2,
    E8M0_EXPONENT_BIAS, E8M0_EXPONENT_NAN_VAL, F4_E2M1_MAX,
    F4_E2M1_MAX_POW2, F6_E2M3_MAX, F6_E2M3_MAX_POW2, F6_E3M2_MAX,
    F6_E3M2_MAX_POW2, F8E4M3_MAX, F8E4M3_MAX_POW2, F8E5M2_MAX,
    F8E5M2_MAX_POW2, F32_MIN_NORMAL, SUPPORTED_ELEM_DTYPES,
)

@dataclass
class DimensionMXTensor:
    _data: torch.Tensor
    _scale_fp: torch.Tensor
    _scale_lp: torch.Tensor
    _elem_dtype: Union[torch.dtype, str]
    _block_size: int
    _orig_dtype: torch.dtype
    _use_fp4_custom_triton_dequant_kernel: bool
    _gemm_kernel_choice: Any
    _max_abs: torch.Tensor
    _max_abs_mask: Optional[torch.Tensor]
    _sm: Optional[torch.Tensor]
    _g: torch.Tensor
    _global_abs_mask: Optional[torch.Tensor]
    _nu: Optional[torch.Tensor]
    _strategy: str = 'encode'
    _use_fp32_scaling: bool = True  # NEW: Precision control

    def to_dtype(self, target_dtype: torch.dtype) -> torch.Tensor:
        return to_dtype_dim(
            self._data, self._scale_lp, self._elem_dtype, self._block_size,
            target_dtype, self._g, self._nu, self._strategy,
            self._use_fp32_scaling
        )

    def __repr__(self):
        return (f"MXTensor(data={self._data.shape}, "
                f"scale_shape={self._scale_lp.shape}, "
                f"elem_dtype={self._elem_dtype}, "
                f"strategy={self._strategy})")


def blockify(data_hp: torch.Tensor, block_size):
    orig_shape = data_hp.shape
    if data_hp.numel() % block_size != 0:
            raise AssertionError("unsupported block size alignment")
    data_hp = data_hp.reshape(-1, block_size)
    return data_hp, orig_shape

# --- KERNEL LOGIC ---
# REMOVED: @torch.compile(dynamic=True) <-- Removed from here
def _core_scaling_math(
    data_hp: torch.Tensor, 
    block_size: int,
    fp_scale_factor: bool,
    strategy: str,
    E: int, M: int, 
    use_tensor_scaling: bool,
    round_mode: Any, 
    nan_handling_mode: str, 
    scale_range_normalisation: bool,
    approx_smooth: str,
    approx_alpha: float,
    use_fp32_scaling: bool = True,  # NEW: Match TE's FP32 precision for scaling
):
    # 1. Blockify
    # if data_hp.numel() > 100000:
    #     print(f"  [Core Scaling DEBUG] block_size: {block_size}, input shape: {data_hp.shape}")
    data_hp_flat, orig_shape = blockify(data_hp, block_size)
    abs_data = torch.abs(data_hp_flat)
    
    # 2. Compute Stats (Max Abs)
    sm = None
    if approx_smooth in ['absmax', 'STE']:
        max_abs = abs_data.amax(dim=1, keepdim=True)
        max_abs = torch.where(max_abs == 0, 1.0, max_abs)
        max_abs_mask = abs_data != max_abs
    elif approx_smooth == 'softsoftmax':
        alpha_abs = approx_alpha * abs_data
        max_abs = (1 / approx_alpha) * torch.logsumexp(alpha_abs, dim=1, keepdim=True)
        max_abs = torch.where(max_abs == 0, 1.0, max_abs)
        sm = torch.softmax(alpha_abs, dim=1)
        max_abs_mask = None 
    else:
        max_abs = abs_data.amax(dim=1, keepdim=True)
        max_abs = torch.where(max_abs == 0, 1.0, max_abs)
        max_abs_mask = None

    # 3. ExMy Logic
    quantizer = ExMy_new(E, M, 
                         use_tensor_scaling=use_tensor_scaling,
                         roundMode=round_mode, 
                         nan_handling_mode=nan_handling_mode, 
                         scale_range_normalisation=scale_range_normalisation,
                         use_fp32_scaling=use_fp32_scaling)
    
    scale_lp, scale_fp, g_final, g_raw = quantizer.compute_scaling(max_abs, strategy)
    glob_max_abs_mask = (abs_data == g_raw)

    # 4. Apply Scaling
    effective_scale = scale_fp if fp_scale_factor else scale_lp
    max_pos = F4_E2M1_MAX 

    # NOTE: When use_fp32_scaling=True, compute in FP32 before FP4 rounding
    # This matches TE which does: bf16->fp32, scale in fp32, then cvt to fp4
    # Old behavior (use_fp32_scaling=False) keeps everything in input dtype
    if use_fp32_scaling:
        if strategy == 'encode':
            data_lp = torch.clamp(data_hp_flat.float() * effective_scale / g_final, -max_pos, max_pos)
        else:
            # NOTE: TE uses multiplication formula: input * (S_enc / S_dec_b)
            # where S_enc = 1/g. This is numerically more stable at tie-breaking
            # boundaries than the equivalent division: input / (S_dec_b * g)
            # Compute: input * (1/g) / scale = input / g / scale
            scale_mult = (1.0 / g_final) / effective_scale
            data_lp = torch.clamp(data_hp_flat.float() * scale_mult, -max_pos, max_pos)
    else:
        # Original behavior: stay in input dtype (may lose precision but ensures exact scale cancellation)
        if strategy == 'encode':
            data_lp = torch.clamp(data_hp_flat * effective_scale / g_final, -max_pos, max_pos) 
        else:
            data_lp = torch.clamp(data_hp_flat / (effective_scale * g_final), -max_pos, max_pos)

    data_lp = data_lp.reshape(orig_shape)
    
    return scale_fp, scale_lp, data_lp, max_abs, max_abs_mask, sm, g_final, glob_max_abs_mask, 1


class MXFPscalingModule(torch.nn.Module):
    def __init__(self,
        elem_dtype: Union[torch.dtype, str],
        block_size: int,
        scale_type: str = 'E8M0',
        roundMode = gfloat.RoundMode.TowardPositive,
        use_approx: dict = {'smooth': False},
        fp_scale_factor: bool = False,
        nan_handling_mode: str = 'nearest_subnormal',
        scale_range_normalisation: bool = False,
        strategy: str = 'encode',
        use_fp32_scaling: bool = True,  # NEW: Match TE's FP32 precision for scaling
    ):
        super().__init__()
        self.elem_dtype = elem_dtype
        self.block_size = block_size
        self.scale_type = scale_type
        self.roundMode = roundMode
        self.use_approx = use_approx if use_approx is not None else {'smooth': False}
        self.fp_scale_factor = fp_scale_factor
        self.strategy = strategy.lower()
        self.E, self.M = extract_e_m(scale_type=self.scale_type)
        self.nan_handling_mode = nan_handling_mode
        self.scale_range_normalisation = scale_range_normalisation
        self.use_fp32_scaling = use_fp32_scaling
        
        self.approx_smooth = self.use_approx.get('smooth', 'absmax')
        self.approx_alpha = self.use_approx.get('alpha', 1.0)
        self.use_tensor_scaling = self.use_approx.get('use_tensor_scaling', True)

        # --- IMPORTANT CHANGE ---
        # Compile the function specifically for THIS instance.
        # This prevents cache thrashing between different layers/instances.
        self.compiled_math_kernel = _core_scaling_math


    def forward(self, data_hp):
        # Call the instance-specific compiled kernel
        return self.compiled_math_kernel(
            data_hp,
            self.block_size,
            self.fp_scale_factor,
            self.strategy,
            self.E,
            self.M,
            self.use_tensor_scaling,
            self.roundMode,
            self.nan_handling_mode,
            self.scale_range_normalisation,
            self.approx_smooth,
            self.approx_alpha,
            self.use_fp32_scaling,
        )

def extract_e_m(scale_type: str):
    if scale_type == 'Ideal':
        return 8, 23
    match = re.match(r"E(\d+)M(\d+)", scale_type)
    if not match:
        raise ValueError(f"Invalid format: {scale_type}")
    E = int(match.group(1))
    M = int(match.group(2))
    return E, M

def to_dtype_dim(data_lp, scale_lp, elem_dtype, block_size, target_dtype, g, nu, strategy='encode', use_fp32_scaling=True):
    orig_shape = data_lp.shape
    # NOTE: Compute in FP32 for precision, then convert to target_dtype (matches TE)
    # Old behavior (use_fp32_scaling=False) keeps everything in input/target dtype
    if use_fp32_scaling:
        data_hp = data_lp.float().reshape(-1, block_size)
    else:
        data_hp = data_lp.to(target_dtype).reshape(-1, block_size)

    if strategy == 'encode':
        if use_fp32_scaling:
             data_hp = (g * data_hp / scale_lp.float())
        else:
             data_hp = (g * data_hp / scale_lp)
    else:
        if use_fp32_scaling:
             data_hp = (g * data_hp * scale_lp.float())
        else:
             data_hp = (g * data_hp * scale_lp)
            
    data_hp = data_hp.reshape(orig_shape).to(target_dtype)
    return data_hp

def new_to_mx(
    tensor: torch.Tensor,
    scalingModule: MXFPscalingModule,
    gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
    fp4_quantiser: torch.nn.Module = None,
):
    tensor_orig_dtype = tensor.dtype
    tensor_orig_shape = tensor.shape
    tensor_hp_r = tensor.reshape(-1, tensor_orig_shape[-1])

    # forward() now calls the instance-compiled kernel
    scale_fp, scale_lp, tensor_pre_quant, max_abs, max_abs_mask, sm, g, glob_max_abs_mask, nu = scalingModule.forward(tensor_hp_r)

    if fp4_quantiser is not None:
        quantized_data = fp4_quantiser(tensor_pre_quant)
    else:
        quantized_data = tensor_pre_quant
        
    tensor_mx = DimensionMXTensor(
        _data=quantized_data,
        _scale_fp=scale_fp,
        _scale_lp=scale_lp,
        _elem_dtype=scalingModule.elem_dtype,
        _block_size=scalingModule.block_size,
        _orig_dtype=tensor_orig_dtype,
        _use_fp4_custom_triton_dequant_kernel=False,
        _gemm_kernel_choice=gemm_kernel_choice,
        _max_abs=max_abs,
        _max_abs_mask=max_abs_mask,
        _sm=sm,
        _g=g,
        _global_abs_mask=glob_max_abs_mask,
        _nu = nu,
        _strategy=scalingModule.strategy,
        _use_fp32_scaling=getattr(scalingModule, 'use_fp32_scaling', True)
    )
    
    return tensor_mx, tensor_orig_shape, tensor_pre_quant