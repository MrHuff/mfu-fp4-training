import torch
from torchao.prototype.mx_formats.constants import (
    F32_MIN_NORMAL,
)
from gfloat.formats import format_info_ocp_e2m1,format_info_ocp_e4m3,format_info_ocp_e8m0,format_info_ocp_e5m2, format_info_bfloat16
from gfloat.types import FormatInfo,Domain

import gfloat
from low_bits_training.quantization.customTorchQuantiser import round_ndarray
from torchao.prototype.mx_formats.constants import (
    F4_E2M1_MAX, F4_E2M1_MAX_POW2,
)

def debug_tensor(mx_tensor: torch.Tensor, tensor_name: str):
    if torch.isnan(mx_tensor).any().item():
        print(f'{tensor_name} has nan')


format_info_ocp_e8m3 = FormatInfo(
    name="ocp_e8m3",
    k=12,
    precision=4,
    bias=2 ** (8 - 1) - 1,
    has_nz=True,
    num_high_nans=0,
    domain=Domain.Finite,

    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)


format_info_ocp_ue5m3 = FormatInfo(
    name="ocp_ue5m3",
    k=8,
    precision=4,
    bias=15,
    has_nz=False,
    domain=Domain.Finite,

    num_high_nans=0,
    has_subnormals=True,
    is_signed=False,
    is_twos_complement=False,
)

class ExMy_new:
    def __init__(self, e_bits: int, m_bits: int, use_tensor_scaling: bool = True, 
                 roundMode: gfloat.RoundMode.TiesToEven = gfloat.RoundMode.TiesToEven, 
                 nan_handling_mode='nearest_subnormal', scale_range_normalisation=False,
                 use_fp32_scaling: bool = False):  # NEW: Match TE's FP32 precision
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.use_tensor_scaling = use_tensor_scaling
        self.roundMode = roundMode
        self.nan_handling_mode = nan_handling_mode
        self.use_fp32_scaling = use_fp32_scaling
        if e_bits == 8 and m_bits == 0:
            self.format = format_info_ocp_e8m0
        if e_bits == 4 and m_bits == 3:
            self.format = format_info_ocp_e4m3
        if e_bits == 5 and m_bits==2:
            self.format = format_info_ocp_e5m2
        if e_bits == 5 and m_bits==3:
            self.format = format_info_ocp_ue5m3
        if e_bits == 8 and m_bits == 3:
            self.format = format_info_ocp_e8m3
        if e_bits == 8 and m_bits == 7:
            self.format = format_info_bfloat16
    
    def dummy_round(self, x):
        rounded = round_ndarray(
            self.format, x, self.roundMode, sat=True
        )
        return rounded
    
    def compute_scaling(self, max_abs, strategy='encode'):
        """
        Main logic hub. Calculates G, applies heuristic, and computes scales.
        """
        # 1. Calculate Raw Global Max (g) based on the flag
        if self.use_tensor_scaling:
            g_raw = max_abs.amax()
            # print(f"  [Custom ExMy DEBUG] g_raw: {g_raw.item():.10f}, max_abs.amax(): {max_abs.amax().item():.10f}")
            # For E8M0 (MXFP4), we don't have a format-defined max multiplier like E4M3.

            # We just want to normalize the tensor to the element max (6.0).
            if self.format.name == "ocp_e8m0":
                divisor = 6.0
            else:
                divisor = self.format.max * 6 
        else:
            g_raw = torch.tensor(1.0, device=max_abs.device, dtype=max_abs.dtype)
            divisor = 1.0

        # 4. Strategy Branching
        if strategy == 'encode':
            # --- ENCODE (Ours) ---
            # We pre-divide g by the divisor to align ranges for inversion
            if self.use_fp32_scaling:
                # NOTE: Keep in FP32 to avoid bf16 precision loss
                g_input = g_raw.float() / divisor
            else:
                g_input = g_raw / divisor
            
            # Compute Multiplier Scale
            scale_lp, scale_fp, _ = self._compute_scale_encode(max_abs, g_input)
            
            # For Encode, we return the pre-divided g. 
            # (Reconstruction: data * g_input / scale)
            g_final = g_input

        else:
            # --- DECODE (Nvidia) ---
            # Quantization uses real g (Magnitude calculation)
            g_input = g_raw
            
            # Compute Magnitude Scale
            scale_lp, scale_fp, _ = self._compute_scale_decode(max_abs, g_input)
            
            # Debug Print
            # if max_abs.numel() > 100: # Only for big layers like matmul
            #     print(f"  [ExMy DEBUG] g_raw: {g_raw.item():.6f}, Raw Scale[0]: {scale_fp.flatten()[0].item():.6f}, Rounded: {scale_lp.flatten()[0].item():.6f}")

            # BAKE Constants into G
            # We want reconstruction: data * scale * g_final
            # Standard Nvidia: data * scale * (g_raw / S_max) ... roughly.
            # To align with the symmetric Code structure:
            if self.use_fp32_scaling:
                # NOTE: Keep g_final in FP32 to preserve precision (matching TE)
                g_final = g_raw.float() / divisor
            else:
                # Old behavior: use BF16 (or input dtype)
                g_final = g_raw / divisor
            
            # if max_abs.numel() > 100:
            #     print(f"  [ExMy DEBUG] divisor: {divisor:.6f}, g_final: {g_final.item():.6f}")
              
        return scale_lp, scale_fp, g_final, g_raw

    def _compute_scale_encode(self, x,  g):
        if self.use_fp32_scaling:
            # NOTE: Compute in FP32 to avoid bf16 precision loss before E4M3 rounding
            scaled = 6 / x.float() * g.float()  # 6/absmax_b * g_input
        else:
            scaled = 6 / x * g #scale downwards
        scale_lp = self._perform_rounding(scaled)
        return scale_lp, scaled, 1

    def _compute_scale_decode(self, x,  g):
        if self.format.name == "ocp_e8m0":
            s_max_val = 1.0
        else:
            s_max_val = self.format.max 
        
        if self.use_fp32_scaling:
            # NOTE: Compute in FP32 to avoid bf16 precision loss before E4M3 rounding
            # BF16 truncates values like 400.516 -> 400.0, causing wrong E4M3 round (384 vs 416)
            scaled = s_max_val * (x.float() / g.float())  # E4M3_max * absmax_b / g_raw
        else:
            scaled = s_max_val * (x / g) #scale downwards   E4M3_max * absmax_b / g_raw 
        
        scale_lp = self._perform_rounding(scaled)
        return scale_lp, scaled, 1

    def _perform_rounding(self, val):
        if self.roundMode == gfloat.RoundMode.Stochastic:
            srbits = torch.randint_like(val, 65536)
            rounded = round_ndarray(self.format, val, self.roundMode, sat=True, srbits=srbits, srnumbits=16)
        else:
            rounded = round_ndarray(self.format, val, self.roundMode, sat=True)

            
        if self.nan_handling_mode == 'nearest_subnormal':
            low = 2**(-127) if self.format.name=='ocp_e8m0' else self.format.smallest_subnormal
            rounded = rounded.clip(low)
        elif self.nan_handling_mode == 'to_one':
            rounded = torch.where(rounded == 0, 1.0, rounded)
            
        return rounded.to(val.dtype)
    

if __name__ == "__main__":
    print("--- Comparing E4M3 Implementations ---")
    finfo = format_info_ocp_ue5m3
    assert finfo.maxexp == 16

    assert finfo.max == 2 ** (16 - 1) * (1 + 0.5 + 0.25 + 0.125)
    print(finfo.max)
    assert finfo.eps == 0.125
    assert finfo.smallest_normal == 2**-15
    assert finfo.smallest_subnormal == 2**-18

    # 1. Setup
    exmy_e4m3 = ExMy_new(e_bits=4, m_bits=3, roundMode=gfloat.RoundMode.TiesToEven,scale_range_normalisation=True)

    # Create some interesting test tensors
    x_input = torch.tensor([0.1, 0.99, 1.0, 50.0, 100.0, 500.0, 0.01,1e3], dtype=torch.bfloat16)
    g_input = x_input.max()
    min_scale = x_input.min()
    target_max_pow2_val = 2
    target_max_mbits_val = 1

    # 2. Define the three versions to test
    
    # Version A: Your eager-mode implementation
    def run_eager_pytorch(x, g):
        return exmy_e4m3.scale_by_format_max(x, target_max_pow2_val, target_max_mbits_val, g,min_scale)[0]

    # Version B: Your implementation with torch.compile
    compiled_scale_func = torch.compile(exmy_e4m3.scale_by_format_max)
    def run_compiled_pytorch(x, g):
        return compiled_scale_func(x, target_max_pow2_val, target_max_mbits_val, g,min_scale)[0]

    # Version C: The original gfloat.round with the same pre/post-processing
    def run_original_gfloat(x, g):
        # Replicate the pre-processing from your function
        m_bits = exmy_e4m3.m_bits
        if m_bits > 0:
            fp4_max = (2 ** target_max_pow2_val) * (1.0 + ((1 << target_max_mbits_val) - 1) / (1 << target_max_mbits_val))
        else:
            fp4_max = 2 ** target_max_pow2_val
        fp4_max = torch.tensor(fp4_max).to(dtype=x.dtype, device=x.device)
        scaled = fp4_max / x * g /1440
        
        # Use the original gfloat rounding function
        # NOTE: gfloat.round already includes saturation logic.
        rounded = gfloat.round_ndarray(exmy_e4m3.format,scaled,exmy_e4m3.roundMode,sat=True)
        
        # Apply the same post-rounding clip to make the comparison fair
        rounded = rounded.clip(exmy_e4m3.format.smallest_subnormal)

        return rounded.to(x.dtype)

    # 3. Run and print results
    result_eager = run_eager_pytorch(x_input, g_input)
    result_compiled = run_compiled_pytorch(x_input, g_input)
    result_gfloat = run_original_gfloat(x_input, g_input)

    print(f"\nInput Tensor: {x_input}")
    print("-" * 35)
    print(f"PyTorch (Eager):   {result_eager}")
    print(f"PyTorch (Compiled):  {result_compiled}")
    print(f"Original gfloat:     {result_gfloat}")
    print("-" * 35)

    # 4. Verification
    print("\n--- Verification ---")
    
    # Compare Eager vs. Compiled
    eager_vs_compiled_match = torch.allclose(result_eager, result_compiled, equal_nan=True)
    print(f"Eager and Compiled results match: {eager_vs_compiled_match}")

    # Compare Eager vs. Original gfloat
    eager_vs_gfloat_match = torch.allclose(result_eager, result_gfloat, equal_nan=True)
    print(f"Eager and Original gfloat results match: {eager_vs_gfloat_match}")

    if eager_vs_compiled_match and eager_vs_gfloat_match:
        print("\n✅ All implementations produce identical results.")
    else:
        print("\n❌ Mismatch found between implementations.")
