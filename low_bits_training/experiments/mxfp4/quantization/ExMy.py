#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
from torchao.prototype.mx_formats.constants import (
    F32_MIN_NORMAL,
)
from gfloat.formats import (
    format_info_ocp_e4m3,
    format_info_ocp_e8m0,
    format_info_ocp_e5m2,
)
import gfloat


def debug_tensor(mx_tensor: torch.Tensor, tensor_name: str):
    if torch.isnan(mx_tensor).any().item():
        print(f"{tensor_name} has nan")


class ExMy_new:
    def __init__(
        self,
        e_bits: int,
        m_bits: int,
        roundMode: gfloat.RoundMode.TiesToEven = gfloat.RoundMode.TiesToEven,
    ):
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.roundMode = roundMode

        if e_bits == 8:
            self.format = format_info_ocp_e8m0
        if e_bits == 4:
            self.format = format_info_ocp_e4m3
        if e_bits == 5:
            self.format = format_info_ocp_e5m2

    def dummy_round(self, x):
        rounded = gfloat.round_ndarray(self.format, x, self.roundMode, sat=True)
        if self.format == format_info_ocp_e8m0:
            rounded = rounded.clip(2 ** (-127))
        else:
            rounded = rounded.clip(self.format.smallest_subnormal)
        return rounded

    def scale_by_format_max(
        self,
        x: torch.Tensor,
        target_max_pow2: int,
        target_max_mbits: int,
        g: torch.Tensor,
    ):
        if self.m_bits > 0:
            fp4_max = (2**target_max_pow2) * (
                1.0 + ((1 << target_max_mbits) - 1) / (1 << target_max_mbits)
            )
        else:
            fp4_max = 2**target_max_pow2
        fp4_max = torch.tensor(fp4_max).to(dtype=x.dtype, device=x.device)
        # dtype = x.dtype
        # finfo = torch.finfo(dtype)
        # max = finfo.max
        # Now scale x by dividing by fp4_max and then quantize.
        scaled = fp4_max / x * g
        rounded = gfloat.round_ndarray(self.format, scaled, self.roundMode, sat=True)
        if self.format == format_info_ocp_e8m0:
            rounded = rounded.clip(2 ** (-127))
        else:
            rounded = rounded.clip(self.format.smallest_subnormal)
        return rounded.to(x.dtype), scaled


class ExMy_old:
    def __init__(self, e_bits: int, m_bits: int, scale_clipping: bool):
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.bias = (1 << (e_bits - 1)) - 1  # Bias for exponent
        self.scale_clipping = scale_clipping

    def float_to_exmy(self, x: torch.Tensor):
        """Convert a tensor of floats to ExMy representation."""
        eps = F32_MIN_NORMAL * (x == 0).type(x.dtype)

        encoded = torch.zeros_like(x, dtype=torch.int32)

        # Handle zero explicitly
        zero_mask = x == 0
        if zero_mask.any():
            encoded[zero_mask] = 0

        # Get raw float components
        f_exp = torch.floor(torch.log2(torch.abs(x) + eps))  # Exponent
        f_mant = torch.abs(x) / (2**f_exp)  # Normalize to [1,2)

        # Encode exponent
        exp_biased = (f_exp + self.bias).clamp(0, (1 << self.e_bits) - 1).int()

        # Extract mantissa (remove leading 1 and scale)
        mantissa = ((f_mant - 1) * (1 << self.m_bits)).int()

        # Pack bits into integer representation
        encoded = (exp_biased << self.m_bits) | mantissa

        # Set zero elements correctly
        encoded[zero_mask] = 0

        return encoded

    def exmy_to_float(self, encoded: torch.Tensor):
        """Convert ExMy encoded tensor back to float."""
        exp_biased = (encoded >> self.m_bits) & (
            (1 << self.e_bits) - 1
        )  # Extract exponent
        mantissa = encoded & ((1 << self.m_bits) - 1)  # Extract mantissa bits

        # Decode exponent
        exponent = exp_biased - self.bias  # Remove bias

        # Decode mantissa
        mantissa_value = 1.0 + (mantissa / (1 << self.m_bits))  # Reconstruct value

        # Compute final value
        result = mantissa_value * (2**exponent)

        # Handle zero explicitly
        result[encoded == 0] = 0.0

        return result

    def float_to_exmy_float(self, x: torch.Tensor):
        # debug_tensor(x,'input tensor abs max ')
        dtype = x.dtype
        x_abs = x.abs()
        sign = x.sign()

        eps = torch.finfo(dtype).eps
        x_safe = torch.clamp(x_abs, min=eps)

        # Compute unbiased log2
        log2_x = torch.log2(x_safe)
        # debug_tensor(x,'post log2 ')

        if self.m_bits > 0:
            # Normal exponent and mantissa handling
            exp_unbiased = torch.floor(log2_x)
            mant = x_abs / (2.0**exp_unbiased)

            min_normal_exp = -self.bias + 1
            # is_subnormal = exp_unbiased < min_normal_exp
            exp_q = exp_unbiased.clamp(min_normal_exp, self.bias)

            mant_scaled = (mant - 1.0) * (1 << self.m_bits)
            mant_q = torch.round(mant_scaled).clamp(0, (1 << self.m_bits) - 1)
            mant_recon = 1.0 + mant_q / (1 << self.m_bits)

            normal_val = mant_recon * (2.0**exp_q)

            sub_val = torch.zeros_like(x)
            if self.m_bits > 0:
                sub_scale = 2 ** (min_normal_exp - self.m_bits)
                sub_mant = torch.round(x_abs / sub_scale).clamp(0, (1 << self.m_bits) - 1)
                sub_val = sub_mant * sub_scale

            result = torch.where(exp_unbiased < min_normal_exp, sub_val, normal_val)

        else:
            # m_bits == 0: exponent-only format like E8M0 — round to nearest power-of-two
            exp_q = torch.round(log2_x).clamp(-self.bias, self.bias)
            result = 2.0**exp_q
        # debug_tensor(x,'post mantissa')
        result = sign * result
        return result

    def scale_by_format_max(
        self, x: torch.Tensor, target_max_pow2: int, target_max_mbits: int
    ):
        """
        Scale the tensor x by FP4_max before quantizing it.

        For an FP4_max given by a format with target_max_pow2 and fp4_mbits,
          - if fp4_mbits > 0 (e.g., E2M1 or E0M8) the maximum is
               2^(target_max_pow2) * (1 + ((2^(fp4_mbits) - 1) / 2^(fp4_mbits)))
          - if fp4_mbits == 0 (e.g., E8M0), the maximum is simply 2^(target_max_pow2).
        """
        if self.m_bits > 0:
            fp4_max = (2**target_max_pow2) * (
                1.0 + ((1 << target_max_mbits) - 1) / (1 << target_max_mbits)
            )
        else:
            fp4_max = 2**target_max_pow2
        fp4_max = torch.tensor(fp4_max).to(dtype=x.dtype, device=x.device)
        # dtype = x.dtype
        # finfo = torch.finfo(dtype)
        # max = finfo.max
        # Now scale x by dividing by fp4_max and then quantize.
        scaled = fp4_max / x
        if self.scale_clipping:
            scaled = torch.clip(scaled, min=0.0019, max=100000)
        # The big true issue is essentially the gradient tensor, it can come it any goddamn values which will fuck up training if you screw up the signal!
        # OK problem found, this guy likes to diverge like crazy meaning x just becomes gradually smaller!
        # presumably we don't want to murder large values as well, so it would make sense to clip this to the smallest possible ExMy representation.
        # Basially, we need to truncate really small incoming gradient values! What happens now is that if we don't truncate them properly,
        # they'll get disproportionally scaled up since the quantisation itself induces an error, i.e. n(x)/q(n(x)).
        # E8M0 doesn't have this problem since it has very large range!!!
        # Introduce scale clipping as a particular feature for a tensor
        exmy_scaled = self.float_to_exmy_float(scaled)
        return exmy_scaled, scaled
