# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Optional, Tuple
import torch
from gfloat.types import FormatInfo, RoundMode, Domain

def _isodd(v: torch.Tensor) -> torch.Tensor:
    return (v & 0x1) == 1


def _ldexp(v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(v, s)


def _frexp(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.frexp returns (mantissa, exponent)
    # The original generic code returned (None, exponent) in its helper wrapper.
    # We return the standard torch tuple, the consumer selects index 1.
    return torch.frexp(v)


def round_ndarray(
    fi: FormatInfo,
    v: torch.Tensor,
    rnd: RoundMode = RoundMode.TiesToEven,
    sat: bool = False,
    srbits: Optional[torch.Tensor] = None,
    srnumbits: int = 0,
) -> torch.Tensor:
    """
    Vectorized version of :meth:`round_float`.

    Round inputs to the given :py:class:`FormatInfo`, given rounding mode and
    saturation flag

    Input NaNs will convert to NaNs in the target, not necessarily preserving payload.
    An input Infinity will convert to the largest float if :paramref:`sat`,
    otherwise to an Inf, if present, otherwise to a NaN.
    Negative zero will be returned if the format has negative zero, otherwise zero.

    Args:
      fi (FormatInfo): Describes the target format
      v (torch.Tensor): Input values to be rounded
      rnd (RoundMode): Rounding mode to use
      sat (bool): Saturation flag: if True, round overflowed values to `fi.max`
      srbits (torch.Tensor): Bits to use for stochastic rounding if rnd == Stochastic.
      srnumbits (int): How many bits are in srbits.  Implies srbits < 2**srnumbits.

    Returns:
      A tensor of floats which is a subset of the format's value set.

    Raises:
       ValueError: The target format cannot represent an input
             (e.g. converting a `NaN`, or an `Inf` when the target has no
             `NaN` or `Inf`, and :paramref:`sat` is false)
    """
    
    # Ensure inputs are tensors
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v)

    p = fi.precision
    bias = fi.bias

    is_negative = torch.signbit(v) & fi.is_signed
    absv = torch.where(is_negative, -v, v)

    finite_nonzero = ~(torch.isnan(v) | torch.isinf(v) | (v == 0))

    # Place 1.0 where finite_nonzero is False, to avoid log of {0,inf,nan}
    absv_masked = torch.where(finite_nonzero, absv, torch.tensor(1.0, dtype=v.dtype, device=v.device))

    # Determine integer type for intermediate calculations
    int_type = torch.int64 if fi.k > 8 or srnumbits > 8 else torch.int16

    def to_int(x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=int_type)

    def to_float(x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=v.dtype)

    # torch.frexp returns (mantissa, exponent)
    expval = _frexp(absv_masked)[1] - 1

    if fi.has_subnormals:
        expval = torch.maximum(expval, torch.tensor(1 - bias, dtype=expval.dtype, device=v.device))

    expval = expval - p + 1
    fsignificand = _ldexp(absv_masked, -expval)

    floorfsignificand = torch.floor(fsignificand)
    isignificand = to_int(floorfsignificand)
    delta = fsignificand - floorfsignificand

    if fi.precision > 1:
        code_is_odd = _isodd(isignificand)
    else:
        # Note: expval + bias might be float, cast to int for bitwise check
        code_is_odd = (isignificand != 0) & _isodd((expval + bias).to(torch.int64))

    # Rounding logic
    should_round_away: torch.Tensor
    
    if rnd == RoundMode.TowardZero:
        should_round_away = torch.zeros_like(delta, dtype=torch.bool)
        
    elif rnd == RoundMode.TowardPositive:
        should_round_away = ~is_negative & (delta > 0)
        
    elif rnd == RoundMode.TowardNegative:
        should_round_away = is_negative & (delta > 0)
        
    elif rnd == RoundMode.TiesToAway:
        should_round_away = delta >= 0.5
        
    elif rnd == RoundMode.TiesToEven:
        should_round_away = (delta > 0.5) | ((delta == 0.5) & code_is_odd)
        
    elif rnd == RoundMode.Stochastic:
        assert srbits is not None
        # RTNE delta to srbits
        d = delta * 2.0 ** float(srnumbits)
        floord = to_int(torch.floor(d))
        dd = d - torch.floor(d)
        should_round_away_tne = (dd > 0.5) | ((dd == 0.5) & _isodd(floord))
        drnd = floord + to_int(should_round_away_tne)

        should_round_away = drnd + srbits >= int(2.0 ** float(srnumbits))

    elif rnd == RoundMode.StochasticOdd:
        assert srbits is not None
        # RTNO delta to srbits
        d = delta * 2.0 ** float(srnumbits)
        floord = to_int(torch.floor(d))
        dd = d - torch.floor(d)
        should_round_away_tno = (dd > 0.5) | ((dd == 0.5) & ~_isodd(floord))
        drnd = floord + to_int(should_round_away_tno)

        should_round_away = drnd + srbits >= int(2.0 ** float(srnumbits))

    elif rnd == RoundMode.StochasticFast:
        assert srbits is not None
        should_round_away = (
            delta + to_float(2 * srbits + 1) * 2.0 ** -float(1 + srnumbits) >= 1.0
        )

    elif rnd == RoundMode.StochasticFastest:
        assert srbits is not None
        should_round_away = delta + to_float(srbits) * 2.0**-srnumbits >= 1.0
        
    else:
        # Fallback or error for unknown RoundMode
        should_round_away = torch.zeros_like(delta, dtype=torch.bool)

    isignificand = torch.where(should_round_away, isignificand + 1, isignificand)

    fresult = _ldexp(to_float(isignificand), expval)

    result = torch.where(finite_nonzero, fresult, absv)

    # Use tensors for min/max to allow broadcasting and device matching
    fi_min = torch.tensor(fi.min, device=v.device, dtype=v.dtype)
    fi_max = torch.tensor(fi.max, device=v.device, dtype=v.dtype)
    
    amax = torch.where(is_negative, -fi_min, fi_max)

    if sat:
        result = torch.where(result > amax, amax, result)
    else:
        put_amax_at: torch.Tensor
        if rnd == RoundMode.TowardNegative:
            put_amax_at = (result > amax) & ~is_negative
        elif rnd == RoundMode.TowardPositive:
            put_amax_at = (result > amax) & is_negative
        elif rnd == RoundMode.TowardZero:
            put_amax_at = result > amax
        else:
            put_amax_at = torch.zeros_like(result, dtype=torch.bool)

        result = torch.where(finite_nonzero & put_amax_at, amax, result)

        # Now anything larger than amax goes to infinity or NaN
        if fi.domain == Domain.Extended:
            result = torch.where(result > amax, torch.tensor(float('inf'), device=v.device, dtype=v.dtype), result)
        elif fi.num_nans > 0:
            result = torch.where(result > amax, torch.tensor(float('nan'), device=v.device, dtype=v.dtype), result)
        else:
            if torch.any(result > amax):
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    result = torch.where(is_negative, -result, result)

    # Make negative zeros negative if has_nz, else make them not negative.
    if fi.has_nz:
        result = torch.where((result == 0) & is_negative, -0.0, result)
    else:
        result = torch.where(result == 0, 0.0, result)

    return result