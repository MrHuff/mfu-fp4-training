#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch
from gfloat.types import FormatInfo, RoundMode


def _isodd(v: torch.Tensor) -> torch.Tensor:
    return (v & 0x1) == 1


def _ldexp(v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # Workaround torch not having ldexp
    offset = 24
    vlo = (v * 2.0**+offset) * torch.pow(2.0, s.to(v.dtype) - offset)
    vhi = (v * 2.0**-offset) * torch.pow(2.0, s.to(v.dtype) + offset)
    return torch.where(v < 1.0, vlo, vhi)


def round_ndarray(
    fi: FormatInfo,
    v: torch.Tensor,
    rnd: RoundMode = RoundMode.TiesToEven,
    sat: bool = False,
    srbits: torch.Tensor | None = None,
    srnumbits: int = 0,
) -> torch.Tensor:
    p = fi.precision
    bias = fi.expBias

    is_negative = torch.signbit(v) & fi.is_signed
    absv = torch.where(is_negative, -v, v)

    finite_nonzero = ~(torch.isnan(v) | torch.isinf(v) | (v == 0))
    absv_masked = torch.where(
        finite_nonzero, absv, torch.tensor(1.0, device=v.device, dtype=v.dtype)
    )

    int_type = torch.int64 if fi.k > 8 or srnumbits > 8 else torch.int16

    def to_int(x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=int_type)

    def to_float(x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=v.dtype)

    expval = to_int(torch.floor(torch.log2(absv_masked)))

    if fi.has_subnormals:
        expval = torch.maximum(
            expval, torch.tensor(1 - bias, device=v.device, dtype=int_type)
        )

    expval = expval - p + 1
    fsignificand = _ldexp(absv_masked, -expval)

    floorfsignificand = torch.floor(fsignificand)
    isignificand = to_int(floorfsignificand)
    delta = fsignificand - floorfsignificand

    if fi.precision > 1:
        code_is_odd = _isodd(isignificand)
    else:
        code_is_odd = (isignificand != 0) & _isodd(expval + bias)

    match rnd:
        case RoundMode.TowardZero:
            should_round_away = torch.zeros_like(delta, dtype=torch.bool)

        case RoundMode.TowardPositive:
            should_round_away = ~is_negative & (delta > 0)

        case RoundMode.TowardNegative:
            should_round_away = is_negative & (delta > 0)

        case RoundMode.TiesToAway:
            should_round_away = delta >= 0.5

        case RoundMode.TiesToEven:
            should_round_away = (delta > 0.5) | ((delta == 0.5) & code_is_odd)

        case RoundMode.Stochastic:
            assert srbits is not None
            d = delta * 2.0 ** float(srnumbits)
            floord = to_int(torch.floor(d))
            dd = d - torch.floor(d)
            should_round_away_tne = (dd > 0.5) | ((dd == 0.5) & _isodd(floord))
            drnd = floord + should_round_away_tne.to(dtype=floord.dtype)
            should_round_away = drnd + srbits >= int(2.0 ** float(srnumbits))

        case RoundMode.StochasticOdd:
            assert srbits is not None
            d = delta * 2.0 ** float(srnumbits)
            floord = to_int(torch.floor(d))
            dd = d - torch.floor(d)
            should_round_away_tno = (dd > 0.5) | ((dd == 0.5) & ~_isodd(floord))
            drnd = floord + should_round_away_tno.to(dtype=floord.dtype)
            should_round_away = drnd + srbits >= int(2.0 ** float(srnumbits))

        case RoundMode.StochasticFast:
            assert srbits is not None
            should_round_away = (
                delta + to_float(2 * srbits + 1) * 2.0 ** -float(1 + srnumbits) >= 1.0
            )

        case RoundMode.StochasticFastest:
            assert srbits is not None
            should_round_away = delta + to_float(srbits) * 2.0**-srnumbits >= 1.0

    isignificand = torch.where(should_round_away, isignificand + 1, isignificand)

    fresult = _ldexp(to_float(isignificand), expval)
    result = torch.where(finite_nonzero, fresult, absv)

    amax = torch.where(is_negative, -fi.min, fi.max)

    if sat:
        result = torch.where(result > amax, amax, result)
    else:
        match rnd:
            case RoundMode.TowardNegative:
                put_amax_at = (result > amax) & ~is_negative
            case RoundMode.TowardPositive:
                put_amax_at = (result > amax) & is_negative
            case RoundMode.TowardZero:
                put_amax_at = result > amax
            case _:
                put_amax_at = torch.zeros_like(result, dtype=torch.bool)

        result = torch.where(finite_nonzero & put_amax_at, amax, result)

        if fi.has_infs:
            result = torch.where(
                result > amax, torch.tensor(torch.inf, device=result.device), result
            )
        elif fi.num_nans > 0:
            result = torch.where(
                result > amax, torch.tensor(torch.nan, device=result.device), result
            )
        else:
            if torch.any(result > amax):
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    result = torch.where(is_negative, -result, result)

    if fi.has_nz:
        result = torch.where(
            (result == 0) & is_negative, torch.tensor(-0.0, device=result.device), result
        )
    else:
        result = torch.where(result == 0, torch.tensor(0.0, device=result.device), result)

    return result
