"""Fused square-ReLU helpers for benchmark FFN paths."""

from __future__ import annotations

import os

import torch


_SQRELU_FWD = None
_SQRELU_BWD = None


def _use_fused_sqrelu(x: torch.Tensor) -> bool:
    return (
        os.environ.get("LBT_USE_FUSED_SQRELU", "1") == "1"
        and x.is_cuda
        and hasattr(torch.cuda, "jiterator")
    )


def _get_sqrelu_fwd():
    global _SQRELU_FWD
    if _SQRELU_FWD is None:
        _SQRELU_FWD = torch.cuda.jiterator._create_jit_fn(
            """
            template <typename T> T sqrelu_fwd(T x) {
              float xf = float(x);
              float r = xf > 0.0f ? xf : 0.0f;
              return T(r * r);
            }
            """
        )
    return _SQRELU_FWD


def _get_sqrelu_bwd():
    global _SQRELU_BWD
    if _SQRELU_BWD is None:
        _SQRELU_BWD = torch.cuda.jiterator._create_jit_fn(
            """
            template <typename T> T sqrelu_bwd(T g, T x) {
              float xf = float(x);
              float r = xf > 0.0f ? xf : 0.0f;
              return T(2.0f * float(g) * r);
            }
            """
        )
    return _SQRELU_BWD


def sqrelu_fwd(x: torch.Tensor) -> torch.Tensor:
    if _use_fused_sqrelu(x):
        return _get_sqrelu_fwd()(x)
    return torch.relu(x).square()


def sqrelu_bwd(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if _use_fused_sqrelu(x):
        return _get_sqrelu_bwd()(grad, x)
    return (grad * torch.relu(x)).mul_(2.0)


class _SqReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return sqrelu_fwd(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return sqrelu_bwd(grad_output, x)


def sqrelu(x: torch.Tensor) -> torch.Tensor:
    return _SqReLUFunction.apply(x)
