"""Pure-CUDA production operators for Nemotron-H Mamba training.

This module deliberately does not import ``mamba_ssm.ops.triton``.  It uses the
compiled selective-scan and causal-convolution CUDA extensions plus the native
gated group-RMSNorm extension maintained in fp4_matmul.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys

import torch


_selective_scan_cuda = None
_causal_conv1d_fn = None
_gated_rmsnorm_cuda = None


def _cuda_site_packages() -> str:
    path = os.environ.get("LBT_NEMOTRON_H_CUDA_SITE_PACKAGES", "").strip()
    if not path:
        raise RuntimeError("LBT_NEMOTRON_H_CUDA_SITE_PACKAGES is not set")
    if not os.path.isdir(path):
        raise RuntimeError(f"Nemotron CUDA site-packages does not exist: {path}")
    return path


def _load_native_extensions() -> None:
    global _selective_scan_cuda, _causal_conv1d_fn, _gated_rmsnorm_cuda
    if _selective_scan_cuda is not None:
        return

    site_packages = _cuda_site_packages()
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    # selective_scan_cuda is a compiled CUDA extension. Import it directly so
    # importing this module can never pull in mamba-ssm's Triton Python layers.
    import selective_scan_cuda  # type: ignore
    from causal_conv1d import causal_conv1d_fn  # type: ignore

    root = os.environ.get("FP4_MATMUL_ROOT", "").strip()
    if not root:
        raise RuntimeError("FP4_MATMUL_ROOT is not set")
    candidates = glob.glob(
        os.path.join(root, "TK_quantisation", "mamba_cuda", "_nemotron_mamba_cuda*.so")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one built _nemotron_mamba_cuda extension under "
            f"{root}, found {candidates}"
        )
    spec = importlib.util.spec_from_file_location("_nemotron_mamba_cuda", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load native Nemotron extension: {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _selective_scan_cuda = selective_scan_cuda
    _causal_conv1d_fn = causal_conv1d_fn
    _gated_rmsnorm_cuda = module


def native_mamba_available() -> tuple[bool, str]:
    try:
        _load_native_extensions()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def causal_conv1d_cuda(x, weight, bias, activation):
    _load_native_extensions()
    return _causal_conv1d_fn(x=x, weight=weight, bias=bias, activation=activation)


class _MambaConvOutputSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, hidden_size, group_state_size):
        ctx.input_shape = tensor.shape
        ctx.hidden_size = hidden_size
        ctx.group_state_size = group_state_size
        return tensor.split(
            [hidden_size, group_state_size, group_state_size],
            dim=-1,
        )

    @staticmethod
    def backward(ctx, grad_hidden, grad_B, grad_C):
        gradients = (grad_hidden, grad_B, grad_C)
        storage_ptrs = {
            gradient.untyped_storage().data_ptr() for gradient in gradients
        }
        adjacent = (
            len(storage_ptrs) == 1
            and grad_B.storage_offset()
            == grad_hidden.storage_offset() + ctx.hidden_size
            and grad_C.storage_offset()
            == grad_B.storage_offset() + ctx.group_state_size
            and grad_hidden.stride()[:-1] == grad_B.stride()[:-1]
            and grad_hidden.stride()[:-1] == grad_C.stride()[:-1]
            and grad_hidden.stride(-1) == 1
            and grad_B.stride(-1) == 1
            and grad_C.stride(-1) == 1
        )
        if adjacent:
            combined_stride = grad_hidden.stride()[:-1] + (1,)
            grad_tensor = grad_hidden.as_strided(
                ctx.input_shape,
                combined_stride,
                grad_hidden.storage_offset(),
            )
        else:
            grad_tensor = torch.cat(gradients, dim=-1)
        return grad_tensor, None, None


def split_mamba_conv_output(tensor, hidden_size, group_state_size):
    """Split the production conv output while preserving adjacent native grads."""
    if os.environ.get("LBT_NEMOTRON_H_ADJACENT_CONV_GRADS", "1") == "0":
        return tensor.split(
            [hidden_size, group_state_size, group_state_size],
            dim=-1,
        )
    return _MambaConvOutputSplit.apply(tensor, hidden_size, group_state_size)


class _SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias):
        _load_native_extensions()
        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        D = D.contiguous()
        delta_bias = delta_bias.contiguous()
        out, intermediates, *_ = _selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, delta_bias, True
        )
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, intermediates)
        return out

    @staticmethod
    def backward(ctx, dout):
        u, delta, A, B, C, D, delta_bias, intermediates = ctx.saved_tensors
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = _selective_scan_cuda.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            None,
            delta_bias,
            dout.contiguous(),
            intermediates,
            None,
            None,
            True,
            False,
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias


def _expanded_selective_scan_inputs(x, dt, A, B, C, D, dt_bias):
    batch, length, heads, head_dim = x.shape
    state = B.shape[-1]
    channels = heads * head_dim
    u = x.permute(0, 2, 3, 1).reshape(batch, channels, length).contiguous()
    delta = (
        dt.permute(0, 2, 1)
        .unsqueeze(2)
        .expand(batch, heads, head_dim, length)
        .reshape(batch, channels, length)
        .contiguous()
    )
    scan_A = (
        A.reshape(heads, 1, 1)
        .expand(heads, head_dim, state)
        .reshape(channels, state)
        .contiguous()
    )
    scan_B = B.permute(0, 2, 3, 1).contiguous()
    scan_C = C.permute(0, 2, 3, 1).contiguous()
    scan_D = D.float().repeat_interleave(head_dim).contiguous()
    scan_bias = dt_bias.float().repeat_interleave(head_dim).contiguous()
    return u, delta, scan_A, scan_B, scan_C, scan_D, scan_bias


class _Mamba2CutlassSsd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D, dt_bias):
        _load_native_extensions()
        ctx.grouped_backward = (
            os.environ.get("LBT_NEMOTRON_H_GROUPED_SSD_BWD", "1") == "1"
        )
        ctx.adjacent_conv_grads = (
            os.environ.get("LBT_NEMOTRON_H_ADJACENT_CONV_GRADS", "1") == "1"
        )
        if ctx.grouped_backward:
            output = _gated_rmsnorm_cuda.mamba2_ssd_fwd_grouped(
                x, dt, A, B, C, D, dt_bias
            )
            ctx.save_for_backward(x, dt, A, B, C, D, dt_bias)
        else:
            output, checkpoints = _gated_rmsnorm_cuda.mamba2_ssd_fwd(
                x, dt, A, B, C, D, dt_bias
            )
            ctx.save_for_backward(x, dt, A, B, C, D, dt_bias, checkpoints)
        return output

    @staticmethod
    def backward(ctx, dout):
        if ctx.grouped_backward:
            x, dt, A, B, C, D, dt_bias = ctx.saved_tensors
            grouped_bwd = (
                _gated_rmsnorm_cuda.mamba2_grouped_bwd_adjacent
                if ctx.adjacent_conv_grads
                else _gated_rmsnorm_cuda.mamba2_grouped_bwd
            )
            return tuple(
                grouped_bwd(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    dt_bias,
                    dout.contiguous(),
                )
            )

        x, dt, A, B, C, D, dt_bias, checkpoints = ctx.saved_tensors
        u, delta, scan_A, scan_B, scan_C, scan_D, scan_bias = (
            _expanded_selective_scan_inputs(x, dt, A, B, C, D, dt_bias)
        )
        du, ddelta, dA, dB, dC, dD, ddelta_bias = (
            _gated_rmsnorm_cuda.mamba2_scan_bwd_256x8(
                u,
                delta,
                scan_A,
                scan_B,
                scan_C,
                scan_D,
                scan_bias,
                dout.permute(0, 2, 3, 1)
                .reshape_as(u)
                .contiguous(),
                checkpoints,
            )
        )
        return tuple(
            _gated_rmsnorm_cuda.mamba2_scan_bwd_collapse(
                du,
                ddelta,
                dA,
                dB,
                dC,
                dD,
                ddelta_bias,
                D,
                dt_bias,
            )
        )


def selective_scan_mamba2_cutlass(x, dt, A, B, C, D, dt_bias):
    """Production Blackwell Mamba2 SSD with CUTLASS forward and CUDA backward."""
    if os.environ.get("LBT_NEMOTRON_H_STRIDED_INPUTS", "1") == "0":
        return _Mamba2CutlassSsd.apply(
            x.contiguous(),
            dt.contiguous(),
            A.contiguous(),
            B.contiguous(),
            C.contiguous(),
            D.contiguous(),
            dt_bias.contiguous(),
        )
    return _Mamba2CutlassSsd.apply(x, dt, A, B, C, D, dt_bias)


def selective_scan_mamba2(x, dt, A, B, C, D, dt_bias):
    """Run grouped Mamba2 through the selected native CUDA implementation.

    x: [batch, length, heads, head_dim]
    dt: [batch, length, heads]
    B/C: [batch, length, groups, state]
    """
    if os.environ.get("LBT_NEMOTRON_H_CUTLASS_SSD", "0") == "1":
        return selective_scan_mamba2_cutlass(x, dt, A, B, C, D, dt_bias)

    batch, length, heads, head_dim = x.shape
    u, delta, scan_A, scan_B, scan_C, scan_D, scan_bias = (
        _expanded_selective_scan_inputs(x, dt, A, B, C, D, dt_bias)
    )
    out = _SelectiveScan.apply(u, delta, scan_A, scan_B, scan_C, scan_D, scan_bias)
    return out.reshape(batch, heads, head_dim, length).permute(0, 3, 1, 2)


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps):
        _load_native_extensions()
        native_layout = (
            x.shape == gate.shape
            and x.dim() in (2, 3)
            and x.stride(-1) == 1
            and gate.stride(-1) == 1
        )
        if (
            os.environ.get("LBT_NEMOTRON_H_STRIDED_INPUTS", "1") == "0"
            or not native_layout
        ):
            x = x.contiguous()
            gate = gate.contiguous()
            weight = weight.contiguous()
        out, inv = _gated_rmsnorm_cuda.gated_rmsnorm_fwd(x, gate, weight, eps)
        ctx.save_for_backward(x, gate, weight, inv)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, gate, weight, inv = ctx.saved_tensors
        dx, dgate, dweight = _gated_rmsnorm_cuda.gated_rmsnorm_bwd(
            dout.contiguous(), x, gate, weight, inv
        )
        return dx, dgate, dweight, None


def gated_rmsnorm_cuda(x, gate, weight, eps, group_size):
    if group_size != 1024:
        raise RuntimeError(f"native Nemotron gated RMSNorm requires group_size=1024, got {group_size}")
    return _GatedRMSNorm.apply(x, gate, weight, eps)


def gated_rmsnorm_backward_cuda(dout, x, gate, weight, inv):
    """Run the native backward for a fused gated-RMS producer."""
    _load_native_extensions()
    return _gated_rmsnorm_cuda.gated_rmsnorm_bwd(
        dout.contiguous(), x, gate, weight, inv
    )
