from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Union
import sys

import torch
from torch import nn

from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from low_bits_training.config import JobConfig


def _prepend_python_path(path: Path) -> None:
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _purge_stale_modules(prefix: str, expected_root: Path) -> None:
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue
        if not module_path.is_relative_to(expected_root):
            del sys.modules[name]


def _extend_fa4_python_paths() -> None:
    codebases_root = Path(__file__).resolve().parents[3]
    workspace_root = codebases_root.parents[1]
    lp_root = codebases_root / "low-precision-functions"
    flash_roots = [
        lp_root / "flash-attention",
        workspace_root / "low-precision-functions" / "flash-attention",
        workspace_root / "mfu_fp4" / "fp4_matmul" / "flash-attention",
    ]
    flash_root = next(
        (root for root in flash_roots if (root / "flash_attn" / "cute").is_dir()),
        flash_roots[0],
    )
    candidate_paths = [
        flash_root.parent,
        flash_root,
        flash_root / "csrc" / "cutlass" / "python" / "CuTeDSL",
    ]
    site_package_roots = [
        *codebases_root.glob("*/.venv*/lib/python*/site-packages"),
        *workspace_root.glob(".venv*/lib/python*/site-packages"),
        *workspace_root.glob("*/.venv*/lib/python*/site-packages"),
        *workspace_root.glob("*/*/.venv*/lib/python*/site-packages"),
    ]
    for site_packages in site_package_roots:
        candidate_paths.append(site_packages)
        candidate_paths.append(site_packages / "nvidia_cutlass_dsl" / "python_packages")
    for path in candidate_paths:
        _prepend_python_path(path)
    _purge_stale_modules("flash_attn", flash_root)


@dataclass(frozen=True)
class _ResolvedFA4Config:
    mode: str
    softcap: float = 0.0
    score_mod: Optional[Callable] = None
    score_mod_bwd: Optional[Callable] = None
    sigmoid_attention: bool = False
    sigmoid_sfu_freq: int = 16
    sigmoid_sfu_res: int = 0
    sigmoid_sfu_freq_bwd: int | None = None
    sigmoid_sfu_res_bwd: int | None = None
    sigmoid_use_direct_bwd_poly: bool = False
    sigmoid_bias: float | None = None
    sigmoid_poly_backend: str = "cute"
    sigmoid_qk_norm: bool = True


@lru_cache(maxsize=1)
def _load_fa4_runtime():
    _extend_fa4_python_paths()
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
    from flash_attn.cute.polynomial_manifest import run_polynomial_coefficient_audit
    from flash_attn.cute.utils import (
        create_softcap_scoremod_backend,
        create_softcap_scoremod_bwd_backend,
    )

    return {
        "_flash_attn_fwd": _flash_attn_fwd,
        "_flash_attn_bwd": _flash_attn_bwd,
        "run_polynomial_coefficient_audit": run_polynomial_coefficient_audit,
        "create_softcap_scoremod_backend": create_softcap_scoremod_backend,
        "create_softcap_scoremod_bwd_backend": create_softcap_scoremod_bwd_backend,
    }


def _resolve_fa4_config(job_config: JobConfig) -> _ResolvedFA4Config:
    runtime = _load_fa4_runtime()
    cfg = job_config.fa4
    if cfg.audit_coefficients:
        audited = runtime["run_polynomial_coefficient_audit"]()
        logger.info("FA4 polynomial audit passed: %s", ", ".join(audited))

    if cfg.mode == "softmax":
        return _ResolvedFA4Config(mode="softmax", sigmoid_qk_norm=False)

    if cfg.mode == "softcap":
        if cfg.softcap_backend == "native":
            return _ResolvedFA4Config(
                mode="softcap",
                softcap=float(cfg.softcap),
                sigmoid_qk_norm=False,
            )
        score_mod = runtime["create_softcap_scoremod_backend"](
            cfg.softcap,
            degree=cfg.softcap_degree,
            backend=cfg.softcap_backend,
        )
        score_mod_bwd = runtime["create_softcap_scoremod_bwd_backend"](
            cfg.softcap,
            degree=cfg.softcap_degree,
            backend=cfg.softcap_backend,
            backward_mode=cfg.softcap_backward_mode,
        )
        return _ResolvedFA4Config(
            mode="softcap",
            softcap=0.0,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            sigmoid_qk_norm=False,
        )

    if cfg.mode == "sigmoid_attention":
        if cfg.sigmoid_variant == "sfu":
            sfu_freq = 1
            sfu_res = 1
        else:
            sfu_freq = cfg.sigmoid_sfu_freq
            sfu_res = cfg.sigmoid_sfu_res
        return _ResolvedFA4Config(
            mode="sigmoid_attention",
            softcap=0.0,
            sigmoid_attention=True,
            sigmoid_sfu_freq=sfu_freq,
            sigmoid_sfu_res=sfu_res,
            sigmoid_sfu_freq_bwd=cfg.sigmoid_sfu_freq_bwd,
            sigmoid_sfu_res_bwd=cfg.sigmoid_sfu_res_bwd,
            sigmoid_use_direct_bwd_poly=(cfg.sigmoid_backward_mode == "direct"),
            sigmoid_bias=cfg.sigmoid_bias,
            sigmoid_poly_backend=cfg.sigmoid_poly_backend,
            sigmoid_qk_norm=cfg.sigmoid_qk_norm,
        )

    raise ValueError(f"Unsupported fa4.mode={cfg.mode!r}")


class _FA4Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        softcap,
        sigmoid_attention,
        sigmoid_sfu_freq,
        sigmoid_sfu_res,
        sigmoid_sfu_freq_bwd,
        sigmoid_sfu_res_bwd,
        sigmoid_use_direct_bwd_poly,
        sigmoid_bias,
        sigmoid_poly_backend,
        score_mod,
        score_mod_bwd,
    ):
        runtime = _load_fa4_runtime()
        if sigmoid_sfu_freq_bwd is None:
            sigmoid_sfu_freq_bwd = sigmoid_sfu_freq
        if sigmoid_sfu_res_bwd is None:
            sigmoid_sfu_res_bwd = sigmoid_sfu_res
        out, lse = runtime["_flash_attn_fwd"](
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap if score_mod is None else 0.0,
            score_mod=score_mod,
            sigmoid_attention=sigmoid_attention,
            sigmoid_sfu_freq=sigmoid_sfu_freq,
            sigmoid_sfu_res=sigmoid_sfu_res,
            sigmoid_bias=sigmoid_bias,
            sigmoid_poly_backend=sigmoid_poly_backend,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = softcap if score_mod is None else 0.0
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.sigmoid_attention = sigmoid_attention
        ctx.sigmoid_sfu_freq_bwd = sigmoid_sfu_freq_bwd
        ctx.sigmoid_sfu_res_bwd = sigmoid_sfu_res_bwd
        ctx.sigmoid_use_direct_bwd_poly = sigmoid_use_direct_bwd_poly
        ctx.sigmoid_bias = sigmoid_bias
        ctx.sigmoid_poly_backend = sigmoid_poly_backend
        return out

    @staticmethod
    def backward(ctx, dout):
        runtime = _load_fa4_runtime()
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = runtime["_flash_attn_bwd"](
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            score_mod=ctx.score_mod,
            score_mod_bwd=ctx.score_mod_bwd,
            sigmoid_attention=ctx.sigmoid_attention,
            sigmoid_bias=ctx.sigmoid_bias,
            sigmoid_sfu_freq=ctx.sigmoid_sfu_freq_bwd,
            sigmoid_sfu_res=ctx.sigmoid_sfu_res_bwd,
            sigmoid_use_direct_bwd_poly=ctx.sigmoid_use_direct_bwd_poly,
            sigmoid_poly_backend=ctx.sigmoid_poly_backend,
        )
        return dq, dk, dv, *((None,) * 13)


def _fa4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    config: _ResolvedFA4Config,
) -> torch.Tensor:
    return _FA4Func.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        config.softcap,
        config.sigmoid_attention,
        config.sigmoid_sfu_freq,
        config.sigmoid_sfu_res,
        config.sigmoid_sfu_freq_bwd,
        config.sigmoid_sfu_res_bwd,
        config.sigmoid_use_direct_bwd_poly,
        config.sigmoid_bias,
        config.sigmoid_poly_backend,
        config.score_mod,
        config.score_mod_bwd,
    )


class FA4AttentionWrapper(nn.Module):
    def __init__(self, config: _ResolvedFA4Config, head_dim: int):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        if config.sigmoid_attention and config.sigmoid_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        score_mod: Callable | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        if score_mod is not None:
            raise NotImplementedError(
                "FA4AttentionWrapper does not accept an external score_mod; configure low_bits_training.fa4 instead."
            )
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        target_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else q.dtype
        )
        out = _fa4_func(
            q.transpose(1, 2).contiguous().to(target_dtype),
            k.transpose(1, 2).contiguous().to(target_dtype),
            v.transpose(1, 2).contiguous().to(target_dtype),
            softmax_scale=scale,
            causal=True,
            config=self.config,
        )
        return out.transpose(1, 2)


def _patch_attention_modules(model: nn.Module, config: _ResolvedFA4Config) -> int:
    patched = 0
    for module in model.modules():
        if not hasattr(module, "inner_attention") or not hasattr(module, "head_dim"):
            continue
        if isinstance(module.inner_attention, FA4AttentionWrapper):
            continue
        wrapper = FA4AttentionWrapper(config=config, head_dim=module.head_dim)
        param = next(module.parameters(), None)
        if param is not None and param.device.type != "meta":
            wrapper = wrapper.to(device=param.device, dtype=param.dtype)
        elif param is not None:
            wrapper = wrapper.to(dtype=param.dtype)
        module.inner_attention = wrapper
        if hasattr(module, "use_flex_attn"):
            module.use_flex_attn = False
        if hasattr(module, "attn_score_modifier"):
            module.attn_score_modifier = None
        patched += 1
    return patched


class FA4AttentionConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.config = _resolve_fa4_config(job_config)

    def convert(self, model: nn.Module):
        patched = _patch_attention_modules(model, self.config)
        logger.info("Patched %d attention modules to FA4 mode=%s", patched, self.config.mode)

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass


register_model_converter(FA4AttentionConverter, "fa4_attention")
