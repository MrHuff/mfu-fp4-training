#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import torchtitan.models.attention
import math
import sys
import os

# =============================================================================
# THE "SURGEON" PATCH: Direct ATen Operator Call (Production Version)
# =============================================================================

def _direct_flash_op(q, k, v, scale):
    return torch.ops.aten._scaled_dot_product_flash_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
        return_debug_mask=False,
        scale=scale,
    )[0]


def direct_pytorch_flash_forward(
    self,
    q,
    k,
    v,
    *,
    score_mod=None,
    scale=None,
    enable_gqa=False,
):
    """
    Bypasses the high-level dispatcher (F.sdpa) and calls the 
    PyTorch internal Flash Attention operator directly.
    Fixes GB200/CUDA13 cuDNN crashes by forcing contiguous memory.
    """
    if score_mod is not None:
        raise NotImplementedError(
            "attn_score_modifier requires use_flex_attn=True; "
            "the direct ATen flash-attention patch has no score_mod hook"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    # enable_gqa is ignored here; this low-level flash op path does not expose
    # the grouped-query attention surface from F.sdpa.
    try:
        if q.size(1) != k.size(1) or q.size(1) != v.size(1):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=scale,
                    enable_gqa=True,
                )
        # Prefer the original layouts first. On current GB200/CUDA13 this works
        # for both strided inputs and grouped-query attention shapes, and avoids
        # the unconditional q/k/v materialization that dominated copy time.
        return _direct_flash_op(q, k, v, scale)
    except (RuntimeError, AttributeError) as e:
        try:
            return _direct_flash_op(q.contiguous(), k.contiguous(), v.contiguous(), scale)
        except (RuntimeError, AttributeError) as fallback_e:
            e = fallback_e
        # Fallback mechanism only prints once per process to avoid log spam
        if not hasattr(self, "_warned_fallback"):
            sys.stdout.write(f"[Rank{torch.cuda.current_device()}] WARNING: Flash Op Failed. Fallback to Math. Error: {e}\n")
            self._warned_fallback = True

        if q.size(1) != k.size(1) and q.size(1) % k.size(1) == 0:
            repeats = q.size(1) // k.size(1)
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Pure Math Implementation (Slow but Safe)
        attn = (q @ k.transpose(-2, -1)) * scale
        L = q.size(-2)
        mask = torch.ones((L, L), device=q.device, dtype=torch.bool).tril(0)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return attn @ v

# =============================================================================
# APPLY PATCH
# =============================================================================
if os.environ.get("LOW_BITS_DISABLE_ATEN_FLASH_PATCH", "1") != "1":
    print("!!! APPLYING PRODUCTION PATCH: Using torch.ops.aten._scaled_dot_product_flash_attention !!!")
    torchtitan.models.attention.ScaledDotProductAttentionWrapper.forward = direct_pytorch_flash_forward
else:
    print("!!! SKIPPING PRODUCTION PATCH: using original ScaledDotProductAttentionWrapper.forward !!!")

_SDPA_BACKEND_ALIASES = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "cudnn_attention": SDPBackend.CUDNN_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "flash_attention": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "efficient_attention": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def _configure_sdpa_backend_order():
    requested = os.environ.get("LOW_BITS_SDPA_BACKENDS", "").strip()
    if not requested:
        return
    backends = []
    for item in requested.split(","):
        key = item.strip().lower()
        if not key:
            continue
        try:
            backends.append(_SDPA_BACKEND_ALIASES[key])
        except KeyError as exc:
            raise ValueError(
                f"Unknown LOW_BITS_SDPA_BACKENDS entry {item!r}; "
                f"valid keys are {sorted(_SDPA_BACKEND_ALIASES)}"
            ) from exc
    if not backends:
        return
    torchtitan.models.attention.ScaledDotProductAttentionWrapper.sdpa_backends = backends
    print(f"!!! LOW_BITS_SDPA_BACKENDS: {requested} -> {backends} !!!")


_configure_sdpa_backend_order()


# =============================================================================
# ORIGINAL IMPORTS
# =============================================================================
from torchtitan.models.llama3 import Transformer as Transformer
from torchtitan.models.deepseek_v3 import DeepSeekV3Model as DeepSeekV3Model

from .models import (
    TrainSpec as TrainSpec,
    get_train_spec as get_train_spec,
    get_model_config as get_model_config,
    add_model_config as add_model_config,
    BaseModelArgs as BaseModelArgs,
) 

from .models import TransformerModelArgs as TransformerModelArgs
from .llama3 import llama3_gc_configs as llama3_gc_configs
from .deepseek_v3 import deepseekv3_gc_configs as deepseekv3_gc_configs
from .nvpaper_transformer import (  # noqa: F401
    PaperTransformer as PaperTransformer,
    PaperTransformerModelArgs as PaperTransformerModelArgs,
)
from . import nemotron_h as nemotron_h  # noqa: F401

from .fuse_linear import (
    AttentionWithFusedLinear as AttentionWithFusedLinear,
    FusedLinearConverter as FusedLinearConverter,
    FeedForwardWithFusedLinear as FeedForwardWithFusedLinear,
)
from .spline_mlp import SplineMLPConverter as SplineMLPConverter
from .fa4_attention import FA4AttentionConverter as FA4AttentionConverter
