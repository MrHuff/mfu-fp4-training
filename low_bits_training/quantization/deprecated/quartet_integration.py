
#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import sys
import os
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import re
import functools

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

# ── TE-native FP4 linear layers ──
from .fused_te_linear import TELinearFP4, FusedTELinearFP4


# Robust recursive setter
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, attr.split('.'), obj)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def _find_norm_for_linear(model, linear_name):
    """Find the RMSNorm that feeds a given linear layer in a Llama-style model.
    
    Llama TransformerBlock structure:
      layers.N.attention_norm → layers.N.attention.{wq, wk, wv}
      layers.N.ffn_norm       → layers.N.feed_forward.{w1, w3}
    
    Returns (norm_name, norm_module, eps) or None if no norm precedes this linear.
    """
    if re.search(r'layers\.\d+\.attention\.w[qkv]$', linear_name):
        norm_name = re.sub(r'\.attention\.w[qkv]$', '.attention_norm', linear_name)
    elif re.search(r'layers\.\d+\.feed_forward\.w[13]$', linear_name):
        norm_name = re.sub(r'\.feed_forward\.w[13]$', '.ffn_norm', linear_name)
    else:
        return None  # No norm for w2, wo, head, etc.
    
    try:
        norm_module = rgetattr(model, norm_name)
        if isinstance(norm_module, nn.RMSNorm):
            eps = norm_module.eps if hasattr(norm_module, 'eps') else 1e-5
            return (norm_name, norm_module, eps)
    except (AttributeError, KeyError):
        pass
    
    return None


class QuartetConverter(ModelConverter):
    """Converts nn.Linear layers to TE-native FP4 layers.
    
    - Layers preceded by RMSNorm (wq, wk, wv, w1, w3) → FusedTELinearFP4
    - Other layers (w2, wo) → TELinearFP4
    - Absorbed RMSNorm layers → nn.Identity
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.quartet_config = job_config.quartet
        self.parallel_dims = parallel_dims

    def convert(self, model: nn.Module):
        logger.info("Converting model to TE-native FP4 Linear layers...")
        
        modules_to_replace = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if "head" in name or "output" in name:
                     continue
                modules_to_replace.append((name, module))

        regular_count = 0
        
        logger.info(f"FP4: Swapping {len(modules_to_replace)} Linear layers.")
        
        for name, module in modules_to_replace:
            # Use standard TELinearFP4 for all layers (no fused norm for now)
            q_layer = TELinearFP4(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=(module.bias is not None),
                dtype=torch.bfloat16,
                device=module.weight.device,
            )
            
            with torch.no_grad():
                q_layer.weight.copy_(module.weight.to(torch.bfloat16))
                if module.bias is not None:
                    q_layer.bias.copy_(module.bias.to(torch.bfloat16))
            
            rsetattr(model, name, q_layer)
            regular_count += 1
        
        logger.info(f"FP4 conversion complete: {regular_count} TELinearFP4 layers.")

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """Invalidate cached quantized weights after optimizer step."""
        models = model if isinstance(model, list) else [model]
        for m in models:
            for module in m.modules():
                if isinstance(module, TELinearFP4):
                    module.invalidate_weight_cache()

register_model_converter(QuartetConverter, "quartet")
