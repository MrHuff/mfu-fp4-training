#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
from typing import Any, Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn
import re
import functools
import os
from contextlib import nullcontext

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

# --- TRANSFORMER ENGINE IMPORTS ---
import transformer_engine.pytorch as te
try:
    import transformer_engine.pytorch.module.linear as te_linear
    import transformer_engine.pytorch.module.base as te_base
except (ModuleNotFoundError, ImportError):
    te_linear = None
    te_base = None
try:
    from transformer_engine.common.recipe import Format, NVFP4BlockScaling, MXFP4BlockScaling
except ImportError:
    try:
        from transformer_engine.common.recipe import Format, NVFP4BlockScaling
    except ImportError:
        NVFP4BlockScaling = None
    MXFP4BlockScaling = None

# --- [FIX 1] Robust Recursive Attribute Setters/Getters ---
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, attr.split('.'), obj)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
# ----------------------------------------------------------

class BoundRecipeLinear(te.Linear):
    """
    A te.Linear layer that carries its own specific FP8/FP4 recipe.
    """
    def __init__(self, in_features, out_features, bias=True, params_dtype=torch.bfloat16, recipe=None, device=None):
        # Passed device to super to handle Meta device initialization correctly
        super().__init__(in_features, out_features, bias=bias, params_dtype=params_dtype, device=device)
        self.bound_recipe = recipe

    def forward(self, inp):
        if inp.dtype != torch.bfloat16:
            inp = inp.to(torch.bfloat16)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if inp.is_cuda and self.weight.dtype != inp.dtype
            else nullcontext()
        )
        if self.bound_recipe is not None:
            with amp_ctx, te.fp8_autocast(enabled=True, fp8_recipe=self.bound_recipe):
                return super().forward(inp)
        else:
            with amp_ctx:
                return super().forward(inp)


def granular_replace_linear(
    model: nn.Module, 
    num_hidden_layers: int,
    mlp_recipe: Optional[Any],       
    attn_recipe: Optional[Any],      
    mamba_recipe: Optional[Any] = None,
    exclude_last_n_layers: int = 0,
    exclude_qkv: bool = False,
    quantize_mamba: bool = True,
    verbose: bool = True
):
    """
    Replaces Linear layers with BoundRecipeLinear.
    Leaves final 'output'/'lm_head' layers untouched; FP4 swaps are only for
    internal transformer MLP/attention linears.
    """
    
    modules_to_replace = []
    
    if num_hidden_layers is None: 
        num_hidden_layers = 0

    cutoff_index = num_hidden_layers - exclude_last_n_layers
    
    # Keywords for both standard Llama and Meta/Titan
    mlp_keywords = [
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
        "mixer.up_proj", "mixer.down_proj",
    ]
    
    attn_keywords = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "attention.wq", "attention.wk", "attention.wv", "attention.wo",
        "mixer.q_proj", "mixer.k_proj", "mixer.v_proj", "mixer.o_proj",
    ]

    qkv_keywords = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "attention.wq", "attention.wk", "attention.wv",
        "mixer.q_proj", "mixer.k_proj", "mixer.v_proj",
    ]

    mamba_keywords = [
        "mixer.in_proj",
        "mixer.out_proj",
    ] if quantize_mamba else []
    
    head_keywords = ["output", "lm_head"]

    if verbose:
        logger.info(f"TE-FP4 Scan: MLP Strategy: {mlp_recipe.__class__.__name__ if mlp_recipe else 'None'}")
        logger.info(f"TE-FP4 Scan: Attn Strategy: {attn_recipe.__class__.__name__ if attn_recipe else 'None'}")
        logger.info(f"TE-FP4 Scan: Total layers {num_hidden_layers}. Swapping up to layer {cutoff_index}.")

    layer_idx_pattern = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
            
        layer_idx = -1
        match = layer_idx_pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
        
        selected_recipe = None
        is_mlp = any(k in name for k in mlp_keywords)
        is_attn = any(k in name for k in attn_keywords)
        is_qkv = any(k in name for k in qkv_keywords)
        is_mamba = any(k in name for k in mamba_keywords)
        is_head = any(k in name for k in head_keywords)

        if layer_idx != -1 and layer_idx < cutoff_index:
            if is_mlp:
                selected_recipe = mlp_recipe
            elif is_attn and not (exclude_qkv and is_qkv):
                selected_recipe = attn_recipe
            elif is_mamba:
                selected_recipe = mamba_recipe
        
        if selected_recipe is not None and not is_head:
            modules_to_replace.append((name, module, selected_recipe))

    if verbose:
        logger.info(f"Transformer Engine: Swapping {len(modules_to_replace)} internal modules to BoundRecipeLinear.")
    
    # 4. Execute Swap
    for name, module, recipe in modules_to_replace:
        # Capture the device of the source module (likely 'meta')
        target_device = module.weight.device
        
        te_layer = BoundRecipeLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=(module.bias is not None),
            params_dtype=module.weight.dtype,
            recipe=recipe,
            device=target_device,
        )
        
        # Only copy weights if we have actual data. 
        # If device is 'meta', we skip copy (TorchTitan will init weights later).
        if target_device.type != 'meta':
            with torch.no_grad():
                te_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    te_layer.bias.copy_(module.bias)
        
        # Use the robust recursive setter
        rsetattr(model, name, te_layer)

    return model


class TEFP4Converter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.mlp_recipe_name = job_config.te_fp4.mlp_recipe
        self.attn_recipe_name = job_config.te_fp4.attn_recipe
        self.mamba_recipe_name = job_config.te_fp4.mamba_recipe
        self.exclude_last_n = job_config.te_fp4.exclude_last_n_layers
        self.exclude_qkv = job_config.te_fp4.exclude_qkv
        self.quantize_mamba = job_config.te_fp4.quantize_mamba

    def _get_recipe_obj(self, name: str):
        if name == "NVFP4":
            return NVFP4BlockScaling()
        elif name == "MXFP4":
            return MXFP4BlockScaling(
                disable_rht=os.getenv("NVTE_NVFP4_DISABLE_RHT", "0") == "1",
                disable_stochastic_rounding=os.getenv("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "0") == "1",
                disable_2d_quantization=os.getenv("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "0") == "1",
                encode=True,
            )
        return None

    def convert(self, model: nn.Module):
        logger.info("Converting model using Transformer Engine (TE) FP4 recipes...")
        
        recipe_mlp = self._get_recipe_obj(self.mlp_recipe_name)
        print(recipe_mlp)
        recipe_attn = self._get_recipe_obj(self.attn_recipe_name)
        print(recipe_attn)
        recipe_mamba = (
            recipe_mlp
            if not self.mamba_recipe_name
            else self._get_recipe_obj(self.mamba_recipe_name)
        )
        num_hidden_layers = None

        # 1. Try Job Config
        if self.job_config.model.n_layers is not None:
             num_hidden_layers = self.job_config.model.n_layers

        # 2. Try Model Config
        if num_hidden_layers is None and hasattr(model, "config"):
            if hasattr(model.config, "num_hidden_layers"):
                num_hidden_layers = model.config.num_hidden_layers
            elif hasattr(model.config, "n_layers"):
                num_hidden_layers = model.config.n_layers

        # 3. Fallback: Auto-detect
        if num_hidden_layers is None:
            max_idx = 0
            pattern = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")
            for name, _ in model.named_modules():
                match = pattern.search(name)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
            if max_idx > 0:
                num_hidden_layers = max_idx + 1
                logger.info(f"TE-FP4: Auto-detected {num_hidden_layers} layers via module scan.")

        # 4. Last Resort
        if num_hidden_layers is None:
            flavor = self.job_config.model.flavor
            logger.warning(f"Could not determine layers for flavor {flavor}. Defaulting to 32.")
            num_hidden_layers = 32

        logger.info(f"TE-FP4: Using {num_hidden_layers} hidden layers for swap logic.")

        granular_replace_linear(
            model,
            num_hidden_layers=num_hidden_layers,
            mlp_recipe=recipe_mlp,
            attn_recipe=recipe_attn,
            mamba_recipe=recipe_mamba,
            exclude_last_n_layers=self.exclude_last_n,
            exclude_qkv=self.exclude_qkv,
            quantize_mamba=self.quantize_mamba,
            verbose=True
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass

register_model_converter(TEFP4Converter, "te_fp4")
