#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
"""
Custom Model Converter for torchtitan using TEParityLinear.
"""

from typing import Any
import os
import re
import torch
import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger
from low_bits_training.quantization.fusedMXFPMatmul import MXLinearDimFused
from low_bits_training.quantization.fused_quant_triton_v3 import TritonFusedQuantLinearV3
from low_bits_training.quantization.fused_quant_triton_v2 import TritonFusedQuantLinear

from low_bits_training.quantization.MXFPconfig import MXLinearDimConfig
from low_bits_training.quantization.float32_linear import Float32Linear
import gfloat

USE_OLD_LAYER = False


def granular_replace_linear(
    model: nn.Module,
    num_hidden_layers: int,
    mx_config: Any,
    mlp_recipe: Any = "Custom",
    attn_recipe: Any = "Custom",
    mamba_recipe: Any = None,
    exclude_last_n_layers: int = 0,
    exclude_qkv: bool = False,
    quantize_mamba: bool = True,
    verbose: bool = True,
):
    """
    Replaces Linear layers with TEParityLinear (in-place modification).
    Also replaces 'output'/'lm_head' with Float32Linear to force FP32 precision.
    """
    modules_to_modify = []

    if num_hidden_layers is None:
        num_hidden_layers = 0

    cutoff_index = num_hidden_layers - exclude_last_n_layers

    # Target specific layers similar to previous logic
    mlp_keywords = [
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "mixer.up_proj",
        "mixer.down_proj",
    ]

    attn_keywords = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "attention.wo",
        "mixer.q_proj",
        "mixer.k_proj",
        "mixer.v_proj",
        "mixer.o_proj",
    ]

    qkv_keywords = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "mixer.q_proj",
        "mixer.k_proj",
        "mixer.v_proj",
    ]

    mamba_keywords = [
        "mixer.in_proj",
        "mixer.out_proj",
    ] if quantize_mamba else []

    head_keywords = ["output", "lm_head"]

    if verbose:
        logger.info(f"Custom TE Parity Replacement Started. Config: {mx_config}")

    def recipe_enabled(recipe: Any) -> bool:
        if recipe is None:
            return False
        if isinstance(recipe, str):
            return recipe.lower() not in {"", "none", "false", "off"}
        return bool(recipe)

    quantize_mlp = recipe_enabled(mlp_recipe)
    quantize_attn = recipe_enabled(attn_recipe)
    if mamba_recipe is None or mamba_recipe == "":
        mamba_recipe = mlp_recipe
    quantize_mamba_recipe = recipe_enabled(mamba_recipe)

    for name, module in model.named_modules():
        # Check if linear
        if not isinstance(module, torch.nn.Linear):
            continue

        # Check layer index
        layer_idx = -1

        # Heuristic for layer index: look for .layers.N.
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break

        # Rules:
        # 1. Must be in mlp or attn keywords
        is_target_module = any(
            k in name for k in mlp_keywords + attn_keywords + mamba_keywords
        )
        is_mlp_module = any(k in name for k in mlp_keywords)
        is_attn_module = any(k in name for k in attn_keywords)
        is_mamba_module = any(k in name for k in mamba_keywords)
        is_qkv_module = any(k in name for k in qkv_keywords)
        is_head_module = any(k in name for k in head_keywords)

        # 2. Must be within layer range [0, cutoff_index)
        is_in_range = layer_idx >= 0 and layer_idx < cutoff_index

        recipe_allows_module = (
            (is_mlp_module and quantize_mlp)
            or (is_attn_module and quantize_attn)
            or (is_mamba_module and quantize_mamba_recipe)
        )
        should_quantize = (
            is_target_module
            and recipe_allows_module
            and is_in_range
            and not (exclude_qkv and is_qkv_module)
        )

        if should_quantize or is_head_module:
            modules_to_modify.append((name, module, is_head_module))

    for name, module, is_head in modules_to_modify:
        if verbose:
            logger.info(
                f"Converting: {name} | Shape: {module.weight.shape} | Head: {is_head}"
            )

        if is_head:
            # Replace with Float32Linear
            new_layer = Float32Linear(
                module.in_features, module.out_features, bias=module.bias is not None
            )
            new_layer = new_layer.to(module.weight.device).to(module.weight.dtype)
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)
        else:
            # Create new layer
            if USE_OLD_LAYER:
                use_approx_default = {
                    "stepGradient": "STE",
                    "smooth": "STE",
                    "qGradient": "STE",
                    "use_tensor_scaling": True,
                    "SR": "None_exact",
                    "tensor_scaling_grad_est": "ignore",
                    "use_hadamard": "None_exact",
                    "dtype": torch.bfloat16,
                }

                # Map string roundMode to gfloat enum
                rm_str = getattr(mx_config, "roundMode", "TiesToEven")
                if rm_str == "TiesToEven":
                    rm = gfloat.RoundMode.TiesToEven
                elif rm_str == "TowardPositive":
                    rm = gfloat.RoundMode.TowardPositive
                elif rm_str == "TowardNegative":
                    rm = gfloat.RoundMode.TowardNegative
                elif rm_str == "TowardZero":
                    rm = gfloat.RoundMode.TowardZero
                else:
                    rm = gfloat.RoundMode.TiesToEven

                mx_dim_config = MXLinearDimConfig(
                    block_size=getattr(mx_config, "block_size", 32),
                    elem_dtype=getattr(mx_config, "elem_dtype", "fp4_e2m1"),
                    scale_type=getattr(mx_config, "scale_type", "E8M0"),
                    strategy=getattr(mx_config, "strategy", "encode"),
                    use_fp32_scaling=True,
                    roundMode=rm,
                    use_approx=use_approx_default,
                    gemm_kernel_choice="emulated",
                    nan_handling_mode="to_one",
                )
                new_layer = MXLinearDimFused.from_float(module, mx_dim_config)
            else:
                # Pass the config object directly, TEParityLinear handles extracting flags
                new_layer = TritonFusedQuantLinear.from_float(module, mx_config)

        # Replace in parent
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.rsplit(".", 1)[-1]

        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model

        setattr(parent, attr_name, new_layer)

    if verbose:
        logger.info(f"Replaced {len(modules_to_modify)} layers with TEParityLinear.")


class TECustomConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.cfg = job_config.mxfp_custom

        self.exclude_last_n = getattr(self.cfg, "exclude_last_n_layers", 0)
        self.verbose = getattr(self.cfg, "verbose", True)

    def convert(self, model: nn.Module):
        logger.info("Converting model to Custom TE Parity Linear...")

        # Robust Layer Counting
        num_hidden_layers = None
        if self.job_config.model.n_layers is not None:
            num_hidden_layers = self.job_config.model.n_layers
        elif hasattr(model, "config"):
            num_hidden_layers = getattr(
                model.config, "num_hidden_layers", getattr(model.config, "n_layers", None)
            )

        if num_hidden_layers is None:
            max_idx = 0
            pattern = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")
            for name, _ in model.named_modules():
                match = pattern.search(name)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))
            if max_idx > 0:
                num_hidden_layers = max_idx + 1

        if num_hidden_layers is None:
            num_hidden_layers = 32

        granular_replace_linear(
            model,
            num_hidden_layers=num_hidden_layers,
            mx_config=self.cfg,  # Pass the config directly
            mlp_recipe=getattr(self.cfg, "mlp_recipe", "Custom"),
            attn_recipe=getattr(self.cfg, "attn_recipe", "Custom"),
            mamba_recipe=getattr(self.cfg, "mamba_recipe", ""),
            exclude_last_n_layers=self.exclude_last_n,
            exclude_qkv=getattr(self.cfg, "exclude_qkv", False),
            quantize_mamba=getattr(self.cfg, "quantize_mamba", True),
            verbose=self.verbose,
        )


register_model_converter(TECustomConverter, "mxfp_custom")
