#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
from __future__ import annotations

import functools
import importlib.metadata as importlib_metadata
import os
import re
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn as nn
import transformer_engine
import transformer_engine.pytorch as te
from packaging.version import Version
try:
    from transformer_engine.common.recipe import MXFP4BlockScaling, NVFP4BlockScaling
except ImportError:
    from transformer_engine.common.recipe import NVFP4BlockScaling
    MXFP4BlockScaling = None

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

_MIN_TE_VERSION = Version("2.12.0")


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, attr.split("."), obj)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def _log_loaded_te_build() -> tuple[str, str]:
    version = importlib_metadata.version("transformer-engine")
    te_path = transformer_engine.__file__
    logger.info("PURE-TE baseline using transformer_engine=%s from %s", version, te_path)
    if Version(version) < _MIN_TE_VERSION and os.getenv("PURE_TE_ALLOW_OLD_TE", "0") != "1":
        raise RuntimeError(
            "pure_te_fp4 requires transformer-engine>=2.12.0 for the native TE "
            f"reference baseline, but imported transformer-engine=={version} from {te_path}. "
            "Set PURE_TE_ALLOW_OLD_TE=1 only for legacy/debug reruns."
        )
    allow_custom = os.getenv("PURE_TE_ALLOW_CUSTOM_BUILD", "0") == "1"
    if "06b44b8e" in version and not allow_custom:
        raise RuntimeError(
            "pure_te_fp4 picked up the custom fp4-custom-quantisation TE build "
            f"({version}); aborting clinical TE baseline"
        )
    if "06b44b8e" in version and allow_custom:
        logger.warning(
            "PURE-TE converter is running against the acknowledged custom TE build %s",
            version,
        )
    return version, te_path


class PureBoundRecipeLinear(te.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        params_dtype: torch.dtype = torch.bfloat16,
        recipe: Any = None,
        device: Any = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            params_dtype=params_dtype,
            device=device,
        )
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
        with amp_ctx, te.fp8_autocast(enabled=True, fp8_recipe=self.bound_recipe):
            return super().forward(inp)


def _granular_replace_linear(
    model: nn.Module,
    *,
    num_hidden_layers: int,
    mlp_recipe: Optional[Any],
    attn_recipe: Optional[Any],
    mamba_recipe: Optional[Any],
    exclude_last_n_layers: int = 0,
    exclude_last_n_ffn_layers: int = 0,
    exclude_qkv: bool = False,
    quantize_mamba: bool = True,
) -> None:
    modules_to_replace = []
    cutoff_index = num_hidden_layers - exclude_last_n_layers
    ffn_cutoff_index = num_hidden_layers - exclude_last_n_ffn_layers
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
    layer_idx_pattern = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
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
        if is_head:
            continue

        if layer_idx != -1 and layer_idx < cutoff_index:
            if is_mlp:
                if layer_idx < ffn_cutoff_index:
                    selected_recipe = mlp_recipe
            elif is_attn and not (exclude_qkv and is_qkv):
                selected_recipe = attn_recipe
            elif is_mamba:
                selected_recipe = mamba_recipe

        if selected_recipe is not None:
            modules_to_replace.append((name, module, selected_recipe))

    logger.info("PURE-TE swapping %d internal modules; output/lm_head layers remain native", len(modules_to_replace))
    for name, module, recipe in modules_to_replace:
        target_device = module.weight.device
        te_layer = PureBoundRecipeLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            params_dtype=module.weight.dtype,
            recipe=recipe,
            device=target_device,
        )
        if target_device.type != "meta":
            with torch.no_grad():
                te_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    te_layer.bias.copy_(module.bias)
        rsetattr(model, name, te_layer)


class PureTEFP4Converter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.mlp_recipe_name = job_config.te_fp4.mlp_recipe
        self.attn_recipe_name = job_config.te_fp4.attn_recipe
        self.mamba_recipe_name = job_config.te_fp4.mamba_recipe
        self.exclude_last_n = job_config.te_fp4.exclude_last_n_layers
        self.exclude_last_n_ffn = job_config.te_fp4.exclude_last_n_ffn_layers
        self.exclude_qkv = job_config.te_fp4.exclude_qkv
        self.quantize_mamba = job_config.te_fp4.quantize_mamba
        env_exclude = os.environ.get("FP4_KEEP_LAST_N_LAYERS_BF16")
        if env_exclude is not None:
            self.exclude_last_n = int(env_exclude)
            logger.info(
                "PURE-TE final-layer BF16 ablation: exclude_last_n_layers=%d from FP4",
                self.exclude_last_n,
            )
        env_exclude_ffn = os.environ.get("FP4_KEEP_LAST_N_FFNS_BF16")
        if env_exclude_ffn is not None:
            self.exclude_last_n_ffn = int(env_exclude_ffn)
            logger.info(
                "PURE-TE final-FFN BF16 ablation: exclude_last_n_ffn_layers=%d from FP4",
                self.exclude_last_n_ffn,
            )

    def _get_recipe_obj(self, name: str):
        if name == "NVFP4":
            return NVFP4BlockScaling()
        if name == "MXFP4":
            if MXFP4BlockScaling is None:
                raise RuntimeError("Loaded TE build does not expose MXFP4BlockScaling")
            return MXFP4BlockScaling(
                disable_rht=os.getenv("NVTE_NVFP4_DISABLE_RHT", "0") == "1",
                disable_stochastic_rounding=os.getenv("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "0") == "1",
                disable_2d_quantization=os.getenv("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "0") == "1",
                encode=True,
            )
        return None

    def convert(self, model: nn.Module):
        _log_loaded_te_build()
        recipe_mlp = self._get_recipe_obj(self.mlp_recipe_name)
        recipe_attn = self._get_recipe_obj(self.attn_recipe_name)
        recipe_mamba = (
            recipe_mlp
            if not self.mamba_recipe_name
            else self._get_recipe_obj(self.mamba_recipe_name)
        )
        num_hidden_layers = self.job_config.model.n_layers
        if num_hidden_layers is None and hasattr(model, "config"):
            num_hidden_layers = getattr(
                model.config,
                "num_hidden_layers",
                getattr(model.config, "n_layers", None),
            )
        if num_hidden_layers is None:
            pattern = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")
            max_idx = -1
            for name, _ in model.named_modules():
                match = pattern.search(name)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))
            if max_idx >= 0:
                num_hidden_layers = max_idx + 1
        if num_hidden_layers is None:
            raise RuntimeError("pure_te_fp4 could not determine num_hidden_layers")
        _granular_replace_linear(
            model,
            num_hidden_layers=num_hidden_layers,
            mlp_recipe=recipe_mlp,
            attn_recipe=recipe_attn,
            mamba_recipe=recipe_mamba,
            exclude_last_n_layers=self.exclude_last_n,
            exclude_last_n_ffn_layers=self.exclude_last_n_ffn,
            exclude_qkv=self.exclude_qkv,
            quantize_mamba=self.quantize_mamba,
        )

    def post_optimizer_hook(self, model):
        pass


register_model_converter(PureTEFP4Converter, "pure_te_fp4")
