"""Fail-closed model contracts shared by the public paper recipes.

The shell launcher owns paths and process topology. This module verifies the
scientific part after the selected low-precision converter has rewritten the
model: body family, layer allocation, ordinary BF16 output head, global batch,
and the route-defining rounding/transform controls.
"""

from __future__ import annotations

from collections import Counter
import os
import re

import torch
import torch.nn as nn

from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger


_ROUTE = os.environ.get("LBT_PAPER_ROUTE_ID", "")
_KNOWN_ROUTES = {
    "bf16-historical-long",
    "te-native-historical-long",
    "pure-v5-fused-v1-long",
    "mxfp4-v4-row-sr-fused-v1-long",
    "localcta-v4-row-sr-fused-v1-long",
    "localcta-v4-row-sr-h16-rht-long",
    "localcta-mxfp4-hybrid-27-5-fused-v1-long",
    "mxfp4-v4-row-sr-h32-rht-long",
    "te-f0l4-long-attempt",
    "mxfp4-col-h16-localcta-dgrad-v1-long",
    "mxfp4-col-h32-localcta-dgrad-v2-continuation",
}
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _require_env(expected: dict[str, str]) -> None:
    drift = {
        name: {"expected": value, "actual": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if drift:
        raise RuntimeError(f"paper route {_ROUTE!r} environment drift: {drift}")


def _layer_indices(model: nn.Module, class_name: str) -> set[int]:
    result: set[int] = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ != class_name:
            continue
        match = _LAYER.search(name)
        if match is None:
            raise RuntimeError(f"{class_name} has no transformer-layer path: {name!r}")
        result.add(int(match.group(1)))
    return result


def _class_counts(model: nn.Module) -> Counter[str]:
    return Counter(module.__class__.__name__ for module in model.modules())


def _require_all_layers(model: nn.Module, attention: str, ffn_names: set[str]) -> None:
    wanted = set(range(32))
    if _layer_indices(model, attention) != wanted:
        raise RuntimeError(f"{attention} is not installed on exactly layers 0..31")
    ffn_layers: set[int] = set()
    for name in ffn_names:
        ffn_layers |= _layer_indices(model, name)
    if ffn_layers != wanted:
        raise RuntimeError(f"{sorted(ffn_names)} are not installed on exactly layers 0..31")


def _find_output_head(model: nn.Module) -> tuple[str, nn.Module]:
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in {"output", "lm_head"}
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one output/lm_head module, found {len(candidates)}")
    return candidates[0]


class PaperRegularBF16Head(ModelConverter):
    def __init__(self, job_config, parallel_dims):
        del job_config, parallel_dims

    def convert(self, model: nn.Module) -> None:
        name, module = _find_output_head(model)
        if type(module) is not nn.Linear:
            raise RuntimeError(
                f"paper output head {name!r} must be exact torch.nn.Linear; "
                f"found {type(module)!r}"
            )
        if module.weight.dtype is not torch.bfloat16:
            raise RuntimeError(
                f"paper output head {name!r} must be BF16 before FP32 master "
                f"conversion; found {module.weight.dtype}"
            )
        if module.bias is not None and module.bias.dtype is not torch.bfloat16:
            raise RuntimeError("paper output-head bias is not BF16")
        logger.info("PAPER ORDINARY BF16 HEAD CONTRACT PASS route=%s name=%s", _ROUTE, name)

    def post_optimizer_hook(self, model) -> None:
        del model


class PaperRouteContract(ModelConverter):
    def __init__(self, job_config, parallel_dims):
        del parallel_dims
        if _ROUTE not in _KNOWN_ROUTES:
            raise RuntimeError(f"LBT_PAPER_ROUTE_ID is missing or unsupported: {_ROUTE!r}")
        if int(job_config.training.global_batch_size) != 512:
            raise RuntimeError("paper recipes require global batch size 512")
        if int(job_config.training.seq_len) != 8192:
            raise RuntimeError("paper recipes require sequence length 8192")
        if int(job_config.training.steps) != 71526 or int(job_config.job.steps) != 38147:
            raise RuntimeError("paper recipes require scheduler horizon 71526 and stop 38147")
        if int(job_config.debug.seed) != 42:
            raise RuntimeError("paper recipes require model seed 42")
        if bool(job_config.training.enable_cce) or bool(job_config.fp4_cce.enabled):
            raise RuntimeError("paper recipes require regular cross entropy, not CCE")
        if not bool(job_config.compile.enable) or "loss" not in job_config.compile.components:
            raise RuntimeError("paper recipes require loss-only torch.compile")
        if "model" in job_config.compile.components:
            raise RuntimeError("paper recipes do not compile the whole model")

    def convert(self, model: nn.Module) -> None:
        counts = _class_counts(model)
        mx_ffn = {
            "FusedFeedForwardMXFP4_TK",
            "FusedSquaredReLUFeedForwardMXFP4_TK",
            "ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK",
        }
        local_ffn = {
            "FusedFeedForwardFP4_TK",
            "FusedSquaredReLUFeedForwardFP4_TK",
        }

        if _ROUTE == "bf16-historical-long":
            forbidden = [
                name for name in counts
                if "FP4" in name or name == "PureBoundRecipeLinear"
            ]
            if forbidden:
                raise RuntimeError(f"BF16 control contains low-precision modules: {forbidden}")

        elif _ROUTE in {"te-native-historical-long", "te-f0l4-long-attempt"}:
            expected = 224 if _ROUTE == "te-native-historical-long" else 196
            if counts["PureBoundRecipeLinear"] != expected:
                raise RuntimeError(
                    f"TE route requires {expected} NVFP4 projections; "
                    f"found {counts['PureBoundRecipeLinear']}"
                )
            _require_env({
                "NVTE_NVFP4_DISABLE_RHT": "0",
                "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "0",
                "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "0",
            })

        elif _ROUTE in {"pure-v5-fused-v1-long", "localcta-v4-row-sr-fused-v1-long", "localcta-v4-row-sr-h16-rht-long"}:
            _require_all_layers(model, "FusedAttentionFP4_TK", local_ffn)
            if _ROUTE == "pure-v5-fused-v1-long":
                _require_env({
                    "USE_TK_LOCALCTA": "0",
                    "USE_TK_V5_2D_WEIGHT_QUANT": "1",
                    "NVFP4_GRAD_SR_AXES": "row",
                    "NVFP4_USE_RHT": "0",
                })
            else:
                expected_rht = "1" if _ROUTE.endswith("h16-rht-long") else "0"
                _require_env({
                    "USE_TK_LOCALCTA": "1",
                    "USE_TK_LOCALCTA_VARIANT": "v4",
                    "USE_TK_LOCALCTA_2D_WEIGHT_QUANT": "1",
                    "NVFP4_GRAD_SR_AXES": "row",
                    "NVFP4_USE_RHT": expected_rht,
                })

        elif _ROUTE == "localcta-mxfp4-hybrid-27-5-fused-v1-long":
            if _layer_indices(model, "FusedAttentionFP4_TK") != set(range(27)):
                raise RuntimeError("layer hybrid localCTA attention allocation drifted")
            local_layers: set[int] = set()
            for name in local_ffn:
                local_layers |= _layer_indices(model, name)
            if local_layers != set(range(27)):
                raise RuntimeError("layer hybrid localCTA FFN allocation drifted")
            if _layer_indices(model, "FusedAttentionMXFP4_TK") != set(range(27, 32)):
                raise RuntimeError("layer hybrid MXFP4 attention allocation drifted")
            mx_layers: set[int] = set()
            for name in mx_ffn:
                mx_layers |= _layer_indices(model, name)
            if mx_layers != set(range(27, 32)):
                raise RuntimeError("layer hybrid MXFP4 FFN allocation drifted")
            _require_env({
                "LBT_FP4_MIXED_LAYERS": "localcta:1-27;mxfp4:28-32",
                "LBT_LOCALCTA_SR_EXPECTED_LOGICAL_PRODUCERS": "108",
                "MXFP4_SR_EXPECTED_PRODUCERS": "20",
            })

        else:
            _require_all_layers(model, "FusedAttentionMXFP4_TK", mx_ffn)
            _require_env({
                "MXFP4_BACKEND_VERSION": "v4",
                "MXFP4_USE_2D_WEIGHT_QUANT": "1",
                "MXFP4_GRAD_SR_AXES": "row",
            })
            if _ROUTE == "mxfp4-v4-row-sr-h32-rht-long":
                _require_env({
                    "MXFP4_USE_LOCALCTA_DGRAD": "0",
                    "MXFP4_USE_RHT": "1",
                    "MXFP4_RHT_BLOCK_SIZE": "32",
                    "MXFP4_RHT_RANDOM_SIGN_MASK": "1",
                })
            elif _ROUTE == "mxfp4-v4-row-sr-fused-v1-long":
                _require_env({"MXFP4_USE_RHT": "0"})
            elif _ROUTE == "mxfp4-col-h16-localcta-dgrad-v1-long":
                _require_env({
                    "MXFP4_USE_LOCALCTA_DGRAD": "1",
                    "MXFP4_RHT_BLOCK_SIZE": "16",
                    "MXFP4_RHT_RANDOM_SIGN_MASK": "0",
                })
            elif _ROUTE == "mxfp4-col-h32-localcta-dgrad-v2-continuation":
                _require_env({
                    "MXFP4_USE_LOCALCTA_DGRAD": "1",
                    "MXFP4_RHT_BLOCK_SIZE": "32",
                    "MXFP4_RHT_RANDOM_SIGN_MASK": "1",
                })

        logger.info("PAPER ROUTE CONTRACT PASS route=%s", _ROUTE)

    def post_optimizer_hook(self, model) -> None:
        del model


register_model_converter(PaperRouteContract, "paper_route_contract_v1")
register_model_converter(PaperRegularBF16Head, "paper_regular_bf16_head_v1")
