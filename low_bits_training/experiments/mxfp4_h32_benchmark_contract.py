"""Live model/state contract for the public MXFP4 fixed-H32 benchmark."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn

from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

from benchmarks.mxfp4_h32.route_contract import (
    expected_logical_keys,
    load_spec,
    validate_environment,
)
from low_bits_training.converters import Float32MasterParamsConverter


SPEC = load_spec()


def _find_head(model: nn.Module) -> tuple[str, nn.Module]:
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in {"output", "lm_head"}
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one output head, found {len(candidates)}")
    return candidates[0]


class MXFP4H32BenchmarkContract(ModelConverter):
    def __init__(self, job_config, parallel_dims):
        del job_config, parallel_dims

    def convert(self, model: nn.Module) -> None:
        validate_environment(SPEC)
        modules = {id(module): module for module in model.modules()}.values()
        counts = Counter(type(module).__name__ for module in modules)
        blocks = SPEC["model"]["blocks"]
        for class_name in ("FusedAttentionMXFP4_TK", "FusedFeedForwardMXFP4_TK"):
            if counts[class_name] != blocks:
                raise RuntimeError(
                    f"benchmark requires {blocks} {class_name}, found "
                    f"{counts[class_name]}"
                )
        if any(count for name, count in counts.items() if "LocalCTA" in name):
            raise RuntimeError("pure MXFP4 benchmark contains a localCTA module")
        name, head = _find_head(model)
        if type(head) is not nn.Linear or head.weight.dtype is not torch.bfloat16:
            raise RuntimeError("benchmark output head is not an ordinary BF16 Linear")
        logger.info(
            "MXFP4 H32 MODEL CONTRACT PASS blocks=%d head=%s row_sr=on "
            "wgrad_h32=on weight_rht=off",
            blocks,
            name,
        )

    def post_optimizer_hook(self, model) -> None:
        del model


register_model_converter(
    MXFP4H32BenchmarkContract, "mxfp4_h32_benchmark_contract"
)


def _install_state_contract() -> None:
    from low_bits_training.quantization import mxfp4_sr_state as state_module

    original = state_module.build_mxfp4_sr_state_for_trainer
    if getattr(original, "_mxfp4_h32_benchmark_checked", False):
        raise RuntimeError("MXFP4 H32 benchmark state contract installed twice")

    def checked(*args, **kwargs):
        state = original(*args, **kwargs)
        if state is None or tuple(state.logical_keys) != expected_logical_keys():
            raise RuntimeError("MXFP4 H32 benchmark requires the exact 128-key SR state")
        world = SPEC["topology"]["world_size"]
        if (
            state.world_size != world
            or state.user_seed != 1234
            or state.user_subsequence_base != 0
        ):
            raise RuntimeError("MXFP4 H32 benchmark SR namespace drifted")
        logger.info(
            "MXFP4 H32 STATE CONTRACT PASS producers=128 world=%d row_sr_seed=1234",
            world,
        )
        return state

    checked._mxfp4_h32_benchmark_checked = True
    state_module.build_mxfp4_sr_state_for_trainer = checked


_install_state_contract()


_original_master_convert = Float32MasterParamsConverter.convert


def _checked_master_convert(self, model: nn.Module) -> None:
    _original_master_convert(self, model)
    name, head = _find_head(model)
    if type(head) is not nn.Linear or head.weight.dtype is not torch.float32:
        raise RuntimeError("FP32 master conversion changed the ordinary output head")
    if self.compute_dtype != "bfloat16":
        raise RuntimeError("FSDP parameter compute dtype is not BF16")
    logger.info("MXFP4 H32 POST-MASTER HEAD CONTRACT PASS name=%s", name)


if getattr(Float32MasterParamsConverter.convert, "_mxfp4_h32_benchmark_checked", False):
    raise RuntimeError("MXFP4 H32 benchmark master contract installed twice")
_checked_master_convert._mxfp4_h32_benchmark_checked = True
Float32MasterParamsConverter.convert = _checked_master_convert
