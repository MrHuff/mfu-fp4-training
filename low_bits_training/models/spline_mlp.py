from __future__ import annotations

from typing import List, Union

import torch
from torch import nn

from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from low_bits_training.config import JobConfig
from .mlp_activation import resolve_mlp_activation_impl


class FeedForwardWithPatchedActivation(nn.Module):
    activation_impl_name: str
    activation_fn: callable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation_fn(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class FeedForwardFusedWithPatchedActivation(nn.Module):
    activation_impl_name: str
    activation_fn: callable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_input, gate = torch.chunk(self.w_in(x), chunks=2, dim=-1)
        return self.w_out(self.activation_fn(act_input) * gate)

    def init_weights(self, init_std: float):
        hidden_dim = self.w_out.weight.shape[-1]
        nn.init.trunc_normal_(self.w_in.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.w_in.weight[hidden_dim:].mul_(init_std / 0.02)
        nn.init.trunc_normal_(self.w_out.weight, mean=0.0, std=init_std)
        if hasattr(self.w_in, "norm_weight"):
            self.w_in.norm_weight.data.fill_(1.0)


def _patch_mlp_modules(module: nn.Module, activation_impl_name: str, activation_fn) -> int:
    patched = 0
    for child in module.modules():
        if hasattr(child, "w1") and hasattr(child, "w2") and hasattr(child, "w3"):
            child.activation_impl_name = activation_impl_name
            child.activation_fn = activation_fn
            if child.__class__ is not FeedForwardWithPatchedActivation:
                child.__class__ = FeedForwardWithPatchedActivation
            patched += 1
        elif hasattr(child, "w_in") and hasattr(child, "w_out"):
            child.activation_impl_name = activation_impl_name
            child.activation_fn = activation_fn
            if child.__class__ is not FeedForwardFusedWithPatchedActivation:
                child.__class__ = FeedForwardFusedWithPatchedActivation
            patched += 1
    return patched


class SplineMLPConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.activation_impl_name, self.activation_fn = resolve_mlp_activation_impl(
            job_config.spline_mlp.activation_impl
        )

    def convert(self, model: nn.Module):
        patched = _patch_mlp_modules(
            model, self.activation_impl_name, self.activation_fn
        )
        logger.info(
            "Patched %d MLP modules to activation_impl=%s",
            patched,
            self.activation_impl_name,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass


register_model_converter(SplineMLPConverter, "spline_mlp")
