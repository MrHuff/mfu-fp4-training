# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
from typing import Any
from torchtitan.components.optimizer import OptimizersContainer
import torchtitan.components.optimizer
import torchtitan.train

from . import lbt_v1
from . import checkpoint_load_decorator_template


class OptimizersContainerCompatV1(OptimizersContainer):
    # Does not support Optimizer in backward or torch FT
    # this is ok because we've not done any runs with those
    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        self.to_pop = {}
        for k in sd.keys():
            if ".decoupled_weight_decay" in k:
                self.to_pop[k] = None
        for to_pop in self.to_pop.keys():
            self.to_pop[to_pop] = sd.pop(to_pop)
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for k, v in self.to_pop.items():
            state_dict[k] = v
        super().load_state_dict(state_dict)


def enable():
    torchtitan.train.train_spec_module.OptimizersContainer = OptimizersContainerCompatV1
    torchtitan.components.optimizer.OptimizersContainer = OptimizersContainerCompatV1
    lbt_v1.enable()


checkpoint_load_decorator = checkpoint_load_decorator_template(
    [".decoupled_weight_decay"], "lbt-v0"
)
