# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
from typing import Any
from torchtitan.components.lr_scheduler import LRSchedulersContainer
import torchtitan.components.lr_scheduler
import torchtitan.train

from . import checkpoint_load_decorator_template


class LRSchedulersContainerCompatV1(LRSchedulersContainer):
    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        _ = sd.pop("_is_initial")
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict["_is_initial"] = False
        super().load_state_dict(state_dict)


def enable():
    torchtitan.train.train_spec_module.LRSchedulersContainer = (
        LRSchedulersContainerCompatV1
    )
    torchtitan.components.lr_scheduler.LRSchedulersContainer = (
        LRSchedulersContainerCompatV1
    )


checkpoint_load_decorator = checkpoint_load_decorator_template(
    ["_is_initial", "ntokens_seen"], "lbt-v1"
)
