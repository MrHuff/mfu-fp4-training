#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from torchtitan.components.lr_scheduler import (
    LRSchedulersContainer,
    OptimizersContainer,
    build_lr_schedulers as tt_build_lr_schedulers,
)
from torchtitan.config import LRScheduler as LRSchedulerConfig


def build_lr_schedulers(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> LRSchedulersContainer:
    """Wrapping TorchTitan LR scheduler factory method, keeping track of the LR scheduler.

    Using Job config `optimizer.lr_scheduler` to choose LR scheduler lambda to use.
    """
    # Build with proper LR scheduler.
    lr_scheduler = tt_build_lr_schedulers(optimizers, lr_scheduler_config, training_steps)
    return lr_scheduler
