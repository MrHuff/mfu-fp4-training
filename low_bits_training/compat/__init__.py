# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
from typing import Callable

from torch.distributed.checkpoint.api import CheckpointException

from ..config import JobConfig

from torchtitan.train import Trainer


def enable_compat_mode(job_config: JobConfig):
    """Enable backward compatibility modes based on the config."""
    from . import lbt_v0
    from . import lbt_v1

    ENABLE_METHODS: dict[str, Callable] = {
        "lbt-v0": lbt_v0.enable,
        "lbt-v1": lbt_v1.enable,
    }
    if job_config.checkpoint.compatibility is None:
        return
    if job_config.checkpoint.compatibility in ENABLE_METHODS:
        ENABLE_METHODS[job_config.checkpoint.compatibility]()
        return


def decorate_trainer_methods(trainer: Trainer):
    from . import lbt_v1
    from . import lbt_v0

    DECORATORS = {
        "checkpointer.load": [
            lbt_v1.checkpoint_load_decorator,
            lbt_v0.checkpoint_load_decorator,
        ]
    }
    original_checkpoint_load = trainer.checkpointer.load
    decorated = original_checkpoint_load
    for decorator in DECORATORS["checkpointer.load"]:
        decorated = decorator(decorated)
    trainer.checkpointer.load = decorated


def checkpoint_load_decorator_template(missing_keys: list[str], version: str):
    def checkpoint_load_decorator(load_func):
        def decorated(*args, **kwargs):
            try:
                return load_func(*args, **kwargs)
            except (
                CheckpointException
            ) as e:  # DCP load returns a RuntimeError on missing keys
                if any(missing in str(e) for missing in missing_keys):
                    raise ValueError(
                        f"Checkpoint could not be loaded, this may be an old checkpoint set: --checkpoint.compatibility={version} "
                    ) from e
                raise

        return decorated

    return checkpoint_load_decorator
