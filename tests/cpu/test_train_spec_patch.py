#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from low_bits_training.models.models import patch_train_specs
from torchtitan.protocols.train_spec import get_train_spec
import torch
from torchtitan.components import loss as tt_loss
from torchtitan.config import JobConfig


def test_patch_loss():
    train_spec = get_train_spec("llama3_gc")
    assert train_spec.build_loss_fn is tt_loss.build_cross_entropy_loss

    def build_loss(job_config: JobConfig):
        def dummy_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return y - x

        return dummy_loss

    patch_train_specs(
        "build_loss_fn", old=tt_loss.build_cross_entropy_loss, new=build_loss
    )

    train_spec = get_train_spec("llama3_gc")
    assert train_spec.build_loss_fn is build_loss
