#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch

from torchtitan.config import JobConfig
from torchtitan.components import loss as tt_loss

from ...models.models import patch_train_specs

try:
    from liger_kernel.transformers.functional import liger_cross_entropy
except ImportError as error:
    raise ImportError(
        "liger-kernel is not installed. "
        "Did you run: uv pip install -e '.[experimental-liger]' ?"
    ) from error


def liger_patched_cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return liger_cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))


def liger_patched_build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    return liger_patched_cross_entropy_loss


patch_train_specs(
    "build_loss_fn",
    tt_loss.build_cross_entropy_loss,
    liger_patched_build_cross_entropy_loss,
)
