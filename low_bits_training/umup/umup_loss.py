#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch
from unit_scaling.scale import scale_fwd


def umup_nll_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """NLL for use with unit-scaled model. You must apply log_softmax to the output of the final matmul to use this."""
    pred = pred.flatten(0, 1).float()
    labels = labels.flatten(0, 1)
    batch_size, _ = pred.shape
    loss = torch.nn.functional.nll_loss(pred, labels, reduction="sum")
    return scale_fwd(loss, 1 / batch_size)
