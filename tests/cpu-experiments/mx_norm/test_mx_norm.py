#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch


def test__rms_norm_backward():
    # Importing from experimental module only as necessary, following `mxfp8` example
    from low_bits_training.experiments.mx_norm.mx_norm import rms_norm_backward

    torch.manual_seed(1472)
    init = torch.randn(4, 4)
    grad_output = torch.randn(4, 4)
    eps = 1e-6

    x_ref = torch.nn.Parameter(init)
    n_ref = torch.nn.functional.rms_norm(x_ref, normalized_shape=(4,), eps=eps)
    n_ref.backward(grad_output)

    assert x_ref.grad is not None
    assert (x_ref.grad - rms_norm_backward(init, eps, grad_output)).abs().max() < 1e-4
