#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch
from low_bits_training.device_patch import (
    get_device_type,
    get_device_module,
    device_type_context,
    get_available_device_type_with_cpu_support,
)
import torchtitan
import torchtitan.tools.utils


def test__get_device_type__on_gpu():
    # Check on GPU we are on `cuda` device by default!
    assert torch.cuda.is_available()
    assert get_device_type() == "cuda"
    assert get_device_module() is torch.cuda


def test__device_type_context__on_gpu():
    device_type = get_device_type()
    assert device_type == "cuda"
    # Local set to `cpu`
    with device_type_context("cpu"):
        assert get_device_type() == "cpu"
        assert torchtitan.tools.utils.device_type == "cpu"
        assert torchtitan.tools.utils.device_module is torch.cpu
        # TODO: decide what it should return here?
        assert torch._utils._get_available_device_type() == "cpu"
    # Reset to original
    assert get_device_type() == device_type
    assert torchtitan.tools.utils.device_type == device_type
    assert (
        torch._utils._get_available_device_type
        is get_available_device_type_with_cpu_support
    )
