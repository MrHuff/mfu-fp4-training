#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import copy
import torch._utils

from low_bits_training.device_patch import (
    get_device_type,
    get_device_module,
    device_type_context,
)
import torchtitan
import torchtitan.tools.utils


def test__torch_get_device_module__support_cpu():
    # Make sure `_get_device_module` support `cpu`
    device_type = "cpu"
    m = torch._utils._get_device_module(device_type)
    assert m is torch.cpu


def test___get_available_device_type__not_returning_none():
    assert torch._utils._get_available_device_type() is not None
    assert torch._utils._get_available_device_type() in {"cpu", "cuda"}


def test__get_device_type_and_module__match_torchtitan():
    # Should return TorchTitan instance
    assert id(get_device_type()) == id(torchtitan.tools.utils.device_type)
    assert id(get_device_module()) == id(torchtitan.tools.utils.device_module)
    assert get_device_type() == torch._utils._get_available_device_type()


def test__get_device_type__never_none_or_cuda_on_cpu():
    assert get_device_type() is not None
    if not torch.cuda.is_available():
        assert get_device_type() != "cuda"


def test__set_device_type__no_op_if_already_set_properly():
    device_type = get_device_type()
    with device_type_context(copy.deepcopy(device_type)):
        assert id(device_type) == id(get_device_type())
    assert id(device_type) == id(get_device_type())
