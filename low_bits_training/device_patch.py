#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os
import warnings
from contextlib import contextmanager
from typing import Any

import torch
import torch._utils
from torch._utils import _get_available_device_type as torch_get_available_device_type

import torchtitan
import torchtitan.tools.utils
import torchtitan.components
import torchtitan.components.metrics
import torchtitan.distributed
import torchtitan.distributed.parallel_dims
import torchtitan.distributed.utils

from torchtitan.tools.logging import logger


def get_available_device_type_with_cpu_support() -> str:
    """`_get_available_device_type` function where it returns `cpu` by default."""
    device_type = torch_get_available_device_type()
    if device_type is None:
        return "cpu"
    return device_type


# Patching `torch` so the default is always CPU, not None
torch._utils._get_available_device_type = get_available_device_type_with_cpu_support


def get_device_type() -> str:
    """Get the device type set in TorchTitan.

    NOTE: using a function to be able to mutate the device type in a context.
    """
    return torchtitan.tools.utils.device_type


def get_device_module() -> Any:
    """Get the device module set in TorchTitan.

    NOTE: using a function to be able to mutate the device type in a context.
    """
    return torchtitan.tools.utils.device_module


def get_device_info():
    return get_device_type(), get_device_module()


def set_device_type(device_type: str):
    """Set device type (and module) used in TorchTitan.

    NOTE: the method is trying to "patch" everywhere in TorchTitan where `device_type`
    is imported, to keep it consistent.
    """
    # No-op if the device type is already properly set.
    if device_type == get_device_type():
        return

    modules_to_patch = [
        torchtitan.tools.utils,
        torchtitan.components.metrics,
        torchtitan.distributed.parallel_dims,
        torchtitan.distributed.utils,
    ]
    device_module = torch._utils._get_device_module(device_type)

    for m in modules_to_patch:
        assert hasattr(m, "device_type"), f"No `device_type` imported in {m}"
        m.device_type = device_type
        m.device_module = device_module

    # Patch `torch` _get_available_device_type if necessary.
    if device_type != get_available_device_type_with_cpu_support():
        torch._utils._get_available_device_type = lambda: device_type
    else:
        torch._utils._get_available_device_type = (
            get_available_device_type_with_cpu_support
        )


@contextmanager
def device_type_context(device_type: str):
    device_type_saved = torchtitan.tools.utils.device_type
    set_device_type(device_type)
    try:
        yield device_type
    finally:
        # Restore previous device type.
        set_device_type(device_type_saved)


def patch_torch_cpu_as_cuda():
    """
    Adds missing CUDA-like methods to `torch.cpu`.

    This is the **minimum set of methods** to make the code work, not a complete
    replacement for `torch.cuda`.
    """
    warnings.warn("No accelerator is available, using CPU emulation.")

    ## Breaking Changes
    # Return int instead of "cpu" device string.
    # Alternatively you can not use the device_ids in torch.distributed.barrier:
    #   - torch.distributed.barrier(device_ids=[device_module.current_device()])
    #   + torch.distributed.barrier()
    #
    torch.cpu.current_device = lambda *_: int(os.environ["LOCAL_RANK"])

    ## Placeholders
    torch.cpu.get_device_name = lambda *_: "CPU"
    torch.cpu.empty_cache = lambda *_: None

    torch.cpu.reset_peak_memory_stats = lambda *_: None
    torch.cpu.memory_stats = lambda *_: dict()

    class _Props:
        total_memory = 1  # 1 byte

    torch.cpu.get_device_properties = lambda *_: _Props()

    return torch.cpu


def patch_cpu_device():
    """Patch `torch.cpu` module to have same interface as `torch.cuda`.
    Only activate if the device type is "cpu"
    """
    if get_device_type() == "cpu":
        patch_torch_cpu_as_cuda()


# Check the consistency between TorchTitan and Torch. Fix it if necessary.
if torch._utils._get_available_device_type() != get_device_type():
    device_type = torch._utils._get_available_device_type()
    logger.info(f"Set device type to: '{device_type}'.")
    set_device_type(device_type)
