#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from low_bits_training.ema_checkpoint import (
    CheckpointManagerLoadOnly,
    CheckpointManagerPatching,
    CheckpointManagerWithEMAWeight,
)

from low_bits_training.utils import find_torchtitan_modules_with_imported_class

import inspect
import torchtitan
import torchtitan.components.checkpoint

import pytest


def test__checkpoint_manager_patching():
    CheckpointManager = torchtitan.components.checkpoint.CheckpointManager

    modules = find_torchtitan_modules_with_imported_class(CheckpointManager)
    assert torchtitan.components.checkpoint in modules
    # Expecting to be used in `train` as well.
    assert len(modules) >= 2

    with CheckpointManagerPatching(CheckpointManagerLoadOnly):
        # All modules should be patched in context
        for m in modules:
            assert getattr(m, "CheckpointManager") is CheckpointManagerLoadOnly
    # Revert back.
    for m in modules:
        assert getattr(m, "CheckpointManager") is CheckpointManager


@pytest.mark.parametrize("fn", ["__init__", "save", "load", "maybe_wait_for_staging"])
def test__checkpoint_manager_with_ema__compatible_signature(fn):
    CheckpointManager = torchtitan.components.checkpoint.CheckpointManager

    tt_fn_args = inspect.signature(getattr(CheckpointManager, fn)).parameters
    ema_fn_args = inspect.signature(
        getattr(CheckpointManagerWithEMAWeight, fn)
    ).parameters

    assert len(ema_fn_args) == len(tt_fn_args)
    assert list(ema_fn_args.keys()) == list(tt_fn_args.keys())
    assert list(ema_fn_args.values()) == list(tt_fn_args.values())
