#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import low_bits_training  # noqa: F401
from torchtitan.protocols.train_spec import get_train_spec


def test__liger_cross_entropy_patching_works():
    from low_bits_training.experiments.liger import liger_patching

    assert (
        get_train_spec("llama3_gc").build_loss_fn
        == liger_patching.liger_patched_build_cross_entropy_loss
    )
