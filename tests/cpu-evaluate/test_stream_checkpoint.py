# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch

from low_bits_training.analysis import stream_checkpoints
from low_bits_training import checkpoints


def test_checkpoint_has_all_tensors(config_and_checkpoint):
    config, checkpoint = config_and_checkpoint
    # Load regular checkpoint for comparison
    loader = checkpoints.REGISTERED_CHECKPOINT_CONVERSIONS[config.model.name]
    model, _ = loader.load_tt_model_and_tokenizer(config, str(checkpoint))
    state_dict = model.state_dict()

    only_streamed = []
    missing = []
    differences = []
    tensors_streamed = []

    # Stream the checkpoint and compare the tensors to those that have been loaded the normal way
    for fqn, tensor in stream_checkpoints.stream_checkpoint_reader(
        checkpoint, batch_tensors=100
    ):
        # Old checkpoints start with this prefix that is not in the model state_dict
        if fqn.startswith("model."):
            fqn = fqn[6:]
        tensors_streamed.append(fqn)

        # The checkpoint contains keys which are not in the model - e.g. optimizer_state
        # Skip those for the comparison
        if fqn not in state_dict:
            only_streamed.append(fqn)
            continue
        # Check tensor values match
        if not torch.all((tensor == state_dict[fqn]).all()):
            differences.append(fqn)

    # Check all tensors in the model state dict where loaded while streaming
    tensors_streamed = set(tensors_streamed)
    for name in state_dict:
        if name not in tensors_streamed:
            missing.append(name)

    assert not missing
    assert not differences
