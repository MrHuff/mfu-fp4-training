#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import pytest
import torch

import low_bits_training


def test__liger_cross_entropy_correct_outputs():
    from low_bits_training.experiments.liger import liger_patching
    from torchtitan.components.loss import cross_entropy_loss

    logits = torch.randn(4, 10, 1000, device="cuda")
    labels = torch.randint(0, 1000, (4, 10), device="cuda")
    result_torch = cross_entropy_loss(logits, labels)
    result_liger = liger_patching.liger_patched_build_cross_entropy_loss(None)(
        logits, labels
    )
    torch.testing.assert_close(result_torch, result_liger)


@pytest.mark.integration
def test__liger_cross_entropy_trains(no_distribution):
    from low_bits_training.config.manager import ConfigManager
    from torchtitan import train

    config = ConfigManager().parse_args(
        [
            "--job.config_file",
            "./train_configs/debug_model.toml",
            "--model.tokenizer_path",
            "tests/assets/test-tokenizer",
            "--wandb.mode",
            "disabled",
            "--model.name",
            "llama3_gc",
            "--model.flavor",
            "unit_test",
            "--training.steps",
            "2",
        ]
    )
    _ = low_bits_training.utils.wandb_init(
        job_config=config,
        project=config.wandb.project,
        entity="graphcore",
    )
    trainer = train.Trainer(config)
    # Missing in TorchTitan upstream. TODO: fix PR.
    trainer.metrics_processor.lr_schedulers = trainer.lr_schedulers
    trainer.train()
