#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import logging
import re

import wandb

from torchtitan.tools.logging import init_logger
from low_bits_training.config import ConfigManager
from low_bits_training.trainer import Trainer


class StepCountHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.max_step_seen = 0
        self._step_pattern = re.compile("step:\\s*(\\d+)")

    def emit(self, record):
        regex_match = self._step_pattern.match(record.getMessage())
        if regex_match is not None:
            (step,) = regex_match.groups()
            self.max_step_seen = max(int(step), self.max_step_seen)


def test_cutoff_step(no_distribution, tmp_path):
    wandb.init()
    init_logger()
    job_config = ConfigManager().parse_args(
        (
            "--wandb.mode offline "
            "--job.config_file ./train_configs/debug_model.toml "
            "--job.steps 3 "
            "--training.steps 10 "
            f"--job.dump_folder {tmp_path} "
            "--model.flavor unit_test "
            "--model.hf_assets_path tests/assets/test-tokenizer "
            "--metrics.disable_color_printing"
        ).split(" ")  # type: ignore
    )
    trainer = Trainer(job_config)

    logger = logging.getLogger()

    step_count_handler = StepCountHandler()
    step_count_handler.setLevel(logging.INFO)
    logger.addHandler(step_count_handler)

    trainer.train()

    assert step_count_handler.max_step_seen == 3
