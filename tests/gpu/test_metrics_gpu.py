#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest

import low_bits_training
from low_bits_training.config import JobConfig, ConfigManager
from low_bits_training.metrics import build_metrics_processor, WBMetricLogger
from low_bits_training.utils import get_parallel_dims

from torchtitan import train


def test__metrics_processor__using_custom_wb_logger(monkeypatch):
    # Make sure the patching works to build a custom WBMetricLogger
    monkeypatch.setenv("WORLD_SIZE", "1")
    cfg = JobConfig()
    parallel_dims = get_parallel_dims(cfg)
    mp = build_metrics_processor(cfg, parallel_dims)
    assert isinstance(mp.logger, WBMetricLogger)


@pytest.fixture
def mock_wandb(mocker):
    """Fixture that mocks common wandb functions"""
    mocks = {
        "init": mocker.patch("wandb.init"),
        "log": mocker.patch("wandb.log"),
        "finish": mocker.patch("wandb.finish"),
        "save": mocker.patch("wandb.save"),
    }

    # Configure default behavior
    mock_run = mocker.MagicMock()
    mocks["init"].return_value = mock_run

    return mocks


def test__metrics__something_is_logged_to_wandb(mock_wandb, no_distribution):
    config = ConfigManager().parse_args(
        [
            "--job.config_file",
            "./train_configs/debug_model.toml",
            "--model.tokenizer_path",
            "tests/assets/test-tokenizer",
            "--wandb.mode",
            "offline",
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

    assert mock_wandb["init"].call_count == 1
    assert mock_wandb["log"].call_count == 2  # 1 per training step
    metrics_tuple, step = mock_wandb["log"].call_args_list[0]
    metrics = metrics_tuple[0]
    expected_metrics = [
        "loss_metrics/global_avg_loss",
        "loss_metrics/global_max_loss",
        "throughput(tps)",
        "tflops",
        "mfu(%)",
        "time_metrics/end_to_end(s)",
        "time_metrics/data_loading(s)",
        "time_metrics/data_loading(%)",
        "memory/max_active(GiB)",
        "memory/max_active(%)",
        "memory/max_reserved(GiB)",
        "memory/max_reserved(%)",
        "memory/num_alloc_retries",
        "memory/num_ooms",
        "LR",
        "total_throughput(tps)",
    ]
    assert set(expected_metrics) & set(metrics.keys()) == set(expected_metrics)
    assert step["step"] == 1
