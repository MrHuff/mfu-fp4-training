#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest

from low_bits_training.config import JobConfig, ConfigManager
from low_bits_training.utils import (
    job_config_to_config_dict,
    get_parallel_dims,
    is_metrics_logging_enabled,
    wandb_logging_mode,
    wandb_group,
    find_torchtitan_modules_with_imported_class,
)
import torchtitan
import torchtitan.tools.utils


def test__job_config_to_config_dict__proper_dict_structure():
    config = JobConfig()
    config_dict = job_config_to_config_dict(config)

    assert isinstance(config_dict, dict)
    assert all([isinstance(v, dict) for v in config_dict.values()])


def test__get_parallel_dims__default_values(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "64")

    config = JobConfig()
    dims = get_parallel_dims(config)
    assert dims.dp_replicate == 1
    assert dims.dp_shard == 64
    assert dims.cp == 1
    assert dims.ep == 1
    assert dims.tp == 1
    assert dims.pp == 1
    assert dims.etp == 1
    assert dims.world_size == 64


@pytest.mark.parametrize(
    "tb,mode,rank,local_rank,expected",
    [
        (False, "all", 0, 0, False),
        (False, "rank_0", 0, 0, False),
        (False, "local_rank_0", 0, 0, False),
        (True, "all", 0, 0, True),
        (True, "all", 7, 3, True),
        (True, "rank_0", 7, 7, False),
        (True, "rank_0", 0, 0, True),
        (True, "local_rank_0", 0, 0, True),
        (True, "local_rank_0", 4, 0, True),
    ],
)
def test__is_metrics_logging_enabled__distributed_mode_all(
    monkeypatch, tb: bool, mode: str, rank: int, local_rank: int, expected: bool
):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))

    config = JobConfig()
    config.metrics.enable_tensorboard = tb
    config.metrics.distributed_mode = mode
    assert is_metrics_logging_enabled(config) == expected


def test__is_metrics_logging_enabled__rank_0__no_pipeline_parallelism(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    config = JobConfig()
    config.metrics.enable_tensorboard = True
    config.metrics.distributed_mode = "rank_0"
    # Default PP == 1 => RANK 0 is the metric logger.
    assert config.parallelism.pipeline_parallel_degree == 1
    assert is_metrics_logging_enabled(config)


@pytest.mark.parametrize("mode", ["rank_0", "local_rank_0"])
def test__is_metrics_logging_enabled__with_pipeline_parallelism(monkeypatch, mode):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "7")

    config = JobConfig()
    config.metrics.enable_tensorboard = True
    config.metrics.distributed_mode = mode
    # PP == 8 => RANK 7 is the metric logger.
    config.parallelism.pipeline_parallel_degree = 8
    assert is_metrics_logging_enabled(config)


def test__wandb_logging_mode__disabled_by_default():
    config = JobConfig()
    assert wandb_logging_mode(config) == "disabled"


@pytest.mark.parametrize("mode", ["online", "offline"])
def test__wandb_logging_mode__env_variable_setting(monkeypatch, mode):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WANDB_MODE", mode)

    config = ConfigManager().parse_args([])
    config.metrics.enable_tensorboard = True
    assert is_metrics_logging_enabled(config)
    assert wandb_logging_mode(config) == mode


def test__wandb_group__proper_default():
    config = JobConfig()
    config.wandb.name = "test"
    # Default case scenario.
    assert config.wandb.group == ""
    assert wandb_group(config) == "test"
    # Group provided scenario.
    config.wandb.group = "group"
    assert wandb_group(config) == "group"


def test__find_torchtitan_modules_with_imported_class():
    from torchtitan.train import Trainer

    # Trainer should be in `train`!
    modules = find_torchtitan_modules_with_imported_class(Trainer)
    assert len(modules) == 1
    assert modules[0] is torchtitan.train


def test__init_logger_patching():
    from low_bits_training.logger import _tt_init_logger as init_logger

    # Check all sub-modules in TorchTitan use the proper new `init_logger`
    modules = find_torchtitan_modules_with_imported_class(init_logger)
    assert len(modules) == 0


def test__build_metric_logger_patching():
    from low_bits_training.metrics import tt_build_metric_logger as _build_metric_logger

    # Check all sub-modules in TorchTitan use the proper new `_build_metric_logger`
    modules = find_torchtitan_modules_with_imported_class(_build_metric_logger)
    assert len(modules) == 0


def test__profiling_patching():
    from low_bits_training.profiling import (
        tt_maybe_enable_profiling as maybe_enable_profiling,
    )

    # Check all sub-modules in TorchTitan use the proper new `maybe_enable_profiling`
    modules = find_torchtitan_modules_with_imported_class(maybe_enable_profiling)
    assert len(modules) == 0
