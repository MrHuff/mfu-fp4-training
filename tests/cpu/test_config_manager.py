#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from pathlib import Path
import json

from low_bits_training.utils import load_config
from low_bits_training.config import JobConfig, ConfigManager, Checkpoint

import pytest


def test__config_manager__cli_args_proper_parsing():
    # Check the default is different from `llama3`
    assert JobConfig().model.name != "llama3"
    config = ConfigManager().parse_args(["--model.name=llama3"])
    assert isinstance(config, JobConfig)
    assert config.model.name == "llama3"


def test__config_manager__multi_passing_single_arg():
    # Last arg passed is the one counting.
    config = ConfigManager().parse_args(
        ["--model.name=deepseek_v2", "--model.name=deepseek_v3"]
    )
    assert isinstance(config, JobConfig)
    assert config.model.name == "deepseek_v3"


def test__config_manager__env_argument(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "llama3")
    config = ConfigManager().parse_args([])
    assert config.model.name == "llama3"


def test__config_manager__cli_env_arg_priority(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "llama3")
    config = ConfigManager().parse_args(["--model.name=llama3_gc"])
    assert config.model.name == "llama3_gc"


def test__config_manager__exception_if_file_not_existing():
    with pytest.raises(FileNotFoundError):
        ConfigManager().parse_args(["--job.config_file=dummy.toml"])
    with pytest.raises(FileNotFoundError):
        # Edge case of argument passed twice, with the first valid.
        ConfigManager().parse_args(
            [
                "--job.config_file=./train_configs/debug_model.toml",
                "--job.config_file=dummy.toml",
            ]
        )


def test__config_manager__validate_checkpoint_interval():
    with pytest.raises(AssertionError):
        ConfigManager().parse_args(
            [
                "--checkpoint.enable",
                "--ema_checkpoint.enable_checkpoint",
                "--checkpoint.interval=2",
                "--ema_checkpoint.save_interval=3",
            ]
        )


def test_config_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: str):
    test_name = "funky-tester-39"
    rank = "2"
    description = "test of the argument parsing"

    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WANDB_NAME", test_name)

    config = ConfigManager().parse_args(
        [f"--job.dump_folder={tmp_path}", f"--job.description={description}"]
    )
    config.dump()

    dump_folder = Path(tmp_path) / test_name
    configs = list(dump_folder.glob("config*rank2*.json"))

    assert (
        len(configs) == 1
    ), f"A config with the expected name was not created by dump {list(dump_folder.iterdir())}"

    loaded_dict = json.loads(configs[0].read_text())
    assert isinstance(loaded_dict, dict)
    assert all([isinstance(v, dict) for v in loaded_dict.values()])
    assert loaded_dict["job"]["description"] == description


def test_config_dump_remote(monkeypatch: pytest.MonkeyPatch, tmp_path):
    test_name = "funky-tester-39"
    monkeypatch.setenv("WANDB_NAME", test_name)
    config = ConfigManager().parse_args(
        [f"--job.dump_folder={tmp_path}", "--job.remote_folder=OBJECT_STORE_URI"]
    )
    # Dump & remote folder with test name in it.
    assert config.job.dump_folder == str(tmp_path / test_name)
    assert config.job.remote_folder == f"OBJECT_STORE_URI"


@pytest.mark.parametrize("converters", ["float6", "float6,mxfp"])
def test_config__parse_string_list__cmd_args(converters):
    config = ConfigManager().parse_args([f"--model.converters={converters}"])

    assert isinstance(config.model.converters, list)
    assert len(config.model.converters) == converters.count(",") + 1
    assert ",".join(config.model.converters) == converters


@pytest.mark.parametrize("converters", ["float6", "float6,mxfp"])
def test_config__parse_string_list__env_args(monkeypatch, converters):
    monkeypatch.setenv("MODEL_CONVERTERS", converters)
    config = ConfigManager().parse_args([])

    assert isinstance(config.model.converters, list)
    assert len(config.model.converters) == converters.count(",") + 1
    assert ",".join(config.model.converters) == converters


def test_config__modified_defaults():
    # Check several different ways of building the defaults.
    assert Checkpoint().keep_latest_k == 0
    assert Checkpoint().keep_latest_k == 0

    assert JobConfig().checkpoint.keep_latest_k == 0
    assert JobConfig().model.name == "llama3_gc"

    config = ConfigManager().parse_args([])
    assert config.checkpoint.keep_latest_k == 0
    assert JobConfig().model.name == "llama3_gc"


def test_config__additional_fields():
    config = JobConfig()
    # Additional fields added to TorchTitan existing sub-configs.
    assert config.job.remote_folder is None
    assert config.job.experimental_modules == []
    assert config.metrics.distributed_mode == "local_rank_0"
    assert config.training.load_dataset_kwargs == "{}"


def test_config__additional_profiling_fields():
    config = JobConfig()
    # Check the defaults profiling fields
    assert config.profiling.consecutive_active_steps == 1
    assert config.profiling.memory_timeline is None
    assert not config.profiling.with_flops_table
    assert not config.profiling.with_summary_metrics
    assert not config.profiling.experimental_cupti_stats
    assert not config.profiling.capture_compiled_kernels


TEST_ASSETS = Path(__file__).resolve().parents[1] / "assets"
TEST_CONFIGS = Path(__file__).resolve().parents[2] / "train_configs"


@pytest.mark.parametrize(
    "config_path",
    [
        *(TEST_ASSETS / "old-configs/").glob("*.toml"),
        *(TEST_ASSETS / "old-configs/").glob("*.json"),
    ],
)
def test_config_upgrade(config_path: str):
    """Checks that all configs in the old_configs directory can still be loaded"""
    config_path = str(config_path)
    config = ConfigManager().parse_args(
        ["--job.config_file", config_path], allow_upgrade=True
    )
    # Check that old config arguments are properly upgraded to new ones.
    assert config.lr_scheduler.decay_type == "cosine"


@pytest.mark.parametrize(
    "filename", sorted([str(f) for f in TEST_CONFIGS.glob("*.toml")])
)
def test__config__load_all_train_configs(filename):
    # All config files should be up to date. Not allowing upgrade!
    ConfigManager().parse_args(["--job.config_file", filename], allow_upgrade=False)
    # Try with util method as well.
    cfg = load_config(filename)
    # No problem with pass through
    assert load_config(cfg) is cfg

    # Make sure we have moved to using `hf_assets_path` in config.
    assert cfg.model.tokenizer_path is None
    assert cfg.model.hf_assets_path is not None
