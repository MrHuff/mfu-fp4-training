# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import subprocess
from pathlib import Path
import os
import shutil
import sys

import torch
import pytest
from low_bits_training.config import JobConfig, ConfigManager


TEST_ROOT = Path(__file__).resolve().parents[0]
REPO_ROOT = TEST_ROOT.parent

TEST_ASSET_DIRECTORY = TEST_ROOT / "assets/"


def generate_test_asset(
    base_config_path="train_configs/debug_model.toml",
    folder_name="test-checkpoint",
    other_args="",
    clear_old=False,
):
    if clear_old and (TEST_ASSET_DIRECTORY / folder_name).exists():
        shutil.rmtree(TEST_ASSET_DIRECTORY / folder_name)
    expected_asset = TEST_ASSET_DIRECTORY / folder_name / "checkpoint/step-0/"
    expected_file = expected_asset / "__0_0.distcp"
    if expected_asset.exists() and expected_asset.is_dir() and expected_file.exists():
        return expected_asset
    if not torch.cuda.is_available():
        pytest.skip(
            "No GPU available - a GPU is required to generate the seed checkpoint - this file is expected to be commited"
        )
    env = os.environ.copy()
    env.update(
        {
            "NGPU": "1",
            "CONFIG_FILE": str(base_config_path),
            "WANDB_MODE": "disabled",
            "WANDB_NAME": str(folder_name),
        }
    )
    command = (
        f"{sys.executable} -m torch.distributed.launch --use-env --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --local-ranks-filter 0 --role rank --tee 3 "
        f"train.py --job.config_file {str(base_config_path)}  --checkpoint.enable --checkpoint.create_seed_checkpoint "
        "--parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 1"
        " --parallelism.pipeline_parallel_degree 1 --parallelism.context_parallel_degree 1 "
        f"--job.dump_folder {TEST_ASSET_DIRECTORY.relative_to(TEST_ROOT.parent)} {other_args}"
    )
    try:
        test_asset_out = subprocess.check_output(
            args=command.strip().split(), env=env, text=True, cwd=TEST_ROOT.parent
        )
    except subprocess.CalledProcessError as e:
        err_str = "Error running command: "
        err_str += str(command)
        err_str += "Output: "
        err_str += str(e.output)
        err_str += "Return code: "
        err_str += str(e.returncode)
        err_str += "STDERR: "
        err_str += str(e.stderr)
        raise RuntimeError(err_str) from e
    print(test_asset_out)
    assert expected_asset.exists() and expected_asset.is_dir() and expected_file.exists()
    return expected_asset


@pytest.fixture(
    scope="session",
    params=["test-checkpoint-unit_test", "test-checkpoint-unit_test-lbt-v0"],
)  # session scope to avoid race conditions
def unit_test_checkpoint(request) -> Path:
    """Creates a 'unit_test' model in the test assets folder

    Requires a GPU to regenerate.
    """
    checkpoint_name = request.param
    if "old" in checkpoint_name:
        checkpoint_path = TEST_ASSET_DIRECTORY / checkpoint_name / "checkpoint/step-0/"
        assert checkpoint_path.exists() and checkpoint_path.is_dir()
        assert (checkpoint_path / "__0_0.distcp").exists()
    else:
        checkpoint_path = generate_test_asset(
            base_config_path="train_configs/debug_model.toml",
            folder_name=checkpoint_name,
            other_args="--model.flavor unit_test --training.seq_len 16 --model.tokenizer_path ./tests/assets/test-tokenizer",
            clear_old=False,
        )
    return checkpoint_path


@pytest.fixture
def no_distribution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "0.0.0.0")
    # Use different ports on each worker to avoid resource contention do *5 as
    # multiple ports are used for different distributed process groups.
    pytest_worker_id = int(os.getenv("PYTEST_XDIST_WORKER", "gw0").strip("gw"))
    monkeypatch.setenv("MASTER_PORT", str(12348 + pytest_worker_id * 5))
    yield
    # We assume that tests that need these environment variables will try
    # to initialise a process group. Depending on what they do, they might
    # not destroy it, make sure we clean up for other tests
    try:
        torch.distributed.destroy_process_group()
    except AssertionError:
        # torch checks with an assert: assert pg is not None
        print("Process group was not initialised")


@pytest.fixture
def config_and_checkpoint(unit_test_checkpoint: Path) -> tuple[JobConfig, Path]:
    config_files = list(unit_test_checkpoint.parents[1].glob("*config*json"))
    assert config_files

    config = ConfigManager().parse_args(
        ["--job.config_file", str(config_files[0])], allow_upgrade=True
    )
    return config, unit_test_checkpoint
