#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Dict, Any, List, Union

import os
import subprocess
import pytest
import torch

import numpy as np
import pandas as pd


def get_wandb_offline_path(result):
    """Extract the wandb offline path from the logs."""
    # Logs in `stderr` for some reason?
    lines = [v.strip() for v in result.stderr.split("\n")]
    # A bit manual parsing!
    wb_substr = "wandb sync"
    line = next((line for line in lines if wb_substr in line), None)
    if line is None:
        raise RuntimeError(
            f"Can not find wandb offline cache directory in logs: {result.stderr}"
        )

    idx = line.find(wb_substr)
    path = line[idx + len(wb_substr) :].strip()
    return path


def load_wandb_offline_metrics(path: str) -> List[Dict[str, Any]]:
    """Load wandb metrics from offline cache."""
    import wunderbar

    def _get_metrics(r):
        d = {k: v for k, v in r.data["item"].items()}
        d["step"] = int(r.data["step"]["num"])
        return d

    filename = next((f for f in os.listdir(path) if f.endswith(".wandb")))
    filename = os.path.join(path, filename)
    # Keep "history" ones corresponding to wandb.log calls.
    records = wunderbar.parse_filepath(path=filename)
    records = [r for r in records if r.type == "history"]
    metrics = [_get_metrics(r) for r in records]
    return metrics


def run_train(command_args: Union[str, Dict[str, Any]], env_vars=None):
    """
    Run the llama training script with the given arguments and environment variables.

    Args:
        command_args (str): Arguments to pass to run_train.sh (or dict)
        env_vars (dict, optional): Environment variables to set for the command

    Returns:
        subprocess.CompletedProcess: The result of the subprocess run
    """
    if isinstance(command_args, dict):
        command_args = " ".join(
            [
                f"--{k} {v}" if not isinstance(v, bool) else f"--{k}"
                for k, v in command_args.items()
            ]
        )

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    # The master port can cause problems in the Asynchronous checkpointers if multiple tests
    # run in parallel. We use a base port and add a multiplier based on the worker
    # ID to ensure each worker uses a different port. Similar to the no_distribution fixture.
    # This is only a problem in tests.
    BASE_MASTER_PORT = 20000
    pytest_worker_id = int(os.getenv("PYTEST_XDIST_WORKER", "gw0").strip("gw"))
    env["MASTER_PORT"] = str(BASE_MASTER_PORT + pytest_worker_id * 5)

    full_command = f"./run_train.sh {command_args}"
    result = subprocess.run(
        full_command, shell=True, capture_output=True, text=True, env=env
    )

    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    return result


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_file, number_of_gpus, expected_out",
    [
        (
            "./train_configs/debug_model.toml",
            1,
            "Building 0-D device mesh",
        ),
        (
            "./train_configs/debug_model.toml",
            2,
            "Building 1-D device mesh with ['dp_shard'], [2]",
        ),
        (
            "./train_configs/deepseek_v3_debug_model.toml",
            1,
            "Building 0-D device mesh",
        ),
        (
            "./train_configs/deepseek_v3_debug_model.toml",
            2,
            "Building 1-D device mesh with ['dp_shard'], [2]",
        ),
    ],
    ids=[
        "debug_model_1gpu",
        "debug_model_2gpu",
        "deepseek_v3_debug_1gpu",
        "deepseek_v3_debug_2gpu",
    ],
)
def test_debug_model_on_n_devices(config_file, number_of_gpus, expected_out):
    """Test running different configs on 1 and 2 devices."""
    if torch.cuda.device_count() < number_of_gpus:
        pytest.skip("Skip test, requiring multiple GPUs.")

    result = run_train(
        f"--wandb.mode offline --job.config_file {config_file}",
        env_vars={"NGPU": str(number_of_gpus)},
    )
    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode} and stderr {result.stderr}"
    assert expected_out in result.stdout or expected_out in result.stderr


@pytest.mark.integration
def test_summary_metrics(tmp_path):
    """Test summary_metrics on 1 device using tmp_path fixture."""
    result = run_train(
        f"--wandb.mode offline --job.config_file ./train_configs/debug_model.toml --profiling.enable_profiling --profiling.with_summary_metrics --job.dump_folder={tmp_path}",
        env_vars={"NGPU": "1"},
    )
    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode} and stderr {result.stderr}"

    # Look for summary_metrics json files
    summary_files = list(tmp_path.rglob("summary_metrics.json"))
    assert (
        len(summary_files) > 0
    ), f"Should find at least one summary_metrics.json file in {tmp_path}. Found files: {list(tmp_path.rglob('*.json'))}"


@pytest.mark.integration
def test_expected_failure():
    """Test a configuration that is expected to fail."""
    result = run_train(
        "--wandb.mode offline --job.config_file ./train_configs/nonexistent_config.toml",
        env_vars={"NGPU": "1"},
    )
    assert (
        result.returncode != 0
    ), "Command unexpectedly succeeded when it should have failed"
    assert (
        "No such file or directory" in result.stderr or "not found" in result.stderr
    ), "Expected error message not found in stderr"


@pytest.mark.xfail(reason="TODO: investigate why broken by 2.9")
@pytest.mark.integration
@pytest.mark.parametrize("number_of_gpus", [2, 1])
def test_ema_weight_averaging(tmp_path, number_of_gpus):
    """We test with 1 and 2 GPUs to make sure the output checkpoint is correctly sharded"""
    if torch.cuda.device_count() < number_of_gpus:
        pytest.skip("Skip test, requiring multiple GPUs.")

    result = run_train(
        "--wandb.mode offline --job.config_file ./train_configs/debug_model.toml "
        "--ema_checkpoint.enable_checkpoint --ema_checkpoint.save_interval 10 --ema_checkpoint.skip_first_k_updates=0 "
        "--checkpoint.enable --checkpoint.interval 20 --checkpoint.async_mode async "
        f" --job.dump_folder {tmp_path} --wandb.name ema_test",
        env_vars={"NGPU": str(number_of_gpus)},
    )
    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode} and stderr {result.stderr}"
    for step in range(10, 50, 10):
        expected_out = f"Requesting EMA checkpoint save at step {step}"
        assert expected_out in result.stdout or expected_out in result.stderr
    ema_checkpoint_dir = tmp_path / "ema_test" / "ema_checkpoint"
    assert ema_checkpoint_dir.exists(), "EMA checkpoint directory not created"
    for step in range(10, 50, 10):
        step_dir = ema_checkpoint_dir / f"step-{step}"
        checkpoint_files = list(step_dir.glob("*.distcp"))
        assert len(checkpoint_files) == number_of_gpus
        assert (step_dir / ".metadata").exists()


@pytest.mark.integration
@pytest.mark.parametrize("number_of_gpus", [2, 1])
def test_checkpoint(tmp_path, number_of_gpus):
    """We test with 1 and 2 GPUs to make sure the output checkpoint is correctly sharded"""
    if torch.cuda.device_count() < number_of_gpus:
        pytest.skip("Skip test, requiring multiple GPUs.")

    result = run_train(
        "--wandb.mode offline --job.config_file ./train_configs/debug_model.toml "
        "--checkpoint.enable --checkpoint.interval 10 --checkpoint.async_mode async "
        " --training.steps 25 "
        " --model.flavor unit_test --model.hf_assets_path tests/assets/test-tokenizer"
        f" --job.dump_folder {tmp_path} --wandb.name checkpoint_test",
        env_vars={"NGPU": str(number_of_gpus)},
    )
    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode} and stderr {result.stderr}"
    checkpoint_dir = tmp_path / "checkpoint_test" / "checkpoint"
    assert checkpoint_dir.exists(), "Checkpoint directory not created"
    problems = []
    for step in [10, 20, 25]:
        step_dir = checkpoint_dir / f"step-{step}"
        checkpoint_files = list(step_dir.glob("*.distcp"))
        if not len(checkpoint_files) == number_of_gpus:
            problems.append(
                f"Step {step} missing some checkpoint files, found: {checkpoint_files}"
            )
        if not (step_dir / ".metadata").exists():
            problems.append(f"Step {step} missing the metadata file")
    assert not problems, "Errors: " + ", ".join(problems)
    expected_out = "Finished saving the checkpoint"
    # Only expect 2 because the log line for the last checkpoint is different.
    matches = result.stdout.count(expected_out) + result.stderr.count(expected_out)
    assert matches == 2

    restart_result = run_train(
        "--wandb.mode offline --job.config_file ./train_configs/debug_model.toml "
        "--checkpoint.enable --checkpoint.interval 10 --checkpoint.async_mode async "
        " --training.steps 30 "
        " --model.flavor unit_test --model.hf_assets_path tests/assets/test-tokenizer"
        f" --job.dump_folder {tmp_path} --wandb.name checkpoint_test",
        env_vars={"NGPU": str(number_of_gpus)},
    )
    # Most outputs are in stderr
    output = restart_result.stdout + restart_result.stderr
    assert "step: 19" not in output
    assert "step: 26" in output
    assert "Training starts at step 26" in output


@pytest.mark.integration
def test__debug_model__deterministic_training_curve(tmp_path):
    """Motivation for this integration test: when upgrading TorchTitan or Torch,
    we should have a fully reproducible training of the debug model given:
    * C4 test data is not changed;
    * Model initialization is the same (frozen seed);

    This test is checking that our training is staying very close to a frozen baseline.
    """
    exp_metrics_path = "./tests/assets/test-training-traces/debug_model_metrics.pq"
    num_gpus = 1
    result = run_train(
        {
            "job.config_file": "./train_configs/debug_model.toml",
            "job.dump_folder": tmp_path,
            "model.flavor": "debugmodel_old",
            "wandb.mode": "offline",
            "wandb.name": "debug_model__deterministic_training_curve",
            "training.steps": 100,
            "debug.seed": 1472,  # deterministic initialization.
            "debug.deterministic": True,
        },
        env_vars={"NGPU": str(num_gpus)},
    )
    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode} and stderr {result.stderr}"

    wb_path = get_wandb_offline_path(result)
    metrics_df = pd.DataFrame(load_wandb_offline_metrics(wb_path))
    exp_metrics_df = pd.read_parquet(exp_metrics_path)

    loss_key = "loss_metrics/global_avg_loss"
    loss = np.asarray(metrics_df[loss_key])
    exp_loss = np.asarray(exp_metrics_df[loss_key])

    # Loss relative difference
    mean_loss_rel_diff = np.mean(2 * np.abs(loss - exp_loss) / (loss + exp_loss))
    # 0.1% maximum loss difference.
    assert mean_loss_rel_diff <= 1e-3
