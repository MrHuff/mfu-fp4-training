#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
from pathlib import Path
from copy import deepcopy
from unittest import mock

import pytest
from torch import nn

from low_bits_training import ema_checkpoint
from low_bits_training.models.models import get_model_config
from low_bits_training.models.llama3 import Transformer
from low_bits_training.config import JobConfig, ConfigManager

# Make sure fixtures are loaded

TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TEST_ROOT.parent

TEST_ASSET_DIRECTORY = TEST_ROOT / "assets/"


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def ema_job_config(no_distribution, tmp_path):
    job_config = ConfigManager().parse_args(
        [
            "--ema_checkpoint.enable_checkpoint",
            "--job.dump_folder",
            str(tmp_path),
            "--wandb.name",
            "ema_test",
            "--ema_checkpoint.save_interval",
            "1",
            "--ema_checkpoint.ema_decay",
            "0.5",
            "--ema_checkpoint.skip_first_k_updates=0",
        ]
    )
    return job_config


def test_ema_checkpoint(ema_job_config):
    """Check that the EMACheckpoint manager produces the expected values

    Does it on a small model with a single 4 by 4 linear layer.
    All weights start at 0 and then are kept at 1.0. The EMA should asymptotically
    converge to 1.0 with a decay of 0.5.

    We load the checkpoints and check the values are what we expect.
    """
    # Note for some reason the test does not work with the VSCode test debugger
    ema_job_config.ema_checkpoint.ema_decay = decay = 0.5
    model = MockModel()
    with torch.no_grad():
        model.linear.weight.data.fill_(0.0)
    checkpointer = ema_checkpoint.EMACheckpointManager(
        model_parts=[model],
        checkpoint_config=ema_job_config.checkpoint,
        base_folder=ema_job_config.job.dump_folder,
    )
    checkpointer.update_ema()
    expected_values = [0.0]
    checkpointer.save(0)
    print("CPU offload state dict before fill_:", checkpointer.cpu_offload_state_dict)
    model.linear.weight.data.fill_(1.0)
    print("CPU offload state dict after fill_:", checkpointer.cpu_offload_state_dict)
    for i in range(5):
        expected_values.append(expected_values[-1] * decay + 1.0 * (1 - decay))
        checkpointer.update_ema()
        checkpointer.save(i + 1)
    checkpointer.wait_for_save_completion()

    # Load checkpoints and check the values corresponds to what we expect
    run_path = Path(ema_job_config.job.dump_folder) / ema_job_config.ema_checkpoint.folder
    assert run_path.exists(), f"Run path {run_path} does not exist"
    for i in range(6):
        checkpoint_dir = run_path / f"step-{i}"
        assert checkpoint_dir.exists()
        ema_checkpoint.dcp_load_helper(
            {"model": model.state_dict()}, checkpoint_id=str(checkpoint_dir)
        )
        assert (model.linear.weight.data == expected_values[i]).all()

    # Now we test the ability to load the checkpoint state
    # We update the internal state of the checkpointer then load a checkpoint and dump it immediately
    # That new checkpoint should have the same values as the one we loaded
    copy_step = 0
    new_step = 6
    checkpointer.update_ema()
    checkpointer.update_ema()
    checkpointer.update_ema()
    checkpointer.load(run_path / f"step-{copy_step}")
    checkpointer.save(new_step)
    checkpointer.wait_for_save_completion()
    ema_checkpoint.dcp_load_helper(
        {"model": model.state_dict()}, checkpoint_id=str(run_path / f"step-{new_step}")
    )
    assert (
        model.linear.weight.data == expected_values[copy_step]
    ).all(), "Loaded checkpoint does not match expected values"

    checkpointer.close()


class TestEMACheckpointTimeouts:
    """The EMACheckpointer should not throw errors and should respect its timeout"""

    small_timeout = 0.001

    def test_init_timeout(
        self,
        ema_job_config,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        # Monkeypatch the class timeout to trigger the timeout in the init
        monkeypatch.setattr(
            ema_checkpoint.EMACheckpointManager, "mp_timeout", self.small_timeout
        )
        checkpointer = ema_checkpoint.EMACheckpointManager(
            model_parts=[MockModel()],
            checkpoint_config=ema_job_config.checkpoint,
            base_folder=ema_job_config.job.dump_folder,
        )
        assert (
            not checkpointer.enable_checkpoint
        ), "The checkpointer should have killed itself"
        captured = "\n".join([c.msg for c in caplog.get_records("call")])

        # Check that we exited how we expect
        assert (
            f"EMA Checkpoint thread failed to start in {self.small_timeout}" in captured
        )
        # This effectively tests the close timeout as well
        assert (
            "EMACheckpoint worker thread did not finish within timeout - force closing"
            in captured
        )
        assert "Force killing EMACheckpoint process" in captured

    def test_update_timeout(self, ema_job_config, caplog: pytest.LogCaptureFixture):
        checkpointer = ema_checkpoint.EMACheckpointManager(
            model_parts=[MockModel()],
            checkpoint_config=ema_job_config.checkpoint,
            base_folder=ema_job_config.job.dump_folder,
        )
        checkpointer.mp_timeout = self.small_timeout
        checkpointer.max_pending_updates = 0
        checkpointer.update_ema()
        checkpointer.save(curr_step=0)
        assert (
            not checkpointer.enable_checkpoint
        ), "The checkpointer should have killed itself"

        captured = "\n".join([c.msg for c in caplog.get_records("call")])
        assert "EMACheckpointer failed to update, turning off EMA updates" in captured


@pytest.fixture
def dummy_checkpoint_kwargs():
    dummy_dataloader = mock.Mock()
    dummy_dataloader.state_dict = mock.Mock(side_effect=lambda: {"dataloader": 1})
    model_args = get_model_config("llama3_gc", "unit_test")
    # print(model_args)
    model_args.vocab_size = 300
    model_args.max_seq_len = 16
    dummy_model_parts = [Transformer(model_args)]
    dummy_optimizers = mock.Mock()
    dummy_optimizers.state_dict = mock.Mock(side_effect=lambda: {"optimizer": 3})
    dummy_lr_schedulers = mock.Mock()
    dummy_lr_schedulers.state_dict = mock.Mock(side_effect=lambda: {"lr_scheduler": 4})
    ft_manager = mock.Mock()
    ft_manager.enabled = False
    trainer_state = mock.Mock()
    trainer_state.state_dict = mock.Mock(side_effect=lambda: {"my_state": 765})
    return {
        "dataloader": dummy_dataloader,
        "model_parts": dummy_model_parts,
        "optimizers": dummy_optimizers,
        "lr_schedulers": dummy_lr_schedulers,
        "ft_manager": ft_manager,
        "states": {"trainer": trainer_state},
        "sd_adapter": None,
    }


@pytest.fixture
def mock_ema_cuda_calls():
    """Mock the calls to the cuda functions used in the EMACheckpointManager.

    This reduces our reliance on GPU resources for the test."""
    # TODO: in order for the test to be moved to CPU we also need to mock the process worker in the
    # `low_bits_training.ema_checkpoint.EMACheckpointManager` class. This can be done once the worker is
    # refactored into a class which separates the process management from the action logic.
    dcp_load_patch = mock.patch("low_bits_training.ema_checkpoint.dcp.load")
    # stream_patch = mock.patch("low_bits_training.ema_checkpoint.torch.cuda.Stream")
    # stream_context_patch = mock.patch(
    #     "low_bits_training.ema_checkpoint.torch.cuda.stream"
    # )
    copy_state_dict_patch = mock.patch(
        "low_bits_training.ema_checkpoint._copy_state_dict"
    )

    def copy_state_dict_patch_decorator(src, trg, *_, **__):
        for key, value in src.items():
            if isinstance(value, dict):
                trg[key] = copy_state_dict_patch_decorator(value, trg.get(key, {}))
            else:
                trg[key] = deepcopy(value)
        return trg

    copy_state_dict_patch.decorate_callable(copy_state_dict_patch_decorator)
    mocks = {
        ## These would need to be patched in the cpu test
        # "torch.cuda.Stream": stream_patch.start(),
        # "torch.cuda.stream": stream_context_patch.start(),
        # "_copy_state_dict": copy_state_dict_patch.start(),
        "dcp.load": dcp_load_patch.start(),
    }
    yield mocks
    for mock_obj in mocks.values():
        mock_obj.stop()


@pytest.fixture
def config_and_checkpoints(config_and_checkpoint) -> tuple[str, tuple[Path, Path]]:
    return config_and_checkpoint[0], (config_and_checkpoint[1], config_and_checkpoint[1])


def test_ema_checkpoint_load(
    no_distribution,
    config_and_checkpoints: tuple[JobConfig, tuple[Path, Path]],
    dummy_checkpoint_kwargs,
    mock_ema_cuda_calls: dict[str, mock.Mock],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
):
    """Checks the EMA checkpoint loads the expected step, when:
    - No step is passed
    - A specific step is passed
    """
    config, (checkpoint, ema_checkpoint_path) = config_and_checkpoints
    config.ema_checkpoint.enable_checkpoint = True
    config.checkpoint.enable = True
    config.ema_checkpoint.skip_first_k_updates = 0
    config.ema_checkpoint.save_interval = 1

    ema_path: Path = tmp_path / config.ema_checkpoint.folder
    ckpt_path: Path = tmp_path / config.checkpoint.folder
    ema_path.mkdir(parents=True)
    ckpt_path.mkdir(parents=True)
    (ema_path / "step-0").symlink_to(ema_checkpoint_path, target_is_directory=True)
    (ckpt_path / "step-0").symlink_to(checkpoint, target_is_directory=True)

    assert (ema_path / "step-0/.metadata").exists()
    assert (ckpt_path / "step-0/.metadata").exists()

    checkpointer = ema_checkpoint.CheckpointManagerWithEMAWeight(
        **dummy_checkpoint_kwargs,
        checkpoint_config=config.checkpoint,
        base_folder=tmp_path,
    )
    caplog.set_level("INFO")
    dcp_mock = mock_ema_cuda_calls["dcp.load"]
    caplog.clear()

    # ====================================
    # Test different loading scenarios
    # ====================================

    # Check that we have loaded step 0 for EMA and normal checkpoints
    has_loaded = checkpointer.load()
    assert (
        len(dcp_mock.mock_calls) == 1
    ), "dcp.load should have been called for the checkpoint loading (ema checkpoint is in a spawned subprocess and cannot be mocked)"
    assert has_loaded, "Checkpoint should have been loaded"
    assert "Loading EMA weights from" in caplog.text
    assert "step-0" in caplog.text
    caplog.clear()
    assert not caplog.text

    # Check that we load step 0 for EMA and normal checkpoints as step 1 is only in the normal checkpoint
    (ckpt_path / "step-1").symlink_to(checkpoint, target_is_directory=True)
    has_loaded = checkpointer.load()
    assert has_loaded, "Checkpoint should have been loaded"
    assert "Loading EMA weights from" in caplog.text
    assert "step-0" in caplog.text
    caplog.clear()
    assert not caplog.text

    has_loaded = checkpointer.load(1)
    assert has_loaded, "Checkpoint should have been loaded for step 1"
    assert "does not exist for EMA weights" in caplog.text
    assert "Loading EMA weights from" not in caplog.text
    assert "step-1" in caplog.text
    caplog.clear()
    assert not caplog.text

    # Try loading a step that does not exist, in that case we should not load anything
    with pytest.raises(FileNotFoundError):
        has_loaded = checkpointer.load(2)
        # assert not has_loaded
    caplog.clear()
    assert not caplog.text

    # When EMA and normal checkpoints have common steps - we load from the latest common one
    # In this case: step 0 as only normal checkpoint has step 1
    (ema_path / "step-2").symlink_to(ema_checkpoint_path, target_is_directory=True)
    has_loaded = checkpointer.load()
    assert has_loaded
    assert "Loading EMA weights from" in caplog.text
    assert (
        "step-0" in caplog.text
    ), "Should have loaded from step-0 as it is the only step in common between normal and EMA checkpoints"
    caplog.clear()
    assert not caplog.text

    # When EMA and normal checkpoints have common steps - we load from the latest common one - now step 2
    (ckpt_path / "step-2").symlink_to(checkpoint, target_is_directory=True)
    has_loaded = checkpointer.load()
    assert has_loaded
    assert "Loading EMA weights from" in caplog.text
    assert "step 2 from both normal and EMA" in caplog.text
    assert (
        "step-2" in caplog.text
    ), "Should have loaded from step-2 as it is the only step in common between normal and EMA checkpoints"
    caplog.clear()
    assert not caplog.text

    # When EMA and normal checkpoints do not have common steps - we don't load
    (ckpt_path / "step-2").unlink()
    (ema_path / "step-0").unlink()
    has_loaded = checkpointer.load()
    assert not has_loaded
    assert "do not have common steps to load" in caplog.text
    caplog.clear()
    assert not caplog.text
