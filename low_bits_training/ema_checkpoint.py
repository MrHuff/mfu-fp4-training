#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import gc
import copy
import enum
import os
import queue
import time
from typing import Any, Dict, List, Literal
import psutil
import re
from packaging.version import Version

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
import torch.distributed.checkpoint as dcp

from torchtitan.components import checkpoint
from torchtitan.tools.logging import init_logger, logger


import torchtitan.components
import torchtitan.components.checkpoint
import torchtitan.train
from torchtitan.components.checkpoint import CheckpointManager as TTCheckpointManager
from torchtitan.components.checkpoint import (
    CheckpointConfig,
    GarbageCollection,
    FTManager,
    BaseDataLoader,
    OptimizersContainer,
    LRSchedulersContainer,
    MODEL,
)
from torchtitan.protocols import BaseStateDictAdapter

from .config.job_config import EMACheckpoint as EMACheckpointConfig


if Version(torch.__version__) >= Version("2.8"):
    # Temporary workaround a PyTorch 2.8 bug: https://github.com/pytorch/pytorch/issues/160983
    # Can be removed once moving EMA logic to new `StateDictStager` class in `torch.distributed`.
    from torch.cuda._pin_memory_utils import unpin_memory as torch_unpin_memory

    def unpin_memory_fixed(ptr_or_tensor):
        if isinstance(ptr_or_tensor, torch.Tensor):
            return torch_unpin_memory(ptr_or_tensor.data_ptr())
        return torch_unpin_memory(ptr_or_tensor)

    # Patch Torch to support Tensor input.
    torch.cuda._pin_memory_utils.unpin_memory = unpin_memory_fixed

LBT_EMA_NUM_UPDATES_KEY = "number_of_updates"
# NOTE: check that this collection hasn't changed when updating torchtitan
# test_torchtitan_monkeypath.py::test_TT_SPECIAL_ENTRIES_STATE_DICT_ENTRIES_is_up_to_date
TT_SPECIAL_ENTRIES_STATE_DICT_ENTRIES = [
    torchtitan.components.checkpoint.MODEL,
    torchtitan.components.checkpoint.OPTIMIZER,
    torchtitan.components.checkpoint.LR_SCHEDULER,
    torchtitan.components.checkpoint.DATALOADER,
    torchtitan.components.checkpoint.TRAIN_STATE,
    LBT_EMA_NUM_UPDATES_KEY,
]

TensorDict = dict[str, torch.Tensor]


def dcp_load_helper(
    state_dict: TensorDict | dict[str, TensorDict],
    checkpoint_id: str,
    *,
    process_group=None,
):
    """Function to load new and old TorchTitan checkpoints into state dicts with and without
    flattened model states

    Old torchtitan checkpoints are in the format:
    - { "model.layers.0.weight", "optimizer."...
    And the new style Torchtitan state_dict flatten the model states:
    - { "layers.0.weight", "optimizer." ...

    This function translates the state dict to match the checkpoint format.
    """

    storage_reader = dcp.state_dict_loader.cast(
        dcp.StorageReader,
        dcp.state_dict_loader._storage_setup(None, checkpoint_id, reader=True),
    )

    def _is_model_key(key):
        return not any(
            key.startswith(tt_special)
            for tt_special in TT_SPECIAL_ENTRIES_STATE_DICT_ENTRIES
            if tt_special != torchtitan.components.checkpoint.MODEL
        )

    model_keys = [k for k in state_dict.keys() if _is_model_key(k)]

    metadata = storage_reader.read_metadata()
    checkpoint_is_nested = any(
        "model." in k and ".weight" in k for k in metadata.state_dict_metadata.keys()
    )
    state_dict_is_nested = "model" in model_keys[0]
    formatted_state_dict = {**state_dict}
    if checkpoint_is_nested and not state_dict_is_nested:
        logger.warning(
            "Nesting state_dict with 'model' because checkpoint is nested but model state dict is not."
        )
        model_state_dict = {k: formatted_state_dict.pop(k) for k in model_keys}
        formatted_state_dict["model"] = model_state_dict
    elif not checkpoint_is_nested and state_dict_is_nested:
        logger.warning(
            "Removing 'model' nesting because checkpoint is not nested but model state dict is. "
            "This behaviour is pending deprecation. Please update your code by removing the leading 'model.'"
            " prefix from state_dict entries related to the model's state."
        )
        if "model" in state_dict:
            formatted_state_dict.update(formatted_state_dict.pop("model"))
        else:
            formatted_state_dict = {
                k.replace("model.", ""): v for k, v in formatted_state_dict.items()
            }

    if process_group is not None:
        logger.info(
            "LBT DCP load planning control receipt: backend=%s rank=%d/%d",
            dist.get_backend(process_group),
            dist.get_rank(process_group),
            dist.get_world_size(process_group),
        )
    dcp.load(
        formatted_state_dict,
        checkpoint_id=checkpoint_id,
        process_group=process_group,
    )


def CheckpointManager_dcp_load_override(
    self,
    state_dict: dict[str, Any],
    checkpoint_id: str,
    from_hf: bool,
    from_quantized: bool,
) -> None:
    """Copy of CheckpointManager.dcp_load which swaps out dcp.load with dcp_load_helper"""
    if from_hf:
        _original_CheckpointManager_dcp_load(
            self, state_dict, checkpoint_id, from_hf, from_quantized
        )
    else:
        dcp_load_helper(
            state_dict,
            checkpoint_id=checkpoint_id,
            process_group=getattr(self, "_lbt_control_process_group", None),
        )  # this line is changed

        if MODEL in self.states:
            self.states[MODEL].load_state_dict(state_dict)


# We don't use inheritance here and directly modify the class to make sure that this
# carries over to all inheriting classes.
# TODO: once this implementation has been default for a few months we can move it behind
# a compatibility flag.
_original_CheckpointManager_dcp_load = TTCheckpointManager.dcp_load
TTCheckpointManager.dcp_load = CheckpointManager_dcp_load_override


_original_CheckpointManager_dcp_save = TTCheckpointManager.dcp_save


@torch.no_grad()
def CheckpointManager_dcp_save_override(
    self,
    state_dict: dict[str, Any],
    checkpoint_id: str,
    async_mode: checkpoint.AsyncMode,
    enable_garbage_collection: bool = False,
    to_hf: bool = False,
):
    """Route synchronous DCP save planning over the opt-in control group."""

    process_group = getattr(self, "_lbt_control_process_group", None)
    if (
        process_group is None
        or async_mode != checkpoint.AsyncMode.DISABLED
        or to_hf
    ):
        return _original_CheckpointManager_dcp_save(
            self,
            state_dict,
            checkpoint_id,
            async_mode,
            enable_garbage_collection,
            to_hf,
        )

    logger.info(
        "LBT DCP save planning control receipt: backend=%s rank=%d/%d",
        dist.get_backend(process_group),
        dist.get_rank(process_group),
        dist.get_world_size(process_group),
    )
    result = dcp.save(
        state_dict,
        checkpoint_id=checkpoint_id,
        process_group=process_group,
    )
    if enable_garbage_collection:
        GarbageCollection.collect("GC collection invoked by checkpointer.")
    return result


TTCheckpointManager.dcp_save = CheckpointManager_dcp_save_override


class EMACommand(enum.Enum):
    UPDATE = "update"
    SAVE = "save"
    TERMINATE = "terminate"
    LOAD = "load"


@torch.no_grad()
def save_with_gc(state, checkpoint_id):
    dcp.save(state, checkpoint_id=checkpoint_id)
    GarbageCollection.collect("GC collection invoked by checkpointer.")


def ema_checkpoint_process(
    recv: mp.Queue, send: mp.Queue, ema_decay: float, export_dtype: torch.dtype
):
    """Process that maintains EMA weights and saves checkpoints.

    This process receives model state dicts, maintains EMA weights,
    and saves checkpoints when requested.

    Args:
        recv: Queue to receive commands and state dicts
        send: Queue to send completion signals
        ema_decay: EMA decay factor
        export_dtype: Data type for weight export
    """
    init_logger()
    os.environ["MASTER_PORT"] = str(
        int(os.environ["MASTER_PORT"]) + 3
    )  # +2 is the regular checkpointer
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    assert os.getenv("WORLD_SIZE")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()

    ema_state_dict = None
    # For the first period of the EMA we use a simple moving average
    # This is to avoid the EMA weights being overly influenced by the
    # initial weights.
    number_of_sma_updates = 1 / (1 - ema_decay)
    try:
        send.put("started_successfully")
        while True:
            # Receive command
            cmd_data = recv.get()

            if cmd_data[0] == EMACommand.TERMINATE:
                logger.info("Terminating EMA checkpoint process.")
                return

            elif cmd_data[0] == EMACommand.UPDATE:
                # Update EMA weights
                _, state_dict = cmd_data
                if ema_state_dict:
                    ema_state_dict[LBT_EMA_NUM_UPDATES_KEY] += 1

                if ema_state_dict is None:
                    # Initialize EMA state dict on first update
                    ema_state_dict = {LBT_EMA_NUM_UPDATES_KEY: 1}

                    def recursive_copy(src, target):
                        for key, value in src.items():
                            if isinstance(value, torch.Tensor):
                                target[key] = value.detach().clone()
                            if isinstance(value, dict):
                                target[key] = {}
                                recursive_copy(value, target[key])
                            else:
                                target[key] = copy.deepcopy(value)

                    recursive_copy(state_dict, ema_state_dict)
                    logger.info(
                        "Initialized EMA state dict with keys: %s", ema_state_dict
                    )
                elif ema_state_dict[LBT_EMA_NUM_UPDATES_KEY] < number_of_sma_updates:
                    # For the first few updates, use a simple moving average
                    def recursive_sma_update(src, target, num_updates):
                        for key, value in src.items():
                            if isinstance(value, torch.Tensor):
                                target[key].mul_((num_updates - 1) / num_updates).add_(
                                    value, alpha=1.0 / num_updates
                                )
                            elif isinstance(value, dict):
                                recursive_sma_update(value, target[key], num_updates)

                    with torch.no_grad():
                        recursive_sma_update(
                            state_dict,
                            ema_state_dict,
                            ema_state_dict[LBT_EMA_NUM_UPDATES_KEY],
                        )
                        logger.debug(
                            "Updated EMA state dict with simple moving average: %s",
                            ema_state_dict,
                        )
                else:
                    # Afterwards use EMA decay
                    def recursive_ema_update(src, target, ema_decay):
                        for key, value in src.items():
                            if isinstance(value, torch.Tensor):
                                target[key].mul_(ema_decay).add_(
                                    value, alpha=1.0 - ema_decay
                                )
                            elif isinstance(value, dict):
                                recursive_ema_update(value, target[key], ema_decay)

                    with torch.no_grad():
                        recursive_ema_update(state_dict, ema_state_dict, ema_decay)
                        logger.debug(
                            "Updated EMA state dict with decay: %s", ema_state_dict
                        )
                # Signal update complete
                send.put("update_done")

            elif cmd_data[0] == EMACommand.SAVE:
                # Save checkpoint with EMA weights
                _, checkpoint_id = cmd_data

                if ema_state_dict is None:
                    logger.warning("No EMA weights to save yet!")
                    send.put("save_done")
                    continue

                begin = time.monotonic()
                # Update model wrapper with EMA weights and save full state
                save_with_gc(ema_state_dict, checkpoint_id=checkpoint_id)
                gc.collect()
                logger.info(
                    f"Saved EMA checkpoint to {checkpoint_id} in {time.monotonic() - begin:.2f} seconds."
                )
                send.put("save_done")
            elif cmd_data[0] == EMACommand.LOAD:
                # Load EMA weights from checkpoint
                _, checkpoint_id = cmd_data
                assert ema_state_dict is not None, "EMA state dict is not initialized!"
                assert ema_state_dict, "EMA state dict is empty!"
                logger.info(f"Loading EMA weights from {checkpoint_id}")
                logger.info(
                    f"Starting load of checkpoint with keys {list(ema_state_dict)}"
                )

                try:
                    try:
                        dcp_load_helper(
                            ema_state_dict,
                            checkpoint_id=checkpoint_id,
                        )
                    except dcp.api.CheckpointException:
                        # Some checkpoints may not have the number_of_updates key - we load them anyway
                        num_updates = ema_state_dict.pop(LBT_EMA_NUM_UPDATES_KEY, None)
                        dcp_load_helper(
                            ema_state_dict,
                            checkpoint_id=checkpoint_id,
                        )
                        ema_state_dict[LBT_EMA_NUM_UPDATES_KEY] = num_updates
                    # If the number_of_updates key is missing, we initialize it to 1
                    if ema_state_dict.get(LBT_EMA_NUM_UPDATES_KEY) is None:
                        ema_state_dict[LBT_EMA_NUM_UPDATES_KEY] = 1
                        logger.warning(
                            "EMA state dict did not have 'number_of_updates' key, initializing to 1."
                        )
                    send.put("load_done")
                    logger.info(f"Loaded EMA weights from {checkpoint_id}")
                    logger.info(f"loaded checkpoint with keys {list(ema_state_dict)}")
                except Exception as e:
                    logger.error(f"Failed to load EMA weights: {e}")
                    send.put("load_failed")
    finally:
        logger.info("Destroying process group.")
        dist.destroy_process_group()


class EMACheckpointManager:
    """EMA checkpoint manager that performs all EMA math in a separate process.

    This manager sends model weights to a separate process that maintains
    the EMA weights and handles all checkpoint saving.

    Args:
        model_parts: List of model parts to track EMA for.
        job_config: Job configuration.
        ema_decay: The decay factor for EMA (default: 0.999).
        states: Optional additional states to save (default: None).
    """

    mp_timeout: float = 20.0
    max_pending_updates = 0  # Seemed like a good idea but isn't - I can't get the asyncness of it all to behave as I expect - at 0 it works - above 0 have fun debugging why the buffer is changing after being put in the queue.

    def __init__(
        self,
        model_parts: List[nn.Module],
        checkpoint_config: CheckpointConfig,
        base_folder: str,
        states: Dict[str, Any] | None = None,
    ):
        logger.warning("TODO: investigate why the EMA checkpointer fails with Torch 2.9.")
        self.model_parts = model_parts if isinstance(model_parts, list) else [model_parts]

        # Extract checkpoint config
        ckpt_config: EMACheckpointConfig = checkpoint_config.ema
        self.folder = os.path.join(base_folder, ckpt_config.folder)
        self.update_interval = ckpt_config.update_interval
        self.save_interval = ckpt_config.save_interval
        self.export_dtype = checkpoint.TORCH_DTYPE_MAP[ckpt_config.export_dtype]
        self.ema_decay = ckpt_config.ema_decay
        self.enable_checkpoint = ckpt_config.enable_checkpoint
        self.skip_first_k_updates = ckpt_config.skip_first_k_updates
        self._current_step_count = 0

        # A timeout which if it gets exceeded will shutdown the ema checkpointer process cleanly
        # set below 30 as 30 is the sigterm -> SIGKILL timeout in most orchestrators (SLURM, K8s)
        # Only support async with pinned memory for EMA
        async_mode = ckpt_config.async_mode.lower()
        if async_mode != checkpoint.AsyncMode.ASYNC_WITH_PINNED_MEM:
            raise ValueError(
                f"EMACheckpointManager only supports async_with_pinned_mem mode, got {async_mode}"
            )
        self.cpu_offload_state_dict = None
        if not self.enable_checkpoint:
            logger.info("EMACheckpointManager is disabled. No checkpoints will be saved.")
            return self
        self.staging_stream = torch.cuda.Stream()
        # Setup states
        self.states = {checkpoint.MODEL: checkpoint.ModelWrapper(model_parts)}
        if states:
            self.states.update(states)

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        self.mp_queue_send = ctx.Queue()
        self.mp_queue_recv = ctx.Queue()
        self.mp = ctx.Process(
            target=ema_checkpoint_process,
            args=(
                self.mp_queue_send,
                self.mp_queue_recv,
                self.ema_decay,
                self.export_dtype,
            ),
            daemon=True,
        )
        self.pending = {
            "update": 0,
            "save": 0,
            "load": 0,
        }
        start_time = time.time()
        self.mp.start()
        try:
            result = self.mp_queue_recv.get(timeout=self.mp_timeout)
            assert result == "started_successfully"
        except (queue.Empty, AssertionError):
            logger.error(
                f"EMA Checkpoint thread failed to start in {self.mp_timeout} - disabling EMA"
            )
            self.close()
            return
        logger.info(
            f"EMACheckpointManager initialized in {time.time() - start_time:.2f}s with decay={self.ema_decay}. "
            f"Checkpoints will be saved to {self.folder}"
        )

    def update_ema(self) -> None:
        """Send current model weights to EMA process for update.

        This is non-blocking - the update happens asynchronously in the
        separate process.
        """
        if not self.enable_checkpoint:
            return
        if not self.mp.is_alive():
            raise RuntimeError("EMA checkpoint process is dead!")
        self._current_step_count += 1
        if not self.warmup_period_finished():
            return

        time_to_get_enqueued_responses = 0.01
        if (
            wait_time := self._wait_for_pending_update(
                max_pending=self.max_pending_updates
            )
        ) > time_to_get_enqueued_responses:
            # TODO: actually implement update interval
            logger.warning(
                "EMA update queue was above the limit of %s and had to wait for %s seconds - consider increasing the --ema_checkpoint.update_interval (currently %s)",
                self.max_pending_updates,
                wait_time,
                self.update_interval,
            )

        # Get current state dict - same as orginal checkpointer _cpu_staging() method
        state_dict = TTCheckpointManager._flattened_model_states_sd(self, self.states)  # type: ignore
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True

        logger.debug("Staged CPU offload state dict: %s", self.cpu_offload_state_dict)
        # Send update command with state dict
        self.mp_queue_send.put_nowait((EMACommand.UPDATE, self.cpu_offload_state_dict))
        self.pending["update"] += 1

    def _wait_for_pending_update(self, max_pending=0) -> float:
        """Wait for any pending EMA update to complete.

        Returns the time spent waiting.
        """
        start_time = time.time()
        self.wait_for_action("update")
        return time.time() - start_time

    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        """Check if we should save at this step."""
        if force:
            return True
        if curr_step % self.save_interval == 0:
            return True
        return False

    def _create_checkpoint_id(self, step: int) -> str:
        """Create checkpoint path for the given step."""
        return os.path.join(self.folder, f"step-{step}")

    def warmup_period_finished(self) -> bool:
        """Check if "update_ema" has been called skip_first_k_updates times. Returns true if it has."""
        return self.skip_first_k_updates <= self._current_step_count

    def save(
        self, curr_step: int, force: bool = False, weights_only: bool = False
    ) -> None:
        """Request checkpoint save with current EMA weights.

        Args:
            curr_step: Current training step.
            force: Force save regardless of interval.
            weights_only: If True, only save model weights without other states.
        """
        if not self.enable_checkpoint:
            return
        if not self._should_save(curr_step, force):
            return

        if not self.mp.is_alive():
            raise RuntimeError("EMA checkpoint process is dead!")

        if not self.warmup_period_finished():
            logger.warning(
                "Skipping EMA save as warmup period is not finished yet. "
                f"Current step: {curr_step}, skip_first_k_updates: {self.skip_first_k_updates}"
            )
            return

        logger.info(f"Requesting EMA checkpoint save at step {curr_step}.")

        # Wait for any pending updates to complete
        try:
            self._wait_for_pending_update(max_pending=self.max_pending_updates)
        except (queue.Empty, RuntimeError) as error:
            logger.error(
                f"EMACheckpointer failed to update, turning off EMA updates, error was: {error}"
            )
            self.close()
            return

        checkpoint_id = self._create_checkpoint_id(curr_step)

        # Send save command
        self.mp_queue_send.put((EMACommand.SAVE, checkpoint_id))
        self.pending["save"] += 1
        # Note: We don't wait for save completion here to keep training async
        # The process will handle the save in the background
        # If force is True we might be on the last step before exiting.
        if force:
            self.wait_for_action("save")

    def load(self, checkpoint_id: str) -> None:
        """Request to load EMA weights from a checkpoint."""
        if not self.enable_checkpoint:
            return
        logger.info(f"Requesting EMA checkpoint load from {checkpoint_id}.")

        self._current_step_count = self.skip_first_k_updates
        # First update the EMA weights to make sure we have initialised the state
        # dict and the process is ready to load weights
        self.update_ema()
        self.mp_queue_send.put((EMACommand.LOAD, checkpoint_id))
        self.pending["load"] += 1

    def wait_for_action(self, action: Literal["update", "save", "load"]) -> None:
        """Wait for a specific action to complete."""
        if not self.enable_checkpoint:
            return
        if not self.mp.is_alive():
            raise RuntimeError("EMA checkpoint process is dead!")

        # Wait for the specific action to complete
        while self.pending[action] > 0:
            result = self.mp_queue_recv.get(timeout=self.mp_timeout)
            last_action = None
            if m := re.match(r"(.*)_done", result):
                last_action = m.group(1)
                if self.pending[last_action] > 0:
                    self.pending[last_action] -= 1
                    continue
            else:
                raise RuntimeError(
                    f"Unexpected response from EMA process: {result} - expected '{action}_done'"
                )
            if result == f"{action}_done":
                self.pending[action] -= 1

    def wait_for_save_completion(self) -> None:
        """Wait for any pending save operations to complete."""
        self.wait_for_action("save")

    def close(self) -> None:
        """Clean up resources."""
        if not self.enable_checkpoint:
            return

        if hasattr(self, "mp") and self.mp and self.mp.is_alive():
            try:
                # Wait for any pending operations
                self._wait_for_pending_update()
                self.wait_for_save_completion()
                # Terminate process
                self.mp_queue_send.put((EMACommand.TERMINATE,))
                self.mp.join(timeout=self.mp_timeout)
            except queue.Empty:
                logger.error(
                    "EMACheckpoint did not update or save withing timeout - force closing"
                )

        if hasattr(self, "mp") and self.mp and self.mp.is_alive():
            logger.error(
                "EMACheckpoint worker thread did not finish within timeout - force closing"
            )
            self.force_close()
        self.enable_checkpoint = False

    def force_close(self):
        def kill_process(proc_pid: int):
            # kill and terminate don't kill descendants, which is some
            # kind of design decision I guess 🙄
            # lifted this code from https://github.com/graphcore/examples-utils/blob/09c7e460946020658baf4b87aad3e439044010a2/examples_utils/benchmarks/run_benchmarks.py#L127C9-L127C21
            process = psutil.Process(proc_pid)
            for proc in process.children(recursive=True):
                logger.info("Killing child process %s", proc.pid)
                proc.kill()
            logger.info("Killing process %s", proc_pid)
            process.kill()

        # First empty the send queue and send terminate again and try to join
        try:
            while True:
                self.mp_queue_send.get_nowait()
        except queue.Empty:
            pass
        self.mp_queue_send.put((EMACommand.TERMINATE,))
        self.mp.join(timeout=self.mp_timeout)
        pid = self.mp.pid
        assert pid, "How is the process reporting alive and without a PID - we're out of ideas - panic!"
        if not self.mp.is_alive():
            return
        # Then if this fails
        logger.error("Force killing EMACheckpoint process %s", pid)
        kill_process(pid)


class CheckpointManagerWithEMAWeight(TTCheckpointManager):
    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
        ft_manager: FTManager | None = None,
    ):
        assert isinstance(checkpoint_config, CheckpointConfig)
        super().__init__(
            dataloader,
            model_parts,
            optimizers,
            lr_schedulers,
            states,
            checkpoint_config,
            sd_adapter,
            base_folder,
            ft_manager,
        )
        if self.ft_manager:
            raise NotImplementedError(
                "CheckpointManagerWithEMAWeight does not support FTManager yet."
            )
        self.ema_checkpointer = EMACheckpointManager(
            model_parts=model_parts,
            checkpoint_config=checkpoint_config,
            base_folder=base_folder,
        )

    def maybe_wait_for_staging(self):
        super().maybe_wait_for_staging()
        # This is... weird. It's relying on implicit knowledge of where the checkpointer is called
        self.ema_checkpointer.update_ema()

    def close(self) -> None:
        """Close the checkpoint manager and EMA checkpointer."""
        super().close()
        if hasattr(self, "ema_checkpointer") and self.ema_checkpointer:
            self.ema_checkpointer.close()

    def save(self, curr_step: int, last_step: bool = False) -> None:
        """Save the current state, including EMA weights if enabled.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.
        """
        super().save(curr_step, last_step)
        self.ema_checkpointer.save(curr_step, last_step)

    def _find_load_steps(self, folder: str = "") -> List[int]:
        """Find the step to load the checkpoint for.

        Note: Implementation copied from CheckpointManager._find_load_step removing the `max` in the return.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            List[int]: The step to load the checkpoint for.

        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return [-1]

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            metadata_probe = os.path.join(folder, filename, ".metadata")
            if match and os.path.isfile(metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return [-1]
        return step_counts

    def load(self, step: int = -1) -> bool:
        """"""
        # Find the highest common step to load from both normal and EMA checkpoints.
        if step == -1:
            normal_steps = self._find_load_steps()
            ema_steps = set(self._find_load_steps(self.ema_checkpointer.folder))
            # find highest common step to load
            for normal_step in sorted(normal_steps, reverse=True):
                if normal_step in ema_steps:
                    step = normal_step
                    logger.info(
                        f"Loading checkpoint at step {step} from both normal and EMA checkpoints."
                    )
                    break
            # if the largest common step is below the skip_first_k_updates, we won't load the EMA
            # and pick the largest step below the skip_first_k_updates
            if self.ema_checkpointer.skip_first_k_updates > max(ema_steps):
                step = max(
                    [
                        s
                        for s in normal_steps
                        if s <= self.ema_checkpointer.skip_first_k_updates
                    ]
                )
                logger.info(
                    f"Skipping EMA loading as the largest common step is below skip_first_k_updates. Chosen step is {step}"
                )
            if step == -1:
                logger.warning(
                    "EMA and normal checkpoints do not have common steps to load, skipping checkpoint loading."
                )
                return False

        load_outcome = super().load(step)
        # We only load the EMA weights if we have loaded the model weights
        if not load_outcome:
            logger.warning(
                f"Failed to load model checkpoint at step {step}, skipping EMA weights loading."
            )
            return False

        checkpoint_id = self._create_checkpoint_id(step, self.ema_checkpointer.folder)
        # For a manually specified step we might not have an ema checkpoint
        if os.path.isdir(checkpoint_id) and os.path.isfile(f"{checkpoint_id}/.metadata"):
            logger.info(f"Loading EMA weights from {checkpoint_id}")
            self.ema_checkpointer.load(checkpoint_id)
        else:
            logger.warning(
                f"Checkpoint directory {checkpoint_id} does not exist for EMA weights."
            )
        return load_outcome


class CheckpointManagerLoadOnly(TTCheckpointManager):
    """Load only checkpoint manager, never writing back any checkpoint.

    Useful for running TorchTitan training loop on evaluation-like type of tasks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self, step: int = -1) -> bool:
        r = super().load(step=step)
        assert r, f"Unsuccessful loading of model checkpoint at step {step}."
        return r

    def save(self, curr_step: int, force: bool = False) -> None:
        # No "warning" when we would not have saved anyway.
        if not self._should_save(curr_step, force):
            return
        logger.warning("Ignoring checkpoint saving in `CheckpointManagerLoadOnly`.")


class CheckpointManagerPatching(object):
    """Context manager for patching `CheckpointManager` TorchTitan class.

    Can be used to patch only a portion of code using:
    ```python
    with CheckpointManagerPatching(CheckpointManagerLoadOnly):
        pass
    ```
    """

    def __init__(self, ckpt_manager_cls):
        self.ckpt_manager_cls = ckpt_manager_cls
        self.orig_ckpt_manager_cls = torchtitan.components.checkpoint.CheckpointManager
        self.modules_to_patch = [
            torchtitan.components.checkpoint,
            torchtitan.train,
        ]

    def __enter__(self):
        for m in self.modules_to_patch:
            m.CheckpointManager = self.ckpt_manager_cls
        logger.info(
            f"Patching TorchTitan `CheckpointManager` with `{self.ckpt_manager_cls.__name__}` class."
        )

    def __exit__(self, type, value, traceback):
        # Reverting patching.
        for m in self.modules_to_patch:
            m.CheckpointManager = self.orig_ckpt_manager_cls
