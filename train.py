#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

# Fix cypari signal handler conflict BEFORE any imports:
# cypari installs SIGSEGV/SIGINT handlers that conflict with NCCL.
import os
os.environ["PARI_NO_SIGNAL"] = "1"  # Hint to pari to not install signal handlers

import sys
import torch
import faulthandler

# First import low_bits_training for MonkeyPatching TorchTitan.
import low_bits_training  # noqa: F401

# After cypari is loaded (via streaming→snappy→SnapPy), neutralize its signal handling.
import signal
signal.signal(signal.SIGINT, signal.default_int_handler)
if os.environ.get("LBT_ENABLE_SIGUSR1_FAULTHANDLER", "1") == "1":
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    except Exception:
        pass
try:
    import cypari._pari as _cp
    # Monkey-patch sig_on/sig_off to prevent cypari from reinstalling handlers
    _cp.sig_on = lambda: None
    _cp.sig_off = lambda: None
    # Also try calling sig_off to clear any active signal state
    try:
        _cp.sig_off()
    except Exception:
        pass
except ImportError:
    pass
import low_bits_training.utils
import wandb
from low_bits_training.config import ConfigManager
from torchtitan.distributed import utils as tt_dist_utils
from torchtitan.tools.logging import init_logger, logger

from typing import Optional

import os
import subprocess
import low_bits_training.profiling as profiling_module
from low_bits_training.trainer import Trainer
import low_bits_training.quantization.mxfp_custom
import low_bits_training.quantization.mxfp
try:
    import low_bits_training.quantization.mxfp_custom_te_fp4
except (ImportError, AttributeError) as e:
    print(f"[-] mxfp_custom_te_fp4 not available: {e}")
try:
    import low_bits_training.quantization.pure_te_fp4
except (ImportError, AttributeError) as e:
    print(f"[-] pure_te_fp4 not available: {e}")
import low_bits_training.quantization.fp4_converter


def get_nvidia_driver_version():
    try:
        # Run nvidia-smi to print driver version
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "NVIDIA driver not found"


def _use_lbt_safe_fast_exit() -> bool:
    return os.environ.get("USE_LBT_SAFE_FAST_EXIT", "1") == "1"


def _finish_and_fast_exit(exit_code: int) -> None:
    try:
        profiling_module.cleanup_hta_worker()
    except Exception:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(exit_code)


def _should_skip_single_rank_timeout_sync() -> bool:
    if os.environ.get("LBT_FORCE_STEP1_PG_TIMEOUT_SYNC", "0") == "1":
        return False
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() == 1


def _patch_single_rank_timeout_sync() -> None:
    if getattr(tt_dist_utils.set_pg_timeouts, "_lbt_single_rank_skip_patch", False):
        return

    orig_set_pg_timeouts = tt_dist_utils.set_pg_timeouts

    def _wrapped_set_pg_timeouts(*, timeout, world_mesh):
        if _should_skip_single_rank_timeout_sync():
            logger.info("Skipping step-1 ProcessGroup timeout sync for single-rank run")
            return
        return orig_set_pg_timeouts(timeout=timeout, world_mesh=world_mesh)

    _wrapped_set_pg_timeouts._lbt_single_rank_skip_patch = True
    tt_dist_utils.set_pg_timeouts = _wrapped_set_pg_timeouts


def main():
    print("TRAINING go brrrrr!")
    init_logger()
    _patch_single_rank_timeout_sync()

    config = ConfigManager().parse_args()
    config.dump()

    # Initialize HTA process where metrics on profile traces are calculated using HolisticTraceAnalysis.
    # This needs to be done before any torch, distributed initialization to avoid resource, signal issues caused by HTA child processes.
    if config.profiling.enable_profiling and config.profiling.with_summary_metrics:
        logger.info("Initializing HTA process...")
        profiling_module.initialize_hta_worker()

    # W&B init for model metrics & checkpoint.
    run = low_bits_training.utils.wandb_init(
        job_config=config,
        project=config.wandb.project,
        entity="graphcore",
    )

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"NVIDIA driver version: {get_nvidia_driver_version()}")
    config.wandb.id = run.id
    config.dump()
    # Main TorchTitan training setup & loop
    trainer: Optional[Trainer] = None
    pending_error: Exception | None = None
    exit_code = 0
    try:
        trainer = Trainer(config)
        if config.checkpoint.create_seed_checkpoint:
            assert int(
                os.environ["WORLD_SIZE"]
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception as e:
        pending_error = e
        exit_code = 1
        # Error logging before process ends, to record in W&B.
        # Keeping formatting similar to `torchrun` error output.
        logger.error("--- Logging error --")
        logger.error(f"{type(e).__module__}.{type(e).__qualname__}: {e}", exc_info=True)
    finally:
        # Note keeping W&B init + finish in `main` for clean exception handling.
        if trainer:
            trainer.close()
        wandb.finish()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if _use_lbt_safe_fast_exit():
            _finish_and_fast_exit(exit_code)
    if pending_error is not None:
        raise pending_error


if __name__ == "__main__":
    main()
