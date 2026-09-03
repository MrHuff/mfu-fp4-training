#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os
from typing import Any, Dict, List, overload, Literal
from pathlib import Path
import pkgutil
import itertools
import logging
import re
import sys
import subprocess
import threading
from queue import Queue, Empty
from deprecated import deprecated

import wandb


import torchtitan
from torchtitan.components.metrics import _get_metrics_rank
from torchtitan.distributed.parallel_dims import ParallelDims

from .config import JobConfig, ConfigManager


@deprecated("Use `JobConfig.to_dict` method directly.")
def job_config_to_config_dict(job_config: JobConfig) -> Dict[str, Dict[str, Any]]:
    """
    JobConfig is created from Argument Parser as a two level object.

    This a hacky method of converting JobConfig to a ConfigDict that can
    cleanly be consumed by wandb.
    """
    return job_config.to_dict()


def flatten_dict(d):
    flat = {}
    for k1, v1 in d.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                flat[f"{k1}.{k2}"] = v2
        else:
            flat[k1] = v1
    return flat


def find_config(run_dir: str | Path, rank=0):
    """Finds a config file in JSON or toml format in the base directory of a run"""
    run_dir = Path(run_dir)

    conf = list(run_dir.glob(f"config-rank{rank}-*.json"))
    if conf and len(conf) == 1:
        return conf[0]
    conf = list(run_dir.glob("config*.toml"))
    if conf and len(conf) == 1:
        return conf[0]
    raise ValueError(
        f"Did not find either JSON or TOML configs in {run_dir} for rank {rank}"
    )


def load_config(
    job_config: JobConfig | str | Path, allow_upgrade: bool = False, validate: bool = True
) -> JobConfig:
    """Loads a config from a file - or passes a config through if it is already a job_config"""
    if isinstance(job_config, (str, Path)):
        job_config_path = str(job_config)
        job_config = ConfigManager().parse_args(
            ["--job.config_file", job_config_path],
            allow_upgrade=allow_upgrade,
            validate=validate,
        )
    return job_config


def set_torch_compile_env_vars(dump_folder):
    os.makedirs(dump_folder, exist_ok=True)
    rank = os.getenv("RANK")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
        f"{dump_folder}/compile_details/inductor_cache_rank{rank}/"
    )
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"
    os.environ["TRITON_CACHE_DIR"] = (
        f"{dump_folder}/compile_details/triton_cache_rank{rank}/"
    )
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = (
        f"{dump_folder}/compile_details/torch_compile_debug_rank{rank}/"
    )


def get_parallel_dims(job_config: JobConfig) -> ParallelDims:
    world_size = int(os.environ["WORLD_SIZE"])
    parallelism_config = job_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    return parallel_dims


def is_metrics_logging_enabled(job_config: JobConfig) -> bool:
    """Is (W&B) metrics logging enabled on this process."""
    # Is metrics logging enabled?
    if not job_config.metrics.enable_tensorboard:
        return False

    if job_config.metrics.distributed_mode == "all":
        # All processes logging metrics!
        return True
    elif job_config.metrics.distributed_mode == "rank_0":
        # Only rank 0 (or last pipeline stage rank).
        metrics_rank = _get_metrics_rank(get_parallel_dims(job_config), job_config)
        return int(os.environ["RANK"]) == metrics_rank
    elif job_config.metrics.distributed_mode == "local_rank_0":
        # Local rank 0 on every node (or adapted to pipeline parallelism)
        metrics_rank = _get_metrics_rank(get_parallel_dims(job_config), job_config)
        # Convert the global rank to a local rank on every node.
        local_metrics_rank = metrics_rank % int(int(os.environ["LOCAL_WORLD_SIZE"]))
        return int(os.environ["LOCAL_RANK"]) == local_metrics_rank
    raise AttributeError(
        f"Unknown job config `metrics.distributed_mode` value: '{job_config.metrics.distributed_mode}'."
    )


def wandb_logging_mode(job_config: JobConfig) -> str:
    """Returns the W&B mode for the current process.

    Combining `WANDB_MODE` env. variable as well as flags in `job_config`
    to decide whether the process should `online` or `disabled`.
    """
    # By default, online on processes where logging is activated.
    wb_mode = job_config.wandb.mode
    assert isinstance(wb_mode, str)
    # Only forward the default mode on processes logging is enabled.
    return wb_mode if is_metrics_logging_enabled(job_config) else "disabled"


def wandb_group(job_config: JobConfig) -> str:
    """Get the W&B job config group, or generate a default one if not set."""
    return job_config.wandb.group or job_config.wandb.name


def wandb_init(job_config: JobConfig, project: str, entity: str):
    """Initialize properly W&B for current process and training job."""
    config_dict = job_config.to_dict()

    rank = os.environ["RANK"]
    wb_training_run_id = job_config.wandb.name
    wb_group_id = wandb_group(job_config)
    # Standardize naming of the run, based on training name and rank.
    wb_run_name = f"{wb_training_run_id}:rank_{rank}"
    # Additional metadata to record on W&B.
    config_dict["metadata"] = {
        "training_run_id": wb_training_run_id,
        "rank": int(rank),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
    }
    # W&B logging mode for this process.
    wb_logging_mode = wandb_logging_mode(job_config)
    return wandb.init(
        project=project,
        entity=entity,
        mode=wb_logging_mode,
        config=config_dict,
        name=wb_run_name,
        group=wb_group_id,
        id=job_config.wandb.id if job_config.wandb.id else None,
        job_type="training",
    )


def wandb_eval_init(
    job_config: JobConfig,
    eval_name: str,
    entity: str = "graphcore",
    delete_previous=False,
):
    """Initialize W&B for evaluation runs."""
    config_dict = job_config_to_config_dict(job_config)

    wb_training_run_id = job_config.wandb.name
    wb_group_id = wandb_group(job_config)
    # Standardize naming of the run, based on training name and rank.
    wb_run_name = f"{wb_training_run_id}:evaluation-{eval_name}"
    # Additional metadata to record on W&B.
    config_dict["metadata"] = {
        "training_run_id": wb_training_run_id,
    }
    project = job_config.wandb.project
    if delete_previous:
        try:
            previous_run_ids = wandb_id_from_project_and_name(
                project=project, name=wb_run_name, entity=entity, allow_multiple=True
            )
        except ValueError as e:
            if "could not be found in project" not in str(e):
                raise
            previous_run_ids = []
            pass
        api = wandb.Api()
        # Delete each run
        for run_id in previous_run_ids:
            run = api.run(f"{entity}/{project}/{run_id}")
            run.delete()
            print(f"Deleted earlier run of {wb_run_name}: {run_id}")
    return wandb.init(
        project=job_config.wandb.project,
        entity=entity,
        config=config_dict,
        name=wb_run_name,
        group=wb_group_id,
        job_type="evaluation",
    )


@overload
def wandb_id_from_project_and_name(
    project: str,
    name: str,
    entity: str = "graphcore",
    allow_multiple: Literal[False] = False,
) -> str: ...


@overload
def wandb_id_from_project_and_name(
    project: str,
    name: str,
    entity: str = "graphcore",
    allow_multiple: Literal[True] = ...,
) -> list[str]: ...


def wandb_id_from_project_and_name(
    project: str, name: str, entity: str = "graphcore", allow_multiple=False
) -> str | list[str]:
    """
    Get the W&B ID for a given project and name.
    """
    runs = wandb.Api().runs(f"{entity}/{project}")
    run_ids = []
    for r in runs:
        if r.name == name:
            run_ids.append(r.id)

    if not run_ids:
        raise ValueError(f"W&B run {name} could not be found in project {project}")
    if allow_multiple:
        return run_ids
    if len(run_ids) > 1:
        raise ValueError(
            f"More than one run with name {name} was found - please specify ID explicitely - IDs that match: {run_ids}"
        )
    run_id = run_ids[0]
    return run_id


def wandb_id_from_config(
    job_config: JobConfig | str, name_suffix: str = ":rank_0"
) -> str:
    """
    Get the W&B ID for a given job config.
    """
    job_config = load_config(job_config)
    if wandb_id := getattr(job_config.wandb, "id", None):
        return wandb_id
    assert job_config.wandb.name, "W&B name is required to get the W&B ID"
    name = f"{job_config.wandb.name}{name_suffix}"
    return wandb_id_from_project_and_name(
        job_config.wandb.project,
        name,
        getattr(job_config.wandb, "entity", "graphcore"),
        allow_multiple=False,
    )


def find_torchtitan_modules_with_imported_class(cls: Any) -> List[Any]:
    """Find all TorchTitan sub-modules with an imported class."""
    # Excluding TorchTitan experiments (not `core` of TorchTitan)
    excludes = ["experiments", "__pycache__"]
    tt_path = torchtitan.__path__[0]
    tt_name = torchtitan.__name__
    # TorchTitan subfolders to scan.
    # Necessary, as `pkgutil.walk_packages` does not scan subfolders without `__init__.py`
    torchtitan_subfolders = [
        v
        for v in os.listdir(tt_path)
        if v not in excludes and os.path.isdir(os.path.join(tt_path, v))
    ]
    # Recursive scan of TorchTitan (sub)modules.
    tt_module_iters = [pkgutil.walk_packages([tt_path], f"{tt_name}.")]
    tt_module_iters += [
        pkgutil.walk_packages([os.path.join(tt_path, m)], f"{tt_name}.{m}.")
        for m in torchtitan_subfolders
    ]
    tt_module_names = sorted(
        set([info.name for info in itertools.chain(*tt_module_iters)])
    )

    # Making sure excludes are not here.
    tt_module_excludes = [f"torchtitan.{v}" for v in excludes]
    tt_module_names = [
        m
        for m in tt_module_names
        if not any([m.startswith(v) for v in tt_module_excludes])
    ]

    # All TorchTitan modules/sub-modules
    tt_all_modules = [torchtitan] + [
        sys.modules[m] for m in tt_module_names if m in sys.modules
    ]

    # Filter modules (making sure it is the exact same object).
    modules = [m for m in tt_all_modules if getattr(m, cls.__name__, None) is cls]
    return modules


def run_process_with_realtime_output(process, buffer_size=8192):
    """
    Runs a subprocess and displays its output in real-time, handling progress bars correctly.
    Fixed version that properly handles bytes vs strings and thread management.

    Args:
        process: A subprocess.Popen object with stdout and stderr set to subprocess.PIPE
        buffer_size: Size of read buffer in bytes (default: 8192)

    Returns:
        The process return code

    Raises:
        RuntimeError: If output threads fail to terminate cleanly
    """

    def read_output(file_object, output_queue, is_stderr=False):
        """Read from file object and put data in queue"""
        try:
            file_descriptor = file_object.fileno()

            while True:
                try:
                    # Use os.read to get bytes directly
                    data = os.read(file_descriptor, buffer_size)
                    if not data:
                        break
                    output_queue.put((data, is_stderr))
                except (OSError, IOError, ValueError):
                    # Handle closed pipes or other IO errors
                    break
        except Exception:
            # Handle any other errors gracefully
            pass
        finally:
            # Signal that this thread is done
            output_queue.put((None, is_stderr))

    def write_output(data, file):
        try:
            print(data.decode(), file=file, sep="", end="")
        except UnicodeDecodeError:
            os.write(file.fileno(), data)

    try:
        output_queue = Queue()

        # Create threads to handle stdout and stderr
        stdout_thread = threading.Thread(
            target=read_output, args=(process.stdout, output_queue, False)
        )
        stderr_thread = threading.Thread(
            target=read_output, args=(process.stderr, output_queue, True)
        )

        # Start threads
        stdout_thread.start()
        stderr_thread.start()

        # Process output as it comes in
        threads_finished = 0
        while threads_finished < 2:
            try:
                data, is_stderr = output_queue.get(timeout=0.1)

                if data is None:
                    # Thread finished
                    threads_finished += 1
                    continue

                # Write bytes directly to the appropriate stream
                if is_stderr:
                    write_output(data, sys.stderr)
                else:
                    write_output(data, sys.stdout)

            except Empty:
                # No data available, check if process is still running
                if process.poll() is not None:
                    # Process finished, but give threads a bit more time
                    # to process any remaining output
                    continue

        # Wait for process to complete
        return_code = process.wait()

        # Give threads reasonable time to finish, but don't wait forever
        stdout_thread.join(timeout=3.0)
        stderr_thread.join(timeout=3.0)

        # Check if threads are still alive and force cleanup if needed
        alive_threads = []
        if stdout_thread.is_alive():
            alive_threads.append("stdout")
        if stderr_thread.is_alive():
            alive_threads.append("stderr")

        if alive_threads:
            print(
                f"Warning: {', '.join(alive_threads)} thread(s) did not finish cleanly, forcing cleanup",
                file=sys.stderr,
            )

            # Close the file descriptors to unblock any pending reads
            try:
                process.stdout.close()
                process.stderr.close()
            except (OSError, IOError, ValueError):
                pass

            # Give threads one more chance to exit after closing descriptors
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

            # Check again and raise error if threads are STILL alive
            still_alive = []
            if stdout_thread.is_alive():
                still_alive.append("stdout")
            if stderr_thread.is_alive():
                still_alive.append("stderr")

            if still_alive:
                raise RuntimeError(
                    f"Failed to cleanly terminate output threads: {', '.join(still_alive)} "
                    f"thread(s) are still running after cleanup attempts. This indicates a "
                    f"serious threading issue that could cause resource leaks."
                )

        # Check return code and print warning if non-zero
        if return_code != 0:
            print(
                f"Warning: process exited with non-zero return code: {return_code}",
                file=sys.stderr,
            )

        return return_code

    finally:
        # Ensure process cleanup
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Close file handles to prevent resource leaks
        try:
            process.stdout.close()
            process.stderr.close()
        except (OSError, IOError, ValueError):
            pass


class NanLossDetectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self._nan_seen = False
        self._nan_loss_pattern = re.compile("loss:\\s*nan")

    def emit(self, record):
        if self._nan_loss_pattern.search(record.getMessage()) is not None:
            self._nan_seen = True

    def nan_seen(self):
        return self._nan_seen
