# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license retained at
# torchtitan_submodule/LICENSE in this source distribution.
from typing import Callable
import contextlib
import os
import time
from pathlib import Path
import json
import warnings
import pandas as pd
import gzip
import torch
from torchtitan.tools.logging import logger
import torchtitan.tools.profiling
from .metrics import get_metrics_processor
import multiprocessing
import atexit
import queue

# Import to keep track of original method + trigger error in future TT upgrade
from torchtitan.tools.profiling import maybe_enable_profiling as tt_maybe_enable_profiling  # noqa:F401
from low_bits_training.config import Profiling as ProfilingConfig

# the number of warmup steps before the active step in each profiling cycle
WARMUP = 3

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


class HTAWorker:
    """
    Manages HTA (Holistic Trace Analysis) worker process for distributed training profiling.

    Why a separate process is essential:
    -----------------------------------
    1. **torch.distributed.destroy_process_group() Deadlock**:
       - Calling HTA analysis in the same process/thread as PyTorch distributed can lock up
         the destroy_process_group() call indefinitely

    2. **Resource Lock Issues**:
       - HTA spawns its own child processes for analysis
       - These conflict with PyTorch's distributed process management
       - Signal handlers clash

    3. **Initialization Timing**:
       - Worker MUST be spawned before torch.distributed.init_process_group()
       - Late initialization after distributed setup still causes conflicts

    Process Separation Solution:
    ---------------------------
    This class creates a completely separate worker process that:
    - Runs independently from PyTorch distributed processes
    - Receives analysis tasks via multiprocessing queues
    - Performs HTA analysis in isolation without resource conflicts
    - Returns results safely without blocking distributed cleanup

    Distributed Training Flow:
    -------------------------
    1. Rank 0 submits HTA analysis task to worker process
    2. Worker process performs HTA analysis independently
    3. All ranks wait at barrier for rank 0 to complete
    4. Distributed cleanup proceeds without deadlocks

    Example usage:
        # In main(), before torch.distributed.init_process_group()
        hta_worker = HTAWorker()
        hta_worker.initialize()

        # During training (rank 0 only)
        success = hta_worker.process_summary_metrics(trace_dir, step, metrics_processor)

        # Automatic cleanup on exit
    """

    def __init__(self):
        self.worker_process = None
        self.task_queue = None
        self.result_queue = None
        self.is_initialized = False

    def initialize(self):
        """Initialize the HTA worker process if not already initialized."""
        if self.is_initialized:
            logger.debug("HTA worker already initialized")
            return

        try:
            # Create queues for communication
            ctx = multiprocessing.get_context("spawn")
            self.task_queue = ctx.Queue()
            self.result_queue = ctx.Queue()

            # Create the worker process (non-daemon)
            self.worker_process = ctx.Process(
                target=self._worker_function,
                args=(self.task_queue, self.result_queue),
                daemon=False,  # Critical: non-daemon so HTA can spawn children
            )
            self.worker_process.start()
            self.is_initialized = True

            logger.info("Initialized HTA worker process")

        except Exception as e:
            logger.error(f"Failed to initialize HTA worker: {e}")
            self._cleanup_resources()
            raise

    def is_alive(self):
        """Check if worker is initialized and alive."""
        return (
            self.is_initialized
            and self.worker_process is not None
            and self.worker_process.is_alive()
        )

    def process_summary_metrics(self, trace_dir: Path, step: int, metrics_processor=None):
        """
        Process summary metrics using the HTA worker process.

        Args:
            trace_dir: Path to the directory containing trace files
            step: Training step number for these metrics
            metrics_processor: Optional MetricsProcessor

        Returns:
            bool: True if processing was successful, False otherwise
        """
        if not self.is_alive():
            logger.error(
                "HTA worker not initialized or died. Call initialize_hta_worker() before training."
            )
            return False

        try:
            # Clear any stale results from the queue first
            self._clear_stale_results()

            # Send task to worker
            task = ("hta_analysis", (str(trace_dir), step))
            self.task_queue.put(task)

            logger.info(f"Submitted HTA analysis for step {step} to worker process")

            # Wait for result with timeout
            try:
                result_type, result_data = self.result_queue.get(timeout=30)
                if result_type == "success":
                    if result_data and metrics_processor:
                        metrics_processor.delayed_log(result_data)
                        logger.info(f"HTA analysis completed for step {step}")
                    return True
                else:  # error
                    logger.error(f"HTA analysis error for step {step}: {result_data}")
                    return False

            except queue.Empty:
                logger.warning(f"HTA analysis for step {step} timed out")
                return False

        except Exception as e:
            logger.error(f"Failed to submit HTA analysis: {e}")
            return False

    def cleanup(self):
        """Clean up the HTA worker process."""
        if not self.is_initialized:
            return

        try:
            if self.worker_process and self.worker_process.is_alive():
                # Signal worker to terminate
                self.task_queue.put(None)

                # Wait for worker to finish
                self.worker_process.join(timeout=10)

                # Force terminate if still alive
                if self.worker_process.is_alive():
                    self.worker_process.terminate()
                    self.worker_process.join(timeout=5)

            logger.info("HTA worker process cleaned up")

        except Exception as e:
            logger.warning(f"Error during HTA worker cleanup: {e}")
        finally:
            self._cleanup_resources()

    def _cleanup_resources(self):
        """Internal method to clean up all resources."""
        self.worker_process = None
        if self.task_queue is not None:
            self.task_queue.close()
            self.task_queue = None
        if self.result_queue is not None:
            self.result_queue.close()
            self.result_queue = None
        self.is_initialized = False

    def _clear_stale_results(self):
        """Clear any stale results from the result queue."""
        while True:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def _worker_function(self, task_queue, result_queue):
        """
        Long-running worker process that waits for HTA analysis tasks.
        This runs in a separate process.
        """
        while True:
            try:
                # Wait for a task (blocking)
                task = task_queue.get()

                if task is None:  # terminate signal
                    break

                task_type, args = task

                if task_type == "hta_analysis":
                    trace_dir_str, step = args
                    try:
                        # Run the HTA analysis
                        success = _process_summary_metrics_worker(trace_dir_str, step)
                        result_queue.put(("success", success))
                    except Exception as e:
                        result_queue.put(("error", str(e)))

            except Exception as e:
                result_queue.put(("error", str(e)))


# Module-level worker instance
_hta_worker = HTAWorker()


def initialize_hta_worker():
    """Initialize the HTA worker."""
    _hta_worker.initialize()


def cleanup_hta_worker():
    """Cleanup the HTA worker."""
    _hta_worker.cleanup()


# Register cleanup function
atexit.register(cleanup_hta_worker)


def _add_flops_to_trace_file(trace_file: Path, prof: torch.profiler.profile):
    """
    Add flops estimates from the profile to the trace file

    Note: this operation requires reading the trace and rewriting it. It will be slow.
    """
    events = prof.events()
    if events is None:
        return []
    flops_json = []
    flops_dict = {}
    for event in events:
        if event.flops != 0:
            flops_dict[event.id] = (event.trace_name, event.flops)
            flops_json.append(
                {"External id": event.id, "flops": event.flops, "name": event.trace_name}
            )
    if trace_file:
        if trace_file.suffix == ".gz":
            with gzip.open(trace_file, "rt") as f:
                trace = json.load(f)
        else:
            trace = json.loads(trace_file.read_text())
        for event in trace["traceEvents"]:
            if args := event.get("args"):
                if name_flops := flops_dict.get(args.get("External id")):
                    assert name_flops[0] == event["name"]
                    args["flops"] = name_flops[1]
        if trace_file.suffix == ".gz":
            with gzip.open(trace_file, "wt") as f:
                json.dump(trace, f)
        else:
            trace_file.write_text(json.dumps(trace))
    return flops_json


def add_aggregated_stats_to_dataframe(df, rank_column="rank", agg_stats=[]):
    """
    Add mean, max, min rows to a dataframe with rank-based metrics
    """

    enhanced_df = df.copy()

    # Create the stat rows using AGG constant
    for stat in agg_stats:
        stat_row = df.select_dtypes(include=["number"]).agg(stat)
        # Convert to object dtype to allow mixed types
        stat_row = stat_row.astype("object")
        stat_row[rank_column] = stat
        enhanced_df = pd.concat([enhanced_df, stat_row.to_frame().T], ignore_index=True)

    return enhanced_df


def filter_dataframe_columns(df, column_names):
    """
    Filter dataframe to only include specified columns plus the rank column

    Args:
        df: Input dataframe
        column_names (list): List of column names to keep

    Returns:
        pandas.DataFrame: Filtered dataframe with only specified columns + rank column
    """
    # Always keep the rank column
    columns_to_keep = ["rank"]

    # Add requested columns that exist in the dataframe
    for col in column_names:
        if col in df.columns and col not in columns_to_keep:
            columns_to_keep.append(col)

    return df[columns_to_keep]


def flatten_enhanced_dataframe(df, category):
    """
    Flatten dataframe into dictionary with hierarchical keys

    Args:
        df: dataframe with rank column and aggregated stats
        category: category name (e.g. "profiler/gpu_compute_breakdown")

    Returns:
        dict: Flattened dictionary with hierarchical keys like "profiler/gpu_compute_breakdown/compute_time_pctg/rank0"
    """
    flattened = {}

    for index, row in df.iterrows():
        rank_value = row["rank"]

        for column in df.columns:
            if column != "rank":  # Skip the rank column itself
                # Create hierarchical key format: category/column/rank_or_stat
                if isinstance(rank_value, int):
                    key = f"{category}/{column}/rank{rank_value}"
                else:
                    key = f"{category}/{column}/{rank_value}"  # For mean, min, max

                flattened[key] = row[column]

    return flattened


# =============================================================================
# CONFIGURATION for HTA analysis, read _add_summary_metrics_from_trace
# =============================================================================

HTA_CONFIG = {
    # Each key corresponds to an HTA function name that will be called on the TraceAnalysis object
    # Structure: "hta_function_name": {"category": "W&B namespace", "columns": ["filtered_columns"]}
    # See https://hta.readthedocs.io/en/latest/ <FEATURES> for further HTA documentation and extend this further
    "get_temporal_breakdown": {  # Calls hta_analyzer.get_temporal_breakdown()
        "category": "profiler/gpu_compute_breakdown",  # Namespace for organizing these metrics
        "columns": [
            # Filter to only these columns from the HTA function's DataFrame output
            "compute_time_pctg",
            "non_compute_time_pctg",
            "idle_time_pctg",
            # Add more columns to be logged as needed:
            # "idle_time(us)",
            # "compute_time(us)",
            # "kernel_time(us)"
        ],
    },
    "get_comm_comp_overlap": {  # Calls hta_analyzer.get_comm_comp_overlap()
        "category": "profiler/overlap",  # Namespace for organizing these metrics
        "columns": [
            # Filter to only these columns from the HTA function's DataFrame output
            "comp_comm_overlap_pctg",
        ],
    },
}

# Aggregation statistics to compute for each metric
AGG = ["mean", "min", "max"]

# =============================================================================


def _add_summary_metrics_from_trace(trace_dir: Path, step: int):
    """
    Calculate performance metrics from profile trace using HTA API
    Perform communication overlap analysis if multiple GPUs are used,

    Args:
        trace_file (Path): Path to the trace files
        step (int): The training step number for which the trace was generated
    Returns:
        dict: A dictionary containing performance metrics
    """

    """
    =============================================================================
    PROCESSING PIPELINE OVERVIEW:

    This function processes HTA (Holistic Trace Analysis) output through a 4-step pipeline:
    1. Extract raw dataframes from HTA using functions specified in HTA_CONFIG
    2. Filter dataframes to include only columns specified in HTA_CONFIG
    3. Enhance filtered dataframes by adding statistics across all ranks from AGG above (e.g mean, max, min)
    4. Flatten enhanced dataframes into a structured dictionary with hierarchical keys

    EXAMPLE DATA TRANSFORMATION:

    Step 1 & 2: HTA Raw Data → Filtered Columns
    ===== GPU compute breakdown (after filtering) =====
    rank  compute_time_pctg  non_compute_time_pctg  idle_time_pctg
    0     75.22              23.81                  0.97
    1     76.04              22.97                  0.99
    2     74.88              24.15                  0.96

    Step 3: Add Aggregated Statistics
    rank  compute_time_pctg  non_compute_time_pctg  idle_time_pctg
    0     75.22              23.81                  0.97
    1     76.04              22.97                  0.99
    2     74.88              24.15                  0.96
    mean  75.38              23.64                  0.97    ← Added
    min   74.88              22.97                  0.96    ← Added
    max   76.04              24.15                  0.99    ← Added

    ===== Computation-Communication Overlap (after filtering) =====
    rank  comp_comm_overlap_pctg
    0     59.86
    1     60.78
    2     58.94
    mean  59.86              ← Added
    min   58.94              ← Added
    max   60.78              ← Added

    Step 4: Flatten to Dictionary Format
    Final output structure with hierarchical keys:
    {
      "profiler/step": 10,
      "profiler/gpu_compute_breakdown/compute_time_pctg/rank0": 75.22,
      "profiler/gpu_compute_breakdown/compute_time_pctg/rank1": 76.04,
      "profiler/gpu_compute_breakdown/compute_time_pctg/rank2": 74.88,
      "profiler/gpu_compute_breakdown/compute_time_pctg/mean": 75.38,
      "profiler/gpu_compute_breakdown/compute_time_pctg/min": 74.88,
      "profiler/gpu_compute_breakdown/compute_time_pctg/max": 76.04,
      "profiler/gpu_compute_breakdown/non_compute_time_pctg/rank0": 23.81,
      "profiler/gpu_compute_breakdown/non_compute_time_pctg/mean": 23.64,
      "profiler/gpu_compute_breakdown/idle_time_pctg/rank0": 0.97,
      "profiler/overlap/comp_comm_overlap_pctg/rank0": 59.86,
      "profiler/overlap/comp_comm_overlap_pctg/mean": 59.86,
      ... (and so on for all configured columns and ranks)
    }

    =============================================================================
    """

    # Get the directory containing the trace file
    # trace_dir = trace_file.parent

    try:
        # Check if we're using multiple GPUs
        trace_files = list(Path(trace_dir).glob("rank*.*.pt.trace.json*"))
        world_size = len(trace_files)
        is_multi_gpu = world_size > 1
        metrics = {}
        metrics["profiler/step"] = step

        try:
            # Create TraceAnalysis object
            from hta.trace_analysis import TraceAnalysis

            hta_analyzer = TraceAnalysis(trace_dir=str(trace_dir))

            temporal_config = HTA_CONFIG["get_temporal_breakdown"]
            overlap_config = HTA_CONFIG["get_comm_comp_overlap"]

            # Collect GPU compute breakdown for all ranks
            temporal_df = hta_analyzer.get_temporal_breakdown(visualize=False)
            temporal_df_filt = filter_dataframe_columns(
                temporal_df, temporal_config["columns"]
            )
            temporal_df_en = add_aggregated_stats_to_dataframe(
                temporal_df_filt, agg_stats=AGG
            )
            # logger.info(f"\n===== GPU compute Breakdown =====\n{temporal_df_en.to_string()}\n")
            temporal_df_en_flat = flatten_enhanced_dataframe(
                temporal_df_en, temporal_config["category"]
            )

            metrics.update(temporal_df_en_flat)

            # Do communication-computation overlap analysis in multi-GPU mode for all ranks
            if is_multi_gpu:
                logger.info(
                    f"Multi-GPU training detected (world_size={world_size}). Analyzing communication-computation overlap."
                )
                overlap_df = hta_analyzer.get_comm_comp_overlap(visualize=False)
                overlap_df_filt = filter_dataframe_columns(
                    overlap_df, overlap_config["columns"]
                )
                overlap_df_en = add_aggregated_stats_to_dataframe(
                    overlap_df_filt, agg_stats=AGG
                )
                # logger.info(f"\n===== Computation-Communication Overlap =====\n{overlap_df_en.to_string()}\n")
                overlap_df_en_flat = flatten_enhanced_dataframe(
                    overlap_df_en, overlap_config["category"]
                )
                metrics.update(overlap_df_en_flat)
            else:
                logger.info(
                    "Single-GPU training detected. Skipping communication-computation overlap analysis."
                )
                metrics[overlap_config["category"]] = (
                    "Single GPU training. Communication metrics not applicable."
                )

        except Exception as e:
            logger.error(f"Error in HTA analysis: {str(e)}")
            metrics["hta_error"] = str(e)

        # logger.info(f"Raw : {json.dumps(metrics, indent=2, default=str)}")
        return metrics

    except Exception as e:
        # If anything goes wrong, log the error and return an empty dict
        logger.error(f"Error calculating metrics: {str(e)}")
        return {"error": str(e)}


def _process_summary_metrics_worker(trace_dir_str: str, step: int):
    """
    Worker function that runs in a separate process.
    calculate metrics, save to file
    """
    trace_dir = Path(trace_dir_str)

    # 1. Calculate summary metrics
    summary_metrics = _add_summary_metrics_from_trace(trace_dir, step)

    # 2. Save to file
    summary_file = trace_dir / "summary_metrics.json"
    summary_file.write_text(json.dumps(summary_metrics, indent=2, default=str))

    if summary_metrics:
        return summary_metrics
    else:
        return None


def process_summary_metrics(trace_dir: Path, step: int, metrics_processor=None):
    """
    Process summary metrics using the long-running HTA worker process

    Args:
        trace_dir: Path to the directory containing trace files
        step: Training step number for these metrics
        metrics_processor: Optional W&B logger for logging metrics

    Returns:
        bool: True if processing was successful, False otherwise
    """
    if not _hta_worker.is_alive():
        logger.error(
            "HTA worker process not initialized or died. Call initialize_hta_worker() before training."
        )
        return False

    return _hta_worker.process_summary_metrics(trace_dir, step, metrics_processor)


def get_trace_handler(
    trace_dir,
    mem_timeline,
    rank,
    with_flops,
    with_summary_metrics,
    device,
    metrics_processor=None,
) -> Callable[[torch.profiler.profile], None]:
    """Get the trace handler with the correct configuration to pass to the
    `on_trace_ready` argument of the Pytorch profiler."""

    # We use this getter to make the trace_handler easier to test.
    # In torchtitan it is simply nested
    def trace_handler(prof: torch.profiler.profile):
        curr_trace_dir_name = "iteration_" + str(prof.step_num)
        curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
        memory_dir = os.path.join(curr_trace_dir, "memory")

        if not os.path.exists(curr_trace_dir):
            os.makedirs(curr_trace_dir, exist_ok=True)

        # Create memory subfolder for memory profiles
        if mem_timeline:
            if not os.path.exists(memory_dir):
                os.makedirs(memory_dir, exist_ok=True)

        logger.info(f"Dumping traces at step {prof.step_num}")
        begin = time.monotonic()
        if mem_timeline:
            if "html" in str(mem_timeline):
                warnings.warn(
                    "Logging memory timeline with HTML is slow. Prefer 'json' and 'raw.json.gz' formats"
                )
            prof.export_memory_timeline(
                f"{memory_dir}/rank{rank}_memory_timeline.{mem_timeline}",
                device=device,
            )
        # This simply dumps the chrome trace with a name that means tensorboard can open it
        # Tensorboard could open it anyway if you renamed - and the files generated in this
        # way can still be opened with https://ui.perfetto.dev/
        torch.profiler.tensorboard_trace_handler(
            curr_trace_dir, worker_name=f"rank{rank}", use_gzip=True
        )(prof)
        # output flops in a way that they can be cross-referenced with the trace.
        if with_flops:
            trace_file = list(Path(curr_trace_dir).glob(f"rank{rank}.*.pt.trace.json*"))[
                0
            ]
            flops_json = _add_flops_to_trace_file(trace_file, prof)
            (Path(curr_trace_dir) / f"rank{rank}_flops.json").write_text(
                json.dumps(flops_json, indent=1)
            )

        logger.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")
        # Profiling is a heavy operation which could cost very different amount of time
        # across all ranks. Insert a barrier to make sure all ranks have finished profiling
        # before moving on.
        # TODO: Can we find a cleaner way?
        torch.distributed.barrier()

        # Calculate and save summary metrics if requested in rank0 when all ranks have finished traces
        if with_summary_metrics and rank == 0:
            trace_file = list(Path(curr_trace_dir).glob(f"rank{rank}.*.pt.trace.json*"))[
                0
            ]
            process_summary_metrics(trace_file.parent, prof.step_num, metrics_processor)

        # All ranks wait till process_summary_metrics is done
        if with_summary_metrics:
            torch.distributed.barrier()

    return trace_handler


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    # get user defined profiler settings
    enable_profiling = profiling_config.enable_profiling

    if enable_profiling:
        dump_dir = base_folder
        save_trace_dir = profiling_config.save_traces_folder
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        profile_freq = profiling_config.profile_freq
        active = profiling_config.consecutive_active_steps
        mem_timeline = profiling_config.memory_timeline
        with_flops = profiling_config.with_flops_table
        with_summary_metrics = profiling_config.with_summary_metrics
        turn_on_mem_timeline = bool(mem_timeline)
        rank = torch.distributed.get_rank()
        device = f"cuda:{os.getenv('LOCAL_RANK')}"
        other_kwargs = {}
        if profiling_config.experimental_cupti_stats:
            # TODO: this does not work, when activated no GPU activity is recorded.
            # This feature is experimental (broken)
            # https://hta.readthedocs.io/en/latest/source/features/cupti_counter_analysis.html#cupti-counter-analysis
            # See this issue for status: https://github.com/pytorch/pytorch/issues/125272
            # assert os.geteuid() == 0, "experimental_cupti_stats require sudo access"
            # other_kwargs = dict(
            #     experimental_config=torch.profiler._ExperimentalConfig(
            #     profiler_metrics=[
            #         "kineto__tensor_core_insts",
            #         "kineto__cuda_core_flops",
            #         "dram__bytes_read.sum",
            #         "dram__bytes_write.sum"],
            #     profiler_measure_per_kernel=True
            #     ),
            # )
            pass

        # Verify that HTA worker is initialized if summary metrics are requested
        if with_summary_metrics and not _hta_worker.is_alive():
            logger.warning(
                "HTA worker not initialized but summary metrics requested. "
                "Make sure to call initialize_hta_worker() in main() before distributed setup."
            )

        metrics_processor = get_metrics_processor()
        trace_handler = get_trace_handler(
            trace_dir,
            mem_timeline,
            rank,
            with_flops,
            with_summary_metrics,
            device,
            metrics_processor,
        )
        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")
        logger.info(f"Number of active steps: {active}")
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = WARMUP, active
        wait = profile_freq - (active + warmup)
        assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=turn_on_mem_timeline,
            with_stack=turn_on_mem_timeline,
            with_flops=with_flops,  # Not in chrome traces, only on the python object
            **other_kwargs,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


# override
torchtitan.tools.profiling.maybe_enable_profiling = maybe_enable_profiling
