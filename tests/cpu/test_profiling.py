# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
import torch
from torch.profiler import profile
import torch.profiler
from pathlib import Path
import json
import gzip
import subprocess
import os
from low_bits_training.profiling import (
    get_trace_handler,
    process_summary_metrics,
    initialize_hta_worker,
    cleanup_hta_worker,
)

import pytest
import shutil


TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TEST_ROOT.parent

TEST_ASSET_DIRECTORY = TEST_ROOT / "assets/"


def generate_test_trace(
    base_config_path="train_configs/llama3_8b_profiling.toml",
    folder_name="test-trace",
    other_args="",
    clear_old=False,
):
    if clear_old and (TEST_ASSET_DIRECTORY / folder_name).exists():
        shutil.rmtree(TEST_ASSET_DIRECTORY / folder_name)
    expected_asset = TEST_ASSET_DIRECTORY / folder_name / "profile_trace/iteration_10/"
    expected_file_glob = "rank*.pt.trace.json*"
    expected_number_of_trace_files = 2

    def check_trace_files():
        trace_files = list(expected_asset.glob(expected_file_glob))
        return len(trace_files) == expected_number_of_trace_files

    if expected_asset.exists() and expected_asset.is_dir() and check_trace_files():
        return expected_asset
    if not torch.cuda.is_available():
        pytest.skip(
            "No GPU available - a GPU is required to generate the seed checkpoint - this file is expected to be commited"
        )
    env = os.environ.copy()
    env.update(
        {
            "NGPU": "2",
            "CONFIG_FILE": str(base_config_path),
            "WANDB_MODE": "disabled",
            "WANDB_NAME": str(folder_name),
        }
    )
    command = f"./run_train.sh --job.dump_folder {TEST_ASSET_DIRECTORY.relative_to(TEST_ROOT.parent)} {other_args}"
    print(f"Running command: {command}")
    test_asset_out = subprocess.check_output(
        args=command.strip().split(), env=env, text=True, cwd=TEST_ROOT.parent
    )
    print(test_asset_out)
    assert expected_asset.exists() and expected_asset.is_dir() and check_trace_files()
    return expected_asset


@pytest.fixture(scope="session", autouse=True)  # session scope to avoid race conditions
def unit_test_trace() -> Path:
    """Creates a 'unit_test' model in the test assets folder

    Requires a GPU to regenerate.
    """
    trace_path = generate_test_trace(
        base_config_path="train_configs/llama3_8b_profiling.toml",
        folder_name="test-trace-unit_test",
        other_args="--model.flavor unit_test --training.seq_len 16 --model.tokenizer_path ./torchtitan_submodule/tests/assets/tokenizer",
        clear_old=False,
    )
    return trace_path


@pytest.fixture()
def initialise_torch_distributed(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_PORT", "8001")
    torch.distributed.init_process_group("gloo", world_size=1)
    yield
    torch.distributed.destroy_process_group()


def check_flops_in_trace(trace_file):
    """Check that the trace contains flops counts in the events"""
    with gzip.open(trace_file, "rt") as f:
        trace_contents = json.load(f)
    flops_count = 0
    for trace_event in trace_contents["traceEvents"]:
        flops_count += trace_event.get("args", {}).get("flops", 0)
    assert flops_count > 0, "no flops were found in the trace"


def get_expected_metrics_keys(trace_dir):
    """Generate expected metrics keys from HTA_CONFIG and AGG"""
    from low_bits_training.profiling import HTA_CONFIG, AGG

    # count number of ranks
    trace_files = list(Path(trace_dir).glob("rank*.*.pt.trace.json*"))
    num_ranks = len(trace_files)

    expected_keys = ["profiler/step"]
    # For each HTA function configuration
    for config in HTA_CONFIG.values():
        category = config["category"]
        columns = config["columns"]
        for column in columns:
            # Expect keys for rank0, rank1, rank2, etc.
            for rank in range(num_ranks):
                expected_keys.append(f"{category}/{column}/rank{rank}")
            # Expect aggregation keys with new hierarchical format
            for agg in AGG:
                expected_keys.append(f"{category}/{column}/{agg}")
    return expected_keys


def test_process_summary_metrics(initialise_torch_distributed, tmp_path, unit_test_trace):
    """
    Test process_summary_metrics() generates correct HTA analysis metrics.

    Automatically detects expected metrics by:
    - Reading HTA_CONFIG to determine which analysis functions and columns to expect
    - Validating hierarchical keys like "profiler/gpu_compute_breakdown/compute_time_pctg/rank0"
      plus aggregation stats (mean, min, max) are present in summary_metrics.json
    - Tests the HTA worker process signalling from process_summary_metrics
    """
    # Initialize the HTA worker process
    initialize_hta_worker()

    # use pregen traces for test, copy to tmp_path
    source_trace_dir = Path(unit_test_trace)
    trace_dir = tmp_path / "traces"
    shutil.copytree(source_trace_dir, trace_dir)

    summary_file = trace_dir / "summary_metrics.json"
    if summary_file.exists():
        summary_file.unlink()
    # Call the function under test with step number
    process_summary_metrics(trace_dir, step=10)
    # Verify the summary_metrics.json file was created
    assert summary_file.exists(), "summary_metrics.json should be created"
    # verify keys for summary metrics are generated correctly
    with open(summary_file, "r") as f:
        metrics = json.load(f)
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert len(metrics) > 0, "Metrics dictionary should not be empty"

    # Get expected keys from config and trace_dir
    expected_keys = get_expected_metrics_keys(trace_dir)

    # Check that we have the expected keys
    found_keys = [key for key in expected_keys if key in metrics]
    assert len(found_keys) == len(
        expected_keys
    ), f"Missing keys: {set(expected_keys) - set(metrics.keys())}"

    cleanup_hta_worker()


@pytest.mark.parametrize("mem_timeline", [None, "json", "raw.json.gz"])
@pytest.mark.parametrize("with_flops", [True, False])
def test_profile_handler(
    initialise_torch_distributed, tmp_path, mem_timeline, with_flops
):
    rank = 1
    with_mem = bool(mem_timeline)
    trace_handler = get_trace_handler(
        tmp_path, mem_timeline, rank, with_flops, False, "cpu"
    )
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_flops=with_flops,
        profile_memory=with_mem,
        with_stack=with_mem,
    ) as profile_with_flops:
        x = torch.randn([100, 100], device="cpu")
        y = torch.randn([100, 100], device="cpu")
        _ = x.mm(y)
        profile_with_flops.step()

    trace_dir = Path(tmp_path) / "iteration_1"
    assert (
        trace_dir.exists()
    ), f"The directory in which traces are expected did not exist: {trace_dir}"
    trace_files = list(trace_dir.glob(f"rank{rank}.*.pt.trace.json*"))
    trace_dir_and_contents = f"{trace_dir} with contents: {list(trace_dir.iterdir())}"
    assert len(trace_files) == 1, f"Trace file not found in {trace_dir_and_contents}"
    trace_file = trace_files[0]
    if mem_timeline:
        mem_dir = trace_dir / "memory"
        assert (
            len(list(mem_dir.glob(f"*memory*{mem_timeline}"))) == 1
        ), f"Memory files not found in {mem_dir}"
    if with_flops:
        check_flops_in_trace(trace_file)
