#!/usr/bin/env python3
#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import List, Dict, Any
import json
import tempfile
import subprocess
import sys
import os
import pytest


@pytest.fixture
def echo_script(tmp_path) -> str:
    """
    Create a temporary shell script that echoes its arguments.
    Args:
        tmp_path: Pytest-provided temporary directory path
    Returns:
        Path to the temporary shell script
    """
    script_path = tmp_path / "echo_script.sh"
    script_content = """#!/bin/bash
echo "$@"
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return str(script_path)


@pytest.fixture
def env_echo_script(tmp_path) -> str:
    """
    Create a temporary shell script that echoes its arguments and environment variables.
    Args:
        tmp_path: Pytest-provided temporary directory path
    Returns:
        Path to the temporary shell script
    """
    script_path = tmp_path / "env_echo_script.sh"
    script_content = """#!/bin/bash
echo "ARGS: $@"
echo "ENV_TEST_GLOBAL_VAR: $TEST_GLOBAL_VAR"
echo "ENV_TEST_PARAM_VAR: $TEST_PARAM_VAR"
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return str(script_path)


def run_sweep_and_capture(
    config_dict: Dict[str, Any], print_only: bool = False
) -> List[str]:
    """Run the sweep script with given config and return output lines."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
        json.dump(config_dict, tf)
        config_path = tf.name
    try:
        cmd = [sys.executable, "scripts/titan_sweep.py", config_path]
        if print_only:
            cmd.append("--print")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().split("\n")
    finally:
        os.unlink(config_path)


def test_grid_sweep(echo_script: str) -> None:
    """Test grid method parameter sweep."""
    config = {
        "command": echo_script,
        "method": "grid",
        "parameters": {
            "wandb.project": {"values": ["my_project"]},
            "training.steps": {"values": [1000, 2000]},
            "training.batch_size": {"values": [16, 32, 64]},
        },
    }
    outputs = run_sweep_and_capture(config)
    expected_outputs = [
        "--wandb.project my_project --training.steps 1000 --training.batch_size 16",
        "--wandb.project my_project --training.steps 1000 --training.batch_size 32",
        "--wandb.project my_project --training.steps 1000 --training.batch_size 64",
        "--wandb.project my_project --training.steps 2000 --training.batch_size 16",
        "--wandb.project my_project --training.steps 2000 --training.batch_size 32",
        "--wandb.project my_project --training.steps 2000 --training.batch_size 64",
    ]
    assert sorted(outputs) == sorted(expected_outputs)


def test_zip_sweep(echo_script: str) -> None:
    """Test zip method parameter sweep with a single-value parameter."""
    config = {
        "command": echo_script,
        "method": "zip",
        "parameters": {
            "wandb.project": {"values": ["my_project"]},
            "model.type": {"values": ["a", "b"]},
            "learning.rate": {"values": [0.1, 0.2]},
        },
    }
    outputs = run_sweep_and_capture(config)
    expected_outputs = [
        "--wandb.project my_project --model.type a --learning.rate 0.1",
        "--wandb.project my_project --model.type b --learning.rate 0.2",
    ]
    assert sorted(outputs) == sorted(expected_outputs)


def test_env_variables(echo_script: str) -> None:
    """Test both global and parameter-specific environment variables."""
    config = {
        "command": echo_script,
        "environment": {
            "TEST_GLOBAL_VAR": "global_value",
        },
        "parameters": {
            "param.name": {"values": ["value1"]},
            "environment": {
                "TEST_PARAM_VAR": "param_value",
            },
        },
    }
    outputs = run_sweep_and_capture(config, print_only=True)
    # Verify both global and parameter env vars are present by printing cmd
    assert any("TEST_GLOBAL_VAR=global_value" in line for line in outputs)
    assert any("TEST_PARAM_VAR=param_value" in line for line in outputs)


def test_env_variables_passed_to_subprocess(env_echo_script: str) -> None:
    """
    Test that environment variables are correctly passed to the subprocess.
    This test verifies our fix for environment variable handling.
    """
    config = {
        "command": env_echo_script,
        "environment": {
            "TEST_GLOBAL_VAR": "global_value",
        },
        "parameters": {
            "param.name": {"values": ["value1"]},
            "environment": {
                "TEST_PARAM_VAR": "param_value",
            },
        },
    }
    outputs = run_sweep_and_capture(config)

    # The script should output the environment variables it receives
    assert any("ARGS: --param.name value1" in line for line in outputs)
    assert any("ENV_TEST_GLOBAL_VAR: global_value" in line for line in outputs)
    assert any("ENV_TEST_PARAM_VAR: param_value" in line for line in outputs)


def test_multiple_commands_with_different_env(env_echo_script: str) -> None:
    """
    Test that each command execution gets its own environment variables.
    This verifies that environment variables don't persist between commands.
    """
    config = {
        "command": env_echo_script,
        "parameters": [
            {
                "set.name": {"values": ["set1"]},
                "environment": {
                    "TEST_GLOBAL_VAR": "value_for_set1",
                    "TEST_PARAM_VAR": "param_for_set1",
                },
            },
            {
                "set.name": {"values": ["set2"]},
                "environment": {
                    "TEST_GLOBAL_VAR": "value_for_set2",
                    "TEST_PARAM_VAR": "param_for_set2",
                },
            },
        ],
    }
    outputs = run_sweep_and_capture(config)

    # Check first command had its specific environment
    assert any("ARGS: --set.name set1" in line for line in outputs)
    assert any("ENV_TEST_GLOBAL_VAR: value_for_set1" in line for line in outputs)
    assert any("ENV_TEST_PARAM_VAR: param_for_set1" in line for line in outputs)

    # Check second command had its specific environment
    assert any("ARGS: --set.name set2" in line for line in outputs)
    assert any("ENV_TEST_GLOBAL_VAR: value_for_set2" in line for line in outputs)
    assert any("ENV_TEST_PARAM_VAR: param_for_set2" in line for line in outputs)


def test_command_line_env_variables(
    env_echo_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test that environment variables passed on the command line are preserved
    and don't interfere with configuration environment variables.

    This test simulates running:
    COMMAND_LINE_VAR=cli_value python titan_sweep.py config.json
    """
    # Set an environment variable that would be present when calling the script from command line
    monkeypatch.setenv("COMMAND_LINE_VAR", "cli_value")

    config = {
        "command": env_echo_script,
        "environment": {
            "TEST_GLOBAL_VAR": "global_value",
        },
        "parameters": {
            "param.name": {"values": ["value1"]},
            "environment": {
                "TEST_PARAM_VAR": "param_value",
            },
        },
    }

    # Update the env_echo_script to also print the COMMAND_LINE_VAR
    with open(env_echo_script, "a") as f:
        f.write('echo "ENV_COMMAND_LINE_VAR: $COMMAND_LINE_VAR"\n')

    outputs = run_sweep_and_capture(config)

    # Verify all environment variables are present
    assert any("ARGS: --param.name value1" in line for line in outputs)
    assert any("ENV_TEST_GLOBAL_VAR: global_value" in line for line in outputs)
    assert any("ENV_TEST_PARAM_VAR: param_value" in line for line in outputs)
    assert any("ENV_COMMAND_LINE_VAR: cli_value" in line for line in outputs)


if __name__ == "__main__":
    pytest.main([__file__])
