# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
import pytest
import subprocess
from low_bits_training.evaluation.environment_manager import EnvironmentManager


def test_environment_manager():
    """Tests that a requirement (pytest) installed in the environment manager does not
    affect the parent environment and that commands run in the child environment show
    the right dependency."""

    def get_pytest_version_in_current_env():
        out = subprocess.check_output(["uv", "pip", "freeze"], text=True)

        return [line for line in out.splitlines() if "pytest==" in line][0]

    start_version = get_pytest_version_in_current_env()
    with EnvironmentManager([f"pytest<{pytest.__version__}"]) as manager:
        end_version = get_pytest_version_in_current_env()

        assert (
            start_version == end_version
        ), f"Pytest version has changed in the current env from {start_version} to {end_version}"
        out_in_env = manager.run_in_env(
            subprocess.check_output, ["uv", "pip", "freeze"], text=True
        )
        assert manager.python_executable.exists()
    version_in_env = [line for line in out_in_env.splitlines() if "pytest==" in line][0]
    assert (
        version_in_env != start_version
    ), f"Version in env ({version_in_env}) is the same as the start version ({start_version}), despite requirement {manager.pip_requirements}"
    assert not manager.python_executable.exists()
