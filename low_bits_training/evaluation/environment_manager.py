# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
import tempfile
from pathlib import Path
import subprocess
import os
import shutil


class EnvironmentManager:
    pop_env_vars = ["PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"]

    def __init__(self, pip_requirements: list[str]):
        # Create a temporary directory to hold the virtual environment
        self.env_dir = tempfile.mkdtemp(prefix="env_manager_")
        self.pip_requirements = pip_requirements
        self.python_executable = Path(self.env_dir) / "bin" / "python"

    def __enter__(self):
        # Create the virtual environment and install dependencies
        env = os.environ.copy()
        for var in self.pop_env_vars:
            env.pop(var, None)
        venv_command = ["uv", "venv", "-p", "python3", self.env_dir]
        install_command = [
            "uv",
            "pip",
            "install",
        ] + self.pip_requirements
        try:
            if not self.python_executable.exists():
                subprocess.run(venv_command, check=True, env=env, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to create virtual environment: {e}, {e.stdout}, {e.stderr}"
            )
        try:
            self.run_in_env(
                subprocess.run, install_command, check=True, env=env, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to install dependencies: {e}, {e.stdout} {e.stderr}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if Path(self.env_dir).exists():
            shutil.rmtree(str(self.env_dir))

    def run_in_env(self, subprocess_function, command: list[str], *args, **kwargs):
        assert (
            self.python_executable.exists()
        ), "Environment not set up. Call setup() first."
        env = kwargs.pop("env", os.environ.copy())
        for var in self.pop_env_vars:
            env.pop(var, None)

        special_chars = " {:=,<>"
        joined_command = " ".join(
            (f"'{c}'" if any(s in c for s in special_chars) else c) for c in command
        )
        script = f"""
        . {self.env_dir}/bin/activate
        {joined_command}
        """
        return subprocess_function(["bash", "-c", script], env=env, *args, **kwargs)
