#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pathlib
import subprocess
import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]


@pytest.mark.xfail(reason="Passes locally but fails in CI with a tiktoken error")
def test_evaluate_prompt():
    """Test running evaluate.py with prompt generation."""
    result = subprocess.run(
        [
            "python",
            "evaluate.py",
            "--model-config",
            "tests/assets/test-checkpoint-unit_test/config-rank0-test-checkpoint-unit_test.json",
            "--model-checkpoint",
            "tests/assets/test-checkpoint-unit_test/checkpoint/step-0/",
            "prompt",
            "--prompt",
            '"Hello world"',
            "--max_new_tokens",
            "5",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode} and stderr {result.stderr}"
