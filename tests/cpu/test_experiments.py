#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import low_bits_training  # import even if not used.
import sys
import pytest
import os

LOW_BITS_TRAINING_PATH = os.path.dirname(low_bits_training.__file__)
EXPERIMENTS_PATH = os.path.join(LOW_BITS_TRAINING_PATH, "experiments")


def test__experiments__no_modules_imported_by_default():
    # No experimental modules should be imported by default, as it may "pollute" the core library.
    modules = [
        m for m in sys.modules.keys() if m.startswith("low_bits_training.experiments.")
    ]
    assert modules == []
    # Root `experiments` loaded.
    assert "low_bits_training.experiments" in [m for m in sys.modules.keys()]


@pytest.mark.parametrize(
    "module",
    [
        f.path
        for f in os.scandir(EXPERIMENTS_PATH)
        if f.is_dir() and not f.name.startswith("__")
    ],
)
def test__experiments__module_has_codeowner(module):
    module_path = module.replace(LOW_BITS_TRAINING_PATH, "low_bits_training")
    codeowners_path = os.path.join(LOW_BITS_TRAINING_PATH, "..", "CODEOWNERS")

    with open(codeowners_path, "r") as f:
        lines = f.readlines()
        lines_with_modules = [v for v in lines if v.startswith(module_path)]
        # Should have at least one codeowner!
        assert (
            len(lines_with_modules) > 0
        ), f"Please provide a CODEOWNER for {module_path}"
