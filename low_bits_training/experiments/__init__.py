#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import importlib


def import_experimental_module(mname: str):
    """Import a low-bits-training experimental module."""
    return importlib.import_module(f"low_bits_training.experiments.{mname}")
