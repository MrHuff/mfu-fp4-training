#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import sys
import low_bits_training


def test__mxfp8__dynamic_loading_module():
    m = low_bits_training.experiments.import_experimental_module("mxfp8")
    assert m.__name__ == "low_bits_training.experiments.mxfp8"
    # Check it has been properly imported.
    modules = [v for v in sys.modules.keys() if v.startswith(m.__name__)]
    assert len(modules) > 0

    # Test calling a second time the import.
    mbis = low_bits_training.experiments.import_experimental_module("mxfp8")
    assert m is mbis
