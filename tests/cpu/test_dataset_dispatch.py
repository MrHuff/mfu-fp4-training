#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
import sys
from types import ModuleType, SimpleNamespace

from low_bits_training.datasets import common


def test_mosaic_loader_is_imported_only_when_selected(monkeypatch):
    result = object()
    calls = []
    module = ModuleType("low_bits_training.datasets.mosaic_datasets")

    def build_mosaic_dataloader(*args):
        calls.append(args)
        return result

    module.build_mosaic_dataloader = build_mosaic_dataloader
    monkeypatch.setitem(
        sys.modules,
        "low_bits_training.datasets.mosaic_datasets",
        module,
    )
    job_config = SimpleNamespace(
        training=SimpleNamespace(dataset="mosaic/test")
    )

    actual = common.build_dataloader(2, 1, None, job_config, False)

    assert actual is result
    assert calls == [(2, 1, None, job_config, False)]
