#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest
import inspect

from low_bits_training.models import get_train_spec
from low_bits_training.metrics import (
    build_metrics_processor,
    MetricsProcessor,
    TTMetricsProcessor,
)


@pytest.mark.parametrize("name", ["llama3_gc", "llumup3"])
def test__train_specs__build_metrics_processor(name):
    spec = get_train_spec(name)
    # All GC train specs should use our custom metrics processor.
    assert spec.build_metrics_processor_fn is build_metrics_processor


@pytest.mark.parametrize("fn", ["__init__", "log"])
def test__metrics_processor__inspect_methods(fn):
    """Check the metrics processor has the right signature."""
    tt_fn_args = inspect.signature(getattr(TTMetricsProcessor, fn)).parameters
    fn_args = inspect.signature(getattr(MetricsProcessor, fn)).parameters

    assert len(fn_args) == len(tt_fn_args)
    assert list(fn_args.keys()) == list(tt_fn_args.keys())
    assert list(fn_args.values()) == list(tt_fn_args.values())
