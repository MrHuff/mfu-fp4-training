#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from low_bits_training.utils import JobConfig


def test__mxfp8_utils__get_swap_filter_fn():
    from low_bits_training.experiments.mxfp8 import get_swap_filter_fn

    fn = get_swap_filter_fn(None, "all_but_output")
    assert fn is not None


def test__mxfp8_utils__register_swap_filter_fn():
    from low_bits_training.experiments.mxfp8 import (
        register_swap_filter_fn,
        get_swap_filter_fn,
        unregister_swap_filter_fn,
    )

    def swap_fn(mod, fqn):
        return True

    fname = "test_swap_fn"
    register_swap_filter_fn(fname, swap_fn)
    assert get_swap_filter_fn(None, fname) is not None
    assert unregister_swap_filter_fn(fname) is swap_fn


def test__make_mxfp_converter_class__proper_patching_config():
    from low_bits_training.experiments.mxfp8.mxfp_utils import (
        make_mxfp_converter_class,
        ScaleCalculationMode,
    )

    mxfp_class, mxfp_name = make_mxfp_converter_class(
        ScaleCalculationMode.EVEN, "custom_filter"
    )
    assert mxfp_name == "mxfp__even__custom_filter"
    cvt = mxfp_class(JobConfig(), None)
    assert cvt._mx_config.scale_rounding_fn == ScaleCalculationMode.EVEN
    assert cvt._mx_config.swap_filter_fn == "custom_filter"
