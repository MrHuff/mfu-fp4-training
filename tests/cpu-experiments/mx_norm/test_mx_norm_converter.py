#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import os
import sys
from copy import deepcopy
import pytest
from unittest.mock import Mock

sys.path.append(os.path.dirname(__file__) + "/../../../torchtitan_submodule")

from torchtitan.protocols.model_converter import (
    _registry_model_converter_cls,
    build_model_converters,
)

# import of low_bits_training will update _registry_model_converter_cls
import low_bits_training  # noqa

from low_bits_training.config import JobConfig  # noqa
from torchtitan.distributed import ParallelDims  # noqa


class TestConverter:
    @pytest.fixture
    def base_config(self):
        return JobConfig()

    @pytest.fixture
    def mock_parallel_dims(self):
        return Mock(spec=ParallelDims)

    @pytest.mark.parametrize(
        "method",
        [
            "fixed_point_iter_with_lut",
            "lut_and_lerp",
            "linear_scale",
            "mean_absmax",
            "rms",
        ],
    )
    def test_mxnorm_converter(self, method, base_config, mock_parallel_dims):
        converter_key = "mx_rmsnorm"
        config = _setup(base_config, converter_key)
        low_bits_training.experiments.import_experimental_module("mx_norm")
        config.mx_rmsnorm.sigma_absmax_mapping_fn = method
        converter = _verify(config, mock_parallel_dims, converter_key)
        assert (
            converter._sigma_absmax_mapping_fn
            == config.mx_rmsnorm.sigma_absmax_mapping_fn
        )

    @pytest.mark.parametrize(
        "norm_mode",
        ["pre", "post"],
    )
    def test_mxnormlinear_converter(self, norm_mode, base_config, mock_parallel_dims):
        converter_key = "mx_norm_linear"
        config = _setup(base_config, converter_key)
        low_bits_training.experiments.import_experimental_module("mx_norm")
        config.mx_norm_linear.norm_mode = norm_mode
        converter = _verify(config, mock_parallel_dims, converter_key)
        assert converter._norm_mode.name.lower() == norm_mode


def _setup(base_config, converter_key):
    config = deepcopy(base_config)
    config.model.converters = [converter_key]
    return config


def _verify(config, parallel_dims, converter_key):
    converter = build_model_converters(config, parallel_dims).converters[0]
    converter_cls = _registry_model_converter_cls[converter_key]
    assert isinstance(converter, converter_cls)
    return converter
