#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import os
import sys
from copy import deepcopy
import pytest
import torch
from unittest.mock import Mock

sys.path.append(os.path.dirname(__file__) + "/../../torchtitan_submodule")
from torchtitan.protocols.model_converter import (
    _registry_model_converter_cls,
    build_model_converters,
    ModelConverter,
)  # retrieve model converters in torchtitan only


# import of low_bits_training will update _registry_model_converter_cls
_tt_converter_registry = deepcopy(_registry_model_converter_cls)

import low_bits_training  # noqa
from low_bits_training.config import JobConfig  # noqa
from low_bits_training.converters import (  # noqa
    Float32MasterParamsConverter,
    ensure_fp32_master_params,
)
from low_bits_training.quantization.mxfp import ScaleCalculationMode  # noqa
from torchtitan.distributed import ParallelDims  # noqa

_lbt_converter_registry = deepcopy(_registry_model_converter_cls)

# diff of converters registered by low_bits_training
LBT_CONVERTER_KEYS = [
    k for k in _lbt_converter_registry.keys() if k not in _tt_converter_registry
]


@pytest.mark.parametrize("cname", ["bfloat16", "fp32_master", "mxfp"])
def test__model_converter__init_from_default_job_config(cname):
    # Testing converter is registered + support default config.
    config = JobConfig()
    config.model.converters = [cname]
    m = build_model_converters(config, None)
    assert len(m.converters) == 1


def test_fp32_master_converter_promotes_parameters_but_not_buffers():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones(4, dtype=torch.bfloat16)
            )
            self.alias = self.weight
            self.register_buffer(
                "freqs_cis",
                torch.ones(2, dtype=torch.complex64),
            )

    model = TinyModel()
    parameter_id = id(model.weight)
    config = JobConfig()
    config.training.mixed_precision_param = "bfloat16"

    Float32MasterParamsConverter(config, None).convert(model)

    assert id(model.weight) == parameter_id
    assert model.alias is model.weight
    assert model.weight.dtype == torch.float32
    assert model.freqs_cis.dtype == torch.complex64


def test_fp32_master_converter_keeps_adamw_updates_and_state_in_fp32():
    model = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    config = JobConfig()
    config.training.mixed_precision_param = "bfloat16"
    Float32MasterParamsConverter(config, None).convert(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = model.weight.detach().clone()
    model(torch.ones(1, 4)).square().sum().backward()
    optimizer.step()

    state = optimizer.state[model.weight]
    assert model.weight.dtype == torch.float32
    assert model.weight.grad.dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert not torch.equal(model.weight, before)


def test_fp32_master_converter_supports_meta_parameters():
    model = torch.nn.Linear(
        4,
        2,
        bias=False,
        device="meta",
        dtype=torch.bfloat16,
    )
    config = JobConfig()
    Float32MasterParamsConverter(config, None).convert(model)

    assert model.weight.device.type == "meta"
    assert model.weight.dtype == torch.float32


@pytest.mark.parametrize(
    ("dtype", "converters", "expected"),
    [
        ("bfloat16", [], ["fp32_master"]),
        ("float32", ["bfloat16", "fp4"], ["bfloat16", "fp4", "fp32_master"]),
        (
            "bfloat16",
            ["fp32_master", "mxfp4_tk"],
            ["mxfp4_tk", "fp32_master"],
        ),
        ("float32", ["fp4"], ["fp4"]),
    ],
)
def test_ensure_fp32_master_params(dtype, converters, expected):
    config = JobConfig()
    config.training.dtype = dtype
    config.model.converters = converters

    ensure_fp32_master_params(config)

    assert config.model.converters == expected


def test_ensure_fp32_master_params_allows_explicit_full_bf16_opt_out():
    config = JobConfig()
    config.training.dtype = "bfloat16"
    config.training.enable_fp32_master_params = False
    config.model.converters = ["bfloat16"]

    ensure_fp32_master_params(config)

    assert config.model.converters == ["bfloat16"]


class TestConverter:
    @pytest.fixture
    def base_config(self):
        return JobConfig()

    @pytest.fixture
    def mock_parallel_dims(self):
        return Mock(spec=ParallelDims)

    @pytest.mark.parametrize("converter_key", LBT_CONVERTER_KEYS)
    def test_converter_initialisation_with_defaults(
        self, converter_key, base_config, mock_parallel_dims
    ):
        config = _setup(base_config, converter_key)
        _verify(config, mock_parallel_dims, converter_key)

    @pytest.mark.parametrize("scale_rounding", list(ScaleCalculationMode))
    def test_mxfp_converter(self, scale_rounding, base_config, mock_parallel_dims):
        converter_key = "mxfp"
        config = _setup(base_config, converter_key)
        config.mxfp.scale_rounding_fn = scale_rounding.name.lower()
        converter = _verify(config, mock_parallel_dims, converter_key)

        assert (
            converter._mx_config.scale_rounding_fn
            == ScaleCalculationMode[config.mxfp.scale_rounding_fn.upper()]
        )

    def test_stats_gather_converter(self, base_config, mock_parallel_dims):
        converter_key = "stats"
        config = _setup(base_config, converter_key)
        config.stats.layer_patterns = [
            "embedding",
            "layer.*.attn",
            "layer.1.ffn",
            "output",
        ]
        config.stats.gather_forward = True
        config.stats.gather_backward = True
        converter = _verify(config, mock_parallel_dims, converter_key)
        assert converter.layer_patterns == config.stats.layer_patterns
        assert converter.gather_forward == config.stats.gather_forward
        assert converter.gather_backward == config.stats.gather_backward

    def test_nonexistent_converter(self, base_config, mock_parallel_dims):
        converter_key = "unicorn"
        config = _setup(base_config, converter_key)
        with pytest.raises(KeyError):
            _verify(config, mock_parallel_dims, converter_key)

    def test_job_config_underspecified_for_new_converter(
        self, base_config, mock_parallel_dims
    ):
        converter_key = "new"

        class NewConverter(ModelConverter):
            def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
                self.arg = job_config.new.arg

        low_bits_training.register_model_converter(NewConverter, converter_key)

        config = _setup(base_config, converter_key)
        with pytest.raises(AttributeError):
            _verify(config, mock_parallel_dims, converter_key)
        _registry_model_converter_cls.pop("new")


def _setup(base_config, converter_key):
    config = deepcopy(base_config)
    config.model.converters = [converter_key]
    return config


def _verify(config, parallel_dims, converter_key):
    converter = build_model_converters(config, parallel_dims).converters[0]
    converter_cls = _registry_model_converter_cls[converter_key]
    assert isinstance(converter, converter_cls)
    return converter
