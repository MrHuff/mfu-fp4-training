#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest
from dataclasses import fields

from low_bits_training.config import ConfigManager
from low_bits_training.config.job_config import (
    Model,
    ModelConfigRegistry,
    generate_flavor_hash,
)
from low_bits_training.models import get_model_config
from low_bits_training.models.llama3 import TransformerModelArgs


def test__generate_flavor_hash__name_structure():
    flavor = generate_flavor_hash("1B", {})
    assert flavor.startswith("1B_override_")


@pytest.mark.parametrize("args", [{}, {"dim": 4096}, {"dim": None}])
def test__model_config_registry__no_new_flavor_added(args):
    # Mapping to an existing flavor of a model => no need to generate a new one.
    registry = ModelConfigRegistry()
    flavor = registry.override_model_args("llama3", "8B", **args)
    assert flavor == "8B"


def test__model_config_registry__new_flavor_added():
    registry = ModelConfigRegistry()
    flavor = registry.override_model_args("llama3", "8B", dim=2048)
    assert "8B_override_" in flavor
    # Check model config generated
    orig_cfg = get_model_config("llama3", "8B")
    hash_cfg = get_model_config("llama3", flavor)
    assert type(hash_cfg) is type(orig_cfg)
    assert hash_cfg.dim == 2048
    # Keep original config values when not overwritten.
    assert hash_cfg.n_layers == orig_cfg.n_layers


def test__get_model_args_type():
    m = Model(name="llama3", flavor="8B")
    model_args_dtype = m.get_model_args_type()
    assert model_args_dtype is TransformerModelArgs


def test__model_args_dict__proper_extracting_args():
    m = Model(name="llama3", flavor="8B", dim=1472)
    args = m.model_args_dict
    assert args["dim"] == 1472
    assert args["n_layers"] is None


def test_model_args_creation_cli():
    """
    Tests model args overriden by command line interface
    """
    config = ConfigManager().parse_args(
        [
            "--model.name=llama3_gc",
            "--model.flavor=debugmodel",
            "--model.dim=1472",
            "--model.n_layers=100",
        ]
    )
    assert config.model.dim == 1472
    assert config.model.n_layers == 100

    assert "override" in config.model.flavor
    assert config.model.flavor.count("override") == 1

    model_args = get_model_config(config.model.name, config.model.flavor)
    assert model_args.dim == 1472
    assert model_args.n_layers == 100


def test_model_args_creation_env(monkeypatch):
    """
    Tests model args overriden by environment variables
    """
    monkeypatch.setenv("MODEL_DIM", "1472")
    monkeypatch.setenv("MODEL_N_LAYERS", "100")
    config = ConfigManager().parse_args(
        ["--model.name=llama3", "--model.flavor=debugmodel"]
    )

    assert config.model.dim == 1472
    assert config.model.n_layers == 100

    assert "override" in config.model.flavor
    assert config.model.flavor.count("override") == 1

    model_args = get_model_config(config.model.name, config.model.flavor)
    assert model_args.dim == 1472
    assert model_args.n_layers == 100


def test_model_args_creation_both(monkeypatch):
    """
    Tests model args overriden by environment variables and CLI
    """
    monkeypatch.setenv("MODEL_DIM", "1472")
    config = ConfigManager().parse_args(
        [
            "--model.name=llama3",
            "--model.flavor=debugmodel",
            "--model.n_layers=100",
        ]
    )

    assert config.model.dim == 1472
    assert config.model.n_layers == 100

    # check exactly one override string in model.flavor
    assert "override" in config.model.flavor
    assert config.model.flavor.count("override") == 1

    model_args = get_model_config(config.model.name, config.model.flavor)
    assert model_args.dim == 1472
    assert model_args.n_layers == 100


def test_model_args_creation_flavor_only():
    """
    Tests model args created by config.model.flavor only"""
    config = ConfigManager().parse_args(
        [
            "--model.name=llama3_gc",
            "--model.flavor=1B",
        ]
    )

    # No overrides should return same arguments as in config.model.flavor
    model_args = get_model_config(config.model.name, config.model.flavor)
    # Check if the model_args are the same as in llama3_gc:1B
    assert model_args == get_model_config("llama3_gc", "1B")
    # Check override string is not present
    assert "override" not in config.model.flavor


@pytest.mark.parametrize("model_name", ["llama3"])
def test_job_config_model_args_synced(model_name):
    """
    Tests that JobConfig model arguments have matching TransformerModelArgs cls attribute
    this test will break when a new attribute is added to TransformerModelArgs by upstream torchtitan.
    To fix it, add the matching argument as a command line option.
    """
    config = ConfigManager().parse_args([f"--model.name={model_name}"])
    model_args_cls = config.model.get_model_args_type()

    # Comparing fields from ModelArgs class and Model config class.
    model_cfg_fields = {f.name for f in fields(config.model)}
    model_args_fields = {f.name for f in fields(model_args_cls)}

    # vocab_size set by Tokenizer, not Model config.
    # TODO: Modify model config to support dataclass-in-dataclass fields (rope_scaling_args).
    model_args_fields -= {"_enforced", "vocab_size", "rope_scaling_args"}

    # All model args should be available in Model config.
    assert model_args_fields <= model_cfg_fields

    # All the default values in Model config should none => not overriding the Model defaults.
    for f in model_args_fields:
        assert config.model.__getattribute__(f) is None
