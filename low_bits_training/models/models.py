#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from torchtitan.models.llama3 import (  # noqa: F401
    TransformerModelArgs,
)

from torchtitan.protocols.train_spec import (
    get_train_spec,
    BaseModelArgs,
    TrainSpec,  # noqa: F401
    _extra_train_specs,
)

from typing import Any, Callable


def get_model_config(model_name: str, model_flavor: str) -> BaseModelArgs:
    """Get a model configuration (if existing) registered in a TorchTitan `TrainSpec`.

    TorchTitan train specs can be retrieve using `get_train_spec`, and model config
    can be added using `add_model_config`.

    Args:
        model_name: General model/architecture name (e.g. `llama3`), key of a TorchTitan `TrainSpec`.
        model_flavor: Model flavor to retrieve (e.g. `1B`, `3B`, ...)
    Returns:
        Model configuration corresponding to the flavor.
    """
    train_spec = get_train_spec(model_name)
    if model_flavor not in train_spec.model_args:
        raise ValueError(
            f"Unknown model flavor '{model_flavor}' for model '{model_name}'. Flavors available: {list(train_spec.model_args.keys())}"
        )
    return train_spec.model_args[model_flavor]


def add_model_config(model_name: str, model_flavor: str, config: BaseModelArgs):
    """Add a new model config to an existing TorchTitan `TrainSpec`.

    Args:
        model_name: General model/architecture name (e.g. `llama3`), key of a TorchTitan `TrainSpec`.
        model_flavor: New model flavor to add (e.g. 1B).
        config: Config of the model flavor. Should be a `BaseModelArgs` (or sub-class) instance.
    """
    assert isinstance(config, BaseModelArgs)
    train_spec = get_train_spec(model_name)
    # TODO: should we allow overriding?
    # At present: using overriding for running unit tests.
    # if model_flavor in train_spec.model_args:
    #     raise ValueError(
    #         f"A model flavor '{model_flavor}' is '{model_name}' already existing."
    #     )
    train_spec.model_args[model_flavor] = config


def apply_to_train_specs(func: Callable[[TrainSpec], TrainSpec]) -> None:
    """Apply a function to all TorchTitan train specs, modifying them in place."""
    global _extra_train_specs
    for name, train_spec in _extra_train_specs.items():
        _extra_train_specs[name] = func(train_spec)


def patch_train_specs(field: str, old: Any, new: Any):
    """Patch a field in all registered train specs, replacing an old instance by a new one (e.g. `build_optimizers_fn`)
    The patching is only done on train specs where the current value correspond to the `old` field.

    NOTE: This function only modifies *externally* registered train specs (i.e., those in `_extra_train_specs`). It will
    not modify built-in TorchTitan train specs.
    """
    from dataclasses import replace

    def _patch_ts(ts: TrainSpec) -> TrainSpec:
        if ts.__getattribute__(field) == old:
            ts = replace(ts, **{field: new})
        return ts

    apply_to_train_specs(_patch_ts)
