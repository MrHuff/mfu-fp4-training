#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Type, Dict, Any
import itertools
import tyro

from torchtitan.config.manager import ConfigManager as TTConfigManager
from torchtitan.config.manager import custom_registry, logger

from dataclasses import fields
import os
import json
import sys

from .job_config import JobConfig


def make_job_config_fields_env_mapping(config_cls: Type[JobConfig]) -> Dict[str, str]:
    """Build the mapping between env. variable names and JobConfig field names.

    Helper function to support env. variables as config arguments.
    """

    def to_env_name(f: str):
        return f.replace(".", "_").upper()

    # Create fields fullnames.
    config_fields = list(
        itertools.chain(
            *[
                [f"{f.name}.{sf.name}" for sf in fields(f.type)]
                for f in fields(config_cls)
            ]
        )
    )
    env_config_fields_map = {to_env_name(f): f for f in config_fields}
    return env_config_fields_map


def get_env_variables_arguments(config_cls: Type[JobConfig]) -> list[str]:
    """Generate a list of arguments corresponding to (optional) env. variables."""
    env_config_fields_map = make_job_config_fields_env_mapping(config_cls)

    def make_arg(k: str, v: str) -> str:
        f = env_config_fields_map[k]
        return f"--{f}={v}"

    env_args = [
        make_arg(k, v) for k, v in os.environ.items() if k in env_config_fields_map
    ]
    return env_args


def parse_job_config_file(args: list[str]) -> str | None:
    # 1. Check CLI
    valid_keys = {"--job.config-file", "--job.config_file"}
    # Starting from the end.
    inverse_args = args[::-1]
    for i, arg in enumerate(inverse_args):
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key in valid_keys:
                file_path = value
                break
        elif arg in valid_keys:
            file_path = inverse_args[i - 1]
            break
    else:
        return None
    return file_path


def config_upgrade_old_arguments(args_dict: Dict[str, Dict[str, Any]]):
    # How to remap old arguments to new arguments either:
    # - a string indicates it's just a renaming
    # - a tuple of string (rename) and argument map - either a dict or a callable
    def append_float_8_converter(mutable_arg, enable_float8_linear: bool):
        mutable_arg = mutable_arg or []
        if enable_float8_linear:
            mutable_arg.append("quantize.linear.float8")
        return mutable_arg

    def replace_float_8_converter(mutable_arg, *args):
        if mutable_arg == "float8":
            return "quantize.linear.float8"
        elif isinstance(mutable_arg, (list, tuple)):
            mutable_arg_ = []
            for c in mutable_arg:
                if c == "float8":
                    mutable_arg_.append("quantize.linear.float8")
                else:
                    mutable_arg_.append(c)
            return mutable_arg_

        return []

    argument_remap = {
        "optimizer.lr_scheduler": (
            "lr_scheduler.decay_type",
            {
                "linear_warmup_linear_decay": "linear",
                "linear_warmup_cosine_decay": "cosine",
            },
        ),
        "optimizer.lr_scheduler_args": None,
        "checkpoint.interval_type": None,
        "experimental.pipeline_parallel_microbatches": None,
        "optimizer.fused": (
            "optimizer.implementation",
            {True: "fused", False: "for-loop"},
        ),
        "experimental.custom_model_path": None,
        "metrics.rank_0_only": (
            "metrics.distributed_mode",
            {True: "rank_0", False: "all"},
        ),
        "training.batch_size": "training.local_batch_size",
        "training.compile": "compile.enable",
        "training.warmup_steps": "lr_scheduler.warmup_steps",
        "training.tensor_parallel_degree": "parallelism.tensor_parallel_degree",
        "parallelism.enable_compiled_autograd": None,
        "parallelism.pipeline_parallel_split_points": None,
        "experimental.enable_compiled_autograd": None,
        "experimental.pipeline_parallel_split_points": None,
        "experimental.context_parallel_degree": "parallelism.context_parallel_degree",
        "experimental.pipeline_parallel_schedule_csv": "parallelism.pipeline_parallel_schedule_csv",
        "experimental.pipeline_parallel_degree": "parallelism.pipeline_parallel_degree",
        "experimental.context_parallel_rotate_method": "parallelism.context_parallel_rotate_method",
        "experimental.enable_async_tensor_parallel": "parallelism.enable_async_tensor_parallel",
        "experimental.pipeline_parallel_schedule": "parallelism.pipeline_parallel_schedule",
        "training.fsdp_reshard_after_forward": "parallelism.fsdp_reshard_after_forward",
        "training.disable_loss_parallel": "parallelism.disable_loss_parallel",
        "training.data_parallel_replicate_degree": "parallelism.data_parallel_replicate_degree",
        "training.data_parallel_shard_degree": "parallelism.data_parallel_shard_degree",
        "training.deterministic": "debug.deterministic",
        "training.seed": "debug.seed",
        "checkpoint.enable_checkpoint": "checkpoint.enable",
        "checkpoint.model_weights_only": None,
        "job.use_for_integration_test": None,
        "job.json_config_file": None,
        "job.print_args": "job.print_config",
        "model.use_flex_attn": None,
        "model.attn_mask_type": None,
        "model.attn_mask_modifier": None,
        "model.attn_score_modifier": None,
        "model.attn_score_modifier_kwargs": None,
        "model.umup_alpha_ffn_act": "umup.alpha_ffn_act",
        "model.umup_alpha_attn_softmax": "umup.alpha_attn_softmax",
        "model.umup_alpha_res": "umup.alpha_res",
        "model.umup_alpha_res_attn_ratio": "umup.alpha_res_attn_ratio",
        "model.umup_alpha_loss_softmax": "umup.alpha_loss_softmax",
        "model.converters": (
            "model.converters",
            replace_float_8_converter,
        ),
        "float8.moe_fqns_prototype": None,
        "float8.enable_float8_linear": ("model.converters", append_float_8_converter),
        "float8.force_recompute_fp8_weight_in_bwd": None,
        "float8": "quantize.linear.float8",
        "mx.moe_fqns_prototype": None,
        "mx": "quantize.linear.mx",
        "memory_estimation.enabled": "memory_estimation.enable",
        "lr_scheduler.lr_min": "lr_scheduler.min_lr_factor",
        "mxfp.scale_rounding": "mxfp.scale_rounding_fn",
        "stats.layer_names": "stats.layer_patterns",
    }

    def pop_nested(d, path):
        """Delete value from nested dict using dot-separated path."""
        keys = path.split(".")
        current = d

        for key in keys[:-1]:
            if key not in current or not isinstance(current, dict):
                return None

            current = current[key]

        # Delete final key, this can be a path or a value
        if keys[-1] in current:
            return current.pop(keys[-1])

        return None

    def set_nested(d, path, value):
        """Set value in nested dict using dot-separated path, creating intermediate dicts as needed."""
        keys = path.split(".")
        current = d
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    for old_path, conversion in argument_remap.items():
        old_val = pop_nested(args_dict, old_path)

        if old_val is None:
            # If path to remap not present in config then skip
            continue

        if conversion is None:
            # If conversion is set to None then drop the argument
            logger.warning(
                "Config upgrade is dropping key %s with value %s", old_path, old_val
            )
            continue

        # Otherwise conversion is a string or a tuple describing a path mapping
        new_path = conversion if isinstance(conversion, str) else conversion[0]

        # Default case (str) is just renaming the path but not the value
        if isinstance(conversion, str):
            new_val = old_val
        elif isinstance(conversion, (tuple, list)):
            # Tuple/list mapping means value is converted for a given key
            converter = conversion[1]
            if isinstance(converter, dict):
                # Either according to a dict mapping
                new_val = converter.get(old_val)
            elif callable(converter):
                # Or according to a function mapping
                current_val = pop_nested(args_dict, new_path)
                new_val = converter(current_val, old_val)
            else:
                raise NotImplementedError(f"Unknown converter type: {type(converter)}")
        else:
            raise NotImplementedError(f"Unknown conversion type: {type(conversion)}")

        set_nested(args_dict, new_path, new_val)

        logger.info(
            "Config upgrade changed key %s=%s to %s=%s",
            old_path,
            old_val,
            new_path,
            new_val,
        )


class ConfigManager(TTConfigManager):
    """
    Parses, merges, and validates a JobConfig from TOML and CLI sources.

    Compared to TorchTitan, also supported env. variables arg passing.

    Configuration precedence:
        CLI args > env. variables > TOML file > JobConfig defaults
    """

    def __init__(self, config_cls: Type[JobConfig] = JobConfig):
        super().__init__(config_cls)

    def _maybe_load_toml_or_json(self, args: list[str]) -> dict[str, Any] | None:
        """Try loading config, from toml or json file."""
        config_file = parse_job_config_file(args)
        if config_file and not os.path.exists(config_file):
            error_msg = f"Job config file '{config_file}' not found."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Supporting additional case of json config.
        if config_file and config_file.endswith(".json"):
            with open(config_file, "rb") as f:
                return json.load(f)
        return self._maybe_load_toml(args)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.ema_checkpoint.enable_checkpoint and cfg.checkpoint.enable:
            assert (
                cfg.checkpoint.interval % cfg.ema_checkpoint.save_interval == 0
            ), f"EMA save interval must divide checkpoint save interval: {cfg.checkpoint.interval} / {cfg.ema_checkpoint.save_interval}."
        if getattr(cfg.training, "enable_cce", False) and getattr(cfg.fp4_cce, "enabled", False):
            raise ValueError(
                "training.enable_cce and fp4_cce.enabled cannot both be true. "
                "Use training.enable_cce only for the legacy Triton path, or fp4_cce for the unified CCE backend family."
            )
        return super()._validate_config()

    def parse_args(
        self,
        args: list[str] = sys.argv[1:],
        allow_upgrade: bool = False,
        validate: bool = True,
    ) -> JobConfig:
        """Parse arguments, supporting integrating env. variables.

        Modifications compared to TorchTitan `JobConfig.parse_args`:
            - Incorporate env. variables.
            - Changing defaults.
            - Upgrade old arguments, supporting older config files.
        """
        logger.info(", ".join(args))
        toml_values = self._maybe_load_toml_or_json(args)
        config_cls = self._maybe_add_custom_config(args, toml_values)
        # New parsing pass to incorporate env. variables (prepend to have proper precedence).
        # Necessary to do it here in case a custom config cls is used.
        args = get_env_variables_arguments(config_cls) + args
        toml_values = self._maybe_load_toml_or_json(args)

        if toml_values and allow_upgrade:
            config_upgrade_old_arguments(toml_values)

        # TorchTitan implementation.
        base_config = (
            self._dict_to_dataclass(config_cls, toml_values)
            if toml_values
            else config_cls()
        )
        self.config = tyro.cli(
            config_cls, args=args, default=base_config, registry=custom_registry
        )
        if validate:
            self._validate_config()
        return self.config
