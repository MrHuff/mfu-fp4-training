#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Type, TypeVar, Literal, Any, Dict

from torchtitan.config.job_config import JobConfig as TTJobConfig
from torchtitan.config.job_config import Training as TTTraining
from torchtitan.config.job_config import Profiling as TTProfiling
from torchtitan.config.job_config import Checkpoint as TTCheckpoint
from torchtitan.config.job_config import Metrics as TTMetrics
from torchtitan.config.job_config import Model as TTModel
from torchtitan.config.job_config import Job as TTJob
from torchtitan.models.llama3.model.args import RoPEScalingArgs
from dataclasses import (
    dataclass,
    field,
    make_dataclass,
    is_dataclass,
    replace,
    fields,
    asdict,
)
import pathlib
import os
import json
import hashlib


T = TypeVar("T")


@dataclass
class Job(TTJob):
    remote_folder: str | None = None
    """Remote (s3) folder to upload `dump_folder` to.
    """
    experimental_modules: list[str] = field(default_factory=list)
    """List of experimental modules to import."""

    steps: int = -1
    """
    Step at which to end training. Distinct from --training.steps, which is used by the learning rate scheduler.
    This way, one can run a debug run of the first N training steps of a long run with the correct learning rates.
    Disabled if set to -1 (which is the default).
    """


@dataclass
class Model(TTModel):
    # TODO: Modify model config to support dataclass-in-dataclass fields
    # TODO: Modify model config to support base config types other than TransformerModelArgs

    name: str = "llama3_gc"
    """Which model to train"""

    # Norm_type removed in upstream TorchTitan
    norm_type: Literal["layernorm", "np_layernorm", "rmsnorm"] = "rmsnorm"
    """Type of layer normalization to use [layernorm, np_layernorm, rmsnorm]"""

    # Additional customization model args
    dim: int | None = None
    """Model width: dimensionality of token embeddings"""
    n_layers: int | None = None
    """Number of transformer layers"""
    n_heads: int | None = None
    """Number of attention heads per query"""
    n_kv_heads: int | None = None
    """Number of attention heads per key/value pair"""
    multiple_of: int | None = None
    """Make SwiGLU hidden layer size multiple of large power of 2"""
    ffn_dim_multiplier: float | None = None
    """Expansion factor for MLP hidden layer dimensionality"""
    norm_eps: float | None = None
    """Small norm layer denominator stability constant"""
    eos_id: int | None = None
    """End Of Sentence token ID"""
    rope_theta: float | None = None
    """Base frequency of rotary positional embeddings"""
    max_seq_len: int | None = None
    """Maximum length of input sequences"""
    depth_init: bool | None = None
    """Transformer block init scaled by layer ID or total number of layer"""
    rope_scaling_args: RoPEScalingArgs | None = None
    """Scaling args for handling long context RoPE extension"""
    use_flex_attn: bool | None = None
    """Use flex attention."""
    attn_mask_type: str | None = None
    """Attention mask type in flex attention."""

    def get_model_args_type(self) -> Any:
        """Get the dataclass type used by for storing the model parameters."""
        from low_bits_training.models import get_train_spec

        possible_model_args_types = set(
            type(args) for args in get_train_spec(self.name).model_args.values()
        )
        if len(possible_model_args_types) == 0:
            raise Exception("No args type associated with this model name")
        if len(possible_model_args_types) > 1:
            raise Exception(
                f"Multiple possible args types found for {self.name} ({possible_model_args_types})"
            )
        # Best way I know of to get a 1-element set's only element
        model_args_type = list(possible_model_args_types)[0]
        return model_args_type

    @property
    def model_args_dict(self):
        """Get the model args override."""
        cfg_fields = [f.name for f in fields(self)]
        model_args_type = self.get_model_args_type()
        model_args_dict_ = {}
        for field_ in fields(model_args_type):
            if field_.name in cfg_fields:
                value = getattr(self, field_.name)
                # Check if it's a dataclass instance (not a dataclass type)
                if is_dataclass(value) and not isinstance(value, type):
                    model_args_dict_[field_.name] = asdict(value)
                else:
                    model_args_dict_[field_.name] = value

        return model_args_dict_


@dataclass
class Metrics(TTMetrics):
    distributed_mode: Literal["all", "local_rank_0", "rank_0"] = "local_rank_0"
    """
    Metrics collection distributed mode.
        all: All ranks reporting metrics.
        local_rank_0: Every node local rank 0 only.
        rank_0: Rank 0 process only.
    When pipeline_parallel_degree is > 1, the option `rank_0` uses the 0th rank of the last stage pipeline group,
    which is the only stage that computes loss metrics.
    """


@dataclass
class Training(TTTraining):
    enable_fp32_master_params: bool = True
    """
    Keep optimizer-owned parameters, gradients, and optimizer state in FP32
    whenever the model would otherwise use BF16 storage. FSDP/autocast still
    uses mixed_precision_param for forward compute. Disable only for an
    intentional full-BF16 memory or throughput experiment.
    """
    load_dataset_kwargs: str = "{}"
    """
    JSON arguments to pass to HF load_dataset method when creating the dataset.
    For example to enable dataset streaming pass: {"streaming": true} (Note the " quotes)
    This only works with compatible datasets (slimpajama)
    """
    dataset_node_distribution: Literal["shard", "hf-flaky-splitting"] = (
        "hf-flaky-splitting"
    )
    """
    Method to use for sharding the dataset across nodes. Shard should be used when it is supported.
     hf-flaky-splitting is kept as default for backward compatibility.
    """
    # --- ADDED: Control flags for CCE Patch and Manual Compilation ---
    enable_cce: bool = False
    """Enable Cut Cross Entropy patch. If true, standard torchtitan compilation should be false."""
    
    compile: bool = False 
    """
    Enable torch.compile. 
    NOTE: If enable_cce is True, this flag controls the *manual* compilation step in the custom Trainer.
    If using standard TorchTitan, this defaults to False here to prevent auto-compilation conflicts.
    """


@dataclass
class Profiling(TTProfiling):
    consecutive_active_steps: int = 1
    """"How many steps should be collected in one go, in iterations. Increase if you need to profile how steps overlap with each other."""
    memory_timeline: Literal[None, "json", "json.gz", "raw.json.gz"] = None
    """Whether to profile with the memory timeline - can cause very big profiles and bugs. Choices are defined by the file formats supported by `profile.export_memory_timeline` To see how to read those files open Pytorch's _memory_profiler.py, Good luck!","""
    with_flops_table: bool = False
    """Export flops estimate calculated by the profiler. This will slow down the profile output."""
    with_summary_metrics: bool = False
    """Summary metrics like comms compute overlap calculated by the profiler. This will slow down the profile output."""
    experimental_cupti_stats: bool = False
    """Turn on CUPTI hardware counters, currently broken - waiting for a fix: to https://github.com/pytorch/pytorch/issues/125272"""
    capture_compiled_kernels: bool = False
    """Capture the triton and inductor caches generated by this model. These will be in the dump folder."""


@dataclass
class Checkpoint(TTCheckpoint):
    compatibility: Literal["lbt-v1", "lbt-v0"] | None = None
    """Argument for turning on backward compatibility mode for checkpointing"""
    keep_latest_k: int = 0
    last_save_model_only: bool = False
    save_aligned_checkpoint: bool = False
    """Whether to save an aligned (BF16 casted) checkpoint at step 0 for parity debugging."""
    load_from_folder: str | None = None
    """Folder to pre-load checkpoint from (e.g. for aligned initialization)."""
    load_step: int | None = None
    """Step to load from load_from_folder."""


### New low-bits-training sub-configs.
@dataclass
class Statistics:
    layer_patterns: list[str] = field(default_factory=list)
    """List of glob-style layer name patterns to gather output statistics for. Supports standard glob patterns: '*' matches any sequence of characters, '?' matches any single character, '[seq]' matches any character in seq, '[!seq]' matches any character not in seq"""
    gather_forward: bool = False
    """Whether to gather statistics during the forward pass."""
    gather_backward: bool = False
    """Whether to gather statistics during the backward pass."""


@dataclass
class MXFP:
    """Standard MXFP Configuration (Deprecated/Legacy use)"""
    activation_dtype: str = "e4m3"
    weight_dtype: str = "e4m3"
    gradient_dtype: str = "e5m2"
    block_size: int = 32
    scale_rounding_fn: str = "floor"
    swap_filter_fn: str = "all_but_output"


@dataclass
class WandB:
    name: str = ""
    """Wandb run name."""
    id: str | None = None
    """Wandb run ID - will be used to restart a run or created at run-time if not specified."""
    group: str = ""
    """Wandb run group name."""
    project: str = "low-bits-training"
    """Wandb project name."""
    mode: Literal["online", "offline", "disabled"] = "online"
    """Wandb logging mode."""


@dataclass
class EMACheckpoint:
    enable_checkpoint: bool = False
    """Whether to enable Exponential moving average of the weights during training."""
    folder: str = "ema_checkpoint"
    """The sub-folder to store the EMA checkpoints in main `dump_folder`."""
    update_interval: int = 1
    """EMA update interval in steps."""
    save_interval: int = 500
    """EMA save checkpointing interval in steps"""
    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """Converts to the specified precision when training completes and model_weights_only=true. Currently supports float32, float16, and bfloat16."""
    async_mode: Literal["async_with_pinned_mem"] = "async_with_pinned_mem"
    """How to save the EMA checkpoint. Currently only supports async_with_pinned_mem."""
    ema_decay: float = 0.999
    """Decay factor for EMA weights. The value should be between 0 and 1.
        W_n = decay * W_n-1 + (1 - decay) * W_t. Where W_n is the EMA weight at step n,
        W_n-1 is the EMA weight at step n-1, and W_t are the training weights.
    """
    skip_first_k_updates: int = 2000
    """Number of updates to skip before updating EMA. Set to match the warmup steps."""


@dataclass
class UMuP:
    alpha_ffn_act: float | None = None
    """Multiplier for FFN activations."""
    alpha_attn_softmax: float | None = None
    """Multiplier for attention softmax."""
    alpha_res: float | None = None
    """Multiplier for residual branches."""
    alpha_res_attn_ratio: float | None = None
    """Additional multiplier for attention residual branches."""
    alpha_loss_softmax: float | None = None
    """Multiplier for final logits before softmax."""


@dataclass
class MXRMSNorm:
    elementwise_affine: bool = True
    """Whether to use elementwise affine parameters in MXRMSNorm."""
    block_size: int = 32
    """MXFP block size used in MXRMSNorm."""
    scale_rounding_fn: str = "ocp"
    """MXFP scale rounding function used in MXRMSNorm. To be used with model.converter 'mx_rmsnorm'."""
    scale_dtype: str = "e8m0"
    """MXFP scale dtype used in MXRMSNorm. To be used with model.converter 'mx_rmsnorm'."""
    data_dtype: str = "e4m3"
    """MXFP data dtype used in MXRMSNorm. To be used with model.converter 'mx_rmsnorm'."""
    sigma_absmax_mapping_fn: str = "fixed_point_iter_with_lut"
    """Name of function used to map absmax to RMS sigma. To be used with model.converter 'mx_rmsnorm'."""
    n_fixed_point_iterations: int = 3
    """Number of fixed point iterations to use with MXRMSNorm. To be used with model.convert 'mx_rmsnom'."""
    n_lut_entries: int = 256
    """Number of LUT entries in MXRMSNorm. To be used with model.convert 'mx_rmsnom'."""


@dataclass
class MXNormLinear:
    norm_mode: str = "pre"
    "Normalise MX cast pre- or post-scale rounding"
    reduction: str = "mean"
    "Reduction to apply over block absmax (mean or rms)"
    clamp_val: float | None = None
    "Value to clamp the output of MXNorm"
    n_lut_entries: int = 256
    "Number of lookup table entries estimating RMS with post-scale rounding"


@dataclass
class MXFPCustom:
    """
    Configuration for the MXLinearDimFused custom layer.
    """
    # --- Basic MXFP Settings ---
    block_size: int = 32
    elem_dtype: str = "fp4_e2m1"
    roundMode: str = "TiesToEven"
    scale_type: str = "E8M0"
    fp_scale_factor: bool = False
    nan_handling_mode: str = "nearest_subnormal"
    swap_filter_fn: str = "all_but_output"
    
    # --- Strategy ---
    strategy: str = "encode"
    use_fp32_scaling: bool = True

    # --- Legacy/Approx ---
    approx_sr: str = "IntelFP4_exact"
    approx_use_tensor_scaling: bool = True           
    approx_tensor_scaling_grad_est: str = "STE"
    approx_dtype: str = "bfloat16"

    # --- TE Integration & Advanced Physics ---
    mlp_recipe: str = "Custom"
    attn_recipe: str = "None"
    # Recipe for Nemotron-H Mamba in/out projections. Empty string inherits mlp_recipe.
    mamba_recipe: str = ""
    exclude_last_n_layers: int = 0
    exclude_qkv: bool = False
    verbose: bool = True
    quantize_mamba: bool = True

    use_global_scale: bool = True
    use_rht: bool = False
    
    # NEW: Rounding mode specifically for the Scale factor (e.g. Stoch for scales)
    scale_round_mode: str = "TiesToEven"
    use_2d_weights: bool = False
    use_2d_weights: bool = False
    use_fp32_matmul: bool = False
    scale_max: float | None = None

@dataclass
class TransformerEngineFP4:
    """
    Configuration for Transformer Engine FP4 selective replacement.
    """
    # Recipe for MLPs: "NVFP4", "MXFP4", or "None"
    mlp_recipe: str = "NVFP4"
    
    # Recipe for Attention: "NVFP4", "MXFP4", or "None"
    attn_recipe: str = "None"

    # Recipe for Nemotron-H Mamba in/out projections. Empty string inherits mlp_recipe.
    mamba_recipe: str = ""
    
    # Number of layers at the end of the model to exclude from quantization
    exclude_last_n_layers: int = 4

    # Number of final FFN/MLP blocks to leave in BF16 while still quantizing
    # attention in those transformer layers.
    exclude_last_n_ffn_layers: int = 0

    # Keep Q/K/V projections in high precision.
    exclude_qkv: bool = False

    # Quantize Nemotron-H Mamba in/out projections when using granular TE paths.
    quantize_mamba: bool = True
    
    # Verbosity of the replacement process
    verbose: bool = True


def config_update_defaults(sub_cls: Type[T], **kwargs) -> Type[T]:
    """Update a JobConfig sub-class defaults."""
    assert is_dataclass(sub_cls)
    sub_cls_name = sub_cls.__name__
    # Update field default, and re-create the config subclass.
    fields = sub_cls.__dataclass_fields__
    for k, v in kwargs.items():
        assert k in fields, f"'{k}' not a field in dataclass '{sub_cls_name}'."
        fields[k].default = v
    sub_cls_new = make_dataclass(sub_cls_name, [], bases=(sub_cls,))

    # Update JobConfig class field (and default factory).
    fields = TTJobConfig.__dataclass_fields__
    fname = next((k for k, v in dict(fields).items() if v.type is sub_cls))
    fields[fname].type = sub_cls_new
    fields[fname].default_factory = sub_cls_new
    return sub_cls_new


# Update some TorchTitan defaults.


@dataclass
class Quartet:
    """
    Configuration for Quartet-II integration.
    """
    four_over_six: bool = True
    """Whether to use 4/6 encoding (True) or standard E2M1 (False)."""
    exclude_last_n_layers: int = 4
    """Number of layers to exclude from quantization."""
    use_bf16_backward: bool = False
    """Use BF16 matmul in backward instead of eden Hadamard transforms. Faster but may affect convergence."""
    verbose: bool = True


@dataclass
class FP4CCE:
    enabled: bool = False
    """Enable the output/loss-only CCE backend patch."""

    backend: Literal[
        "triton_bf16",
        "torch_compile_bf16",
        "native_mxfp4",
        "nvfp4",
        "mxfp4",
    ] = "triton_bf16"
    """CCE backend implementation to use."""

    implementation: Literal["v2", "v3", "v4", "v5"] = "v2"
    """Backend implementation version where supported."""

    quant_mode: Literal["native", "enc", "dec", "rte"] = "enc"
    """Quantization mode for FP4 backends."""

    ignore_index: int = -100
    """Ignore index used in the internal loss path."""

    filter_eps: float | Literal["auto"] | None = 0.0
    """Optional CCE filter epsilon. Use 'auto' for Triton's BF16 default."""

    forward_precision: Literal["bf16", "fp4"] = "bf16"
    """Logits GEMM precision for the native MXFP4 CCE ablation."""

    backward_precision: Literal["bf16", "fp4"] = "bf16"
    """dHidden/dWeight GEMM precision for the native MXFP4 CCE ablation."""


@dataclass
class FA4:
    enabled: bool = True
    """Enable the FA4 attention converter when selected in model.converters."""

    mode: Literal["softmax", "softcap", "sigmoid_attention"] = "softmax"
    """Attention scoring mode exposed by the FA4 wrapper."""

    audit_coefficients: bool = False
    """Run the FA4 polynomial coefficient audit once during converter initialization."""

    softcap: float = 50.0
    """Softcap value used when mode='softcap'."""

    softcap_degree: int = 3
    """Polynomial degree for approximated softcap score modifiers."""

    softcap_backend: str = "cute"
    """Backend used to materialize softcap score modifiers."""

    softcap_backward_mode: str = "algebraic"
    """Backward implementation mode for approximated softcap score modifiers."""

    sigmoid_variant: Literal["poly", "sfu"] = "poly"
    """Sigmoid-attention frontend variant."""

    sigmoid_sfu_freq: int = 16
    """Forward SFU approximation frequency knob."""

    sigmoid_sfu_res: int = 0
    """Forward SFU approximation resolution knob."""

    sigmoid_sfu_freq_bwd: int | None = None
    """Optional backward override for sigmoid SFU frequency."""

    sigmoid_sfu_res_bwd: int | None = None
    """Optional backward override for sigmoid SFU resolution."""

    sigmoid_backward_mode: Literal["algebraic", "direct"] = "algebraic"
    """Select backward polynomial path for sigmoid attention."""

    sigmoid_bias: float | None = None
    """Optional bias term forwarded into FA4 sigmoid attention."""

    sigmoid_poly_backend: str = "cute"
    """Backend used for polynomial sigmoid attention."""

    sigmoid_qk_norm: bool = True
    """Apply per-head RMSNorm to Q/K before FA4 when supported."""


@dataclass
class SplineMLP:
    activation_impl: Literal["native_silu", "spline_silu", "native_gelu", "spline_gelu"] = "spline_silu"
    """Activation implementation used by the spline MLP converter."""


@dataclass
class JobConfig(TTJobConfig):
    """Extension of TorchTitan JobConfig."""

    # Torchtitan configs with additional fields
    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    metrics: Metrics = field(default_factory=Metrics)
    profiling: Profiling = field(default_factory=Profiling)
    training: Training = field(default_factory=Training)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)

    # Low-bits-training additional configs.
    ema_checkpoint: EMACheckpoint = field(default_factory=EMACheckpoint)
    stats: Statistics = field(default_factory=Statistics)
    mxfp: MXFP = field(default_factory=MXFP)
    wandb: WandB = field(default_factory=WandB)
    umup: UMuP = field(default_factory=UMuP)
    mx_rmsnorm: MXRMSNorm = field(default_factory=MXRMSNorm)
    mx_norm_linear: MXNormLinear = field(default_factory=MXNormLinear)

    # --- NEW SECTIONS ADDED HERE ---
    mxfp_custom: MXFPCustom = field(default_factory=MXFPCustom)
    te_fp4: TransformerEngineFP4 = field(default_factory=TransformerEngineFP4)
    quartet: Quartet = field(default_factory=lambda: Quartet())
    fp4_cce: FP4CCE = field(default_factory=FP4CCE)
    fa4: FA4 = field(default_factory=FA4)
    spline_mlp: SplineMLP = field(default_factory=SplineMLP)



    def __post_init__(self):
        # If W&B training run name existing, append to output directory.
        if len(self.wandb.name):
            self.job.dump_folder = os.path.join(self.job.dump_folder, self.wandb.name)
            if self.job.remote_folder is not None:
                self.job.remote_folder = os.path.join(
                    self.job.remote_folder, self.wandb.name
                )

        # Add a reference to `ema` config in main checkpoint config.
        self.checkpoint.ema = self.ema_checkpoint

        # Generate a new flavor when overriding model args.
        print("MODEL ARGS DICT:", self.model.model_args_dict)
        self.model.flavor = model_registry.override_model_args(
            self.model.name, self.model.flavor, **self.model.model_args_dict
        )

    def dump(self):
        """Write the configuration as JSON into the dump_folder with the wandb name and
        the rank in the name of the file."""

        config_id = ""
        if rank := os.getenv("RANK"):
            config_id += f"-rank{rank}"
        if self.wandb.name:
            config_id += f"-{self.wandb.name}"

        pathlib.Path(self.job.dump_folder).mkdir(parents=True, exist_ok=True)
        config_dict = self.to_dict()
        (pathlib.Path(self.job.dump_folder) / f"config{config_id}.json").write_text(
            json.dumps(config_dict, indent=2)
        )


def generate_flavor_hash(base_flavor: str, overrides: Dict[str, Any]) -> str:
    """Generate a deterministic hash for a set of model argument overrides."""
    sorted_items = sorted(overrides.items())
    override_str = json.dumps(sorted_items, sort_keys=True)

    hash_obj = hashlib.sha256(override_str.encode())
    short_hash = hash_obj.hexdigest()[:8]

    return f"{base_flavor}_override_{short_hash}"


def reconstruct_nested(value: Any, base_type: Type[Any] | str | Any | None) -> Any:
    """Recursively reconstruct dataclass instances from dictionaries."""
    if is_dataclass(base_type) and isinstance(base_type, type):
        field_types = {f.name: f.type for f in fields(base_type)}

        fields_ = {k: reconstruct_nested(v, field_types.get(k)) for k, v in value.items()}
        return base_type(**fields_)

    return value


class ModelConfigRegistry:
    """Registry for managing dynamic model configurations"""

    def __init__(self):
        self.override_args: Dict[str, Any] = {}

    def override_model_args(self, model_name: str, model_flavor: str, **kwargs) -> str:
        """
        Override specific ModelArgs parameters for a given model configuration.
        Returns new flavor name hashed with override args
        """
        from low_bits_training.models import get_model_config, add_model_config

        # Get model config. Raising an error if not existing.
        base_config = get_model_config(model_name, model_flavor)
        # prune kwargs if values are same as in base_config or are None
        base_config_dict = asdict(base_config)
        base_config_types = {f.name: f.type for f in fields(base_config)}

        kwargs = {
            k: reconstruct_nested(v, base_config_types.get(k))
            for k, v in kwargs.items()
            if v != base_config_dict.get(k) and v is not None
        }

        if kwargs:
            new_config = replace(base_config, **kwargs)
            new_flavor = generate_flavor_hash(model_flavor, kwargs)

            # Add new model flavor config.
            add_model_config(model_name, new_flavor, new_config)

            key = f"{model_name}:{model_flavor}"
            self.override_args[key] = kwargs
            return new_flavor
        else:
            return model_flavor


model_registry = ModelConfigRegistry()
