#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.validate import build_validator
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from low_bits_training.components.tokenizer import build_hf_or_tiktoken_tokenizer

from ..datasets import build_dataloader
from ..lr_scheduler import build_lr_schedulers
from ..metrics import build_metrics_processor
from ..optimizer import build_optimizers
from .models import add_model_config


NEMOTRON_NANO_12B_PATTERN = (
    "M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-"
)

NEMOTRON_H_8B_PAPER_PATTERN = (
    "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
)

NEMOTRON_SDPA_BACKEND_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

NEMOTRON_SDPA_NO_CUDNN_BACKEND_PRIORITY = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _load_nemotron_h_config(hf_assets_path: str):
    def _load_vendored_config():
        assets_path = Path(hf_assets_path)
        logger.warning(
            "AutoConfig could not load Nemotron-H assets from %s. "
            "Falling back to the vendored Nemotron-H remote-code classes.",
            hf_assets_path,
        )
        try:
            from .nemotron_h_hf.configuration_nemotron_h import NemotronHConfig
            from .nemotron_h_hf.modeling_nemotron_h import NemotronHForCausalLM
        except ImportError as import_exc:
            raise RuntimeError(
                "Could not import vendored Nemotron-H remote-code classes"
            ) from import_exc

        config_kwargs = {}
        config_json = assets_path / "config.json"
        if config_json.exists():
            with config_json.open("r", encoding="utf-8") as f:
                config_kwargs = json.load(f)
        config_kwargs["model_type"] = "nemotron_h"
        auto_map = config_kwargs.setdefault("auto_map", {})
        auto_map.setdefault("AutoConfig", "configuration_nemotron_h.NemotronHConfig")
        auto_map.setdefault(
            "AutoModelForCausalLM",
            "modeling_nemotron_h.NemotronHForCausalLM",
        )

        config = NemotronHConfig(**config_kwargs)
        config.name_or_path = str(assets_path)
        config._name_or_path = str(assets_path)
        return config, NemotronHForCausalLM

    if not _env_flag("LBT_NEMOTRON_H_USE_NATIVE_TRANSFORMERS", False):
        return _load_vendored_config()

    try:
        config = AutoConfig.from_pretrained(
            hf_assets_path,
            trust_remote_code=True,
        )
        if getattr(config, "model_type", None) == "nemotron_h":
            return config, None
        logger.warning(
            "AutoConfig loaded %s from %s, expected model_type='nemotron_h'.",
            type(config).__name__,
            hf_assets_path,
        )
        return _load_vendored_config()
    except ValueError:
        return _load_vendored_config()


@dataclass
class NemotronHModelArgs(BaseModelArgs):
    hf_assets_path: str = "./assets/hf/NVIDIA-Nemotron-Nano-12B-v2-Base"
    vocab_size: int = 131072
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 62
    hybrid_override_pattern: str = NEMOTRON_NANO_12B_PATTERN
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_hidden_act: str = "relu2"
    use_bias: bool = False
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 131072
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = False
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = True
    use_cache: bool = False
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    ssm_state_size: int = 128
    mamba_num_heads: int = 32
    mamba_head_dim: int = 80
    n_groups: int = 8
    conv_kernel: int = 4
    mamba_hidden_act: str = "silu"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    use_conv_bias: bool = True
    mamba_proj_bias: bool = False
    chunk_size: int = 128

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        self.hf_assets_path = job_config.model.hf_assets_path
        seq_len = job_config.training.seq_len
        if seq_len > self.max_position_embeddings:
            logger.warning(
                f"Sequence length {seq_len} exceeds configured maximum "
                f"{self.max_position_embeddings}. Expanding model max_position_embeddings."
            )
            self.max_position_embeddings = seq_len

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        param_count = sum(p.numel() for p in model.parameters())
        # Rough dense-model proxy. Good enough for logging/comparison while we
        # bring the architecture up in this codebase.
        flops_per_token = 6.0 * param_count
        return param_count, flops_per_token


class NemotronHForCausalLMTitan(nn.Module):
    def __init__(self, model_args: NemotronHModelArgs):
        super().__init__()
        self.model_args = model_args

        config, model_cls = _load_nemotron_h_config(model_args.hf_assets_path)
        config.vocab_size = model_args.vocab_size
        config.hidden_size = model_args.hidden_size
        config.intermediate_size = model_args.intermediate_size
        config.num_hidden_layers = model_args.num_hidden_layers
        config.hybrid_override_pattern = model_args.hybrid_override_pattern
        config.num_attention_heads = model_args.num_attention_heads
        config.num_key_value_heads = model_args.num_key_value_heads
        config.head_dim = model_args.head_dim
        config.attention_bias = model_args.attention_bias
        config.attention_dropout = model_args.attention_dropout
        config.hidden_dropout = model_args.hidden_dropout
        config.mlp_bias = model_args.mlp_bias
        config.mlp_hidden_act = model_args.mlp_hidden_act
        config.use_bias = model_args.use_bias
        config.tie_word_embeddings = model_args.tie_word_embeddings
        config.max_position_embeddings = model_args.max_position_embeddings
        config.layer_norm_epsilon = model_args.layer_norm_epsilon
        config.rms_norm_eps = model_args.layer_norm_epsilon
        config.residual_in_fp32 = model_args.residual_in_fp32
        config.initializer_range = model_args.initializer_range
        config.rescale_prenorm_residual = model_args.rescale_prenorm_residual
        config.use_cache = model_args.use_cache
        config.ssm_state_size = model_args.ssm_state_size
        config.mamba_num_heads = model_args.mamba_num_heads
        config.mamba_head_dim = model_args.mamba_head_dim
        config.n_groups = model_args.n_groups
        config.conv_kernel = model_args.conv_kernel
        config.mamba_hidden_act = model_args.mamba_hidden_act
        config.time_step_min = model_args.time_step_min
        config.time_step_max = model_args.time_step_max
        config.time_step_floor = model_args.time_step_floor
        config.use_conv_bias = model_args.use_conv_bias
        config.mamba_proj_bias = model_args.mamba_proj_bias
        config.chunk_size = model_args.chunk_size
        # The released HF NemotronH remote-code model currently refuses
        # `sdpa` in `_autoset_attn_implementation`. Construct in "eager"
        # mode, then use PyTorch SDPA with explicit cuDNN-first backend
        # priority inside this wrapper's attention calls.
        config._attn_implementation = "eager"

        if model_cls is None:
            hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        else:
            hf_model = model_cls(config)
        backbone = hf_model.backbone

        self.config = config
        self.tok_embeddings = backbone.embeddings
        self.layers = backbone.layers
        self.norm = backbone.norm_f
        self.output = hf_model.lm_head
        self.gradient_checkpointing = False
        self._cudnn_sdpa_enabled = _env_flag(
            "TORCH_CUDNN_SDPA_ENABLED",
            True,
        ) and not _env_flag(
            "LBT_FORCE_DIRECT_FLASH_ATTN",
            False,
        )
        self._sdpa_backends = (
            NEMOTRON_SDPA_BACKEND_PRIORITY
            if self._cudnn_sdpa_enabled
            else NEMOTRON_SDPA_NO_CUDNN_BACKEND_PRIORITY
        )
        self._swap_attention_blocks_to_compiled_backend()

        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(self._cudnn_sdpa_enabled)

        if self.attention_impl == "sdpa":
            priority = (
                "CUDNN_ATTENTION > FLASH_ATTENTION > EFFICIENT_ATTENTION > MATH"
                if self._cudnn_sdpa_enabled
                else "FLASH_ATTENTION > EFFICIENT_ATTENTION > MATH"
            )
            logger.info(
                "Nemotron attention backend: NemotronHSdpaAttention with PyTorch SDPA "
                "priority %s",
                priority,
            )
        else:
            logger.info(
                "Nemotron attention backend: NemotronHFlashAttention2 "
                "(compiled flash-attention path)"
            )

    def _attention_context(self, device: torch.device):
        if device.type == "cuda" and self.attention_impl == "sdpa":
            return sdpa_kernel(self._sdpa_backends)
        return contextlib.nullcontext()

    def _swap_attention_blocks_to_compiled_backend(self) -> None:
        attention_layers = [
            layer for layer in self.layers if getattr(layer, "block_type", None) == "attention"
        ]
        if not attention_layers:
            self.attention_impl = "sdpa"
            self.attention_backend = "nemotron_sdpa_cudnn_first"
            return

        attention_module_name = type(attention_layers[0].mixer).__module__
        attention_module = importlib.import_module(attention_module_name)
        attention_classes = getattr(attention_module, "NEMOTRONH_ATTENTION_CLASSES", None)
        if attention_classes is None:
            raise RuntimeError(
                f"Could not find Nemotron attention class map in module {attention_module_name}"
            )

        chosen_impl = "sdpa"

        if chosen_impl not in attention_classes:
            raise RuntimeError(
                f"Requested Nemotron attention implementation '{chosen_impl}' is not available "
                f"in module {attention_module_name}"
            )
        if chosen_impl == "sdpa" and self.config.hidden_size != (
            self.config.num_attention_heads * self.config.head_dim
        ):
            raise RuntimeError(
                "Nemotron SDPA path requires hidden_size == num_attention_heads * head_dim. "
                "Use the strict shrink or another architecture-preserving configuration."
            )

        compiled_attention_cls = attention_classes[chosen_impl]
        converted_count = 0
        for layer in attention_layers:
            old_mixer = layer.mixer
            if isinstance(old_mixer, compiled_attention_cls):
                continue

            new_mixer = compiled_attention_cls(self.config, layer_idx=old_mixer.layer_idx)
            target_device = old_mixer.q_proj.weight.device
            target_dtype = old_mixer.q_proj.weight.dtype

            if target_device.type == "meta":
                if hasattr(new_mixer, "to_empty"):
                    new_mixer = new_mixer.to_empty(device=target_device)
                else:
                    new_mixer = new_mixer.to(device=target_device)
            else:
                new_mixer = new_mixer.to(device=target_device, dtype=target_dtype)
                new_mixer.load_state_dict(old_mixer.state_dict())

            new_mixer.train(old_mixer.training)
            layer.mixer = new_mixer
            converted_count += 1

        self.attention_impl = chosen_impl
        self.config._attn_implementation = chosen_impl
        if chosen_impl == "sdpa":
            self.attention_backend = "nemotron_sdpa_cudnn_first"
        else:
            self.attention_backend = f"nemotron_{chosen_impl}"
        logger.info(
            "Replaced %d Nemotron attention blocks with compiled implementation '%s'.",
            converted_count,
            chosen_impl,
        )

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        if buffer_device is not None:
            for buffer_name, buffer in self.named_buffers(recurse=True):
                if buffer is not None:
                    setattr(
                        self.get_submodule(buffer_name.rsplit(".", 1)[0])
                        if "." in buffer_name
                        else self,
                        buffer_name.rsplit(".", 1)[-1],
                        buffer.to(buffer_device),
                    )

        for module in self.modules():
            if module is self:
                continue
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
            elif hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                nn.init.ones_(module.weight)

        for module in self.modules():
            if (
                hasattr(module, "A_log")
                and hasattr(module, "D")
                and hasattr(module, "dt_bias")
            ):
                module.A_log._no_weight_decay = True
                module.D._no_weight_decay = True
                dt = torch.exp(
                    torch.rand(
                        self.config.mamba_num_heads,
                        device=module.dt_bias.device,
                        dtype=torch.float32,
                    )
                    * (
                        math.log(self.config.time_step_max)
                        - math.log(self.config.time_step_min)
                    )
                    + math.log(self.config.time_step_min)
                ).clamp(min=self.config.time_step_floor)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    inv_dt = inv_dt.to(module.dt_bias.dtype)
                    if isinstance(module.dt_bias, DTensor):
                        inv_dt = distribute_tensor(
                            inv_dt,
                            module.dt_bias.device_mesh,
                            module.dt_bias.placements,
                        )
                    module.dt_bias.copy_(inv_dt)
                module.dt_bias._no_reinit = True

            if isinstance(module, nn.Linear):
                if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                nn.init.ones_(module.weight)

            if self.config.rescale_prenorm_residual:
                for name, param in module.named_parameters(recurse=False):
                    if name == "out_proj.weight":
                        nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                        with torch.no_grad():
                            param /= math.sqrt(self.config.num_hidden_layers)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = int(cache_position[-1].item()) + 1

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[..., :mask_length].eq(0.0)
                    * attention_mask[:, None, None, :].eq(0.0)
                )
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                    padding_mask,
                    min_dtype,
                )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _update_mamba_mask(
        self,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            mamba_mask = None
        return mamba_mask

    def forward_hidden_states_for_cce(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: object | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.tok_embeddings(input_ids)

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        for mixer_block in self.layers:
            if mixer_block.block_type == "mamba":
                layer_mask = mamba_mask
            elif mixer_block.block_type == "attention":
                layer_mask = causal_mask
            elif mixer_block.block_type == "mlp":
                layer_mask = None
            else:
                raise ValueError(f"Invalid block_type: {mixer_block.block_type}")

            if mixer_block.block_type == "attention":
                with self._attention_context(hidden_states.device):
                    hidden_states = mixer_block(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=layer_mask,
                    )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )

        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_params: object | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.forward_hidden_states_for_cce(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_params=cache_params,
            cache_position=cache_position,
        )
        logits = self.output(hidden_states.to(self.output.weight.dtype)).float()

        if labels is None:
            return logits

        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )


def apply_nemotron_fsdp(
    model: nn.Module,
    dp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    preserve_input_dtype_mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = True
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    fully_shard(
        model.tok_embeddings,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

    for layer in model.layers:
        layer_fsdp_config = fsdp_config
        if getattr(layer, "_fsdp_preserve_forward_input_dtypes", False):
            layer_fsdp_config = {
                **fsdp_config,
                "mp_policy": preserve_input_dtype_mp_policy,
            }
        fully_shard(
            layer,
            **layer_fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(
        [model.norm, model.output],
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )
    fully_shard(model, **fsdp_config)


def parallelize_nemotron_h(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Nemotron HF wrapper currently does not support tensor parallelism."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Nemotron HF wrapper currently does not support context parallelism."
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Nemotron HF wrapper currently does not support pipeline parallelism."
        )
    if job_config.activation_checkpoint.mode != "none":
        raise NotImplementedError(
            "Nemotron HF wrapper currently expects activation_checkpoint.mode = 'none'."
        )

    world_mesh = parallel_dims.world_mesh
    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if parallel_dims.fsdp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        apply_nemotron_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )
        logger.info("Applied FSDP to Nemotron HF wrapper")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(model, world_mesh, enable_compile=model_compile_enabled)

    return model


register_train_spec(
    "nemotron_h_gc",
    TrainSpec(
        model_cls=NemotronHForCausalLMTitan,
        model_args={},
        parallelize_fn=parallelize_nemotron_h,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_or_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_metrics_processor_fn=build_metrics_processor,
        build_validator_fn=build_validator,
        state_dict_adapter=None,
    ),
)

add_model_config(
    "nemotron_h_gc",
    "1.08B_strict",
    NemotronHModelArgs(),
)

add_model_config(
    "nemotron_h_gc",
    "8B_paper",
    NemotronHModelArgs(
        hidden_size=4096,
        intermediate_size=21504,
        num_hidden_layers=52,
        hybrid_override_pattern=NEMOTRON_H_8B_PAPER_PATTERN,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=8192,
        ssm_state_size=128,
        mamba_num_heads=128,
        mamba_head_dim=64,
        n_groups=8,
        conv_kernel=4,
        chunk_size=256,
    ),
)
