#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch

from .models import add_model_config

from torchtitan.models.deepseek_v3 import (  # noqa: F401
    DeepSeekV3ModelArgs,
    DeepSeekV3Model,
    deepseekv3_args,
    MoEArgs,
)
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.models.deepseek_v3 import parallelize_deepseekv3
from torchtitan.models.deepseek_v3 import DeepSeekV3StateDictAdapter
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm

from low_bits_training.components.tokenizer import build_hf_or_tiktoken_tokenizer

from ..optimizer import build_optimizers
from ..lr_scheduler import build_lr_schedulers
from ..datasets import build_dataloader
from ..metrics import build_metrics_processor

from dataclasses import replace

# Copy the TorchTitan collection of config (to avoid directly mutating it).
deepseekv3_gc_configs = {name: replace(cfg) for name, cfg in deepseekv3_args.items()}

register_train_spec(
    "deepseek_v3_gc",
    TrainSpec(
        model_cls=DeepSeekV3Model,
        model_args=deepseekv3_gc_configs,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_or_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_metrics_processor_fn=build_metrics_processor,
        build_validator_fn=build_validator,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    ),
)

add_model_config(
    "deepseek_v3_gc",
    "debugmodel",
    DeepSeekV3ModelArgs(
        vocab_size=2048,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=6,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=2,
            top_k=3,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    ),
)

add_model_config(
    "deepseek_v3_gc",
    "16A3B",
    DeepSeekV3ModelArgs(
        vocab_size=129280,  # Use the V3-base tokenizer
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
)


# A deepseek model sized to be similar in number of active parameters to our 1B llama 3 1B
add_model_config(
    "deepseek_v3_gc",
    "18A1B",
    DeepSeekV3ModelArgs(
        vocab_size=129280,  # Use the V3-base tokenizer
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=17,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
)


def estimate_moe_params(self: DeepSeekV3ModelArgs):
    qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
    mla_params = (
        self.dim * ((self.q_lora_rank) or qk_head_dim * self.n_heads)  # Q proj
        + self.q_lora_rank * qk_head_dim * self.n_heads
        + self.kv_lora_rank
        * (self.v_head_dim * self.n_heads + qk_head_dim * self.n_heads)  # KV proj
        + self.dim * self.dim  # Q proj and Output
    )
    # Factor of 3 as we use gated units
    dense_params = self.dim * self.inter_dim * 3
    single_expert_params = self.dim * self.moe_inter_dim * 3
    moe_params = single_expert_params * (
        self.moe_args.num_experts + self.moe_args.num_shared_experts
    )
    output = self.dim * self.vocab_size
    return dict(
        matmul_params=mla_params * self.n_layers
        + output
        + self.n_dense_layers * dense_params
        + (self.n_layers - self.n_dense_layers) * moe_params,
        active_params=mla_params * self.n_layers
        + output
        + self.n_dense_layers * dense_params
        + (self.n_layers - self.n_dense_layers)
        * single_expert_params
        * (self.moe_args.num_shared_experts + self.moe_args.top_k),
        embedding_params=output,
    )


# Deepseek's apply compile calls into this private function which does not
# exist on some versions of torch. We pray that making it return nothing
# does the trick
# torchtitan_submodule/torchtitan/models/llama4/infra/parallelize.py
if not hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"):
    torch._C._dynamo.eval_frame._set_lru_cache = lambda x: None
