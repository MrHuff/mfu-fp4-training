#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from torchtitan.components.validate import build_validator
from torchtitan.models.llama3 import (
    parallelize_llama,
    Llama3StateDictAdapter,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm

from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from low_bits_training.components.tokenizer import build_hf_or_tiktoken_tokenizer

from .umup_llama import UmupTransformer, UmupTransformerModelArgs
from .umup_loss import umup_nll_loss
from .umup_optimizer import create_umup_adamw

from ..datasets import build_dataloader
from ..lr_scheduler import build_lr_schedulers
from ..optimizer import build_optimizers
from ..metrics import build_metrics_processor


__all__ = [
    "UmupTransformer",
    "UmupTransformerModelArgs",
    "llumup3_configs",
    "create_umup_adamw",
    "build_dataloader",
]


llumup3_configs = {
    "debugmodel": UmupTransformerModelArgs(
        dim=256, n_layers=8, n_heads=16, rope_theta=500000
    ),
    "8B": UmupTransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": UmupTransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": UmupTransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}

register_train_spec(
    "llumup3",
    TrainSpec(
        model_cls=UmupTransformer,
        model_args=llumup3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_or_tiktoken_tokenizer,
        build_loss_fn=lambda cfg: umup_nll_loss,
        build_metrics_processor_fn=build_metrics_processor,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    ),
)
