#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from .models import add_model_config

from torchtitan.models.llama3 import (  # noqa: F401
    TransformerModelArgs,
    Transformer,
    llama3_args,
)
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.models.llama3 import parallelize_llama
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.llama3 import Llama3StateDictAdapter
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.validate import build_validator

from low_bits_training.components.tokenizer import build_hf_or_tiktoken_tokenizer

from ..optimizer import build_optimizers
from ..lr_scheduler import build_lr_schedulers
from ..datasets import build_dataloader
from ..metrics import build_metrics_processor

from dataclasses import replace

# Copy the TorchTitan collection of config (to avoid directly mutating it).
llama3_gc_configs = {name: replace(cfg) for name, cfg in llama3_args.items()}

register_train_spec(
    "llama3_gc",
    TrainSpec(
        model_cls=Transformer,
        model_args=llama3_gc_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_or_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_metrics_processor_fn=build_metrics_processor,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    ),
)
# Note - for new configs
# The ffn_multiplier is not a straight multiplier at the time of writing the equations below
# are applied to the "dim" parameter - check the implementation before creating a config.
#
# dim = 2048
# ffn_multiplier = 3/2
# multiple_of = 256
#
# post_ffn = int(ffn_multiplier * int(2 * (4 * dim) / 3))
# final = multiple_of * ((post_ffn + multiple_of - 1) // multiple_of)
# print(f"{post_ffn=}, {final=}")
# If you know the ffn intermediate size you want express it as
# ffn_dim / 4 / dim * 3 / 2

# Adding small Llama3 models
# https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
add_model_config(
    "llama3_gc",
    "debugmodel_old",
    TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2004, rope_theta=500000
    ),
)

add_model_config(
    "llama3_gc",
    "1B",
    TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=8192 / 4 / 2048 * 3 / 2,
        multiple_of=256,
        rope_theta=500000,
    ),  # TODO: share embedding; d_head=64 (default=128) -> 1235746816 params
)

add_model_config(
    "llama3_gc",
    "8B_llama3_blog",
    TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        rope_scaling_args=None,
        max_seq_len=8192,
    ),
)

# Proxy for the 8B model in "Pretraining LLMs with NVFP4".
#
# The paper model is a 52-block Nemotron-H hybrid Mamba-Transformer
# (4 attention, 24 FFN, 24 Mamba-2 blocks), which this Llama3-only codepath
# cannot represent exactly. This proxy keeps the Llama/TorchTitan transformer
# implementation but matches the paper's hidden size, FFN size, Q heads, KV
# heads, sequence length, and 24 FFN-block count while staying near 8B params.
add_model_config(
    "llama3_gc",
    "8B_nvfp4_paper_proxy",
    TransformerModelArgs(
        dim=4096,
        n_layers=24,
        n_heads=32,
        n_kv_heads=4,
        ffn_dim_multiplier=21504 / 4 / 4096 * 3 / 2,
        multiple_of=1024,
        rope_theta=500000,
        rope_scaling_args=None,
        max_seq_len=8192,
    ),
)
# https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
add_model_config(
    "llama3_gc",
    "3B",
    TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=8192 / 4 / 3072 * 3 / 2,  # ffn_dim = 8192
        multiple_of=256,
        rope_theta=500000,
    ),  # TODO: share embedding -> 3212574720 params
)

add_model_config(
    "llama3_gc",
    "unit_test",
    TransformerModelArgs(
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim_multiplier=1,
        multiple_of=8,
        rope_theta=32,
        vocab_size=300,
    ),
)
# Deprecated configs which had the wrong numbers of parameters
add_model_config(
    "llama3_gc",
    "fake-1B",
    TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=4,
        multiple_of=256,
        rope_theta=500000,
    ),  # TODO: share embedding; d_head=64 (default=128) -> 1235746816 params
)
# https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
add_model_config(
    "llama3_gc",
    "fake-3B",
    TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=2.66,  # ffn_dim = 8192
        multiple_of=256,
        rope_theta=500000,
    ),  # TODO: share embedding -> 3212574720 params
)

add_model_config(
    "llama3_gc",
    "1B_legacy",
    TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=32,
        ffn_dim_multiplier=1.0, 
        multiple_of=256,
        rope_theta=10000, 
        vocab_size=32000, 
    ),
)
