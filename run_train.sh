#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e
# use envs as local overrides for convenience
# export CUDA_LAUNCH_BLOCKING=1
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
NGPU=${NGPU:-"4"}
LOG_RANK=${LOG_RANK:-0}
#CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_1B_e5m3_bf16_simulation_encode_neg_wikitext.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_1B_nvfp4_te_wikitext.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_1B_fused_fp4_wiki.toml"}
# Example configuration path is relative to the repository root.
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi
# W&B credentials, when needed, must be supplied by the runtime environment.
# Generate a group ID
export WANDB_NAME="${WANDB_NAME:-$(python scripts/random_name.py)}"
export WANDB_GROUP_ID="${WANDB_GROUP_ID:-${WANDB_NAME}}"
# AWS credentials must come from the runtime default credential chain.

export NVTE_FUSED_ATTN=0
export TORCH_CUDNN_SDPA_ENABLED=1
export LOW_BITS_DISABLE_ATEN_FLASH_PATCH=1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export TORCH_LOGS="recompiles"
# Hugging Face credentials, when needed, must be supplied by the runtime.
# Llama 3.1 tokenizer
export NVTE_NVFP4_DISABLE_RHT=1
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1
export NVTE_NVFP4_ENCODE_CENTRIC=0
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
export NVTE_CUSTOM_QUANT=1
export USE_TK_GEMM=0
export FUSED_TE_QUANT=0
# Checkpoint creation, resumption, and retention are controlled by the selected
# TorchTitan configuration and command-line overrides. This generic launcher
# never deletes a checkpoint directory.
# export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} ${overrides}
