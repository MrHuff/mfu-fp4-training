#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Default to the stable 1.2B MXFP4 high-water route. The current reproducible
# 500-step route keeps FFN residual fusion, disables attention residual fusion,
# and overlaps QKV/FFN wgrad with a wait before QKV RMSNorm backward. The older
# 99%+ 80-step mark was from an unsafe residual-overlap route and is kept behind
# MXFP4_HIGHWATER_UNSAFE=1 for debugging only.
ROOT=${ROOT:-${REPO_ROOT}}
BACKEND_ROOT=${FP4_MATMUL_ROOT:-/opt/mfu/EXTERNAL_PATH}
# Use the in-tree MXFP4 quant backend by default so benchmark runs pick up the
# latest kernel fixes. Set FP4_MXFP4_ROOT explicitly to pin an older build.
MXFP4_ROOT=${FP4_MXFP4_ROOT:-${BACKEND_ROOT}}
MXFP4_GEMM_ROOT=${FP4_MATMUL_GEMM_ROOT:-/opt/mfu/EXTERNAL_PATH}
CONFIG=${CONFIG:-train_configs/nvpaper_transformer_1p2b_mxfp4_tk_matrix.toml}
STEPS=${STEPS:-80}
NPROC_PER_NODE=${NPROC_PER_NODE:-${GPUS_PER_NODE:-1}}
NNODES=${NNODES:-${PET_NNODES:-${SLURM_JOB_NUM_NODES:-1}}}
NODE_RANK=${NODE_RANK:-${PET_NODE_RANK:-${SLURM_NODEID:-0}}}
MASTER_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-29500}}
RDZV_ID=${RDZV_ID:-mxfp4-highwater-${USER:-user}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU="${CUDA_VISIBLE_DEVICES}"
elif (( NPROC_PER_NODE > 1 )); then
  GPU="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
else
  GPU=0
fi
if [[ "${ALLOW_GPU2:-0}" != "1" ]]; then
  IFS=',' read -r -a _selected_gpus <<< "${GPU}"
  for _gpu in "${_selected_gpus[@]}"; do
    if [[ "${_gpu}" == "2" ]]; then
      echo "Refusing to run on blacklisted GPU2; set ALLOW_GPU2=1 only for explicit diagnostics." >&2
      exit 2
    fi
  done
  unset _gpu _selected_gpus
fi
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-/tmp/lbt_mxfp4_highwater_repro_${STAMP}}
LOG=${LOG:-${OUT_DIR}/train.log}

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE=${WANDB_MODE:-offline}
export FP4_MATMUL_ROOT="${BACKEND_ROOT}"
export FP4_MXFP4_ROOT="${MXFP4_ROOT}"
export FP4_MATMUL_GEMM_ROOT="${MXFP4_GEMM_ROOT}"
if [[ -z "${FP4_CCE_ASSUME_NONEMPTY_LABELS+x}" ]]; then
  if [[ "${MXFP4_HIGHWATER_FORCE_NO_RHT_SR:-1}" == "1" ]]; then
    export FP4_CCE_ASSUME_NONEMPTY_LABELS=0
  else
    export FP4_CCE_ASSUME_NONEMPTY_LABELS=1
  fi
fi

# Match the no-RHT/no-SR high-water route by default. Set
# MXFP4_HIGHWATER_FORCE_NO_RHT_SR=0 to reuse the same launch shell for RHT/SR
# screening without drifting the rest of the route.
if [[ "${MXFP4_HIGHWATER_FORCE_NO_RHT_SR:-1}" == "1" ]]; then
  export MXFP4_USE_RHT=0
  export MXFP4_RHT_ACTIVATION=0
  export MXFP4_RHT_GRAD=0
  export MXFP4_RHT_WEIGHT=0
  export MXFP4_RHT_RANDOM_SIGN_MASK=0
  export MXFP4_USE_STOCHASTIC_ROUNDING=0
  export MXFP4_SR_ACTIVATION=0
  export MXFP4_SR_GRAD=0
  export MXFP4_SR_WEIGHT=0
  export MXFP4_USE_SCALE_STOCHASTIC_ROUNDING=0
  export MXFP4_SCALE_SR_ACTIVATION=0
  export MXFP4_SCALE_SR_GRAD=0
  export MXFP4_SCALE_SR_WEIGHT=0
else
  # Semantic RHT/SR route: apply col-axis activation RHT and stochastic
  # rounding to gradient quantization.  The faster weight-SR screen can still
  # be reproduced with MXFP4_SR_GRAD=0 MXFP4_SR_WEIGHT=1.
  export MXFP4_USE_RHT=${MXFP4_USE_RHT:-1}
  export MXFP4_RHT_ACTIVATION=${MXFP4_RHT_ACTIVATION:-1}
  export MXFP4_RHT_GRAD=${MXFP4_RHT_GRAD:-1}
  export MXFP4_RHT_WEIGHT=${MXFP4_RHT_WEIGHT:-0}
  export MXFP4_RHT_TE_STYLE=${MXFP4_RHT_TE_STYLE:-1}
  export MXFP4_RHT_RANDOM_SIGN_MASK=${MXFP4_RHT_RANDOM_SIGN_MASK:-0}
  export MXFP4_USE_STOCHASTIC_ROUNDING=${MXFP4_USE_STOCHASTIC_ROUNDING:-1}
  export MXFP4_SR_ACTIVATION=${MXFP4_SR_ACTIVATION:-0}
  export MXFP4_SR_GRAD=${MXFP4_SR_GRAD:-1}
  export MXFP4_SR_WEIGHT=${MXFP4_SR_WEIGHT:-0}
  export MXFP4_USE_SCALE_STOCHASTIC_ROUNDING=${MXFP4_USE_SCALE_STOCHASTIC_ROUNDING:-0}
  export MXFP4_SCALE_SR_ACTIVATION=${MXFP4_SCALE_SR_ACTIVATION:-0}
  export MXFP4_SCALE_SR_GRAD=${MXFP4_SCALE_SR_GRAD:-0}
  export MXFP4_SCALE_SR_WEIGHT=${MXFP4_SCALE_SR_WEIGHT:-0}
fi

# High-water MXFP4 trainer path knobs.
export MXFP4_BACKEND_VERSION=${MXFP4_BACKEND_VERSION:-v4}
export MXFP4_USE_QKV_ROPE_EPILOGUE=${MXFP4_USE_QKV_ROPE_EPILOGUE:-1}
export MXFP4_USE_QKV_DIRECT_OUTPUTS=${MXFP4_USE_QKV_DIRECT_OUTPUTS:-1}
if [[ "${MXFP4_HIGHWATER_FORCE_NO_RHT_SR:-1}" == "1" ]]; then
  export MXFP4_USE_QKV_RMSNORM_QUANT_FUSION=${MXFP4_USE_QKV_RMSNORM_QUANT_FUSION:-1}
else
  export MXFP4_USE_QKV_RMSNORM_QUANT_FUSION=${MXFP4_USE_QKV_RMSNORM_QUANT_FUSION:-1}
fi
export MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD=${MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD:-0}
export MXFP4_QKV_BWD_STATE_SLOTS=${MXFP4_QKV_BWD_STATE_SLOTS:-4}
export MXFP4_USE_QKV_BF16_WGRAD=${MXFP4_USE_QKV_BF16_WGRAD:-0}
# Keep qkv wgrad overlap, but wait before qkv rmsnorm backward. Without this
# wait the high-water route can trip a delayed CUDA launch failure in long runs.
export MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM=${MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM:-1}
export MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM_DGAMMA=${MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM_DGAMMA:-1}
export MXFP4_USE_QKV_FWD_WEIGHT_QUANT_OVERLAP=${MXFP4_USE_QKV_FWD_WEIGHT_QUANT_OVERLAP:-0}
export MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD=${MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD:-1}
export MXFP4_USE_SPLIT2_FFN_INPLACE_QUANT=${MXFP4_USE_SPLIT2_FFN_INPLACE_QUANT:-1}
export MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP=${MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP:-0}
export MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT=${MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT:-1}
export MXFP4_USE_SPLIT2_FFN_PRODUCER_SPLIT=${MXFP4_USE_SPLIT2_FFN_PRODUCER_SPLIT:-0}
# Keep global backward/wgrad overlap opt-in. On high-clock GB200 runs this path
# can trip delayed CUDA launch failures, while the no-overlap route keeps the
# same 93%+ packed-C4 MFU band with the QKV RMSNorm waits enabled.
if [[ -z "${MXFP4_USE_BWD_WGRAD_OVERLAP+x}" ]]; then
  export MXFP4_USE_BWD_WGRAD_OVERLAP=0
fi
export MXFP4_USE_BWD_STATE_CACHE=${MXFP4_USE_BWD_STATE_CACHE:-0}
export MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP=${MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP:--1}
export MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP_M4096_N2048=${MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP_M4096_N2048:-199}
export MXFP4_USE_RESIDUAL_FUSION=${MXFP4_USE_RESIDUAL_FUSION:-1}
export MXFP4_USE_RESIDUAL_FUSION_FFN=${MXFP4_USE_RESIDUAL_FUSION_FFN:-1}
export MXFP4_UNSAFE_RESIDUAL_FALLBACK=${MXFP4_UNSAFE_RESIDUAL_FALLBACK:-prefer_ffn}
export MXFP4_USE_GEMM_RESIDUAL_KERNEL=${MXFP4_USE_GEMM_RESIDUAL_KERNEL:-1}
if [[ -z "${FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED+x}" ]]; then
  if [[ "${MXFP4_USE_BWD_WGRAD_OVERLAP:-1}" == "0" ]]; then
    export FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED=1
  else
    export FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED=0
  fi
fi
export FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE=${FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE:-1}
export FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE=${FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE:-1}
if [[ -z "${USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER+x}" && -n "${USE_TK_LOCALCTA_V4_RMSNORM_FINAL_SG_PRODUCER+x}" ]]; then
  export USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER="${USE_TK_LOCALCTA_V4_RMSNORM_FINAL_SG_PRODUCER}"
fi
export USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER=${USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER:-1}
if [[ -z "${MXFP4_RHT_AXES+x}" ]]; then
  if [[ "${MXFP4_RHT_TE_STYLE:-0}" == "1" ]]; then
    export MXFP4_RHT_AXES=col
  else
    export MXFP4_RHT_AXES=row
  fi
fi
export MXFP4_USE_FUSED_RMSNORM_QUANT_RHT=${MXFP4_USE_FUSED_RMSNORM_QUANT_RHT:-1}
if [[ "${MXFP4_HIGHWATER_FORCE_NO_RHT_SR:-1}" == "1" ]]; then
  export MXFP4_USE_FUSED_SILU_FFN_QUANT=${MXFP4_USE_FUSED_SILU_FFN_QUANT:-0}
else
  export MXFP4_USE_FUSED_SILU_FFN_QUANT=${MXFP4_USE_FUSED_SILU_FFN_QUANT:-1}
fi
export MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT=${MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT:-0}
export MXFP4_USE_FUSED_SQRELU_QUANT=${MXFP4_USE_FUSED_SQRELU_QUANT:-1}
export MXFP4_USE_TMA_SQRELU_QUANT=${MXFP4_USE_TMA_SQRELU_QUANT:-1}
if [[ "${MXFP4_HIGHWATER_FORCE_NO_RHT_SR:-1}" == "1" ]]; then
  export MXFP4_USE_FUSED_SQRELU_DERIV_QUANT=${MXFP4_USE_FUSED_SQRELU_DERIV_QUANT:-0}
else
  # The col-RHT+grad-SR fused derivative producer recovered a small amount of
  # MFU and passed the 1.2B 500-step RHT/SR screen on GPU3. Keep it disableable
  # for quick bisects.
  export MXFP4_USE_FUSED_SQRELU_DERIV_QUANT=${MXFP4_USE_FUSED_SQRELU_DERIV_QUANT:-1}
  export MXFP4_USE_SQRELU_DERIV_RHT_SR=${MXFP4_USE_SQRELU_DERIV_RHT_SR:-1}
fi
export MXFP4_USE_SQRELU_FUSED_RMS_W1=${MXFP4_USE_SQRELU_FUSED_RMS_W1:-0}
export MXFP4_USE_SIMPLE_SQRELU_FUSED_W2=${MXFP4_USE_SIMPLE_SQRELU_FUSED_W2:-1}
export MXFP4_USE_SQRELU_SPLIT_COL_OVERLAP=${MXFP4_USE_SQRELU_SPLIT_COL_OVERLAP:-0}
export MXFP4_USE_SQRELU_SPLIT_COL_QUANT=${MXFP4_USE_SQRELU_SPLIT_COL_QUANT:-0}
export MXFP4_USE_SQRELU_SPLIT_COL_WAIT_FORWARD=${MXFP4_USE_SQRELU_SPLIT_COL_WAIT_FORWARD:-1}
export MXFP4_USE_SQRELU_W2_WGRAD_OVERLAP=${MXFP4_USE_SQRELU_W2_WGRAD_OVERLAP:-0}
export MXFP4_USE_SQRELU_W2_WGRAD_AFTER_DGRAD_OVERLAP=${MXFP4_USE_SQRELU_W2_WGRAD_AFTER_DGRAD_OVERLAP:-0}
export MXFP4_USE_SQRELU_DERIV_GEMM_EPILOGUE=${MXFP4_USE_SQRELU_DERIV_GEMM_EPILOGUE:-0}
export MXFP4_USE_WO_ATTN_LAYOUT=${MXFP4_USE_WO_ATTN_LAYOUT:-0}
# Fusing both attention and FFN residual epilogues can race the backward overlap
# path under the default multi-connection scheduler. The stable high-water route
# keeps FFN residual fusion and leaves the old combined route behind an explicit
# opt-in for repro/debugging.
if [[ "${MXFP4_HIGHWATER_UNSAFE:-0}" == "1" ]]; then
  export MXFP4_USE_RESIDUAL_FUSION_ATTN=${MXFP4_USE_RESIDUAL_FUSION_ATTN:-1}
  export MXFP4_ALLOW_UNSAFE_ATTN_FFN_RESIDUAL_OVERLAP=1
else
  export MXFP4_USE_RESIDUAL_FUSION_ATTN=${MXFP4_USE_RESIDUAL_FUSION_ATTN:-0}
  export MXFP4_ALLOW_UNSAFE_ATTN_FFN_RESIDUAL_OVERLAP=0
fi

# The 91.94% log did not require connection pinning. Keep it unset unless the
# caller explicitly asks for the CDM=1 experiment.
if [[ "${MXFP4_HIGHWATER_CDM1:-0}" != "1" ]]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS || true
else
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi

{
  echo "repo=${ROOT}"
  git -C "${ROOT}" rev-parse --short HEAD
  git -C "${ROOT}" status --short
  echo "backend=${BACKEND_ROOT}"
  git -C "${BACKEND_ROOT}" rev-parse --short HEAD
  git -C "${BACKEND_ROOT}" status --short
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "NNODES=${NNODES}"
  echo "NODE_RANK=${NODE_RANK}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "RDZV_ID=${RDZV_ID}"
  echo "FP4_MATMUL_ROOT=${FP4_MATMUL_ROOT}"
  echo "FP4_MXFP4_ROOT=${FP4_MXFP4_ROOT}"
  echo "FP4_MATMUL_GEMM_ROOT=${FP4_MATMUL_GEMM_ROOT}"
  if [[ -d "${FP4_MATMUL_GEMM_ROOT}/.git" ]]; then
    echo "gemm_backend=${FP4_MATMUL_GEMM_ROOT}"
    git -C "${FP4_MATMUL_GEMM_ROOT}" rev-parse --short HEAD
    git -C "${FP4_MATMUL_GEMM_ROOT}" status --short
    if [[ -d "${FP4_MATMUL_GEMM_ROOT}/ThunderKittens/.git" ]]; then
      git -C "${FP4_MATMUL_GEMM_ROOT}/ThunderKittens" rev-parse --short HEAD
    fi
  fi
  echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}"
  timeout 5 nvidia-smi || true
  timeout 5 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv || true
  env | grep -E '^(MXFP4|FP4_CCE|FP4_MATMUL_ROOT|FP4_MXFP4_ROOT|FP4_MATMUL_GEMM_ROOT|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_MAX_CONNECTIONS|USE_TK)' | sort
  echo "log=${LOG}"
  echo "dump=${OUT_DIR}/dump"

  torchrun_args=(--nproc_per_node "${NPROC_PER_NODE}")
  if (( NNODES > 1 )); then
    torchrun_args+=(
      --nnodes "${NNODES}"
      --node_rank "${NODE_RANK}"
      --rdzv_id "${RDZV_ID}"
      --rdzv_backend c10d
      --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
      --rdzv-conf timeout=3600
    )
  else
    torchrun_args+=(--standalone)
  fi

  extra_train_args=()
  if (( NNODES * NPROC_PER_NODE > 1 )) && [[ "${FP4_MULTI_KEEP_CONFIG_GLOBAL_BATCH:-0}" != "1" ]]; then
    extra_train_args+=(--training.global-batch-size -1)
  fi

  torchrun "${torchrun_args[@]}" train.py \
    --job.config_file "${CONFIG}" \
    --job.dump_folder "${OUT_DIR}/dump" \
    --training.steps "${STEPS}" \
    --metrics.disable-color-printing \
    --fp4-cce.backend "${FP4_CCE_BACKEND:-nvfp4}" \
    --fp4-cce.implementation "${FP4_CCE_IMPLEMENTATION:-v4}" \
    --fp4-cce.quant-mode "${FP4_CCE_QUANT_MODE:-enc}" \
    "${extra_train_args[@]}" \
    "$@"
} 2>&1 | tee "${LOG}"
