#!/usr/bin/env bash
set -euo pipefail

# Focused final-layer loss benchmark for the NVIDIA-paper 1.2B shape:
# local batch 8, seq len 8192, hidden 2048, vocab 131072.
#
# This isolates the CCE layer so the result answers: how much faster is FP4 CCE
# v4 than a torch.compile BF16 CE final layer?

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
cd "${repo_root}"

timestamp=$(date +%Y%m%d_%H%M%S)
gpu="${GPU:-3}"
warmup="${WARMUP:-2}"
iters="${ITERS:-5}"
variants="${VARIANTS:-bf16-torch-compile,nv-v4-enc,mx-v4-enc}"
shape_set="${SHAPE_SET:-nvpaper_1p2b_final}"
timeout_sec="${ISOLATED_TIMEOUT_SEC:-1200}"
out_dir="${OUT_DIR:-/tmp/lbt_fp4_cce_final_layer_${timestamp}}"
mkdir -p "${out_dir}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export FP4_MATMUL_ROOT="${FP4_MATMUL_ROOT:-/opt/mfu/EXTERNAL_PATH}"
export FP4_MATMUL_GEMM_ROOT="${FP4_MATMUL_GEMM_ROOT:-/opt/mfu/EXTERNAL_PATH}"
export FP4_CCE_ASSUME_NONEMPTY_LABELS="${FP4_CCE_ASSUME_NONEMPTY_LABELS:-1}"
export FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE="${FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE:-1}"
export FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE="${FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE:-1}"

markdown_out="${out_dir}/fp4_cce_final_layer_${shape_set}.md"
log_out="${out_dir}/bench.log"

python tools/bench_fp4_cce_e2e.py \
  --shape-set "${shape_set}" \
  --variants "${variants}" \
  --warmup "${warmup}" \
  --iters "${iters}" \
  --isolated-variants \
  --isolated-timeout-sec "${timeout_sec}" \
  --markdown-out "${markdown_out}" \
  2>&1 | tee "${log_out}"

echo "Wrote log: ${log_out}"
echo "Wrote markdown: ${markdown_out}"
