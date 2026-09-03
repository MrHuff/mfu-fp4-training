#!/usr/bin/env bash
set -euo pipefail

# Builds a local pretokenized SlimPajama token pack from MosaicML S3 shards.
# Credentials may be exported already or placed in /tmp/lbt_bench_env.

MAX_TOKENS="${MAX_TOKENS:-67108864}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/tmp/lbt_packed/slimpajama_64m_tokens}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TOKENS_PER_SHARD="${TOKENS_PER_SHARD:-1073741824}"
CACHE_DIR="${CACHE_DIR:-/tmp/lbt_mosaic_cache/cerebras_slim_pajama_627b_small_shards}"
DATASET_PATH="${DATASET_PATH:-OBJECT_STORE_URI}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./torchtitan_submodule/tests/assets/tokenizer}"
PREDOWNLOAD="${PREDOWNLOAD:-16}"
SHUFFLE="${SHUFFLE:-false}"

out_args=(--output-prefix "${OUTPUT_PREFIX}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  out_args=(
    --output-dir "${OUTPUT_DIR}"
    --tokens-per-shard "${TOKENS_PER_SHARD}"
  )
fi

shuffle_args=()
if [[ "${SHUFFLE}" == "1" || "${SHUFFLE}" == "true" || "${SHUFFLE}" == "yes" ]]; then
  shuffle_args=(--shuffle)
fi

python tools/build_packed_binary_dataset.py \
  --source mosaic \
  --dataset mosaic/cerebras___slim_pajama-627_b \
  --dataset-path "${DATASET_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  "${out_args[@]}" \
  --max-tokens "${MAX_TOKENS}" \
  --predownload "${PREDOWNLOAD}" \
  "${shuffle_args[@]}" \
  "$@"
