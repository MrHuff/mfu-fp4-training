#!/usr/bin/env bash
set -euo pipefail

# Build a no-repeat SlimPajama packed-token corpus large enough for multi-node
# FP4 convergence runs. Public Mosaic S3 shards are read unsigned when AWS
# credentials are absent.

if [[ -f /tmp/lbt_bench_env ]]; then
  set -a
  # shellcheck source=/dev/null
  source /tmp/lbt_bench_env
  set +a
fi

# 20B tokens covers 500 steps at seq=8192, local_batch=8, world_size=512 with
# about 20% headroom. Override for larger runs.
export MAX_TOKENS="${MAX_TOKENS:-20132659200}"
export OUTPUT_DIR="${OUTPUT_DIR:-/local_nvme/lbt_packed/slimpajama_20b_tokens}"
export TOKENS_PER_SHARD="${TOKENS_PER_SHARD:-1073741824}"
export CACHE_DIR="${CACHE_DIR:-/local_nvme/lbt_mosaic_cache/cerebras_slim_pajama_627b_small_shards}"
export PREDOWNLOAD="${PREDOWNLOAD:-64}"
# Keep this false for the fast local-MDS builder path. The source shards are
# consumed without replacement, and PackedBinaryDataset enforces no-repeat
# capacity at training startup.
export SHUFFLE="${SHUFFLE:-false}"

tools/build_slimpajama_packed_binary.sh --overwrite "$@"
