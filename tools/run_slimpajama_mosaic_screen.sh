#!/usr/bin/env bash
set -euo pipefail

# Credential-safe SlimPajama MosaicML smoke/training screen.
#
# Expected credentials:
#   - already exported in the shell, or
#   - stored in /tmp/lbt_bench_env as export lines.
#
# The script deliberately avoids `set -x` and does not print credential values.

if [[ -f /tmp/lbt_bench_env ]]; then
  set -a
  # shellcheck source=/dev/null
  source /tmp/lbt_bench_env
  set +a
fi

missing=0
for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION AWS_REGION; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    missing=1
  fi
done
if [[ "${missing}" == "1" ]]; then
  echo "Create /tmp/lbt_bench_env or export AWS credentials in the parent shell, then rerun." >&2
  exit 2
fi

GPU="${GPU:-3}"
STEPS="${STEPS:-20}"
LOG_FREQ="${LOG_FREQ:-5}"
SEED="${SEED:-1234}"
OUT_DIR="${OUT_DIR:-/tmp/lbt_slimpajama_mosaic_screen_${STEPS}_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"

DATASET_NAME="${DATASET_NAME:-mosaic/cerebras___slim_pajama-627_b}"
DATASET_PATH="${DATASET_PATH:-OBJECT_STORE_URI}"
CACHE_DIR="${CACHE_DIR:-/tmp/lbt_mosaic_cache/cerebras_slim_pajama_627b_small_shards}"
MOSAIC_NUM_WORKERS="${MOSAIC_NUM_WORKERS:-8}"
MOSAIC_SHUFFLE="${MOSAIC_SHUFFLE:-false}"
MOSAIC_PREDOWNLOAD="${MOSAIC_PREDOWNLOAD:-8}"
MOSAIC_CACHE_LIMIT="${MOSAIC_CACHE_LIMIT:-}"
MOSAIC_PREFETCH_FACTOR="${MOSAIC_PREFETCH_FACTOR:-4}"
MOSAIC_PIN_MEMORY="${MOSAIC_PIN_MEMORY:-false}"
CASE="${CASE:-bf16}"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

DATASET_KWARGS=$(python - <<PY
import json
kwargs = {
    "split": "train",
    "shuffle": "${MOSAIC_SHUFFLE}".lower() == "true",
    "num_workers": int("${MOSAIC_NUM_WORKERS}"),
    "cache_dir": "${CACHE_DIR}",
    "predownload": int("${MOSAIC_PREDOWNLOAD}"),
    "prefetch_factor": int("${MOSAIC_PREFETCH_FACTOR}"),
    "pin_memory": "${MOSAIC_PIN_MEMORY}".lower() == "true",
}
cache_limit = "${MOSAIC_CACHE_LIMIT}"
if cache_limit:
    kwargs["cache_limit"] = cache_limit
print(json.dumps(kwargs, separators=(",", ":")))
PY
)

echo "SlimPajama Mosaic screen"
echo "  case: ${CASE}"
echo "  gpu: ${GPU}"
echo "  steps: ${STEPS}"
echo "  dataset: ${DATASET_NAME}"
echo "  dataset path: ${DATASET_PATH}"
echo "  cache dir: ${CACHE_DIR}"
echo "  out dir: ${OUT_DIR}"

echo "Running Mosaic dataset smoke..."
python - <<PY
from streaming import Stream, StreamingDataset
import itertools
import json
import os
import time

dataset_path = "${DATASET_PATH}"
cache_dir = "${CACHE_DIR}"
kwargs = json.loads('${DATASET_KWARGS}')
split = kwargs.pop("split")
num_workers = kwargs.pop("num_workers", None)
cache_dir = kwargs.pop("cache_dir", cache_dir)
prefetch_factor = kwargs.pop("prefetch_factor", None)
pin_memory = kwargs.pop("pin_memory", None)
print(
    f"  smoke split={split} cache_dir={cache_dir} num_workers={num_workers} "
    f"prefetch_factor={prefetch_factor} pin_memory={pin_memory} kwargs={kwargs}",
    flush=True,
)
t0 = time.perf_counter()
ds = StreamingDataset(
    streams=[Stream(remote=dataset_path, local=cache_dir, split=split)],
    batch_size=8,
    **kwargs,
)
print(f"  loaded len={len(ds):,} init_s={time.perf_counter() - t0:.3f}", flush=True)
t1 = time.perf_counter()
n = 0
chars = 0
for sample in itertools.islice(ds, 32):
    n += 1
    text = sample.get("text", "")
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    chars += len(text)
print(f"  iter_n={n} chars={chars} iter_s={time.perf_counter() - t1:.3f}", flush=True)
PY

COMMON_ARGS=(
  --job.dump-folder "${OUT_DIR}/dump"
  --training.steps "${STEPS}"
  --metrics.log-freq "${LOG_FREQ}"
  --metrics.disable-color-printing
  --training.dataset "${DATASET_NAME}"
  --training.dataset-path "${DATASET_PATH}"
  --training.load-dataset-kwargs "${DATASET_KWARGS}"
  --debug.seed "${SEED}"
)

export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ "${CASE}" == "bf16" ]]; then
  python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --job.config-file train_configs/nvpaper_transformer_1p2b_bf16_matrix.toml \
    "${COMMON_ARGS[@]}" \
    --fp4-cce.enabled --fp4-cce.backend triton_bf16
elif [[ "${CASE}" == "mxfp4" ]]; then
  FP4_MATMUL_ROOT="${FP4_MATMUL_ROOT:-/opt/mfu/EXTERNAL_PATH}" \
  FP4_MXFP4_ROOT="${FP4_MXFP4_ROOT:-/opt/mfu/EXTERNAL_PATH}" \
  FP4_MATMUL_GEMM_ROOT="${FP4_MATMUL_GEMM_ROOT:-/opt/mfu/EXTERNAL_PATH}" \
  tools/run_mxfp4_highwater_repro.sh \
    --job.dump-folder "${OUT_DIR}/dump" \
    --training.steps "${STEPS}" \
    --metrics.log-freq "${LOG_FREQ}" \
    --training.dataset "${DATASET_NAME}" \
    --training.dataset-path "${DATASET_PATH}" \
    --training.load-dataset-kwargs "${DATASET_KWARGS}" \
    --debug.seed "${SEED}"
else
  echo "unknown CASE=${CASE}; supported: bf16, mxfp4" >&2
  exit 2
fi
