#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
STEPS="${STEPS:-20}"
LOG_FREQ="${LOG_FREQ:-5}"
SEED="${SEED:-1234}"
CASE="${CASE:-bf16}"
DATA_BIN="${DATA_BIN:-/tmp/lbt_packed/slimpajama_64m_tokens.bin}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PIN_MEMORY="${PIN_MEMORY:-false}"
REPEAT="${REPEAT:-false}"
REQUIRE_FULL_RUN="${REQUIRE_FULL_RUN:-true}"
MAX_OPEN_SHARDS="${MAX_OPEN_SHARDS:-8}"
OUT_DIR="${OUT_DIR:-/tmp/lbt_packed_binary_${CASE}_${STEPS}_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "${DATA_BIN}" && ! -d "${DATA_BIN}" ]]; then
  echo "packed token file/directory not found: ${DATA_BIN}" >&2
  echo "Build it with tools/build_packed_binary_dataset.py first." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

LOAD_KWARGS=$(python - <<PY
import json
print(json.dumps({
    "num_workers": int("${NUM_WORKERS}"),
    "prefetch_factor": int("${PREFETCH_FACTOR}"),
    "pin_memory": "${PIN_MEMORY}".lower() == "true",
    "repeat": "${REPEAT}".lower() == "true",
    "require_full_run": "${REQUIRE_FULL_RUN}".lower() == "true",
    "max_open_shards": int("${MAX_OPEN_SHARDS}"),
}, separators=(",", ":")))
PY
)

COMMON_ARGS=(
  --training.dataset packed-bin
  --training.dataset-path "${DATA_BIN}"
  --training.load-dataset-kwargs "${LOAD_KWARGS}"
  --metrics.log-freq "${LOG_FREQ}"
  --metrics.disable-color-printing
  --debug.seed "${SEED}"
)

export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "Packed binary screen"
echo "  case: ${CASE}"
echo "  gpu: ${GPU}"
echo "  steps: ${STEPS}"
echo "  data: ${DATA_BIN}"
echo "  workers: ${NUM_WORKERS}"
echo "  prefetch_factor: ${PREFETCH_FACTOR}"
echo "  pin_memory: ${PIN_MEMORY}"
echo "  repeat: ${REPEAT}"
echo "  require_full_run: ${REQUIRE_FULL_RUN}"
echo "  out dir: ${OUT_DIR}"

if [[ "${CASE}" == "bf16" ]]; then
  python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    --job.config-file train_configs/nvpaper_transformer_1p2b_bf16_matrix.toml \
    --job.dump-folder "${OUT_DIR}/dump" \
    --training.steps "${STEPS}" \
    "${COMMON_ARGS[@]}" \
    --fp4-cce.enabled --fp4-cce.backend triton_bf16
elif [[ "${CASE}" == "mxfp4" ]]; then
  OUT_DIR="${OUT_DIR}" \
  STEPS="${STEPS}" \
  tools/run_mxfp4_highwater_repro.sh \
    "${COMMON_ARGS[@]}"
else
  echo "unknown CASE=${CASE}; supported: bf16, mxfp4" >&2
  exit 2
fi
