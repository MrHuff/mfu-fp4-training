#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/opt/mfu/EXTERNAL_PATH}
BACKEND_ROOT=${FP4_MATMUL_ROOT:-/opt/mfu/EXTERNAL_PATH}
CONFIG=${CONFIG:-train_configs/llama3_1B_fused_fp4_wiki_v4_nvfp4_cce_v4.toml}
GPU=${CUDA_VISIBLE_DEVICES:-3}
STEPS=${STEPS:-50}
LOG_FREQ=${LOG_FREQ:-10}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_BASE=${OUT_BASE:-/tmp/lbt_localcta_v4_rht_matrix_${STAMP}}

mkdir -p "${OUT_BASE}"
cd "${ROOT}"

run_mode() {
  local name=$1
  shift
  local out="${OUT_BASE}/${name}"
  mkdir -p "${out}"

  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export FP4_MATMUL_ROOT="${BACKEND_ROOT}"

    export USE_TK_GEMM=1
    export USE_TK_QUANT=1
    export USE_TK_LOCALCTA=1
    export USE_TK_LOCALCTA_VARIANT=v4
    export USE_TK_LOCALCTA_FUSED=0
    export USE_TK_LOCALCTA_V4_FFN_RESIDUAL_EPILOGUE="${USE_TK_LOCALCTA_V4_FFN_RESIDUAL_EPILOGUE:-1}"

    export NVFP4_USE_RHT=0
    export NVFP4_RHT_ACTIVATION=0
    export NVFP4_RHT_GRAD=0
    export NVFP4_RHT_WEIGHT=0
    export NVFP4_RHT_AXES=col
    export NVFP4_RHT_RANDOM_SIGNS=0

    export NVFP4_USE_STOCHASTIC_ROUNDING=0
    export NVFP4_SR_ACTIVATION=0
    export NVFP4_SR_GRAD=0
    export NVFP4_SR_WEIGHT=0

    export NVFP4_USE_SCALE_STOCHASTIC_ROUNDING=0
    export NVFP4_SCALE_SR_ACTIVATION=0
    export NVFP4_SCALE_SR_GRAD=0
    export NVFP4_SCALE_SR_WEIGHT=0

    for kv in "$@"; do
      export "${kv}"
    done

    {
      echo "mode=${name}"
      echo "out=${out}"
      git -C "${ROOT}" rev-parse --short HEAD
      git -C "${ROOT}" status --short
      echo "backend=${BACKEND_ROOT}"
      git -C "${BACKEND_ROOT}" rev-parse --short HEAD
      git -C "${BACKEND_ROOT}" status --short
      env | grep -E '^(CUDA_VISIBLE_DEVICES|FP4_MATMUL_ROOT|USE_TK|NVFP4)' | sort
      torchrun --standalone --nproc_per_node=1 train.py \
        --job.config_file "${CONFIG}" \
        --job.dump_folder "${out}/dump" \
        --training.steps "${STEPS}" \
        --metrics.log_freq "${LOG_FREQ}" \
        --metrics.disable-color-printing \
        --fp4-cce.backend nvfp4 \
        --fp4-cce.implementation v4 \
        --fp4-cce.quant-mode enc
    } 2>&1 | tee "${out}/train.log"
  )
}

run_mode "col_act_rht_no_sr" \
  NVFP4_USE_RHT=1 \
  NVFP4_RHT_ACTIVATION=1 \
  NVFP4_RHT_GRAD=0 \
  NVFP4_RHT_WEIGHT=0

run_mode "col_act_grad_rht_grad_sr" \
  NVFP4_USE_RHT=1 \
  NVFP4_RHT_ACTIVATION=1 \
  NVFP4_RHT_GRAD=1 \
  NVFP4_RHT_WEIGHT=0 \
  NVFP4_USE_STOCHASTIC_ROUNDING=1 \
  NVFP4_SR_ACTIVATION=0 \
  NVFP4_SR_GRAD=1 \
  NVFP4_SR_WEIGHT=0

run_mode "col_act_grad_rht_grad_scale_sr" \
  NVFP4_USE_RHT=1 \
  NVFP4_RHT_ACTIVATION=1 \
  NVFP4_RHT_GRAD=1 \
  NVFP4_RHT_WEIGHT=0 \
  NVFP4_USE_STOCHASTIC_ROUNDING=1 \
  NVFP4_SR_ACTIVATION=0 \
  NVFP4_SR_GRAD=1 \
  NVFP4_SR_WEIGHT=0 \
  NVFP4_USE_SCALE_STOCHASTIC_ROUNDING=1 \
  NVFP4_SCALE_SR_ACTIVATION=0 \
  NVFP4_SCALE_SR_GRAD=1 \
  NVFP4_SCALE_SR_WEIGHT=0

run_mode "col_act_rht_weight_sr" \
  NVFP4_USE_RHT=1 \
  NVFP4_RHT_ACTIVATION=1 \
  NVFP4_RHT_GRAD=0 \
  NVFP4_RHT_WEIGHT=0 \
  NVFP4_USE_STOCHASTIC_ROUNDING=1 \
  NVFP4_SR_ACTIVATION=0 \
  NVFP4_SR_GRAD=0 \
  NVFP4_SR_WEIGHT=1

python - "${OUT_BASE}" <<'PY'
import re
import statistics as st
import sys
from pathlib import Path

ansi = re.compile(r"\x1b\[[0-9;:]*m")
step_re = re.compile(r"\bstep:\s*(\d+)", re.I)
loss_re = re.compile(r"\bloss:\s*([0-9]+(?:\.[0-9]+)?)", re.I)
tps_re = re.compile(r"\btps:\s*([0-9,]+)", re.I)
mfu_re = re.compile(r"\bmfu:\s*([0-9]+(?:\.[0-9]+)?)%", re.I)

print("mode\tlogged_steps\tstep_range\tmean_mfu\tmedian_mfu\tpeak_mfu\tmean_tps\tfinal_loss\tlog")
for log in sorted(Path(sys.argv[1]).glob("*/train.log")):
    rows = []
    for raw in log.read_text(errors="ignore").splitlines():
        line = ansi.sub("", raw)
        ms, mm = step_re.search(line), mfu_re.search(line)
        if not (ms and mm):
            continue
        mt, ml = tps_re.search(line), loss_re.search(line)
        rows.append(
            (
                int(ms.group(1)),
                float(mm.group(1)),
                float(mt.group(1).replace(",", "")) if mt else None,
                float(ml.group(1)) if ml else None,
            )
        )
    rows = [row for row in rows if row[0] >= 2]
    if not rows:
        print(f"{log.parent.name}\t0\t-\t-\t-\t-\t-\t-\t{log}")
        continue
    mfus = [row[1] for row in rows]
    tps = [row[2] for row in rows if row[2] is not None]
    losses = [row[3] for row in rows if row[3] is not None]
    print(
        f"{log.parent.name}\t{len(rows)}\t{rows[0][0]}-{rows[-1][0]}\t"
        f"{st.mean(mfus):.3f}\t{st.median(mfus):.3f}\t{max(mfus):.3f}\t"
        f"{st.mean(tps):.0f}\t{losses[-1] if losses else '-'}\t{log}"
    )
PY
