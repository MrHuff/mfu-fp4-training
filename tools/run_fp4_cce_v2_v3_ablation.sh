#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model_size="${MODEL_SIZE:-1b}"
case "${model_size}" in
  1b|8b) ;;
  *)
    echo "MODEL_SIZE must be one of: 1b, 8b" >&2
    exit 2
    ;;
esac

config="${CONFIG:-train_configs/ablations/fp4_cce_slimpajama/${model_size}/fp4_cce_matrix.toml}"
variants="${VARIANTS:-mx-v2-enc,mx-v3-enc,mx-v4-enc,nv-v2-enc,nv-v3-enc,nv-v4-enc}"
steps="${STEPS:-50}"
label="${LABEL:-slimpajama_${model_size}_fp4_cce_v2_v3_v4}"

if [[ -z "${FP4_MATMUL_ROOT:-}" ]]; then
  for candidate in \
    "${repo_root}/../fp4_matmul" \
    "/opt/mfu/EXTERNAL_PATH" \
    "/opt/mfu/EXTERNAL_PATH" \
    "/tmp/fp4_matmul_v4_pcache"; do
    if [[ -d "${candidate}" ]]; then
      export FP4_MATMUL_ROOT="${candidate}"
      break
    fi
  done
  export FP4_MATMUL_ROOT="${FP4_MATMUL_ROOT:-${repo_root}/../fp4_matmul}"
fi

cd "${repo_root}"
exec python tools/run_fp4_cce_train_matrix.py \
  --config "${config}" \
  --variants "${variants}" \
  --steps "${steps}" \
  --label "${label}" \
  "$@"
