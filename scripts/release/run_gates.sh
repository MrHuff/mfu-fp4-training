#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime_root=${FP4_RUNTIME_ROOT:-"$repo_root/fp4_runtime"}
mode=${1:-full}
cd "$repo_root"
export PYTHONPATH="$repo_root/torchtitan_submodule${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

if [[ "$mode" != "--cpu-only" && "$mode" != "full" ]]; then
  echo "usage: $0 [--cpu-only]" >&2
  exit 2
fi

python tools/check_public_references.py --root "$repo_root"

# Structural development gates may run before the final publication seal, but
# unsafe content and ledger mismatches always remain hard failures.
review_args=()
if [[ "$mode" == "--cpu-only" ]]; then
  if [[ -f "$repo_root/release/components.json" ]]; then
    review_args+=(--allow-publication-blockers)
  else
    review_args+=(--allow-staging)
  fi
fi
python tools/release_capsule.py doctor --phase source "${review_args[@]}"
python tools/release_capsule.py doctor \
  --phase runtime \
  --runtime-root "$runtime_root" \
  "${review_args[@]}"

python -m pytest -q -p no:cacheprovider \
  tests/cpu/test_mxfp4_mixed_localcta_dgrad.py \
  tests/cpu/test_distributed_control.py \
  tests/cpu/test_hsdp_accumulation.py

python -m pytest -q -p no:cacheprovider \
  "$runtime_root/TK_quantisation/nvfp4_CTA_local_v4/test_mixed_mx_localcta_source.py" \
  "$runtime_root/TK_quantisation/nvfp4_CTA_local_v4/test_mixed_split2_fp8x4_rescale_source.py" \
  "$runtime_root/TK_quantisation/mxfp4_v4/test_autograd_cuda_context_guards.py" \
  "$runtime_root/TK_quantisation/nvfp4_CTA_local_v4/test_atomic_paired_col_rht_source.py" \
  "$runtime_root/TK_quantisation/nvfp4_CTA_local_v4/test_persistent_counter_stream_scope_source.py"

if [[ "$mode" == "--cpu-only" ]]; then
  echo "CPU/source development gates passed; GPU and distributed resume gates were not run"
  exit 0
fi

# A full gate may only run from a sealed release candidate whose production
# extensions have already been built by scripts/release/build_kernels.sh.
python tools/release_capsule.py doctor --phase source
scripts/release/run_gpu_gates.sh

echo "source, ABI, and single-GPU numerical release gates passed"
