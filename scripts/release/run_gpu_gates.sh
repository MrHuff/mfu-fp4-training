#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime_root=${FP4_RUNTIME_ROOT:-"$repo_root/fp4_runtime"}
cd "$repo_root"
export PYTHONDONTWRITEBYTECODE=1

if (($#)); then
  echo "usage: $0" >&2
  exit 2
fi

# This validates the software installed on the current host against the
# NGC-derived lock.  It does not assert that a container engine independently
# pulled the recorded image digest; that scope is explicit in the receipt.
python tools/verify_container_contract.py

# The ABI checker imports all six production extensions by exact filename and
# verifies the Python symbols required by the supported training routes.
python tools/check_fp4_runtime_abi.py --runtime-root "$runtime_root"

# Deterministic GEMM checks against BF16 cover the two global-scaling kernel
# families.  The localCTA producer and GEMM have their own 2D-weight contract
# below because their outer-scale ABI is deliberately different.
python tools/check_fp4_runtime_numerics.py --runtime-root "$runtime_root"

python "$runtime_root/TK_quantisation/mxfp4_v4/test_correlated_dual_sr.py"
python -m pytest -q -p no:cacheprovider \
  "$runtime_root/TK_quantisation/nvfp4_v5/test_fused_norm_barrier_free.py"

LOCALCTA_GEMM_MODULE_DIR="$runtime_root/ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3" \
LOCALCTA_GEMM_MODULE_NAME=_C_nv_localcta_gemm_v3 \
PYTHONPATH="$runtime_root/TK_quantisation/nvfp4_CTA_local_v4${PYTHONPATH:+:$PYTHONPATH}" \
python "$runtime_root/TK_quantisation/nvfp4_CTA_local_v4/test_weight_2d_common_outer_scale.py"

cat <<'EOF'
single-GPU production runtime gates passed on the recorded sm_100a software contract
note: this gate does not claim an independent container-digest pull or a distributed training replay
EOF
