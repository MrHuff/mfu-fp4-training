#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime_root=${FP4_RUNTIME_ROOT:-"$repo_root/fp4_runtime"}
cd "$repo_root"

python tools/release_capsule.py doctor --phase source
python tools/release_capsule.py doctor \
  --phase runtime \
  --runtime-root "$runtime_root"

test -x /usr/local/cuda/bin/nvcc || {
  echo "error: pinned CUDA compiler /usr/local/cuda/bin/nvcc is unavailable" >&2
  exit 2
}
python -c 'import pybind11, torch; assert torch.version.cuda == "13.0"'

make -C "$runtime_root/TK_quantisation/mxfp4_v4" all
make -C "$runtime_root/TK_quantisation/nvfp4_CTA_local_v4" all
make -C "$runtime_root/TK_quantisation/nvfp4_v5" all
make -C "$runtime_root/ThunderKittens/kernels/gemm/mxfp4_gb200" GPU=B200 all
make -C "$runtime_root/ThunderKittens/kernels/gemm/nvfp4_b200" GPU=B200 all
make -C "$runtime_root/ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3" GPU=B200 all

python tools/check_fp4_runtime_abi.py --runtime-root "$runtime_root"

echo "production FP4 quantizers and GEMMs built and ABI-checked from pinned sources"
echo "run scripts/release/run_gpu_gates.sh for the numerical and behavior gates"
