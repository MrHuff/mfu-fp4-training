#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

mode=verify-container
case ${1:-} in
  "") ;;
  --verify-only)
    mode=verify-source
    shift
    ;;
  --install-vendored)
    mode=install-vendored
    shift
    ;;
  *)
    echo "usage: $0 [--verify-only|--install-vendored]" >&2
    exit 2
    ;;
esac
if (($#)); then
  echo "usage: $0 [--verify-only|--install-vendored]" >&2
  exit 2
fi

review_args=()
if [[ "$mode" == verify-source && -f release/components.json ]]; then
  review_args+=(--allow-publication-blockers)
fi
python tools/release_capsule.py doctor --phase source "${review_args[@]}"

if [[ -f release/components.json ]]; then
  # Clean public exports contain ordinary vendored source trees.  Their exact
  # upstream identities are verified through the content ledger; there is no
  # inherited submodule metadata to initialize.
  python tools/release_capsule.py doctor \
    --phase runtime \
    --runtime-root "$repo_root/fp4_runtime" \
    "${review_args[@]}"
  if [[ "$mode" == verify-source ]]; then
    echo "vendored source and component ledgers verified"
    exit 0
  fi
else
  if [[ "$mode" == verify-source ]]; then
    echo "error: --verify-only requires a flattened public export" >&2
    exit 2
  fi
fi

if [[ ! -f release/components.json ]]; then
  git submodule sync --recursive
  git submodule update --init --recursive
fi

# PyTorch and CUDA are baked into the digest-pinned NVIDIA image. Installing a
# nominally similar PyPI wheel would not reproduce that ABI. Verify the exact
# observed container contract instead of mutating it.
python tools/verify_container_contract.py

if [[ "$mode" == install-vendored ]]; then
  # The NGC base image already contains the locked compiler and Python build
  # stack.  Do not apt-install, curl-install, or resolve packages from PyPI:
  # those operations would create a second, mutable environment contract.
  for command in cmake gcc g++ make ninja; do
    command -v "$command" >/dev/null || {
      echo "error: required image-baked build tool is unavailable: $command" >&2
      exit 2
    }
  done
  test -x /usr/local/cuda/bin/nvcc || {
    echo "error: pinned CUDA compiler /usr/local/cuda/bin/nvcc is unavailable" >&2
    exit 2
  }
  python - <<'PY'
import importlib

for module in ("packaging", "pybind11", "setuptools", "torch", "wheel"):
    importlib.import_module(module)
PY

  te_wheel_dir=$(mktemp -d)
  cleanup() {
    rm -rf -- "$te_wheel_dir"
  }
  trap cleanup EXIT
  mkdir -p "$te_wheel_dir/source" "$te_wheel_dir/wheels"
  cp -a "$repo_root/TransformerEngine/." "$te_wheel_dir/source/"
  PIP_NO_INDEX=1 \
  NVTE_CUDA_ARCHS=100a \
  NVTE_FRAMEWORK=pytorch \
  NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
    python -m pip wheel \
      --no-build-isolation \
      --no-deps \
      --no-index \
      --wheel-dir "$te_wheel_dir/wheels" \
      "$te_wheel_dir/source"

  mapfile -t te_wheels < <(find "$te_wheel_dir/wheels" -maxdepth 1 -type f \
    -name 'transformer_engine-*.whl' -print)
  if ((${#te_wheels[@]} != 1)); then
    echo "error: vendored Transformer Engine build produced ${#te_wheels[@]} wheels" >&2
    exit 2
  fi
  PIP_NO_INDEX=1 python -m pip install \
    --force-reinstall \
    --no-deps \
    --no-index \
    "${te_wheels[0]}"
  cleanup
  trap - EXIT
fi

# Authenticate the installed Python distribution against the custom vendored
# source rather than merely accepting whichever Transformer Engine happened to
# be baked into the base image.
python - "$repo_root" <<'PY'
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import sys

root = Path(sys.argv[1])
relative = Path("transformer_engine/pytorch/custom_recipes/quantization_custom_format.py")
source = root / "TransformerEngine" / relative
distribution = metadata.distribution("transformer-engine")
installed = Path(distribution.locate_file(relative))
expected_version = (root / "TransformerEngine/build_tools/VERSION.txt").read_text().strip()

if distribution.version != expected_version:
    raise RuntimeError(
        "vendored Transformer Engine version mismatch: "
        f"expected {expected_version}, found {distribution.version}"
    )
if not installed.is_file():
    raise RuntimeError(f"installed custom Transformer Engine module is absent: {relative}")
if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(installed.read_bytes()).digest():
    raise RuntimeError("installed custom Transformer Engine source does not match the vendored file")
print(f"vendored Transformer Engine {distribution.version} verified")
PY

python tools/release_capsule.py doctor \
  --phase runtime \
  --runtime-root "$repo_root/fp4_runtime"

if [[ "$mode" == install-vendored ]]; then
  echo "container build bootstrap complete; attach an SM100 GPU before building FP4 runtime kernels"
else
  echo "container and vendored Transformer Engine verification complete"
fi
