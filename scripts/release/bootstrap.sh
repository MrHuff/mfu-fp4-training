#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

verify_only=false
if [[ ${1:-} == "--verify-only" ]]; then
  verify_only=true
  shift
fi
if (($#)); then
  echo "usage: $0 [--verify-only]" >&2
  exit 2
fi

review_args=()
if [[ "$verify_only" == true && -f release/components.json ]]; then
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
  if [[ "$verify_only" == true ]]; then
    echo "vendored source and component ledgers verified"
    exit 0
  fi
else
  if [[ "$verify_only" == true ]]; then
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

python tools/release_capsule.py doctor \
  --phase runtime \
  --runtime-root "$repo_root/fp4_runtime"

echo "container-anchored bootstrap verification complete"
