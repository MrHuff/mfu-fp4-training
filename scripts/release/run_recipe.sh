#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
route=""
inputs=""
runtime_root="${repo_root}/fp4_runtime"
topology=()

while (($#)); do
  case "$1" in
    --route) route=${2:?}; shift 2 ;;
    --inputs) inputs=${2:?}; shift 2 ;;
    --runtime-root) runtime_root=${2:?}; shift 2 ;;
    --nnodes|--nproc-per-node|--node-rank|--master-addr|--master-port)
      topology+=("$1" "${2:?}"); shift 2 ;;
    --execute) topology+=("$1"); shift ;;
    --wandb-mode) topology+=("$1" "${2:?}"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$route" && -n "$inputs" ]] || {
  echo "usage: $0 --route ROUTE --inputs LOCAL_JSON --nnodes N --nproc-per-node N --node-rank N --master-addr HOST [--execute]" >&2
  exit 2
}

cd "$repo_root"
python tools/release_capsule.py doctor \
  --phase all \
  --runtime-root "$runtime_root" \
  --inputs "$inputs" \
  --route "$route"
exec python tools/paper_recipe.py \
  --route "$route" --inputs "$inputs" --runtime-root "$runtime_root" \
  "${topology[@]}"
