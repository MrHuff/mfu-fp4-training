#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
destination=${1:-}
if [[ -z "$destination" || "$destination" != /* ]]; then
  echo "usage: $0 ABSOLUTE_VENV_PATH" >&2
  exit 2
fi
if [[ -e "$destination" ]]; then
  echo "error: refusing to overwrite existing evaluation environment" >&2
  exit 2
fi
cd "$repo_root"

python tools/verify_container_contract.py
base_torch=$(python -c 'import torch; print(torch.__version__)')
base_cuda=$(python -c 'import torch; print(torch.version.cuda)')
python -m venv --system-site-packages "$destination"
"$destination/bin/python" -m pip install \
  --disable-pip-version-check --no-cache-dir --no-deps --require-hashes \
  -r release/evaluation_requirements.lock

"$destination/bin/python" - "$base_torch" "$base_cuda" <<'PY'
from importlib.metadata import version
import sys
import torch

wanted = {
    "accelerate": "1.7.0",
    "datasets": "3.6.0",
    "lm-eval": "0.4.12",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.1",
    "transformers": "4.48.2",
}
observed = {name: version(name) for name in wanted}
if observed != wanted:
    raise SystemExit(f"evaluation dependency drift: {observed}")
if torch.__version__ != sys.argv[1] or torch.version.cuda != sys.argv[2]:
    raise SystemExit("evaluation overlay changed the training PyTorch/CUDA ABI")
from lm_eval.__main__ import cli_evaluate  # noqa: F401
print("hash-locked evaluation overlay verified")
PY
