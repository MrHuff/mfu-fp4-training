#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bundle=${1:-}
if [[ -z "$bundle" || ! -f "$bundle" ]]; then
  echo "usage: $0 /path/to/mfu-fp4-training.bundle" >&2
  exit 2
fi
bundle=$(realpath "$bundle")

check_root=$(mktemp -d /tmp/mfu-public-bundle-check-XXXXXX)
cleanup() {
  # check_root is always an explicit directory returned by mktemp above.
  find "$check_root" -depth -delete
}
trap cleanup EXIT

git -C "$repo_root" bundle verify "$bundle" >/dev/null
git clone --quiet "$bundle" "$check_root/repository"
cd "$check_root/repository"

[[ $(git rev-list --count HEAD) == 1 ]]
[[ -z $(git submodule status --recursive) ]]
python tools/public_clean_export.py verify --tree . --allow-blocked >/dev/null
scripts/release/bootstrap.sh --verify-only >/dev/null
scripts/release/run_gates.sh --cpu-only >/dev/null
make -C docs/technical_report >/dev/null
make -C docs/technical_report arxiv-source >/dev/null
test -f docs/technical_report/build/fp4_training_systems_arxiv.tar.gz
test -f docs/technical_report/build/fp4_training_systems_overleaf.zip
(
  cd docs/technical_report/build
  sha256sum -c fp4_training_systems_arxiv.tar.gz.sha256 >/dev/null
  sha256sum -c fp4_training_systems_overleaf.zip.sha256 >/dev/null
)

printf 'public_bundle_cold_check=pass\n'
printf 'clean_commit=%s\n' "$(git rev-parse HEAD)"
