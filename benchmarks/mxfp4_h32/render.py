"""Bind local external inputs into a non-cluster benchmark plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from .route_contract import DEFAULT_SPEC, file_sha256, load_spec


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _local_directory(path: Path, name: str, *, must_exist: bool = True) -> Path:
    raw = str(path)
    if "://" in raw:
        raise ValueError(f"{name} must be a local directory")
    resolved = path.expanduser().resolve()
    if must_exist and not resolved.is_dir():
        raise ValueError(f"{name} is not a directory: {resolved}")
    return resolved


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def render_plan(
    *,
    spec_path: Path,
    model_assets: Path,
    dataset: Path,
    output_dir: Path,
    plan_path: Path,
    repository: Path = REPO_ROOT,
    runtime: Path | None = None,
) -> dict[str, Any]:
    repository = _local_directory(repository, "repository")
    runtime = _local_directory(runtime or repository / "fp4_runtime", "runtime")
    model_assets = _local_directory(model_assets, "model_assets")
    dataset = _local_directory(dataset, "dataset")
    output_dir = _local_directory(output_dir, "output_dir", must_exist=False)
    plan_path = plan_path.expanduser().resolve()
    if _inside(output_dir, repository) or _inside(plan_path, repository):
        raise ValueError("generated plans and results must remain outside the checkout")
    if plan_path.parent != output_dir:
        raise ValueError("plan must be written directly inside output_dir")

    spec = load_spec(spec_path)
    runtime_head = _git_head(runtime)
    if runtime_head != spec["runtime"]["fp4_commit"]:
        raise ValueError(
            f"runtime commit mismatch: {runtime_head} != "
            f"{spec['runtime']['fp4_commit']}"
        )
    torchtitan = repository / "torchtitan_submodule"
    torchtitan_head = _git_head(torchtitan)
    if torchtitan_head != spec["runtime"]["torchtitan_commit"]:
        raise ValueError("TorchTitan commit mismatch")

    plan: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_spec": spec,
        "bindings": {
            "repository": str(repository),
            "runtime": str(runtime),
            "model_assets": str(model_assets),
            "dataset": str(dataset),
            "output_dir": str(output_dir),
        },
        "source": {
            "repository_commit": _git_head(repository),
            "runtime_commit": runtime_head,
            "torchtitan_commit": torchtitan_head,
            "benchmark_spec_sha256": file_sha256(spec_path),
            "route_contract_sha256": file_sha256(
                Path(__file__).resolve().parent / "route_contract.py"
            ),
        },
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan["plan_sha256"] = sha256(canonical).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--model-assets", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    plan = render_plan(
        spec_path=args.spec,
        model_assets=args.model_assets,
        dataset=args.dataset,
        output_dir=args.output_dir,
        plan_path=args.plan,
        repository=args.repository,
        runtime=args.runtime,
    )
    print(f"rendered local plan sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
