#!/usr/bin/env python3
"""Verify the immutable-container runtime recorded by the public capsule."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

import torch
import triton


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    contract = json.loads((ROOT / "release/container_dependency_lock.json").read_text())
    runtime = contract["image_baked_runtime"]
    observed = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }
    mismatch = {
        key: (runtime[key], value)
        for key, value in observed.items()
        if value != runtime[key]
    }
    packages = contract["required_python_packages_observed_in_image"]
    for distribution, wanted in packages.items():
        try:
            found = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            found = None
        if found != wanted:
            mismatch[f"package:{distribution}"] = (wanted, found)
    if mismatch:
        details = ", ".join(
            f"{name}=expected:{wanted!r}/found:{found!r}"
            for name, (wanted, found) in sorted(mismatch.items())
        )
        raise RuntimeError(f"container dependency contract mismatch: {details}")
    print(
        "container dependency contract verified; immutable base reference: "
        + contract["base_image"]["reference"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
