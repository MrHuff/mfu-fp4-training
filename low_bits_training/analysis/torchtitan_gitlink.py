# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Fail-closed validation for the TorchTitan source dependency.

The public repository records the expected gitlink in ``release/``.  A normal
Git checkout is checked by commit ID; a source-only export may instead carry
the same one-line ``.lbt_torchtitan_commit`` marker inside its TorchTitan tree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


PINNED_TORCHTITAN_COMMIT = "20b3de7585696c327bd5aa9f9627f0300abdbf9d"
TORCHTITAN_GITLINK_MARKER_NAME = ".lbt_torchtitan_commit"
PACKAGED_TORCHTITAN_GITLINK_RELATIVE = Path("release/torchtitan_gitlink.txt")
TORCHTITAN_GITLINK_MARKER_BYTES = (PINNED_TORCHTITAN_COMMIT + "\n").encode("ascii")
TORCHTITAN_GITLINK_MARKER_SHA256 = (
    "581ad66793039269f45b189794301cab27d231118fb52d714c568bd73e4c0b86"
)


def _read_exact_marker(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    payload = path.read_bytes()
    if (
        payload != TORCHTITAN_GITLINK_MARKER_BYTES
        or hashlib.sha256(payload).hexdigest() != TORCHTITAN_GITLINK_MARKER_SHA256
    ):
        raise RuntimeError(f"{label} drift")
    return payload


def _git_head(root: Path) -> str | None:
    """Return HEAD for a Git checkout, or ``None`` for a source-only tree."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def validate_torchtitan_gitlink_marker(
    root: str | Path, *, packaged_marker: str | Path | None = None
) -> str:
    """Validate an initialized checkout or source-only export against the pin."""

    candidate = Path(root)
    try:
        resolved_root = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"TorchTitan root is absent: {candidate}") from error
    if not resolved_root.is_dir():
        raise RuntimeError(f"TorchTitan root is not a directory: {resolved_root}")

    project_root = Path(__file__).resolve().parents[2]
    authority = (
        project_root / PACKAGED_TORCHTITAN_GITLINK_RELATIVE
        if packaged_marker is None
        else Path(packaged_marker)
    )
    expected_payload = _read_exact_marker(
        authority, "packaged TorchTitan gitlink marker authority"
    )

    head = _git_head(resolved_root)
    if head is None:
        actual_payload = _read_exact_marker(
            resolved_root / TORCHTITAN_GITLINK_MARKER_NAME,
            "source-only TorchTitan gitlink marker",
        )
        if actual_payload != expected_payload:
            raise RuntimeError("TorchTitan source marker differs from release authority")
    elif head != PINNED_TORCHTITAN_COMMIT:
        raise RuntimeError("TorchTitan checkout commit drift")

    package = resolved_root / "torchtitan" / "__init__.py"
    if not package.is_file() or resolved_root not in package.resolve().parents:
        raise RuntimeError(f"initialized TorchTitan package is absent: {package}")
    return PINNED_TORCHTITAN_COMMIT
