# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Fail-closed validation for the TorchTitan source dependency.

The public repository records the expected gitlink in ``release/``.  A normal
Git checkout is checked by commit ID; a source-only export may instead carry
the same one-line ``.lbt_torchtitan_commit`` marker inside its TorchTitan tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess


PINNED_TORCHTITAN_COMMIT = "20b3de7585696c327bd5aa9f9627f0300abdbf9d"
TORCHTITAN_GITLINK_MARKER_NAME = ".lbt_torchtitan_commit"
PACKAGED_TORCHTITAN_GITLINK_RELATIVE = Path("release/torchtitan_gitlink.txt")
TORCHTITAN_GITLINK_MARKER_BYTES = (PINNED_TORCHTITAN_COMMIT + "\n").encode("ascii")
TORCHTITAN_GITLINK_MARKER_SHA256 = (
    "581ad66793039269f45b189794301cab27d231118fb52d714c568bd73e4c0b86"
)
PINNED_TORCHTITAN_GIT_TREE = "4464fcc17914b0253cdf761e39997afddfdf40f5"
PINNED_TORCHTITAN_PUBLIC_FILE_COUNT = 266
PINNED_TORCHTITAN_PUBLIC_LEDGER_SHA256 = (
    "6c763d2517b13e4464c0721bf265926ae4a9b509fee7d2275ceacc2ba9132eda"
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
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # A vendored source directory may live below the public repository's
        # Git root.  In that case ``git -C`` discovers the *parent* checkout;
        # its HEAD is not the vendored TorchTitan identity.  Treat only a Git
        # worktree rooted exactly at ``root`` as a standalone checkout.
        if Path(top_level).resolve() != root.resolve():
            return None
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _require_clean_git_tree(root: Path) -> None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("TorchTitan checkout cleanliness could not be verified") from error
    if result.stdout:
        raise RuntimeError("TorchTitan checkout is dirty")


def _safe_ledger_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("public file ledger contains an unsafe path")
    return path


def _read_file_ledger(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("public file ledger is absent")
    result: dict[str, str] = {}
    previous = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.fullmatch(r"[0-9a-f]{64}  .+", line):
            raise RuntimeError("public file ledger has an invalid line")
        digest, relative = line.split("  ", 1)
        _safe_ledger_path(relative)
        if relative <= previous or relative in result or relative == "SHA256SUMS":
            raise RuntimeError("public file ledger is not strictly sorted and unique")
        previous = relative
        result[relative] = digest
    return result


def _path_digest(path: Path) -> str:
    payload = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _path_mode(path: Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return "120000"
    if not stat.S_ISREG(mode):
        raise RuntimeError("TorchTitan public source contains a non-file object")
    return "100755" if mode & stat.S_IXUSR else "100644"


def _validate_flattened_source(root: Path, authority: Path) -> None:
    """Bind source-only TorchTitan bytes to the sealed public component ledger."""

    public_root = authority.resolve(strict=True).parent.parent
    expected_root = (public_root / "torchtitan_submodule").resolve(strict=True)
    if root != expected_root:
        raise RuntimeError("source-only TorchTitan root is outside the sealed public tree")

    components_path = public_root / "release/components.json"
    ledger_path = public_root / "SHA256SUMS"
    if components_path.is_symlink() or not components_path.is_file():
        raise RuntimeError("public component inventory is absent")
    try:
        inventory = json.loads(components_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("public component inventory is invalid") from error
    matches = [
        item
        for item in inventory.get("components", [])
        if isinstance(item, dict)
        and item.get("id") == "torchtitan"
        and item.get("path") == "torchtitan_submodule"
    ]
    if len(matches) != 1:
        raise RuntimeError("public TorchTitan component inventory is missing or ambiguous")
    component = matches[0]
    expected_component = {
        "commit": PINNED_TORCHTITAN_COMMIT,
        "git_tree": PINNED_TORCHTITAN_GIT_TREE,
        "file_count": PINNED_TORCHTITAN_PUBLIC_FILE_COUNT,
        "file_ledger_sha256": PINNED_TORCHTITAN_PUBLIC_LEDGER_SHA256,
    }
    if any(component.get(key) != value for key, value in expected_component.items()):
        raise RuntimeError("public TorchTitan component identity drift")

    ledger = _read_file_ledger(ledger_path)
    prefix = "torchtitan_submodule/"
    ledger_entries = {
        relative: digest for relative, digest in ledger.items() if relative.startswith(prefix)
    }
    actual_entries: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() or path.is_symlink():
            relative = path.relative_to(public_root).as_posix()
            actual_entries[relative] = path
    if set(actual_entries) != set(ledger_entries):
        raise RuntimeError("public TorchTitan file inventory drift")

    records: list[str] = []
    for relative, path in sorted(actual_entries.items()):
        digest = _path_digest(path)
        if digest != ledger_entries[relative]:
            raise RuntimeError("public TorchTitan file ledger mismatch")
        component_relative = relative[len(prefix) :]
        records.append(f"{_path_mode(path)} {digest} {component_relative}\n")
    fingerprint = hashlib.sha256("".join(records).encode()).hexdigest()
    if (
        len(records) != PINNED_TORCHTITAN_PUBLIC_FILE_COUNT
        or fingerprint != PINNED_TORCHTITAN_PUBLIC_LEDGER_SHA256
    ):
        raise RuntimeError("public TorchTitan component ledger mismatch")


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
    authority = authority.resolve(strict=True)

    head = _git_head(resolved_root)
    if head is None:
        actual_payload = _read_exact_marker(
            resolved_root / TORCHTITAN_GITLINK_MARKER_NAME,
            "source-only TorchTitan gitlink marker",
        )
        if actual_payload != expected_payload:
            raise RuntimeError("TorchTitan source marker differs from release authority")
        _validate_flattened_source(resolved_root, authority)
    elif head != PINNED_TORCHTITAN_COMMIT:
        raise RuntimeError("TorchTitan checkout commit drift")
    else:
        _require_clean_git_tree(resolved_root)

    package = resolved_root / "torchtitan" / "__init__.py"
    if not package.is_file() or resolved_root not in package.resolve().parents:
        raise RuntimeError(f"initialized TorchTitan package is absent: {package}")
    return PINNED_TORCHTITAN_COMMIT
