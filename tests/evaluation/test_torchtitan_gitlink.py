import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

os.environ.setdefault("LBT_LIGHT_IMPORT", "1")

import low_bits_training.analysis as _analysis_package

_ANALYSIS_PATH = Path(__file__).resolve().parents[2] / "low_bits_training/analysis"
if str(_ANALYSIS_PATH) not in _analysis_package.__path__:
    _analysis_package.__path__.insert(0, str(_ANALYSIS_PATH))

from low_bits_training.analysis import torchtitan_gitlink


def source_tree(root: Path) -> Path:
    package = root / "torchtitan"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (root / torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_NAME).write_bytes(
        torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_BYTES
    )
    return root


def authority(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_BYTES)
    return path


def public_source_tree(
    public_root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    root = source_tree(public_root / "torchtitan_submodule")
    marker = authority(public_root / "release/torchtitan_gitlink.txt")
    entries: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = path.relative_to(public_root).as_posix()
        digest = torchtitan_gitlink._path_digest(path)
        component_relative = path.relative_to(root).as_posix()
        entries.append(
            (
                relative,
                digest,
                f"{torchtitan_gitlink._path_mode(path)} {digest} {component_relative}\n",
            )
        )
    fingerprint = hashlib.sha256(
        "".join(record for _, _, record in entries).encode()
    ).hexdigest()
    monkeypatch.setattr(
        torchtitan_gitlink, "PINNED_TORCHTITAN_PUBLIC_FILE_COUNT", len(entries)
    )
    monkeypatch.setattr(
        torchtitan_gitlink, "PINNED_TORCHTITAN_PUBLIC_LEDGER_SHA256", fingerprint
    )
    (public_root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest, _ in entries),
        encoding="utf-8",
    )
    (public_root / "release/components.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "id": "torchtitan",
                        "path": "torchtitan_submodule",
                        "commit": torchtitan_gitlink.PINNED_TORCHTITAN_COMMIT,
                        "git_tree": torchtitan_gitlink.PINNED_TORCHTITAN_GIT_TREE,
                        "file_count": len(entries),
                        "file_ledger_sha256": fingerprint,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return root, marker


def test_source_export_accepts_exact_ledger_bound_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, marker = public_source_tree(tmp_path / "public", monkeypatch)

    assert (
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )
        == torchtitan_gitlink.PINNED_TORCHTITAN_COMMIT
    )


def test_source_export_rejects_marker_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, marker = public_source_tree(tmp_path / "public", monkeypatch)
    (root / torchtitan_gitlink.TORCHTITAN_GITLINK_MARKER_NAME).write_text(
        "0" * 40 + "\n", encoding="ascii"
    )

    with pytest.raises(RuntimeError, match="marker drift"):
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )


def test_source_export_rejects_symlinked_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, marker = public_source_tree(tmp_path / "public", monkeypatch)
    target = marker.with_name("marker-target")
    marker.rename(target)
    marker.symlink_to(target.name)

    with pytest.raises(RuntimeError, match="marker authority is absent"):
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )


def test_source_export_rejects_source_tampering_with_copied_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, marker = public_source_tree(tmp_path / "public", monkeypatch)
    (root / "torchtitan/__init__.py").write_text("# tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="file ledger mismatch"):
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )


def test_checkout_rejects_wrong_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = source_tree(tmp_path / "source")
    marker = authority(tmp_path / "authority")
    monkeypatch.setattr(torchtitan_gitlink, "_git_head", lambda _: "0" * 40)

    with pytest.raises(RuntimeError, match="checkout commit drift"):
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )


def test_checkout_rejects_dirty_pinned_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    package = root / "torchtitan"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(root)], check=True
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    marker = authority(tmp_path / "authority")
    monkeypatch.setattr(
        torchtitan_gitlink,
        "_git_head",
        lambda _: torchtitan_gitlink.PINNED_TORCHTITAN_COMMIT,
    )
    (root / "untracked.py").write_text("# dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkout is dirty"):
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )


def test_nested_source_tree_does_not_inherit_parent_git_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(repository)],
        check=True,
    )
    root, marker = public_source_tree(repository, monkeypatch)

    assert torchtitan_gitlink._git_head(root) is None
    assert (
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            root, packaged_marker=marker
        )
        == torchtitan_gitlink.PINNED_TORCHTITAN_COMMIT
    )


def test_packaged_public_tree_markers_validate() -> None:
    project_root = Path(__file__).resolve().parents[2]

    assert (
        torchtitan_gitlink.validate_torchtitan_gitlink_marker(
            project_root / "torchtitan_submodule"
        )
        == torchtitan_gitlink.PINNED_TORCHTITAN_COMMIT
    )
