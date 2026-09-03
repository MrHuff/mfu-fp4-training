from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/release/build_public_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_public_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SNAPSHOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SNAPSHOT
SPEC.loader.exec_module(SNAPSHOT)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _fixture_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    (repo / "release").mkdir(parents=True)
    (repo / "tools").mkdir()
    (repo / "scripts/release").mkdir(parents=True)
    (repo / "README.md").write_text("public fixture\n", encoding="utf-8")
    (repo / "SHA256SUMS").write_text("fixture ledger\n", encoding="utf-8")
    (repo / "release/components.json").write_text("{}\n", encoding="utf-8")
    (repo / "release/public_release_audit.json").write_text("{}\n", encoding="utf-8")
    (repo / "tools/public_clean_export.py").write_text("# fixture\n", encoding="utf-8")
    verifier = repo / "scripts/release/verify_public_bundle.sh"
    verifier.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    verifier.chmod(0o755)
    subprocess.run(
        ("git", "init", "--quiet", "--initial-branch=main", str(repo)), check=True
    )
    _git(repo, "add", "-f", "--all")
    subprocess.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ),
        check=True,
        env={
            **dict(os.environ),
            "GIT_AUTHOR_DATE": "1700000000 +0000",
            "GIT_COMMITTER_DATE": "1700000000 +0000",
        },
    )
    return repo


def test_build_snapshot_creates_one_root_tree_and_invokes_both_verifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fixture_repository(tmp_path)
    output = tmp_path / "snapshot"
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        SNAPSHOT,
        "_verify_source_tree",
        lambda tree: calls.append(("source", tree)),
    )
    monkeypatch.setattr(
        SNAPSHOT,
        "_verify_bundle",
        lambda source, bundle: calls.append(("bundle", bundle)),
    )

    report = SNAPSHOT.build_snapshot(repo, output)

    assert report["status"] == "complete"
    assert report["source_git_tree"] == report["clean_git_tree"]
    assert [kind for kind, _ in calls] == ["source", "bundle"]
    assert not (output / "source-tree/.git").exists()
    assert json.loads((output / "BUILD_REPORT.json").read_text()) == report

    bundle = output / "mfu-fp4-training.bundle"
    _git(repo, "bundle", "verify", str(bundle))
    clone = tmp_path / "clone"
    subprocess.run(("git", "clone", "--quiet", str(bundle), str(clone)), check=True)
    assert _git(clone, "rev-list", "--count", "HEAD") == "1"
    assert _git(clone, "rev-parse", "HEAD^{tree}") == _git(repo, "rev-parse", "HEAD^{tree}")

    with tarfile.open(output / "mfu-fp4-training.tar.gz", mode="r:gz") as archive:
        names = archive.getnames()
    assert names
    assert all(
        name == "mfu-fp4-training" or name.startswith("mfu-fp4-training/")
        for name in names
    )


def test_build_snapshot_rejects_dirty_public_checkout(tmp_path: Path) -> None:
    repo = _fixture_repository(tmp_path)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(SNAPSHOT.SnapshotError, match="not clean"):
        SNAPSHOT.build_snapshot(repo, tmp_path / "snapshot")


def test_archive_guard_allows_internal_parent_link_but_rejects_escape() -> None:
    safe = tarfile.TarInfo("component/subdir/requirements.txt")
    safe.type = tarfile.SYMTYPE
    safe.linkname = "../requirements.txt"
    SNAPSHOT._validate_archive_member(safe)

    unsafe = tarfile.TarInfo("component/link")
    unsafe.type = tarfile.SYMTYPE
    unsafe.linkname = "../../outside"
    with pytest.raises(SNAPSHOT.SnapshotError, match="unsafe link"):
        SNAPSHOT._validate_archive_member(unsafe)
