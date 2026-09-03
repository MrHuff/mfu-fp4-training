#!/usr/bin/env python3
"""Build and cold-verify a one-root release snapshot from public Git HEAD."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import BinaryIO, Sequence


ROOT = Path(__file__).resolve().parents[2]
RELEASE_NAME = "mfu-fp4-training"
BUILDER_NAME = "MFU FP4 public snapshot builder"
BUILDER_EMAIL = "release-builder@example.invalid"
COMMIT_MESSAGE = "Public MFU FP4 reproducibility snapshot"


class SnapshotError(RuntimeError):
    """A release-snapshot failure that does not expose source contents."""


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout: int | BinaryIO = subprocess.PIPE,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            tuple(arguments),
            cwd=cwd,
            env=env,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SnapshotError("release snapshot command failed") from error


def _git(repo: Path, *arguments: str) -> str:
    return _run(("git", "-C", str(repo), *arguments)).stdout.decode().strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_archive_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotError("Git archive contains an unsafe path")
    if member.isdev() or member.isfifo():
        raise SnapshotError("Git archive contains a non-source object")
    if member.islnk():
        raise SnapshotError("Git archive contains an unsupported hard link")
    if member.issym():
        target = PurePosixPath(member.linkname)
        combined = posixpath.normpath((path.parent / target).as_posix())
        if target.is_absolute() or combined == ".." or combined.startswith("../"):
            raise SnapshotError("Git archive contains an unsafe link")


def _extract_tracked_head(repo: Path, revision: str, destination: Path) -> None:
    archive = destination.parent / "tracked-head.tar"
    with archive.open("wb") as handle:
        _run(
            ("git", "-C", str(repo), "archive", "--format=tar", revision),
            stdout=handle,
        )
    destination.mkdir()
    with tarfile.open(archive, mode="r:") as source:
        for member in source.getmembers():
            _validate_archive_member(member)
        source.extractall(destination, filter="data")
    archive.unlink()


def _verify_source_tree(tree: Path) -> None:
    verifier = tree / "tools/public_clean_export.py"
    if not verifier.is_file():
        raise SnapshotError("public clean-export verifier is absent")
    _run(
        (
            sys.executable,
            str(verifier),
            "verify",
            "--tree",
            str(tree),
        ),
        cwd=tree,
        stdout=None,
    )


def _init_one_root_history(tree: Path, epoch: int) -> tuple[str, str]:
    _run(("git", "init", "--quiet", "--initial-branch=main", str(tree)))
    _git(tree, "config", "gc.auto", "0")
    _git(tree, "config", "user.name", BUILDER_NAME)
    _git(tree, "config", "user.email", BUILDER_EMAIL)
    _git(tree, "config", "commit.gpgSign", "false")
    _git(tree, "add", "-f", "--all")
    environment = dict(os.environ)
    stamp = f"{epoch} +0000"
    environment.update(
        {
            "GIT_AUTHOR_NAME": BUILDER_NAME,
            "GIT_AUTHOR_EMAIL": BUILDER_EMAIL,
            "GIT_COMMITTER_NAME": BUILDER_NAME,
            "GIT_COMMITTER_EMAIL": BUILDER_EMAIL,
            "GIT_AUTHOR_DATE": stamp,
            "GIT_COMMITTER_DATE": stamp,
            "TZ": "UTC",
        }
    )
    _run(("git", "-C", str(tree), "commit", "--quiet", "-m", COMMIT_MESSAGE), env=environment)
    if _git(tree, "rev-list", "--count", "HEAD") != "1":
        raise SnapshotError("snapshot did not produce one root commit")
    if any(line.startswith("160000 ") for line in _git(tree, "ls-files", "--stage").splitlines()):
        raise SnapshotError("snapshot retained a Git submodule entry")
    return _git(tree, "rev-parse", "HEAD"), _git(tree, "rev-parse", "HEAD^{tree}")


def _write_archive(repository: Path, destination: Path) -> None:
    process = subprocess.Popen(
        (
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            f"--prefix={RELEASE_NAME}/",
            "HEAD",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            shutil.copyfileobj(process.stdout, compressed)
    _, error = process.communicate()
    if process.returncode:
        destination.unlink(missing_ok=True)
        raise SnapshotError("deterministic source archive creation failed") from subprocess.CalledProcessError(
            process.returncode, process.args, stderr=error
        )


def _verify_bundle(repo: Path, bundle: Path) -> None:
    verifier = repo / "scripts/release/verify_public_bundle.sh"
    if not verifier.is_file():
        raise SnapshotError("public bundle verifier is absent")
    _run((str(verifier), str(bundle)), cwd=repo, stdout=None)


def build_snapshot(repo: Path, output: Path) -> dict[str, object]:
    repo = repo.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise SnapshotError("output path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    head = _git(repo, "rev-parse", "HEAD^{commit}")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise SnapshotError("public source checkout is not clean")
    source_tree = _git(repo, "rev-parse", f"{head}^{{tree}}")
    source_epoch = int(_git(repo, "show", "-s", "--format=%ct", head))
    staged = _git(repo, "ls-tree", "-r", head)
    if any(line.startswith("160000 ") for line in staged.splitlines()):
        raise SnapshotError("public source checkout contains a Git submodule")
    for required in (
        "SHA256SUMS",
        "release/components.json",
        "release/public_release_audit.json",
        "tools/public_clean_export.py",
        "scripts/release/verify_public_bundle.sh",
    ):
        _git(repo, "cat-file", "-e", f"{head}:{required}")

    temporary = Path(
        tempfile.mkdtemp(prefix=".mfu-public-snapshot-", dir=output.parent)
    )
    tree = temporary / "source-tree"
    try:
        _extract_tracked_head(repo, head, tree)
        _verify_source_tree(tree)
        clean_commit, clean_tree = _init_one_root_history(tree, source_epoch)
        if clean_tree != source_tree:
            raise SnapshotError("snapshot root tree differs from public source HEAD")

        archive = temporary / f"{RELEASE_NAME}.tar.gz"
        bundle = temporary / f"{RELEASE_NAME}.bundle"
        _write_archive(tree, archive)
        _git(
            tree,
            "-c",
            "pack.threads=1",
            "bundle",
            "create",
            str(bundle),
            "HEAD",
            "refs/heads/main",
        )
        _git(tree, "bundle", "verify", str(bundle))
        shutil.rmtree(tree / ".git")
        _verify_bundle(repo, bundle)

        report: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "source_commit": head,
            "source_git_tree": source_tree,
            "clean_commit": clean_commit,
            "clean_git_tree": clean_tree,
            "commit_count": 1,
            "archive_sha256": _sha256(archive),
            "bundle_sha256": _sha256(bundle),
            "source_verification": "pass",
            "cold_bundle_verification": "pass",
            "local_paths_recorded": False,
        }
        (temporary / "BUILD_REPORT.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_snapshot(ROOT, args.output)
    except (SnapshotError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"public_snapshot_status={report['status']}")
    print(f"clean_commit={report['clean_commit']}")
    print(f"archive_sha256={report['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
