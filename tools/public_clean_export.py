#!/usr/bin/env python3
"""Build and verify a scrubbed public source export.

The builder reads only Git objects at explicit commits.  It never copies a
working tree, inherited Git metadata, credentials, checkpoint material, or
rendered scheduler objects.  Every dependency is recursively materialized at
the gitlink recorded by its parent, then represented by a content ledger in a
new one-commit repository.  The source tree and compressed source archive are
byte-deterministic for a fixed input contract.  The Git bundle is a cloneable
transport for that one-commit repository; its container bytes are not claimed
to be deterministic across independent builds.

Findings contain rule names and public-relative paths only.  Matched content,
local cache locations, and credential-shaped values are never emitted.
"""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass, field
import fnmatch
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "release" / "public_export_manifest.json"
COMPONENT_INVENTORY = PurePosixPath("release/components.json")
AUDIT_REPORT = PurePosixPath("release/public_release_audit.json")
FILE_LEDGER = PurePosixPath("SHA256SUMS")


class ExportError(RuntimeError):
    """A fail-closed export error whose message contains no source content."""


@dataclass(frozen=True, order=True)
class Finding:
    rule: str
    path: str


@dataclass(frozen=True)
class GitEntry:
    mode: str
    kind: str
    object_id: str
    path: str


@dataclass(frozen=True)
class FileRecord:
    path: str
    mode: str
    sha256: str
    component_path: str
    component_relative_path: str


@dataclass
class ComponentRecord:
    component_id: str
    path: str
    url: str | None
    commit: str
    tree: str
    parent: str | None
    files: list[FileRecord] = field(default_factory=list)


@dataclass(frozen=True)
class SanitizationRecord:
    path: str
    source_sha256: str
    output_sha256: str
    rules: tuple[str, ...]


@dataclass(frozen=True)
class SourceRewriteRecord:
    path: str
    source_sha256: str
    output_sha256: str
    rule: str


SECRET_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "aws_access_key_value",
        re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    (
        "private_key_block",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "github_token_value",
        re.compile(
            rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|"
            rb"github_pat_[A-Za-z0-9_]{30,})"
        ),
    ),
    (
        "huggingface_token_value",
        re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"),
    ),
    (
        "slack_token_value",
        re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}"),
    ),
    (
        "credential_assignment",
        re.compile(
            rb"(?i)(?:secretAccessKey|aws_secret_access_key|sessionToken|"
            rb"aws_session_token|WANDB_API_KEY|HF_TOKEN|GITHUB_TOKEN)"
            rb"[\x20\t]*=[\x20\t\"\x27]+"
            rb"(?!\$\{|\$[A-Za-z_]|<|REPLACE|EXAMPLE)[^\s\"\x27]{12,}"
        ),
    ),
    (
        "credential_json_value",
        re.compile(
            rb"(?i)[\"\x27](?:secretAccessKey|aws_secret_access_key|sessionToken|"
            rb"aws_session_token|WANDB_API_KEY|HF_TOKEN|GITHUB_TOKEN)[\"\x27]"
            rb"[\x20\t]*:[\x20\t]*[\"\x27]"
            rb"(?!\$\{|\$[A-Za-z_]|<|REPLACE|EXAMPLE)[^\"\x27]{12,}[\"\x27]"
        ),
    ),
)

_CLUSTER_CONTROL_IDENTIFIERS = (
    rb"kube" + rb"ctl",
    rb"kue" + rb"ue",
    rb"py" + rb"torchjob",
    rb"volt" + rb"-dev",
    rb"volt" + rb"_dev",
    rb"volt" + rb" dev",
    rb"volt" + rb"-prod",
    rb"volt" + rb"_prod",
    rb"volt" + rb" prod",
)
_CLUSTER_CONTROL_PATTERN = re.compile(
    rb"(?i)(?<![A-Za-z0-9_])(?:"
    + rb"|".join(re.escape(value) for value in _CLUSTER_CONTROL_IDENTIFIERS)
    + rb")(?:s)?(?![A-Za-z0-9_])"
)
CLUSTER_CONTROL_PATH_FRAGMENTS = tuple(
    value.decode("ascii").replace(" ", "_") for value in _CLUSTER_CONTROL_IDENTIFIERS
) + (
    "k" + "8s",
    "kube" + "rnetes",
)
INTERNAL_AGENT_DIRECTORIES = {".claude", ".codex", ".gemini"}


IDENTITY_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private_host_path",
        re.compile(
            rb"(?<![A-Za-z0-9_}])/(?:workspace|volt|Users|data|mnt|private/tmp)/"
        ),
    ),
    (
        "private_home_path",
        re.compile(
            rb"(?<![A-Za-z0-9_])/" + rb"home/(?!USER(?:/|\b)|<)[^/\s]+/"
        ),
    ),
    (
        "storage_uri",
        re.compile(rb"(?i)(?:s3|gs|abfs|az|azure)://[^\s\"\x27<>]+"),
    ),
    (
        "object_store_https_uri",
        re.compile(rb"(?i)https?://[^\s/\"\x27<>]*\.s3[^\s\"\x27<>]*/"),
    ),
    (
        "wandb_identity",
        re.compile(
            rb"(?i)(?:(?<![A-Za-z0-9_.-])wandb\.ai/"
            rb"[^\s/\"\x27<>]+/[^\s\"\x27<>]+|"
            rb"WANDB_(?:ENTITY|PROJECT|NAME|RUN_ID|GROUP)[\x20\t]*="
            rb"[\x20\t]*[\"\x27]?(?!\$|<|REPLACE)[A-Za-z0-9_.-]+)"
        ),
    ),
    (
        "uuid_identity",
        re.compile(
            rb"(?i)(?:owner[_-]?uid|job[_-]?uid|\buid)"
            rb"[\x20\t\"\x27]*[:=][\x20\t\"\x27]+"
            rb"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            rb"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])"
        ),
    ),
    (
        "static_job_or_run_identity",
        re.compile(
            rb"(?i)(?:job[_-]?(?:id|name)|run[_-]?id|owner[_-]?uid)"
            rb"[\x20\t\"\x27]*[:=][\x20\t]*[\"\x27]"
            rb"(?!\$|<|REPLACE|EXAMPLE|None\b|null\b)[A-Za-z0-9][A-Za-z0-9_.-]{7,}"
        ),
    ),
    ("cluster_control_identity", _CLUSTER_CONTROL_PATTERN),
)


SANITIZE_PRIVATE_HOST = re.compile(
    rb"(?<![A-Za-z0-9_}])/(?:workspace|volt|Users|data|mnt|private/tmp)/"
    rb"[^\s\"\x27<>(){}\[\],;`]*"
)
SANITIZE_PRIVATE_HOME = re.compile(
    rb"(?<![A-Za-z0-9_])/"
    + rb"home/(?!USER(?:/|\b)|<)[^/\s]+/[^\s\"\x27<>(){}\[\],;`]*"
)
SANITIZE_STORAGE_URI = re.compile(
    rb"(?i)(?:s3|gs|abfs|az|azure)://[^\s\"\x27<>(){}\[\],;`]+"
)
SANITIZE_OBJECT_STORE_HTTPS = re.compile(
    rb"(?i)https?://[^\s/\"\x27<>]*\.s3"
    rb"[^\s\"\x27<>(){}\[\],;`]*"
)
SANITIZE_WANDB_URL = re.compile(
    rb"(?i)(?<![A-Za-z0-9_.-])wandb\.ai/[^\s/\"\x27<>]+/"
    rb"[^\s\"\x27<>(){}\[\],;`]+"
)
SANITIZE_WANDB_ASSIGNMENT = re.compile(
    rb"(?i)(WANDB_(?:ENTITY|PROJECT|NAME|RUN_ID|GROUP)[\x20\t]*="
    rb"[\x20\t]*[\"\x27]?)(?!\$|<|REPLACE)[A-Za-z0-9_.-]+"
)
SANITIZE_UUID_ASSIGNMENT = re.compile(
    rb"(?i)((?:owner[_-]?uid|job[_-]?uid|\buid)"
    rb"[\x20\t\"\x27]*[:=][\x20\t\"\x27]+)"
    rb"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    rb"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])"
)
SANITIZE_STATIC_ID_ASSIGNMENT = re.compile(
    rb"(?i)((?:job[_-]?(?:id|name)|run[_-]?id|owner[_-]?uid)"
    rb"[\x20\t\"\x27]*[:=][\x20\t]*[\"\x27])"
    rb"(?!\$|<|REPLACE|EXAMPLE|None\b|null\b)[A-Za-z0-9][A-Za-z0-9_.-]{7,}"
)


FORBIDDEN_BINARY_SUFFIXES = {
    ".a",
    ".arrow",
    ".bin",
    ".bz2",
    ".co",
    ".ckpt",
    ".cubin",
    ".distcp",
    ".dll",
    ".dylib",
    ".fatbin",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".ncu-rep",
    ".nsys-rep",
    ".npy",
    ".npz",
    ".o",
    ".onnx",
    ".otf",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".hsaco",
    ".inv",
    ".jar",
    ".pq",
    ".so",
    ".trace",
    ".ttf",
    ".tar",
    ".whl",
    ".woff",
    ".woff2",
    ".xz",
    ".zip",
    ".zst",
}


def _run(
    command: Iterable[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.run(
        tuple(command),
        cwd=cwd,
        env=env,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and process.returncode:
        executable = Path(tuple(command)[0]).name
        raise ExportError(f"{executable} failed with status {process.returncode}")
    return process


def _git(repo: Path, *args: str, check: bool = True) -> bytes:
    return _run(("git", "-C", str(repo), *args), check=check).stdout


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ExportError("export manifest must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o644)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitize_identity_text(data: bytes) -> tuple[bytes, dict[str, int]]:
    """Replace identity-bearing defaults/examples without touching algorithms.

    This is deliberately separate from the secret scanner.  Credentials are
    never rewritten: any credential-shaped value remains a hard audit failure.
    Sanitization is available only for paths explicitly named by the manifest,
    and the source/output hashes are recorded in the public inventory.
    """

    if b"\0" in data[:8192]:
        raise ExportError("an explicitly sanitized file is not text")
    counts: dict[str, int] = {}

    def replace(
        value: bytes,
        *,
        rule: str,
        pattern: re.Pattern[bytes],
        replacement: bytes | Any,
    ) -> bytes:
        updated, count = pattern.subn(replacement, value)
        if count:
            counts[rule] = counts.get(rule, 0) + count
        return updated

    data = replace(
        data,
        rule="private_host_path",
        pattern=SANITIZE_PRIVATE_HOST,
        replacement=lambda match: b"/opt/mfu/EXTERNAL_PATH"
        + (b"/" if match.group(0).endswith(b"/") else b""),
    )
    data = replace(
        data,
        rule="private_home_path",
        pattern=SANITIZE_PRIVATE_HOME,
        replacement=b"/home/USER/EXTERNAL_PATH",
    )
    data = replace(
        data,
        rule="storage_uri",
        pattern=SANITIZE_STORAGE_URI,
        replacement=b"OBJECT_STORE_URI",
    )
    data = replace(
        data,
        rule="object_store_https_uri",
        pattern=SANITIZE_OBJECT_STORE_HTTPS,
        replacement=b"OBJECT_STORE_HTTPS_URI",
    )
    data = replace(
        data,
        rule="wandb_identity",
        pattern=SANITIZE_WANDB_URL,
        replacement=b"EXPERIMENT_TRACKER_RUN",
    )
    data = replace(
        data,
        rule="wandb_identity",
        pattern=SANITIZE_WANDB_ASSIGNMENT,
        replacement=lambda match: match.group(1) + b"REPLACE",
    )
    data = replace(
        data,
        rule="uuid_identity",
        pattern=SANITIZE_UUID_ASSIGNMENT,
        replacement=lambda match: match.group(1) + b"EXAMPLE",
    )
    data = replace(
        data,
        rule="static_job_or_run_identity",
        pattern=SANITIZE_STATIC_ID_ASSIGNMENT,
        replacement=lambda match: match.group(1) + b"EXAMPLE",
    )
    return data, counts


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\x00" in value
    ):
        raise ExportError("manifest contains an unsafe relative path")
    return path


def _matches(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatchcase(path, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]):
            return True
    return False


def _canonical_git_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ExportError("dependency URL is not anonymous HTTPS")
    if parsed.hostname.lower() != "github.com":
        raise ExportError("dependency URL host is not allowlisted")
    if parsed.query or parsed.fragment or not parsed.path.strip("/"):
        raise ExportError("dependency URL is not a plain repository URL")
    path = parsed.path.rstrip("/")
    if not path.endswith(".git"):
        path += ".git"
    return f"https://github.com{path}"


def _git_entries(repo: Path, revision: str) -> list[GitEntry]:
    raw = _git(repo, "ls-tree", "-rz", "--full-tree", revision)
    entries: list[GitEntry] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, raw_path = item.split(b"\t", 1)
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ExportError("repository contains a non-UTF-8 path") from error
        mode, kind, object_id = metadata.decode("ascii").split()
        _safe_relative(path)
        entries.append(GitEntry(mode, kind, object_id, path))
    return entries


def _resolve_commit(repo: Path, revision: str) -> tuple[str, str]:
    commit = _git(repo, "rev-parse", f"{revision}^{{commit}}").decode().strip()
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(
        r"[0-9a-f]{40}", tree
    ):
        raise ExportError("resolved revision is not a SHA-1 commit")
    return commit, tree


def _gitlink(repo: Path, revision: str, path: str) -> str | None:
    output = _git(repo, "ls-tree", "-z", revision, "--", path)
    items = [item for item in output.split(b"\0") if item]
    if len(items) != 1:
        return None
    metadata, actual = items[0].split(b"\t", 1)
    mode, kind, object_id = metadata.decode("ascii").split()
    if actual.decode("utf-8") != path or mode != "160000" or kind != "commit":
        return None
    return object_id


def _submodules(repo: Path, revision: str) -> list[tuple[str, str, str]]:
    process = _run(
        ("git", "-C", str(repo), "show", f"{revision}:.gitmodules"),
        check=False,
    )
    if process.returncode:
        gitlinks = [entry for entry in _git_entries(repo, revision) if entry.mode == "160000"]
        if gitlinks:
            raise ExportError("gitlinks exist without a readable .gitmodules file")
        return []
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read_string(process.stdout.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as error:
        raise ExportError("invalid .gitmodules data") from error
    result: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()
    for section in parser.sections():
        match = re.fullmatch(r'submodule "(.+)"', section)
        if not match or "path" not in parser[section] or "url" not in parser[section]:
            raise ExportError("invalid submodule declaration")
        path = parser[section]["path"]
        url = parser[section]["url"]
        _safe_relative(path)
        if path in seen_paths:
            raise ExportError("duplicate submodule path")
        seen_paths.add(path)
        object_id = _gitlink(repo, revision, path)
        if object_id is None:
            # Stale declarations are not dependencies and are not exported.
            continue
        result.append((path, url, object_id))
    declared = {path for path, _, _ in result}
    actual = {entry.path for entry in _git_entries(repo, revision) if entry.mode == "160000"}
    if declared != actual:
        raise ExportError("not every gitlink has one exact submodule declaration")
    return sorted(result)


class RepositoryResolver:
    def __init__(
        self,
        cache: Path,
        repo_map: dict[str, Path] | None = None,
        offline: bool = False,
    ) -> None:
        self.cache = cache
        self.repo_map = {
            _canonical_git_url(url): path.resolve() for url, path in (repo_map or {}).items()
        }
        self.offline = offline
        self.cache.mkdir(parents=True, exist_ok=True)

    def resolve(self, url: str, commit: str) -> Path:
        canonical = _canonical_git_url(url)
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ExportError("dependency pin is not a full commit ID")
        mapped = self.repo_map.get(canonical)
        if mapped is not None:
            self._verify_object(mapped, commit)
            return mapped
        key = hashlib.sha256(canonical.encode()).hexdigest()
        repository = self.cache / f"{key}.git"
        if not repository.exists():
            if self.offline:
                raise ExportError("dependency is unavailable in the offline repository map")
            _run(("git", "init", "--bare", "--quiet", str(repository)))
            _git(repository, "remote", "add", "origin", canonical)
        if not self._has_object(repository, commit):
            if self.offline:
                raise ExportError("pinned dependency commit is absent from the offline cache")
            anonymous_env = dict(os.environ)
            anonymous_env.update(
                {
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_ASKPASS": "/bin/false",
                }
            )
            fetch = _run(
                (
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "credential.helper=",
                    "-c",
                    "protocol.version=2",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "--depth=1",
                    "origin",
                    commit,
                ),
                env=anonymous_env,
                check=False,
            )
            if fetch.returncode:
                _run(
                    (
                        "git",
                        "-C",
                        str(repository),
                        "-c",
                        "credential.helper=",
                        "-c",
                        "protocol.version=2",
                        "fetch",
                        "--quiet",
                        "--no-tags",
                        "origin",
                        "+refs/heads/*:refs/heads/*",
                        "+refs/tags/*:refs/tags/*",
                    ),
                    env=anonymous_env,
                )
        self._verify_object(repository, commit)
        return repository

    @staticmethod
    def _has_object(repo: Path, commit: str) -> bool:
        process = _run(
            ("git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"),
            check=False,
        )
        return process.returncode == 0

    def _verify_object(self, repo: Path, commit: str) -> None:
        process = _run(
            ("git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"),
            check=False,
        )
        if process.returncode:
            raise ExportError("repository does not contain the pinned dependency commit")


class ExportBuilder:
    def __init__(
        self,
        *,
        source_repo: Path,
        source_revision: str,
        manifest: dict[str, Any],
        resolver: RepositoryResolver,
    ) -> None:
        self.source_repo = source_repo.resolve()
        self.source_revision = source_revision
        self.manifest = manifest
        self.resolver = resolver
        self.exclude = tuple(manifest["policy"]["exclude"])
        declarations = list(manifest.get("components", [])) + list(
            manifest.get("required_nested_components", [])
        )
        self.component_declarations: dict[str, dict[str, Any]] = {}
        for declaration in declarations:
            path = str(_safe_relative(declaration["path"]))
            if path in self.component_declarations:
                raise ExportError("component paths must be unique")
            include = declaration.get("include")
            if include is not None and (
                not isinstance(include, list)
                or not include
                or not all(isinstance(pattern, str) and pattern for pattern in include)
            ):
                raise ExportError("component include rules must be a non-empty string list")
            self.component_declarations[path] = declaration
        raw_binary_allowlist = manifest["policy"].get("binary_allowlist", {})
        if not isinstance(raw_binary_allowlist, dict):
            raise ExportError("binary allowlist must bind exact paths to SHA-256 values")
        self.binary_hashes: dict[str, str] = {}
        for raw_path, expected_sha256 in raw_binary_allowlist.items():
            path = str(_safe_relative(raw_path))
            if any(character in path for character in "*?["):
                raise ExportError("binary allowlist paths must be exact")
            if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
                raise ExportError("binary allowlist contains an invalid SHA-256 value")
            self.binary_hashes[path] = str(expected_sha256)
        self.binary_allowlist = tuple(sorted(self.binary_hashes))
        self.text_sanitize = tuple(manifest["policy"].get("text_sanitize", []))
        self.sanitize_matches = {pattern: 0 for pattern in self.text_sanitize}
        self.sanitizations: list[SanitizationRecord] = []
        self.source_rewrites: dict[str, dict[str, str]] = {}
        self.source_rewrite_matches: set[str] = set()
        self.source_rewrite_records: list[SourceRewriteRecord] = []
        for raw_rewrite in manifest["policy"].get("source_rewrites", []):
            required_keys = {"path", "source_sha256", "rule", "old", "new"}
            if not isinstance(raw_rewrite, dict) or set(raw_rewrite) != required_keys:
                raise ExportError("each source rewrite must have an exact bound schema")
            path = str(_safe_relative(raw_rewrite["path"]))
            if any(character in path for character in "*?[") or path in self.source_rewrites:
                raise ExportError("source rewrite paths must be exact and unique")
            source_sha256 = raw_rewrite["source_sha256"]
            rule = raw_rewrite["rule"]
            old = raw_rewrite["old"]
            new = raw_rewrite["new"]
            if not isinstance(source_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", source_sha256
            ):
                raise ExportError("source rewrite has an invalid SHA-256")
            if not isinstance(rule, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", rule):
                raise ExportError("source rewrite has an invalid rule")
            if not isinstance(old, str) or not isinstance(new, str) or not old or old == new:
                raise ExportError("source rewrite has invalid replacement text")
            self.source_rewrites[path] = {
                "source_sha256": source_sha256,
                "rule": rule,
                "old": old,
                "new": new,
            }
        self.records: list[FileRecord] = []
        self.components: list[ComponentRecord] = []
        self.destinations: set[str] = set()
        self.exclusion_counts: dict[str, int] = {}

    def build_tree(self, tree_root: Path) -> tuple[str, str]:
        tree_root.mkdir(parents=True)
        source_commit, source_tree = _resolve_commit(self.source_repo, self.source_revision)
        required = self.manifest["source"].get("required_ancestor")
        if required:
            process = _run(
                (
                    "git",
                    "-C",
                    str(self.source_repo),
                    "merge-base",
                    "--is-ancestor",
                    required,
                    source_commit,
                ),
                check=False,
            )
            if process.returncode:
                raise ExportError("source revision lacks the required release lineage")

        source_component = ComponentRecord(
            component_id="mfu_fp4_training",
            path=".",
            url=None,
            commit=source_commit,
            tree=source_tree,
            parent=None,
        )
        self.components.append(source_component)
        entries = _git_entries(self.source_repo, source_commit)
        include = tuple(self.manifest["source"]["include"])
        optional = set(self.manifest["source"].get("optional_include", []))
        matched = {pattern: 0 for pattern in include}
        for entry in entries:
            for pattern in include:
                if _matches(entry.path, (pattern,)):
                    matched[pattern] += 1
            if entry.mode == "160000" or not _matches(entry.path, include):
                continue
            if self._excluded(entry.path):
                continue
            record = self._write_entry(
                self.source_repo,
                entry,
                tree_root,
                entry.path,
                source_component,
            )
            source_component.files.append(record)
        missing = [pattern for pattern, count in matched.items() if not count and pattern not in optional]
        if missing:
            raise ExportError("a required source allowlist entry did not match the source commit")

        for overlay in self.manifest["source"].get("overlays", []):
            source_path = str(_safe_relative(overlay["source"]))
            destination = str(_safe_relative(overlay["destination"]))
            by_path = {entry.path: entry for entry in entries}
            if source_path not in by_path or by_path[source_path].mode == "160000":
                raise ExportError("a required public overlay is absent from the source commit")
            if self._excluded(destination):
                raise ExportError("a public overlay destination is excluded by policy")
            if destination in self.destinations:
                self._remove_record(destination, source_component)
                target = tree_root / destination
                if target.is_symlink() or target.is_file():
                    target.unlink()
            record = self._write_entry(
                self.source_repo,
                by_path[source_path],
                tree_root,
                destination,
                source_component,
                component_relative_path=destination,
            )
            source_component.files.append(record)

        source_submodules = {
            path: (url, commit) for path, url, commit in _submodules(self.source_repo, source_commit)
        }
        for item in self.manifest["components"]:
            path = str(_safe_relative(item["path"]))
            if path not in source_submodules:
                raise ExportError("required component is not an exact source gitlink")
            declared_url, gitlink = source_submodules[path]
            if gitlink != item["commit"]:
                raise ExportError("required component gitlink does not match its manifest pin")
            if _canonical_git_url(declared_url) != _canonical_git_url(item["url"]):
                raise ExportError("required component URL does not match the source declaration")
            repo = self.resolver.resolve(item["url"], item["commit"])
            self._materialize_component(
                repo=repo,
                url=item["url"],
                commit=item["commit"],
                export_path=path,
                component_id=item["id"],
                parent="mfu_fp4_training",
                tree_root=tree_root,
            )

        by_source_path = {entry.path: entry for entry in entries}
        by_component_path = {component.path: component for component in self.components}
        for overlay in self.manifest["source"].get("component_overlays", []):
            if not isinstance(overlay, dict) or set(overlay) != {
                "source",
                "destination",
                "component",
            }:
                raise ExportError("component overlays require source, destination, and component")
            source_path = str(_safe_relative(overlay["source"]))
            destination = str(_safe_relative(overlay["destination"]))
            component_path = str(_safe_relative(overlay["component"]))
            entry = by_source_path.get(source_path)
            component = by_component_path.get(component_path)
            if entry is None or entry.mode == "160000":
                raise ExportError("a component-overlay source is absent from the source commit")
            if component is None:
                raise ExportError("a component-overlay owner was not materialized")
            if not (
                destination.startswith(component_path + "/")
                and destination != component_path
            ):
                raise ExportError("a component-overlay destination is outside its owner")
            if destination in self.destinations:
                raise ExportError("a component-overlay destination already exists")
            relative = destination[len(component_path) + 1 :]
            record = self._write_entry(
                self.source_repo,
                entry,
                tree_root,
                destination,
                component,
                component_relative_path=relative,
            )
            component.files.append(record)

        expected = {
            item["path"]: (_canonical_git_url(item["url"]), item["commit"])
            for item in self.manifest.get("required_nested_components", [])
        }
        actual = {
            component.path: (_canonical_git_url(component.url or ""), component.commit)
            for component in self.components
            if component.url is not None
        }
        for path, identity in expected.items():
            if actual.get(path) != identity:
                raise ExportError("a required nested dependency pin was not materialized")
        unmatched_sanitizers = [
            pattern for pattern, count in self.sanitize_matches.items() if not count
        ]
        if unmatched_sanitizers:
            raise ExportError("a required text-sanitization path did not match the source graph")
        unmatched_rewrites = set(self.source_rewrites) - self.source_rewrite_matches
        if unmatched_rewrites:
            raise ExportError("a required source rewrite did not match the source graph")
        return source_commit, source_tree

    def _materialize_component(
        self,
        *,
        repo: Path,
        url: str,
        commit: str,
        export_path: str,
        component_id: str,
        parent: str,
        tree_root: Path,
    ) -> None:
        if any(component.path == export_path for component in self.components):
            raise ExportError("duplicate recursive component path")
        resolved_commit, tree = _resolve_commit(repo, commit)
        if resolved_commit != commit:
            raise ExportError("dependency did not resolve to its exact commit")
        component = ComponentRecord(
            component_id=component_id,
            path=export_path,
            url=_canonical_git_url(url),
            commit=resolved_commit,
            tree=tree,
            parent=parent,
        )
        self.components.append(component)
        declaration = self.component_declarations.get(export_path)
        if declaration is None:
            raise ExportError("materialized component lacks an explicit manifest declaration")
        include = tuple(declaration.get("include", ()))
        matched = {pattern: 0 for pattern in include}
        for entry in _git_entries(repo, commit):
            if entry.mode == "160000":
                continue
            if include:
                for pattern in include:
                    if _matches(entry.path, (pattern,)):
                        matched[pattern] += 1
                if not _matches(entry.path, include):
                    continue
            destination = str(PurePosixPath(export_path) / entry.path)
            if self._excluded(destination) or self._excluded(entry.path):
                continue
            record = self._write_entry(
                repo,
                entry,
                tree_root,
                destination,
                component,
                component_relative_path=entry.path,
            )
            component.files.append(record)
        missing = [pattern for pattern, count in matched.items() if not count]
        if missing:
            raise ExportError("a required component allowlist entry did not match its pin")
        for sub_path, sub_url, sub_commit in _submodules(repo, commit):
            canonical_url = _canonical_git_url(sub_url)
            child_export = str(PurePosixPath(export_path) / sub_path)
            child_declaration = self.component_declarations.get(child_export)
            if child_declaration is None:
                rule = "undeclared_nested_component"
                self.exclusion_counts[rule] = self.exclusion_counts.get(rule, 0) + 1
                continue
            if child_declaration["commit"] != sub_commit:
                raise ExportError("nested component gitlink does not match its manifest pin")
            if _canonical_git_url(child_declaration["url"]) != canonical_url:
                raise ExportError("nested component URL does not match its parent declaration")
            child_repo = self.resolver.resolve(canonical_url, sub_commit)
            self._materialize_component(
                repo=child_repo,
                url=canonical_url,
                commit=sub_commit,
                export_path=child_export,
                component_id=f"{component_id}:{sub_path}",
                parent=component_id,
                tree_root=tree_root,
            )

    def _excluded(self, path: str) -> bool:
        if _matches(path, self.binary_allowlist):
            return False
        if _matches(path, self.exclude):
            rule = next(pattern for pattern in self.exclude if _matches(path, (pattern,)))
            self.exclusion_counts[rule] = self.exclusion_counts.get(rule, 0) + 1
            return True
        return False

    def _remove_record(self, destination: str, component: ComponentRecord) -> None:
        self.destinations.remove(destination)
        self.records = [record for record in self.records if record.path != destination]
        component.files = [record for record in component.files if record.path != destination]

    def _write_entry(
        self,
        repo: Path,
        entry: GitEntry,
        tree_root: Path,
        destination: str,
        component: ComponentRecord,
        component_relative_path: str | None = None,
    ) -> FileRecord:
        _safe_relative(destination)
        if destination in self.destinations:
            raise ExportError("two source objects map to the same export path")
        if entry.kind != "blob" or entry.mode not in {"100644", "100755", "120000"}:
            raise ExportError("unsupported Git entry in allowlisted source")
        data = _git(repo, "cat-file", "blob", entry.object_id)
        source_sha256 = _sha256_bytes(data)
        expected_binary_hash = self.binary_hashes.get(destination)
        if expected_binary_hash is not None and source_sha256 != expected_binary_hash:
            raise ExportError("an allowlisted binary does not match its manifest SHA-256")
        rewrite = self.source_rewrites.get(destination)
        if rewrite is not None:
            if entry.mode == "120000":
                raise ExportError("an explicitly rewritten path is a symlink")
            if source_sha256 != rewrite["source_sha256"]:
                raise ExportError("a source rewrite does not match its bound SHA-256")
            old = rewrite["old"].encode()
            if data.count(old) != 1:
                raise ExportError("a source rewrite did not match exactly once")
            data = data.replace(old, rewrite["new"].encode(), 1)
            self.source_rewrite_matches.add(destination)
            self.source_rewrite_records.append(
                SourceRewriteRecord(
                    path=destination,
                    source_sha256=source_sha256,
                    output_sha256=_sha256_bytes(data),
                    rule=rewrite["rule"],
                )
            )
        matching_sanitizers = [
            pattern for pattern in self.text_sanitize if _matches(destination, (pattern,))
        ]
        if matching_sanitizers:
            if entry.mode == "120000":
                raise ExportError("an explicitly sanitized path is a symlink")
            for pattern in matching_sanitizers:
                self.sanitize_matches[pattern] += 1
            data, sanitized_counts = _sanitize_identity_text(data)
            if not sanitized_counts:
                raise ExportError("an explicitly sanitized file had no identity-bearing text")
            self.sanitizations.append(
                SanitizationRecord(
                    path=destination,
                    source_sha256=source_sha256,
                    output_sha256=_sha256_bytes(data),
                    rules=tuple(sorted(sanitized_counts)),
                )
            )
        target = tree_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.mode == "120000":
            try:
                link = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ExportError("non-UTF-8 symlink target") from error
            link_path = PurePosixPath(link)
            combined = PurePosixPath(os.path.normpath(str(PurePosixPath(destination).parent / link_path)))
            if link_path.is_absolute() or str(combined).startswith("../") or combined == PurePosixPath(".."):
                raise ExportError("symlink escapes the public tree")
            target.symlink_to(link)
        else:
            target.write_bytes(data)
            target.chmod(0o755 if entry.mode == "100755" else 0o644)
        record = FileRecord(
            path=destination,
            mode=entry.mode,
            sha256=_sha256_bytes(data),
            component_path=component.path,
            component_relative_path=component_relative_path or entry.path,
        )
        self.destinations.add(destination)
        self.records.append(record)
        return record


def _record_ledger_sha256(records: Iterable[FileRecord]) -> str:
    lines = [
        f"{record.mode} {record.sha256} {record.component_relative_path}\n"
        for record in sorted(records, key=lambda value: value.component_relative_path)
    ]
    return _sha256_bytes("".join(lines).encode())


def _legal_document_name(name: str, stem: str) -> bool:
    """Match conventional legal-document names without matching source code.

    For example, this accepts ``LICENSE``, ``LICENSE.txt``, ``LICENSE-MIT``,
    and embedded names such as ``furo.js.LICENSE.txt`` while rejecting
    ``license_header.txt`` and ``copyright_checker.py``. The latter two are
    notices/source files rather than the terms that license a component.
    """

    return (
        name == stem
        or name.startswith((f"{stem}.", f"{stem}-"))
        or f".{stem}." in name
        or f".{stem}-" in name
    )


def _legal_file_inventory(
    component: ComponentRecord,
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {
        "license_files": [],
        "notice_files": [],
        "attribution_files": [],
        "patent_files": [],
    }
    for record in sorted(component.files, key=lambda value: value.component_relative_path):
        path = PurePosixPath(record.component_relative_path)
        name = path.name.lower()
        destination: str | None = None
        kind: str | None = None
        if path.parts and path.parts[0].lower() == "licenses":
            destination = "license_files"
            kind = "scoped_license_notice"
        elif any(_legal_document_name(name, stem) for stem in ("license", "copying")):
            destination = "license_files"
            kind = "license_terms"
        elif _legal_document_name(name, "eula"):
            destination = "license_files"
            kind = "eula_terms"
        elif (
            name == "third_party_notices.md"
            or name.startswith("license_header")
            or any(
                _legal_document_name(name, stem)
                for stem in (
                    "notice",
                    "copyright",
                    "acknowledgement",
                    "acknowledgements",
                    "acknowledgment",
                    "acknowledgments",
                )
            )
        ):
            destination = "notice_files"
            kind = "notice"
        elif any(
            _legal_document_name(name, stem)
            for stem in ("author", "authors", "contributor", "contributors")
        ):
            destination = "attribution_files"
            kind = "attribution"
        elif any(_legal_document_name(name, stem) for stem in ("patent", "patents")):
            destination = "patent_files"
            kind = "patent_notice"
        if destination is not None and kind is not None:
            result[destination].append(
                {
                    "path": record.component_relative_path,
                    "sha256": record.sha256,
                    "kind": kind,
                }
            )
    return result


def _declared_legal_review_findings(
    manifest: dict[str, Any], root: Path
) -> list[Finding]:
    raw_gates = manifest.get("policy", {}).get("legal_review_gates", [])
    if not isinstance(raw_gates, list):
        raise ExportError("legal review gates must be a list")
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for raw_gate in raw_gates:
        if not isinstance(raw_gate, dict) or set(raw_gate) != {"rule", "path"}:
            raise ExportError("each legal review gate must contain only rule and path")
        rule = raw_gate["rule"]
        if not isinstance(rule, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", rule):
            raise ExportError("legal review gate has an invalid rule")
        raw_path = raw_gate["path"]
        if not isinstance(raw_path, str):
            raise ExportError("legal review gate has an invalid path")
        path = str(_safe_relative(raw_path))
        if not (root / path).is_file():
            raise ExportError("legal review gate points to an absent exported file")
        key = (rule, path)
        if key in seen:
            raise ExportError("legal review gate is duplicated")
        seen.add(key)
        findings.append(Finding(rule, path))
    return findings


def _component_inventory(
    builder: ExportBuilder,
    source_commit: str,
    source_tree: str,
) -> tuple[dict[str, Any], list[Finding]]:
    components: list[dict[str, Any]] = []
    blockers: list[Finding] = []
    declared = {
        item["path"]: item
        for item in (
            list(builder.manifest.get("components", []))
            + list(builder.manifest.get("required_nested_components", []))
        )
    }
    for component in sorted(builder.components, key=lambda value: value.path):
        legal_files = _legal_file_inventory(component)
        licenses = legal_files["license_files"]
        top_level = [item for item in licenses if "/" not in item["path"]]
        if not top_level:
            blockers.append(Finding("missing_top_level_license", component.path))
        release_gate = declared.get(component.path, {}).get("release_gate")
        if release_gate:
            blockers.append(Finding("component_release_gate", component.path))
        components.append(
            {
                "id": component.component_id,
                "path": component.path,
                "parent": component.parent,
                "url": component.url,
                "commit": component.commit,
                "git_tree": component.tree,
                "file_count": len(component.files),
                "file_ledger_sha256": _record_ledger_sha256(component.files),
                "license_files": licenses,
                "notice_files": legal_files["notice_files"],
                "attribution_files": legal_files["attribution_files"],
                "patent_files": legal_files["patent_files"],
                "license_interpretation": None,
                "declared_release_gate": release_gate,
                "license_status": (
                    "candidate_files_found_not_interpreted"
                    if licenses
                    else "no_candidate_file_found"
                ),
            }
        )
    binary_assets = [
        {
            "path": record.path,
            "sha256": record.sha256,
            "source_component": record.component_path,
        }
        for record in sorted(builder.records, key=lambda value: value.path)
        if _matches(record.path, builder.binary_allowlist)
    ]
    sanitizations = [
        {
            "path": record.path,
            "source_sha256": record.source_sha256,
            "output_sha256": record.output_sha256,
            "rules": list(record.rules),
        }
        for record in sorted(builder.sanitizations, key=lambda value: value.path)
    ]
    source_rewrites = [
        {
            "path": record.path,
            "source_sha256": record.source_sha256,
            "output_sha256": record.output_sha256,
            "rule": record.rule,
        }
        for record in sorted(builder.source_rewrite_records, key=lambda value: value.path)
    ]
    return (
        {
            "schema_version": 1,
            "layout": "flattened_vendored_sources",
            "history_contract": "one_new_root_commit_no_inherited_git_metadata",
            "binary_allowlist": [
                {"path": path, "sha256": builder.binary_hashes[path]}
                for path in builder.binary_allowlist
            ],
            "binary_assets": binary_assets,
            "text_sanitizations": sanitizations,
            "source_rewrites": source_rewrites,
            "source_commit": source_commit,
            "source_git_tree": source_tree,
            "components": components,
            "license_policy": {
                "interpretation_performed": False,
                "missing_top_level_license_is_release_blocker": True,
                "legal_file_categories": [
                    "license_files",
                    "notice_files",
                    "attribution_files",
                    "patent_files",
                ],
            },
        },
        blockers,
    )


def _iter_tree_files(root: Path) -> Iterator[tuple[str, Path]]:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        if ".git" in PurePosixPath(relative).parts:
            continue
        if path.is_file() or path.is_symlink():
            yield relative, path


def _path_mode(path: Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return "120000"
    if not stat.S_ISREG(mode):
        raise ExportError("public tree contains a non-file filesystem object")
    return "100755" if mode & stat.S_IXUSR else "100644"


def _path_digest(path: Path) -> str:
    if path.is_symlink():
        return _sha256_bytes(os.readlink(path).encode())
    return _sha256_file(path)


def audit_tree(root: Path, binary_allowlist: Iterable[str] = ()) -> list[Finding]:
    findings: set[Finding] = set()
    for relative, path in _iter_tree_files(root):
        parts = PurePosixPath(relative).parts
        lower_parts = tuple(part.lower() for part in parts)
        lower_relative = relative.lower()
        lower_name = PurePosixPath(relative).name.lower()
        suffixes = {suffix.lower() for suffix in PurePosixPath(relative).suffixes}
        if ".git" in parts or lower_name == ".gitmodules":
            findings.add(Finding("inherited_git_metadata", relative))
        binary_allowed = _matches(relative, binary_allowlist)
        if (
            suffixes & FORBIDDEN_BINARY_SUFFIXES or lower_name.endswith(".so")
        ) and not binary_allowed:
            findings.add(Finding("binary_or_compiled_artifact", relative))
        if lower_name in {"agents.md", "claude.md", "gemini.md"}:
            findings.add(Finding("internal_agent_document", relative))
        if any(part in INTERNAL_AGENT_DIRECTORIES for part in lower_parts):
            findings.add(Finding("internal_agent_document", relative))
        if any(
            fragment in lower_relative for fragment in CLUSTER_CONTROL_PATH_FRAGMENTS
        ):
            findings.add(Finding("cluster_control_identity", relative))
        if (
            lower_name == ".metadata"
            or lower_name.startswith("checkpoint_manifest.")
            or lower_name.startswith("checkpoint-manifest.")
            or lower_name.startswith("checkpoint_metadata.")
            or lower_name.startswith("checkpoint-metadata.")
        ):
            findings.add(Finding("checkpoint_metadata", relative))
        if path.is_symlink():
            if not path.exists():
                findings.add(Finding("broken_symlink", relative))
            continue
        if binary_allowed:
            # The builder copied this exact Git blob and records its digest in
            # both the component inventory and global file ledger.  Binary
            # content is not decoded or searched as text.
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            findings.add(Finding("binary_payload", relative))
            continue
        for rule, pattern in SECRET_RULES + IDENTITY_RULES:
            if pattern.search(data):
                findings.add(Finding(rule, relative))
    return sorted(findings)


def _environment_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    public_contract = root / "release" / "environment.json"
    if public_contract.is_file():
        try:
            contract = _load_json(public_contract)
        except (ValueError, json.JSONDecodeError):
            return [Finding("invalid_environment_contract", "release/environment.json")]
        observed = contract.get("observed_production_contract", {})
        required = {
            "container",
            "container_digest",
            "container_digest_observed_utc",
            "python",
            "pytorch",
            "cuda",
            "cuda_architecture",
        }
        if not isinstance(observed, dict) or set(observed) != required:
            findings.append(Finding("invalid_environment_contract", "release/environment.json"))
        if contract.get("status") != "sealed_reproducible":
            findings.append(Finding("environment_contract_unsealed", "release/environment.json"))
        if not observed.get("container_digest"):
            findings.append(Finding("immutable_image_digest_missing", "release/environment.json"))
        lock = contract.get("dependency_lock")
        if not isinstance(lock, dict) or set(lock) != {"path", "sha256"}:
            findings.append(Finding("coherent_dependency_lock_missing", "release/environment.json"))
        else:
            lock_path = root / str(lock["path"])
            if not lock_path.is_file():
                findings.append(Finding("coherent_dependency_lock_missing", str(lock["path"])))
            elif not re.fullmatch(r"[0-9a-f]{64}", str(lock["sha256"])):
                findings.append(Finding("invalid_dependency_lock_hash", "release/environment.json"))
            elif _sha256_file(lock_path) != lock["sha256"]:
                findings.append(Finding("dependency_lock_hash_mismatch", str(lock["path"])))
        return findings
    version_path = root / ".python-version"
    project_path = root / "pyproject.toml"
    lock_path = root / "uv.lock"
    if not (version_path.is_file() and project_path.is_file() and lock_path.is_file()):
        findings.append(Finding("incomplete_environment_contract", "."))
        return findings
    version = version_path.read_text().strip()
    project = project_path.read_text()
    lock = lock_path.read_text()
    project_match = re.search(r'(?m)^requires-python\s*=\s*["\x27]([^"\x27]+)', project)
    lock_match = re.search(r'(?m)^requires-python\s*=\s*["\x27]([^"\x27]+)', lock)
    if not project_match or not lock_match:
        findings.append(Finding("unverifiable_python_contract", "pyproject.toml"))
    else:
        major_minor = ".".join(version.split(".")[:2])
        project_req = project_match.group(1)
        lock_req = lock_match.group(1)
        if major_minor not in project_req or major_minor not in lock_req or project_req != lock_req:
            findings.append(Finding("python_environment_version_conflict", "pyproject.toml"))
    return findings


def _write_audit(
    root: Path,
    *,
    unsafe_findings: list[Finding],
    release_blockers: list[Finding],
    exclusion_counts: dict[str, int],
    sanitizations: Iterable[SanitizationRecord] = (),
) -> dict[str, Any]:
    sanitization_counts: dict[str, int] = {}
    sanitized_paths: list[dict[str, Any]] = []
    for record in sorted(sanitizations, key=lambda value: value.path):
        sanitized_paths.append({"path": record.path, "rules": list(record.rules)})
        for rule in record.rules:
            sanitization_counts[rule] = sanitization_counts.get(rule, 0) + 1
    payload = {
        "schema_version": 1,
        "status": "pass" if not unsafe_findings and not release_blockers else "blocked",
        "safe_content_status": "pass" if not unsafe_findings else "blocked",
        "release_blocker_count": len(release_blockers),
        "release_blockers": [finding.__dict__ for finding in sorted(set(release_blockers))],
        "unsafe_finding_count": len(unsafe_findings),
        "unsafe_findings": [finding.__dict__ for finding in sorted(set(unsafe_findings))],
        "excluded_counts_by_rule": dict(sorted(exclusion_counts.items())),
        "sanitized_file_counts_by_rule": dict(sorted(sanitization_counts.items())),
        "sanitized_paths": sanitized_paths,
        "matched_values_recorded": False,
    }
    _write_json(root / AUDIT_REPORT, payload)
    return payload


def _write_file_ledger(root: Path) -> str:
    lines: list[str] = []
    for relative, path in _iter_tree_files(root):
        if relative == str(FILE_LEDGER):
            continue
        lines.append(f"{_path_digest(path)}  {relative}\n")
    payload = "".join(lines)
    (root / FILE_LEDGER).write_text(payload)
    (root / FILE_LEDGER).chmod(0o644)
    return _sha256_bytes(payload.encode())


def _parse_file_ledger(root: Path) -> dict[str, str]:
    path = root / FILE_LEDGER
    if not path.is_file():
        raise ExportError("public file ledger is missing")
    result: dict[str, str] = {}
    previous = ""
    for line in path.read_text().splitlines():
        if not re.fullmatch(r"[0-9a-f]{64}  .+", line):
            raise ExportError("public file ledger has an invalid line")
        digest, relative = line.split("  ", 1)
        _safe_relative(relative)
        if relative <= previous or relative in result or relative == str(FILE_LEDGER):
            raise ExportError("public file ledger is not strictly sorted and unique")
        previous = relative
        result[relative] = digest
    return result


def verify_export_tree(root: Path, allow_blocked: bool = False) -> list[Finding]:
    inventory_path = root / COMPONENT_INVENTORY
    inventory: dict[str, Any] | None = None
    binary_allowlist: list[str] = []
    if inventory_path.is_file():
        try:
            inventory = _load_json(inventory_path)
            binary_allowlist = [
                item["path"] for item in inventory.get("binary_allowlist", [])
            ]
        except (ValueError, TypeError, json.JSONDecodeError):
            inventory = None
    findings: set[Finding] = set(audit_tree(root, binary_allowlist))
    try:
        ledger = _parse_file_ledger(root)
    except ExportError:
        return sorted(findings | {Finding("invalid_file_ledger", str(FILE_LEDGER))})
    for relative, expected in ledger.items():
        path = root / relative
        if not (path.is_file() or path.is_symlink()) or _path_digest(path) != expected:
            findings.add(Finding("file_ledger_mismatch", relative))
    actual = {
        relative for relative, _ in _iter_tree_files(root) if relative != str(FILE_LEDGER)
    }
    for relative in actual - set(ledger):
        findings.add(Finding("unlisted_export_file", relative))

    if not inventory_path.is_file():
        findings.add(Finding("component_inventory_missing", str(COMPONENT_INVENTORY)))
    elif inventory is None:
        findings.add(Finding("component_inventory_invalid", str(COMPONENT_INVENTORY)))
    else:
        try:
            allowlisted: dict[str, str] = {}
            for item in inventory.get("binary_allowlist", []):
                relative = item["path"]
                digest = item["sha256"]
                _safe_relative(relative)
                if relative in allowlisted:
                    findings.add(Finding("duplicate_binary_allowlist_entry", relative))
                allowlisted[relative] = digest
                if (
                    not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or ledger.get(relative) != digest
                ):
                    findings.add(Finding("binary_allowlist_digest_mismatch", relative))

            assets: dict[str, str] = {}
            for item in inventory.get("binary_assets", []):
                relative = item["path"]
                digest = item["sha256"]
                _safe_relative(relative)
                if relative in assets:
                    findings.add(Finding("duplicate_binary_asset_entry", relative))
                assets[relative] = digest
                if (
                    not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or ledger.get(relative) != digest
                ):
                    findings.add(Finding("binary_asset_digest_mismatch", relative))
            for relative in set(allowlisted) ^ set(assets):
                findings.add(Finding("binary_inventory_membership_mismatch", relative))

            sanitized_seen: set[str] = set()
            for item in inventory.get("text_sanitizations", []):
                relative = item["path"]
                _safe_relative(relative)
                if relative in sanitized_seen:
                    findings.add(Finding("duplicate_sanitization_record", relative))
                sanitized_seen.add(relative)
                if ledger.get(relative) != item.get("output_sha256"):
                    findings.add(Finding("sanitization_output_mismatch", relative))
                if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("source_sha256", ""))):
                    findings.add(Finding("sanitization_source_hash_invalid", relative))
                if not item.get("rules"):
                    findings.add(Finding("sanitization_rules_missing", relative))
            for component in inventory.get("components", []):
                prefix = component["path"]
                records: list[tuple[str, str]] = []
                for relative, digest in ledger.items():
                    if prefix == ".":
                        nested = [
                            item["path"]
                            for item in inventory["components"]
                            if item["path"] != "."
                        ]
                        if any(relative == value or relative.startswith(value + "/") for value in nested):
                            continue
                        component_relative = relative
                    elif relative.startswith(prefix + "/"):
                        component_relative = relative[len(prefix) + 1 :]
                        child_paths = [
                            item["path"]
                            for item in inventory["components"]
                            if item.get("parent") == component.get("id")
                        ]
                        if any(relative == value or relative.startswith(value + "/") for value in child_paths):
                            continue
                    else:
                        continue
                    if relative in {str(COMPONENT_INVENTORY), str(AUDIT_REPORT)}:
                        continue
                    records.append(
                        (
                            component_relative,
                            f"{_path_mode(root / relative)} {digest} {component_relative}\n",
                        )
                    )
                fingerprint = _sha256_bytes(
                    "".join(line for _, line in sorted(records)).encode()
                )
                if fingerprint != component.get("file_ledger_sha256"):
                    findings.add(Finding("component_ledger_mismatch", prefix))
                if len(records) != component.get("file_count"):
                    findings.add(Finding("component_file_count_mismatch", prefix))
        except (ExportError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            findings.add(Finding("component_inventory_invalid", str(COMPONENT_INVENTORY)))

    audit_path = root / AUDIT_REPORT
    if not audit_path.is_file():
        findings.add(Finding("public_audit_missing", str(AUDIT_REPORT)))
    else:
        try:
            audit = _load_json(audit_path)
            if audit.get("safe_content_status") != "pass":
                findings.add(Finding("public_audit_unsafe", str(AUDIT_REPORT)))
            if not allow_blocked and audit.get("status") != "pass":
                findings.add(Finding("public_audit_blocked", str(AUDIT_REPORT)))
        except (ValueError, json.JSONDecodeError):
            findings.add(Finding("public_audit_invalid", str(AUDIT_REPORT)))
    return sorted(findings)


def _init_clean_history(root: Path, manifest: dict[str, Any]) -> str:
    clean = manifest["clean_commit"]
    epoch = int(manifest["archive_epoch"])
    _run(("git", "init", "--quiet", "--initial-branch=main", str(root)))
    # Keep this one-shot repository loose until the bundle is written.  Git's
    # automatic maintenance and multithreaded pack delta search otherwise make
    # bundle container bytes vary across identical builds.
    _git(root, "config", "gc.auto", "0")
    _git(root, "config", "user.name", clean["author_name"])
    _git(root, "config", "user.email", clean["author_email"])
    _git(root, "config", "commit.gpgSign", "false")
    # The source repository's ignore rules are useful to developers but must
    # not silently erase manifest-selected files (for example route `.env`
    # presets) from the new clean history.
    _git(root, "add", "-f", "--all")
    env = dict(os.environ)
    stamp = f"{epoch} +0000"
    env.update(
        {
            "GIT_AUTHOR_NAME": clean["author_name"],
            "GIT_AUTHOR_EMAIL": clean["author_email"],
            "GIT_COMMITTER_NAME": clean["author_name"],
            "GIT_COMMITTER_EMAIL": clean["author_email"],
            "GIT_AUTHOR_DATE": stamp,
            "GIT_COMMITTER_DATE": stamp,
            "TZ": "UTC",
        }
    )
    _run(("git", "-C", str(root), "commit", "--quiet", "-m", clean["message"]), env=env)
    count = _git(root, "rev-list", "--count", "HEAD").decode().strip()
    if count != "1":
        raise ExportError("clean export did not produce exactly one commit")
    staged = _git(root, "ls-files", "--stage").decode()
    if any(line.startswith("160000 ") for line in staged.splitlines()):
        raise ExportError("clean export retained a Git submodule entry")
    expected = {
        relative: (_path_mode(path), _path_digest(path))
        for relative, path in _iter_tree_files(root)
    }
    committed: dict[str, tuple[str, str]] = {}
    for entry in _git_entries(root, "HEAD"):
        if entry.kind != "blob":
            raise ExportError("clean history contains a non-blob entry")
        committed[entry.path] = (
            entry.mode,
            _sha256_bytes(_git(root, "cat-file", "blob", entry.object_id)),
        )
    if committed != expected:
        raise ExportError("clean history does not exactly match the audited source tree")
    return _git(root, "rev-parse", "HEAD").decode().strip()


def _write_deterministic_archive(root: Path, destination: Path, prefix: str) -> None:
    tar_data = _git(root, "archive", "--format=tar", f"--prefix={prefix}/", "HEAD")
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(tar_data)


def _write_blocked_output(
    output: Path,
    report: dict[str, Any],
    *,
    inventory: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "BUILD_REPORT.json", report)
    _write_json(output / "COMPONENTS.json", inventory)
    _write_json(output / "PUBLIC_RELEASE_AUDIT.json", audit)


def build_export(
    *,
    source_repo: Path,
    source_revision: str,
    manifest_path: Path,
    output: Path,
    cache: Path,
    repo_map: dict[str, Path] | None = None,
    offline: bool = False,
    allow_release_blockers: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise ExportError("output path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ExportError("unsupported public export manifest version")
    resolver = RepositoryResolver(cache, repo_map, offline)
    temporary = Path(tempfile.mkdtemp(prefix=".mfu-public-export-", dir=output.parent))
    tree = temporary / "source-tree"
    try:
        builder = ExportBuilder(
            source_repo=source_repo,
            source_revision=source_revision,
            manifest=manifest,
            resolver=resolver,
        )
        source_commit, source_tree = builder.build_tree(tree)
        inventory, license_blockers = _component_inventory(builder, source_commit, source_tree)
        _write_json(tree / COMPONENT_INVENTORY, inventory)
        unsafe = audit_tree(tree, builder.binary_allowlist)
        release_blockers = list(license_blockers)
        release_blockers.extend(_declared_legal_review_findings(manifest, tree))
        release_blockers.extend(_environment_findings(tree))
        audit = _write_audit(
            tree,
            unsafe_findings=unsafe,
            release_blockers=release_blockers,
            exclusion_counts=builder.exclusion_counts,
            sanitizations=builder.sanitizations,
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "building",
            "source_commit": source_commit,
            "source_git_tree": source_tree,
            "safe_content_status": audit["safe_content_status"],
            "release_status": audit["status"],
            "component_count": len(inventory["components"]),
            "release_blockers": audit["release_blockers"],
            "unsafe_findings": audit["unsafe_findings"],
            "local_paths_recorded": False,
        }
        if unsafe:
            report["status"] = "blocked_unsafe_content"
            shutil.rmtree(temporary)
            _write_blocked_output(output, report, inventory=inventory, audit=audit)
            return report
        if release_blockers and not allow_release_blockers:
            report["status"] = "blocked_release_gates"
            shutil.rmtree(temporary)
            _write_blocked_output(output, report, inventory=inventory, audit=audit)
            return report

        ledger_sha256 = _write_file_ledger(tree)
        verification = verify_export_tree(tree, allow_blocked=allow_release_blockers)
        if verification:
            raise ExportError("generated public tree failed its own verification")
        clean_commit = _init_clean_history(tree, manifest)
        archive = temporary / f"{manifest['release_name']}.tar.gz"
        bundle = temporary / f"{manifest['release_name']}.bundle"
        _write_deterministic_archive(tree, archive, manifest["release_name"])
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
        if any(path.name == ".git" for path in tree.rglob(".git")):
            raise ExportError("Git metadata survived clean-tree finalization")
        report.update(
            {
                "status": "complete" if not release_blockers else "complete_blocked_not_publishable",
                "clean_commit": clean_commit,
                "commit_count": 1,
                "file_ledger_sha256": ledger_sha256,
                "archive_sha256": _sha256_file(archive),
                "bundle_sha256": _sha256_file(bundle),
                "inherited_git_metadata": False,
            }
        )
        _write_json(temporary / "BUILD_REPORT.json", report)
        os.replace(temporary, output)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parse_repo_map(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ExportError("repository map must use URL=LOCAL_REPOSITORY")
        url, raw_path = value.split("=", 1)
        canonical = _canonical_git_url(url)
        if canonical in result:
            raise ExportError("duplicate repository-map URL")
        result[canonical] = Path(raw_path)
    return result


def _render_report(value: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(f"public_export_status={value['status']}")
        if "verification_status" in value:
            print(f"verification_status={value['verification_status']}")
        if "clean_commit" in value:
            print(f"clean_commit={value['clean_commit']}")
            print(f"archive_sha256={value['archive_sha256']}")
        print(f"safe_content_status={value.get('safe_content_status', 'unknown')}")
        print(f"release_status={value.get('release_status', 'unknown')}")
        print(f"release_blockers={len(value.get('release_blockers', []))}")
        print(f"unsafe_findings={len(value.get('unsafe_findings', []))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="materialize a clean export")
    build.add_argument("--source-repo", type=Path, default=ROOT)
    build.add_argument("--source-revision", default="HEAD")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--cache", type=Path)
    build.add_argument("--repo-map", action="append", default=[], metavar="URL=PATH")
    build.add_argument("--offline", action="store_true")
    build.add_argument("--allow-release-blockers", action="store_true")
    build.add_argument("--json", action="store_true")
    verify = subparsers.add_parser("verify", help="verify a flattened source tree")
    verify.add_argument("--tree", type=Path, required=True)
    verify.add_argument("--allow-blocked", action="store_true")
    verify.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "verify":
            findings = verify_export_tree(args.tree, args.allow_blocked)
            try:
                embedded_audit = _load_json(args.tree / AUDIT_REPORT)
            except (OSError, ValueError, json.JSONDecodeError):
                embedded_audit = {}
            release_blockers = embedded_audit.get("release_blockers", [])
            if not isinstance(release_blockers, list):
                release_blockers = []
            embedded_unsafe = embedded_audit.get("unsafe_findings", [])
            if not isinstance(embedded_unsafe, list):
                embedded_unsafe = []
            integrity_findings = [
                finding
                for finding in findings
                if finding.rule != "public_audit_blocked"
            ]
            verification_status = "pass" if not findings else "blocked"
            safe_content_status = (
                "pass"
                if embedded_audit.get("safe_content_status") == "pass"
                and not embedded_unsafe
                and not integrity_findings
                else "blocked"
            )
            release_status = (
                "pass"
                if embedded_audit.get("status") == "pass" and not release_blockers
                else "blocked"
            )
            report = {
                "status": verification_status,
                "verification_status": verification_status,
                "finding_count": len(findings),
                "findings": [finding.__dict__ for finding in findings],
                "safe_content_status": safe_content_status,
                "release_status": release_status,
                "release_blockers": release_blockers,
                "unsafe_findings": [
                    *embedded_unsafe,
                    *(finding.__dict__ for finding in integrity_findings),
                ],
            }
            _render_report(report, args.json)
            return 0 if not findings else 2
        cache = args.cache or (args.output.parent / ".public-export-git-cache")
        report = build_export(
            source_repo=args.source_repo,
            source_revision=args.source_revision,
            manifest_path=args.manifest,
            output=args.output,
            cache=cache,
            repo_map=_parse_repo_map(args.repo_map),
            offline=args.offline,
            allow_release_blockers=args.allow_release_blockers,
        )
        _render_report(report, args.json)
        return 0 if report["status"] == "complete" else 2
    except (ExportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
