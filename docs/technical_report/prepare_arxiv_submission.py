#!/usr/bin/env python3
"""Build and verify the minimal arXiv source bundle for this report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = PAPER_ROOT / "build" / "arxiv-source"
DEFAULT_ARCHIVE = PAPER_ROOT / "build" / "fp4_training_systems_arxiv.tar.gz"
DEFAULT_OVERLEAF_ARCHIVE = (
    PAPER_ROOT / "build" / "fp4_training_systems_overleaf.zip"
)

INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}")
BIB_RE = re.compile(r"\\bibliography\s*\{([^}]+)\}")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_+.,=-]+$")
FORBIDDEN_SOURCE_PATTERNS = {
    "absolute workspace path": re.compile(r"/(?:workspace|volt/restore/workspace)/"),
    "shell escape": re.compile(r"\\(?:immediate\s*)?write18\b|\\ShellEscape\b"),
    "minted cache dependency": re.compile(r"\\usepackage(?:\[[^]]*\])?\{minted\}"),
}
GENERATED_BUILD_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".lof",
    ".lot",
    ".out",
    ".pdf",
    ".run.xml",
    ".toc",
}
PACKAGING_FORBIDDEN_SUFFIXES = GENERATED_BUILD_SUFFIXES - {".pdf"}


def scan_text_source(relative: Path) -> None:
    text = (PAPER_ROOT / relative).read_text(encoding="utf-8")
    for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"{relative}: contains forbidden {label}")


def resolve_dependency(raw: str, suffixes: tuple[str, ...]) -> Path:
    candidate = Path(raw.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe dependency path: {raw!r}")
    options = [candidate] if candidate.suffix else [candidate.with_suffix(s) for s in suffixes]
    for option in options:
        path = PAPER_ROOT / option
        if path.is_file():
            return option
    raise FileNotFoundError(f"missing dependency {raw!r}; tried {options}")


def collect_dependencies() -> list[Path]:
    pending = [Path("main.tex"), Path("public_report.sty")]
    found: set[Path] = {Path("main.tex"), Path("public_report.sty")}

    while pending:
        relative = pending.pop()
        scan_text_source(relative)
        text = (PAPER_ROOT / relative).read_text(encoding="utf-8")

        dependencies: list[Path] = []
        dependencies.extend(
            resolve_dependency(value, (".tex",)) for value in INPUT_RE.findall(text)
        )
        dependencies.extend(
            resolve_dependency(value, (".pdf", ".png", ".jpg", ".jpeg"))
            for value in GRAPHIC_RE.findall(text)
        )
        for bibliography_group in BIB_RE.findall(text):
            dependencies.extend(
                resolve_dependency(value, (".bib",))
                for value in bibliography_group.split(",")
            )

        for dependency in dependencies:
            if dependency not in found:
                found.add(dependency)
                if dependency.suffix in {".tex", ".sty"}:
                    pending.append(dependency)

    # A matching pre-generated bbl makes reference processing robust on arXiv.
    bbl = Path("main.bbl")
    if not (PAPER_ROOT / bbl).is_file():
        raise FileNotFoundError("main.bbl is missing; run `make` before packaging")
    found.add(bbl)
    paths = sorted(found, key=lambda path: path.as_posix())
    for relative in paths:
        if relative.suffix in {".tex", ".sty", ".bib", ".bbl"}:
            scan_text_source(relative)
    return paths


def validate_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.parts or path.parts[0] == "build":
            raise ValueError(f"generated build path is not a paper source: {path}")
        if any(
            path.name.endswith(suffix) for suffix in PACKAGING_FORBIDDEN_SUFFIXES
        ):
            raise ValueError(f"generated TeX artifact is not a paper source: {path}")
        for component in path.parts:
            if component.startswith(".") or not SAFE_COMPONENT_RE.fullmatch(component):
                raise ValueError(f"arXiv-incompatible path: {path}")


def stage_sources(paths: list[Path], stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    for relative in paths:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER_ROOT / relative, destination)


def write_deterministic_archive(stage: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
                    if not path.is_file():
                        continue
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(stage).as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as source:
                        tar.addfile(info, source)


def _staged_files(stage: Path) -> list[Path]:
    return sorted(
        (path for path in stage.rglob("*") if path.is_file()),
        key=lambda item: item.relative_to(stage).as_posix(),
    )


def write_deterministic_overleaf_archive(stage: Path, archive: Path) -> None:
    """Write a byte-stable, flat-rooted ZIP accepted by Overleaf.

    ZIP_STORED avoids compressor-version drift. Images already dominate this
    bundle and are compressed formats themselves, so the size difference is
    small while the reproducibility contract is stronger.
    """

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, mode="w") as output:
        for path in _staged_files(stage):
            relative = path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.extra = b""
            info.comment = b""
            output.writestr(info, path.read_bytes())


def verify_overleaf_archive(stage: Path, archive: Path, *, build: bool) -> None:
    expected = {
        path.relative_to(stage).as_posix(): path.read_bytes()
        for path in _staged_files(stage)
    }
    if "main.tex" not in expected:
        raise RuntimeError("Overleaf archive would not contain main.tex at its root")

    with zipfile.ZipFile(archive, mode="r") as source:
        infos = source.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("Overleaf archive contains duplicate members")
        if names != sorted(expected):
            raise RuntimeError(
                "Overleaf archive does not exactly match the ordered arXiv source set"
            )
        validate_paths([Path(name) for name in names])
        if any(info.is_dir() or info.flag_bits & 0x1 for info in infos):
            raise RuntimeError("Overleaf archive contains a directory or encrypted member")
        for info in infos:
            if source.read(info) != expected[info.filename]:
                raise RuntimeError(f"Overleaf member differs from staged source: {info.filename}")

        if not build:
            return
        with tempfile.TemporaryDirectory(prefix="fp4-overleaf-check-") as temp_dir:
            extracted = Path(temp_dir) / "source"
            extracted.mkdir()
            for info in infos:
                destination = extracted / info.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read(info))
            verify_clean_build(extracted)


def verify_recorded_dependencies(check_root: Path, declared: set[Path]) -> None:
    recorder = check_root / "main.fls"
    if not recorder.is_file():
        raise RuntimeError("clean staged build did not produce main.fls")
    undeclared: set[Path] = set()
    for line in recorder.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        raw = line.removeprefix("INPUT ").strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = check_root / candidate
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(check_root.resolve())
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or relative in declared:
            continue
        if any(relative.name.endswith(suffix) for suffix in GENERATED_BUILD_SUFFIXES):
            continue
        undeclared.add(relative)
    if undeclared:
        rendered = ", ".join(path.as_posix() for path in sorted(undeclared))
        raise RuntimeError(f"clean staged build read undeclared local inputs: {rendered}")


def verify_clean_build(stage: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fp4-arxiv-check-") as temp_dir:
        check_root = Path(temp_dir) / "source"
        shutil.copytree(stage, check_root)
        declared = {
            path.relative_to(check_root)
            for path in check_root.rglob("*")
            if path.is_file()
        }
        result = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                "-recorder",
                "main.tex",
            ],
            cwd=check_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-80:])
            raise RuntimeError(f"clean staged build failed:\n{tail}")
        log = (check_root / "main.log").read_text(encoding="utf-8", errors="replace")
        fatal_markers = ("LaTeX Error", "Undefined control sequence", "Citation `", "Reference `")
        present = [marker for marker in fatal_markers if marker in log]
        if present:
            raise RuntimeError(f"clean staged build contains unresolved markers: {present}")
        verify_recorded_dependencies(check_root, declared)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--overleaf-archive", type=Path, default=DEFAULT_OVERLEAF_ARCHIVE
    )
    parser.add_argument("--skip-build-check", action="store_true")
    return parser.parse_args()


def normalize_generated_path(path: Path, *, kind: str) -> Path:
    if not path.is_absolute():
        path = PAPER_ROOT / path
    resolved = path.resolve()
    build_root = (PAPER_ROOT / "build").resolve()
    if resolved == build_root or build_root not in resolved.parents:
        raise ValueError(f"{kind} must be a child of {build_root}: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    args.stage = normalize_generated_path(args.stage, kind="stage directory")
    args.archive = normalize_generated_path(args.archive, kind="archive")
    args.overleaf_archive = normalize_generated_path(
        args.overleaf_archive, kind="Overleaf archive"
    )
    if (
        args.stage in args.archive.parents
        or args.stage in args.overleaf_archive.parents
    ):
        raise ValueError("archives must not be placed inside the staged source directory")
    if args.archive == args.overleaf_archive:
        raise ValueError("arXiv and Overleaf archives must use different paths")
    if not args.archive.name.endswith(".tar.gz"):
        raise ValueError("archive name must end in .tar.gz")
    if args.overleaf_archive.suffix != ".zip":
        raise ValueError("Overleaf archive name must end in .zip")
    paths = collect_dependencies()
    validate_paths(paths)
    stage_sources(paths, args.stage)
    write_deterministic_archive(args.stage, args.archive)
    write_deterministic_overleaf_archive(args.stage, args.overleaf_archive)
    verify_overleaf_archive(
        args.stage, args.overleaf_archive, build=not args.skip_build_check
    )
    archive_hash = sha256(args.archive)
    checksum_path = args.archive.with_suffix(args.archive.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {args.archive.name}\n", encoding="utf-8")
    overleaf_hash = sha256(args.overleaf_archive)
    overleaf_checksum = args.overleaf_archive.with_suffix(
        args.overleaf_archive.suffix + ".sha256"
    )
    overleaf_checksum.write_text(
        f"{overleaf_hash}  {args.overleaf_archive.name}\n", encoding="utf-8"
    )
    print(f"staged {len(paths)} source files in {args.stage}")
    print(f"verified archive: {args.archive}")
    print(f"sha256: {archive_hash}")
    print(f"verified Overleaf archive: {args.overleaf_archive}")
    print(f"sha256: {overleaf_hash}")


if __name__ == "__main__":
    main()
