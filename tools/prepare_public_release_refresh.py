#!/usr/bin/env python3
"""Prepare a scrubbed review tree for a later paper/evaluation refresh.

The scientific manuscript lives on a history that is intentionally separate
from the release-source history.  This tool reads both sides through immutable
Git objects, reports their exact identities, and can materialize a *review
tree* containing the candidate paper changes.  It never modifies either input
repository and it never claims that the review tree is a public release.

The final publication boundary remains ``public_clean_export.py``.  A human
must port the reviewed scientific delta to a descendant of the release source,
update the hash-bound binary allowlist, and run the complete export gates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EXPORT_TOOL = ROOT / "tools" / "public_clean_export.py"
PAPER_PREFIX = "docs/technical_report/"


def _load_public_export_module():
    spec = importlib.util.spec_from_file_location(
        "mfu_public_clean_export", PUBLIC_EXPORT_TOOL
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load public export policy")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PUBLIC = _load_public_export_module()


@dataclass(frozen=True)
class GitBlob:
    path: str
    mode: str
    object_id: str


GENERATED_OR_NONPUBLIC_PATTERNS = (
    re.compile(r"^build(?:/|$)"),
    re.compile(r"^fonts(?:/|$)"),
    re.compile(r"^gc(?:/|$)"),
    re.compile(r"^assets/graphcore-symbol\.png$"),
    re.compile(r"^graphcore_report\.sty$"),
    re.compile(r".*\.(?:aux|bbl|blg|fdb_latexmk|fls|log|out|pdf)$"),
)

# These files are valuable in the immutable evidence branch but should not be
# copied into a public source release.  The compact CSV ledgers and public
# methodological descriptions are the publication artifacts.
OPERATIONAL_EVIDENCE_PATTERNS = (
    re.compile(r"(?:^|/)[^/]*receipt[^/]*\.(?:json|md)$", re.I),
    re.compile(r"(?:^|/)COMPLETED\.json$", re.I),
    re.compile(r"(?:^|/)conversion_manifest\.json$", re.I),
    re.compile(r"(?:^|/)canonical-parity\.json$", re.I),
    re.compile(r"(?:^|/)[^/]*manifest[^/]*\.json$", re.I),
    re.compile(r"(?:^|/)[^/]*checkpoint_inventory[^/]*\.csv$", re.I),
    re.compile(r"^claim_validation_experiments\.md$"),
    re.compile(r"^reproducibility\.md$"),
    re.compile(r"^data/README\.md$"),
)

IDENTITY_COLUMNS = {
    "job_id",
    "job_name",
    "owner_uid",
    "run_id",
    "run_name",
    "seed42_job",
    "source_job",
    "source_job_or_lineage",
    "source_pod",
    "source_run_id",
    "source_run_name",
    "source_uri",
    "storage_uri",
    "wandb_entity",
    "wandb_project",
    "wandb_run_id",
    "wandb_run_name",
    "wandb_url",
    "workload_id",
    "workload_name",
}


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    process = subprocess.run(
        ("git", "-C", str(repo), *args),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError("Git object query failed; repository and revision were not accepted")
    return process.stdout


def _resolve(repo: Path, revision: str) -> tuple[str, str]:
    commit = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    tree = _git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}").decode().strip()
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (commit, tree)):
        raise RuntimeError("revision did not resolve to full Git identities")
    return commit, tree


def _paper_tree(repo: Path, commit: str) -> str:
    tree = _git(repo, "rev-parse", "--verify", f"{commit}:{PAPER_PREFIX[:-1]}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise RuntimeError("candidate revision has no paper tree")
    return tree


def _paper_blobs(repo: Path, commit: str) -> list[GitBlob]:
    output = _git(repo, "ls-tree", "-r", "-z", commit, "--", PAPER_PREFIX)
    blobs: list[GitBlob] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        metadata, raw_path = raw.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        path = raw_path.decode()
        if not path.startswith(PAPER_PREFIX):
            raise RuntimeError("paper traversal escaped its prefix")
        relative = path.removeprefix(PAPER_PREFIX)
        PurePosixPath(relative)
        if relative.startswith("/") or ".." in PurePosixPath(relative).parts:
            raise RuntimeError("paper contains an unsafe path")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeError("paper contains a symlink, gitlink, or unsupported mode")
        blobs.append(GitBlob(relative, mode, object_id))
    if not blobs:
        raise RuntimeError("candidate paper tree is empty")
    return sorted(blobs, key=lambda item: item.path)


def _blob(repo: Path, object_id: str) -> bytes:
    return _git(repo, "cat-file", "blob", object_id)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _matches(path: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.fullmatch(path) or pattern.search(path) for pattern in patterns)


def _scan_rules(data: bytes, rules) -> list[str]:
    return sorted(name for name, pattern in rules if pattern.search(data))


def _identity_columns(path: str, data: bytes) -> list[str]:
    if not path.endswith(".csv"):
        return []
    try:
        first = data.decode("utf-8").splitlines()[0]
        fields = next(csv.reader([first]))
    except (UnicodeDecodeError, csv.Error, IndexError, StopIteration):
        return []
    return sorted(field for field in fields if field.lower() in IDENTITY_COLUMNS)


def _remove_identity_columns(data: bytes, columns: Iterable[str]) -> bytes:
    """Drop identity-only CSV columns without retaining their values."""

    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames
    if not fieldnames:
        raise RuntimeError("identity-bearing CSV has no header")
    removed = set(columns)
    kept = [field for field in fieldnames if field not in removed]
    if not kept or not removed <= set(fieldnames):
        raise RuntimeError("identity-bearing CSV schema changed during staging")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=kept, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        writer.writerow({field: row[field] for field in kept})
    return output.getvalue().encode("utf-8")


def _baseline_blob_map(base_repo: Path, base_commit: str) -> dict[str, str]:
    return {blob.path: blob.object_id for blob in _paper_blobs(base_repo, base_commit)}


def audit(
    *, base_repo: Path, base_revision: str, paper_repo: Path, paper_revision: str
) -> tuple[dict[str, object], dict[str, bytes]]:
    base_commit, base_tree = _resolve(base_repo, base_revision)
    paper_commit, paper_repo_tree = _resolve(paper_repo, paper_revision)
    paper_tree = _paper_tree(paper_repo, paper_commit)
    base_paper_tree = _paper_tree(base_repo, base_commit)
    baseline = _baseline_blob_map(base_repo, base_commit)

    candidate_data: dict[str, bytes] = {}
    generated: list[str] = []
    operational: list[str] = []
    binary_figures: list[dict[str, str]] = []
    secrets: list[dict[str, str]] = []
    private: list[dict[str, object]] = []
    csv_identity_schema: list[dict[str, object]] = []

    blobs = _paper_blobs(paper_repo, paper_commit)
    candidate_ids = {blob.path: blob.object_id for blob in blobs}
    for item in blobs:
        data = _blob(paper_repo, item.object_id)
        candidate_data[item.path] = data
        if _matches(item.path, GENERATED_OR_NONPUBLIC_PATTERNS):
            generated.append(item.path)
            continue
        if _matches(item.path, OPERATIONAL_EVIDENCE_PATTERNS):
            operational.append(item.path)
        if item.path.startswith("figures/") and item.path.endswith(".png"):
            binary_figures.append({"path": item.path, "sha256": _sha256(data)})
        secret_rules = _scan_rules(data, PUBLIC.SECRET_RULES)
        for rule in secret_rules:
            secrets.append({"path": item.path, "rule": rule})
        private_rules = _scan_rules(data, PUBLIC.IDENTITY_RULES)
        if private_rules:
            private.append({"path": item.path, "rules": private_rules})
        columns = _identity_columns(item.path, data)
        if columns:
            csv_identity_schema.append({"path": item.path, "columns": columns})

    added = sorted(set(candidate_ids) - set(baseline))
    deleted = sorted(set(baseline) - set(candidate_ids))
    modified = sorted(
        path for path in set(candidate_ids) & set(baseline)
        if candidate_ids[path] != baseline[path]
    )
    status = "blocked_credentials" if secrets else "review_required"
    if not (generated or operational or private or csv_identity_schema or added or deleted or modified):
        status = "no_change"
    report: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "publication_ready": False,
        "base": {
            "commit": base_commit,
            "repository_tree": base_tree,
            "paper_tree": base_paper_tree,
        },
        "candidate": {
            "commit": paper_commit,
            "repository_tree": paper_repo_tree,
            "paper_tree": paper_tree,
        },
        "change_summary": {
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
        },
        "changed_paths": {"added": added, "modified": modified, "deleted": deleted},
        "candidate_file_count": len(blobs),
        "excluded_generated_or_nonpublic_paths": generated,
        "excluded_operational_evidence_paths": operational,
        "private_identity_findings": private,
        "credential_findings": secrets,
        "csv_identity_schemas": csv_identity_schema,
        "paper_figure_allowlist_candidate": binary_figures,
        "matched_values_recorded": False,
        "next_gate": (
            "remove credential-bearing source before staging"
            if secrets
            else "curate scientific delta, refresh hashes, then run public_clean_export.py"
        ),
    }
    return report, candidate_data


def _public_main(data: bytes) -> bytes:
    text = data.decode("utf-8")
    replacements = {
        r"\usepackage{graphcore_report}": r"\usepackage{public_report}",
        "linkcolor=GraphcoreInk": "linkcolor=ReportInk",
        "citecolor=GraphcoreCoralDark": "citecolor=ReportAccentDark",
        "urlcolor=GraphcoreCoralDark": "urlcolor=ReportAccentDark",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError("candidate main.tex no longer matches the reviewed public-style transform")
        text = text.replace(old, new, 1)
    date = r"\date{September 2026}"
    if date not in text:
        raise RuntimeError("candidate main.tex date no longer matches the release contract")
    text = text.replace(
        date,
        "\\date{September 2026\\\\[3mm]\n"
        "\\small \\textcopyright{} 2026 Robert Hu. Original manuscript and figures\n"
        "licensed under \\href{https://creativecommons.org/licenses/by/4.0/}{CC BY 4.0}.}",
        1,
    )
    return text.encode()


def _public_makefile(data: bytes) -> bytes:
    text = data.decode("utf-8")
    marker = "SECTIONS := $(wildcard sections/*.tex) $(wildcard appendices/*.tex)\n"
    if marker not in text:
        raise RuntimeError("candidate Makefile no longer matches the deterministic-build transform")
    if "SOURCE_DATE_EPOCH" not in text:
        text = text.replace(
            marker,
            marker
            + "SOURCE_DATE_EPOCH ?= 1788307200\n"
            + "export SOURCE_DATE_EPOCH\n"
            + "export FORCE_SOURCE_DATE = 1\n",
            1,
        )
    dependency = "graphcore_report.sty \\\n\tassets/graphcore-symbol.png"
    if dependency not in text:
        raise RuntimeError("candidate Makefile no longer matches the public-style dependency transform")
    return text.replace(dependency, "public_report.sty", 1).encode()


def _public_arxiv_builder(data: bytes) -> bytes:
    text = data.decode("utf-8")
    occurrences = text.count('Path("graphcore_report.sty")')
    if occurrences != 2:
        raise RuntimeError("candidate arXiv builder no longer matches the public-style transform")
    return text.replace('Path("graphcore_report.sty")', 'Path("public_report.sty")').encode()


def stage_review(
    *,
    output: Path,
    report: dict[str, object],
    candidate_data: dict[str, bytes],
    base_repo: Path,
    base_commit: str,
) -> dict[str, object]:
    if output.exists():
        raise RuntimeError("output already exists; refusing to overwrite a review tree")
    if report["credential_findings"]:
        raise RuntimeError("credential-shaped content blocks review-tree materialization")
    paper = output / "paper"
    paper.mkdir(parents=True)
    excluded = set(report["excluded_generated_or_nonpublic_paths"])
    excluded.update(report["excluded_operational_evidence_paths"])
    identity_columns = {
        item["path"]: item["columns"] for item in report["csv_identity_schemas"]
    }

    baseline = _baseline_blob_map(base_repo, base_commit)
    public_overlay_paths = {
        "README.md",
        "reproducibility.md",
        "claim_validation_experiments.md",
        "data/README.md",
        "public_report.sty",
    }
    for path, data in sorted(candidate_data.items()):
        if path in excluded or path in public_overlay_paths:
            continue
        if path == "main.tex":
            data = _public_main(data)
        elif path == "Makefile":
            data = _public_makefile(data)
        elif path == "prepare_arxiv_submission.py":
            data = _public_arxiv_builder(data)
        if path in identity_columns:
            data = _remove_identity_columns(data, identity_columns[path])
        if _scan_rules(data, PUBLIC.IDENTITY_RULES):
            sanitized, _ = PUBLIC._sanitize_identity_text(data)
        else:
            sanitized = data
        target = paper / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sanitized)

    for path in sorted(public_overlay_paths):
        object_id = baseline.get(path)
        if object_id is None:
            continue
        target = paper / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_blob(base_repo, object_id))

    # The report is safe metadata: it records paths, rule names, and hashes,
    # never the matched content or a local repository path.
    (output / "PAPER_REFRESH_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    binary_allowlist = {
        f"paper/{item['path']}": item["sha256"]
        for item in report["paper_figure_allowlist_candidate"]
        if (paper / item["path"]).is_file()
    }
    findings = PUBLIC.audit_tree(output, binary_allowlist=binary_allowlist)
    staged = {
        "schema_version": 1,
        "status": "safe_review_tree" if not findings else "blocked_review_tree",
        "publication_ready": False,
        "build_ready": False,
        "build_blockers": [
            "identity-column redaction changes hash-bound scientific inputs; reseal public builders",
            "operational receipts/manifests were omitted; replace required inputs with public summaries",
            "candidate figure hashes require explicit binary-allowlist review",
        ],
        "finding_count": len(findings),
        "findings": [
            {"path": finding.path, "rule": finding.rule} for finding in findings
        ],
        "file_count": sum(1 for path in output.rglob("*") if path.is_file()),
        "matched_values_recorded": False,
    }
    (output / "STAGE_STATUS.json").write_text(
        json.dumps(staged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return staged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "stage"))
    parser.add_argument("--base-repo", type=Path, default=ROOT)
    parser.add_argument("--base-revision", default="HEAD")
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--paper-revision", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_repo = args.base_repo.resolve()
    paper_repo = args.paper_repo.resolve()
    report, data = audit(
        base_repo=base_repo,
        base_revision=args.base_revision,
        paper_repo=paper_repo,
        paper_revision=args.paper_revision,
    )
    result: dict[str, object] = report
    if args.action == "stage":
        if args.output is None:
            raise SystemExit("--output is required for stage")
        base_commit = report["base"]["commit"]
        result = {
            "audit": report,
            "stage": stage_review(
                output=args.output.resolve(),
                report=report,
                candidate_data=data,
                base_repo=base_repo,
                base_commit=base_commit,
            ),
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if args.action == "stage":
            print(f"release_refresh_stage_status={result['stage']['status']}")
        else:
            print(f"release_refresh_audit_status={report['status']}")
        print(f"candidate_commit={report['candidate']['commit']}")
        print(f"candidate_paper_tree={report['candidate']['paper_tree']}")
        print(f"credential_findings={len(report['credential_findings'])}")
        print(f"private_identity_findings={len(report['private_identity_findings'])}")
        print(f"operational_evidence_paths={len(report['excluded_operational_evidence_paths'])}")
    return 2 if report["credential_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
