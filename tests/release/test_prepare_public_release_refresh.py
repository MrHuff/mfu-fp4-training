from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "prepare_public_release_refresh.py"
SPEC = importlib.util.spec_from_file_location("prepare_public_release_refresh", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFRESH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFRESH
SPEC.loader.exec_module(REFRESH)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _repo(path: Path, files: dict[str, str | bytes]) -> str:
    path.mkdir()
    subprocess.run(("git", "init", "--quiet", "--initial-branch=main", str(path)), check=True)
    _git(path, "config", "user.name", "Fixture")
    _git(path, "config", "user.email", "fixture@example.invalid")
    for relative, value in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if isinstance(value, bytes) else value.encode())
    _git(path, "add", "--all")
    _git(path, "commit", "-m", "fixture")
    return _git(path, "rev-parse", "HEAD")


def _main_tex() -> str:
    return r"""\documentclass{article}
\usepackage{graphcore_report}
\date{September 2026}
\hypersetup{linkcolor=GraphcoreInk,citecolor=GraphcoreCoralDark,urlcolor=GraphcoreCoralDark}
\begin{document}safe\end{document}
"""


def _makefile() -> str:
    return """SECTIONS := $(wildcard sections/*.tex) $(wildcard appendices/*.tex)
report: graphcore_report.sty \\
\tassets/graphcore-symbol.png
\ttrue
"""


def _arxiv() -> str:
    return 'a = Path("graphcore_report.sty")\nb = {Path("graphcore_report.sty")}\n'


def test_audit_and_stage_never_retain_private_values(tmp_path: Path) -> None:
    public_style = "\\ProvidesPackage{public_report}\n"
    private_path = "/" + "work" + "space/private/project"
    private_job = "private-job-123"
    job_field = "job_" + "id"
    base = tmp_path / "base"
    base_commit = _repo(
        base,
        {
            "docs/technical_report/main.tex": _main_tex(),
            "docs/technical_report/Makefile": _makefile(),
            "docs/technical_report/prepare_arxiv_submission.py": _arxiv(),
            "docs/technical_report/public_report.sty": public_style,
            "docs/technical_report/README.md": "public readme\n",
            "docs/technical_report/reproducibility.md": "public reproducibility\n",
        },
    )
    paper = tmp_path / "paper"
    candidate = _repo(
        paper,
        {
            "docs/technical_report/main.tex": _main_tex(),
            "docs/technical_report/Makefile": _makefile(),
            "docs/technical_report/prepare_arxiv_submission.py": _arxiv(),
            "docs/technical_report/README.md": private_path + "\n",
            "docs/technical_report/data/curve.csv": "step,source_run_id,loss\n1,private-run-123,2.0\n",
            "docs/technical_report/data/run/INPUT_RECEIPT.json": json.dumps(
                {job_field: private_job}
            ),
            "docs/technical_report/figures/result.png": b"\x89PNG\r\n\x1a\nfixture",
            "docs/technical_report/fonts/private.otf": b"private font",
        },
    )

    report, data = REFRESH.audit(
        base_repo=base,
        base_revision=base_commit,
        paper_repo=paper,
        paper_revision=candidate,
    )
    assert report["credential_findings"] == []
    assert report["private_identity_findings"] == [
        {"path": "README.md", "rules": ["private_host_path"]},
        {
            "path": "data/run/INPUT_RECEIPT.json",
            "rules": ["static_job_or_run_identity"],
        },
    ]
    assert report["csv_identity_schemas"] == [
        {"path": "data/curve.csv", "columns": ["source_run_id"]}
    ]
    assert "data/run/INPUT_RECEIPT.json" in report["excluded_operational_evidence_paths"]
    assert "fonts/private.otf" in report["excluded_generated_or_nonpublic_paths"]

    output = tmp_path / "review"
    staged = REFRESH.stage_review(
        output=output,
        report=report,
        candidate_data=data,
        base_repo=base,
        base_commit=base_commit,
    )
    serialized = b"\n".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert private_path.encode() not in serialized
    assert private_job.encode() not in serialized
    assert not (output / "paper/fonts/private.otf").exists()
    assert not (output / "paper/data/run/INPUT_RECEIPT.json").exists()
    assert (output / "paper/public_report.sty").read_text() == public_style
    assert staged["status"] == "safe_review_tree"


def test_credential_shaped_content_blocks_stage(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base_commit = _repo(
        base,
        {"docs/technical_report/main.tex": _main_tex()},
    )
    paper = tmp_path / "paper"
    token = "ASIA" + "A" * 16
    candidate = _repo(
        paper,
        {"docs/technical_report/main.tex": _main_tex() + f"\n% {token}\n"},
    )
    report, data = REFRESH.audit(
        base_repo=base,
        base_revision=base_commit,
        paper_repo=paper,
        paper_revision=candidate,
    )
    assert report["status"] == "blocked_credentials"
    assert token not in repr(report)
    try:
        REFRESH.stage_review(
            output=tmp_path / "blocked",
            report=report,
            candidate_data=data,
            base_repo=base,
            base_commit=base_commit,
        )
    except RuntimeError as error:
        assert "credential-shaped" in str(error)
    else:
        raise AssertionError("credential-bearing paper must not be staged")
