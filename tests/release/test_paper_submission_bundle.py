from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs" / "technical_report" / "prepare_arxiv_submission.py"
SPEC = importlib.util.spec_from_file_location("prepare_arxiv_submission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBMISSION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBMISSION
SPEC.loader.exec_module(SUBMISSION)


def test_overleaf_zip_is_deterministic_and_matches_staged_sources(tmp_path: Path) -> None:
    stage = tmp_path / "source"
    (stage / "sections").mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "main.tex").write_text("\\input{sections/body}\n", encoding="utf-8")
    (stage / "sections" / "body.tex").write_text("paper\n", encoding="utf-8")
    (stage / "figures" / "result.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    SUBMISSION.write_deterministic_overleaf_archive(stage, first)
    SUBMISSION.write_deterministic_overleaf_archive(stage, second)
    SUBMISSION.verify_overleaf_archive(stage, first, build=False)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "figures/result.png",
            "main.tex",
            "sections/body.tex",
        ]
        assert all(not info.is_dir() for info in archive.infolist())
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0)
            for info in archive.infolist()
        )
        assert all(
            info.compress_type == zipfile.ZIP_STORED
            for info in archive.infolist()
        )


@pytest.mark.parametrize(
    "unsafe",
    [Path(".latexmkrc"), Path("build/generated.tex"), Path("main.aux")],
)
def test_submission_bundle_rejects_hidden_or_generated_paths(unsafe: Path) -> None:
    with pytest.raises(ValueError):
        SUBMISSION.validate_paths([Path("main.tex"), unsafe])
