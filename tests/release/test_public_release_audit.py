from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/public_release_audit.py"
SPEC = importlib.util.spec_from_file_location("public_release_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_scanner_never_retains_matched_secret(tmp_path: Path, monkeypatch) -> None:
    tracked = tmp_path / "launch.sh"
    tracked.write_text(
        "set -x\n"
        "WANDB_API_KEY=" + "a" * 40 + "\n"
        "export HF_TOKEN=hf_" + "b" * 40 + "\n"
    )
    monkeypatch.setattr(AUDIT, "tracked_paths", lambda _repo: [tracked.name])

    findings, count, skipped = AUDIT.scan_tip(tmp_path, 1024 * 1024)

    assert count == 1
    assert skipped == 0
    assert {finding.rule for finding in findings} == {
        "huggingface_token_value",
        "wandb_key_assignment",
        "xtrace_with_secret_environment",
    }
    rendered = repr(findings)
    assert "a" * 40 not in rendered
    assert "hf_" + "b" * 40 not in rendered


def test_path_rules_identify_non_distributable_assets(tmp_path: Path, monkeypatch) -> None:
    paths = [
        ".DS_Store",
        "analysis_outputs/private-capsule/result.json",
        "profiling_dumps/trace.json",
        "scripts/kubernetes/render.py",
        "TransformerEngine_v29_backup/binary.so.broken",
        "docs/technical_report/fonts/private.otf",
        "safe/source.py",
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pass\n")
    monkeypatch.setattr(AUDIT, "tracked_paths", lambda _repo: paths)

    findings, count, _ = AUDIT.scan_tip(tmp_path, 1024 * 1024)

    assert count == 7
    rules = {(finding.rule, finding.path) for finding in findings}
    assert ("macos_metadata", ".DS_Store") in rules
    assert (
        "vendored_transformer_engine_backup",
        "TransformerEngine_v29_backup/binary.so.broken",
    ) in rules
    assert (
        "bundled_proprietary_font",
        "docs/technical_report/fonts/private.otf",
    ) in rules
    assert (
        "historical_analysis_artifact",
        "analysis_outputs/private-capsule/result.json",
    ) in rules
    assert ("profiling_artifact", "profiling_dumps/trace.json") in rules
    assert ("cluster_control_debris", "scripts/kubernetes/render.py") in rules
    assert not any(path == "safe/source.py" for _, path in rules)
