from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "release_capsule.py"
SPEC = importlib.util.spec_from_file_location("release_capsule", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CAPSULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAPSULE
SPEC.loader.exec_module(CAPSULE)


def _hashed_file(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_external_input_binding_accepts_only_hash_bound_local_files(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    dataset = tmp_path / "dataset.json"
    cursor = tmp_path / "cursor.json"
    for path, content in (
        (tokenizer / "tokenizer.json", b"public tokenizer fixture"),
        (tokenizer / "tokenizer_config.json", b"{}"),
        (tokenizer / "special_tokens_map.json", b"{}"),
        (dataset, b"[]"),
        (cursor, b"{}"),
    ):
        path.write_bytes(content)
    binding = {
        "schema_version": 1,
        "tokenizer": {
            "directory": str(tokenizer),
            "files": {
                name: hashlib.sha256((tokenizer / name).read_bytes()).hexdigest()
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                )
            },
        },
        "dataset": {
            "manifest": _hashed_file(dataset),
            "cursor_contract": _hashed_file(cursor),
        },
        "output": {"directory": str(tmp_path / "output")},
        "authentication": {"mode": "provider_default_chain"},
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(binding))

    assert CAPSULE.validate_inputs(path) == []


def test_external_input_binding_rejects_remote_uri_and_credential_value(tmp_path: Path) -> None:
    binding = {
        "schema_version": 1,
        "tokenizer": {
            "directory": "remote://tokenizer",
            "files": {
                "tokenizer.json": "0" * 64,
                "tokenizer_config.json": "0" * 64,
                "special_tokens_map.json": "0" * 64,
            },
        },
        "dataset": {
            "manifest": {"path": "remote://dataset", "sha256": "0" * 64},
            "cursor_contract": {"path": "remote://cursor", "sha256": "0" * 64},
        },
        "output": {"directory": "remote://output"},
        "api_key": "must-not-be-retained",
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(binding))

    findings = CAPSULE.validate_inputs(path)
    rules = {finding.rule for finding in findings}
    assert "credential_value_field_forbidden" in rules
    assert "remote_uri_forbidden" in rules
    assert "non_local_input_path" in rules
    assert "non_local_output_path" in rules
    assert "must-not-be-retained" not in repr(findings)


def test_withheld_invalid_route_cannot_launch_or_resume() -> None:
    findings = CAPSULE.validate_route("v5-mxfp4-hybrid-27-5-invalid", resume=True)
    rules = {finding.rule for finding in findings}
    assert "withheld_route_rejected" in rules
    assert "route_not_executable" in rules


def test_manifest_records_public_release_boundary_without_exact_replay_claim() -> None:
    manifest = json.loads((ROOT / "release" / "capsule_manifest.json").read_text())
    recipes = json.loads(
        (ROOT / "configs" / "paper" / "llama8b_160b_recipe_contracts.json").read_text()
    )

    assert manifest["status"] == "published_clean_export"
    assert manifest["publication"]["release_commit"] == (
        "1f7b1b1d206e4779dd977771833e0736c8ef4f79"
    )
    assert manifest["publication"]["release_tree"] == (
        "c55184b1b914676faa7633e2ba245dc61f83e675"
    )
    assert manifest["publication"]["release_tag"] == "v0.1.0"
    assert not manifest["publication"]["must_not_publish_current_history"]
    assert not any(
        route["reproduction_level"] == "exact_replay_ready"
        for route in recipes["routes"]
    )


def test_evaluation_overlay_is_separate_and_hash_bound() -> None:
    contract = json.loads((ROOT / "release/evaluation_environment.json").read_text())
    lock = ROOT / contract["requirements"]["path"]
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == contract["requirements"]["sha256"]
    assert contract["training_abi_invariants"]["pytorch"] == "2.9.0a0+145a3a7bda.nv25.10"
    assert contract["canonical_direct_versions"]["lm-eval"] == "0.4.12"
    assert contract["install_mode"].startswith("isolated virtual environment")
