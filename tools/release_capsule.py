#!/usr/bin/env python3
"""Fail-closed checks for the public cold-start continuation capsule.

This tool does not fetch dependencies, copy data, inspect credential values,
or submit work. It verifies local code identities and externally supplied,
hash-bound input files. Findings contain rule names and public repository paths;
input contents are never printed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


# Verifying a sealed tree must not create an unlisted cache file inside it.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "release" / "capsule_manifest.json"
RECIPES_PATH = ROOT / "configs" / "paper" / "llama8b_160b_recipe_contracts.json"
ROUTE_EXECUTION_PATH = ROOT / "configs" / "paper" / "route_execution.json"
PUBLIC_COMPONENTS_PATH = ROOT / "release" / "components.json"
PUBLIC_EXPORT_TOOL = ROOT / "tools" / "public_clean_export.py"


@dataclass(frozen=True, order=True)
class Finding:
    rule: str
    subject: str
    detail: str


def _run_git(repo: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if check and process.returncode:
        raise RuntimeError(f"git command failed in {repo}: {' '.join(args)}")
    return process.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _gitlink(repo: Path, revision: str, path: str) -> str | None:
    line = _run_git(repo, "ls-tree", revision, "--", path, check=False)
    if not line:
        return None
    metadata, actual_path = line.split("\t", 1)
    mode, kind, object_id = metadata.split()
    if actual_path != path or mode != "160000" or kind != "commit":
        return None
    return object_id


def _submodule_urls(repo: Path) -> dict[str, str]:
    output = _run_git(
        repo,
        "config",
        "--file",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.url$",
        check=False,
    )
    result: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        key, url = line.split(maxsplit=1)
        name = key[len("submodule.") : -len(".url")]
        result[name] = url
    return result


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    process = subprocess.run(
        ("git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.returncode == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_keys(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            yield path, child
            yield from _flatten_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten_keys(child, f"{prefix}[{index}]")


def validate_source(manifest: dict[str, Any], allow_staging: bool) -> list[Finding]:
    findings: list[Finding] = []
    source = manifest["training_source"]
    required = source["route_complete_commit"]
    head = _run_git(ROOT, "rev-parse", "HEAD")
    if not _is_ancestor(ROOT, required, head):
        findings.append(Finding("missing_route_complete_source", "HEAD", required))
    required_tree = _run_git(ROOT, "rev-parse", f"{required}^{{tree}}", check=False)
    if required_tree != source["route_complete_tree"]:
        findings.append(
            Finding(
                "route_complete_tree_mismatch",
                required,
                f"expected {source['route_complete_tree']}; found {required_tree or 'missing'}",
            )
        )
    for item in manifest["source_inventory"]:
        actual = _run_git(ROOT, "rev-parse", f"{required}:{item['path']}", check=False)
        if actual != item["tree"]:
            findings.append(
                Finding(
                    "source_tree_mismatch",
                    item["path"],
                    f"expected {item['tree']}; found {actual or 'missing'}",
                )
            )

    components = {item["id"]: item for item in manifest["components"]}
    urls = _submodule_urls(ROOT)
    for component_id in ("torchtitan", "transformer_engine", "fp4_runtime"):
        component = components[component_id]
        actual = _gitlink(ROOT, "HEAD", component["path"])
        if actual != component["commit"]:
            findings.append(
                Finding(
                    "component_gitlink_mismatch",
                    component["path"],
                    f"expected {component['commit']}; found {actual or 'missing'}",
                )
            )

    expected_urls = {
        "torchtitan": components["torchtitan"]["url"],
        "TransformerEngine": components["transformer_engine"]["url"],
        "fp4_runtime": components["fp4_runtime"]["url"],
    }
    for name, expected in expected_urls.items():
        actual = urls.get(name)
        if actual != expected:
            findings.append(
                Finding(
                    "component_url_mismatch",
                    name,
                    f"expected {expected}; found {actual or 'missing'}",
                )
            )

    recipes = _load_json(RECIPES_PATH)
    if recipes.get("document_status") != "specification_not_executable":
        findings.append(
            Finding("unexpected_recipe_document_status", str(RECIPES_PATH.relative_to(ROOT)), "review")
        )
    route_ids = [route.get("id") for route in recipes.get("routes", [])]
    if not route_ids or len(route_ids) != len(set(route_ids)):
        findings.append(Finding("invalid_recipe_ids", str(RECIPES_PATH.relative_to(ROOT)), "missing or duplicate"))
    if any(route.get("reproduction_level") == "exact_replay_ready" for route in recipes.get("routes", [])):
        findings.append(Finding("unsupported_exact_replay_claim", str(RECIPES_PATH.relative_to(ROOT)), "route"))

    if not allow_staging:
        if manifest.get("status") != "release_candidate":
            findings.append(Finding("release_status_blocked", "release/capsule_manifest.json", str(manifest.get("status"))))
        publication = manifest.get("publication", {})
        for key in ("release_commit", "release_tree"):
            if not publication.get(key):
                findings.append(Finding("unsealed_public_export", f"publication.{key}", "missing"))
        environment = manifest.get("environment", {})
        if environment.get("python_status") != "consistent_locked":
            findings.append(Finding("environment_metadata_conflict", "environment.python_status", str(environment.get("python_status"))))
        for component in manifest.get("components", []):
            if component.get("release_gate"):
                findings.append(Finding("component_release_gate_open", component["id"], component["release_gate"]))
        if not manifest.get("paper", {}).get("included_in_staging"):
            findings.append(Finding("authoritative_paper_missing", "paper", "not imported"))
        forbidden_tracked_roots = (
            ".DS_Store",
            "TransformerEngine_v29_backup",
            "analysis_outputs",
            "profiling_dumps",
            "scripts/kubernetes",
            "scripts/nvl72",
        )
        tracked = set(_run_git(ROOT, "ls-files").splitlines())
        for prefix in forbidden_tracked_roots:
            count = sum(path == prefix or path.startswith(prefix + "/") for path in tracked)
            if count:
                findings.append(
                    Finding("public_export_forbidden_content", prefix, f"{count} tracked entries")
                )
    return findings


def validate_runtime(manifest: dict[str, Any], runtime_root: Path | None) -> list[Finding]:
    findings: list[Finding] = []
    if runtime_root is None:
        findings.append(Finding("runtime_not_bound", "fp4_runtime", "pass --runtime-root"))
        return findings
    runtime_root = runtime_root.resolve()
    if not (runtime_root / ".git").exists() and not _run_git(runtime_root, "rev-parse", "--git-dir", check=False):
        findings.append(Finding("runtime_not_git_checkout", "fp4_runtime", "missing checkout"))
        return findings
    runtime_component = next(item for item in manifest["components"] if item["id"] == "fp4_runtime")
    head = _run_git(runtime_root, "rev-parse", "HEAD", check=False)
    if head != runtime_component["commit"]:
        findings.append(
            Finding(
                "runtime_commit_mismatch",
                "fp4_runtime",
                f"expected {runtime_component['commit']}; found {head or 'missing'}",
            )
        )
    if _run_git(runtime_root, "status", "--porcelain", "--untracked-files=no", check=False):
        findings.append(Finding("runtime_tracked_tree_dirty", "fp4_runtime", "tracked changes present"))
    urls = _submodule_urls(runtime_root)
    for component in manifest["fp4_runtime_submodules"]:
        actual = _gitlink(runtime_root, "HEAD", component["path"])
        if actual != component["commit"]:
            findings.append(
                Finding(
                    "runtime_submodule_gitlink_mismatch",
                    component["path"],
                    f"expected {component['commit']}; found {actual or 'missing'}",
                )
            )
        actual_url = urls.get(component["id"], urls.get(component["path"]))
        if actual_url != component["url"]:
            findings.append(
                Finding(
                    "runtime_submodule_url_mismatch",
                    component["path"],
                    f"expected {component['url']}; found {actual_url or 'missing'}",
                )
            )
    for item in manifest["runtime_inventory"]:
        actual = _run_git(runtime_root, "rev-parse", f"HEAD:{item['path']}", check=False)
        if actual != item["tree"]:
            findings.append(
                Finding(
                    "runtime_tree_mismatch",
                    item["path"],
                    f"expected {item['tree']}; found {actual or 'missing'}",
                )
            )
    return findings


def _load_public_export_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mfu_public_clean_export_verifier", PUBLIC_EXPORT_TOOL
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("public export verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module while class decorators execute.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_vendored_source(allow_blocked: bool) -> list[Finding]:
    """Verify a flattened export without relying on inherited Git objects."""
    try:
        verifier = _load_public_export_tool()
        public_findings = verifier.verify_export_tree(ROOT, allow_blocked=allow_blocked)
    except (OSError, RuntimeError, ValueError) as error:
        return [Finding("public_export_verifier_failed", ".", type(error).__name__)]
    return [
        Finding(item.rule, item.path, "vendored source verification")
        for item in public_findings
    ]


def validate_vendored_runtime(runtime_root: Path | None, allow_blocked: bool) -> list[Finding]:
    findings: list[Finding] = []
    expected = (ROOT / "fp4_runtime").resolve()
    if runtime_root is None:
        findings.append(Finding("runtime_not_bound", "fp4_runtime", "pass --runtime-root"))
    elif runtime_root.resolve() != expected:
        findings.append(
            Finding(
                "runtime_path_mismatch",
                "fp4_runtime",
                "flattened release uses its vendored runtime",
            )
        )
    elif not expected.is_dir():
        findings.append(Finding("runtime_missing", "fp4_runtime", "vendored directory absent"))
    try:
        inventory = _load_json(PUBLIC_COMPONENTS_PATH)
        matches = [
            component
            for component in inventory.get("components", [])
            if component.get("id") == "fp4_runtime" and component.get("path") == "fp4_runtime"
        ]
        if len(matches) != 1:
            findings.append(
                Finding("runtime_inventory_missing", "fp4_runtime", "exact component absent")
            )
    except (OSError, ValueError, json.JSONDecodeError):
        findings.append(
            Finding("component_inventory_invalid", "release/components.json", "unreadable")
        )
    findings.extend(validate_vendored_source(allow_blocked))
    return findings


LOCAL_PATH_RE = re.compile(r"^(?![A-Za-z][A-Za-z0-9+.-]*://).+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_KEY_RE = re.compile(r"(?:secret|password|access.?key|session.?token|api.?key)", re.I)


def _validate_hashed_file(label: str, item: Any) -> list[Finding]:
    findings: list[Finding] = []
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        return [Finding("invalid_hashed_file_binding", label, "expected path and sha256")]
    raw_path = item.get("path")
    expected = item.get("sha256")
    if not isinstance(raw_path, str) or not LOCAL_PATH_RE.match(raw_path):
        findings.append(Finding("non_local_input_path", label, "URI or empty path rejected"))
        return findings
    if not isinstance(expected, str) or not SHA256_RE.match(expected):
        findings.append(Finding("invalid_sha256", label, "expected 64 lowercase hex characters"))
        return findings
    path = Path(raw_path).expanduser()
    if not path.is_file():
        findings.append(Finding("external_input_missing", label, "file not found"))
    elif _sha256(path) != expected:
        findings.append(Finding("external_input_hash_mismatch", label, "sha256 mismatch"))
    return findings


TOKENIZER_ASSETS = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}


def _validate_tokenizer_directory(item: Any) -> list[Finding]:
    label = "tokenizer"
    if not isinstance(item, dict) or set(item) != {"directory", "files"}:
        return [Finding("invalid_tokenizer_binding", label, "expected directory and files")]
    raw_directory = item.get("directory")
    files = item.get("files")
    findings: list[Finding] = []
    if not isinstance(raw_directory, str) or not LOCAL_PATH_RE.match(raw_directory):
        return [Finding("non_local_input_path", label, "URI or empty path rejected")]
    if not isinstance(files, dict) or set(files) != TOKENIZER_ASSETS:
        return [Finding("invalid_tokenizer_binding", label, "expected the three pinned HF tokenizer assets")]
    directory = Path(raw_directory).expanduser()
    if not directory.is_dir():
        return [Finding("external_input_missing", label, "directory not found")]
    for name in sorted(TOKENIZER_ASSETS):
        expected = files[name]
        if not isinstance(expected, str) or not SHA256_RE.match(expected):
            findings.append(Finding("invalid_sha256", f"tokenizer.files.{name}", "expected 64 lowercase hex characters"))
            continue
        path = directory / name
        if not path.is_file():
            findings.append(Finding("external_input_missing", f"tokenizer.files.{name}", "file not found"))
        elif _sha256(path) != expected:
            findings.append(Finding("external_input_hash_mismatch", f"tokenizer.files.{name}", "sha256 mismatch"))
    return findings


def validate_inputs(inputs_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    inputs = _load_json(inputs_path)
    allowed_top = {"schema_version", "tokenizer", "dataset", "checkpoint", "output", "authentication"}
    unknown = set(inputs) - allowed_top
    if unknown:
        findings.append(Finding("unknown_input_fields", "inputs", ",".join(sorted(unknown))))
    for key, value in _flatten_keys(inputs):
        if FORBIDDEN_KEY_RE.search(key):
            findings.append(Finding("credential_value_field_forbidden", key, "use provider-default authentication"))
        if isinstance(value, str) and "://" in value:
            findings.append(Finding("remote_uri_forbidden", key, "bind a local authorized input"))
    if inputs.get("schema_version") != 1:
        findings.append(Finding("input_schema_version", "schema_version", "expected 1"))
    findings.extend(_validate_tokenizer_directory(inputs.get("tokenizer")))
    dataset = inputs.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {"manifest", "cursor_contract"}:
        findings.append(Finding("invalid_dataset_binding", "dataset", "expected manifest and cursor_contract"))
    else:
        findings.extend(_validate_hashed_file("dataset.manifest", dataset["manifest"]))
        findings.extend(_validate_hashed_file("dataset.cursor_contract", dataset["cursor_contract"]))
    output = inputs.get("output")
    if not isinstance(output, dict) or set(output) != {"directory"}:
        findings.append(Finding("invalid_output_binding", "output", "expected directory"))
    elif not isinstance(output["directory"], str) or not LOCAL_PATH_RE.match(output["directory"]):
        findings.append(Finding("non_local_output_path", "output.directory", "URI or empty path rejected"))
    checkpoint = inputs.get("checkpoint")
    if checkpoint is not None:
        expected_checkpoint_keys = {"directory", "manifest", "route_id", "step"}
        if not isinstance(checkpoint, dict) or set(checkpoint) != expected_checkpoint_keys:
            findings.append(Finding("invalid_checkpoint_binding", "checkpoint", "unexpected fields"))
        else:
            directory = checkpoint["directory"]
            if not isinstance(directory, str) or not LOCAL_PATH_RE.match(directory):
                findings.append(Finding("non_local_input_path", "checkpoint.directory", "URI rejected"))
            elif not Path(directory).expanduser().is_dir():
                findings.append(Finding("external_input_missing", "checkpoint.directory", "directory not found"))
            findings.extend(_validate_hashed_file("checkpoint.manifest", checkpoint["manifest"]))
            if not isinstance(checkpoint["step"], int) or checkpoint["step"] < 0:
                findings.append(Finding("invalid_checkpoint_step", "checkpoint.step", "expected non-negative integer"))
    authentication = inputs.get("authentication", {"mode": "none"})
    if not isinstance(authentication, dict) or authentication.get("mode") not in {"none", "provider_default_chain"}:
        findings.append(Finding("invalid_authentication_mode", "authentication.mode", "unsupported"))
    elif set(authentication) - {"mode", "required_environment_variable_names"}:
        findings.append(Finding("invalid_authentication_fields", "authentication", "unexpected fields"))
    return findings


def validate_route(route_id: str | None, resume: bool) -> list[Finding]:
    if route_id is None:
        return []
    recipes = _load_json(RECIPES_PATH)
    recorded_routes = {route["id"]: route for route in recipes.get("routes", [])}
    execution = _load_json(ROUTE_EXECUTION_PATH)
    routes = {route["id"]: route for route in execution.get("routes", [])}
    if route_id not in routes:
        return [Finding("unknown_route", route_id, "not in the release route contract")]
    route = routes[route_id]
    findings: list[Finding] = []
    recorded = recorded_routes.get(route_id, {})
    if (
        route.get("execution_status") == "withheld_invalid"
        or recorded.get("reproduction_level") == "withheld_invalid"
    ):
        findings.append(
            Finding(
                "withheld_route_rejected",
                route_id,
                str(route.get("paper_evidence_status", recorded.get("result_status"))),
            )
        )
    if not route.get("fresh_start_allowed", False):
        findings.append(Finding("route_not_executable", route_id, "fresh launch forbidden"))
    if resume and not route.get("resume_allowed", False):
        findings.append(Finding("route_resume_contract_unproven", route_id, "resume forbidden"))
    if not route.get("environment") or not route.get("converters"):
        findings.append(Finding("route_not_executable", route_id, "missing environment or converters"))
    return findings


def render(findings: list[Finding], as_json: bool) -> None:
    unique = sorted(set(findings))
    if as_json:
        print(
            json.dumps(
                {
                    "status": "pass" if not unique else "blocked",
                    "finding_count": len(unique),
                    "findings": [finding.__dict__ for finding in unique],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(f"release_capsule_status={'pass' if not unique else 'blocked'}")
    for finding in unique:
        print(f"{finding.rule}: {finding.subject}: {finding.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor",))
    parser.add_argument("--phase", choices=("source", "runtime", "inputs", "all"), default="all")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--route")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-staging", action="store_true", help="validate structure without requiring publication seal")
    parser.add_argument(
        "--allow-publication-blockers",
        action="store_true",
        help="for a flattened export, verify safe content while retaining publication blockers",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    public_export = PUBLIC_COMPONENTS_PATH.is_file()
    if args.allow_publication_blockers and not public_export:
        parser.error("--allow-publication-blockers requires a flattened public export")
    manifest = None if public_export else _load_json(MANIFEST_PATH)
    findings: list[Finding] = []
    if args.phase in {"source", "all"}:
        if public_export:
            # The explicit review mode suppresses publication-gate status only;
            # unsafe content and every ledger mismatch remain hard failures.
            findings.extend(validate_vendored_source(args.allow_publication_blockers))
        else:
            assert manifest is not None
            findings.extend(validate_source(manifest, args.allow_staging))
    if args.phase in {"runtime", "all"}:
        if public_export:
            findings.extend(
                validate_vendored_runtime(
                    args.runtime_root, args.allow_publication_blockers
                )
            )
        else:
            assert manifest is not None
            findings.extend(validate_runtime(manifest, args.runtime_root))
    if args.phase in {"inputs", "all"}:
        if args.inputs is None:
            findings.append(Finding("external_inputs_not_bound", "inputs", "pass --inputs"))
        elif not args.inputs.is_file():
            findings.append(Finding("external_inputs_file_missing", "inputs", "file not found"))
        else:
            findings.extend(validate_inputs(args.inputs))
    findings.extend(validate_route(args.route, args.resume))
    render(findings, args.json)
    return 0 if not findings else 2


if __name__ == "__main__":
    sys.exit(main())
