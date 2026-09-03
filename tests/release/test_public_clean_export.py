from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "public_clean_export.py"
REFERENCE_CHECK = ROOT / "tools" / "check_public_references.py"
SPEC = importlib.util.spec_from_file_location("public_clean_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _init_repo(path: Path, files: dict[str, bytes | str]) -> str:
    path.mkdir(parents=True)
    subprocess.run(("git", "init", "--quiet", "--initial-branch=main", str(path)), check=True)
    _git(path, "config", "user.name", "Fixture")
    _git(path, "config", "user.email", "fixture@example.invalid")
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)
    _git(path, "add", "--all")
    env = {
        **dict(__import__("os").environ),
        "GIT_AUTHOR_DATE": "946684800 +0000",
        "GIT_COMMITTER_DATE": "946684800 +0000",
    }
    subprocess.run(
        ("git", "-C", str(path), "commit", "--quiet", "-m", "fixture"),
        check=True,
        env=env,
    )
    return _git(path, "rev-parse", "HEAD")


def _add_gitlink(
    repo: Path,
    *,
    name: str,
    path: str,
    url: str,
    commit: str,
) -> str:
    modules = repo / ".gitmodules"
    previous = modules.read_text() if modules.exists() else ""
    modules.write_text(
        previous
        + f'[submodule "{name}"]\n'
        + f"\tpath = {path}\n"
        + f"\turl = {url}\n"
    )
    _git(repo, "add", ".gitmodules")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},{path}")
    env = {
        **dict(__import__("os").environ),
        "GIT_AUTHOR_DATE": "946684801 +0000",
        "GIT_COMMITTER_DATE": "946684801 +0000",
    }
    subprocess.run(
        ("git", "-C", str(repo), "commit", "--quiet", "-m", f"pin {name}"),
        check=True,
        env=env,
    )
    return _git(repo, "rev-parse", "HEAD")


def _environment() -> dict[str, object]:
    lock = b"fixture-lock\n"
    return {
        "schema_version": 1,
        "status": "sealed_reproducible",
        "observed_production_contract": {
            "container": "registry.example.invalid/training@sha256:" + "1" * 64,
            "container_digest": "sha256:" + "1" * 64,
            "container_digest_observed_utc": "2026-09-02",
            "python": "3.12.3",
            "pytorch": "2.9.0",
            "cuda": "13.0",
            "cuda_architecture": "sm_100a",
        },
        "dependency_lock": {
            "path": "requirements.lock",
            "sha256": hashlib.sha256(lock).hexdigest(),
        },
        "observed_packages_not_an_install_lock": {},
        "release_blockers": [],
        "legacy_environment_files_are_authoritative": False,
    }


def _manifest(
    path: Path,
    *,
    source_commit: str,
    components: list[dict[str, str]],
    nested: list[dict[str, str]],
) -> Path:
    document = {
        "schema_version": 1,
        "release_name": "fixture-release",
        "archive_epoch": 946684800,
        "clean_commit": {
            "author_name": "Fixture builder",
            "author_email": "fixture@example.invalid",
            "message": "Clean fixture export",
        },
        "source": {
            "required_ancestor": source_commit,
            "include": [
                ".gitignore",
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "requirements.lock",
                "src/**",
                "tools/public_clean_export.py",
                "tools/release_capsule.py",
                "scripts/release/bootstrap.sh",
            ],
            "optional_include": [],
            "overlays": [
                {"source": "release/public/README.md", "destination": "README.md"},
                {
                    "source": "release/public/environment.json",
                    "destination": "release/environment.json",
                },
            ],
        },
        "components": components,
        "required_nested_components": nested,
        "policy": {
            "binary_allowlist": {},
            "text_sanitize": [],
            "exclude": [
                ".git",
                ".git/**",
                ".gitmodules",
                "**/.git",
                "**/.git/**",
                "**/.gitmodules",
                "**/.env",
                "**/*.log",
                "**/*.bin",
                "**/.metadata",
                "**/checkpoint*/*.json",
                "**/checkpoint*/**/*.json",
                "**/checkpoint*/*.yaml",
                "**/checkpoint*/**/*.yaml",
                "**/checkpoint*/*.yml",
                "**/checkpoint*/**/*.yml",
                "**/checkpoint_manifest.*",
                "**/checkpoint-manifest.*",
                "**/checkpoint_metadata.*",
                "**/checkpoint-metadata.*",
                "**/AGENTS.md",
            ],
        },
    }
    path.write_text(json.dumps(document))
    return path


def _fixture_graph(tmp_path: Path) -> tuple[Path, str, Path, dict[str, Path]]:
    urls = {
        "torchtitan": "https://github.com/example/torchtitan.git",
        "transformer_engine": "https://github.com/example/transformer-engine.git",
        "fp4_runtime": "https://github.com/example/fp4-runtime.git",
        "kernel": "https://github.com/example/kernel.git",
    }
    kernel = tmp_path / "kernel"
    kernel_commit = _init_repo(
        kernel,
        {"LICENSE": "kernel terms\n", "include/kernel.cuh": "// kernel\n"},
    )
    runtime = tmp_path / "runtime"
    _init_repo(runtime, {"LICENSE": "runtime terms\n", "src/runtime.cu": "// runtime\n"})
    runtime_commit = _add_gitlink(
        runtime,
        name="kernel",
        path="kernel",
        url=urls["kernel"],
        commit=kernel_commit,
    )
    torchtitan = tmp_path / "torchtitan"
    torchtitan_commit = _init_repo(
        torchtitan,
        {"LICENSE": "torchtitan terms\n", "torchtitan/train.py": "pass\n"},
    )
    transformer_engine = tmp_path / "transformer-engine"
    transformer_engine_commit = _init_repo(
        transformer_engine,
        {"LICENSE": "TE terms\n", "transformer_engine/__init__.py": "\n"},
    )

    source = tmp_path / "source"
    source_files = {
        ".gitignore": "*.ignored\n",
        "LICENSE": "project terms\n",
        "THIRD_PARTY_NOTICES.md": "# Third-party notices\n",
        "requirements.lock": b"fixture-lock\n",
        "src/train.py": "pass\n",
        "release/public/README.md": "# Public fixture\n",
        "release/public/environment.json": json.dumps(_environment()),
        "tools/public_clean_export.py": SCRIPT.read_bytes(),
        "tools/release_capsule.py": (ROOT / "tools/release_capsule.py").read_bytes(),
        "scripts/release/bootstrap.sh": (ROOT / "scripts/release/bootstrap.sh").read_bytes(),
    }
    _init_repo(source, source_files)
    pins = [
        ("torchtitan", "torchtitan_submodule", urls["torchtitan"], torchtitan_commit),
        (
            "transformer_engine",
            "TransformerEngine",
            urls["transformer_engine"],
            transformer_engine_commit,
        ),
        ("fp4_runtime", "fp4_runtime", urls["fp4_runtime"], runtime_commit),
    ]
    source_commit = ""
    for name, export_path, url, commit in pins:
        source_commit = _add_gitlink(
            source,
            name=name,
            path=export_path,
            url=url,
            commit=commit,
        )
    components = [
        {"id": name, "path": export_path, "url": url, "commit": commit}
        for name, export_path, url, commit in pins
    ]
    nested = [
        {
            "path": "fp4_runtime/kernel",
            "url": urls["kernel"],
            "commit": kernel_commit,
        }
    ]
    manifest = _manifest(
        tmp_path / "manifest.json",
        source_commit=source_commit,
        components=components,
        nested=nested,
    )
    mapping = {
        urls["torchtitan"]: torchtitan,
        urls["transformer_engine"]: transformer_engine,
        urls["fp4_runtime"]: runtime,
        urls["kernel"]: kernel,
    }
    return source, source_commit, manifest, mapping


def _component_with_legal_files(paths: list[str]) -> object:
    component = EXPORT.ComponentRecord(
        component_id="fixture",
        path=".",
        url=None,
        commit="1" * 40,
        tree="2" * 40,
        parent=None,
    )
    component.files.extend(
        EXPORT.FileRecord(
            path=path,
            mode="100644",
            sha256=hashlib.sha256(path.encode()).hexdigest(),
            component_path=".",
            component_relative_path=path,
        )
        for path in paths
    )
    return component


def test_legal_file_inventory_distinguishes_terms_and_attribution() -> None:
    inventory = EXPORT._legal_file_inventory(
        _component_with_legal_files(
            [
                "LICENSE",
                "LICENSES/CONTENT.md",
                "COPYING.LESSER",
                "EULA.txt",
                "docs/furo.js.LICENSE.txt",
                "NOTICE.md",
                "THIRD_PARTY_NOTICES.md",
                "COPYRIGHT",
                "ACKNOWLEDGEMENTS.md",
                "assets/license_header.txt",
                "AUTHORS",
                "CONTRIBUTORS.md",
                "PATENTS",
                "qa/L0_license/copyright_checker.py",
                "src/license_helper.py",
            ]
        )
    )

    by_category = {
        category: {item["path"]: item["kind"] for item in items}
        for category, items in inventory.items()
    }
    assert by_category["license_files"] == {
        "COPYING.LESSER": "license_terms",
        "EULA.txt": "eula_terms",
        "LICENSE": "license_terms",
        "LICENSES/CONTENT.md": "scoped_license_notice",
        "docs/furo.js.LICENSE.txt": "license_terms",
    }
    assert by_category["notice_files"] == {
        "ACKNOWLEDGEMENTS.md": "notice",
        "COPYRIGHT": "notice",
        "NOTICE.md": "notice",
        "THIRD_PARTY_NOTICES.md": "notice",
        "assets/license_header.txt": "notice",
    }
    assert by_category["attribution_files"] == {
        "AUTHORS": "attribution",
        "CONTRIBUTORS.md": "attribution",
    }
    assert by_category["patent_files"] == {"PATENTS": "patent_notice"}
    assert all(
        "copyright_checker.py" not in item and "license_helper.py" not in item
        for paths in by_category.values()
        for item in paths
    )


def test_declared_legal_review_gates_are_validated(tmp_path: Path) -> None:
    reviewed = tmp_path / "terms" / "EULA.txt"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_text("terms\n")
    manifest = {
        "policy": {
            "legal_review_gates": [
                {"rule": "third_party_eula_review_required", "path": "terms/EULA.txt"}
            ]
        }
    }

    assert EXPORT._declared_legal_review_findings(manifest, tmp_path) == [
        EXPORT.Finding("third_party_eula_review_required", "terms/EULA.txt")
    ]
    manifest["policy"]["legal_review_gates"][0]["path"] = "terms/missing.txt"
    with pytest.raises(EXPORT.ExportError, match="absent exported file"):
        EXPORT._declared_legal_review_findings(manifest, tmp_path)


def test_public_manifest_carries_curated_scope_and_resolved_license_map() -> None:
    manifest = json.loads((ROOT / "release/public_export_manifest.json").read_text())
    assert "THIRD_PARTY_NOTICES.md" in manifest["source"]["include"]
    assert "NOTICE" in manifest["source"]["include"]
    assert "LICENSES/**" in manifest["source"]["include"]
    assert manifest["policy"]["legal_review_gates"] == []
    rewrites = manifest["policy"]["source_rewrites"]
    assert {item["path"] for item in rewrites} == {
        "fp4_runtime/ThunderKittens/kernels/gemm/nvfp4_b200/"
        "localCTA_epilogue/bench_qkv_split3_onepass.py",
        "fp4_runtime/ThunderKittens/kernels/gemm/nvfp4_b200/"
        "localCTA_epilogue_v3/bench_qkv_split3_onepass.py",
    }
    assert all(item["rule"] == "portable_torch_extensions_path" for item in rewrites)
    assert all(
        item["source_sha256"]
        == "e066a8d6f7e30a61e348c90278e2571c4949df0dcbd317f2ccffe3797a16eae7"
        for item in rewrites
    )
    assert all("/root/.cache/torch_extensions" in item["old"] for item in rewrites)
    assert all("TE_FUSED_RMSNORM_EXTENSION" in item["new"] for item in rewrites)
    assert all("TORCH_EXTENSIONS_DIR" in item["new"] for item in rewrites)
    assert all("/root/" not in item["new"] for item in rewrites)
    runtime = next(item for item in manifest["components"] if item["id"] == "fp4_runtime")
    assert "TK_quantisation/mxfp4_v1/**" in runtime["include"]
    assert "TK_quantisation/nvfp4_v6/**" in runtime["include"]
    assert "TK_quantisation/nvfp4_CTA_local_v4/**" in runtime["include"]
    assert "fp4_cce_TK/v4_sparse_correct.cu" in runtime["include"]
    assert "fp4_cross_entropy/**" not in runtime["include"]
    assert "torchtitan_submodule/benchmarks/**" in manifest["policy"]["exclude"]
    nested_paths = {item["path"] for item in manifest["required_nested_components"]}
    assert nested_paths == {
        "TransformerEngine/3rdparty/cudnn-frontend",
        "TransformerEngine/3rdparty/cutlass",
        "fp4_runtime/ThunderKittens",
    }
    thunderkittens = next(
        item
        for item in manifest["required_nested_components"]
        if item["path"] == "fp4_runtime/ThunderKittens"
    )
    assert thunderkittens["include"] == [
        "LICENSE",
        "README.md",
        "include/**",
        "prototype/**",
        "kernels/common.mk",
        "kernels/gemm/common/**",
        "kernels/gemm/mxfp4_gb200/**",
        "kernels/gemm/nvfp4_b200/**",
    ]
    excluded = set(manifest["policy"]["exclude"])
    assert "low_bits_training/evaluation/simple_evals.py" in excluded
    assert "low_bits_training/experiments/mxfp4/quantization/stable_spam.py" in excluded
    assert "torchtitan_submodule/torchtitan/models/flux/**" in excluded
    overlays = manifest["source"]["component_overlays"]
    assert overlays == [
        {
            "source": "release/public/fp4_runtime_LICENSE.txt",
            "destination": "fp4_runtime/LICENSE",
            "component": "fp4_runtime",
        },
        {
            "source": "release/torchtitan_gitlink.txt",
            "destination": "torchtitan_submodule/.lbt_torchtitan_commit",
            "component": "torchtitan_submodule",
        },
    ]
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "LICENSES/PYTORCH.txt" in notice
    assert "LICENSES/META_LLAMA_3.txt" in notice
    assert "LICENSES/APACHE-2.0.txt" in notice
    assert "Graphcore Research `gc-training`" in notice
    assert "fp4_runtime/TK_quantisation/nvfp4*" in notice
    assert "were modified" in notice

    repository_notice = (ROOT / "NOTICE").read_text()
    public_readme_path = ROOT / "release/public/README.md"
    if not public_readme_path.is_file():
        # The clean export promotes this overlay to the repository root.
        public_readme_path = ROOT / "README.md"
    public_readme = public_readme_path.read_text()
    assert "Built with Meta Llama 3" in repository_notice
    assert "Built with Meta Llama 3" in public_readme

    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert "Graphcore Ltd. All rights reserved." not in contributing
    assert "LICENSES/APACHE-2.0.txt" in contributing
    assert "scripts/release/run_gates.sh --cpu-only" in contributing

    portable_sources = (
        ROOT / "tools/check_localcta_cde_exact_boundary.py",
        ROOT / "tools/profile_bwd.py",
        ROOT / "tools/test_fused.py",
    )
    assert all(
        "/root/.cache/torch_extensions" not in path.read_text()
        for path in portable_sources
    )


def test_exact_source_rewrite_is_hash_bound_and_recorded(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    document = json.loads(manifest.read_text())
    document["policy"]["source_rewrites"] = [
        {
            "path": "src/train.py",
            "source_sha256": hashlib.sha256(b"pass\n").hexdigest(),
            "rule": "correct_vendored_license_path",
            "old": "pass",
            "new": "pass  # exact public rewrite",
        }
    ]
    manifest.write_text(json.dumps(document))
    output = tmp_path / "output"

    EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    rewritten = b"pass  # exact public rewrite\n"
    assert (output / "source-tree/src/train.py").read_bytes() == rewritten
    inventory = json.loads((output / "source-tree/release/components.json").read_text())
    assert inventory["source_rewrites"] == [
        {
            "path": "src/train.py",
            "source_sha256": hashlib.sha256(b"pass\n").hexdigest(),
            "output_sha256": hashlib.sha256(rewritten).hexdigest(),
            "rule": "correct_vendored_license_path",
        }
    ]

    document["policy"]["source_rewrites"][0]["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(EXPORT.ExportError, match="bound SHA-256"):
        EXPORT.build_export(
            source_repo=source,
            source_revision=source_commit,
            manifest_path=manifest,
            output=tmp_path / "bad-output",
            cache=tmp_path / "cache",
            repo_map=mapping,
            offline=True,
        )


def test_deterministic_tree_and_archive_with_cloneable_bundles(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    reports = []
    for output in (first, second):
        reports.append(
            EXPORT.build_export(
                source_repo=source,
                source_revision=source_commit,
                manifest_path=manifest,
                output=output,
                cache=tmp_path / "cache",
                repo_map=mapping,
                offline=True,
            )
        )

    assert reports[0]["status"] == "complete"
    assert reports[0]["clean_commit"] == reports[1]["clean_commit"]
    assert reports[0]["archive_sha256"] == reports[1]["archive_sha256"]
    # Single-threaded packing makes same-toolchain builds byte-identical.  The
    # semantic portability contract remains the sealed commit verified below.
    assert reports[0]["bundle_sha256"] == reports[1]["bundle_sha256"]
    assert (first / "source-tree/fp4_runtime/kernel/include/kernel.cuh").is_file()
    assert not list((first / "source-tree").rglob(".git"))
    assert not list((first / "source-tree").rglob(".gitmodules"))
    assert EXPORT.verify_export_tree(first / "source-tree") == []
    inventory = json.loads((first / "source-tree/release/components.json").read_text())
    components = {item["path"]: item for item in inventory["components"]}
    assert set(components) == {
        ".",
        "TransformerEngine",
        "fp4_runtime",
        "fp4_runtime/kernel",
        "torchtitan_submodule",
    }
    assert components["fp4_runtime/kernel"]["commit"] == _git(
        mapping["https://github.com/example/kernel.git"], "rev-parse", "HEAD"
    )
    assert all(item["license_interpretation"] is None for item in components.values())
    assert all(item["license_files"] for item in components.values())
    assert all(
        {"notice_files", "attribution_files", "patent_files"} <= set(item)
        for item in components.values()
    )
    assert {
        item["path"] for item in components["."]["notice_files"]
    } >= {"THIRD_PARTY_NOTICES.md"}

    clones = (tmp_path / "clone-first", tmp_path / "clone-second")
    for output, clone in zip((first, second), clones, strict=True):
        subprocess.run(
            (
                "git",
                "clone",
                "--quiet",
                str(output / "fixture-release.bundle"),
                str(clone),
            ),
            check=True,
        )
        assert _git(clone, "rev-list", "--count", "HEAD") == "1"
        assert _git(clone, "rev-parse", "HEAD") == reports[0]["clean_commit"]
        assert _git(clone, "submodule", "status", "--recursive") == ""

    clone = clones[0]
    subprocess.run(
        (sys.executable, "tools/release_capsule.py", "doctor", "--phase", "source"),
        cwd=clone,
        check=True,
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ("bash", "scripts/release/bootstrap.sh", "--verify-only"),
        cwd=clone,
        check=True,
        stdout=subprocess.PIPE,
    )


def test_component_allowlist_and_scoped_license_overlay(tmp_path: Path) -> None:
    source, _, manifest, mapping = _fixture_graph(tmp_path)
    overlay = source / "release/public/runtime-license.txt"
    overlay.write_text("scoped runtime terms\n")
    _git(source, "add", "-f", "release/public/runtime-license.txt")
    _git(source, "commit", "-m", "add scoped runtime terms")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    document["source"]["component_overlays"] = [
        {
            "source": "release/public/runtime-license.txt",
            "destination": "fp4_runtime/LICENSE",
            "component": "fp4_runtime",
        }
    ]
    runtime = next(item for item in document["components"] if item["id"] == "fp4_runtime")
    runtime["include"] = ["src/**"]
    manifest.write_text(json.dumps(document))
    output = tmp_path / "output"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "complete"
    tree = output / "source-tree"
    assert (tree / "fp4_runtime/src/runtime.cu").is_file()
    assert (tree / "fp4_runtime/LICENSE").read_text() == "scoped runtime terms\n"
    assert (tree / "fp4_runtime/kernel/include/kernel.cuh").is_file()


def test_public_reference_checker_skips_non_export_tree(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[legacy private UI](missing-ui.example)\n")
    result = subprocess.run(
        (sys.executable, str(REFERENCE_CHECK), "--root", str(tmp_path)),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert result.stdout.strip() == "public_reference_check=skipped_non_export_tree"


def test_public_reference_checker_validates_third_party_notice_paths(
    tmp_path: Path,
) -> None:
    for relative in (
        "README.md",
        "configs/paper/README.md",
        "docs/technical_report/README.md",
        "release/components.json",
        "legal/LICENSE.txt",
        "legal/nested/EULA.txt",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n" if relative.endswith(".json") else "fixture\n")
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text(
        "`legal/LICENSE.txt`; `legal/nested/*.txt`; `See LICENSE`\n"
    )

    result = subprocess.run(
        (sys.executable, str(REFERENCE_CHECK), "--root", str(tmp_path)),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert result.stdout.strip() == "public_reference_check=pass"

    (tmp_path / "legal/LICENSE.txt").unlink()
    result = subprocess.run(
        (sys.executable, str(REFERENCE_CHECK), "--root", str(tmp_path)),
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 2
    assert (
        "missing_public_reference=THIRD_PARTY_NOTICES.md -> legal/LICENSE.txt"
        in result.stdout
    )


def test_clean_history_force_adds_manifest_selected_ignored_file(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    ignored = source / "src/required.ignored"
    ignored.write_text("required route preset\n")
    _git(source, "add", "-f", "src/required.ignored")
    _git(source, "commit", "-m", "required ignored fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    manifest.write_text(json.dumps(document))
    output = tmp_path / "output"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "complete"
    clone = tmp_path / "clone"
    subprocess.run(
        ("git", "clone", "--quiet", str(output / "fixture-release.bundle"), str(clone)),
        check=True,
    )
    assert (clone / "src/required.ignored").read_text() == "required route preset\n"
    ledger_paths = {
        line.split("  ", 1)[1]
        for line in (clone / "SHA256SUMS").read_text().splitlines()
    }
    assert set(_git(clone, "ls-files").splitlines()) == ledger_paths | {"SHA256SUMS"}
    subprocess.run(
        ("bash", "scripts/release/bootstrap.sh", "--verify-only"),
        cwd=clone,
        check=True,
        stdout=subprocess.PIPE,
    )


def test_excluded_material_never_enters_tree_or_ledger(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    (source / "src/private.log").write_text("log\n")
    (source / "src/.env").write_text("not-public\n")
    (source / "src/checkpoint_manifest.json").write_text("{}\n")
    (source / "src/checkpoint-123/state.json").parent.mkdir(parents=True)
    (source / "src/checkpoint-123/state.json").write_text("{}\n")
    (source / "src/checkpoint-123/.metadata").write_text("not public\n")
    (source / "src/checkpoint-metadata.yaml").write_text("not: public\n")
    (source / "src/checkpoint_metadata.json").write_text("{}\n")
    (source / "src/checkpoint-manifest.yml").write_text("not: public\n")
    (source / "src/checkpoint_conversion/convert.py").parent.mkdir(parents=True)
    (source / "src/checkpoint_conversion/convert.py").write_text("def convert():\n    pass\n")
    (source / "src/payload.bin").write_bytes(b"\0payload")
    _git(
        source,
        "add",
        "src/private.log",
        "src/.env",
        "src/checkpoint_manifest.json",
        "src/checkpoint-123/state.json",
        "src/checkpoint-123/.metadata",
        "src/checkpoint-metadata.yaml",
        "src/checkpoint_metadata.json",
        "src/checkpoint-manifest.yml",
        "src/checkpoint_conversion/convert.py",
        "src/payload.bin",
    )
    _git(source, "commit", "-m", "excluded fixtures")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    manifest.write_text(json.dumps(document))
    output = tmp_path / "output"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "complete"
    ledger = (output / "source-tree/SHA256SUMS").read_text()
    for name in (
        "private.log",
        ".env",
        "checkpoint_manifest.json",
        "state.json",
        ".metadata",
        "checkpoint-metadata.yaml",
        "checkpoint_metadata.json",
        "checkpoint-manifest.yml",
        "payload.bin",
    ):
        assert name not in ledger
        assert not list((output / "source-tree").rglob(name))
    assert "src/checkpoint_conversion/convert.py" in ledger
    assert (output / "source-tree/src/checkpoint_conversion/convert.py").is_file()


def test_unsafe_included_content_fails_without_retaining_value(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    shaped_value = "ASIA" + "A" * 16
    (source / "src/unsafe.py").write_text(f'value = "{shaped_value}"\n')
    _git(source, "add", "src/unsafe.py")
    _git(source, "commit", "-m", "unsafe fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    manifest.write_text(json.dumps(document))
    output = tmp_path / "blocked"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "blocked_unsafe_content"
    assert not (output / "source-tree").exists()
    serialized = (output / "BUILD_REPORT.json").read_text()
    assert shaped_value not in serialized
    assert "aws_access_key_value" in serialized
    assert (output / "COMPONENTS.json").is_file()
    assert (output / "PUBLIC_RELEASE_AUDIT.json").is_file()


def test_explicit_text_sanitization_is_hash_bound_and_preserves_python_structure(
    tmp_path: Path,
) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    private_root = "/" + "workspace/private-project/cache"
    tracker = "wandb." + "ai/private-team/private-run"
    storage = "s3" + "://private-bucket/private-job/checkpoint"
    original = (
        f'DEFAULT_ROOT = "{private_root}"\n'
        f'TRACKER = "{tracker}"\n'
        f'STORAGE = "{storage}"\n'
        'def add(left: int, right: int) -> int:\n'
        '    """Algorithmic behavior must not change."""\n'
        '    return left + right\n'
    )
    target = source / "src/sanitized.py"
    target.write_text(original)
    _git(source, "add", "src/sanitized.py")
    _git(source, "commit", "-m", "identity-bearing defaults fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    document["policy"]["text_sanitize"] = ["src/sanitized.py"]
    manifest.write_text(json.dumps(document))
    output = tmp_path / "output"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "complete"
    sanitized_path = output / "source-tree/src/sanitized.py"
    sanitized = sanitized_path.read_text()
    compile(sanitized, str(sanitized_path), "exec")
    assert "/" + "workspace/" not in sanitized
    assert "wandb." + "ai/" not in sanitized
    assert "private-project/cache" not in sanitized
    assert "private-bucket/private-job/checkpoint" not in sanitized

    class NormalizeStrings(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value="<text>"), node)
            return node

    original_structure = ast.dump(
        NormalizeStrings().visit(ast.parse(original)), include_attributes=False
    )
    sanitized_structure = ast.dump(
        NormalizeStrings().visit(ast.parse(sanitized)), include_attributes=False
    )
    assert sanitized_structure == original_structure

    inventory = json.loads(
        (output / "source-tree/release/components.json").read_text()
    )
    assert inventory["text_sanitizations"] == [
        {
            "path": "src/sanitized.py",
            "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(sanitized.encode()).hexdigest(),
            "rules": ["private_host_path", "storage_uri", "wandb_identity"],
        }
    ]
    audit = json.loads(
        (output / "source-tree/release/public_release_audit.json").read_text()
    )
    assert audit["safe_content_status"] == "pass"
    assert audit["sanitized_paths"] == [
        {
            "path": "src/sanitized.py",
            "rules": ["private_host_path", "storage_uri", "wandb_identity"],
        }
    ]
    assert EXPORT.verify_export_tree(output / "source-tree") == []


def test_identity_sanitization_preserves_generic_code_and_shell_delimiters(
    tmp_path: Path,
) -> None:
    private_root = b"/" + b"work" + b"space/private-project/cache"
    private_run = b"private-" + b"run-123"
    arrow_suffix = b"/" + b"data/**/*.arrow"
    python_source = (
        b'ROOT = "' + private_root + b'"\n'
        b'ALT = f"' + private_root + b'/{variant}"\n'
        b'ARROW = f"{path.rstrip(chr(47))}' + arrow_suffix + b'"\n'
        b'source_job_id = args.expected_source_job_id\n'
        b'manifest = {"source_job_id": source_job_id}\n'
        b'DOCS = "https://docs.wandb.ai/guides/track/log/customize-logging-axes/"\n'
        b'JOB_NAME = "' + private_run + b'"\n'
    )

    sanitized, counts = EXPORT._sanitize_identity_text(python_source)

    assert counts == {"private_host_path": 2, "static_job_or_run_identity": 1}
    assert b'ROOT = "/opt/mfu/EXTERNAL_PATH"' in sanitized
    assert b'ALT = f"/opt/mfu/EXTERNAL_PATH/{variant}"' in sanitized
    assert arrow_suffix in sanitized
    assert b'manifest = {"source_job_id": source_job_id}' in sanitized
    assert b"https://docs.wandb.ai/" in sanitized
    assert b'JOB_NAME = "EXAMPLE"' in sanitized
    compile(sanitized, "<sanitized fixture>", "exec")
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    (scan_root / "safe.py").write_bytes(sanitized)
    assert EXPORT.audit_tree(scan_root) == []

    object_store = b"s" + b"3://example.invalid/" + b"data/"
    shell_source = b'DATASET_PATH="${DATASET_PATH:-' + object_store + b'}"\n'
    sanitized_shell, shell_counts = EXPORT._sanitize_identity_text(shell_source)
    assert shell_counts == {"storage_uri": 1}
    assert sanitized_shell == b'DATASET_PATH="${DATASET_PATH:-OBJECT_STORE_URI}"\n'
    shell_path = scan_root / "safe.sh"
    shell_path.write_bytes(sanitized_shell)
    subprocess.run(("bash", "-n", str(shell_path)), check=True)
    assert EXPORT.audit_tree(scan_root) == []


def test_public_core_runtime_defaults_use_the_vendored_tree() -> None:
    core_paths = [
        ROOT / "low_bits_training/bridge_mcore_fp4.py",
        ROOT / "low_bits_training/cce/backend.py",
        ROOT / "low_bits_training/quantization/mxfp4_backend.py",
        ROOT / "low_bits_training/quantization/tk_gemm.py",
    ]
    for path in core_paths:
        source = path.read_text()
        assert "fp4_runtime" in source
        assert "/opt/mfu/EXTERNAL_PATH" not in source
    v7 = ROOT / "low_bits_training/quantization/v7_fused_linear.py"
    v7_source = v7.read_text()
    assert "FP4_V7_CSRC" in v7_source
    assert "/opt/mfu/EXTERNAL_PATH" not in v7_source
    for name in ("fused_te_quant_v7.cu", "fused_te_quant_v7_torch.cpp"):
        assert (ROOT / "fp4_runtime/fused_ops/csrc/old_ideas" / name).is_file()
    for name in ("vec.cuh", "utils.cuh"):
        assert (ROOT / "fp4_runtime/fused_ops/csrc" / name).is_file()
    assert "extra_include_paths=[CSRC, include_root]" in v7_source
    manifest = json.loads((ROOT / "release/public_export_manifest.json").read_text())
    runtime = next(item for item in manifest["components"] if item["id"] == "fp4_runtime")
    assert "fused_ops/csrc/old_ideas/fused_te_quant_v7.cu" in runtime["include"]
    assert "fused_ops/csrc/old_ideas/fused_te_quant_v7_torch.cpp" in runtime["include"]


def test_archival_runtime_drivers_have_portable_configurable_roots() -> None:
    paths = [
        ROOT / "fp4_runtime/TK_quantisation/nvfp4_v2/test_v2_vs_v1.py",
        ROOT / "fp4_runtime/TK_quantisation/nvfp4_v3/bench_v3_vs_v2.py",
        ROOT / "fp4_runtime/TK_quantisation/nvfp4_v3/test_all_v3.py",
        ROOT / "fp4_runtime/TK_quantisation/nvfp4_v3/test_v3_gemm.py",
        ROOT / "fp4_runtime/TK_quantisation/nvfp4_v3/test_v3_vs_v2.py",
        ROOT / "fp4_runtime/TK_quantisation/nvfp4_v5/test_split_dyn.py",
        ROOT / "tools/bench_split_dgrad.py",
        ROOT / "fp4_runtime/ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue/bench_qkv_split3_onepass.py",
        ROOT / "fp4_runtime/ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/bench_qkv_split3_onepass.py",
    ]
    for path in paths:
        source = path.read_text()
        compile(source, str(path), "exec")
        assert "EXTERNAL_PATH" not in source
        assert "FP4_RUNTIME_ROOT" in source
    assert (ROOT / "fp4_runtime/TK_quantisation/nvfp4_v2").is_dir()
    assert (ROOT / "fp4_runtime/TK_quantisation/nvfp4_v3").is_dir()
    assert (ROOT / "fp4_runtime/TK_quantisation/nvfp4_v5").is_dir()
    assert (ROOT / "fp4_runtime/ThunderKittens/kernels/gemm/nvfp4_b200").is_dir()
    assert (ROOT / "TransformerEngine/transformer_engine").is_dir()
    assert (ROOT / "torchtitan_submodule/torchtitan").is_dir()


def test_text_sanitization_never_masks_credentials(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    shaped_value = "ASIA" + "B" * 16
    private_root = "/" + "workspace/private-project"
    (source / "src/sanitized.py").write_text(
        f'ROOT = "{private_root}"\nTOKEN = "{shaped_value}"\n'
    )
    _git(source, "add", "src/sanitized.py")
    _git(source, "commit", "-m", "credential-shaped sanitized fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    document["policy"]["text_sanitize"] = ["src/sanitized.py"]
    manifest.write_text(json.dumps(document))
    output = tmp_path / "blocked"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "blocked_unsafe_content"
    assert shaped_value not in (output / "BUILD_REPORT.json").read_text()
    assert any(
        item["rule"] == "aws_access_key_value"
        for item in report["unsafe_findings"]
    )


def test_exact_component_pin_mismatch_is_rejected(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    document = json.loads(manifest.read_text())
    document["components"][0]["commit"] = "0" * 40
    manifest.write_text(json.dumps(document))

    with pytest.raises(EXPORT.ExportError, match="gitlink"):
        EXPORT.build_export(
            source_repo=source,
            source_revision=source_commit,
            manifest_path=manifest,
            output=tmp_path / "output",
            cache=tmp_path / "cache",
            repo_map=mapping,
            offline=True,
        )


def test_missing_license_is_explicit_release_blocker(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    runtime = mapping["https://github.com/example/fp4-runtime.git"]
    (runtime / "LICENSE").unlink()
    _git(runtime, "add", "--update", "LICENSE")
    _git(runtime, "commit", "-m", "license absent fixture")
    new_runtime_commit = _git(runtime, "rev-parse", "HEAD")
    _git(
        source,
        "update-index",
        "--cacheinfo",
        f"160000,{new_runtime_commit},fp4_runtime",
    )
    _git(source, "commit", "-m", "update runtime fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    runtime_item = next(item for item in document["components"] if item["id"] == "fp4_runtime")
    runtime_item["commit"] = new_runtime_commit
    manifest.write_text(json.dumps(document))
    output = tmp_path / "blocked"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "blocked_release_gates"
    assert report["safe_content_status"] == "pass"
    assert {item["rule"] for item in report["release_blockers"]} == {
        "missing_top_level_license"
    }
    assert not (output / "source-tree").exists()
    components = json.loads((output / "COMPONENTS.json").read_text())
    assert any(item["path"] == "fp4_runtime" for item in components["components"])


def test_explicit_paper_binary_is_hash_bound_and_other_binary_is_excluded(
    tmp_path: Path,
) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    allowed = b"\x89PNG\r\n\x1a\nfixture"
    figure = source / "docs/technical_report/figures/result.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(allowed)
    (source / "docs/technical_report/figures/unlisted.png").write_bytes(b"\x89PNG\0other")
    _git(source, "add", "docs/technical_report/figures")
    _git(source, "commit", "-m", "paper figure fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    document["source"]["include"].append("docs/technical_report/figures/**")
    document["policy"]["binary_allowlist"] = {
        "docs/technical_report/figures/result.png": hashlib.sha256(allowed).hexdigest()
    }
    document["policy"]["exclude"].append("**/*.png")
    manifest.write_text(json.dumps(document))
    output = tmp_path / "output"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "complete"
    tree = output / "source-tree"
    assert (tree / "docs/technical_report/figures/result.png").read_bytes() == allowed
    assert not (tree / "docs/technical_report/figures/unlisted.png").exists()
    inventory = json.loads((tree / "release/components.json").read_text())
    assert inventory["binary_assets"] == [
        {
            "path": "docs/technical_report/figures/result.png",
            "sha256": hashlib.sha256(allowed).hexdigest(),
            "source_component": ".",
        }
    ]
    assert inventory["binary_allowlist"] == [
        {
            "path": "docs/technical_report/figures/result.png",
            "sha256": hashlib.sha256(allowed).hexdigest(),
        }
    ]
    assert EXPORT.verify_export_tree(tree) == []


def test_verifier_rejects_stale_binary_inventory_digest(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    payload = b"\x89PNG\r\n\x1a\nfixture"
    figure = source / "docs/technical_report/figures/result.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(payload)
    _git(source, "add", "docs/technical_report/figures/result.png")
    _git(source, "commit", "-m", "paper figure fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    document["source"]["include"].append("docs/technical_report/figures/**")
    document["policy"]["binary_allowlist"] = {
        "docs/technical_report/figures/result.png": hashlib.sha256(payload).hexdigest()
    }
    document["policy"]["exclude"].append("**/*.png")
    manifest.write_text(json.dumps(document))
    output = tmp_path / "output"
    EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )
    tree = output / "source-tree"
    inventory_path = tree / "release/components.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["binary_allowlist"][0]["sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    ledger_path = tree / "SHA256SUMS"
    inventory_digest = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    lines = ledger_path.read_text().splitlines()
    lines = [
        f"{inventory_digest}  release/components.json"
        if line.endswith("  release/components.json")
        else line
        for line in lines
    ]
    ledger_path.write_text("\n".join(lines) + "\n")

    findings = EXPORT.verify_export_tree(tree)
    assert EXPORT.Finding(
        "binary_allowlist_digest_mismatch",
        "docs/technical_report/figures/result.png",
    ) in findings


def test_allowlisted_binary_mutation_is_rejected(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    original = b"\x89PNG\r\n\x1a\noriginal"
    figure = source / "docs/technical_report/figures/result.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(original)
    _git(source, "add", "docs/technical_report/figures/result.png")
    _git(source, "commit", "-m", "paper figure fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    document["source"]["include"].append("docs/technical_report/figures/**")
    document["policy"]["binary_allowlist"] = {
        "docs/technical_report/figures/result.png": hashlib.sha256(original).hexdigest()
    }
    document["policy"]["exclude"].append("**/*.png")
    manifest.write_text(json.dumps(document))

    figure.write_bytes(b"\x89PNG\r\n\x1a\nmutated")
    _git(source, "add", "docs/technical_report/figures/result.png")
    _git(source, "commit", "-m", "mutate paper figure fixture")
    descendant = _git(source, "rev-parse", "HEAD")

    with pytest.raises(EXPORT.ExportError, match="manifest SHA-256"):
        EXPORT.build_export(
            source_repo=source,
            source_revision=descendant,
            manifest_path=manifest,
            output=tmp_path / "output",
            cache=tmp_path / "cache",
            repo_map=mapping,
            offline=True,
        )


def test_unsealed_environment_blocks_release_but_not_safe_content(tmp_path: Path) -> None:
    source, source_commit, manifest, mapping = _fixture_graph(tmp_path)
    environment_path = source / "release/public/environment.json"
    environment = json.loads(environment_path.read_text())
    environment["status"] = "blocked_unsealed_environment"
    environment["dependency_lock"] = None
    environment_path.write_text(json.dumps(environment))
    _git(source, "add", "release/public/environment.json")
    _git(source, "commit", "-m", "unsealed environment fixture")
    source_commit = _git(source, "rev-parse", "HEAD")
    document = json.loads(manifest.read_text())
    document["source"]["required_ancestor"] = source_commit
    manifest.write_text(json.dumps(document))
    output = tmp_path / "blocked"

    report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=output,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
    )

    assert report["status"] == "blocked_release_gates"
    assert report["safe_content_status"] == "pass"
    assert {item["rule"] for item in report["release_blockers"]} == {
        "coherent_dependency_lock_missing",
        "environment_contract_unsealed",
    }

    review = tmp_path / "review"
    review_report = EXPORT.build_export(
        source_repo=source,
        source_revision=source_commit,
        manifest_path=manifest,
        output=review,
        cache=tmp_path / "cache",
        repo_map=mapping,
        offline=True,
        allow_release_blockers=True,
    )
    assert review_report["status"] == "complete_blocked_not_publishable"
    tree = review / "source-tree"
    blocked = subprocess.run(
        (sys.executable, "tools/release_capsule.py", "doctor", "--phase", "source"),
        cwd=tree,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert blocked.returncode == 2
    assert "public_audit_blocked" in blocked.stdout
    accepted = subprocess.run(
        (
            sys.executable,
            "tools/release_capsule.py",
            "doctor",
            "--phase",
            "source",
            "--allow-publication-blockers",
        ),
        cwd=tree,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert accepted.returncode == 0
    assert "release_capsule_status=pass" in accepted.stdout
    bootstrap = subprocess.run(
        ("bash", "scripts/release/bootstrap.sh", "--verify-only"),
        cwd=tree,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert bootstrap.returncode == 0
    assert "vendored source and component ledgers verified" in bootstrap.stdout

    verified = subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            "verify",
            "--tree",
            str(tree),
            "--allow-blocked",
            "--json",
        ),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    verification = json.loads(verified.stdout)
    assert verification["verification_status"] == "pass"
    assert verification["safe_content_status"] == "pass"
    assert verification["release_status"] == "blocked"
    assert len(verification["release_blockers"]) == 2
    assert verification["unsafe_findings"] == []


@pytest.mark.parametrize(
    "url",
    (
        "ssh://git@example.invalid/project.git",
        "https://user:password@github.com/example/project.git",
        "https://example.invalid/project.git",
        "../relative-project.git",
    ),
)
def test_dependency_urls_must_be_anonymous_allowlisted_https(url: str) -> None:
    with pytest.raises(EXPORT.ExportError):
        EXPORT._canonical_git_url(url)


@pytest.mark.parametrize(
    "suffix",
    (
        ".cubin",
        ".fatbin",
        ".dylib",
        ".dll",
        ".npy",
        ".npz",
        ".parquet",
        ".arrow",
        ".distcp",
        ".zst",
        ".xz",
        ".bz2",
    ),
)
def test_generated_payload_suffixes_are_rejected(suffix: str, tmp_path: Path) -> None:
    target = tmp_path / f"generated{suffix}"
    target.write_bytes(b"generated artifact")
    findings = EXPORT.audit_tree(tmp_path)
    assert EXPORT.Finding("binary_or_compiled_artifact", target.name) in findings


@pytest.mark.parametrize(
    "identifier",
    (
        "kube" + "ctl",
        "kue" + "ue",
        "py" + "torchjob",
        "volt" + "-dev",
        "volt" + "_prod",
    ),
)
def test_cluster_control_content_is_rejected(identifier: str, tmp_path: Path) -> None:
    target = tmp_path / "neutral.txt"
    target.write_text(f"command={identifier}\n")
    findings = EXPORT.audit_tree(tmp_path)
    assert EXPORT.Finding("cluster_control_identity", target.name) in findings


@pytest.mark.parametrize(
    "relative",
    (
        ".cl" + "aude/settings.json",
        ".co" + "dex/instructions.md",
        ".ge" + "mini/settings.json",
    ),
)
def test_internal_agent_directories_are_rejected(relative: str, tmp_path: Path) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text("inert\n")
    findings = EXPORT.audit_tree(tmp_path)
    assert EXPORT.Finding("internal_agent_document", relative) in findings


def test_cluster_control_path_is_rejected_without_operational_content(
    tmp_path: Path,
) -> None:
    relative = "tools/run_probe_" + "k" + "8s.sh"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\ntrue\n")
    findings = EXPORT.audit_tree(tmp_path)
    assert EXPORT.Finding("cluster_control_identity", relative) in findings
