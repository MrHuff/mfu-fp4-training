from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/nvl72/install_prebuilt_localcta_module.py"
PREPARE = REPO_ROOT / "scripts/nvl72/prepare_fp4_head_b300_runtime.sh"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "nvl72"))

import install_prebuilt_localcta_module as installer  # noqa: E402

X86_SUFFIX = ".cpython-312-x86_64-linux-gnu.so"


@pytest.fixture(autouse=True)
def _x86_extension_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(installer, "extension_suffix", lambda: X86_SUFFIX)


def _elf(machine: int = 62) -> bytes:
    contents = bytearray(4096)
    contents[:4] = b"\x7fELF"
    contents[4] = 2
    contents[5] = 1
    contents[18:20] = machine.to_bytes(2, byteorder="little")
    contents[128:] = b"pinned-localcta-producer" * 64
    return bytes(contents)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source_dir = tmp_path / "proof"
    destination_dir = tmp_path / "runtime" / "TK_quantisation" / "nvfp4_CTA_local_v4"
    source_dir.mkdir()
    destination_dir.mkdir(parents=True)
    source = source_dir / f"{installer.MODULE_STEM}{X86_SUFFIX}"
    destination = destination_dir / source.name
    source.write_bytes(_elf())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return source, destination, tmp_path / "runtime", digest


def test_check_then_atomic_install_preserves_exact_hash_and_mode(tmp_path: Path):
    source, destination, runtime_root, digest = _paths(tmp_path)

    checked = installer.validate_or_install(
        str(source),
        digest,
        str(destination),
        str(runtime_root),
        check_only=True,
    )
    assert checked == (digest, len(_elf()))
    assert not destination.exists()

    installed = installer.validate_or_install(
        str(source),
        digest,
        str(destination),
        str(runtime_root),
        check_only=False,
    )
    assert installed == checked
    assert destination.read_bytes() == source.read_bytes()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    assert stat.S_IMODE(os.lstat(destination).st_mode) == 0o555
    assert not destination.is_symlink()
    assert not list(destination.parent.glob(f".{destination.name}.prebuilt-*"))


def test_optimized_cli_executes_all_checks_and_installs(tmp_path: Path):
    source, destination, runtime_root, digest = _paths(tmp_path)
    optimized_driver = (
        "import sys; "
        f"sys.path.insert(0, {str(SCRIPT.parent)!r}); "
        "import install_prebuilt_localcta_module as m; "
        f"m.extension_suffix=lambda: {X86_SUFFIX!r}; "
        "sys.argv=['install_prebuilt_localcta_module.py']+sys.argv[1:]; "
        "m.main()"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            optimized_driver,
            "--source",
            str(source),
            "--expected-sha256",
            digest,
            "--destination",
            str(destination),
            "--forbidden-root",
            str(runtime_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == (
        "LOCALCTA_PREBUILT_MODULE_INSTALL PASS " f"sha256={digest} bytes={len(_elf())}"
    )
    assert destination.is_file() and not destination.is_symlink()


@pytest.mark.parametrize(
    ("digest", "message"),
    [
        ("0" * 63, "exactly 64 lowercase"),
        ("A" * 64, "exactly 64 lowercase"),
        ("0" * 64, "SHA256 mismatch"),
    ],
)
def test_rejects_malformed_or_mismatched_hash_without_install(
    tmp_path: Path,
    digest: str,
    message: str,
):
    source, destination, runtime_root, _ = _paths(tmp_path)
    with pytest.raises(installer.PrebuiltModuleError, match=message):
        installer.validate_or_install(
            str(source),
            digest,
            str(destination),
            str(runtime_root),
            check_only=False,
        )
    assert not destination.exists()


def test_rejects_wrong_basename_host_arch_and_existing_destination(tmp_path: Path):
    source, destination, runtime_root, digest = _paths(tmp_path)
    wrong_name = source.with_name(f"other{X86_SUFFIX}")
    source.rename(wrong_name)
    with pytest.raises(installer.PrebuiltModuleError, match="basename must be exactly"):
        installer.validate_or_install(
            str(wrong_name),
            digest,
            str(destination),
            str(runtime_root),
            check_only=True,
        )

    source = wrong_name.rename(source)
    source.write_bytes(_elf(machine=183))
    foreign_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(installer.PrebuiltModuleError, match="x86-64 ELF machine 62"):
        installer.validate_or_install(
            str(source),
            foreign_digest,
            str(destination),
            str(runtime_root),
            check_only=True,
        )

    source.write_bytes(_elf())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.write_bytes(b"owner data")
    with pytest.raises(
        installer.PrebuiltModuleError, match="destination already exists"
    ):
        installer.validate_or_install(
            str(source), digest, str(destination), str(runtime_root), check_only=False
        )
    assert destination.read_bytes() == b"owner data"


def test_rejects_source_symlinks_symlink_components_and_runtime_sources(tmp_path: Path):
    source, destination, runtime_root, digest = _paths(tmp_path)
    symlink_dir = tmp_path / "symlink-proof"
    symlink_dir.mkdir()
    source_link = symlink_dir / source.name
    source_link.symlink_to(source)
    with pytest.raises(
        installer.PrebuiltModuleError, match="must not traverse a symlink"
    ):
        installer.validate_or_install(
            str(source_link),
            digest,
            str(destination),
            str(runtime_root),
            check_only=True,
        )

    linked_parent = tmp_path / "linked-proof"
    linked_parent.symlink_to(source.parent, target_is_directory=True)
    linked_source = linked_parent / source.name
    with pytest.raises(
        installer.PrebuiltModuleError, match="must not traverse a symlink"
    ):
        installer.validate_or_install(
            str(linked_source),
            digest,
            str(destination),
            str(runtime_root),
            check_only=True,
        )

    in_runtime = destination.parent / source.name
    in_runtime.write_bytes(source.read_bytes())
    with pytest.raises(
        installer.PrebuiltModuleError, match="outside the purged runtime root"
    ):
        installer.validate_or_install(
            str(in_runtime),
            digest,
            str(destination),
            str(runtime_root),
            check_only=True,
        )


def test_prepare_contract_is_both_or_neither_and_keeps_nonproducer_builds():
    text = PREPARE.read_text(encoding="utf-8")
    purge = "find \"${runtime_root}\" -type f -name '*.so' -delete"
    install = "LOCALCTA_PREBUILT_MODULE_BUILD_SKIP producer="
    assert "FP4_LOCALCTA_PREBUILT_MODULE_PATH" in text
    assert "FP4_LOCALCTA_PREBUILT_MODULE_SHA256" in text
    assert "prebuilt_localcta_path_is_set != prebuilt_localcta_sha_is_set" in text
    assert (
        "prebuilt localCTA module requires strict pre-resource capture/publication"
        in text
    )
    assert text.index(purge) < text.index(
        '--forbidden-root "${runtime_root}"', text.index(purge)
    )
    assert text.index(
        "install_prebuilt_localcta_module.py", text.index(purge)
    ) < text.index('case "${arm}" in\n  bf16)', text.index(purge))
    assert install in text

    localcta = text.split("  localcta-v4-body-bf16-regce)\n", 1)[1].split(
        "  hybrid-localcta-mxfp4-v4-body-bf16-regce)\n", 1
    )[0]
    assert localcta.count("queue_build") == 3
    assert "localcta-v4-body-producer" in localcta
    assert "localcta-v4-body-gemm" in localcta
    assert "localcta-v4-qkv-rope" in localcta


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("FP4_LOCALCTA_PREBUILT_MODULE_PATH", "/proof/module.so"),
        ("FP4_LOCALCTA_PREBUILT_MODULE_SHA256", "0" * 64),
    ],
)
def test_prepare_rejects_partial_prebuilt_environment_before_source_mutation(
    tmp_path: Path,
    variable: str,
    value: str,
):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "fp4_cce_TK").mkdir(parents=True)
    (runtime_root / "ThunderKittens").mkdir()
    fp4_commit = "1" * 40
    tk_commit = "2" * 40
    (runtime_root / ".lbt_fp4_matmul_commit").write_text(
        fp4_commit + "\n", encoding="utf-8"
    )
    (runtime_root / "ThunderKittens/.lbt_thunderkittens_commit").write_text(
        tk_commit + "\n", encoding="utf-8"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text("#!/bin/sh\nprintf '%s\\n' x86_64\n", encoding="utf-8")
    fake_uname.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FP4_MATMUL_EXPECTED_COMMIT"] = fp4_commit
    environment["FP4_THUNDERKITTENS_EXPECTED_COMMIT"] = tk_commit
    environment.pop("FP4_LOCALCTA_PREBUILT_MODULE_PATH", None)
    environment.pop("FP4_LOCALCTA_PREBUILT_MODULE_SHA256", None)
    environment[variable] = value

    result = subprocess.run(
        [
            "bash",
            str(PREPARE),
            str(runtime_root),
            "localcta-v4-body-bf16-regce",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be set together" in result.stderr
    assert not (runtime_root / ".lbt_mxfp4_gemm_default_use_pdl").exists()


def test_default_source_build_branches_remain_byte_identical_to_parent():
    parent = subprocess.run(
        [
            "git",
            "show",
            "8a856c38001afc92cf3fb156040cccd5c6e6afc3:"
            "scripts/nvl72/prepare_fp4_head_b300_runtime.sh",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    current = PREPARE.read_text(encoding="utf-8")
    marker = 'case "${arm}" in\n  bf16)'
    parent_builds = parent[
        parent.index(marker) : parent.index("\nesac", parent.index(marker))
    ]
    current_builds = current[
        current.index(marker) : current.index("\nesac", current.index(marker))
    ]
    assert current_builds == parent_builds
