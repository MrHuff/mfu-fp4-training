from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = (ROOT / "Dockerfile", ROOT / "FP4Dockerfile")


def _from_reference(text: str) -> str:
    matches = re.findall(r"^FROM\s+(\S+)\s*$", text, flags=re.MULTILINE)
    assert len(matches) == 1
    return matches[0]


def test_container_files_use_the_locked_image_and_shared_bootstrap() -> None:
    lock = json.loads((ROOT / "release/container_dependency_lock.json").read_text())
    reference = lock["base_image"]["reference"]
    assert "@sha256:" in reference

    texts = [path.read_text() for path in DOCKERFILES]
    for text in texts:
        assert _from_reference(text) == reference
        assert "WORKDIR /opt/mfu" in text
        assert "COPY . /opt/mfu" in text
        assert "RUN scripts/release/bootstrap.sh --install-vendored" in text
        assert 'CMD ["/bin/bash"]' in text
        assert "PYTHONPATH=/opt/mfu:/opt/mfu/torchtitan_submodule" in text
        assert "NVTE_CUDA_ARCHS=100a" in text
        assert "NVTE_FRAMEWORK=pytorch" in text

        # The exact NGC image supplies the toolchain. Container construction
        # must not add a second mutable package-manager or curl-install path.
        lowered = text.lower()
        for forbidden in (
            "apt-get",
            "curl ",
            "pip install",
            "uv pip",
            "wget ",
            "awscliv2",
            "openssh-server",
            "rm -rf /opt/mfu",
        ):
            assert forbidden not in lowered


def test_compatibility_dockerfile_matches_critical_primary_settings() -> None:
    primary, compatibility = (path.read_text() for path in DOCKERFILES)
    for line in (
        "ENV CUDA_HOME=/usr/local/cuda \\",
        "    MAX_JOBS=2 \\",
        "    NVTE_CUDA_ARCHS=100a \\",
        "    NVTE_FRAMEWORK=pytorch \\",
        "    NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \\",
        "    PIP_DISABLE_PIP_VERSION_CHECK=1 \\",
        "    PYTHONDONTWRITEBYTECODE=1 \\",
        "    PYTHONPATH=/opt/mfu:/opt/mfu/torchtitan_submodule",
    ):
        assert line in primary
        assert line in compatibility


def test_container_context_excludes_local_state_and_credentials() -> None:
    ignored = set((ROOT / ".dockerignore").read_text().splitlines())
    assert {
        ".git",
        ".git/**",
        ".env",
        ".env.*",
        "**/.env",
        "**/.env.*",
        "external_inputs.local.json",
        "release/external_inputs.local.json",
        "wandb/",
        "**/wandb/",
    } <= ignored


def test_bootstrap_builds_only_the_vendored_transformer_engine() -> None:
    script = ROOT / "scripts/release/bootstrap.sh"
    text = script.read_text()
    assert "--install-vendored" in text
    assert 'cp -a "$repo_root/TransformerEngine/."' in text
    assert "python -m pip wheel" in text
    assert "python -m pip install" in text
    assert text.count("--no-index") >= 2
    assert text.count("--no-deps") >= 2
    assert "--no-build-isolation" in text
    assert "--force-reinstall" in text
    assert "NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1" in text
    assert "TransformerEngine/build_tools/VERSION.txt" in text
    assert "quantization_custom_format.py" in text
    assert "hashlib.sha256(source.read_bytes())" in text
    assert "scripts/release/build_kernels.sh" not in text
    assert "apt-get" not in text
    assert "curl " not in text

    subprocess.run(("bash", "-n", str(script)), check=True)
    invalid = subprocess.run(
        (str(script), "--not-a-mode"),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert invalid.returncode == 2
    assert "usage:" in invalid.stderr


def test_readme_describes_the_executable_container_boundary() -> None:
    text = (ROOT / "README.md").read_text()
    assert "docker build --pull --file Dockerfile --tag mfu-fp4:25.10 ." in text
    assert "docker run --rm --gpus all --interactive --tty mfu-fp4:25.10" in text
    assert "scripts/release/bootstrap.sh" in text
    assert "scripts/release/build_kernels.sh" in text
    assert "scripts/release/run_gpu_gates.sh" in text
    assert re.search(r"package indexes\s+disabled", text)
    assert "built after start" in text
    assert re.search(r"requires an attached\s+SM100 GPU", text)
