from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "nvl72"))

import package_fp4_8b_dev_asset as packager  # noqa: E402
PATCH_PATH = (
    REPO_ROOT
    / "scripts/nvl72/patches/"
    "fp4_matmul_c033a29_tk_ece25c4_cluster_lifetime_r2.patch"
)
EXPECTED_SHA256 = "853c161fa6be289e8e238cb8c90d26322b9e0463a42585a0a9558e05fb03fe8d"
EXPECTED_HEADERS = {
    "ThunderKittens/kernels/gemm/mxfp4_gb200/mxfp4_atb_gemm.cuh",
    "ThunderKittens/kernels/gemm/mxfp4_gb200/mxfp4_batched_gemm.cuh",
    "ThunderKittens/kernels/gemm/mxfp4_gb200/mxfp4_gemm.cuh",
    "ThunderKittens/kernels/gemm/mxfp4_gb200/mxfp4_silu_dgrad_quant_gemm.cuh",
    "ThunderKittens/kernels/gemm/mxfp4_gb200/mxfp4_split2_accum_gemm.cuh",
    "ThunderKittens/kernels/gemm/mxfp4_gb200/mxfp4_split3_accum_gemm.cuh",
    "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_accum_gemm.cuh",
    "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_batched_accum_gemm.cuh",
    "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_batched_gemm.cuh",
    "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_gemm.cuh",
    "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_split2_accum_gemm.cuh",
    "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_split3_accum_gemm.cuh",
    (
        "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/"
        "nvfp4_localcta_batched_kernel.cuh"
    ),
    (
        "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/"
        "nvfp4_localcta_kernel.cuh"
    ),
    (
        "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/"
        "nvfp4_localcta_silu_dgrad_quant_gemm.cuh"
    ),
}


def test_cluster_lifetime_patch_is_the_validated_runtime_delta() -> None:
    contents = PATCH_PATH.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == EXPECTED_SHA256

    text = contents.decode("utf-8")
    touched = {
        line.removeprefix("+++ b/")
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }
    assert touched == EXPECTED_HEADERS

    added = [
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    # Eighteen clustered kernels live in the fifteen headers. Every warp
    # completes the entry phase, and every CTA remains live through a second
    # cluster phase at function exit.
    assert added.count("    everyone::tma::cluster::wait_aligned();") == 18
    assert (
        added.count(
            '    asm volatile("barrier.cluster.arrive.relaxed.aligned;\\n");'
        )
        == 18
    )
    assert (
        added.count('    asm volatile("barrier.cluster.wait.aligned;\\n");')
        == 18
    )


def _write_cluster_headers(root: Path) -> None:
    kernel = "\n".join(
        (
            "everyone::tma::cluster::arrive_aligned();",
            "everyone::tma::cluster::wait_aligned();",
            'asm volatile("barrier.cluster.arrive.relaxed.aligned;\\n");',
            'asm volatile("barrier.cluster.wait.aligned;\\n");',
        )
    )
    for relative_path, count in packager.CLUSTER_LIFETIME_HEADERS.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((kernel + "\n") * count, encoding="utf-8")


def test_packager_accepts_complete_cluster_lifetime_phases(tmp_path: Path) -> None:
    _write_cluster_headers(tmp_path)
    packager._validate_cluster_lifetime_policy(tmp_path)


def test_packager_rejects_incomplete_cluster_lifetime_phase(tmp_path: Path) -> None:
    _write_cluster_headers(tmp_path)
    relative_path = next(iter(packager.CLUSTER_LIFETIME_HEADERS))
    path = tmp_path / relative_path
    contents = path.read_text(encoding="utf-8")
    path.write_text(
        contents.replace("barrier.cluster.wait.aligned;", "barrier.cluster.wait;", 1),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="clustered GEMM lifetime policy mismatch"):
        packager._validate_cluster_lifetime_policy(tmp_path)
