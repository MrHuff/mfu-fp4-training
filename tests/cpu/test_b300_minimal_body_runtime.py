# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "scripts/nvl72/prepare_fp4_head_b300_runtime.sh"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "nvl72"))

import check_fp4_runtime_extensions as extension_check  # noqa: E402
import check_localcta_qkv_rope_runtime as rope_check  # noqa: E402
import select_fp4_b200_runtime_arm as arm_selector  # noqa: E402


LEGACY_BUILD_BRANCH_SHA256 = {
    "bf16": "869bc8ba78e6ecdb2fc908d09d8731db7290e336a344537f4d3db425bf8d208c",
    "mxfp4-gemm": "7c75f47cb7c14218b24222a9b4da20d2e222af308570be701c8fde0fee479729",
    "mxfp4": "7dca60b481a0c93e12e2a1ae7e432052bb8082432ff74d27e67fc3be69d7a0ab",
    "nvfp4|nvfp4-localcta": (
        "70aec7a321593c4a2470a7abd75d9d237d1f5a01e80d012339969b9bf60c068e"
    ),
    "mxfp8-mxfp4": "0f823ced3684c97ab5e2e64422a153668093a4bfef0ac3433cf43e5cedf3f13c",
    "mxfp8-nvfp4": "878fc10c73a704b8ee777816509f7e94398495c12ebc6c1b258b6605aaa00747",
    "mxfp4-mxfp8-nvfp4": (
        "010979a1a145570eefcd6540cf1e0491c8f0e59b2dda918da9edb42a0107f822"
    ),
    "mxfp4-nvfp4": "8d72ecfd300ebeaae932839636a3912d85daeaed18e3ff74a0060887bc3db88a",
    "all": "40bcb85f085f36b8bda45f33182c7f69c068591fb89c49024162bc9b28bcfeb4",
}

LEGACY_REQUIRED_BRANCH_SHA256 = {
    "mxfp4-gemm": "db9be70ded7602ebce7f312473c0a4d6b7765b7b51e9d26f08ba2d9162a54e17",
    "mxfp4": "a6ea71df554acf970863e210e17fb162d22b8371bdf268281667754a981b0a03",
    "nvfp4|nvfp4-localcta": (
        "ac90e9ccbb981d852a6b4042ad60edf343d8278e68872b37b3e3986d9327f310"
    ),
    "mxfp8-mxfp4": "ab171ede0c5b8c8bf19df3e41e3a1e95cc7e83f712c21c408a33501e9343ffca",
    "mxfp8-nvfp4": "ebc079b11247c389f098ef58b08744b59719eb6b45900eb44a33a0a2e104e06d",
    "mxfp4-mxfp8-nvfp4": (
        "e557de0d157ff56647f3c27ceb440a6235f77407e96801c43f19f9fb06083399"
    ),
    "mxfp4-nvfp4": "b2e6a1035a2de113c81f0f86a9ae959b02b64224cd5ac95f2cf28b23bd7a4bec",
    "all": "47d359e7690f6e9efa9abde1485a2cd1d48c486d58e27f1bd5e3e8bd3ad67bba",
}


def _case_branches(text: str, anchor: str) -> dict[str, str]:
    start = text.index(anchor)
    case_start = text.index('case "${arm}" in', start)
    payload_start = case_start + len('case "${arm}" in\n')
    payload = text[payload_start : text.index("\nesac", payload_start)]
    pattern = re.compile(
        r"^  (?P<label>\S[^\n]*)\)\n(?P<body>.*?)(?=^  \S[^\n]*\)\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    return {match["label"]: match["body"] for match in pattern.finditer(payload)}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _elf(machine: int = 62) -> bytes:
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[18:20] = machine.to_bytes(2, byteorder="little")
    return bytes(header)


def _fake_cuobjdump(path: Path, arch: str) -> Path:
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' 'ELF file    1: kernel.{arch}.cubin'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_prepare_script_is_valid_shell_and_registers_explicit_minimal_arms():
    subprocess.run(["bash", "-n", str(PREPARE)], check=True)
    text = PREPARE.read_text(encoding="utf-8")

    assert "mxfp4-v4-2d-body-bf16-regce" in text
    assert "localcta-v4-body-bf16-regce" in text
    assert "hybrid-localcta-mxfp4-v4-body-bf16-regce" in text
    assert "hybrid-v5-mxfp4-v4-body-bf16-regce" in text
    assert "check_fp4_runtime_extensions.py" in text
    assert "check_localcta_qkv_rope_runtime.py" in text
    assert '--not-older-than "${build_start_marker}"' in text
    assert 'forbidden_extension_globs+=("${runtime_root}/**/*.so")' in text


def test_minimal_build_plans_have_only_the_required_independent_body_outputs():
    text = PREPARE.read_text(encoding="utf-8")
    build = _case_branches(text, 'case "${arm}" in\n  bf16)')
    required = _case_branches(text, "required_extensions=()")

    mx_build = build["mxfp4-v4-2d-body-bf16-regce"]
    assert mx_build.count("queue_build") == 2
    assert "TK_quantisation/mxfp4_v4" in mx_build
    assert "gemm/mxfp4_gb200" in mx_build
    assert "mxfp4_v3" not in mx_build
    assert "cce" not in mx_build.lower()

    localcta_build = build["localcta-v4-body-bf16-regce"]
    assert localcta_build.count("queue_build") == 3
    assert "TK_quantisation/nvfp4_CTA_local_v4" in localcta_build
    assert "localCTA_epilogue_v3" in localcta_build
    assert '"${runtime_root}/ThunderKittens/kernels/gemm/nvfp4_b200"' in localcta_build
    assert "nvfp4_v5" not in localcta_build
    assert "build_nvfp4_cce_runtime" not in localcta_build

    mx_required = required["mxfp4-v4-2d-body-bf16-regce"]
    assert mx_required.count("${extension_suffix}") == 2
    assert "mxfp4_quant_v4${extension_suffix}" in mx_required
    assert "_C_mx${extension_suffix}" in mx_required

    localcta_required = required["localcta-v4-body-bf16-regce"]
    assert localcta_required.count("${extension_suffix}") == 3
    assert "_tk_quant_localcta_v4${extension_suffix}" in localcta_required
    assert "_C_nv_localcta_gemm_v3${extension_suffix}" in localcta_required
    assert "nvfp4_b200/_C${extension_suffix}" in localcta_required

    hybrid_build = build["hybrid-localcta-mxfp4-v4-body-bf16-regce"]
    assert hybrid_build.count("queue_build") == 5
    assert "TK_quantisation/nvfp4_CTA_local_v4" in hybrid_build
    assert "localCTA_epilogue_v3" in hybrid_build
    assert '"${runtime_root}/ThunderKittens/kernels/gemm/nvfp4_b200"' in hybrid_build
    assert "TK_quantisation/mxfp4_v4" in hybrid_build
    assert "gemm/mxfp4_gb200" in hybrid_build
    assert "mxfp4_v3" not in hybrid_build
    assert "nvfp4_v5" not in hybrid_build
    assert "cce" not in hybrid_build.lower()

    hybrid_required = required["hybrid-localcta-mxfp4-v4-body-bf16-regce"]
    assert hybrid_required.count("${extension_suffix}") == 5
    assert "_tk_quant_localcta_v4${extension_suffix}" in hybrid_required
    assert "_C_nv_localcta_gemm_v3${extension_suffix}" in hybrid_required
    assert "nvfp4_b200/_C${extension_suffix}" in hybrid_required
    assert "mxfp4_quant_v4${extension_suffix}" in hybrid_required
    assert "_C_mx${extension_suffix}" in hybrid_required

    v5_hybrid_build = build["hybrid-v5-mxfp4-v4-body-bf16-regce"]
    assert v5_hybrid_build.count("queue_build") == 4
    assert "TK_quantisation/nvfp4_v5" in v5_hybrid_build
    assert 'gemm/nvfp4_b200"' in v5_hybrid_build
    assert "TK_quantisation/mxfp4_v4" in v5_hybrid_build
    assert "gemm/mxfp4_gb200" in v5_hybrid_build
    assert "nvfp4_CTA_local_v4" not in v5_hybrid_build
    assert "localCTA_epilogue_v3" not in v5_hybrid_build
    assert "build_nvfp4_cce_runtime" not in v5_hybrid_build
    assert "build_mxfp4_cce_runtime" not in v5_hybrid_build

    v5_hybrid_required = required["hybrid-v5-mxfp4-v4-body-bf16-regce"]
    assert v5_hybrid_required.count("${extension_suffix}") == 4
    assert "nvfp4_v5/_tk_quant_v5${extension_suffix}" in v5_hybrid_required
    assert "nvfp4_b200/_C${extension_suffix}" in v5_hybrid_required
    assert "mxfp4_quant_v4${extension_suffix}" in v5_hybrid_required
    assert "_C_mx${extension_suffix}" in v5_hybrid_required
    assert "nvfp4_CTA_local_v4" not in v5_hybrid_required
    assert "localCTA_epilogue_v3" not in v5_hybrid_required
    assert "_cce" not in v5_hybrid_required


def test_localcta_qkv_rope_contract_rejects_missing_or_noncallable_symbol():
    class Missing:
        pass

    class Noncallable:
        nvfp4_forward_rope_packed_qk = object()
        nvfp4_inverse_rope_packed_qk = lambda: None

    with pytest.raises(RuntimeError, match="missing callable"):
        rope_check.require_rope_symbols(Missing())
    with pytest.raises(RuntimeError, match="missing callable"):
        rope_check.require_rope_symbols(Noncallable())


def test_localcta_qkv_rope_contract_accepts_callable_symbol():
    forward = lambda: None
    inverse = lambda: None
    module = type("Module", (), {})()
    module.nvfp4_forward_rope_packed_qk = forward
    module.nvfp4_inverse_rope_packed_qk = inverse
    assert rope_check.require_rope_symbols(module) == (forward, inverse)


def test_runtime_loaders_match_the_minimal_extension_contracts():
    mx = (
        REPO_ROOT / "low_bits_training/quantization/mxfp4_backend.py"
    ).read_text(encoding="utf-8")
    localcta = (
        REPO_ROOT / "low_bits_training/quantization/tk_gemm.py"
    ).read_text(encoding="utf-8")
    fused = (
        REPO_ROOT / "low_bits_training/quantization/fused_te_linear.py"
    ).read_text(encoding="utf-8")

    assert 'module_basename = f"mxfp4_quant_{version}"' in mx
    assert 'mxfp4_gb200/_C_mx*.so' in mx
    assert 'if "_cce" not in os.path.basename(path)' in mx
    assert "'v4': ('nvfp4_CTA_local_v4', '_tk_quant_localcta_v4')" in localcta
    assert "return 'localCTA_epilogue_v3', '_C_nv_localcta_gemm_v3'" in localcta
    assert "def _get_tk_plain():" in localcta
    assert "_get_tk_plain(), 'nvfp4_forward_rope_packed_qk', None" in fused
    assert (
        "return use_tk_localcta() and "
        "os.environ.get('USE_TK_LOCALCTA_DIRECT_CONTRACT', '0') == '1'"
    ) in localcta
    assert (
        "return os.environ.get('USE_TK_QKV_LOCALCTA_TK_PREPARED_ACT', '0') "
        "== '1'"
    ) in localcta


def test_minimal_compile_plan_reduces_make_invocations_and_parallelizes_safely():
    text = PREPARE.read_text(encoding="utf-8")
    build = _case_branches(text, 'case "${arm}" in\n  bf16)')

    # The full helpers contain four MXFP4 CCE and five NVFP4 CCE/sidecar makes.
    assert build["mxfp4"].count("build_kernel") + 4 == 7
    assert build["nvfp4|nvfp4-localcta"].count("build_kernel") + 5 == 9
    assert build["mxfp4-v4-2d-body-bf16-regce"].count("queue_build") == 2
    assert build["localcta-v4-body-bf16-regce"].count("queue_build") == 3
    assert build["hybrid-localcta-mxfp4-v4-body-bf16-regce"].count("queue_build") == 5
    assert build["hybrid-v5-mxfp4-v4-body-bf16-regce"].count("queue_build") == 4

    # Every output builds in a different directory and has a distinct
    # allowlisted extension, which permits bounded parallel execution.
    required = _case_branches(text, "required_extensions=()")
    for arm in (
        "mxfp4-v4-2d-body-bf16-regce",
        "localcta-v4-body-bf16-regce",
        "hybrid-localcta-mxfp4-v4-body-bf16-regce",
        "hybrid-v5-mxfp4-v4-body-bf16-regce",
    ):
        paths = [
            path
            for path in re.findall(r'"\$\{runtime_root\}(/[^\"]+)', required[arm])
            if "${extension_suffix}" in path
        ]
        expected = {
            "mxfp4-v4-2d-body-bf16-regce": 2,
            "localcta-v4-body-bf16-regce": 3,
            "hybrid-localcta-mxfp4-v4-body-bf16-regce": 5,
            "hybrid-v5-mxfp4-v4-body-bf16-regce": 4,
        }[arm]
        assert len(paths) == len(set(paths)) == expected
        assert len({str(Path(path).parent) for path in paths}) == expected


def _pure_v5_environment() -> dict[str, str]:
    return {
        "FP4_ATTN_BACKEND": "tk",
        "FP4_FFN_BACKEND": "tk",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_FUSED": "0",
        "USE_TK_V5_2D_WEIGHT_QUANT": "1",
    }


def test_nonlocalcta_v5_routes_select_the_minimal_arm():
    minimal = arm_selector.HYBRID_V5_MXFP4_BF16_REGCE_ARM
    pure_v5 = _pure_v5_environment()
    assert arm_selector.select_runtime_arm(minimal, pure_v5) == minimal
    assert arm_selector.select_runtime_arm("auto", pure_v5) == minimal

    hybrid = dict(pure_v5)
    hybrid["LBT_FP4_MIXED_LAYERS"] = "v5:1-27;mxfp4:28-32"
    assert arm_selector.select_runtime_arm(minimal, hybrid) == minimal
    assert arm_selector.select_runtime_arm("auto", hybrid) == minimal


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("USE_TK_LOCALCTA", "1"),
        ("USE_TK_LOCALCTA_FUSED", "1"),
        ("FP4_ATTN_BACKEND", "localcta"),
        ("FP4_FFN_BACKEND", "localcta"),
        ("LBT_LOCALCTA_V4_PROFILE", "highwater"),
        ("LBT_FP4_MIXED_LAYERS", "localcta:1-27;mxfp4:28-32"),
    ],
)
def test_localcta_routes_cannot_bypass_the_resource_gate(name, value):
    minimal = arm_selector.HYBRID_V5_MXFP4_BF16_REGCE_ARM
    environment = _pure_v5_environment()
    environment[name] = value
    with pytest.raises(ValueError, match="localCTA|USE_TK_LOCALCTA=0"):
        arm_selector.select_runtime_arm(minimal, environment)
    assert arm_selector.select_runtime_arm("auto", environment) == "all"


def test_true_localcta_route_keeps_the_resource_gated_all_arm():
    environment = {
        "FP4_ATTN_BACKEND": "localcta",
        "FP4_FFN_BACKEND": "localcta",
        "LBT_FP4_MIXED_LAYERS": "localcta:1-27;mxfp4:28-32",
        "LBT_LOCALCTA_V4_PROFILE": "highwater",
        "USE_TK_LOCALCTA": "1",
        "USE_TK_LOCALCTA_FUSED": "1",
        "USE_TK_V5_2D_WEIGHT_QUANT": "0",
    }
    assert arm_selector.select_runtime_arm("all", environment) == "all"
    assert arm_selector.select_runtime_arm("auto", environment) == "all"

    text = PREPARE.read_text(encoding="utf-8")
    before_resource_gate = text.split("localcta_max_stack_bytes=16", 1)[0]
    labels = before_resource_gate.rsplit('case "${arm}" in\n  ', 1)[1]
    labels = labels.split(")", 1)[0]
    assert "localcta-v4-body-bf16-regce" in labels
    assert "hybrid-localcta-mxfp4-v4-body-bf16-regce" in labels
    assert "all" in labels
    assert arm_selector.HYBRID_V5_MXFP4_BF16_REGCE_ARM not in labels


def test_minimal_v5_arm_requires_an_explicit_v5_route():
    environment = _pure_v5_environment()
    environment["USE_TK_V5_2D_WEIGHT_QUANT"] = "0"
    with pytest.raises(ValueError, match="requires a v5 body route"):
        arm_selector.select_runtime_arm(
            arm_selector.HYBRID_V5_MXFP4_BF16_REGCE_ARM,
            environment,
        )


def test_every_legacy_full_build_and_required_list_is_unchanged():
    text = PREPARE.read_text(encoding="utf-8")
    build = _case_branches(text, 'case "${arm}" in\n  bf16)')
    required = _case_branches(text, "required_extensions=()")

    assert {
        arm: _sha256(build[arm]) for arm in LEGACY_BUILD_BRANCH_SHA256
    } == LEGACY_BUILD_BRANCH_SHA256
    assert {
        arm: _sha256(required[arm]) for arm in LEGACY_REQUIRED_BRANCH_SHA256
    } == LEGACY_REQUIRED_BRANCH_SHA256


def test_cuda_arch_parser_requires_an_exact_single_target():
    assert extension_check.parse_cuda_arches(
        "ELF file 1: a.sm_103a.cubin\nELF file 2: b.sm_103a.cubin\n"
    ) == {"sm_103a"}


def test_extension_validator_accepts_fresh_x86_b300_binary(tmp_path):
    marker = tmp_path / "build-start"
    extension = tmp_path / "body.so"
    marker.write_text("start", encoding="utf-8")
    extension.write_bytes(_elf())
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(extension, ns=(2_000_000_000, 2_000_000_000))
    cuobjdump = _fake_cuobjdump(tmp_path / "cuobjdump", "sm_103a")

    extension_check.validate_extension(
        extension,
        expected_sm="sm_103a",
        not_older_than=marker,
        cuobjdump=str(cuobjdump),
    )


def test_extension_validator_rejects_stale_binary(tmp_path):
    marker = tmp_path / "build-start"
    extension = tmp_path / "body.so"
    marker.write_text("start", encoding="utf-8")
    extension.write_bytes(_elf())
    os.utime(extension, ns=(1_000_000_000, 1_000_000_000))
    os.utime(marker, ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(ValueError, match="predates this build"):
        extension_check.validate_extension(
            extension,
            expected_sm="sm_103a",
            not_older_than=marker,
            cuobjdump=str(tmp_path / "unused"),
        )


def test_extension_validator_rejects_a_compatibility_symlink(tmp_path):
    marker = tmp_path / "build-start"
    target = tmp_path / "real-body.so"
    extension = tmp_path / "body.so"
    marker.write_text("start", encoding="utf-8")
    target.write_bytes(_elf())
    extension.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        extension_check.validate_extension(
            extension,
            expected_sm="sm_103a",
            not_older_than=marker,
            cuobjdump=str(tmp_path / "unused"),
        )


def test_extension_validator_rejects_wrong_host_or_device_arch(tmp_path):
    marker = tmp_path / "build-start"
    marker.write_text("start", encoding="utf-8")
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    cuobjdump = _fake_cuobjdump(tmp_path / "cuobjdump", "sm_100a")

    foreign_host = tmp_path / "aarch64.so"
    foreign_host.write_bytes(_elf(machine=183))
    os.utime(foreign_host, ns=(2_000_000_000, 2_000_000_000))
    with pytest.raises(ValueError, match="x86-64 ELF machine 62"):
        extension_check.validate_extension(
            foreign_host,
            expected_sm="sm_103a",
            not_older_than=marker,
            cuobjdump=str(cuobjdump),
        )

    wrong_device = tmp_path / "b200.so"
    wrong_device.write_bytes(_elf())
    os.utime(wrong_device, ns=(2_000_000_000, 2_000_000_000))
    with pytest.raises(ValueError, match="expected only sm_103a cubins, found sm_100a"):
        extension_check.validate_extension(
            wrong_device,
            expected_sm="sm_103a",
            not_older_than=marker,
            cuobjdump=str(cuobjdump),
        )


def test_exact_allowlist_rejects_unused_cce_or_head_extension(tmp_path):
    required = (tmp_path / "body.so").resolve()
    required.write_bytes(_elf())
    extra = tmp_path / "_C_mx_cce_backward_v3.so"
    extra.write_bytes(_elf())

    with pytest.raises(ValueError, match="forbidden extensions were produced"):
        extension_check.validate_forbidden(
            (),
            (str(tmp_path / "**/*.so"),),
            allowed=frozenset({required}),
        )
