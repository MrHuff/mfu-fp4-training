from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
NUMERICS_PATH = ROOT / "tools/check_fp4_runtime_numerics.py"
SPEC = importlib.util.spec_from_file_location(
    "check_fp4_runtime_numerics", NUMERICS_PATH
)
assert SPEC is not None and SPEC.loader is not None
NUMERICS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NUMERICS
SPEC.loader.exec_module(NUMERICS)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_gpu_receipt_is_path_free_and_matches_executable_thresholds() -> None:
    receipt = json.loads((ROOT / "release/gpu_gate_receipt.json").read_text())
    schema = json.loads((ROOT / "release/gpu_gate_receipt.schema.json").read_text())

    assert receipt["status"] == "pass_on_matching_host"
    assert set(schema["required"]) <= set(receipt)
    assert len(receipt["kernel_builds"]) == 6
    assert {item["extension"] for item in receipt["kernel_builds"]} == {
        "mxfp4_quant_v4",
        "_tk_quant_localcta_v4",
        "_tk_quant_v5",
        "_C_mx",
        "_C",
        "_C_nv_localcta_gemm_v3",
    }
    thresholds = receipt["numerics"]["thresholds"]
    assert thresholds == {
        "rel_l2_strict_upper_bound": NUMERICS.MAX_REL_L2,
        "cosine_strict_lower_bound": NUMERICS.MIN_COSINE,
        "norm_ratio_closed_interval": [
            NUMERICS.MIN_NORM_RATIO,
            NUMERICS.MAX_NORM_RATIO,
        ],
    }
    for route in ("mxfp4_v4_signed_h32", "nvfp4_v5"):
        values = receipt["numerics"][route]
        assert values["rel_l2"] < NUMERICS.MAX_REL_L2
        assert values["cosine"] > NUMERICS.MIN_COSINE
        assert (
            NUMERICS.MIN_NORM_RATIO
            <= values["norm_ratio"]
            <= NUMERICS.MAX_NORM_RATIO
        )

    serialized_strings = list(_strings(receipt))
    assert not any(value.startswith("/") for value in serialized_strings)
    assert not any(
        "/workspace" in value or "/tmp/" in value for value in serialized_strings
    )


def test_environment_contract_records_the_verification_boundary() -> None:
    environment_path = ROOT / "release/environment.json"
    if not environment_path.is_file():
        # Staging keeps the overlay below release/public; the flattened
        # release installs it at release/environment.json.
        environment_path = ROOT / "release/public/environment.json"
    environment = json.loads(environment_path.read_text())
    lock_path = ROOT / environment["dependency_lock"]["path"]
    lock = json.loads(lock_path.read_text())

    assert environment["status"] == "sealed_reproducible"
    assert environment["release_blockers"] == []
    assert environment["gpu_validation"][
        "six_production_extensions_built_and_abi_checked"
    ]
    assert environment["gpu_validation"]["single_gpu_bf16_reference_numerics_passed"]
    assert not lock["claims"]["container_digest_independently_pulled"]
    assert not lock["claims"]["cold_container_kernel_build_validated"]
    assert not lock["claims"]["distributed_training_validated_from_public_export"]
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == environment[
        "dependency_lock"
    ]["sha256"]


def test_public_gpu_gate_runs_abi_numerics_and_localcta_contract() -> None:
    gate = (ROOT / "scripts/release/run_gpu_gates.sh").read_text()
    aggregate = (ROOT / "scripts/release/run_gates.sh").read_text()
    export = json.loads((ROOT / "release/public_export_manifest.json").read_text())

    assert "tools/check_fp4_runtime_abi.py" in gate
    assert "tools/check_fp4_runtime_numerics.py" in gate
    assert "test_weight_2d_common_outer_scale.py" in gate
    assert "LOCALCTA_GEMM_MODULE_NAME=_C_nv_localcta_gemm_v3" in gate
    assert "scripts/release/run_gpu_gates.sh" in aggregate
    assert "release/gpu_gate_receipt.json" in export["source"]["include"]
    assert "release/gpu_gate_receipt.schema.json" in export["source"]["include"]
