from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture
def competing_v5_module_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA", "0")
    from low_bits_training.quantization import tk_gemm

    explicit_root = tmp_path / "explicit"
    legacy_root = tmp_path / "legacy"
    for root, marker in (
        (explicit_root, "explicit"),
        (legacy_root, "legacy"),
    ):
        module_dir = root / "TK_quantisation" / "nvfp4_v5"
        module_dir.mkdir(parents=True)
        (module_dir / "_tk_quant_v5.py").write_text(
            f"ORIGIN = {marker!r}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(tk_gemm, "_fp4_matmul_root", lambda: str(explicit_root))
    monkeypatch.setattr(tk_gemm, "_LEGACY_FP4_MATMUL_ROOT", str(legacy_root))
    monkeypatch.setattr(sys, "path", list(sys.path))
    previous = sys.modules.pop("_tk_quant_v5", None)
    try:
        yield tk_gemm, explicit_root
    finally:
        sys.modules.pop("_tk_quant_v5", None)
        if previous is not None:
            sys.modules["_tk_quant_v5"] = previous


@pytest.mark.parametrize(
    "loader_name",
    (
        "_get_tk_quant_plain",
        "_get_tk_quant_for_gemm",
        "_get_tk_quant_standalone",
    ),
)
def test_explicit_v5_module_root_outranks_legacy_fallback(
    competing_v5_module_roots,
    loader_name,
):
    tk_gemm, explicit_root = competing_v5_module_roots
    tk_gemm.reset_tk_runtime_caches()
    tk_gemm._tk_quant_standalone_mod_cache = None

    module = getattr(tk_gemm, loader_name)()

    assert module.ORIGIN == "explicit"
    assert Path(module.__file__).resolve().is_relative_to(explicit_root.resolve())
