from __future__ import annotations

import importlib
import sys
from pathlib import Path

from low_bits_training.cce import backend
from low_bits_training.cce import head_sr_state as sr


def _write_fake_runtime(root: Path, marker: str) -> None:
    package = root / "fp4_cce_TK"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "v4_common.py").write_text(
        "_state = None\n"
        f"MARKER = {marker!r}\n"
        "def set_checkpointed_output_head_sr_state(state):\n"
        "    global _state\n"
        "    _state = state\n"
        "def get_checkpointed_output_head_sr_state():\n"
        "    return _state\n",
        encoding="utf-8",
    )
    (package / "nvfp4_cce_tk.py").write_text(
        "from . import v4_common\n"
        "def nvfp4_cce_tk_v4_pcache(*args, **kwargs):\n"
        "    return v4_common.get_checkpointed_output_head_sr_state()\n",
        encoding="utf-8",
    )
    (package / "mxfp4_cce_tk.py").write_text(
        "from . import v4_common\n"
        "def mxfp4_cce_tk_v4_pcache(*args, **kwargs):\n"
        "    return v4_common.get_checkpointed_output_head_sr_state()\n",
        encoding="utf-8",
    )


def _reset_loader() -> None:
    backend._FP4_CCE_TK_V4 = None
    backend._clear_fp4_cce_tk_imports()


def test_lazy_loader_preserves_installed_sr_state_from_selected_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_root = tmp_path / "selected"
    _write_fake_runtime(runtime_root, "selected")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.syspath_prepend(str(runtime_root))
    monkeypatch.setenv("FP4_CCE_TK_ROOT", str(runtime_root))
    for name in (
        "FP4_CCE_V4_CHECKPOINTED_HEAD_SR",
        "FP4_CCE_V4_NVFP4_G_ROW_DATA_SR",
        "FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW",
        "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE",
        "FP4_CCE_V4_MIXED_DW_MXFP8_COLS",
    ):
        monkeypatch.setenv(name, "1")
    for name in (
        "FP4_CCE_V4_NVFP4_G_COL_DATA_SR",
        "FP4_CCE_V4_NVFP4_X_COL_DATA_SR",
        "FP4_CCE_V4_NVFP4_DATA_SR",
        "FP4_CCE_V4_NVFP4_USE_STOCHASTIC_ROUNDING",
    ):
        monkeypatch.setenv(name, "0")
    _reset_loader()

    try:
        step = [0]
        state = sr.build_output_head_sr_state_for_trainer(
            device="cpu",
            training_steps=4,
            gradient_accumulation_steps=1,
            step_getter=lambda: step[0],
        )
        assert state is not None
        installed_common = importlib.import_module("fp4_cce_TK.v4_common")
        owner_tensor = state.get()

        runtime = backend._load_fp4_cce_tk_v4()
        loaded_common = importlib.import_module("fp4_cce_TK.v4_common")
        loaded_nv = importlib.import_module("fp4_cce_TK.nvfp4_cce_tk")

        assert loaded_common is installed_common
        assert loaded_common.get_checkpointed_output_head_sr_state() is owner_tensor
        assert loaded_nv.v4_common is installed_common
        assert runtime.nvfp4_cce_tk_v4_pcache() is owner_tensor
    finally:
        _reset_loader()


def test_lazy_loader_still_evicts_modules_from_a_different_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stale_root = tmp_path / "stale"
    selected_root = tmp_path / "selected"
    _write_fake_runtime(stale_root, "stale")
    _write_fake_runtime(selected_root, "selected")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.syspath_prepend(str(stale_root))
    _reset_loader()

    try:
        stale_common = importlib.import_module("fp4_cce_TK.v4_common")
        stale_common.set_checkpointed_output_head_sr_state(object())
        monkeypatch.setenv("FP4_CCE_TK_ROOT", str(selected_root))

        backend._load_fp4_cce_tk_v4()
        selected_common = importlib.import_module("fp4_cce_TK.v4_common")

        assert selected_common is not stale_common
        assert selected_common.MARKER == "selected"
        assert selected_common.get_checkpointed_output_head_sr_state() is None
        assert Path(selected_common.__file__).is_relative_to(
            selected_root / "fp4_cce_TK"
        )
    finally:
        _reset_loader()
