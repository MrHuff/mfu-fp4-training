from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import torch.nn.functional as F


PUBLIC_MLP_ACTIVATIONS = (
    "native_silu",
    "spline_silu",
    "native_gelu",
    "spline_gelu",
)


def _prepend_python_path(path: Path) -> None:
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _extend_spline_python_paths() -> None:
    codebases_root = Path(__file__).resolve().parents[3]
    spline_root = codebases_root / "low-precision-functions" / "autonumerics_zero" / "spline_ops"
    _prepend_python_path(spline_root)


@lru_cache(maxsize=1)
def _load_spline_activation_functions():
    _extend_spline_python_paths()
    try:
        from spline_compile import spline_silu, spline_gelu
    except ImportError as exc:
        raise RuntimeError(
            "spline_compile is required for spline MLP activations. "
            "Build the low-precision-functions spline_ops extension first."
        ) from exc
    return spline_silu, spline_gelu


def resolve_mlp_activation_impl(name: str):
    if name == "native_silu":
        return "native_silu", F.silu
    if name == "native_gelu":
        return "native_gelu", F.gelu
    spline_silu, spline_gelu = _load_spline_activation_functions()
    if name == "spline_silu":
        return "spline_silu", spline_silu
    if name == "spline_gelu":
        return "spline_gelu", spline_gelu
    valid = ", ".join(PUBLIC_MLP_ACTIVATIONS)
    raise ValueError(f"Unknown mlp activation_impl '{name}'. Expected one of: {valid}")
