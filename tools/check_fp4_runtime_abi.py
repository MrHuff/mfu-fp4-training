#!/usr/bin/env python3
"""Import the production FP4 extensions and verify their required ABI surface."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import sysconfig

import torch


MODULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "mxfp4_quant_v4",
        "TK_quantisation/mxfp4_v4/mxfp4_quant_v4",
        ("mxfp4_quantize_for_gemm_opt_rht", "mxfp4_quantize_weight_2d"),
    ),
    (
        "_tk_quant_localcta_v4",
        "TK_quantisation/nvfp4_CTA_local_v4/_tk_quant_localcta_v4",
        ("tk_localcta_quantize_for_gemm_opt", "tk_mixed_grad_localcta_row_mx_col_alloc"),
    ),
    (
        "_tk_quant_v5",
        "TK_quantisation/nvfp4_v5/_tk_quant_v5",
        ("tk_quantize_for_gemm_opt", "tk_quantize_weight_2d"),
    ),
    (
        "_C_mx",
        "ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx",
        ("mxfp4_gemm", "mxfp4_gemm_config"),
    ),
    (
        "_C",
        "ThunderKittens/kernels/gemm/nvfp4_b200/_C",
        ("nvfp4_gemm", "nvfp4_forward_rope_packed_qk"),
    ),
    (
        "_C_nv_localcta_gemm_v3",
        "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/_C_nv_localcta_gemm_v3",
        ("nvfp4_localcta_gemm", "nvfp4_localcta_fast_gemm"),
    ),
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct loader for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"expected CUDA 13.0, found {torch.version.cuda}")
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 0):
        raise RuntimeError(f"expected GB200/B200 compute capability (10, 0), found {capability}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension suffix is unavailable")
    for name, relative_stem, symbols in MODULES:
        path = root / f"{relative_stem}{suffix}"
        if not path.is_file():
            raise RuntimeError(f"required extension is absent: {path.relative_to(root)}")
        module = _load(name, path)
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            raise RuntimeError(f"{name} lacks required ABI symbols: {', '.join(missing)}")
    print(f"verified {len(MODULES)} production extension ABIs on sm_100a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
