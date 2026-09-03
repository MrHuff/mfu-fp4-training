"""Benchmark-oriented MXFP4 backend loader for TK quantize/GEMM paths.

This module intentionally stays thin:
- load the selected MXFP4 quantizer extension from ``TK_quantisation/mxfp4_*``
- load the existing MXFP4 GEMM kernels from ``ThunderKittens/.../mxfp4_gb200``
- expose direct wrappers for isolation benches and future fused integration

DeepSeek MXFP4 training defaults to the ``mxfp4_v3`` quantizer.  The
``mxfp4_v4`` tree also contains experimental MoE producer/combine helper
extensions; those helpers are only loaded by default when the selected quantizer
backend is v4, or when explicitly enabled with ``MXFP4_ALLOW_V4_MOE_HELPERS=1``.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import torch


logger = logging.getLogger(__name__)
_QUANT_MODULES: dict[str, object] = {}
_GEMM_MODULE = None
_MOE_COMBINE_MODULE = None
_MOE_PRODUCER_MODULE = None
_QUANT_IMPORT_ATTEMPTED: dict[str, bool] = {}
_GEMM_IMPORT_ATTEMPTED = False
_MOE_COMBINE_IMPORT_ATTEMPTED = False
_MOE_PRODUCER_IMPORT_ATTEMPTED = False
_QUANT_IMPORT_ERROR: dict[str, Exception] = {}
_GEMM_IMPORT_ERROR = None
_MOE_COMBINE_IMPORT_ERROR = None
_MOE_PRODUCER_IMPORT_ERROR = None


def _use_mxfp4_batched_gemm_tuning() -> bool:
    return os.environ.get("LBT_DISABLE_MXFP4_BATCHED_TUNE", "0") != "1"


def _env_int_or_none(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer %s=%r", name, raw)
        return None


def _config_override_from_keys(keys: Iterable[str]) -> tuple[bool, int | None]:
    """Return whether an override was set and its normalized config ID.

    Negative config IDs explicitly request the native, unconfigured entrypoint.
    Keeping that state separate from an absent override prevents an opt-out from
    falling back into an automatic selector.
    """
    for key in keys:
        if key not in os.environ:
            continue
        value = _env_int_or_none(key)
        if value is None:
            continue
        return True, None if value < 0 else value
    return False, None


def _mxfp4_gemm_config_override_status(
    prefix: str,
    M: int,
    N: int,
    K: int,
) -> tuple[bool, int | None]:
    return _config_override_from_keys(
        (
            f"{prefix}_M{M}_N{N}_K{K}",
            f"{prefix}_M{M}_N{N}",
            f"{prefix}_ID",
        )
    )


def _mxfp4_batched_gemm_config_override(
    M: int,
    N: int,
    K: int,
    num_batches: int,
    default: int | None,
) -> int | None:
    default_value = default if default is not None else -1
    for key in (
        f"MXFP4_BATCHED_GEMM_CONFIG_M{M}_N{N}_K{K}_B{num_batches}",
        f"MXFP4_BATCHED_GEMM_CONFIG_M{M}_N{N}_K{K}",
        f"MXFP4_BATCHED_GEMM_CONFIG_M{M}_N{N}_B{num_batches}",
        f"MXFP4_BATCHED_GEMM_CONFIG_M{M}_N{N}",
    ):
        value = _env_int_or_none(key)
        if value is not None:
            return None if value < 0 else value
    return default


def _fp4_root_candidates() -> list[str]:
    roots = []
    quant_root = os.environ.get("FP4_MXFP4_ROOT") or os.environ.get("FP4_MATMUL_QUANT_ROOT")
    if quant_root:
        roots.append(os.path.abspath(quant_root))
    env_root = os.environ.get("FP4_MATMUL_ROOT")
    if env_root:
        roots.append(os.path.abspath(env_root))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    roots.append(os.path.join(repo_root, "fp4_runtime"))
    roots.append(os.path.abspath(os.path.join(repo_root, "..", "fp4_matmul")))
    deduped = []
    seen = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return deduped


def _fp4_gemm_root_candidates() -> list[str]:
    roots = []
    env_root = os.environ.get("FP4_MATMUL_GEMM_ROOT")
    if env_root:
        roots.append(os.path.abspath(env_root))
    roots.extend(_fp4_root_candidates())
    deduped = []
    seen = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return deduped


def use_mxfp4_tk_backend() -> bool:
    return os.environ.get("USE_MXFP4_TK_BACKEND", "0") == "1"


def mxfp4_backend_version() -> str:
    return os.environ.get("MXFP4_BACKEND_VERSION", "v3")


def _allow_v4_moe_helpers() -> bool:
    raw = os.environ.get("MXFP4_ALLOW_V4_MOE_HELPERS")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return mxfp4_backend_version().strip().lower() == "v4"


@dataclass
class MXFP4Quantized:
    fp4: torch.Tensor
    sc: torch.Tensor
    bf16: torch.Tensor | None = None


@dataclass
class MXFP4HTileCarrier:
    z_out: torch.Tensor
    row_fp4: torch.Tensor
    row_sc: torch.Tensor
    col_fp4: torch.Tensor
    col_sc: torch.Tensor
    r_tile: torch.Tensor


def _candidate_quant_paths(version: str | None = None) -> list[str]:
    version = version or mxfp4_backend_version()
    module_basename = f"mxfp4_quant_{version}"
    patterns = [
        f"{root}/TK_quantisation/mxfp4_{version}/{module_basename}*.so"
        for root in _fp4_root_candidates()
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(glob.glob(pattern)))
    return matches


def _candidate_gemm_paths() -> list[str]:
    patterns = [
        f"{root}/ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx*.so"
        for root in _fp4_gemm_root_candidates()
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(glob.glob(pattern)))
    return [path for path in matches if "_cce" not in os.path.basename(path)]


def _candidate_moe_combine_paths() -> list[str]:
    patterns = [
        f"{root}/TK_quantisation/mxfp4_v4/mxfp4_moe_combine*.so"
        for root in _fp4_root_candidates()
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(glob.glob(pattern)))
    return matches


def _candidate_moe_producer_paths() -> list[str]:
    patterns = [
        f"{root}/TK_quantisation/mxfp4_v4/mxfp4_moe_producer*.so"
        for root in _fp4_root_candidates()
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(glob.glob(pattern)))
    return matches


def _module_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    if ".cpython-" in base:
        return base.split(".cpython-", maxsplit=1)[0]
    if base.endswith(".so"):
        return base[:-3]
    return os.path.splitext(base)[0]


def _load_extension(module_name: str, so_path: str):
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_cuda_ready():
    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()


def _load_quant_module(version: str | None = None):
    version = version or mxfp4_backend_version()
    if _QUANT_IMPORT_ATTEMPTED.get(version, False):
        return _QUANT_MODULES.get(version)
    _QUANT_IMPORT_ATTEMPTED[version] = True

    try:
        candidates = _candidate_quant_paths(version)
        if not candidates:
            raise FileNotFoundError(
                "MXFP4 quantizer .so not found. Build one of:\n"
                + "\n".join(
                    f"  {root}/TK_quantisation/mxfp4_{version}"
                    for root in _fp4_root_candidates()
                )
            )

        _ensure_cuda_ready()
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        ctypes.CDLL(os.path.join(torch_lib, "libtorch_python.so"), mode=ctypes.RTLD_GLOBAL)

        so_path = candidates[0]
        module_name = _module_name_from_path(so_path)
        _QUANT_MODULES[version] = _load_extension(module_name, so_path)
        logger.info("Loaded MXFP4 quant backend version=%s from %s", version, so_path)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        _QUANT_IMPORT_ERROR[version] = exc
        raise
    return _QUANT_MODULES[version]


def _load_gemm_module():
    global _GEMM_MODULE, _GEMM_IMPORT_ATTEMPTED, _GEMM_IMPORT_ERROR
    if _GEMM_IMPORT_ATTEMPTED:
        return _GEMM_MODULE
    _GEMM_IMPORT_ATTEMPTED = True

    try:
        candidates = _candidate_gemm_paths()
        if not candidates:
            raise FileNotFoundError(
                "MXFP4 GEMM .so not found. Build one of:\n"
                + "\n".join(
                    f"  {root}/ThunderKittens/kernels/gemm/mxfp4_gb200"
                    for root in _fp4_gemm_root_candidates()
                )
            )

        _ensure_cuda_ready()
        so_path = candidates[0]
        module_name = _module_name_from_path(so_path)
        old_module = sys.modules.pop(module_name, None)
        _GEMM_MODULE = _load_extension(module_name, so_path)
        logger.info("Loaded MXFP4 GEMM backend from %s", so_path)
        if old_module is not None:
            sys.modules[module_name] = old_module
        elif module_name in sys.modules:
            del sys.modules[module_name]
    except Exception as exc:  # pragma: no cover - surfaced to caller
        _GEMM_IMPORT_ERROR = exc
        raise
    return _GEMM_MODULE


def _load_moe_combine_module():
    global _MOE_COMBINE_MODULE, _MOE_COMBINE_IMPORT_ATTEMPTED, _MOE_COMBINE_IMPORT_ERROR
    if _MOE_COMBINE_IMPORT_ATTEMPTED:
        return _MOE_COMBINE_MODULE
    _MOE_COMBINE_IMPORT_ATTEMPTED = True

    try:
        if not _allow_v4_moe_helpers():
            raise AttributeError(
                "MXFP4 MoE combine helpers are v4-only; set "
                "MXFP4_ALLOW_V4_MOE_HELPERS=1 to opt in while using a non-v4 "
                "quant backend."
            )
        candidates = _candidate_moe_combine_paths()
        if not candidates:
            raise FileNotFoundError(
                "MXFP4 MoE combine .so not found. Build:\n"
                + "\n".join(
                    f"  {root}/TK_quantisation/mxfp4_v4 make moe_combine_ext"
                    for root in _fp4_root_candidates()
                )
            )

        _ensure_cuda_ready()
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        ctypes.CDLL(os.path.join(torch_lib, "libtorch_python.so"), mode=ctypes.RTLD_GLOBAL)

        so_path = candidates[0]
        module_name = _module_name_from_path(so_path)
        _MOE_COMBINE_MODULE = _load_extension(module_name, so_path)
        logger.info("Loaded MXFP4 MoE combine backend from %s", so_path)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        _MOE_COMBINE_IMPORT_ERROR = exc
        raise
    return _MOE_COMBINE_MODULE


def _load_moe_producer_module():
    global _MOE_PRODUCER_MODULE, _MOE_PRODUCER_IMPORT_ATTEMPTED, _MOE_PRODUCER_IMPORT_ERROR
    if _MOE_PRODUCER_IMPORT_ATTEMPTED:
        return _MOE_PRODUCER_MODULE
    _MOE_PRODUCER_IMPORT_ATTEMPTED = True

    try:
        if not _allow_v4_moe_helpers():
            raise AttributeError(
                "MXFP4 MoE producer helpers are v4-only; set "
                "MXFP4_ALLOW_V4_MOE_HELPERS=1 to opt in while using a non-v4 "
                "quant backend."
            )
        candidates = _candidate_moe_producer_paths()
        if not candidates:
            raise FileNotFoundError(
                "MXFP4 MoE producer .so not found. Build:\n"
                + "\n".join(
                    f"  {root}/TK_quantisation/mxfp4_v4 make moe_producer_ext"
                    for root in _fp4_root_candidates()
                )
            )

        _ensure_cuda_ready()
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        ctypes.CDLL(os.path.join(torch_lib, "libtorch_python.so"), mode=ctypes.RTLD_GLOBAL)

        so_path = candidates[0]
        module_name = _module_name_from_path(so_path)
        _MOE_PRODUCER_MODULE = _load_extension(module_name, so_path)
        logger.info("Loaded MXFP4 MoE producer backend from %s", so_path)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        _MOE_PRODUCER_IMPORT_ERROR = exc
        raise
    return _MOE_PRODUCER_MODULE


def mxfp4_backend_capabilities() -> dict[str, bool]:
    version = mxfp4_backend_version()
    quant_ok = bool(_candidate_quant_paths(version))
    gemm_ok = bool(_candidate_gemm_paths())
    silu_dgrad_quant_ok = False
    silu_dgrad_from_sigmoid_quant_ok = False
    if gemm_ok:
        try:
            gemm_mod = _load_gemm_module()
            silu_dgrad_quant_ok = hasattr(gemm_mod, "mxfp4_gemm_silu_dgrad_quant")
            silu_dgrad_from_sigmoid_quant_ok = hasattr(
                gemm_mod, "mxfp4_gemm_silu_dgrad_from_sigmoid_quant"
            )
        except Exception:
            silu_dgrad_quant_ok = False
            silu_dgrad_from_sigmoid_quant_ok = False
    moe_helpers_ok = _allow_v4_moe_helpers()
    moe_combine_ok = moe_helpers_ok and bool(_candidate_moe_combine_paths())
    moe_producer_ok = moe_helpers_ok and bool(_candidate_moe_producer_paths())
    return {
        "quant_version": version,
        "quant_available": quant_ok,
        "gemm_available": gemm_ok,
        "silu_dgrad_quant_available": silu_dgrad_quant_ok,
        "silu_dgrad_from_sigmoid_quant_available": silu_dgrad_from_sigmoid_quant_ok,
        "moe_combine_available": moe_combine_ok,
        "moe_producer_available": moe_producer_ok,
        "backend_available": quant_ok and gemm_ok,
    }


def mxfp4_quantize_for_gemm(input_tensor: torch.Tensor, mode: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_quant_module().mxfp4_quantize_for_gemm(input_tensor, mode)


def mxfp4_quantize_for_gemm_opt(
    input_tensor: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_for_gemm_opt"):
        if data_stochastic_rounding or scale_stochastic_rounding:
            raise AttributeError("mxfp4_quantize_for_gemm_opt is unavailable in this backend")
        return mod.mxfp4_quantize_for_gemm(input_tensor, mode)
    return mod.mxfp4_quantize_for_gemm_opt(
        input_tensor,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_for_gemm_opt_rht(
    input_tensor: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    rht_axes: str | None = None,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_for_gemm_opt_rht"):
        raise AttributeError("mxfp4_quantize_for_gemm_opt_rht is unavailable in this backend")
    return mod.mxfp4_quantize_for_gemm_opt_rht(
        input_tensor,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_group_quantize_dim0(
    input_tensor: torch.Tensor, group_sizes: Iterable[int]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _load_quant_module().mxfp4_group_quantize_dim0(input_tensor, list(group_sizes))


def mxfp4_quantize_col_only(
    input_tensor: torch.Tensor, mode: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if hasattr(mod, "mxfp4_quantize_col_only"):
        return mod.mxfp4_quantize_col_only(input_tensor, mode)
    transposed = input_tensor.transpose(0, 1).contiguous()
    return mod.mxfp4_quantize_for_gemm(transposed, mode)


def mxfp4_quantize_col_only_opt(
    input_tensor: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_col_only_opt"):
        if data_stochastic_rounding or scale_stochastic_rounding:
            raise AttributeError("mxfp4_quantize_col_only_opt is unavailable in this backend")
        return mxfp4_quantize_col_only(input_tensor, mode)
    return mod.mxfp4_quantize_col_only_opt(
        input_tensor,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_col_only_opt_rht(
    input_tensor: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    rht_axes: str | None = None,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_col_only_opt_rht"):
        raise AttributeError("mxfp4_quantize_col_only_opt_rht is unavailable in this backend")
    return mod.mxfp4_quantize_col_only_opt_rht(
        input_tensor,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_row_and_col(
    input_tensor: torch.Tensor, mode: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _load_quant_module().mxfp4_quantize_row_and_col(input_tensor, mode)


def mxfp4_quantize_weight_2d(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_weight_2d"):
        raise AttributeError("mxfp4_quantize_weight_2d is unavailable in this backend")
    return mod.mxfp4_quantize_weight_2d(input_tensor)


def mxfp4_copy_col_slices(
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    row_starts: Iterable[int],
    rows: Iterable[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_copy_col_slices"):
        raise AttributeError("mxfp4_copy_col_slices is unavailable in this backend")
    fp4_slices, sc_slices = mod.mxfp4_copy_col_slices(
        col_fp4,
        col_sc,
        [int(x) for x in row_starts],
        [int(x) for x in rows],
    )
    return list(fp4_slices), list(sc_slices)


def mxfp4_pack_grouped_rows_bf16(
    input_tensor: torch.Tensor,
    starts: Iterable[int],
    rows: Iterable[int],
    padded_rows: Iterable[int],
    output_cols: int,
) -> torch.Tensor:
    mod = _load_quant_module()
    name = "mxfp4_pack_grouped_rows_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        [int(x) for x in starts],
        [int(x) for x in rows],
        [int(x) for x in padded_rows],
        int(output_cols),
    )


def mxfp4_pack_indexed_rows_bf16(
    input_tensor: torch.Tensor,
    token_indices: torch.Tensor,
    num_batches: int,
    live_rows_per_batch: int,
    padded_rows_per_batch: int,
    output_cols: int,
) -> torch.Tensor:
    mod = _load_quant_module()
    name = "mxfp4_pack_indexed_rows_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        token_indices,
        int(num_batches),
        int(live_rows_per_batch),
        int(padded_rows_per_batch),
        int(output_cols),
    )


def mxfp4_pack_shared_routed_rows_bf16(
    input_tensor: torch.Tensor,
    token_indices: torch.Tensor,
    route_starts: Iterable[int],
    rows: Iterable[int],
    padded_starts: Iterable[int],
    padded_rows: Iterable[int],
    shared_rows_padded: int,
    output_cols: int,
) -> torch.Tensor:
    name = "mxfp4_pack_shared_routed_rows_bf16"
    mod = _load_moe_producer_module()
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        token_indices,
        [int(x) for x in route_starts],
        [int(x) for x in rows],
        [int(x) for x in padded_starts],
        [int(x) for x in padded_rows],
        int(shared_rows_padded),
        int(output_cols),
    )


def mxfp4_pack_indexed_scaled_rows_bf16_variable(
    input_tensor: torch.Tensor,
    token_indices: torch.Tensor,
    scores: torch.Tensor,
    route_starts: Iterable[int],
    rows: Iterable[int],
    padded_starts: Iterable[int],
    padded_rows: Iterable[int],
    output_cols: int,
) -> torch.Tensor:
    name = "mxfp4_pack_indexed_scaled_rows_bf16_variable"
    mod = _load_moe_producer_module()
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        token_indices,
        scores,
        [int(x) for x in route_starts],
        [int(x) for x in rows],
        [int(x) for x in padded_starts],
        [int(x) for x in padded_rows],
        int(output_cols),
    )


def mxfp4_dot_and_pack_indexed_scaled_rows_bf16_variable(
    input_tensor: torch.Tensor,
    token_indices: torch.Tensor,
    scores: torch.Tensor,
    expert_out: torch.Tensor,
    route_starts: Iterable[int],
    rows: Iterable[int],
    padded_starts: Iterable[int],
    padded_rows: Iterable[int],
    output_cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    name = "mxfp4_dot_and_pack_indexed_scaled_rows_bf16_variable"
    mod = _load_moe_producer_module()
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        token_indices,
        scores,
        expert_out,
        [int(x) for x in route_starts],
        [int(x) for x in rows],
        [int(x) for x in padded_starts],
        [int(x) for x in padded_rows],
        int(output_cols),
    )


def mxfp4_pack_indexed_rmsnorm_rows_bf16_variable(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    inv_rms: torch.Tensor,
    token_indices: torch.Tensor,
    route_starts: Iterable[int],
    rows: Iterable[int],
    padded_starts: Iterable[int],
    padded_rows: Iterable[int],
    output_cols: int,
) -> torch.Tensor:
    name = "mxfp4_pack_indexed_rmsnorm_rows_bf16_variable"
    mod = _load_moe_producer_module()
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        norm_weight,
        inv_rms,
        token_indices,
        [int(x) for x in route_starts],
        [int(x) for x in rows],
        [int(x) for x in padded_starts],
        [int(x) for x in padded_rows],
        int(output_cols),
    )


def mxfp4_pack_grouped_rows_quantize_row_and_col(
    input_tensor: torch.Tensor,
    num_batches: int,
    live_rows_per_batch: int,
    padded_rows_per_batch: int,
    output_cols: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_pack_grouped_rows_quantize_row_and_col"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        int(num_batches),
        int(live_rows_per_batch),
        int(padded_rows_per_batch),
        int(output_cols),
        int(mode),
    )


def mxfp4_pack_indexed_scaled_rows_quantize_row_and_col(
    input_tensor: torch.Tensor,
    token_indices: torch.Tensor,
    scores: torch.Tensor,
    num_batches: int,
    live_rows_per_batch: int,
    padded_rows_per_batch: int,
    output_cols: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_pack_indexed_scaled_rows_quantize_row_and_col"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        token_indices,
        scores,
        int(num_batches),
        int(live_rows_per_batch),
        int(padded_rows_per_batch),
        int(output_cols),
        int(mode),
    )


def mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable(
    input_tensor: torch.Tensor,
    token_indices: torch.Tensor,
    scores: torch.Tensor,
    route_starts: Iterable[int],
    rows: Iterable[int],
    padded_starts: Iterable[int],
    padded_rows: Iterable[int],
    output_cols: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    name = "mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable"
    try:
        mod = _load_moe_producer_module()
    except (AttributeError, FileNotFoundError, ImportError):
        mod = _load_quant_module()
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        token_indices,
        scores,
        [int(x) for x in route_starts],
        [int(x) for x in rows],
        [int(x) for x in padded_starts],
        [int(x) for x in padded_rows],
        int(output_cols),
        int(mode),
    )


def mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    inv_rms: torch.Tensor,
    token_indices: torch.Tensor,
    num_batches: int,
    live_rows_per_batch: int,
    padded_rows_per_batch: int,
    output_cols: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        norm_weight,
        inv_rms,
        token_indices,
        int(num_batches),
        int(live_rows_per_batch),
        int(padded_rows_per_batch),
        int(output_cols),
        int(mode),
    )


def mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col_variable(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    inv_rms: torch.Tensor,
    token_indices: torch.Tensor,
    route_starts: Iterable[int],
    rows: Iterable[int],
    padded_starts: Iterable[int],
    padded_rows: Iterable[int],
    output_cols: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    name = "mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col_variable"
    try:
        mod = _load_moe_producer_module()
    except (AttributeError, FileNotFoundError, ImportError):
        mod = _load_quant_module()
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        norm_weight,
        inv_rms,
        token_indices,
        [int(x) for x in route_starts],
        [int(x) for x in rows],
        [int(x) for x in padded_starts],
        [int(x) for x in padded_rows],
        int(output_cols),
        int(mode),
    )


def mxfp4_moe_scale_scatter_add_bf16(
    src: torch.Tensor,
    scores: torch.Tensor,
    token_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_scale_scatter_add_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(src, scores, token_indices, out)


def mxfp4_moe_scatter_add_bf16(
    src: torch.Tensor,
    token_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_scatter_add_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(src, token_indices, out)


def mxfp4_moe_indexed_dot_rows_bf16(
    lhs: torch.Tensor,
    token_indices: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_indexed_dot_rows_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(lhs, token_indices, rhs)


def mxfp4_moe_build_route_inverse(route_positions: torch.Tensor) -> torch.Tensor:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_build_route_inverse"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(route_positions)


def mxfp4_moe_build_route_inverse_padded(
    route_positions: torch.Tensor,
    live_rows_per_batch: int,
    padded_rows_per_batch: int,
) -> torch.Tensor:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_build_route_inverse_padded"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        route_positions,
        int(live_rows_per_batch),
        int(padded_rows_per_batch),
    )


def mxfp4_moe_gather_scores(
    scores: torch.Tensor,
    route_positions: torch.Tensor,
) -> torch.Tensor:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_gather_scores"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(scores, route_positions)


def mxfp4_moe_scatter_scores(
    grad_sorted: torch.Tensor,
    route_positions: torch.Tensor,
    num_scores: int,
) -> torch.Tensor:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_scatter_scores"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(grad_sorted, route_positions, int(num_scores))


def mxfp4_moe_reorder_indices(
    selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_reorder_indices"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(selected_experts, int(num_experts), int(top_k))


def mxfp4_moe_reorder_scores_full(
    scores: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
    pad_granularity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_reorder_scores_full"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        scores,
        selected_experts,
        int(num_experts),
        int(top_k),
        int(pad_granularity),
    )


def mxfp4_moe_route_combine_bf16(
    expert_out: torch.Tensor,
    sorted_scores: torch.Tensor,
    inverse: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> None:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_route_combine_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(expert_out, sorted_scores, inverse, out, int(top_k))


def mxfp4_moe_route_combine_padded_index_bf16(
    expert_out_padded: torch.Tensor,
    sorted_scores: torch.Tensor,
    inverse: torch.Tensor,
    inverse_padded: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> None:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_route_combine_padded_index_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        expert_out_padded,
        sorted_scores,
        inverse,
        inverse_padded,
        out,
        int(top_k),
    )


def mxfp4_moe_route_combine_padded_bf16(
    expert_out_padded: torch.Tensor,
    sorted_scores: torch.Tensor,
    inverse: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    live_rows_per_batch: int,
    padded_rows_per_batch: int,
) -> None:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_route_combine_padded_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        expert_out_padded,
        sorted_scores,
        inverse,
        out,
        int(top_k),
        int(live_rows_per_batch),
        int(padded_rows_per_batch),
    )


def mxfp4_moe_indexed_dot_rows_padded_bf16(
    lhs: torch.Tensor,
    token_indices: torch.Tensor,
    rhs_padded: torch.Tensor,
    live_rows_per_batch: int,
    padded_rows_per_batch: int,
) -> torch.Tensor:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_indexed_dot_rows_padded_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        lhs,
        token_indices,
        rhs_padded,
        int(live_rows_per_batch),
        int(padded_rows_per_batch),
    )


def mxfp4_moe_route_scatter_gradx_bf16(
    grad_routed: torch.Tensor,
    inverse: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> None:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_route_scatter_gradx_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(grad_routed, inverse, out, int(top_k))


def mxfp4_moe_route_scatter_gradx_padded_index_bf16(
    grad_routed_padded: torch.Tensor,
    inverse_padded: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> None:
    mod = _load_moe_combine_module()
    name = "mxfp4_moe_route_scatter_gradx_padded_index_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(grad_routed_padded, inverse_padded, out, int(top_k))


def mxfp4_pack_w13_bf16(
    w1: torch.Tensor,
    w3: torch.Tensor,
    H13n: int,
    Dk: int,
) -> torch.Tensor:
    mod = _load_quant_module()
    name = "mxfp4_pack_w13_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(w1, w3, int(H13n), int(Dk))


def mxfp4_split_w13_bf16(
    grad_w13: torch.Tensor,
    H: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_split_w13_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(grad_w13, int(H), int(D))


def mxfp4_scatter_grouped_rows_bf16(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    starts: Iterable[int],
    rows: Iterable[int],
    padded_rows: Iterable[int],
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_scatter_grouped_rows_bf16"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        input_tensor,
        output_tensor,
        [int(x) for x in starts],
        [int(x) for x in rows],
        [int(x) for x in padded_rows],
    )


def mxfp4_quantize_nhsd_wo_row_and_col(
    input_tensor: torch.Tensor, mode: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_nhsd_wo_row_and_col"):
        raise AttributeError("mxfp4_quantize_nhsd_wo_row_and_col is unavailable in this backend")
    return mod.mxfp4_quantize_nhsd_wo_row_and_col(input_tensor, mode)


def mxfp4_quantize_row_and_col_opt(
    input_tensor: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_row_and_col_opt"):
        raise AttributeError("mxfp4_quantize_row_and_col_opt is unavailable in this backend")
    return mod.mxfp4_quantize_row_and_col_opt(
        input_tensor,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_row_and_col_opt_rht(
    input_tensor: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    rht_axes: str = "col",
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    axes = rht_axes.strip().lower().replace("-", "_")
    if axes == "row":
        name = "mxfp4_quantize_row_and_col_opt_rht_row_only"
    elif axes == "both":
        name = "mxfp4_quantize_row_and_col_opt_rht_both"
    elif axes == "col":
        name = "mxfp4_quantize_row_and_col_opt_rht"
    else:
        raise ValueError(f"Unsupported MXFP4 rht_axes={rht_axes!r}; expected row, col, or both")
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_row_and_col_launch_inplace(
    input_tensor: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_row_and_col_launch_inplace"):
        raise AttributeError("mxfp4_quantize_row_and_col_launch_inplace is unavailable in this backend")
    mod.mxfp4_quantize_row_and_col_launch_inplace(
        input_tensor, row_fp4, row_sc, col_fp4, col_sc, mode
    )


def mxfp4_fused_rmsnorm_quantize_row_and_col(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_rmsnorm_quantize_row_and_col"):
        raise AttributeError("mxfp4_fused_rmsnorm_quantize_row_and_col is unavailable in this backend")
    return mod.mxfp4_fused_rmsnorm_quantize_row_and_col(input_tensor, norm_weight, float(epsilon), mode)


def mxfp4_fused_rmsnorm_quantize_row_and_col_from_row_rms_partial(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    row_rms_partial: torch.Tensor,
    epsilon: float,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_fused_rmsnorm_quantize_row_and_col_from_row_rms_partial"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        norm_weight,
        row_rms_partial,
        float(epsilon),
        mode,
    )


def mxfp4_fused_rmsnorm_quantize_row_and_col_opt(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    row_with_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if row_with_rht:
        raise NotImplementedError(
            "mxfp4_fused_rmsnorm_quantize_row_and_col_opt currently supports col-axis RHT only"
        )
    mod = _load_quant_module()
    name = "mxfp4_fused_rmsnorm_quantize_row_and_col_opt"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(
        input_tensor,
        norm_weight,
        float(epsilon),
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_fused_rmsnorm_to_bf16(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_rmsnorm_to_bf16"):
        raise AttributeError("mxfp4_fused_rmsnorm_to_bf16 is unavailable in this backend")
    return mod.mxfp4_fused_rmsnorm_to_bf16(input_tensor, norm_weight, float(epsilon))


def mxfp4_quantize_split3_row_and_col(
    input0: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split3_row_and_col"):
        raise AttributeError("mxfp4_quantize_split3_row_and_col is unavailable in this backend")
    return mod.mxfp4_quantize_split3_row_and_col(input0, input1, input2, mode)


def mxfp4_quantize_split3_row_and_col_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split3_row_and_col_launch_inplace"):
        raise AttributeError("mxfp4_quantize_split3_row_and_col_launch_inplace is unavailable in this backend")
    mod.mxfp4_quantize_split3_row_and_col_launch_inplace(
        input0, input1, input2, row_fp4, row_sc, col_fp4, col_sc, mode
    )


def mxfp4_quantize_split3_row_and_col_opt_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    row_with_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> None:
    if row_with_rht:
        raise NotImplementedError(
            "mxfp4_quantize_split3_row_and_col_opt_launch_inplace currently supports col-axis RHT only"
        )
    mod = _load_quant_module()
    name = "mxfp4_quantize_split3_row_and_col_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        input0,
        input1,
        input2,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_split3_row_and_col_inverse_rope_live64(
    input0: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split3_row_and_col_inverse_rope_live64"):
        raise AttributeError("mxfp4_quantize_split3_row_and_col_inverse_rope_live64 is unavailable in this backend")
    return mod.mxfp4_quantize_split3_row_and_col_inverse_rope_live64(
        input0, input1, input2, rope_cs, int(rope_seq_len), mode
    )


def mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace"):
        raise AttributeError("mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace is unavailable in this backend")
    mod.mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace(
        input0, input1, input2, rope_cs, int(rope_seq_len), row_fp4, row_sc, col_fp4, col_sc, mode
    )


def mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    row_with_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> None:
    if row_with_rht:
        raise NotImplementedError(
            "mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace currently supports col-axis RHT only"
        )
    mod = _load_quant_module()
    name = "mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        input0,
        input1,
        input2,
        rope_cs,
        int(rope_seq_len),
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_split2_row_and_col(
    input0: torch.Tensor,
    input1: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split2_row_and_col"):
        raise AttributeError("mxfp4_quantize_split2_row_and_col is unavailable in this backend")
    return mod.mxfp4_quantize_split2_row_and_col(input0, input1, mode)


def mxfp4_quantize_split2_row_and_col_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split2_row_and_col_launch_inplace"):
        raise AttributeError("mxfp4_quantize_split2_row_and_col_launch_inplace is unavailable in this backend")
    mod.mxfp4_quantize_split2_row_and_col_launch_inplace(
        input0, input1, row_fp4, row_sc, col_fp4, col_sc, mode
    )


def mxfp4_quantize_split2_row_and_col_opt_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    row_with_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> None:
    if row_with_rht:
        raise NotImplementedError(
            "mxfp4_quantize_split2_row_and_col_opt_launch_inplace currently supports col-axis RHT only"
        )
    mod = _load_quant_module()
    name = "mxfp4_quantize_split2_row_and_col_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        input0,
        input1,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        input0,
        input1,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_split2_row_only_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split2_row_only_launch_inplace"):
        raise AttributeError("mxfp4_quantize_split2_row_only_launch_inplace is unavailable in this backend")
    mod.mxfp4_quantize_split2_row_only_launch_inplace(
        input0, input1, row_fp4, row_sc, mode
    )


def mxfp4_quantize_split2_row_only_opt_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_quantize_split2_row_only_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        input0,
        input1,
        row_fp4,
        row_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_split2_col_only_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split2_col_only_launch_inplace"):
        raise AttributeError("mxfp4_quantize_split2_col_only_launch_inplace is unavailable in this backend")
    mod.mxfp4_quantize_split2_col_only_launch_inplace(
        input0, input1, col_fp4, col_sc, mode
    )


def mxfp4_quantize_split2_col_only_opt_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_quantize_split2_col_only_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        input0,
        input1,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
    )


def mxfp4_quantize_split2_row_and_col_splitcols(
    input0: torch.Tensor,
    input1: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split2_row_and_col_splitcols"):
        raise AttributeError("mxfp4_quantize_split2_row_and_col_splitcols is unavailable in this backend")
    return mod.mxfp4_quantize_split2_row_and_col_splitcols(input0, input1, mode)


def mxfp4_quantize_split2_row_and_col_splitcols_launch_inplace(
    input0: torch.Tensor,
    input1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col0_fp4: torch.Tensor,
    col0_sc: torch.Tensor,
    col1_fp4: torch.Tensor,
    col1_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_quantize_split2_row_and_col_splitcols_launch_inplace"):
        raise AttributeError("mxfp4_quantize_split2_row_and_col_splitcols_launch_inplace is unavailable in this backend")
    mod.mxfp4_quantize_split2_row_and_col_splitcols_launch_inplace(
        input0, input1, row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc, mode
    )


def mxfp4_fused_silu_deriv_quantize_split2_row_and_col(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_silu_deriv_quantize_split2_row_and_col"):
        raise AttributeError("mxfp4_fused_silu_deriv_quantize_split2_row_and_col is unavailable in this backend")
    return mod.mxfp4_fused_silu_deriv_quantize_split2_row_and_col(dh, h3, h1_raw, mode)


def mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace"):
        raise AttributeError("mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace is unavailable in this backend")
    mod.mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace(
        dh, h3, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode
    )


def mxfp4_fused_silu_deriv_quantize_split2_row_and_col_opt_launch_inplace(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
    row_with_rht: bool = False,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_deriv_quantize_split2_row_and_col_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        dh,
        h3,
        h1_raw,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
        row_with_rht,
    )


def mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols"):
        raise AttributeError("mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols is unavailable in this backend")
    return mod.mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols(dh, h3, h1_raw, mode)


def mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col0_fp4: torch.Tensor,
    col0_sc: torch.Tensor,
    col1_fp4: torch.Tensor,
    col1_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace"):
        raise AttributeError("mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace is unavailable in this backend")
    mod.mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace(
        dh, h3, h1_raw, row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc, mode
    )


def mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_and_col_splitcols_launch_inplace(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    sig_h1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col0_fp4: torch.Tensor,
    col0_sc: torch.Tensor,
    col1_fp4: torch.Tensor,
    col1_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_and_col_splitcols_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        dh,
        h3,
        h1_raw,
        sig_h1,
        row_fp4,
        row_sc,
        col0_fp4,
        col0_sc,
        col1_fp4,
        col1_sc,
        mode,
    )


def mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined(
    dh: torch.Tensor,
    h13: torch.Tensor,
    hidden_dim: int,
    h3_offset: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(dh, h13, int(hidden_dim), int(h3_offset), mode)


def mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    dh1_out: torch.Tensor,
    dh3_out: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace"):
        raise AttributeError("mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace is unavailable in this backend")
    mod.mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace(
        dh, h3, h1_raw, dh1_out, dh3_out, row_fp4, row_sc, mode
    )


def mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_bf16_launch_inplace(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    sig_h1: torch.Tensor,
    dh1_out: torch.Tensor,
    dh3_out: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_bf16_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(dh, h3, h1_raw, sig_h1, dh1_out, dh3_out, row_fp4, row_sc, mode)


def mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace(
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    dh1_out: torch.Tensor,
    dh3_out: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(dh, h3, h1_raw, dh1_out, dh3_out, row_fp4, row_sc, mode)


def mxfp4_fused_silu_mul_quantize_row_and_col(
    h1_raw: torch.Tensor,
    h3: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_silu_mul_quantize_row_and_col"):
        raise AttributeError("mxfp4_fused_silu_mul_quantize_row_and_col is unavailable in this backend")
    return mod.mxfp4_fused_silu_mul_quantize_row_and_col(h1_raw, h3, mode)


def mxfp4_fused_silu_mul_quantize_row_and_col_strided(
    h13: torch.Tensor,
    hidden_dim: int,
    h3_offset: int,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_mul_quantize_row_and_col_strided"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(h13, int(hidden_dim), int(h3_offset), mode)


def mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace(
    h1_raw: torch.Tensor,
    h3: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    if not hasattr(mod, "mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace"):
        raise AttributeError("mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace is unavailable in this backend")
    mod.mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace(
        h1_raw, h3, row_fp4, row_sc, col_fp4, col_sc, mode
    )


def mxfp4_fused_silu_mul_sigmoid_quantize_row_and_col_launch_inplace(
    h1_raw: torch.Tensor,
    h3: torch.Tensor,
    sig_h1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_mul_sigmoid_quantize_row_and_col_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(h1_raw, h3, sig_h1, row_fp4, row_sc, col_fp4, col_sc, mode)


def mxfp4_fused_sqrelu_quantize_row_and_col(
    h1_raw: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_fused_sqrelu_quantize_row_and_col"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(h1_raw, mode)


def mxfp4_fused_sqrelu_quantize_row_and_col_launch_inplace(
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_sqrelu_quantize_row_and_col_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode)


def mxfp4_fused_sqrelu_deriv_quantize_row_and_col(
    dh: torch.Tensor,
    h1_raw: torch.Tensor,
    mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _load_quant_module()
    name = "mxfp4_fused_sqrelu_deriv_quantize_row_and_col"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    return getattr(mod, name)(dh, h1_raw, mode)


def mxfp4_fused_sqrelu_deriv_quantize_row_and_col_launch_inplace(
    dh: torch.Tensor,
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_sqrelu_deriv_quantize_row_and_col_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode)


def mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace(
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
    row_with_rht: bool = False,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        h1_raw,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
        row_with_rht,
    )


def mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace(
    dh: torch.Tensor,
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
    row_with_rht: bool = False,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        dh,
        h1_raw,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
        row_with_rht,
    )


def mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace(
    h1_raw: torch.Tensor,
    h3: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    mode: int = 1,
    *,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    use_rht: bool = False,
    rht_block_size: int = 16,
    with_random_sign_mask: bool = True,
    rng_seed: int = 1234,
    rng_subsequence: int = 0,
    row_with_rht: bool = False,
) -> None:
    mod = _load_quant_module()
    name = "mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace"
    if not hasattr(mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    getattr(mod, name)(
        h1_raw,
        h3,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        use_rht,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence,
        row_with_rht,
    )


def quantize_mxfp4_tensor(
    input_tensor: torch.Tensor, keep_bf16: bool = False, mode: int = 1
) -> MXFP4Quantized:
    fp4, sc = mxfp4_quantize_for_gemm(input_tensor, mode)
    return MXFP4Quantized(fp4=fp4, sc=sc, bf16=input_tensor if keep_bf16 else None)


def quantize_mxfp4_row_and_col(
    input_tensor: torch.Tensor, keep_bf16: bool = False, mode: int = 1
) -> tuple[MXFP4Quantized, MXFP4Quantized]:
    row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_row_and_col(input_tensor, mode)
    bf16 = input_tensor if keep_bf16 else None
    return (
        MXFP4Quantized(fp4=row_fp4, sc=row_sc, bf16=bf16),
        MXFP4Quantized(fp4=col_fp4, sc=col_sc),
    )


def _use_mxfp4_dense_gemm_shape_config() -> bool:
    return os.environ.get("MXFP4_DENSE_GEMM_SHAPE_CONFIG", "1") != "0"


def _use_mxfp4_residual_gemm_shape_config() -> bool:
    value = os.environ.get("MXFP4_GEMM_RESIDUAL_SHAPE_CONFIG")
    if value is not None:
        return value == "1"
    # Residual cfg10 is faster in isolation, but races with the backward
    # wgrad-overlap path. Overlap is now opt-in, so the tuned residual config
    # should be the default when the env is absent.
    return os.environ.get("MXFP4_USE_BWD_WGRAD_OVERLAP", "0") == "0"


@dataclass(frozen=True)
class MXFP4GemmSelectorKey:
    """Exact dense GEMM selector coordinates."""

    orientation: str
    M: int
    N: int
    K: int


_MXFP4_GEMM_SELECTOR_ORIENTATIONS = frozenset(("forward", "dgrad", "wgrad"))

# Retained only when two uncontended GB200 runs beat the native entrypoint by
# at least 0.5% with bitwise parity. See tools/mxfp4_selector_evidence_0e9ab834.md.
_MXFP4_EXACT_GEMM_CONFIGS: dict[MXFP4GemmSelectorKey, int] = {
    MXFP4GemmSelectorKey("forward", 32768, 4096, 8192): 10,
    MXFP4GemmSelectorKey("dgrad", 32768, 8192, 4096): 10,
    MXFP4GemmSelectorKey("forward", 32768, 21504, 4096): 10,
    MXFP4GemmSelectorKey("dgrad", 32768, 21504, 4096): 10,
    MXFP4GemmSelectorKey("dgrad", 32768, 4096, 5120): 10,
    MXFP4GemmSelectorKey("dgrad", 32768, 4096, 6144): 10,
}


def _use_mxfp4_exact_gemm_selectors() -> bool:
    return os.environ.get("MXFP4_USE_EXACT_GEMM_SELECTORS", "1") != "0"


def _mxfp4_gemm_selector_key(
    M: int,
    N: int,
    K: int,
    *,
    orientation: str,
) -> MXFP4GemmSelectorKey:
    normalized_orientation = orientation.strip().lower().replace("-", "_")
    if normalized_orientation not in _MXFP4_GEMM_SELECTOR_ORIENTATIONS:
        raise ValueError(
            f"Unknown MXFP4 GEMM selector orientation {orientation!r}"
        )

    key = MXFP4GemmSelectorKey(
        orientation=normalized_orientation,
        M=int(M),
        N=int(N),
        K=int(K),
    )
    if min(key.M, key.N, key.K) <= 0:
        raise ValueError(f"MXFP4 GEMM selector dimensions must be positive: {key}")
    return key


def mxfp4_gemm_selector_table() -> dict[MXFP4GemmSelectorKey, int]:
    """Return a copy of the validated exact-shape selector table."""
    return dict(_MXFP4_EXACT_GEMM_CONFIGS)


def mxfp4_gemm_config_for_selector(
    M: int,
    N: int,
    K: int,
    *,
    orientation: str | None = None,
) -> int | None:
    """Look up a validated config for an exact physical shape and orientation.

    An orientation-less lookup is accepted only when every matching semantic
    orientation selects the same config. This lets legacy call sites consume
    unambiguous wins without collapsing the selector table back to raw shapes.
    """
    if not _use_mxfp4_exact_gemm_selectors():
        return None

    if orientation is not None:
        key = _mxfp4_gemm_selector_key(
            M,
            N,
            K,
            orientation=orientation,
        )
        return _MXFP4_EXACT_GEMM_CONFIGS.get(key)

    dimensions = (int(M), int(N), int(K))
    if min(dimensions) <= 0:
        raise ValueError(
            f"MXFP4 GEMM selector dimensions must be positive: {dimensions}"
        )
    configs = {
        config_id
        for key, config_id in _MXFP4_EXACT_GEMM_CONFIGS.items()
        if (key.M, key.N, key.K) == dimensions
    }
    return next(iter(configs)) if len(configs) == 1 else None


_NEMOTRON_PROJECTION_GEMM_CONFIGS = {
    (8192, 18688, 4096): 10,
    (8192, 4096, 18688): 10,
    (16384, 18688, 4096): 10,
    (16384, 4096, 18688): 0,
    (24576, 18688, 4096): 3,
    (24576, 4096, 18688): 0,
    (32768, 18688, 4096): 10,
    (32768, 4096, 18688): 0,
}


def mxfp4_dense_gemm_config_for_shape(
    M: int,
    N: int,
    K: int,
    *,
    orientation: str | None = None,
    _allow_exact_selector: bool = True,
) -> int | None:
    forced_config = _env_int_or_none("MXFP4_DENSE_GEMM_CONFIG_ID")
    if forced_config is not None:
        return None if forced_config < 0 else forced_config
    if not _use_mxfp4_dense_gemm_shape_config() or K % 256 != 0:
        return None

    if _allow_exact_selector:
        exact_config = mxfp4_gemm_config_for_selector(
            M,
            N,
            K,
            orientation=orientation,
        )
        if exact_config is not None:
            return exact_config

    # Exact Nemotron-H 8B Mamba input projection orientations with the native
    # 128-column padding route. All selected configs are bitwise-identical.
    nemotron_projection_config = _NEMOTRON_PROJECTION_GEMM_CONFIGS.get(
        (M, N, K)
    )
    if nemotron_projection_config is not None:
        return nemotron_projection_config

    # Exact GB200 sweeps on the Llama 1B legacy MXFP4 training hot shapes.
    if M >= 32768:
        # Llama-3 8B blog SwiGLU path: QKV/WO/FFN forward hot shapes.
        if K == 4096 and N in {4096, 6144, 14336}:
            return 10
        if K == 14336 and N == 4096:
            return 10
        if K == 2048 and N in {2048, 5632, 6144}:
            return 10
        if K == 5632 and N == 2048:
            return 10
        if K == 6144 and N == 2048:
            return 4

    # Col-major backward wgrad GEMMs expose the token dimension as K.
    if K >= 32768:
        if (M, N) in {(4096, 4096), (4096, 14336), (14336, 4096)}:
            return 10
        if M == 6144 and N == 4096:
            return 10
        if M == 2048 and N == 2048:
            return 0
        if M == 6144 and N == 2048:
            return 7
        if M == 2048 and N == 5632:
            return 4
        if M == 5632 and N == 2048:
            return 7

    return None


def mxfp4_gemm(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    orientation: str | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    mod = _load_gemm_module()
    k = int(A_fp4.size(1) * 2)
    M = int(A_fp4.size(0))
    N = int(B_fp4.size(0))
    qkv_override, qkv_config = _config_override_from_keys(
        (f"MXFP4_QKV_GEMM_CONFIG_M{M}_N{N}_K{k}",)
    )
    config_id = (
        qkv_config
        if qkv_override
        else mxfp4_dense_gemm_config_for_shape(
            M,
            N,
            k,
            orientation=orientation,
        )
    )
    if config_id is not None and hasattr(mod, "mxfp4_gemm_config"):
        mod.mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, int(config_id))
        return out
    if k % 256 != 0 and hasattr(mod, "mxfp4_gemm_k128"):
        mod.mxfp4_gemm_k128(A_fp4, A_sc, B_fp4, B_sc, out)
    else:
        mod.mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
    return out


def _use_gemm_residual_kernel() -> bool:
    return os.environ.get("MXFP4_USE_GEMM_RESIDUAL_KERNEL", "1") == "1"


def mxfp4_gemm_residual(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    mod = _load_gemm_module()
    config_id = _env_int_or_none("MXFP4_GEMM_RESIDUAL_CONFIG_ID")
    if config_id is None and _use_mxfp4_residual_gemm_shape_config():
        k = int(A_fp4.size(1) * 2)
        config_id = mxfp4_dense_gemm_config_for_shape(
            int(A_fp4.size(0)),
            int(B_fp4.size(0)),
            k,
            _allow_exact_selector=False,
        )
    if config_id is not None and hasattr(mod, "mxfp4_gemm_residual_config"):
        mod.mxfp4_gemm_residual_config(A_fp4, A_sc, B_fp4, B_sc, residual, out, int(config_id))
    elif _use_gemm_residual_kernel() and hasattr(mod, "mxfp4_gemm_residual"):
        mod.mxfp4_gemm_residual(A_fp4, A_sc, B_fp4, B_sc, residual, out)
    else:
        mod.mxfp4_gemm(A_fp4, A_sc, B_fp4, B_sc, out)
        out.add_(residual)
    return out


def mxfp4_gemm_residual_rms(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty(
            A_fp4.size(0), B_fp4.size(0),
            dtype=torch.bfloat16, device=A_fp4.device,
        )
    row_rms_partial = torch.empty(
        A_fp4.size(0), B_fp4.size(0) // 256,
        dtype=torch.float32, device=A_fp4.device,
    )
    mod = _load_gemm_module()
    if not hasattr(mod, "mxfp4_gemm_residual_rms"):
        raise AttributeError("mxfp4_gemm_residual_rms is unavailable in this backend")
    mod.mxfp4_gemm_residual_rms(
        A_fp4, A_sc, B_fp4, B_sc, residual, out, row_rms_partial
    )
    return out, row_rms_partial


def mxfp4_h_residual_carrier(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1.0e-5,
) -> MXFP4HTileCarrier:
    """Dispatch the native MX 128x128 RMS carrier producer."""
    if (
        residual.ndim != 2
        or residual.dtype is not torch.bfloat16
        or not residual.is_cuda
        or not residual.is_contiguous()
    ):
        raise RuntimeError("MX H residual must be contiguous rank-2 CUDA BF16")
    m, n = residual.shape
    if m <= 0 or n <= 0 or m % 128 or n % 256:
        raise RuntimeError(
            "MX H residual shape must be positive and divisible by 128x256"
        )
    if (
        gamma.shape != (n,)
        or gamma.dtype is not torch.bfloat16
        or not gamma.is_cuda
        or not gamma.is_contiguous()
        or gamma.device != residual.device
    ):
        raise RuntimeError(
            "MX H gamma must be contiguous CUDA BF16 [N] on the residual device"
        )
    mod = _load_gemm_module()
    if not hasattr(mod, "mxfp4_h_residual_carrier"):
        raise RuntimeError("MX H native carrier symbol is unavailable")
    row_fp4 = torch.empty(
        (m, n // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=residual.device,
    )
    row_sc = torch.empty(
        (m // 128, n // 128, 32, 16),
        dtype=torch.uint8,
        device=residual.device,
    )
    col_fp4 = torch.empty(
        (n, m // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=residual.device,
    )
    col_sc = torch.empty(
        (n // 128, m // 128, 32, 16),
        dtype=torch.uint8,
        device=residual.device,
    )
    r_tile = torch.empty(
        (m // 128, n // 128),
        dtype=torch.float32,
        device=residual.device,
    )
    z_out = torch.empty_like(residual)
    mod.mxfp4_h_residual_carrier(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        residual,
        gamma,
        z_out,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        r_tile,
        float(eps),
    )
    return MXFP4HTileCarrier(
        z_out, row_fp4, row_sc, col_fp4, col_sc, r_tile
    )


def mxfp4_h_tile_backward(
    du: torch.Tensor,
    z: torch.Tensor,
    gamma: torch.Tensor,
    r_tile: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch native tile-RMS dx and fixed-order dgamma reduction."""
    if (
        du.ndim != 2
        or du.dtype is not torch.bfloat16
        or not du.is_cuda
        or not du.is_contiguous()
    ):
        raise RuntimeError("MX H du must be contiguous rank-2 CUDA BF16")
    if (
        z.shape != du.shape
        or z.dtype is not torch.bfloat16
        or not z.is_contiguous()
    ):
        raise RuntimeError("MX H z must be contiguous BF16 matching du")
    m, n = du.shape
    if m % 128 or n % 128:
        raise RuntimeError("MX H backward shape must be divisible by 128x128")
    if (
        gamma.shape != (n,)
        or gamma.dtype is not torch.bfloat16
        or not gamma.is_contiguous()
    ):
        raise RuntimeError("MX H backward gamma must be contiguous BF16 [N]")
    if (
        r_tile.shape != (m // 128, n // 128)
        or r_tile.dtype is not torch.float32
        or not r_tile.is_contiguous()
    ):
        raise RuntimeError("MX H backward r_tile shape/dtype mismatch")
    if (
        z.device != du.device
        or gamma.device != du.device
        or r_tile.device != du.device
    ):
        raise RuntimeError("MX H backward tensors must share one CUDA device")
    mod = _load_gemm_module()
    if not hasattr(mod, "mxfp4_h_tile_backward"):
        raise RuntimeError("MX H native backward symbol is unavailable")
    dx = torch.empty_like(du)
    partial = torch.empty(
        (m // 128, n), dtype=torch.float32, device=du.device
    )
    dgamma = torch.empty_like(gamma)
    mod.mxfp4_h_tile_backward(
        du, z, gamma, r_tile, dx, partial, dgamma
    )
    return dx, dgamma


def mxfp4_gemm_residual_config(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor | None = None,
    config_id: int = 0,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    mod = _load_gemm_module()
    if _use_gemm_residual_kernel() and hasattr(mod, "mxfp4_gemm_residual_config"):
        mod.mxfp4_gemm_residual_config(A_fp4, A_sc, B_fp4, B_sc, residual, out, config_id)
    elif _use_gemm_residual_kernel() and hasattr(mod, "mxfp4_gemm_residual"):
        mod.mxfp4_gemm_residual(A_fp4, A_sc, B_fp4, B_sc, residual, out)
    else:
        mod.mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, config_id)
        out.add_(residual)
    return out


def mxfp4_gemm_rope(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_seq_len: int,
    rope_head_dim: int,
    rope_rotary_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    _load_gemm_module().mxfp4_gemm_rope(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        out,
        rope_cos,
        rope_sin,
        rope_seq_len,
        rope_head_dim,
        rope_rotary_dim,
    )
    return out


def mxfp4_rope_live_head_dim_available(head_dim: int) -> bool:
    """Return whether the loaded GEMM asset has the packed RoPE ABI for a head size."""
    mod = _load_gemm_module()
    if head_dim == 64:
        names = (
            "mxfp4_gemm_rope_live64",
            "mxfp4_gemm_rope_live64_config",
            "mxfp4_batched_gemm_rope_live64",
        )
    elif head_dim == 128:
        names = (
            "mxfp4_gemm_rope_live",
            "mxfp4_gemm_rope_live_config",
            "mxfp4_batched_gemm_rope_live",
        )
    else:
        return False
    return all(hasattr(mod, name) for name in names)


def _uses_general_live_rope(rope_cs: torch.Tensor) -> bool:
    return rope_cs.dim() >= 2 and rope_cs.size(1) != 32


def mxfp4_gemm_rope_live64(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    mod = _load_gemm_module()
    symbol = (
        "mxfp4_gemm_rope_live"
        if _uses_general_live_rope(rope_cs)
        else "mxfp4_gemm_rope_live64"
    )
    fn = getattr(mod, symbol, None)
    if fn is None:
        raise AttributeError(f"{symbol} is unavailable in this backend")
    fn(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        out,
        rope_cs,
        int(rope_seq_len),
    )
    return out


def mxfp4_gemm_config(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    out: torch.Tensor | None = None,
    config_id: int = 0,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    _load_gemm_module().mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, config_id)
    return out


def mxfp4_gemm_sqrelu_deriv_config(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    config_id: int = 0,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    mod = _load_gemm_module()
    if hasattr(mod, "mxfp4_gemm_sqrelu_deriv_config"):
        mod.mxfp4_gemm_sqrelu_deriv_config(A_fp4, A_sc, B_fp4, B_sc, x, out, config_id)
    else:
        mod.mxfp4_gemm_config(A_fp4, A_sc, B_fp4, B_sc, out, config_id)
        out.mul_(torch.relu(x)).mul_(2.0)
    return out


def mxfp4_gemm_silu_dgrad_quant(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col0_fp4: torch.Tensor,
    col0_sc: torch.Tensor,
    col1_fp4: torch.Tensor,
    col1_sc: torch.Tensor,
    config_id: int = 4,
    mode: int = 1,
) -> bool:
    mod = _load_gemm_module()
    if not hasattr(mod, "mxfp4_gemm_silu_dgrad_quant"):
        return False
    mod.mxfp4_gemm_silu_dgrad_quant(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        h3,
        h1_raw,
        row_fp4,
        row_sc,
        col0_fp4,
        col0_sc,
        col1_fp4,
        col1_sc,
        int(config_id),
        int(mode),
    )
    return True


def mxfp4_gemm_silu_dgrad_from_sigmoid_quant(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    sig_h1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col0_fp4: torch.Tensor,
    col0_sc: torch.Tensor,
    col1_fp4: torch.Tensor,
    col1_sc: torch.Tensor,
    config_id: int = 4,
    mode: int = 1,
) -> bool:
    mod = _load_gemm_module()
    if not hasattr(mod, "mxfp4_gemm_silu_dgrad_from_sigmoid_quant"):
        return False
    mod.mxfp4_gemm_silu_dgrad_from_sigmoid_quant(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        h3,
        h1_raw,
        sig_h1,
        row_fp4,
        row_sc,
        col0_fp4,
        col0_sc,
        col1_fp4,
        col1_sc,
        int(config_id),
        int(mode),
    )
    return True


def mxfp4_gemm_silu_dgrad_from_sigmoid_row_bf16_quant(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    h3: torch.Tensor,
    h1_raw: torch.Tensor,
    sig_h1: torch.Tensor,
    dh0: torch.Tensor,
    dh1: torch.Tensor,
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    config_id: int = 44,
    mode: int = 1,
) -> bool:
    mod = _load_gemm_module()
    name = "mxfp4_gemm_silu_dgrad_from_sigmoid_row_bf16_quant"
    if not hasattr(mod, name):
        return False
    getattr(mod, name)(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        h3,
        h1_raw,
        sig_h1,
        dh0,
        dh1,
        row_fp4,
        row_sc,
        int(config_id),
        int(mode),
    )
    return True


def mxfp4_gemm_rope_config(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_seq_len: int,
    rope_head_dim: int,
    rope_rotary_dim: int,
    out: torch.Tensor | None = None,
    config_id: int = 0,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    _load_gemm_module().mxfp4_gemm_rope_config(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        out,
        rope_cos,
        rope_sin,
        rope_seq_len,
        rope_head_dim,
        rope_rotary_dim,
        config_id,
    )
    return out


def mxfp4_gemm_rope_live64_config(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    B_fp4: torch.Tensor,
    B_sc: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    out: torch.Tensor | None = None,
    config_id: int = 0,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(A_fp4.size(0), B_fp4.size(0), dtype=torch.bfloat16, device=A_fp4.device)
    mod = _load_gemm_module()
    symbol = (
        "mxfp4_gemm_rope_live64_config"
        if not _uses_general_live_rope(rope_cs)
        else "mxfp4_gemm_rope_live_config"
    )
    fn = getattr(mod, symbol, None)
    if fn is None:
        raise AttributeError(f"{symbol} is unavailable in this backend")
    fn(
        A_fp4,
        A_sc,
        B_fp4,
        B_sc,
        out,
        rope_cs,
        int(rope_seq_len),
        config_id,
    )
    return out


def mxfp4_batched_gemm(
    A_list: list[torch.Tensor],
    A_sc_list: list[torch.Tensor],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    D_list: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    if D_list is None:
        D_list = [
            torch.empty(A.size(0), B.size(0), dtype=torch.bfloat16, device=A.device)
            for A, B in zip(A_list, B_list)
        ]
    config_id = None
    forced_config = _env_int_or_none("MXFP4_BATCHED_GEMM_CONFIG_ID")
    if forced_config is not None:
        config_id = forced_config
    elif _use_mxfp4_batched_gemm_tuning() and A_list and B_list and D_list:
        M = int(D_list[0].size(0))
        N_out = int(D_list[0].size(1))
        K = int(A_list[0].size(1) * 2)
        num_batches = len(A_list)
        # Exact-shape wins from live batched sweeps on the real FFN-up hot paths.
        if M >= 65536 and K == 2048 and num_batches == 2 and N_out == 1024:
            config_id = 4
        elif M >= 65536 and K == 2048 and num_batches == 2 and N_out == 5632:
            config_id = 6
        elif M >= 65536 and K == 2048 and num_batches == 2 and N_out == 8192:
            config_id = 0
        elif M == 14336 and K == 32768 and num_batches == 2 and N_out == 4096:
            config_id = 2
        config_id = _mxfp4_batched_gemm_config_override(
            M, N_out, K, num_batches, config_id
        )
    gemm_mod = _load_gemm_module()
    if config_id is None or not hasattr(gemm_mod, "mxfp4_batched_gemm_config"):
        gemm_mod.mxfp4_batched_gemm(A_list, A_sc_list, B_list, B_sc_list, D_list)
    else:
        gemm_mod.mxfp4_batched_gemm_config(A_list, A_sc_list, B_list, B_sc_list, D_list, config_id)
    return D_list


def mxfp4_grouped_gemm_strided(
    A: torch.Tensor,
    A_sc: torch.Tensor,
    B: torch.Tensor,
    B_sc: torch.Tensor,
    D: torch.Tensor,
    num_batches: int,
    m_per_batch: int,
    n_per_batch: int,
    k_per_batch: int,
    a_row_stride: int,
    a_k_stride: int,
    b_row_stride: int,
    b_k_stride: int,
    d_row_stride: int,
) -> torch.Tensor:
    gemm_mod = _load_gemm_module()
    name = "mxfp4_grouped_gemm_strided"
    if not hasattr(gemm_mod, name):
        raise AttributeError(f"{name} is unavailable in this backend")
    raw_config = os.environ.get("MXFP4_DEEPSEEK_GROUPED_STRIDED_GEMM_CONFIG", "").strip()
    config_id = int(raw_config) if raw_config else -1
    fn = getattr(gemm_mod, name)
    args = (
        A,
        A_sc,
        B,
        B_sc,
        D,
        int(num_batches),
        int(m_per_batch),
        int(n_per_batch),
        int(k_per_batch),
        int(a_row_stride),
        int(a_k_stride),
        int(b_row_stride),
        int(b_k_stride),
        int(d_row_stride),
    )
    if config_id >= 0:
        try:
            fn(*args, config_id)
        except TypeError:
            fn(*args)
    else:
        fn(*args)
    return D


def mxfp4_batched_gemm_config(
    A_list: list[torch.Tensor],
    A_sc_list: list[torch.Tensor],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    D_list: list[torch.Tensor] | None = None,
    config_id: int = 0,
) -> list[torch.Tensor]:
    if D_list is None:
        D_list = [
            torch.empty(A.size(0), B.size(0), dtype=torch.bfloat16, device=A.device)
            for A, B in zip(A_list, B_list)
        ]
    _load_gemm_module().mxfp4_batched_gemm_config(
        A_list, A_sc_list, B_list, B_sc_list, D_list, int(config_id)
    )
    return D_list


def mxfp4_batched_gemm_rope(
    A_list: list[torch.Tensor],
    A_sc_list: list[torch.Tensor],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    rope_cos_list: list[torch.Tensor],
    rope_sin_list: list[torch.Tensor],
    rope_seq_len_list: list[int],
    rope_head_dim_list: list[int],
    rope_rotary_dim_list: list[int],
    D_list: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    if D_list is None:
        D_list = [
            torch.empty(A.size(0), B.size(0), dtype=torch.bfloat16, device=A.device)
            for A, B in zip(A_list, B_list)
        ]
    _load_gemm_module().mxfp4_batched_gemm_rope(
        A_list,
        A_sc_list,
        B_list,
        B_sc_list,
        D_list,
        rope_cos_list,
        rope_sin_list,
        rope_seq_len_list,
        rope_head_dim_list,
        rope_rotary_dim_list,
    )
    return D_list


def mxfp4_batched_gemm_rope_live64(
    A_list: list[torch.Tensor],
    A_sc_list: list[torch.Tensor],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    rope_cs_list: list[torch.Tensor],
    rope_seq_len_list: list[int],
    D_list: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    if D_list is None:
        D_list = [
            torch.empty(A.size(0), B.size(0), dtype=torch.bfloat16, device=A.device)
            for A, B in zip(A_list, B_list)
        ]
    mod = _load_gemm_module()
    use_general_live = any(
        t.numel() and _uses_general_live_rope(t) for t in rope_cs_list
    )
    native_symbol = (
        "mxfp4_batched_gemm_rope_live"
        if use_general_live
        else "mxfp4_batched_gemm_rope_live64"
    )
    config_symbol = (
        "mxfp4_batched_gemm_rope_live_config"
        if use_general_live
        else "mxfp4_batched_gemm_rope_live64_config"
    )
    global_config = _env_int_or_none(
        "MXFP4_BATCHED_GEMM_ROPE_LIVE64_CONFIG_ID"
    )
    if global_config is None:
        global_config = _env_int_or_none("MXFP4_BATCHED_GEMM_CONFIG_ID")
    global_override = global_config is not None
    forced_config = (
        None
        if global_config is not None and global_config < 0
        else global_config
    )
    if not global_override and A_list and D_list:
        M = int(D_list[0].size(0))
        N_out = int(D_list[0].size(1))
        K = int(A_list[0].size(1) * 2)
        num_batches = len(A_list)
        if _use_mxfp4_dense_gemm_shape_config():
            if (
                M >= 65536
                and K == 2048
                and num_batches == 2
                and N_out == 1024
            ):
                forced_config = 10
            elif (
                M >= 65536
                and K == 2048
                and num_batches == 3
                and N_out == 2048
            ):
                forced_config = 6
        forced_config = _mxfp4_batched_gemm_config_override(
            M,
            N_out,
            K,
            num_batches,
            forced_config,
        )
    if forced_config is not None and hasattr(mod, config_symbol):
        getattr(mod, config_symbol)(
            A_list,
            A_sc_list,
            B_list,
            B_sc_list,
            D_list,
            rope_cs_list,
            [int(seq_len) for seq_len in rope_seq_len_list],
            int(forced_config),
        )
        return D_list
    fn = getattr(mod, native_symbol, None)
    if fn is None:
        raise AttributeError(f"{native_symbol} is unavailable in this backend")
    fn(
        A_list,
        A_sc_list,
        B_list,
        B_sc_list,
        D_list,
        rope_cs_list,
        [int(seq_len) for seq_len in rope_seq_len_list],
    )
    return D_list


def mxfp4_batched_gemm_rope_live64_config(
    A_list: list[torch.Tensor],
    A_sc_list: list[torch.Tensor],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    rope_cs_list: list[torch.Tensor],
    rope_seq_len_list: list[int],
    D_list: list[torch.Tensor] | None = None,
    config_id: int = 0,
) -> list[torch.Tensor]:
    if D_list is None:
        D_list = [
            torch.empty(A.size(0), B.size(0), dtype=torch.bfloat16, device=A.device)
            for A, B in zip(A_list, B_list)
        ]
    mod = _load_gemm_module()
    use_general_live = any(
        t.numel() and _uses_general_live_rope(t) for t in rope_cs_list
    )
    symbol = (
        "mxfp4_batched_gemm_rope_live_config"
        if use_general_live
        else "mxfp4_batched_gemm_rope_live64_config"
    )
    fn = getattr(mod, symbol, None)
    if fn is None:
        raise AttributeError(f"{symbol} is unavailable in this backend")
    fn(
        A_list,
        A_sc_list,
        B_list,
        B_sc_list,
        D_list,
        rope_cs_list,
        [int(seq_len) for seq_len in rope_seq_len_list],
        int(config_id),
    )
    return D_list


def mxfp4_batched_qkv_gemm_rope_live64(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    q_fp4: torch.Tensor,
    q_sc: torch.Tensor,
    k_fp4: torch.Tensor,
    k_sc: torch.Tensor,
    v_fp4: torch.Tensor,
    v_sc: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> list[torch.Tensor]:
    rope_empty = torch.empty(0, dtype=torch.float32, device=rope_cs.device)
    A_list = [A_fp4, A_fp4, A_fp4]
    A_sc_list = [A_sc, A_sc, A_sc]
    B_list = [q_fp4, k_fp4, v_fp4]
    B_sc_list = [q_sc, k_sc, v_sc]
    rope_cs_list = [rope_cs, rope_cs, rope_empty]
    rope_seq_len_list = [rope_seq_len, rope_seq_len, 0]
    D_list = [q_out, k_out, v_out]

    M = int(q_out.size(0))
    N = sum(int(out.size(1)) for out in D_list)
    K = int(A_fp4.size(1) * 2)
    qkv_override, qkv_config = _mxfp4_gemm_config_override_status(
        "MXFP4_QKV_GEMM_CONFIG",
        M,
        N,
        K,
    )
    if qkv_override:
        mod = _load_gemm_module()
        seq_lens = [int(seq_len) for seq_len in rope_seq_len_list]
        use_general_live = _uses_general_live_rope(rope_cs)
        native_symbol = (
            "mxfp4_batched_gemm_rope_live"
            if use_general_live
            else "mxfp4_batched_gemm_rope_live64"
        )
        config_symbol = (
            "mxfp4_batched_gemm_rope_live_config"
            if use_general_live
            else "mxfp4_batched_gemm_rope_live64_config"
        )
        if (
            qkv_config is not None
            and hasattr(mod, config_symbol)
        ):
            getattr(mod, config_symbol)(
                A_list,
                A_sc_list,
                B_list,
                B_sc_list,
                D_list,
                rope_cs_list,
                seq_lens,
                int(qkv_config),
            )
        else:
            fn = getattr(mod, native_symbol, None)
            if fn is None:
                raise AttributeError(f"{native_symbol} is unavailable in this backend")
            fn(
                A_list,
                A_sc_list,
                B_list,
                B_sc_list,
                D_list,
                rope_cs_list,
                seq_lens,
            )
        return D_list

    return mxfp4_batched_gemm_rope_live64(
        A_list,
        A_sc_list,
        B_list,
        B_sc_list,
        rope_cs_list,
        rope_seq_len_list,
        D_list,
    )


def mxfp4_batched_kv_gemm_rope_live64(
    A_fp4: torch.Tensor,
    A_sc: torch.Tensor,
    k_fp4: torch.Tensor,
    k_sc: torch.Tensor,
    v_fp4: torch.Tensor,
    v_sc: torch.Tensor,
    rope_cs: torch.Tensor,
    rope_seq_len: int,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> list[torch.Tensor]:
    rope_empty = torch.empty(0, dtype=torch.float32, device=rope_cs.device)
    return mxfp4_batched_gemm_rope_live64(
        [A_fp4, A_fp4],
        [A_sc, A_sc],
        [k_fp4, v_fp4],
        [k_sc, v_sc],
        [rope_cs, rope_empty],
        [rope_seq_len, 0],
        [k_out, v_out],
    )


def mxfp4_split2_dgrad_strided_onepass_gemm(
    A_full: torch.Tensor,
    A_sc_list: list[torch.Tensor],
    A_col_offsets: list[int],
    A_col_widths: list[int],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    D_out: torch.Tensor,
    config_idx: int = -1,
) -> torch.Tensor:
    _load_gemm_module().mxfp4_split2_dgrad_strided_onepass_gemm(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        D_out,
        config_idx,
    )


def mxfp4_split2_dgrad_strided_onepass_h_gemm(
    A_full: torch.Tensor,
    A_sc_list: list[torch.Tensor],
    A_col_offsets: list[int],
    A_col_widths: list[int],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    z: torch.Tensor,
    gamma: torch.Tensor,
    r_tile: torch.Tensor,
    D_out: torch.Tensor,
    config_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load_gemm_module()
    if not hasattr(mod, "mxfp4_split2_dgrad_strided_onepass_h_gemm"):
        raise RuntimeError("MX H fused split2 dgrad symbol is unavailable")
    m, n = D_out.shape
    partial = torch.empty((m // 128, n), dtype=torch.float32, device=D_out.device)
    dgamma = torch.empty_like(gamma)
    mod.mxfp4_split2_dgrad_strided_onepass_h_gemm(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        z,
        gamma,
        r_tile,
        D_out,
        partial,
        dgamma,
        config_idx,
    )
    return D_out, dgamma


def mxfp4_split3_dgrad_strided_onepass_gemm(
    A_full: torch.Tensor,
    A_sc_list: list[torch.Tensor],
    A_col_offsets: list[int],
    A_col_widths: list[int],
    B_list: list[torch.Tensor],
    B_sc_list: list[torch.Tensor],
    D_out: torch.Tensor,
    config_idx: int = -1,
) -> None:
    mod = _load_gemm_module()
    if not hasattr(mod, "mxfp4_split3_dgrad_strided_onepass_gemm"):
        raise AttributeError("mxfp4_split3_dgrad_strided_onepass_gemm is unavailable in this backend")
    mod.mxfp4_split3_dgrad_strided_onepass_gemm(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        D_out,
        config_idx,
    )
    return D_out
