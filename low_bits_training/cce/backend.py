#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import ctypes
import glob
import importlib
import importlib.util
import logging
import os
import sys
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)
_COMMON_EVAL_COUNTER = 0


try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:  # pragma: no cover - older torch builds without DTensor
    DTensor = None
    Replicate = None


try:
    from cut_cross_entropy import linear_cross_entropy as _linear_cross_entropy

    linear_cross_entropy = torch._dynamo.disable(_linear_cross_entropy)
    raw_linear_cross_entropy = _linear_cross_entropy
except ImportError:
    linear_cross_entropy = None
    raw_linear_cross_entropy = None


_NV_RUNTIME = None
_MX_RUNTIME = None
_MXFP4_BACKEND_PY = None
_FP4_CCE_TK_V4 = None


def cce_path_handles_loss(job_config) -> bool:
    return bool(
        getattr(job_config.training, "enable_cce", False)
        or getattr(job_config.fp4_cce, "enabled", False)
    )


def _guard_mxfp4_cce_env() -> None:
    if os.environ.get("FP4_CCE_MXFP4_ALLOW_BWD_OVERLAP", "0") == "1":
        logger.warning(
            "FP4_CCE_MXFP4_ALLOW_BWD_OVERLAP=1 leaves MXFP4 fused backward "
            "overlap/cache enabled for mxfp4 CCE; this path is known to hit "
            "asynchronous CUDA launch failures."
        )
        return

    changed = []
    for name in ("MXFP4_USE_BWD_WGRAD_OVERLAP", "MXFP4_USE_BWD_STATE_CACHE"):
        old = os.environ.get(name)
        if old != "0":
            os.environ[name] = "0"
            changed.append(f"{name}={old if old is not None else '<unset>'}->0")
    if changed:
        logger.info(
            "Disabled MXFP4 fused backward overlap/cache for mxfp4 CCE: %s",
            ", ".join(changed),
        )


def make_training_loss_backend(
    backend: str,
    implementation: str = "v2",
    quant_mode: str = "enc",
    ignore_index: int = -100,
    filter_eps: float = 0.0,
    forward_precision: str = "bf16",
    backward_precision: str = "bf16",
):
    if backend == "triton_bf16":
        return _TritonBF16Backend(ignore_index=ignore_index, filter_eps=filter_eps)
    if backend == "torch_compile_bf16":
        return _TorchCompileBF16Backend(ignore_index=ignore_index)
    if backend == "native_mxfp4":
        return _NativeMXFP4PrecisionBackend(
            ignore_index=ignore_index,
            implementation=implementation,
            quant_mode=quant_mode,
            filter_eps=filter_eps,
            forward_precision=forward_precision,
            backward_precision=backward_precision,
        )
    if backend == "nvfp4":
        if quant_mode not in ("native", "enc", "dec"):
            raise ValueError(
                f"NVFP4 backend requires quant_mode in {{'native','enc','dec'}}, got {quant_mode!r}."
            )
        return _NVFP4Backend(
            ignore_index=ignore_index,
            implementation=implementation,
            quant_mode=quant_mode,
            filter_eps=filter_eps,
        )
    if backend == "mxfp4":
        if quant_mode not in ("rte", "enc", "dec"):
            raise ValueError(
                f"MXFP4 backend requires quant_mode in {{'rte','enc','dec'}}, got {quant_mode!r}."
            )
        _guard_mxfp4_cce_env()
        return _MXFP4Backend(
            ignore_index=ignore_index,
            implementation=implementation,
            quant_mode=quant_mode,
            filter_eps=filter_eps,
        )
    raise ValueError(f"Unsupported backend {backend!r}")


def _resolve_filter_eps(filter_eps, dtype=torch.bfloat16):
    if filter_eps is None:
        return 0.0
    if isinstance(filter_eps, str):
        val = filter_eps.strip().lower()
        if val in {"", "0", "none"}:
            return 0.0
        if val == "auto":
            return float(torch.finfo(dtype).eps / 32)
        raise ValueError(f"Unsupported filter_eps string {filter_eps!r}")
    return float(filter_eps)


def _cut_cross_entropy_filter_eps(filter_eps):
    """Translate our zero-is-exact setting to cut-cross-entropy semantics."""
    filter_eps = float(filter_eps)
    return filter_eps if filter_eps > 0.0 else None


def _nv_kernel_filter_eps(filter_eps: float, grad_scale: float) -> float:
    filter_eps = float(filter_eps)
    if filter_eps <= 0.0:
        return 0.0
    return filter_eps * abs(float(grad_scale))


def _nv_row_sr_enabled():
    val = os.environ.get("LBT_NV_ROW_SR", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _nv_fwd_hidden_row_sr_enabled():
    val = os.environ.get("LBT_NV_FWD_HIDDEN_ROW_SR", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _flag_enabled(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _flag_value(name: str) -> bool | None:
    val = os.environ.get(name)
    if val is None:
        return None
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _assume_nonempty_labels() -> bool:
    return _flag_enabled("FP4_CCE_ASSUME_NONEMPTY_LABELS")


def _nvfp4_v4_prequant_x_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_NVFP4_PREQUANT_X")


def _nvfp4_mxfp8_forward_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_NVFP4_MXFP8_FORWARD")


def _nvfp4_mxfp8_native_col_enabled() -> bool:
    return (
        _nvfp4_mxfp8_forward_enabled()
        and not _flag_enabled("FP4_CCE_V4_MXFP4_G_CACHE")
        and not _flag_enabled("FP4_CCE_V4_MXFP8_G_CACHE")
    )


def _mixed_dw_mxfp8_cols_enabled() -> bool:
    """Use MXFP8 only for the mixed head's dWeight column operands."""

    return _flag_enabled("FP4_CCE_V4_MIXED_DW_MXFP8_COLS")


def _nvfp4_direct_fp8_forward_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD")


def _nvfp4_mxfp4_forward_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_NVFP4_MXFP4_FORWARD")


def _mxfp4_v4_prequant_x_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_MXFP4_PREQUANT_X")


def _v4_fused_x_producer_enabled(backend_name: str) -> bool:
    scoped = _flag_value(f"FP4_CCE_V4_{backend_name.upper()}_FUSED_X_PRODUCER")
    if scoped is not None:
        return scoped
    generic = _flag_value("FP4_CCE_V4_FUSED_X_PRODUCER")
    if generic is not None:
        return generic
    return False


def _v4_fused_x_producer_enabled_for_backend(backend) -> bool:
    if backend.name not in ("nvfp4", "mxfp4"):
        return False
    if getattr(backend, "implementation", None) != "v4":
        return False
    return _v4_fused_x_producer_enabled(backend.name)


def _v4_fused_x_producer_sync_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_FUSED_X_PRODUCER_SYNC")


def _v4_fused_x_producer_pre_sync_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_FUSED_X_PRODUCER_PRE_SYNC")


def _v4_fused_x_producer_quant_only_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_FUSED_X_PRODUCER_QUANT_ONLY")


def _fp4_cce_bf16_weight(weight):
    weight = _local_tensor_for_cce(weight)
    if weight.dtype == torch.bfloat16:
        return weight.contiguous()
    return weight.to(dtype=torch.bfloat16).contiguous()


def _local_tensor_for_cce(tensor):
    if DTensor is not None and isinstance(tensor, DTensor):
        placements = tuple(getattr(tensor, "placements", ()))
        if Replicate is not None and placements and not all(isinstance(p, Replicate) for p in placements):
            tensor = tensor.redistribute(placements=tuple(Replicate() for _ in placements))
        return tensor.to_local()
    return tensor


def _flag_disabled(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in {"0", "false", "no", "off"}


def _nv_cce_pre_quant_sync_enabled() -> bool:
    value = os.environ.get("LBT_NV_CCE_PRE_QUANT_SYNC")
    if value is not None:
        return not _flag_disabled("LBT_NV_CCE_PRE_QUANT_SYNC")
    return os.environ.get("USE_TK_GEMM", "0").strip() == "1"


def _nv_cce_internal_sync_enabled() -> bool:
    return not _flag_disabled("LBT_NV_CCE_INTERNAL_SYNC")


def _nv_cce_sync_cuda(device):
    if _nv_cce_internal_sync_enabled():
        _sync_cuda(device)


def _nv_v5_bwd_mode() -> str:
    raw = os.environ.get("LBT_NV_V5_BWD_MODE", "split").strip().lower()
    aliases = {
        "": "split",
        "default": "split",
        "triton": "triton_l4_sg8",
        "triton-style": "triton_l4_sg8",
        "triton_style": "triton_l4_sg8",
        "triton_l4": "triton_l4_sg8",
        "triton_e2": "triton_l4_sg8_e2",
        "triton-e2": "triton_l4_sg8_e2",
        "triton_e1": "triton_l4_sg8_e1",
        "triton-e1": "triton_l4_sg8_e1",
        "triton_l1e1": "triton_l1_sg8_e1",
        "triton-l1e1": "triton_l1_sg8_e1",
        "triton_l1sg4e1": "triton_l1_sg4_e1",
        "triton-l1sg4e1": "triton_l1_sg4_e1",
        "publicv3": "publicv3",
        "public-v3": "publicv3",
    }
    mode = aliases.get(raw, raw)
    valid = {
        "split",
        "triton_l4_sg8",
        "triton_l4_sg8_e2",
        "triton_l4_sg8_e1",
        "triton_l1_sg8_e1",
        "triton_l1_sg4_e1",
        "publicv3",
    }
    if mode not in valid:
        raise ValueError(
            f"Unsupported LBT_NV_V5_BWD_MODE={raw!r}; expected one of {sorted(valid)}."
        )
    return mode


def _nv_v5_bwd_uses_bf16_saved(mode: str) -> bool:
    return mode.startswith("triton_")


def _nv_nuclear_bwd_enabled():
    return _flag_enabled("LBT_FP4_CCE_NUCLEAR_BWD") or _flag_enabled("LBT_NV_NUCLEAR_BWD")


def _mx_nuclear_bwd_enabled():
    return _flag_enabled("LBT_FP4_CCE_NUCLEAR_BWD") or _flag_enabled("LBT_MX_NUCLEAR_BWD")


def _nv_true_nuclear_bwd_enabled():
    return _flag_enabled("LBT_FP4_CCE_TRUE_NUCLEAR_BWD") or _flag_enabled("LBT_NV_TRUE_NUCLEAR_BWD")


def _mx_true_nuclear_bwd_enabled():
    return _flag_enabled("LBT_FP4_CCE_TRUE_NUCLEAR_BWD") or _flag_enabled("LBT_MX_TRUE_NUCLEAR_BWD")


def _nv_native_v3_dense_enabled():
    return _flag_enabled("LBT_FORCE_NV_NATIVE_V3_DENSE")


def _mx_native_v3_dense_enabled():
    return _flag_enabled("LBT_FORCE_MX_NATIVE_V3_DENSE")


def _common_eval_enabled():
    return _flag_enabled("LBT_FP4_CCE_COMMON_EVAL")


def _common_eval_filter_eps():
    raw = os.environ.get("LBT_FP4_CCE_COMMON_EVAL_FILTER_EPS", "").strip()
    if not raw:
        return 0.0
    return _resolve_filter_eps(raw)


def _common_eval_every():
    raw = os.environ.get("LBT_FP4_CCE_COMMON_EVAL_EVERY", "").strip()
    if not raw:
        return 1
    return max(int(raw), 1)


def _common_eval_backend():
    raw = os.environ.get(
        "LBT_FP4_CCE_COMMON_EVAL_BACKEND", "native_mxfp4"
    ).strip().lower()
    aliases = {
        "native": "native_mxfp4",
        "native_mxfp4": "native_mxfp4",
        "cut_ce": "triton_bf16",
        "cut_cross_entropy": "triton_bf16",
        "triton_bf16": "triton_bf16",
    }
    try:
        return aliases[raw]
    except KeyError as error:
        raise ValueError(
            "LBT_FP4_CCE_COMMON_EVAL_BACKEND must be native_mxfp4 or "
            f"triton_bf16, got {raw!r}"
        ) from error


def _common_eval_max_relative_gap():
    raw = os.environ.get("LBT_FP4_CCE_COMMON_EVAL_MAX_REL_GAP", "").strip()
    if not raw:
        return None
    value = float(raw)
    if value < 0.0:
        raise ValueError("LBT_FP4_CCE_COMMON_EVAL_MAX_REL_GAP must be non-negative")
    return value


def _module_name_from_path(path):
    base = os.path.basename(path)
    if ".cpython-" in base:
        return base.split(".cpython-", maxsplit=1)[0]
    if base.endswith(".so"):
        return base[:-3]
    return os.path.splitext(base)[0]


def _preload_torch_python():
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    ctypes.CDLL(os.path.join(torch_lib, "libtorch_python.so"), mode=ctypes.RTLD_GLOBAL)


def _load_so(path):
    module_name = _module_name_from_path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_attr(module, *names):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    joined = ", ".join(names)
    raise AttributeError(f"{module.__name__} does not export any of: {joined}")


def _resolve_optional_attr(module, *names):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _load_mxfp4_backend_py():
    global _MXFP4_BACKEND_PY
    if _MXFP4_BACKEND_PY is not None:
        return _MXFP4_BACKEND_PY
    mod_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "quantization", "mxfp4_backend.py")
    )
    spec = importlib.util.spec_from_file_location("low_bits_training_quantization_mxfp4_backend", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MXFP4_BACKEND_PY = module
    return module


def _fp4_matmul_roots():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    env_root = os.environ.get("FP4_MATMUL_ROOT")
    sibling_peer = os.path.join(
        os.path.dirname(repo_root),
        os.path.basename(repo_root).replace("low-bits-training", "fp4_matmul", 1),
    )
    sibling_roots = sorted(glob.glob(os.path.join(repo_root, "..", "fp4_matmul*")))
    candidates = [
        env_root,
        os.path.normpath(os.path.join(repo_root, "..", "cce", "fp4_matmul")),
        os.path.normpath(os.path.join(repo_root, "..", "fp4_matmul")),
        os.path.normpath(sibling_peer),
    ]
    candidates.extend(os.path.normpath(path) for path in sibling_roots)
    candidates.extend([
        "/opt/mfu/EXTERNAL_PATH",
        "/opt/mfu/EXTERNAL_PATH",
        "/tmp/fp4_matmul_v4_pcache",
        "/opt/mfu/EXTERNAL_PATH",
        "/opt/mfu/EXTERNAL_PATH",
        "/opt/mfu/EXTERNAL_PATH",
    ])
    out = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.normpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            out.append(candidate)
    return out


def _resolve_existing_path(label, rel_patterns):
    checked = []
    for root in _fp4_matmul_roots():
        for rel in rel_patterns:
            pattern = rel if os.path.isabs(rel) else os.path.join(root, rel)
            checked.append(pattern)
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[0]
    joined = "\n  ".join(checked)
    raise FileNotFoundError(
        f"Could not find {label}. Checked:\n  {joined}\n"
        "Build the missing fp4_matmul extension before enabling fp4_cce."
    )


def _try_resolve_existing_path(rel_patterns):
    for root in _fp4_matmul_roots():
        for rel in rel_patterns:
            pattern = rel if os.path.isabs(rel) else os.path.join(root, rel)
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[0]
    return None


def _fp4_cce_tk_v4_roots():
    candidates = [os.environ.get("FP4_CCE_TK_ROOT")]
    candidates.extend(_fp4_matmul_roots())
    out = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.normpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            out.append(candidate)
    return out


def _clear_fp4_cce_tk_imports():
    for name in list(sys.modules):
        if name == "fp4_cce_TK" or name.startswith("fp4_cce_TK."):
            sys.modules.pop(name, None)


def _fp4_cce_tk_imports_match_root(root):
    """Return whether every loaded runtime module belongs to ``root``.

    The trainer may import ``v4_common`` early to install checkpoint-owned SR
    state.  Re-importing the same package later would silently orphan that
    module-global state, so preserve modules that already resolve to the exact
    selected runtime root.  Any ambiguous or stale module remains fail-closed
    and is evicted by the caller.
    """

    package_dir = os.path.realpath(os.path.join(root, "fp4_cce_TK"))
    loaded = [
        module
        for name, module in sys.modules.items()
        if name == "fp4_cce_TK" or name.startswith("fp4_cce_TK.")
    ]
    for module in loaded:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        module_file = os.path.realpath(module_file)
        try:
            if os.path.commonpath((package_dir, module_file)) != package_dir:
                return False
        except ValueError:
            return False
    return True


def _load_fp4_cce_tk_v4():
    global _FP4_CCE_TK_V4
    if _FP4_CCE_TK_V4 is not None:
        return _FP4_CCE_TK_V4

    checked = []
    for root in _fp4_cce_tk_v4_roots():
        pkg_dir = os.path.join(root, "fp4_cce_TK")
        nv_path = os.path.join(pkg_dir, "nvfp4_cce_tk.py")
        mx_path = os.path.join(pkg_dir, "mxfp4_cce_tk.py")
        checked.extend([nv_path, mx_path])
        if not (os.path.isfile(nv_path) and os.path.isfile(mx_path)):
            continue

        if not _fp4_cce_tk_imports_match_root(root):
            _clear_fp4_cce_tk_imports()
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)

        nv_mod = importlib.import_module("fp4_cce_TK.nvfp4_cce_tk")
        mx_mod = importlib.import_module("fp4_cce_TK.mxfp4_cce_tk")
        _FP4_CCE_TK_V4 = types.SimpleNamespace(
            NVFP4Quantized=getattr(nv_mod, "NVFP4Quantized", None),
            MXFP8Quantized=getattr(nv_mod, "MXFP8Quantized", None),
            MXFP4Quantized=getattr(nv_mod, "MXFP4Quantized", None),
            DirectFP8Quantized=getattr(nv_mod, "DirectFP8Quantized", None),
            nvfp4_cce_tk_v4_pcache=nv_mod.nvfp4_cce_tk_v4_pcache,
            nvfp4_cce_tk_v4_pcache_prequantized_x=getattr(
                nv_mod, "nvfp4_cce_tk_v4_pcache_prequantized_x", None
            ),
            nvfp4_cce_tk_v4_vocab_parallel=getattr(
                nv_mod, "nvfp4_cce_tk_v4_vocab_parallel", None
            ),
            quantize_nvfp4_row_and_col_tk=getattr(nv_mod, "quantize_nvfp4_row_and_col_tk", None),
            quantize_nvfp4_norm_row_and_col_tk=getattr(nv_mod, "quantize_nvfp4_norm_row_and_col_tk", None),
            quantize_nvfp4_norm_row_and_col_with_output_tk=getattr(
                nv_mod, "quantize_nvfp4_norm_row_and_col_with_output_tk", None
            ),
            quantize_mxfp8_row_mxfp4_col=getattr(
                nv_mod, "quantize_mxfp8_row_mxfp4_col", None
            ),
            quantize_mxfp8_row_nvfp4_col_localcta_v4=getattr(
                nv_mod, "quantize_mxfp8_row_nvfp4_col_localcta_v4", None
            ),
            quantize_mxfp8_row_and_col_fused=getattr(
                nv_mod, "quantize_mxfp8_row_and_col_fused", None
            ),
            quantize_direct_fp8_row_mxfp4_col=getattr(
                nv_mod, "quantize_direct_fp8_row_mxfp4_col", None
            ),
            quantize_mxfp8_norm_row_mxfp4_col_with_output_localcta_v4=getattr(
                nv_mod,
                "quantize_mxfp8_norm_row_mxfp4_col_with_output_localcta_v4",
                None,
            ),
            quantize_mxfp8_norm_row_nvfp4_col_with_output_localcta_v4=getattr(
                nv_mod,
                "quantize_mxfp8_norm_row_nvfp4_col_with_output_localcta_v4",
                None,
            ),
            quantize_direct_fp8_norm_row_mxfp4_col_with_output_localcta_v4=getattr(
                nv_mod,
                "quantize_direct_fp8_norm_row_mxfp4_col_with_output_localcta_v4",
                None,
            ),
            quantize_mxfp4_row_nvfp4_col_v5=getattr(
                nv_mod, "quantize_mxfp4_row_nvfp4_col_v5", None
            ),
            mxfp4_cce_tk_v4_pcache=mx_mod.mxfp4_cce_tk_v4_pcache,
            mxfp4_cce_tk_native_precision=getattr(
                mx_mod, "mxfp4_cce_tk_native_precision", None
            ),
            mxfp4_cce_tk_v4_pcache_prequantized_x=getattr(
                mx_mod, "mxfp4_cce_tk_v4_pcache_prequantized_x", None
            ),
            mxfp4_cce_tk_v4_vocab_parallel=getattr(
                mx_mod, "mxfp4_cce_tk_v4_vocab_parallel", None
            ),
            mxfp4_cce_tk_v4_vocab_parallel_prequantized_x=getattr(
                mx_mod, "mxfp4_cce_tk_v4_vocab_parallel_prequantized_x", None
            ),
            quantize_mxfp4_row_and_col_tk=getattr(mx_mod, "quantize_mxfp4_row_and_col_tk", None),
            quantize_mxfp4_norm_row_and_col_tk=getattr(mx_mod, "quantize_mxfp4_norm_row_and_col_tk", None),
            quantize_mxfp4_norm_row_and_col_with_output_tk=getattr(
                mx_mod, "quantize_mxfp4_norm_row_and_col_with_output_tk", None
            ),
        )
        logger.info("Loaded FP4 CCE TK v4 pcache backend from %s", root)
        return _FP4_CCE_TK_V4

    joined = "\n  ".join(checked)
    raise FileNotFoundError(
        f"Could not find fp4_cce_TK v4 Python backends. Checked:\n  {joined}\n"
        "Set FP4_CCE_TK_ROOT to the fp4_matmul checkout containing fp4_cce_TK."
    )


def _get_mlce_root():
    for root in _fp4_matmul_roots():
        candidate = os.path.join(root, "ml-cross-entropy")
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError("Could not locate ml-cross-entropy under fp4_matmul roots.")


@dataclass(frozen=True)
class NVRuntime:
    fwd_fn: object
    bwd_v2_fn: object
    bwd_v3_bf16_fn: object
    v3_native_fn: object
    v3_enc_fn: object
    v3_dec_fn: object
    v5_dE_fn: object
    v5_dC_fn: object
    v5_triton_l4_sg8_fn: object
    v5_triton_l4_sg8_e2_fn: object
    v5_triton_l4_sg8_e1_fn: object
    v5_triton_l1_sg8_e1_fn: object
    v5_triton_l1_sg4_e1_fn: object
    v5_publicv3_fn: object
    quant: object
    gemm: object


@dataclass(frozen=True)
class MXRuntime:
    fwd_fn: object
    bwd_v2_bf16_fn: object
    bwd_v3_bf16_fn: object
    v2_rte_fn: object
    v2_enc_fn: object
    v2_dec_fn: object
    v2_rte_rowcol_fn: object
    v2_enc_rowcol_fn: object
    v2_dec_rowcol_fn: object
    v2_rte_rowcol_masked_fn: object
    v2_enc_rowcol_masked_fn: object
    v2_dec_rowcol_masked_fn: object
    v3_rte_fn: object
    v3_enc_fn: object
    v3_dec_fn: object
    v3_rte_masked_fn: object
    v3_enc_masked_fn: object
    v3_dec_masked_fn: object
    v3_enc_colonly_masked_fn: object
    quant: object
    gemm: object


def _get_nv_runtime():
    global _NV_RUNTIME
    if _NV_RUNTIME is not None:
        return _NV_RUNTIME
    _preload_torch_python()
    fwd_v2_path = _try_resolve_existing_path(
        [
            "ThunderKittens/kernels/gemm/nvfp4_b200/_C_nv_cce_v2*.so",
            "fp4_cce_TK/_C_nv_cce_v2*.so",
        ]
    )
    if fwd_v2_path is not None:
        fwd = _load_so(fwd_v2_path)
        fwd_fn = _resolve_attr(fwd, "pp_L3_SG8")
    else:
        raise FileNotFoundError("Could not find the NV fused forward CCE v2 kernel.")

    bwd_v2 = _load_so(
        _resolve_existing_path(
            "NV backward v2",
            [
                "ThunderKittens/kernels/gemm/nvfp4_b200/_C_nv_cce_backward_v2*.so",
                "fp4_cce_TK/_C_nv_cce_backward_v2*.so",
                "ThunderKittens/kernels/gemm/nvfp4_b200/_C_nv_cce_backward.cpython-*.so",
                "fp4_cce_TK/_C_nv_cce_backward.cpython-*.so",
            ],
        )
    )
    bwd_v3 = _load_so(
        _resolve_existing_path(
            "NV backward v3",
            [
                "ThunderKittens/kernels/gemm/nvfp4_b200/_C_nv_cce_backward_v3*.so",
                "fp4_cce_TK/_C_nv_cce_backward_v3*.so",
            ],
        )
    )
    bwd_v5_path = _try_resolve_existing_path(
        [
            "ThunderKittens/kernels/gemm/nvfp4_b200/_C_nv_cce_backward_v5*.so",
            "fp4_cce_TK/_C_nv_cce_backward_v5*.so",
        ]
    )
    bwd_v5 = _load_so(bwd_v5_path) if bwd_v5_path is not None else None
    quant = _load_so(
        _resolve_existing_path(
            "NV quantization",
            [
                "TK_quantisation/nvfp4_v5/_tk_quant_v5*.so",
            ],
        )
    )
    gemm = _load_so(
        _resolve_existing_path(
            "NV GEMM",
            [
                "ThunderKittens/kernels/gemm/nvfp4_b200/_C_nv_gemm*.so",
            ],
        )
    )
    _NV_RUNTIME = NVRuntime(
        fwd_fn=fwd_fn,
        bwd_v2_fn=_resolve_attr(bwd_v2, "backward_v2_bf16_L5_SG8", "backward_v2_bf16_L4_SG8", "backward_L5_SG8", "backward_L4_SG8"),
        bwd_v3_bf16_fn=_resolve_attr(bwd_v3, "backward_v3_bf16_L5_SG8", "backward_v3_bf16_L4_SG8"),
        v3_native_fn=_resolve_attr(bwd_v3, "backward_v3_fp4_L4_SG8"),
        v3_enc_fn=_resolve_attr(bwd_v3, "backward_v3_fp4_enc_L4_SG8"),
        v3_dec_fn=_resolve_attr(bwd_v3, "backward_v3_fp4_dec_L4_SG8"),
        v5_dE_fn=_resolve_optional_attr(bwd_v5, "backward_v5_dE_fp4_L4_SG8") if bwd_v5 is not None else None,
        v5_dC_fn=_resolve_optional_attr(bwd_v5, "backward_v5_dC_fp4_L4_SG8") if bwd_v5 is not None else None,
        v5_triton_l4_sg8_fn=_resolve_optional_attr(bwd_v5, "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8") if bwd_v5 is not None else None,
        v5_triton_l4_sg8_e2_fn=_resolve_optional_attr(bwd_v5, "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E2") if bwd_v5 is not None else None,
        v5_triton_l4_sg8_e1_fn=_resolve_optional_attr(bwd_v5, "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E1") if bwd_v5 is not None else None,
        v5_triton_l1_sg8_e1_fn=_resolve_optional_attr(bwd_v5, "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L1_SG8_E1") if bwd_v5 is not None else None,
        v5_triton_l1_sg4_e1_fn=_resolve_optional_attr(bwd_v5, "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L1_SG4_E1") if bwd_v5 is not None else None,
        v5_publicv3_fn=_resolve_optional_attr(bwd_v5, "experimental_backward_v5_combo_publicv3_fp4_L4_SG8") if bwd_v5 is not None else None,
        quant=quant,
        gemm=gemm,
    )
    return _NV_RUNTIME


def _get_mx_runtime():
    global _MX_RUNTIME
    if _MX_RUNTIME is not None:
        return _MX_RUNTIME
    _preload_torch_python()
    fwd = _load_so(
        _resolve_existing_path(
            "MX fused forward CCE v2",
            [
                "ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx_cce_v2*.so",
                "fp4_cce_TK/_C_mx_cce_v2*.so",
            ],
        )
    )
    fwd_fn = _resolve_attr(fwd, "pp_L4_SG8")

    bwd_v2 = _load_so(
        _resolve_existing_path(
            "MX backward v2",
            [
                "ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx_cce_backward_v2*.so",
                "fp4_cce_TK/_C_mx_cce_backward_v2*.so",
                "ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx_cce_backward.cpython-*.so",
                "fp4_cce_TK/_C_mx_cce_backward.cpython-*.so",
            ],
        )
    )
    bwd_v3 = _load_so(
        _resolve_existing_path(
            "MX backward v3",
            [
                "ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx_cce_backward_v3*.so",
                "fp4_cce_TK/_C_mx_cce_backward_v3*.so",
            ],
        )
    )
    quant = _load_so(
        _resolve_existing_path(
            "MX quantization v3",
            [
                "TK_quantisation/mxfp4_v3/mxfp4_quant_v3*.so",
            ],
        )
    )
    gemm = _load_so(
        _resolve_existing_path(
            "MX GEMM",
            [
                "ThunderKittens/kernels/gemm/mxfp4_gb200/_C_mx.cpython-*.so",
            ],
        )
    )
    _MX_RUNTIME = MXRuntime(
        fwd_fn=fwd_fn,
        bwd_v2_bf16_fn=_resolve_attr(bwd_v2, "backward_v2_bf16_L5_SG8", "backward_v2_bf16_L4_SG8", "backward_L5_SG8", "backward_L4_SG8"),
        bwd_v3_bf16_fn=_resolve_attr(bwd_v3, "backward_v3_bf16_L5_SG8", "backward_v3_bf16_L4_SG8"),
        v2_rte_fn=_resolve_attr(bwd_v2, "backward_v2_fp4_L4_SG8"),
        v2_enc_fn=_resolve_attr(bwd_v2, "backward_v2_fp4_enc_L4_SG8"),
        v2_dec_fn=_resolve_attr(bwd_v2, "backward_v2_fp4_dec_L4_SG8"),
        v2_rte_rowcol_fn=_resolve_attr(bwd_v2, "backward_v2_fp4_rowcol_L4_SG8"),
        v2_enc_rowcol_fn=_resolve_attr(bwd_v2, "backward_v2_fp4_enc_rowcol_L4_SG8"),
        v2_dec_rowcol_fn=_resolve_attr(bwd_v2, "backward_v2_fp4_dec_rowcol_L4_SG8"),
        v2_rte_rowcol_masked_fn=_resolve_optional_attr(bwd_v2, "backward_v2_fp4_rowcol_masked_L4_SG8"),
        v2_enc_rowcol_masked_fn=_resolve_optional_attr(bwd_v2, "backward_v2_fp4_enc_rowcol_masked_L4_SG8"),
        v2_dec_rowcol_masked_fn=_resolve_optional_attr(bwd_v2, "backward_v2_fp4_dec_rowcol_masked_L4_SG8"),
        v3_rte_fn=_resolve_attr(bwd_v3, "backward_v3_fp4_L4_SG8"),
        v3_enc_fn=_resolve_attr(bwd_v3, "backward_v3_fp4_enc_L4_SG8"),
        v3_dec_fn=_resolve_attr(bwd_v3, "backward_v3_fp4_dec_L4_SG8"),
        v3_rte_masked_fn=_resolve_optional_attr(bwd_v3, "backward_v3_fp4_masked_L4_SG8"),
        v3_enc_masked_fn=_resolve_optional_attr(bwd_v3, "backward_v3_fp4_enc_masked_L4_SG8"),
        v3_dec_masked_fn=_resolve_optional_attr(bwd_v3, "backward_v3_fp4_dec_masked_L4_SG8"),
        v3_enc_colonly_masked_fn=_resolve_optional_attr(bwd_v3, "backward_v3_fp4_enc_colonly_masked_L4_SG8"),
        quant=quant,
        gemm=gemm,
    )
    return _MX_RUNTIME


def _require_cuda_bf16(hidden_2d, weight):
    if hidden_2d.device.type != "cuda" or weight.device.type != "cuda":
        raise RuntimeError("fp4_cce requires CUDA tensors for hidden states and output weights.")
    if hidden_2d.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(
            f"fp4_cce currently requires bfloat16 hidden/weight tensors, got "
            f"{hidden_2d.dtype} and {weight.dtype}."
        )


def _fp4_cce_inputs_for_kernel(hidden_2d, weight):
    hidden_2d = _local_tensor_for_cce(hidden_2d)
    weight = _local_tensor_for_cce(weight)
    if hidden_2d.device.type != "cuda" or weight.device.type != "cuda":
        raise RuntimeError("fp4_cce requires CUDA tensors for hidden states and output weights.")
    if hidden_2d.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        return hidden_2d.contiguous(), weight.contiguous()
    return hidden_2d.to(torch.bfloat16).contiguous(), weight.to(torch.bfloat16).contiguous()


def _aligned_size(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _pad_hidden_and_weight(hidden_2d, weight):
    m, k = hidden_2d.shape
    v, weight_k = weight.shape
    if weight_k != k:
        raise ValueError(f"Hidden dim {k} does not match output weight dim {weight_k}.")
    if v % 256 != 0:
        raise ValueError(
            f"fp4_cce currently requires vocab/output rows to be a multiple of 256, got {v}. "
            "This is required to avoid padded-class drift in the public kernels."
        )
    mp = _aligned_size(m, 256)
    kp = _aligned_size(k, 128)
    if mp == m and kp == k:
        hidden_pad = hidden_2d.contiguous()
    else:
        hidden_pad = hidden_2d.new_zeros((mp, kp))
        hidden_pad[:m, :k].copy_(hidden_2d)
    if kp == k:
        weight_pad = weight.contiguous()
    else:
        weight_pad = weight.new_zeros((v, kp))
        weight_pad[:, :k].copy_(weight)
    return hidden_pad.contiguous(), weight_pad.contiguous(), m, k, v, mp, kp


def _pad_targets(labels_1d, mp, ignore_index):
    padded = labels_1d.new_full((mp,), ignore_index)
    padded[: labels_1d.numel()].copy_(labels_1d)
    return padded


def _mx_scale_view(sc):
    return sc.view(-1, sc.shape[1], 32, 16).view(torch.float8_e8m0fnu)


def _mx_zero_fp4x2(shape, device):
    t = torch.empty(*shape, dtype=torch.float4_e2m1fn_x2, device=device)
    t.view(torch.uint8).zero_()
    return t


def _mx_empty_fp4x2(shape, device):
    return torch.empty(*shape, dtype=torch.float4_e2m1fn_x2, device=device)


def _mx_fp4x2_narrow_contiguous(t, start, length):
    return t.view(torch.uint8).narrow(1, start, length).contiguous().view(torch.float4_e2m1fn_x2)


def _valid_label_count(labels_1d, ignore_index):
    return int(labels_1d.ne(ignore_index).sum().item())


def _check_empty_cce_labels() -> bool:
    return os.environ.get("LBT_CCE_CHECK_EMPTY_LABELS", "0") == "1"


def _labels_all_ignored(labels_1d, ignore_index):
    if _assume_nonempty_labels():
        return False
    if not _check_empty_cce_labels():
        return False
    return _valid_label_count(labels_1d, ignore_index) == 0


def _loss_from_logits(logits, labels_1d, ignore_index):
    return F.cross_entropy(logits.float(), labels_1d, ignore_index=ignore_index, reduction="mean")


def _cce_scratch(device):
    return torch.zeros((128, 32), dtype=torch.bfloat16, device=device)


def _masked_nll_loss(lse, neg_logit, labels_1d, ignore_index, neg_scale):
    valid = labels_1d.ne(ignore_index)
    if not bool(valid.any()):
        return lse.sum() * 0.0
    nll = lse + (neg_logit * float(neg_scale))
    return nll[valid].mean()


def _split_label_correction_enabled():
    val = os.environ.get("LBT_FP4_SPLIT_LABEL_CORRECTION", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _valid_label_rows(labels_1d, ignore_index):
    valid = labels_1d.ne(ignore_index)
    if not bool(valid.any()):
        return None, None
    row_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    label_idx = labels_1d.index_select(0, row_idx).to(torch.long)
    return row_idx, label_idx


def _promote_cce_grad_to_probs_inplace(g_bf16, labels_1d, ignore_index, grad_scale):
    row_idx, label_idx = _valid_label_rows(labels_1d, ignore_index)
    if row_idx is None:
        return
    g_bf16[row_idx, label_idx] += float(grad_scale)


def _apply_exact_label_correction_(d_hidden, d_weight, hidden_pad, weight_pad, labels_1d, ignore_index, grad_scale):
    row_idx, label_idx = _valid_label_rows(labels_1d, ignore_index)
    if row_idx is None:
        return
    scaled = float(-grad_scale)
    d_hidden[row_idx] += weight_pad.index_select(0, label_idx).to(d_hidden.dtype) * scaled
    d_weight.index_add_(
        0,
        label_idx,
        hidden_pad.index_select(0, row_idx).to(d_weight.dtype) * scaled,
    )


def _bf16_tail_grads(g_bf16, hidden_pad, weight_pad):
    d_hidden = torch.matmul(g_bf16, weight_pad)
    d_weight = torch.matmul(g_bf16.transpose(0, 1).contiguous(), hidden_pad)
    return d_hidden, d_weight


def _sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _get_triton_linear_cross_entropy():
    global linear_cross_entropy
    if linear_cross_entropy is None:
        mlce_root = _get_mlce_root()
        if mlce_root not in sys.path:
            sys.path.insert(0, mlce_root)
        from cut_cross_entropy import linear_cross_entropy as _lce

        linear_cross_entropy = torch._dynamo.disable(_lce)
    return linear_cross_entropy


def _get_raw_linear_cross_entropy():
    global raw_linear_cross_entropy
    if raw_linear_cross_entropy is None:
        mlce_root = _get_mlce_root()
        if mlce_root not in sys.path:
            sys.path.insert(0, mlce_root)
        from cut_cross_entropy import linear_cross_entropy as _lce

        raw_linear_cross_entropy = _lce
    return raw_linear_cross_entropy


def _bf16_cce_backward_from_saved(hidden_2d, weight, labels_1d, ignore_index, filter_eps, grad_output):
    lce = _get_triton_linear_cross_entropy()
    with torch.enable_grad():
        hidden_ref = hidden_2d.detach().requires_grad_(True)
        weight_ref = weight.detach().requires_grad_(True)
        kwargs = dict(
            shift=False,
            reduction="mean",
            ignore_index=ignore_index,
            filter_eps=_cut_cross_entropy_filter_eps(filter_eps),
        )
        loss = lce(hidden_ref, weight_ref, labels_1d, **kwargs)
        d_hidden, d_weight = torch.autograd.grad(loss, (hidden_ref, weight_ref), grad_outputs=grad_output)
    return d_hidden, d_weight


def _bf16_cce_dhidden_from_saved(
    hidden_2d,
    weight,
    labels_1d,
    ignore_index,
    filter_eps,
    grad_output,
):
    """Recompute only the BF16 body-facing CCE gradient.

    Cut CCE tiles the vocabulary calculation, so this does not retain or
    materialize a full ``[tokens, vocab]`` BF16 gradient tensor.  The weight is
    deliberately detached: the low-precision CCE graph remains responsible
    for ``dWeight`` in the selective-backward ablation.
    """
    lce = _get_triton_linear_cross_entropy()
    with torch.enable_grad():
        hidden_ref = hidden_2d.detach().requires_grad_(True)
        weight_ref = weight.detach()
        kwargs = dict(
            shift=False,
            reduction="mean",
            ignore_index=ignore_index,
            filter_eps=_cut_cross_entropy_filter_eps(filter_eps),
        )
        loss = lce(hidden_ref, weight_ref, labels_1d, **kwargs)
        (d_hidden,) = torch.autograd.grad(
            loss,
            (hidden_ref,),
            grad_outputs=grad_output,
        )
    return d_hidden


class _BF16DHiddenLowPrecisionDWeight(torch.autograd.Function):
    """Keep the low-precision scalar/dWeight but replace its dHidden.

    ``lowp_loss`` must have been constructed from ``hidden.detach()``.  Passing
    the incoming scalar gradient back to that input preserves the original
    low-precision dWeight graph, while the explicit ``hidden`` input receives
    a straight-through BF16 Cut-CCE gradient computed during backward.
    """

    @staticmethod
    def forward(
        ctx,
        lowp_loss,
        hidden,
        weight,
        labels,
        ignore_index,
        filter_eps,
    ):
        ctx.save_for_backward(hidden, weight, labels)
        ctx.ignore_index = int(ignore_index)
        ctx.filter_eps = float(filter_eps)
        return lowp_loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, labels = ctx.saved_tensors
        d_hidden = _bf16_cce_dhidden_from_saved(
            hidden,
            weight,
            labels,
            ctx.ignore_index,
            ctx.filter_eps,
            grad_output,
        )
        return (
            grad_output,
            d_hidden.contiguous(),
            None,
            None,
            None,
            None,
        )


def _bf16_dhidden_only_enabled() -> bool:
    return _flag_enabled("LBT_FP4_CCE_BF16_DHIDDEN_ONLY")


def _lowp_logits_bf16_dhidden_enabled() -> bool:
    return _flag_enabled("FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN")


def _validate_bf16_dhidden_only_backend(backend, *, prequantized_x: bool) -> None:
    """Fail closed outside the one selective-backward path we measured.

    Detaching ``hidden`` only saves work in the v4 prequantized-X autograd
    function.  The ordinary CCE entry point and the other FP4 cache formats
    still compute (or assume) a low-precision dHidden, so accepting them would
    make this flag look faster without proving that the discarded work was
    actually removed.
    """
    if not prequantized_x:
        raise RuntimeError(
            "LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1 requires the fused final-norm "
            "prequantized-X path"
        )
    if backend.name != "nvfp4" or getattr(backend, "implementation", None) != "v4":
        raise RuntimeError(
            "LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1 is currently validated only "
            "for the nvfp4 v4 CCE backend"
        )

    required_flags = (
        "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER",
        "FP4_CCE_V4_NVFP4_MXFP8_FORWARD",
        "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE",
        "FP4_CCE_V4_MIXED_DW_MXFP8_COLS",
        "FP4_CCE_V4_NVFP4_G_TARGET_SPLIT",
        "FP4_CCE_V4_NVFP4_FUSED_G_CACHE",
        "FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW",
        "FP4_CCE_V4_EXACT_TARGET_TOPK_LOGITS",
        "FP4_CCE_V4_MX_COMPACT_DW_REPAIR",
    )
    missing = [name for name in required_flags if not _flag_enabled(name)]
    if missing:
        raise RuntimeError(
            "LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1 requires the measured mixed "
            "MXFP8-forward/MXFP8-dWeight candidate; missing: "
            + ", ".join(missing)
        )
    conflicting = (
        "FP4_CCE_V4_MXFP4_G_CACHE",
        "FP4_CCE_V4_MXFP8_G_CACHE",
    )
    enabled_conflicts = [name for name in conflicting if _flag_enabled(name)]
    if enabled_conflicts:
        raise RuntimeError(
            "LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1 rejects conflicting G-cache "
            "formats: " + ", ".join(enabled_conflicts)
        )

    if os.environ.get("FP4_CCE_V4_NVFP4_QUANT_BACKEND") != "localcta_v4":
        raise RuntimeError(
            "LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1 requires "
            "FP4_CCE_V4_NVFP4_QUANT_BACKEND=localcta_v4"
        )
    if os.environ.get("FP4_CCE_V4_NVFP4_FUSED_G_CACHE_IMPL") != "tiled":
        raise RuntimeError(
            "LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1 requires the tiled fused G-cache"
        )
    try:
        topk = int(os.environ.get("FP4_CCE_V4_NVFP4_G_TOPK_SPLIT", "0"))
        exact_topk = int(os.environ.get("FP4_CCE_V4_EXACT_SELECTED_TOPK", "0"))
    except ValueError as exc:
        raise RuntimeError(
            "selective BF16 dHidden requires integer top-k settings"
        ) from exc
    if (topk, exact_topk) != (16, 16):
        raise RuntimeError(
            "LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1 is validated only for "
            "G_TOPK_SPLIT=16 and EXACT_SELECTED_TOPK=16"
        )


def _attach_bf16_dhidden_to_lowp_loss(
    lowp_loss,
    hidden_2d,
    weight,
    labels_1d,
    backend,
):
    """Attach the selective BF16-dHidden/lowp-dWeight autograd bridge."""
    hidden_local = _local_tensor_for_cce(hidden_2d)
    weight_local = _fp4_cce_bf16_weight(weight.detach())
    labels_local = _local_tensor_for_cce(labels_1d).contiguous()
    return _BF16DHiddenLowPrecisionDWeight.apply(
        lowp_loss,
        hidden_local,
        weight_local,
        labels_local,
        int(backend.ignore_index),
        float(getattr(backend, "filter_eps", 0.0)),
    )


def _validate_lowp_logits_bf16_dhidden_backend(
    backend,
    *,
    prequantized_x: bool,
) -> None:
    """Fail closed unless the internal FP4 path receives its measured profile."""
    flag = "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1"
    if _bf16_dhidden_only_enabled():
        raise RuntimeError(
            f"{flag} conflicts with LBT_FP4_CCE_BF16_DHIDDEN_ONLY=1"
        )
    if not prequantized_x:
        raise RuntimeError(
            f"{flag} requires the fused final-norm prequantized-X path"
        )
    if backend.name != "nvfp4" or getattr(backend, "implementation", None) != "v4":
        raise RuntimeError(f"{flag} is validated only for nvfp4 v4 CCE")
    required_flags = (
        "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER",
        "FP4_CCE_V4_NVFP4_MXFP8_FORWARD",
        "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE",
        "FP4_CCE_V4_MIXED_DW_MXFP8_COLS",
        "FP4_CCE_V4_NVFP4_G_TARGET_SPLIT",
        "FP4_CCE_V4_NVFP4_FUSED_G_CACHE",
        "FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW",
        "FP4_CCE_V4_EXACT_TARGET_TOPK_LOGITS",
        "FP4_CCE_V4_MX_COMPACT_DW_REPAIR",
    )
    missing = [name for name in required_flags if not _flag_enabled(name)]
    if missing:
        raise RuntimeError(
            f"{flag} requires the measured mixed MXFP8 candidate; missing: "
            + ", ".join(missing)
        )
    conflicts = (
        "FP4_CCE_V4_MXFP4_G_CACHE",
        "FP4_CCE_V4_MXFP8_G_CACHE",
        "FP4_CCE_V4_MXFP8_FP8_LOGITS",
        "FP4_CCE_V4_MXFP8_CENTERED_FP8_LOGITS",
        "FP4_CCE_V4_SPARSE_REPAIR_OVERLAP",
        "FP4_CCE_V4_MX_BACKWARD_GEMM_OVERLAP",
    )
    enabled_conflicts = [name for name in conflicts if _flag_enabled(name)]
    if enabled_conflicts:
        raise RuntimeError(
            f"{flag} rejects conflicting flags: "
            + ", ".join(enabled_conflicts)
        )
    if os.environ.get("FP4_CCE_V4_NVFP4_QUANT_BACKEND") != "localcta_v4":
        raise RuntimeError(f"{flag} requires localcta_v4 quantization")
    if os.environ.get("FP4_CCE_V4_NVFP4_FUSED_G_CACHE_IMPL") != "tiled":
        raise RuntimeError(f"{flag} requires the tiled fused G-cache")
    try:
        topk = int(os.environ.get("FP4_CCE_V4_NVFP4_G_TOPK_SPLIT", "0"))
        exact_topk = int(os.environ.get("FP4_CCE_V4_EXACT_SELECTED_TOPK", "0"))
    except ValueError as exc:
        raise RuntimeError(f"{flag} requires integer top-k settings") from exc
    if (topk, exact_topk) != (16, 16):
        raise RuntimeError(f"{flag} requires exact target/top-16 repair")


def _queue_common_eval_metric(
    hidden_2d,
    weight,
    labels_1d,
    ignore_index,
    backend_loss,
):
    global _COMMON_EVAL_COUNTER
    if not _common_eval_enabled():
        return
    _COMMON_EVAL_COUNTER += 1
    if _COMMON_EVAL_COUNTER % _common_eval_every() != 0:
        return
    maximum_relative_gap = _common_eval_max_relative_gap()
    metrics_processor = None
    try:
        from low_bits_training.metrics import get_metrics_processor
        metrics_processor = get_metrics_processor()
    except Exception:
        pass
    if metrics_processor is None and maximum_relative_gap is None:
        return
    _sync_cuda(hidden_2d.device)
    if _common_eval_filter_eps() != 0.0:
        logger.warning(
            "Native BF16 common CCE evaluation ignores filter_eps=%s.",
            _common_eval_filter_eps(),
        )
    common_eval_backend = _common_eval_backend()
    if common_eval_backend == "triton_bf16":
        backend = _TritonBF16Backend(
            ignore_index=ignore_index,
            filter_eps=0.0,
        )
    else:
        backend = _NativeMXFP4PrecisionBackend(
            ignore_index=ignore_index,
            implementation="v4",
            quant_mode="enc",
            forward_precision="bf16",
            backward_precision="bf16",
        )
    hidden_eval = _local_tensor_for_cce(hidden_2d.detach()).to(
        dtype=torch.bfloat16
    ).contiguous()
    weight_eval = _fp4_cce_bf16_weight(weight.detach())
    labels_eval = _local_tensor_for_cce(labels_1d.detach()).contiguous()
    with torch.no_grad():
        loss = backend.training_loss(hidden_eval, weight_eval, labels_eval)
    backend_loss_value = float(backend_loss.detach().item())
    loss_value = float(loss.detach().item())
    loss_gap = loss_value - backend_loss_value
    relative_loss_gap = abs(loss_gap) / max(abs(loss_value), 1e-12)
    loss_ratio = (
        loss_value / backend_loss_value
        if backend_loss_value != 0.0
        else float("nan")
    )
    if metrics_processor is not None:
        metrics_processor.delayed_log(
            {
                "eval_backend/loss": backend_loss_value,
                "eval_bf16/loss": loss_value,
                "eval_gap/bf16_minus_backend": loss_gap,
                "eval_gap/abs_bf16_minus_backend": abs(loss_gap),
                "eval_gap/relative_abs_bf16": relative_loss_gap,
                "eval_gap/bf16_over_backend": loss_ratio,
            }
        )
    logger.info(
        "paired CCE eval: backend=%.6f bf16=%.6f bf16-backend=%+.6f",
        backend_loss_value,
        loss_value,
        loss_gap,
    )
    if (
        maximum_relative_gap is not None
        and relative_loss_gap > maximum_relative_gap
    ):
        raise RuntimeError(
            "paired CCE relative loss gap "
            f"{relative_loss_gap:.6f} exceeds {maximum_relative_gap:.6f}: "
            f"backend={backend_loss_value:.6f}, bf16={loss_value:.6f}"
        )


def _nv_encode_inputs_for_mode(quant_mode):
    return quant_mode != "dec"


def _nv_tk_quantize_for_gemm(runtime, tensor, return_transpose: bool, encode_centric: bool):
    result = runtime.quant.tk_quantize_for_gemm(tensor, return_transpose, encode_centric)
    return result[:6], tuple(result[6:])


def _mx_mode_id(quant_mode):
    mapping = {"rte": 0, "enc": 1, "dec": 2}
    return mapping[quant_mode]


def _mx_filter_block_cols():
    raw = os.environ.get("LBT_MX_FILTER_BLOCK_COLS", "").strip()
    if not raw:
        return 32
    val = int(raw)
    if val not in (32, 64, 128, 256):
        raise ValueError(f"LBT_MX_FILTER_BLOCK_COLS must be one of 32, 64, 128, 256; got {val}")
    return val


def _mx_consumer_chunk_cols(runtime):
    raw = os.environ.get("LBT_MX_CONSUMER_CHUNK_COLS", "").strip()
    if raw:
        val = int(raw)
        if val not in (128, 256):
            raise ValueError(f"LBT_MX_CONSUMER_CHUNK_COLS must be 128 or 256; got {val}")
    else:
        val = 256
    if val == 128 and not hasattr(runtime.gemm, "mxfp4_gemm_k128"):
        return 256
    return val


def _mx_masked_consumer_chunk_cols(runtime):
    raw = os.environ.get("LBT_MX_CONSUMER_CHUNK_COLS", "").strip()
    if raw:
        return _mx_consumer_chunk_cols(runtime)
    return _mx_consumer_chunk_cols(runtime)


def _mx_dense_gemm(runtime, A, A_sc, B, B_sc, D, consumer_chunk_cols=None):
    chunk_cols = _mx_consumer_chunk_cols(runtime) if consumer_chunk_cols is None else consumer_chunk_cols
    if chunk_cols == 128 and hasattr(runtime.gemm, "mxfp4_gemm_k128"):
        runtime.gemm.mxfp4_gemm_k128(A, A_sc, B, B_sc, D)
        return
    runtime.gemm.mxfp4_gemm(A, A_sc, B, B_sc, D)


def _mx_masked_gemm(runtime, A, A_sc, B, B_sc, tilemask, tilemask_transposed, D, consumer_chunk_cols=None):
    chunk_cols = _mx_consumer_chunk_cols(runtime) if consumer_chunk_cols is None else consumer_chunk_cols
    if chunk_cols == 128 and hasattr(runtime.gemm, "mxfp4_gemm_masked_k128"):
        runtime.gemm.mxfp4_gemm_masked_k128(A, A_sc, B, B_sc, tilemask, tilemask_transposed, D)
        return
    runtime.gemm.mxfp4_gemm_masked(A, A_sc, B, B_sc, tilemask, tilemask_transposed, D)


def _mx_v2_rowcol_launch_fn(runtime, quant_mode, masked):
    if quant_mode == "rte":
        return runtime.v2_rte_rowcol_masked_fn if masked else runtime.v2_rte_rowcol_fn
    if quant_mode == "enc":
        return runtime.v2_enc_rowcol_masked_fn if masked else runtime.v2_enc_rowcol_fn
    return runtime.v2_dec_rowcol_masked_fn if masked else runtime.v2_dec_rowcol_fn


def _mx_v3_launch_fn(runtime, quant_mode, masked):
    if quant_mode == "rte":
        return runtime.v3_rte_masked_fn if masked else runtime.v3_rte_fn
    if quant_mode == "enc":
        return runtime.v3_enc_masked_fn if masked else runtime.v3_enc_fn
    return runtime.v3_dec_masked_fn if masked else runtime.v3_dec_fn


def _mx_active_reduction_chunks(g_tilemask, g_sc_row=None, consumer_chunk_cols=256):
    if g_tilemask is None:
        if g_sc_row is None:
            return [], []
        active_tiles = g_sc_row.ne(0).view(g_sc_row.shape[0], g_sc_row.shape[1], -1).any(dim=2)
    else:
        active_tiles = g_tilemask.ne(0)
    if g_sc_row is not None and not bool(active_tiles.any().item()):
        active_tiles = g_sc_row.ne(0).view(g_sc_row.shape[0], g_sc_row.shape[1], -1).any(dim=2)
    if consumer_chunk_cols % 128 != 0:
        raise ValueError(f"consumer_chunk_cols ({consumer_chunk_cols}) must be divisible by 128")
    tile_group = consumer_chunk_cols // 128
    d_hidden_chunks = active_tiles.view(
        active_tiles.shape[0], active_tiles.shape[1] // tile_group, tile_group
    ).any(dim=2).any(dim=0)
    d_weight_chunks = active_tiles.view(
        active_tiles.shape[0] // tile_group, tile_group, active_tiles.shape[1]
    ).any(dim=1).any(dim=1)
    d_hidden_idx = d_hidden_chunks.nonzero(as_tuple=False).flatten().tolist()
    d_weight_idx = d_weight_chunks.nonzero(as_tuple=False).flatten().tolist()
    return d_hidden_idx, d_weight_idx


def _mx_tilemask_from_scales(g_sc_row):
    return g_sc_row.ne(0).view(g_sc_row.shape[0], g_sc_row.shape[1], -1).any(dim=2).to(torch.uint8)


def _mx_active_reduction_chunks_from_fine_mask(active_fine, block_cols, consumer_chunk_cols=256):
    if active_fine is None:
        return [], []
    if block_cols <= 0 or consumer_chunk_cols % block_cols != 0:
        raise ValueError(
            f"consumer_chunk_cols ({consumer_chunk_cols}) must be divisible by block_cols ({block_cols})"
        )
    red_group = consumer_chunk_cols // block_cols
    d_hidden_chunks = active_fine.view(
        active_fine.shape[0], active_fine.shape[1] // red_group, red_group
    ).any(dim=2).any(dim=0)
    row_group = consumer_chunk_cols // 128
    d_weight_chunks = active_fine.view(
        active_fine.shape[0] // row_group, row_group, active_fine.shape[1]
    ).any(dim=1).any(dim=1)
    d_hidden_idx = d_hidden_chunks.nonzero(as_tuple=False).flatten().tolist()
    d_weight_idx = d_weight_chunks.nonzero(as_tuple=False).flatten().tolist()
    return d_hidden_idx, d_weight_idx


def _mx_block_filter_bf16_inplace(g_bf16, filter_eps, block_cols=128):
    if filter_eps <= 0.0:
        return None
    row_tiles = g_bf16.shape[0] // 128
    col_tiles = g_bf16.shape[1] // block_cols
    tiles = g_bf16.view(row_tiles, 128, col_tiles, block_cols).transpose(1, 2)
    active = tiles.abs().float().amax(dim=(2, 3)) >= float(filter_eps)
    inactive = ~active
    if bool(inactive.any().item()):
        tiles.masked_fill_(inactive[:, :, None, None], 0)
    return active.to(torch.uint8)


def _mx_tilemask_to_128_tiles(active_fine, block_cols):
    if active_fine is None:
        return None
    if block_cols <= 0 or block_cols % 32 != 0:
        raise ValueError(f"block_cols must be a positive multiple of 32; got {block_cols}")
    if block_cols == 128:
        return active_fine.to(torch.uint8)
    if block_cols < 128:
        group = 128 // block_cols
        return active_fine.view(
            active_fine.shape[0], active_fine.shape[1] // group, group
        ).any(dim=2).to(torch.uint8)
    repeat = block_cols // 128
    return active_fine.repeat_interleave(repeat, dim=1).to(torch.uint8)


def _mx_compacted_gemm(runtime, A_full, A_sc_full, B_full, B_sc_full, active_chunks, D_out, consumer_chunk_cols=None):
    consumer_chunk_cols = _mx_consumer_chunk_cols(runtime) if consumer_chunk_cols is None else consumer_chunk_cols
    if not active_chunks:
        D_out.zero_()
        return

    total_chunks = 2 * A_full.shape[1] // consumer_chunk_cols
    if len(active_chunks) == total_chunks:
        _mx_dense_gemm(
            runtime,
            A_full,
            _mx_scale_view(A_sc_full),
            B_full,
            _mx_scale_view(B_sc_full),
            D_out,
            consumer_chunk_cols=consumer_chunk_cols,
        )
        return

    D_out.zero_()
    D_acc = torch.zeros(D_out.shape, dtype=torch.float32, device=D_out.device)
    chunk_fp4x2_width = consumer_chunk_cols // 2
    chunk_scale_width = consumer_chunk_cols // 128
    segments = []
    seg_start = active_chunks[0]
    seg_len = 1
    for chunk_idx in active_chunks[1:]:
        if chunk_idx == seg_start + seg_len:
            seg_len += 1
        else:
            segments.append((seg_start, seg_len))
            seg_start = chunk_idx
            seg_len = 1
    segments.append((seg_start, seg_len))
    for seg_start, seg_len in segments:
        fp4_off = seg_start * chunk_fp4x2_width
        sc_off = seg_start * chunk_scale_width
        fp4_width = seg_len * chunk_fp4x2_width
        sc_width = seg_len * chunk_scale_width
        A_sub = _mx_fp4x2_narrow_contiguous(A_full, fp4_off, fp4_width)
        A_sc_sub = _mx_scale_view(A_sc_full.narrow(1, sc_off, sc_width).contiguous())
        B_sub = _mx_fp4x2_narrow_contiguous(B_full, fp4_off, fp4_width)
        B_sc_sub = _mx_scale_view(B_sc_full.narrow(1, sc_off, sc_width).contiguous())
        D_partial = torch.empty_like(D_out)
        _mx_dense_gemm(
            runtime,
            A_sub,
            A_sc_sub,
            B_sub,
            B_sc_sub,
            D_partial,
            consumer_chunk_cols=consumer_chunk_cols,
        )
        D_acc.add_(D_partial.float())
    D_out.copy_(D_acc.to(dtype=D_out.dtype))


def _mx_active_iter_ratio(tilemask, consumer_chunk_cols=256):
    if tilemask.numel() == 0:
        return 0.0
    if consumer_chunk_cols == 128:
        return float(tilemask.ne(0).float().mean().item())
    if consumer_chunk_cols == 256:
        row_pairs = tilemask.shape[0] // 2
        red_pairs = tilemask.shape[1] // 2
        if row_pairs == 0 or red_pairs == 0:
            return 0.0
        block_active = (
            tilemask[0::2, 0::2].ne(0)
            | tilemask[0::2, 1::2].ne(0)
            | tilemask[1::2, 0::2].ne(0)
            | tilemask[1::2, 1::2].ne(0)
        )
        return float(block_active.float().mean().item())
    raise ValueError(f"Unsupported consumer_chunk_cols: {consumer_chunk_cols}")


def _mx_should_use_sparse_gemm(runtime, tilemask, consumer_chunk_cols=None, threshold=None):
    consumer_chunk_cols = _mx_consumer_chunk_cols(runtime) if consumer_chunk_cols is None else consumer_chunk_cols
    force_env = os.environ.get("LBT_FORCE_MX_SPARSE_GEMM", "").strip().lower()
    if threshold is None:
        threshold = 0.90 if consumer_chunk_cols == 128 else 0.95
    if force_env in {"0", "false", "no", "off"}:
        return False
    return _mx_active_iter_ratio(tilemask, consumer_chunk_cols=consumer_chunk_cols) < threshold


def _mx_tilemask_has_any(tilemask):
    return bool(tilemask.ne(0).any().item())


def _mx_should_enable_masked_filter(implementation, quant_mode, m, k, v, filter_eps):
    if filter_eps <= 0.0:
        return False

    force_env = os.environ.get("LBT_FORCE_MX_FILTER_EPS", "").strip().lower()
    if force_env in {"1", "true", "yes", "on"}:
        return True
    if force_env in {"0", "false", "no", "off"}:
        return False

    if quant_mode != "enc":
        return False
    if implementation == "v2":
        if v < 131072:
            return False
        if m > 8192:
            return False
        return True
    if implementation == "v3":
        if v < 128000:
            return False
        if k < 7168:
            return False
        if m > 16384:
            return False
        return True
    return False


def _mx_should_save_filter_cols(implementation, quant_mode, m, k, v, filter_eps):
    if filter_eps <= 0.0:
        return False

    force_env = os.environ.get("LBT_FORCE_MX_SAVE_FILTER_COLS", "").strip().lower()
    if force_env in {"1", "true", "yes", "on"}:
        return True
    if force_env in {"0", "false", "no", "off"}:
        return False

    return _mx_should_enable_masked_filter(implementation, quant_mode, m, k, v, filter_eps)


def _nv_v5_triton_combo_fn(runtime, mode: str):
    mapping = {
        "triton_l4_sg8": runtime.v5_triton_l4_sg8_fn,
        "triton_l4_sg8_e2": runtime.v5_triton_l4_sg8_e2_fn,
        "triton_l4_sg8_e1": runtime.v5_triton_l4_sg8_e1_fn,
        "triton_l1_sg8_e1": runtime.v5_triton_l1_sg8_e1_fn,
        "triton_l1_sg4_e1": runtime.v5_triton_l1_sg4_e1_fn,
    }
    fn = mapping.get(mode)
    if fn is None:
        raise RuntimeError(f"NVFP4 v5 combo entrypoint {mode!r} is unavailable in this build.")
    return fn


class _NVFP4CCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_2d, weight, labels_1d, ignore_index, implementation, quant_mode, filter_eps):
        runtime = _get_nv_runtime()
        _require_cuda_bf16(hidden_2d, weight)
        # Some custom TK/localCTA producers launch on the default stream. Order the
        # CCE consumer stream after those launches without a host-side device drain.
        torch.cuda.current_stream(hidden_2d.device).wait_stream(torch.cuda.default_stream(hidden_2d.device))
        if _nv_cce_pre_quant_sync_enabled():
            # Regular TK still has non-captured producer launches that can race the
            # CCE input quantizers. Keep the barrier scoped away from localCTA/v4.
            _sync_cuda(hidden_2d.device)
        hidden_pad, weight_pad, m, k, v, mp, kp = _pad_hidden_and_weight(hidden_2d, weight)
        encode_inputs = _nv_encode_inputs_for_mode(quant_mode)
        resolved_filter_eps = _resolve_filter_eps(filter_eps)
        split_label_correction = _split_label_correction_enabled()
        nuclear_bwd = _nv_nuclear_bwd_enabled()
        true_nuclear_bwd = _nv_true_nuclear_bwd_enabled()
        v5_bwd_mode = _nv_v5_bwd_mode() if implementation == "v5" else "split"
        if _nv_fwd_hidden_row_sr_enabled() and encode_inputs and hasattr(runtime.quant, "tk_quantize_row_for_gemm_sr"):
            e_row, e_sc_r, e_sg_r, _ = runtime.quant.tk_quantize_row_for_gemm_sr(hidden_pad, encode_inputs)
            e_col, e_sc_c, e_sg_c, _ = runtime.quant.tk_quantize_col_only(hidden_pad, e_sg_r)
            e_keepalive = ()
        else:
            (e_row, e_sc_r, e_col, e_sc_c, e_sg_r, e_sg_c), e_keepalive = _nv_tk_quantize_for_gemm(
                runtime, hidden_pad, True, encode_inputs
            )
        (w_row, w_sc_r, w_col, w_sc_c, w_sg_r, w_sg_c), w_keepalive = _nv_tk_quantize_for_gemm(
            runtime, weight_pad, True, encode_inputs
        )
        # The NV ping-pong forward kernel is stable across repeated launches only
        # when the preceding quantization work has fully completed.
        _nv_cce_sync_cuda(hidden_pad.device)
        lse = torch.full((m,), -float("inf"), dtype=torch.float32, device=hidden_pad.device)
        neg = torch.zeros((m,), dtype=torch.float32, device=hidden_pad.device)
        runtime.fwd_fn(
            e_row,
            e_sc_r,
            e_sg_r,
            w_row,
            w_sc_r,
            w_sg_r,
            lse,
            neg,
            labels_1d,
            _cce_scratch(hidden_pad.device),
            m,
            v,
        )
        _nv_cce_sync_cuda(hidden_pad.device)
        # The NV forward kernels apply the combined global scale to the accumulator
        # before extracting both `lse` and `neg_correct_logit`. Re-scaling the
        # target term here double-scales it and corrupts the loss.
        loss = _masked_nll_loss(lse, neg, labels_1d, ignore_index, 1.0)

        if true_nuclear_bwd:
            ctx.save_for_backward()
        else:
            ctx.save_for_backward(
                e_row,
                e_sc_r,
                e_col,
                e_sc_c,
                e_sg_r,
                e_sg_c,
                w_row,
                w_sc_r,
                w_col,
                w_sc_c,
                w_sg_r,
                w_sg_c,
                lse,
                labels_1d,
            )
        ctx.m = m
        ctx.k = k
        ctx.v = v
        ctx.mp = mp
        ctx.kp = kp
        ctx.implementation = implementation
        ctx.quant_mode = quant_mode
        ctx.ignore_index = ignore_index
        ctx.filter_eps = resolved_filter_eps
        ctx._nv_hidden_pad = hidden_pad.detach() if split_label_correction else None
        ctx._nv_weight_pad = weight_pad.detach() if split_label_correction else None
        ctx._nv_bwd_hidden_pad = hidden_pad.detach() if nuclear_bwd else None
        ctx._nv_bwd_weight_pad = weight_pad.detach() if nuclear_bwd else None
        ctx._nv_v5_bwd_mode = v5_bwd_mode
        ctx._nv_v5_hidden_pad = hidden_pad.detach() if _nv_v5_bwd_uses_bf16_saved(v5_bwd_mode) else None
        ctx._nv_v5_weight_pad = weight_pad.detach() if _nv_v5_bwd_uses_bf16_saved(v5_bwd_mode) else None
        ctx._nv_true_bwd_hidden = hidden_2d.detach() if true_nuclear_bwd else None
        ctx._nv_true_bwd_weight = weight.detach() if true_nuclear_bwd else None
        ctx._nv_true_bwd_labels = labels_1d.detach() if true_nuclear_bwd else None
        ctx._nv_tk_quant_keepalive = e_keepalive + w_keepalive
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        use_true_nuclear_bwd = (
            ctx._nv_true_bwd_hidden is not None and ctx._nv_true_bwd_weight is not None
        )
        if use_true_nuclear_bwd:
            d_hidden, d_weight = _bf16_cce_backward_from_saved(
                ctx._nv_true_bwd_hidden,
                ctx._nv_true_bwd_weight,
                ctx._nv_true_bwd_labels,
                ctx.ignore_index,
                ctx.filter_eps,
                grad_output,
            )
            _nv_cce_sync_cuda(d_hidden.device)
            return (
                d_hidden.contiguous(),
                d_weight.contiguous(),
                None,
                None,
                None,
                None,
                None,
            )

        (
            e_row,
            e_sc_r,
            e_col,
            e_sc_c,
            e_sg_r,
            e_sg_c,
            w_row,
            w_sc_r,
            w_col,
            w_sc_c,
            w_sg_r,
            w_sg_c,
            lse,
            labels_1d,
        ) = ctx.saved_tensors
        runtime = _get_nv_runtime()
        valid = labels_1d.ne(ctx.ignore_index)
        denom = max(int(valid.sum().item()), 1)
        grad_scale = float((grad_output / denom).item())
        kernel_filter_eps = _nv_kernel_filter_eps(ctx.filter_eps, grad_scale)
        encode_inputs = _nv_encode_inputs_for_mode(ctx.quant_mode)
        use_row_sr = _nv_row_sr_enabled() and encode_inputs
        split_label_correction = (
            ctx._nv_hidden_pad is not None and ctx._nv_weight_pad is not None
        )
        use_nuclear_bwd = (
            ctx._nv_bwd_hidden_pad is not None and ctx._nv_bwd_weight_pad is not None
        )

        if use_nuclear_bwd:
            grad_bf16 = e_row.new_zeros((ctx.mp, ctx.v), dtype=torch.bfloat16)
            runtime.bwd_v2_fn(
                e_row,
                e_sc_r,
                e_sg_r,
                w_row,
                w_sc_r,
                w_sg_r,
                grad_bf16,
                lse,
                labels_1d,
                grad_scale,
                ctx.m,
                ctx.v,
                kernel_filter_eps,
            )
            _nv_cce_sync_cuda(e_row.device)
            if split_label_correction:
                _promote_cce_grad_to_probs_inplace(grad_bf16, labels_1d, ctx.ignore_index, grad_scale)
            d_hidden, d_weight = _bf16_tail_grads(
                grad_bf16,
                ctx._nv_bwd_hidden_pad,
                ctx._nv_bwd_weight_pad,
            )
            if split_label_correction:
                _apply_exact_label_correction_(
                    d_hidden,
                    d_weight,
                    ctx._nv_hidden_pad,
                    ctx._nv_weight_pad,
                    labels_1d,
                    ctx.ignore_index,
                    grad_scale,
                )
            _nv_cce_sync_cuda(d_hidden.device)
            return (
                d_hidden[: ctx.m, : ctx.k].contiguous(),
                d_weight[: ctx.v, : ctx.k].contiguous(),
                None,
                None,
                None,
                None,
                None,
            )

        d_hidden = e_row.new_zeros((ctx.mp, ctx.kp), dtype=torch.bfloat16)
        d_weight = w_row.new_zeros((ctx.v, ctx.kp), dtype=torch.bfloat16)

        if ctx.implementation == "v2":
            grad_bf16 = e_row.new_zeros((ctx.mp, ctx.v), dtype=torch.bfloat16)
            runtime.bwd_v2_fn(
                e_row, e_sc_r, e_sg_r,
                w_row, w_sc_r, w_sg_r,
                grad_bf16, lse, labels_1d, grad_scale, ctx.m, ctx.v, kernel_filter_eps,
            )
            _nv_cce_sync_cuda(e_row.device)
            if split_label_correction:
                _promote_cce_grad_to_probs_inplace(grad_bf16, labels_1d, ctx.ignore_index, grad_scale)
            if use_row_sr:
                g_row, g_sc_r, g_sg_r, _ = runtime.quant.tk_quantize_row_for_gemm_sr(
                    grad_bf16, encode_inputs,
                )
                g_col, g_sc_c, g_sg_c, _ = runtime.quant.tk_quantize_col_only(grad_bf16, g_sg_r)
            else:
                (g_row, g_sc_r, g_col, g_sc_c, g_sg_r, g_sg_c), _g_keepalive = _nv_tk_quantize_for_gemm(
                    runtime, grad_bf16, True, encode_inputs
                )
            _nv_cce_sync_cuda(e_row.device)
            runtime.gemm.nvfp4_gemm(g_row, g_sc_r, g_sg_r, w_col, w_sc_c, w_sg_c, d_hidden)
            _nv_cce_sync_cuda(e_row.device)
            runtime.gemm.nvfp4_gemm(g_col, g_sc_c, g_sg_c, e_col, e_sc_c, e_sg_c, d_weight)
            _nv_cce_sync_cuda(e_row.device)
            if split_label_correction:
                _apply_exact_label_correction_(
                    d_hidden,
                    d_weight,
                    ctx._nv_hidden_pad,
                    ctx._nv_weight_pad,
                    labels_1d,
                    ctx.ignore_index,
                    grad_scale,
                )
        elif ctx.implementation == "v3":
            use_bf16_v3_fallback = ctx.filter_eps > 0.0 or not _nv_native_v3_dense_enabled()
            if use_bf16_v3_fallback:
                grad_bf16 = e_row.new_zeros((ctx.mp, ctx.v), dtype=torch.bfloat16)
                runtime.bwd_v2_fn(
                    e_row, e_sc_r, e_sg_r,
                    w_row, w_sc_r, w_sg_r,
                    grad_bf16, lse, labels_1d, grad_scale, ctx.m, ctx.v, kernel_filter_eps,
                )
                _nv_cce_sync_cuda(e_row.device)
                if split_label_correction:
                    _promote_cce_grad_to_probs_inplace(grad_bf16, labels_1d, ctx.ignore_index, grad_scale)
                if use_row_sr:
                    g_row, g_sc_r, g_sg_r, _ = runtime.quant.tk_quantize_row_for_gemm_sr(
                        grad_bf16, encode_inputs,
                    )
                    g_col, g_sc_c, g_sg_c, _ = runtime.quant.tk_quantize_col_only(grad_bf16, g_sg_r)
                else:
                    (g_row, g_sc_r, g_col, g_sc_c, g_sg_r, g_sg_c), _g_keepalive = _nv_tk_quantize_for_gemm(
                        runtime, grad_bf16, True, encode_inputs
                    )
                _nv_cce_sync_cuda(e_row.device)
                runtime.gemm.nvfp4_gemm(g_row, g_sc_r, g_sg_r, w_col, w_sc_c, w_sg_c, d_hidden)
                _nv_cce_sync_cuda(e_row.device)
                runtime.gemm.nvfp4_gemm(g_col, g_sc_c, g_sg_c, e_col, e_sc_c, e_sg_c, d_weight)
                _nv_cce_sync_cuda(e_row.device)
                if split_label_correction:
                    _apply_exact_label_correction_(
                        d_hidden,
                        d_weight,
                        ctx._nv_hidden_pad,
                        ctx._nv_weight_pad,
                        labels_1d,
                        ctx.ignore_index,
                        grad_scale,
                    )
                return (
                    d_hidden[: ctx.m, : ctx.k].contiguous(),
                    d_weight[: ctx.v, : ctx.k].contiguous(),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            g_fp4_row = torch.empty(ctx.mp, ctx.v // 2, dtype=torch.float4_e2m1fn_x2, device=e_row.device)
            g_sc_row = torch.zeros(ctx.mp // 128, ctx.v // 64, 512, dtype=torch.uint8, device=e_row.device)
            g_sg_row = torch.empty(1, dtype=torch.float32, device=e_row.device)
            g_fp4_col = torch.zeros(ctx.v, ctx.mp // 2, dtype=torch.uint8, device=e_row.device)
            g_sc_col = torch.zeros(ctx.v // 128, ctx.mp // 64, 512, dtype=torch.uint8, device=e_row.device)

            if ctx.quant_mode == "native":
                runtime.v3_native_fn(
                    e_row,
                    e_sc_r,
                    e_sg_r,
                    w_row,
                    w_sc_r,
                    w_sg_r,
                    g_fp4_row,
                    g_sc_row.view(torch.float16),
                    g_sg_row,
                    g_fp4_col,
                    g_sc_col.view(torch.float16),
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    0.0,
                    False,
                )
            elif ctx.quant_mode == "enc":
                runtime.v3_enc_fn(
                    e_row,
                    e_sc_r,
                    e_sg_r,
                    w_row,
                    w_sc_r,
                    w_sg_r,
                    g_fp4_row,
                    g_sc_row.view(torch.float16),
                    g_sg_row,
                    g_fp4_col,
                    g_sc_col.view(torch.float16),
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    kernel_filter_eps,
                )
            else:
                runtime.v3_dec_fn(
                    e_row,
                    e_sc_r,
                    e_sg_r,
                    w_row,
                    w_sc_r,
                    w_sg_r,
                    g_fp4_row,
                    g_sc_row.view(torch.float16),
                    g_sg_row,
                    g_fp4_col,
                    g_sc_col.view(torch.float16),
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    kernel_filter_eps,
                )
            _nv_cce_sync_cuda(e_row.device)

            runtime.gemm.nvfp4_gemm(
                g_fp4_row,
                g_sc_row.view(torch.float8_e4m3fn),
                g_sg_row,
                w_col,
                w_sc_c,
                w_sg_c,
                d_hidden,
            )
            _nv_cce_sync_cuda(e_row.device)
            runtime.gemm.nvfp4_gemm(
                g_fp4_col.view(torch.float4_e2m1fn_x2),
                g_sc_col.view(torch.float8_e4m3fn),
                g_sg_row,
                e_col,
                e_sc_c,
                e_sg_c,
                d_weight,
            )
            _nv_cce_sync_cuda(e_row.device)
        elif ctx.implementation == "v5":
            v5_bwd_mode = getattr(ctx, "_nv_v5_bwd_mode", "split")
            if v5_bwd_mode.startswith("triton_"):
                if ctx._nv_v5_hidden_pad is None or ctx._nv_v5_weight_pad is None:
                    raise RuntimeError(
                        f"NVFP4 v5 {v5_bwd_mode} requires saved BF16 hidden/weight pads."
                    )
                _nv_v5_triton_combo_fn(runtime, v5_bwd_mode)(
                    e_row,
                    e_sc_r,
                    e_sg_r,
                    w_row,
                    w_sc_r,
                    w_sg_r,
                    ctx._nv_v5_hidden_pad,
                    ctx._nv_v5_weight_pad,
                    d_hidden,
                    d_weight,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.mp,
                    ctx.v,
                    ctx.kp,
                    kernel_filter_eps,
                )
                _nv_cce_sync_cuda(e_row.device)
            elif v5_bwd_mode == "publicv3":
                if runtime.v5_publicv3_fn is None:
                    raise RuntimeError(
                        "NVFP4 v5 public-v3 combo entrypoint is unavailable in this build."
                    )
                if kernel_filter_eps != 0.0:
                    raise RuntimeError("NVFP4 v5 public-v3 combo does not support filter_eps.")
                g_fp4_row = torch.empty(
                    ctx.mp, ctx.v // 2, dtype=torch.float4_e2m1fn_x2, device=e_row.device
                )
                g_sc_row = torch.zeros(
                    ctx.mp // 128, ctx.v // 64, 512, dtype=torch.uint8, device=e_row.device
                )
                g_sg_row = torch.empty(1, dtype=torch.float32, device=e_row.device)
                g_fp4_col = torch.zeros(ctx.v, ctx.mp // 2, dtype=torch.uint8, device=e_row.device)
                g_sc_col = torch.zeros(
                    ctx.v // 128, ctx.mp // 64, 512, dtype=torch.uint8, device=e_row.device
                )
                runtime.v5_publicv3_fn(
                    e_row,
                    e_sc_r,
                    e_sg_r,
                    w_row,
                    w_sc_r,
                    w_sg_r,
                    w_col,
                    w_sc_c,
                    w_sg_c,
                    e_col,
                    e_sc_c,
                    e_sg_c,
                    d_hidden,
                    d_weight,
                    g_fp4_row,
                    g_sc_row.view(torch.float16),
                    g_sg_row,
                    g_fp4_col,
                    g_sc_col.view(torch.float16),
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.mp,
                    ctx.v,
                    ctx.kp,
                    0.0,
                    encode_inputs,
                )
                _nv_cce_sync_cuda(e_row.device)
            else:
                if runtime.v5_dE_fn is None or runtime.v5_dC_fn is None:
                    raise RuntimeError("NVFP4 v5 split entrypoints are unavailable in this build.")
                runtime.v5_dE_fn(
                    e_row,
                    e_sc_r,
                    e_sg_r,
                    w_row,
                    w_sc_r,
                    w_sg_r,
                    w_col,
                    w_sc_c,
                    w_sg_c,
                    d_hidden,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.mp,
                    ctx.v,
                    ctx.kp,
                    kernel_filter_eps,
                )
                _nv_cce_sync_cuda(e_row.device)
                runtime.v5_dC_fn(
                    e_row,
                    e_sc_r,
                    e_sg_r,
                    w_row,
                    w_sc_r,
                    w_sg_r,
                    e_col,
                    e_sc_c,
                    e_sg_c,
                    d_weight,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.mp,
                    ctx.v,
                    ctx.kp,
                    kernel_filter_eps,
                )
                _nv_cce_sync_cuda(e_row.device)
        else:
            raise ValueError(f"Unsupported NVFP4 implementation {ctx.implementation!r}")

        return (
            d_hidden[: ctx.m, : ctx.k].contiguous(),
            d_weight[: ctx.v, : ctx.k].contiguous(),
            None,
            None,
            None,
            None,
            None,
        )


class _MXFP4CCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_2d, weight, labels_1d, ignore_index, implementation, quant_mode, filter_eps):
        runtime = _get_mx_runtime()
        _require_cuda_bf16(hidden_2d, weight)
        hidden_pad, weight_pad, m, k, v, mp, kp = _pad_hidden_and_weight(hidden_2d, weight)
        mode_id = _mx_mode_id(quant_mode)
        resolved_filter_eps = _resolve_filter_eps(filter_eps)
        split_label_correction = _split_label_correction_enabled()
        nuclear_bwd = _mx_nuclear_bwd_enabled()
        true_nuclear_bwd = _mx_true_nuclear_bwd_enabled()
        ctx._mx_exact_label_hidden_pad = hidden_pad.detach() if split_label_correction else None
        ctx._mx_exact_label_weight_pad = weight_pad.detach() if split_label_correction else None
        ctx._mx_nuclear_hidden_pad = hidden_pad.detach() if nuclear_bwd else None
        ctx._mx_nuclear_weight_pad = weight_pad.detach() if nuclear_bwd else None
        ctx._mx_true_bwd_hidden = hidden_2d.detach() if true_nuclear_bwd else None
        ctx._mx_true_bwd_weight = weight.detach() if true_nuclear_bwd else None

        if resolved_filter_eps > 0.0:
            if _mx_should_save_filter_cols(implementation, quant_mode, m, k, v, resolved_filter_eps):
                e_fp4, e_sc, e_col_fp4, e_col_sc = runtime.quant.mxfp4_quantize_row_and_col(hidden_pad, mode_id)
                w_fp4, w_sc, w_col_fp4, w_col_sc = runtime.quant.mxfp4_quantize_row_and_col(weight_pad, mode_id)
                ctx._mx_saved_cols = True
                ctx._mx_hidden_pad = None
                ctx._mx_weight_pad = None
            else:
                e_fp4, e_sc = runtime.quant.mxfp4_quantize_for_gemm(hidden_pad, mode_id)
                w_fp4, w_sc = runtime.quant.mxfp4_quantize_for_gemm(weight_pad, mode_id)
                ctx._mx_saved_cols = False
                ctx._mx_hidden_pad = hidden_pad.detach()
                ctx._mx_weight_pad = weight_pad.detach()
        else:
            e_fp4, e_sc, e_col_fp4, e_col_sc = runtime.quant.mxfp4_quantize_row_and_col(hidden_pad, mode_id)
            w_fp4, w_sc, w_col_fp4, w_col_sc = runtime.quant.mxfp4_quantize_row_and_col(weight_pad, mode_id)
            ctx._mx_saved_cols = True
            ctx._mx_hidden_pad = None
            ctx._mx_weight_pad = None
        lse = torch.full((m,), -float("inf"), dtype=torch.float32, device=hidden_pad.device)
        neg = torch.zeros((m,), dtype=torch.float32, device=hidden_pad.device)
        # The MX forward path is sensitive to back-to-back launches from the
        # quantizer into the fused CE kernel. Keeping the launch ordering
        # explicit here avoids latent async failures under repeated training.
        _sync_cuda(hidden_pad.device)
        runtime.fwd_fn(
            e_fp4,
            e_sc,
            w_fp4,
            w_sc,
            lse,
            neg,
            labels_1d,
            _cce_scratch(hidden_pad.device),
            m,
            v,
        )
        _sync_cuda(hidden_pad.device)
        # The MX forward kernels apply MXFP4_ALPHA before extracting both
        # `lse` and `neg_correct_logit`, so the target term is already scaled.
        loss = _masked_nll_loss(lse, neg, labels_1d, ignore_index, 1.0)
        if ctx._mx_saved_cols:
            ctx.save_for_backward(
                e_fp4,
                e_sc,
                e_col_fp4,
                e_col_sc,
                w_fp4,
                w_sc,
                w_col_fp4,
                w_col_sc,
                lse,
                labels_1d,
            )
        else:
            ctx.save_for_backward(
                e_fp4,
                e_sc,
                w_fp4,
                w_sc,
                lse,
                labels_1d,
            )
        ctx.m = m
        ctx.k = k
        ctx.v = v
        ctx.mp = mp
        ctx.kp = kp
        ctx.implementation = implementation
        ctx.quant_mode = quant_mode
        ctx.ignore_index = ignore_index
        ctx.filter_eps = resolved_filter_eps
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx._mx_saved_cols:
            (
                e_fp4,
                e_sc,
                e_col_fp4,
                e_col_sc,
                w_fp4,
                w_sc,
                w_col_fp4,
                w_col_sc,
                lse,
                labels_1d,
            ) = ctx.saved_tensors
        else:
            (
                e_fp4,
                e_sc,
                w_fp4,
                w_sc,
                lse,
                labels_1d,
            ) = ctx.saved_tensors
            e_col_fp4 = e_col_sc = w_col_fp4 = w_col_sc = None
        runtime = _get_mx_runtime()
        valid = labels_1d.ne(ctx.ignore_index)
        denom = max(int(valid.sum().item()), 1)
        grad_scale = float((grad_output / denom).item())
        split_label_correction = (
            ctx._mx_exact_label_hidden_pad is not None and ctx._mx_exact_label_weight_pad is not None
        )
        use_true_nuclear_bwd = (
            ctx._mx_true_bwd_hidden is not None and ctx._mx_true_bwd_weight is not None
        )
        use_nuclear_bwd = (
            ctx._mx_nuclear_hidden_pad is not None and ctx._mx_nuclear_weight_pad is not None
        )
        if use_true_nuclear_bwd:
            d_hidden, d_weight = _bf16_cce_backward_from_saved(
                ctx._mx_true_bwd_hidden,
                ctx._mx_true_bwd_weight,
                labels_1d,
                ctx.ignore_index,
                ctx.filter_eps,
                grad_output,
            )
            return (
                d_hidden.contiguous(),
                d_weight.contiguous(),
                None,
                None,
                None,
                None,
                None,
            )
        if use_nuclear_bwd:
            g_bf16 = e_fp4.new_zeros((ctx.mp, ctx.v), dtype=torch.bfloat16)
            if ctx.implementation == "v2":
                runtime.bwd_v2_bf16_fn(
                    e_fp4,
                    e_sc,
                    w_fp4,
                    w_sc,
                    g_bf16,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    0.0,
                )
            else:
                runtime.bwd_v3_bf16_fn(
                    e_fp4,
                    e_sc,
                    w_fp4,
                    w_sc,
                    g_bf16,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    0.0,
                )
            if split_label_correction:
                _promote_cce_grad_to_probs_inplace(g_bf16, labels_1d, ctx.ignore_index, grad_scale)
            if ctx.filter_eps > 0.0:
                _mx_block_filter_bf16_inplace(
                    g_bf16,
                    _nv_kernel_filter_eps(ctx.filter_eps, grad_scale),
                    block_cols=_mx_filter_block_cols(),
                )
            d_hidden, d_weight = _bf16_tail_grads(
                g_bf16,
                ctx._mx_nuclear_hidden_pad,
                ctx._mx_nuclear_weight_pad,
            )
            if split_label_correction:
                _apply_exact_label_correction_(
                    d_hidden,
                    d_weight,
                    ctx._mx_exact_label_hidden_pad,
                    ctx._mx_exact_label_weight_pad,
                    labels_1d,
                    ctx.ignore_index,
                    grad_scale,
                )
            _sync_cuda(d_hidden.device)
            return (
                d_hidden[: ctx.m, : ctx.k].contiguous(),
                d_weight[: ctx.v, : ctx.k].contiguous(),
                None,
                None,
                None,
                None,
                None,
            )
        use_masked_filter = _mx_should_enable_masked_filter(
            ctx.implementation,
            ctx.quant_mode,
            ctx.m,
            ctx.k,
            ctx.v,
            ctx.filter_eps,
        )

        d_hidden = None
        d_weight = None

        def _alloc_grads():
            nonlocal d_hidden, d_weight
            hidden_ref = e_fp4 if e_fp4 is not None else (e_col_fp4 if e_col_fp4 is not None else ctx._mx_hidden_pad)
            weight_ref = w_fp4 if w_fp4 is not None else (w_col_fp4 if w_col_fp4 is not None else ctx._mx_weight_pad)
            if d_hidden is None:
                d_hidden = hidden_ref.new_zeros((ctx.mp, ctx.kp), dtype=torch.bfloat16)
            if d_weight is None:
                d_weight = weight_ref.new_zeros((ctx.v, ctx.kp), dtype=torch.bfloat16)

        def _ensure_col_quants():
            nonlocal e_col_fp4, e_col_sc, w_col_fp4, w_col_sc
            if e_col_fp4 is not None and e_col_sc is not None and w_col_fp4 is not None and w_col_sc is not None:
                return
            mxfp4_quantize_col_only = _load_mxfp4_backend_py().mxfp4_quantize_col_only
            mode_id = _mx_mode_id(ctx.quant_mode)
            if e_col_fp4 is None or e_col_sc is None:
                e_col_fp4, e_col_sc = mxfp4_quantize_col_only(ctx._mx_hidden_pad, mode_id)
            if w_col_fp4 is None or w_col_sc is None:
                w_col_fp4, w_col_sc = mxfp4_quantize_col_only(ctx._mx_weight_pad, mode_id)
            ctx._mx_hidden_pad = None
            ctx._mx_weight_pad = None

        force_bf16_filter_fallback = os.environ.get("LBT_FORCE_MX_BF16_FILTER_FALLBACK", "").strip().lower()
        force_native_masked_producer = os.environ.get("LBT_FORCE_MX_NATIVE_MASKED_PRODUCER", "").strip().lower()
        if force_bf16_filter_fallback in {"1", "true", "yes", "on"}:
            use_bf16_filter_fallback = True
        elif force_native_masked_producer in {"0", "false", "no", "off"}:
            use_bf16_filter_fallback = True
        else:
            use_bf16_filter_fallback = False

        if ctx.filter_eps > 0.0 and use_bf16_filter_fallback:
            g_bf16 = e_fp4.new_zeros((ctx.mp, ctx.v), dtype=torch.bfloat16)
            if ctx.implementation == "v2":
                runtime.bwd_v2_bf16_fn(
                    e_fp4,
                    e_sc,
                    w_fp4,
                    w_sc,
                    g_bf16,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    0.0,
                )
            else:
                runtime.bwd_v3_bf16_fn(
                    e_fp4,
                    e_sc,
                    w_fp4,
                    w_sc,
                    g_bf16,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    0.0,
                )

            if split_label_correction:
                _promote_cce_grad_to_probs_inplace(g_bf16, labels_1d, ctx.ignore_index, grad_scale)
            block_cols = _mx_filter_block_cols()
            consumer_chunk_cols = max(_mx_masked_consumer_chunk_cols(runtime), block_cols)
            g_tilemask_fine = _mx_block_filter_bf16_inplace(
                g_bf16,
                _nv_kernel_filter_eps(ctx.filter_eps, grad_scale),
                block_cols=block_cols,
            )
            g_tilemask = _mx_tilemask_to_128_tiles(g_tilemask_fine, block_cols=block_cols)
            mode_id = _mx_mode_id(ctx.quant_mode)
            g_fp4_row, g_sc_row, g_fp4_col, g_sc_col = runtime.quant.mxfp4_quantize_row_and_col(
                g_bf16, mode_id
            )
            d_hidden_idx, d_weight_idx = _mx_active_reduction_chunks_from_fine_mask(
                g_tilemask_fine,
                block_cols=block_cols,
                consumer_chunk_cols=consumer_chunk_cols,
            )
            labels_for_correction = labels_1d
            e_fp4 = e_sc = w_fp4 = w_sc = lse = None
            _ensure_col_quants()
            _alloc_grads()
            if _mx_tilemask_has_any(g_tilemask) and _mx_should_use_sparse_gemm(
                runtime, g_tilemask, consumer_chunk_cols=consumer_chunk_cols
            ):
                _mx_masked_gemm(
                    runtime,
                    g_fp4_row,
                    _mx_scale_view(g_sc_row),
                    w_col_fp4,
                    _mx_scale_view(w_col_sc),
                    g_tilemask,
                    False,
                    d_hidden,
                    consumer_chunk_cols=consumer_chunk_cols,
                )
                _mx_masked_gemm(
                    runtime,
                    g_fp4_col.view(torch.float4_e2m1fn_x2),
                    _mx_scale_view(g_sc_col),
                    e_col_fp4.view(torch.float4_e2m1fn_x2),
                    _mx_scale_view(e_col_sc),
                    g_tilemask,
                    True,
                    d_weight,
                    consumer_chunk_cols=consumer_chunk_cols,
                )
            else:
                _mx_compacted_gemm(
                    runtime,
                    g_fp4_row,
                    g_sc_row,
                    w_col_fp4,
                    w_col_sc,
                    d_hidden_idx,
                    d_hidden,
                    consumer_chunk_cols=consumer_chunk_cols,
                )
                _mx_compacted_gemm(
                    runtime,
                    g_fp4_col.view(torch.float4_e2m1fn_x2),
                    g_sc_col,
                    e_col_fp4.view(torch.float4_e2m1fn_x2),
                    e_col_sc,
                    d_weight_idx,
                    d_weight,
                    consumer_chunk_cols=consumer_chunk_cols,
                )
            if split_label_correction:
                _apply_exact_label_correction_(
                    d_hidden,
                    d_weight,
                    ctx._mx_exact_label_hidden_pad,
                    ctx._mx_exact_label_weight_pad,
                    labels_for_correction,
                    ctx.ignore_index,
                    grad_scale,
                )
            return (
                d_hidden[: ctx.m, : ctx.k].contiguous(),
                d_weight[: ctx.v, : ctx.k].contiguous(),
                None,
                None,
                None,
                None,
                None,
            )

        if ctx.implementation == "v2":
            if use_masked_filter:
                g_fp4_row = _mx_empty_fp4x2((ctx.mp, ctx.v // 2), e_fp4.device)
                g_sc_row = torch.zeros(ctx.mp // 128, ctx.v // 128, 512, dtype=torch.uint8, device=e_fp4.device)
                g_fp4_col = torch.empty(ctx.v, ctx.mp // 2, dtype=torch.uint8, device=e_fp4.device)
                g_sc_col = torch.zeros(ctx.v // 128, ctx.mp // 128, 512, dtype=torch.uint8, device=e_fp4.device)
                g_tilemask = torch.zeros(ctx.mp // 128, ctx.v // 128, dtype=torch.uint8, device=e_fp4.device)

                launch_fn = _mx_v2_rowcol_launch_fn(runtime, ctx.quant_mode, masked=True)
                force_v3_masked_for_v2_env = os.environ.get("LBT_FORCE_V3_MASKED_PRODUCER_FOR_V2", "").strip().lower()
                force_native_v2_masked_for_v2_env = os.environ.get("LBT_FORCE_V2_NATIVE_MASKED_PRODUCER_FOR_V2", "").strip().lower()
                force_v2_v3_hybrid_for_v2_env = os.environ.get("LBT_FORCE_V2_V3_HYBRID_MASKED_PRODUCER_FOR_V2", "").strip().lower()
                if force_v3_masked_for_v2_env in {"1", "true", "yes", "on"}:
                    force_v3_masked_for_v2 = True
                elif force_v3_masked_for_v2_env in {"0", "false", "no", "off"}:
                    force_v3_masked_for_v2 = False
                else:
                    force_v3_masked_for_v2 = (
                        ctx.quant_mode == "enc"
                        and ctx.filter_eps > 0.0
                        and runtime.v3_enc_masked_fn is not None
                    )
                use_v3_full_masked_for_v2 = (
                    force_v3_masked_for_v2
                    and ctx.quant_mode == "enc"
                    and ctx.filter_eps > 0.0
                    and runtime.v3_enc_masked_fn is not None
                )
                if force_native_v2_masked_for_v2_env in {"1", "true", "yes", "on"}:
                    use_v3_full_masked_for_v2 = False
                use_v2_v3_hybrid_masked = (
                    force_v2_v3_hybrid_for_v2_env in {"1", "true", "yes", "on"}
                    and not use_v3_full_masked_for_v2
                    and force_native_v2_masked_for_v2_env not in {"1", "true", "yes", "on"}
                    and ctx.quant_mode == "enc"
                    and ctx.filter_eps > 0.0
                    and runtime.v3_enc_colonly_masked_fn is not None
                )
                if launch_fn is None and not use_v3_full_masked_for_v2:
                    raise RuntimeError("MXFP4 v2 masked row+col launcher is unavailable in this build.")
                if use_v3_full_masked_for_v2:
                    runtime.v3_enc_masked_fn(
                        e_fp4,
                        e_sc,
                        w_fp4,
                        w_sc,
                        g_fp4_row,
                        g_sc_row,
                        g_fp4_col,
                        g_sc_col,
                        g_tilemask,
                        lse,
                        labels_1d,
                        grad_scale,
                        ctx.m,
                        ctx.v,
                        ctx.filter_eps,
                    )
                elif use_v2_v3_hybrid_masked:
                    empty_u8 = torch.empty((0,), dtype=torch.uint8, device=e_fp4.device)
                    launch_fn(
                        e_fp4,
                        e_sc,
                        w_fp4,
                        w_sc,
                        g_fp4_row,
                        g_sc_row,
                        empty_u8,
                        empty_u8,
                        g_tilemask,
                        lse,
                        labels_1d,
                        grad_scale,
                        ctx.m,
                        ctx.v,
                        ctx.filter_eps,
                    )
                    g_tilemask_col = torch.zeros_like(g_tilemask)
                    runtime.v3_enc_colonly_masked_fn(
                        e_fp4,
                        e_sc,
                        w_fp4,
                        w_sc,
                        empty_u8,
                        empty_u8,
                        g_fp4_col,
                        g_sc_col,
                        g_tilemask_col,
                        lse,
                        labels_1d,
                        grad_scale,
                        ctx.m,
                        ctx.v,
                        ctx.filter_eps,
                    )
                    g_tilemask = (g_tilemask | g_tilemask_col).to(torch.uint8)
                else:
                    launch_fn(
                        e_fp4,
                        e_sc,
                        w_fp4,
                        w_sc,
                        g_fp4_row,
                        g_sc_row,
                        g_fp4_col,
                        g_sc_col,
                        g_tilemask,
                        lse,
                        labels_1d,
                        grad_scale,
                        ctx.m,
                        ctx.v,
                        ctx.filter_eps,
                    )
                if not _mx_tilemask_has_any(g_tilemask):
                    fallback_tilemask = _mx_tilemask_from_scales(g_sc_row)
                    if _mx_tilemask_has_any(fallback_tilemask):
                        g_tilemask = fallback_tilemask
                if not _mx_tilemask_has_any(g_tilemask):
                    _alloc_grads()
                elif _mx_should_use_sparse_gemm(
                    runtime, g_tilemask, consumer_chunk_cols=_mx_masked_consumer_chunk_cols(runtime)
                ):
                    e_fp4 = e_sc = w_fp4 = w_sc = lse = labels_1d = None
                    _ensure_col_quants()
                    _alloc_grads()
                    consumer_chunk_cols = _mx_masked_consumer_chunk_cols(runtime)
                    _mx_masked_gemm(
                        runtime,
                        g_fp4_row,
                        _mx_scale_view(g_sc_row),
                        w_col_fp4,
                        _mx_scale_view(w_col_sc),
                        g_tilemask,
                        False,
                        d_hidden,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
                    _mx_masked_gemm(
                        runtime,
                        g_fp4_col.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(g_sc_col),
                        e_col_fp4.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(e_col_sc),
                        g_tilemask,
                        True,
                        d_weight,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
                else:
                    e_fp4 = e_sc = w_fp4 = w_sc = lse = labels_1d = None
                    _ensure_col_quants()
                    _alloc_grads()
                    consumer_chunk_cols = _mx_masked_consumer_chunk_cols(runtime)
                    d_hidden_idx, d_weight_idx = _mx_active_reduction_chunks(
                        g_tilemask, g_sc_row, consumer_chunk_cols=consumer_chunk_cols
                    )
                    _mx_compacted_gemm(
                        runtime,
                        g_fp4_row,
                        g_sc_row,
                        w_col_fp4,
                        w_col_sc,
                        d_hidden_idx,
                        d_hidden,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
                    _mx_compacted_gemm(
                        runtime,
                        g_fp4_col.view(torch.float4_e2m1fn_x2),
                        g_sc_col,
                        e_col_fp4.view(torch.float4_e2m1fn_x2),
                        e_col_sc,
                        d_weight_idx,
                        d_weight,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
            else:
                g_bf16 = e_fp4.new_zeros((ctx.mp, ctx.v), dtype=torch.bfloat16)
                runtime.bwd_v2_bf16_fn(
                    e_fp4,
                    e_sc,
                    w_fp4,
                    w_sc,
                    g_bf16,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    0.0,
                )
                if split_label_correction:
                    _promote_cce_grad_to_probs_inplace(g_bf16, labels_1d, ctx.ignore_index, grad_scale)
                mode_id = _mx_mode_id(ctx.quant_mode)
                g_fp4_row, g_sc_row, g_fp4_col, g_sc_col = runtime.quant.mxfp4_quantize_row_and_col(
                    g_bf16, mode_id
                )
                labels_for_correction = labels_1d
                e_fp4 = e_sc = w_fp4 = w_sc = lse = g_bf16 = None
                _ensure_col_quants()
                _alloc_grads()
                runtime.gemm.mxfp4_gemm(
                    g_fp4_row,
                    _mx_scale_view(g_sc_row),
                    w_col_fp4,
                    _mx_scale_view(w_col_sc),
                    d_hidden,
                )
                runtime.gemm.mxfp4_gemm(
                    g_fp4_col.view(torch.float4_e2m1fn_x2),
                    _mx_scale_view(g_sc_col),
                    e_col_fp4.view(torch.float4_e2m1fn_x2),
                    _mx_scale_view(e_col_sc),
                    d_weight,
                )
                if split_label_correction:
                    _apply_exact_label_correction_(
                        d_hidden,
                        d_weight,
                        ctx._mx_exact_label_hidden_pad,
                        ctx._mx_exact_label_weight_pad,
                        labels_for_correction,
                        ctx.ignore_index,
                        grad_scale,
                    )
        else:
            g_fp4_row = _mx_empty_fp4x2((ctx.mp, ctx.v // 2), e_fp4.device)
            g_sc_row = torch.zeros(ctx.mp // 128, ctx.v // 128, 512, dtype=torch.uint8, device=e_fp4.device)
            g_fp4_col = torch.empty(ctx.v, ctx.mp // 2, dtype=torch.uint8, device=e_fp4.device)
            g_sc_col = torch.zeros(ctx.v // 128, ctx.mp // 128, 512, dtype=torch.uint8, device=e_fp4.device)

            launch_fn = _mx_v3_launch_fn(runtime, ctx.quant_mode, masked=use_masked_filter)
            if launch_fn is None and (use_masked_filter or _mx_native_v3_dense_enabled()):
                raise RuntimeError("MXFP4 v3 masked launcher is unavailable in this build.")

            if use_masked_filter:
                g_tilemask = torch.zeros(ctx.mp // 128, ctx.v // 128, dtype=torch.uint8, device=e_fp4.device)
                launch_fn(
                    e_fp4,
                    e_sc,
                    w_fp4,
                    w_sc,
                    g_fp4_row,
                    g_sc_row,
                    g_fp4_col,
                    g_sc_col,
                    g_tilemask,
                    lse,
                    labels_1d,
                    grad_scale,
                    ctx.m,
                    ctx.v,
                    ctx.filter_eps,
                )
                if not _mx_tilemask_has_any(g_tilemask):
                    fallback_tilemask = _mx_tilemask_from_scales(g_sc_row)
                    if _mx_tilemask_has_any(fallback_tilemask):
                        g_tilemask = fallback_tilemask
                if not _mx_tilemask_has_any(g_tilemask):
                    _alloc_grads()
                elif _mx_should_use_sparse_gemm(
                    runtime, g_tilemask, consumer_chunk_cols=_mx_masked_consumer_chunk_cols(runtime)
                ):
                    e_fp4 = e_sc = w_fp4 = w_sc = lse = labels_1d = None
                    _ensure_col_quants()
                    _alloc_grads()
                    consumer_chunk_cols = _mx_masked_consumer_chunk_cols(runtime)
                    _mx_masked_gemm(
                        runtime,
                        g_fp4_row,
                        _mx_scale_view(g_sc_row),
                        w_col_fp4,
                        _mx_scale_view(w_col_sc),
                        g_tilemask,
                        False,
                        d_hidden,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
                    _mx_masked_gemm(
                        runtime,
                        g_fp4_col.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(g_sc_col),
                        e_col_fp4.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(e_col_sc),
                        g_tilemask,
                        True,
                        d_weight,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
                else:
                    e_fp4 = e_sc = w_fp4 = w_sc = lse = labels_1d = None
                    _ensure_col_quants()
                    _alloc_grads()
                    consumer_chunk_cols = _mx_masked_consumer_chunk_cols(runtime)
                    d_hidden_idx, d_weight_idx = _mx_active_reduction_chunks(
                        g_tilemask, g_sc_row, consumer_chunk_cols=consumer_chunk_cols
                    )
                    _mx_compacted_gemm(
                        runtime,
                        g_fp4_row,
                        g_sc_row,
                        w_col_fp4,
                        w_col_sc,
                        d_hidden_idx,
                        d_hidden,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
                    _mx_compacted_gemm(
                        runtime,
                        g_fp4_col.view(torch.float4_e2m1fn_x2),
                        g_sc_col,
                        e_col_fp4.view(torch.float4_e2m1fn_x2),
                        e_col_sc,
                        d_weight_idx,
                        d_weight,
                        consumer_chunk_cols=consumer_chunk_cols,
                    )
            else:
                if _mx_native_v3_dense_enabled():
                    launch_fn(
                        e_fp4,
                        e_sc,
                        w_fp4,
                        w_sc,
                        g_fp4_row,
                        g_sc_row,
                        g_fp4_col,
                        g_sc_col,
                        lse,
                        labels_1d,
                        grad_scale,
                        ctx.m,
                        ctx.v,
                        0.0,
                    )
                    e_fp4 = e_sc = w_fp4 = w_sc = lse = labels_1d = None
                    _ensure_col_quants()
                    _alloc_grads()
                    runtime.gemm.mxfp4_gemm(
                        g_fp4_row,
                        _mx_scale_view(g_sc_row),
                        w_col_fp4,
                        _mx_scale_view(w_col_sc),
                        d_hidden,
                    )
                    runtime.gemm.mxfp4_gemm(
                        g_fp4_col.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(g_sc_col),
                        e_col_fp4.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(e_col_sc),
                        d_weight,
                    )
                else:
                    g_bf16 = e_fp4.new_zeros((ctx.mp, ctx.v), dtype=torch.bfloat16)
                    runtime.bwd_v2_bf16_fn(
                        e_fp4,
                        e_sc,
                        w_fp4,
                        w_sc,
                        g_bf16,
                        lse,
                        labels_1d,
                        grad_scale,
                        ctx.m,
                        ctx.v,
                        0.0,
                    )
                    if split_label_correction:
                        _promote_cce_grad_to_probs_inplace(g_bf16, labels_1d, ctx.ignore_index, grad_scale)
                    mode_id = _mx_mode_id(ctx.quant_mode)
                    g_fp4_row, g_sc_row, g_fp4_col, g_sc_col = runtime.quant.mxfp4_quantize_row_and_col(
                        g_bf16, mode_id
                    )
                    labels_for_correction = labels_1d
                    e_fp4 = e_sc = w_fp4 = w_sc = lse = g_bf16 = None
                    _ensure_col_quants()
                    _alloc_grads()
                    runtime.gemm.mxfp4_gemm(
                        g_fp4_row,
                        _mx_scale_view(g_sc_row),
                        w_col_fp4,
                        _mx_scale_view(w_col_sc),
                        d_hidden,
                    )
                    runtime.gemm.mxfp4_gemm(
                        g_fp4_col.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(g_sc_col),
                        e_col_fp4.view(torch.float4_e2m1fn_x2),
                        _mx_scale_view(e_col_sc),
                        d_weight,
                    )
                    if split_label_correction:
                        _apply_exact_label_correction_(
                            d_hidden,
                            d_weight,
                            ctx._mx_exact_label_hidden_pad,
                            ctx._mx_exact_label_weight_pad,
                            labels_for_correction,
                            ctx.ignore_index,
                            grad_scale,
                        )

        _sync_cuda(d_hidden.device)
        return (
            d_hidden[: ctx.m, : ctx.k].contiguous(),
            d_weight[: ctx.v, : ctx.k].contiguous(),
            None,
            None,
            None,
            None,
            None,
        )


class _BaseCCEBackend:
    name = "base"
    requires_biasless_training = True

    def __init__(self, ignore_index):
        self.ignore_index = ignore_index

    def training_loss(self, hidden_2d, weight, labels_1d):
        raise NotImplementedError

    def quantize_final_norm_x(self, pre_norm_2d, norm_weight, epsilon):
        raise NotImplementedError

    def training_loss_prequantized_x(self, hidden_2d, x_q, x_col_q, weight, labels_1d):
        raise NotImplementedError

    def training_loss_vocab_parallel(
        self,
        hidden_2d,
        weight,
        labels_1d,
        *,
        tp_group,
        vocab_start: int,
        global_vocab_size: int,
        reduce_dE: bool = False,
    ):
        raise NotImplementedError

    def training_loss_vocab_parallel_prequantized_x(
        self,
        hidden_2d,
        x_q,
        x_col_q,
        weight,
        labels_1d,
        *,
        tp_group,
        vocab_start: int,
        global_vocab_size: int,
        reduce_dE: bool = False,
    ):
        raise NotImplementedError


class _TritonBF16Backend(_BaseCCEBackend):
    name = "triton_bf16"

    def __init__(self, ignore_index, filter_eps=0.0):
        super().__init__(ignore_index)
        self.filter_eps = _resolve_filter_eps(filter_eps)
        if linear_cross_entropy is None:
            mlce_root = _get_mlce_root()
            if mlce_root not in sys.path:
                sys.path.insert(0, mlce_root)
            from cut_cross_entropy import linear_cross_entropy as _lce

            self._linear_cross_entropy = torch._dynamo.disable(_lce)
        else:
            self._linear_cross_entropy = linear_cross_entropy

    def training_loss(self, hidden_2d, weight, labels_1d):
        if _labels_all_ignored(labels_1d, self.ignore_index):
            return hidden_2d.sum() * 0.0
        kwargs = dict(
            shift=False,
            reduction="mean",
            ignore_index=self.ignore_index,
            filter_eps=_cut_cross_entropy_filter_eps(self.filter_eps),
        )
        return self._linear_cross_entropy(
            hidden_2d,
            weight,
            labels_1d,
            **kwargs,
        )


class _TorchCompileBF16Backend(_BaseCCEBackend):
    name = "torch_compile_bf16"

    def __init__(self, ignore_index):
        super().__init__(ignore_index)
        self._linear_cross_entropy = _get_raw_linear_cross_entropy()

    def training_loss(self, hidden_2d, weight, labels_1d):
        if _labels_all_ignored(labels_1d, self.ignore_index):
            return hidden_2d.sum() * 0.0
        return self._linear_cross_entropy(
            hidden_2d,
            weight,
            labels_1d,
            shift=False,
            reduction="mean",
            ignore_index=self.ignore_index,
            impl="torch_compile",
        )


class _NativeMXFP4PrecisionBackend(_BaseCCEBackend):
    name = "native_mxfp4"

    def __init__(
        self,
        ignore_index,
        implementation,
        quant_mode,
        filter_eps=0.0,
        forward_precision="bf16",
        backward_precision="bf16",
    ):
        super().__init__(ignore_index)
        if implementation != "v4":
            raise ValueError("native_mxfp4 CCE requires implementation='v4'.")
        if quant_mode != "enc":
            raise ValueError("native_mxfp4 CCE requires quant_mode='enc'.")
        if _resolve_filter_eps(filter_eps) != 0.0:
            raise ValueError("native_mxfp4 CCE does not support filter_eps.")
        if forward_precision not in {"bf16", "fp4"}:
            raise ValueError(
                "native_mxfp4 forward_precision must be 'bf16' or 'fp4'."
            )
        if backward_precision not in {"bf16", "fp4"}:
            raise ValueError(
                "native_mxfp4 backward_precision must be 'bf16' or 'fp4'."
            )
        self.implementation = implementation
        self.quant_mode = quant_mode
        self.forward_precision = forward_precision
        self.backward_precision = backward_precision

    def training_loss(self, hidden_2d, weight, labels_1d):
        if _labels_all_ignored(labels_1d, self.ignore_index):
            return hidden_2d.sum() * 0.0
        v4 = _load_fp4_cce_tk_v4()
        if v4.mxfp4_cce_tk_native_precision is None:
            raise RuntimeError(
                "native_mxfp4 requires fp4_matmul native precision CCE support."
            )
        return v4.mxfp4_cce_tk_native_precision(
            hidden_2d.contiguous(),
            _fp4_cce_bf16_weight(weight),
            labels_1d.contiguous(),
            ignore_index=self.ignore_index,
            mode=_mx_mode_id(self.quant_mode),
            forward_precision=self.forward_precision,
            backward_precision=self.backward_precision,
        )


class _NVFP4Backend(_BaseCCEBackend):
    name = "nvfp4"

    def __init__(self, ignore_index, implementation, quant_mode, filter_eps=0.0):
        super().__init__(ignore_index)
        self.implementation = implementation
        self.quant_mode = quant_mode
        self.filter_eps = _resolve_filter_eps(filter_eps)
        self._weight_quant_cache_key = None
        self._weight_quant_cache = None
        if self.implementation == "v4":
            if self.filter_eps > 0.0:
                raise ValueError("NVFP4 v4 pcache path does not support filter_eps.")

    def invalidate_weight_cache(self):
        self._weight_quant_cache_key = None
        self._weight_quant_cache = None

    def _forward_weight_quantized(self, weight, v4, *, cache_owner=None):
        if not _flag_enabled("FP4_CCE_V4_WEIGHT_QUANT_CACHE"):
            return None
        direct_fp8 = _nvfp4_direct_fp8_forward_enabled()
        mxfp8 = _nvfp4_mxfp8_forward_enabled()
        if not (direct_fp8 or mxfp8):
            raise RuntimeError(
                "FP4_CCE_V4_WEIGHT_QUANT_CACHE currently requires direct-FP8 "
                "or MXFP8 forward."
            )
        mixed_g_cache = mxfp8 and _flag_enabled(
            "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE"
        )
        native_mxfp8_col = mxfp8 and _nvfp4_mxfp8_native_col_enabled()
        producer = (
            v4.quantize_direct_fp8_row_mxfp4_col
            if direct_fp8
            else v4.quantize_mxfp8_row_and_col_fused
            if mixed_g_cache
            else v4.quantize_mxfp8_row_nvfp4_col_localcta_v4
            if native_mxfp8_col
            else v4.quantize_mxfp8_row_mxfp4_col
        )
        if producer is None:
            raise RuntimeError(
                "FP8 weight caching requires a matching fp4_matmul runtime."
            )
        owner = cache_owner if cache_owner is not None else weight
        key = (
            "direct_fp8"
            if direct_fp8
            else "mxfp8_mxfp8_col"
            if mixed_g_cache
            else "mxfp8_nvfp4_col"
            if native_mxfp8_col
            else "mxfp8_mxfp4_col",
            # FSDP may update persistent unsharded parameters without bumping
            # the Python tensor's version counter. Reuse within accumulation
            # microbatches, but force a fresh quantization after each optimizer
            # step.
            os.environ.get("LBT_TRACE_ACTIVE_STEP"),
            id(owner),
            int(getattr(owner, "_version", getattr(weight, "_version", -1))),
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.device.type,
            weight.device.index,
            weight.dtype,
        )
        if key != self._weight_quant_cache_key:
            producer_kwargs = (
                {"role": "W"}
                if direct_fp8
                else {
                    "encode_centric": _nv_encode_inputs_for_mode(self.quant_mode),
                    "four_over_six_mae": _flag_enabled(
                        "FP4_CCE_V4_NVFP4_W_FOUROVERSIX_MAE"
                    ),
                }
                if native_mxfp8_col and not mixed_g_cache
                else {}
            )
            self._weight_quant_cache = producer(weight.detach(), **producer_kwargs)
            self._weight_quant_cache_key = key
        return self._weight_quant_cache

    def _prequantized_weight_kwargs(self, weight, v4, *, cache_owner=None):
        quantized = self._forward_weight_quantized(
            weight, v4, cache_owner=cache_owner
        )
        return {"weight_quantized": quantized} if quantized is not None else {}

    def training_loss(self, hidden_2d, weight, labels_1d):
        if _labels_all_ignored(labels_1d, self.ignore_index):
            return hidden_2d.sum() * 0.0
        if self.implementation == "v4":
            v4 = _load_fp4_cce_tk_v4()
            hidden = hidden_2d.contiguous()
            weight_contig = _fp4_cce_bf16_weight(weight)
            labels = labels_1d.contiguous()
            encode_centric = _nv_encode_inputs_for_mode(self.quant_mode)
            forward_formats = (
                _nvfp4_mxfp8_forward_enabled(),
                _nvfp4_mxfp4_forward_enabled(),
                _nvfp4_direct_fp8_forward_enabled(),
            )
            if sum(forward_formats) > 1:
                raise RuntimeError(
                    "MXFP8, MXFP4, and direct-FP8 head forward modes are "
                    "mutually exclusive."
                )
            if _nvfp4_mxfp8_forward_enabled():
                native_mxfp8_col = _nvfp4_mxfp8_native_col_enabled()
                quantize_x = (
                    v4.quantize_mxfp8_row_nvfp4_col_localcta_v4
                    if native_mxfp8_col
                    else v4.quantize_mxfp8_row_mxfp4_col
                )
                if (
                    v4.MXFP8Quantized is None
                    or quantize_x is None
                    or v4.nvfp4_cce_tk_v4_pcache_prequantized_x is None
                ):
                    raise RuntimeError(
                        "mixed MXFP8 output-head training requires a "
                        "matching fp4_matmul runtime."
                    )
                producer_kwargs = (
                    {
                        "encode_centric": encode_centric,
                        "four_over_six_mae": _flag_enabled(
                            "FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE"
                        ),
                    }
                    if native_mxfp8_col
                    else {}
                )
                x_q, x_col_q = quantize_x(hidden.detach(), **producer_kwargs)
                return v4.nvfp4_cce_tk_v4_pcache_prequantized_x(
                    hidden,
                    x_q,
                    x_col_q,
                    weight_contig,
                    labels,
                    ignore_index=self.ignore_index,
                    encode_centric=encode_centric,
                    **self._prequantized_weight_kwargs(
                        weight_contig, v4, cache_owner=weight
                    ),
                )
            if _nvfp4_direct_fp8_forward_enabled():
                if not _flag_enabled("FP4_CCE_V4_MXFP4_G_CACHE"):
                    raise RuntimeError(
                        "FP4_CCE_V4_NVFP4_DIRECT_FP8_FORWARD currently requires "
                        "FP4_CCE_V4_MXFP4_G_CACHE=1."
                    )
                if (
                    v4.DirectFP8Quantized is None
                    or v4.quantize_direct_fp8_row_mxfp4_col is None
                    or v4.nvfp4_cce_tk_v4_pcache_prequantized_x is None
                ):
                    raise RuntimeError(
                        "direct-FP8/MXFP4 output-head training requires a "
                        "matching fp4_matmul runtime."
                    )
                x_q, x_col_q = v4.quantize_direct_fp8_row_mxfp4_col(
                    hidden.detach(), role="X"
                )
                return v4.nvfp4_cce_tk_v4_pcache_prequantized_x(
                    hidden,
                    x_q,
                    x_col_q,
                    weight_contig,
                    labels,
                    ignore_index=self.ignore_index,
                    encode_centric=encode_centric,
                    **self._prequantized_weight_kwargs(
                        weight_contig, v4, cache_owner=weight
                    ),
                )
            if _nvfp4_mxfp4_forward_enabled():
                if _flag_enabled("FP4_CCE_V4_MXFP8_G_CACHE"):
                    raise RuntimeError(
                        "FP4_CCE_V4_NVFP4_MXFP4_FORWARD is incompatible with "
                        "FP4_CCE_V4_MXFP8_G_CACHE=1."
                    )
                use_mxfp4_g_cache = _flag_enabled(
                    "FP4_CCE_V4_MXFP4_G_CACHE"
                )
                quantize_x = (
                    v4.quantize_mxfp4_row_and_col_tk
                    if use_mxfp4_g_cache
                    else v4.quantize_mxfp4_row_nvfp4_col_v5
                )
                if (
                    v4.MXFP4Quantized is None
                    or quantize_x is None
                    or v4.nvfp4_cce_tk_v4_pcache_prequantized_x is None
                ):
                    raise RuntimeError(
                        "MXFP4-forward output-head training "
                        "requires a matching fp4_matmul runtime."
                    )
                if use_mxfp4_g_cache:
                    x_q, x_col_q = quantize_x(
                        hidden.detach(),
                        mode=_mx_mode_id(self.quant_mode),
                        role="X",
                    )
                else:
                    x_q, x_col_q = quantize_x(
                        hidden.detach(),
                        encode_centric=encode_centric,
                        role="X",
                    )
                return v4.nvfp4_cce_tk_v4_pcache_prequantized_x(
                    hidden,
                    x_q,
                    x_col_q,
                    weight_contig,
                    labels,
                    ignore_index=self.ignore_index,
                    encode_centric=encode_centric,
                )
            if _nvfp4_v4_prequant_x_enabled():
                if (
                    v4.quantize_nvfp4_row_and_col_tk is None
                    or v4.nvfp4_cce_tk_v4_pcache_prequantized_x is None
                ):
                    raise RuntimeError(
                        "FP4_CCE_V4_NVFP4_PREQUANT_X=1 requires a fp4_matmul "
                        "checkout with NVFP4 v4 prequantized-x support."
                    )
                x_q, x_col_q = v4.quantize_nvfp4_row_and_col_tk(
                    hidden.detach(),
                    encode_centric=encode_centric,
                )
                return v4.nvfp4_cce_tk_v4_pcache_prequantized_x(
                    hidden,
                    x_q,
                    x_col_q,
                    weight_contig,
                    labels,
                    ignore_index=self.ignore_index,
                    encode_centric=encode_centric,
                )
            return v4.nvfp4_cce_tk_v4_pcache(
                hidden,
                weight_contig,
                labels,
                ignore_index=self.ignore_index,
                encode_centric=encode_centric,
            )
        hidden_kernel, weight_kernel = _fp4_cce_inputs_for_kernel(hidden_2d, weight)
        return _NVFP4CCEFunction.apply(
            hidden_kernel,
            weight_kernel,
            labels_1d.contiguous(),
            self.ignore_index,
            self.implementation,
            self.quant_mode,
            self.filter_eps,
        )

    def quantize_final_norm_x(self, pre_norm_2d, norm_weight, epsilon):
        if self.implementation != "v4":
            raise NotImplementedError("NVFP4 fused x producer is only implemented for v4.")
        if _nvfp4_mxfp4_forward_enabled():
            raise RuntimeError(
                "MXFP4-forward/native-v5-G currently requires "
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER=0."
            )
        pre_norm_2d = _local_tensor_for_cce(pre_norm_2d)
        norm_weight = _local_tensor_for_cce(norm_weight)
        v4 = _load_fp4_cce_tk_v4()
        if _nvfp4_mxfp8_forward_enabled():
            native_mxfp8_col = _nvfp4_mxfp8_native_col_enabled()
            fn = (
                v4.quantize_mxfp8_norm_row_nvfp4_col_with_output_localcta_v4
                if native_mxfp8_col
                else v4.quantize_mxfp8_norm_row_mxfp4_col_with_output_localcta_v4
            )
            if fn is None:
                raise RuntimeError(
                    "fused MXFP8 final RMSNorm requires a matching "
                    "fp4_matmul runtime."
                )
            producer_kwargs = (
                {
                    "encode_centric": _nv_encode_inputs_for_mode(self.quant_mode),
                    "four_over_six_mae": _flag_enabled(
                        "FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE"
                    ),
                }
                if native_mxfp8_col
                else {}
            )
            _normed, x_q, x_col_q, _inv_rms, _scratch = fn(
                pre_norm_2d.detach().contiguous(),
                norm_weight.detach().contiguous(),
                float(epsilon),
                **producer_kwargs,
            )
            return x_q, x_col_q
        if _nvfp4_direct_fp8_forward_enabled():
            fn = v4.quantize_direct_fp8_norm_row_mxfp4_col_with_output_localcta_v4
            if fn is None:
                raise RuntimeError(
                    "fused direct-FP8/MXFP4 final RMSNorm requires a matching "
                    "fp4_matmul runtime."
                )
            _normed, x_q, x_col_q, _inv_rms, _scratch = fn(
                pre_norm_2d.detach().contiguous(),
                norm_weight.detach().contiguous(),
                float(epsilon),
                role="X",
            )
            return x_q, x_col_q
        if v4.quantize_nvfp4_norm_row_and_col_tk is None:
            raise RuntimeError(
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER requires a fp4_matmul checkout "
                "with quantize_nvfp4_norm_row_and_col_tk."
            )
        x_q, x_col_q, _inv_rms, _amax = v4.quantize_nvfp4_norm_row_and_col_tk(
            pre_norm_2d.detach().contiguous(),
            norm_weight.detach().contiguous(),
            float(epsilon),
            encode_centric=_nv_encode_inputs_for_mode(self.quant_mode),
        )
        return x_q, x_col_q

    def training_loss_prequantized_x(self, hidden_2d, x_q, x_col_q, weight, labels_1d):
        if self.implementation != "v4":
            raise NotImplementedError("NVFP4 prequantized-x consumption is only implemented for v4.")
        if _nvfp4_mxfp4_forward_enabled():
            raise RuntimeError(
                "MXFP4-forward/native-v5-G currently requires "
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER=0."
            )
        v4 = _load_fp4_cce_tk_v4()
        if v4.nvfp4_cce_tk_v4_pcache_prequantized_x is None:
            raise RuntimeError(
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER requires a fp4_matmul checkout "
                "with NVFP4 v4 prequantized-x support."
            )
        hidden = hidden_2d.contiguous()
        weight_contig = _fp4_cce_bf16_weight(weight)
        labels = labels_1d.contiguous()

        return v4.nvfp4_cce_tk_v4_pcache_prequantized_x(
            hidden,
            x_q,
            x_col_q,
            weight_contig,
            labels,
            ignore_index=self.ignore_index,
            encode_centric=_nv_encode_inputs_for_mode(self.quant_mode),
            **self._prequantized_weight_kwargs(
                weight_contig, v4, cache_owner=weight
            ),
        )

    def training_loss_vocab_parallel(
        self,
        hidden_2d,
        weight,
        labels_1d,
        *,
        tp_group,
        vocab_start: int,
        global_vocab_size: int,
        reduce_dE: bool = False,
    ):
        if self.implementation != "v4":
            raise NotImplementedError("NVFP4 vocab-parallel CCE is only implemented for v4.")
        v4 = _load_fp4_cce_tk_v4()
        if v4.nvfp4_cce_tk_v4_vocab_parallel is None:
            raise RuntimeError(
                "LBT_BRIDGE_FP4_CCE_TP_MODE=vocab_parallel requires a fp4_matmul "
                "checkout with NVFP4 v4 vocab-parallel CCE support."
            )
        hidden = hidden_2d.contiguous()
        weight_contig = _fp4_cce_bf16_weight(weight)
        labels = labels_1d.contiguous()
        return v4.nvfp4_cce_tk_v4_vocab_parallel(
            hidden,
            weight_contig,
            labels,
            ignore_index=self.ignore_index,
            global_vocab_size=int(global_vocab_size),
            vocab_start=int(vocab_start),
            tp_group=tp_group,
            reduce_dE=bool(reduce_dE),
            encode_centric=_nv_encode_inputs_for_mode(self.quant_mode),
        )


class _MXFP4Backend(_BaseCCEBackend):
    name = "mxfp4"

    def __init__(self, ignore_index, implementation, quant_mode, filter_eps=0.0):
        super().__init__(ignore_index)
        self.implementation = implementation
        self.quant_mode = quant_mode
        self.filter_eps = _resolve_filter_eps(filter_eps)
        if self.implementation == "v4":
            if self.filter_eps > 0.0:
                raise ValueError("MXFP4 v4 pcache path does not support filter_eps.")

    def training_loss(self, hidden_2d, weight, labels_1d):
        if _labels_all_ignored(labels_1d, self.ignore_index):
            return hidden_2d.sum() * 0.0
        if self.implementation == "v4":
            v4 = _load_fp4_cce_tk_v4()
            hidden = hidden_2d.contiguous()
            weight_contig = _fp4_cce_bf16_weight(weight)
            labels = labels_1d.contiguous()
            mode = _mx_mode_id(self.quant_mode)
            if _mxfp4_v4_prequant_x_enabled():
                if (
                    v4.quantize_mxfp4_row_and_col_tk is None
                    or v4.mxfp4_cce_tk_v4_pcache_prequantized_x is None
                ):
                    raise RuntimeError(
                        "FP4_CCE_V4_MXFP4_PREQUANT_X=1 requires a fp4_matmul "
                        "checkout with MXFP4 v4 prequantized-x support."
                    )
                x_q, x_col_q = v4.quantize_mxfp4_row_and_col_tk(hidden.detach(), mode=mode)
                return v4.mxfp4_cce_tk_v4_pcache_prequantized_x(
                    hidden,
                    x_q,
                    x_col_q,
                    weight_contig,
                    labels,
                    ignore_index=self.ignore_index,
                    mode=mode,
                )
            return v4.mxfp4_cce_tk_v4_pcache(
                hidden,
                weight_contig,
                labels,
                ignore_index=self.ignore_index,
                mode=mode,
            )
        hidden_kernel, weight_kernel = _fp4_cce_inputs_for_kernel(hidden_2d, weight)
        return _MXFP4CCEFunction.apply(
            hidden_kernel,
            weight_kernel,
            labels_1d.contiguous(),
            self.ignore_index,
            self.implementation,
            self.quant_mode,
            self.filter_eps,
        )

    def quantize_final_norm_x(self, pre_norm_2d, norm_weight, epsilon):
        if self.implementation != "v4":
            raise NotImplementedError("MXFP4 fused x producer is only implemented for v4.")
        pre_norm_2d = _local_tensor_for_cce(pre_norm_2d)
        norm_weight = _local_tensor_for_cce(norm_weight)
        v4 = _load_fp4_cce_tk_v4()
        if v4.quantize_mxfp4_norm_row_and_col_tk is None:
            raise RuntimeError(
                "FP4_CCE_V4_MXFP4_FUSED_X_PRODUCER requires a fp4_matmul checkout "
                "with quantize_mxfp4_norm_row_and_col_tk."
            )
        return v4.quantize_mxfp4_norm_row_and_col_tk(
            pre_norm_2d.detach().contiguous(),
            norm_weight.detach().contiguous(),
            float(epsilon),
            mode=_mx_mode_id(self.quant_mode),
        )[:2]

    def training_loss_prequantized_x(self, hidden_2d, x_q, x_col_q, weight, labels_1d):
        if self.implementation != "v4":
            raise NotImplementedError("MXFP4 prequantized-x consumption is only implemented for v4.")
        v4 = _load_fp4_cce_tk_v4()
        if v4.mxfp4_cce_tk_v4_pcache_prequantized_x is None:
            raise RuntimeError(
                "FP4_CCE_V4_MXFP4_FUSED_X_PRODUCER requires a fp4_matmul checkout "
                "with MXFP4 v4 prequantized-x support."
            )
        hidden = hidden_2d.contiguous()
        weight_contig = _fp4_cce_bf16_weight(weight)
        labels = labels_1d.contiguous()

        return v4.mxfp4_cce_tk_v4_pcache_prequantized_x(
            hidden,
            x_q,
            x_col_q,
            weight_contig,
            labels,
            ignore_index=self.ignore_index,
            mode=_mx_mode_id(self.quant_mode),
        )

    def training_loss_vocab_parallel(
        self,
        hidden_2d,
        weight,
        labels_1d,
        *,
        tp_group,
        vocab_start: int,
        global_vocab_size: int,
        reduce_dE: bool = False,
    ):
        if self.implementation != "v4":
            raise NotImplementedError("MXFP4 vocab-parallel CCE is only implemented for v4.")
        v4 = _load_fp4_cce_tk_v4()
        if v4.mxfp4_cce_tk_v4_vocab_parallel is None:
            raise RuntimeError(
                "LBT_BRIDGE_FP4_CCE_TP_MODE=vocab_parallel requires a fp4_matmul "
                "checkout with MXFP4 v4 vocab-parallel CCE support."
            )
        hidden = hidden_2d.contiguous()
        weight_contig = _fp4_cce_bf16_weight(weight)
        labels = labels_1d.contiguous()
        return v4.mxfp4_cce_tk_v4_vocab_parallel(
            hidden,
            weight_contig,
            labels,
            ignore_index=self.ignore_index,
            global_vocab_size=int(global_vocab_size),
            vocab_start=int(vocab_start),
            tp_group=tp_group,
            reduce_dE=bool(reduce_dE),
            mode=_mx_mode_id(self.quant_mode),
        )

    def training_loss_vocab_parallel_prequantized_x(
        self,
        hidden_2d,
        x_q,
        x_col_q,
        weight,
        labels_1d,
        *,
        tp_group,
        vocab_start: int,
        global_vocab_size: int,
        reduce_dE: bool = False,
    ):
        if self.implementation != "v4":
            raise NotImplementedError("MXFP4 vocab-parallel prequantized-x CCE is only implemented for v4.")
        v4 = _load_fp4_cce_tk_v4()
        fn = getattr(v4, "mxfp4_cce_tk_v4_vocab_parallel_prequantized_x", None)
        if fn is None:
            raise RuntimeError(
                "FP4_CCE_V4_MXFP4_FUSED_X_PRODUCER requires a fp4_matmul checkout "
                "with MXFP4 v4 vocab-parallel prequantized-x support."
            )
        hidden = hidden_2d.contiguous()
        weight_contig = _fp4_cce_bf16_weight(weight)
        labels = labels_1d.contiguous()
        return fn(
            hidden,
            x_q,
            x_col_q,
            weight_contig,
            labels,
            ignore_index=self.ignore_index,
            global_vocab_size=int(global_vocab_size),
            vocab_start=int(vocab_start),
            tp_group=tp_group,
            reduce_dE=bool(reduce_dE),
            mode=_mx_mode_id(self.quant_mode),
        )


def _build_backend(job_config):
    if getattr(job_config.training, "enable_cce", False):
        return _TritonBF16Backend(
            ignore_index=-100,
            filter_eps=getattr(job_config.fp4_cce, "filter_eps", 0.0),
        )

    cfg = job_config.fp4_cce
    if not cfg.enabled:
        raise ValueError("apply_cce_backend_patch called but fp4_cce is disabled.")
    if cfg.backend == "triton_bf16":
        return _TritonBF16Backend(ignore_index=cfg.ignore_index, filter_eps=getattr(cfg, "filter_eps", 0.0))
    if cfg.backend == "torch_compile_bf16":
        return _TorchCompileBF16Backend(ignore_index=cfg.ignore_index)
    if cfg.backend == "native_mxfp4":
        return _NativeMXFP4PrecisionBackend(
            ignore_index=cfg.ignore_index,
            implementation=cfg.implementation,
            quant_mode=cfg.quant_mode,
            filter_eps=getattr(cfg, "filter_eps", 0.0),
            forward_precision=cfg.forward_precision,
            backward_precision=cfg.backward_precision,
        )
    if cfg.backend == "nvfp4":
        return _NVFP4Backend(
            ignore_index=cfg.ignore_index,
            implementation=cfg.implementation,
            quant_mode=cfg.quant_mode,
            filter_eps=getattr(cfg, "filter_eps", 0.0),
        )
    if cfg.backend == "mxfp4":
        _guard_mxfp4_cce_env()
        return _MXFP4Backend(
            ignore_index=cfg.ignore_index,
            implementation=cfg.implementation,
            quant_mode=cfg.quant_mode,
            filter_eps=getattr(cfg, "filter_eps", 0.0),
        )
    raise ValueError(f"Unsupported fp4_cce backend: {cfg.backend!r}")


class _FinalRMSNormQuantProducerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre_norm_2d, norm_weight, epsilon: float, backend_name: str, mode: int):
        v4 = _load_fp4_cce_tk_v4()
        if backend_name == "nvfp4":
            if _nvfp4_mxfp8_forward_enabled():
                native_mxfp8_col = _nvfp4_mxfp8_native_col_enabled()
                fn = (
                    v4.quantize_mxfp8_norm_row_nvfp4_col_with_output_localcta_v4
                    if native_mxfp8_col
                    else v4.quantize_mxfp8_norm_row_mxfp4_col_with_output_localcta_v4
                )
                error = "fused MXFP8 final RMSNorm producer"
                kwargs = (
                    {
                        "encode_centric": bool(mode),
                        "four_over_six_mae": _flag_enabled(
                            "FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE"
                        ),
                    }
                    if native_mxfp8_col
                    else {}
                )
            elif _nvfp4_direct_fp8_forward_enabled():
                fn = v4.quantize_direct_fp8_norm_row_mxfp4_col_with_output_localcta_v4
                error = "fused direct-FP8/MXFP4 final RMSNorm producer"
                kwargs = {"role": "X"}
            else:
                fn = v4.quantize_nvfp4_norm_row_and_col_with_output_tk
                error = "fused NVFP4 final RMSNorm producer"
                kwargs = {"encode_centric": bool(mode)}
            if fn is None:
                raise RuntimeError(
                    f"{error} requires a matching fp4_matmul runtime."
                )
            normed, x_q, x_col_q, inv_rms, _amax = fn(
                pre_norm_2d,
                norm_weight,
                float(epsilon),
                **kwargs,
            )
            if _nvfp4_mxfp8_forward_enabled() or _nvfp4_direct_fp8_forward_enabled():
                row_data = x_q.fp8
                mixed_dw_mxfp8_cols = (
                    _nvfp4_mxfp8_forward_enabled()
                    and _mixed_dw_mxfp8_cols_enabled()
                )
                col_data = (
                    x_col_q.fp8 if mixed_dw_mxfp8_cols else x_col_q.fp4
                )
                row_sg = torch.empty(0, dtype=torch.float32, device=pre_norm_2d.device)
                col_sg = (
                    x_col_q.sg
                    if (
                        _nvfp4_mxfp8_native_col_enabled()
                        and not mixed_dw_mxfp8_cols
                    )
                    else torch.empty(0, dtype=torch.float32, device=pre_norm_2d.device)
                )
            else:
                row_data = x_q.fp4
                col_data = x_col_q.fp4
                row_sg = x_q.sg
                col_sg = x_col_q.sg
        elif backend_name == "mxfp4":
            fn = v4.quantize_mxfp4_norm_row_and_col_with_output_tk
            if fn is None:
                raise RuntimeError(
                    "FP4_CCE_V4_MXFP4_FUSED_X_PRODUCER requires a fp4_matmul checkout "
                    "with quantize_mxfp4_norm_row_and_col_with_output_tk."
                )
            normed, x_q, x_col_q, inv_rms = fn(
                pre_norm_2d,
                norm_weight,
                float(epsilon),
                mode=int(mode),
            )
            row_data = x_q.fp4
            col_data = x_col_q.fp4
            row_sg = torch.empty(0, dtype=torch.float32, device=pre_norm_2d.device)
            col_sg = torch.empty(0, dtype=torch.float32, device=pre_norm_2d.device)
        else:
            raise RuntimeError(f"Unsupported fused final RMSNorm quant backend: {backend_name!r}")

        ctx.set_materialize_grads(False)
        ctx.save_for_backward(pre_norm_2d, norm_weight, inv_rms)
        ctx.hidden_size = pre_norm_2d.shape[1]
        ctx.mark_non_differentiable(
            row_data,
            x_q.sc,
            col_data,
            x_col_q.sc,
            row_sg,
            col_sg,
        )
        return (
            normed,
            row_data,
            x_q.sc,
            col_data,
            x_col_q.sc,
            row_sg,
            col_sg,
        )

    @staticmethod
    def backward(ctx, grad_normed, *_unused):
        if grad_normed is None:
            return None, None, None, None, None

        pre_norm_2d, norm_weight, inv_rms = ctx.saved_tensors
        grad_input, grad_gamma = torch.ops.aten._fused_rms_norm_backward(
            grad_normed.contiguous(),
            pre_norm_2d,
            [ctx.hidden_size],
            inv_rms,
            norm_weight,
            [True, True],
        )
        return (
            grad_input,
            grad_gamma,
            None,
            None,
            None,
        )


def _produce_final_norm_x_with_quant(pre_norm_2d, norm_weight, epsilon: float, backend):
    pre_norm_2d = _local_tensor_for_cce(pre_norm_2d)
    norm_weight = _local_tensor_for_cce(norm_weight)
    mode = (
        _mx_mode_id(backend.quant_mode)
        if backend.name == "mxfp4"
        else int(_nv_encode_inputs_for_mode(backend.quant_mode))
    )
    if _v4_fused_x_producer_pre_sync_enabled():
        torch.cuda.synchronize(pre_norm_2d.device)
    (
        normed,
        row_data,
        row_sc,
        col_data,
        col_sc,
        row_sg,
        col_sg,
    ) = _FinalRMSNormQuantProducerFunction.apply(
        pre_norm_2d,
        norm_weight,
        float(epsilon),
        backend.name,
        int(mode),
    )
    if _v4_fused_x_producer_sync_enabled():
        torch.cuda.synchronize(pre_norm_2d.device)
    if backend.name == "nvfp4":
        v4 = _load_fp4_cce_tk_v4()
        if _nvfp4_mxfp8_forward_enabled():
            if _nvfp4_mxfp8_native_col_enabled():
                if _mixed_dw_mxfp8_cols_enabled():
                    return (
                        normed,
                        v4.MXFP8Quantized(row_data, row_sc),
                        v4.MXFP8Quantized(col_data, col_sc),
                    )
                return (
                    normed,
                    v4.MXFP8Quantized(row_data, row_sc),
                    v4.NVFP4Quantized(
                        col_data,
                        col_sc,
                        col_sg,
                        layout="localcta",
                    ),
                )
            return (
                normed,
                v4.MXFP8Quantized(row_data, row_sc),
                v4.MXFP4Quantized(col_data, col_sc),
            )
        if _nvfp4_direct_fp8_forward_enabled():
            return (
                normed,
                v4.DirectFP8Quantized(row_data, row_sc),
                v4.MXFP4Quantized(col_data, col_sc),
            )
        return (
            normed,
            types.SimpleNamespace(fp4=row_data, sc=row_sc, sg=row_sg),
            types.SimpleNamespace(fp4=col_data, sc=col_sc, sg=col_sg),
        )
    return (
        normed,
        types.SimpleNamespace(fp4=row_data, sc=row_sc),
        types.SimpleNamespace(fp4=col_data, sc=col_sc),
    )


class _FinalRMSNormQuantOnlyProducerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre_norm_2d, norm_weight, epsilon: float, backend_name: str, mode: int):
        if backend_name != "mxfp4":
            raise RuntimeError(
                f"Unsupported quant-only final RMSNorm quant backend: {backend_name!r}"
            )

        v4 = _load_fp4_cce_tk_v4()
        fn = v4.quantize_mxfp4_norm_row_and_col_tk
        if fn is None:
            raise RuntimeError(
                "LBT_BRIDGE_FP4_CCE_FINAL_NORM_PREQUANT_QUANT_ONLY=1 requires a "
                "fp4_matmul checkout with quantize_mxfp4_norm_row_and_col_tk."
            )
        x_q, x_col_q, inv_rms = fn(
            pre_norm_2d,
            norm_weight,
            float(epsilon),
            mode=int(mode),
        )

        # The downstream prequantized CCE path never reads x values in forward;
        # this carrier exists only to receive dE and route it through RMSNorm bwd.
        carrier = torch.empty_strided(
            tuple(pre_norm_2d.shape),
            tuple(pre_norm_2d.stride()),
            dtype=pre_norm_2d.dtype,
            device=pre_norm_2d.device,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(pre_norm_2d, norm_weight, inv_rms)
        ctx.hidden_size = pre_norm_2d.shape[1]
        ctx.mark_non_differentiable(x_q.fp4, x_q.sc, x_col_q.fp4, x_col_q.sc)
        return carrier, x_q.fp4, x_q.sc, x_col_q.fp4, x_col_q.sc

    @staticmethod
    def backward(ctx, grad_carrier, *_unused):
        if grad_carrier is None:
            return None, None, None, None, None

        pre_norm_2d, norm_weight, inv_rms = ctx.saved_tensors
        grad_input, grad_gamma = torch.ops.aten._fused_rms_norm_backward(
            grad_carrier.contiguous(),
            pre_norm_2d,
            [ctx.hidden_size],
            inv_rms,
            norm_weight,
            [True, True],
        )
        return (
            grad_input,
            grad_gamma,
            None,
            None,
            None,
        )


def _produce_final_norm_x_quant_only_for_cce(pre_norm_2d, norm_weight, epsilon: float, backend):
    pre_norm_2d = _local_tensor_for_cce(pre_norm_2d)
    norm_weight = _local_tensor_for_cce(norm_weight)
    if backend.name != "mxfp4" or getattr(backend, "implementation", None) != "v4":
        raise RuntimeError("Quant-only final norm CCE producer is only implemented for MXFP4 v4.")
    mode = _mx_mode_id(backend.quant_mode)
    if _v4_fused_x_producer_pre_sync_enabled():
        torch.cuda.synchronize(pre_norm_2d.device)
    carrier, row_fp4, row_sc, col_fp4, col_sc = _FinalRMSNormQuantOnlyProducerFunction.apply(
        pre_norm_2d,
        norm_weight,
        float(epsilon),
        backend.name,
        int(mode),
    )
    if _v4_fused_x_producer_sync_enabled():
        torch.cuda.synchronize(pre_norm_2d.device)
    return (
        carrier,
        types.SimpleNamespace(fp4=row_fp4, sc=row_sc),
        types.SimpleNamespace(fp4=col_fp4, sc=col_sc),
    )


class TitanCCEHead(nn.Module):
    def __init__(self, original_linear, backend):
        super().__init__()
        self.weight = nn.Parameter(original_linear.weight.detach())
        self.bias = None
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.detach())
        self.backend = backend

    def invalidate_weight_cache(self):
        invalidate = getattr(self.backend, "invalidate_weight_cache", None)
        if callable(invalidate):
            invalidate()

    @torch._dynamo.disable
    def forward(self, x, labels=None):
        if labels is None:
            return F.linear(x, self.weight, self.bias)
        if self.bias is not None:
            raise NotImplementedError(
                "fp4_cce training path does not support an output bias term. "
                "Inference remains supported."
            )
        hidden_2d = x.reshape(-1, x.shape[-1])
        labels_1d = labels.reshape(-1).to(device=hidden_2d.device, dtype=torch.int64)
        selective_bf16_dhidden = _bf16_dhidden_only_enabled()
        if selective_bf16_dhidden:
            _validate_bf16_dhidden_only_backend(
                self.backend,
                prequantized_x=False,
            )
        if _lowp_logits_bf16_dhidden_enabled():
            _validate_lowp_logits_bf16_dhidden_backend(
                self.backend,
                prequantized_x=False,
            )
        loss = self.backend.training_loss(hidden_2d, self.weight, labels_1d)
        _queue_common_eval_metric(
            hidden_2d,
            self.weight,
            labels_1d,
            self.backend.ignore_index,
            loss,
        )
        return loss

    @torch._dynamo.disable
    def forward_from_pre_norm(self, pre_norm_x, norm, labels=None):
        if labels is None or not _v4_fused_x_producer_enabled_for_backend(self.backend):
            x = norm(pre_norm_x) if norm is not None else pre_norm_x
            return self.forward(x, labels=labels)
        if self.bias is not None:
            raise NotImplementedError(
                "fp4_cce training path does not support an output bias term. "
                "Inference remains supported."
            )
        if norm is None or not hasattr(norm, "weight"):
            x = pre_norm_x
            return self.forward(x, labels=labels)

        pre_norm_2d = pre_norm_x.reshape(-1, pre_norm_x.shape[-1]).contiguous()
        producer_pre_norm_2d = pre_norm_2d
        if producer_pre_norm_2d.dtype != torch.bfloat16:
            producer_pre_norm_2d = producer_pre_norm_2d.to(torch.bfloat16)
        labels_1d = labels.reshape(-1).to(device=pre_norm_2d.device, dtype=torch.int64)
        epsilon = getattr(norm, "eps", 1e-5)
        if epsilon is None:
            epsilon = 1e-5
        norm_weight = _local_tensor_for_cce(norm.weight)
        norm_weight = norm_weight.to(device=pre_norm_2d.device, dtype=torch.bfloat16).contiguous()

        if _v4_fused_x_producer_quant_only_enabled():
            x_q, x_col_q = self.backend.quantize_final_norm_x(
                producer_pre_norm_2d,
                norm_weight,
                float(epsilon),
            )
            hidden = norm(pre_norm_x)
            hidden_2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        else:
            hidden_2d, x_q, x_col_q = _produce_final_norm_x_with_quant(
                producer_pre_norm_2d,
                norm_weight,
                float(epsilon),
                self.backend,
            )
        selective_bf16_dhidden = _bf16_dhidden_only_enabled()
        if selective_bf16_dhidden:
            _validate_bf16_dhidden_only_backend(
                self.backend,
                prequantized_x=True,
            )
        if _lowp_logits_bf16_dhidden_enabled():
            _validate_lowp_logits_bf16_dhidden_backend(
                self.backend,
                prequantized_x=True,
            )
        loss = self.backend.training_loss_prequantized_x(
            hidden_2d.detach() if selective_bf16_dhidden else hidden_2d,
            x_q,
            x_col_q,
            self.weight,
            labels_1d,
        )
        if selective_bf16_dhidden:
            loss = _attach_bf16_dhidden_to_lowp_loss(
                loss,
                hidden_2d,
                self.weight,
                labels_1d,
                self.backend,
            )
        _queue_common_eval_metric(
            hidden_2d,
            self.weight,
            labels_1d,
            self.backend.ignore_index,
            loss,
        )
        return loss


def _get_raw_model(obj):
    max_depth = 20
    depth = 0
    while depth < max_depth:
        if hasattr(obj, "_fsdp_wrapped_module"):
            obj = obj._fsdp_wrapped_module
        elif hasattr(obj, "module"):
            obj = obj.module
        elif hasattr(obj, "_orig_mod"):
            obj = obj._orig_mod
        else:
            return obj
        depth += 1
    return obj


def _forward_with_internal_loss(self, tokens: torch.Tensor, start_pos: int = 0, labels: torch.Tensor = None):
    raw_model = _get_raw_model(self)
    tok_embeddings = getattr(raw_model, "tok_embeddings", None)
    layers = getattr(raw_model, "layers", None)
    norm = getattr(raw_model, "norm", None)
    output = getattr(raw_model, "output", None)
    if tok_embeddings is None or layers is None or norm is None or output is None:
        raise AttributeError(
            f"Could not find tok_embeddings/layers/norm/output on raw model: {type(raw_model)}"
        )

    freqs_cis = getattr(raw_model, "freqs_cis", None)
    mask = getattr(raw_model, "mask", None)
    if freqs_cis is not None:
        seqlen = tokens.shape[1]
        freqs_cis = freqs_cis.to(tokens.device)
        current_freqs = freqs_cis[start_pos : start_pos + seqlen]
    else:
        current_freqs = None

    h = tok_embeddings(tokens)
    layer_iter = layers.values() if isinstance(layers, (nn.ModuleDict, dict)) else layers
    for layer in layer_iter:
        h = layer(h, current_freqs, mask)
    if labels is not None:
        if (
            hasattr(output, "forward_from_pre_norm")
            and hasattr(output, "backend")
            and _v4_fused_x_producer_enabled_for_backend(output.backend)
        ):
            return output.forward_from_pre_norm(h, norm, labels=labels)
        h = norm(h)
        return output(h, labels=labels)
    h = norm(h)
    return output(h)


def apply_cce_backend_patch(model, job_config):
    raw_model = _get_raw_model(model)
    if not hasattr(raw_model, "output"):
        raise AttributeError(f"Could not find 'output' on model of type {type(raw_model)}")
    cfg = getattr(job_config, "fp4_cce", None)
    if cfg is not None and "NVTE_NVFP4_ENCODE_CENTRIC" not in os.environ:
        os.environ["NVTE_NVFP4_ENCODE_CENTRIC"] = "1" if getattr(cfg, "quant_mode", "enc") == "enc" else "0"
    backend = _build_backend(job_config)
    raw_model.output = TitanCCEHead(raw_model.output, backend)
    model.forward = types.MethodType(_forward_with_internal_loss, model)
    logger.info("Applied CCE backend patch: backend=%s", backend.name)
    return model
