"""
TK GEMM helper — wraps ThunderKittens NVFP4 GEMM for the training pipeline.

TK computes: D(M,N) = A(M,K) @ B(N,K)^T

Data source mapping for each GEMM type:
  | GEMM    | Math          | TK A source      | TK B source       |
  |---------|---------------|------------------|-------------------|
  | forward | y = x @ W^T   | x._tk_row (M,K)  | W._tk_row (N,K)   |
  | dgrad   | dx = dY @ W   | dY._tk_row(M,N)  | W._tk_col (K,N)   |
  | wgrad   | dW = dY^T @ x | dY._tk_col(N,M)  | x._tk_col (K,M)   |

All inputs are _TKQuantized objects with pre-cached _tk_row / _tk_col tuples.
"""

import importlib.machinery
import logging
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
import torch

logger = logging.getLogger(__name__)

_LEGACY_FP4_MATMUL_ROOT = "/opt/mfu/EXTERNAL_PATH"

_backend_trace_once: set[str] = set()
_last_qkv_backward_debug_payload = None


def _trace_backend_choice(key: str, value: str) -> None:
    """Emit a single backend-choice log line per process when tracing is enabled."""
    if os.environ.get('USE_TK_LOCALCTA_BACKEND_TRACE', '0') != '1':
        return
    token = f"{key}={value}"
    if token in _backend_trace_once:
        return
    _backend_trace_once.add(token)
    logger.info("[TK BACKEND] %s=%s", key, value)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _packed_fp4_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize packed FP4 through bytes without invoking FP4 ``copy_``.

    PyTorch does not implement ``copy_`` for ``float4_e2m1fn_x2``.  Calling
    ``contiguous()`` on a noncontiguous packed tensor therefore fails even
    though the dtype occupies exactly one byte per packed pair.  Copying the
    uint8 view and reinterpreting it is bit-exact and preserves the tensor's
    logical packed shape.
    """
    if tensor.dtype != torch.float4_e2m1fn_x2 or tensor.element_size() != 1:
        raise RuntimeError(
            "packed FP4 materialization requires float4_e2m1fn_x2 bytes"
        )
    result = (
        tensor.view(torch.uint8)
        .contiguous()
        .view(torch.float4_e2m1fn_x2)
    )
    if result.shape != tensor.shape or not result.is_contiguous():
        raise RuntimeError("packed FP4 byte materialization contract failed")
    return result


def _narrow_packed_fp4_contiguous(
    tensor: torch.Tensor,
    dim: int,
    start: int,
    length: int,
) -> torch.Tensor:
    """Return one bit-exact contiguous packed-FP4 narrow."""
    if tensor.dtype != torch.float4_e2m1fn_x2 or tensor.element_size() != 1:
        raise RuntimeError(
            "packed FP4 contiguous slicing requires float4_e2m1fn_x2 bytes"
        )
    result = (
        tensor.view(torch.uint8)
        .narrow(dim, start, length)
        .contiguous()
        .view(torch.float4_e2m1fn_x2)
    )
    expected_shape = list(tensor.shape)
    expected_shape[dim] = length
    if tuple(result.shape) != tuple(expected_shape) or not result.is_contiguous():
        raise RuntimeError("packed FP4 byte-copy slice contract failed")
    return result


def _regular_tk_nvfp4_rht_needs_nopdl() -> bool:
    """Route regular-TK RHT runs away from PDL GEMMs.

    The row-RHT + grad-SR trainer path can trip CUDA launch failures with PDL
    GEMMs under normal async launch ordering. No-PDL keeps the same fast path
    stable without forcing global CUDA_LAUNCH_BLOCKING or conn1.
    """
    if use_tk_localcta():
        return False
    return (
        _env_flag("NVFP4_USE_RHT", False)
        or _env_flag("NVFP4_RHT_ACTIVATION", False)
        or _env_flag("NVFP4_RHT_GRAD", False)
        or _env_flag("NVFP4_RHT_WEIGHT", False)
    )


def use_nvfp4_mxfp4_live_path() -> bool:
    """Use the NVFP4 GEMM route that mirrors MXFP4 live-path defaults."""
    return (
        _env_flag('USE_NVFP4_MXFP4_LIVE_PATH', False)
        or _env_flag('NVFP4_MIMIC_MXFP4_LIVE_PATH', False)
    )


def use_tk_qkv_backward_capture_debug() -> bool:
    return os.environ.get('USE_TK_QKV_BWD_CAPTURE_DEBUG', '0') == '1'


def use_tk_serial_rmsnorm_backward_debug() -> bool:
    """Debug-only: disable overlapped RMSNorm backward side-stream launches."""
    return os.environ.get('USE_TK_SERIAL_RMSNORM_BWD', '0') == '1'


def use_tk_rmsnorm_bwd_single_out() -> bool:
    """Use the single-kernel preallocated RMSNorm backward output path.

    The native single-output helper accumulates ``dgamma`` with one FP32
    ``atomicAdd`` per row.  CTA arrival order is not deterministic, so two
    otherwise identical localCTA-v4 resumes can update RMSNorm weights
    differently.  Keep the helper available to other backends, but reject it
    for localCTA v4 rather than silently defeating exact-resume guarantees.
    """
    enabled = _env_flag('USE_TK_RMSNORM_BWD_SINGLE_OUT', False)
    if (
        enabled
        and use_tk_localcta()
        and get_tk_localcta_variant() == 'v4'
    ):
        raise RuntimeError(
            "USE_TK_RMSNORM_BWD_SINGLE_OUT=1 is unsafe for localCTA v4: "
            "the native helper uses schedule-dependent atomic dgamma "
            "accumulation. Set it to 0 to use the fixed-order tiled dgamma "
            "reduction."
        )
    return enabled


def _clear_last_qkv_backward_debug_payload() -> None:
    global _last_qkv_backward_debug_payload
    _last_qkv_backward_debug_payload = None


def _set_last_qkv_backward_debug_payload(payload: dict) -> None:
    global _last_qkv_backward_debug_payload
    _last_qkv_backward_debug_payload = payload


def _get_last_qkv_backward_debug_payload(clear: bool = False):
    payload = _last_qkv_backward_debug_payload
    if clear:
        _clear_last_qkv_backward_debug_payload()
    return payload


def _const_chunk_grid(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    return torch.ones((rows // 128, cols // 128), device=device, dtype=torch.float32)


def _empty_chunk_sg(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.float32)


def _chunk_sg_or_empty(obj, rowwise: bool, ref_fp4: torch.Tensor) -> torch.Tensor:
    attr = '_tk_row_chunk_sg' if rowwise else '_tk_col_chunk_sg'
    chunk = getattr(obj, attr, None)
    if torch.is_tensor(chunk) and chunk.numel() > 0:
        return chunk
    return _empty_chunk_sg(ref_fp4.device)


def _has_virtual_rescale_chunk(obj, rowwise: bool) -> bool:
    attr = '_tk_row_chunk_sg' if rowwise else '_tk_col_chunk_sg'
    chunk = getattr(obj, attr, None)
    return torch.is_tensor(chunk) and chunk.numel() > 0

# ---------------------------------------------------------------------------
# Lazy TK import
# ---------------------------------------------------------------------------
_tk_module = None
_tk_import_attempted = False
_tk_import_error = None
_tk_backend_info = {}
_tk_plain_module = None
_tk_plain_import_attempted = False
_tk_plain_import_error = None
_tk_plain_quant_mod = None
_tk_plain_quant_import_attempted = False
_tk_plain_quant_import_error = None
_tk_localcta_direct_module = None
_tk_localcta_direct_import_attempted = False
_tk_localcta_direct_import_error = None
_tk_mixed_mx_localcta_quant_module = None
_tk_mixed_mx_localcta_quant_import_attempted = False
_tk_mixed_mx_localcta_quant_import_error = None
_localcta_v4_cpp_only_override: ContextVar[bool | None] = ContextVar(
    "_localcta_v4_cpp_only_override",
    default=None,
)


@contextmanager
def localcta_v4_cpp_only_scope(enabled: bool):
    token = _localcta_v4_cpp_only_override.set(enabled)
    try:
        yield
    finally:
        _localcta_v4_cpp_only_override.reset(token)


def set_localcta_v4_cpp_only_override(enabled: bool):
    return _localcta_v4_cpp_only_override.set(enabled)


def reset_localcta_v4_cpp_only_override(token) -> None:
    _localcta_v4_cpp_only_override.reset(token)
_col_quant_stream = None
_rmsnorm_bwd_stream = None
_wgrad_stream = None
_rmsnorm_bwd_cache = {}
_debug_qkv_capture_count = 0


def _use_tk_stage_trace() -> bool:
    return os.environ.get("USE_TK_STAGE_TRACE", "0") == "1"


_tk_stage_trace_starts: dict[tuple[str, str, str], float] = {}


def _tk_stage_trace(stage: str, event: str, name: str | None) -> None:
    if not _use_tk_stage_trace():
        return
    active_step = os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip()
    step_filter = os.environ.get("TK_STAGE_TRACE_STEP", "").strip()
    if step_filter and active_step != step_filter:
        return
    stage_filter = os.environ.get("TK_STAGE_TRACE_STAGE_FILTER", "").strip()
    if stage_filter and stage_filter not in stage:
        return
    event_filter = os.environ.get("TK_STAGE_TRACE_EVENT_FILTER", "").strip()
    if event_filter and event_filter not in event:
        return
    label = name or stage
    name_filter = os.environ.get("TK_STAGE_TRACE_FILTER", "").strip()
    if name_filter and name_filter not in label:
        return
    if os.environ.get("USE_TK_STAGE_TRACE_SYNC", "0") == "1" and torch.cuda.is_available():
        torch.cuda.synchronize()
    prefix = f"[TK TRACE step={active_step}]" if active_step else "[TK TRACE]"
    timer_key = None
    if event == "start":
        timer_key = (active_step, stage, label)
        _tk_stage_trace_starts[timer_key] = time.perf_counter()
    elif event.endswith("_start"):
        timer_key = (active_step, stage, f"{label}:{event[:-6]}")
        _tk_stage_trace_starts[timer_key] = time.perf_counter()

    elapsed = None
    if event == "end":
        timer_key = (active_step, stage, label)
        start = _tk_stage_trace_starts.pop(timer_key, None)
        if start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0
    elif event.endswith("_done"):
        timer_key = (active_step, stage, f"{label}:{event[:-5]}")
        start = _tk_stage_trace_starts.pop(timer_key, None)
        if start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0

    suffix = f" elapsed_ms={elapsed:.3f}" if elapsed is not None else ""
    print(f"{prefix} {stage} {event} {label}{suffix}", file=sys.stderr, flush=True)


def _fp4_matmul_root() -> str:
    root = os.environ.get("FP4_MATMUL_ROOT")
    if root:
        return os.path.abspath(root)
    base_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
        )
    )
    candidates = [
        os.path.join(base_dir, "fp4_matmul"),
        os.path.join(base_dir, "fp4_matmul-54-debug"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _prepend_import_paths_in_priority_order(paths) -> None:
    """Put existing module roots on sys.path without reversing priority."""
    ordered = []
    for path in paths:
        normalized = os.path.abspath(path)
        if os.path.isdir(normalized) and normalized not in ordered:
            ordered.append(normalized)
    for path in ordered:
        while path in sys.path:
            sys.path.remove(path)
    for path in reversed(ordered):
        sys.path.insert(0, path)


def _debug_qkv_capture_path() -> str:
    return os.environ.get("USE_TK_DEBUG_QKV_CAPTURE_JSONL", "")


def _tensor_debug_stats(t: torch.Tensor | None) -> dict | None:
    if t is None:
        return None
    if not torch.is_tensor(t):
        return {"type": str(type(t))}
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    if t.numel() == 0:
        out["numel"] = 0
        return out
    if t.dtype == torch.float4_e2m1fn_x2:
        u8 = t.view(torch.uint8)
        out.update({
            "numel": int(t.numel()),
            "byte_zero_fraction": float((u8 == 0).float().mean().item()),
            "byte_nonzero_fraction": float((u8 != 0).float().mean().item()),
        })
        return out
    x = t.detach()
    if not x.is_floating_point() or x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = x.to(torch.float32)
    finite = torch.isfinite(x)
    x_abs = x.abs()
    out.update({
        "numel": int(x.numel()),
        "finite_fraction": float(finite.float().mean().item()),
        "zero_fraction": float((x == 0).float().mean().item()),
        "rms": float(torch.sqrt((x * x).mean()).item()),
        "mean_abs": float(x_abs.mean().item()),
        "max_abs": float(x_abs.max().item()),
    })
    return out


def _append_qkv_capture(payload: dict) -> None:
    global _debug_qkv_capture_count
    path = _debug_qkv_capture_path()
    if not path:
        return
    record = dict(payload)
    record["capture_index"] = _debug_qkv_capture_count
    _debug_qkv_capture_count += 1
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _artifact_info(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "sha256": digest.hexdigest(),
    }


def get_tk_backend_info() -> dict:
    return dict(_tk_backend_info)


def reset_tk_runtime_caches() -> None:
    """Clear cached TK runtime modules so env-driven variant switches take effect."""
    global _tk_module, _tk_import_attempted, _tk_import_error, _tk_backend_info
    global _tk_plain_module, _tk_plain_import_attempted, _tk_plain_import_error
    global _tk_plain_quant_mod, _tk_plain_quant_import_attempted, _tk_plain_quant_import_error
    global _tk_localcta_direct_module, _tk_localcta_direct_import_attempted
    global _tk_localcta_direct_import_error, _tk_quant_mod_cache
    _tk_module = None
    _tk_import_attempted = False
    _tk_import_error = None
    _tk_backend_info = {}
    _tk_plain_module = None
    _tk_plain_import_attempted = False
    _tk_plain_import_error = None
    _tk_plain_quant_mod = None
    _tk_plain_quant_import_attempted = False
    _tk_plain_quant_import_error = None
    _tk_localcta_direct_module = None
    _tk_localcta_direct_import_attempted = False
    _tk_localcta_direct_import_error = None
    _tk_quant_mod_cache = None


def clear_tk_step_caches() -> None:
    """Release step-scoped TK tensor caches after an optimizer update.

    The restored isolated localCTA/TK paths intentionally keep several large
    payload caches alive within a step to avoid repeated packing work. Those
    tensors are keyed by data pointers and become stale as soon as weights are
    updated. In full training, keeping them across optimizer steps can retain
    tens of GiB of dead tensors.
    """
    for cache in (
        _dgrad_buf_cache,
        _weight_split_cache,
        _wgrad_buf_cache,
        _dgrad_sum_cache,
        _fused_bwd_cache,
        _wgrad_sg_idx_cache,
        _wgrad_direct_buf_cache,
        _grouped_wgrad_cat_cache,
        _split_wgrad_cache,
    ):
        cache.clear()


def use_tk_localcta() -> bool:
    """Check if localCTA quant/GEMM path is enabled."""
    return os.environ.get('USE_TK_LOCALCTA', '0') == '1'


def use_tk_localcta_direct_contract() -> bool:
    """Use the debug direct localCTA quant/GEMM contract.

    The production localCTA target is the fast v1 path. Opt into the direct
    contract explicitly when a local experiment needs it; do not silently route
    normal localCTA runs onto the slower surrogate path.
    """
    return use_tk_localcta() and os.environ.get('USE_TK_LOCALCTA_DIRECT_CONTRACT', '0') == '1'


def _tk_localcta_forward_min_m() -> int:
    value = os.environ.get('USE_TK_LOCALCTA_FORWARD_MIN_M')
    if value is None or value == '':
        return 256
    try:
        return max(0, int(value))
    except ValueError:
        return 256


def _use_tk_localcta_for_m(m: int) -> bool:
    return use_tk_localcta() and m >= _tk_localcta_forward_min_m()


def use_tk_localcta_fused() -> bool:
    """Check if localCTA fused quant helpers are enabled.

    USE_TK_LOCALCTA_FUSED is the primary flag. The older
    USE_TK_LOCALCTA_FUSED_SPLIT remains as a deprecated alias for now.
    """
    value = os.environ.get('USE_TK_LOCALCTA_FUSED')
    if value is not None:
        return value == '1'
    return os.environ.get('USE_TK_LOCALCTA_FUSED_SPLIT', '0') == '1'


def use_tk_localcta_v3() -> bool:
    """Check if the experimental localCTA v3 quant module should be loaded."""
    return os.environ.get('USE_TK_LOCALCTA_V3', '0') == '1'


def get_tk_localcta_variant() -> str:
    """Resolve the internal localCTA quant variant to load."""
    variant = os.environ.get('USE_TK_LOCALCTA_VARIANT')
    if variant is not None:
        variant = variant.strip().lower()
        if variant not in {'v1', 'v2', 'v3', 'v4'}:
            raise ValueError(
                "USE_TK_LOCALCTA_VARIANT must be one of {'v1', 'v2', 'v3', 'v4'}, "
                f"got {variant!r}"
            )
        if variant == 'v2':
            logger.warning(
                "USE_TK_LOCALCTA_VARIANT=v2 is deprecated and excluded from the active "
                "matrix/debug path; use v1 unless you are explicitly doing v2 archaeology."
            )
        return variant
    if use_tk_localcta_v3():
        return 'v3'
    return 'v1'


def _extension_candidate_names(module_name: str) -> list[str]:
    return [
        f"{module_name}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ]


def _find_extension_in_dirs(module_name: str, directories: list[str]) -> str | None:
    """Find a native extension without assuming the host architecture suffix."""
    candidate_names = _extension_candidate_names(module_name)
    # Retain the historical extensionless lookup for runtime trees that provide
    # an explicit compatibility symlink.
    candidate_names.append(module_name)
    for directory in directories:
        for candidate_name in candidate_names:
            candidate = os.path.join(directory, candidate_name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _localcta_gemm_variant_spec() -> tuple[str, str]:
    variant = get_tk_localcta_variant()
    if variant in {'v3', 'v4'}:
        return 'localCTA_epilogue_v3', '_C_nv_localcta_gemm_v3'
    return 'localCTA_epilogue', '_C_nv_localcta_gemm'


def _apply_localcta_v3_perf_defaults() -> None:
    if get_tk_localcta_variant() not in {'v3', 'v4'}:
        return
    os.environ.setdefault('USE_TK_LOCALCTA_V3_MULTIINPUT_QUANT', 'splitfinal')
    os.environ.setdefault('USE_TK_LOCALCTA_V3_DEFER_COL_DGRAD', '1')
    os.environ.setdefault('USE_TK_LOCALCTA_V3_SPLIT2_ONEPASS', '1')
    os.environ.setdefault('USE_TK_LOCALCTA_V3_SPLIT3_BATCHED_ACCUM', '1')
    os.environ.setdefault('USE_TK_LOCALCTA_V3_SPLIT3_BATCHED_ACCUM_SMALLN', '1')
    os.environ.setdefault('USE_TK_LOCALCTA_V4_SPLIT3_ROPE_SCAN_THREADS', '128')


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == '':
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == '':
        return None
    return value != '0'


def _localcta_global_scale_num() -> float | None:
    value = os.environ.get('USE_TK_LOCALCTA_SCALE_NUM')
    if value is None or value == '':
        return None
    try:
        scale_num = float(value)
    except ValueError as exc:
        raise ValueError(
            f'USE_TK_LOCALCTA_SCALE_NUM must be a finite positive number, got {value!r}'
        ) from exc
    if not math.isfinite(scale_num) or scale_num <= 0.0:
        raise ValueError(
            f'USE_TK_LOCALCTA_SCALE_NUM must be a finite positive number, got {value!r}'
        )
    return scale_num


def _v5_ffn_scale_target() -> float | None:
    value = os.environ.get('USE_TK_V5_FFN_SCALE_TARGET')
    if value is None or value == '':
        return None
    try:
        target = float(value)
    except ValueError as exc:
        raise ValueError(
            f'USE_TK_V5_FFN_SCALE_TARGET must be finite and in (0, 512], got {value!r}'
        ) from exc
    if not math.isfinite(target) or not 0.0 < target <= 512.0:
        raise ValueError(
            f'USE_TK_V5_FFN_SCALE_TARGET must be finite and in (0, 512], got {value!r}'
        )
    return target


class _V5FFNQuantAdapter:
    """Override generic quantization only while executing an FFN."""

    def __init__(self, mod, target: float) -> None:
        self._mod = mod
        self._target = target

    def __getattr__(self, name):
        return getattr(self._mod, name)

    def tk_quantize_for_gemm(self, input, return_transpose=True, encode_centric=True):
        return self._mod.tk_quantize_for_gemm_opt(
            input,
            return_transpose,
            encode_centric,
            False,
            False,
            'none',
            False,
            42,
            0,
            self._target,
        )


_v5_ffn_quant_scope_active = ContextVar('v5_ffn_quant_scope_active', default=False)
_v5_ffn_quant_adapter_cache = {}


@contextmanager
def v5_ffn_quant_scope():
    token = _v5_ffn_quant_scope_active.set(True)
    try:
        yield
    finally:
        _v5_ffn_quant_scope_active.reset(token)


def _maybe_wrap_v5_ffn_quantizer(mod):
    """Return an FFN-calibrated view without mutating the shared extension."""
    if not _v5_ffn_quant_scope_active.get():
        return mod
    target = _v5_ffn_scale_target()
    if target is None or target == 448.0:
        return mod
    if getattr(mod, 'is_localcta', False):
        return mod
    if not hasattr(mod, 'tk_quantize_for_gemm_opt'):
        raise RuntimeError('native v5 quantizer does not expose scale-target control')
    key = (id(mod), target)
    adapter = _v5_ffn_quant_adapter_cache.get(key)
    if adapter is None:
        adapter = _V5FFNQuantAdapter(mod, target)
        _v5_ffn_quant_adapter_cache[key] = adapter
    return adapter


def _maybe_apply_localcta_quant_tuning(mod) -> None:
    variant = get_tk_localcta_variant()
    if variant not in ('v3', 'v4'):
        return

    prefix = 'USE_TK_LOCALCTA_V3' if variant == 'v3' else 'USE_TK_LOCALCTA_V4'

    global_scale_num = _localcta_global_scale_num()
    if global_scale_num is not None:
        setter = getattr(mod, 'tk_localcta_set_global_scale_num', None)
        if setter is None:
            raise RuntimeError(
                f'localCTA {variant} does not expose global-scale control'
            )
        setter(global_scale_num)

    threads_2cta = _env_int(f'{prefix}_2CTA_PREPARED_THREADS')
    pipe_2cta = _env_int(f'{prefix}_2CTA_PREPARED_PIPE_DEPTH')
    shared_amax_2cta = _env_bool(f'{prefix}_2CTA_PREPARED_SHARED_AMAX')
    if (
        hasattr(mod, 'tk_localcta_set_2cta_prepared_tuning')
        and threads_2cta is not None
        and pipe_2cta is not None
        and shared_amax_2cta is not None
    ):
        mod.tk_localcta_set_2cta_prepared_tuning(
            threads_2cta, pipe_2cta, shared_amax_2cta
        )

    threads_1cta = _env_int(f'{prefix}_1CTA_PREPARED_THREADS')
    pipe_1cta = _env_int(f'{prefix}_1CTA_PREPARED_PIPE_DEPTH')
    if (
        hasattr(mod, 'tk_localcta_set_1cta_prepared_tuning')
        and threads_1cta is not None
        and pipe_1cta is not None
    ):
        mod.tk_localcta_set_1cta_prepared_tuning(
            threads_1cta, pipe_1cta
        )

    split2_threads_2cta = _env_int(f'{prefix}_SPLIT2_2CTA_PREPARED_THREADS')
    split2_pipe_2cta = _env_int(f'{prefix}_SPLIT2_2CTA_PREPARED_PIPE_DEPTH')
    split2_shared_amax_2cta = _env_bool(f'{prefix}_SPLIT2_2CTA_PREPARED_SHARED_AMAX')
    if (
        hasattr(mod, 'tk_localcta_set_2cta_prepared_split2_tuning')
        and split2_threads_2cta is not None
        and split2_pipe_2cta is not None
        and split2_shared_amax_2cta is not None
    ):
        mod.tk_localcta_set_2cta_prepared_split2_tuning(
            split2_threads_2cta, split2_pipe_2cta, split2_shared_amax_2cta
        )


def use_tk_qkv_bf16_wgrad() -> bool:
    """Use a debug-only BF16 QKV wgrad path while keeping TK/localCTA dgrad fast.

    This remains available for attribution only. The production baseline should
    stay on the real TK QKV wgrad path now that the shared quant backend fix is
    in place.
    """
    return os.environ.get('USE_TK_QKV_BF16_WGRAD', '0') == '1'


def use_tk_qkv_bf16_dgrad() -> bool:
    """Use a diagnostic BF16 QKV dgrad path with raw BF16 weights.

    This is narrower than a full fallback: it exists to confirm whether the
    remaining attention NaNs come from the shared QKV dgrad path after the
    QKV wgrad fix is already in place.
    """
    return os.environ.get('USE_TK_QKV_BF16_DGRAD', '0') == '1'


def use_tk_qkv_bf16_underflow_rescue() -> bool:
    """Rescue localCTA QKV backward when prepared grad scales collapse."""
    value = os.environ.get('USE_TK_QKV_BF16_RESCUE_ON_UNDERFLOW')
    if value is not None:
        return value == '1'
    return False


def use_tk_qkv_bf16_rmsnorm_bwd() -> bool:
    """Use a BF16 reference RMSNorm backward for QKV debug isolation.

    This stays internal-only and is intended to answer whether the remaining
    step-2 NaN belongs to the shared RMSNorm-backward contract after QKV dgrad
    and wgrad are already forced onto safer paths.
    """
    return os.environ.get('USE_TK_QKV_BF16_RMSNORM_BWD', '0') == '1'


def use_tk_qkv_bwd_nopdl() -> bool:
    """Prefer no-PDL QKV backward GEMMs when conn1 exposes TK PDL races."""
    mode = os.environ.get('USE_TK_QKV_BWD_NOPDL', 'auto').strip().lower()
    if mode in ('1', 'true', 'yes', 'on'):
        return True
    if mode in ('0', 'false', 'no', 'off'):
        return False
    if _regular_tk_nvfp4_rht_needs_nopdl():
        return True
    return os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '').strip() == '1'


def use_tk_qkv_dgrad_nopdl() -> bool:
    """Use the no-PDL QKV dgrad GEMM without forcing QKV wgrad no-PDL."""
    mode = os.environ.get('USE_TK_QKV_DGRAD_NOPDL')
    if mode is not None:
        return mode.strip().lower() in ('1', 'true', 'yes', 'on')
    return use_tk_qkv_bwd_nopdl()


def use_tk_qkv_wgrad_nopdl() -> bool:
    """Use the no-PDL QKV wgrad GEMM without forcing QKV dgrad no-PDL."""
    mode = os.environ.get('USE_TK_QKV_WGRAD_NOPDL')
    if mode is not None:
        return mode.strip().lower() in ('1', 'true', 'yes', 'on')
    return use_tk_qkv_bwd_nopdl()


def get_tk_qkv_wgrad_nopdl_config() -> int | None:
    """Optional config selector for the regular-TK QKV grouped wgrad no-PDL GEMM."""
    value = os.environ.get('USE_TK_QKV_WGRAD_NOPDL_CONFIG')
    if value is None or value == '':
        value = os.environ.get('USE_TK_QKV_WGRAD_NOPDL_CONFIG_ID')
    if value is None or value == '':
        return None
    config_id = int(value)
    if config_id < 0:
        return None
    return config_id


def use_tk_qkv_overlap_wgrad_nopdl() -> bool:
    """Launch regular-TK QKV no-PDL wgrad early and overlap it with dgrad/RMS."""
    mode = os.environ.get('USE_TK_QKV_OVERLAP_WGRAD_NOPDL')
    if mode is not None:
        return mode.strip().lower() in ('1', 'true', 'yes', 'on')
    # The side-stream no-PDL wgrad overlap can hang the regular-TK NVFP4
    # RHT/SR trainer path. Keep the stable fused sum+RMS path as the default
    # QKV fusion and require explicit opt-in for this overlap experiment.
    return False


def use_tk_qkv_overlap_rms_wgrad() -> bool:
    """Overlap regular-TK QKV RMSNorm backward with QKV wgrad after dgrad."""
    mode = os.environ.get('USE_TK_QKV_OVERLAP_RMS_WGRAD')
    if mode is not None:
        return mode.strip().lower() in ('1', 'true', 'yes', 'on')
    # Concurrent QKV RMS/wgrad occasionally corrupts the CUDA context at the
    # 8B training shape. It provides no measurable end-to-end gain there, so
    # keep the serial route as the production default and require opt-in.
    return False


def use_tk_qkv_plain_batched_accum_dgrad() -> bool:
    """Use the grouped in-kernel accumulation dgrad consumer for plain TK QKV."""
    mode = os.environ.get('USE_TK_QKV_PLAIN_BATCHED_ACCUM_DGRAD')
    if mode is not None:
        return mode.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_qkv_fused_sum_rms() -> bool:
    """Fuse regular-TK QKV split dgrad summation into RMSNorm backward."""
    return _env_flag('USE_TK_QKV_FUSED_SUM_RMS', False)


def use_tk_qkv_cached_return_transpose() -> bool:
    """Use the TK tiled BF16 transpose into per-layer cached QKV wgrad storage."""
    return _env_flag('USE_TK_QKV_CACHED_RETURN_TRANSPOSE', True)


def use_tk_ffn_cached_return_transpose() -> bool:
    """Use the TK tiled BF16 transpose into per-layer cached FFN wgrad storage."""
    return _env_flag('USE_TK_FFN_CACHED_RETURN_TRANSPOSE', True)


def use_tk_qkv_native_split3_quant() -> bool:
    """Use regular-TK native split3 grouped dim1 quant to avoid BF16 QKV concat."""
    return _env_flag('USE_TK_QKV_NATIVE_SPLIT3_QUANT', True)


def use_tk_ffn_plain_batched_accum_dgrad() -> bool:
    """Use the grouped in-kernel accumulation dgrad consumer for plain TK FFN."""
    mode = os.environ.get('USE_TK_FFN_PLAIN_BATCHED_ACCUM_DGRAD')
    if mode is not None:
        return mode.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_gemm_nopdl() -> bool:
    """Prefer no-PDL regular TK GEMMs under conn1 unless explicitly disabled."""
    mode = os.environ.get('USE_TK_GEMM_NOPDL', 'auto').strip().lower()
    if mode in ('1', 'true', 'yes', 'on'):
        return True
    if mode in ('0', 'false', 'no', 'off'):
        return False
    if _regular_tk_nvfp4_rht_needs_nopdl():
        return True
    return os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '').strip() == '1'


def use_tk_wo_dgrad_nopdl() -> bool:
    """Disable PDL only for the WO dgrad consumed by SDPA backward.

    This isolates the cross-library producer/consumer boundary without
    changing the QKV, FFN, or WO-wgrad launch contracts.
    """
    return _env_flag('USE_TK_WO_DGRAD_NOPDL', False)


_TK_V5_PDL_PRODUCTION_GEMM_CONFIGS = {
    # Llama 8B, batch=4, sequence=8192.
    (32768, 14336, 4096): 12,
    (4096, 14336, 32768): 28,
    # Nemotron-H 8B, batch=3, sequence=8192.
    (4096, 4096, 24576): 12,
    (24576, 21504, 4096): 12,
    (24576, 18688, 4096): 12,
    (24576, 4096, 18688): 12,
    (24576, 4096, 8192): 28,
    (24576, 8192, 4096): 27,
}
_TK_V5_PDL_PRODUCTION_GEMM_CONFIG_ENVS = {
    shape: f'USE_TK_GEMM_CONFIG_M{shape[0]}_N{shape[1]}_K{shape[2]}'
    for shape in _TK_V5_PDL_PRODUCTION_GEMM_CONFIGS
}


def _read_tk_gemm_config_env(name: str) -> tuple[bool, int | None]:
    value = os.environ.get(name)
    if value is None or value == '':
        return False, None
    config_id = int(value)
    return True, config_id if config_id >= 0 else None


def get_tk_gemm_config(
    shape: tuple[int, int, int] | None = None,
    *,
    use_production_default: bool = False,
) -> int | None:
    """Resolve a regular-TK single-GEMM config with explicit overrides first."""
    if shape is not None:
        exact_env = _TK_V5_PDL_PRODUCTION_GEMM_CONFIG_ENVS.get(shape)
        if exact_env is not None:
            is_set, config_id = _read_tk_gemm_config_env(exact_env)
            if is_set:
                return config_id
    for name in ('USE_TK_GEMM_CONFIG', 'USE_TK_GEMM_CONFIG_ID'):
        is_set, config_id = _read_tk_gemm_config_env(name)
        if is_set:
            return config_id
    if shape is not None and use_production_default:
        return _TK_V5_PDL_PRODUCTION_GEMM_CONFIGS.get(shape)
    return None


def use_tk_qkv_debug_sync_check() -> bool:
    """Force CUDA sync checkpoints around TK QKV backward stages for diagnosis."""
    return os.environ.get('USE_TK_QKV_DEBUG_SYNC_CHECK', '0') == '1'


def use_tk_qkv_disable_strided_dgrad() -> bool:
    """Debug-only: bypass the strided TK QKV dgrad kernel variant."""
    return os.environ.get('USE_TK_QKV_DISABLE_STRIDED_DGRAD', '0') == '1'


def use_tk_qkv_localcta_raw_dim1_quant() -> bool:
    """Debug-only: use raw localCTA split3 dim1 quant instead of prepared scales."""
    return os.environ.get('USE_TK_QKV_LOCALCTA_RAW_DIM1_QUANT', '0') == '1'


def use_tk_qkv_localcta_floor_prepared_scales() -> bool:
    """Debug-only: floor zeroed localCTA prepared QKV scales to the smallest FP8 value."""
    return os.environ.get('USE_TK_QKV_LOCALCTA_FLOOR_PREPARED_SCALES', '0') == '1'


def use_tk_qkv_localcta_scale_backoff() -> bool:
    """Debug-only: retry localCTA QKV quant with smaller scale numerators on underflow."""
    return os.environ.get('USE_TK_QKV_LOCALCTA_SCALE_BACKOFF', '0') == '1'


def use_tk_qkv_localcta_prepared_split3_separate() -> bool:
    """Debug-only: build localCTA split3 prepared QKV payloads from separate per-split quant calls."""
    return os.environ.get('USE_TK_QKV_LOCALCTA_PREPARED_SPLIT3_SEPARATE', '0') == '1'


def use_tk_qkv_localcta_consistent_nofold_operands() -> bool:
    """Debug-only: requantize QKV backward operands onto the no-fold prepared contract."""
    return os.environ.get('USE_TK_QKV_LOCALCTA_CONSISTENT_NOFOLD', '0') == '1'


def get_tk_qkv_localcta_scale_backoff_values() -> list[float]:
    raw = os.environ.get(
        'USE_TK_QKV_LOCALCTA_SCALE_BACKOFF_LIST',
        '1344,1200,1120,1024,896,746.5',
    )
    values: list[float] = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def get_tk_qkv_localcta_fixed_grad_boost() -> float:
    value = os.environ.get('USE_TK_QKV_LOCALCTA_FIXED_GRAD_BOOST')
    if value is None or value == '':
        return 1.0
    boost = float(value)
    if boost <= 0.0:
        raise ValueError(
            f"USE_TK_QKV_LOCALCTA_FIXED_GRAD_BOOST must be > 0, got {boost!r}"
        )
    return boost


def _localcta_adaptive_grad_boost_value(
    *tensors: torch.Tensor | None,
    target_amax: float = 1.0 / 32.0,
    max_boost: float = float(1 << 24),
) -> float:
    """Choose the smallest per-call boost that keeps tiny localCTA grads representable.

    The fast localCTA backward path folds global chunk scale into fp8 prepared
    microscales. On very small gradients that folded scale underflows to zero
    before GEMM. We scale the backward inputs up just enough to avoid that
    underflow, then divide the GEMM outputs back down after the kernel.
    """
    amax = 0.0
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        local_amax = float(torch.amax(torch.abs(tensor.detach().to(torch.float32))).item())
        if local_amax > amax:
            amax = local_amax
    if amax <= 0.0:
        return 1.0
    boost = target_amax / amax
    if boost <= 1.0:
        return 1.0
    return min(boost, max_boost)


def _qkv_localcta_scale_num_override() -> float | None:
    value = os.environ.get('USE_TK_QKV_LOCALCTA_SCALE_NUM')
    if value is None or value == '':
        return None
    return float(value)


def _set_localcta_qkv_scale_num(tkq) -> float | None:
    scale_num = _qkv_localcta_scale_num_override()
    if scale_num is None or not hasattr(tkq, 'tk_set_global_scale_num'):
        return None
    prev = tkq.tk_get_global_scale_num() if hasattr(tkq, 'tk_get_global_scale_num') else None
    tkq.tk_set_global_scale_num(scale_num)
    return prev


def _restore_localcta_qkv_scale_num(tkq, prev: float | None) -> None:
    scale_num = _qkv_localcta_scale_num_override()
    if scale_num is None or not hasattr(tkq, 'tk_set_global_scale_num'):
        return
    if prev is None:
        if hasattr(tkq, 'tk_reset_global_scale_num'):
            tkq.tk_reset_global_scale_num()
    else:
        tkq.tk_set_global_scale_num(prev)


def _run_localcta_qkv_quant_with_scale_override(tkq, fn, *args, **kwargs):
    prev = _set_localcta_qkv_scale_num(tkq)
    try:
        return fn(*args, **kwargs)
    finally:
        _restore_localcta_qkv_scale_num(tkq, prev)


def use_tk_localcta_prepared_sc_clamp_tiny() -> bool:
    """Debug-only: clamp prepared microscales to the smallest nonzero FP8 value."""
    return os.environ.get('USE_TK_LOCALCTA_PREPARED_SC_CLAMP_TINY', '0') == '1'


def get_tk_qkv_localcta_dgrad_backend_override() -> str:
    """Resolve a debug-only override for the localCTA QKV dgrad backend."""
    value = os.environ.get('USE_TK_QKV_LOCALCTA_DGRAD_BACKEND', '').strip().lower()
    if value in ('', 'auto'):
        return 'auto'
    allowed = {
        'strided_onepass',
        'strided_sum',
        'strided',
        'split3',
        'batched_accum',
        'direct_split',
    }
    if value not in allowed:
        raise ValueError(
            "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND must be one of "
            f"{sorted(allowed | {'auto'})}, got {value!r}"
        )
    return value


def use_tk_qkv_localcta_tk_prepared_activation() -> bool:
    """Use standalone TK payloads folded into the localCTA prepared contract."""
    return os.environ.get('USE_TK_QKV_LOCALCTA_TK_PREPARED_ACT', '0') == '1'


def use_tk_localcta_wo_prepared_split2_backward() -> bool:
    """Debug-only: use prepared split2 localCTA WO backward payloads."""
    value = os.environ.get('USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD')
    if value is not None:
        return value == '1'
    return False


def use_tk_localcta_wo_bf16_underflow_rescue() -> bool:
    """Rescue localCTA WO backward when split2-prepared dy scales underflow."""
    value = os.environ.get('USE_TK_LOCALCTA_WO_BF16_RESCUE_ON_UNDERFLOW')
    if value is not None:
        return value == '1'
    return True


def use_tk_localcta_v4_wo_raw_fast_outer() -> bool:
    """Use fast outer-SG GEMM consumers for raw localCTA-v4 WO split2 backward."""
    value = os.environ.get('USE_TK_LOCALCTA_V4_WO_RAW_FAST_OUTER')
    if value is not None:
        return value == '1'
    return False


def get_tk_localcta_wo_fixed_grad_boost() -> float:
    value = os.environ.get('USE_TK_LOCALCTA_WO_FIXED_GRAD_BOOST')
    if value is None or value == '':
        return 1.0
    boost = float(value)
    if boost <= 0.0:
        raise ValueError(
            f"USE_TK_LOCALCTA_WO_FIXED_GRAD_BOOST must be > 0, got {boost!r}"
        )
    return boost


def use_tk_localcta_v4_fast_qkv_dim1_concat() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_QKV_DIM1_CONCAT', '0') == '1'


def use_tk_localcta_v4_fast_qkv_grouped_wgrad() -> bool:
    value = os.environ.get('USE_TK_LOCALCTA_V4_FAST_QKV_GROUPED_WGRAD')
    if value is not None:
        return value == '1'
    return get_tk_localcta_variant() == 'v4' and not use_tk_localcta_v4_strict_path()


def use_tk_localcta_v4_fast_qkv_split_wgrad() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_QKV_SPLIT_WGRAD', '0') == '1'


def _tk_qkv_debug_sync_checkpoint(label: str) -> None:
    if not use_tk_qkv_debug_sync_check():
        return
    labels = os.environ.get('USE_TK_QKV_DEBUG_SYNC_LABELS', '').strip()
    if labels:
        allowed = {item.strip() for item in labels.split(',') if item.strip()}
        if label not in allowed:
            return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"TK QKV debug sync failed after {label}: {exc}") from exc


def _tk_qkv_debug_assert_finite(
    stage: str,
    debug_name: str | None,
    named_tensors,
) -> None:
    """Fail at the first non-finite QKV boundary without a matrix-sized mask."""
    if os.environ.get("USE_TK_DEBUG_QKV_FINITE", "0") != "1":
        return
    filtered = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    ]
    if not filtered:
        return

    # foreach_norm emits one scalar per tensor.  Unlike torch.isfinite(), it
    # does not allocate a boolean tensor as large as the 32768x4096 QKV dgrad.
    norms = torch._foreach_norm(
        [tensor.detach() for _, tensor in filtered], float("inf")
    )
    bad = []
    for (name, tensor), norm in zip(filtered, norms):
        max_abs = float(norm.item())
        if not math.isfinite(max_abs):
            bad.append(
                f"{name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"max_abs={max_abs}"
            )
    if bad:
        owner = debug_name or "<unknown>"
        raise RuntimeError(
            f"Non-finite TK QKV tensor at {owner}:{stage}: " + "; ".join(bad)
        )


def use_tk_ffn_split_wgrad_eager() -> bool:
    """Select the route-specific plain-TK FFN split wgrad policy.

    This bypasses nvfp4_batched_gemm for the 2-way FFN W1/W3 weight-gradient
    update and runs two independent nvfp4_gemm calls instead. The batched
    split-wgrad kernel can stall the regular-TK v5 trainer under sustained
    launch pressure, so regular v5 keeps the eager split route. Delayed v5
    benefits from the lower batched launch/service cost. Explicit selectors
    continue to override either route policy, and localCTA remains outside
    this regular-TK selector.
    """
    value = os.environ.get('USE_TK_FFN_SPLIT_WGRAD_EAGER')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    if use_tk_localcta():
        return False
    return os.environ.get('USE_TK_FFN_V5_DELAYED_DIRECT_SPLIT', '0') != '1'


def use_tk_ffn_direct_split_wgrad_layout() -> bool:
    """Emit regular-TK FFN split wgrad directly as (H, K), avoiding return transposes."""
    return _env_flag('USE_TK_FFN_DIRECT_SPLIT_WGRAD_LAYOUT', False)


def use_tk_localcta_v3_enable_prepared_split2() -> bool:
    """Expose prepared split2 FFN dgrad backends under localCTA v3.

    This is an experiment gate only. The default v3 path stays on the exact
    outerscale contract.
    """
    return os.environ.get('USE_TK_LOCALCTA_V3_ENABLE_PREPARED_SPLIT2', '0') == '1'


def use_tk_localcta_v4_sg_direct_consumers() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_SG_DIRECT_CONSUMERS', '0') == '1'


def use_tk_localcta_v4_gemm_virtual_rescale() -> bool:
    return _env_flag('USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE', False)


def use_tk_localcta_v4_final_sg_producer() -> bool:
    """Use scan-first v4 producers that quantize directly with final outer SG."""
    return _env_flag('USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER', True)


def use_tk_localcta_v4_atomic_final_sg_producer() -> bool:
    """Use the v4 producer whose final outer SG is reduced atomically."""
    return _env_flag('USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER', False)


def use_tk_localcta_v4_silu_atomic_final_sg_producer() -> bool:
    """Use the fused SwiGLU producer with atomic final outer-SG reduction."""
    return _env_flag('USE_TK_LOCALCTA_V4_SILU_ATOMIC_FINAL_SG_PRODUCER', True)


def use_tk_localcta_v4_ffn_residual_epilogue() -> bool:
    """Fuse the FFN W2 residual add into the localCTA v4 GEMM epilogue."""
    value = os.environ.get('USE_TK_LOCALCTA_V4_FFN_RESIDUAL_EPILOGUE')
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return use_tk_localcta() and get_tk_localcta_variant() == 'v4'


def use_tk_v5_ffn_residual_epilogue() -> bool:
    """Use the native regular-v5 W2 residual epilogue when requested."""
    if use_tk_localcta() or os.environ.get('MXFP4_BACKEND_VERSION') is not None:
        return False
    return _env_flag('USE_TK_V5_FFN_RESIDUAL_EPILOGUE', True)


def use_tk_v5_ffn_residual_epilogue_for_shape(M: int, K: int, H: int) -> bool:
    """Limit regular-v5 residual fusion to the measured Llama-8B W2 shape."""
    return (
        use_tk_v5_ffn_residual_epilogue()
        and int(M) == 32768
        and int(K) == 4096
        and int(H) == 14336
    )


def use_tk_ffn_rms_residual_bwd_for_shape(
    M: int,
    K: int,
    H: int,
    *,
    use_localcta: bool = False,
) -> bool:
    """Fold an aliased FFN residual gradient into RMS backward at the 8B shape."""
    return (
        _env_flag('USE_TK_FFN_RMS_RESIDUAL_BWD', True)
        and int(M) == 32768
        and int(K) == 4096
        and int(H) == 14336
        and not (
            use_localcta
            and (
                _env_flag('USE_TK_FFN_LOCALCTA_DELAYED_SPLIT', False)
                or _env_flag('USE_TK_FFN_H13_TILE_DELAYED_AMAX', False)
            )
        )
    )


def _tensor_autograd_alias_root(tensor: torch.Tensor) -> torch.Tensor:
    root = tensor
    while torch.is_tensor(getattr(root, '_base', None)):
        root = root._base
    return root


def _ffn_rms_residual_aliases_input(
    input_tensor: torch.Tensor,
    residual: torch.Tensor | None,
) -> bool:
    """Require identical layout, storage address, and differentiable view root."""
    return (
        torch.is_tensor(residual)
        and residual.dtype == input_tensor.dtype
        and residual.device == input_tensor.device
        and tuple(residual.shape) == tuple(input_tensor.shape)
        and tuple(residual.stride()) == tuple(input_tensor.stride())
        and residual.data_ptr() == input_tensor.data_ptr()
        and _tensor_autograd_alias_root(residual)
        is _tensor_autograd_alias_root(input_tensor)
    )


def use_tk_localcta_v4_cpp_only() -> bool:
    """Allow Python to orchestrate localCTA v4 kernels, but not do tensor math.

    In this mode the wrapper must not fold SG into scales, run fallback matmuls,
    or otherwise perform numeric work in Python. Wrapper-side view/reshape and
    argument packaging is still allowed.
    """
    override = _localcta_v4_cpp_only_override.get()
    if override is not None:
        return override
    return os.environ.get('USE_TK_LOCALCTA_V4_CPP_ONLY', '0') == '1'


def use_tk_localcta_v4_strict_path() -> bool:
    """Strict v4 C++ contract.

    This path is only valid for localCTA v4 and is intentionally fail-closed:
    wrappers may package tensors, but numeric work must flow through C++
    producers/consumers. The default v4 GEMM contract is outer SG epilogue
    scaling, not raw chunk-grid SG consumption.
    """
    return get_tk_localcta_variant() == 'v4' and use_tk_localcta_v4_cpp_only()


def use_tk_localcta_v4_fast_grouped_forward() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    value = os.environ.get('USE_TK_LOCALCTA_V4_FAST_GROUPED_FORWARD')
    if value is not None:
        return value == '1'
    return use_nvfp4_mxfp4_live_path()


def use_tk_localcta_v4_fast_grouped_wgrad() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    value = os.environ.get('USE_TK_LOCALCTA_V4_FAST_GROUPED_WGRAD')
    if value is not None:
        return value == '1'
    return use_nvfp4_mxfp4_live_path()


def use_tk_localcta_v4_direct_grouped_wgrad_layout() -> bool:
    value = os.environ.get('USE_TK_LOCALCTA_V4_DIRECT_GROUPED_WGRAD_LAYOUT')
    if value is not None:
        return value == '1'
    # The direct (N, K) grouped-wgrad output layout is not safe for the 8B
    # SwiGLU path yet: it can report the failure asynchronously during gradient
    # clipping. Keep it explicit-only instead of inheriting from broader v4
    # grouped-wgrad experiment flags.
    return False


def use_tk_localcta_v4_ffn_direct_grouped_wgrad_layout() -> bool:
    """Emit production FFN W13 gradients directly in parameter layout."""
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get(
        'USE_TK_LOCALCTA_V4_FFN_DIRECT_GROUPED_WGRAD_LAYOUT', '0'
    ) == '1'


def use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout() -> bool:
    if not use_tk_localcta() or get_tk_localcta_variant() != 'v4':
        return False
    value = os.environ.get('USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT')
    if value is not None:
        return value == '1'
    return False


def use_tk_localcta_v4_fast_single_dgrad() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_SINGLE_DGRAD', '0') == '1'


def use_tk_localcta_v4_fast_single_wgrad() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_SINGLE_WGRAD', '0') == '1'


def use_tk_localcta_v4_raw_backward_fallbacks(rows: int | None = None) -> bool:
    """Enable the older raw-direct localCTA v4 backward fallbacks.

    These paths were useful to keep v4 running while SG consumers were being
    debugged, but they are approximate and materially affect both numerics and
    MFU on the real standard-1B training path. Keep them available for narrow
    isolation runs, but enable them automatically only on small-M shapes where
    the SG-preserving consumers are still not as stable.
    """
    if use_tk_localcta_v4_cpp_only():
        return False
    value = os.environ.get('USE_TK_LOCALCTA_V4_RAW_BACKWARD_FALLBACKS')
    if value is not None:
        return value == '1'
    if rows is None:
        return False
    return rows < tk_localcta_v4_split2_large_m_threshold()


def tk_localcta_v4_split3_dgrad_sg_mode() -> str:
    if use_tk_localcta_v4_cpp_only():
        return 'a_b'
    return os.environ.get('USE_TK_LOCALCTA_V4_SPLIT3_DGRAD_SG_MODE', 'a_b').strip().lower()

def tk_localcta_v4_split2_large_m_threshold() -> int:
    value = os.environ.get('USE_TK_LOCALCTA_V4_SPLIT2_LARGE_M_THRESHOLD')
    if value is None:
        return 4096
    try:
        return int(value)
    except ValueError:
        return 4096


def tk_localcta_v4_split2_dgrad_sg_mode(rows: int | None = None) -> str:
    if use_tk_localcta_v4_cpp_only():
        return 'a_b'
    value = os.environ.get('USE_TK_LOCALCTA_V4_SPLIT2_DGRAD_SG_MODE')
    if value is not None:
        return value.strip().lower()
    if rows is not None and rows >= tk_localcta_v4_split2_large_m_threshold():
        return 'v3_outer'
    return 'afold_b'

def tk_localcta_v4_grouped_wgrad_sg_mode() -> str:
    if use_tk_localcta_v4_cpp_only():
        return 'a_b'
    return os.environ.get('USE_TK_LOCALCTA_V4_GROUPED_WGRAD_SG_MODE', 'a_b').strip().lower()


def tk_localcta_v4_split3_onepass_config_idx() -> int:
    value = os.environ.get('USE_TK_LOCALCTA_V4_SPLIT3_ONEPASS_CONFIG_IDX')
    if value is None:
        return -1
    try:
        return int(value)
    except ValueError:
        return -1


def use_tk_localcta_v4_fast_qkv_onepass_dgrad() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD', '0') == '1'


def use_tk_localcta_v4_fullcol_qkv_dgrad() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD', '0') == '1'


def use_tk_localcta_v4_fullcol_qkv_dgrad_direct_sg() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD_DIRECT_SG', '0') == '1'


def use_tk_localcta_v4_split3_fold_row_sg_in_producer() -> bool:
    if os.environ.get('USE_TK_LOCALCTA_V4_SPLIT3_FOLD_ROW_SG_IN_PRODUCER', '0') != '1':
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_SPLIT3_TWO_PHASE', '1') != '0'


def tk_localcta_v4_split2_b_sg_scale(rows: int | None = None, mode: str | None = None) -> float:
    value = os.environ.get('USE_TK_LOCALCTA_V4_SPLIT2_B_SG_SCALE')
    if value is not None:
        try:
            return float(value)
        except ValueError:
            return 0.724
    if mode == 'a1_bmean':
        return 0.05
    if rows is not None and rows >= tk_localcta_v4_split2_large_m_threshold():
        return 0.05
    return 0.724


def tk_localcta_v4_split_wgrad_a_sg_scale() -> float:
    value = os.environ.get('USE_TK_LOCALCTA_V4_SPLIT_WGRAD_A_SG_SCALE')
    if value is None:
        return 0.709
    try:
        return float(value)
    except ValueError:
        return 0.709


def tk_localcta_v4_w2_dgrad_b_sg_scale() -> float:
    value = os.environ.get('USE_TK_LOCALCTA_V4_W2_DGRAD_B_SG_SCALE')
    if value is None:
        return 0.731
    try:
        return float(value)
    except ValueError:
        return 0.731


def use_tk_localcta_v3_split2_onepass() -> bool:
    """Enable the exact SG-aware split2 one-pass FFN dgrad backend for v3."""
    return os.environ.get('USE_TK_LOCALCTA_V3_SPLIT2_ONEPASS', '1') == '1'


def tk_localcta_v3_split2_onepass_config_idx() -> int:
    value = os.environ.get('USE_TK_LOCALCTA_V3_SPLIT2_ONEPASS_CONFIG')
    if value is None:
        if use_tk_localcta_v4_strict_path():
            return 7
        return 3
    try:
        return int(value)
    except ValueError:
        if use_tk_localcta_v4_strict_path():
            return 7
        return 3


def _rmsnorm_backward_bf16_reference(d_normed, input_tensor, norm_weight, inv_rms):
    """BF16/FP32 RMSNorm backward reference for QKV debug experiments."""
    x = input_tensor.float()
    dy = d_normed.float()
    w = norm_weight.float().view(1, -1)
    inv = inv_rms.float().view(-1, 1)
    gw = dy * w
    dot = (gw * x).sum(dim=-1, keepdim=True)
    k = float(x.shape[-1])
    dx = gw * inv - x * (inv * inv * inv) * (dot / k)
    dgamma = (dy * x * inv).sum(dim=0)
    return dx.to(torch.bfloat16), dgamma


def _get_col_quant_stream():
    """Get (or create) a cached CUDA stream for col-only quant overlap."""
    global _col_quant_stream
    if _col_quant_stream is None:
        _col_quant_stream = torch.cuda.Stream()
    return _col_quant_stream


def _get_rmsnorm_bwd_stream():
    """Get (or create) a cached CUDA stream for RMSNorm backward overlap."""
    global _rmsnorm_bwd_stream
    if _rmsnorm_bwd_stream is None:
        _rmsnorm_bwd_stream = torch.cuda.Stream()
    return _rmsnorm_bwd_stream


def _get_wgrad_stream():
    """Get (or create) a cached CUDA stream for grouped wgrad overlap."""
    global _wgrad_stream
    if _wgrad_stream is None:
        _wgrad_stream = torch.cuda.Stream()
    return _wgrad_stream


def _record_tensors_on_stream(obj, stream) -> None:
    if obj is None:
        return
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj.record_stream(stream)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _record_tensors_on_stream(item, stream)


def _rmsnorm_bwd_stream_owner_key(tag: str, owner_key, caller_stream):
    """Scope reusable RMSNorm scratch by role and caller stream."""
    if not tag.endswith(("qkv", "ffn")):
        return owner_key
    logical_owner = tag if owner_key is None else _cache_owner_tag(owner_key)
    if os.environ.get('USE_TK_TRANSIENT_RMSNORM_RETURNS', '0') == '1':
        logical_owner = tag
    return logical_owner, int(caller_stream.cuda_stream)


def _get_rmsnorm_bwd_state(
    M: int,
    K: int,
    device: torch.device,
    *,
    tag: str = "default",
    owner_key=None,
):
    """Get cached eager buffers for overlapped RMSNorm backward.

    Attention and FFN can hit the same `(M, K, device)` in one backward pass,
    and independent callers can enter on different CUDA streams.  Only internal
    scratch and events are cached here.  Autograd-returned outputs are allocated
    separately for every launch so FSDP controls their lifetime.
    """
    owner_tag = tag if owner_key is None else _cache_owner_tag(owner_key)
    key = (owner_tag, M, K, device.index)
    state = _rmsnorm_bwd_cache.get(key)
    if state is None:
        row_tiles = (M + 255) // 256
        state = {
            'dgamma_partials': torch.empty(row_tiles, K, dtype=torch.float32, device=device),
            'input_bf16': torch.empty(M, K, dtype=torch.bfloat16, device=device),
            'norm_weight_bf16': torch.empty(K, dtype=torch.bfloat16, device=device),
            'ready_event': torch.cuda.Event(),
            'done_event': torch.cuda.Event(),
        }
        if os.environ.get('USE_TK_TRANSIENT_RMSNORM_RETURNS', '0') != '1':
            state.update({
                'grad_input': torch.empty(M, K, dtype=torch.bfloat16, device=device),
                'dgamma': torch.empty(K, dtype=torch.float32, device=device),
                'dgamma_bf16': torch.empty(K, dtype=torch.bfloat16, device=device),
            })
        _rmsnorm_bwd_cache[key] = state
    return state


def _prepare_rmsnorm_bwd_return_state(
    scratch: dict,
    M: int,
    K: int,
    device: torch.device,
):
    """Attach launch-owned outputs to reusable RMSNorm scratch."""
    if os.environ.get('USE_TK_TRANSIENT_RMSNORM_RETURNS', '0') != '1':
        return scratch
    state = dict(scratch)
    state.update({
        'grad_input': torch.empty(M, K, dtype=torch.bfloat16, device=device),
        'dgamma': torch.empty(K, dtype=torch.float32, device=device),
        'dgamma_bf16': torch.empty(K, dtype=torch.bfloat16, device=device),
    })
    return state


def _launch_native_sum3_rmsnorm_bwd_out_async(
    d_normed,
    input_tensor,
    norm_weight,
    inv_rms,
    d_sum,
    native_sum3_rmsnorm_bwd_out,
    ready_event=None,
    owner_key=None,
    *,
    tag: str = "default",
    force_current_stream: bool = False,
):
    """Launch the native TK sum3 + RMSNorm backward into cached outputs."""
    if not isinstance(d_normed, (tuple, list)) or len(d_normed) != 3:
        raise RuntimeError("native sum3 + RMSNorm backward requires exactly three inputs")
    d0, d1, d2 = d_normed
    M, K = d0.shape
    device = d0.device
    caller_stream = torch.cuda.current_stream(device)
    stream_owner_key = _rmsnorm_bwd_stream_owner_key(
        tag, owner_key, caller_stream
    )
    scratch = _get_rmsnorm_bwd_state(
        M, K, device, tag=tag, owner_key=stream_owner_key
    )
    state = _prepare_rmsnorm_bwd_return_state(scratch, M, K, device)
    use_bf16_dgamma_out = norm_weight.dtype == torch.bfloat16
    dgamma_out = state['dgamma_bf16'] if use_bf16_dgamma_out else state['dgamma']
    state['dgamma_out'] = dgamma_out

    rms_stream = (
        caller_stream if force_current_stream else _get_rmsnorm_bwd_stream()
    )
    if not force_current_stream:
        state['ready_event'].record(caller_stream)
        rms_stream.wait_event(state['ready_event'])

    with torch.cuda.stream(rms_stream):
        if input_tensor.dtype == torch.bfloat16 and input_tensor.is_contiguous():
            input_bf16 = input_tensor
        else:
            state['input_bf16'].copy_(input_tensor)
            input_bf16 = state['input_bf16']
        if norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous():
            norm_weight_bf16 = norm_weight
        else:
            state['norm_weight_bf16'].copy_(norm_weight)
            norm_weight_bf16 = state['norm_weight_bf16']
        _record_tensors_on_stream(
            (
                d0, d1, d2, input_bf16, norm_weight_bf16, inv_rms,
                d_sum, state['grad_input'], state['dgamma_partials'], dgamma_out,
            ),
            rms_stream,
        )
        native_sum3_rmsnorm_bwd_out(
            d0,
            d1,
            d2,
            input_bf16,
            norm_weight_bf16,
            inv_rms,
            d_sum,
            state['grad_input'],
            state['dgamma_partials'],
            dgamma_out,
        )
        state['done_event'].record(rms_stream)
    return state, rms_stream


def _launch_native_rmsnorm_residual_bwd_out_async(
    d_normed,
    input_tensor,
    norm_weight,
    inv_rms,
    residual_grad,
    ready_event=None,
    owner_key=None,
    *,
    tag: str = "default",
    force_current_stream: bool = False,
):
    """Launch native RMS backward with its BF16 residual add in the dx kernel."""
    native = _get_native_rmsnorm_bwd_residual_out()
    if native is None:
        raise RuntimeError(
            "USE_TK_FFN_RMS_RESIDUAL_BWD requires rmsnorm_bwd_residual_out"
        )
    if (
        d_normed.dtype != torch.bfloat16
        or not d_normed.is_contiguous()
        or residual_grad.dtype != torch.bfloat16
        or not residual_grad.is_contiguous()
    ):
        raise RuntimeError(
            "native FFN residual RMS backward requires contiguous BF16 gradients"
        )
    M, K = d_normed.shape
    device = d_normed.device
    caller_stream = torch.cuda.current_stream(device)
    stream_owner_key = _rmsnorm_bwd_stream_owner_key(
        tag, owner_key, caller_stream
    )
    scratch = _get_rmsnorm_bwd_state(
        M, K, device, tag=tag, owner_key=stream_owner_key
    )
    state = _prepare_rmsnorm_bwd_return_state(scratch, M, K, device)
    state['dgamma_out'] = state['dgamma']
    rms_stream = caller_stream if force_current_stream else _get_rmsnorm_bwd_stream()
    if not force_current_stream:
        if ready_event is not None:
            rms_stream.wait_event(ready_event)
        else:
            rms_stream.wait_stream(caller_stream)

    with torch.cuda.stream(rms_stream):
        if input_tensor.dtype == torch.bfloat16 and input_tensor.is_contiguous():
            input_bf16 = input_tensor
        else:
            state['input_bf16'].copy_(input_tensor)
            input_bf16 = state['input_bf16']
        if norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous():
            norm_weight_bf16 = norm_weight
        else:
            state['norm_weight_bf16'].copy_(norm_weight)
            norm_weight_bf16 = state['norm_weight_bf16']
        _record_tensors_on_stream(
            (
                d_normed, input_bf16, norm_weight_bf16, inv_rms,
                residual_grad, state['grad_input'],
                state['dgamma'],
            ),
            rms_stream,
        )
        native(
            d_normed,
            input_bf16,
            norm_weight_bf16,
            inv_rms,
            residual_grad,
            state['grad_input'],
            state['dgamma'],
        )
    return state, rms_stream


def _launch_rmsnorm_bwd_out_async(
    d_normed,
    input_tensor,
    norm_weight,
    inv_rms,
    te_fused,
    ready_event=None,
    owner_key=None,
    *,
    tag: str = "default",
    force_current_stream: bool = False,
    force_single_out: bool = False,
    force_fp32_dgamma: bool = False,
    force_norm_weight_copy: bool = False,
    residual_grad=None,
):
    """Launch fused RMSNorm backward into cached outputs on the selected stream."""
    if residual_grad is not None:
        if isinstance(d_normed, (tuple, list)):
            raise RuntimeError(
                "residual RMS backward requires one accumulated d_normed"
            )
        return _launch_native_rmsnorm_residual_bwd_out_async(
            d_normed,
            input_tensor,
            norm_weight,
            inv_rms,
            residual_grad,
            ready_event=ready_event,
            owner_key=owner_key,
            tag=tag,
            force_current_stream=force_current_stream,
        )
    sum3_input = isinstance(d_normed, (tuple, list))
    if sum3_input:
        d0, d1, d2 = d_normed
        M, K = d0.shape
        device = d0.device
    else:
        M, K = d_normed.shape
        device = d_normed.device
    caller_stream = torch.cuda.current_stream(device)
    stream_owner_key = _rmsnorm_bwd_stream_owner_key(
        tag, owner_key, caller_stream
    )
    scratch = _get_rmsnorm_bwd_state(
        M, K, device, tag=tag, owner_key=stream_owner_key
    )
    state = _prepare_rmsnorm_bwd_return_state(scratch, M, K, device)
    state['dgamma_out'] = state['dgamma']
    norm_weight_snapshot = None
    if force_norm_weight_copy:
        state['norm_weight_bf16'].copy_(norm_weight)
        norm_weight_snapshot = state['norm_weight_bf16']
    use_bf16_dgamma_out = (
        not force_fp32_dgamma
        and norm_weight.dtype == torch.bfloat16
        and hasattr(
            te_fused,
            'fused_rmsnorm_backward_sum3_dgamma_tiled_bf16_out'
            if sum3_input else 'fused_rmsnorm_backward_dgamma_tiled_bf16_out'
        )
    )
    use_single_out = force_single_out or use_tk_rmsnorm_bwd_single_out()
    if force_current_stream or use_tk_serial_rmsnorm_backward_debug():
        if input_tensor.dtype == torch.bfloat16 and input_tensor.is_contiguous():
            input_bf16 = input_tensor
        else:
            state['input_bf16'].copy_(input_tensor)
            input_bf16 = state['input_bf16']
        if norm_weight_snapshot is not None:
            norm_weight_bf16 = norm_weight_snapshot
        elif norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous():
            norm_weight_bf16 = norm_weight
        else:
            state['norm_weight_bf16'].copy_(norm_weight)
            norm_weight_bf16 = state['norm_weight_bf16']
        if (
            use_single_out
            and sum3_input
            and hasattr(te_fused, 'fused_rmsnorm_backward_sum3_out')
        ):
            te_fused.fused_rmsnorm_backward_sum3_out(
                d0, d1, d2, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
            if use_bf16_dgamma_out:
                state['dgamma_bf16'].copy_(state['dgamma'])
                state['dgamma_out'] = state['dgamma_bf16']
            else:
                state['dgamma_out'] = state['dgamma']
        elif (
            use_single_out
            and not sum3_input
            and hasattr(te_fused, 'fused_rmsnorm_backward_out')
        ):
            te_fused.fused_rmsnorm_backward_out(
                d_normed, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
            if use_bf16_dgamma_out:
                state['dgamma_bf16'].copy_(state['dgamma'])
                state['dgamma_out'] = state['dgamma_bf16']
            else:
                state['dgamma_out'] = state['dgamma']
        elif sum3_input and hasattr(te_fused, 'fused_rmsnorm_backward_sum3_dgamma_tiled_out'):
            dgamma_out = state['dgamma_bf16'] if use_bf16_dgamma_out else state['dgamma']
            state['dgamma_out'] = dgamma_out
            if use_bf16_dgamma_out:
                te_fused.fused_rmsnorm_backward_sum3_dgamma_tiled_bf16_out(
                    d0, d1, d2, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma_bf16']
                )
            else:
                te_fused.fused_rmsnorm_backward_sum3_dgamma_tiled_out(
                    d0, d1, d2, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma']
                )
            te_fused.fused_rmsnorm_backward_sum3_dx_only_out(
                d0, d1, d2, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input']
            )
        elif hasattr(te_fused, 'fused_rmsnorm_backward_dgamma_tiled_out'):
            dgamma_out = state['dgamma_bf16'] if use_bf16_dgamma_out else state['dgamma']
            state['dgamma_out'] = dgamma_out
            if use_bf16_dgamma_out:
                te_fused.fused_rmsnorm_backward_dgamma_tiled_bf16_out(
                    d_normed, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma_bf16']
                )
            else:
                te_fused.fused_rmsnorm_backward_dgamma_tiled_out(
                    d_normed, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma']
                )
            if hasattr(te_fused, 'fused_rmsnorm_backward_grad_input_out'):
                te_fused.fused_rmsnorm_backward_grad_input_out(
                    d_normed, input_bf16, norm_weight_bf16, inv_rms,
                    state['grad_input']
                )
            else:
                te_fused.fused_rmsnorm_backward_dx_only_out(
                    d_normed, input_bf16, norm_weight_bf16, inv_rms,
                    state['grad_input']
                )
        elif sum3_input:
            te_fused.fused_rmsnorm_backward_sum3_out(
                d0, d1, d2, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
        else:
            te_fused.fused_rmsnorm_backward_out(
                d_normed, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
        return state, caller_stream
    rms_stream = _get_rmsnorm_bwd_stream()
    if ready_event is not None:
        rms_stream.wait_event(ready_event)
    else:
        rms_stream.wait_stream(caller_stream)

    with torch.cuda.stream(rms_stream):
        if input_tensor.dtype == torch.bfloat16 and input_tensor.is_contiguous():
            input_bf16 = input_tensor
        else:
            state['input_bf16'].copy_(input_tensor)
            input_bf16 = state['input_bf16']

        if norm_weight_snapshot is not None:
            norm_weight_bf16 = norm_weight_snapshot
        elif norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous():
            norm_weight_bf16 = norm_weight
        else:
            state['norm_weight_bf16'].copy_(norm_weight)
            norm_weight_bf16 = state['norm_weight_bf16']

        rms_inputs = (d0, d1, d2) if sum3_input else (d_normed,)
        _record_tensors_on_stream(
            (
                *rms_inputs,
                input_bf16,
                norm_weight_bf16,
                inv_rms,
                state['grad_input'],
                state['dgamma'],
                state['dgamma_bf16'],
                state['dgamma_partials'],
            ),
            rms_stream,
        )

        if (
            use_single_out
            and sum3_input
            and hasattr(te_fused, "fused_rmsnorm_backward_sum3_out")
        ):
            te_fused.fused_rmsnorm_backward_sum3_out(
                d0, d1, d2, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
            if use_bf16_dgamma_out:
                state['dgamma_bf16'].copy_(state['dgamma'])
                state['dgamma_out'] = state['dgamma_bf16']
            else:
                state['dgamma_out'] = state['dgamma']
        elif (
            use_single_out
            and not sum3_input
            and hasattr(te_fused, "fused_rmsnorm_backward_out")
        ):
            te_fused.fused_rmsnorm_backward_out(
                d_normed, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
            if use_bf16_dgamma_out:
                state['dgamma_bf16'].copy_(state['dgamma'])
                state['dgamma_out'] = state['dgamma_bf16']
            else:
                state['dgamma_out'] = state['dgamma']
        elif (
            sum3_input
            and hasattr(te_fused, "fused_rmsnorm_backward_sum3_dx_only_out")
            and hasattr(te_fused, "fused_rmsnorm_backward_sum3_dgamma_tiled_out")
        ):
            dgamma_out = state['dgamma_bf16'] if use_bf16_dgamma_out else state['dgamma']
            state['dgamma_out'] = dgamma_out
            te_fused.fused_rmsnorm_backward_sum3_dx_only_out(
                d0, d1, d2, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input']
            )
            if use_bf16_dgamma_out:
                te_fused.fused_rmsnorm_backward_sum3_dgamma_tiled_bf16_out(
                    d0, d1, d2, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma_bf16']
                )
            else:
                te_fused.fused_rmsnorm_backward_sum3_dgamma_tiled_out(
                    d0, d1, d2, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma']
                )
        elif (
            hasattr(te_fused, "fused_rmsnorm_backward_dx_only_out")
            and hasattr(te_fused, "fused_rmsnorm_backward_dgamma_tiled_out")
        ):
            dgamma_out = state['dgamma_bf16'] if use_bf16_dgamma_out else state['dgamma']
            state['dgamma_out'] = dgamma_out
            te_fused.fused_rmsnorm_backward_dx_only_out(
                d_normed, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input']
            )
            if use_bf16_dgamma_out:
                te_fused.fused_rmsnorm_backward_dgamma_tiled_bf16_out(
                    d_normed, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma_bf16']
                )
            else:
                te_fused.fused_rmsnorm_backward_dgamma_tiled_out(
                    d_normed, input_bf16, inv_rms,
                    state['dgamma_partials'], state['dgamma']
                )
        elif sum3_input:
            te_fused.fused_rmsnorm_backward_sum3_out(
                d0, d1, d2, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
        else:
            te_fused.fused_rmsnorm_backward_out(
                d_normed, input_bf16, norm_weight_bf16, inv_rms,
                state['grad_input'], state['dgamma']
            )
    return state, rms_stream


def _get_tk():
    """Lazy-load the TK _C module (GEMM kernels)."""
    global _tk_module, _tk_import_attempted, _tk_import_error, _tk_backend_info
    if _tk_import_attempted:
        return _tk_module
    _tk_import_attempted = True
    try:
        import importlib.util
        _apply_localcta_v3_perf_defaults()

        localcta_enabled = use_tk_localcta()
        if localcta_enabled:
            fp4_root = _fp4_matmul_root()
            localcta_gemm_dir, extension_name = _localcta_gemm_variant_spec()
            tk_dir = os.path.join(
                fp4_root,
                'ThunderKittens', 'kernels', 'gemm', 'nvfp4_b200', localcta_gemm_dir,
            )
            alt_tk_dir = f'/opt/mfu/EXTERNAL_PATH{localcta_gemm_dir}'
        else:
            fp4_root = _fp4_matmul_root()
            tk_dir = os.path.join(
                fp4_root,
                'ThunderKittens', 'kernels', 'gemm', 'nvfp4_b200',
            )
            alt_tk_dir = '/opt/mfu/EXTERNAL_PATH'
            extension_name = '_C'

        so_path = None
        candidate_names = _extension_candidate_names(extension_name)
        if not localcta_enabled:
            candidate_names.extend(_extension_candidate_names('_C_nv_gemm'))
        so_path = _select_existing_tk_so([tk_dir, alt_tk_dir], candidate_names)

        if so_path is None:
            raise FileNotFoundError(
                f"TK GEMM _C.so not found in {tk_dir} or {alt_tk_dir}; "
                f"tried {candidate_names}"
            )

        if not torch.cuda.is_initialized():
            torch.cuda.init()
            _ = torch.zeros(1, device='cuda')
            torch.cuda.synchronize()

        # Load from explicit path to avoid _C name collisions with TK quant.
        # The module name must match the compiled PyInit_<name> symbol.
        import sys as _sys
        old_c = _sys.modules.pop('_C', None)
        so_base = os.path.basename(so_path)
        module_name = so_base.split('.cpython-')[0]
        spec = importlib.util.spec_from_file_location(module_name, so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Restore old _C if it existed (e.g. TK quant's _C)
        if old_c is not None:
            _sys.modules['_C'] = old_c
        elif '_C' in _sys.modules:
            del _sys.modules['_C']

        if localcta_enabled:
            direct_contract_enabled = use_tk_localcta_direct_contract()
            localcta_direct_mod = _get_tk_localcta_direct()
            localcta_variant = get_tk_localcta_variant()
            _localcta_regular_backend = getattr(mod, 'nvfp4_localcta_gemm', None)
            _localcta_grouped_backend = getattr(mod, 'nvfp4_localcta_grouped_gemm', None)
            _localcta_grouped_rope_live64_backend = getattr(
                mod, 'nvfp4_localcta_grouped_gemm_rope_live64', None
            )
            _localcta_grouped_rope_backend = getattr(
                mod, 'nvfp4_localcta_grouped_gemm_rope', None
            )
            _localcta_v3_regular_backend = (
                getattr(mod, 'nvfp4_localcta_gemm', None)
                if localcta_variant == 'v3'
                else getattr(mod, 'nvfp4_localcta_v3_regular_gemm', None)
            )
            _localcta_batched_backend = getattr(mod, 'nvfp4_localcta_batched_gemm', None)
            _localcta_batched_accum_backend = getattr(mod, 'nvfp4_localcta_batched_accum_gemm', None)
            _localcta_direct_gemm_backend = (
                getattr(localcta_direct_mod, 'nvfp4_gemm', None)
                if localcta_direct_mod is not None else None
            )
            _localcta_direct_grouped_backend = (
                getattr(localcta_direct_mod, 'nvfp4_grouped_gemm', None)
                if localcta_direct_mod is not None else None
            )
            use_v4_sg_direct = (
                localcta_variant == 'v4' and use_tk_localcta_v4_sg_direct_consumers()
            )
            strict_v4_contract = (
                localcta_variant == 'v4' and use_tk_localcta_v4_strict_path()
            )

            if _localcta_regular_backend is not None:
                def _localcta_regular_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D):
                    return _localcta_regular_backend(A, A_sc, A_sg, B, B_sc, B_sg, D)

            if _localcta_direct_gemm_backend is not None:
                def _localcta_direct_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D):
                    return _localcta_direct_gemm_backend(A, A_sc, A_sg, B, B_sc, B_sg, D)

            if _localcta_direct_grouped_backend is not None:
                def _localcta_direct_grouped_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D, *extra):
                    return _localcta_direct_grouped_backend(
                        A, A_sc, A_sg, B, B_sc, B_sg, D, *extra
                    )

            if _localcta_grouped_backend is not None:
                def _localcta_regular_grouped_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D, *extra):
                    D_K_opt = extra[0] if len(extra) > 0 else None
                    D_V_opt = extra[1] if len(extra) > 1 else None
                    silu_dim = extra[2] if len(extra) > 2 else 0
                    return _localcta_grouped_backend(
                        A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim
                    )

            if _localcta_grouped_rope_live64_backend is not None:
                def _localcta_regular_grouped_gemm_rope_live64(
                    A, A_sc, A_sg, B, B_sc, B_sg, D, D_K, D_V, rope_cs, rope_seq_len, silu_dim=0
                ):
                    return _localcta_grouped_rope_live64_backend(
                        A, A_sc, A_sg, B, B_sc, B_sg,
                        D, D_K, D_V, rope_cs, int(rope_seq_len), silu_dim
                    )

            if _localcta_grouped_rope_backend is not None:
                def _localcta_regular_grouped_gemm_rope(
                    A, A_sc, A_sg, B, B_sc, B_sg, D, D_K, D_V,
                    rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim,
                    silu_dim=0,
                ):
                    return _localcta_grouped_rope_backend(
                        A, A_sc, A_sg, B, B_sc, B_sg,
                        D, D_K, D_V, rope_cos, rope_sin,
                        int(rope_seq_len), int(rope_head_dim), int(rope_rotary_dim),
                        silu_dim,
                    )

            if _localcta_batched_backend is not None:
                def _localcta_regular_batched_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_list,
                ):
                    return _localcta_batched_backend(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_list,
                    )

            if _localcta_batched_accum_backend is not None:
                def _localcta_regular_batched_accum_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                ):
                    return _localcta_batched_accum_backend(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_out,
                    )

            if localcta_variant == 'v3':
                def _localcta_v3_regular_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D):
                    if _localcta_v3_regular_backend is not None:
                        return _localcta_v3_regular_backend(
                            A, A_sc, A_sg, B, B_sc, B_sg, D
                        )
                    return _localcta_regular_grouped_gemm(
                        A, A_sc, A_sg, B, B_sc, B_sg, D, None, None, 0
                    )

                def _localcta_v3_batched_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_list,
                ):
                    if _localcta_batched_backend is not None:
                        return _localcta_batched_backend(
                            A_list, A_sc_list, A_sg_list,
                            B_list, B_sc_list, B_sg_list,
                            D_list,
                        )
                    for i in range(len(A_list)):
                        _localcta_v3_regular_gemm(
                            A_list[i], A_sc_list[i], A_sg_list[i],
                            B_list[i], B_sc_list[i], B_sg_list[i],
                            D_list[i],
                        )
                    return None

                def _localcta_v3_batched_accum_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                ):
                    if _localcta_batched_accum_backend is not None:
                        return _localcta_batched_accum_backend(
                            A_list, A_sc_list, A_sg_list,
                            B_list, B_sc_list, B_sg_list,
                            D_out,
                        )
                    D_out.zero_()
                    tmp = torch.empty_like(D_out)
                    for i in range(len(A_list)):
                        _localcta_v3_regular_gemm(
                            A_list[i], A_sc_list[i], A_sg_list[i],
                            B_list[i], B_sc_list[i], B_sg_list[i],
                            tmp,
                        )
                        D_out.add_(tmp)
                    return None

            _fast_gemm_sg_backend = getattr(mod, 'nvfp4_localcta_fast_gemm_sg', None)
            _fast_gemm_outer_sg_backend = getattr(mod, 'nvfp4_localcta_fast_gemm_outer_sg', None)
            _fast_gemm_virtual_rescale_backend = getattr(
                mod, 'nvfp4_localcta_fast_gemm_virtual_rescale', None
            )
            _fast_gemm_residual_backend = getattr(mod, 'nvfp4_localcta_gemm_residual', None)
            _fast_gemm_residual_rms_backend = getattr(
                mod, 'nvfp4_localcta_gemm_residual_rms', None
            )
            _fast_gemm_prepared_residual_backend = getattr(
                mod, 'nvfp4_localcta_fast_gemm_residual', None
            )
            use_chunk_grid_fast_consumer = use_v4_sg_direct
            def _is_localcta_v4_chunkgrid_sg(sg, fp4):
                return _is_localcta_v4_chunkgrid_sg_tensor(sg, fp4)

            def _has_localcta_v4_chunkgrid_contract(A, A_sg, B, B_sg):
                return (
                    _is_localcta_v4_chunkgrid_sg(A_sg, A)
                    and _is_localcta_v4_chunkgrid_sg(B_sg, B)
                )

            def _is_localcta_v4_outer_sg(sg, fp4):
                if not (torch.is_tensor(sg) and torch.is_tensor(fp4)):
                    return False
                rows = int(fp4.size(0))
                if rows % 256 != 0:
                    return False
                tiles = rows // 256
                if sg.dim() == 1:
                    return int(sg.numel()) == tiles
                if sg.dim() == 2:
                    return tuple(sg.shape) in ((tiles, 1), (1, tiles))
                return False

            def _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg):
                return (
                    _is_localcta_v4_outer_sg(A_sg, A)
                    and _is_localcta_v4_outer_sg(B_sg, B)
                )

            debug_sg_contract = (
                os.environ.get('USE_TK_DEBUG_LOCALCTA_SG_CONTRACT', '0') == '1'
            )

            def _localcta_v4_sg_contract_kind(sg, fp4):
                if _is_localcta_v4_outer_sg(sg, fp4):
                    return 'outer'
                if _is_localcta_v4_chunkgrid_sg(sg, fp4):
                    return 'chunk_grid'
                if torch.is_tensor(sg) and int(sg.numel()) == 1:
                    return 'unit'
                return 'unknown'

            def _debug_check_localcta_v4_sg_pair(A, A_sg, B, B_sg):
                if not debug_sg_contract:
                    return
                a_kind = _localcta_v4_sg_contract_kind(A_sg, A)
                b_kind = _localcta_v4_sg_contract_kind(B_sg, B)
                if a_kind == b_kind and a_kind in ('outer', 'chunk_grid'):
                    return
                raise RuntimeError(
                    'localCTA v4 SG contract mismatch before GEMM: '
                    f'A={a_kind} fp4={tuple(A.shape)} sg={tuple(A_sg.shape)}; '
                    f'B={b_kind} fp4={tuple(B.shape)} sg={tuple(B_sg.shape)}'
                )

            def _debug_check_localcta_v4_sg_list(
                A_list, A_sg_list, B_list, B_sg_list,
            ):
                if not debug_sg_contract:
                    return
                if A_sg_list is None or B_sg_list is None:
                    raise RuntimeError(
                        'localCTA v4 SG contract list is missing before GEMM'
                    )
                lengths = tuple(
                    len(values)
                    for values in (A_list, A_sg_list, B_list, B_sg_list)
                )
                if len(set(lengths)) != 1:
                    raise RuntimeError(
                        'localCTA v4 SG contract list length mismatch before '
                        f'GEMM: {lengths}'
                    )
                for index, (A, A_sg, B, B_sg) in enumerate(
                    zip(A_list, A_sg_list, B_list, B_sg_list)
                ):
                    try:
                        _debug_check_localcta_v4_sg_pair(A, A_sg, B, B_sg)
                    except RuntimeError as error:
                        raise RuntimeError(
                            f'localCTA v4 SG contract mismatch at group {index}'
                        ) from error

            def _has_localcta_v4_outer_sg_contract_list(A_list, A_sg_list, B_list, B_sg_list):
                if A_sg_list is None or B_sg_list is None:
                    return False
                try:
                    same_len = (
                        len(A_list) == len(A_sg_list)
                        and len(A_list) == len(B_list)
                        and len(A_list) == len(B_sg_list)
                    )
                except TypeError:
                    return False
                if not same_len:
                    return False
                return all(
                    _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg)
                    for A, A_sg, B, B_sg in zip(A_list, A_sg_list, B_list, B_sg_list)
                )

            def _localcta_fast_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D):
                _debug_check_localcta_v4_sg_pair(A, A_sg, B, B_sg)
                if (
                    use_chunk_grid_fast_consumer
                    and _fast_gemm_sg_backend is not None
                    and _has_localcta_v4_chunkgrid_contract(A, A_sg, B, B_sg)
                ):
                    return _fast_gemm_sg_backend(A, A_sc, A_sg, B, B_sc, B_sg, D)
                if use_chunk_grid_fast_consumer:
                    return _localcta_regular_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D)
                if (
                    _fast_gemm_outer_sg_backend is not None
                    and _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg)
                ):
                    return _fast_gemm_outer_sg_backend(A, A_sc, A_sg, B, B_sc, B_sg, D)
                if (
                    _localcta_regular_backend is not None
                    and _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg)
                ):
                    return _localcta_regular_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D)
                return mod.nvfp4_localcta_fast_gemm(A, A_sc, B, B_sc, D)

            def _localcta_fast_gemm_residual(A, A_sc, A_sg, B, B_sc, B_sg, R, D):
                if (
                    _fast_gemm_residual_backend is not None
                    and _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg)
                ):
                    _trace_backend_choice('localcta_ffn_residual_dispatch', 'outer_sg_kernel')
                    return _fast_gemm_residual_backend(A, A_sc, A_sg, B, B_sc, B_sg, R, D)
                if (
                    use_chunk_grid_fast_consumer
                    and _has_localcta_v4_chunkgrid_contract(A, A_sg, B, B_sg)
                ):
                    _trace_backend_choice('localcta_ffn_residual_dispatch', 'chunkgrid_fallback_add')
                    _localcta_fast_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D)
                    D.add_(R)
                    return None
                if _fast_gemm_prepared_residual_backend is not None:
                    _trace_backend_choice('localcta_ffn_residual_dispatch', 'prepared_kernel')
                    return _fast_gemm_prepared_residual_backend(A, A_sc, B, B_sc, R, D)
                _trace_backend_choice('localcta_ffn_residual_dispatch', 'fallback_add')
                _localcta_fast_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D)
                D.add_(R)
                return None

            def _localcta_fast_gemm_residual_rms(
                A, A_sc, A_sg, B, B_sc, B_sg, R, D, row_rms_partial
            ):
                if (
                    _fast_gemm_residual_rms_backend is not None
                    and _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg)
                ):
                    _trace_backend_choice(
                        'localcta_exact_cde_w2_dispatch', 'outer_sg_kernel'
                    )
                    return _fast_gemm_residual_rms_backend(
                        A, A_sc, A_sg, B, B_sc, B_sg,
                        R, D, row_rms_partial,
                    )
                raise RuntimeError(
                    'localCTA exact C/D/E requires the native v4 outer-SG '
                    'residual RMS kernel'
                )

            if _fast_gemm_virtual_rescale_backend is not None:
                def _localcta_fast_gemm_virtual_rescale(
                    A, A_sc, A_sg, A_sg_chunks,
                    B, B_sc, B_sg, B_sg_chunks,
                    D,
                ):
                    return _fast_gemm_virtual_rescale_backend(
                        A, A_sc, A_sg, A_sg_chunks,
                        B, B_sc, B_sg, B_sg_chunks,
                        D,
                    )

            _fast_grouped_gemm_sg_backend = getattr(mod, 'nvfp4_localcta_fast_grouped_gemm_sg', None)
            _fast_grouped_gemm_outer_sg_backend = getattr(mod, 'nvfp4_localcta_fast_grouped_gemm_outer_sg', None)
            _fast_grouped_gemm_virtual_rescale_backend = getattr(
                mod, 'nvfp4_localcta_fast_grouped_gemm_virtual_rescale', None
            )
            _fast_batched_accum_outer_sg_backend = getattr(
                mod, 'nvfp4_localcta_fast_batched_accum_gemm_outer_sg', None
            )
            def _localcta_fast_grouped_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D, *extra):
                _debug_check_localcta_v4_sg_pair(A, A_sg, B, B_sg)
                D_K_opt = extra[0] if len(extra) > 0 else None
                D_V_opt = extra[1] if len(extra) > 1 else None
                silu_dim = extra[2] if len(extra) > 2 else 0
                if (
                    use_chunk_grid_fast_consumer
                    and _fast_grouped_gemm_sg_backend is not None
                    and _has_localcta_v4_chunkgrid_contract(A, A_sg, B, B_sg)
                ):
                    return _fast_grouped_gemm_sg_backend(
                        A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim
                    )
                if use_chunk_grid_fast_consumer:
                    return _localcta_regular_grouped_gemm(
                        A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim
                    )
                if (
                    _fast_grouped_gemm_outer_sg_backend is not None
                    and _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg)
                ):
                    return _fast_grouped_gemm_outer_sg_backend(
                        A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim
                    )
                if (
                    _localcta_grouped_backend is not None
                    and _has_localcta_v4_outer_sg_contract(A, A_sg, B, B_sg)
                ):
                    return _localcta_regular_grouped_gemm(
                        A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim
                    )
                return mod.nvfp4_localcta_fast_grouped_gemm(
                    A, A_sc, B, B_sc, D, D_K_opt, D_V_opt, silu_dim
                )

            if _fast_grouped_gemm_virtual_rescale_backend is not None:
                def _localcta_fast_grouped_gemm_virtual_rescale(
                    A, A_sc, A_sg, A_sg_chunks,
                    B, B_sc, B_sg, B_sg_chunks,
                    D, *extra,
                ):
                    D_K_opt = extra[0] if len(extra) > 0 else None
                    D_V_opt = extra[1] if len(extra) > 1 else None
                    silu_dim = extra[2] if len(extra) > 2 else 0
                    return _fast_grouped_gemm_virtual_rescale_backend(
                        A, A_sc, A_sg, A_sg_chunks,
                        B, B_sc, B_sg, B_sg_chunks,
                        D, D_K_opt, D_V_opt, silu_dim,
                    )

            def _localcta_fast_gemm_prepared(A, A_sc, A_sg, B, B_sc, B_sg, D):
                return mod.nvfp4_localcta_fast_gemm(A, A_sc, B, B_sc, D)

            def _localcta_fast_grouped_gemm_prepared(A, A_sc, A_sg, B, B_sc, B_sg, D, *extra):
                D_K_opt = extra[0] if len(extra) > 0 else None
                D_V_opt = extra[1] if len(extra) > 1 else None
                silu_dim = extra[2] if len(extra) > 2 else 0
                return mod.nvfp4_localcta_fast_grouped_gemm(
                    A, A_sc, B, B_sc, D, D_K_opt, D_V_opt, silu_dim
                )

            def _localcta_fast_batched_gemm(
                A_list, A_sc_list, A_sg_list,
                B_list, B_sc_list, B_sg_list,
                D_list,
            ):
                _debug_check_localcta_v4_sg_list(
                    A_list, A_sg_list, B_list, B_sg_list
                )
                if use_chunk_grid_fast_consumer:
                    return _localcta_regular_batched_gemm(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_list,
                    )
                if (
                    _localcta_batched_backend is not None
                    and _has_localcta_v4_outer_sg_contract_list(A_list, A_sg_list, B_list, B_sg_list)
                ):
                    return _localcta_regular_batched_gemm(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_list,
                    )
                return mod.nvfp4_localcta_fast_batched_gemm(
                    A_list, A_sc_list, B_list, B_sc_list, D_list
                )

            def _localcta_fast_batched_accum_gemm(
                A_list, A_sc_list, A_sg_list,
                B_list, B_sc_list, B_sg_list,
                D_out,
            ):
                _debug_check_localcta_v4_sg_list(
                    A_list, A_sg_list, B_list, B_sg_list
                )
                if use_chunk_grid_fast_consumer:
                    return _localcta_regular_batched_accum_gemm(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_out,
                    )
                if (
                    _fast_batched_accum_outer_sg_backend is not None
                    and _has_localcta_v4_outer_sg_contract_list(A_list, A_sg_list, B_list, B_sg_list)
                ):
                    return _fast_batched_accum_outer_sg_backend(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_out,
                    )
                if (
                    _localcta_batched_accum_backend is not None
                    and _has_localcta_v4_outer_sg_contract_list(A_list, A_sg_list, B_list, B_sg_list)
                ):
                    return _localcta_regular_batched_accum_gemm(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_out,
                    )
                return mod.nvfp4_localcta_fast_batched_accum_gemm(
                    A_list, A_sc_list, B_list, B_sc_list, D_out
                )

            def _localcta_fast_batched_gemm_strided(
                A_full, A_sc_list, A_sg_list, A_col_offsets, A_col_widths,
                B_list, B_sc_list, B_sg_list,
                D_list,
            ):
                return mod.nvfp4_localcta_fast_batched_gemm_strided(
                    A_full, A_sc_list, A_sg_list, A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list, D_list
                )
            _split3_backend = getattr(mod, 'nvfp4_localcta_fast_split3_dgrad_gemm', None)
            if _split3_backend is not None:
                def _localcta_fast_split3_dgrad_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                ):
                    _debug_check_localcta_v4_sg_list(
                        A_list, A_sg_list, B_list, B_sg_list
                    )
                    return _split3_backend(
                        A_list, A_sc_list, B_list, B_sc_list, D_out
                    )

            _split2_onepass_backend = getattr(mod, 'nvfp4_localcta_fast_split2_dgrad_onepass_gemm', None)
            _split2_onepass_sg_backend = getattr(mod, 'nvfp4_localcta_fast_split2_dgrad_onepass_gemm_sg', None)
            _split2_onepass_outer_sg_backend = getattr(
                mod, 'nvfp4_localcta_fast_split2_dgrad_onepass_gemm_outer_sg', None
            )
            if _split2_onepass_outer_sg_backend is None and localcta_variant == 'v4':
                _split2_onepass_outer_sg_backend = getattr(
                    mod, 'nvfp4_localcta_v3_split2_dgrad_onepass_gemm', None
                )
            if _split2_onepass_backend is not None:
                def _localcta_fast_split2_dgrad_onepass_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                    config_idx=-1,
                ):
                    _debug_check_localcta_v4_sg_list(
                        A_list, A_sg_list, B_list, B_sg_list
                    )
                    if (
                        _split2_onepass_outer_sg_backend is not None
                        and _has_localcta_v4_outer_sg_contract_list(A_list, A_sg_list, B_list, B_sg_list)
                    ):
                        return _split2_onepass_outer_sg_backend(
                            A_list, A_sc_list, A_sg_list,
                            B_list, B_sc_list, B_sg_list,
                            D_out, config_idx
                        )
                    if (
                        use_chunk_grid_fast_consumer
                        and
                        _split2_onepass_sg_backend is not None
                        and all(
                            _is_localcta_v4_chunkgrid_sg(sg, fp4)
                            for sg, fp4 in zip(A_sg_list, A_list)
                        )
                        and all(
                            _is_localcta_v4_chunkgrid_sg(sg, fp4)
                            for sg, fp4 in zip(B_sg_list, B_list)
                        )
                    ):
                        return _split2_onepass_sg_backend(
                            A_list, A_sc_list, A_sg_list,
                            B_list, B_sc_list, B_sg_list,
                            D_out, config_idx
                        )
                    if config_idx in (6, 7):
                        config_idx = 1
                    return _split2_onepass_backend(
                        A_list, A_sc_list, B_list, B_sc_list, D_out, config_idx
                    )

            _split2_onepass_v3_backend = getattr(mod, 'nvfp4_localcta_v3_split2_dgrad_onepass_gemm', None)
            if _split2_onepass_v3_backend is not None:
                def _localcta_v3_split2_dgrad_onepass_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                    config_idx=-1,
                ):
                    return _split2_onepass_v3_backend(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_out, config_idx
                    )

            def _localcta_v4_split2_dgrad_direct_gemm(
                A_list, A_sc_list, A_sg_list,
                B_list, B_sc_list, B_sg_list,
                D_out,
                config_idx=-1,
            ):
                del config_idx
                def _neutral_sg_like(sg, ref_tensor):
                    if torch.is_tensor(sg):
                        return torch.ones_like(sg, dtype=torch.float32)
                    return torch.ones(1, dtype=torch.float32, device=ref_tensor.device)
                rows = A_list[0].size(0) if len(A_list) > 0 else None
                mode = tk_localcta_v4_split2_dgrad_sg_mode(rows)
                if mode == 'v3_outer':
                    if _split2_onepass_v3_backend is None:
                        raise RuntimeError("localCTA v3 split2 onepass backend is unavailable")
                    A_sg_outer = [
                        _prepare_localcta_v4_outer_sg_for_direct(
                            A_sg_list[i], A_list[i].size(0) // 256, A_list[i].device
                        )
                        for i in range(len(A_list))
                    ]
                    B_sg_outer = [
                        _prepare_localcta_v4_outer_sg_for_direct(
                            B_sg_list[i], B_list[i].size(0) // 256, B_list[i].device
                        )
                        for i in range(len(B_list))
                    ]
                    return _localcta_v3_split2_dgrad_onepass_gemm(
                        A_list, A_sc_list, A_sg_outer,
                        B_list, B_sc_list, B_sg_outer,
                        D_out,
                        tk_localcta_v3_split2_onepass_config_idx(),
                    )
                elif mode == 'afold_b':
                    A_sc_use = [
                        _fold_localcta_v4_sg_into_prepared_sc(
                            A_sc_list[i],
                            A_sg_list[i],
                            A_list[i].size(0),
                            A_list[i].size(1) * 2,
                        )
                        for i in range(len(A_list))
                    ]
                    A_sg_use = [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))]
                    B_sg_use = B_sg_list
                elif mode == 'afold_bmean':
                    A_sc_use = [
                        _fold_localcta_v4_sg_into_prepared_sc(
                            A_sc_list[i],
                            A_sg_list[i],
                            A_list[i].size(0),
                            A_list[i].size(1) * 2,
                        )
                        for i in range(len(A_list))
                    ]
                    A_sg_use = [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))]
                    B_sg_use = [
                        _prepare_localcta_v4_ffn_split2_b_sg_for_direct(
                            B_sg_list[i],
                            B_list[i].size(0) // 256,
                            B_list[i].device,
                            rows=A_list[i].size(0),
                            mode=mode,
                        )
                        for i in range(len(B_list))
                    ]
                elif mode == 'afold_b1':
                    A_sc_use = [
                        _fold_localcta_v4_sg_into_prepared_sc(
                            A_sc_list[i],
                            A_sg_list[i],
                            A_list[i].size(0),
                            A_list[i].size(1) * 2,
                        )
                        for i in range(len(A_list))
                    ]
                    A_sg_use = [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))]
                    B_sg_use = [_neutral_sg_like(B_sg_list[i], B_list[i]) for i in range(len(B_list))]
                elif mode == 'afold_bfold1':
                    A_sc_use = [
                        _fold_localcta_v4_sg_into_prepared_sc(
                            A_sc_list[i],
                            A_sg_list[i],
                            A_list[i].size(0),
                            A_list[i].size(1) * 2,
                        )
                        for i in range(len(A_list))
                    ]
                    A_sg_use = [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))]
                    B_sc_list = [
                        _fold_localcta_v4_sg_into_prepared_sc(
                            B_sc_list[i],
                            B_sg_list[i],
                            B_list[i].size(0),
                            B_list[i].size(1) * 2,
                        )
                        for i in range(len(B_list))
                    ]
                    B_sg_use = [_neutral_sg_like(B_sg_list[i], B_list[i]) for i in range(len(B_list))]
                elif mode == 'afold_bfold_fast':
                    if _split2_onepass_backend is None:
                        raise RuntimeError("localCTA fast split2 onepass backend is unavailable")
                    A_sc_use = [
                        _fold_localcta_v4_sg_into_prepared_sc(
                            A_sc_list[i],
                            A_sg_list[i],
                            A_list[i].size(0),
                            A_list[i].size(1) * 2,
                        )
                        for i in range(len(A_list))
                    ]
                    B_sc_use = [
                        _fold_localcta_v4_sg_into_prepared_sc(
                            B_sc_list[i],
                            B_sg_list[i],
                            B_list[i].size(0),
                            B_list[i].size(1) * 2,
                        )
                        for i in range(len(B_list))
                    ]
                    return _localcta_fast_split2_dgrad_onepass_gemm(
                        A_list, A_sc_use, [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))],
                        B_list, B_sc_use, [_neutral_sg_like(B_sg_list[i], B_list[i]) for i in range(len(B_list))],
                        D_out,
                        -1,
                    )
                elif mode == 'a_b':
                    A_sc_use = A_sc_list
                    A_sg_use = A_sg_list
                    B_sg_use = B_sg_list
                elif mode == 'a_b1':
                    A_sc_use = A_sc_list
                    A_sg_use = A_sg_list
                    B_sg_use = [_neutral_sg_like(B_sg_list[i], B_list[i]) for i in range(len(B_list))]
                elif mode == 'a1_bmean':
                    A_sc_use = A_sc_list
                    A_sg_use = [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))]
                    B_sg_use = [
                        _prepare_localcta_v4_ffn_split2_b_sg_for_direct(
                            B_sg_list[i],
                            B_list[i].size(0) // 256,
                            B_list[i].device,
                            rows=A_list[i].size(0),
                            mode=mode,
                        )
                        for i in range(len(B_list))
                    ]
                elif mode == 'a1_b':
                    A_sc_use = A_sc_list
                    A_sg_use = [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))]
                    B_sg_use = B_sg_list
                elif mode == 'a1_b1':
                    A_sc_use = A_sc_list
                    A_sg_use = [_neutral_sg_like(A_sg_list[i], A_list[i]) for i in range(len(A_list))]
                    B_sg_use = [_neutral_sg_like(B_sg_list[i], B_list[i]) for i in range(len(B_list))]
                else:
                    raise ValueError(
                        "USE_TK_LOCALCTA_V4_SPLIT2_DGRAD_SG_MODE must be one of "
                        "{'v3_outer', 'afold_b', 'afold_bmean', 'afold_b1', 'afold_bfold1', 'afold_bfold_fast', "
                        "'a_b', 'a_b1', 'a1_bmean', 'a1_b', 'a1_b1'}"
                    )
                return _localcta_regular_batched_accum_gemm(
                    A_list, A_sc_use, A_sg_use,
                    B_list, B_sc_list, B_sg_use,
                    D_out,
                )

            def _localcta_v4_split2_dgrad_strict_gemm(
                A_list, A_sc_list, A_sg_list,
                B_list, B_sc_list, B_sg_list,
                D_out,
                config_idx=-1,
            ):
                if (
                    use_chunk_grid_fast_consumer
                    and
                    _split2_onepass_sg_backend is not None
                    and all(
                        _is_localcta_v4_chunkgrid_sg(sg, fp4)
                        for sg, fp4 in zip(A_sg_list, A_list)
                    )
                    and all(
                        _is_localcta_v4_chunkgrid_sg(sg, fp4)
                        for sg, fp4 in zip(B_sg_list, B_list)
                    )
                ):
                    return _localcta_fast_split2_dgrad_onepass_gemm(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_out,
                        config_idx,
                    )
                return _localcta_regular_batched_accum_gemm(
                    A_list, A_sc_list, A_sg_list,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                )

            _split3_strided_backend = getattr(mod, 'nvfp4_localcta_fast_split3_dgrad_strided_gemm', None)
            _split3_strided_sg_backend = getattr(mod, 'nvfp4_localcta_fast_split3_dgrad_strided_gemm_sg', None)
            if _split3_strided_backend is not None:
                def _localcta_fast_split3_dgrad_strided_gemm(
                    A_full, A_sc_list, A_sg_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                ):
                    if (
                        _split3_strided_sg_backend is not None
                        and all(torch.is_tensor(sg) for sg in A_sg_list)
                        and all(torch.is_tensor(sg) for sg in B_sg_list)
                    ):
                        return _split3_strided_sg_backend(
                            A_full, A_sc_list, A_sg_list,
                            A_col_offsets, A_col_widths,
                            B_list, B_sc_list, B_sg_list, D_out
                        )
                    return _split3_strided_backend(
                        A_full, A_sc_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, D_out
                    )

            _split3_strided_v3_backend = getattr(mod, 'nvfp4_localcta_v3_split3_dgrad_strided_gemm', None)
            if _split3_strided_v3_backend is not None:
                def _localcta_v3_split3_dgrad_strided_gemm(
                    A_full, A_sc_list, A_sg_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                ):
                    return _split3_strided_v3_backend(
                        A_full, A_sc_list, A_sg_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, B_sg_list, D_out
                    )

            _split3_strided_sum_backend = getattr(mod, 'nvfp4_localcta_fast_split3_dgrad_strided_sum_gemm', None)
            if _split3_strided_sum_backend is not None:
                def _localcta_fast_split3_dgrad_strided_sum_gemm(
                    A_full, A_sc_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                ):
                    return _split3_strided_sum_backend(
                        A_full, A_sc_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, D_out
                    )

            _split3_strided_onepass_backend = getattr(mod, 'nvfp4_localcta_fast_split3_dgrad_strided_onepass_gemm', None)
            _split3_strided_onepass_full_b_backend = getattr(
                mod, 'nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm', None
            )
            _split3_strided_onepass_full_b_sg_backend = getattr(
                mod, 'nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_sg', None
            )
            if _split3_strided_onepass_backend is not None:
                def _localcta_fast_split3_dgrad_strided_onepass_gemm(
                    A_full, A_sc_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                    config_idx=-1,
                ):
                    del B_sg_list
                    return _split3_strided_onepass_backend(
                        A_full, A_sc_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, D_out, config_idx
                    )
            if _split3_strided_onepass_full_b_backend is not None:
                def _localcta_fast_split3_dgrad_strided_onepass_full_b_gemm(
                    A_full, A_sc_list,
                    A_col_offsets, A_col_widths,
                    B_full, B_sc_full,
                    D_out,
                    config_idx=-1,
                ):
                    return _split3_strided_onepass_full_b_backend(
                        A_full, A_sc_list,
                        A_col_offsets, A_col_widths,
                        B_full, B_sc_full, D_out, config_idx
                    )
            if _split3_strided_onepass_full_b_sg_backend is not None:
                def _localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_sg(
                    A_full, A_sc_list, A_sg_list,
                    A_col_offsets, A_col_widths,
                    B_full, B_sc_full, B_sg_full,
                    D_out,
                    config_idx=-1,
                ):
                    return _split3_strided_onepass_full_b_sg_backend(
                        A_full, A_sc_list, A_sg_list,
                        A_col_offsets, A_col_widths,
                        B_full, B_sc_full, B_sg_full, D_out, config_idx
                    )
            _split2_strided_sum_backend = getattr(mod, 'nvfp4_localcta_fast_split2_dgrad_strided_sum_gemm', None)
            if _split2_strided_sum_backend is not None:
                def _localcta_fast_split2_dgrad_strided_sum_gemm(
                    A_full, A_sc_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                ):
                    return _split2_strided_sum_backend(
                        A_full, A_sc_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, D_out
                    )

            _split2_strided_onepass_backend = getattr(mod, 'nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm', None)
            if _split2_strided_onepass_backend is not None:
                def _localcta_fast_split2_dgrad_strided_onepass_gemm(
                    A_full, A_sc_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                    config_idx=-1,
                ):
                    del B_sg_list
                    return _split2_strided_onepass_backend(
                        A_full, A_sc_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, D_out,
                        config_idx
                    )

            _split2_strided_onepass_outer_sg_backend = getattr(
                mod, 'nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg', None
            )
            if _split2_strided_onepass_outer_sg_backend is not None:
                def _localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg(
                    A_full, A_sc_list, A_sg_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                    config_idx=-1,
                ):
                    return _split2_strided_onepass_outer_sg_backend(
                        A_full, A_sc_list, A_sg_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, B_sg_list,
                        D_out, config_idx
                    )

            _split2_strided_onepass_sg_backend = getattr(
                mod, 'nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_sg', None
            )
            if _split2_strided_onepass_sg_backend is not None:
                def _localcta_fast_split2_dgrad_strided_onepass_gemm_sg(
                    A_full, A_sc_list, A_sg_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_out,
                    config_idx=-1,
                ):
                    return _split2_strided_onepass_sg_backend(
                        A_full, A_sc_list, A_sg_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, B_sg_list,
                        D_out, config_idx
                    )

            _w2_dgrad_silu_quant_backend = getattr(
                mod, 'nvfp4_localcta_w2_dgrad_silu_quant_gemm', None
            )
            if _w2_dgrad_silu_quant_backend is not None:
                def _localcta_w2_dgrad_silu_quant_gemm(
                    A, A_sc, A_sg,
                    B, B_sc, B_sg,
                    h3, h1_raw,
                    row_fp4_full, row_sc_full, row_sg_full,
                    col_fp4_full, col_sc_full, col_sg_full,
                    config_idx=-1,
                ):
                    return _w2_dgrad_silu_quant_backend(
                        A, A_sc, A_sg,
                        B, B_sc, B_sg,
                        h3, h1_raw,
                        row_fp4_full, row_sc_full, row_sg_full,
                        col_fp4_full, col_sc_full, col_sg_full,
                        config_idx,
                    )

            _batched_strided_backend = (
                getattr(mod, 'nvfp4_localcta_v3_batched_gemm_strided', None)
                if localcta_variant == 'v3'
                else getattr(mod, 'nvfp4_localcta_fast_batched_gemm_strided', None)
            )
            if _batched_strided_backend is not None:
                def _localcta_fast_batched_gemm_strided(
                    A_full, A_sc_list, A_sg_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_list,
                ):
                    return _batched_strided_backend(
                        A_full, A_sc_list, A_sg_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, B_sg_list,
                        D_list
                    )

            _batched_strided_nopdl_backend = getattr(mod, 'nvfp4_localcta_fast_batched_gemm_strided_nopdl', None)
            if _batched_strided_nopdl_backend is not None:
                def _localcta_fast_batched_gemm_strided_nopdl(
                    A_full, A_sc_list, A_sg_list,
                    A_col_offsets, A_col_widths,
                    B_list, B_sc_list, B_sg_list,
                    D_list,
                ):
                    return _batched_strided_nopdl_backend(
                        A_full, A_sc_list, A_sg_list,
                        A_col_offsets, A_col_widths,
                        B_list, B_sc_list, B_sg_list,
                        D_list
                    )

            _split_dgrad_sum = getattr(mod, 'nvfp4_localcta_split_dgrad_sum', None)
            _sum3 = getattr(mod, 'sum3_bf16', None)
            _sum2 = getattr(mod, 'sum2_bf16', None)

            gemm_impl = _localcta_fast_gemm
            grouped_impl = _localcta_fast_grouped_gemm
            batched_accum_impl = staticmethod(_localcta_fast_batched_accum_gemm)
            batched_impl = staticmethod(_localcta_fast_batched_gemm)
            if localcta_variant == 'v3':
                gemm_impl = _localcta_v3_regular_gemm
                grouped_impl = _localcta_regular_grouped_gemm
                batched_accum_impl = staticmethod(_localcta_v3_batched_accum_gemm)
                batched_impl = staticmethod(_localcta_v3_batched_gemm)
            if direct_contract_enabled:
                if _localcta_direct_grouped_backend is None:
                    raise RuntimeError(
                        "USE_TK_LOCALCTA_DIRECT_CONTRACT=1 requires the direct localCTA GEMM module"
                    )
                # Folded prepared localCTA payloads belong on the fast
                # unit-scale consumers. Only the explicit direct-contract path
                # should use the SG-aware grouped / batched consumers.
                grouped_impl = _localcta_direct_grouped_gemm
                batched_accum_impl = (
                    staticmethod(localcta_direct_mod.nvfp4_batched_accum_gemm)
                    if localcta_direct_mod is not None and hasattr(localcta_direct_mod, 'nvfp4_batched_accum_gemm')
                    else staticmethod(_localcta_fast_batched_accum_gemm)
                )
                batched_impl = (
                    staticmethod(localcta_direct_mod.nvfp4_batched_gemm)
                    if localcta_direct_mod is not None and hasattr(localcta_direct_mod, 'nvfp4_batched_gemm')
                    else staticmethod(_localcta_fast_batched_gemm)
                )

            use_v3_prepared_split2 = (
                localcta_variant == 'v3' and use_tk_localcta_v3_enable_prepared_split2()
            )
            if use_v3_prepared_split2 and not direct_contract_enabled:
                grouped_impl = _localcta_fast_grouped_gemm
            if use_v4_sg_direct and not direct_contract_enabled:
                grouped_impl = _localcta_regular_grouped_gemm
                batched_accum_impl = staticmethod(_localcta_regular_batched_accum_gemm)
            if strict_v4_contract:
                # The v4 production contract is the fast outer-SG consumer:
                # local row/col scales are applied once in the GEMM epilogue.
                # Keep strict/cpp-only from falling back to the slow SG-aware
                # regular grouped path unless the explicit direct contract is
                # selected above.
                batched_accum_impl = staticmethod(_localcta_regular_batched_accum_gemm)
                batched_impl = staticmethod(_localcta_regular_batched_gemm)
            attrs = {
                'nvfp4_gemm': staticmethod(gemm_impl),
                'nvfp4_grouped_gemm': staticmethod(grouped_impl),
                'nvfp4_batched_accum_gemm': batched_accum_impl,
                'nvfp4_batched_gemm': batched_impl,
                '_is_localcta': True,
            }
            attrs['nvfp4_gemm_fast'] = staticmethod(_localcta_fast_gemm)
            _h_residual_carrier = getattr(
                mod, 'nvfp4_localcta_h_residual_carrier', None
            )
            if _h_residual_carrier is not None:
                attrs['nvfp4_h_residual_carrier'] = staticmethod(
                    _h_residual_carrier
                )
            if (
                _fast_gemm_residual_backend is not None
                or _fast_gemm_prepared_residual_backend is not None
            ):
                attrs['nvfp4_gemm_residual'] = staticmethod(_localcta_fast_gemm_residual)
            if _fast_gemm_residual_rms_backend is not None:
                attrs['nvfp4_gemm_residual_rms'] = staticmethod(
                    _localcta_fast_gemm_residual_rms
                )
            attrs['nvfp4_batched_accum_gemm_fast'] = staticmethod(_localcta_fast_batched_accum_gemm)
            attrs['nvfp4_batched_gemm_fast'] = staticmethod(_localcta_fast_batched_gemm)
            if _fast_gemm_virtual_rescale_backend is not None:
                attrs['nvfp4_gemm_virtual_rescale'] = staticmethod(
                    _localcta_fast_gemm_virtual_rescale
                )
            if _localcta_fast_grouped_gemm is not None:
                attrs['nvfp4_grouped_gemm_fast'] = staticmethod(_localcta_fast_grouped_gemm)
            if _fast_grouped_gemm_virtual_rescale_backend is not None:
                attrs['nvfp4_grouped_gemm_virtual_rescale'] = staticmethod(
                    _localcta_fast_grouped_gemm_virtual_rescale
                )
            if _localcta_grouped_rope_live64_backend is not None:
                attrs['nvfp4_grouped_gemm_rope_live64'] = staticmethod(
                    _localcta_regular_grouped_gemm_rope_live64
                )
            if _localcta_grouped_rope_backend is not None:
                attrs['nvfp4_grouped_gemm_rope'] = staticmethod(
                    _localcta_regular_grouped_gemm_rope
                )
            if _localcta_regular_backend is not None:
                attrs['nvfp4_gemm_highacc'] = staticmethod(_localcta_regular_gemm)
            if use_v4_sg_direct:
                attrs['nvfp4_split3_dgrad_gemm'] = staticmethod(_localcta_regular_batched_accum_gemm)
            elif _split3_backend is not None and localcta_variant != 'v3':
                attrs['nvfp4_split3_dgrad_gemm'] = staticmethod(
                    _localcta_regular_batched_accum_gemm if localcta_variant == 'v3' else _localcta_fast_split3_dgrad_gemm
                )
            allow_v3_prepared_split2 = localcta_variant != 'v3' or use_v3_prepared_split2
            if strict_v4_contract:
                attrs['nvfp4_split2_dgrad_onepass_gemm'] = staticmethod(_localcta_v4_split2_dgrad_strict_gemm)
            elif use_v4_sg_direct:
                attrs['nvfp4_split2_dgrad_onepass_gemm'] = staticmethod(_localcta_v4_split2_dgrad_direct_gemm)
            elif _split2_onepass_backend is not None and allow_v3_prepared_split2:
                attrs['nvfp4_split2_dgrad_onepass_gemm'] = staticmethod(
                    _localcta_fast_split2_dgrad_onepass_gemm
                )
            if localcta_variant == 'v3':
                if _split3_strided_v3_backend is not None:
                    attrs['nvfp4_split3_dgrad_strided_gemm'] = staticmethod(_localcta_v3_split3_dgrad_strided_gemm)
            elif _split3_strided_backend is not None:
                attrs['nvfp4_split3_dgrad_strided_gemm'] = staticmethod(_localcta_fast_split3_dgrad_strided_gemm)
            if _split3_strided_sum_backend is not None and localcta_variant != 'v3' and not use_v4_sg_direct:
                attrs['nvfp4_split3_dgrad_strided_sum_gemm'] = staticmethod(_localcta_fast_split3_dgrad_strided_sum_gemm)
            if _split3_strided_onepass_backend is not None and localcta_variant != 'v3' and not use_v4_sg_direct:
                attrs['nvfp4_split3_dgrad_strided_onepass_gemm'] = staticmethod(_localcta_fast_split3_dgrad_strided_onepass_gemm)
            if _split3_strided_onepass_full_b_backend is not None and localcta_variant != 'v3' and not use_v4_sg_direct:
                attrs['nvfp4_split3_dgrad_strided_onepass_full_b_gemm'] = staticmethod(
                    _localcta_fast_split3_dgrad_strided_onepass_full_b_gemm
                )
            if _split3_strided_onepass_full_b_sg_backend is not None and localcta_variant != 'v3' and not use_v4_sg_direct:
                attrs['nvfp4_split3_dgrad_strided_onepass_full_b_gemm_sg'] = staticmethod(
                    _localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_sg
                )
            if _split2_strided_sum_backend is not None and allow_v3_prepared_split2 and not use_v4_sg_direct:
                attrs['nvfp4_split2_dgrad_strided_sum_gemm'] = staticmethod(_localcta_fast_split2_dgrad_strided_sum_gemm)
            if _split2_strided_onepass_backend is not None and allow_v3_prepared_split2 and not use_v4_sg_direct:
                attrs['nvfp4_split2_dgrad_strided_onepass_gemm'] = staticmethod(_localcta_fast_split2_dgrad_strided_onepass_gemm)
            if _split2_strided_onepass_outer_sg_backend is not None and localcta_variant != 'v3':
                attrs['nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg'] = staticmethod(
                    _localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg
                )
            if _split2_strided_onepass_sg_backend is not None and localcta_variant != 'v3':
                attrs['nvfp4_split2_dgrad_strided_onepass_gemm_sg'] = staticmethod(
                    _localcta_fast_split2_dgrad_strided_onepass_gemm_sg
                )
            if _w2_dgrad_silu_quant_backend is not None and localcta_variant == 'v4':
                attrs['nvfp4_w2_dgrad_silu_quant_gemm'] = staticmethod(
                    _localcta_w2_dgrad_silu_quant_gemm
                )
            if _split2_onepass_v3_backend is not None and localcta_variant == 'v3' and use_tk_localcta_v3_split2_onepass():
                attrs['nvfp4_v3_split2_dgrad_onepass_gemm'] = staticmethod(_localcta_v3_split2_dgrad_onepass_gemm)
            if _batched_strided_backend is not None:
                attrs['nvfp4_batched_gemm_strided'] = staticmethod(_localcta_fast_batched_gemm_strided)
            if _batched_strided_nopdl_backend is not None and localcta_variant != 'v3':
                attrs['nvfp4_batched_gemm_strided_nopdl'] = staticmethod(_localcta_fast_batched_gemm_strided_nopdl)
            if _split_dgrad_sum is not None:
                attrs['nvfp4_split_dgrad_sum'] = staticmethod(_split_dgrad_sum)
            if _sum3 is not None:
                attrs['sum3_bf16'] = staticmethod(_sum3)
            if _sum2 is not None:
                attrs['sum2_bf16'] = staticmethod(_sum2)
        else:
            attrs = {
                'nvfp4_gemm': staticmethod(mod.nvfp4_gemm),
                'nvfp4_grouped_gemm': staticmethod(mod.nvfp4_grouped_gemm),
                'nvfp4_split_dgrad_sum': staticmethod(mod.nvfp4_split_dgrad_sum),
                'nvfp4_batched_accum_gemm': staticmethod(mod.nvfp4_batched_accum_gemm),
                'nvfp4_batched_gemm': staticmethod(mod.nvfp4_batched_gemm),
                '_is_localcta': False,
            }
            _grouped_k = getattr(mod, 'nvfp4_grouped_k_gemm', None)
            if _grouped_k is not None:
                attrs['nvfp4_grouped_k_gemm'] = staticmethod(_grouped_k)
            _strided = getattr(mod, 'nvfp4_batched_gemm_strided', None)
            if _strided is not None:
                attrs['nvfp4_batched_gemm_strided'] = staticmethod(_strided)
            _gemm_config = getattr(mod, 'nvfp4_gemm_config', None)
            if _gemm_config is not None:
                attrs['nvfp4_gemm_config'] = staticmethod(_gemm_config)
            _gemm_config_nopdl = getattr(mod, 'nvfp4_gemm_config_nopdl', None)
            if _gemm_config_nopdl is not None:
                attrs['nvfp4_gemm_config_nopdl'] = staticmethod(_gemm_config_nopdl)
            _nopdl = getattr(mod, 'nvfp4_grouped_gemm_nopdl', None)
            if _nopdl is not None:
                attrs['nvfp4_grouped_gemm_nopdl'] = staticmethod(_nopdl)
            _rope_live64 = getattr(mod, 'nvfp4_grouped_gemm_rope_live64', None)
            if _rope_live64 is not None:
                attrs['nvfp4_grouped_gemm_rope_live64'] = staticmethod(_rope_live64)
            _rope_packed_cat = getattr(mod, 'nvfp4_grouped_gemm_rope_packed_cat', None)
            if _rope_packed_cat is not None:
                attrs['nvfp4_grouped_gemm_rope_packed_cat'] = staticmethod(_rope_packed_cat)
            _rope_packed_split = getattr(mod, 'nvfp4_grouped_gemm_rope_packed_split', None)
            if _rope_packed_split is not None:
                attrs['nvfp4_grouped_gemm_rope_packed_split'] = staticmethod(_rope_packed_split)
            _inverse_rope_packed = getattr(mod, 'nvfp4_inverse_rope_packed_qk', None)
            if _inverse_rope_packed is not None:
                attrs['nvfp4_inverse_rope_packed_qk'] = staticmethod(_inverse_rope_packed)
            _forward_rope_packed = getattr(mod, 'nvfp4_forward_rope_packed_qk', None)
            if _forward_rope_packed is not None:
                attrs['nvfp4_forward_rope_packed_qk'] = staticmethod(_forward_rope_packed)
            _nopdl_config = getattr(mod, 'nvfp4_grouped_gemm_config_nopdl', None)
            if _nopdl_config is not None:
                attrs['nvfp4_grouped_gemm_config_nopdl'] = staticmethod(_nopdl_config)
            _gemm_nopdl = getattr(mod, 'nvfp4_gemm_nopdl', None)
            if _gemm_nopdl is not None:
                attrs['nvfp4_gemm_nopdl'] = staticmethod(_gemm_nopdl)
            _gemm_residual_rms = getattr(mod, 'nvfp4_gemm_residual_rms', None)
            _row_rms_reduce = getattr(mod, 'nvfp4_row_rms_reduce', None)
            if _gemm_residual_rms is not None:
                attrs['nvfp4_gemm_residual_rms'] = staticmethod(_gemm_residual_rms)
            if _row_rms_reduce is not None:
                attrs['nvfp4_row_rms_reduce'] = staticmethod(_row_rms_reduce)
            _batched_strided_nopdl = getattr(mod, 'nvfp4_batched_gemm_strided_nopdl', None)
            if _batched_strided_nopdl is not None:
                attrs['nvfp4_batched_gemm_strided_nopdl'] = staticmethod(_batched_strided_nopdl)
            _sum3 = getattr(mod, 'sum3_bf16', None)
            if _sum3 is not None:
                attrs['sum3_bf16'] = staticmethod(_sum3)
            _accum_v2 = getattr(mod, 'nvfp4_accum_gemm_v2', None)
            if _accum_v2 is not None:
                attrs['nvfp4_accum_gemm_v2'] = staticmethod(_accum_v2)
        _tk_module = type('TK', (), attrs)()
        if localcta_enabled and localcta_variant == 'v3':
            _v3_regular_backend = getattr(mod, 'nvfp4_localcta_gemm')
            _v3_batched_backend = getattr(mod, 'nvfp4_localcta_batched_gemm', None)
            _v3_batched_accum_backend = getattr(mod, 'nvfp4_localcta_batched_accum_gemm', None)

            def _v3_instance_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D):
                B_sg = _normalize_localcta_v3_tilegrid_b_sg(B, B_sg)
                return _v3_regular_backend(A, A_sc, A_sg, B, B_sc, B_sg, D)

            def _v3_instance_batched_gemm(
                A_list, A_sc_list, A_sg_list,
                B_list, B_sc_list, B_sg_list,
                D_list,
            ):
                if _v3_batched_backend is not None:
                    return _v3_batched_backend(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_list,
                    )
                for i in range(len(A_list)):
                    _v3_instance_gemm(
                        A_list[i], A_sc_list[i], A_sg_list[i],
                        B_list[i], B_sc_list[i], B_sg_list[i],
                        D_list[i],
                    )
                return None

            def _v3_instance_batched_accum_gemm(
                A_list, A_sc_list, A_sg_list,
                B_list, B_sc_list, B_sg_list,
                D_out,
            ):
                if _v3_batched_accum_backend is not None:
                    return _v3_batched_accum_backend(
                        A_list, A_sc_list, A_sg_list,
                        B_list, B_sc_list, B_sg_list,
                        D_out,
                    )
                D_out.zero_()
                tmp = torch.empty_like(D_out)
                for i in range(len(A_list)):
                    _v3_instance_gemm(
                        A_list[i], A_sc_list[i], A_sg_list[i],
                        B_list[i], B_sc_list[i], B_sg_list[i],
                        tmp,
                    )
                    D_out.add_(tmp)
                return None

            _tk_module.nvfp4_gemm = _v3_instance_gemm
            _tk_module.nvfp4_batched_gemm = _v3_instance_batched_gemm
            _tk_module.nvfp4_batched_accum_gemm = _v3_instance_batched_accum_gemm
        _tk_backend_info = {
            "fp4_matmul_root": fp4_root,
            "mode": "localcta" if localcta_enabled else "tk",
            "localcta_variant": get_tk_localcta_variant() if localcta_enabled else None,
            "localcta_direct_contract": bool(direct_contract_enabled) if localcta_enabled else False,
            "gemm": _artifact_info(so_path),
            "gemm_module_name": module_name,
            "gemm_module_origin": getattr(getattr(mod, "__spec__", None), "origin", None),
            "has_localcta_regular_gemm": bool(getattr(mod, 'nvfp4_localcta_gemm', None)),
            "has_localcta_fast_gemm": bool(getattr(mod, 'nvfp4_localcta_fast_gemm', None)),
            "has_localcta_fast_grouped_gemm": bool(getattr(mod, 'nvfp4_localcta_fast_grouped_gemm', None)),
            "has_localcta_direct_gemm": bool(_localcta_direct_gemm_backend) if localcta_enabled else False,
            "has_localcta_direct_grouped_gemm": bool(_localcta_direct_grouped_backend) if localcta_enabled else False,
        }
        mode = "localCTA TK" if localcta_enabled else "ThunderKittens NVFP4"
        logger.info("[TK GEMM] %s GEMM loaded successfully", mode)
        print(f"[TK GEMM] ✅ {mode} GEMM loaded — USE_TK_GEMM=1", flush=True)
    except Exception as e:
        _tk_import_error = str(e)
        _tk_module = None
        logger.warning(f"[TK GEMM] Failed to load ThunderKittens: {e}")
    return _tk_module


def _load_tk_extension_from_path(so_path: str):
    import importlib.util
    import sys as _sys

    old_c = _sys.modules.pop('_C', None)
    so_base = os.path.basename(so_path)
    module_name = so_base.split('.cpython-')[0]
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if old_c is not None:
        _sys.modules['_C'] = old_c
    elif '_C' in _sys.modules:
        del _sys.modules['_C']
    return mod


def _select_existing_tk_so(search_dirs, candidate_names):
    """Pick the first requested backend alias from the first directory that has one."""
    for d in search_dirs:
        for candidate_name in candidate_names:
            candidate = os.path.join(d, candidate_name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _get_tk_plain():
    """Load the plain TK GEMM module regardless of USE_TK_LOCALCTA."""
    global _tk_plain_module, _tk_plain_import_attempted, _tk_plain_import_error
    if _tk_plain_import_attempted:
        return _tk_plain_module
    _tk_plain_import_attempted = True
    try:
        fp4_root = _fp4_matmul_root()
        tk_dir = os.path.join(
            fp4_root,
            'ThunderKittens', 'kernels', 'gemm', 'nvfp4_b200',
        )
        alt_tk_dir = '/opt/mfu/EXTERNAL_PATH'
        candidate_names = [
            *_extension_candidate_names('_C'),
            *_extension_candidate_names('_C_nv_gemm'),
        ]
        so_path = _select_existing_tk_so([tk_dir, alt_tk_dir], candidate_names)
        if so_path is None:
            raise FileNotFoundError(
                f"plain TK GEMM _C.so not found in {tk_dir} or {alt_tk_dir}; "
                f"tried {candidate_names}"
            )
        if not torch.cuda.is_initialized():
            torch.cuda.init()
            _ = torch.zeros(1, device='cuda')
            torch.cuda.synchronize()
        mod = _load_tk_extension_from_path(so_path)
        attrs = {
            'nvfp4_gemm': staticmethod(mod.nvfp4_gemm),
            'nvfp4_grouped_gemm': staticmethod(mod.nvfp4_grouped_gemm),
            'nvfp4_split_dgrad_sum': staticmethod(mod.nvfp4_split_dgrad_sum),
            'nvfp4_batched_accum_gemm': staticmethod(mod.nvfp4_batched_accum_gemm),
            'nvfp4_batched_gemm': staticmethod(mod.nvfp4_batched_gemm),
            '_is_localcta': False,
        }
        _grouped_k = getattr(mod, 'nvfp4_grouped_k_gemm', None)
        if _grouped_k is not None:
            attrs['nvfp4_grouped_k_gemm'] = staticmethod(_grouped_k)
        _strided = getattr(mod, 'nvfp4_batched_gemm_strided', None)
        if _strided is not None:
            attrs['nvfp4_batched_gemm_strided'] = staticmethod(_strided)
        _gemm_config = getattr(mod, 'nvfp4_gemm_config', None)
        if _gemm_config is not None:
            attrs['nvfp4_gemm_config'] = staticmethod(_gemm_config)
        _gemm_config_nopdl = getattr(mod, 'nvfp4_gemm_config_nopdl', None)
        if _gemm_config_nopdl is not None:
            attrs['nvfp4_gemm_config_nopdl'] = staticmethod(_gemm_config_nopdl)
        _strided_nopdl = getattr(mod, 'nvfp4_batched_gemm_strided_nopdl', None)
        if _strided_nopdl is not None:
            attrs['nvfp4_batched_gemm_strided_nopdl'] = staticmethod(_strided_nopdl)
        _nopdl = getattr(mod, 'nvfp4_grouped_gemm_nopdl', None)
        if _nopdl is not None:
            attrs['nvfp4_grouped_gemm_nopdl'] = staticmethod(_nopdl)
        _rope_live64 = getattr(mod, 'nvfp4_grouped_gemm_rope_live64', None)
        if _rope_live64 is not None:
            attrs['nvfp4_grouped_gemm_rope_live64'] = staticmethod(_rope_live64)
        _rope_packed_cat = getattr(mod, 'nvfp4_grouped_gemm_rope_packed_cat', None)
        if _rope_packed_cat is not None:
            attrs['nvfp4_grouped_gemm_rope_packed_cat'] = staticmethod(_rope_packed_cat)
        _rope_packed_split = getattr(mod, 'nvfp4_grouped_gemm_rope_packed_split', None)
        if _rope_packed_split is not None:
            attrs['nvfp4_grouped_gemm_rope_packed_split'] = staticmethod(_rope_packed_split)
        _inverse_rope_packed = getattr(mod, 'nvfp4_inverse_rope_packed_qk', None)
        if _inverse_rope_packed is not None:
            attrs['nvfp4_inverse_rope_packed_qk'] = staticmethod(_inverse_rope_packed)
        _forward_rope_packed = getattr(mod, 'nvfp4_forward_rope_packed_qk', None)
        if _forward_rope_packed is not None:
            attrs['nvfp4_forward_rope_packed_qk'] = staticmethod(_forward_rope_packed)
        _nopdl_config = getattr(mod, 'nvfp4_grouped_gemm_config_nopdl', None)
        if _nopdl_config is not None:
            attrs['nvfp4_grouped_gemm_config_nopdl'] = staticmethod(_nopdl_config)
        _gemm_nopdl = getattr(mod, 'nvfp4_gemm_nopdl', None)
        if _gemm_nopdl is not None:
            attrs['nvfp4_gemm_nopdl'] = staticmethod(_gemm_nopdl)
        _gemm_residual = getattr(mod, 'nvfp4_gemm_residual', None)
        if _gemm_residual is not None:
            attrs['nvfp4_gemm_residual'] = staticmethod(_gemm_residual)
        _gemm_residual_rms = getattr(mod, 'nvfp4_gemm_residual_rms', None)
        _row_rms_reduce = getattr(mod, 'nvfp4_row_rms_reduce', None)
        if _gemm_residual_rms is not None:
            attrs['nvfp4_gemm_residual_rms'] = staticmethod(_gemm_residual_rms)
        if _row_rms_reduce is not None:
            attrs['nvfp4_row_rms_reduce'] = staticmethod(_row_rms_reduce)
        _sum3 = getattr(mod, 'sum3_bf16', None)
        if _sum3 is not None:
            attrs['sum3_bf16'] = staticmethod(_sum3)
        _sum3_rmsnorm_bwd_out = getattr(mod, 'sum3_rmsnorm_bwd_out', None)
        if _sum3_rmsnorm_bwd_out is not None:
            attrs['sum3_rmsnorm_bwd_out'] = staticmethod(_sum3_rmsnorm_bwd_out)
        _rmsnorm_bwd_residual_out = getattr(
            mod, 'rmsnorm_bwd_residual_out', None
        )
        if _rmsnorm_bwd_residual_out is not None:
            attrs['rmsnorm_bwd_residual_out'] = staticmethod(
                _rmsnorm_bwd_residual_out
            )
        _tk_plain_module = type('PlainTK', (), attrs)()
    except Exception as exc:
        _tk_plain_import_error = exc
        logger.exception("Failed to load plain TK GEMM _C module")
        _tk_plain_module = None
    return _tk_plain_module


def _get_native_sum3_rmsnorm_bwd_out():
    """Return the native plain-TK sum3 + RMSNorm backward ABI, if built."""
    tk_mod = _get_tk_plain()
    if tk_mod is None:
        return None
    return getattr(tk_mod, 'sum3_rmsnorm_bwd_out', None)


def _get_native_rmsnorm_bwd_residual_out():
    """Return the format-independent plain-TK residual RMS backward ABI."""
    tk_mod = _get_tk_plain()
    if tk_mod is None:
        return None
    return getattr(tk_mod, 'rmsnorm_bwd_residual_out', None)


def _get_tk_quant_plain():
    """Load the plain TK quant module regardless of USE_TK_LOCALCTA."""
    global _tk_plain_quant_mod, _tk_plain_quant_import_attempted, _tk_plain_quant_import_error
    if _tk_plain_quant_import_attempted:
        return _tk_plain_quant_mod
    _tk_plain_quant_import_attempted = True
    try:
        import sys

        base_dir = os.path.join(_fp4_matmul_root(), 'TK_quantisation')
        alt_base_dir = os.path.join(_LEGACY_FP4_MATMUL_ROOT, 'TK_quantisation')
        _prepend_import_paths_in_priority_order(
            (
                os.path.join(base_dir, 'nvfp4_v5'),
                os.path.join(alt_base_dir, 'nvfp4_v5'),
                os.path.join(base_dir, 'nvfp4'),
                os.path.join(alt_base_dir, 'nvfp4'),
            )
        )
        try:
            import _tk_quant_v5 as _mod
        except ImportError:
            import _tk_quant as _mod
        _tk_plain_quant_mod = _mod
    except Exception as exc:
        _tk_plain_quant_import_error = str(exc)
        _tk_plain_quant_mod = None
    return _tk_plain_quant_mod


def _get_tk_localcta_direct():
    """Load the direct localCTA GEMM module without swapping in the fast aliases."""
    global _tk_localcta_direct_module, _tk_localcta_direct_import_attempted, _tk_localcta_direct_import_error
    if _tk_localcta_direct_import_attempted:
        return _tk_localcta_direct_module
    _tk_localcta_direct_import_attempted = True
    try:
        import importlib.util
        import sys as _sys

        fp4_root = _fp4_matmul_root()
        localcta_gemm_dir, so_name = _localcta_gemm_variant_spec()
        tk_dir = os.path.join(
            fp4_root,
            'ThunderKittens', 'kernels', 'gemm', 'nvfp4_b200', localcta_gemm_dir,
        )
        alt_tk_dir = f'/opt/mfu/EXTERNAL_PATH{localcta_gemm_dir}'

        so_path = _find_extension_in_dirs(so_name, [tk_dir, alt_tk_dir])
        if so_path is None:
            candidates = ", ".join(_extension_candidate_names(so_name))
            raise FileNotFoundError(
                "localCTA direct GEMM extension not found in "
                f"{tk_dir} or {alt_tk_dir}; tried: {candidates}"
            )

        if not torch.cuda.is_initialized():
            torch.cuda.init()
            _ = torch.zeros(1, device='cuda')
            torch.cuda.synchronize()

        module_name = so_name
        old_c = _sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if old_c is not None:
            _sys.modules[module_name] = old_c
        elif module_name in _sys.modules:
            del _sys.modules[module_name]

        attrs = {
            'nvfp4_grouped_gemm': staticmethod(mod.nvfp4_localcta_grouped_gemm),
            '_is_localcta_direct': True,
            '_extension_path': os.path.realpath(so_path),
        }
        if getattr(mod, 'nvfp4_localcta_grouped_gemm_rope_live64', None) is not None:
            attrs['nvfp4_grouped_gemm_rope_live64'] = staticmethod(
                mod.nvfp4_localcta_grouped_gemm_rope_live64
            )
        if getattr(mod, 'nvfp4_localcta_grouped_gemm_rope', None) is not None:
            attrs['nvfp4_grouped_gemm_rope'] = staticmethod(
                mod.nvfp4_localcta_grouped_gemm_rope
            )
        if getattr(mod, 'nvfp4_localcta_gemm', None) is not None:
            attrs['nvfp4_gemm'] = staticmethod(mod.nvfp4_localcta_gemm)
        if getattr(mod, 'nvfp4_localcta_batched_accum_gemm', None) is not None:
            attrs['nvfp4_batched_accum_gemm'] = staticmethod(mod.nvfp4_localcta_batched_accum_gemm)
        if getattr(mod, 'nvfp4_localcta_batched_gemm', None) is not None:
            attrs['nvfp4_batched_gemm'] = staticmethod(mod.nvfp4_localcta_batched_gemm)
        if getattr(mod, 'nvfp4_localcta_prepare_outer_sg', None) is not None:
            attrs['prepare_outer_sg'] = staticmethod(mod.nvfp4_localcta_prepare_outer_sg)
        if getattr(mod, 'nvfp4_localcta_prepare_split_wgrad_a_sg', None) is not None:
            attrs['prepare_split_wgrad_a_sg'] = staticmethod(mod.nvfp4_localcta_prepare_split_wgrad_a_sg)
        if getattr(mod, 'nvfp4_localcta_prepare_split2_b_sg', None) is not None:
            attrs['prepare_split2_b_sg'] = staticmethod(mod.nvfp4_localcta_prepare_split2_b_sg)
        if getattr(mod, 'nvfp4_localcta_prepare_w2_dgrad_b_sg', None) is not None:
            attrs['prepare_w2_dgrad_b_sg'] = staticmethod(mod.nvfp4_localcta_prepare_w2_dgrad_b_sg)
        if getattr(mod, 'nvfp4_localcta_w2_dgrad_silu_quant_gemm', None) is not None:
            attrs['nvfp4_w2_dgrad_silu_quant_gemm'] = staticmethod(
                mod.nvfp4_localcta_w2_dgrad_silu_quant_gemm
            )
        if getattr(mod, 'nvfp4_localcta_fold_sg_into_prepared_sc', None) is not None:
            attrs['fold_sg_into_prepared_sc'] = staticmethod(mod.nvfp4_localcta_fold_sg_into_prepared_sc)
        if getattr(mod, 'nvfp4_localcta_fold_outer_sg_into_prepared_sc', None) is not None:
            attrs['fold_outer_sg_into_prepared_sc'] = staticmethod(
                mod.nvfp4_localcta_fold_outer_sg_into_prepared_sc
            )
        if getattr(mod, 'nvfp4_localcta_fast_split3_dgrad_strided_onepass_gemm', None) is not None:
            attrs['nvfp4_split3_dgrad_strided_onepass_gemm'] = staticmethod(
                mod.nvfp4_localcta_fast_split3_dgrad_strided_onepass_gemm
            )
        if getattr(mod, 'nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm', None) is not None:
            attrs['nvfp4_split3_dgrad_strided_onepass_full_b_gemm'] = staticmethod(
                mod.nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm
            )
        if getattr(mod, 'nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm', None) is not None:
            attrs['nvfp4_split2_dgrad_strided_onepass_gemm'] = staticmethod(
                mod.nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm
            )
        if getattr(mod, 'nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg', None) is not None:
            attrs['nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg'] = staticmethod(
                mod.nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg
            )
        if getattr(mod, 'nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_sg', None) is not None:
            attrs['nvfp4_split2_dgrad_strided_onepass_gemm_sg'] = staticmethod(
                mod.nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_sg
            )
        _tk_localcta_direct_module = type('TKLocalCTADirect', (), attrs)()
    except Exception as e:
        _tk_localcta_direct_import_error = str(e)
        _tk_localcta_direct_module = None
        logger.warning(f"[TK GEMM] Failed to load direct localCTA GEMM: {e}")
    return _tk_localcta_direct_module


def _get_tk_mixed_mx_localcta_quant():
    """Load the localCTA-v4 producer ABI without changing the active backend.

    The mixed MXFP4 route remains an MXFP4 route for forward and wgrad.  It
    borrows only the fused localCTA-v4 producer symbols used to build the
    dgrad row/weight-column carriers, so routing this through
    ``_get_tk_quant_for_gemm`` would be incorrect: that loader follows the
    process-wide ``USE_TK_LOCALCTA`` backend selection.
    """
    global _tk_mixed_mx_localcta_quant_module
    global _tk_mixed_mx_localcta_quant_import_attempted
    global _tk_mixed_mx_localcta_quant_import_error
    if _tk_mixed_mx_localcta_quant_import_attempted:
        return _tk_mixed_mx_localcta_quant_module
    _tk_mixed_mx_localcta_quant_import_attempted = True
    try:
        variant_dir = "nvfp4_CTA_local_v4"
        module_name = "_tk_quant_localcta_v4"
        runtime_root = os.path.realpath(_fp4_matmul_root())
        candidates = [
            os.path.join(runtime_root, "TK_quantisation", variant_dir)
        ]
        _prepend_import_paths_in_priority_order(candidates)
        if not any(os.path.isdir(path) for path in candidates):
            raise FileNotFoundError(
                "mixed MXFP4/localCTA producer requires the localCTA-v4 "
                f"extension under one of {candidates}"
            )
        module = __import__(module_name)
        module_file = os.path.realpath(getattr(module, "__file__", ""))
        expected_dir = os.path.realpath(candidates[0])
        if not module_file or os.path.commonpath(
            (module_file, expected_dir)
        ) != expected_dir:
            raise RuntimeError(
                "mixed MXFP4/localCTA producer resolved outside the pinned "
                f"FP4_MATMUL_ROOT: {module_file or '<unknown>'}"
            )
        _tk_mixed_mx_localcta_quant_module = module
    except Exception as exc:
        _tk_mixed_mx_localcta_quant_import_error = exc
        _tk_mixed_mx_localcta_quant_module = None
    return _tk_mixed_mx_localcta_quant_module


def tk_mixed_mx_localcta_quant_capabilities():
    """Return the sealed fused-producer capability receipt, or ``None``."""
    module = _get_tk_mixed_mx_localcta_quant()
    if module is None:
        return None
    query = getattr(module, "tk_mixed_mx_localcta_capabilities", None)
    return query() if query is not None else None


def _get_tk_mixed_localcta_direct():
    """Return only a direct GEMM loaded from the explicitly pinned runtime."""
    direct = _get_tk_localcta_direct()
    if direct is None:
        return None
    runtime_root = os.path.realpath(_fp4_matmul_root())
    expected_dir = os.path.realpath(
        os.path.join(
            runtime_root,
            "ThunderKittens",
            "kernels",
            "gemm",
            "nvfp4_b200",
            _localcta_gemm_variant_spec()[0],
        )
    )
    raw_extension_path = getattr(direct, "_extension_path", "")
    extension_path = (
        os.path.realpath(raw_extension_path) if raw_extension_path else ""
    )
    if not extension_path or os.path.commonpath(
        (extension_path, expected_dir)
    ) != expected_dir:
        raise RuntimeError(
            "mixed MXFP4/localCTA GEMM resolved outside the pinned "
            f"FP4_MATMUL_ROOT: {extension_path or '<unknown>'}"
        )
    return direct


def tk_mixed_localcta_dgrad(
    a_fp4,
    a_sc,
    a_sg,
    b_fp4,
    b_sc,
    b_sg,
    out=None,
):
    """Run one localCTA-v4 dgrad GEMM for the mixed MXFP4 route.

    ``a`` is the localCTA row-SR gradient carrier and ``b`` is the exact-2D
    localCTA weight-column carrier.  This helper deliberately calls the direct
    localCTA GEMM module: selecting the process-wide localCTA backend would
    also change forward/wgrad, violating the mixed-route identity.
    """
    if out is None:
        out = torch.empty(
            a_fp4.size(0),
            b_fp4.size(0),
            dtype=torch.bfloat16,
            device=a_fp4.device,
        )
    direct = _get_tk_mixed_localcta_direct()
    if direct is None or not hasattr(direct, "nvfp4_gemm"):
        raise RuntimeError(
            "mixed MXFP4/localCTA dgrad requires the direct localCTA-v4 GEMM"
        )
    direct.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, out)
    return out


def tk_mixed_localcta_split2_dgrad(
    a_fp4,
    a_sc,
    a_sg0,
    a_sg1,
    b0_fp4,
    b0_sc,
    b0_sg,
    b1_fp4,
    b1_sc,
    b1_sg,
    out=None,
    *,
    config_idx=-1,
):
    """Consume the fused ``[dh1|dh3]`` localCTA carrier in one GEMM.

    The producer stores the two logical arms next to one another without a
    BF16 concatenation: packed data uses two width-``H/2`` slices and scale
    data uses two width-``H/64`` slices.  Both arms intentionally share the
    stochastic coordinate, but each arm has its own independently finalized
    row outer-scale grid.  This wrapper keeps the process-wide GEMM backend
    unchanged and calls only the direct localCTA-v4 one-pass consumer.
    """
    if a_fp4.dim() != 2 or a_sc.dim() < 2:
        raise RuntimeError("mixed localCTA split2 dgrad requires 2D carriers")
    m = int(a_fp4.size(0))
    h = int(a_fp4.size(1))
    if h <= 0 or h % 64:
        raise RuntimeError(
            "mixed localCTA split2 packed width must be a positive multiple "
            f"of 64, got {h}"
        )
    if int(a_sc.size(1)) != h // 32:
        raise RuntimeError(
            "mixed localCTA split2 scale layout mismatch: "
            f"packed_width={h}, scale_width={int(a_sc.size(1))}"
        )
    if int(b0_fp4.size(0)) != int(b1_fp4.size(0)):
        raise RuntimeError("mixed localCTA split2 weights disagree on output width")
    if out is None:
        out = torch.empty(
            m,
            b0_fp4.size(0),
            dtype=torch.bfloat16,
            device=a_fp4.device,
        )
    direct = _get_tk_mixed_localcta_direct()
    symbol = (
        None
        if direct is None
        else getattr(
            direct,
            "nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg",
            None,
        )
    )
    if symbol is None:
        raise RuntimeError(
            "mixed MXFP4/localCTA FFN dgrad requires the direct localCTA-v4 "
            "split2 one-pass outer-SG GEMM"
        )
    h_packed = h // 2
    h_sc = h // 64
    # Mixed weight finalization has already reduced each localCTA weight to a
    # single common outer SG, folded that value into SC, and broadcast the
    # common value across the exact 128x128 grid.  The direct split-2 consumer
    # accepts one SG per 256-row output tile.  Under this dedicated mixed-route
    # contract a flattened prefix is therefore the exact outer vector and a
    # zero-copy view; the generic direct adapter would launch a needless GPU
    # reduction.  Do not apply the empirical FFN B-SG multiplier used by the
    # legacy localCTA training path.
    b0_sg_direct = _prepare_mixed_localcta_common_weight_sg_for_split2_direct(
        b0_sg, b0_fp4
    )
    b1_sg_direct = _prepare_mixed_localcta_common_weight_sg_for_split2_direct(
        b1_sg, b1_fp4
    )
    symbol(
        a_fp4,
        [
            a_sc.narrow(1, 0, h_sc),
            a_sc.narrow(1, h_sc, h_sc),
        ],
        [a_sg0, a_sg1],
        [0, h_packed],
        [h_packed, h_packed],
        [b0_fp4, b1_fp4],
        [b0_sc, b1_sc],
        [b0_sg_direct, b1_sg_direct],
        out,
        int(config_idx),
    )
    return out


def _prepare_mixed_localcta_common_weight_sg_for_split2_direct(
    sg: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Return the zero-copy direct-GEMM view of a mixed weight's common SG.

    This helper is intentionally stricter than the generic localCTA outer-SG
    adapter.  Its caller is the mixed fused producer, whose sealed ABI emits a
    contiguous FP32 128x128 grid containing one broadcast common value.
    Metadata that could describe any other carrier is rejected before launch.
    """
    if not torch.is_tensor(sg) or not torch.is_tensor(packed_weight):
        raise RuntimeError("mixed split2 weight SG and packed weight must be tensors")
    if packed_weight.dim() != 2 or packed_weight.dtype != torch.float4_e2m1fn_x2:
        raise RuntimeError(
            "mixed split2 packed weight must be 2D packed FP4, got "
            f"shape={tuple(packed_weight.shape)} dtype={packed_weight.dtype}"
        )
    rows = int(packed_weight.size(0))
    cols = int(packed_weight.size(1)) * 2
    if rows <= 0 or cols <= 0 or rows % 256 or cols % 256:
        raise RuntimeError(
            "mixed split2 packed weight must be 256-aligned, got "
            f"logical_shape=({rows}, {cols})"
        )
    expected_shape = (rows // 128, cols // 128)
    if (
        sg.dtype != torch.float32
        or sg.device != packed_weight.device
        or not sg.is_contiguous()
        or tuple(sg.shape) != expected_shape
    ):
        raise RuntimeError(
            "mixed split2 weight SG violates the common-broadcast 128x128 "
            "carrier contract: "
            f"shape={tuple(sg.shape)} expected={expected_shape} "
            f"dtype={sg.dtype} device={sg.device} contiguous={sg.is_contiguous()}"
        )
    tiles = rows // 256
    return sg.view(-1).narrow(0, 0, tiles).view(tiles, 1)


def use_tk_gemm() -> bool:
    """Check if TK GEMM should be used (USE_TK_GEMM env, default=1)."""
    if os.environ.get('USE_TK_GEMM', '1') == '0':
        return False
    return _get_tk() is not None


def is_tk_available() -> bool:
    """Check if TK module can be loaded."""
    return _get_tk() is not None


def _use_localcta_v3_runtime() -> bool:
    return use_tk_localcta() and get_tk_localcta_variant() == 'v3'


def _localcta_v3_contract() -> str:
    return os.environ.get('USE_TK_LOCALCTA_V3_CONTRACT', 'outerscale')


def _use_localcta_v3_tilegrid256() -> bool:
    return _use_localcta_v3_runtime() and _localcta_v3_contract() == 'tilegrid256'


def _use_localcta_v3_defer_col_dgrad() -> bool:
    return (
        _use_localcta_v3_runtime()
        and not _use_localcta_v3_tilegrid256()
        and os.environ.get('USE_TK_LOCALCTA_V3_DEFER_COL_DGRAD', '0') == '1'
    )


def _normalize_localcta_v3_tilegrid_b_sg(B_fp4, B_sg):
    if not (_use_localcta_v3_tilegrid256() and torch.is_tensor(B_sg) and B_sg.dim() == 2):
        return B_sg
    expected_k_tiles = (B_fp4.shape[1] * 2) // 256
    expected_n_tiles = B_fp4.shape[0] // 256
    if tuple(B_sg.shape) == (expected_n_tiles, expected_k_tiles):
        return B_sg.transpose(0, 1).contiguous()
    return B_sg


def _normalize_localcta_grouped_col_sg(sg_col):
    if not torch.is_tensor(sg_col):
        return sg_col
    if sg_col.dim() <= 2:
        return sg_col
    if sg_col.dim() == 3:
        parts = [sg_col[i] for i in range(sg_col.size(0))]
        if not parts:
            return sg_col
        cat_dim = 1 if parts[0].dim() == 2 and parts[0].size(0) == 1 else 0
        return torch.cat(parts, dim=cat_dim).contiguous()
    raise RuntimeError(f"Unsupported localCTA grouped SG rank: {tuple(sg_col.shape)}")


def _prepare_localcta_v4_chunkgrid_for_batched(sg, rows, cols, device=None):
    row_tiles = rows // 128
    col_tiles = cols // 128
    row_outer_tiles = rows // 256
    col_outer_tiles = cols // 256
    if not torch.is_tensor(sg):
        if device is None:
            raise RuntimeError("localCTA v4 direct SG adapter needs a device for scalar SG")
        return torch.full(
            (row_tiles, col_tiles),
            float(sg),
            dtype=torch.float32,
            device=device,
        )

    x = sg if sg.dtype == torch.float32 else sg.to(torch.float32)
    if x.dim() == 0:
        return torch.full(
            (row_tiles, col_tiles),
            float(x.item()),
            dtype=torch.float32,
            device=x.device,
        )
    if x.dim() == 1:
        if x.numel() == row_tiles * col_tiles:
            return x.contiguous().view(row_tiles, col_tiles)
        if x.numel() == row_tiles:
            return x.contiguous().view(row_tiles, 1).expand(row_tiles, col_tiles).contiguous()
        if x.numel() == col_tiles:
            return x.contiguous().view(1, col_tiles).expand(row_tiles, col_tiles).contiguous()
        if x.numel() == row_outer_tiles:
            return x.contiguous().view(row_outer_tiles, 1).repeat_interleave(2, dim=0).expand(row_tiles, col_tiles).contiguous()
        if x.numel() == col_outer_tiles:
            return x.contiguous().view(1, col_outer_tiles).repeat_interleave(2, dim=1).expand(row_tiles, col_tiles).contiguous()
        if x.numel() == 1:
            return x.contiguous().view(1, 1).expand(row_tiles, col_tiles).contiguous()
        return x.contiguous()
    if x.dim() != 2:
        return x.contiguous()

    if tuple(x.shape) == (row_tiles, col_tiles):
        return x.contiguous()
    if tuple(x.shape) == (col_tiles, row_tiles):
        return x.transpose(0, 1).contiguous()
    if tuple(x.shape) == (row_tiles, 1):
        return x.expand(row_tiles, col_tiles).contiguous()
    if tuple(x.shape) == (1, col_tiles):
        return x.expand(row_tiles, col_tiles).contiguous()
    if tuple(x.shape) == (row_outer_tiles, 1):
        return x.repeat_interleave(2, dim=0).expand(row_tiles, col_tiles).contiguous()
    if tuple(x.shape) == (1, row_outer_tiles):
        return x.transpose(0, 1).contiguous().repeat_interleave(2, dim=0).expand(row_tiles, col_tiles).contiguous()
    if tuple(x.shape) == (1, col_outer_tiles):
        return x.repeat_interleave(2, dim=1).expand(row_tiles, col_tiles).contiguous()
    if tuple(x.shape) == (col_outer_tiles, 1):
        return x.transpose(0, 1).contiguous().repeat_interleave(2, dim=1).expand(row_tiles, col_tiles).contiguous()
    if tuple(x.shape) == (1, 1):
        return x.expand(row_tiles, col_tiles).contiguous()
    return x.contiguous()


def _is_localcta_v4_chunkgrid_sg_tensor(sg, fp4):
    if not (torch.is_tensor(sg) and torch.is_tensor(fp4) and sg.dim() == 2):
        return False
    rows = int(fp4.size(0))
    cols = int(fp4.size(1)) * 2
    if rows % 128 != 0 or cols % 128 != 0:
        return False
    return tuple(sg.shape) == (rows // 128, cols // 128)


def _prepare_localcta_v4_outer_sg_for_direct(sg, tiles, device=None, row_axis: bool = True):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is not None and hasattr(tk_direct, 'prepare_outer_sg'):
        return tk_direct.prepare_outer_sg(sg, tiles, row_axis)
    if not torch.is_tensor(sg):
        if device is None:
            raise RuntimeError("localCTA v4 direct SG adapter needs a device for scalar SG")
        return torch.full((tiles, 1), float(sg), dtype=torch.float32, device=device)

    x = sg if sg.dtype == torch.float32 else sg.to(torch.float32)
    if x.dim() == 0:
        return torch.full((tiles, 1), float(x.item()), dtype=torch.float32, device=x.device)
    if x.dim() == 1:
        if x.numel() == tiles:
            return x.contiguous().view(tiles, 1)
        if x.numel() == tiles * 2:
            return x.contiguous().view(tiles, 2).amax(dim=1, keepdim=True)
        if x.numel() == 1:
            return x.contiguous().view(1, 1).expand(tiles, 1).contiguous()
        return x.contiguous()
    if x.dim() != 2:
        return x.contiguous()
    if tuple(x.shape) == (tiles, 1) or tuple(x.shape) == (1, tiles):
        return x.contiguous()
    if x.size(0) in (tiles, tiles * 2):
        if x.size(0) == tiles * 2:
            x = x.contiguous().view(tiles, 2, x.size(1)).amax(dim=1)
        return x.amax(dim=1, keepdim=True).contiguous()
    if x.size(1) in (tiles, tiles * 2):
        if x.size(1) == tiles * 2:
            x = x.contiguous().view(x.size(0), tiles, 2).amax(dim=2)
        return x.amax(dim=0, keepdim=True).transpose(0, 1).contiguous()
    return x.contiguous()


def _prepare_localcta_v4_ffn_split2_b_sg_for_direct(
    sg, tiles, device=None, *, rows: int | None = None, mode: str | None = None
):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is not None and hasattr(tk_direct, 'prepare_split2_b_sg'):
        return tk_direct.prepare_split2_b_sg(
            sg,
            tiles,
            float(tk_localcta_v4_split2_b_sg_scale(rows=rows, mode=mode)),
        )
    if not torch.is_tensor(sg):
        return _prepare_localcta_v4_outer_sg_for_direct(sg, tiles, device)

    x = sg if sg.dtype == torch.float32 else sg.to(torch.float32)
    if x.dim() == 2 and x.size(0) == tiles * 2:
        # For localCTA v4 FFN dgrad weights, the direct split2 consumer wants
        # one outer SG per 256-row K tile. The localCTA quantizer emits a
        # 128x128 chunk grid; collapsing by mean over the paired 128-row tiles
        # and the split width matches the regular TK grouped dgrad contract
        # far better than the old amax collapse.
        outer = x.contiguous().view(tiles, 2, -1).mean(dim=1).mean(dim=1, keepdim=True)
        return (outer * tk_localcta_v4_split2_b_sg_scale(rows=rows, mode=mode)).contiguous()
    return _prepare_localcta_v4_outer_sg_for_direct(x, tiles, device)


def _prepare_localcta_v4_split_wgrad_a_sg_for_direct(sg, tiles, device=None):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is not None and hasattr(tk_direct, 'prepare_split_wgrad_a_sg'):
        return tk_direct.prepare_split_wgrad_a_sg(
            sg,
            tiles,
            float(tk_localcta_v4_split_wgrad_a_sg_scale()),
        )
    if not torch.is_tensor(sg):
        return _prepare_localcta_v4_outer_sg_for_direct(sg, tiles, device)

    x = sg if sg.dtype == torch.float32 else sg.to(torch.float32)
    if x.dim() == 2 and x.size(0) == tiles * 2:
        scalar = x.mean() * tk_localcta_v4_split_wgrad_a_sg_scale()
        return torch.full((tiles, 1), float(scalar.item()), dtype=torch.float32, device=x.device)
    return _prepare_localcta_v4_outer_sg_for_direct(x, tiles, device)


def _prepare_localcta_v4_w2_dgrad_b_sg_for_direct(sg, tiles, device=None):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is not None and hasattr(tk_direct, 'prepare_w2_dgrad_b_sg'):
        return tk_direct.prepare_w2_dgrad_b_sg(
            sg,
            tiles,
            float(tk_localcta_v4_w2_dgrad_b_sg_scale()),
        )
    if not torch.is_tensor(sg):
        return _prepare_localcta_v4_outer_sg_for_direct(sg, tiles, device)

    x = sg if sg.dtype == torch.float32 else sg.to(torch.float32)
    if x.dim() == 2:
        scalar = x.mean() * tk_localcta_v4_w2_dgrad_b_sg_scale()
        return torch.full((tiles, 1), float(scalar.item()), dtype=torch.float32, device=x.device)
    return _prepare_localcta_v4_outer_sg_for_direct(x, tiles, device)


def _fold_localcta_v4_sg_into_prepared_sc(sc_raw, sg, rows, cols):
    if torch.is_tensor(sg) and rows % 256 == 0:
        if (
            (sg.dim() == 1 and sg.numel() == rows // 256)
            or (sg.dim() == 2 and sg.size(0) == rows // 256 and sg.size(1) == 1)
            or (sg.dim() == 2 and sg.size(0) == 1 and sg.size(1) == rows // 256)
        ):
            return _fold_localcta_v4_outer_sg_into_prepared_sc(sc_raw, sg, rows, cols)
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is not None and hasattr(tk_direct, 'fold_sg_into_prepared_sc'):
        return tk_direct.fold_sg_into_prepared_sc(sc_raw, sg, rows, cols)
    if not torch.is_tensor(sg):
        raise TypeError(f'expected tensor sg, got {type(sg)!r}')
    sg_grid = sg.to(torch.float32) if sg.dtype != torch.float32 else sg
    if sg_grid.dim() == 0:
        sg_grid = torch.full(
            (rows // 128, cols // 128),
            float(sg_grid.item()),
            dtype=torch.float32,
            device=sc_raw.device,
        )
    elif sg_grid.dim() == 1:
        if sg_grid.numel() == rows // 128:
            sg_grid = sg_grid.view(rows // 128, 1).expand(rows // 128, cols // 128)
        elif sg_grid.numel() == cols // 128:
            sg_grid = sg_grid.view(1, cols // 128).expand(rows // 128, cols // 128)
        elif sg_grid.numel() == 1:
            sg_grid = sg_grid.view(1, 1).expand(rows // 128, cols // 128)
    sg_prepared = sg_grid.repeat_interleave(2, dim=1).unsqueeze(-1)
    return (sc_raw.float() * sg_prepared).contiguous().to(torch.float8_e4m3fn)


def _fold_localcta_v4_outer_sg_into_prepared_sc(sc_raw, sg, rows, cols):
    """Fold v4 256-row outer SG into a prepared FP8 scale tensor."""
    tk_direct = _get_tk_localcta_direct()
    if sc_raw.dtype != torch.float8_e4m3fn:
        sc_raw = sc_raw.view(torch.float8_e4m3fn)
    if tk_direct is not None and hasattr(tk_direct, 'fold_outer_sg_into_prepared_sc'):
        return tk_direct.fold_outer_sg_into_prepared_sc(sc_raw, sg, rows, cols)
    sc_raw = sc_raw.contiguous()
    sg_outer = _prepare_localcta_v4_outer_sg_for_direct(
        sg, rows // 256, sc_raw.device
    ).reshape(-1)
    sg_prepared = sg_outer.repeat_interleave(2).view(rows // 128, 1, 1)
    return (sc_raw.float() * sg_prepared).contiguous().to(torch.float8_e4m3fn)


def tk_dispatch_gemm(
    tk,
    A_fp4,
    A_sc,
    A_sg,
    B_fp4,
    B_sc,
    B_sg,
    D,
    *,
    force_nopdl: bool | None = None,
):
    B_sg = _normalize_localcta_v3_tilegrid_b_sg(B_fp4, B_sg)
    gemm_fn = tk.nvfp4_gemm
    use_nopdl = use_tk_gemm_nopdl() if force_nopdl is None else force_nopdl
    if use_nopdl:
        gemm_fn = getattr(tk, 'nvfp4_gemm_nopdl', gemm_fn)
    if not getattr(tk, '_is_localcta', False):
        shape = (D.shape[0], D.shape[1], A_fp4.shape[1] * 2)
        config_id = get_tk_gemm_config(
            shape,
            use_production_default=not use_nopdl,
        )
        config_name = (
            'nvfp4_gemm_config_nopdl' if use_nopdl else 'nvfp4_gemm_config'
        )
        config_fn = getattr(tk, config_name, None)
        if config_id is not None and config_fn is not None:
            _trace_backend_choice(
                'regular_v5_gemm_config',
                f'M{shape[0]}_N{shape[1]}_K{shape[2]}={config_id}',
            )
            return config_fn(
                A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D, config_id
            )
    return gemm_fn(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D)


def tk_dispatch_batched_gemm(
    tk,
    A_list, A_sc_list, A_sg_list,
    B_list, B_sc_list, B_sg_list,
    D_list,
):
    return tk.nvfp4_batched_gemm(
        A_list, A_sc_list, A_sg_list,
        B_list, B_sc_list, B_sg_list,
        D_list,
    )


def tk_dispatch_batched_accum_gemm(
    tk,
    A_list, A_sc_list, A_sg_list,
    B_list, B_sc_list, B_sg_list,
    D_out,
):
    return tk.nvfp4_batched_accum_gemm(
        A_list, A_sc_list, A_sg_list,
        B_list, B_sc_list, B_sg_list,
        D_out,
    )


def _use_plain_tk_small_m_qkv_dgrad_eager(m: int, n_groups: int) -> bool:
    return m < 256 and n_groups > 1


def _pad_rows_bf16(t: torch.Tensor, target_rows: int) -> torch.Tensor:
    if t.dim() != 2:
        raise ValueError(f"_pad_rows_bf16 expects 2D tensor, got shape={tuple(t.shape)}")
    rows, cols = t.shape
    if rows >= target_rows:
        return t
    out = torch.zeros(target_rows, cols, dtype=t.dtype, device=t.device)
    out[:rows].copy_(t)
    return out


# ---------------------------------------------------------------------------
# Single GEMM wrappers — inputs are _TKQuantized with cached _tk_row/_tk_col
# ---------------------------------------------------------------------------
def tk_forward_gemm(x_nvfp4, w_nvfp4, out=None, use_localcta=None):
    """y = x @ w^T.  Both inputs must have _tk_row cached."""
    if use_localcta is None:
        use_localcta = use_tk_localcta()
    tk = _get_tk() if use_localcta else _get_tk_plain()
    x_fp4, x_sc, x_sg = x_nvfp4._tk_row
    w_fp4, w_sc, w_sg = w_nvfp4._tk_row
    if out is None:
        out = torch.empty(x_nvfp4.shape[0], w_nvfp4.shape[0],
                          dtype=torch.bfloat16, device=x_fp4.device)
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_gemm_virtual_rescale()
        and hasattr(tk, 'nvfp4_gemm_virtual_rescale')
        and (_has_virtual_rescale_chunk(x_nvfp4, True) or _has_virtual_rescale_chunk(w_nvfp4, True))
    ):
        tk.nvfp4_gemm_virtual_rescale(
            x_fp4, x_sc, x_sg, _chunk_sg_or_empty(x_nvfp4, True, x_fp4),
            w_fp4, w_sc, w_sg, _chunk_sg_or_empty(w_nvfp4, True, w_fp4),
            out,
        )
        return out
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and os.environ.get('USE_TK_LOCALCTA_V4_FAST_FORWARD_GEMM', '0') == '1'
        and hasattr(tk, 'nvfp4_gemm_fast')
    ):
        tk.nvfp4_gemm_fast(x_fp4, x_sc, x_sg, w_fp4, w_sc, w_sg, out)
        return out
    tk_dispatch_gemm(tk, x_fp4, x_sc, x_sg, w_fp4, w_sc, w_sg, out)
    return out


def tk_forward_gemm_residual(x_nvfp4, w_nvfp4, residual, out=None, use_localcta=None):
    """y = x @ w^T + residual, using a native epilogue when selected."""
    if use_localcta is None:
        use_localcta = use_tk_localcta()
    tk = _get_tk() if use_localcta else _get_tk_plain()
    x_fp4, x_sc, x_sg = x_nvfp4._tk_row
    w_fp4, w_sc, w_sg = w_nvfp4._tk_row
    if out is None:
        out = torch.empty(x_nvfp4.shape[0], w_nvfp4.shape[0],
                          dtype=torch.bfloat16, device=x_fp4.device)
    use_v5_residual = bool(
        not use_localcta
        and use_tk_v5_ffn_residual_epilogue_for_shape(
            int(x_fp4.shape[0]), int(w_fp4.shape[0]), int(x_fp4.shape[1]) * 2
        )
    )
    if use_v5_residual and not hasattr(tk, 'nvfp4_gemm_residual'):
        raise RuntimeError(
            "selected regular-v5 FFN residual epilogue requires "
            "nvfp4_gemm_residual"
        )
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_ffn_residual_epilogue()
        and hasattr(tk, 'nvfp4_gemm_residual')
    ):
        tk.nvfp4_gemm_residual(x_fp4, x_sc, x_sg, w_fp4, w_sc, w_sg, residual, out)
        return out
    if use_v5_residual:
        tk.nvfp4_gemm_residual(
            x_fp4, x_sc, x_sg, w_fp4, w_sc, w_sg, residual, out
        )
        return out
    tk_forward_gemm(x_nvfp4, w_nvfp4, out, use_localcta=use_localcta)
    out.add_(residual)
    return out


def tk_forward_gemm_residual_rms_partial(
    x_nvfp4,
    w_nvfp4,
    residual: torch.Tensor,
    out=None,
    use_localcta: bool = False,
):
    """Native residual GEMM that also emits row sum-of-squares partials."""
    tk = _get_tk() if use_localcta else _get_tk_plain()
    if (
        tk is None
        or not hasattr(tk, 'nvfp4_gemm_residual_rms')
    ):
        raise RuntimeError(
            "exact C/D/E requires nvfp4_gemm_residual_rms in the selected "
            "native TK extension"
        )

    x_fp4, x_sc, x_sg = x_nvfp4._tk_row
    w_fp4, w_sc, w_sg = w_nvfp4._tk_row
    rows = int(x_nvfp4.shape[0])
    cols = int(w_nvfp4.shape[0])
    if cols <= 0 or cols > 4096 or cols % 32:
        raise RuntimeError(
            f"exact C/D/E requires a 32-aligned hidden size <= 4096, got {cols}"
        )
    if tuple(residual.shape) != (rows, cols):
        raise RuntimeError(
            f"exact C/D/E residual shape {tuple(residual.shape)} != {(rows, cols)}"
        )
    if out is None:
        out = torch.empty(
            (rows, cols), dtype=torch.bfloat16, device=x_fp4.device
        )
    partial_width = cols // (256 if use_localcta else 32)
    row_rms_partial = torch.empty(
        (rows, partial_width), dtype=torch.float32, device=x_fp4.device
    )
    tk.nvfp4_gemm_residual_rms(
        x_fp4, x_sc, x_sg,
        w_fp4, w_sc, w_sg,
        residual, out, row_rms_partial,
    )
    return out, row_rms_partial


def tk_forward_gemm_h_carrier(
    x_nvfp4,
    w_nvfp4,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
    out=None,
    use_localcta=None,
):
    """Residual GEMM plus native tile-RMS row/column NVFP4 carrier."""
    if use_localcta is None:
        use_localcta = use_tk_localcta()
    if not use_localcta:
        raise RuntimeError("H tile carrier is retained only for localCTA v4")
    tk = _get_tk()
    if tk is None or not hasattr(tk, 'nvfp4_h_residual_carrier'):
        raise RuntimeError(
            "H tile carrier requires the native NVFP4 residual GEMM ABI"
        )
    quant = _get_tk_quant_plain()
    if quant is None:
        raise RuntimeError("H tile carrier requires the v5 native quant extension")

    x_fp4, x_sc, x_sg = x_nvfp4._tk_row
    w_fp4, w_sc, w_sg = w_nvfp4._tk_row
    rows = int(x_nvfp4.shape[0])
    cols = int(w_nvfp4.shape[0])
    if rows % 128 or cols % 128:
        raise RuntimeError(
            f"H tile carrier requires 128-aligned output, got {(rows, cols)}"
        )
    if rows % 256 or cols % 256:
        raise RuntimeError(
            f"localCTA H carrier requires 256-aligned output, got {(rows, cols)}"
        )
    residual = residual.contiguous()
    gamma = gamma.contiguous()
    if residual.dtype != torch.bfloat16 or gamma.dtype != torch.bfloat16:
        raise RuntimeError("H tile carrier requires bf16 residual and gamma")
    if out is None:
        out = torch.empty(
            (rows, cols), dtype=torch.bfloat16, device=x_fp4.device
        )

    row_fp4 = torch.empty(
        (rows, cols // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=x_fp4.device,
    )
    row_sc = torch.empty(
        (rows // 128, cols // 64, 512),
        dtype=torch.float8_e4m3fn,
        device=x_fp4.device,
    )
    col_fp4 = torch.empty(
        (cols, rows // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=x_fp4.device,
    )
    col_sc = torch.empty(
        (cols // 128, rows // 64, 512),
        dtype=torch.float8_e4m3fn,
        device=x_fp4.device,
    )
    r_tile = torch.empty(
        (rows // 128, cols // 128),
        dtype=torch.float32,
        device=x_fp4.device,
    )
    amax_tile = torch.empty_like(r_tile)
    work_counter = torch.empty((1,), dtype=torch.int32, device=x_fp4.device)

    row_sg = torch.empty(
        (rows // 256, 1), dtype=torch.float32, device=x_fp4.device
    )
    col_sg = torch.empty(
        (1, cols // 256), dtype=torch.float32, device=x_fp4.device
    )
    tk.nvfp4_h_residual_carrier(
        x_fp4, x_sc, x_sg,
        w_fp4, w_sc, w_sg,
        residual, gamma, out,
        r_tile, amax_tile, row_sg, col_sg, float(epsilon),
    )
    quant.tk_localcta_h_tile_quantize_out(
        out, gamma, r_tile, row_sg, col_sg,
        row_fp4, row_sc, col_fp4, col_sc, work_counter,
    )

    return (
        out,
        row_fp4,
        row_sc,
        row_sg,
        col_fp4,
        col_sc,
        col_sg,
        r_tile,
    )


def tk_h_tile_backward(
    du: torch.Tensor,
    z: torch.Tensor,
    gamma: torch.Tensor,
    r_tile: torch.Tensor,
):
    """Native 128x128 tile-RMS backward shared by v5 and localCTA."""
    quant = _get_tk_quant_plain()
    if quant is None or not hasattr(quant, 'tk_h_tile_backward_out'):
        raise RuntimeError("H tile backward requires the v5 native CUDA ABI")
    du = du.contiguous()
    z = z.contiguous()
    gamma = gamma.contiguous()
    rows, cols = du.shape
    dx = torch.empty_like(du)
    dgamma_partial = torch.empty(
        (rows // 128, cols), dtype=torch.float32, device=du.device
    )
    dgamma = torch.empty((cols,), dtype=torch.bfloat16, device=du.device)
    quant.tk_h_tile_backward_out(
        du, z, gamma, r_tile, dx, dgamma_partial, dgamma
    )
    return dx, dgamma


def tk_dgrad_gemm(
    dY_nvfp4,
    w_nvfp4,
    dx=None,
    use_localcta=None,
    *,
    use_nopdl: bool | None = None,
):
    """dx = dY @ W.  dY uses _tk_row, W uses _tk_col."""
    if use_localcta is None:
        use_localcta = use_tk_localcta()
    a_fp4, a_sc, a_sg = dY_nvfp4._tk_row
    b_fp4, b_sc, b_sg = w_nvfp4._tk_col
    if dx is None:
        dx = torch.empty(dY_nvfp4.shape[0], w_nvfp4.shape[1],
                         dtype=torch.bfloat16, device=a_fp4.device)
    tk = _get_tk() if use_localcta else _get_tk_plain()
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_gemm_virtual_rescale()
        and hasattr(tk, 'nvfp4_gemm_virtual_rescale')
        and (_has_virtual_rescale_chunk(dY_nvfp4, True) or _has_virtual_rescale_chunk(w_nvfp4, False))
    ):
        tk.nvfp4_gemm_virtual_rescale(
            a_fp4, a_sc, a_sg, _chunk_sg_or_empty(dY_nvfp4, True, a_fp4),
            b_fp4, b_sc, b_sg, _chunk_sg_or_empty(w_nvfp4, False, b_fp4),
            dx,
        )
        return dx
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_fast_single_dgrad()
        and hasattr(tk, 'nvfp4_gemm_fast')
    ):
        tk.nvfp4_gemm_fast(
            a_fp4, a_sc, a_sg,
            b_fp4, b_sc, b_sg,
            dx,
        )
        return dx
    tk_dispatch_gemm(
        tk,
        a_fp4,
        a_sc,
        a_sg,
        b_fp4,
        b_sc,
        b_sg,
        dx,
        force_nopdl=use_nopdl,
    )
    return dx


def tk_wgrad_gemm(x_nvfp4, dY_nvfp4, dW=None, use_localcta=None):
    """dW = dY^T @ x.  Both use _tk_col."""
    if use_localcta is None:
        use_localcta = use_tk_localcta()
    a_fp4, a_sc, a_sg = dY_nvfp4._tk_col
    b_fp4, b_sc, b_sg = x_nvfp4._tk_col
    M_dy, N = dY_nvfp4.shape
    _, K = x_nvfp4.shape
    if dW is None:
        dW = torch.empty(N, K, dtype=torch.bfloat16, device=a_fp4.device)
    tk = _get_tk() if use_localcta else _get_tk_plain()
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_gemm_virtual_rescale()
        and hasattr(tk, 'nvfp4_gemm_virtual_rescale')
        and (_has_virtual_rescale_chunk(dY_nvfp4, False) or _has_virtual_rescale_chunk(x_nvfp4, False))
    ):
        tk.nvfp4_gemm_virtual_rescale(
            a_fp4, a_sc, a_sg, _chunk_sg_or_empty(dY_nvfp4, False, a_fp4),
            b_fp4, b_sc, b_sg, _chunk_sg_or_empty(x_nvfp4, False, b_fp4),
            dW,
        )
        return dW
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_fast_single_wgrad()
        and hasattr(tk, 'nvfp4_gemm_fast')
    ):
        tk.nvfp4_gemm_fast(
            a_fp4, a_sc, a_sg,
            b_fp4, b_sc, b_sg,
            dW,
        )
        return dW
    tk_dispatch_gemm(tk, a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dW)
    return dW


def tk_v4_direct_wgrad_col_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dW):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is None or not hasattr(tk_direct, 'nvfp4_gemm'):
        raise RuntimeError("localCTA direct GEMM backend is unavailable for v4 SG wgrad")
    if hasattr(tk_direct, 'prepare_outer_sg'):
        a_sg = tk_direct.prepare_outer_sg(a_sg, a_fp4.size(0) // 256, True)
        b_sg = tk_direct.prepare_outer_sg(b_sg, b_fp4.size(0) // 256, False)
    else:
        a_sg = _prepare_localcta_v4_outer_sg_for_direct(
            a_sg, a_fp4.size(0) // 256, a_fp4.device
        )
        b_sg = _prepare_localcta_v4_outer_sg_for_direct(
            b_sg, b_fp4.size(0) // 256, b_fp4.device
        )
    tk_direct.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dW)


def tk_v4_direct_dgrad_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dx):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is None or not hasattr(tk_direct, 'nvfp4_gemm'):
        raise RuntimeError("localCTA direct GEMM backend is unavailable for v4 dgrad")
    if hasattr(tk_direct, 'prepare_outer_sg'):
        a_sg = torch.ones_like(tk_direct.prepare_outer_sg(a_sg, a_fp4.size(0) // 256, True))
        b_sg = tk_direct.prepare_outer_sg(b_sg, b_fp4.size(0) // 256, False)
    else:
        a_sg = torch.ones_like(
            _prepare_localcta_v4_outer_sg_for_direct(
                a_sg, a_fp4.size(0) // 256, a_fp4.device
            )
        )
        b_sg = _prepare_localcta_v4_outer_sg_for_direct(
            b_sg, b_fp4.size(0) // 256, b_fp4.device
        )
    tk_direct.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dx)


def tk_v4_direct_raw_dgrad_gemm(dy_bf16, w_col_raw, dx):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is None or not hasattr(tk_direct, 'nvfp4_gemm'):
        raise RuntimeError("localCTA direct GEMM backend is unavailable for v4 raw dgrad")
    tkq = _get_tk_quant_for_gemm()
    if not hasattr(tkq, 'tk_quantize_for_gemm_direct'):
        raise RuntimeError("localCTA direct quant helper is unavailable for v4 raw dgrad")
    dy = dy_bf16 if dy_bf16.dtype == torch.bfloat16 and dy_bf16.is_contiguous() else dy_bf16.to(torch.bfloat16).contiguous()
    dy_quant = tkq.tk_quantize_for_gemm_direct(dy, True, True)
    a_fp4, a_sc, a_sg = dy_quant[0], dy_quant[1], dy_quant[4]
    b_fp4, b_sc, b_sg = w_col_raw
    tk_direct.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dx)


def tk_v4_direct_raw_wgrad_gemm(dy_bf16, x_bf16, dW,
                                dy_encode_centric=False,
                                x_encode_centric=False):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is None or not hasattr(tk_direct, 'nvfp4_gemm'):
        raise RuntimeError("localCTA direct GEMM backend is unavailable for v4 raw wgrad")
    tkq = _get_tk_quant_for_gemm()
    if not hasattr(tkq, 'tk_quantize_for_gemm_direct'):
        raise RuntimeError("localCTA direct quant helper is unavailable for v4 raw wgrad")
    dy = dy_bf16 if dy_bf16.dtype == torch.bfloat16 and dy_bf16.is_contiguous() else dy_bf16.to(torch.bfloat16).contiguous()
    x = x_bf16 if x_bf16.dtype == torch.bfloat16 and x_bf16.is_contiguous() else x_bf16.to(torch.bfloat16).contiguous()
    dy_quant = tkq.tk_quantize_for_gemm_direct(dy, True, dy_encode_centric)
    x_quant = tkq.tk_quantize_for_gemm_direct(x, True, x_encode_centric)
    a_fp4, a_sc = dy_quant[2], dy_quant[3]
    a_sg = dy_quant[5] if len(dy_quant) > 5 and torch.is_tensor(dy_quant[5]) and dy_quant[5].numel() > 0 else dy_quant[4]
    b_fp4, b_sc = x_quant[2], x_quant[3]
    b_sg = x_quant[5] if len(x_quant) > 5 and torch.is_tensor(x_quant[5]) and x_quant[5].numel() > 0 else x_quant[4]
    tk_direct.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dW)


def tk_v4_direct_split_wgrad_col_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dW):
    tk_direct = _get_tk_localcta_direct()
    if tk_direct is None or not hasattr(tk_direct, 'nvfp4_gemm'):
        raise RuntimeError("localCTA direct GEMM backend is unavailable for v4 split wgrad")
    a_sg = _prepare_localcta_v4_split_wgrad_a_sg_for_direct(
        a_sg, a_fp4.size(0) // 256, a_fp4.device
    )
    if hasattr(tk_direct, 'prepare_outer_sg'):
        b_sg = tk_direct.prepare_outer_sg(b_sg, b_fp4.size(0) // 256, False)
    else:
        b_sg = _prepare_localcta_v4_outer_sg_for_direct(
            b_sg, b_fp4.size(0) // 256, b_fp4.device
        )
    tk_direct.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dW)


# ---------------------------------------------------------------------------
# Grouped GEMM wrappers — pre-concatenated TK tensors + b_sg from kernel
# ---------------------------------------------------------------------------
def tk_grouped_forward_gemm(x_nvfp4, wc_fp4, wc_sc, b_sg_per_tile, N_dims, out=None, use_localcta=None):
    """Grouped forward GEMM with pre-built TK weight tensors.

    Args:
        x_nvfp4: _TKQuantized input (M, K)
        wc_fp4: concatenated weight FP4 (N_total, K/2) fp4x2
        wc_sc:  concatenated weight scales (total_ntm_r, ntk_r, 512) fp8
        b_sg_per_tile: per-tile B_sg (total_fwd_tiles,) float32
        N_dims: list of int — output sizes per group
        out: optional pre-allocated output (M, N_total)

    Returns:
        list of (M, N_g) bf16 tensors, one per group
    """
    if use_localcta is None:
        use_localcta = use_tk_localcta()
    tk = _get_tk() if use_localcta else _get_tk_plain()
    x_fp4, x_sc, x_sg = x_nvfp4._tk_row
    M = x_nvfp4.shape[0]
    K = x_nvfp4.shape[1]
    N_total = sum(N_dims)
    if use_localcta:
        b_sg_per_tile = _normalize_localcta_v3_tilegrid_b_sg(wc_fp4, b_sg_per_tile)
    if out is None:
        out = torch.empty(M, N_total, dtype=torch.bfloat16, device=x_fp4.device)
    if use_tk_localcta() and use_tk_localcta_direct_contract():
        if torch.is_tensor(x_sg) and x_sg.dim() != 2:
            x_sg = _localcta_expand_sg_grid(x_sg, M, K)
        if torch.is_tensor(b_sg_per_tile) and b_sg_per_tile.dim() != 2:
            b_sg_per_tile = _localcta_group_sg_grid_from_scalars(
                b_sg_per_tile, N_dims, K, x_fp4.device
            )
    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_fast_grouped_forward()
        and hasattr(tk, 'nvfp4_grouped_gemm_fast')
    ):
        tk.nvfp4_grouped_gemm_fast(
            x_fp4, x_sc, x_sg,
            wc_fp4, wc_sc, b_sg_per_tile,
            out,
        )
    else:
        tk.nvfp4_grouped_gemm(x_fp4, x_sc, x_sg, wc_fp4, wc_sc, b_sg_per_tile, out)
    return list(torch.split(out, N_dims, dim=1))


def tk_grouped_forward_gemm_split(x_nvfp4, wc_fp4, wc_sc, b_sg_per_tile, N_dims, outs=None, use_localcta=None):
    """Grouped forward GEMM that writes each split directly to its own bf16 output."""
    if use_localcta is None:
        use_localcta = use_tk_localcta()
    tk = _get_tk() if use_localcta else _get_tk_plain()
    x_fp4, x_sc, x_sg = x_nvfp4._tk_row
    M = x_nvfp4.shape[0]
    K = x_nvfp4.shape[1]
    if use_localcta:
        b_sg_per_tile = _normalize_localcta_v3_tilegrid_b_sg(wc_fp4, b_sg_per_tile)

    if len(N_dims) not in (2, 3):
        raise ValueError(f"split output path expects 2 or 3 groups, got {len(N_dims)}")

    if outs is None:
        outs = [
            torch.empty(M, n_dim, dtype=torch.bfloat16, device=x_fp4.device)
            for n_dim in N_dims
        ]
    if len(outs) != len(N_dims):
        raise ValueError("outs must have the same length as N_dims")

    if use_tk_localcta() and use_tk_localcta_direct_contract():
        if torch.is_tensor(x_sg) and x_sg.dim() != 2:
            x_sg = _localcta_expand_sg_grid(x_sg, M, K)
        if torch.is_tensor(b_sg_per_tile) and b_sg_per_tile.dim() != 2:
            b_sg_per_tile = _localcta_group_sg_grid_from_scalars(
                b_sg_per_tile, N_dims, K, x_fp4.device
            )

    if (
        use_localcta
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_fast_grouped_forward()
        and hasattr(tk, 'nvfp4_grouped_gemm_fast')
    ):
        if len(N_dims) == 2:
            tk.nvfp4_grouped_gemm_fast(
                x_fp4, x_sc, x_sg, wc_fp4, wc_sc, b_sg_per_tile,
                outs[0], outs[1]
            )
        else:
            tk.nvfp4_grouped_gemm_fast(
                x_fp4, x_sc, x_sg, wc_fp4, wc_sc, b_sg_per_tile,
                outs[0], outs[1], outs[2]
            )
    else:
        if len(N_dims) == 2:
            tk.nvfp4_grouped_gemm(
                x_fp4, x_sc, x_sg, wc_fp4, wc_sc, b_sg_per_tile,
                outs[0], outs[1]
            )
        else:
            tk.nvfp4_grouped_gemm(
                x_fp4, x_sc, x_sg, wc_fp4, wc_sc, b_sg_per_tile,
                outs[0], outs[1], outs[2]
            )
    return outs


def tk_grouped_dgrad_gemm(dy_nvfp4, wc_fp4_cols, wc_sc_cols, sg_cat, N_dims):
    """Per-split dgrad GEMM: dx = sum_i( dY_i @ W_i^T ).

    Uses the fused C++ function nvfp4_split_dgrad_sum to perform all
    split slicing + GEMMs + accumulation in a single C++ call,
    eliminating Python loop overhead.

    Args:
        dy_nvfp4: _TKQuantized gradient (M, N_total)
        wc_fp4_cols: list of per-split col FP4 (K, M_i/2) fp4x2
        wc_sc_cols:  list of per-split col scales (ntm_c, ntk_c_i, 512) fp8
        sg_cat: per-split sg values (n_splits,) float32
        N_dims: list of int — output dim per split [q_dim, k_dim, v_dim]

    Returns:
        dx: (M, K) bf16 tensor
    """
    if use_tk_localcta():
        raise RuntimeError("tk_grouped_dgrad_gemm is not used in localCTA mode; use tk_grouped_k_dgrad_gemm")

    tk = _get_tk()
    a_fp4, a_sc, a_sg = dy_nvfp4._tk_row  # (M, N_total)
    M = dy_nvfp4.shape[0]
    K = wc_fp4_cols[0].size(0)  # all col FP4 have K rows

    dx = torch.empty(M, K, dtype=torch.bfloat16, device=a_fp4.device)
    tk.nvfp4_split_dgrad_sum(
        a_fp4, a_sc, a_sg,
        wc_fp4_cols, wc_sc_cols, sg_cat,
        N_dims, dx
    )
    return dx


# ---------------------------------------------------------------------------
# Grouped dim-1 quant + grouped-K GEMM for QKV backward dgrad
# ---------------------------------------------------------------------------
_tk_quant_mod_cache = None
_tk_quant_standalone_mod_cache = None


class _LocalCTAQuantAdapter:
    """Normalize localCTA quant entrypoints to the generic TK quant API."""

    def __init__(self, mod):
        self._mod = mod
        self.is_localcta = True
        self.variant = get_tk_localcta_variant()
        self._silu_deriv_tile_amax_cache = {}
        self._silu_deriv_tile_amax_cache_age = {}
        self._silu_deriv_tile_amax_work_cache = {}
        self._silu_deriv_tile_amax_work_next_slot = {}

    def _make_silu_deriv_tile_amax_work_slot(self, dh, H):
        alloc = getattr(
            self._mod,
            'tk_localcta_silu_deriv_quantize_split_for_gemm_alloc',
            None,
        )
        if alloc is None:
            return None
        bufs = tuple(alloc(int(dh.shape[0]), int(H), dh.device))
        return (
            torch.empty_like(dh),
            torch.empty_like(dh),
            bufs,
            torch.empty_like(bufs[12]),
            torch.empty_like(bufs[14]),
            torch.empty_like(bufs[4]),
            torch.empty_like(bufs[5]),
            torch.empty_like(bufs[10]),
            torch.empty_like(bufs[11]),
        )

    @staticmethod
    def _slot_aliases_silu_deriv_tile_amax_state(slot, prev_state):
        if slot is None or not prev_state:
            return False
        prev_ptrs = {
            int(t.data_ptr())
            for t in prev_state
            if hasattr(t, 'data_ptr') and t.numel() > 0
        }
        if not prev_ptrs:
            return False
        return any(
            int(t.data_ptr()) in prev_ptrs
            for t in slot[3:]
            if hasattr(t, 'data_ptr') and t.numel() > 0
        )

    def _get_silu_deriv_tile_amax_work_slot(self, cache_key, dh, H, prev_state):
        slots = self._silu_deriv_tile_amax_work_cache.get(cache_key)
        if slots is None:
            slots = []
            for _ in range(2):
                slot = self._make_silu_deriv_tile_amax_work_slot(dh, H)
                if slot is None:
                    return None
                slots.append(slot)
            self._silu_deriv_tile_amax_work_cache[cache_key] = slots
            self._silu_deriv_tile_amax_work_next_slot[cache_key] = 0

        next_slot = self._silu_deriv_tile_amax_work_next_slot.get(cache_key, 0)
        for offset in range(len(slots)):
            slot_index = (next_slot + offset) % len(slots)
            slot = slots[slot_index]
            if not self._slot_aliases_silu_deriv_tile_amax_state(slot, prev_state):
                self._silu_deriv_tile_amax_work_next_slot[cache_key] = slot_index + 1
                return slot

        slot = self._make_silu_deriv_tile_amax_work_slot(dh, H)
        if slot is None:
            return None
        slots.append(slot)
        self._silu_deriv_tile_amax_work_next_slot[cache_key] = len(slots)
        return slot

    @staticmethod
    def _alloc_localcta_dim1_bundle(mod, input, split_sections):
        M, N_total = input.shape
        device = input.device
        fp4_row_full, row_sc_prepared_full, fp4_col_full, col_sc_prepared_full, row_sg_full, col_sg_full = (
            mod.tk_localcta_quantize_for_gemm_prepared_alloc(
                M, N_total, True, device
            )
        )
        sg_per_group = torch.ones(len(split_sections), dtype=torch.float32, device=device)
        row_sc_prepared_list = []
        col_fp4_list = []
        col_sc_prepared_list = []
        sc_offset = 0
        sg_offset = 0
        col_offset = 0
        for cols_i in split_sections:
            row_sc_prepared_list.append(
                row_sc_prepared_full.narrow(1, sc_offset, cols_i // 64)
            )
            col_fp4_list.append(
                fp4_col_full.narrow(0, col_offset, cols_i)
            )
            col_sc_prepared_list.append(
                col_sc_prepared_full.narrow(0, sg_offset, cols_i // 128)
            )
            sc_offset += cols_i // 64
            sg_offset += cols_i // 128
            col_offset += cols_i
        dummy = torch.empty(0, dtype=torch.uint8, device=device)
        return (
            fp4_row_full, fp4_col_full, sg_per_group,
            row_sc_prepared_full, row_sg_full, col_sc_prepared_full, col_sg_full,
            row_sc_prepared_list, col_fp4_list, col_sc_prepared_list, dummy,
        )

    @staticmethod
    def _launch_localcta_dim1_bundle(mod, input, split_sections,
                                     fp4_row_full, fp4_col_full, sg_per_group,
                                     row_sc_prepared_full, row_sg_full,
                                     col_sc_prepared_full, col_sg_full,
                                     row_sc_prepared_list, col_fp4_list, col_sc_prepared_list,
                                     _tma_host_buf,
                                     skip_cat=False):
        del skip_cat
        mod.tk_localcta_quantize_for_gemm_prepared_launch(
            input, True, True,
            fp4_row_full, row_sc_prepared_full,
            fp4_col_full, col_sc_prepared_full,
            row_sg_full, col_sg_full,
        )
        row_fp4_list = []
        col_offset = 0
        for cols_i in split_sections:
            row_fp4_list.append(fp4_row_full.narrow(1, col_offset // 2, cols_i // 2))
            col_offset += cols_i
        return (
            row_fp4_list, row_sc_prepared_list, sg_per_group,
            col_fp4_list, col_sc_prepared_list,
            fp4_row_full, row_sc_prepared_full, fp4_col_full, col_sc_prepared_full,
        )

    @staticmethod
    def _normalize_group_quantize_outer_sg_result(result):
        """Normalize v4 direct grouped quantize to the prepared-path tuple shape."""
        if len(result) >= 12:
            row_fp4_list, row_sc_list, row_sg_list, \
                col_fp4_list, col_sc_list, col_sg_list, \
                row_fp4_cat, row_sc_cat, row_sg_cat, \
                col_fp4_cat, col_sc_cat, col_sg_cat = result[:12]
            return (
                row_fp4_cat, row_sc_cat, row_sg_cat,
                col_fp4_list, col_sc_list, col_sg_cat,
                row_sg_list, col_sg_list,
                col_fp4_cat, col_sc_cat,
            )
        return result

    def tk_quantize_for_gemm(self, input, return_transpose=True, encode_centric=True):
        if use_tk_localcta_direct_contract():
            return _standalone_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        if self.variant == 'v3':
            return self._mod.tk_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        return self._mod.tk_localcta_quantize_for_gemm(
            input, return_transpose, encode_centric
        )

    def tk_quantize_for_gemm_padded(
        self,
        input,
        output_rows,
        output_cols,
        return_transpose=True,
        encode_centric=True,
    ):
        if not hasattr(self._mod, "tk_localcta_quantize_for_gemm_padded"):
            raise AttributeError(
                "localCTA quant module is missing its native padded producer"
            )
        return self._mod.tk_localcta_quantize_for_gemm_padded(
            input,
            output_rows,
            output_cols,
            return_transpose,
            encode_centric,
        )

    def tk_quantize_nhsd_wo_for_gemm(self, input, encode_centric=True):
        return self._mod.tk_localcta_quantize_nhsd_wo_for_gemm(
            input, encode_centric
        )

    def tk_quantize_for_gemm_fast(self, input, return_transpose=True, encode_centric=True):
        return self._mod.tk_localcta_quantize_for_gemm_fast(
            input, return_transpose, encode_centric
        )

    def tk_quantize_weight_2d(self, input):
        if not hasattr(self._mod, "tk_localcta_quantize_weight_2d"):
            raise AttributeError(
                "localCTA quant module is missing its native 2D weight producer"
            )
        return self._mod.tk_localcta_quantize_weight_2d(input)

    def tk_quantize_for_gemm_row_prepared_col_outer(
        self,
        input,
        return_transpose=True,
        encode_centric=True,
    ):
        return self._mod.tk_localcta_quantize_for_gemm_row_prepared_col_outer(
            input, return_transpose, encode_centric
        )

    def tk_quantize_for_gemm_raw_outer_tma(
        self,
        input,
        return_transpose=True,
        encode_centric=True,
    ):
        if not hasattr(self._mod, 'tk_localcta_quantize_for_gemm_raw_outer_tma'):
            raise AttributeError('localCTA quant module is missing raw-outer TMA quantizer')
        return self._mod.tk_localcta_quantize_for_gemm_raw_outer_tma(
            input, return_transpose, encode_centric
        )

    def tk_quantize_for_gemm_opt(self, input, return_transpose=True, encode_centric=True,
                                 data_stochastic_rounding=False,
                                 scale_stochastic_rounding=False,
                                 rht_axes="none",
                                 with_random_sign_mask=False,
                                 rng_seed=0,
                                 rng_subsequence_base=0,
                                 data_sr_axes="both",
                                 persistent_rng_state=None):
        if not hasattr(self._mod, 'tk_localcta_quantize_for_gemm_opt'):
            raise AttributeError('localCTA quant module is missing native v4 opt quantizer')
        args = (
            input, return_transpose, encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
            data_sr_axes,
        )
        return self._mod.tk_localcta_quantize_for_gemm_opt(
            *args,
            *(() if persistent_rng_state is None else (persistent_rng_state,)),
        )

    def tk_quantize_for_gemm_final_sg_opt(
        self,
        input,
        return_transpose=True,
        encode_centric=True,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        rht_axes="none",
        with_random_sign_mask=False,
        rng_seed=0,
        rng_subsequence_base=0,
    ):
        if not hasattr(self._mod, 'tk_localcta_quantize_for_gemm_final_sg_opt'):
            raise AttributeError(
                'localCTA quant module is missing native v4 final-SG quantizer'
            )
        return self._mod.tk_localcta_quantize_for_gemm_final_sg_opt(
            input,
            return_transpose,
            encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_rmsnorm_to_bf16(self, input, gamma, epsilon):
        if not hasattr(self._mod, 'tk_localcta_rmsnorm_to_bf16'):
            raise AttributeError(
                'localCTA quant module is missing native v4 BF16 RMSNorm producer'
            )
        return self._mod.tk_localcta_rmsnorm_to_bf16(
            input, gamma, float(epsilon)
        )

    def tk_rmsnorm_quantize_for_gemm_opt(self, input, gamma, epsilon,
                                         return_transpose=True, encode_centric=True,
                                         data_stochastic_rounding=False,
                                         scale_stochastic_rounding=False,
                                         rht_axes="none",
                                         with_random_sign_mask=False,
                                         rng_seed=0,
                                         rng_subsequence_base=0):
        if not hasattr(self._mod, 'tk_localcta_rmsnorm_quantize_for_gemm_opt'):
            raise AttributeError('localCTA quant module is missing native v4 RMSNorm opt quantizer')
        return self._mod.tk_localcta_rmsnorm_quantize_for_gemm_opt(
            input, gamma, float(epsilon), return_transpose, encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_rmsnorm_quantize_for_gemm_final_sg_opt(self, input, gamma, epsilon,
                                                  return_transpose=True, encode_centric=True,
                                                  data_stochastic_rounding=False,
                                                  scale_stochastic_rounding=False,
                                                  rht_axes="none",
                                                  with_random_sign_mask=False,
                                                  rng_seed=0,
                                                  rng_subsequence_base=0):
        if not hasattr(self._mod, 'tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt'):
            raise AttributeError('localCTA quant module is missing native v4 RMSNorm final-SG opt quantizer')
        return self._mod.tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt(
            input, gamma, float(epsilon), return_transpose, encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_rmsnorm_quantize_from_row_rms_partial_final_sg(
        self,
        input,
        gamma,
        row_rms_partial,
        epsilon,
        return_transpose=True,
        encode_centric=True,
    ):
        name = 'tk_localcta_rmsnorm_quantize_from_row_rms_partial_final_sg'
        if not hasattr(self._mod, name):
            raise AttributeError(
                'localCTA quant module is missing the native exact C/D/E '
                'final-SG quantizer'
            )
        return getattr(self._mod, name)(
            input,
            gamma,
            row_rms_partial,
            float(epsilon),
            return_transpose,
            encode_centric,
        )

    def tk_rmsnorm_quantize_for_gemm_row_prepared_col_outer(
        self,
        input,
        gamma,
        epsilon,
        return_transpose=True,
        encode_centric=True,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        rht_axes="none",
        with_random_sign_mask=False,
        rng_seed=0,
        rng_subsequence_base=0,
    ):
        if not hasattr(self._mod, 'tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer'):
            raise AttributeError(
                'localCTA quant module is missing native v4 RMSNorm row-prepared/col-outer quantizer'
            )
        return self._mod.tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer(
            input,
            gamma,
            float(epsilon),
            return_transpose,
            encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_localcta_sqrelu_quantize_for_gemm_prepared(self, input, encode_centric=True,
                                                      data_stochastic_rounding=False,
                                                      scale_stochastic_rounding=False,
                                                      rht_axes="none",
                                                      with_random_sign_mask=False,
                                                      rng_seed=0,
                                                      rng_subsequence_base=0):
        if not hasattr(self._mod, 'tk_localcta_sqrelu_quantize_for_gemm_prepared'):
            raise AttributeError('localCTA quant module is missing square-ReLU prepared quantizer')
        return self._mod.tk_localcta_sqrelu_quantize_for_gemm_prepared(
            input,
            encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer(
        self,
        input,
        encode_centric=True,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        rht_axes="none",
        with_random_sign_mask=False,
        rng_seed=0,
        rng_subsequence_base=0,
    ):
        if not hasattr(self._mod, 'tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer'):
            raise AttributeError('localCTA quant module is missing square-ReLU row-prepared/col-outer quantizer')
        return self._mod.tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer(
            input,
            encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_localcta_sqrelu_quantize_row_only_prepared(self, input, encode_centric=True,
                                                      data_stochastic_rounding=False,
                                                      scale_stochastic_rounding=False,
                                                      rht_axes="none",
                                                      with_random_sign_mask=False,
                                                      rng_seed=0,
                                                      rng_subsequence_base=0):
        if not hasattr(self._mod, 'tk_localcta_sqrelu_quantize_row_only_prepared'):
            raise AttributeError('localCTA quant module is missing square-ReLU row-only quantizer')
        return self._mod.tk_localcta_sqrelu_quantize_row_only_prepared(
            input,
            encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_localcta_sqrelu_quantize_col_only_raw_outer(self, input, encode_centric=True):
        if not hasattr(self._mod, 'tk_localcta_sqrelu_quantize_col_only_raw_outer'):
            raise AttributeError('localCTA quant module is missing square-ReLU col-only quantizer')
        return self._mod.tk_localcta_sqrelu_quantize_col_only_raw_outer(
            input,
            encode_centric,
        )

    def tk_localcta_sqrelu_deriv_quantize_for_gemm_prepared(self, dh, h1_raw, encode_centric=True,
                                                            data_stochastic_rounding=False,
                                                            scale_stochastic_rounding=False,
                                                            rht_axes="none",
                                                            with_random_sign_mask=False,
                                                            rng_seed=0,
                                                            rng_subsequence_base=0):
        if not hasattr(self._mod, 'tk_localcta_sqrelu_deriv_quantize_for_gemm_prepared'):
            raise AttributeError('localCTA quant module is missing square-ReLU derivative prepared quantizer')
        return self._mod.tk_localcta_sqrelu_deriv_quantize_for_gemm_prepared(
            dh,
            h1_raw,
            encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer(self, dh, h1_raw, encode_centric=True,
                                                             data_stochastic_rounding=False,
                                                             scale_stochastic_rounding=False,
                                                             rht_axes="none",
                                                             with_random_sign_mask=False,
                                                             rng_seed=0,
                                                             rng_subsequence_base=0):
        if not hasattr(self._mod, 'tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer'):
            raise AttributeError('localCTA quant module is missing square-ReLU derivative raw-outer quantizer')
        return self._mod.tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer(
            dh,
            h1_raw,
            encode_centric,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rht_axes,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
        )

    def tk_quantize_for_gemm_direct(self, input, return_transpose=True, encode_centric=True):
        if use_tk_localcta_direct_contract():
            return _standalone_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        if (
            self.variant == 'v4'
            and use_tk_localcta_v4_final_sg_producer()
            and hasattr(self._mod, 'tk_localcta_quantize_for_gemm_final_sg')
        ):
            return self._mod.tk_localcta_quantize_for_gemm_final_sg(
                input, return_transpose, encode_centric
            )
        if use_tk_localcta_v4_strict_path() and hasattr(self._mod, 'tk_localcta_quantize_for_gemm'):
            return self._mod.tk_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        return self._mod.tk_localcta_quantize_for_gemm(
            input, return_transpose, encode_centric
        )

    def tk_quantize_for_gemm_direct_forward(self, input, return_transpose=True, encode_centric=True):
        if self.variant == 'v3':
            return self._mod.tk_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        return self.tk_quantize_for_gemm_direct(input, return_transpose, encode_centric)

    def _direct_forward_producer(self):
        """Name the producer selected by ``tk_quantize_for_gemm_direct_forward``."""
        if self.variant == 'v3':
            return 'v3'
        if use_tk_localcta_direct_contract():
            return 'standalone'
        if (
            self.variant == 'v4'
            and use_tk_localcta_v4_final_sg_producer()
            and hasattr(self._mod, 'tk_localcta_quantize_for_gemm_final_sg')
        ):
            return 'legacy_final_sg'
        return 'public_selector'

    def _paired_col_rht_direct_forward_symbol(self):
        contract = _localcta_v3_contract().strip().lower()
        if self.variant != 'v4' or contract in {'tilegrid256', 'tilegrid', '2d'}:
            return None
        producer = self._direct_forward_producer()
        if producer == 'legacy_final_sg':
            symbol = 'tk_localcta_quantize_for_gemm_final_sg_paired_col_rht'
            return symbol if hasattr(self._mod, symbol) else None
        if producer == 'public_selector' and use_tk_localcta_v4_atomic_final_sg_producer():
            symbol = 'tk_localcta_quantize_for_gemm_atomic_paired_col_rht'
            return symbol if hasattr(self._mod, symbol) else None
        return None

    def supports_paired_col_rht_direct_forward(self):
        """Whether one native call preserves the exact selected row producer."""
        return self._paired_col_rht_direct_forward_symbol() is not None

    def tk_quantize_for_gemm_direct_forward_paired_col_rht(
        self,
        input,
        return_transpose=True,
        encode_centric=True,
    ):
        if not return_transpose:
            raise RuntimeError(
                "paired column-RHT QKV producer requires return_transpose=True"
            )
        symbol = self._paired_col_rht_direct_forward_symbol()
        if symbol is None:
            raise RuntimeError(
                "native paired column-RHT QKV producer requires localCTA v4, "
                "the outer-scale contract, and a route-matched fused extension symbol"
            )
        return getattr(self._mod, symbol)(
            input,
            return_transpose,
            encode_centric,
        )

    def tk_quantize_for_gemm_maybe_borrow(self, input, staging_input,
                                          return_transpose=True, encode_centric=True):
        if use_tk_localcta_direct_contract():
            del staging_input
            return _standalone_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        if self.variant == 'v3':
            del staging_input
            return self._mod.tk_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        if not input.is_contiguous():
            staging_input.copy_(input)
            input = staging_input
        return self._mod.tk_localcta_quantize_for_gemm(
            input, return_transpose, encode_centric
        )

    def tk_quantize_for_gemm_prepared_nofold_maybe_borrow(self, input, staging_input,
                                                          return_transpose=True, encode_centric=True):
        if use_tk_localcta_direct_contract():
            del staging_input
            return _standalone_localcta_quantize_for_gemm(
                input, return_transpose, encode_centric
            )
        quantize = getattr(
            self._mod,
            'tk_localcta_quantize_for_gemm_prepared_nofold_maybe_borrow',
            None,
        )
        if quantize is None:
            quantize = getattr(
                self._mod,
                'tk_localcta_quantize_for_gemm_prepared_maybe_borrow',
                None,
            )
        if quantize is None:
            raise AttributeError('localCTA quant module is missing prepared maybe-borrow entrypoint')
        return quantize(
            input, staging_input, return_transpose, encode_centric
        )

    def tk_fused_norm_quantize(self, input, gamma, epsilon, with_silu=False,
                               return_transpose=True, encode_centric=True):
        try:
            result = self._mod.tk_localcta_fused_norm_quantize(
                input, gamma, epsilon, with_silu, return_transpose, encode_centric
            )
        except TypeError as exc:
            # Older localCTA pybinds do not expose the encode-centric argument.
            if "incompatible function arguments" not in str(exc):
                raise
            result = self._mod.tk_localcta_fused_norm_quantize(
                input, gamma, epsilon, with_silu, return_transpose
            )
        return result

    def tk_quantize_col_only_prepared(self, input, sg_tensor):
        if self.variant == 'v3':
            raise NotImplementedError("localCTA v3 does not support folded prepared col-only quantization")
        return self._mod.tk_localcta_quantize_col_only_prepared(input, sg_tensor)

    def tk_group_quantize_for_gemm(self, input, split_sections):
        if use_tk_localcta_direct_contract():
            return _adapt_standalone_group_quant_for_localcta(
                _as_bf16_contiguous(input), split_sections
            )
        if self.variant == 'v3':
            result = self._mod.tk_localcta_group_quantize_for_gemm(input, split_sections)
            if len(result) >= 10:
                row_fp4_cat, row_sc_cat, row_sg_cat, \
                    col_fp4_list, col_sc_list, col_sg_cat, \
                    row_sg_parts, col_sg_list, \
                    col_fp4_cat, col_sc_cat = result[:10]
            else:
                row_fp4_cat, row_sc_cat, row_sg_cat, \
                    col_fp4_list, col_sc_list, col_sg_cat, \
                    row_sg_parts, col_sg_list = result
                col_fp4_cat = torch.cat(
                    [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in col_fp4_list], dim=1
                ).view(torch.float4_e2m1fn_x2)
                col_sc_cat = torch.cat(
                    [sc.contiguous().view(torch.uint8) for sc in col_sc_list], dim=1
                ).view(torch.float8_e4m3fn)
            fwd_sg = col_sg_cat if _use_localcta_v3_tilegrid256() else row_sg_cat
            return (
                row_fp4_cat, row_sc_cat, fwd_sg,
                col_fp4_list, col_sc_list, col_sg_cat,
                row_sg_parts, col_sg_list,
                col_fp4_cat, col_sc_cat,
            )
        return self.tk_group_quantize_for_gemm_direct(input, split_sections)

    def tk_group_quantize_for_gemm_fast(self, input, split_sections):
        return self._mod.tk_localcta_group_quantize_for_gemm_fast(
            input, split_sections
        )

    def tk_group_quantize_for_gemm_direct(self, input, split_sections):
        if use_tk_localcta_direct_contract():
            return _adapt_standalone_group_quant_for_localcta(
                _as_bf16_contiguous(input), split_sections
            )
        if self.variant == 'v3':
            return self.tk_group_quantize_for_gemm(input, split_sections)
        if (
            self.variant == 'v4'
            and use_tk_localcta_v4_final_sg_producer()
            and hasattr(self._mod, 'tk_localcta_group_quantize_for_gemm_final_sg')
        ):
            return self._normalize_group_quantize_outer_sg_result(
                self._mod.tk_localcta_group_quantize_for_gemm_final_sg(
                    input, split_sections
                )
            )
        return self._normalize_group_quantize_outer_sg_result(
            self._mod.tk_localcta_group_quantize_for_gemm(
                input, split_sections
            )
        )

    def tk_set_global_scale_num(self, value: float):
        return self._mod.tk_localcta_set_global_scale_num(float(value))

    def tk_get_global_scale_num(self) -> float:
        return float(self._mod.tk_localcta_get_global_scale_num())

    def tk_reset_global_scale_num(self):
        return self._mod.tk_localcta_reset_global_scale_num()

    def tk_group_quantize_for_gemm_v2(self, input, split_sections):
        if use_tk_localcta_direct_contract():
            return _adapt_standalone_group_quant_for_localcta(
                _as_bf16_contiguous(input), split_sections
            )
        if self.variant == 'v3':
            return self.tk_group_quantize_for_gemm(input, split_sections)
        return self.tk_group_quantize_for_gemm_direct(input, split_sections)

    def tk_group_quantize_split_for_gemm_v2(self, input0, input1):
        if use_tk_localcta_direct_contract():
            input_cat = torch.cat([input0, input1], dim=0)
            split_sections = [input0.shape[0], input1.shape[0]]
            return _adapt_standalone_group_quant_for_localcta(
                _as_bf16_contiguous(input_cat), split_sections
            )
        if (
            self.variant == 'v4'
            and use_tk_localcta_v4_final_sg_producer()
            and hasattr(self._mod, 'tk_localcta_group_quantize_split2_for_gemm_final_sg')
        ):
            return self._normalize_group_quantize_outer_sg_result(
                self._mod.tk_localcta_group_quantize_split2_for_gemm_final_sg(
                    _as_bf16_contiguous(input0),
                    _as_bf16_contiguous(input1),
                )
            )
        if self.variant == 'v3':
            result = self.tk_batched_quantize_for_gemm(
                [_as_bf16_contiguous(input0), _as_bf16_contiguous(input1)],
                True,
                True,
            )
            if len(result) >= 9:
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs, \
                    col_fp4_full, col_sc_full, col_sg_full = result[:9]
            else:
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs = result
                col_fp4_full = col_sc_full = col_sg_full = None

            row_fp4_cat = torch.cat(
                [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in row_fp4s], dim=0
            ).view(torch.float4_e2m1fn_x2)
            row_sc_cat = torch.cat(
                [sc.contiguous().view(torch.uint8) for sc in row_scs], dim=0
            ).view(torch.float8_e4m3fn)
            row_sg_cat = torch.cat(row_sgs, dim=0)

            if col_sgs and col_sgs[0].dim() == 2 and col_sgs[0].size(0) == 1:
                col_sg_cat = torch.cat(col_sgs, dim=1)
            else:
                col_sg_cat = torch.cat(col_sgs, dim=0)

            if col_fp4_full is not None and col_sc_full is not None:
                return (
                    row_fp4_cat, row_sc_cat, row_sg_cat,
                    col_fp4s, col_scs, col_sg_cat,
                    row_sgs, col_sgs,
                    col_fp4_full, col_sc_full,
                )
            return (
                row_fp4_cat, row_sc_cat, row_sg_cat,
                col_fp4s, col_scs, col_sg_cat,
                row_sgs, col_sgs,
            )
        return self.tk_group_quantize_for_gemm_direct(
            torch.cat([_as_bf16_contiguous(input0), _as_bf16_contiguous(input1)], dim=0),
            [int(input0.shape[0]), int(input1.shape[0])],
        )

    def tk_silu_quantize_split_for_gemm(self, h1_raw, h3):
        if self.variant == 'v3':
            return self._mod.tk_localcta_silu_quantize_split_for_gemm(h1_raw, h3)
        return self._mod.tk_localcta_silu_quantize_split_for_gemm(h1_raw, h3)

    def supports_silu_paired_col_rht(self, h1_raw=None, h3=None):
        """Whether W2 can produce its preserved row and paired RHT col at once."""
        marker = getattr(
            self._mod,
            'tk_localcta_silu_supports_paired_col_rht',
            None,
        )
        contract = _localcta_v3_contract().strip().lower()
        supported = (
            self.variant == 'v4'
            and contract not in {'tilegrid256', 'tilegrid', '2d'}
            and use_tk_localcta_v4_silu_atomic_final_sg_producer()
            and not use_tk_localcta_v4_gemm_virtual_rescale()
            and _env_flag('USE_TK_LOCALCTA_V4_FUSED_SILU_RAW', True)
            and not _env_flag('NVTE_NVFP4_ENCODE_CENTRIC', False)
            and callable(marker)
            and bool(marker())
            and hasattr(
                self._mod,
                'tk_localcta_silu_quantize_split_for_gemm_paired_col_rht',
            )
        )
        if not supported:
            return False
        if hasattr(h1_raw, 'shape') and hasattr(h3, 'shape'):
            h1_shape = tuple(h1_raw.shape)
            h3_shape = tuple(h3.shape)
            return (
                h1_shape == h3_shape
                and len(h1_shape) == 2
                and h1_shape[0] % 256 == 0
                and h1_shape[1] % 256 == 0
            )
        return True

    def tk_silu_quantize_split_for_gemm_paired_col_rht(self, h1_raw, h3):
        if not self.supports_silu_paired_col_rht(h1_raw, h3):
            raise RuntimeError(
                "native paired column-RHT W2 producer requires localCTA v4, "
                "the outer-scale contract, atomic final-SG, fused raw SiLU, "
                "decode-centric columns, virtual rescale off, 256-aligned shapes, "
                "and its fused extension symbols"
            )
        return self._mod.tk_localcta_silu_quantize_split_for_gemm_paired_col_rht(
            h1_raw,
            h3,
        )

    @staticmethod
    def _localcta_tile_delayed_refresh_interval() -> int:
        if (
            _env_flag('USE_TK_FFN_LOCALCTA_DELAYED_NO_COLLECT', False)
            or _env_flag('USE_TK_FFN_H13_TILE_DELAYED_NO_COLLECT', False)
        ):
            return 0
        value = os.environ.get(
            'USE_TK_FFN_LOCALCTA_DELAYED_REFRESH_INTERVAL',
            os.environ.get('USE_TK_FFN_H13_TILE_DELAYED_REFRESH_INTERVAL', '1'),
        )
        try:
            return max(0, int(value))
        except ValueError:
            return 1

    def _tk_silu_deriv_quantize_split_for_gemm_impl(
        self,
        dh,
        h3,
        h1_raw,
        use_delayed_scaling=False,
        state_key=None,
    ):
        dh = _as_bf16_contiguous(dh)
        h3 = _as_bf16_contiguous(h3)
        h1_raw = _as_bf16_contiguous(h1_raw)
        H = int(dh.shape[1])
        if use_delayed_scaling:
            use_tile_delayed = (
                _env_flag('USE_TK_FFN_LOCALCTA_TILE_DELAYED_AMAX', False)
                or _env_flag('USE_TK_FFN_H13_TILE_DELAYED_AMAX', False)
            )
            if use_tile_delayed:
                use_split_collect = (
                    _env_flag('USE_TK_FFN_LOCALCTA_DELAYED_SPLIT_COLLECT', False)
                    or _env_flag('USE_TK_FFN_H13_TILE_DELAYED_SPLIT_COLLECT', False)
                )
                collect = getattr(
                    self._mod,
                    'tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax_outer',
                    None,
                )
                tile_delayed = getattr(
                    self._mod,
                    'tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_split_collect'
                    if use_split_collect
                    else 'tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer',
                    None,
                )
                tile_delayed_no_collect = getattr(
                    self._mod,
                    'tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect',
                    None,
                )
                tile_delayed_inplace = (
                    None if use_split_collect else getattr(
                        self._mod,
                        'tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_launch_inplace',
                        None,
                    )
                )
                tile_delayed_no_collect_inplace = getattr(
                    self._mod,
                    'tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect_launch_inplace',
                    None,
                )
                cache_slice = slice(12, 20)
                if collect is None or tile_delayed is None:
                    collect = getattr(
                        self._mod,
                        'tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax',
                        None,
                    )
                    tile_delayed = getattr(
                        self._mod,
                        'tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed',
                        None,
                    )
                    tile_delayed_no_collect = None
                    tile_delayed_inplace = None
                    tile_delayed_no_collect_inplace = None
                    cache_slice = slice(12, 16)
                if self.variant != 'v4' or collect is None or tile_delayed is None:
                    raise NotImplementedError(
                        "localCTA tile-delayed SiLU-deriv scaling requires the v4 quant "
                        "extension with tile-amax collect and delayed entrypoints"
                    )
                device_index = dh.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                cache_key = (
                    state_key,
                    int(device_index),
                    int(dh.shape[0]),
                    int(H),
                )
                prev_tile_amax = self._silu_deriv_tile_amax_cache.get(cache_key)
                if prev_tile_amax is None:
                    result = collect(dh, h3, h1_raw)
                    next_age = 0
                else:
                    refresh_interval = self._localcta_tile_delayed_refresh_interval()
                    age = self._silu_deriv_tile_amax_cache_age.get(cache_key, 0)
                    use_no_collect = (
                        tile_delayed_no_collect is not None
                        and (refresh_interval == 0 or (refresh_interval > 1 and age < refresh_interval - 1))
                    )
                    work_slot = None
                    if (
                        cache_slice.stop == 20
                        and _env_flag(
                            'USE_TK_FFN_LOCALCTA_DELAYED_PREALLOC',
                            _env_flag('USE_TK_FFN_H13_TILE_DELAYED_PREALLOC', True),
                        )
                        and (
                            (use_no_collect and tile_delayed_no_collect_inplace is not None)
                            or ((not use_no_collect) and tile_delayed_inplace is not None)
                        )
                    ):
                        work_slot = self._get_silu_deriv_tile_amax_work_slot(
                            cache_key, dh, H, prev_tile_amax
                        )
                    if use_no_collect:
                        if work_slot is not None and tile_delayed_no_collect_inplace is not None:
                            dh1, dh3_out, bufs = work_slot[:3]
                            result = tile_delayed_no_collect_inplace(
                                dh, h3, h1_raw, dh1, dh3_out, *bufs, *prev_tile_amax
                            )
                        else:
                            result = tile_delayed_no_collect(dh, h3, h1_raw, *prev_tile_amax)
                        next_age = age + 1
                    else:
                        if work_slot is not None and tile_delayed_inplace is not None:
                            dh1, dh3_out, bufs = work_slot[:3]
                            cur_row_amax_0, cur_row_amax_1 = work_slot[3:5]
                            cur_row_sg_0, cur_col_sg_0, cur_row_sg_1, cur_col_sg_1 = work_slot[5:9]
                            result = tile_delayed_inplace(
                                dh, h3, h1_raw, dh1, dh3_out, *bufs, *prev_tile_amax,
                                cur_row_amax_0, cur_row_amax_1,
                                cur_row_sg_0, cur_col_sg_0, cur_row_sg_1, cur_col_sg_1
                            )
                        else:
                            result = tile_delayed(dh, h3, h1_raw, *prev_tile_amax)
                        next_age = 0
                if len(result) >= 16:
                    self._silu_deriv_tile_amax_cache[cache_key] = tuple(
                        t.detach() for t in result[cache_slice]
                    )
                    self._silu_deriv_tile_amax_cache_age[cache_key] = next_age
                return result
            delayed = getattr(
                self._mod,
                'tk_localcta_silu_deriv_quantize_split_for_gemm_delayed',
                None,
            )
            if self.variant != 'v4' or delayed is None:
                raise NotImplementedError(
                    "localCTA delayed SiLU-deriv scaling requires the v4 quant extension "
                    "with tk_localcta_silu_deriv_quantize_split_for_gemm_delayed"
                )
            return delayed(dh, h3, h1_raw)
        return self._mod.tk_localcta_silu_deriv_quantize_split_for_gemm(dh, h3, h1_raw)

    def tk_silu_deriv_quantize_for_gemm(
        self,
        dh,
        h13,
        H,
        use_delayed_scaling=False,
        state_key=None,
    ):
        h1_raw = h13[:, :H]
        h3 = h13[:, H:]
        return self._tk_silu_deriv_quantize_split_for_gemm_impl(
            dh, h3, h1_raw, use_delayed_scaling, state_key
        )

    def tk_silu_deriv_quantize_split_for_gemm(self, dh, h3, h1_raw, state_key=None):
        use_delayed_scaling = (
            _env_flag('USE_TK_FFN_LOCALCTA_DELAYED_SPLIT', False)
            or _env_flag('USE_TK_FFN_LOCALCTA_TILE_DELAYED_AMAX', False)
            or _env_flag('USE_TK_FFN_H13_TILE_DELAYED_AMAX', False)
        )
        return self._tk_silu_deriv_quantize_split_for_gemm_impl(
            dh, h3, h1_raw, use_delayed_scaling, state_key
        )

    def tk_group_quantize_dim1_for_gemm(self, input, split_sections):
        if use_tk_localcta_direct_contract():
            tk_q = _get_tk_quant_standalone()
            result = tk_q.tk_group_quantize_dim1_for_gemm(
                _as_bf16_contiguous(input), split_sections
            )
            return _adapt_standalone_dim1_quant_for_localcta(
                result, split_sections, input.shape[0], input.device
            )
        if self.variant == 'v3':
            return self._mod.tk_localcta_group_quantize_dim1_for_gemm(
                input, split_sections
            )
        bundle = self._alloc_localcta_dim1_bundle(self._mod, input, split_sections)
        return self._launch_localcta_dim1_bundle(
            self._mod, input, split_sections, *bundle
        )

    def tk_group_quantize_dim1_alloc(self, input, split_sections):
        if self.variant == 'v3':
            raise NotImplementedError("localCTA v3 dim1 alloc/launch helper is not wired yet")
        return self._alloc_localcta_dim1_bundle(self._mod, input, split_sections)

    def tk_group_quantize_dim1_launch(self, input, split_sections,
                                      fp4_row_full, fp4_col_full, sg_per_group,
                                      amax_tensor, sync_tensor, psync_tensor,
                                      tma_host_buf, tma_dev_buf,
                                      sc_row_allocs, fp4_col_allocs, sc_col_allocs,
                                      skip_cat=False):
        if self.variant == 'v3':
            raise NotImplementedError("localCTA v3 dim1 alloc/launch helper is not wired yet")
        return self._launch_localcta_dim1_bundle(
            self._mod, input, split_sections,
            fp4_row_full, fp4_col_full, sg_per_group,
            amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
            sc_row_allocs, fp4_col_allocs, sc_col_allocs, tma_host_buf,
            skip_cat=skip_cat,
        )

    def _tk_localcta_v4_group_quantize_dim1_split3_for_gemm(
        self, input0, input1, input2, persistent_rng_state=None
    ):
        data_sr, data_sr_axes, rng_seed, rng_subsequence = (
            _plain_qkv_split3_sr_policy()
        )
        args = (
            input0,
            input1,
            input2,
            data_sr,
            rng_seed,
            rng_subsequence,
            data_sr_axes,
        )
        return self._mod.tk_localcta_group_quantize_dim1_split3_for_gemm(
            *args,
            *(() if persistent_rng_state is None else (persistent_rng_state,)),
        )

    def tk_concat_group_quantize_dim1_for_gemm(self, input, split_sections):
        if use_tk_localcta_direct_contract():
            adapted = self.tk_group_quantize_dim1_for_gemm(input, split_sections)
            row_fp4_list, row_sc_list, row_sg_list, \
                col_fp4_list, col_sc_list, col_sg_list, \
                a_fp4_full, _a_sc_cat, _a_sg_full, \
                col_fp4_full, col_sc_cat, col_sg_cat = adapted
            if a_fp4_full is None:
                a_fp4_full = torch.cat(
                    [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in row_fp4_list], dim=1
                ).view(torch.float4_e2m1fn_x2)
            return (
                a_fp4_full,
                row_sc_list,
                row_sg_list,
                col_fp4_list,
                col_sc_list,
                col_sg_list,
                col_fp4_full,
                col_sc_cat,
                col_sg_cat,
            )
        if self.variant == 'v3':
            raise NotImplementedError("localCTA v3 concat prepared dim1 helper is not wired yet")
        split_sections = [int(s) for s in split_sections]
        if len(set(split_sections)) != 1:
            if len(split_sections) == 3 and hasattr(self._mod, 'tk_localcta_group_quantize_dim1_split3_for_gemm'):
                g0, g1, g2 = [
                    _as_bf16_contiguous(part)
                    for part in torch.split(_as_bf16_contiguous(input), split_sections, dim=1)
                ]
                return self._tk_localcta_v4_group_quantize_dim1_split3_for_gemm(
                    g0, g1, g2
                )
            return self.tk_group_quantize_dim1_for_gemm(input, split_sections)
        # Native v4 contract: keep raw fp8 microscales + fp32 SG separate and
        # let the C++ fast consumers apply SG in the epilogue. The legacy
        # prepared path folds SG into e4m3 too early and reproduces the old
        # underflow mode on low-amplitude backward tensors.
        if hasattr(self._mod, 'tk_localcta_group_quantize_dim1_for_gemm_fast'):
            return self._mod.tk_localcta_group_quantize_dim1_for_gemm_fast(
                input, split_sections
            )
        return self._mod.tk_localcta_concat_group_quantize_dim1_for_gemm_prepared(
            input, split_sections
        )

    def tk_concat_group_quantize_dim1_for_gemm_fast(self, input, split_sections):
        if self.variant == 'v3':
            raise NotImplementedError("localCTA v3 concat fast dim1 helper is not wired yet")
        return self._mod.tk_localcta_group_quantize_dim1_for_gemm_fast(
            input, split_sections
        )

    def tk_group_quantize_dim1_split3_for_gemm(
        self, input0, input1, input2, persistent_rng_state=None
    ):
        if use_tk_localcta_direct_contract():
            if persistent_rng_state is not None:
                raise RuntimeError(
                    "checkpointed localCTA SR is unsupported by the standalone "
                    "direct-contract QKV quantizer"
                )
            tk_q = _get_tk_quant_standalone()
            g0 = _as_bf16_contiguous(input0)
            g1 = _as_bf16_contiguous(input1)
            g2 = _as_bf16_contiguous(input2)
            if hasattr(tk_q, 'tk_group_quantize_dim1_split3_for_gemm'):
                result = _plain_qkv_split3_quantize_eager(tk_q, g0, g1, g2)
            else:
                result = tk_q.tk_group_quantize_dim1_for_gemm(
                    torch.cat([g0, g1, g2], dim=1),
                    [g0.shape[1], g1.shape[1], g2.shape[1]],
                )
            return _adapt_standalone_dim1_quant_for_localcta(
                result,
                [g0.shape[1], g1.shape[1], g2.shape[1]],
                g0.shape[0],
                g0.device,
            )
        if self.variant == 'v3':
            if persistent_rng_state is not None:
                raise RuntimeError(
                    "checkpointed localCTA SR requires the v4 split3 ABI"
                )
            result = self._mod.tk_localcta_group_quantize_dim1_split3_for_gemm(
                input0, input1, input2
            )
            if len(result) == 12:
                return result
            row_fp4s, row_scs, row_sgs, \
                col_fp4s, col_scs, col_sgs, \
                col_fp4_cat, col_sc_cat_raw, col_sg_cat_raw = result
            return (
                row_fp4s, row_scs, row_sgs,
                col_fp4s, col_scs, col_sgs,
                col_fp4_cat, col_sc_cat_raw, col_sg_cat_raw,
            )
        if use_tk_qkv_localcta_raw_dim1_quant():
            g0 = _as_bf16_contiguous(input0)
            g1 = _as_bf16_contiguous(input1)
            g2 = _as_bf16_contiguous(input2)
            raw_result = self._tk_localcta_v4_group_quantize_dim1_split3_for_gemm(
                g0, g1, g2, persistent_rng_state
            )
            return _adapt_raw_localcta_dim1_quant_for_fast(
                raw_result,
                [g0.shape[1], g1.shape[1], g2.shape[1]],
                g0.shape[0],
                g0.device,
            )
        if hasattr(self._mod, 'tk_localcta_group_quantize_dim1_split3_for_gemm'):
            return self._tk_localcta_v4_group_quantize_dim1_split3_for_gemm(
                input0, input1, input2, persistent_rng_state
            )
        if persistent_rng_state is not None:
            raise RuntimeError(
                "localCTA extension lacks the checkpointed v4 split3 SR ABI"
            )
        return self._mod.tk_localcta_group_quantize_dim1_split3_for_gemm_prepared(
            input0, input1, input2
        )

    def tk_localcta_split3_supports_paired_rht(self) -> bool:
        marker = getattr(
            self._mod, "tk_localcta_split3_supports_paired_rht", None
        )
        return bool(marker is not None and marker())

    def tk_group_quantize_dim1_split3_for_gemm_paired_rht(
        self,
        input0,
        input1,
        input2,
        *,
        data_stochastic_rounding,
        rng_seed,
        rng_subsequence_base,
        data_sr_axes,
        persistent_rng_state,
        encode_centric,
    ):
        if not self.tk_localcta_split3_supports_paired_rht():
            raise RuntimeError(
                "localCTA extension does not advertise native paired split3 RHT"
            )
        return self._mod.tk_localcta_group_quantize_dim1_split3_for_gemm(
            input0,
            input1,
            input2,
            data_stochastic_rounding,
            rng_seed,
            rng_subsequence_base,
            data_sr_axes,
            persistent_rng_state,
            "col",
            True,
            encode_centric,
        )

    def tk_group_quantize_dim1_split3_for_gemm_inverse_rope_live64(
        self, input0, input1, input2, rope_cs, rope_seq_len,
        persistent_rng_state=None,
    ):
        if self.variant == 'v3':
            raise NotImplementedError("split3 inverse RoPE is only wired for localCTA v4")
        if not hasattr(self._mod, 'tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64'):
            raise AttributeError(
                "tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64 "
                "is unavailable in this localCTA extension"
            )
        data_sr, data_sr_axes, rng_seed, rng_subsequence = (
            _plain_qkv_split3_sr_policy()
        )
        args = (
            input0,
            input1,
            input2,
            rope_cs,
            int(rope_seq_len),
            data_sr,
            False,
            "none",
            False,
            rng_seed,
            rng_subsequence,
            data_sr_axes,
        )
        return self._mod.tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64(
            *args,
            *(() if persistent_rng_state is None else (persistent_rng_state,)),
        )

    def tk_group_quantize_dim1_split3_rowphase_for_gemm(self, input0, input1, input2):
        if self.variant != 'v3':
            raise NotImplementedError("split3 rowphase is only wired for localCTA v3")
        return self._mod.tk_localcta_group_quantize_dim1_split3_rowphase_for_gemm(
            input0, input1, input2
        )

    def tk_group_quantize_dim1_split3_finalize_col_inplace(
        self, col_sc_cat, col_sg_cat, col_sg_chunk_0, col_sg_chunk_1, col_sg_chunk_2
    ):
        if self.variant != 'v3':
            raise NotImplementedError("split3 deferred col finalize is only wired for localCTA v3")
        return self._mod.tk_localcta_group_quantize_dim1_split3_finalize_col_inplace(
            col_sc_cat, col_sg_cat, col_sg_chunk_0, col_sg_chunk_1, col_sg_chunk_2
        )

    def tk_batched_quantize_for_gemm(self, inputs, return_transpose=True, encode_centric=True):
        if use_tk_localcta_direct_contract():
            tk_q = _get_tk_quant_standalone()
            row_fp4_list, row_sc_list = [], []
            col_fp4_list, col_sc_list = [], []
            row_sg_list, col_sg_list = [], []
            for input in inputs:
                result = tk_q.tk_quantize_for_gemm(
                    _as_bf16_contiguous(input),
                    return_transpose,
                    encode_centric,
                )
                row_fp4_list.append(result[0])
                row_sc_list.append(result[1])
                col_fp4_list.append(result[2])
                col_sc_list.append(result[3])
                row_sg_list.append(result[4].to(torch.float32))
                col_sg_list.append(
                    result[5].to(torch.float32)
                    if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0
                    else result[4].to(torch.float32)
                )
            return (
                row_fp4_list, row_sc_list,
                col_fp4_list, col_sc_list,
                row_sg_list, col_sg_list,
            )
        if self.variant == 'v3':
            result = self._mod.tk_localcta_batched_quantize_for_gemm(
                inputs, return_transpose, encode_centric
            )
            if len(result) >= 9:
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs, \
                    col_fp4_full, col_sc_full, col_sg_full = result[:9]
                return (
                    row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs,
                    col_fp4_full, col_sc_full, col_sg_full,
                )
            row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs = result
            return row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs
        if use_tk_localcta_v4_strict_path() and hasattr(self._mod, 'tk_localcta_batched_quantize_for_gemm'):
            result = self._mod.tk_localcta_batched_quantize_for_gemm(
                inputs, return_transpose, encode_centric
            )
            if len(result) >= 9:
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs, \
                    col_fp4_full, col_sc_full, col_sg_full = result[:9]
                return (
                    row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs,
                    col_fp4_full, col_sc_full, col_sg_full,
                )
            return result
        if hasattr(self._mod, 'tk_localcta_batched_quantize_for_gemm'):
            result = self._mod.tk_localcta_batched_quantize_for_gemm(
                inputs, return_transpose, encode_centric
            )
            if len(result) >= 9:
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs, \
                    col_fp4_full, col_sc_full, col_sg_full = result[:9]
                return (
                    row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs,
                    col_fp4_full, col_sc_full, col_sg_full,
                )
            return result
        return self._mod.tk_localcta_batched_quantize_for_gemm_prepared(
            inputs, return_transpose, encode_centric
        )

    def tk_silu_deriv_quantize_for_gemm_alloc(self, M, H, device):
        if self.variant == 'v3':
            raise NotImplementedError("localCTA v3 split2 prepared helper is not wired yet")
        dummy = torch.empty(0, dtype=torch.uint8, device=device)
        out1 = self._mod.tk_localcta_quantize_for_gemm_prepared_alloc(M, H, True, device)
        out2 = self._mod.tk_localcta_quantize_for_gemm_prepared_alloc(M, H, True, device)
        h13_buf = torch.empty(M, 2 * H, dtype=torch.bfloat16, device=device)
        return (
            out1[0], out1[1], out1[2], out1[3],
            out2[0], out2[1], out2[2], out2[3],
            torch.empty(2, dtype=torch.float32, device=device),
            dummy, dummy, dummy,
            dummy, dummy, h13_buf, dummy, dummy,
        )

    def tk_silu_deriv_quantize_for_gemm_launch(self, dh, h13, H,
                                               out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
                                               out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
                                               sg_buf, amax_buf, sync_buf, psync_buf,
                                               fp4_row_full, fp4_col_full, dh13_bf16,
                                               tma_host_buf, tma_dev_buf):
        if self.variant == 'v3':
            raise NotImplementedError("localCTA v3 split2 prepared helper is not wired yet")
        del amax_buf, sync_buf, psync_buf, fp4_row_full, fp4_col_full, tma_host_buf, tma_dev_buf
        dh13_bf16.copy_(h13)
        result = self.tk_silu_deriv_quantize_for_gemm(dh, dh13_bf16, H)
        out1_fp4.view(torch.uint8).copy_(result[0].view(torch.uint8))
        out1_sc.copy_(result[1])
        out1_fp4_t.view(torch.uint8).copy_(result[2].view(torch.uint8))
        out1_sc_t.copy_(result[3])
        out2_fp4.view(torch.uint8).copy_(result[6].view(torch.uint8))
        out2_sc.copy_(result[7])
        out2_fp4_t.view(torch.uint8).copy_(result[8].view(torch.uint8))
        out2_sc_t.copy_(result[9])
        sg_buf[0:1].copy_(result[4].view(-1).to(torch.float32)[:1])
        sg_buf[1:2].copy_(result[10].view(-1).to(torch.float32)[:1])
        return (
            out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
            sg_buf.narrow(0, 0, 1), torch.zeros(1, dtype=torch.float32, device=dh.device),
            out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
            sg_buf.narrow(0, 1, 1), torch.zeros(1, dtype=torch.float32, device=dh.device),
        )


def _is_localcta_quant_mod(tkq) -> bool:
    return bool(getattr(tkq, 'is_localcta', False))

def _get_tk_quant_for_gemm():
    """Lazy-load TK standalone quant module (prefers v5 with fused-amax dim=1)."""
    global _tk_quant_mod_cache, _tk_backend_info
    if _tk_quant_mod_cache is None:
        import sys
        _apply_localcta_v3_perf_defaults()
        if use_tk_localcta():
            variant = get_tk_localcta_variant()
            variant_specs = {
                'v1': ('nvfp4_CTA_local_v1', '_tk_quant_localcta'),
                'v2': ('nvfp4_CTA_local_v2', '_tk_quant_localcta_v2'),
                'v3': ('nvfp4_CTA_local_v3', '_tk_quant_localcta_v3'),
                'v4': ('nvfp4_CTA_local_v4', '_tk_quant_localcta_v4'),
            }
            variant_dir, module_name = variant_specs[variant]
            tk_quant_dir = os.path.join(_fp4_matmul_root(), 'TK_quantisation', variant_dir)
            alt = os.path.join(
                _LEGACY_FP4_MATMUL_ROOT,
                'TK_quantisation',
                variant_dir,
            )
            found_dir = None
            candidates = [tk_quant_dir, alt]
            _prepend_import_paths_in_priority_order(candidates)
            for d in candidates:
                if os.path.isdir(d):
                    found_dir = d
                    break
            if found_dir is None:
                raise FileNotFoundError(
                    f"Requested localCTA quant variant {variant!r}, but {variant_dir} "
                    f"was not found under {tk_quant_dir} or {alt}"
                )
            _mod = __import__(module_name)
            _maybe_apply_localcta_quant_tuning(_mod)
            logger.info("[TK QUANT] Loaded %s quant module from %s", variant_dir, found_dir)
            _tk_quant_mod_cache = _LocalCTAQuantAdapter(_mod)
            _tk_backend_info["quant"] = _artifact_info(getattr(_mod, "__file__", None))
            _tk_backend_info["quant_module_name"] = module_name
            _tk_backend_info["quant_variant_dir"] = variant_dir
        else:
            tk_quant_dir = os.path.join(_fp4_matmul_root(), 'TK_quantisation', 'nvfp4')
            tk_quant_v5_dir = os.path.join(_fp4_matmul_root(), 'TK_quantisation', 'nvfp4_v5')
            alt = os.path.join(
                _LEGACY_FP4_MATMUL_ROOT, 'TK_quantisation', 'nvfp4'
            )
            alt_v5 = os.path.join(
                _LEGACY_FP4_MATMUL_ROOT, 'TK_quantisation', 'nvfp4_v5'
            )
            _prepend_import_paths_in_priority_order(
                (tk_quant_v5_dir, alt_v5, tk_quant_dir, alt)
            )
            try:
                import _tk_quant_v5 as _mod
                logger.info("[TK QUANT] Loaded v5 (persistent fused-amax dim=1)")
            except ImportError:
                import _tk_quant as _mod
                logger.info("[TK QUANT] Loaded v3 (fallback)")
            _tk_quant_mod_cache = _mod
            _tk_backend_info["quant"] = _artifact_info(getattr(_mod, "__file__", None))
            _tk_backend_info["quant_module_name"] = getattr(_mod, "__name__", None)
    return _maybe_wrap_v5_ffn_quantizer(_tk_quant_mod_cache)


def _get_tk_quant_standalone():
    """Load the standalone TK quant module regardless of localCTA mode."""
    global _tk_quant_standalone_mod_cache
    if _tk_quant_standalone_mod_cache is not None:
        return _tk_quant_standalone_mod_cache

    import sys

    tk_quant_dir = os.path.join(_fp4_matmul_root(), 'TK_quantisation', 'nvfp4')
    tk_quant_v5_dir = os.path.join(_fp4_matmul_root(), 'TK_quantisation', 'nvfp4_v5')
    alt = os.path.join(
        _LEGACY_FP4_MATMUL_ROOT, 'TK_quantisation', 'nvfp4'
    )
    alt_v5 = os.path.join(
        _LEGACY_FP4_MATMUL_ROOT, 'TK_quantisation', 'nvfp4_v5'
    )
    _prepend_import_paths_in_priority_order(
        (tk_quant_v5_dir, alt_v5, tk_quant_dir, alt)
    )
    try:
        import _tk_quant_v5 as _mod
    except ImportError:
        import _tk_quant as _mod
    _tk_quant_standalone_mod_cache = _mod
    return _tk_quant_standalone_mod_cache


def _normalize_localcta_sg_scalar_list(sg_values, split_sections, device):
    """Normalize grouped SG metadata to one float32 scalar tensor per split."""
    if torch.is_tensor(sg_values):
        flat = sg_values.to(torch.float32).reshape(-1)
        if flat.numel() == len(split_sections):
            return [flat[i:i + 1] for i in range(len(split_sections))]
        if flat.numel() == 1 and len(split_sections) == 1:
            return [flat]

    normalized = []
    for sg in sg_values:
        if torch.is_tensor(sg):
            normalized.append(sg.to(torch.float32).reshape(-1)[:1])
        else:
            normalized.append(torch.tensor([float(sg)], dtype=torch.float32, device=device))
    if len(normalized) != len(split_sections):
        raise RuntimeError(
            f"expected {len(split_sections)} SG values, got {len(normalized)}"
        )
    return normalized


def _localcta_expand_sg_grid(sg, rows, cols):
    """Expand a scalar or vector SG payload into the 2D grid expected by direct GEMMs."""
    if not torch.is_tensor(sg):
        raise TypeError(f"expected tensor sg, got {type(sg)!r}")
    sg = sg.to(torch.float32)
    if sg.dim() == 2:
        return sg
    rows_tiles = rows // 128
    cols_tiles = cols // 128
    flat = sg.reshape(-1)
    if flat.numel() == 1:
        return torch.full(
            (rows_tiles, cols_tiles),
            float(flat[0].item()),
            dtype=torch.float32,
            device=sg.device,
        )
    if flat.numel() == rows_tiles * cols_tiles:
        return flat.reshape(rows_tiles, cols_tiles).contiguous()
    if flat.numel() == rows_tiles:
        return flat.view(rows_tiles, 1).expand(rows_tiles, cols_tiles).contiguous()
    if flat.numel() == cols_tiles:
        return flat.view(1, cols_tiles).expand(rows_tiles, cols_tiles).contiguous()
    raise RuntimeError(
        f"cannot expand SG payload with {flat.numel()} values to grid {(rows_tiles, cols_tiles)}"
    )


def _localcta_group_sg_grid_from_scalars(sg_values, split_sections, cols, device):
    """Build the row-oriented grouped SG grid for direct localCTA grouped GEMMs."""
    sg_list = _normalize_localcta_sg_scalar_list(sg_values, split_sections, device)
    return torch.cat(
        [_localcta_expand_sg_grid(sg, rows, cols) for sg, rows in zip(sg_list, split_sections)],
        dim=0,
    )


def _adapt_standalone_group_quant_for_localcta(input, split_sections):
    """Map standalone TK grouped-weight quantization onto the localCTA grouped contract."""
    tk_q = _get_tk_quant_standalone()
    if hasattr(tk_q, 'tk_group_quantize_for_gemm_v2'):
        result = tk_q.tk_group_quantize_for_gemm_v2(input, split_sections)
    else:
        result = tk_q.tk_group_quantize_for_gemm(input, split_sections)

    row_fp4, row_sc, _fwd_b_sg, col_fp4_list, col_sc_list, _dgrad_b_sg, sg_cat, keepalive = result[:8]
    sg_list = _normalize_localcta_sg_scalar_list(sg_cat, split_sections, input.device)
    fwd_b_sg = _localcta_group_sg_grid_from_scalars(
        sg_list, split_sections, input.shape[1], input.device
    )
    col_sg_cat = torch.cat(sg_list, dim=0)
    return (
        row_fp4,
        row_sc,
        fwd_b_sg,
        col_fp4_list,
        col_sc_list,
        col_sg_cat,
        keepalive,
        sg_list,
    )


def _fold_sg_into_localcta_prepared_sc(sc_raw, sg, rows, cols):
    """Fold a TK SG payload into the FP8 microscale tensor expected by fast localCTA GEMMs."""
    sg_grid = _localcta_expand_sg_grid(sg, rows, cols)
    sg_prepared = sg_grid.repeat_interleave(2, dim=1).unsqueeze(-1)
    sc_prepared = (sc_raw.float() * sg_prepared).contiguous()
    if use_tk_localcta_prepared_sc_clamp_tiny():
        sc_prepared = sc_prepared.clamp_min(torch.finfo(torch.float8_e4m3fn).tiny)
    return sc_prepared.to(torch.float8_e4m3fn)


def _adapt_standalone_dim1_quant_for_localcta(result, split_sections, M, device):
    """Map standalone TK dim1 grouped quantization onto the localCTA split package."""
    if len(result) >= 9:
        row_fp4_list, row_sc_list, sg_per_group, \
            col_fp4_list, col_sc_list, \
            a_fp4_full, a_sc_cat, col_fp4_full, col_sc_cat = result[:9]
    else:
        row_fp4_list, row_sc_list, sg_per_group, col_fp4_list, col_sc_list = result[:5]
        a_fp4_full = a_sc_cat = col_fp4_full = col_sc_cat = None

    sg_list = _normalize_localcta_sg_scalar_list(sg_per_group, split_sections, device)
    if col_fp4_full is None:
        col_fp4_full = torch.cat(
            [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in col_fp4_list], dim=0
        ).view(torch.float4_e2m1fn_x2)
    if col_sc_cat is None:
        col_sc_cat = torch.cat(
            [sc.contiguous().view(torch.uint8) for sc in col_sc_list], dim=0
        ).view(torch.float8_e4m3fn)
    col_sg_cat = torch.cat(sg_list, dim=0)
    return (
        row_fp4_list,
        row_sc_list,
        sg_list,
        col_fp4_list,
        col_sc_list,
        sg_list,
        a_fp4_full,
        a_sc_cat,
        None,
        col_fp4_full,
        col_sc_cat,
        col_sg_cat,
    )


def _adapt_standalone_dim1_quant_for_localcta_fast(result, split_sections, M, device):
    """Map standalone TK dim1 quantization onto fast localCTA prepared payloads."""
    if len(result) >= 9:
        row_fp4_list, row_sc_raw_list, sg_per_group, \
            col_fp4_list, col_sc_raw_list, \
            a_fp4_full, _a_sc_cat_raw, col_fp4_full, _col_sc_cat_raw = result[:9]
    else:
        row_fp4_list, row_sc_raw_list, sg_per_group, col_fp4_list, col_sc_raw_list = result[:5]
        a_fp4_full = col_fp4_full = None

    sg_list = _normalize_localcta_sg_scalar_list(sg_per_group, split_sections, device)
    row_sc_list = [
        _fold_sg_into_localcta_prepared_sc(sc_raw, sg, M, cols)
        for sc_raw, sg, cols in zip(row_sc_raw_list, sg_list, split_sections)
    ]
    col_sc_list = [
        _fold_sg_into_localcta_prepared_sc(sc_raw, sg, cols, M)
        for sc_raw, sg, cols in zip(col_sc_raw_list, sg_list, split_sections)
    ]

    if a_fp4_full is None:
        a_fp4_full = torch.cat(
            [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in row_fp4_list], dim=1
        ).view(torch.float4_e2m1fn_x2)
    a_sc_cat = torch.cat(
        [sc.contiguous().view(torch.uint8) for sc in row_sc_list], dim=1
    ).view(torch.float8_e4m3fn)

    if col_fp4_full is None:
        col_fp4_full = torch.cat(
            [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in col_fp4_list], dim=0
        ).view(torch.float4_e2m1fn_x2)
    col_sc_cat = torch.cat(
        [sc.contiguous().view(torch.uint8) for sc in col_sc_list], dim=0
    ).view(torch.float8_e4m3fn)

    sg_unit_list = [
        torch.ones(1, dtype=torch.float32, device=device)
        for _ in split_sections
    ]
    col_sg_cat = torch.ones(len(split_sections), dtype=torch.float32, device=device)
    return (
        row_fp4_list,
        row_sc_list,
        sg_unit_list,
        col_fp4_list,
        col_sc_list,
        sg_unit_list,
        a_fp4_full,
        a_sc_cat,
        None,
        col_fp4_full,
        col_sc_cat,
        col_sg_cat,
    )


def _adapt_raw_localcta_dim1_quant_for_fast(result, split_sections, M, device):
    """Fold raw localCTA SG payloads into prepared scales for fast localCTA GEMMs."""
    if len(result) != 9:
        raise RuntimeError(
            "raw localCTA split3 dim1 quant must return 9 items, "
            f"got {len(result)}"
        )
    row_fp4_list, row_sc_raw_list, row_sg_list, \
        col_fp4_list, col_sc_raw_list, col_sg_list, \
        col_fp4_full, _col_sc_cat_raw, _col_sg_cat_raw = result

    row_sg_list = [
        sg.to(torch.float32) if sg.dtype != torch.float32 else sg
        for sg in row_sg_list
    ]
    col_sg_list = [
        sg.to(torch.float32) if sg.dtype != torch.float32 else sg
        for sg in col_sg_list
    ]

    row_sc_list = [
        _fold_sg_into_localcta_prepared_sc(sc_raw, sg, M, cols)
        for sc_raw, sg, cols in zip(row_sc_raw_list, row_sg_list, split_sections)
    ]
    col_sc_list = [
        _fold_sg_into_localcta_prepared_sc(sc_raw, sg, cols, M)
        for sc_raw, sg, cols in zip(col_sc_raw_list, col_sg_list, split_sections)
    ]

    row_fp4_full = torch.cat(
        [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in row_fp4_list], dim=1
    ).view(torch.float4_e2m1fn_x2)
    row_sc_cat = torch.cat(
        [sc.contiguous().view(torch.uint8) for sc in row_sc_list], dim=1
    ).view(torch.float8_e4m3fn)
    col_sc_cat = torch.cat(
        [sc.contiguous().view(torch.uint8) for sc in col_sc_list], dim=0
    ).view(torch.float8_e4m3fn)

    sg_unit_list = [
        torch.ones(1, dtype=torch.float32, device=device)
        for _ in split_sections
    ]
    col_sg_cat = torch.ones(len(split_sections), dtype=torch.float32, device=device)
    return (
        row_fp4_list,
        row_sc_list,
        sg_unit_list,
        col_fp4_list,
        col_sc_list,
        sg_unit_list,
        row_fp4_full,
        row_sc_cat,
        None,
        col_fp4_full,
        col_sc_cat,
        col_sg_cat,
    )


def _standalone_localcta_quantize_for_gemm(input, return_transpose=True, encode_centric=True):
    """Use the standalone TK quantizer while keeping the localCTA adapter surface."""
    tk_q = _get_tk_quant_standalone()
    return tk_q.tk_quantize_for_gemm(
        _as_bf16_contiguous(input),
        return_transpose,
        encode_centric,
    )



# Module-level cache for dgrad D buffers to avoid per-call allocation
_dgrad_buf_cache = {}
_weight_split_cache = {}  # key: data_ptr -> (B_fp4_list, B_sc_list, B_sg_list)
_weight_sg_split_cache = {}
_weight_split_outer_fold_cache = {}


def _sg_cache_sig(sg):
    if torch.is_tensor(sg):
        return ('tensor', sg.data_ptr(), tuple(sg.shape), str(sg.dtype))
    if isinstance(sg, (list, tuple)):
        return (
            type(sg).__name__,
            tuple(_sg_cache_sig(x) for x in sg),
        )
    return (type(sg).__name__, repr(sg))


def _split_weight_col_sg_tensors(w_fp4_c, w_sg_c, N_dims, use_localcta_runtime=None):
    """Split only the colwise SG payload without touching FP4/FP8 scale tensors."""
    if use_localcta_runtime is None:
        use_localcta_runtime = bool(use_tk_localcta())
    key = (
        w_fp4_c.data_ptr(),
        tuple(N_dims),
        bool(use_localcta_runtime),
        get_tk_localcta_variant() if use_localcta_runtime else None,
        _sg_cache_sig(w_sg_c),
    )
    cached = _weight_sg_split_cache.get(key)
    if cached is not None:
        return cached

    B_sg_list = []
    offset_sg = 0
    localcta_v3 = _use_localcta_v3_runtime()
    localcta_v3_tilegrid = _use_localcta_v3_tilegrid256()
    if use_localcta_runtime:
        if localcta_v3 and isinstance(w_sg_c, (list, tuple)) and len(w_sg_c) != len(N_dims):
            raise ValueError(
                f"localCTA v3 expected {len(N_dims)} per-split col SG tensors, got {len(w_sg_c)}"
            )
        if (not localcta_v3) and isinstance(w_sg_c, (list, tuple)) and len(w_sg_c) != len(N_dims):
            raise ValueError(
                f"localCTA v4 expected {len(N_dims)} per-split col SG tensors, got {len(w_sg_c)}"
            )
        for i, n_i in enumerate(N_dims):
            if localcta_v3:
                if isinstance(w_sg_c, (list, tuple)):
                    B_sg_list.append(w_sg_c[i].contiguous())
                elif w_sg_c.dim() == 1:
                    sg_tiles_i = n_i // (256 if localcta_v3_tilegrid else 128)
                    B_sg_list.append(w_sg_c[offset_sg:offset_sg + sg_tiles_i].contiguous())
                    offset_sg += sg_tiles_i
                elif w_sg_c.dim() == 2 and w_sg_c.size(0) == 1:
                    B_sg_list.append(w_sg_c.contiguous())
                else:
                    sg_tiles_i = n_i // (256 if localcta_v3_tilegrid else 128)
                    sg_view = w_sg_c[:, offset_sg:offset_sg + sg_tiles_i]
                    if localcta_v3_tilegrid:
                        sg_view = sg_view.transpose(0, 1)
                    B_sg_list.append(sg_view.contiguous())
                    offset_sg += sg_tiles_i
            else:
                k_outer_tiles = w_fp4_c.size(0) // 256
                if isinstance(w_sg_c, (list, tuple)):
                    B_sg_list.append(w_sg_c[i].contiguous())
                elif w_sg_c.dim() == 1 and w_sg_c.numel() == k_outer_tiles * len(N_dims):
                    start = i * k_outer_tiles
                    B_sg_list.append(w_sg_c[start:start + k_outer_tiles].contiguous())
                elif w_sg_c.dim() == 2 and tuple(w_sg_c.shape) == (len(N_dims), k_outer_tiles):
                    B_sg_list.append(w_sg_c[i, :].contiguous())
                elif w_sg_c.dim() == 2 and tuple(w_sg_c.shape) == (k_outer_tiles, len(N_dims)):
                    B_sg_list.append(w_sg_c[:, i].contiguous())
                elif w_sg_c.dim() == 1 and w_sg_c.numel() == k_outer_tiles:
                    B_sg_list.append(w_sg_c.contiguous())
                elif (
                    w_sg_c.dim() == 2
                    and (
                        tuple(w_sg_c.shape) == (1, k_outer_tiles)
                        or tuple(w_sg_c.shape) == (k_outer_tiles, 1)
                    )
                ):
                    B_sg_list.append(w_sg_c.contiguous())
                else:
                    sg_tiles_i = n_i // 128
                    if w_sg_c.dim() == 1:
                        B_sg_list.append(w_sg_c[offset_sg:offset_sg + sg_tiles_i].contiguous())
                    else:
                        B_sg_list.append(w_sg_c[:, offset_sg:offset_sg + sg_tiles_i].contiguous())
                    offset_sg += sg_tiles_i
    else:
        for i, _ in enumerate(N_dims):
            B_sg_list.append(w_sg_c[i:i + 1].to(torch.float32))

    cached = B_sg_list
    _weight_sg_split_cache[key] = cached
    return cached


def _split_weight_col_tensors(w_fp4_c, w_sc_c, w_sg_c, N_dims, use_localcta_runtime=None):
    """Split cached colwise weight tensors into per-group tensors."""
    if use_localcta_runtime is None:
        use_localcta_runtime = bool(use_tk_localcta())
    key = (
        w_fp4_c.data_ptr(),
        w_sc_c.data_ptr(),
        tuple(N_dims),
        bool(use_localcta_runtime),
        get_tk_localcta_variant() if use_localcta_runtime else None,
        _sg_cache_sig(w_sg_c),
    )
    cached = _weight_split_cache.get(key)
    if cached is not None:
        return cached

    w_fp4_bytes = w_fp4_c.view(torch.uint8)
    w_sc_bytes = w_sc_c.view(torch.uint8)
    B_fp4_list, B_sc_list, B_sg_list = [], [], []
    offset_fp4 = offset_sc = offset_sg = 0
    localcta_v3 = _use_localcta_v3_runtime()
    localcta_v3_tilegrid = _use_localcta_v3_tilegrid256()
    if localcta_v3 and isinstance(w_sg_c, (list, tuple)):
        if len(w_sg_c) != len(N_dims):
            raise ValueError(
                f"localCTA v3 expected {len(N_dims)} per-split col SG tensors, got {len(w_sg_c)}"
            )
    for i, n_i in enumerate(N_dims):
        fp4_cols_i = n_i // 2
        sc_tiles_i = n_i // 64
        B_fp4_list.append(
            w_fp4_bytes[:, offset_fp4:offset_fp4 + fp4_cols_i]
            .contiguous().view(torch.float4_e2m1fn_x2)
        )
        B_sc_list.append(
            w_sc_bytes[:, offset_sc:offset_sc + sc_tiles_i]
            .contiguous().view(torch.float8_e4m3fn)
        )
        if use_localcta_runtime:
            if localcta_v3:
                if isinstance(w_sg_c, (list, tuple)):
                    B_sg_list.append(w_sg_c[i].contiguous())
                elif w_sg_c.dim() == 1:
                    sg_tiles_i = n_i // (256 if localcta_v3_tilegrid else 128)
                    B_sg_list.append(w_sg_c[offset_sg:offset_sg + sg_tiles_i].contiguous())
                    offset_sg += sg_tiles_i
                elif w_sg_c.dim() == 2 and w_sg_c.size(0) == 1:
                    B_sg_list.append(w_sg_c.contiguous())
                else:
                    sg_tiles_i = n_i // (256 if localcta_v3_tilegrid else 128)
                    sg_view = w_sg_c[:, offset_sg:offset_sg + sg_tiles_i]
                    if localcta_v3_tilegrid:
                        sg_view = sg_view.transpose(0, 1)
                    B_sg_list.append(sg_view.contiguous())
                    offset_sg += sg_tiles_i
            else:
                k_outer_tiles = w_fp4_c.size(0) // 256
                if isinstance(w_sg_c, (list, tuple)):
                    if len(w_sg_c) != len(N_dims):
                        raise ValueError(
                            f"localCTA v4 expected {len(N_dims)} per-split col SG tensors, got {len(w_sg_c)}"
                        )
                    B_sg_list.append(w_sg_c[i].contiguous())
                    offset_sg = 0
                elif (
                    w_sg_c.dim() == 1
                    and w_sg_c.numel() == k_outer_tiles * len(N_dims)
                ):
                    start = i * k_outer_tiles
                    B_sg_list.append(w_sg_c[start:start + k_outer_tiles].contiguous())
                    offset_sg = 0
                elif (
                    w_sg_c.dim() == 2
                    and tuple(w_sg_c.shape) == (len(N_dims), k_outer_tiles)
                ):
                    B_sg_list.append(w_sg_c[i, :].contiguous())
                    offset_sg = 0
                elif (
                    w_sg_c.dim() == 2
                    and tuple(w_sg_c.shape) == (k_outer_tiles, len(N_dims))
                ):
                    B_sg_list.append(w_sg_c[:, i].contiguous())
                    offset_sg = 0
                elif (
                    w_sg_c.dim() == 1
                    and w_sg_c.numel() == k_outer_tiles
                ):
                    B_sg_list.append(w_sg_c.contiguous())
                    offset_sg = 0
                elif (
                    w_sg_c.dim() == 2
                    and (
                        tuple(w_sg_c.shape) == (1, k_outer_tiles)
                        or tuple(w_sg_c.shape) == (k_outer_tiles, 1)
                    )
                ):
                    B_sg_list.append(w_sg_c.contiguous())
                    offset_sg = 0
                else:
                    sg_tiles_i = n_i // 128
                    if w_sg_c.dim() == 1:
                        B_sg_list.append(w_sg_c[offset_sg:offset_sg + sg_tiles_i].contiguous())
                    else:
                        B_sg_list.append(w_sg_c[:, offset_sg:offset_sg + sg_tiles_i].contiguous())
                    offset_sg += sg_tiles_i
        else:
            B_sg_list.append(w_sg_c[i:i + 1].to(torch.float32))
        offset_fp4 += fp4_cols_i
        offset_sc += sc_tiles_i

    cached = (B_fp4_list, B_sc_list, B_sg_list)
    _weight_split_cache[key] = cached
    return cached


def _fold_localcta_v4_weight_sc_outer_sg_cached(B_fp4_list, B_sc_list, B_sg_list):
    key = tuple(
        (
            B_fp4_list[i].data_ptr(),
            B_sc_list[i].data_ptr(),
            _sg_cache_sig(B_sg_list[i]),
            tuple(B_fp4_list[i].shape),
            tuple(B_sc_list[i].shape),
        )
        for i in range(len(B_fp4_list))
    )
    cached = _weight_split_outer_fold_cache.get(key)
    if cached is not None:
        return cached
    folded = [
        _fold_localcta_v4_outer_sg_into_prepared_sc(
            B_sc_list[i],
            B_sg_list[i],
            B_fp4_list[i].size(0),
            B_fp4_list[i].size(1) * 2,
        )
        for i in range(len(B_fp4_list))
    ]
    _weight_split_outer_fold_cache[key] = folded
    return folded


def _split_col_fp4_sc_tensors(w_fp4_c, w_sc_c, N_dims):
    """Split colwise FP4 and scale tensors without touching SG metadata."""
    w_fp4_bytes = w_fp4_c.view(torch.uint8)
    w_sc_bytes = w_sc_c.view(torch.uint8)
    B_fp4_list, B_sc_list = [], []
    offset_fp4 = 0
    offset_sc = 0
    for n_i in N_dims:
        fp4_cols_i = n_i // 2
        sc_tiles_i = n_i // 64
        B_fp4_list.append(
            w_fp4_bytes[:, offset_fp4:offset_fp4 + fp4_cols_i]
            .contiguous().view(torch.float4_e2m1fn_x2)
        )
        B_sc_list.append(
            w_sc_bytes[:, offset_sc:offset_sc + sc_tiles_i]
            .contiguous().view(torch.float8_e4m3fn)
        )
        offset_fp4 += fp4_cols_i
        offset_sc += sc_tiles_i
    return B_fp4_list, B_sc_list


def _localcta_split2_sections(total_cols: int) -> list[int] | None:
    """Return an equal 2-way split only when the localCTA runtime can support it."""
    if total_cols < 256 or total_cols % 256 != 0:
        return None
    half = total_cols // 2
    if half % 128 != 0:
        return None
    return [half, half]


def _localcta_raw_dim1_quant_package(input_tensor, split_sections):
    """Run the raw localCTA dim1 quant op and keep the explicit SG payloads."""
    tkq = _get_tk_quant_for_gemm()
    if not _is_localcta_quant_mod(tkq):
        raise RuntimeError("raw localCTA dim1 quant requires localCTA quant mode")

    mod = getattr(tkq, '_mod', None)
    if mod is None or not hasattr(mod, 'tk_localcta_group_quantize_dim1_for_gemm'):
        raise RuntimeError("localCTA quant module does not expose raw dim1 group quant")

    result = mod.tk_localcta_group_quantize_dim1_for_gemm(
        _as_bf16_contiguous(input_tensor),
        split_sections,
    )
    if len(result) != 12:
        raise RuntimeError(
            "localCTA WO split2 path requires the 12-item raw dim1 quant payload, "
            f"got {len(result)} items"
        )

    row_fp4_list, row_sc_list, row_sg_list, \
        col_fp4_list, col_sc_list, col_sg_list, \
        row_fp4_full, row_sc_cat, row_sg_cat, \
        col_fp4_full, col_sc_cat, col_sg_cat = result

    return {
        'row_fp4_list': row_fp4_list,
        'row_sc_list': row_sc_list,
        'row_sg_list': [
            sg.to(torch.float32) if sg.dtype != torch.float32 else sg
            for sg in row_sg_list
        ],
        'col_fp4_list': col_fp4_list,
        'col_sc_list': col_sc_list,
        'col_sg_list': [
            sg.to(torch.float32) if sg.dtype != torch.float32 else sg
            for sg in col_sg_list
        ],
        'row_fp4_full': row_fp4_full,
        'row_sc_cat': row_sc_cat,
        'row_sg_cat': row_sg_cat.to(torch.float32) if row_sg_cat.dtype != torch.float32 else row_sg_cat,
        'col_fp4_full': col_fp4_full,
        'col_sc_cat': col_sc_cat,
        'col_sg_cat': col_sg_cat.to(torch.float32) if col_sg_cat.dtype != torch.float32 else col_sg_cat,
    }


def _localcta_prepared_dim1_split2_quant_package(input_tensor, split_sections):
    """Run the prepared localCTA split2 dim1 quant op and keep explicit payloads."""
    if len(split_sections) != 2:
        raise RuntimeError("prepared localCTA split2 quant requires exactly two split sections")
    tkq = _get_tk_quant_for_gemm()
    if not _is_localcta_quant_mod(tkq):
        raise RuntimeError("prepared localCTA split2 quant requires localCTA quant mode")

    mod = getattr(tkq, '_mod', None)
    if mod is None or not hasattr(mod, 'tk_localcta_group_quantize_dim1_split2_for_gemm_prepared'):
        raise RuntimeError(
            "localCTA quant module does not expose prepared split2 dim1 group quant"
        )

    n0, n1 = split_sections
    input0 = _as_bf16_contiguous(input_tensor[:, :n0])
    input1 = _as_bf16_contiguous(input_tensor[:, n0:n0 + n1])
    result = mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared(input0, input1)
    if len(result) != 12:
        raise RuntimeError(
            "localCTA prepared split2 path requires the 12-item payload, "
            f"got {len(result)} items"
        )

    row_fp4_list, row_sc_list, row_sg_list, \
        col_fp4_list, col_sc_list, col_sg_list, \
        row_fp4_full, row_sc_cat, row_sg_cat, \
        col_fp4_full, col_sc_cat, col_sg_cat = result

    return {
        'row_fp4_list': [
            fp4.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
            for fp4 in row_fp4_list
        ],
        'row_sc_list': [
            sc.contiguous().view(torch.float8_e4m3fn)
            for sc in row_sc_list
        ],
        'row_sg_list': [
            sg.to(torch.float32).contiguous() if sg.dtype != torch.float32 else sg.contiguous()
            for sg in row_sg_list
        ],
        'row_sg_raw_list': [
            sg.to(torch.float32) if sg.dtype != torch.float32 else sg
            for sg in row_sg_list
        ],
        'col_fp4_list': [
            fp4.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
            for fp4 in col_fp4_list
        ],
        'col_sc_list': [
            sc.contiguous().view(torch.float8_e4m3fn)
            for sc in col_sc_list
        ],
        'col_sg_list': [
            sg.to(torch.float32).contiguous() if sg.dtype != torch.float32 else sg.contiguous()
            for sg in col_sg_list
        ],
        'col_sg_raw_list': [
            sg.to(torch.float32) if sg.dtype != torch.float32 else sg
            for sg in col_sg_list
        ],
        'row_fp4_full': row_fp4_full.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2),
        'row_sc_cat': row_sc_cat.contiguous().view(torch.float8_e4m3fn),
        'row_sg_cat': row_sg_cat.to(torch.float32).contiguous() if row_sg_cat.dtype != torch.float32 else row_sg_cat.contiguous(),
        'row_sg_raw_cat': row_sg_cat.to(torch.float32) if row_sg_cat.dtype != torch.float32 else row_sg_cat,
        'col_fp4_full': col_fp4_full.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2),
        'col_sc_cat': col_sc_cat.contiguous().view(torch.float8_e4m3fn),
        'col_sg_cat': col_sg_cat.to(torch.float32).contiguous() if col_sg_cat.dtype != torch.float32 else col_sg_cat.contiguous(),
        'col_sg_raw_cat': col_sg_cat.to(torch.float32) if col_sg_cat.dtype != torch.float32 else col_sg_cat,
    }


def _localcta_prepare_col_payload(fp4_c, sc_c, sg_c, rows, cols):
    """Fold SG into colwise microscales for fast localCTA kernels."""
    fp4_c = fp4_c.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
    sc_prepared = _fold_sg_into_localcta_prepared_sc(sc_c, sg_c, rows, cols)
    sg_unit = torch.ones(1, dtype=torch.float32, device=sc_prepared.device)
    return fp4_c, sc_prepared, sg_unit


def _scale_bytes_all_zero(scale_tensor: torch.Tensor) -> bool:
    if scale_tensor.numel() == 0:
        return True
    scale_bytes = scale_tensor.contiguous().view(torch.uint8)
    return not bool(torch.count_nonzero(scale_bytes).item())


def _scale_bytes_debug(scale_tensor: torch.Tensor) -> dict[str, object]:
    if scale_tensor.numel() == 0:
        return {
            'present': True,
            'numel': 0,
            'byte_nonzero': 0,
            'all_zero': True,
        }
    scale_bytes = scale_tensor.contiguous().view(torch.uint8)
    byte_nonzero = int(torch.count_nonzero(scale_bytes).item())
    return {
        'present': True,
        'numel': int(scale_tensor.numel()),
        'byte_nonzero': byte_nonzero,
        'all_zero': byte_nonzero == 0,
    }


def _scale_sequence_any_underflow(scale_seq) -> bool:
    if scale_seq is None:
        return False
    if torch.is_tensor(scale_seq):
        return _scale_bytes_all_zero(scale_seq)
    for item in scale_seq:
        if torch.is_tensor(item) and _scale_bytes_all_zero(item):
            return True
    return False


def _localcta_qkv_package_underflow(dgrad_package) -> bool:
    return (
        _scale_sequence_any_underflow(dgrad_package.get('a_sc_list'))
        or _scale_sequence_any_underflow(dgrad_package.get('sc_col_list'))
        or _scale_sequence_any_underflow(dgrad_package.get('a_sc_cat'))
        or _scale_sequence_any_underflow(dgrad_package.get('sc_col_cat'))
    )


def _localcta_qkv_package_underflow_details(dgrad_package) -> dict[str, object]:
    details = {}
    for key in ('a_sc_list', 'sc_col_list'):
        seq = dgrad_package.get(key)
        if seq is None:
            details[key] = {'present': False}
            continue
        if torch.is_tensor(seq):
            details[key] = _scale_bytes_debug(seq)
            continue
        items = []
        any_zero = False
        for item in seq:
            if torch.is_tensor(item):
                item_stats = _scale_bytes_debug(item)
                any_zero = any_zero or bool(item_stats['all_zero'])
                items.append(item_stats)
        details[key] = {
            'present': True,
            'items': items,
            'any_all_zero': any_zero,
        }
    for key in ('a_sc_cat', 'sc_col_cat'):
        tensor = dgrad_package.get(key)
        details[key] = {'present': False} if tensor is None else _scale_bytes_debug(tensor)
    details['underflow'] = (
        bool(details.get('a_sc_list', {}).get('any_all_zero'))
        or bool(details.get('sc_col_list', {}).get('any_all_zero'))
        or bool(details.get('a_sc_cat', {}).get('all_zero'))
        or bool(details.get('sc_col_cat', {}).get('all_zero'))
    )
    return details


def _tensor_any_nonzero(tensor: torch.Tensor | None) -> bool:
    if tensor is None or not torch.is_tensor(tensor):
        return False
    return bool(torch.count_nonzero(tensor).item())


def _sequence_any_nonzero(seq) -> bool:
    if seq is None:
        return False
    if torch.is_tensor(seq):
        return _tensor_any_nonzero(seq)
    return any(_tensor_any_nonzero(item) for item in seq if torch.is_tensor(item))


def _localcta_qkv_package_has_live_sg(dgrad_package) -> bool:
    return (
        _sequence_any_nonzero(dgrad_package.get('a_sg_list'))
        or _sequence_any_nonzero(dgrad_package.get('col_sg_list'))
        or _tensor_any_nonzero(dgrad_package.get('a_sg_full'))
        or _tensor_any_nonzero(dgrad_package.get('col_sg_cat'))
    )


def _floor_zero_fp8_scale_tensor_(tensor: torch.Tensor) -> dict[str, int]:
    scale_bytes = tensor.contiguous().view(torch.uint8)
    zero_mask = scale_bytes == 0
    replaced = int(torch.count_nonzero(zero_mask).item())
    if replaced:
        scale_bytes[zero_mask] = 1
    return {'numel': int(scale_bytes.numel()), 'replaced': replaced}


def _floor_zero_localcta_qkv_prepared_scales_(dgrad_package) -> dict[str, object]:
    seen: set[tuple[int, int]] = set()
    touched: list[dict[str, int | str]] = []

    def visit(name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or not torch.is_tensor(tensor):
            return
        key = (tensor.data_ptr(), tensor.numel())
        if key in seen:
            return
        seen.add(key)
        stats = _floor_zero_fp8_scale_tensor_(tensor)
        stats['name'] = name
        touched.append(stats)

    for idx, tensor in enumerate(dgrad_package.get('a_sc_list') or []):
        visit(f'a_sc_list[{idx}]', tensor)
    for idx, tensor in enumerate(dgrad_package.get('sc_col_list') or []):
        visit(f'sc_col_list[{idx}]', tensor)
    visit('a_sc_cat', dgrad_package.get('a_sc_cat'))
    visit('sc_col_cat', dgrad_package.get('sc_col_cat'))

    return {
        'taken': any(int(item['replaced']) > 0 for item in touched),
        'path': 'prepared_scale_floor',
        'touched': touched,
        'total_replaced': sum(int(item['replaced']) for item in touched),
    }


def _try_localcta_qkv_scale_backoff_package(
    tkq,
    grad_splits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    N_dims,
    rope_live64_cs=None,
    rope_seq_len: int = 0,
    persistent_rng_state=None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not hasattr(tkq, 'tk_set_global_scale_num') or not hasattr(tkq, 'tk_get_global_scale_num'):
        return None, None
    prev_scale = float(tkq.tk_get_global_scale_num())
    attempts: list[dict[str, object]] = []
    retry_rng_state = None
    if persistent_rng_state is not None:
        # The initial producer has already reserved this logical invocation's
        # subsequence.  Scale-backoff requantizations reuse that same random
        # draw through scratch state so the checkpointed primary counter still
        # advances exactly once, independent of data-dependent retry count.
        retry_rng_state = persistent_rng_state.clone()
    try:
        for scale_num in get_tk_qkv_localcta_scale_backoff_values():
            tkq.tk_set_global_scale_num(scale_num)
            if retry_rng_state is not None:
                retry_rng_state.copy_(persistent_rng_state)
                retry_rng_state[1].sub_(1 << 32)
            package = _localcta_grouped_k_dgrad_package(
                grad_splits,
                N_dims,
                rope_live64_cs=rope_live64_cs,
                rope_seq_len=rope_seq_len,
                persistent_rng_state=retry_rng_state,
            )
            underflow_details = _localcta_qkv_package_underflow_details(package)
            attempts.append({
                'scale_num': scale_num,
                'underflow': bool(underflow_details.get('underflow')),
                'underflow_details': underflow_details,
            })
            if not underflow_details.get('underflow'):
                return package, {
                    'taken': True,
                    'path': 'scale_backoff_requant',
                    'scale_num': scale_num,
                    'attempts': attempts,
                }
    finally:
        tkq.tk_set_global_scale_num(prev_scale)
    return None, {
        'taken': False,
        'path': 'scale_backoff_requant',
        'attempts': attempts,
    }


def can_use_localcta_split2_wo_backward(dy: torch.Tensor) -> bool:
    """WO backward split2 path is only valid on the fast localCTA stack."""
    return (
        use_tk_localcta()
        and _localcta_split2_sections(dy.shape[1]) is not None
    )


def tk_localcta_split2_wo_backward(
    dy: torch.Tensor,
    x_nvfp4,
    w_nvfp4,
    dx=None,
    input_bf16: torch.Tensor | None = None,
    w_bf16: torch.Tensor | None = None,
):
    """Run WO backward on the localCTA split2 path.

    The default path preserves the exact fast-stack raw localCTA contract.
    A debug-only prepared split2 variant is available to test whether the
    current WO backward corruption is caused by the raw split2 payloads rather
    than by the grouped / batched localCTA kernels themselves.
    """
    split_sections = _localcta_split2_sections(dy.shape[1])
    if split_sections is None:
        raise RuntimeError(
            f"localCTA WO split2 backend requires N divisible by 256, got N={dy.shape[1]}"
        )

    use_prepared = use_tk_localcta_wo_prepared_split2_backward()
    use_raw_fast_outer = (
        not use_prepared
        and get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_wo_raw_fast_outer()
    )
    wo_grad_boost = 1.0
    dy_quant = dy
    tk_direct = None
    tk_fast = None
    if use_prepared or use_raw_fast_outer:
        tk_fast = _get_tk()
        if tk_fast is None or not getattr(tk_fast, '_is_localcta', False):
            raise RuntimeError("localCTA WO split2 fast outer backend requires fast localCTA GEMM")
    if use_prepared:
        wo_grad_boost = get_tk_localcta_wo_fixed_grad_boost()
        if wo_grad_boost != 1.0:
            dy_quant = (dy.float() * wo_grad_boost).to(torch.bfloat16).contiguous()
            _trace_backend_choice('localcta_wo_bwd', f'adaptive_grad_scale_{wo_grad_boost:g}')
        dy_pkg = _localcta_prepared_dim1_split2_quant_package(dy_quant, split_sections)
    else:
        tk_direct = _get_tk_localcta_direct()
        if tk_direct is None:
            raise RuntimeError(
                f"localCTA direct runtime not available: {_tk_localcta_direct_import_error}"
            )
        if not hasattr(tk_direct, 'nvfp4_batched_accum_gemm'):
            raise RuntimeError("localCTA direct runtime is missing nvfp4_batched_accum_gemm")
        if not hasattr(tk_direct, 'nvfp4_grouped_gemm'):
            raise RuntimeError("localCTA direct runtime is missing nvfp4_grouped_gemm")
        dy_pkg = _localcta_raw_dim1_quant_package(dy, split_sections)

    if (
        use_prepared
        and input_bf16 is not None
        and w_bf16 is not None
        and use_tk_localcta_wo_bf16_underflow_rescue()
        and (
            _scale_bytes_all_zero(dy_pkg['row_sc_cat'])
            or _scale_bytes_all_zero(dy_pkg['col_sc_cat'])
        )
    ):
        if dx is None:
            dx = torch.empty(
                dy.shape[0],
                w_nvfp4.shape[1],
                dtype=torch.bfloat16,
                device=dy.device,
            )
        dx.copy_(torch.matmul(dy.float(), w_bf16.float()).to(torch.bfloat16))
        grad_w = torch.matmul(dy.transpose(0, 1).float(), input_bf16.float()).to(torch.bfloat16)
        return dx, grad_w, {
            'taken': True,
            'reason': 'zero_dy_sc',
            'path': 'bf16_underflow_rescue',
            'prepared': True,
        }

    w_fp4_c, w_sc_c, w_sg_c = w_nvfp4._tk_col
    if use_prepared:
        if get_tk_localcta_variant() == 'v4':
            # Keep the true v4 dgrad contract explicit: raw local row/col
            # microscales on both operands, plus one outer SG per output tile.
            b_fp4_list, b_sc_list, b_sg_list = _split_weight_col_tensors(
                w_fp4_c,
                w_sc_c,
                w_sg_c,
                split_sections,
            )
            a_sg_list = [
                _prepare_localcta_v4_outer_sg_for_direct(
                    sg,
                    dy.shape[0] // 256,
                    dy.device,
                )
                for sg in dy_pkg['row_sg_list']
            ]
            b_sg_grid_list = [
                _prepare_localcta_v4_outer_sg_for_direct(
                    sg,
                    int(b_fp4.size(0)) // 256,
                    dy.device,
                )
                for b_fp4, sg in zip(b_fp4_list, b_sg_list)
            ]
        else:
            b_fp4_list, b_sc_list, b_sg_list = _split_weight_col_tensors(
                w_fp4_c,
                w_sc_c,
                w_sg_c,
                split_sections,
            )
            a_sg_list = [
                _localcta_expand_sg_grid(sg, dy.shape[0], split_sections[i])
                if torch.is_tensor(sg) and sg.dim() != 2 else sg
                for i, sg in enumerate(dy_pkg['row_sg_list'])
            ]
            b_sg_grid_list = [
                _localcta_expand_sg_grid(sg, w_nvfp4.shape[1], split_sections[i])
                if torch.is_tensor(sg) and sg.dim() != 2 else sg
                for i, sg in enumerate(b_sg_list)
            ]
    else:
        b_fp4_list, b_sc_list, b_sg_list = _split_weight_col_tensors(
            w_fp4_c, w_sc_c, w_sg_c, split_sections
        )
        a_sg_list = [
            _localcta_expand_sg_grid(sg, dy.shape[0], split_sections[i])
            if torch.is_tensor(sg) and sg.dim() != 2 else sg
            for i, sg in enumerate(dy_pkg['row_sg_list'])
        ]
        b_sg_grid_list = [
            _localcta_expand_sg_grid(sg, w_nvfp4.shape[1], split_sections[i])
            if torch.is_tensor(sg) and sg.dim() != 2 else sg
            for i, sg in enumerate(b_sg_list)
        ]

    if dx is None:
        dx = torch.empty(
            dy.shape[0],
            w_nvfp4.shape[1],
            dtype=torch.bfloat16,
            device=dy.device,
        )
    _trace_backend_choice(
        'localcta_wo_dgrad',
        (
            'split2_prepared_fast'
            if use_prepared else
            'split2_raw_fast_outer'
            if use_raw_fast_outer else
            'split2_raw_chunkgrid'
        )
    )
    if use_prepared or use_raw_fast_outer:
        batched_accum = (
            getattr(tk_fast, 'nvfp4_batched_accum_gemm_fast', None)
            if (get_tk_localcta_variant() == 'v4' and (use_prepared or use_raw_fast_outer)) else None
        )
        if batched_accum is None:
            batched_accum = tk_fast.nvfp4_batched_accum_gemm
        batched_accum(
            dy_pkg['row_fp4_list'],
            dy_pkg['row_sc_list'],
            a_sg_list,
            b_fp4_list,
            b_sc_list,
            b_sg_grid_list,
            dx,
        )
    else:
        tk_direct.nvfp4_batched_accum_gemm(
            dy_pkg['row_fp4_list'],
            dy_pkg['row_sc_list'],
            a_sg_list,
            b_fp4_list,
            b_sc_list,
            b_sg_grid_list,
            dx,
        )

    x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
    if use_prepared:
        col_sg_cat = dy_pkg['col_sg_cat']
        if torch.is_tensor(x_sg_c) and x_sg_c.dim() != 2:
            x_sg_c = _localcta_expand_sg_grid(x_sg_c, x_nvfp4.shape[1], x_nvfp4.shape[0])
        if torch.is_tensor(col_sg_cat) and col_sg_cat.dim() != 2:
            col_sg_cat = _localcta_group_sg_grid_from_scalars(
                dy_pkg['col_sg_list'],
                split_sections,
                x_nvfp4.shape[0],
                x_fp4_c.device,
            )
    else:
        if torch.is_tensor(x_sg_c) and x_sg_c.dim() != 2:
            x_sg_c = _localcta_expand_sg_grid(x_sg_c, x_nvfp4.shape[1], x_nvfp4.shape[0])
        col_sg_cat = dy_pkg['col_sg_cat']
        if torch.is_tensor(col_sg_cat) and col_sg_cat.dim() != 2:
            col_sg_cat = _localcta_group_sg_grid_from_scalars(
                dy_pkg['col_sg_list'],
                split_sections,
                x_nvfp4.shape[0],
                x_fp4_c.device,
            )
    n_total = dy.shape[1]
    k = x_fp4_c.shape[0]
    dW_T = _get_wgrad_buf(k, n_total, x_fp4_c.device)
    _trace_backend_choice(
        'localcta_wo_wgrad',
        (
            'split2_prepared_fast'
            if use_prepared else
            'split2_raw_fast_outer'
            if use_raw_fast_outer else
            'split2_raw_chunkgrid'
        )
    )
    if use_prepared or use_raw_fast_outer:
        grouped_gemm = (
            getattr(tk_fast, 'nvfp4_grouped_gemm_fast', None)
            if get_tk_localcta_variant() == 'v4' else None
        )
        if grouped_gemm is None:
            grouped_gemm = tk_fast.nvfp4_grouped_gemm
    else:
        grouped_gemm = tk_direct.nvfp4_grouped_gemm
    grouped_gemm(
        x_fp4_c,
        x_sc_c,
        x_sg_c,
        dy_pkg['col_fp4_full'],
        dy_pkg['col_sc_cat'],
        col_sg_cat,
        dW_T,
    )
    if use_prepared and wo_grad_boost != 1.0:
        dx.div_(wo_grad_boost)
        dW_T.div_(wo_grad_boost)
    return dx, dW_T.T, None


def tk_localcta_direct_wo_backward(dy_nvfp4, x_nvfp4, w_nvfp4, dx=None, dW=None):
    """WO backward for the direct-contract localCTA path.

    The fast localCTA single-GEMM entrypoint ignores SG tensors and therefore
    only works when SG has already been folded into the microscales. The
    current direct-contract WO backward caches raw localCTA row/col payloads
    with explicit SG, so it must use a single-GEMM backend that consumes SG.
    """
    tk = _get_tk()
    gemm = getattr(tk, 'nvfp4_gemm_highacc', None)
    if gemm is None:
        tk_direct = _get_tk_localcta_direct()
        gemm = getattr(tk_direct, 'nvfp4_gemm', None) if tk_direct is not None else None
    if gemm is None:
        raise RuntimeError("localCTA SG-aware single GEMM backend is not available")

    dy_fp4_r, dy_sc_r, dy_sg_r = dy_nvfp4._tk_row
    w_fp4_c, w_sc_c, w_sg_c = w_nvfp4._tk_col
    if dx is None:
        dx = torch.empty(
            dy_nvfp4.shape[0],
            w_nvfp4.shape[1],
            dtype=torch.bfloat16,
            device=dy_fp4_r.device,
        )
    _trace_backend_choice('localcta_wo_dgrad', 'sg_single_gemm')
    gemm(
        dy_fp4_r,
        dy_sc_r,
        _localcta_expand_sg_grid(dy_sg_r, dy_nvfp4.shape[0], dy_nvfp4.shape[1]),
        w_fp4_c,
        w_sc_c,
        _localcta_expand_sg_grid(w_sg_c, w_nvfp4.shape[1], w_nvfp4.shape[0]),
        dx,
    )

    dy_fp4_c, dy_sc_c, dy_sg_c = dy_nvfp4._tk_col
    x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
    if dW is None:
        dW = torch.empty(
            dy_nvfp4.shape[1],
            x_nvfp4.shape[1],
            dtype=torch.bfloat16,
            device=dy_fp4_c.device,
        )
    _trace_backend_choice('localcta_wo_wgrad', 'sg_single_gemm')
    gemm(
        dy_fp4_c,
        dy_sc_c,
        _localcta_expand_sg_grid(dy_sg_c, dy_nvfp4.shape[1], dy_nvfp4.shape[0]),
        x_fp4_c,
        x_sc_c,
        _localcta_expand_sg_grid(x_sg_c, x_nvfp4.shape[1], x_nvfp4.shape[0]),
        dW,
    )
    return dx, dW


def _as_bf16_contiguous(tensor):
    """Normalize an input tensor to contiguous bf16 for quant backends."""
    if tensor.is_contiguous() and tensor.dtype == torch.bfloat16:
        return tensor
    if tensor.dtype == torch.bfloat16:
        return tensor.contiguous()
    return tensor.contiguous().to(torch.bfloat16)


def _use_localcta_v4_strided_qkv_grad_quant() -> bool:
    return (
        get_tk_localcta_variant() == 'v4'
        and os.environ.get('USE_TK_LOCALCTA_V4_QKV_STRIDED_GRAD_QUANT', '1') == '1'
    )


def use_tk_qkv_strided_grad_quant() -> bool:
    """Preserve row-strided QKV grad split views when the quant backend supports it."""
    return _env_flag('USE_TK_QKV_STRIDED_GRAD_QUANT', True)


def _as_bf16_quant_input(tensor, allow_strided: bool = False):
    """Normalize to BF16 while preserving row-strided matrices when supported."""
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if (
        allow_strided
        and tensor.is_cuda
        and tensor.dim() == 2
        and tensor.stride(1) == 1
    ):
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _use_tk_qkv_fast_expanded_sum_grad() -> bool:
    value = os.environ.get('USE_TK_QKV_FAST_EXPANDED_SUM_GRAD')
    if value is not None:
        return value == '1'
    return True


def _is_zero_stride_scalar_bf16_grad(tensor) -> bool:
    if not torch.is_tensor(tensor):
        return False
    base = getattr(tensor, "_base", None)
    return (
        tensor.dtype == torch.bfloat16
        and tensor.is_cuda
        and not tensor.is_contiguous()
        and tensor.dim() == 2
        and base is not None
        and base.numel() == 1
        and all(stride == 0 for stride in tensor.stride())
    )


def _make_localcta_v4_split3_scalar_grad_package(grad_splits, N_dims, scale_num: float):
    """Build the localCTA-v4 split3 package for scalar-expanded QKV grads."""
    g0, g1, g2 = grad_splits
    M = int(g0.shape[0])
    n0, n1, n2 = [int(n) for n in N_dims]
    total_n = n0 + n1 + n2
    device = g0.device

    row_fp4_cat = torch.empty(
        (M, total_n // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=device,
    )
    row_sc_cat = torch.empty(
        (M // 128, total_n // 64, 512),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    col_fp4_cat = torch.empty(
        (total_n, M // 2),
        dtype=torch.float4_e2m1fn_x2,
        device=device,
    )
    col_sc_cat = torch.empty(
        (total_n // 128, M // 64, 512),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    # For nonzero scalar BF16 grads, localCTA-v4 encodes each element as the
    # saturated FP4 endpoint and carries the magnitude through the outer SG.
    row_fp4_cat.view(torch.uint8).fill_(0x77)
    col_fp4_cat.view(torch.uint8).fill_(0x77)
    row_sc_cat.view(torch.uint8).fill_(0x78)
    col_sc_cat.view(torch.uint8).fill_(0x78)

    row_fp4_list = [
        row_fp4_cat.narrow(1, 0, n0 // 2),
        row_fp4_cat.narrow(1, n0 // 2, n1 // 2),
        row_fp4_cat.narrow(1, (n0 + n1) // 2, n2 // 2),
    ]
    row_sc_list = [
        row_sc_cat.narrow(1, 0, n0 // 64),
        row_sc_cat.narrow(1, n0 // 64, n1 // 64),
        row_sc_cat.narrow(1, (n0 + n1) // 64, n2 // 64),
    ]
    col_fp4_list = [
        col_fp4_cat.narrow(0, 0, n0),
        col_fp4_cat.narrow(0, n0, n1),
        col_fp4_cat.narrow(0, n0 + n1, n2),
    ]
    col_sc_list = [
        col_sc_cat.narrow(0, 0, n0 // 128),
        col_sc_cat.narrow(0, n0 // 128, n1 // 128),
        col_sc_cat.narrow(0, (n0 + n1) // 128, n2 // 128),
    ]

    inv_scale_num = 1.0 / float(scale_num)
    row_sg_list = [
        torch.empty((M // 256, 1), dtype=torch.float32, device=device),
        torch.empty((M // 256, 1), dtype=torch.float32, device=device),
        torch.empty((M // 256, 1), dtype=torch.float32, device=device),
    ]
    col_sg_cat = torch.empty((1, total_n // 256), dtype=torch.float32, device=device)
    col_sg_list = [
        col_sg_cat.narrow(1, 0, n0 // 256),
        col_sg_cat.narrow(1, n0 // 256, n1 // 256),
        col_sg_cat.narrow(1, (n0 + n1) // 256, n2 // 256),
    ]

    for grad, row_sg, col_sg in zip((g0, g1, g2), row_sg_list, col_sg_list):
        base = getattr(grad, "_base", None)
        sg_value = base.abs().to(torch.float32).mul(inv_scale_num)
        row_sg.copy_(sg_value.expand_as(row_sg))
        col_sg.copy_(sg_value.expand_as(col_sg))

    return {
        'M': M,
        'device': device,
        'from_concat': False,
        'prepared_nofold': False,
        'consumer_contract': 'folded_prepared',
        'N_dims': list(N_dims),
        'fp4_row_list': row_fp4_list,
        'sc_row_list': row_sc_list,
        'row_sg_list': row_sg_list,
        'a_fp4_list': None,
        'a_sc_list': list(row_sc_list),
        'a_sg_list': row_sg_list,
        'fp4_col_list': col_fp4_list,
        'sc_col_list': col_sc_list,
        'col_sg_list': col_sg_list,
        'a_fp4_full': row_fp4_cat,
        'a_sc_cat': row_sc_cat,
        'a_sg_full': torch.empty((0,), dtype=torch.float32, device=device),
        'fp4_col_full': col_fp4_cat,
        'sc_col_cat': col_sc_cat,
        'col_sg_cat': col_sg_cat,
        'scalar_expanded_sum_grad': True,
    }


def _parse_localcta_split3_dgrad_package(result, M, device):
    """Normalize localCTA split3 dim1 quant output into a reusable package."""
    if len(result) == 12:
        fp4_row_list, sc_row_list, row_sg_list, \
            fp4_col_list, sc_col_list, col_sg_list, \
            a_fp4_full, a_sc_cat, a_sg_full, \
            fp4_col_full, sc_col_cat, col_sg_cat = result
    else:
        fp4_row_list, sc_row_list, row_sg_list, \
            fp4_col_list, sc_col_list, col_sg_list, \
            fp4_col_full, sc_col_cat, col_sg_cat = result
        a_fp4_full = a_sc_cat = a_sg_full = None

    have_full_row = a_fp4_full is not None
    return {
        'M': M,
        'device': device,
        'from_concat': False,
        'prepared_nofold': False,
        'consumer_contract': 'folded_prepared',
        'N_dims': None,
        'a_sc_outer_folded': use_tk_localcta_v4_split3_fold_row_sg_in_producer(),
        'fp4_row_list': fp4_row_list,
        'sc_row_list': sc_row_list,
        'row_sg_list': row_sg_list,
        # localCTA split3 already provides a contiguous full-row payload. Keep
        # per-split row views lazy so the fast strided path can consume the
        # full tensors directly without rebuilding contiguous split copies in
        # Python on every backward call.
        'a_fp4_list': None if have_full_row else [
            fp4.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
            for fp4 in fp4_row_list
        ],
        'a_sc_list': list(sc_row_list) if have_full_row else [
            sc.contiguous().view(torch.float8_e4m3fn)
            for sc in sc_row_list
        ],
        'a_sg_list': [
            sg.to(torch.float32) if sg.dtype != torch.float32 else sg
            for sg in row_sg_list
        ],
        'fp4_col_list': fp4_col_list,
        'sc_col_list': sc_col_list,
        'col_sg_list': col_sg_list,
        'a_fp4_full': a_fp4_full,
        'a_sc_cat': a_sc_cat,
        'a_sg_full': a_sg_full,
        'fp4_col_full': fp4_col_full,
        'sc_col_cat': sc_col_cat,
        'col_sg_cat': col_sg_cat,
    }


def _localcta_single_quant_to_split3_package(
    tkq,
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    *,
    prepared_nofold: bool = False,
) -> dict[str, object]:
    pieces = []
    for g in (g0, g1, g2):
        if prepared_nofold:
            result = tkq.tk_quantize_for_gemm_prepared_nofold_maybe_borrow(
                g, g, True, True
            )
        else:
            result = tkq.tk_quantize_for_gemm(
                g,
                True,
                True,
            )
        row_sg = result[4].to(torch.float32) if result[4].dtype != torch.float32 else result[4]
        col_sg = (
            result[5].to(torch.float32)
            if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0
            else row_sg
        )
        pieces.append((result[0], result[1], row_sg, result[2], result[3], col_sg))

    row_fp4_list = [piece[0] for piece in pieces]
    row_sc_list = [piece[1] for piece in pieces]
    row_sg_list = [piece[2] for piece in pieces]
    col_fp4_list = [piece[3] for piece in pieces]
    col_sc_list = [piece[4] for piece in pieces]
    col_sg_list = [piece[5] for piece in pieces]

    row_fp4_cat = torch.cat(
        [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in row_fp4_list],
        dim=1,
    ).view(torch.float4_e2m1fn_x2)
    row_sc_cat = torch.cat(
        [sc.contiguous().view(torch.uint8) for sc in row_sc_list],
        dim=1,
    ).view(torch.float8_e4m3fn)
    row_sg_cat = torch.cat(row_sg_list, dim=1)
    col_fp4_cat = torch.cat(
        [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in col_fp4_list],
        dim=0,
    ).view(torch.float4_e2m1fn_x2)
    col_sc_cat = torch.cat(
        [sc.contiguous().view(torch.uint8) for sc in col_sc_list],
        dim=0,
    ).view(torch.float8_e4m3fn)
    col_sg_cat = torch.cat(col_sg_list, dim=0)

    return {
        'M': g0.shape[0],
        'device': g0.device,
        'from_concat': False,
        'prepared_nofold': prepared_nofold,
        'consumer_contract': 'nofold_prepared' if prepared_nofold else 'folded_prepared',
        'N_dims': [g0.shape[1], g1.shape[1], g2.shape[1]],
        'fp4_row_list': row_fp4_list,
        'sc_row_list': row_sc_list,
        'row_sg_list': row_sg_list,
        'a_fp4_list': [
            _packed_fp4_contiguous(fp4)
            for fp4 in row_fp4_list
        ],
        'a_sc_list': [
            sc.contiguous().view(torch.float8_e4m3fn)
            for sc in row_sc_list
        ],
        'a_sg_list': row_sg_list,
        'fp4_col_list': col_fp4_list,
        'sc_col_list': col_sc_list,
        'col_sg_list': col_sg_list,
        'a_fp4_full': row_fp4_cat,
        'a_sc_cat': row_sc_cat,
        'a_sg_full': row_sg_cat,
        'fp4_col_full': col_fp4_cat,
        'sc_col_cat': col_sc_cat,
        'col_sg_cat': col_sg_cat,
    }


def _narrow_localcta_col_payload(full_tensor, N_dims, divisor):
    """Split a full colwise localCTA tensor into per-group narrows."""
    pieces = []
    offset = 0
    for n_i in N_dims:
        size_i = n_i // divisor
        pieces.append(full_tensor.narrow(0, offset, size_i))
        offset += size_i
    return pieces


def _parse_localcta_concat_dgrad_package(result, N_dims, M, device):
    """Normalize concat localCTA dim1 quant output into the dgrad package shape."""
    if len(result) == 5:
        row_fp4_full, row_sc_prepared_list, \
            col_fp4_full, col_sc_prepared_full, col_sg_full = result
        row_sg_list = None
        col_fp4_list = _narrow_localcta_col_payload(col_fp4_full, N_dims, 1)
        col_sc_prepared_list = _narrow_localcta_col_payload(col_sc_prepared_full, N_dims, 128)
        col_sg_list = _narrow_localcta_col_payload(
            col_sg_full, N_dims, 256 if _use_localcta_v3_tilegrid256() else 128
        )
    elif len(result) == 16:
        row_fp4_list, row_sc_list, row_sg_list, \
            col_fp4_list, col_sc_list, col_sg_list, \
            row_fp4_full, row_sc_full, row_sg_full, \
            col_fp4_full, col_sc_full, col_sg_full, \
            row_sc_prepared_list, col_sc_prepared_list, \
            row_sc_prepared_full, col_sc_prepared_full = result
    else:
        row_fp4_full, row_sc_prepared_list, row_sg_list, \
            col_fp4_list, col_sc_prepared_list, col_sg_list, \
            col_fp4_full, col_sc_prepared_full, col_sg_full = result

    return {
        'M': M,
        'device': device,
        'from_concat': True,
        'prepared_nofold': False,
        'consumer_contract': 'folded_prepared',
        'N_dims': list(N_dims),
        'fp4_row_list': None,
        'sc_row_list': row_sc_list if len(result) == 16 else None,
        'row_sg_list': row_sg_list,
        'a_fp4_list': None,
        'a_sc_list': [
            sc.contiguous().view(torch.float8_e4m3fn)
            for sc in row_sc_prepared_list
        ],
        'a_sg_list': row_sg_list,
        'a_fp4_full': row_fp4_full,
        'a_sc_cat': None,
        'a_sc_full': row_sc_prepared_full if len(result) == 16 else None,
        'a_sg_full': row_sg_full if len(result) == 16 else None,
        'fp4_col_list': col_fp4_list,
        'sc_col_list': col_sc_prepared_list,
        'col_sg_list': col_sg_list,
        'fp4_col_full': col_fp4_full,
        'sc_col_cat': col_sc_prepared_full,
        'col_sg_cat': col_sg_full,
    }


def _ensure_localcta_row_split_payload(dgrad_package, N_dims):
    """Materialize per-split row tensors only when a backend needs them."""
    if dgrad_package.get('a_fp4_list') is not None:
        return (
            dgrad_package['a_fp4_list'],
            dgrad_package['a_sc_list'],
            dgrad_package['a_sg_list'],
        )

    row_fp4_full = dgrad_package['a_fp4_full']
    row_fp4_u8 = row_fp4_full.view(torch.uint8)
    existing_a_sc_list = dgrad_package.get('a_sc_list')
    existing_a_sg_list = dgrad_package.get('a_sg_list')
    missing_a_sg_full = dgrad_package.get('a_sg_full') is None and existing_a_sg_list is None

    a_fp4_list = []
    a_sc_list = [] if existing_a_sc_list is None else list(existing_a_sc_list)
    a_sg_list = [] if existing_a_sg_list is None else list(existing_a_sg_list)
    sg_divisor = 256 if _use_localcta_v3_tilegrid256() else 128
    fp4_off = sc_off = sg_off = 0
    for n_i in N_dims:
        fp4_cols = n_i // 2
        a_fp4_list.append(
            row_fp4_u8.narrow(1, fp4_off, fp4_cols).contiguous().view(torch.float4_e2m1fn_x2)
        )
        fp4_off += fp4_cols
        if existing_a_sc_list is None:
            sc_tiles = n_i // 64
            a_sc_list.append(
                dgrad_package['a_sc_full'].narrow(1, sc_off, sc_tiles).contiguous().view(torch.float8_e4m3fn)
            )
            sc_off += sc_tiles
        if existing_a_sg_list is None:
            sg_tiles = n_i // sg_divisor
            if missing_a_sg_full:
                a_sg_list.append(None)
            else:
                a_sg_list.append(
                    dgrad_package['a_sg_full'].narrow(1, sg_off, sg_tiles).contiguous().to(torch.float32)
                )
            sg_off += sg_tiles

    dgrad_package['a_fp4_list'] = a_fp4_list
    dgrad_package['a_sc_list'] = a_sc_list
    dgrad_package['a_sg_list'] = a_sg_list
    return a_fp4_list, a_sc_list, a_sg_list


def _use_localcta_paired_rht_carrier() -> bool:
    return _env_flag("USE_TK_LOCALCTA_PAIRED_RHT_CARRIER", False)


def _localcta_paired_rht_role_enabled(role: str) -> bool:
    explicit = os.environ.get(f"NVFP4_RHT_{role.upper()}")
    if explicit is not None:
        return _env_flag(f"NVFP4_RHT_{role.upper()}", False)
    return _env_flag("NVFP4_USE_RHT", False)


def _localcta_paired_rht_split3_package(
    tkq,
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    N_dims,
    *,
    persistent_rng_state,
):
    """Build split3 payloads through one common SR/RHT-capable producer.

    New localCTA binaries consume Q/K/V separately while enumerating their
    tiles as one logical concatenation.  This removes the BF16 cat and output
    split copies without changing RHT/SR coordinates or the single persistent
    RNG reservation.  Older binaries retain the exact concatenated fallback.
    """
    if get_tk_localcta_variant() != "v4":
        raise RuntimeError("paired localCTA QKV carrier requires variant v4")
    if (
        not hasattr(tkq, "tk_quantize_for_gemm_opt")
        and not hasattr(
            tkq, "tk_group_quantize_dim1_split3_for_gemm_paired_rht"
        )
    ):
        raise RuntimeError(
            "paired localCTA QKV carrier requires a native split3 producer "
            "or tk_quantize_for_gemm_opt fallback"
        )
    if persistent_rng_state is None:
        raise RuntimeError(
            "paired localCTA QKV carrier requires checkpointed SR state"
        )
    if len(N_dims) != 3:
        raise RuntimeError("paired localCTA QKV carrier requires three splits")

    grads = tuple(_as_bf16_contiguous(g) for g in (g0, g1, g2))
    M = grads[0].shape[0]
    if any(g.dim() != 2 or g.shape[0] != M for g in grads):
        raise RuntimeError("paired localCTA QKV carrier requires aligned 2D inputs")
    if [g.shape[1] for g in grads] != list(N_dims):
        raise RuntimeError("paired localCTA QKV split widths do not match N_dims")
    if M % 256 or any(int(n) % 256 for n in N_dims):
        raise RuntimeError(
            "paired localCTA QKV carrier requires 256-aligned dimensions"
        )

    data_sr, sr_axes, rng_seed, rng_subsequence = _plain_qkv_split3_sr_policy()
    if not data_sr or sr_axes != "row":
        raise RuntimeError(
            "paired localCTA QKV carrier requires row-only gradient data SR"
        )
    if _env_flag("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", False) or _env_flag(
        "NVFP4_SCALE_SR_GRAD", False
    ):
        raise RuntimeError(
            "paired localCTA QKV carrier requires gradient scale SR off"
        )

    use_rht = _localcta_paired_rht_role_enabled("grad")
    use_activation_rht = _localcta_paired_rht_role_enabled("activation")
    use_weight_rht = _localcta_paired_rht_role_enabled("weight")
    if use_rht != use_activation_rht or use_weight_rht:
        raise RuntimeError(
            "paired localCTA QKV carrier requires matched activation/gradient "
            "RHT enable bits and weight RHT off"
        )
    axes = os.environ.get("NVFP4_RHT_AXES", "both").strip().lower()
    if axes not in {"col", "cols", "column", "columns"}:
        raise RuntimeError("paired localCTA QKV carrier supports column RHT only")
    fixed_sign_geometry = _env_flag("NVFP4_RHT_RANDOM_SIGNS", False)
    if not fixed_sign_geometry:
        raise RuntimeError(
            "paired localCTA QKV carrier requires the sealed fixed-sign geometry"
        )

    native_marker = getattr(
        tkq, "tk_localcta_split3_supports_paired_rht", None
    )
    native_call = getattr(
        tkq, "tk_group_quantize_dim1_split3_for_gemm_paired_rht", None
    )
    use_native_split3 = (
        use_rht
        and _env_flag("USE_TK_LOCALCTA_NATIVE_PAIRED_RHT_SPLIT3", True)
        and native_marker is not None
        and native_call is not None
        and native_marker()
    )
    if use_native_split3:
        result = native_call(
            *grads,
            data_stochastic_rounding=True,
            rng_seed=rng_seed,
            rng_subsequence_base=rng_subsequence,
            data_sr_axes="row",
            persistent_rng_state=persistent_rng_state,
            encode_centric=_env_flag("NVTE_NVFP4_ENCODE_CENTRIC", False),
        )
        if len(result) != 12:
            raise RuntimeError(
                "native paired localCTA QKV split3 producer returned a stale ABI"
            )
        package = _parse_localcta_split3_dgrad_package(result, M, grads[0].device)
        package["N_dims"] = list(N_dims)
        package["from_concat"] = True
        # The native split3 producer aliases all three per-split row SG views
        # to the one shared QKV outer scale, but its legacy full-row tuple slot
        # may still be an empty compatibility tensor.  Normalize the package
        # to the same full-carrier ABI as the concatenated opt fallback.
        package["a_sc_full"] = package["a_sc_cat"]
        package["a_sg_full"] = package["a_sg_list"][0]
        # The native RHT branch uses the same raw-scale + outer-SG finalizer as
        # the concatenated opt producer, independent of the legacy split3
        # fold-in-producer tuning flag.
        package["a_sc_outer_folded"] = False
        package["paired_rht_backend"] = "native_split3"
        package["paired_rht_keepalive"] = (result,)
        return package

    combined = torch.cat(grads, dim=1)
    result = tkq.tk_quantize_for_gemm_opt(
        combined,
        True,
        _env_flag("NVTE_NVFP4_ENCODE_CENTRIC", False),
        True,
        False,
        "col" if use_rht else "none",
        use_rht and fixed_sign_geometry,
        rng_seed,
        rng_subsequence,
        "row",
        persistent_rng_state,
    )
    if len(result) < 6:
        raise RuntimeError("paired localCTA QKV opt producer returned a stale ABI")
    row_fp4_full, row_sc_full, col_fp4_full, col_sc_full, row_sg, col_sg_full = result[:6]

    row_fp4_list = []
    row_sc_list = []
    col_fp4_list = []
    col_sc_list = []
    col_sg_list = []
    fp4_offset = row_sc_offset = col_offset = col_sc_offset = col_sg_offset = 0
    for width in N_dims:
        width = int(width)
        row_fp4_list.append(
            _narrow_packed_fp4_contiguous(
                row_fp4_full, 1, fp4_offset, width // 2
            )
        )
        row_sc_list.append(
            row_sc_full.narrow(1, row_sc_offset, width // 64).contiguous()
        )
        col_fp4_list.append(
            _narrow_packed_fp4_contiguous(
                col_fp4_full, 0, col_offset, width
            )
        )
        col_sc_list.append(
            col_sc_full.narrow(0, col_sc_offset, width // 128).contiguous()
        )
        col_sg_list.append(
            col_sg_full.narrow(1, col_sg_offset, width // 256)
        )
        fp4_offset += width // 2
        row_sc_offset += width // 64
        col_offset += width
        col_sc_offset += width // 128
        col_sg_offset += width // 256

    row_sg_list = [row_sg, row_sg, row_sg]
    return {
        "M": M,
        "device": grads[0].device,
        "from_concat": True,
        "prepared_nofold": False,
        "consumer_contract": "folded_prepared",
        "N_dims": list(N_dims),
        "fp4_row_list": row_fp4_list,
        "sc_row_list": row_sc_list,
        "row_sg_list": row_sg_list,
        "a_fp4_list": row_fp4_list,
        "a_sc_list": row_sc_list,
        "a_sg_list": row_sg_list,
        "a_fp4_full": row_fp4_full,
        "a_sc_cat": row_sc_full,
        "a_sc_full": row_sc_full,
        "a_sg_full": row_sg,
        "fp4_col_list": col_fp4_list,
        "sc_col_list": col_sc_list,
        "col_sg_list": col_sg_list,
        "fp4_col_full": col_fp4_full,
        "sc_col_cat": col_sc_full,
        "col_sg_cat": col_sg_full,
        "paired_rht_backend": "concat_opt_fallback",
        "paired_rht_keepalive": (combined, result),
    }


def _localcta_grouped_k_dgrad_package(
    dy_input,
    N_dims,
    defer_col=False,
    rope_live64_cs=None,
    rope_seq_len: int = 0,
    persistent_rng_state=None,
):
    """Quantize localCTA QKV grads into reusable split3 dgrad/wgrad payloads."""
    tkq = _get_tk_quant_for_gemm()
    if not _is_localcta_quant_mod(tkq):
        raise RuntimeError("_localcta_grouped_k_dgrad_package requires localCTA quant")
    if len(N_dims) != 3:
        raise ValueError(f"split3 package expects 3 groups, got {len(N_dims)}")

    debug_pkg_timings = os.environ.get('USE_TK_QKV_PACKAGE_DEBUG_TIMINGS', '0') == '1'
    pkg_timing_events = []

    def _pkg_mark(label: str) -> None:
        if not debug_pkg_timings:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream())
        pkg_timing_events.append((label, event))

    def _layout(t: torch.Tensor) -> str:
        base = getattr(t, '_base', None)
        return (
            f"shape={tuple(t.shape)} stride={tuple(t.stride())} "
            f"dtype={t.dtype} contig={t.is_contiguous()} "
            f"offset={t.storage_offset()} base_numel={base.numel() if base is not None else None}"
        )

    def _pkg_emit(mode: str, g0: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor) -> None:
        if not debug_pkg_timings or not pkg_timing_events:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(torch.cuda.current_stream())
        end_event.synchronize()
        events = pkg_timing_events + [('end', end_event)]
        parts = [
            f"{label}->{next_label}={event.elapsed_time(next_event):.3f}ms"
            for (label, event), (next_label, next_event) in zip(events, events[1:])
        ]
        print(
            f"[TK QKV PACKAGE TIMINGS] mode={mode} "
            + " ".join(parts)
            + f" q({_layout(g0)}) k({_layout(g1)}) v({_layout(g2)})",
            file=sys.stderr,
            flush=True,
        )

    _pkg_mark('start')

    if use_tk_qkv_localcta_tk_prepared_activation():
        if persistent_rng_state is not None:
            raise RuntimeError(
                "checkpointed localCTA SR cannot use the standalone prepared "
                "QKV activation quantizer"
            )
        tk_q = _get_tk_quant_standalone()
        use_split_inputs = (
            isinstance(dy_input, (tuple, list))
            and len(dy_input) == len(N_dims)
            and hasattr(tk_q, 'tk_group_quantize_dim1_split3_for_gemm')
        )
        if use_split_inputs:
            g0 = _as_bf16_contiguous(dy_input[0])
            g1 = _as_bf16_contiguous(dy_input[1])
            g2 = _as_bf16_contiguous(dy_input[2])
            result = _plain_qkv_split3_quantize_eager(tk_q, g0, g1, g2)
            M = g0.shape[0]
            device = g0.device
        else:
            if isinstance(dy_input, (tuple, list)):
                g0, g1, g2 = [
                    _as_bf16_contiguous(part) for part in dy_input
                ]
                dy_cat = torch.cat((g0, g1, g2), dim=1)
                M = g0.shape[0]
                device = g0.device
            else:
                dy_cat = _as_bf16_contiguous(dy_input)
                M = dy_cat.shape[0]
                device = dy_cat.device
            result = tk_q.tk_group_quantize_dim1_for_gemm(dy_cat, N_dims)

        package = _parse_localcta_split3_dgrad_package(
            _adapt_standalone_dim1_quant_for_localcta_fast(
                result, N_dims, M, device
            ),
            M,
            device,
        )
        package['N_dims'] = list(N_dims)
        return package

    use_split_inputs = (
        isinstance(dy_input, (tuple, list))
        and len(dy_input) == len(N_dims)
        and hasattr(tkq, 'tk_group_quantize_dim1_split3_for_gemm')
    )
    if _use_localcta_paired_rht_carrier():
        if not isinstance(dy_input, (tuple, list)) or len(dy_input) != 3:
            raise RuntimeError(
                "paired localCTA QKV carrier requires explicit Q/K/V inputs"
            )
        if rope_live64_cs is not None:
            raise RuntimeError(
                "paired localCTA QKV carrier requires BF16 inverse RoPE upstream"
            )
        g0, g1, g2 = (
            _as_bf16_contiguous(dy_input[0]),
            _as_bf16_contiguous(dy_input[1]),
            _as_bf16_contiguous(dy_input[2]),
        )
        package = _run_localcta_qkv_quant_with_scale_override(
            tkq,
            _localcta_paired_rht_split3_package,
            tkq,
            g0,
            g1,
            g2,
            N_dims,
            persistent_rng_state=persistent_rng_state,
        )
        _trace_backend_choice(
            'localcta_qkv_bwd_quant',
            'paired_rht_split3_carrier_'
            + package.get('paired_rht_backend', 'unknown'),
        )
        return package
    use_fast_concat = (
        get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_fast_qkv_dim1_concat()
        and isinstance(dy_input, (tuple, list))
        and len(dy_input) == len(N_dims)
        and hasattr(tkq, 'tk_concat_group_quantize_dim1_for_gemm_fast')
        and persistent_rng_state is None
    )
    if use_fast_concat:
        g0 = _as_bf16_contiguous(dy_input[0])
        g1 = _as_bf16_contiguous(dy_input[1])
        g2 = _as_bf16_contiguous(dy_input[2])
        M = g0.shape[0]
        device = g0.device
        fast_results = [
            tkq.tk_quantize_for_gemm_fast(g, True, True)
            for g in (g0, g1, g2)
        ]
        row_fp4_list = [r[0] for r in fast_results]
        row_sc_prepared_list = [r[6] for r in fast_results]
        row_sg_list = [
            r[4].to(torch.float32) if r[4].dtype != torch.float32 else r[4]
            for r in fast_results
        ]
        col_fp4_list = [r[2] for r in fast_results]
        col_sc_prepared_list = [r[7] for r in fast_results]
        col_sg_list = [
            r[5].to(torch.float32) if r[5].dtype != torch.float32 else r[5]
            for r in fast_results
        ]
        return {
            'M': M,
            'device': device,
            'from_concat': True,
            'N_dims': list(N_dims),
            'fp4_row_list': None,
            'sc_row_list': None,
            'row_sg_list': row_sg_list,
            'a_fp4_list': None,
            'a_sc_list': [
                sc.contiguous().view(torch.float8_e4m3fn)
                for sc in row_sc_prepared_list
            ],
            'a_sg_list': row_sg_list,
            'a_fp4_full': torch.cat(row_fp4_list, dim=1),
            'a_sc_cat': None,
            'a_sc_full': torch.cat(row_sc_prepared_list, dim=1),
            'a_sg_full': None,
            'fp4_col_list': col_fp4_list,
            'sc_col_list': col_sc_prepared_list,
            'col_sg_list': col_sg_list,
            'fp4_col_full': torch.cat(col_fp4_list, dim=0),
            'sc_col_cat': torch.cat(col_sc_prepared_list, dim=0),
            'col_sg_cat': torch.cat(col_sg_list, dim=0),
        }
    if use_split_inputs:
        allow_strided_inputs = _use_localcta_v4_strided_qkv_grad_quant()
        g0 = _as_bf16_quant_input(dy_input[0], allow_strided=allow_strided_inputs)
        g1 = _as_bf16_quant_input(dy_input[1], allow_strided=allow_strided_inputs)
        g2 = _as_bf16_quant_input(dy_input[2], allow_strided=allow_strided_inputs)
        _pkg_mark('inputs_ready')
        if (
            rope_live64_cs is not None
            and get_tk_localcta_variant() == 'v4'
            and hasattr(tkq, 'tk_group_quantize_dim1_split3_for_gemm_inverse_rope_live64')
        ):
            M = g0.shape[0]
            device = g0.device
            _trace_backend_choice('localcta_qkv_bwd_quant', 'inverse_rope_live64_split3')
            result = _run_localcta_qkv_quant_with_scale_override(
                tkq,
                tkq.tk_group_quantize_dim1_split3_for_gemm_inverse_rope_live64,
                g0,
                g1,
                g2,
                rope_live64_cs,
                int(rope_seq_len),
                persistent_rng_state=persistent_rng_state,
            )
            _pkg_mark('quant_done')
            package = _parse_localcta_split3_dgrad_package(result, M, device)
            package['N_dims'] = list(N_dims)
            package['inverse_rope_live64'] = True
            _pkg_mark('parse_done')
            _pkg_emit('inverse_rope_live64_split3', g0, g1, g2)
            return package
        if (
            use_tk_qkv_localcta_prepared_split3_separate()
            and persistent_rng_state is None
        ):
            package = _localcta_single_quant_to_split3_package(tkq, g0, g1, g2)
            package['N_dims'] = list(N_dims)
            return package
        M = g0.shape[0]
        device = g0.device
        if (
            defer_col
            and persistent_rng_state is None
            and hasattr(tkq, 'tk_group_quantize_dim1_split3_rowphase_for_gemm')
            and hasattr(tkq, 'tk_group_quantize_dim1_split3_finalize_col_inplace')
        ):
            result = tkq.tk_group_quantize_dim1_split3_rowphase_for_gemm(g0, g1, g2)
            row_fp4_list, sc_row_list, row_sg_list, \
                a_fp4_full, a_sc_cat, a_sg_full, \
                fp4_col_full, sc_col_cat, col_sg_cat, \
                col_sg_chunk_0, col_sg_chunk_1, col_sg_chunk_2 = result
            return {
                'M': M,
                'device': device,
                'from_concat': False,
                'N_dims': list(N_dims),
                'fp4_row_list': row_fp4_list,
                'sc_row_list': sc_row_list,
                'row_sg_list': row_sg_list,
                'a_fp4_list': None,
                'a_sc_list': list(sc_row_list),
                'a_sg_list': [
                    sg.to(torch.float32) if sg.dtype != torch.float32 else sg
                    for sg in row_sg_list
                ],
                'fp4_col_list': None,
                'sc_col_list': None,
                'col_sg_list': None,
                'a_fp4_full': a_fp4_full,
                'a_sc_cat': a_sc_cat,
                'a_sg_full': a_sg_full,
                'fp4_col_full': fp4_col_full,
                'sc_col_cat': sc_col_cat,
                'col_sg_cat': col_sg_cat,
                'deferred_col_state': {
                    'col_sc_cat': sc_col_cat,
                    'col_sg_cat': col_sg_cat,
                    'col_sg_chunk_0': col_sg_chunk_0,
                    'col_sg_chunk_1': col_sg_chunk_1,
                    'col_sg_chunk_2': col_sg_chunk_2,
                },
            }
        result = _run_localcta_qkv_quant_with_scale_override(
            tkq,
            tkq.tk_group_quantize_dim1_split3_for_gemm,
            g0,
            g1,
            g2,
            persistent_rng_state=persistent_rng_state,
        )
        _pkg_mark('quant_done')
        package = _parse_localcta_split3_dgrad_package(result, M, device)
        package['N_dims'] = list(N_dims)
        _pkg_mark('parse_done')
        _pkg_emit('split3', g0, g1, g2)
        return package

    use_concat_inputs = (
        isinstance(dy_input, (tuple, list))
        and len(dy_input) == len(N_dims)
        and hasattr(tkq, 'tk_concat_group_quantize_dim1_for_gemm')
        and persistent_rng_state is None
    )
    if use_concat_inputs:
        g0 = _as_bf16_contiguous(dy_input[0])
        g1 = _as_bf16_contiguous(dy_input[1])
        g2 = _as_bf16_contiguous(dy_input[2])
        M = g0.shape[0]
        device = g0.device
        dy_cat = torch.cat((g0, g1, g2), dim=1)
        result = _run_localcta_qkv_quant_with_scale_override(
            tkq,
            tkq.tk_concat_group_quantize_dim1_for_gemm,
            dy_cat,
            N_dims,
        )
        if len(result) != 9:
            package = _parse_localcta_split3_dgrad_package(result, M, device)
            package['N_dims'] = list(N_dims)
            return package
        return _parse_localcta_concat_dgrad_package(result, N_dims, M, device)
    else:
        dy_cat = dy_input if isinstance(dy_input, torch.Tensor) else torch.cat(list(dy_input), dim=1)
        dy_cat = _as_bf16_contiguous(dy_cat)
        if (
            persistent_rng_state is None
            and hasattr(tkq, 'tk_concat_group_quantize_dim1_for_gemm')
        ):
            result = _run_localcta_qkv_quant_with_scale_override(
                tkq,
                tkq.tk_concat_group_quantize_dim1_for_gemm,
                dy_cat,
                N_dims,
            )
            M = dy_cat.shape[0]
            device = dy_cat.device
            if len(result) != 9:
                package = _parse_localcta_split3_dgrad_package(result, M, device)
                package['N_dims'] = list(N_dims)
                return package
            return _parse_localcta_concat_dgrad_package(result, N_dims, M, device)
        if hasattr(tkq, 'tk_group_quantize_dim1_split3_for_gemm'):
            g0, g1, g2 = [
                part.contiguous() for part in torch.split(dy_cat, N_dims, dim=1)
            ]
            if (
                use_tk_qkv_localcta_prepared_split3_separate()
                and persistent_rng_state is None
            ):
                package = _localcta_single_quant_to_split3_package(tkq, g0, g1, g2)
                package['N_dims'] = list(N_dims)
                return package
            result = _run_localcta_qkv_quant_with_scale_override(
                tkq,
                tkq.tk_group_quantize_dim1_split3_for_gemm,
                g0,
                g1,
                g2,
                persistent_rng_state=persistent_rng_state,
            )
        else:
            if persistent_rng_state is not None:
                raise RuntimeError(
                    "localCTA extension lacks the checkpointed v4 split3 SR ABI"
                )
            result = _run_localcta_qkv_quant_with_scale_override(
                tkq,
                tkq.tk_group_quantize_dim1_for_gemm,
                dy_cat,
                N_dims,
            )
        M = dy_cat.shape[0]
        device = dy_cat.device

    package = _parse_localcta_split3_dgrad_package(result, M, device)
    package['N_dims'] = list(N_dims)
    return package


def _localcta_grouped_k_dgrad_backend(dgrad_package, w_nvfp4, N_dims, dx=None,
                                      prefer_split3=False, prefer_strided=False,
                                      a_col_offsets=None, a_col_widths=None,
                                      debug_name=None):
    """Run the localCTA QKV split3 dgrad backend on a prebuilt package."""
    tk = _get_tk()
    tk_sg = _get_tk_localcta_direct()
    use_nofold_consumer = bool(dgrad_package.get('prepared_nofold'))
    dgrad_tk = tk_sg if use_nofold_consumer and tk_sg is not None else tk
    w_fp4_c, w_sc_c, w_sg_c = w_nvfp4._tk_col
    K = w_fp4_c.shape[0]
    cached_col_splits = getattr(w_nvfp4, '_tk_col_splits', None)
    cached_col_sg_splits = getattr(w_nvfp4, '_tk_col_sg_splits', None)
    if cached_col_splits is not None:
        B_fp4_list, B_sc_list = cached_col_splits
        if cached_col_sg_splits is not None:
            B_sg_list = cached_col_sg_splits
        else:
            B_sg_list = _split_weight_col_sg_tensors(
                w_fp4_c, w_sg_c, N_dims, use_localcta_runtime=True
            )
    else:
        B_fp4_list, B_sc_list, B_sg_list = _split_weight_col_tensors(
            w_fp4_c, w_sc_c, w_sg_c, N_dims, use_localcta_runtime=True
        )
    if dx is None:
        dx = torch.empty(
            dgrad_package['M'], K,
            dtype=torch.bfloat16, device=dgrad_package['device']
        )

    backend_override = get_tk_qkv_localcta_dgrad_backend_override()
    if backend_override == 'direct_split':
        _trace_backend_choice('localcta_qkv_dgrad', 'direct_split')
        return _localcta_direct_split_dgrad_backend(dgrad_package, w_nvfp4, N_dims, dx=dx)

    force_strided_onepass = backend_override == 'strided_onepass'
    force_strided_sum = backend_override in {'strided_sum', 'strided'}
    force_split3 = backend_override == 'split3'
    force_batched_accum = backend_override == 'batched_accum'

    allow_strided = prefer_strided and not use_tk_qkv_disable_strided_dgrad()
    if backend_override != 'auto':
        allow_strided = allow_strided and backend_override in {
            'strided_onepass', 'strided_sum', 'strided'
        }

    strict_v4_contract = (
        use_tk_localcta_v4_strict_path()
        and _use_tk_localcta_for_m(dgrad_package['M'])
    )

    if strict_v4_contract:
        if (
            dgrad_package.get('a_fp4_full') is not None
            and hasattr(tk, 'nvfp4_split3_dgrad_strided_onepass_gemm')
            and use_tk_localcta_v4_fast_qkv_onepass_dgrad()
        ):
            if a_col_offsets is None or a_col_widths is None:
                a_col_offsets = []
                a_col_widths = []
                fp4_off = 0
                for n_i in N_dims:
                    fp4_cols = n_i // 2
                    a_col_offsets.append(fp4_off)
                    a_col_widths.append(fp4_cols)
                    fp4_off += fp4_cols
            if dgrad_package.get('a_sc_outer_folded', False):
                a_sc_folded = dgrad_package['a_sc_list']
            else:
                a_sc_folded = [
                    _fold_localcta_v4_outer_sg_into_prepared_sc(
                        dgrad_package['a_sc_list'][i],
                        dgrad_package['a_sg_list'][i],
                        dgrad_package['M'],
                        N_dims[i],
                    )
                    for i in range(len(N_dims))
                ]
            if (
                use_tk_localcta_v4_fullcol_qkv_dgrad()
                and hasattr(tk, 'nvfp4_split3_dgrad_strided_onepass_full_b_gemm')
            ):
                b_sg_full = B_sg_list[0] if isinstance(w_sg_c, (list, tuple)) else w_sg_c
                if (
                    use_tk_localcta_v4_fullcol_qkv_dgrad_direct_sg()
                    and hasattr(tk, 'nvfp4_split3_dgrad_strided_onepass_full_b_gemm_sg')
                ):
                    _trace_backend_choice('localcta_qkv_dgrad', 'strict_strided_onepass_full_b_direct_sg')
                    tk.nvfp4_split3_dgrad_strided_onepass_full_b_gemm_sg(
                        dgrad_package['a_fp4_full'],
                        dgrad_package['a_sc_list'],
                        dgrad_package['a_sg_list'],
                        a_col_offsets,
                        a_col_widths,
                        w_fp4_c,
                        w_sc_c,
                        b_sg_full,
                        dx,
                        tk_localcta_v4_split3_onepass_config_idx(),
                    )
                    return dx
                B_sc_folded_full = _fold_localcta_v4_outer_sg_into_prepared_sc(
                    w_sc_c,
                    b_sg_full,
                    w_fp4_c.size(0),
                    sum(N_dims),
                )
                _trace_backend_choice('localcta_qkv_dgrad', 'strict_strided_onepass_full_b')
                tk.nvfp4_split3_dgrad_strided_onepass_full_b_gemm(
                    dgrad_package['a_fp4_full'],
                    a_sc_folded,
                    a_col_offsets,
                    a_col_widths,
                    w_fp4_c,
                    B_sc_folded_full,
                    dx,
                    tk_localcta_v4_split3_onepass_config_idx(),
                )
                return dx
            _trace_backend_choice('localcta_qkv_dgrad', 'strict_strided_onepass')
            B_sc_folded = _fold_localcta_v4_weight_sc_outer_sg_cached(
                B_fp4_list, B_sc_list, B_sg_list
            )
            tk.nvfp4_split3_dgrad_strided_onepass_gemm(
                dgrad_package['a_fp4_full'],
                a_sc_folded,
                a_col_offsets,
                a_col_widths,
                B_fp4_list,
                B_sc_folded,
                B_sg_list,
                dx,
                tk_localcta_v4_split3_onepass_config_idx(),
            )
            return dx
        if (
            dgrad_package.get('a_fp4_full') is not None
            and hasattr(tk, 'nvfp4_split3_dgrad_strided_gemm')
        ):
            if a_col_offsets is None or a_col_widths is None:
                a_col_offsets = []
                a_col_widths = []
                fp4_off = 0
                for n_i in N_dims:
                    fp4_cols = n_i // 2
                    a_col_offsets.append(fp4_off)
                    a_col_widths.append(fp4_cols)
                    fp4_off += fp4_cols
            tk.nvfp4_split3_dgrad_strided_gemm(
                dgrad_package['a_fp4_full'],
                dgrad_package['a_sc_list'],
                dgrad_package['a_sg_list'],
                a_col_offsets,
                a_col_widths,
                B_fp4_list,
                B_sc_list,
                B_sg_list,
                dx,
            )
            _trace_backend_choice('localcta_qkv_dgrad', 'strict_strided_sum')
            return dx

        a_fp4_list, a_sc_list, a_sg_list = _ensure_localcta_row_split_payload(
            dgrad_package, N_dims
        )
        if len(N_dims) == 3 and hasattr(tk, 'nvfp4_split3_dgrad_gemm'):
            _trace_backend_choice('localcta_qkv_dgrad', 'strict_split3')
            tk.nvfp4_split3_dgrad_gemm(
                a_fp4_list,
                a_sc_list,
                a_sg_list,
                B_fp4_list,
                B_sc_list,
                B_sg_list,
                dx,
            )
            return dx
        _trace_backend_choice('localcta_qkv_dgrad', 'strict_batched_accum')
        tk.nvfp4_batched_accum_gemm(
            a_fp4_list,
            a_sc_list,
            a_sg_list,
            B_fp4_list,
            B_sc_list,
            B_sg_list,
            dx,
        )
        return dx

    if (
        get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_sg_direct_consumers()
        and _use_tk_localcta_for_m(dgrad_package['M'])
    ):
        tk_direct = _get_tk_localcta_direct()
        if (
            use_tk_localcta_v4_cpp_only()
            and dgrad_package.get('a_fp4_full') is not None
            and hasattr(tk, 'nvfp4_split3_dgrad_strided_gemm')
            and use_tk_localcta_v4_fast_qkv_onepass_dgrad()
        ):
            if a_col_offsets is None or a_col_widths is None:
                a_col_offsets = []
                a_col_widths = []
                fp4_off = 0
                for n_i in N_dims:
                    fp4_cols = n_i // 2
                    a_col_offsets.append(fp4_off)
                    a_col_widths.append(fp4_cols)
                    fp4_off += fp4_cols
            _trace_backend_choice('localcta_qkv_dgrad', 'v4_cpp_strided_sum')
            tk.nvfp4_split3_dgrad_strided_gemm(
                dgrad_package['a_fp4_full'],
                dgrad_package['a_sc_list'],
                dgrad_package['a_sg_list'],
                a_col_offsets,
                a_col_widths,
                B_fp4_list,
                B_sc_list,
                B_sg_list,
                dx,
            )
            return dx
        if (
            use_tk_localcta_v4_cpp_only()
            and dgrad_package.get('a_fp4_full') is not None
            and tk_direct is not None
            and hasattr(tk_direct, 'nvfp4_split3_dgrad_strided_onepass_gemm')
            and use_tk_localcta_v4_fast_qkv_onepass_dgrad()
        ):
            if a_col_offsets is None or a_col_widths is None:
                a_col_offsets = []
                a_col_widths = []
                fp4_off = 0
                for n_i in N_dims:
                    fp4_cols = n_i // 2
                    a_col_offsets.append(fp4_off)
                    a_col_widths.append(fp4_cols)
                    fp4_off += fp4_cols
            _trace_backend_choice('localcta_qkv_dgrad', 'v4_cpp_strided_onepass')
            tk_direct.nvfp4_split3_dgrad_strided_onepass_gemm(
                dgrad_package['a_fp4_full'],
                dgrad_package['a_sc_list'],
                a_col_offsets,
                a_col_widths,
                B_fp4_list,
                B_sc_list,
                dx,
                tk_localcta_v4_split3_onepass_config_idx(),
            )
            return dx
        mode = tk_localcta_v4_split3_dgrad_sg_mode()
        if mode != 'batched':
            a_fp4_list, a_sc_list, a_sg_list = _ensure_localcta_row_split_payload(
                dgrad_package, N_dims
            )
            if tk_direct is None or not hasattr(tk_direct, 'nvfp4_gemm'):
                raise RuntimeError("localCTA direct GEMM backend is unavailable for v4 split3 dgrad")
            bufs = _get_dgrad_bufs(len(N_dims), dgrad_package['M'], K, dgrad_package['device'])
            dx.zero_()
            for i in range(len(a_fp4_list)):
                a_sg_outer = _prepare_localcta_v4_outer_sg_for_direct(
                    a_sg_list[i], a_fp4_list[i].size(0) // 256, a_fp4_list[i].device
                )
                b_sg_outer = _prepare_localcta_v4_outer_sg_for_direct(
                    B_sg_list[i], B_fp4_list[i].size(0) // 256, B_fp4_list[i].device
                )
                if mode == 'a1_b':
                    a_sg_direct = torch.ones_like(a_sg_outer)
                    b_sg_direct = b_sg_outer
                elif mode == 'a_b1':
                    a_sg_direct = a_sg_outer
                    b_sg_direct = torch.ones_like(b_sg_outer)
                elif mode == 'a_b':
                    a_sg_direct = a_sg_outer
                    b_sg_direct = b_sg_outer
                elif mode == 'a1_b1':
                    a_sg_direct = torch.ones_like(a_sg_outer)
                    b_sg_direct = torch.ones_like(b_sg_outer)
                elif mode == 'afold_b':
                    a_sc_list[i] = _fold_localcta_v4_sg_into_prepared_sc(
                        a_sc_list[i],
                        a_sg_list[i],
                        a_fp4_list[i].size(0),
                        a_fp4_list[i].size(1) * 2,
                    )
                    a_sg_direct = torch.ones_like(a_sg_outer)
                    b_sg_direct = b_sg_outer
                else:
                    raise ValueError(
                        "USE_TK_LOCALCTA_V4_SPLIT3_DGRAD_SG_MODE must be one of "
                        "{'a1_b', 'a_b1', 'a_b', 'a1_b1', 'afold_b', 'batched'}"
                    )
                if _debug_qkv_capture_path():
                    _append_qkv_capture({
                        "event": "qkv_split3_direct_dgrad_prelaunch",
                        "debug_name": debug_name,
                        "split_index": i,
                        "N_dims": list(N_dims),
                        "sg_mode": mode,
                        "inputs": {
                            "a_fp4_shape": list(a_fp4_list[i].shape),
                            "a_sc_shape": list(a_sc_list[i].shape),
                            "a_sg_shape": list(a_sg_list[i].shape),
                            "a_sg_outer_shape": list(a_sg_outer.shape),
                            "a_sg_direct_shape": list(a_sg_direct.shape),
                            "b_fp4_shape": list(B_fp4_list[i].shape),
                            "b_sc_shape": list(B_sc_list[i].shape),
                            "b_sg_shape": list(B_sg_list[i].shape),
                            "b_sg_outer_shape": list(b_sg_outer.shape),
                            "b_sg_direct_shape": list(b_sg_direct.shape),
                        },
                    })
                tk_direct.nvfp4_gemm(
                    a_fp4_list[i],
                    a_sc_list[i].contiguous(),
                    a_sg_direct,
                    B_fp4_list[i],
                    B_sc_list[i].contiguous(),
                    b_sg_direct,
                    bufs[i],
                )
                dx.add_(bufs[i])
            if _debug_qkv_capture_path():
                _append_qkv_capture({
                    "event": "qkv_split3_direct_dgrad",
                    "debug_name": debug_name,
                    "N_dims": list(N_dims),
                    "sg_mode": mode,
                    "outputs": {
                        "splits": [_tensor_debug_stats(buf) for buf in bufs],
                        "sum": _tensor_debug_stats(dx),
                        "a_sg": [_tensor_debug_stats(sg) for sg in a_sg_list],
                        "b_sg": [_tensor_debug_stats(sg) for sg in B_sg_list],
                        "a_sg_direct": [
                            _tensor_debug_stats(
                                _prepare_localcta_v4_outer_sg_for_direct(
                                    sg, a_fp4_list[i].size(0) // 256, a_fp4_list[i].device
                                )
                            )
                            for i, sg in enumerate(a_sg_list)
                        ],
                        "b_sg_direct": [
                            _tensor_debug_stats(
                                _prepare_localcta_v4_outer_sg_for_direct(
                                    sg, B_fp4_list[i].size(0) // 256, B_fp4_list[i].device
                                )
                            )
                            for i, sg in enumerate(B_sg_list)
                        ],
                    },
                })
            return dx

    if (
        allow_strided
        and _use_localcta_v3_runtime()
        and dgrad_package.get('a_fp4_full') is not None
        and hasattr(dgrad_tk, 'nvfp4_batched_gemm_strided')
        and not (
            len(N_dims) == 3 and (
                hasattr(dgrad_tk, 'nvfp4_split3_dgrad_strided_onepass_gemm') or
                hasattr(dgrad_tk, 'nvfp4_split3_dgrad_strided_gemm')
            )
        )
    ):
        a_sc_list = dgrad_package['a_sc_list']
        a_sg_list = dgrad_package['a_sg_list']
        if a_col_offsets is None or a_col_widths is None:
            a_col_offsets = []
            a_col_widths = []
            fp4_off = 0
            for n_i in N_dims:
                fp4_cols = n_i // 2
                a_col_offsets.append(fp4_off)
                a_col_widths.append(fp4_cols)
                fp4_off += fp4_cols

        bufs = _get_dgrad_bufs(len(N_dims), dgrad_package['M'], K, dgrad_package['device'])
        dgrad_tk.nvfp4_batched_gemm_strided(
            dgrad_package['a_fp4_full'],
            a_sc_list,
            a_sg_list,
            a_col_offsets,
            a_col_widths,
            B_fp4_list,
            B_sc_list,
            B_sg_list,
            bufs,
        )
        dx.copy_(bufs[0])
        for i in range(1, len(N_dims)):
            dx.add_(bufs[i])
        _trace_backend_choice('localcta_qkv_dgrad', 'v3_batched_strided')
        return dx

    native_v4_raw_contract = (
        get_tk_localcta_variant() == 'v4'
        and use_tk_localcta_v4_cpp_only()
    )

    if ((force_strided_onepass or (
            allow_strided and not native_v4_raw_contract and not _use_localcta_v3_tilegrid256()))
            and
            dgrad_package.get('a_fp4_full') is not None and
            hasattr(dgrad_tk, 'nvfp4_split3_dgrad_strided_onepass_gemm')):
        a_sc_list = dgrad_package['a_sc_list']
        if a_col_offsets is None or a_col_widths is None:
            a_col_offsets = []
            a_col_widths = []
            fp4_off = 0
            for n_i in N_dims:
                fp4_cols = n_i // 2
                a_col_offsets.append(fp4_off)
                a_col_widths.append(fp4_cols)
                fp4_off += fp4_cols

        dgrad_tk.nvfp4_split3_dgrad_strided_onepass_gemm(
            dgrad_package['a_fp4_full'],
            a_sc_list,
            a_col_offsets,
            a_col_widths,
            B_fp4_list,
            B_sc_list,
            B_sg_list,
            dx,
        )
        _trace_backend_choice('localcta_qkv_dgrad', 'strided_onepass')
        return dx

    if ((force_strided_sum or (allow_strided and not _use_localcta_v3_tilegrid256()))
            and dgrad_package.get('a_fp4_full') is not None and
            hasattr(dgrad_tk, 'nvfp4_split3_dgrad_strided_gemm')):
        a_sc_list = dgrad_package['a_sc_list']
        a_sg_list = dgrad_package['a_sg_list']
        if a_col_offsets is None or a_col_widths is None:
            a_col_offsets = []
            a_col_widths = []
            fp4_off = 0
            for n_i in N_dims:
                fp4_cols = n_i // 2
                a_col_offsets.append(fp4_off)
                a_col_widths.append(fp4_cols)
                fp4_off += fp4_cols

        dgrad_tk.nvfp4_split3_dgrad_strided_gemm(
            dgrad_package['a_fp4_full'],
            a_sc_list,
            a_sg_list,
            a_col_offsets,
            a_col_widths,
            B_fp4_list,
            B_sc_list,
            B_sg_list,
            dx,
        )
        _trace_backend_choice('localcta_qkv_dgrad', 'strided_sum')
        return dx

    a_fp4_list, a_sc_list, a_sg_list = _ensure_localcta_row_split_payload(
        dgrad_package, N_dims
    )
    a_sc_list = [
        sc.contiguous() if torch.is_tensor(sc) else sc
        for sc in a_sc_list
    ]
    a_sg_list = [
        sg.contiguous() if torch.is_tensor(sg) else sg
        for sg in a_sg_list
    ]
    B_sg_list = [
        sg.contiguous() if torch.is_tensor(sg) else sg
        for sg in B_sg_list
    ]
    if not force_batched_accum and (
        force_split3 or (prefer_split3 and len(N_dims) == 3 and hasattr(dgrad_tk, 'nvfp4_split3_dgrad_gemm'))
    ) and hasattr(dgrad_tk, 'nvfp4_split3_dgrad_gemm'):
        dgrad_tk.nvfp4_split3_dgrad_gemm(
            a_fp4_list,
            a_sc_list,
            a_sg_list,
            B_fp4_list,
            B_sc_list,
            B_sg_list,
            dx,
        )
        _trace_backend_choice('localcta_qkv_dgrad', 'split3')
    elif prefer_split3 and _use_localcta_v3_runtime():
        bufs = _get_dgrad_bufs(len(N_dims), dgrad_package['M'], K, dgrad_package['device'])
        for i in range(len(N_dims)):
            tk_dispatch_gemm(
                tk,
                a_fp4_list[i], a_sc_list[i], a_sg_list[i],
                B_fp4_list[i], B_sc_list[i], B_sg_list[i],
                bufs[i],
            )
        dx.copy_(bufs[0])
        for i in range(1, len(N_dims)):
            dx.add_(bufs[i])
        _trace_backend_choice('localcta_qkv_dgrad', 'v3_single_gemm_sum')
    else:
        dgrad_tk.nvfp4_batched_accum_gemm(
            a_fp4_list,
            a_sc_list,
            a_sg_list,
            B_fp4_list,
            B_sc_list,
            B_sg_list,
            dx,
        )
        _trace_backend_choice('localcta_qkv_dgrad', 'batched_accum')
    return dx


def _localcta_direct_split_dgrad_backend(dgrad_package, w_nvfp4, N_dims, dx=None):
    """Correctness-first localCTA dgrad using direct single-GEMM launches."""
    tk = _get_tk()
    tk_direct = _get_tk_localcta_direct()
    direct_gemm = getattr(tk_direct, 'nvfp4_gemm', None) if tk_direct is not None else None
    w_fp4_c, w_sc_c, w_sg_c = w_nvfp4._tk_col
    K = w_fp4_c.shape[0]
    cached_col_splits = getattr(w_nvfp4, '_tk_col_splits', None)
    cached_col_sg_splits = getattr(w_nvfp4, '_tk_col_sg_splits', None)
    if cached_col_splits is not None:
        B_fp4_list, B_sc_list = cached_col_splits
        if cached_col_sg_splits is not None:
            B_sg_list = cached_col_sg_splits
        else:
            B_sg_list = _split_weight_col_sg_tensors(
                w_fp4_c, w_sg_c, N_dims, use_localcta_runtime=True
            )
    else:
        B_fp4_list, B_sc_list, B_sg_list = _split_weight_col_tensors(
            w_fp4_c, w_sc_c, w_sg_c, N_dims, use_localcta_runtime=True
        )
    if dx is None:
        dx = torch.empty(
            dgrad_package['M'], K,
            dtype=torch.bfloat16, device=dgrad_package['device']
        )
    tmp_bufs = _get_dgrad_bufs(len(N_dims), dgrad_package['M'], K, dgrad_package['device'])
    a_fp4_list, a_sc_list, a_sg_list = _ensure_localcta_row_split_payload(
        dgrad_package, N_dims
    )
    for i in range(len(N_dims)):
        a_sg_i = a_sg_list[i]
        b_sg_i = B_sg_list[i]
        if direct_gemm is not None:
            if torch.is_tensor(a_sg_i) and a_sg_i.dim() != 2:
                a_sg_i = _localcta_expand_sg_grid(
                    a_sg_i, dgrad_package['M'], N_dims[i]
                )
            if torch.is_tensor(a_sg_i):
                a_sg_i = a_sg_i.contiguous()
            if torch.is_tensor(b_sg_i) and b_sg_i.dim() != 2:
                b_sg_i = _localcta_expand_sg_grid(b_sg_i, K, N_dims[i])
            if torch.is_tensor(b_sg_i):
                b_sg_i = b_sg_i.contiguous()
            direct_gemm(
                a_fp4_list[i],
                a_sc_list[i].contiguous(),
                a_sg_i,
                B_fp4_list[i],
                B_sc_list[i].contiguous(),
                b_sg_i,
                tmp_bufs[i],
            )
        else:
            tk.nvfp4_gemm(
                a_fp4_list[i],
                a_sc_list[i],
                a_sg_i,
                B_fp4_list[i],
                B_sc_list[i],
                b_sg_i,
                tmp_bufs[i],
            )
    dx.copy_(tmp_bufs[0])
    for buf in tmp_bufs[1:]:
        dx.add_(buf)
    return dx

def _get_dgrad_bufs(n_splits, M, K, device, owner_key=None):
    """Get or allocate cached per-split output buffers for dgrad."""
    key = (n_splits, M, K, device, _cache_owner_tag(owner_key))
    bufs = _dgrad_buf_cache.get(key)
    if bufs is None or bufs[0].shape != (M, K):
        bufs = [torch.empty(M, K, dtype=torch.bfloat16, device=device) for _ in range(n_splits)]
        _dgrad_buf_cache[key] = bufs
    return bufs


# Module-level cache for wgrad D buffer to avoid per-call allocation
_wgrad_buf_cache = {}

def _get_wgrad_buf(K, N_total, device, owner_key=None):
    """Get or allocate cached output buffer for wgrad."""
    key = (K, N_total, device, _cache_owner_tag(owner_key))
    buf = _wgrad_buf_cache.get(key)
    if buf is None or buf.shape != (K, N_total):
        buf = torch.empty(K, N_total, dtype=torch.bfloat16, device=device)
        _wgrad_buf_cache[key] = buf
    return buf


# Module-level cache for fused dgrad sum buffer
_dgrad_sum_cache = {}


def _cache_owner_tag(owner_key):
    return owner_key if owner_key is not None else "__shared__"


def _get_dgrad_sum_buf(M, K, device, owner_key=None):
    """Get or allocate cached output buffer for fused 3-way dgrad sum."""
    key = (M, K, device, _cache_owner_tag(owner_key))
    buf = _dgrad_sum_cache.get(key)
    if buf is None or buf.shape != (M, K):
        buf = torch.empty(M, K, dtype=torch.bfloat16, device=device)
        _dgrad_sum_cache[key] = buf
    return buf


def tk_split_dgrad(grad_splits, w_col, N_dims):
    """Streamlined per-split quant + batched GEMM + fused 3-way sum.

    Eliminates Python dispatch overhead vs calling _fast_quantize + batched_gemm
    separately:
      - Direct tk_quantize_for_gemm calls (no _TKQuantized wrapper)
      - Pre-allocated D_sum buffer with in-place add_ (no temp allocs)
      - Single function call from backward

    Args:
        grad_splits: tuple of (gq, gk, gv) bf16 tensors, each (M, N_i)
        w_col: _TKQuantized weight with _tk_col = (fp4, sc, sg)
        N_dims: list of int — [q_dim, k_dim, v_dim]

    Returns:
        dx_normed: (M, K) bf16 tensor
        dy_col_quant: 5-tuple for wgrad (fp4_col_list, sc_col_list, sg_per_group, None, None)
    """
    use_localcta_runtime = _use_tk_localcta_for_m(grad_splits[0].shape[0])
    tk_mod = _get_tk() if use_localcta_runtime else _get_tk_plain()
    tkq = _get_tk_quant_for_gemm() if use_localcta_runtime else _get_tk_quant_plain()
    n_groups = len(N_dims)
    M = grad_splits[0].shape[0]

    # 1. Quantize gradient splits
    #    When TK_SPLIT_ROW_COL_BWD=1:
    #      Row-only quant → [col-only on side stream ∥ GEMM+sum3 on main stream]
    #      Col-only overlaps with GEMM, hiding its cost behind compute.
    #    When TK_SPLIT_ROW_COL_BWD=0 (default):
    #      Row+col quant together → GEMM+sum3 (original path)
    _use_split = os.environ.get('TK_SPLIT_ROW_COL_BWD', '0') == '1'
    _has_col_only = hasattr(tkq, 'tk_quantize_col_only')
    _split = _use_split and _has_col_only

    # Prepare contiguous bf16 inputs
    grad_inputs = []
    for gs in grad_splits:
        g = gs if gs.is_contiguous() else gs.contiguous()
        if g.dtype != torch.bfloat16:
            g = g.to(torch.bfloat16)
        grad_inputs.append(g)

    # Row quant: return_transpose=False (split) or True (original)
    quant_results = []
    for g in grad_inputs:
        quant_results.append(tkq.tk_quantize_for_gemm(g, not _split))

    A_fp4_list = [r[0] for r in quant_results]
    A_sc_list = [r[1] for r in quant_results]
    A_sg_list = [r[4].to(torch.float32) if r[4].dtype != torch.float32 else r[4]
                 for r in quant_results]

    # 2. Weight col splits — cached by data_ptr
    w_fp4_c, w_sc_c, w_sg_c = w_col._tk_col
    K = w_fp4_c.shape[0]
    B_fp4_list, B_sc_list, B_sg_list = _split_weight_col_tensors(
        w_fp4_c, w_sc_c, w_sg_c, N_dims, use_localcta_runtime=use_localcta_runtime
    )

    # 3. Launch col-only on side stream (if split), overlapping with GEMM
    if _split:
        col_stream = _get_col_quant_stream()
        # Side stream waits for row quant to finish (same input tensors)
        col_stream.wait_stream(torch.cuda.current_stream())
        col_results = [None] * n_groups
        with torch.cuda.stream(col_stream):
            for i, g in enumerate(grad_inputs):
                col_results[i] = tkq.tk_quantize_col_only(g, A_sg_list[i])

    # 4. Batched GEMM + sum3 (main stream, runs concurrently with col-only)
    D_sum = _get_dgrad_sum_buf(M, K, A_fp4_list[0].device)
    if (
        hasattr(tk_mod, 'nvfp4_batched_accum_gemm')
        and (
            use_tk_localcta()
            or (not use_localcta_runtime and use_tk_ffn_plain_batched_accum_dgrad())
        )
    ):
        tk_mod.nvfp4_batched_accum_gemm(
            A_fp4_list, A_sc_list, A_sg_list,
            B_fp4_list, B_sc_list, B_sg_list,
            D_sum
        )
    else:
        D_list = _get_dgrad_bufs(n_groups, M, K, A_fp4_list[0].device)
        tk_dispatch_batched_gemm(
            tk_mod,
            A_fp4_list, A_sc_list, A_sg_list,
            B_fp4_list, B_sc_list, B_sg_list,
            D_list
        )
        if M <= 16384 and n_groups == 3 and hasattr(tk_mod, 'sum3_bf16'):
            tk_mod.sum3_bf16(D_list[0], D_list[1], D_list[2], D_sum)
        else:
            torch.add(D_list[0], D_list[1], out=D_sum)
            for i in range(2, n_groups):
                D_sum.add_(D_list[i])

    # 5. Build col quant output — sync side stream if needed
    if _split:
        torch.cuda.current_stream().wait_stream(col_stream)
        fp4_col_list = [r[0] for r in col_results]
        sc_col_list = [r[1] for r in col_results]
        sg_payload = [r[2] if len(r) > 2 else A_sg_list[i] for i, r in enumerate(col_results)]
    else:
        fp4_col_list = [r[2] for r in quant_results]
        sc_col_list = [r[3] for r in quant_results]
        sg_payload = [r[5] if use_tk_localcta() else r[4] for r in quant_results]

    if use_tk_localcta():
        fp4_col_full = torch.cat(
            [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in fp4_col_list], dim=0
        ).view(torch.float4_e2m1fn_x2)
        sc_col_cat = torch.cat(
            [sc.contiguous().view(torch.uint8) for sc in sc_col_list], dim=0
        ).view(torch.float8_e4m3fn)
        sg_col_cat = torch.cat(sg_payload, dim=0)
        dy_col_quant = (fp4_col_list, sc_col_list, sg_payload, fp4_col_full, sc_col_cat, sg_col_cat)
    else:
        sg_per_group = torch.stack([sg.view(-1)[0] for sg in A_sg_list]).squeeze()
        if sg_per_group.dim() == 0:
            sg_per_group = sg_per_group.unsqueeze(0)
        dy_col_quant = (fp4_col_list, sc_col_list, sg_per_group, None, None)

    return D_sum, dy_col_quant

# ---------------------------------------------------------------------------
# Fused QKV backward: dgrad + wgrad + rmsnorm_bwd in one tight call
# ---------------------------------------------------------------------------
_fused_bwd_cache = {}  # key: (M, K, N_total, N_dims, device) → cached state
_plain_qkv_split3_graph_cache = {}


def _qkv_fused_bwd_aux_buffer_policy(
    M: int,
    n_groups: int,
    *,
    use_localcta_runtime: bool,
    has_batched_accum: bool = False,
    has_bf16_transpose: bool = False,
    has_bf16_weight: bool = True,
) -> dict[str, bool]:
    """Return which large fallback-only QKV scratch buffers are required."""
    use_small_m_plain_qkv = _use_plain_tk_small_m_qkv_dgrad_eager(M, n_groups)
    use_bf16_wgrad = use_tk_qkv_bf16_wgrad() or use_small_m_plain_qkv
    use_bf16_dgrad = (
        (use_tk_qkv_bf16_dgrad() or use_small_m_plain_qkv)
        and has_bf16_weight
    )
    use_localcta_raw_wgrad = (
        use_localcta_runtime
        and get_tk_localcta_variant() == "v4"
        and use_tk_localcta_v4_sg_direct_consumers()
        and use_tk_localcta_v4_raw_backward_fallbacks(M)
    )
    use_localcta_split_wgrad = (
        use_localcta_runtime
        and get_tk_localcta_variant() == "v4"
        and use_tk_localcta_v4_fast_qkv_split_wgrad()
    )
    use_localcta_direct_wgrad = (
        use_localcta_runtime
        and get_tk_localcta_variant() == "v4"
        and not use_bf16_wgrad
        and not use_localcta_raw_wgrad
        and not use_localcta_split_wgrad
        and use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
    )
    use_plain_accum_dgrad = (
        not use_localcta_runtime
        and not use_bf16_dgrad
        and not use_small_m_plain_qkv
        and has_batched_accum
        and use_tk_qkv_plain_batched_accum_dgrad()
    )
    use_localcta_nofold = (
        use_localcta_runtime
        and use_tk_qkv_localcta_consistent_nofold_operands()
    )
    use_localcta_rescue = (
        use_localcta_runtime
        and use_tk_qkv_bf16_underflow_rescue()
    )
    use_debug_scratch = bool(
        _env_flag("USE_TK_DEBUG_QKV_DGRAD_REF", False)
        or _env_flag("USE_TK_DEBUG_QKV_WGRAD_REF", False)
        or use_tk_qkv_backward_capture_debug()
        or _debug_qkv_capture_path()
        or os.environ.get("USE_TK_QKV_DEBUG_DW_OUT_PATH", "").strip()
    )
    return {
        "D_list": (
            not (
                use_bf16_dgrad
                or (use_localcta_runtime and has_batched_accum)
                or use_plain_accum_dgrad
            )
            or use_debug_scratch
        ),
        "dW_T": not use_localcta_direct_wgrad or use_debug_scratch,
        "grad_w_materialized": (
            not use_localcta_runtime
            and has_bf16_transpose
            and use_tk_qkv_cached_return_transpose()
        ),
        "gw_list": (
            use_localcta_raw_wgrad
            or use_localcta_split_wgrad
            or use_debug_scratch
        ),
        "dy_cat": (
            use_small_m_plain_qkv
            or use_bf16_wgrad
            or use_bf16_dgrad
            or use_localcta_rescue
            or use_debug_scratch
        ),
        "normed": (
            use_small_m_plain_qkv
            or use_bf16_wgrad
            or use_localcta_raw_wgrad
            or use_localcta_nofold
            or use_localcta_rescue
            or use_debug_scratch
        ),
        "inv_rms_bf16": (
            use_small_m_plain_qkv
            or use_bf16_wgrad
            or use_localcta_raw_wgrad
            or use_localcta_nofold
            or use_localcta_rescue
            or use_debug_scratch
        ),
    }


def _plain_qkv_split3_sr_policy():
    explicit = os.environ.get("NVFP4_SR_GRAD")
    data_sr = (
        _env_flag("NVFP4_SR_GRAD", False)
        if explicit is not None
        else _env_flag("NVFP4_USE_STOCHASTIC_ROUNDING", False)
    )
    return (
        data_sr,
        _normalize_qkv_grad_sr_axes() if data_sr else "none",
        int(os.environ.get("NVFP4_RNG_SEED", "0")),
        int(os.environ.get("NVFP4_RNG_SUBSEQUENCE_BASE", "0")),
    )


def _normalize_qkv_grad_sr_axes():
    value = os.environ.get("NVFP4_GRAD_SR_AXES", "both")
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "all": "both",
        "row_col": "both",
        "rowcol": "both",
        "dgrad": "row",
        "wgrad": "col",
        "column": "col",
        "columns": "col",
        "off": "none",
        "0": "none",
    }
    value = aliases.get(value, value)
    if value not in {"none", "row", "col", "both"}:
        raise ValueError(
            f"Unsupported NVFP4_GRAD_SR_AXES={value!r}; expected none, row, col, or both"
        )
    return value


def _plain_qkv_split3_quantize_eager(tkq, g0, g1, g2):
    data_sr, data_sr_axes, rng_seed, rng_subsequence = _plain_qkv_split3_sr_policy()
    return tkq.tk_group_quantize_dim1_split3_for_gemm(
        g0,
        g1,
        g2,
        data_sr,
        rng_seed,
        rng_subsequence,
        data_sr_axes,
    )


def _plain_qkv_split3_graph_key(grad_splits, debug_name):
    first = grad_splits[0]
    return (
        _cache_owner_tag(debug_name),
        first.device.index,
        int(torch.cuda.current_stream(first.device).cuda_stream),
        tuple((tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
              for tensor in grad_splits),
        _plain_qkv_split3_sr_policy(),
    )


def _plain_qkv_split3_input_signature(grad_splits):
    return tuple(
        (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device.index,
        )
        for tensor in grad_splits
    )


def clear_tk_qkv_split3_graph_cache() -> None:
    """Maintenance-only release after all referencing CUDA graphs are dead."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("cannot clear QKV split3 graph state during capture")
    _plain_qkv_split3_graph_cache.clear()


def _plain_qkv_split3_quantize_graph_safe(
    tkq,
    grad_splits,
    debug_name=None,
):
    """Keep eager split3 unchanged; use exact-pointer state only in capture.

    A graph-enabled eager warmup first executes the ordinary monolithic native
    split3 ABI.  Its output storage is then retained while the capture allocator
    freezes both the exact split-input and scale descriptors.  Capture can only
    call the allocation-free launch ABI from the same owner and CUDA stream.
    """
    g0, g1, g2 = grad_splits
    graph_policy = _env_flag("USE_CUDA_GRAPH", False)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        if not graph_policy:
            # Preserve legacy behavior for unrelated captures.  The monolithic
            # ABI will reject unsupported capture rather than silently changing
            # the production route.
            return _plain_qkv_split3_quantize_eager(tkq, g0, g1, g2)
        launch = getattr(tkq, "tk_group_quantize_dim1_split3_launch", None)
        if launch is None:
            raise RuntimeError(
                "USE_CUDA_GRAPH=1 requires tk_group_quantize_dim1_split3_launch"
            )
        key = _plain_qkv_split3_graph_key(grad_splits, debug_name)
        cached = _plain_qkv_split3_graph_cache.get(key)
        if cached is None:
            raise RuntimeError(
                "regular QKV split3 graph state was not primed on this owner/stream; "
                "run an eager full-module warmup on the capture stream first"
            )
        current_signature = _plain_qkv_split3_input_signature(grad_splits)
        if cached["input_signature"] != current_signature:
            raise RuntimeError(
                "regular QKV split3 capture inputs do not match the exact pointers "
                "and strides frozen during warmup"
            )
        if cached.get("captured", False):
            raise RuntimeError(
                "regular QKV split3 state is already leased by a CUDA graph; "
                "use a distinct owner/stream or clear it after destroying the graph"
            )
        captured_result = launch(g0, g1, g2, cached["state"])
        cached["captured"] = True
        return captured_result

    result = _plain_qkv_split3_quantize_eager(tkq, g0, g1, g2)
    if not graph_policy:
        return result
    capture_alloc = getattr(
        tkq, "tk_group_quantize_dim1_split3_capture_alloc", None
    )
    launch = getattr(tkq, "tk_group_quantize_dim1_split3_launch", None)
    if capture_alloc is None or launch is None:
        raise RuntimeError(
            "USE_CUDA_GRAPH=1 requires split3 capture alloc/launch native ABIs"
        )
    if len(result) < 11:
        raise RuntimeError(
            "regular QKV split3 eager result must retain TMA device/host storage"
        )
    key = _plain_qkv_split3_graph_key(grad_splits, debug_name)
    input_signature = _plain_qkv_split3_input_signature(grad_splits)
    cached = _plain_qkv_split3_graph_cache.get(key)
    if cached is None or cached["input_signature"] != input_signature:
        data_sr, data_sr_axes, rng_seed, rng_subsequence = _plain_qkv_split3_sr_policy()
        state = capture_alloc(
            g0,
            g1,
            g2,
            result[5].view(torch.uint8),
            result[7].view(torch.uint8),
            result[2],
            [tensor.view(torch.uint8) for tensor in result[1]],
            [tensor.view(torch.uint8) for tensor in result[3]],
            [tensor.view(torch.uint8) for tensor in result[4]],
            result[6].view(torch.uint8),
            result[8].view(torch.uint8),
            result[9],
            result[10],
            data_sr,
            rng_seed,
            rng_subsequence,
            data_sr_axes,
        )
        state_tensor_manifest = tuple(tuple(row) for row in state.tensor_manifest)
        _plain_qkv_split3_graph_cache[key] = {
            "input_signature": input_signature,
            "state": state,
            "state_tensor_manifest": state_tensor_manifest,
            "state_stream": int(state.caller_stream),
            "captured": False,
        }
    return result


def tk_fused_qkv_backward(grad_splits, w_col, N_dims, x_nvfp4,
                           input_bf16, norm_weight_bf16, inv_rms,
                           w_bf16,
                           te_fused,
                           debug_name=None,
                           rope_live64_cs=None,
                           rope_seq_len: int = 0,
                           h_tile: bool = False):
    """Fused QKV backward: dgrad + wgrad + rmsnorm_bwd with minimal Python overhead.

    Caches all intermediate state (weight splits, sg indices, buffers,
    function pointers) so the hot path is just kernel launches with
    zero Python allocation or lookup overhead between them.

    Returns: (grad_input, grad_w_qkv, grad_norm_weight, rescue_info)
    """
    n_groups = len(N_dims)
    M = grad_splits[0].shape[0]
    use_localcta_runtime = _use_tk_localcta_for_m(M)
    tk_mod = _get_tk() if use_localcta_runtime else _get_tk_plain()
    tkq = _get_tk_quant_for_gemm() if use_localcta_runtime else _get_tk_quant_plain()
    w_fp4_c, w_sc_c, w_sg_c = w_col._tk_col
    K = w_fp4_c.shape[0]
    N_total = sum(N_dims)
    device = w_fp4_c.device
    qkv_sr_state = None
    if use_localcta_runtime:
        from .localcta_sr_state import active_localcta_sr_state, qkv_grad_key

        sr_manager = active_localcta_sr_state()
        if sr_manager is not None:
            qkv_sr_state = sr_manager.get(qkv_grad_key(debug_name), device)
    caller_stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    qkv_debug_timings = os.environ.get('USE_TK_QKV_DEBUG_TIMINGS', '0') == '1'
    qkv_timing_events = []

    def _qkv_timing_mark(label: str) -> None:
        if not qkv_debug_timings:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        qkv_timing_events.append((label, event))

    def _qkv_timing_emit(return_mode: str) -> None:
        if not qkv_debug_timings or not qkv_timing_events:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        events = qkv_timing_events + [('end', end_event)]
        parts = []
        total_ms = events[0][1].elapsed_time(events[-1][1])
        for (label, event), (next_label, next_event) in zip(events, events[1:]):
            parts.append(f"{label}->{next_label}={event.elapsed_time(next_event):.3f}ms")
        print(
            f"[TK QKV TIMINGS] mode={return_mode} total={total_ms:.3f}ms "
            + " ".join(parts),
            file=sys.stderr,
            flush=True,
        )

    # ── One-time setup: cache everything that doesn't change ──
    n_dims_tuple = tuple(int(n) for n in N_dims)
    cache_key = (
        "localcta" if use_localcta_runtime else "tk",
        M, K, N_total, n_dims_tuple, device.index if device.index is not None else 0,
        caller_stream_id,
        _cache_owner_tag(debug_name),
    )
    state = _fused_bwd_cache.get(cache_key)
    if state is None:
        aux_buffer_policy = _qkv_fused_bwd_aux_buffer_policy(
            M,
            n_groups,
            use_localcta_runtime=use_localcta_runtime,
            has_batched_accum=hasattr(tk_mod, 'nvfp4_batched_accum_gemm'),
            has_bf16_transpose=hasattr(tkq, 'bf16_transpose_into'),
            has_bf16_weight=w_bf16 is not None,
        )
        a_col_offsets = []
        a_col_widths = []
        fp4_off = 0
        for n_i in n_dims_tuple:
            fp4_cols = n_i // 2
            a_col_offsets.append(fp4_off)
            a_col_widths.append(fp4_cols)
            fp4_off += fp4_cols
        state = {
            'D_list': (
                [torch.empty(M, K, dtype=torch.bfloat16, device=device) for _ in range(n_groups)]
                if aux_buffer_policy['D_list']
                else None
            ),
            'D_sum': torch.empty(M, K, dtype=torch.bfloat16, device=device),
            'dW_T': (
                torch.empty(K, N_total, dtype=torch.bfloat16, device=device)
                if aux_buffer_policy['dW_T']
                else None
            ),
            'grad_w_materialized': (
                torch.empty(N_total, K, dtype=torch.bfloat16, device=device)
                if aux_buffer_policy['grad_w_materialized']
                else None
            ),
            'gw_list': (
                [torch.empty(n_i, K, dtype=torch.bfloat16, device=device) for n_i in n_dims_tuple]
                if aux_buffer_policy['gw_list']
                else None
            ),
            'dy_cat': (
                torch.empty(M, N_total, dtype=torch.bfloat16, device=device)
                if aux_buffer_policy['dy_cat']
                else None
            ),
            'normed': (
                torch.empty(M, K, dtype=torch.bfloat16, device=device)
                if aux_buffer_policy['normed']
                else None
            ),
            'inv_rms_bf16': (
                torch.empty(M, 1, dtype=torch.bfloat16, device=device)
                if aux_buffer_policy['inv_rms_bf16']
                else None
            ),
            'sg_idx': _get_sg_tile_indices(N_dims, device),
            'has_sum3': hasattr(tk_mod, 'sum3_bf16'),
            'has_strided': (
                hasattr(tk_mod, 'nvfp4_batched_gemm_strided')
                and not use_tk_qkv_disable_strided_dgrad()
            ),
            'a_col_offsets': a_col_offsets,
            'a_col_widths': a_col_widths,
            'ready_event': torch.cuda.Event(),
        }
        _fused_bwd_cache[cache_key] = state

    D_list = state['D_list']
    D_sum = state['D_sum']
    dW_T = state['dW_T']
    grad_w_materialized_buf = state['grad_w_materialized']
    gw_list = state['gw_list']
    dy_cat_buf = state['dy_cat']
    normed_buf = state['normed']
    inv_rms_bf16 = state['inv_rms_bf16']
    sg_idx = state['sg_idx']
    has_sum3 = state['has_sum3']
    has_strided = state['has_strided']
    a_col_offsets = state['a_col_offsets']
    a_col_widths = state['a_col_widths']
    ready_event = state['ready_event']

    # ── Weight splits (cached by data_ptr + layout) ──
    cached_col_splits = getattr(w_col, '_tk_col_splits', None)
    cached_col_sg_splits = getattr(w_col, '_tk_col_sg_splits', None)
    if cached_col_splits is not None:
        B_fp4_list, B_sc_list = cached_col_splits
        if cached_col_sg_splits is not None:
            B_sg_list = cached_col_sg_splits
        else:
            B_sg_list = _split_weight_col_sg_tensors(
                w_fp4_c, w_sg_c, N_dims, use_localcta_runtime=use_localcta_runtime
            )
    else:
        B_fp4_list, B_sc_list, B_sg_list = _split_weight_col_tensors(
            w_fp4_c, w_sc_c, w_sg_c, N_dims, use_localcta_runtime=use_localcta_runtime
        )
    x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
    w_col_for_dgrad = w_col

    use_consistent_nofold = (
        use_tk_qkv_localcta_consistent_nofold_operands()
        and hasattr(tkq, 'tk_quantize_for_gemm_prepared_nofold_maybe_borrow')
    )
    if (
        _is_localcta_quant_mod(tkq)
        and w_bf16 is not None
        and use_consistent_nofold
    ):
        if inv_rms.dim() == 1:
            inv_rms_bf16.copy_(inv_rms.view(M, 1))
        else:
            inv_rms_bf16.copy_(inv_rms.view(M, 1))
        normed_buf.copy_(input_bf16)
        normed_buf.mul_(inv_rms_bf16)
        normed_buf.mul_(norm_weight_bf16.view(1, K))
        x_quant = tkq.tk_quantize_for_gemm_prepared_nofold_maybe_borrow(
            normed_buf, normed_buf, True, True
        )
        x_fp4_c, x_sc_c = x_quant[2], x_quant[3]
        x_sg_c = (
            x_quant[5]
            if len(x_quant) > 5 and torch.is_tensor(x_quant[5]) and x_quant[5].numel() > 0
            else x_quant[4]
        )

        w_bf16_q = _as_bf16_contiguous(w_bf16)
        w_quant = tkq.tk_quantize_for_gemm_prepared_nofold_maybe_borrow(
            w_bf16_q, w_bf16_q, True, True
        )
        w_fp4_c_consistent, w_sc_c_consistent = w_quant[2], w_quant[3]
        w_sg_c_consistent = (
            w_quant[5]
            if len(w_quant) > 5 and torch.is_tensor(w_quant[5]) and w_quant[5].numel() > 0
            else w_quant[4]
        )

        class _LocalCTAColRef:
            __slots__ = ('_tk_col',)

            def __init__(self, tk_col):
                self._tk_col = tk_col

        w_col_for_dgrad = _LocalCTAColRef(
            (w_fp4_c_consistent, w_sc_c_consistent, w_sg_c_consistent)
        )
        B_fp4_list, B_sc_list, B_sg_list = _split_weight_col_tensors(
            w_fp4_c_consistent, w_sc_c_consistent, w_sg_c_consistent, N_dims
        )
        _trace_backend_choice('localcta_qkv_bwd', 'consistent_nofold_operands')

    if _use_plain_tk_small_m_qkv_dgrad_eager(M, n_groups) and w_bf16 is not None:
        B_fp4_list = []
        B_sc_list = []
        B_sg_list = []
        for w_i in torch.split(_as_bf16_contiguous(w_bf16), list(N_dims), dim=0):
            w_q = tkq.tk_quantize_for_gemm(_as_bf16_contiguous(w_i), True)
            B_fp4_list.append(w_q[2])
            B_sc_list.append(w_q[3])
            B_sg_list.append((w_q[5] if len(w_q) > 5 else w_q[4]).to(torch.float32))

    # ════════════════════════════════════════════════════════════════
    # HOT PATH: minimal Python between kernel launches
    # ════════════════════════════════════════════════════════════════

    # 1. Quantize grad splits.
    _qkv_timing_mark('start')
    _tk_stage_trace('qkv_bwd_sub', 'dy_quant_start', debug_name)
    raw_g0, raw_g1, raw_g2 = grad_splits
    can_use_scalar_expanded_package = (
        _use_tk_qkv_fast_expanded_sum_grad()
        and _is_localcta_quant_mod(tkq)
        and get_tk_localcta_variant() == 'v4'
        and rope_live64_cs is None
        and _is_zero_stride_scalar_bf16_grad(raw_g0)
        and _is_zero_stride_scalar_bf16_grad(raw_g1)
        and _is_zero_stride_scalar_bf16_grad(raw_g2)
        and qkv_sr_state is None
    )
    if can_use_scalar_expanded_package:
        g0, g1, g2 = raw_g0, raw_g1, raw_g2
    else:
        allow_strided_qkv_grad = (
            _use_localcta_v4_strided_qkv_grad_quant()
            or (
                not _is_localcta_quant_mod(tkq)
                and use_tk_qkv_strided_grad_quant()
                and use_tk_qkv_native_split3_quant()
                and hasattr(tkq, 'tk_group_quantize_dim1_split3_for_gemm')
            )
        )
        g0 = _as_bf16_quant_input(raw_g0, allow_strided=allow_strided_qkv_grad)
        g1 = _as_bf16_quant_input(raw_g1, allow_strided=allow_strided_qkv_grad)
        g2 = _as_bf16_quant_input(raw_g2, allow_strided=allow_strided_qkv_grad)
    use_small_m_plain_qkv = _use_plain_tk_small_m_qkv_dgrad_eager(M, n_groups)
    use_bf16_wgrad = use_tk_qkv_bf16_wgrad() or use_small_m_plain_qkv
    use_bf16_dgrad = (
        (use_tk_qkv_bf16_dgrad() and w_bf16 is not None)
        or (use_small_m_plain_qkv and w_bf16 is not None)
    )
    use_bf16_rmsnorm_bwd = use_tk_qkv_bf16_rmsnorm_bwd()
    if h_tile:
        use_bf16_rmsnorm_bwd = False
    rescue_info = None
    use_dgrad_nopdl = use_tk_qkv_dgrad_nopdl()
    use_wgrad_nopdl = use_tk_qkv_wgrad_nopdl()
    dgrad_package = None
    qkv_grad_boost = 1.0
    g0_q = g0
    g1_q = g1
    g2_q = g2
    if can_use_scalar_expanded_package and (use_bf16_wgrad or use_bf16_dgrad):
        can_use_scalar_expanded_package = False
        g0 = _as_bf16_contiguous(raw_g0)
        g1 = _as_bf16_contiguous(raw_g1)
        g2 = _as_bf16_contiguous(raw_g2)
        g0_q, g1_q, g2_q = g0, g1, g2
    if _is_localcta_quant_mod(tkq) and not (use_bf16_wgrad or use_bf16_dgrad):
        qkv_grad_boost = get_tk_qkv_localcta_fixed_grad_boost()
        if qkv_grad_boost != 1.0:
            can_use_scalar_expanded_package = False
            g0_q = (g0.float() * qkv_grad_boost).to(torch.bfloat16).contiguous()
            g1_q = (g1.float() * qkv_grad_boost).to(torch.bfloat16).contiguous()
            g2_q = (g2.float() * qkv_grad_boost).to(torch.bfloat16).contiguous()
            _trace_backend_choice('localcta_qkv_bwd', f'adaptive_grad_scale_{qkv_grad_boost:g}')
    if use_bf16_wgrad or use_bf16_dgrad:
        n0 = N_dims[0]
        n1 = N_dims[1]
        dy_cat_buf[:, :n0].copy_(g0)
        dy_cat_buf[:, n0:n0 + n1].copy_(g1)
        dy_cat_buf[:, n0 + n1:].copy_(g2)

    if can_use_scalar_expanded_package:
        scale_num = (
            tkq.tk_get_global_scale_num()
            if hasattr(tkq, 'tk_get_global_scale_num')
            else 1493.0
        )
        dgrad_package = _make_localcta_v4_split3_scalar_grad_package(
            (g0, g1, g2),
            N_dims,
            scale_num,
        )
        _trace_backend_choice('localcta_qkv_bwd', 'v4_scalar_expanded_sum_grad')
        fp4_row_list = dgrad_package['fp4_row_list']
        sc_row_list = dgrad_package['sc_row_list']
        A_sg_list = dgrad_package['a_sg_list']
        fp4_col_list = dgrad_package['fp4_col_list']
        sc_col_list = dgrad_package['sc_col_list']
        col_sg_list = dgrad_package['col_sg_list']
        a_fp4_full = dgrad_package['a_fp4_full']
        a_sc_cat = dgrad_package['a_sc_cat']
        fp4_col_full = dgrad_package['fp4_col_full']
        sc_col_cat = dgrad_package['sc_col_cat']
        col_sg_cat = dgrad_package['col_sg_cat']
        deferred_col_state = None
    elif _is_localcta_quant_mod(tkq) and hasattr(tkq, 'tk_group_quantize_dim1_split3_for_gemm'):
        if (
            w_col_for_dgrad is not w_col
            and use_consistent_nofold
            and qkv_sr_state is None
        ):
            dgrad_package = _localcta_single_quant_to_split3_package(
                tkq,
                g0_q,
                g1_q,
                g2_q,
                prepared_nofold=True,
            )
            dgrad_package['N_dims'] = list(N_dims)
            _trace_backend_choice('localcta_qkv_bwd', 'consistent_nofold_grad_operands')
        else:
            dgrad_package = _localcta_grouped_k_dgrad_package(
                (g0_q, g1_q, g2_q),
                N_dims,
                defer_col=_use_localcta_v3_defer_col_dgrad(),
                rope_live64_cs=rope_live64_cs,
                rope_seq_len=rope_seq_len,
                persistent_rng_state=qkv_sr_state,
            )
        fp4_row_list = dgrad_package['fp4_row_list']
        sc_row_list = dgrad_package['sc_row_list']
        A_sg_list = dgrad_package['a_sg_list']
        fp4_col_list = dgrad_package['fp4_col_list']
        sc_col_list = dgrad_package['sc_col_list']
        col_sg_list = dgrad_package['col_sg_list']
        a_fp4_full = dgrad_package['a_fp4_full']
        a_sc_cat = dgrad_package['a_sc_cat']
        fp4_col_full = dgrad_package['fp4_col_full']
        sc_col_cat = dgrad_package['sc_col_cat']
        col_sg_cat = dgrad_package['col_sg_cat']
        deferred_col_state = dgrad_package.get('deferred_col_state')
        if _debug_qkv_capture_path():
            _append_qkv_capture({
                "event": "qkv_package",
                "debug_name": debug_name,
                "N_dims": list(N_dims),
                "grad_inputs": {
                    "g0": _tensor_debug_stats(g0),
                    "g1": _tensor_debug_stats(g1),
                    "g2": _tensor_debug_stats(g2),
                },
                "package": {
                    "a_fp4_full": _tensor_debug_stats(a_fp4_full),
                    "a_sc_cat": _tensor_debug_stats(a_sc_cat),
                    "a_sg_list": [_tensor_debug_stats(x) for x in A_sg_list],
                    "fp4_col_full": _tensor_debug_stats(fp4_col_full),
                    "sc_col_cat": _tensor_debug_stats(sc_col_cat),
                    "col_sg_cat": _tensor_debug_stats(col_sg_cat),
                },
            })
    else:
        deferred_col_state = None
        plain_quant_keepalive = None
        if (
            use_tk_qkv_native_split3_quant()
            and hasattr(tkq, 'tk_group_quantize_dim1_split3_for_gemm')
        ):
            if _env_flag("USE_CUDA_GRAPH", False):
                result = _plain_qkv_split3_quantize_graph_safe(
                    tkq,
                    (g0_q, g1_q, g2_q),
                    debug_name=debug_name,
                )
            else:
                result = _plain_qkv_split3_quantize_eager(
                    tkq, g0_q, g1_q, g2_q
                )
            if len(result) > 9:
                plain_quant_keepalive = tuple(result[9:])
            _trace_backend_choice('regular_tk_qkv_quant', 'native_split3')
        else:
            result = tkq.tk_group_quantize_dim1_for_gemm(
                torch.cat([g0_q, g1_q, g2_q], dim=1),
                N_dims,
            )

    if _is_localcta_quant_mod(tkq):
        a_fp4_full = a_sc_cat = None
    elif not _is_localcta_quant_mod(tkq):
        if len(result) >= 9:
            fp4_row_list, sc_row_list, sg_per_group, \
                fp4_col_list, sc_col_list, \
                a_fp4_full, a_sc_cat, fp4_col_full, sc_col_cat = result[:9]
        else:
            fp4_row_list, sc_row_list, sg_per_group, fp4_col_list, sc_col_list = result
            a_fp4_full = a_sc_cat = fp4_col_full = sc_col_cat = None
        if plain_quant_keepalive is not None:
            state['plain_qkv_quant_keepalive'] = plain_quant_keepalive
            _record_tensors_on_stream(plain_quant_keepalive, torch.cuda.current_stream())
        A_sg_list = [sg_per_group[i:i+1].to(torch.float32) for i in range(n_groups)]
        col_sg_list = None
        col_sg_cat = None
        if _debug_qkv_capture_path():
            _append_qkv_capture({
                "event": "qkv_package_plain",
                "debug_name": debug_name,
                "N_dims": list(N_dims),
                "grad_inputs": {
                    "g0": _tensor_debug_stats(g0),
                    "g1": _tensor_debug_stats(g1),
                    "g2": _tensor_debug_stats(g2),
                },
                "package": {
                    "sc_row_list": [_tensor_debug_stats(x) for x in sc_row_list],
                    "a_sg_list": [_tensor_debug_stats(x) for x in A_sg_list],
                    "sc_col_list": [_tensor_debug_stats(x) for x in sc_col_list],
                    "fp4_col_full": _tensor_debug_stats(fp4_col_full),
                    "sc_col_cat": _tensor_debug_stats(sc_col_cat),
                    "B_sc_list": [_tensor_debug_stats(x) for x in B_sc_list],
                    "B_sg_list": [_tensor_debug_stats(x) for x in B_sg_list],
                },
            })
    _tk_stage_trace('qkv_bwd_sub', 'dy_quant_done', debug_name)
    _qkv_timing_mark('dy_quant_done')

    if (
        _is_localcta_quant_mod(tkq)
        and use_tk_qkv_localcta_scale_backoff()
        and _localcta_qkv_package_underflow(dgrad_package)
    ):
        retried_package, retried_info = _try_localcta_qkv_scale_backoff_package(
            tkq,
            (g0_q, g1_q, g2_q),
            N_dims,
            rope_live64_cs=rope_live64_cs,
            rope_seq_len=rope_seq_len,
            persistent_rng_state=qkv_sr_state,
        )
        if retried_package is not None:
            dgrad_package = retried_package
            rescue_info = retried_info
            _trace_backend_choice(
                'localcta_qkv_bwd',
                f"scale_backoff_{retried_info['scale_num']}",
            )

    if (
        _is_localcta_quant_mod(tkq)
        and use_tk_qkv_localcta_floor_prepared_scales()
        and _localcta_qkv_package_underflow(dgrad_package)
        and _localcta_qkv_package_has_live_sg(dgrad_package)
    ):
        rescue_info = _floor_zero_localcta_qkv_prepared_scales_(dgrad_package)
        if rescue_info.get('taken'):
            _trace_backend_choice('localcta_qkv_bwd', 'prepared_scale_floor')

    if _is_localcta_quant_mod(tkq):
        fp4_row_list = dgrad_package['fp4_row_list']
        sc_row_list = dgrad_package['sc_row_list']
        A_sg_list = dgrad_package['a_sg_list']
        fp4_col_list = dgrad_package['fp4_col_list']
        sc_col_list = dgrad_package['sc_col_list']
        col_sg_list = dgrad_package['col_sg_list']
        a_fp4_full = dgrad_package['a_fp4_full']
        a_sc_cat = dgrad_package['a_sc_cat']
        fp4_col_full = dgrad_package['fp4_col_full']
        sc_col_cat = dgrad_package['sc_col_cat']
        col_sg_cat = dgrad_package['col_sg_cat']

    if (
        _is_localcta_quant_mod(tkq)
        and w_bf16 is not None
        and use_tk_qkv_bf16_underflow_rescue()
        and _localcta_qkv_package_underflow(dgrad_package)
    ):
        underflow_details = _localcta_qkv_package_underflow_details(dgrad_package)
        _trace_backend_choice('localcta_qkv_bwd', 'bf16_underflow_rescue')
        n0 = N_dims[0]
        n1 = N_dims[1]
        dy_cat_buf[:, :n0].copy_(g0)
        dy_cat_buf[:, n0:n0 + n1].copy_(g1)
        dy_cat_buf[:, n0 + n1:].copy_(g2)
        if inv_rms.dim() == 1:
            inv_rms_bf16.copy_(inv_rms.view(M, 1))
        else:
            inv_rms_bf16.copy_(inv_rms.view(M, 1))
        normed_buf.copy_(input_bf16)
        normed_buf.mul_(inv_rms_bf16)
        normed_buf.mul_(norm_weight_bf16.view(1, K))
        torch.mm(normed_buf.transpose(0, 1), dy_cat_buf, out=dW_T)
        torch.mm(dy_cat_buf, _as_bf16_contiguous(w_bf16), out=D_sum)
        gi, dg = _rmsnorm_backward_bf16_reference(
            D_sum, input_bf16, norm_weight_bf16, inv_rms
        )
        rescue_info = {
            'taken': True,
            'reason': 'zero_qkv_scales',
            'path': 'bf16_underflow_rescue',
            'underflow_details': underflow_details,
        }
        return gi, dW_T.T, dg, rescue_info

    wgrad_input_dump_path = os.environ.get("USE_TK_QKV_DEBUG_WGRAD_INPUT_PATH", "").strip()
    if wgrad_input_dump_path and not getattr(tk_fused_qkv_backward, "_debug_wgrad_input_dumped", False):
        if inv_rms.dim() == 1:
            inv_rms_dbg = inv_rms.view(M, 1)
        else:
            inv_rms_dbg = inv_rms.view(M, 1)
        normed_dbg = input_bf16.detach().to(torch.bfloat16).contiguous()
        normed_dbg.mul_(inv_rms_dbg.detach().to(torch.bfloat16))
        normed_dbg.mul_(norm_weight_bf16.view(1, K))
        torch.save({
            "site": "tk_fused_qkv_backward_wgrad_inputs",
            "debug_name": debug_name,
            "N_dims": list(N_dims),
            "normed_buf": normed_dbg.cpu(),
            "g0": g0.detach().cpu(),
            "g1": g1.detach().cpu(),
            "g2": g2.detach().cpu(),
        }, wgrad_input_dump_path)
        tk_fused_qkv_backward._debug_wgrad_input_dumped = True

    col_stream = None
    col_results = None
    if (
        _is_localcta_quant_mod(tkq)
        and not use_bf16_wgrad
        and fp4_col_list is None
        and deferred_col_state is None
        and hasattr(tkq, 'tk_quantize_col_only_prepared')
    ):
        col_stream = _get_col_quant_stream()
        col_stream.wait_stream(torch.cuda.current_stream())
        col_results = [None] * n_groups
        with torch.cuda.stream(col_stream):
            _record_tensors_on_stream(((g0_q, g1_q, g2_q), A_sg_list), col_stream)
            for i, g in enumerate((g0_q, g1_q, g2_q)):
                col_results[i] = tkq.tk_quantize_col_only_prepared(g, A_sg_list[i])

    grad_w_qkv = None
    wgrad_stream = None
    plain_wgrad_stream = None
    plain_wgrad_launched = False
    plain_wgrad_waited = False

    def _plain_qkv_wgrad_operands():
        x_fp4_c_, x_sc_c_, x_sg_c_ = x_nvfp4._tk_col
        if fp4_col_full is not None and sc_col_cat is not None:
            dy_fp4_cat_ = fp4_col_full
            dy_sc_cat_ = sc_col_cat
        else:
            dy_fp4_cat_ = torch.cat(
                [fp4.view(torch.uint8) for fp4 in fp4_col_list], dim=0
            ).view(torch.float4_e2m1fn_x2)
            dy_sc_cat_ = torch.cat(
                [sc.view(torch.uint8) for sc in sc_col_list], dim=0
            ).view(torch.float8_e4m3fn)
        sg_stack = sg_per_group.to(torch.float32) if sg_per_group.dtype != torch.float32 else sg_per_group
        if sg_stack.dim() == 0:
            sg_stack = sg_stack.unsqueeze(0)
        b_sg_per_tile_ = sg_stack[sg_idx]
        x_sg_f32_ = x_sg_c_.to(torch.float32) if x_sg_c_.dtype != torch.float32 else x_sg_c_
        grouped_gemm_ = tk_mod.nvfp4_grouped_gemm
        if use_wgrad_nopdl:
            grouped_gemm_ = getattr(tk_mod, 'nvfp4_grouped_gemm_nopdl', grouped_gemm_)
            config_id = get_tk_qkv_wgrad_nopdl_config()
            grouped_gemm_config = getattr(tk_mod, 'nvfp4_grouped_gemm_config_nopdl', None)
            if config_id is not None and grouped_gemm_config is not None:
                def _configured_grouped_gemm(A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt=None, D_V_opt=None, silu_dim=0):
                    return grouped_gemm_config(
                        A, A_sc, A_sg, B, B_sc, B_sg, D,
                        D_K_opt, D_V_opt, silu_dim, config_id,
                    )
                grouped_gemm_ = _configured_grouped_gemm
        return grouped_gemm_, x_fp4_c_, x_sc_c_, x_sg_f32_, dy_fp4_cat_, dy_sc_cat_, b_sg_per_tile_

    def _wait_plain_qkv_wgrad_if_needed():
        nonlocal plain_wgrad_waited
        if not plain_wgrad_launched or plain_wgrad_waited:
            return
        torch.cuda.current_stream().wait_stream(plain_wgrad_stream)
        plain_wgrad_waited = True
        _tk_qkv_debug_sync_checkpoint('qkv_wgrad')
        _tk_stage_trace('qkv_bwd_sub', 'wgrad_done', debug_name)
        _qkv_timing_mark('wgrad_wait_done')

    plain_can_overlap_wgrad = (
        not use_bf16_wgrad
        and not _is_localcta_quant_mod(tkq)
        and not use_small_m_plain_qkv
        and use_wgrad_nopdl
        and use_tk_qkv_overlap_wgrad_nopdl()
        and not (
            get_tk_localcta_variant() == 'v4'
            and use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
        )
    )
    if plain_can_overlap_wgrad:
        _trace_backend_choice('regular_tk_qkv_wgrad', 'overlap_nopdl')
        _tk_stage_trace('qkv_bwd_sub', 'wgrad_start', debug_name)
        grouped_gemm, x_fp4_c, x_sc_c, x_sg_f32, dy_fp4_cat, dy_sc_cat, b_sg_per_tile = (
            _plain_qkv_wgrad_operands()
        )
        plain_wgrad_stream = _get_wgrad_stream()
        plain_wgrad_stream.wait_stream(torch.cuda.current_stream())
        if col_stream is not None:
            plain_wgrad_stream.wait_stream(col_stream)
        with torch.cuda.stream(plain_wgrad_stream):
            _record_tensors_on_stream(
                (
                    x_fp4_c, x_sc_c, x_sg_f32,
                    dy_fp4_cat, dy_sc_cat, b_sg_per_tile,
                    dW_T,
                ),
                plain_wgrad_stream,
            )
            grouped_gemm(
                x_fp4_c, x_sc_c, x_sg_f32,
                dy_fp4_cat, dy_sc_cat, b_sg_per_tile,
                dW_T,
            )
        plain_wgrad_launched = True
    if use_bf16_wgrad:
        _trace_backend_choice('localcta_qkv_wgrad', 'bf16_mm')
        _tk_stage_trace('qkv_bwd_sub', 'wgrad_start', debug_name)
        wgrad_stream = _get_wgrad_stream()
        wgrad_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(wgrad_stream):
            if inv_rms.dim() == 1:
                inv_rms_bf16.copy_(inv_rms.view(M, 1))
            else:
                inv_rms_bf16.copy_(inv_rms.view(M, 1))

            normed_buf.copy_(input_bf16)
            normed_buf.mul_(inv_rms_bf16)
            normed_buf.mul_(norm_weight_bf16.view(1, K))
            torch.mm(normed_buf.transpose(0, 1), dy_cat_buf, out=dW_T)
        _tk_stage_trace('qkv_bwd_sub', 'wgrad_done', debug_name)
    elif _is_localcta_quant_mod(tkq):
        _tk_stage_trace('qkv_bwd_sub', 'wgrad_start', debug_name)
        wgrad_stream = _get_wgrad_stream()
        wgrad_stream.wait_stream(torch.cuda.current_stream())
        use_v4_raw_wgrad = (
            get_tk_localcta_variant() == 'v4'
            and use_tk_localcta_v4_sg_direct_consumers()
            and use_tk_localcta_v4_raw_backward_fallbacks(M)
            and use_localcta_runtime
        )
        split_fast_wgrad_requested = (
            os.environ.get('USE_TK_LOCALCTA_V4_FAST_QKV_SPLIT_WGRAD', '0') == '1'
        )
        use_split_fast_wgrad = (
            get_tk_localcta_variant() == 'v4'
            and use_tk_localcta_v4_fast_qkv_split_wgrad()
        )
        direct_wgrad_requested = (
            get_tk_localcta_variant() == 'v4'
            and use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
        )
        if split_fast_wgrad_requested and direct_wgrad_requested:
            raise RuntimeError(
                "USE_TK_LOCALCTA_V4_FAST_QKV_SPLIT_WGRAD and "
                "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT are "
                "mutually exclusive"
            )
        use_localcta_direct_wgrad = (
            not use_v4_raw_wgrad
            and not use_split_fast_wgrad
            and direct_wgrad_requested
        )
        if use_localcta_direct_wgrad:
            # The side stream writes this caller-owned return before the
            # existing caller-stream wait below.
            grad_w_qkv = torch.empty(
                N_total, K, dtype=torch.bfloat16, device=x_fp4_c.device
            )
        with torch.cuda.stream(wgrad_stream):
            if use_v4_raw_wgrad:
                _record_tensors_on_stream(
                    (
                        g0, g1, g2,
                        input_bf16, norm_weight_bf16, inv_rms_bf16,
                        normed_buf, dW_T, gw_list,
                    ),
                    wgrad_stream,
                )
                if inv_rms.dim() == 1:
                    inv_rms_bf16.copy_(inv_rms.view(M, 1))
                else:
                    inv_rms_bf16.copy_(inv_rms.view(M, 1))
                normed_buf.copy_(input_bf16)
                normed_buf.mul_(inv_rms_bf16)
                normed_buf.mul_(norm_weight_bf16.view(1, K))
                _trace_backend_choice('localcta_qkv_wgrad', 'v4_raw_direct_wgrad')
                dump_path = os.environ.get("USE_TK_QKV_DEBUG_WGRAD_DUMP_PATH", "").strip()
                if dump_path and not getattr(tk_fused_qkv_backward, "_debug_wgrad_dumped", False):
                    torch.save({
                        "site": "tk_fused_qkv_backward_localcta_v4_raw_direct_wgrad",
                        "N_dims": list(N_dims),
                        "normed_buf": normed_buf.detach().cpu(),
                        "g0": g0.detach().cpu(),
                        "g1": g1.detach().cpu(),
                        "g2": g2.detach().cpu(),
                    }, dump_path)
                    tk_fused_qkv_backward._debug_wgrad_dumped = True
                tk_v4_direct_raw_wgrad_gemm(g0, normed_buf, gw_list[0], dy_encode_centric=True)
                tk_v4_direct_raw_wgrad_gemm(g1, normed_buf, gw_list[1], dy_encode_centric=True)
                tk_v4_direct_raw_wgrad_gemm(g2, normed_buf, gw_list[2], dy_encode_centric=True)
                col_off = 0
                for i, n_i in enumerate(N_dims):
                    dW_T[:, col_off:col_off + n_i].copy_(gw_list[i].transpose(0, 1))
                    col_off += n_i
            else:
                _trace_backend_choice('localcta_qkv_wgrad', 'grouped_gemm')
                x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
                if deferred_col_state is not None:
                    tkq.tk_group_quantize_dim1_split3_finalize_col_inplace(
                        deferred_col_state['col_sc_cat'],
                        deferred_col_state['col_sg_cat'],
                        deferred_col_state['col_sg_chunk_0'],
                        deferred_col_state['col_sg_chunk_1'],
                        deferred_col_state['col_sg_chunk_2'],
                    )
                if col_stream is not None:
                    wgrad_stream.wait_stream(col_stream)
                    fp4_col_list = [r[0] for r in col_results]
                    sc_col_list = [r[1] for r in col_results]
                    col_sg_list = [r[2] for r in col_results]
                    fp4_col_full = torch.cat(
                        [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in fp4_col_list], dim=0
                    ).view(torch.float4_e2m1fn_x2)
                    sc_col_cat = torch.cat(
                        [sc.contiguous().view(torch.uint8) for sc in sc_col_list], dim=0
                    ).view(torch.float8_e4m3fn)
                    col_sg_cat = torch.cat(col_sg_list, dim=0)

                dy_fp4_cat = fp4_col_full
                dy_sc_cat = sc_col_cat
                if dy_fp4_cat is None or dy_sc_cat is None:
                    dy_fp4_cat = torch.cat(
                        [fp4.view(torch.uint8) for fp4 in fp4_col_list], dim=0
                    ).view(torch.float4_e2m1fn_x2)
                    dy_sc_cat = torch.cat(
                        [sc.view(torch.uint8) for sc in sc_col_list], dim=0
                    ).view(torch.float8_e4m3fn)
                if col_sg_cat is None:
                    col_sg_cat = torch.cat(col_sg_list, dim=0)
                col_sg_cat = _normalize_localcta_v3_tilegrid_b_sg(dy_fp4_cat, col_sg_cat)
                col_sg_cat = col_sg_cat.contiguous()
                tk_wgrad = _get_tk_localcta_direct() if dgrad_package.get('prepared_nofold') else tk_mod
                if tk_wgrad is None:
                    tk_wgrad = tk_mod
                _record_tensors_on_stream(
                    (
                        x_fp4_c, x_sc_c, x_sg_c,
                        dy_fp4_cat, dy_sc_cat, col_sg_cat,
                        fp4_col_list, sc_col_list, col_sg_list,
                        dW_T, grad_w_qkv, gw_list,
                    ),
                    wgrad_stream,
                )

                if use_split_fast_wgrad:
                    col_off = 0
                    for i, n_i in enumerate(N_dims):
                        if hasattr(tk_mod, 'nvfp4_gemm_fast'):
                            tk_mod.nvfp4_gemm_fast(
                                fp4_col_list[i],
                                sc_col_list[i],
                                col_sg_list[i],
                                x_fp4_c,
                                x_sc_c,
                                x_sg_c,
                                gw_list[i],
                            )
                        else:
                            tk_dispatch_gemm(
                                tk_mod,
                                fp4_col_list[i],
                                sc_col_list[i],
                                col_sg_list[i],
                                x_fp4_c,
                                x_sc_c,
                                x_sg_c,
                                gw_list[i],
                            )
                        dW_T[:, col_off:col_off + n_i].copy_(gw_list[i].transpose(0, 1))
                        col_off += n_i
                else:
                    gemm_fn = tk_wgrad.nvfp4_grouped_gemm
                    if (
                        get_tk_localcta_variant() == 'v4'
                        and use_tk_localcta_v4_fast_qkv_grouped_wgrad()
                        and hasattr(tk_mod, 'nvfp4_grouped_gemm_fast')
                    ):
                        if use_localcta_direct_wgrad:
                            tk_mod.nvfp4_grouped_gemm_fast(
                                dy_fp4_cat, dy_sc_cat, col_sg_cat,
                                x_fp4_c, x_sc_c, x_sg_c,
                                grad_w_qkv
                            )
                        else:
                            tk_mod.nvfp4_grouped_gemm_fast(
                                x_fp4_c, x_sc_c, x_sg_c,
                                dy_fp4_cat, dy_sc_cat, col_sg_cat,
                                dW_T
                            )
                    else:
                        if use_localcta_direct_wgrad:
                            gemm_fn(
                                dy_fp4_cat, dy_sc_cat, col_sg_cat,
                                x_fp4_c, x_sc_c, x_sg_c,
                                grad_w_qkv
                            )
                        else:
                            gemm_fn(
                                x_fp4_c, x_sc_c, x_sg_c,
                                dy_fp4_cat, dy_sc_cat, col_sg_cat,
                                dW_T
                            )
            _tk_qkv_debug_sync_checkpoint('qkv_wgrad_localcta')
        _tk_stage_trace('qkv_bwd_sub', 'wgrad_done', debug_name)
    _qkv_timing_mark('wgrad_launch_done')

    # 2. Batched dgrad GEMM — 1 kernel launch
    _tk_stage_trace('qkv_bwd_sub', 'dgrad_start', debug_name)
    if use_bf16_dgrad:
        _trace_backend_choice('localcta_qkv_dgrad', 'bf16_mm')
        torch.mm(dy_cat_buf, _as_bf16_contiguous(w_bf16), out=D_sum)
    elif _is_localcta_quant_mod(tkq) and hasattr(tk_mod, 'nvfp4_batched_accum_gemm'):
        _localcta_grouped_k_dgrad_backend(
            dgrad_package,
            w_col_for_dgrad,
            N_dims,
            dx=D_sum,
            prefer_split3=_use_localcta_v3_runtime(),
            prefer_strided=dgrad_package.get('a_fp4_full') is not None,
            a_col_offsets=a_col_offsets,
            a_col_widths=a_col_widths,
            debug_name=debug_name,
        )
    elif _use_plain_tk_small_m_qkv_dgrad_eager(M, n_groups):
        target_rows = 256
        for i, g in enumerate((g0, g1, g2)):
            g_pad = _pad_rows_bf16(g, target_rows)
            g_quant_pad = tkq.tk_quantize_for_gemm(g_pad, True)
            A_fp4_pad = g_quant_pad[0].view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
            A_sc_pad = g_quant_pad[1].contiguous().view(torch.float8_e4m3fn)
            A_sg_pad = g_quant_pad[4].to(torch.float32)
            tmp_full = torch.empty(target_rows, K, dtype=torch.bfloat16, device=device)
            tk_dispatch_gemm(
                tk_mod,
                A_fp4_pad, A_sc_pad, A_sg_pad,
                B_fp4_list[i], B_sc_list[i], B_sg_list[i],
                tmp_full,
            )
            D_list[i].copy_(tmp_full[:M])
    elif (
        not _is_localcta_quant_mod(tkq)
        and use_tk_qkv_plain_batched_accum_dgrad()
        and hasattr(tk_mod, 'nvfp4_batched_accum_gemm')
    ):
        A_fp4_list = [fp4.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
                      for fp4 in fp4_row_list]
        A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
        tk_mod.nvfp4_batched_accum_gemm(
            A_fp4_list, A_sc_list, A_sg_list,
            B_fp4_list, B_sc_list, B_sg_list,
            D_sum,
        )
    elif a_fp4_full is not None and has_strided:
        a_fp4_u8 = a_fp4_full.view(torch.uint8)
        A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
        strided_gemm = tk_mod.nvfp4_batched_gemm_strided
        if use_dgrad_nopdl:
            strided_gemm = getattr(tk_mod, 'nvfp4_batched_gemm_strided_nopdl', strided_gemm)
        strided_gemm(
            a_fp4_u8.view(torch.float4_e2m1fn_x2),
            A_sc_list,
            A_sg_list,
            a_col_offsets,
            a_col_widths,
            B_fp4_list,
            B_sc_list,
            B_sg_list,
            D_list,
        )
    else:
        A_fp4_list = [fp4.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2)
                      for fp4 in fp4_row_list]
        A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
        tk_dispatch_batched_gemm(
            tk_mod,
            A_fp4_list, A_sc_list, A_sg_list,
            B_fp4_list, B_sc_list, B_sg_list,
            D_list
        )
    _tk_stage_trace('qkv_bwd_sub', 'dgrad_done', debug_name)
    _tk_qkv_debug_assert_finite(
        "dgrad_inputs",
        debug_name,
        (("g0", g0), ("g1", g1), ("g2", g2)),
    )
    _qkv_timing_mark('dgrad_done')

    # 3. Sum3 — 1 kernel launch (or already accumulated for localCTA)
    _tk_stage_trace('qkv_bwd_sub', 'sum_start', debug_name)
    qkv_dgrad_already_accumulated = (
        use_bf16_dgrad
        or (_is_localcta_quant_mod(tkq) and hasattr(tk_mod, 'nvfp4_batched_accum_gemm'))
        or (
            not _is_localcta_quant_mod(tkq)
            and not use_bf16_dgrad
            and not _use_plain_tk_small_m_qkv_dgrad_eager(M, n_groups)
            and use_tk_qkv_plain_batched_accum_dgrad()
            and hasattr(tk_mod, 'nvfp4_batched_accum_gemm')
        )
    )
    dgrad_debug_outputs = (
        (("D_sum", D_sum),)
        if qkv_dgrad_already_accumulated
        else tuple(
            (f"D_list[{index}]", value)
            for index, value in enumerate(D_list)
        )
    )
    _tk_qkv_debug_assert_finite(
        "dgrad_outputs", debug_name, dgrad_debug_outputs
    )
    qkv_dgrad_ref_filter = os.environ.get('USE_TK_DEBUG_QKV_DGRAD_REF_FILTER', '').strip()
    qkv_dgrad_ref_enabled = (
        os.environ.get('USE_TK_DEBUG_QKV_DGRAD_REF', '0') == '1'
        and w_bf16 is not None
        and (not qkv_dgrad_ref_filter or (debug_name is not None and qkv_dgrad_ref_filter in debug_name))
    )
    qkv_rms_ref_filter = os.environ.get('USE_TK_DEBUG_QKV_RMS_REF_FILTER', '').strip()
    qkv_rms_ref_enabled = (
        os.environ.get('USE_TK_DEBUG_QKV_RMS_REF', '0') == '1'
        and (not qkv_rms_ref_filter or (debug_name is not None and qkv_rms_ref_filter in debug_name))
    )
    plain_sum_rms_requested = use_tk_qkv_fused_sum_rms()
    native_sum3_rmsnorm_bwd_out = (
        _get_native_sum3_rmsnorm_bwd_out() if plain_sum_rms_requested else None
    )
    # The cached K=4096 kernel is the only production specialization that has
    # cleared the end-to-end gate; keep the generic ABI out of model dispatch.
    plain_sum_rms_eligible = (
        not _is_localcta_quant_mod(tkq)
        and not h_tile
        and not use_bf16_dgrad
        and not qkv_dgrad_already_accumulated
        and not use_bf16_rmsnorm_bwd
        and n_groups == 3
        and M == 32768
        and K == 4096
        and tuple(N_dims) == (4096, 1024, 1024)
        and not qkv_dgrad_ref_enabled
        and not qkv_rms_ref_enabled
        and not _debug_qkv_capture_path()
    )
    if (
        plain_sum_rms_requested
        and plain_sum_rms_eligible
        and native_sum3_rmsnorm_bwd_out is None
    ):
        raise RuntimeError(
            "USE_TK_QKV_FUSED_SUM_RMS=1 requires the native plain-TK "
            "sum3_rmsnorm_bwd_out symbol"
        )
    plain_can_fuse_sum_rms = (
        plain_sum_rms_requested
        and plain_sum_rms_eligible
        and native_sum3_rmsnorm_bwd_out is not None
    )
    if plain_can_fuse_sum_rms:
        _trace_backend_choice(
            'regular_tk_qkv_sum_rms', 'native_sum3_rmsnorm_bwd_out'
        )
    rms_d_normed = (D_list[0], D_list[1], D_list[2]) if plain_can_fuse_sum_rms else D_sum
    if not use_bf16_dgrad and not qkv_dgrad_already_accumulated and not plain_can_fuse_sum_rms:
        if has_sum3:
            tk_mod.sum3_bf16(D_list[0], D_list[1], D_list[2], D_sum)
        else:
            torch.add(D_list[0], D_list[1], out=D_sum)
            D_sum.add_(D_list[2])
    if (
        _is_localcta_quant_mod(tkq)
        and not use_bf16_dgrad
        and qkv_grad_boost != 1.0
    ):
        D_sum.div_(qkv_grad_boost)
    _tk_stage_trace('qkv_bwd_sub', 'sum_done', debug_name)
    _tk_qkv_debug_sync_checkpoint('qkv_dgrad')
    _qkv_timing_mark('sum_done')
    if qkv_dgrad_ref_enabled:
        n0 = N_dims[0]
        n1 = N_dims[1]
        dy_cat_buf[:, :n0].copy_(g0)
        dy_cat_buf[:, n0:n0 + n1].copy_(g1)
        dy_cat_buf[:, n0 + n1:].copy_(g2)
        D_ref = torch.matmul(dy_cat_buf, _as_bf16_contiguous(w_bf16))
        diff = D_sum.detach().to(torch.float32) - D_ref.detach().to(torch.float32)
        _append_qkv_capture({
            "event": "qkv_dgrad_ref",
            "debug_name": debug_name,
            "N_dims": list(N_dims),
            "outputs": {
                "actual": _tensor_debug_stats(D_sum),
                "ref": _tensor_debug_stats(D_ref),
                "diff": _tensor_debug_stats(diff),
            },
        })
    if _debug_qkv_capture_path():
        _wait_plain_qkv_wgrad_if_needed()
        _append_qkv_capture({
            "event": "qkv_outputs",
            "debug_name": debug_name,
            "N_dims": list(N_dims),
            "outputs": {
                "D_list": [_tensor_debug_stats(x) for x in D_list],
                "D_sum": _tensor_debug_stats(D_sum),
                "dW_T": _tensor_debug_stats(dW_T),
                "dy_fp4_cat": _tensor_debug_stats(fp4_col_full),
                "dy_sc_cat": _tensor_debug_stats(sc_col_cat),
                "dy_sg_cat": _tensor_debug_stats(col_sg_cat),
            },
        })

    # 4. RMSNorm backward and wgrad are independent once D_sum is ready.
    # Overlap them where it does not contend with the QKV dgrad critical path.
    gi = dg = None
    plain_rms_async = (
        not _is_localcta_quant_mod(tkq)
        and not h_tile
        and not use_bf16_rmsnorm_bwd
        and use_tk_qkv_overlap_rms_wgrad()
        and not torch.cuda.is_current_stream_capturing()
    )
    if h_tile:
        _tk_stage_trace('qkv_bwd_sub', 'h_tile_backward_start', debug_name)
        gi, dg = tk_h_tile_backward(
            D_sum, input_bf16, norm_weight_bf16, inv_rms
        )
        _tk_stage_trace('qkv_bwd_sub', 'h_tile_backward_done', debug_name)
    elif plain_can_fuse_sum_rms:
        _tk_stage_trace('qkv_bwd_sub', 'rmsnorm_start', debug_name)
        _tk_qkv_debug_assert_finite(
            "fused_sum_rms_inputs",
            debug_name,
            (
                ("input", input_bf16),
                ("norm_weight", norm_weight_bf16),
                ("inv_rms", inv_rms),
            ),
        )
        rms_state, rms_stream = _launch_native_sum3_rmsnorm_bwd_out_async(
            rms_d_normed,
            input_bf16,
            norm_weight_bf16,
            inv_rms,
            D_sum,
            native_sum3_rmsnorm_bwd_out,
            owner_key=debug_name,
            tag="qkv",
            force_current_stream=not plain_rms_async,
        )
    elif (_is_localcta_quant_mod(tkq) or plain_rms_async) and not use_bf16_rmsnorm_bwd:
        _tk_stage_trace('qkv_bwd_sub', 'rmsnorm_start', debug_name)
        ready_event.record(torch.cuda.current_stream())
        rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
            rms_d_normed, input_bf16, norm_weight_bf16, inv_rms, te_fused,
            ready_event=ready_event,
            owner_key=debug_name,
            tag="qkv",
        )
    _qkv_timing_mark('rms_launch_done')

    # 5. Wgrad: build col inputs and launch grouped GEMM — 1 kernel launch
    if use_bf16_wgrad:
        torch.cuda.current_stream().wait_stream(wgrad_stream)
    elif _is_localcta_quant_mod(tkq):
        torch.cuda.current_stream().wait_stream(wgrad_stream)
        if qkv_grad_boost != 1.0:
            if grad_w_qkv is not None:
                grad_w_qkv.div_(qkv_grad_boost)
            else:
                dW_T.div_(qkv_grad_boost)
        if use_tk_qkv_backward_capture_debug():
            _set_last_qkv_backward_debug_payload({
                'mode': 'localcta_qkv_wgrad',
                'x_fp4_c': x_fp4_c.detach(),
                'x_sc_c': x_sc_c.detach(),
                'x_sg_c': x_sg_c.detach() if torch.is_tensor(x_sg_c) else x_sg_c,
                'dy_fp4_cat': dy_fp4_cat.detach(),
                'dy_sc_cat': dy_sc_cat.detach(),
                'dy_sg_cat': col_sg_cat.detach() if torch.is_tensor(col_sg_cat) else col_sg_cat,
                'dW_T': dW_T.detach(),
                'shape': {'M': M, 'K': K, 'N_total': N_total},
                'n_dims': list(N_dims),
            })
    else:
        if not plain_wgrad_launched:
            _tk_stage_trace('qkv_bwd_sub', 'wgrad_start', debug_name)
            if _use_plain_tk_small_m_qkv_dgrad_eager(M, n_groups):
                if inv_rms.dim() == 1:
                    inv_rms_bf16.copy_(inv_rms.view(M, 1))
                else:
                    inv_rms_bf16.copy_(inv_rms.view(M, 1))
                normed_buf.copy_(input_bf16)
                normed_buf.mul_(inv_rms_bf16)
                normed_buf.mul_(norm_weight_bf16.view(1, K))

                target_rows = 256
                normed_pad = _pad_rows_bf16(normed_buf, target_rows)
                x_quant_pad = tkq.tk_quantize_for_gemm(normed_pad, True)
                x_fp4_c = x_quant_pad[2]
                x_sc_c = x_quant_pad[3]
                x_sg_c = x_quant_pad[4].to(torch.float32)

                col_off = 0
                for g, n_i in zip((g0, g1, g2), N_dims):
                    g_pad = _pad_rows_bf16(g, target_rows)
                    g_quant_pad = tkq.tk_quantize_for_gemm(g_pad, True)
                    g_fp4_c = g_quant_pad[2]
                    g_sc_c = g_quant_pad[3]
                    g_sg_c = g_quant_pad[4].to(torch.float32)
                    tmp = torch.empty(K, n_i, dtype=torch.bfloat16, device=x_fp4_c.device)
                    tk_dispatch_gemm(
                        tk_mod,
                        x_fp4_c, x_sc_c, x_sg_c,
                        g_fp4_c, g_sc_c, g_sg_c,
                        tmp,
                    )
                    dW_T[:, col_off:col_off + n_i].copy_(tmp)
                    col_off += n_i
            else:
                grouped_gemm, x_fp4_c, x_sc_c, x_sg_f32, dy_fp4_cat, dy_sc_cat, b_sg_per_tile = (
                    _plain_qkv_wgrad_operands()
                )
                if (
                    get_tk_localcta_variant() == 'v4'
                    and use_tk_localcta_v4_qkv_direct_grouped_wgrad_layout()
                ):
                    grad_w_qkv = torch.empty(sum(N_dims), K, dtype=torch.bfloat16, device=x_fp4_c.device)
                    grouped_gemm(
                        dy_fp4_cat, dy_sc_cat, b_sg_per_tile,
                        x_fp4_c, x_sc_c, x_sg_f32,
                        grad_w_qkv
                    )
                else:
                    grouped_gemm(
                        x_fp4_c, x_sc_c, x_sg_f32,
                        dy_fp4_cat, dy_sc_cat, b_sg_per_tile,
                        dW_T
                    )
            _tk_qkv_debug_sync_checkpoint('qkv_wgrad')
            _tk_stage_trace('qkv_bwd_sub', 'wgrad_done', debug_name)
    if not plain_wgrad_launched:
        _qkv_timing_mark('wgrad_wait_done')

    dw_dump_path = os.environ.get("USE_TK_QKV_DEBUG_DW_OUT_PATH", "").strip()
    if dw_dump_path and not getattr(tk_fused_qkv_backward, "_debug_dw_out_dumped", False):
        _wait_plain_qkv_wgrad_if_needed()
        torch.save({
            "site": "tk_fused_qkv_backward_dW_T",
            "debug_name": debug_name,
            "N_dims": list(N_dims),
            "dW_T": dW_T.detach().cpu(),
            "grad_w_qkv": None if grad_w_qkv is None else grad_w_qkv.detach().cpu(),
        }, dw_dump_path)
        tk_fused_qkv_backward._debug_dw_out_dumped = True

    qkv_wgrad_ref_filter = os.environ.get('USE_TK_DEBUG_QKV_WGRAD_REF_FILTER', '').strip()
    if (
        os.environ.get('USE_TK_DEBUG_QKV_WGRAD_REF', '0') == '1'
        and (not qkv_wgrad_ref_filter or (debug_name is not None and qkv_wgrad_ref_filter in debug_name))
    ):
        _wait_plain_qkv_wgrad_if_needed()
        n0 = N_dims[0]
        n1 = N_dims[1]
        dy_cat_buf[:, :n0].copy_(g0)
        dy_cat_buf[:, n0:n0 + n1].copy_(g1)
        dy_cat_buf[:, n0 + n1:].copy_(g2)
        if inv_rms.dim() == 1:
            inv_rms_ref = inv_rms.view(M, 1)
        else:
            inv_rms_ref = inv_rms.view(M, 1)
        normed_ref = (
            input_bf16.float()
            * inv_rms_ref.float()
            * norm_weight_bf16.float().view(1, K)
        ).to(torch.bfloat16)
        dW_ref_T = torch.matmul(
            normed_ref.transpose(0, 1),
            dy_cat_buf.to(torch.bfloat16),
        )
        actual_T = grad_w_qkv.transpose(0, 1) if grad_w_qkv is not None else dW_T
        diff = actual_T.detach().to(torch.float32) - dW_ref_T.detach().to(torch.float32)
        _append_qkv_capture({
            "event": "qkv_wgrad_ref",
            "debug_name": debug_name,
            "N_dims": list(N_dims),
            "outputs": {
                "actual_T": _tensor_debug_stats(actual_T),
                "ref_T": _tensor_debug_stats(dW_ref_T),
                "diff": _tensor_debug_stats(diff),
            },
        })

    # 6. RMSNorm backward — overlapped on side stream where enabled, serial otherwise.
    if h_tile:
        pass
    elif _is_localcta_quant_mod(tkq) or plain_rms_async or plain_can_fuse_sum_rms:
        if use_bf16_rmsnorm_bwd:
            gi, dg = _rmsnorm_backward_bf16_reference(
                D_sum, input_bf16, norm_weight_bf16, inv_rms
            )
        else:
            if plain_can_fuse_sum_rms:
                torch.cuda.current_stream().wait_event(rms_state['done_event'])
            else:
                torch.cuda.current_stream().wait_stream(rms_stream)
            gi = rms_state['grad_input']
            dg = rms_state.get('dgamma_out', rms_state['dgamma'])
    else:
        if use_bf16_rmsnorm_bwd:
            gi, dg = _rmsnorm_backward_bf16_reference(
                D_sum, input_bf16, norm_weight_bf16, inv_rms
            )
        else:
            gi, dg = te_fused.fused_rmsnorm_backward(
                D_sum, input_bf16, norm_weight_bf16, inv_rms)
    _tk_qkv_debug_assert_finite(
        "rms_outputs",
        debug_name,
        (("D_sum", D_sum), ("grad_input", gi), ("dgamma", dg)),
    )
    _tk_stage_trace('qkv_bwd_sub', 'rmsnorm_done', debug_name)
    _qkv_timing_mark('rms_done')
    _wait_plain_qkv_wgrad_if_needed()
    _tk_qkv_debug_assert_finite(
        "wgrad_output",
        debug_name,
        (("dW_T", dW_T), ("grad_w_qkv", grad_w_qkv)),
    )
    qkv_rms_ref_filter = os.environ.get('USE_TK_DEBUG_QKV_RMS_REF_FILTER', '').strip()
    if (
        os.environ.get('USE_TK_DEBUG_QKV_RMS_REF', '0') == '1'
        and (not qkv_rms_ref_filter or (debug_name is not None and qkv_rms_ref_filter in debug_name))
    ):
        gi_ref, dg_ref = _rmsnorm_backward_bf16_reference(
            D_sum, input_bf16, norm_weight_bf16, inv_rms
        )
        gi_diff = gi.detach().to(torch.float32) - gi_ref.detach().to(torch.float32)
        dg_diff = dg.detach().to(torch.float32) - dg_ref.detach().to(torch.float32)
        _append_qkv_capture({
            "event": "qkv_rmsnorm_ref",
            "debug_name": debug_name,
            "N_dims": list(N_dims),
            "outputs": {
                "grad_input": _tensor_debug_stats(gi),
                "grad_input_ref": _tensor_debug_stats(gi_ref),
                "grad_input_diff": _tensor_debug_stats(gi_diff),
                "grad_norm_weight": _tensor_debug_stats(dg),
                "grad_norm_weight_ref": _tensor_debug_stats(dg_ref),
                "grad_norm_weight_diff": _tensor_debug_stats(dg_diff),
            },
        })

    if use_tk_qkv_backward_capture_debug() and _is_localcta_quant_mod(tkq):
        x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
        w_fp4_c_dbg, w_sc_c_dbg, w_sg_c_dbg = w_col._tk_col
        payload = {
            'mode': 'localcta_qkv_backward',
            'dgrad_backend': 'bf16_mm' if use_bf16_dgrad else get_tk_qkv_localcta_dgrad_backend_override(),
            'use_bf16_dgrad': bool(use_bf16_dgrad),
            'use_bf16_wgrad': bool(use_bf16_wgrad),
            'use_bf16_rmsnorm_bwd': bool(use_bf16_rmsnorm_bwd),
            'qkv_grad_boost': float(qkv_grad_boost),
            'shape': {'M': M, 'K': K, 'N_total': N_total},
            'n_dims': list(N_dims),
            'x_fp4_c': x_fp4_c.detach(),
            'x_sc_c': x_sc_c.detach(),
            'x_sg_c': x_sg_c.detach() if torch.is_tensor(x_sg_c) else x_sg_c,
            'w_fp4_c': w_fp4_c_dbg.detach(),
            'w_sc_c': w_sc_c_dbg.detach(),
            'w_sg_c': w_sg_c_dbg.detach() if torch.is_tensor(w_sg_c_dbg) else w_sg_c_dbg,
            'dy_cat': dy_cat_buf.detach().clone(),
            'D_sum': D_sum.detach().clone(),
            'grad_input': gi.detach().clone(),
            'grad_norm_weight': dg.detach().clone(),
            'dW_T': dW_T.detach().clone(),
        }
        if 'dgrad_package' in locals():
            payload.update({
                'a_fp4_full': dgrad_package.get('a_fp4_full').detach() if torch.is_tensor(dgrad_package.get('a_fp4_full')) else None,
                'a_sc_cat': dgrad_package.get('a_sc_cat').detach() if torch.is_tensor(dgrad_package.get('a_sc_cat')) else None,
                'a_sg_full': dgrad_package.get('a_sg_full').detach() if torch.is_tensor(dgrad_package.get('a_sg_full')) else None,
                'fp4_row_list': [t.detach() for t in dgrad_package.get('fp4_row_list') or []],
                'sc_row_list': [t.detach() for t in dgrad_package.get('sc_row_list') or []],
                'row_sg_list': [
                    t.detach() if torch.is_tensor(t) else t
                    for t in (dgrad_package.get('row_sg_list') or [])
                ],
                'fp4_col_list': [t.detach() for t in dgrad_package.get('fp4_col_list') or []],
                'sc_col_list': [t.detach() for t in dgrad_package.get('sc_col_list') or []],
                'col_sg_list': [
                    t.detach() if torch.is_tensor(t) else t
                    for t in (dgrad_package.get('col_sg_list') or [])
                ],
                'fp4_col_full': dgrad_package.get('fp4_col_full').detach() if torch.is_tensor(dgrad_package.get('fp4_col_full')) else None,
                'sc_col_cat': dgrad_package.get('sc_col_cat').detach() if torch.is_tensor(dgrad_package.get('sc_col_cat')) else None,
                'col_sg_cat': dgrad_package.get('col_sg_cat').detach() if torch.is_tensor(dgrad_package.get('col_sg_cat')) else None,
            })
        if 'dy_fp4_cat' in locals():
            payload['dy_fp4_cat'] = dy_fp4_cat.detach()
        if 'dy_sc_cat' in locals():
            payload['dy_sc_cat'] = dy_sc_cat.detach()
        if 'col_sg_cat' in locals():
            payload['dy_sg_cat'] = col_sg_cat.detach() if torch.is_tensor(col_sg_cat) else col_sg_cat
        _set_last_qkv_backward_debug_payload(payload)

    if _debug_qkv_capture_path():
        _append_qkv_capture({
            "event": "qkv_rmsnorm",
            "debug_name": debug_name,
            "N_dims": list(N_dims),
            "outputs": {
                "grad_input": _tensor_debug_stats(gi),
                "grad_norm_weight": _tensor_debug_stats(dg),
            },
        })

    if grad_w_qkv is not None:
        _qkv_timing_mark('return_direct')
        _qkv_timing_emit('direct_grad_w_qkv')
        return gi, grad_w_qkv, dg, rescue_info

    # Materialize the transposed weight gradient before returning. dW_T is a
    # cached scratch buffer, so returning a view here lets later launches
    # overwrite the storage before autograd accumulates it.
    _qkv_timing_mark('return_copy_start')
    if (
        not _is_localcta_quant_mod(tkq)
        and debug_name is not None
        and use_tk_qkv_cached_return_transpose()
        and hasattr(tkq, 'bf16_transpose_into')
    ):
        tkq.bf16_transpose_into(dW_T, grad_w_materialized_buf)
        grad_w_materialized = grad_w_materialized_buf
    else:
        grad_w_materialized = dW_T.transpose(0, 1).contiguous()
    _qkv_timing_mark('return_copy_done')
    _qkv_timing_emit('materialized_grad_w')
    return gi, grad_w_materialized, dg, rescue_info

def tk_grouped_k_dgrad_gemm(dy_input, w_nvfp4, N_dims, debug_name=None):
    """Grouped dim-1 quant + z-dim batched GEMM for QKV backward dgrad.

    dx(M, K) = dy_q @ Wq^T + dy_k @ Wk^T + dy_v @ Wv^T

    Uses:
      1. tk_group_quantize_dim1_for_gemm or split-input equivalent:
         per-column-group quant → per-split row tensors
      2. nvfp4_batched_gemm (z-dim parallel): one kernel, separate per-batch outputs
      3. In-place sum of per-batch outputs

    Optimizations vs old path:
      - Uses per-split row tensors DIRECTLY from quantizer (no concatenate→re-slice)
      - Calls nvfp4_batched_gemm directly (no C++ nvfp4_split_dgrad_sum overhead)
      - Pre-allocated D buffers (cached by shape, no at::empty per call)
      - Uses pre-split weight col tensors from forward (no .contiguous() slicing)

    Returns:
        dx: (M, K) bf16 tensor
        dy_col_quant: tuple for wgrad reuse
    """
    M = dy_input[0].shape[0] if isinstance(dy_input, (tuple, list)) else dy_input.shape[0]
    use_localcta_runtime = use_tk_localcta() and _use_tk_localcta_for_m(M)
    tk = _get_tk() if use_localcta_runtime else _get_tk_plain()
    tkq = _get_tk_quant_for_gemm() if use_localcta_runtime else _get_tk_quant_plain()
    if use_localcta_runtime and _is_localcta_quant_mod(tkq) and len(N_dims) == 3:
        from .localcta_sr_state import active_localcta_sr_state, qkv_grad_key

        sr_manager = active_localcta_sr_state()
        persistent_rng_state = (
            None
            if sr_manager is None
            else sr_manager.get(qkv_grad_key(debug_name), w_nvfp4._tk_col[0].device)
        )
        dgrad_package = _localcta_grouped_k_dgrad_package(
            dy_input,
            N_dims,
            persistent_rng_state=persistent_rng_state,
        )
        if use_tk_localcta_direct_contract():
            dx = _localcta_direct_split_dgrad_backend(
                dgrad_package, w_nvfp4, N_dims,
            )
        else:
            dx = _localcta_grouped_k_dgrad_backend(
                dgrad_package, w_nvfp4, N_dims,
                prefer_strided=dgrad_package.get('a_fp4_full') is not None,
            )
        return dx, (
            dgrad_package['fp4_col_list'],
            dgrad_package['sc_col_list'],
            dgrad_package['col_sg_list'],
            dgrad_package['fp4_col_full'],
            dgrad_package['sc_col_cat'],
            dgrad_package['col_sg_cat'],
        )

    use_split_inputs = (
        isinstance(dy_input, (tuple, list))
        and len(dy_input) == len(N_dims)
        and hasattr(tkq, 'tk_group_quantize_dim1_split3_for_gemm')
    )
    if use_split_inputs:
        result = _plain_qkv_split3_quantize_eager(
            tkq, dy_input[0], dy_input[1], dy_input[2]
        )
        dy_device = dy_input[0].device
    else:
        dy_cat = dy_input if isinstance(dy_input, torch.Tensor) else torch.cat(list(dy_input), dim=1)
        result = tkq.tk_group_quantize_dim1_for_gemm(dy_cat, N_dims)
        dy_device = dy_cat.device

    w_fp4_c, w_sc_c, w_sg_c = w_nvfp4._tk_col
    K = w_fp4_c.shape[0]
    n_groups = len(N_dims)

    # 1. Grouped dim-1 quantize — returns per-split row tensors
    if not _is_localcta_quant_mod(tkq):
        if len(result) >= 9:
            fp4_row_list, sc_row_list, sg_per_group, \
                fp4_col_list, sc_col_list, \
                a_fp4_full, a_sc_cat, fp4_col_full, sc_col_cat = result[:9]
        else:
            fp4_row_list, sc_row_list, sg_per_group, fp4_col_list, sc_col_list = result
            a_fp4_full = a_sc_cat = fp4_col_full = sc_col_cat = None
        A_sg_list = [sg_per_group[i:i+1].to(torch.float32) for i in range(n_groups)]
        col_sg_list = None
        col_sg_cat = None

    # 2. Weight col splits — cached by data_ptr
    B_fp4_list, B_sc_list, B_sg_list = _split_weight_col_tensors(
        w_fp4_c, w_sc_c, w_sg_c, N_dims, use_localcta_runtime=use_localcta_runtime
    )

    # 3. GEMM
    D_list = _get_dgrad_bufs(n_groups, M, K, dy_device)
    use_strided = (a_fp4_full is not None
                   and hasattr(tk, 'nvfp4_batched_gemm_strided'))
    if _use_plain_tk_small_m_qkv_dgrad_eager(M, n_groups):
        A_fp4_list = [fp4.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2) for fp4 in fp4_row_list]
        A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
        for i in range(n_groups):
            tk_dispatch_gemm(
                tk,
                A_fp4_list[i], A_sc_list[i], A_sg_list[i],
                B_fp4_list[i], B_sc_list[i], B_sg_list[i],
                D_list[i],
            )
    elif use_strided:
        a_fp4_u8 = a_fp4_full.view(torch.uint8) if a_fp4_full.dtype != torch.uint8 else a_fp4_full
        A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
        col_offsets = []
        col_widths = []
        off = 0
        for n_i in N_dims:
            col_offsets.append(off)
            col_widths.append(n_i // 2)
            off += n_i // 2
        tk.nvfp4_batched_gemm_strided(
            a_fp4_u8.view(torch.float4_e2m1fn_x2),
            A_sc_list, A_sg_list,
            col_offsets, col_widths,
            B_fp4_list, B_sc_list, B_sg_list,
            D_list
        )
    else:
        A_fp4_list = [fp4.view(torch.uint8).contiguous().view(torch.float4_e2m1fn_x2) for fp4 in fp4_row_list]
        A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
        tk_dispatch_batched_gemm(
            tk,
            A_fp4_list, A_sc_list, A_sg_list,
            B_fp4_list, B_sc_list, B_sg_list,
            D_list
        )

    dx = _get_dgrad_sum_buf(M, K, dy_device)
    if n_groups == 3 and hasattr(tk, 'sum3_bf16'):
        tk.sum3_bf16(D_list[0], D_list[1], D_list[2], dx)
    elif n_groups == 3:
        torch.add(D_list[0], D_list[1], out=dx)
        dx.add_(D_list[2])
    elif n_groups == 2:
        torch.add(D_list[0], D_list[1], out=dx)
    else:
        dx.copy_(D_list[0])
        for i in range(1, n_groups):
            dx.add_(D_list[i])

    has_full_views = len(result) >= 9
    return dx, (fp4_col_list, sc_col_list, sg_per_group,
                fp4_col_full if has_full_views else None,
                sc_col_cat if has_full_views else None)

# ---------------------------------------------------------------------------
# Grouped wgrad GEMM: dW = dy^T @ x, using per-group col-quantized dy
# ---------------------------------------------------------------------------

# Module-level caches for wgrad
_wgrad_sg_idx_cache = {}  # key: (N_dims_tuple, device) -> index tensor
_wgrad_buf_cache = {}
_wgrad_direct_buf_cache = {}
_grouped_wgrad_cat_cache = {}
_split_wgrad_cache = {}

def _get_sg_tile_indices(N_dims, device):
    """Get or build cached tile indices for b_sg_per_tile gather."""
    key = (tuple(N_dims), device)
    idx = _wgrad_sg_idx_cache.get(key)
    if idx is None:
        Nb = 256
        indices = []
        for i, N_i in enumerate(N_dims):
            indices.extend([i] * (N_i // Nb))
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        _wgrad_sg_idx_cache[key] = idx
    return idx

def _get_wgrad_buf(K, N_total, device):
    """Get or allocate cached output buffer for wgrad."""
    key = (K, N_total, device)
    buf = _wgrad_buf_cache.get(key)
    if buf is None or buf.shape != (K, N_total):
        buf = torch.empty(K, N_total, dtype=torch.bfloat16, device=device)
        _wgrad_buf_cache[key] = buf
    return buf


def _get_wgrad_direct_buf(
    N_total,
    K,
    device,
    owner_key=None,
    caller_stream=None,
):
    """Get direct-layout output storage for grouped wgrad."""
    if (
        owner_key is not None
        and os.environ.get('USE_TK_TRANSIENT_DIRECT_WGRAD', '0') == '1'
        and not torch.cuda.is_current_stream_capturing()
    ):
        return torch.empty(
            N_total, K, dtype=torch.bfloat16, device=device
        )
    stream_key = (
        int(caller_stream.cuda_stream) if caller_stream is not None else None
    )
    key = (
        N_total,
        K,
        device,
        _cache_owner_tag(owner_key),
        stream_key,
    )
    buf = _wgrad_direct_buf_cache.get(key)
    if buf is None or buf.shape != (N_total, K):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "direct grouped-WGRAD output must be primed before graph capture"
            )
        buf = torch.empty(N_total, K, dtype=torch.bfloat16, device=device)
        _wgrad_direct_buf_cache[key] = buf
    return buf


def _split_wgrad_cache_key(M, K, H, device, owner_key, caller_stream):
    """Scope returned FFN WGRAD storage to its logical owner and stream."""
    return (
        M,
        K,
        H,
        device,
        _cache_owner_tag(owner_key),
        int(caller_stream.cuda_stream),
    )


def _get_split_wgrad_state(M, K, H, device, owner_key=None):
    """Get owner/stream-scoped cached state for split FFN WGRAD."""
    caller_stream = torch.cuda.current_stream(device)
    key = _split_wgrad_cache_key(M, K, H, device, owner_key, caller_stream)
    state = _split_wgrad_cache.get(key)
    if state is None:
        a_sg_bufs = [
            torch.empty(1, dtype=torch.float32, device=device),
            torch.empty(1, dtype=torch.float32, device=device),
        ]
        b_sg_bufs = [
            torch.empty(1, dtype=torch.float32, device=device),
            torch.empty(1, dtype=torch.float32, device=device),
        ]
        state = {
            'a_fp4_list': [None, None],
            'a_sc_list': [None, None],
            'a_sg_list': a_sg_bufs,
            'b_fp4_list': [None, None],
            'b_sc_list': [None, None],
            'b_sg_list': b_sg_bufs,
            'dW_T_list': [
                torch.empty(K, H, dtype=torch.bfloat16, device=device),
                torch.empty(K, H, dtype=torch.bfloat16, device=device),
            ],
            'dW_list': [
                torch.empty(H, K, dtype=torch.bfloat16, device=device),
                torch.empty(H, K, dtype=torch.bfloat16, device=device),
            ],
        }
        _split_wgrad_cache[key] = state
    return state


def _transpose_split_wgrad_outputs(dW_T_list, state, *, use_localcta: bool, owner_key=None):
    if (
        not use_localcta
        and owner_key is not None
        and use_tk_ffn_cached_return_transpose()
    ):
        tkq = _get_tk_quant_plain()
        if hasattr(tkq, 'bf16_transpose_into'):
            dW_list = state['dW_list']
            tkq.bf16_transpose_into(dW_T_list[0], dW_list[0])
            tkq.bf16_transpose_into(dW_T_list[1], dW_list[1])
            return dW_list[0], dW_list[1]
    return (
        dW_T_list[0].transpose(0, 1).contiguous(),
        dW_T_list[1].transpose(0, 1).contiguous(),
    )


def _grouped_wgrad_cat_key(fp4_col_list, sc_col_list, sg_col_list):
    fp4_sig = tuple(
        (t.data_ptr(), tuple(t.shape), str(t.dtype))
        for t in fp4_col_list
    )
    sc_sig = tuple(
        (t.data_ptr(), tuple(t.shape), str(t.dtype))
        for t in sc_col_list
    )
    sg_sig = _sg_cache_sig(sg_col_list)
    return fp4_sig, sc_sig, sg_sig


def _get_grouped_wgrad_col_payload(fp4_col_list, sc_col_list, sg_col_list):
    key = _grouped_wgrad_cat_key(fp4_col_list, sc_col_list, sg_col_list)
    cached = _grouped_wgrad_cat_cache.get(key)
    if cached is not None:
        return cached
    dy_fp4_cat = torch.cat(
        [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in fp4_col_list], dim=0
    ).view(torch.float4_e2m1fn_x2)
    dy_sc_cat = torch.cat(
        [sc.contiguous().view(torch.uint8) for sc in sc_col_list], dim=0
    ).view(torch.float8_e4m3fn)
    if torch.is_tensor(sg_col_list):
        sg_col_cat = _normalize_localcta_grouped_col_sg(sg_col_list)
    else:
        cat_dim = 1 if len(sg_col_list) > 0 and sg_col_list[0].dim() == 2 and sg_col_list[0].size(0) == 1 else 0
        sg_col_cat = torch.cat(sg_col_list, dim=cat_dim)
    cached = (dy_fp4_cat, dy_sc_cat, sg_col_cat)
    _grouped_wgrad_cat_cache[key] = cached
    return cached


def tk_grouped_wgrad_gemm(
    dy_col_quant,
    x_nvfp4,
    N_dims,
    nopdl=False,
    b_sg_buf=None,
    sg_f32_buf=None,
    owner_key=None,
    caller_stream=None,
):
    """Grouped wgrad GEMM: dW = dy^T @ x with per-group dy scaling.

    Uses nvfp4_grouped_gemm with swapped operands:
      A = x_col (K, M)           — shared, scalar a_sg = x_sg
      B = dy_col_cat (N_total, M) — per-group, b_sg_per_tile from sg_per_group
      D = A @ B^T = x^T @ dy = dW^T (K, N_total)

    Then transposes D to get dW (N_total, K).

    Args:
        dy_col_quant: tuple from dgrad — 5-tuple with pre-concatenated tensors
            or legacy 3-tuple (fp4_col_list, sc_col_list, sg_per_group)
        x_nvfp4: _TKQuantized input with _tk_col = (fp4, sc, sg)
        N_dims: list of int — [q_dim, k_dim, v_dim]
        nopdl: if True, use non-PDL kernel variant (safe for multi-stream/CUDA graphs)

    Returns:
        grad_w_qkv: (N_total, K) bf16 tensor
    """
    use_localcta_runtime = use_tk_localcta() and _use_tk_localcta_for_m(x_nvfp4.shape[0])
    tk = _get_tk() if use_localcta_runtime else _get_tk_plain()

    if use_localcta_runtime:
        if len(dy_col_quant) >= 6:
            fp4_col_list, sc_col_list, sg_col_list, fp4_col_full, sc_col_cat, sg_col_cat = dy_col_quant[:6]
        else:
            fp4_col_list, sc_col_list, sg_col_list = dy_col_quant
            fp4_col_full = sc_col_cat = sg_col_cat = None

        x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
        N_total = sum(N_dims)
        K = x_fp4_c.shape[0]
        M = int(x_nvfp4.shape[0])
        use_ffn_direct_layout = (
            use_tk_localcta_v4_ffn_direct_grouped_wgrad_layout()
            and isinstance(owner_key, tuple)
            and len(owner_key) == 3
            and owner_key[0] == 'localcta_ffn_direct_w13'
        )
        if use_ffn_direct_layout:
            dims = tuple(int(value) for value in N_dims)
            if (
                get_tk_localcta_variant() != 'v4'
                or (M, K, dims) != (32768, 4096, (14336, 14336))
            ):
                raise RuntimeError(
                    "localCTA FFN direct W13 return received a non-production contract"
                )
            if (
                not isinstance(owner_key[1], int)
                or owner_key[1] <= 0
                or not isinstance(owner_key[2], str)
                or not owner_key[2]
            ):
                raise RuntimeError(
                    "localCTA FFN direct W13 return requires a valid layer owner"
                )
            current_stream = torch.cuda.current_stream(x_fp4_c.device)
            if (
                caller_stream is None
                or int(current_stream.cuda_stream)
                != int(caller_stream.cuda_stream)
            ):
                raise RuntimeError(
                    "localCTA FFN direct W13 return must run on its caller stream"
                )
        use_direct_output_layout = (
            use_tk_localcta_v4_direct_grouped_wgrad_layout()
            or use_ffn_direct_layout
        )
        if fp4_col_full is not None and sc_col_cat is not None:
            dy_fp4_cat = fp4_col_full
            dy_sc_cat = sc_col_cat
        else:
            dy_fp4_cat, dy_sc_cat, sg_col_cat = _get_grouped_wgrad_col_payload(
                fp4_col_list, sc_col_list, sg_col_list
            )
        if sg_col_cat is None:
            _, _, sg_col_cat = _get_grouped_wgrad_col_payload(
                fp4_col_list, sc_col_list, sg_col_list
            )
        if use_tk_localcta_direct_contract():
            if torch.is_tensor(x_sg_c) and x_sg_c.dim() != 2:
                x_sg_c = _localcta_expand_sg_grid(x_sg_c, K, x_nvfp4.shape[0])
            if torch.is_tensor(sg_col_cat) and sg_col_cat.dim() != 2:
                sg_col_cat = _localcta_group_sg_grid_from_scalars(
                    sg_col_list if sg_col_list is not None else sg_col_cat,
                    N_dims,
                    x_nvfp4.shape[0],
                    x_fp4_c.device,
                )
        else:
            sg_col_cat = _normalize_localcta_grouped_col_sg(sg_col_cat)
        sg_col_cat = _normalize_localcta_v3_tilegrid_b_sg(dy_fp4_cat, sg_col_cat)
        dW_T = _get_wgrad_buf(K, N_total, x_fp4_c.device)
        use_v4_direct_grouped_wgrad = (
            get_tk_localcta_variant() == 'v4'
            and (
                use_tk_localcta_v4_sg_direct_consumers()
                or use_tk_localcta_v4_fast_grouped_wgrad()
                or use_ffn_direct_layout
            )
        )
        if use_ffn_direct_layout and not use_v4_direct_grouped_wgrad:
            raise RuntimeError(
                "localCTA FFN direct W13 return requires the direct grouped GEMM"
            )
        if (
            torch.is_tensor(x_sg_c)
            and x_sg_c.dim() <= 1
            and torch.is_tensor(sg_col_cat)
            and sg_col_cat.dim() <= 1
        ):
            sg_per_group = sg_col_cat.to(torch.float32).reshape(-1)
            sg_idx = _get_sg_tile_indices(N_dims, sg_per_group.device)
            b_sg_per_tile = sg_per_group[sg_idx]
            if use_v4_direct_grouped_wgrad:
                tk_direct = _get_tk_localcta_direct()
                if tk_direct is not None and hasattr(tk_direct, 'nvfp4_grouped_gemm'):
                    a_sg_per_tile = (
                        x_sg_c.reshape(-1)
                        .to(torch.float32)[:1]
                        .expand(K // 256)
                        .contiguous()
                    )
                    if use_direct_output_layout:
                        dW = _get_wgrad_direct_buf(
                            N_total, K, x_fp4_c.device,
                            owner_key=owner_key if use_ffn_direct_layout else None,
                            caller_stream=caller_stream if use_ffn_direct_layout else None,
                        )
                        tk_direct.nvfp4_grouped_gemm(
                            dy_fp4_cat, dy_sc_cat, b_sg_per_tile,
                            x_fp4_c, x_sc_c, a_sg_per_tile,
                            dW
                        )
                        return dW
                    tk_direct.nvfp4_grouped_gemm(
                        x_fp4_c, x_sc_c, a_sg_per_tile,
                        dy_fp4_cat, dy_sc_cat, b_sg_per_tile,
                        dW_T
                    )
                    return dW_T.transpose(0, 1)
            tk_plain = _get_tk_plain()
            tk_plain.nvfp4_grouped_gemm(
                x_fp4_c, x_sc_c, x_sg_c,
                dy_fp4_cat, dy_sc_cat, b_sg_per_tile,
                dW_T
            )
            return dW_T.transpose(0, 1)
        if (
            use_v4_direct_grouped_wgrad
            and torch.is_tensor(x_sg_c)
            and x_sg_c.dim() > 1
            and torch.is_tensor(sg_col_cat)
            and sg_col_cat.dim() <= 1
        ):
            tk_direct = _get_tk_localcta_direct()
            if tk_direct is None or not hasattr(tk_direct, 'nvfp4_grouped_gemm'):
                raise RuntimeError("localCTA direct grouped GEMM backend is unavailable for v4 grouped wgrad")
            sg_per_group = sg_col_cat.to(torch.float32).reshape(-1)
            sg_idx = _get_sg_tile_indices(N_dims, sg_per_group.device)
            b_sg_per_tile = sg_per_group[sg_idx]
            a_sg_outer = _prepare_localcta_v4_outer_sg_for_direct(
                x_sg_c, x_fp4_c.size(0) // 256, x_fp4_c.device
            )
            mode = tk_localcta_v4_grouped_wgrad_sg_mode()
            if mode == 'a1_b':
                a_sg_direct = torch.ones_like(a_sg_outer)
                b_sg_direct = b_sg_per_tile
            elif mode == 'a_b1':
                a_sg_direct = a_sg_outer
                b_sg_direct = torch.ones_like(b_sg_per_tile)
            elif mode == 'a_b':
                a_sg_direct = a_sg_outer
                b_sg_direct = b_sg_per_tile
            elif mode == 'a1_b1':
                a_sg_direct = torch.ones_like(a_sg_outer)
                b_sg_direct = torch.ones_like(b_sg_per_tile)
            else:
                raise ValueError(
                    "USE_TK_LOCALCTA_V4_GROUPED_WGRAD_SG_MODE must be one of "
                    "{'a1_b', 'a_b1', 'a_b', 'a1_b1'}"
                )
            if use_direct_output_layout:
                dW = _get_wgrad_direct_buf(
                    N_total, K, x_fp4_c.device,
                    owner_key=owner_key if use_ffn_direct_layout else None,
                    caller_stream=caller_stream if use_ffn_direct_layout else None,
                )
                tk_direct.nvfp4_grouped_gemm(
                    dy_fp4_cat, dy_sc_cat, b_sg_direct,
                    x_fp4_c, x_sc_c, a_sg_direct,
                    dW
                )
                return dW
            tk_direct.nvfp4_grouped_gemm(
                x_fp4_c, x_sc_c, a_sg_direct,
                dy_fp4_cat, dy_sc_cat, b_sg_direct,
                dW_T
            )
            return dW_T.transpose(0, 1)
        use_v4_direct_grouped_wgrad = (
            use_v4_direct_grouped_wgrad
            and torch.is_tensor(x_sg_c)
            and x_sg_c.dim() > 1
            and torch.is_tensor(sg_col_cat)
            and sg_col_cat.dim() > 1
        )
        if use_v4_direct_grouped_wgrad:
            tk_direct = _get_tk_localcta_direct()
            if tk_direct is None or not hasattr(tk_direct, 'nvfp4_grouped_gemm'):
                raise RuntimeError("localCTA direct grouped GEMM backend is unavailable for v4 grouped wgrad")
            a_sg_outer = _prepare_localcta_v4_outer_sg_for_direct(
                x_sg_c, x_fp4_c.size(0) // 256, x_fp4_c.device
            )
            b_sg_outer = _prepare_localcta_v4_outer_sg_for_direct(
                sg_col_cat, dy_fp4_cat.size(0) // 256, dy_fp4_cat.device
            )
            mode = tk_localcta_v4_grouped_wgrad_sg_mode()
            if mode == 'a1_b':
                a_sg_direct = torch.ones_like(a_sg_outer)
                b_sg_direct = b_sg_outer
            elif mode == 'a_b1':
                a_sg_direct = a_sg_outer
                b_sg_direct = torch.ones_like(b_sg_outer)
            elif mode == 'a_b':
                a_sg_direct = a_sg_outer
                b_sg_direct = b_sg_outer
            elif mode == 'a1_b1':
                a_sg_direct = torch.ones_like(a_sg_outer)
                b_sg_direct = torch.ones_like(b_sg_outer)
            else:
                raise ValueError(
                    "USE_TK_LOCALCTA_V4_GROUPED_WGRAD_SG_MODE must be one of "
                    "{'a1_b', 'a_b1', 'a_b', 'a1_b1'}"
                )
            if use_direct_output_layout:
                dW = _get_wgrad_direct_buf(
                    N_total, K, x_fp4_c.device,
                    owner_key=owner_key if use_ffn_direct_layout else None,
                    caller_stream=caller_stream if use_ffn_direct_layout else None,
                )
                tk_direct.nvfp4_grouped_gemm(
                    dy_fp4_cat, dy_sc_cat, b_sg_direct,
                    x_fp4_c, x_sc_c, a_sg_direct,
                    dW
                )
                return dW
            tk_direct.nvfp4_grouped_gemm(
                x_fp4_c, x_sc_c, a_sg_direct,
                dy_fp4_cat, dy_sc_cat, b_sg_direct,
                dW_T
            )
        else:
            tk.nvfp4_grouped_gemm(
                x_fp4_c, x_sc_c, x_sg_c,
                dy_fp4_cat, dy_sc_cat, sg_col_cat,
                dW_T
            )
        return dW_T.transpose(0, 1)

    # Unpack — support both legacy 3-tuple and optimized 5-tuple
    if len(dy_col_quant) == 5:
        fp4_col_list, sc_col_list, sg_per_group, fp4_col_full, sc_col_cat = dy_col_quant
    else:
        fp4_col_list, sc_col_list, sg_per_group = dy_col_quant
        fp4_col_full = sc_col_cat = None

    x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
    N_total = sum(N_dims)

    K = x_fp4_c.shape[0]
    M = x_fp4_c.shape[1] * 2

    # 1. Col FP4/scales — use pre-concatenated if available (avoids 0.040ms cat)
    if _use_plain_tk_small_m_qkv_dgrad_eager(M, len(N_dims)):
        dW_T = _get_wgrad_buf(K, N_total, x_fp4_c.device)
        col_off = 0
        for i, n_i in enumerate(N_dims):
            tmp = torch.empty(K, n_i, dtype=torch.bfloat16, device=x_fp4_c.device)
            tk_dispatch_gemm(
                tk,
                x_fp4_c, x_sc_c, x_sg_c,
                fp4_col_list[i], sc_col_list[i],
                sg_per_group[i:i + 1].to(torch.float32),
                tmp,
            )
            dW_T[:, col_off:col_off + n_i].copy_(tmp)
            col_off += n_i
        return dW_T.transpose(0, 1).contiguous()
    if fp4_col_full is not None and sc_col_cat is not None:
        dy_fp4_cat = fp4_col_full
        dy_sc_cat = sc_col_cat
    else:
        dy_fp4_cat = torch.cat(
            [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in fp4_col_list], dim=0
        ).view(torch.float4_e2m1fn_x2)
        dy_sc_cat = torch.cat(
            [sc.contiguous().view(torch.uint8) for sc in sc_col_list], dim=0
        ).view(torch.float8_e4m3fn)

    sg_idx = _get_sg_tile_indices(N_dims, sg_per_group.device)
    if b_sg_buf is not None:
        # Graph-safe: avoid graph-pool allocs from .to() and [sg_idx]
        # Use sg_f32_buf if provided for the intermediate .to(float32) conversion
        if sg_f32_buf is not None:
            sg_f32_buf.copy_(sg_per_group)
            torch.index_select(sg_f32_buf, 0, sg_idx, out=b_sg_buf)
        else:
            b_sg_buf.copy_(sg_per_group.to(torch.float32)[sg_idx])
        b_sg_per_tile = b_sg_buf
    else:
        b_sg_per_tile = sg_per_group.to(torch.float32)[sg_idx]

    # 3. Launch grouped GEMM: D(K, N_total) = x_col @ dy_col_cat^T = dW^T
    dW_T = _get_wgrad_buf(K, N_total, x_fp4_c.device)
    gemm_fn = tk.nvfp4_grouped_gemm_nopdl if nopdl else tk.nvfp4_grouped_gemm
    gemm_fn(
        x_fp4_c, x_sc_c, x_sg_c,         # A = x_col (K, M), scalar a_sg
        dy_fp4_cat, dy_sc_cat,             # B = dy_col_cat (N_total, M)
        b_sg_per_tile,                     # per-tile b_sg
        dW_T                               # D = dW^T (K, N_total)
    )

    # Materialize the transposed output. Returning a view into the cached
    # scratch buffer is unsafe because later layers can overwrite the buffer
    # before autograd finishes accumulating parameter grads.
    return dW_T.transpose(0, 1).contiguous()


def tk_split_wgrad_gemm(dy_col_quant, x_nvfp4, use_localcta=None, owner_key=None):
    """2-way eager FFN wgrad using batched GEMM on split col-quantized grads.

    Computes:
      dW1^T = x_col @ dh1_col^T
      dW3^T = x_col @ dh3_col^T

    Returns transposed views `(dW1, dW3)` with shapes `(H, K)`.
    """
    if use_localcta is None:
        use_localcta = _use_tk_localcta_for_m(x_nvfp4.shape[0])
    tk = _get_tk() if use_localcta else _get_tk_plain()

    fp4_col_list, sc_col_list, sg_list = dy_col_quant
    if len(fp4_col_list) != 2 or len(sc_col_list) != 2 or len(sg_list) != 2:
        raise ValueError("tk_split_wgrad_gemm expects exactly 2 split gradients")

    if _use_localcta_v3_runtime():
        H = fp4_col_list[0].shape[0]
        grad_w13 = tk_grouped_wgrad_gemm(
            (fp4_col_list, sc_col_list, sg_list),
            x_nvfp4,
            [H, H],
        )
        return grad_w13.split(H, dim=0)

    x_fp4_c, x_sc_c, x_sg_c = x_nvfp4._tk_col
    M, K = x_nvfp4.shape
    H = fp4_col_list[0].shape[0]
    state = _get_split_wgrad_state(M, K, H, x_fp4_c.device, owner_key=owner_key)
    dW_T_list = state['dW_T_list']
    dW_list = state.get('dW_list')

    if use_localcta and use_tk_localcta_v4_strict_path():
        if os.environ.get('USE_TK_LOCALCTA_V4_STRICT_GROUPED_WGRAD', '0') == '1':
            dy_fp4_cat, dy_sc_cat, dy_sg_cat = _get_grouped_wgrad_col_payload(
                fp4_col_list, sc_col_list, sg_list
            )
            if not (
                torch.is_tensor(dy_sg_cat)
                and dy_sg_cat.dim() == 2
                and torch.is_tensor(x_sg_c)
                and x_sg_c.dim() == 2
            ):
                raise RuntimeError("strict grouped v4 wgrad requires raw 2D chunk-grid SG tensors")
            x_sg_for_wgrad = _prepare_localcta_v4_chunkgrid_for_batched(
                x_sg_c,
                x_fp4_c.size(0),
                x_fp4_c.size(1) * 2,
                x_fp4_c.device,
            )
            dW_T = _get_wgrad_buf(K, 2 * H, x_fp4_c.device)
            tk.nvfp4_grouped_gemm(
                x_fp4_c,
                x_sc_c,
                x_sg_for_wgrad,
                dy_fp4_cat,
                dy_sc_cat,
                dy_sg_cat,
                dW_T,
            )
            grad_w13 = dW_T.transpose(0, 1).contiguous()
            return grad_w13.split(H, dim=0)
        # Strict v4 uses the outer-SG epilogue contract by default: two C++
        # GEMM dispatches with one row/output-tile SG and one col/output-tile
        # SG. Raw chunk-grid expansion is only for explicit SG-direct probes.
        use_chunkgrid_b_sg = (
            torch.is_tensor(sg_list[0])
            and sg_list[0].dim() == 2
            and _is_localcta_v4_chunkgrid_sg_tensor(sg_list[0], fp4_col_list[0])
        )
        x_sg_for_wgrad = (
            _prepare_localcta_v4_chunkgrid_for_batched(
                x_sg_c,
                x_fp4_c.size(0),
                x_fp4_c.size(1) * 2,
                x_fp4_c.device,
            )
            if use_chunkgrid_b_sg else x_sg_c
        )
        tk_dispatch_gemm(
            tk,
            x_fp4_c,
            x_sc_c,
            x_sg_for_wgrad,
            fp4_col_list[0],
            sc_col_list[0],
            sg_list[0] if sg_list[0].dim() > 0 else sg_list[0].view(1),
            dW_T_list[0],
        )
        tk_dispatch_gemm(
            tk,
            x_fp4_c,
            x_sc_c,
            x_sg_for_wgrad,
            fp4_col_list[1],
            sc_col_list[1],
            sg_list[1] if sg_list[1].dim() > 0 else sg_list[1].view(1),
            dW_T_list[1],
        )
        return _transpose_split_wgrad_outputs(
            dW_T_list, state, use_localcta=use_localcta, owner_key=owner_key
        )

    a_fp4_list = state['a_fp4_list']
    a_sc_list = state['a_sc_list']
    a_sg_list = state['a_sg_list']
    b_fp4_list = state['b_fp4_list']
    b_sc_list = state['b_sc_list']
    b_sg_list = state['b_sg_list']

    if (
        not use_localcta
        and use_tk_ffn_direct_split_wgrad_layout()
        and dW_list is not None
        and torch.is_tensor(x_sg_c)
        and x_sg_c.numel() == 1
        and all(torch.is_tensor(sg) and sg.numel() == 1 for sg in sg_list)
    ):
        a_fp4_list[0] = fp4_col_list[0]
        a_fp4_list[1] = fp4_col_list[1]
        a_sc_list[0] = sc_col_list[0]
        a_sc_list[1] = sc_col_list[1]
        a_sg_list[0] = sg_list[0] if sg_list[0].dim() > 0 else sg_list[0].view(1)
        a_sg_list[1] = sg_list[1] if sg_list[1].dim() > 0 else sg_list[1].view(1)

        b_fp4_list[0] = x_fp4_c
        b_fp4_list[1] = x_fp4_c
        b_sc_list[0] = x_sc_c
        b_sc_list[1] = x_sc_c
        x_sg_f32 = x_sg_c if x_sg_c.dtype == torch.float32 else x_sg_c.to(torch.float32)
        b_sg_list[0] = x_sg_f32
        b_sg_list[1] = x_sg_f32

        tk_dispatch_batched_gemm(
            tk,
            a_fp4_list, a_sc_list, a_sg_list,
            b_fp4_list, b_sc_list, b_sg_list,
            dW_list,
        )
        if owner_key is not None and use_tk_ffn_cached_return_transpose():
            return dW_list[0], dW_list[1]
        # Without a layer owner key this is a shape-global scratch buffer.
        return (
            dW_list[0].clone(memory_format=torch.contiguous_format),
            dW_list[1].clone(memory_format=torch.contiguous_format),
        )

    if use_localcta or use_tk_ffn_split_wgrad_eager():
        if (
            get_tk_localcta_variant() == 'v4'
            and use_tk_localcta_v4_sg_direct_consumers()
            and torch.is_tensor(x_sg_c)
            and x_sg_c.dim() > 1
            and torch.is_tensor(sg_list[0])
            and torch.is_tensor(sg_list[1])
            and sg_list[0].dim() > 1
            and sg_list[1].dim() > 1
        ):
            tk_v4_direct_split_wgrad_col_gemm(
                fp4_col_list[0],
                sc_col_list[0],
                sg_list[0] if sg_list[0].dim() > 0 else sg_list[0].view(1),
                x_fp4_c,
                x_sc_c,
                x_sg_c,
                dW_T_list[0],
            )
            tk_v4_direct_split_wgrad_col_gemm(
                fp4_col_list[1],
                sc_col_list[1],
                sg_list[1] if sg_list[1].dim() > 0 else sg_list[1].view(1),
                x_fp4_c,
                x_sc_c,
                x_sg_c,
                dW_T_list[1],
            )
            return _transpose_split_wgrad_outputs(
                dW_T_list, state, use_localcta=use_localcta, owner_key=owner_key
            )

        # localCTA split FFN wgrad needs the same per-tile SG contract as the
        # single-GEMM localCTA paths above; the batched kernel can emit NaNs or
        # long-run stalls here even when both split producers are finite.
        tk_dispatch_gemm(
            tk,
            x_fp4_c,
            x_sc_c,
            x_sg_c,
            fp4_col_list[0],
            sc_col_list[0],
            sg_list[0] if sg_list[0].dim() > 0 else sg_list[0].view(1),
            dW_T_list[0],
        )
        tk_dispatch_gemm(
            tk,
            x_fp4_c,
            x_sc_c,
            x_sg_c,
            fp4_col_list[1],
            sc_col_list[1],
            sg_list[1] if sg_list[1].dim() > 0 else sg_list[1].view(1),
            dW_T_list[1],
        )
        return _transpose_split_wgrad_outputs(
            dW_T_list, state, use_localcta=use_localcta, owner_key=owner_key
        )

    a_fp4_list[0] = x_fp4_c
    a_fp4_list[1] = x_fp4_c
    a_sc_list[0] = x_sc_c
    a_sc_list[1] = x_sc_c
    if x_sg_c.dtype == torch.float32:
        a_sg_list[0].copy_(x_sg_c)
        a_sg_list[1].copy_(x_sg_c)
    else:
        a_sg_list[0].copy_(x_sg_c.to(torch.float32))
        a_sg_list[1].copy_(x_sg_c.to(torch.float32))

    b_fp4_list[0] = fp4_col_list[0]
    b_fp4_list[1] = fp4_col_list[1]
    b_sc_list[0] = sc_col_list[0]
    b_sc_list[1] = sc_col_list[1]
    b_sg_list[0] = sg_list[0] if sg_list[0].dim() > 0 else sg_list[0].view(1)
    b_sg_list[1] = sg_list[1] if sg_list[1].dim() > 0 else sg_list[1].view(1)

    tk_dispatch_batched_gemm(
        tk,
        a_fp4_list, a_sc_list, a_sg_list,
        b_fp4_list, b_sc_list, b_sg_list,
        dW_T_list,
    )
    return _transpose_split_wgrad_outputs(
        dW_T_list, state, use_localcta=use_localcta, owner_key=owner_key
    )


# ---------------------------------------------------------------------------
# Legacy compat stubs — imported but not called in active code paths
# ---------------------------------------------------------------------------
_NVFP4_SCALE_RECIP = 1.0 / (6.0 * 448.0)

def prepare_tk_tensors(nvfp4_tensor, M, K):
    """Legacy stub — _TKQuantized objects already have _tk_row/_tk_col."""
    pass

def te_nvfp4_to_tk_format(nvfp4_tensor, M, K):
    """Legacy fallback — returns _tk_row if cached."""
    return nvfp4_tensor._tk_row

def te_nvfp4_to_tk_format_t(nvfp4_tensor, orig_rows, orig_cols):
    """Legacy fallback — returns _tk_col if cached."""
    return nvfp4_tensor._tk_col
