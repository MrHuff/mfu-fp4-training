"""
Fused TE Linear layers for FP4 training.

Provides:
  - TELinearFP4:         TE linear with NVFP4 recipe (no norm/act fusion)
  - NormTELinearFP4:     Fused RMSNorm + optional SiLU + FP4 quant → TE GEMM
  - FusedFeedForwardFP4: Replaces Llama FeedForward + ffn_norm as a single module

Forward quantization uses the TE-fused kernel from nvfp4_transpose_fused.cuh:
  Pass 1: Custom kernel computes inv_rms (per-row) + global_amax
  Pass 2: TE kernel fuses rmsnorm(x, gamma) → activation → NVFP4 quantization

Backward uses fused_silu_rmsnorm_backward.cu for dx + dgamma, and standard
TE NVFP4 GEMMs for dgrad/wgrad.
"""

import os
import sys
import math
import json
import logging
import time
from contextlib import contextmanager, nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


_delayed_fsdp_backward_prefetch_by_device: dict[int, tuple[object, ...]] = {}


class _DelayedFSDPBackwardPrefetchSentinel:
    _fsdp_param_group = None


_DELAYED_FSDP_BACKWARD_PREFETCH_SENTINEL = (
    _DelayedFSDPBackwardPrefetchSentinel()
)


def _install_delayed_fsdp_backward_prefetch() -> None:
    if os.environ.get(
        "TORCHTITAN_FSDP_DELAY_BACKWARD_PREFETCH_UNTIL_FFN_PRODUCER",
        "0",
    ) != "1":
        return

    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    if getattr(FSDPState, "_lbt_delayed_ffn_prefetch_installed", False):
        return

    original_pre_backward = FSDPState._pre_backward

    def _pre_backward_after_ffn_producer(self, grad):
        target_states = tuple(self._states_to_backward_prefetch)
        param_group = self._fsdp_param_group
        module_fqn = getattr(param_group, "_module_fqn", "") if param_group else ""
        delay_prefetch = bool(target_states) and "layers." in (module_fqn or "")
        if not delay_prefetch:
            return original_pre_backward(self, grad)

        device_index = grad.device.index
        if device_index is None:
            raise RuntimeError("delayed FSDP backward prefetch requires CUDA gradients")
        if device_index in _delayed_fsdp_backward_prefetch_by_device:
            raise RuntimeError(
                "an earlier delayed FSDP backward prefetch was not released"
            )

        self._states_to_backward_prefetch = [
            _DELAYED_FSDP_BACKWARD_PREFETCH_SENTINEL
        ]
        try:
            result = original_pre_backward(self, grad)
        finally:
            self._states_to_backward_prefetch = list(target_states)
        _delayed_fsdp_backward_prefetch_by_device[device_index] = target_states
        return result

    FSDPState._pre_backward = _pre_backward_after_ffn_producer
    FSDPState._lbt_delayed_ffn_prefetch_installed = True


def _release_delayed_fsdp_backward_prefetch(device: torch.device) -> bool:
    device_index = device.index
    if device_index is None:
        return False
    target_states = _delayed_fsdp_backward_prefetch_by_device.pop(
        device_index,
        None,
    )
    if target_states is None:
        return False

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
    )

    producer_done = torch.cuda.Event()
    producer_done.record(torch.cuda.current_stream(device))
    for target_state in target_states:
        target_param_group = getattr(target_state, "_fsdp_param_group", None)
        if target_param_group is not None:
            target_param_group._wait_all_gather_streams_on_event(producer_done)
            FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
    return True


_install_delayed_fsdp_backward_prefetch()

_NATIVE_LIGHT_IMPORT = (
    os.environ.get("LBT_LIGHT_IMPORT", "0") == "1"
    or os.environ.get("LBT_QUANTIZATION_LIGHT_IMPORT", "0") == "1"
)


class _LazyModule:
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            import importlib

            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


class _LazyCallable:
    def __init__(self, module_name: str, attr_name: str):
        self._module_name = module_name
        self._attr_name = attr_name
        self._target = None

    def _load(self):
        if self._target is None:
            import importlib

            self._target = getattr(importlib.import_module(self._module_name), self._attr_name)
        return self._target

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)


class _LazyMapping:
    def __init__(self, module_name: str, attr_name: str):
        self._module_name = module_name
        self._attr_name = attr_name
        self._target = None

    def _load(self):
        if self._target is None:
            import importlib

            self._target = getattr(importlib.import_module(self._module_name), self._attr_name)
        return self._target

    def __getitem__(self, key):
        return self._load()[key]


class _LazyOptionalCallable(_LazyCallable):
    def _load(self):
        if self._target is None:
            try:
                return super()._load()
            except Exception:
                self._target = False
        return None if self._target is False else self._target

    def __call__(self, *args, **kwargs):
        target = self._load()
        if target is None:
            raise RuntimeError(f"{self._module_name}.{self._attr_name} is unavailable")
        return target(*args, **kwargs)

    def __bool__(self):
        return self._load() is not None


class _NVFP4RoleMarker:
    def __init__(self, role: str, **kwargs):
        self._lbt_nvfp4_role = role
        self.kwargs = kwargs

    def copy(self):
        # TE 2.10+ copies quantizer metadata when wrapping prequantized payloads.
        return type(self)(self._lbt_nvfp4_role, **self.kwargs)

    def quantize(self, *_args, **_kwargs):
        raise RuntimeError(
            "NVFP4 role marker cannot quantize; use a native TK/localCTA producer"
        )


if _NATIVE_LIGHT_IMPORT:
    te = _LazyModule("transformer_engine.pytorch")
    tex = _LazyModule("transformer_engine_torch")
    NVFP4Quantizer = _LazyCallable("transformer_engine.pytorch", "NVFP4Quantizer")
    NVFP4Tensor = _LazyCallable(
        "transformer_engine.pytorch.tensor.nvfp4_tensor", "NVFP4Tensor"
    )
    TE_DType = _LazyMapping("transformer_engine.pytorch.constants", "TE_DType")
    NVFP4_BLOCK_SCALING_SIZE = 16
    te_apply_rotary_pos_emb = _LazyOptionalCallable(
        "transformer_engine.pytorch.attention.rope", "apply_rotary_pos_emb"
    )
    NVFP4BlockScaling = _LazyOptionalCallable(
        "transformer_engine.common.recipe", "NVFP4BlockScaling"
    )
else:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import NVFP4Quantizer
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor
    from transformer_engine.pytorch.constants import TE_DType, NVFP4_BLOCK_SCALING_SIZE
    try:
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb as te_apply_rotary_pos_emb
    except Exception:
        te_apply_rotary_pos_emb = None

    try:
        from transformer_engine.common.recipe import NVFP4BlockScaling
    except ImportError:
        NVFP4BlockScaling = None

    from .mxfp_custom_te_fp4 import BoundRecipeLinear

from .sqrelu import sqrelu_bwd, sqrelu_fwd
from .tk_gemm import (
    use_tk_gemm, use_tk_localcta, use_tk_localcta_direct_contract, use_tk_localcta_fused, prepare_tk_tensors,
    get_tk_localcta_variant,
    tk_forward_gemm, tk_forward_gemm_residual,
    tk_forward_gemm_residual_rms_partial,
    tk_dgrad_gemm, tk_wgrad_gemm,
    tk_v4_direct_dgrad_gemm, tk_v4_direct_raw_dgrad_gemm, tk_v4_direct_raw_wgrad_gemm, tk_v4_direct_wgrad_col_gemm,
    tk_dispatch_gemm, tk_dispatch_batched_gemm, tk_dispatch_batched_accum_gemm,
    tk_grouped_forward_gemm, tk_grouped_forward_gemm_split, tk_grouped_dgrad_gemm,
    tk_grouped_k_dgrad_gemm, tk_grouped_wgrad_gemm, tk_split_wgrad_gemm,
    can_use_localcta_split2_wo_backward, tk_localcta_split2_wo_backward,
    use_tk_localcta_wo_bf16_underflow_rescue,
    use_tk_localcta_wo_prepared_split2_backward,
    _localcta_adaptive_grad_boost_value, _get_tk_localcta_direct,
    tk_localcta_v3_split2_onepass_config_idx, use_tk_localcta_v4_cpp_only,
    use_tk_localcta_v4_strict_path, localcta_v4_cpp_only_scope,
    use_tk_localcta_v4_raw_backward_fallbacks,
    use_tk_localcta_v4_ffn_direct_grouped_wgrad_layout,
    use_tk_localcta_v4_gemm_virtual_rescale,
    use_tk_localcta_v4_ffn_residual_epilogue,
    use_tk_v5_ffn_residual_epilogue_for_shape,
    use_tk_ffn_rms_residual_bwd_for_shape,
    _ffn_rms_residual_aliases_input,
    _empty_chunk_sg, _chunk_sg_or_empty, _has_virtual_rescale_chunk,
    _packed_fp4_contiguous, _narrow_packed_fp4_contiguous,
    _get_tk, _fold_localcta_v4_sg_into_prepared_sc,
    _prepare_localcta_v4_outer_sg_for_direct,
    _prepare_localcta_v4_chunkgrid_for_batched,
    use_tk_ffn_plain_batched_accum_dgrad,
    _get_wgrad_stream, _launch_rmsnorm_bwd_out_async, _record_tensors_on_stream,
    _maybe_wrap_v5_ffn_quantizer, v5_ffn_quant_scope,
)


logger = logging.getLogger(__name__)
_backend_trace_once: set[str] = set()


def _get_bound_recipe_linear_cls():
    if not _NATIVE_LIGHT_IMPORT:
        return BoundRecipeLinear
    from .mxfp_custom_te_fp4 import BoundRecipeLinear

    return BoundRecipeLinear


def _trace_backend_choice(key: str, value: str) -> None:
    if os.environ.get('USE_TK_LOCALCTA_BACKEND_TRACE', '0') != '1':
        return
    token = f"{key}={value}"
    if token in _backend_trace_once:
        return
    _backend_trace_once.add(token)
    logger.info("[TK BACKEND] %s=%s", key, value)


def _use_tk_debug_log_localcta_function_grads() -> bool:
    return os.environ.get("USE_TK_DEBUG_LOG_LOCALCTA_FUNCTION_GRADS", "0") == "1"


def _should_emit_localcta_function_grad_debug(debug_name: Optional[str]) -> bool:
    if not _use_tk_debug_log_localcta_function_grads():
        return False
    name_filter = os.environ.get("USE_TK_DEBUG_LOCALCTA_FUNCTION_GRADS_FILTER", "").strip()
    return not name_filter or (debug_name is not None and name_filter in debug_name)


def _emit_localcta_function_grad_debug(kind: str, stats: dict[str, object]) -> None:
    if not _use_tk_debug_log_localcta_function_grads():
        return
    payload = {
        "step": int(os.environ.get("LBT_TRACE_ACTIVE_STEP", "-1")),
        "kind": kind,
        "stats": stats,
    }
    print(
        f"[LBT LOCALCTA FUNC] {json.dumps(payload, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def _tensor_debug_stats(tensor: Optional[torch.Tensor]) -> dict[str, object]:
    if tensor is None:
        return {"present": False}
    local = tensor.detach()
    to_local = getattr(local, "to_local", None)
    if callable(to_local):
        try:
            maybe_local = to_local()
        except Exception:
            maybe_local = None
        if maybe_local is not None:
            local = maybe_local
    flat = local.reshape(-1).to(torch.float32)
    numel = int(flat.numel())
    if numel == 0:
        return {
            "present": True,
            "numel": 0,
            "finite_count": 0,
            "nonfinite_count": 0,
            "zero_fraction": None,
            "rms": None,
            "max_abs": None,
        }
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    nonfinite_count = numel - finite_count
    zero_fraction = float((flat == 0).sum().item()) / numel
    rms = None
    max_abs = None
    if finite_count > 0:
        finite_flat = flat[finite]
        rms = math.sqrt(float(torch.sum(finite_flat * finite_flat).item()) / finite_count)
        max_abs = float(torch.max(torch.abs(finite_flat)).item())
    return {
        "present": True,
        "numel": numel,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "zero_fraction": zero_fraction,
        "rms": rms,
        "max_abs": max_abs,
    }


def _fp4_matmul_root(repo_root: str | None = None) -> str:
    root = os.environ.get("FP4_MATMUL_ROOT")
    if root:
        return os.path.abspath(root)
    if repo_root is None:
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(_this_dir))
    parent = os.path.dirname(repo_root)
    candidates = [
        os.path.join(parent, "fp4_matmul"),
        os.path.join(parent, "fp4_matmul-54-debug"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def use_custom_quant() -> bool:
    """Check if custom quantisation path is enabled via NVTE_CUSTOM_QUANT=1."""
    import os
    return os.environ.get('NVTE_CUSTOM_QUANT', '0') == '1'


def use_fused_te_quant() -> bool:
    """Check if TE-fused rmsnorm+quantize path is enabled via FUSED_TE_QUANT=1.

    When enabled, uses nvte_quantize_rmsnorm_silu / nvte_quantize_rmsnorm
    (single C++ dispatch) instead of the 2-step approach (rmsnorm→bf16→nvte_quantize_v2).
    This is the path that achieved 60% MFU.
    """
    return os.environ.get('FUSED_TE_QUANT', '0') == '1'


def use_tk_quant() -> bool:
    """Check if TK standalone quantisation is enabled via USE_TK_QUANT=1.

    When enabled, uses the pre-compiled _tk_quant module for FP4 quantisation
    instead of the TE JIT-compiled fp4_quantize_ext. Requires USE_TK_GEMM=1.
    """
    return os.environ.get('USE_TK_QUANT', '0') == '1'


def use_tk_ffn_debug_timings() -> bool:
    return os.environ.get('USE_TK_FFN_DEBUG_TIMINGS', '0') == '1'


def use_tk_wo_debug_timings() -> bool:
    return os.environ.get('USE_TK_WO_DEBUG_TIMINGS', '0') == '1'


def use_tk_qkv_debug_timings() -> bool:
    return os.environ.get('USE_TK_QKV_DEBUG_TIMINGS', '0') == '1'


def use_localcta_mamba_out_weight_quant_overlap(
    debug_name: str | None,
    m: int,
    n: int,
    k: int,
) -> bool:
    return (
        _env_flag("USE_TK_LOCALCTA_V4_MAMBA_OUT_WEIGHT_QUANT_OVERLAP", False)
        and use_tk_localcta()
        and get_tk_localcta_variant() == "v4"
        and (m, n, k) in {
            (8192, 4096, 8192),
            (16384, 4096, 8192),
            (24576, 4096, 8192),
            (32768, 4096, 8192),
        }
        and ".mixer.out_proj" in (debug_name or "")
    )


def _debug_timing_name_enabled(env_name: str, debug_name: str | None) -> bool:
    name_filter = os.environ.get(env_name, "").strip()
    if not name_filter:
        return True
    return name_filter in (debug_name or "")


def _debug_timing_step_enabled(env_name: str) -> bool:
    step_filter = (
        os.environ.get(env_name, "").strip()
        or os.environ.get("TK_STAGE_TRACE_STEP", "").strip()
    )
    if not step_filter:
        return True
    return os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip() == step_filter


def use_tk_ffn_debug_timings_for(debug_name: str | None) -> bool:
    return (
        use_tk_ffn_debug_timings()
        and _debug_timing_step_enabled("USE_TK_FFN_DEBUG_TIMINGS_STEP")
        and _debug_timing_name_enabled("USE_TK_FFN_DEBUG_TIMINGS_FILTER", debug_name)
    )


def use_tk_wo_debug_timings_for(debug_name: str | None) -> bool:
    return (
        use_tk_wo_debug_timings()
        and _debug_timing_step_enabled("USE_TK_WO_DEBUG_TIMINGS_STEP")
        and _debug_timing_name_enabled("USE_TK_WO_DEBUG_TIMINGS_FILTER", debug_name)
    )


def use_tk_qkv_debug_timings_for(debug_name: str | None) -> bool:
    return (
        use_tk_qkv_debug_timings()
        and _debug_timing_step_enabled("USE_TK_QKV_DEBUG_TIMINGS_STEP")
        and _debug_timing_name_enabled("USE_TK_QKV_DEBUG_TIMINGS_FILTER", debug_name)
    )


@contextmanager
def _ffn_cuda_timed(timings, name: str, stream=None):
    if timings is None:
        yield
        return
    if stream is None:
        stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    try:
        yield
    except Exception:
        raise
    else:
        end.record(stream)
        timings.append((name, start, end))


def _emit_ffn_debug_timings_once(debug_name, M: int, K: int, H: int, timings):
    if not timings:
        return
    key = (debug_name or "ffn", int(M), int(K), int(H), torch.cuda.current_device())
    emitted = getattr(_emit_ffn_debug_timings_once, "_emitted", set())
    if key in emitted:
        return
    emitted.add(key)
    _emit_ffn_debug_timings_once._emitted = emitted
    if _env_flag("USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_DEVICE", False):
        torch.cuda.synchronize()
    else:
        torch.cuda.current_stream().synchronize()
    rank = "na"
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = str(torch.distributed.get_rank())
    except Exception:
        rank = "na"
    parts = [f"{name}={start.elapsed_time(end):.3f}ms" for name, start, end in timings]
    print(
        "[FFN TIMING] "
        f"rank={rank} debug={debug_name or 'ffn'} M={M} K={K} H={H} " + " ".join(parts),
        flush=True,
    )


def _emit_wo_debug_timings_once(debug_name, phase: str, M: int, K: int, N: int, timings):
    if not timings:
        return
    key = (phase, debug_name or "wo", int(M), int(K), int(N), torch.cuda.current_device())
    emitted = getattr(_emit_wo_debug_timings_once, "_emitted", set())
    if key in emitted:
        return
    emitted.add(key)
    _emit_wo_debug_timings_once._emitted = emitted
    torch.cuda.synchronize()
    parts = [f"{name}={start.elapsed_time(end):.3f}ms" for name, start, end in timings]
    print(
        "[WO TIMING] "
        f"phase={phase} debug={debug_name or 'wo'} M={M} K={K} N={N} " + " ".join(parts),
        flush=True,
    )


def _emit_qkv_debug_timings_once(debug_name, phase: str, M: int, K: int, N: int, timings):
    if not timings:
        return
    key = (phase, debug_name or "qkv", int(M), int(K), int(N), torch.cuda.current_device())
    emitted = getattr(_emit_qkv_debug_timings_once, "_emitted", set())
    if key in emitted:
        return
    emitted.add(key)
    _emit_qkv_debug_timings_once._emitted = emitted
    torch.cuda.synchronize()
    parts = [f"{name}={start.elapsed_time(end):.3f}ms" for name, start, end in timings]
    print(
        "[QKV TIMING] "
        f"phase={phase} debug={debug_name or 'qkv'} M={M} K={K} N={N} " + " ".join(parts),
        flush=True,
    )


def use_nvfp4_encode_centric() -> bool:
    """Shared encode-centric toggle for TE/TK NVFP4 quantization."""
    return os.environ.get('NVTE_NVFP4_ENCODE_CENTRIC', '0') == '1'


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _sync_after_ffn_bwd_if_enabled(debug_name, M: int, K: int, H: int) -> None:
    """Env-gated replacement for the debug-timing FFN backward boundary sync."""
    if not _env_flag("USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD", False):
        return
    if not _debug_timing_step_enabled("USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_STEP"):
        return
    if not _debug_timing_name_enabled("USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_FILTER", debug_name):
        return
    if _env_flag("USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_ONCE", True):
        key = (debug_name or "ffn", int(M), int(K), int(H), torch.cuda.current_device())
        synced = getattr(_sync_after_ffn_bwd_if_enabled, "_synced", set())
        if key in synced:
            return
        synced.add(key)
        _sync_after_ffn_bwd_if_enabled._synced = synced
    if _env_flag("USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_DEVICE", False):
        torch.cuda.synchronize()
    else:
        torch.cuda.current_stream().synchronize()


def use_nvfp4_mxfp4_live_path() -> bool:
    """Use the NVFP4 wrapper route that mirrors MXFP4 live-path defaults."""
    return (
        _env_flag('USE_NVFP4_MXFP4_LIVE_PATH', False)
        or _env_flag('NVFP4_MIMIC_MXFP4_LIVE_PATH', False)
    )


def _nvfp4_any_role_flag_enabled(prefix: str) -> bool:
    return any(
        _env_flag(f"{prefix}_{role.upper()}", False)
        for role in ("activation", "grad", "weight")
    )


def use_nvfp4_rht() -> bool:
    return _env_flag('NVFP4_USE_RHT', False) or _nvfp4_any_role_flag_enabled("NVFP4_RHT")


def use_nvfp4_rht_for_role(role: str) -> bool:
    role = _normalize_nvfp4_role(role)
    value = os.environ.get(f"NVFP4_RHT_{role.upper()}")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    if not _env_flag('NVFP4_USE_RHT', False):
        return False
    defaults = {
        "activation": True,
        "grad": True,
        "weight": False,
    }
    return defaults.get(role, True)


def use_nvfp4_data_stochastic_rounding_for_role(role: str) -> bool:
    role = _normalize_nvfp4_role(role)
    value = os.environ.get(f"NVFP4_SR_{role.upper()}")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    if not _env_flag('NVFP4_USE_STOCHASTIC_ROUNDING', False):
        return False
    return True


def _nvfp4_grad_sr_axes() -> str:
    """Select which localCTA gradient orientation receives data SR.

    The native producer emits independent row and column quantizations.  The
    row payload feeds dgrad while the column payload feeds wgrad.
    """
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


def use_nvfp4_scale_stochastic_rounding() -> bool:
    return (
        _env_flag('NVFP4_USE_SCALE_STOCHASTIC_ROUNDING', False)
        or _nvfp4_any_role_flag_enabled("NVFP4_SCALE_SR")
    )


def use_nvfp4_scale_stochastic_rounding_for_role(role: str) -> bool:
    role = _normalize_nvfp4_role(role)
    value = os.environ.get(f"NVFP4_SCALE_SR_{role.upper()}")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    if not _env_flag('NVFP4_USE_SCALE_STOCHASTIC_ROUNDING', False):
        return False
    value = os.environ.get(f"NVFP4_SR_{role.upper()}")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return True


def _normalize_nvfp4_role(role: str | None) -> str:
    role = (role or "activation").strip().lower()
    aliases = {
        "act": "activation",
        "input": "activation",
        "dy": "grad",
        "gradient": "grad",
        "w": "weight",
    }
    role = aliases.get(role, role)
    if role not in {"activation", "weight", "grad"}:
        raise ValueError(f"Unsupported NVFP4 quantization role {role!r}")
    return role


def _nvfp4_rht_axes() -> str:
    axes = os.environ.get("NVFP4_RHT_AXES", "both").strip().lower().replace("-", "_")
    aliases = {
        "rowcol": "both",
        "row_col": "both",
        "all": "both",
        "cols": "col",
        "columns": "col",
        "rows": "row",
    }
    axes = aliases.get(axes, axes)
    if axes not in {"row", "col", "both"}:
        raise ValueError(f"Unsupported NVFP4_RHT_AXES={axes!r}; expected row, col, or both")
    return axes


def _nvfp4_rht_random_sign_mask() -> bool:
    return _env_flag("NVFP4_RHT_RANDOM_SIGNS", False)


def use_tk_localcta_paired_rht_carrier() -> bool:
    """Use the common SR/RHT-capable localCTA ablation carrier.

    The production split2/split3 and NHSD producers predate axis-selective
    paired column RHT.  They either reject the transform or silently omit it.
    Paired ablations therefore opt both control and treatment into a common
    carrier that keeps the proven row-gradient SR policy while routing the
    transformed column payload through the native v4 opt quantizer.
    """
    return _env_flag("USE_TK_LOCALCTA_PAIRED_RHT_CARRIER", False)


def _validate_nvfp4_rht_contract(role: str) -> None:
    """Reject gradient RHT unless the wgrad contraction is transformed in pairs."""
    role = _normalize_nvfp4_role(role)
    if role != "grad" or not use_nvfp4_rht_for_role(role):
        return
    if (
        _nvfp4_rht_axes() == "col"
        and use_nvfp4_rht_for_role("activation")
        and not use_nvfp4_rht_for_role("weight")
    ):
        # Forward and dgrad consume the untouched row payloads. Wgrad consumes
        # the column payloads, where the same orthogonal block-16 transform is
        # applied to dY and its matching saved activation.
        return
    raise RuntimeError(
        "NVFP4 gradient RHT is not a valid standalone quantization transform: "
        "the current backward path rotates dY without applying the matching "
        "transform to the weight (dgrad) or activation (wgrad) operand. This "
        "corrupts the resulting gradients. The supported paired policy is "
        "NVFP4_RHT_AXES=col with RHT enabled for activation and gradient only."
    )


def _nvfp4_rng_seed() -> int:
    return int(os.environ.get("NVFP4_RNG_SEED", "0"))


def _nvfp4_rng_subsequence_base() -> int:
    return int(os.environ.get("NVFP4_RNG_SUBSEQUENCE_BASE", "0"))


def _localcta_ffn_sr_states(
    debug_name: str | None, device: torch.device | str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    from .localcta_sr_state import (
        active_localcta_sr_state,
        ffn_deriv_grad_key,
        ffn_w2_grad_key,
    )

    state = active_localcta_sr_state()
    if state is None:
        return None, None
    return (
        state.get(ffn_w2_grad_key(debug_name), device),
        state.get(ffn_deriv_grad_key(debug_name), device),
    )


def _localcta_wo_sr_state(
    debug_name: str | None, device: torch.device | str
) -> torch.Tensor | None:
    from .localcta_sr_state import active_localcta_sr_state, wo_grad_key

    state = active_localcta_sr_state()
    return None if state is None else state.get(wo_grad_key(debug_name), device)


def _validate_checkpointed_localcta_wo_sr_route(
    persistent_rng_state: torch.Tensor | None,
    *,
    skip_generic_v4_wo_dy_quant: bool,
) -> None:
    """Reject the alternate WO split2 producer until it accepts explicit SR state."""
    if persistent_rng_state is not None and skip_generic_v4_wo_dy_quant:
        raise RuntimeError(
            "checkpointed localCTA SR cannot use the alternate WO split2 "
            "gradient producer because it lacks the explicit-state v4 ABI"
        )


def _call_with_optional_localcta_sr_state(
    fn, *args, persistent_rng_state: torch.Tensor | None
):
    if persistent_rng_state is None:
        return fn(*args)
    return fn(*args, persistent_rng_state)


def _tk_localcta_silu_deriv_split2_supports_rht(tkq_mod) -> bool:
    """Return whether the loaded extension explicitly supports split2 RHT.

    Older localCTA-v4 extensions expose the same split2 launch symbol but do
    not accept the RHT arguments.  Never infer support from that launch symbol:
    only the dedicated marker may opt this path into the native carrier.
    """
    if tkq_mod is None:
        return False
    marker = getattr(
        tkq_mod, "tk_localcta_silu_deriv_split2_supports_rht", None
    )
    if marker is None:
        return False
    if not callable(marker):
        raise RuntimeError(
            "localCTA split2 RHT capability marker exists but is not callable"
        )
    try:
        supported = marker()
    except Exception as exc:
        raise RuntimeError(
            "localCTA split2 RHT capability marker failed"
        ) from exc
    if not isinstance(supported, bool):
        raise RuntimeError(
            "localCTA split2 RHT capability marker must return bool"
        )
    if supported and not hasattr(
        tkq_mod,
        "tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace",
    ):
        raise RuntimeError(
            "localCTA extension advertises split2 RHT but lacks its launch API"
        )
    return supported


def _use_tk_localcta_native_paired_rht_split2(
    tkq_mod,
    *,
    paired_rht_carrier: bool,
) -> bool:
    """Select native split2 only for the actual column-RHT treatment."""
    return (
        paired_rht_carrier
        and _env_flag("USE_TK_LOCALCTA_NATIVE_PAIRED_RHT_SPLIT2", True)
        and use_nvfp4_rht_for_role("grad")
        and _tk_localcta_silu_deriv_split2_supports_rht(tkq_mod)
    )


def _validate_tk_localcta_paired_rht_split2_contract(
    persistent_rng_state: torch.Tensor | None,
) -> tuple[str, bool]:
    """Validate and return the native paired split2 RHT launch policy."""
    if persistent_rng_state is None:
        raise RuntimeError(
            "paired localCTA split2 carrier requires checkpointed SR state"
        )
    if not use_nvfp4_data_stochastic_rounding_for_role("grad"):
        raise RuntimeError("paired localCTA carrier requires gradient data SR")
    if _nvfp4_grad_sr_axes() != "row":
        raise RuntimeError("paired localCTA carrier requires row-only gradient SR")
    if use_nvfp4_scale_stochastic_rounding_for_role("grad"):
        raise RuntimeError("paired localCTA carrier requires gradient scale SR off")

    rht_axes = _nvfp4_native_rht_axes_for_role("grad")
    random_sign = (
        use_nvfp4_rht_for_role("grad") and _nvfp4_rht_random_sign_mask()
    )
    if use_nvfp4_rht_for_role("grad"):
        _validate_nvfp4_rht_contract("grad")
        if rht_axes != "col":
            raise RuntimeError("paired localCTA carrier supports column RHT only")
        if not random_sign:
            raise RuntimeError(
                "paired localCTA carrier requires the fixed-sign mask"
            )
    return rht_axes, random_sign


def _call_localcta_silu_deriv_split2(
    fn,
    *args,
    persistent_rng_state: torch.Tensor | None,
    native_paired_rht: bool,
):
    """Call split2 through the old ABI or the marker-gated native RHT ABI."""
    if not native_paired_rht:
        return _call_with_optional_localcta_sr_state(
            fn,
            *args,
            persistent_rng_state=persistent_rng_state,
        )

    rht_axes, random_sign = _validate_tk_localcta_paired_rht_split2_contract(
        persistent_rng_state
    )
    return fn(
        *args,
        rht_axes=rht_axes,
        with_random_sign_mask=random_sign,
        derivatives_precomputed=True,
        encode_centric=use_nvfp4_encode_centric(),
        persistent_rng_state=persistent_rng_state,
    )


_nvfp4_quantizer_role_by_id: dict[int, str] = {}


def _nvfp4_quantizer_role(quantizer, role: str | None = None) -> str:
    if role is not None:
        return _normalize_nvfp4_role(role)
    if quantizer is None:
        return "activation"
    mapped = _nvfp4_quantizer_role_by_id.get(id(quantizer))
    if mapped is not None:
        return mapped
    return _normalize_nvfp4_role(getattr(quantizer, "_lbt_nvfp4_role", "activation"))


def _nvfp4_quantizer_extras_enabled(role: str) -> bool:
    role = _normalize_nvfp4_role(role)
    return (
        use_nvfp4_rht_for_role(role)
        or use_nvfp4_data_stochastic_rounding_for_role(role)
        or use_nvfp4_scale_stochastic_rounding_for_role(role)
    )


def use_tk_localcta_v4_fused_sqrelu_quant() -> bool:
    return _env_flag("USE_TK_LOCALCTA_V4_FUSED_SQRELU_QUANT", False)


def use_tk_localcta_v4_fused_sqrelu_deriv_quant() -> bool:
    return _env_flag("USE_TK_LOCALCTA_V4_FUSED_SQRELU_DERIV_QUANT", False)


def use_tk_regular_fused_sqrelu_quant(role: str) -> bool:
    value = os.environ.get("USE_TK_REGULAR_FUSED_SQRELU_QUANT")
    if value is not None:
        return _env_flag("USE_TK_REGULAR_FUSED_SQRELU_QUANT", False)
    return _nvfp4_quantizer_extras_enabled(role)


def use_tk_regular_fused_sqrelu_deriv_quant(role: str) -> bool:
    value = os.environ.get("USE_TK_REGULAR_FUSED_SQRELU_DERIV_QUANT")
    if value is not None:
        return _env_flag("USE_TK_REGULAR_FUSED_SQRELU_DERIV_QUANT", False)
    return _nvfp4_quantizer_extras_enabled(role)


def use_tk_localcta_v4_sqrelu_delay_col_quant() -> bool:
    return _env_flag("USE_TK_LOCALCTA_V4_SQRELU_DELAY_COL_QUANT", False)


def _check_nvfp4_native_extras_supported(role: str, path: str) -> None:
    role = _normalize_nvfp4_role(role)
    if not _nvfp4_quantizer_extras_enabled(role):
        return
    raise NotImplementedError(
        f"NVFP4 RHT/SR for role={role} is not implemented in native {path} yet. "
        "Use the regular TK/TE quantizer path for SR/RHT experiments, or port the "
        "native producer before enabling these flags on localCTA/v4."
    )


def _nvfp4_native_rht_axes_for_role(role: str) -> str:
    role = _normalize_nvfp4_role(role)
    return _nvfp4_rht_axes() if use_nvfp4_rht_for_role(role) else "none"


def _nvfp4_quantizer_kwargs(role: str, te_dtype=None) -> dict:
    role = _normalize_nvfp4_role(role)
    if use_nvfp4_scale_stochastic_rounding_for_role(role):
        raise NotImplementedError(
            "NVFP4 scale stochastic rounding requires a native TK/localCTA producer path. "
            "Use USE_TK_QUANT=1 or USE_TK_LOCALCTA=1, or disable "
            "NVFP4_USE_SCALE_STOCHASTIC_ROUNDING for the pure TE path."
        )
    if te_dtype is None:
        te_dtype = tex.DType.kFloat4E2M1
    use_rht = use_nvfp4_rht_for_role(role)
    if use_rht and _nvfp4_rht_axes() != "both":
        raise NotImplementedError(
            "NVFP4 RHT is currently wired through TE's quantizer, which does not expose "
            "row/col axis selection. Use NVFP4_RHT_AXES=both or port native axis-selective RHT."
        )
    return {
        "fp4_dtype": te_dtype,
        "rowwise": True,
        "columnwise": True,
        "with_amax_reduction": False,
        "amax_reduction_group": None,
        "with_rht": use_rht,
        "with_post_rht_amax": use_rht,
        "with_2d_quantization": False,
        "stochastic_rounding": use_nvfp4_data_stochastic_rounding_for_role(role),
        "with_random_sign_mask": use_rht and _nvfp4_rht_random_sign_mask(),
        "encode_centric": use_nvfp4_encode_centric(),
    }


def _make_nvfp4_quantizer_for_role(role: str, te_dtype=None):
    role = _normalize_nvfp4_role(role)
    _validate_nvfp4_rht_contract(role)
    if use_tk_quant() or use_tk_localcta():
        quantizer = _NVFP4RoleMarker(role, fp4_dtype=te_dtype)
        _nvfp4_quantizer_role_by_id[id(quantizer)] = role
        return quantizer
    try:
        kwargs = _nvfp4_quantizer_kwargs(role, te_dtype)
    except NotImplementedError:
        native_tk_extras = (
            _nvfp4_quantizer_extras_enabled(role)
            and (
                (use_tk_localcta() and get_tk_localcta_variant() == 'v4')
                or (use_tk_quant() and not use_tk_localcta())
            )
        )
        if not native_tk_extras:
            raise
        # Native TK producers apply axis-selective RHT/SR themselves.
        # The TE quantizer object is still threaded through several call sites
        # as a role marker, so construct a neutral placeholder instead of
        # asking TE to enable unsupported row/col-selective RHT.
        kwargs = {
            "fp4_dtype": te_dtype or tex.DType.kFloat4E2M1,
            "rowwise": True,
            "columnwise": True,
            "with_amax_reduction": False,
            "amax_reduction_group": None,
            "with_rht": False,
            "with_post_rht_amax": False,
            "with_2d_quantization": False,
            "stochastic_rounding": False,
        }
        kwargs["encode_centric"] = use_nvfp4_encode_centric()
    quantizer = _make_nvfp4_quantizer_compat(**kwargs)
    _nvfp4_quantizer_role_by_id[id(quantizer)] = role
    try:
        quantizer._lbt_nvfp4_role = role
    except Exception:
        pass
    return quantizer


def use_tk_localcta_v4_fast_prepared_producer() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_PREPARED_PRODUCER', '0') == '1'


def use_tk_localcta_v4_fast_prepared_producer_for_shape(m: int, k: int) -> bool:
    if not use_tk_localcta_v4_fast_prepared_producer():
        return False
    if os.environ.get('USE_TK_LOCALCTA_V4_FAST_PREPARED_PRODUCER_ALLOW_HIGH_M', '0') == '1':
        return True
    # The 8B TP2 Bridge activation shape (M=32768,K=4096) can trip a CUDA
    # launch failure in the current prepared producer path. Leave smaller
    # shapes and explicit unsafe probes untouched.
    return not (m >= 32768 and k >= 4096)


def use_tk_localcta_v4_row_prepared_col_outer() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_ROW_PREPARED_COL_OUTER', '1') == '1'


def use_tk_localcta_v4_row_prepared_rmsnorm_quant() -> bool:
    return _env_flag("USE_TK_LOCALCTA_V4_ROW_PREPARED_RMSNORM_QUANT", False)


def use_tk_localcta_v4_raw_outer_tma_grad() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_RAW_OUTER_TMA_GRAD', '0') == '1'


def _make_nvfp4_quantizer_compat(**kwargs):
    try:
        return NVFP4Quantizer(**kwargs)
    except TypeError as exc:
        if 'encode_centric' not in kwargs or 'encode_centric' not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs.pop('encode_centric', None)
        return NVFP4Quantizer(**kwargs)


def _make_nvfp4_tensor_compat(*args, **kwargs):
    try:
        return NVFP4Tensor(*args, **kwargs)
    except TypeError as exc:
        if 'with_gemm_swizzled_scales' not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs.setdefault('with_gemm_swizzled_scales', False)
        return NVFP4Tensor(*args, **kwargs)


def use_tk_qkv_forward_cat_debug() -> bool:
    """Disable split-D QKV output writes when conn1 exposes the TK scheduling race."""
    mode = os.environ.get('USE_TK_QKV_FORWARD_CAT_DEBUG', 'auto').strip().lower()
    if mode in ('1', 'true', 'yes', 'on'):
        return True
    if mode in ('0', 'false', 'no', 'off'):
        return False
    # The regular TK split-D QKV writer can wedge in full-model training on
    # GB200. Keep localCTA/v4 on its dedicated path, but route plain TK through
    # the single-output grouped GEMM unless the caller explicitly opts out.
    if use_tk_gemm() and not use_tk_localcta():
        return True
    if use_tk_localcta() and get_tk_localcta_variant() == 'v4':
        return False
    return os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '').strip() == '1'


def use_tk_qkv_forward_nopdl() -> bool:
    """Prefer no-PDL QKV split-D when conn1 exposes the TK PDL scheduling race."""
    mode = os.environ.get('USE_TK_QKV_FORWARD_NOPDL', 'auto').strip().lower()
    if mode in ('1', 'true', 'yes', 'on'):
        return True
    if mode in ('0', 'false', 'no', 'off'):
        return False
    # Plain regular-TK QKV forward can wedge on GB200 under the split-D PDL
    # route even without RHT/SR.  Keep localCTA/v4 on its dedicated path, but
    # prefer the no-PDL QKV forward GEMM for regular TK unless explicitly opted
    # out above.
    if use_tk_gemm() and not use_tk_localcta():
        return True
    return os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '').strip() == '1'


def use_tk_qkv_te_act_quant() -> bool:
    """Use TE/custom swizzled activation quant for TK/localCTA QKV forward.

    This keeps the fast TK/localCTA grouped weight path and GEMMs, but bypasses
    the standalone TK activation quantizer. It is internal-only and targets the
    shared attention numerics gap where:
      - TE activation quant + TK GEMM is effectively exact
      - TK activation quant + TK GEMM carries almost the whole QKV error
    """
    return os.environ.get('USE_TK_QKV_TE_ACT_QUANT', '0') == '1'


def use_tk_qkv_tk_act_quant() -> bool:
    """Use TK-row/localCTA-col activation quant for strict-v4 QKV forward.

    This is not the final localCTA-v4 producer fix. It is a C++-only routing
    change that keeps strict-v4 GEMMs, localCTA weight quantization, and the
    localCTA column/backward path intact while replacing only the numerically
    bad localCTA rowwise forward producer contract.
    """
    return os.environ.get('USE_TK_QKV_TK_ACT_QUANT', '0') == '1'


def use_tk_qkv_fused_norm_quant() -> bool:
    """Use plain-TK fused RMSNorm row/col quant for QKV activations.

    This keeps the shared TK grouped GEMM path intact while replacing the
    decomposed `rmsnorm_only + tk_quantize_for_gemm` activation contract with
    the backend's fused row+col producer. The default is intentionally narrow:
    only the validated regular-TK activation-extras row-RHT contract uses the
    new opt producer automatically. Other contracts remain opt-in because the
    older non-extras fused producer was neutral and previously showed unstable
    full-model behavior.
    """
    value = os.environ.get('USE_TK_QKV_FUSED_NORM_QUANT')
    if value is not None:
        return value == '1'
    if (
        use_tk_quant()
        and not use_tk_localcta()
        and _nvfp4_quantizer_extras_enabled("activation")
        and _nvfp4_native_rht_axes_for_role("activation") == "row"
        and not _nvfp4_rht_random_sign_mask()
    ):
        return True
    return False


def use_tk_qkv_localcta_fused_rmsnorm_quant() -> bool:
    """Fuse localCTA-v4 QKV RMSNorm and activation quant under the direct contract."""
    value = os.environ.get('USE_TK_QKV_LOCALCTA_FUSED_RMSNORM_QUANT')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_fused_norm_quant() -> bool:
    """Opt-in regular-TK fused RMSNorm row/col quant for FFN activations."""
    value = os.environ.get('USE_TK_FFN_FUSED_NORM_QUANT')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_localcta_v4_qkv_fused_producer() -> bool:
    """Debug-only opt-in for the legacy localCTA-v4 fused QKV producer.

    The validated v4 route uses RMSNorm-only followed by the v4 QKV quantizer.
    The older fused RMSNorm+quant producer emits a different scale contract and
    over-scales QKV outputs when consumed by the current v4 GEMM path.
    """
    return os.environ.get('USE_TK_LOCALCTA_V4_QKV_FUSED_PRODUCER', '0') == '1'


def use_tk_qkv_rope_epilogue() -> bool:
    value = os.environ.get('USE_TK_QKV_ROPE_EPILOGUE')
    if value is not None:
        return value == '1'
    if use_tk_localcta():
        return os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
    return os.environ.get('USE_TK_GEMM', '0') == '1' and not use_tk_qkv_forward_cat_debug()


def use_tk_qkv_localcta_direct_forward_debug() -> bool:
    """Debug-only: localCTA QKV forward uses TK act quant + direct localCTA GEMM."""
    return os.environ.get('USE_TK_QKV_LOCALCTA_DIRECT_FORWARD_DEBUG', '0') == '1'


def use_tk_qkv_localcta_encode_centric() -> bool:
    """Internal override for localCTA QKV activation encode-centric mode.

    Leave the production default tied to the shared NVTE flag unless this env
    is explicitly set. LocalCTA attention numerics experiments can then flip
    the mode without silently changing the recovered default path.
    """
    value = os.environ.get('USE_TK_QKV_LOCALCTA_ENCODE_CENTRIC')
    if value is not None:
        return value == '1'
    return use_nvfp4_encode_centric()


def use_tk_qkv_localcta_tk_prepared_activation() -> bool:
    """Use TK raw activation quant with prepared-scale folding for localCTA QKV.

    This is internal-only and targets the remaining production localCTA QKV
    numerics gap. It leaves localCTA grouped weights and fast grouped GEMM
    unchanged, and only swaps the activation quantization contract.
    """
    value = os.environ.get('USE_TK_QKV_LOCALCTA_TK_PREPARED_ACT')
    if value is not None:
        return value == '1'
    return False


def use_tk_qkv_localcta_fast_activation() -> bool:
    """Use localCTA dual-write quant for QKV activations on the fast path.

    This keeps the fast localCTA GEMM contract intact, but swaps the activation
    producer from the prepared-only quant kernel to the dual-write
    `tk_quantize_for_gemm_fast` path. It is a strict localCTA-native control to
    answer whether the remaining QKV forward drift is created by the prepared-only
    activation producer itself.
    """
    value = os.environ.get('USE_TK_QKV_LOCALCTA_FAST_ACT')
    if value is not None:
        return value == '1'
    return False


def use_tk_qkv_localcta_fast_weights() -> bool:
    """Use localCTA dual-write grouped quant for QKV weights on the fast path.

    This keeps localCTA grouped GEMMs active, but swaps the grouped weight
    producer from the prepared-only kernel to the dual-write
    `tk_group_quantize_for_gemm_fast` path and feeds the prepared row scales
    that it emits. It isolates whether the shared QKV forward drift lives in
    the grouped weight producer.
    """
    value = os.environ.get('USE_TK_QKV_LOCALCTA_FAST_W')
    if value is not None:
        return value == '1'
    return False


def use_tk_qkv_localcta_weight_overlap() -> bool:
    """Overlap localCTA QKV weight quant with RMSNorm+activation quant.

    This is intentionally opt-in: it uses a side stream around the v4 direct
    producer, so keep the default single-stream path as the conservative
    numerics/debug baseline.
    """
    value = os.environ.get('USE_TK_QKV_LOCALCTA_WEIGHT_OVERLAP')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def tk_localcta_forward_min_m() -> int:
    value = os.environ.get('USE_TK_LOCALCTA_FORWARD_MIN_M')
    if value is None or value == '':
        return 256
    try:
        return max(0, int(value))
    except ValueError:
        return 256


def use_tk_localcta_forward_for_m(m: int) -> bool:
    return use_tk_localcta() and m >= tk_localcta_forward_min_m()


def use_tk_qkv_bf16_rmsnorm_bwd() -> bool:
    """Debug-only RMSNorm backward fallback for QKV numerics isolation."""
    return os.environ.get('USE_TK_QKV_BF16_RMSNORM_BWD', '0') == '1'


def use_tk_qkv_backward_debug_fallback() -> bool:
    """Disable QKV CUDA-graph fast path when explicit debug fallbacks are active."""
    return (
        os.environ.get('USE_TK_QKV_BF16_WGRAD', '0') == '1'
        or os.environ.get('USE_TK_QKV_BF16_DGRAD', '0') == '1'
        or use_tk_qkv_bf16_rmsnorm_bwd()
        or use_tk_attn_debug_finite()
    )


def _qkv_localcta_scale_num_override() -> float | None:
    value = os.environ.get('USE_TK_QKV_LOCALCTA_SCALE_NUM')
    if value is None or value == '':
        return None
    return float(value)


def _set_localcta_qkv_scale_num(tk_q) -> float | None:
    scale_num = _qkv_localcta_scale_num_override()
    if scale_num is None or not hasattr(tk_q, 'tk_set_global_scale_num'):
        return None
    prev = tk_q.tk_get_global_scale_num() if hasattr(tk_q, 'tk_get_global_scale_num') else None
    tk_q.tk_set_global_scale_num(scale_num)
    return prev


def _restore_localcta_qkv_scale_num(tk_q, prev: float | None) -> None:
    scale_num = _qkv_localcta_scale_num_override()
    if scale_num is None or not hasattr(tk_q, 'tk_set_global_scale_num'):
        return
    if prev is None:
        if hasattr(tk_q, 'tk_reset_global_scale_num'):
            tk_q.tk_reset_global_scale_num()
    else:
        tk_q.tk_set_global_scale_num(prev)


def use_tk_ms() -> bool:
    """Check if TK multi-stream overlap is enabled via USE_TK_MS=1.

    When enabled, weight quantisation runs on a secondary CUDA stream in
    parallel with input RMSNorm + quantisation. Requires USE_TK_QUANT=1.
    Uses v5 split API (v5_alloc on s0, v5_launch on s1).
    """
    return os.environ.get('USE_TK_MS', '0') == '1'


def use_cuda_graph() -> bool:
    """Check if CUDA graph capture is enabled via USE_CUDA_GRAPH=1.

    When enabled, forward+backward of QKV and Wo are captured into CUDA graphs
    on first call, then replayed. Requires fixed-shape inputs per layer.
    """
    return os.environ.get('USE_CUDA_GRAPH', '0') == '1'


def use_tk_localcta_ffn_fused_row_producer() -> bool:
    """Benchmark-only FFN eager localCTA path for fused producer + row quant."""
    return os.environ.get('USE_TK_LOCALCTA_FFN_FUSED_ROW_PRODUCER', '0') == '1'


def use_tk_localcta_v4_split2_two_stage() -> bool:
    """Experimental FFN split2 v4 path.

    Stage 1 owns row production and emits row payloads.
    Stage 2 reduces final col tile scales and writes col payloads once.
    """
    return os.environ.get('USE_TK_LOCALCTA_V4_SPLIT2_TWO_STAGE', '0') == '1'


def _localcta_ffn_experiment_min_m(env_name: str, default: int = 65536) -> int:
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _tk_localcta_ffn_experiment_mode() -> str:
    """Benchmark-only FFN eager localCTA experiments."""
    return os.environ.get('USE_TK_LOCALCTA_FFN_EXPERIMENT', 'off').strip().lower()


def use_tk_localcta_ffn_saved_sigmoid() -> bool:
    mode = _tk_localcta_ffn_experiment_mode()
    return mode in ('saved_sigmoid', 'saved_sigmoid_overlap', 'saved_sigmoid_overlap_w2highacc')


def use_tk_localcta_ffn_saved_sigmoid_overlap() -> bool:
    mode = _tk_localcta_ffn_experiment_mode()
    return mode in ('saved_sigmoid_overlap', 'saved_sigmoid_overlap_w2highacc')


def use_tk_localcta_ffn_saved_sigmoid_w2highacc() -> bool:
    return _tk_localcta_ffn_experiment_mode() == 'saved_sigmoid_overlap_w2highacc'


def use_tk_localcta_ffn_bf16_w2_backward_debug() -> bool:
    """Debug-only: bypass localCTA grad-output quant for the FFN W2 backward."""
    value = os.environ.get('USE_TK_LOCALCTA_FFN_BF16_W2_BWD')
    if value is not None:
        return value == '1'
    return False


def use_tk_localcta_ffn_bf16_dgrad_debug() -> bool:
    """Debug-only: bypass localCTA W1/W3 dgrad GEMMs in BF16."""
    value = os.environ.get('USE_TK_LOCALCTA_FFN_BF16_DGRAD')
    if value is not None:
        return value == '1'
    return False


def use_tk_debug_ffn_bf16_w2_dgrad() -> bool:
    """Diagnostic-only BF16 W2 dgrad for the regular-TK FFN path."""
    return os.environ.get('USE_TK_DEBUG_FFN_BF16_W2_DGRAD', '0') == '1'


def use_tk_debug_ffn_bf16_split_dgrad() -> bool:
    """Diagnostic-only BF16 W1/W3 dgrad for the regular-TK FFN path."""
    return os.environ.get('USE_TK_DEBUG_FFN_BF16_SPLIT_DGRAD', '0') == '1'


def use_tk_localcta_ffn_dequant_dgrad_debug() -> bool:
    """Debug-only: dequantize localCTA W1/W3 operands before dgrad GEMMs."""
    return get_tk_localcta_ffn_dequant_dgrad_debug_mode() != 'off'


def get_tk_localcta_ffn_dequant_dgrad_debug_mode() -> str:
    value = os.environ.get('USE_TK_LOCALCTA_FFN_DEQUANT_DGRAD_MODE')
    if value is None:
        value = (
            'both'
            if os.environ.get('USE_TK_LOCALCTA_FFN_DEQUANT_DGRAD') == '1'
            else 'off'
        )
    value = value.strip().lower()
    if value not in ('off', 'both', 'activation', 'weight'):
        raise ValueError(
            "USE_TK_LOCALCTA_FFN_DEQUANT_DGRAD_MODE must be one of "
            f"off/both/activation/weight, got {value!r}"
        )
    return value


def use_tk_localcta_ffn_bf16_rescue_on_zero_dy_sc() -> bool:
    """Use a BF16 rescue path when localCTA FFN grad-output scales underflow."""
    value = os.environ.get('USE_TK_LOCALCTA_FFN_BF16_RESCUE_ON_ZERO_DY_SC')
    if value is not None:
        return value == '1'
    return False


def use_tk_localcta_ffn_check_zero_dy_sc() -> bool:
    """Debug/diagnostic check for all-zero localCTA grad-output scale bytes."""
    value = os.environ.get('USE_TK_LOCALCTA_FFN_CHECK_ZERO_DY_SC')
    if value is not None:
        return value == '1'
    return use_tk_localcta_ffn_bf16_rescue_on_zero_dy_sc()


def get_tk_localcta_ffn_fixed_grad_boost() -> float:
    value = os.environ.get('USE_TK_LOCALCTA_FFN_FIXED_GRAD_BOOST')
    if value is None or value == '':
        return 1.0
    boost = float(value)
    if boost <= 0.0:
        raise ValueError(
            f"USE_TK_LOCALCTA_FFN_FIXED_GRAD_BOOST must be > 0, got {boost!r}"
        )
    return boost


def use_tk_debug_clone_ffn_input() -> bool:
    value = os.environ.get('USE_TK_DEBUG_CLONE_FFN_INPUT')
    if value is not None:
        return value == '1'
    return False


def use_tk_debug_clone_ffn_grad_output() -> bool:
    value = os.environ.get('USE_TK_DEBUG_CLONE_FFN_GRAD_OUTPUT')
    if value is not None:
        return value == '1'
    return False


def use_tk_ffn_bwd_safe_producer(
    m: int | None = None,
    k: int | None = None,
    h: int | None = None,
) -> bool:
    """Use the stable plain-TK FFN backward producer.

    The fused producer is enabled only at the measured regular-v5 Llama-8B
    shape. Delayed scaling and all other shapes retain their existing producer.
    """
    value = os.environ.get('USE_TK_FFN_BWD_SAFE_PRODUCER')
    if value is not None:
        return value == '1'
    regular_v5 = (
        os.environ.get('USE_TK_GEMM', '0') == '1'
        and not use_tk_localcta()
    )
    if (
        regular_v5
        and (m, k, h) == (32768, 4096, 14336)
        and not use_tk_ffn_v5_delayed_split_silu_deriv()
    ):
        return False
    return regular_v5


def use_tk_ffn_split_dgrad_eager() -> bool:
    """Use a debug-only eager plain-TK FFN split dgrad path."""
    value = os.environ.get('USE_TK_FFN_SPLIT_DGRAD_EAGER')
    if value is not None:
        return value == '1'
    return False


def use_tk_ffn_split_quant_eager() -> bool:
    """Use a debug-only eager plain-TK FFN split quant path."""
    value = os.environ.get('USE_TK_FFN_SPLIT_QUANT_EAGER')
    if value is not None:
        return value == '1'
    return False


def use_tk_ffn_split2_opt_producer() -> bool:
    """Use the regular-TK split2 producer that fuses SiLU-deriv with RHT/SR scans."""
    value = os.environ.get('USE_TK_FFN_SPLIT2_OPT_PRODUCER')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return True


def use_tk_ffn_split2_persistent_producer() -> bool:
    """Use the experimental persistent fused regular-TK split2 producer."""
    value = os.environ.get('USE_TK_FFN_SPLIT2_PERSISTENT_PRODUCER')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_split_cache() -> bool:
    """Use split h1/h3 FFN cache instead of saving the concatenated h13."""
    value = os.environ.get('USE_TK_FFN_SPLIT_CACHE')
    if value is not None:
        return value == '1'
    return True


def use_tk_ffn_recompute_h13() -> bool:
    """Recompute SwiGLU preactivations in backward instead of saving BF16 h1/h3."""
    value = os.environ.get('USE_TK_FFN_RECOMPUTE_H13')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_recompute_h_for_w2_wgrad() -> bool:
    """Recreate the quantized SwiGLU activation in backward for W2 wgrad."""
    value = os.environ.get('USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_requant_h13_operands() -> bool:
    """Requantize localCTA GEMM1 row operands during H1/H3 recomputation."""
    value = os.environ.get('USE_TK_FFN_REQUANT_H13_OPERANDS')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_requant_h13_activation() -> bool:
    """Requantize only the regular-v5 GEMM1 activation row in backward."""
    value = os.environ.get('USE_TK_FFN_REQUANT_H13_ACTIVATION')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_localcta_inplace_h13_deriv() -> bool:
    """Overwrite recomputed H1/H3 with their derivative outputs."""
    value = os.environ.get('USE_TK_FFN_LOCALCTA_INPLACE_H13_DERIV')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_localcta_v4_ffn_deriv_w2_wgrad_overlap() -> bool:
    """Overlap the v4 SiLU-derivative quant producer with W2 wgrad."""
    value = os.environ.get('USE_TK_LOCALCTA_V4_FFN_DERIV_W2_WGRAD_OVERLAP')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_prealloc_split_producer() -> bool:
    """Reuse regular-TK FFN split producer output/staging buffers."""
    value = os.environ.get('USE_TK_FFN_PREALLOC_SPLIT_PRODUCER')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_ffn_h13_delayed_silu_deriv() -> bool:
    """Use the h13 single-pass delayed-scaling SiLU-deriv producer."""
    if os.environ.get('USE_TK_FFN_H13_TILE_DELAYED_AMAX', '0') == '1':
        return True
    value = os.environ.get('USE_TK_FFN_H13_DELAYED_SILU_DERIV')
    if value is not None:
        return value == '1'
    return False


def tk_ffn_h13_delayed_refresh_interval() -> int:
    if (
        os.environ.get('USE_TK_FFN_H13_DELAYED_NO_COLLECT', '0') == '1'
        or os.environ.get('USE_TK_FFN_H13_TILE_DELAYED_NO_COLLECT', '0') == '1'
    ):
        return 0
    value = os.environ.get(
        'USE_TK_FFN_H13_DELAYED_REFRESH_INTERVAL',
        os.environ.get('USE_TK_FFN_H13_TILE_DELAYED_REFRESH_INTERVAL', '1'),
    )
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def use_tk_ffn_v5_delayed_split_silu_deriv() -> bool:
    """Use regular TK v5 split-cache SiLU-deriv quantization with delayed scaling."""
    value = os.environ.get('USE_TK_FFN_V5_DELAYED_SILU_DERIV')
    if value is None:
        return False
    return value == '1' and use_tk_quant() and not use_tk_localcta()


def tk_ffn_v5_delayed_refresh_interval() -> int:
    if os.environ.get('USE_TK_FFN_V5_DELAYED_NO_COLLECT', '0') == '1':
        return 0
    value = os.environ.get('USE_TK_FFN_V5_DELAYED_REFRESH_INTERVAL', '1')
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def _record_delayed_amax_event(tensor: torch.Tensor):
    if not torch.is_tensor(tensor) or not tensor.is_cuda:
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(tensor.device))
    return event


def _wait_delayed_amax_event(event, tensor: torch.Tensor) -> None:
    if event is None or not torch.is_tensor(tensor) or not tensor.is_cuda:
        return
    torch.cuda.current_stream(tensor.device).wait_event(event)


def use_tk_ffn_overlap_rms_wgrad() -> bool:
    """Overlap regular-TK FFN RMSNorm backward with split wgrad."""
    value = os.environ.get('USE_TK_FFN_OVERLAP_RMS_WGRAD')
    if value is not None:
        return value == '1'
    return False


def use_tk_sqrelu_ffn_overlap_w1_wgrad_rms() -> bool:
    """Overlap square-ReLU FFN W1 wgrad with RMSNorm backward."""
    value = os.environ.get('USE_TK_SQRELU_FFN_OVERLAP_W1_WGRAD_RMS')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_sqrelu_ffn_overlap_w2_wgrad_deriv() -> bool:
    """Overlap square-ReLU FFN W2 wgrad with derivative quantization."""
    value = os.environ.get('USE_TK_SQRELU_FFN_OVERLAP_W2_WGRAD_DERIV')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_tk_sqrelu_ffn_cached_rms_bwd() -> bool:
    """Use cached out-buffer RMSNorm backward in square-ReLU FFN backward."""
    value = os.environ.get('USE_TK_SQRELU_FFN_CACHED_RMS_BWD')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def use_localcta_sqrelu_w2_weight_quant_overlap(
    debug_name: str | None,
    m: int,
    n: int,
    k: int,
    h: int,
) -> bool:
    return (
        _env_flag(
            "USE_TK_LOCALCTA_V4_SQRELU_W2_WEIGHT_QUANT_OVERLAP",
            False,
        )
        and use_tk_localcta()
        and get_tk_localcta_variant() == "v4"
        and (m, n, k, h) in {
            (8192, 4096, 4096, 21504),
            (16384, 4096, 4096, 21504),
            (24576, 4096, 4096, 21504),
            (32768, 4096, 4096, 21504),
        }
        and ".mixer" in (debug_name or "")
    )


def use_tk_ffn_fused_sum_rms() -> bool:
    """Fuse regular-TK FFN split-dgrad summation into RMSNorm backward."""
    value = os.environ.get('USE_TK_FFN_FUSED_SUM_RMS')
    if value is not None:
        return value == '1'
    return True


def use_tk_ffn_safe_input_quant() -> bool:
    """Debug-only plain-TK FFN input-quant fallback.

    Uses TE RMSNorm-only + standard FP4 quant instead of the fused
    tk_fused_norm_quantize input producer. This is intended to isolate the
    remaining full-TK trainer wedge where TK attention outputs feed TK FFN.
    """
    value = os.environ.get('USE_TK_FFN_SAFE_INPUT_QUANT')
    if value is not None:
        return value == '1'
    return False


def use_tk_ffn_fwd_safe_producer() -> bool:
    """Debug-only plain-TK FFN forward producer fallback.

    Computes silu(h1_raw) * h3 in BF16 first, then quantizes with the standard
    FP4 path before the W2 GEMM. This isolates whether the fused TK forward
    producer is the owner of the remaining full-TK trainer wedge.
    """
    value = os.environ.get('USE_TK_FFN_FWD_SAFE_PRODUCER')
    if value is not None:
        return value == '1'
    return False


def use_tk_ffn_localcta_tk_quant_contract() -> bool:
    """Use the fast localCTA-v4 FFN outer-scale route.

    FFN v4 should not enter the old strict/native split2 formulation by
    default. The production route is the fast full-K GEMM consumer with one
    outer epilogue scale per output tile.
    """
    if not (
        use_tk_localcta()
        and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
    ):
        return False
    return True


def use_tk_localcta_v4_strict_prepared_split2() -> bool:
    """Use the localCTA-v4 prepared split2 producer inside the strict FFN path.

    This is still a C++ localCTA-v4 producer path, but it emits GEMM-ready
    prepared scales directly instead of raw chunk SG plus a separate finalize.
    """
    return os.environ.get('USE_TK_LOCALCTA_V4_STRICT_PREPARED_SPLIT2', '0') == '1'


def use_tk_localcta_v4_ffn_prepared_split2_producer() -> bool:
    """Use the prepared one-pass SiLU-deriv split2 producer for fast v4 FFN."""
    value = os.environ.get('USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_PRODUCER')
    if value is not None:
        return value == '1'
    return False


def use_tk_localcta_v4_w2_dgrad_silu_producer() -> bool:
    """Fuse W2 dgrad with SiLU derivative split2 quantization for localCTA v4."""
    return os.environ.get('USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER', '0') == '1'


def use_tk_localcta_v4_w2_dgrad_silu_producer_fresh_payload() -> bool:
    """Allocate a fresh split2 output payload for each W2-dgrad producer launch.

    The high-H clustered producer is stable in isolation when each launch uses
    a newly allocated payload, but it can fail when reusing prepared output
    buffers. Keep this explicit so the default cached path remains unchanged.
    """
    return os.environ.get(
        'USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER_FRESH_PAYLOAD', '0'
    ) == '1'


def use_tk_localcta_v4_w2_dgrad_silu_producer_priority_stream() -> bool:
    """Launch the clustered W2 producer on a high-priority CUDA stream."""
    return os.environ.get(
        'USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER_PRIORITY_STREAM', '0'
    ) == '1'


def use_tk_localcta_v4_sync_ffn_rms_bwd() -> bool:
    """Run localCTA-v4 FFN RMSNorm backward on the current stream."""
    return os.environ.get('USE_TK_LOCALCTA_V4_SYNC_FFN_RMS_BWD', '0') == '1'


def use_tk_localcta_v4_w2_dgrad_silu_producer_unit_sg() -> bool:
    """Use unit outer scales for the fused W2-dgrad producer.

    This is a diagnostic escape hatch. The fused W2 producer performs the
    W2-dgrad GEMM internally, so it needs the same outer scales as the regular
    localCTA GEMM. Passing unit scales over-amplifies the derived split2
    payload and corrupts gradient magnitudes.
    """
    value = os.environ.get('USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER_UNIT_SG')
    if value is None:
        return False
    return value == '1'


def tk_localcta_v4_w2_dgrad_silu_producer_shape_safe(m: int, h: int) -> bool:
    """Avoid the current high-H clustered-launch failure in the v4 producer.

    The fused producer launches at the 8B TP2 FFN shape in isolation, but both
    the PDL and non-PDL variants have failed live/valid-payload validation.
    Keep high-H disabled unless an unsafe experiment explicitly overrides it.
    """
    if not (m >= 32768 and h >= 7168):
        return True
    return (
        os.environ.get('USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER_ALLOW_UNSAFE_HIGH_H', '0') == '1'
        or os.environ.get('USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER_ALLOW_HIGH_H', '0') == '1'
    )


def tk_localcta_v4_w2_dgrad_silu_producer_config_id() -> int:
    value = os.environ.get('USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER_CONFIG_ID', '4')
    try:
        return int(value)
    except ValueError:
        return 4


def disable_tk_localcta_v4_ffn_fused_prepared_deriv_quant() -> bool:
    """Route strict v4 split2 through BF16 dh1/dh3 before direct fused quant."""
    value = os.environ.get('USE_TK_LOCALCTA_V4_FFN_DISABLE_FUSED_PREPARED_DERIV_QUANT')
    if value is not None:
        return value == '1'
    return True


def use_tk_localcta_v4_ffn_row_bf16_prepared_deriv_quant() -> bool:
    """Use the v4 row-fused BF16 derivative producer before col-only quant."""
    return os.environ.get('USE_TK_LOCALCTA_V4_FFN_ROW_BF16_PREPARED_DERIV_QUANT', '0') == '1'


def use_tk_localcta_v4_ffn_w2_weight_quant_overlap() -> bool:
    """Overlap localCTA-v4 FFN W2 weight quant with W13 producer work."""
    return _env_flag("USE_TK_LOCALCTA_V4_FFN_W2_WEIGHT_QUANT_OVERLAP", False)


def use_tk_localcta_v4_ffn_cpp_only() -> bool:
    """Disable the removed strict/native v4 FFN route.

    The old `USE_TK_LOCALCTA_V4_FFN_CPP_ONLY=1` setting used to force FFN into
    the strict split2 formulation. That path is slower than the fast outer-SG
    FFN route, so ignore the legacy flag for FFN.
    """
    return False


def use_te_ffn_bwd_safe_producer() -> bool:
    """Use the safer TE FFN backward producer by default.

    This mirrors the stable plain-TK FFN recovery: compute BF16 dh1/dh3 first,
    then quantize each branch with the standard FP4 quantizer before GEMMs.
    """
    value = os.environ.get('USE_TE_FFN_BWD_SAFE_PRODUCER')
    if value is not None:
        return value == '1'
    return True


def use_te_ffn_fwd_safe_producer() -> bool:
    """Use the safer TE FFN forward producer by default.

    This computes `silu(h1_raw) * h3` in BF16 first, then quantizes the result
    with the standard FP4 quantizer before the W2 GEMM.
    """
    value = os.environ.get('USE_TE_FFN_FWD_SAFE_PRODUCER')
    if value is not None:
        return value == '1'
    return True


def use_te_ffn_safe_input_quant() -> bool:
    """Use the stable RMSNorm-only + standard quant path for TE FFN input.

    The older fused_rmsnorm_quantize + manual NVFP4Tensor path has shown
    step-to-step trainer instability on the real 1B_legacy wiki run even when
    the BF16 input and BF16 weights remain sane. Default to the same contract
    already used by the stable TK/localCTA FFN paths, while keeping an env
    override for A/B comparison.
    """
    value = os.environ.get('USE_TE_FFN_SAFE_INPUT_QUANT')
    if value is not None:
        return value == '1'
    return True


def use_te_ffn_safe_rmsnorm() -> bool:
    """Use a reference RMSNorm implementation for TE FFN by default.

    On the real short trainer run, TE FFN can produce corrupt RMSNorm outputs
    at the first layer of step 2 even when the BF16 input and BF16 weights are
    still sane. The stable fallback is to compute RMSNorm in explicit PyTorch
    math and keep the rest of the TE FP4 path unchanged.
    """
    value = os.environ.get('USE_TE_FFN_SAFE_RMSNORM')
    if value is not None:
        return value == '1'
    return True


def use_tk_localcta_ffn_direct_split2() -> bool:
    """Gate the localCTA FFN direct split2 producer.

    Restore the fast plain-localCTA split2 backend by default for debug/perf
    comparisons. It is numerically weaker, but it is the path that produced
    the historical ~5.6ms FFN reference on the isolated 1B_legacy gate; keep
    the explicit opt-out for numerics archaeology.
    """
    if os.environ.get('USE_TK_LOCALCTA_FFN_DISABLE_DIRECT_SPLIT2', '0') == '1':
        return False
    value = os.environ.get('USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2')
    if value is not None:
        return value == '1'
    return use_tk_localcta()


def use_tk_localcta_v3_prepared_split2() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V3_ENABLE_PREPARED_SPLIT2', '0') == '1'

def use_tk_localcta_v4_prepared_split2_dgrad_weights() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_ENABLE_PREPARED_SPLIT2_DGRAD_WEIGHTS', '0') == '1'


def use_tk_localcta_v4_tk_ffn_dgrad_weights() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_USE_TK_FFN_DGRAD_WEIGHTS', '0') == '1'


def use_tk_localcta_v4_tk_ffn_dgrad_acts() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_USE_TK_FFN_DGRAD_ACTS', '0') == '1'


def use_tk_ffn_debug_finite() -> bool:
    """Debug-only finite checks for FFN forward/backward stage boundaries."""
    return os.environ.get('USE_TK_FFN_DEBUG_FINITE', '0') == '1'


def use_tk_ffn_sync_fwd() -> bool:
    """Debug-only synchronization for the FFN forward output boundary."""
    return os.environ.get('USE_TK_FFN_SYNC_FWD', '0') == '1'


def use_tk_attn_debug_finite() -> bool:
    """Debug-only finite checks for attention forward/backward stage boundaries."""
    return os.environ.get('USE_TK_ATTN_DEBUG_FINITE', '0') == '1'


def use_tk_attn_sync_qkv_fwd(debug_name: str | None = None) -> bool:
    if os.environ.get('USE_TK_ATTN_SYNC_QKV_FWD', '0') != '1':
        return False
    name_filter = os.environ.get('USE_TK_ATTN_SYNC_QKV_FWD_FILTER', '').strip()
    return not name_filter or (
        isinstance(debug_name, str) and name_filter in debug_name
    )


def use_tk_attn_sync_before_qkv_fwd(debug_name: str | None = None) -> bool:
    if os.environ.get('USE_TK_ATTN_SYNC_BEFORE_QKV_FWD', '0') != '1':
        return False
    name_filter = os.environ.get(
        'USE_TK_ATTN_SYNC_BEFORE_QKV_FWD_FILTER', ''
    ).strip()
    return not name_filter or (
        isinstance(debug_name, str) and name_filter in debug_name
    )


def use_tk_attn_safe_qkv_fwd_sync() -> bool:
    """Serialize TK/localCTA QKV forward completion for debug-only isolation."""
    value = os.environ.get('USE_TK_ATTN_SAFE_QKV_FWD_SYNC')
    if value is not None:
        return value == '1'
    return False


def use_tk_wo_bf16_wgrad() -> bool:
    """Use a debug-only BF16 WO wgrad path."""
    return os.environ.get("USE_TK_WO_BF16_WGRAD", "0") == "1"


def use_tk_attn_sync_wo_fwd() -> bool:
    return os.environ.get('USE_TK_ATTN_SYNC_WO_FWD', '0') == '1'


def use_tk_attn_sync_wo_bwd() -> bool:
    return os.environ.get('USE_TK_ATTN_SYNC_WO_BWD', '0') == '1'


def use_tk_wo_rowonly_input_quant() -> bool:
    """Use TE row-only swizzled input quant for TK WO forward in debug mode."""
    value = os.environ.get('USE_TK_WO_ROWONLY_INPUT_QUANT')
    if value is not None:
        return value == '1'
    return False


_TK_REGULAR_NHSD_WO_QUANT_AVAILABLE: Optional[bool] = None


def _tk_regular_nhsd_wo_quant_available() -> bool:
    global _TK_REGULAR_NHSD_WO_QUANT_AVAILABLE
    if _TK_REGULAR_NHSD_WO_QUANT_AVAILABLE is None:
        try:
            _TK_REGULAR_NHSD_WO_QUANT_AVAILABLE = hasattr(
                _get_tk_quant(), "tk_quantize_nhsd_wo_for_gemm"
            )
        except Exception:
            _TK_REGULAR_NHSD_WO_QUANT_AVAILABLE = False
    return _TK_REGULAR_NHSD_WO_QUANT_AVAILABLE


def use_tk_regular_wo_nhsd_quant() -> bool:
    """Use regular-TK v5 quantizer that consumes attention output in NHSD layout."""
    value = os.environ.get('USE_TK_WO_NHSD_QUANT')
    if value is not None:
        enabled = value.strip().lower() in {'1', 'true', 'yes', 'on'}
    else:
        enabled = True
    return (
        enabled
        and use_tk_quant()
        and not use_tk_localcta()
        and not _nvfp4_quantizer_extras_enabled("activation")
        and _tk_regular_nhsd_wo_quant_available()
    )


def use_tk_wo_attn_layout() -> bool:
    """Consume SDPA's NHSD output directly when the active WO producer supports it."""
    value = os.environ.get('USE_TK_WO_ATTN_LAYOUT')
    if value is not None:
        return value == '1'
    if use_tk_localcta_v4_strict_path():
        return False
    if (
        use_tk_localcta()
        and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
    ):
        return True
    if use_tk_regular_wo_nhsd_quant():
        return True
    return False


def use_tk_localcta_v4_fast_wo_dgrad() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_WO_DGRAD', '0') == '1'


def use_tk_localcta_v4_fast_wo_wgrad() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_WO_WGRAD', '0') == '1'


def use_tk_localcta_v4_wo_rht_weight_quant_overlap() -> bool:
    """Overlap deterministic WO weight quant with its RHT activation producer.

    This is an explicit, default-off performance candidate.  The weight path
    must remain free of SR/RHT extras so moving it to the established weight
    stream cannot change a checkpointed RNG producer's ordering.
    """
    return _env_flag("USE_TK_LOCALCTA_V4_WO_RHT_WEIGHT_QUANT_OVERLAP", False)


def use_tk_localcta_v4_wo_attn_layout() -> bool:
    value = os.environ.get('USE_TK_LOCALCTA_V4_WO_ATTN_LAYOUT')
    if value is not None:
        return value == '1'
    return use_tk_wo_attn_layout()


def use_tk_localcta_v4_wo_attn_layout_strided_dx() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_V4_WO_ATTN_LAYOUT_STRIDED_DX', '1') == '1'


def _nhsd_attention_output_matrix_view(
    input: torch.Tensor,
    B: int,
    H: int,
    S: int,
    D: int,
) -> torch.Tensor | None:
    """View a [B,H,S,D] transposed attention output as [B*S,H*D].

    cuDNN SDPA returns the logical NHSD tensor backed by BSHD-contiguous
    storage. For WO this is exactly the desired row-major matrix layout, so
    materializing NHSD first is pure copy overhead.
    """
    if input.storage_offset() != 0:
        return None
    expected = (H * S * D, D, H * D, 1)
    if tuple(input.stride()) != expected:
        return None
    return input.as_strided((B * S, H * D), (H * D, 1))


def use_tk_localcta_v4_fast_w2_wgrad() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_W2_WGRAD', '0') == '1'


def use_tk_localcta_v4_fast_ffn_fused_norm() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_FFN_FUSED_NORM', '0') == '1'


def use_tk_localcta_v4_fast_ffn_rmsnorm_quant() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_FAST_FFN_RMSNORM_QUANT', '0') == '1'


def use_tk_localcta_v4_ffn_separate_bf16_final_sg() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get(
        'USE_TK_LOCALCTA_V4_FFN_SEPARATE_BF16_FINAL_SG', '0'
    ) == '1'


def use_tk_ffn_split_weight_quant() -> bool:
    """Use the FFN split grouped-weight quant fast path."""
    value = os.environ.get('USE_TK_FFN_SPLIT_WEIGHT_QUANT')
    if value is not None:
        return value == '1'
    # Regular TK intermittently livelocks in the split-weight producer during
    # trainer runs. Keep the v4/localCTA path on its tuned split producer by
    # default, and leave regular TK opt-in until that producer is fixed.
    return use_tk_localcta()


def use_tk_ffn_split_weight_fast() -> bool:
    value = os.environ.get('USE_TK_FFN_SPLIT_WEIGHT_FAST')
    if value is not None:
        return value == '1'
    return use_nvfp4_mxfp4_live_path()


def use_tk_ffn_weight_quant_v2() -> bool:
    """Use the stable v2 FFN grouped-weight quantizer for regular TK."""
    value = os.environ.get('USE_TK_FFN_WEIGHT_QUANT_V2')
    if value is not None:
        return value == '1'
    return os.environ.get('USE_TK_GEMM', '0') == '1' and not use_tk_localcta()


def use_tk_ffn_decomposed_weight_quant() -> bool:
    """Avoid the regular TK grouped FFN weight producer that can livelock."""
    value = os.environ.get('USE_TK_FFN_DECOMPOSED_WEIGHT_QUANT')
    if value is not None:
        return value == '1'
    return os.environ.get('USE_TK_GEMM', '0') == '1' and not use_tk_localcta()


def use_tk_ffn_debug_sync_check() -> bool:
    """Force CUDA sync checkpoints around TK FFN backward stages."""
    return os.environ.get('USE_TK_FFN_DEBUG_SYNC_CHECK', '0') == '1'


def _tk_ffn_debug_sync_checkpoint(label: str) -> None:
    if not use_tk_ffn_debug_sync_check():
        return
    labels = os.environ.get('USE_TK_FFN_DEBUG_SYNC_LABELS', '').strip()
    if labels:
        allowed = {item.strip() for item in labels.split(',') if item.strip()}
        if label not in allowed:
            return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"TK FFN debug sync failed after {label}: {exc}") from exc


def use_tk_attn_sync_qkv_bwd() -> bool:
    return os.environ.get('USE_TK_ATTN_SYNC_QKV_BWD', '0') == '1'


def use_tk_stage_trace() -> bool:
    return os.environ.get('USE_TK_STAGE_TRACE', '0') == '1'


_tk_stage_trace_starts: dict[tuple[str, str, str], float] = {}


def _tk_stage_trace(stage: str, event: str, name: Optional[str]) -> None:
    if not use_tk_stage_trace():
        return
    active_step = os.environ.get('LBT_TRACE_ACTIVE_STEP', '').strip()
    step_filter = os.environ.get('TK_STAGE_TRACE_STEP', '').strip()
    if step_filter and active_step != step_filter:
        return
    stage_filter = os.environ.get('TK_STAGE_TRACE_STAGE_FILTER', '').strip()
    if stage_filter and stage_filter not in stage:
        return
    event_filter = os.environ.get('TK_STAGE_TRACE_EVENT_FILTER', '').strip()
    if event_filter and event_filter not in event:
        return
    label = name or stage
    name_filter = os.environ.get('TK_STAGE_TRACE_FILTER', '').strip()
    if name_filter and name_filter not in label:
        return
    if os.environ.get('USE_TK_STAGE_TRACE_SYNC', '0') == '1' and torch.cuda.is_available():
        torch.cuda.synchronize()
    prefix = f"[TK TRACE step={active_step}]" if active_step else "[TK TRACE]"
    timer_key = None
    if event == 'start':
        timer_key = (active_step, stage, label)
        _tk_stage_trace_starts[timer_key] = time.perf_counter()
    elif event.endswith('_start'):
        timer_key = (active_step, stage, f"{label}:{event[:-6]}")
        _tk_stage_trace_starts[timer_key] = time.perf_counter()

    elapsed = None
    if event == 'end':
        timer_key = (active_step, stage, label)
        start = _tk_stage_trace_starts.pop(timer_key, None)
        if start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0
    elif event.endswith('_done'):
        timer_key = (active_step, stage, f"{label}:{event[:-5]}")
        start = _tk_stage_trace_starts.pop(timer_key, None)
        if start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0

    suffix = f" elapsed_ms={elapsed:.3f}" if elapsed is not None else ""
    print(f"{prefix} {stage} {event} {label}{suffix}", file=sys.stderr, flush=True)


def use_tk_debug_prints() -> bool:
    return os.environ.get('USE_TK_DEBUG_PRINTS', '0') == '1'


def _tk_debug_print(stage: str, event: str, name: Optional[str]) -> None:
    if not use_tk_debug_prints():
        return
    label = name or stage
    active_step = os.environ.get('LBT_TRACE_ACTIVE_STEP', '').strip()
    prefix = f"[TK DEBUG step={active_step}]" if active_step else "[TK DEBUG]"
    print(f"{prefix} {stage} {event} {label}", file=sys.stderr, flush=True)


_te_ffn_fwd_debug_call_idx = 0
_te_ffn_bwd_debug_call_idx = 0
_tk_attn_qkv_fwd_debug_call_idx = 0
_tk_attn_wo_fwd_debug_call_idx = 0
_tk_attn_capture_count = 0
_tk_ffn_capture_count = 0
_TK_QKV_LIVE64_ROPE_CACHE: dict[tuple[int, int, torch.dtype, str, int | None], torch.Tensor] = {}
_TK_QKV_PACKED_ROPE_CACHE: dict[tuple[int, int, torch.dtype, str, int | None], torch.Tensor] = {}
_TK_QKV_ROPE_TABLE_CACHE: dict[tuple[int, int, torch.dtype, str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
_TK_QKV_TE_ROPE_CACHE: dict[tuple[int, int, torch.dtype, str, int | None], torch.Tensor] = {}
_TK_QKV_PACKED_INVERSE_GRAPH_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_TK_QKV_PACKED_V_GRAPH_CACHE: dict[tuple, torch.Tensor] = {}
_TK_QKV_FORWARD_GRAPH_KEEPALIVE: list[tuple[object, ...]] = []
_TK_QKV_DEBUG_RETAINED_INCOMING_GRADS: list[tuple[torch.Tensor, ...]] = []


def clear_tk_qkv_packed_graph_caches() -> None:
    """Maintenance-only release after all referencing CUDA graphs are dead."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("cannot clear packed QKV graph state during capture")
    _TK_QKV_PACKED_INVERSE_GRAPH_CACHE.clear()
    _TK_QKV_PACKED_V_GRAPH_CACHE.clear()
    _TK_QKV_FORWARD_GRAPH_KEEPALIVE.clear()
    from .tk_gemm import clear_tk_qkv_split3_graph_cache

    clear_tk_qkv_split3_graph_cache()
    clear_tk_qkv_persistent_weight_quant_state()


def _retain_tk_qkv_forward_graph_state(*values: object) -> None:
    """Keep internal forward buffers alive for every captured graph replay."""
    if torch.cuda.is_current_stream_capturing():
        _TK_QKV_FORWARD_GRAPH_KEEPALIVE.append(tuple(values))


def _is_power_of_two_int(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _get_tk_live64_rope_cs(freqs_cis: torch.Tensor, seq_len: int) -> torch.Tensor:
    key = (
        freqs_cis.data_ptr(),
        int(seq_len),
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _TK_QKV_LIVE64_ROPE_CACHE.get(key)
    if cached is None:
        if len(_TK_QKV_LIVE64_ROPE_CACHE) >= 8:
            _TK_QKV_LIVE64_ROPE_CACHE.clear()
        freqs_slice = freqs_cis[:seq_len]
        cached = torch.stack((freqs_slice.real[:, :32], freqs_slice.imag[:, :32]), dim=-1).contiguous()
        _TK_QKV_LIVE64_ROPE_CACHE[key] = cached
    return cached


def _get_tk_packed_rope_cs(freqs_cis: torch.Tensor, seq_len: int) -> torch.Tensor:
    key = (
        freqs_cis.data_ptr(),
        int(seq_len),
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _TK_QKV_PACKED_ROPE_CACHE.get(key)
    if cached is None:
        if len(_TK_QKV_PACKED_ROPE_CACHE) >= 8:
            _TK_QKV_PACKED_ROPE_CACHE.clear()
        freqs_slice = freqs_cis[:seq_len]
        cached = torch.stack((freqs_slice.real, freqs_slice.imag), dim=-1).contiguous()
        _TK_QKV_PACKED_ROPE_CACHE[key] = cached
    return cached


def _get_tk_rope_tables(freqs_cis: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        freqs_cis.data_ptr(),
        int(seq_len),
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _TK_QKV_ROPE_TABLE_CACHE.get(key)
    if cached is None:
        if len(_TK_QKV_ROPE_TABLE_CACHE) >= 8:
            _TK_QKV_ROPE_TABLE_CACHE.clear()
        freqs_slice = freqs_cis[:seq_len]
        cached = (freqs_slice.real.contiguous(), freqs_slice.imag.contiguous())
        _TK_QKV_ROPE_TABLE_CACHE[key] = cached
    return cached


def _get_tk_te_rope_freqs(freqs_cis: torch.Tensor, seq_len: int) -> torch.Tensor:
    key = (
        freqs_cis.data_ptr(),
        int(seq_len),
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _TK_QKV_TE_ROPE_CACHE.get(key)
    if cached is None:
        if len(_TK_QKV_TE_ROPE_CACHE) >= 8:
            _TK_QKV_TE_ROPE_CACHE.clear()
        angles = torch.angle(freqs_cis[:seq_len])
        cached = (
            torch.stack((angles, angles), dim=-1)
            .flatten(-2)
            .view(seq_len, 1, 1, -1)
            .contiguous()
        )
        _TK_QKV_TE_ROPE_CACHE[key] = cached
    return cached


def _apply_inverse_tk_rotary_qk(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    freqs_cis: torch.Tensor,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    packed_selected: bool = False,
    owner_key=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if packed_selected:
        from .tk_gemm import _get_tk_plain
        tk_mod = _get_tk_plain()
        inverse_packed = getattr(tk_mod, 'nvfp4_inverse_rope_packed_qk', None)
        if inverse_packed is None:
            raise RuntimeError("packed QKV RoPE selected but native inverse symbol is unavailable")
        if (
            grad_q.dim() != 2
            or grad_k.dim() != 2
            or not grad_q.is_cuda
            or not grad_k.is_cuda
            or grad_q.dtype != torch.bfloat16
            or grad_k.dtype != torch.bfloat16
            or not grad_q.is_contiguous()
            or not grad_k.is_contiguous()
            or head_dim != 128
            or seq_len <= 0
            or grad_q.size(0) % seq_len != 0
            or freqs_cis.dim() != 2
            or int(freqs_cis.size(1)) * 2 != head_dim
        ):
            raise RuntimeError("packed QKV RoPE native inverse contract is unsupported")
        rope_cs = _get_tk_packed_rope_cs(freqs_cis, seq_len)
        if use_cuda_graph():
            stream_id = int(torch.cuda.current_stream(grad_q.device).cuda_stream)
            graph_key = (
                owner_key if owner_key is not None else "__shared__",
                grad_q.device.index,
                stream_id,
                tuple(grad_q.shape),
                tuple(grad_k.shape),
                grad_q.dtype,
                grad_k.dtype,
            )
            graph_outputs = _TK_QKV_PACKED_INVERSE_GRAPH_CACHE.get(graph_key)
            if graph_outputs is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "packed QKV inverse graph scratch was not primed on the "
                        "capture stream; run an eager full-module warmup first"
                    )
                graph_outputs = (torch.empty_like(grad_q), torch.empty_like(grad_k))
                _TK_QKV_PACKED_INVERSE_GRAPH_CACHE[graph_key] = graph_outputs
            grad_q_out, grad_k_out = graph_outputs
        else:
            grad_q_out = torch.empty_like(grad_q)
            grad_k_out = torch.empty_like(grad_k)
        inverse_packed(
            grad_q, grad_k, rope_cs, int(seq_len), int(head_dim),
            grad_q_out, grad_k_out,
        )
        _trace_backend_choice('regular_tk_qkv_bwd_rope', 'inverse_packed_native')
        return grad_q_out, grad_k_out

    gq_4d = grad_q.view(batch_size, seq_len, -1, head_dim)
    gk_4d = grad_k.view(batch_size, seq_len, -1, head_dim)

    if te_apply_rotary_pos_emb is not None and grad_q.is_cuda and grad_k.is_cuda and freqs_cis.is_cuda:
        rope_freqs = _get_tk_te_rope_freqs(freqs_cis, seq_len)
        gq_4d = te_apply_rotary_pos_emb(
            gq_4d,
            -rope_freqs,
            tensor_format="bshd",
            fused=True,
            interleaved=True,
        )
        gk_4d = te_apply_rotary_pos_emb(
            gk_4d,
            -rope_freqs,
            tensor_format="bshd",
            fused=True,
            interleaved=True,
        )
        return gq_4d.reshape_as(grad_q), gk_4d.reshape_as(grad_k)

    from torchtitan.models.llama3.model.model import apply_rotary_emb

    gq_4d, gk_4d = apply_rotary_emb(gq_4d, gk_4d, freqs_cis=freqs_cis[:seq_len].conj())
    return gq_4d.reshape_as(grad_q), gk_4d.reshape_as(grad_k)


def _stable_packed_graph_grad_v(
    grad_v: torch.Tensor,
    owner_key=None,
) -> torch.Tensor:
    """Stage packed-route V grad at a descriptor-stable graph address."""
    if not use_cuda_graph():
        return grad_v
    stream_id = int(torch.cuda.current_stream(grad_v.device).cuda_stream)
    key = (
        owner_key if owner_key is not None else "__shared__",
        grad_v.device.index,
        stream_id,
        tuple(grad_v.shape),
        grad_v.dtype,
    )
    stable = _TK_QKV_PACKED_V_GRAPH_CACHE.get(key)
    if stable is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "packed QKV V-grad graph scratch was not primed on the capture "
                "stream; run an eager full-module warmup first"
            )
        stable = torch.empty_like(grad_v)
        _TK_QKV_PACKED_V_GRAPH_CACHE[key] = stable
    stable.copy_(grad_v)
    return stable


def _tk_qkv_rope_live64_supported(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    seq_len: int,
    freqs_cis: torch.Tensor | None,
) -> bool:
    if not use_tk_qkv_rope_epilogue():
        return False
    if freqs_cis is None or not freqs_cis.is_cuda or not torch.is_complex(freqs_cis):
        return False
    return (
        M > 0
        and K > 0
        and seq_len > 0
        and M % seq_len == 0
        and head_dim == 64
        and _is_power_of_two_int(int(seq_len))
        and q_dim % 64 == 0
        and k_dim % 64 == 0
        and v_dim % 128 == 0
        and freqs_cis.size(0) >= seq_len
        and freqs_cis.size(1) >= 32
    )


def _tk_qkv_rope_generic_supported(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    seq_len: int,
    freqs_cis: torch.Tensor | None,
) -> bool:
    if not use_tk_qkv_rope_epilogue():
        return False
    if os.environ.get("USE_TK_QKV_GENERIC_ROPE_EPILOGUE", "0") != "1":
        return False
    if freqs_cis is None or not freqs_cis.is_cuda or not torch.is_complex(freqs_cis):
        return False
    rotary_dim = int(freqs_cis.size(1)) * 2 if freqs_cis.dim() >= 2 else 0
    return (
        M > 0
        and K > 0
        and seq_len > 0
        and M % seq_len == 0
        and head_dim > 0
        and rotary_dim > 0
        and rotary_dim <= head_dim
        and (rotary_dim % 2) == 0
        and q_dim % head_dim == 0
        and k_dim % head_dim == 0
        and v_dim % 128 == 0
        and freqs_cis.size(0) >= seq_len
    )


def _use_tk_qkv_packed_rope_policy(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    seq_len: int,
) -> bool:
    if use_tk_localcta():
        return os.environ.get("USE_TK_LOCALCTA_V4_NATIVE_QK_ROPE", "0") == "1"
    policy = os.environ.get("USE_TK_QKV_PACKED_ROPE_EPILOGUE")
    if policy is not None:
        return policy == "1"
    return (
        M == 32768
        and K == 4096
        and q_dim == 4096
        and k_dim == 1024
        and v_dim == 1024
        and head_dim == 128
        and seq_len == 8192
    )


def _tk_qkv_rope_packed_supported(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    seq_len: int,
    freqs_cis: torch.Tensor | None,
) -> bool:
    if not _use_tk_qkv_packed_rope_policy(
        M, K, q_dim, k_dim, v_dim, head_dim, seq_len
    ):
        return False
    if freqs_cis is None or not freqs_cis.is_cuda or not torch.is_complex(freqs_cis):
        return False
    rotary_dim = int(freqs_cis.size(1)) * 2 if freqs_cis.dim() == 2 else 0
    return (
        M > 0
        and K > 0
        and M % 256 == 0
        and K % 256 == 0
        and seq_len > 0
        and M % seq_len == 0
        and head_dim == 128
        and rotary_dim == 128
        and _is_power_of_two_int(int(seq_len))
        and q_dim > 0
        and k_dim > 0
        and v_dim > 0
        and q_dim % 256 == 0
        and k_dim % 256 == 0
        and v_dim % 256 == 0
        and freqs_cis.size(0) >= seq_len
    )


def _tk_qkv_rope_supported(
    M: int,
    K: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    seq_len: int,
    freqs_cis: torch.Tensor | None,
) -> bool:
    return _tk_qkv_rope_live64_supported(
        M, K, q_dim, k_dim, v_dim, head_dim, seq_len, freqs_cis
    ) or _tk_qkv_rope_packed_supported(
        M, K, q_dim, k_dim, v_dim, head_dim, seq_len, freqs_cis
    ) or _tk_qkv_rope_generic_supported(
        M, K, q_dim, k_dim, v_dim, head_dim, seq_len, freqs_cis
    )


def _tk_qkv_rope_backend_available() -> bool:
    try:
        from .tk_gemm import _get_tk_localcta_direct, _get_tk_plain, get_tk_localcta_variant
        if use_tk_localcta():
            if get_tk_localcta_variant() != 'v4':
                return False
            direct = _get_tk_localcta_direct()
            return direct is not None and (
                hasattr(direct, 'nvfp4_grouped_gemm_rope_live64')
                or hasattr(direct, 'nvfp4_grouped_gemm_rope')
            )
        tk_mod = _get_tk_plain()
        return tk_mod is not None and (
            hasattr(tk_mod, 'nvfp4_grouped_gemm_rope_live64')
            or hasattr(tk_mod, 'nvfp4_grouped_gemm_rope_packed_cat')
            or hasattr(tk_mod, 'nvfp4_grouped_gemm_rope_packed_split')
            or hasattr(tk_mod, 'nvfp4_grouped_gemm_rope')
        )
    except Exception:
        return False


def _tk_qkv_rope_packed_backend_available() -> bool:
    if use_tk_localcta():
        return False
    try:
        from .tk_gemm import _get_tk_plain
        tk_mod = _get_tk_plain()
        return tk_mod is not None and hasattr(
            tk_mod, 'nvfp4_grouped_gemm_rope_packed_split'
        )
    except Exception:
        return False


def _attn_capture_path() -> str:
    return os.environ.get("USE_TK_DEBUG_ATTN_CAPTURE_JSONL", "")


def _attn_layout_path() -> str:
    return os.environ.get("USE_TK_DEBUG_ATTN_LAYOUT_JSONL", "")


def _attn_debug_name_matches(debug_name: str | None, filter_env: str) -> bool:
    name_filter = os.environ.get(filter_env, "").strip()
    return not name_filter or (
        isinstance(debug_name, str) and name_filter in debug_name
    )


def _attn_layout_event_enabled(event: str, debug_name: str | None) -> bool:
    if not _attn_layout_path():
        return False
    event_filter = {
        value.strip()
        for value in os.environ.get(
            "USE_TK_DEBUG_ATTN_LAYOUT_EVENTS", ""
        ).split(",")
        if value.strip()
    }
    return (
        (not event_filter or event in event_filter)
        and _attn_debug_name_matches(
            debug_name, "USE_TK_DEBUG_ATTN_LAYOUT_FILTER"
        )
    )


def _tensor_layout_metadata(t: torch.Tensor | None) -> dict | None:
    """Return CUDA tensor metadata without launching work or synchronizing."""
    if t is None:
        return None
    if not torch.is_tensor(t):
        return {"type": str(type(t))}

    storage = t.untyped_storage()
    base = getattr(t, "_base", None)
    metadata = {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "storage_offset": int(t.storage_offset()),
        "data_ptr": int(t.data_ptr()),
        "storage_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "numel": int(t.numel()),
        "element_size": int(t.element_size()),
        "is_contiguous": bool(t.is_contiguous()),
        "requires_grad": bool(t.requires_grad),
        "version": int(t._version),
    }
    if torch.is_tensor(base):
        metadata["base_data_ptr"] = int(base.data_ptr())
        metadata["base_storage_ptr"] = int(base.untyped_storage().data_ptr())
        metadata["base_shape"] = list(base.shape)
        metadata["base_stride"] = list(base.stride())
    if t.is_cuda:
        stream = torch.cuda.current_stream(t.device)
        metadata["cuda_stream"] = int(stream.cuda_stream)
        metadata["cuda_stream_device"] = int(stream.device_index)
    return metadata


def _tensor_layout_group(named_tensors) -> dict:
    if os.environ.get("USE_TK_DEBUG_ATTN_LAYOUT_NO_TENSORS", "0") == "1":
        return {"tensors": {}, "storage_aliases": []}
    tensor_filter = {
        name.strip()
        for name in os.environ.get(
            "USE_TK_DEBUG_ATTN_LAYOUT_TENSORS", ""
        ).split(",")
        if name.strip()
    }
    if tensor_filter:
        named_tensors = tuple(
            (name, tensor)
            for name, tensor in named_tensors
            if name in tensor_filter
        )
    tensors = {
        name: _tensor_layout_metadata(tensor)
        for name, tensor in named_tensors
    }
    aliases = []
    materialized = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_tensor(tensor)
    ]
    for index, (left_name, left) in enumerate(materialized):
        left_storage = int(left.untyped_storage().data_ptr())
        for right_name, right in materialized[index + 1:]:
            if left_storage == int(right.untyped_storage().data_ptr()):
                aliases.append([left_name, right_name])
    return {"tensors": tensors, "storage_aliases": aliases}


def _append_attn_layout(payload: dict) -> None:
    """Append metadata-only attention records; never inspect tensor values."""
    path = _attn_layout_path()
    if not path:
        return
    debug_name = payload.get("debug_name")
    if not _attn_layout_event_enabled(payload.get("event", ""), debug_name):
        return
    record = dict(payload)
    record["rank"] = int(
        os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    )
    if os.environ.get("USE_TK_DEBUG_ATTN_LAYOUT_BUFFERED", "0") == "1":
        records = getattr(_append_attn_layout, "_buffered_records", None)
        if records is None:
            import atexit

            records = []
            setattr(_append_attn_layout, "_buffered_records", records)

            def _flush_buffered_records() -> None:
                if not records:
                    return
                rank = int(
                    os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
                )
                rank_path = f"{path}.rank{rank}.jsonl"
                with open(rank_path, "w", encoding="utf-8") as f:
                    for buffered_record in records:
                        f.write(
                            json.dumps(buffered_record, sort_keys=True) + "\n"
                        )

            atexit.register(_flush_buffered_records)
        records.append(record)
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _ffn_capture_path() -> str:
    return os.environ.get("USE_TK_DEBUG_FFN_CAPTURE_JSONL", "")


def _ffn_dump_dir() -> str:
    return os.environ.get("USE_TK_DEBUG_FFN_DUMP_DIR", "")


def _ffn_dump_match() -> str:
    return os.environ.get("USE_TK_DEBUG_FFN_DUMP_MATCH", "")


def _should_dump_ffn(debug_name: str | None, tag: str) -> bool:
    dump_dir = _ffn_dump_dir()
    if not dump_dir:
        return False
    match = _ffn_dump_match()
    if match and (debug_name is None or match not in debug_name):
        return False
    once_key = f"{debug_name or 'unknown'}::{tag}"
    if os.environ.get("USE_TK_DEBUG_FFN_DUMP_ONCE", "1") != "0":
        dumped = getattr(_should_dump_ffn, "_dumped", None)
        if dumped is None:
            dumped = set()
            setattr(_should_dump_ffn, "_dumped", dumped)
        if once_key in dumped:
            return False
        dumped.add(once_key)
    return True


def _clone_dump_value(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return [_clone_dump_value(v) for v in x]
    if isinstance(x, dict):
        return {k: _clone_dump_value(v) for k, v in x.items()}
    return x


def _should_clone_autograd_return(m: int | None) -> bool:
    if os.environ.get("USE_TK_DISABLE_AUTOGRAD_RETURN_CLONES", "1") == "0":
        return True
    if m is None:
        return False
    return m < int(os.environ.get("USE_TK_AUTOGRAD_RETURN_CLONE_MAX_M", "256"))


def _maybe_clone_autograd_return(x: torch.Tensor, m: int | None) -> torch.Tensor:
    return x.clone() if _should_clone_autograd_return(m) else x


def _maybe_own_qkv_grad_input(x: torch.Tensor, m: int | None) -> torch.Tensor:
    """Give autograd unique QKV input-gradient storage when requested.

    The fused RMS backward writes ``grad_input`` into shape-cached scratch.  A
    large-shape QKV backward normally bypasses the generic return clone, so the
    next layer or accumulation microbatch can overwrite storage still reachable
    through the residual graph.  This switch isolates that ownership boundary
    without cloning the much larger set of fused backward returns.
    """
    if os.environ.get("USE_TK_QKV_OWNED_GRAD_INPUT", "0") == "1":
        return x.clone(memory_format=torch.contiguous_format)
    return _maybe_clone_autograd_return(x, m)


def _maybe_clone_localcta_v4_ffn_return(x: torch.Tensor, m: int | None) -> torch.Tensor:
    if _clone_localcta_v4_ffn_backward_returns():
        return x.clone()
    return _maybe_clone_autograd_return(x, m)


def _maybe_clone_localcta_v4_ffn_grad_input(x: torch.Tensor, m: int | None) -> torch.Tensor:
    if _env_flag("USE_TK_LOCALCTA_V4_CLONE_FFN_BWD_GRAD_INPUT", False):
        return x.clone()
    return _maybe_clone_localcta_v4_ffn_return(x, m)


def _clone_localcta_v4_ffn_backward_returns() -> bool:
    value = os.environ.get("USE_TK_LOCALCTA_V4_CLONE_FFN_BWD_RETURNS")
    if value is not None:
        return value == "1"
    return False


def _dump_ffn_tensors(tag: str, debug_name: str | None, payload: dict) -> None:
    if not _should_dump_ffn(debug_name, tag):
        return
    dump_dir = _ffn_dump_dir()
    try:
        os.makedirs(dump_dir, exist_ok=True)
        safe_name = (debug_name or "unknown").replace("/", "_").replace(":", "_").replace(".", "_")
        path = os.path.join(dump_dir, f"{safe_name}_{tag}.pt")
        torch.save(_clone_dump_value(payload), path)
    except Exception:
        logger.exception("Failed to dump FFN tensors for %s (%s)", debug_name, tag)


def _tensor_capture_stats(t: torch.Tensor | None) -> dict | None:
    if t is None:
        return None
    if not torch.is_tensor(t):
        return {"type": str(type(t))}
    x = t.detach()
    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "numel": int(x.numel()),
    }
    if x.numel() == 0:
        return out
    if x.dtype == torch.float4_e2m1fn_x2:
        u8 = x.view(torch.uint8)
        low = u8 & 0x0F
        high = u8 >> 4
        zero_count = (
            (low == 0).sum() + (low == 8).sum()
            + (high == 0).sum() + (high == 8).sum()
        )
        max_count = (
            (low == 7).sum() + (low == 15).sum()
            + (high == 7).sum() + (high == 15).sum()
        )
        out.update({
            "byte_zero_fraction": float((u8 == 0).float().mean().item()),
            "byte_nonzero_fraction": float((u8 != 0).float().mean().item()),
            "value_zero_fraction": float(zero_count.item() / (2 * u8.numel())),
            "value_max_fraction": float(max_count.item() / (2 * u8.numel())),
        })
        return out
    if not x.is_floating_point() or x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = x.to(torch.float32)
    finite = torch.isfinite(x)
    x_abs = x.abs()
    out.update({
        "finite_fraction": float(finite.float().mean().item()),
        "mean": float(x.mean().item()),
        "zero_fraction": float((x == 0).float().mean().item()),
        "rms": float(torch.sqrt((x * x).mean()).item()),
        "mean_abs": float(x_abs.mean().item()),
        "max_abs": float(x_abs.max().item()),
    })
    return out


def _append_attn_capture(payload: dict) -> None:
    global _tk_attn_capture_count
    path = _attn_capture_path()
    if not path:
        return
    name_filter = os.environ.get("USE_TK_DEBUG_ATTN_CAPTURE_FILTER", "").strip()
    if name_filter:
        debug_name = payload.get("debug_name")
        if not isinstance(debug_name, str) or name_filter not in debug_name:
            return
    record = dict(payload)
    record["capture_index"] = _tk_attn_capture_count
    _tk_attn_capture_count += 1
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _append_ffn_capture(payload: dict) -> None:
    global _tk_ffn_capture_count
    path = _ffn_capture_path()
    if not path:
        return
    name_filter = os.environ.get("USE_TK_DEBUG_FFN_CAPTURE_FILTER", "").strip()
    if name_filter:
        debug_name = payload.get("debug_name")
        if not isinstance(debug_name, str) or name_filter not in debug_name:
            return
    record = dict(payload)
    record["capture_index"] = _tk_ffn_capture_count
    _tk_ffn_capture_count += 1
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _debug_forward_ref_enabled(debug_name: Optional[str]) -> bool:
    if os.environ.get("USE_TK_DEBUG_FWD_REF", "0") != "1":
        return False
    name_filter = os.environ.get("USE_TK_DEBUG_FWD_REF_FILTER", "").strip()
    if not name_filter:
        return True
    return debug_name is not None and name_filter in debug_name


def _debug_wgrad_ref_enabled(debug_name: Optional[str]) -> bool:
    if os.environ.get("USE_TK_DEBUG_WGRAD_REF", "0") != "1":
        return False
    name_filter = os.environ.get("USE_TK_DEBUG_WGRAD_REF_FILTER", "").strip()
    if not name_filter:
        return True
    return debug_name is not None and name_filter in debug_name


def _debug_dgrad_ref_enabled(debug_name: Optional[str]) -> bool:
    if os.environ.get("USE_TK_DEBUG_DGRAD_REF", "0") != "1":
        return False
    name_filter = os.environ.get("USE_TK_DEBUG_DGRAD_REF_FILTER", "").strip()
    if not name_filter:
        return True
    return debug_name is not None and name_filter in debug_name


def _tensor_delta_stats(actual: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = actual.detach().to(torch.float32) - ref.detach().to(torch.float32)
    return {
        "actual": _tensor_capture_stats(actual),
        "ref": _tensor_capture_stats(ref),
        "diff": _tensor_capture_stats(diff),
    }


def _next_te_ffn_debug_call(kind: str) -> int:
    global _te_ffn_fwd_debug_call_idx, _te_ffn_bwd_debug_call_idx
    if kind == 'fwd':
        _te_ffn_fwd_debug_call_idx += 1
        return _te_ffn_fwd_debug_call_idx
    _te_ffn_bwd_debug_call_idx += 1
    return _te_ffn_bwd_debug_call_idx


def _next_tk_attn_debug_call(kind: str) -> int:
    global _tk_attn_qkv_fwd_debug_call_idx, _tk_attn_wo_fwd_debug_call_idx
    if kind == 'qkv_fwd':
        _tk_attn_qkv_fwd_debug_call_idx += 1
        return _tk_attn_qkv_fwd_debug_call_idx
    _tk_attn_wo_fwd_debug_call_idx += 1
    return _tk_attn_wo_fwd_debug_call_idx


def _debug_log_tensor_stats(name: str, tensor: torch.Tensor | None):
    if not use_tk_ffn_debug_finite() or tensor is None or not torch.is_tensor(tensor):
        return
    if tensor.numel() == 0:
        return

    with torch.no_grad():
        flat = tensor.detach().reshape(-1)
        try:
            finite = torch.isfinite(flat)
            stats_source = flat
        except NotImplementedError:
            stats_source = flat.float()
            finite = torch.isfinite(stats_source)
        finite_count = int(finite.sum().item())
        total_count = flat.numel()
        nan_count = int(torch.isnan(stats_source).sum().item())
        posinf_count = int(torch.isposinf(stats_source).sum().item())
        neginf_count = int(torch.isneginf(stats_source).sum().item())

        max_abs = float('nan')
        mean_abs = float('nan')
        rms = float('nan')
        min_val = float('nan')
        max_val = float('nan')
        if finite_count:
            finite_vals = stats_source[finite].float()
            abs_vals = finite_vals.abs()
            max_abs = float(abs_vals.max().item())
            mean_abs = float(abs_vals.mean().item())
            rms = float(torch.sqrt((finite_vals * finite_vals).mean()).item())
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())

    print(
        f"[FFN DBG] {name} "
        f"finite={finite_count}/{total_count} nan={nan_count} +inf={posinf_count} -inf={neginf_count} "
        f"maxabs={max_abs:.6e} meanabs={mean_abs:.6e} rms={rms:.6e} "
        f"min={min_val:.6e} max={max_val:.6e}",
        flush=True,
    )


def _debug_check_finite(name: str, tensor: torch.Tensor | None):
    if not use_tk_ffn_debug_finite() or tensor is None or not torch.is_tensor(tensor):
        return
    if tensor.numel() == 0:
        return
    _debug_log_tensor_stats(name, tensor)
    try:
        finite = torch.isfinite(tensor)
    except NotImplementedError:
        finite = torch.isfinite(tensor.float())
    if not bool(finite.all().item()):
        raise RuntimeError(f"FFN finite check failed: {name}")


def _tk_ffn_debug_assert_finite(
    stage: str,
    debug_name: str | None,
    named_tensors,
) -> None:
    """Fail at the first non-finite FFN boundary without a full-size mask."""
    if os.environ.get("USE_TK_DEBUG_FFN_FINITE", "0") != "1":
        return
    filtered = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    ]
    if not filtered:
        return

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
            f"Non-finite TK FFN tensor at {owner}:{stage}: " + "; ".join(bad)
        )


def _tk_attn_debug_assert_finite(
    stage: str,
    debug_name: str | None,
    named_tensors,
) -> None:
    """Fail at the first non-finite attention boundary without a full mask."""
    if os.environ.get("USE_TK_DEBUG_ATTN_FINITE", "0") != "1":
        return
    filtered = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    ]
    if not filtered:
        return

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
            f"Non-finite TK attention tensor at {owner}:{stage}: "
            + "; ".join(bad)
        )


def _attn_debug_log_tensor_stats(name: str, tensor: torch.Tensor | None):
    if not use_tk_attn_debug_finite() or tensor is None or not torch.is_tensor(tensor):
        return
    if tensor.numel() == 0:
        return

    with torch.no_grad():
        flat = tensor.detach().reshape(-1)
        try:
            finite = torch.isfinite(flat)
            stats_source = flat
        except NotImplementedError:
            stats_source = flat.float()
            finite = torch.isfinite(stats_source)
        finite_count = int(finite.sum().item())
        total_count = flat.numel()
        nan_count = int(torch.isnan(stats_source).sum().item())
        posinf_count = int(torch.isposinf(stats_source).sum().item())
        neginf_count = int(torch.isneginf(stats_source).sum().item())

        max_abs = float('nan')
        mean_abs = float('nan')
        rms = float('nan')
        min_val = float('nan')
        max_val = float('nan')
        if finite_count:
            finite_vals = stats_source[finite].float()
            abs_vals = finite_vals.abs()
            max_abs = float(abs_vals.max().item())
            mean_abs = float(abs_vals.mean().item())
            rms = float(torch.sqrt((finite_vals * finite_vals).mean()).item())
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())

    print(
        f"[ATTN DBG] {name} "
        f"finite={finite_count}/{total_count} nan={nan_count} +inf={posinf_count} -inf={neginf_count} "
        f"maxabs={max_abs:.6e} meanabs={mean_abs:.6e} rms={rms:.6e} "
        f"min={min_val:.6e} max={max_val:.6e}",
        flush=True,
    )


def _attn_debug_check_finite(name: str, tensor: torch.Tensor | None):
    if not use_tk_attn_debug_finite() or tensor is None or not torch.is_tensor(tensor):
        return
    if tensor.numel() == 0:
        return
    _attn_debug_log_tensor_stats(name, tensor)
    try:
        finite = torch.isfinite(tensor)
    except NotImplementedError:
        finite = torch.isfinite(tensor.float())
    if not bool(finite.all().item()):
        raise RuntimeError(f"Attention finite check failed: {name}")


def _tk_qkv_forward_stage_probe(
    stage: str,
    debug_call_id: int | None,
    debug_name: str | None,
    named_tensors,
) -> None:
    """Compare small QKV boundary snapshots before and after a stream drain."""
    requested_stage = os.environ.get("USE_TK_DEBUG_QKV_FWD_STAGE", "").strip()
    if requested_stage != stage or debug_call_id is None:
        return
    name_filter = os.environ.get(
        "USE_TK_DEBUG_QKV_FWD_FILTER", ""
    ).strip()
    if name_filter and (
        not isinstance(debug_name, str) or name_filter not in debug_name
    ):
        return
    requested_call = os.environ.get(
        "USE_TK_DEBUG_QKV_FWD_CALL", "all"
    ).strip().lower()
    if requested_call not in {"all", "*"} and debug_call_id != int(
        requested_call
    ):
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    requested_rank = os.environ.get(
        "USE_TK_DEBUG_QKV_FWD_RANK", "0"
    ).strip().lower()
    if requested_rank not in {"all", "*"} and rank != int(requested_rank):
        return

    def _sample(tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach()
        if value.dim() == 2 and value.size(0) > 1:
            width = min(4096, value.size(1))
            return torch.cat((value[0, :width], value[-1, :width])).clone()
        if not value.is_contiguous():
            value = value.contiguous()
        flat = value.view(-1)
        width = min(4096, flat.numel())
        if flat.numel() <= width:
            return flat.clone()
        return torch.cat((flat[:width], flat[-width:])).clone()

    tensors = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_tensor(tensor) and tensor.numel()
    ]
    if not tensors:
        return
    early = {name: _sample(tensor) for name, tensor in tensors}
    torch.cuda.current_stream(tensors[0][1].device).synchronize()
    late = {name: _sample(tensor) for name, tensor in tensors}
    torch.cuda.current_stream(tensors[0][1].device).synchronize()

    summaries = []
    for name, tensor in tensors:
        changed = early[name].view(torch.uint8) != late[name].view(torch.uint8)
        summary = {
            "name": name,
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "dtype": str(tensor.dtype),
            "ptr": tensor.data_ptr(),
            "changed_bytes": int(changed.sum().item()),
            "sample_numel": early[name].numel(),
        }
        if tensor.dtype == torch.float4_e2m1fn_x2:
            early_bytes = early[name].view(torch.uint8)
            late_bytes = late[name].view(torch.uint8)
            summary.update(
                {
                    "early_nonzero_bytes": int((early_bytes != 0).sum().item()),
                    "late_nonzero_bytes": int((late_bytes != 0).sum().item()),
                }
            )
        else:
            early_float = early[name].float()
            late_float = late[name].float()
            early_finite = torch.isfinite(early_float)
            late_finite = torch.isfinite(late_float)
            summary.update(
                {
                    "early_nonfinite": int((~early_finite).sum().item()),
                    "late_nonfinite": int((~late_finite).sum().item()),
                    "early_max_finite_abs": float(
                        early_float[early_finite].abs().max().item()
                    )
                    if bool(early_finite.any().item())
                    else float("nan"),
                    "late_max_finite_abs": float(
                        late_float[late_finite].abs().max().item()
                    )
                    if bool(late_finite.any().item())
                    else float("nan"),
                }
            )
        summaries.append(summary)
    logger.warning(
        "TK QKV forward stage probe stage=%s call=%d owner=%s rank=%d %s",
        stage,
        debug_call_id,
        debug_name,
        rank,
        summaries,
    )


def _te_ffn_rmsnorm_forward_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.float()
    gamma_f = gamma.float()
    inv_rms = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + float(epsilon))
    normed = (x_f * inv_rms * gamma_f).to(torch.bfloat16)
    return normed, inv_rms


def _te_ffn_rmsnorm_backward_reference(
    d_normed: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    inv_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.float()
    d_normed_f = d_normed.float()
    gamma_f = gamma.float()
    inv_rms_f = inv_rms.float()
    if inv_rms_f.ndim == 1:
        inv_rms_f = inv_rms_f.unsqueeze(-1)

    proj = d_normed_f * gamma_f
    dot = (proj * x_f).mean(dim=-1, keepdim=True)
    grad_input = inv_rms_f * proj - (inv_rms_f ** 3) * x_f * dot
    grad_gamma = (d_normed_f * x_f * inv_rms_f).sum(dim=0)
    return grad_input.to(torch.bfloat16), grad_gamma.to(torch.float32)


def _scale_bytes_all_zero(scale_tensor: torch.Tensor) -> bool:
    if scale_tensor.numel() == 0:
        return True
    scale_bytes = scale_tensor.contiguous().view(torch.uint8)
    return not bool(torch.count_nonzero(scale_bytes).item())


def _localcta_ffn_backward_bf16_rescue(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    inv_rms: torch.Tensor,
    h1_raw: torch.Tensor,
    h3: torch.Tensor,
    w1_bf16: torch.Tensor,
    w3_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover FFN backward in BF16 when localCTA grad-output scales collapse."""
    grad_output_f = grad_output.float()
    w1_f = w1_bf16.float()
    w3_f = w3_bf16.float()
    w2_f = w2_bf16.float()
    h1_f = h1_raw.float()
    h3_f = h3.float()
    sig_h1 = torch.sigmoid(h1_f)
    silu_h1 = h1_f * sig_h1
    silu_deriv = sig_h1 * (1.0 + h1_f * (1.0 - sig_h1))

    h_f = silu_h1 * h3_f
    grad_w2 = torch.matmul(grad_output_f.transpose(0, 1), h_f).to(torch.bfloat16)
    dh_f = torch.matmul(grad_output_f, w2_f)
    dh1_f = dh_f * h3_f * silu_deriv
    dh3_f = dh_f * silu_h1

    inv_rms_f = inv_rms.float()
    if inv_rms_f.ndim == 1:
        inv_rms_f = inv_rms_f.unsqueeze(-1)
    normed_f = input_tensor.float() * inv_rms_f * norm_weight.float().view(1, -1)
    grad_w1 = torch.matmul(dh1_f.transpose(0, 1), normed_f).to(torch.bfloat16)
    grad_w3 = torch.matmul(dh3_f.transpose(0, 1), normed_f).to(torch.bfloat16)

    d_normed = (
        torch.matmul(dh1_f, w1_f) +
        torch.matmul(dh3_f, w3_f)
    ).to(torch.bfloat16)
    grad_input, grad_norm_weight = _te_ffn_rmsnorm_backward_reference(
        d_normed,
        input_tensor,
        norm_weight,
        inv_rms,
    )
    return grad_input, grad_w1, grad_w3, grad_w2, _as_param_grad_dtype(grad_norm_weight, norm_weight)


# Lazy-initialized secondary stream for multi-stream overlap
_ms_weight_stream: torch.cuda.Stream | None = None

def _get_ms_stream() -> torch.cuda.Stream:
    global _ms_weight_stream
    if _ms_weight_stream is None:
        _ms_weight_stream = torch.cuda.Stream()
    return _ms_weight_stream


# ---------------------------------------------------------------------------
# Lazy-load TE-fused C++ extension (uses nvfp4_transpose_fused.cuh)
# ---------------------------------------------------------------------------
def _resolve_te_include(te_root: str) -> str:
    """Find TE common include dir: source tree first, then build output."""
    import glob
    src_include = os.path.join(te_root, 'transformer_engine/common/include')
    if os.path.isdir(src_include):
        return src_include
    # Build output: build/lib.<platform>/transformer_engine/common/include
    pattern = os.path.join(te_root, 'build', 'lib.*', 'transformer_engine', 'common', 'include')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Last resort: return the source path (will fail at compile time with a clear error)
    return src_include

_bwd_side_stream = None

def _get_bwd_side_stream():
    """Get (or create) a cached CUDA stream for overlapping wgrad with RMSNorm bwd."""
    global _bwd_side_stream
    if _bwd_side_stream is None:
        _bwd_side_stream = torch.cuda.Stream()
    return _bwd_side_stream

_te_fused_ext = None

def _get_te_fused():
    """Load the TE-fused extension that wraps nvfp4_transpose_fused.cuh.

    Provides:
      - fused_te_quantize_rmsnorm_silu_2pass(x, w, eps) → (fp4, scales, inv_rms, amax)
      - fused_te_quantize_rmsnorm_silu(x, inv_rms, w, amax) → (fp4, scales)
      - fused_te_quantize_rmsnorm(x, inv_rms, w, amax) → (fp4, scales)
      - fused_silu_rmsnorm_backward(dx_proj, x_raw, w, inv_rms) → (dx, dgamma)
    """
    global _te_fused_ext
    if _te_fused_ext is None:
        import os
        import ctypes
        from torch.utils.cpp_extension import load

        # Derive paths relative to this file
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(os.path.dirname(_this_dir))
        _fp4_root = _fp4_matmul_root(_repo_root)

        # Locate the installed TE tree without importing its Python package.
        # Importing transformer_engine.pytorch also imports its Triton helpers,
        # even though this native CUDA extension only needs headers and the
        # shared library.
        import importlib.util

        _te_spec = importlib.util.find_spec("transformer_engine")
        if _te_spec is None or not _te_spec.submodule_search_locations:
            raise RuntimeError("Cannot locate the installed transformer_engine package")
        _te_pkg_dir = os.path.dirname(
            os.path.abspath(next(iter(_te_spec.submodule_search_locations)))
        )
        _TE_CANDIDATES = [
            _te_pkg_dir,  # installed TE package root (highest priority)
            os.path.join(_repo_root, 'TransformerEngine_v29_backup'),
            os.path.join(_repo_root, 'TransformerEngine'),
        ]
        # Deduplicate while preserving order
        seen = set()
        _TE_CANDIDATES = [c for c in _TE_CANDIDATES if not (c in seen or seen.add(c))]

        TE_ROOT = None
        TE_LIB_DIR = None
        for _cand in _TE_CANDIDATES:
            for _lib_dir in (
                os.path.join(_cand, 'build/cmake'),
                os.path.join(_cand, 'transformer_engine/wheel_lib'),
                os.path.join(_cand, 'transformer_engine'),
                _cand,
            ):
                if os.path.exists(os.path.join(_lib_dir, 'libtransformer_engine.so')):
                    TE_ROOT = _cand
                    TE_LIB_DIR = _lib_dir
                    break
            if TE_ROOT is not None:
                break
        if TE_ROOT is None:
            raise RuntimeError(f"Cannot find libtransformer_engine.so in any of: {_TE_CANDIDATES}")

        for _dep in ['/usr/local/cuda/lib64/libnvrtc.so',
                     '/usr/local/cuda/lib64/libcudart.so',
                     os.path.join(TE_LIB_DIR, 'libtransformer_engine.so')]:
            if os.path.exists(_dep):
                ctypes.CDLL(_dep, mode=ctypes.RTLD_GLOBAL)

        TE_INCLUDE = _resolve_te_include(TE_ROOT)
        TE_COMMON = os.path.join(TE_ROOT, 'transformer_engine/common')
        TE_CUSTOM = os.path.join(TE_ROOT, 'transformer_engine/common/cast/nvfp4/custom_quantisation')
        TE_CAST_NVFP4 = os.path.join(TE_ROOT, 'transformer_engine/common/cast/nvfp4')
        CSRC = os.path.join(_fp4_root, 'fused_ops', 'csrc')
        CUDA_LIB = '/usr/local/cuda/lib64'

        sources = [
            os.path.join(CSRC, 'te_fused_rmsnorm_ext.cpp'),
            os.path.join(CSRC, 'te_fused_pass1.cu'),
            os.path.join(CSRC, 'fused_silu_rmsnorm_backward.cu'),
            os.path.join(CSRC, 'elementwise_mul.cu'),
            os.path.join(CSRC, 'fused_amax_bf16.cu'),
        ]
        include_paths = [TE_INCLUDE, '/usr/local/cuda/include', CSRC]
        extra_cflags = ['-std=c++17']
        extra_cuda_cflags = ['-std=c++17', '--expt-relaxed-constexpr', '-O3']

        # Check if TE custom quantisation headers are available.
        # Old TE had common.h in cast/nvfp4/, new TE moved it.
        # Check for the custom_quantisation dir which exists in both versions.
        _custom_quant_dir = os.path.join(TE_CAST_NVFP4, 'custom_quantisation')
        if use_custom_quant() and os.path.isdir(_custom_quant_dir):
            sources.append(os.path.join(CSRC, 'custom_quantize.cu'))
            # TE_CAST_NVFP4: ../../common.h resolves via -I fallback
            # TE_PARENT: matches TE CMake's include_directories(${PROJECT_SOURCE_DIR}/..)
            #   needed for #include "common/utils.cuh" etc.
            TE_PARENT = os.path.join(TE_ROOT, 'transformer_engine')
            include_paths.extend([TE_COMMON, TE_CUSTOM, TE_CAST_NVFP4, TE_PARENT])
            extra_cflags.append('-DCUSTOM_QUANT_ENABLED')
            extra_cuda_cflags.append('-DCUSTOM_QUANT_ENABLED')
            # TE kernels need arch-specific features (sm100a, not generic sm100)
            os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'

        _te_fused_ext = load(
            name='te_fused_rmsnorm_ext_linear',
            sources=sources,
            extra_include_paths=include_paths,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=[
                f'-L{TE_LIB_DIR}', '-ltransformer_engine',
                f'-Wl,-rpath,{TE_LIB_DIR}',
                f'-L{CUDA_LIB}', '-lcudart', '-lnvrtc',
                f'-Wl,-rpath,{CUDA_LIB}',
            ],
            verbose=False,
        )

        # Set custom quant flag at C++ level
        if use_custom_quant():
            _te_fused_ext.set_custom_quant(True)

    return _te_fused_ext


# ---------------------------------------------------------------------------
# Minimal FP4 quantize extension (always buildable, no fused rmsnorm deps)
# ---------------------------------------------------------------------------
_fp4_ext = None

# ---------------------------------------------------------------------------
# CUTLASS grouped GEMM extension (optional, for per-group FP4 GEMM)
# ---------------------------------------------------------------------------
_cutlass_gemm_ext = None
_cutlass_gemm_ext_attempted = False

def _get_cutlass_gemm_ext():
    """Load the CUTLASS grouped NVFP4 GEMM extension.

    Provides:
      - forward(A_data, A_sf, B_data_list, B_sf_list, N_dims, M, K, alpha)
      - convert_te_sf_to_cutlass(te_sf, rows, cols)
    """
    global _cutlass_gemm_ext, _cutlass_gemm_ext_attempted
    if _cutlass_gemm_ext is not None:
        return _cutlass_gemm_ext
    if _cutlass_gemm_ext_attempted:
        return None
    _cutlass_gemm_ext_attempted = True
    try:
        import os
        from torch.utils.cpp_extension import load
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(os.path.dirname(_this_dir))
        _fp4_root = _fp4_matmul_root(_repo_root)
        CUTLASS_ROOT = os.path.join(_fp4_root, 'cutlass')
        _cutlass_gemm_ext = load(
            name='fp4_grouped_gemm_ext',
            sources=[os.path.join(_fp4_root, 'fused_ops', 'csrc', 'fp4_grouped_gemm.cu')],
            extra_include_paths=[
                os.path.join(CUTLASS_ROOT, 'include'),
                os.path.join(CUTLASS_ROOT, 'tools/util/include'),
                os.path.join(CUTLASS_ROOT, 'examples/common'),
            ],
            extra_cuda_cflags=[
                '-std=c++17', '-O3', '--expt-relaxed-constexpr',
                '-gencode=arch=compute_100a,code=sm_100a',
                '-DCUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED',
                '-DCUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED',
                '-DCUTE_ARCH_TCGEN05_TMEM_ENABLED',
            ],
            verbose=False,
        )
        print("[FusedQKV] CUTLASS grouped GEMM extension loaded successfully")
    except Exception as e:
        print(f"[FusedQKV] CUTLASS grouped GEMM extension not available: {e}")
        _cutlass_gemm_ext = None
    return _cutlass_gemm_ext

def _get_fp4_ext():
    """Load the minimal FP4 quantize extension.

    Provides:
      - fast_nvfp4_quantize_v2(tensor, encode_centric) → (fp4, si, fp4_t, si_t, amax, amax_t)
      - group_nvfp4_quantize(tensor, split_sections) → [(fp4, si, fp4_t, si_t, amax), ...]
    """
    global _fp4_ext
    if _fp4_ext is None:
        import os
        import ctypes
        from torch.utils.cpp_extension import load

        # Derive paths relative to this file
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(os.path.dirname(_this_dir))
        _fp4_root = _fp4_matmul_root(_repo_root)

        import transformer_engine as _te_mod
        _te_pkg_dir = os.path.dirname(os.path.abspath(_te_mod.__path__[0]))
        _TE_CANDIDATES = [
            _te_pkg_dir,
            os.path.join(_repo_root, 'TransformerEngine'),
        ]
        seen = set()
        _TE_CANDIDATES = [c for c in _TE_CANDIDATES if not (c in seen or seen.add(c))]

        TE_ROOT = None
        TE_LIB_DIR = None
        for _cand in _TE_CANDIDATES:
            for _lib_dir in (
                os.path.join(_cand, 'build/cmake'),
                os.path.join(_cand, 'transformer_engine/wheel_lib'),
                os.path.join(_cand, 'transformer_engine'),
                _cand,
            ):
                if os.path.exists(os.path.join(_lib_dir, 'libtransformer_engine.so')):
                    TE_ROOT = _cand
                    TE_LIB_DIR = _lib_dir
                    break
            if TE_ROOT is not None:
                break
        if TE_ROOT is None:
            raise RuntimeError(f"Cannot find libtransformer_engine.so in any of: {_TE_CANDIDATES}")

        # Pre-load TE shared lib
        te_lib = os.path.join(TE_LIB_DIR, 'libtransformer_engine.so')
        if os.path.exists(te_lib):
            ctypes.CDLL(te_lib, mode=ctypes.RTLD_GLOBAL)

        TE_INCLUDE = _resolve_te_include(TE_ROOT)
        TE_COMMON = os.path.join(TE_ROOT, 'transformer_engine/common')
        TE_CUSTOM = os.path.join(TE_ROOT, 'transformer_engine/common/cast/nvfp4/custom_quantisation')
        TE_CAST_NVFP4 = os.path.join(TE_ROOT, 'transformer_engine/common/cast/nvfp4')
        CSRC = os.path.join(_fp4_root, 'fused_ops', 'csrc')

        sources = [
            os.path.join(CSRC, 'fp4_quantize_ext.cpp'),
            os.path.join(CSRC, 'fused_amax_bf16.cu'),
        ]
        include_paths = [TE_INCLUDE, CSRC]
        extra_cflags = []
        extra_cuda_cflags = []
        
        _common_h = os.path.join(TE_ROOT, 'transformer_engine/common/cast/nvfp4', 'common.h')
        if (int(os.environ.get("USE_TK_GEMM", "0")) or use_custom_quant()) and os.path.isfile(_common_h):
            sources.append(os.path.join(CSRC, 'custom_quantize.cu'))
            TE_PARENT = os.path.join(TE_ROOT, 'transformer_engine')
            include_paths.extend([TE_COMMON, TE_CUSTOM, TE_CAST_NVFP4, TE_PARENT])
            if "-DCUSTOM_QUANT_ENABLED" not in extra_cflags:
                extra_cflags.append('-DCUSTOM_QUANT_ENABLED')
            if "-DCUSTOM_QUANT_ENABLED" not in extra_cuda_cflags:
                extra_cuda_cflags.append('-DCUSTOM_QUANT_ENABLED')
            os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
            os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'

        _fp4_ext = load(
            name='fp4_quantize_ext',
            sources=sources,
            extra_include_paths=include_paths,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=[
                f'-L{TE_LIB_DIR}', '-ltransformer_engine',
                f'-Wl,-rpath,{TE_LIB_DIR}',
            ],
            verbose=False,
        )
    return _fp4_ext


# ---------------------------------------------------------------------------
# Lazy-load TK standalone quant module (_tk_quant.so)
# ---------------------------------------------------------------------------
_tk_quant_mod = None

def _get_tk_quant():
    """Load the pre-compiled TK standalone quantisation module.

    Provides:
      - tk_quantize_for_gemm(tensor, transpose) → (fp4, sc_3d, fp4_t, sc_3d_t, sg, sg_t)
      - tk_quantize_transpose(tensor, amax, amax, transpose) → (fp4, sc, fp4_t, sc_t)
    """
    global _tk_quant_mod
    if _tk_quant_mod is None:
        import sys
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(os.path.dirname(_this_dir))
        base_dir = os.path.join(_fp4_matmul_root(_repo_root), 'TK_quantisation')
        # Add both nvfp4 and nvfp4_v5 so either module can be found
        for subdir in ['nvfp4_v5', 'nvfp4']:
            d = os.path.join(base_dir, subdir)
            if d not in sys.path:
                sys.path.insert(0, d)
        try:
            import _tk_quant_v5 as _tk_quant
        except ImportError:
            import _tk_quant
        _tk_quant_mod = _tk_quant
    return _maybe_wrap_v5_ffn_quantizer(_tk_quant_mod)


def _regular_tk_sg_1d(sg, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(sg):
        return sg.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.tensor([float(sg)], dtype=torch.float32, device=device)


def _regular_tk_expand_sg_tiles(sg: torch.Tensor, n_tiles: int) -> torch.Tensor:
    sg = sg.reshape(-1)
    if sg.numel() == n_tiles:
        return sg.contiguous()
    if sg.numel() == 1:
        return sg.expand(n_tiles).contiguous()
    raise RuntimeError(
        f"Cannot expand regular TK SG payload with {sg.numel()} values to {n_tiles} tiles"
    )


_regular_qkv_weight_quant_state_cache = {}
_REGULAR_QKV_WEIGHT_QUANT_STATE_MAX_SLOTS = 4


def clear_tk_qkv_persistent_weight_quant_state() -> None:
    """Release reusable regular-v5 QKV weight-quantization state.

    Callers must first quiesce every eager launch and captured graph that uses
    the state.  Captured graphs retain the addresses in their launch nodes, so
    this is deliberately an explicit maintenance hook rather than a step-cache
    cleanup.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "cannot clear persistent QKV weight-quant graph state during capture"
        )
    _regular_qkv_weight_quant_state_cache.clear()


class _RegularQKVWeightQuantStateLease:
    __slots__ = ('slot', 'forward_recorded', 'finished')

    def __init__(self, slot):
        self.slot = slot
        self.forward_recorded = False
        self.finished = False

    def record_forward_consumed(self) -> None:
        if self.finished or self.forward_recorded:
            return
        stream = torch.cuda.current_stream(self.slot['device'])
        if (
            not self.slot['graph_pinned']
            and int(stream.cuda_stream) != self.slot['stream_id']
        ):
            self.slot['event'].record(stream)
            self.slot['event_pending'] = True
        else:
            self.slot['event_pending'] = False
        self.forward_recorded = True

    def release_after_backward(self) -> None:
        if self.finished:
            return
        if self.slot['graph_pinned']:
            # A captured graph keeps the state addresses for its whole
            # lifetime.  Never return that slot to eager callers.
            self.finished = True
            return
        stream = torch.cuda.current_stream(self.slot['device'])
        if int(stream.cuda_stream) != self.slot['stream_id']:
            self.slot['event'].record(stream)
            self.slot['event_pending'] = True
        else:
            self.slot['event_pending'] = False
        self.slot['released'] = True
        self.finished = True

    def __del__(self):
        if (
            not self.finished
            and self.forward_recorded
            and not self.slot['graph_pinned']
        ):
            self.slot['released'] = True
            self.finished = True


def _regular_qkv_weight_quant_state_lease(keepalive):
    if isinstance(keepalive, tuple):
        for value in keepalive:
            if isinstance(value, _RegularQKVWeightQuantStateLease):
                return value
    return None


def _acquire_regular_qkv_weight_quant_state(
    tk_q,
    w_bf16: torch.Tensor,
    split_sections,
    owner_key=None,
):
    """Acquire a stable regular-v5 grouped-weight output package.

    The native ``v5_alloc`` entrypoint performs pinned-host allocation and a
    descriptor copy, neither of which may run during CUDA graph capture.  An
    eager full forward/backward warmup creates a reusable slot; capture then
    pins that released slot permanently and invokes only ``v5_launch``.
    """
    split_sections = tuple(int(value) for value in split_sections)
    stream = torch.cuda.current_stream(w_bf16.device)
    stream_id = int(stream.cuda_stream)
    capturing = bool(torch.cuda.is_current_stream_capturing())
    key = (
        None if owner_key is None else str(owner_key),
        id(tk_q),
        int(w_bf16.data_ptr()),
        w_bf16.device.index,
        stream_id,
        tuple(int(value) for value in w_bf16.shape),
        split_sections,
    )
    slots = _regular_qkv_weight_quant_state_cache.get(key)
    if slots is not None:
        for slot in slots:
            if slot['released'] and not slot['graph_pinned']:
                if slot['event_pending']:
                    if capturing:
                        # A pre-capture event must already be complete.  Adding
                        # an external event wait to the graph would make replay
                        # depend on state that is not owned by the graph.
                        if not slot['event'].query():
                            raise RuntimeError(
                                'regular TK QKV graph capture requires its '
                                'eager warmup stream to be synchronized'
                            )
                    else:
                        stream.wait_event(slot['event'])
                    slot['event_pending'] = False
                slot['released'] = False
                if capturing:
                    slot['graph_pinned'] = True
                return slot, _RegularQKVWeightQuantStateLease(slot)

    if capturing:
        raise RuntimeError(
            'regular TK QKV graph capture requires a completed eager '
            'forward/backward warmup on the same weight and CUDA stream'
        )

    if slots is None:
        slots = []
        _regular_qkv_weight_quant_state_cache[key] = slots
    if len(slots) >= _REGULAR_QKV_WEIGHT_QUANT_STATE_MAX_SLOTS:
        return None, None

    alloc = tuple(
        tk_q.tk_group_quantize_v5_alloc(w_bf16, list(split_sections))
    )
    if len(alloc) != 12:
        raise RuntimeError(
            'regular TK QKV v5_alloc returned an unexpected state package: '
            f'{len(alloc)} values'
        )
    (
        wc_fp4_row,
        wc_fp4_col,
        sg_cat,
        fwd_b_sg,
        dgrad_b_sg,
        _amax,
        _sync,
        _psync,
        _tma_dev,
        sc_row_list,
        _fp4_col_scratch,
        sc_col_list,
    ) = alloc
    total_rows, K = map(int, w_bf16.shape)
    wc_sc_row_raw = torch.empty(
        total_rows // 128,
        K // 64,
        512,
        dtype=torch.uint8,
        device=w_bf16.device,
    )
    wc_fp4_col_raw_parts = [
        torch.empty(K, rows // 2, dtype=torch.uint8, device=w_bf16.device)
        for rows in split_sections
    ]
    slot = {
        'alloc': alloc,
        'wc_fp4_row': (
            wc_fp4_row
            if wc_fp4_row.dtype == torch.float4_e2m1fn_x2
            else wc_fp4_row.view(torch.float4_e2m1fn_x2)
        ),
        'wc_sc_row_raw': wc_sc_row_raw,
        'wc_sc_row': wc_sc_row_raw.view(torch.float8_e4m3fn),
        'wc_fp4_col_raw': wc_fp4_col,
        'fwd_b_sg': fwd_b_sg,
        'wc_fp4_col_raw_parts': wc_fp4_col_raw_parts,
        'wc_fp4_cols': [
            value.view(torch.float4_e2m1fn_x2)
            for value in wc_fp4_col_raw_parts
        ],
        'wc_sc_cols': [value.view(torch.float8_e4m3fn) for value in sc_col_list],
        'dgrad_b_sg': dgrad_b_sg,
        'sg_cat': sg_cat,
        'sc_row_list': sc_row_list,
        'event': torch.cuda.Event(),
        'device': w_bf16.device,
        'stream_id': stream_id,
        'event_pending': False,
        'released': False,
        'graph_pinned': False,
    }
    slots.append(slot)
    return slot, _RegularQKVWeightQuantStateLease(slot)


def _regular_tk_group_quantize_qkv_weights(
    tk_q,
    w_bf16: torch.Tensor,
    split_sections,
    *,
    owner_key=None,
):
    """Run graph-safe regular-v5 QKV grouped weight quantization."""
    capturing = bool(torch.cuda.is_current_stream_capturing())
    if use_tk_v5_2d_weight_quant() or _nvfp4_quantizer_extras_enabled("weight"):
        if capturing:
            raise RuntimeError(
                "QKV 2D/RHT/SR weight quantization decomposes the grouped producer "
                "and is not graph-safe"
            )
        weights = w_bf16.split(list(split_sections), dim=0)
        if use_tk_v5_2d_weight_quant():
            results = [
                _tk_quantized_as_result_tuple(
                    _fast_quantize_v5_2d_weight_swizzled(weight)
                )
                for weight in weights
            ]
        else:
            results = [
                _tk_quantized_as_result_tuple(
                    _fast_quantize_tk_regular_opt(weight, nvfp4_role="weight")
                )
                for weight in weights
            ]
        wc_fp4_row = torch.cat(
            [result[0].contiguous().view(torch.uint8) for result in results],
            dim=0,
        ).view(torch.float4_e2m1fn_x2)
        wc_sc_row = torch.cat(
            [result[1].contiguous().view(torch.uint8) for result in results],
            dim=0,
        ).view(torch.float8_e4m3fn)
        wc_fp4_cols = [
            result[2].contiguous().view(torch.float4_e2m1fn_x2)
            for result in results
        ]
        wc_sc_cols = [
            result[3].contiguous().view(torch.float8_e4m3fn)
            for result in results
        ]
        sg_values = [
            _regular_tk_sg_1d(result[4], w_bf16.device)
            for result in results
        ]
        sg_cat = torch.cat([sg[:1] for sg in sg_values], dim=0).contiguous()
        fwd_b_sg = torch.cat(
            [
                _regular_tk_expand_sg_tiles(sg, int(rows) // 256)
                for sg, rows in zip(sg_values, split_sections, strict=True)
            ],
            dim=0,
        ).contiguous()
        dgrad_b_sg = torch.cat(
            [
                _regular_tk_expand_sg_tiles(sg, w_bf16.size(1) // 256)
                for sg in sg_values
            ],
            dim=0,
        ).contiguous()
        keepalive = tuple(
            value
            for result in results
            for value in result[6:]
        )
        return (
            wc_fp4_row,
            wc_sc_row,
            fwd_b_sg,
            wc_fp4_cols,
            wc_sc_cols,
            dgrad_b_sg,
            sg_cat,
            keepalive,
        )
    if not (use_cuda_graph() or capturing):
        # Preserve the exact established eager wrapper, including its output
        # allocations and descriptor-copy chronology, unless graph execution
        # was explicitly requested.
        return tk_q.tk_group_quantize_for_gemm(w_bf16, list(split_sections))
    if owner_key is None:
        raise RuntimeError(
            'regular TK QKV graph execution requires an explicit QKV owner key'
        )

    has_split_api = all(
        hasattr(tk_q, name)
        for name in ('tk_group_quantize_v5_alloc', 'tk_group_quantize_v5_launch')
    )
    if not has_split_api:
        raise RuntimeError(
            'regular TK QKV graph execution requires '
            'tk_group_quantize_v5_alloc/tk_group_quantize_v5_launch'
        )

    slot, lease = _acquire_regular_qkv_weight_quant_state(
        tk_q, w_bf16, split_sections, owner_key=owner_key
    )
    if slot is None:
        # This is reachable only in eager mode after all bounded state slots
        # are concurrently leased.  Preserve the established eager fallback;
        # capture instead fails closed above because it cannot allocate.
        return tk_q.tk_group_quantize_for_gemm(w_bf16, list(split_sections))

    tk_q.tk_group_quantize_v5_launch(
        w_bf16,
        list(split_sections),
        *slot['alloc'],
    )
    col_offset = 0
    for rows, destination in zip(
        split_sections, slot['wc_fp4_col_raw_parts'], strict=True
    ):
        # Match the monolithic wrapper's
        # wc_fp4_col.narrow(...).contiguous() return contract.  The v5 launch
        # writes the full transpose through its output TMA map; the slot owns
        # separate contiguous graph storage for these semantic copies.
        destination.copy_(
            slot['wc_fp4_col_raw'].narrow(1, col_offset // 2, int(rows) // 2)
        )
        col_offset += int(rows)
    torch.cat(slot['sc_row_list'], dim=0, out=slot['wc_sc_row_raw'])
    keepalive = (slot['alloc'], lease)
    return (
        slot['wc_fp4_row'],
        slot['wc_sc_row'],
        slot['fwd_b_sg'],
        slot['wc_fp4_cols'],
        slot['wc_sc_cols'],
        slot['dgrad_b_sg'],
        slot['sg_cat'],
        keepalive,
    )


def _regular_tk_group_quantize_ffn_weights_decomposed(
    tk_q,
    w1_bf16: torch.Tensor,
    w3_bf16: torch.Tensor,
):
    if use_tk_v5_2d_weight_quant():
        q1 = _tk_quantized_as_result_tuple(
            _fast_quantize_v5_2d_weight_swizzled(w1_bf16)
        )
        q3 = _tk_quantized_as_result_tuple(
            _fast_quantize_v5_2d_weight_swizzled(w3_bf16)
        )
    elif _nvfp4_quantizer_extras_enabled("weight"):
        q1 = _tk_quantized_as_result_tuple(
            _fast_quantize_tk_regular_opt(w1_bf16, nvfp4_role="weight")
        )
        q3 = _tk_quantized_as_result_tuple(
            _fast_quantize_tk_regular_opt(w3_bf16, nvfp4_role="weight")
        )
    else:
        q1 = tk_q.tk_quantize_for_gemm(w1_bf16, True)
        q3 = tk_q.tk_quantize_for_gemm(w3_bf16, True)

    wc_fp4_row = torch.cat(
        [q1[0].contiguous().view(torch.uint8), q3[0].contiguous().view(torch.uint8)],
        dim=0,
    ).view(torch.float4_e2m1fn_x2)
    wc_sc_row = torch.cat(
        [q1[1].contiguous().view(torch.uint8), q3[1].contiguous().view(torch.uint8)],
        dim=0,
    ).view(torch.float8_e4m3fn)

    wc_fp4_cols = [
        q1[2].contiguous().view(torch.float4_e2m1fn_x2),
        q3[2].contiguous().view(torch.float4_e2m1fn_x2),
    ]
    wc_sc_cols = [
        q1[3].contiguous().view(torch.float8_e4m3fn),
        q3[3].contiguous().view(torch.float8_e4m3fn),
    ]

    device = w1_bf16.device
    row_sg1 = _regular_tk_sg_1d(q1[4], device)
    row_sg3 = _regular_tk_sg_1d(q3[4], device)
    col_sg1 = _regular_tk_sg_1d(q1[5], device)
    col_sg3 = _regular_tk_sg_1d(q3[5], device)
    # sg_cat is retained for backward consumers.  Row/column SGs are equal in
    # the legacy path, but row-only RHT gives them distinct amax domains.
    sg_cat = torch.cat([col_sg1[:1], col_sg3[:1]], dim=0).contiguous()

    fwd_b_sg = torch.cat(
        [
            _regular_tk_expand_sg_tiles(row_sg1, w1_bf16.size(0) // 256),
            _regular_tk_expand_sg_tiles(row_sg3, w3_bf16.size(0) // 256),
        ],
        dim=0,
    ).contiguous()
    dgrad_b_sg = torch.cat(
        [
            _regular_tk_expand_sg_tiles(col_sg1, w1_bf16.size(1) // 256),
            _regular_tk_expand_sg_tiles(col_sg3, w3_bf16.size(1) // 256),
        ],
        dim=0,
    ).contiguous()

    # The concatenated row payloads own their storage, while the returned column
    # lists already retain q1/q3's column payloads.  Keeping both complete
    # quantizer results alive duplicated the row payloads in every transformer
    # layer until backward.
    keepalive = tuple(
        value
        for result in (q1, q3)
        for value in result[6:]
    )
    return (
        wc_fp4_row, wc_sc_row, fwd_b_sg,
        wc_fp4_cols, wc_sc_cols, dgrad_b_sg, sg_cat, keepalive,
    )


def _localcta_group_quantize_weights_2d(
    weight_bf16: torch.Tensor,
    split_sections,
):
    """Build the localCTA grouped contract from consistent 2D weight payloads."""
    sections = [int(rows) for rows in split_sections]
    weights = weight_bf16.split(sections, dim=0)
    results = [
        _tk_quantized_as_result_tuple(
            _fast_quantize_localcta_2d_weight_swizzled(weight)
        )
        for weight in weights
    ]

    wc_fp4_row = torch.cat(
        [result[0].contiguous().view(torch.uint8) for result in results],
        dim=0,
    ).view(torch.float4_e2m1fn_x2)
    wc_sc_row = torch.cat(
        [result[1].contiguous().view(torch.uint8) for result in results],
        dim=0,
    ).view(torch.float8_e4m3fn)
    wc_fp4_cols = [
        result[2].contiguous().view(torch.float4_e2m1fn_x2)
        for result in results
    ]
    wc_sc_cols = [
        result[3].contiguous().view(torch.float8_e4m3fn)
        for result in results
    ]

    device = weight_bf16.device
    row_sg_scalars = [
        _regular_tk_sg_1d(result[4], device)[:1] for result in results
    ]
    col_sg_scalars = [
        _regular_tk_sg_1d(result[5], device)[:1] for result in results
    ]
    row_sg_parts = [
        _regular_tk_expand_sg_tiles(sg, rows // 256)
        for sg, rows in zip(row_sg_scalars, sections, strict=True)
    ]
    col_sg_parts = [
        _regular_tk_expand_sg_tiles(sg, weight_bf16.size(1) // 256)
        for sg in col_sg_scalars
    ]
    fwd_b_sg = torch.cat(row_sg_parts, dim=0).contiguous()
    col_sg_cat = torch.cat(col_sg_parts, dim=0).contiguous()

    keepalive = tuple(
        tensor
        for result in results
        for tensor in result[6:]
        if torch.is_tensor(tensor)
    )
    return (
        wc_fp4_row,
        wc_sc_row,
        fwd_b_sg,
        wc_fp4_cols,
        wc_sc_cols,
        col_sg_cat,
        row_sg_parts,
        col_sg_parts,
        *keepalive,
    )


def _tk_group_quantize_ffn_weights(
    tk_q,
    w1_bf16: torch.Tensor,
    w3_bf16: torch.Tensor | None = None,
    split_sections=None,
    prefer_split: bool = True,
):
    """Quantize FFN grouped weights with the fastest measured eager path.

    For the FFN [W1; W3] forward case at M=65536, the v2 grouped dim0 path
    benchmarks slightly faster than the default persistent grouped path.
    Keep this FFN-specific so other grouped call sites stay on their existing,
    separately-tuned behavior until they are benchmarked in isolation.
    """
    w1_bf16 = _as_contiguous_bf16(w1_bf16)
    if w1_bf16.dim() != 2:
        raise RuntimeError(
            f"Expected FFN W1 weight to be 2D after normalization, got shape={tuple(w1_bf16.shape)}"
        )
    if w3_bf16 is not None:
        w3_bf16 = _as_contiguous_bf16(w3_bf16)
        if w3_bf16.dim() != 2:
            raise RuntimeError(
                f"Expected FFN W3 weight to be 2D after normalization, got shape={tuple(w3_bf16.shape)}"
            )
    if (
        w3_bf16 is not None
        and getattr(tk_q, 'is_localcta', False)
        and use_tk_localcta_2d_weight_quant()
    ):
        return _localcta_group_quantize_weights_2d(
            torch.cat([w1_bf16, w3_bf16], dim=0),
            [w1_bf16.size(0), w3_bf16.size(0)],
        )
    if (
        w3_bf16 is not None
        and use_tk_ffn_decomposed_weight_quant()
        and hasattr(tk_q, 'tk_quantize_for_gemm')
    ):
        return _regular_tk_group_quantize_ffn_weights_decomposed(
            tk_q, w1_bf16, w3_bf16
        )
    _check_nvfp4_native_extras_supported("weight", "grouped FFN weight quantizer")
    if (
        w3_bf16 is not None
        and prefer_split
        and use_tk_ffn_split_weight_quant()
        and hasattr(tk_q, 'tk_group_quantize_split_for_gemm_v2')
    ):
        from .tk_gemm import get_tk_localcta_variant, use_tk_localcta_v4_strict_path
        if (
            get_tk_localcta_variant() == 'v4'
            and getattr(tk_q, 'is_localcta', False)
            and hasattr(tk_q, 'tk_group_quantize_split_for_gemm_v2')
        ):
            return tk_q.tk_group_quantize_split_for_gemm_v2(w1_bf16, w3_bf16)
        if use_tk_ffn_split_weight_fast() and hasattr(tk_q, 'tk_group_quantize_for_gemm_fast'):
            if split_sections is None:
                split_sections = [w1_bf16.shape[0], w3_bf16.shape[0]]
            w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=0)
            return tk_q.tk_group_quantize_for_gemm_fast(w13_bf16, split_sections)
        return tk_q.tk_group_quantize_split_for_gemm_v2(w1_bf16, w3_bf16)
    if w3_bf16 is not None:
        if split_sections is None:
            split_sections = [w1_bf16.shape[0], w3_bf16.shape[0]]
        w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=0)
        if (
            not use_tk_ffn_weight_quant_v2()
            and hasattr(tk_q, 'tk_group_quantize_for_gemm')
        ):
            return tk_q.tk_group_quantize_for_gemm(w13_bf16, split_sections)
        return tk_q.tk_group_quantize_for_gemm_v2(w13_bf16, split_sections)
    if (
        not use_tk_ffn_weight_quant_v2()
        and hasattr(tk_q, 'tk_group_quantize_for_gemm')
    ):
        return tk_q.tk_group_quantize_for_gemm(w1_bf16, split_sections)
    return tk_q.tk_group_quantize_for_gemm_v2(w1_bf16, split_sections)


class _TKQuantized:
    """Lightweight wrapper for TK-native quantized tensors.
    Bypasses NVFP4Tensor and NVFP4Quantizer construction overhead (~20µs each).
    Only stores the tuples that TK GEMM wrappers actually read."""
    __slots__ = (
        '_tk_row', '_tk_col', '_tk_row_chunk_sg', '_tk_col_chunk_sg',
        '_with_gemm_swizzled_scales', '_keepalive', 'shape'
    )
    def __init__(self, fp4, si, sg, fp4_t, si_t, sg_t=None,
                 row_chunk_sg=None, col_chunk_sg=None, keepalive=()):
        # Ensure fp4/scale tensors have the correct dtype (some C++ paths return uint8)
        fp4 = fp4.view(torch.float4_e2m1fn_x2) if fp4.dtype != torch.float4_e2m1fn_x2 else fp4
        si = si.view(torch.float8_e4m3fn) if si.dtype != torch.float8_e4m3fn else si
        fp4_t = fp4_t.view(torch.float4_e2m1fn_x2) if fp4_t.numel() > 0 and fp4_t.dtype != torch.float4_e2m1fn_x2 else fp4_t
        si_t = si_t.view(torch.float8_e4m3fn) if si_t.numel() > 0 and si_t.dtype != torch.float8_e4m3fn else si_t
        if sg_t is None:
            sg_t = sg
        self._tk_row = (fp4, si, sg)
        self._tk_col = (fp4_t, si_t, sg_t)
        self._tk_row_chunk_sg = row_chunk_sg
        self._tk_col_chunk_sg = col_chunk_sg
        self._keepalive = tuple(keepalive)
        self._with_gemm_swizzled_scales = True
        # fp4 is (M, K/2) in fp4x2 dtype → logical shape is (M, K)
        self.shape = (fp4.shape[0], fp4.shape[1] * 2)


def _release_tk_row_storage(quantized) -> None:
    """Release forward-only TK row and producer workspace storage."""
    row = getattr(quantized, '_tk_row', None)
    if not isinstance(row, tuple):
        return
    quantized._tk_row = tuple(
        value.new_empty((0,)) if torch.is_tensor(value) else value
        for value in row
    )
    quantized._tk_row_chunk_sg = None
    quantized._keepalive = ()


def _optional_result_tensor(result, idx: int):
    if len(result) > idx and torch.is_tensor(result[idx]) and result[idx].numel() > 0:
        return result[idx]
    return None


def _result_keepalive(result, start: int = 6):
    if len(result) <= start:
        return ()
    return tuple(t for t in result[start:] if torch.is_tensor(t))


def _localcta_require_paired_col_rht(path: str) -> bool:
    """Validate the only RHT contract supported by localCTA wgrad carriers.

    Forward and dgrad consume the original row payload.  Wgrad may consume a
    column-RHT payload only when both of its contracted operands use that same
    transform.  The mixed-carrier producers below preserve the established
    forward row byte-for-byte and replace only the cached activation column.
    """
    activation_rht = use_nvfp4_rht_for_role("activation")
    grad_rht = use_nvfp4_rht_for_role("grad")
    if activation_rht != grad_rht:
        raise RuntimeError(
            f"{path} requires matched activation/gradient RHT enable bits; "
            f"got activation={activation_rht}, grad={grad_rht}"
        )
    if not activation_rht:
        return False
    if not use_tk_localcta_paired_rht_carrier():
        raise RuntimeError(
            f"{path} requires USE_TK_LOCALCTA_PAIRED_RHT_CARRIER=1 "
            "whenever paired column RHT is enabled"
        )
    if not _nvfp4_rht_random_sign_mask():
        raise RuntimeError(
            f"{path} requires NVFP4_RHT_RANDOM_SIGNS=1 so forward and "
            "backward use the sealed fixed-sign RHT geometry"
        )
    activation_axes = _nvfp4_native_rht_axes_for_role("activation")
    grad_axes = _nvfp4_native_rht_axes_for_role("grad")
    if activation_axes != "col" or grad_axes != "col":
        raise RuntimeError(f"{path} preserves the forward row and supports column RHT only")
    if use_nvfp4_rht_for_role("weight"):
        raise RuntimeError(f"{path} requires weight RHT to remain disabled")
    if use_nvfp4_data_stochastic_rounding_for_role("activation"):
        raise RuntimeError(
            f"{path} cannot preserve the established forward row while activation data SR is enabled"
        )
    if use_nvfp4_scale_stochastic_rounding_for_role("activation"):
        raise RuntimeError(
            f"{path} cannot preserve the established forward row while activation scale SR is enabled"
        )
    return True


def _localcta_quantized_from_result(result) -> '_TKQuantized':
    return _TKQuantized(
        result[0], result[1], result[4],
        result[2], result[3],
        result[5]
        if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0
        else result[4],
        row_chunk_sg=_optional_result_tensor(result, 6),
        col_chunk_sg=_optional_result_tensor(result, 7),
        keepalive=_result_keepalive(result, 8),
    )


def _localcta_replace_col_with_paired_rht(
    base: '_TKQuantized',
    activation_bf16: torch.Tensor,
    *,
    path: str,
) -> '_TKQuantized':
    """Keep ``base`` row payload and attach a v4-opt activation RHT column."""
    if not _localcta_require_paired_col_rht(path):
        return base
    transformed = _fast_quantize_localcta_v4_opt(
        activation_bf16,
        nvfp4_role="activation",
    )
    base._tk_col = transformed._tk_col
    base._tk_col_chunk_sg = transformed._tk_col_chunk_sg
    base._keepalive = (*base._keepalive, *transformed._keepalive)
    return base


def _localcta_silu_quantize_split_for_gemm(tk_q, h1_raw, h3):
    """Fused SwiGLU row plus a paired-RHT W2-wgrad column when requested."""
    path = "localCTA FFN W2 activation producer"
    needs_paired_col_rht = _localcta_require_paired_col_rht(path)
    if needs_paired_col_rht and tk_q.supports_silu_paired_col_rht(h1_raw, h3):
        _trace_backend_choice(
            "localcta_rht_w2_carrier",
            "native_fused_paired",
        )
        return _localcta_quantized_from_result(
            tk_q.tk_silu_quantize_split_for_gemm_paired_col_rht(h1_raw, h3)
        )

    result = tk_q.tk_silu_quantize_split_for_gemm(h1_raw, h3)
    quantized = _localcta_quantized_from_result(result)
    if not needs_paired_col_rht:
        return quantized

    te_fused = _get_te_fused()
    if not hasattr(te_fused, 'fused_silu_mul_bf16_out_no_amax'):
        raise RuntimeError(
            "paired localCTA FFN W2 RHT requires the exact BF16 SwiGLU output producer"
        )
    _trace_backend_choice(
        "localcta_rht_w2_carrier",
        "python_two_pass_fallback",
    )
    h_bf16 = torch.empty_like(h1_raw)
    te_fused.fused_silu_mul_bf16_out_no_amax(h1_raw, h3, h_bf16)
    return _localcta_replace_col_with_paired_rht(
        quantized,
        h_bf16,
        path=path,
    )


_tk_scale_swizzle_idx = None
_last_qkv_forward_debug_payload = None


def _clear_last_qkv_forward_debug_payload():
    global _last_qkv_forward_debug_payload
    _last_qkv_forward_debug_payload = None


def _set_last_qkv_forward_debug_payload(payload: dict):
    global _last_qkv_forward_debug_payload
    _last_qkv_forward_debug_payload = payload


def _get_last_qkv_forward_debug_payload(clear: bool = False):
    payload = _last_qkv_forward_debug_payload
    if clear:
        _clear_last_qkv_forward_debug_payload()
    return payload


def _apply_tk_scale_swizzle(tiles: torch.Tensor) -> torch.Tensor:
    global _tk_scale_swizzle_idx
    if _tk_scale_swizzle_idx is None or _tk_scale_swizzle_idx.device != tiles.device:
        idx = torch.empty(512, dtype=torch.long, device=tiles.device)
        for row in range(128):
            for k in range(4):
                dst = (row % 32) * 16 + (row // 32) * 4 + k
                src = row * 4 + k
                idx[dst] = src
        _tk_scale_swizzle_idx = idx
    flat = tiles.reshape(-1, 512)
    return flat[:, _tk_scale_swizzle_idx].reshape(*tiles.shape)


def _make_te_rowonly_quantizer():
    quantizer = _make_nvfp4_quantizer_compat(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    quantizer.optimize_for_gemm = False
    return quantizer


def use_tk_v5_2d_weight_quant() -> bool:
    """Use one orientation-consistent 16x16 NVFP4 encoding per v5 weight."""
    value = os.environ.get("USE_TK_V5_2D_WEIGHT_QUANT")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    # Keep the diagnostic spelling as a temporary compatibility alias.
    return _env_flag("USE_TK_DEBUG_TE_2D_WEIGHT_QUANT", False)


def use_tk_localcta_2d_weight_quant() -> bool:
    """Use one orientation-consistent 16x16 NVFP4 encoding per localCTA weight."""
    return _env_flag("USE_TK_LOCALCTA_2D_WEIGHT_QUANT", False)


def _make_te_2d_weight_quantizer():
    quantizer = _make_nvfp4_quantizer_compat(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
        with_random_sign_mask=False,
        encode_centric=use_nvfp4_encode_centric(),
    )
    # The adapter below performs the scale-layout conversion expected by TK.
    quantizer.optimize_for_gemm = False
    return quantizer


def _te_payload_to_tk_row(data: torch.Tensor, scale_inv: torch.Tensor, amax: torch.Tensor, rows: int, cols: int):
    fp4x2 = data.view(torch.float4_e2m1fn_x2)
    ntm = rows // 128
    ntk = cols // 64
    scales_flat = scale_inv.contiguous().view(torch.uint8)
    scales_tiled = scales_flat.reshape(ntm, 128, ntk, 4).permute(0, 2, 1, 3).contiguous()
    sc = _apply_tk_scale_swizzle(scales_tiled).reshape(ntm, ntk, 512).view(torch.float8_e4m3fn)
    sg = amax / (6.0 * 448.0)
    return fp4x2, sc, sg


def _te_rowwise_to_tk_row(nvfp4_tensor, M: int, K: int):
    return _te_payload_to_tk_row(
        nvfp4_tensor._rowwise_data,
        nvfp4_tensor._rowwise_scale_inv,
        nvfp4_tensor._amax_rowwise,
        M,
        K,
    )


def _te_nvfp4_to_tk_quantized(nvfp4_tensor, M: int, K: int) -> '_TKQuantized':
    from .nvfp4_2d_weight_adapter import swizzle_2d_nvfp4_scales_for_tk

    row_sc, col_sc = swizzle_2d_nvfp4_scales_for_tk(
        nvfp4_tensor._rowwise_scale_inv,
        nvfp4_tensor._columnwise_scale_inv,
        M,
        K,
    )
    row_fp4 = nvfp4_tensor._rowwise_data.view(torch.float4_e2m1fn_x2)
    col_fp4 = nvfp4_tensor._columnwise_data.view(torch.float4_e2m1fn_x2)
    row_sc = row_sc.view(torch.float8_e4m3fn)
    col_sc = col_sc.view(torch.float8_e4m3fn)
    row_sg = nvfp4_tensor._amax_rowwise / (6.0 * 448.0)
    col_sg = nvfp4_tensor._amax_columnwise / (6.0 * 448.0)
    return _TKQuantized(
        row_fp4,
        row_sc,
        row_sg,
        col_fp4,
        col_sc,
        col_sg,
        keepalive=(nvfp4_tensor,),
    )


def _fast_quantize_te_2d_weight_swizzled(tensor: torch.Tensor) -> '_TKQuantized':
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape
    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"TE 2D weight quantization requires 128-aligned inputs, got {(M, K)}"
        )
    quantized = _make_te_2d_weight_quantizer().quantize(tensor)
    return _te_nvfp4_to_tk_quantized(quantized, M, K)


def _fast_quantize_v5_2d_weight_swizzled(
    tensor: torch.Tensor,
) -> '_TKQuantized':
    """Quantize a v5 weight once into orientation-consistent TK payloads."""
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape
    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"native v5 2D weight quantization requires 128-aligned inputs, got {(M, K)}"
        )

    from .tk_gemm import _get_tk_quant_for_gemm

    tk_q = _get_tk_quant_for_gemm()
    producer = getattr(tk_q, "tk_quantize_weight_2d", None)
    if producer is None:
        raise RuntimeError(
            "USE_TK_V5_2D_WEIGHT_QUANT=1 requires the native v5 "
            "tk_quantize_weight_2d runtime symbol"
        )
    result = producer(tensor)
    if len(result) < 6:
        raise RuntimeError(
            "native v5 2D weight producer returned an incomplete TK contract"
        )
    return _TKQuantized(
        result[0], result[1], result[4],
        result[2], result[3], result[5],
        keepalive=_result_keepalive(result, 6),
    )


def _fast_quantize_localcta_2d_weight_swizzled(
    tensor: torch.Tensor,
) -> '_TKQuantized':
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape
    if M % 256 != 0 or K % 256 != 0:
        raise RuntimeError(
            "native localCTA 2D weight quantization requires 256-aligned "
            f"inputs, got {(M, K)}"
        )
    from .tk_gemm import _get_tk_quant_for_gemm

    result = _get_tk_quant_for_gemm().tk_quantize_weight_2d(tensor)
    # The native producer folds each 128x128 tile's decode scale into the
    # prepared block scales, then fills its scale workspace with ones. Expose
    # those consumed scales in the compact outer-SG shape required by the v4
    # GEMM contract instead of forwarding the producer's 2D workspace shape.
    row_sg = result[4].reshape(-1)[: M // 256]
    col_sg = result[5].reshape(-1)[: K // 256]
    return _TKQuantized(
        result[0], result[1], row_sg,
        result[2], result[3], col_sg,
        keepalive=_result_keepalive(result, 6),
    )


def _const_sg_grid(sg: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    sg_scalar = sg.to(dtype=torch.float32).reshape(1, 1)
    return sg_scalar.expand(rows // 128, cols // 128).contiguous()


def _fold_sg_into_prepared_sc(sc_raw: torch.Tensor, sg, rows: int, cols: int) -> torch.Tensor:
    if torch.is_tensor(sg) and sg.dim() == 2:
        sg_grid = sg.to(torch.float32)
    elif torch.is_tensor(sg):
        sg_grid = _const_sg_grid(sg.reshape(-1)[0], rows, cols)
    else:
        raise TypeError(f"expected tensor sg, got {type(sg)!r}")
    sg_prepared = sg_grid.repeat_interleave(2, dim=1).unsqueeze(-1)
    return (sc_raw.float() * sg_prepared).contiguous().to(torch.float8_e4m3fn)


def _as_contiguous_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Return a BF16 contiguous view with no-op fast path for the common case."""
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _as_param_grad_dtype(grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Return parameter-gradient storage that autograd can own past this call.

    Several fused RMSNorm backward paths write ``dgamma`` into shape-global
    scratch.  Returning that tensor directly lets the next layer or gradient
    accumulation microbatch overwrite a gradient still owned by autograd or
    FSDP.  A dtype conversion already allocates; the no-conversion path needs
    an explicit (small) copy.
    """
    if grad.dtype == param.dtype:
        return grad.clone(memory_format=torch.contiguous_format)
    return grad.to(param.dtype)


def _safe_trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """CUDA bf16/fp16 trunc_normal_ can produce corrupted tails; initialize via fp32."""
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        _safe_trunc_normal_(tensor.to_local(), mean=mean, std=std, a=a, b=b)
        return tensor
    if tensor.is_cuda and tensor.dtype in (torch.bfloat16, torch.float16):
        with torch.no_grad():
            tmp = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
            if std > 0 and (mean - a) / std >= 8.0 and (b - mean) / std >= 8.0:
                tmp.normal_(mean=mean, std=std)
            else:
                nn.init.trunc_normal_(tmp, mean=mean, std=std, a=a, b=b)
            tensor.copy_(tmp.to(dtype=tensor.dtype))
        return tensor
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def _allow_v4_qkv_strided_grad_quant() -> bool:
    if use_tk_localcta_v4_strict_path():
        return False
    return os.environ.get('USE_TK_LOCALCTA_V4_QKV_STRIDED_GRAD_QUANT', '1') == '1'


def _as_qkv_grad_bf16_for_quant(tensor: torch.Tensor) -> torch.Tensor:
    """Keep row-strided QKV grads as views when the v4 producer can consume them."""
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if (
        _allow_v4_qkv_strided_grad_quant()
        and get_tk_localcta_variant() == 'v4'
        and tensor.is_cuda
        and tensor.dim() == 2
        and tensor.stride(1) == 1
    ):
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _maybe_print_qkv_grad_layouts(
    label: str,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    debug_name: Optional[str],
) -> None:
    if os.environ.get('USE_TK_QKV_DEBUG_LAYOUTS', '0') != '1':
        return
    count = getattr(_FusedQKVFunction_TK, '_layout_debug_count', 0)
    limit = int(os.environ.get('USE_TK_QKV_DEBUG_LAYOUTS_LIMIT', '8'))
    if count >= limit:
        return
    setattr(_FusedQKVFunction_TK, '_layout_debug_count', count + 1)

    def _desc(t: torch.Tensor) -> str:
        base = getattr(t, '_base', None)
        return (
            f"shape={tuple(t.shape)} stride={tuple(t.stride())} "
            f"dtype={t.dtype} contig={t.is_contiguous()} "
            f"storage_offset={t.storage_offset()} base_numel={base.numel() if base is not None else None}"
        )

    print(
        f"[TK QKV LAYOUT] {label} name={debug_name} "
        f"q({_desc(grad_q)}) k({_desc(grad_k)}) v({_desc(grad_v)})",
        file=sys.stderr,
        flush=True,
    )


def _is_zero_stride_scalar_bf16_grad(tensor: torch.Tensor) -> bool:
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


def _fast_quantize_tk_localcta_nhsd_wo(tensor: torch.Tensor) -> '_TKQuantized':
    """Quantize flash-attention output [B,H,S,D] as logical WO input [B*S,H*D]."""
    _check_nvfp4_native_extras_supported("activation", "localCTA/v4 NHSD WO quantizer")
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant
    if get_tk_localcta_variant() != 'v4':
        raise RuntimeError("NHSD WO quantization is only implemented for localCTA v4")
    tk_q = _get_tk_quant_for_gemm()
    if not hasattr(tk_q, 'tk_quantize_nhsd_wo_for_gemm'):
        raise RuntimeError("localCTA v4 quant module lacks tk_localcta_quantize_nhsd_wo_for_gemm")
    result = tk_q.tk_quantize_nhsd_wo_for_gemm(tensor, use_nvfp4_encode_centric())
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], result[5])


def _fast_quantize_tk_nhsd_wo(tensor: torch.Tensor) -> '_TKQuantized':
    """Regular-TK v5 global-amax quantization of [B,H,S,D] as logical [B*S,H*D]."""
    _check_nvfp4_native_extras_supported("activation", "regular-TK NHSD WO quantizer")
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.dim() != 4:
        raise RuntimeError(f"regular-TK NHSD WO quantizer expects [B,H,S,D], got {tuple(tensor.shape)}")
    tk_q = _get_tk_quant()
    if not hasattr(tk_q, 'tk_quantize_nhsd_wo_for_gemm'):
        raise RuntimeError("regular TK quant module lacks tk_quantize_nhsd_wo_for_gemm")
    result = tk_q.tk_quantize_nhsd_wo_for_gemm(
        tensor,
        True,
        use_nvfp4_encode_centric(),
    )
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], result[5])


def _fast_quantize_localcta_v4_opt(
    tensor: torch.Tensor,
    nvfp4_role: str = "activation",
    persistent_rng_state: torch.Tensor | None = None,
) -> '_TKQuantized':
    """Native localCTA-v4 row/col producer with NVFP4 SR/scale-SR/RHT controls."""
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape
    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"localCTA v4 opt quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )

    from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant

    if get_tk_localcta_variant() != 'v4':
        raise NotImplementedError(
            "Native NVFP4 SR/RHT is currently implemented only for the localCTA v4 producer."
        )

    tk_q = _get_tk_quant_for_gemm()
    if not hasattr(tk_q, 'tk_quantize_for_gemm_opt'):
        raise RuntimeError(
            "localCTA v4 quant module lacks tk_localcta_quantize_for_gemm_opt; "
            "rebuild the localCTA_epilogue_v3 extension."
        )

    role = _normalize_nvfp4_role(nvfp4_role)
    data_sr = use_nvfp4_data_stochastic_rounding_for_role(role)
    scale_sr = use_nvfp4_scale_stochastic_rounding_for_role(role)
    rht_axes = _nvfp4_native_rht_axes_for_role(role)
    random_sign = use_nvfp4_rht_for_role(role) and _nvfp4_rht_random_sign_mask()

    grad_sr_axes = _nvfp4_grad_sr_axes() if role == "grad" and data_sr else "both"
    args = (
        tensor,
        True,
        use_nvfp4_encode_centric(),
        data_sr and grad_sr_axes != "none",
        scale_sr,
        rht_axes,
        random_sign,
        _nvfp4_rng_seed(),
        _nvfp4_rng_subsequence_base(),
        grad_sr_axes,
    )
    # Preserve compatibility with extensions built before the explicit-state
    # ABI when checkpointed SR is not active.  A live manager always supplies
    # the final state argument and therefore fails loudly against a stale ABI.
    result = tk_q.tk_quantize_for_gemm_opt(
        *args,
        *(() if persistent_rng_state is None else (persistent_rng_state,)),
    )
    sg_t = result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4]
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], sg_t,
                        row_chunk_sg=_optional_result_tensor(result, 6),
                        col_chunk_sg=_optional_result_tensor(result, 7))


def _fast_quantize_localcta_v4_split2_paired_rht_carrier(
    input0: torch.Tensor,
    input1: torch.Tensor,
    *,
    persistent_rng_state: torch.Tensor | None,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[object, ...],
]:
    """Quantize two derivative tensors through one causal SR/RHT carrier.

    One horizontal concatenation gives both arms the same quantizer and uses
    exactly one checkpointed SR reservation per logical FFN derivative
    producer.  The row outer scale is intentionally shared by the two dgrad
    views; column outer scales remain independently sliceable for wgrad.  The
    common carrier is a documented route change held fixed across the pair.
    """
    if not use_tk_localcta_paired_rht_carrier():
        raise RuntimeError("paired localCTA split2 carrier was not enabled")
    _validate_tk_localcta_paired_rht_split2_contract(persistent_rng_state)
    input0 = _as_contiguous_bf16(input0)
    input1 = _as_contiguous_bf16(input1)
    if input0.dim() != 2 or input1.dim() != 2 or input0.shape != input1.shape:
        raise RuntimeError(
            "paired localCTA split2 carrier requires equal contiguous 2D inputs"
        )
    M, width = input0.shape
    if M % 256 or width % 256:
        raise RuntimeError(
            "paired localCTA split2 carrier requires 256-aligned dimensions, "
            f"got {(M, width)}"
        )

    combined = torch.cat((input0, input1), dim=1)
    quantized = _fast_quantize_localcta_v4_opt(
        combined,
        nvfp4_role="grad",
        persistent_rng_state=persistent_rng_state,
    )
    row_fp4_full, row_sc_full, row_sg = quantized._tk_row
    col_fp4_full, col_sc_full, col_sg_full = quantized._tk_col

    row_fp4s = [
        _narrow_packed_fp4_contiguous(
            row_fp4_full, 1, index * (width // 2), width // 2
        )
        for index in range(2)
    ]
    row_scs = [
        row_sc_full.narrow(1, index * (width // 64), width // 64).contiguous()
        for index in range(2)
    ]
    row_sgs = [row_sg, row_sg]
    col_fp4s = [
        _narrow_packed_fp4_contiguous(col_fp4_full, 0, index * width, width)
        for index in range(2)
    ]
    col_scs = [
        col_sc_full.narrow(0, index * (width // 128), width // 128).contiguous()
        for index in range(2)
    ]
    col_sgs = [
        col_sg_full.narrow(1, index * (width // 256), width // 256).contiguous()
        for index in range(2)
    ]
    keepalive: tuple[object, ...] = (combined, quantized)
    return (
        row_fp4s,
        row_scs,
        row_sgs,
        col_fp4s,
        col_scs,
        col_sgs,
        col_fp4_full,
        col_sc_full,
        col_sg_full,
        keepalive,
    )


def _fast_rmsnorm_quantize_localcta_v4_opt(
    tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
    nvfp4_role: str = "activation",
    prefer_row_prepared_col_outer: bool = False,
    encode_centric_override: Optional[bool] = None,
    separate_bf16_final_sg: bool = False,
) -> tuple['_TKQuantized', torch.Tensor]:
    """Native localCTA-v4 RMSNorm + row/col producer with NVFP4 SR/RHT controls."""
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    norm_weight = _as_contiguous_bf16(norm_weight)
    M, K = tensor.shape
    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"localCTA v4 RMSNorm opt quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )
    if norm_weight.dim() != 1 or norm_weight.shape[0] != K:
        raise RuntimeError(
            f"localCTA v4 RMSNorm opt quant expected norm_weight shape=({K},), got {tuple(norm_weight.shape)}"
        )

    from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant

    if get_tk_localcta_variant() != 'v4':
        raise NotImplementedError(
            "Native RMSNorm NVFP4 SR/RHT is currently implemented only for the localCTA v4 producer."
        )

    tk_q = _get_tk_quant_for_gemm()
    encode_centric = (
        use_nvfp4_encode_centric()
        if encode_centric_override is None
        else bool(encode_centric_override)
    )
    use_row_prepared_col_outer = (
        prefer_row_prepared_col_outer
        and use_tk_localcta_v4_row_prepared_rmsnorm_quant()
        and M % 256 == 0
        and K % 256 == 0
        and hasattr(tk_q, 'tk_rmsnorm_quantize_for_gemm_row_prepared_col_outer')
    )
    if separate_bf16_final_sg:
        supported = (
            not prefer_row_prepared_col_outer
            and M % 256 == 0
            and K % 256 == 0
            and not (
                use_nvfp4_rht_for_role(nvfp4_role)
                and _nvfp4_rht_random_sign_mask()
            )
        )
        if not supported:
            raise RuntimeError(
                "localCTA v4 separate BF16/final-SG producer requires the "
                "production final-SG contract"
            )
        if not (
            hasattr(tk_q, 'tk_rmsnorm_to_bf16')
            and hasattr(tk_q, 'tk_quantize_for_gemm_final_sg_opt')
        ):
            raise RuntimeError(
                "localCTA v4 separate BF16/final-SG producer requires both "
                "native ABIs"
            )
        normed, inv_rms = tk_q.tk_rmsnorm_to_bf16(
            tensor, norm_weight, float(epsilon)
        )
        result = tk_q.tk_quantize_for_gemm_final_sg_opt(
            normed,
            True,
            encode_centric,
            use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
            use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
            _nvfp4_native_rht_axes_for_role(nvfp4_role),
            use_nvfp4_rht_for_role(nvfp4_role)
            and _nvfp4_rht_random_sign_mask(),
            _nvfp4_rng_seed(),
            _nvfp4_rng_subsequence_base(),
        )
        sg_t = (
            result[5]
            if len(result) > 5
            and torch.is_tensor(result[5])
            and result[5].numel() > 0
            else result[4]
        )
        quantized = _TKQuantized(
            result[0],
            result[1],
            result[4],
            result[2],
            result[3],
            sg_t,
            row_chunk_sg=_optional_result_tensor(result, 6),
            col_chunk_sg=_optional_result_tensor(result, 7),
            keepalive=(normed,),
        )
        return quantized, inv_rms

    if use_row_prepared_col_outer:
        result = tk_q.tk_rmsnorm_quantize_for_gemm_row_prepared_col_outer(
            tensor,
            norm_weight,
            float(epsilon),
            True,
            encode_centric,
            use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
            use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
            _nvfp4_native_rht_axes_for_role(nvfp4_role),
            use_nvfp4_rht_for_role(nvfp4_role) and _nvfp4_rht_random_sign_mask(),
            _nvfp4_rng_seed(),
            _nvfp4_rng_subsequence_base(),
        )
        quantized = _TKQuantized(
            result[0], result[1], result[2],
            result[3], result[4], result[5],
        )
        return quantized, result[6]

    use_final_sg = (
        not prefer_row_prepared_col_outer
        and
        M % 256 == 0
        and K % 256 == 0
        and not (use_nvfp4_rht_for_role(nvfp4_role) and _nvfp4_rht_random_sign_mask())
        and hasattr(tk_q, 'tk_rmsnorm_quantize_for_gemm_final_sg_opt')
    )
    if use_final_sg:
        method = tk_q.tk_rmsnorm_quantize_for_gemm_final_sg_opt
    elif hasattr(tk_q, 'tk_rmsnorm_quantize_for_gemm_opt'):
        method = tk_q.tk_rmsnorm_quantize_for_gemm_opt
    else:
        raise RuntimeError(
            "localCTA v4 quant module lacks RMSNorm opt quantizer; "
            "rebuild the localCTA_epilogue_v3 extension."
        )

    result = method(
        tensor,
        norm_weight,
        float(epsilon),
        True,
        encode_centric,
        use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
        use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
        _nvfp4_native_rht_axes_for_role(nvfp4_role),
        use_nvfp4_rht_for_role(nvfp4_role) and _nvfp4_rht_random_sign_mask(),
        _nvfp4_rng_seed(),
        _nvfp4_rng_subsequence_base(),
    )
    sg_t = result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4]
    quantized = _TKQuantized(
        result[0], result[1], result[4],
        result[2], result[3], sg_t,
        row_chunk_sg=_optional_result_tensor(result, 7),
        col_chunk_sg=_optional_result_tensor(result, 8),
    )
    return quantized, result[6]


def _fast_sqrelu_quantize_localcta_v4(tensor: torch.Tensor, nvfp4_role: str = "activation") -> Optional['_TKQuantized']:
    if not use_tk_localcta_v4_fused_sqrelu_quant():
        return None
    extras_enabled = _nvfp4_quantizer_extras_enabled(nvfp4_role)
    if extras_enabled:
        rht_axes = _nvfp4_native_rht_axes_for_role(nvfp4_role)
        if rht_axes not in ("none", "off", "0", "row", "col", "both"):
            return None
        if use_nvfp4_rht_for_role(nvfp4_role) and _nvfp4_rht_random_sign_mask():
            return None
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape
    if M % 128 != 0 or K % 128 != 0:
        return None

    from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant, use_tk_localcta_v4_strict_path

    if get_tk_localcta_variant() != 'v4':
        return None
    if not (use_tk_localcta_v4_fast_prepared_producer_for_shape(M, K) or use_tk_localcta_v4_strict_path()):
        return None
    tk_q = _get_tk_quant_for_gemm()
    if use_tk_localcta_v4_row_prepared_col_outer():
        if (
            nvfp4_role != "grad"
            and M % 256 == 0
            and K % 256 == 0
            and hasattr(tk_q, 'tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer')
        ):
            if (
                extras_enabled
                and nvfp4_role == "activation"
                and rht_axes in ("row", "none", "off", "0")
                and use_tk_localcta_v4_sqrelu_delay_col_quant()
                and hasattr(tk_q, 'tk_localcta_sqrelu_quantize_row_only_prepared')
            ):
                result = tk_q.tk_localcta_sqrelu_quantize_row_only_prepared(
                    tensor,
                    use_nvfp4_encode_centric(),
                    use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
                    use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
                    rht_axes,
                    False,
                    _nvfp4_rng_seed(),
                    _nvfp4_rng_subsequence_base(),
                )
                empty_fp4 = torch.empty((0,), dtype=torch.float4_e2m1fn_x2, device=tensor.device)
                empty_sc = torch.empty((0,), dtype=torch.float8_e4m3fn, device=tensor.device)
                empty_sg = torch.empty((0,), dtype=torch.float32, device=tensor.device)
                return _TKQuantized(
                    result[0], result[1], result[2],
                    empty_fp4, empty_sc, empty_sg,
                )
            if extras_enabled:
                result = tk_q.tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer(
                    tensor,
                    use_nvfp4_encode_centric(),
                    use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
                    use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
                    rht_axes,
                    False,
                    _nvfp4_rng_seed(),
                    _nvfp4_rng_subsequence_base(),
                )
            else:
                result = tk_q.tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer(
                    tensor, use_nvfp4_encode_centric()
                )
            return _TKQuantized(
                result[0], result[1], result[2],
                result[3], result[4], result[5],
            )
        return None
    if not hasattr(tk_q, 'tk_localcta_sqrelu_quantize_for_gemm_prepared'):
        return None

    if extras_enabled:
        result = tk_q.tk_localcta_sqrelu_quantize_for_gemm_prepared(
            tensor,
            use_nvfp4_encode_centric(),
            use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
            use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
            rht_axes,
            False,
            _nvfp4_rng_seed(),
            _nvfp4_rng_subsequence_base(),
        )
    else:
        result = tk_q.tk_localcta_sqrelu_quantize_for_gemm_prepared(
            tensor, use_nvfp4_encode_centric()
        )
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], result[5])


def _materialize_sqrelu_col_localcta_v4(
    q: '_TKQuantized',
    h1_raw: torch.Tensor,
) -> '_TKQuantized':
    col_fp4 = q._tk_col[0]
    if torch.is_tensor(col_fp4) and col_fp4.numel() > 0:
        return q
    from .tk_gemm import _get_tk_quant_for_gemm

    tk_q = _get_tk_quant_for_gemm()
    if not hasattr(tk_q, 'tk_localcta_sqrelu_quantize_col_only_raw_outer'):
        raise RuntimeError("localCTA quant module lacks square-ReLU col-only producer")
    h1_raw = _as_contiguous_bf16(h1_raw)
    if h1_raw.dim() > 2:
        h1_raw = h1_raw.reshape(-1, h1_raw.shape[-1])
    result = tk_q.tk_localcta_sqrelu_quantize_col_only_raw_outer(
        h1_raw,
        use_nvfp4_encode_centric(),
    )
    q._tk_col = (result[0], result[1], result[2])
    return q


def _fast_sqrelu_deriv_quantize_localcta_v4(
    dh: torch.Tensor,
    h1_raw: torch.Tensor,
    nvfp4_role: str = "grad",
) -> Optional['_TKQuantized']:
    if not use_tk_localcta_v4_fused_sqrelu_deriv_quant():
        return None
    extras_enabled = _nvfp4_quantizer_extras_enabled(nvfp4_role)
    if extras_enabled:
        rht_axes = _nvfp4_native_rht_axes_for_role(nvfp4_role)
        if rht_axes not in ("none", "off", "0", "row", "col", "both"):
            return None
        if use_nvfp4_rht_for_role(nvfp4_role) and _nvfp4_rht_random_sign_mask():
            return None
    dh = _as_contiguous_bf16(dh)
    h1_raw = _as_contiguous_bf16(h1_raw)
    if dh.dim() > 2:
        dh = dh.reshape(-1, dh.shape[-1])
    if h1_raw.dim() > 2:
        h1_raw = h1_raw.reshape(-1, h1_raw.shape[-1])
    if dh.shape != h1_raw.shape:
        return None
    M, K = dh.shape
    if M % 128 != 0 or K % 128 != 0:
        return None

    from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant, use_tk_localcta_v4_strict_path

    if get_tk_localcta_variant() != 'v4':
        return None
    if not (use_tk_localcta_v4_fast_prepared_producer_for_shape(M, K) or use_tk_localcta_v4_strict_path()):
        return None
    if M % 256 != 0 or K % 256 != 0:
        return None
    tk_q = _get_tk_quant_for_gemm()
    if not hasattr(tk_q, 'tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer'):
        return None

    if extras_enabled:
        result = tk_q.tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer(
            dh,
            h1_raw,
            use_nvfp4_encode_centric(),
            use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
            use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
            rht_axes,
            False,
            _nvfp4_rng_seed(),
            _nvfp4_rng_subsequence_base(),
        )
    else:
        result = tk_q.tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer(
            dh, h1_raw, use_nvfp4_encode_centric()
        )
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], result[5])


def _fast_sqrelu_quantize_tk_regular_opt(
    tensor: torch.Tensor,
    nvfp4_role: str = "activation",
) -> Optional['_TKQuantized']:
    if not use_tk_quant() or use_tk_localcta():
        return None
    if not use_nvfp4_encode_centric():
        return None
    if not use_tk_regular_fused_sqrelu_quant(nvfp4_role):
        return None
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2:
        return None
    M, K = tensor.shape
    if M % 128 != 0 or K % 128 != 0:
        return None
    tk_q = _get_tk_quant()
    if not hasattr(tk_q, 'tk_sqrelu_quantize_for_gemm_opt'):
        return None

    extras_enabled = _nvfp4_quantizer_extras_enabled(nvfp4_role)
    data_sr = use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role) if extras_enabled else False
    scale_sr = use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role) if extras_enabled else False
    rht_axes = _nvfp4_native_rht_axes_for_role(nvfp4_role) if extras_enabled else "none"
    if rht_axes not in ("none", "off", "0", "row", "col", "both"):
        return None
    if data_sr or scale_sr:
        return None
    if use_nvfp4_rht_for_role(nvfp4_role) and _nvfp4_rht_random_sign_mask():
        return None

    result = tk_q.tk_sqrelu_quantize_for_gemm_opt(
        tensor,
        True,
        use_nvfp4_encode_centric(),
        False,
        False,
        "none" if rht_axes in ("off", "0") else rht_axes,
        False,
        _nvfp4_rng_seed(),
        _nvfp4_rng_subsequence_base(),
    )
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], result[5])


def _fast_sqrelu_deriv_quantize_tk_regular_opt(
    dh: torch.Tensor,
    h1_raw: torch.Tensor,
    nvfp4_role: str = "grad",
) -> Optional['_TKQuantized']:
    if not use_tk_quant() or use_tk_localcta():
        return None
    if not use_nvfp4_encode_centric():
        return None
    if not use_tk_regular_fused_sqrelu_deriv_quant(nvfp4_role):
        return None
    dh = _as_contiguous_bf16(dh)
    h1_raw = _as_contiguous_bf16(h1_raw)
    if dh.dim() > 2:
        dh = dh.reshape(-1, dh.shape[-1])
    if h1_raw.dim() > 2:
        h1_raw = h1_raw.reshape(-1, h1_raw.shape[-1])
    if dh.dim() != 2 or dh.shape != h1_raw.shape:
        return None
    M, K = dh.shape
    if M % 128 != 0 or K % 128 != 0:
        return None
    tk_q = _get_tk_quant()
    if not hasattr(tk_q, 'tk_sqrelu_deriv_quantize_for_gemm_opt'):
        return None

    extras_enabled = _nvfp4_quantizer_extras_enabled(nvfp4_role)
    data_sr = use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role) if extras_enabled else False
    scale_sr = use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role) if extras_enabled else False
    rht_axes = _nvfp4_native_rht_axes_for_role(nvfp4_role) if extras_enabled else "none"
    if rht_axes not in ("none", "off", "0", "row", "col", "both"):
        return None
    if rht_axes in ("col", "both"):
        return None
    if use_nvfp4_rht_for_role(nvfp4_role) and _nvfp4_rht_random_sign_mask():
        return None

    result = tk_q.tk_sqrelu_deriv_quantize_for_gemm_opt(
        dh,
        h1_raw,
        True,
        use_nvfp4_encode_centric(),
        data_sr,
        scale_sr,
        "none" if rht_axes in ("off", "0") else rht_axes,
        False,
        _nvfp4_rng_seed(),
        _nvfp4_rng_subsequence_base(),
    )
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], result[5])


def _fast_quantize_tk_regular_opt(tensor: torch.Tensor, nvfp4_role: str = "activation") -> '_TKQuantized':
    """Native regular-TK producer for NVFP4 payload SR and block-16 RHT experiments."""
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape
    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"regular-TK opt quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )
    tk_q = _get_tk_quant()
    if not hasattr(tk_q, 'tk_quantize_for_gemm_opt'):
        raise RuntimeError(
            "regular TK quant module lacks tk_quantize_for_gemm_opt; rebuild TK_quantisation/nvfp4_v5."
        )
    role = _normalize_nvfp4_role(nvfp4_role)
    data_sr = use_nvfp4_data_stochastic_rounding_for_role(role)
    grad_sr_axes = _nvfp4_grad_sr_axes() if role == "grad" and data_sr else "both"
    result = tk_q.tk_quantize_for_gemm_opt(
        tensor,
        True,
        use_nvfp4_encode_centric(),
        data_sr and grad_sr_axes != "none",
        use_nvfp4_scale_stochastic_rounding_for_role(role),
        _nvfp4_native_rht_axes_for_role(role),
        use_nvfp4_rht_for_role(role) and _nvfp4_rht_random_sign_mask(),
        _nvfp4_rng_seed(),
        _nvfp4_rng_subsequence_base(),
        data_sr_axes=grad_sr_axes,
    )
    sg_t = result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4]
    return _TKQuantized(result[0], result[1], result[4],
                        result[2], result[3], sg_t)


def _can_fast_rmsnorm_quantize_tk_regular_opt(tensor: torch.Tensor, nvfp4_role: str = "activation") -> bool:
    """Return whether the native regular-TK fused RMSNorm extras producer is valid."""
    if (
        not use_tk_quant()
        or use_tk_localcta()
        or not use_tk_ffn_fused_norm_quant()
        or not _nvfp4_quantizer_extras_enabled(nvfp4_role)
        or _nvfp4_native_rht_axes_for_role(nvfp4_role) != "row"
        or _nvfp4_rht_random_sign_mask()
    ):
        return False
    if tensor.dtype != torch.bfloat16:
        return False
    shape = tensor.reshape(-1, tensor.shape[-1]).shape if tensor.dim() > 2 else tensor.shape
    if len(shape) != 2:
        return False
    M, K = shape
    if M % 128 != 0 or K % 128 != 0:
        return False
    return hasattr(_get_tk_quant(), 'tk_fused_norm_quantize_opt')


def _fast_rmsnorm_quantize_tk_regular_opt(
    tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
    nvfp4_role: str = "activation",
) -> tuple['_TKQuantized', torch.Tensor]:
    """Native regular-TK RMSNorm + row-RHT/SR-aware NVFP4 quant producer."""
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    tk_q = _get_tk_quant()
    result = tk_q.tk_fused_norm_quantize_opt(
        tensor,
        _as_contiguous_bf16(norm_weight),
        float(epsilon),
        True,
        use_nvfp4_encode_centric(),
        use_nvfp4_data_stochastic_rounding_for_role(nvfp4_role),
        use_nvfp4_scale_stochastic_rounding_for_role(nvfp4_role),
        _nvfp4_native_rht_axes_for_role(nvfp4_role),
        False,
        _nvfp4_rng_seed(),
        _nvfp4_rng_subsequence_base(),
    )
    quantized = _TKQuantized(
        result[0], result[1], result[4],
        result[2], result[3], result[5],
    )
    return quantized, result[6]


def _tk_quantized_as_result_tuple(q: '_TKQuantized') -> tuple[object, ...]:
    row_fp4, row_sc, row_sg = q._tk_row
    col_fp4, col_sc, col_sg = q._tk_col
    return (
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        row_sg,
        col_sg,
        *q._keepalive,
    )


def _fast_quantize_tk_padded(
    tensor: torch.Tensor,
    padded_rows: int,
    padded_cols: int,
    nvfp4_role: str,
) -> '_TKQuantized':
    """Quantize a logical BF16 matrix directly into a 256-aligned TK extent."""
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    rows, cols = tensor.shape
    if (
        rows % 128 != 0
        or cols % 128 != 0
        or padded_rows % 256 != 0
        or padded_cols % 256 != 0
        or padded_rows < rows
        or padded_cols < cols
    ):
        raise RuntimeError(
            "native padded NVFP4 quantization requires a 128-aligned logical "
            f"shape covered by a 256-aligned target, got {(rows, cols)} -> "
            f"{(padded_rows, padded_cols)}"
        )
    if _nvfp4_quantizer_extras_enabled(nvfp4_role):
        raise RuntimeError(
            "native padded NVFP4 quantization does not yet support SR/RHT extras"
        )

    if use_tk_localcta():
        from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant

        if get_tk_localcta_variant() != "v4":
            raise RuntimeError("native padded localCTA quantization requires v4")
        tk_q = _get_tk_quant_for_gemm()
    else:
        tk_q = _get_tk_quant()
    if not hasattr(tk_q, "tk_quantize_for_gemm_padded"):
        raise RuntimeError(
            "TK quant module lacks its native padded producer; rebuild the "
            "nvfp4_v5/localCTA_v4 extension"
        )
    result = tk_q.tk_quantize_for_gemm_padded(
        tensor,
        padded_rows,
        padded_cols,
        True,
        use_nvfp4_encode_centric(),
    )
    return _TKQuantized(
        result[0], result[1], result[4],
        result[2], result[3], result[5],
        keepalive=_result_keepalive(result, 6),
    )


def _fast_quantize(
    tensor: torch.Tensor,
    quantizer=None,
    tk_swizzle=False,
    use_localcta_override: Optional[bool] = None,
    nvfp4_role: str | None = None,
    persistent_rng_state: torch.Tensor | None = None,
) -> 'NVFP4Tensor':
    """Drop-in replacement for quantizer.quantize(tensor).

    Uses fast_nvfp4_quantize_v2 (C++ bypass + custom fused amax kernel)
    for ~2.2x speedup over TE's Python quantize path.

    When USE_TK_QUANT=1, uses the pre-compiled TK standalone quant module
    instead of TE's JIT-compiled extension for further speedup.

    Args:
        tensor: Input tensor to quantize.
        quantizer: Optional NVFP4Quantizer (cosmetic, wraps result).
        tk_swizzle: Pass True for TK-native format (_TKQuantized).
                    Default False for TE format (NVFP4Tensor).
        nvfp4_role: Optional activation/weight/grad role for SR/RHT policy.

    Returns an NVFP4Tensor (TE path) or _TKQuantized (TK path).
    """
    tensor = _as_contiguous_bf16(tensor)
    # Ensure 2D for the C++ kernel (batch dims flattened)
    orig_shape = tensor.shape
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape
    nvfp4_role = _nvfp4_quantizer_role(quantizer, nvfp4_role)



    # ===== TK swizzle path: pure TK quantize (no TE calls) =====
    if tk_swizzle and M % 128 == 0 and K % 128 == 0:
        localcta_enabled = (
            use_tk_localcta()
            if use_localcta_override is None
            else use_localcta_override
        )
        if (
            nvfp4_role == "weight"
            and (
                (not localcta_enabled and use_tk_v5_2d_weight_quant())
                or (localcta_enabled and use_tk_localcta_2d_weight_quant())
            )
        ):
            if localcta_enabled:
                return _fast_quantize_localcta_2d_weight_swizzled(tensor)
            return _fast_quantize_v5_2d_weight_swizzled(tensor)
        if _nvfp4_quantizer_extras_enabled(nvfp4_role):
            if localcta_enabled:
                return _fast_quantize_localcta_v4_opt(
                    tensor,
                    nvfp4_role=nvfp4_role,
                    persistent_rng_state=persistent_rng_state,
                )
            if use_tk_quant():
                return _fast_quantize_tk_regular_opt(tensor, nvfp4_role=nvfp4_role)
            active_quantizer = quantizer or _make_nvfp4_quantizer_for_role(nvfp4_role)
            return _te_nvfp4_to_tk_quantized(active_quantizer.quantize(tensor), M, K)
        if localcta_enabled and use_tk_localcta_direct_contract():
            return _fast_quantize_tk_standalone_swizzled(tensor, nvfp4_role=nvfp4_role)
        if use_tk_quant() or localcta_enabled:
            encode_centric = use_nvfp4_encode_centric()
            if localcta_enabled:
                from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant, use_tk_localcta_v4_strict_path
                tk_q = _get_tk_quant_for_gemm()
                if (
                    get_tk_localcta_variant() == 'v4'
                    and use_tk_localcta_v4_fast_prepared_producer_for_shape(M, K)
                    and use_tk_localcta_v4_row_prepared_col_outer()
                    and use_tk_localcta_v4_raw_outer_tma_grad()
                    and nvfp4_role == "grad"
                    and hasattr(tk_q, 'tk_quantize_for_gemm_raw_outer_tma')
                    and M % 256 == 0
                    and K % 256 == 0
                ):
                    result = tk_q.tk_quantize_for_gemm_raw_outer_tma(
                        tensor, True, encode_centric
                    )
                    return _TKQuantized(result[0], result[1], result[4],
                                        result[2], result[3], result[5])
                if (
                    get_tk_localcta_variant() == 'v4'
                    and use_tk_localcta_v4_fast_prepared_producer_for_shape(M, K)
                    and use_tk_localcta_v4_row_prepared_col_outer()
                    and nvfp4_role != "grad"
                    and hasattr(tk_q, 'tk_quantize_for_gemm_row_prepared_col_outer')
                    and M % 256 == 0
                    and K % 256 == 0
                ):
                    result = tk_q.tk_quantize_for_gemm_row_prepared_col_outer(
                        tensor, True, encode_centric
                    )
                    return _TKQuantized(
                        result[0], result[1], result[2],
                        result[3], result[4], result[5],
                    )
                if (
                    get_tk_localcta_variant() == 'v4'
                    and (
                        use_tk_localcta_v4_strict_path()
                        or (
                            use_tk_localcta_v4_fast_prepared_producer_for_shape(M, K)
                            and nvfp4_role != "grad"
                        )
                    )
                    and hasattr(tk_q, 'tk_quantize_for_gemm_fast')
                ):
                    result = tk_q.tk_quantize_for_gemm_fast(tensor, True, encode_centric)
                    return _TKQuantized(
                        result[0], result[6], result[4],
                        result[2], result[7], result[5],
                    )
                if (
                    get_tk_localcta_variant() == 'v4'
                    and use_tk_localcta_v4_strict_path()
                    and hasattr(tk_q, 'tk_quantize_for_gemm_direct')
                ):
                    result = tk_q.tk_quantize_for_gemm_direct(tensor, True, encode_centric)
                    sg_t = result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4]
                    return _TKQuantized(
                        result[0], result[1], result[4],
                        result[2], result[3], sg_t,
                        keepalive=_result_keepalive(result, 6),
                    )
            else:
                tk_q = _get_tk_quant()
            result = tk_q.tk_quantize_for_gemm(tensor, True, encode_centric)
            sg_t = result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4]
            return _TKQuantized(
                result[0], result[1], result[4],
                result[2], result[3], sg_t,
                keepalive=_result_keepalive(result, 6),
            )
        else:
            ext = _get_te_fused()
            result = ext.fused_amax_quantize(tensor, True)
            return _TKQuantized(result[0], result[1], result[4],
                                result[2], result[3], result[4])

    # ===== TE quant path (non-swizzle) =====
    if _nvfp4_quantizer_extras_enabled(nvfp4_role):
        active_quantizer = quantizer or _make_nvfp4_quantizer_for_role(nvfp4_role)
        return active_quantizer.quantize(tensor.reshape(orig_shape))

    # Alignment check — fall back to TE Python path if not aligned
    if M % 32 != 0 or K % 32 != 0:
        if quantizer is not None:
            return quantizer.quantize(tensor.reshape(orig_shape))
        return _make_nvfp4_quantizer_compat(
            fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True,
        ).quantize(tensor.reshape(orig_shape))

    if use_fused_te_quant():
        ext = _get_te_fused()
    else:
        ext = _get_fp4_ext()

    if use_fused_te_quant():
        # te_fused path: returns 6 values (no sg), no swizzle/custom_quant support
        fp4, si, fp4_t, si_t, amax, amax_t = ext.fast_nvfp4_quantize_v2(tensor, False)
        tk_swizzle = False  # force TE path
    else:
        cq = use_custom_quant()
        fp4, si, fp4_t, si_t, amax, amax_t, sg = ext.fast_nvfp4_quantize_v2(
            tensor, False, tk_swizzle, cq)

    # TK fast path: skip NVFP4Tensor + NVFP4Quantizer construction entirely
    if tk_swizzle:
        return _TKQuantized(fp4, si, sg, fp4_t, si_t)

    # TE path: full NVFP4Tensor construction
    if quantizer is None:
        quantizer = _make_nvfp4_quantizer_compat(
            fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True,
        )
    result = _make_nvfp4_tensor_compat(
        (M, K),
        tensor.dtype,
        rowwise_data=fp4,
        rowwise_scale_inv=si,
        columnwise_data=fp4_t,
        columnwise_scale_inv=si_t,
        amax_rowwise=amax,
        amax_columnwise=amax_t,
        fp4_dtype=tex.DType.kFloat4E2M1,
        quantizer=quantizer,
    )
    result._with_gemm_swizzled_scales = False
    return result


def _fast_quantize_te_swizzled(tensor: torch.Tensor, nvfp4_role: str = "activation") -> '_TKQuantized':
    """Return `_TKQuantized` with TE rowwise activation quant and TK col payload.

    The forward QKV GEMM only consumes `_tk_row`, so using TE's exact rowwise
    activation quantization isolates the standalone TK activation quantizer as a
    potential numerical bug. `_tk_col` is still sourced from the TK standalone
    quantizer so existing backward/debug paths keep working.
    """
    _check_nvfp4_native_extras_supported(nvfp4_role, "TE-row/TK-col debug quantizer")
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape

    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"TE rowwise TK quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )

    tk_q = _get_tk_quant()
    tk_result = tk_q.tk_quantize_for_gemm(tensor, True)
    tk_col_fp4 = tk_result[2]
    tk_col_sc = tk_result[3]
    tk_col_sg = tk_result[5] if len(tk_result) > 5 and torch.is_tensor(tk_result[5]) and tk_result[5].numel() > 0 else tk_result[4]

    te_row_q = _make_te_rowonly_quantizer()(tensor)
    te_row_fp4, te_row_sc, te_row_sg = _te_rowwise_to_tk_row(te_row_q, M, K)
    return _TKQuantized(
        te_row_fp4, te_row_sc, te_row_sg, tk_col_fp4, tk_col_sc, tk_col_sg,
        keepalive=_result_keepalive(tk_result, 6),
    )


def _fast_quantize_te_rowonly_swizzled(tensor: torch.Tensor, nvfp4_role: str = "activation") -> '_TKQuantized':
    """Return `_TKQuantized` with TE rowwise payload only for forward-only use."""
    _check_nvfp4_native_extras_supported(nvfp4_role, "TE-rowonly debug quantizer")
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape

    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"TE rowonly TK quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )

    te_row_q = _make_te_rowonly_quantizer()(tensor)
    te_row_fp4, te_row_sc, te_row_sg = _te_rowwise_to_tk_row(te_row_q, M, K)
    return _TKQuantized(
        te_row_fp4, te_row_sc, te_row_sg,
        te_row_fp4, te_row_sc, te_row_sg,
    )


def _fast_quantize_tk_standalone_swizzled(tensor: torch.Tensor, nvfp4_role: str = "activation") -> '_TKQuantized':
    """Return `_TKQuantized` from standalone TK quant even when localCTA is enabled."""
    _check_nvfp4_native_extras_supported(nvfp4_role, "standalone TK quantizer")
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape

    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"TK standalone swizzled quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )

    tk_q = _get_tk_quant()
    result = tk_q.tk_quantize_for_gemm(tensor, True, use_nvfp4_encode_centric())
    sg_t = result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4]
    return _TKQuantized(
        result[0], result[1], result[4], result[2], result[3], sg_t,
        keepalive=_result_keepalive(result, 6),
    )


def _fast_quantize_localcta_regular_hybrid(tensor: torch.Tensor, nvfp4_role: str = "activation") -> '_TKQuantized':
    """Single-pass localCTA regular quant with prepared row and raw col payloads."""
    _check_nvfp4_native_extras_supported(nvfp4_role, "localCTA/v4 regular hybrid quantizer")
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape

    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"localCTA hybrid regular quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )

    from .tk_gemm import _get_tk_quant_for_gemm

    tk_q = _get_tk_quant_for_gemm()
    result = tk_q.tk_quantize_for_gemm_fast(tensor, True, use_nvfp4_encode_centric())
    return _TKQuantized(
        result[0], result[6], result[4],
        result[2], result[3], result[5],
    )


def _fast_quantize_tk_standalone_localcta_prepared(tensor: torch.Tensor, nvfp4_role: str = "activation") -> '_TKQuantized':
    """Build localCTA-prepared row/col payloads from standalone TK quantization."""
    _check_nvfp4_native_extras_supported(nvfp4_role, "standalone TK localCTA-prepared quantizer")
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape

    x_tk = _fast_quantize_tk_standalone_swizzled(tensor, nvfp4_role=nvfp4_role)
    x_fp4, x_sc_raw, x_sg = x_tk._tk_row
    x_fp4_t, x_sc_t_raw, x_sg_t = x_tk._tk_col
    x_sc_prepared = _fold_sg_into_prepared_sc(x_sc_raw, x_sg, M, K)
    x_sc_t_prepared = _fold_sg_into_prepared_sc(x_sc_t_raw, x_sg_t, K, M)
    return _TKQuantized(
        x_fp4, x_sc_prepared, x_sg,
        x_fp4_t, x_sc_t_prepared, x_sg_t,
    )


def _fast_quantize_tk_row_localcta_col_swizzled(tensor: torch.Tensor, nvfp4_role: str = "activation") -> '_TKQuantized':
    """Return TK row contract with localCTA direct col contract."""
    _check_nvfp4_native_extras_supported(nvfp4_role, "TK-row/localCTA-col mixed quantizer")
    tensor = _as_contiguous_bf16(tensor)
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    M, K = tensor.shape

    if M % 128 != 0 or K % 128 != 0:
        raise RuntimeError(
            f"TK/localCTA mixed swizzled quant requires 128-aligned 2D inputs, got shape={(M, K)}"
        )

    from .tk_gemm import _get_tk_quant_for_gemm

    tk_row = _fast_quantize_tk_standalone_swizzled(tensor, nvfp4_role=nvfp4_role)
    tk_q = _get_tk_quant_for_gemm()
    prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
    try:
        localcta = tk_q.tk_quantize_for_gemm_direct(
            tensor,
            True,
            use_tk_qkv_localcta_encode_centric(),
        )
    finally:
        _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
    col_sg = (
        localcta[5]
        if len(localcta) > 5 and torch.is_tensor(localcta[5]) and localcta[5].numel() > 0
        else localcta[4]
    )
    row_fp4, row_sc, row_sg = tk_row._tk_row
    return _TKQuantized(row_fp4, row_sc, row_sg, localcta[2], localcta[3], col_sg)


def _pad_rows_bf16(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"_pad_rows_bf16 expects 2D tensor, got shape={tuple(tensor.shape)}")
    rows = tensor.size(0)
    if rows >= target_rows:
        return tensor
    pad_rows = target_rows - rows
    pad = torch.zeros(pad_rows, tensor.size(1), device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def _qkv_forward_quantize(
    normed: torch.Tensor,
    input_quantizer: NVFP4Quantizer,
    use_localcta: Optional[bool] = None,
):
    """Quantize QKV activations for TK/localCTA forward with debug-selectable source."""
    # Internal-only debug override: drive both TK and localCTA QKV forward from
    # the same TE-rowwise/TK-col contract so trainer probes can isolate whether
    # the remaining failure is activation quantization or downstream GEMM/use.
    if use_tk_qkv_te_act_quant():
        return _fast_quantize_te_swizzled(normed, nvfp4_role="activation")
    if use_tk_qkv_tk_act_quant():
        return _fast_quantize_tk_row_localcta_col_swizzled(normed, nvfp4_role="activation")
    localcta_enabled = use_tk_localcta() if use_localcta is None else use_localcta
    if localcta_enabled and _nvfp4_quantizer_extras_enabled("activation"):
        from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant
        if get_tk_localcta_variant() != 'v4':
            _check_nvfp4_native_extras_supported("activation", "localCTA QKV activation quantizer")
        tk_q = _get_tk_quant_for_gemm()
        prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
        try:
            return _fast_quantize_localcta_v4_opt(normed, nvfp4_role="activation")
        finally:
            _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
    if localcta_enabled and use_tk_localcta_direct_contract():
        return _fast_quantize_tk_standalone_swizzled(normed, nvfp4_role="activation")
    if localcta_enabled:
        if use_tk_qkv_localcta_fast_activation():
            return _fast_quantize_localcta_regular_hybrid(normed, nvfp4_role="activation")
        from .tk_gemm import _get_tk_quant_for_gemm, get_tk_localcta_variant, use_tk_localcta_v4_strict_path
        tk_q = _get_tk_quant_for_gemm()
        prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
        try:
            if (
                get_tk_localcta_variant() == 'v4'
                and use_tk_localcta_v4_strict_path()
                and hasattr(tk_q, 'tk_quantize_for_gemm_direct_forward')
            ):
                result = tk_q.tk_quantize_for_gemm_direct_forward(
                    normed,
                    True,
                    use_tk_qkv_localcta_encode_centric(),
                )
            else:
                result = tk_q.tk_quantize_for_gemm(
                    normed,
                    True,
                    use_tk_qkv_localcta_encode_centric(),
                )
        finally:
            _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
        sg_t = (
            result[5]
            if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0
            else result[4]
        )
        return _TKQuantized(result[0], result[1], result[4], result[2], result[3], sg_t)
    return _fast_quantize(
        normed,
        input_quantizer,
        tk_swizzle=True,
        use_localcta_override=False,
    )


# ---------------------------------------------------------------------------
# TELinearFP4: plain TE linear with NVFP4 recipe (no norm/act fusion)
# ---------------------------------------------------------------------------
if not _NATIVE_LIGHT_IMPORT:
    class TELinearFP4(BoundRecipeLinear):
        """TE Linear layer with NVFP4 recipe."""

        def __init__(self, in_features, out_features, bias=False,
                     device=None, dtype=torch.bfloat16, recipe=None):
            if recipe is None:
                recipe = NVFP4BlockScaling()
            super().__init__(
                in_features, out_features, bias=bias,
                params_dtype=dtype, recipe=recipe, device=device,
            )

        def invalidate_weight_cache(self):
            """No-op for API compat with benchmarks."""
            pass
else:
    class TELinearFP4(nn.Module):
        """Lazy TE wrapper used only by native-light import audits."""

        def __init__(self, in_features, out_features, bias=False,
                     device=None, dtype=torch.bfloat16, recipe=None):
            super().__init__()
            if recipe is None:
                recipe = NVFP4BlockScaling()
            bound_cls = _get_bound_recipe_linear_cls()
            self.inner = bound_cls(
                in_features, out_features, bias=bias,
                params_dtype=dtype, recipe=recipe, device=device,
            )

        def forward(self, inp):
            return self.inner(inp)

        @property
        def weight(self):
            return self.inner.weight

        @property
        def bias(self):
            return getattr(self.inner, "bias", None)

        def invalidate_weight_cache(self):
            """No-op for API compat with benchmarks."""
            pass


# ---------------------------------------------------------------------------
# SimpleFP4Linear: plain FP4 linear using _fast_quantize + tex.generic_gemm
# No TE quantizer API, no context managers. Drop-in for nn.Linear.
# ---------------------------------------------------------------------------
class _SimpleFP4Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        workspace,
        debug_name=None,
        residual=None,
        cde_emit=False,
    ):
        M, K = input.shape
        N = weight.shape[0]
        trace = use_tk_stage_trace()
        if trace:
            _tk_stage_trace('simple_fp4_fwd', 'start', debug_name)

        _tk = (
            use_tk_gemm()
            and bias is None
            and M % 128 == 0
            and K % 128 == 0
            and N % 128 == 0
        )
        if (
            _tk
            and residual is not None
            and (M % 256 != 0 or K % 256 != 0 or N % 256 != 0)
        ):
            _tk = False
        if cde_emit and (
            not _tk
            or residual is None
            or not use_tk_localcta()
            or get_tk_localcta_variant() != "v4"
            or (M, N, K) != (24576, 4096, 8192)
        ):
            raise RuntimeError(
                "Nemotron Mamba CDE emission requires localCTA v4 native "
                "residual GEMM at MNK=(24576,4096,8192)"
            )
        # The production TK GEMM uses 256-row, 256-column, and 256-reduction
        # tiles even though its public contract accepts 128-aligned shapes.
        # Pad every participating dimension so an isolated 128 tail is neither
        # left unwritten in forward/wgrad nor omitted from dgrad reduction.
        Mp = (M + 255) // 256 * 256 if _tk else M
        Kp = (K + 255) // 256 * 256 if _tk else K
        Np = (N + 255) // 256 * 256 if _tk else N
        tk_needs_padding = _tk and (Mp != M or Kp != K or Np != N)
        if trace:
            _tk_stage_trace('simple_fp4_fwd', 'quant_start', debug_name)
        overlap_weight_quant = (
            _tk
            and use_localcta_mamba_out_weight_quant_overlap(
                debug_name, M, N, K
            )
        )
        caller_stream = None
        weight_quant_stream = None
        w_nvfp4 = None
        if overlap_weight_quant:
            caller_stream = torch.cuda.current_stream(input.device)
            weight_quant_stream = _get_ms_stream()
            weight_quant_stream.wait_stream(caller_stream)
            with torch.cuda.stream(weight_quant_stream):
                _record_tensors_on_stream((weight,), weight_quant_stream)
                w_nvfp4 = _fast_quantize(
                    weight, tk_swizzle=True, nvfp4_role="weight"
                )
        if _tk and (Mp != M or Kp != K):
            x_nvfp4 = _fast_quantize_tk_padded(
                input, Mp, Kp, nvfp4_role="activation"
            )
        else:
            x_nvfp4 = _fast_quantize(
                input, tk_swizzle=_tk, nvfp4_role="activation"
            )
        if weight_quant_stream is not None:
            assert caller_stream is not None
            caller_stream.wait_stream(weight_quant_stream)
            _record_tensors_on_stream(
                (
                    w_nvfp4._tk_row,
                    w_nvfp4._tk_col,
                    w_nvfp4._tk_row_chunk_sg,
                    w_nvfp4._tk_col_chunk_sg,
                    w_nvfp4._keepalive,
                ),
                caller_stream,
            )
        elif _tk and (Np != N or Kp != K):
            w_nvfp4 = _fast_quantize_tk_padded(
                weight, Np, Kp, nvfp4_role="weight"
            )
        else:
            w_nvfp4 = _fast_quantize(
                weight, tk_swizzle=_tk, nvfp4_role="weight"
            )
        if trace:
            _tk_stage_trace('simple_fp4_fwd', 'quant_done', debug_name)

        y_storage = torch.empty(
            (Mp, Np), dtype=torch.bfloat16, device=input.device
        )
        residual_bf16 = _as_contiguous_bf16(residual) if residual is not None else None
        if trace:
            _tk_stage_trace('simple_fp4_fwd', 'gemm_start', debug_name)
        row_rms_partial = None
        if _tk:
            if cde_emit:
                y_storage, row_rms_partial = tk_forward_gemm_residual_rms_partial(
                    x_nvfp4,
                    w_nvfp4,
                    residual_bf16,
                    y_storage,
                    use_localcta=True,
                )
            elif residual_bf16 is not None:
                tk_forward_gemm_residual(
                    x_nvfp4, w_nvfp4, residual_bf16, y_storage
                )
            else:
                tk_forward_gemm(x_nvfp4, w_nvfp4, y_storage)
        else:
            tex.generic_gemm(
                w_nvfp4, True, x_nvfp4, False,
                y_storage, None, TE_DType[torch.bfloat16],
                bias, TE_DType[bias.dtype] if bias is not None else TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )
            if residual_bf16 is not None:
                y_storage.add_(residual_bf16)
        if trace:
            _tk_stage_trace('simple_fp4_fwd', 'gemm_done', debug_name)
        y = y_storage[:M, :N] if tk_needs_padding else y_storage

        # Cache quantized tensors — TE quantize already produces both dimensions,
        # so backward can reuse them without re-quantizing.
        ctx.x_nvfp4 = x_nvfp4
        ctx.w_nvfp4 = w_nvfp4
        ctx.workspace = workspace
        ctx.has_bias = bias is not None
        ctx._lbt_debug_name = debug_name
        ctx.has_residual = residual is not None
        ctx.cde_output = bool(cde_emit)
        ctx.tk_path = _tk
        ctx.tk_orig_shape = (M, N, K)
        ctx.tk_padded_shape = (Mp, Np, Kp)
        if trace:
            _tk_stage_trace('simple_fp4_fwd', 'end', debug_name)
        if ctx.cde_output:
            ctx.mark_non_differentiable(row_rms_partial)
            return y, row_rms_partial
        return y

    @staticmethod
    def backward(ctx, grad_output, grad_row_rms_partial=None):
        workspace = ctx.workspace
        debug_name = getattr(ctx, '_lbt_debug_name', None)
        trace = use_tk_stage_trace()
        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'start', debug_name)

        # Reuse quantized tensors from forward (no re-quantization!)
        w_nvfp4 = ctx.w_nvfp4
        x_nvfp4 = ctx.x_nvfp4

        _tk = getattr(ctx, 'tk_path', False)
        M, N, K = ctx.tk_orig_shape
        Mp, Np, Kp = ctx.tk_padded_shape
        tk_needs_padding = _tk and (Mp != M or Np != N or Kp != K)
        grad_output_q_src = _as_contiguous_bf16(grad_output)
        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'dy_quant_start', debug_name)
        if _tk and (Mp != M or Np != N):
            dY_nvfp4 = _fast_quantize_tk_padded(
                grad_output_q_src, Mp, Np, nvfp4_role="grad"
            )
        else:
            dY_nvfp4 = _fast_quantize(
                grad_output_q_src, tk_swizzle=_tk, nvfp4_role="grad"
            )
        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'dy_quant_done', debug_name)

        # dgrad = dY @ W
        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'dgrad_start', debug_name)
        if _tk:
            grad_input = tk_dgrad_gemm(dY_nvfp4, w_nvfp4)
        else:
            grad_input = tex.generic_gemm(
                w_nvfp4, False, dY_nvfp4, False,
                None, None, TE_DType[torch.bfloat16],
                None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )[0]
        if tk_needs_padding and (Mp != M or Kp != K):
            grad_input = grad_input[:M, :K].contiguous()
        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'dgrad_done', debug_name)

        # wgrad = dY^T @ x (reuse cached x_nvfp4)
        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'wgrad_start', debug_name)
        if _tk:
            grad_weight = tk_wgrad_gemm(x_nvfp4, dY_nvfp4)
        else:
            grad_weight = tex.generic_gemm(
                x_nvfp4, False, dY_nvfp4, True,
                None, None, TE_DType[torch.bfloat16],
                None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )[0]
        if tk_needs_padding and (Np != N or Kp != K):
            grad_weight = grad_weight[:N, :K].contiguous()
        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'wgrad_done', debug_name)

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        # Free cached quantized tensors
        ctx.x_nvfp4 = None
        ctx.w_nvfp4 = None

        if trace:
            _tk_stage_trace('simple_fp4_bwd', 'end', debug_name)
        grad_residual = grad_output if getattr(ctx, 'has_residual', False) else None
        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            grad_residual,
            None,
        )


def _fast_mamba_gated_quantize_tk_linear(
    scan: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
) -> tuple[_TKQuantized, torch.Tensor]:
    """Emit a gated group-RMS result directly as native NVFP4 row/col payloads."""
    if use_tk_localcta():
        raise RuntimeError(
            "the fused Nemotron gated RMS projection is enabled only for "
            "regular NVFP4 v5; localCTA's direct producer is slower"
        )

    if not use_tk_quant():
        raise RuntimeError(
            "fused Nemotron gated RMS projection requires the native v5 "
            "quantizer"
        )
    tk_q = _get_tk_quant()
    method = getattr(
        tk_q,
        "tk_gated_group_rmsnorm_quantize_for_gemm",
        None,
    )
    if method is None:
        raise RuntimeError(
            "regular TK quant module lacks the native gated group-RMS "
            "producer; rebuild TK_quantisation/nvfp4_v5"
        )
    result = method(
        scan,
        gate,
        norm_weight,
        float(epsilon),
        use_nvfp4_encode_centric(),
    )
    quantized = _TKQuantized(
        result[0],
        result[1],
        result[4],
        result[2],
        result[3],
        result[5],
        keepalive=_result_keepalive(result, 6),
    )
    return quantized, result[8]


class _NVFP4MambaGatedLinearFunction(torch.autograd.Function):
    """Native gated group-RMS producer followed by regular v5 TK GEMMs."""

    @staticmethod
    def forward(
        ctx,
        scan: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        epsilon: float,
        debug_name: Optional[str] = None,
    ):
        inp = _as_contiguous_bf16(scan)
        if gate.dtype != torch.bfloat16 or gate.stride(1) != 1:
            raise RuntimeError(
                "fused Nemotron gated RMS projection requires a BF16 gate "
                "with unit inner stride"
            )
        w = _as_contiguous_bf16(weight.detach())
        nw = _as_contiguous_bf16(norm_weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        if (
            not use_tk_gemm()
            or inp.shape != gate.shape
            or K != 8192
            or M not in {8192, 16384, 24576, 32768}
            or gate.stride(0) != 18688
            or N % 128
        ):
            raise RuntimeError(
                "fused Nemotron gated RMS projection requires native TK with "
                "scan/gate [M,8192], gate row stride 18688, "
                f"M in {{8192,16384,24576,32768}}, and 128-aligned N; got "
                f"scan={tuple(inp.shape)}, gate={tuple(gate.shape)}, "
                f"gate_stride={tuple(gate.stride())}, N={N}"
            )

        _tk_stage_trace(
            "nemotron_mamba_gated_linear_fwd",
            "input_quant_start",
            debug_name,
        )
        x_nvfp4, inv_rms = _fast_mamba_gated_quantize_tk_linear(
            inp,
            gate,
            nw,
            float(epsilon),
        )
        _tk_stage_trace(
            "nemotron_mamba_gated_linear_fwd",
            "input_quant_done",
            debug_name,
        )
        w_nvfp4 = _fast_quantize(
            w,
            tk_swizzle=True,
            nvfp4_role="weight",
        )
        _tk_stage_trace(
            "nemotron_mamba_gated_linear_fwd",
            "gemm_start",
            debug_name,
        )
        output = tk_forward_gemm(x_nvfp4, w_nvfp4)
        _tk_stage_trace(
            "nemotron_mamba_gated_linear_fwd",
            "gemm_done",
            debug_name,
        )
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(inp, gate, nw, inv_rms)
        ctx.x_nvfp4 = x_nvfp4
        ctx.w_nvfp4 = w_nvfp4
        ctx.has_bias = bias is not None
        ctx._lbt_debug_name = debug_name
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, gate, nw, inv_rms = ctx.saved_tensors
        debug_name = getattr(ctx, "_lbt_debug_name", None)
        dY = _as_contiguous_bf16(grad_output)

        _tk_stage_trace(
            "nemotron_mamba_gated_linear_bwd",
            "grad_quant_start",
            debug_name,
        )
        dY_nvfp4 = _fast_quantize(
            dY,
            tk_swizzle=True,
            nvfp4_role="grad",
        )
        _tk_stage_trace(
            "nemotron_mamba_gated_linear_bwd",
            "grad_quant_done",
            debug_name,
        )
        d_normed = tk_dgrad_gemm(dY_nvfp4, ctx.w_nvfp4)
        grad_weight = tk_wgrad_gemm(ctx.x_nvfp4, dY_nvfp4)

        from low_bits_training.models.nemotron_h_hf.mamba_cuda import (
            gated_rmsnorm_backward_cuda,
        )

        grad_scan, grad_gate, grad_norm_weight = gated_rmsnorm_backward_cuda(
            d_normed,
            inp,
            gate,
            nw,
            inv_rms,
        )
        grad_bias = dY.sum(dim=0) if ctx.has_bias else None
        ctx.x_nvfp4 = None
        ctx.w_nvfp4 = None
        return (
            grad_scan,
            grad_gate,
            grad_weight,
            grad_bias,
            _as_param_grad_dtype(grad_norm_weight, nw),
            None,
            None,
        )


class SimpleFP4Linear(nn.Module):
    """Minimal FP4 linear: _fast_quantize + tex.generic_gemm.

    Drop-in replacement for nn.Linear. No TE quantizer API, no context managers.
    Uses the same proven codepath as the fused FFN.
    """

    def __init__(self, in_features, out_features, bias=False,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)
        self._workspace = None

    def _ensure_workspace(self, device):
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=device)

    def forward(self, input, residual=None, cde_emit=False):
        is_3d = input.dim() == 3
        if is_3d:
            B, S, H = input.shape
            input = input.reshape(B * S, H)
            residual = residual.reshape(B * S, self.out_features) if residual is not None else None

        self._ensure_workspace(input.device)
        out = _SimpleFP4Function.apply(
            input, self.weight, self.bias, self._workspace,
            getattr(self, '_lbt_debug_name', None), residual, cde_emit)

        if is_3d:
            if isinstance(out, tuple):
                return out[0].view(B, S, self.out_features), out[1]
            return out.view(B, S, self.out_features)
        return out

    def forward_mamba_gated(
        self,
        scan: torch.Tensor,
        gate: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        if scan.shape != gate.shape or scan.shape[-1] != self.in_features:
            raise RuntimeError(
                "fused Nemotron gated RMS projection received incompatible "
                f"scan/gate shapes {tuple(scan.shape)} and {tuple(gate.shape)} "
                f"for in_features={self.in_features}"
            )
        output_shape = (*scan.shape[:-1], self.out_features)
        scan_2d = scan.reshape(-1, self.in_features)
        try:
            gate_2d = gate.view(-1, self.in_features)
        except RuntimeError as exc:
            raise RuntimeError(
                "fused Nemotron gated RMS projection requires a viewable "
                "production-strided gate"
            ) from exc
        output = _NVFP4MambaGatedLinearFunction.apply(
            scan_2d,
            gate_2d,
            self.weight,
            self.bias,
            norm_weight,
            float(epsilon),
            getattr(self, "_lbt_debug_name", None),
        )
        return output.view(output_shape)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "SimpleFP4Linear":
        out = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        if linear.weight.device.type != "meta":
            with torch.no_grad():
                out.weight.copy_(linear.weight)
                if linear.bias is not None:
                    out.bias.copy_(linear.bias)
        return out


def _fast_rmsnorm_quantize_tk_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    epsilon: float,
) -> tuple[_TKQuantized, torch.Tensor]:
    """Quantize a native RMSNorm result without materializing BF16 output."""
    if use_tk_localcta():
        if get_tk_localcta_variant() != "v4":
            raise RuntimeError(
                "fused Nemotron RMSNorm projection requires localCTA v4"
            )
        return _fast_rmsnorm_quantize_localcta_v4_opt(
            input,
            norm_weight,
            float(epsilon),
            nvfp4_role="activation",
        )

    if not use_tk_quant():
        raise RuntimeError(
            "fused Nemotron RMSNorm projection requires the native v5 quantizer"
        )
    if _nvfp4_quantizer_extras_enabled("activation"):
        if not _can_fast_rmsnorm_quantize_tk_regular_opt(input, "activation"):
            raise RuntimeError(
                "the requested v5 activation extras are not supported by the "
                "native fused RMSNorm projection producer"
            )
        return _fast_rmsnorm_quantize_tk_regular_opt(
            input,
            norm_weight,
            float(epsilon),
            nvfp4_role="activation",
        )

    tk_q = _get_tk_quant()
    if not hasattr(tk_q, "tk_fused_norm_quantize"):
        raise RuntimeError(
            "regular TK quant module lacks tk_fused_norm_quantize; rebuild "
            "TK_quantisation/nvfp4_v5"
        )
    result = tk_q.tk_fused_norm_quantize(
        input,
        norm_weight,
        float(epsilon),
        False,
        True,
    )
    quantized = _TKQuantized(
        result[0],
        result[1],
        result[4],
        result[2],
        result[3],
        keepalive=_result_keepalive(result, 6),
    )
    return quantized, result[5]


class _NVFP4RMSNormLinearFunction(torch.autograd.Function):
    """Fused native RMSNorm producer plus padded v5/localCTA TK GEMMs."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
        debug_name: Optional[str] = None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
    ):
        inp = _as_contiguous_bf16(input)
        w = _as_contiguous_bf16(weight.detach())
        nw = _as_contiguous_bf16(norm_weight.detach())
        M, K = inp.shape
        N = w.shape[0]
        Mp = (M + 255) // 256 * 256
        Kp = (K + 255) // 256 * 256
        Np = (N + 255) // 256 * 256
        if (
            not use_tk_gemm()
            or M == 0
            or Mp != M
            or Kp != K
            or M % 128
            or K % 128
            or N % 128
        ):
            raise RuntimeError(
                "fused Nemotron NVFP4 RMSNorm projection requires native TK "
                f"with 256-aligned M/K and 128-aligned N, got MKN={(M, K, N)}"
            )
        cde_input = (
            torch.is_tensor(cde_row_rms_partial)
            and cde_row_rms_partial.numel() != 0
        )
        if cde_input and (
            not use_tk_localcta()
            or get_tk_localcta_variant() != "v4"
            or (M, K, N) != (24576, 4096, 18560)
            or cde_row_rms_partial.dtype != torch.float32
            or not cde_row_rms_partial.is_cuda
            or not cde_row_rms_partial.is_contiguous()
            or tuple(cde_row_rms_partial.shape) != (M, K // 256)
        ):
            raise RuntimeError(
                "Nemotron Mamba input CDE requires localCTA v4 at "
                "MKN=(24576,4096,18560) with contiguous FP32 [M,K/256] partials"
            )

        _tk_stage_trace(
            "nemotron_mamba_rms_linear_fwd", "input_quant_start", debug_name
        )
        if cde_input:
            from .tk_gemm import _get_tk_quant_for_gemm

            result = (
                _get_tk_quant_for_gemm()
                .tk_rmsnorm_quantize_from_row_rms_partial_final_sg(
                    inp,
                    nw,
                    cde_row_rms_partial,
                    float(epsilon),
                    True,
                    use_nvfp4_encode_centric(),
                )
            )
            x_nvfp4 = _TKQuantized(
                result[0],
                result[1],
                result[4],
                result[2],
                result[3],
                result[5],
                keepalive=_result_keepalive(result, 7),
            )
            inv_rms = result[6]
            _trace_backend_choice(
                "localcta_nemotron_interlayer_cde",
                "native",
            )
        else:
            x_nvfp4, inv_rms = _fast_rmsnorm_quantize_tk_linear(
                inp, nw, float(epsilon)
            )
        _tk_stage_trace(
            "nemotron_mamba_rms_linear_fwd", "input_quant_done", debug_name
        )
        if Np != N:
            w_nvfp4 = _fast_quantize_tk_padded(
                w, Np, Kp, nvfp4_role="weight"
            )
        else:
            w_nvfp4 = _fast_quantize(
                w, tk_swizzle=True, nvfp4_role="weight"
            )

        y_storage = torch.empty(
            (Mp, Np), dtype=torch.bfloat16, device=inp.device
        )
        _tk_stage_trace(
            "nemotron_mamba_rms_linear_fwd", "gemm_start", debug_name
        )
        tk_forward_gemm(x_nvfp4, w_nvfp4, y_storage)
        _tk_stage_trace(
            "nemotron_mamba_rms_linear_fwd", "gemm_done", debug_name
        )

        ctx.save_for_backward(inp, nw, inv_rms)
        ctx.x_nvfp4 = x_nvfp4
        ctx.w_nvfp4 = w_nvfp4
        ctx.orig_shape = (M, N, K)
        ctx.padded_shape = (Mp, Np, Kp)
        ctx._lbt_debug_name = debug_name
        return y_storage[:, :N] if Np != N else y_storage

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, nw, inv_rms = ctx.saved_tensors
        x_nvfp4 = ctx.x_nvfp4
        w_nvfp4 = ctx.w_nvfp4
        M, N, K = ctx.orig_shape
        Mp, Np, Kp = ctx.padded_shape
        debug_name = getattr(ctx, "_lbt_debug_name", None)
        dY = _as_contiguous_bf16(grad_output)

        _tk_stage_trace(
            "nemotron_mamba_rms_linear_bwd", "grad_quant_start", debug_name
        )
        if Np != N:
            dY_nvfp4 = _fast_quantize_tk_padded(
                dY, Mp, Np, nvfp4_role="grad"
            )
        else:
            dY_nvfp4 = _fast_quantize(
                dY, tk_swizzle=True, nvfp4_role="grad"
            )
        _tk_stage_trace(
            "nemotron_mamba_rms_linear_bwd", "grad_quant_done", debug_name
        )

        dx_normed = tk_dgrad_gemm(dY_nvfp4, w_nvfp4)
        if Kp != K:
            dx_normed = dx_normed[:M, :K].contiguous()
        grad_weight = tk_wgrad_gemm(x_nvfp4, dY_nvfp4)
        if Np != N or Kp != K:
            grad_weight = grad_weight[:N, :K].contiguous()

        te_fused = _get_te_fused()
        grad_input, grad_norm_weight = te_fused.fused_rmsnorm_backward(
            dx_normed.contiguous(),
            inp,
            nw,
            inv_rms,
        )
        ctx.x_nvfp4 = None
        ctx.w_nvfp4 = None
        return (
            grad_input,
            grad_weight,
            _as_param_grad_dtype(grad_norm_weight, nw),
            None,
            None,
            None,
        )


class NVFP4RMSNormLinearTK(nn.Module):
    """Nemotron Mamba input projection with native fused RMS quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.norm_weight = nn.Parameter(
            torch.ones(in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        _safe_trunc_normal_(self.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.norm_weight)

    def forward(
        self,
        input: torch.Tensor,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_nd = input.dim() > 2
        if is_nd:
            orig_shape = input.shape[:-1]
            input = input.reshape(-1, input.shape[-1])
        output = _NVFP4RMSNormLinearFunction.apply(
            input,
            self.weight,
            self.norm_weight,
            self.eps,
            getattr(self, "_lbt_debug_name", None),
            cde_row_rms_partial,
        )
        if is_nd:
            output = output.reshape(*orig_shape, self.out_features)
        return output

    def invalidate_weight_cache(self):
        pass


# ---------------------------------------------------------------------------
# Helper: round up for TE's padded scale layout



# _FusedFFNFunctionV2: Custom autograd for full FFN with fused CUDA kernels
#
# Architecture: w2(silu(w1(rms_norm(x))) * w3(rms_norm(x)))
# Fusion strategy:
#   Forward:  fused kernel: rms_norm + silu + fp4_quant → GEMM w1
#             quantizer:    rms_norm(x) → fp4_quant   → GEMM w3
#             quantizer:    h1*h3      → fp4_quant    → GEMM w2
#   Backward: tex.generic_gemm for all dgrads/wgrads
#             fused_silu_rmsnorm_backward for combined dx + dgamma
# ---------------------------------------------------------------------------
class _FusedFFNFunctionV2_TE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,           # (M, K) bf16 — raw pre-norm input
        w1_weight: torch.Tensor,        # (H, K) bf16 — gate projection
        w3_weight: torch.Tensor,        # (H, K) bf16 — up projection
        w2_weight: torch.Tensor,        # (K, H) bf16 — down projection
        norm_weight: torch.Tensor,      # (K,) bf16 — RMSNorm gamma
        epsilon: float,
        # TE quantizers
        w1_weight_quantizer: NVFP4Quantizer,
        w3_input_quantizer: NVFP4Quantizer,
        w3_weight_quantizer: NVFP4Quantizer,
        w2_input_quantizer: NVFP4Quantizer,
        w2_weight_quantizer: NVFP4Quantizer,
        grad_quantizer_w1: NVFP4Quantizer,
        grad_quantizer_w3: NVFP4Quantizer,
        grad_quantizer_w2: NVFP4Quantizer,
        # Dummy quantizer to wrap fused kernel output
        w1_input_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
        debug_name: Optional[str] = None,
    ):
        M, K = input.shape
        H = w1_weight.shape[0]
        N = w2_weight.shape[0]
        debug_prefix = f"te_ffn_fwd[{_next_te_ffn_debug_call('fwd')}]"

        inp = _as_contiguous_bf16(input)
        nw = norm_weight.detach()
        _debug_check_finite(f'{debug_prefix}.input', inp)
        _debug_check_finite(f'{debug_prefix}.norm_weight', nw)
        _debug_check_finite(f'{debug_prefix}.w1_weight', w1_weight)
        _debug_check_finite(f'{debug_prefix}.w3_weight', w3_weight)
        _debug_check_finite(f'{debug_prefix}.w2_weight', w2_weight)

        # ---- Quantize all weights (cached for backward) ----
        w1_nvfp4 = _fast_quantize(w1_weight, w1_weight_quantizer)
        w3_nvfp4 = _fast_quantize(w3_weight, w3_weight_quantizer)
        w2_nvfp4 = _fast_quantize(w2_weight, w2_weight_quantizer)
        _dump_ffn_tensors(
            "w2_weight_src",
            debug_prefix,
            {
                "debug_name": debug_prefix,
                "w2_weight": w2_weight,
            },
        )

        # ---- RMSNorm + FP4 quantize shared input for W1/W3 ----
        te_fused = _get_te_fused()
        if use_te_ffn_safe_rmsnorm():
            normed, inv_rms = _te_ffn_rmsnorm_forward_reference(inp, nw, float(epsilon))
            _debug_check_finite(f'{debug_prefix}.normed', normed)
            x_nvfp4 = _fast_quantize(normed, w1_input_quantizer)
        elif use_te_ffn_safe_input_quant():
            normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
            _debug_check_finite(f'{debug_prefix}.normed', normed)
            x_nvfp4 = _fast_quantize(normed, w1_input_quantizer)
        else:
            # Single shared input: rmsnorm+quant (no silu)
            # fused_rmsnorm_quantize: 2 kernels (norm+amax, quantize) → returns
            # (fp4, si, fp4_t, si_t, amax, inv_rms)
            fp4_norm, si_norm, fp4_norm_t, si_norm_t, amax_norm, inv_rms = \
                te_fused.fused_rmsnorm_quantize(
                    inp.detach(), nw, float(epsilon), False)

            x_nvfp4 = _make_nvfp4_tensor_compat(
                (M, K), torch.bfloat16,
                rowwise_data=fp4_norm, rowwise_scale_inv=si_norm,
                columnwise_data=fp4_norm_t, columnwise_scale_inv=si_norm_t,
                amax_rowwise=amax_norm, amax_columnwise=amax_norm,  # shared amax
                fp4_dtype=tex.DType.kFloat4E2M1,
                quantizer=w1_input_quantizer,
            )

        def _fwd_gemm(x_q, w_q, out):
            tex.generic_gemm(
                w_q, True, x_q, False, out, None,
                TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                False, None, False, workspace, workspace.shape[0], False, False,
            )

        # h1_raw = W1 @ rmsnorm(x) — raw projection, NO silu yet
        h1_raw = torch.empty((M, H), dtype=torch.bfloat16, device=inp.device)
        _fwd_gemm(x_nvfp4, w1_nvfp4, h1_raw)
        _debug_check_finite(f'{debug_prefix}.h1_raw', h1_raw)

        # h3 = W3 @ rmsnorm(x) — gate projection (shared input)
        h3 = torch.empty((M, H), dtype=torch.bfloat16, device=inp.device)
        _fwd_gemm(x_nvfp4, w3_nvfp4, h3)
        _debug_check_finite(f'{debug_prefix}.h3', h3)

        # ---- h = silu(h1_raw) * h3 + FP4 quantize (CORRECTED SwiGLU) ----
        if use_te_ffn_fwd_safe_producer():
            # Conservative reference path: keep the producer in explicit PyTorch
            # math so step-to-step trainer probes are not exposed to fused SiLU
            # producer kernel bugs.
            h = (F.silu(h1_raw.float()) * h3.float()).to(torch.bfloat16)
            _debug_check_finite(f'{debug_prefix}.h', h)
            h_nvfp4 = _fast_quantize(h, w2_input_quantizer)
        else:
            fp4, si, fp4_t, si_t, amax, amax_t, sg_h = te_fused.fused_te_silu_mul_quantize(
                h1_raw, h3, False, False)
            h_nvfp4 = _make_nvfp4_tensor_compat(
                (M, H), torch.bfloat16,
                rowwise_data=fp4, rowwise_scale_inv=si,
                columnwise_data=fp4_t, columnwise_scale_inv=si_t,
                amax_rowwise=amax, amax_columnwise=amax_t,
                fp4_dtype=tex.DType.kFloat4E2M1,
                quantizer=w2_input_quantizer,
            )
            h_nvfp4._with_gemm_swizzled_scales = False

        y = torch.empty((M, N), dtype=torch.bfloat16, device=inp.device)
        _fwd_gemm(h_nvfp4, w2_nvfp4, y)
        _debug_check_finite(f'{debug_prefix}.output', y)

        # ---- Save for backward ----
        ctx.save_for_backward(inp, nw, inv_rms, h1_raw, h3)
        ctx.w1_nvfp4 = w1_nvfp4
        ctx.w3_nvfp4 = w3_nvfp4
        ctx.w2_nvfp4 = w2_nvfp4
        ctx.w2_dgrad_col = None
        ctx.x_nvfp4 = x_nvfp4    # shared normed quantized input (for both wgrads)
        ctx.h_nvfp4 = h_nvfp4     # h=silu(h1_raw)*h3 quantized
        ctx.epsilon = epsilon
        ctx.grad_quantizer_w1 = grad_quantizer_w1
        ctx.grad_quantizer_w3 = grad_quantizer_w3
        ctx.grad_quantizer_w2 = grad_quantizer_w2
        ctx.workspace = workspace

        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, norm_weight, inv_rms, h1_raw, h3 = ctx.saved_tensors
        workspace = ctx.workspace
        debug_prefix = f"te_ffn_bwd[{_next_te_ffn_debug_call('bwd')}]"
        M = grad_output.size(0)

        w1_nvfp4 = ctx.w1_nvfp4
        w3_nvfp4 = ctx.w3_nvfp4
        w2_nvfp4 = ctx.w2_nvfp4
        x_nvfp4 = ctx.x_nvfp4  # shared normed quantized input
        h_nvfp4 = ctx.h_nvfp4

        te_fused = _get_te_fused()

        def _dgrad(dy_q, w_q):
            return tex.generic_gemm(
                w_q, False, dy_q, False, None, None,
                TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                False, None, False, workspace, workspace.shape[0], False, False,
            )[0]

        def _wgrad(x_q, dy_q):
            return tex.generic_gemm(
                x_q, False, dy_q, True, None, None,
                TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                False, None, False, workspace, workspace.shape[0], False, False,
            )[0]

        # 1. Quantize dY
        dY_nvfp4 = _fast_quantize(grad_output, ctx.grad_quantizer_w2)
        _debug_check_finite(f'{debug_prefix}.grad_output', grad_output)

        # 2–3. w2 backward: dh = dY @ W2, dW2 = h^T @ dY
        dh     = _dgrad(dY_nvfp4, w2_nvfp4)
        grad_w2 = _wgrad(h_nvfp4, dY_nvfp4)
        _debug_check_finite(f'{debug_prefix}.dh', dh)
        _debug_check_finite(f'{debug_prefix}.grad_w2', grad_w2)
        if hasattr(dY_nvfp4, "_tk_row") and hasattr(w2_nvfp4, "_tk_col"):
            _dump_ffn_tensors(
                "w2_dgrad",
                debug_prefix,
                {
                    "debug_name": debug_prefix,
                    "grad_output": grad_output,
                    "dY_fp4_row": dY_nvfp4._tk_row[0],
                    "dY_sc_row": dY_nvfp4._tk_row[1],
                    "dY_sg_row": dY_nvfp4._tk_row[2],
                    "w2_fp4_col": w2_nvfp4._tk_col[0],
                    "w2_sc_col": w2_nvfp4._tk_col[1],
                    "w2_sg_col": w2_nvfp4._tk_col[2],
                    "dh": dh,
                },
            )

        # 4. SiLU-derivative producer:
        #    dh1_raw = dh * h3 * silu'(h1_raw)  — grad w.r.t. raw W1 output
        #    dh3     = dh * silu(h1_raw)         — grad w.r.t. gate output
        #
        # The safer producer computes BF16 branches first, then quantizes each
        # branch independently. This has been more stable on the TK/localCTA FFN
        # paths and keeps the GEMM side unchanged.
        if use_te_ffn_bwd_safe_producer():
            dh1_raw = torch.empty_like(dh)
            dh3 = torch.empty_like(dh)
            if hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
                te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                    dh, h3, h1_raw, dh1_raw, dh3,
                )
            elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out'):
                amax_1 = torch.zeros(1, dtype=torch.float32, device=dh.device)
                amax_2 = torch.zeros(1, dtype=torch.float32, device=dh.device)
                te_fused.fused_silu_deriv_dual_mul_bf16_out(
                    dh, h3, h1_raw, dh1_raw, dh3, amax_1, amax_2,
                )
            else:
                dh1_tmp, dh3_tmp, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1_raw)
                dh1_raw.copy_(dh1_tmp)
                dh3.copy_(dh3_tmp)
            _debug_check_finite(f'{debug_prefix}.safe.dh1_raw', dh1_raw)
            _debug_check_finite(f'{debug_prefix}.safe.dh3', dh3)
            dh1_raw_nvfp4 = _fast_quantize(dh1_raw, ctx.grad_quantizer_w1)
            dh3_nvfp4 = _fast_quantize(dh3, ctx.grad_quantizer_w3)
        else:
            (fp4_1, si_1, fp4_1t, si_1t, amax_1, amax_1t,
             fp4_2, si_2, fp4_2t, si_2t, amax_2, amax_2t,
             sg_1, sg_2) = te_fused.fused_te_silu_deriv_dual_mul_quantize(
                dh, h3, h1_raw, False, False)

            M_g, H_g = dh.shape
            dh1_raw_nvfp4 = _make_nvfp4_tensor_compat(
                (M_g, H_g), torch.bfloat16,
                rowwise_data=fp4_1, rowwise_scale_inv=si_1,
                columnwise_data=fp4_1t, columnwise_scale_inv=si_1t,
                amax_rowwise=amax_1, amax_columnwise=amax_1t,
                fp4_dtype=tex.DType.kFloat4E2M1, quantizer=ctx.grad_quantizer_w1,
            )
            dh1_raw_nvfp4._with_gemm_swizzled_scales = False

            dh3_nvfp4 = _make_nvfp4_tensor_compat(
                (M_g, H_g), torch.bfloat16,
                rowwise_data=fp4_2, rowwise_scale_inv=si_2,
                columnwise_data=fp4_2t, columnwise_scale_inv=si_2t,
                amax_rowwise=amax_2, amax_columnwise=amax_2t,
                fp4_dtype=tex.DType.kFloat4E2M1, quantizer=ctx.grad_quantizer_w3,
            )
            dh3_nvfp4._with_gemm_swizzled_scales = False

        # 5–6. w1 backward: d_normed_w1 = dh1_raw @ W1, dW1 = normed^T @ dh1_raw
        d_normed_w1 = _dgrad(dh1_raw_nvfp4, w1_nvfp4)
        grad_w1     = _wgrad(x_nvfp4, dh1_raw_nvfp4)
        _debug_check_finite(f'{debug_prefix}.d_normed_w1', d_normed_w1)
        _debug_check_finite(f'{debug_prefix}.grad_w1', grad_w1)

        # 7–8. w3 backward: d_normed_w3 = dh3 @ W3, dW3 = normed^T @ dh3
        d_normed_w3 = _dgrad(dh3_nvfp4, w3_nvfp4)
        grad_w3     = _wgrad(x_nvfp4, dh3_nvfp4)
        _debug_check_finite(f'{debug_prefix}.d_normed_w3', d_normed_w3)
        _debug_check_finite(f'{debug_prefix}.grad_w3', grad_w3)

        # 9. Pure RMSNorm backward (no SiLU — silu' already in element-wise step)
        d_normed_total = (d_normed_w1 + d_normed_w3).contiguous()
        if use_te_ffn_safe_rmsnorm():
            try:
                from .tk_gemm import _launch_rmsnorm_bwd_out_async

                rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                    d_normed_total,
                    input.contiguous(),
                    norm_weight,
                    inv_rms,
                    te_fused,
                    owner_key=("te_ffn_safe_rmsnorm", d_normed_total.device.index),
                )
                torch.cuda.current_stream().wait_stream(rms_stream)
                grad_input = rms_state["grad_input"]
                dgamma = rms_state.get("dgamma_out", rms_state["dgamma"])
            except Exception:
                grad_input, dgamma = _te_ffn_rmsnorm_backward_reference(
                    d_normed_total,
                    input.contiguous(),
                    norm_weight,
                    inv_rms,
                )
        else:
            grad_input, dgamma = te_fused.fused_rmsnorm_backward(
                d_normed_total, input.contiguous(),
                norm_weight, inv_rms)
        grad_norm_weight = _as_param_grad_dtype(dgamma, norm_weight)
        _debug_check_finite(f'{debug_prefix}.grad_input', grad_input)
        _debug_check_finite(f'{debug_prefix}.grad_norm_weight', grad_norm_weight)

        # Free cached tensors
        ctx.w1_nvfp4 = None
        ctx.w3_nvfp4 = None
        ctx.w2_nvfp4 = None
        ctx.w2_dgrad_col = None
        ctx.x_nvfp4 = None
        ctx.h_nvfp4 = None

        # 16 inputs to forward
        return (
            _maybe_clone_autograd_return(grad_input, M),        # input
            _maybe_clone_autograd_return(grad_w1, M),           # w1_weight
            _maybe_clone_autograd_return(grad_w3, M),           # w3_weight
            _maybe_clone_autograd_return(grad_w2, M),           # w2_weight
            _maybe_clone_autograd_return(grad_norm_weight, M),  # norm_weight
            None,              # epsilon
            None,              # w1_weight_quantizer
            None, None,        # w3 quantizers
            None, None,        # w2 quantizers
            None, None, None,  # grad quantizers
            None,              # w1_input_quantizer
            None,              # workspace
        )


# ---------------------------------------------------------------------------
# FFN backward CUDA graph
# ---------------------------------------------------------------------------
_ffn_bwd_graph_cache = {}  # (M, K, H, device) → (graph, static_bufs)
_ffn_sb_cache = {}         # (M, K, H, device) → sb dict (shared with forward)
_ffn_localcta_fwd_cache = {}  # (M, H, device) → eager localCTA forward scratch
_ffn_localcta_bwd_cache = {}  # (M, K, H, device) → eager localCTA scratch
_ffn_localcta_owned_grad_cache = {}  # (role, layer, shape, device) → param grad
_ffn_localcta_deriv_streams = {}  # device → derivative producer stream
_ffn_localcta_w2_producer_streams = {}  # device → priority W2 producer stream


def use_tk_localcta_persistent_step_scratch() -> bool:
    """Keep shape-keyed localCTA scratch allocated across optimizer steps."""
    value = os.environ.get('USE_TK_LOCALCTA_PERSISTENT_STEP_SCRATCH')
    if value is not None:
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return False


def clear_fused_fp4_step_caches() -> None:
    """Release step-scoped fused FP4 caches after optimizer updates."""
    _ffn_bwd_graph_cache.clear()
    _ffn_sb_cache.clear()
    if not use_tk_localcta_persistent_step_scratch():
        _ffn_localcta_fwd_cache.clear()
        _ffn_localcta_bwd_cache.clear()
    _qkv_full_graph_cache.clear()
    _qkv_bwd_graph_cache.clear()
    if hasattr(_FusedQKVFunction_TK, "_w_col_cache"):
        _FusedQKVFunction_TK._w_col_cache.clear()


def _get_ffn_localcta_fwd_state(M: int, H: int, device: torch.device):
    """Get cached eager buffers for the localCTA FFN forward fast path."""
    key = (M, H, device.index)
    state = _ffn_localcta_fwd_cache.get(key)
    if state is None:
        state = {
            'h': torch.empty(M, H, dtype=torch.bfloat16, device=device),
            'amax': torch.zeros(1, dtype=torch.float32, device=device),
        }
        _ffn_localcta_fwd_cache[key] = state
    return state


def _get_ffn_localcta_deriv_stream(device: torch.device) -> torch.cuda.Stream:
    """Return the per-device stream used for FFN derivative quantization."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _ffn_localcta_deriv_streams.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream(device=device_index)
        _ffn_localcta_deriv_streams[device_index] = stream
    return stream


def _get_ffn_localcta_w2_producer_stream(device: torch.device) -> torch.cuda.Stream:
    """Return the per-device high-priority stream for the clustered W2 producer."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _ffn_localcta_w2_producer_streams.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream(device=device_index, priority=-1)
        _ffn_localcta_w2_producer_streams[device_index] = stream
    return stream


class _LazyFFNScratch(dict):
    """Materialize large localCTA fallback buffers only when a path uses them."""

    def __init__(self, device: torch.device, specs):
        super().__init__()
        self._device = device
        self._specs = specs

    def __missing__(self, key):
        try:
            shape, dtype, fill = self._specs[key]
        except KeyError:
            raise KeyError(key) from None
        factory = {
            'ones': torch.ones,
            'zeros': torch.zeros,
        }.get(fill, torch.empty)
        value = factory(shape, dtype=dtype, device=self._device)
        self[key] = value
        return value


def _get_ffn_localcta_dh_scratch(localcta_state, *, fused_producer: bool):
    """Materialize the BF16 W2-dgrad output only when a consumer needs it."""
    if fused_producer:
        return None
    return localcta_state['dh']


def _get_ffn_localcta_owned_grad_buffer(
    role: str,
    debug_name: str | None,
    owner: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Allocate an autograd-owned parameter-gradient output.

    Keeping one output in a module cache for every layer prevents FSDP from
    releasing the full gradient after reduce-scatter.  The CUDA allocator can
    recycle this storage once autograd and FSDP are done with it, while the
    tensor's normal lifetime guarantees that a later layer cannot overwrite it.
    """
    shape = tuple(int(dim) for dim in shape)
    if os.environ.get('USE_TK_LOCALCTA_TRANSIENT_W2_GRAD', '0') == '1':
        return torch.empty(
            shape,
            dtype=torch.bfloat16,
            device=owner.device,
        )
    layer_key = (
        debug_name
        if isinstance(debug_name, str) and debug_name
        else f"parameter:{int(owner.data_ptr())}"
    )
    key = (role, layer_key, shape, owner.device.index)
    output = _ffn_localcta_owned_grad_cache.get(key)
    if output is None:
        output = torch.empty(shape, dtype=torch.bfloat16, device=owner.device)
        _ffn_localcta_owned_grad_cache[key] = output
    return output


def _get_ffn_localcta_bwd_state(M: int, K: int, H: int, device: torch.device):
    """Get cached eager buffers for the localCTA FFN backward fast path."""
    key = (M, K, H, device.index)
    state = _ffn_localcta_bwd_cache.get(key)
    if state is None:
        specs = {
            'dY_bf16': ((M, K), torch.bfloat16, 'empty'),
            'dh': ((M, H), torch.bfloat16, 'empty'),
            'h_bf16': ((M, H), torch.bfloat16, 'empty'),
            'w2_bf16': ((K, H), torch.bfloat16, 'empty'),
            'x_normed': ((M, K), torch.bfloat16, 'empty'),
            'dh1': ((M, H), torch.bfloat16, 'empty'),
            'dh3': ((M, H), torch.bfloat16, 'empty'),
            'd_normed_tmp': ((M, K), torch.bfloat16, 'empty'),
            'dh1_row_sg': ((M // 128, H // 128), torch.float32, 'empty'),
            'dh3_row_sg': ((M // 128, H // 128), torch.float32, 'empty'),
            'dh_col_fp4_full': ((2 * H, M // 2), torch.float4_e2m1fn_x2, 'empty'),
            'dh_col_sc_full': (((2 * H) // 128, M // 64, 512), torch.float8_e4m3fn, 'empty'),
            'dh_col_sg_full': (((2 * H) // 128, M // 128), torch.float32, 'empty'),
            'unit_sg_m256': ((max(M // 256, 1), 1), torch.float32, 'ones'),
            'unit_sg_h256': ((max(H // 256, 1), 1), torch.float32, 'ones'),
            'amax1': ((1,), torch.float32, 'zeros'),
            'amax2': ((1,), torch.float32, 'zeros'),
            'grad_w1': ((H, K), torch.bfloat16, 'empty'),
            'grad_w3': ((H, K), torch.bfloat16, 'empty'),
            'grad_w2': ((K, H), torch.bfloat16, 'empty'),
            'd_normed': ((M, K), torch.bfloat16, 'empty'),
            'grad_input': ((M, K), torch.bfloat16, 'empty'),
            'dgamma': ((K,), torch.float32, 'empty'),
        }
        state = _LazyFFNScratch(device, specs)
        state.update({
            'dh_ready_event': torch.cuda.Event(),
            'split_quant_ready_event': torch.cuda.Event(),
            'deriv_quant_ready_event': torch.cuda.Event(),
            'wgrad_done_event': torch.cuda.Event(),
            'w2_dgrad_silu_payload_ready_event': torch.cuda.Event(),
            'w2_dgrad_silu_payload_ready_recorded': False,
        })
        _ffn_localcta_bwd_cache[key] = state
    return state


def _get_ffn_localcta_h13_recompute_buffers(
    M: int,
    K: int,
    H: int,
    device: torch.device,
):
    """Allocate or alias the two large H1/H3 recompute outputs."""
    state = _get_ffn_localcta_bwd_state(M, K, H, device)
    if (
        use_tk_ffn_localcta_inplace_h13_deriv()
        and not use_tk_localcta_ffn_bf16_dgrad_debug()
    ):
        if not use_tk_ffn_recompute_h13():
            raise RuntimeError(
                "USE_TK_FFN_LOCALCTA_INPLACE_H13_DERIV requires "
                "USE_TK_FFN_RECOMPUTE_H13=1"
            )
        return state['dh1'], state['dh3']
    return (
        torch.empty(M, H, dtype=torch.bfloat16, device=device),
        torch.empty(M, H, dtype=torch.bfloat16, device=device),
    )


def _get_ffn_localcta_deriv_outputs(localcta_state, h1_raw, h3):
    """Select distinct or in-place SiLU derivative output buffers."""
    if (
        not use_tk_ffn_localcta_inplace_h13_deriv()
        or use_tk_localcta_ffn_bf16_dgrad_debug()
    ):
        return localcta_state['dh1'], localcta_state['dh3']
    if (
        h1_raw.data_ptr() != localcta_state['dh1'].data_ptr()
        or h3.data_ptr() != localcta_state['dh3'].data_ptr()
    ):
        raise RuntimeError(
            "in-place localCTA H1/H3 derivatives require the cached recompute buffers"
        )
    return h1_raw, h3


def _produce_ffn_localcta_derivatives_with_te(
    te_fused,
    dh,
    h3,
    h1_raw,
    dh1,
    dh3_out,
    amax1,
    amax2,
) -> None:
    """Reproduce the established paired-carrier BF16 derivative payload."""
    if hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
        te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
            dh, h3, h1_raw, dh1, dh3_out,
        )
    elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out'):
        te_fused.fused_silu_deriv_dual_mul_bf16_out(
            dh, h3, h1_raw, dh1, dh3_out, amax1, amax2,
        )
    else:
        dh1_tmp, dh3_tmp, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(
            dh, h3, h1_raw
        )
        dh1.copy_(dh1_tmp)
        dh3_out.copy_(dh3_tmp)


def _localcta_ffn_dequant_split2_dgrad(
    tkq_mod,
    row_fp4s,
    row_scs,
    row_sgs,
    weight_fp4_cols,
    weight_sc_cols,
    weight_sg_cols,
    output,
    M,
    H,
    K,
    mode='both',
    exact_activations=None,
    exact_weights=None,
    input_tensor=None,
    inv_rms=None,
    debug_name=None,
):
    if tkq_mod is None or not (
        hasattr(tkq_mod, 'tk_localcta_reconstruct_row')
        and hasattr(tkq_mod, 'tk_localcta_reconstruct_col')
    ):
        raise RuntimeError(
            "dequant FFN dgrad debug path requires localCTA reconstruction kernels"
        )

    def _expand_outer_sg(sg, rows, cols, *, rowwise):
        outer_tiles = rows // 256 if rowwise else cols // 256
        if sg.numel() != outer_tiles:
            raise RuntimeError(
                "dequant FFN dgrad expected outer-SG payload with "
                f"{outer_tiles} values, got shape={tuple(sg.shape)}"
            )
        expanded = sg.float().reshape(-1).repeat_interleave(2)
        if rowwise:
            return (
                expanded.reshape(rows // 128, 1)
                .expand(rows // 128, cols // 128)
                .contiguous()
            )
        return (
            expanded.reshape(cols // 128, 1)
            .expand(cols // 128, rows // 128)
            .contiguous()
        )

    xhat = None
    if _ffn_capture_path() and input_tensor is not None and inv_rms is not None:
        xhat = input_tensor.float() * inv_rms.float().reshape(-1, 1)

    for split_index in range(2):
        if mode == 'weight':
            if exact_activations is None:
                raise RuntimeError(
                    "weight-only dgrad decomposition requires exact activations"
                )
            dh_dq = exact_activations[split_index]
        else:
            dh_dq = tkq_mod.tk_localcta_reconstruct_row(
                row_fp4s[split_index],
                row_scs[split_index],
                _expand_outer_sg(row_sgs[split_index], M, H, rowwise=True),
            )
        if mode == 'activation':
            if exact_weights is None:
                raise RuntimeError(
                    "activation-only dgrad decomposition requires exact weights"
                )
            weight_dq = exact_weights[split_index]
        else:
            weight_dq = tkq_mod.tk_localcta_reconstruct_col(
                weight_fp4_cols[split_index],
                weight_sc_cols[split_index],
                _expand_outer_sg(
                    weight_sg_cols[split_index], H, K, rowwise=False
                ),
            ).t().contiguous()
        contribution = torch.matmul(
            dh_dq.float(), weight_dq.float()
        ).to(torch.bfloat16)
        if xhat is not None:
            branch_dgamma = (contribution.float() * xhat).sum(dim=0)
            _append_ffn_capture({
                "event": "ffn_dequant_dgrad_branch",
                "debug_name": debug_name,
                "mode": mode,
                "split_index": split_index,
                "d_normed": _tensor_capture_stats(contribution),
                "dgamma": _tensor_capture_stats(branch_dgamma),
            })
            del branch_dgamma
        if split_index == 0:
            output.copy_(contribution)
        else:
            output.add_(contribution)
        del contribution
        if mode != 'weight':
            del dh_dq
        if mode != 'activation':
            del weight_dq
    if xhat is not None:
        del xhat


def _localcta_v4_outer_sg_contract(fp4: torch.Tensor, sg: torch.Tensor) -> bool:
    if not (torch.is_tensor(fp4) and torch.is_tensor(sg)):
        return False
    rows = int(fp4.size(0))
    if rows <= 0 or rows % 256 != 0:
        return False
    tiles = rows // 256
    if sg.dim() == 1:
        return int(sg.numel()) == tiles
    if sg.dim() == 2:
        return tuple(sg.shape) in ((tiles, 1), (1, tiles))
    return False


def _localcta_v4_wo_outer_sg_pair(a_parts, b_parts) -> bool:
    a_fp4, _a_sc, a_sg = a_parts
    b_fp4, _b_sc, b_sg = b_parts
    return (
        _localcta_v4_outer_sg_contract(a_fp4, a_sg)
        and _localcta_v4_outer_sg_contract(b_fp4, b_sg)
    )


def _localcta_v4_wo_outer_sg_gemm(
    tk,
    a_fp4: torch.Tensor,
    a_sc: torch.Tensor,
    a_sg: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sc: torch.Tensor,
    b_sg: torch.Tensor,
    output: torch.Tensor,
) -> None:
    tk_dispatch_gemm(tk, a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, output)


def _ffn_bwd_graphed(
    grad_output, input, norm_weight, inv_rms,
    h13, sg_cat,
    wc_fp4_cols, wc_sc_cols, w2_nvfp4, x_nvfp4, h_nvfp4,
    w1_bf16, w3_bf16, w2_bf16,
    N_dims_13, K, H, M,
    workspace,
    w2_dgrad_col=None,
    w13_dgrad_cols=None,
    h1_raw=None, h3=None, sig_h1=None,
    wc_sg_cols=None,
    dgrad_wc_fp4_cols=None,
    dgrad_wc_sc_cols=None,
    dgrad_wc_sg_cols=None,
    debug_name=None,
    residual_grad=None,
    h_tile: bool = False,
):
    """Graph-captured FFN backward: full dgrad + wgrad + RMSNorm bwd.

    First call per (M, K, H) captures the computation as a CUDA graph.
    Subsequent calls copy inputs to static buffers and replay the graph.
    """
    from .tk_gemm import (
        _get_tk, _get_tk_plain, _get_tk_quant_for_gemm, tk_grouped_wgrad_gemm,
        _launch_rmsnorm_bwd_out_async, _get_sg_tile_indices, _get_wgrad_buf,
        _get_wgrad_stream, _get_rmsnorm_bwd_stream, _record_tensors_on_stream,
    )

    te_fused = _get_te_fused()
    use_localcta = use_tk_localcta() and use_tk_localcta_forward_for_m(M)
    tk = _get_tk() if use_localcta else _get_tk_plain()
    use_ffn_localcta_tk_quant_contract = (
        use_localcta and use_tk_ffn_localcta_tk_quant_contract()
    )
    if use_localcta:
        tkq = _get_tk_quant_for_gemm()
    else:
        tkq = _get_tk_quant()
    tkq_mod = getattr(tkq, '_mod', None)
    use_ffn_localcta_tk_quant_contract = (
        use_localcta and use_tk_ffn_localcta_tk_quant_contract()
    )
    if dgrad_wc_fp4_cols is None:
        dgrad_wc_fp4_cols = wc_fp4_cols
    if dgrad_wc_sc_cols is None:
        dgrad_wc_sc_cols = wc_sc_cols
    if dgrad_wc_sg_cols is None:
        dgrad_wc_sg_cols = wc_sg_cols
    n_groups = len(wc_fp4_cols)
    if use_tk_debug_clone_ffn_grad_output():
        grad_output = grad_output.clone()
    strict_v4_localcta = use_localcta and use_tk_localcta_v4_strict_path()
    localcta_variant = os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower()
    paired_rht_carrier = (
        use_localcta
        and localcta_variant == 'v4'
        and use_tk_localcta_paired_rht_carrier()
    )
    native_paired_rht_split2 = _use_tk_localcta_native_paired_rht_split2(
        tkq_mod,
        paired_rht_carrier=paired_rht_carrier,
    )

    if use_localcta:
        if _nvfp4_quantizer_extras_enabled("activation") and localcta_variant != 'v4':
            _check_nvfp4_native_extras_supported(
                "activation", "localCTA FFN backward activation producer"
            )
        if _nvfp4_quantizer_extras_enabled("grad") and localcta_variant != 'v4':
            _check_nvfp4_native_extras_supported(
                "grad", "localCTA FFN backward grad producer"
            )
        _check_nvfp4_native_extras_supported("weight", "localCTA/v4 FFN backward weight producer")
        localcta_state = _get_ffn_localcta_bwd_state(M, K, H, grad_output.device)
        ffn_timings = [] if use_tk_ffn_debug_timings_for(debug_name) else None
        _debug_check_finite('ffn_bwd.localcta.grad_output', grad_output)
        use_localcta_direct_ffn = (
            use_tk_localcta_direct_contract()
            and not _nvfp4_quantizer_extras_enabled("grad")
        )
        use_split_cache = (
            h1_raw is not None and h3 is not None
            and h1_raw.numel() != 0 and h3.numel() != 0
        )
        use_saved_sigmoid_experiment = (
            not use_tk_localcta_fused()
            and M >= _localcta_ffn_experiment_min_m('USE_TK_LOCALCTA_FFN_SAVED_SIGMOID_MIN_M')
            and use_split_cache
            and sig_h1 is not None
            and sig_h1.numel() != 0
            and use_tk_localcta_ffn_saved_sigmoid()
            and hasattr(te_fused, 'fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax')
        )
        use_saved_sigmoid_overlap = (
            use_saved_sigmoid_experiment
            and use_tk_localcta_ffn_saved_sigmoid_overlap()
        )
        use_w2_dgrad_highacc = (
            use_saved_sigmoid_experiment
            and use_tk_localcta_ffn_saved_sigmoid_w2highacc()
            and hasattr(tk, 'nvfp4_gemm_highacc')
        )
        use_bf16_w2_backward_debug = (
            not use_localcta_direct_ffn
            and use_tk_localcta_ffn_bf16_w2_backward_debug()
        )
        use_bf16_dgrad_debug = (
            not use_localcta_direct_ffn
            and use_tk_localcta_ffn_bf16_dgrad_debug()
        )
        disable_wgrad_stream_env = os.environ.get('USE_TK_FFN_DISABLE_WGRAD_STREAM')
        localcta_v4_bwd = (
            use_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
        )
        if disable_wgrad_stream_env is None:
            disable_wgrad_stream = (
                localcta_v4_bwd
            )
        elif disable_wgrad_stream_env.strip().lower() in ('auto', ''):
            disable_min_m = _localcta_ffn_experiment_min_m(
                'USE_TK_LOCALCTA_FFN_WGRAD_STREAM_DISABLE_MIN_M'
            )
            try:
                disable_min_h = int(
                    os.environ.get('USE_TK_LOCALCTA_FFN_WGRAD_STREAM_DISABLE_MIN_H', '7168')
                )
            except ValueError:
                disable_min_h = 7168
            disable_wgrad_stream = (
                localcta_v4_bwd
                and M >= disable_min_m
                and H >= disable_min_h
            )
        else:
            disable_wgrad_stream = disable_wgrad_stream_env == '1'
        request_localcta_direct_w13_return = (
            use_tk_localcta_v4_ffn_direct_grouped_wgrad_layout()
        )
        use_localcta_direct_w13_return = (
            request_localcta_direct_w13_return
            and localcta_variant == 'v4'
            and (M, K, H) == (32768, 4096, 14336)
        )
        localcta_direct_w13_owner = None
        if use_localcta_direct_w13_return:
            if use_localcta_direct_ffn:
                raise RuntimeError(
                    "localCTA FFN direct W13 return does not support the "
                    "TE-linked direct FFN fallback"
                )
            if not disable_wgrad_stream:
                raise RuntimeError(
                    "localCTA FFN direct W13 return requires caller-stream WGRAD"
                )
            if os.environ.get('USE_TK_FFN_FORCE_SPLIT_WGRAD', '0') == '1':
                raise RuntimeError(
                    "localCTA FFN direct W13 return does not support split WGRAD"
                )
            if not isinstance(debug_name, str) or not debug_name:
                raise RuntimeError(
                    "localCTA FFN direct W13 return requires a production layer owner"
                )
            localcta_direct_w13_owner = (
                'localcta_ffn_direct_w13',
                int(norm_weight.data_ptr()),
                debug_name,
            )
        localcta_grad_w2 = _get_ffn_localcta_owned_grad_buffer(
            'w2', debug_name, norm_weight, (K, H)
        )
        ffn_grad_boost = 1.0
        grad_data_sr = use_nvfp4_data_stochastic_rounding_for_role("grad")
        grad_scale_sr = use_nvfp4_scale_stochastic_rounding_for_role("grad")
        grad_data_sr_axes = _nvfp4_grad_sr_axes() if grad_data_sr else "none"
        ffn_w2_sr_state, ffn_deriv_sr_state = _localcta_ffn_sr_states(
            debug_name, grad_output.device
        )
        grad_output_q = grad_output
        dY_fp4c = dY_scc = dY_sgc = None
        h_fp4c = h_scc = h_sgc = None
        dY_underflow_requant_info = None
        w2_dgrad_silu_split2_q = None
        overlapped_v4_cat_split2 = None
        paired_rht_split2_keepalive = None
        if use_localcta_direct_ffn:
            dY_bf16 = _as_contiguous_bf16(grad_output)
            dY_nvfp4 = _fast_quantize(dY_bf16, tk_swizzle=False, nvfp4_role="grad")
            dh_tmp = tex.generic_gemm(
                w2_nvfp4, False, dY_nvfp4, False,
                None, None, TE_DType[torch.bfloat16],
                None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )[0]
            localcta_state['dh'].copy_(dh_tmp)
            dh = localcta_state['dh']
            _debug_check_finite('ffn_bwd.localcta.dh', dh)

            grad_w2_tmp = tex.generic_gemm(
                h_nvfp4, False, dY_nvfp4, True,
                None, None, TE_DType[torch.bfloat16],
                None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )[0]
            localcta_grad_w2.copy_(grad_w2_tmp)
            grad_w2 = localcta_grad_w2
            _debug_check_finite('ffn_bwd.localcta.grad_w2', grad_w2)
            use_w2_wgrad_overlap = False
            wgrad_stream = None
        elif use_bf16_w2_backward_debug:
            _trace_backend_choice('localcta_ffn_bwd', 'bf16_w2_debug')
            if w2_bf16 is None:
                raise RuntimeError("BF16 W2 debug path requires ctx.w2_bf16 to be saved")
            dh = localcta_state['dh']
            h_bf16 = localcta_state['h_bf16']
            grad_w2 = localcta_grad_w2
            if use_split_cache:
                if hasattr(te_fused, 'fused_silu_mul_bf16_out_no_amax'):
                    te_fused.fused_silu_mul_bf16_out_no_amax(h1_raw, h3, h_bf16)
                elif hasattr(te_fused, 'fused_silu_mul_bf16_out'):
                    te_fused.fused_silu_mul_bf16_out(
                        h1_raw, h3, h_bf16, localcta_state['amax1']
                    )
                else:
                    h_bf16_tmp, _ = te_fused.fused_silu_mul_bf16(h1_raw, h3)
                    h_bf16.copy_(h_bf16_tmp)
            else:
                h_bf16_tmp, _ = te_fused.fused_silu_mul_strided_bf16(h13, H)
                h_bf16.copy_(h_bf16_tmp)
            dh.copy_(torch.matmul(grad_output.float(), w2_bf16.float()).to(torch.bfloat16))
            grad_w2.copy_(torch.matmul(grad_output.float().t(), h_bf16.float()).to(torch.bfloat16))
            _debug_check_finite('ffn_bwd.localcta.dh', dh)
            _debug_check_finite('ffn_bwd.localcta.grad_w2', grad_w2)
            use_w2_wgrad_overlap = False
            wgrad_stream = None
        else:
            ffn_grad_boost = get_tk_localcta_ffn_fixed_grad_boost()
            localcta_direct_tk = _get_tk_localcta_direct() if use_tk_localcta() else None
            dY_underflow_requant_info = None
            use_direct_w2_localcta = False
            use_nofold_direct_w2_operands = False
            if ffn_grad_boost != 1.0:
                grad_output_q = (grad_output.float() * ffn_grad_boost).to(torch.bfloat16).contiguous()
                _trace_backend_choice('localcta_ffn_bwd', f'adaptive_grad_scale_{ffn_grad_boost:g}')
            # Borrow grad_output directly when the localCTA descriptor accepts it,
            # and fall back to the cached staging buffer only when it does not.
            _tk_stage_trace('ffn_bwd_localcta_sub', 'dy_quant_start', debug_name)
            with _ffn_cuda_timed(ffn_timings, "dy_quant"):
                if _nvfp4_quantizer_extras_enabled("grad") and localcta_variant == 'v4':
                    dY_nvfp4 = _fast_quantize_localcta_v4_opt(
                        grad_output_q,
                        nvfp4_role="grad",
                        persistent_rng_state=ffn_w2_sr_state,
                    )
                    dY_fp4, dY_sc, dY_sg = dY_nvfp4._tk_row
                    dY_fp4c, dY_scc, dY_sgc = dY_nvfp4._tk_col
                    dY_quant = (dY_fp4, dY_sc, dY_fp4c, dY_scc, dY_sg, dY_sgc)
                elif hasattr(tkq, 'tk_quantize_for_gemm_maybe_borrow'):
                    dY_quant = tkq.tk_quantize_for_gemm_maybe_borrow(
                        grad_output_q, localcta_state['dY_bf16'], True, True
                    )
                    dY_fp4, dY_sc, dY_sg = dY_quant[0], dY_quant[1], dY_quant[4]
                    dY_fp4c, dY_scc = dY_quant[2], dY_quant[3]
                    dY_sgc = dY_quant[5] if len(dY_quant) > 5 and torch.is_tensor(dY_quant[5]) and dY_quant[5].numel() > 0 else dY_sg
                else:
                    localcta_state['dY_bf16'].copy_(grad_output_q)
                    dY_quant = tkq.tk_quantize_for_gemm(localcta_state['dY_bf16'], True)
                    dY_fp4, dY_sc, dY_sg = dY_quant[0], dY_quant[1], dY_quant[4]
                    dY_fp4c, dY_scc = dY_quant[2], dY_quant[3]
                    dY_sgc = dY_quant[5] if len(dY_quant) > 5 and torch.is_tensor(dY_quant[5]) and dY_quant[5].numel() > 0 else dY_sg
            _tk_stage_trace('ffn_bwd_localcta_sub', 'dy_quant_done', debug_name)
            _debug_check_finite('ffn_bwd.localcta.dY_sc', dY_sc)
            _debug_check_finite('ffn_bwd.localcta.dY_sg', dY_sg)

            dY_sc_all_zero = False
            if use_tk_localcta_ffn_check_zero_dy_sc():
                dY_sc_all_zero = _scale_bytes_all_zero(dY_sc)

            if dY_sc_all_zero:
                if (
                    localcta_direct_tk is not None
                    and hasattr(localcta_direct_tk, 'nvfp4_gemm')
                    and hasattr(tkq, 'tk_quantize_for_gemm_prepared_nofold_maybe_borrow')
                ):
                    dY_quant = tkq.tk_quantize_for_gemm_prepared_nofold_maybe_borrow(
                        grad_output_q, localcta_state['dY_bf16'], True, True
                    )
                    dY_fp4, dY_sc, dY_sg = dY_quant[0], dY_quant[1], dY_quant[4]
                    dY_fp4c, dY_scc = dY_quant[2], dY_quant[3]
                    dY_sgc = (
                        dY_quant[5]
                        if len(dY_quant) > 5 and torch.is_tensor(dY_quant[5]) and dY_quant[5].numel() > 0
                        else dY_sg
                    )
                    _trace_backend_choice('localcta_ffn_bwd', 'prepared_nofold_underflow_requant')
                    dY_underflow_requant_info = {
                        "taken": True,
                        "reason": "zero_dY_sc",
                        "path": "prepared_nofold_underflow_requant",
                        "fused": use_tk_localcta_fused(),
                    }
                    use_direct_w2_localcta = True
                    use_nofold_direct_w2_operands = True
                elif localcta_direct_tk is not None and hasattr(localcta_direct_tk, 'nvfp4_gemm'):
                    dY_quant = tkq.tk_quantize_for_gemm_direct(grad_output_q, True, True)
                    dY_fp4, dY_sc, dY_sg = dY_quant[0], dY_quant[1], dY_quant[4]
                    dY_fp4c, dY_scc = dY_quant[2], dY_quant[3]
                    dY_sgc = (
                        dY_quant[5]
                        if len(dY_quant) > 5 and torch.is_tensor(dY_quant[5]) and dY_quant[5].numel() > 0
                        else dY_sg
                    )
                    _trace_backend_choice('localcta_ffn_bwd', 'raw_dy_underflow_requant')
                    dY_underflow_requant_info = {
                        "taken": True,
                        "reason": "zero_dY_sc",
                        "path": "raw_localcta_underflow_requant",
                        "fused": use_tk_localcta_fused(),
                    }
                    use_direct_w2_localcta = True
                if not h_tile and use_tk_localcta_ffn_bf16_rescue_on_zero_dy_sc():
                    if w1_bf16 is None or w3_bf16 is None or w2_bf16 is None:
                        raise RuntimeError(
                            "FFN BF16 underflow rescue requires ctx.w1_bf16, ctx.w3_bf16, and ctx.w2_bf16"
                        )
                    _trace_backend_choice('localcta_ffn_bwd', 'bf16_zero_dysc_rescue')
                    grad_input, grad_w1, grad_w3, grad_w2, dgamma = _localcta_ffn_backward_bf16_rescue(
                        grad_output,
                        input,
                        norm_weight,
                        inv_rms,
                        h1_raw,
                        h3,
                        w1_bf16,
                        w3_bf16,
                        w2_bf16,
                    )
                    return (
                        grad_input,
                        grad_w1,
                        grad_w3,
                        grad_w2,
                        dgamma,
                        {
                            "taken": True,
                            "reason": "zero_dY_sc",
                            "path": "bf16_underflow_rescue",
                            "fused": use_tk_localcta_fused(),
                        },
                    )
                if not use_tk_localcta_fused():
                    grad_w1 = localcta_state['grad_w1']
                    grad_w3 = localcta_state['grad_w3']
                    grad_w2 = localcta_grad_w2
                    d_normed = localcta_state['d_normed']
                    grad_input = localcta_state['grad_input']
                    dgamma = localcta_state['dgamma']
                    grad_w1.zero_()
                    grad_w3.zero_()
                    grad_w2.zero_()
                    d_normed.zero_()
                    grad_input.zero_()
                    dgamma.zero_()
                    return grad_input, grad_w1, grad_w3, grad_w2, dgamma, {
                        "taken": False,
                        "reason": "zero_dY_sc",
                        "path": "legacy_zero_return",
                        "fused": False,
                    }

            w2_fp4c, w2_scc, w2_sgc = w2_nvfp4._tk_col
            h_fp4c, h_scc, h_sgc = h_nvfp4._tk_col
            if use_nofold_direct_w2_operands:
                h_bf16 = localcta_state['h_bf16']
                if use_split_cache:
                    if hasattr(te_fused, 'fused_silu_mul_bf16_out_no_amax'):
                        te_fused.fused_silu_mul_bf16_out_no_amax(h1_raw, h3, h_bf16)
                    elif hasattr(te_fused, 'fused_silu_mul_bf16_out'):
                        te_fused.fused_silu_mul_bf16_out(
                            h1_raw, h3, h_bf16, localcta_state['amax1']
                        )
                    else:
                        h_bf16_tmp, _ = te_fused.fused_silu_mul_bf16(h1_raw, h3)
                        h_bf16.copy_(h_bf16_tmp)
                else:
                    h_bf16_tmp, _ = te_fused.fused_silu_mul_strided_bf16(h13, H)
                    h_bf16.copy_(h_bf16_tmp)
                h_quant = tkq.tk_quantize_for_gemm_prepared_nofold_maybe_borrow(
                    h_bf16, h_bf16, True, True
                )
                h_fp4c, h_scc = h_quant[2], h_quant[3]
                h_sgc = (
                    h_quant[5]
                    if len(h_quant) > 5 and torch.is_tensor(h_quant[5]) and h_quant[5].numel() > 0
                    else h_quant[4]
                )
                w2_bf16_cached = localcta_state['w2_bf16']
                w2_bf16_cached.copy_(w2_bf16)
                w2_quant = tkq.tk_quantize_for_gemm_prepared_nofold_maybe_borrow(
                    w2_bf16_cached, w2_bf16_cached, True, True
                )
                w2_fp4c, w2_scc = w2_quant[2], w2_quant[3]
                w2_sgc = (
                    w2_quant[5]
                    if len(w2_quant) > 5 and torch.is_tensor(w2_quant[5]) and w2_quant[5].numel() > 0
                    else w2_quant[4]
                )
                _trace_backend_choice('localcta_ffn_bwd', 'underflow_requant_consistent_w2_operands')
            use_v4_raw_w2_dgrad = (
                use_tk_localcta()
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and os.environ.get('USE_TK_LOCALCTA_V4_SG_DIRECT_CONSUMERS', '0') == '1'
                and not use_w2_dgrad_highacc
                and w2_dgrad_col is not None
                and not use_direct_w2_localcta
            )
            w2_dgrad_silu_split2_q = None
            use_w2_dgrad_silu_producer = (
                use_tk_localcta()
                and localcta_variant == 'v4'
                and use_tk_localcta_v4_w2_dgrad_silu_producer()
                and use_split_cache
                and ffn_grad_boost == 1.0
                and not use_saved_sigmoid_experiment
                and not use_direct_w2_localcta
                and not use_v4_raw_w2_dgrad
                and not use_w2_dgrad_highacc
                and not _nvfp4_quantizer_extras_enabled("grad")
                and hasattr(tk, 'nvfp4_w2_dgrad_silu_quant_gemm')
                and tkq_mod is not None
                and hasattr(tkq_mod, 'tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc')
                and (M % 256) == 0
                and (K % 256) == 0
                and (H % 256) == 0
                and tk_localcta_v4_w2_dgrad_silu_producer_shape_safe(M, H)
            )
            dh = _get_ffn_localcta_dh_scratch(
                localcta_state,
                fused_producer=use_w2_dgrad_silu_producer,
            )
            if use_w2_dgrad_silu_producer:
                if use_tk_localcta_v4_w2_dgrad_silu_producer_fresh_payload():
                    with _ffn_cuda_timed(ffn_timings, "w2_dgrad_silu_split2_alloc"):
                        w2_dgrad_silu_split2_q = tkq_mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
                            M, H, H, grad_output.device
                        )
                    localcta_state.setdefault(
                        'w2_dgrad_silu_split2_q_retained_payloads', []
                    ).append(w2_dgrad_silu_split2_q)
                else:
                    w2_dgrad_silu_split2_q = localcta_state.get('w2_dgrad_silu_split2_q_bufs')
                    if w2_dgrad_silu_split2_q is None:
                        with _ffn_cuda_timed(ffn_timings, "w2_dgrad_silu_split2_alloc"):
                            w2_dgrad_silu_split2_q = tkq_mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
                                M, H, H, grad_output.device
                            )
                            localcta_state['w2_dgrad_silu_split2_q_bufs'] = w2_dgrad_silu_split2_q
            _tk_stage_trace('ffn_bwd_localcta_sub', 'w2_dgrad_start', debug_name)
            with _ffn_cuda_timed(ffn_timings, "w2_dgrad"):
                if use_w2_dgrad_silu_producer:
                    if localcta_state['w2_dgrad_silu_payload_ready_recorded']:
                        torch.cuda.current_stream().wait_event(
                            localcta_state['w2_dgrad_silu_payload_ready_event']
                        )
                    if not use_tk_localcta_v4_w2_dgrad_silu_producer_unit_sg():
                        w2_dgrad_silu_a_sg = _prepare_localcta_v4_outer_sg_for_direct(
                            dY_sg,
                            max(M // 256, 1),
                            grad_output.device,
                            True,
                        )
                        w2_dgrad_silu_b_sg = _prepare_localcta_v4_outer_sg_for_direct(
                            w2_sgc,
                            max(H // 256, 1),
                            grad_output.device,
                            False,
                        )
                    else:
                        w2_dgrad_silu_a_sg = _prepare_localcta_v4_outer_sg_for_direct(
                            localcta_state['unit_sg_m256'],
                            max(M // 256, 1),
                            grad_output.device,
                            True,
                        )
                        w2_dgrad_silu_b_sg = _prepare_localcta_v4_outer_sg_for_direct(
                            localcta_state['unit_sg_h256'],
                            max(H // 256, 1),
                            grad_output.device,
                            False,
                        )
                    if (
                        os.environ.get('USE_TK_DEBUG_FFN_W2_PRODUCER_INPUTS', '0') == '1'
                        and _ffn_capture_path()
                    ):
                        _append_ffn_capture({
                            "event": "ffn_w2_dgrad_silu_producer_inputs",
                            "debug_name": debug_name,
                            "M": int(M),
                            "K": int(K),
                            "H": int(H),
                            "config_id": int(tk_localcta_v4_w2_dgrad_silu_producer_config_id()),
                            "dY_fp4": _tensor_capture_stats(dY_fp4),
                            "dY_sc": _tensor_capture_stats(dY_sc),
                            "dY_sg": _tensor_capture_stats(w2_dgrad_silu_a_sg),
                            "w2_fp4c": _tensor_capture_stats(w2_fp4c),
                            "w2_scc": _tensor_capture_stats(w2_scc),
                            "w2_sgc": _tensor_capture_stats(w2_dgrad_silu_b_sg),
                            "h1_raw": _tensor_capture_stats(h1_raw),
                            "h3": _tensor_capture_stats(h3),
                        })
                    producer_tensors = (
                        dY_fp4, dY_sc, w2_dgrad_silu_a_sg,
                        w2_fp4c, w2_scc, w2_dgrad_silu_b_sg,
                        h3, h1_raw,
                        w2_dgrad_silu_split2_q[6:12],
                    )
                    caller_stream = torch.cuda.current_stream(grad_output.device)
                    if use_tk_localcta_v4_w2_dgrad_silu_producer_priority_stream():
                        producer_stream = _get_ffn_localcta_w2_producer_stream(
                            grad_output.device
                        )
                        producer_stream.wait_stream(caller_stream)
                        with torch.cuda.stream(producer_stream):
                            _record_tensors_on_stream(producer_tensors, producer_stream)
                            tk.nvfp4_w2_dgrad_silu_quant_gemm(
                                dY_fp4, dY_sc, w2_dgrad_silu_a_sg,
                                w2_fp4c, w2_scc, w2_dgrad_silu_b_sg,
                                h3, h1_raw,
                                w2_dgrad_silu_split2_q[6],
                                w2_dgrad_silu_split2_q[7],
                                w2_dgrad_silu_split2_q[8],
                                w2_dgrad_silu_split2_q[9],
                                w2_dgrad_silu_split2_q[10],
                                w2_dgrad_silu_split2_q[11],
                                tk_localcta_v4_w2_dgrad_silu_producer_config_id(),
                            )
                        caller_stream.wait_stream(producer_stream)
                    else:
                        tk.nvfp4_w2_dgrad_silu_quant_gemm(
                            dY_fp4, dY_sc, w2_dgrad_silu_a_sg,
                            w2_fp4c, w2_scc, w2_dgrad_silu_b_sg,
                            h3, h1_raw,
                            w2_dgrad_silu_split2_q[6],
                            w2_dgrad_silu_split2_q[7],
                            w2_dgrad_silu_split2_q[8],
                            w2_dgrad_silu_split2_q[9],
                            w2_dgrad_silu_split2_q[10],
                            w2_dgrad_silu_split2_q[11],
                            tk_localcta_v4_w2_dgrad_silu_producer_config_id(),
                        )
                        _record_tensors_on_stream(producer_tensors, caller_stream)
                    if _release_delayed_fsdp_backward_prefetch(
                        grad_output.device
                    ):
                        _trace_backend_choice(
                            'localcta_ffn_bwd',
                            'released_delayed_fsdp_prefetch',
                        )
                    dh = None
                    _trace_backend_choice('localcta_ffn_bwd', 'w2_dgrad_silu_split2_producer')
                elif use_v4_raw_w2_dgrad:
                    tk_v4_direct_raw_dgrad_gemm(grad_output_q, w2_dgrad_col, dh)
                else:
                    w2_dgrad_gemm = (
                        localcta_direct_tk.nvfp4_gemm
                        if use_direct_w2_localcta and localcta_direct_tk is not None
                        else (tk.nvfp4_gemm_highacc if use_w2_dgrad_highacc else tk.nvfp4_gemm)
                    )
                    w2_dgrad_gemm(
                        dY_fp4, dY_sc, dY_sg, w2_fp4c, w2_scc, w2_sgc, dh
                    )
            _tk_stage_trace('ffn_bwd_localcta_sub', 'w2_dgrad_done', debug_name)
            if ffn_grad_boost != 1.0:
                dh.div_(ffn_grad_boost)
            _debug_check_finite('ffn_bwd.localcta.dh', dh)

            prefer_v4_raw_w13_dgrad_for_overlap = (
                not strict_v4_localcta
                and localcta_variant == 'v4'
                and w13_dgrad_cols is not None
                and os.environ.get('USE_TK_LOCALCTA_V4_SG_DIRECT_CONSUMERS', '0') == '1'
                and use_tk_localcta_v4_raw_backward_fallbacks(M)
            )
            overlap_v4_cat_split2 = (
                use_tk_localcta_v4_ffn_deriv_w2_wgrad_overlap()
                and not paired_rht_carrier
                and not strict_v4_localcta
                and use_ffn_localcta_tk_quant_contract
                and use_split_cache
                and not use_saved_sigmoid_experiment
                and w2_dgrad_silu_split2_q is None
                and not prefer_v4_raw_w13_dgrad_for_overlap
                and not use_tk_localcta_v4_ffn_prepared_split2_producer()
                and hasattr(tkq, 'tk_silu_deriv_quantize_split_for_gemm')
                and tkq_mod is not None
                and hasattr(tkq_mod, 'tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc')
                and hasattr(tkq_mod, 'tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace')
            )
            if overlap_v4_cat_split2:
                strict_split2_q_overlap = localcta_state.get('v4_cat_split2_q_bufs')
                if strict_split2_q_overlap is None:
                    with _ffn_cuda_timed(ffn_timings, "split2_alloc"):
                        strict_split2_q_overlap = tkq_mod.tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc(
                            M, H, grad_output.device
                        )
                        localcta_state['v4_cat_split2_q_bufs'] = strict_split2_q_overlap
                dh1_overlap, dh3_overlap = _get_ffn_localcta_deriv_outputs(
                    localcta_state, h1_raw, h3
                )
                deriv_stream = _get_ffn_localcta_deriv_stream(grad_output.device)
                deriv_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(deriv_stream):
                    _record_tensors_on_stream(
                        (
                            dh, h1_raw, h3, dh1_overlap, dh3_overlap,
                            strict_split2_q_overlap,
                        ),
                        deriv_stream,
                    )
                    with _ffn_cuda_timed(
                        ffn_timings,
                        "split2_fused_tk_split_producer_overlap",
                        deriv_stream,
                    ):
                        fused_overlap = _call_with_optional_localcta_sr_state(
                            tkq_mod.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace,
                            dh, h3, h1_raw,
                            dh1_overlap, dh3_overlap,
                            *strict_split2_q_overlap[:16],
                            True,
                            grad_data_sr,
                            grad_scale_sr,
                            _nvfp4_rng_seed(),
                            _nvfp4_rng_subsequence_base(),
                            grad_data_sr_axes,
                            persistent_rng_state=ffn_deriv_sr_state,
                        )
                    deriv_stream.record_event(localcta_state['deriv_quant_ready_event'])
                overlapped_v4_cat_split2 = (
                    fused_overlap,
                    strict_split2_q_overlap,
                )
                _trace_backend_choice(
                    'localcta_ffn_bwd',
                    'v4_cat_split2_w2_wgrad_overlap',
                )

            use_w2_wgrad_overlap = (
                M >= _localcta_ffn_experiment_min_m('USE_TK_LOCALCTA_FFN_W2_WGRAD_OVERLAP_MIN_M')
                and not use_saved_sigmoid_experiment
                and not disable_wgrad_stream
            )

            grad_w2 = localcta_grad_w2
            wgrad_stream = _get_wgrad_stream() if use_w2_wgrad_overlap else None
            use_v4_fast_w2_wgrad = (
                use_tk_localcta()
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and use_tk_localcta_v4_fast_w2_wgrad()
            )
            use_v4_raw_w2_wgrad = (
                use_tk_localcta()
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and not use_ffn_localcta_tk_quant_contract
                and use_tk_localcta_v4_raw_backward_fallbacks(grad_output.size(0))
            )
            use_v4_virtual_w2_wgrad = (
                use_tk_localcta()
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and use_tk_localcta_v4_gemm_virtual_rescale()
                and hasattr(tk, 'nvfp4_gemm_virtual_rescale')
                and not use_nofold_direct_w2_operands
                and _has_virtual_rescale_chunk(h_nvfp4, False)
            )
            empty_dy_chunk_sg = _empty_chunk_sg(dY_fp4c.device) if use_v4_virtual_w2_wgrad else None
            h_col_chunk_sg = (
                _chunk_sg_or_empty(h_nvfp4, False, h_fp4c)
                if use_v4_virtual_w2_wgrad
                else None
            )
            w2_wgrad_timing_stream = wgrad_stream if use_w2_wgrad_overlap else None
            _tk_stage_trace('ffn_bwd_localcta_sub', 'w2_wgrad_start', debug_name)
            with _ffn_cuda_timed(ffn_timings, "w2_wgrad", w2_wgrad_timing_stream):
                if use_w2_wgrad_overlap:
                    wgrad_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(wgrad_stream):
                        extra_virtual_tensors = (
                            (empty_dy_chunk_sg, h_col_chunk_sg)
                            if use_v4_virtual_w2_wgrad
                            else ()
                        )
                        _record_tensors_on_stream(
                            (
                                grad_output_q, dY_fp4c, dY_scc, dY_sgc,
                                h_fp4c, h_scc, h_sgc, grad_w2,
                                *extra_virtual_tensors,
                            ),
                            wgrad_stream,
                        )
                        if use_v4_virtual_w2_wgrad:
                            tk.nvfp4_gemm_virtual_rescale(
                                dY_fp4c, dY_scc, dY_sgc, empty_dy_chunk_sg,
                                h_fp4c, h_scc, h_sgc, h_col_chunk_sg,
                                grad_w2,
                            )
                        elif use_v4_fast_w2_wgrad:
                            tk.nvfp4_gemm_fast(
                                dY_fp4c, dY_scc, dY_sgc,
                                h_fp4c, h_scc, h_sgc,
                                grad_w2,
                            )
                        elif use_v4_raw_w2_wgrad:
                            tk_v4_direct_wgrad_col_gemm(
                                dY_fp4c, dY_scc, dY_sgc,
                                h_fp4c, h_scc, h_sgc,
                                grad_w2,
                            )
                        else:
                            (localcta_direct_tk.nvfp4_gemm if use_direct_w2_localcta and localcta_direct_tk is not None else tk.nvfp4_gemm)(
                                dY_fp4c, dY_scc, dY_sgc, h_fp4c, h_scc, h_sgc, grad_w2
                            )
                else:
                    if use_v4_virtual_w2_wgrad:
                        tk.nvfp4_gemm_virtual_rescale(
                            dY_fp4c, dY_scc, dY_sgc, empty_dy_chunk_sg,
                            h_fp4c, h_scc, h_sgc, h_col_chunk_sg,
                            grad_w2,
                        )
                    elif use_v4_fast_w2_wgrad:
                        tk.nvfp4_gemm_fast(
                            dY_fp4c, dY_scc, dY_sgc,
                            h_fp4c, h_scc, h_sgc,
                            grad_w2,
                        )
                    elif use_v4_raw_w2_wgrad:
                        tk_v4_direct_wgrad_col_gemm(
                            dY_fp4c, dY_scc, dY_sgc,
                            h_fp4c, h_scc, h_sgc,
                            grad_w2,
                        )
                    else:
                        (localcta_direct_tk.nvfp4_gemm if use_direct_w2_localcta and localcta_direct_tk is not None else tk.nvfp4_gemm)(
                            dY_fp4c, dY_scc, dY_sgc, h_fp4c, h_scc, h_sgc, grad_w2
                        )
            _tk_stage_trace('ffn_bwd_localcta_sub', 'w2_wgrad_done', debug_name)
            _debug_check_finite('ffn_bwd.localcta.grad_w2', grad_w2)

        split2_grad_boost = 1.0
        dh_localcta = dh
        if not use_localcta_direct_ffn:
            split2_grad_boost = 1.0

        if use_localcta_direct_ffn:
            _trace_backend_choice('localcta_ffn_bwd', 'direct_two_gemm')
            if use_split_cache:
                dh1 = localcta_state['dh1']
                dh3_out = localcta_state['dh3']
                if use_saved_sigmoid_experiment:
                    te_fused.fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax(
                        dh, h3, h1_raw, sig_h1,
                        dh1, dh3_out,
                    )
                elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
                    te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                        dh, h3, h1_raw,
                        dh1, dh3_out,
                    )
                elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out'):
                    te_fused.fused_silu_deriv_dual_mul_bf16_out(
                        dh, h3, h1_raw,
                        dh1, dh3_out,
                        localcta_state['amax1'], localcta_state['amax2'],
                    )
                else:
                    dh1, dh3_out, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1_raw)
            else:
                dh1, dh3_out, _, _ = te_fused.fused_silu_deriv_dual_mul_strided_bf16(dh, h13)
            _debug_check_finite('ffn_bwd.localcta.dh1', dh1)
            _debug_check_finite('ffn_bwd.localcta.dh3', dh3_out)

            split2_q = tkq.tk_batched_quantize_for_gemm([dh1, dh3_out], True, True)
            row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs = split2_q[:6]
            col_fp4_full = torch.cat(
                [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in col_fp4s], dim=0
            ).view(torch.float4_e2m1fn_x2)
            col_sc_cat = torch.cat(
                [sc.contiguous().view(torch.uint8) for sc in col_scs], dim=0
            ).view(torch.float8_e4m3fn)
            col_sg_cat = torch.cat(col_sgs, dim=0)
            grad_w13 = tk_grouped_wgrad_gemm(
                (col_fp4s, col_scs, col_sgs, col_fp4_full, col_sc_cat, col_sg_cat),
                x_nvfp4,
                N_dims_13,
            )
            grad_w1, grad_w3 = grad_w13.split(H, dim=0)
            _debug_check_finite('ffn_bwd.localcta.grad_w1', grad_w1)
            _debug_check_finite('ffn_bwd.localcta.grad_w3', grad_w3)

            if wc_sg_cols is None:
                raise RuntimeError("localCTA FFN backward requires per-split weight col SG tensors")
            d_normed = localcta_state['d_normed']
            d_normed_tmp = localcta_state['d_normed_tmp']
            # localCTA FFN dgrad kernels may accumulate into the destination.
            # These buffers are cached across calls, so clear them before reuse.
            d_normed.zero_()
            d_normed_tmp.zero_()
            if use_tk_localcta_ffn_dequant_dgrad_debug():
                _trace_backend_choice('localcta_ffn_dgrad', 'dequant_debug')
                dequant_mode = get_tk_localcta_ffn_dequant_dgrad_debug_mode()
                _localcta_ffn_dequant_split2_dgrad(
                    tkq_mod,
                    row_fp4s,
                    row_scs,
                    row_sgs,
                    dgrad_wc_fp4_cols,
                    dgrad_wc_sc_cols,
                    dgrad_wc_sg_cols,
                    d_normed,
                    M,
                    H,
                    K,
                    mode=dequant_mode,
                    exact_activations=(dh1, dh3_out),
                    exact_weights=(w1_bf16, w3_bf16),
                    input_tensor=input,
                    inv_rms=inv_rms,
                    debug_name=debug_name,
                )
            elif use_bf16_dgrad_debug:
                _trace_backend_choice('localcta_ffn_dgrad', 'bf16_debug')
                if w1_bf16 is None or w3_bf16 is None:
                    raise RuntimeError("BF16 FFN dgrad debug path requires ctx.w1_bf16 and ctx.w3_bf16")
                d_normed.copy_(
                    torch.matmul(dh1.float(), w1_bf16.float()).to(torch.bfloat16)
                )
                d_normed_tmp.copy_(
                    torch.matmul(dh3_out.float(), w3_bf16.float()).to(torch.bfloat16)
                )
                d_normed.add_(d_normed_tmp)
            else:
                tk.nvfp4_gemm(
                    row_fp4s[0], row_scs[0], row_sgs[0],
                    wc_fp4_cols[0], wc_sc_cols[0], wc_sg_cols[0],
                    d_normed,
                )
                tk.nvfp4_gemm(
                    row_fp4s[1], row_scs[1], row_sgs[1],
                    wc_fp4_cols[1], wc_sc_cols[1], wc_sg_cols[1],
                    d_normed_tmp,
                )
                d_normed.add_(d_normed_tmp)
            _debug_check_finite('ffn_bwd.localcta.d_normed', d_normed)

            if h_tile:
                from .tk_gemm import tk_h_tile_backward
                grad_input, dgamma = tk_h_tile_backward(
                    d_normed, input, norm_weight, inv_rms
                )
            else:
                rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                    d_normed, input, norm_weight, inv_rms, te_fused,
                    tag='ffn',
                    residual_grad=residual_grad,
                )
                torch.cuda.current_stream().wait_stream(rms_stream)
                grad_input = rms_state['grad_input']
                dgamma = rms_state.get('dgamma_out', rms_state['dgamma'])
            _debug_check_finite('ffn_bwd.localcta.grad_input', grad_input)
            _debug_check_finite('ffn_bwd.localcta.dgamma', dgamma)
            return grad_input, grad_w1, grad_w3, grad_w2, _as_param_grad_dtype(dgamma, norm_weight), None

        use_native_v4_split_contract = strict_v4_localcta
        use_direct_split2 = (
            not use_native_v4_split_contract
            and not use_ffn_localcta_tk_quant_contract
            and not _nvfp4_quantizer_extras_enabled("grad")
            and
            use_split_cache
            and use_tk_localcta_ffn_direct_split2()
            and tkq_mod is not None
            and hasattr(tkq_mod, "tk_localcta_group_quantize_dim1_split2_for_gemm_prepared")
            and hasattr(tk, "nvfp4_split2_dgrad_strided_onepass_gemm")
        )
        use_two_single_row_split2_quant = (
            not use_native_v4_split_contract
            and not use_ffn_localcta_tk_quant_contract
            and not _nvfp4_quantizer_extras_enabled("grad")
            and
            not use_direct_split2
            and use_split_cache
            and tkq_mod is not None
            and hasattr(tkq_mod, "tk_localcta_quantize_for_gemm_prepared_alloc")
            and hasattr(tkq_mod, "tk_localcta_quantize_for_gemm_prepared_launch")
            and hasattr(tkq_mod, "tk_localcta_quantize_col_only_prepared_launch_inplace")
            and hasattr(tk, "nvfp4_split2_dgrad_onepass_gemm")
        )
        use_fused_split2_quant = (
            use_split_cache
            and use_direct_split2
            and tkq_mod is not None
            and hasattr(tkq_mod, "tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace")
        )
        if os.environ.get('USE_TK_LOCALCTA_FFN_DISABLE_FUSED_SPLIT2_QUANT', '0') == '1':
            use_fused_split2_quant = False
        use_row_only_split2_quant = (
            use_saved_sigmoid_overlap
            and use_direct_split2
            and tkq_mod is not None
            and hasattr(tkq_mod, "tk_localcta_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace")
            and hasattr(tkq_mod, "tk_localcta_quantize_col_only_prepared_launch_inplace")
        )
        use_fused_row_producer_split2_quant = (
            use_split_cache
            and use_direct_split2
            and M >= _localcta_ffn_experiment_min_m('USE_TK_LOCALCTA_FFN_ROW_PRODUCER_MIN_M')
            and not use_saved_sigmoid_experiment
            and use_tk_localcta_ffn_fused_row_producer()
            and tkq_mod is not None
            and hasattr(tkq_mod, "tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace")
            and hasattr(tkq_mod, "tk_localcta_quantize_col_only_prepared_launch_inplace")
        )
        use_v4_split2_two_stage = (
            use_split_cache
            and use_direct_split2
            and M >= _localcta_ffn_experiment_min_m('USE_TK_LOCALCTA_V4_SPLIT2_TWO_STAGE_MIN_M')
            and not use_saved_sigmoid_experiment
            and use_tk_localcta_v4_split2_two_stage()
            and tkq_mod is not None
            and hasattr(tkq_mod, "tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace")
        )
        if use_v4_split2_two_stage:
            _trace_backend_choice('localcta_ffn_bwd', 'v4_split2_two_stage')
        elif use_direct_split2:
            _trace_backend_choice('localcta_ffn_bwd', 'split2_onepass')
        elif use_two_single_row_split2_quant:
            _trace_backend_choice('localcta_ffn_bwd', 'two_single_row_split2')
        else:
            _trace_backend_choice('localcta_ffn_bwd', 'legacy_path')
        single_row_q = None
        dh1 = None
        dh3_out = None
        strict_v4_raw_split2_consumers = False
        has_v4_split2_strided_sg = hasattr(tk, 'nvfp4_split2_dgrad_strided_onepass_gemm_sg')
        has_v4_split2_strided_outer_sg = hasattr(tk, 'nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg')
        has_v4_prepared_split2_finalizer = (
            tkq_mod is not None
            and hasattr(tkq_mod, 'tk_localcta_finalize_split2_for_gemm_prepared_inplace')
        )
        use_v4_prepared_split2_finalizer = (
            has_v4_split2_strided_outer_sg
            and has_v4_prepared_split2_finalizer
            and os.environ.get('USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_FINALIZE', '1') == '1'
        )
        use_v4_split2_strided_sg_dgrad = (
            has_v4_split2_strided_sg
            and os.environ.get('USE_TK_LOCALCTA_V4_FFN_STRIDED_SG_DGRAD', '0') == '1'
        )
        strict_v4_prepared_split2_finalized = False

        use_fast_v4_prepared_split2 = (
            use_ffn_localcta_tk_quant_contract
            and use_tk_localcta_v4_ffn_prepared_split2_producer()
            and use_v4_split2_strided_sg_dgrad
            and use_split_cache
            and tkq_mod is not None
            and hasattr(tkq_mod, 'tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc')
            and hasattr(tkq_mod, 'tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace')
            and (has_v4_split2_strided_sg or hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm'))
        )
        use_strict_v4_prepared_split2 = (
            (
                strict_v4_localcta
                and use_tk_localcta_v4_strict_prepared_split2()
                and use_v4_split2_strided_sg_dgrad
            )
            or use_fast_v4_prepared_split2
        ) and use_split_cache and (
            tkq_mod is not None
            and hasattr(tkq_mod, 'tk_localcta_silu_deriv_split_bf16_launch_inplace')
            and hasattr(tkq_mod, 'tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc')
            and hasattr(tkq_mod, 'tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace')
            and (has_v4_split2_strided_sg or hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm'))
        )
        if _nvfp4_quantizer_extras_enabled("grad"):
            use_strict_v4_prepared_split2 = False
        if _ffn_capture_path():
            _append_ffn_capture({
                "event": "ffn_split2_path_flags",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "strict_v4_localcta": bool(strict_v4_localcta),
                "use_split_cache": bool(use_split_cache),
                "strict_prepared_env": bool(use_tk_localcta_v4_strict_prepared_split2()),
                "fast_prepared_env": bool(use_tk_localcta_v4_ffn_prepared_split2_producer()),
                "use_fast_v4_prepared_split2": bool(use_fast_v4_prepared_split2),
                "use_strict_v4_prepared_split2": bool(use_strict_v4_prepared_split2),
                "has_split_bf16": bool(tkq_mod is not None and hasattr(tkq_mod, 'tk_localcta_silu_deriv_split_bf16_launch_inplace')),
                "has_prepared_alloc": bool(tkq_mod is not None and hasattr(tkq_mod, 'tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc')),
                "has_prepared_launch": bool(tkq_mod is not None and hasattr(tkq_mod, 'tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace')),
                "has_split2_dgrad_strided": bool(hasattr(tk, 'nvfp4_split2_dgrad_strided_onepass_gemm')),
                "has_split2_dgrad_strided_sg": bool(has_v4_split2_strided_sg),
                "has_split2_dgrad_strided_outer_sg": bool(has_v4_split2_strided_outer_sg),
                "use_split2_dgrad_strided_sg": bool(use_v4_split2_strided_sg_dgrad),
                "use_v4_prepared_split2_finalizer": bool(use_v4_prepared_split2_finalizer),
            })
        if use_two_single_row_split2_quant:
            single_row_q = localcta_state.get('single_row_q_bufs')
            if single_row_q is None:
                single_row_q = (
                    tkq_mod.tk_localcta_quantize_for_gemm_prepared_alloc(M, H, True, grad_output.device),
                    tkq_mod.tk_localcta_quantize_for_gemm_prepared_alloc(M, H, True, grad_output.device),
                )
                localcta_state['single_row_q_bufs'] = single_row_q
        _tk_stage_trace('ffn_bwd_localcta_sub', 'split_prod_start', debug_name)
        if paired_rht_carrier and not native_paired_rht_split2:
            with _ffn_cuda_timed(ffn_timings, "split2_quant_prep"):
                dh1, dh3_out = _get_ffn_localcta_deriv_outputs(
                    localcta_state, h1_raw, h3
                )
                with _ffn_cuda_timed(ffn_timings, "split2_bf16_producer"):
                    _produce_ffn_localcta_derivatives_with_te(
                        te_fused,
                        dh,
                        h3,
                        h1_raw,
                        dh1,
                        dh3_out,
                        localcta_state['amax1'],
                        localcta_state['amax2'],
                    )
                with _ffn_cuda_timed(ffn_timings, "split2_paired_rht_carrier"):
                    (
                        row_fp4s,
                        row_scs,
                        row_sgs,
                        col_fp4s,
                        col_scs,
                        col_sgs,
                        col_fp4_full,
                        col_sc_full,
                        col_sg_full,
                        paired_rht_split2_keepalive,
                    ) = _fast_quantize_localcta_v4_split2_paired_rht_carrier(
                        dh1,
                        dh3_out,
                        persistent_rng_state=ffn_deriv_sr_state,
                    )
                row_fp4_full = None
                strict_v4_raw_split2_consumers = False
                _trace_backend_choice(
                    'localcta_ffn_bwd', 'paired_rht_split2_fallback_carrier'
                )
        elif use_direct_split2:
            with _ffn_cuda_timed(ffn_timings, "split2_quant_prep"):
                split2_q = localcta_state.get('split2_q_bufs')
                if split2_q is None:
                    with _ffn_cuda_timed(ffn_timings, "split2_alloc"):
                        alloc = getattr(tkq_mod, "tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc", None)
                        if alloc is not None:
                            split2_q = alloc(M, H, H, grad_output.device)
                            localcta_state['split2_q_bufs'] = split2_q
                if w2_dgrad_silu_split2_q is not None:
                    split2_q = w2_dgrad_silu_split2_q
                elif split2_q is not None and use_v4_split2_two_stage:
                    with _ffn_cuda_timed(ffn_timings, "split2_twostage_launch"):
                        tkq_mod.tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace(
                            dh, h3, h1_raw,
                            split2_q[6], split2_q[7],
                            split2_q[9], split2_q[10],
                            split2_q[8], split2_q[11],
                        )
                else:
                    if split2_q is not None and use_fused_split2_quant:
                        with _ffn_cuda_timed(ffn_timings, "split2_fused_deriv_quant_launch"):
                            tkq_mod.tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
                                dh, h3, h1_raw,
                                split2_q[6], split2_q[7],
                                split2_q[9], split2_q[10],
                                split2_q[8], split2_q[11],
                            )
                    else:
                        if use_fused_row_producer_split2_quant:
                            with _ffn_cuda_timed(ffn_timings, "split2_row_producer_launch"):
                                dh1 = localcta_state['dh1']
                                dh3_out = localcta_state['dh3']
                                tkq_mod.tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace(
                                    dh, h3, h1_raw,
                                    dh1, dh3_out,
                                    split2_q[6], split2_q[7], split2_q[8],
                                )
                        elif use_split_cache:
                            with _ffn_cuda_timed(ffn_timings, "split2_bf16_producer"):
                                dh1 = localcta_state['dh1']
                                dh3_out = localcta_state['dh3']
                                if use_saved_sigmoid_experiment:
                                    te_fused.fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax(
                                        dh, h3, h1_raw, sig_h1,
                                        dh1, dh3_out,
                                    )
                                elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
                                    te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                                        dh, h3, h1_raw,
                                        dh1, dh3_out,
                                    )
                                elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out'):
                                    te_fused.fused_silu_deriv_dual_mul_bf16_out(
                                        dh, h3, h1_raw,
                                        dh1, dh3_out,
                                        localcta_state['amax1'], localcta_state['amax2'],
                                    )
                                else:
                                    dh1, dh3_out, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1_raw)
                        else:
                            with _ffn_cuda_timed(ffn_timings, "split2_bf16_producer"):
                                dh1, dh3_out, _, _ = te_fused.fused_silu_deriv_dual_mul_strided_bf16(dh, h13)
                    _debug_check_finite('ffn_bwd.localcta.dh1', dh1)
                    _debug_check_finite('ffn_bwd.localcta.dh3', dh3_out)
                    if _ffn_capture_path():
                        _append_ffn_capture({
                            "event": "ffn_split2_inputs",
                            "debug_name": debug_name,
                            "M": int(M),
                            "K": int(K),
                            "H": int(H),
                            "use_direct_split2": bool(use_direct_split2),
                            "use_two_single_row_split2_quant": bool(use_two_single_row_split2_quant),
                            "use_row_only_split2_quant": bool(use_row_only_split2_quant),
                            "use_fused_row_producer_split2_quant": bool(use_fused_row_producer_split2_quant),
                            "dh": _tensor_capture_stats(dh),
                            "dh1": _tensor_capture_stats(dh1),
                            "dh3_out": _tensor_capture_stats(dh3_out),
                        })

                    if use_v4_split2_two_stage:
                        pass
                    elif use_two_single_row_split2_quant and not use_fused_split2_quant:
                        with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                            tkq_mod.tk_localcta_quantize_for_gemm_prepared_launch(
                                dh1, True, True,
                                single_row_q[0][0], single_row_q[0][1],
                                single_row_q[0][2], single_row_q[0][3],
                                single_row_q[0][4], single_row_q[0][5],
                            )
                            tkq_mod.tk_localcta_quantize_for_gemm_prepared_launch(
                                dh3_out, True, True,
                                single_row_q[1][0], single_row_q[1][1],
                                single_row_q[1][2], single_row_q[1][3],
                                single_row_q[1][4], single_row_q[1][5],
                            )
                        row_fp4s = [single_row_q[0][0], single_row_q[1][0]]
                        row_scs = [single_row_q[0][1], single_row_q[1][1]]
                        row_sgs = [single_row_q[0][4], single_row_q[1][4]]
                        row_fp4_full = None
                        col_fp4s = col_scs = col_sgs = None
                    elif split2_q is not None and not use_fused_split2_quant and use_row_only_split2_quant:
                        with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                            tkq_mod.tk_localcta_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace(
                                dh1, dh3_out,
                                split2_q[6], split2_q[7], split2_q[8],
                            )
                    elif split2_q is not None and not use_fused_split2_quant and tkq_mod is not None and hasattr(tkq_mod, "tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace"):
                        with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                            tkq_mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
                                dh1, dh3_out,
                                split2_q[6], split2_q[7],
                                split2_q[9], split2_q[10],
                                split2_q[8], split2_q[11],
                            )
                    elif split2_q is not None and not use_fused_split2_quant and tkq_mod is not None and hasattr(tkq_mod, "tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch"):
                        with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                            split2_q = tkq_mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch(
                                dh1, dh3_out,
                                split2_q[6], split2_q[7],
                                split2_q[9], split2_q[10],
                                split2_q[8], split2_q[11],
                            )
                        localcta_state['split2_q_bufs'] = split2_q
                    elif not use_fused_split2_quant:
                        with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                            split2_q = tkq_mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared(dh1, dh3_out)
                    col_fp4_full = col_sc_full = col_sg_full = None
                    if not use_two_single_row_split2_quant:
                        row_fp4s, row_scs, row_sgs, \
                            col_fp4s, col_scs, col_sgs, \
                            row_fp4_full, _, _, _, _, _ = split2_q
                        if len(split2_q) >= 12:
                            col_fp4_full = split2_q[9]
                            col_sc_full = split2_q[10]
                            col_sg_full = split2_q[11]
                        if use_direct_split2:
                            # Prepared split2 returns col SG on the chunk grid
                            # [N_128_tiles, M_128_tiles]. Grouped wgrad consumes
                            # outer-scale tiles at the GEMM B-tile granularity
                            # (256 rows), so collapse both the M-tile axis and each
                            # adjacent pair of 128-row tiles.
                            def _collapse_wgrad_col_sg(t):
                                if torch.is_tensor(t) and t.dim() == 2 and t.size(1) > 1:
                                    if (t.size(0) % 2) == 0:
                                        return t.view(t.size(0) // 2, 2, t.size(1)).amax(dim=(1, 2)).contiguous()
                                    return t.amax(dim=1).contiguous()
                                return t

                            with _ffn_cuda_timed(ffn_timings, "split2_sg_collapse"):
                                col_sgs = [_collapse_wgrad_col_sg(t) for t in col_sgs]
                                col_sg_full = _collapse_wgrad_col_sg(col_sg_full)
                    if _ffn_capture_path():
                        _append_ffn_capture({
                            "event": "ffn_split2_quant",
                            "debug_name": debug_name,
                            "M": int(M),
                            "K": int(K),
                            "H": int(H),
                            "use_direct_split2": bool(use_direct_split2),
                            "use_two_single_row_split2_quant": bool(use_two_single_row_split2_quant),
                            "strict_v4_prepared_split2_finalized": bool(strict_v4_prepared_split2_finalized),
                            "row_fp4s": [_tensor_capture_stats(t) for t in row_fp4s] if row_fp4s is not None else None,
                            "row_scs": [_tensor_capture_stats(t) for t in row_scs] if row_scs is not None else None,
                            "row_sgs": [_tensor_capture_stats(t) for t in row_sgs] if row_sgs is not None else None,
                            "col_fp4s": [_tensor_capture_stats(t) for t in col_fp4s] if col_fp4s is not None else None,
                            "col_scs": [_tensor_capture_stats(t) for t in col_scs] if col_scs is not None else None,
                            "col_sgs": [_tensor_capture_stats(t) for t in col_sgs] if col_sgs is not None else None,
                            "row_fp4_full": _tensor_capture_stats(row_fp4_full),
                            "col_fp4_full": _tensor_capture_stats(col_fp4_full),
                            "col_sc_full": _tensor_capture_stats(col_sc_full),
                            "col_sg_full": _tensor_capture_stats(col_sg_full),
                        })
        else:
            with _ffn_cuda_timed(ffn_timings, "split2_quant_prep"):
                prefer_v4_raw_w13_dgrad = (
                    not strict_v4_localcta
                    and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                    and w13_dgrad_cols is not None
                    and os.environ.get('USE_TK_LOCALCTA_V4_SG_DIRECT_CONSUMERS', '0') == '1'
                    and use_tk_localcta_v4_raw_backward_fallbacks(M)
                )
                use_fused_tk_split_producer = (
                    (strict_v4_localcta or use_ffn_localcta_tk_quant_contract)
                    and use_split_cache
                    and hasattr(tkq, 'tk_silu_deriv_quantize_split_for_gemm')
                    and not prefer_v4_raw_w13_dgrad
                )
                use_strict_v4_inplace_split_producer = (
                    strict_v4_localcta
                    and tkq_mod is not None
                    and hasattr(tkq_mod, 'tk_localcta_silu_deriv_quantize_split_for_gemm_alloc')
                    and hasattr(tkq_mod, 'tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace')
                )
                use_v4_cat_split_producer = (
                    use_ffn_localcta_tk_quant_contract
                    and (
                        not strict_v4_localcta
                        or use_tk_localcta_v4_ffn_prepared_split2_producer()
                    )
                    and tkq_mod is not None
                    and hasattr(tkq_mod, 'tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc')
                    and hasattr(tkq_mod, 'tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace')
                )
                if native_paired_rht_split2 and not (
                    use_v4_cat_split_producer
                    or use_strict_v4_inplace_split_producer
                ):
                    raise RuntimeError(
                        "localCTA extension advertises native paired split2 RHT, "
                        "but the fused split2 producer route is unavailable"
                    )
                strict_v4_raw_split2_consumers = False
                if w2_dgrad_silu_split2_q is not None:
                    strict_split2_q = w2_dgrad_silu_split2_q
                    row_fp4s, row_scs, row_sgs = strict_split2_q[0], strict_split2_q[1], strict_split2_q[2]
                    col_fp4s, col_scs, col_sgs = strict_split2_q[3], strict_split2_q[4], strict_split2_q[5]
                    row_fp4_full = strict_split2_q[6]
                    col_fp4_full = strict_split2_q[9]
                    col_sc_full = strict_split2_q[10]
                    col_sg_full = strict_split2_q[11]
                    if use_v4_prepared_split2_finalizer:
                        outer_sgs = localcta_state.get('w2_dgrad_silu_split2_outer_sgs')
                        expected_row_shape = (M // 256, 1)
                        expected_col_full_shape = (1, 2 * (H // 256))
                        if (
                            outer_sgs is None
                            or tuple(outer_sgs[0].shape) != expected_row_shape
                            or tuple(outer_sgs[4].shape) != expected_col_full_shape
                        ):
                            row_sg_outer0 = torch.empty(expected_row_shape, dtype=torch.float32, device=grad_output.device)
                            row_sg_outer1 = torch.empty(expected_row_shape, dtype=torch.float32, device=grad_output.device)
                            col_sg_outer_full = torch.empty(expected_col_full_shape, dtype=torch.float32, device=grad_output.device)
                            col_sg_outer0 = col_sg_outer_full.narrow(1, 0, H // 256)
                            col_sg_outer1 = col_sg_outer_full.narrow(1, H // 256, H // 256)
                            outer_sgs = (
                                row_sg_outer0,
                                row_sg_outer1,
                                col_sg_outer0,
                                col_sg_outer1,
                                col_sg_outer_full,
                            )
                            localcta_state['w2_dgrad_silu_split2_outer_sgs'] = outer_sgs
                        row_sg_outer0, row_sg_outer1, col_sg_outer0, col_sg_outer1, col_sg_outer_full = outer_sgs
                        with _ffn_cuda_timed(ffn_timings, "split2_finalize_prepared"):
                            tkq_mod.tk_localcta_finalize_split2_for_gemm_prepared_inplace(
                                row_scs[0], row_sgs[0], row_sg_outer0,
                                col_scs[0], col_sgs[0], col_sg_outer0,
                                row_scs[1], row_sgs[1], row_sg_outer1,
                                col_scs[1], col_sgs[1], col_sg_outer1,
                            )
                        row_sgs = [row_sg_outer0, row_sg_outer1]
                        col_sgs = [col_sg_outer0, col_sg_outer1]
                        col_sg_full = col_sg_outer_full
                        strict_v4_prepared_split2_finalized = True
                    dh1 = None
                    dh3_out = None
                    strict_v4_raw_split2_consumers = False
                elif use_strict_v4_prepared_split2:
                    strict_split2_q = localcta_state.get('strict_prepared_split2_q_bufs')
                    if strict_split2_q is None:
                        with _ffn_cuda_timed(ffn_timings, "split2_alloc"):
                            strict_split2_q = tkq_mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
                                M, H, H, grad_output.device
                            )
                            localcta_state['strict_prepared_split2_q_bufs'] = strict_split2_q
                    row_bf16_prepared = (
                        use_tk_localcta_v4_ffn_row_bf16_prepared_deriv_quant()
                        and hasattr(
                            tkq_mod,
                            'tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace',
                        )
                        and hasattr(tkq_mod, 'tk_localcta_quantize_col_only_prepared_launch_inplace')
                    )
                    if row_bf16_prepared:
                        dh1 = localcta_state['dh1']
                        dh3_out = localcta_state['dh3']
                        with _ffn_cuda_timed(ffn_timings, "split2_row_bf16_quant_launch"):
                            tkq_mod.tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace(
                                dh, h3, h1_raw,
                                dh1, dh3_out,
                                strict_split2_q[6], strict_split2_q[7], strict_split2_q[8],
                            )
                        col_fp4_tmp, col_sc_tmp, col_sg_tmp = strict_split2_q[3], strict_split2_q[4], strict_split2_q[5]
                        row_sg_tmp = strict_split2_q[2]
                        with _ffn_cuda_timed(ffn_timings, "split2_col_quant_launch"):
                            tkq_mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                                dh1, row_sg_tmp[0].contiguous(),
                                col_fp4_tmp[0], col_sc_tmp[0], col_sg_tmp[0],
                            )
                            tkq_mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                                dh3_out, row_sg_tmp[1].contiguous(),
                                col_fp4_tmp[1], col_sc_tmp[1], col_sg_tmp[1],
                            )
                    else:
                        fused_prepared = None
                        if use_tk_localcta_v4_split2_two_stage():
                            fused_prepared = getattr(
                                tkq_mod,
                                'tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace',
                                None,
                            )
                        if (
                            fused_prepared is None
                            and not disable_tk_localcta_v4_ffn_fused_prepared_deriv_quant()
                        ):
                            fused_prepared = getattr(
                                tkq_mod,
                                'tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace',
                                None,
                            )
                        if fused_prepared is not None:
                            with _ffn_cuda_timed(ffn_timings, "split2_fused_deriv_quant_launch"):
                                fused_prepared(
                                    dh, h3, h1_raw,
                                    strict_split2_q[6], strict_split2_q[7],
                                    strict_split2_q[9], strict_split2_q[10],
                                    strict_split2_q[8], strict_split2_q[11],
                                )
                            dh1 = None
                            dh3_out = None
                        else:
                            dh1 = localcta_state['dh1']
                            dh3_out = localcta_state['dh3']
                            with _ffn_cuda_timed(ffn_timings, "split2_bf16_producer"):
                                tkq_mod.tk_localcta_silu_deriv_split_bf16_launch_inplace(
                                    dh, h3, h1_raw, dh1, dh3_out,
                                )
                            with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                                tkq_mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
                                    dh1, dh3_out,
                                    strict_split2_q[6], strict_split2_q[7],
                                    strict_split2_q[9], strict_split2_q[10],
                                    strict_split2_q[8], strict_split2_q[11],
                                )
                    row_fp4s, row_scs, row_sgs = strict_split2_q[0], strict_split2_q[1], strict_split2_q[2]
                    col_fp4s, col_scs, col_sgs = strict_split2_q[3], strict_split2_q[4], strict_split2_q[5]
                    row_fp4_full = strict_split2_q[6]
                    col_fp4_full = strict_split2_q[9]
                    col_sc_full = strict_split2_q[10]
                    col_sg_full = strict_split2_q[11]
                    if use_v4_prepared_split2_finalizer:
                        outer_sgs = localcta_state.get('strict_prepared_split2_outer_sgs')
                        expected_row_shape = (M // 256, 1)
                        expected_col_full_shape = (1, 2 * (H // 256))
                        if (
                            outer_sgs is None
                            or tuple(outer_sgs[0].shape) != expected_row_shape
                            or tuple(outer_sgs[4].shape) != expected_col_full_shape
                        ):
                            row_sg_outer0 = torch.empty(expected_row_shape, dtype=torch.float32, device=grad_output.device)
                            row_sg_outer1 = torch.empty(expected_row_shape, dtype=torch.float32, device=grad_output.device)
                            col_sg_outer_full = torch.empty(expected_col_full_shape, dtype=torch.float32, device=grad_output.device)
                            col_sg_outer0 = col_sg_outer_full.narrow(1, 0, H // 256)
                            col_sg_outer1 = col_sg_outer_full.narrow(1, H // 256, H // 256)
                            outer_sgs = (
                                row_sg_outer0,
                                row_sg_outer1,
                                col_sg_outer0,
                                col_sg_outer1,
                                col_sg_outer_full,
                            )
                            localcta_state['strict_prepared_split2_outer_sgs'] = outer_sgs
                        row_sg_outer0, row_sg_outer1, col_sg_outer0, col_sg_outer1, col_sg_outer_full = outer_sgs
                        with _ffn_cuda_timed(ffn_timings, "split2_finalize_prepared"):
                            tkq_mod.tk_localcta_finalize_split2_for_gemm_prepared_inplace(
                                row_scs[0], row_sgs[0], row_sg_outer0,
                                col_scs[0], col_sgs[0], col_sg_outer0,
                                row_scs[1], row_sgs[1], row_sg_outer1,
                                col_scs[1], col_sgs[1], col_sg_outer1,
                            )
                        row_sgs = [row_sg_outer0, row_sg_outer1]
                        col_sgs = [col_sg_outer0, col_sg_outer1]
                        col_sg_full = col_sg_outer_full
                        strict_v4_prepared_split2_finalized = True
                    strict_v4_raw_split2_consumers = False
                elif use_fused_tk_split_producer:
                    if use_v4_cat_split_producer:
                        if overlapped_v4_cat_split2 is not None:
                            fused, strict_split2_q = overlapped_v4_cat_split2
                            torch.cuda.current_stream().wait_event(
                                localcta_state['deriv_quant_ready_event']
                            )
                        else:
                            strict_split2_q = localcta_state.get('v4_cat_split2_q_bufs')
                            if strict_split2_q is None:
                                with _ffn_cuda_timed(ffn_timings, "split2_alloc"):
                                    strict_split2_q = tkq_mod.tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc(
                                        M, H, grad_output.device
                                    )
                                    localcta_state['v4_cat_split2_q_bufs'] = strict_split2_q
                            dh1, dh3_out = _get_ffn_localcta_deriv_outputs(
                                localcta_state, h1_raw, h3
                            )
                            if native_paired_rht_split2:
                                with _ffn_cuda_timed(
                                    ffn_timings, "split2_bf16_producer"
                                ):
                                    _produce_ffn_localcta_derivatives_with_te(
                                        te_fused,
                                        dh,
                                        h3,
                                        h1_raw,
                                        dh1,
                                        dh3_out,
                                        localcta_state['amax1'],
                                        localcta_state['amax2'],
                                    )
                            with _ffn_cuda_timed(ffn_timings, "split2_fused_tk_split_producer"):
                                fused = _call_localcta_silu_deriv_split2(
                                    tkq_mod.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace,
                                    dh, h3, h1_raw,
                                    dh1, dh3_out,
                                    *strict_split2_q[:16],
                                    not strict_v4_raw_split2_consumers,
                                    grad_data_sr,
                                    grad_scale_sr,
                                    _nvfp4_rng_seed(),
                                    _nvfp4_rng_subsequence_base(),
                                    grad_data_sr_axes,
                                    persistent_rng_state=ffn_deriv_sr_state,
                                    native_paired_rht=native_paired_rht_split2,
                                )
                    elif use_strict_v4_inplace_split_producer:
                        strict_split2_q = localcta_state.get('strict_split2_q_bufs')
                        if strict_split2_q is None:
                            with _ffn_cuda_timed(ffn_timings, "split2_alloc"):
                                strict_split2_q = tkq_mod.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
                                    M, H, grad_output.device
                                )
                                localcta_state['strict_split2_q_bufs'] = strict_split2_q
                        dh1, dh3_out = _get_ffn_localcta_deriv_outputs(
                            localcta_state, h1_raw, h3
                        )
                        if native_paired_rht_split2:
                            with _ffn_cuda_timed(
                                ffn_timings, "split2_bf16_producer"
                            ):
                                _produce_ffn_localcta_derivatives_with_te(
                                    te_fused,
                                    dh,
                                    h3,
                                    h1_raw,
                                    dh1,
                                    dh3_out,
                                    localcta_state['amax1'],
                                    localcta_state['amax2'],
                                )
                        with _ffn_cuda_timed(ffn_timings, "split2_fused_tk_split_producer"):
                            fused = _call_localcta_silu_deriv_split2(
                                tkq_mod.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace,
                                dh, h3, h1_raw,
                                dh1, dh3_out,
                                *strict_split2_q,
                                not strict_v4_raw_split2_consumers,
                                grad_data_sr,
                                grad_scale_sr,
                                _nvfp4_rng_seed(),
                                _nvfp4_rng_subsequence_base(),
                                grad_data_sr_axes,
                                persistent_rng_state=ffn_deriv_sr_state,
                                native_paired_rht=native_paired_rht_split2,
                            )
                    else:
                        if ffn_deriv_sr_state is not None:
                            raise RuntimeError(
                                "checkpointed localCTA SR requires the explicit-state "
                                "v4 fused split2 derivative producer"
                            )
                        strict_v4_raw_split2_consumers = False
                        with _ffn_cuda_timed(ffn_timings, "split2_fused_tk_split_producer"):
                            fused = tkq.tk_silu_deriv_quantize_split_for_gemm(dh, h3, h1_raw)
                    if native_paired_rht_split2:
                        _trace_backend_choice(
                            'localcta_ffn_bwd', 'native_paired_rht_split2_carrier'
                        )
                    dh1 = None
                    dh3_out = None
                    row_fp4s = [fused[0], fused[6]]
                    row_scs = [fused[1], fused[7]]
                    col_fp4s = [fused[2], fused[8]]
                    col_scs = [fused[3], fused[9]]
                    row_sgs = [fused[4], fused[10]]
                    col_sgs = [fused[5], fused[11]]
                    row_fp4_full = None
                    if use_v4_cat_split_producer:
                        col_fp4_full = strict_split2_q[16]
                        col_sc_full = strict_split2_q[17]
                        col_sg_full = strict_split2_q[18]
                    else:
                        col_fp4_full = col_sc_full = col_sg_full = None
                else:
                    if ffn_deriv_sr_state is not None:
                        raise RuntimeError(
                            "checkpointed localCTA SR cannot fall back to a "
                            "derivative quantizer without the explicit-state v4 ABI"
                        )
                    strict_v4_raw_split2_consumers = False
                    if use_split_cache:
                        with _ffn_cuda_timed(ffn_timings, "split2_bf16_producer"):
                            dh1 = localcta_state['dh1']
                            dh3_out = localcta_state['dh3']
                            if use_saved_sigmoid_experiment:
                                te_fused.fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax(
                                    dh, h3, h1_raw, sig_h1,
                                    dh1, dh3_out,
                                )
                            elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
                                te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                                    dh, h3, h1_raw,
                                    dh1, dh3_out,
                                )
                            elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out'):
                                te_fused.fused_silu_deriv_dual_mul_bf16_out(
                                    dh, h3, h1_raw,
                                    dh1, dh3_out,
                                    localcta_state['amax1'], localcta_state['amax2'],
                                )
                            else:
                                dh1, dh3_out, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1_raw)
                    else:
                        with _ffn_cuda_timed(ffn_timings, "split2_bf16_producer"):
                            dh1, dh3_out, _, _ = te_fused.fused_silu_deriv_dual_mul_strided_bf16(dh, h13)
                    _debug_check_finite('ffn_bwd.localcta.dh1', dh1)
                    _debug_check_finite('ffn_bwd.localcta.dh3', dh3_out)
                    with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                        split2_q = tkq.tk_batched_quantize_for_gemm([dh1, dh3_out], True, True)
                    if len(split2_q) >= 9:
                        row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs, \
                            col_fp4_full, col_sc_full, col_sg_full = split2_q[:9]
                    else:
                        row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs = split2_q[:6]
                        col_fp4_full = col_sc_full = col_sg_full = None
                    row_fp4_full = None

        use_tk_split2_acts = (
            os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
            and use_tk_localcta_v4_tk_ffn_dgrad_acts()
        )
        if (
            use_tk_split2_acts
            and not use_ffn_localcta_tk_quant_contract
            and dh1 is not None
            and dh3_out is not None
        ):
            tkq_std = _get_tk_quant()
            split2_q = tkq_std.tk_batched_quantize_for_gemm([dh1, dh3_out], True, True)
            if len(split2_q) >= 9:
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs, \
                    col_fp4_full, col_sc_full, col_sg_full = split2_q[:9]
            else:
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs = split2_q[:6]
                col_fp4_full = col_sc_full = col_sg_full = None
            row_fp4_full = None

        if use_two_single_row_split2_quant and not use_fused_split2_quant:
            with _ffn_cuda_timed(ffn_timings, "split2_quant_launch"):
                tkq_mod.tk_localcta_quantize_for_gemm_prepared_launch(
                    dh1, True, True,
                    single_row_q[0][0], single_row_q[0][1],
                    single_row_q[0][2], single_row_q[0][3],
                    single_row_q[0][4], single_row_q[0][5],
                )
                tkq_mod.tk_localcta_quantize_for_gemm_prepared_launch(
                    dh3_out, True, True,
                    single_row_q[1][0], single_row_q[1][1],
                    single_row_q[1][2], single_row_q[1][3],
                    single_row_q[1][4], single_row_q[1][5],
                )
            row_fp4s = [single_row_q[0][0], single_row_q[1][0]]
            row_scs = [single_row_q[0][1], single_row_q[1][1]]
            row_sgs = [single_row_q[0][4], single_row_q[1][4]]
            row_fp4_full = None
            col_fp4s = col_scs = col_sgs = None
            col_fp4_full = col_sc_full = col_sg_full = None
        _tk_stage_trace('ffn_bwd_localcta_sub', 'split_prod_done', debug_name)

        if _ffn_capture_path():
            _append_ffn_capture({
                "event": "ffn_split2_inputs",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "use_direct_split2": bool(use_direct_split2),
                "use_two_single_row_split2_quant": bool(use_two_single_row_split2_quant),
                "use_row_only_split2_quant": bool(use_row_only_split2_quant),
                "use_fused_row_producer_split2_quant": bool(use_fused_row_producer_split2_quant),
                "dh": _tensor_capture_stats(dh),
                "dh1": _tensor_capture_stats(dh1),
                "dh3_out": _tensor_capture_stats(dh3_out),
            })
            _append_ffn_capture({
                "event": "ffn_split2_quant",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "use_direct_split2": bool(use_direct_split2),
                "use_two_single_row_split2_quant": bool(use_two_single_row_split2_quant),
                "strict_v4_prepared_split2_finalized": bool(strict_v4_prepared_split2_finalized),
                "row_fp4s": [_tensor_capture_stats(t) for t in row_fp4s] if row_fp4s is not None else None,
                "row_scs": [_tensor_capture_stats(t) for t in row_scs] if row_scs is not None else None,
                "row_sgs": [_tensor_capture_stats(t) for t in row_sgs] if row_sgs is not None else None,
                "col_fp4s": [_tensor_capture_stats(t) for t in col_fp4s] if col_fp4s is not None else None,
                "col_scs": [_tensor_capture_stats(t) for t in col_scs] if col_scs is not None else None,
                "col_sgs": [_tensor_capture_stats(t) for t in col_sgs] if col_sgs is not None else None,
                "row_fp4_full": _tensor_capture_stats(row_fp4_full),
                "col_fp4_full": _tensor_capture_stats(col_fp4_full),
                "col_sc_full": _tensor_capture_stats(col_sc_full),
                "col_sg_full": _tensor_capture_stats(col_sg_full),
            })

        _dump_ffn_tensors(
            "split2_pre",
            debug_name,
            {
                "debug_name": debug_name,
                "use_direct_split2": bool(use_direct_split2),
                "use_two_single_row_split2_quant": bool(use_two_single_row_split2_quant),
                "dh": dh,
                "dh1": dh1,
                "dh3_out": dh3_out,
                "row_fp4s": row_fp4s,
                "row_scs": row_scs,
                "row_sgs": row_sgs,
                "row_fp4_full": row_fp4_full,
                "col_fp4s": col_fp4s,
                "col_scs": col_scs,
                "col_sgs": col_sgs,
                "col_fp4_full": col_fp4_full,
                "col_sc_full": col_sc_full,
                "col_sg_full": col_sg_full,
                "dgrad_wc_fp4_cols": dgrad_wc_fp4_cols,
                "dgrad_wc_sc_cols": dgrad_wc_sc_cols,
                "dgrad_wc_sg_cols": dgrad_wc_sg_cols,
                "wc_fp4_cols": wc_fp4_cols,
                "wc_sc_cols": wc_sc_cols,
                "wc_sg_cols": wc_sg_cols,
                "input": input,
                "x_tk_col": x_nvfp4._tk_col if hasattr(x_nvfp4, "_tk_col") else None,
                "inv_rms": inv_rms,
                "norm_weight": norm_weight,
            },
        )

        if wc_sg_cols is None:
            raise RuntimeError("localCTA FFN backward requires per-split weight col SG tensors")

        def _collapse_wgrad_col_sg(t):
            if torch.is_tensor(t) and t.dim() == 2 and t.size(1) > 1:
                if (t.size(0) % 2) == 0:
                    return t.view(t.size(0) // 2, 2, t.size(1)).amax(dim=(1, 2)).contiguous()
                return t.amax(dim=1).contiguous()
            return t

        def _reduce_row_sg_to_col_tiles(t):
            if torch.is_tensor(t) and t.dim() == 2 and t.size(1) > 1:
                col_max = t.amax(dim=0)
                if (col_max.numel() % 2) == 0:
                    return col_max.view(col_max.numel() // 2, 2).amax(dim=1).contiguous()
                return col_max.contiguous()
            return t

        def _grouped_wgrad_sg_group(sgs):
            if torch.is_tensor(sgs):
                flat = sgs.to(torch.float32).reshape(-1)
                if flat.numel() == len(N_dims_13):
                    return flat.contiguous()
                return flat[:len(N_dims_13)].contiguous()
            return torch.stack(
                [sg.reshape(-1).to(torch.float32)[0] for sg in sgs],
                dim=0,
            ).contiguous()

        if use_saved_sigmoid_overlap:
            torch.cuda.current_stream().record_event(localcta_state['split_quant_ready_event'])

        if wgrad_stream is None and not disable_wgrad_stream:
            wgrad_stream = _get_wgrad_stream()
        if wgrad_stream is None:
            wgrad_stream = torch.cuda.current_stream()
        elif use_saved_sigmoid_overlap:
            wgrad_stream.wait_event(localcta_state['split_quant_ready_event'])
        else:
            wgrad_stream.wait_stream(torch.cuda.current_stream())
        _tk_stage_trace('ffn_bwd_localcta_sub', 'w13_wgrad_start', debug_name)
        with torch.cuda.stream(wgrad_stream), _ffn_cuda_timed(ffn_timings, "w13_wgrad", wgrad_stream):
            _record_tensors_on_stream(
                (
                    grad_output, dY_fp4c, dY_scc, dY_sgc,
                    h_fp4c, h_scc, h_sgc,
                    h1_raw, h3, grad_w2,
                    dh1, dh3_out,
                    row_fp4s, row_scs, row_sgs,
                    col_fp4s, col_scs, col_sgs,
                    col_fp4_full, col_sc_full, col_sg_full,
                    paired_rht_split2_keepalive,
                    x_nvfp4._tk_col if hasattr(x_nvfp4, '_tk_col') else None,
                ),
                wgrad_stream,
            )
            if (
                use_tk_localcta()
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and not use_tk_localcta_v4_cpp_only()
                and not use_ffn_localcta_tk_quant_contract
                and dh1 is not None
                and dh3_out is not None
            ):
                x_normed = localcta_state['x_normed']
                x_normed.copy_(
                    (input.float() * inv_rms.float().view(-1, 1) * norm_weight.float().view(1, -1)).to(torch.bfloat16)
                )
                grad_w1 = localcta_state['grad_w1']
                grad_w3 = localcta_state['grad_w3']
                tk_v4_direct_raw_wgrad_gemm(dh1, x_normed, grad_w1)
                tk_v4_direct_raw_wgrad_gemm(dh3_out, x_normed, grad_w3)
            elif use_two_single_row_split2_quant:
                dh_col_fp4_full = localcta_state['dh_col_fp4_full']
                dh_col_sc_full = localcta_state['dh_col_sc_full']
                dh_col_sg_full = localcta_state['dh_col_sg_full']
                dh1_col_fp4 = dh_col_fp4_full.narrow(0, 0, H)
                dh3_col_fp4 = dh_col_fp4_full.narrow(0, H, H)
                dh1_col_sc = dh_col_sc_full.narrow(0, 0, H // 128)
                dh3_col_sc = dh_col_sc_full.narrow(0, H // 128, H // 128)
                dh1_col_sg = dh_col_sg_full.narrow(0, 0, H // 128)
                dh3_col_sg = dh_col_sg_full.narrow(0, H // 128, H // 128)
                dh1_col_fp4.copy_(single_row_q[0][2])
                dh3_col_fp4.copy_(single_row_q[1][2])
                dh1_col_sc.copy_(single_row_q[0][3])
                dh3_col_sc.copy_(single_row_q[1][3])
                dh1_col_sg.copy_(single_row_q[0][5])
                dh3_col_sg.copy_(single_row_q[1][5])
                grad_w13 = tk_grouped_wgrad_gemm((
                    [dh1_col_fp4, dh3_col_fp4],
                    [dh1_col_sc, dh3_col_sc],
                    [dh1_col_sg, dh3_col_sg],
                    dh_col_fp4_full,
                    dh_col_sc_full,
                    dh_col_sg_full,
                ), x_nvfp4, N_dims_13,
                    owner_key=localcta_direct_w13_owner,
                    caller_stream=wgrad_stream)
                grad_w1, grad_w3 = grad_w13[:H, :], grad_w13[H:, :]
            elif (
                (use_row_only_split2_quant or use_fused_row_producer_split2_quant)
                and use_direct_split2
                and w2_dgrad_silu_split2_q is None
            ):
                dh_col_fp4_full = localcta_state['dh_col_fp4_full']
                dh_col_sc_full = localcta_state['dh_col_sc_full']
                dh_col_sg_full = localcta_state['dh_col_sg_full']
                dh1_col_fp4 = dh_col_fp4_full.narrow(0, 0, H)
                dh3_col_fp4 = dh_col_fp4_full.narrow(0, H, H)
                dh1_col_sc = dh_col_sc_full.narrow(0, 0, H // 128)
                dh3_col_sc = dh_col_sc_full.narrow(0, H // 128, H // 128)
                dh1_col_sg = dh_col_sg_full.narrow(0, 0, H // 128)
                dh3_col_sg = dh_col_sg_full.narrow(0, H // 128, H // 128)
                localcta_state['dh1_row_sg'].copy_(row_sgs[0])
                localcta_state['dh3_row_sg'].copy_(row_sgs[1])
                if use_v4_split2_two_stage:
                    dh1_col_sg_tiles = _reduce_row_sg_to_col_tiles(localcta_state['dh1_row_sg'])
                    dh3_col_sg_tiles = _reduce_row_sg_to_col_tiles(localcta_state['dh3_row_sg'])
                    dh1_col_sg_ratio = localcta_state['dh1_row_sg'] / dh1_col_sg_tiles.repeat_interleave(2).view(1, -1).clamp_min(1e-12)
                    dh3_col_sg_ratio = localcta_state['dh3_row_sg'] / dh3_col_sg_tiles.repeat_interleave(2).view(1, -1).clamp_min(1e-12)
                    tkq_mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                        dh1, dh1_col_sg_ratio.contiguous(),
                        dh1_col_fp4,
                        dh1_col_sc,
                        dh1_col_sg,
                    )
                    tkq_mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                        dh3_out, dh3_col_sg_ratio.contiguous(),
                        dh3_col_fp4,
                        dh3_col_sc,
                        dh3_col_sg,
                    )
                    dh1_col_sg_reduced = dh1_col_sg_tiles
                    dh3_col_sg_reduced = dh3_col_sg_tiles
                    dh_col_sg_full_reduced = torch.cat([dh1_col_sg_tiles, dh3_col_sg_tiles], dim=0)
                else:
                    tkq._mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                        dh1, localcta_state['dh1_row_sg'],
                        dh1_col_fp4,
                        dh1_col_sc,
                        dh1_col_sg,
                    )
                    tkq._mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                        dh3_out, localcta_state['dh3_row_sg'],
                        dh3_col_fp4,
                        dh3_col_sc,
                        dh3_col_sg,
                    )
                    dh1_col_sg_reduced = _collapse_wgrad_col_sg(dh1_col_sg)
                    dh3_col_sg_reduced = _collapse_wgrad_col_sg(dh3_col_sg)
                    dh_col_sg_full_reduced = _collapse_wgrad_col_sg(dh_col_sg_full)
                grad_w13 = tk_grouped_wgrad_gemm((
                    [dh1_col_fp4, dh3_col_fp4],
                    [dh1_col_sc, dh3_col_sc],
                    [dh1_col_sg_reduced, dh3_col_sg_reduced],
                    dh_col_fp4_full,
                    dh_col_sc_full,
                    dh_col_sg_full_reduced,
                ), x_nvfp4, N_dims_13,
                    owner_key=localcta_direct_w13_owner,
                    caller_stream=wgrad_stream)
                grad_w1, grad_w3 = grad_w13[:H, :], grad_w13[H:, :]
            else:
                if (
                    col_fp4_full is not None
                    and col_sc_full is not None
                    and col_sg_full is not None
                    and (
                        use_ffn_localcta_tk_quant_contract
                        or use_direct_split2
                        or use_strict_v4_prepared_split2
                    )
                ):
                    sg_full_for_wgrad = (
                        col_sg_full
                        if torch.is_tensor(col_sg_full)
                        else _grouped_wgrad_sg_group(col_sgs)
                    )
                    dy_col_quant = (
                        col_fp4s,
                        col_scs,
                        col_sgs,
                        col_fp4_full,
                        col_sc_full,
                        sg_full_for_wgrad,
                    )
                    if (
                        use_ffn_localcta_tk_quant_contract
                        and os.environ.get('USE_TK_FFN_FORCE_SPLIT_WGRAD', '0') == '1'
                    ):
                        grad_w1, grad_w3 = tk_split_wgrad_gemm(
                            (col_fp4s, col_scs, col_sgs),
                            x_nvfp4,
                            use_localcta=use_localcta,
                        )
                    else:
                        grad_w1, grad_w3 = tk_grouped_wgrad_gemm(
                            dy_col_quant,
                            x_nvfp4,
                            N_dims_13,
                            owner_key=localcta_direct_w13_owner,
                            caller_stream=wgrad_stream,
                        ).split(H, dim=0)
                elif use_ffn_localcta_tk_quant_contract:
                    grad_w1, grad_w3 = tk_grouped_wgrad_gemm(
                        (
                            col_fp4s,
                            col_scs,
                            col_sgs,
                        ),
                        x_nvfp4,
                        N_dims_13,
                        owner_key=localcta_direct_w13_owner,
                        caller_stream=wgrad_stream,
                    ).split(H, dim=0)
                else:
                    grad_w1, grad_w3 = tk_split_wgrad_gemm((col_fp4s, col_scs, col_sgs), x_nvfp4)
            _debug_check_finite('ffn_bwd.localcta.grad_w1', grad_w1)
            _debug_check_finite('ffn_bwd.localcta.grad_w3', grad_w3)
            if _ffn_capture_path():
                _append_ffn_capture({
                    "event": "ffn_w13_wgrad",
                    "debug_name": debug_name,
                    "M": int(M),
                    "K": int(K),
                    "H": int(H),
                    "use_direct_split2": bool(use_direct_split2),
                    "use_two_single_row_split2_quant": bool(use_two_single_row_split2_quant),
                    "grad_w1": _tensor_capture_stats(grad_w1),
                    "grad_w3": _tensor_capture_stats(grad_w3),
                })
            if use_saved_sigmoid_overlap:
                wgrad_stream.record_event(localcta_state['wgrad_done_event'])
        _tk_stage_trace('ffn_bwd_localcta_sub', 'w13_wgrad_done', debug_name)

        d_normed = localcta_state['d_normed']
        use_v4_raw_split2 = (
            not strict_v4_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
            and dh1 is not None
            and w13_dgrad_cols is not None
            and os.environ.get('USE_TK_LOCALCTA_V4_SG_DIRECT_CONSUMERS', '0') == '1'
            and use_tk_localcta_v4_raw_backward_fallbacks(dh1.size(0))
        )
        dgrad_wc_sg_for_split2 = dgrad_wc_sg_cols
        if strict_v4_raw_split2_consumers:
            dgrad_wc_sg_for_split2 = [
                _prepare_localcta_v4_chunkgrid_for_batched(
                    sg,
                    fp4.size(0),
                    fp4.size(1) * 2,
                    fp4.device,
                )
                for sg, fp4 in zip(dgrad_wc_sg_cols, dgrad_wc_fp4_cols)
            ]
        use_w2_dgrad_silu_strided_split2 = (
            w2_dgrad_silu_split2_q is not None
            and row_fp4_full is not None
            and has_v4_split2_strided_sg
        )
        split2_dgrad_overwrites_d_normed = (
            os.environ.get('USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO', '0') == '1'
            and os.environ.get('USE_TK_FFN_DISABLE_WGRAD_STREAM', '0') == '1'
            and not use_v4_raw_split2
            and (
                (
                    (use_strict_v4_prepared_split2 or use_w2_dgrad_silu_strided_split2)
                    and row_fp4_full is not None
                    and (
                        (
                            strict_v4_prepared_split2_finalized
                            and hasattr(tk, 'nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg')
                        )
                        or hasattr(tk, 'nvfp4_split2_dgrad_strided_onepass_gemm_sg')
                    )
                )
                or (
                    (strict_v4_localcta or use_ffn_localcta_tk_quant_contract)
                    and row_fp4s is not None
                    and len(row_fp4s) == 2
                    and hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm')
                )
                or (
                    (use_two_single_row_split2_quant or use_tk_split2_acts)
                    and hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm')
                )
                or (
                    use_direct_split2
                    and row_fp4_full is not None
                    and hasattr(tk, 'nvfp4_split2_dgrad_strided_onepass_gemm')
                )
                or (
                    not (use_strict_v4_prepared_split2 or use_w2_dgrad_silu_strided_split2)
                    and not (strict_v4_localcta or use_ffn_localcta_tk_quant_contract)
                    and not (use_two_single_row_split2_quant or use_tk_split2_acts)
                    and not use_direct_split2
                    and (
                        hasattr(tk, 'nvfp4_v3_split2_dgrad_onepass_gemm')
                        or hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm')
                    )
                )
            )
        )
        # Experimental only: some split2 dgrad kernels appear to overwrite the
        # full D_out tile, but skipping this clear caused intermittent launch
        # failures in full 8B trainer runs. Keep the safe zero as the default.
        if split2_dgrad_overwrites_d_normed:
            _tk_stage_trace('ffn_bwd_localcta_sub', 'd_normed_zero_skip_start', debug_name)
            _tk_stage_trace('ffn_bwd_localcta_sub', 'd_normed_zero_skip_done', debug_name)
        else:
            _tk_stage_trace('ffn_bwd_localcta_sub', 'd_normed_zero_start', debug_name)
            d_normed.zero_()
            _tk_stage_trace('ffn_bwd_localcta_sub', 'd_normed_zero_done', debug_name)
        _tk_stage_trace('ffn_bwd_localcta_sub', 'split_dgrad_start', debug_name)
        with _ffn_cuda_timed(ffn_timings, "split2_dgrad"):
            if use_tk_localcta_ffn_dequant_dgrad_debug():
                _trace_backend_choice('localcta_ffn_dgrad', 'dequant_debug')
                dequant_mode = get_tk_localcta_ffn_dequant_dgrad_debug_mode()
                exact_activations = None
                if dequant_mode == 'weight':
                    if use_tk_ffn_localcta_inplace_h13_deriv():
                        exact_activations = (h1_raw, h3)
                    elif dh1 is not None and dh3_out is not None:
                        exact_activations = (dh1, dh3_out)
                    else:
                        exact_activations = (
                            localcta_state['dh1'], localcta_state['dh3']
                        )
                _localcta_ffn_dequant_split2_dgrad(
                    tkq_mod,
                    row_fp4s,
                    row_scs,
                    row_sgs,
                    dgrad_wc_fp4_cols,
                    dgrad_wc_sc_cols,
                    dgrad_wc_sg_cols,
                    d_normed,
                    M,
                    H,
                    K,
                    mode=dequant_mode,
                    exact_activations=exact_activations,
                    exact_weights=(w1_bf16, w3_bf16),
                    input_tensor=input,
                    inv_rms=inv_rms,
                    debug_name=debug_name,
                )
            elif use_bf16_dgrad_debug:
                _trace_backend_choice('localcta_ffn_dgrad', 'bf16_debug')
                if w1_bf16 is None or w3_bf16 is None:
                    raise RuntimeError(
                        "BF16 FFN dgrad debug path requires ctx.w1_bf16 and ctx.w3_bf16"
                    )
                if dh1 is None or dh3_out is None:
                    dh1 = localcta_state['dh1']
                    dh3_out = localcta_state['dh3']
                    if hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
                        te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                            dh, h3, h1_raw, dh1, dh3_out,
                        )
                    else:
                        dh1_tmp, dh3_tmp, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(
                            dh, h3, h1_raw
                        )
                        dh1.copy_(dh1_tmp)
                        dh3_out.copy_(dh3_tmp)
                d_normed.copy_(
                    torch.matmul(dh1.float(), w1_bf16.float()).to(torch.bfloat16)
                )
                d_normed.add_(
                    torch.matmul(dh3_out.float(), w3_bf16.float()).to(torch.bfloat16)
                )
            elif (
                (use_strict_v4_prepared_split2 or use_w2_dgrad_silu_strided_split2)
                and row_fp4_full is not None
                and strict_v4_prepared_split2_finalized
                and hasattr(tk, 'nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg')
            ):
                tk.nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg(
                    row_fp4_full,
                    row_scs,
                    row_sgs,
                    [0, H // 2],
                    [H // 2, H // 2],
                    dgrad_wc_fp4_cols,
                    dgrad_wc_sc_cols,
                    dgrad_wc_sg_cols,
                    d_normed,
                    -1,
                )
            elif (
                (use_strict_v4_prepared_split2 or use_w2_dgrad_silu_strided_split2)
                and row_fp4_full is not None
            ):
                row_sgs_for_dgrad = [
                    _prepare_localcta_v4_chunkgrid_for_batched(
                        sg,
                        row_fp4_full.size(0),
                        H,
                        row_fp4_full.device,
                    )
                    for sg in row_sgs
                ]
                dgrad_wc_sg_for_dgrad = [
                    _prepare_localcta_v4_chunkgrid_for_batched(
                        sg,
                        fp4.size(0),
                        fp4.size(1) * 2,
                        fp4.device,
                    )
                    for sg, fp4 in zip(dgrad_wc_sg_cols, dgrad_wc_fp4_cols)
                ]
                tk.nvfp4_split2_dgrad_strided_onepass_gemm_sg(
                    row_fp4_full,
                    row_scs,
                    row_sgs_for_dgrad,
                    [0, H // 2],
                    [H // 2, H // 2],
                    dgrad_wc_fp4_cols,
                    dgrad_wc_sc_cols,
                    dgrad_wc_sg_for_dgrad,
                    d_normed,
                    -1,
                )
            elif use_v4_raw_split2:
                d_normed_tmp = localcta_state['d_normed_tmp']
                d_normed_tmp.zero_()
                tk_v4_direct_raw_dgrad_gemm(dh1, w13_dgrad_cols[0], d_normed)
                tk_v4_direct_raw_dgrad_gemm(dh3_out, w13_dgrad_cols[1], d_normed_tmp)
                d_normed.add_(d_normed_tmp)
            elif strict_v4_localcta or use_ffn_localcta_tk_quant_contract:
                if (
                    len(row_fp4s) == 2
                    and hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm')
                ):
                    tk.nvfp4_split2_dgrad_onepass_gemm(
                        row_fp4s, row_scs, row_sgs,
                        dgrad_wc_fp4_cols, dgrad_wc_sc_cols, dgrad_wc_sg_for_split2,
                        d_normed,
                        tk_localcta_v3_split2_onepass_config_idx(),
                    )
                else:
                    tk.nvfp4_batched_accum_gemm(
                        row_fp4s, row_scs, row_sgs,
                        dgrad_wc_fp4_cols, dgrad_wc_sc_cols, dgrad_wc_sg_for_split2,
                        d_normed,
                    )
            elif use_two_single_row_split2_quant or use_tk_split2_acts:
                tk.nvfp4_split2_dgrad_onepass_gemm(
                    row_fp4s, row_scs, row_sgs,
                    dgrad_wc_fp4_cols, dgrad_wc_sc_cols, dgrad_wc_sg_cols,
                    d_normed,
                    -1,
                )
            elif use_direct_split2:
                split2_onepass_cfg = 3 if use_v4_split2_two_stage else -1
                tk.nvfp4_split2_dgrad_strided_onepass_gemm(
                    row_fp4_full,
                    row_scs,
                    [0, H // 2],
                    [H // 2, H // 2],
                    dgrad_wc_fp4_cols,
                    dgrad_wc_sc_cols,
                    dgrad_wc_sg_cols,
                    d_normed,
                    split2_onepass_cfg,
                )
            elif hasattr(tk, 'nvfp4_v3_split2_dgrad_onepass_gemm'):
                tk.nvfp4_v3_split2_dgrad_onepass_gemm(
                    row_fp4s, row_scs, row_sgs,
                    dgrad_wc_fp4_cols, dgrad_wc_sc_cols, dgrad_wc_sg_cols,
                    d_normed,
                    tk_localcta_v3_split2_onepass_config_idx(),
                )
            elif hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm'):
                tk.nvfp4_split2_dgrad_onepass_gemm(
                    row_fp4s, row_scs, row_sgs,
                    dgrad_wc_fp4_cols, dgrad_wc_sc_cols, dgrad_wc_sg_cols,
                    d_normed,
                    -1,
                )
            else:
                tk.nvfp4_batched_accum_gemm(
                    row_fp4s, row_scs, row_sgs,
                    dgrad_wc_fp4_cols, dgrad_wc_sc_cols, dgrad_wc_sg_cols,
                    d_normed,
                )
        _tk_stage_trace('ffn_bwd_localcta_sub', 'split_dgrad_done', debug_name)
        _debug_check_finite('ffn_bwd.localcta.d_normed', d_normed)
        _dump_ffn_tensors(
            "split2_post",
            debug_name,
            {
                "debug_name": debug_name,
                "d_normed": d_normed,
                "inv_rms": inv_rms,
                "norm_weight": norm_weight,
            },
        )
        if _ffn_capture_path():
            _append_ffn_capture({
                "event": "ffn_d_normed",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "use_direct_split2": bool(use_direct_split2),
                "use_two_single_row_split2_quant": bool(use_two_single_row_split2_quant),
                "d_normed": _tensor_capture_stats(d_normed),
            })

        _tk_stage_trace('ffn_bwd_localcta_sub', 'rmsnorm_start', debug_name)
        with _ffn_cuda_timed(ffn_timings, "rmsnorm_launch"):
            if h_tile:
                from .tk_gemm import tk_h_tile_backward
                grad_input, dgamma = tk_h_tile_backward(
                    d_normed, input, norm_weight, inv_rms
                )
                rms_state = {
                    'grad_input': grad_input,
                    'dgamma': dgamma,
                }
                rms_stream = None
            else:
                rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                    d_normed, input, norm_weight, inv_rms, te_fused,
                    tag='ffn',
                    force_current_stream=(
                        strict_v4_localcta
                        and use_tk_localcta_v4_sync_ffn_rms_bwd()
                    ),
                    residual_grad=residual_grad,
                )
        _tk_stage_trace('ffn_bwd_localcta_sub', 'rmsnorm_done', debug_name)

        _tk_stage_trace('ffn_bwd_localcta_sub', 'final_waits_start', debug_name)
        with _ffn_cuda_timed(ffn_timings, "final_waits"):
            if use_saved_sigmoid_overlap:
                torch.cuda.current_stream().wait_event(localcta_state['wgrad_done_event'])
            else:
                torch.cuda.current_stream().wait_stream(wgrad_stream)
            if split2_grad_boost != 1.0:
                grad_w1.div_(split2_grad_boost)
                grad_w3.div_(split2_grad_boost)
                d_normed.div_(split2_grad_boost)
            if not use_localcta_direct_ffn and ffn_grad_boost != 1.0:
                grad_w2.div_(ffn_grad_boost)
            if rms_stream is not None:
                torch.cuda.current_stream().wait_stream(rms_stream)
            if w2_dgrad_silu_split2_q is not None:
                torch.cuda.current_stream().record_event(
                    localcta_state['w2_dgrad_silu_payload_ready_event']
                )
                localcta_state['w2_dgrad_silu_payload_ready_recorded'] = True
        _tk_stage_trace('ffn_bwd_localcta_sub', 'final_waits_done', debug_name)
        grad_input = rms_state['grad_input']
        dgamma = rms_state.get('dgamma_out', rms_state['dgamma'])
        _sync_after_ffn_bwd_if_enabled(debug_name, M, K, H)
        _emit_ffn_debug_timings_once(debug_name, M, K, H, ffn_timings)
        _debug_check_finite('ffn_bwd.localcta.grad_input', grad_input)
        if _ffn_capture_path():
            _append_ffn_capture({
                "event": "ffn_rmsnorm_bwd",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "inv_rms": _tensor_capture_stats(inv_rms),
                "grad_input": _tensor_capture_stats(grad_input),
                "dgamma": _tensor_capture_stats(dgamma),
            })
        _debug_check_finite('ffn_bwd.localcta.dgamma', dgamma)
        # Materialize all fallback outputs before returning. The eager FFN
        # path mixes cached scratch buffers, TE fused outputs, and tensors
        # produced by helper wrappers. Returning them directly has repeatedly
        # led to later layers clobbering earlier layers' grads before autograd
        # finishes accumulation on standard 1B small-M runs.
        grad_norm_weight = _as_param_grad_dtype(dgamma, norm_weight)
        return (
            _maybe_clone_localcta_v4_ffn_grad_input(grad_input, M),
            _maybe_clone_localcta_v4_ffn_return(grad_w1, M),
            _maybe_clone_localcta_v4_ffn_return(grad_w3, M),
            _maybe_clone_localcta_v4_ffn_return(grad_w2, M),
            _maybe_clone_localcta_v4_ffn_return(grad_norm_weight, M),
            dY_underflow_requant_info,
        )

    key = (M, K, H, grad_output.device.index)
    # Adaptive threshold: CUDA graphs help at M≤4096 (1.10-1.37x speedup from
    # eliminating CPU kernel launch overhead). At large M the D2D copy overhead
    # to fixed graph addresses exceeds the diminishing launch savings.
    # See cuda_graph_analysis.md for the full profiling deep-dive.
    _graph_max_m = int(os.environ.get('FFN_GRAPH_MAX_M', '4096'))
    cache = _ffn_bwd_graph_cache.get(key) if M <= _graph_max_m and not h_tile else None

    if cache is not None:
        # ── Replay path ──
        # h13/input: zero-copy (forward wrote directly to sb addresses).
        # Other tensors: written to sb on the same stream during forward.
        # Only dY needs a backward copy (comes from upstream backward).
        graph, sb = cache
        sb['dY'].copy_(grad_output)
        if 'h1_raw' in sb:
            sb['h1_raw'].copy_(h1_raw)
        if 'h3' in sb:
            sb['h3'].copy_(h3)
        if 'h13' in sb and h13.numel() != 0:
            sb['h13'].copy_(h13)
        # No synchronize here — all ops are on the same CUDA stream, so
        # GPU-side ordering is guaranteed. The forward's sync (line 1334)
        # handles the TMA descriptor drain for M>=32768. Adding sync here
        # creates a CPU bubble that serializes fwd/bwd, costing ~0.5ms at
        # M=65536 (confirmed via split profiling).
        graph.replay()
        # Materialize replay outputs before returning. The graph state is
        # shared per (M, K, H, device), so every FFN layer with the same shape
        # reuses the same backing buffers. Returning the static buffers
        # directly lets later layer replays overwrite earlier layers' grads
        # before autograd has finished consuming them.
        return (
            sb['grad_input'].clone(),
            sb['grad_w1'].clone(),
            sb['grad_w3'].clone(),
            sb['grad_w2'].clone(),
            sb['grad_nw'].clone(),
        )

    if h_tile or not use_cuda_graph() or M > _graph_max_m:
        # ── Non-graphed fallback (fully inlined C++ calls) ──
        # All wrapper functions (_fast_quantize, tk_dgrad_gemm, etc.) are
        # bypassed. Module references and output buffers are cached to
        # eliminate per-call Python overhead (~0.1ms per wrapper × 5 = ~0.5ms).
        ffn_timings = [] if use_tk_ffn_debug_timings_for(debug_name) else None

        def _ffn_timing_mark(label: str) -> None:
            if ffn_timings is None:
                return
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            ffn_timings.append((label, event))

        def _ffn_timing_emit(return_mode: str) -> None:
            if not ffn_timings:
                return
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            events = ffn_timings + [('end', end_event)]
            total_ms = events[0][1].elapsed_time(events[-1][1])
            parts = [
                f"{label}->{next_label}={event.elapsed_time(next_event):.3f}ms"
                for (label, event), (next_label, next_event) in zip(events, events[1:])
            ]
            print(
                f"[FFN TIMING] mode={return_mode} debug={debug_name or 'ffn'} "
                f"M={M} K={K} H={H} total={total_ms:.3f}ms " + " ".join(parts),
                file=sys.stderr,
                flush=True,
            )

        grad_input = dgamma = None
        rms_stream = None
        rms_launched = False

        def _launch_regular_ffn_rms_async(d_normed_tensor) -> bool:
            nonlocal grad_input, dgamma, rms_stream, rms_launched
            if h_tile:
                return False
            if not use_tk_ffn_overlap_rms_wgrad():
                return False
            _tk_stage_trace('ffn_bwd_sub', 'rmsnorm_start', debug_name)
            rms_stream = _get_rmsnorm_bwd_stream()
            rms_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(rms_stream):
                if residual_grad is not None:
                    rms_state, _ = _launch_rmsnorm_bwd_out_async(
                        d_normed_tensor,
                        input,
                        norm_weight,
                        inv_rms,
                        te_fused,
                        tag='ffn',
                        force_current_stream=True,
                        residual_grad=residual_grad,
                    )
                    grad_input = rms_state['grad_input']
                    dgamma = rms_state.get('dgamma_out', rms_state['dgamma'])
                else:
                    _record_tensors_on_stream(
                        (d_normed_tensor, input, norm_weight, inv_rms),
                        rms_stream,
                    )
                    grad_input, dgamma = te_fused.fused_rmsnorm_backward(
                        d_normed_tensor, input, norm_weight, inv_rms
                    )
                _record_tensors_on_stream((grad_input, dgamma), rms_stream)
            rms_launched = True
            _ffn_timing_mark('rms_launch_done')
            return True

        def _finish_regular_ffn_rms(d_normed_tensor):
            nonlocal grad_input, dgamma, rms_stream
            if rms_launched:
                torch.cuda.current_stream().wait_stream(rms_stream)
            else:
                _tk_stage_trace('ffn_bwd_sub', 'rmsnorm_start', debug_name)
                if h_tile:
                    from .tk_gemm import tk_h_tile_backward
                    grad_input, dgamma = tk_h_tile_backward(
                        d_normed_tensor, input, norm_weight, inv_rms
                    )
                elif residual_grad is not None:
                    rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                        d_normed_tensor,
                        input,
                        norm_weight,
                        inv_rms,
                        te_fused,
                        tag='ffn',
                        force_current_stream=True,
                        residual_grad=residual_grad,
                    )
                    grad_input = rms_state['grad_input']
                    dgamma = rms_state.get('dgamma_out', rms_state['dgamma'])
                elif isinstance(d_normed_tensor, (tuple, list)):
                    grad_input, dgamma = te_fused.fused_rmsnorm_backward_sum2(
                        d_normed_tensor[0], d_normed_tensor[1],
                        input, norm_weight, inv_rms,
                    )
                else:
                    grad_input, dgamma = te_fused.fused_rmsnorm_backward(
                        d_normed_tensor, input, norm_weight, inv_rms
                    )
            _tk_stage_trace('ffn_bwd_sub', 'rmsnorm_done', debug_name)
            _tk_ffn_debug_sync_checkpoint('ffn_rmsnorm')
            _ffn_timing_mark('rms_done')
            return grad_input, dgamma

        _tk_mod = tk                        # already resolved at function entry
        _tk_q_mod = _get_tk_quant()         # cached singleton
        dev = input.device
        can_fuse_ffn_sum_rms = (
            not use_localcta
            and not h_tile
            and residual_grad is None
            and use_tk_ffn_fused_sum_rms()
            and hasattr(te_fused, 'fused_rmsnorm_backward_sum2')
            and not use_tk_ffn_overlap_rms_wgrad()
            and not use_tk_ffn_debug_finite()
            and not _ffn_capture_path()
        )

        # ─── 1. Quantize dY (direct C++ call, no _fast_quantize wrapper) ───
        # grad_output from autograd may be non-contiguous
        _ffn_timing_mark('start')
        _go = grad_output.contiguous() if not grad_output.is_contiguous() else grad_output
        _tk_ffn_debug_assert_finite(
            "backward_inputs",
            debug_name,
            (
                ("grad_output", _go),
                ("input", input),
                ("norm_weight", norm_weight),
                ("inv_rms", inv_rms),
                ("residual_grad", residual_grad),
                ("h1_raw", h1_raw),
                ("h3", h3),
            ),
        )
        _tk_stage_trace('ffn_bwd_sub', 'dy_quant_start', debug_name)
        if _nvfp4_quantizer_extras_enabled("grad"):
            dY_quant = _tk_quantized_as_result_tuple(
                _fast_quantize_tk_regular_opt(_go, nvfp4_role="grad")
            )
        else:
            dY_quant = _tk_q_mod.tk_quantize_for_gemm(_go, True)
        _tk_stage_trace('ffn_bwd_sub', 'dy_quant_done', debug_name)
        _tk_ffn_debug_sync_checkpoint('ffn_dy_quant')
        _ffn_timing_mark('dy_quant_done')
        # dY_quant = (fp4, sc, fp4_t, sc_t, sg, sg_t)
        dY_fp4, dY_sc, dY_sg = dY_quant[0], dY_quant[1], dY_quant[4]
        dY_fp4c, dY_scc, dY_sgc = (
            dY_quant[2],
            dY_quant[3],
            dY_quant[5] if len(dY_quant) > 5 and torch.is_tensor(dY_quant[5]) and dY_quant[5].numel() > 0 else dY_quant[4],
        )

        # ─── 2. W2 dgrad: dh = dY @ W2^T (direct nvfp4_gemm) ───
        w2_fp4c, w2_scc, w2_sgc = w2_nvfp4._tk_col
        # Cache dh buffer
        _bwd_key = (M, H, K, dev.index)
        if not hasattr(_ffn_bwd_graphed, '_buf_cache') or \
           _ffn_bwd_graphed._buf_key != _bwd_key:
            _ffn_bwd_graphed._buf_cache = {
                'dh': torch.empty(M, H, dtype=torch.bfloat16, device=dev),
                'h_w2': torch.empty(M, H, dtype=torch.bfloat16, device=dev),
                'grad_w2': torch.empty(K, H, dtype=torch.bfloat16, device=dev),
                'd1': torch.empty(M, K, dtype=torch.bfloat16, device=dev),
                'd2': torch.empty(M, K, dtype=torch.bfloat16, device=dev),
                'dh1': torch.empty(M, H, dtype=torch.bfloat16, device=dev),
                'dh3': torch.empty(M, H, dtype=torch.bfloat16, device=dev),
                'amax1': torch.zeros(1, dtype=torch.float32, device=dev),
                'amax2': torch.zeros(1, dtype=torch.float32, device=dev),
            }
            _ffn_bwd_graphed._buf_key = _bwd_key
        bufs = _ffn_bwd_graphed._buf_cache
        dh = bufs['dh']
        _tk_stage_trace('ffn_bwd_sub', 'w2_dgrad_start', debug_name)
        if (
            use_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
            and use_tk_localcta_v4_raw_backward_fallbacks(grad_output.size(0))
            and w2_dgrad_col is not None
        ):
            tk_v4_direct_raw_dgrad_gemm(
                grad_output,
                w2_dgrad_col,
                dh,
            )
        elif use_tk_debug_ffn_bf16_w2_dgrad():
            if w2_bf16 is None:
                raise RuntimeError(
                    "BF16 W2 dgrad diagnostic requires ctx.w2_bf16"
                )
            dh.copy_(
                torch.matmul(_go.float(), w2_bf16.float()).to(torch.bfloat16)
            )
        else:
            tk_dispatch_gemm(_tk_mod, dY_fp4, dY_sc, dY_sg, w2_fp4c, w2_scc, w2_sgc, dh)
        _tk_stage_trace('ffn_bwd_sub', 'w2_dgrad_done', debug_name)
        _tk_ffn_debug_sync_checkpoint('ffn_w2_dgrad')
        _ffn_timing_mark('w2_dgrad_done')
        _tk_ffn_debug_assert_finite(
            "w2_dgrad_output", debug_name, (("dh", dh),)
        )
        _debug_check_finite('ffn_bwd.dh', dh)
        _dump_ffn_tensors(
            "w2_dgrad",
            debug_name,
            {
                "debug_name": debug_name,
                "grad_output": grad_output,
                "dY_fp4_row": dY_fp4,
                "dY_sc_row": dY_sc,
                "dY_sg_row": dY_sg,
                "w2_fp4_col": w2_fp4c,
                "w2_sc_col": w2_scc,
                "w2_sg_col": w2_sgc,
                "dh": dh,
            },
        )

        # ─── 3. W2 wgrad: grad_w2 = dY^T @ h (direct nvfp4_gemm) ───
        h_fp4c, h_scc, h_sgc = h_nvfp4._tk_col
        grad_w2 = bufs['grad_w2']
        h_w2 = bufs['h_w2']
        _tk_stage_trace('ffn_bwd_sub', 'w2_wgrad_start', debug_name)
        if (
            use_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
            and not use_ffn_localcta_tk_quant_contract
            and use_tk_localcta_v4_raw_backward_fallbacks(grad_output.size(0))
        ):
            if hasattr(te_fused, 'fused_silu_mul_bf16_out_no_amax'):
                te_fused.fused_silu_mul_bf16_out_no_amax(h1_raw, h3, h_w2)
            else:
                h_w2.copy_((F.silu(h1_raw.float()) * h3.float()).to(torch.bfloat16))
            tk_v4_direct_raw_wgrad_gemm(
                grad_output,
                h_w2,
                grad_w2,
            )
        else:
            tk_dispatch_gemm(_tk_mod, dY_fp4c, dY_scc, dY_sgc, h_fp4c, h_scc, h_sgc, grad_w2)
        _tk_stage_trace('ffn_bwd_sub', 'w2_wgrad_done', debug_name)
        _tk_ffn_debug_sync_checkpoint('ffn_w2_wgrad')
        _ffn_timing_mark('w2_wgrad_done')
        _debug_check_finite('ffn_bwd.grad_w2', grad_w2)
        _dump_ffn_tensors(
            "w2_wgrad",
            debug_name,
            {
                "debug_name": debug_name,
                "grad_output": grad_output,
                "h1_raw": h1_raw,
                "h3": h3,
                "dY_fp4_col": dY_fp4c,
                "dY_sc_col": dY_scc,
                "dY_sg_col": dY_sgc,
                "h_fp4_col": h_fp4c,
                "h_sc_col": h_scc,
                "h_sg_col": h_sgc,
                "grad_w2": grad_w2,
            },
        )

        # ─── 4. Fused silu_deriv + quantize (direct C++ call) ───
        use_split_cache = (
            h1_raw is not None and h3 is not None
            and h1_raw.numel() != 0 and h3.numel() != 0
            and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_split_for_gemm')
        )
        use_v5_delayed_split_producer = (
            use_split_cache
            and not use_localcta
            and use_tk_ffn_v5_delayed_split_silu_deriv()
            and os.environ.get('USE_TK_FFN_V5_DELAYED_DIRECT_SPLIT', '0') == '1'
            and not _nvfp4_quantizer_extras_enabled("grad")
            and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_split_for_gemm_delayed')
        )
        use_v5_delayed_safe_quant = (
            use_split_cache
            and not use_localcta
            and use_tk_ffn_v5_delayed_split_silu_deriv()
            and not use_v5_delayed_split_producer
            and not _nvfp4_quantizer_extras_enabled("grad")
            and hasattr(_tk_q_mod, 'tk_quantize_for_gemm_delayed')
            and M >= 256
        )
        use_prealloc_split_producer = (
            use_split_cache
            and not use_v5_delayed_split_producer
            and not use_localcta
            and use_tk_ffn_prealloc_split_producer()
            and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_split_for_gemm_alloc')
            and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_split_for_gemm_launch')
        )
        use_safe_split_producer = (
            use_split_cache
            and not use_v5_delayed_split_producer
            and not use_localcta
            and (
                M < 256
                or use_tk_ffn_bwd_safe_producer(M, K, H)
                or _nvfp4_quantizer_extras_enabled("grad")
                or use_tk_debug_ffn_bf16_split_dgrad()
            )
            and hasattr(_tk_q_mod, 'tk_batched_quantize_for_gemm')
        )
        use_split2_opt_producer = (
            use_safe_split_producer
            and M >= 256
            and use_tk_ffn_split2_opt_producer()
            and _nvfp4_quantizer_extras_enabled("grad")
            and not use_tk_debug_ffn_bf16_split_dgrad()
            and _nvfp4_native_rht_axes_for_role("grad") == "row"
            and not (use_nvfp4_rht_for_role("grad") and _nvfp4_rht_random_sign_mask())
            and not _ffn_capture_path()
            and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_split_for_gemm_opt')
        )
        use_prealloc_split2_opt_producer = (
            use_split2_opt_producer
            and use_tk_ffn_split2_persistent_producer()
            and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_split_for_gemm_opt_alloc')
            and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_split_for_gemm_opt_launch')
        )
        if use_prealloc_split_producer and not use_safe_split_producer:
            split_prod_alloc = bufs.get('split_prod_alloc')
            if split_prod_alloc is None:
                split_prod_alloc = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm_alloc(M, H, dev)
                bufs['split_prod_alloc'] = split_prod_alloc
        if use_prealloc_split2_opt_producer:
            split2_opt_alloc = bufs.get('split2_opt_alloc')
            if split2_opt_alloc is None:
                split2_opt_alloc = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm_opt_alloc(M, H, dev)
                bufs['split2_opt_alloc'] = split2_opt_alloc
        _tk_stage_trace('ffn_bwd_sub', 'split_prod_start', debug_name)
        if use_v5_delayed_split_producer:
            device_index = dh.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cache_key = (debug_name, int(device_index), int(M), int(H))
            if not hasattr(_ffn_bwd_graphed, '_v5_split_delayed_amax_cache'):
                _ffn_bwd_graphed._v5_split_delayed_amax_cache = {}
                _ffn_bwd_graphed._v5_split_delayed_amax_age = {}
                _ffn_bwd_graphed._v5_split_delayed_amax_event = {}
            elif not hasattr(_ffn_bwd_graphed, '_v5_split_delayed_amax_event'):
                _ffn_bwd_graphed._v5_split_delayed_amax_event = {}
            amax_cache = _ffn_bwd_graphed._v5_split_delayed_amax_cache
            age_cache = _ffn_bwd_graphed._v5_split_delayed_amax_age
            event_cache = _ffn_bwd_graphed._v5_split_delayed_amax_event
            prev_amax = amax_cache.get(cache_key)
            if prev_amax is None:
                fused = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm(dh, h3, h1_raw)
                if len(fused) > 12 and torch.is_tensor(fused[12]) and fused[12].numel() >= 2:
                    amax_cache[cache_key] = fused[12].detach()
                    event_cache[cache_key] = _record_delayed_amax_event(amax_cache[cache_key])
                    age_cache[cache_key] = 0
            else:
                _wait_delayed_amax_event(event_cache.get(cache_key), prev_amax)
                refresh_interval = tk_ffn_v5_delayed_refresh_interval()
                age = age_cache.get(cache_key, 0)
                collect_current_amax = (
                    refresh_interval > 0
                    and (refresh_interval <= 1 or age >= refresh_interval - 1)
                )
                fused = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm_delayed(
                    dh, h3, h1_raw, prev_amax, collect_current_amax
                )
                if collect_current_amax:
                    if len(fused) > 12 and torch.is_tensor(fused[12]) and fused[12].numel() >= 2:
                        amax_cache[cache_key] = fused[12].detach()
                        event_cache[cache_key] = _record_delayed_amax_event(amax_cache[cache_key])
                    age_cache[cache_key] = 0
                else:
                    age_cache[cache_key] = age + 1
        elif use_safe_split_producer:
            target_rows = 256
            dh1 = bufs['dh1']
            dh3_out = bufs['dh3']
            safe_delayed_prev_amax = None
            safe_delayed_collect_amax = False
            safe_delayed_cache_key = None
            if use_v5_delayed_safe_quant:
                device_index = dh.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                safe_delayed_cache_key = (debug_name, int(device_index), int(M), int(H))
                if not hasattr(_ffn_bwd_graphed, '_v5_safe_delayed_amax_cache'):
                    _ffn_bwd_graphed._v5_safe_delayed_amax_cache = {}
                    _ffn_bwd_graphed._v5_safe_delayed_amax_age = {}
                    _ffn_bwd_graphed._v5_safe_delayed_amax_event = {}
                elif not hasattr(_ffn_bwd_graphed, '_v5_safe_delayed_amax_event'):
                    _ffn_bwd_graphed._v5_safe_delayed_amax_event = {}
                amax_cache = _ffn_bwd_graphed._v5_safe_delayed_amax_cache
                age_cache = _ffn_bwd_graphed._v5_safe_delayed_amax_age
                event_cache = _ffn_bwd_graphed._v5_safe_delayed_amax_event
                safe_delayed_prev_amax = amax_cache.get(safe_delayed_cache_key)
                if safe_delayed_prev_amax is None:
                    safe_delayed_collect_amax = True
                else:
                    _wait_delayed_amax_event(event_cache.get(safe_delayed_cache_key), safe_delayed_prev_amax)
                    refresh_interval = tk_ffn_v5_delayed_refresh_interval()
                    age = age_cache.get(safe_delayed_cache_key, 0)
                    safe_delayed_collect_amax = (
                        refresh_interval > 0
                        and (refresh_interval <= 1 or age >= refresh_interval - 1)
                    )
            if use_split2_opt_producer:
                split2_args = (
                    dh, h3, h1_raw,
                    True,
                    use_nvfp4_encode_centric(),
                    use_nvfp4_data_stochastic_rounding_for_role("grad"),
                    use_nvfp4_scale_stochastic_rounding_for_role("grad"),
                    _nvfp4_native_rht_axes_for_role("grad"),
                    False,
                    _nvfp4_rng_seed(),
                    _nvfp4_rng_subsequence_base(),
                )
                split2_data_sr_axes = (
                    _nvfp4_grad_sr_axes()
                    if use_nvfp4_data_stochastic_rounding_for_role("grad")
                    else "none"
                )
                if use_prealloc_split2_opt_producer:
                    fused = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm_opt_launch(
                        *split2_args,
                        *bufs['split2_opt_alloc'],
                        split2_data_sr_axes,
                    )
                else:
                    fused = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm_opt(
                        *split2_args,
                        split2_data_sr_axes,
                    )
                row_fp4s = [fused[0], fused[6]]
                row_scs = [fused[1], fused[7]]
                col_fp4s = [fused[2], fused[8]]
                col_scs = [fused[3], fused[9]]
                row_sgs = [fused[4], fused[10]]
                col_sgs = [fused[5], fused[11]]
            else:
                if (
                    safe_delayed_collect_amax
                    and hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out')
                ):
                    te_fused.fused_silu_deriv_dual_mul_bf16_out(
                        dh, h3, h1_raw, dh1, dh3_out,
                        bufs['amax1'], bufs['amax2'],
                    )
                elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
                    te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                        dh, h3, h1_raw, dh1, dh3_out,
                    )
                elif hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out'):
                    te_fused.fused_silu_deriv_dual_mul_bf16_out(
                        dh, h3, h1_raw, dh1, dh3_out,
                        bufs['amax1'], bufs['amax2'],
                    )
                else:
                    _dh1, _dh3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1_raw)
                    dh1.copy_(_dh1)
                    dh3_out.copy_(_dh3)
                _debug_check_finite('ffn_bwd.safe.dh1', dh1)
                _debug_check_finite('ffn_bwd.safe.dh3', dh3_out)
                if use_v5_delayed_safe_quant:
                    if safe_delayed_prev_amax is None:
                        q_amax1 = bufs['amax1']
                        q_amax2 = bufs['amax2']
                    else:
                        q_amax1 = safe_delayed_prev_amax.narrow(0, 0, 1)
                        q_amax2 = safe_delayed_prev_amax.narrow(0, 1, 1)
                    dh1_q = _tk_q_mod.tk_quantize_for_gemm_delayed(
                        dh1, q_amax1, True, use_nvfp4_encode_centric(), False
                    )
                    dh3_q = _tk_q_mod.tk_quantize_for_gemm_delayed(
                        dh3_out, q_amax2, True, use_nvfp4_encode_centric(), False
                    )
                    if safe_delayed_collect_amax and safe_delayed_cache_key is not None:
                        _ffn_bwd_graphed._v5_safe_delayed_amax_cache[safe_delayed_cache_key] = torch.cat(
                            (bufs['amax1'].detach(), bufs['amax2'].detach())
                        )
                        _ffn_bwd_graphed._v5_safe_delayed_amax_event[safe_delayed_cache_key] = _record_delayed_amax_event(
                            _ffn_bwd_graphed._v5_safe_delayed_amax_cache[safe_delayed_cache_key]
                        )
                        _ffn_bwd_graphed._v5_safe_delayed_amax_age[safe_delayed_cache_key] = 0
                    elif safe_delayed_cache_key is not None:
                        age_cache = _ffn_bwd_graphed._v5_safe_delayed_amax_age
                        age_cache[safe_delayed_cache_key] = age_cache.get(safe_delayed_cache_key, 0) + 1
                    row_fp4s = [dh1_q[0], dh3_q[0]]
                    row_scs = [dh1_q[1], dh3_q[1]]
                    col_fp4s = [dh1_q[2], dh3_q[2]]
                    col_scs = [dh1_q[3], dh3_q[3]]
                    row_sgs = [dh1_q[4], dh3_q[4]]
                    col_sgs = [
                        dh1_q[5] if len(dh1_q) > 5 else dh1_q[4],
                        dh3_q[5] if len(dh3_q) > 5 else dh3_q[4],
                    ]
                elif _nvfp4_quantizer_extras_enabled("grad"):
                    if M < 256:
                        dh1_for_quant = _pad_rows_bf16(dh1, target_rows)
                        dh3_for_quant = _pad_rows_bf16(dh3_out, target_rows)
                    else:
                        dh1_for_quant = dh1
                        dh3_for_quant = dh3_out
                    dh1_q = _tk_quantized_as_result_tuple(
                        _fast_quantize_tk_regular_opt(dh1_for_quant, nvfp4_role="grad")
                    )
                    dh3_q = _tk_quantized_as_result_tuple(
                        _fast_quantize_tk_regular_opt(dh3_for_quant, nvfp4_role="grad")
                    )
                    row_fp4s = [dh1_q[0], dh3_q[0]]
                    row_scs = [dh1_q[1], dh3_q[1]]
                    col_fp4s = [dh1_q[2], dh3_q[2]]
                    col_scs = [dh1_q[3], dh3_q[3]]
                    row_sgs = [dh1_q[4], dh3_q[4]]
                    col_sgs = [dh1_q[5], dh3_q[5]]
                elif M < 256:
                    dh1_pad = _pad_rows_bf16(dh1, target_rows)
                    dh3_pad = _pad_rows_bf16(dh3_out, target_rows)
                    dh1_q = _tk_q_mod.tk_quantize_for_gemm(dh1_pad, True)
                    dh3_q = _tk_q_mod.tk_quantize_for_gemm(dh3_pad, True)
                    row_fp4s = [dh1_q[0], dh3_q[0]]
                    row_scs = [dh1_q[1], dh3_q[1]]
                    col_fp4s = [dh1_q[2], dh3_q[2]]
                    col_scs = [dh1_q[3], dh3_q[3]]
                    row_sgs = [dh1_q[4], dh3_q[4]]
                    col_sgs = [
                        dh1_q[5] if len(dh1_q) > 5 else dh1_q[4],
                        dh3_q[5] if len(dh3_q) > 5 else dh3_q[4],
                    ]
                elif use_tk_ffn_split_quant_eager():
                    dh1_q = _tk_q_mod.tk_quantize_for_gemm(dh1, True)
                    dh3_q = _tk_q_mod.tk_quantize_for_gemm(dh3_out, True)
                    row_fp4s = [dh1_q[0], dh3_q[0]]
                    row_scs = [dh1_q[1], dh3_q[1]]
                    col_fp4s = [dh1_q[2], dh3_q[2]]
                    col_scs = [dh1_q[3], dh3_q[3]]
                    row_sgs = [dh1_q[4], dh3_q[4]]
                    col_sgs = [dh1_q[5] if len(dh1_q) > 5 else dh1_q[4],
                               dh3_q[5] if len(dh3_q) > 5 else dh3_q[4]]
                else:
                    split2_q = _tk_q_mod.tk_batched_quantize_for_gemm([dh1, dh3_out], True, True)
                    row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs = split2_q[:6]
            _tk_ffn_debug_assert_finite(
                "silu_deriv_outputs",
                debug_name,
                (("dh1", dh1), ("dh3", dh3_out)),
            )
            for idx, tensor in enumerate(row_scs):
                _debug_check_finite(f'ffn_bwd.safe.row_sc_{idx}', tensor)
            for idx, tensor in enumerate(row_sgs):
                _debug_check_finite(f'ffn_bwd.safe.row_sg_{idx}', tensor)
            for idx, tensor in enumerate(col_scs):
                _debug_check_finite(f'ffn_bwd.safe.col_sc_{idx}', tensor)
            for idx, tensor in enumerate(col_sgs):
                _debug_check_finite(f'ffn_bwd.safe.col_sg_{idx}', tensor)
            if _ffn_capture_path():
                _append_ffn_capture({
                    "event": "ffn_split2_inputs",
                    "debug_name": debug_name,
                    "M": int(M),
                    "K": int(K),
                    "H": int(H),
                    "use_direct_split2": False,
                    "use_two_single_row_split2_quant": False,
                    "use_row_only_split2_quant": False,
                    "use_fused_row_producer_split2_quant": False,
                    "dh": _tensor_capture_stats(dh),
                    "dh1": _tensor_capture_stats(dh1),
                    "dh3_out": _tensor_capture_stats(dh3_out),
                })
                _append_ffn_capture({
                    "event": "ffn_split2_quant",
                    "debug_name": debug_name,
                    "M": int(M),
                    "K": int(K),
                    "H": int(H),
                    "row_fp4_0": _tensor_capture_stats(row_fp4s[0]),
                    "row_fp4_1": _tensor_capture_stats(row_fp4s[1]),
                    "row_sc_0": _tensor_capture_stats(row_scs[0]),
                    "row_sc_1": _tensor_capture_stats(row_scs[1]),
                    "row_sg_0": _tensor_capture_stats(row_sgs[0]),
                    "row_sg_1": _tensor_capture_stats(row_sgs[1]),
                    "col_sc_0": _tensor_capture_stats(col_scs[0]),
                    "col_sc_1": _tensor_capture_stats(col_scs[1]),
                    "col_sg_0": _tensor_capture_stats(col_sgs[0]),
                    "col_sg_1": _tensor_capture_stats(col_sgs[1]),
                    "w1_sc_c": _tensor_capture_stats(wc_sc_cols[0]),
                    "w3_sc_c": _tensor_capture_stats(wc_sc_cols[1]),
                    "w1_fp4_c": _tensor_capture_stats(wc_fp4_cols[0]),
                    "w3_fp4_c": _tensor_capture_stats(wc_fp4_cols[1]),
                    "w1_sg": _tensor_capture_stats(sg_cat[0:1]),
                    "w3_sg": _tensor_capture_stats(sg_cat[1:2]),
                })
        elif use_split_cache:
            if use_prealloc_split_producer:
                fused = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm_launch(
                    dh, h3, h1_raw, *bufs['split_prod_alloc']
                )
            elif getattr(_tk_q_mod, 'is_localcta', False):
                fused = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm(
                    dh, h3, h1_raw, debug_name
                )
            else:
                fused = _tk_q_mod.tk_silu_deriv_quantize_split_for_gemm(dh, h3, h1_raw)
        else:
            use_delayed_h13 = use_tk_ffn_h13_delayed_silu_deriv()
            if (
                use_delayed_h13
                and not getattr(_tk_q_mod, 'is_localcta', False)
                and hasattr(_tk_q_mod, 'tk_silu_deriv_quantize_for_gemm_delayed')
            ):
                device_index = dh.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                cache_key = (debug_name, int(device_index), int(M), int(H))
                if not hasattr(_ffn_bwd_graphed, '_v5_delayed_amax_cache'):
                    _ffn_bwd_graphed._v5_delayed_amax_cache = {}
                    _ffn_bwd_graphed._v5_delayed_amax_age = {}
                    _ffn_bwd_graphed._v5_delayed_amax_event = {}
                elif not hasattr(_ffn_bwd_graphed, '_v5_delayed_amax_event'):
                    _ffn_bwd_graphed._v5_delayed_amax_event = {}
                if not hasattr(_ffn_bwd_graphed, '_v5_delayed_amax_age'):
                    _ffn_bwd_graphed._v5_delayed_amax_age = {}
                amax_cache = _ffn_bwd_graphed._v5_delayed_amax_cache
                age_cache = _ffn_bwd_graphed._v5_delayed_amax_age
                event_cache = _ffn_bwd_graphed._v5_delayed_amax_event
                prev_amax = amax_cache.get(cache_key)
                if prev_amax is None:
                    fused = _tk_q_mod.tk_silu_deriv_quantize_for_gemm(dh, h13, H, False)
                    collect_current_amax = True
                else:
                    _wait_delayed_amax_event(event_cache.get(cache_key), prev_amax)
                    refresh_interval = tk_ffn_h13_delayed_refresh_interval()
                    age = age_cache.get(cache_key, 0)
                    collect_current_amax = (
                        refresh_interval > 0
                        and (refresh_interval <= 1 or age >= refresh_interval - 1)
                    )
                    fused = _tk_q_mod.tk_silu_deriv_quantize_for_gemm_delayed(
                        dh, h13, H, prev_amax, collect_current_amax
                    )
                if collect_current_amax and len(fused) > 12 and torch.is_tensor(fused[12]) and fused[12].numel() >= 2:
                    amax_cache[cache_key] = fused[12].detach()
                    event_cache[cache_key] = _record_delayed_amax_event(amax_cache[cache_key])
                    age_cache[cache_key] = 0
                elif prev_amax is not None:
                    age_cache[cache_key] = age_cache.get(cache_key, 0) + 1
            elif getattr(_tk_q_mod, 'is_localcta', False):
                fused = _tk_q_mod.tk_silu_deriv_quantize_for_gemm(
                    dh, h13, H, use_delayed_h13, debug_name
                )
            else:
                fused = _tk_q_mod.tk_silu_deriv_quantize_for_gemm(
                    dh, h13, H, use_delayed_h13
                )
        _tk_stage_trace('ffn_bwd_sub', 'split_prod_done', debug_name)
        _tk_ffn_debug_sync_checkpoint('ffn_split_producer')
        _ffn_timing_mark('split_prod_done')
        # fused = (dh1_fp4, dh1_sc, dh1_fp4_t, dh1_sc_t, dh1_sg, zeros,
        #          dh3_fp4, dh3_sc, dh3_fp4_t, dh3_sc_t, dh3_sg, zeros)

        # ─── 5. W1/W3 batched dgrad (direct nvfp4_batched_gemm) ───
        w1_fp4_c = wc_fp4_cols[0].view(torch.float4_e2m1fn_x2)
        w1_sc_c = wc_sc_cols[0].view(torch.float8_e4m3fn)
        w3_fp4_c = wc_fp4_cols[1].view(torch.float4_e2m1fn_x2)
        w3_sc_c = wc_sc_cols[1].view(torch.float8_e4m3fn)

        # Regular TK uses scalar global SGs. Keep existing float32 SG tensors
        # on the hot path instead of launching tiny copy kernels into scratch.
        if sg_cat.dtype == torch.float32:
            _sg_f32 = sg_cat
        else:
            _sg_key = (id(sg_cat), dev.index)
            if not hasattr(_ffn_bwd_graphed, '_sg_f32') or \
               _ffn_bwd_graphed._sg_f32_key != _sg_key:
                _ffn_bwd_graphed._sg_f32 = sg_cat.float()
                _ffn_bwd_graphed._sg_f32_key = _sg_key
            else:
                _ffn_bwd_graphed._sg_f32.copy_(sg_cat)
            _sg_f32 = _ffn_bwd_graphed._sg_f32
        w1_sg = _sg_f32[0:1]
        w3_sg = _sg_f32[1:2]

        if use_safe_split_producer:
            D_list = [bufs['d1'], bufs['d2']]
            _tk_stage_trace('ffn_bwd_sub', 'split_dgrad_start', debug_name)
            if use_tk_debug_ffn_bf16_split_dgrad():
                if w1_bf16 is None or w3_bf16 is None:
                    raise RuntimeError(
                        "BF16 split dgrad diagnostic requires ctx.w1_bf16 and ctx.w3_bf16"
                    )
                D_list[0].copy_(
                    torch.matmul(dh1.float(), w1_bf16.float()).to(torch.bfloat16)
                )
                D_list[1].copy_(
                    torch.matmul(dh3_out.float(), w3_bf16.float()).to(torch.bfloat16)
                )
            elif M < 256:
                tmp0 = torch.empty(target_rows, K, dtype=torch.bfloat16, device=dev)
                tmp1 = torch.empty(target_rows, K, dtype=torch.bfloat16, device=dev)
                tk_dispatch_gemm(
                    _tk_mod,
                    row_fp4s[0], row_scs[0], row_sgs[0],
                    w1_fp4_c, w1_sc_c, w1_sg, tmp0
                )
                tk_dispatch_gemm(
                    _tk_mod,
                    row_fp4s[1], row_scs[1], row_sgs[1],
                    w3_fp4_c, w3_sc_c, w3_sg, tmp1
                )
                D_list[0].copy_(tmp0[:M])
                D_list[1].copy_(tmp1[:M])
            elif (
                not use_localcta
                and use_tk_ffn_plain_batched_accum_dgrad()
                and hasattr(_tk_mod, 'nvfp4_batched_accum_gemm')
            ):
                tk_dispatch_batched_accum_gemm(
                    _tk_mod,
                    row_fp4s, row_scs, row_sgs,
                    [w1_fp4_c, w3_fp4_c], [w1_sc_c, w3_sc_c],
                    [w1_sg, w3_sg], D_list[0])
            elif use_tk_ffn_split_dgrad_eager():
                tk_dispatch_gemm(
                    _tk_mod,
                    row_fp4s[0], row_scs[0], row_sgs[0],
                    w1_fp4_c, w1_sc_c, w1_sg, D_list[0]
                )
                tk_dispatch_gemm(
                    _tk_mod,
                    row_fp4s[1], row_scs[1], row_sgs[1],
                    w3_fp4_c, w3_sc_c, w3_sg, D_list[1]
                )
            else:
                tk_dispatch_batched_gemm(
                    _tk_mod,
                    row_fp4s, row_scs, row_sgs,
                    [w1_fp4_c, w3_fp4_c], [w1_sc_c, w3_sc_c],
                    [w1_sg, w3_sg], D_list)

            if (
                not use_localcta
                and use_tk_ffn_plain_batched_accum_dgrad()
                and hasattr(_tk_mod, 'nvfp4_batched_accum_gemm')
                and M >= 256
            ):
                d_normed = D_list[0]
            elif can_fuse_ffn_sum_rms and M >= 256:
                d_normed = (D_list[0], D_list[1])
            else:
                d_normed = D_list[0].add_(D_list[1])
            _tk_stage_trace('ffn_bwd_sub', 'split_dgrad_done', debug_name)
            _tk_ffn_debug_sync_checkpoint('ffn_split_dgrad')
            _ffn_timing_mark('split_dgrad_done')
            _tk_ffn_debug_assert_finite(
                "split_dgrad_output",
                debug_name,
                (
                    ("d0", D_list[0]),
                    (
                        "d1",
                        None
                        if (
                            not use_localcta
                            and use_tk_ffn_plain_batched_accum_dgrad()
                            and hasattr(_tk_mod, 'nvfp4_batched_accum_gemm')
                            and M >= 256
                        )
                        else D_list[1],
                    ),
                    ("d_normed", d_normed if torch.is_tensor(d_normed) else None),
                ),
            )
            _debug_check_finite('ffn_bwd.safe.d_normed', d_normed)
            if _ffn_capture_path():
                _append_ffn_capture({
                    "event": "ffn_split2_dgrad",
                    "debug_name": debug_name,
                    "M": int(M),
                    "K": int(K),
                    "H": int(H),
                    "d0": _tensor_capture_stats(D_list[0]),
                    "d1": _tensor_capture_stats(D_list[1]),
                        "d_normed": _tensor_capture_stats(d_normed),
                    })
            _launch_regular_ffn_rms_async(d_normed)
            _tk_stage_trace('ffn_bwd_sub', 'split_wgrad_start', debug_name)
            if M < 256:
                normed_buf = _as_contiguous_bf16(input).clone()
                if inv_rms.dim() == 1:
                    normed_buf.mul_(inv_rms.view(M, 1).to(normed_buf.dtype))
                else:
                    normed_buf.mul_(inv_rms.view(M, 1).to(normed_buf.dtype))
                normed_buf.mul_(norm_weight.view(1, K).to(normed_buf.dtype))
                normed_pad = _pad_rows_bf16(normed_buf, target_rows)
                x_nvfp4_pad = _fast_quantize(
                    normed_pad,
                    None,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                dh1_wgrad_pad = _pad_rows_bf16(dh1, target_rows)
                dh3_wgrad_pad = _pad_rows_bf16(dh3_out, target_rows)
                dh1_nvfp4_pad = _fast_quantize(
                    dh1_wgrad_pad,
                    None,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                dh3_nvfp4_pad = _fast_quantize(
                    dh3_wgrad_pad,
                    None,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                grad_w1 = tk_wgrad_gemm(x_nvfp4_pad, dh1_nvfp4_pad, use_localcta=False)
                grad_w3 = tk_wgrad_gemm(x_nvfp4_pad, dh3_nvfp4_pad, use_localcta=False)
            else:
                grad_w1, grad_w3 = tk_split_wgrad_gemm(
                    (col_fp4s, col_scs, col_sgs), x_nvfp4,
                    use_localcta=use_localcta,
                    owner_key=debug_name,
                )
            _tk_stage_trace('ffn_bwd_sub', 'split_wgrad_done', debug_name)
            _tk_ffn_debug_sync_checkpoint('ffn_split_wgrad')
            _ffn_timing_mark('split_wgrad_done')
            _debug_check_finite('ffn_bwd.safe.grad_w1', grad_w1)
            _debug_check_finite('ffn_bwd.safe.grad_w3', grad_w3)
            if _ffn_capture_path():
                _append_ffn_capture({
                    "event": "ffn_split2_wgrad",
                    "debug_name": debug_name,
                    "M": int(M),
                    "K": int(K),
                    "H": int(H),
                    "grad_w1": _tensor_capture_stats(grad_w1),
                    "grad_w3": _tensor_capture_stats(grad_w3),
                })
        else:
            _fsg = [fused[4], fused[10]]

            if hasattr(_tk_mod, 'nvfp4_split2_dgrad_onepass_gemm'):
                d_normed = bufs['d1']
                _tk_stage_trace('ffn_bwd_sub', 'split_dgrad_start', debug_name)
                _tk_mod.nvfp4_split2_dgrad_onepass_gemm(
                    [fused[0], fused[6]], [fused[1], fused[7]],
                    _fsg,
                    [w1_fp4_c, w3_fp4_c], [w1_sc_c, w3_sc_c],
                    [w1_sg, w3_sg], d_normed, -1)
                _tk_stage_trace('ffn_bwd_sub', 'split_dgrad_done', debug_name)
            else:
                D_list = [bufs['d1'], bufs['d2']]
                _tk_stage_trace('ffn_bwd_sub', 'split_dgrad_start', debug_name)
                if (
                    not use_localcta
                    and use_tk_ffn_plain_batched_accum_dgrad()
                    and hasattr(_tk_mod, 'nvfp4_batched_accum_gemm')
                ):
                    tk_dispatch_batched_accum_gemm(
                        _tk_mod,
                        [fused[0], fused[6]], [fused[1], fused[7]],
                        _fsg,
                        [w1_fp4_c, w3_fp4_c], [w1_sc_c, w3_sc_c],
                        [w1_sg, w3_sg], D_list[0])
                elif use_tk_ffn_split_dgrad_eager():
                    tk_dispatch_gemm(
                        _tk_mod,
                        fused[0], fused[1], _fsg[0],
                        w1_fp4_c, w1_sc_c, w1_sg, D_list[0]
                    )
                    tk_dispatch_gemm(
                        _tk_mod,
                        fused[6], fused[7], _fsg[1],
                        w3_fp4_c, w3_sc_c, w3_sg, D_list[1]
                    )
                else:
                    tk_dispatch_batched_gemm(
                        _tk_mod,
                        [fused[0], fused[6]], [fused[1], fused[7]],
                        _fsg,
                        [w1_fp4_c, w3_fp4_c], [w1_sc_c, w3_sc_c],
                        [w1_sg, w3_sg], D_list)

                # ─── 6. d_normed = D_list[0] + D_list[1] (in-place) ───
                if (
                    not use_localcta
                    and use_tk_ffn_plain_batched_accum_dgrad()
                    and hasattr(_tk_mod, 'nvfp4_batched_accum_gemm')
                ):
                    d_normed = D_list[0]
                elif can_fuse_ffn_sum_rms:
                    d_normed = (D_list[0], D_list[1])
                else:
                    d_normed = D_list[0].add_(D_list[1])
                _tk_stage_trace('ffn_bwd_sub', 'split_dgrad_done', debug_name)
            _ffn_timing_mark('split_dgrad_done')
            _tk_ffn_debug_sync_checkpoint('ffn_split_dgrad')
            _tk_ffn_debug_assert_finite(
                "split_dgrad_output",
                debug_name,
                (
                    ("d0", d_normed[0] if isinstance(d_normed, (tuple, list)) else d_normed),
                    ("d1", d_normed[1] if isinstance(d_normed, (tuple, list)) else None),
                ),
            )
            _debug_check_finite('ffn_bwd.d_normed', d_normed)

            # ─── 7. W1/W3 wgrad (split eager helper, avoids cat + sg expansion) ───
            _launch_regular_ffn_rms_async(d_normed)
            _tk_stage_trace('ffn_bwd_sub', 'split_wgrad_start', debug_name)
            if use_localcta and use_tk_ffn_localcta_tk_quant_contract():
                grad_w13 = tk_grouped_wgrad_gemm(
                    ([fused[2], fused[8]], [fused[3], fused[9]], [fused[4], fused[10]]),
                    x_nvfp4,
                    N_dims_13,
                )
                grad_w1, grad_w3 = grad_w13[:H, :], grad_w13[H:, :]
            else:
                grad_w1, grad_w3 = tk_split_wgrad_gemm(
                    ([fused[2], fused[8]], [fused[3], fused[9]], [fused[4], fused[10]]),
                    x_nvfp4,
                    use_localcta=use_localcta,
                    owner_key=debug_name,
                )
            _tk_stage_trace('ffn_bwd_sub', 'split_wgrad_done', debug_name)
            _tk_ffn_debug_sync_checkpoint('ffn_split_wgrad')
            _ffn_timing_mark('split_wgrad_done')
            _debug_check_finite('ffn_bwd.grad_w1', grad_w1)
            _debug_check_finite('ffn_bwd.grad_w3', grad_w3)
            if _ffn_capture_path():
                _append_ffn_capture({
                    "event": "ffn_split2_wgrad",
                    "debug_name": debug_name,
                    "M": int(M),
                    "K": int(K),
                    "H": int(H),
                    "grad_w1": _tensor_capture_stats(grad_w1),
                    "grad_w3": _tensor_capture_stats(grad_w3),
                })

        # ─── 8. RMSNorm backward ───
        grad_input, dgamma = _finish_regular_ffn_rms(d_normed)
        _tk_ffn_debug_assert_finite(
            "rms_outputs",
            debug_name,
            (("grad_input", grad_input), ("dgamma", dgamma)),
        )
        _debug_check_finite('ffn_bwd.grad_input', grad_input)
        _debug_check_finite('ffn_bwd.dgamma', dgamma)
        if _ffn_capture_path():
            _append_ffn_capture({
                "event": "ffn_rmsnorm_bwd",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "d_normed": _tensor_capture_stats(d_normed),
                "inv_rms": _tensor_capture_stats(inv_rms),
                "grad_input": _tensor_capture_stats(grad_input),
                "dgamma": _tensor_capture_stats(dgamma),
            })

        # grad_w2 is backed by the shape-global FFN scratch buffer. Return
        # owned storage so later layer backward calls cannot overwrite it while
        # autograd is still accumulating parameter gradients.
        grad_w2_materialized = grad_w2.clone(memory_format=torch.contiguous_format)
        _ffn_timing_emit('regular_tk_bwd')
        return grad_input, grad_w1, grad_w3, grad_w2_materialized, _as_param_grad_dtype(dgamma, norm_weight), None

    # ── CUDA graph path: first call setup ──
    sb = {
        'dY': grad_output.clone(),
        'input': input.clone(),
        'norm_weight': norm_weight.clone(),
        'inv_rms': inv_rms.clone(),
        'h13': h13.clone(),
        'h1_raw': h1_raw.clone(),
        'h3': h3.clone(),
        'sg_cat': sg_cat.clone(),
        'w2_fp4_c': w2_nvfp4._tk_col[0].clone(),
        'w2_sc_c': w2_nvfp4._tk_col[1].clone(),
        'w2_sg_c': w2_nvfp4._tk_col[2].clone(),
        'x_fp4_c': x_nvfp4._tk_col[0].clone(),
        'x_sc_c': x_nvfp4._tk_col[1].clone(),
        'x_sg_c': x_nvfp4._tk_col[2].clone(),
        'h_fp4_c': h_nvfp4._tk_col[0].clone(),
        'h_sc_c': h_nvfp4._tk_col[1].clone(),
        'h_sg_c': h_nvfp4._tk_col[2].clone(),
    }
    for i in range(n_groups):
        sb[f'wc_fp4_c_{i}'] = wc_fp4_cols[i].clone()
        sb[f'wc_sc_c_{i}'] = wc_sc_cols[i].clone()

    # Pre-allocate dY quant buffers (single group [K])
    dy_alloc = tkq.tk_group_quantize_dim1_alloc(sb['dY'].to(torch.bfloat16), [K])
    (dy_fp4_row, dy_fp4_col, dy_sg_buf, dy_amax, dy_sync, dy_psync, dy_tma,
     dy_sc_row, dy_fp4_col_list, dy_sc_col, dy_tma_host) = dy_alloc

    # Pre-allocate dh1+dh3 quant buffers (2 groups [H, H])
    dh_cat_dummy = torch.empty(M, 2 * H, dtype=torch.bfloat16, device=grad_output.device)
    dh_alloc = tkq.tk_group_quantize_dim1_alloc(dh_cat_dummy, [H, H])
    (dh_fp4_row, dh_fp4_col, dh_sg_buf, dh_amax, dh_sync, dh_psync, dh_tma,
     dh_sc_row, dh_fp4_col_list, dh_sc_col, dh_tma_host) = dh_alloc

    use_graph_direct_split2 = False
    use_graph_fused_split2_quant = (
        use_graph_direct_split2
        and hasattr(tkq._mod, "tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace")
    )
    use_graph_fused_split2_quant = False
    use_graph_two_single_row_split2_quant = (
        not use_graph_direct_split2
        and use_localcta
        and h1_raw is not None and h1_raw.numel() != 0
        and h3 is not None and h3.numel() != 0
        and hasattr(tkq, "_mod")
        and hasattr(tkq._mod, "tk_localcta_quantize_for_gemm_prepared_alloc")
        and hasattr(tkq._mod, "tk_localcta_quantize_for_gemm_prepared_launch")
        and hasattr(tkq._mod, "tk_localcta_quantize_col_only_prepared_launch_inplace")
        and hasattr(tk, "nvfp4_split2_dgrad_onepass_gemm")
    )
    use_graph_row_only_split2_quant = False
    split2_q_graph = None
    if use_graph_direct_split2 or use_graph_two_single_row_split2_quant:
        split2_q_graph = tkq._mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
            M, H, H, grad_output.device
        )
    single_row_q_graph = None
    if use_graph_two_single_row_split2_quant:
        single_row_q_graph = (
            tkq._mod.tk_localcta_quantize_for_gemm_prepared_alloc(M, H, True, grad_output.device),
            tkq._mod.tk_localcta_quantize_for_gemm_prepared_alloc(M, H, True, grad_output.device),
        )

    # Pre-warm caches
    _get_sg_tile_indices(N_dims_13, grad_output.device)
    _get_wgrad_buf(K, sum(N_dims_13), grad_output.device)

    static_x = type('_S', (), {'_tk_col': (sb['x_fp4_c'], sb['x_sc_c'], sb['x_sg_c'])})()

    # Pre-allocate per-batch dgrad output buffers (strided GEMM outputs)
    D_list = [torch.empty(M, K, dtype=torch.bfloat16, device=grad_output.device) for _ in range(n_groups)]

    # Pre-allocate ALL intermediate buffers OUTSIDE graph capture.
    # Graph-pool allocated tensors (from C++ functions like fused_silu_deriv,
    # fused_rmsnorm_backward) must NOT be passed directly to TMA-dependent
    # kernels or returned as graph outputs — their addresses may not replay
    # correctly.  Copy them to these stable buffers instead.
    _dh_buf = torch.empty(M, H, dtype=torch.bfloat16, device=grad_output.device)
    _grad_w2_buf = torch.empty(K, H, dtype=torch.bfloat16, device=grad_output.device)
    _dh_cat_buf = torch.empty(M, 2 * H, dtype=torch.bfloat16, device=grad_output.device)
    _dh1_buf = torch.empty(M, H, dtype=torch.bfloat16, device=grad_output.device)
    _dh3_buf = torch.empty(M, H, dtype=torch.bfloat16, device=grad_output.device)
    _dh1_row_sg_buf = torch.empty(M // 128, H // 128, dtype=torch.float32, device=grad_output.device)
    _dh3_row_sg_buf = torch.empty(M // 128, H // 128, dtype=torch.float32, device=grad_output.device)
    # Pre-allocated outputs for fused_silu_deriv (avoids graph-pool allocs)
    _silu_amax1 = torch.zeros(1, dtype=torch.float32, device=grad_output.device)
    _silu_amax2 = torch.zeros(1, dtype=torch.float32, device=grad_output.device)
    _d_normed_buf = torch.empty(M, K, dtype=torch.bfloat16, device=grad_output.device)
    _grad_input_buf = torch.empty(M, K, dtype=torch.bfloat16, device=grad_output.device)
    _dgamma_buf = torch.empty(K, dtype=torch.float32, device=grad_output.device)
    # Pre-allocated output for fused_rmsnorm_backward_out (avoids graph-pool allocs)
    _rmsnorm_grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=grad_output.device)
    _rmsnorm_dgamma = torch.zeros(K, dtype=torch.float32, device=grad_output.device)
    _rmsnorm_dgamma_partials = torch.empty((M + 255) // 256, K, dtype=torch.float32, device=grad_output.device)
    # Pre-allocate for torch::cat outputs from dim1_launch (graph-pool tensors).
    # sc_col_cat = cat of sc_col_allocs along dim 0: shape (sum(N_g/128), M/64, 512) u8
    _dh_sc_col_cat_buf = torch.empty_like(
        torch.cat([sc.view(torch.float8_e4m3fn) for sc in dh_sc_col], dim=0))
    # fp4_col_full = cat of fp4_col_allocs along dim 0: shape (N_total, M//2) fp4
    # Used directly as dy_fp4_cat in wgrad GEMM → TMA descriptor creation
    N_total_13 = sum(N_dims_13)  # = 2*H
    _dh_fp4_col_full_buf = torch.empty(N_total_13, M // 2, dtype=torch.uint8, device=grad_output.device)

    # Pre-allocate buffers for every .to() / .contiguous() call inside graph
    # that creates a graph-pool tensor fed to GEMM TMA descriptor creation.
    # Without these, graph-pool addresses alternate on replay → wrong results.
    #
    # dY dgrad GEMM: dy_fp4_rows[0] is (M, K//2) fp4, dy_sc_rows[0] is (M//128, K//64, 512) fp8
    # fp4 doesn't support .copy_() so we use uint8 and .view() at call sites
    _dy_fp4_row0_contig = torch.empty(M, K // 2, dtype=torch.uint8, device=grad_output.device)
    _dy_sc_row0_contig = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device=grad_output.device)
    # dy_sg_out[0:1].to(float32) — used by both dgrad GEMMs
    _dy_sg_f32 = torch.empty(1, dtype=torch.float32, device=grad_output.device)
    # dY wgrad GEMM: dy_fp4_cols[0] is (K, M//2) fp4, dy_sc_cols[0] is (K//128, M//64, 512) fp8
    _dy_fp4_col0_contig = torch.empty(K, M // 2, dtype=torch.uint8, device=grad_output.device)
    _dy_sc_col0_contig = torch.empty(K // 128, M // 64, 512, dtype=torch.float8_e4m3fn, device=grad_output.device)
    # Batched dgrad: dh_sg_out[i].to(float32), sb['sg_cat'][i].to(float32)
    _dh_sg_f32 = [torch.empty(1, dtype=torch.float32, device=grad_output.device) for _ in range(n_groups)]
    _b_sg_f32 = [torch.empty(1, dtype=torch.float32, device=grad_output.device) for _ in range(n_groups)]
    # Batched dgrad: dh_sc_rows[i] is (M//128, H//64, 512) fp8
    _dh_sc_row_contig = [torch.empty(M // 128, H // 64, 512, dtype=torch.float8_e4m3fn, device=grad_output.device) for _ in range(n_groups)]
    # RMSNorm backward: sb['input'].contiguous().to(bf16), sb['norm_weight'].to(bf16)
    _input_bf16 = torch.empty(M, K, dtype=torch.bfloat16, device=grad_output.device)
    _norm_weight_bf16 = torch.empty(K, dtype=torch.bfloat16, device=grad_output.device)
    # Return: _dgamma_buf.to(norm_weight.dtype)
    _dgamma_cast = torch.empty(K, dtype=norm_weight.dtype, device=grad_output.device)
    # dY.to(bf16) — input to dim1_launch which creates TMA descriptors
    _dY_bf16 = torch.empty(M, K, dtype=torch.bfloat16, device=grad_output.device)
    # Wgrad b_sg_per_tile: .to(float32)[sg_idx] creates 2 graph-pool tensors
    _wgrad_b_sg_buf = torch.empty(sum(N_dims_13) // 256, dtype=torch.float32, device=grad_output.device)
    # Intermediate float32 conversion of sg_per_group (avoids graph-pool alloc from .to())
    _wgrad_sg_f32_buf = torch.empty(n_groups, dtype=torch.float32, device=grad_output.device)

    def _run():
        # ---- Fully graph-safe _run(): ZERO graph-pool allocations.
        # Every tensor is pre-allocated outside graph capture scope.
        # Previous attempts using _out variants individually failed because
        # the MIX of pre-allocated and graph-pool tensors caused aliasing.
        # With zero graph-pool allocs, the graph is fully deterministic. ----

        # 1. Quantize dY — single group [K]
        _dY_bf16.copy_(sb['dY'])                      # sb['dY'].to(bf16)
        dy_qr = tkq.tk_group_quantize_dim1_launch(
            _dY_bf16, [K],
            dy_fp4_row, dy_fp4_col, dy_sg_buf, dy_amax, dy_sync, dy_psync,
            dy_tma_host, dy_tma, dy_sc_row, dy_fp4_col_list, dy_sc_col)
        dy_fp4_rows, dy_sc_rows, dy_sg_out, \
            dy_fp4_cols, dy_sc_cols, \
            dy_a_fp4, dy_a_sc, dy_fp4_col_full, dy_sc_col_cat = dy_qr

        # 2. W2 dgrad: dh = dY @ W2^T → pre-allocated _dh_buf
        _dy_fp4_row0_contig.copy_(dy_fp4_rows[0].view(torch.uint8))
        _dy_sc_row0_contig.copy_(dy_sc_rows[0])
        _dy_sg_f32.copy_(dy_sg_out[0:1])
        tk_dispatch_batched_gemm(
            tk,
            [_dy_fp4_row0_contig.view(torch.float4_e2m1fn_x2)],
            [_dy_sc_row0_contig],
            [_dy_sg_f32],
            [sb['w2_fp4_c']], [sb['w2_sc_c']], [sb['w2_sg_c']],
            [_dh_buf])

        # 3. W2 wgrad: grad_w2 = h^T @ dY → pre-allocated _grad_w2_buf
        _dy_fp4_col0_contig.copy_(dy_fp4_cols[0].view(torch.uint8))
        _dy_sc_col0_contig.copy_(dy_sc_cols[0])
        tk_dispatch_batched_gemm(
            tk,
            [_dy_fp4_col0_contig.view(torch.float4_e2m1fn_x2)],
            [_dy_sc_col0_contig],
            [_dy_sg_f32],
            [sb['h_fp4_c']], [sb['h_sc_c']], [sb['h_sg_c']],
            [_grad_w2_buf])

        # 4. SiLU derivative → pre-allocated _dh_cat_buf, _silu_amax1/2
        if hasattr(te_fused, 'fused_silu_deriv_dual_mul_strided_interleaved_bf16_out_no_amax'):
            te_fused.fused_silu_deriv_dual_mul_strided_interleaved_bf16_out_no_amax(
                _dh_buf, sb['h13'], _dh_cat_buf)
        else:
            te_fused.fused_silu_deriv_dual_mul_strided_interleaved_bf16_out(
                _dh_buf, sb['h13'], _dh_cat_buf, _silu_amax1, _silu_amax2)
        dh_qr = tkq.tk_group_quantize_dim1_launch(
            _dh_cat_buf, [H, H],
            dh_fp4_row, dh_fp4_col, dh_sg_buf, dh_amax, dh_sync, dh_psync,
            dh_tma_host, dh_tma, dh_sc_row, dh_fp4_col_list, dh_sc_col)
        dh_fp4_rows, dh_sc_rows, dh_sg_out, \
            dh_fp4_cols, dh_sc_cols, \
            dh_a_fp4, dh_a_sc, dh_fp4_col_full, dh_sc_col_cat = dh_qr

        # 6. W1/W3 dgrad: d_normed = dh1@W1^T + dh3@W3^T
        #    Pre-copy all .to()/.contiguous() intermediates to stable buffers
        for i in range(n_groups):
            _dh_sg_f32[i].copy_(dh_sg_out[i:i+1])
            _b_sg_f32[i].copy_(sb['sg_cat'][i:i+1])
            _dh_sc_row_contig[i].copy_(dh_sc_rows[i])
        B_fp4_list = [sb[f'wc_fp4_c_{i}'].view(torch.float4_e2m1fn_x2) for i in range(n_groups)]
        B_sc_list = [sb[f'wc_sc_c_{i}'].view(torch.float8_e4m3fn) for i in range(n_groups)]
        A_sc_list = [_dh_sc_row_contig[i] for i in range(n_groups)]
        if hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm'):
            tk.nvfp4_split2_dgrad_onepass_gemm(
                dh_fp4_rows,
                A_sc_list,
                _dh_sg_f32,
                B_fp4_list,
                B_sc_list,
                _b_sg_f32,
                _d_normed_buf,
                -1,
            )
        else:
            a_fp4_u8 = dh_a_fp4.view(torch.uint8) if dh_a_fp4.dtype != torch.uint8 else dh_a_fp4
            col_offsets = [0, H // 2]
            col_widths = [H // 2, H // 2]
            tk.nvfp4_batched_gemm_strided(
                a_fp4_u8.view(torch.float4_e2m1fn_x2),
                A_sc_list,
                _dh_sg_f32,
                col_offsets, col_widths,
                B_fp4_list, B_sc_list, _b_sg_f32, D_list)
            torch.add(D_list[0], D_list[1], out=_d_normed_buf)

        # Wgrad GEMM
        dy_col_quant = (dh_fp4_cols, dh_sc_cols, dh_sg_out,
                        dh_fp4_col_full, dh_sc_col_cat)
        grad_w13 = tk_grouped_wgrad_gemm(dy_col_quant, static_x, N_dims_13,
                                          b_sg_buf=_wgrad_b_sg_buf, sg_f32_buf=_wgrad_sg_f32_buf)

        # RMSNorm backward → pre-allocated _rmsnorm_grad_input, _rmsnorm_dgamma
        # (padded to absorb TE kernel overrun at small M)
        _input_bf16.copy_(sb['input'])
        _norm_weight_bf16.copy_(sb['norm_weight'])
        if (
            hasattr(te_fused, "fused_rmsnorm_backward_dx_only_out")
            and hasattr(te_fused, "fused_rmsnorm_backward_dgamma_tiled_out")
        ):
            te_fused.fused_rmsnorm_backward_dx_only_out(
                _d_normed_buf, _input_bf16, _norm_weight_bf16, sb['inv_rms'],
                _rmsnorm_grad_input
            )
            te_fused.fused_rmsnorm_backward_dgamma_tiled_out(
                _d_normed_buf, _input_bf16, sb['inv_rms'],
                _rmsnorm_dgamma_partials, _rmsnorm_dgamma
            )
        else:
            te_fused.fused_rmsnorm_backward_out(
                _d_normed_buf, _input_bf16, _norm_weight_bf16, sb['inv_rms'],
                _rmsnorm_grad_input, _rmsnorm_dgamma)
        _dgamma_cast.copy_(_rmsnorm_dgamma)
        return _rmsnorm_grad_input, grad_w1_w3_split(grad_w13), _grad_w2_buf, _dgamma_cast

    def grad_w1_w3_split(gw13):
        return gw13[:H, :], gw13[H:, :]

    # Modify _run to return flat tuple
    def _run_flat():
        _dY_bf16.copy_(sb['dY'])
        dy_qr = tkq.tk_group_quantize_dim1_launch(
            _dY_bf16, [K],
            dy_fp4_row, dy_fp4_col, dy_sg_buf, dy_amax, dy_sync, dy_psync,
            dy_tma_host, dy_tma, dy_sc_row, dy_fp4_col_list, dy_sc_col,
            skip_cat=True)
        dy_fp4_rows, dy_sc_rows, dy_sg_out, \
            dy_fp4_cols, dy_sc_cols, \
            dy_a_fp4, dy_a_sc, dy_fp4_col_full, dy_sc_col_cat = dy_qr

        # Use graph-safe batched GEMM (batch=1) with pre-allocated intermediate buffers
        _dy_fp4_row0_contig.copy_(dy_fp4_rows[0].view(torch.uint8))
        _dy_sc_row0_contig.copy_(dy_sc_rows[0])
        _dy_sg_f32.copy_(dy_sg_out[0:1])
        tk_dispatch_batched_gemm(
            tk,
            [_dy_fp4_row0_contig.view(torch.float4_e2m1fn_x2)],
            [_dy_sc_row0_contig],
            [_dy_sg_f32],
            [sb['w2_fp4_c']], [sb['w2_sc_c']], [sb['w2_sg_c']], [_dh_buf])

        _dy_fp4_col0_contig.copy_(dy_fp4_cols[0].view(torch.uint8))
        _dy_sc_col0_contig.copy_(dy_sc_cols[0])
        tk_dispatch_batched_gemm(
            tk,
            [_dy_fp4_col0_contig.view(torch.float4_e2m1fn_x2)],
            [_dy_sc_col0_contig],
            [_dy_sg_f32],
            [sb['h_fp4_c']], [sb['h_sc_c']], [sb['h_sg_c']], [_grad_w2_buf])

        if use_graph_direct_split2 or use_graph_two_single_row_split2_quant:
            if use_graph_fused_split2_quant:
                tkq._mod.tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
                    _dh_buf, sb['h3'], sb['h1_raw'],
                    split2_q_graph[6], split2_q_graph[7],
                    split2_q_graph[9], split2_q_graph[10],
                    split2_q_graph[8], split2_q_graph[11],
                )
            else:
                if hasattr(te_fused, 'fused_silu_deriv_dual_mul_bf16_out_no_amax'):
                    te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                        _dh_buf, sb['h3'], sb['h1_raw'], _dh1_buf, _dh3_buf
                    )
                else:
                    te_fused.fused_silu_deriv_dual_mul_bf16_out(
                        _dh_buf, sb['h3'], sb['h1_raw'],
                        _dh1_buf, _dh3_buf, _silu_amax1, _silu_amax2
                    )
                if use_graph_two_single_row_split2_quant:
                    tkq._mod.tk_localcta_quantize_for_gemm_prepared_launch(
                        _dh1_buf, True, True,
                        single_row_q_graph[0][0], single_row_q_graph[0][1],
                        single_row_q_graph[0][2], single_row_q_graph[0][3],
                        single_row_q_graph[0][4], single_row_q_graph[0][5],
                    )
                    tkq._mod.tk_localcta_quantize_for_gemm_prepared_launch(
                        _dh3_buf, True, True,
                        single_row_q_graph[1][0], single_row_q_graph[1][1],
                        single_row_q_graph[1][2], single_row_q_graph[1][3],
                        single_row_q_graph[1][4], single_row_q_graph[1][5],
                    )
                    split2_q_graph[3][0].copy_(single_row_q_graph[0][2])
                    split2_q_graph[3][1].copy_(single_row_q_graph[1][2])
                    split2_q_graph[4][0].copy_(single_row_q_graph[0][3])
                    split2_q_graph[4][1].copy_(single_row_q_graph[1][3])
                    split2_q_graph[5][0].copy_(single_row_q_graph[0][5])
                    split2_q_graph[5][1].copy_(single_row_q_graph[1][5])
                elif use_graph_row_only_split2_quant:
                    tkq._mod.tk_localcta_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace(
                        _dh1_buf, _dh3_buf,
                        split2_q_graph[6], split2_q_graph[7], split2_q_graph[8],
                    )
                    _dh1_row_sg_buf.copy_(split2_q_graph[2][0])
                    _dh3_row_sg_buf.copy_(split2_q_graph[2][1])
                    tkq._mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                        _dh1_buf,
                        _dh1_row_sg_buf,
                        split2_q_graph[3][0],
                        split2_q_graph[4][0],
                        split2_q_graph[5][0],
                    )
                    tkq._mod.tk_localcta_quantize_col_only_prepared_launch_inplace(
                        _dh3_buf,
                        _dh3_row_sg_buf,
                        split2_q_graph[3][1],
                        split2_q_graph[4][1],
                        split2_q_graph[5][1],
                    )
                else:
                    tkq._mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
                        _dh1_buf, _dh3_buf,
                        split2_q_graph[6], split2_q_graph[7],
                        split2_q_graph[9], split2_q_graph[10],
                        split2_q_graph[8], split2_q_graph[11],
                    )
            if use_graph_two_single_row_split2_quant:
                dh_fp4_rows = [single_row_q_graph[0][0], single_row_q_graph[1][0]]
                dh_sc_rows = [single_row_q_graph[0][1], single_row_q_graph[1][1]]
                dh_sg_out = [single_row_q_graph[0][4], single_row_q_graph[1][4]]
            else:
                dh_fp4_rows = split2_q_graph[0]
                dh_sc_rows = split2_q_graph[1]
                dh_sg_out = split2_q_graph[2]
            dh_fp4_cols = split2_q_graph[3]
            dh_sc_cols = split2_q_graph[4]
            dh_fp4_col_full = split2_q_graph[9]
            dh_sc_col_cat = split2_q_graph[10]
            dh_sg_col_cat = split2_q_graph[11]
        else:
            # Graph-safe: write to pre-allocated output buffers (no graph-pool allocs)
            if hasattr(te_fused, 'fused_silu_deriv_dual_mul_strided_interleaved_bf16_out_no_amax'):
                te_fused.fused_silu_deriv_dual_mul_strided_interleaved_bf16_out_no_amax(
                    _dh_buf, sb['h13'], _dh_cat_buf)
            else:
                te_fused.fused_silu_deriv_dual_mul_strided_interleaved_bf16_out(
                    _dh_buf, sb['h13'], _dh_cat_buf, _silu_amax1, _silu_amax2)

            dh_qr = tkq.tk_group_quantize_dim1_launch(
                _dh_cat_buf, [H, H],
                dh_fp4_row, dh_fp4_col, dh_sg_buf, dh_amax, dh_sync, dh_psync,
                dh_tma_host, dh_tma, dh_sc_row, dh_fp4_col_list, dh_sc_col,
                skip_cat=True)
            dh_fp4_rows, dh_sc_rows, dh_sg_out, \
                dh_fp4_cols, dh_sc_cols, \
                _dh_a_fp4, _dh_a_sc, _dh_fp4_col_full_unused, _dh_sc_col_cat_unused = dh_qr

            # Manual cat of sc_col into pre-allocated buffer (skip_cat=True)
            sc_offset = 0
            for i in range(n_groups):
                sc_i_u8 = dh_sc_col[i]  # pre-allocated, written by kernel
                n_rows = sc_i_u8.size(0)
                _dh_sc_col_cat_buf.view(torch.uint8)[sc_offset:sc_offset + n_rows].copy_(sc_i_u8)
                sc_offset += n_rows
            # fp4_col_full is a view of the pre-allocated dh_fp4_col — just copy
            _dh_fp4_col_full_buf.copy_(dh_fp4_col.view(torch.uint8))
            dh_fp4_col_full = _dh_fp4_col_full_buf.view(torch.float4_e2m1fn_x2)
            dh_sc_col_cat = _dh_sc_col_cat_buf
            dh_sg_col_cat = None

        # Pre-copy only the per-group B SG tensors for dgrad GEMM. LocalCTA's
        # split2 one-pass backends consume prepared A scales directly, so the
        # row-SG payload can stay in its native tiled shape.
        for i in range(n_groups):
            _b_sg_f32[i].copy_(sb['sg_cat'][i:i+1])
            if not (use_graph_direct_split2 or use_graph_two_single_row_split2_quant):
                _dh_sg_f32[i].copy_(dh_sg_out[i:i+1])
                _dh_sc_row_contig[i].copy_(dh_sc_rows[i])
        A_sg_list = dh_sg_out if (use_graph_direct_split2 or use_graph_two_single_row_split2_quant) else _dh_sg_f32
        B_fp4_list = [sb[f'wc_fp4_c_{i}'].view(torch.float4_e2m1fn_x2) for i in range(n_groups)]
        B_sc_list = [sb[f'wc_sc_c_{i}'].view(torch.float8_e4m3fn) for i in range(n_groups)]
        B_sg_list = _b_sg_f32
        A_sc_list = dh_sc_rows if (use_graph_direct_split2 or use_graph_two_single_row_split2_quant) else [_dh_sc_row_contig[i] for i in range(n_groups)]
        if use_graph_two_single_row_split2_quant:
            _d_normed_buf.zero_()
            tk.nvfp4_split2_dgrad_onepass_gemm(
                dh_fp4_rows,
                dh_sc_rows,
                A_sg_list,
                B_fp4_list,
                B_sc_list,
                B_sg_list,
                _d_normed_buf,
                -1,
            )
        elif use_graph_direct_split2:
            _d_normed_buf.zero_()
            tk.nvfp4_split2_dgrad_strided_onepass_gemm(
                split2_q_graph[6],
                A_sc_list,
                [0, H // 2],
                [H // 2, H // 2],
                B_fp4_list,
                B_sc_list,
                B_sg_list,
                _d_normed_buf,
                -1,
            )
        elif hasattr(tk, 'nvfp4_split2_dgrad_onepass_gemm'):
            for i in range(n_groups):
                _dh_sc_row_contig[i].copy_(dh_sc_rows[i])
            _d_normed_buf.zero_()
            tk.nvfp4_split2_dgrad_onepass_gemm(
                dh_fp4_rows,
                [_dh_sc_row_contig[i] for i in range(n_groups)],
                A_sg_list,
                B_fp4_list,
                B_sc_list,
                B_sg_list,
                _d_normed_buf,
                -1,
            )
        else:
            a_fp4_u8 = dh_a_fp4.view(torch.uint8) if dh_a_fp4.dtype != torch.uint8 else dh_a_fp4
            col_offsets = [0, H // 2]
            col_widths = [H // 2, H // 2]
            tk.nvfp4_batched_gemm_strided(
                a_fp4_u8.view(torch.float4_e2m1fn_x2),
                A_sc_list, A_sg_list,
                col_offsets, col_widths,
                B_fp4_list, B_sc_list, B_sg_list, D_list)
            # In-place add to pre-allocated buffer (avoids graph pool allocation)
            torch.add(D_list[0], D_list[1], out=_d_normed_buf)

        # Wgrad GEMM
        if use_graph_direct_split2 or use_graph_two_single_row_split2_quant:
            dy_col_quant = (
                dh_fp4_cols, dh_sc_cols, split2_q_graph[5],
                dh_fp4_col_full, dh_sc_col_cat, _grouped_wgrad_sg_group(split2_q_graph[5])
            )
        else:
            dy_col_quant = (
                dh_fp4_cols, dh_sc_cols, dh_sg_out,
                _dh_fp4_col_full_buf.view(torch.float4_e2m1fn_x2), _dh_sc_col_cat_buf,
                _grouped_wgrad_sg_group(dh_sg_out),
            )
        grad_w13 = tk_grouped_wgrad_gemm(dy_col_quant, static_x, N_dims_13,
                                          b_sg_buf=_wgrad_b_sg_buf, sg_f32_buf=_wgrad_sg_f32_buf)
        # Graph-safe rmsnorm backward: write to pre-allocated output buffers
        _input_bf16.copy_(sb['input'])
        _norm_weight_bf16.copy_(sb['norm_weight'])
        if (
            hasattr(te_fused, "fused_rmsnorm_backward_dx_only_out")
            and hasattr(te_fused, "fused_rmsnorm_backward_dgamma_tiled_out")
        ):
            te_fused.fused_rmsnorm_backward_dx_only_out(
                _d_normed_buf, _input_bf16, _norm_weight_bf16, sb['inv_rms'],
                _rmsnorm_grad_input
            )
            te_fused.fused_rmsnorm_backward_dgamma_tiled_out(
                _d_normed_buf, _input_bf16, sb['inv_rms'],
                _rmsnorm_dgamma_partials, _rmsnorm_dgamma
            )
        else:
            te_fused.fused_rmsnorm_backward_out(
                _d_normed_buf, _input_bf16, _norm_weight_bf16, sb['inv_rms'],
                _rmsnorm_grad_input, _rmsnorm_dgamma)
        _grad_input_buf.copy_(_rmsnorm_grad_input)
        _dgamma_buf.copy_(_rmsnorm_dgamma)
        _dgamma_cast.copy_(_dgamma_buf)
        return _grad_input_buf, grad_w13[:H, :], grad_w13[H:, :], _grad_w2_buf, _dgamma_cast

    # Warmup — sync between iterations to drain PDL chain at large M
    torch.cuda.synchronize()
    for _ in range(10):
        _run_flat()
        torch.cuda.synchronize()

    # Capture using _run() — hybrid approach:
    # - torch.empty() for GEMM outputs (safe at all M, graph-pool managed)
    # - _out silu variant → pre-allocated _dh_cat_buf (no C++ internal alloc)
    # - pre-copied .to()/.contiguous() via stable buffers
    # - regular fused_rmsnorm_backward (internal allocs are OK for returned tensors)
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        gi, gw1, gw3, gw2, gnw = _run_flat()
    sb['grad_input'] = gi
    sb['grad_w1'] = gw1
    sb['grad_w3'] = gw3
    sb['grad_w2'] = gw2
    sb['grad_nw'] = gnw
    _ffn_bwd_graph_cache[key] = (graph, sb)
    _ffn_sb_cache[key] = sb  # Export sb so forward can write directly to it

    # First call: copy real data to sb and replay.
    # We DISCARD the capture-time result (it can be corrupted at small M
    # due to graph-pool allocator interactions during capture).
    # Replays are correct and deterministic at all M.
    sb['dY'].copy_(grad_output)
    sb['input'].copy_(input)
    sb['norm_weight'].copy_(norm_weight)
    sb['inv_rms'].copy_(inv_rms)
    sb['h13'].copy_(h13)
    sb['h1_raw'].copy_(h1_raw)
    sb['h3'].copy_(h3)
    sb['sg_cat'].copy_(sg_cat)
    sb['w2_fp4_c'].copy_(w2_nvfp4._tk_col[0])
    sb['w2_sc_c'].copy_(w2_nvfp4._tk_col[1])
    sb['w2_sg_c'].copy_(w2_nvfp4._tk_col[2])
    sb['x_fp4_c'].copy_(x_nvfp4._tk_col[0])
    sb['x_sc_c'].copy_(x_nvfp4._tk_col[1])
    sb['x_sg_c'].copy_(x_nvfp4._tk_col[2])
    sb['h_fp4_c'].copy_(h_nvfp4._tk_col[0])
    sb['h_sc_c'].copy_(h_nvfp4._tk_col[1])
    sb['h_sg_c'].copy_(h_nvfp4._tk_col[2])
    for i in range(n_groups):
        sb[f'wc_fp4_c_{i}'].copy_(wc_fp4_cols[i])
        sb[f'wc_sc_c_{i}'].copy_(wc_sc_cols[i])
    torch.cuda.synchronize()
    graph.replay()
    return (sb['grad_input'].clone(), sb['grad_w1'].clone(),
            sb['grad_w3'].clone(), sb['grad_w2'].clone(),
            sb['grad_nw'].clone())


class _FusedFFNFunctionV2_TK(torch.autograd.Function):
    @staticmethod
    @v5_ffn_quant_scope()
    def forward(
        ctx,
        input: torch.Tensor,           # (M, K) bf16 — raw pre-norm input
        w1_weight: torch.Tensor,        # (H, K) bf16 — gate projection
        w3_weight: torch.Tensor,        # (H, K) bf16 — up projection
        w2_weight: torch.Tensor,        # (K, H) bf16 — down projection
        norm_weight: torch.Tensor,      # (K,) bf16 — RMSNorm gamma
        epsilon: float,
        # TE quantizers
        w1_weight_quantizer: NVFP4Quantizer,
        w3_input_quantizer: NVFP4Quantizer,
        w3_weight_quantizer: NVFP4Quantizer,
        w2_input_quantizer: NVFP4Quantizer,
        w2_weight_quantizer: NVFP4Quantizer,
        grad_quantizer_w1: NVFP4Quantizer,
        grad_quantizer_w3: NVFP4Quantizer,
        grad_quantizer_w2: NVFP4Quantizer,
        # Dummy quantizer to wrap fused kernel output
        w1_input_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
        debug_name: Optional[str] = None,
        residual: Optional[torch.Tensor] = None,
        h_row_fp4: Optional[torch.Tensor] = None,
        h_row_sc: Optional[torch.Tensor] = None,
        h_row_sg: Optional[torch.Tensor] = None,
        h_col_fp4: Optional[torch.Tensor] = None,
        h_col_sc: Optional[torch.Tensor] = None,
        h_col_sg: Optional[torch.Tensor] = None,
        h_r_tile: Optional[torch.Tensor] = None,
        h_next_gamma: Optional[torch.Tensor] = None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
        cde_emit: bool = False,
    ):
        M, K = input.shape
        H = w1_weight.shape[0]
        N = w2_weight.shape[0]
        residual_2d = None
        if torch.is_tensor(residual):
            if tuple(residual.shape) != (M, N):
                raise RuntimeError(
                    f"FFN residual shape {tuple(residual.shape)} does not match output {(M, N)}"
                )
            residual_2d = residual if residual.is_contiguous() else residual.contiguous()
            if residual_2d.dtype != torch.bfloat16:
                residual_2d = residual_2d.to(torch.bfloat16)
        ctx.has_residual = residual_2d is not None
        ctx.h_tile = h_row_fp4 is not None and h_row_fp4.numel() != 0
        ctx.h_output = h_next_gamma is not None and h_next_gamma.numel() != 0
        ctx.cde_input = (
            cde_row_rms_partial is not None
            and cde_row_rms_partial.numel() != 0
        )
        cde_input_row_rms_partial = (
            cde_row_rms_partial if ctx.cde_input else None
        )
        ctx.cde_output = bool(cde_emit)
        if (ctx.cde_input or ctx.cde_output) and (ctx.h_tile or ctx.h_output):
            raise RuntimeError("exact C/D/E and H tile carriers are mutually exclusive")
        if (ctx.h_tile or ctx.h_output) and (M < 256 or K % 256 or N % 256):
            raise RuntimeError(
                f"H FFN carrier requires production-aligned output, got MKN={(M, K, N)}"
            )
        if ctx.h_output and residual_2d is None:
            raise RuntimeError("H FFN output carrier requires the residual stream")
        ctx._ffn_v4_cpp_only = use_tk_localcta_v4_cpp_only()

        # No sync needed: all forward/backward ops are on the default CUDA stream,
        # so GPU ordering guarantees previous kernels complete before new ones run.
        # TMA host buffers from dim1_alloc are per-(M,K,H) and same-stream ordered.

        # ── Zero-copy input: use sb['input'] as compute buffer ──
        use_localcta = use_tk_localcta_forward_for_m(M)
        if ctx.cde_input:
            if not use_localcta:
                raise RuntimeError(
                    "exact Wo-to-FFN C/D/E is retained only for localCTA v4"
                )
            if get_tk_localcta_variant() != 'v4':
                raise RuntimeError("exact C/D/E localCTA FFN support requires variant v4")
            if use_tk_localcta_direct_contract():
                raise RuntimeError(
                    "exact C/D/E does not support the localCTA direct-TE contract"
                )
            if _nvfp4_quantizer_extras_enabled("activation"):
                raise RuntimeError("exact C/D/E does not support NVFP4 activation RHT/SR")
            expected_partial_width = K // 256
            if (
                M % 256
                or K != 4096
                or not cde_row_rms_partial.is_cuda
                or not cde_row_rms_partial.is_contiguous()
                or cde_row_rms_partial.dtype != torch.float32
                or tuple(cde_row_rms_partial.shape) != (M, expected_partial_width)
            ):
                raise RuntimeError(
                    "exact C/D/E FFN row RMS partial must be contiguous CUDA "
                    f"float32 {(M, expected_partial_width)}, got "
                    f"shape={tuple(cde_row_rms_partial.shape)} "
                    f"dtype={cde_row_rms_partial.dtype}"
                )
            if not use_tk_quant():
                raise RuntimeError("exact C/D/E FFN requires the native TK quantizer")
            _trace_backend_choice('localcta_exact_cde_wo_ffn', 'native')
        if ctx.cde_output:
            if use_localcta and get_tk_localcta_variant() != 'v4':
                raise RuntimeError("exact C/D/E localCTA support requires variant v4")
            if use_localcta and use_tk_localcta_direct_contract():
                raise RuntimeError(
                    "exact C/D/E does not support the localCTA direct-TE contract"
                )
            if residual_2d is None:
                raise RuntimeError("exact C/D/E requires the W2 residual stream")
            if M < 256 or K % 128 or H % 128 or N != K:
                raise RuntimeError(
                    f"exact C/D/E requires an aligned square residual output, got MKNH={(M, K, N, H)}"
                )
        residual_aliases_input = _ffn_rms_residual_aliases_input(
            input, residual_2d
        )
        residual_rms_shape_enabled = use_tk_ffn_rms_residual_bwd_for_shape(
            M, K, H, use_localcta=use_localcta
        )
        ctx._fuse_rms_residual_bwd = (
            residual_aliases_input and residual_rms_shape_enabled and not ctx.h_tile
        )
        if residual_rms_shape_enabled:
            _trace_backend_choice(
                'ffn_rms_residual_eligibility',
                (
                    f"alias={int(residual_aliases_input)},"
                    f"same_ptr={int(torch.is_tensor(residual_2d) and residual_2d.data_ptr() == input.data_ptr())},"
                    f"input_stride={tuple(input.stride())},"
                    f"residual_stride={tuple(residual_2d.stride()) if torch.is_tensor(residual_2d) else None},"
                    f"input_grad={int(input.requires_grad)},"
                    f"residual_grad={int(torch.is_tensor(residual_2d) and residual_2d.requires_grad)}"
                ),
            )
        use_localcta_direct_ffn = use_localcta and use_tk_localcta_direct_contract()
        sb_key = (M, K, H, input.device.index)
        _graph_max_m = int(os.environ.get('FFN_GRAPH_MAX_M', '4096'))
        sb = _ffn_sb_cache.get(sb_key) if (not use_localcta and use_cuda_graph() and M <= _graph_max_m) else None
        if sb is not None:
            sb['input'].copy_(input)
            inp = sb['input']  # forward compute uses stable address
        else:
            inp = input if (input.is_contiguous() and input.dtype == torch.bfloat16) \
                else input.contiguous().to(torch.bfloat16)
        if use_localcta and use_tk_debug_clone_ffn_input():
            inp = inp.clone()
        # norm_weight is an nn.Parameter(dtype=bf16) — always contiguous
        nw = _as_contiguous_bf16(norm_weight.detach())
        debug_prefix = 'ffn_fwd.localcta' if use_localcta else 'ffn_fwd.plain'
        _debug_check_finite(f'{debug_prefix}.input', inp)
        _debug_check_finite(f'{debug_prefix}.norm_weight', nw)

        from .tk_gemm import _get_tk, _get_tk_plain, _get_tk_quant_for_gemm
        tk_mod = _get_tk() if use_localcta else _get_tk_plain()
        te_fused = _get_te_fused()

        def _quantize_from_cde_row_rms_partial():
            tk_q_cde = _get_tk_quant_for_gemm()
            result = tk_q_cde.tk_rmsnorm_quantize_from_row_rms_partial_final_sg(
                inp,
                nw,
                cde_row_rms_partial,
                float(epsilon),
                True,
                use_nvfp4_encode_centric(),
            )
            quantized = _TKQuantized(
                result[0], result[1], result[4],
                result[2], result[3], result[5],
                keepalive=_result_keepalive(result, 7),
            )
            return quantized, result[6]

        w1_bf16 = w1_weight.detach()
        w3_bf16 = w3_weight.detach()
        N_dims_13 = [H, H]
        _dump_ffn_tensors(
            "w13_weight_src",
            debug_name,
            {
                "debug_name": debug_name,
                "w1_weight": w1_weight,
                "w3_weight": w3_weight,
            },
        )
        localcta_variant = os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() if use_localcta else 'v1'
        use_small_m_plain_ffn = (not use_localcta and M < 256)

        use_ffn_localcta_tk_quant_contract = (
            use_localcta and use_tk_ffn_localcta_tk_quant_contract()
        )

        # Quantize W2 separately (different shape: K×H)
        w2_weight_quant_stream = None
        _tk_stage_trace('ffn_fwd_sub', 'w2_weight_quant_start', debug_name)
        if (
            use_localcta
            and localcta_variant == 'v4'
            and not use_localcta_direct_ffn
            and use_tk_localcta_v4_ffn_w2_weight_quant_overlap()
        ):
            w2_weight_quant_stream = _get_ms_stream()
            caller_stream = torch.cuda.current_stream()
            w2_weight_quant_stream.wait_stream(caller_stream)
            _record_tensors_on_stream(w2_weight, w2_weight_quant_stream)
            with torch.cuda.stream(w2_weight_quant_stream):
                w2_nvfp4 = _fast_quantize(
                    w2_weight,
                    w2_weight_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=True,
                )
        else:
            w2_nvfp4 = _fast_quantize(
                w2_weight,
                w2_weight_quantizer,
                tk_swizzle=not use_localcta_direct_ffn,
                use_localcta_override=(use_localcta and not use_localcta_direct_ffn),
            )
        _tk_stage_trace('ffn_fwd_sub', 'w2_weight_quant_done', debug_name)
        w2_dgrad_col = None
        w13_dgrad_cols = None
        need_v4_direct_dgrad_cols = (
            os.environ.get('USE_TK_LOCALCTA_V4_SG_DIRECT_CONSUMERS', '0') == '1'
            or use_tk_localcta_v4_raw_backward_fallbacks(M)
        )
        if (
            use_localcta
            and localcta_variant == 'v4'
            and not use_tk_localcta_2d_weight_quant()
            and not use_ffn_localcta_tk_quant_contract
            and not use_tk_localcta_v4_cpp_only()
            and need_v4_direct_dgrad_cols
            and hasattr(_get_tk_quant_for_gemm(), 'tk_quantize_for_gemm_direct')
        ):
            w2_direct = _get_tk_quant_for_gemm().tk_quantize_for_gemm_direct(
                _as_contiguous_bf16(w2_weight), True, True
            )
            w2_dgrad_col = (
                w2_direct[2],
                w2_direct[3],
                w2_direct[5] if len(w2_direct) > 5 and torch.is_tensor(w2_direct[5]) and w2_direct[5].numel() > 0 else w2_direct[4],
            )
            w1_direct = _get_tk_quant_for_gemm().tk_quantize_for_gemm_direct(
                _as_contiguous_bf16(w1_weight), True, False
            )
            w3_direct = _get_tk_quant_for_gemm().tk_quantize_for_gemm_direct(
                _as_contiguous_bf16(w3_weight), True, False
            )
            w13_dgrad_cols = (
                (
                    w1_direct[2],
                    w1_direct[3],
                    w1_direct[5] if len(w1_direct) > 5 and torch.is_tensor(w1_direct[5]) and w1_direct[5].numel() > 0 else w1_direct[4],
                ),
                (
                    w3_direct[2],
                    w3_direct[3],
                    w3_direct[5] if len(w3_direct) > 5 and torch.is_tensor(w3_direct[5]) and w3_direct[5].numel() > 0 else w3_direct[4],
                ),
            )
        _dump_ffn_tensors(
            "w2_weight_src",
            debug_name,
            {
                "debug_name": debug_name,
                "w2_weight": w2_weight,
            },
        )

        qkv_weight_quant_keepalive = None
        ffn_weight_quant_keepalive = None
        ffn_dgrad_weight_quant_keepalive = None
        dgrad_wc_fp4_cols = None
        dgrad_wc_sc_cols = None
        dgrad_wc_sg_cols = None
        w1_nvfp4 = None
        w3_nvfp4 = None
        if ctx.h_tile:
            x_nvfp4 = _TKQuantized(
                h_row_fp4, h_row_sc, h_row_sg,
                h_col_fp4, h_col_sc, h_col_sg,
            )
            inv_rms = h_r_tile
        if use_localcta:
            if (
                _nvfp4_quantizer_extras_enabled("activation")
                and localcta_variant != 'v4'
            ):
                _check_nvfp4_native_extras_supported(
                    "activation", "localCTA FFN forward activation producer"
                )
            _check_nvfp4_native_extras_supported("weight", "localCTA/v4 FFN forward weight producer")
            tk_q = _get_tk_quant_for_gemm()
            # Keep FFN localCTA eager input quant on the baseline RMSNorm-only
            # + prepared quant path for now. The current localCTA fused norm
            # helper is exact, but on the real M=65536 FFN forward it has not
            # cleared the perf gate yet; the backward-side localCTA_fused wins
            # are what make the mode faster overall.
            use_localcta_fused_ffn_norm = (
                not use_ffn_localcta_tk_quant_contract
                and
                use_tk_localcta_fused()
                and localcta_variant == 'v4'
                and use_tk_localcta_v4_fast_ffn_fused_norm()
                and not _nvfp4_quantizer_extras_enabled("activation")
            )
            use_localcta_row_prepared_col_outer_contract = (
                use_ffn_localcta_tk_quant_contract
                and localcta_variant == 'v4'
                and use_tk_localcta_v4_row_prepared_col_outer()
                and M % 256 == 0
                and K % 256 == 0
            )
            use_localcta_row_prepared_rmsnorm_quant = (
                use_localcta_row_prepared_col_outer_contract
                and use_tk_localcta_v4_row_prepared_rmsnorm_quant()
            )
            use_localcta_native_extras_ffn_norm = (
                localcta_variant == 'v4'
                and (
                    _nvfp4_quantizer_extras_enabled("activation")
                    or use_tk_localcta_v4_fast_ffn_rmsnorm_quant()
                )
                and (
                    hasattr(getattr(tk_q, '_mod', None), 'tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt')
                    or (
                        use_localcta_row_prepared_rmsnorm_quant
                        and hasattr(
                            getattr(tk_q, '_mod', None),
                            'tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer',
                        )
                    )
                )
            )
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_start', debug_name)
            if ctx.h_tile:
                pass
            elif ctx.cde_input:
                x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
            elif use_localcta_native_extras_ffn_norm:
                x_nvfp4, inv_rms = _fast_rmsnorm_quantize_localcta_v4_opt(
                    inp,
                    nw,
                    float(epsilon),
                    nvfp4_role="activation",
                    prefer_row_prepared_col_outer=use_localcta_row_prepared_rmsnorm_quant,
                    separate_bf16_final_sg=(
                        use_tk_localcta_v4_ffn_separate_bf16_final_sg()
                    ),
                )
            elif use_localcta_fused_ffn_norm and use_tk_localcta_fused() and hasattr(tk_q, 'tk_fused_norm_quantize'):
                result = tk_q.tk_fused_norm_quantize(
                    inp,
                    nw,
                    float(epsilon),
                    False,
                    True,
                )
                x_nvfp4 = _TKQuantized(result[0], result[1], result[4],
                                       result[2], result[3], result[5],
                                       keepalive=_result_keepalive(result, 7))
                inv_rms = result[6]
            else:
                normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                _debug_check_finite('ffn_fwd.localcta.normed', normed)
                x_nvfp4 = _fast_quantize(
                    normed,
                    w1_input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_done', debug_name)

            _tk_stage_trace('ffn_fwd_sub', 'weight_quant_start', debug_name)
            group_result = _tk_group_quantize_ffn_weights(
                tk_q,
                w1_bf16,
                w3_bf16,
                N_dims_13,
                prefer_split=(M >= 256),
            )
            _tk_stage_trace('ffn_fwd_sub', 'weight_quant_done', debug_name)
            # Backward retains the unpacked col payloads directly. Keeping the
            # complete localCTA result also retained its forward-only row
            # payloads in every layer.
            ffn_weight_quant_keepalive = _result_keepalive(group_result, 8)
            wc_fp4_row, wc_sc_row, fwd_b_sg, \
                wc_fp4_cols, wc_sc_cols, sg_cat, _, wc_sg_cols = \
                group_result[:8]
            dgrad_wc_fp4_cols = wc_fp4_cols
            dgrad_wc_sc_cols = wc_sc_cols
            dgrad_wc_sg_cols = wc_sg_cols
        else:
            wc_sg_cols = None

        # Grouped weight quantization for [W1; W3]
        if not use_localcta:
            tk_q = _get_tk_quant()
        if not use_localcta and (use_tk_ffn_safe_input_quant() or M < 256):
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_start', debug_name)
            if ctx.h_tile:
                pass
            elif ctx.cde_input:
                x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
            else:
                normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                x_nvfp4 = _fast_quantize(
                    normed,
                    w1_input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_done', debug_name)
            _tk_stage_trace('ffn_fwd_sub', 'weight_quant_start', debug_name)
            if use_small_m_plain_ffn:
                w1_nvfp4 = _fast_quantize(
                    w1_weight,
                    w1_weight_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                w3_nvfp4 = _fast_quantize(
                    w3_weight,
                    w3_weight_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                wc_fp4_row = wc_sc_row = fwd_b_sg = None
                wc_fp4_cols = [w1_nvfp4._tk_col[0], w3_nvfp4._tk_col[0]]
                wc_sc_cols = [w1_nvfp4._tk_col[1], w3_nvfp4._tk_col[1]]
                dgrad_wc_fp4_cols = wc_fp4_cols
                dgrad_wc_sc_cols = wc_sc_cols
                dgrad_wc_sg_cols = None
                sg1 = w1_nvfp4._tk_col[2]
                sg3 = w3_nvfp4._tk_col[2]
                if not torch.is_tensor(sg1):
                    sg1 = torch.tensor([float(sg1)], dtype=torch.float32, device=inp.device)
                else:
                    sg1 = sg1.to(torch.float32).reshape(-1)
                if not torch.is_tensor(sg3):
                    sg3 = torch.tensor([float(sg3)], dtype=torch.float32, device=inp.device)
                else:
                    sg3 = sg3.to(torch.float32).reshape(-1)
                sg_cat = torch.cat([sg1, sg3], dim=0)
                ffn_weight_quant_keepalive = (w1_nvfp4, w3_nvfp4)
            else:
                group_result = _tk_group_quantize_ffn_weights(
                    tk_q,
                    w1_bf16,
                    w3_bf16,
                    N_dims_13,
                    prefer_split=(M >= 256),
                )
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, dgrad_b_sg, sg_cat, ffn_weight_quant_keepalive = \
                    group_result
                dgrad_wc_fp4_cols = wc_fp4_cols
                dgrad_wc_sc_cols = wc_sc_cols
                dgrad_wc_sg_cols = dgrad_b_sg
            _tk_stage_trace('ffn_fwd_sub', 'weight_quant_done', debug_name)
        elif not use_localcta and use_tk_quant() and use_tk_ms():
            # Multi-stream: input quant on s0 ∥ weight quant on s1
            s0 = torch.cuda.current_stream()
            s1 = _get_ms_stream()

            s1.wait_stream(s0)
            with torch.cuda.stream(s1):
                _tk_stage_trace('ffn_fwd_sub', 'weight_quant_start', debug_name)
                group_result = _tk_group_quantize_ffn_weights(
                    tk_q,
                    w1_bf16,
                    w3_bf16,
                    N_dims_13,
                    prefer_split=(M >= 256),
                )
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, dgrad_b_sg, sg_cat, ffn_weight_quant_keepalive = \
                    group_result
                dgrad_wc_fp4_cols = wc_fp4_cols
                dgrad_wc_sc_cols = wc_sc_cols
                dgrad_wc_sg_cols = dgrad_b_sg
                _tk_stage_trace('ffn_fwd_sub', 'weight_quant_done', debug_name)

            # Meanwhile, fused RMSNorm + FP4 quantize on s0 (TK-swizzled scales)
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_start', debug_name)
            if ctx.h_tile:
                pass
            elif ctx.cde_input:
                x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
            elif _can_fast_rmsnorm_quantize_tk_regular_opt(inp, "activation"):
                x_nvfp4, inv_rms = _fast_rmsnorm_quantize_tk_regular_opt(
                    inp,
                    nw,
                    float(epsilon),
                    nvfp4_role="activation",
                )
            elif _nvfp4_quantizer_extras_enabled("activation"):
                normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                x_nvfp4 = _fast_quantize(
                    normed,
                    w1_input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
            else:
                result = tk_q.tk_fused_norm_quantize(
                    inp,
                    nw,
                    float(epsilon),
                    False,
                    True,
                )
                fp4, si, fp4_t, si_t, sg, inv_rms = result[:6]
                x_nvfp4 = _TKQuantized(
                    fp4, si, sg, fp4_t, si_t,
                    keepalive=_result_keepalive(result, 6),
                )
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_done', debug_name)

            s0.wait_stream(s1)
            _record_tensors_on_stream(group_result, s0)
        elif not use_localcta:
            # Single-stream: fused RMSNorm + FP4 quantize (TK-swizzled scales)
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_start', debug_name)
            if ctx.h_tile:
                pass
            elif ctx.cde_input:
                x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
            elif _can_fast_rmsnorm_quantize_tk_regular_opt(inp, "activation"):
                x_nvfp4, inv_rms = _fast_rmsnorm_quantize_tk_regular_opt(
                    inp,
                    nw,
                    float(epsilon),
                    nvfp4_role="activation",
                )
            elif _nvfp4_quantizer_extras_enabled("activation"):
                normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                x_nvfp4 = _fast_quantize(
                    normed,
                    w1_input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
            else:
                result = tk_q.tk_fused_norm_quantize(
                    inp,
                    nw,
                    float(epsilon),
                    False,
                    True,
                )
                fp4, si, fp4_t, si_t, sg, inv_rms = result[:6]
                x_nvfp4 = _TKQuantized(
                    fp4, si, sg, fp4_t, si_t,
                    keepalive=_result_keepalive(result, 6),
                )
            _tk_stage_trace('ffn_fwd_sub', 'input_quant_done', debug_name)

            _tk_stage_trace('ffn_fwd_sub', 'weight_quant_start', debug_name)
            group_result = _tk_group_quantize_ffn_weights(
                tk_q,
                w1_bf16,
                w3_bf16,
                N_dims_13,
                prefer_split=(M >= 256),
            )
            wc_fp4_row, wc_sc_row, fwd_b_sg, \
                wc_fp4_cols, wc_sc_cols, dgrad_b_sg, sg_cat, ffn_weight_quant_keepalive = \
                group_result
            dgrad_wc_fp4_cols = wc_fp4_cols
            dgrad_wc_sc_cols = wc_sc_cols
            dgrad_wc_sg_cols = dgrad_b_sg
            _tk_stage_trace('ffn_fwd_sub', 'weight_quant_done', debug_name)

        if (
            use_localcta
            and not use_tk_localcta_2d_weight_quant()
            and not use_ffn_localcta_tk_quant_contract
            and (
                (localcta_variant != 'v4' and use_tk_localcta_v3_prepared_split2())
                or (localcta_variant == 'v4' and use_tk_localcta_v4_prepared_split2_dgrad_weights())
            )
            and hasattr(tk_q, '_mod')
            and hasattr(tk_q._mod, 'tk_localcta_group_quantize_for_gemm_prepared')
        ):
            prepared_group_result = tk_q._mod.tk_localcta_group_quantize_for_gemm_prepared(
                torch.cat([w1_bf16, w3_bf16], dim=0).contiguous(),
                N_dims_13,
            )
            dgrad_wc_fp4_cols = prepared_group_result[3]
            dgrad_wc_sc_cols = prepared_group_result[4]
            dgrad_wc_sg_cols = wc_sg_cols
            ffn_dgrad_weight_quant_keepalive = prepared_group_result

        if (
            use_localcta
            and not use_tk_localcta_2d_weight_quant()
            and not use_ffn_localcta_tk_quant_contract
            and localcta_variant == 'v4'
            and use_tk_localcta_v4_tk_ffn_dgrad_weights()
        ):
            tk_q_ref = _get_tk_quant()
            tk_group_result = _tk_group_quantize_ffn_weights(
                tk_q_ref, w1_bf16, w3_bf16, N_dims_13
            )
            dgrad_wc_fp4_cols = tk_group_result[3]
            dgrad_wc_sc_cols = tk_group_result[4]
            dgrad_b_sg = tk_group_result[5]
            dgrad_b_tiles = [fp4.size(0) // 256 for fp4 in dgrad_wc_fp4_cols]
            dgrad_wc_sg_cols = list(dgrad_b_sg.split(dgrad_b_tiles, dim=0))
            ffn_dgrad_weight_quant_keepalive = tk_group_result

        if _ffn_capture_path():
            _append_ffn_capture({
                "event": "ffn_dgrad_weight_payload",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "use_localcta": bool(use_localcta),
                "localcta_variant": localcta_variant if use_localcta else None,
                "wc_sc_cols": [_tensor_capture_stats(t) for t in wc_sc_cols] if wc_sc_cols is not None else None,
                "wc_sg_cols": [_tensor_capture_stats(t) for t in wc_sg_cols] if wc_sg_cols is not None else None,
                "dgrad_wc_sc_cols": [_tensor_capture_stats(t) for t in dgrad_wc_sc_cols] if dgrad_wc_sc_cols is not None else None,
                "dgrad_wc_sg_cols": [_tensor_capture_stats(t) for t in dgrad_wc_sg_cols] if dgrad_wc_sg_cols is not None else None,
            })

        # ── sb copy batch 1: x cols, inv_rms, wc_*, sg_cat, norm_weight ──
        # Interleaved BEFORE GEMM1 on same stream (multi-stream causes HBM
        # contention with GEMMs; same-stream ordering is more efficient).
        if sb is not None:
            sb['inv_rms'].copy_(inv_rms)
            sb['norm_weight'].copy_(nw)
            sb['sg_cat'].copy_(sg_cat)
            sb['x_fp4_c'].copy_(x_nvfp4._tk_col[0])
            sb['x_sc_c'].copy_(x_nvfp4._tk_col[1])
            sb['x_sg_c'].copy_(x_nvfp4._tk_col[2])
            n_groups_fwd = len(wc_fp4_cols)
            for i in range(n_groups_fwd):
                sb[f'wc_fp4_c_{i}'].copy_(wc_fp4_cols[i])
                sb[f'wc_sc_c_{i}'].copy_(wc_sc_cols[i])

        # ---- Single grouped GEMM for [W1; W3] ---- 
        x_fp4, x_sc, x_sg = x_nvfp4._tk_row
        x_nvfp4_fwd = x_nvfp4
        h_nvfp4_fwd = None
        use_split_cache = use_small_m_plain_ffn or use_localcta or (
            use_tk_ffn_split_cache()
            and
            sb is None
            and not (use_cuda_graph() and M <= _graph_max_m)
            and use_tk_quant()
            and hasattr(tk_q, 'tk_silu_quantize_split_for_gemm')
            and hasattr(tk_q, 'tk_silu_deriv_quantize_split_for_gemm')
        )
        empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=inp.device)
        use_saved_sigmoid_experiment = (
            use_localcta
            and not use_tk_localcta_fused()
            and M >= _localcta_ffn_experiment_min_m('USE_TK_LOCALCTA_FFN_SAVED_SIGMOID_MIN_M')
            and use_tk_localcta_ffn_saved_sigmoid()
            and hasattr(te_fused, 'fused_silu_mul_and_sigmoid_bf16_out_no_amax')
        )

        # Zero-copy: if backward's stable buffers exist, use sb['h13'] as the
        # GEMM output directly — avoids a 1.4 GB copy at M=65536.
        if use_small_m_plain_ffn:
            normed_fwd = _pad_rows_bf16(normed, 256)
            x_nvfp4_fwd = _fast_quantize(
                normed_fwd,
                w1_input_quantizer,
                tk_swizzle=True,
                use_localcta_override=False,
            )
            h1_raw_full = torch.empty(normed_fwd.size(0), H, dtype=torch.bfloat16, device=inp.device)
            h3_full = torch.empty_like(h1_raw_full)
            tk_forward_gemm(x_nvfp4_fwd, w1_nvfp4, h1_raw_full, use_localcta=False)
            tk_forward_gemm(x_nvfp4_fwd, w3_nvfp4, h3_full, use_localcta=False)
            h1_raw = h1_raw_full[:M]
            h3 = h3_full[:M]
            _debug_check_finite('ffn_fwd.plain.h1_raw', h1_raw)
            _debug_check_finite('ffn_fwd.plain.h3', h3)
            h_full, _ = te_fused.fused_silu_mul_bf16(h1_raw_full, h3_full)
            h = h_full[:M]
            sig_h1 = empty_bf16
            h_nvfp4 = _fast_quantize(
                h,
                w1_input_quantizer,
                tk_swizzle=True,
                use_localcta_override=False,
            )
            h_nvfp4_fwd = _fast_quantize(
                h_full,
                w1_input_quantizer,
                tk_swizzle=True,
                use_localcta_override=False,
            )
            h13 = empty_bf16
        elif use_localcta:
            h1_raw = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
            h3 = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
            _tk_stage_trace('ffn_fwd_sub', 'group_gemm_start', debug_name)
            h1_raw, h3 = tk_grouped_forward_gemm_split(
                x_nvfp4, wc_fp4_row, wc_sc_row, fwd_b_sg, N_dims_13,
                outs=[h1_raw, h3],
                use_localcta=use_localcta,
            )
            _tk_stage_trace('ffn_fwd_sub', 'group_gemm_done', debug_name)
            _debug_check_finite('ffn_fwd.localcta.h1_raw', h1_raw)
            _debug_check_finite('ffn_fwd.localcta.h3', h3)
            _tk_stage_trace('ffn_fwd_sub', 'producer_start', debug_name)
            if use_localcta_direct_ffn:
                _trace_backend_choice('localcta_ffn_fwd', 'direct_bf16_producer')
                fwd_state = _get_ffn_localcta_fwd_state(M, H, inp.device)
                h = fwd_state['h']
                if hasattr(te_fused, 'fused_silu_mul_bf16_out_no_amax'):
                    te_fused.fused_silu_mul_bf16_out_no_amax(
                        h1_raw, h3, h
                    )
                elif hasattr(te_fused, 'fused_silu_mul_bf16_out'):
                    te_fused.fused_silu_mul_bf16_out(
                        h1_raw, h3, h, fwd_state['amax']
                    )
                else:
                    h, _ = te_fused.fused_silu_mul_bf16(h1_raw, h3)
                sig_h1 = empty_bf16
                h_nvfp4 = _fast_quantize(h, w1_input_quantizer, tk_swizzle=False)
            elif (
                not use_saved_sigmoid_experiment
                and hasattr(tk_q, 'tk_silu_quantize_split_for_gemm')
            ):
                _trace_backend_choice('localcta_ffn_fwd', 'split_silu_quant')
                h_nvfp4 = _localcta_silu_quantize_split_for_gemm(
                    tk_q,
                    h1_raw,
                    h3,
                )
                sig_h1 = empty_bf16
            elif hasattr(te_fused, 'fused_silu_mul_bf16_out'):
                _trace_backend_choice(
                    'localcta_ffn_fwd',
                    'saved_sigmoid_bf16_producer' if use_saved_sigmoid_experiment else 'bf16_producer'
                )
                fwd_state = _get_ffn_localcta_fwd_state(M, H, inp.device)
                h = fwd_state['h']
                if use_saved_sigmoid_experiment:
                    sig_h1 = torch.empty_like(h1_raw)
                    te_fused.fused_silu_mul_and_sigmoid_bf16_out_no_amax(
                        h1_raw, h3, h, sig_h1
                    )
                elif hasattr(te_fused, 'fused_silu_mul_bf16_out_no_amax'):
                    te_fused.fused_silu_mul_bf16_out_no_amax(
                        h1_raw, h3, h
                    )
                    sig_h1 = empty_bf16
                else:
                    te_fused.fused_silu_mul_bf16_out(
                        h1_raw, h3, h, fwd_state['amax']
                    )
                    sig_h1 = empty_bf16
                h_nvfp4 = _fast_quantize(
                    h,
                    w1_input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
            else:
                _trace_backend_choice('localcta_ffn_fwd', 'fallback_bf16_producer')
                h, _ = te_fused.fused_silu_mul_bf16(h1_raw, h3)
                sig_h1 = empty_bf16
                h_nvfp4 = _fast_quantize(
                    h,
                    w1_input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
            _tk_stage_trace('ffn_fwd_sub', 'producer_done', debug_name)
            h13 = empty_bf16
        elif use_split_cache:
            h1_raw = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
            h3 = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
            sig_h1 = empty_bf16
        else:
            h13 = sb['h13'] if sb is not None else torch.empty(M, 2 * H, dtype=torch.bfloat16, device=inp.device)
            sig_h1 = empty_bf16

        is_inference = not any(ctx.needs_input_grad)
        if use_small_m_plain_ffn:
            pass
        elif use_localcta:
            pass
        elif use_split_cache:
            _tk_stage_trace('ffn_fwd_sub', 'group_gemm_start', debug_name)
            h1_raw, h3 = tk_grouped_forward_gemm_split(
                x_nvfp4, wc_fp4_row, wc_sc_row, fwd_b_sg, N_dims_13,
                outs=[h1_raw, h3],
            )
            _tk_stage_trace('ffn_fwd_sub', 'group_gemm_done', debug_name)
            _tk_stage_trace('ffn_fwd_sub', 'producer_start', debug_name)
            if use_tk_ffn_fwd_safe_producer():
                h, _ = te_fused.fused_silu_mul_bf16(h1_raw, h3)
                h_nvfp4 = _fast_quantize(
                    h,
                    w1_input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
            else:
                result = tk_q.tk_silu_quantize_split_for_gemm(h1_raw, h3)
                h_nvfp4 = _TKQuantized(
                    result[0], result[1], result[4],
                    result[2], result[3],
                    result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4],
                    keepalive=_result_keepalive(result, 6),
                )
            _tk_stage_trace('ffn_fwd_sub', 'producer_done', debug_name)
        else:
            _tk_stage_trace('ffn_fwd_sub', 'group_gemm_start', debug_name)
            tk_mod.nvfp4_grouped_gemm(
                x_fp4, x_sc, x_sg, wc_fp4_row, wc_sc_row, fwd_b_sg,
                h13
            )
            _tk_stage_trace('ffn_fwd_sub', 'group_gemm_done', debug_name)

            # Fused silu(h1)*h3 + FP4 quantize in 1 kernel (saves ~1.1ms at M=65536)
            _tk_stage_trace('ffn_fwd_sub', 'producer_start', debug_name)
            if use_tk_ffn_fwd_safe_producer():
                h, _ = te_fused.fused_silu_mul_strided_bf16(h13, H)
                h_nvfp4 = _fast_quantize(h, w1_input_quantizer, tk_swizzle=True)
            elif use_tk_quant() and hasattr(tk_q, 'tk_silu_quantize_for_gemm'):
                result = tk_q.tk_silu_quantize_for_gemm(h13, H)
                h_nvfp4 = _TKQuantized(
                    result[0], result[1], result[4],
                    result[2], result[3],
                    result[5] if len(result) > 5 and torch.is_tensor(result[5]) and result[5].numel() > 0 else result[4],
                    keepalive=_result_keepalive(result, 6),
                )
            else:
                h, _ = te_fused.fused_silu_mul_strided_bf16(h13, H)
                h_nvfp4 = _fast_quantize(h, w1_input_quantizer, tk_swizzle=True)
            _tk_stage_trace('ffn_fwd_sub', 'producer_done', debug_name)

        recompute_h13_for_bwd = (
            use_tk_ffn_recompute_h13()
            and use_split_cache
            and not is_inference
            and not use_small_m_plain_ffn
            and wc_fp4_row is not None
            and wc_sc_row is not None
            and fwd_b_sg is not None
        )
        requant_h13_operands_for_bwd = (
            recompute_h13_for_bwd
            and use_tk_ffn_requant_h13_operands()
            and use_localcta
            and localcta_variant == 'v4'
            and use_localcta_native_extras_ffn_norm
            and not ctx.h_tile
        )
        requant_h13_activation_for_bwd = (
            recompute_h13_for_bwd
            and use_tk_ffn_requant_h13_activation()
            and not use_localcta
            and not ctx.h_tile
            and not _nvfp4_quantizer_extras_enabled("activation")
        )
        recompute_h_for_w2_wgrad = (
            recompute_h13_for_bwd
            and use_tk_ffn_recompute_h_for_w2_wgrad()
            and use_localcta
            and localcta_variant == 'v4'
            and not use_localcta_direct_ffn
            and not ctx.h_tile
            and not _nvfp4_quantizer_extras_enabled("activation")
            and hasattr(tk_q, 'tk_silu_quantize_split_for_gemm')
        )
        if not use_localcta_direct_ffn:
            # GEMM1 has consumed these row-oriented operands. Backward uses
            # only their col-oriented caches, so retaining the rows through
            # the whole autograd graph wastes hundreds of MiB per layer.
            if (
                requant_h13_operands_for_bwd
                or requant_h13_activation_for_bwd
                or not recompute_h13_for_bwd
            ):
                _release_tk_row_storage(x_nvfp4)
                x_fp4 = x_sc = x_sg = None
                group_result = None
                if requant_h13_operands_for_bwd or not recompute_h13_for_bwd:
                    wc_fp4_row = wc_sc_row = fwd_b_sg = None
                if requant_h13_operands_for_bwd:
                    # The backward producer recreates both orientations, so its
                    # column payloads replace the forward wgrad and dgrad
                    # caches as well.
                    x_nvfp4 = None
                    ffn_weight_quant_keepalive = ()
                    wc_fp4_cols = wc_sc_cols = wc_sg_cols = None
                    dgrad_wc_fp4_cols = dgrad_wc_sc_cols = None
                    dgrad_wc_sg_cols = None
        if not use_small_m_plain_ffn:
            normed = None

        if w2_weight_quant_stream is not None:
            _tk_stage_trace('ffn_fwd_sub', 'w2_weight_quant_wait_start', debug_name)
            caller_stream.wait_stream(w2_weight_quant_stream)
            _record_tensors_on_stream(
                (
                    w2_nvfp4._tk_row,
                    w2_nvfp4._tk_col,
                    w2_nvfp4._tk_row_chunk_sg,
                    w2_nvfp4._tk_col_chunk_sg,
                    w2_nvfp4._keepalive,
                ),
                caller_stream,
            )
            _tk_stage_trace('ffn_fwd_sub', 'w2_weight_quant_wait_done', debug_name)

        # ── sb copy batch 2: h cols, w2 cols ──
        # Interleaved BEFORE GEMM2.
        if sb is not None:
            sb['w2_fp4_c'].copy_(w2_nvfp4._tk_col[0])
            sb['w2_sc_c'].copy_(w2_nvfp4._tk_col[1])
            sb['w2_sg_c'].copy_(w2_nvfp4._tk_col[2])
            sb['h_fp4_c'].copy_(h_nvfp4._tk_col[0])
            sb['h_sc_c'].copy_(h_nvfp4._tk_col[1])
            sb['h_sg_c'].copy_(h_nvfp4._tk_col[2])

        y = torch.empty((M, N), dtype=torch.bfloat16, device=inp.device)
        cde_row_rms_partial = None
        _tk_stage_trace('ffn_fwd_sub', 'w2_gemm_start', debug_name)
        if use_localcta_direct_ffn and not ctx.h_output:
            tex.generic_gemm(
                w2_nvfp4, True, h_nvfp4, False,
                y, None, TE_DType[torch.bfloat16],
                None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )
            if residual_2d is not None:
                y.add_(residual_2d)
        elif use_small_m_plain_ffn and not ctx.cde_output:
            y_fwd = torch.empty((normed_fwd.size(0), N), dtype=torch.bfloat16, device=inp.device)
            tk_forward_gemm(h_nvfp4_fwd, w2_nvfp4, y_fwd, use_localcta=False)
            y.copy_(y_fwd[:M])
            if residual_2d is not None:
                y.add_(residual_2d)
        else:
            if residual_2d is not None:
                if ctx.cde_output:
                    _trace_backend_choice(
                        'localcta_exact_cde_w2' if use_localcta
                        else 'v5_exact_cde_w2',
                        'native',
                    )
                    y, cde_row_rms_partial = tk_forward_gemm_residual_rms_partial(
                        h_nvfp4,
                        w2_nvfp4,
                        residual_2d,
                        out=y,
                        use_localcta=use_localcta,
                    )
                elif ctx.h_output:
                    from .tk_gemm import tk_forward_gemm_h_carrier

                    h_carrier = tk_forward_gemm_h_carrier(
                        h_nvfp4,
                        w2_nvfp4,
                        residual_2d,
                        _as_contiguous_bf16(h_next_gamma),
                        float(epsilon),
                        out=y,
                        use_localcta=use_localcta,
                    )
                elif (
                    use_localcta
                    and get_tk_localcta_variant() == 'v4'
                    and use_tk_localcta_v4_ffn_residual_epilogue()
                ):
                    _trace_backend_choice('localcta_ffn_residual', 'requested')
                elif (
                    not use_localcta
                    and use_tk_v5_ffn_residual_epilogue_for_shape(M, K, H)
                ):
                    _trace_backend_choice('v5_ffn_residual', 'native')
                else:
                    _trace_backend_choice('localcta_ffn_residual', 'fallback_add')
                if not ctx.h_output and not ctx.cde_output:
                    tk_forward_gemm_residual(
                        h_nvfp4, w2_nvfp4, residual_2d, y, use_localcta=use_localcta
                    )
            else:
                tk_forward_gemm(h_nvfp4, w2_nvfp4, y, use_localcta=use_localcta)
        _tk_stage_trace('ffn_fwd_sub', 'w2_gemm_done', debug_name)
        if not use_localcta_direct_ffn:
            # GEMM2 has consumed both row payloads. The backward paths use
            # w2/h col payloads exclusively for dgrad and wgrad.
            _release_tk_row_storage(h_nvfp4)
            _release_tk_row_storage(w2_nvfp4)
            result = None
        _debug_check_finite('ffn_fwd.localcta.output', y)
        if _debug_forward_ref_enabled(debug_name) and not use_small_m_plain_ffn:
            # Diagnostic only: attribute the forward error to the grouped
            # up/gate projections versus the down projection.  Keeping this
            # behind the explicit reference flag avoids any production cost.
            normed_ref = F.rms_norm(inp, (K,), nw, float(epsilon))
            h1_ref = F.linear(normed_ref, w1_bf16)
            h3_ref = F.linear(normed_ref, w3_bf16)
            h_from_fp4_projections = (
                F.silu(h1_raw.float()) * h3.float()
            ).to(torch.bfloat16)
            h_ref = (F.silu(h1_ref.float()) * h3_ref.float()).to(torch.bfloat16)
            y_from_fp4_projections = F.linear(
                h_from_fp4_projections, w2_weight.detach()
            )
            y_ref = F.linear(h_ref, w2_weight.detach())
            if residual_2d is not None:
                y_from_fp4_projections = y_from_fp4_projections + residual_2d
                y_ref = y_ref + residual_2d
            _append_ffn_capture({
                "event": "ffn_forward_stage_ref",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "h1": _tensor_delta_stats(h1_raw, h1_ref),
                "h3": _tensor_delta_stats(h3, h3_ref),
                "h_from_fp4_projections": _tensor_delta_stats(
                    h_from_fp4_projections, h_ref
                ),
                "w2_stage": _tensor_delta_stats(y, y_from_fp4_projections),
                "end_to_end": _tensor_delta_stats(y, y_ref),
            })
        if use_small_m_plain_ffn and _debug_forward_ref_enabled(debug_name):
            y_ref = torch.matmul(h.to(torch.bfloat16), w2_weight.detach().to(torch.bfloat16).transpose(0, 1))
            _append_ffn_capture({
                "event": "ffn_forward_ref",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "stats": _tensor_delta_stats(y, y_ref),
            })
        if _ffn_capture_path():
            _append_ffn_capture({
                "event": "ffn_forward_io",
                "debug_name": debug_name,
                "M": int(M),
                "K": int(K),
                "H": int(H),
                "input": _tensor_capture_stats(inp),
                "inv_rms": _tensor_capture_stats(inv_rms),
                "h1_raw": _tensor_capture_stats(h1_raw),
                "h3": _tensor_capture_stats(h3),
                "output": _tensor_capture_stats(y),
            })
        if use_tk_ffn_sync_fwd() and (use_localcta or use_tk_gemm()):
            torch.cuda.synchronize(y.device)

        h13_save = h13 if (not use_split_cache and not is_inference) else empty_bf16
        h1_raw_save = h1_raw if (use_split_cache and not is_inference and not recompute_h13_for_bwd) else empty_bf16
        h3_save = h3 if (use_split_cache and not is_inference and not recompute_h13_for_bwd) else empty_bf16
        sig_h1_save = sig_h1 if (use_saved_sigmoid_experiment and not is_inference) else empty_bf16
        ctx.save_for_backward(inp, nw, inv_rms, h13_save, h1_raw_save, h3_save, sig_h1_save, sg_cat)
        ctx._wc_fp4_cols = wc_fp4_cols  # per-group colwise weights for dgrad
        ctx._wc_sc_cols = wc_sc_cols
        ctx._wc_sg_cols = wc_sg_cols
        ctx._dgrad_wc_fp4_cols = dgrad_wc_fp4_cols
        ctx._dgrad_wc_sc_cols = dgrad_wc_sc_cols
        ctx._dgrad_wc_sg_cols = dgrad_wc_sg_cols
        ctx.w2_nvfp4 = w2_nvfp4
        save_ffn_bf16_rescue = use_tk_localcta() and use_tk_localcta_ffn_bf16_rescue_on_zero_dy_sc()
        save_h13_requant_weights = bool(requant_h13_operands_for_bwd)
        ctx.w1_bf16 = w1_weight.detach() if (
            save_ffn_bf16_rescue
            or use_tk_localcta_ffn_bf16_dgrad_debug()
            or use_tk_debug_ffn_bf16_split_dgrad()
            or get_tk_localcta_ffn_dequant_dgrad_debug_mode() == 'activation'
            or save_h13_requant_weights
        ) else None
        ctx.w3_bf16 = w3_weight.detach() if (
            save_ffn_bf16_rescue
            or use_tk_localcta_ffn_bf16_dgrad_debug()
            or use_tk_debug_ffn_bf16_split_dgrad()
            or get_tk_localcta_ffn_dequant_dgrad_debug_mode() == 'activation'
            or save_h13_requant_weights
        ) else None
        ctx.w2_bf16 = w2_weight.detach() if (
            use_tk_localcta()
            or save_ffn_bf16_rescue
            or use_tk_localcta_ffn_bf16_w2_backward_debug()
            or use_tk_debug_ffn_bf16_w2_dgrad()
        ) else None
        ctx.w2_dgrad_col = w2_dgrad_col
        ctx.w13_dgrad_cols = w13_dgrad_cols
        ctx.x_nvfp4 = x_nvfp4    # shared normed quantized input (for wgrad)
        ctx.h_nvfp4 = (
            None if recompute_h_for_w2_wgrad else h_nvfp4
        )  # h=silu(h1_raw)*h3 quantized (for w2 wgrad)
        ctx.N_dims_13 = N_dims_13
        ctx.epsilon = epsilon
        ctx.grad_quantizer_w1 = grad_quantizer_w1
        ctx.grad_quantizer_w3 = grad_quantizer_w3
        ctx.grad_quantizer_w2 = grad_quantizer_w2
        ctx.workspace = workspace
        ctx._K = K
        ctx._H = H
        ctx._lbt_debug_name = debug_name
        ctx._ffn_weight_quant_keepalive = ffn_weight_quant_keepalive
        ctx._ffn_dgrad_weight_quant_keepalive = ffn_dgrad_weight_quant_keepalive
        ctx._recompute_h13_for_bwd = bool(recompute_h13_for_bwd)
        ctx._requant_h13_operands_for_bwd = bool(requant_h13_operands_for_bwd)
        ctx._requant_h13_activation_for_bwd = bool(
            requant_h13_activation_for_bwd
        )
        ctx._recompute_h_for_w2_wgrad = bool(recompute_h_for_w2_wgrad)
        ctx._recompute_wc_fp4_row = (
            wc_fp4_row
            if recompute_h13_for_bwd and not requant_h13_operands_for_bwd
            else None
        )
        ctx._recompute_wc_sc_row = (
            wc_sc_row
            if recompute_h13_for_bwd and not requant_h13_operands_for_bwd
            else None
        )
        ctx._recompute_fwd_b_sg = (
            fwd_b_sg
            if recompute_h13_for_bwd and not requant_h13_operands_for_bwd
            else None
        )
        ctx._recompute_cde_row_rms_partial = (
            cde_input_row_rms_partial.detach()
            if (
                requant_h13_operands_for_bwd
                or requant_h13_activation_for_bwd
            ) and ctx.cde_input
            else None
        )
        ctx._recompute_use_localcta = bool(use_localcta)

        if ctx.cde_output:
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(cde_row_rms_partial)
            return y, cde_row_rms_partial
        if ctx.h_output:
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(*h_carrier[1:])
            return h_carrier
        return y

    @staticmethod
    @v5_ffn_quant_scope()
    def backward(ctx, grad_output, *carrier_grads):
        input, norm_weight, inv_rms, h13, h1_raw, h3, sig_h1, sg_cat = ctx.saved_tensors

        wc_fp4_cols = ctx._wc_fp4_cols
        ctx._wc_fp4_cols = None
        wc_sc_cols = ctx._wc_sc_cols
        ctx._wc_sc_cols = None
        wc_sg_cols = ctx._wc_sg_cols
        ctx._wc_sg_cols = None
        dgrad_wc_fp4_cols = ctx._dgrad_wc_fp4_cols
        ctx._dgrad_wc_fp4_cols = None
        dgrad_wc_sc_cols = ctx._dgrad_wc_sc_cols
        ctx._dgrad_wc_sc_cols = None
        dgrad_wc_sg_cols = ctx._dgrad_wc_sg_cols
        ctx._dgrad_wc_sg_cols = None
        w2_nvfp4 = ctx.w2_nvfp4
        w1_bf16 = ctx.w1_bf16
        w3_bf16 = ctx.w3_bf16
        w2_bf16 = ctx.w2_bf16
        x_nvfp4 = ctx.x_nvfp4
        h_nvfp4 = ctx.h_nvfp4
        ctx._ffn_weight_quant_keepalive = None
        ctx._ffn_dgrad_weight_quant_keepalive = None
        if dgrad_wc_fp4_cols is None:
            dgrad_wc_fp4_cols = wc_fp4_cols
        if dgrad_wc_sc_cols is None:
            dgrad_wc_sc_cols = wc_sc_cols
        if dgrad_wc_sg_cols is None:
            dgrad_wc_sg_cols = wc_sg_cols
        N_dims_13 = ctx.N_dims_13
        K = ctx._K
        H = ctx._H
        M = input.shape[0]
        debug_name = getattr(ctx, '_lbt_debug_name', None)

        _tk_stage_trace('ffn_bwd', 'start', debug_name)
        _tk_debug_print('ffn_bwd', 'start', debug_name)
        fuse_rms_residual_bwd = (
            getattr(ctx, '_fuse_rms_residual_bwd', False)
            and grad_output.dtype == torch.bfloat16
            and grad_output.is_contiguous()
        )
        if getattr(ctx, '_fuse_rms_residual_bwd', False):
            _trace_backend_choice(
                'ffn_rms_residual_grad_contract',
                (
                    f"dtype={grad_output.dtype},"
                    f"contiguous={int(grad_output.is_contiguous())},"
                    f"stride={tuple(grad_output.stride())}"
                ),
            )
        if fuse_rms_residual_bwd:
            from .tk_gemm import _get_native_rmsnorm_bwd_residual_out
            if _get_native_rmsnorm_bwd_residual_out() is None:
                raise RuntimeError(
                    'USE_TK_FFN_RMS_RESIDUAL_BWD requires '
                    'rmsnorm_bwd_residual_out before FFN backward work'
                )
            _trace_backend_choice('ffn_rms_residual_bwd', 'native_tk')
        if getattr(ctx, '_recompute_h13_for_bwd', False):
            x_h13 = x_nvfp4
            h13_requant_keepalive = None
            if getattr(ctx, '_requant_h13_operands_for_bwd', False):
                if w1_bf16 is None or w3_bf16 is None:
                    raise RuntimeError(
                        "USE_TK_FFN_REQUANT_H13_OPERANDS lost BF16 W1/W3 handles"
                    )
                from .tk_gemm import _get_tk_quant_for_gemm

                tk_q = _get_tk_quant_for_gemm()
                cde_row_rms_partial = getattr(
                    ctx, '_recompute_cde_row_rms_partial', None
                )
                if cde_row_rms_partial is not None:
                    result = tk_q.tk_rmsnorm_quantize_from_row_rms_partial_final_sg(
                        input,
                        norm_weight,
                        cde_row_rms_partial,
                        float(ctx.epsilon),
                        True,
                        use_nvfp4_encode_centric(),
                    )
                    x_h13 = _TKQuantized(
                        result[0], result[1], result[4],
                        result[2], result[3], result[5],
                        keepalive=_result_keepalive(result, 7),
                    )
                else:
                    use_row_prepared_rmsnorm_quant = (
                        use_tk_ffn_localcta_tk_quant_contract()
                        and get_tk_localcta_variant() == 'v4'
                        and use_tk_localcta_v4_row_prepared_col_outer()
                        and use_tk_localcta_v4_row_prepared_rmsnorm_quant()
                        and M % 256 == 0
                        and K % 256 == 0
                    )
                    x_h13, _ = _fast_rmsnorm_quantize_localcta_v4_opt(
                        input,
                        norm_weight,
                        float(ctx.epsilon),
                        nvfp4_role="activation",
                        prefer_row_prepared_col_outer=use_row_prepared_rmsnorm_quant,
                        separate_bf16_final_sg=(
                            use_tk_localcta_v4_ffn_separate_bf16_final_sg()
                        ),
                    )
                ctx._recompute_cde_row_rms_partial = None
                x_nvfp4 = x_h13
                h13_requant_result = _tk_group_quantize_ffn_weights(
                    tk_q,
                    w1_bf16,
                    w3_bf16,
                    N_dims_13,
                    prefer_split=(M >= 256),
                )
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, sg_cat, _, wc_sg_cols = \
                    h13_requant_result[:8]
                dgrad_wc_fp4_cols = wc_fp4_cols
                dgrad_wc_sc_cols = wc_sc_cols
                dgrad_wc_sg_cols = wc_sg_cols
                h13_requant_keepalive = _result_keepalive(
                    h13_requant_result, 8
                )
            else:
                if getattr(ctx, '_requant_h13_activation_for_bwd', False):
                    cde_row_rms_partial = getattr(
                        ctx, '_recompute_cde_row_rms_partial', None
                    )
                    if cde_row_rms_partial is not None:
                        from .tk_gemm import _get_tk_quant_for_gemm

                        tk_q = _get_tk_quant_for_gemm()
                        result = tk_q.tk_rmsnorm_quantize_from_row_rms_partial_final_sg(
                            input,
                            norm_weight,
                            cde_row_rms_partial,
                            float(ctx.epsilon),
                            True,
                            use_nvfp4_encode_centric(),
                        )
                        x_h13 = _TKQuantized(
                            result[0], result[1], result[4],
                            result[2], result[3], result[5],
                            keepalive=_result_keepalive(result, 7),
                        )
                    else:
                        tk_q = _get_tk_quant()
                        result = tk_q.tk_fused_norm_quantize(
                            input,
                            norm_weight,
                            float(ctx.epsilon),
                            False,
                            True,
                        )
                        x_h13 = _TKQuantized(
                            result[0], result[1], result[4],
                            result[2], result[3],
                            keepalive=_result_keepalive(result, 6),
                        )
                    ctx._recompute_cde_row_rms_partial = None
                wc_fp4_row = ctx._recompute_wc_fp4_row
                wc_sc_row = ctx._recompute_wc_sc_row
                fwd_b_sg = ctx._recompute_fwd_b_sg
                if wc_fp4_row is None or wc_sc_row is None or fwd_b_sg is None:
                    raise RuntimeError(
                        "USE_TK_FFN_RECOMPUTE_H13 lost FFN forward quantized weights"
                    )
            h1_raw, h3 = _get_ffn_localcta_h13_recompute_buffers(
                M, K, H, input.device
            )
            tk_grouped_forward_gemm_split(
                x_h13,
                wc_fp4_row,
                wc_sc_row,
                fwd_b_sg,
                N_dims_13,
                outs=[h1_raw, h3],
                use_localcta=getattr(ctx, '_recompute_use_localcta', False),
            )
            ctx._recompute_wc_fp4_row = None
            ctx._recompute_wc_sc_row = None
            ctx._recompute_fwd_b_sg = None
            wc_fp4_row = wc_sc_row = fwd_b_sg = None
            h13_requant_result = None

        if getattr(ctx, '_recompute_h_for_w2_wgrad', False):
            if h1_raw.numel() == 0 or h3.numel() == 0:
                raise RuntimeError(
                    "USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD requires recomputed H1/H3"
                )
            from .tk_gemm import _get_tk_quant_for_gemm

            h_nvfp4 = _localcta_silu_quantize_split_for_gemm(
                _get_tk_quant_for_gemm(),
                h1_raw,
                h3,
            )
            _release_tk_row_storage(h_nvfp4)
            _trace_backend_choice('localcta_ffn_bwd', 'recomputed_h_for_w2_wgrad')

        with localcta_v4_cpp_only_scope(getattr(ctx, '_ffn_v4_cpp_only', False)):
            grad_input, grad_w1, grad_w3, grad_w2, grad_norm_weight, rescue_info = _ffn_bwd_graphed(
                grad_output, input, norm_weight, inv_rms,
                h13, sg_cat,
                wc_fp4_cols, wc_sc_cols, w2_nvfp4, x_nvfp4, h_nvfp4, w1_bf16, w3_bf16, w2_bf16,
                N_dims_13, K, H, M, ctx.workspace,
                w2_dgrad_col=getattr(ctx, 'w2_dgrad_col', None),
                w13_dgrad_cols=getattr(ctx, 'w13_dgrad_cols', None),
                h1_raw=h1_raw, h3=h3, sig_h1=sig_h1,
                wc_sg_cols=wc_sg_cols,
                dgrad_wc_fp4_cols=dgrad_wc_fp4_cols,
                dgrad_wc_sc_cols=dgrad_wc_sc_cols,
                dgrad_wc_sg_cols=dgrad_wc_sg_cols,
                debug_name=debug_name,
                residual_grad=grad_output if fuse_rms_residual_bwd else None,
                h_tile=getattr(ctx, 'h_tile', False),
            )
        _tk_stage_trace('ffn_bwd', 'end', debug_name)
        _tk_debug_print('ffn_bwd', 'end', debug_name)
        if _should_emit_localcta_function_grad_debug(debug_name):
            _emit_localcta_function_grad_debug(
                "ffn_backward_return",
                {
                    "debug_name": debug_name,
                    "grad_output": _tensor_debug_stats(grad_output),
                    "grad_input": _tensor_debug_stats(grad_input),
                    "grad_w1": _tensor_debug_stats(grad_w1),
                    "grad_w3": _tensor_debug_stats(grad_w3),
                    "grad_w2": _tensor_debug_stats(grad_w2),
                    "grad_norm_weight": _tensor_debug_stats(grad_norm_weight),
                    "rescue": rescue_info,
                },
            )


        # Free cached tensors
        ctx.w2_nvfp4 = None
        ctx.w1_bf16 = None
        ctx.w3_bf16 = None
        ctx.w2_bf16 = None
        ctx.w2_dgrad_col = None
        ctx.w13_dgrad_cols = None
        ctx.x_nvfp4 = None
        ctx.h_nvfp4 = None

        if _clone_localcta_v4_ffn_backward_returns():
            grad_input = grad_input.clone()
            grad_w1 = grad_w1.clone()
            grad_w3 = grad_w3.clone()
            grad_w2 = grad_w2.clone()
            grad_norm_weight = grad_norm_weight.clone()

        grad_residual = (
            None if fuse_rms_residual_bwd
            else (grad_output if ctx.has_residual else None)
        )

        # 28 inputs to forward
        return (
            grad_input,        # input
            grad_w1,           # w1_weight
            grad_w3,           # w3_weight
            grad_w2,           # w2_weight
            grad_norm_weight,  # norm_weight
            None,              # epsilon
            None,              # w1_weight_quantizer
            None, None,        # w3 quantizers
            None, None,        # w2 quantizers
            None, None, None,  # grad quantizers
            None,              # w1_input_quantizer
            None,              # workspace
            None,              # debug_name
            grad_residual,     # residual
            None, None, None, None, None, None, None,  # H input carrier
            None,              # h_next_gamma
            None,              # cde_row_rms_partial
            None,              # cde_emit
        )


class _FusedSquaredReLUFFNFunctionV2_TK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
        workspace: torch.Tensor,
        debug_name: Optional[str] = None,
        residual: Optional[torch.Tensor] = None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
        cde_emit: bool = False,
    ):
        M, K = input.shape
        H = w1_weight.shape[0]
        N = w2_weight.shape[0]
        residual_2d = None
        if torch.is_tensor(residual):
            if tuple(residual.shape) != (M, N):
                raise RuntimeError(
                    f"Square-ReLU FFN residual shape {tuple(residual.shape)} does not match output {(M, N)}"
                )
            residual_2d = _as_contiguous_bf16(residual)
        ctx.has_residual = residual_2d is not None
        ctx._ffn_v4_cpp_only = use_tk_localcta_v4_ffn_cpp_only()

        inp = _as_contiguous_bf16(input)
        nw = _as_contiguous_bf16(norm_weight.detach())
        te_fused = _get_te_fused()
        use_localcta = use_tk_localcta_forward_for_m(M)
        cde_input = (
            torch.is_tensor(cde_row_rms_partial)
            and cde_row_rms_partial.numel() != 0
        )
        if cde_input and (
            not use_localcta
            or get_tk_localcta_variant() != "v4"
            or (M, K, H) != (24576, 4096, 21504)
            or cde_row_rms_partial.dtype != torch.float32
            or not cde_row_rms_partial.is_cuda
            or not cde_row_rms_partial.is_contiguous()
            or tuple(cde_row_rms_partial.shape) != (M, K // 256)
        ):
            raise RuntimeError(
                "Nemotron square-ReLU CDE input requires localCTA v4 at "
                "MKH=(24576,4096,21504) with contiguous FP32 [M,K/256] partials"
            )
        if cde_emit and (
            not use_localcta
            or get_tk_localcta_variant() != "v4"
            or residual_2d is None
            or (M, N, H) != (24576, 4096, 21504)
        ):
            raise RuntimeError(
                "Nemotron square-ReLU CDE emission requires localCTA v4 "
                "residual W2 at MNK=(24576,4096,21504)"
            )

        _tk_stage_trace('ffn_sqrelu_fwd', 'start', debug_name)
        with localcta_v4_cpp_only_scope(ctx._ffn_v4_cpp_only):
            overlap_w2_weight_quant = (
                use_localcta
                and use_localcta_sqrelu_w2_weight_quant_overlap(
                    debug_name,
                    M,
                    N,
                    K,
                    H,
                )
            )
            caller_stream = None
            w2_weight_quant_stream = None
            w2_nvfp4 = None
            if overlap_w2_weight_quant:
                caller_stream = torch.cuda.current_stream(inp.device)
                w2_weight_quant_stream = _get_ms_stream()
                w2_weight_quant_stream.wait_stream(caller_stream)
                with torch.cuda.stream(w2_weight_quant_stream):
                    _record_tensors_on_stream(
                        (w2_weight,),
                        w2_weight_quant_stream,
                    )
                    w2_nvfp4 = _fast_quantize(
                        w2_weight,
                        tk_swizzle=True,
                        use_localcta_override=True,
                        nvfp4_role="weight",
                    )

            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'x_quant_start', debug_name)
            if cde_input:
                from .tk_gemm import _get_tk_quant_for_gemm

                tk_q_cde = _get_tk_quant_for_gemm()
                result = (
                    tk_q_cde
                    .tk_rmsnorm_quantize_from_row_rms_partial_final_sg(
                        inp,
                        nw,
                        cde_row_rms_partial,
                        float(epsilon),
                        True,
                        use_nvfp4_encode_centric(),
                    )
                )
                x_nvfp4 = _TKQuantized(
                    result[0],
                    result[1],
                    result[4],
                    result[2],
                    result[3],
                    result[5],
                    keepalive=_result_keepalive(result, 7),
                )
                inv_rms = result[6]
                _trace_backend_choice(
                    "localcta_nemotron_interlayer_cde",
                    "native",
                )
            elif (
                use_localcta
                and get_tk_localcta_variant() == 'v4'
                and _nvfp4_quantizer_extras_enabled("activation")
            ):
                x_nvfp4, inv_rms = _fast_rmsnorm_quantize_localcta_v4_opt(
                    inp,
                    nw,
                    float(epsilon),
                    nvfp4_role="activation",
                )
            elif _can_fast_rmsnorm_quantize_tk_regular_opt(inp, "activation"):
                x_nvfp4, inv_rms = _fast_rmsnorm_quantize_tk_regular_opt(
                    inp,
                    nw,
                    float(epsilon),
                    nvfp4_role="activation",
                )
            elif (
                use_localcta
                and use_tk_localcta_fused()
                and get_tk_localcta_variant() == 'v4'
                and use_tk_localcta_v4_fast_ffn_fused_norm()
                and not _nvfp4_quantizer_extras_enabled("activation")
            ):
                from .tk_gemm import _get_tk_quant_for_gemm

                tk_q = _get_tk_quant_for_gemm()
                if hasattr(tk_q, 'tk_fused_norm_quantize'):
                    result = tk_q.tk_fused_norm_quantize(
                        inp,
                        nw,
                        float(epsilon),
                        False,
                        True,
                    )
                    x_nvfp4 = _TKQuantized(
                        result[0], result[1], result[4],
                        result[2], result[3], result[5],
                    )
                    inv_rms = result[6]
                else:
                    normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                    x_nvfp4 = _fast_quantize(
                        normed,
                        tk_swizzle=True,
                        use_localcta_override=use_localcta,
                        nvfp4_role="activation",
                    )
            else:
                normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                x_nvfp4 = _fast_quantize(
                    normed,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                    nvfp4_role="activation",
                )
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'x_quant_done', debug_name)

            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w1_quant_start', debug_name)
            w1_nvfp4 = _fast_quantize(
                w1_weight,
                tk_swizzle=True,
                use_localcta_override=use_localcta,
                nvfp4_role="weight",
            )
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w1_quant_done', debug_name)
            h1_raw = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w1_gemm_start', debug_name)
            tk_forward_gemm(x_nvfp4, w1_nvfp4, h1_raw, use_localcta=use_localcta)
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w1_gemm_done', debug_name)
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'sqrelu_quant_start', debug_name)
            h_nvfp4 = (
                _fast_sqrelu_quantize_localcta_v4(h1_raw, nvfp4_role="activation")
                if use_localcta
                else _fast_sqrelu_quantize_tk_regular_opt(h1_raw, nvfp4_role="activation")
            )
            if h_nvfp4 is None:
                h = sqrelu_fwd(h1_raw)
                h_nvfp4 = _fast_quantize(
                    h,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                    nvfp4_role="activation",
                )
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'sqrelu_quant_done', debug_name)
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w2_quant_start', debug_name)
            if w2_weight_quant_stream is not None:
                assert caller_stream is not None
                caller_stream.wait_stream(w2_weight_quant_stream)
                _record_tensors_on_stream(
                    (
                        w2_nvfp4._tk_row,
                        w2_nvfp4._tk_col,
                        w2_nvfp4._tk_row_chunk_sg,
                        w2_nvfp4._tk_col_chunk_sg,
                        w2_nvfp4._keepalive,
                    ),
                    caller_stream,
                )
            else:
                w2_nvfp4 = _fast_quantize(
                    w2_weight,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                    nvfp4_role="weight",
                )
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w2_quant_done', debug_name)
            y = torch.empty((M, N), dtype=torch.bfloat16, device=inp.device)
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w2_gemm_start', debug_name)
            row_rms_partial = None
            if cde_emit:
                y, row_rms_partial = tk_forward_gemm_residual_rms_partial(
                    h_nvfp4,
                    w2_nvfp4,
                    residual_2d,
                    y,
                    use_localcta=True,
                )
            elif residual_2d is not None:
                tk_forward_gemm_residual(h_nvfp4, w2_nvfp4, residual_2d, y, use_localcta=use_localcta)
            else:
                tk_forward_gemm(h_nvfp4, w2_nvfp4, y, use_localcta=use_localcta)
            _tk_stage_trace('ffn_sqrelu_fwd_sub', 'w2_gemm_done', debug_name)
        _tk_stage_trace('ffn_sqrelu_fwd', 'end', debug_name)

        ctx.save_for_backward(inp, nw, inv_rms, h1_raw)
        ctx.x_nvfp4 = x_nvfp4
        ctx.h_nvfp4 = h_nvfp4
        ctx.w1_nvfp4 = w1_nvfp4
        ctx.w2_nvfp4 = w2_nvfp4
        ctx.workspace = workspace
        ctx._lbt_debug_name = debug_name
        ctx.cde_output = bool(cde_emit)
        if ctx.cde_output:
            ctx.mark_non_differentiable(row_rms_partial)
            return y, row_rms_partial
        return y

    @staticmethod
    def backward(ctx, grad_output, grad_row_rms_partial=None):
        input, norm_weight, inv_rms, h1_raw = ctx.saved_tensors
        x_nvfp4 = ctx.x_nvfp4
        h_nvfp4 = ctx.h_nvfp4
        w1_nvfp4 = ctx.w1_nvfp4
        w2_nvfp4 = ctx.w2_nvfp4
        debug_name = getattr(ctx, '_lbt_debug_name', None)
        use_localcta = use_tk_localcta_forward_for_m(grad_output.shape[0])

        _tk_stage_trace('ffn_sqrelu_bwd', 'start', debug_name)
        with localcta_v4_cpp_only_scope(getattr(ctx, '_ffn_v4_cpp_only', False)):
            dY = _as_contiguous_bf16(grad_output)
            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'dy_quant_start', debug_name)
            dY_nvfp4 = _fast_quantize(
                dY,
                tk_swizzle=True,
                use_localcta_override=use_localcta,
                nvfp4_role="grad",
            )
            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'dy_quant_done', debug_name)

            pending_wgrad_stream = None
            delayed_h_col = (
                use_localcta
                and torch.is_tensor(h_nvfp4._tk_col[0])
                and h_nvfp4._tk_col[0].numel() == 0
            )
            if (
                delayed_h_col
                and use_tk_sqrelu_ffn_overlap_w2_wgrad_deriv()
                and torch.cuda.is_available()
                and dY.is_cuda
            ):
                pending_wgrad_stream = _get_wgrad_stream()
                pending_wgrad_stream.wait_stream(torch.cuda.current_stream())
                _record_tensors_on_stream((h1_raw,), pending_wgrad_stream)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'h_col_quant_start', debug_name)
                with torch.cuda.stream(pending_wgrad_stream):
                    _materialize_sqrelu_col_localcta_v4(h_nvfp4, h1_raw)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'h_col_quant_done', debug_name)

            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w2_dgrad_start', debug_name)
            dh = tk_dgrad_gemm(dY_nvfp4, w2_nvfp4, use_localcta=use_localcta)
            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w2_dgrad_done', debug_name)
            overlap_w2_deriv = (
                use_tk_sqrelu_ffn_overlap_w2_wgrad_deriv()
                and torch.cuda.is_available()
                and dh.is_cuda
            )
            if delayed_h_col and pending_wgrad_stream is None:
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'h_col_quant_start', debug_name)
                _materialize_sqrelu_col_localcta_v4(h_nvfp4, h1_raw)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'h_col_quant_done', debug_name)
            if overlap_w2_deriv:
                if pending_wgrad_stream is None:
                    pending_wgrad_stream = _get_wgrad_stream()
                    pending_wgrad_stream.wait_stream(torch.cuda.current_stream())
                _record_tensors_on_stream((h_nvfp4._tk_col, dY_nvfp4._tk_col), pending_wgrad_stream)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w2_wgrad_start', debug_name)
                with torch.cuda.stream(pending_wgrad_stream):
                    grad_w2 = tk_wgrad_gemm(h_nvfp4, dY_nvfp4, use_localcta=use_localcta)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w2_wgrad_done', debug_name)
            else:
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w2_wgrad_start', debug_name)
                grad_w2 = tk_wgrad_gemm(h_nvfp4, dY_nvfp4, use_localcta=use_localcta)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w2_wgrad_done', debug_name)
            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'sqrelu_deriv_quant_start', debug_name)
            dh1_nvfp4 = (
                _fast_sqrelu_deriv_quantize_localcta_v4(dh, h1_raw, nvfp4_role="grad")
                if use_localcta
                else _fast_sqrelu_deriv_quantize_tk_regular_opt(dh, h1_raw, nvfp4_role="grad")
            )
            if dh1_nvfp4 is None:
                dh1 = sqrelu_bwd(dh, h1_raw)
                dh1_nvfp4 = _fast_quantize(
                    dh1,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                    nvfp4_role="grad",
                )
            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'sqrelu_deriv_quant_done', debug_name)
            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w1_dgrad_start', debug_name)
            dx_normed = tk_dgrad_gemm(dh1_nvfp4, w1_nvfp4, use_localcta=use_localcta)
            _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w1_dgrad_done', debug_name)
            te_fused = _get_te_fused()
            overlap_w1_rms = (
                use_tk_sqrelu_ffn_overlap_w1_wgrad_rms()
                and torch.cuda.is_available()
                and dx_normed.is_cuda
            )
            use_cached_rms = use_tk_sqrelu_ffn_cached_rms_bwd()
            if overlap_w1_rms:
                wgrad_stream = pending_wgrad_stream or _get_wgrad_stream()
                wgrad_stream.wait_stream(torch.cuda.current_stream())
                _record_tensors_on_stream((x_nvfp4._tk_col, dh1_nvfp4._tk_col), wgrad_stream)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w1_wgrad_start', debug_name)
                with torch.cuda.stream(wgrad_stream):
                    grad_w1 = tk_wgrad_gemm(x_nvfp4, dh1_nvfp4, use_localcta=use_localcta)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w1_wgrad_done', debug_name)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'rmsnorm_start', debug_name)
                if use_cached_rms:
                    rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                        dx_normed, input, norm_weight, inv_rms, te_fused,
                        tag='sqrelu_ffn',
                    )
                    torch.cuda.current_stream().wait_stream(rms_stream)
                    grad_input = rms_state['grad_input']
                    dgamma = rms_state.get('dgamma_out', rms_state['dgamma'])
                else:
                    grad_input, dgamma = te_fused.fused_rmsnorm_backward(
                        dx_normed.contiguous(),
                        input.contiguous(),
                        norm_weight,
                        inv_rms,
                    )
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'rmsnorm_done', debug_name)
                torch.cuda.current_stream().wait_stream(wgrad_stream)
            else:
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w1_wgrad_start', debug_name)
                grad_w1 = tk_wgrad_gemm(x_nvfp4, dh1_nvfp4, use_localcta=use_localcta)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'w1_wgrad_done', debug_name)
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'rmsnorm_start', debug_name)
                if use_cached_rms:
                    rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                        dx_normed, input, norm_weight, inv_rms, te_fused,
                        tag='sqrelu_ffn',
                    )
                    torch.cuda.current_stream().wait_stream(rms_stream)
                    grad_input = rms_state['grad_input']
                    dgamma = rms_state.get('dgamma_out', rms_state['dgamma'])
                else:
                    grad_input, dgamma = te_fused.fused_rmsnorm_backward(
                        dx_normed.contiguous(),
                        input.contiguous(),
                        norm_weight,
                        inv_rms,
                    )
                _tk_stage_trace('ffn_sqrelu_bwd_sub', 'rmsnorm_done', debug_name)
                if pending_wgrad_stream is not None:
                    torch.cuda.current_stream().wait_stream(pending_wgrad_stream)
        _tk_stage_trace('ffn_sqrelu_bwd', 'end', debug_name)

        ctx.x_nvfp4 = None
        ctx.h_nvfp4 = None
        ctx.w1_nvfp4 = None
        ctx.w2_nvfp4 = None
        grad_norm_weight = _as_param_grad_dtype(dgamma, norm_weight)
        grad_residual = grad_output if ctx.has_residual else None
        return (
            grad_input,
            grad_w1,
            grad_w2,
            grad_norm_weight,
            None,
            None,
            None,
            grad_residual,
            None,
            None,
        )


def _sqrelu_ffn_projection_pair(ffn):
    if hasattr(ffn, "w1") and hasattr(ffn, "w2"):
        return ffn.w1, ffn.w2
    if hasattr(ffn, "up_proj") and hasattr(ffn, "down_proj"):
        return ffn.up_proj, ffn.down_proj
    raise AttributeError(
        "square-ReLU FFN fusion expects w1/w2 or up_proj/down_proj linear projections"
    )


class FusedSquaredReLUFeedForwardFP4_TK(nn.Module):
    """Fused TK/localCTA FP4 square-ReLU FFN for two-projection paper models."""

    def __init__(self, dim, hidden_dim, norm_eps=1e-5,
                 bias=False, device=None, dtype=torch.bfloat16,
                 recipe=None):
        super().__init__()
        if bias:
            raise NotImplementedError("FusedSquaredReLUFeedForwardFP4_TK does not support bias")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epsilon = norm_eps
        self.norm_weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.w1_weight = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(dim, hidden_dim, device=device, dtype=dtype))
        self._workspace = None
        self._workspace_device = None
        self.init_weights()

    def _ensure_workspace(self, device):
        if self._workspace_device != device:
            self._workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._workspace_device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_residual(x, residual=None)

    def forward_with_residual(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
        cde_emit: bool = False,
    ) -> torch.Tensor:
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x_2d = x.reshape(B * S, H)
            residual_2d = residual.reshape(B * S, self.dim) if residual is not None else None
        else:
            x_2d = x
            residual_2d = residual

        self._ensure_workspace(x.device)
        out = _FusedSquaredReLUFFNFunctionV2_TK.apply(
            x_2d,
            self.w1_weight,
            self.w2_weight,
            self.norm_weight,
            self.epsilon,
            self._workspace,
            getattr(self, '_lbt_debug_name', None),
            residual_2d,
            cde_row_rms_partial,
            cde_emit,
        )
        if is_3d:
            if isinstance(out, tuple):
                return out[0].view(B, S, self.dim), out[1]
            return out.view(B, S, self.dim)
        return out

    def invalidate_weight_cache(self):
        pass

    def init_weights(self, init_std: float = 0.02):
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self.w1_weight, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.w2_weight, mean=0.0, std=init_std)

    @classmethod
    def from_unfused(cls, ffn, norm, recipe=None):
        up_proj, down_proj = _sqrelu_ffn_projection_pair(ffn)
        dim = up_proj.in_features
        hidden_dim = up_proj.out_features
        device = up_proj.weight.device
        dtype = up_proj.weight.dtype
        eps = getattr(norm, 'eps', None)
        if eps is None:
            eps = 1e-5

        fused = cls(
            dim, hidden_dim, norm_eps=eps,
            bias=getattr(up_proj, "bias", None) is not None,
            device=device, dtype=dtype, recipe=recipe,
        )
        if device.type != 'meta':
            with torch.no_grad():
                fused.w1_weight.copy_(up_proj.weight)
                fused.w2_weight.copy_(down_proj.weight)
                if hasattr(norm, 'weight') and norm.weight is not None:
                    fused.norm_weight.copy_(norm.weight)
        return fused


# ---------------------------------------------------------------------------
# FusedFeedForwardFP4: replaces FeedForward + ffn_norm
# ---------------------------------------------------------------------------
class FusedFeedForwardFP4_TE(nn.Module):
    """Fused FP4 FeedForward with custom autograd using fused CUDA kernels.

    Architecture: w2(silu(w1(rms_norm(x))) * w3(rms_norm(x)))

    Fusion strategy:
      Forward:  fused CUDA kernel: rms_norm + fp4_quant (shared) → GEMM w1, GEMM w3
                fused CUDA kernel: silu(h1)*h3 + fp4_quant       → GEMM w2
      Backward: tex.generic_gemm for all dgrads/wgrads
                fused silu'+mul element-wise kernel for dh1/dh3
                fused_rmsnorm_backward for dx + dgamma
    """

    def __init__(self, dim, hidden_dim, norm_eps=1e-5,
                 bias=False, device=None, dtype=torch.bfloat16,
                 recipe=None, packed_w13: bool = False):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epsilon = norm_eps
        self.packed_w13 = packed_w13

        # Single shared norm weight
        self.norm_weight = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype)
        )

        # Raw weight parameters (not TELinearFP4 — we handle quantization).
        # Bridge/MCore stores SwiGLU fc1 as one packed [w1; w3] parameter; keep
        # that layout available so DDP/optimizer bucketization matches MCore.
        if packed_w13:
            self.w13_weight = nn.Parameter(
                torch.empty(2 * hidden_dim, dim, device=device, dtype=dtype)
            )
        else:
            self.w1_weight = nn.Parameter(
                torch.empty(hidden_dim, dim, device=device, dtype=dtype)
            )
            self.w3_weight = nn.Parameter(
                torch.empty(hidden_dim, dim, device=device, dtype=dtype)
            )
        self.w2_weight = nn.Parameter(
            torch.empty(dim, hidden_dim, device=device, dtype=dtype)
        )

        # Initialize weights
        if packed_w13:
            _safe_trunc_normal_(self.w13_weight[:hidden_dim], mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w13_weight[hidden_dim:], mean=0.0, std=0.02)
        else:
            _safe_trunc_normal_(self.w1_weight, mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w3_weight, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.w2_weight, mean=0.0, std=0.02)

        # TE quantizers (one per GEMM input/weight/grad)
        te_dtype = tex.DType.kFloat4E2M1

        def _make_quantizer(role: str):
            return _make_nvfp4_quantizer_for_role(role, te_dtype)

        # w1 quantizers
        self.w1_input_quantizer = _make_quantizer("activation")  # dummy for NVFP4Tensor wrapper
        self.w1_weight_quantizer = _make_quantizer("weight")
        # w3 quantizers
        self.w3_input_quantizer = _make_quantizer("activation")
        self.w3_weight_quantizer = _make_quantizer("weight")
        # w2 quantizers
        self.w2_input_quantizer = _make_quantizer("activation")
        self.w2_weight_quantizer = _make_quantizer("weight")
        # grad quantizers
        self.grad_quantizer_w1 = _make_quantizer("grad")
        self.grad_quantizer_w3 = _make_quantizer("grad")
        self.grad_quantizer_w2 = _make_quantizer("grad")

        # Workspace (lazy init)
        self._workspace = None
        self._workspace_device = None

    def _ensure_workspace(self, device):
        if self._workspace_device != device:
            self._workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._workspace_device = device

    def _w1_weight_view(self):
        if self.packed_w13:
            return self.w13_weight[:self.hidden_dim]
        return self.w1_weight

    def _w3_weight_view(self):
        if self.packed_w13:
            return self.w13_weight[self.hidden_dim:]
        return self.w3_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x_2d = x.reshape(B * S, H)
        else:
            x_2d = x

        self._ensure_workspace(x.device)

        out = _FusedFFNFunctionV2_TE.apply(
            x_2d, self._w1_weight_view(), self._w3_weight_view(), self.w2_weight,
            self.norm_weight, self.epsilon,
            self.w1_weight_quantizer,
            self.w3_input_quantizer, self.w3_weight_quantizer,
            self.w2_input_quantizer, self.w2_weight_quantizer,
            self.grad_quantizer_w1, self.grad_quantizer_w3,
            self.grad_quantizer_w2,
            self.w1_input_quantizer,
            self._workspace,
        )

        if is_3d:
            return out.view(B, S, self.dim)
        return out

    def invalidate_weight_cache(self):
        """For benchmark API compatibility."""
        pass

    def init_weights(self, init_std: float = 0.02):
        """Initialize weights, matching Llama's FeedForward.init_weights().

        NOTE: norm_weight MUST be reset to ones here because the model is
        constructed on meta device and init_weights() is called after FSDP
        materialization. Without this, norm_weight contains uninitialized data.
        """
        nn.init.ones_(self.norm_weight)
        if self.packed_w13:
            _safe_trunc_normal_(self.w13_weight[:self.hidden_dim], mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w13_weight[self.hidden_dim:], mean=0.0, std=init_std)
        else:
            _safe_trunc_normal_(self.w1_weight, mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w3_weight, mean=0.0, std=init_std)
        _safe_trunc_normal_(self.w2_weight, mean=0.0, std=init_std)

    @classmethod
    def from_unfused(cls, ffn, norm, recipe=None):
        """Create from an existing FeedForward module + RMSNorm.

        Args:
            ffn: FeedForward module with .w1, .w2, .w3 (nn.Linear)
            norm: nn.RMSNorm (the ffn_norm being absorbed)
            recipe: NVFP4BlockScaling recipe for w2
        """
        dim = ffn.w1.in_features
        hidden_dim = ffn.w1.out_features
        device = ffn.w1.weight.device
        dtype = ffn.w1.weight.dtype
        eps = getattr(norm, 'eps', None)
        if eps is None:
            eps = 1e-5

        fused = cls(
            dim, hidden_dim, norm_eps=eps, bias=False,
            device=device, dtype=dtype, recipe=recipe,
        )

        # Copy weights
        if device.type != 'meta':
            with torch.no_grad():
                fused._w1_weight_view().copy_(ffn.w1.weight)
                fused._w3_weight_view().copy_(ffn.w3.weight)
                fused.w2_weight.copy_(ffn.w2.weight)

                # Copy norm weight
                if hasattr(norm, 'weight') and norm.weight is not None:
                    fused.norm_weight.copy_(norm.weight)

        return fused

class FusedFeedForwardFP4_TK(nn.Module):
    """Fused FP4 FeedForward with custom autograd using fused CUDA kernels (TK path).

    Architecture: w2(silu(w1(rms_norm(x))) * w3(rms_norm(x)))

    Fusion strategy:
      Forward:  fused CUDA kernel: rms_norm (no silu, shared) → FP4 quant → TK GEMM w1, w3
                fused CUDA kernel: silu(h1)*h3 + fp4_quant    → TK GEMM w2
      Backward: TK nvfp4_gemm for all dgrads/wgrads
                fused silu'+mul element-wise kernel for dh1/dh3
                fused_rmsnorm_backward for dx + dgamma
    """

    def __init__(self, dim, hidden_dim, norm_eps=1e-5,
                 bias=False, device=None, dtype=torch.bfloat16,
                 recipe=None, packed_w13: bool = False):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epsilon = norm_eps
        self.packed_w13 = packed_w13

        # Single shared norm weight
        self.norm_weight = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype)
        )

        # Raw weight parameters (not TELinearFP4 — we handle quantization).
        if packed_w13:
            self.w13_weight = nn.Parameter(
                torch.empty(2 * hidden_dim, dim, device=device, dtype=dtype)
            )
        else:
            self.w1_weight = nn.Parameter(
                torch.empty(hidden_dim, dim, device=device, dtype=dtype)
            )
            self.w3_weight = nn.Parameter(
                torch.empty(hidden_dim, dim, device=device, dtype=dtype)
            )
        self.w2_weight = nn.Parameter(
            torch.empty(dim, hidden_dim, device=device, dtype=dtype)
        )

        # Initialize weights
        if packed_w13:
            _safe_trunc_normal_(self.w13_weight[:hidden_dim], mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w13_weight[hidden_dim:], mean=0.0, std=0.02)
        else:
            _safe_trunc_normal_(self.w1_weight, mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w3_weight, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.w2_weight, mean=0.0, std=0.02)

        # TE quantizers (one per GEMM input/weight/grad)
        te_dtype = None if (use_tk_quant() or use_tk_localcta()) else tex.DType.kFloat4E2M1

        def _make_quantizer(role: str):
            return _make_nvfp4_quantizer_for_role(role, te_dtype)

        # w1 quantizers
        self.w1_input_quantizer = _make_quantizer("activation")  # dummy for NVFP4Tensor wrapper
        self.w1_weight_quantizer = _make_quantizer("weight")
        # w3 quantizers
        self.w3_input_quantizer = _make_quantizer("activation")
        self.w3_weight_quantizer = _make_quantizer("weight")
        # w2 quantizers
        self.w2_input_quantizer = _make_quantizer("activation")
        self.w2_weight_quantizer = _make_quantizer("weight")
        # grad quantizers
        self.grad_quantizer_w1 = _make_quantizer("grad")
        self.grad_quantizer_w3 = _make_quantizer("grad")
        self.grad_quantizer_w2 = _make_quantizer("grad")

        # Workspace (lazy init)
        self._workspace = None
        self._workspace_device = None

    def _ensure_workspace(self, device):
        if self._workspace_device != device:
            self._workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._workspace_device = device

    def _w1_weight_view(self):
        if self.packed_w13:
            return self.w13_weight[:self.hidden_dim]
        return self.w1_weight

    def _w3_weight_view(self):
        if self.packed_w13:
            return self.w13_weight[self.hidden_dim:]
        return self.w3_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_residual(x, residual=None)

    def forward_with_residual(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
        cde_emit: bool = False,
    ):
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x_2d = x.reshape(B * S, H)
            residual_2d = residual.reshape(B * S, self.dim) if residual is not None else None
        else:
            x_2d = x
            residual_2d = residual

        self._ensure_workspace(x.device)
        debug_name = getattr(self, '_lbt_debug_name', None)
        if debug_name is None:
            debug_name = f"{self.__class__.__name__}:{id(self)}"
            self._lbt_debug_name = debug_name
        _tk_stage_trace('ffn_fwd', 'start', debug_name)
        _tk_debug_print('ffn_fwd', 'start', debug_name)

        empty_fp4 = torch.empty(
            0, dtype=torch.float4_e2m1fn_x2, device=x_2d.device
        )
        empty_sc = torch.empty(
            0, dtype=torch.float8_e4m3fn, device=x_2d.device
        )
        empty_sg = torch.empty(0, dtype=torch.float32, device=x_2d.device)
        empty_r = torch.empty(0, dtype=torch.float32, device=x_2d.device)
        empty_gamma = torch.empty(
            0, dtype=torch.bfloat16, device=x_2d.device
        )

        with localcta_v4_cpp_only_scope(use_tk_localcta_v4_ffn_cpp_only()):
            out = _FusedFFNFunctionV2_TK.apply(
                x_2d, self._w1_weight_view(), self._w3_weight_view(), self.w2_weight,
                self.norm_weight, self.epsilon,
                self.w1_weight_quantizer,
                self.w3_input_quantizer, self.w3_weight_quantizer,
                self.w2_input_quantizer, self.w2_weight_quantizer,
                self.grad_quantizer_w1, self.grad_quantizer_w3,
                self.grad_quantizer_w2,
                self.w1_input_quantizer,
                self._workspace,
                debug_name,
                residual_2d,
                empty_fp4, empty_sc, empty_sg,
                empty_fp4, empty_sc, empty_sg,
                empty_r, empty_gamma, cde_row_rms_partial, cde_emit,
            )
        _tk_stage_trace('ffn_fwd', 'end', debug_name)
        _tk_debug_print('ffn_fwd', 'end', debug_name)

        if is_3d:
            if isinstance(out, tuple):
                return out[0].view(B, S, self.dim), out[1]
            return out.view(B, S, self.dim)
        return out

    def forward_with_h_carrier(
        self,
        carrier,
        next_attention_gamma: Optional[torch.Tensor] = None,
    ):
        (
            z, row_fp4, row_sc, row_sg,
            col_fp4, col_sc, col_sg, r_tile,
        ) = carrier
        is_3d = z.dim() == 3
        if is_3d:
            B, S, H = z.shape
            z_2d = z.reshape(B * S, H)
        else:
            z_2d = z
        self._ensure_workspace(z.device)
        debug_name = getattr(self, '_lbt_debug_name', None)
        if debug_name is None:
            debug_name = f"{self.__class__.__name__}:{id(self)}"
            self._lbt_debug_name = debug_name
        next_gamma = next_attention_gamma
        if next_gamma is None:
            next_gamma = torch.empty(
                0, dtype=torch.bfloat16, device=z.device
            )
        with localcta_v4_cpp_only_scope(use_tk_localcta_v4_ffn_cpp_only()):
            out = _FusedFFNFunctionV2_TK.apply(
                z_2d,
                self._w1_weight_view(),
                self._w3_weight_view(),
                self.w2_weight,
                self.norm_weight,
                self.epsilon,
                self.w1_weight_quantizer,
                self.w3_input_quantizer,
                self.w3_weight_quantizer,
                self.w2_input_quantizer,
                self.w2_weight_quantizer,
                self.grad_quantizer_w1,
                self.grad_quantizer_w3,
                self.grad_quantizer_w2,
                self.w1_input_quantizer,
                self._workspace,
                debug_name,
                z_2d,
                row_fp4, row_sc, row_sg,
                col_fp4, col_sc, col_sg,
                r_tile, next_gamma, None, False,
            )
        if next_attention_gamma is not None:
            z_next = out[0]
            if is_3d:
                z_next = z_next.view(B, S, self.dim)
            return (z_next, *out[1:])
        return out.view(B, S, self.dim) if is_3d else out

    def invalidate_weight_cache(self):
        """For benchmark API compatibility."""
        pass

    def init_weights(self, init_std: float = 0.02):
        """Initialize weights, matching Llama's FeedForward.init_weights().

        NOTE: norm_weight MUST be reset to ones here because the model is
        constructed on meta device and init_weights() is called after FSDP
        materialization. Without this, norm_weight contains uninitialized data.
        """
        nn.init.ones_(self.norm_weight)
        if self.packed_w13:
            _safe_trunc_normal_(self.w13_weight[:self.hidden_dim], mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w13_weight[self.hidden_dim:], mean=0.0, std=init_std)
        else:
            _safe_trunc_normal_(self.w1_weight, mean=0.0, std=0.02)
            _safe_trunc_normal_(self.w3_weight, mean=0.0, std=init_std)
        _safe_trunc_normal_(self.w2_weight, mean=0.0, std=init_std)

    @classmethod
    def from_unfused(cls, ffn, norm, recipe=None):
        """Create from an existing FeedForward module + RMSNorm.

        Args:
            ffn: FeedForward module with .w1, .w2, .w3 (nn.Linear)
            norm: nn.RMSNorm (the ffn_norm being absorbed)
            recipe: NVFP4BlockScaling recipe for w2
        """
        dim = ffn.w1.in_features
        hidden_dim = ffn.w1.out_features
        device = ffn.w1.weight.device
        dtype = ffn.w1.weight.dtype
        eps = getattr(norm, 'eps', None)
        if eps is None:
            eps = 1e-5

        fused = cls(
            dim, hidden_dim, norm_eps=eps, bias=False,
            device=device, dtype=dtype, recipe=recipe,
        )

        # Copy weights
        if device.type != 'meta':
            with torch.no_grad():
                fused._w1_weight_view().copy_(ffn.w1.weight)
                fused._w3_weight_view().copy_(ffn.w3.weight)
                fused.w2_weight.copy_(ffn.w2.weight)

                # Copy norm weight
                if hasattr(norm, 'weight') and norm.weight is not None:
                    fused.norm_weight.copy_(norm.weight)

        return fused


# ---------------------------------------------------------------------------
# _FusedQKVFunction: custom autograd for stacked QKV with fused RMSNorm
# ---------------------------------------------------------------------------
class _FusedQKVFunction_TE(torch.autograd.Function):
    """Fused RMSNorm + grouped FP4 quant + single QKV GEMM.

    Forward:
      1. RMSNorm(x, gamma, eps) → normed_x (PyTorch F.rms_norm)
      2. quantize(normed_x) → x_q (ONCE — shared for Q, K, V)
      3. group_quantize(w_qkv, [q_dim, k_dim, v_dim]) → w_q (1 kernel, per-split amax!)
      4. Single GEMM: y = w_qkv @ x → split → xq, xk, xv
    
    Savings vs TE: 2 fewer input quants + 2 fewer weight quants + 2 fewer GEMMs.
    Fix vs old stacked: per-split amax via grouped kernel → better FP4 resolution.

    Backward:
      1. Combine dQ, dK, dV → d_y
      2. dgrad: dx_normed = W_qkv^T @ d_y (single GEMM)
      3. wgrad: dW_qkv = x_normed^T @ d_y (single GEMM)
      4. RMSNorm backward: dx, dgamma (manual math)
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,         # (M, K) bf16 — raw pre-norm input
        w_qkv: torch.Tensor,         # (Q+K+V, K) bf16 — stacked weights
        norm_weight: torch.Tensor,   # (K,) bf16 — RMSNorm gamma
        epsilon: float,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        # TE quantizers (for fallback / backward)
        input_quantizer: NVFP4Quantizer,
        weight_quantizer: NVFP4Quantizer,
        grad_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
        debug_name: Optional[str] = None,
    ):
        M, K = input.shape
        total_out = q_dim + k_dim + v_dim

        # Force bf16 — FSDP may materialize in float32
        inp = _as_contiguous_bf16(input)
        nw = _as_contiguous_bf16(norm_weight.detach())

        # 1+2. RMSNorm + NVFP4 quant (TE path: no swizzle)
        te_fused = _get_te_fused()
        normed, inv_rms = te_fused.fused_rmsnorm_only(inp.detach(), nw, float(epsilon))
        x_nvfp4 = _fast_quantize(normed, input_quantizer, tk_swizzle=False)

        # 3+4. Weight quantization + GEMM (TE per-split)
        w_bf16 = _as_contiguous_bf16(w_qkv.detach())
        N_dims = [q_dim, k_dim, v_dim]

        w_splits = w_bf16.split(N_dims, dim=0)  # [Wq, Wk, Wv]
        w_nvfp4_list = []
        outputs = []
        for w_split in w_splits:
            w_q = _fast_quantize(w_split, weight_quantizer, tk_swizzle=False)
            w_nvfp4_list.append(w_q)
            y_i = tex.generic_gemm(
                w_q, True, x_nvfp4, False,
                None, None, TE_DType[torch.bfloat16],
                None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )[0]
            outputs.append(y_i)
        xq, xk, xv = outputs

        ctx.save_for_backward(inp, nw, inv_rms)
        ctx.x_nvfp4 = x_nvfp4
        ctx.w_nvfp4_splits = w_nvfp4_list

        ctx.N_dims = N_dims
        ctx.grad_quantizer = grad_quantizer
        ctx.workspace = workspace
        ctx.epsilon = epsilon
        ctx.q_dim = q_dim
        ctx.k_dim = k_dim
        ctx.v_dim = v_dim

        return xq.contiguous(), xk.contiguous(), xv.contiguous()

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        workspace = ctx.workspace

        input, norm_weight, inv_rms = ctx.saved_tensors
        w_nvfp4_splits = ctx.w_nvfp4_splits
        x_nvfp4 = ctx.x_nvfp4

        grad_splits = [
            _as_contiguous_bf16(grad_q),
            _as_contiguous_bf16(grad_k),
            _as_contiguous_bf16(grad_v),
        ]

        dx_parts = []
        grad_w_parts = []
        for w_q, dy_split in zip(w_nvfp4_splits, grad_splits):
            dy_q = _fast_quantize(dy_split, ctx.grad_quantizer, tk_swizzle=False)
            dx_i = tex.generic_gemm(
                w_q, False, dy_q, False, None, None,
                TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )[0]
            dx_parts.append(dx_i)
            dw_i = tex.generic_gemm(
                x_nvfp4, False, dy_q, True, None, None,
                TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
                False, None, False,
                workspace, workspace.shape[0], False, False,
            )[0]
            grad_w_parts.append(dw_i)

        dx_normed = dx_parts[0] + dx_parts[1] + dx_parts[2]
        grad_w_qkv = torch.cat(grad_w_parts, dim=0)

        ctx.w_nvfp4_splits = None
        ctx.x_nvfp4 = None

        te_fused = _get_te_fused()
        inp_bf16 = _as_contiguous_bf16(input)
        grad_input, dgamma = te_fused.fused_rmsnorm_backward(
            dx_normed.contiguous(), inp_bf16,
            _as_contiguous_bf16(norm_weight), inv_rms)

        grad_norm_weight = _as_param_grad_dtype(dgamma, norm_weight)

        return (
            grad_input,
            grad_w_qkv,
            grad_norm_weight,
            None, None, None, None,
            None, None, None, None,
        )

# ── Full forward+backward CUDA-graphed QKV ──────────────────────
_qkv_full_graph_cache = {}  # (M, K, N_total, device) → (graph, sb)

def _qkv_full_graphed(x_2d, w_qkv, norm_weight, epsilon,
                       q_dim, k_dim, v_dim,
                       input_quantizer, weight_quantizer,
                       grad_output_tuple,
                       workspace):
    """Full forward+backward QKV captured as a SINGLE CUDA graph.

    Bypasses autograd entirely. All TK kernels' TMA descriptors are
    captured with stable tensor addresses (static buffers or graph-pool).

    Args:
        x_2d: (M, K) bf16 input
        w_qkv: (N_total, K) bf16 weights
        norm_weight: (K,) bf16 rmsnorm gamma
        epsilon: rmsnorm eps
        q_dim, k_dim, v_dim: output split dims
        input_quantizer, weight_quantizer: TE quantizers
        grad_output_tuple: (grad_q, grad_k, grad_v) bf16
        workspace: workspace buffer

    Returns:
        (xq, xk, xv, grad_input, grad_w_qkv, grad_norm_weight)
    """
    from .tk_gemm import (
        _get_tk, _get_tk_quant_for_gemm,
        _get_dgrad_bufs, _get_wgrad_buf, _get_sg_tile_indices,
        _weight_split_cache, tk_grouped_wgrad_gemm,
    )

    M, K = x_2d.shape
    N_dims = [q_dim, k_dim, v_dim]
    N_total = sum(N_dims)
    key = (M, K, N_total, x_2d.device.index)
    cache = _qkv_full_graph_cache.get(key)

    grad_q, grad_k, grad_v = grad_output_tuple

    te_fused = _get_te_fused()
    tk_mod = _get_tk()
    tk_q = _get_tk_quant()
    tkq = _get_tk_quant_for_gemm()

    if cache is not None:
        # ── Replay path ──
        graph, sb = cache
        sb['x'].copy_(x_2d)
        sb['w_qkv'].copy_(w_qkv)
        sb['norm_weight'].copy_(norm_weight)
        sb['grad_q'].copy_(grad_q)
        sb['grad_k'].copy_(grad_k)
        sb['grad_v'].copy_(grad_v)
        graph.replay()
        return (sb['xq'].clone(), sb['xk'].clone(), sb['xv'].clone(),
                sb['grad_input'].clone(), sb['grad_w'].clone(), sb['grad_nw'].clone())

    # ── First call: create static buffers, warmup, capture ──
    sb = {
        'x': x_2d.clone(),
        'w_qkv': w_qkv.detach().clone(),
        'norm_weight': norm_weight.detach().clone(),
        'grad_q': grad_q.clone(),
        'grad_k': grad_k.clone(),
        'grad_v': grad_v.clone(),
    }

    # --- Pre-allocate weight quant buffers OUTSIDE graph (v5_alloc) ---
    w_bf16_static = sb['w_qkv'].contiguous().to(torch.bfloat16)
    wq_alloc = tk_q.tk_group_quantize_v5_alloc(w_bf16_static, N_dims)
    (wq_fp4_row, wq_fp4_col, wq_sg_cat, wq_fwd_b_sg, wq_dgrad_b_sg,
     wq_amax, wq_sync, wq_psync, wq_tma_dev,
     wq_sc_row_list, wq_fp4_col_list, wq_sc_col_list) = wq_alloc

    # --- Pre-allocate dy quant buffers OUTSIDE graph (dim1_alloc) ---
    dy_static = torch.cat([sb['grad_q'], sb['grad_k'], sb['grad_v']], dim=1).to(torch.bfloat16)
    dq_alloc = tkq.tk_group_quantize_dim1_alloc(dy_static, list(N_dims))
    (dq_fp4_row, dq_fp4_col, dq_sg, dq_amax, dq_sync, dq_psync,
     dq_tma, dq_sc_row, dq_fp4_col_list, dq_sc_col, dq_tma_host) = dq_alloc

    def _fwd_bwd():
        inp = sb['x'].contiguous().to(torch.bfloat16)
        nw = sb['norm_weight'].contiguous().to(torch.bfloat16)
        w_bf16 = sb['w_qkv'].contiguous().to(torch.bfloat16)

        # ── FORWARD ──
        # 1. RMSNorm
        normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))

        # 2. Quantize input in TK-consumable layout. This can optionally use
        # the TE/custom swizzled path to isolate activation-quant numerics.
        x_nvfp4 = _qkv_forward_quantize(normed, input_quantizer)

        # 3. Quantize weights (graph-safe v5_launch — NO allocations)
        wq_result = tk_q.tk_group_quantize_v5_launch(
            w_bf16, N_dims,
            wq_fp4_row, wq_fp4_col, wq_sg_cat, wq_fwd_b_sg, wq_dgrad_b_sg,
            wq_amax, wq_sync, wq_psync, wq_tma_dev,
            wq_sc_row_list, wq_fp4_col_list, wq_sc_col_list
        )
        wc_fp4_view, fwd_b_sg, dgrad_b_sg, sg_cat = wq_result
        # Build scale row tensor from pre-allocated list
        wc_sc_row = torch.cat(
            [sc.view(torch.uint8) for sc in wq_sc_row_list], dim=1
        ).view(torch.float8_e4m3fn)

        # 4. Forward GEMM (graph-safe — TMA by value in cudaLaunchKernelEx)
        x_fp4, x_sc, x_sg = x_nvfp4._tk_row
        if use_tk_localcta() and use_tk_localcta_direct_contract() and (not torch.is_tensor(x_sg) or x_sg.dim() != 2):
            x_sg = _const_sg_grid(x_sg.reshape(-1)[0], M, K)
        xq = torch.empty(M, q_dim, dtype=torch.bfloat16, device=inp.device)
        xk = torch.empty(M, k_dim, dtype=torch.bfloat16, device=inp.device)
        xv = torch.empty(M, v_dim, dtype=torch.bfloat16, device=inp.device)
        tk_mod.nvfp4_grouped_gemm(
            x_fp4, x_sc, x_sg, wc_fp4_view, wc_sc_row, fwd_b_sg,
            xq, xk, xv
        )

        # ── BACKWARD ──
        # 5. Assemble dy_cat
        gq = sb['grad_q'].to(torch.bfloat16)
        gk = sb['grad_k'].to(torch.bfloat16)
        gv = sb['grad_v'].to(torch.bfloat16)
        dy_cat = torch.cat([gq, gk, gv], dim=1)

        # 6. Assemble weight col tensors for dgrad (from pre-allocated v5 buffers)
        col_fp4_cat = torch.cat(
            [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in wq_fp4_col_list], dim=1
        ).view(torch.float4_e2m1fn_x2)
        col_sc_cat = torch.cat(
            [sc.contiguous().view(torch.uint8) for sc in wq_sc_col_list], dim=1
        ).view(torch.float8_e4m3fn)
        w_sg_per_split = sg_cat.float()

        # 7. Quantize dy (graph-safe dim1_launch — NO allocations)
        quant_result = tkq.tk_group_quantize_dim1_launch(
            dy_cat, list(N_dims),
            dq_fp4_row, dq_fp4_col, dq_sg, dq_amax, dq_sync, dq_psync,
            dq_tma_host, dq_tma, dq_sc_row, dq_fp4_col_list, dq_sc_col
        )
        fp4_row_list, sc_row_list, sg_per_group, \
            fp4_col_list, sc_col_list, \
            a_fp4_full, a_sc_cat, fp4_col_full, sc_col_cat = quant_result

        # 8. Dgrad GEMM (graph-safe — TMA by value)
        n_groups = len(N_dims)
        A_sg_list = [sg_per_group[i:i+1].to(torch.float32) for i in range(n_groups)]

        # Weight col splits
        w_fp4_bytes = col_fp4_cat.view(torch.uint8)
        w_sc_bytes = col_sc_cat.view(torch.uint8)
        B_fp4_list, B_sc_list = [], []
        offset_fp4, offset_sc = 0, 0
        for n_i in N_dims:
            B_fp4_list.append(
                w_fp4_bytes[:, offset_fp4:offset_fp4 + n_i // 2]
                .contiguous().view(torch.float4_e2m1fn_x2))
            B_sc_list.append(
                w_sc_bytes[:, offset_sc:offset_sc + n_i // 64]
                .contiguous().view(torch.float8_e4m3fn))
            offset_fp4 += n_i // 2
            offset_sc += n_i // 64
        B_sg_list = [w_sg_per_split[i:i+1] for i in range(n_groups)]

        D_list = [torch.empty(M, K, dtype=torch.bfloat16, device=inp.device) for _ in range(n_groups)]

        use_strided = (a_fp4_full is not None
                       and hasattr(tk_mod, 'nvfp4_batched_gemm_strided'))
        if use_strided:
            a_fp4_u8 = a_fp4_full.view(torch.uint8)
            A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
            col_offsets, col_widths = [], []
            off = 0
            for n_i in N_dims:
                col_offsets.append(off)
                col_widths.append(n_i // 2)
                off += n_i // 2
            tk_mod.nvfp4_batched_gemm_strided(
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
                tk_mod,
                A_fp4_list, A_sc_list, A_sg_list,
                B_fp4_list, B_sc_list, B_sg_list,
                D_list
            )

        dx_normed = D_list[0] + D_list[1] + D_list[2]

        # 9. Wgrad (graph-safe — TMA by value)
        dy_col_quant = (fp4_col_list, sc_col_list, sg_per_group,
                        fp4_col_full, sc_col_cat)
        class _ColRef:
            __slots__ = ('_tk_col',)
            def __init__(self, c): self._tk_col = c
        static_x_col = _ColRef((x_nvfp4._tk_col[0], x_nvfp4._tk_col[1], x_nvfp4._tk_col[2]))
        gw = tk_grouped_wgrad_gemm(dy_col_quant, static_x_col, N_dims)

        # 10. RMSNorm backward
        gi, dg = te_fused.fused_rmsnorm_backward(
            dx_normed.contiguous(), inp.contiguous(),
            nw, inv_rms)

        return xq, xk, xv, gi, gw, dg.to(sb['norm_weight'].dtype)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(3):
        _fwd_bwd()
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        xq, xk, xv, gi, gw, gnw = _fwd_bwd()

    sb['xq'] = xq
    sb['xk'] = xk
    sb['xv'] = xv
    sb['grad_input'] = gi
    sb['grad_w'] = gw
    sb['grad_nw'] = gnw
    _qkv_full_graph_cache[key] = (graph, sb)

    # First call: replay with real data
    sb['x'].copy_(x_2d)
    sb['w_qkv'].copy_(w_qkv)
    sb['norm_weight'].copy_(norm_weight)
    sb['grad_q'].copy_(grad_q)
    sb['grad_k'].copy_(grad_k)
    sb['grad_v'].copy_(grad_v)
    graph.replay()
    return (sb['xq'].clone(), sb['xk'].clone(), sb['xv'].clone(),
            sb['grad_input'].clone(), sb['grad_w'].clone(), sb['grad_nw'].clone())


# ── CUDA-graphed QKV backward ──────────────────────────────────
_qkv_bwd_graph_cache = {}  # (M, K, N_total, device) → (graph, static_bufs)

def _qkv_bwd_graphed(dy_cat, w_col, N_dims, x_nvfp4,
                      input, norm_weight, inv_rms, K, M,
                      w_bf16=None,
                      grad_splits=None,
                      debug_name=None,
                      rope_live64_cs=None,
                      rope_seq_len: int = 0,
                      h_tile: bool = False):
    """Graph-captured QKV backward: dgrad + wgrad + RMSNorm bwd.

    Non-graphed path uses per-split quantize + batched GEMM:
    - 3× _fast_quantize on already-contiguous grad splits (zero copy)
    - 1× batched GEMM (single kernel launch for all dgrad GEMMs)
    - Cached weight splits + pre-allocated D buffers + in-place sum

    When grad_splits=(gq,gk,gv) is provided, avoids the 0.867ms dy_cat copy.

    First call per (M, K, N_total) with CUDA graphs captures the computation
    as a CUDA graph with multi-stream overlap. Subsequent calls copy inputs
    to static buffers and replay the graph.
    """
    N_total = sum(N_dims)
    device = w_col._tk_col[0].device if grad_splits is not None else dy_cat.device
    key = (M, K, N_total, device.index)
    if h_tile:
        if grad_splits is None:
            raise RuntimeError("H QKV backward requires split gradients")
        from .tk_gemm import tk_fused_qkv_backward

        te_fused = _get_te_fused()
        inp_bf16 = input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)
        nw_bf16 = norm_weight if norm_weight.dtype == torch.bfloat16 else norm_weight.to(torch.bfloat16)
        gi, gw, dg, rescue_info = tk_fused_qkv_backward(
            grad_splits, w_col, N_dims, x_nvfp4,
            inp_bf16, nw_bf16, inv_rms, w_bf16, te_fused,
            debug_name=debug_name,
            rope_live64_cs=rope_live64_cs,
            rope_seq_len=rope_seq_len,
            h_tile=True,
        )
        return gi, gw, _as_param_grad_dtype(dg, norm_weight), rescue_info
    if use_tk_localcta():
        te_fused = _get_te_fused()
        from .tk_gemm import _get_wgrad_stream, _launch_rmsnorm_bwd_out_async, tk_fused_qkv_backward
        inp_bf16 = input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)
        nw_bf16 = norm_weight if norm_weight.dtype == torch.bfloat16 else norm_weight.to(torch.bfloat16)
        if use_tk_localcta_direct_contract():
            if grad_splits is not None and use_tk_qkv_backward_debug_fallback():
                gi, gw, dg, rescue_info = tk_fused_qkv_backward(
                    grad_splits, w_col, N_dims, x_nvfp4,
                    inp_bf16, nw_bf16, inv_rms, w_bf16, te_fused,
                    rope_live64_cs=rope_live64_cs,
                    rope_seq_len=rope_seq_len,
                )
                return gi, gw, _as_param_grad_dtype(dg, norm_weight), rescue_info
            dy_input = grad_splits if grad_splits is not None else dy_cat
            dx_normed, dy_col_quant = tk_grouped_k_dgrad_gemm(
                dy_input,
                w_col,
                N_dims,
                debug_name=debug_name,
            )
            gw = tk_grouped_wgrad_gemm(dy_col_quant, x_nvfp4, N_dims)
            rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                dx_normed.contiguous(), inp_bf16.contiguous(),
                nw_bf16, inv_rms, te_fused,
                tag='qkv',
            )
            torch.cuda.current_stream().wait_stream(rms_stream)
            gi = rms_state['grad_input']
            dg = rms_state.get('dgamma_out', rms_state['dgamma'])
        elif grad_splits is not None:
            gi, gw, dg, rescue_info = tk_fused_qkv_backward(
                grad_splits, w_col, N_dims, x_nvfp4,
                inp_bf16, nw_bf16, inv_rms, w_bf16, te_fused,
                debug_name=debug_name,
                rope_live64_cs=rope_live64_cs,
                rope_seq_len=rope_seq_len,
            )
            return gi, gw, _as_param_grad_dtype(dg, norm_weight), rescue_info
        else:
            dx_normed, dy_col_quant = tk_grouped_k_dgrad_gemm(
                dy_cat,
                w_col,
                N_dims,
                debug_name=debug_name,
            )
            wgrad_stream = _get_wgrad_stream()
            wgrad_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(wgrad_stream):
                gw = tk_grouped_wgrad_gemm(dy_col_quant, x_nvfp4, N_dims)
            rms_state, rms_stream = _launch_rmsnorm_bwd_out_async(
                dx_normed.contiguous(), inp_bf16.contiguous(),
                nw_bf16, inv_rms, te_fused,
                tag='qkv',
            )
            torch.cuda.current_stream().wait_stream(wgrad_stream)
            torch.cuda.current_stream().wait_stream(rms_stream)
            gi = rms_state['grad_input']
            dg = rms_state.get('dgamma_out', rms_state['dgamma'])
        return gi, gw, _as_param_grad_dtype(dg, norm_weight), None
    cache = _qkv_bwd_graph_cache.get(key)

    if cache is not None:
        # ── Replay path: full backward graph ──
        graph, sb = cache
        if grad_splits is not None:
            # Copy per-split grads into static dy_cat
            offsets = [0]
            for n in N_dims[:-1]:
                offsets.append(offsets[-1] + n)
            for i, gs in enumerate(grad_splits):
                sb['dy_cat'][:, offsets[i]:offsets[i]+N_dims[i]].copy_(gs)
        else:
            sb['dy_cat'].copy_(dy_cat)
        sb['col_fp4'].copy_(w_col._tk_col[0])
        sb['col_sc'].copy_(w_col._tk_col[1])
        sb['w_sg'].copy_(w_col._tk_col[2])
        sb['x_fp4_c'].copy_(x_nvfp4._tk_col[0])
        sb['x_sc_c'].copy_(x_nvfp4._tk_col[1])
        sb['x_sg_c'].copy_(x_nvfp4._tk_col[2])
        sb['input'].copy_(input)
        sb['norm_weight'].copy_(norm_weight)
        sb['inv_rms'].copy_(inv_rms)
        graph.replay()
        return sb['grad_input'].clone(), sb['grad_w'].clone(), sb['grad_nw'].clone(), None


    # ── First call: set up static buffers, warm up, capture graph ──
    te_fused = _get_te_fused()

    # Decide: CUDA graph (small M, eliminates Python overhead) vs
    # zero-copy per-split (large M, avoids 0.87ms dy_cat copy in graph replay)
    _graph_max_m = int(os.environ.get('QKV_GRAPH_MAX_M', '16384'))
    use_plain_small_m_eager = (not use_tk_localcta()) and M < 256 and len(N_dims) > 1
    _use_graph = (
        False
        and (use_cuda_graph() or M <= _graph_max_m)
        and not use_tk_qkv_backward_debug_fallback()
        and not use_plain_small_m_eager
    )

    if not _use_graph:
        if grad_splits is not None:
            # ── Fused path: quant + dgrad + wgrad + rmsnorm_bwd in one call ──
            # Eliminates ~0.5ms Python overhead at M=65536 vs separate calls.
            from .tk_gemm import tk_fused_qkv_backward

            inp_bf16 = input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)
            nw_bf16 = norm_weight if norm_weight.dtype == torch.bfloat16 else norm_weight.to(torch.bfloat16)
            gi, gw, dg, rescue_info = tk_fused_qkv_backward(
                grad_splits, w_col, N_dims, x_nvfp4,
                inp_bf16, nw_bf16, inv_rms, w_bf16, te_fused,
                debug_name=debug_name,
                rope_live64_cs=rope_live64_cs,
                rope_seq_len=rope_seq_len,
            )
            return gi, gw, _as_param_grad_dtype(dg, norm_weight), rescue_info

        else:
            # ── Fallback: grouped quantize path (used when dy_cat is provided) ──
            from .tk_gemm import tk_grouped_k_dgrad_gemm as _grp_dgrad

            class _ColRef:
                __slots__ = ('_tk_col',)
                def __init__(self, c): self._tk_col = c
            w_col_ref = _ColRef((w_col._tk_col[0], w_col._tk_col[1], w_col._tk_col[2]))

            dx_normed, dy_col_quant = _grp_dgrad(
                dy_cat, w_col_ref, N_dims, debug_name=debug_name
            )
            gw = tk_grouped_wgrad_gemm(dy_col_quant, x_nvfp4, N_dims)
            inp_bf16 = input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)
            gi, dg = te_fused.fused_rmsnorm_backward(
                dx_normed.contiguous(), inp_bf16.contiguous(),
                norm_weight.to(torch.bfloat16), inv_rms)
            return gi, gw, _as_param_grad_dtype(dg, norm_weight), None


    # ── CUDA graph path: graph the FULL backward ──
    # dim1_launch: graph-safe quant (no pinned TMA buffer)
    # GEMMs: graph-safe (TMA descriptors passed by value via cudaLaunchKernelEx)
    # rmsnorm_backward: graph-safe (standard CUDA kernel)
    from .tk_gemm import (
        _get_tk, _get_tk_quant_for_gemm, _get_dgrad_bufs,
        _weight_split_cache, _get_sg_tile_indices, _get_wgrad_buf,
    )

    tk = _get_tk()
    tkq = _get_tk_quant_for_gemm()

    # Build dy_cat for graph capture if we got grad_splits instead
    if dy_cat is None and grad_splits is not None:
        dy_cat = torch.cat(list(grad_splits), dim=1)

    M = dy_cat.shape[0]
    w_fp4_c, w_sc_c, w_sg_c = w_col._tk_col
    K = w_fp4_c.shape[0]
    n_groups = len(N_dims)
    N_total = sum(N_dims)

    # --- Static input buffers ---
    sb = {
        'dy_cat': dy_cat.clone(),
        'col_fp4': w_fp4_c.clone(),
        'col_sc': w_sc_c.clone(),
        'w_sg': w_sg_c.clone(),
        'x_fp4_c': x_nvfp4._tk_col[0].clone(),
        'x_sc_c': x_nvfp4._tk_col[1].clone(),
        'x_sg_c': x_nvfp4._tk_col[2].clone(),
        'input': input.clone(),
        'norm_weight': norm_weight.clone(),
        'inv_rms': inv_rms.clone(),
    }

    # --- Pre-allocate dy quant buffers OUTSIDE graph (dim1_alloc) ---
    quant_alloc = tkq.tk_group_quantize_dim1_alloc(sb['dy_cat'], list(N_dims))
    (q_fp4_row, q_fp4_col, q_sg, q_amax, q_sync, q_psync, q_tma,
     q_sc_row, q_fp4_col_list, q_sc_col, q_tma_host) = quant_alloc

    # --- Pre-allocate D buffers and wgrad buf ---
    D_list = _get_dgrad_bufs(n_groups, M, K, dy_cat.device)
    _get_sg_tile_indices(list(N_dims), dy_cat.device)
    _get_wgrad_buf(K, N_total, dy_cat.device)

    # --- Static x_nvfp4 for wgrad ---
    class _SQuant:
        __slots__ = ('_tk_col',)
        def __init__(self, fp4_c, sc_c, sg_c):
            self._tk_col = (fp4_c, sc_c, sg_c)
    static_x = _SQuant(sb['x_fp4_c'], sb['x_sc_c'], sb['x_sg_c'])

    def _run():
        # 1. Quantize dy (graph-safe dim1_launch)
        quant_result = tkq.tk_group_quantize_dim1_launch(
            sb['dy_cat'], list(N_dims),
            q_fp4_row, q_fp4_col, q_sg, q_amax, q_sync, q_psync,
            q_tma_host, q_tma, q_sc_row, q_fp4_col_list, q_sc_col
        )
        fp4_row_list, sc_row_list, sg_per_group, \
            fp4_col_list, sc_col_list, \
            a_fp4_full, a_sc_cat, fp4_col_full, sc_col_cat = quant_result

        # 2. Dgrad GEMM (graph-safe — TMA by value)
        A_sg_list = [sg_per_group[i:i+1].to(torch.float32) for i in range(n_groups)]

        # Weight col splits — computed INSIDE graph so .contiguous() copies
        # are captured and re-execute on replay with fresh sb['col_fp4'] data
        w_fp4_bytes = sb['col_fp4'].view(torch.uint8)
        w_sc_bytes = sb['col_sc'].view(torch.uint8)
        B_fp4_list, B_sc_list = [], []
        offset_fp4, offset_sc = 0, 0
        for n_i in N_dims:
            B_fp4_list.append(
                w_fp4_bytes[:, offset_fp4:offset_fp4 + n_i // 2]
                .contiguous().view(torch.float4_e2m1fn_x2))
            B_sc_list.append(
                w_sc_bytes[:, offset_sc:offset_sc + n_i // 64]
                .contiguous().view(torch.float8_e4m3fn))
            offset_fp4 += n_i // 2
            offset_sc += n_i // 64
        B_sg_list = [sb['w_sg'][i:i+1].to(torch.float32) for i in range(n_groups)]

        use_strided = (a_fp4_full is not None
                       and hasattr(tk, 'nvfp4_batched_gemm_strided'))
        if use_strided:
            a_fp4_u8 = a_fp4_full.view(torch.uint8)
            A_sc_list = [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list]
            col_offsets, col_widths = [], []
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

        # 3. Sum dgrad outputs
        dx_normed = D_list[0] + D_list[1] + D_list[2] if n_groups == 3 else sum(D_list)

        # 4+5. Overlap wgrad (side stream) with rmsnorm_bwd (main stream)
        # CUDA graph captures stream events for automatic parallel replay
        s1 = _get_bwd_side_stream()
        s1.wait_stream(torch.cuda.current_stream())

        # Wgrad on side stream (graph-safe — TMA by value)
        with torch.cuda.stream(s1):
            dy_col_quant = (fp4_col_list, sc_col_list, sg_per_group,
                            fp4_col_full, sc_col_cat)
            gw = tk_grouped_wgrad_gemm(dy_col_quant, static_x, N_dims)

        # RMSNorm backward on main stream (concurrent with wgrad)
        inp_bf16 = sb['input'] if sb['input'].dtype == torch.bfloat16 else sb['input'].to(torch.bfloat16)
        gi, dg = te_fused.fused_rmsnorm_backward(
            dx_normed.contiguous(), inp_bf16.contiguous(),
            sb['norm_weight'].to(torch.bfloat16), sb['inv_rms'])

        # Wait for wgrad to finish
        torch.cuda.current_stream().wait_stream(s1)
        return gi, gw, dg.to(sb['norm_weight'].dtype)

    # Warmup — sync between iterations to drain PDL chain at large M
    torch.cuda.synchronize()
    for _ in range(3):
        _run()
        torch.cuda.synchronize()

    # Capture full backward
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        gi, gw, gnw = _run()
    sb['grad_input'] = gi
    sb['grad_w'] = gw
    sb['grad_nw'] = gnw
    _qkv_bwd_graph_cache[key] = (graph, sb)

    # First call: replay with real data
    sb['dy_cat'].copy_(dy_cat)
    sb['col_fp4'].copy_(w_col._tk_col[0])
    sb['col_sc'].copy_(w_col._tk_col[1])
    sb['w_sg'].copy_(w_col._tk_col[2])
    sb['x_fp4_c'].copy_(x_nvfp4._tk_col[0])
    sb['x_sc_c'].copy_(x_nvfp4._tk_col[1])
    sb['x_sg_c'].copy_(x_nvfp4._tk_col[2])
    sb['input'].copy_(input)
    sb['norm_weight'].copy_(norm_weight)
    sb['inv_rms'].copy_(inv_rms)
    graph.replay()
    return sb['grad_input'].clone(), sb['grad_w'].clone(), sb['grad_nw'].clone(), None



class _FusedQKVFunction_TK(torch.autograd.Function):
    """Fused RMSNorm + grouped FP4 quant + single QKV GEMM.

    Forward:
      1. RMSNorm(x, gamma, eps) → normed_x (PyTorch F.rms_norm)
      2. quantize(normed_x) → x_q (ONCE — shared for Q, K, V)
      3. group_quantize(w_qkv, [q_dim, k_dim, v_dim]) → w_q (1 kernel, per-split amax!)
      4. Single GEMM: y = w_qkv @ x → split → xq, xk, xv
    
    Savings vs TE: 2 fewer input quants + 2 fewer weight quants + 2 fewer GEMMs.
    Fix vs old stacked: per-split amax via grouped kernel → better FP4 resolution.

    Backward:
      1. Combine dQ, dK, dV → d_y
      2. dgrad: dx_normed = W_qkv^T @ d_y (single GEMM)
      3. wgrad: dW_qkv = x_normed^T @ d_y (single GEMM)
      4. RMSNorm backward: dx, dgamma (manual math)
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,         # (M, K) bf16 — raw pre-norm input
        w_qkv: torch.Tensor,         # (Q+K+V, K) bf16 — stacked weights
        norm_weight: torch.Tensor,   # (K,) bf16 — RMSNorm gamma
        epsilon: float,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        rope_freqs_cis: torch.Tensor | None,
        rope_batch_size: int,
        rope_seq_len: int,
        rope_head_dim: int,
        # TE quantizers (for fallback / backward)
        input_quantizer: NVFP4Quantizer,
        weight_quantizer: NVFP4Quantizer,
        grad_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
        debug_name: Optional[str] = None,
        h_row_fp4: Optional[torch.Tensor] = None,
        h_row_sc: Optional[torch.Tensor] = None,
        h_row_sg: Optional[torch.Tensor] = None,
        h_col_fp4: Optional[torch.Tensor] = None,
        h_col_sc: Optional[torch.Tensor] = None,
        h_col_sg: Optional[torch.Tensor] = None,
        h_r_tile: Optional[torch.Tensor] = None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
    ):
        if use_tk_attn_sync_before_qkv_fwd(debug_name):
            if os.environ.get(
                'USE_TK_ATTN_SYNC_BEFORE_QKV_FWD_CURRENT_STREAM', '0'
            ) == '1':
                torch.cuda.current_stream(input.device).synchronize()
            else:
                torch.cuda.synchronize(input.device)
        M, K = input.shape
        total_out = q_dim + k_dim + v_dim
        rope_live64_cs = None
        rope_applied = False
        rope_packed_applied = False
        rope_live64_applied = False
        debug_call_id = (
            _next_tk_attn_debug_call('qkv_fwd')
            if (
                use_tk_attn_debug_finite()
                or os.environ.get("USE_TK_DEBUG_ATTN_FINITE", "0") == "1"
                or bool(os.environ.get("USE_TK_DEBUG_QKV_FWD_STAGE", "").strip())
            )
            else None
        )
        qkv_weight_quant_keepalive = None
        from .tk_gemm import (
            reset_localcta_v4_cpp_only_override,
            set_localcta_v4_cpp_only_override,
        )
        _qkv_v4_cpp_token = None
        if os.environ.get('USE_TK_LOCALCTA_V4_QKV_CPP_ONLY', '1') == '1':
            _qkv_v4_cpp_token = set_localcta_v4_cpp_only_override(True)

        # Avoid redundant .contiguous().to(bf16) — check first
        inp = input if (input.is_contiguous() and input.dtype == torch.bfloat16) \
            else input.contiguous().to(torch.bfloat16)
        nw = norm_weight.detach() if norm_weight.dtype == torch.bfloat16 \
            else norm_weight.detach().to(torch.bfloat16)
        _tk_stage_trace('qkv_fwd_sub', 'input_ready', debug_name)
        if debug_call_id is not None:
            _attn_debug_check_finite(f'qkv_fwd[{debug_call_id}].input', inp)
            _attn_debug_check_finite(f'qkv_fwd[{debug_call_id}].norm_weight', nw)
            _attn_debug_check_finite(f'qkv_fwd[{debug_call_id}].w_qkv', w_qkv)
        qkv_timings = [] if use_tk_qkv_debug_timings_for(debug_name) else None
        qkv_last_event = None

        def _qkv_mark(name: str) -> None:
            nonlocal qkv_last_event
            if qkv_timings is None:
                return
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream())
            if qkv_last_event is not None:
                qkv_timings.append((name, qkv_last_event, event))
            qkv_last_event = event

        _qkv_mark("input_ready")

        # 1+2. Fused RMSNorm + amax + NVFP4 quant
        te_fused = _get_te_fused()

        # 3+4. Weight quantization + GEMM
        w_bf16 = w_qkv.detach() if (w_qkv.is_contiguous() and w_qkv.dtype == torch.bfloat16) \
            else w_qkv.detach().contiguous().to(torch.bfloat16)
        N_dims = [q_dim, k_dim, v_dim]
        from .tk_gemm import (
            _get_tk,
            _get_tk_quant_for_gemm,
            get_tk_localcta_variant,
            use_tk_localcta_v4_cpp_only,
            use_tk_localcta_v4_sg_direct_consumers,
            use_tk_qkv_bf16_dgrad,
            use_tk_qkv_bf16_underflow_rescue,
        )
        use_localcta = use_tk_localcta_forward_for_m(M)
        tk_mod = _get_tk()
        use_localcta_direct_forward = use_localcta and (
            use_tk_qkv_localcta_direct_forward_debug()
            or (
                get_tk_localcta_variant() == 'v4'
                and use_tk_localcta_v4_cpp_only()
            )
        )
        use_localcta_direct_prod = use_localcta and use_tk_localcta_direct_contract()
        use_localcta_v4_raw_forward = (
            use_localcta
            and get_tk_localcta_variant() == 'v4'
            and use_tk_localcta_v4_sg_direct_consumers()
            and not use_localcta_direct_forward
        )
        qkv_weight_quant_keepalive = None
        ctx.h_tile = h_row_fp4 is not None and h_row_fp4.numel() != 0
        ctx.cde_exact = (
            cde_row_rms_partial is not None
            and cde_row_rms_partial.numel() != 0
        )
        if ctx.h_tile and ctx.cde_exact:
            raise RuntimeError("exact C/D/E and H tile QKV carriers are mutually exclusive")
        if ctx.cde_exact:
            if use_localcta and get_tk_localcta_variant() != 'v4':
                raise RuntimeError("exact C/D/E localCTA QKV support requires variant v4")
            if use_localcta and use_tk_localcta_direct_contract():
                raise RuntimeError(
                    "exact C/D/E does not support the localCTA direct-TE contract"
                )
            if _nvfp4_quantizer_extras_enabled("activation"):
                raise RuntimeError("exact C/D/E does not support NVFP4 activation RHT/SR")
            if not use_tk_quant():
                raise RuntimeError("exact C/D/E requires the native TK v5 quantizer")
            expected_partial_width = K // (256 if use_localcta else 32)
            if (
                not cde_row_rms_partial.is_cuda
                or not cde_row_rms_partial.is_contiguous()
                or cde_row_rms_partial.dtype != torch.float32
                or tuple(cde_row_rms_partial.shape) != (M, expected_partial_width)
            ):
                raise RuntimeError(
                    "exact C/D/E row RMS partial must be contiguous CUDA float32 "
                    f"{(M, expected_partial_width)}, "
                    f"got shape={tuple(cde_row_rms_partial.shape)} "
                    f"dtype={cde_row_rms_partial.dtype}"
                )
            if K > 4096 or K % 128:
                raise RuntimeError(
                    f"exact C/D/E row RMS reducer requires 128-aligned K <= 4096, got {K}"
                )
            _trace_backend_choice(
                'localcta_exact_cde_qkv' if use_localcta
                else 'v5_exact_cde_qkv',
                'native',
            )

        def _quantize_from_cde_row_rms_partial():
            if use_localcta:
                tk_q_cde = _get_tk_quant_for_gemm()
                result = tk_q_cde.tk_rmsnorm_quantize_from_row_rms_partial_final_sg(
                    inp,
                    nw,
                    cde_row_rms_partial,
                    float(epsilon),
                    True,
                    use_tk_qkv_localcta_encode_centric(),
                )
                quantized = _TKQuantized(
                    result[0], result[1], result[4],
                    result[2], result[3], result[5],
                    keepalive=_result_keepalive(result, 7),
                )
                return quantized, result[6]
            tk_q_cde = _get_tk_quant()
            if not hasattr(
                tk_q_cde, 'tk_fused_norm_quantize_from_row_rms_partial'
            ):
                raise RuntimeError(
                    "exact C/D/E requires "
                    "tk_fused_norm_quantize_from_row_rms_partial"
                )
            result = tk_q_cde.tk_fused_norm_quantize_from_row_rms_partial(
                inp,
                nw,
                cde_row_rms_partial,
                float(epsilon),
                False,
                True,
            )
            quantized = _TKQuantized(
                result[0], result[1], result[4],
                result[2], result[3], result[4],
                keepalive=_result_keepalive(result, 6),
            )
            return quantized, result[5]

        if ctx.h_tile:
            if M < 256:
                raise RuntimeError("H QKV carrier requires M >= 256")
            x_nvfp4 = _TKQuantized(
                h_row_fp4, h_row_sc, h_row_sg,
                h_col_fp4, h_col_sc, h_col_sg,
            )
            inv_rms = h_r_tile

        if use_localcta:
            if (
                _nvfp4_quantizer_extras_enabled("activation")
                and get_tk_localcta_variant() != 'v4'
            ):
                _check_nvfp4_native_extras_supported(
                    "activation", "localCTA QKV forward activation producer"
                )
            _check_nvfp4_native_extras_supported("weight", "localCTA/v4 QKV forward grouped weight producer")
            tk_q = _get_tk_quant_for_gemm()
            if use_localcta_direct_forward:
                prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
                try:
                    overlap_qkv_w = use_tk_qkv_localcta_weight_overlap()
                    if overlap_qkv_w:
                        s0 = torch.cuda.current_stream()
                        s1 = _get_ms_stream()
                        s1.wait_stream(s0)
                        _record_tensors_on_stream(w_bf16, s1)
                        with torch.cuda.stream(s1):
                            _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                            group_result = (
                                _localcta_group_quantize_weights_2d(w_bf16, N_dims)
                                if use_tk_localcta_2d_weight_quant()
                                else tk_q.tk_group_quantize_for_gemm_direct(w_bf16, N_dims)
                            )
                            _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    if ctx.h_tile:
                        x_result = None
                    elif ctx.cde_exact:
                        x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
                        x_result = None
                    elif use_tk_qkv_localcta_fused_rmsnorm_quant():
                        x_nvfp4, inv_rms = _fast_rmsnorm_quantize_localcta_v4_opt(
                            inp,
                            nw,
                            float(epsilon),
                            nvfp4_role="activation",
                            encode_centric_override=use_tk_qkv_localcta_encode_centric(),
                        )
                        x_result = None
                    else:
                        normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                        qkv_activation_path = "localCTA QKV activation producer"
                        needs_paired_col_rht = _localcta_require_paired_col_rht(
                            qkv_activation_path
                        )
                        qkv_encode_centric = use_tk_qkv_localcta_encode_centric()
                        native_paired_col_rht = (
                            needs_paired_col_rht
                            and tk_q.supports_paired_col_rht_direct_forward()
                            and qkv_encode_centric == use_nvfp4_encode_centric()
                        )
                        if native_paired_col_rht:
                            _trace_backend_choice(
                                "localcta_rht_qkv_carrier",
                                "native_route_matched_paired",
                            )
                            x_result = (
                                tk_q.tk_quantize_for_gemm_direct_forward_paired_col_rht(
                                    normed,
                                    True,
                                    qkv_encode_centric,
                                )
                            )
                            x_nvfp4 = _localcta_quantized_from_result(x_result)
                        else:
                            x_result = tk_q.tk_quantize_for_gemm_direct_forward(
                                normed,
                                True,
                                qkv_encode_centric,
                            )
                            # Keep the validated direct-producer row byte exact.
                            # Older extensions attach the paired fixed-sign RHT
                            # column with a second quantization call.
                            if needs_paired_col_rht:
                                _trace_backend_choice(
                                    "localcta_rht_qkv_carrier",
                                    "python_two_pass_fallback",
                                )
                            x_nvfp4 = _localcta_replace_col_with_paired_rht(
                                _localcta_quantized_from_result(x_result),
                                normed,
                                path=qkv_activation_path,
                            )
                        x_result = None
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                    _qkv_mark("act_quant")
                    if overlap_qkv_w:
                        s0.wait_stream(s1)
                        # These tensors are allocated on the producer stream but
                        # consumed by the QKV GEMM and backward on this stream.
                        # Track that use before forward-only row references die.
                        _record_tensors_on_stream(group_result, s0)
                        _qkv_mark("weight_quant_wait")
                    else:
                        _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                        group_result = (
                            _localcta_group_quantize_weights_2d(w_bf16, N_dims)
                            if use_tk_localcta_2d_weight_quant()
                            else tk_q.tk_group_quantize_for_gemm_direct(w_bf16, N_dims)
                        )
                        _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
                        _qkv_mark("weight_quant")
                finally:
                    _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
                x_col_sg = (
                    x_result[5]
                    if x_result is not None and len(x_result) > 5 and torch.is_tensor(x_result[5]) and x_result[5].numel() > 0
                    else (x_result[4] if x_result is not None else None)
                )
                if x_result is not None:
                    x_nvfp4 = _TKQuantized(
                        x_result[0], x_result[1], x_result[4],
                        x_result[2], x_result[3], x_col_sg,
                    )
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, col_sg_cat, _, wc_sg_cols = \
                    group_result[:8]
                wc_fp4_col_cat = group_result[8] if len(group_result) > 8 else None
                wc_sc_col_cat = group_result[9] if len(group_result) > 9 else None
            elif use_localcta_v4_raw_forward:
                if ctx.cde_exact:
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                elif not ctx.h_tile:
                    _tk_stage_trace('qkv_fwd_sub', 'norm_start', debug_name)
                    normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                    _tk_stage_trace('qkv_fwd_sub', 'norm_done', debug_name)
                prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
                try:
                    _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                    group_result = (
                        _localcta_group_quantize_weights_2d(w_bf16, N_dims)
                        if use_tk_localcta_2d_weight_quant()
                        else tk_q.tk_group_quantize_for_gemm_direct(w_bf16, N_dims)
                    )
                    _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
                finally:
                    _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
                if not ctx.h_tile and not ctx.cde_exact:
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    x_nvfp4 = _qkv_forward_quantize(
                        normed,
                        input_quantizer,
                        use_localcta=True,
                    )
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, col_sg_cat, _, wc_sg_cols = \
                    group_result[:8]
                wc_fp4_col_cat = group_result[8] if len(group_result) > 8 else None
                wc_sc_col_cat = group_result[9] if len(group_result) > 9 else None
            else:
                use_localcta_fused_qkv_producer = use_tk_localcta_fused() and not (
                    get_tk_localcta_variant() == 'v4'
                    and not use_tk_localcta_v4_qkv_fused_producer()
                )
                if ctx.cde_exact:
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                elif use_localcta_direct_prod and not ctx.h_tile:
                    _tk_stage_trace('qkv_fwd_sub', 'norm_start', debug_name)
                    normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                    _tk_stage_trace('qkv_fwd_sub', 'norm_done', debug_name)
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    x_nvfp4 = _qkv_forward_quantize(normed, input_quantizer)
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                elif use_tk_qkv_localcta_tk_prepared_activation() and not ctx.h_tile:
                    _tk_stage_trace('qkv_fwd_sub', 'norm_start', debug_name)
                    normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                    _tk_stage_trace('qkv_fwd_sub', 'norm_done', debug_name)
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    x_nvfp4 = _fast_quantize_tk_standalone_localcta_prepared(normed)
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                elif (
                    not ctx.h_tile
                    and use_localcta_fused_qkv_producer
                    and hasattr(tk_q, 'tk_fused_norm_quantize')
                ):
                    prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
                    try:
                        _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                        result = tk_q.tk_fused_norm_quantize(
                            inp,
                            nw,
                            float(epsilon),
                            False,
                            True,
                            use_tk_qkv_localcta_encode_centric(),
                        )
                        _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                    finally:
                        _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
                    x_nvfp4 = _TKQuantized(
                        result[0], result[1], result[4],
                        result[2], result[3], result[5],
                        keepalive=_result_keepalive(result, 7),
                    )
                    inv_rms = result[6]
                elif not ctx.h_tile:
                    _tk_stage_trace('qkv_fwd_sub', 'norm_start', debug_name)
                    normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                    _tk_stage_trace('qkv_fwd_sub', 'norm_done', debug_name)
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    x_nvfp4 = _qkv_forward_quantize(
                        normed,
                        input_quantizer,
                        use_localcta=use_localcta,
                    )
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                if (
                    use_tk_qkv_localcta_fast_weights()
                    and not use_tk_localcta_2d_weight_quant()
                ):
                    prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
                    try:
                        _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                        group_result = tk_q.tk_group_quantize_for_gemm_fast(w_bf16, N_dims)
                        _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
                    finally:
                        _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
                    wc_fp4_row = group_result[0]
                    wc_sc_row = group_result[8]
                    fwd_b_sg = group_result[2]
                    wc_fp4_cols = group_result[3]
                    wc_sc_cols = group_result[9]
                    col_sg_cat = group_result[5]
                    wc_sg_cols = group_result[7]
                    wc_fp4_col_cat = None
                    wc_sc_col_cat = None
                else:
                    prev_scale_num = _set_localcta_qkv_scale_num(tk_q)
                    try:
                        _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                        group_result = (
                            _localcta_group_quantize_weights_2d(w_bf16, N_dims)
                            if use_tk_localcta_2d_weight_quant()
                            else tk_q.tk_group_quantize_for_gemm(w_bf16, N_dims)
                        )
                        _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
                    finally:
                        _restore_localcta_qkv_scale_num(tk_q, prev_scale_num)
                    wc_fp4_row, wc_sc_row, fwd_b_sg, \
                        wc_fp4_cols, wc_sc_cols, col_sg_cat, _, wc_sg_cols = \
                        group_result[:8]
                    wc_fp4_col_cat = group_result[8] if len(group_result) > 8 else None
                    wc_sc_col_cat = group_result[9] if len(group_result) > 9 else None
            _tk_stage_trace('qkv_fwd_sub', 'act_ready', debug_name)
            _tk_debug_print('qkv_fwd_sub', 'act_ready', debug_name)
            _tk_stage_trace('qkv_fwd_sub', 'weight_ready', debug_name)
            _tk_debug_print('qkv_fwd_sub', 'weight_ready', debug_name)
        elif use_tk_quant() and use_tk_ms():
            # ── Multi-stream path: input quant on s0 ∥ weight quant on s1 ──
            tk_q = _get_tk_quant()
            s0 = torch.cuda.current_stream()
            s1 = _get_ms_stream()

            # Phase 1: Launch weight quant on s1
            s1.wait_stream(s0)
            with torch.cuda.stream(s1):
                _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                quant_result = _regular_tk_group_quantize_qkv_weights(
                    tk_q, w_bf16, N_dims, owner_key=debug_name
                )
                _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, dgrad_b_sg, sg_cat, qkv_weight_quant_keepalive = \
                    quant_result
            wc_fp4_col_cat = None
            wc_sc_col_cat = None

            # Phase 2: Input RMSNorm + quant on s0 (overlaps with s1)
            # Step 1: RMSNorm only → get normed + inv_rms (1 kernel)
            if not ctx.h_tile:
                _tk_stage_trace('qkv_fwd_sub', 'norm_start', debug_name)
                if ctx.cde_exact:
                    x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
                    normed = None
                else:
                    normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                _tk_stage_trace('qkv_fwd_sub', 'norm_done', debug_name)
                # Step 2: Quantize normed input with TK-swizzled layout
                if not ctx.cde_exact:
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    x_nvfp4 = _qkv_forward_quantize(
                        normed,
                        input_quantizer,
                        use_localcta=False,
                    )
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)

            # Sync s1 → s0 before using weight results
            s0.wait_stream(s1)
            _record_tensors_on_stream(quant_result, s0)
            _tk_stage_trace('qkv_fwd_sub', 'act_ready', debug_name)
            _tk_stage_trace('qkv_fwd_sub', 'weight_ready', debug_name)
            _tk_debug_print('qkv_fwd_sub', 'act_ready', debug_name)
            _tk_debug_print('qkv_fwd_sub', 'weight_ready', debug_name)
        else:
            # ── Single-stream path ──
            qkv_weight_quant_keepalive = None
            use_tk_fused_norm_quant = (
                use_tk_qkv_fused_norm_quant()
                and use_tk_quant()
                and not use_localcta
                and not ctx.h_tile
            )
            if ctx.cde_exact:
                _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                x_nvfp4, inv_rms = _quantize_from_cde_row_rms_partial()
                _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
            elif use_tk_fused_norm_quant:
                tk_q = _get_tk_quant()
                if _nvfp4_quantizer_extras_enabled("activation"):
                    if not hasattr(tk_q, 'tk_fused_norm_quantize_opt'):
                        raise RuntimeError(
                            'USE_TK_QKV_FUSED_NORM_QUANT=1 with NVFP4 RHT/SR requires '
                            'tk_fused_norm_quantize_opt in the standalone TK quant module'
                        )
                    axes = _nvfp4_native_rht_axes_for_role("activation")
                    if axes != "row":
                        raise NotImplementedError(
                            "regular TK QKV fused RMSNorm quant currently supports "
                            "NVFP4_RHT_AXES=row only"
                        )
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    result = tk_q.tk_fused_norm_quantize_opt(
                        inp,
                        nw,
                        float(epsilon),
                        True,
                        use_nvfp4_encode_centric(),
                        use_nvfp4_data_stochastic_rounding_for_role("activation"),
                        use_nvfp4_scale_stochastic_rounding_for_role("activation"),
                        axes,
                        use_nvfp4_rht_for_role("activation") and _nvfp4_rht_random_sign_mask(),
                        _nvfp4_rng_seed(),
                        _nvfp4_rng_subsequence_base(),
                    )
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                    x_nvfp4 = _TKQuantized(
                        result[0], result[1], result[4],
                        result[2], result[3], result[5],
                        keepalive=_result_keepalive(result, 7),
                    )
                    inv_rms = result[6]
                else:
                    if not hasattr(tk_q, 'tk_fused_norm_quantize'):
                        raise RuntimeError(
                            'USE_TK_QKV_FUSED_NORM_QUANT=1 requires tk_fused_norm_quantize '
                            'in the standalone TK quant module'
                        )
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                    result = tk_q.tk_fused_norm_quantize(
                        inp, nw, float(epsilon), False, True
                    )
                    _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
                    x_nvfp4 = _TKQuantized(
                        result[0], result[1], result[4],
                        result[2], result[3], result[4],
                        keepalive=_result_keepalive(result, 6),
                    )
                    inv_rms = result[5]
            elif not ctx.h_tile:
                # Step 1: RMSNorm only → get normed + inv_rms (1 kernel)
                _tk_stage_trace('qkv_fwd_sub', 'norm_start', debug_name)
                normed, inv_rms = te_fused.fused_rmsnorm_only(inp, nw, float(epsilon))
                _tk_stage_trace('qkv_fwd_sub', 'norm_done', debug_name)

                # Step 2: Quantize normed input into TK-consumable layout. This can
                # optionally use the TE/custom swizzled path to isolate activation
                # quantization from the grouped GEMM numerics.
                _tk_stage_trace('qkv_fwd_sub', 'act_quant_start', debug_name)
                x_nvfp4 = _qkv_forward_quantize(
                    normed,
                    input_quantizer,
                    use_localcta=False,
                )
                _tk_stage_trace('qkv_fwd_sub', 'act_quant_done', debug_name)
            _tk_stage_trace('qkv_fwd_sub', 'act_ready', debug_name)
            _tk_debug_print('qkv_fwd_sub', 'act_ready', debug_name)

            # Weight quant
            if use_tk_quant():
                tk_q = _get_tk_quant()
                _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                quant_result = _regular_tk_group_quantize_qkv_weights(
                    tk_q, w_bf16, N_dims, owner_key=debug_name
                )
                _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, dgrad_b_sg, sg_cat, qkv_weight_quant_keepalive = \
                    quant_result
            else:
                _check_nvfp4_native_extras_supported("weight", "fused extension QKV grouped weight quantizer")
                fp4_ext = _get_fp4_ext()
                _tk_stage_trace('qkv_fwd_sub', 'weight_quant_start', debug_name)
                wc_fp4_row, wc_sc_row, fwd_b_sg, \
                    wc_fp4_cols, wc_sc_cols, dgrad_b_sg, sg_cat, _ = \
                    fp4_ext.group_nvfp4_quantize_tk(w_bf16, N_dims)
                _tk_stage_trace('qkv_fwd_sub', 'weight_quant_done', debug_name)
            wc_fp4_col_cat = None
            wc_sc_col_cat = None
            _tk_stage_trace('qkv_fwd_sub', 'weight_ready', debug_name)
            _tk_debug_print('qkv_fwd_sub', 'weight_ready', debug_name)

        use_small_m_plain_qkv = (not use_localcta and M < 256)
        x_fp4, x_sc, x_sg = x_nvfp4._tk_row
        if use_localcta_direct_prod and (not torch.is_tensor(x_sg) or x_sg.dim() != 2):
            x_sg = _const_sg_grid(x_sg.reshape(-1)[0], M, K)
        _tk_qkv_forward_stage_probe(
            "pre_gemm",
            debug_call_id,
            debug_name,
            (
                ("x_fp4", x_fp4),
                ("x_sc", x_sc),
                ("x_sg", x_sg),
                ("w_fp4", locals().get("wc_fp4_row")),
                ("w_sc", locals().get("wc_sc_row")),
                ("w_sg", locals().get("fwd_b_sg")),
            ),
        )
        _tk_stage_trace('qkv_fwd_sub', 'gemm_start', debug_name)
        _tk_debug_print('qkv_fwd_sub', 'gemm_start', debug_name)
        if use_small_m_plain_qkv:
            normed_fwd = normed if 'normed' in locals() else te_fused.fused_rmsnorm_only(
                inp, nw, float(epsilon)
            )[0]
            normed_pad = _pad_rows_bf16(normed_fwd, 256)
            x_nvfp4_pad = _qkv_forward_quantize(
                normed_pad,
                input_quantizer,
                use_localcta=False,
            )
            wq_bf16, kv_rest = w_bf16.split([q_dim, k_dim + v_dim], dim=0)
            wk_bf16, wv_bf16 = kv_rest.split([k_dim, v_dim], dim=0)
            wq_nvfp4 = _fast_quantize(
                wq_bf16,
                weight_quantizer,
                tk_swizzle=True,
                use_localcta_override=False,
            )
            wk_nvfp4 = _fast_quantize(
                wk_bf16,
                weight_quantizer,
                tk_swizzle=True,
                use_localcta_override=False,
            )
            wv_nvfp4 = _fast_quantize(
                wv_bf16,
                weight_quantizer,
                tk_swizzle=True,
                use_localcta_override=False,
            )
            xq_pad = torch.empty(normed_pad.size(0), q_dim, dtype=torch.bfloat16, device=inp.device)
            xk_pad = torch.empty(normed_pad.size(0), k_dim, dtype=torch.bfloat16, device=inp.device)
            xv_pad = torch.empty(normed_pad.size(0), v_dim, dtype=torch.bfloat16, device=inp.device)
            tk_forward_gemm(x_nvfp4_pad, wq_nvfp4, xq_pad, use_localcta=False)
            tk_forward_gemm(x_nvfp4_pad, wk_nvfp4, xk_pad, use_localcta=False)
            tk_forward_gemm(x_nvfp4_pad, wv_nvfp4, xv_pad, use_localcta=False)
            xq = xq_pad[:M]
            xk = xk_pad[:M]
            xv = xv_pad[:M]
            x_nvfp4 = x_nvfp4_pad
            x_fp4, x_sc, x_sg = x_nvfp4._tk_row
            wc_fp4_row = None
            wc_sc_row = None
            fwd_b_sg = None
            wc_fp4_cols = [wq_nvfp4._tk_col[0], wk_nvfp4._tk_col[0], wv_nvfp4._tk_col[0]]
            wc_sc_cols = [wq_nvfp4._tk_col[1], wk_nvfp4._tk_col[1], wv_nvfp4._tk_col[1]]
            wc_sg_cols = None
            wc_fp4_col_cat = None
            wc_sc_col_cat = None
            def _flat_sg(sg):
                if not torch.is_tensor(sg):
                    return torch.tensor([float(sg)], dtype=torch.float32, device=inp.device)
                return sg.to(torch.float32).reshape(-1)
            sg_cat = torch.cat([
                _flat_sg(wq_nvfp4._tk_col[2]),
                _flat_sg(wk_nvfp4._tk_col[2]),
                _flat_sg(wv_nvfp4._tk_col[2]),
            ], dim=0)
            qkv_weight_quant_keepalive = (wq_nvfp4, wk_nvfp4, wv_nvfp4)
        elif use_localcta_direct_forward:
            from .tk_gemm import _get_tk_localcta_direct
            tk_localcta_direct = _get_tk_localcta_direct()
            if tk_localcta_direct is None:
                raise RuntimeError("direct localCTA GEMM module failed to load")
            x_sg_direct = x_sg
            if (
                get_tk_localcta_variant() == 'v4'
                and torch.is_tensor(x_sg_direct)
                and x_sg_direct.dim() == 2
                and hasattr(tk_localcta_direct, 'prepare_outer_sg')
            ):
                x_sg_direct = tk_localcta_direct.prepare_outer_sg(
                    x_sg_direct,
                    x_fp4.size(0) // 256,
                    True,
            )
            if use_tk_qkv_forward_cat_debug():
                _trace_backend_choice('localcta_qkv_fwd', 'cat_debug_grouped_gemm')
                if rope_freqs_cis is not None:
                    _trace_backend_choice('localcta_qkv_fwd_rope_skip', 'cat_debug_grouped_gemm')
                y_cat = torch.empty(M, total_out, dtype=torch.bfloat16, device=x_fp4.device)
                tk_localcta_direct.nvfp4_grouped_gemm(
                    x_fp4, x_sc, x_sg_direct, wc_fp4_row, wc_sc_row, fwd_b_sg,
                    y_cat
                )
                xq, xk, xv = y_cat.split(N_dims, dim=1)
            else:
                xq = torch.empty(M, q_dim, dtype=torch.bfloat16, device=x_fp4.device)
                xk = torch.empty(M, k_dim, dtype=torch.bfloat16, device=x_fp4.device)
                xv = torch.empty(M, v_dim, dtype=torch.bfloat16, device=x_fp4.device)
                qkv_gemm_rope_live64 = getattr(tk_localcta_direct, 'nvfp4_grouped_gemm_rope_live64', None)
                qkv_gemm_rope = getattr(tk_localcta_direct, 'nvfp4_grouped_gemm_rope', None)
                from .tk_gemm import _get_tk_plain
                qkv_native_rope = getattr(
                    _get_tk_plain(), 'nvfp4_forward_rope_packed_qk', None
                )
                _qkv_mark("pre_gemm_setup")
                if (
                    rope_freqs_cis is not None
                    and qkv_native_rope is not None
                    and _tk_qkv_rope_packed_supported(
                        M, K, q_dim, k_dim, v_dim,
                        int(rope_head_dim), int(rope_seq_len), rope_freqs_cis
                    )
                ):
                    tk_localcta_direct.nvfp4_grouped_gemm(
                        x_fp4, x_sc, x_sg_direct, wc_fp4_row, wc_sc_row, fwd_b_sg,
                        xq, xk, xv
                    )
                    rope_packed_cs = _get_tk_packed_rope_cs(
                        rope_freqs_cis, int(rope_seq_len)
                    )
                    xq_rotated = torch.empty_like(xq)
                    xk_rotated = torch.empty_like(xk)
                    qkv_native_rope(
                        xq, xk, rope_packed_cs,
                        int(rope_seq_len), int(rope_head_dim),
                        xq_rotated, xk_rotated,
                    )
                    xq, xk = xq_rotated, xk_rotated
                    _trace_backend_choice('localcta_qkv_fwd_rope', 'native_qk_post')
                    rope_applied = True
                    rope_packed_applied = True
                elif (
                    rope_freqs_cis is not None
                    and use_tk_qkv_rope_epilogue()
                    and qkv_gemm_rope_live64 is not None
                    and _tk_qkv_rope_live64_supported(
                        M, K, q_dim, k_dim, v_dim, int(rope_head_dim), int(rope_seq_len), rope_freqs_cis
                    )
                ):
                    rope_live64_cs = _get_tk_live64_rope_cs(rope_freqs_cis, int(rope_seq_len))
                    qkv_gemm_rope_live64(
                        x_fp4, x_sc, x_sg_direct, wc_fp4_row, wc_sc_row, fwd_b_sg,
                        xq, xk, xv, rope_live64_cs, int(rope_seq_len)
                    )
                    _trace_backend_choice('localcta_qkv_fwd_rope', 'epilogue_live64')
                    rope_applied = True
                    rope_live64_applied = True
                elif (
                    rope_freqs_cis is not None
                    and use_tk_qkv_rope_epilogue()
                    and qkv_gemm_rope is not None
                    and _tk_qkv_rope_generic_supported(
                        M, K, q_dim, k_dim, v_dim, int(rope_head_dim), int(rope_seq_len), rope_freqs_cis
                    )
                ):
                    rope_cos, rope_sin = _get_tk_rope_tables(rope_freqs_cis, int(rope_seq_len))
                    rope_rotary_dim = min(int(rope_freqs_cis.size(1)) * 2, int(rope_head_dim))
                    qkv_gemm_rope(
                        x_fp4, x_sc, x_sg_direct, wc_fp4_row, wc_sc_row, fwd_b_sg,
                        xq, xk, xv, rope_cos, rope_sin,
                        int(rope_seq_len), int(rope_head_dim), int(rope_rotary_dim)
                    )
                    _trace_backend_choice('localcta_qkv_fwd_rope', 'epilogue_generic')
                    rope_applied = True
                elif rope_freqs_cis is not None:
                    raise RuntimeError("localCTA v4 QKV RoPE epilogue requested but backend/support check failed")
                else:
                    tk_localcta_direct.nvfp4_grouped_gemm(
                        x_fp4, x_sc, x_sg_direct, wc_fp4_row, wc_sc_row, fwd_b_sg,
                        xq, xk, xv
                    )
            _set_last_qkv_forward_debug_payload({
                'mode': 'localcta_direct_debug',
                'cat_debug': use_tk_qkv_forward_cat_debug(),
                'x_fp4': x_fp4.detach(),
                'x_sc': x_sc.detach(),
                'x_sg': x_sg.detach() if torch.is_tensor(x_sg) else x_sg,
                'x_sg_direct': x_sg_direct.detach() if torch.is_tensor(x_sg_direct) else x_sg_direct,
                'wc_fp4_row': wc_fp4_row.detach(),
                'wc_sc_row': wc_sc_row.detach(),
                'fwd_b_sg': fwd_b_sg.detach() if torch.is_tensor(fwd_b_sg) else fwd_b_sg,
                'xq': xq.detach(),
                'xk': xk.detach(),
                'xv': xv.detach(),
            })
        elif use_tk_qkv_forward_cat_debug():
            _trace_backend_choice('localcta_qkv_fwd', 'cat_debug_grouped_gemm')
            qkv_gemm = tk_mod.nvfp4_grouped_gemm
            if use_tk_qkv_forward_nopdl():
                qkv_gemm = getattr(tk_mod, 'nvfp4_grouped_gemm_nopdl', qkv_gemm)
            qkv_gemm_rope_packed_split = getattr(
                tk_mod, 'nvfp4_grouped_gemm_rope_packed_split', None
            )
            if (
                rope_freqs_cis is not None
                and qkv_gemm_rope_packed_split is not None
                and _tk_qkv_rope_packed_supported(
                    M, K, q_dim, k_dim, v_dim,
                    int(rope_head_dim), int(rope_seq_len), rope_freqs_cis
                )
            ):
                rope_packed_cs = _get_tk_packed_rope_cs(rope_freqs_cis, int(rope_seq_len))
                rope_rotary_dim = int(rope_freqs_cis.size(1)) * 2
                xq = torch.empty(M, q_dim, dtype=torch.bfloat16, device=x_fp4.device)
                xk = torch.empty(M, k_dim, dtype=torch.bfloat16, device=x_fp4.device)
                xv = torch.empty(M, v_dim, dtype=torch.bfloat16, device=x_fp4.device)
                _tk_stage_trace('qkv_fwd_sub', 'qkv_rope_packed_split_epilogue_start', debug_name)
                qkv_gemm_rope_packed_split(
                    x_fp4, x_sc, x_sg, wc_fp4_row, wc_sc_row, fwd_b_sg,
                    xq, xk, xv, rope_packed_cs,
                    int(rope_seq_len), int(rope_head_dim), int(rope_rotary_dim),
                )
                _tk_stage_trace('qkv_fwd_sub', 'qkv_rope_packed_split_epilogue_done', debug_name)
                _trace_backend_choice('regular_tk_qkv_fwd_rope', 'epilogue_packed_split')
                rope_applied = True
                rope_packed_applied = True
            else:
                if rope_freqs_cis is not None:
                    raise RuntimeError("TK packed split RoPE epilogue requested but backend/support check failed")
                y_cat = torch.empty(M, total_out, dtype=torch.bfloat16, device=x_fp4.device)
                qkv_gemm(
                    x_fp4, x_sc, x_sg, wc_fp4_row, wc_sc_row, fwd_b_sg,
                    y_cat
                )
                xq, xk, xv = y_cat.split(N_dims, dim=1)
        else:
            # Forward GEMM: use split_D to write Q/K/V directly (avoids split+contiguous copy)
            xq = torch.empty(M, q_dim, dtype=torch.bfloat16, device=x_fp4.device)
            xk = torch.empty(M, k_dim, dtype=torch.bfloat16, device=x_fp4.device)
            xv = torch.empty(M, v_dim, dtype=torch.bfloat16, device=x_fp4.device)
            qkv_gemm = tk_mod.nvfp4_grouped_gemm
            if use_tk_qkv_forward_nopdl():
                qkv_gemm = getattr(tk_mod, 'nvfp4_grouped_gemm_nopdl', qkv_gemm)
            qkv_gemm_rope_live64 = getattr(tk_mod, 'nvfp4_grouped_gemm_rope_live64', None)
            qkv_gemm_rope = getattr(tk_mod, 'nvfp4_grouped_gemm_rope', None)
            if (
                rope_freqs_cis is not None
                and use_tk_qkv_rope_epilogue()
                and qkv_gemm_rope_live64 is not None
                and _tk_qkv_rope_live64_supported(
                    M, K, q_dim, k_dim, v_dim, int(rope_head_dim), int(rope_seq_len), rope_freqs_cis
                )
            ):
                rope_live64_cs = _get_tk_live64_rope_cs(rope_freqs_cis, int(rope_seq_len))
                qkv_gemm_rope_live64(
                    x_fp4, x_sc, x_sg, wc_fp4_row, wc_sc_row, fwd_b_sg,
                    xq, xk, xv, rope_live64_cs, int(rope_seq_len)
                )
                _trace_backend_choice('localcta_qkv_fwd_rope', 'epilogue_live64')
                rope_applied = True
                rope_live64_applied = True
            elif (
                rope_freqs_cis is not None
                and use_tk_qkv_rope_epilogue()
                and qkv_gemm_rope is not None
                and _tk_qkv_rope_generic_supported(
                    M, K, q_dim, k_dim, v_dim, int(rope_head_dim), int(rope_seq_len), rope_freqs_cis
                )
            ):
                rope_cos, rope_sin = _get_tk_rope_tables(rope_freqs_cis, int(rope_seq_len))
                rope_rotary_dim = min(int(rope_freqs_cis.size(1)) * 2, int(rope_head_dim))
                qkv_gemm_rope(
                    x_fp4, x_sc, x_sg, wc_fp4_row, wc_sc_row, fwd_b_sg,
                    xq, xk, xv, rope_cos, rope_sin,
                    int(rope_seq_len), int(rope_head_dim), int(rope_rotary_dim)
                )
                _trace_backend_choice('localcta_qkv_fwd_rope', 'epilogue_generic')
                rope_applied = True
            elif rope_freqs_cis is not None:
                raise RuntimeError("TK QKV RoPE epilogue requested but backend/support check failed")
            else:
                qkv_gemm(
                    x_fp4, x_sc, x_sg, wc_fp4_row, wc_sc_row, fwd_b_sg,
                    xq, xk, xv  # use_split_D: Q->D, K->D_K, V->D_V
                )
        if use_localcta and use_tk_qkv_forward_cat_debug():
            y_cat_payload = y_cat if 'y_cat' in locals() else torch.cat([xq, xk, xv], dim=1)
            _set_last_qkv_forward_debug_payload({
                'mode': 'localcta_fast',
                'cat_debug': use_tk_qkv_forward_cat_debug(),
                'x_fp4': x_fp4.detach(),
                'x_sc': x_sc.detach(),
                'x_sg': x_sg.detach() if torch.is_tensor(x_sg) else x_sg,
                'wc_fp4_row': wc_fp4_row.detach(),
                'wc_sc_row': wc_sc_row.detach(),
                'fwd_b_sg': fwd_b_sg.detach() if torch.is_tensor(fwd_b_sg) else fwd_b_sg,
                'w_bf16': w_bf16.detach(),
                'y_cat': y_cat_payload.detach(),
                'xq': xq.detach(),
                'xk': xk.detach(),
                'xv': xv.detach(),
                'n_dims': list(N_dims),
                'shape': {'M': M, 'K': K, 'N_total': total_out},
                })
        _tk_stage_trace('qkv_fwd_sub', 'gemm_done', debug_name)
        _tk_debug_print('qkv_fwd_sub', 'gemm_done', debug_name)
        _qkv_mark("gemm")
        _tk_qkv_forward_stage_probe(
            "post_gemm",
            debug_call_id,
            debug_name,
            (("xq", xq), ("xk", xk), ("xv", xv)),
        )
        qkv_weight_quant_state_lease = _regular_qkv_weight_quant_state_lease(
            qkv_weight_quant_keepalive
        )
        if qkv_weight_quant_state_lease is not None:
            qkv_weight_quant_state_lease.record_forward_consumed()

        _retain_tk_qkv_forward_graph_state(
            inp,
            nw,
            locals().get('normed'),
            inv_rms,
            x_nvfp4,
            wc_fp4_row,
            wc_sc_row,
            fwd_b_sg,
            wc_fp4_cols,
            wc_sc_cols,
            locals().get('wc_sg_cols'),
            wc_fp4_col_cat,
            wc_sc_col_cat,
            qkv_weight_quant_keepalive,
        )

        # Save per-split colwise weight tensors for dgrad.
        _tk_stage_trace('qkv_fwd_sub', 'ctx_save_start', debug_name)
        localcta_variant = os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower()
        weight_sg = col_sg_cat if use_localcta else sg_cat
        ctx.save_for_backward(inp, nw, inv_rms, weight_sg)
        ctx._wc_fp4_cols = wc_fp4_cols
        ctx._wc_sc_cols = wc_sc_cols
        ctx._wc_sg_cols = wc_sg_cols if use_localcta else None
        ctx._wc_fp4_col_cat = wc_fp4_col_cat if use_localcta else None
        ctx._wc_sc_col_cat = wc_sc_col_cat if use_localcta else None
        ctx._x_nvfp4 = x_nvfp4  # holds refs to row+col tensors
        ctx._qkv_weight_quant_keepalive = qkv_weight_quant_keepalive if use_tk_quant() else None
        ctx._qkv_weight_quant_state_lease = qkv_weight_quant_state_lease
        save_qkv_bf16_rescue = use_localcta and use_tk_qkv_bf16_underflow_rescue()
        ctx._w_qkv_bf16 = w_bf16 if (
            use_localcta
            or save_qkv_bf16_rescue
            or use_tk_qkv_bf16_dgrad()
            or use_small_m_plain_qkv
        ) else None
        ctx.N_dims = N_dims
        ctx._K = K
        ctx._use_localcta = use_localcta
        ctx.rope_applied = rope_applied
        ctx.rope_packed_applied = rope_packed_applied
        ctx.rope_live64_applied = rope_live64_applied
        ctx.rope_freqs_cis = rope_freqs_cis if rope_applied else None
        ctx.rope_batch_size = int(rope_batch_size) if rope_applied else 0
        ctx.rope_seq_len = int(rope_seq_len) if rope_applied else 0
        ctx.rope_head_dim = int(rope_head_dim) if rope_applied else 0

        ctx.grad_quantizer = grad_quantizer
        ctx.workspace = workspace
        ctx.epsilon = epsilon
        ctx.q_dim = q_dim
        ctx.k_dim = k_dim
        ctx.v_dim = v_dim
        ctx._attn_debug_qkv_call_id = debug_call_id
        ctx._lbt_debug_name = debug_name
        _tk_stage_trace('qkv_fwd_sub', 'ctx_save_done', debug_name)
        _qkv_mark("ctx_save")
        _tk_stage_trace('qkv_fwd_sub', 'debug_tail_start', debug_name)
        if use_tk_attn_safe_qkv_fwd_sync() or use_tk_attn_sync_qkv_fwd(debug_name):
            if os.environ.get(
                'USE_TK_ATTN_SYNC_QKV_FWD_CURRENT_STREAM', '0'
            ) == '1':
                torch.cuda.current_stream(xq.device).synchronize()
            else:
                torch.cuda.synchronize(xq.device)
        if debug_call_id is not None:
            _attn_debug_check_finite(f'qkv_fwd[{debug_call_id}].xq', xq)
            _attn_debug_check_finite(f'qkv_fwd[{debug_call_id}].xk', xk)
            _attn_debug_check_finite(f'qkv_fwd[{debug_call_id}].xv', xv)
        if _attn_capture_path():
            _append_attn_capture({
                "event": "qkv_forward",
                "debug_name": debug_name,
                "debug_call_id": debug_call_id,
                "inputs": {
                    "input": _tensor_capture_stats(inp),
                    "norm_weight": _tensor_capture_stats(nw),
                    "inv_rms": _tensor_capture_stats(inv_rms),
                },
                "payloads": {
                    "x_fp4": _tensor_capture_stats(x_fp4),
                    "x_sc": _tensor_capture_stats(x_sc),
                    "x_sg": _tensor_capture_stats(x_sg),
                    "wc_fp4_row": _tensor_capture_stats(wc_fp4_row),
                    "wc_sc_row": _tensor_capture_stats(wc_sc_row),
                    "fwd_b_sg": _tensor_capture_stats(fwd_b_sg),
                },
                "outputs": {
                    "xq": _tensor_capture_stats(xq),
                    "xk": _tensor_capture_stats(xk),
                    "xv": _tensor_capture_stats(xv),
                },
            })
        if _attn_layout_event_enabled("qkv_forward_return", debug_name):
            _append_attn_layout({
                "event": "qkv_forward_return",
                "debug_name": debug_name,
                **_tensor_layout_group(
                    (("xq", xq), ("xk", xk), ("xv", xv))
                ),
            })
        qkv_dump_dir = os.environ.get("USE_TK_QKV_DEBUG_FWD_DUMP_DIR", "").strip()
        qkv_dump_match = os.environ.get("USE_TK_QKV_DEBUG_FWD_DUMP_MATCH", "").strip()
        if qkv_dump_dir and (not qkv_dump_match or qkv_dump_match in debug_name):
            once_key = f"{debug_name}::qkv_forward_dump"
            dumped = getattr(_FusedQKVFunction_TK, "_debug_forward_dumped", None)
            if dumped is None:
                dumped = set()
                setattr(_FusedQKVFunction_TK, "_debug_forward_dumped", dumped)
            if once_key not in dumped:
                dumped.add(once_key)
                os.makedirs(qkv_dump_dir, exist_ok=True)
                safe_name = debug_name.replace("/", "_").replace(":", "_").replace(".", "_")
                torch.save({
                    "debug_name": debug_name,
                    "input": inp.detach().cpu(),
                    "inv_rms": inv_rms.detach().cpu(),
                    "xq": xq.detach().cpu(),
                    "xk": xk.detach().cpu(),
                    "xv": xv.detach().cpu(),
                }, os.path.join(qkv_dump_dir, f"{safe_name}.pt"))

        if _qkv_v4_cpp_token is not None:
            reset_localcta_v4_cpp_only_override(_qkv_v4_cpp_token)
        _tk_stage_trace('qkv_fwd_sub', 'debug_tail_done', debug_name)
        _qkv_mark("debug_tail")
        _emit_qkv_debug_timings_once(debug_name, "forward", M, K, total_out, qkv_timings)
        return xq, xk, xv

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        workspace = ctx.workspace

        saved = ctx.saved_tensors
        input, norm_weight, inv_rms, weight_sg = saved

        wc_fp4_cols = ctx._wc_fp4_cols
        ctx._wc_fp4_cols = None
        wc_sc_cols = ctx._wc_sc_cols
        ctx._wc_sc_cols = None
        wc_sg_cols = getattr(ctx, '_wc_sg_cols', None)
        ctx._wc_sg_cols = None
        wc_fp4_col_cat = getattr(ctx, '_wc_fp4_col_cat', None)
        ctx._wc_fp4_col_cat = None
        wc_sc_col_cat = getattr(ctx, '_wc_sc_col_cat', None)
        ctx._wc_sc_col_cat = None
        x_nvfp4 = ctx._x_nvfp4
        ctx._x_nvfp4 = None
        qkv_weight_quant_state_lease = getattr(
            ctx, '_qkv_weight_quant_state_lease', None
        )
        ctx._qkv_weight_quant_keepalive = None
        ctx._qkv_weight_quant_state_lease = None
        w_qkv_bf16 = getattr(ctx, '_w_qkv_bf16', None)
        ctx._w_qkv_bf16 = None
        K = ctx._K
        M = input.shape[0]
        use_localcta = getattr(ctx, '_use_localcta', False)
        paired_rht_carrier = (
            use_localcta
            and os.environ.get(
                'USE_TK_LOCALCTA_VARIANT', 'v1'
            ).strip().lower() == 'v4'
            and use_tk_localcta_paired_rht_carrier()
        )

        q_d = ctx.q_dim; k_d = ctx.k_dim; v_d = ctx.v_dim
        N_total = q_d + k_d + v_d
        N_dims = list(ctx.N_dims)
        debug_call_id = getattr(ctx, '_attn_debug_qkv_call_id', None)
        debug_name = getattr(ctx, '_lbt_debug_name', None)
        retain_incoming_grads = (
            os.environ.get('USE_TK_QKV_RETAIN_INCOMING_GRADS', '0') == '1'
        )
        retain_incoming_filter = os.environ.get(
            'USE_TK_QKV_RETAIN_INCOMING_GRADS_FILTER', ''
        ).strip()
        retain_incoming_call = os.environ.get(
            'USE_TK_QKV_RETAIN_INCOMING_GRADS_CALL', ''
        ).strip()
        if (
            retain_incoming_grads
            and (
                not retain_incoming_filter
                or (
                    isinstance(debug_name, str)
                    and retain_incoming_filter in debug_name
                )
            )
            and (
                not retain_incoming_call
                or debug_call_id == int(retain_incoming_call)
            )
            and len(_TK_QKV_DEBUG_RETAINED_INCOMING_GRADS)
            < int(os.environ.get('USE_TK_QKV_RETAIN_INCOMING_GRADS_MAX', '1'))
        ):
            retained_names = {
                value.strip().lower()
                for value in os.environ.get(
                    'USE_TK_QKV_RETAIN_INCOMING_GRADS_TENSORS', 'q'
                ).split(',')
                if value.strip()
            }
            if not retained_names.issubset({'q', 'k', 'v'}):
                raise ValueError(
                    'USE_TK_QKV_RETAIN_INCOMING_GRADS_TENSORS must be a '
                    'comma-separated subset of q,k,v'
                )
            retained = tuple(
                tensor
                for name, tensor in (
                    ('q', grad_q), ('k', grad_k), ('v', grad_v)
                )
                if name in retained_names
            )
            _TK_QKV_DEBUG_RETAINED_INCOMING_GRADS.append(retained)
        own_incoming_grads = (
            os.environ.get('USE_TK_QKV_OWN_INCOMING_GRADS', '0') == '1'
        )
        own_incoming_filter = os.environ.get(
            'USE_TK_QKV_OWN_INCOMING_GRADS_FILTER', ''
        ).strip()
        own_incoming_call = os.environ.get(
            'USE_TK_QKV_OWN_INCOMING_GRADS_CALL', ''
        ).strip()
        fence_incoming_grads = (
            os.environ.get('USE_TK_QKV_INCOMING_GRAD_EVENT_FENCE', '0') == '1'
        )
        if (
            fence_incoming_grads
            and (
                not own_incoming_filter
                or (
                    isinstance(debug_name, str)
                    and own_incoming_filter in debug_name
                )
            )
            and (
                not own_incoming_call
                or debug_call_id == int(own_incoming_call)
            )
        ):
            incoming_grad_event = torch.cuda.Event()
            incoming_grad_event.record(torch.cuda.current_stream(grad_q.device))
            torch.cuda.current_stream(grad_q.device).wait_event(
                incoming_grad_event
            )
        if (
            own_incoming_grads
            and (
                not own_incoming_filter
                or (
                    isinstance(debug_name, str)
                    and own_incoming_filter in debug_name
                )
            )
            and (
                not own_incoming_call
                or debug_call_id == int(own_incoming_call)
            )
        ):
            owned_names = {
                value.strip().lower()
                for value in os.environ.get(
                    'USE_TK_QKV_OWN_INCOMING_GRADS_TENSORS', 'q,k,v'
                ).split(',')
                if value.strip()
            }
            if not owned_names.issubset({'q', 'k', 'v'}):
                raise ValueError(
                    'USE_TK_QKV_OWN_INCOMING_GRADS_TENSORS must be a '
                    'comma-separated subset of q,k,v'
                )
            retain_owned_grads = (
                os.environ.get(
                    'USE_TK_QKV_OWN_INCOMING_GRADS_RETAIN', '1'
                )
                == '1'
            )
            owned_grad_action = os.environ.get(
                'USE_TK_QKV_OWN_INCOMING_GRADS_ACTION', 'clone'
            ).strip().lower()
            if owned_grad_action not in {'clone', 'allocate'}:
                raise ValueError(
                    'USE_TK_QKV_OWN_INCOMING_GRADS_ACTION must be clone or '
                    'allocate'
                )

            def _own_incoming_grad(tensor: torch.Tensor) -> torch.Tensor:
                if owned_grad_action == 'allocate':
                    return torch.empty_like(
                        tensor, memory_format=torch.contiguous_format
                    )
                prefix_elements = int(
                    os.environ.get(
                        'USE_TK_QKV_OWN_INCOMING_GRADS_PREFIX_ELEMENTS', '0'
                    )
                )
                if prefix_elements:
                    if retain_owned_grads:
                        raise ValueError(
                            'prefix gradient ownership probes require '
                            'USE_TK_QKV_OWN_INCOMING_GRADS_RETAIN=0'
                        )
                    return tensor.reshape(-1)[:prefix_elements].clone()
                return tensor.clone(memory_format=torch.contiguous_format)

            owned_grad_q = None
            owned_grad_k = None
            owned_grad_v = None
            if 'q' in owned_names:
                owned_grad_q = _own_incoming_grad(grad_q)
                if retain_owned_grads:
                    grad_q = owned_grad_q
            if 'k' in owned_names:
                owned_grad_k = _own_incoming_grad(grad_k)
                if retain_owned_grads:
                    grad_k = owned_grad_k
            if 'v' in owned_names:
                owned_grad_v = _own_incoming_grad(grad_v)
                if retain_owned_grads:
                    grad_v = owned_grad_v
        rope_live64_cs = None
        rope_seq_len = 0
        if getattr(ctx, 'rope_applied', False):
            rope_seq_len = int(getattr(ctx, 'rope_seq_len', 0))
            if getattr(ctx, 'rope_live64_applied', False):
                rope_live64_cs = _get_tk_live64_rope_cs(ctx.rope_freqs_cis, rope_seq_len)
                _trace_backend_choice('localcta_qkv_bwd_rope', 'inverse_live64_requested')
        _tk_stage_trace('qkv_bwd', 'start', debug_name)
        _tk_debug_print('qkv_bwd', 'start', debug_name)
        if _attn_layout_event_enabled("qkv_backward_entry", debug_name):
            _append_attn_layout({
                "event": "qkv_backward_entry",
                "debug_name": debug_name,
                **_tensor_layout_group(
                    (("grad_q", grad_q), ("grad_k", grad_k), ("grad_v", grad_v))
                ),
            })
        _maybe_print_qkv_grad_layouts('raw', grad_q, grad_k, grad_v, debug_name)
        raw_grad_stage = (
            "raw_qkv_grads"
            if debug_call_id is None
            else f"raw_qkv_grads[call={debug_call_id}]"
        )
        _tk_attn_debug_assert_finite(
            raw_grad_stage,
            debug_name,
            (("grad_q", grad_q), ("grad_k", grad_k), ("grad_v", grad_v)),
        )

        # Small-M runs have historically needed eager materialization here to
        # avoid upstream storage reuse. Large-M training is stable without the
        # extra clone, and the clone cost is visible in the step trace.
        if M < 256:
            gq = _as_contiguous_bf16(grad_q).clone()
            gk = _as_contiguous_bf16(grad_k).clone()
            gv = _as_contiguous_bf16(grad_v).clone()
        elif (
            _is_zero_stride_scalar_bf16_grad(grad_q)
            and _is_zero_stride_scalar_bf16_grad(grad_k)
            and _is_zero_stride_scalar_bf16_grad(grad_v)
        ):
            # Benchmark sum-loss gradients are scalar zero-stride expansions.
            # Keep them lazy so tk_gemm can build the v4 split3 package without
            # first writing three dense BF16 grad tensors.
            gq, gk, gv = grad_q, grad_k, grad_v
        else:
            gq = _as_qkv_grad_bf16_for_quant(grad_q)
            gk = _as_qkv_grad_bf16_for_quant(grad_k)
            gv = _as_qkv_grad_bf16_for_quant(grad_v)
        _maybe_print_qkv_grad_layouts('normalized', gq, gk, gv, debug_name)
        _tk_attn_debug_assert_finite(
            "normalized_qkv_grads",
            debug_name,
            (("gq", gq), ("gk", gk), ("gv", gv)),
        )
        if debug_call_id is not None:
            _attn_debug_check_finite(f'qkv_bwd[{debug_call_id}].grad_q', gq)
            _attn_debug_check_finite(f'qkv_bwd[{debug_call_id}].grad_k', gk)
            _attn_debug_check_finite(f'qkv_bwd[{debug_call_id}].grad_v', gv)
        if (
            getattr(ctx, 'rope_applied', False)
            and (
                not getattr(ctx, 'rope_live64_applied', False)
                or paired_rht_carrier
            )
        ):
            rope_packed_applied = bool(getattr(ctx, 'rope_packed_applied', False))
            gq, gk = _apply_inverse_tk_rotary_qk(
                gq,
                gk,
                ctx.rope_freqs_cis,
                int(getattr(ctx, 'rope_batch_size', 0)),
                int(getattr(ctx, 'rope_seq_len', 0)),
                int(getattr(ctx, 'rope_head_dim', 0)),
                rope_packed_applied,
                owner_key=debug_name,
            )
            if rope_packed_applied and use_cuda_graph():
                gv = _stable_packed_graph_grad_v(gv, owner_key=debug_name)
            if getattr(ctx, 'rope_live64_applied', False):
                # The fused inverse-live64 split3 producer predates paired
                # column RHT and silently omits it.  Restore Q/K in BF16, then
                # route all three gradients through the generic opt producer.
                rope_live64_cs = None
                _trace_backend_choice(
                    'localcta_qkv_bwd_rope',
                    'paired_rht_carrier_inverse_bf16',
                )
            elif not rope_packed_applied:
                _trace_backend_choice('localcta_qkv_bwd_rope', 'inverse_generic_applied')
        _tk_attn_debug_assert_finite(
            "inverse_rope_outputs",
            debug_name,
            (("gq", gq), ("gk", gk), ("gv", gv)),
        )

        # ── Zero-copy backward: pass grad splits directly ──
        # Profiling showed dy_cat copy costs 0.867ms (28% of bwd) at M=65536.
        # Instead, pass gq/gk/gv directly — they're already contiguous from autograd.

        can_use_split_weight_col_ref = (
            use_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
            and os.environ.get('USE_TK_LOCALCTA_V4_QKV_CPP_ONLY', '1') == '1'
            and wc_fp4_cols is not None
            and wc_sc_cols is not None
            and wc_sg_cols is not None
        )
        can_use_graph_regular_split_weight_col_ref = (
            not use_localcta
            and use_cuda_graph()
            and wc_fp4_cols is not None
            and wc_sc_cols is not None
        )
        if wc_fp4_col_cat is not None and wc_sc_col_cat is not None:
            col_fp4_cat, col_sc_cat = wc_fp4_col_cat, wc_sc_col_cat
        elif (
            can_use_split_weight_col_ref
            or can_use_graph_regular_split_weight_col_ref
        ):
            # The v4 fused backward and graph-mode regular backward consume the
            # live per-split weight col tensors through _tk_col_splits.  The
            # first split is only a shape/device carrier for _tk_col; avoiding
            # the concatenation also avoids pinning the first warmup's weight
            # contents behind either weight-split cache.
            col_fp4_cat = wc_fp4_cols[0]
            col_sc_cat = wc_sc_cols[0]
        else:
            # Cache weight col cat across backward calls when the backend does
            # not hand back the grouped col payload directly.
            if not hasattr(_FusedQKVFunction_TK, '_w_col_cache'):
                _FusedQKVFunction_TK._w_col_cache = {}
            _w_cache = _FusedQKVFunction_TK._w_col_cache
            _w_key = id(wc_fp4_cols[0])
            cached_w = _w_cache.get(_w_key)
            if cached_w is None:
                col_fp4_cat = torch.cat(
                    [_packed_fp4_contiguous(fp4).view(torch.uint8) for fp4 in wc_fp4_cols], dim=1
                ).view(torch.float4_e2m1fn_x2)
                col_sc_cat = torch.cat(
                    [sc.contiguous().view(torch.uint8) for sc in wc_sc_cols], dim=1
                ).view(torch.float8_e4m3fn)
                cached_w = (col_fp4_cat, col_sc_cat)
                _w_cache[_w_key] = cached_w
            col_fp4_cat, col_sc_cat = cached_w
        if use_localcta and wc_sg_cols is not None:
            w_sg_payload = wc_sg_cols
        else:
            w_sg_payload = weight_sg if use_localcta else weight_sg.float()

        class _ColRef:
            __slots__ = ('_tk_col', '_tk_col_splits', '_tk_col_sg_splits')
            def __init__(self, c, splits=None, sg_splits=None):
                self._tk_col = c
                self._tk_col_splits = splits
                self._tk_col_sg_splits = sg_splits

        split_payload = None
        split_sg_payload = None
        if (
            use_localcta or can_use_graph_regular_split_weight_col_ref
        ) and wc_fp4_cols is not None and wc_sc_cols is not None:
            split_payload = (wc_fp4_cols, wc_sc_cols)
            if wc_sg_cols is not None:
                split_sg_payload = wc_sg_cols
            elif can_use_graph_regular_split_weight_col_ref:
                split_sg_payload = [
                    w_sg_payload[index:index + 1]
                    for index in range(len(N_dims))
                ]

        w_col = _ColRef(
            (col_fp4_cat, col_sc_cat, w_sg_payload),
            splits=split_payload,
            sg_splits=split_sg_payload,
        )

        from .tk_gemm import localcta_v4_cpp_only_scope
        if os.environ.get('USE_TK_LOCALCTA_V4_QKV_CPP_ONLY', '1') == '1':
            with localcta_v4_cpp_only_scope(True):
                grad_input, grad_w_qkv, grad_norm_weight, rescue_info = _qkv_bwd_graphed(
                    None, w_col, N_dims, x_nvfp4,
                    input, norm_weight, inv_rms, K, M,
                    w_bf16=w_qkv_bf16,
                    grad_splits=(gq, gk, gv),
                    debug_name=debug_name,
                    rope_live64_cs=rope_live64_cs,
                    rope_seq_len=rope_seq_len,
                    h_tile=getattr(ctx, 'h_tile', False),
                )
        else:
            grad_input, grad_w_qkv, grad_norm_weight, rescue_info = _qkv_bwd_graphed(
                None, w_col, N_dims, x_nvfp4,
                input, norm_weight, inv_rms, K, M,
                w_bf16=w_qkv_bf16,
                grad_splits=(gq, gk, gv),
                debug_name=debug_name,
                rope_live64_cs=rope_live64_cs,
                rope_seq_len=rope_seq_len,
                h_tile=getattr(ctx, 'h_tile', False),
            )
        if use_tk_attn_sync_qkv_bwd():
            torch.cuda.synchronize(grad_input.device)
        if qkv_weight_quant_state_lease is not None:
            qkv_weight_quant_state_lease.release_after_backward()
        _tk_stage_trace('qkv_bwd', 'end', debug_name)
        _tk_debug_print('qkv_bwd', 'end', debug_name)
        if debug_call_id is not None:
            _attn_debug_check_finite(f'qkv_bwd[{debug_call_id}].grad_input', grad_input)
            _attn_debug_check_finite(f'qkv_bwd[{debug_call_id}].grad_w_qkv', grad_w_qkv)
            _attn_debug_check_finite(f'qkv_bwd[{debug_call_id}].grad_norm_weight', grad_norm_weight)
        if _should_emit_localcta_function_grad_debug(debug_name):
            _emit_localcta_function_grad_debug(
                "qkv_backward_return",
                {
                    "debug_name": debug_name,
                    "grad_q": _tensor_debug_stats(gq),
                    "grad_k": _tensor_debug_stats(gk),
                    "grad_v": _tensor_debug_stats(gv),
                    "grad_input": _tensor_debug_stats(grad_input),
                    "grad_w_qkv": _tensor_debug_stats(grad_w_qkv),
                    "grad_norm_weight": _tensor_debug_stats(grad_norm_weight),
                    "rescue": rescue_info,
                },
            )

        # 24 inputs to forward
        if debug_name is not None:
            return (
                _maybe_own_qkv_grad_input(grad_input, M),            # input
                _maybe_clone_autograd_return(grad_w_qkv, M),         # w_qkv
                _maybe_clone_autograd_return(grad_norm_weight, M),   # norm_weight
                None,               # epsilon
                None, None, None,   # q_dim, k_dim, v_dim
                None, None, None, None,  # rope_freqs_cis, rope_batch_size, rope_seq_len, rope_head_dim
                None,               # input_quantizer
                None,               # weight_quantizer
                None,               # grad_quantizer
                None,               # workspace
                None,               # debug_name
                None, None, None, None, None, None, None,  # H carrier payload
                None,               # cde_row_rms_partial
            )
        return (
            _maybe_own_qkv_grad_input(grad_input, M),            # input
            _maybe_clone_autograd_return(grad_w_qkv, M),         # w_qkv
            _maybe_clone_autograd_return(grad_norm_weight, M),   # norm_weight
            None,               # epsilon
            None, None, None,   # q_dim, k_dim, v_dim
            None, None, None, None,  # rope_freqs_cis, rope_batch_size, rope_seq_len, rope_head_dim
            None,               # input_quantizer
            None,               # weight_quantizer
            None,               # grad_quantizer
            None,               # workspace
            None,               # debug_name
            None, None, None, None, None, None, None,  # H carrier payload
            None,               # cde_row_rms_partial
        )



# ---------------------------------------------------------------------------
# _WoFunction: custom autograd for wo output projection
# ---------------------------------------------------------------------------
class _WoFunction_TE(torch.autograd.Function):
    """Explicit forward/backward for wo projection: y = W_o @ x.

    Forward: quantize input + weight → GEMM
    Backward: quantize dY → dgrad GEMM (dx = W_o^T @ dY) + wgrad GEMM (dW = x^T @ dY)
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,           # (M, N_in) bf16 — attention output
        wo_weight: torch.Tensor,        # (N_out, N_in) bf16
        input_quantizer: NVFP4Quantizer,
        weight_quantizer: NVFP4Quantizer,
        grad_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
        debug_name: Optional[str] = None,
    ):
        M = input.shape[0]
        N_out = wo_weight.shape[0]

        inp = input if input.is_contiguous() else input.contiguous()

        # FP4 quantize input and weight (TK or TE scale layout)
        
        x_nvfp4 = _fast_quantize(inp, input_quantizer, tk_swizzle=False)
        w_nvfp4 = _fast_quantize(wo_weight, weight_quantizer, tk_swizzle=False)

        # GEMM: y = W_o @ x (TE path)
        y = torch.empty((M, N_out), dtype=torch.bfloat16, device=inp.device)
        tex.generic_gemm(
            w_nvfp4, True, x_nvfp4, False,
            y, None, TE_DType[torch.bfloat16],
            None, TE_DType[torch.bfloat16],
            False, None, False,
            workspace, workspace.shape[0], False, False,
        )

        # Only cache FP4 data — bf16 input not needed (dgrad/wgrad use FP4)
        ctx.w_nvfp4 = w_nvfp4
        ctx.x_nvfp4 = x_nvfp4
        ctx.grad_quantizer = grad_quantizer
        ctx.workspace = workspace

        return y

    @staticmethod
    def backward(ctx, grad_output):
        workspace = ctx.workspace

        w_nvfp4 = ctx.w_nvfp4
        x_nvfp4 = ctx.x_nvfp4

        dY = _as_contiguous_bf16(grad_output)

        # Quantize gradient (TK or TE scale layout)
        
        dY_nvfp4 = _fast_quantize(dY, ctx.grad_quantizer, tk_swizzle=False)

        # dgrad: dx = dY @ W_o
        dx = tex.generic_gemm(
            w_nvfp4, False, dY_nvfp4, False,
            None, None, TE_DType[torch.bfloat16],
            None, TE_DType[torch.bfloat16],
            False, None, False,
            workspace, workspace.shape[0], False, False,
        )[0]

        # wgrad: dW_o = dY^T @ x
        grad_w = tex.generic_gemm(
            x_nvfp4, False, dY_nvfp4, True,
            None, None, TE_DType[torch.bfloat16],
            None, TE_DType[torch.bfloat16],
            False, None, False,
            workspace, workspace.shape[0], False, False,
        )[0]

        # Free cached tensors
        ctx.w_nvfp4 = None
        ctx.x_nvfp4 = None

        return (
            dx,         # input
            grad_w,     # wo_weight
            None,       # input_quantizer
            None,       # weight_quantizer
            None,       # grad_quantizer
            None,       # workspace
        )

class _WoFunction_TK(torch.autograd.Function):
    """Explicit forward/backward for wo projection: y = W_o @ x.

    Forward: quantize input + weight → GEMM
    Backward: quantize dY → dgrad GEMM (dx = W_o^T @ dY) + wgrad GEMM (dW = x^T @ dY)
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,           # (M, N_in) bf16 — attention output
        wo_weight: torch.Tensor,        # (N_out, N_in) bf16
        input_quantizer: NVFP4Quantizer,
        weight_quantizer: NVFP4Quantizer,
        grad_quantizer: NVFP4Quantizer,
        workspace: torch.Tensor,
        debug_name: Optional[str] = None,
        residual: Optional[torch.Tensor] = None,
        h_gamma: Optional[torch.Tensor] = None,
        cde_emit: bool = False,
    ):
        input_nhsd_shape = tuple(input.shape) if input.dim() == 4 else None
        if input_nhsd_shape is not None:
            B, H, S, D = input_nhsd_shape
            M = B * S
            K_in = H * D
        else:
            M = input.shape[0]
            K_in = input.shape[-1]
        N_out = wo_weight.shape[0]
        residual_2d = None
        if torch.is_tensor(residual) and residual.numel() != 0:
            residual_2d = residual.reshape(M, N_out).contiguous()
            if residual_2d.dtype != torch.bfloat16:
                residual_2d = residual_2d.to(torch.bfloat16)
        ctx.has_residual = residual_2d is not None
        ctx.h_output = h_gamma is not None and h_gamma.numel() != 0
        ctx.cde_output = bool(cde_emit)
        if ctx.h_output and ctx.cde_output:
            raise RuntimeError("exact C/D/E and H Wo carriers are mutually exclusive")
        if ctx.h_output and residual_2d is None:
            raise RuntimeError("H Wo carrier requires the residual stream")
        if ctx.h_output and M < 256:
            raise RuntimeError("H Wo carrier requires M >= 256")
        use_localcta = use_tk_localcta_forward_for_m(M)
        if ctx.cde_output:
            if residual_2d is None:
                raise RuntimeError("exact C/D/E Wo producer requires the residual stream")
            if M % 256 or K_in != 4096 or N_out != 4096:
                raise RuntimeError(
                    "exact C/D/E Wo producer requires [M,4096,4096] with "
                    "M divisible by 256"
                )
            if not use_localcta:
                raise RuntimeError(
                    "exact Wo-to-FFN C/D/E is retained only for localCTA v4"
                )
            if get_tk_localcta_variant() != 'v4':
                raise RuntimeError("exact C/D/E Wo localCTA support requires variant v4")
            if use_tk_localcta_direct_contract():
                raise RuntimeError(
                    "exact C/D/E does not support the localCTA direct-TE contract"
                )
            if _nvfp4_quantizer_extras_enabled("activation"):
                raise RuntimeError("exact C/D/E does not support NVFP4 activation RHT/SR")
        use_small_m_plain_wo = (not use_localcta and M < 256)
        debug_call_id = _next_tk_attn_debug_call('wo_fwd') if use_tk_attn_debug_finite() else None
        wo_timings = [] if use_tk_wo_debug_timings_for(debug_name) else None
        wo_last_event = None

        def _wo_mark(name: str) -> None:
            nonlocal wo_last_event
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream())
            if wo_last_event is not None:
                wo_timings.append((name, wo_last_event, event))
            wo_last_event = event

        if wo_timings is not None:
            _wo_mark("start")

        use_localcta_v4 = (
            use_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
        )
        use_attn_layout_view = input_nhsd_shape is not None and use_tk_wo_attn_layout()
        use_regular_nhsd_wo_quant = (
            use_attn_layout_view
            and not use_localcta
            and use_tk_regular_wo_nhsd_quant()
        )
        if use_regular_nhsd_wo_quant and input_nhsd_shape is not None and input_nhsd_shape[-1] != 64:
            use_regular_nhsd_wo_quant = False
        use_nhsd_wo_quant = (
            use_attn_layout_view
            and (use_localcta_v4 or use_regular_nhsd_wo_quant)
            and not (
                use_localcta_v4
                and use_tk_localcta_paired_rht_carrier()
            )
        )
        overlap_localcta_v4_wo_weight_quant = (
            use_localcta_v4
            and use_nvfp4_rht_for_role("activation")
            and use_tk_localcta_v4_wo_rht_weight_quant_overlap()
        )
        if (
            overlap_localcta_v4_wo_weight_quant
            and _nvfp4_quantizer_extras_enabled("weight")
        ):
            raise RuntimeError(
                "localCTA v4 WO RHT weight overlap requires deterministic "
                "weight quantization without SR/RHT extras"
            )
        if overlap_localcta_v4_wo_weight_quant and use_cuda_graph():
            raise RuntimeError(
                "localCTA v4 WO RHT weight overlap is an eager-only candidate"
            )
        if use_attn_layout_view:
            B, H, S, D = input_nhsd_shape
            matrix_view = _nhsd_attention_output_matrix_view(input, B, H, S, D)
            if matrix_view is not None:
                inp = matrix_view
                use_nhsd_wo_quant = False
            elif use_nhsd_wo_quant:
                inp = input if input.is_contiguous() else input.contiguous()
            else:
                inp = input.transpose(1, 2).contiguous().view(B * S, H * D)
        elif input_nhsd_shape is not None:
            # Fallback keeps the old materialized [B,S,H,D] -> [B*S,H*D] contract.
            B, H, S, D = input_nhsd_shape
            inp = input.transpose(1, 2).contiguous().view(B * S, H * D)
        else:
            inp = input if input.is_contiguous() else input.contiguous()
        if wo_timings is not None:
            _wo_mark("input_ready")
        inp_fwd = _pad_rows_bf16(inp, 256) if use_small_m_plain_wo else inp
        if debug_call_id is not None:
            _attn_debug_check_finite(f'wo_fwd[{debug_call_id}].input', inp)
            _attn_debug_check_finite(f'wo_fwd[{debug_call_id}].wo_weight', wo_weight)

        # FP4 quantize input and weight
        if use_tk_localcta() and not use_localcta_v4:
            if use_nhsd_wo_quant:
                x_nvfp4 = (
                    _fast_quantize_tk_localcta_nhsd_wo(inp)
                    if use_localcta_v4 else _fast_quantize_tk_nhsd_wo(inp)
                )
            elif use_tk_wo_rowonly_input_quant():
                x_nvfp4 = _fast_quantize_te_rowonly_swizzled(inp, nvfp4_role="activation")
            else:
                x_nvfp4 = _fast_quantize_localcta_regular_hybrid(inp, nvfp4_role="activation")
            w_nvfp4 = _fast_quantize_localcta_regular_hybrid(wo_weight, nvfp4_role="weight")
        elif overlap_localcta_v4_wo_weight_quant:
            # WO's activation RHT producer is independent of deterministic
            # weight quantization.  Launch the latter on the established
            # weight stream and retain both source and result allocations on
            # every stream that touches them.  The wait remains before GEMM,
            # so this changes scheduling only, never quantization math or RNG
            # reservation order.
            s0 = torch.cuda.current_stream()
            s1 = _get_ms_stream()
            s1.wait_stream(s0)
            _record_tensors_on_stream((wo_weight,), s1)
            with torch.cuda.stream(s1):
                _tk_stage_trace('wo_fwd_sub', 'weight_quant_start', debug_name)
                w_nvfp4 = _fast_quantize(
                    wo_weight,
                    weight_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
                _tk_stage_trace('wo_fwd_sub', 'weight_quant_done', debug_name)
            _tk_stage_trace('wo_fwd_sub', 'input_quant_start', debug_name)
            if use_nhsd_wo_quant:
                x_nvfp4 = _fast_quantize_tk_localcta_nhsd_wo(inp)
            elif use_tk_wo_rowonly_input_quant() and use_localcta:
                x_nvfp4 = _fast_quantize_te_rowonly_swizzled(
                    inp, nvfp4_role="activation"
                )
            else:
                x_nvfp4 = _fast_quantize(
                    inp,
                    input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
            _tk_stage_trace('wo_fwd_sub', 'input_quant_done', debug_name)
            s0.wait_stream(s1)
            _record_tensors_on_stream(
                (
                    w_nvfp4._tk_row,
                    w_nvfp4._tk_col,
                    w_nvfp4._tk_row_chunk_sg,
                    w_nvfp4._tk_col_chunk_sg,
                    w_nvfp4._keepalive,
                ),
                s0,
            )
        elif use_tk_ms():
            # Multi-stream: input quant on s0 ∥ weight quant on s1
            s0 = torch.cuda.current_stream()
            s1 = _get_ms_stream()
            s1.wait_stream(s0)
            with torch.cuda.stream(s1):
                _tk_stage_trace('wo_fwd_sub', 'weight_quant_start', debug_name)
                w_nvfp4 = _fast_quantize(
                    wo_weight,
                    weight_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
                _tk_stage_trace('wo_fwd_sub', 'weight_quant_done', debug_name)
            _tk_stage_trace('wo_fwd_sub', 'input_quant_start', debug_name)
            if use_nhsd_wo_quant:
                x_nvfp4 = (
                    _fast_quantize_tk_localcta_nhsd_wo(inp)
                    if use_localcta_v4 else _fast_quantize_tk_nhsd_wo(inp)
                )
            elif use_tk_wo_rowonly_input_quant() and use_localcta:
                x_nvfp4 = _fast_quantize_te_rowonly_swizzled(inp, nvfp4_role="activation")
            else:
                x_nvfp4 = _fast_quantize(
                    inp,
                    input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
            _tk_stage_trace('wo_fwd_sub', 'input_quant_done', debug_name)
            s0.wait_stream(s1)
        else:
            _tk_stage_trace('wo_fwd_sub', 'input_quant_start', debug_name)
            if use_nhsd_wo_quant:
                x_nvfp4 = (
                    _fast_quantize_tk_localcta_nhsd_wo(inp)
                    if use_localcta_v4 else _fast_quantize_tk_nhsd_wo(inp)
                )
            elif use_tk_wo_rowonly_input_quant() and use_localcta:
                x_nvfp4 = _fast_quantize_te_rowonly_swizzled(inp, nvfp4_role="activation")
            else:
                x_nvfp4 = _fast_quantize(
                    inp,
                    input_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                )
            _tk_stage_trace('wo_fwd_sub', 'input_quant_done', debug_name)
            _tk_stage_trace('wo_fwd_sub', 'weight_quant_start', debug_name)
            w_nvfp4 = _fast_quantize(
                wo_weight,
                weight_quantizer,
                tk_swizzle=True,
                use_localcta_override=use_localcta,
            )
            _tk_stage_trace('wo_fwd_sub', 'weight_quant_done', debug_name)
        if wo_timings is not None:
            _wo_mark("quant_done")

        x_nvfp4_fwd = x_nvfp4
        if use_small_m_plain_wo:
            _tk_stage_trace('wo_fwd_sub', 'small_m_input_quant_start', debug_name)
            x_nvfp4_fwd = _fast_quantize(
                inp_fwd,
                input_quantizer,
                tk_swizzle=True,
                use_localcta_override=False,
            )
            _tk_stage_trace('wo_fwd_sub', 'small_m_input_quant_done', debug_name)

        # GEMM: y = W_o @ x
        y_fwd = torch.empty((x_nvfp4_fwd.shape[0], N_out), dtype=torch.bfloat16, device=inp.device)
        _tk_stage_trace('wo_fwd_sub', 'gemm_start', debug_name)
        if ctx.h_output:
            from .tk_gemm import tk_forward_gemm_h_carrier

            h_carrier = tk_forward_gemm_h_carrier(
                x_nvfp4_fwd,
                w_nvfp4,
                residual_2d,
                _as_contiguous_bf16(h_gamma),
                1.0e-5,
                out=y_fwd,
                use_localcta=use_localcta,
            )
        elif ctx.cde_output:
            _trace_backend_choice('localcta_exact_cde_wo', 'native')
            y_fwd, cde_row_rms_partial = tk_forward_gemm_residual_rms_partial(
                x_nvfp4_fwd,
                w_nvfp4,
                residual_2d,
                out=y_fwd,
                use_localcta=use_localcta,
            )
        elif residual_2d is not None:
            tk_forward_gemm_residual(
                x_nvfp4_fwd,
                w_nvfp4,
                residual_2d,
                y_fwd,
                use_localcta=use_localcta,
            )
        else:
            tk_forward_gemm(
                x_nvfp4_fwd, w_nvfp4, y_fwd, use_localcta=use_localcta
            )
        _tk_stage_trace('wo_fwd_sub', 'gemm_done', debug_name)
        _tk_stage_trace('wo_fwd_sub', 'tail_start', debug_name)
        y = y_fwd[:M]
        if wo_timings is not None:
            _wo_mark("gemm_done")
        if _debug_forward_ref_enabled(debug_name):
            ref_inp = (
                inp.transpose(1, 2).contiguous().view(M, K_in)
                if use_nhsd_wo_quant else inp
            )
            y_ref = torch.matmul(ref_inp.to(torch.bfloat16), wo_weight.detach().to(torch.bfloat16).transpose(0, 1))
            _append_attn_capture({
                "event": "wo_forward_ref",
                "debug_call_id": debug_call_id,
                "debug_name": debug_name,
                "stats": _tensor_delta_stats(y, y_ref),
            })

        _tk_stage_trace('wo_fwd_sub', 'ctx_save_start', debug_name)
        ctx.w_nvfp4 = w_nvfp4
        ctx.x_nvfp4 = x_nvfp4
        save_wo_bf16_rescue = use_tk_localcta() and use_tk_localcta_wo_bf16_underflow_rescue()
        save_wo_bf16_debug = _debug_dgrad_ref_enabled(debug_name)
        save_wo_wgrad_debug = _debug_wgrad_ref_enabled(debug_name)
        save_nhsd_bf16_for_bwd = (
            use_nhsd_wo_quant
            and use_localcta_v4
            and (
                use_tk_wo_bf16_wgrad()
                or os.environ.get('USE_TK_LOCALCTA_V4_WO_BF16_BWD', '0') == '1'
                or save_wo_bf16_debug
                or save_wo_wgrad_debug
                or (
                    save_wo_bf16_rescue
                    and use_tk_localcta_wo_prepared_split2_backward()
                )
            )
        )
        ctx.input_bf16 = None if use_nhsd_wo_quant else (
            inp if (use_tk_wo_bf16_wgrad() or use_localcta_v4 or use_small_m_plain_wo) else None
        )
        ctx.input_nhsd_bf16 = inp if save_nhsd_bf16_for_bwd else None
        ctx.localcta_input_bf16 = inp if (save_wo_bf16_rescue and not use_nhsd_wo_quant) else None
        ctx.wo_weight_bf16 = wo_weight.detach() if (save_wo_bf16_rescue or save_wo_bf16_debug) else None
        ctx._input_nhsd_shape = input_nhsd_shape
        ctx._use_small_m_plain_wo = use_small_m_plain_wo
        ctx.grad_quantizer = grad_quantizer
        ctx.workspace = workspace
        ctx._attn_debug_wo_call_id = debug_call_id
        ctx._lbt_debug_name = debug_name
        _tk_stage_trace('wo_fwd_sub', 'ctx_save_done', debug_name)
        if use_tk_attn_sync_wo_fwd():
            torch.cuda.synchronize(y.device)
        if debug_call_id is not None:
            _attn_debug_check_finite(f'wo_fwd[{debug_call_id}].output', y)
        _emit_wo_debug_timings_once(debug_name, "forward", M, K_in, N_out, wo_timings)
        _tk_stage_trace('wo_fwd_sub', 'tail_done', debug_name)

        if ctx.h_output:
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(*h_carrier[1:])
            return h_carrier
        if ctx.cde_output:
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(cde_row_rms_partial)
            return y, cde_row_rms_partial
        return y

    @staticmethod
    def backward(ctx, grad_output, *carrier_grads):
        workspace = ctx.workspace

        w_nvfp4 = ctx.w_nvfp4
        x_nvfp4 = ctx.x_nvfp4
        input_bf16 = ctx.input_bf16
        input_nhsd_bf16 = getattr(ctx, 'input_nhsd_bf16', None)
        localcta_input_bf16 = getattr(ctx, 'localcta_input_bf16', None)
        wo_weight_bf16 = getattr(ctx, 'wo_weight_bf16', None)

        input_nhsd_shape = getattr(ctx, '_input_nhsd_shape', None)
        dY = _as_contiguous_bf16(grad_output)
        if input_nhsd_shape is not None and dY.dim() == 3:
            B, _H, S, _D = input_nhsd_shape
            dY = dY.reshape(B * S, dY.shape[-1]).contiguous()
        use_localcta = use_tk_localcta_forward_for_m(dY.shape[0])
        use_small_m_plain_wo = getattr(ctx, '_use_small_m_plain_wo', False)
        use_localcta_v4 = (
            use_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
        )
        keep_nhsd_on_split2_wo_bwd = (
            input_bf16 is None
            and input_nhsd_bf16 is not None
            and use_localcta_v4
            and can_use_localcta_split2_wo_backward(dY)
        )
        use_split2_wo_bwd = can_use_localcta_split2_wo_backward(dY)
        skip_generic_v4_wo_dy_quant = (
            use_localcta_v4
            and not use_small_m_plain_wo
            and not use_tk_localcta_v4_fast_wo_dgrad()
            and not use_tk_localcta_v4_fast_wo_wgrad()
            and not use_tk_localcta_v4_cpp_only()
            and use_split2_wo_bwd
        )
        debug_call_id = getattr(ctx, '_attn_debug_wo_call_id', None)
        debug_name = getattr(ctx, '_lbt_debug_name', None)
        wo_sr_state = _localcta_wo_sr_state(debug_name, dY.device)
        _validate_checkpointed_localcta_wo_sr_route(
            wo_sr_state,
            skip_generic_v4_wo_dy_quant=skip_generic_v4_wo_dy_quant,
        )
        wo_timings = [] if use_tk_wo_debug_timings_for(debug_name) else None
        wo_last_event = None

        def _wo_mark(name: str) -> None:
            nonlocal wo_last_event
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream())
            if wo_last_event is not None:
                wo_timings.append((name, wo_last_event, event))
            wo_last_event = event

        if wo_timings is not None:
            _wo_mark("start")
        _tk_stage_trace('wo_bwd', 'start', debug_name)
        _tk_debug_print('wo_bwd', 'start', debug_name)
        _tk_attn_debug_assert_finite(
            "wo_backward_input", debug_name, (("grad_output", dY),)
        )
        if _attn_layout_event_enabled("wo_backward_entry", debug_name):
            _append_attn_layout({
                "event": "wo_backward_entry",
                "debug_name": debug_name,
                **_tensor_layout_group((("grad_output", grad_output), ("dY", dY))),
            })
        if debug_call_id is not None:
            _attn_debug_check_finite(f'wo_bwd[{debug_call_id}].grad_output', dY)

        # Quantize gradient (TK or TE scale layout)
        dY_nvfp4 = None
        if not skip_generic_v4_wo_dy_quant:
            dY_nvfp4 = _fast_quantize(
                dY,
                ctx.grad_quantizer,
                tk_swizzle=True,
                use_localcta_override=use_localcta,
                persistent_rng_state=wo_sr_state,
            )
        if wo_timings is not None:
            _wo_mark("dy_quant_done")
        if _attn_capture_path():
            if dY_nvfp4 is None:
                dY_nvfp4 = _fast_quantize(
                    dY,
                    ctx.grad_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=use_localcta,
                    persistent_rng_state=wo_sr_state,
                )
            dy_row_fp4, dy_row_sc, dy_row_sg = dY_nvfp4._tk_row
            dy_col_fp4, dy_col_sc, dy_col_sg = dY_nvfp4._tk_col
            w_col_fp4, w_col_sc, w_col_sg = w_nvfp4._tk_col
            x_col_fp4, x_col_sc, x_col_sg = x_nvfp4._tk_col
            _append_attn_capture({
                "event": "wo_quant",
                "debug_call_id": debug_call_id,
                "debug_name": debug_name,
                "stats": {
                    "dy_row_fp4": _tensor_capture_stats(dy_row_fp4),
                    "dy_row_sc": _tensor_capture_stats(dy_row_sc),
                    "dy_row_sg": _tensor_capture_stats(dy_row_sg),
                    "dy_col_fp4": _tensor_capture_stats(dy_col_fp4),
                    "dy_col_sc": _tensor_capture_stats(dy_col_sc),
                    "dy_col_sg": _tensor_capture_stats(dy_col_sg),
                    "w_col_fp4": _tensor_capture_stats(w_col_fp4),
                    "w_col_sc": _tensor_capture_stats(w_col_sc),
                    "w_col_sg": _tensor_capture_stats(w_col_sg),
                    "x_col_fp4": _tensor_capture_stats(x_col_fp4),
                    "x_col_sc": _tensor_capture_stats(x_col_sc),
                    "x_col_sg": _tensor_capture_stats(x_col_sg),
                },
            })

        wo_rescue = None
        if input_bf16 is None and input_nhsd_bf16 is not None and not keep_nhsd_on_split2_wo_bwd:
            B, H, S, D = getattr(ctx, '_input_nhsd_shape')
            input_bf16 = input_nhsd_bf16.transpose(1, 2).contiguous().view(B * S, H * D)

        split2_input_bf16 = localcta_input_bf16
        if (
            split2_input_bf16 is None
            and input_bf16 is None
            and input_nhsd_bf16 is not None
            and keep_nhsd_on_split2_wo_bwd
            and use_localcta_v4
        ):
            B, H, S, D = getattr(ctx, '_input_nhsd_shape')
            split2_input_bf16 = input_nhsd_bf16.transpose(1, 2).contiguous().view(B * S, H * D)

        use_v4_bf16_wo_bwd = (
            use_localcta_v4
            and os.environ.get('USE_TK_LOCALCTA_V4_WO_BF16_BWD', '0') == '1'
            and wo_weight_bf16 is not None
            and (input_bf16 is not None or split2_input_bf16 is not None)
        )
        if use_v4_bf16_wo_bwd:
            wgrad_input_bf16 = input_bf16 if input_bf16 is not None else split2_input_bf16
            dx = torch.matmul(
                dY.float(),
                wo_weight_bf16.float(),
            ).to(torch.bfloat16)
            grad_w = torch.matmul(
                dY.transpose(0, 1).float(),
                wgrad_input_bf16.float(),
            ).to(torch.bfloat16)
            wo_rescue = {
                'taken': True,
                'reason': 'forced_v4_bf16_wo_bwd',
                'path': 'bf16_wo_backward',
            }
        elif input_bf16 is None and use_split2_wo_bwd:
            dx, grad_w, wo_rescue = tk_localcta_split2_wo_backward(
                dY,
                x_nvfp4,
                w_nvfp4,
                input_bf16=split2_input_bf16,
                w_bf16=wo_weight_bf16,
            )
        elif (
            use_localcta
            and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
            and use_tk_localcta_v4_fast_wo_dgrad()
            and _localcta_v4_wo_outer_sg_pair(
                dY_nvfp4._tk_row, w_nvfp4._tk_col
            )
        ):
            tk = _get_tk()
            dy_row_fp4, dy_row_sc, dy_row_sg = dY_nvfp4._tk_row
            w_col_fp4, w_col_sc, w_col_sg = w_nvfp4._tk_col
            dx = torch.empty(
                dY.shape[0],
                w_nvfp4.shape[1],
                dtype=torch.bfloat16,
                device=dY.device,
            )
            _localcta_v4_wo_outer_sg_gemm(
                tk,
                dy_row_fp4, dy_row_sc, dy_row_sg,
                w_col_fp4, w_col_sc, w_col_sg,
                dx,
            )
            _trace_backend_choice('localcta_wo_dgrad', 'fast_outer_sg')
            # wgrad: dW_o = dY^T @ x
            if (
                use_localcta
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and use_tk_localcta_v4_fast_wo_wgrad()
                and dY.shape[0] % 256 == 0
                and _localcta_v4_wo_outer_sg_pair(
                    dY_nvfp4._tk_col, x_nvfp4._tk_col
                )
            ):
                tk = _get_tk()
                dy_col_fp4, dy_col_sc, dy_col_sg = dY_nvfp4._tk_col
                x_col_fp4, x_col_sc, x_col_sg = x_nvfp4._tk_col
                grad_w = torch.empty(
                    dY.shape[1],
                    x_nvfp4.shape[1],
                    dtype=torch.bfloat16,
                    device=dY.device,
                )
                _localcta_v4_wo_outer_sg_gemm(
                    tk,
                    dy_col_fp4, dy_col_sc, dy_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    grad_w,
                )
                _trace_backend_choice('localcta_wo_wgrad', 'fast_outer_sg')
            elif (
                use_localcta
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and use_tk_localcta_v4_cpp_only()
            ):
                grad_w = tk_wgrad_gemm(x_nvfp4, dY_nvfp4, use_localcta=True)
            elif use_localcta and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4' and input_bf16 is not None:
                grad_w = torch.empty(
                    dY.shape[1],
                    input_bf16.shape[1],
                    dtype=torch.bfloat16,
                    device=dY.device,
                )
                tk_v4_direct_raw_wgrad_gemm(dY, input_bf16, grad_w)
            elif use_small_m_plain_wo and input_bf16 is not None:
                input_pad = _pad_rows_bf16(input_bf16, 256)
                dY_pad = _pad_rows_bf16(dY, 256)
                x_nvfp4_pad = _fast_quantize(
                    input_pad,
                    None,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                dY_nvfp4_pad = _fast_quantize(
                    dY_pad,
                    ctx.grad_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                grad_w = tk_wgrad_gemm(x_nvfp4_pad, dY_nvfp4_pad, use_localcta=False)
            elif input_bf16 is not None:
                grad_w = torch.mm(dY.transpose(0, 1), input_bf16)
            else:
                grad_w = tk_wgrad_gemm(x_nvfp4, dY_nvfp4, use_localcta=use_localcta)
        else:
            # dgrad: dx = dY @ W_o
            if use_small_m_plain_wo:
                dY_pad = _pad_rows_bf16(dY, 256)
                dY_nvfp4_pad = _fast_quantize(
                    dY_pad,
                    ctx.grad_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                dx_full = tk_dgrad_gemm(dY_nvfp4_pad, w_nvfp4, use_localcta=False)
                dx = dx_full[:dY.shape[0]]
            elif (
                use_localcta
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
            ):
                dx = torch.empty(
                    dY.shape[0],
                    w_nvfp4.shape[1],
                    dtype=torch.bfloat16,
                    device=dY.device,
                )
                tk_v4_direct_raw_dgrad_gemm(dY, w_nvfp4._tk_col, dx)
            else:
                from .tk_gemm import use_tk_wo_dgrad_nopdl

                dx = tk_dgrad_gemm(
                    dY_nvfp4,
                    w_nvfp4,
                    use_localcta=use_localcta,
                    use_nopdl=(
                        use_tk_wo_dgrad_nopdl() if not use_localcta else None
                    ),
                )

            # wgrad: dW_o = dY^T @ x
            if (
                use_localcta
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and use_tk_localcta_v4_fast_wo_wgrad()
                and dY.shape[0] % 256 == 0
                and _localcta_v4_wo_outer_sg_pair(
                    dY_nvfp4._tk_col, x_nvfp4._tk_col
                )
            ):
                tk = _get_tk()
                dy_col_fp4, dy_col_sc, dy_col_sg = dY_nvfp4._tk_col
                x_col_fp4, x_col_sc, x_col_sg = x_nvfp4._tk_col
                grad_w = torch.empty(
                    dY.shape[1],
                    x_nvfp4.shape[1],
                    dtype=torch.bfloat16,
                    device=dY.device,
                )
                _localcta_v4_wo_outer_sg_gemm(
                    tk,
                    dy_col_fp4, dy_col_sc, dy_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    grad_w,
                )
                _trace_backend_choice('localcta_wo_wgrad', 'fast_outer_sg')
            elif (
                use_localcta
                and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'
                and use_tk_localcta_v4_cpp_only()
            ):
                grad_w = tk_wgrad_gemm(x_nvfp4, dY_nvfp4, use_localcta=True)
            elif use_localcta and os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4' and input_bf16 is not None:
                grad_w = torch.empty(
                    dY.shape[1],
                    input_bf16.shape[1],
                    dtype=torch.bfloat16,
                    device=dY.device,
                )
                tk_v4_direct_raw_wgrad_gemm(dY, input_bf16, grad_w)
            elif use_small_m_plain_wo and input_bf16 is not None:
                input_pad = _pad_rows_bf16(input_bf16, 256)
                dY_pad = _pad_rows_bf16(dY, 256)
                x_nvfp4_pad = _fast_quantize(
                    input_pad,
                    None,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                dY_nvfp4_pad = _fast_quantize(
                    dY_pad,
                    ctx.grad_quantizer,
                    tk_swizzle=True,
                    use_localcta_override=False,
                )
                grad_w = tk_wgrad_gemm(x_nvfp4_pad, dY_nvfp4_pad, use_localcta=False)
            elif input_bf16 is not None:
                grad_w = torch.mm(dY.transpose(0, 1), input_bf16)
            else:
                grad_w = tk_wgrad_gemm(x_nvfp4, dY_nvfp4, use_localcta=use_localcta)
        if wo_timings is not None:
            _wo_mark("gemms_done")
        _tk_attn_debug_assert_finite(
            "wo_backward_outputs",
            debug_name,
            (("grad_input", dx), ("grad_weight", grad_w)),
        )
        if use_tk_attn_sync_wo_bwd():
            torch.cuda.synchronize(dx.device)
        _tk_stage_trace('wo_bwd', 'end', debug_name)
        _tk_debug_print('wo_bwd', 'end', debug_name)
        if debug_call_id is not None:
            _attn_debug_check_finite(f'wo_bwd[{debug_call_id}].grad_input', dx)
        if debug_call_id is not None:
            _attn_debug_check_finite(f'wo_bwd[{debug_call_id}].grad_w', grad_w)
        wgrad_ref_input = input_bf16
        if wgrad_ref_input is None:
            wgrad_ref_input = split2_input_bf16
        if (
            wgrad_ref_input is None
            and input_nhsd_shape is not None
            and input_nhsd_bf16 is not None
        ):
            B, H, S, D = input_nhsd_shape
            wgrad_ref_input = input_nhsd_bf16.transpose(1, 2).contiguous().view(B * S, H * D)
        if _debug_wgrad_ref_enabled(debug_name) and wgrad_ref_input is not None:
            grad_w_ref = torch.matmul(
                dY.transpose(0, 1).to(torch.bfloat16),
                wgrad_ref_input.to(torch.bfloat16),
            )
            _append_attn_capture({
                "event": "wo_wgrad_ref",
                "debug_call_id": debug_call_id,
                "debug_name": debug_name,
                "stats": {
                    "input": _tensor_capture_stats(wgrad_ref_input),
                    "delta": _tensor_delta_stats(grad_w, grad_w_ref),
                },
            })
        if _debug_dgrad_ref_enabled(debug_name) and wo_weight_bf16 is not None:
            dx_ref = torch.matmul(
                dY.to(torch.bfloat16),
                wo_weight_bf16.to(torch.bfloat16),
            )
            _append_attn_capture({
                "event": "wo_dgrad_ref",
                "debug_call_id": debug_call_id,
                "debug_name": debug_name,
                "stats": _tensor_delta_stats(dx, dx_ref),
            })
        if _should_emit_localcta_function_grad_debug(debug_name):
            _emit_localcta_function_grad_debug(
                "wo_backward_return",
                {
                    "debug_name": debug_name,
                    "grad_output": _tensor_debug_stats(dY),
                    "wgrad_input": _tensor_debug_stats(wgrad_ref_input) if wgrad_ref_input is not None else None,
                    "grad_input": _tensor_debug_stats(dx),
                    "grad_w": _tensor_debug_stats(grad_w),
                    "rescue": wo_rescue,
                },
            )
        if _attn_capture_path():
            _append_attn_capture({
                "event": "wo_backward",
                "debug_call_id": debug_call_id,
                "debug_name": debug_name,
                "stats": {
                    "grad_output": _tensor_capture_stats(dY),
                    "wgrad_input": _tensor_capture_stats(wgrad_ref_input),
                    "grad_input": _tensor_capture_stats(dx),
                    "grad_w": _tensor_capture_stats(grad_w),
                },
            })

        input_grad = dx
        if input_nhsd_shape is not None:
            B, H, S, D = input_nhsd_shape
            input_grad = dx.view(B, S, H, D).transpose(1, 2)
            if not use_tk_localcta_v4_wo_attn_layout_strided_dx():
                input_grad = input_grad.contiguous()
        if _attn_layout_event_enabled("wo_backward_return", debug_name):
            _append_attn_layout({
                "event": "wo_backward_return",
                "debug_name": debug_name,
                **_tensor_layout_group(
                    (("dx", dx), ("input_grad", input_grad), ("grad_w", grad_w))
                ),
            })
        if wo_timings is not None:
            _wo_mark("reshape_done")
        _emit_wo_debug_timings_once(
            debug_name, "backward", dY.shape[0], w_nvfp4.shape[1], dY.shape[1], wo_timings
        )

        # Free cached tensors
        ctx.w_nvfp4 = None
        ctx.x_nvfp4 = None
        ctx.input_bf16 = None
        ctx.localcta_input_bf16 = None
        ctx.wo_weight_bf16 = None
        ctx._input_nhsd_shape = None

        grad_residual = dY if ctx.has_residual else None
        if debug_name is not None:
            return (
                _maybe_clone_autograd_return(input_grad, dY.shape[0]), # input
                _maybe_clone_autograd_return(grad_w, dY.shape[0]),     # wo_weight
                None,       # input_quantizer
                None,       # weight_quantizer
                None,       # grad_quantizer
                None,       # workspace
                None,       # debug_name
                grad_residual,
                None,       # h_gamma
                None,       # cde_emit
            )
        return (
            _maybe_clone_autograd_return(input_grad, dY.shape[0]), # input
            _maybe_clone_autograd_return(grad_w, dY.shape[0]),     # wo_weight
            None,       # input_quantizer
            None,       # weight_quantizer
            None,       # grad_quantizer
            None,       # workspace
            None,       # debug_name
            grad_residual,
            None,       # h_gamma
            None,       # cde_emit
        )


# ---------------------------------------------------------------------------
# FusedAttentionFP4: replaces Attention + attention_norm
# ---------------------------------------------------------------------------
class FusedAttentionFP4_TE(nn.Module):
    """Fused FP4 Attention with absorbed RMSNorm and grouped QKV quantization.

    Replaces:
      attention_norm(x) → wq(x), wk(x), wv(x) (3 separate quant + 3 GEMMs)
    With:
      fused_rmsnorm(x) + quantize(normed, ONCE) →
      group_quantize(w_qkv, per-split amax, 1 kernel) →
      single GEMM → split

    Savings per layer: 2 input FP4 quants, 2 weight FP4 quants, 2 GEMMs eliminated.
    Per-split amax preserves FP4 resolution for convergence.
    The output projection (wo) remains separate.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.epsilon = norm_eps

        self.q_dim = n_heads * head_dim
        self.k_dim = n_kv_heads * head_dim
        self.v_dim = n_kv_heads * head_dim
        self.total_out = self.q_dim + self.k_dim + self.v_dim

        # Absorbed norm weight
        self.norm_weight = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype)
        )

        # Stacked QKV weight: rows = [wq; wk; wv]
        self.w_qkv = nn.Parameter(
            torch.empty(self.total_out, dim, device=device, dtype=dtype)
        )

        # Output projection
        self.wo_weight = nn.Parameter(
            torch.empty(dim, self.q_dim, device=device, dtype=dtype)
        )

        # TE quantizers
        te_dtype = tex.DType.kFloat4E2M1

        def _make_quantizer(role: str):
            return _make_nvfp4_quantizer_for_role(role, te_dtype)

        # QKV quantizers
        self.qkv_input_quantizer = _make_quantizer("activation")
        self.qkv_weight_quantizer = _make_quantizer("weight")
        self.qkv_grad_quantizer = _make_quantizer("grad")

        # wo quantizers
        self.wo_input_quantizer = _make_quantizer("activation")
        self.wo_weight_quantizer = _make_quantizer("weight")
        self.wo_grad_quantizer = _make_quantizer("grad")

        # Workspace
        self._workspace = None
        self._workspace_device = None

    def _ensure_workspace(self, device):
        if self._workspace_device != device:
            self._workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._workspace_device = device

    def forward_qkv(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        h_carrier=None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
    ):
        """Forward pass for QKV projection only."""
        del freqs_cis
        if h_carrier is not None:
            raise RuntimeError("TE QKV does not support an H tile carrier")
        if cde_row_rms_partial is not None:
            raise RuntimeError("TE QKV does not support an exact C/D/E carrier")
        is_3d = x.dim() == 3
        if is_3d:
            B, S, D = x.shape
            x_2d = x.reshape(B * S, D)
        else:
            x_2d = x

        self._ensure_workspace(x.device)

        xq, xk, xv = _FusedQKVFunction_TE.apply(
            x_2d, self.w_qkv, self.norm_weight, self.epsilon,
            self.q_dim, self.k_dim, self.v_dim,
            self.qkv_input_quantizer,
            self.qkv_weight_quantizer,
            self.qkv_grad_quantizer,
            self._workspace,
        )
        self._last_qkv_rope_applied = False

        if is_3d:
            xq = xq.view(B, S, -1)
            xk = xk.view(B, S, -1)
            xv = xv.view(B, S, -1)

        return xq, xk, xv

    def forward_wo(
        self,
        attn_output: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        h_gamma: Optional[torch.Tensor] = None,
        cde_emit: bool = False,
    ):
        """Output projection: wo @ attn_output."""
        if h_gamma is not None:
            raise RuntimeError("TE Wo does not support an H tile carrier")
        if cde_emit:
            raise RuntimeError("TE Wo does not support an exact C/D/E carrier")
        is_nhsd = attn_output.dim() == 4
        is_3d = attn_output.dim() == 3
        if is_nhsd:
            B, H, S, D = attn_output.shape
            out_2d = attn_output.transpose(1, 2).contiguous().view(B * S, H * D)
        elif is_3d:
            B, S, D = attn_output.shape
            out_2d = attn_output.reshape(B * S, D)
        else:
            out_2d = attn_output

        self._ensure_workspace(attn_output.device)

        y = _WoFunction_TE.apply(
            out_2d, self.wo_weight,
            self.wo_input_quantizer, self.wo_weight_quantizer,
            self.wo_grad_quantizer, self._workspace,
        )

        if residual is not None:
            y = y + residual.reshape_as(y)

        if is_nhsd or is_3d:
            return y.view(B, S, self.dim)
        return y

    def init_weights(self, init_std: float = 0.02):
        """Initialize weights, matching Llama's Attention.init_weights()."""
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self.w_qkv, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.wo_weight, mean=0.0, std=init_std)

    @classmethod
    def from_attention(cls, attention, norm, model_args=None):
        """Create from existing Attention + attention_norm modules."""
        q_proj = getattr(attention, "wq", getattr(attention, "q_proj", None))
        k_proj = getattr(attention, "wk", getattr(attention, "k_proj", None))
        v_proj = getattr(attention, "wv", getattr(attention, "v_proj", None))
        o_proj = getattr(attention, "wo", getattr(attention, "o_proj", None))
        if not all(isinstance(p, nn.Linear) for p in (q_proj, k_proj, v_proj, o_proj)):
            raise AttributeError("FusedAttentionFP4_TE expects wq/wk/wv/wo or q_proj/k_proj/v_proj/o_proj linears")
        n_heads = getattr(attention, "n_heads", None)
        if n_heads is None:
            n_heads = attention.num_heads
        n_kv_heads = getattr(attention, 'n_kv_heads', None)
        if n_kv_heads is None:
            n_kv_heads = getattr(attention, "num_key_value_heads", n_heads)
        head_dim = attention.head_dim

        dim = q_proj.in_features
        device = q_proj.weight.device
        dtype = q_proj.weight.dtype
        eps = getattr(norm, 'eps', getattr(norm, "variance_epsilon", 1e-5))

        fused = cls(
            dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
            head_dim=head_dim, norm_eps=eps,
            device=device, dtype=dtype,
        )

        if device.type != 'meta':
            with torch.no_grad():
                # Stack QKV weights
                fused.w_qkv.copy_(torch.cat([
                    q_proj.weight,
                    k_proj.weight,
                    v_proj.weight,
                ], dim=0))
                fused.wo_weight.copy_(o_proj.weight)

                if hasattr(norm, 'weight') and norm.weight is not None:
                    fused.norm_weight.copy_(norm.weight)

        return fused

# ---------------------------------------------------------------------------
# Thin nn.Module wrappers for CUDA graph capture via make_graphed_callables
# ---------------------------------------------------------------------------
class _QKVGraphHelper(nn.Module):
    """Wraps _FusedQKVFunction_TK.apply for make_graphed_callables."""
    def __init__(self, parent: 'FusedAttentionFP4_TK'):
        super().__init__()
        self._parent = parent

    def forward(self, x_2d):
        p = self._parent
        debug_name = f"{getattr(p, '_lbt_debug_name', p.__class__.__name__)}:qkv"
        empty_fp4 = torch.empty(
            0, dtype=torch.float4_e2m1fn_x2, device=x_2d.device
        )
        empty_sc = torch.empty(
            0, dtype=torch.float8_e4m3fn, device=x_2d.device
        )
        empty_sg = torch.empty(0, dtype=torch.float32, device=x_2d.device)
        empty_r = torch.empty(0, dtype=torch.float32, device=x_2d.device)
        return _FusedQKVFunction_TK.apply(
            x_2d, p.w_qkv, p.norm_weight, p.epsilon,
            p.q_dim, p.k_dim, p.v_dim,
            None, 0, 0, p.head_dim,
            p.qkv_input_quantizer,
            p.qkv_weight_quantizer,
            p.qkv_grad_quantizer,
            p._qkv_workspace,
            debug_name,
            empty_fp4, empty_sc, empty_sg,
            empty_fp4, empty_sc, empty_sg, empty_r, empty_r,
        )

class _WoGraphHelper(nn.Module):
    """Wraps _WoFunction_TK.apply for make_graphed_callables."""
    def __init__(self, parent: 'FusedAttentionFP4_TK'):
        super().__init__()
        self._parent = parent

    def forward(self, out_2d):
        p = self._parent
        debug_name = f"{getattr(p, '_lbt_debug_name', p.__class__.__name__)}:wo"
        empty = torch.empty(0, dtype=torch.bfloat16, device=out_2d.device)
        return _WoFunction_TK.apply(
            out_2d, p.wo_weight,
            p.wo_input_quantizer, p.wo_weight_quantizer,
            p.wo_grad_quantizer, p._wo_workspace,
            debug_name, empty, empty, False,
        )

class FusedAttentionFP4_TK(nn.Module):
    """Fused FP4 Attention with absorbed RMSNorm and grouped QKV quantization.

    Replaces:
      attention_norm(x) → wq(x), wk(x), wv(x) (3 separate quant + 3 GEMMs)
    With:
      fused_rmsnorm(x) + quantize(normed, ONCE) →
      group_quantize(w_qkv, per-split amax, 1 kernel) →
      single GEMM → split

    Savings per layer: 2 input FP4 quants, 2 weight FP4 quants, 2 GEMMs eliminated.
    Per-split amax preserves FP4 resolution for convergence.
    The output projection (wo) remains separate.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.epsilon = norm_eps

        self.q_dim = n_heads * head_dim
        self.k_dim = n_kv_heads * head_dim
        self.v_dim = n_kv_heads * head_dim
        self.total_out = self.q_dim + self.k_dim + self.v_dim

        # Absorbed norm weight
        self.norm_weight = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype)
        )

        # Stacked QKV weight: rows = [wq; wk; wv]
        self.w_qkv = nn.Parameter(
            torch.empty(self.total_out, dim, device=device, dtype=dtype)
        )

        # Output projection
        self.wo_weight = nn.Parameter(
            torch.empty(dim, self.q_dim, device=device, dtype=dtype)
        )

        # TE quantizers
        te_dtype = None if (use_tk_quant() or use_tk_localcta()) else tex.DType.kFloat4E2M1

        def _make_quantizer(role: str):
            return _make_nvfp4_quantizer_for_role(role, te_dtype)

        # QKV quantizers
        self.qkv_input_quantizer = _make_quantizer("activation")
        self.qkv_weight_quantizer = _make_quantizer("weight")
        self.qkv_grad_quantizer = _make_quantizer("grad")

        # wo quantizers
        self.wo_input_quantizer = _make_quantizer("activation")
        self.wo_weight_quantizer = _make_quantizer("weight")
        self.wo_grad_quantizer = _make_quantizer("grad")

        # QKV and WO can still have kernels in flight at the same time. Give
        # them disjoint scratch so they never alias the same workspace buffer.
        self._qkv_workspace = None
        self._wo_workspace = None
        self._workspace_device = None

        # CUDA graph state (lazy-initialized)
        self._qkv_graphed = None  # graphed callable for QKV
        self._wo_graphed = None   # graphed callable for Wo
        self._qkv_cached_M = None
        self._wo_cached_M = None

    def _forward_wo_bf16(self, attn_output: torch.Tensor):
        is_nhsd = attn_output.dim() == 4
        is_3d = attn_output.dim() == 3
        if is_nhsd:
            B, H, S, D = attn_output.shape
            out_2d = attn_output.transpose(1, 2).contiguous().view(B * S, H * D)
        elif is_3d:
            B, S, D = attn_output.shape
            out_2d = attn_output.reshape(B * S, D)
        else:
            out_2d = attn_output

        y = F.linear(out_2d.to(self.wo_weight.dtype), self.wo_weight)
        if is_nhsd:
            return y.view(B, S, self.dim)
        if is_3d:
            return y.view(B, S, self.dim)
        return y

    def _ensure_workspace(self, device):
        if self._workspace_device != device:
            self._qkv_workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._wo_workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._workspace_device = device

    def forward_qkv(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        h_carrier=None,
        cde_row_rms_partial: Optional[torch.Tensor] = None,
    ):
        """Forward pass for QKV projection only."""
        if h_carrier is not None and cde_row_rms_partial is not None:
            raise RuntimeError("exact C/D/E and H tile QKV carriers are mutually exclusive")
        if h_carrier is not None:
            (
                x, h_row_fp4, h_row_sc, h_row_sg,
                h_col_fp4, h_col_sc, h_col_sg, h_r_tile,
            ) = h_carrier
        else:
            h_row_fp4 = torch.empty(
                0, dtype=torch.float4_e2m1fn_x2, device=x.device
            )
            h_row_sc = torch.empty(
                0, dtype=torch.float8_e4m3fn, device=x.device
            )
            h_row_sg = torch.empty(0, dtype=torch.float32, device=x.device)
            h_col_fp4 = h_row_fp4
            h_col_sc = h_row_sc
            h_col_sg = h_row_sg
            h_r_tile = torch.empty(0, dtype=torch.float32, device=x.device)
        if cde_row_rms_partial is None:
            cde_row_rms_partial = torch.empty(
                0, dtype=torch.float32, device=x.device
            )
        is_3d = x.dim() == 3
        rope_freqs = None
        rope_batch_size = 0
        rope_seq_len = 0
        if is_3d:
            B, S, D = x.shape
            rope_supported = (
                freqs_cis is not None
                and _tk_qkv_rope_supported(
                    B * S, D, self.q_dim, self.k_dim, self.v_dim, self.head_dim, S, freqs_cis
                )
            )
            rope_split_path = use_tk_localcta() or not use_tk_qkv_forward_cat_debug()
            rope_packed_supported = _tk_qkv_rope_packed_supported(
                B * S, D, self.q_dim, self.k_dim, self.v_dim,
                self.head_dim, S, freqs_cis
            )
            rope_packed_requested = (
                not use_tk_localcta()
                and (
                    rope_packed_supported
                    or os.environ.get("USE_TK_QKV_PACKED_ROPE_EPILOGUE") == "1"
                )
            )
            rope_packed_backend = _tk_qkv_rope_packed_backend_available()
            if rope_packed_requested and not rope_packed_supported:
                raise RuntimeError("packed QKV RoPE requested for an unsupported contract")
            if rope_packed_requested and not rope_packed_backend:
                raise RuntimeError("packed QKV RoPE requested but native split symbol is unavailable")
            rope_packed_split_path = (
                not rope_split_path
                and rope_packed_supported
                and rope_packed_backend
            )
            rope_dispatch_path = rope_split_path or rope_packed_split_path
            rope_backend = (
                _tk_qkv_rope_backend_available()
                if rope_supported and rope_dispatch_path
                else False
            )
            if rope_supported and rope_backend:
                from .tk_gemm import use_tk_localcta_direct_contract
                if not use_tk_localcta_direct_contract():
                    rope_freqs = freqs_cis
                    rope_batch_size = B
                    rope_seq_len = S
                else:
                    _trace_backend_choice('localcta_qkv_fwd_rope_skip', 'direct_contract')
            elif freqs_cis is not None:
                reason = 'unsupported'
                if not use_tk_qkv_rope_epilogue():
                    reason = 'disabled'
                elif not freqs_cis.is_cuda:
                    reason = 'freqs_not_cuda'
                elif not torch.is_complex(freqs_cis):
                    reason = f'freqs_dtype_{freqs_cis.dtype}'
                elif not rope_dispatch_path:
                    reason = 'cat_debug_grouped_gemm'
                elif not rope_backend:
                    reason = 'backend_unavailable'
                _trace_backend_choice('localcta_qkv_fwd_rope_skip', reason)
            x_2d = x.reshape(B * S, D)
        elif freqs_cis is not None:
            _trace_backend_choice('localcta_qkv_fwd_rope_skip', f'input_dim_{x.dim()}')
        else:
            x_2d = x

        self._ensure_workspace(x.device)
        debug_name = f"{getattr(self, '_lbt_debug_name', self.__class__.__name__)}:qkv"
        _tk_stage_trace('qkv_fwd', 'start', debug_name)
        _tk_debug_print('qkv_fwd', 'start', debug_name)

        # Note: USE_CUDA_GRAPH=1 is handled in the backward via _qkv_bwd_graphed().
        # We do NOT use make_graphed_callables here because it's incompatible with
        # custom autograd Function.apply() (non-tensor args cause allow_unused errors).
        xq, xk, xv = _FusedQKVFunction_TK.apply(
            x_2d, self.w_qkv, self.norm_weight, self.epsilon,
            self.q_dim, self.k_dim, self.v_dim,
            rope_freqs, rope_batch_size, rope_seq_len, self.head_dim,
            self.qkv_input_quantizer,
            self.qkv_weight_quantizer,
            self.qkv_grad_quantizer,
            self._qkv_workspace,
            debug_name,
            h_row_fp4, h_row_sc, h_row_sg,
            h_col_fp4, h_col_sc, h_col_sg, h_r_tile,
            cde_row_rms_partial,
        )
        self._last_qkv_rope_applied = rope_freqs is not None
        _tk_stage_trace('qkv_fwd', 'end', debug_name)
        _tk_debug_print('qkv_fwd', 'end', debug_name)

        if is_3d:
            xq = xq.view(B, S, -1)
            xk = xk.view(B, S, -1)
            xv = xv.view(B, S, -1)

        return xq, xk, xv

    def forward_wo(
        self,
        attn_output: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        h_gamma: Optional[torch.Tensor] = None,
        cde_emit: bool = False,
    ):
        """Output projection: wo @ attn_output."""
        if getattr(self, "_force_wo_bf16", False):
            if h_gamma is not None or cde_emit:
                raise RuntimeError("Wo carriers are unavailable on the BF16 fallback")
            return self._forward_wo_bf16(attn_output)

        is_nhsd = attn_output.dim() == 4
        is_3d = attn_output.dim() == 3
        if is_nhsd:
            B, H, S, D = attn_output.shape
            out_2d = attn_output
        elif is_3d:
            B, S, D = attn_output.shape
            out_2d = attn_output.reshape(B * S, D)
        else:
            out_2d = attn_output

        residual_2d = None
        if residual is not None:
            residual_2d = residual.reshape(-1, self.dim)
        gamma_arg = h_gamma
        if gamma_arg is None:
            gamma_arg = torch.empty(
                0, dtype=torch.bfloat16, device=attn_output.device
            )

        self._ensure_workspace(attn_output.device)
        debug_name = f"{getattr(self, '_lbt_debug_name', self.__class__.__name__)}:wo"
        _tk_stage_trace('wo_fwd', 'start', debug_name)
        _tk_debug_print('wo_fwd', 'start', debug_name)

        if (
            use_cuda_graph()
            and not use_tk_attn_debug_finite()
            and not is_nhsd
            and residual_2d is None
            and h_gamma is None
        ):
            M = out_2d.shape[0]
            if self._wo_graphed is None or self._wo_cached_M != M:
                self._wo_cached_M = M
                sample_input = torch.randn_like(out_2d)
                helper = _WoGraphHelper(self)
                from .localcta_sr_state import (
                    preserve_localcta_sr_state_during_cuda_graph_capture,
                    wo_grad_key,
                )

                # make_graphed_callables executes synthetic backward warmups
                # and a capture.  Roll back only those reservations; the
                # captured prep kernel keeps the live state pointer and the
                # first real backward replay advances it exactly once.
                with preserve_localcta_sr_state_during_cuda_graph_capture(
                    (wo_grad_key(debug_name),)
                ):
                    self._wo_graphed = torch.cuda.make_graphed_callables(
                        helper, (sample_input,), num_warmup_iters=3,
                    )
            y = self._wo_graphed(out_2d)
        else:
            y = _WoFunction_TK.apply(
                out_2d, self.wo_weight,
                self.wo_input_quantizer, self.wo_weight_quantizer,
                self.wo_grad_quantizer, self._wo_workspace,
                debug_name,
                residual_2d,
                gamma_arg,
                cde_emit,
            )
        _tk_stage_trace('wo_fwd', 'end', debug_name)
        _tk_debug_print('wo_fwd', 'end', debug_name)

        if h_gamma is not None:
            z = y[0]
            if is_nhsd or is_3d:
                z = z.view(B, S, self.dim)
            return (z, *y[1:])
        if cde_emit:
            z, row_rms_partial = y
            if is_nhsd or is_3d:
                z = z.view(B, S, self.dim)
            return z, row_rms_partial
        if is_nhsd:
            return y.view(B, S, self.dim)
        if is_3d:
            return y.view(B, S, self.dim)
        return y

    def init_weights(self, init_std: float = 0.02):
        """Initialize weights, matching Llama's Attention.init_weights()."""
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self.w_qkv, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.wo_weight, mean=0.0, std=init_std)

    @classmethod
    def from_attention(cls, attention, norm, model_args=None):
        """Create from existing Attention + attention_norm modules."""
        q_proj = getattr(attention, "wq", getattr(attention, "q_proj", None))
        k_proj = getattr(attention, "wk", getattr(attention, "k_proj", None))
        v_proj = getattr(attention, "wv", getattr(attention, "v_proj", None))
        o_proj = getattr(attention, "wo", getattr(attention, "o_proj", None))
        if not all(isinstance(p, nn.Linear) for p in (q_proj, k_proj, v_proj, o_proj)):
            raise AttributeError("FusedAttentionFP4_TK expects wq/wk/wv/wo or q_proj/k_proj/v_proj/o_proj linears")
        n_heads = getattr(attention, "n_heads", None)
        if n_heads is None:
            n_heads = attention.num_heads
        n_kv_heads = getattr(attention, 'n_kv_heads', None)
        if n_kv_heads is None:
            n_kv_heads = getattr(attention, "num_key_value_heads", n_heads)
        head_dim = attention.head_dim

        dim = q_proj.in_features
        device = q_proj.weight.device
        dtype = q_proj.weight.dtype
        eps = getattr(norm, 'eps', getattr(norm, "variance_epsilon", 1e-5))

        fused = cls(
            dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
            head_dim=head_dim, norm_eps=eps,
            device=device, dtype=dtype,
        )

        if device.type != 'meta':
            with torch.no_grad():
                # Stack QKV weights
                fused.w_qkv.copy_(torch.cat([
                    q_proj.weight,
                    k_proj.weight,
                    v_proj.weight,
                ], dim=0))
                fused.wo_weight.copy_(o_proj.weight)

                if hasattr(norm, 'weight') and norm.weight is not None:
                    fused.norm_weight.copy_(norm.weight)

        return fused


# ---------------------------------------------------------------------------
# Dispatch aliases: select TE or TK variant based on USE_TK_GEMM env var
# ---------------------------------------------------------------------------
def FusedAttentionFP4(*args, **kwargs):
    """Factory: returns FusedAttentionFP4_TK when USE_TK_GEMM=1, else _TE."""
    from .tk_gemm import use_tk_gemm
    cls = FusedAttentionFP4_TK if use_tk_gemm() else FusedAttentionFP4_TE
    return cls(*args, **kwargs)


def FusedFeedForwardFP4(*args, **kwargs):
    """Factory: returns FusedFeedForwardFP4_TK when USE_TK_GEMM=1, else _TE."""
    from .tk_gemm import use_tk_gemm
    cls = FusedFeedForwardFP4_TK if use_tk_gemm() else FusedFeedForwardFP4_TE
    return cls(*args, **kwargs)
