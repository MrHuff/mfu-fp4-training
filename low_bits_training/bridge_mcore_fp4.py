"""Megatron-Core adapters for the local FP4 fused kernels.

The target Bridge path is a layer swap, not a separate model stack: keep
Megatron-Core's distributed attention/CP machinery, but replace the projection
and MLP modules with the same fused FP4 kernels used by the TorchTitan/LBT path.
"""

from __future__ import annotations

import os
import sys
import copy
import types
import weakref
from typing import Any

import torch
import torch.nn as nn

_BRIDGE_FP4_WORKSPACES: dict[tuple[str, int | None, int], torch.Tensor] = {}
_BRIDGE_WRAPPER_TIMING_COUNTS: dict[tuple[str, int | None, int], int] = {}


def _current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _bridge_debug_enabled() -> bool:
    return os.environ.get("LBT_BRIDGE_FP4_CCE_DEBUG", "0") == "1"


def _bridge_sync_if_enabled(name: str) -> None:
    if os.environ.get(name, "0") == "1" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _bridge_trace_if_enabled(name: str, message: str) -> None:
    if os.environ.get(name, "0") != "1":
        return
    rank = os.environ.get("RANK", "na")
    print(f"[LBT_BRIDGE_TRACE] rank={rank} {message}", file=sys.stderr, flush=True)


def _bridge_rank() -> str:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return str(torch.distributed.get_rank())
    except Exception:
        pass
    return os.environ.get("RANK", "na")


def _bridge_wrapper_timing_begin(label: str, layer_number: int | None):
    if os.environ.get("LBT_BRIDGE_WRAPPER_TIMING", "0") != "1" or not torch.cuda.is_available():
        return None
    rank = _bridge_rank()
    rank_filter = os.environ.get("LBT_BRIDGE_WRAPPER_TIMING_RANK", "0").strip()
    if rank_filter not in ("", "*", "all") and rank != rank_filter:
        return None
    step_filter = os.environ.get("LBT_BRIDGE_WRAPPER_TIMING_STEP", "").strip()
    if step_filter:
        active_step = os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip()
        if active_step != step_filter:
            return None
    layer_filter = os.environ.get("LBT_BRIDGE_WRAPPER_TIMING_LAYER", "0").strip()
    if layer_filter not in ("", "*", "all"):
        if layer_number is None or str(layer_number) != layer_filter:
            return None
    label_filter = os.environ.get("LBT_BRIDGE_WRAPPER_TIMING_FILTER", "").strip()
    if label_filter and label_filter not in label:
        return None

    try:
        limit = int(os.environ.get("LBT_BRIDGE_WRAPPER_TIMING_LIMIT", "4"))
    except ValueError:
        limit = 4
    key = (label, layer_number, torch.cuda.current_device())
    count = _BRIDGE_WRAPPER_TIMING_COUNTS.get(key, 0)
    if count >= limit:
        return None
    _BRIDGE_WRAPPER_TIMING_COUNTS[key] = count + 1

    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    return label, layer_number, rank, count + 1, stream, start, end


def _bridge_wrapper_timing_end(token) -> None:
    if token is None:
        return
    label, layer_number, rank, count, stream, start, end = token
    end.record(stream)
    end.synchronize()
    step = os.environ.get("LBT_TRACE_ACTIVE_STEP", "na")
    print(
        "[LBT_BRIDGE_WRAP_TIMING] "
        f"rank={rank} step={step} layer={layer_number} "
        f"sample={count} label={label} ms={start.elapsed_time(end):.3f}",
        file=sys.stderr,
        flush=True,
    )


def _print_bridge_grad_stats(label: str, grad: torch.Tensor) -> None:
    if not _bridge_debug_enabled():
        return
    finite = bool(torch.isfinite(grad).all().item())
    limit = int(os.environ.get("LBT_BRIDGE_FP4_QKV_DEBUG_LIMIT", "24"))
    count = getattr(_print_bridge_grad_stats, "_count", 0)
    if finite and count >= limit:
        return
    setattr(_print_bridge_grad_stats, "_count", count + 1)
    safe = torch.nan_to_num(
        grad.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    nbad = int((~torch.isfinite(grad)).sum().item()) if not finite else 0
    print(
        "[LBT_BRIDGE_QKV] "
        f"rank={int(os.environ.get('RANK', '0'))} {label} "
        f"shape={tuple(grad.shape)} stride={tuple(grad.stride())} "
        f"dtype={grad.dtype} finite={finite} nbad={nbad} "
        f"max_abs={float(safe.abs().max().item()):.8e}",
        file=sys.stderr,
        flush=True,
    )


class _NVFP4QuantizerBundle:
    def __init__(self) -> None:
        from low_bits_training.quantization.fused_te_linear import (
            _make_nvfp4_quantizer_for_role,
        )

        import transformer_engine_torch as tex

        te_dtype = tex.DType.kFloat4E2M1
        self.activation = _make_nvfp4_quantizer_for_role("activation", te_dtype)
        self.weight = _make_nvfp4_quantizer_for_role("weight", te_dtype)
        self.grad = _make_nvfp4_quantizer_for_role("grad", te_dtype)


def _require_tp1(config: Any, module_name: str) -> None:
    tp_size = int(getattr(config, "tensor_model_parallel_size", 1))
    if tp_size != 1:
        raise NotImplementedError(
            f"{module_name} currently supports TP=1 only. "
            "The local fused parameter layout is not TP-sharded yet."
        )


def _tp_info(tp_group: Any | None = None) -> tuple[Any | None, int, int]:
    try:
        from megatron.core.utils import get_pg_rank, get_pg_size, get_tensor_model_parallel_group_if_none

        group = get_tensor_model_parallel_group_if_none(tp_group)
        return group, int(get_pg_size(group)), int(get_pg_rank(group))
    except Exception:
        return tp_group, 1, 0


def _divide_exact(numerator: int, denominator: int, label: str) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{label}={numerator} is not divisible by TP size {denominator}")
    return numerator // denominator


def _normalize_bridge_fp4_backend(backend: str) -> str:
    value = backend.strip().lower().replace("-", "_")
    aliases = {
        "localcta": "nvfp4_localcta_v4",
        "localcta_v4": "nvfp4_localcta_v4",
        "nvfp4_localcta": "nvfp4_localcta_v4",
        "nvfp4_localcta_v4": "nvfp4_localcta_v4",
        "nvfp4_tk": "nvfp4_tk_v5",
        "tk": "nvfp4_tk_v5",
        "tk_v5": "nvfp4_tk_v5",
        "nvfp4_tk_v5": "nvfp4_tk_v5",
        "mx": "mxfp4",
        "mxfp4": "mxfp4",
        "mixed": "mixed_localcta_mxfp4",
        "mixed_localcta_mxfp4": "mixed_localcta_mxfp4",
        "localcta_mxfp4": "mixed_localcta_mxfp4",
        "nvfp4_localcta_v4_mxfp4": "mixed_localcta_mxfp4",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported Bridge FP4 backend: {backend!r}")
    return aliases[value]


def _is_bridge_layerwise_mixed_backend(backend: str) -> bool:
    return _normalize_bridge_fp4_backend(backend) == "mixed_localcta_mxfp4"


def _normalize_bridge_localcta_v4_tp2_profile(profile: str | None) -> str:
    value = (profile or "").strip().lower().replace("-", "_")
    aliases = {
        "": "off",
        "0": "off",
        "false": "off",
        "none": "off",
        "off": "off",
        "default": "off",
        "prepared_split2": "tp2_fused_split",
        "tp2_prepared_split2": "tp2_fused_split",
        "tp2_fused_split": "tp2_fused_split",
        "overlap": "tp2_overlap",
        "tp2_overlap": "tp2_overlap",
        "prepared_split2_overlap": "tp2_fused_split_overlap",
        "tp2_prepared_split2_overlap": "tp2_fused_split_overlap",
        "tp2_fused_split_overlap": "tp2_fused_split_overlap",
        "highwater": "tp2_fused_split_overlap",
        "hw": "tp2_fused_split_overlap",
        "mxfp4_highwater": "tp2_fused_split_overlap",
    }
    if value not in aliases:
        raise ValueError(
            "LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE must be one of off, "
            "tp2_fused_split, tp2_overlap, tp2_fused_split_overlap, or highwater; "
            f"got {profile!r}"
        )
    return aliases[value]


def _parse_bridge_layer_ranges(raw: str, layer_number: int) -> str | None:
    """Parse BACKEND:1-8,17;BACKEND:9-16 style layer routing."""

    if not raw.strip():
        return None
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(
                "LBT_BRIDGE_FP4_MIXED_LAYERS entries must be BACKEND:RANGES, "
                f"got {spec!r}"
            )
        backend, ranges = spec.split(":", 1)
        backend = _normalize_bridge_fp4_backend(backend)
        if backend == "mixed_localcta_mxfp4":
            raise ValueError("LBT_BRIDGE_FP4_MIXED_LAYERS cannot route a layer to mixed")
        for item in ranges.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                left, right = item.split("-", 1)
                start = int(left)
                end = int(right)
            else:
                start = end = int(item)
            if start <= layer_number <= end:
                return backend
    return None


def _bridge_layerwise_mixed_backend(layer_number: int, total_layers: int | None) -> str:
    explicit = os.environ.get("LBT_BRIDGE_FP4_MIXED_LAYERS", "")
    selected = _parse_bridge_layer_ranges(explicit, layer_number)
    if selected is not None:
        return selected

    default_backend = _normalize_bridge_fp4_backend(
        os.environ.get("LBT_BRIDGE_FP4_MIXED_DEFAULT_BACKEND", "nvfp4_localcta_v4")
    )
    if default_backend == "mixed_localcta_mxfp4":
        default_backend = "nvfp4_localcta_v4"
    other_backend = "mxfp4" if default_backend != "mxfp4" else "nvfp4_localcta_v4"
    policy = os.environ.get("LBT_BRIDGE_FP4_MIXED_POLICY", "front_localcta").strip().lower()
    if total_layers is None or total_layers <= 0:
        total_layers = layer_number
    split = int(
        os.environ.get(
            "LBT_BRIDGE_FP4_MIXED_SPLIT_LAYER",
            str(max(1, (int(total_layers) + 1) // 2)),
        )
    )
    if policy in {"front_localcta", "localcta_front", "front_nvfp4"}:
        return "nvfp4_localcta_v4" if layer_number <= split else "mxfp4"
    if policy in {"front_mxfp4", "mxfp4_front"}:
        return "mxfp4" if layer_number <= split else "nvfp4_localcta_v4"
    if policy in {"alternate", "alternate_localcta_odd"}:
        return "nvfp4_localcta_v4" if layer_number % 2 == 1 else "mxfp4"
    if policy == "alternate_mxfp4_odd":
        return "mxfp4" if layer_number % 2 == 1 else "nvfp4_localcta_v4"
    if policy in {"tail_mxfp4", "final_mxfp4", "last_mxfp4"}:
        tail_count = max(
            0,
            int(
                os.environ.get(
                    "LBT_BRIDGE_FP4_MIXED_TAIL_LAYERS",
                    os.environ.get("LBT_FP4_MIXED_TAIL_LAYERS", "4"),
                )
                or 0
            ),
        )
        return "mxfp4" if tail_count and layer_number > total_layers - tail_count else "nvfp4_localcta_v4"
    if policy in {"tail_localcta", "final_localcta", "last_localcta"}:
        tail_count = max(
            0,
            int(
                os.environ.get(
                    "LBT_BRIDGE_FP4_MIXED_TAIL_LAYERS",
                    os.environ.get("LBT_FP4_MIXED_TAIL_LAYERS", "4"),
                )
                or 0
            ),
        )
        return "nvfp4_localcta_v4" if tail_count and layer_number > total_layers - tail_count else "mxfp4"
    if policy in {"default_front", "front_default"}:
        return default_backend if layer_number <= split else other_backend
    raise ValueError(
        "LBT_BRIDGE_FP4_MIXED_POLICY must be one of front_localcta, "
        "front_mxfp4, tail_mxfp4, tail_localcta, alternate, "
        "alternate_mxfp4_odd, or default_front; "
        f"got {policy!r}"
    )


def _mark_tp_param(param: torch.nn.Parameter, *, dim: int, stride: int = 1) -> None:
    try:
        from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes

        set_tensor_model_parallel_attributes(param, True, dim, stride)
    except Exception:
        setattr(param, "tensor_model_parallel", True)
        setattr(param, "partition_dim", dim)
        setattr(param, "partition_stride", stride)


def _bridge_fp4_workspace(device: torch.device) -> torch.Tensor | None:
    if os.environ.get("LBT_BRIDGE_FP4_SHARED_WORKSPACE", "1") != "1":
        return None
    size_mb = int(os.environ.get("LBT_BRIDGE_FP4_WORKSPACE_MB", "32"))
    key = (device.type, device.index, size_mb)
    workspace = _BRIDGE_FP4_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8, device=device)
        _BRIDGE_FP4_WORKSPACES[key] = workspace
    return workspace


def _slice_sequence_parallel_batch_first(tensor: torch.Tensor, local_seq_len: int) -> torch.Tensor:
    """Slice a [batch, seq] tensor to the local sequence-parallel shard."""

    if tensor.dim() != 2:
        raise ValueError(f"expected [batch, seq] tensor for SP slice, got {tuple(tensor.shape)}")
    global_seq_len = int(tensor.shape[1])
    if global_seq_len == local_seq_len:
        return tensor
    try:
        from megatron.core.parallel_state import get_tensor_model_parallel_rank
    except Exception:
        return tensor
    rank = int(get_tensor_model_parallel_rank())
    expected_global = local_seq_len * max(rank + 1, 1)
    if global_seq_len < expected_global:
        raise ValueError(
            "sequence-parallel local sequence length does not fit labels: "
            f"rank={rank} local_seq={local_seq_len} labels={tuple(tensor.shape)}"
        )
    start = rank * local_seq_len
    end = start + local_seq_len
    return tensor[:, start:end]


def _bridge_hidden_needs_sp_gather(
    hidden_states: torch.Tensor,
    labels: torch.Tensor | None,
    tp_size: int,
) -> bool:
    if labels is None or labels.dim() != 2 or tp_size <= 1:
        return False
    hidden_seq = int(hidden_states.shape[0])
    label_seq = int(labels.shape[1])
    return hidden_seq != label_seq and hidden_seq * int(tp_size) == label_seq


def _attention_dims(config: Any) -> tuple[int, int, int, int, int, int]:
    hidden_size = int(getattr(config, "hidden_size"))
    n_heads = int(getattr(config, "num_attention_heads"))
    n_kv_heads = int(getattr(config, "num_query_groups") or n_heads)
    head_dim = int(getattr(config, "kv_channels", hidden_size // n_heads))
    q_dim = n_heads * head_dim
    k_dim = n_kv_heads * head_dim
    v_dim = n_kv_heads * head_dim
    return hidden_size, n_heads, n_kv_heads, head_dim, q_dim, k_dim + v_dim


def _make_bridge_cce_backend():
    backend_name = os.environ.get("LBT_BRIDGE_FP4_CCE_BACKEND", "nvfp4").strip().lower()
    implementation = os.environ.get("LBT_BRIDGE_FP4_CCE_IMPLEMENTATION", "v4").strip().lower()
    quant_mode = os.environ.get("LBT_BRIDGE_FP4_CCE_QUANT_MODE", "enc").strip().lower()
    ignore_index = int(os.environ.get("LBT_BRIDGE_FP4_CCE_IGNORE_INDEX", "-100"))
    if backend_name == "nvfp4":
        from low_bits_training.cce.backend import _NVFP4Backend

        return _NVFP4Backend(
            ignore_index=ignore_index,
            implementation=implementation,
            quant_mode=quant_mode,
        )
    if backend_name == "mxfp4":
        from low_bits_training.cce.backend import _MXFP4Backend, _guard_mxfp4_cce_env

        _guard_mxfp4_cce_env()
        return _MXFP4Backend(
            ignore_index=ignore_index,
            implementation=implementation,
            quant_mode=quant_mode,
        )
    raise ValueError(f"Unsupported Bridge FP4 CCE backend: {backend_name!r}")


def _bridge_final_norm_prequant_enabled() -> bool:
    value = os.environ.get("LBT_BRIDGE_FP4_CCE_FINAL_NORM_PREQUANT")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _bridge_final_norm_prequant_quant_only_enabled() -> bool:
    return os.environ.get(
        "LBT_BRIDGE_FP4_CCE_FINAL_NORM_PREQUANT_QUANT_ONLY", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _bridge_backend_supports_final_norm_prequant(backend: Any) -> bool:
    return (
        getattr(backend, "name", None) == "mxfp4"
        and getattr(backend, "implementation", None) == "v4"
        and hasattr(backend, "training_loss_vocab_parallel_prequantized_x")
    )


def _install_bridge_final_norm_prequant_patch(model: Any) -> None:
    decoder = getattr(model, "decoder", None)
    norm = getattr(decoder, "final_layernorm", None)
    if norm is None or getattr(norm, "_lbt_bridge_cce_final_norm_patched", False):
        return

    original_forward = norm.forward
    model_ref = weakref.ref(model)

    def _lbt_bridge_final_norm_forward(self_norm, hidden_states, *args, **kwargs):
        owner = model_ref()
        if owner is None or not getattr(owner, "_lbt_bridge_cce_final_norm_active", False):
            return original_forward(hidden_states, *args, **kwargs)

        backend = getattr(owner, "_lbt_bridge_fp4_cce_backend", None)
        if backend is None:
            backend = _make_bridge_cce_backend()
            setattr(owner, "_lbt_bridge_fp4_cce_backend", backend)
        if not _bridge_backend_supports_final_norm_prequant(backend):
            return original_forward(hidden_states, *args, **kwargs)
        if not torch.cuda.is_available() or hidden_states.shape[-1] % 128 != 0:
            return original_forward(hidden_states, *args, **kwargs)

        pre_norm_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        if pre_norm_2d.shape[0] % 128 != 0:
            return original_forward(hidden_states, *args, **kwargs)
        producer_pre_norm_2d = pre_norm_2d
        if producer_pre_norm_2d.dtype != torch.bfloat16:
            producer_pre_norm_2d = producer_pre_norm_2d.to(torch.bfloat16)

        try:
            from low_bits_training.cce.backend import (
                _local_tensor_for_cce,
                _produce_final_norm_x_with_quant,
                _produce_final_norm_x_quant_only_for_cce,
            )

            norm_weight = _local_tensor_for_cce(self_norm.weight)
            norm_weight = norm_weight.to(
                device=producer_pre_norm_2d.device,
                dtype=torch.bfloat16,
            ).contiguous()
            epsilon = getattr(self_norm, "eps", 1e-5)
            if epsilon is None:
                epsilon = 1e-5
            quant_only = _bridge_final_norm_prequant_quant_only_enabled()
            producer = (
                _produce_final_norm_x_quant_only_for_cce
                if quant_only
                else _produce_final_norm_x_with_quant
            )
            hidden_2d, x_q, x_col_q = producer(
                producer_pre_norm_2d,
                norm_weight,
                float(epsilon),
                backend,
            )
        except Exception as exc:
            if os.environ.get("LBT_BRIDGE_FP4_CCE_FINAL_NORM_PREQUANT_STRICT", "0") == "1":
                raise
            _bridge_trace_if_enabled(
                "LBT_BRIDGE_TRACE_CCE_FINAL_NORM_PREQUANT",
                f"disabled_after_error={type(exc).__name__}: {exc}",
            )
            setattr(owner, "_lbt_bridge_cce_final_norm_active", False)
            setattr(owner, "_lbt_bridge_cce_final_norm_cache", None)
            return original_forward(hidden_states, *args, **kwargs)

        setattr(
            owner,
            "_lbt_bridge_cce_final_norm_cache",
            {
                "hidden_2d": hidden_2d,
                "x_q": x_q,
                "x_col_q": x_col_q,
                "shape": tuple(hidden_states.shape),
                "device": hidden_2d.device,
                "requires_prequant_consume": quant_only,
            },
        )
        return hidden_2d.reshape(hidden_states.shape)

    norm._lbt_bridge_cce_original_forward = original_forward
    norm.forward = types.MethodType(_lbt_bridge_final_norm_forward, norm)
    norm._lbt_bridge_cce_final_norm_patched = True


def _bridge_cce_fallback_impl() -> str:
    return os.environ.get("LBT_BRIDGE_FP4_CCE_TP_FALLBACK", "te").strip().lower()


def _bridge_cce_tp_mode() -> str:
    mode = os.environ.get("LBT_BRIDGE_FP4_CCE_TP_MODE", "fallback").strip().lower()
    if mode == "local":
        mode = "local_shard"
    if mode in {"vp", "vocab", "vocab-parallel"}:
        mode = "vocab_parallel"
    if mode not in {"fallback", "gather_full", "local_shard", "vocab_parallel"}:
        raise ValueError(
            "LBT_BRIDGE_FP4_CCE_TP_MODE must be one of "
            f"'fallback', 'gather_full', 'local_shard', or 'vocab_parallel', got {mode!r}"
        )
    return mode


def install_bridge_fp4_cce_postprocess_patch() -> None:
    """Install a Megatron-Core GPTModel postprocess hook for local FP4 CCE.

    This is Bridge-specific glue for the existing FP4 CCE kernels. It bypasses
    BF16 logit materialization when labels are present and
    ``config.cross_entropy_fusion_impl == "lbt_fp4_cce"``. The model still uses
    the normal output layer for inference/logit-only calls.
    """

    from megatron.core.models.gpt.gpt_model import GPTModel

    if getattr(GPTModel, "_lbt_bridge_fp4_cce_patched", False):
        return

    original_forward = GPTModel.forward
    original_postprocess = GPTModel._postprocess

    def _lbt_fp4_cce_forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        if labels is None and len(args) >= 5:
            labels = args[4]

        use_bridge_cce = (
            labels is not None
            and getattr(self.config, "cross_entropy_fusion_impl", None) == "lbt_fp4_cce"
            and _bridge_final_norm_prequant_enabled()
        )
        if use_bridge_cce:
            _install_bridge_final_norm_prequant_patch(self)

        previous_active = getattr(self, "_lbt_bridge_cce_final_norm_active", False)
        previous_cache = getattr(self, "_lbt_bridge_cce_final_norm_cache", None)
        setattr(self, "_lbt_bridge_cce_final_norm_active", bool(use_bridge_cce))
        setattr(self, "_lbt_bridge_cce_final_norm_cache", None)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            setattr(self, "_lbt_bridge_cce_final_norm_active", previous_active)
            setattr(self, "_lbt_bridge_cce_final_norm_cache", previous_cache)

    def _lbt_fp4_cce_postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        is_spec_decode=None,
    ):
        use_bridge_cce = (
            labels is not None
            and getattr(self.config, "cross_entropy_fusion_impl", None) == "lbt_fp4_cce"
        )
        tp_size = int(getattr(self.config, "tensor_model_parallel_size", 1))
        tp_cce_mode = "tp1"
        if use_bridge_cce and tp_size > 1:
            tp_cce_mode = _bridge_cce_tp_mode()
            if (
                tp_cce_mode == "fallback"
                and os.environ.get("LBT_BRIDGE_FP4_CCE_ALLOW_TP", "0") == "1"
            ):
                tp_cce_mode = "vocab_parallel"
            use_bridge_cce = tp_cce_mode != "fallback"
        if not use_bridge_cce:
            original_impl = getattr(self.config, "cross_entropy_fusion_impl", "native")
            if original_impl == "lbt_fp4_cce":
                self.config.cross_entropy_fusion_impl = _bridge_cce_fallback_impl()
            try:
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "before_fallback_postprocess_sync",
                )
                _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_BEFORE_FALLBACK_POSTPROCESS")
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "before_fallback_postprocess_call",
                )
                result = original_postprocess(
                    self,
                    hidden_states,
                    input_ids,
                    position_ids,
                    labels,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    mtp_in_postprocess=mtp_in_postprocess,
                    loss_mask=loss_mask,
                    decoder_input=decoder_input,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    runtime_gather_output=runtime_gather_output,
                    extra_block_kwargs=extra_block_kwargs,
                    inference_context=inference_context,
                    is_spec_decode=is_spec_decode,
                )
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "after_fallback_postprocess_call",
                )
                _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_AFTER_FALLBACK_POSTPROCESS")
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "after_fallback_postprocess_sync",
                )
                return result
            finally:
                self.config.cross_entropy_fusion_impl = original_impl

        if inference_context is not None and not self.training:
            return original_postprocess(
                self,
                hidden_states,
                input_ids,
                position_ids,
                labels,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                mtp_in_postprocess=mtp_in_postprocess,
                loss_mask=loss_mask,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                runtime_gather_output=runtime_gather_output,
                extra_block_kwargs=extra_block_kwargs,
                inference_context=inference_context,
                is_spec_decode=is_spec_decode,
            )
        if mtp_in_postprocess or getattr(self.config, "mtp_num_layers", 0):
            raise NotImplementedError("Bridge FP4 CCE hook does not support MTP yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("Bridge FP4 CCE hook does not support packed sequences yet.")
        if not self.post_process:
            return hidden_states
        if getattr(self.config, "use_mup", False):
            raise NotImplementedError("Bridge FP4 CCE hook does not support MuP logit scaling yet.")

        output_weight = (
            self.shared_embedding_or_output_weight()
            if self.share_embeddings_and_output_weights
            else None
        )
        weight = output_weight if output_weight is not None else self.output_layer.weight
        hidden_for_cce = hidden_states
        weight_for_cce = weight
        tp_group = getattr(getattr(self, "pg_collection", None), "tp", None)
        cache = getattr(self, "_lbt_bridge_cce_final_norm_cache", None)
        cache_requires_prequant_consume = bool(
            cache is not None and cache.get("requires_prequant_consume")
        )
        used_sp_gather_for_cce = False
        needs_sp_gather_for_cce = (
            tp_size > 1
            and tp_cce_mode in {"gather_full", "vocab_parallel"}
            and _bridge_hidden_needs_sp_gather(hidden_states, labels, tp_size)
        )
        if cache_requires_prequant_consume and (
            tp_cce_mode != "vocab_parallel" or needs_sp_gather_for_cce
        ):
            raise RuntimeError(
                "Quant-only Bridge final-norm prequant requires direct vocab-parallel "
                "CCE consumption without sequence-parallel gather."
            )
        if (
            needs_sp_gather_for_cce
        ):
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

            hidden_for_cce = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=True,
                group=tp_group,
            )
            used_sp_gather_for_cce = True
        if tp_size > 1 and tp_cce_mode == "gather_full":
            from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

            weight_for_cce = gather_from_tensor_model_parallel_region(
                weight.t().contiguous(),
                group=tp_group,
            ).t().contiguous()

        padded_vocab_size = getattr(self.config, "padded_vocab_size", None)
        if (
            padded_vocab_size is not None
            and int(weight_for_cce.shape[0]) != int(padded_vocab_size)
            and tp_cce_mode != "local_shard"
        ):
            if cache_requires_prequant_consume:
                raise RuntimeError(
                    "Quant-only Bridge final-norm prequant cannot fall back to "
                    "materialized logits postprocess."
                )
            original_impl = getattr(self.config, "cross_entropy_fusion_impl", "native")
            self.config.cross_entropy_fusion_impl = _bridge_cce_fallback_impl()
            try:
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "before_fallback_postprocess_sync",
                )
                _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_BEFORE_FALLBACK_POSTPROCESS")
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "before_fallback_postprocess_call",
                )
                result = original_postprocess(
                    self,
                    hidden_states,
                    input_ids,
                    position_ids,
                    labels,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    mtp_in_postprocess=mtp_in_postprocess,
                    loss_mask=loss_mask,
                    decoder_input=decoder_input,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    runtime_gather_output=runtime_gather_output,
                    extra_block_kwargs=extra_block_kwargs,
                    inference_context=inference_context,
                    is_spec_decode=is_spec_decode,
                )
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "after_fallback_postprocess_call",
                )
                _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_AFTER_FALLBACK_POSTPROCESS")
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_FALLBACK_POSTPROCESS",
                    "after_fallback_postprocess_sync",
                )
                return result
            finally:
                self.config.cross_entropy_fusion_impl = original_impl
        backend = getattr(self, "_lbt_bridge_fp4_cce_backend", None)
        if backend is None:
            backend = _make_bridge_cce_backend()
            setattr(self, "_lbt_bridge_fp4_cce_backend", backend)

        hidden_2d = hidden_for_cce.reshape(-1, hidden_for_cce.shape[-1]).contiguous()
        final_norm_prequant_cache = None
        if (
            cache is not None
            and not used_sp_gather_for_cce
            and tuple(hidden_for_cce.shape) == cache.get("shape")
        ):
            cached_hidden_2d = cache.get("hidden_2d")
            if (
                cached_hidden_2d is not None
                and tuple(cached_hidden_2d.shape) == tuple(hidden_2d.shape)
                and cached_hidden_2d.device == hidden_2d.device
            ):
                hidden_2d = cached_hidden_2d
                final_norm_prequant_cache = cache
                _bridge_trace_if_enabled(
                    "LBT_BRIDGE_TRACE_CCE_FINAL_NORM_PREQUANT",
                    f"using_cached_prequant shape={tuple(hidden_2d.shape)}",
                )
        if cache_requires_prequant_consume and final_norm_prequant_cache is None:
            raise RuntimeError(
                "Quant-only Bridge final-norm prequant cache was not consumed by CCE."
            )
        labels_local = labels
        loss_mask_local = loss_mask
        if labels.dim() == 2 and labels.shape[1] != hidden_for_cce.shape[0]:
            labels_local = _slice_sequence_parallel_batch_first(
                labels,
                int(hidden_for_cce.shape[0]),
            )
            if loss_mask is not None:
                loss_mask_local = _slice_sequence_parallel_batch_first(
                    loss_mask,
                    int(hidden_for_cce.shape[0]),
                )
        labels_sb = labels_local.transpose(0, 1).contiguous().to(
            device=hidden_2d.device,
            dtype=torch.int64,
        )
        if loss_mask_local is not None:
            mask_sb = loss_mask_local.transpose(0, 1).contiguous().to(device=hidden_2d.device)
            labels_sb = torch.where(
                mask_sb.to(torch.bool),
                labels_sb,
                torch.full_like(labels_sb, int(backend.ignore_index)),
            )
        labels_1d = labels_sb.reshape(-1)
        debug = os.environ.get("LBT_BRIDGE_FP4_CCE_DEBUG", "0") == "1"
        if debug and not getattr(self, "_lbt_bridge_fp4_cce_weight_hooked", False):
            rank = int(os.environ.get("RANK", "0"))

            def _debug_weight_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                finite = bool(torch.isfinite(grad).all().item())
                nbad = int((~torch.isfinite(grad)).sum().item()) if not finite else 0
                max_abs = float(
                    torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                    .abs()
                    .max()
                    .item()
                )
                print(
                    "[LBT_BRIDGE_CCE] "
                    f"rank={rank} output_weight_grad finite={finite} nbad={nbad} max_abs={max_abs:.8e}",
                    file=sys.stderr,
                    flush=True,
                )
                return grad

            weight.register_hook(_debug_weight_grad_hook)
            setattr(self, "_lbt_bridge_fp4_cce_weight_hooked", True)
        if debug and not getattr(self, "_lbt_bridge_fp4_cce_hidden_hooked", False):
            rank = int(os.environ.get("RANK", "0"))

            def _debug_hidden_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                finite = bool(torch.isfinite(grad).all().item())
                nbad = int((~torch.isfinite(grad)).sum().item()) if not finite else 0
                max_abs = float(
                    torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                    .abs()
                    .max()
                    .item()
                )
                print(
                    "[LBT_BRIDGE_CCE] "
                    f"rank={rank} hidden_grad finite={finite} nbad={nbad} max_abs={max_abs:.8e}",
                    file=sys.stderr,
                    flush=True,
                )
                return grad

            hidden_states.register_hook(_debug_hidden_grad_hook)
            setattr(self, "_lbt_bridge_fp4_cce_hidden_hooked", True)
        if debug and not getattr(self, "_lbt_bridge_fp4_cce_debug_printed", False):
            valid_count = int((labels_1d != int(backend.ignore_index)).sum().item())
            rank = int(os.environ.get("RANK", "0"))
            hidden_max = float(hidden_2d.detach().float().abs().max().item())
            weight_max = float(weight.detach().float().abs().max().item())
            print(
                "[LBT_BRIDGE_CCE] "
                f"rank={rank} hidden={tuple(hidden_states.shape)} "
                f"hidden_for_cce={tuple(hidden_for_cce.shape)} hidden_2d={tuple(hidden_2d.shape)} "
                f"labels={tuple(labels.shape)} labels_local={tuple(labels_local.shape)} "
                f"labels_1d={tuple(labels_1d.shape)} "
                f"loss_mask={None if loss_mask is None else tuple(loss_mask.shape)} "
                f"loss_mask_local={None if loss_mask_local is None else tuple(loss_mask_local.shape)} "
                f"valid={valid_count} weight={tuple(weight.shape)} "
                f"weight_for_cce={tuple(weight_for_cce.shape)} tp_cce_mode={tp_cce_mode} "
                f"backend={backend.name} "
                f"hidden_max={hidden_max:.8e} weight_max={weight_max:.8e}",
                file=sys.stderr,
                flush=True,
            )
            setattr(self, "_lbt_bridge_fp4_cce_debug_printed", True)
        valid_mask_1d = labels_1d != int(backend.ignore_index)
        if tp_size > 1 and tp_cce_mode == "vocab_parallel":
            tp_group_for_cce, actual_tp_size, tp_rank = _tp_info(tp_group)
            local_vocab_size = int(weight_for_cce.shape[0])
            global_vocab_size = (
                int(padded_vocab_size)
                if padded_vocab_size is not None
                else local_vocab_size * int(actual_tp_size)
            )
            vocab_start = int(tp_rank) * local_vocab_size
            if final_norm_prequant_cache is not None:
                loss_mean = backend.training_loss_vocab_parallel_prequantized_x(
                    hidden_2d,
                    final_norm_prequant_cache["x_q"],
                    final_norm_prequant_cache["x_col_q"],
                    weight_for_cce,
                    labels_1d,
                    tp_group=tp_group_for_cce,
                    vocab_start=vocab_start,
                    global_vocab_size=global_vocab_size,
                    reduce_dE=not used_sp_gather_for_cce,
                )
            else:
                loss_mean = backend.training_loss_vocab_parallel(
                    hidden_2d,
                    weight_for_cce,
                    labels_1d,
                    tp_group=tp_group_for_cce,
                    vocab_start=vocab_start,
                    global_vocab_size=global_vocab_size,
                    reduce_dE=not used_sp_gather_for_cce,
                )
        else:
            loss_mean = backend.training_loss(hidden_2d, weight_for_cce, labels_1d)
        if debug:
            rank = int(os.environ.get("RANK", "0"))
            finite = bool(torch.isfinite(loss_mean.detach()).item())
            print(
                f"[LBT_BRIDGE_CCE] rank={rank} loss_mean={float(loss_mean.detach().float().item()):.8f} finite={finite}",
                file=sys.stderr,
                flush=True,
            )
        local_losses = loss_mean.reshape(1).expand(labels_1d.numel())
        return local_losses, valid_mask_1d.to(device=hidden_2d.device, dtype=torch.float32)

    GPTModel._lbt_bridge_original_postprocess = original_postprocess
    GPTModel._lbt_bridge_original_forward = original_forward
    GPTModel.forward = _lbt_fp4_cce_forward
    GPTModel._postprocess = _lbt_fp4_cce_postprocess
    GPTModel._lbt_bridge_fp4_cce_patched = True


def install_bridge_fp4_ddp_debug_patch() -> None:
    """Debug-only patch to identify nonfinite Megatron DDP bucket slices."""

    if os.environ.get("LBT_BRIDGE_FP4_CCE_DEBUG", "0") != "1":
        return
    import megatron.core.distributed.param_and_grad_buffer as pgb

    if getattr(pgb, "_lbt_bridge_ddp_debug_patched", False):
        return

    original_init = pgb._ParamAndGradBuffer.__init__
    original_check_grads = pgb._ParamAndGradBucketGroup.check_grads

    def _debug_init(self, *args, **kwargs):
        params_with_names = kwargs.get("params_with_names", None)
        if params_with_names is None and len(args) >= 4:
            params_with_names = args[3]
        if params_with_names is not None:
            for param, name in params_with_names:
                setattr(param, "_lbt_param_name", name)
        return original_init(self, *args, **kwargs)

    def _debug_check_grads(self, check_for_nan_or_inf, check_for_large):
        rank = int(os.environ.get("RANK", "0"))
        for bucket_index, bucket in enumerate(self.buckets):
            grad_data = bucket.grad_data
            if torch.isfinite(grad_data).all():
                continue
            bad_total = int((~torch.isfinite(grad_data)).sum().item())
            print(
                f"[LBT_BRIDGE_DDP] rank={rank} bucket={bucket_index} "
                f"grad_data_shape={tuple(grad_data.shape)} nonfinite={bad_total}",
                file=sys.stderr,
                flush=True,
            )
            printed = 0
            for param in bucket.params_list:
                start, end = bucket.param_to_index[param]
                view = grad_data[start:end]
                if torch.isfinite(view).all():
                    continue
                name = getattr(param, "_lbt_param_name", "<unnamed>")
                nbad = int((~torch.isfinite(view)).sum().item())
                max_abs = float(
                    torch.nan_to_num(view.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                    .abs()
                    .max()
                    .item()
                )
                main_grad = getattr(param, "main_grad", None)
                main_grad_finite = (
                    None if main_grad is None else bool(torch.isfinite(main_grad).all().item())
                )
                print(
                    "[LBT_BRIDGE_DDP] "
                    f"rank={rank} param={name} shape={tuple(param.shape)} "
                    f"slice=({start},{end}) nonfinite={nbad} max_abs={max_abs:.8e} "
                    f"main_grad_finite={main_grad_finite}",
                    file=sys.stderr,
                    flush=True,
                )
                printed += 1
                if printed >= 20:
                    break
        return original_check_grads(self, check_for_nan_or_inf, check_for_large)

    pgb._ParamAndGradBuffer.__init__ = _debug_init
    pgb._ParamAndGradBucketGroup.check_grads = _debug_check_grads
    pgb._lbt_bridge_ddp_debug_patched = True


def install_bridge_fp4_separate_qkv_patch() -> None:
    """Let MCore attention consume local FP4 Q/K/V without mixed-QKV packing."""

    import megatron.core.transformer.attention as attention_mod

    SelfAttention = attention_mod.SelfAttention
    if getattr(SelfAttention, "_lbt_bridge_fp4_separate_qkv_patched", False):
        return

    original_get_qkv = SelfAttention.get_query_key_value_tensors
    apply_module = original_get_qkv.__globals__["apply_module"]

    def _lbt_get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ):
        linear_qkv = getattr(self, "linear_qkv", None)
        if (
            key_value_states is None
            and not output_gate
            and split_qkv
            and isinstance(linear_qkv, MCoreFusedQKVLinear)
        ):
            xq, xk, xv = apply_module(linear_qkv).forward_separate(hidden_states)
            query = xq.view(
                xq.size(0),
                xq.size(1),
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            key = xk.view(
                xk.size(0),
                xk.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
            value = xv.view(
                xv.size(0),
                xv.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
            if self.q_layernorm is not None:
                query = apply_module(self.q_layernorm)(query)
            if self.k_layernorm is not None:
                key = apply_module(self.k_layernorm)(key)
            if self.config.test_mode:
                self.run_realtime_tests()
            return query, key, value

        return original_get_qkv(
            self,
            hidden_states,
            key_value_states=key_value_states,
            output_gate=output_gate,
            split_qkv=split_qkv,
        )

    SelfAttention._lbt_bridge_original_get_query_key_value_tensors = original_get_qkv
    SelfAttention.get_query_key_value_tensors = _lbt_get_query_key_value_tensors
    SelfAttention._lbt_bridge_fp4_separate_qkv_patched = True


def install_bridge_fp4_transformer_block_sync_patch() -> None:
    """Debug-only final TransformerBlock boundary sync probes."""

    if not (
        os.environ.get("LBT_BRIDGE_SYNC_BEFORE_TRANSFORMER_POSTPROCESS", "0") == "1"
        or os.environ.get("LBT_BRIDGE_TRACE_TRANSFORMER_POSTPROCESS", "0") == "1"
    ):
        return

    from megatron.core.transformer.transformer_block import TransformerBlock

    if getattr(TransformerBlock, "_lbt_bridge_fp4_sync_patched", False):
        return

    original_forward = TransformerBlock.forward

    def _lbt_transformer_forward(self, *args, **kwargs):
        hidden_states = original_forward(self, *args, **kwargs)
        _bridge_trace_if_enabled(
            "LBT_BRIDGE_TRACE_TRANSFORMER_POSTPROCESS",
            "before_transformer_block_output_sync",
        )
        _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_BEFORE_TRANSFORMER_POSTPROCESS")
        _bridge_trace_if_enabled(
            "LBT_BRIDGE_TRACE_TRANSFORMER_POSTPROCESS",
            "after_transformer_block_output_sync",
        )
        return hidden_states

    TransformerBlock.forward = _lbt_transformer_forward
    TransformerBlock._lbt_bridge_fp4_sync_patched = True


class MCoreFusedQKVLinear(nn.Module):
    """MCore ``linear_qkv`` replacement using fused RMSNorm + stacked QKV.

    MCore expects mixed QKV grouped as ``q... k v | q... k v`` per KV group.
    The local kernel returns contiguous ``[all q], [all k], [all v]`` outputs,
    so this wrapper repacks the views before handing the tensor back to MCore's
    normal RoPE/CP/SDPA path.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: Any,
        init_method: Any,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        fp4_backend: str = "nvfp4_tk_v5",
        tp_comm_buffer_name: str | None = None,
        tp_group: Any | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        del gather_output, bias, skip_bias_add, is_expert, tp_comm_buffer_name

        (
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            q_dim,
            kv_total,
        ) = _attention_dims(config)
        k_dim = kv_total // 2
        v_dim = kv_total // 2
        total_out = q_dim + k_dim + v_dim
        if int(input_size) != hidden_size or int(output_size) != total_out:
            raise ValueError(
                "MCoreFusedQKVLinear shape mismatch: "
                f"input={input_size}, output={output_size}, expected=({hidden_size}, {total_out})"
            )
        self.tp_group, self.tp_size, self.tp_rank = _tp_info(tp_group)
        if self.tp_size > 1:
            if n_kv_heads < self.tp_size:
                raise NotImplementedError(
                    "MCoreFusedQKVLinear does not support TP larger than the number "
                    f"of KV heads yet: kv_heads={n_kv_heads}, TP={self.tp_size}"
                )
            if n_heads % self.tp_size != 0 or n_kv_heads % self.tp_size != 0:
                raise ValueError(
                    "MCoreFusedQKVLinear requires attention heads and KV heads to "
                    f"divide TP exactly: heads={n_heads}, kv_heads={n_kv_heads}, TP={self.tp_size}"
                )
        self.sequence_parallel = bool(getattr(config, "sequence_parallel", False)) and self.tp_size > 1
        n_heads_local = _divide_exact(n_heads, self.tp_size, "num_attention_heads")
        n_kv_heads_local = _divide_exact(n_kv_heads, self.tp_size, "num_query_groups")
        q_dim_local = n_heads_local * head_dim
        k_dim_local = n_kv_heads_local * head_dim
        v_dim_local = n_kv_heads_local * head_dim

        self.hidden_size = hidden_size
        self.n_heads = n_heads_local
        self.n_kv_heads = n_kv_heads_local
        self.head_dim = head_dim
        self.q_dim = q_dim_local
        self.k_dim = k_dim_local
        self.v_dim = v_dim_local
        self.q_heads_per_group = n_heads_local // n_kv_heads_local
        self.epsilon = float(getattr(config, "layernorm_epsilon", 1e-5))
        self.fp4_backend = fp4_backend.lower()
        if self.fp4_backend not in {"nvfp4_tk_v5", "nvfp4_localcta_v4", "mxfp4"}:
            raise ValueError(f"Unsupported Bridge QKV FP4 backend: {fp4_backend}")
        dtype = getattr(config, "params_dtype", torch.bfloat16)
        device = _current_device()

        self.norm_weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.w_qkv = nn.Parameter(
            torch.empty(
                q_dim_local + k_dim_local + v_dim_local,
                hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        if self.tp_size > 1:
            _mark_tp_param(self.w_qkv, dim=0)
        setattr(self.w_qkv, "allreduce", True)
        setattr(self.norm_weight, "allreduce", True)
        self.qkv_quantizers = (
            _NVFP4QuantizerBundle() if self.fp4_backend != "mxfp4" else None
        )
        self._workspace: torch.Tensor | None = None
        self._workspace_device: torch.device | None = None
        self._debug_name = f"mcore_bridge:{self.fp4_backend}:qkv"
        self.layer_number: int | None = None
        self._debug_params_hooked = False

        if bool(getattr(config, "perform_initialization", True)):
            with torch.no_grad():
                self.norm_weight.fill_(1.0)
                init_method(self.w_qkv)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.w_qkv

    def _ensure_workspace(self, device: torch.device) -> torch.Tensor:
        shared = _bridge_fp4_workspace(device)
        if shared is not None:
            self._workspace = shared
            self._workspace_device = device
            return shared
        if self._workspace is None or self._workspace_device != device:
            self._workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
            self._workspace_device = device
        return self._workspace

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number
        self._debug_name = f"mcore_bridge:{self.fp4_backend}:qkv:{layer_number}"

    def _debug_label(self, suffix: str) -> str:
        name = getattr(self.w_qkv, "_lbt_param_name", None) or self._debug_name
        return f"{name}:{suffix}"

    def _install_debug_param_hooks(self) -> None:
        if self._debug_params_hooked or not _bridge_debug_enabled():
            return

        def _param_hook(param: torch.nn.Parameter, suffix: str):
            def _hook(grad: torch.Tensor) -> torch.Tensor:
                name = getattr(param, "_lbt_param_name", None) or self._debug_name
                _print_bridge_grad_stats(f"{name}:{suffix}", grad)
                return grad

            return _hook

        self.w_qkv.register_hook(_param_hook(self.w_qkv, "w_qkv_grad"))
        self.norm_weight.register_hook(_param_hook(self.norm_weight, "norm_weight_grad"))
        self._debug_params_hooked = True

    def _pack_mcore_mixed_qkv(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        seq_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        q = xq.view(seq_len, batch_size, self.n_kv_heads, self.q_heads_per_group, self.head_dim)
        k = xk.view(seq_len, batch_size, self.n_kv_heads, 1, self.head_dim)
        v = xv.view(seq_len, batch_size, self.n_kv_heads, 1, self.head_dim)
        return torch.cat((q, k, v), dim=3).reshape(seq_len, batch_size, -1)

    def forward_separate(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Q/K/V as [seq, batch, hidden] tensors for the patched MCore path."""

        if hidden_states.dim() != 3:
            raise ValueError(
                f"MCoreFusedQKVLinear expects [seq, batch, hidden], got {tuple(hidden_states.shape)}"
            )

        x = hidden_states
        if self.tp_size > 1:
            if self.sequence_parallel:
                from megatron.core.tensor_parallel.mappings import (
                    gather_from_sequence_parallel_region,
                )

                x = gather_from_sequence_parallel_region(
                    x,
                    tensor_parallel_output_grad=True,
                    group=self.tp_group,
                )
            else:
                from megatron.core.tensor_parallel.mappings import (
                    copy_to_tensor_model_parallel_region,
                )

                x = copy_to_tensor_model_parallel_region(x, group=self.tp_group)

        seq_len, batch_size, hidden_size = x.shape
        x_2d = x.reshape(seq_len * batch_size, hidden_size)
        workspace = self._ensure_workspace(x.device)
        if _bridge_debug_enabled():
            hidden_states.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("qkv_hidden_states_grad"), grad)
                    or grad
                )
            )
            x_2d.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("qkv_input_2d_grad"), grad)
                    or grad
                )
            )
        timing = _bridge_wrapper_timing_begin("qkv_kernel", self.layer_number)
        try:
            if self.fp4_backend == "mxfp4":
                from low_bits_training.quantization.mxfp4_fused_linear import (
                    _FusedQKVFunction_MXFP4_TK,
                )

                xq, xk, xv = _FusedQKVFunction_MXFP4_TK.apply(
                    x_2d,
                    self.w_qkv,
                    self.norm_weight,
                    self.epsilon,
                    self.q_dim,
                    self.k_dim,
                    self.v_dim,
                    None,
                    0,
                    0,
                    self.head_dim,
                    self._debug_name,
                )
            else:
                from low_bits_training.quantization.fused_te_linear import _FusedQKVFunction_TK

                assert self.qkv_quantizers is not None
                xq, xk, xv = _FusedQKVFunction_TK.apply(
                    x_2d,
                    self.w_qkv,
                    self.norm_weight,
                    self.epsilon,
                    self.q_dim,
                    self.k_dim,
                    self.v_dim,
                    None,
                    0,
                    0,
                    self.head_dim,
                    self.qkv_quantizers.activation,
                    self.qkv_quantizers.weight,
                    self.qkv_quantizers.grad,
                    workspace,
                    self._debug_name,
                )
        finally:
            _bridge_wrapper_timing_end(timing)
        if _bridge_debug_enabled():
            self._install_debug_param_hooks()
            xq.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("xq_grad"), grad) or grad
                )
            )
            xk.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("xk_grad"), grad) or grad
                )
            )
            xv.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("xv_grad"), grad) or grad
                )
            )
        return (
            xq.view(seq_len, batch_size, self.q_dim),
            xk.view(seq_len, batch_size, self.k_dim),
            xv.view(seq_len, batch_size, self.v_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        xq, xk, xv = self.forward_separate(hidden_states)
        seq_len, batch_size = xq.shape[:2]
        mixed_qkv = self._pack_mcore_mixed_qkv(xq, xk, xv, seq_len, batch_size)
        if _bridge_debug_enabled():
            self._install_debug_param_hooks()
            mixed_qkv.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("mixed_qkv_grad"), grad) or grad
                )
            )
            xq.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("xq_grad"), grad) or grad
                )
            )
            xk.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("xk_grad"), grad) or grad
                )
            )
            xv.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("xv_grad"), grad) or grad
                )
            )
        return mixed_qkv, None

    def backward_dw(self) -> None:
        return None


class MCoreFusedWoLinear(nn.Module):
    """MCore output projection replacement using the local FP4 WO kernel."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: Any,
        init_method: Any,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        fp4_backend: str = "nvfp4_tk_v5",
        tp_comm_buffer_name: str | None = None,
        tp_group: Any | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        del bias, input_is_parallel, skip_bias_add, is_expert, tp_comm_buffer_name

        hidden_size, _n_heads, _n_kv_heads, _head_dim, q_dim, _kv_total = _attention_dims(config)
        if int(input_size) != q_dim or int(output_size) != hidden_size:
            raise ValueError(
                "MCoreFusedWoLinear shape mismatch: "
                f"input={input_size}, output={output_size}, expected=({q_dim}, {hidden_size})"
            )
        self.tp_group, self.tp_size, self.tp_rank = _tp_info(tp_group)
        self.sequence_parallel = bool(getattr(config, "sequence_parallel", False)) and self.tp_size > 1
        q_dim_per_partition = _divide_exact(q_dim, self.tp_size, "q_dim")

        dtype = getattr(config, "params_dtype", torch.bfloat16)
        device = _current_device()
        self.hidden_size = hidden_size
        self.q_dim = q_dim_per_partition
        self.fp4_backend = fp4_backend.lower()
        if self.fp4_backend not in {"nvfp4_tk_v5", "nvfp4_localcta_v4", "mxfp4"}:
            raise ValueError(f"Unsupported Bridge WO FP4 backend: {fp4_backend}")
        self.wo_weight = nn.Parameter(
            torch.empty(hidden_size, q_dim_per_partition, device=device, dtype=dtype)
        )
        if self.tp_size > 1:
            _mark_tp_param(self.wo_weight, dim=1)
        setattr(self.wo_weight, "allreduce", True)
        self.wo_quantizers = _NVFP4QuantizerBundle() if self.fp4_backend != "mxfp4" else None
        self._workspace: torch.Tensor | None = None
        self._workspace_device: torch.device | None = None
        self._debug_name = f"mcore_bridge:{self.fp4_backend}:wo"
        self.layer_number: int | None = None
        self._debug_params_hooked = False

        if bool(getattr(config, "perform_initialization", True)):
            init_method(self.wo_weight)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.wo_weight

    def _ensure_workspace(self, device: torch.device) -> torch.Tensor:
        shared = _bridge_fp4_workspace(device)
        if shared is not None:
            self._workspace = shared
            self._workspace_device = device
            return shared
        if self._workspace is None or self._workspace_device != device:
            self._workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
            self._workspace_device = device
        return self._workspace

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number
        self._debug_name = f"mcore_bridge:{self.fp4_backend}:wo:{layer_number}"

    def _debug_label(self, suffix: str) -> str:
        name = getattr(self.wo_weight, "_lbt_param_name", None) or self._debug_name
        return f"{name}:{suffix}"

    def _install_debug_param_hooks(self) -> None:
        if self._debug_params_hooked or not _bridge_debug_enabled():
            return

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            name = getattr(self.wo_weight, "_lbt_param_name", None) or self._debug_name
            _print_bridge_grad_stats(f"{name}:wo_weight_grad", grad)
            return grad

        self.wo_weight.register_hook(_hook)
        self._debug_params_hooked = True

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"MCoreFusedWoLinear expects [seq, batch, hidden], got {tuple(hidden_states.shape)}"
            )
        seq_len, batch_size, hidden_size = hidden_states.shape
        if hidden_size != self.q_dim:
            raise ValueError(
                "MCoreFusedWoLinear input shard mismatch: "
                f"got {hidden_size}, expected {self.q_dim}"
            )
        x_2d = hidden_states.reshape(seq_len * batch_size, hidden_size)
        workspace = self._ensure_workspace(hidden_states.device)
        timing = _bridge_wrapper_timing_begin("wo_kernel", self.layer_number)
        try:
            if self.fp4_backend == "mxfp4":
                from low_bits_training.quantization.mxfp4_fused_linear import _WoFunction_MXFP4_TK

                y = _WoFunction_MXFP4_TK.apply(
                    x_2d,
                    self.wo_weight,
                    self._debug_name,
                    None,
                )
            else:
                from low_bits_training.quantization.fused_te_linear import _WoFunction_TK

                assert self.wo_quantizers is not None
                y = _WoFunction_TK.apply(
                    x_2d,
                    self.wo_weight,
                    self.wo_quantizers.activation,
                    self.wo_quantizers.weight,
                    self.wo_quantizers.grad,
                    workspace,
                    self._debug_name,
                )
        finally:
            _bridge_wrapper_timing_end(timing)
        output = y.view(seq_len, batch_size, self.hidden_size)
        if self.tp_size > 1:
            if self.sequence_parallel:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_scatter_to_sequence_parallel_region,
                )

                timing = _bridge_wrapper_timing_begin("wo_sp_reduce_scatter", self.layer_number)
                try:
                    output = reduce_scatter_to_sequence_parallel_region(output, group=self.tp_group)
                finally:
                    _bridge_wrapper_timing_end(timing)
            else:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_from_tensor_model_parallel_region,
                )

                timing = _bridge_wrapper_timing_begin("wo_tp_reduce", self.layer_number)
                try:
                    output = reduce_from_tensor_model_parallel_region(output, group=self.tp_group)
                finally:
                    _bridge_wrapper_timing_end(timing)
        if _bridge_debug_enabled():
            self._install_debug_param_hooks()
            hidden_states.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("wo_input_grad"), grad) or grad
                )
            )
            y.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("wo_output_2d_grad"), grad)
                    or grad
                )
            )
            output.register_hook(
                lambda grad: (
                    _print_bridge_grad_stats(self._debug_label("wo_output_grad"), grad) or grad
                )
            )
        return output, None

    def backward_dw(self) -> None:
        return None


class MCoreFusedSwiGLUFFN(nn.Module):
    """MCore-compatible wrapper around fused FP4 SwiGLU FFN modules.

    MCore's regular MLP owns only fc1/activation/fc2 because the TE spec fuses
    the pre-MLP RMSNorm into fc1. Our kernels own RMSNorm plus all FFN GEMMs, so
    this module is intended to be installed as ``TransformerLayerSubmodules.mlp``
    with ``pre_mlp_layernorm=IdentityOp``.
    """

    def __init__(
        self,
        *,
        config: Any,
        fp4_backend: str,
        submodules: Any | None = None,
        tp_group: Any | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        del submodules

        if not bool(getattr(config, "gated_linear_unit", False)):
            raise ValueError("MCoreFusedSwiGLUFFN requires a gated SwiGLU MLP config.")

        hidden_size = int(getattr(config, "hidden_size"))
        ffn_hidden_size = int(getattr(config, "ffn_hidden_size"))
        norm_eps = float(getattr(config, "layernorm_epsilon", 1e-5))
        dtype = getattr(config, "params_dtype", torch.bfloat16)
        device = _current_device()

        backend = fp4_backend.lower()
        if backend in {"nvfp4_tk_v5", "nvfp4_localcta_v4"}:
            from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4_TK

            ffn_cls = FusedFeedForwardFP4_TK
        elif backend == "mxfp4":
            from low_bits_training.quantization.mxfp4_fused_linear import FusedFeedForwardMXFP4_TK

            ffn_cls = FusedFeedForwardMXFP4_TK
        else:
            raise ValueError(f"Unknown FP4 MCore FFN backend: {fp4_backend}")

        self.config = config
        self.fp4_backend = backend
        self.tp_group, self.tp_size, self.tp_rank = _tp_info(tp_group)
        self.sequence_parallel = bool(getattr(config, "sequence_parallel", False)) and self.tp_size > 1
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn_hidden_size_per_partition = _divide_exact(
            ffn_hidden_size, self.tp_size, "ffn_hidden_size"
        )
        self.layer_number: int | None = None
        ffn_kwargs = {}
        if backend in {"nvfp4_tk_v5", "nvfp4_localcta_v4", "mxfp4"}:
            ffn_kwargs["packed_w13"] = True
        self.ffn = ffn_cls(
            hidden_size,
            self.ffn_hidden_size_per_partition,
            norm_eps=norm_eps,
            bias=False,
            device=device,
            dtype=dtype,
            **ffn_kwargs,
        )
        if self.tp_size > 1:
            _mark_tp_param(self.ffn.w13_weight, dim=0)
            _mark_tp_param(self.ffn.w2_weight, dim=1)
        setattr(self.ffn.w13_weight, "allreduce", True)
        setattr(self.ffn.w2_weight, "allreduce", True)
        setattr(self.ffn.norm_weight, "allreduce", True)
        self._initialize_like_mcore(config)

    def _initialize_like_mcore(self, config: Any) -> None:
        if not bool(getattr(config, "perform_initialization", True)):
            return
        init_method = getattr(config, "init_method", None)
        output_init_method = getattr(config, "output_layer_init_method", None)
        with torch.no_grad():
            self.ffn.norm_weight.fill_(1.0)
            if init_method is not None:
                if getattr(self.ffn, "packed_w13", False):
                    init_method(self.ffn._w1_weight_view())
                    init_method(self.ffn._w3_weight_view())
                else:
                    init_method(self.ffn.w1_weight)
                    init_method(self.ffn.w3_weight)
            if output_init_method is not None:
                output_init_method(self.ffn.w2_weight)

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number
        setattr(self.ffn, "_lbt_debug_name", f"mcore_bridge:{self.fp4_backend}:ffn:{layer_number}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        del padding_mask
        x = hidden_states
        if self.tp_size > 1 and self.sequence_parallel:
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

            x = gather_from_sequence_parallel_region(
                x,
                tensor_parallel_output_grad=True,
                group=self.tp_group,
            )
        timing = _bridge_wrapper_timing_begin("ffn_kernel", self.layer_number)
        try:
            out = self.ffn(x)
        finally:
            _bridge_wrapper_timing_end(timing)
        if self.tp_size > 1:
            if self.sequence_parallel:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_scatter_to_sequence_parallel_region,
                )

                timing = _bridge_wrapper_timing_begin("ffn_sp_reduce_scatter", self.layer_number)
                try:
                    out = reduce_scatter_to_sequence_parallel_region(out, group=self.tp_group)
                finally:
                    _bridge_wrapper_timing_end(timing)
            else:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_from_tensor_model_parallel_region,
                )

                _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_BEFORE_FFN_TP_REDUCE")
                timing = _bridge_wrapper_timing_begin("ffn_tp_reduce", self.layer_number)
                try:
                    out = reduce_from_tensor_model_parallel_region(out, group=self.tp_group)
                finally:
                    _bridge_wrapper_timing_end(timing)
                _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_AFTER_FFN_TP_REDUCE")
        _bridge_trace_if_enabled(
            "LBT_BRIDGE_TRACE_FFN_FORWARD_BOUNDARY",
            f"ffn_layer={self.layer_number} before_after_ffn_forward_sync",
        )
        _bridge_sync_if_enabled("LBT_BRIDGE_SYNC_AFTER_FFN_FORWARD")
        _bridge_trace_if_enabled(
            "LBT_BRIDGE_TRACE_FFN_FORWARD_BOUNDARY",
            f"ffn_layer={self.layer_number} after_after_ffn_forward_sync",
        )
        return out, None

    def backward_dw(self) -> None:
        """MCore fine-grained schedules expect this hook on MLP modules."""
        return None


def fp4_bridge_env(fp4_backend: str) -> dict[str, str]:
    """Environment preset for Bridge runs using the local FP4 MLP adapter."""

    def _env(name: str, default: str) -> str:
        return os.environ.get(name, default)

    def _set_profile_default(env: dict[str, str], name: str, default: str) -> None:
        env[name] = _env(name, default)

    def _apply_localcta_v4_tp2_profile(env: dict[str, str]) -> None:
        profile = _normalize_bridge_localcta_v4_tp2_profile(
            os.environ.get("LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE")
        )
        env["LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE"] = profile
        if profile == "off":
            return
        if profile in {"tp2_fused_split", "tp2_fused_split_overlap"}:
            _set_profile_default(env, "USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_PRODUCER", "1")
            _set_profile_default(env, "USE_TK_LOCALCTA_V4_FFN_STRIDED_SG_DGRAD", "1")
            _set_profile_default(env, "USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_FINALIZE", "1")
        if profile in {"tp2_overlap", "tp2_fused_split_overlap"}:
            _set_profile_default(env, "USE_TK_FFN_DISABLE_WGRAD_STREAM", "0")
            _set_profile_default(env, "USE_TK_LOCALCTA_FFN_W2_WGRAD_OVERLAP_MIN_M", "32768")
            _set_profile_default(env, "USE_TK_LOCALCTA_V4_FFN_W2_WEIGHT_QUANT_OVERLAP", "1")

    noextras = {
        "NVTE_NVFP4_DISABLE_RHT": "1",
        "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "NVTE_NVFP4_ENCODE_CENTRIC": "0",
        "NVFP4_USE_RHT": "0",
        "NVFP4_RHT_ACTIVATION": "0",
        "NVFP4_RHT_GRAD": "0",
        "NVFP4_RHT_WEIGHT": "0",
        "NVFP4_USE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SR_ACTIVATION": "0",
        "NVFP4_SR_GRAD": "0",
        "NVFP4_SR_WEIGHT": "0",
        "NVFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SCALE_SR_ACTIVATION": "0",
        "NVFP4_SCALE_SR_GRAD": "0",
        "NVFP4_SCALE_SR_WEIGHT": "0",
    }
    backend = _normalize_bridge_fp4_backend(fp4_backend)
    if backend == "mixed_localcta_mxfp4":
        env = fp4_bridge_env("nvfp4_localcta_v4")
        mxfp4_env = fp4_bridge_env("mxfp4")
        for key, value in mxfp4_env.items():
            if (
                key.startswith("MXFP4_")
                or key.startswith("FP4_MXFP4")
                or key.startswith("FP4_MATMUL")
                or key in {"FP4_CCE_TK_ROOT"}
            ):
                env[key] = value
        env.update(
            {
                "LBT_BRIDGE_FP4_LAYERWISE_MIXED": "1",
                "LBT_BRIDGE_FP4_MIXED_POLICY": os.environ.get(
                    "LBT_BRIDGE_FP4_MIXED_POLICY", "front_localcta"
                ),
                "LBT_BRIDGE_FP4_MIXED_CCE_BACKEND": os.environ.get(
                    "LBT_BRIDGE_FP4_MIXED_CCE_BACKEND",
                    os.environ.get("LBT_BRIDGE_FP4_CCE_BACKEND", "nvfp4"),
                ),
            }
        )
        env["LBT_BRIDGE_FP4_CCE_BACKEND"] = env["LBT_BRIDGE_FP4_MIXED_CCE_BACKEND"]
        return env
    if backend == "mxfp4":
        return {
            "FP4_MXFP4_ROOT": os.environ.get(
                "FP4_MXFP4_ROOT", "/opt/mfu/EXTERNAL_PATH"
            ),
            "FP4_MATMUL_GEMM_ROOT": os.environ.get(
                "FP4_MATMUL_GEMM_ROOT",
                "/opt/mfu/EXTERNAL_PATH",
            ),
            "FP4_CCE_TK_ROOT": os.environ.get(
                "FP4_CCE_TK_ROOT",
                "/opt/mfu/EXTERNAL_PATH",
            ),
            "LBT_BRIDGE_FP4_CCE_BACKEND": os.environ.get(
                "LBT_BRIDGE_FP4_CCE_BACKEND", "mxfp4"
            ),
            "LBT_BRIDGE_FP4_CCE_IMPLEMENTATION": os.environ.get(
                "LBT_BRIDGE_FP4_CCE_IMPLEMENTATION", "v4"
            ),
            "LBT_BRIDGE_FP4_CCE_QUANT_MODE": os.environ.get(
                "LBT_BRIDGE_FP4_CCE_QUANT_MODE", "enc"
            ),
            "LBT_BRIDGE_FP4_CCE_ALLOW_TP": os.environ.get(
                "LBT_BRIDGE_FP4_CCE_ALLOW_TP", "1"
            ),
            "MXFP4_BACKEND_VERSION": "v4",
            "MXFP4_USE_RHT": "0",
            "MXFP4_RHT_ACTIVATION": "0",
            "MXFP4_RHT_GRAD": "0",
            "MXFP4_RHT_WEIGHT": "0",
            "MXFP4_RHT_AXES": os.environ.get("MXFP4_RHT_AXES", "row"),
            "MXFP4_RHT_RANDOM_SIGN_MASK": os.environ.get("MXFP4_RHT_RANDOM_SIGN_MASK", "0"),
            "MXFP4_USE_STOCHASTIC_ROUNDING": "0",
            "MXFP4_SR_ACTIVATION": "0",
            "MXFP4_SR_GRAD": "0",
            "MXFP4_SR_WEIGHT": "0",
            "MXFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
            "MXFP4_SCALE_SR_ACTIVATION": "0",
            "MXFP4_SCALE_SR_GRAD": "0",
            "MXFP4_SCALE_SR_WEIGHT": "0",
            "MXFP4_USE_QKV_ROPE_EPILOGUE": _env("MXFP4_USE_QKV_ROPE_EPILOGUE", "1"),
            "MXFP4_USE_QKV_DIRECT_OUTPUTS": _env("MXFP4_USE_QKV_DIRECT_OUTPUTS", "1"),
            "MXFP4_USE_QKV_RMSNORM_QUANT_FUSION": _env("MXFP4_USE_QKV_RMSNORM_QUANT_FUSION", "1"),
            "MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD": _env("MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD", "0"),
            "MXFP4_QKV_BWD_STATE_SLOTS": _env("MXFP4_QKV_BWD_STATE_SLOTS", "4"),
            "MXFP4_USE_QKV_BF16_WGRAD": _env("MXFP4_USE_QKV_BF16_WGRAD", "0"),
            "MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD": _env("MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD", "1"),
            "MXFP4_SPLIT2_FFN_ONEPASS_CONFIG_IDX": _env("MXFP4_SPLIT2_FFN_ONEPASS_CONFIG_IDX", "3"),
            "MXFP4_USE_SPLIT2_FFN_INPLACE_QUANT": _env("MXFP4_USE_SPLIT2_FFN_INPLACE_QUANT", "1"),
            "MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP": _env("MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP", "1"),
            "MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT": _env("MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT", "1"),
            "MXFP4_USE_SPLIT2_FFN_PRODUCER_SPLIT": _env("MXFP4_USE_SPLIT2_FFN_PRODUCER_SPLIT", "0"),
            "MXFP4_USE_QKV_COMBINED_BWD": os.environ.get(
                "MXFP4_USE_QKV_COMBINED_BWD", "1"
            ),
            "MXFP4_USE_SPLIT3_QKV_STAGE_COPY": os.environ.get(
                "MXFP4_USE_SPLIT3_QKV_STAGE_COPY", "1"
            ),
            "MXFP4_SPLIT3_QKV_STAGE_COPY_MASK": os.environ.get(
                "MXFP4_SPLIT3_QKV_STAGE_COPY_MASK", "none"
            ),
            "MXFP4_QKV_GEMM_CONFIG_M32768_N3072_K4096": os.environ.get(
                "MXFP4_QKV_GEMM_CONFIG_M32768_N3072_K4096", "0"
            ),
            "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM": os.environ.get(
                "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM", "0"
            ),
            "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM_DGAMMA": os.environ.get(
                "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM_DGAMMA", "0"
            ),
            "MXFP4_USE_QKV_FWD_WEIGHT_QUANT_OVERLAP": os.environ.get(
                "MXFP4_USE_QKV_FWD_WEIGHT_QUANT_OVERLAP", "0"
            ),
            "MXFP4_USE_FFN_WGRAD_OVERLAP": os.environ.get(
                "MXFP4_USE_FFN_WGRAD_OVERLAP", "1"
            ),
            "MXFP4_FFN_WGRAD_OVERLAP_MIN_M": os.environ.get(
                "MXFP4_FFN_WGRAD_OVERLAP_MIN_M", "32768"
            ),
            "MXFP4_USE_BWD_WGRAD_OVERLAP": os.environ.get(
                "MXFP4_USE_BWD_WGRAD_OVERLAP", "0"
            ),
            "MXFP4_USE_BWD_STATE_CACHE": os.environ.get(
                "MXFP4_USE_BWD_STATE_CACHE", "0"
            ),
            "MXFP4_WGRAD_GEMM_CONFIG_M4096_N2048": os.environ.get(
                "MXFP4_WGRAD_GEMM_CONFIG_M4096_N2048", "18"
            ),
            "MXFP4_WGRAD_GEMM_CONFIG_M2048_N2048": os.environ.get(
                "MXFP4_WGRAD_GEMM_CONFIG_M2048_N2048", "18"
            ),
            "MXFP4_WGRAD_GEMM_CONFIG_M2048_N6144": os.environ.get(
                "MXFP4_WGRAD_GEMM_CONFIG_M2048_N6144", "18"
            ),
            "MXFP4_USE_RESIDUAL_FUSION": _env("MXFP4_USE_RESIDUAL_FUSION", "1"),
            "MXFP4_USE_RESIDUAL_FUSION_FFN": _env("MXFP4_USE_RESIDUAL_FUSION_FFN", "1"),
            "MXFP4_USE_RESIDUAL_FUSION_ATTN": _env("MXFP4_USE_RESIDUAL_FUSION_ATTN", "0"),
            "MXFP4_ALLOW_UNSAFE_ATTN_FFN_RESIDUAL_OVERLAP": _env("MXFP4_ALLOW_UNSAFE_ATTN_FFN_RESIDUAL_OVERLAP", "0"),
            "MXFP4_UNSAFE_RESIDUAL_FALLBACK": os.environ.get(
                "MXFP4_UNSAFE_RESIDUAL_FALLBACK", "prefer_ffn"
            ),
            "MXFP4_USE_GEMM_RESIDUAL_KERNEL": _env("MXFP4_USE_GEMM_RESIDUAL_KERNEL", "1"),
            "MXFP4_USE_FUSED_RMSNORM_QUANT_RHT": _env("MXFP4_USE_FUSED_RMSNORM_QUANT_RHT", "1"),
            "MXFP4_USE_FUSED_SILU_FFN_QUANT": _env("MXFP4_USE_FUSED_SILU_FFN_QUANT", "1"),
            "MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN": _env("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN", "1"),
            "MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT": _env("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT", "0"),
            "MXFP4_USE_SIMPLE_SQRELU_FUSED_W2": _env("MXFP4_USE_SIMPLE_SQRELU_FUSED_W2", "0"),
            "MXFP4_USE_WO_ATTN_LAYOUT": _env("MXFP4_USE_WO_ATTN_LAYOUT", "0"),
            "MXFP4_USE_QKV_FORWARD_SYNC": os.environ.get(
                "MXFP4_USE_QKV_FORWARD_SYNC",
                os.environ.get("LBT_BRIDGE_SYNC_QKV_FWD", "0"),
            ),
        }

    env = {
        **noextras,
        "USE_TK_GEMM": "1",
        "USE_TK_QUANT": "1",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_FUSED": "0",
        "FP4_ATTN_BACKEND": "tk",
        "FP4_FFN_BACKEND": "tk",
        "FP4_CCE_TK_ROOT": os.environ.get(
            "FP4_CCE_TK_ROOT",
            "/opt/mfu/EXTERNAL_PATH",
        ),
        "LBT_BRIDGE_FP4_CCE_BACKEND": os.environ.get(
            "LBT_BRIDGE_FP4_CCE_BACKEND", "nvfp4"
        ),
        "LBT_BRIDGE_FP4_CCE_IMPLEMENTATION": os.environ.get(
            "LBT_BRIDGE_FP4_CCE_IMPLEMENTATION", "v4"
        ),
        "LBT_BRIDGE_FP4_CCE_QUANT_MODE": os.environ.get(
            "LBT_BRIDGE_FP4_CCE_QUANT_MODE", "enc"
        ),
        "LBT_BRIDGE_FP4_CCE_ALLOW_TP": os.environ.get(
            "LBT_BRIDGE_FP4_CCE_ALLOW_TP", "1"
        ),
        "NVTE_CUSTOM_QUANT": "1",
        "USE_TK_QKV_ROPE_EPILOGUE": "1",
        "USE_TK_V5_STRIDED_Q_ATTN": _env("USE_TK_V5_STRIDED_Q_ATTN", "1"),
        "USE_TK_ATTN_SYNC_QKV_FWD": os.environ.get(
            "USE_TK_ATTN_SYNC_QKV_FWD",
            os.environ.get("LBT_BRIDGE_SYNC_QKV_FWD", "0"),
        ),
        "USE_TK_ATTN_SYNC_QKV_BWD": os.environ.get(
            "USE_TK_ATTN_SYNC_QKV_BWD",
            os.environ.get("LBT_BRIDGE_SYNC_QKV_BWD", "1"),
        ),
        "USE_TK_FFN_SPLIT2_OPT_PRODUCER": "1",
        "USE_TK_FFN_SPLIT_CACHE": "1",
        "USE_TK_FFN_RECOMPUTE_H13": "1",
        "USE_TK_FFN_CACHED_RETURN_TRANSPOSE": _env(
            "USE_TK_FFN_CACHED_RETURN_TRANSPOSE", "1"
        ),
        "USE_TK_FFN_FUSED_SUM_RMS": "1",
        "USE_TK_FFN_OVERLAP_RMS_WGRAD": "1",
        "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
        "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
        "FP4_CCE_NVFP4_EXACT_NORM_QUANT": "0",
        "FP4_CCE_ASSUME_NONEMPTY_LABELS": "1",
        "FP4_CCE_V4_NVFP4_G_CACHE": "0",
        "FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE": "1",
        "FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE": "0",
        "FP4_CCE_V4_NVFP4_P_DATA_SR": "1",
        "FP4_CCE_V4_NVFP4_P_SCALE_SR": "0",
        "FP4_CCE_V4_NVFP4_P_TARGET_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_P_TOP1_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_STAGED_P_CACHE": "1",
        "FP4_CCE_V4_NVFP4_TILED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_TMA_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_FUSED_STAGED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_DIRECT_SOFTMAX": "0",
        "FP4_CCE_V4_STRICT_FUSED_SPARSE": "1",
        "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_G_CACHE": os.environ.get(
            "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_G_CACHE", "auto"
        ),
        "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_CHUNK": os.environ.get(
            "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_CHUNK", "1024"
        ),
    }
    if backend == "nvfp4_localcta_v4":
        env.update(
            {
                "USE_TK_LOCALCTA": "1",
                "USE_TK_LOCALCTA_VARIANT": "v4",
                "FP4_FFN_BACKEND": "localcta",
                "FP4_ATTN_BACKEND": "localcta",
                "USE_TK_LOCALCTA_V4_FAST_PREPARED_PRODUCER": _env(
                    "USE_TK_LOCALCTA_V4_FAST_PREPARED_PRODUCER", "0"
                ),
                "USE_TK_LOCALCTA_V4_ROW_PREPARED_COL_OUTER": _env(
                    "USE_TK_LOCALCTA_V4_ROW_PREPARED_COL_OUTER", "1"
                ),
                "USE_TK_LOCALCTA_V4_FAST_FORWARD_GEMM": _env(
                    "USE_TK_LOCALCTA_V4_FAST_FORWARD_GEMM", "1"
                ),
                "USE_TK_LOCALCTA_V4_FAST_SINGLE_DGRAD": _env(
                    "USE_TK_LOCALCTA_V4_FAST_SINGLE_DGRAD", "1"
                ),
                "USE_TK_LOCALCTA_V4_FAST_SINGLE_WGRAD": _env(
                    "USE_TK_LOCALCTA_V4_FAST_SINGLE_WGRAD", "1"
                ),
                "USE_TK_LOCALCTA_V4_FAST_FFN_RMSNORM_QUANT": os.environ.get(
                    "USE_TK_LOCALCTA_V4_FAST_FFN_RMSNORM_QUANT", "1"
                ),
                "USE_TK_LOCALCTA_V4_FFN_SEPARATE_BF16_FINAL_SG": _env(
                    "USE_TK_LOCALCTA_V4_FFN_SEPARATE_BF16_FINAL_SG", "1"
                ),
                "USE_TK_LOCALCTA_V4_ROW_PREPARED_RMSNORM_QUANT": _env(
                    "USE_TK_LOCALCTA_V4_ROW_PREPARED_RMSNORM_QUANT", "0"
                ),
                "USE_TK_LOCALCTA_V4_FFN_W2_WEIGHT_QUANT_OVERLAP": _env(
                    "USE_TK_LOCALCTA_V4_FFN_W2_WEIGHT_QUANT_OVERLAP", "0"
                ),
                "USE_TK_FFN_DISABLE_WGRAD_STREAM": _env(
                    "USE_TK_FFN_DISABLE_WGRAD_STREAM", "1"
                ),
                "USE_TK_LOCALCTA_V4_FAST_W2_WGRAD": os.environ.get(
                    "USE_TK_LOCALCTA_V4_FAST_W2_WGRAD", "1"
                ),
                "USE_TK_LOCALCTA_V4_FFN_DIRECT_GROUPED_WGRAD_LAYOUT": _env(
                    "USE_TK_LOCALCTA_V4_FFN_DIRECT_GROUPED_WGRAD_LAYOUT", "1"
                ),
                "USE_TK_RMSNORM_BWD_SINGLE_OUT": _env(
                    "USE_TK_RMSNORM_BWD_SINGLE_OUT", "0"
                ),
                "USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_PRODUCER": _env(
                    "USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_PRODUCER", "0"
                ),
                "USE_TK_LOCALCTA_V4_FFN_STRIDED_SG_DGRAD": _env(
                    "USE_TK_LOCALCTA_V4_FFN_STRIDED_SG_DGRAD", "0"
                ),
                "USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD": _env(
                    "USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD", "1"
                ),
                "USE_TK_ATTN_SYNC_QKV_BWD": os.environ.get(
                    "USE_TK_ATTN_SYNC_QKV_BWD",
                    os.environ.get("LBT_BRIDGE_SYNC_QKV_BWD", "0"),
                ),
                "USE_TK_QKV_LOCALCTA_FAST_ACT": _env(
                    "USE_TK_QKV_LOCALCTA_FAST_ACT", "1"
                ),
                "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": _env(
                    "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND", "split3"
                ),
                "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT": _env(
                    "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", "1"
                ),
                "USE_TK_QKV_LOCALCTA_FUSED_RMSNORM_QUANT": _env(
                    "USE_TK_QKV_LOCALCTA_FUSED_RMSNORM_QUANT", "0"
                ),
                "USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD": _env(
                    "USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD", "1"
                ),
                "USE_TK_LOCALCTA_V4_FAST_WO_DGRAD": _env(
                    "USE_TK_LOCALCTA_V4_FAST_WO_DGRAD", "1"
                ),
                "USE_TK_LOCALCTA_V4_FAST_WO_WGRAD": _env(
                    "USE_TK_LOCALCTA_V4_FAST_WO_WGRAD", "1"
                ),
                "USE_TK_LOCALCTA_V4_WO_ATTN_LAYOUT": _env(
                    "USE_TK_LOCALCTA_V4_WO_ATTN_LAYOUT", "0"
                ),
                "USE_TK_LOCALCTA_V4_FAST_DATA_SR": _env("USE_TK_LOCALCTA_V4_FAST_DATA_SR", "1"),
                "USE_TK_LOCALCTA_V4_DIRECT_GRID_SM_MULT": _env(
                    "USE_TK_LOCALCTA_V4_DIRECT_GRID_SM_MULT", "1.0"
                ),
                "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW": _env(
                    "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW", "1"
                ),
                "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_RAW_MULTIPLIER": _env(
                    "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_RAW_MULTIPLIER", "2.0"
                ),
                "USE_TK_LOCALCTA_V4_SPLIT2_PRECOMPUTE_AMAX": _env(
                    "USE_TK_LOCALCTA_V4_SPLIT2_PRECOMPUTE_AMAX", "0"
                ),
                "USE_TK_LOCALCTA_V4_FFN_ROW_BF16_PREPARED_DERIV_QUANT": _env(
                    "USE_TK_LOCALCTA_V4_FFN_ROW_BF16_PREPARED_DERIV_QUANT", "0"
                ),
                "USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER": _env(
                    "USE_TK_LOCALCTA_V4_W2_DGRAD_SILU_PRODUCER", "0"
                ),
                "USE_TK_LOCALCTA_V4_CLONE_FFN_BWD_RETURNS": _env(
                    "USE_TK_LOCALCTA_V4_CLONE_FFN_BWD_RETURNS", "0"
                ),
                "USE_TK_LOCALCTA_V4_CLONE_FFN_BWD_GRAD_INPUT": _env(
                    "USE_TK_LOCALCTA_V4_CLONE_FFN_BWD_GRAD_INPUT", "0"
                ),
                "USE_TK_LOCALCTA_V4_SYNC_FFN_RMS_BWD": _env(
                    "USE_TK_LOCALCTA_V4_SYNC_FFN_RMS_BWD", "0"
                ),
                "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD": _env(
                    "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD", "1"
                ),
                "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_FILTER": _env(
                    "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_FILTER", "ffn:16"
                ),
                "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_ONCE": _env(
                    "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_ONCE", "1"
                ),
                "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_DEVICE": _env(
                    "USE_TK_LOCALCTA_V4_SYNC_AFTER_FFN_BWD_DEVICE", "1"
                ),
            }
        )
        _apply_localcta_v4_tp2_profile(env)
    elif backend != "nvfp4_tk_v5":
        raise ValueError(f"Unknown FP4 Bridge env backend: {fp4_backend}")
    return env


def _module_spec_with_fp4_backend(spec: Any, backend: str) -> Any:
    from megatron.core.transformer.spec_utils import ModuleSpec

    backend = _normalize_bridge_fp4_backend(backend)
    if isinstance(spec, ModuleSpec) and spec.module in {
        MCoreFusedQKVLinear,
        MCoreFusedWoLinear,
        MCoreFusedSwiGLUFFN,
    }:
        copied = copy.copy(spec)
        params = dict(getattr(spec, "params", {}) or {})
        params["fp4_backend"] = backend
        copied.params = params
        return copied
    return spec


def _submodules_with_layer_backend(submodules: Any, backend: str) -> Any:
    from megatron.core.transformer.spec_utils import ModuleSpec

    backend = _normalize_bridge_fp4_backend(backend)
    mixed_submodules = copy.copy(submodules)
    mixed_submodules.mlp = _module_spec_with_fp4_backend(mixed_submodules.mlp, backend)

    self_attention = getattr(mixed_submodules, "self_attention", None)
    if isinstance(self_attention, ModuleSpec) and self_attention.submodules is not None:
        self_attention_spec = copy.copy(self_attention)
        attention_submodules = copy.copy(self_attention.submodules)
        attention_submodules.linear_qkv = _module_spec_with_fp4_backend(
            attention_submodules.linear_qkv,
            backend,
        )
        attention_submodules.linear_proj = _module_spec_with_fp4_backend(
            attention_submodules.linear_proj,
            backend,
        )
        self_attention_spec.submodules = attention_submodules
        mixed_submodules.self_attention = self_attention_spec
    return mixed_submodules


def _make_layerwise_mixed_transformer_layer_cls():
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        get_transformer_layer_offset,
    )
    from megatron.core.utils import get_pg_rank

    class LayerwiseMixedFP4TransformerLayer(TransformerLayer):
        def __init__(
            self,
            config: Any,
            submodules: Any,
            layer_number: int = 1,
            hidden_dropout: Any | None = None,
            pg_collection: Any | None = None,
            vp_stage: Any | None = None,
            is_mtp_layer: bool = False,
            add_layer_offset: bool = True,
            pp_layer_offset: int | None = None,
        ) -> None:
            if is_mtp_layer or not add_layer_offset:
                absolute_layer_number = layer_number
            elif (
                getattr(config, "pipeline_model_parallel_size", 1) <= 1
                and getattr(config, "virtual_pipeline_model_parallel_size", None) is None
            ):
                absolute_layer_number = layer_number
            else:
                if pg_collection is None:
                    pg_collection_for_offset = ProcessGroupCollection.use_mpu_process_groups()
                else:
                    pg_collection_for_offset = pg_collection
                absolute_layer_number = layer_number + get_transformer_layer_offset(
                    config,
                    vp_stage,
                    get_pg_rank(pg_collection_for_offset.pp),
                )
            total_layers = getattr(config, "num_layers", None)
            backend = _bridge_layerwise_mixed_backend(
                int(absolute_layer_number),
                int(total_layers) if total_layers is not None else None,
            )
            if os.environ.get("LBT_BRIDGE_FP4_MIXED_VERBOSE", "0") == "1":
                print(
                    "[LBT_BRIDGE_FP4_MIXED] "
                    f"layer={absolute_layer_number} backend={backend}",
                    file=sys.stderr,
                    flush=True,
                )
            mixed_submodules = _submodules_with_layer_backend(submodules, backend)
            super().__init__(
                config=config,
                submodules=mixed_submodules,
                layer_number=layer_number,
                hidden_dropout=hidden_dropout,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
                is_mtp_layer=is_mtp_layer,
                add_layer_offset=add_layer_offset,
                pp_layer_offset=pp_layer_offset,
            )

    return LayerwiseMixedFP4TransformerLayer


def make_fp4_mlp_layer_spec(provider: Any, *, fp4_backend: str):
    """Return a TE-attention Bridge layer spec with the MLP replaced by FP4."""

    from megatron.core.extensions.transformer_engine import HAVE_TE
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if not HAVE_TE:
        raise RuntimeError("Transformer Engine is required for the Bridge FP4 MLP spec.")

    submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=getattr(provider, "num_moe_experts", None),
        moe_grouped_gemm=getattr(provider, "moe_grouped_gemm", False),
        qk_layernorm=getattr(provider, "qk_layernorm", False),
        multi_latent_attention=getattr(provider, "multi_latent_attention", False),
        qk_l2_norm=False,
        use_kitchen=getattr(provider, "use_kitchen", False),
        use_te_activation_func=getattr(provider, "use_te_activation_func", False),
        use_kitchen_attention=getattr(provider, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(provider, "kitchen_attention_backend", "sdpa"),
        mla_down_proj_fusion=getattr(provider, "mla_down_proj_fusion", False),
    )
    if _is_bridge_layerwise_mixed_backend(fp4_backend):
        initial_backend = "nvfp4_localcta_v4"
        layer_module = _make_layerwise_mixed_transformer_layer_cls()
    else:
        initial_backend = _normalize_bridge_fp4_backend(fp4_backend)
        layer_module = TransformerLayer
    submodules.mlp = ModuleSpec(
        module=MCoreFusedSwiGLUFFN,
        params={"fp4_backend": initial_backend},
    )
    return ModuleSpec(module=layer_module, submodules=submodules)


def make_fp4_projection_mlp_layer_spec(provider: Any, *, fp4_backend: str):
    """Return a Bridge layer spec with fused QKV, WO, and MLP FP4 modules."""

    from megatron.core.extensions.transformer_engine import HAVE_TE
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if not HAVE_TE:
        raise RuntimeError("Transformer Engine is required for the Bridge FP4 projection spec.")
    if bool(getattr(provider, "qk_layernorm", False)):
        raise NotImplementedError("The Bridge FP4 projection wrapper does not support QK LN yet.")
    if bool(getattr(provider, "multi_latent_attention", False)):
        raise NotImplementedError("The Bridge FP4 projection wrapper only supports regular GQA.")

    submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=getattr(provider, "num_moe_experts", None),
        moe_grouped_gemm=getattr(provider, "moe_grouped_gemm", False),
        qk_layernorm=False,
        multi_latent_attention=False,
        qk_l2_norm=False,
        use_kitchen=getattr(provider, "use_kitchen", False),
        use_te_activation_func=getattr(provider, "use_te_activation_func", False),
        use_kitchen_attention=getattr(provider, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(provider, "kitchen_attention_backend", "sdpa"),
        mla_down_proj_fusion=False,
    )
    if _is_bridge_layerwise_mixed_backend(fp4_backend):
        initial_backend = "nvfp4_localcta_v4"
        layer_module = _make_layerwise_mixed_transformer_layer_cls()
    else:
        initial_backend = _normalize_bridge_fp4_backend(fp4_backend)
        layer_module = TransformerLayer

    submodules.input_layernorm = IdentityOp
    submodules.pre_mlp_layernorm = IdentityOp
    submodules.self_attention.submodules.linear_qkv = ModuleSpec(
        module=MCoreFusedQKVLinear,
        params={"fp4_backend": initial_backend},
    )
    submodules.self_attention.submodules.linear_proj = ModuleSpec(
        module=MCoreFusedWoLinear,
        params={"fp4_backend": initial_backend},
    )
    submodules.self_attention.submodules.q_layernorm = IdentityOp
    submodules.self_attention.submodules.k_layernorm = IdentityOp
    submodules.mlp = ModuleSpec(
        module=MCoreFusedSwiGLUFFN,
        params={"fp4_backend": initial_backend},
    )
    submodules.sharded_state_dict_keys_map = {}
    return ModuleSpec(module=layer_module, submodules=submodules)
