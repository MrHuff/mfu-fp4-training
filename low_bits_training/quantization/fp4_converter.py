"""
FP4 converter for Llama models — module-level replacement.

Strategy:
  - Attention (wq/wk/wv/wo) + attention_norm: FusedAttentionFP4
    Absorbs attention_norm, stacks QKV weights, fused rmsnorm+quant, single GEMM.

  - FFN: Replace ENTIRE feed_forward + ffn_norm with FusedFeedForwardFP4:
    Absorbs ffn_norm, uses custom autograd for w1/w2/w3.

  - Output/lm_head: Float32Linear for numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

try:
    from transformer_engine.common.recipe import NVFP4BlockScaling
except ImportError:
    NVFP4BlockScaling = None

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb as te_apply_rotary_pos_emb
except ImportError:
    te_apply_rotary_pos_emb = None

import functools
import math
import types
import weakref

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, attr.split('.'), obj)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

try:
    from .mxfp_custom_te_fp4 import BoundRecipeLinear
except (ImportError, AttributeError):
    BoundRecipeLinear = None
try:
    from .te_parity_linear_tex import TEParityLinearTex
except (ImportError, OSError):
    TEParityLinearTex = None
from .fused_te_linear import (
    FusedFeedForwardFP4_TE, FusedAttentionFP4_TE,
    FusedFeedForwardFP4_TK, FusedAttentionFP4_TK,
    FusedSquaredReLUFeedForwardFP4_TK,
    SimpleFP4Linear, NVFP4RMSNormLinearTK,
    _tk_stage_trace,
    use_tk_stage_trace,
    _trace_backend_choice,
    _attn_debug_check_finite,
    use_tk_attn_debug_finite,
    _attn_capture_path,
    _append_attn_capture,
    _tensor_capture_stats,
    _attn_layout_path,
    _append_attn_layout,
    _attn_layout_event_enabled,
    _tensor_layout_group,
    clear_fused_fp4_step_caches,
    use_tk_localcta_v4_wo_attn_layout,
    _get_te_fused,
    _as_contiguous_bf16,
    _safe_trunc_normal_,
)
from .float32_linear import Float32Linear
from .nemotron_h_projection_policy import (
    NemotronHFusedAttentionWrapper,
    is_nemotron_h_attention_block,
    nemotron_h_fp4_output_head_enabled,
    replace_nemotron_h_projection_linears,
    use_nemotron_h_fused_attention,
)
from .sqrelu import sqrelu
from .tk_gemm import (
    clear_tk_step_caches,
    use_tk_localcta_v4_ffn_residual_epilogue,
    use_tk_v5_ffn_residual_epilogue,
    use_tk_rmsnorm_bwd_single_out,
)

import inspect
import os
import re


_VALID_FP4_BACKENDS = {"te", "tk", "localcta", "localcta_fused"}


def _maybe_enable_nvfp4_live_path(tk_backend: str) -> None:
    """Default NVFP4 TK/localCTA runs onto the MXFP4-style live wrapper route.

    The lower-level helpers keep individual env overrides, so setting
    USE_NVFP4_MXFP4_LIVE_PATH=0 remains a hard opt-out.
    """
    if tk_backend == 'te':
        return
    if (
        os.environ.get('USE_NVFP4_MXFP4_LIVE_PATH') is None
        and os.environ.get('NVFP4_MIMIC_MXFP4_LIVE_PATH') is None
    ):
        os.environ['USE_NVFP4_MXFP4_LIVE_PATH'] = '1'


def _normalize_localcta_v4_profile(profile: str | None) -> str:
    value = (profile or "").strip().lower().replace("-", "_")
    aliases = {
        "": "off",
        "0": "off",
        "false": "off",
        "none": "off",
        "off": "off",
        "default": "off",
        "fused_split": "fused_split",
        "prepared_split2": "fused_split",
        "tp2_fused_split": "fused_split",
        "tp2_prepared_split2": "fused_split",
        "overlap": "overlap",
        "tp2_overlap": "overlap",
        "fused_split_overlap": "fused_split_overlap",
        "prepared_split2_overlap": "fused_split_overlap",
        "tp2_fused_split_overlap": "fused_split_overlap",
        "highwater": "highwater",
        "hw": "highwater",
        "mxfp4_highwater": "highwater",
    }
    if value not in aliases:
        raise ValueError(
            "LBT_LOCALCTA_V4_PROFILE must be one of off, fused_split, overlap, "
            f"fused_split_overlap, or highwater; got {profile!r}"
        )
    return aliases[value]


def apply_localcta_v4_profile_defaults() -> str:
    """Apply opt-in localCTA-v4 FFN defaults for regular TorchTitan runs."""

    profile = _normalize_localcta_v4_profile(
        os.environ.get(
            "LBT_LOCALCTA_V4_PROFILE",
            os.environ.get("LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE"),
        )
    )
    if profile == "off":
        return profile
    if profile in {"fused_split", "fused_split_overlap"}:
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_PRODUCER", "1")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FFN_STRIDED_SG_DGRAD", "1")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_FINALIZE", "1")
    if profile in {"overlap", "fused_split_overlap", "highwater"}:
        os.environ.setdefault("USE_TK_FFN_DISABLE_WGRAD_STREAM", "0")
        os.environ.setdefault("USE_TK_LOCALCTA_FFN_W2_WGRAD_OVERLAP_MIN_M", "32768")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FFN_W2_WEIGHT_QUANT_OVERLAP", "1")
    if profile == "highwater":
        # Folding the tiny chunk SG into FP8 before outer-SG finalization loses
        # scale dynamic range, so keep the correct raw CUDA/TK path by default.
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FFN_PREPARED_SPLIT2_PRODUCER", "0")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FAST_FFN_RMSNORM_QUANT", "1")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FAST_W2_WGRAD", "1")
        # The single-output native helper accumulates dgamma with atomics
        # across rows, so CTA scheduling changes exact resume results.  The
        # tiled path writes fixed row partials and reduces them in a fixed
        # order; make that the only supported localCTA-v4 production policy.
        os.environ.setdefault("USE_TK_RMSNORM_BWD_SINGLE_OUT", "0")
        # The one-pass QKV dgrad folds outer scales into E4M3 prepared scales.
        # At the Llama-8B GQA shape this is both slower and measurably less
        # accurate than the scale-preserving strided-sum kernel.
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD", "0")
        os.environ.setdefault("USE_TK_QKV_LOCALCTA_FAST_ACT", "1")
        os.environ.setdefault("USE_TK_QKV_LOCALCTA_DGRAD_BACKEND", "split3")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", "1")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD", "0")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_NATIVE_QK_ROPE", "1")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER", "1")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FAST_WO_DGRAD", "1")
        os.environ.setdefault("USE_TK_LOCALCTA_V4_FAST_WO_WGRAD", "1")
        if os.environ.get("USE_TK_FFN_DISABLE_WGRAD_STREAM") == "1":
            os.environ.setdefault("USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO", "0")
    # Validate here so an inherited/explicit unsafe override fails during
    # conversion rather than on the first backward pass.  The selector repeats
    # this check as defense in depth for callers that bypass the converter.
    use_tk_rmsnorm_bwd_single_out()
    return profile


def _is_localcta_v4() -> bool:
    return os.environ.get('USE_TK_LOCALCTA_VARIANT', 'v1').strip().lower() == 'v4'


_ATTN_ENABLE_GQA_CACHE: dict[type, bool] = {}
_FP4_TE_ROPE_CACHE: dict[tuple[int, tuple[int, ...], torch.dtype, str, int | None], torch.Tensor] = {}
_FP4_TE_ROPE_FALLBACK_WARNED = False


def _use_fp4_te_fused_rope() -> bool:
    value = os.environ.get("USE_FP4_TK_TE_FUSED_ROPE")
    if value is not None:
        return value == "1"
    return True


def _get_fp4_te_rope_freqs(freqs_cis: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(freqs_cis):
        raise RuntimeError(f"expected complex freqs_cis, got {freqs_cis.dtype}")
    key = (
        freqs_cis.data_ptr(),
        tuple(freqs_cis.shape),
        freqs_cis.dtype,
        freqs_cis.device.type,
        freqs_cis.device.index,
    )
    cached = _FP4_TE_ROPE_CACHE.get(key)
    if cached is None:
        if len(_FP4_TE_ROPE_CACHE) >= 8:
            _FP4_TE_ROPE_CACHE.clear()
        angles = torch.angle(freqs_cis)
        cached = (
            torch.stack((angles, angles), dim=-1)
            .flatten(-2)
            .view(freqs_cis.shape[0], 1, 1, -1)
            .contiguous()
        )
        _FP4_TE_ROPE_CACHE[key] = cached
    return cached


def _apply_rotary_emb_fast(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _FP4_TE_ROPE_FALLBACK_WARNED
    from torchtitan.models.llama3.model.model import apply_rotary_emb

    if (
        not _use_fp4_te_fused_rope()
        or te_apply_rotary_pos_emb is None
        or not xq.is_cuda
        or not xk.is_cuda
        or not freqs_cis.is_cuda
    ):
        return apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    try:
        rope_freqs = _get_fp4_te_rope_freqs(freqs_cis)
        return (
            te_apply_rotary_pos_emb(
                xq,
                rope_freqs,
                tensor_format="bshd",
                fused=True,
                interleaved=True,
            ),
            te_apply_rotary_pos_emb(
                xk,
                rope_freqs,
                tensor_format="bshd",
                fused=True,
                interleaved=True,
            ),
        )
    except RuntimeError as exc:
        if not _FP4_TE_ROPE_FALLBACK_WARNED:
            logger.warning("FP4 fused TE RoPE unavailable; falling back to TorchTitan RoPE: %s", exc)
            _FP4_TE_ROPE_FALLBACK_WARNED = True
        return apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)


def _attention_accepts_enable_gqa(module: nn.Module) -> bool:
    typ = type(module)
    cached = _ATTN_ENABLE_GQA_CACHE.get(typ)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(module.forward)
        cached = "enable_gqa" in sig.parameters
    except (TypeError, ValueError, AttributeError):
        cached = False
    _ATTN_ENABLE_GQA_CACHE[typ] = cached
    return cached


def _env_truthy_default(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sdpa_attention(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    enable_gqa: bool,
) -> torch.Tensor:
    backends = getattr(module, "sdpa_backends", None)
    kwargs = {
        "dropout_p": 0.0,
        "is_causal": True,
        "enable_gqa": enable_gqa,
    }
    if backends:
        with sdpa_kernel(backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


_sdpa_lifetime_watches: dict[
    tuple[int, str, int], tuple[torch.Tensor, torch.Tensor | None]
] = {}


def _sdpa_q_edge_sample(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 4:
        return torch.cat(
            (q[0, :, 0, :].reshape(-1), q[-1, :, -1, :].reshape(-1))
        )
    flat = q.reshape(-1)
    return torch.cat((flat[:4096], flat[-4096:]))


def _maybe_watch_sdpa_q_lifetime(q: torch.Tensor, debug_name: str) -> None:
    target = os.environ.get("USE_TK_DEBUG_SDPA_Q_LIFETIME_TARGET", "").strip()
    if not target or target not in debug_name:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    requested_rank = os.environ.get(
        "USE_TK_DEBUG_SDPA_Q_LIFETIME_RANK", "0"
    ).strip().lower()
    if requested_rank not in {"all", "*"} and rank != int(requested_rank):
        return
    call_counts = getattr(_maybe_watch_sdpa_q_lifetime, "_call_counts", None)
    if call_counts is None:
        call_counts = {}
        setattr(_maybe_watch_sdpa_q_lifetime, "_call_counts", call_counts)
    key = (rank, debug_name)
    call_index = call_counts.get(key, 0) + 1
    call_counts[key] = call_index
    requested_call = os.environ.get(
        "USE_TK_DEBUG_SDPA_Q_LIFETIME_CALL", "2"
    ).strip().lower()
    if requested_call not in {"all", "*"} and call_index != int(requested_call):
        return
    baseline = None
    if _env_truthy_default("USE_TK_DEBUG_SDPA_Q_LIFETIME_SNAPSHOT", False):
        baseline = _sdpa_q_edge_sample(q.detach()).clone()
    _sdpa_lifetime_watches[(rank, debug_name, call_index)] = (
        q.detach(),
        baseline,
    )
    logger.warning(
        "Watching SDPA Q lifetime %s call=%d ptr=%d shape=%s stride=%s",
        debug_name,
        call_index,
        q.data_ptr(),
        tuple(q.shape),
        tuple(q.stride()),
    )


def _maybe_probe_sdpa_q_lifetime(debug_name: str, stage: str) -> None:
    checkpoint = os.environ.get(
        "USE_TK_DEBUG_SDPA_Q_LIFETIME_CHECKPOINT", ""
    ).strip()
    if not checkpoint or checkpoint not in debug_name:
        return
    requested_stage = os.environ.get(
        "USE_TK_DEBUG_SDPA_Q_LIFETIME_CHECKPOINT_STAGE", "after_sdpa"
    ).strip()
    if requested_stage and requested_stage != stage:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    call_counts = getattr(_maybe_probe_sdpa_q_lifetime, "_call_counts", None)
    if call_counts is None:
        call_counts = {}
        setattr(_maybe_probe_sdpa_q_lifetime, "_call_counts", call_counts)
    key = (rank, debug_name, stage)
    call_index = call_counts.get(key, 0) + 1
    call_counts[key] = call_index
    requested_call = os.environ.get(
        "USE_TK_DEBUG_SDPA_Q_LIFETIME_CHECKPOINT_CALL", "2"
    ).strip().lower()
    if requested_call not in {"all", "*"} and call_index != int(requested_call):
        return
    for (watch_rank, watch_name, watch_call), (q, baseline) in tuple(
        _sdpa_lifetime_watches.items()
    ):
        if watch_rank != rank:
            continue
        sample = _sdpa_q_edge_sample(q)
        nonfinite_mask = ~torch.isfinite(sample)
        nonfinite_indices = torch.nonzero(
            nonfinite_mask, as_tuple=False
        ).flatten()
        nan_count = int(torch.isnan(sample).sum().item())
        inf_count = int(torch.isinf(sample).sum().item())
        finite_values = sample[~nonfinite_mask]
        max_abs = (
            float(finite_values.abs().max().item())
            if finite_values.numel()
            else float("nan")
        )
        first_indices = nonfinite_indices[:128].cpu().tolist()
        decoded_indices = []
        if q.dim() == 4:
            edge_size = q.shape[1] * q.shape[3]
            for sample_index in first_indices:
                edge = "first" if sample_index < edge_size else "last"
                edge_index = sample_index % edge_size
                decoded_indices.append(
                    (edge, edge_index // q.shape[3], edge_index % q.shape[3])
                )
        changed_count = None
        first_changed_indices = None
        first_changed_bit_pairs = None
        if baseline is not None:
            sample_bits = sample.view(torch.int16)
            baseline_bits = baseline.view(torch.int16)
            changed_indices = torch.nonzero(
                sample_bits != baseline_bits, as_tuple=False
            ).flatten()
            changed_count = int(changed_indices.numel())
            first_changed_indices = changed_indices[:128].cpu().tolist()
            before_bits = baseline_bits[changed_indices[:32]].cpu().tolist()
            after_bits = sample_bits[changed_indices[:32]].cpu().tolist()
            first_changed_bit_pairs = [
                (before & 0xFFFF, after & 0xFFFF)
                for before, after in zip(before_bits, after_bits, strict=True)
            ]
        logger.warning(
            "SDPA Q lifetime probe target=%s call=%d checkpoint=%s "
            "stage=%s call=%d "
            "ptr=%d sample_nan_count=%d sample_inf_count=%d/%d "
            "sample_max_finite_abs=%s first_nonfinite_indices=%s "
            "first_nonfinite_edge_head_channel=%s changed_count=%s "
            "first_changed_indices=%s first_changed_bf16_bits=%s",
            watch_name,
            watch_call,
            debug_name,
            stage,
            call_index,
            q.data_ptr(),
            nan_count,
            inf_count,
            sample.numel(),
            max_abs,
            first_indices,
            decoded_indices,
            changed_count,
            first_changed_indices,
            first_changed_bit_pairs,
        )


def _maybe_register_sdpa_replay_dump(
    *,
    output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    debug_name: str,
    backend_names: object,
) -> None:
    """Capture exact SDPA inputs and upstream gradient for offline replay."""
    dump_dir = os.environ.get("USE_TK_DEBUG_SDPA_REPLAY_DIR", "").strip()
    if not dump_dir or not output.requires_grad:
        return
    name_filter = os.environ.get("USE_TK_DEBUG_SDPA_REPLAY_FILTER", "").strip()
    if name_filter and name_filter not in debug_name:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    requested_rank_value = os.environ.get("USE_TK_DEBUG_SDPA_REPLAY_RANK", "0")
    if requested_rank_value.strip().lower() not in {"all", "*"} and rank != int(
        requested_rank_value
    ):
        return
    on_nonfinite = _env_truthy_default(
        "USE_TK_DEBUG_SDPA_REPLAY_ON_NONFINITE", False
    )
    snapshot_forward = _env_truthy_default(
        "USE_TK_DEBUG_SDPA_REPLAY_SNAPSHOT_FORWARD", False
    )
    snapshot_boundaries = _env_truthy_default(
        "USE_TK_DEBUG_SDPA_REPLAY_SNAPSHOT_BOUNDARIES", False
    )
    snapshot_after_forward = _env_truthy_default(
        "USE_TK_DEBUG_SDPA_REPLAY_SNAPSHOT_AFTER_FORWARD",
        snapshot_boundaries,
    )
    snapshot_before_backward = _env_truthy_default(
        "USE_TK_DEBUG_SDPA_REPLAY_SNAPSHOT_BEFORE_BACKWARD",
        snapshot_boundaries,
    )
    sample_only = _env_truthy_default(
        "USE_TK_DEBUG_SDPA_REPLAY_SAMPLE_ONLY", False
    )
    call_counts = getattr(_maybe_register_sdpa_replay_dump, "_call_counts", None)
    if call_counts is None:
        call_counts = {}
        setattr(_maybe_register_sdpa_replay_dump, "_call_counts", call_counts)
    call_key = (rank, debug_name)
    call_index = call_counts.get(call_key, 0) + 1
    call_counts[call_key] = call_index
    requested_call_value = os.environ.get(
        "USE_TK_DEBUG_SDPA_REPLAY_CALL", "1"
    ).strip().lower()
    if requested_call_value not in {"all", "*"}:
        requested_call = int(requested_call_value)
        if call_index != requested_call:
            return

    once_key = (rank, debug_name, call_index)
    dumped = getattr(_maybe_register_sdpa_replay_dump, "_dumped", None)
    if dumped is None:
        dumped = set()
        setattr(_maybe_register_sdpa_replay_dump, "_dumped", dumped)
    if once_key in dumped:
        return
    dumped.add(once_key)

    q_stride = tuple(q.stride())
    k_stride = tuple(k.stride())
    v_stride = tuple(v.stride())
    forward_versions = {
        name: int(tensor._version)
        for name, tensor in (("q", q), ("k", k), ("v", v))
    }

    def _edge_token_samples(tensor: torch.Tensor) -> torch.Tensor:
        # The observed corruption is channel-stationary across tokens. Sampling
        # the first and last token keeps this probe small enough not to alter
        # the full-model allocator layout like a 256 MiB Q clone does.
        if tensor.dim() == 4:
            first = tensor[0, :, 0, :].reshape(-1)
            last = tensor[-1, :, -1, :].reshape(-1)
            return torch.stack((first, last)).clone()
        flat = tensor.reshape(-1)
        return torch.stack((flat[:4096], flat[-4096:])).clone()

    def _sample_group(named_tensors) -> dict[str, torch.Tensor]:
        return {
            name: _edge_token_samples(tensor.detach())
            for name, tensor in named_tensors
        }

    def _samples_to_cpu(samples):
        if samples is None:
            return None
        return {
            name: tensor.detach().cpu()
            for name, tensor in samples.items()
        }

    forward_samples = (
        _sample_group((("q", q), ("k", k), ("v", v)))
        if snapshot_after_forward
        else None
    )
    forward_tensors = None
    if snapshot_forward:
        # The fused producer's output storage may be released or reused before
        # an output-gradient hook runs.  Take the diagnostic copy while the
        # forward values are still authoritative.
        forward_tensors = {
            name: tensor.detach().contiguous().cpu()
            for name, tensor in (("q", q), ("k", k), ("v", v))
        }
    forward_max_abs = (
        {
            name: float(tensor.float().abs().max().item())
            for name, tensor in forward_tensors.items()
        }
        if forward_tensors is not None
        else None
    )

    def _save(grad_output: torch.Tensor, grad_q: torch.Tensor | None = None) -> None:
        os.makedirs(dump_dir, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", debug_name)
        path = os.path.join(
            dump_dir, f"rank{rank}_call{call_index}_{safe_name}.pt"
        )
        postback_named_tensors = [
            ("q", q),
            ("k", k),
            ("v", v),
            ("grad_output", grad_output),
        ]
        if grad_q is not None:
            postback_named_tensors.append(("grad_q", grad_q))
        torch.save(
            {
                "debug_name": debug_name,
                "rank": rank,
                "call_index": call_index,
                "backend_names": [str(item) for item in (backend_names or [])],
                "forward_max_abs": forward_max_abs,
                "backward_entry_max_abs": {
                    name: float(tensor.detach().norm(float("inf")).item())
                    for name, tensor in (("q", q), ("k", k), ("v", v))
                },
                "forward_versions": forward_versions,
                "backward_entry_versions": {
                    name: int(tensor._version)
                    for name, tensor in (("q", q), ("k", k), ("v", v))
                },
                "boundary_samples": {
                    "after_forward": _samples_to_cpu(forward_samples),
                    "before_backward": _samples_to_cpu(
                        state.get("preback_samples")
                    ),
                    "after_backward": _samples_to_cpu(
                        _sample_group(postback_named_tensors)
                    ),
                },
                "q": None if sample_only else (
                    forward_tensors["q"]
                    if forward_tensors is not None
                    else q.detach().contiguous().cpu()
                ),
                "k": None if sample_only else (
                    forward_tensors["k"]
                    if forward_tensors is not None
                    else k.detach().contiguous().cpu()
                ),
                "v": None if sample_only else (
                    forward_tensors["v"]
                    if forward_tensors is not None
                    else v.detach().contiguous().cpu()
                ),
                "grad_output": (
                    None
                    if sample_only
                    else grad_output.detach().contiguous().cpu()
                ),
                "observed_grad_q": (
                    None
                    if sample_only or grad_q is None
                    else grad_q.detach().contiguous().cpu()
                ),
                "strides": {
                    "q": q_stride,
                    "k": k_stride,
                    "v": v_stride,
                    "grad_output": tuple(grad_output.stride()),
                },
            },
            path,
        )
        logger.warning("Saved exact SDPA replay tensors to %s", path)

    state = {"grad_output": None, "preback_samples": None}

    def _dump_hook(grad_output: torch.Tensor) -> torch.Tensor:
        if snapshot_before_backward:
            state["preback_samples"] = _sample_group(
                (
                    ("q", q),
                    ("k", k),
                    ("v", v),
                    ("grad_output", grad_output),
                )
            )
        if on_nonfinite:
            state["grad_output"] = grad_output
        else:
            _save(grad_output)
        return grad_output

    output.register_hook(_dump_hook)
    if on_nonfinite:
        def _q_grad_hook(grad_q: torch.Tensor) -> torch.Tensor:
            max_abs = float(grad_q.detach().norm(float("inf")).item())
            if not math.isfinite(max_abs):
                grad_output = state["grad_output"]
                if grad_output is None:
                    raise RuntimeError(
                        f"SDPA replay hook at {debug_name} did not observe grad_output"
                    )
                _save(grad_output, grad_q)
            state["grad_output"] = None
            return grad_q

        q.register_hook(_q_grad_hook)


def _maybe_check_sdpa_inputs_after_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    debug_name: str,
) -> None:
    """Bracket input corruption across the SDPA forward launch."""
    if not _env_truthy_default(
        "USE_TK_DEBUG_SDPA_INPUTS_AFTER_FORWARD", False
    ):
        return
    name_filter = os.environ.get(
        "USE_TK_DEBUG_SDPA_INPUTS_AFTER_FORWARD_FILTER", ""
    ).strip()
    if name_filter and name_filter not in debug_name:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    call_counts = getattr(
        _maybe_check_sdpa_inputs_after_forward, "_call_counts", None
    )
    if call_counts is None:
        call_counts = {}
        setattr(
            _maybe_check_sdpa_inputs_after_forward, "_call_counts", call_counts
        )
    key = (rank, debug_name)
    call_index = call_counts.get(key, 0) + 1
    call_counts[key] = call_index
    requested_call = os.environ.get(
        "USE_TK_DEBUG_SDPA_INPUTS_AFTER_FORWARD_CALL", "all"
    ).strip().lower()
    if requested_call not in {"all", "*"} and call_index != int(requested_call):
        return

    tensors = (("q", q), ("k", k), ("v", v))
    norms = torch._foreach_norm(
        [tensor.detach() for _, tensor in tensors], float("inf")
    )
    values = {name: float(norm.item()) for (name, _), norm in zip(tensors, norms)}
    logger.warning(
        "SDPA post-forward input check %s call=%d max_abs=%s",
        debug_name,
        call_index,
        values,
    )
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(
            f"SDPA forward mutated an input at {debug_name} call={call_index}: "
            f"{values}"
        )


def _maybe_register_sdpa_pre_backward_check(
    *,
    output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    debug_name: str,
) -> None:
    """Check saved SDPA inputs immediately before its backward executes."""
    if not _env_truthy_default(
        "USE_TK_DEBUG_SDPA_INPUTS_BEFORE_BACKWARD", False
    ):
        return
    name_filter = os.environ.get(
        "USE_TK_DEBUG_SDPA_INPUTS_BEFORE_BACKWARD_FILTER", ""
    ).strip()
    if name_filter and name_filter not in debug_name:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    call_counts = getattr(
        _maybe_register_sdpa_pre_backward_check, "_call_counts", None
    )
    if call_counts is None:
        call_counts = {}
        setattr(
            _maybe_register_sdpa_pre_backward_check,
            "_call_counts",
            call_counts,
        )
    key = (rank, debug_name)
    call_index = call_counts.get(key, 0) + 1
    call_counts[key] = call_index
    requested_call = os.environ.get(
        "USE_TK_DEBUG_SDPA_INPUTS_BEFORE_BACKWARD_CALL", "all"
    ).strip().lower()
    if requested_call not in {"all", "*"} and call_index != int(requested_call):
        return

    def _check(grad_output: torch.Tensor) -> torch.Tensor:
        tensors = (("q", q), ("k", k), ("v", v), ("dout", grad_output))
        norms = torch._foreach_norm(
            [tensor.detach() for _, tensor in tensors], float("inf")
        )
        values = {
            name: float(norm.item())
            for (name, _), norm in zip(tensors, norms)
        }
        logger.warning(
            "SDPA pre-backward input check %s call=%d max_abs=%s",
            debug_name,
            call_index,
            values,
        )
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(
                f"SDPA input corrupted before backward at {debug_name} "
                f"call={call_index}: {values}"
            )
        return grad_output

    output.register_hook(_check)


def _fp4_tk_strided_attention_input_mask(
    *,
    batch_size: int | None = None,
    seq_len: int | None = None,
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> frozenset[str]:
    # K/V strided plans are not interchangeable with the measured Q-only plan,
    # so defaults stay backend- and shape-scoped.
    value = os.environ.get("USE_FP4_TK_STRIDED_ATTN_INPUTS")
    if value is not None:
        mode = value.strip().lower()
        if mode in {"1", "true", "yes", "on"}:
            return frozenset({"q", "k", "v"})
        if mode in {"0", "false", "no", "off", ""}:
            return frozenset()
        mask = frozenset(part.strip() for part in mode.split(",") if part.strip())
        if not mask or not mask.issubset({"q", "k", "v"}):
            raise ValueError(
                "USE_FP4_TK_STRIDED_ATTN_INPUTS must be a boolean or a "
                "comma-separated subset of q,k,v"
            )
        return mask
    localcta_enabled = (
        os.environ.get('USE_TK_LOCALCTA', '0') == '1'
        and _is_localcta_v4()
        and os.environ.get("USE_NVFP4_MXFP4_LIVE_PATH", "0") != "0"
    )
    if localcta_enabled:
        return frozenset({"q", "k", "v"})

    v5_value = os.environ.get("USE_TK_V5_STRIDED_Q_ATTN")
    if v5_value is None:
        v5_enabled = True
    else:
        mode = v5_value.strip().lower()
        if mode in {"1", "true", "yes", "on"}:
            v5_enabled = True
        elif mode in {"0", "false", "no", "off", ""}:
            v5_enabled = False
        else:
            raise ValueError("USE_TK_V5_STRIDED_Q_ATTN must be a boolean")
    v5_locked_shape = (
        os.environ.get("USE_TK_GEMM", "0") == "1"
        and os.environ.get("USE_TK_LOCALCTA", "0") != "1"
        and os.environ.get("USE_MXFP4_TK_BACKEND", "0") != "1"
        and os.environ.get("FP4_ATTN_BACKEND", "").strip().lower() == "tk"
        and batch_size == 4
        and seq_len == 8192
        and n_heads == 32
        and n_kv_heads == 8
        and head_dim == 128
    )
    return frozenset({"q"}) if v5_enabled and v5_locked_shape else frozenset()


def _use_fp4_tk_strided_attention_inputs() -> bool:
    return bool(_fp4_tk_strided_attention_input_mask())


def _normalize_backend_override(value: str | None, *, selector: str) -> str | None:
    if value is None:
        return None
    backend = value.strip().lower()
    if backend not in _VALID_FP4_BACKENDS:
        raise ValueError(
            f"{selector} must be one of {sorted(_VALID_FP4_BACKENDS)}, got {value!r}"
        )
    return backend


def _default_fp4_backend() -> str:
    if os.environ.get('USE_TK_GEMM', '0') != '1':
        return 'te'
    if os.environ.get('USE_TK_LOCALCTA', '0') == '1':
        fused_requested = (
            os.environ.get('USE_TK_LOCALCTA_FUSED', os.environ.get('USE_TK_LOCALCTA_FUSED_SPLIT', '0')) == '1'
        )
        # The validated v4 full-model route is the plain localCTA backend. The
        # legacy localcta_fused QKV producer uses an incompatible scale contract.
        if fused_requested and not _is_localcta_v4():
            return 'localcta_fused'
        return 'localcta'
    return 'tk'


def _resolve_backend_split() -> tuple[str, str]:
    attn_backend = _normalize_backend_override(
        os.environ.get('FP4_ATTN_BACKEND'),
        selector='FP4_ATTN_BACKEND',
    ) or _default_fp4_backend()
    ffn_backend = _normalize_backend_override(
        os.environ.get('FP4_FFN_BACKEND'),
        selector='FP4_FFN_BACKEND',
    ) or _default_fp4_backend()

    if _is_localcta_v4():
        if attn_backend == 'localcta_fused':
            logger.warning(
                "localCTA v4 ignores attention backend localcta_fused; using localcta "
                "to avoid the fused QKV scale-contract mismatch."
            )
            attn_backend = 'localcta'
        if ffn_backend == 'localcta_fused':
            logger.warning(
                "localCTA v4 ignores FFN backend localcta_fused; using localcta "
                "to keep the validated v4 route."
            )
            ffn_backend = 'localcta'

    active_tk_backends = {b for b in (attn_backend, ffn_backend) if b != 'te'}
    if len(active_tk_backends) > 1:
        raise RuntimeError(
            "Mixed TK backend flavors across attention and FFN are not supported yet. "
            f"Got attention={attn_backend!r}, ffn={ffn_backend!r}."
        )

    tk_backend = next(iter(active_tk_backends), 'te')
    _maybe_enable_nvfp4_live_path(tk_backend)
    if tk_backend == 'te':
        os.environ['USE_TK_GEMM'] = '0'
        os.environ['USE_TK_LOCALCTA'] = '0'
        os.environ['USE_TK_LOCALCTA_FUSED'] = '0'
    elif tk_backend == 'tk':
        os.environ['USE_TK_GEMM'] = '1'
        os.environ['USE_TK_LOCALCTA'] = '0'
        os.environ['USE_TK_LOCALCTA_FUSED'] = '0'
        os.environ.setdefault('USE_TK_QUANT', '1')
    elif tk_backend == 'localcta':
        os.environ['USE_TK_GEMM'] = '1'
        os.environ['USE_TK_LOCALCTA'] = '1'
        os.environ['USE_TK_LOCALCTA_FUSED'] = '0'
        if _is_localcta_v4():
            apply_localcta_v4_profile_defaults()
    elif tk_backend == 'localcta_fused':
        os.environ['USE_TK_GEMM'] = '1'
        os.environ['USE_TK_LOCALCTA'] = '1'
        os.environ['USE_TK_LOCALCTA_FUSED'] = '1'
        if _is_localcta_v4():
            apply_localcta_v4_profile_defaults()

    return attn_backend, ffn_backend


def _call_with_optional_backend_mode(factory, *args, backend_mode: str, **kwargs):
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    if "backend_mode" in params:
        return factory(*args, backend_mode=backend_mode, **kwargs)
    return factory(*args, **kwargs)


class _NormIdentity(nn.Module):
    """Pass-through replacement for absorbed norms."""
    def __init__(self, dim, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)
    def forward(self, x):
        return x
    def reset_parameters(self):
        nn.init.ones_(self.weight)


class _FusedAttentionWrapper(nn.Module):
    """Wrapper that replaces Attention + attention_norm with FusedAttentionFP4.

    The FusedAttentionFP4 handles:
      - Absorbed attention_norm (RMSNorm)
      - Stacked QKV weights with single quant + single GEMM
      - Separate wo projection

    The wrapper handles:
      - RoPE (rotary position embedding)
      - Flash/flex attention computation
      - repeat_kv for GQA
    """

    def __init__(self, orig_attention, fused_attn_fp4):
        super().__init__()
        self.fused = fused_attn_fp4
        self.n_heads = orig_attention.n_heads
        self.n_kv_heads = getattr(orig_attention, 'n_kv_heads', self.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = orig_attention.head_dim
        self.use_flex_attn = orig_attention.use_flex_attn
        self.inner_attention = orig_attention.inner_attention

    def _forward_impl(
        self,
        x,
        freqs_cis,
        attention_masks,
        residual=None,
        h_gamma=None,
        input_h_carrier=None,
        input_cde_partial=None,
        cde_emit=False,
    ):
        """
        Args:
            x: (B, S, dim) bf16 — RAW input (pre-norm)
        """
        from torchtitan.models.llama3.model.model import repeat_kv
        from torch.nn.attention.flex_attention import BlockMask

        bs, seqlen, _ = x.shape
        debug_name = getattr(self.fused, '_lbt_debug_name', self.__class__.__name__)
        _maybe_probe_sdpa_q_lifetime(debug_name, "entry")
        _tk_stage_trace('attn_wrap', 'start', debug_name)
        debug_finite = use_tk_attn_debug_finite()

        def _record(name: str, tensor: torch.Tensor):
            if debug_finite:
                _attn_debug_check_finite(name, tensor)
            return tensor

        capture_filter = os.environ.get(
            "USE_TK_DEBUG_ATTN_CAPTURE_FILTER", ""
        ).strip()
        capture_attn = bool(_attn_capture_path()) and (
            not capture_filter or capture_filter in debug_name
        )
        capture_hooks = capture_attn and os.environ.get(
            "USE_TK_DEBUG_ATTN_CAPTURE_HOOKS", "0"
        ) == "1"
        hook_tags_value = os.environ.get(
            "USE_TK_DEBUG_ATTN_CAPTURE_HOOK_TAGS", ""
        ).strip()
        hook_tags = frozenset(
            tag.strip() for tag in hook_tags_value.split(",") if tag.strip()
        )

        def _capture_hook_enabled(tag: str) -> bool:
            return capture_hooks and (not hook_tags or tag in hook_tags)

        _capture_hook = None
        if capture_hooks:
            def _make_capture_hook(tag: str):
                def _hook(grad):
                    _append_attn_capture({
                        "event": "attn_core_backward",
                        "debug_name": debug_name,
                        "tensor": tag,
                        "stats": _tensor_capture_stats(grad),
                    })
                    return grad
                return _hook
            _capture_hook = _make_capture_hook

        # Fused QKV: rmsnorm + quant + single GEMM → split
        if os.environ.get('USE_TK_LOCALCTA_BACKEND_TRACE', '0') == '1':
            if freqs_cis is None:
                _tk_stage_trace('attn_wrap', 'freqs_none', debug_name)
            else:
                _trace_backend_choice(
                    'attn_wrap_freqs',
                    f"dtype={freqs_cis.dtype},cuda={freqs_cis.is_cuda},shape={tuple(freqs_cis.shape)}",
                )
        xq, xk, xv = self.fused.forward_qkv(
            x,
            freqs_cis=freqs_cis,
            h_carrier=input_h_carrier,
            cde_row_rms_partial=input_cde_partial,
        )
        _tk_stage_trace('attn_wrap', 'qkv_done', debug_name)
        _maybe_probe_sdpa_q_lifetime(debug_name, "after_qkv")

        # Reshape to (B, S, n_heads, head_dim)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        _record('attn_core.q_view', xq)
        _record('attn_core.k_view', xk)
        _record('attn_core.v_view', xv)

        xq_pre_rope = xq
        xk_pre_rope = xk
        xv_pre_repeat = xv
        if capture_hooks:
            if _capture_hook_enabled("xq_pre_rope") and xq_pre_rope.requires_grad:
                xq_pre_rope.register_hook(_capture_hook("xq_pre_rope"))
            if _capture_hook_enabled("xk_pre_rope") and xk_pre_rope.requires_grad:
                xk_pre_rope.register_hook(_capture_hook("xk_pre_rope"))
            if _capture_hook_enabled("xv_pre_repeat") and xv_pre_repeat.requires_grad:
                xv_pre_repeat.register_hook(_capture_hook("xv_pre_repeat"))

        # RoPE may already be fused into the Q/K GEMM epilogue.
        if not getattr(self.fused, "_last_qkv_rope_applied", False):
            xq, xk = _apply_rotary_emb_fast(xq, xk, freqs_cis=freqs_cis)
        _record('attn_core.q_rope', xq)
        _record('attn_core.k_rope', xk)
        if capture_hooks:
            if _capture_hook_enabled("xq_post_rope") and xq.requires_grad:
                xq.register_hook(_capture_hook("xq_post_rope"))
            if _capture_hook_enabled("xk_post_rope") and xk.requires_grad:
                xk.register_hook(_capture_hook("xk_post_rope"))

        accepts_enable_gqa = (
            not self.use_flex_attn
            and _attention_accepts_enable_gqa(self.inner_attention)
        )
        use_native_gqa = (
            _env_truthy_default("USE_FP4_TK_NATIVE_GQA", True)
            and not self.use_flex_attn
            and self.n_rep > 1
        )
        # GQA: prefer native grouped-query attention when the backend supports
        # it, rather than materializing repeated K/V heads in Python.
        if use_native_gqa:
            keys = xk
            values = xv
        else:
            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)
        _record('attn_core.keys', keys)
        _record('attn_core.values', values)
        if capture_hooks:
            if _capture_hook_enabled("k_post_repeat") and keys.requires_grad:
                keys.register_hook(_capture_hook("k_post_repeat"))
            if _capture_hook_enabled("v_post_repeat") and values.requires_grad:
                values.register_hook(_capture_hook("v_post_repeat"))

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)
        strided_attention_inputs = _fp4_tk_strided_attention_input_mask(
            batch_size=bs,
            seq_len=seqlen,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )
        if "q" not in strided_attention_inputs:
            xq = xq.contiguous()
        if "k" not in strided_attention_inputs:
            xk = xk.contiguous()
        if "v" not in strided_attention_inputs:
            xv = xv.contiguous()

        if capture_hooks:
            if _capture_hook_enabled("xq_attn_in") and xq.requires_grad:
                xq.register_hook(_capture_hook("xq_attn_in"))
            if _capture_hook_enabled("xk_attn_in") and xk.requires_grad:
                xk.register_hook(_capture_hook("xk_attn_in"))
            if _capture_hook_enabled("xv_attn_in") and xv.requires_grad:
                xv.register_hook(_capture_hook("xv_attn_in"))
        if _attn_layout_event_enabled("sdpa_inputs", debug_name):
            _append_attn_layout({
                "event": "sdpa_inputs",
                "debug_name": debug_name,
                **_tensor_layout_group((("q", xq), ("k", xk), ("v", xv))),
            })
        _record('attn_core.q_attn', xq)
        _record('attn_core.k_attn', xk)
        _record('attn_core.v_attn', xv)

        # Attention computation
        assert (
            isinstance(attention_masks, BlockMask) or attention_masks is None
        ), attention_masks

        if self.use_flex_attn:
            assert isinstance(attention_masks, BlockMask), attention_masks
            output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
        else:
            assert attention_masks is None
            if accepts_enable_gqa:
                output = self.inner_attention(xq, xk, xv, enable_gqa=use_native_gqa)
            elif use_native_gqa:
                output = _sdpa_attention(self.inner_attention, xq, xk, xv, enable_gqa=True)
            else:
                output = self.inner_attention(xq, xk, xv)
        _maybe_watch_sdpa_q_lifetime(xq, debug_name)
        _maybe_probe_sdpa_q_lifetime(debug_name, "after_sdpa")
        _maybe_check_sdpa_inputs_after_forward(
            q=xq,
            k=xk,
            v=xv,
            debug_name=debug_name,
        )
        _maybe_register_sdpa_pre_backward_check(
            output=output,
            q=xq,
            k=xk,
            v=xv,
            debug_name=debug_name,
        )
        _maybe_register_sdpa_replay_dump(
            output=output,
            q=xq,
            k=xk,
            v=xv,
            debug_name=debug_name,
            backend_names=getattr(self.inner_attention, "sdpa_backends", None),
        )
        _tk_stage_trace('attn_wrap', 'inner_done', debug_name)
        _record('attn_core.inner_output', output)
        if _attn_layout_event_enabled("sdpa_output", debug_name):
            _append_attn_layout({
                "event": "sdpa_output",
                "debug_name": debug_name,
                **_tensor_layout_group((("output", output),)),
            })

        if (
            _capture_hook_enabled("attn_output")
            and output.requires_grad
        ):
            output.register_hook(_capture_hook("attn_output"))

        if use_tk_localcta_v4_wo_attn_layout():
            # Direct WO quant consumes flash-attention's [B,H,S,D] layout and
            # avoids materializing the intermediate [B,S,H,D] activation.
            out = self.fused.forward_wo(
                output, residual=residual, h_gamma=h_gamma, cde_emit=cde_emit
            )
        else:
            # Reshape back: (B, S, n_heads*head_dim)
            output = output.transpose(1, 2).contiguous()
            _record('attn_core.output_bshd', output)
            output = output.view(bs, seqlen, -1)
            _record('attn_core.pre_wo', output)
            if _attn_layout_event_enabled("wo_input", debug_name):
                _append_attn_layout({
                    "event": "wo_input",
                    "debug_name": debug_name,
                    **_tensor_layout_group((("input", output),)),
                })

            # Output projection
            out = self.fused.forward_wo(
                output, residual=residual, h_gamma=h_gamma, cde_emit=cde_emit
            )
        if capture_attn:
            _append_attn_capture({
                "event": "attn_forward_io",
                "debug_name": debug_name,
                "stats": {
                    "attn_input": _tensor_capture_stats(x),
                    "xq_pre_rope": _tensor_capture_stats(xq_pre_rope),
                    "xk_pre_rope": _tensor_capture_stats(xk_pre_rope),
                    "xv_pre_repeat": _tensor_capture_stats(xv_pre_repeat),
                    "xq_post_rope": _tensor_capture_stats(xq.transpose(1, 2)),
                    "xk_post_repeat": _tensor_capture_stats(xk.transpose(1, 2)),
                    "xv_post_repeat": _tensor_capture_stats(xv.transpose(1, 2)),
                    "xq": _tensor_capture_stats(xq),
                    "xk": _tensor_capture_stats(xk),
                    "xv": _tensor_capture_stats(xv),
                    "attn_core_output": _tensor_capture_stats(output),
                    "wo_output": _tensor_capture_stats(
                        out[0] if isinstance(out, tuple) else out
                    ),
                },
            })
        _tk_stage_trace('attn_wrap', 'end', debug_name)
        return out

    def forward(self, x, freqs_cis, attention_masks):
        return self._forward_impl(x, freqs_cis, attention_masks)

    def forward_with_residual(
        self,
        x,
        freqs_cis,
        attention_masks,
        residual,
        h_gamma=None,
        input_h_carrier=None,
        input_cde_partial=None,
        cde_emit=False,
    ):
        return self._forward_impl(
            x,
            freqs_cis,
            attention_masks,
            residual=residual,
            h_gamma=h_gamma,
            input_h_carrier=input_h_carrier,
            input_cde_partial=input_cde_partial,
            cde_emit=cde_emit,
        )

    def forward_with_cde_partial(
        self,
        x,
        freqs_cis,
        attention_masks,
        input_cde_partial=None,
    ):
        return self._forward_impl(
            x,
            freqs_cis,
            attention_masks,
            input_cde_partial=input_cde_partial,
        )

    def init_weights(self, init_std: float):
        self.fused.init_weights(init_std)


def _ffn_residual_fused_block_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attention_masks):
    use_h = os.environ.get("USE_FP4_CODA_H_TILE_RMS", "0") == "1"
    use_exact_cde = os.environ.get("USE_FP4_CODA_EXACT_CDE", "0") == "1"
    use_exact_wo = os.environ.get("USE_FP4_CODA_EXACT_CDE_WO", "0") == "1"
    if use_h and (use_exact_cde or use_exact_wo):
        raise RuntimeError("exact C/D/E and H tile carriers are mutually exclusive")
    if use_exact_cde or use_exact_wo:
        attention_method = (
            "forward_with_residual" if use_exact_wo
            else "forward_with_cde_partial"
        )
        if not (
            hasattr(self.attention, attention_method)
            and hasattr(self.feed_forward, "forward_with_residual")
        ):
            raise RuntimeError("exact C/D/E requires fused attention and FFN owners")
        if isinstance(x, tuple):
            if len(x) != 2:
                raise RuntimeError(
                    "exact C/D/E expected a (residual, row_rms_partial) carrier"
                )
            residual, input_cde_partial = x
        else:
            residual = x
            input_cde_partial = None
        if use_exact_wo:
            h, wo_row_rms_partial = self.attention.forward_with_residual(
                self.attention_norm(residual),
                freqs_cis,
                attention_masks,
                residual=residual,
                input_cde_partial=input_cde_partial,
                cde_emit=True,
            )
        else:
            h = residual + self.attention.forward_with_cde_partial(
                self.attention_norm(residual),
                freqs_cis,
                attention_masks,
                input_cde_partial=input_cde_partial,
            )
            wo_row_rms_partial = None
        return self.feed_forward.forward_with_residual(
            self.ffn_norm(h),
            residual=h,
            cde_row_rms_partial=wo_row_rms_partial,
            cde_emit=(
                use_exact_cde
                and bool(getattr(self, "_fp4_cde_has_next", False))
            ),
        )
    if (
        use_h
        and hasattr(self.attention, "forward_with_residual")
        and hasattr(self.feed_forward, "forward_with_h_carrier")
    ):
        owner_ref = getattr(self, "_fp4_coda_h_next_attention_owner", None)
        owner = owner_ref() if owner_ref is not None else None
        next_attention_gamma = getattr(owner, "norm_weight", None)
        input_carrier = x if isinstance(x, tuple) else None
        residual = input_carrier[0] if input_carrier is not None else x
        attention_input = residual if input_carrier is not None else self.attention_norm(x)
        carrier = self.attention.forward_with_residual(
            attention_input,
            freqs_cis,
            attention_masks,
            residual=residual,
            h_gamma=self.feed_forward.norm_weight,
            input_h_carrier=input_carrier,
        )
        return self.feed_forward.forward_with_h_carrier(
            carrier, next_attention_gamma=next_attention_gamma
        )
    h = x + self.attention(self.attention_norm(x), freqs_cis, attention_masks)
    if (
        (use_tk_localcta_v4_ffn_residual_epilogue()
         or use_tk_v5_ffn_residual_epilogue())
        and hasattr(self.feed_forward, "forward_with_residual")
    ):
        return self.feed_forward.forward_with_residual(self.ffn_norm(h), residual=h)
    return h + self.feed_forward(self.ffn_norm(h))


def _nemotron_h_mlp_residual_fused_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    residual = hidden_states
    hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
    if getattr(self, "residual_in_fp32", False):
        residual = residual.to(torch.float32)
    return self.mixer.forward_with_residual(hidden_states, residual=residual)


def _nemotron_h_mamba_cde_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    if isinstance(hidden_states, tuple):
        if len(hidden_states) != 2:
            raise RuntimeError(
                "Nemotron Mamba CDE consumer expected "
                "(residual, row_rms_partial)"
            )
        residual, row_rms_partial = hidden_states
    else:
        residual = hidden_states
        row_rms_partial = None
    hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
    if getattr(self, "residual_in_fp32", False):
        raise RuntimeError("Nemotron Mamba CDE does not support FP32 residuals")
    return self.mixer(
        hidden_states,
        cache_params=cache_params,
        cache_position=cache_position,
        attention_mask=attention_mask,
        residual=residual,
        cde_row_rms_partial=row_rms_partial,
        cde_emit=bool(getattr(self, "_fp4_nemotron_cde_emit", False)),
    )


def _nemotron_h_mlp_cde_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    del cache_params, cache_position, attention_mask
    if isinstance(hidden_states, tuple):
        if len(hidden_states) != 2:
            raise RuntimeError(
                "Nemotron square-ReLU CDE consumer expected "
                "(residual, row_rms_partial)"
            )
        residual, row_rms_partial = hidden_states
    else:
        residual = hidden_states
        row_rms_partial = None
    hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
    if getattr(self, "residual_in_fp32", False):
        raise RuntimeError("Nemotron square-ReLU CDE does not support FP32 residuals")
    return self.mixer.forward_with_residual(
        hidden_states,
        residual=residual,
        cde_row_rms_partial=row_rms_partial,
        cde_emit=bool(getattr(self, "_fp4_nemotron_cde_emit", False)),
    )


def _nemotron_h_attention_cde_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    if isinstance(hidden_states, tuple):
        if len(hidden_states) != 2:
            raise RuntimeError(
                "Nemotron attention CDE consumer expected "
                "(residual, row_rms_partial)"
            )
        residual, row_rms_partial = hidden_states
    else:
        residual = hidden_states
        row_rms_partial = None
    hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
    if getattr(self, "residual_in_fp32", False):
        raise RuntimeError("Nemotron attention CDE does not support FP32 residuals")
    output = self.mixer(
        hidden_states,
        cache_params=cache_params,
        cache_position=cache_position,
        attention_mask=attention_mask,
        residual=residual,
        cde_row_rms_partial=row_rms_partial,
        cde_emit=bool(getattr(self, "_fp4_nemotron_cde_emit", False)),
    )
    return output[0]


def _use_simple_sqrelu_residual_epilogue() -> bool:
    return os.environ.get("USE_FP4_SIMPLE_SQRELU_W2_RESIDUAL_EPILOGUE", "0") == "1"


class FusedSquaredReLUFeedForwardFP4(nn.Module):
    """Two-linear squared-ReLU FFN for paper-style 1.2B models."""

    def __init__(self, dim, hidden_dim, norm_eps=1e-5, bias=False, device=None, dtype=torch.bfloat16, recipe=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epsilon = norm_eps
        self.norm_weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.w1 = SimpleFP4Linear(dim, hidden_dim, bias=bias, device=device, dtype=dtype)
        self.w2 = SimpleFP4Linear(hidden_dim, dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_residual(x, residual=None)

    def forward_with_residual(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        is_3d = x.dim() == 3
        if is_3d:
            B, S, H = x.shape
            x_2d = x.reshape(B * S, H)
            residual_2d = residual.reshape(B * S, self.dim) if residual is not None else None
        else:
            x_2d = x
            residual_2d = residual

        debug_name = getattr(self, '_lbt_debug_name', self.__class__.__name__)
        trace = use_tk_stage_trace()
        if trace:
            self.w1._lbt_debug_name = f"{debug_name}:w1"
            self.w2._lbt_debug_name = f"{debug_name}:w2"
            _tk_stage_trace('ffn_sqrelu_simple_fwd', 'start', debug_name)
            _tk_stage_trace('ffn_sqrelu_simple_fwd', 'rmsnorm_start', debug_name)
        normed, _ = _get_te_fused().fused_rmsnorm_only(
            _as_contiguous_bf16(x_2d),
            _as_contiguous_bf16(self.norm_weight),
            float(self.epsilon),
        )
        if trace:
            _tk_stage_trace('ffn_sqrelu_simple_fwd', 'rmsnorm_done', debug_name)
        hidden = self.w1(normed)
        if trace:
            _tk_stage_trace('ffn_sqrelu_simple_fwd', 'sqrelu_start', debug_name)
        hidden = sqrelu(hidden)
        if trace:
            _tk_stage_trace('ffn_sqrelu_simple_fwd', 'sqrelu_done', debug_name)
        if residual_2d is not None and _use_simple_sqrelu_residual_epilogue():
            out = self.w2(hidden, residual=residual_2d)
        else:
            out = self.w2(hidden)
            if residual_2d is not None:
                if trace:
                    _tk_stage_trace('ffn_sqrelu_simple_fwd', 'residual_start', debug_name)
                out = out + residual_2d
                if trace:
                    _tk_stage_trace('ffn_sqrelu_simple_fwd', 'residual_done', debug_name)
        if is_3d:
            out = out.view(B, S, self.dim)
        if trace:
            _tk_stage_trace('ffn_sqrelu_simple_fwd', 'end', debug_name)
        return out

    def invalidate_weight_cache(self):
        pass

    def init_weights(self, init_std: float = 0.02):
        nn.init.ones_(self.norm_weight)
        _safe_trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.w2.weight, mean=0.0, std=init_std)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    @classmethod
    def from_unfused(cls, ffn, norm, recipe=None):
        up_proj, down_proj = _sqrelu_ffn_projection_pair(ffn)
        fused = cls(
            dim=up_proj.in_features,
            hidden_dim=up_proj.out_features,
            norm_eps=getattr(norm, "eps", 1e-5),
            bias=getattr(up_proj, "bias", None) is not None,
            device=up_proj.weight.device,
            dtype=up_proj.weight.dtype,
            recipe=recipe,
        )
        if up_proj.weight.device.type != 'meta':
            with torch.no_grad():
                fused.w1.weight.copy_(up_proj.weight)
                fused.w2.weight.copy_(down_proj.weight)
                if up_proj.bias is not None and fused.w1.bias is not None:
                    fused.w1.bias.copy_(up_proj.bias)
                if down_proj.bias is not None and fused.w2.bias is not None:
                    fused.w2.bias.copy_(down_proj.bias)
                if hasattr(norm, 'weight') and norm.weight is not None:
                    fused.norm_weight.copy_(norm.weight)
        return fused


# Keywords matching output/lm_head layers
_HEAD_KEYWORDS = ["output", "lm_head"]


def _is_output_head_name(name: str) -> bool:
    return any(k in name for k in _HEAD_KEYWORDS)


def _use_fp4_convert_output_head() -> bool:
    return os.environ.get("USE_FP4_CONVERT_OUTPUT_HEAD", "0") == "1"


def _tail_bf16_linear_names(model: nn.Module) -> set[str]:
    count = int(os.environ.get("FP4_KEEP_TAIL_BF16_LINEAR_COUNT", "0") or 0)
    if count <= 0:
        return set()
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        or (_is_output_head_name(name) and isinstance(getattr(module, "weight", None), nn.Parameter))
    ]
    tail = set(names[-count:])
    logger.info("FP4 tail BF16 ablation: keeping last %d linear modules BF16: %s", count, sorted(tail))
    return tail


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def _layer_index_from_name(name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(name)
    return int(match.group(1)) if match else None


def _is_regular_transformer_block(module: nn.Module) -> bool:
    return hasattr(module, "attention") and hasattr(module, "feed_forward")


def _is_nemotron_h_block(module: nn.Module) -> bool:
    return hasattr(module, "mixer") and hasattr(module, "norm") and hasattr(module, "block_type")


def _is_nemotron_h_mlp_block(module: nn.Module) -> bool:
    mixer = getattr(module, "mixer", None)
    return (
        getattr(module, "block_type", None) == "mlp"
        and mixer is not None
        and hasattr(mixer, "up_proj")
        and isinstance(mixer.up_proj, nn.Linear)
        and hasattr(mixer, "down_proj")
        and isinstance(mixer.down_proj, nn.Linear)
    )


def _is_transformer_layer_module(module: nn.Module) -> bool:
    return _is_regular_transformer_block(module) or _is_nemotron_h_block(module)


def _is_ffn_layer_module(module: nn.Module) -> bool:
    return (
        (hasattr(module, "feed_forward") and hasattr(module, "ffn_norm"))
        or _is_nemotron_h_mlp_block(module)
    )


def _sqrelu_ffn_projection_pair(ffn: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    if hasattr(ffn, "w1") and isinstance(ffn.w1, nn.Linear) and hasattr(ffn, "w2"):
        return ffn.w1, ffn.w2
    if (
        hasattr(ffn, "up_proj")
        and isinstance(ffn.up_proj, nn.Linear)
        and hasattr(ffn, "down_proj")
    ):
        return ffn.up_proj, ffn.down_proj
    raise AttributeError("square-ReLU FFN fusion expects w1/w2 or up_proj/down_proj")


def _last_bf16_layer_indices(model: nn.Module) -> set[int]:
    count = int(os.environ.get("FP4_KEEP_LAST_N_LAYERS_BF16", "0") or 0)
    if count <= 0:
        return set()
    indices = []
    for name, module in model.named_modules():
        if not _is_transformer_layer_module(module):
            continue
        idx = _layer_index_from_name(name)
        if idx is not None:
            indices.append(idx)
    if not indices:
        logger.warning("FP4_KEEP_LAST_N_LAYERS_BF16=%d set, but no transformer layer indices were found", count)
        return set()
    last = max(indices)
    selected = set(range(max(0, last - count + 1), last + 1))
    logger.info("FP4 final-layer BF16 ablation: keeping transformer layer indices BF16: %s", sorted(selected))
    return selected


def _last_bf16_ffn_layer_indices(model: nn.Module) -> set[int]:
    count = int(os.environ.get("FP4_KEEP_LAST_N_FFNS_BF16", "0") or 0)
    if count <= 0:
        return set()
    indices = []
    for name, module in model.named_modules():
        if not _is_ffn_layer_module(module):
            continue
        idx = _layer_index_from_name(name)
        if idx is not None:
            indices.append(idx)
    if not indices:
        logger.warning("FP4_KEEP_LAST_N_FFNS_BF16=%d set, but no FFN layer indices were found", count)
        return set()
    last = max(indices)
    selected = set(range(max(0, last - count + 1), last + 1))
    logger.info("FP4 final-FFN BF16 ablation: keeping FFN layer indices BF16: %s", sorted(selected))
    return selected


class FP4Converter(ModelConverter):
    """Converts Llama model to FP4:

    - Attention + attention_norm → FusedAttentionFP4 (stacked QKV, absorbed norm)
    - FFN + ffn_norm → FusedFeedForwardFP4 (absorbed norm + BoundRecipeLinear w1/w2/w3)
    - Output head → Float32Linear
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config

    def convert(self, model: nn.Module):
        attn_backend, ffn_backend = _resolve_backend_split()

        _FusedAttentionFP4 = FusedAttentionFP4_TE if attn_backend == 'te' else FusedAttentionFP4_TK
        _FusedFeedForwardFP4 = FusedFeedForwardFP4_TE if ffn_backend == 'te' else FusedFeedForwardFP4_TK

        logger.info(
            "FP4Converter: attention backend=%s (%s), ffn backend=%s (%s)",
            attn_backend, _FusedAttentionFP4.__name__,
            ffn_backend, _FusedFeedForwardFP4.__name__,
        )

        recipe = NVFP4BlockScaling()
        attn_count = 0
        ffn_count = 0
        ffn_impl_counts = {}
        sqrelu_request_ignored_count = 0
        head_count = 0
        norm_identity_count = 0
        tail_bf16_names = _tail_bf16_linear_names(model)
        final_bf16_layer_indices = _last_bf16_layer_indices(model)
        final_bf16_ffn_layer_indices = _last_bf16_ffn_layer_indices(model)

        # ── Pass 1: swap output/lm_head ──
        if _use_fp4_convert_output_head() and not nemotron_h_fp4_output_head_enabled(False):
            modules_to_replace = []
            for name, module in model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                is_head = any(k in name for k in _HEAD_KEYWORDS)
                if is_head and name not in tail_bf16_names:
                    modules_to_replace.append((name, module))

            for name, module in modules_to_replace:
                device = module.weight.device
                dtype = module.weight.dtype
                new_layer = Float32Linear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                )
                new_layer = new_layer.to(device).to(dtype)
                if device.type != 'meta':
                    with torch.no_grad():
                        new_layer.weight.copy_(module.weight)
                        if module.bias is not None:
                            new_layer.bias.copy_(module.bias)
                rsetattr(model, name, new_layer)
                head_count += 1
                logger.info(f"  HEAD: {name}")

        # ── Pass 2: replace attention + attention_norm → FusedAttentionFP4 ──
        # Absorbs attention_norm, stacks QKV weights, routes through TK-enabled
        # _FusedQKVFunction + _WoFunction instead of TE generic_gemm.
        blocks_to_fuse_attn = []
        for block_name, block in model.named_modules():
            if hasattr(block, 'attention') and hasattr(block, 'attention_norm'):
                attn = block.attention
                norm = block.attention_norm
                # Only fuse if attention has unfused wq/wk/wv/wo (nn.Linear)
                if hasattr(attn, 'wq') and isinstance(attn.wq, nn.Linear):
                    layer_idx = _layer_index_from_name(block_name)
                    if layer_idx in final_bf16_layer_indices:
                        logger.info("  KEEP BF16 ATTN: %s.attention final-layer ablation", block_name)
                        continue
                    qkv_tail = {
                        f"{block_name}.attention.wq",
                        f"{block_name}.attention.wk",
                        f"{block_name}.attention.wv",
                    } & tail_bf16_names
                    if qkv_tail:
                        logger.info("  KEEP BF16 ATTN: %s.attention tail=%s", block_name, sorted(qkv_tail))
                        continue
                    wo_bf16 = f"{block_name}.attention.wo" in tail_bf16_names
                    blocks_to_fuse_attn.append((block_name, block, attn, norm, wo_bf16))

        for block_name, block, attn, norm, wo_bf16 in blocks_to_fuse_attn:
            fused_attn = _call_with_optional_backend_mode(
                _FusedAttentionFP4.from_attention,
                attn,
                norm,
                backend_mode=attn_backend,
            )
            if wo_bf16:
                fused_attn._force_wo_bf16 = True
            fused_attn._lbt_debug_name = f"{block_name}.attention"
            wrapper = _FusedAttentionWrapper(attn, fused_attn)
            block.attention = wrapper
            attn_count += 1
            logger.info(
                f"  ATTN: {block_name}.attention → {_FusedAttentionFP4.__name__}"
                f"{' (BF16 wo)' if wo_bf16 else ''}"
            )

            # Replace attention_norm with identity (absorbed into fused)
            dim = norm.weight.shape[0] if hasattr(norm, 'weight') else 0
            norm_dtype = norm.weight.dtype if hasattr(norm, 'weight') else torch.bfloat16
            block.attention_norm = _NormIdentity(dim, dtype=norm_dtype)
            norm_identity_count += 1
            logger.info(f"  IDENTITY: {block_name}.attention_norm")

        if use_nemotron_h_fused_attention():
            blocks_to_fuse_nemotron_attn = []
            for block_name, block in model.named_modules():
                if not is_nemotron_h_attention_block(block):
                    continue
                layer_idx = _layer_index_from_name(block_name)
                if layer_idx in final_bf16_layer_indices:
                    logger.info("  KEEP BF16 NEMOTRON-H ATTN: %s.mixer final-layer ablation", block_name)
                    continue
                qkv_tail = {
                    f"{block_name}.mixer.q_proj",
                    f"{block_name}.mixer.k_proj",
                    f"{block_name}.mixer.v_proj",
                } & tail_bf16_names
                if qkv_tail:
                    logger.info("  KEEP BF16 NEMOTRON-H ATTN: %s.mixer tail=%s", block_name, sorted(qkv_tail))
                    continue
                wo_bf16 = f"{block_name}.mixer.o_proj" in tail_bf16_names
                blocks_to_fuse_nemotron_attn.append((block_name, block, block.mixer, block.norm, wo_bf16))

            for block_name, block, attn, norm, wo_bf16 in blocks_to_fuse_nemotron_attn:
                fused_attn = _call_with_optional_backend_mode(
                    _FusedAttentionFP4.from_attention,
                    attn,
                    norm,
                    backend_mode=attn_backend,
                )
                if wo_bf16:
                    fused_attn._force_wo_bf16 = True
                fused_attn._lbt_debug_name = f"{block_name}.mixer"
                block.mixer = NemotronHFusedAttentionWrapper(
                    attn,
                    fused_attn,
                    use_direct_wo_layout=(
                        use_tk_localcta_v4_wo_attn_layout
                        if attn_backend != "te"
                        else (lambda: False)
                    ),
                )
                attn_count += 1
                logger.info(
                    f"  NEMOTRON-H ATTN: {block_name}.mixer -> {_FusedAttentionFP4.__name__}"
                    f"{' (BF16 o_proj)' if wo_bf16 else ''}"
                )
                dim = norm.weight.shape[0] if hasattr(norm, 'weight') else 0
                norm_dtype = norm.weight.dtype if hasattr(norm, 'weight') else torch.bfloat16
                block.norm = _NormIdentity(dim, dtype=norm_dtype)
                norm_identity_count += 1
                logger.info(f"  IDENTITY: {block_name}.norm")

        # ── Pass 3: replace FFN modules (feed_forward + ffn_norm) ──
        blocks_to_fuse = []
        for block_name, block in model.named_modules():
            if hasattr(block, 'feed_forward') and hasattr(block, 'ffn_norm'):
                ffn = block.feed_forward
                norm = block.ffn_norm
                # Only fuse if feed_forward has unfused w1/w2/w3 (nn.Linear)
                if (hasattr(ffn, 'w1') and isinstance(ffn.w1, nn.Linear)):
                    layer_idx = _layer_index_from_name(block_name)
                    if layer_idx in final_bf16_layer_indices:
                        logger.info("  KEEP BF16 FFN: %s.feed_forward final-layer ablation", block_name)
                        continue
                    if layer_idx in final_bf16_ffn_layer_indices:
                        logger.info("  KEEP BF16 FFN: %s.feed_forward final-FFN ablation", block_name)
                        continue
                    ffn_tail = {
                        f"{block_name}.feed_forward.w1",
                        f"{block_name}.feed_forward.w2",
                        f"{block_name}.feed_forward.w3",
                    } & tail_bf16_names
                    if ffn_tail:
                        logger.info("  KEEP BF16 FFN: %s.feed_forward tail=%s", block_name, sorted(ffn_tail))
                        continue
                    blocks_to_fuse.append(
                        (block_name, block, ffn, norm, "feed_forward", "ffn_norm", f"{block_name}.feed_forward")
                    )

        for block_name, block in model.named_modules():
            if not _is_nemotron_h_mlp_block(block):
                continue
            layer_idx = _layer_index_from_name(block_name)
            if layer_idx in final_bf16_layer_indices:
                logger.info("  KEEP BF16 NEMOTRON-H MLP: %s.mixer final-layer ablation", block_name)
                continue
            if layer_idx in final_bf16_ffn_layer_indices:
                logger.info("  KEEP BF16 NEMOTRON-H MLP: %s.mixer final-FFN ablation", block_name)
                continue
            ffn_tail = {
                f"{block_name}.mixer.up_proj",
                f"{block_name}.mixer.down_proj",
            } & tail_bf16_names
            if ffn_tail:
                logger.info("  KEEP BF16 NEMOTRON-H MLP: %s.mixer tail=%s", block_name, sorted(ffn_tail))
                continue
            blocks_to_fuse.append((block_name, block, block.mixer, block.norm, "mixer", "norm", f"{block_name}.mixer"))

        for block_name, block, ffn, norm, ffn_attr, norm_attr, debug_name in blocks_to_fuse:
            use_sqrelu_tk = os.environ.get("USE_FP4_SQRELU_FFN_TK")
            if use_sqrelu_tk is None:
                use_sqrelu_tk = os.environ.get("USE_FP4_EXPERIMENTAL_SQRELU_FFN_TK", "0")
            if hasattr(ffn, 'w3'):
                ffn_cls = _FusedFeedForwardFP4
                ffn_kind = "SwiGLU/SiLU"
                if use_sqrelu_tk == "1":
                    sqrelu_request_ignored_count += 1
            elif ffn_backend != 'te' and use_sqrelu_tk == "1":
                ffn_cls = FusedSquaredReLUFeedForwardFP4_TK
                ffn_kind = "Square-ReLU"
            else:
                ffn_cls = FusedSquaredReLUFeedForwardFP4
                ffn_kind = "Square-ReLU"
            fused_ffn = _call_with_optional_backend_mode(
                ffn_cls.from_unfused,
                ffn,
                norm,
                recipe=recipe,
                backend_mode=ffn_backend,
            )
            fused_ffn._lbt_debug_name = debug_name
            setattr(block, ffn_attr, fused_ffn)
            ffn_count += 1
            ffn_impl_key = (ffn_kind, ffn_cls.__name__)
            ffn_impl_counts[ffn_impl_key] = ffn_impl_counts.get(ffn_impl_key, 0) + 1
            logger.info(
                f"  FFN: {debug_name} -> {ffn_cls.__name__} "
                f"({ffn_kind}, norm absorbed)"
            )

            # Replace ffn_norm with identity
            dim = norm.weight.shape[0] if hasattr(norm, 'weight') else 0
            norm_dtype = norm.weight.dtype if hasattr(norm, 'weight') else torch.bfloat16
            setattr(block, norm_attr, _NormIdentity(dim, dtype=norm_dtype))
            norm_identity_count += 1
            logger.info(f"  IDENTITY: {block_name}.{norm_attr}")

        fused_mamba_rms_count = 0
        if (
            attn_backend != "te"
            and _env_truthy_default(
                "LBT_NEMOTRON_H_FUSED_MAMBA_RMS_IN_PROJ", False
            )
            and _env_truthy_default(
                "LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ", True
            )
        ):
            fused_mamba_rms_projections = []
            for block_name, block in model.named_modules():
                if getattr(block, "block_type", None) != "mamba":
                    continue
                layer_idx = _layer_index_from_name(block_name)
                if layer_idx in final_bf16_layer_indices:
                    continue
                in_proj = getattr(getattr(block, "mixer", None), "in_proj", None)
                norm = getattr(block, "norm", None)
                projection_name = f"{block_name}.mixer.in_proj"
                if (
                    not isinstance(in_proj, nn.Linear)
                    or in_proj.bias is not None
                    or in_proj.in_features != 4096
                    or in_proj.out_features != 18560
                    or projection_name in tail_bf16_names
                    or not isinstance(getattr(norm, "weight", None), nn.Parameter)
                ):
                    continue
                fused_mamba_rms_projections.append(
                    (block_name, block, in_proj, norm, projection_name)
                )

            for (
                block_name,
                block,
                in_proj,
                norm,
                projection_name,
            ) in fused_mamba_rms_projections:
                fused = NVFP4RMSNormLinearTK(
                    in_proj.in_features,
                    in_proj.out_features,
                    eps=float(
                        getattr(
                            norm,
                            "variance_epsilon",
                            getattr(norm, "eps", 1e-5),
                        )
                    ),
                    device=in_proj.weight.device,
                    dtype=in_proj.weight.dtype,
                )
                if in_proj.weight.device.type != "meta":
                    with torch.no_grad():
                        fused.weight.copy_(in_proj.weight)
                        fused.norm_weight.copy_(norm.weight)
                fused._lbt_debug_name = projection_name
                block.mixer.in_proj = fused
                block.norm = _NormIdentity(
                    in_proj.in_features,
                    dtype=norm.weight.dtype,
                )
                fused_mamba_rms_count += 1
                norm_identity_count += 1
                logger.info(
                    "  NVFP4 FUSED NEMOTRON-H MAMBA RMS+IN: %s -> %s",
                    projection_name,
                    attn_backend,
                )

        projection_counts = replace_nemotron_h_projection_linears(
            model,
            make_linear=lambda linear, name, backend: SimpleFP4Linear.from_linear(linear),
            backend_for_layer=lambda layer_idx: attn_backend,
            tail_bf16_names=tail_bf16_names,
            final_bf16_layer_indices=final_bf16_layer_indices,
            label="FP4",
        )
        projection_counts["mamba_in"] += fused_mamba_rms_count
        if fused_mamba_rms_count:
            logger.info(
                "FP4 Nemotron-H projection conversion including fused RMS: "
                "attention=%d mamba_in=%d mamba_out=%d head=%d",
                projection_counts["attention"],
                projection_counts["mamba_in"],
                projection_counts["mamba_out"],
                projection_counts["head"],
            )

        localcta_residual = use_tk_localcta_v4_ffn_residual_epilogue()
        v5_residual = use_tk_v5_ffn_residual_epilogue()
        h_tile_requested = os.environ.get("USE_FP4_CODA_H_TILE_RMS", "0") == "1"
        exact_cde_requested = os.environ.get("USE_FP4_CODA_EXACT_CDE", "0") == "1"
        exact_wo_requested = os.environ.get("USE_FP4_CODA_EXACT_CDE_WO", "0") == "1"
        if exact_cde_requested or exact_wo_requested:
            if h_tile_requested:
                raise RuntimeError("exact C/D/E and H tile carriers are mutually exclusive")
            localcta_exact = os.environ.get("USE_TK_LOCALCTA", "0") == "1"
            if exact_wo_requested and not localcta_exact:
                raise RuntimeError(
                    "exact Wo-to-FFN C/D/E is retained only for localCTA v4"
                )
            if localcta_exact:
                if os.environ.get("USE_TK_LOCALCTA_VARIANT", "v1").strip().lower() != "v4":
                    raise RuntimeError("exact C/D/E localCTA support requires variant v4")
                if os.environ.get("USE_TK_LOCALCTA_DIRECT_CONTRACT", "0") == "1":
                    raise RuntimeError(
                        "exact C/D/E does not support the localCTA direct-TE contract"
                    )
                if not localcta_residual:
                    raise RuntimeError(
                        "exact C/D/E localCTA support requires the v4 residual epilogue"
                    )
            if (
                os.environ.get("MXFP4_BACKEND_VERSION") is not None
                or os.environ.get("USE_MXFP4_TK_BACKEND", "0") == "1"
            ):
                raise RuntimeError("exact C/D/E is not yet implemented for MXFP4")
            if not localcta_exact and not v5_residual:
                raise RuntimeError("exact C/D/E requires the regular-v5 residual epilogue")
        h_tile_enabled = (
            h_tile_requested
            and os.environ.get("USE_TK_LOCALCTA", "0") == "1"
            and os.environ.get("USE_TK_LOCALCTA_VARIANT", "v1").strip().lower()
            == "v4"
        )
        if (
            localcta_residual or v5_residual or h_tile_enabled
            or exact_cde_requested or exact_wo_requested
        ):
            for block_name, block in model.named_modules():
                feed_forward = getattr(block, "feed_forward", None)
                mixer = getattr(block, "mixer", None)
                v5_shape_supported = bool(
                    v5_residual
                    and (
                        (
                            getattr(feed_forward, "dim", None) == 4096
                            and getattr(feed_forward, "hidden_dim", None) == 14336
                        )
                        or (
                            getattr(mixer, "dim", None) == 4096
                            and getattr(mixer, "hidden_dim", None) == 14336
                        )
                    )
                )
                if (
                    (localcta_residual or v5_shape_supported or h_tile_enabled
                     or exact_cde_requested or exact_wo_requested)
                    and
                    hasattr(block, "attention")
                    and hasattr(block, "feed_forward")
                    and hasattr(block.feed_forward, "forward_with_residual")
                ):
                    block.forward = types.MethodType(_ffn_residual_fused_block_forward, block)
                    logger.info(f"  NVFP4 FFN RESIDUAL FUSED BLOCK: {block_name}")
                elif (
                    (localcta_residual or v5_shape_supported or h_tile_enabled
                     or exact_cde_requested or exact_wo_requested)
                    and
                    getattr(block, "block_type", None) == "mlp"
                    and hasattr(block, "mixer")
                    and hasattr(block.mixer, "forward_with_residual")
                ):
                    block.forward = types.MethodType(_nemotron_h_mlp_residual_fused_block_forward, block)
                    logger.info(f"  NVFP4 NEMOTRON-H MLP RESIDUAL FUSED BLOCK: {block_name}")

            if h_tile_enabled:
                layers = getattr(model, "layers", None)
                if layers is None:
                    raise RuntimeError("NVFP4 H requires a model.layers container")
                h_layers = list(layers.values()) if hasattr(layers, "values") else list(layers)
                if not h_layers:
                    raise RuntimeError("NVFP4 H found no model layers to wire")
                for index, layer in enumerate(h_layers):
                    attention = getattr(layer, "attention", None)
                    feed_forward = getattr(layer, "feed_forward", None)
                    if not (
                        hasattr(attention, "forward_with_residual")
                        and hasattr(feed_forward, "forward_with_h_carrier")
                    ):
                        raise RuntimeError(
                            f"NVFP4 H layer {index} lacks a fused attention/FFN carrier owner"
                        )
                    owner_ref = None
                    if index + 1 < len(h_layers):
                        next_attention = getattr(h_layers[index + 1], "attention", None)
                        next_owner = getattr(next_attention, "fused", None)
                        if next_owner is None:
                            raise RuntimeError(
                                "NVFP4 H next-layer fused attention owner is unavailable"
                            )
                        owner_ref = weakref.ref(next_owner)
                    object.__setattr__(
                        layer, "_fp4_coda_h_next_attention_owner", owner_ref
                    )
                    object.__setattr__(
                        layer, "_fsdp_preserve_forward_input_dtypes", True
                    )
                logger.info("  NVFP4 H TILE CARRIER WIRED: %d layers", len(h_layers))

            if exact_cde_requested:
                layers = getattr(model, "layers", None)
                if layers is None:
                    raise RuntimeError("exact C/D/E requires a model.layers container")
                cde_layers = list(layers.values()) if hasattr(layers, "values") else list(layers)
                if not cde_layers:
                    raise RuntimeError("exact C/D/E found no model layers to wire")
                for index, layer in enumerate(cde_layers):
                    attention = getattr(layer, "attention", None)
                    feed_forward = getattr(layer, "feed_forward", None)
                    if not (
                        hasattr(attention, "forward_with_cde_partial")
                        and hasattr(feed_forward, "forward_with_residual")
                    ):
                        raise RuntimeError(
                            f"exact C/D/E layer {index} lacks fused attention/FFN owners"
                        )
                    if index + 1 < len(cde_layers):
                        next_attention = getattr(cde_layers[index + 1], "attention", None)
                        next_owner = getattr(next_attention, "fused", None)
                        producer_eps = float(getattr(feed_forward, "epsilon"))
                        consumer_eps = float(getattr(next_owner, "epsilon"))
                        if producer_eps != consumer_eps:
                            raise RuntimeError(
                                "exact C/D/E requires matching W2-producer and next-QKV "
                                f"epsilon, got {producer_eps} and {consumer_eps} at layer {index}"
                            )
                    object.__setattr__(
                        layer, "_fp4_cde_has_next", index + 1 < len(cde_layers)
                    )
                    object.__setattr__(
                        layer, "_fsdp_preserve_forward_input_dtypes", True
                    )
                logger.info("  NVFP4 EXACT C/D/E CARRIER WIRED: %d layers", len(cde_layers))

            if exact_wo_requested:
                layers = getattr(model, "layers", None)
                if layers is None:
                    raise RuntimeError("exact Wo C/D/E requires a model.layers container")
                wo_layers = list(layers.values()) if hasattr(layers, "values") else list(layers)
                if not wo_layers:
                    raise RuntimeError("exact Wo C/D/E found no model layers to wire")
                for index, layer in enumerate(wo_layers):
                    attention = getattr(layer, "attention", None)
                    feed_forward = getattr(layer, "feed_forward", None)
                    if not (
                        hasattr(attention, "forward_with_residual")
                        and hasattr(feed_forward, "forward_with_residual")
                    ):
                        raise RuntimeError(
                            f"exact Wo C/D/E layer {index} lacks fused attention/FFN owners"
                        )
                    if (
                        getattr(feed_forward, "dim", None) != 4096
                        or getattr(feed_forward, "hidden_dim", None)
                        not in {14336, 21504}
                    ):
                        raise RuntimeError(
                            f"exact Wo C/D/E layer {index} is not a supported "
                            "production 4096x{14336,21504} FFN"
                        )
                logger.info(
                    "  NVFP4 EXACT WO C/D/E CARRIER WIRED: %d layers", len(wo_layers)
                )

        nemotron_interlayer_cde = (
            os.environ.get("USE_FP4_NEMOTRON_INTERLAYER_CDE", "0") == "1"
        )
        if nemotron_interlayer_cde:
            if (
                os.environ.get("USE_TK_LOCALCTA", "0") != "1"
                or os.environ.get("USE_TK_LOCALCTA_VARIANT", "v1")
                .strip()
                .lower()
                != "v4"
            ):
                raise RuntimeError(
                    "Nemotron inter-layer CDE requires localCTA v4"
                )
            layers = getattr(model, "layers", None)
            if layers is None:
                raise RuntimeError(
                    "Nemotron inter-layer CDE requires model.layers"
                )
            cde_layers = (
                list(layers.values()) if hasattr(layers, "values") else list(layers)
            )
            mamba_mlp_pairs = 0
            mamba_attention_pairs = 0
            attention_mlp_pairs = 0
            mlp_mamba_pairs = 0
            for index, layer in enumerate(cde_layers):
                object.__setattr__(
                    layer, "_fsdp_preserve_forward_input_dtypes", True
                )
                block_type = getattr(layer, "block_type", None)
                next_type = (
                    getattr(cde_layers[index + 1], "block_type", None)
                    if index + 1 < len(cde_layers)
                    else None
                )
                if block_type == "mamba":
                    out_proj = getattr(
                        getattr(layer, "mixer", None),
                        "out_proj",
                        None,
                    )
                    in_proj = getattr(
                        getattr(layer, "mixer", None),
                        "in_proj",
                        None,
                    )
                    if not (
                        callable(getattr(out_proj, "forward", None))
                        and isinstance(in_proj, NVFP4RMSNormLinearTK)
                    ):
                        raise RuntimeError(
                            f"Nemotron Mamba CDE layer {index} lacks native FP4 owners"
                        )
                    object.__setattr__(
                        layer,
                        "_fp4_nemotron_cde_emit",
                        next_type in {"mlp", "attention"},
                    )
                    layer.forward = types.MethodType(
                        _nemotron_h_mamba_cde_block_forward,
                        layer,
                    )
                    if next_type == "mlp":
                        mamba_mlp_pairs += 1
                    elif next_type == "attention":
                        mamba_attention_pairs += 1
                elif block_type == "attention":
                    mixer = getattr(layer, "mixer", None)
                    if not (
                        isinstance(mixer, NemotronHFusedAttentionWrapper)
                        and hasattr(mixer.fused, "forward_qkv")
                        and hasattr(mixer.fused, "forward_wo")
                    ):
                        raise RuntimeError(
                            f"Nemotron attention CDE layer {index} lacks native FP4 owners"
                        )
                    object.__setattr__(
                        layer,
                        "_fp4_nemotron_cde_emit",
                        next_type == "mlp",
                    )
                    layer.forward = types.MethodType(
                        _nemotron_h_attention_cde_block_forward,
                        layer,
                    )
                    if next_type == "mlp":
                        attention_mlp_pairs += 1
                elif block_type == "mlp":
                    mixer = getattr(layer, "mixer", None)
                    if not (
                        hasattr(mixer, "forward_with_residual")
                        and getattr(mixer, "dim", None) == 4096
                        and getattr(mixer, "hidden_dim", None) == 21504
                    ):
                        raise RuntimeError(
                            f"Nemotron MLP CDE layer {index} lacks a native FP4 owner"
                        )
                    object.__setattr__(
                        layer,
                        "_fp4_nemotron_cde_emit",
                        next_type == "mamba",
                    )
                    layer.forward = types.MethodType(
                        _nemotron_h_mlp_cde_block_forward,
                        layer,
                    )
                    if next_type == "mamba":
                        mlp_mamba_pairs += 1
            if (
                mamba_mlp_pairs,
                mamba_attention_pairs,
                attention_mlp_pairs,
                mlp_mamba_pairs,
            ) != (20, 4, 4, 23):
                raise RuntimeError(
                    "Nemotron 8B CDE expected production pair counts "
                    "(Mamba->MLP, Mamba->attention, attention->MLP, "
                    "MLP->Mamba)=(20,4,4,23), wired "
                    f"({mamba_mlp_pairs},{mamba_attention_pairs},"
                    f"{attention_mlp_pairs},{mlp_mamba_pairs})"
                )
            logger.info(
                "  NVFP4 NEMOTRON CDE WIRED: MAMBA->MLP=%d "
                "MAMBA->ATTN=%d ATTN->MLP=%d MLP->MAMBA=%d",
                mamba_mlp_pairs,
                mamba_attention_pairs,
                attention_mlp_pairs,
                mlp_mamba_pairs,
            )

        ffn_summary = ", ".join(
            f"{count} {kind} ({cls_name})"
            for (kind, cls_name), count in sorted(ffn_impl_counts.items())
        ) or "none"
        logger.info(
            f"FP4 conversion done: {attn_count} attn ({_FusedAttentionFP4.__name__}), "
            f"{ffn_count} FFN [{ffn_summary}], "
            f"{head_count} head (Float32Linear), "
            f"{norm_identity_count} norms→identity"
        )
        if sqrelu_request_ignored_count:
            logger.info(
                "FP4Converter: ignored USE_FP4_SQRELU_FFN_TK for %d three-linear gated FFNs; "
                "those layers use the model architecture's SwiGLU/SiLU path.",
                sqrelu_request_ignored_count,
            )

    def post_optimizer_hook(self, model):
        """Release step-scoped FP4/TK caches after weights change."""
        modules = model if isinstance(model, (list, tuple)) else [model]
        for root in modules:
            for module in root.modules():
                invalidate = getattr(module, "invalidate_weight_cache", None)
                if callable(invalidate):
                    invalidate()
        clear_fused_fp4_step_caches()
        clear_tk_step_caches()


register_model_converter(FP4Converter, "fp4")
