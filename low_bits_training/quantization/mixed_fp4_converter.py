"""Layer-wise mixed NVFP4 / MXFP4 TK converter.

This converter is for regular TorchTitan DDP-style runs. It routes complete
transformer blocks to either one NVFP4 backend (localCTA v4 or native v5) or
the MXFP4 TK fused wrappers so we can test true layer-wise mixed precision
without letting two independent converters race over the same modules.
"""

from __future__ import annotations

import os
import types
from collections.abc import Callable

import torch
import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger

try:
    from transformer_engine.common.recipe import NVFP4BlockScaling
except ImportError:
    NVFP4BlockScaling = None

from .fused_te_linear import (
    NVFP4RMSNormLinearTK,
    SimpleFP4Linear,
    use_tk_localcta_v4_wo_attn_layout,
)
from .float32_linear import Float32Linear
from .nemotron_h_projection_policy import (
    NemotronHFusedAttentionWrapper,
    is_nemotron_h_attention_block,
    nemotron_h_fp4_output_head_enabled,
    replace_nemotron_h_projection_linears,
    use_nemotron_h_fused_attention,
)
from .fp4_converter import (
    FusedAttentionFP4_TK,
    FusedFeedForwardFP4_TK,
    FusedSquaredReLUFeedForwardFP4,
    FusedSquaredReLUFeedForwardFP4_TK,
    _FusedAttentionWrapper as NVFP4AttentionWrapper,
    _NormIdentity as NVFP4NormIdentity,
    _call_with_optional_backend_mode,
    _is_output_head_name,
    _is_nemotron_h_mlp_block,
    _is_transformer_layer_module,
    _last_bf16_ffn_layer_indices,
    _last_bf16_layer_indices,
    _layer_index_from_name,
    _maybe_enable_nvfp4_live_path,
    _nemotron_h_mlp_residual_fused_block_forward,
    _tail_bf16_linear_names,
    apply_localcta_v4_profile_defaults,
    clear_fused_fp4_step_caches,
    rsetattr,
)
from .mxfp4_tk_converter import (
    ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK,
    FusedAttentionMXFP4_TK,
    FusedFeedForwardMXFP4_TK,
    FusedSquaredReLUFeedForwardMXFP4_TK,
    MXFP4LinearTK,
    MXFP4RMSNormLinearTK,
    _FusedAttentionWrapper as MXFP4AttentionWrapper,
    _NormIdentity as MXFP4NormIdentity,
    _is_llama_stacked_qkv_attention,
    _log_mxfp4_highwater_route_once,
    _log_mxfp4_rht_route_once,
    _mxfp4_bool_env,
    _keep_mxfp4_tk_output_head_bf16,
    _require_mxfp4_tk_llama_bf16_output_head,
    _residual_fused_block_forward as _mxfp4_residual_fused_block_forward,
    _use_mxfp4_tk_convert_output_head,
    use_mxfp4_wo_attn_layout,
    use_mxfp4_residual_fusion_attn,
    use_mxfp4_residual_fusion_ffn,
)
from .mxfp4_fused_linear import _register_mixed_mxfp4_fsdp_layer_indices
from .tk_gemm import clear_tk_step_caches, use_tk_localcta_v4_ffn_residual_epilogue


_BACKEND_ALIASES = {
    "localcta": "localcta",
    "nvfp4": "localcta",
    "nvfp4_localcta": "localcta",
    "nvfp4_localcta_v4": "localcta",
    "localcta_v4": "localcta",
    "v5": "v5",
    "tk": "v5",
    "tk_v5": "v5",
    "nvfp4_tk_v5": "v5",
    "mxfp4": "mxfp4",
    "mx": "mxfp4",
    "mxfp4_tk": "mxfp4",
}


def _normalize_backend(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key not in _BACKEND_ALIASES:
        raise ValueError(
            f"Unknown LBT_FP4_MIXED backend {value!r}; "
            "expected localcta/localcta_v4, v5/tk_v5, or mxfp4."
        )
    return _BACKEND_ALIASES[key]


def _projection_backend(
    routes: dict[int, str],
    head_backend: str,
    layer_idx: int | None,
    kind: str,
) -> str:
    if kind == "head":
        return head_backend
    override = os.environ.get(f"LBT_FP4_MIXED_{kind.upper()}_BACKEND")
    if override is None and kind in {"mamba_in", "mamba_out"}:
        override = os.environ.get("LBT_FP4_MIXED_MAMBA_BACKEND")
    if override:
        return _normalize_backend(override)
    return routes.get(layer_idx, "localcta") if layer_idx is not None else head_backend


def _parse_layer_spec(value: str) -> dict[int, str]:
    """Parse `backend:1-16,20;mxfp4:17-19` into 0-based layer routing."""

    routes: dict[int, str] = {}
    for clause in value.replace("|", ";").split(";"):
        clause = clause.strip()
        if not clause:
            continue
        if ":" not in clause:
            raise ValueError(f"LBT_FP4_MIXED_LAYERS clause must be backend:layers, got {clause!r}")
        backend_raw, layers_raw = clause.split(":", 1)
        backend = _normalize_backend(backend_raw)
        for item in layers_raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                left, right = item.split("-", 1)
                start = int(left)
                end = int(right)
                if start > end:
                    raise ValueError(f"Descending LBT_FP4_MIXED_LAYERS range: {item!r}")
                layer_numbers = range(start, end + 1)
            else:
                layer_numbers = (int(item),)
            for layer_no in layer_numbers:
                if layer_no <= 0:
                    raise ValueError("LBT_FP4_MIXED_LAYERS uses 1-based layer numbers")
                routes[layer_no - 1] = backend
    return routes


def _block_layer_indices(model: nn.Module) -> list[int]:
    indices: list[int] = []
    for name, module in model.named_modules():
        if _is_transformer_layer_module(module):
            idx = _layer_index_from_name(name)
            if idx is not None:
                indices.append(idx)
    return sorted(set(indices))


def _mixed_tail_count(all_layers: list[int]) -> int:
    raw_count = int(os.environ.get("LBT_FP4_MIXED_TAIL_LAYERS", "4") or 0)
    return min(len(all_layers), max(0, raw_count))


def _default_backend_for_layer(layer_idx: int, all_layers: list[int]) -> str:
    policy = os.environ.get("LBT_FP4_MIXED_POLICY", "front_localcta").strip().lower().replace("-", "_")
    if policy in {"default", "default_front"}:
        policy = "front_localcta"
    if policy in {"all_localcta", "localcta", "nvfp4"}:
        return "localcta"
    if policy in {"all_v5", "v5", "tk_v5"}:
        return "v5"
    if policy in {"all_mxfp4", "mxfp4"}:
        return "mxfp4"
    if policy in {"alternate", "alternate_localcta_odd"}:
        return "localcta" if (layer_idx % 2 == 0) else "mxfp4"
    if policy == "alternate_mxfp4_odd":
        return "mxfp4" if (layer_idx % 2 == 0) else "localcta"
    if not all_layers:
        return os.environ.get("LBT_FP4_MIXED_DEFAULT_BACKEND", "localcta")
    if policy in {"tail_mxfp4", "final_mxfp4", "last_mxfp4"}:
        tail_count = _mixed_tail_count(all_layers)
        tail_start = all_layers[-tail_count] if tail_count else max(all_layers) + 1
        return "mxfp4" if layer_idx >= tail_start else "localcta"
    if policy in {"tail_localcta", "final_localcta", "last_localcta"}:
        tail_count = _mixed_tail_count(all_layers)
        tail_start = all_layers[-tail_count] if tail_count else max(all_layers) + 1
        return "localcta" if layer_idx >= tail_start else "mxfp4"
    if policy not in {"front_localcta", "front_mxfp4"}:
        raise ValueError(
            "LBT_FP4_MIXED_POLICY must be one of front_localcta, front_mxfp4, "
            "tail_mxfp4, tail_localcta, alternate, alternate_mxfp4_odd, "
            "all_localcta, all_v5, or all_mxfp4."
        )
    split_raw = os.environ.get("LBT_FP4_MIXED_SPLIT_LAYER")
    if split_raw:
        split_idx = int(split_raw) - 1
    else:
        split_idx = all_layers[(len(all_layers) - 1) // 2]
    front_backend = "localcta" if policy == "front_localcta" else "mxfp4"
    back_backend = "mxfp4" if front_backend == "localcta" else "localcta"
    return front_backend if layer_idx <= split_idx else back_backend


def _build_layer_routes(model: nn.Module) -> dict[int, str]:
    all_layers = _block_layer_indices(model)
    if not all_layers:
        raise ValueError(
            "Mixed FP4 routing did not discover any transformer layers."
        )
    explicit = _parse_layer_spec(os.environ.get("LBT_FP4_MIXED_LAYERS", ""))
    unknown_layers = sorted(set(explicit) - set(all_layers))
    if unknown_layers:
        available = ", ".join(str(idx + 1) for idx in all_layers)
        unknown = ", ".join(str(idx + 1) for idx in unknown_layers)
        raise ValueError(
            "LBT_FP4_MIXED_LAYERS references layers not present in the "
            f"model: {unknown}; available layers: {available}"
        )
    default_backend = _normalize_backend(os.environ.get("LBT_FP4_MIXED_DEFAULT_BACKEND", "localcta"))
    routes: dict[int, str] = {}
    for idx in all_layers:
        routes[idx] = explicit.get(idx, _default_backend_for_layer(idx, all_layers) or default_backend)
    return routes


def _configure_regular_localcta_v4() -> str:
    os.environ["USE_TK_GEMM"] = "1"
    os.environ["USE_TK_QUANT"] = "1"
    os.environ["USE_TK_LOCALCTA"] = "1"
    os.environ["USE_TK_LOCALCTA_VARIANT"] = "v4"
    os.environ["USE_TK_LOCALCTA_FUSED"] = "0"
    os.environ["FP4_ATTN_BACKEND"] = "localcta"
    os.environ["FP4_FFN_BACKEND"] = "localcta"
    _maybe_enable_nvfp4_live_path("localcta")
    return apply_localcta_v4_profile_defaults()


def _configure_regular_v5() -> str:
    # Mixed launchers can inherit a localCTA tuning profile from the outer
    # experiment. Remove v4-only selectors before regular-v5 wrappers inspect
    # process-global environment state.
    for key in tuple(os.environ):
        if key.startswith(
            (
                "USE_TK_LOCALCTA_",
                "USE_TK_FFN_LOCALCTA_",
                "USE_TK_QKV_LOCALCTA_",
            )
        ):
            os.environ.pop(key)
    for key in (
        "USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD",
        "USE_TK_FFN_REQUANT_H13_OPERANDS",
        "USE_TK_FFN_SEPARATE_WGRAD_STREAM",
    ):
        os.environ.pop(key, None)
    # localCTA high-water disables the ordinary wgrad stream.  That generic
    # flag is also read by the regular-v5 FFN, so absence is not sufficient:
    # overwrite a value inherited from a preceding matrix arm.
    os.environ["USE_TK_FFN_DISABLE_WGRAD_STREAM"] = "0"
    os.environ["LBT_LOCALCTA_V4_PROFILE"] = "off"
    # Exact C/D/E is also process-global: the shared residual block forward
    # reads these flags at runtime, after layers have been routed.  A mixed-v5
    # model therefore cannot safely inherit localCTA's Wo carrier (and this
    # converter does not wire the inter-layer carrier across mixed backends).
    # Keep pure-v5 and localCTA exact-C/D/E support in their dedicated
    # converters; mixed localCTA routes do not enter this helper.
    exact_cde_keys = (
        "USE_FP4_CODA_EXACT_CDE",
        "USE_FP4_CODA_EXACT_CDE_WO",
    )
    if any(os.environ.get(key, "0") == "1" for key in exact_cde_keys):
        logger.warning(
            "Mixed v5 routing disables exact C/D/E carriers; use the regular "
            "v5 or localCTA converter for a homogeneous exact-C/D/E route."
        )
    for key in exact_cde_keys:
        os.environ[key] = "0"
    os.environ["USE_TK_GEMM"] = "1"
    os.environ["USE_TK_QUANT"] = "1"
    os.environ["USE_TK_LOCALCTA"] = "0"
    os.environ["USE_TK_LOCALCTA_FUSED"] = "0"
    os.environ["FP4_ATTN_BACKEND"] = "tk"
    os.environ["FP4_FFN_BACKEND"] = "tk"
    _maybe_enable_nvfp4_live_path("tk")
    return "v5"


def _configure_regular_nvfp4(routes: dict[int, str]) -> str:
    """Select the one global NVFP4 kernel family used beside MXFP4 modules."""

    nv_backends = sorted(set(routes.values()) - {"mxfp4"})
    if len(nv_backends) > 1:
        raise ValueError(
            "A mixed model may use MXFP4 plus localCTA v4 or MXFP4 plus v5, "
            "but localCTA v4 and v5 cannot coexist because their fused "
            f"wrappers share global backend selectors; got {nv_backends}."
        )
    if not nv_backends:
        return "mxfp4_only"
    if nv_backends[0] == "localcta":
        return _configure_regular_localcta_v4()
    if nv_backends[0] == "v5":
        return _configure_regular_v5()
    raise AssertionError(f"Unhandled mixed NVFP4 backend: {nv_backends[0]}")


def _nvfp4_backend_mode(backend: str) -> str:
    if backend == "localcta":
        return "localcta"
    if backend == "v5":
        return "tk"
    raise ValueError(f"Expected an NVFP4 backend, got {backend!r}")


def _norm_identity_like(norm: nn.Module, *, backend: str) -> nn.Module:
    dim = norm.weight.shape[0] if hasattr(norm, "weight") else 0
    norm_dtype = norm.weight.dtype if hasattr(norm, "weight") else torch.bfloat16
    if backend == "mxfp4":
        return MXFP4NormIdentity(dim, dtype=norm_dtype, trainable=False)
    return NVFP4NormIdentity(dim, dtype=norm_dtype)


def _replace_output_heads(model: nn.Module, tail_bf16_names: set[str]) -> int:
    output_heads = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _is_output_head_name(name)
    ]
    require_llama_bf16_head = _require_mxfp4_tk_llama_bf16_output_head()
    if not _use_mxfp4_tk_convert_output_head():
        if require_llama_bf16_head and len(output_heads) != 1:
            raise RuntimeError(
                "mixed localCTA/MXFP4 production requires exactly one root "
                f"output module, found {[name for name, _ in output_heads]}"
            )
        for name, module in output_heads:
            _keep_mxfp4_tk_output_head_bf16(name, module)
        return 0
    if require_llama_bf16_head:
        raise RuntimeError(
            "mixed localCTA/MXFP4 ordinary-BF16-head requirement conflicts "
            "with an explicitly enabled legacy output-head conversion"
        )
    if nemotron_h_fp4_output_head_enabled(False):
        return 0
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_output_head_name(name):
            if name in tail_bf16_names:
                logger.info("  MIXED KEEP HEAD BF16: %s", name)
                continue
            replacements.append((name, module))
    for name, module in replacements:
        new_layer = Float32Linear(module.in_features, module.out_features, bias=module.bias is not None)
        new_layer = new_layer.to(module.weight.device).to(module.weight.dtype)
        if module.weight.device.type != "meta":
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)
        rsetattr(model, name, new_layer)
        logger.info("  MIXED HEAD: %s -> Float32Linear", name)
    return len(replacements)


def _use_fp4_sqrelu_tk() -> bool:
    value = os.environ.get("USE_FP4_SQRELU_FFN_TK")
    if value is None:
        value = os.environ.get("USE_FP4_EXPERIMENTAL_SQRELU_FFN_TK", "0")
    return value == "1"


def _replace_nemotron_h_mamba_rms_in_projections(
    model: nn.Module,
    *,
    backend_for_projection: Callable[[int | None, str, str], str],
    tail_bf16_names: set[str],
    final_bf16_layer_indices: set[int],
) -> dict[str, int]:
    counts = {"localcta": 0, "v5": 0, "mxfp4": 0}
    if not _mxfp4_bool_env(
        "LBT_NEMOTRON_H_FUSED_MAMBA_RMS_IN_PROJ",
        True,
    ):
        return counts
    if os.environ.get("LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ", "1") == "0":
        return counts

    candidates = []
    for block_name, block in model.named_modules():
        if getattr(block, "block_type", None) != "mamba":
            continue
        layer_idx = _layer_index_from_name(block_name)
        if layer_idx is None or layer_idx in final_bf16_layer_indices:
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
        backend = backend_for_projection(
            layer_idx,
            "mamba_in",
            projection_name,
        )
        candidates.append(
            (block, in_proj, norm, projection_name, layer_idx, backend)
        )

    for (
        block,
        in_proj,
        norm,
        projection_name,
        layer_idx,
        backend,
    ) in candidates:
        fused_cls = (
            MXFP4RMSNormLinearTK
            if backend == "mxfp4"
            else NVFP4RMSNormLinearTK
        )
        fused = fused_cls(
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
        block.norm = _norm_identity_like(norm, backend=backend)
        counts[backend] += 1
        logger.info(
            "  MIXED FUSED NEMOTRON-H MAMBA RMS+IN L%d %s: %s",
            layer_idx + 1,
            backend,
            projection_name,
        )
    return counts


class MixedFP4LocalCTAMXFP4Converter(ModelConverter):
    """Route layers between one regular NVFP4 backend and MXFP4."""

    def __init__(self, job_config: JobConfig | None, parallel_dims: ParallelDims | None):
        self.job_config = job_config

    def convert(self, model: nn.Module):
        routes = _build_layer_routes(model)
        profile = _configure_regular_nvfp4(routes)
        _log_mxfp4_highwater_route_once()
        _log_mxfp4_rht_route_once()

        recipe = NVFP4BlockScaling()
        head_backend = _normalize_backend(
            os.environ.get(
                "LBT_FP4_MIXED_HEAD_BACKEND",
                "mxfp4"
                if os.environ.get("LBT_FP4_MIXED_POLICY", "").strip().lower().replace("-", "_")
                in {"tail_mxfp4", "final_mxfp4", "last_mxfp4", "front_localcta"}
                else "localcta",
            )
        )
        verbose = os.environ.get("LBT_FP4_MIXED_VERBOSE", "0") == "1"
        tail_bf16_names = _tail_bf16_linear_names(model)
        final_bf16_layer_indices = _last_bf16_layer_indices(model)
        final_bf16_ffn_layer_indices = _last_bf16_ffn_layer_indices(model)

        def projection_backend(layer_idx: int | None, kind: str, name: str) -> str:
            del name
            return _projection_backend(routes, head_backend, layer_idx, kind)

        head_count = _replace_output_heads(model, tail_bf16_names)
        attn_counts = {"localcta": 0, "v5": 0, "mxfp4": 0}
        ffn_counts = {"localcta": 0, "v5": 0, "mxfp4": 0}
        mxfp4_forward_prefetch_layers: set[int] = set()
        mxfp4_backward_prefetch_layers: set[int] = set()
        norm_identity_count = 0

        blocks_to_fuse_attn = []
        for block_name, block in model.named_modules():
            if not (hasattr(block, "attention") and hasattr(block, "attention_norm")):
                continue
            attn = block.attention
            norm = block.attention_norm
            layer_idx = _layer_index_from_name(block_name)
            if layer_idx is None:
                continue
            if layer_idx in final_bf16_layer_indices:
                logger.info("  MIXED KEEP BF16 ATTN: %s.attention final-layer ablation", block_name)
                continue
            if not _is_llama_stacked_qkv_attention(attn):
                continue
            qkv_tail = {
                f"{block_name}.attention.wq",
                f"{block_name}.attention.wk",
                f"{block_name}.attention.wv",
            } & tail_bf16_names
            if qkv_tail:
                logger.info("  MIXED KEEP BF16 ATTN: %s.attention tail=%s", block_name, sorted(qkv_tail))
                continue
            backend = projection_backend(
                layer_idx, "attention", f"{block_name}.attention"
            )
            wo_bf16 = f"{block_name}.attention.wo" in tail_bf16_names
            blocks_to_fuse_attn.append((block_name, block, attn, norm, layer_idx, backend, wo_bf16))

        for block_name, block, attn, norm, layer_idx, backend, wo_bf16 in blocks_to_fuse_attn:
            if backend == "mxfp4":
                fused_attn = FusedAttentionMXFP4_TK.from_attention(attn, norm)
                block.attention = MXFP4AttentionWrapper(attn, fused_attn)
                mxfp4_forward_prefetch_layers.add(layer_idx)
            else:
                fused_attn = _call_with_optional_backend_mode(
                    FusedAttentionFP4_TK.from_attention,
                    attn,
                    norm,
                    backend_mode=_nvfp4_backend_mode(backend),
                )
                block.attention = NVFP4AttentionWrapper(attn, fused_attn)
            if wo_bf16:
                fused_attn._force_wo_bf16 = True
            fused_attn._lbt_debug_name = f"{block_name}.attention"
            block.attention_norm = _norm_identity_like(norm, backend=backend)
            attn_counts[backend] += 1
            norm_identity_count += 1
            if verbose:
                logger.info(
                    "  MIXED ATTN L%d %s: %s.attention%s",
                    layer_idx + 1,
                    backend,
                    block_name,
                    " (BF16 wo)" if wo_bf16 else "",
                )

        if use_nemotron_h_fused_attention():
            blocks_to_fuse_nemotron_attn = []
            for block_name, block in model.named_modules():
                if not is_nemotron_h_attention_block(block):
                    continue
                layer_idx = _layer_index_from_name(block_name)
                if layer_idx is None:
                    continue
                if layer_idx in final_bf16_layer_indices:
                    logger.info("  MIXED KEEP BF16 NEMOTRON-H ATTN: %s.mixer final-layer ablation", block_name)
                    continue
                qkv_tail = {
                    f"{block_name}.mixer.q_proj",
                    f"{block_name}.mixer.k_proj",
                    f"{block_name}.mixer.v_proj",
                } & tail_bf16_names
                if qkv_tail:
                    logger.info("  MIXED KEEP BF16 NEMOTRON-H ATTN: %s.mixer tail=%s", block_name, sorted(qkv_tail))
                    continue
                backend = projection_backend(
                    layer_idx, "attention", f"{block_name}.mixer"
                )
                wo_bf16 = f"{block_name}.mixer.o_proj" in tail_bf16_names
                blocks_to_fuse_nemotron_attn.append((block_name, block, block.mixer, block.norm, layer_idx, backend, wo_bf16))

            for block_name, block, attn, norm, layer_idx, backend, wo_bf16 in blocks_to_fuse_nemotron_attn:
                if backend == "mxfp4":
                    fused_attn = FusedAttentionMXFP4_TK.from_attention(attn, norm)
                    direct_wo = use_mxfp4_wo_attn_layout
                    mxfp4_forward_prefetch_layers.add(layer_idx)
                else:
                    fused_attn = _call_with_optional_backend_mode(
                        FusedAttentionFP4_TK.from_attention,
                        attn,
                        norm,
                        backend_mode=_nvfp4_backend_mode(backend),
                    )
                    direct_wo = use_tk_localcta_v4_wo_attn_layout
                if wo_bf16:
                    fused_attn._force_wo_bf16 = True
                fused_attn._lbt_debug_name = f"{block_name}.mixer"
                block.mixer = NemotronHFusedAttentionWrapper(
                    attn,
                    fused_attn,
                    use_direct_wo_layout=direct_wo,
                )
                block.norm = _norm_identity_like(norm, backend=backend)
                attn_counts[backend] += 1
                norm_identity_count += 1
                if verbose:
                    logger.info(
                        "  MIXED NEMOTRON-H ATTN L%d %s: %s.mixer%s",
                        layer_idx + 1,
                        backend,
                        block_name,
                        " (BF16 o_proj)" if wo_bf16 else "",
                    )

        blocks_to_fuse_ffn = []
        for block_name, block in model.named_modules():
            if not (hasattr(block, "feed_forward") and hasattr(block, "ffn_norm")):
                continue
            ffn = block.feed_forward
            norm = block.ffn_norm
            layer_idx = _layer_index_from_name(block_name)
            if layer_idx is None:
                continue
            if layer_idx in final_bf16_layer_indices:
                logger.info("  MIXED KEEP BF16 FFN: %s.feed_forward final-layer ablation", block_name)
                continue
            if layer_idx in final_bf16_ffn_layer_indices:
                logger.info("  MIXED KEEP BF16 FFN: %s.feed_forward final-FFN ablation", block_name)
                continue
            if not (hasattr(ffn, "w1") and isinstance(ffn.w1, nn.Linear)):
                continue
            ffn_tail = {
                f"{block_name}.feed_forward.w1",
                f"{block_name}.feed_forward.w2",
                f"{block_name}.feed_forward.w3",
            } & tail_bf16_names
            if ffn_tail:
                logger.info("  MIXED KEEP BF16 FFN: %s.feed_forward tail=%s", block_name, sorted(ffn_tail))
                continue
            backend = projection_backend(
                layer_idx, "ffn", f"{block_name}.feed_forward"
            )
            blocks_to_fuse_ffn.append(
                (block_name, block, ffn, norm, layer_idx, backend, "feed_forward", "ffn_norm", f"{block_name}.feed_forward")
            )

        for block_name, block in model.named_modules():
            if not _is_nemotron_h_mlp_block(block):
                continue
            norm = block.norm
            ffn = block.mixer
            layer_idx = _layer_index_from_name(block_name)
            if layer_idx is None:
                continue
            if layer_idx in final_bf16_layer_indices:
                logger.info("  MIXED KEEP BF16 NEMOTRON-H MLP: %s.mixer final-layer ablation", block_name)
                continue
            if layer_idx in final_bf16_ffn_layer_indices:
                logger.info("  MIXED KEEP BF16 NEMOTRON-H MLP: %s.mixer final-FFN ablation", block_name)
                continue
            ffn_tail = {
                f"{block_name}.mixer.up_proj",
                f"{block_name}.mixer.down_proj",
            } & tail_bf16_names
            if ffn_tail:
                logger.info("  MIXED KEEP BF16 NEMOTRON-H MLP: %s.mixer tail=%s", block_name, sorted(ffn_tail))
                continue
            backend = projection_backend(
                layer_idx, "ffn", f"{block_name}.mixer"
            )
            blocks_to_fuse_ffn.append(
                (block_name, block, ffn, norm, layer_idx, backend, "mixer", "norm", f"{block_name}.mixer")
            )

        skip_mxfp4_fused_ffn = _mxfp4_bool_env("MXFP4_SKIP_FUSED_FFN", False)
        for block_name, block, ffn, norm, layer_idx, backend, ffn_attr, norm_attr, debug_name in blocks_to_fuse_ffn:
            if backend == "mxfp4":
                if skip_mxfp4_fused_ffn:
                    for child_name in ("w1", "w2", "w3"):
                        child = getattr(ffn, child_name, None)
                        if isinstance(child, nn.Linear):
                            setattr(ffn, child_name, MXFP4LinearTK.from_linear(child))
                    for child_name in ("up_proj", "down_proj"):
                        child = getattr(ffn, child_name, None)
                        if isinstance(child, nn.Linear):
                            setattr(ffn, child_name, MXFP4LinearTK.from_linear(child))
                    logger.info("  MIXED MXFP4 UNFUSED FFN LINEARS: %s", debug_name)
                    continue
                if hasattr(ffn, "w3"):
                    ffn_cls = FusedFeedForwardMXFP4_TK
                elif os.environ.get("MXFP4_USE_EXPERIMENTAL_SQRELU_FFN", "0") == "1":
                    ffn_cls = ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK
                else:
                    ffn_cls = FusedSquaredReLUFeedForwardMXFP4_TK
                fused_ffn = ffn_cls.from_unfused(ffn, norm)
                mxfp4_backward_prefetch_layers.add(layer_idx)
            else:
                if hasattr(ffn, "w3"):
                    ffn_cls = FusedFeedForwardFP4_TK
                elif _use_fp4_sqrelu_tk():
                    ffn_cls = FusedSquaredReLUFeedForwardFP4_TK
                else:
                    ffn_cls = FusedSquaredReLUFeedForwardFP4
                fused_ffn = _call_with_optional_backend_mode(
                    ffn_cls.from_unfused,
                    ffn,
                    norm,
                    recipe=recipe,
                    backend_mode=_nvfp4_backend_mode(backend),
                )
            fused_ffn._lbt_debug_name = debug_name
            setattr(block, ffn_attr, fused_ffn)
            setattr(block, norm_attr, _norm_identity_like(norm, backend=backend))
            ffn_counts[backend] += 1
            norm_identity_count += 1
            if verbose:
                logger.info(
                    "  MIXED FFN L%d %s: %s -> %s",
                    layer_idx + 1,
                    backend,
                    debug_name,
                    ffn_cls.__name__,
                )

        fused_mamba_rms_counts = (
            _replace_nemotron_h_mamba_rms_in_projections(
                model,
                backend_for_projection=projection_backend,
                tail_bf16_names=tail_bf16_names,
                final_bf16_layer_indices=final_bf16_layer_indices,
            )
        )
        norm_identity_count += sum(fused_mamba_rms_counts.values())

        replace_nemotron_h_projection_linears(
            model,
            make_linear=(
                lambda linear, name, backend: MXFP4LinearTK.from_linear(linear)
                if backend == "mxfp4"
                else SimpleFP4Linear.from_linear(linear)
            ),
            backend_for_layer=lambda layer_idx: routes.get(layer_idx, "localcta") if layer_idx is not None else head_backend,
            backend_for_projection=projection_backend,
            tail_bf16_names=tail_bf16_names,
            final_bf16_layer_indices=final_bf16_layer_indices,
            label="MIXED",
        )

        localcta_residual = use_tk_localcta_v4_ffn_residual_epilogue()
        v5_residual_value = os.environ.get("USE_TK_V5_FFN_RESIDUAL_EPILOGUE")
        v5_residual = (
            v5_residual_value is None
            or v5_residual_value.strip().lower() in {"1", "true", "yes", "on"}
        )
        if localcta_residual or v5_residual:
            for block_name, block in model.named_modules():
                layer_idx = _layer_index_from_name(block_name)
                backend = routes.get(layer_idx) if layer_idx is not None else None
                feed_forward = getattr(block, "feed_forward", None)
                mixer = getattr(block, "mixer", None)
                residual_supported = bool(
                    (backend == "localcta" and localcta_residual)
                    or (
                        backend == "v5"
                        and v5_residual
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
                )
                if (
                    residual_supported
                    and
                    hasattr(block, "attention")
                    and hasattr(block, "feed_forward")
                    and layer_idx is not None
                    and hasattr(block.feed_forward, "forward_with_residual")
                ):
                    from .fp4_converter import _ffn_residual_fused_block_forward

                    block.forward = types.MethodType(_ffn_residual_fused_block_forward, block)
                    logger.info(
                        "  MIXED %s FFN RESIDUAL FUSED BLOCK: %s",
                        backend,
                        block_name,
                    )
                elif (
                    residual_supported
                    and
                    getattr(block, "block_type", None) == "mlp"
                    and layer_idx is not None
                    and hasattr(block, "mixer")
                    and hasattr(block.mixer, "forward_with_residual")
                ):
                    block.forward = types.MethodType(_nemotron_h_mlp_residual_fused_block_forward, block)
                    logger.info(
                        "  MIXED %s NEMOTRON-H MLP RESIDUAL FUSED BLOCK: %s",
                        backend,
                        block_name,
                    )

        residual_attn_enabled = use_mxfp4_residual_fusion_attn()
        residual_ffn_enabled = use_mxfp4_residual_fusion_ffn()
        if residual_attn_enabled or residual_ffn_enabled:
            for block_name, block in model.named_modules():
                layer_idx = _layer_index_from_name(block_name)
                if (
                    hasattr(block, "attention")
                    and hasattr(block, "feed_forward")
                    and layer_idx is not None
                    and routes.get(layer_idx) == "mxfp4"
                    and hasattr(block.attention, "forward_with_residual")
                    and hasattr(block.feed_forward, "forward_with_residual")
                ):
                    block.forward = types.MethodType(_mxfp4_residual_fused_block_forward, block)
                    logger.info("  MIXED MXFP4 RESIDUAL FUSED BLOCK: %s", block_name)
                elif (
                    getattr(block, "block_type", None) == "mlp"
                    and layer_idx is not None
                    and routes.get(layer_idx) == "mxfp4"
                    and hasattr(block, "mixer")
                    and hasattr(block.mixer, "forward_with_residual")
                ):
                    block.forward = types.MethodType(_nemotron_h_mlp_residual_fused_block_forward, block)
                    logger.info("  MIXED MXFP4 NEMOTRON-H MLP RESIDUAL FUSED BLOCK: %s", block_name)

        route_summary = ", ".join(
            f"L{idx + 1}:{backend}" for idx, backend in sorted(routes.items())
        )
        logger.info(
            "MixedFP4LocalCTAMXFP4Converter done: profile=%s "
            "attn(localcta=%d v5=%d mxfp4=%d) "
            "ffn(localcta=%d v5=%d mxfp4=%d) heads=%d norms=%d routes=[%s]",
            profile,
            attn_counts["localcta"],
            attn_counts["v5"],
            attn_counts["mxfp4"],
            ffn_counts["localcta"],
            ffn_counts["v5"],
            ffn_counts["mxfp4"],
            head_count,
            norm_identity_count,
            route_summary,
        )
        _register_mixed_mxfp4_fsdp_layer_indices(
            forward=mxfp4_forward_prefetch_layers,
            backward=mxfp4_backward_prefetch_layers,
        )

    def post_optimizer_hook(self, model):
        modules = model if isinstance(model, (list, tuple)) else [model]
        for root in modules:
            for module in root.modules():
                invalidate = getattr(module, "invalidate_weight_cache", None)
                if callable(invalidate):
                    invalidate()
        clear_fused_fp4_step_caches()
        clear_tk_step_caches()


register_model_converter(MixedFP4LocalCTAMXFP4Converter, "fp4_mixed_localcta_mxfp4")
