# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Fail-closed schemas and defusion helpers for Llama-8B DCP evaluation.

The fused production routes do not store the unfused TorchTitan Llama
parameter names.  Each layer contains a stacked QKV tensor and absorbed norm
parameters.  Pure-v5 and localCTA additionally serialize two frozen
``_NormIdentity`` aliases per layer, while MXFP4-v4+row-SR does not.  The
production localCTA/MXFP4 hybrid has those aliases only in its first 27
localCTA layers; its final five layers use MXFP4.  Evaluation must account for
those exact layouts before mapping the weights to a common Hugging Face Llama
model.  Treating a fused checkpoint as an ordinary unfused checkpoint silently
misses most of the learned weights.

This module deliberately has no Transformer Engine or custom-kernel imports.
It can validate and defuse a checkpoint on a CPU conversion host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


PURE_V5_FUSED = "pure-v5-fused-v1"
LOCALCTA_V4_FUSED = "localcta-v4-fused-v1"
MXFP4_V4_ROW_SR_FUSED = "mxfp4-v4-row-sr-fused-v1"
LOCALCTA_MXFP4_HYBRID_FUSED = "localcta-mxfp4-hybrid-27-5-fused-v1"
LOCALCTA_MXFP4_HYBRID_LOCALCTA_LAYERS = 27
# Short names are kept as the public route constants used by the command-line
# tools.  The values retain the algorithm/version identity of the exact source
# checkpoints rather than collapsing structurally similar routes together.
LOCALCTA_FUSED = LOCALCTA_V4_FUSED
MXFP4_FUSED = MXFP4_V4_ROW_SR_FUSED
BF16_UNFUSED = "bf16-unfused-v1"
TE_NATIVE_NVFP4_UNFUSED = "te-native-nvfp4-unfused-v1"
TE_FOL4_NVFP4_UNFUSED = "te-fol4-nvfp4-final4-bf16-unfused-v1"
TE_FOL4_BF16_FINAL_LAYERS = 4
FUSED_ROUTES = (
    PURE_V5_FUSED,
    LOCALCTA_FUSED,
    MXFP4_FUSED,
    LOCALCTA_MXFP4_HYBRID_FUSED,
)
ALIASED_FUSED_ROUTES = (
    PURE_V5_FUSED,
    LOCALCTA_FUSED,
    LOCALCTA_MXFP4_HYBRID_FUSED,
)
UNFUSED_ROUTES = (
    BF16_UNFUSED,
    TE_NATIVE_NVFP4_UNFUSED,
    TE_FOL4_NVFP4_UNFUSED,
)
SUPPORTED_ROUTES = (*FUSED_ROUTES, *UNFUSED_ROUTES)

OPTIMIZER_STATE_FIELDS = ("exp_avg", "exp_avg_sq", "step")
OPTIMIZER_GROUP_FIELDS = (
    "amsgrad",
    "betas",
    "capturable",
    "decoupled_weight_decay",
    "differentiable",
    "eps",
    "foreach",
    "fused",
    "initial_lr",
    "lr",
    "maximize",
    "weight_decay",
)


@dataclass(frozen=True)
class LlamaSpec:
    """Structural dimensions required to validate and defuse a Llama model."""

    layers: int = 32
    dim: int = 4096
    hidden_dim: int = 14336
    vocab_size: int = 128256
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1.0e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072

    def __post_init__(self) -> None:
        if (
            min(
                self.layers,
                self.dim,
                self.hidden_dim,
                self.vocab_size,
                self.heads,
                self.kv_heads,
                self.head_dim,
            )
            <= 0
        ):
            raise ValueError("all Llama dimensions must be positive")
        if self.dim != self.heads * self.head_dim:
            raise ValueError("dim must equal heads * head_dim")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")

    @property
    def q_rows(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_rows(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkv_rows(self) -> int:
        return self.q_rows + 2 * self.kv_rows


LLAMA3_8B = LlamaSpec()


@dataclass(frozen=True)
class RouteValidation:
    route: str
    model_tensors: int
    extra_state_tensors: int
    optimizer_parameters: int
    frozen_aliases: int
    source_dtype: str


def _global_shapes(spec: LlamaSpec) -> dict[str, tuple[int, ...]]:
    return {
        "tok_embeddings.weight": (spec.vocab_size, spec.dim),
        "norm.weight": (spec.dim,),
        "output.weight": (spec.vocab_size, spec.dim),
    }


def fused_trainable_shapes(
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, tuple[int, ...]]:
    """Exact trainable manifest shared by the three fused routes."""

    result = _global_shapes(spec)
    per_layer = {
        "attention.fused.norm_weight": (spec.dim,),
        "attention.fused.w_qkv": (spec.qkv_rows, spec.dim),
        "attention.fused.wo_weight": (spec.dim, spec.dim),
        "feed_forward.norm_weight": (spec.dim,),
        "feed_forward.w1_weight": (spec.hidden_dim, spec.dim),
        "feed_forward.w2_weight": (spec.dim, spec.hidden_dim),
        "feed_forward.w3_weight": (spec.hidden_dim, spec.dim),
    }
    result.update(
        {
            f"layers.{layer}.{suffix}": shape
            for layer in range(spec.layers)
            for suffix, shape in per_layer.items()
        }
    )
    return result


def fused_alias_keys(spec: LlamaSpec = LLAMA3_8B) -> set[str]:
    """Frozen absorbed-norm aliases present only in aliased fused routes."""

    return {
        f"layers.{layer}.{norm}.weight"
        for layer in range(spec.layers)
        for norm in ("attention_norm", "ffn_norm")
    }


def localcta_mxfp4_hybrid_alias_keys(
    spec: LlamaSpec = LLAMA3_8B,
) -> set[str]:
    """Frozen aliases in the exact 27-localCTA/5-MXFP4 production hybrid."""

    # Reduced specs are useful for converter tests; production LLAMA3_8B has
    # all 32 layers and therefore takes the exact 27-layer alias prefix.
    localcta_layers = min(LOCALCTA_MXFP4_HYBRID_LOCALCTA_LAYERS, spec.layers)
    return {
        f"layers.{layer}.{norm}.weight"
        for layer in range(localcta_layers)
        for norm in ("attention_norm", "ffn_norm")
    }


def _aliased_fused_shapes(spec: LlamaSpec) -> dict[str, tuple[int, ...]]:
    result = fused_trainable_shapes(spec)
    result.update({key: (spec.dim,) for key in sorted(fused_alias_keys(spec))})
    return result


def pure_v5_fused_shapes(spec: LlamaSpec = LLAMA3_8B) -> dict[str, tuple[int, ...]]:
    """Exact pure-v5 model tensor manifest, including frozen aliases."""

    return _aliased_fused_shapes(spec)


def pure_v5_alias_keys(spec: LlamaSpec = LLAMA3_8B) -> set[str]:
    """Backward-compatible pure-v5 alias helper."""

    return fused_alias_keys(spec)


def pure_v5_trainable_shapes(
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, tuple[int, ...]]:
    """Backward-compatible pure-v5 trainable helper."""

    return fused_trainable_shapes(spec)


def localcta_fused_shapes(spec: LlamaSpec = LLAMA3_8B) -> dict[str, tuple[int, ...]]:
    """Exact repaired localCTA model manifest, including frozen aliases."""

    return _aliased_fused_shapes(spec)


def localcta_alias_keys(spec: LlamaSpec = LLAMA3_8B) -> set[str]:
    return fused_alias_keys(spec)


def localcta_trainable_shapes(
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, tuple[int, ...]]:
    return fused_trainable_shapes(spec)


def mxfp4_fused_shapes(spec: LlamaSpec = LLAMA3_8B) -> dict[str, tuple[int, ...]]:
    """Exact MXFP4-v4+row-SR model manifest (no frozen aliases)."""

    return fused_trainable_shapes(spec)


def mxfp4_trainable_shapes(
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, tuple[int, ...]]:
    return fused_trainable_shapes(spec)


def localcta_mxfp4_hybrid_fused_shapes(
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, tuple[int, ...]]:
    """Exact 27-localCTA/5-MXFP4 hybrid manifest, including 54 aliases."""

    result = fused_trainable_shapes(spec)
    result.update(
        {
            key: (spec.dim,)
            for key in sorted(localcta_mxfp4_hybrid_alias_keys(spec))
        }
    )
    return result


def bf16_unfused_shapes(spec: LlamaSpec = LLAMA3_8B) -> dict[str, tuple[int, ...]]:
    result = _global_shapes(spec)
    per_layer = {
        "attention.wq.weight": (spec.q_rows, spec.dim),
        "attention.wk.weight": (spec.kv_rows, spec.dim),
        "attention.wv.weight": (spec.kv_rows, spec.dim),
        "attention.wo.weight": (spec.dim, spec.dim),
        "attention_norm.weight": (spec.dim,),
        "feed_forward.w1.weight": (spec.hidden_dim, spec.dim),
        "feed_forward.w2.weight": (spec.dim, spec.hidden_dim),
        "feed_forward.w3.weight": (spec.hidden_dim, spec.dim),
        "ffn_norm.weight": (spec.dim,),
    }
    result.update(
        {
            f"layers.{layer}.{suffix}": shape
            for layer in range(spec.layers)
            for suffix, shape in per_layer.items()
        }
    )
    return result


def te_native_extra_state_shapes(
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, tuple[int, ...]]:
    """Exact empty Transformer Engine state tensors in the native NVFP4 DCP."""

    modules = (
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "attention.wo",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    )
    return {
        f"layers.{layer}.{module}._extra_state": (0,)
        for layer in range(spec.layers)
        for module in modules
    }


def te_fol4_extra_state_shapes(
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, tuple[int, ...]]:
    """TE state for the exact final-four-layers-in-BF16 recipe.

    Transformer Engine owns layers ``0..layers-5``.  The final four blocks are
    ordinary BF16 modules and therefore must not serialize TE ``_extra_state``
    entries.  For production Llama-8B this is 28 TE layers and 196 state
    tensors.  Clipping at zero keeps reduced structural tests well-defined.
    """

    modules = (
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "attention.wo",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    )
    te_layers = max(spec.layers - TE_FOL4_BF16_FINAL_LAYERS, 0)
    return {
        f"layers.{layer}.{module}._extra_state": (0,)
        for layer in range(te_layers)
        for module in modules
    }


def route_model_shapes(
    route: str, spec: LlamaSpec = LLAMA3_8B
) -> dict[str, tuple[int, ...]]:
    if route == PURE_V5_FUSED:
        return pure_v5_fused_shapes(spec)
    if route == LOCALCTA_FUSED:
        return localcta_fused_shapes(spec)
    if route == MXFP4_FUSED:
        return mxfp4_fused_shapes(spec)
    if route == LOCALCTA_MXFP4_HYBRID_FUSED:
        return localcta_mxfp4_hybrid_fused_shapes(spec)
    if route in UNFUSED_ROUTES:
        return bf16_unfused_shapes(spec)
    raise ValueError(f"unsupported checkpoint route: {route!r}")


def route_trainable_shapes(
    route: str, spec: LlamaSpec = LLAMA3_8B
) -> dict[str, tuple[int, ...]]:
    if route in FUSED_ROUTES:
        return fused_trainable_shapes(spec)
    if route in UNFUSED_ROUTES:
        return bf16_unfused_shapes(spec)
    raise ValueError(f"unsupported checkpoint route: {route!r}")


def route_alias_keys(route: str, spec: LlamaSpec = LLAMA3_8B) -> set[str]:
    if route in (PURE_V5_FUSED, LOCALCTA_FUSED):
        return fused_alias_keys(spec)
    if route == LOCALCTA_MXFP4_HYBRID_FUSED:
        return localcta_mxfp4_hybrid_alias_keys(spec)
    if route == MXFP4_FUSED or route in UNFUSED_ROUTES:
        return set()
    raise ValueError(f"unsupported checkpoint route: {route!r}")


def route_extra_state_shapes(
    route: str, spec: LlamaSpec = LLAMA3_8B
) -> dict[str, tuple[int, ...]]:
    if route == TE_NATIVE_NVFP4_UNFUSED:
        return te_native_extra_state_shapes(spec)
    if route == TE_FOL4_NVFP4_UNFUSED:
        return te_fol4_extra_state_shapes(spec)
    if route in FUSED_ROUTES or route == BF16_UNFUSED:
        return {}
    raise ValueError(f"unsupported checkpoint route: {route!r}")


def route_model_fqns(route: str, spec: LlamaSpec = LLAMA3_8B) -> set[str]:
    """Exact layer/global FQN inventory, including metadata-only route state."""

    return set(route_model_shapes(route, spec)) | set(
        route_extra_state_shapes(route, spec)
    )


def _metadata_mapping(metadata: Any) -> Mapping[str, Any]:
    state = getattr(metadata, "state_dict_metadata", None)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("checkpoint metadata has no state_dict_metadata mapping")
    return state


def _tensor_shape_and_dtype(value: Any) -> tuple[tuple[int, ...], torch.dtype] | None:
    properties = getattr(value, "properties", None)
    size = getattr(value, "size", None)
    dtype = getattr(properties, "dtype", None)
    if size is None or not isinstance(dtype, torch.dtype):
        return None
    return tuple(int(dimension) for dimension in size), dtype


def _model_keys(keys: Iterable[str]) -> set[str]:
    return {
        key
        for key in keys
        if key in {"tok_embeddings.weight", "norm.weight", "output.weight"}
        or key.startswith("layers.")
    }


def detect_route(metadata: Any, spec: LlamaSpec = LLAMA3_8B) -> str:
    actual = _model_keys(str(key) for key in _metadata_mapping(metadata))
    matches = [
        route
        for route in SUPPORTED_ROUTES
        if actual == route_model_fqns(route, spec)
    ]
    if not matches:
        counts = {
            route: len(actual & route_model_fqns(route, spec))
            for route in SUPPORTED_ROUTES
        }
        raise RuntimeError(
            "checkpoint model schema is not an exact supported route; "
            f"actual={len(actual)} overlaps={counts}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            "checkpoint model schema is ambiguous between exact supported routes; "
            f"explicit route is required: actual={len(actual)} matches={matches}"
        )
    return matches[0]


def _optimizer_fields(keys: set[str], parameter: str, root: str) -> set[str]:
    prefix = f"{root}.{parameter}."
    return {key[len(prefix) :] for key in keys if key.startswith(prefix)}


def validate_route_metadata(
    metadata: Any,
    route: str = "auto",
    *,
    spec: LlamaSpec = LLAMA3_8B,
    expected_dtype: torch.dtype | None = torch.float32,
    require_optimizer: bool = True,
) -> RouteValidation:
    """Validate exact model and optimizer manifests for an evaluation route.

    ``expected_dtype`` defaults to FP32 because the production jobs used FP32
    master/export storage even though their forward compute used BF16/FP4.
    Set it to ``None`` only when auditing a known older export policy.
    """

    state = _metadata_mapping(metadata)
    route = detect_route(metadata, spec) if route == "auto" else route
    expected_model = route_model_shapes(route, spec)
    expected_extra_state = route_extra_state_shapes(route, spec)
    expected_model_fqns = route_model_fqns(route, spec)
    actual_model = _model_keys(str(key) for key in state)
    if actual_model != expected_model_fqns:
        missing = sorted(expected_model_fqns - actual_model)
        extra = sorted(actual_model - expected_model_fqns)
        raise RuntimeError(
            f"{route} model schema mismatch: missing={missing[:8]} extra={extra[:8]}"
        )

    observed_dtypes: set[torch.dtype] = set()
    for name, wanted_shape in expected_model.items():
        tensor_info = _tensor_shape_and_dtype(state[name])
        if tensor_info is None:
            raise RuntimeError(f"model entry is not tensor metadata: {name}")
        shape, dtype = tensor_info
        if shape != wanted_shape:
            raise RuntimeError(
                f"model shape mismatch for {name}: {shape} != {wanted_shape}"
            )
        if expected_dtype is not None and dtype != expected_dtype:
            raise RuntimeError(
                f"model dtype mismatch for {name}: {dtype} != {expected_dtype}"
            )
        observed_dtypes.add(dtype)

    for name, wanted_shape in expected_extra_state.items():
        tensor_info = _tensor_shape_and_dtype(state[name])
        if tensor_info is None:
            raise RuntimeError(f"TE extra-state entry is not tensor metadata: {name}")
        shape, dtype = tensor_info
        if shape != wanted_shape or dtype != torch.uint8:
            raise RuntimeError(
                f"TE extra-state tensor mismatch for {name}: "
                f"shape={shape} dtype={dtype}"
            )

    trainable = route_trainable_shapes(route, spec)
    aliases = route_alias_keys(route, spec)
    all_keys = {str(key) for key in state}
    optimizer_parameters = 0
    for name, wanted_shape in trainable.items():
        state_fields = _optimizer_fields(all_keys, name, "optimizer.state")
        group_fields = _optimizer_fields(all_keys, name, "optimizer.param_groups")
        if require_optimizer:
            if not set(OPTIMIZER_STATE_FIELDS) <= state_fields:
                raise RuntimeError(f"optimizer state is incomplete for {name}")
            if not set(OPTIMIZER_GROUP_FIELDS) <= group_fields:
                raise RuntimeError(f"optimizer parameter group is incomplete for {name}")
        if state_fields:
            optimizer_parameters += 1
            for field in OPTIMIZER_STATE_FIELDS:
                key = f"optimizer.state.{name}.{field}"
                tensor_info = _tensor_shape_and_dtype(state.get(key))
                if tensor_info is None:
                    raise RuntimeError(f"optimizer entry is not tensor metadata: {key}")
                shape, dtype = tensor_info
                expected_shape = () if field == "step" else wanted_shape
                if shape != expected_shape or dtype != torch.float32:
                    raise RuntimeError(
                        f"optimizer tensor mismatch for {key}: shape={shape} dtype={dtype}"
                    )

    for alias in aliases:
        if _optimizer_fields(all_keys, alias, "optimizer.state") or _optimizer_fields(
            all_keys, alias, "optimizer.param_groups"
        ):
            raise RuntimeError(f"frozen {route} alias has optimizer state: {alias}")
    for extra_state in expected_extra_state:
        if _optimizer_fields(
            all_keys, extra_state, "optimizer.state"
        ) or _optimizer_fields(all_keys, extra_state, "optimizer.param_groups"):
            raise RuntimeError(
                f"TE extra-state tensor has optimizer entries: {extra_state}"
            )

    if require_optimizer:
        known_optimizer = set(trainable)
        expected_optimizer_state = {
            f"optimizer.state.{name}.{field}"
            for name in known_optimizer
            for field in OPTIMIZER_STATE_FIELDS
        }
        actual_optimizer_state = {
            key for key in all_keys if key.startswith("optimizer.state.")
        }
        if actual_optimizer_state != expected_optimizer_state:
            missing = sorted(expected_optimizer_state - actual_optimizer_state)
            extra = sorted(actual_optimizer_state - expected_optimizer_state)
            raise RuntimeError(
                "optimizer state manifest mismatch: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        expected_optimizer_groups = {
            f"optimizer.param_groups.{name}.{field}"
            for name in known_optimizer
            for field in OPTIMIZER_GROUP_FIELDS
        }
        actual_optimizer_groups = {
            key for key in all_keys if key.startswith("optimizer.param_groups.")
        }
        if actual_optimizer_groups != expected_optimizer_groups:
            missing = sorted(expected_optimizer_groups - actual_optimizer_groups)
            extra = sorted(actual_optimizer_groups - expected_optimizer_groups)
            raise RuntimeError(
                "optimizer parameter-group manifest mismatch: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )

    source_dtype = ",".join(sorted(str(dtype) for dtype in observed_dtypes))
    return RouteValidation(
        route=route,
        model_tensors=len(expected_model),
        extra_state_tensors=len(expected_extra_state),
        optimizer_parameters=optimizer_parameters,
        frozen_aliases=len(aliases),
        source_dtype=source_dtype,
    )


def _layer_and_suffix(key: str) -> tuple[int, str]:
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit():
        raise ValueError(f"not a layer parameter: {key}")
    return int(parts[1]), parts[2]


def reorder_tt_rope_rows(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Convert TorchTitan's interleaved Q/K RoPE rows to HF ordering."""

    if weight.ndim != 2 or weight.shape[0] % head_dim:
        raise ValueError(
            f"invalid Q/K tensor for head_dim={head_dim}: {tuple(weight.shape)}"
        )
    viewed = weight.reshape(-1, head_dim, weight.shape[1])
    even = torch.arange(0, head_dim, 2, device=weight.device)
    reordered = torch.cat((viewed[:, even], viewed[:, even + 1]), dim=1)
    return reordered.flatten(0, 1).contiguous()


def defuse_fused_tensor(
    key: str,
    tensor: torch.Tensor,
    route: str,
    spec: LlamaSpec = LLAMA3_8B,
) -> dict[str, torch.Tensor]:
    """Map one fused-route tensor to ordinary TorchTitan parameter names."""

    if route not in FUSED_ROUTES:
        raise ValueError(f"route does not use the fused schema: {route!r}")
    wanted = route_model_shapes(route, spec)
    if key not in wanted:
        raise KeyError(f"not a {route} model tensor: {key}")
    if tuple(tensor.shape) != wanted[key]:
        raise ValueError(
            f"shape mismatch for {key}: {tuple(tensor.shape)} != {wanted[key]}"
        )
    if key in route_alias_keys(route, spec):
        return {}
    if key in _global_shapes(spec):
        return {key: tensor}

    layer, suffix = _layer_and_suffix(key)
    prefix = f"layers.{layer}."
    if suffix == "attention.fused.norm_weight":
        return {prefix + "attention_norm.weight": tensor}
    if suffix == "attention.fused.w_qkv":
        q, k, v = tensor.split((spec.q_rows, spec.kv_rows, spec.kv_rows), dim=0)
        return {
            prefix + "attention.wq.weight": q,
            prefix + "attention.wk.weight": k,
            prefix + "attention.wv.weight": v,
        }
    if suffix == "attention.fused.wo_weight":
        return {prefix + "attention.wo.weight": tensor}
    if suffix == "feed_forward.norm_weight":
        return {prefix + "ffn_norm.weight": tensor}
    ffn = {
        "feed_forward.w1_weight": "feed_forward.w1.weight",
        "feed_forward.w2_weight": "feed_forward.w2.weight",
        "feed_forward.w3_weight": "feed_forward.w3.weight",
    }
    if suffix in ffn:
        return {prefix + ffn[suffix]: tensor}
    raise AssertionError(f"unhandled {route} tensor: {key}")


def defuse_pure_v5_tensor(
    key: str, tensor: torch.Tensor, spec: LlamaSpec = LLAMA3_8B
) -> dict[str, torch.Tensor]:
    """Backward-compatible pure-v5 defusion helper."""

    return defuse_fused_tensor(key, tensor, PURE_V5_FUSED, spec)


def tt_to_hf_tensors(
    key: str,
    tensor: torch.Tensor,
    route: str,
    *,
    spec: LlamaSpec = LLAMA3_8B,
    output_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Convert one route tensor into one or more HF Llama tensors."""

    if route in FUSED_ROUTES:
        unfused = defuse_fused_tensor(key, tensor, route, spec)
    elif route in UNFUSED_ROUTES:
        wanted = bf16_unfused_shapes(spec)
        if key not in wanted or tuple(tensor.shape) != wanted.get(key):
            raise ValueError(
                f"invalid {route} model tensor: {key} {tuple(tensor.shape)}"
            )
        unfused = {key: tensor}
    else:
        raise ValueError(f"unsupported checkpoint route: {route!r}")

    result: dict[str, torch.Tensor] = {}
    global_map = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    suffix_map = {
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "attention_norm.weight": "input_layernorm.weight",
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
    }
    for unfused_key, value in unfused.items():
        if unfused_key in global_map:
            hf_key = global_map[unfused_key]
        else:
            layer, suffix = _layer_and_suffix(unfused_key)
            if suffix not in suffix_map:
                raise AssertionError(f"unhandled unfused tensor: {unfused_key}")
            hf_key = f"model.layers.{layer}.{suffix_map[suffix]}"
            if suffix in {"attention.wq.weight", "attention.wk.weight"}:
                value = reorder_tt_rope_rows(value, spec.head_dim)
        result[hf_key] = value.to(dtype=output_dtype).contiguous()
    return result


def expected_hf_shapes(spec: LlamaSpec = LLAMA3_8B) -> dict[str, tuple[int, ...]]:
    """Exact destination tensor manifest shared by all evaluation routes."""

    result = {
        "model.embed_tokens.weight": (spec.vocab_size, spec.dim),
        "model.norm.weight": (spec.dim,),
        "lm_head.weight": (spec.vocab_size, spec.dim),
    }
    per_layer = {
        "self_attn.q_proj.weight": (spec.q_rows, spec.dim),
        "self_attn.k_proj.weight": (spec.kv_rows, spec.dim),
        "self_attn.v_proj.weight": (spec.kv_rows, spec.dim),
        "self_attn.o_proj.weight": (spec.dim, spec.dim),
        "input_layernorm.weight": (spec.dim,),
        "mlp.gate_proj.weight": (spec.hidden_dim, spec.dim),
        "mlp.down_proj.weight": (spec.dim, spec.hidden_dim),
        "mlp.up_proj.weight": (spec.hidden_dim, spec.dim),
        "post_attention_layernorm.weight": (spec.dim,),
    }
    result.update(
        {
            f"model.layers.{layer}.{suffix}": shape
            for layer in range(spec.layers)
            for suffix, shape in per_layer.items()
        }
    )
    return result


def hf_config(spec: LlamaSpec = LLAMA3_8B, dtype: str = "bfloat16") -> dict[str, Any]:
    """Generate the exact Transformers-4.48 Llama config used for comparison."""

    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "head_dim": spec.head_dim,
        "hidden_act": "silu",
        "hidden_size": spec.dim,
        "initializer_range": 0.02,
        "intermediate_size": spec.hidden_dim,
        "max_position_embeddings": spec.max_position_embeddings,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": spec.heads,
        "num_hidden_layers": spec.layers,
        "num_key_value_heads": spec.kv_heads,
        "pad_token_id": 128001,
        "pretraining_tp": 1,
        "rms_norm_eps": spec.norm_eps,
        "rope_scaling": {"rope_type": "default"},
        "rope_theta": spec.rope_theta,
        "tie_word_embeddings": False,
        "torch_dtype": dtype,
        "transformers_version": "4.48.2",
        "use_cache": True,
        "vocab_size": spec.vocab_size,
    }
