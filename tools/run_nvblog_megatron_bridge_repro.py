#!/usr/bin/env python3
"""Run a fair NVIDIA-blog TE reproduction with Megatron Bridge.

This is intentionally separate from tools/run_nvblog_llama3_8b_matrix.py.
The local matrix uses Torchtitan/LBT plus optional TE Linear replacement. A
fair NVIDIA TE reproduction should use the NeMo Megatron Bridge Llama 3 8B
recipes directly, because the public NVIDIA result was measured on that stack.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NVIDIA_BLOG_URL = (
    "https://developer.nvidia.com/blog/"
    "using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/"
)
BRIDGE_LLAMA_DOC_URL = (
    "https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/llama3.html"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_LOCAL_LLAMA3_HF_CONFIG = REPO_ROOT / "assets" / "hf" / "Meta-Llama-3-8B-config-only"
REMOTE_LLAMA3_8B = "meta-llama/Meta-Llama-3-8B"

os.environ.setdefault(
    "TORCHINDUCTOR_COMPILE_THREADS",
    os.environ.get("LBT_BRIDGE_TORCHINDUCTOR_COMPILE_THREADS", "1"),
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    recipe: str
    micro_batch_size: int
    fp4_backend: str | None = None
    env: dict[str, str] | None = None


def _bridge_tk_v5_highwater_env(
    *,
    delayed: bool = False,
    no_collect: bool = False,
    qkv_fused_sum_rms: bool = False,
) -> dict[str, str]:
    env = {
        "USE_TK_QKV_NATIVE_SPLIT3_QUANT": "1",
        "USE_TK_QKV_OVERLAP_RMS_WGRAD": "1",
        "USE_TK_QKV_PLAIN_BATCHED_ACCUM_DGRAD": "1",
        "USE_TK_FFN_PLAIN_BATCHED_ACCUM_DGRAD": "1",
        "USE_TK_FFN_FUSED_SUM_RMS": "1",
        "USE_TK_FFN_OVERLAP_RMS_WGRAD": "1",
    }
    if delayed:
        env.update(
            {
                "USE_TK_FFN_V5_DELAYED_SILU_DERIV": "1",
                "USE_TK_FFN_V5_DELAYED_DIRECT_SPLIT": "1",
                "USE_TK_FFN_V5_DELAYED_REFRESH_INTERVAL": "1",
            }
        )
        if no_collect:
            env["USE_TK_FFN_V5_DELAYED_NO_COLLECT"] = "1"
    if qkv_fused_sum_rms:
        env["USE_TK_QKV_FUSED_SUM_RMS"] = "1"
    return env


def _bridge_localcta_v4_highwater_env() -> dict[str, str]:
    return {"LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE": "highwater"}


@contextmanager
def _scoped_env(updates: dict[str, str] | None):
    if not updates:
        yield
        return
    previous = dict(os.environ)
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _import_bridge() -> tuple[Any, Any, Any]:
    try:
        from megatron.bridge.recipes.llama import (
            llama3_8b_low_precision_pretrain_config,
            llama3_8b_pretrain_config,
        )
        from megatron.bridge.training.gpt_step import forward_step
        from megatron.bridge.training.pretrain import pretrain
    except Exception as exc:  # pragma: no cover - exercised on non-Bridge envs.
        raise SystemExit(
            "Megatron Bridge is not importable in this environment. Run this "
            "script inside a NeMo/Megatron Bridge container or install "
            "`megatron.bridge` first. The current local TE/LBT runner is not a "
            "fair reproduction of NVIDIA Table 2.\n"
            f"Bridge docs: {BRIDGE_LLAMA_DOC_URL}"
        ) from exc
    return llama3_8b_pretrain_config, llama3_8b_low_precision_pretrain_config, (
        pretrain,
        forward_step,
    )


def ensure_local_llama3_8b_config(path: Path) -> None:
    """Create a config-only HF directory for Llama 3 8B.

    The Meta Llama 3 8B repository is gated. For throughput reproduction from
    random initialization we only need the architecture config because Bridge is
    called with load_weights=False by the recipe.
    """

    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "config.json"
    if config_path.exists():
        return
    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "5.3.0",
        "use_cache": True,
        "vocab_size": 128256,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


@contextmanager
def remap_llama3_hf_repo(local_hf_path: Path | None):
    if local_hf_path is None:
        yield
        return

    import megatron.bridge.recipes.llama.llama3 as llama3_module

    bridge_cls = llama3_module.AutoBridge
    original_descriptor = bridge_cls.__dict__["from_hf_pretrained"]
    original_call = bridge_cls.from_hf_pretrained

    def patched_from_hf_pretrained(cls: type, path: str | Path, **kwargs: Any) -> Any:
        if str(path) == REMOTE_LLAMA3_8B:
            path = str(local_hf_path)
        return original_call(path, **kwargs)

    bridge_cls.from_hf_pretrained = classmethod(patched_from_hf_pretrained)
    try:
        yield
    finally:
        bridge_cls.from_hf_pretrained = original_descriptor


def patch_dataset_helper_compile_if_present() -> None:
    """Avoid wheel-runtime `make` when the compiled helper module is already present."""

    if importlib.util.find_spec("megatron.core.datasets.helpers_cpp") is None:
        return
    import megatron.core.datasets.utils as dataset_utils

    noop = lambda: None
    dataset_utils.compile_helpers = noop
    for module_name in (
        "megatron.bridge.training.initialize",
        "megatron.training.initialize",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "compile_helpers"):
            setattr(module, "compile_helpers", noop)


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _set_required(obj: Any, name: str, value: Any) -> None:
    if not hasattr(obj, name):
        raise AttributeError(f"{type(obj).__name__} has no attribute {name!r}")
    setattr(obj, name, value)


def _get_required(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        raise AttributeError(f"{type(obj).__name__} has no attribute {name!r}")
    return getattr(obj, name)


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    if raw in {"1", "true", "True", "yes", "YES", "on", "ON"}:
        return True
    if raw in {"0", "false", "False", "no", "NO", "off", "OFF"}:
        return False
    raise SystemExit(f"{name} must be a boolean value, got {raw!r}")


def apply_bridge_runtime_overrides(cfg: Any, optimizer: Any) -> None:
    """Apply local compatibility/debug knobs across TE and custom FP4 cases."""

    model = getattr(cfg, "model", None)
    train = getattr(cfg, "train", None)
    ddp = getattr(cfg, "ddp", None)
    mixed_precision = getattr(cfg, "mixed_precision", None)

    recompute_modules = os.environ.get("LBT_BRIDGE_RECOMPUTE_MODULES")
    if recompute_modules is not None and model is not None:
        modules = [item.strip() for item in recompute_modules.split(",") if item.strip()]
        _set_if_present(model, "recompute_modules", modules)

    use_dist_opt = _env_bool("LBT_BRIDGE_USE_DISTRIBUTED_OPTIMIZER")
    if use_dist_opt is not None:
        if ddp is not None and hasattr(ddp, "use_distributed_optimizer"):
            setattr(ddp, "use_distributed_optimizer", use_dist_opt)
        if hasattr(optimizer, "use_distributed_optimizer"):
            setattr(optimizer, "use_distributed_optimizer", use_dist_opt)

    fp4_param = _env_bool("LBT_BRIDGE_FP4_PARAM")
    if fp4_param is not None and mixed_precision is not None:
        _set_if_present(mixed_precision, "fp4_param", fp4_param)

    fp4_param_gather = _env_bool("LBT_BRIDGE_FP4_PARAM_GATHER")
    if fp4_param_gather is not None:
        if ddp is not None:
            _set_if_present(ddp, "fp4_param_gather", fp4_param_gather)
        if mixed_precision is not None:
            _set_if_present(mixed_precision, "fp4_param_gather", fp4_param_gather)

    overlap_grad_reduce = _env_bool("LBT_BRIDGE_OVERLAP_GRAD_REDUCE")
    if overlap_grad_reduce is not None and ddp is not None:
        _set_if_present(ddp, "overlap_grad_reduce", overlap_grad_reduce)

    overlap_param_gather = _env_bool("LBT_BRIDGE_OVERLAP_PARAM_GATHER")
    if overlap_param_gather is not None and ddp is not None:
        _set_if_present(ddp, "overlap_param_gather", overlap_param_gather)

    check_nan_grad = _env_bool("LBT_BRIDGE_CHECK_FOR_NAN_IN_GRAD")
    if check_nan_grad is not None and ddp is not None:
        _set_if_present(ddp, "check_for_nan_in_grad", check_nan_grad)

    check_large_grads = _env_bool("LBT_BRIDGE_CHECK_FOR_LARGE_GRADS")
    if check_large_grads is not None and ddp is not None:
        _set_if_present(ddp, "check_for_large_grads", check_large_grads)

    deallocate_pipeline_outputs = _env_bool("LBT_BRIDGE_DEALLOCATE_PIPELINE_OUTPUTS")
    if deallocate_pipeline_outputs is not None and model is not None:
        _set_if_present(model, "deallocate_pipeline_outputs", deallocate_pipeline_outputs)

    cuda_graph_impl = os.environ.get("LBT_BRIDGE_CUDA_GRAPH_IMPL")
    if cuda_graph_impl is not None and model is not None:
        _set_if_present(model, "cuda_graph_impl", cuda_graph_impl)

    cuda_graph_scope = os.environ.get("LBT_BRIDGE_CUDA_GRAPH_SCOPE")
    if cuda_graph_scope is not None and model is not None:
        _set_if_present(model, "cuda_graph_scope", cuda_graph_scope)

    cuda_graph_warmup_steps = os.environ.get("LBT_BRIDGE_CUDA_GRAPH_WARMUP_STEPS")
    if cuda_graph_warmup_steps is not None and model is not None:
        try:
            warmup_steps = int(cuda_graph_warmup_steps)
        except ValueError as exc:
            raise SystemExit(
                "LBT_BRIDGE_CUDA_GRAPH_WARMUP_STEPS must be an integer, "
                f"got {cuda_graph_warmup_steps!r}"
            ) from exc
        _set_if_present(model, "cuda_graph_warmup_steps", warmup_steps)

    cuda_graph_single_mempool = _env_bool("LBT_BRIDGE_CUDA_GRAPH_USE_SINGLE_MEMPOOL")
    if cuda_graph_single_mempool is not None and model is not None:
        _set_if_present(model, "cuda_graph_use_single_mempool", cuda_graph_single_mempool)

    rerun_check_nan_loss = _env_bool("LBT_BRIDGE_RERUN_CHECK_FOR_NAN_IN_LOSS")
    if rerun_check_nan_loss is not None:
        rerun_state_machine = getattr(cfg, "rerun_state_machine", None)
        if rerun_state_machine is not None:
            _set_if_present(rerun_state_machine, "check_for_nan_in_loss", rerun_check_nan_loss)

    use_te_rng_tracker = _env_bool("LBT_BRIDGE_USE_TE_RNG_TRACKER")
    if use_te_rng_tracker is not None:
        if model is not None:
            _set_if_present(model, "use_te_rng_tracker", use_te_rng_tracker)
        rng = getattr(cfg, "rng", None)
        if rng is not None:
            _set_if_present(rng, "te_rng_tracker", use_te_rng_tracker)

    empty_unused_memory_level = os.environ.get("LBT_BRIDGE_EMPTY_UNUSED_MEMORY_LEVEL")
    if empty_unused_memory_level is not None and train is not None:
        try:
            level = int(empty_unused_memory_level)
        except ValueError as exc:
            raise SystemExit(
                "LBT_BRIDGE_EMPTY_UNUSED_MEMORY_LEVEL must be an integer, "
                f"got {empty_unused_memory_level!r}"
            ) from exc
        _set_if_present(train, "empty_unused_memory_level", level)

    check_optimizer_step_success = _env_bool("LBT_BRIDGE_CHECK_OPTIMIZER_STEP_SUCCESS")
    if check_optimizer_step_success is not None and train is not None:
        _set_if_present(train, "check_optimizer_step_success", check_optimizer_step_success)

    skip_sync_grad_norm = _env_bool("LBT_BRIDGE_SKIP_SYNC_GRAD_NORM_ACROSS_MP")
    if skip_sync_grad_norm is not None and train is not None:
        _set_if_present(train, "skip_sync_grad_norm_across_mp", skip_sync_grad_norm)


def validate_bridge_te_fp4_compat(cfg: Any) -> None:
    """Fail early for known incompatible TE/MCore FP4-param combinations."""

    mixed_precision = getattr(cfg, "mixed_precision", None)
    fp4_param = bool(getattr(mixed_precision, "fp4_param", False))
    fp4_param_gather = bool(getattr(mixed_precision, "fp4_param_gather", False))
    ddp = getattr(cfg, "ddp", None)
    ddp_fp4_param_gather = bool(getattr(ddp, "fp4_param_gather", False))
    if not (fp4_param or fp4_param_gather or ddp_fp4_param_gather):
        return

    try:
        from transformer_engine.pytorch.tensor import utils as te_tensor_utils
    except Exception as exc:
        raise SystemExit(
            "TE NVFP4 with FP4 parameter storage/gather requires "
            "transformer_engine.pytorch.tensor.utils.quantize_master_weights, "
            "but Transformer Engine tensor utils could not be imported. "
            "Use scripts/run_nvblog_bridge_repro_with_te.sh, a newer compatible "
            "TE/CUDA container, or set "
            "LBT_BRIDGE_FP4_PARAM=0 LBT_BRIDGE_FP4_PARAM_GATHER=0 "
            "LBT_BRIDGE_USE_DISTRIBUTED_OPTIMIZER=0 for a local compute-path smoke."
        ) from exc

    if not hasattr(te_tensor_utils, "quantize_master_weights"):
        raise SystemExit(
            "TE NVFP4 FP4-parameter storage/gather is not compatible with this "
            "Transformer Engine install: quantize_master_weights is missing. "
            "Exact NVIDIA-blog FP4-param reproduction needs a newer TE/CUDA "
            "runtime pair. Locally, use scripts/run_nvblog_bridge_repro_with_te.sh "
            "to stage TE 2.14.1 plus a compatible cuBLASLt. For a Bridge NVFP4 "
            "compute-path smoke, run with "
            "LBT_BRIDGE_FP4_PARAM=0 LBT_BRIDGE_FP4_PARAM_GATHER=0 "
            "LBT_BRIDGE_USE_DISTRIBUTED_OPTIMIZER=0 LBT_BRIDGE_OVERLAP_PARAM_GATHER=0."
        )


def _runtime_env_summary() -> dict[str, str]:
    prefixes = (
        "FP4_",
        "LBT_BRIDGE_",
        "MXFP4_",
        "NVFP4_",
        "NVTE_NVFP4_",
        "USE_TK_",
    )
    exact_names = {
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "TORCHINDUCTOR_COMPILE_THREADS",
    }
    selected = {}
    for name, value in os.environ.items():
        if name in exact_names or any(name.startswith(prefix) for prefix in prefixes):
            selected[name] = value
    return dict(sorted(selected.items()))


def _public_summary(cfg: Any, spec: CaseSpec) -> dict[str, Any]:
    model = _get_required(cfg, "model")
    train = _get_required(cfg, "train")
    dataset = _get_required(cfg, "dataset")
    optimizer = _get_required(cfg, "optimizer")
    scheduler = _get_required(cfg, "scheduler")
    ddp = getattr(cfg, "ddp", None)
    mixed_precision = getattr(cfg, "mixed_precision", None)
    rerun_state_machine = getattr(cfg, "rerun_state_machine", None)
    rng = getattr(cfg, "rng", None)
    return {
        "name": spec.name,
        "recipe": spec.recipe,
        "model": REMOTE_LLAMA3_8B,
        "hf_config_source": getattr(cfg, "_nvblog_hf_config_source", REMOTE_LLAMA3_8B),
        "transformer_impl": getattr(model, "transformer_impl", None),
        "attention_backend": getattr(model, "attention_backend", None),
        "cross_entropy_loss_fusion": getattr(model, "cross_entropy_loss_fusion", None),
        "cross_entropy_fusion_impl": getattr(model, "cross_entropy_fusion_impl", None),
        "deallocate_pipeline_outputs": getattr(model, "deallocate_pipeline_outputs", None),
        "cuda_graph_impl": getattr(model, "cuda_graph_impl", None),
        "cuda_graph_scope": getattr(model, "cuda_graph_scope", None),
        "cuda_graph_warmup_steps": getattr(model, "cuda_graph_warmup_steps", None),
        "cuda_graph_use_single_mempool": getattr(model, "cuda_graph_use_single_mempool", None),
        "rerun_check_for_nan_in_loss": getattr(
            rerun_state_machine,
            "check_for_nan_in_loss",
            None,
        ),
        "use_te_rng_tracker": getattr(model, "use_te_rng_tracker", None),
        "rng_te_rng_tracker": getattr(rng, "te_rng_tracker", None),
        "tp": getattr(model, "tensor_model_parallel_size", None),
        "pp": getattr(model, "pipeline_model_parallel_size", None),
        "cp": getattr(model, "context_parallel_size", None),
        "sequence_parallel": getattr(model, "sequence_parallel", None),
        "seq_length": getattr(model, "seq_length", None),
        "train_iters": getattr(train, "train_iters", None),
        "global_batch_size": getattr(train, "global_batch_size", None),
        "micro_batch_size": getattr(train, "micro_batch_size", None),
        "empty_unused_memory_level": getattr(train, "empty_unused_memory_level", None),
        "check_optimizer_step_success": getattr(train, "check_optimizer_step_success", None),
        "skip_sync_grad_norm_across_mp": getattr(train, "skip_sync_grad_norm_across_mp", None),
        "fp4_backend": getattr(cfg, "_nvblog_fp4_backend", None),
        "fp4_full_fusions": getattr(cfg, "_nvblog_fp4_full_fusions", False),
        "custom_mlp_recompute": getattr(cfg, "_nvblog_custom_mlp_recompute", False),
        "ddp_use_distributed_optimizer": getattr(ddp, "use_distributed_optimizer", None),
        "ddp_overlap_grad_reduce": getattr(ddp, "overlap_grad_reduce", None),
        "ddp_overlap_param_gather": getattr(ddp, "overlap_param_gather", None),
        "dataset_blend": getattr(dataset, "blend", None),
        "dataset_seq_length": getattr(dataset, "seq_length", None),
        "lr": getattr(optimizer, "lr", None),
        "min_lr": getattr(optimizer, "min_lr", None),
        "adam_eps": getattr(optimizer, "adam_eps", None),
        "lr_warmup_iters": getattr(scheduler, "lr_warmup_iters", None),
        "lr_decay_iters": getattr(scheduler, "lr_decay_iters", None),
        "precision_config": repr(mixed_precision),
        "runtime_env": _runtime_env_summary(),
    }


def build_config(args: argparse.Namespace, spec: CaseSpec) -> Any:
    (
        llama3_8b_pretrain_config,
        llama3_8b_low_precision_pretrain_config,
        _training,
    ) = _import_bridge()

    hf_config_source = args.hf_model_path or REMOTE_LLAMA3_8B
    with remap_llama3_hf_repo(args.hf_model_path):
        if spec.recipe == "bf16":
            cfg = llama3_8b_pretrain_config()
        elif spec.recipe in {
            "bf16_with_nvfp4_mixed",
            "bridge_fp4_mlp",
            "bridge_fp4_projection_mlp",
        }:
            cfg = llama3_8b_low_precision_pretrain_config(
                mixed_precision_recipe="bf16_with_nvfp4_mixed"
            )
        else:  # pragma: no cover - parser constrains this.
            raise ValueError(f"unknown recipe: {spec.recipe}")
    setattr(cfg, "_nvblog_hf_config_source", str(hf_config_source))

    model = _get_required(cfg, "model")
    train = _get_required(cfg, "train")
    dataset = _get_required(cfg, "dataset")
    optimizer = _get_required(cfg, "optimizer")
    scheduler = _get_required(cfg, "scheduler")
    logger = getattr(cfg, "logger", None)
    validation = getattr(cfg, "validation", None)
    checkpoint = getattr(cfg, "checkpoint", None)

    _set_required(model, "seq_length", args.seq_length)
    _set_required(dataset, "seq_length", args.seq_length)
    _set_required(train, "train_iters", args.train_iters)
    _set_required(train, "global_batch_size", args.global_batch_size)
    _set_required(train, "micro_batch_size", spec.micro_batch_size)
    _set_required(scheduler, "lr_warmup_iters", args.lr_warmup_iters)
    _set_if_present(scheduler, "lr_decay_iters", args.lr_decay_iters)
    _set_if_present(optimizer, "lr", args.lr)
    _set_if_present(optimizer, "min_lr", args.min_lr)
    _set_if_present(optimizer, "adam_eps", args.adam_eps)

    _set_required(model, "tensor_model_parallel_size", args.tp)
    _set_required(model, "pipeline_model_parallel_size", args.pp)
    _set_required(model, "context_parallel_size", args.cp)
    _set_required(model, "sequence_parallel", args.sequence_parallel)
    if spec.recipe in {"bridge_fp4_mlp", "bridge_fp4_projection_mlp"}:
        if spec.fp4_backend is None:
            raise ValueError(f"{spec.recipe} requires CaseSpec.fp4_backend")
        from low_bits_training.bridge_mcore_fp4 import (
            fp4_bridge_env,
            install_bridge_fp4_cce_postprocess_patch,
            install_bridge_fp4_ddp_debug_patch,
            install_bridge_fp4_separate_qkv_patch,
            install_bridge_fp4_transformer_block_sync_patch,
            make_fp4_mlp_layer_spec,
            make_fp4_projection_mlp_layer_spec,
        )

        os.environ.update(fp4_bridge_env(spec.fp4_backend))
        if spec.fp4_backend.lower() == "mxfp4":
            os.environ.setdefault("MXFP4_USE_WEIGHT_QUANT_CACHE", "1")
            os.environ.setdefault("MXFP4_USE_SAVED_SIGMOID_FWD_INPLACE_QUANT", "1")
            os.environ.setdefault("MXFP4_FFN_W13_FWD_BATCHED_GEMM_CONFIG_M32768_N7168_K4096", "7")
            if os.environ.get("MXFP4_USE_WEIGHT_QUANT_CACHE", "0") == "1":
                os.environ.setdefault("MXFP4_USE_FFN_FWD_W2_WEIGHT_QUANT_OVERLAP", "0")
        install_bridge_fp4_cce_postprocess_patch()
        install_bridge_fp4_ddp_debug_patch()
        install_bridge_fp4_separate_qkv_patch()
        install_bridge_fp4_transformer_block_sync_patch()
        _set_if_present(model, "cross_entropy_loss_fusion", True)
        _set_if_present(model, "cross_entropy_fusion_impl", "lbt_fp4_cce")
        mixed_precision = getattr(cfg, "mixed_precision", None)
        if mixed_precision is not None:
            for attr, value in (
                ("first_last_layers_bf16", False),
                ("num_layers_at_start_in_bf16", 0),
                ("num_layers_at_end_in_bf16", 0),
                ("fp4_param_gather", False),
            ):
                if hasattr(mixed_precision, attr):
                    setattr(mixed_precision, attr, value)
        ddp = getattr(cfg, "ddp", None)
        if ddp is not None:
            if os.environ.get("LBT_BRIDGE_CHECK_FOR_NAN_IN_GRAD") is None:
                os.environ["LBT_BRIDGE_CHECK_FOR_NAN_IN_GRAD"] = "0"
            use_dist_opt = os.environ.get("LBT_BRIDGE_USE_DISTRIBUTED_OPTIMIZER")
            if use_dist_opt is None:
                use_dist_opt_value = getattr(ddp, "use_distributed_optimizer", True)
            else:
                use_dist_opt_value = use_dist_opt == "1"
            overlap_grad_reduce = os.environ.get("LBT_BRIDGE_OVERLAP_GRAD_REDUCE")
            if overlap_grad_reduce is None:
                overlap_grad_reduce_value = True
            else:
                overlap_grad_reduce_value = overlap_grad_reduce == "1"
            for attr, value in (
                # Keep gradient bucketing enabled by default. Single-GPU 8B
                # memory smokes can opt out with LBT_BRIDGE_OVERLAP_GRAD_REDUCE=0.
                ("overlap_grad_reduce", overlap_grad_reduce_value),
                ("overlap_param_gather", False),
                ("use_distributed_optimizer", use_dist_opt_value),
                ("bucket_size", int(os.environ.get("LBT_BRIDGE_DDP_BUCKET_SIZE", "40000000"))),
            ):
                if hasattr(ddp, attr):
                    setattr(ddp, attr, value)
        if hasattr(optimizer, "use_distributed_optimizer") and ddp is not None:
            setattr(
                optimizer,
                "use_distributed_optimizer",
                getattr(ddp, "use_distributed_optimizer", optimizer.use_distributed_optimizer),
            )
        spec_factory = (
            make_fp4_projection_mlp_layer_spec
            if spec.recipe == "bridge_fp4_projection_mlp"
            else make_fp4_mlp_layer_spec
        )
        model.transformer_layer_spec = (
            lambda provider, fp4_backend=spec.fp4_backend, spec_factory=spec_factory: spec_factory(
                provider, fp4_backend=fp4_backend
            )
        )
        if args.custom_mlp_recompute:
            modules = list(getattr(model, "recompute_modules", None) or [])
            if "mlp" not in modules:
                modules.append("mlp")
            _set_if_present(model, "recompute_modules", modules)
            setattr(cfg, "_nvblog_custom_mlp_recompute", True)
        setattr(cfg, "_nvblog_fp4_backend", spec.fp4_backend)
        setattr(cfg, "_nvblog_fp4_full_fusions", spec.recipe == "bridge_fp4_projection_mlp")

    if args.mock_data:
        _set_required(dataset, "blend", None)
    else:
        if not args.data_path:
            raise SystemExit("--data-path is required unless --mock-data is set.")
        _set_required(dataset, "blend", [(str(args.data_path), 1.0)])

    _set_if_present(dataset, "num_workers", args.num_workers)
    if validation is not None:
        _set_if_present(validation, "eval_interval", args.eval_interval)
        _set_if_present(validation, "eval_iters", args.eval_iters)
    if checkpoint is not None and args.no_checkpoint_save:
        _set_if_present(checkpoint, "save", None)
        _set_if_present(checkpoint, "save_interval", args.train_iters + 1)
    if logger is not None:
        _set_if_present(logger, "log_interval", args.log_interval)
        _set_if_present(logger, "tensorboard_dir", str(args.out_dir / "tensorboard"))
        _set_if_present(logger, "wandb_project", os.environ.get("WANDB_PROJECT"))
        _set_if_present(logger, "wandb_exp_name", spec.name)

    apply_bridge_runtime_overrides(cfg, optimizer)

    _set_if_present(cfg, "name", spec.name)
    _set_if_present(cfg, "dir", str(args.out_dir / spec.name))
    return cfg


def _case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    specs = {
        "bf16": CaseSpec("bridge_bf16", "bf16", args.bf16_micro_batch_size),
        "nvfp4": CaseSpec(
            "bridge_te_nvfp4_f0l4",
            "bf16_with_nvfp4_mixed",
            args.nvfp4_micro_batch_size,
        ),
        "ours_nvfp4_tk_v5_mlp": CaseSpec(
            "bridge_ours_nvfp4_tk_v5_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_tk_v5",
        ),
        "ours_nvfp4_tk_v5_full": CaseSpec(
            "bridge_ours_nvfp4_tk_v5_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_tk_v5",
        ),
        "ours_nvfp4_tk_v5_highwater_mlp": CaseSpec(
            "bridge_ours_nvfp4_tk_v5_highwater_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_tk_v5",
            _bridge_tk_v5_highwater_env(),
        ),
        "ours_nvfp4_tk_v5_highwater_full": CaseSpec(
            "bridge_ours_nvfp4_tk_v5_highwater_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_tk_v5",
            _bridge_tk_v5_highwater_env(),
        ),
        "ours_nvfp4_tk_v5_highwater_delayed_mlp": CaseSpec(
            "bridge_ours_nvfp4_tk_v5_highwater_delayed_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_tk_v5",
            _bridge_tk_v5_highwater_env(delayed=True),
        ),
        "ours_nvfp4_tk_v5_highwater_delayed_full": CaseSpec(
            "bridge_ours_nvfp4_tk_v5_highwater_delayed_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_tk_v5",
            _bridge_tk_v5_highwater_env(delayed=True),
        ),
        "ours_nvfp4_localcta_v4_mlp": CaseSpec(
            "bridge_ours_nvfp4_localcta_v4_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_localcta_v4",
        ),
        "ours_nvfp4_localcta_v4_full": CaseSpec(
            "bridge_ours_nvfp4_localcta_v4_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_localcta_v4",
        ),
        "ours_nvfp4_localcta_v4_tp2_fused_mlp": CaseSpec(
            "bridge_ours_nvfp4_localcta_v4_tp2_fused_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_localcta_v4",
            {"LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE": "tp2_fused_split"},
        ),
        "ours_nvfp4_localcta_v4_tp2_fused_full": CaseSpec(
            "bridge_ours_nvfp4_localcta_v4_tp2_fused_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_localcta_v4",
            {"LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE": "tp2_fused_split"},
        ),
        "ours_nvfp4_localcta_v4_tp2_highwater_mlp": CaseSpec(
            "bridge_ours_nvfp4_localcta_v4_tp2_highwater_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_localcta_v4",
            _bridge_localcta_v4_highwater_env(),
        ),
        "ours_nvfp4_localcta_v4_tp2_highwater_full": CaseSpec(
            "bridge_ours_nvfp4_localcta_v4_tp2_highwater_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "nvfp4_localcta_v4",
            _bridge_localcta_v4_highwater_env(),
        ),
        "ours_mixed_localcta_mxfp4_mlp": CaseSpec(
            "bridge_ours_mixed_localcta_mxfp4_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "mixed_localcta_mxfp4",
        ),
        "ours_mixed_localcta_mxfp4_full": CaseSpec(
            "bridge_ours_mixed_localcta_mxfp4_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "mixed_localcta_mxfp4",
        ),
        "ours_mixed_localcta_mxfp4_tp2_fused_mlp": CaseSpec(
            "bridge_ours_mixed_localcta_mxfp4_tp2_fused_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "mixed_localcta_mxfp4",
            {"LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE": "tp2_fused_split"},
        ),
        "ours_mixed_localcta_mxfp4_tp2_fused_full": CaseSpec(
            "bridge_ours_mixed_localcta_mxfp4_tp2_fused_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "mixed_localcta_mxfp4",
            {"LBT_BRIDGE_LOCALCTA_V4_TP2_PROFILE": "tp2_fused_split"},
        ),
        "ours_mixed_localcta_mxfp4_tp2_highwater_mlp": CaseSpec(
            "bridge_ours_mixed_localcta_mxfp4_tp2_highwater_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "mixed_localcta_mxfp4",
            _bridge_localcta_v4_highwater_env(),
        ),
        "ours_mixed_localcta_mxfp4_tp2_highwater_full": CaseSpec(
            "bridge_ours_mixed_localcta_mxfp4_tp2_highwater_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "mixed_localcta_mxfp4",
            _bridge_localcta_v4_highwater_env(),
        ),
        "ours_mxfp4_mlp": CaseSpec(
            "bridge_ours_mxfp4_mlp",
            "bridge_fp4_mlp",
            args.nvfp4_micro_batch_size,
            "mxfp4",
        ),
        "ours_mxfp4_full": CaseSpec(
            "bridge_ours_mxfp4_full",
            "bridge_fp4_projection_mlp",
            args.nvfp4_micro_batch_size,
            "mxfp4",
        ),
    }
    if args.precision == "both":
        return [specs["bf16"], specs["nvfp4"]]
    return [specs[args.precision]]


def _bridge_trace_step_forward_step(
    forward_step: Any,
    args: argparse.Namespace,
    spec: CaseSpec,
) -> Any:
    """Optionally expose Bridge train-iteration numbers to local timing hooks.

    Megatron Bridge calls ``forward_step`` once per microbatch. The local FP4
    timing hooks key off ``LBT_TRACE_ACTIVE_STEP`` because the non-Bridge trainer
    sets it around each optimizer step. Recreate that marker here only when
    explicitly requested so normal throughput runs are unchanged.
    """

    if os.environ.get("LBT_BRIDGE_SET_TRACE_STEP", "0") != "1":
        return forward_step

    microbatches = max(1, int(args.global_batch_size) // int(spec.micro_batch_size))
    call_idx = 0

    @wraps(forward_step)
    def wrapped_forward_step(*step_args: Any, **step_kwargs: Any) -> Any:
        nonlocal call_idx
        # Bridge invokes this callback during forward, while most local FP4
        # timing hooks emit from autograd backward. Keep the marker live until
        # the next microbatch callback so backward sees the same train step.
        os.environ["LBT_TRACE_ACTIVE_STEP"] = str(call_idx // microbatches + 1)
        call_idx += 1
        return forward_step(*step_args, **step_kwargs)

    return wrapped_forward_step


def _bridge_fp4_callbacks() -> list[Any] | None:
    if os.environ.get("MXFP4_USE_WEIGHT_QUANT_CACHE", "0") != "1":
        return None

    from megatron.bridge.training.callbacks import Callback

    class _FP4WeightQuantCacheInvalidator(Callback):
        @staticmethod
        def _clear(context: Any) -> None:
            try:
                from low_bits_training.quantization.mxfp4_fused_linear import (
                    clear_mxfp4_weight_quant_cache,
                )

                clear_mxfp4_weight_quant_cache()
            except Exception:
                pass

        def on_train_start(self, context: Any) -> None:
            self._clear(context)

        def on_train_step_start(self, context: Any) -> None:
            self._clear(context)

        def on_train_step_end(self, context: Any) -> None:
            self._clear(context)

    return [_FP4WeightQuantCacheInvalidator()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=[
            "bf16",
            "nvfp4",
            "both",
            "ours_nvfp4_tk_v5_mlp",
            "ours_nvfp4_tk_v5_full",
            "ours_nvfp4_tk_v5_highwater_mlp",
            "ours_nvfp4_tk_v5_highwater_full",
            "ours_nvfp4_tk_v5_highwater_delayed_mlp",
            "ours_nvfp4_tk_v5_highwater_delayed_full",
            "ours_nvfp4_localcta_v4_mlp",
            "ours_nvfp4_localcta_v4_full",
            "ours_nvfp4_localcta_v4_tp2_fused_mlp",
            "ours_nvfp4_localcta_v4_tp2_fused_full",
            "ours_nvfp4_localcta_v4_tp2_highwater_mlp",
            "ours_nvfp4_localcta_v4_tp2_highwater_full",
            "ours_mixed_localcta_mxfp4_mlp",
            "ours_mixed_localcta_mxfp4_full",
            "ours_mixed_localcta_mxfp4_tp2_fused_mlp",
            "ours_mixed_localcta_mxfp4_tp2_fused_full",
            "ours_mixed_localcta_mxfp4_tp2_highwater_mlp",
            "ours_mixed_localcta_mxfp4_tp2_highwater_full",
            "ours_mxfp4_mlp",
            "ours_mxfp4_full",
        ],
        default="both",
    )
    parser.add_argument("--train-iters", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=0)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--bf16-micro-batch-size", type=int, default=2)
    parser.add_argument("--nvfp4-micro-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-6)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lr-warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=500)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument(
        "--custom-mlp-recompute",
        action="store_true",
        help=(
            "For local FP4 Bridge MLP adapter cases, also recompute MLP "
            "activations. This is a memory diagnostic, not part of the "
            "default NVIDIA recipe."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/nvblog_bridge_repro"))
    parser.add_argument(
        "--hf-model-path",
        type=Path,
        default=None,
        help=(
            "Optional local HF model/config directory to use in place of the "
            f"gated {REMOTE_LLAMA3_8B} repo. Useful for config-only throughput runs."
        ),
    )
    parser.add_argument(
        "--use-local-config-only",
        action="store_true",
        help=(
            "Use/create a local config-only Llama 3 8B HF directory. This avoids "
            "pulling gated Meta weights/config and is valid only for random-init "
            "throughput reproduction."
        ),
    )
    parser.add_argument("--no-checkpoint-save", action="store_true", default=True)
    parser.add_argument(
        "--allow-checkpoint-save",
        dest="no_checkpoint_save",
        action="store_false",
        help="Allow Bridge recipe checkpoint saves instead of disabling them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-dataset-helper-compile",
        action="store_true",
        help="Do not skip MCore's dataset-helper make step even if helpers_cpp is importable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.use_local_config_only and args.hf_model_path is None:
        args.hf_model_path = DEFAULT_LOCAL_LLAMA3_HF_CONFIG
    if args.hf_model_path is not None:
        ensure_local_llama3_8b_config(args.hf_model_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for spec in _case_specs(args):
        with _scoped_env(spec.env):
            cfg = build_config(args, spec)
            summary = _public_summary(cfg, spec)
            summaries.append(summary)
            case_dir = args.out_dir / spec.name
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "config_summary.json").write_text(
                json.dumps(summary, indent=2, default=str) + "\n"
            )

            print(json.dumps({"case": asdict(spec), "config": summary}, indent=2, default=str))
            if args.dry_run:
                continue

            if not args.force_dataset_helper_compile:
                patch_dataset_helper_compile_if_present()

            validate_bridge_te_fp4_compat(cfg)

            _llama3_8b_pretrain_config, _low_precision, training = _import_bridge()
            pretrain, forward_step = training
            forward_step = _bridge_trace_step_forward_step(forward_step, args, spec)
            pretrain(cfg, forward_step, callbacks=_bridge_fp4_callbacks())

    (args.out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
