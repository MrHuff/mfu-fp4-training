#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os

import torch
import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import build_optimizers as tt_build_optimizers
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.config import Optimizer as OptimizerConfig

from torchtitan.distributed import ParallelDims


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
):
    """Wrapping TorchTitan optimizer factory method, keeping track of the optimizer."""
    # Circular import because of umup.__init__.py structure.
    from .umup.umup_optimizer import create_umup_adamw
    from .quantization.stable_spam import create_stable_spam

    try:
        optimizer = tt_build_optimizers(
            model_parts, optimizer_config, parallel_dims, ft_manager
        )
    except NotImplementedError:
        # Copy and paste old build_optimizers code
        # TT now hardcodes a dict of supported optimizers in build_optimizers
        optim_in_bwd = optimizer_config.early_step_in_backward
        if optim_in_bwd and parallel_dims.pp > 1:
            raise NotImplementedError(
                "Optimizers in backward is not supported with pipeline parallelism."
            )
        name = optimizer_config.name
        lr = optimizer_config.lr
        optim_implementation = optimizer_config.implementation
        assert optim_implementation in ["fused", "foreach", "for-loop"]

        fused = optim_implementation == "fused"
        foreach = optim_implementation == "foreach"
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": foreach,
        }

        # Ugh
        if name == "UmupAdamW":
            model_cls = create_umup_adamw
        elif name == "StableSPAM":
            model_cls = create_stable_spam
        else:
            raise NotImplementedError

        optimizer = (
            OptimizersContainer(model_parts, model_cls, optimizer_kwargs)
            if not optim_in_bwd
            else OptimizersInBackwardContainer(model_parts, model_cls, optimizer_kwargs)
        )
    # Caching for logging purposes.
    return optimizer


def build_optimizers_with_moe_load_balancing(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
):
    """Build LBT optimizers and preserve TorchTitan's MoE expert-bias update.

    TorchTitan's DeepSeek registration uses an optimizer pre-hook to update the
    auxiliary-loss-free expert bias from tokens-per-expert statistics. The LBT
    DeepSeek registration needs the same hook; otherwise early routing skew can
    dominate EP all-to-all time and hide local MXFP4 compute wins.
    """

    optimizer = build_optimizers(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            for transformer_block in model_part.layers.values():
                if transformer_block.moe_enabled:
                    return bool(transformer_block.moe.load_balance_coeff)
        return False

    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _load_balance_coeff(moe) -> float:
        override = os.environ.get("LBT_MOE_LOAD_BALANCE_COEFF", "").strip()
        if override:
            return float(override)
        return float(moe.load_balance_coeff)

    def _ep_group_balance_coeff() -> float:
        override = os.environ.get("LBT_MOE_EP_GROUP_BALANCE_COEFF", "").strip()
        return float(override) if override else 0.0

    def _ep_group_balance_mode() -> str:
        mode = os.environ.get("LBT_MOE_EP_GROUP_BALANCE_MODE", "proportional").strip().lower()
        if mode not in ("proportional", "sign"):
            raise ValueError(
                "Unsupported LBT_MOE_EP_GROUP_BALANCE_MODE="
                f"{mode}. Expected proportional or sign."
            )
        return mode

    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    def _dist_rank() -> int:
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return torch.distributed.get_rank()
        return 0

    def _moe_balance_reduce_mode() -> str:
        mode = os.environ.get("LBT_MOE_BALANCE_REDUCE", "dp_cp").strip().lower()
        if mode in ("", "default"):
            return "dp_cp"
        if mode not in ("dp_cp", "world", "none"):
            raise ValueError(
                "Unsupported LBT_MOE_BALANCE_REDUCE="
                f"{mode}. Expected dp_cp, world, or none."
            )
        return mode

    def _reduce_tokens_per_expert(
        tokens_per_expert_by_layer: torch.Tensor,
        dp_cp_mesh,
    ) -> tuple[str, int]:
        mode = _moe_balance_reduce_mode()
        if (
            mode == "none"
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return mode, 1

        if mode == "world":
            torch.distributed.all_reduce(
                tokens_per_expert_by_layer,
                op=torch.distributed.ReduceOp.SUM,
            )
            return mode, torch.distributed.get_world_size()

        if dp_cp_mesh is None:
            return "none", 1

        pg = dp_cp_mesh.get_group()
        torch.distributed.all_reduce(
            tokens_per_expert_by_layer,
            group=pg,
            op=torch.distributed.ReduceOp.SUM,
        )
        return mode, torch.distributed.get_world_size(pg)

    def _format_topk(counts: torch.Tensor, largest: bool, k: int = 4) -> str:
        k = min(k, counts.numel())
        if k == 0:
            return "[]"
        values, indices = torch.topk(counts, k=k, largest=largest)
        return "[" + ",".join(
            f"{int(idx.item())}:{float(value.item()):.0f}"
            for idx, value in zip(indices, values)
        ) + "]"

    def _debug_moe_balance_layer(
        *,
        step: int,
        model_part_idx: int,
        block_name,
        moe_layer_idx: int,
        tokens_per_expert: torch.Tensor,
        expert_bias: torch.Tensor | None,
        coeff: float,
        ep_group_coeff: float,
        reduce_mode: str,
        reduce_world_size: int,
    ) -> None:
        if not _env_flag("LBT_MOE_BALANCE_DEBUG"):
            return
        limit = _env_int("LBT_MOE_BALANCE_DEBUG_LIMIT_STEPS", 4)
        if limit >= 0 and step >= limit:
            return
        every = max(_env_int("LBT_MOE_BALANCE_DEBUG_EVERY", 1), 1)
        if step % every != 0:
            return
        if _env_flag("LBT_MOE_BALANCE_DEBUG_RANK0_ONLY", True) and _dist_rank() != 0:
            return

        counts = tokens_per_expert.detach().float().cpu()
        if counts.numel() == 0:
            return

        total = float(counts.sum().item())
        mean = float(counts.mean().item())
        minimum = float(counts.min().item())
        maximum = float(counts.max().item())
        std = float(counts.std(unbiased=False).item())
        skew = maximum / mean if mean > 0 else 0.0
        cv = std / mean if mean > 0 else 0.0

        ep_degree = _env_int(
            "LBT_MOE_BALANCE_EP_DEGREE",
            int(parallel_dims.ep) if parallel_dims.ep > 1 else 0,
        )
        ep_summary = ""
        if ep_degree > 1 and counts.numel() % ep_degree == 0:
            ep_totals = counts.view(ep_degree, counts.numel() // ep_degree).sum(dim=1)
            ep_mean = float(ep_totals.mean().item())
            ep_max = float(ep_totals.max().item())
            ep_skew = ep_max / ep_mean if ep_mean > 0 else 0.0
            ep_values = ",".join(f"{float(value.item()):.0f}" for value in ep_totals)
            ep_summary = (
                f" ep_degree={ep_degree} ep_skew={ep_skew:.3f}"
                f" ep_totals=[{ep_values}]"
            )

        bias_summary = ""
        if expert_bias is not None:
            bias = expert_bias.detach().float().cpu()
            bias_summary = (
                f" bias_min={float(bias.min().item()):.4f}"
                f" bias_max={float(bias.max().item()):.4f}"
                f" bias_std={float(bias.std(unbiased=False).item()):.4f}"
            )

        print(
            "[lbt_moe_balance]"
            f" step={step}"
            f" rank={_dist_rank()}"
            f" model_part={model_part_idx}"
            f" block={block_name}"
            f" moe_layer={moe_layer_idx}"
            f" reduce={reduce_mode}"
            f" reduce_world_size={reduce_world_size}"
            f" coeff={coeff:g}"
            f" ep_group_coeff={ep_group_coeff:g}"
            f" total={total:.0f}"
            f" mean={mean:.1f}"
            f" min={minimum:.0f}"
            f" max={maximum:.0f}"
            f" skew={skew:.3f}"
            f" cv={cv:.3f}"
            f"{ep_summary}"
            f" top={_format_topk(counts, largest=True)}"
            f" bottom={_format_topk(counts, largest=False)}"
            f"{bias_summary}",
            flush=True,
        )

    balance_debug_step = 0

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        nonlocal balance_debug_step

        dp_cp_mesh = (
            parallel_dims.world_mesh["dp_cp"] if parallel_dims.dp_cp_enabled else None
        )

        tokens_per_expert_entries = []
        for model_part_idx, model_part in enumerate(model_parts):
            for block_name, transformer_block in model_part.layers.items():
                if not transformer_block.moe_enabled:
                    continue
                if transformer_block.moe.load_balance_coeff is None:
                    return
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_entries.append(
                    (model_part_idx, block_name, transformer_block.moe, tokens_per_expert)
                )

        if not tokens_per_expert_entries:
            return

        tokens_per_expert_by_layer = torch.vstack(
            [entry[3] for entry in tokens_per_expert_entries]
        )

        reduce_mode, reduce_world_size = _reduce_tokens_per_expert(
            tokens_per_expert_by_layer,
            dp_cp_mesh,
        )

        with torch.no_grad():
            for moe_layer_idx, (
                model_part_idx,
                block_name,
                moe,
                _,
            ) in enumerate(tokens_per_expert_entries):
                tokens_per_expert = tokens_per_expert_by_layer[moe_layer_idx].float()
                coeff = _load_balance_coeff(moe)
                ep_group_coeff = _ep_group_balance_coeff()
                _debug_moe_balance_layer(
                    step=balance_debug_step,
                    model_part_idx=model_part_idx,
                    block_name=block_name,
                    moe_layer_idx=moe_layer_idx,
                    tokens_per_expert=tokens_per_expert,
                    expert_bias=moe.expert_bias,
                    coeff=coeff,
                    ep_group_coeff=ep_group_coeff,
                    reduce_mode=reduce_mode,
                    reduce_world_size=reduce_world_size,
                )

                expert_bias_delta = coeff * torch.sign(
                    tokens_per_expert.mean() - tokens_per_expert
                )
                expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                ep_group_degree = _env_int(
                    "LBT_MOE_EP_GROUP_BALANCE_DEGREE",
                    int(parallel_dims.ep) if parallel_dims.ep > 1 else 0,
                )
                if (
                    ep_group_coeff != 0.0
                    and ep_group_degree > 1
                    and tokens_per_expert.numel() % ep_group_degree == 0
                ):
                    experts_per_group = tokens_per_expert.numel() // ep_group_degree
                    group_counts = tokens_per_expert.view(
                        ep_group_degree,
                        experts_per_group,
                    ).sum(dim=1)
                    if _ep_group_balance_mode() == "sign":
                        group_bias_delta = ep_group_coeff * torch.sign(
                            group_counts.mean() - group_counts
                        )
                    else:
                        group_bias_delta = ep_group_coeff * torch.clamp(
                            (group_counts.mean() - group_counts)
                            / torch.clamp(group_counts.mean(), min=1.0),
                            min=-1.0,
                            max=1.0,
                        )
                    group_bias_delta = group_bias_delta - group_bias_delta.mean()
                    expert_bias_delta = expert_bias_delta + group_bias_delta.repeat_interleave(
                        experts_per_group
                    )
                moe.expert_bias.add_(expert_bias_delta)
                moe.tokens_per_expert.zero_()

        balance_debug_step += 1

    if _should_register_moe_balancing_hook(model_parts):
        optimizer.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )

    return optimizer
