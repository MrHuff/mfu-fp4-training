"""Shared helpers for FP4 CCE v4 prototypes."""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.cpp_extension import load


_CHECKPOINTED_OUTPUT_HEAD_SR_STATE: torch.Tensor | None = None


def set_checkpointed_output_head_sr_state(state: torch.Tensor | None) -> None:
    """Install the persistent fused output-head row-SR state.

    The owning training runtime checkpoints this tensor. Keeping the setter in
    the Python wrapper avoids coupling the CUDA extension to a particular
    trainer/checkpointer implementation while retaining a single explicit
    tensor ABI at the kernel boundary.
    """
    if state is not None:
        if (
            not torch.is_tensor(state)
            or state.dtype != torch.int64
            or state.ndim != 1
            or state.numel() != 2
            or not state.is_contiguous()
        ):
            raise ValueError(
                "checkpointed output-head SR state must be contiguous int64[2]"
            )
    global _CHECKPOINTED_OUTPUT_HEAD_SR_STATE
    _CHECKPOINTED_OUTPUT_HEAD_SR_STATE = state


def _checkpointed_output_head_sr_state_for(
    reference: torch.Tensor,
) -> torch.Tensor | None:
    enabled = os.environ.get("FP4_CCE_V4_CHECKPOINTED_HEAD_SR", "0") != "0"
    state = _CHECKPOINTED_OUTPUT_HEAD_SR_STATE
    if not enabled:
        if state is not None:
            raise RuntimeError(
                "checkpointed output-head SR state is installed but "
                "FP4_CCE_V4_CHECKPOINTED_HEAD_SR is disabled"
            )
        return None
    if state is None:
        raise RuntimeError(
            "FP4_CCE_V4_CHECKPOINTED_HEAD_SR=1 but no output-head SR state "
            "was installed"
        )
    if state.device != reference.device:
        raise RuntimeError(
            "checkpointed output-head SR state device mismatch: "
            f"state={state.device}, reference={reference.device}"
        )
    return state


def _use_compile_helpers() -> bool:
    return os.environ.get("FP4_CCE_V4_COMPILE_HELPERS", "1") != "0"


def use_direct_mxfp4_p_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_DIRECT_MXFP4_P_CACHE", "1") != "0"


def _direct_mxfp4_p_cache_mode() -> str:
    return os.environ.get("FP4_CCE_V4_DIRECT_MXFP4_P_CACHE_MODE", "auto").strip().lower()


def _stage_direct_mxfp4_exp() -> bool:
    return os.environ.get("FP4_CCE_V4_DIRECT_MXFP4_STAGE_EXP", "0") != "0"


def _mxfp4_p_constant_scale() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_P_CONSTANT_SCALE", "0") != "0"


def use_direct_nvfp4_softmax() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_DIRECT_SOFTMAX", "0") != "0"


def use_staged_nvfp4_p_cache() -> bool:
    return (
        os.environ.get("FP4_CCE_V4_NVFP4_STAGED_P_CACHE", "1") != "0"
        or use_direct_nvfp4_softmax()
    )


def use_tiled_nvfp4_p_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_TILED_P_CACHE", "0") != "0"


def use_tma_nvfp4_p_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_TMA_P_CACHE", "0") != "0"


def _use_direct_lse_producer() -> bool:
    return os.environ.get("FP4_CCE_V4_DIRECT_LSE_PRODUCER", "1") != "0"


def assume_all_valid_full_vocab(logits: torch.Tensor, vocab_size: int) -> bool:
    return (
        os.environ.get("FP4_CCE_V4_SOFTMAX_ASSUME_ALL_VALID_FULL_VOCAB", "0") != "0"
        and int(vocab_size) == int(logits.shape[1])
    )


@torch.compile(fullgraph=False, options={"triton.cudagraphs": False})
def _loss_and_probs_nomask_default_ignore(logits: torch.Tensor, targets: torch.Tensor):
    logits_f = logits.float()
    loss = F.cross_entropy(logits_f, targets, ignore_index=-100)
    probs = torch.softmax(logits_f, dim=-1).to(torch.bfloat16)
    return loss, probs


@torch.compile(fullgraph=False, options={"triton.cudagraphs": False})
def _loss_and_probs_masked_default_ignore(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
):
    logits_f = logits.float()
    loss = F.cross_entropy(logits_f, targets, ignore_index=-100)
    probs = torch.softmax(logits_f, dim=-1).to(torch.bfloat16)
    return loss, probs.masked_fill(~valid[:, None], 0)


@torch.compile(fullgraph=False, options={"triton.cudagraphs": False})
def _loss_and_probs_nomask(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int):
    logits_f = logits.float()
    loss = F.cross_entropy(logits_f, targets, ignore_index=ignore_index)
    probs = torch.softmax(logits_f, dim=-1).to(torch.bfloat16)
    return loss, probs


@torch.compile(fullgraph=False, options={"triton.cudagraphs": False})
def _loss_and_probs_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int,
):
    logits_f = logits.float()
    loss = F.cross_entropy(logits_f, targets, ignore_index=ignore_index)
    probs = torch.softmax(logits_f, dim=-1).to(torch.bfloat16)
    return loss, probs.masked_fill(~valid[:, None], 0)


def loss_and_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int,
):
    if _use_compile_helpers():
        if ignore_index == -100:
            if bool(valid.all()):
                return _loss_and_probs_nomask_default_ignore(logits, targets)
            return _loss_and_probs_masked_default_ignore(logits, targets, valid)
        if bool(valid.all()):
            return _loss_and_probs_nomask(logits, targets, ignore_index)
        return _loss_and_probs_masked(logits, targets, valid, ignore_index)

    logits_f = logits.float()
    loss = F.cross_entropy(logits_f, targets, ignore_index=ignore_index)
    probs = torch.softmax(logits_f, dim=-1).to(torch.bfloat16)
    if not bool(valid.all()):
        probs = probs.clone()
        probs[~valid] = 0
    return loss, probs


@torch.compile(fullgraph=False, options={"triton.cudagraphs": False})
def _loss_and_lse_nomask(logits: torch.Tensor, targets: torch.Tensor):
    logits_f = logits.float()
    lse = logits_f.logsumexp(dim=-1)
    target_logits = logits_f.gather(1, targets[:, None]).squeeze(1)
    loss = (lse - target_logits).mean()
    return loss, lse


@torch.compile(fullgraph=False, options={"triton.cudagraphs": False})
def _loss_and_lse_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
):
    logits_f = logits.float()
    lse = logits_f.logsumexp(dim=-1)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    target_logits = logits_f.gather(1, safe_targets[:, None]).squeeze(1)
    denom = valid.sum().clamp(min=1)
    loss = ((lse - target_logits) * valid).sum() / denom
    return loss, lse


def loss_and_lse(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
):
    if _use_compile_helpers():
        if bool(valid.all()):
            return _loss_and_lse_nomask(logits, targets)
        return _loss_and_lse_masked(logits, targets, valid)

    logits_f = logits.float()
    lse = logits_f.logsumexp(dim=-1)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    target_logits = logits_f.gather(1, safe_targets[:, None]).squeeze(1)
    denom = valid.sum().clamp(min=1)
    loss = ((lse - target_logits) * valid).sum() / denom
    return loss, lse


@functools.lru_cache(maxsize=1)
def _load_mxfp4_softmax_quant():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(this_dir, "v4_mxfp4_softmax_quant.cu")
    build_dir = os.environ.get("FP4_CCE_V4_MXFP4_EXT_BUILD_DIR", "/tmp/fp4_cce_v4_mxfp4_softmax_quant")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="fp4_cce_v4_mxfp4_softmax_quant",
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        verbose=os.environ.get("FP4_CCE_V4_EXT_VERBOSE", "0") == "1",
    )


def mxfp4_direct_p_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    if not use_direct_mxfp4_p_cache():
        raise RuntimeError("direct MXFP4 P-cache disabled")
    if _direct_mxfp4_p_cache_mode() in ("auto", "staged"):
        loss, row_fp4, row_sc, col_fp4, col_sc = (
            _load_mxfp4_softmax_quant().mxfp4_softmax_quant_row_col_staged(
                logits,
                targets.contiguous(),
                valid.contiguous(),
                int(vocab_size),
                _stage_direct_mxfp4_exp(),
                _mxfp4_p_constant_scale(),
            )
        )
        return loss, row_fp4, row_sc, col_fp4, col_sc

    loss, lse = p_cache_loss_and_lse(logits, targets, valid, vocab_size)
    row_fp4, row_sc, col_fp4, col_sc = _load_mxfp4_softmax_quant().mxfp4_softmax_quant_row_col(
        logits,
        lse.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )
    return loss, row_fp4, row_sc, col_fp4, col_sc


def mxfp4_staged_g_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    loss, row_fp4, row_sc, col_fp4, col_sc = (
        _load_mxfp4_softmax_quant().mxfp4_softmax_grad_quant_row_col_staged(
            logits,
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            _stage_direct_mxfp4_exp(),
        )
    )
    return loss, row_fp4, row_sc, col_fp4, col_sc


def mxfp4_tiled_g_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    loss, lse = p_cache_loss_and_lse(logits, targets, valid, vocab_size)
    row_fp4, row_sc, col_fp4, col_sc = (
        _load_mxfp4_softmax_quant().mxfp4_softmax_grad_quant_row_col_tiled(
            logits,
            lse.contiguous(),
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
        )
    )
    return loss, row_fp4, row_sc, col_fp4, col_sc


def mxfp4_tiled_g_cache_target_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int = 0,
    *,
    x: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    exact_selected_logits: bool = False,
):
    """Emit an MXFP4 softmax tail after masking exact target/top-k entries."""
    if exact_selected_logits and (x is None or weight is None):
        raise ValueError("exact selected-logit repair requires BF16 x and weight")
    (
        loss,
        lse,
        target_probs,
        topk_probs,
        topk_indices,
    ) = (
        direct_loss_lse_target_topk_split_exact_logits(
            logits,
            x,
            weight,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
        if exact_selected_logits
        else direct_loss_lse_target_topk_split(
            logits,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
    )
    if topk_split == 0:
        topk_probs = None
        topk_indices = None

    row_fp4, row_sc, col_fp4, col_sc = mxfp4_softmax_tail_quant_row_col(
        logits,
        lse,
        valid,
        vocab_size,
    )
    return (
        loss,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        target_probs,
        topk_indices,
        topk_probs,
    )


def mxfp4_softmax_tail_quant_row_col(
    logits: torch.Tensor,
    lse: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    """Quantize a selected-masked softmax tail to row/column MXFP4."""
    scale_floor_ratio = float(
        os.environ.get("FP4_CCE_V4_MXFP4_G_SCALE_FLOOR_RATIO", "1.125")
    )
    if not 1.0 <= scale_floor_ratio <= 2.0:
        raise ValueError(
            "FP4_CCE_V4_MXFP4_G_SCALE_FLOOR_RATIO must be in [1, 2]"
        )
    return (
        _load_mxfp4_softmax_quant()
        .mxfp4_softmax_quant_row_col_with_floor_ratio(
            logits,
            lse.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            scale_floor_ratio,
        )
    )


def mxfp4_col_requant_from_row(
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    row_normalization: torch.Tensor,
):
    scale_floor_ratio = float(
        os.environ.get("FP4_CCE_V4_MXFP4_G_SCALE_FLOOR_RATIO", "1.125")
    )
    if not 1.0 <= scale_floor_ratio <= 2.0:
        raise ValueError(
            "FP4_CCE_V4_MXFP4_G_SCALE_FLOOR_RATIO must be in [1, 2]"
        )
    return _load_mxfp4_softmax_quant().mxfp4_col_requant_from_row(
        row_fp4,
        row_sc,
        row_normalization.contiguous(),
        scale_floor_ratio,
    )


@functools.lru_cache(maxsize=1)
def _load_softmax_probs():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(this_dir, "v4_softmax_probs.cu")
    build_dir = os.environ.get("FP4_CCE_V4_SOFTMAX_EXT_BUILD_DIR", "/tmp/fp4_cce_v4_softmax_probs")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="fp4_cce_v4_softmax_probs",
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_ldflags=["-lcublas"],
        verbose=os.environ.get("FP4_CCE_V4_EXT_VERBOSE", "0") == "1",
    )


def direct_loss_and_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    return _load_softmax_probs().softmax_loss_probs(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def replace_target_logits_bf16(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    vocab_size: int,
) -> None:
    _load_softmax_probs().replace_target_logits_bf16(
        logits,
        x,
        weight,
        targets.contiguous(),
        int(ignore_index),
        int(vocab_size),
    )


def direct_loss_and_probs_target_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    return _load_softmax_probs().softmax_loss_probs_target_split(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def direct_loss_and_probs_target_top1_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    return _load_softmax_probs().softmax_loss_probs_target_top1_split(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def direct_loss_and_probs_target_top2_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    return _load_softmax_probs().softmax_loss_probs_target_top2_split(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def direct_loss_and_probs_target_top4_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    return _load_softmax_probs().softmax_loss_probs_target_top4_split(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def direct_loss_and_grad_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    if assume_all_valid_full_vocab(logits, vocab_size):
        return _load_softmax_probs().softmax_loss_grad_probs_all_valid_full_vocab(
            logits,
            targets.contiguous(),
            int(vocab_size),
        )
    return _load_softmax_probs().softmax_loss_grad_probs(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def bf16_logits_cuda(hidden: torch.Tensor, weight: torch.Tensor):
    return _load_softmax_probs().bf16_logits_cuda(
        hidden.contiguous(),
        weight.contiguous(),
    )


def bf16_tail_grads_cuda(
    grad_logits: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    valid_count: torch.Tensor,
):
    return _load_softmax_probs().bf16_tail_grads_cuda(
        grad_logits.contiguous(),
        hidden.contiguous(),
        weight.contiguous(),
        grad_output.contiguous(),
        valid_count.contiguous(),
    )


def valid_mask_count_cuda(targets: torch.Tensor, ignore_index: int):
    return _load_softmax_probs().valid_mask_count_cuda(
        targets.contiguous(),
        int(ignore_index),
    )


def backward_scale_cuda(
    grad_output: torch.Tensor,
    valid_count: torch.Tensor,
):
    return _load_softmax_probs().backward_scale_cuda(
        grad_output.contiguous(),
        valid_count.contiguous(),
    )


def use_vocab_parallel_direct_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_VOCAB_PARALLEL_DIRECT_G_CACHE", "1") != "0"


def use_mxfp4_vocab_parallel_direct_g_cache() -> bool:
    value = os.environ.get("FP4_CCE_V4_MXFP4_VOCAB_PARALLEL_DIRECT_G_CACHE")
    if value is not None:
        return value != "0"
    return use_vocab_parallel_direct_g_cache()


def use_nvfp4_vocab_parallel_direct_g_cache() -> bool:
    value = os.environ.get("FP4_CCE_V4_NVFP4_VOCAB_PARALLEL_DIRECT_G_CACHE")
    if value is not None:
        return value != "0"
    value = os.environ.get("FP4_CCE_V4_VOCAB_PARALLEL_DIRECT_G_CACHE")
    if value is not None:
        return value != "0"
    return False


def use_vocab_parallel_cuda_grad_probs() -> bool:
    return os.environ.get("FP4_CCE_V4_VOCAB_PARALLEL_CUDA_GRAD_PROBS", "1") != "0"


def use_vocab_parallel_cuda_lse() -> bool:
    return os.environ.get("FP4_CCE_V4_VOCAB_PARALLEL_CUDA_LSE", "1") != "0"


def use_mxfp4_vocab_parallel_cuda_lse_targets() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_VOCAB_PARALLEL_CUDA_LSE_TARGETS", "1") != "0"


def use_nvfp4_vocab_parallel_cuda_lse_targets() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_VOCAB_PARALLEL_CUDA_LSE_TARGETS", "1") != "0"


def vocab_parallel_loss_lse_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_start: int,
    global_vocab_size: int,
    tp_group,
    *,
    fuse_cuda_targets: bool = False,
):
    """Return vocab-parallel CE loss plus local-shard LSE inputs.

    `targets_for_quant` uses local vocab indices only for rows whose label is
    owned by this shard; other valid rows get -1 so the local G-cache producer
    emits softmax probabilities without subtracting the one-hot term.
    """
    local_vocab_size = int(logits.shape[1])
    vocab_start = int(vocab_start)
    global_vocab_size = int(global_vocab_size)
    local_valid_cols = max(min(global_vocab_size - vocab_start, local_vocab_size), 0)

    have_target_payload = False
    if (
        logits.is_cuda
        and use_vocab_parallel_cuda_lse()
        and fuse_cuda_targets
        and local_valid_cols > 0
    ):
        local_lse, target_logits, targets_for_quant = _load_softmax_probs().softmax_lse_targets(
            logits,
            targets.contiguous(),
            valid.contiguous(),
            vocab_start,
            global_vocab_size,
            int(local_valid_cols),
        )
        have_target_payload = True
    elif logits.is_cuda and use_vocab_parallel_cuda_lse() and local_valid_cols > 0:
        local_lse = _load_softmax_probs().softmax_lse(
            logits,
            valid.contiguous(),
            int(local_valid_cols),
        )
    elif local_valid_cols > 0:
        local_lse = torch.logsumexp(logits.float()[:, :local_valid_cols], dim=1)
    else:
        local_lse = torch.empty((logits.shape[0],), dtype=torch.float32, device=logits.device)
        local_lse.fill_(-float("inf"))

    if not have_target_payload:
        local_targets = targets.to(torch.long) - vocab_start
        in_range = (
            valid
            & (local_targets >= 0)
            & (local_targets < local_valid_cols)
            & (targets < global_vocab_size)
        )
        targets_for_quant = torch.where(
            in_range,
            local_targets,
            torch.full_like(local_targets, -1),
        )
        target_logits = torch.empty((logits.shape[0],), dtype=torch.float32, device=logits.device)
        target_logits.fill_(-float("inf"))
        rows = torch.where(in_range)[0]
        target_logits[rows] = logits[rows, local_targets[rows]].float()

    use_dist = (
        tp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=tp_group) > 1
    )
    target_reduce = None
    if use_dist:
        target_reduce = dist.all_reduce(
            target_logits, op=dist.ReduceOp.MAX, group=tp_group, async_op=True
        )

    lse_base = local_lse.clone()
    if use_dist:
        dist.all_reduce(lse_base, op=dist.ReduceOp.MAX, group=tp_group)
        lse_weight = torch.exp(local_lse - lse_base)
        global_lse_weight = lse_weight.clone()
        dist.all_reduce(global_lse_weight, op=dist.ReduceOp.SUM, group=tp_group)
        global_lse = lse_base + torch.log(global_lse_weight)
    else:
        global_lse = local_lse

    if target_reduce is not None:
        target_reduce.wait()

    denom = valid.sum().clamp(min=1)
    row_loss = torch.where(valid, global_lse - target_logits, torch.zeros_like(global_lse))
    loss = row_loss.sum() / denom
    return loss, global_lse, targets_for_quant, valid, local_valid_cols


def vocab_parallel_loss_and_grad_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_start: int,
    global_vocab_size: int,
    tp_group,
):
    """Correct vocab-parallel CE over local logits.

    The local FP4 CCE kernels assume a full local vocabulary. Bridge TP output
    weights are vocab-sharded, so the CE denominator and target logit must be
    reduced across TP ranks before producing the local softmax gradient cache.
    """
    if logits.is_cuda and use_vocab_parallel_cuda_grad_probs():
        loss, lse, local_targets, valid, local_valid_cols = vocab_parallel_loss_lse_targets(
            logits,
            targets,
            valid,
            vocab_start,
            global_vocab_size,
            tp_group,
            fuse_cuda_targets=use_nvfp4_vocab_parallel_cuda_lse_targets(),
        )
        grad_probs = _load_softmax_probs().softmax_grad_probs_from_lse(
            logits,
            lse.contiguous(),
            local_targets.contiguous(),
            valid.contiguous(),
            int(local_valid_cols),
        )
        return loss, grad_probs

    logits_f = logits.float()
    local_vocab_size = int(logits_f.shape[1])
    vocab_start = int(vocab_start)
    global_vocab_size = int(global_vocab_size)
    local_valid_cols = max(min(global_vocab_size - vocab_start, local_vocab_size), 0)
    if local_valid_cols < local_vocab_size:
        logits_f = logits_f.clone()
        logits_f[:, local_valid_cols:] = -float("inf")

    local_max = logits_f.max(dim=1).values
    global_max = local_max.clone()
    use_dist = (
        tp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=tp_group) > 1
    )
    if use_dist:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

    local_targets = targets.to(torch.long) - vocab_start
    in_range = (
        valid
        & (local_targets >= 0)
        & (local_targets < local_valid_cols)
        & (targets < global_vocab_size)
    )
    target_logits = logits_f.new_full((logits_f.shape[0],), -float("inf"))
    if bool(in_range.any()):
        rows = torch.where(in_range)[0]
        target_logits[rows] = logits_f[rows, local_targets[rows]]
    target_reduce = None
    if use_dist:
        target_reduce = dist.all_reduce(
            target_logits, op=dist.ReduceOp.MAX, group=tp_group, async_op=True
        )

    local_exp = torch.exp(logits_f - global_max[:, None])
    local_den = local_exp.sum(dim=1)
    global_den = local_den.clone()
    den_reduce = None
    if use_dist:
        den_reduce = dist.all_reduce(global_den, op=dist.ReduceOp.SUM, group=tp_group, async_op=True)
    if den_reduce is not None:
        den_reduce.wait()
    global_lse = global_max + torch.log(global_den)
    if target_reduce is not None:
        target_reduce.wait()

    denom = valid.sum().clamp(min=1)
    row_loss = torch.where(valid, global_lse - target_logits, torch.zeros_like(global_lse))
    loss = row_loss.sum() / denom

    grad_probs = local_exp / global_den[:, None]
    if bool((~valid).any()):
        grad_probs = grad_probs.masked_fill(~valid[:, None], 0.0)
    if bool(in_range.any()):
        rows = torch.where(in_range)[0]
        grad_probs[rows, local_targets[rows]] -= 1.0
    return loss, grad_probs.to(torch.bfloat16)


def mxfp4_vocab_parallel_tiled_g_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_start: int,
    global_vocab_size: int,
    tp_group,
):
    loss, lse, local_targets, valid, local_valid_cols = vocab_parallel_loss_lse_targets(
        logits,
        targets,
        valid,
        vocab_start,
        global_vocab_size,
        tp_group,
        fuse_cuda_targets=use_mxfp4_vocab_parallel_cuda_lse_targets(),
    )
    row_fp4, row_sc, col_fp4, col_sc = (
        _load_mxfp4_softmax_quant().mxfp4_softmax_grad_quant_row_col_tiled(
            logits,
            lse.contiguous(),
            local_targets.contiguous(),
            valid.contiguous(),
            int(local_valid_cols),
        )
    )
    return loss, row_fp4, row_sc, col_fp4, col_sc


def direct_loss_and_lse(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    return _load_softmax_probs().softmax_loss_lse(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def direct_loss_lse_target_topk_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int,
):
    return _load_softmax_probs().softmax_loss_lse_target_topk_split(
        logits,
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
        int(topk_split),
    )


def direct_loss_lse_target_topk_split_exact_logits(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int,
):
    return _load_softmax_probs().softmax_loss_lse_target_topk_split_exact_logits(
        logits,
        x.contiguous(),
        weight.contiguous(),
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
        int(topk_split),
    )


def direct_loss_lse_target_topk_split_exact_logits_nvfp4_row(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int,
    global_scale_max: float,
    exact_selected_logits: bool = True,
):
    return (
        _load_softmax_probs()
        .softmax_loss_lse_target_topk_split_exact_logits_nvfp4_row(
            logits,
            x.contiguous(),
            weight.contiguous(),
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            int(topk_split),
            float(global_scale_max),
            bool(exact_selected_logits),
        )
    )


def direct_loss_lse_target_topk_split_exact_logits_mxfp4_row(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int,
    scale_floor_ratio: float,
    exact_selected_logits: bool = True,
):
    return (
        _load_softmax_probs()
        .softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row(
            logits,
            x.contiguous(),
            weight.contiguous(),
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            int(topk_split),
            float(scale_floor_ratio),
            bool(exact_selected_logits),
        )
    )


def direct_loss_lse_target_topk_split_exact_logits_mxfp4_row_centered(
    logit_residuals: torch.Tensor,
    logit_centers: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int,
    scale_floor_ratio: float,
    exact_selected_logits: bool = True,
):
    return (
        _load_softmax_probs()
        .softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_centered(
            logit_residuals,
            logit_centers,
            x.contiguous(),
            weight.contiguous(),
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            int(topk_split),
            float(scale_floor_ratio),
            bool(exact_selected_logits),
        )
    )


def direct_loss_lse_target_topk_split_exact_logits_mxfp8_row(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int,
    quant_max: float,
    exact_selected_logits: bool = True,
):
    args = (
        logits,
        x.contiguous(),
        weight.contiguous(),
        targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
        int(topk_split),
        float(quant_max),
        bool(exact_selected_logits),
    )
    state = _checkpointed_output_head_sr_state_for(logits)
    producer = (
        _load_softmax_probs()
        .softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row
    )
    if state is None:
        return producer(*args)
    producer_with_state = getattr(
        _load_softmax_probs(),
        "softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row_with_sr_state",
        None,
    )
    if producer_with_state is None:
        raise RuntimeError(
            "checkpointed output-head SR requires the explicit fused-row state ABI"
        )
    return producer_with_state(*args, state)


def softmax_grad_probs_from_lse(
    logits: torch.Tensor,
    lse: torch.Tensor,
    local_targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    """Materialize BF16 softmax gradients from an existing corrected LSE."""
    return _load_softmax_probs().softmax_grad_probs_from_lse(
        logits,
        lse.contiguous(),
        local_targets.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )


def softmax_tail_grad_probs_from_lse(
    logits: torch.Tensor,
    lse: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    logit_temperature: float,
):
    """Materialize the temperature-scaled BF16 tail from a corrected LSE."""
    return _load_softmax_probs().softmax_tail_grad_probs_from_lse(
        logits,
        lse.contiguous(),
        valid.contiguous(),
        int(vocab_size),
        float(logit_temperature),
    )


def softmax_repaired_grad_probs_from_lse(
    logits: torch.Tensor,
    lse: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    target_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    vocab_size: int,
    logit_temperature: float,
):
    """Materialize the repaired BF16 gradient with a compact CUDA scatter."""
    return _load_softmax_probs().softmax_repaired_grad_probs_from_lse(
        logits,
        lse.contiguous(),
        targets.contiguous(),
        valid.contiguous(),
        target_probs.contiguous(),
        topk_indices.contiguous(),
        topk_probs.contiguous(),
        int(vocab_size),
        float(logit_temperature),
    )


def softmax_repaired_grad_probs_from_lse_inplace(
    logits: torch.Tensor,
    lse: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    target_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    vocab_size: int,
    logit_temperature: float,
):
    """Overwrite dead BF16 logits with the bitwise-identical repaired G."""
    return _load_softmax_probs().softmax_repaired_grad_probs_from_lse_inplace(
        logits,
        lse.contiguous(),
        targets.contiguous(),
        valid.contiguous(),
        target_probs.contiguous(),
        topk_indices.contiguous(),
        topk_probs.contiguous(),
        int(vocab_size),
        float(logit_temperature),
    )


def p_cache_loss_and_lse(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    if _use_direct_lse_producer():
        return direct_loss_and_lse(logits, targets, valid, vocab_size)
    return loss_and_lse(logits, targets, valid)


@functools.lru_cache(maxsize=1)
def _load_nvfp4_softmax_quant():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)
    src = os.path.join(this_dir, "v4_nvfp4_softmax_quant.cu")
    build_dir = os.environ.get("FP4_CCE_V4_NVFP4_EXT_BUILD_DIR", "/tmp/fp4_cce_v4_nvfp4_softmax_quant")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="fp4_cce_v4_nvfp4_softmax_quant",
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_include_paths=[
            repo_root,
            os.path.join(repo_root, "TK_quantisation", "nvfp4_v5"),
        ],
        verbose=os.environ.get("FP4_CCE_V4_EXT_VERBOSE", "0") == "1",
    )


def nvfp4_tiled_p_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    constant_scale: bool,
):
    loss, lse = p_cache_loss_and_lse(logits, targets, valid, vocab_size)
    row_fp4, row_sc, col_fp4, col_sc, sg = _load_nvfp4_softmax_quant().nvfp4_softmax_quant_row_col(
        logits,
        lse.contiguous(),
        valid.contiguous(),
        int(vocab_size),
        bool(constant_scale),
    )
    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def nvfp4_tma_p_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    loss, lse = p_cache_loss_and_lse(logits, targets, valid, vocab_size)
    row_fp4, row_sc, col_fp4, col_sc, sg = _load_nvfp4_softmax_quant().nvfp4_softmax_quant_row_col_tma(
        logits,
        lse.contiguous(),
        valid.contiguous(),
        int(vocab_size),
    )
    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def nvfp4_staged_p_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    loss, row_fp4, row_sc, col_fp4, col_sc, sg = (
        _load_nvfp4_softmax_quant().nvfp4_softmax_quant_row_col_staged(
            logits,
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
        )
    )
    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def nvfp4_staged_g_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    loss, row_fp4, row_sc, col_fp4, col_sc, sg = (
        _load_nvfp4_softmax_quant().nvfp4_softmax_grad_quant_row_col_staged(
            logits,
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
        )
    )
    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def nvfp4_tiled_g_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    global_scale_max: float = 448.0,
    block_scale: bool = False,
):
    loss, lse = p_cache_loss_and_lse(logits, targets, valid, vocab_size)
    row_fp4, row_sc, col_fp4, col_sc, sg = (
        _load_nvfp4_softmax_quant().nvfp4_softmax_grad_quant_row_col_tiled(
            logits,
            lse.contiguous(),
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            float(global_scale_max),
            bool(block_scale),
        )
    )
    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def nvfp4_tiled_g_cache_target_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int = 0,
    *,
    x: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    exact_selected_logits: bool = False,
    constant_scale: bool = True,
    global_scale_max: float = 448.0,
    chunk_scale: bool = False,
):
    if exact_selected_logits and (x is None or weight is None):
        raise ValueError("exact selected-logit repair requires BF16 x and weight")
    fused_row = (
        os.environ.get("FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW", "0") != "0"
    )
    if fused_row:
        if not constant_scale or chunk_scale or topk_split not in (0, 8, 12, 16):
            raise ValueError(
                "fused NVFP4 softmax-row production requires constant scale, "
                "no chunk scaling, and top-k 0, 8, 12, or 16"
            )
        (
            loss,
            _lse,
            target_probs,
            topk_probs,
            topk_indices,
            row_fp4,
            row_sc,
            row_sg,
            row_normalization,
        ) = direct_loss_lse_target_topk_split_exact_logits_nvfp4_row(
            logits,
            x,
            weight,
            targets,
            valid,
            vocab_size,
            topk_split,
            global_scale_max,
            exact_selected_logits,
        )
        if topk_split == 0:
            topk_probs = None
            topk_indices = None
        col_fp4, col_sc = (
            _load_nvfp4_softmax_quant().nvfp4_col_requant_from_row(
                row_fp4, row_sc, row_normalization
            )
        )
        return (
            loss,
            row_fp4,
            row_sc,
            row_sg,
            col_fp4,
            col_sc,
            row_sg,
            target_probs,
            topk_indices,
            topk_probs,
            row_normalization,
        )
    (
        loss,
        lse,
        target_probs,
        topk_probs,
        topk_indices,
    ) = (
        direct_loss_lse_target_topk_split_exact_logits(
            logits,
            x,
            weight,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
        if exact_selected_logits
        else direct_loss_lse_target_topk_split(
            logits,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
    )
    if topk_split == 0:
        topk_probs = None
        topk_indices = None

    if chunk_scale:
        row_fp4, row_sc, row_sg, col_fp4, col_sc, col_sg = (
            _load_nvfp4_softmax_quant().nvfp4_softmax_quant_row_col_chunk_scaled(
                logits,
                lse.contiguous(),
                valid.contiguous(),
                int(vocab_size),
                float(global_scale_max),
            )
        )
    else:
        row_fp4, row_sc, col_fp4, col_sc, sg = (
            _load_nvfp4_softmax_quant().nvfp4_softmax_quant_row_col(
                logits,
                lse.contiguous(),
                valid.contiguous(),
                int(vocab_size),
                bool(constant_scale),
                float(global_scale_max),
            )
        )
        row_sg = sg
        col_sg = sg
    return (
        loss,
        row_fp4,
        row_sc,
        row_sg,
        col_fp4,
        col_sc,
        col_sg,
        target_probs,
        topk_indices,
        topk_probs,
    )


def mxfp8_quant_row_col(
    value: torch.Tensor,
    quant_max: float = 448.0,
):
    """Emit native TK MXFP8 operands for ``value`` and ``value.T`` together."""
    return _load_nvfp4_softmax_quant().mxfp8_quant_row_col(
        value.contiguous(),
        float(quant_max),
    )


def mxfp8_quant_row(
    value: torch.Tensor,
    quant_max: float = 448.0,
):
    """Emit the native TK MXFP8 row while omitting the transpose operand."""
    extension = _load_nvfp4_softmax_quant()
    fn = getattr(extension, "mxfp8_quant_row", None)
    if fn is None:
        raise RuntimeError(
            "v4 softmax quant extension lacks the row-only MXFP8 producer; "
            "rebuild it from the pinned source before enabling cache elision"
        )
    return fn(
        value.contiguous(),
        float(quant_max),
    )


def mxfp8_quant_col(
    value: torch.Tensor,
    quant_max: float = 448.0,
):
    """Emit the native TK MXFP8 operand for ``value.T`` only."""
    return _load_nvfp4_softmax_quant().mxfp8_quant_col(
        value.contiguous(),
        float(quant_max),
    )


def mxfp4_quant_row_mxfp8_col(value: torch.Tensor):
    """Emit MXFP4 rows and an MXFP8 operand for ``value.T`` together."""
    return _load_nvfp4_softmax_quant().mxfp4_quant_row_mxfp8_col(
        value.contiguous()
    )


def nvfp4_col_requant_from_mxfp8_row(
    row_fp8: torch.Tensor,
    row_sc: torch.Tensor,
    row_normalization: torch.Tensor,
    global_scale_max: float = 448.0,
):
    """Build the native-NVFP4 transposed operand from an MXFP8 row."""
    return _load_nvfp4_softmax_quant().nvfp4_col_requant_from_mxfp8_row(
        row_fp8,
        row_sc,
        row_normalization.contiguous(),
        float(global_scale_max),
    )


def mxfp8_col_requant_from_mxfp8_row(
    row_fp8: torch.Tensor,
    row_sc: torch.Tensor,
    row_normalization: torch.Tensor,
    quant_max: float = 448.0,
):
    """Build the native-MXFP8 transposed operand from an MXFP8 row."""
    return _load_nvfp4_softmax_quant().mxfp8_col_requant_from_mxfp8_row(
        row_fp8,
        row_sc,
        row_normalization.contiguous(),
        float(quant_max),
    )


def mxfp8_rmsnorm_quant_row_col(
    value: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
    quant_max: float = 448.0,
):
    """Fuse RMSNorm with native MXFP8 row/transpose operand production."""
    return _load_nvfp4_softmax_quant().mxfp8_rmsnorm_quant_row_col(
        value.contiguous(),
        gamma.contiguous(),
        float(epsilon),
        float(quant_max),
    )


def mxfp8_tiled_g_cache_target_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int = 0,
    *,
    x: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    exact_selected_logits: bool = False,
    quant_max: float = 448.0,
):
    """Fuse the selected-masked softmax tail into row/transpose MXFP8."""
    if exact_selected_logits and (x is None or weight is None):
        raise ValueError("exact selected-logit repair requires BF16 x and weight")
    (
        loss,
        lse,
        target_probs,
        topk_probs,
        topk_indices,
    ) = (
        direct_loss_lse_target_topk_split_exact_logits(
            logits,
            x,
            weight,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
        if exact_selected_logits
        else direct_loss_lse_target_topk_split(
            logits,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
    )
    if topk_split == 0:
        topk_probs = None
        topk_indices = None

    row_fp8, row_sc, col_fp8, col_sc = (
        _load_nvfp4_softmax_quant().mxfp8_softmax_quant_row_col(
            logits,
            lse.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            float(quant_max),
        )
    )
    return (
        loss,
        row_fp8,
        row_sc,
        col_fp8,
        col_sc,
        target_probs,
        topk_indices,
        topk_probs,
    )


def mxfp8_row_nvfp4_col_tiled_g_cache_target_split(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    topk_split: int = 0,
    *,
    x: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    exact_selected_logits: bool = False,
    mxfp8_quant_max: float = 448.0,
    nvfp4_global_scale_max: float = 448.0,
):
    """Emit a selected-masked MXFP8 row and native-NVFP4 column G tail."""
    if exact_selected_logits and (x is None or weight is None):
        raise ValueError("exact selected-logit repair requires BF16 x and weight")
    (
        loss,
        lse,
        target_probs,
        topk_probs,
        topk_indices,
    ) = (
        direct_loss_lse_target_topk_split_exact_logits(
            logits,
            x,
            weight,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
        if exact_selected_logits
        else direct_loss_lse_target_topk_split(
            logits,
            targets,
            valid,
            vocab_size,
            topk_split,
        )
    )
    if topk_split == 0:
        topk_probs = None
        topk_indices = None

    row_fp8, row_sc, col_fp4, col_sc, col_sg = (
        _load_nvfp4_softmax_quant().mxfp8_row_nvfp4_col_softmax_quant(
            logits,
            lse.contiguous(),
            valid.contiguous(),
            int(vocab_size),
            float(mxfp8_quant_max),
            float(nvfp4_global_scale_max),
        )
    )
    return (
        loss,
        row_fp8,
        row_sc,
        col_fp4,
        col_sc,
        col_sg,
        target_probs,
        topk_indices,
        topk_probs,
    )


def nvfp4_tma_g_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    loss, lse = p_cache_loss_and_lse(logits, targets, valid, vocab_size)
    row_fp4, row_sc, col_fp4, col_sc, sg = (
        _load_nvfp4_softmax_quant().nvfp4_softmax_grad_quant_row_col_tma(
            logits,
            lse.contiguous(),
            targets.contiguous(),
            valid.contiguous(),
            int(vocab_size),
        )
    )
    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def nvfp4_vocab_parallel_tiled_g_cache(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_start: int,
    global_vocab_size: int,
    tp_group,
):
    loss, lse, local_targets, valid, local_valid_cols = vocab_parallel_loss_lse_targets(
        logits,
        targets,
        valid,
        vocab_start,
        global_vocab_size,
        tp_group,
        fuse_cuda_targets=use_nvfp4_vocab_parallel_cuda_lse_targets(),
    )
    row_fp4, row_sc, col_fp4, col_sc, sg = (
        _load_nvfp4_softmax_quant().nvfp4_softmax_grad_quant_row_col_tiled(
            logits,
            lse.contiguous(),
            local_targets.contiguous(),
            valid.contiguous(),
            int(local_valid_cols),
        )
    )
    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


@functools.lru_cache(maxsize=1)
def _load_sparse_correct():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(this_dir, "v4_sparse_correct.cu")
    build_dir = os.environ.get(
        "FP4_CCE_V4_EXT_BUILD_DIR", "/tmp/fp4_cce_v4_sparse_correct_v4"
    )
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="fp4_cce_v4_sparse_correct_v4",
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        verbose=os.environ.get("FP4_CCE_V4_EXT_VERBOSE", "0") == "1",
    )


def _fallback_sparse_correct(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    valid = targets.ne(ignore_index)
    if bool(valid.all()):
        dE.add_((weight.index_select(0, targets).float() * (-scale)).to(torch.bfloat16))
        dC.index_add_(0, targets, (x.float() * (-scale)).to(torch.bfloat16))
        return

    rows = torch.where(valid)[0]
    cols = targets[rows]
    if bool(rows.numel() == 0):
        return
    dE[rows] = (dE[rows].float() - weight[cols].float() * scale).to(torch.bfloat16)
    dC.index_add_(0, cols, (x[rows].float() * (-scale)).to(torch.bfloat16))


def _use_sparse_vec2(weight: torch.Tensor) -> bool:
    mode = os.environ.get("FP4_CCE_V4_SPARSE_VEC2")
    if mode is not None:
        return mode != "0"
    return int(weight.shape[0]) >= 65536


def _launch_grouped_sparse_correct_dC(
    extension,
    dC: torch.Tensor,
    x: torch.Tensor,
    sorted_vocab_rows: torch.Tensor,
    sorted_x_rows: torch.Tensor,
    sorted_coefficients: torch.Tensor,
    use_vec2: bool,
) -> None:
    entries_override = os.environ.get(
        "FP4_CCE_V4_SPARSE_DC_ENTRIES_PER_BLOCK"
    )
    entries_per_block = (
        int(entries_override)
        if entries_override is not None
        else (4 if sorted_vocab_rows.numel() <= x.shape[0] else 8)
    )
    if entries_per_block not in (1, 2, 3, 4, 5, 8):
        raise ValueError(
            "FP4_CCE_V4_SPARSE_DC_ENTRIES_PER_BLOCK must be 1, 2, 3, 4, 5, or 8"
        )
    threads_override = os.environ.get("FP4_CCE_V4_SPARSE_DC_THREADS")
    threads_per_block = (
        int(threads_override)
        if threads_override is not None
        else (512 if x.shape[1] >= 2048 else 256)
    )
    if threads_per_block not in (128, 256, 512):
        raise ValueError(
            "FP4_CCE_V4_SPARSE_DC_THREADS must be 128, 256, or 512"
        )
    extension.grouped_sparse_correct_dC(
        dC,
        x,
        sorted_vocab_rows,
        sorted_x_rows,
        sorted_coefficients,
        use_vec2,
        entries_per_block,
        threads_per_block,
    )


def _prepare_grouped_sparse_correct_dC(
    vocab_rows: torch.Tensor,
    x_rows: torch.Tensor,
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_rows = vocab_rows.reshape(-1).to(dtype=torch.int32)
    x_rows = x_rows.reshape(-1).to(dtype=torch.int32)
    coefficients = coefficients.reshape(-1).to(dtype=torch.float32)
    sorted_vocab_rows, order = torch.sort(vocab_rows)
    return (
        sorted_vocab_rows.contiguous(),
        x_rows.index_select(0, order).contiguous(),
        coefficients.index_select(0, order).contiguous(),
    )


def _grouped_sparse_correct_dC(
    extension,
    dC: torch.Tensor,
    x: torch.Tensor,
    vocab_rows: torch.Tensor,
    x_rows: torch.Tensor,
    coefficients: torch.Tensor,
    use_vec2: bool,
) -> None:
    """Accumulate shared vocabulary rows in FP32 before one BF16 write."""
    prepared = _prepare_grouped_sparse_correct_dC(
        vocab_rows, x_rows, coefficients
    )
    _launch_grouped_sparse_correct_dC(
        extension, dC, x, *prepared, use_vec2
    )


def _target_dC_updates(
    targets: torch.Tensor,
    coefficients: torch.Tensor,
    ignore_index: int,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(targets.numel(), device=targets.device, dtype=torch.int32)
    if valid is None:
        valid = targets.ne(ignore_index)
    return (
        targets.to(dtype=torch.int32),
        rows,
        torch.where(valid, coefficients.float(), 0.0),
    )


def _target_topk_dC_updates(
    targets: torch.Tensor,
    target_coefficients: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_coefficients: torch.Tensor,
    ignore_index: int,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_vocab, rows, target_coefficients = _target_dC_updates(
        targets,
        target_coefficients,
        ignore_index,
        valid,
    )
    topk_indices = topk_indices.reshape(targets.numel(), -1)
    topk_coefficients = topk_coefficients.reshape(targets.numel(), -1)
    topk_rows = rows[:, None].expand_as(topk_indices).reshape(-1)
    if valid is None:
        valid = targets.ne(ignore_index)
    valid = valid[:, None].expand_as(topk_indices)
    return (
        torch.cat((target_vocab, topk_indices.reshape(-1).to(torch.int32))),
        torch.cat((rows, topk_rows)),
        torch.cat(
            (
                target_coefficients,
                torch.where(
                    valid.reshape(-1), topk_coefficients.reshape(-1), 0.0
                ),
            )
        ),
    )


def prepare_sparse_correct_target_topk_dC(
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    ignore_index: int,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare and sort scale-free top-k dW repair descriptors."""
    vocab_rows, x_rows, coefficients = _target_topk_dC_updates(
        targets,
        -(1.0 - target_probs),
        topk_indices,
        topk_probs,
        ignore_index,
        valid,
    )
    return _prepare_grouped_sparse_correct_dC(
        vocab_rows, x_rows, coefficients
    )


def sparse_correct(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    if os.environ.get("FP4_CCE_V4_FUSED_SPARSE", "1") == "0":
        _fallback_sparse_correct(dE, dC, x, weight, targets, scale, ignore_index)
        return

    try:
        extension = _load_sparse_correct()
        fused = extension.sparse_correct
        getattr(extension, "grouped_sparse_correct_dC")
    except Exception:
        if os.environ.get("FP4_CCE_V4_STRICT_FUSED_SPARSE", "0") == "1":
            raise
        _fallback_sparse_correct(dE, dC, x, weight, targets, scale, ignore_index)
        return

    use_vec2 = _use_sparse_vec2(weight)
    fused(
        dE,
        dC,
        x,
        weight,
        targets,
        scale.reshape(1),
        int(ignore_index),
        use_vec2,
    )
    vocab_rows, x_rows, coefficients = _target_dC_updates(
        targets,
        torch.ones_like(targets, dtype=torch.float32) * -scale,
        ignore_index,
    )
    _grouped_sparse_correct_dC(
        extension, dC, x, vocab_rows, x_rows, coefficients, use_vec2
    )


def sparse_correct_target_split(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    try:
        extension = _load_sparse_correct()
        fused = extension.sparse_correct_target_split
        getattr(extension, "grouped_sparse_correct_dC")
    except Exception:
        if os.environ.get("FP4_CCE_V4_STRICT_FUSED_SPARSE", "0") == "1":
            raise
        valid = targets.ne(ignore_index)
        rows = torch.where(valid)[0]
        if bool(rows.numel() == 0):
            return
        cols = targets[rows]
        correction = ((1.0 - target_probs[rows]) * scale).float()
        dE[rows] = (
            dE[rows].float()
            - weight[cols].float() * correction[:, None]
        ).to(torch.bfloat16)
        dC.index_add_(
            0,
            cols,
            (-x[rows].float() * correction[:, None]).to(torch.bfloat16),
        )
        return

    use_vec2 = _use_sparse_vec2(weight)
    fused(
        dE,
        dC,
        x,
        weight,
        targets,
        target_probs,
        scale.reshape(1),
        int(ignore_index),
        use_vec2,
    )
    vocab_rows, x_rows, coefficients = _target_dC_updates(
        targets,
        -(1.0 - target_probs) * scale,
        ignore_index,
    )
    _grouped_sparse_correct_dC(
        extension, dC, x, vocab_rows, x_rows, coefficients, use_vec2
    )


def _topk_split_size(topk_indices: torch.Tensor) -> int:
    return 1 if topk_indices.dim() == 1 else int(topk_indices.shape[1])


def sparse_correct_target_topk_dE(
    dE: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    extension = _load_sparse_correct()
    use_vec2 = _use_sparse_vec2(weight)
    topk = _topk_split_size(topk_indices)
    extension.sparse_correct_target_topk_dE(
        dE,
        weight,
        targets,
        target_probs,
        topk_indices,
        topk_probs,
        scale.reshape(1),
        topk,
        int(ignore_index),
        use_vec2,
    )


def sparse_correct_target_topk_dC(
    dC: torch.Tensor,
    x: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    extension = _load_sparse_correct()
    use_vec2 = _use_sparse_vec2(dC)
    vocab_rows, x_rows, coefficients = _target_topk_dC_updates(
        targets,
        -(1.0 - target_probs) * scale,
        topk_indices,
        topk_probs * scale,
        ignore_index,
    )
    _grouped_sparse_correct_dC(
        extension, dC, x, vocab_rows, x_rows, coefficients, use_vec2
    )


def sparse_correct_target_topk_dC_prepared(
    dC: torch.Tensor,
    x: torch.Tensor,
    prepared: tuple[torch.Tensor, ...],
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply pre-sorted dW repairs, leaving only scalar scaling in backward."""
    extension = _load_sparse_correct()
    use_vec2 = _use_sparse_vec2(dC)
    sorted_vocab_rows, sorted_x_rows, sorted_coefficients = prepared[:3]
    scaled_coefficients = (sorted_coefficients * scale).contiguous()
    _launch_grouped_sparse_correct_dC(
        extension,
        dC,
        x,
        sorted_vocab_rows,
        sorted_x_rows,
        scaled_coefficients,
        use_vec2,
    )
    return scaled_coefficients


def compact_target_topk_dC(
    x: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build and multiply the compact repair matrix without sorting descriptors."""
    extension = _load_sparse_correct()
    raw_chunk_rows = os.environ.get(
        "FP4_CCE_V4_MX_COMPACT_DW_CHUNK_ROWS", "4096"
    ).strip()
    chunk_rows = int(raw_chunk_rows)
    if chunk_rows < 0:
        raise ValueError(
            "FP4_CCE_V4_MX_COMPACT_DW_CHUNK_ROWS must be non-negative"
        )
    if chunk_rows:
        if not hasattr(extension, "compact_target_topk_correction"):
            raise RuntimeError(
                "chunked compact dW repair requires "
                "compact_target_topk_correction"
            )
        unique_vocab_rows, compact_correction = (
            extension.compact_target_topk_correction(
                x,
                targets,
                target_probs,
                topk_indices,
                topk_probs,
                int(vocab_size),
                int(ignore_index),
                chunk_rows,
            )
        )
    else:
        unique_vocab_rows, coefficient_matrix = (
            extension.compact_target_topk_coefficients(
                targets,
                target_probs,
                topk_indices,
                topk_probs,
                int(vocab_size),
                int(ignore_index),
            )
        )
        compact_correction = torch.mm(coefficient_matrix, x)
    compact_rows = torch.arange(
        unique_vocab_rows.numel(), dtype=torch.int32, device=x.device
    )
    return unique_vocab_rows, compact_correction, compact_rows


def add_compact_target_topk_dC(
    dC: torch.Tensor,
    unique_vocab_rows: torch.Tensor,
    compact_correction: torch.Tensor,
    compact_rows: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Add one pre-accumulated correction vector per vocabulary row."""
    if unique_vocab_rows.numel() == 0:
        return
    safe_vocab_rows = unique_vocab_rows.clamp(0, dC.shape[0] - 1).contiguous()
    coefficients = scale.float().expand(unique_vocab_rows.numel()).contiguous()
    _launch_grouped_sparse_correct_dC(
        _load_sparse_correct(),
        dC,
        compact_correction,
        safe_vocab_rows,
        compact_rows,
        coefficients,
        _use_sparse_vec2(dC),
    )


def sparse_correct_target_topk_split(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    sparse_correct_target_topk_dE(
        dE,
        weight,
        targets,
        target_probs,
        topk_indices,
        topk_probs,
        scale,
        ignore_index,
    )
    sparse_correct_target_topk_dC(
        dC,
        x,
        targets,
        target_probs,
        topk_indices,
        topk_probs,
        scale,
        ignore_index,
    )


def sparse_correct_target_top1_split(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    top1_indices: torch.Tensor,
    top1_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    sparse_correct_target_topk_split(
        dE, dC, x, weight, targets, target_probs,
        top1_indices, top1_probs, scale, ignore_index,
    )


def sparse_correct_target_top2_split(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    top2_indices: torch.Tensor,
    top2_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    sparse_correct_target_topk_split(
        dE, dC, x, weight, targets, target_probs,
        top2_indices, top2_probs, scale, ignore_index,
    )


def sparse_correct_target_top4_split(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    top4_indices: torch.Tensor,
    top4_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    sparse_correct_target_topk_split(
        dE, dC, x, weight, targets, target_probs,
        top4_indices, top4_probs, scale, ignore_index,
    )


def sparse_correct_target_top6_split(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_probs: torch.Tensor,
    top6_indices: torch.Tensor,
    top6_probs: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    sparse_correct_target_topk_split(
        dE, dC, x, weight, targets, target_probs,
        top6_indices, top6_probs, scale, ignore_index,
    )


def sparse_correct_scaled_dE(
    dE: torch.Tensor,
    dC: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> None:
    if os.environ.get("FP4_CCE_V4_FUSED_SPARSE", "1") == "0":
        dE.mul_(scale)
        _fallback_sparse_correct(dE, dC, x, weight, targets, scale, ignore_index)
        return

    try:
        extension = _load_sparse_correct()
        fused = extension.sparse_correct_scaled_dE
        getattr(extension, "grouped_sparse_correct_dC")
    except Exception:
        if os.environ.get("FP4_CCE_V4_STRICT_FUSED_SPARSE", "0") == "1":
            raise
        dE.mul_(scale)
        _fallback_sparse_correct(dE, dC, x, weight, targets, scale, ignore_index)
        return

    use_vec2 = _use_sparse_vec2(weight)
    fused(
        dE,
        dC,
        x,
        weight,
        targets,
        scale.reshape(1),
        int(ignore_index),
        use_vec2,
    )
    vocab_rows, x_rows, coefficients = _target_dC_updates(
        targets,
        torch.ones_like(targets, dtype=torch.float32) * -scale,
        ignore_index,
    )
    _grouped_sparse_correct_dC(
        extension, dC, x, vocab_rows, x_rows, coefficients, use_vec2
    )
