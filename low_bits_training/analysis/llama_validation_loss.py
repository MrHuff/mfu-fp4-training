#!/usr/bin/env python3
"""Memory-bounded Llama validation loss adapted from the recovered UE5M3 path."""

from __future__ import annotations

import math
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load_file



def sha256_file(path: str | Path) -> str:
    """Hash an input without depending on an experiment-capsule module."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidationShard:
    path: str
    sha256: str
    shape: tuple[int, int]
    sequence_offset: int


class LlamaCausalLMValidationAdapter(torch.nn.Module):
    """Expose Hugging Face ``model.model`` through the evaluator's backbone ABI."""

    def __init__(self, causal_lm: torch.nn.Module) -> None:
        super().__init__()
        backbone = getattr(causal_lm, "model", None)
        lm_head = getattr(causal_lm, "lm_head", None)
        if not isinstance(backbone, torch.nn.Module):
            raise TypeError("Llama causal LM must expose model.model")
        if not isinstance(lm_head, torch.nn.Module):
            raise TypeError("Llama causal LM must expose model.lm_head")
        self.causal_lm = causal_lm

    @property
    def backbone(self) -> torch.nn.Module:
        return self.causal_lm.model

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.causal_lm.lm_head

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.causal_lm(*args, **kwargs)


def _extract_hidden_states(outputs: Any) -> torch.Tensor:
    value = getattr(outputs, "last_hidden_state", None)
    if value is None and isinstance(outputs, (tuple, list)) and outputs:
        value = outputs[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("backbone output does not expose last_hidden_state")
    return value


def _extract_logits(outputs: Any) -> torch.Tensor:
    value = getattr(outputs, "logits", None)
    if value is None and isinstance(outputs, (tuple, list)) and outputs:
        value = outputs[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("model output does not expose logits")
    return value


def _validate_targets(targets: torch.Tensor, vocab_size: int) -> None:
    if targets.ndim != 2 or targets.numel() == 0:
        raise ValueError("targets must have non-empty [batch, sequence] shape")
    token_min = int(targets.min().item())
    token_max = int(targets.max().item())
    if token_min < 0 or token_max >= vocab_size:
        raise ValueError(
            f"target IDs [{token_min}, {token_max}] are outside vocabulary [0, {vocab_size})"
        )


def _cross_entropy_sums(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    if not bool(torch.isfinite(token_losses).all().item()):
        raise FloatingPointError("non-finite validation loss")
    return token_losses.sum(dim=1, dtype=torch.float64)


def sequence_loss_sums_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, *, chunk_tokens: int
) -> torch.Tensor:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets are not position-aligned")
    _validate_targets(targets, logits.shape[-1])
    sums = torch.zeros(logits.shape[0], dtype=torch.float64, device=logits.device)
    for start in range(0, targets.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, targets.shape[1])
        sums += _cross_entropy_sums(logits[:, start:stop], targets[:, start:stop])
    return sums.cpu()


def sequence_loss_sums_from_hidden(
    adapter: LlamaCausalLMValidationAdapter,
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if hidden_states.ndim != 3 or hidden_states.shape[:2] != targets.shape:
        raise ValueError("hidden states and targets are not position-aligned")
    vocab_size = getattr(adapter.lm_head, "out_features", None)
    if not isinstance(vocab_size, int):
        raise TypeError("Llama LM head must expose integer out_features")
    _validate_targets(targets, vocab_size)
    try:
        head_dtype = next(adapter.lm_head.parameters()).dtype
    except StopIteration as error:
        raise TypeError("Llama LM head must have learned parameters") from error
    sums = torch.zeros(hidden_states.shape[0], dtype=torch.float64, device=hidden_states.device)
    for start in range(0, targets.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, targets.shape[1])
        logits = adapter.lm_head(hidden_states[:, start:stop].to(head_dtype))
        sums += _cross_entropy_sums(logits, targets[:, start:stop])
        del logits
    return sums.cpu()


def evaluate_token_batch(
    adapter: LlamaCausalLMValidationAdapter,
    tokens: torch.Tensor,
    *,
    ce_chunk_tokens: int,
    logit_path: Literal["chunked_lm_head", "model_forward"] = "chunked_lm_head",
) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.shape[1] <= 1:
        raise ValueError("tokens must have shape [batch, sequence + 1]")
    input_ids = tokens[:, :-1].to(dtype=torch.long).contiguous()
    targets = tokens[:, 1:].to(dtype=torch.long).contiguous()
    if logit_path == "chunked_lm_head":
        outputs = adapter.backbone(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = _extract_hidden_states(outputs)
        return sequence_loss_sums_from_hidden(
            adapter, hidden, targets, chunk_tokens=ce_chunk_tokens
        )
    if logit_path == "model_forward":
        outputs = adapter(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = _extract_logits(outputs)
        return sequence_loss_sums_from_logits(
            logits, targets, chunk_tokens=ce_chunk_tokens
        )
    raise ValueError(f"unknown logit path: {logit_path}")


def evaluate_frozen_validation(
    adapter: LlamaCausalLMValidationAdapter,
    shard_paths: Sequence[str | Path],
    *,
    device: str | torch.device,
    batch_size: int = 1,
    ce_chunk_tokens: int = 256,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0 or ce_chunk_tokens <= 0:
        raise ValueError("batch size and CE chunk size must be positive")
    shards = []
    offset = 0
    for raw_path in shard_paths:
        path = Path(raw_path).resolve()
        tensors = load_file(path, device="cpu")
        if set(tensors) != {"tokens"}:
            raise ValueError(f"{path} does not contain only the tokens tensor")
        tokens = tensors["tokens"]
        if tokens.dtype is not torch.int32 or tokens.ndim != 2 or tokens.shape[1] <= 1:
            raise ValueError(f"{path} violates the int32 [N, S+1] token contract")
        shards.append(
            ValidationShard(
                path=str(path),
                sha256=sha256_file(path),
                shape=tuple(tokens.shape),
                sequence_offset=offset,
            )
        )
        offset += tokens.shape[0]
    if not shards or len({shard.shape[1] for shard in shards}) != 1:
        raise ValueError("validation shards must be non-empty and equal width")

    adapter.eval()
    evaluation_device = torch.device(device)
    total_sequences = sum(shard.shape[0] for shard in shards)
    sequence_tokens = shards[0].shape[1] - 1
    loss_sums: list[float] = []
    provenance = []
    started = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        for shard_index, shard in enumerate(shards):
            tokens = load_file(shard.path, device="cpu")["tokens"]
            if sha256_file(shard.path) != shard.sha256:
                raise RuntimeError(f"validation shard changed during evaluation: {shard.path}")
            for row_start in range(0, tokens.shape[0], batch_size):
                row_stop = min(row_start + batch_size, tokens.shape[0])
                batch = tokens[row_start:row_stop].to(evaluation_device)
                sums = evaluate_token_batch(
                    adapter,
                    batch,
                    ce_chunk_tokens=ce_chunk_tokens,
                    logit_path="chunked_lm_head",
                )
                for batch_index, value in enumerate(sums):
                    row = row_start + batch_index
                    loss_sums.append(float(value.item()))
                    provenance.append(
                        {
                            "sequence_index": shard.sequence_offset + row,
                            "shard_index": shard_index,
                            "row_index": row,
                            "shard_sha256": shard.sha256,
                        }
                    )
                completed += row_stop - row_start
                if progress_callback:
                    progress_callback(completed, total_sequences)
    token_count = total_sequences * sequence_tokens
    total_loss = math.fsum(loss_sums)
    nll = total_loss / token_count
    return {
        "metrics": {
            "loss_sum": total_loss,
            "token_count": token_count,
            "nll": nll,
            "perplexity": math.exp(nll),
        },
        "per_sequence": {
            "loss_sums": loss_sums,
            "token_counts": [sequence_tokens] * total_sequences,
            "provenance": provenance,
        },
        "evaluation": {
            "model_parameter_dtype": "bfloat16",
            "cross_entropy_dtype": "float32",
            "accumulation_dtype": "float64",
            "batch_size": batch_size,
            "ce_chunk_tokens": ce_chunk_tokens,
            "logit_path": "chunked_lm_head",
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
