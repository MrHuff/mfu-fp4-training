#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license retained at
# torchtitan_submodule/LICENSE in this source distribution.
"""
This script is copied from torchtian_submodule/scripts/generate/_generation.py

It was impossible to reliably import the script from the submodule so it is copied
and modified here to work with KV caching.
"""

from typing import Optional

import torch


def multinomial_sample_one(
    probs: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    seq_lens: Optional[torch.Tensor] = None,
    **other_model_kwargs,
) -> torch.Tensor:
    if seq_lens is not None:
        logits = model(x, seq_lens=seq_lens, **other_model_kwargs)  # (B, T, vocab_size)
    else:
        logits = model(x, **other_model_kwargs)  # (B, T, vocab_size)
    if seq_lens is not None and logits.shape[1] > 1:
        # If seq_lens is provided, we need to gather the logits for the last token of each sequence
        s = (seq_lens.view(logits.shape[0], 1, 1) - 1).expand(logits.shape)
        logits = logits.gather(dim=1, index=s)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **other_model_kwargs,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()
    batch_size = generated_tokens.shape[0]

    # Track which sequences have finished generating
    finished_sequences = torch.zeros(
        batch_size, dtype=torch.bool, device=input_ids.device
    )

    for _ in range(max_new_tokens):
        # If all sequences are finished, stop generation
        if eos_token_id is not None and finished_sequences.all():
            break

        next_token = generate_next_token(
            model,
            generated_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
            **other_model_kwargs,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        # Check for EOS token and mark sequences as finished
        if eos_token_id is not None:
            finished_sequences = finished_sequences | (
                next_token.squeeze(-1) == eos_token_id
            )

    return generated_tokens
