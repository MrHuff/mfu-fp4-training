#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import logging
from typing import List, Union, Optional, Tuple
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from low_bits_training.generate._generation import generate_next_token

from torchtitan.config import JobConfig
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.models.llama3.model.model import (
    apply_rotary_emb,
    repeat_kv,
    Attention,
    reshape_for_broadcast,
    AttentionMasksType,
)
from torchtitan.distributed import ParallelDims

logger = logging.getLogger(__name__)


@torch.compiler.disable(recursive=True)
def attention_forward_with_kv_cache(
    self: Attention,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    attention_masks: AttentionMasksType | None = None,
):
    """
    Forward pass of the attention module.

    Args:
        x (torch.Tensor): Input tensor.
        freqs_cis (torch.Tensor): Precomputed frequency tensor.

    Returns:
        torch.Tensor: Output tensor after attention.

    """
    bs, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
    # local heads from sizes of xq, xk, and xv as TP may have sharded them
    # after the above linear ops.
    xq = xq.view(bs, seqlen, -1, self.head_dim)
    xk = xk.view(bs, seqlen, -1, self.head_dim)
    xv = xv.view(bs, seqlen, -1, self.head_dim)

    # If the layer is in inference mode
    if not self.training:
        if x.shape[1] > 1 or self.k_cache is None:
            # Only set seq_lens if it hasn't been set by the transformer forward pass
            if not hasattr(self, "seq_lens") or self.seq_lens is None:
                self.seq_lens = (
                    torch.ones(bs, device=x.device, dtype=torch.int64) * x.shape[1]
                )
            self.k_cache = xk
            self.v_cache = xv

        else:
            # if the max sequence length is greater than the cache size we need to extend the cache by concatenating with zeros
            # Expand the cache by 10 % of the cache size each time
            next_seq_len = getattr(self, "max_seq_len", self.seq_lens.max()) + 1
            assert (
                next_seq_len < freqs_cis.shape[0]
            ), "Trying to generate for more than the maximum"
            if next_seq_len > self.k_cache.shape[1]:
                size_increase = min(
                    max(
                        next_seq_len - self.k_cache.shape[1],
                        int(self.k_cache.shape[1] * 0.1),
                    ),
                    freqs_cis.shape[0] - self.k_cache.shape[1],
                )
                self.k_cache = torch.cat(
                    [
                        self.k_cache,
                        torch.zeros(
                            [bs, size_increase, *xk.shape[2:]],
                            device=xk.device,
                            dtype=xk.dtype,
                        ),
                    ],
                    dim=1,
                )
                self.v_cache = torch.cat(
                    [
                        self.v_cache,
                        torch.zeros(
                            [bs, size_increase, *xv.shape[2:]],
                            device=xv.device,
                            dtype=xv.dtype,
                        ),
                    ],
                    dim=1,
                )
            s = (
                (self.seq_lens - 1)
                .to(torch.int64)
                .reshape(-1, 1, 1, 1)
                .broadcast_to(xk.shape)
            )
            self.k_cache.scatter_(dim=1, index=s, src=xk)
            self.v_cache.scatter_(dim=1, index=s, src=xv)
            xk = self.k_cache
            xv = self.v_cache

    # Determine if this is a decode step
    is_decode = (
        not self.training
        and x.shape[1] == 1
        and hasattr(self, "k_cache")
        and self.k_cache is not None
    )

    xq, xk = apply_rotary_emb_with_cache(
        xq,
        xk,
        freqs_cis=freqs_cis,
        is_decode=is_decode,
        seq_lens=getattr(self, "seq_lens", None),
    )

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
    values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

    if self.training or x.shape[1] != 1:
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    else:
        # Mask out the padding tokens from the cache
        causal_mask = (
            torch.tensor([1], dtype=torch.bool, device=xq.device)
            .broadcast_to([bs, 1, 1, xv.shape[2]])
            .clone()
        )
        for i, bs_seqlen in enumerate(self.seq_lens):
            causal_mask[i, 0, 0, bs_seqlen:] = False
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=causal_mask)
    output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
    output = output.view(bs, seqlen, -1)
    return self.wo(output)


def apply_rotary_emb_with_cache(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    is_decode: bool = False,
    seq_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings with proper handling for KV cache scenarios.

    Args:
        xq: Query tensor
        xk: Key tensor (either new token or full cache)
        freqs_cis: Precomputed frequency tensor
        is_decode: Whether this is a decode step (single new token)
        seq_lens: Actual length of the sequences when doing decode with
            single queries.

    Returns:
        Tuple of query and key tensors with rotary embeddings applied
    """
    if not is_decode:
        # Prefill case: apply rotary embeddings normally
        return apply_rotary_emb(xq, xk, freqs_cis)

    # Decode case: xq is new token, xk might be full cache
    if xq.shape[1] == 1 and xk.shape[1] > 1:
        # xq is new query token, xk is full cache
        # Only apply RoPE to the query at the current position
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        # TODO: handle the special case where xk_ is reaching the max_sequence_length
        freqs_cis_k = reshape_for_broadcast(freqs_cis, xk_)
        # Do seq_lens-1 to account for the 0 indexing of python
        freqs_cis_q = reshape_for_broadcast_for_decode(
            freqs_cis, xq_, query_pos_in_seq=seq_lens - 1
        )
        xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)

    else:
        # Both are single tokens or same length - apply normally
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def reshape_for_broadcast_for_decode(
    freqs_cis: torch.Tensor, x: torch.Tensor, query_pos_in_seq: torch.Tensor
) -> torch.Tensor:
    """
    Version of Torchtitan's reshape_for_broadcast that is adapted
    to process single queries in decode

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        query_pos_in_seq: The real position of the queries inside the sequence
            - used to index the rotary position embeddings

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    freqs_cis = freqs_cis[query_pos_in_seq]
    assert freqs_cis.shape == (query_pos_in_seq.shape[0], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 or i == 0 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


@torch.no_grad()
def generate_with_kv_cache(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    seq_lens: torch.Tensor | None = None,
    use_kv_cache: bool = True,
    eos_token_id: Optional[int | list[int]] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    get_timing: bool = False,
    progress_bar: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[float, float]]]:
    """Generate a sequence of tokens using the model with KV caching enabled.

    This function supports ragged sequences batched together by specifying the seq_lens tensor.
    If it is not specified, it is assumed that all sequences in the batch are of the same length.

    Args:
        model: The model to use for generation.
        input_ids: Input tensor of shape (batch_size, sequence_length)
        max_new_tokens: Number of tokens to generate
        seq_lens: Tensor of shape (batch_size) containing the sequence lengths
            of each sequence in the batch. This number is the total length of the
            sequence including previously cached tokens and the new token.
        use_kv_cache: Whether to use KV caching.
        eos_token_id: The token id of the end of sequence token.
        temperature: The temperature to use for the generation.
        top_k: The top k tokens to use for the generation.
        seed: The seed to use for the generation.
        get_timing: Whether to get the timing of the generation or not.
        progress_bar: Whether to show a progress bar or not.
    Returns:
        Tuple of (generated sequence tensor, generation time)
    """

    # Clear any existing cache
    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)
    if use_kv_cache and not hasattr(model, "clear_kv_cache"):
        logger.warning(
            "KV caching is enabled but the model does not have a clear_kv_cache method, disabling kv caching"
        )
        use_kv_cache = False
    if use_kv_cache:
        model.clear_kv_cache()
    if seq_lens is not None and not isinstance(seq_lens, torch.Tensor):
        seq_lens = torch.tensor(seq_lens, device=input_ids.device, dtype=torch.int64)
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize()
    if eos_token_id is not None:
        eos_token_id = torch.tensor(
            eos_token_id, dtype=torch.int64, device=input_ids.device
        )
    # First token
    next_input = input_ids.clone()
    batch_size = next_input.shape[0]

    final_output = input_ids.clone()
    start_time = time.time()
    finished_sequences = torch.zeros(
        batch_size, dtype=torch.bool, device=input_ids.device
    )
    last_step_time = None
    # Subsequent tokens
    for i in tqdm.tqdm(range(max_new_tokens), disable=not progress_bar):
        if i == max_new_tokens - 1 and get_timing:
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize()
            last_step_start_time = time.time()
        if eos_token_id is not None and finished_sequences.all():
            break
        with torch.profiler.record_function(f"Step {i}"):
            new_token = generate_next_token(
                model,
                next_input,
                temperature=temperature,
                top_k=top_k,
                seq_lens=seq_lens,
                rng=rng,
            )
        if i == max_new_tokens - 1 and get_timing:
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize()
            last_step_time = time.time() - last_step_start_time
        if seq_lens is not None:
            # if the max sequence length is greater than the cache size we need to extend the cache by concatenating with zeros
            if seq_lens.max() + 1 > final_output.shape[1]:
                final_output = torch.cat(
                    [
                        final_output,
                        torch.zeros(
                            [final_output.shape[0], 1],
                            device=final_output.device,
                            dtype=final_output.dtype,
                        ),
                    ],
                    dim=1,
                )
            # scatter the new token into the final output - if the sequence is already finished
            # - we scatter 0s this is not an idea solution as really we would want to not scatter
            # the output to finished sequences. However a naive implementation with indexing returns
            # a copy and means the final_output is not modified in place by the scatter.
            new_token[finished_sequences] = 0
            s = seq_lens.to(torch.int64).reshape(-1, 1).broadcast_to(new_token.shape)
            final_output.scatter_(dim=1, index=s, src=new_token)
            seq_lens += 1
        else:
            new_token[finished_sequences] = 0
            final_output = torch.cat([final_output, new_token], dim=1)
        if eos_token_id is not None:
            finished_sequences = finished_sequences | (
                new_token.squeeze(-1).view(batch_size, 1) == eos_token_id.view(1, -1)
            ).any(dim=1)
        if use_kv_cache:
            next_input = new_token
        else:
            next_input = final_output

    if input_ids.device.type == "cuda":
        torch.cuda.synchronize()
    generation_time = time.time() - start_time
    if get_timing:
        return final_output, (generation_time, last_step_time)
    else:
        return final_output


class KVCacheConverter(ModelConverter):
    """Converter pass for enabling KV caching in transformer models.

    This converter modifies the attention modules to support KV caching,
    which can significantly speed up inference by reusing previously computed
    key and value tensors.
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        # Parse attention module names from comma-separated string if needed
        pass

    def convert(self, model: nn.Module):
        """Convert the model to use KV caching.

        This method finds all attention modules in the model and adds KV caching
        functionality to them.

        Args:
            model: The model to convert
        """

        # Find all attention modules in the model
        attention_modules = []
        for name, module in model.named_modules():
            if "attention" in module.__class__.__name__.lower():
                attention_modules.append((name, module))

        if not attention_modules:
            logger.warning(
                "No attention modules found. These are modules whose class name contains 'attention'."
            )
            return

        logger.info(f"Found {len(attention_modules)} attention modules to convert")
        self._cached_modules = {}
        # Add KV caching to each attention module
        for name, module in attention_modules:
            self._cached_modules[name] = module
            self._add_kv_caching_to_attention_module(module)

        # Add cache management methods to the model
        model.clear_kv_cache = self._clear_kv_cache.__get__(model, type(model))
        self._transformer_forward_with_kv_cache(model)
        logger.info("Successfully added KV caching to the model")

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """Post-optimizer hook (not used for KV caching)."""
        pass

    # To support kv caching I need to make the lengths of the sequences accessible to all the attention modules. To do that I want to replace the forward function of the transformer to add an extra argument where the caller of the model can give input tokens as well as the sequence lengths of each sequence in a batch. Do not modify the code for the transformer - I want it to be intercepted by the kv cache pass and dynamically bound to accept the new argument
    def _transformer_forward_with_kv_cache(self, model: nn.Module):
        """Add KV caching to the transformer forward method.

        This method replaces the forward method of the transformer with a new method that
        accepts the input tokens as well as the sequence lengths of each sequence in a batch.

        Args:
            model: The model to modify
        """
        original_forward = model.forward

        def transformer_forward_with_kv_cache(model, x, *args, seq_lens=None, **kwargs):
            """Forward pass of the transformer with KV caching.

            This method intercepts the sequence lengths of the input tokens and sets them
            on the attention modules.

            Args:
                model: The model
                x: The input tokens
                seq_lens: The sequence lengths of the input tokens
            """
            if seq_lens is None:
                seq_lens = (
                    torch.ones(x.shape[0], device=x.device, dtype=torch.int64)
                    * x.shape[1]
                )
            else:
                seq_lens = seq_lens.to(device=x.device, dtype=torch.int64)
            max_seq_len = seq_lens.max().to("cpu")
            for attention_module in self._cached_modules.values():
                attention_module.max_seq_len = max_seq_len
                attention_module.seq_lens = seq_lens
            return original_forward(x, *args, **kwargs)

        model.forward = transformer_forward_with_kv_cache.__get__(model, type(model))

    def _add_kv_caching_to_attention_module(self, module):
        """Add KV caching functionality to an attention module.

        This method modifies the forward method of the attention module to
        support KV caching.

        See `attention_forward_with_kv_cache` for the implementation of the new
        forward method.

        This method also adds a `clear_kv_cache` method to the attention module.
        Args:
            module: The attention module to modify
        """

        def clear_kv_cache(self):
            if hasattr(self, "k_cache"):
                del self.k_cache
            if hasattr(self, "v_cache"):
                del self.v_cache
            if hasattr(self, "seq_lens"):
                del self.seq_lens
            self.k_cache = None
            self.v_cache = None
            self.seq_lens = None

        # TODO: register the buffers for the k_cache and v_cache
        module.forward = attention_forward_with_kv_cache.__get__(module, type(module))
        module.clear_kv_cache = clear_kv_cache.__get__(module, type(module))

    def _clear_kv_cache(self):
        """Clear the KV cache."""
        # iterate over all attention modules and clear the kv cache
        for module in self._cached_modules.values():
            module.clear_kv_cache()


# Register the converter

register_model_converter(KVCacheConverter, "kv_cache")
