#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from typing import Callable, Union, List

from copy import deepcopy
import os

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.tools.logging import logger
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

from torchtitan.models.llama3.model.model import (
    apply_rotary_emb,
    repeat_kv,
    Attention,
    FeedForward,
    BlockMask,
    AttentionMasksType,
)


class AttentionWithFusedLinear(torch.nn.Module):
    """
    Multi-head attention module with fused linear parameters.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wqkv (Linear): Fused Linear transformation for queries, keys, and values.
        wo (Linear): Linear transformation for output.

    """

    @classmethod
    @torch.no_grad()
    def from_unfused(cls, mod):
        """
        Convert from an unfused attention module to a fused one.
        """
        mod = deepcopy(mod)
        mod.__class__ = cls
        dim = mod.wq.weight.shape[-1]
        device = mod.wq.weight.device
        dtype = mod.wq.weight.dtype

        total_qkv_heads = mod.n_heads + 2 * mod.n_kv_heads

        mod.wqkv = nn.Linear(
            dim,
            total_qkv_heads * mod.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if device != torch.device("meta"):
            # If unfused model weights have been initialised
            # with data, copy the data from the unfused layers
            mod.wqkv.weight.data.copy_(
                torch.cat(
                    [
                        mod.wq.weight.data.clone(),
                        mod.wk.weight.data.clone(),
                        mod.wv.weight.data.clone(),
                    ],
                    dim=0,
                )
            )
        assert mod.wqkv.weight.shape[0] == total_qkv_heads * mod.head_dim
        del mod.wq, mod.wk, mod.wv
        return mod

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.wqkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if hasattr(self.wqkv, "norm_weight"):
            self.wqkv.norm_weight.data.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        # TODO: Should this call into SDPA and FlexAttention as the base Attention module does?
        attention_masks: AttentionMasksType | None = None,
        **kwargs,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        if isinstance(x, tuple):
            bs, seqlen, _ = x[0].shape
        else:
            bs, seqlen, _ = x.shape

        xqkv = self.wqkv(x)
        xq, xk, xv = torch.split(
            xqkv,
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        if self.use_flex_attn:
            assert isinstance(attention_masks, BlockMask), attention_masks
            output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
        else:
            assert attention_masks is None
            output = self.inner_attention(xq, xk, xv)

        attn_dump_dir = os.environ.get("USE_TK_ATTN_CORE_DUMP_DIR", "").strip()
        attn_dump_match = os.environ.get("USE_TK_ATTN_CORE_DUMP_MATCH", "").strip()
        debug_name = getattr(self, "_lbt_debug_name", self.__class__.__name__)
        if attn_dump_dir and (not attn_dump_match or attn_dump_match in debug_name):
            once_key = f"{debug_name}::attn_core"
            dumped = getattr(AttentionWithFusedLinear, "_debug_core_dumped", None)
            if dumped is None:
                dumped = set()
                setattr(AttentionWithFusedLinear, "_debug_core_dumped", dumped)
            if once_key not in dumped:
                dumped.add(once_key)
                os.makedirs(attn_dump_dir, exist_ok=True)
                safe_name = debug_name.replace("/", "_").replace(":", "_").replace(".", "_")
                torch.save({
                    "debug_name": debug_name,
                    "xq": xq.detach().cpu(),
                    "xk": xk.detach().cpu(),
                    "xv": xv.detach().cpu(),
                    "attn_output": output.detach().cpu(),
                }, os.path.join(attn_dump_dir, f"{safe_name}.pt"))

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForwardWithFusedLinear(torch.nn.Module):
    """
    FeedForward module with fused linear parameters.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w_in (Linear): Linear transformation of the input.
        w_out (Linear): Linear transformation for the output.
    """

    @classmethod
    @torch.no_grad()
    def from_unfused(cls, mod):
        """
        Convert from an unfused feedforward module to a fused one.
        """
        mod = deepcopy(mod)
        mod.__class__ = cls
        dim, hidden_dim = mod.w2.weight.shape
        device = mod.w1.weight.device
        dtype = mod.w1.weight.dtype

        mod.w_in = nn.Linear(dim, 2 * hidden_dim, bias=False, device=device, dtype=dtype)
        mod.w_out = nn.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)

        if device != torch.device("meta"):
            # If unfused model weights have been initialised
            # with data, copy the data from the unfused layers
            mod.w_in.weight.data.copy_(
                torch.cat([mod.w1.weight.data.clone(), mod.w3.weight.data.clone()], dim=0)
            )
            mod.w_out.weight.data.copy_(mod.w2.weight.data.clone())
        del mod.w1, mod.w2, mod.w3
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = torch.chunk(self.w_in(x), chunks=2, dim=-1)
        h = F.silu(y) * gate
        return self.w_out(h)

    def init_weights(self, init_std: float):
        hidden_dim = self.w_out.weight.shape[-1]
        nn.init.trunc_normal_(self.w_in.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.w_in.weight[hidden_dim:].mul_(init_std / 0.02)
        nn.init.trunc_normal_(self.w_out.weight, mean=0.0, std=init_std)
        if hasattr(self.w_in, "norm_weight"):
            self.w_in.norm_weight.data.fill_(1.0)


def _replace(
    model: nn.Module,
    unfused_cls: nn.Module,
    replacement_fn: Callable[[nn.Module], nn.Module],
):
    for name, child in model.named_children():
        if isinstance(child, unfused_cls):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            _replace(child, unfused_cls, replacement_fn)


def swap_unfused_with_fused(
    model: nn.Module, unfused_cls: nn.Module, fused_cls: nn.Module, **kwargs
):
    """
    Swaps layers in the model with those that use fused linear operations
    """
    _replace(model, unfused_cls, lambda mod: fused_cls.from_unfused(mod, **kwargs))


class FusedLinearConverter(ModelConverter):
    """
    Converts model to use fused linear layers in attention
    and feedforward blocks

    This converter pass should be used to
    a) Minimise overheads introduced by separate nn.Linear modules, e.g.,
       operations applied to the input in a pre forward hook.
    b) To simplify/canonicalise the computational graph to be more chain-like
       for further converter passes
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        pass

    def convert(self, model: nn.Module):
        """
        Convert the model to use fused linear layer in
        attention and feedforward blocks
        """

        logger.info("Converting model to use fused linear layers")
        swap_unfused_with_fused(model, Attention, AttentionWithFusedLinear)
        swap_unfused_with_fused(
            model,
            FeedForward,
            FeedForwardWithFusedLinear,
        )
        from low_bits_training.metrics import _logger_model_cache

        _logger_model_cache["model"] = model

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass


register_model_converter(FusedLinearConverter, "fuse_linear")
