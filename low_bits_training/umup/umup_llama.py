#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import math

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import unit_scaling as uu
from torch import nn
from unit_scaling import functional as U
from unit_scaling.core.functional import logarithmic_interpolation
from unit_scaling.scale import scale_bwd, scale_fwd

from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol


@dataclass
class UmupTransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    norm_type: str = "rmsnorm"
    eos_id: int = 0
    umup_alpha_ffn_act: float = 1.0
    umup_alpha_attn_softmax: float = 1.0
    umup_alpha_res: float = 1.0
    umup_alpha_res_attn_ratio: float = 1.0
    umup_alpha_loss_softmax: float = 1.0

    def __post_init__(self) -> None:
        if self.norm_type != "rmsnorm":
            raise ValueError("Norms other than RMSNorm are currently not supported.")

    def update_from_config(self, job_config, tokenizer) -> None:
        self.norm_type = job_config.model.norm_type
        self.vocab_size = tokenizer.n_words
        self.max_seq_len = job_config.training.seq_len

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        L, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * L * h * q * t

        return nparams, num_flops_per_token


def rotate(x, cos, sin):
    x0, x1 = x.split(x.shape[-1] // 2, dim=-1)
    return torch.cat((cos * x0 - sin * x1, cos * x1 + sin * x0), dim=-1)


def embed_rope(q, k, cos_sin):
    cos, sin = cos_sin.to(q.dtype)
    assert q.shape[-2] == k.shape[-2]
    sequence_length = q.shape[-2]
    cos, sin = cos[:sequence_length], sin[:sequence_length]
    return rotate(q, cos, sin), rotate(k, cos, sin)


def precompute_cos_sin(
    head_dim: int, sequence_length: int, base: float = 10000.0
) -> torch.Tensor:
    freq = base ** -torch.arange(0, head_dim, 2, dtype=torch.float32).div_(head_dim)
    position_ids = torch.arange(0, sequence_length, dtype=torch.float32)
    angle = freq[None, :] * position_ids[:, None]
    cos, sin = angle.cos(), angle.sin()
    return torch.stack([cos, sin])


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    mult: float = 1.0,
) -> torch.Tensor:
    *_, seq_len, d_head = value.shape
    # Empirical model of attention output std given mult and seq_len
    scale = (1 - dropout_p) ** 0.5 / logarithmic_interpolation(
        alpha=1 / (1 + 4 * d_head / mult**2),  # = sigmoid(log(mult**2 / (4 * d_head)))
        lower=((math.log(seq_len) if is_causal else 1) / seq_len) ** 0.5,
        upper=1.0,
    )
    query, key, value = (scale_bwd(t, scale) for t in (query, key, value))
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=mult / d_head,
    )
    return scale_fwd(out, scale)


class Attention(nn.Module):
    """
    Attention module that can be used for (standard) MHA, GQA or MQA.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: UmupTransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = uu.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = uu.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = uu.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = uu.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        self.alpha_attn_softmax = model_args.umup_alpha_attn_softmax

    def init_weights(self):
        for linear in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(linear.weight, mean=0.0, std=1.0)
            linear.weight = uu.Parameter(linear.weight, "weight")

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.

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

        # Using u-muP pos embed
        # This could be rewritten if this combination of shapes is problematic
        xq = xq.transpose(1, 2).reshape(-1, seqlen, self.head_dim)
        xk = xk.transpose(1, 2).reshape(-1, seqlen, self.head_dim)
        xq, xk = embed_rope(xq, xk, cos_sin)
        xq = xq.reshape(bs, -1, seqlen, self.head_dim).transpose(1, 2)
        xk = xk.reshape(bs, -1, seqlen, self.head_dim).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = scaled_dot_product_attention(
            xq, xk, xv, is_causal=True, mult=self.alpha_attn_softmax
        )
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        output = self.wo(output)
        return output


def _gated_swish(x, gate, mult):
    return F.sigmoid(x * mult) * x * gate


def gated_swish(x, gate, mult):
    # Matches definition in u-muP paper
    scale = 1 / logarithmic_interpolation(
        alpha=1 / (1 + 1 / mult**2),
        lower=2**-1,
        upper=2**-0.5,
    )
    x = scale_bwd(x, scale)
    gate = scale_bwd(gate, scale)
    out = _gated_swish(x, gate, mult)
    return scale_fwd(out, scale)


class GatedSwish(nn.Module):
    def forward(self, silu_input, gate, alpha_ffn_act):
        return gated_swish(silu_input, gate, alpha_ffn_act)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        alpha_ffn_act: float,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = uu.Linear(dim, hidden_dim, bias=False)
        self.w2 = uu.Linear(hidden_dim, dim, bias=False)
        self.w3 = uu.Linear(dim, hidden_dim, bias=False)

        self.alpha_ffn_act = alpha_ffn_act

        self.gated_swish = GatedSwish()

    def forward(self, x):
        # Original code:
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))
        # Gated swish is defined as:
        # F.sigmoid(x * mult) * x * gate
        silu_input = self.w1(x)
        gate = self.w3(x)
        activation = self.gated_swish(silu_input, gate, self.alpha_ffn_act)
        output = self.w2(activation)
        return output

    def init_weights(self):
        for linear in (self.w1, self.w2, self.w3):
            nn.init.normal_(linear.weight, mean=0.0, std=1.0)
            linear.weight = uu.Parameter(linear.weight, "weight")


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: UmupTransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            alpha_ffn_act=model_args.umup_alpha_ffn_act,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = nn.RMSNorm(
            model_args.dim, eps=model_args.norm_eps, elementwise_affine=False
        )
        self.ffn_norm = nn.RMSNorm(
            model_args.dim, eps=model_args.norm_eps, elementwise_affine=False
        )

        # Compute residual scales based on Section G.2.2, Equations (25) - (31)

        alpha_hat_f_squared = (
            2 / (model_args.umup_alpha_res_attn_ratio**2 + 1) * model_args.umup_alpha_res
        )
        alpha_hat_a_squared = (
            model_args.umup_alpha_res_attn_ratio**2 * alpha_hat_f_squared
        )

        # L / 2 = number of transformer layers, each of which is an attention layer and an FFN

        tau_squared_attn_denominator = (
            model_args.n_layers
            + self.layer_id * alpha_hat_a_squared
            + self.layer_id * alpha_hat_f_squared
        )

        tau_squared_attn = alpha_hat_a_squared / tau_squared_attn_denominator
        self.tau_attn = math.sqrt(tau_squared_attn)

        tau_squared_ffn_denominator = (
            model_args.n_layers
            + (self.layer_id + 1) * alpha_hat_a_squared
            + self.layer_id * alpha_hat_f_squared
        )

        tau_squared_ffn = alpha_hat_f_squared / tau_squared_ffn_denominator
        self.tau_ffn = math.sqrt(tau_squared_ffn)

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        x, attn_skip = U.residual_split(x, self.tau_attn)
        x = self.attention_norm(x)
        attn_resid = self.attention(x, cos_sin)
        attn_out = U.residual_add(attn_resid, attn_skip, self.tau_attn)

        h, ffn_skip = U.residual_split(attn_out, self.tau_ffn)
        h = self.ffn_norm(h)
        ffn_resid = self.feed_forward(h)
        ffn_out = U.residual_add(ffn_resid, ffn_skip, self.tau_ffn)

        return ffn_out

    def init_weights(self):
        self.attention.init_weights()
        self.feed_forward.init_weights()


class ReadoutLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        if bias:
            raise ValueError("bias=True not yet supported for this class.")

        weight_data = torch.empty(out_features, in_features, **factory_kwargs)
        self.weight = uu.Parameter(weight_data, mup_type="output", mup_scaling_depth=None)

        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.weight = uu.Parameter(self.weight, "output")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Different for readout
        output_scale = self.in_features**-1
        grad_input_scale = self.out_features ** (-1 / 2)

        weight_grad_reduction_length = x.numel() // self.in_features
        grad_weight_scale = weight_grad_reduction_length ** (-1 / 2)

        x = scale_bwd(x, grad_input_scale)
        weight = scale_bwd(self.weight, grad_weight_scale)
        output = F.linear(x, weight)
        output = scale_fwd(output, output_scale)

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def log_softmax_for_nll(input: torch.Tensor, mult: float, dim: int) -> torch.Tensor:
    """A specialised version of nn.functional.log_softmax, computing a
    unit-scaled softmax, assuming the next operation is U.nll_loss.

    This method always uses separate forward and backward scaling factors.
    """
    dim_size = input.shape[dim]
    input = scale_fwd(input, mult)
    input = scale_bwd(input, dim_size / (dim_size - 1) ** 0.5)
    return F.log_softmax(input, dim)


class UmupTransformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.

    """

    def __init__(self, model_args: UmupTransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id

        self.tok_embeddings = uu.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cos_sin to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cos_sin", self._precompute_cos_sin(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(
            model_args.dim, eps=model_args.norm_eps, elementwise_affine=False
        )

        self.output = ReadoutLinear(model_args.dim, model_args.vocab_size, bias=False)
        self.alpha_loss_softmax = model_args.umup_alpha_loss_softmax
        self.init_weights()

        # The low-bits-training config manager expects all model args to be attributes of the model?

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cos_sin.device
        with torch.device(buffer_device):
            self.freqs_cos_sin = self._precompute_cos_sin()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0)
            self.tok_embeddings.weight.mup_type = "weight"
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        self.output.init_weights()

    def _precompute_cos_sin(self) -> torch.Tensor:
        return precompute_cos_sin(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cos_sin)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h

        return log_softmax_for_nll(output, mult=self.alpha_loss_softmax, dim=-1)

    @classmethod
    def from_model_args(cls, model_args: UmupTransformerModelArgs) -> "UmupTransformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
