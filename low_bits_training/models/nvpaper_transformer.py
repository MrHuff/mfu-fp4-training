from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    ScaledDotProductAttentionWrapper,
)
from torchtitan.models.llama3 import parallelize_llama
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import AttentionMasksType, BaseModelArgs
from torchtitan.protocols.train_spec import ModelProtocol, register_train_spec, TrainSpec
from torchtitan.tools.logging import logger

from low_bits_training.components.tokenizer import build_hf_or_tiktoken_tokenizer
from low_bits_training.datasets import build_dataloader
from low_bits_training.lr_scheduler import build_lr_schedulers
from low_bits_training.metrics import build_metrics_processor
from low_bits_training.optimizer import build_optimizers

from .models import add_model_config


def safe_trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        safe_trunc_normal_(tensor.to_local(), mean=mean, std=std, a=a, b=b)
        return tensor
    if tensor.is_cuda and tensor.dtype in (torch.bfloat16, torch.float16):
        tmp = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        nn.init.trunc_normal_(tmp, mean=mean, std=std, a=a, b=b)
        tensor.copy_(tmp.to(tensor.dtype))
        return tensor
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


@dataclass
class PaperTransformerModelArgs(BaseModelArgs):
    dim: int = 2048
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 131072
    ffn_hidden_dim: int = 6144
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 8192
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 2
    enable_weight_tying: bool = False

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                "Sequence length %s exceeds original maximum %s.",
                seq_len,
                self.max_seq_len,
            )
        self.max_seq_len = seq_len

    def get_nparams_and_flops(
        self,
        model: nn.Module,
        seq_len: int,
    ) -> tuple[int, float]:
        return get_dense_model_nparams_and_flops(
            self,
            model,
            2 * (self.dim // self.n_heads),
            seq_len,
        )


class Attention(nn.Module):
    def __init__(self, model_args: PaperTransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.wq = nn.Linear(model_args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, model_args.dim, bias=False)
        self.use_flex_attn = model_args.use_flex_attn
        self.inner_attention = (
            FlexAttentionWrapper()
            if self.use_flex_attn
            else ScaledDotProductAttentionWrapper()
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            safe_trunc_normal_(linear.weight, mean=0.0, std=0.02)
        safe_trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        assert isinstance(attention_masks, BlockMask) or attention_masks is None
        if self.use_flex_attn:
            assert isinstance(attention_masks, BlockMask), attention_masks
            output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
        else:
            assert attention_masks is None
            output = self.inner_attention(xq, xk, xv)

        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.relu(x).square()
        return self.w2(x)

    def init_weights(self, init_std: float):
        safe_trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        safe_trunc_normal_(self.w2.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: PaperTransformerModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(model_args.dim, model_args.ffn_hidden_dim)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / math.sqrt(2 * (layer_id + 1))
        else:
            self.weight_init_std = 0.02 / math.sqrt(2 * model_args.n_layers)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, attention_masks)
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class PaperTransformer(nn.Module, ModelProtocol):
    def __init__(self, model_args: PaperTransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)
        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            layer.init_weights()
        self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        safe_trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-3 * final_out_std,
            b=3 * final_out_std,
        )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward_hidden_states_for_cce(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks=attention_masks)
        return self.norm(h)

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                batch = 1
            case "block_causal":
                batch = input_batch.shape[0]
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(f"Unknown attention mask type: {self.model_args.attn_mask_type}")
        return create_attention_mask(
            and_masks(*mask_mods),
            batch,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        h = self.forward_hidden_states_for_cce(tokens, attention_masks=attention_masks)
        return self.output(h)


def parallelize_paper_transformer(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if parallel_dims.tp_enabled or parallel_dims.pp_enabled or parallel_dims.cp_enabled:
        raise NotImplementedError(
            "PaperTransformer FP4 path currently supports DP/FSDP only; "
            "TP/PP/CP need explicit FP4-kernel validation."
        )
    return parallelize_llama(model, parallel_dims, job_config)


register_train_spec(
    "nvpaper_transformer_gc",
    TrainSpec(
        model_cls=PaperTransformer,
        model_args={},
        parallelize_fn=parallelize_paper_transformer,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_or_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_metrics_processor_fn=build_metrics_processor,
        build_validator_fn=build_validator,
        state_dict_adapter=None,
    ),
)

add_model_config(
    "nvpaper_transformer_gc",
    "1.2B_paper",
    PaperTransformerModelArgs(),
)

add_model_config(
    "nvpaper_transformer_gc",
    "1.2B_paper_ffn8192",
    PaperTransformerModelArgs(ffn_hidden_dim=8192),
)

add_model_config(
    "nvpaper_transformer_gc",
    "1.2B_paper_mha64_ffn8192",
    PaperTransformerModelArgs(n_heads=32, n_kv_heads=32, ffn_hidden_dim=8192),
)
