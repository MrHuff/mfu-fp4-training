#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import pytest

import torch
from low_bits_training.models.llama3 import llama3_gc_configs, Transformer
from low_bits_training.models import AttentionWithFusedLinear, FeedForwardWithFusedLinear
from low_bits_training.models.fuse_linear import swap_unfused_with_fused
from torchtitan.models.llama3.model.model import (
    Attention,
    FeedForward,
    precompute_freqs_cis,
)


def test_swap_unfused_with_fused():
    model_args = llama3_gc_configs["debugmodel"]
    model_args.vocab_size = 1024
    model = Transformer(model_args)

    swap_unfused_with_fused(model, Attention, AttentionWithFusedLinear)
    swap_unfused_with_fused(model, FeedForward, FeedForwardWithFusedLinear)

    assert hasattr(model.layers["0"].attention, "wqkv")
    assert not hasattr(model.layers["0"].attention, "wq")
    assert isinstance(model.layers["0"].attention, AttentionWithFusedLinear)

    assert hasattr(model.layers["0"].feed_forward, "w_in")
    assert not hasattr(model.layers["0"].feed_forward, "w1")
    assert hasattr(model.layers["0"].feed_forward, "w_out")
    assert isinstance(model.layers["0"].feed_forward, FeedForwardWithFusedLinear)


def _generate_modules(ref_cls, new_cls):
    model_args = llama3_gc_configs["debugmodel"]
    init_std = 0.02 / (2 * model_args.n_layers) ** 0.5
    if ref_cls == Attention:
        ref_mod = ref_cls(model_args)
    elif ref_cls == FeedForward:
        ref_mod = ref_cls(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
    else:
        raise ValueError(f"Unsupported reference class: {ref_cls}")

    ref_mod.init_weights(init_std=init_std)
    new_mod = new_cls.from_unfused(ref_mod)
    return ref_mod, new_mod


@pytest.mark.parametrize(
    "ref_cls,new_cls",
    [(Attention, AttentionWithFusedLinear), (FeedForward, FeedForwardWithFusedLinear)],
)
def test_numeric_equivalence(
    ref_cls: Attention | FeedForward,
    new_cls: AttentionWithFusedLinear | FeedForwardWithFusedLinear,
):
    torch.manual_seed(1472)
    ref_mod, new_mod = _generate_modules(ref_cls, new_cls)

    model_args = llama3_gc_configs["unit_test"]
    init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    if ref_cls == Attention:
        ref_mod = ref_cls(model_args)
    elif ref_cls == FeedForward:
        ref_mod = ref_cls(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
    else:
        raise ValueError(f"Unsupported reference class: {ref_cls}")
    ref_mod.init_weights(init_std=init_std)
    new_mod = new_cls.from_unfused(ref_mod)

    x = torch.randn(8, 256, model_args.dim, requires_grad=True)
    grad = torch.randn_like(x)
    if ref_cls == Attention:
        kwargs = dict(
            freqs_cis=precompute_freqs_cis(
                model_args.dim // model_args.n_heads,
                model_args.max_seq_len,
                model_args.rope_theta,
            ),
            attention_masks=None,
        )
    else:
        kwargs = {}

    ref_out = ref_mod(x, **kwargs)
    ref_out.backward(grad)
    ref_grad_x = x.grad.clone()
    if ref_cls == Attention:
        ref_grad_weight = torch.cat(
            [
                ref_mod.wq.weight.grad.clone(),
                ref_mod.wk.weight.grad.clone(),
                ref_mod.wv.weight.grad.clone(),
            ]
        )
    elif ref_cls == FeedForward:
        ref_grad_weight = torch.cat(
            [
                ref_mod.w1.weight.grad.clone(),
                ref_mod.w3.weight.grad.clone(),
            ]
        )
    x.grad.zero_()

    new_out = new_mod(x, **kwargs)
    new_out.backward(grad)
    new_grad_x = x.grad.clone()
    if ref_cls == Attention:
        new_grad_weight = new_mod.wqkv.weight.grad.clone()
    elif ref_cls == FeedForward:
        new_grad_weight = new_mod.w_in.weight.grad.clone()

    with torch.no_grad():
        assert ref_out.norm() > 1e-5

    torch.testing.assert_close(ref_out, new_out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ref_grad_x, new_grad_x, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ref_grad_weight, new_grad_weight, rtol=1e-6, atol=1e-6)


def test_init_weights():
    torch.manual_seed(1472)

    """Check that weight initialisation draws from the same distribution"""
    attn_ref_mod, attn_new_mod = _generate_modules(Attention, AttentionWithFusedLinear)
    ffn_ref_mod, ffn_new_mod = _generate_modules(FeedForward, FeedForwardWithFusedLinear)

    with torch.no_grad():
        # carries over initialised weights from unfused impl
        wqkv_std_ref = torch.stack(
            [
                attn_ref_mod.wq.weight,
                attn_ref_mod.wk.weight,
                attn_ref_mod.wv.weight,
            ]
        ).std(dim=(1, 2))
        w_in_std_ref = torch.stack(
            [
                ffn_ref_mod.w1.weight,
                ffn_ref_mod.w3.weight,
            ],
        ).std(dim=(1, 2))

        # reinitialise weights in fused impl
        model_args = llama3_gc_configs["debugmodel"]
        init_std = 0.02 / (2 * model_args.n_layers) ** 0.5
        attn_new_mod.init_weights(init_std=init_std)
        ffn_new_mod.init_weights(init_std=init_std)
        wqkv_std_new = torch.stack(
            torch.chunk(attn_new_mod.wqkv.weight, chunks=3, dim=0)
        ).std(dim=(1, 2))
        w_in_std_new = torch.stack(
            torch.chunk(ffn_new_mod.w_in.weight, chunks=2, dim=0)
        ).std(dim=(1, 2))

        # checks that std of weights are close
        # increase tolerance as std estimate has high variance
        torch.testing.assert_close(wqkv_std_ref, wqkv_std_new, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w_in_std_ref, w_in_std_new, rtol=1e-4, atol=1e-4)
