#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest
from low_bits_training.models import (
    get_model_config,
    TransformerModelArgs,
    get_train_spec,
)
import transformers
import inspect


@pytest.mark.parametrize("model, flavour", [("llama3_gc", "1B"), ("llama3_gc", "3B")])
def test_get_model_config__models_are_registered(model, flavour):
    # Just test models have been properly added to the collection.
    assert isinstance(get_model_config(model, flavour), TransformerModelArgs)


@pytest.mark.parametrize(
    "flavour, hf_name",
    [("1B", "meta-llama/Llama-3.2-1B"), ("3B", "meta-llama/Llama-3.2-3B")],
)
def test__llama3_config__proper_parameters(flavour, hf_name):
    model = "llama3_gc"
    config = get_model_config(model, flavour)
    hf_config = transformers.LlamaConfig.from_pretrained(hf_name)

    assert config.dim == hf_config.hidden_size
    assert config.n_layers == hf_config.num_hidden_layers
    assert config.n_heads == hf_config.num_attention_heads
    assert config.n_kv_heads == hf_config.num_key_value_heads
    assert config.rope_theta == hf_config.rope_theta
    # FIX: tie embeddings?
    # assert hf_config.tie_word_embeddings


@pytest.mark.parametrize(
    "flavour, hf_name",
    [("1B", "meta-llama/Llama-3.2-1B"), ("3B", "meta-llama/Llama-3.2-3B")],
)
def test__llama3_config__ffn_intermediate_size(flavour, hf_name):
    from torchtitan.models.llama3.model.model import FeedForward

    model = "llama3_gc"
    config = get_model_config(model, flavour)
    hf_config = transformers.LlamaConfig.from_pretrained(hf_name)
    # Build a single FFN layer, to check the intermediate size and TorchTitan math!
    # Copy-paste of the call in TorchTitan Transformer module __init__.
    ffn = FeedForward(
        dim=config.dim,
        hidden_dim=4 * config.dim,
        multiple_of=config.multiple_of,
        ffn_dim_multiplier=config.ffn_dim_multiplier,
    )
    w1_shape = ffn.w1.weight.shape
    assert w1_shape[0] == hf_config.intermediate_size


@pytest.mark.parametrize("name", ["llama3_gc", "llumup3"])
@pytest.mark.parametrize(
    "field",
    [
        "parallelize_fn",
        "pipelining_fn",
        "build_optimizers_fn",
        "build_lr_schedulers_fn",
        "build_dataloader_fn",
        "build_tokenizer_fn",
        "build_loss_fn",
    ],
)
def test__train_specs__method_signature(name, field):
    """Validate that Graphcore defined training specs define methods with the correct signature."""
    ref_spec = get_train_spec("llama3")
    spec = get_train_spec("llama3_gc")

    ref_fn_args = inspect.signature(getattr(ref_spec, field)).parameters
    fn_args = inspect.signature(getattr(spec, field)).parameters

    # Signature of the train spec method should be the same as `llama3` TorchTitan reference.
    assert len(fn_args) == len(ref_fn_args)
    assert list(fn_args.keys()) == list(ref_fn_args.keys())
    # TODO: adding a check on type annotation, taking into account sub-classing.
    for t0, t1 in zip(fn_args.values(), ref_fn_args.values()):
        assert t0 == t1
