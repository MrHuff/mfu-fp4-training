#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from low_bits_training.config import ConfigManager
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from low_bits_training.components.tokenizer import (
    build_hf_or_tiktoken_tokenizer,
    TikTokenizer,
)


def test__build_hf_or_tiktoken_tokenizer__backward_compatible_tiktoken():
    filename = "./tests/assets/unit_test_tiktoken.model"
    cfg = ConfigManager().parse_args(["--model.tokenizer_path", filename])
    tokenizer = build_hf_or_tiktoken_tokenizer(cfg)

    assert isinstance(tokenizer, TikTokenizer)
    assert tokenizer.get_vocab_size() == 300
    assert tokenizer.decode([0, 1]) == "Hello world"
    assert tokenizer.encode("Hello world", add_eos=True, add_bos=True) == [44, 0, 1, 45]


def test__build_hf_or_tiktoken_tokenizer__support_hf_tokenizer():
    filename = "./tests/assets/test-tokenizer"
    cfg = ConfigManager().parse_args(["--model.tokenizer_path", filename])
    tokenizer = build_hf_or_tiktoken_tokenizer(cfg)

    assert isinstance(tokenizer, HuggingFaceTokenizer)
    assert tokenizer.get_vocab_size() == 300
    assert tokenizer.decode([0, 1]) == "Hello world"
    assert tokenizer.encode("Hello world", add_eos=True, add_bos=True) == [44, 0, 1, 45]


def test__build_hf_or_tiktoken_tokenizer__support_hf_assets_path():
    filename = "./tests/assets/test-tokenizer"
    cfg = ConfigManager().parse_args(["--model.hf_assets_path", filename])
    tokenizer = build_hf_or_tiktoken_tokenizer(cfg)

    assert isinstance(tokenizer, HuggingFaceTokenizer)
    assert tokenizer.get_vocab_size() == 300


def test__debug_tiktoken_tokenizer__properties():
    filename = "./tests/assets/torchtitan_test_tiktoken.model"
    cfg = ConfigManager().parse_args(["--model.tokenizer_path", filename])
    tokenizer = build_hf_or_tiktoken_tokenizer(cfg)

    assert isinstance(tokenizer, TikTokenizer)
    assert tokenizer.get_vocab_size() == 2256


def test__debug_tiktoken_tokenizer__encode_c4_test_result():
    filename = "./tests/assets/torchtitan_test_tiktoken.model"
    cfg = ConfigManager().parse_args(["--model.tokenizer_path", filename])
    tokenizer = build_hf_or_tiktoken_tokenizer(cfg)

    # Example from the test C4 dataset => make sure the debug tokenizer does not change.
    tokens = tokenizer.encode(
        "Beginners BBQ Class Taking Place in Missoula!", add_bos=True, add_eos=True
    )
    assert len(tokens) == 26
    assert tokens == [
        2000,
        66,
        101,
        103,
        259,
        110,
        495,
        447,
        66,
        81,
        497,
        108,
        520,
        323,
        762,
        390,
        108,
        707,
        292,
        460,
        1112,
        266,
        108,
        97,
        33,
        2001,
    ]
