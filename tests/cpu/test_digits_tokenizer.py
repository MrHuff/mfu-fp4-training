#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import low_bits_training  # noqa: F401
from digits_tokenizer import DigitsTokenizer


def test__digits_tokenizer__encode():
    tokenizer = DigitsTokenizer()
    token_ids = tokenizer.encode("1,2,3,44,5,6")
    assert token_ids == [1, 2, 3, 44, 5, 6]


def test__digits_tokenizer__decode():
    tokenizer = DigitsTokenizer()
    text = tokenizer.decode([1, 2, 3, 44, 5, 6])
    assert text == "1,2,3,44,5,6"
