#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from torchtitan.components.tokenizer import BaseTokenizer


class DigitsTokenizer(BaseTokenizer):
    """Very basic tokenizer, useful for testing dataset pipeline + tokenizer.

    Transforming a string of the form `1,3,2,5` into [1,3,2,5] tokens.
    """

    def __init__(self):
        super().__init__()
        # Max. token id (except BOS, EOS, ...)
        self._max_token_id = 65535
        self._num_special_tokens = 0

    def encode(self, *args, **kwargs) -> list[int]:
        # Extract arguments
        if len(args) >= 1:
            text = args[0]
        else:
            text = kwargs.get("text", "")

        token_ids = [int(v) for v in text.split(",")]
        assert all([v <= self._max_token_id for v in token_ids])
        return token_ids

    def decode(self, *args, **kwargs) -> str:
        # Extract token_ids from arguments
        if len(args) >= 1:
            token_ids = args[0]
        else:
            token_ids = kwargs.pop("token_ids", [])
        text = ",".join([str(v) for v in token_ids])
        return text

    def get_vocab_size(self) -> int:
        return self._max_token_id + self._num_special_tokens

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()
