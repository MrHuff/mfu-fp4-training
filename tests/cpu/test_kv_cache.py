#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest
import torch
from pathlib import Path
import numpy as np
import random

import low_bits_training  # noqa

from low_bits_training.generate.kv_cache import (
    attention_forward_with_kv_cache,
    apply_rotary_emb_with_cache,
    generate_with_kv_cache,
)
from torchtitan.models.llama3.model.model import Attention, TransformerModelArgs

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model_config_path():
    # Use the debug model config
    return str(REPO_ROOT / "train_configs" / "debug_model.toml")


def test_kv_cache_token_divergence(model_config_path, monkeypatch: pytest.MonkeyPatch):
    """Test that KV cache implementation produces expected token divergence."""
    examples_dir = str(REPO_ROOT / "examples")
    monkeypatch.syspath_prepend(examples_dir)
    from kv_cache_example import KVCacheExample

    # Initialize the KV cache example with debug configuration
    example = KVCacheExample(
        config_path=model_config_path,
        vocab_size=1000,
        device="cpu",  # Use CPU for testing
    )
    generation_length = 10
    # Run comparison with small sequence length for testing
    results = example.run_comparison(
        sequence_length=128, batch_size=1, generation_length=generation_length
    )

    # Check that the first token divergence is exactly 10
    # This means the first 10 tokens generated are the same
    assert (
        results["first_token_diff"][0] == generation_length
    ).all(), f"Expected first token divergence to be {generation_length}, but got {results['first_token_diff'][0]}"

    # Verify outputs match in length
    assert results[
        "outputs_match"
    ], "Outputs from KV cache and non-KV cache runs should match"


def get_attention_test_inputs(batch_size, seq_len, max_seq_len=None):
    # Set the seed before the layers and the inputs are initialized for repeatability
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(1472)
    np.random.seed(1472)
    random.seed(1472)
    torch.use_deterministic_algorithms(True)

    # Create a small attention configuration
    torch.manual_seed(1473)  # for test reproducibility
    model_args = TransformerModelArgs(
        dim=128,
        n_heads=8,
        n_kv_heads=4,
        norm_eps=1e-5,
    )
    if max_seq_len is None:
        max_seq_len = seq_len * 2
    # Create attention layer and set to eval mode
    attn = Attention(model_args)
    attn.eval()
    attn.debug_print = lambda *args, **kwargs: None

    # Create test inputs
    x = torch.randn(batch_size, seq_len, model_args.dim)
    # Create rotary embeddings (needed for both forward passes)
    # freqs_cis = torch.ones(max_seq_len, model_args.n_heads)
    freqs_cis = torch.linspace(-2, 2, steps=max_seq_len * model_args.n_heads).reshape(
        max_seq_len, model_args.n_heads
    )

    return attn, x, freqs_cis


class TestAttentionForwardWithKVCache:
    @torch.no_grad()
    def test_prefill_cache(self):
        """Unit test that the output of the attention forward with and without kv cache are the same"""
        batch_size = 2
        seq_len = 16
        attn, x, freqs_cis = get_attention_test_inputs(batch_size, seq_len)

        # Run with the original forward function
        output_no_cache = attn(x, freqs_cis, attention_masks=None)
        # Enable KV cache and run again
        # Monkey patch the forward function
        output_prefill = attention_forward_with_kv_cache(attn, x, freqs_cis)
        # Compare outputs
        torch.testing.assert_close(
            output_no_cache,
            output_prefill,
            rtol=1e-7,  # Relative tolerance
            atol=1e-7,  # Absolute tolerance
            msg="Attention outputs from reference and prefill cache should match within numerical tolerance",
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    @torch.no_grad()
    def test_decode_cache(self, batch_size):
        """Unit test that the output of the attention forward with and without kv cache are the same"""
        seq_len = 16
        attn, x, freqs_cis = get_attention_test_inputs(batch_size, seq_len)
        output_no_cache = attn(x, freqs_cis, attention_masks=None)
        attn.seq_lens = torch.ones([batch_size], dtype=torch.int64) * (seq_len - 1)
        _ = attention_forward_with_kv_cache(attn, x[:, :-1], freqs_cis)
        attn.seq_lens = torch.ones([batch_size], dtype=torch.int64) * (seq_len)
        output_decode = attention_forward_with_kv_cache(attn, x[:, -1:], freqs_cis)
        torch.testing.assert_close(
            output_no_cache[:, [-1]],
            output_decode,
            rtol=1e-7,  # Relative tolerance
            atol=1e-7,  # Absolute tolerance
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    @torch.no_grad()
    def test_rotary_embedding_with_kv_cache(self, batch_size):
        """Unit test that the rope gives the same result in prefill and decode

        We test that given the same input queries
        """
        seq_len = 16
        attn, x, freqs_cis = get_attention_test_inputs(batch_size, seq_len)
        # Test setup calcute
        bs, seqlen, _ = x.shape
        xq_ = attn.wq(x)
        xk_ = attn.wk(x)
        xq_ = xq_.view(bs, seqlen, -1, attn.head_dim)
        xk_ = xk_.view(bs, seqlen, -1, attn.head_dim)
        # Calculate rotary embeddings without cache
        xq_ref, xk_ref = apply_rotary_emb_with_cache(
            xq_.clone(), xk_.clone(), freqs_cis, is_decode=False
        )
        seq_lens = torch.ones([batch_size], dtype=torch.int64) * (seq_len)
        xq, xk = apply_rotary_emb_with_cache(
            xq_.clone()[:, [-1], ...],
            xk_.clone(),
            freqs_cis,
            is_decode=True,
            seq_lens=seq_lens,
        )

        torch.testing.assert_close(
            xk,
            xk_ref,
            rtol=1e-7,
            atol=1e-7,
        )
        torch.testing.assert_close(
            xq_ref[:, [-1]],
            xq,
            rtol=1e-7,
            atol=1e-7,
        )

    @torch.no_grad()
    @pytest.mark.parametrize("prefill_default_token", [0, 1])
    def test_decode_ragged_sequence(self, prefill_default_token):
        """Check the correct implementation of ragged sequence decoding with KV cache"""
        batch_size = 2
        seq_len = 16
        ragged_seq_lens = torch.tensor([10, 12], dtype=torch.int64)
        attn, x, freqs_cis = get_attention_test_inputs(batch_size, seq_len)

        reference_output = attn(x, freqs_cis, attention_masks=None)
        # Prefill and truncate the input to make sure the cache is not cheating
        # and seeing the entire sequence - we set it to 0 and 1 to make sure
        # that we are not zeroing out part of the computation to reach equivalence
        x_for_prefill = x.clone()
        for i, s in enumerate(ragged_seq_lens):
            x_for_prefill[i, s - 1 :, ...] = prefill_default_token
        _ = attention_forward_with_kv_cache(
            attn, x_for_prefill, freqs_cis
        )  # Populates the KV cache
        if prefill_default_token == 0:
            for i, s in enumerate(ragged_seq_lens):
                assert (attn.k_cache[i, s - 1, ...] == 0.0).all()
                assert (attn.v_cache[i, s - 1, ...] == 0.0).all()

        attn.seq_lens = ragged_seq_lens
        ragged_x = torch.cat(
            [
                x[i, s - 1, ...].unsqueeze(0).unsqueeze(0)
                for i, s in enumerate(ragged_seq_lens)
            ],
            dim=0,
        )
        output_decode = attention_forward_with_kv_cache(attn, ragged_x, freqs_cis)
        ragged_reference_output = torch.cat(
            [
                reference_output[i, s - 1, ...].unsqueeze(0).unsqueeze(0)
                for i, s in enumerate(ragged_seq_lens)
            ]
        )
        torch.testing.assert_close(
            ragged_reference_output,
            output_decode,
        )


class DummyIncrementModel(torch.nn.Module):
    """Dummy model that returns logits for token = last_input_token + 1"""

    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, seq_lens=None):
        batch_size, seq_len = input_ids.shape
        # Get the last token in each sequence

        # Create logits tensor filled with very negative values
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )

        # Set high probability for next_token = last_token + 1 (with wraparound)
        next_tokens = (input_ids + 1) % self.vocab_size

        # Set the logits for the last position in the sequence
        for i in range(batch_size):
            for j, next_token in enumerate(next_tokens[i]):
                logits[i, j, next_token] = 10.0  # High logit for the next token

        return logits

    def clear_kv_cache(self):
        """Required for KV cache compatibility"""
        pass


class TestGenerateWithKVCache:
    """Tests the generate_with_kv_cache function with a dummy"""

    @pytest.mark.parametrize("use_cache", [False, True])
    @pytest.mark.parametrize("ragged_input", [False, True])
    def test_eos_token_termination_multiple_ids(self, use_cache, ragged_input):
        """Test that generation terminates correctly with multiple EOS token IDs"""
        vocab_size = 20
        model = DummyIncrementModel(vocab_size=vocab_size)

        # Test with multiple EOS tokens
        eos_token_ids = [5, 12]  # Will terminate when we hit token 5 or 12

        # Start with different tokens for each batch item
        batch_size = 3

        input_ids = [
            [2],  # Will generate: 3, 4, 5 (stops at 5 - EOS)
            [10],  # Will generate: 11, 12 (stops at 12 - EOS)
            [1],  # Will generate: 2, 3, 4, 5 (stops at 5 - EOS)
        ]
        seq_lens = None
        if ragged_input:
            # Extend the first sequence with the next token it will generate
            # (this keeps the expected output the same) - pad the other sequences
            seq_lens = [2, 1, 1]
            input_ids[0].append(3)
            input_ids[1].append(0)
            input_ids[2].append(0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        expected_seqs = [
            # Sequence 0: [2] -> [2, 3, 4, 5] (stops at 5)
            torch.tensor([2, 3, 4, 5], dtype=torch.long),
            # Sequence 1: [10] -> [10, 11, 12] (stops at 12)
            torch.tensor([10, 11, 12], dtype=torch.long),
            # Sequence 2: [1] -> [1, 2, 3, 4, 5] (stops at 5)
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        ]

        max_new_tokens = 10  # More than needed to check early termination

        output = generate_with_kv_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_ids,
            use_kv_cache=use_cache,
            seq_lens=seq_lens,
            progress_bar=False,
        )

        fails = []
        for i, expected_seq in enumerate(expected_seqs):
            # Check that sequences terminated at the right points
            actual_seq = output[i, : len(expected_seq)]
            if not torch.equal(actual_seq, expected_seq):
                fails.append(f"Expected {expected_seq}, got {actual_seq}")
            if not (output[i, len(expected_seq) :] == 0).all():
                fails.append(
                    f"Sequence didn't terminate early enough: Expected {expected_seq}, {output[i,:]=}"
                )
        assert not fails, f"{len(fails)} errors: {fails}"
        # Verify that generation didn't continue beyond EOS tokens
        # All sequences should be shorter than max_new_tokens + 1 (original length)
        for i in range(batch_size):
            # Find the first occurrence of padding (assuming padding with 0 or last valid token repetition)
            seq = output[i]
            # Check that we don't have more than 5 tokens (1 original + 4 generated max before hitting EOS)
            non_zero_tokens = (seq != 0).sum()
            assert (
                non_zero_tokens <= 5
            ), f"Sequence {i} didn't terminate early enough: {seq}"

    def test_eos_token_termination_single_id(self):
        """Test that generation terminates correctly with a single EOS token ID"""
        vocab_size = 10
        model = DummyIncrementModel(vocab_size=vocab_size)

        eos_token_id = 7
        input_ids = torch.tensor(
            [[5]], dtype=torch.long
        )  # Will generate: 6, 7 (stops at 7)

        output = generate_with_kv_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=5,
            eos_token_id=eos_token_id,
            use_kv_cache=False,
            progress_bar=False,
        )

        expected = torch.tensor([5, 6, 7], dtype=torch.long)
        actual = output[0, : len(expected)]
        assert torch.equal(actual, expected), f"Expected {expected}, got {actual}"

    def test_no_eos_termination(self):
        """Test that generation continues for max_new_tokens when no EOS is hit"""
        vocab_size = 10
        model = DummyIncrementModel(vocab_size=vocab_size)

        # Use EOS token that won't be hit in the range we're testing
        eos_token_id = 99  # Won't be hit with vocab_size=10
        input_ids = torch.tensor([[0]], dtype=torch.long)
        max_new_tokens = 3

        output = generate_with_kv_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            use_kv_cache=False,
            progress_bar=False,
        )

        # Should generate exactly max_new_tokens
        expected = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 0 + 3 new tokens
        actual = output[0, : len(expected)]
        assert torch.equal(actual, expected), f"Expected {expected}, got {actual}"
        assert (
            output.shape[1] == 1 + max_new_tokens
        ), f"Expected {1 + max_new_tokens} tokens, got {output.shape[1]}"
