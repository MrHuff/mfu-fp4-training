#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import pytest
import torch
import torch.nn as nn
from typing import Dict, List

from low_bits_training.metrics import append_model_metrics
from low_bits_training.layer_stats import convert_model_with_stats, resolve_layer_patterns
from low_bits_training.models import Transformer, TransformerModelArgs


def run_training_step(
    model: nn.Module, device: torch.device, batch_size: int = 8, seq_len: int = 128
) -> Dict[str, float]:
    """Run a single training step and return metrics."""
    # Create dummy data
    input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Forward pass
    output = model(input_ids)
    loss = nn.functional.cross_entropy(
        output.reshape(-1, output.size(-1)), labels.reshape(-1)
    )

    # Backward pass
    loss.backward()

    # Create metrics and append model stats
    metrics = {"step": 0, "loss": loss.item(), "lr": 0.001}

    metrics = append_model_metrics(metrics, model)

    return metrics


class TestLayerPatternMatching:
    """Test suite for layer pattern matching functionality."""

    @pytest.fixture
    def model(self):
        """Create a test transformer model."""
        model_args = TransformerModelArgs(n_layers=4, dim=256, vocab_size=256)
        return Transformer(model_args)

    @pytest.fixture
    def all_layer_names(self, model):
        """Get all layer names from the model."""
        return [name for name, _ in model.named_modules() if name]

    def test_exact_match(self, model):
        """Test exact layer name matching."""
        patterns = ["tok_embeddings", "output"]
        matched = resolve_layer_patterns(model, patterns)

        assert "tok_embeddings" in matched
        assert "output" in matched
        assert len(matched) == 2

    def test_wildcard_match(self, model):
        """Test wildcard pattern matching."""
        patterns = ["*norm*"]
        matched = resolve_layer_patterns(model, patterns)

        # Should match attention_norm and ffn_norm layers
        norm_layers = [name for name in matched if "norm" in name]
        assert (
            len(norm_layers) == model.n_layers * 2 + 1
        )  # 2 norms per layer + 1 final norm
        assert all("norm" in name for name in norm_layers)

    def test_nested_module_match(self, model):
        """Test matching nested modules."""
        patterns = ["layers.*.attention.*"]
        matched = resolve_layer_patterns(model, patterns)

        # Should match all attention sublayers
        attention_layers = [name for name in matched if "attention" in name]
        assert (
            len(attention_layers) == model.n_layers * 5
        )  # 4 linears + 1 sdpa per attn layer
        assert all(name.startswith("layers.") for name in attention_layers)

    def test_specific_layer_indices(self, model):
        """Test matching specific layer indices."""
        patterns = ["layers.0.*", "layers.2.*"]
        matched = resolve_layer_patterns(model, patterns)

        layer_0 = [name for name in matched if name.startswith("layers.0")]
        layer_2 = [name for name in matched if name.startswith("layers.2")]

        assert len(layer_0) == 12  # 2 blocks + 6 modules per attn + 4 modules per ffn
        assert len(layer_2) == 12
        # Should not match layers 1 or 3
        assert not any(name.startswith("layers.1") for name in matched)
        assert not any(name.startswith("layers.3") for name in matched)

    def test_multiple_patterns(self, model):
        """Test combining multiple patterns."""
        patterns = ["*.attention", "output", "*norm*"]
        matched = resolve_layer_patterns(model, patterns)

        assert "output" in matched
        attention_layers = [name for name in matched if "attention" in name]
        norm_layers = [name for name in matched if "norm" in name]

        assert (
            len(attention_layers) == model.n_layers * 2
        )  # attention blocks + attention norms
        assert len(norm_layers) == 9

    def test_question_mark_wildcard(self, model):
        """Test single character wildcard."""
        patterns = ["layers.?.*"]
        matched = resolve_layer_patterns(model, patterns)

        # Should match layers.0, layers.1, layers.2, layers.3
        assert (
            len(matched) == model.n_layers * 12
        )  # 2 blocks + 6 modules per attn + 4 modules per ff
        assert all(name.startswith("layers.") for name in matched)

    def test_empty_pattern(self, model):
        """Test with empty pattern list."""
        patterns = []
        matched = resolve_layer_patterns(model, patterns)

        assert len(matched) == 0

    def test_no_match(self, model):
        """Test pattern that doesn't match anything."""
        patterns = ["nonexistent.layer.*"]
        matched = resolve_layer_patterns(model, patterns)

        assert len(matched) == 0

    def test_order_preservation(self, model, all_layer_names):
        """Test that layer order is preserved from the model."""
        # Match layers in different pattern order
        patterns = ["output", "tok_embeddings", "layers.*.attention"]
        matched = resolve_layer_patterns(model, patterns)

        # Verify matched layers appear in the same order as in the model
        matched_indices = [all_layer_names.index(name) for name in matched]
        assert matched_indices == sorted(matched_indices), "Layer order not preserved!"

        # Verify tok_embeddings comes before output in results (as it does in model)
        assert matched.index("tok_embeddings") < matched.index("output")

    def test_order_with_overlapping_patterns(self, model, all_layer_names):
        """Test order preservation with patterns that match overlapping layers."""
        # These patterns will match some of the same layers
        patterns = ["layers.*", "*.attention", "layers.*.feed_forward.*"]
        matched = resolve_layer_patterns(model, patterns)

        # Verify order is preserved
        matched_indices = [all_layer_names.index(name) for name in matched]
        assert matched_indices == sorted(
            matched_indices
        ), "Layer order not preserved with overlapping patterns!"


class TestStatsGathering:
    """Test suite for statistics gathering."""

    @pytest.fixture
    def layer_names(self):
        """Layer names to monitor."""
        return [
            "tok_embeddings",
            "layers.0.attention",
            "layers.0.attention_norm",
            "layers.1.attention.wo",
            "layers.2.feed_forward.w2",
            "layers.3.ffn_norm",
            "output",
        ]

    @pytest.fixture
    def expected_stats(self):
        """Expected statistics types."""
        return ["mean", "std", "absmax", "norm"]

    def test_stats_gathering(self, layer_names, expected_stats):
        """Test statistics gathering on CPU."""
        # Create model
        model_args = TransformerModelArgs(n_layers=4, dim=256, vocab_size=256)
        model = Transformer(model_args)

        # Convert model with stats
        model, _ = convert_model_with_stats(
            model,
            layer_names,
            gather_forward=True,
            gather_backward=True,
            gather_interval=1,
        )

        # Run training step
        device = torch.device("cpu")
        metrics = run_training_step(model, device, batch_size=4, seq_len=64)

        # Verify statistics were collected
        self._verify_metrics(metrics, layer_names, expected_stats)

    def _verify_metrics(
        self, metrics: Dict, layer_names: List[str], expected_stats: List[str]
    ):
        """Verify that metrics contain expected statistics."""
        # Check that we have loss and other base metrics
        assert "loss" in metrics
        assert "step" in metrics
        assert "lr" in metrics

        # Check for layer statistics
        stats_found = False
        for layer_name in layer_names:
            for direction in ["forward", "backward"]:
                for stat in expected_stats:
                    key = f"model_metrics/{layer_name}_{direction}.{stat}"
                    if key in metrics:
                        stats_found = True
                        # Verify it's a number
                        assert isinstance(metrics[key], (int, float))
                        # Verify it's not NaN
                        assert not torch.isnan(torch.tensor(metrics[key]))

        # Ensure we found at least some statistics
        assert stats_found, f"No statistics found in metrics: {metrics.keys()}"

    def test_gather_interval(self, layer_names, expected_stats):
        """Test that gather_interval works correctly."""
        # Create model with gather_interval=3
        model_args = TransformerModelArgs(n_layers=4, dim=256, vocab_size=256)
        model = Transformer(model_args)
        model, _ = convert_model_with_stats(
            model,
            layer_names[:3],  # Use fewer layers for speed
            gather_forward=True,
            gather_backward=False,
            gather_interval=3,
        )

        device = torch.device("cpu")

        # Run multiple steps
        for step in range(5):
            metrics = run_training_step(model, device, batch_size=2, seq_len=32)

            # Should only have stats on steps 0 and 3
            if step % 3 == 0:
                (
                    self._verify_metrics(metrics, layer_names, expected_stats),
                    f"Expected stats at step {step}",
                )
            else:
                try:
                    self._verify_metrics(metrics, layer_names, expected_stats)
                    assert False, f"Unexpected stats at step {step}"

                except AssertionError:
                    pass  # Expected no stats on these steps

    def test_forward_only(self, layer_names):
        """Test gathering only forward pass statistics."""
        model_args = TransformerModelArgs(n_layers=4, dim=256, vocab_size=256)
        model = Transformer(model_args)
        model, _ = convert_model_with_stats(
            model,
            layer_names[:3],
            gather_forward=True,
            gather_backward=False,
            gather_interval=1,
        )

        device = torch.device("cpu")
        metrics = run_training_step(model, device, batch_size=2, seq_len=32)

        # Check we only have forward stats
        for key in metrics.keys():
            if "layer_stats" in key:
                assert "_forward" in key
                assert "_backward" not in key

    def test_backward_only(self, layer_names):
        """Test gathering only backward pass statistics."""
        model_args = TransformerModelArgs(n_layers=4, dim=256, vocab_size=256)
        model = Transformer(model_args)
        model, _ = convert_model_with_stats(
            model,
            layer_names[:3],
            gather_forward=False,
            gather_backward=True,
            gather_interval=1,
        )

        device = torch.device("cpu")
        metrics = run_training_step(model, device, batch_size=2, seq_len=32)

        # Check we only have backward stats
        for key in metrics.keys():
            if "layer_stats" in key:
                assert "_backward" in key
                assert "_forward" not in key
