#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Dict, List
import functools

from low_bits_training.metrics import append_model_metrics
from low_bits_training.layer_stats import convert_model_with_stats
from low_bits_training.models import Transformer, TransformerModelArgs
from torchtitan.models.llama3.model.model import TransformerBlock


def setup_distributed(monkeypatch, rank: int, world_size: int):
    """Initialize distributed training."""
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12355")

    # Initialize process group
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_training_step(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    seq_len: int = 128,
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
            "layers.2.ff.w2",
            "layers.3.ffn_norm",
            "output",
        ]

    @pytest.fixture
    def expected_stats(self):
        """Expected statistics types."""
        return ["mean", "std", "absmax", "norm"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_gpu_stats_gathering(self, layer_names, expected_stats):
        """Test statistics gathering on single GPU."""
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

        # Move to GPU
        device = torch.device("cuda:0")
        model = model.to(device)

        # Run training step
        metrics = run_training_step(model, device)

        # Verify statistics were collected
        self._verify_metrics(metrics, layer_names, expected_stats)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need at least 2 GPUs",
    )
    def test_fsdp_multi_gpu_stats_gathering(
        self, monkeypatch, layer_names, expected_stats
    ):
        """Test statistics gathering with FSDP on multiple GPUs."""
        import torch.multiprocessing as mp

        world_size = 2
        mp.spawn(
            self._fsdp_worker,
            args=(monkeypatch, world_size, layer_names, expected_stats),
            nprocs=world_size,
            join=True,
        )

    def _fsdp_worker(
        self,
        rank: int,
        monkeypatch,
        world_size: int,
        layer_names: List[str],
        expected_stats: List[str],
    ):
        """Worker function for FSDP testing."""
        # Setup distributed
        setup_distributed(monkeypatch, rank, world_size)

        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Create model
        model_args = TransformerModelArgs(n_layers=4, dim=256, vocab_size=256)
        model = Transformer(model_args)

        # Convert model with stats before wrapping with FSDP
        model, _ = convert_model_with_stats(
            model,
            layer_names,
            gather_forward=True,
            gather_backward=True,
            gather_interval=1,
        )

        # Wrap with FSDP
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock}
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=rank,
            use_orig_params=True,  # Important for accessing parameters
        )

        # Run training step
        metrics = run_training_step(model, device)

        # Only rank 0 verifies metrics (they should be synchronized)
        if rank == 0:
            self._verify_metrics(metrics, layer_names, expected_stats)

        # Cleanup
        cleanup_distributed()

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
