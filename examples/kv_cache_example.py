#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
"""
Example script demonstrating how to use the KV caching converter with low_bits_training.

This script shows how to:
1. Load a model configuration from a TOML or JSON file
2. Apply the KV caching converter
3. Run inference with KV caching enabled and compare performance
"""

import argparse

import torch
import low_bits_training  # noqa

from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools.logging import init_logger, logger

from low_bits_training.config import ConfigManager
from low_bits_training.models import get_model_config
from low_bits_training.models.llama3 import Transformer
from low_bits_training.generate.kv_cache import generate_with_kv_cache


class KVCacheExample:
    def __init__(
        self,
        config_path: str,
        vocab_size: int = 32000,
        device: str = None,
    ):
        """
        Initialize the KV Cache Example with model configuration.

        Args:
            config_path: Path to the model configuration file (.toml or .json)
            vocab_size: Vocabulary size for the model
            device: Device to run the example on
        """
        init_logger()

        # Load configuration from toml/json file
        kv_cache_args = []
        kv_cache_args += [f"--job.config_file={config_path}"]
        # Add KV caching configuration
        kv_cache_args += ["--model.converters", "kv_cache"]

        # Parse additional arguments
        self.config = ConfigManager().parse_args(kv_cache_args)

        # Initialize device
        self.device = torch.device(
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self._initialize_model(vocab_size)

    def _initialize_model(self, vocab_size: int):
        """Initialize the model and apply KV caching converter."""
        # Create parallel dimensions (single GPU for this example)
        self.parallel_dims = ParallelDims(
            dp_shard=1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )

        # Get model config
        model_config = get_model_config(self.config.model.name, self.config.model.flavor)

        # Ensure vocab_size is set
        if not hasattr(model_config, "vocab_size") or model_config.vocab_size <= 0:
            logger.info(
                f"Setting vocab_size to {vocab_size} (was: {getattr(model_config, 'vocab_size', 'not set')})"
            )
            model_config.vocab_size = vocab_size
        # Create model
        with torch.device("meta"):
            self.model = Transformer(model_config)

        # Apply KV caching converter if enabled
        if "kv_cache" in self.config.model.converters:
            logger.info("Applying KV caching converter...")
            converters = build_model_converters(self.config, self.parallel_dims)
            converters.convert(self.model)
            logger.info("KV caching converter applied successfully")

        self.model.to_empty(device=self.device)
        with torch.no_grad():
            self.model.init_weights(buffer_device=None)

        self.model.eval()  # Set to evaluation mode
        self.model_config = model_config

    def run_comparison(
        self,
        sequence_length: int = 1024,
        batch_size: int = 1,
        generation_length: int = 10,
        use_compile: bool = False,
        profile: bool = False,
    ):
        """
        Run inference with and without KV caching and compare performance.

        Args:
            sequence_length: Length of the input sequence
            batch_size: Batch size for inference

        Returns:
            Dictionary containing performance metrics
        """
        logger.info(
            f"Running comparison with sequence_length={sequence_length}, batch_size={batch_size}"
        )
        if use_compile:
            self.model = torch.compile(self.model)
        # warmup
        seq_lens = torch.tensor(
            [sequence_length] * batch_size + [sequence_length - 10] * batch_size,
            device=self.device,
            dtype=torch.int64,
        )
        generate_with_kv_cache(
            self.model,
            torch.randint(
                0,
                self.model_config.vocab_size,
                (batch_size * 2, sequence_length),
                device=self.device,
            ),
            2,
            seq_lens=seq_lens.clone(),
            seed=1,
            use_kv_cache=False,
        )
        generate_with_kv_cache(
            self.model,
            torch.randint(
                0,
                self.model_config.vocab_size,
                (batch_size * 2, sequence_length),
                device=self.device,
            ),
            3,
            seq_lens=seq_lens.clone(),
            seed=1,
            use_kv_cache=True,
        )
        self.model.clear_kv_cache()

        # Create random input of batch size 2 * bs with seq_len sequence lenght and seq_Len - 10 sequence length
        # Need to handle padding of the shorter sequence
        input_ids = torch.cat(
            [
                torch.randint(
                    0,
                    self.model_config.vocab_size,
                    (batch_size, sequence_length),
                    device=self.device,
                ),
                torch.cat(
                    [
                        torch.randint(
                            0,
                            self.model_config.vocab_size,
                            (batch_size, sequence_length - 10),
                            device=self.device,
                        ),
                        torch.zeros(
                            (batch_size, 10),
                            device=self.device,
                            dtype=torch.int32,
                        ),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )
        seq_lens = torch.tensor(
            [sequence_length] * batch_size + [sequence_length - 10] * batch_size,
            device=self.device,
            dtype=torch.int64,
        )

        # Run inference without KV caching
        # add profiling
        if profile:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            )
        else:
            import contextlib

            prof = contextlib.nullcontext()

        with prof:
            # Add an event matching the log call]
            with torch.profiler.record_function(
                "Running inference without KV caching..."
            ):
                output_without_cache, time_without_cache = generate_with_kv_cache(
                    self.model,
                    input_ids,
                    generation_length,
                    seed=1,
                    seq_lens=seq_lens.clone(),
                    use_kv_cache=False,
                    eos_token_id=1,
                    get_timing=True,
                )

            # Run inference with KV caching
            with torch.profiler.record_function("Running inference with KV caching..."):
                output_with_cache, time_with_cache = generate_with_kv_cache(
                    self.model,
                    input_ids,
                    generation_length,
                    seed=1,
                    seq_lens=seq_lens.clone(),
                    use_kv_cache=True,
                    eos_token_id=1,
                    get_timing=True,
                )

        if profile:
            device_name = "_cpu" if self.device == "cpu" else ""
            is_compiled = "_compiled" if use_compile else ""
            prof.export_chrome_trace(f"kv_cache_trace{device_name}{is_compiled}.json")
        # Compare outputs
        # Calculate first token divergence for each sequence
        first_token_diff = (output_without_cache == output_with_cache).sum(
            dim=1
        ) - sequence_length
        outputs_match = (output_without_cache == output_with_cache).all()

        # Calculate speedup
        speedup = (
            time_without_cache[0] / time_with_cache[0]
            if time_with_cache[0] > 0
            else float("inf")
        )
        if time_with_cache[1] is not None:
            speedup_last_step = (
                time_without_cache[1] / time_with_cache[1]
                if time_with_cache[1] > 0
                else float("inf")
            )
        else:
            speedup_last_step = None

        # Return results
        return {
            "time_without_cache": time_without_cache,
            "time_with_cache": time_with_cache,
            "speedup": speedup,
            "speedup_last_step": speedup_last_step,
            "first_token_diff": first_token_diff,
            "outputs_match": outputs_match,
        }


def main():
    parser = argparse.ArgumentParser(description="KV caching example")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model configuration file (.toml or .json)",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=10,
        help="Generation length for inference",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1024,
        help="Sequence length for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--attention_module_names",
        type=str,
        default="attention",
        help="Comma-separated list of attention module names to apply KV caching to",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for the model",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the example on",
    )
    args = parser.parse_args()

    # Create KV cache example
    example = KVCacheExample(
        config_path=args.config_path,
        vocab_size=args.vocab_size,
        device=args.device,
    )

    # Run comparison
    results = example.run_comparison(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        generation_length=args.generation_length,
        use_compile=args.compile,
        profile=args.profile,
    )

    # Print results
    print("\nResults:")
    print(f"Time without KV caching: {results['time_without_cache']} seconds")

    if results["time_with_cache"] is not None:
        print(f"Time with KV caching: {results['time_with_cache']} seconds")
        print(f"Speedup: {results['speedup']:.2f}x")
        print(f"First token difference: {results['first_token_diff']}")
        print(f"Speedup last step: {results['speedup_last_step']:.2f}x")
        if results["outputs_match"]:
            print("Outputs match! KV caching is working correctly.")
        else:
            print(
                "Warning: Outputs don't match exactly. There might be an issue with KV caching."
            )
    else:
        print("KV caching was not enabled. No comparison available.")


if __name__ == "__main__":
    main()
