#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license retained at
# torchtitan_submodule/LICENSE in this source distribution.
"""
This script is copied from torchtian_submodule/scripts/generate/test_generate.py

It was impossible to reliably import the script from the submodule so it is copied
and modified here to work with KV caching.
"""

import argparse
import json
import os
import time

from typing import Optional, Dict, Any, List, TypedDict


import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

from .kv_cache import generate_with_kv_cache
from ..device_patch import get_device_info, get_device_module
from ..checkpoints import load_config, dcp_load_helper
from torchtitan.config.job_config import Debug as DebugConfig

from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import utils as tools_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import build_model_converters

from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.components.tokenizer import Tokenizer


def apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


class GeneratorSettings(TypedDict):
    """
    temperature: Sampling temperature
    max_new_tokens: Maximum number of tokens to generate
    top_k: If specified, only sample from the top k most likely tokens
    seed: Random seed for generation
    """

    temperature: float
    max_new_tokens: int
    top_k: Optional[int]
    seed: Optional[int]
    extra_eos_token_ids: list[str]


DEFAULT_GENERATOR_SETTINGS = GeneratorSettings(
    temperature=0.5,
    max_new_tokens=32,
    top_k=None,
    seed=None,
    extra_eos_token_ids=[],
)


class Generator:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        seed: Optional[int] = None,
        deterministic: bool = False,
        add_kv_cache: bool = True,
        device: str | None = None,
        **_,  # for forward compatibility with vllm and vllm-engine
    ):
        """
        Initialize the Generator with model configuration and checkpoint.

        Args:
            config_path: Path to the model configuration file (.toml or .json)
            checkpoint_path: Path to the model checkpoint
            seed: Random seed for reproducibility
            deterministic: Whether to use deterministic algorithms
        """
        init_logger()

        # Load configuration from toml/json file
        self.config = load_config(config_path)

        if self.config.debug.seed is not None:
            logger.warning(
                "For generation, the `seed` value in the job config is ignored.",
                "In order to specify a seed, use the `seed` argument to initialise ",
                "the `Generator` class or when calling the `generate` method.",
            )

        if self.config.debug.deterministic:
            logger.warning(
                "For generation, the `deterministic` value in the job config is ignored.",
                "In order to specify deterministic behavior, use the `deterministic` ",
                "argument to initialise the `Generator` class or when calling the `generate` method.",
            )

        # Initialize distributed environment
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # TODO: simplify & unify this logic.
        if device is None:
            self.device_type = get_device_info()[0]
        else:
            self.device_type = device

        if self.device_type == "cpu":
            self.device = torch.device("cpu")
        elif ":" not in self.device_type:
            self.device = torch.device(f"{self.device_type}:{self.local_rank}")
        else:
            self.device = torch.device(self.device_type)
            self.device_type, _ = self.device_type.split(":")

        get_device_module().set_device(self.device)

        # TODO: refactor with a utils that handles CPU device monitoring
        self.device_memory_monitor = None
        if self.device_type != "cpu":
            self.device_memory_monitor = build_device_memory_monitor()

        logger.info(
            f"World Size: {self.world_size}, Local Rank: {self.local_rank} on {self.device}"
        )
        self.has_kv_cache = add_kv_cache
        # Initialize model
        self._initialize_model(checkpoint_path, seed, deterministic, add_kv_cache)

    def _initialize_model(
        self,
        checkpoint_path: str,
        seed: Optional[int],
        deterministic: bool,
        add_kv_cache: bool,
    ):
        """Initialize the model, tokenizer, and load checkpoint."""
        # TODO: replace with the function that is in `checkpoints`
        train_spec = get_train_spec(self.config.model.name)

        # Tokenizer setup
        self.tokenizer: Tokenizer = train_spec.build_tokenizer_fn(self.config)

        model_config = train_spec.model_args[self.config.model.flavor]
        model_config.norm_type = self.config.model.norm_type
        model_config.max_seq_len = self.config.training.seq_len
        model_config.vocab_size = self.tokenizer.get_vocab_size()

        # Model initialization
        model_cls = train_spec.model_cls
        init_device = "meta" if self.world_size > 1 else self.device
        with torch.device(init_device):
            logger.info(f"Init model on init_device: {init_device}")
            self.model = model_cls(model_config)

        self.world_mesh = None
        # Init distributed env
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )
        distinct_seed_mesh_dims = []
        if self.world_size > 1:
            dist_utils.init_distributed(self.config)
            # Build world mesh for parallelism
            self.world_mesh = parallel_dims.build_mesh(device_type=self.device_type)
            distinct_seed_mesh_dims = list(self.world_mesh.mesh_dim_names)
            # apply_tp (with Sequence Parallel) on unevenly sharded
            # sequences would require https://github.com/pytorch/torchtitan/pull/686
            apply_tp_minus_sp(self.model, self.world_mesh["tp"])

        if add_kv_cache:
            if isinstance(self.config.model.converters, str):
                self.config.model.converters = [self.config.model.converters]
            elif self.config.model.converters is None:
                self.config.model.converters = []
            self.config.model.converters.append("kv_cache")

        if self.config.model.converters:
            logger.info("Applying model converters...")
        converters = build_model_converters(self.config, parallel_dims)
        converters.convert(self.model)
        logger.info(
            f"Model converters: {self.config.model.converters} applied successfully"
        )

        # TODO: Ideally this should be pre-defined in the config.
        debug_config = DebugConfig(seed=seed, deterministic=deterministic)
        dist_utils.set_determinism(
            self.world_mesh, self.device, debug_config, distinct_seed_mesh_dims
        )

        # Materialize model
        self.model.to_empty(device=self.device_type)
        self.model.eval()

        state_dict = {"model": self.model.state_dict()}

        # Checkpoint Loading
        begin = time.monotonic()
        logger.info(f"Loading chkpt at: {checkpoint_path}")
        dcp_load_helper(state_dict, checkpoint_id=checkpoint_path)
        logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")

        if self.device_memory_monitor is not None:
            device_mem_stats = self.device_memory_monitor.get_peak_stats()
            logger.info(
                f"{self.device_type.upper()} memory usage for model: "
                f"{device_mem_stats.max_reserved_gib:.2f}GiB"
                f"({device_mem_stats.max_reserved_pct:.2f}%)"
            )

    def generate(
        self,
        prompt: str | List[str],
        batch_size: int = 1,
        output_json: bool = False,
        progress_bar: bool = True,
        settings: Optional[GeneratorSettings] = None,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text based on the input prompt.

        Args:
            prompt: Input text prompt
            batch_size: Number of samples to generate in parallel
            output_json: Whether to return the output as a JSON object
            progress_bar: Show a progress bar during generation
            settings: A settings dictionary - defaults to DEFAULT_GENERATOR_SETTINGS

        Returns:
            Dictionary containing generated responses and metadata
        """
        if settings is None:
            settings = DEFAULT_GENERATOR_SETTINGS.copy()
        settings.update(**generation_kwargs)
        if len(prompt) == 0:
            logger.warning(
                "The input prompt is empty, model will respond from a empty sequence."
            )

        # Reset memory monitor
        if self.device_memory_monitor is not None:
            self.device_memory_monitor.reset_peak_stats()
        # Tokenize prompts and repeat batch_size times
        # allowed_special="all" is needed to correctly encode tokens for instruct models.
        encode_settings = dict(add_bos=True, add_eos=False, allowed_special="all")
        if isinstance(prompt, str):
            logger.debug(f"Prompt: {prompt}")
            input_ids = (
                torch.tensor(
                    self.tokenizer.encode(prompt, **encode_settings), dtype=torch.long
                )
                .view(1, -1)
                .repeat(batch_size, 1)
            )
            initial_seq_lens = torch.tensor(
                [input_ids.size(1) for _ in range(batch_size)], dtype=torch.long
            )
        elif isinstance(prompt, list):
            assert len(prompt) == batch_size, "Prompt list must be of length batch_size"
            tokenized_prompts = [
                self.tokenizer.encode(p, **encode_settings) for p in prompt
            ]
            initial_seq_lens = torch.tensor(
                [len(t) for t in tokenized_prompts], dtype=torch.long
            )
            logger.debug(
                f"Prompt: has {len(prompt)} prompts of sizes: {initial_seq_lens}"
            )
            max_length = max(len(t) for t in tokenized_prompts)
            input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
            for i, t in enumerate(tokenized_prompts):
                input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        logger.info(f"Processing batch of size {batch_size}")
        if (
            self.model.model_args.max_seq_len
            < input_ids.size(1) + settings["max_new_tokens"]
        ):
            logger.error(
                f"Max sequence length is {self.model.model_args.max_seq_len}, but the input prompt"
                f" is {input_ids.size(1)} tokens long and {settings['max_new_tokens']} tokens are requested to "
                f"be generated. This may cause problems."
            )
            return {
                "metadata": {},
                "responses": [
                    {
                        "response_idx": 0,
                        "input_text": prompt,
                        "output_text": "Error: Max sequence length is too short.",
                    }
                ],
            }
        # Run generation
        t0 = time.monotonic()
        eos_token_ids = [self.tokenizer.eos_id]
        for extra_eos_str in settings["extra_eos_token_ids"]:
            extra_eos_token = self.tokenizer.encode(
                extra_eos_str, add_bos=False, add_eos=False, allowed_special="all"
            )
            assert (
                len(extra_eos_token) == 1
            ), f"Incorrect generation config settings {extra_eos_str}, should encode to a single token, encoded to {extra_eos_token}"
            eos_token_ids.append(extra_eos_token[0])
        responses = generate_with_kv_cache(
            # TODO: refactor generate with kv_cache so that it receives the GenerationSettings
            self.model,
            input_ids.clone().to(self.device),
            seq_lens=initial_seq_lens.clone().to(self.device),
            max_new_tokens=settings["max_new_tokens"],
            eos_token_id=eos_token_ids,
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            seed=settings["seed"],
            progress_bar=progress_bar,
            get_timing=False,
            use_kv_cache=self.has_kv_cache,
        )
        t1 = time.monotonic()
        elapsed_sec = t1 - t0

        # Post process
        B, T = responses.size()  # B: batch_size, T: total seq length
        input_n_tokens = input_ids.size(1)
        generated_n_tokens = T - input_n_tokens  # == max_new_tokens

        output_data = {
            "metadata": {},
            "responses": [],
        }

        if self.local_rank != 0:
            return output_data

        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")

        color = tools_utils.Color
        r, b = color.red, color.blue
        total_generated_tokens = 0
        total_input_tokens = 0
        for i, tokens in enumerate(responses):
            input_n_tokens = initial_seq_lens[i]
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            # Trim trailing zeros
            truncate_idx = len(out_tok)
            for j in range(len(out_tok) - 1, -1, -1):
                if out_tok[j] != 0:
                    truncate_idx = j + 1
                    break
            out_tok = out_tok[:truncate_idx]
            total_input_tokens += len(inp_tok)
            total_generated_tokens += len(out_tok)
            input_text = self.tokenizer.decode(inp_tok, skip_special_tokens=False)
            output_text = self.tokenizer.decode(out_tok, skip_special_tokens=False)

            _data = {
                "response_idx": i,
                "input_text": input_text,
                "output_text": output_text,
            }
            output_data["responses"].append(_data)

            logger.debug(f"{r}\n{input_text}{b}{output_text}\n{color.reset}")

        output_data["metadata"] = {
            "generated_n_tokens": generated_n_tokens,
            "input_n_tokens": input_n_tokens.tolist(),
            "generation_time_sec": elapsed_sec,
            "tokens_per_sec": total_generated_tokens / elapsed_sec,
            "batching_efficiency": total_generated_tokens / (B * T),
            "batch_size": B,
            "generator_settings": settings,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "world_size": self.world_size,
            "torch_version": torch.__version__,
        }
        logger.info(
            "[%s] Prefill %s tokens, Generated %s tokens at %s tok/s (batching efficiency: %s)",
            self.device,
            input_n_tokens,
            total_generated_tokens,
            total_generated_tokens / elapsed_sec,
            output_data["metadata"]["batching_efficiency"],
        )
        if self.device_memory_monitor is not None:
            device_mem_stats = self.device_memory_monitor.get_peak_stats()
            output_data["metadata"].update(
                {
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
            )

        if output_json:
            logger.info(json.dumps(output_data, indent=4))

        return output_data


@record
def test_generate(
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    output_json: bool = False,
):
    """Legacy function for backward compatibility"""

    generator = Generator(config_path, checkpoint_path, seed, deterministic)
    return generator.generate(
        prompt=prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        top_k=top_k,
        seed=seed,
        output_json=output_json,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Test generation")

    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    add_generation_args(parser)
    return parser


def add_generation_args(parser: argparse.ArgumentParser):
    parser.add_argument("--prompt", required=True, type=str, help="Input prompt")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default is 1.0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate. Default is 32",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples to run in batch"
    )
    parser.add_argument(
        "--top_k", type=int, help="Prune to select from top_k probabilities. Optional"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms wherever possible, may be slower",
    )

    parser.add_argument(
        "--out",
        action="store_true",
        help="If specified, prints the report to stdout. Defaults to no output.",
    )

    return parser


def main(args: argparse.Namespace):
    test_generate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        deterministic=args.deterministic,
        output_json=args.out,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(get_parser().parse_args())
