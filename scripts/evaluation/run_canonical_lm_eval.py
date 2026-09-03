#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Run lm-eval inside the pinned fully canonical Llama inference context."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import os
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


os.environ["LBT_LIGHT_IMPORT"] = "1"
sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_bits_training.analysis.llama_conversion_parity import (  # noqa: E402
    PINNED_TORCHTITAN_COMMIT,
    PINNED_TRANSFORMERS_VERSION,
)
from low_bits_training.analysis.torchtitan_gitlink import (  # noqa: E402
    validate_torchtitan_gitlink_marker,
)
from scripts.evaluation.validate_llama8b_conversion_parity import (  # noqa: E402
    _canonical_hf_rmsnorm,
    _canonical_hf_rope,
)


def _pinned_frequencies(torchtitan_root: Path) -> torch.Tensor:
    root = torchtitan_root.resolve()
    if validate_torchtitan_gitlink_marker(root) != PINNED_TORCHTITAN_COMMIT:
        raise RuntimeError("canonical evaluator TorchTitan commit drift")
    sys.path.insert(0, str(root))
    from torchtitan.models.llama3.model.args import RoPEScalingArgs
    from torchtitan.models.llama3.model.model import precompute_freqs_cis

    resolved = Path(sys.modules[precompute_freqs_cis.__module__].__file__).resolve()
    if root not in resolved.parents:
        raise RuntimeError("canonical evaluator imported TorchTitan outside pin")
    scaling = RoPEScalingArgs(
        scaling_factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )
    if scaling != RoPEScalingArgs():
        raise RuntimeError("pinned Llama-3.1 8B RoPE defaults drift")
    frequencies = precompute_freqs_cis(
        dim=128,
        end=8192,
        theta=500000.0,
        scaling_args=scaling,
    )
    no_scaling = precompute_freqs_cis(
        dim=128,
        end=8192,
        theta=500000.0,
        scaling_args=None,
    )
    if (
        frequencies.shape != (8192, 64)
        or frequencies.dtype is not torch.complex64
        or torch.equal(frequencies, no_scaling)
    ):
        raise RuntimeError("scaled canonical evaluator frequency contract drift")
    return frequencies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("lm_eval_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    forwarded = list(args.lm_eval_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    if not forwarded or forwarded[0] != "run":
        raise SystemExit("canonical evaluator requires an lm-eval `run` command")
    if version("transformers") != PINNED_TRANSFORMERS_VERSION:
        raise SystemExit("canonical evaluator Transformers version drift")
    if version("lm-eval") != "0.4.12":
        raise SystemExit("canonical evaluator lm-eval version drift")
    if not torch.cuda.is_available():
        raise SystemExit("canonical evaluator requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    frequencies = _pinned_frequencies(args.torchtitan_root).to(device)

    from lm_eval.__main__ import cli_evaluate

    original_argv = sys.argv
    sys.argv = ["lm_eval", *forwarded]
    print(
        "[CANONICAL LM-EVAL START] semantics=torchtitan-llama31-8b-"
        "scaled-rope-rmsnorm rope_theta=500000 rope_scaling=8,1,4,8192 "
        "attention=math dtype=bfloat16 stock_hf_computed=0",
        flush=True,
    )
    try:
        with (
            _canonical_hf_rope(frequencies),
            _canonical_hf_rmsnorm(),
            sdpa_kernel([SDPBackend.MATH], set_priority=True),
        ):
            cli_evaluate()
    finally:
        sys.argv = original_argv
    print(
        "[CANONICAL LM-EVAL COMPLETE] semantics=torchtitan-llama31-8b-"
        "scaled-rope-rmsnorm stock_hf_computed=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
