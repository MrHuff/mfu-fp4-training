#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Canonical-only, bit-exact parity gate for converted Llama-8B checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


os.environ["LBT_LIGHT_IMPORT"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_bits_training.analysis.llama_canonical_parity import (  # noqa: E402
    CANONICAL_PARITY_METHOD,
    CANONICAL_PARITY_POLICY,
    CANONICAL_PARITY_SCHEMA_VERSION,
    seal_canonical_receipt,
    validate_canonical_receipt,
)
from low_bits_training.analysis.llama_checkpoint_routes import (  # noqa: E402
    SUPPORTED_ROUTES,
)
from low_bits_training.analysis.llama_conversion_parity import (  # noqa: E402
    CANONICAL_FIXED_TOKEN_IDS,
    CANONICAL_LOGITS_SHAPE,
    CANONICAL_SEMANTIC_TOLERANCES,
    canonical_json_bytes,
    compare_semantic_logits,
    sha256_bytes,
    sha256_file,
    token_ids_sha256,
    write_atomic_receipt,
)
from scripts.evaluation.validate_llama8b_conversion_parity import (  # noqa: E402
    _canonical_hf_rmsnorm,
    _canonical_hf_rope,
    _device,
    build_native_reference,
    load_pinned_torchtitan,
    validate_inputs,
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _run_canonical_transformers(
    converted: Path,
    token_ids: torch.Tensor,
    device: torch.device,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    from transformers import AutoModelForCausalLM
    from transformers.models.llama import modeling_llama

    model = AutoModelForCausalLM.from_pretrained(
        converted,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    rmsnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, modeling_llama.LlamaRMSNorm)
    ]
    if len(rmsnorm_modules) != 65:
        raise RuntimeError("canonical HF model must contain 65 Llama RMSNorm modules")
    with (
        _canonical_hf_rope(freqs_cis),
        _canonical_hf_rmsnorm(),
        torch.inference_mode(),
        sdpa_kernel([SDPBackend.MATH], set_priority=True),
    ):
        logits = model(input_ids=token_ids, use_cache=False).logits.float().cpu()
    del model
    return logits


def _code_bundle(
    torchtitan_root: Path, project_root: Path
) -> tuple[dict[str, str], str]:
    files = {
        "canonical_parity_tool": Path(__file__).resolve(),
        "canonical_receipt_module": project_root
        / "low_bits_training/analysis/llama_canonical_parity.py",
        "base_parity_tool": project_root
        / "scripts/evaluation/validate_llama8b_conversion_parity.py",
        "parity_measurement_module": project_root
        / "low_bits_training/analysis/llama_conversion_parity.py",
        "checkpoint_routes": project_root
        / "low_bits_training/analysis/llama_checkpoint_routes.py",
        "checkpoint_streamer": project_root
        / "low_bits_training/analysis/stream_checkpoints.py",
        "canonical_eval_wrapper": project_root
        / "scripts/evaluation/run_canonical_lm_eval.py",
        "torchtitan_llama_model": torchtitan_root
        / "torchtitan/models/llama3/model/model.py",
        "torchtitan_llama_args": torchtitan_root
        / "torchtitan/models/llama3/model/args.py",
        "torchtitan_llama_adapter": torchtitan_root
        / "torchtitan/models/llama3/model/state_dict_adapter.py",
        "torchtitan_attention": torchtitan_root / "torchtitan/models/attention.py",
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    return hashes, sha256_bytes(canonical_json_bytes(hashes))


def _environment(
    device: torch.device, torchtitan_commit: str
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "safetensors": _package_version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torchtitan_commit": torchtitan_commit,
        "native_attention": "TorchTitan scaled_dot_product_attention causal math",
        "converted_attention": "Transformers SDPA causal math",
        "attention_backend": "SDPBackend.MATH",
        "compute_dtype": "torch.bfloat16",
        "canonical_semantic_rope": (
            "TorchTitan interleaved complex64 RoPE in converted HF model"
        ),
        "canonical_semantic_rmsnorm": (
            "TorchTitan torch.nn.functional.rms_norm in converted HF model"
        ),
        "stock_hf_computed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.receipt.exists():
        raise RuntimeError(f"refusing to overwrite parity receipt: {args.receipt}")
    manifest, _, weight_map, manifest_sha = validate_inputs(args)
    device = _device(args.device)
    torchtitan_root = args.torchtitan_root.resolve()
    torchtitan_types = load_pinned_torchtitan(torchtitan_root)
    torchtitan_commit = torchtitan_types[-1]
    project_root = Path(__file__).resolve().parents[2]
    code_files, code_bundle_sha = _code_bundle(torchtitan_root, project_root)

    token_ids = list(CANONICAL_FIXED_TOKEN_IDS)
    native_model, state_measurements = build_native_reference(
        args.checkpoint.resolve(),
        args.converted.resolve(),
        args.expected_route,
        weight_map,
        device,
        len(token_ids),
        torchtitan_types,
    )
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    torch.manual_seed(42)
    with torch.inference_mode():
        native_logits = native_model(tokens, attention_masks=None).float().cpu()
    native_freqs_cis = native_model.freqs_cis.detach()
    del native_model
    gc.collect()
    torch.cuda.empty_cache()

    canonical_logits = _run_canonical_transformers(
        args.converted.resolve(), tokens, device, native_freqs_cis
    )
    semantic = compare_semantic_logits(
        native_logits, canonical_logits, CANONICAL_SEMANTIC_TOLERANCES
    )
    measurements = {
        "passed": semantic["passed"],
        "canonical_semantic": semantic,
        **state_measurements,
    }
    print(
        "[LLAMA CANONICAL PARITY MEASUREMENTS] "
        + json.dumps(measurements, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    payload = {
        "schema_version": CANONICAL_PARITY_SCHEMA_VERSION,
        "method": CANONICAL_PARITY_METHOD,
        "policy": CANONICAL_PARITY_POLICY,
        "passed": measurements["passed"],
        "can_authorize_downstream_evaluation": measurements["passed"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_manifest_sha256": manifest_sha,
        "route": args.expected_route,
        "step": args.expected_step,
        "ntokens_seen": args.expected_ntokens_seen,
        "checkpoint_metadata_sha256": args.expected_metadata_sha256,
        "source_job_id": EXAMPLE,
        "source_uri_sha256": args.expected_source_uri_sha256,
        "fixed_token_ids": token_ids,
        "fixed_token_ids_sha256": token_ids_sha256(token_ids),
        "expected_logits_shape": list(CANONICAL_LOGITS_SHAPE),
        "tool_sha256": code_files["canonical_parity_tool"],
        "code_bundle_sha256": code_bundle_sha,
        "code_files_sha256": code_files,
        "environment": _environment(device, torchtitan_commit),
        "tolerances": CANONICAL_SEMANTIC_TOLERANCES.to_dict(),
        "measurements": measurements,
        "limitations": [],
    }
    receipt = seal_canonical_receipt(payload)
    if receipt["passed"]:
        validate_canonical_receipt(receipt)
    write_atomic_receipt(args.receipt, receipt)
    if not receipt["passed"]:
        raise RuntimeError(
            "canonical fixed-token parity failed; failure receipt was written"
        )
    print(
        "[LLAMA CANONICAL PARITY PASS] "
        f"route={args.expected_route} step={args.expected_step} "
        f"mismatches={semantic['mismatch_count']} "
        f"max_abs={semantic['max_abs_error']:.8g} receipt={args.receipt}",
        flush=True,
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--expected-route", choices=SUPPORTED_ROUTES, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-ntokens-seen", type=int, required=True)
    parser.add_argument("--expected-source-job-id", required=True)
    parser.add_argument("--expected-source-uri-sha256", required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.expected_step < 0
        or args.expected_ntokens_seen < 0
        or args.expected_shards <= 0
    ):
        raise SystemExit(
            "expected step and token count must be nonnegative and expected "
            "shards must be positive"
        )
    try:
        run(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"[LLAMA CANONICAL PARITY FAIL] {error}") from error


if __name__ == "__main__":
    main()
