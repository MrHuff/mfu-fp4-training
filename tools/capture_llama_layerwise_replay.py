#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Build sealed calibration batches and capture canonical Llama replay tensors.

Example::

    python tools/capture_llama_layerwise_replay.py build-batch \
      --arrow /opt/mfu/EXTERNAL_PATH --arrow /opt/mfu/EXTERNAL_PATH \
      --tokenizer /opt/mfu/EXTERNAL_PATH --output-dir /opt/mfu/EXTERNAL_PATH \
      --seq-len 8192 --batch-size 1 --num-batches 1

    python tools/capture_llama_layerwise_replay.py capture \
      --converted /opt/mfu/EXTERNAL_PATH \
      --batch /opt/mfu/EXTERNAL_PATH \
      --batch-manifest /opt/mfu/EXTERNAL_PATH \
      --output-dir /opt/mfu/EXTERNAL_PATH --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


os.environ.setdefault("LBT_LIGHT_IMPORT", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_bits_training.analysis.layerwise_replay_capture import (  # noqa: E402
    CALIBRATION_METHOD,
    CAPTURE_METHOD,
    SCHEMA_VERSION,
    LayerCaptureWriter,
    build_token_batches,
    calibration_tensor_ledger,
    canonical_hf_rmsnorm,
    canonical_hf_rope,
    canonical_json_bytes,
    chunked_causal_ce_hidden_gradient,
    directory_file_ledger,
    install_hf_layer_capture_hooks,
    load_json_object,
    seal_receipt,
    sha256_file,
    staged_output_directory,
    tensor_sha256,
    validate_converted_model,
    validate_receipt_seal,
)


PINNED_TORCHTITAN_COMMIT = "20b3de7585696c327bd5aa9f9627f0300abdbf9d"
PINNED_TRANSFORMERS_VERSION = "4.48.2"


def _code_ledger(torchtitan_root: Path | None = None) -> dict[str, str]:
    files = {
        "capture_tool": Path(__file__).resolve(),
        "capture_helpers": PROJECT_ROOT
        / "low_bits_training/analysis/layerwise_replay_capture.py",
    }
    if torchtitan_root is not None:
        files["torchtitan_llama_model"] = (
            torchtitan_root.resolve()
            / "torchtitan/models/llama3/model/model.py"
        )
    return {name: sha256_file(path) for name, path in sorted(files.items())}


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tracked_dirty": bool(
            _git(root, "status", "--porcelain", "--untracked-files=no")
        ),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite JSON evidence: {path}")
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _validate_optional_expected(
    actual: str, expected: str | None, label: str
) -> None:
    if expected is not None and actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected}, found {actual}")


def build_calibration(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    arrows = sorted((path.resolve() for path in args.arrow), key=str)
    if len({str(path) for path in arrows}) != 2:
        raise RuntimeError("build-batch requires two distinct Arrow files")
    source_files = []
    for index, path in enumerate(arrows):
        if not path.is_file():
            raise RuntimeError(f"Arrow source does not exist: {path}")
        digest = sha256_file(path)
        expected = (
            None
            if args.expected_arrow_sha256 is None
            else args.expected_arrow_sha256[index]
        )
        _validate_optional_expected(digest, expected, f"Arrow source {path.name}")
        source_files.append(
            {
                "index": index,
                "filename": path.name,
                "resolved_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    tokenizer_root = args.tokenizer.resolve()
    tokenizer_files = directory_file_ledger(tokenizer_root)
    tokenizer_bundle_sha = sha256(canonical_json_bytes(tokenizer_files)).hexdigest()
    _validate_optional_expected(
        tokenizer_bundle_sha,
        args.expected_tokenizer_bundle_sha256,
        "tokenizer bundle",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    payload, documents = build_token_batches(
        arrows,
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )
    tensor_ledger = calibration_tensor_ledger(payload)
    code_files = _code_ledger()

    with staged_output_directory(args.output_dir) as staging:
        batch_path = staging / "calibration_batch.pt"
        torch.save(payload, batch_path)
        document_ledger_sha = sha256(canonical_json_bytes(documents)).hexdigest()
        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "method": CALIBRATION_METHOD,
            "source_files": source_files,
            "tokenizer": {
                "resolved_path": str(tokenizer_root),
                "class": type(tokenizer).__name__,
                "vocab_size": len(tokenizer),
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "files": tokenizer_files,
                "bundle_sha256": tokenizer_bundle_sha,
                "transformers_version": version("transformers"),
                "tokenizers_version": version("tokenizers"),
            },
            "packing": {
                "source_order": [item["filename"] for item in source_files],
                "round_robin": True,
                "token_quantum": max(
                    1,
                    min(
                        256,
                        (
                            args.num_batches
                            * args.batch_size
                            * args.seq_len
                        )
                        // 2,
                    ),
                ),
                "bos_once_at_stream_start": True,
                "eos_after_each_document": True,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
            },
            "documents": documents,
            "document_ledger_sha256": document_ledger_sha,
            "tensors": tensor_ledger,
            "batch_file": {
                "filename": batch_path.name,
                "bytes": batch_path.stat().st_size,
                "sha256": sha256_file(batch_path),
            },
            "code_files_sha256": code_files,
            "code_bundle_sha256": sha256(canonical_json_bytes(code_files)).hexdigest(),
            "project_git": _git_identity(PROJECT_ROOT),
        }
        receipt = seal_receipt(manifest_payload)
        _write_json(staging / "calibration_manifest.json", receipt)
    return receipt


def _load_calibration(
    batch_path: Path, manifest_path: Path, batch_index: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest = load_json_object(manifest_path)
    validate_receipt_seal(manifest)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("method") != CALIBRATION_METHOD
    ):
        raise RuntimeError("calibration manifest schema/method mismatch")
    batch_record = manifest.get("batch_file")
    if not isinstance(batch_record, dict):
        raise RuntimeError("calibration manifest has no batch-file binding")
    if batch_record.get("filename") != batch_path.name:
        raise RuntimeError("calibration batch filename mismatch")
    if sha256_file(batch_path) != batch_record.get("sha256"):
        raise RuntimeError("calibration batch file SHA-256 mismatch")
    payload = torch.load(batch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("calibration batch payload is not a mapping")
    actual_ledger = calibration_tensor_ledger(payload)
    if actual_ledger != manifest.get("tensors"):
        raise RuntimeError("calibration tensor ledger mismatch")
    num_batches = payload["input_ids"].shape[0]
    if batch_index < 0 or batch_index >= num_batches:
        raise RuntimeError(
            f"batch index {batch_index} is outside [0, {num_batches})"
        )
    selected = {
        "input_ids": payload["input_ids"][batch_index].contiguous(),
        "target_ids": payload["target_ids"][batch_index].contiguous(),
    }
    identity = {
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "calibration_receipt_sha256": manifest["receipt_sha256"],
        "calibration_batch_file_sha256": batch_record["sha256"],
        "batch_index": batch_index,
        "input_ids_sha256": tensor_sha256(selected["input_ids"]),
        "target_ids_sha256": tensor_sha256(selected["target_ids"]),
        "source_files": manifest["source_files"],
        "tokenizer_bundle_sha256": manifest["tokenizer"]["bundle_sha256"],
        "packing": manifest["packing"],
    }
    return selected, identity


def _load_pinned_frequencies(
    torchtitan_root: Path,
    *,
    head_dim: int,
    seq_len: int,
    theta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    root = torchtitan_root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if commit != PINNED_TORCHTITAN_COMMIT:
        raise RuntimeError(
            "TorchTitan commit drift: "
            f"expected {PINNED_TORCHTITAN_COMMIT}, found {commit}"
        )
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("TorchTitan tracked worktree is dirty")
    sys.path.insert(0, str(root))
    from torchtitan.models.llama3.model.model import precompute_freqs_cis

    origin = Path(sys.modules[precompute_freqs_cis.__module__].__file__).resolve()
    if root not in origin.parents:
        raise RuntimeError("TorchTitan frequency function imported outside pin")
    frequencies = precompute_freqs_cis(
        dim=head_dim,
        end=seq_len,
        theta=theta,
        scaling_args=None,
    )
    if frequencies.shape != (seq_len, head_dim // 2):
        raise RuntimeError("TorchTitan frequency shape mismatch")
    if frequencies.dtype is not torch.complex64:
        raise RuntimeError("TorchTitan frequencies are not complex64")
    return frequencies, {
        "commit": commit,
        "model_source": str(origin),
        "model_source_sha256": sha256_file(origin),
        "rope_scaling_args": None,
        "rope_theta": theta,
        "head_dim": head_dim,
        "seq_len": seq_len,
    }


def capture_reference(args: argparse.Namespace) -> dict[str, Any]:
    if version("transformers") != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Transformers must be {PINNED_TRANSFORMERS_VERSION}, "
            f"found {version('transformers')}"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capture requires a CUDA device")
    torch.cuda.set_device(device)

    batch, calibration_identity = _load_calibration(
        args.batch.resolve(),
        args.batch_manifest.resolve(),
        args.batch_index,
    )
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    batch_size, seq_len = input_ids.shape
    if seq_len > 8192:
        raise RuntimeError("canonical sealed RoPE capture is limited to 8192 tokens")

    conversion_identity = validate_converted_model(
        args.converted,
        verify_weight_files=not args.skip_weight_file_hashes,
    )
    _validate_optional_expected(
        conversion_identity["conversion_manifest_sha256"],
        args.expected_conversion_manifest_sha256,
        "conversion manifest",
    )
    if conversion_identity["transformers_version"] != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError("conversion manifest Transformers version drift")

    config = load_json_object(args.converted.resolve() / "config.json")
    expected_geometry = {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    }
    for field, wanted in expected_geometry.items():
        if config.get(field) != wanted:
            raise RuntimeError(
                f"converted Llama geometry drift for {field}: {config.get(field)}"
            )
    if input_ids.min().item() < 0 or input_ids.max().item() >= config["vocab_size"]:
        raise RuntimeError("calibration input token lies outside converted vocabulary")
    if target_ids.min().item() < 0 or target_ids.max().item() >= config["vocab_size"]:
        raise RuntimeError("calibration target token lies outside converted vocabulary")

    frequencies, torchtitan_identity = _load_pinned_frequencies(
        args.torchtitan_root,
        head_dim=config["head_dim"],
        seq_len=seq_len,
        theta=float(config["rope_theta"]),
    )
    code_files = _code_ledger(args.torchtitan_root)
    bindings = {
        "conversion": conversion_identity,
        "calibration": calibration_identity,
        "code_files_sha256": code_files,
        "code_bundle_sha256": sha256(canonical_json_bytes(code_files)).hexdigest(),
        "project_git": _git_identity(PROJECT_ROOT),
        "torchtitan": torchtitan_identity,
        "semantics": {
            "dtype": "torch.bfloat16",
            "rope": "TorchTitan interleaved complex64",
            "rmsnorm": "torch.nn.functional.rms_norm",
            "attention": "SDPBackend.MATH",
            "loss": "regular cross entropy",
            "parameters_require_grad": False,
        },
    }

    from transformers import AutoModelForCausalLM
    import transformers.models.llama.modeling_llama as modeling_llama

    model = AutoModelForCausalLM.from_pretrained(
        args.converted.resolve(),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    for parameter in model.parameters():
        if parameter.dtype is not torch.bfloat16 or parameter.device != device:
            raise RuntimeError("converted model parameter materialization drift")
        parameter.requires_grad_(False)
    rmsnorm_count = sum(
        isinstance(module, modeling_llama.LlamaRMSNorm)
        for module in model.modules()
    )
    if rmsnorm_count != 65 or len(model.model.layers) != 32:
        raise RuntimeError("converted model module inventory drift")

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    frequencies = frequencies.to(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with staged_output_directory(args.output_dir) as staging:
        writer = LayerCaptureWriter(
            staging,
            batch_size=batch_size,
            seq_len=seq_len,
            sample_rows=args.sample_rows,
            sample_seed=args.sample_seed,
            num_layers=32,
            bindings=bindings,
        )
        handles = install_hf_layer_capture_hooks(model, writer)
        try:
            inputs_embeds = (
                model.model.embed_tokens(input_ids).detach().requires_grad_(True)
            )
            with (
                canonical_hf_rope(
                    frequencies,
                    writer=writer,
                    expected_layers=32,
                ),
                canonical_hf_rmsnorm(),
                sdpa_kernel([SDPBackend.MATH], set_priority=True),
            ):
                body_output = model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden_states = body_output.last_hidden_state
                loss, hidden_gradient = chunked_causal_ce_hidden_gradient(
                    hidden_states,
                    model.lm_head,
                    target_ids,
                    chunk_tokens=args.ce_chunk_tokens,
                )
                hidden_states.backward(hidden_gradient)
            torch.cuda.synchronize(device)
        finally:
            for handle in handles:
                handle.remove()
        receipt = writer.finalize(
            loss=loss,
            extra={
                "seed": args.seed,
                "sample_seed": args.sample_seed,
                "ce_chunk_tokens": args.ce_chunk_tokens,
                "device_name": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "rmsnorm_modules": rmsnorm_count,
            },
        )
    del model, hidden_states, hidden_gradient, inputs_embeds, body_output
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("build-batch", help="build sealed token batches")
    batch.add_argument("--arrow", action="append", type=Path, required=True)
    batch.add_argument("--tokenizer", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--seq-len", type=int, default=1024)
    batch.add_argument("--batch-size", type=int, default=1)
    batch.add_argument("--num-batches", type=int, default=1)
    batch.add_argument("--expected-arrow-sha256", nargs=2)
    batch.add_argument("--expected-tokenizer-bundle-sha256")

    capture = subparsers.add_parser("capture", help="capture canonical BF16 replay")
    capture.add_argument("--converted", type=Path, required=True)
    capture.add_argument("--batch", type=Path, required=True)
    capture.add_argument("--batch-manifest", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument(
        "--torchtitan-root",
        type=Path,
        default=PROJECT_ROOT / "torchtitan_submodule",
    )
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--batch-index", type=int, default=0)
    capture.add_argument("--sample-rows", type=int, default=256)
    capture.add_argument("--sample-seed", type=int, default=42)
    capture.add_argument("--seed", type=int, default=42)
    capture.add_argument("--ce-chunk-tokens", type=int, default=128)
    capture.add_argument("--expected-conversion-manifest-sha256")
    capture.add_argument(
        "--skip-weight-file-hashes",
        action="store_true",
        help="record but skip the expensive per-shard HF verification",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-batch":
        if len(args.arrow) != 2:
            raise SystemExit("build-batch requires exactly two --arrow arguments")
        receipt = build_calibration(args)
        summary = {
            "command": args.command,
            "output_dir": str(args.output_dir.resolve()),
            "receipt_sha256": receipt["receipt_sha256"],
            "input_ids_sha256": receipt["tensors"]["input_ids"]["sha256"],
        }
    else:
        receipt = capture_reference(args)
        summary = {
            "command": args.command,
            "output_dir": str(args.output_dir.resolve()),
            "receipt_sha256": receipt["receipt_sha256"],
            "loss": receipt["loss"],
            "file_ledger_sha256": receipt["file_ledger_sha256"],
        }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
