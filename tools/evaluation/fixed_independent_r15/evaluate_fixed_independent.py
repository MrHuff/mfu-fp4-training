#!/usr/bin/env python3
"""Evaluate a local checkpoint on the fixed-independent r15 token panel.

Inputs are local files named explicitly by the caller.  The result records
content hashes and scientific settings, never storage locations, credentials,
or scheduler identities.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import torch
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM


FIXED_VALIDATION_SCHEMA = "mfu_llama_fixed_independent_validation_manifest_v1"
FIXED_STREAM_ID = "llama31-fixed-independent-82dclm18olmo-r15-20260902"
RESULT_SCHEMA = "mfu_llama_fixed_independent_prefix_fullparity_result_v1"
ALLOWED_TASKS = {
    8: ("bf16", "bf16-unfused-v1", 2000, 32),
    31: ("bf16", "bf16-unfused-v1", 29000, 32),
    6: ("localcta_h16", "localcta-v4-fused-v1", 38000, 64),
}
PREFIX_TARGETS = (512, 2048, 8192)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def validate_task(task: dict, task_index: int) -> None:
    required = {
        "index",
        "route_key",
        "checkpoint_route",
        "step",
        "shard_count",
        "expected_ntokens_seen",
        "checkpoint_metadata_sha256",
    }
    if set(task) != required or task.get("index") != task_index:
        raise RuntimeError("task contract fields drift")
    if task_index not in ALLOWED_TASKS:
        raise RuntimeError("task is outside the fixed-independent r15 panel")
    route, checkpoint_route, step, shards = ALLOWED_TASKS[task_index]
    if (
        task["route_key"],
        task["checkpoint_route"],
        task["step"],
        task["shard_count"],
    ) != (route, checkpoint_route, step, shards):
        raise RuntimeError("task scientific binding drift")
    digest = task["checkpoint_metadata_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("checkpoint metadata digest is malformed")


def validate_fixed_manifest(manifest: dict) -> None:
    unsigned = dict(manifest)
    observed_seal = unsigned.pop("manifest_sha256", None)
    if observed_seal != sha256_bytes(canonical_json_bytes(unsigned)):
        raise RuntimeError("fixed-independent manifest seal drift")
    if (
        manifest.get("schema") != FIXED_VALIDATION_SCHEMA
        or manifest.get("stream_id") != FIXED_STREAM_ID
        or manifest.get("claim") != "fixed-independent-not-proven-held-out"
    ):
        raise RuntimeError("fixed-independent validation identity drift")
    if manifest.get("geometry") != {
        "world_size": 32,
        "sequences_per_rank": 24,
        "sequences": 768,
        "stored_tokens_per_sequence": 8193,
        "scored_tokens_per_sequence": 8192,
        "stored_tokens": 6_292_224,
        "validation_tokens": 6_291_456,
        "padding": False,
    }:
        raise RuntimeError("fixed-independent validation geometry drift")
    if manifest.get("stratification", {}).get("selected_tokens") != {
        "dclm": 5_159_624,
        "olmo-no-dclm": 1_132_600,
    }:
        raise RuntimeError("fixed-independent validation stratum drift")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 32:
        raise RuntimeError("fixed-independent shard count drift")
    for rank, record in enumerate(shards):
        expected_path = f"rank-{rank:02d}.safetensors"
        if (
            record.get("rank") != rank
            or record.get("path") != expected_path
            or record.get("shape") != [24, 8193]
            or record.get("sequences") != 24
            or record.get("validation_tokens") != 196_608
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or int(record.get("bytes", 0)) <= 0
        ):
            raise RuntimeError(f"fixed-independent shard {rank} identity drift")


def load_validation(manifest_path: Path, manifest: dict) -> list[Path]:
    root = manifest_path.resolve().parent
    result: list[Path] = []
    for record in manifest["shards"]:
        path = (root / record["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(f"validation shard is absent: rank {record['rank']}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"validation shard content drift: rank {record['rank']}")
        tensor = load_file(path, device="cpu")["tokens"]
        if list(tensor.shape) != record["shape"] or tensor.dtype != torch.int32:
            raise RuntimeError(f"validation shard tensor drift: rank {record['rank']}")
        result.append(path)
    return result


def aggregate(losses: list[float], targets_per_sequence: int) -> dict:
    loss_sum = math.fsum(losses)
    token_count = len(losses) * targets_per_sequence
    nll = loss_sum / token_count
    return {
        "sequences": len(losses),
        "targets_per_sequence": targets_per_sequence,
        "token_count": token_count,
        "loss_sum": loss_sum,
        "nll": nll,
        "perplexity": math.exp(nll),
        "per_sequence_loss_sums": losses,
    }


def evaluate_hf_prefixes(model, shard_paths: list[Path], device: torch.device, modules) -> dict:
    adapter = modules.LlamaCausalLMValidationAdapter(model)
    output = {}
    for prefix in PREFIX_TARGETS:
        losses: list[float] = []
        with torch.inference_mode():
            for path in shard_paths:
                tokens = load_file(path, device="cpu")["tokens"][:, : prefix + 1]
                for row in range(tokens.shape[0]):
                    value = modules.evaluate_token_batch(
                        adapter,
                        tokens[row : row + 1].to(device),
                        ce_chunk_tokens=256,
                        logit_path="chunked_lm_head",
                    )
                    losses.append(float(value[0].item()))
        output[str(prefix)] = aggregate(losses, prefix)
    return output


def evaluate_native_full(model, shard_paths: list[Path], device: torch.device, modules) -> dict:
    head = model.output
    model.output = torch.nn.Identity()
    adapter = SimpleNamespace(lm_head=head)
    losses: list[float] = []
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH], set_priority=True):
        for path in shard_paths:
            tokens = load_file(path, device="cpu")["tokens"]
            for row in range(tokens.shape[0]):
                batch = tokens[row : row + 1].to(device=device, dtype=torch.long)
                hidden = model(batch[:, :-1], attention_masks=None)
                value = modules.sequence_loss_sums_from_hidden(
                    adapter, hidden, batch[:, 1:], chunk_tokens=256
                )
                losses.append(float(value[0].item()))
    return aggregate(losses, 8192)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--tokenizer-assets", type=Path, required=True)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.work_root.exists() or args.output.exists():
        raise FileExistsError("work and output paths must be new")

    task = json.loads(args.task.read_bytes())
    validate_task(task, args.task_index)
    metadata = args.checkpoint / ".metadata"
    if sha256_file(metadata) != task["checkpoint_metadata_sha256"]:
        raise RuntimeError("checkpoint metadata content drift")
    manifest = json.loads(args.validation_manifest.read_bytes())
    validate_fixed_manifest(manifest)
    validation_data = load_validation(args.validation_manifest, manifest)

    args.work_root.mkdir(parents=True)
    args.output.mkdir(parents=True)
    os.environ["LBT_LIGHT_IMPORT"] = "1"
    sys.path.insert(0, str(args.project_root))
    from low_bits_training.analysis import llama_validation_loss as validation_modules
    from scripts.evaluation import validate_llama8b_conversion_parity as parity_modules
    from scripts.evaluation.run_canonical_lm_eval import _pinned_frequencies

    converted = args.work_root / "converted"
    parity_path = args.output / "canonical-parity.json"
    local_source_id = "public-local-checkpoint"
    local_source_digest = sha256_bytes(local_source_id.encode())
    subprocess.run(
        [
            sys.executable,
            str(args.project_root / "scripts/evaluation/convert_llama8b_dcp_to_hf.py"),
            "--checkpoint", str(args.checkpoint),
            "--output", str(converted),
            "--tokenizer-assets", str(args.tokenizer_assets),
            "--source-job-id", local_source_id,
            "--source-uri-sha256", local_source_digest,
            "--route", task["checkpoint_route"],
            "--source-dtype", "float32",
            "--output-dtype", "bfloat16",
            "--expected-shards", str(task["shard_count"]),
            "--expected-ntokens-seen", str(task["expected_ntokens_seen"]),
        ],
        cwd=args.project_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(args.project_root / "scripts/evaluation/validate_llama8b_canonical_parity.py"),
            "--checkpoint", str(args.checkpoint),
            "--converted", str(converted),
            "--receipt", str(parity_path),
            "--torchtitan-root", str(args.torchtitan_root),
            "--expected-route", task["checkpoint_route"],
            "--expected-step", str(task["step"]),
            "--expected-ntokens-seen", str(task["expected_ntokens_seen"]),
            "--expected-source-job-id", local_source_id,
            "--expected-source-uri-sha256", local_source_digest,
            "--expected-metadata-sha256", task["checkpoint_metadata_sha256"],
            "--expected-shards", str(task["shard_count"]),
            "--device", args.device,
        ],
        cwd=args.project_root,
        check=True,
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    frequencies = _pinned_frequencies(args.torchtitan_root).to(device)
    model = AutoModelForCausalLM.from_pretrained(
        converted,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    with (
        parity_modules._canonical_hf_rope(frequencies),
        parity_modules._canonical_hf_rmsnorm(),
        sdpa_kernel([SDPBackend.MATH], set_priority=True),
    ):
        hf = evaluate_hf_prefixes(model, validation_data, device, validation_modules)
    del model, frequencies
    gc.collect()
    torch.cuda.empty_cache()

    index = json.loads((converted / "model.safetensors.index.json").read_text())
    torchtitan_types = parity_modules.load_pinned_torchtitan(args.torchtitan_root)
    native, native_load = parity_modules.build_native_reference(
        args.checkpoint,
        converted,
        task["checkpoint_route"],
        index["weight_map"],
        device,
        8192,
        torchtitan_types,
    )
    native_full = evaluate_native_full(native, validation_data, device, validation_modules)
    differences = [
        actual - expected
        for actual, expected in zip(
            native_full["per_sequence_loss_sums"],
            hf["8192"]["per_sequence_loss_sums"],
            strict=True,
        )
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "route_key": task["route_key"],
        "step": task["step"],
        "validation_stream_id": FIXED_STREAM_ID,
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "checkpoint_metadata_sha256": task["checkpoint_metadata_sha256"],
        "contract": {
            "sequences": 768,
            "prefix_targets": list(PREFIX_TARGETS),
            "batch_size": 1,
            "ce_chunk_tokens": 256,
            "parameter_dtype": "bfloat16",
            "cross_entropy_dtype": "float32",
            "accumulation_dtype": "float64",
            "attention_backend": "SDPBackend.MATH",
        },
        "hf_prefixes": hf,
        "torchtitan_native_8192": native_full,
        "full_8192_parity": {
            "native_minus_hf_nll": native_full["nll"] - hf["8192"]["nll"],
            "per_sequence_max_abs_loss_sum_error": max(map(abs, differences)),
            "per_sequence_mean_abs_loss_sum_error": math.fsum(map(abs, differences))
            / len(differences),
        },
        "native_load": native_load,
        "canonical_parity_sha256": sha256_file(parity_path),
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    (args.output / "result.json").write_bytes(payload)
    (args.output / "SHA256SUMS").write_text(
        f"{sha256_bytes(payload)}  result.json\n"
        f"{sha256_file(parity_path)}  canonical-parity.json\n"
    )
    print(
        "MFU_LLAMA_FIXED_INDEPENDENT_VALIDATION_PASS "
        f"route={task['route_key']} step={task['step']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
