#!/usr/bin/env python3
"""Single-block localCTA FFN parity harness.

This script compares a single 1B_legacy Transformer block across four backend
combinations:

- te_te
- localcta_te
- te_localcta
- localcta_localcta

Each combo runs in a fresh subprocess so the FP4 backend env contract is fixed
before imports. The child run captures:

- block-level forward/backward boundaries shared by all combos
- localCTA FFN internal boundaries exposed by `_debug_check_finite`
- sampled tensor stats suitable for cross-combo parity comparison

The parent process merges the child JSON files and emits a compact summary that
highlights the first materially bad boundary when comparing:

- each combo against te_te
- localcta_localcta against te_localcta
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _configure_import_paths(lbt_root: Path, torchtitan_root: Path | None) -> None:
    candidates = [lbt_root]
    if torchtitan_root is not None:
        candidates.append(torchtitan_root)
    else:
        candidates.extend(
            [
                lbt_root / "torchtitan_submodule",
                Path("/opt/mfu/EXTERNAL_PATH"),
            ]
        )

    keep_paths = {candidate.resolve() for candidate in candidates if candidate.exists()}
    for existing in list(sys.path):
        if not existing:
            continue
        try:
            existing_path = Path(existing).resolve()
        except OSError:
            continue
        if existing_path in keep_paths:
            continue
        if existing_path.name in {"low-bits-training", "torchtitan_submodule"}:
            sys.path.remove(existing)

    for candidate in reversed(candidates):
        if not candidate.exists():
            continue
        candidate_str = str(candidate.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


COMBO_BACKENDS = {
    "te_te": ("te", "te"),
    "localcta_te": ("localcta", "te"),
    "te_localcta": ("te", "localcta"),
    "localcta_localcta": ("localcta", "localcta"),
    "localcta_fused_te": ("localcta_fused", "te"),
    "te_localcta_fused": ("te", "localcta_fused"),
    "localcta_fused_localcta_fused": ("localcta_fused", "localcta_fused"),
}

COMBOS = tuple(COMBO_BACKENDS)

FORWARD_BOUNDARIES = (
    "attn_raw_input",
    "attn_norm_ref",
    "attn_out",
    "post_attn_resid",
    "ffn_raw_input",
    "ffn_norm_ref",
    "ffn_out",
    "block_out",
)

BACKWARD_BOUNDARIES = (
    "grad.block_out",
    "grad.ffn_out",
    "grad.ffn_raw_input",
    "grad.post_attn_resid",
    "grad.attn_out",
    "grad.input",
)

QKV_DEBUG_BOUNDARIES = (
    "qkv_fwd.input",
    "qkv_fwd.norm_weight",
    "qkv_fwd.w_qkv",
    "qkv_fwd.xq",
    "qkv_fwd.xk",
    "qkv_fwd.xv",
    "attn_core.q_view",
    "attn_core.k_view",
    "attn_core.v_view",
    "attn_core.q_rope",
    "attn_core.k_rope",
    "attn_core.keys",
    "attn_core.values",
    "attn_core.q_attn",
    "attn_core.k_attn",
    "attn_core.v_attn",
    "attn_core.inner_output",
    "attn_core.output_bshd",
    "attn_core.pre_wo",
    "qkv_bwd.grad_q",
    "qkv_bwd.grad_k",
    "qkv_bwd.grad_v",
    "qkv_bwd.grad_input",
    "qkv_bwd.grad_w_qkv",
    "qkv_bwd.grad_norm_weight",
    "wo_fwd.input",
    "wo_fwd.wo_weight",
    "wo_fwd.output",
    "wo_bwd.grad_output",
    "wo_bwd.grad_input",
    "wo_bwd.grad_w",
)

LOCALCTA_FFN_DEBUG_BOUNDARIES = (
    "ffn_fwd.localcta.input",
    "ffn_fwd.localcta.norm_weight",
    "ffn_fwd.localcta.normed",
    "ffn_fwd.localcta.h1_raw",
    "ffn_fwd.localcta.h3",
    "ffn_fwd.localcta.h",
    "ffn_fwd.localcta.output",
    "ffn_bwd.localcta.dh",
    "ffn_bwd.localcta.dY_sc",
    "ffn_bwd.localcta.dY_sg",
    "ffn_bwd.localcta.grad_w2",
    "ffn_bwd.localcta.dh1",
    "ffn_bwd.localcta.dh3",
    "ffn_bwd.localcta.d_normed_w1",
    "ffn_bwd.localcta.grad_w1",
    "ffn_bwd.localcta.d_normed_w3",
    "ffn_bwd.localcta.grad_w3",
    "ffn_bwd.localcta.d_normed",
    "ffn_bwd.localcta.grad_input",
    "ffn_bwd.localcta.dgamma",
)

QKV_COMPARE_ORDER = (
    "qkv_fwd.input",
    "qkv_fwd.norm_weight",
    "qkv_fwd.w_qkv",
    "qkv_fwd.xq",
    "qkv_fwd.xk",
    "qkv_fwd.xv",
    "attn_core.q_view",
    "attn_core.k_view",
    "attn_core.v_view",
    "attn_core.q_rope",
    "attn_core.k_rope",
    "attn_core.keys",
    "attn_core.values",
    "attn_core.q_attn",
    "attn_core.k_attn",
    "attn_core.v_attn",
    "attn_core.inner_output",
    "attn_core.output_bshd",
    "attn_core.pre_wo",
    "wo_fwd.input",
    "wo_fwd.output",
    "attn_out",
    "post_attn_resid",
    "qkv_bwd.grad_q",
    "qkv_bwd.grad_k",
    "qkv_bwd.grad_v",
    "wo_bwd.grad_output",
    "wo_bwd.grad_input",
    "wo_bwd.grad_w",
    "qkv_bwd.grad_input",
    "qkv_bwd.grad_w_qkv",
    "qkv_bwd.grad_norm_weight",
)

FFN_INCREMENTAL_COMPARE_ORDER = (
    "ffn_norm_ref",
    "ffn_fwd.localcta.input",
    "ffn_fwd.localcta.norm_weight",
    "ffn_fwd.localcta.normed",
    "ffn_fwd.localcta.h1_raw",
    "ffn_fwd.localcta.h3",
    "ffn_fwd.localcta.h",
    "ffn_fwd.localcta.output",
    "ffn_out",
    "block_out",
    "ffn_bwd.localcta.dh",
    "ffn_bwd.localcta.dY_sc",
    "ffn_bwd.localcta.dY_sg",
    "ffn_bwd.localcta.dh1",
    "ffn_bwd.localcta.dh3",
    "ffn_bwd.localcta.d_normed_w1",
    "ffn_bwd.localcta.grad_w1",
    "ffn_bwd.localcta.d_normed_w3",
    "ffn_bwd.localcta.grad_w3",
    "ffn_bwd.localcta.d_normed",
    "ffn_bwd.localcta.grad_input",
)

INTERACTION_COMPARE_ORDER = (
    "attn_core.inner_output",
    "attn_core.pre_wo",
    "wo_fwd.input",
    "wo_fwd.output",
    "attn_out",
    "post_attn_resid",
    "ffn_norm_ref",
    "ffn_out",
    "block_out",
    "wo_bwd.grad_input",
    "qkv_bwd.grad_q",
    "qkv_bwd.grad_k",
    "qkv_bwd.grad_v",
    "qkv_bwd.grad_w_qkv",
    "qkv_bwd.grad_norm_weight",
    "grad.ffn_out",
    "ffn_bwd.localcta.dY_sc",
    "ffn_bwd.localcta.dY_sg",
    "ffn_bwd.localcta.dh",
    "ffn_bwd.localcta.dh1",
    "ffn_bwd.localcta.dh3",
    "ffn_bwd.localcta.d_normed",
    "ffn_bwd.localcta.grad_input",
    "grad.ffn_raw_input",
    "grad.input",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "child"),
        default="all",
        help="Parent orchestration mode or single child run mode.",
    )
    parser.add_argument(
        "--combo",
        choices=COMBOS,
        help="Child combo to execute.",
    )
    parser.add_argument(
        "--device",
        default="cuda:3",
        help="CUDA device, for example cuda:3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size. The exact 54%% stack uses 64.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length. The exact 54%% stack uses 1024.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed used for weights, inputs, and targets.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4096,
        help="Number of sampled elements to retain per captured tensor.",
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=0.1,
        help="Scale applied to the synthetic block input.",
    )
    parser.add_argument(
        "--fp4-matmul-root",
        default="/tmp/fp4_0406_clean",
        help="Runtime root used for localCTA/TK imports.",
    )
    parser.add_argument(
        "--lbt-root",
        type=Path,
        default=REPO_ROOT,
        help="low-bits-training root used for imports inside child runs.",
    )
    parser.add_argument(
        "--torchtitan-root",
        type=Path,
        default=None,
        help="Optional torchtitan root. Defaults to <lbt-root>/torchtitan_submodule.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Child JSON output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/localcta_block_parity"),
        help="Parent output directory.",
    )
    parser.add_argument(
        "--first-bad-cosine-threshold",
        type=float,
        default=0.995,
        help="Heuristic threshold for flagging the first materially bad boundary.",
    )
    parser.add_argument(
        "--first-bad-rms-ratio-threshold",
        type=float,
        default=1.25,
        help="Heuristic RMS ratio threshold for flagging materially inflated outputs.",
    )
    parser.add_argument(
        "--first-bad-zero-fraction-threshold",
        type=float,
        default=0.01,
        help="Heuristic zero-fraction delta threshold for flagging zeroed gradients.",
    )
    return parser.parse_args()


def _set_combo_env(combo: str, fp4_matmul_root: str) -> None:
    attn_backend, ffn_backend = COMBO_BACKENDS[combo]
    os.environ["FP4_MATMUL_ROOT"] = fp4_matmul_root
    os.environ["USE_CUDA_GRAPH"] = "0"
    os.environ["USE_TK_ATTN_DEBUG_FINITE"] = "1"
    os.environ["USE_TK_FFN_DEBUG_FINITE"] = "1"
    os.environ["USE_TK_LOCALCTA_VARIANT"] = "v1"
    os.environ["USE_TK_LOCALCTA_FFN_DISABLE_DIRECT_SPLIT2"] = "0"
    os.environ["USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2"] = "1"
    os.environ["FP4_ATTN_BACKEND"] = attn_backend
    os.environ["FP4_FFN_BACKEND"] = ffn_backend

    # The block harness is for the fast localCTA v1 stack. If we leave the
    # direct-contract fallback enabled, "localcta" probes silently switch to a
    # different backend and the attribution no longer matches the 54% target.
    if "localcta" in attn_backend or "localcta" in ffn_backend:
        os.environ["USE_TK_LOCALCTA_DIRECT_CONTRACT"] = "0"
    else:
        os.environ.pop("USE_TK_LOCALCTA_DIRECT_CONTRACT", None)

    if combo == "te_te":
        os.environ["USE_TK_GEMM"] = "0"
        os.environ["USE_TK_LOCALCTA"] = "0"
        os.environ["USE_TK_LOCALCTA_FUSED"] = "0"
    else:
        os.environ["USE_TK_GEMM"] = "1"
        if "localcta" in attn_backend or "localcta" in ffn_backend:
            os.environ["USE_TK_LOCALCTA"] = "1"
        else:
            os.environ["USE_TK_LOCALCTA"] = "0"
        if "localcta_fused" in attn_backend or "localcta_fused" in ffn_backend:
            os.environ["USE_TK_LOCALCTA_FUSED"] = "1"
        else:
            os.environ["USE_TK_LOCALCTA_FUSED"] = "0"


def _sample_seed(name: str, total_count: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}:{total_count}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _safe_float(value: float | int) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _capture_tensor(name: str, tensor, sample_size: int, seed: int) -> dict:
    import torch

    with torch.no_grad():
        detached = tensor.detach()
        shape = list(detached.shape)
        dtype = str(detached.dtype)
        flat = detached.reshape(-1)
        stats_source = flat
        try:
            finite_mask = torch.isfinite(flat)
        except NotImplementedError:
            stats_source = flat.float()
            finite_mask = torch.isfinite(stats_source)
        total_count = int(flat.numel())
        finite_count = int(finite_mask.sum().item())
        nonfinite_count = total_count - finite_count

        max_abs = None
        mean_abs = None
        rms = None
        zero_fraction = None
        min_val = None
        max_val = None
        if finite_count:
            finite_vals = stats_source[finite_mask].float()
            abs_vals = finite_vals.abs()
            max_abs = _safe_float(abs_vals.max().item())
            mean_abs = _safe_float(abs_vals.mean().item())
            rms = _safe_float(torch.sqrt((finite_vals * finite_vals).mean()).item())
            zero_fraction = _safe_float((finite_vals == 0).float().mean().item())
            min_val = _safe_float(finite_vals.min().item())
            max_val = _safe_float(finite_vals.max().item())

        sample_indices: list[int] = []
        if total_count == 0:
            sample_values: list[float] = []
        elif total_count <= sample_size:
            sample_values = stats_source.float().cpu().tolist()
            sample_indices = list(range(total_count))
        else:
            sample_seed = _sample_seed(name, total_count, seed)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(sample_seed)
            sample_idx = torch.randint(total_count, (sample_size,), generator=gen)
            sample_idx_device = sample_idx.to(stats_source.device)
            sample_values = (
                stats_source.index_select(0, sample_idx_device).float().cpu().tolist()
            )
            sample_indices = sample_idx.tolist()

    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "device": str(detached.device),
        "stride": list(detached.stride()),
        "storage_offset": int(detached.storage_offset()),
        "is_contiguous": bool(detached.is_contiguous()),
        "is_cuda": bool(detached.is_cuda),
        "data_ptr": int(detached.data_ptr()) if detached.numel() else None,
        "storage_data_ptr": int(detached.untyped_storage().data_ptr()) if detached.numel() else None,
        "base_data_ptr": (
            int(detached._base.data_ptr())
            if getattr(detached, "_base", None) is not None and detached._base.numel()
            else None
        ),
        "base_storage_data_ptr": (
            int(detached._base.untyped_storage().data_ptr())
            if getattr(detached, "_base", None) is not None and detached._base.numel()
            else None
        ),
        "total_count": total_count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "zero_fraction": zero_fraction,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "min": min_val,
        "max": max_val,
        "sample_size": len(sample_values),
        "sample_values": sample_values,
        "_sample_indices": sample_indices,
    }


class _TensorRecorder:
    def __init__(self, sample_size: int, seed: int):
        self.sample_size = sample_size
        self.seed = seed
        self.tensors: dict[str, dict] = {}
        self.trace: dict[str, str] = {}
        self._capture_index = 0
        self._live_records: dict[str, dict] = {}

    def _phase_group_for_name(self, name: str) -> str:
        if name.startswith("attn_") or name.startswith("qkv_") or name.startswith("wo_"):
            return "attention"
        if name.startswith("ffn_"):
            return "ffn"
        if name.startswith("grad."):
            return "gradient"
        return "other"

    def _should_track_mutation(self, name: str) -> bool:
        tracked_prefixes = (
            "attn_core.",
            "wo_fwd.",
            "wo_bwd.",
            "qkv_fwd.",
            "qkv_bwd.",
            "attn_out",
            "post_attn_resid",
            "ffn_raw_input",
            "ffn_out",
            "block_out",
            "ffn_bwd.localcta.",
            "grad.",
        )
        return name.startswith(tracked_prefixes)

    def record(self, name: str, tensor, *, phase: str = "unknown") -> None:
        if tensor is None:
            return
        name = _normalize_debug_name(name)
        if name in self.tensors:
            return
        record = _capture_tensor(name, tensor, self.sample_size, self.seed)
        sample_indices = record.pop("_sample_indices", [])
        self._capture_index += 1
        record["capture_index"] = self._capture_index
        record["capture_phase"] = phase
        record["phase_group"] = self._phase_group_for_name(name)
        self.tensors[name] = record
        if self._should_track_mutation(name):
            self._live_records[name] = {
                "tensor": tensor.detach(),
                "sample_indices": sample_indices,
                "sample_values": list(record["sample_values"]),
                "capture_index": record["capture_index"],
                "capture_phase": phase,
            }

    def trace_backend(self, key: str, value: str) -> None:
        self.trace[key] = value

    def alias_groups(self) -> dict[str, list[str]]:
        storage_groups: dict[str, list[str]] = {}
        for name, record in self.tensors.items():
            storage_ptr = record.get("storage_data_ptr")
            if storage_ptr is None:
                continue
            key = str(storage_ptr)
            storage_groups.setdefault(key, []).append(name)
        return {key: names for key, names in storage_groups.items() if len(names) > 1}

    def cross_stage_alias_groups(self) -> dict[str, list[str]]:
        cross_stage = {}
        for storage_ptr, names in self.alias_groups().items():
            phase_groups = {self.tensors[name].get("phase_group") for name in names}
            if len(phase_groups) > 1:
                cross_stage[storage_ptr] = names
        return cross_stage

    def finalize_mutation_checks(self) -> None:
        import torch

        for name, live in self._live_records.items():
            record = self.tensors.get(name)
            if record is None:
                continue
            tensor = live["tensor"]
            try:
                flat = tensor.reshape(-1)
            except RuntimeError:
                record["mutation_check_error"] = "reshape_failed"
                continue
            sample_indices = live["sample_indices"]
            try:
                if not sample_indices:
                    current_samples = flat.float().cpu()
                else:
                    idx = torch.tensor(sample_indices, device=flat.device, dtype=torch.long)
                    current_samples = flat.index_select(0, idx).float().cpu()
            except (NotImplementedError, RuntimeError) as exc:
                record["mutation_check_error"] = str(exc)
                continue
            snapshot = torch.tensor(live["sample_values"], dtype=torch.float32)
            if current_samples.numel() != snapshot.numel():
                record["mutation_sample_size_changed"] = True
                continue
            diff = current_samples - snapshot
            max_abs = _safe_float(diff.abs().max().item()) if diff.numel() else 0.0
            rms_diff = _safe_float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0
            cosine = None
            denom = float(current_samples.norm().item() * snapshot.norm().item()) if diff.numel() else 0.0
            if denom:
                cosine = _safe_float(torch.nn.functional.cosine_similarity(current_samples, snapshot, dim=0).item())
            record["mutation_checked"] = True
            record["mutation_max_abs"] = max_abs
            record["mutation_rms_diff"] = rms_diff
            record["mutation_cosine"] = cosine
            record["mutated_after_capture"] = bool(
                (max_abs is not None and max_abs > 0.0)
                or (rms_diff is not None and rms_diff > 0.0)
            )


def _normalize_debug_name(name: str) -> str:
    return re.sub(r"\[\d+\]", "", name)


def _install_debug_hooks(recorder: _TensorRecorder):
    import torch
    from low_bits_training.quantization import fused_te_linear as ftl
    from low_bits_training.quantization import fp4_converter as fp4c

    original_ffn_check = ftl._debug_check_finite
    original_attn_check = ftl._attn_debug_check_finite
    original_converter_attn_check = fp4c._attn_debug_check_finite
    original_qkv_payload = getattr(ftl, "_get_last_qkv_forward_debug_payload", None)
    original_trace = getattr(ftl, "_trace_backend_choice", None)

    def _check_and_record(name: str, tensor) -> None:
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        recorder.record(name, tensor, phase="debug_hook")
        try:
            finite = torch.isfinite(tensor)
        except NotImplementedError:
            finite = torch.isfinite(tensor.float())
        if not bool(finite.all().item()):
            raise RuntimeError(f"Non-finite tensor captured at {name}")

    def _ffn_wrapper(name: str, tensor) -> None:
        _check_and_record(name, tensor)

    def _attn_wrapper(name: str, tensor) -> None:
        _check_and_record(name, tensor)

    def _trace_wrapper(key: str, value: str) -> None:
        recorder.trace_backend(key, value)
        if original_trace is not None:
            original_trace(key, value)

    ftl._debug_check_finite = _ffn_wrapper
    ftl._attn_debug_check_finite = _attn_wrapper
    fp4c._attn_debug_check_finite = _attn_wrapper
    if original_trace is not None:
        ftl._trace_backend_choice = _trace_wrapper

    def _restore() -> None:
        ftl._debug_check_finite = original_ffn_check
        ftl._attn_debug_check_finite = original_attn_check
        fp4c._attn_debug_check_finite = original_converter_attn_check
        if original_trace is not None:
            ftl._trace_backend_choice = original_trace

    return _restore, original_qkv_payload


def _install_attention_recorders(block, recorder: _TensorRecorder):
    fused_attn = block.attention.fused
    originals = {
        "forward_qkv": fused_attn.forward_qkv,
        "forward_wo": fused_attn.forward_wo,
    }
    saved = {}

    def _retain(name: str, tensor) -> None:
        if tensor is None:
            return
        recorder.record(name, tensor, phase="attn_wrapper")
        if getattr(tensor, "requires_grad", False):
            tensor.retain_grad()
        saved[name] = tensor

    def _forward_qkv(x):
        recorder.record("qkv_fwd.input", x, phase="attn_wrapper")
        recorder.record("qkv_fwd.norm_weight", fused_attn.norm_weight, phase="attn_wrapper")
        recorder.record("qkv_fwd.w_qkv", fused_attn.w_qkv, phase="attn_wrapper")
        xq, xk, xv = originals["forward_qkv"](x)
        _retain("qkv_fwd.xq", xq)
        _retain("qkv_fwd.xk", xk)
        _retain("qkv_fwd.xv", xv)
        return xq, xk, xv

    def _forward_wo(attn_output):
        recorder.record("wo_fwd.input", attn_output, phase="attn_wrapper")
        recorder.record("wo_fwd.wo_weight", fused_attn.wo_weight, phase="attn_wrapper")
        if getattr(attn_output, "requires_grad", False):
            attn_output.retain_grad()
        saved["wo_fwd.input"] = attn_output
        y = originals["forward_wo"](attn_output)
        _retain("wo_fwd.output", y)
        return y

    fused_attn.forward_qkv = _forward_qkv
    fused_attn.forward_wo = _forward_wo

    def _restore():
        fused_attn.forward_qkv = originals["forward_qkv"]
        fused_attn.forward_wo = originals["forward_wo"]

    return _restore, saved


def _build_model_args():
    from low_bits_training.models.models import get_model_config

    return copy.deepcopy(get_model_config("llama3_gc", "1B_legacy"))


def _make_block(combo: str, device: str, seed: int):
    import torch
    from low_bits_training.quantization.fp4_converter import _FusedAttentionWrapper, _NormIdentity
    from low_bits_training.quantization.fused_te_linear import (
        FusedAttentionFP4_TE,
        FusedAttentionFP4_TK,
        FusedFeedForwardFP4_TE,
        FusedFeedForwardFP4_TK,
    )
    from torchtitan.models.llama3.model.model import TransformerBlock

    model_args = _build_model_args()
    torch.manual_seed(seed)
    block = TransformerBlock(0, model_args).to(device=device, dtype=torch.bfloat16)
    block.init_weights()

    attn_backend, ffn_backend = COMBO_BACKENDS[combo]

    if attn_backend == "te":
        fused_attn = FusedAttentionFP4_TE.from_attention(
            block.attention,
            block.attention_norm,
            model_args=model_args,
        )
    else:
        fused_attn = FusedAttentionFP4_TK.from_attention(
            block.attention,
            block.attention_norm,
            model_args=model_args,
        )
    fused_attn._lbt_debug_name = f"{combo}:attn"
    block.attention = _FusedAttentionWrapper(block.attention, fused_attn)
    block.attention_norm = _NormIdentity(model_args.dim, dtype=torch.bfloat16).to(device)

    if ffn_backend == "te":
        fused_ffn = FusedFeedForwardFP4_TE.from_unfused(
            block.feed_forward,
            block.ffn_norm,
        )
    else:
        fused_ffn = FusedFeedForwardFP4_TK.from_unfused(
            block.feed_forward,
            block.ffn_norm,
        )
    fused_ffn._lbt_debug_name = f"{combo}:ffn"
    block.feed_forward = fused_ffn
    block.ffn_norm = _NormIdentity(model_args.dim, dtype=torch.bfloat16).to(device)

    return block, model_args


def _fused_norm_reference(module, raw_input):
    import torch.nn.functional as F

    if hasattr(module, "fused"):
        weight = module.fused.norm_weight
        epsilon = module.fused.epsilon
    else:
        weight = module.norm_weight
        epsilon = module.epsilon
    return F.rms_norm(raw_input, (raw_input.shape[-1],), weight, epsilon)


def _run_child(args: argparse.Namespace) -> int:
    if args.combo is None:
        raise SystemExit("--combo is required in child mode")
    if args.json_out is None:
        raise SystemExit("--json-out is required in child mode")

    _configure_import_paths(args.lbt_root, args.torchtitan_root)
    _set_combo_env(args.combo, args.fp4_matmul_root)

    import torch
    import torch.nn.functional as F
    from low_bits_training.quantization import fused_te_linear as ftl
    from torchtitan.models.llama3.model.model import precompute_freqs_cis

    torch.cuda.set_device(torch.device(args.device))
    recorder = _TensorRecorder(sample_size=args.sample_size, seed=args.seed)
    restore_hooks, get_qkv_payload = _install_debug_hooks(recorder)
    restore_attention_recorders = lambda: None

    try:
        block, model_args = _make_block(args.combo, args.device, args.seed)
        restore_attention_recorders, saved_attention = _install_attention_recorders(
            block,
            recorder,
        )
        block.train()

        torch.manual_seed(args.seed + 1)
        x = torch.randn(
            args.batch_size,
            args.seq_len,
            model_args.dim,
            device=args.device,
            dtype=torch.bfloat16,
        ).mul_(args.input_scale).requires_grad_(True)
        target = x.detach().clone()
        rope_scaling = getattr(
            model_args,
            "rope_scaling",
            getattr(model_args, "rope_scaling_args", None),
        )
        freqs_cis = precompute_freqs_cis(
            model_args.dim // model_args.n_heads,
            args.seq_len,
            theta=model_args.rope_theta,
            scaling_args=rope_scaling,
        ).to(device=args.device)

        recorder.record("attn_raw_input", x, phase="block_forward")
        x.retain_grad()
        attn_norm_ref = _fused_norm_reference(block.attention, x)
        recorder.record("attn_norm_ref", attn_norm_ref, phase="block_forward")
        attn_out = block.attention(block.attention_norm(x), freqs_cis, None)
        if get_qkv_payload is not None:
            payload = ftl._get_last_qkv_forward_debug_payload(clear=True)
        else:
            payload = None
        recorder.record("attn_out", attn_out, phase="block_forward")
        attn_out.retain_grad()

        post_attn = x + attn_out
        recorder.record("post_attn_resid", post_attn, phase="block_forward")
        post_attn.retain_grad()

        recorder.record("ffn_raw_input", post_attn, phase="block_forward")
        ffn_norm_ref = _fused_norm_reference(block.feed_forward, post_attn)
        recorder.record("ffn_norm_ref", ffn_norm_ref, phase="block_forward")
        ffn_out = block.feed_forward(block.ffn_norm(post_attn))
        recorder.record("ffn_out", ffn_out, phase="block_forward")
        ffn_out.retain_grad()

        out = post_attn + ffn_out
        recorder.record("block_out", out, phase="block_forward")
        out.retain_grad()
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()

        if out.grad is not None:
            recorder.record("grad.block_out", out.grad, phase="block_backward")
        if ffn_out.grad is not None:
            recorder.record("grad.ffn_out", ffn_out.grad, phase="block_backward")
        if post_attn.grad is not None:
            recorder.record("grad.post_attn_resid", post_attn.grad, phase="block_backward")
            recorder.record("grad.ffn_raw_input", post_attn.grad, phase="block_backward")
        if attn_out.grad is not None:
            recorder.record("grad.attn_out", attn_out.grad, phase="block_backward")
        if x.grad is not None:
            recorder.record("grad.input", x.grad, phase="block_backward")
        if saved_attention.get("qkv_fwd.xq") is not None and saved_attention["qkv_fwd.xq"].grad is not None:
            recorder.record("qkv_bwd.grad_q", saved_attention["qkv_fwd.xq"].grad, phase="block_backward")
        if saved_attention.get("qkv_fwd.xk") is not None and saved_attention["qkv_fwd.xk"].grad is not None:
            recorder.record("qkv_bwd.grad_k", saved_attention["qkv_fwd.xk"].grad, phase="block_backward")
        if saved_attention.get("qkv_fwd.xv") is not None and saved_attention["qkv_fwd.xv"].grad is not None:
            recorder.record("qkv_bwd.grad_v", saved_attention["qkv_fwd.xv"].grad, phase="block_backward")
        if saved_attention.get("wo_fwd.output") is not None and saved_attention["wo_fwd.output"].grad is not None:
            recorder.record("wo_bwd.grad_output", saved_attention["wo_fwd.output"].grad, phase="block_backward")
        if saved_attention.get("wo_fwd.input") is not None and saved_attention["wo_fwd.input"].grad is not None:
            recorder.record("wo_bwd.grad_input", saved_attention["wo_fwd.input"].grad, phase="block_backward")
        fused_attn = block.attention.fused
        if fused_attn.w_qkv.grad is not None:
            recorder.record("qkv_bwd.grad_w_qkv", fused_attn.w_qkv.grad, phase="block_backward")
        if fused_attn.norm_weight.grad is not None:
            recorder.record("qkv_bwd.grad_norm_weight", fused_attn.norm_weight.grad, phase="block_backward")
        if fused_attn.wo_weight.grad is not None:
            recorder.record("wo_bwd.grad_w", fused_attn.wo_weight.grad, phase="block_backward")

        recorder.finalize_mutation_checks()

        parameter_grads = {}
        for name, param in block.named_parameters():
            if param.grad is None:
                continue
            parameter_grads[name] = _capture_tensor(
                f"param_grad.{name}",
                param.grad,
                args.sample_size,
                args.seed,
            )

        result = {
            "combo": args.combo,
            "device": args.device,
            "fp4_matmul_root": args.fp4_matmul_root,
            "lbt_root": str(args.lbt_root),
            "torchtitan_root": str(args.torchtitan_root) if args.torchtitan_root is not None else None,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "loss": _safe_float(loss.item()),
            "trace": recorder.trace,
            "alias_groups": recorder.alias_groups(),
            "cross_stage_alias_groups": recorder.cross_stage_alias_groups(),
            "tensors": recorder.tensors,
            "parameter_grads": parameter_grads,
            "env": {
                key: os.environ.get(key)
                for key in (
                    "FP4_MATMUL_ROOT",
                    "FP4_ATTN_BACKEND",
                    "FP4_FFN_BACKEND",
                    "USE_TK_GEMM",
                    "USE_TK_LOCALCTA",
                    "USE_TK_LOCALCTA_VARIANT",
                    "USE_TK_LOCALCTA_FUSED",
                    "USE_TK_LOCALCTA_DIRECT_CONTRACT",
                    "USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2",
                    "USE_TK_LOCALCTA_FFN_DISABLE_DIRECT_SPLIT2",
                    "USE_TK_LOCALCTA_FFN_BF16_W2_BWD",
                    "USE_TK_LOCALCTA_FFN_BF16_DGRAD",
                    "USE_TK_QKV_BF16_WGRAD",
                    "USE_TK_QKV_BF16_DGRAD",
                    "USE_TK_QKV_BF16_RMSNORM_BWD",
                    "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND",
                    "USE_TK_SERIAL_RMSNORM_BWD",
                    "USE_CUDA_GRAPH",
                )
            },
            "qkv_payload_meta": _capture_payload_meta(payload),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        restore_attention_recorders()
        return 0
    finally:
        restore_attention_recorders()
        restore_hooks()


def _compare_record(lhs: dict | None, rhs: dict | None) -> dict | None:
    if lhs is None or rhs is None:
        return None
    if lhs["total_count"] != rhs["total_count"]:
        return {
            "shape_mismatch": {
                "lhs": lhs["shape"],
                "rhs": rhs["shape"],
            }
        }

    import torch
    import torch.nn.functional as F

    a = torch.tensor(lhs["sample_values"], dtype=torch.float32)
    b = torch.tensor(rhs["sample_values"], dtype=torch.float32)
    if a.numel() == 0 or b.numel() == 0:
        cosine = None
        max_abs = None
        rms = None
    else:
        denom = float(a.norm().item() * b.norm().item())
        cosine = _safe_float(F.cosine_similarity(a, b, dim=0).item()) if denom else None
        diff = a - b
        max_abs = _safe_float(diff.abs().max().item())
        rms = _safe_float(torch.sqrt((diff * diff).mean()).item())

    lhs_rms = lhs["rms"]
    rhs_rms = rhs["rms"]
    rms_ratio = None
    if lhs_rms is not None and rhs_rms is not None and rhs_rms != 0.0:
        rms_ratio = _safe_float(lhs_rms / rhs_rms)
    zero_fraction_delta = None
    lhs_zero_fraction = lhs.get("zero_fraction")
    rhs_zero_fraction = rhs.get("zero_fraction")
    if lhs_zero_fraction is not None and rhs_zero_fraction is not None:
        zero_fraction_delta = _safe_float(lhs_zero_fraction - rhs_zero_fraction)

    return {
        "shape": {
            "lhs": lhs["shape"],
            "rhs": rhs["shape"],
        },
        "dtype": lhs["dtype"],
        "stride": {
            "lhs": lhs.get("stride"),
            "rhs": rhs.get("stride"),
        },
        "storage_offset": {
            "lhs": lhs.get("storage_offset"),
            "rhs": rhs.get("storage_offset"),
        },
        "is_contiguous": {
            "lhs": lhs.get("is_contiguous"),
            "rhs": rhs.get("is_contiguous"),
        },
        "data_ptr": {
            "lhs": lhs.get("data_ptr"),
            "rhs": rhs.get("data_ptr"),
        },
        "storage_data_ptr": {
            "lhs": lhs.get("storage_data_ptr"),
            "rhs": rhs.get("storage_data_ptr"),
        },
        "capture_phase": {
            "lhs": lhs.get("capture_phase"),
            "rhs": rhs.get("capture_phase"),
        },
        "capture_index": {
            "lhs": lhs.get("capture_index"),
            "rhs": rhs.get("capture_index"),
        },
        "sample_size": min(lhs["sample_size"], rhs["sample_size"]),
        "cosine": cosine,
        "max_abs": max_abs,
        "rms_diff": rms,
        "lhs_rms": lhs_rms,
        "rhs_rms": rhs_rms,
        "rms_ratio": rms_ratio,
        "lhs_zero_fraction": lhs_zero_fraction,
        "rhs_zero_fraction": rhs_zero_fraction,
        "zero_fraction_delta": zero_fraction_delta,
        "lhs_nonfinite": lhs["nonfinite_count"],
        "rhs_nonfinite": rhs["nonfinite_count"],
        "lhs_mutated_after_capture": lhs.get("mutated_after_capture"),
        "rhs_mutated_after_capture": rhs.get("mutated_after_capture"),
        "lhs_mutation_rms_diff": lhs.get("mutation_rms_diff"),
        "rhs_mutation_rms_diff": rhs.get("mutation_rms_diff"),
    }


def _capture_payload_meta(payload: dict | None) -> dict | None:
    import torch

    if not payload:
        return None
    meta = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            meta[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (list, tuple)):
            meta[key] = {
                "kind": type(value).__name__,
                "length": len(value),
            }
        else:
            meta[key] = value
    return meta


def _first_bad_boundary(
    comp_map: dict[str, dict],
    order: tuple[str, ...],
    cosine_threshold: float,
    rms_ratio_threshold: float,
    zero_fraction_threshold: float,
) -> dict | None:
    first_bad = None
    for name in order:
        comp = comp_map.get(name)
        if comp is None or "shape_mismatch" in comp:
            continue
        cosine = comp.get("cosine")
        rms_ratio = comp.get("rms_ratio")
        zero_fraction_delta = comp.get("zero_fraction_delta")
        has_bad_cosine = cosine is not None and cosine < cosine_threshold
        has_bad_rms = (
            rms_ratio is not None
            and (rms_ratio > rms_ratio_threshold or rms_ratio < (1.0 / rms_ratio_threshold))
        )
        has_bad_zero_fraction = (
            zero_fraction_delta is not None
            and abs(zero_fraction_delta) > zero_fraction_threshold
        )
        has_nonfinite = (comp.get("lhs_nonfinite") or 0) > 0 or (comp.get("rhs_nonfinite") or 0) > 0
        if has_bad_cosine or has_bad_rms or has_bad_zero_fraction or has_nonfinite:
            first_bad = {
                "boundary": name,
                "cosine": cosine,
                "rms_diff": comp.get("rms_diff"),
                "max_abs": comp.get("max_abs"),
                "rms_ratio": rms_ratio,
                "zero_fraction_delta": zero_fraction_delta,
                "lhs_zero_fraction": comp.get("lhs_zero_fraction"),
                "rhs_zero_fraction": comp.get("rhs_zero_fraction"),
            }
            break
    if first_bad is not None:
        return first_bad

    lowest_name = None
    lowest_cosine = None
    for name in order:
        comp = comp_map.get(name)
        if comp is None or "shape_mismatch" in comp:
            continue
        cosine = comp.get("cosine")
        if cosine is None:
            continue
        if lowest_cosine is None or cosine < lowest_cosine:
            lowest_cosine = cosine
            lowest_name = name
    if lowest_name is None:
        return None
    comp = comp_map[lowest_name]
    return {
        "boundary": lowest_name,
        "cosine": comp.get("cosine"),
        "rms_diff": comp.get("rms_diff"),
        "max_abs": comp.get("max_abs"),
        "rms_ratio": comp.get("rms_ratio"),
        "zero_fraction_delta": comp.get("zero_fraction_delta"),
        "lhs_zero_fraction": comp.get("lhs_zero_fraction"),
        "rhs_zero_fraction": comp.get("rhs_zero_fraction"),
    }


def _run_parent(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combo_results = {}
    summary_lines = []

    for combo in COMBOS:
        combo_json = args.output_dir / f"{combo}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "child",
            "--combo",
            combo,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--seed",
            str(args.seed),
            "--sample-size",
            str(args.sample_size),
            "--input-scale",
            str(args.input_scale),
            "--fp4-matmul-root",
            args.fp4_matmul_root,
            "--lbt-root",
            str(args.lbt_root),
            "--json-out",
            str(combo_json),
        ]
        if args.torchtitan_root is not None:
            cmd.extend(["--torchtitan-root", str(args.torchtitan_root)])
        env = os.environ.copy()
        run = subprocess.run(cmd, check=False, env=env)
        if run.returncode != 0:
            raise RuntimeError(f"{combo} child run failed with exit code {run.returncode}")
        combo_results[combo] = json.loads(combo_json.read_text())

    te_te = combo_results["te_te"]
    comparisons = {
        "vs_te_te": {},
    }

    ordered_boundaries = (
        FORWARD_BOUNDARIES
        + BACKWARD_BOUNDARIES
        + QKV_DEBUG_BOUNDARIES
        + LOCALCTA_FFN_DEBUG_BOUNDARIES
    )
    for combo in COMBOS:
        if combo == "te_te":
            continue
        combo_comp = {}
        for name in ordered_boundaries:
            combo_comp[name] = _compare_record(
                combo_results[combo]["tensors"].get(name),
                te_te["tensors"].get(name),
            )
        comparisons["vs_te_te"][combo] = combo_comp

    family_specs = {
        "localcta": {
            "attn_only": "localcta_te",
            "ffn_only": "te_localcta",
            "full": "localcta_localcta",
        },
        "localcta_fused": {
            "attn_only": "localcta_fused_te",
            "ffn_only": "te_localcta_fused",
            "full": "localcta_fused_localcta_fused",
        },
    }
    family_analysis = {}
    for family, spec in family_specs.items():
        attn_only = combo_results[spec["attn_only"]]
        full = combo_results[spec["full"]]
        ffn_only = combo_results[spec["ffn_only"]]
        attn_vs_te = {}
        for name in QKV_COMPARE_ORDER:
            attn_vs_te[name] = _compare_record(
                attn_only["tensors"].get(name),
                te_te["tensors"].get(name),
            )
        full_vs_attn = {}
        for name in FFN_INCREMENTAL_COMPARE_ORDER:
            full_vs_attn[name] = _compare_record(
                full["tensors"].get(name),
                attn_only["tensors"].get(name),
            )
        full_vs_ffn = {}
        for name in QKV_COMPARE_ORDER:
            full_vs_ffn[name] = _compare_record(
                full["tensors"].get(name),
                ffn_only["tensors"].get(name),
            )
        family_analysis[family] = {
            "attn_vs_te_te": attn_vs_te,
            "full_vs_attn_only": full_vs_attn,
            "full_vs_ffn_only": full_vs_ffn,
            "first_bad_qkv_boundary": _first_bad_boundary(
                attn_vs_te,
                QKV_COMPARE_ORDER,
                args.first_bad_cosine_threshold,
                args.first_bad_rms_ratio_threshold,
                args.first_bad_zero_fraction_threshold,
            ),
        }
    comparisons["families"] = family_analysis

    interaction_specs = {
        "localcta": {
            "attention_only_delta": ("localcta_te", "te_te"),
            "ffn_only_delta": ("te_localcta", "te_te"),
            "ffn_added_on_localcta_attention": ("localcta_localcta", "localcta_te"),
            "attention_added_on_localcta_ffn": ("localcta_localcta", "te_localcta"),
        },
        "localcta_fused": {
            "attention_only_delta": ("localcta_fused_te", "te_te"),
            "ffn_only_delta": ("te_localcta_fused", "te_te"),
            "ffn_added_on_localcta_attention": ("localcta_fused_localcta_fused", "localcta_fused_te"),
            "attention_added_on_localcta_ffn": ("localcta_fused_localcta_fused", "te_localcta_fused"),
        },
    }
    interaction_analysis = {}
    for family, spec in interaction_specs.items():
        family_result = {}
        for label, (lhs_combo, rhs_combo) in spec.items():
            comp_map = {}
            for name in INTERACTION_COMPARE_ORDER:
                comp_map[name] = _compare_record(
                    combo_results[lhs_combo]["tensors"].get(name),
                    combo_results[rhs_combo]["tensors"].get(name),
                )
            family_result[label] = {
                "lhs_combo": lhs_combo,
                "rhs_combo": rhs_combo,
                "comparisons": comp_map,
                "first_bad_boundary": _first_bad_boundary(
                    comp_map,
                    INTERACTION_COMPARE_ORDER,
                    args.first_bad_cosine_threshold,
                    args.first_bad_rms_ratio_threshold,
                    args.first_bad_zero_fraction_threshold,
                ),
            }
        interaction_analysis[family] = family_result
    comparisons["interaction"] = interaction_analysis

    summary_lines.append(
        f"shape: batch={args.batch_size} seq={args.seq_len} device={args.device} "
        f"fp4_root={args.fp4_matmul_root} lbt_root={args.lbt_root}"
    )
    for combo in COMBOS:
        result = combo_results[combo]
        summary_lines.append(
            f"{combo}: loss={result['loss']:.6f} "
            f"trace={result.get('trace', {})}"
        )
        alias_groups = result.get("alias_groups", {})
        if alias_groups:
            summary_lines.append(f"  alias_groups={alias_groups}")
        cross_stage_alias_groups = result.get("cross_stage_alias_groups", {})
        if cross_stage_alias_groups:
            summary_lines.append(f"  cross_stage_alias_groups={cross_stage_alias_groups}")
        summary_lines.append("")
    for family, analysis in family_analysis.items():
        summary_lines.append(
            f"{family} attention-only vs te_te:"
        )
        for name in QKV_COMPARE_ORDER:
            comp = analysis["attn_vs_te_te"].get(name)
            if comp is None or "shape_mismatch" in comp:
                continue
            summary_lines.append(
                "  "
                + f"{name}: cosine={comp.get('cosine')} rms_diff={comp.get('rms_diff')} "
                + f"rms_ratio={comp.get('rms_ratio')} zero_delta={comp.get('zero_fraction_delta')} "
                + f"max_abs={comp.get('max_abs')}"
            )
        if analysis["first_bad_qkv_boundary"] is not None:
            first_bad_boundary = analysis["first_bad_qkv_boundary"]
            summary_lines.append(
                "  "
                + f"first_bad_qkv_boundary={first_bad_boundary['boundary']} "
                + f"(cosine={first_bad_boundary['cosine']}, "
                + f"rms_diff={first_bad_boundary['rms_diff']}, "
                + f"rms_ratio={first_bad_boundary['rms_ratio']}, "
                + f"zero_delta={first_bad_boundary['zero_fraction_delta']}, "
                + f"max_abs={first_bad_boundary['max_abs']})"
            )
        summary_lines.append("")

    for family, family_result in interaction_analysis.items():
        summary_lines.append(f"{family} interaction deltas:")
        for label, payload in family_result.items():
            first_bad = payload["first_bad_boundary"]
            if first_bad is None:
                summary_lines.append(f"  {label}: no materially bad boundary")
                continue
            summary_lines.append(
                "  "
                + f"{label}: {payload['lhs_combo']} vs {payload['rhs_combo']} "
                + f"first_bad={first_bad['boundary']} "
                + f"cosine={first_bad['cosine']} rms_ratio={first_bad['rms_ratio']} "
                + f"zero_delta={first_bad['zero_fraction_delta']} max_abs={first_bad['max_abs']}"
            )
        summary_lines.append("")
        summary_lines.append(f"{family} full vs attention-only:")
        for name in FFN_INCREMENTAL_COMPARE_ORDER:
            comp = analysis["full_vs_attn_only"].get(name)
            if comp is None or "shape_mismatch" in comp:
                continue
            summary_lines.append(
                "  "
                + f"{name}: cosine={comp.get('cosine')} rms_diff={comp.get('rms_diff')} "
                + f"rms_ratio={comp.get('rms_ratio')} zero_delta={comp.get('zero_fraction_delta')} "
                + f"max_abs={comp.get('max_abs')}"
            )
        summary_lines.append("")

    result = {
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "device": args.device,
        },
        "fp4_matmul_root": args.fp4_matmul_root,
        "lbt_root": str(args.lbt_root),
        "torchtitan_root": str(args.torchtitan_root) if args.torchtitan_root is not None else None,
        "combos": combo_results,
        "comparisons": comparisons,
        "summary": summary_lines,
    }
    summary_json = args.output_dir / "summary.json"
    summary_json.write_text(json.dumps(result, indent=2))
    print("\n".join(summary_lines), flush=True)
    print(f"\nsummary_json: {summary_json}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.mode == "child":
        return _run_child(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
