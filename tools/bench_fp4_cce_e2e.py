#!/usr/bin/env python3
"""End-to-end autograd benchmark for FP4 CCE backends vs Triton BF16.

This exercises the same forward+backward loss path that `low_bits_training`
uses after the output-head patch, but without instantiating a full trainer.

It measures:
- forward+backward wall time
- loss difference vs Triton BF16
- hidden-gradient cosine vs Triton BF16
- weight-gradient cosine vs Triton BF16
- peak CUDA allocation / reservation deltas over the raw BF16 input baseline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BACKEND_PATH = os.path.join(REPO_ROOT, "low_bits_training", "cce", "backend.py")
FAST_FP4_CCE_V4_ROOT = "/tmp/fp4_matmul_v4_pcache"


def default_fp4_matmul_root() -> str:
    env_root = os.environ.get("FP4_MATMUL_ROOT")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    candidates = [
        os.path.join(os.path.dirname(REPO_ROOT), "fp4_matmul"),
        "/opt/mfu/EXTERNAL_PATH",
        "/opt/mfu/EXTERNAL_PATH",
        FAST_FP4_CCE_V4_ROOT,
        os.path.join(os.path.dirname(REPO_ROOT), "cce", "fp4_matmul"),
        "/opt/mfu/EXTERNAL_PATH",
        "/opt/mfu/EXTERNAL_PATH",
        "/opt/mfu/EXTERNAL_PATH",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(os.path.dirname(REPO_ROOT), "fp4_matmul"))


os.environ.setdefault("FP4_MATMUL_ROOT", default_fp4_matmul_root())


def _load_backend_module():
    spec = importlib.util.spec_from_file_location("lbt_cce_backend", BACKEND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BACKEND = _load_backend_module()


SHAPE_SETS = {
    "smoke2": [
        ("256x256->512", 256, 256, 512),
        ("512x256->512", 512, 256, 512),
    ],
    "default4": [
        ("256x256->512", 256, 256, 512),
        ("512x256->512", 512, 256, 512),
        ("2Kx2K->8K", 2048, 2048, 8192),
        ("4Kx4K->32K", 4096, 4096, 32000),
    ],
    "llama4": [
        ("2Kx2K->8K", 2048, 2048, 8192),
        ("2Kx4K->32K", 2048, 4096, 32000),
        ("4Kx4K->32K", 4096, 4096, 32000),
        ("4Kx8K->32K", 4096, 8192, 32000),
    ],
    "large4": [
        ("4Kx4K->32K", 4096, 4096, 32000),
        ("4Kx8K->32K", 4096, 8192, 32000),
        ("8Kx4K->32K", 8192, 4096, 32000),
        ("4Kx4K->128K", 4096, 4096, 128000),
    ],
    "xlarge4": [
        ("4Kx7K->256K", 4096, 7168, 256000),
        ("16Kx4K->32K", 16384, 4096, 32000),
        ("8Kx8K->128K", 8192, 8192, 128000),
        ("16Kx8K->128K", 16384, 8192, 128000),
    ],
    "prod_final_layer": [
        ("llama3-1b-final", 65536, 2048, 128256),
        ("llama3-8b-final", 16384, 4096, 128256),
    ],
    "nvpaper_1p2b_final": [
        ("nvpaper-1p2b-final", 65536, 2048, 131072),
    ],
}


@dataclass(frozen=True)
class Variant:
    label: str
    backend: str
    implementation: str
    quant_mode: str
    filter_eps: object = 0.0


DEFAULT_VARIANTS = [
    Variant("triton-bf16", "triton_bf16", "v2", "enc"),
    Variant("triton-bf16-auto", "triton_bf16", "v2", "enc", "auto"),
    Variant("bf16-torch-compile", "torch_compile_bf16", "v2", "enc"),
    Variant("nv-v2-enc", "nvfp4", "v2", "enc"),
    Variant("nv-v2-auto", "nvfp4", "v2", "enc", "auto"),
    Variant("nv-v3-enc", "nvfp4", "v3", "enc"),
    Variant("nv-v3-auto", "nvfp4", "v3", "enc", "auto"),
    Variant("nv-v4-enc", "nvfp4", "v4", "enc"),
    Variant("nv-v5", "nvfp4", "v5", "enc"),
    Variant("nv-v5-auto", "nvfp4", "v5", "enc", "auto"),
    Variant("mx-v2-enc", "mxfp4", "v2", "enc"),
    Variant("mx-v2-auto", "mxfp4", "v2", "enc", "auto"),
    Variant("mx-v3-enc", "mxfp4", "v3", "enc"),
    Variant("mx-v3-auto", "mxfp4", "v3", "enc", "auto"),
    Variant("mx-v4-enc", "mxfp4", "v4", "enc"),
]

VARIANT_BY_LABEL = {variant.label: variant for variant in DEFAULT_VARIANTS}


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def error_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = a.float() - b.float()
    return float(diff.abs().max().item()), float(diff.square().mean().sqrt().item())


def relative_error_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a_float = a.float()
    b_float = b.float()
    reference_norm = torch.linalg.vector_norm(b_float)
    relative_l2 = torch.linalg.vector_norm(a_float - b_float) / reference_norm
    norm_ratio = torch.linalg.vector_norm(a_float) / reference_norm
    return float(relative_l2.item()), float(norm_ratio.item())


def bench(fn, setup_fn=None, warmup=2, iters=5) -> float:
    for _ in range(warmup):
        if setup_fn is not None:
            setup_fn()
        fn()
    torch.cuda.synchronize()
    if setup_fn is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    times_ms = []
    for _ in range(iters):
        setup_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return sum(times_ms) / len(times_ms)


def format_mb(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "-"
    return f"{num_bytes / (1024 * 1024):.2f}"


def format_float(val: float | None, precision: int = 3) -> str:
    if val is None:
        return "-"
    return f"{val:.{precision}f}"


def format_sci(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.3e}"


def _argmax_labels(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Find each row's top logit without materializing the full logits matrix."""
    best_values = torch.full(
        (hidden.shape[0],),
        -torch.inf,
        dtype=torch.float32,
        device=hidden.device,
    )
    best_indices = torch.zeros(hidden.shape[0], dtype=torch.int64, device=hidden.device)
    with torch.no_grad():
        for start in range(0, weight.shape[0], chunk_size):
            end = min(start + chunk_size, weight.shape[0])
            logits = hidden @ weight[start:end].T
            chunk_values, chunk_indices = logits.float().max(dim=1)
            update = chunk_values > best_values
            best_values = torch.where(update, chunk_values, best_values)
            best_indices = torch.where(update, chunk_indices + start, best_indices)
    return best_indices


def build_inputs(
    m: int,
    k: int,
    v: int,
    seed: int,
    device: str,
    *,
    hidden_std: float,
    weight_std: float,
    label_mode: str,
    label_chunk_size: int,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    hidden = (
        torch.randn(m, k, generator=g, device=device, dtype=torch.bfloat16)
        * hidden_std
    ).contiguous()
    weight = (
        torch.randn(v, k, generator=g, device=device, dtype=torch.bfloat16)
        * weight_std
    ).contiguous()
    if label_mode == "random":
        labels = torch.randint(0, v, (m,), generator=g, device=device, dtype=torch.int64)
    elif label_mode == "argmax":
        labels = _argmax_labels(hidden, weight, chunk_size=label_chunk_size)
    else:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    return hidden, weight, labels


def run_reference(variant: Variant, hidden_base: torch.Tensor, weight_base: torch.Tensor, labels: torch.Tensor):
    backend = BACKEND.make_training_loss_backend(
        backend=variant.backend,
        implementation=variant.implementation,
        quant_mode=variant.quant_mode,
        ignore_index=-100,
        filter_eps=variant.filter_eps,
    )
    hidden = hidden_base.detach().clone().requires_grad_(True)
    weight = weight_base.detach().clone().requires_grad_(True)
    loss = backend.training_loss(hidden, weight, labels)
    loss.backward()
    torch.cuda.synchronize()
    return {
        "loss": float(loss.item()),
        "d_hidden": hidden.grad.detach().clone(),
        "d_weight": weight.grad.detach().clone(),
    }


def reference_variant_for(variant: Variant) -> Variant:
    if variant.filter_eps not in (0.0, None):
        return VARIANT_BY_LABEL["triton-bf16-auto"]
    return VARIANT_BY_LABEL["triton-bf16"]


def run_timed_case(
    variant: Variant,
    hidden_base: torch.Tensor,
    weight_base: torch.Tensor,
    labels: torch.Tensor,
    ref: dict,
    warmup: int,
    iters: int,
):
    backend = BACKEND.make_training_loss_backend(
        backend=variant.backend,
        implementation=variant.implementation,
        quant_mode=variant.quant_mode,
        ignore_index=-100,
        filter_eps=variant.filter_eps,
    )
    state = {}

    def setup():
        state["hidden"] = hidden_base.detach().clone().requires_grad_(True)
        state["weight"] = weight_base.detach().clone().requires_grad_(True)
        state["loss"] = None

    def run():
        loss = backend.training_loss(state["hidden"], state["weight"], labels)
        state["loss"] = loss
        loss.backward()

    ms = bench(run, setup_fn=setup, warmup=warmup, iters=iters)

    setup()
    run()
    torch.cuda.synchronize()
    loss = float(state["loss"].item())
    d_hidden = state["hidden"].grad.detach()
    d_weight = state["weight"].grad.detach()
    hidden_max_abs_err, hidden_rmse = error_stats(d_hidden, ref["d_hidden"])
    weight_max_abs_err, weight_rmse = error_stats(d_weight, ref["d_weight"])
    hidden_rel_l2, hidden_norm_ratio = relative_error_stats(
        d_hidden, ref["d_hidden"]
    )
    weight_rel_l2, weight_norm_ratio = relative_error_stats(
        d_weight, ref["d_weight"]
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    setup()
    run()
    torch.cuda.synchronize()
    peak_alloc = max(torch.cuda.max_memory_allocated() - baseline_alloc, 0)
    peak_reserved = max(torch.cuda.max_memory_reserved() - baseline_reserved, 0)

    return {
        "time_ms": ms,
        "loss": loss,
        "loss_abs_err": abs(loss - ref["loss"]),
        "cos_hidden": cosine_sim(d_hidden, ref["d_hidden"]),
        "cos_weight": cosine_sim(d_weight, ref["d_weight"]),
        "max_hidden_abs_err": hidden_max_abs_err,
        "rmse_hidden": hidden_rmse,
        "rel_l2_hidden": hidden_rel_l2,
        "norm_ratio_hidden": hidden_norm_ratio,
        "max_weight_abs_err": weight_max_abs_err,
        "rmse_weight": weight_rmse,
        "rel_l2_weight": weight_rel_l2,
        "norm_ratio_weight": weight_norm_ratio,
        "peak_alloc_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "status": "OK",
    }


def render_markdown(results_by_shape: list[tuple[str, list[dict]]], args) -> str:
    lines = []
    lines.append("# FP4 CCE E2E Autograd Benchmark")
    lines.append("")
    lines.append(f"- device: `{args.device}`")
    lines.append(f"- warmup: `{args.warmup}`")
    lines.append(f"- iters: `{args.iters}`")
    lines.append(f"- shape set: `{args.shape_set}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append("")
    for label, rows in results_by_shape:
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Variant | Time (ms) | Loss | |Δloss| vs matching Triton BF16 | cos(dHidden) | rel-L2(dHidden) | norm(dHidden)/ref | cos(dWeight) | rel-L2(dWeight) | norm(dWeight)/ref | PeakAlloc (MB) | PeakResv (MB) | Status |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['variant']} | {format_float(row.get('time_ms'), 3)} | {format_float(row.get('loss'), 6)} | "
                f"{format_sci(row.get('loss_abs_err'))} | {format_float(row.get('cos_hidden'), 6)} | {format_sci(row.get('rel_l2_hidden'))} | "
                f"{format_float(row.get('norm_ratio_hidden'), 6)} | {format_float(row.get('cos_weight'), 6)} | "
                f"{format_sci(row.get('rel_l2_weight'))} | {format_float(row.get('norm_ratio_weight'), 6)} | "
                f"{format_mb(row.get('peak_alloc_bytes'))} | {format_mb(row.get('peak_reserved_bytes'))} | {row.get('status', 'OK')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def selected_variants(variant_labels: str | None) -> list[Variant]:
    if not variant_labels:
        return list(DEFAULT_VARIANTS)
    labels = [part.strip() for part in variant_labels.split(",") if part.strip()]
    out = []
    for label in labels:
        if label not in VARIANT_BY_LABEL:
            raise ValueError(f"Unknown variant label {label!r}")
        out.append(VARIANT_BY_LABEL[label])
    return out


def selected_shapes(shape_set: str, shape_label: str | None):
    shapes = SHAPE_SETS[shape_set]
    if not shape_label:
        return shapes
    matched = [shape for shape in shapes if shape[0] == shape_label]
    if not matched:
        raise ValueError(f"Unknown shape label {shape_label!r} for shape set {shape_set!r}")
    return matched


def run_isolated_variant(shape_set: str, shape_label: str, variant: Variant, args) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--shape-set",
        shape_set,
        "--shape-label",
        shape_label,
        "--variants",
        variant.label,
        "--hidden-std",
        str(args.hidden_std),
        "--weight-std",
        str(args.weight_std),
        "--label-mode",
        args.label_mode,
        "--label-chunk-size",
        str(args.label_chunk_size),
        "--json-row-out",
        json_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.isolated_timeout_sec or max(600, 120 * max(args.iters, 1)),
        )
    except subprocess.TimeoutExpired:
        return {
            "variant": variant.label,
            "time_ms": None,
            "loss": None,
            "loss_abs_err": None,
            "cos_hidden": None,
            "max_hidden_abs_err": None,
            "rmse_hidden": None,
            "cos_weight": None,
            "max_weight_abs_err": None,
            "rmse_weight": None,
            "peak_alloc_bytes": None,
            "peak_reserved_bytes": None,
            "status": "ERROR: variant timeout",
        }
    try:
        if proc.returncode != 0:
            status = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"subprocess failed ({proc.returncode})"
            return {
                "variant": variant.label,
                "time_ms": None,
                "loss": None,
                "loss_abs_err": None,
                "cos_hidden": None,
                "max_hidden_abs_err": None,
                "rmse_hidden": None,
                "cos_weight": None,
                "max_weight_abs_err": None,
                "rmse_weight": None,
                "peak_alloc_bytes": None,
                "peak_reserved_bytes": None,
                "status": f"ERROR: {status}",
            }
        with open(json_path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["variant"] = variant.label
        return row
    finally:
        try:
            os.remove(json_path)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--shape-set", choices=sorted(SHAPE_SETS), default="default4")
    parser.add_argument("--shape-label", type=str, default=None)
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--isolated-variants", action="store_true")
    parser.add_argument("--isolated-timeout-sec", type=int, default=None)
    parser.add_argument("--json-row-out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-std", type=float, default=0.02)
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--label-mode", choices=("random", "argmax"), default="random")
    parser.add_argument("--label-chunk-size", type=int, default=8192)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("This benchmark currently supports CUDA only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.hidden_std <= 0 or args.weight_std <= 0:
        raise ValueError("Input standard deviations must be positive.")
    if args.label_chunk_size <= 0:
        raise ValueError("--label-chunk-size must be positive.")

    variants = selected_variants(args.variants)
    shapes = selected_shapes(args.shape_set, args.shape_label)

    results_by_shape = []
    for idx, (label, m, k, v) in enumerate(shapes):
        if args.isolated_variants:
            hidden_base = weight_base = labels = None
            refs = {}
        else:
            hidden_base, weight_base, labels = build_inputs(
                m,
                k,
                v,
                args.seed + idx,
                args.device,
                hidden_std=args.hidden_std,
                weight_std=args.weight_std,
                label_mode=args.label_mode,
                label_chunk_size=args.label_chunk_size,
            )
            refs = {
                "triton-bf16": run_reference(VARIANT_BY_LABEL["triton-bf16"], hidden_base, weight_base, labels),
                "triton-bf16-auto": run_reference(VARIANT_BY_LABEL["triton-bf16-auto"], hidden_base, weight_base, labels),
            }
        rows = []
        print()
        print("=" * 120)
        print(f"{label}  [M={m}, K={k}, V={v}]")
        print("=" * 120)
        print(
            f"{'Variant':<12} | {'Time (ms)':>10} | {'Loss':>10} | {'|Δloss|':>11} | "
            f"{'cos(dH)':>10} | {'relL2(dH)':>10} | {'normH/ref':>10} | "
            f"{'cos(dW)':>10} | {'relL2(dW)':>10} | {'normW/ref':>10} | "
            f"{'PeakAlloc(MB)':>13} | {'PeakResv(MB)':>12} | {'Status':<20}"
        )
        print("─" * 120)
        for variant in variants:
            if args.isolated_variants:
                row = run_isolated_variant(args.shape_set, label, variant, args)
            else:
                try:
                    row = run_timed_case(
                        variant=variant,
                        hidden_base=hidden_base,
                        weight_base=weight_base,
                        labels=labels,
                        ref=refs[reference_variant_for(variant).label],
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                except Exception as ex:
                    row = {
                        "time_ms": None,
                        "loss": None,
                        "loss_abs_err": None,
                        "cos_hidden": None,
                        "max_hidden_abs_err": None,
                        "rmse_hidden": None,
                        "cos_weight": None,
                        "max_weight_abs_err": None,
                        "rmse_weight": None,
                        "peak_alloc_bytes": None,
                        "peak_reserved_bytes": None,
                        "status": f"ERROR: {ex}",
                    }
                row["variant"] = variant.label
            rows.append(row)
            print(
                f"{variant.label:<12} | {format_float(row.get('time_ms'), 3):>10} | {format_float(row.get('loss'), 6):>10} | {format_sci(row.get('loss_abs_err')):>11} | "
                f"{format_float(row.get('cos_hidden'), 6):>10} | {format_sci(row.get('rel_l2_hidden')):>10} | {format_float(row.get('norm_ratio_hidden'), 6):>10} | "
                f"{format_float(row.get('cos_weight'), 6):>10} | {format_sci(row.get('rel_l2_weight')):>10} | {format_float(row.get('norm_ratio_weight'), 6):>10} | "
                f"{format_mb(row.get('peak_alloc_bytes')):>13} | {format_mb(row.get('peak_reserved_bytes')):>12} | {row.get('status', 'OK'):<20}"
            )
        results_by_shape.append((label, rows))

    if args.markdown_out:
        markdown = render_markdown(results_by_shape, args)
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write(markdown)
        print()
        print(f"Wrote markdown summary to {args.markdown_out}")

    if args.json_row_out:
        if len(results_by_shape) != 1 or len(results_by_shape[0][1]) != 1:
            raise ValueError("--json-row-out requires exactly one selected shape and one selected variant")
        with open(args.json_row_out, "w", encoding="utf-8") as f:
            json.dump(results_by_shape[0][1][0], f)


if __name__ == "__main__":
    main()
