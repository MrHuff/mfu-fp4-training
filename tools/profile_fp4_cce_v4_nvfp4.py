#!/usr/bin/env python3
"""Stage profile the NVFP4 v4 CCE final-layer path.

This intentionally calls the fp4_matmul v4 Python entrypoints directly so we
can separate quantization, logits GEMM, G-cache production, and backward GEMMs.
It does not change trainer behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass

import torch


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_FP4_ROOT = "/opt/mfu/EXTERNAL_PATH"


def _add_backend_paths(fp4_root: str) -> None:
    paths = [
        fp4_root,
        os.path.join(fp4_root, "fp4_cce_TK"),
        os.path.join(fp4_root, "fp4_cross_entropy"),
        os.path.join(fp4_root, "ml-cross-entropy"),
    ]
    for path in reversed(paths):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


@dataclass
class StageResult:
    name: str
    ms: float


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def _load_lbt_backend():
    path = os.path.join(REPO_ROOT, "low_bits_training", "cce", "backend.py")
    spec = importlib.util.spec_from_file_location("lbt_cce_backend_profile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=65536)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--v", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp4-root", default=os.environ.get("FP4_MATMUL_ROOT", DEFAULT_FP4_ROOT))
    parser.add_argument("--skip-autograd", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ.setdefault("FP4_MATMUL_ROOT", args.fp4_root)
    os.environ.setdefault("FP4_CCE_ASSUME_NONEMPTY_LABELS", "1")
    os.environ.setdefault("FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE", "1")
    os.environ.setdefault("FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE", "1")
    _add_backend_paths(args.fp4_root)

    from fp4_cce_TK.nvfp4_cce_tk import (  # pylint: disable=import-error
        NVFP4Quantized,
        _select_nvfp4_g_quantizer,
        _select_nvfp4_xw_quantizer,
        direct_loss_and_grad_probs,
        tk_nvfp4_gemm,
    )

    torch.manual_seed(args.seed)
    device = "cuda"
    x = (torch.randn(args.m, args.k, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
    w = (torch.randn(args.v, args.k, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
    labels = torch.randint(0, args.v, (args.m,), device=device, dtype=torch.int64)
    valid = labels.ne(-100)

    quant_xw = _select_nvfp4_xw_quantizer()
    quant_g = _select_nvfp4_g_quantizer()

    # Build one reusable state for dependent stages.
    q_x, q_x_col = quant_xw(x, encode_centric=True)
    q_w, q_w_col = quant_xw(w, encode_centric=True)
    logits = tk_nvfp4_gemm(q_x, q_w)
    loss, grad_probs = direct_loss_and_grad_probs(logits, labels, valid, int(args.v))
    g_row_q, g_col_q = quant_g(grad_probs, encode_centric=True)
    scale = torch.tensor(1.0 / float(args.m), device=device, dtype=torch.float32)
    g_row = NVFP4Quantized(g_row_q.fp4, g_row_q.sc, g_row_q.sg * scale)
    g_col = NVFP4Quantized(g_col_q.fp4, g_col_q.sc, g_col_q.sg * scale)
    torch.cuda.synchronize()

    def full_manual():
        lx, lx_col = quant_xw(x, encode_centric=True)
        lw, lw_col = quant_xw(w, encode_centric=True)
        llogits = tk_nvfp4_gemm(lx, lw)
        _loss, lg = direct_loss_and_grad_probs(llogits, labels, valid, int(args.v))
        lg_row_q, lg_col_q = quant_g(lg, encode_centric=True)
        lg_row = NVFP4Quantized(lg_row_q.fp4, lg_row_q.sc, lg_row_q.sg * scale)
        lg_col = NVFP4Quantized(lg_col_q.fp4, lg_col_q.sc, lg_col_q.sg * scale)
        _de = tk_nvfp4_gemm(lg_row, lw_col)
        _dc = tk_nvfp4_gemm(lg_col, lx_col)
        return _loss, _de, _dc

    results = [
        StageResult("quant_x_row_col", _time_cuda(lambda: quant_xw(x, encode_centric=True), args.warmup, args.iters)),
        StageResult("quant_w_row_col", _time_cuda(lambda: quant_xw(w, encode_centric=True), args.warmup, args.iters)),
        StageResult("logits_fp4_gemm", _time_cuda(lambda: tk_nvfp4_gemm(q_x, q_w), args.warmup, args.iters)),
        StageResult(
            "loss_grad_probs",
            _time_cuda(lambda: direct_loss_and_grad_probs(logits, labels, valid, int(args.v)), args.warmup, args.iters),
        ),
        StageResult("quant_g_row_col", _time_cuda(lambda: quant_g(grad_probs, encode_centric=True), args.warmup, args.iters)),
        StageResult("d_hidden_gemm", _time_cuda(lambda: tk_nvfp4_gemm(g_row, q_w_col), args.warmup, args.iters)),
        StageResult("d_weight_gemm", _time_cuda(lambda: tk_nvfp4_gemm(g_col, q_x_col), args.warmup, args.iters)),
        StageResult("manual_total", _time_cuda(full_manual, max(1, args.warmup // 2), max(1, args.iters // 2))),
    ]

    if not args.skip_autograd:
        backend_mod = _load_lbt_backend()
        backend = backend_mod.make_training_loss_backend(
            backend="nvfp4",
            implementation="v4",
            quant_mode="enc",
            ignore_index=-100,
            filter_eps=0.0,
        )
        state: dict[str, torch.Tensor] = {}

        def setup_autograd() -> None:
            state["x"] = x.detach().clone().requires_grad_(True)
            state["w"] = w.detach().clone().requires_grad_(True)

        def run_autograd():
            state["x"].grad = None
            state["w"].grad = None
            out = backend.training_loss(state["x"], state["w"], labels)
            out.backward()
            return out

        def autograd_once():
            setup_autograd()
            return run_autograd()

        results.append(StageResult("lbt_autograd_total", _time_cuda(autograd_once, 1, max(1, args.iters // 2))))

    approx_sum = sum(r.ms for r in results if r.name not in {"manual_total", "lbt_autograd_total"})
    print(f"shape M={args.m} K={args.k} V={args.v}")
    print(f"loss_sample={float(loss.item()):.6f}")
    print(f"{'stage':<22} {'ms':>10} {'share_vs_sum':>13}")
    print("-" * 49)
    for row in results:
        share = row.ms / approx_sum * 100.0 if approx_sum > 0 and row.name not in {"manual_total", "lbt_autograd_total"} else 0.0
        print(f"{row.name:<22} {row.ms:10.3f} {share:12.1f}%")
    print("-" * 49)
    print(f"{'sum_separate_stages':<22} {approx_sum:10.3f}")


if __name__ == "__main__":
    main()
