import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import _tk_quant_localcta as q_local  # type: ignore


def parse_shapes(items):
    shapes = []
    for item in items:
        m_str, k_str = item.lower().split("x")
        shapes.append((int(m_str), int(k_str)))
    return shapes


def make_input(kind: str, m: int, k: int) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    if kind == "normal":
        x = torch.randn(m, k, generator=g, device="cuda", dtype=torch.float32)
    elif kind == "laplace":
        u = torch.rand(m, k, generator=g, device="cuda", dtype=torch.float32) - 0.5
        x = -torch.sign(u) * torch.log1p(-2 * u.abs().clamp(max=0.499999))
    elif kind == "sparse_spikes":
        x = torch.randn(m, k, generator=g, device="cuda", dtype=torch.float32)
        idx = torch.rand(m, k, generator=g, device="cuda") < 0.002
        x[idx] *= 8.0
    elif kind == "row_spikes":
        x = torch.randn(m, k, generator=g, device="cuda", dtype=torch.float32)
        rows = torch.randperm(m, generator=g, device="cuda")[: max(1, m // 64)]
        x[rows] *= 8.0
    else:
        raise ValueError(f"unsupported distribution: {kind}")
    return (x / (k ** 0.25)).to(torch.bfloat16).contiguous()


def benchmark_ms(fn, x: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def reconstruct_prepared(fn, x: torch.Tensor) -> torch.Tensor:
    row_fp4, row_sc_prepared, _, _, _, _ = fn(x, True, True)
    ones = torch.ones((x.size(0) // 128, x.size(1) // 128), device=x.device, dtype=torch.float32)
    return q_local.tk_localcta_reconstruct_row(row_fp4, row_sc_prepared, ones)


def qdq_metrics(recon: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (recon.float() - ref.float()).abs().flatten()
    return {
        "rms": diff.square().mean().sqrt().item(),
        "p99_abs": torch.quantile(diff, 0.99).item(),
        "max_abs": diff.max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["2048x2048", "4096x4096", "16384x2048", "65536x2048", "128000x2048"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--qdq-shape", default="4096x4096")
    parser.add_argument("--qdq-dists", nargs="+", default=["normal", "laplace", "sparse_spikes", "row_spikes"])
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    shapes = parse_shapes(args.shapes)
    qdq_m, qdq_k = parse_shapes([args.qdq_shape])[0]

    configs = [None]
    for threads in (160, 192, 256, 384, 512):
        for pipe_depth in (2, 3, 4):
            for shared_amax in (False, True):
                configs.append((threads, pipe_depth, shared_amax))

    results = {"quant_ms": {}, "qdq": {}}
    for m, k in shapes:
        x = make_input("normal", m, k)
        key = f"{m}x{k}"
        shape_results = {}
        for cfg in configs:
            if cfg is None:
                label = "1cta_prepared"
                ms = benchmark_ms(
                    lambda y: q_local.tk_localcta_quantize_for_gemm_prepared(y, True, True),
                    x, args.warmup, args.iters)
            else:
                threads, pipe_depth, shared_amax = cfg
                label = f"2cta_t{threads}_d{pipe_depth}_{'shared' if shared_amax else 'local'}"
                q_local.tk_localcta_set_2cta_prepared_tuning(threads, pipe_depth, shared_amax)
                ms = benchmark_ms(
                    lambda y: q_local.tk_localcta2_quantize_for_gemm_prepared(y, True, True),
                    x, args.warmup, args.iters)
            shape_results[label] = ms
        results["quant_ms"][key] = shape_results

    for dist in args.qdq_dists:
        x = make_input(dist, qdq_m, qdq_k)
        base = reconstruct_prepared(q_local.tk_localcta_quantize_for_gemm_prepared, x)
        dist_results = {"1cta_prepared": qdq_metrics(base, x)}
        for threads, pipe_depth, shared_amax in ((160, 2, False), (160, 2, True), (192, 2, False), (256, 2, False)):
            q_local.tk_localcta_set_2cta_prepared_tuning(threads, pipe_depth, shared_amax)
            recon = reconstruct_prepared(q_local.tk_localcta2_quantize_for_gemm_prepared, x)
            label = f"2cta_t{threads}_d{pipe_depth}_{'shared' if shared_amax else 'local'}"
            dist_results[label] = qdq_metrics(recon, x)
        results["qdq"][dist] = dist_results

    print(json.dumps(results, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
