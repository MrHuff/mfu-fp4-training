#!/usr/bin/env python3
"""Profile one synthetic 1B FP4 forward+backward step."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os


def _self_cuda_us(evt) -> float:
    return float(
        getattr(
            evt,
            "self_cuda_time_total",
            getattr(evt, "self_device_time_total", 0.0),
        )
    )


def _cuda_total_us(evt) -> float:
    return float(
        getattr(
            evt,
            "cuda_time_total",
            getattr(evt, "device_time_total", 0.0),
        )
    )


def _load_bench_module():
    bench_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_synth_1b_fp4.py")
    spec = importlib.util.spec_from_file_location("bench_synth_1b_fp4", bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_copy_trace_module():
    trace_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "trace_python_copies.py",
    )
    spec = importlib.util.spec_from_file_location("trace_python_copies", trace_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    bench = _load_bench_module()

    parser.add_argument("--mode", required=True, choices=bench.MODES)
    parser.add_argument("--flavor", default="1B", choices=["1B", "1B_legacy"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--block", choices=["full", "qkv", "wo", "ffn"], default="full")
    parser.add_argument("--isolation-m", type=int, default=65536)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--mxfp4-backend-version", choices=["v3", "v4"], default="v4")
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument(
        "--activities",
        default="cpu",
        choices=["cpu", "cuda", "cpu,cuda"],
        help="Use CPU-only profiling by default for stable copy traces; enable CUDA when needed.",
    )
    parser.add_argument("--export-trace", default=None)
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    bench.configure_env(args.mode, args.mxfp4_backend_version)

    import torch
    from torch.profiler import profile, ProfilerActivity

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device_index)

    model, model_args = bench.build_model(args.flavor, args.mode, f"cuda:{args.device_index}")
    shape_meta = bench.get_shape_metadata(model, model_args)
    backend_info = bench._runtime_backend_info(args.mode)
    print(json.dumps(
        {
            "mode": args.mode,
            "flavor": args.flavor,
            "block": args.block,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "isolation_m": args.isolation_m if args.block != "full" else None,
            "backend_info": backend_info,
            **shape_meta,
        },
        sort_keys=True,
    ))

    if args.block == "full":
        vocab = model_args.vocab_size
        torch.manual_seed(1234)
        tokens = torch.randint(0, vocab, (args.batch_size, args.seq_len), device="cuda", dtype=torch.long)
        targets = torch.randint(0, vocab, (args.batch_size, args.seq_len), device="cuda", dtype=torch.long)

        def step():
            model.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
    else:
        step, _ = bench.build_isolation_step(
            model, model_args, args.block, args.isolation_m, f"cuda:{args.device_index}"
        )

    step()
    torch.cuda.synchronize()

    activities = []
    if "cpu" in args.activities:
        activities.append(ProfilerActivity.CPU)
    if "cuda" in args.activities:
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=bool(args.export_trace or args.copy_report),
    ) as prof:
        step()
        torch.cuda.synchronize()

    if ProfilerActivity.CUDA in activities:
        sort_by = "self_cuda_time_total"
    else:
        sort_by = "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit))

    copy_rows = None
    if args.export_trace:
        prof.export_chrome_trace(args.export_trace)
        print(f"Exported trace to {args.export_trace}")
        if args.copy_report:
            copy_mod = _load_copy_trace_module()
            rows = copy_mod.summarize_trace_copy_callers(args.export_trace)
            copy_rows = rows
            print("Aggregate PyTorch Copies mapped to Python functions:")
            for row in rows:
                print(f"{row['duration_ms']:8.2f} ms ({row['count']:4d} calls) : {row['caller']}")

    if args.summary_json:
        events = prof.key_averages()
        top_events = sorted(
            events,
            key=lambda evt: _self_cuda_us(evt) if ProfilerActivity.CUDA in activities else evt.self_cpu_time_total,
            reverse=True,
        )[:args.row_limit]
        summary = {
            "mode": args.mode,
            "flavor": args.flavor,
            "block": args.block,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "isolation_m": args.isolation_m if args.block != "full" else None,
            "backend_info": backend_info,
            **shape_meta,
            "activities": args.activities,
            "top_events": [
                {
                    "key": evt.key,
                    "count": evt.count,
                    "self_cpu_us": evt.self_cpu_time_total,
                    "cpu_total_us": evt.cpu_time_total,
                    "self_cuda_us": _self_cuda_us(evt),
                    "cuda_total_us": _cuda_total_us(evt),
                }
                for evt in top_events
            ],
            "copy_rows": copy_rows,
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Wrote summary JSON to {args.summary_json}")


if __name__ == "__main__":
    main()
