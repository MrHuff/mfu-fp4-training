#!/usr/bin/env python3
"""Compare localCTA v4 fused W2-dgrad SiLU producer with the stable split2 path.

Run with torch.distributed.run so each rank exercises one visible GPU.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CYPARI_NO_SIGNALS", "1")
os.environ.setdefault("USE_TK_GEMM", "1")
os.environ.setdefault("USE_TK_QUANT", "1")
os.environ.setdefault("USE_TK_LOCALCTA", "1")
os.environ.setdefault("USE_TK_LOCALCTA_VARIANT", "v4")
os.environ.setdefault("NVTE_NVFP4_DISABLE_RHT", "1")
os.environ.setdefault("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "1")
os.environ.setdefault("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "1")
os.environ.setdefault("NVTE_NVFP4_ENCODE_CENTRIC", "0")
os.environ.setdefault("NVFP4_USE_RHT", "0")
os.environ.setdefault("NVFP4_USE_STOCHASTIC_ROUNDING", "0")
os.environ.setdefault("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", "0")

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _max_u8_delta(a: torch.Tensor, b: torch.Tensor) -> int:
    av = a.view(torch.uint8).reshape(-1)
    bv = b.view(torch.uint8).reshape(-1)
    if av.numel() == 0:
        return 0
    return int((av.to(torch.int16) - bv.to(torch.int16)).abs().max().item())


def _rel_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    if tuple(a.shape) != tuple(b.shape):
        return float("nan"), float("nan"), float("nan")
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    denom = bf.abs().clamp_min(1.0e-12)
    return (
        float(diff.max().item()),
        float((diff / denom).max().item()),
        float(torch.nn.functional.cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1)).item()),
    )


def _signal_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float, float, float]:
    if tuple(a.shape) != tuple(b.shape):
        return (float("nan"),) * 5
    af = a.float()
    bf = b.float()
    a_rms = float(af.square().mean().sqrt().item())
    b_rms = float(bf.square().mean().sqrt().item())
    norm_ratio = float(af.norm().div(bf.norm().clamp_min(1.0e-30)).item())
    a_zero = float((af == 0).float().mean().item())
    b_zero = float((bf == 0).float().mean().item())
    return a_rms, b_rms, norm_ratio, a_zero, b_zero


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", "--m", dest="m", type=int, default=256)
    parser.add_argument("--input-dim", "--k", dest="k", type=int, default=2048)
    parser.add_argument("--hidden-dim", "--h", dest="h", type=int, default=1024)
    parser.add_argument("--dy-scale", type=float, default=0.01)
    parser.add_argument("--w2-scale", type=float, default=0.01)
    parser.add_argument("--act-scale", type=float, default=0.01)
    parser.add_argument("--config-id", type=int, default=4)
    parser.add_argument("--raw-threads", type=int, choices=(160, 192, 256), default=256)
    parser.add_argument("--raw-pipe-depth", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--producer",
        choices=("w2", "stable", "twostage", "cat", "deriv"),
        default="w2",
        help=(
            "Producer under test: w2 fuses W2 dgrad with split2 quant; "
            "stable tests the prepared localCTA v4 fused SiLU-deriv split2 "
            "producer; twostage tests the experimental twostage variant; "
            "cat tests the live Bridge cat split producer; deriv tests only "
            "the localCTA SiLU derivative BF16 producer."
        ),
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="For --producer cat, skip final outer-scale contract finalization.",
    )
    parser.add_argument(
        "--production-finalize",
        action="store_true",
        help=(
            "Finalize prepared split2 scales and use the outer-SG consumers, "
            "matching the production FFN backward path."
        ),
    )
    parser.add_argument(
        "--live-quant",
        action="store_true",
        help="Use the same standard localCTA v4 quant route as the live Bridge backward.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Repeat the fused producer launch into the same prepared buffers.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=0,
        help="Print payload deltas every N repeated launches; 0 reports only the final launch.",
    )
    parser.add_argument(
        "--sync-each",
        action="store_true",
        help="Synchronize after every repeated launch instead of only at the end.",
    )
    parser.add_argument(
        "--fresh-buffers",
        action="store_true",
        help="Allocate a fresh fused output payload before each repeated producer launch.",
    )
    parser.add_argument(
        "--retain-fresh-buffers",
        action="store_true",
        help="Keep every fresh fused payload alive so CUDA allocator addresses are not reused.",
    )
    parser.add_argument(
        "--buffer-ring",
        type=int,
        default=1,
        help="Rotate across this many fused output payloads during repeated launches.",
    )
    parser.add_argument(
        "--zero-each",
        action="store_true",
        help="Zero the fused output payload before each repeated producer launch.",
    )
    parser.add_argument("--real-sg", action="store_true")
    parser.add_argument("--unit-sg", action="store_true")
    parser.add_argument(
        "--check-consumers",
        action="store_true",
        help="Also run the W13-wgrad and split2-dgrad consumers on ref/fused payloads.",
    )
    parser.add_argument(
        "--check-bf16-consumers",
        action="store_true",
        help="Also compare the quantized consumers against direct BF16 GEMMs.",
    )
    parser.add_argument(
        "--overlap-matmul-size",
        type=int,
        default=0,
        help="Run square BF16 GEMMs on a side stream while launching the producer.",
    )
    parser.add_argument(
        "--overlap-matmul-repeats",
        type=int,
        default=8,
        help="Number of side-stream GEMMs launched per producer iteration.",
    )
    parser.add_argument(
        "--overlap-all-reduce-mib",
        type=int,
        default=0,
        help="Launch an asynchronous NCCL all-reduce of this size before the producer.",
    )
    parser.add_argument(
        "--producer-priority-stream",
        action="store_true",
        help="Launch the producer on a high-priority stream with event handoff.",
    )
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1234 + rank)

    from low_bits_training.quantization import fused_te_linear as fte
    from low_bits_training.quantization.tk_gemm import (
        _get_tk,
        _get_tk_quant_for_gemm,
        _prepare_localcta_v4_outer_sg_for_direct,
        _prepare_localcta_v4_chunkgrid_for_batched,
        tk_grouped_wgrad_gemm,
        tk_split_wgrad_gemm,
    )

    tk = _get_tk()
    tkq = _get_tk_quant_for_gemm()
    te_fused = fte._get_te_fused()
    if hasattr(tkq._mod, "tk_localcta_set_2cta_raw_split2_tuning"):
        tkq._mod.tk_localcta_set_2cta_raw_split2_tuning(
            args.raw_threads, args.raw_pipe_depth
        )

    M, K, H = args.m, args.k, args.h
    dY = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * args.dy_scale).contiguous()
    w2 = (torch.randn(K, H, device=device, dtype=torch.bfloat16) * args.w2_scale).contiguous()
    h1 = (torch.randn(M, H, device=device, dtype=torch.bfloat16) * args.act_scale).contiguous()
    h3 = (torch.randn(M, H, device=device, dtype=torch.bfloat16) * args.act_scale).contiguous()

    if args.live_quant:
        dY_staging = torch.empty_like(dY)
        dY_result = tkq.tk_quantize_for_gemm_maybe_borrow(
            dY,
            dY_staging,
            True,
            fte.use_nvfp4_encode_centric(),
        )
        dY_sg_t = (
            dY_result[5]
            if len(dY_result) > 5 and torch.is_tensor(dY_result[5]) and dY_result[5].numel() > 0
            else dY_result[4]
        )
        dY_q = fte._TKQuantized(
            dY_result[0],
            dY_result[1],
            dY_result[4],
            dY_result[2],
            dY_result[3],
            dY_sg_t,
        )
        w2_q = fte._fast_quantize(
            w2,
            tk_swizzle=True,
            use_localcta_override=True,
            nvfp4_role="weight",
        )
    else:
        dY_q = fte._fast_quantize_localcta_v4_opt(dY, nvfp4_role="grad")
        w2_q = fte._fast_quantize_localcta_v4_opt(w2, nvfp4_role="weight")

    dh = torch.empty(M, H, device=device, dtype=torch.bfloat16)
    tk.nvfp4_gemm(*dY_q._tk_row, *w2_q._tk_col, dh)

    dh1 = torch.empty_like(h1)
    dh3 = torch.empty_like(h3)
    if hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
        te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(dh, h3, h1, dh1, dh3)
    else:
        te_fused.fused_silu_deriv_dual_mul_bf16_out(
            dh,
            h3,
            h1,
            dh1,
            dh3,
            torch.empty(1, device=device, dtype=torch.float32),
            torch.empty(1, device=device, dtype=torch.float32),
        )
    ref = tkq._mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
        M, H, H, device
    )
    tkq._mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
        dh1, dh3, ref[6], ref[7], ref[9], ref[10], ref[8], ref[11]
    )

    def _alloc_outer_sgs():
        row0 = torch.empty((M // 256, 1), dtype=torch.float32, device=device)
        row1 = torch.empty_like(row0)
        col_full = torch.empty((1, 2 * (H // 256)), dtype=torch.float32, device=device)
        return row0, row1, col_full.narrow(1, 0, H // 256), col_full.narrow(1, H // 256, H // 256), col_full

    outer_sgs: dict[int, tuple[torch.Tensor, ...]] = {}

    def _finalize_prepared(payload):
        key = id(payload)
        outer = outer_sgs.get(key)
        if outer is None:
            outer = _alloc_outer_sgs()
            outer_sgs[key] = outer
        row0, row1, col0, col1, _ = outer
        tkq._mod.tk_localcta_finalize_split2_for_gemm_prepared_inplace(
            payload[1][0], payload[2][0], row0,
            payload[4][0], payload[5][0], col0,
            payload[1][1], payload[2][1], row1,
            payload[4][1], payload[5][1], col1,
        )
        return outer

    if args.production_finalize:
        if args.producer == "cat":
            raise ValueError("--production-finalize does not apply to the cat producer")
        _finalize_prepared(ref)
    elif args.producer == "cat" and not args.no_finalize:
        # The live cat producer finalizes its scale contract in the launch. Do
        # the same to the stable prepared reference before comparing payloads.
        _finalize_prepared(ref)
    dh1_ref = dh1.clone() if args.producer == "deriv" else None
    dh3_ref = dh3.clone() if args.producer == "deriv" else None

    def _alloc_fused():
        return tkq._mod.tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
            M, H, H, device
        )

    def _alloc_test_payload():
        if args.producer == "cat":
            return tkq._mod.tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc(
                M, H, device
            )
        return _alloc_fused()

    ring_size = max(1, int(args.buffer_ring))
    fused_ring = [_alloc_test_payload() for _ in range(ring_size)]
    retained_fused = list(fused_ring)
    fused = fused_ring[0]
    def _zero_tensor_tree(obj, seen: set[int]) -> None:
        if torch.is_tensor(obj):
            key = obj.data_ptr()
            if key in seen:
                return
            seen.add(key)
            try:
                obj.zero_()
            except RuntimeError:
                obj.view(torch.uint8).zero_()
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _zero_tensor_tree(item, seen)

    def _zero_fused() -> None:
        _zero_tensor_tree(fused, set())

    if args.real_sg and args.unit_sg:
        raise ValueError("--real-sg and --unit-sg are mutually exclusive")
    use_unit_sg = (
        args.unit_sg
        if (args.real_sg or args.unit_sg)
        else fte.use_tk_localcta_v4_w2_dgrad_silu_producer_unit_sg()
    )
    if not use_unit_sg:
        a_sg = _prepare_localcta_v4_outer_sg_for_direct(
            dY_q._tk_row[2],
            max(M // 256, 1),
            device,
            True,
        )
        b_sg = _prepare_localcta_v4_outer_sg_for_direct(
            w2_q._tk_col[2],
            max(H // 256, 1),
            device,
            False,
        )
    else:
        a_sg = _prepare_localcta_v4_outer_sg_for_direct(
            torch.ones(max(M // 256, 1), 1, device=device, dtype=torch.float32),
            max(M // 256, 1),
            device,
            True,
        )
        b_sg = _prepare_localcta_v4_outer_sg_for_direct(
            torch.ones(max(H // 256, 1), 1, device=device, dtype=torch.float32),
            max(H // 256, 1),
            device,
            False,
        )
    def _launch_fused() -> None:
        if args.producer == "deriv":
            tkq._mod.tk_localcta_silu_deriv_split_bf16_launch_inplace(
                dh,
                h3,
                h1,
                dh1,
                dh3,
            )
            return
        if args.producer == "stable":
            tkq._mod.tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
                dh,
                h3,
                h1,
                fused[6],
                fused[7],
                fused[9],
                fused[10],
                fused[8],
                fused[11],
            )
            return
        if args.producer == "twostage":
            tkq._mod.tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace(
                dh,
                h3,
                h1,
                fused[6],
                fused[7],
                fused[9],
                fused[10],
                fused[8],
                fused[11],
            )
            return
        if args.producer == "cat":
            tkq._mod.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
                dh,
                h3,
                h1,
                dh1,
                dh3,
                *fused[:16],
                not args.no_finalize,
            )
            return
        tk.nvfp4_w2_dgrad_silu_quant_gemm(
            dY_q._tk_row[0],
            dY_q._tk_row[1],
            a_sg,
            w2_q._tk_col[0],
            w2_q._tk_col[1],
            b_sg,
            h3,
            h1,
            fused[6],
            fused[7],
            fused[8],
            fused[9],
            fused[10],
            fused[11],
            args.config_id,
        )

    def _payload_stats() -> tuple[int, int, int, int, float, float, float, float, float, float]:
        if args.producer == "deriv":
            assert dh1_ref is not None and dh3_ref is not None
            dh1_abs, dh1_rel, dh1_cos = _rel_stats(dh1, dh1_ref)
            dh3_abs, dh3_rel, dh3_cos = _rel_stats(dh3, dh3_ref)
            return (
                _max_u8_delta(dh1, dh1_ref),
                0,
                _max_u8_delta(dh3, dh3_ref),
                0,
                dh1_abs,
                dh1_rel,
                dh1_cos,
                dh3_abs,
                dh3_rel,
                dh3_cos,
            )
        if args.producer == "cat":
            row_delta = max(
                _max_u8_delta(fused[0], ref[0][0]),
                _max_u8_delta(fused[6], ref[0][1]),
            )
            row_sc_delta = max(
                _max_u8_delta(fused[1], ref[1][0]),
                _max_u8_delta(fused[7], ref[1][1]),
            )
            col_delta = _max_u8_delta(fused[16], ref[9])
            col_sc_delta = _max_u8_delta(fused[17], ref[10])
            if args.no_finalize:
                row_sg_refs = ref[2]
                col_sg_ref = ref[11]
                row_sg_values = [fused[12], fused[14]]
                col_sg_value = torch.cat([fused[13], fused[15]], dim=0)
            else:
                row0, row1, _, _, col_full = outer_sgs[id(ref)]
                row_sg_refs = [row0, row1]
                col_sg_ref = col_full
                row_sg_values = [fused[4], fused[10]]
                col_sg_value = fused[18]
            row_sg_abs_0, row_sg_rel_0, row_sg_cos_0 = _rel_stats(
                row_sg_values[0], row_sg_refs[0]
            )
            row_sg_abs_1, row_sg_rel_1, row_sg_cos_1 = _rel_stats(
                row_sg_values[1], row_sg_refs[1]
            )
            col_sg_abs, col_sg_rel, col_sg_cos = _rel_stats(col_sg_value, col_sg_ref)
            return (
                row_delta,
                row_sc_delta,
                col_delta,
                col_sc_delta,
                max(row_sg_abs_0, row_sg_abs_1),
                max(row_sg_rel_0, row_sg_rel_1),
                min(row_sg_cos_0, row_sg_cos_1),
                col_sg_abs,
                col_sg_rel,
                col_sg_cos,
            )
        row_delta = _max_u8_delta(fused[6], ref[6])
        row_sc_delta = _max_u8_delta(fused[7], ref[7])
        col_delta = _max_u8_delta(fused[9], ref[9])
        col_sc_delta = _max_u8_delta(fused[10], ref[10])
        row_sg_abs, row_sg_rel, row_sg_cos = _rel_stats(fused[8], ref[8])
        col_sg_abs, col_sg_rel, col_sg_cos = _rel_stats(fused[11], ref[11])
        return (
            row_delta,
            row_sc_delta,
            col_delta,
            col_sc_delta,
            row_sg_abs,
            row_sg_rel,
            row_sg_cos,
            col_sg_abs,
            col_sg_rel,
            col_sg_cos,
        )

    def _print_payload_stats(iter_idx: int) -> None:
        (
            row_delta,
            row_sc_delta,
            col_delta,
            col_sc_delta,
            row_sg_abs,
            row_sg_rel,
            row_sg_cos,
            col_sg_abs,
            col_sg_rel,
            col_sg_cos,
        ) = _payload_stats()
        print(
            f"[rank {rank}] iter={iter_idx} producer={args.producer} "
            f"M={M} K={K} H={H} cfg={args.config_id} "
            f"unit_sg={use_unit_sg} row_u8_delta={row_delta} "
            f"row_sc_u8_delta={row_sc_delta} col_u8_delta={col_delta} "
            f"col_sc_u8_delta={col_sc_delta} row_sg_abs={row_sg_abs:.3e} "
            f"row_sg_rel={row_sg_rel:.3e} row_sg_cos={row_sg_cos:.6f} "
            f"col_sg_abs={col_sg_abs:.3e} col_sg_rel={col_sg_rel:.3e} "
            f"col_sg_cos={col_sg_cos:.6f}",
            flush=True,
        )

    overlap_stream = None
    overlap_matmul = None
    if args.overlap_matmul_size > 0:
        overlap_stream = torch.cuda.Stream(device=device)
        overlap_dim = int(args.overlap_matmul_size)
        overlap_a = torch.randn(
            overlap_dim, overlap_dim, device=device, dtype=torch.bfloat16
        )
        overlap_b = torch.randn_like(overlap_a)
        overlap_out = torch.empty_like(overlap_a)
        overlap_matmul = (overlap_a, overlap_b, overlap_out)

    overlap_all_reduce = None
    if args.overlap_all_reduce_mib > 0:
        overlap_elements = int(args.overlap_all_reduce_mib) * 1024 * 1024 // 2
        overlap_all_reduce = torch.ones(
            overlap_elements, device=device, dtype=torch.bfloat16
        )

    pending_collectives = []
    producer_stream = (
        torch.cuda.Stream(device=device, priority=-1)
        if args.producer_priority_stream
        else None
    )

    def _launch_overlap() -> None:
        if overlap_stream is not None and overlap_matmul is not None:
            overlap_stream.wait_stream(torch.cuda.current_stream(device))
            overlap_a, overlap_b, overlap_out = overlap_matmul
            with torch.cuda.stream(overlap_stream):
                for _ in range(max(1, int(args.overlap_matmul_repeats))):
                    torch.mm(overlap_a, overlap_b, out=overlap_out)
        if overlap_all_reduce is not None:
            pending_collectives.append(
                dist.all_reduce(overlap_all_reduce, async_op=True)
            )

    iters = max(1, int(args.iters))
    report_every = max(0, int(args.report_every))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for iter_idx in range(1, iters + 1):
        if args.fresh_buffers and iter_idx > 1:
            fused = _alloc_test_payload()
            if args.retain_fresh_buffers:
                retained_fused.append(fused)
        elif ring_size > 1:
            fused = fused_ring[(iter_idx - 1) % ring_size]
        if args.zero_each:
            _zero_fused()
        _launch_overlap()
        if producer_stream is None:
            _launch_fused()
        else:
            caller_stream = torch.cuda.current_stream(device)
            producer_stream.wait_stream(caller_stream)
            with torch.cuda.stream(producer_stream):
                _launch_fused()
            caller_stream.wait_stream(producer_stream)
        if args.production_finalize and args.producer != "deriv":
            _finalize_prepared(fused)
        is_intermediate_report = (
            iter_idx != iters and report_every and iter_idx % report_every == 0
        )
        if args.sync_each or is_intermediate_report:
            torch.cuda.synchronize()
        if is_intermediate_report:
            _print_payload_stats(iter_idx)
    end_event.record()
    for work in pending_collectives:
        work.wait()
    torch.cuda.synchronize()
    total_ms = float(start_event.elapsed_time(end_event))
    _print_payload_stats(iters)
    print(
        f"[rank {rank}] producer={args.producer} iters={iters} "
        f"total_ms={total_ms:.3f} avg_ms={total_ms / iters:.3f}",
        flush=True,
    )

    if args.check_consumers:
        if args.producer == "cat" and args.no_finalize:
            raise ValueError("--check-consumers requires the finalized cat contract")
        x = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.01).contiguous()
        w1 = (torch.randn(H, K, device=device, dtype=torch.bfloat16) * 0.01).contiguous()
        w3 = (torch.randn(H, K, device=device, dtype=torch.bfloat16) * 0.01).contiguous()
        x_q = fte._fast_quantize_localcta_v4_opt(x, nvfp4_role="activation")
        group_result = fte._tk_group_quantize_ffn_weights(tkq, w1, w3, [H, H], prefer_split=True)
        dgrad_wc_fp4_cols = group_result[3]
        dgrad_wc_sc_cols = group_result[4]
        dgrad_wc_sg_cols = group_result[5]
        standard_dh1_q = fte._fast_quantize_localcta_v4_opt(dh1, nvfp4_role="grad")
        standard_dh3_q = fte._fast_quantize_localcta_v4_opt(dh3, nvfp4_role="grad")

        def _consumer_quant(payload, *, cat_payload: bool):
            if cat_payload:
                return {
                    "row_fp4s": [payload[0], payload[6]],
                    "row_scs": [payload[1], payload[7]],
                    "row_sgs": [payload[4], payload[10]],
                    "col_fp4s": [payload[2], payload[8]],
                    "col_scs": [payload[3], payload[9]],
                    "col_sgs": [payload[5], payload[11]],
                    "col_fp4_full": payload[16],
                    "col_sc_full": payload[17],
                    "col_sg_full": payload[18],
                }
            if id(payload) in outer_sgs:
                row0, row1, col0, col1, col_full = outer_sgs[id(payload)]
                row_sgs = [row0, row1]
                col_sgs = [col0, col1]
                col_sg_full = col_full
            else:
                row_sgs = payload[2]
                col_sgs = payload[5]
                col_sg_full = payload[11]
            return {
                "row_fp4s": payload[0],
                "row_scs": payload[1],
                "row_sgs": row_sgs,
                "col_fp4s": payload[3],
                "col_scs": payload[4],
                "col_sgs": col_sgs,
                "col_fp4_full": payload[9],
                "col_sc_full": payload[10],
                "col_sg_full": col_sg_full,
                "row_fp4_full": payload[6],
            }

        def _w13(payload, *, cat_payload: bool):
            quant = _consumer_quant(payload, cat_payload=cat_payload)
            if cat_payload or id(payload) in outer_sgs:
                col_sgs = quant["col_sgs"]
                col_sg_full = quant["col_sg_full"]
            elif args.production_finalize:
                _, _, col0, col1, col_full = outer_sgs[id(payload)]
                col_sgs = [col0, col1]
                col_sg_full = col_full
            else:
                col_sgs = quant["col_sgs"]
                col_sg_full = quant["col_sg_full"]
            return tk_grouped_wgrad_gemm(
                (
                    quant["col_fp4s"],
                    quant["col_scs"],
                    col_sgs,
                    quant["col_fp4_full"],
                    quant["col_sc_full"],
                    col_sg_full,
                ),
                x_q,
                [H, H],
        )

        def _w13_standard():
            grad_w1, grad_w3 = tk_split_wgrad_gemm(
                (
                    [standard_dh1_q._tk_col[0], standard_dh3_q._tk_col[0]],
                    [standard_dh1_q._tk_col[1], standard_dh3_q._tk_col[1]],
                    [standard_dh1_q._tk_col[2], standard_dh3_q._tk_col[2]],
                ),
                x_q,
                use_localcta=True,
            )
            return torch.cat([grad_w1, grad_w3], dim=0)

        print(f"[rank {rank}] consumer=w13_ref start", flush=True)
        # tk_grouped_wgrad_gemm may recycle an internal output allocation, so
        # retain the reference before launching the second consumer.
        if args.producer == "cat":
            w13_ref = _w13_standard()
        else:
            w13_ref = _w13(ref, cat_payload=False).clone()
        torch.cuda.synchronize()
        print(f"[rank {rank}] consumer=w13_ref done", flush=True)

        print(f"[rank {rank}] consumer=w13_fused start", flush=True)
        w13_fused = _w13(fused, cat_payload=args.producer == "cat")
        torch.cuda.synchronize()
        w13_abs, w13_rel, w13_cos = _rel_stats(w13_fused, w13_ref)
        w13_rms, w13_ref_rms, w13_norm_ratio, w13_zero, w13_ref_zero = _signal_stats(
            w13_fused, w13_ref
        )
        print(
            f"[rank {rank}] consumer=w13_fused done "
            f"abs={w13_abs:.3e} rel={w13_rel:.3e} cos={w13_cos:.6f} "
            f"rms={w13_rms:.3e} ref_rms={w13_ref_rms:.3e} "
            f"norm_ratio={w13_norm_ratio:.6f} zero={w13_zero:.6f} "
            f"ref_zero={w13_ref_zero:.6f}",
            flush=True,
        )
        if args.check_bf16_consumers:
            w13_bf16 = torch.cat(
                [dh1.transpose(0, 1) @ x, dh3.transpose(0, 1) @ x], dim=0
            )
            w13_bf16_abs, w13_bf16_rel, w13_bf16_cos = _rel_stats(
                w13_fused, w13_bf16
            )
            (
                w13_bf16_rms,
                w13_true_rms,
                w13_bf16_norm_ratio,
                w13_bf16_zero,
                w13_true_zero,
            ) = _signal_stats(w13_fused, w13_bf16)
            print(
                f"[rank {rank}] consumer=w13_fused_vs_bf16 "
                f"abs={w13_bf16_abs:.3e} rel={w13_bf16_rel:.3e} "
                f"cos={w13_bf16_cos:.6f} rms={w13_bf16_rms:.3e} "
                f"ref_rms={w13_true_rms:.3e} norm_ratio={w13_bf16_norm_ratio:.6f} "
                f"zero={w13_bf16_zero:.6f} ref_zero={w13_true_zero:.6f}",
                flush=True,
            )

        def _split2_dgrad(payload, *, cat_payload: bool):
            quant = _consumer_quant(payload, cat_payload=cat_payload)
            if args.producer == "cat":
                out = torch.empty(M, K, device=device, dtype=torch.bfloat16)
                out.zero_()
                tk.nvfp4_split2_dgrad_onepass_gemm(
                    [value.contiguous() for value in quant["row_fp4s"]],
                    [value.contiguous() for value in quant["row_scs"]],
                    [value.contiguous() for value in quant["row_sgs"]],
                    dgrad_wc_fp4_cols,
                    dgrad_wc_sc_cols,
                    list(dgrad_wc_sg_cols),
                    out,
                    fte.tk_localcta_v3_split2_onepass_config_idx(),
                )
                return out
            if args.production_finalize:
                row0, row1, _, _, _ = outer_sgs[id(payload)]
                out = torch.empty(M, K, device=device, dtype=torch.bfloat16)
                out.zero_()
                tk.nvfp4_split2_dgrad_strided_onepass_gemm_outer_sg(
                    payload[6],
                    payload[1],
                    [row0, row1],
                    [0, H // 2],
                    [H // 2, H // 2],
                    dgrad_wc_fp4_cols,
                    dgrad_wc_sc_cols,
                    list(dgrad_wc_sg_cols),
                    out,
                    -1,
                )
                return out
            row_sgs_for_dgrad = [
                _prepare_localcta_v4_chunkgrid_for_batched(
                    sg,
                    payload[6].size(0),
                    H,
                    payload[6].device,
                )
                for sg in payload[2]
            ]
            dgrad_wc_sg_for_dgrad = [
                _prepare_localcta_v4_chunkgrid_for_batched(
                    sg,
                    fp4.size(0),
                    fp4.size(1) * 2,
                    fp4.device,
                )
                for sg, fp4 in zip(dgrad_wc_sg_cols, dgrad_wc_fp4_cols)
            ]
            out = torch.empty(M, K, device=device, dtype=torch.bfloat16)
            out.zero_()
            tk.nvfp4_split2_dgrad_strided_onepass_gemm_sg(
                payload[6],
                payload[1],
                row_sgs_for_dgrad,
                [0, H // 2],
                [H // 2, H // 2],
                dgrad_wc_fp4_cols,
                dgrad_wc_sc_cols,
                dgrad_wc_sg_for_dgrad,
                out,
                -1,
            )
            return out

        def _split2_dgrad_standard():
            out = torch.empty(M, K, device=device, dtype=torch.bfloat16)
            out.zero_()
            tk.nvfp4_split2_dgrad_onepass_gemm(
                [standard_dh1_q._tk_row[0], standard_dh3_q._tk_row[0]],
                [standard_dh1_q._tk_row[1], standard_dh3_q._tk_row[1]],
                [standard_dh1_q._tk_row[2], standard_dh3_q._tk_row[2]],
                dgrad_wc_fp4_cols,
                dgrad_wc_sc_cols,
                list(dgrad_wc_sg_cols),
                out,
                fte.tk_localcta_v3_split2_onepass_config_idx(),
            )
            return out

        print(f"[rank {rank}] consumer=split2_dgrad_ref start", flush=True)
        if args.producer == "cat":
            dgrad_ref = _split2_dgrad_standard()
        else:
            dgrad_ref = _split2_dgrad(ref, cat_payload=False)
        torch.cuda.synchronize()
        print(f"[rank {rank}] consumer=split2_dgrad_ref done", flush=True)

        print(f"[rank {rank}] consumer=split2_dgrad_fused start", flush=True)
        dgrad_fused = _split2_dgrad(fused, cat_payload=args.producer == "cat")
        torch.cuda.synchronize()
        dgrad_abs, dgrad_rel, dgrad_cos = _rel_stats(dgrad_fused, dgrad_ref)
        dgrad_rms, dgrad_ref_rms, dgrad_norm_ratio, dgrad_zero, dgrad_ref_zero = _signal_stats(
            dgrad_fused, dgrad_ref
        )
        print(
            f"[rank {rank}] consumer=split2_dgrad_fused done "
            f"abs={dgrad_abs:.3e} rel={dgrad_rel:.3e} cos={dgrad_cos:.6f} "
            f"rms={dgrad_rms:.3e} ref_rms={dgrad_ref_rms:.3e} "
            f"norm_ratio={dgrad_norm_ratio:.6f} zero={dgrad_zero:.6f} "
            f"ref_zero={dgrad_ref_zero:.6f}",
            flush=True,
        )
        if args.check_bf16_consumers:
            dgrad_bf16 = dh1 @ w1 + dh3 @ w3
            dgrad_bf16_abs, dgrad_bf16_rel, dgrad_bf16_cos = _rel_stats(
                dgrad_fused, dgrad_bf16
            )
            (
                dgrad_bf16_rms,
                dgrad_true_rms,
                dgrad_bf16_norm_ratio,
                dgrad_bf16_zero,
                dgrad_true_zero,
            ) = _signal_stats(dgrad_fused, dgrad_bf16)
            print(
                f"[rank {rank}] consumer=split2_dgrad_fused_vs_bf16 "
                f"abs={dgrad_bf16_abs:.3e} rel={dgrad_bf16_rel:.3e} "
                f"cos={dgrad_bf16_cos:.6f} rms={dgrad_bf16_rms:.3e} "
                f"ref_rms={dgrad_true_rms:.3e} "
                f"norm_ratio={dgrad_bf16_norm_ratio:.6f} "
                f"zero={dgrad_bf16_zero:.6f} ref_zero={dgrad_true_zero:.6f}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
