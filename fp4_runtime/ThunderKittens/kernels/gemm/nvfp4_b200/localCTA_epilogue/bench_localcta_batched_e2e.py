import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
QUANT_ROOT = ROOT.parents[4] / "TK_quantisation" / "nvfp4_CTA_local_v1"
V5_ROOT = ROOT.parents[4] / "TK_quantisation" / "nvfp4_v5"
LEGACY_GEMM_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_ROOT))
sys.path.insert(0, str(V5_ROOT))
sys.path.insert(0, str(LEGACY_GEMM_ROOT))

import _C_nv_localcta_gemm as local_gemm  # type: ignore
import _C as legacy_gemm  # type: ignore
import _tk_quant_localcta as local_q  # type: ignore
import _tk_quant_v5 as q_v5  # type: ignore


def bench(fn, warmup=5, iters=20) -> float:
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
    return start.elapsed_time(end) / iters


def build_batch(M: int, N: int, K: int, batches: int):
    torch.manual_seed(0)
    A_list = [torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25) for _ in range(batches)]
    B_list = [torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25) for _ in range(batches)]
    return A_list, B_list


def main() -> None:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    batches = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    A_list, B_list = build_batch(M, N, K, batches)
    D_direct = [torch.empty(M, N, dtype=torch.bfloat16, device="cuda") for _ in range(batches)]
    D_prepared = [torch.empty(M, N, dtype=torch.bfloat16, device="cuda") for _ in range(batches)]
    D_v5 = [torch.empty(M, N, dtype=torch.bfloat16, device="cuda") for _ in range(batches)]
    D_direct_accum = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_prepared_accum = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_v5_accum = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def direct_quant():
        for A, B in zip(A_list, B_list):
            local_q.tk_localcta_quantize_for_gemm(A, False, True)
            local_q.tk_localcta_quantize_for_gemm(B, False, True)

    def prepared_quant():
        for A, B in zip(A_list, B_list):
            local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
            local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)

    def baseline_quant():
        for A, B in zip(A_list, B_list):
            q_v5.tk_quantize_for_gemm(A, False, True)
            q_v5.tk_quantize_for_gemm(B, False, True)

    t_direct_q = bench(direct_quant)
    t_prepared_q = bench(prepared_quant)
    t_v5_q = bench(baseline_quant)

    A_direct = [local_q.tk_localcta_quantize_for_gemm(A, False, True) for A in A_list]
    B_direct = [local_q.tk_localcta_quantize_for_gemm(B, False, True) for B in B_list]
    A_prepared = [local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True) for A in A_list]
    B_prepared = [local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True) for B in B_list]
    A_v5 = [q_v5.tk_quantize_for_gemm(A, False, True) for A in A_list]
    B_v5 = [q_v5.tk_quantize_for_gemm(B, False, True) for B in B_list]

    t_direct_gemm = bench(lambda: local_gemm.nvfp4_localcta_batched_gemm(
        [x[0] for x in A_direct], [x[1] for x in A_direct], [x[4] for x in A_direct],
        [x[0] for x in B_direct], [x[1] for x in B_direct], [x[4] for x in B_direct],
        D_direct,
    ))
    t_prepared_gemm = bench(lambda: local_gemm.nvfp4_localcta_fast_batched_gemm(
        [x[0] for x in A_prepared], [x[1] for x in A_prepared],
        [x[0] for x in B_prepared], [x[1] for x in B_prepared],
        D_prepared,
    ))
    t_v5_gemm = bench(lambda: legacy_gemm.nvfp4_batched_gemm(
        [x[0] for x in A_v5], [x[1] for x in A_v5], [x[4] for x in A_v5],
        [x[0] for x in B_v5], [x[1] for x in B_v5], [x[4] for x in B_v5],
        D_v5,
    ))

    t_direct_accum = bench(lambda: local_gemm.nvfp4_localcta_batched_accum_gemm(
        [x[0] for x in A_direct], [x[1] for x in A_direct], [x[4] for x in A_direct],
        [x[0] for x in B_direct], [x[1] for x in B_direct], [x[4] for x in B_direct],
        D_direct_accum,
    ))
    t_prepared_accum = bench(lambda: local_gemm.nvfp4_localcta_fast_batched_accum_gemm(
        [x[0] for x in A_prepared], [x[1] for x in A_prepared],
        [x[0] for x in B_prepared], [x[1] for x in B_prepared],
        D_prepared_accum,
    ))
    t_v5_accum = bench(lambda: legacy_gemm.nvfp4_batched_accum_gemm(
        [x[0] for x in A_v5], [x[1] for x in A_v5], [x[4] for x in A_v5],
        [x[0] for x in B_v5], [x[1] for x in B_v5], [x[4] for x in B_v5],
        D_v5_accum,
    ))

    def direct_e2e():
        Aq = [local_q.tk_localcta_quantize_for_gemm(A, False, True) for A in A_list]
        Bq = [local_q.tk_localcta_quantize_for_gemm(B, False, True) for B in B_list]
        local_gemm.nvfp4_localcta_batched_gemm(
            [x[0] for x in Aq], [x[1] for x in Aq], [x[4] for x in Aq],
            [x[0] for x in Bq], [x[1] for x in Bq], [x[4] for x in Bq],
            D_direct,
        )

    def prepared_e2e():
        Aq = [local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True) for A in A_list]
        Bq = [local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True) for B in B_list]
        local_gemm.nvfp4_localcta_fast_batched_gemm(
            [x[0] for x in Aq], [x[1] for x in Aq],
            [x[0] for x in Bq], [x[1] for x in Bq],
            D_prepared,
        )

    def baseline_e2e():
        Aq = [q_v5.tk_quantize_for_gemm(A, False, True) for A in A_list]
        Bq = [q_v5.tk_quantize_for_gemm(B, False, True) for B in B_list]
        legacy_gemm.nvfp4_batched_gemm(
            [x[0] for x in Aq], [x[1] for x in Aq], [x[4] for x in Aq],
            [x[0] for x in Bq], [x[1] for x in Bq], [x[4] for x in Bq],
            D_v5,
        )

    def direct_accum_e2e():
        Aq = [local_q.tk_localcta_quantize_for_gemm(A, False, True) for A in A_list]
        Bq = [local_q.tk_localcta_quantize_for_gemm(B, False, True) for B in B_list]
        local_gemm.nvfp4_localcta_batched_accum_gemm(
            [x[0] for x in Aq], [x[1] for x in Aq], [x[4] for x in Aq],
            [x[0] for x in Bq], [x[1] for x in Bq], [x[4] for x in Bq],
            D_direct_accum,
        )

    def prepared_accum_e2e():
        Aq = [local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True) for A in A_list]
        Bq = [local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True) for B in B_list]
        local_gemm.nvfp4_localcta_fast_batched_accum_gemm(
            [x[0] for x in Aq], [x[1] for x in Aq],
            [x[0] for x in Bq], [x[1] for x in Bq],
            D_prepared_accum,
        )

    def baseline_accum_e2e():
        Aq = [q_v5.tk_quantize_for_gemm(A, False, True) for A in A_list]
        Bq = [q_v5.tk_quantize_for_gemm(B, False, True) for B in B_list]
        legacy_gemm.nvfp4_batched_accum_gemm(
            [x[0] for x in Aq], [x[1] for x in Aq], [x[4] for x in Aq],
            [x[0] for x in Bq], [x[1] for x in Bq], [x[4] for x in Bq],
            D_v5_accum,
        )

    t_direct_e2e = bench(direct_e2e, warmup=2, iters=5)
    t_prepared_e2e = bench(prepared_e2e, warmup=2, iters=5)
    t_v5_e2e = bench(baseline_e2e, warmup=2, iters=5)

    t_direct_accum_e2e = bench(direct_accum_e2e, warmup=2, iters=5)
    t_prepared_accum_e2e = bench(prepared_accum_e2e, warmup=2, iters=5)
    t_v5_accum_e2e = bench(baseline_accum_e2e, warmup=2, iters=5)

    print(f"batches {batches}")
    print(f"quant_only_ms localcta_direct={t_direct_q:.3f} localcta_prepared={t_prepared_q:.3f} baseline_v5={t_v5_q:.3f}")
    print(f"batched_gemm_only_ms localcta_direct={t_direct_gemm:.3f} localcta_prepared={t_prepared_gemm:.3f} baseline_v5={t_v5_gemm:.3f}")
    print(f"batched_accum_only_ms localcta_direct={t_direct_accum:.3f} localcta_prepared={t_prepared_accum:.3f} baseline_v5={t_v5_accum:.3f}")
    print(f"batched_e2e_ms localcta_direct={t_direct_e2e:.3f} localcta_prepared={t_prepared_e2e:.3f} baseline_v5={t_v5_e2e:.3f}")
    print(f"batched_accum_e2e_ms localcta_direct={t_direct_accum_e2e:.3f} localcta_prepared={t_prepared_accum_e2e:.3f} baseline_v5={t_v5_accum_e2e:.3f}")


if __name__ == "__main__":
    main()
