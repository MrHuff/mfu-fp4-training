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


def parse_splits(arg: str | None) -> list[int]:
    if not arg:
        return [2048, 2048, 2048]
    return [int(part) for part in arg.split(",") if part]


def main() -> None:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    splits = parse_splits(sys.argv[3] if len(sys.argv) > 3 else None)
    N_total = sum(splits)

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    W = torch.randn(N_total, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)

    D_direct = torch.empty(M, N_total, dtype=torch.bfloat16, device="cuda")
    D_prepared = torch.empty(M, N_total, dtype=torch.bfloat16, device="cuda")
    D_v5 = torch.empty(M, N_total, dtype=torch.bfloat16, device="cuda")

    def direct_quant():
        local_q.tk_localcta_quantize_for_gemm(A, False, True)
        local_q.tk_localcta_group_quantize_for_gemm(W, splits)

    def prepared_quant():
        local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
        local_q.tk_localcta_group_quantize_for_gemm_prepared(W, splits)

    def baseline_quant():
        q_v5.tk_quantize_for_gemm(A, False, True)
        q_v5.tk_group_quantize_for_gemm(W, splits)

    t_direct_q = bench(direct_quant)
    t_prepared_q = bench(prepared_quant)
    t_v5_q = bench(baseline_quant)

    A_direct = local_q.tk_localcta_quantize_for_gemm(A, False, True)
    W_direct = local_q.tk_localcta_group_quantize_for_gemm(W, splits)
    A_prepared = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
    W_prepared = local_q.tk_localcta_group_quantize_for_gemm_prepared(W, splits)
    A_v5 = q_v5.tk_quantize_for_gemm(A, False, True)
    W_v5 = q_v5.tk_group_quantize_for_gemm(W, splits)

    t_direct_gemm = bench(lambda: local_gemm.nvfp4_localcta_grouped_gemm(
        A_direct[0], A_direct[1], A_direct[4],
        W_direct[0], W_direct[1], W_direct[2],
        D_direct,
    ))
    t_prepared_gemm = bench(lambda: local_gemm.nvfp4_localcta_fast_grouped_gemm(
        A_prepared[0], A_prepared[1],
        W_prepared[0], W_prepared[1],
        D_prepared,
    ))
    t_v5_gemm = bench(lambda: legacy_gemm.nvfp4_grouped_gemm(
        A_v5[0], A_v5[1], A_v5[4],
        W_v5[0], W_v5[1], W_v5[2],
        D_v5,
    ))

    def direct_e2e():
        Aq = local_q.tk_localcta_quantize_for_gemm(A, False, True)
        Wq = local_q.tk_localcta_group_quantize_for_gemm(W, splits)
        local_gemm.nvfp4_localcta_grouped_gemm(
            Aq[0], Aq[1], Aq[4],
            Wq[0], Wq[1], Wq[2],
            D_direct,
        )

    def prepared_e2e():
        Aq = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
        Wq = local_q.tk_localcta_group_quantize_for_gemm_prepared(W, splits)
        local_gemm.nvfp4_localcta_fast_grouped_gemm(
            Aq[0], Aq[1],
            Wq[0], Wq[1],
            D_prepared,
        )

    def v5_e2e():
        Aq = q_v5.tk_quantize_for_gemm(A, False, True)
        Wq = q_v5.tk_group_quantize_for_gemm(W, splits)
        legacy_gemm.nvfp4_grouped_gemm(
            Aq[0], Aq[1], Aq[4],
            Wq[0], Wq[1], Wq[2],
            D_v5,
        )

    t_direct_e2e = bench(direct_e2e, warmup=2, iters=5)
    t_prepared_e2e = bench(prepared_e2e, warmup=2, iters=5)
    t_v5_e2e = bench(v5_e2e, warmup=2, iters=5)

    print(f"grouped_splits {splits}")
    print(f"quant_only_ms localcta_direct={t_direct_q:.3f} localcta_prepared={t_prepared_q:.3f} baseline_v5={t_v5_q:.3f}")
    print(f"gemm_only_ms localcta_direct={t_direct_gemm:.3f} localcta_prepared={t_prepared_gemm:.3f} baseline_v5={t_v5_gemm:.3f}")
    print(f"e2e_ms localcta_direct={t_direct_e2e:.3f} localcta_prepared={t_prepared_e2e:.3f} baseline_v5={t_v5_e2e:.3f}")


if __name__ == "__main__":
    main()
