import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
QUANT_ROOT = ROOT.parents[4] / "TK_quantisation" / "nvfp4_CTA_local_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_ROOT))

import _C_nv_localcta_gemm as local_gemm  # type: ignore
import _tk_quant_localcta as local_q  # type: ignore


def stats(name: str, x: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (x.float() - ref.float()).abs()
    print(f"{name}: max={diff.max().item():.6e} mean={diff.mean().item():.6e}")


def run_regular(M: int, N: int, K: int) -> None:
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    Aq = local_q.tk_localcta_quantize_for_gemm(A, False, True)
    Bq = local_q.tk_localcta_quantize_for_gemm(B, False, True)
    local_gemm.nvfp4_localcta_gemm(
        Aq[0], Aq[1], Aq[4],
        Bq[0], Bq[1], Bq[4],
        D,
    )
    D_fast = torch.empty_like(D)
    Aq_fast = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
    Bq_fast = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)
    local_gemm.nvfp4_localcta_fast_gemm(
        Aq_fast[0], Aq_fast[1],
        Bq_fast[0], Bq_fast[1],
        D_fast,
    )
    ref = torch.matmul(A, B.t()).to(torch.bfloat16)
    stats("regular localcta_direct GEMM vs bf16", D, ref)
    stats("regular localcta_prepared GEMM vs bf16", D_fast, ref)


def run_grouped(M: int, K: int, splits: list[int]) -> None:
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    W = torch.randn(sum(splits), K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    out = torch.empty(M, sum(splits), dtype=torch.bfloat16, device="cuda")

    Aq = local_q.tk_localcta_quantize_for_gemm(A, False, True)
    Wq = local_q.tk_localcta_group_quantize_for_gemm(W, splits)
    local_gemm.nvfp4_localcta_grouped_gemm(
        Aq[0], Aq[1], Aq[4],
        Wq[0], Wq[1], Wq[2],
        out,
    )
    out_fast = torch.empty_like(out)
    Aq_fast = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
    Wq_fast = local_q.tk_localcta_group_quantize_for_gemm_prepared(W, splits)
    local_gemm.nvfp4_localcta_fast_grouped_gemm(
        Aq_fast[0], Aq_fast[1],
        Wq_fast[0], Wq_fast[1],
        out_fast,
    )
    ref = torch.matmul(A, W.t()).to(torch.bfloat16)
    stats("grouped localcta_direct GEMM vs bf16", out, ref)
    stats("grouped localcta_prepared GEMM vs bf16", out_fast, ref)


def run_batched(M: int, N: int, K: int, batches: int) -> None:
    A_list, A_sc_list, A_sg_list = [], [], []
    B_list, B_sc_list, B_sg_list = [], [], []
    A_fast_list, A_sc_prepared_list = [], []
    B_fast_list, B_sc_prepared_list = [], []
    D_list = []
    D_fast_list = []
    refs = []
    for _ in range(batches):
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
        Aq = local_q.tk_localcta_quantize_for_gemm(A, False, True)
        Aq_fast = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
        Bq = local_q.tk_localcta_quantize_for_gemm(B, False, True)
        Bq_fast = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)
        A_list.append(Aq[0]); A_sc_list.append(Aq[1]); A_sg_list.append(Aq[4])
        A_fast_list.append(Aq_fast[0]); A_sc_prepared_list.append(Aq_fast[1])
        B_list.append(Bq[0]); B_sc_list.append(Bq[1]); B_sg_list.append(Bq[4])
        B_fast_list.append(Bq_fast[0]); B_sc_prepared_list.append(Bq_fast[1])
        D_list.append(torch.empty(M, N, dtype=torch.bfloat16, device="cuda"))
        D_fast_list.append(torch.empty(M, N, dtype=torch.bfloat16, device="cuda"))
        refs.append(torch.matmul(A, B.t()).to(torch.bfloat16))

    local_gemm.nvfp4_localcta_batched_gemm(
        A_list, A_sc_list, A_sg_list,
        B_list, B_sc_list, B_sg_list,
        D_list,
    )
    local_gemm.nvfp4_localcta_fast_batched_gemm(
        A_fast_list, A_sc_prepared_list,
        B_fast_list, B_sc_prepared_list,
        D_fast_list,
    )
    for i, (out, ref) in enumerate(zip(D_list, refs)):
        stats(f"batched localcta_direct GEMM vs bf16 [{i}]", out, ref)
    for i, (out, ref) in enumerate(zip(D_fast_list, refs)):
        stats(f"batched localcta_prepared GEMM vs bf16 [{i}]", out, ref)


def run_batched_accum(M: int, N: int, K: int, batches: int) -> None:
    A_list, A_sc_list, A_sg_list = [], [], []
    B_list, B_sc_list, B_sg_list = [], [], []
    A_fast_list, A_sc_prepared_list = [], []
    B_fast_list, B_sc_prepared_list = [], []
    ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    for _ in range(batches):
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
        Aq = local_q.tk_localcta_quantize_for_gemm(A, False, True)
        Aq_fast = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
        Bq = local_q.tk_localcta_quantize_for_gemm(B, False, True)
        Bq_fast = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)
        A_list.append(Aq[0]); A_sc_list.append(Aq[1]); A_sg_list.append(Aq[4])
        A_fast_list.append(Aq_fast[0]); A_sc_prepared_list.append(Aq_fast[1])
        B_list.append(Bq[0]); B_sc_list.append(Bq[1]); B_sg_list.append(Bq[4])
        B_fast_list.append(Bq_fast[0]); B_sc_prepared_list.append(Bq_fast[1])
        ref.add_(torch.matmul(A, B.t()).to(torch.bfloat16))

    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    local_gemm.nvfp4_localcta_batched_accum_gemm(
        A_list, A_sc_list, A_sg_list,
        B_list, B_sc_list, B_sg_list,
        out,
    )
    stats("batched_accum localcta_direct GEMM vs bf16", out, ref)

    out_fast = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    local_gemm.nvfp4_localcta_fast_batched_accum_gemm(
        A_fast_list, A_sc_prepared_list,
        B_fast_list, B_sc_prepared_list,
        out_fast,
    )
    stats("batched_accum localcta_prepared GEMM vs bf16", out_fast, ref)


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

    torch.manual_seed(0)
    run_regular(M, N, K)
    run_grouped(M, K, [N // 2, N // 2])
    run_batched(M, N, K, 2)
    run_batched_accum(M, N, K, 2)
