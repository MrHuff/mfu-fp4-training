import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
QUANT_ROOT = ROOT.parents[4] / "TK_quantisation" / "nvfp4_CTA_local_v1"
LEGACY_GEMM_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_ROOT))
sys.path.insert(0, str(LEGACY_GEMM_ROOT))

import _C_nv_localcta_gemm as local_gemm  # type: ignore
import _C as legacy_gemm  # type: ignore
import _tk_quant_localcta as local_q  # type: ignore


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


def main() -> None:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    config_lo = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    config_hi = int(sys.argv[5]) if len(sys.argv) > 5 else 46

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    Aq = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
    Bq = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)

    def localcta_default() -> None:
        local_gemm.nvfp4_localcta_fast_gemm(Aq[0], Aq[1], Bq[0], Bq[1], D)

    unit = torch.ones(1, device="cuda", dtype=torch.float32)
    default_ms = bench(localcta_default)
    print(f"default_ms {default_ms:.3f}")

    best_cfg = None
    best_ms = None
    for config_id in range(config_lo, config_hi + 1):
        def run_cfg() -> None:
            legacy_gemm.nvfp4_gemm_config(Aq[0], Aq[1], unit, Bq[0], Bq[1], unit, D, config_id)

        try:
            t = bench(run_cfg)
        except RuntimeError as exc:
            print(f"config {config_id:02d} ERROR {exc}")
            continue
        print(f"config {config_id:02d} {t:.3f}")
        if best_ms is None or t < best_ms:
            best_ms = t
            best_cfg = config_id

    if best_cfg is not None and best_ms is not None:
        print(f"best_config {best_cfg} best_ms {best_ms:.3f}")
        print(f"speedup_vs_default {default_ms / best_ms:.4f}")


if __name__ == "__main__":
    main()
