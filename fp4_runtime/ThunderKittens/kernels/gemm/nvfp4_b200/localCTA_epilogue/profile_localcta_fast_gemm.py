import sys
import time
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


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "localcta_prepared"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    config_id = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "profile" else -1
    pre_sleep = float(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "profile" else 0.0
    profile_region = "profile" in sys.argv[5:]

    torch.manual_seed(0)
    if pre_sleep > 0:
        time.sleep(pre_sleep)

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    if mode == "localcta_prepared":
        Aq = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
        Bq = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)

        def run_once() -> None:
            local_gemm.nvfp4_localcta_fast_gemm(Aq[0], Aq[1], Bq[0], Bq[1], D)

    elif mode == "baseline_v5":
        Aq = q_v5.tk_quantize_for_gemm(A, False, True)
        Bq = q_v5.tk_quantize_for_gemm(B, False, True)

        def run_once() -> None:
            legacy_gemm.nvfp4_gemm(Aq[0], Aq[1], Aq[4], Bq[0], Bq[1], Bq[4], D)

    elif mode == "localcta_prepared_config":
        if config_id < 0:
            raise ValueError("config_id must be >= 0 for localcta_prepared_config")
        Aq = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
        Bq = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)
        unit = torch.ones(1, device="cuda", dtype=torch.float32)

        def run_once() -> None:
            legacy_gemm.nvfp4_gemm_config(Aq[0], Aq[1], unit, Bq[0], Bq[1], unit, D, config_id)

    else:
        raise ValueError(f"unknown mode: {mode}")

    run_once()
    torch.cuda.synchronize()

    if profile_region:
        torch.cuda.cudart().cudaProfilerStart()
        run_once()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        run_once()
        torch.cuda.synchronize()

    print(mode, M, N, K, {"config_id": config_id, "pre_sleep": pre_sleep})


if __name__ == "__main__":
    main()
