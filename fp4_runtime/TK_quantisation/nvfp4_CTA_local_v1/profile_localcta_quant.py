import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "nvfp4_v5"))

import _tk_quant_localcta as q_local  # type: ignore
import _tk_quant_v5 as q_v5  # type: ignore


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "localcta_prepared"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    profile_region = "profile" in sys.argv[4:]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "profile" else 160
    pipe_depth = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "profile" else 2
    shared_amax = False
    if len(sys.argv) > 6 and sys.argv[6] != "profile":
        shared_amax = sys.argv[6].lower() in {"1", "true", "shared", "yes"}
    pre_sleep = float(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != "profile" else 0.0

    torch.manual_seed(0)
    if pre_sleep > 0:
        time.sleep(pre_sleep)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)

    def run_once():
        if mode == "localcta_direct":
            return q_local.tk_localcta_quantize_for_gemm(x, False, True)
        if mode == "localcta_fast_debug":
            return q_local.tk_localcta_quantize_for_gemm_fast(x, False, True)
        if mode == "localcta_prepared":
            return q_local.tk_localcta_quantize_for_gemm_prepared(x, False, True)
        if mode == "localcta1_prepared":
            q_local.tk_localcta_set_1cta_prepared_tuning(threads, pipe_depth)
            return q_local.tk_localcta_quantize_for_gemm_prepared(x, False, True)
        if mode == "localcta2_prepared":
            q_local.tk_localcta_set_2cta_prepared_tuning(threads, pipe_depth, shared_amax)
            return q_local.tk_localcta2_quantize_for_gemm_prepared(x, False, True)
        if mode == "baseline_v5":
            return q_v5.tk_quantize_for_gemm(x, False, True)
        raise ValueError(f"unknown mode: {mode}")

    if profile_region:
        run_once()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        out = run_once()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        out = run_once()

    torch.cuda.synchronize()
    print(mode, tuple(out[0].shape), tuple(out[1].shape),
          {"threads": threads, "pipe_depth": pipe_depth, "shared_amax": shared_amax, "pre_sleep": pre_sleep})


if __name__ == "__main__":
    main()
