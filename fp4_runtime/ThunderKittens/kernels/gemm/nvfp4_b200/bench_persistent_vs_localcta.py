"""
Apples-to-apples localCTA benchmark.

All curves stay within the localCTA quantization/GEMM family:

1. Separate direct localCTA:
   tk_localcta_quantize_for_gemm_launch(...) + nvfp4_localcta_gemm(...)
2. Optimized separate prepared localCTA:
   tk_localcta_quantize_for_gemm_prepared_launch(...) + nvfp4_localcta_fast_gemm(...)
3. Persistent 1CTA localCTA quant:
   tk_localcta_quantize_for_gemm_fast_launch(...) + nvfp4_localcta_fast_gemm(...)
4. Forced 2CTA prepared localCTA:
   tk_localcta2_quantize_for_gemm_prepared_launch(...) + nvfp4_localcta_fast_gemm(...)

The benchmark uses preallocated outputs so we time kernel work rather than
Python/Torch allocation overhead.
"""

import sys
from pathlib import Path

import torch

torch.random.manual_seed(42)

ROOT = Path(__file__).resolve().parent
LOCALCTA_GEMM = ROOT / "localCTA_epilogue"
LOCALCTA_QUANT = ROOT.parents[3] / "TK_quantisation" / "nvfp4_CTA_local_v1"

sys.path.insert(0, str(LOCALCTA_GEMM))
sys.path.insert(0, str(LOCALCTA_QUANT))

try:
    import _C_nv_localcta_gemm as local_gemm  # type: ignore
    import _tk_quant_localcta as local_q  # type: ignore
except ImportError as exc:
    raise RuntimeError(
        "Missing localCTA extension modules.\n"
        f"  Quant path: {LOCALCTA_QUANT}\n"
        f"  GEMM path:  {LOCALCTA_GEMM}\n"
        "Build commands:\n"
        f"  make -C {LOCALCTA_QUANT}\n"
        f"  make -C {LOCALCTA_GEMM}"
    ) from exc


def bench(fn, warmup=10, iters=20) -> float:
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


def prepared_quant_mode(M: int, K: int) -> str:
    blocks_y = M // 128
    blocks_x = K // 128
    macro_tiles_y = (blocks_y + 1) // 2
    total_macro_tiles = blocks_x * macro_tiles_y
    if total_macro_tiles > 0 and (blocks_x <= 16 or total_macro_tiles <= 1152):
        return "prepared_2cta_tuned"
    return "prepared_1cta_tuned"


def alloc_prepared_outputs(M: int, K: int):
    return local_q.tk_localcta_quantize_for_gemm_prepared_alloc(
        M, K, False, torch.device("cuda")
    )


def alloc_direct_outputs(M: int, K: int):
    return local_q.tk_localcta_quantize_for_gemm_alloc(
        M, K, False, torch.device("cuda")
    )


def alloc_persistent_outputs(M: int, K: int):
    return local_q.tk_localcta_quantize_for_gemm_fast_alloc(
        M, K, False, torch.device("cuda")
    )


def alloc_forced_2cta_outputs(M: int, K: int):
    return local_q.tk_localcta2_quantize_for_gemm_prepared_alloc(
        M, K, False, torch.device("cuda")
    )


def launch_prepared_quant(x: torch.Tensor, outputs) -> None:
    local_q.tk_localcta_quantize_for_gemm_prepared_launch(
        x,
        False,
        True,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
    )


def launch_direct_quant(x: torch.Tensor, outputs) -> None:
    local_q.tk_localcta_quantize_for_gemm_launch(
        x,
        False,
        True,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
    )


def launch_persistent_quant(x: torch.Tensor, outputs) -> None:
    local_q.tk_localcta_quantize_for_gemm_fast_launch(
        x,
        False,
        True,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
    )


def launch_forced_2cta_quant(x: torch.Tensor, outputs) -> None:
    local_q.tk_localcta2_quantize_for_gemm_prepared_launch(
        x,
        False,
        True,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
    )


def run_fast_gemm(A_fp4, A_sc_prepared, B_fp4, B_sc_prepared, D) -> None:
    local_gemm.nvfp4_localcta_fast_gemm(A_fp4, A_sc_prepared, B_fp4, B_sc_prepared, D)


def run_direct_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D) -> None:
    local_gemm.nvfp4_localcta_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D)


def run_case(M: int, N: int, K: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"  M={M}, N={N}, K={K}")
    print(f"{'=' * 72}")
    print(f"    Optimized quant mode:  {prepared_quant_mode(M, K)}")
    print("    Comparison curves: separate_direct_localcta, optimized_auto_prepared, persistent_1cta_fast, forced_2cta_prepared")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)

    A_direct = alloc_direct_outputs(M, K)
    B_direct = alloc_direct_outputs(N, K)
    A_opt = alloc_prepared_outputs(M, K)
    B_opt = alloc_prepared_outputs(N, K)
    A_pers = alloc_persistent_outputs(M, K)
    B_pers = alloc_persistent_outputs(N, K)
    A_forced2 = alloc_forced_2cta_outputs(M, K)
    B_forced2 = alloc_forced_2cta_outputs(N, K)

    D_direct = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_opt = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_pers = torch.empty_like(D_opt)
    D_forced2 = torch.empty_like(D_opt)

    # The raw+prepared 2CTA fast path shares a cluster amax, so we exclude it
    # from this benchmark and compare against the numerically valid prepared-only 2CTA path.

    def direct_quant() -> None:
        launch_direct_quant(A, A_direct)
        launch_direct_quant(B, B_direct)

    def opt_quant() -> None:
        launch_prepared_quant(A, A_opt)
        launch_prepared_quant(B, B_opt)

    def pers_quant() -> None:
        launch_persistent_quant(A, A_pers)
        launch_persistent_quant(B, B_pers)

    def forced2_quant() -> None:
        launch_forced_2cta_quant(A, A_forced2)
        launch_forced_2cta_quant(B, B_forced2)

    def direct_gemm() -> None:
        run_direct_gemm(A_direct[0], A_direct[1], A_direct[4], B_direct[0], B_direct[1], B_direct[4], D_direct)

    def opt_gemm() -> None:
        run_fast_gemm(A_opt[0], A_opt[1], B_opt[0], B_opt[1], D_opt)

    def pers_gemm() -> None:
        run_fast_gemm(A_pers[0], A_pers[6], B_pers[0], B_pers[6], D_pers)

    def forced2_gemm() -> None:
        run_fast_gemm(A_forced2[0], A_forced2[1], B_forced2[0], B_forced2[1], D_forced2)

    def direct_e2e() -> None:
        launch_direct_quant(A, A_direct)
        launch_direct_quant(B, B_direct)
        run_direct_gemm(A_direct[0], A_direct[1], A_direct[4], B_direct[0], B_direct[1], B_direct[4], D_direct)

    def opt_e2e() -> None:
        launch_prepared_quant(A, A_opt)
        launch_prepared_quant(B, B_opt)
        run_fast_gemm(A_opt[0], A_opt[1], B_opt[0], B_opt[1], D_opt)

    def pers_e2e() -> None:
        launch_persistent_quant(A, A_pers)
        launch_persistent_quant(B, B_pers)
        run_fast_gemm(A_pers[0], A_pers[6], B_pers[0], B_pers[6], D_pers)

    def forced2_e2e() -> None:
        launch_forced_2cta_quant(A, A_forced2)
        launch_forced_2cta_quant(B, B_forced2)
        run_fast_gemm(A_forced2[0], A_forced2[1], B_forced2[0], B_forced2[1], D_forced2)

    ms_direct_q = bench(direct_quant)
    ms_opt_q = bench(opt_quant)
    ms_pers_q = bench(pers_quant)
    ms_forced2_q = bench(forced2_quant)

    direct_quant()
    opt_quant()
    pers_quant()
    forced2_quant()

    ms_direct_g = bench(direct_gemm)
    ms_opt_g = bench(opt_gemm)
    ms_pers_g = bench(pers_gemm)
    ms_forced2_g = bench(forced2_gemm)
    ms_direct = bench(direct_e2e)
    ms_opt = bench(opt_e2e)
    ms_pers = bench(pers_e2e)
    ms_forced2 = bench(forced2_e2e)

    direct_e2e()
    opt_e2e()
    pers_e2e()
    forced2_e2e()
    torch.cuda.synchronize()

    flops = 2.0 * M * N * K
    direct_tflops = flops / (ms_direct * 1e-3) / 1e12
    opt_tflops = flops / (ms_opt * 1e-3) / 1e12
    pers_tflops = flops / (ms_pers * 1e-3) / 1e12
    forced2_tflops = flops / (ms_forced2 * 1e-3) / 1e12

    cosine_direct = torch.nn.functional.cosine_similarity(
        D_direct.flatten().float().unsqueeze(0),
        D_pers.flatten().float().unsqueeze(0),
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        D_opt.flatten().float().unsqueeze(0),
        D_pers.flatten().float().unsqueeze(0),
    ).item()
    cosine_forced2 = torch.nn.functional.cosine_similarity(
        D_opt.flatten().float().unsqueeze(0),
        D_forced2.flatten().float().unsqueeze(0),
    ).item()
    max_abs_direct = (D_direct.float() - D_pers.float()).abs().max().item()
    max_abs = (D_opt.float() - D_pers.float()).abs().max().item()
    max_abs_forced2 = (D_opt.float() - D_forced2.float()).abs().max().item()

    print(
        f"    localCTA direct:      {ms_direct:.4f} ms  ({direct_tflops:.0f} TFLOPs)"
        f"  [Q={ms_direct_q:.4f} G={ms_direct_g:.4f}]"
    )
    print(
        f"    localCTA optimized:   {ms_opt:.4f} ms  ({opt_tflops:.0f} TFLOPs)"
        f"  [Q={ms_opt_q:.4f} G={ms_opt_g:.4f}]"
    )
    print(
        f"    localCTA persistent:  {ms_pers:.4f} ms  ({pers_tflops:.0f} TFLOPs)"
        f"  [Q={ms_pers_q:.4f} G={ms_pers_g:.4f}]"
    )
    print(
        f"    localCTA forced-2cta: {ms_forced2:.4f} ms  ({forced2_tflops:.0f} TFLOPs)"
        f"  [Q={ms_forced2_q:.4f} G={ms_forced2_g:.4f}]"
    )

    delta_direct = ms_direct - ms_pers
    if delta_direct > 0:
        print(
            f"    -> persistent_1cta_fast beats separate_direct_localcta by {delta_direct:.4f} ms "
            f"({delta_direct / ms_direct * 100:.1f}%)"
        )
    else:
        print(
            f"    -> separate_direct_localcta beats persistent_1cta_fast by {-delta_direct:.4f} ms "
            f"({-delta_direct / ms_pers * 100:.1f}%)"
        )

    best_persistent_label = "persistent_1cta_fast"
    best_persistent_ms = ms_pers
    if ms_forced2 < best_persistent_ms:
        best_persistent_label = "forced_2cta_prepared"
        best_persistent_ms = ms_forced2

    delta = best_persistent_ms - ms_opt
    if delta > 0:
        print(
            f"    -> Optimized wins vs best persistent ({best_persistent_label}) by {delta:.4f} ms "
            f"({delta / best_persistent_ms * 100:.1f}%)"
        )
    else:
        print(
            f"    -> Best persistent ({best_persistent_label}) wins by {-delta:.4f} ms "
            f"({-delta / ms_opt * 100:.1f}%)"
        )
    delta_forced2 = ms_pers - ms_forced2
    if delta_forced2 > 0:
        print(
            f"    -> forced_2cta_prepared beats persistent_1cta_fast by {delta_forced2:.4f} ms "
            f"({delta_forced2 / ms_pers * 100:.1f}%)"
        )
    else:
        print(
            f"    -> persistent_1cta_fast beats forced_2cta_prepared by {-delta_forced2:.4f} ms "
            f"({-delta_forced2 / ms_forced2 * 100:.1f}%)"
        )
    print(f"    Cos(direct vs pers1): {cosine_direct:.6f}")
    print(f"    Cos(opt vs pers1): {cosine:.6f}")
    print(f"    Cos(opt vs forced2): {cosine_forced2:.6f}")
    print(f"    MaxAbs(direct vs pers1): {max_abs_direct:.6f}")
    print(f"    MaxAbs(opt vs pers1): {max_abs:.6f}")
    print(f"    MaxAbs(opt vs forced2): {max_abs_forced2:.6f}")


def run_suite(title: str, cases) -> None:
    print(f"\n*** {title} ***")
    for M, N, K in cases:
        try:
            run_case(M, N, K)
        except Exception as exc:
            print(f"  M={M}, N={N}, K={K}: FAILED: {exc}")


def main() -> None:
    if len(sys.argv) == 4:
        run_case(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        return

    run_suite(
        "QKV Forward shapes",
        [(M, 6144, 2048) for M in [2048, 4096, 8192, 16384, 32768, 65536]],
    )
    run_suite(
        "QKV Dgrad shapes",
        [(M, 2048, 6144) for M in [2048, 4096, 8192, 16384, 32768]],
    )
    run_suite("Square shapes", [(s, s, s) for s in [4096, 8192]])


if __name__ == "__main__":
    main()
