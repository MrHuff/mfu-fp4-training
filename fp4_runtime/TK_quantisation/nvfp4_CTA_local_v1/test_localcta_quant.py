import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "nvfp4_v5"))

import _tk_quant_localcta as q_local  # type: ignore
import _tk_quant_v5 as q_v5  # type: ignore


def stats(name: str, x: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (x.float() - ref.float()).abs()
    print(f"{name}: max={diff.max().item():.6e} mean={diff.mean().item():.6e}")


def constant_chunk_grid(rows: int, cols: int, sg: torch.Tensor) -> torch.Tensor:
    return torch.full((rows // 128, cols // 128), sg.item(), device=sg.device, dtype=torch.float32)


def main() -> None:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)

    row_fp4_l, row_sc_l, col_fp4_l, col_sc_l, row_sg_l, col_sg_l = \
        q_local.tk_localcta_quantize_for_gemm(x, True, True)
    recon_l = q_local.tk_localcta_reconstruct_row(row_fp4_l, row_sc_l, row_sg_l)

    row_fp4_f, row_sc_f, col_fp4_f, col_sc_f, row_sg_f, col_sg_f, row_sc_prepared_f, col_sc_prepared_f = \
        q_local.tk_localcta_quantize_for_gemm_fast(x, True, True)
    recon_f = q_local.tk_localcta_reconstruct_row(row_fp4_f, row_sc_f, row_sg_f)
    row_fp4_p, row_sc_prepared_p, col_fp4_p, col_sc_prepared_p, row_sg_p, col_sg_p = \
        q_local.tk_localcta_quantize_for_gemm_prepared(x, True, True)

    row_fp4_v, row_sc_v, _, _, sg_v, _ = q_v5.tk_quantize_for_gemm(x, True, True)
    sg_grid_v = constant_chunk_grid(M, K, sg_v)
    recon_v = q_local.tk_localcta_reconstruct_row(row_fp4_v, row_sc_v, sg_grid_v)

    stats("localcta_direct reconstruction vs bf16", recon_l, x)
    stats("localcta_fast_debug reconstruction vs bf16", recon_f, x)
    stats("baseline_v5 reconstruction vs bf16", recon_v, x)
    stats("localcta_direct vs baseline_v5 reconstruction", recon_l, recon_v)
    stats("localcta_fast_debug vs localcta_direct reconstruction", recon_f, recon_l)

    print(f"row_sg_chunks: {tuple(row_sg_l.shape)}")
    print(f"col_sg_chunks: {tuple(col_sg_l.shape)}")
    print(f"transpose fp4:  {tuple(col_fp4_l.shape)}")
    print(f"transpose sc:   {tuple(col_sc_l.shape)}")
    print(f"prepared row sc: {tuple(row_sc_prepared_f.shape)}")
    print(f"prepared col sc: {tuple(col_sc_prepared_f.shape)}")
    print(f"prepared-only row sc: {tuple(row_sc_prepared_p.shape)}")
    print(f"prepared-only col sc: {tuple(col_sc_prepared_p.shape)}")


if __name__ == "__main__":
    main()
