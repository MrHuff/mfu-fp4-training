import torch
import torch.nn as nn
import sys
import os

# Add local path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "tests"))

from low_bits_training.quantization.debug_fused_quant import debug_fake_quant_simultaneous
from low_bits_training.quantization.fused_quant_triton_v2 import TritonFusedQuantLinear
from debug_llama_parity import MockCfg, debug_ref_wrapper

# Import Transformer Engine QGEMM
try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.experimental.quantization import GEMMType
    from transformer_engine.pytorch.experimental.quantization_custom_triton import (
        QuantizationMetadata,
        get_triton_quantizer_factory,
    )

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("WARNING: Transformer Engine not found, QGEMM ablation might fail.")


def expand_scales(s, block_size, target_shape, axis=1):
    # s shape: (M, K//B) or similar
    # Expand to (M, K)
    # If axis=1: repeat_interleave dim 1
    return s.repeat_interleave(block_size, dim=axis)


def ablate_divergence():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 1. Setup Simulation (Same as debug_llama_parity Step 0)
    B, S, H = 2, 128, 512  # Small model
    M = B * S
    K = H
    N = H

    input_data = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    weight_data = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02

    # Reference (Golden)
    print("--- Running Reference ---")
    ref_out = debug_ref_wrapper([(input_data, weight_data)], use_dequant_gemm=True)
    ref_y = ref_out[0].detach()  # (M, N)

    # Fused (Baseline)
    print("--- Running Fused Quant Extraction ---")
    # We use debug_fake_quant_simultaneous to get intermediates
    # It acts on X (M, K) and W.t() (K, N)

    # Prepare inputs for debug kernel
    x_in = input_data.to(torch.float32)
    w_t = weight_data.t().to(torch.float32).contiguous()  # (K, N)

    # Global AMAX (Simulated)
    ga_x = x_in.abs().max().view(1)
    ga_w = w_t.abs().max().view(1)

    # Call Debug Kernel
    # returns: out_a, out_b, a_q, b_q, s_a, s_b
    # a_q is (M, K), s_a is (M, K//B)
    # b_q is (K, N), s_b is (K//B, N) - Wait, check debug_fused_quant logic for s_b layout
    # In debug_fused_quant: s_b = torch.empty((num_blocks_k, N)) -> (K//B, N)

    out_a, out_b, a_q, b_q, s_a, s_b = debug_fake_quant_simultaneous(
        x_in,
        w_t,
        use_global_scale=True,
        scale_max_a=448.0,
        scale_max_b=448.0,
        ga_a=ga_x,
        ga_b=ga_w,
        scale_type="E4M3",
        block_size=16,
        return_encoded=True,  # Get Integers!
    )

    # Calculate Global Alpha
    DATA_MAX = 6.0  # E2M1
    SCALE_MAX = 448.0
    factor = (SCALE_MAX * DATA_MAX) ** 2
    alpha = (ga_x * ga_w) / factor

    print("\n=== ABLATION 1: PyTorch Dequantization + Float MM ===")

    # Dequantize A
    # s_a is (M, K/B). Expand to (M, K)
    s_a_exp = s_a.repeat_interleave(16, dim=1)
    a_dq_py = a_q * s_a_exp

    # Dequantize B
    # s_b is (K/B, N). Expand to (K, N)
    s_b_exp = s_b.repeat_interleave(16, dim=0)
    b_dq_py = b_q * s_b_exp

    # Matmul
    y_py = torch.mm(a_dq_py, b_dq_py)

    # Apply Global Scale
    y_py = y_py * alpha

    # Verify vs Reference
    diff_py = (y_py.to(torch.bfloat16) - ref_y).abs().max()
    print(f"PyTorch Dequant Diff: {diff_py:.4e}")
    if diff_py < 1e-5:
        print(">> PASS: PyTorch Dequant matches Reference.")
    else:
        print(">> FAIL: PyTorch Dequant diverges.")

    print("\n=== ABLATION 2: QGEMM ===")
    # Need factory for QGEMM? Or call kernel directly?
    # te.quantization.qgemm is exposed? No, it's usually on the quantizer object.

    if TE_AVAILABLE:
        # Create a helper quantizer to access qgemm
        factory = get_triton_quantizer_factory(scale_format="E4M3", block_size=16)
        dummy_q = factory("input")

        # Prepare Metadata
        meta_x = QuantizationMetadata(ga_x, ga_x)
        meta_w = QuantizationMetadata(ga_w, ga_w)

        # QGEMM arguments
        # qx: (M, K), sx: (M, K/B)
        # qw: (K, N), sw: (K/B, N) -> Need Transpose for QGEMM?
        # TE QGEMM usually expects [N, K] for weight if transposition handled?
        # Let's check signature.
        # qgemm(qx, qw, sx, sw, ...)
        # Typically qw is (N, K) packed?
        # If we use `debug_fake_quant`, we got `b_q` as (K, N).
        # We might need to transpose B for QGEMM if it expects W_T.

        # Try passing as is (torch.mm semantics) -> (M, K) @ (K, N)
        # But QGEMM usually assumes one operand transposed?
        # References say: qx (M, K), qw (N, K). Output (M, N).
        # Our `b_q` is (K, N). So we pass `b_q.t()` -> (N, K).
        # Scales for B: `s_b` is (K/B, N). Transpose -> (N, K/B).

        b_q_t = b_q.t().contiguous()
        s_b_t = s_b.t().contiguous()

        try:
            y_qgemm = dummy_q.qgemm(
                qx=a_q,
                qw=b_q_t,
                sx=s_a,
                sw=s_b_t,
                m_params=None,
                out_dtype=torch.float32,  # or bf16
                bias=None,
                qresult_x=meta_x,
                qresult_w=meta_w,
                accumulate_in_fp32=True,
            )

            # y_qgemm includes global scaling applied internally?
            # TEParityLinear QGEMM path passes metadata, so yes.

            diff_qgemm = (y_qgemm.to(torch.bfloat16) - ref_y).abs().max()
            print(f"QGEMM Diff: {diff_qgemm:.4e}")
            if diff_qgemm < 1e-5:
                print(">> PASS: QGEMM matches Reference.")
            else:
                print(">> FAIL: QGEMM diverges.")

        except Exception as e:
            print(f"QGEMM Execution Failed: {e}")

    else:
        print("SKIPPING Ablation 2 (TE not available)")


if __name__ == "__main__":
    try:
        ablate_divergence()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Ablation Crashed: {e}")
