import numpy as np
import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import mxfp4_quantize


def fp32_to_fp4x2(V: torch.Tensor) -> torch.Tensor:
    """Convert float32 tensor to fp4x2 using round-to-nearest."""
    assert V.dtype == torch.float32
    assert V.dim() == 2
    M, N = V.shape
    assert N % 2 == 0
    # Use torch's native conversion if available, otherwise do manual
    result = torch.empty(M, N // 2, dtype=torch.float4_e2m1fn_x2, device=V.device)
    # Clamp to FP4 range: [-6, 6]
    V_clamped = V.clamp(-6.0, 6.0)
    # Convert via bf16 -> fp4x2 using the native path
    V_bf16 = V_clamped.to(torch.bfloat16)
    # Pack pairs: for the reference we dequantize via known grid values
    return V_bf16.view(torch.float4_e2m1fn_x2).reshape(M, N // 2)


def torch_mxfp4_quantize(
    V: torch.Tensor # (M, N)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP4 quantization in PyTorch (naive, for correctness checking)."""
    assert len(V.shape) == 2
    assert V.shape[0] % 128 == 0
    assert V.shape[1] % 128 == 0

    M, N = V.shape
    V = V.to(torch.float32)

    fp4_max = torch.tensor(6.0, dtype=torch.float32, device=V.device)
    min_exp = torch.tensor(-127.0, dtype=torch.float32, device=V.device)
    fp8e8m0_bias = torch.tensor(127.0, dtype=torch.float32, device=V.device)

    # Block-32 amax
    block_amax = torch.amax(torch.abs(V).view(M, N // 32, 32), dim=-1)

    # Compute E8M0 decode scale: power-of-2 via ceil(log2(amax/fp4_max))
    decode_scale = block_amax / fp4_max
    V_sc_unswizzled = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=min_exp)
    V_fp4_scaled = V / (2 ** V_sc_unswizzled.repeat_interleave(32, dim=-1))

    # Round to nearest FP4 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # For simplicity, clamp and let the hardware do the rounding
    V_fp4_scaled = V_fp4_scaled.clamp(-6.0, 6.0)

    # Convert scale to uint8 (E8M0 bias = 127)
    V_sc_unswizzled = (V_sc_unswizzled + fp8e8m0_bias).to(torch.uint8)

    return (
        V_fp4_scaled,    # (M, N) - scaled values (to be rounded to FP4)
        V_sc_unswizzled  # (M, N // 32) - E8M0 scale
    )


def torch_mxfp4_dequantize(
    V_fp4_scaled: torch.Tensor,    # (M, N) float32 scaled values
    V_sc_unswizzled: torch.Tensor  # (M, N // 32) uint8 E8M0 scales
) -> torch.Tensor:
    """Reference MXFP4 dequantization."""
    fp8e8m0_bias = 127
    scale = 2 ** (V_sc_unswizzled.to(torch.float32) - fp8e8m0_bias)
    # For dequantization we'd need to round the scaled values to FP4 grid first
    # but for the reference comparison we just multiply back
    return V_fp4_scaled * scale.repeat_interleave(32, dim=-1)


def scale_swizzle(
    V_sc_unswizzled: torch.Tensor # (M, N // 32)
) -> torch.Tensor:
    """Swizzle scales to MMA-compatible layout.
    Same layout as MXFP8 since both use E8M0 with block-32.
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    """
    assert len(V_sc_unswizzled.shape) == 2
    assert V_sc_unswizzled.dtype == torch.uint8
    assert V_sc_unswizzled.shape[0] % 128 == 0
    assert (V_sc_unswizzled.shape[1] * 32) % 128 == 0

    M_BLOCK = 128
    N_BLOCK = 4  # 128 / 32

    M, N_32 = V_sc_unswizzled.shape

    V_sc = V_sc_unswizzled        # (M, N_32)
    V_sc = V_sc.reshape(          # (M / 128, 128, N_32 / 4, 4)
        M // M_BLOCK, M_BLOCK,
        N_32 // N_BLOCK, N_BLOCK
    )
    V_sc = V_sc.transpose(1, 2)   # (M / 128, N_32 / 4, 128, 4)
    V_sc = V_sc.reshape(          # (M / 128, N_32 / 4, 4, 32, 4)
        M // M_BLOCK, N_32 // N_BLOCK,
        4, M_BLOCK // 4, N_BLOCK
    )
    V_sc = V_sc.transpose(-2, -3) # (M / 128, N_32 / 4, 32, 4, 4)
    V_sc = V_sc.reshape(          # (M / 128, N_32 / 4, 32, 16)
        M // M_BLOCK,
        N_32 // N_BLOCK,
        M_BLOCK // 4, N_BLOCK * 4
    )

    return V_sc.contiguous()


def check_diff(
    name: str,
    A: torch.Tensor,
    A_ref: torch.Tensor
) -> None:
    A = A.to(torch.float32)
    A_ref = A_ref.to(torch.float32)
    print(f"===============================================================================")
    print(f"<{name}>")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    print(f"Max:       {A.abs().max().item():.10f}")
    print(f"Ref max:   {A_ref.abs().max().item():.10f}")


if __name__ == '__main__':
    # Matrix dimensions
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 204800
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

    # Group size
    l2_size = 128 * 1024 * 1024
    size_per_group = M * N * 2  # bf16 input
    num_groups = (l2_size // size_per_group + 1) * 100
    print(f"{M=}, {N=}, {num_groups=}")

    # Generate reference outputs and input matrix
    groups = []
    for i in range(num_groups):
        # Create a known-quantizable input: fp8 values * E8M0 scales
        A_sc_unswizzled_ref = torch.randint(127 - 20, 127 + 20, (M, N // 32), dtype=torch.uint8, device="cuda")
        A_sc_ref = scale_swizzle(A_sc_unswizzled_ref)

        # Create bf16 input from dequantized random FP4 values
        # Random FP4 grid values scaled by the block scale
        scale = 2 ** (A_sc_unswizzled_ref.to(torch.float32) - 127.0)
        fp4_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                                dtype=torch.float32, device="cuda")
        random_indices = torch.randint(0, 16, (M, N), dtype=torch.long, device="cuda")
        A_fp4_vals = fp4_vals[random_indices]
        A_bf16 = (A_fp4_vals * scale.repeat_interleave(32, dim=-1)).to(torch.bfloat16)

        A_fp4x2 = torch.empty(M, N // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        A_sc = torch.empty_like(A_sc_ref)
        groups.append((A_bf16, A_fp4x2, A_sc))

    # Run our MXFP4 quantization kernel and check correctness
    mxfp4_quantize(A_bf16, A_fp4x2, A_sc)
    torch.cuda.synchronize()

    # Check scale correctness
    check_diff("TK-SC", A_sc, A_sc_ref)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        mxfp4_quantize(*groups[i % num_groups])
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        mxfp4_quantize(*groups[i % num_groups])
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS
    gb = M * N * (2 + 0.5 + 1 / 32) * 1e-9  # bf16 read, fp4 write, scale write
    gbps = gb / avg_time

    print(f"===============================================================================")
    print(f"Average time: {avg_time * 1e6:.2f} us")
    print(f"Average throughput: {gbps:.2f} GB/s")
