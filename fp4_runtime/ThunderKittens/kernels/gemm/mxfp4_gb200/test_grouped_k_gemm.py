import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import mxfp4_grouped_k_gemm, mxfp4_quantize


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
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    NUM_K_GROUPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    print(f"{M=}, {N=}, {K=}, {NUM_K_GROUPS=}")

    # Generate input matrices
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25

    # Quantize to MXFP4
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    mxfp4_quantize(A, A_fp4x2, A_sc)
    mxfp4_quantize(B, B_fp4x2, B_sc)

    # Create K-tile group boundaries
    # K-tiles = 2*K / Kb. With Kb=128, num_k_tiles = K/64
    num_k_tiles = K // 64  # each K-tile covers 64 cols in both A halves
    tiles_per_group = num_k_tiles // NUM_K_GROUPS
    assert tiles_per_group * NUM_K_GROUPS == num_k_tiles, \
        f"K-tiles {num_k_tiles} must be divisible by NUM_K_GROUPS {NUM_K_GROUPS}"

    group_k_start = torch.zeros(NUM_K_GROUPS + 1, dtype=torch.int32, device="cuda")
    for i in range(NUM_K_GROUPS + 1):
        group_k_start[i] = min(i * tiles_per_group, num_k_tiles)

    print(f"K-tile boundaries: {group_k_start.cpu().tolist()}")

    # Output
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Run grouped-K GEMM
    mxfp4_grouped_k_gemm(A_fp4x2, A_sc, B_fp4x2, B_sc, D, group_k_start)

    # Reference: full matmul
    D_ref = torch.matmul(A, B.T).to(torch.bfloat16)
    check_diff("TK-MXFP4-Grouped-K-GEMM", D, D_ref)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        mxfp4_grouped_k_gemm(A_fp4x2, A_sc, B_fp4x2, B_sc, D, group_k_start)
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        mxfp4_grouped_k_gemm(A_fp4x2, A_sc, B_fp4x2, B_sc, D, group_k_start)
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS
    flops = 2.0 * M * N * K
    tflops = flops * 1e-12

    print(f"Average time: {avg_time * 1e6:.2f} us")
    print(f"Average TFLOPs: {tflops / avg_time:.2f} TFLOp/s")
