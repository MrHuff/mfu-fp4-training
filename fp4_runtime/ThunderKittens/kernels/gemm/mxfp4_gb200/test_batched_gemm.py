import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import mxfp4_batched_gemm, mxfp4_quantize


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
    NUM_BATCHES = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    print(f"{M=}, {N=}, {K=}, {NUM_BATCHES=}")

    # Generate per-batch inputs and quantize them
    A_list, A_sc_list, B_list, B_sc_list, D_list = [], [], [], [], []
    A_src, B_src = [], []

    for i in range(NUM_BATCHES):
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        A_src.append(A_bf16)
        B_src.append(B_bf16)

        A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
        B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        B_sc = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")

        mxfp4_quantize(A_bf16, A_fp4x2, A_sc)
        mxfp4_quantize(B_bf16, B_fp4x2, B_sc)

        A_list.append(A_fp4x2)
        A_sc_list.append(A_sc)
        B_list.append(B_fp4x2)
        B_sc_list.append(B_sc)
        D_list.append(torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))

    # Run batched GEMM
    mxfp4_batched_gemm(A_list, A_sc_list, B_list, B_sc_list, D_list)

    # Validate each batch independently
    for i in range(NUM_BATCHES):
        D_ref = torch.matmul(A_src[i], B_src[i].T).to(torch.bfloat16)
        check_diff(f"Batch {i}", D_list[i], D_ref)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        mxfp4_batched_gemm(A_list, A_sc_list, B_list, B_sc_list, D_list)
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        mxfp4_batched_gemm(A_list, A_sc_list, B_list, B_sc_list, D_list)
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS
    total_flops = 2.0 * M * N * K * NUM_BATCHES
    tflops = total_flops * 1e-12

    print(f"===============================================================================")
    print(f"Average time: {avg_time * 1e6:.2f} us")
    print(f"Average TFLOPs: {tflops / avg_time:.2f} TFLOp/s ({NUM_BATCHES} batches)")
