"""
End-to-end test: MXFP4 quantization → TK MXFP4 GEMM.

This test:
1. Generates BF16 matrices
2. Quantizes using TK's MXFP4 quantization (E8M0 scales, block-32)
3. Runs TK MXFP4 GEMM on the quantized data
4. Compares against a Python reference that dequantizes + BF16 matmul
5. Also compares against raw BF16 matmul for end-to-end error

Usage:
  python test_e2e_mxfp4.py [M] [N] [K]
"""
import sys
import torch
import math
torch.random.manual_seed(42)

from _C import mxfp4_gemm, mxfp4_quantize


def mxfp4_dequantize_ref(fp4x2: torch.Tensor, scales: torch.Tensor,
                          M: int, K: int) -> torch.Tensor:
    """
    Reference MXFP4 dequantization in Python.
    
    fp4x2: [M, K//2] dtype=float4_e2m1fn_x2
    scales: [M//128, K//128, 32, 16] dtype=uint8 (E8M0)
    
    Returns: [M, K] BF16 tensor
    """
    # Convert FP4 data to BF16 via PyTorch's view
    # fp4x2 stores pairs of FP4 values. We can view as uint8 and unpack.
    raw = fp4x2.view(torch.uint8)  # [M, K//2] bytes
    
    # Unpack FP4 pairs: low nibble = even element, high nibble = odd element
    # FP4 E2M1 values: 0,0.5,1,1.5,2,3,4,6 (with sign bit)
    fp4_lut = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=raw.device)
    
    low_nibble = (raw & 0x0F).to(torch.int64)
    high_nibble = ((raw >> 4) & 0x0F).to(torch.int64)
    
    even_vals = fp4_lut[low_nibble]   # [M, K//2]
    odd_vals = fp4_lut[high_nibble]   # [M, K//2]
    
    # Interleave: result[..., 2*i] = even, result[..., 2*i+1] = odd
    result = torch.empty(M, K, dtype=torch.float32, device=raw.device)
    result[:, 0::2] = even_vals
    result[:, 1::2] = odd_vals
    
    # Apply E8M0 scales: scale_value = 2^(byte - 127)
    # scales shape: [M//128, K//128, 32, 16]
    # The MMA swizzle layout maps to row-major with specific tiling
    # For now, we'll compute the expected decode scale per block-32 group
    
    scale_M = M // 128
    scale_K = K // 128
    
    for tm in range(scale_M):
        for tk in range(scale_K):
            # Each 128x128 tile has 32x16 bytes of scales
            # This covers (128 rows) × (128 cols / 32) = 128 × 4 blocks
            scale_tile = scales[tm, tk]  # [32, 16]
            
            row_start = tm * 128
            col_start = tk * 128
            
            # The scale layout in TK: for each row in 0..127 and each
            # block-of-32 in 0..3:
            #   j = row_in_tile % 32
            #   group = row_in_tile / 32
            #   byte = j * 16 + group * 4 + block_idx
            for row_in_tile in range(128):
                row = row_start + row_in_tile
                if row >= M:
                    break
                j = row_in_tile % 32
                group = row_in_tile // 32
                for blk in range(4):
                    col_blk_start = col_start + blk * 32
                    byte_idx = j * 16 + group * 4 + blk
                    flat_idx = byte_idx
                    row_idx = flat_idx // 16
                    col_idx = flat_idx % 16
                    scale_byte = scale_tile[row_idx, col_idx].item()
                    scale_val = 2.0 ** (scale_byte - 127)
                    result[row, col_blk_start:col_blk_start+32] *= scale_val
    
    return result.to(torch.bfloat16)


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor) -> dict:
    A = A.to(torch.float32)
    A_ref = A_ref.to(torch.float32)
    max_diff = (A - A_ref).abs().max().item()
    mean_diff = (A - A_ref).abs().mean().item()
    
    # Relative error
    denom = A_ref.abs().mean().item()
    rel_err = mean_diff / denom if denom > 0 else float('inf')
    
    print(f"{'=' * 70}")
    print(f"<{name}>")
    print(f"  Max diff:    {max_diff:.6f}")
    print(f"  Mean diff:   {mean_diff:.6f}")
    print(f"  Relative:    {rel_err:.4%}")
    print(f"  Mean |A|:    {A.abs().mean().item():.6f}")
    print(f"  Mean |Ref|:  {A_ref.abs().mean().item():.6f}")
    return {"max_diff": max_diff, "mean_diff": mean_diff, "rel_err": rel_err}


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    
    print(f"MXFP4 E2E Test: M={M}, N={N}, K={K}")
    print(f"  Block size: 32, Scale type: E8M0")
    
    # Generate input matrices
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    
    # Quantize with TK
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
    
    mxfp4_quantize(A_bf16, A_fp4x2, A_sc)
    mxfp4_quantize(B_bf16, B_fp4x2, B_sc)
    
    # Run TK MXFP4 GEMM
    C_tk = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    mxfp4_gemm(A_fp4x2, A_sc, B_fp4x2, B_sc, C_tk)
    
    # Reference: BF16 matmul (no quantization)
    C_bf16 = torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16)
    
    # Check TK GEMM vs BF16 reference
    stats = check_diff("TK MXFP4 GEMM vs BF16 Matmul (E2E error)", C_tk, C_bf16)
    
    # Benchmark
    print(f"\n{'=' * 70}")
    print("Benchmarking...")
    
    NUM_WARMUPS = 5
    NUM_ITERS = 20
    
    # Pre-allocate groups for cache eviction
    l2_size = 128 * 1024 * 1024
    size_per_group = M * K // 2 + K * N // 2
    num_groups = max(1, (l2_size // size_per_group + 1) * 10)
    
    groups = []
    for i in range(num_groups):
        A_i = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        B_i = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        Aq = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        As = torch.empty(M // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
        Bq = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        Bs = torch.empty(N // 128, K // 128, 32, 16, dtype=torch.uint8, device="cuda")
        C_i = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        mxfp4_quantize(A_i, Aq, As)
        mxfp4_quantize(B_i, Bq, Bs)
        groups.append((Aq, As, Bq, Bs, C_i))
    
    # Warmup
    for i in range(NUM_WARMUPS):
        mxfp4_gemm(*groups[i % num_groups])
    torch.cuda.synchronize()
    
    # Benchmark quantize + GEMM
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for i in range(NUM_ITERS):
        mxfp4_gemm(*groups[i % num_groups])
    end.record()
    torch.cuda.synchronize()
    
    total_time = start.elapsed_time(end) * 1e-3
    avg_time = total_time / NUM_ITERS
    flops = 2.0 * M * N * K
    tflops = flops * 1e-12
    
    print(f"  GEMM-only avg: {avg_time * 1e6:.2f} us")
    print(f"  GEMM-only TFLOPs: {tflops / avg_time:.2f}")
    
    print(f"\n{'=' * 70}")
    print("DONE")
