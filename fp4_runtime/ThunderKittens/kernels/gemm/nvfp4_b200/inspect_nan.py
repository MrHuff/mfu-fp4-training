import torch
import _C

M, N, K = 1024, 1024, 1024
A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K**0.25)
B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.25)
B_fp4x2 = torch.zeros(N, K//2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
B_sc = torch.zeros(N//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
B_sc_global = torch.zeros(1, dtype=torch.float32, device="cuda")
_C.nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

A_fp4x2 = torch.zeros(M, K//2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
A_sc = torch.zeros(M//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
A_sc_global = torch.zeros(1, dtype=torch.float32, device="cuda")
_C.nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)

for i in range(5):
    D_fused = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    _C.nvfp4_fused_gemm(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused)
    print(f"Iter {i} Constant SCALE MAX D_fused has NaNs?", torch.isnan(D_fused).any().item())
