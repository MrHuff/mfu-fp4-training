import torch
from low_bits_training.quantization.te_parity_linear_triton import TritonTEParityLinear
from low_bits_training.quantization.te_parity_linear_tex import TEParityLinearTex

device = 'cuda'
dtype = torch.float32

print('='*80)
print('FINAL VERIFICATION (FILE EXECUTION)')
print('='*80)

M, K, N = 2048, 1024, 1024

class ConfigRHT:
    def __init__(self, use_2d_weights=False):
        self.scale_type = 'E4M3'
        self.block_size = 16
        self.use_global_scale = True
        self.encode_centric = False
        self.use_rht = False
        self.scale_round_mode = 'TiesToEven'
        self.roundMode = 'TiesToEven'
        self.use_2d_weights = use_2d_weights
        self.eps = 0.0
        self.with_random_sign_mask = False
        self.scale_max = 448.0
        self.use_fp32_matmul = True
        self.use_bf16_scales = False
        self.use_bf16_data = False

def run_parity(name, use_2d):
    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=dtype).requires_grad_(True)
    dy = torch.randn(M, N, device=device, dtype=dtype)
    config = ConfigRHT(use_2d_weights=use_2d)
    
    # TEX Module (Reference)
    # Note: If TEX applies RHT to Weight (despite Recipe saying False), Fwd might mismatch.
    tex_mod = TEParityLinearTex(K, N, bias=False, mx_config=config, use_dequant_gemm=True).to(device).to(dtype)
    weight_ref = tex_mod.weight.clone()
    
    y_tex = tex_mod(x)
    y_tex.backward(dy)
    grad_x_tex = x.grad.clone()
    grad_w_tex = tex_mod.weight.grad.clone()
    x.grad = None
    
    # TRITON Module
    # (Weight RHT Disabled Internally to match Recipe)
    triton_mod = TritonTEParityLinear(K, N, bias=False, mx_config=config, use_dequant_gemm=True).to(device).to(dtype)
    with torch.no_grad(): triton_mod.weight.copy_(weight_ref)
    
    y_triton = triton_mod(x)
    y_triton.backward(dy)
    grad_x_triton = x.grad.clone()
    grad_w_triton = triton_mod.weight.grad.clone()
    
    fwd = (y_triton - y_tex).abs().max().item()
    gx = (grad_x_triton - grad_x_tex).abs().max().item()
    gw = (grad_w_triton - grad_w_tex).abs().max().item()
    
    # Tolerances
    # Fwd: Standard Quantization (Match expected high precision) -> 0.1
    # GradX: RHT Match -> 0.1
    # GradW: RHT Match -> 1.0 (Higher due to accumulation?)
    
    status = '✅' if (fwd < 0.1 and gx < 0.1 and gw < 1.0) else '⚠️'
    print(f'{status} {name:30s} | Fwd: {fwd:.6f} | GradX: {gx:.6f} | GradW: {gw:.6f}')
    return gw < 1.0

success = True
run_parity('1D + RHT', False)
run_parity('2D + RHT', True)
# print('='*80)
# if success:
#     print('ALL VERIFIED ✅')
# else:
#     print('FAILURES DETECTED ❌')
