import torch
import unittest
import time
from low_bits_training.quantization.dimensionQuantisationClass import *
from torchao.prototype.mx_formats.mx_linear import replace_with_custom_fn_if_matches_filter, _is_linear
from low_bits_training.quantization.MXFPconfig import MXLinearDimConfig
from low_bits_training.quantization.mx_ops_dim import *

# ==============================================================================
# UNIT TEST CLASS
# ==============================================================================

class TestMatMulComparison(unittest.TestCase):

    def setUp(self):
        """Set up common tensors and modules for the tests."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not torch.cuda.is_available() or not hasattr(torch, 'float8_e4m3fn'):
            self.skipTest("CUDA with FP8 support is required for this test.")

        # Use bfloat16 as it's common for LLMs
        self.dtype = torch.bfloat16
        
        # Quantization settings
        self.block_size = 32
        self.scaling_module = MXFPscalingModule(elem_dtype=DTYPE_FP4, block_size=self.block_size,use_approx={'smooth': 'STE', 'alpha': 80, 'stepGradient': 'STE','use_tensor_scaling':True},fp_scale_factor=False).to(self.device)
        self.quantiser_FP4 = E2M1Quantizer()

    # def test_numerical_correctness(self):
    #     """Verify that the FP8-accelerated mode is numerically close to the emulated one."""
    #     print("\n--- Running Numerical Correctness Test ---")
        
    #     # ARRANGE
    #     M, K, N = 2048, 4096, 2048
    #     a_tensor = torch.randn(M, K, device=self.device, dtype=self.dtype)
    #     b_tensor = torch.randn(K, N, device=self.device, dtype=self.dtype) # b is the weight

    #     # Quantize tensors to MXFP4 format
    #     # Here we use a symmetric setup for a direct comparison of the matmul logic.
    #     a_mx,_,_ = new_to_mx(a_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
    #     b_mx,_,_= new_to_mx(b_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
    #     # We quantize b.t() row-wise, which is equivalent to quantizing b column-wise.


    #     # ACT
    #     # Result from the original, emulated method (our ground truth)
    #     expected_output = mx_matmul(a_mx, b_mx)
        
    #     # Result from the new, FP8-accelerated method
    #     actual_output = mx_matmul_fp8(a_tensor, b_tensor)

    #     # ASSERT
    #     print("Comparing outputs...")
    #     # The accumulator precision in FP8 hardware can be slightly different from bf16.
    #     # We use torch.testing.assert_close to check for numerical similarity.
    #     torch.testing.assert_close(actual_output, expected_output, rtol=1e-2, atol=1e-2)
    #     print("✅ Numerical Correctness Test Passed!")

    def test_performance(self):
        """Benchmark the performance of both matmul modes."""
        print("\n--- Running Performance Test ---")

        # ARRANGE
        M, K, N = 2048, 4096, 2048
        a_tensor = torch.randn(M, K, device=self.device, dtype=self.dtype).contiguous()
        b_tensor = torch.randn(K, N, device=self.device, dtype=self.dtype).contiguous()

        a_mx,_,_ = new_to_mx(a_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
        b_mx,_,_ = new_to_mx(b_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
        
        # Warm-up GPU
        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            a_mx,_,_ = new_to_mx(a_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
            b_mx,_,_ = new_to_mx(b_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
            _ = mx_matmul(a_mx, b_mx)
            _ = mx_matmul_fp8(a_tensor, b_tensor)
        
        torch.cuda.synchronize()

        # ACT & MEASURE
        num_runs = 20
        
        start_time = time.time()
        for _ in range(num_runs):
            torch.compiler.cudagraph_mark_step_begin()
            a_mx,_,_ = new_to_mx(a_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
            b_mx,_,_ = new_to_mx(b_tensor, self.scaling_module, fp4_quantiser=self.quantiser_FP4)
            _ = mx_matmul(a_mx, b_mx)
        torch.cuda.synchronize()
        emulated_time = (time.time() - start_time) / num_runs

        start_time = time.time()
        for _ in range(num_runs):
            _ = mx_matmul_fp8(a_tensor, b_tensor)
        torch.cuda.synchronize()
        accelerated_time = (time.time() - start_time) / num_runs

        # ASSERT & REPORT
        print(f"Emulated Mode Average Time:    {emulated_time:.6f} seconds")
        print(f"Regular BF16 Avg Time: {accelerated_time:.6f} seconds")
        speedup = emulated_time / accelerated_time
        print(f"🚀 Speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 1.5, "Expected significant speedup from FP8 acceleration.")


if __name__ == '__main__':
    unittest.main()