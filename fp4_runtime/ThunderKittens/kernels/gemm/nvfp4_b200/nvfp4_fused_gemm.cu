// ================================================================
// NVFP4 Fused Quantize + GEMM — Compilation unit
// Takes bf16 A activations + pre-quantized NVFP4 B weights,
// quantizes A on-the-fly inside the GEMM kernel.
// ================================================================
#include "nvfp4_fused_gemm.cuh"

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void fused_kernel_entrypoint(const __grid_constant__ nvfp4_fused_gemm::globals<C> g) {
    nvfp4_fused_gemm::kernel<C>(g);
}

#else // TORCH_COMPILE — PyTorch extension mode

#include "kittens.cuh"
#include <optional>

// ================================================================
// Fused GEMM entrypoint: A is bf16, B is pre-quantized NVFP4
// D = quantize(A_bf16) × B_fp4^T
// ================================================================
void nvfp4_fused_gemm_entrypoint(
    const at::Tensor &A_bf16,       // [M, K] bf16 activations
    const at::Tensor &B,            // [N, K/2] fp4x2
    const at::Tensor &B_sc,         // [N/128, K/64, 512] fp8
    const at::Tensor &B_sc_global,  // [1] float32
    at::Tensor &D                   // [M, N] bf16 output
) {
    int M = A_bf16.size(0);
    int K = A_bf16.size(1);
    int N = D.size(1);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::kFloat4_e2m1fn_x2, "B must be fp4x2");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");
    TORCH_CHECK(N % 128 == 0, "N must be multiple of 128");

    using C = nvfp4_fused_gemm::config<256, 3, 4, 4, 2, false>;
    using G = nvfp4_fused_gemm::globals<C>;

    // Build globals step-by-step (nvcc chokes on designated initializers with templates)
    G g;
    g.A_bf16 = kittens::py::tensor_to_gl<typename G::A_bf16_gl>(A_bf16);
    g.B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B);
    g.B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
        B_sc, 1,
        B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0),
        B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1),
        256);
    g.B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global);
    g.D = kittens::py::tensor_to_gl<typename G::D_gl>(D);
    g.D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D);
    g.D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D);
    g.q_dim = 0;
    g.k_dim = 0;
    g.use_split_D = false;
    g.b_sg_per_tile = nullptr;
    g.silu_dim = 0;

    kittens::py::launch_kernel<C, G, nvfp4_fused_gemm::kernel<C>>(g);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_fused_gemm", &nvfp4_fused_gemm_entrypoint,
          "Fused Quantize+GEMM: bf16 A × pre-quantized fp4 B",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"));
}

#endif
