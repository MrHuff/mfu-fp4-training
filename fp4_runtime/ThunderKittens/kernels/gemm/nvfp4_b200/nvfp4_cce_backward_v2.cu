// NVFP4 CCE Backward v2 — Fused softmax gradient + optional FP4 quantization
#include "nvfp4_cce_backward_v2.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

// BF16 mode: outputs BF16 grad_logits (identical interface to v1)
template <typename C>
static void launch_backward_v2_bf16(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    static_assert(C::USE_BF16_ACCUM, "Must use BF16 config for this function");
    using G = nvfp4_cce_backward_v2::globals<C>;
    const auto padded_M = A.size(0);
    const auto padded_N = B.size(0);

    // Create dummy tensors for unused FP4 output fields
    auto dummy_fp4 = A.new_empty({padded_M, padded_N / 2}, A.options().dtype(c10::kFloat4_e2m1fn_x2));
    auto dummy_sc  = A.new_empty(
        {std::max<int64_t>(1, padded_M / 128), std::max<int64_t>(1, padded_N / 64), 256},
        A.options().dtype(c10::kHalf));
    auto dummy_sg  = A.new_empty({1}, A.options().dtype(c10::kFloat));

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(grad_out),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(dummy_fp4),
        .G_sc_row = kittens::py::tensor_to_gl<typename G::G_sc_row_gl, false>(dummy_sc, 1, 1, 1, 256),
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(dummy_sg),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N,
        .encode_centric = false
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

// FP4 mode: outputs quantized G in NVFP4 format for separate GEMM calls
template <typename C>
static void launch_backward_v2_fp4(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v2::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    // Dummy BF16 output — must be properly sized for valid TMA descriptors
    // (even though D_out is never written in FP4 mode, the TMA desc is still created)
    auto dummy_bf16 = A.new_empty({M, N}, A.options().dtype(c10::kBFloat16));
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "NVFP4 v2 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 v2 analytic G_sg_row: ",
                cudaGetErrorString(err));

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(dummy_bf16),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(G_fp4_row),
        .G_sc_row = kittens::py::tensor_to_gl<typename G::G_sc_row_gl, false>(
            G_sc_row, 1,
            G_sc_row.dim()==2 ? G_sc_row.size(0)/128 : G_sc_row.size(0),
            G_sc_row.dim()==2 ? G_sc_row.size(1)/4 : G_sc_row.size(1), 256),
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(G_sg_row),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

template <typename C>
static void launch_backward_v2_fp4_enc(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    launch_backward_v2_fp4<C>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        G_fp4_row, G_sc_row, G_sg_row,
        lse, targets,
        grad_scale, M, N, filter_eps, true);
}

template <typename C>
static void launch_backward_v2_fp4_dec(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    launch_backward_v2_fp4<C>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        G_fp4_row, G_sc_row, G_sg_row,
        lse, targets,
        grad_scale, M, N, filter_eps, false);
}

// Config instantiations
using bwd_v2_bf16_L4_SG8 = nvfp4_cce_backward_v2::config<4, 8, true, true>;
using bwd_v2_fp4_L4_SG8  = nvfp4_cce_backward_v2::config<4, 8, false, true>;
using bwd_v2_bf16_epipipe_L4_SG8 = nvfp4_cce_backward_v2::experimental_config_epipipe<4, 8, true, true>;
using bwd_v2_fp4_epipipe_L4_SG8  = nvfp4_cce_backward_v2::experimental_config_epipipe<4, 8, false, true>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v2_bf16_L4_SG8", &launch_backward_v2_bf16<bwd_v2_bf16_L4_SG8>,
          "NVFP4 CCE backward v2 (BF16 output) L4 SG8");
    m.def("backward_v2_fp4_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_L4_SG8>,
          "NVFP4 CCE backward v2 (FP4 output) L4 SG8");
    m.def("backward_v2_fp4_enc_L4_SG8", &launch_backward_v2_fp4_enc<bwd_v2_fp4_L4_SG8>,
          "NVFP4 CCE backward v2 (FP4 encode-centric output) L4 SG8");
    m.def("backward_v2_fp4_dec_L4_SG8", &launch_backward_v2_fp4_dec<bwd_v2_fp4_L4_SG8>,
          "NVFP4 CCE backward v2 (FP4 decode-centric output) L4 SG8");
    m.def("experimental_backward_v2_bf16_epipipe_L4_SG8", &launch_backward_v2_bf16<bwd_v2_bf16_epipipe_L4_SG8>,
          "NVFP4 CCE backward v2 experimental epipipe (BF16 output) L4 SG8");
    m.def("experimental_backward_v2_fp4_epipipe_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_epipipe_L4_SG8>,
          "NVFP4 CCE backward v2 experimental epipipe (FP4 output) L4 SG8");
}
#endif
