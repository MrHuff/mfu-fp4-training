// NVFP4 CCE Backward v5 — owner-flipped fused passes
#include "nvfp4_cce_backward_v5_dE.cuh"
#include "nvfp4_cce_backward_v5_dC.cuh"
#include "nvfp4_cce_backward_v5_dE_superk4_experimental.cuh"
#include "nvfp4_cce_backward_v5_dC_superk4_experimental.cuh"
#include "nvfp4_cce_backward_v5_dE_fp4p1_bf16p3_experimental.cuh"
#include "nvfp4_cce_backward_v5_dC_fp4p1_bf16p3_experimental.cuh"
#include "nvfp4_cce_backward_v5_combo_tritonstyle_experimental.cuh"
#include "nvfp4_cce_backward_v3.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

template <typename C>
static void launch_backward_v5_dC(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps);

template <typename C>
static void launch_backward_v5_dE(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    at::Tensor &dE_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dE::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .C_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(C_col),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            C_col_sc, 1, C_col_sc.size(0), C_col_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(C_col_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dE_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dE::kernel<C>>(g);
}

using bwd_v5_dE_fp4_L4_SG8 = nvfp4_cce_backward_v5_dE::config<2, 8, true>;
using bwd_v5_dC_fp4_L4_SG8 = nvfp4_cce_backward_v5_dC::config<2, 8, true>;
using exp_bwd_v5_dE_fp4_L4_SG8 = nvfp4_cce_backward_v5_dE_superk4_experimental::config<2, 8, true>;
using exp_bwd_v5_dC_fp4_L4_SG8 = nvfp4_cce_backward_v5_dC_superk4_experimental::config<2, 8, true>;
using exp_bwd_v5_dE_fp4p1_bf16p3_L4_SG8 = nvfp4_cce_backward_v5_dE_fp4p1_bf16p3_experimental::config<2, 8, true>;
using exp_bwd_v5_dC_fp4p1_bf16p3_L4_SG8 = nvfp4_cce_backward_v5_dC_fp4p1_bf16p3_experimental::config<2, 8, true>;
using exp_bwd_v5_dE_fp4p1_bf16p3_L2_SG4 = nvfp4_cce_backward_v5_dE_fp4p1_bf16p3_experimental::config<2, 4, true>;
using exp_bwd_v5_dC_fp4p1_bf16p3_L2_SG4 = nvfp4_cce_backward_v5_dC_fp4p1_bf16p3_experimental::config<2, 4, true>;
using exp_bwd_v5_dE_fp4p1_bf16p3_L1_SG8 = nvfp4_cce_backward_v5_dE_fp4p1_bf16p3_experimental::config<1, 8, true>;
using exp_bwd_v5_dC_fp4p1_bf16p3_L1_SG8 = nvfp4_cce_backward_v5_dC_fp4p1_bf16p3_experimental::config<1, 8, true>;
using exp_bwd_v5_dE_fp4p1_bf16p3_L1_SG4 = nvfp4_cce_backward_v5_dE_fp4p1_bf16p3_experimental::config<1, 4, true>;
using exp_bwd_v5_dC_fp4p1_bf16p3_L1_SG4 = nvfp4_cce_backward_v5_dC_fp4p1_bf16p3_experimental::config<1, 4, true>;
using exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L4_SG8 =
    nvfp4_cce_backward_v5_combo_tritonstyle_experimental::config<2, 8, true, 4>;
using exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E2 =
    nvfp4_cce_backward_v5_combo_tritonstyle_experimental::config<2, 8, true, 2>;
using exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E1 =
    nvfp4_cce_backward_v5_combo_tritonstyle_experimental::config<2, 8, true, 1>;
using exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L1_SG8_E1 =
    nvfp4_cce_backward_v5_combo_tritonstyle_experimental::config<1, 8, true, 1>;
using exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L1_SG4_E1 =
    nvfp4_cce_backward_v5_combo_tritonstyle_experimental::config<1, 4, true, 1>;
using exp_bwd_v5_combo_publicv3_fp4_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd<4, 8, true, 4>;

template <typename C, int ComboMode>
static void launch_experimental_backward_v5_combo_publicv3_fp4(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dE_out, at::Tensor &dC_out,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    using G = nvfp4_cce_backward_v3::globals_3wg<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;
    (void)K;

    TORCH_CHECK(filter_eps == 0.0f,
                "experimental_backward_v5_combo_publicv3_fp4_L4_SG8 does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Combo public-v3 path expects G_sg_row to be a CUDA tensor with one float element.");

    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update combo public-v3 analytic G_sg_row: ",
                cudaGetErrorString(err));

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(G_fp4_row),
        .G_sc_row_ptr = reinterpret_cast<uint8_t*>(G_sc_row.data_ptr()),
        .G_sc_row_kgroups = N / 64,
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(G_sg_row),
        .G_fp4_col_ptr = reinterpret_cast<uint8_t*>(G_fp4_col.data_ptr()),
        .G_sc_col_ptr = reinterpret_cast<uint8_t*>(G_sc_col.data_ptr()),
        .G_sc_col_kgroups = M / 64,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .encode_centric = encode_centric,
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(C_col),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            C_col_sc, 1, C_col_sc.size(0), C_col_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(C_col_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(E_col_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(dE_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(dC_out),
        .combo_mode = ComboMode,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v5_dE(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    at::Tensor &dE_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dE_superk4_experimental::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .C_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(C_col),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            C_col_sc, 1, C_col_sc.size(0), C_col_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(C_col_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dE_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dE_superk4_experimental::kernel<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v5_dE_fp4p1_bf16p3(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col_bf16,
    at::Tensor &dE_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dE_fp4p1_bf16p3_experimental::globals<C>;

    TORCH_CHECK(C_col_bf16.scalar_type() == at::kBFloat16,
                "experimental_backward_v5_dE_fp4p1_bf16p3_L4_SG8 expects C_col_bf16 to have dtype bfloat16");
    TORCH_CHECK(C_col_bf16.is_contiguous(),
                "experimental_backward_v5_dE_fp4p1_bf16p3_L4_SG8 expects C_col_bf16 to be contiguous and pre-transposed into [K, N_pad]");

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .C_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(C_col_bf16),
        .dE_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dE_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dE_fp4p1_bf16p3_experimental::kernel<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v5_dC(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    // The 3-WG dC sandbox remains available through the dedicated debug trace
    // entrypoints, but it is not a viable standalone performance path. Route
    // the exported experimental dC entrypoint through the proven public v5 dC
    // kernel so the "experimental combo" stays aligned with real best-known use.
    (void)sizeof(C);
    launch_backward_v5_dC<bwd_v5_dC_fp4_L4_SG8>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        E_col, E_col_sc, E_col_sc_global,
        dC_out, lse, targets,
        grad_scale, M, N, K, filter_eps);
}

template <typename C>
static void launch_experimental_backward_v5_dC_fp4p1_bf16p3(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col_bf16,
    at::Tensor &dC_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dC_fp4p1_bf16p3_experimental::globals<C>;

    TORCH_CHECK(E_col_bf16.scalar_type() == at::kBFloat16,
                "experimental_backward_v5_dC_fp4p1_bf16p3_L4_SG8 expects E_col_bf16 to have dtype bfloat16");
    TORCH_CHECK(E_col_bf16.is_contiguous(),
                "experimental_backward_v5_dC_fp4p1_bf16p3_L4_SG8 expects E_col_bf16 to be contiguous and pre-transposed into [K, M_pad]");

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col_bf16),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC_fp4p1_bf16p3_experimental::kernel<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v5_combo_fp4p1_tritonstyle_exact(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_bf16, const at::Tensor &C_bf16,
    at::Tensor &dE_out, at::Tensor &dC_out,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_combo_tritonstyle_experimental::globals<C>;

    TORCH_CHECK(E_bf16.scalar_type() == at::kBFloat16,
                "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8 expects E_bf16 to have dtype bfloat16");
    TORCH_CHECK(C_bf16.scalar_type() == at::kBFloat16,
                "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8 expects C_bf16 to have dtype bfloat16");
    TORCH_CHECK(E_bf16.is_contiguous(),
                "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8 expects E_bf16 to be contiguous in row-major [M_pad, K]");
    TORCH_CHECK(C_bf16.is_contiguous(),
                "experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8 expects C_bf16 to be contiguous in row-major [N_pad, K]");

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_bf16 = kittens::py::tensor_to_gl<typename G::E_gl>(E_bf16),
        .C_bf16 = kittens::py::tensor_to_gl<typename G::C_gl>(C_bf16),
        .dE_out = kittens::py::tensor_to_gl<typename G::dE_gl>(dE_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::dC_gl>(dC_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_combo_tritonstyle_experimental::kernel<C>>(g);
}

template <typename C>
static void launch_debug_experimental_backward_v5_dC_trace(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out, at::Tensor &debug_trace, at::Tensor &debug_trace_count,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K,
    int debug_trace_mode, int debug_breakpoint,
    int debug_block_start, int debug_block_stride,
    float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dC_superk4_experimental::debug_globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(E_col_sc_global),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .debug_p3_b_fp4_ptr = nullptr,
        .debug_p3_b_sc_ptr = nullptr,
        .debug_gt_fp4_ptr = nullptr,
        .debug_gt_sc_ptr = nullptr,
        .debug_p3_out_ptr = nullptr,
        .debug_p3_out_raw_ptr = nullptr,
        .debug_p3_b_fp4_stride = 0,
        .debug_gt_fp4_stride = 0,
        .debug_p3_out_stride = 0,
        .debug_p3_out_raw_stride = 0,
        .debug_trace_mode = debug_trace_mode,
        .debug_breakpoint = debug_breakpoint,
        .debug_block_start = debug_block_start,
        .debug_block_stride = debug_block_stride,
        .debug_trace_ptr = reinterpret_cast<int32_t*>(debug_trace.data_ptr()),
        .debug_trace_count_ptr = reinterpret_cast<int32_t*>(debug_trace_count.data_ptr()),
        .debug_trace_capacity = static_cast<int>(debug_trace.size(0)),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC_superk4_experimental::debug_kernel<C>>(g);
}

template <typename C>
static void launch_backward_v5_dC(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps)
{
    using G = nvfp4_cce_backward_v5_dC::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(E_col_sc_global),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::kernel<C>>(g);
}

template <typename C>
static void launch_debug_backward_v5_dC_stage(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out,
    at::Tensor &debug_gt_fp4, at::Tensor &debug_gt_sc,
    at::Tensor &debug_p3_b_fp4, at::Tensor &debug_p3_b_sc, at::Tensor &debug_p3_out,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dC::debug_globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(E_col_sc_global),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .debug_p3_b_fp4_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_fp4.data_ptr()),
        .debug_p3_b_sc_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_sc.data_ptr()),
        .debug_gt_fp4_ptr = reinterpret_cast<uint8_t*>(debug_gt_fp4.data_ptr()),
        .debug_gt_sc_ptr = reinterpret_cast<uint8_t*>(debug_gt_sc.data_ptr()),
        .debug_p3_out_ptr = reinterpret_cast<bf16*>(debug_p3_out.data_ptr()),
        .debug_p3_out_raw_ptr = nullptr,
        .debug_p3_b_fp4_stride = static_cast<int>(debug_p3_b_fp4.size(1)),
        .debug_gt_fp4_stride = static_cast<int>(debug_gt_fp4.size(1)),
        .debug_p3_out_stride = static_cast<int>(debug_p3_out.size(1)),
        .debug_p3_out_raw_stride = 0,
        .debug_trace_mode = 0,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::debug_kernel<C>>(g);
}

template <typename C>
static void launch_debug_backward_v5_dC_p3_probe(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out,
    at::Tensor &debug_gt_fp4, at::Tensor &debug_gt_sc,
    at::Tensor &debug_gt_sc_cluster, at::Tensor &debug_p3_a_sc_chunks,
    at::Tensor &debug_p3_b_fp4, at::Tensor &debug_p3_b_sc,
    at::Tensor &debug_p3_out_raw, at::Tensor &debug_p3_out,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, int debug_trace_mode, int debug_p3_contract, int debug_b_payload_source, int debug_b_sc_source,
    float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dC::debug_globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(E_col_sc_global),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .debug_p3_b_fp4_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_fp4.data_ptr()),
        .debug_p3_b_sc_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_sc.data_ptr()),
        .debug_gt_fp4_ptr = reinterpret_cast<uint8_t*>(debug_gt_fp4.data_ptr()),
        .debug_gt_sc_ptr = reinterpret_cast<uint8_t*>(debug_gt_sc.data_ptr()),
        .debug_p3_out_ptr = reinterpret_cast<bf16*>(debug_p3_out.data_ptr()),
        .debug_p3_out_raw_ptr = reinterpret_cast<float*>(debug_p3_out_raw.data_ptr()),
        .debug_p3_b_fp4_stride = static_cast<int>(debug_p3_b_fp4.size(1)),
        .debug_gt_fp4_stride = static_cast<int>(debug_gt_fp4.size(1)),
        .debug_p3_out_stride = static_cast<int>(debug_p3_out.size(1)),
        .debug_p3_out_raw_stride = static_cast<int>(debug_p3_out_raw.size(1)),
        .debug_trace_mode = debug_trace_mode,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::debug_kernel<C>>(g);
}

template <typename C>
static void launch_debug_backward_v5_dC_mm2_probe(
    const at::Tensor &gt_fp4,
    const at::Tensor &gt_sc,
    const at::Tensor &p3_b_fp4,
    const at::Tensor &p3_b_sc,
    at::Tensor &debug_p3_out_raw,
    at::Tensor &debug_p3_out,
    float gt_sg,
    float p3_b_sg,
    int probe_mode)
{
    using G = nvfp4_cce_backward_v5_dC::p3_probe_globals<C>;
    G g {
        .gt_fp4_ptr = reinterpret_cast<const uint8_t*>(gt_fp4.data_ptr()),
        .gt_sc_ptr = reinterpret_cast<const uint8_t*>(gt_sc.data_ptr()),
        .p3_b_fp4_ptr = reinterpret_cast<const uint8_t*>(p3_b_fp4.data_ptr()),
        .p3_b_sc_ptr = reinterpret_cast<const uint8_t*>(p3_b_sc.data_ptr()),
        .gt_sg = gt_sg,
        .p3_b_sg = p3_b_sg,
        .out_bf_ptr = reinterpret_cast<bf16*>(debug_p3_out.data_ptr()),
        .out_raw_ptr = reinterpret_cast<float*>(debug_p3_out_raw.data_ptr()),
        .gt_fp4_stride = static_cast<int>(gt_fp4.size(1)),
        .p3_b_fp4_stride = static_cast<int>(p3_b_fp4.size(1)),
        .out_stride = static_cast<int>(debug_p3_out.size(1)),
        .out_raw_stride = static_cast<int>(debug_p3_out_raw.size(1)),
        .probe_mode = probe_mode,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::p3_probe_kernel<C>>(g);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v5_dE_fp4_L4_SG8", &launch_backward_v5_dE<bwd_v5_dE_fp4_L4_SG8>,
          "NVFP4 CCE backward v5 fused dE pass L4 SG8");
    m.def("backward_v5_dC_fp4_L4_SG8", &launch_backward_v5_dC<bwd_v5_dC_fp4_L4_SG8>,
          "NVFP4 CCE backward v5 fused dC pass L4 SG8");
    m.def("debug_v5_dC_stage_fp4_L4_SG8", &launch_debug_backward_v5_dC_stage<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC stage dump L4 SG8");
    m.def("debug_v5_dC_p3_probe_fp4_L4_SG8", &launch_debug_backward_v5_dC_p3_probe<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC phase-3 contract probe L4 SG8");
    m.def("debug_v5_dC_trace_fp4_L4_SG8", &launch_debug_backward_v5_dC_p3_probe<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC tiny trace L4 SG8");
    m.def("debug_v5_dC_mm2_probe_fp4_L4_SG8", &launch_debug_backward_v5_dC_mm2_probe<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only isolated NVFP4 dC phase-3 mm2 probe L4 SG8");
    m.def("experimental_backward_v5_dE_fp4_L4_SG8", &launch_experimental_backward_v5_dE<exp_bwd_v5_dE_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dE experimental sandbox");
    m.def("experimental_backward_v5_dE_fp4p1_bf16p3_L4_SG8", &launch_experimental_backward_v5_dE_fp4p1_bf16p3<exp_bwd_v5_dE_fp4p1_bf16p3_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dE hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_dE_fp4p1_bf16p3_L2_SG4", &launch_experimental_backward_v5_dE_fp4p1_bf16p3<exp_bwd_v5_dE_fp4p1_bf16p3_L2_SG4>,
          "Developer-only NVFP4 CCE v5 dE hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_dE_fp4p1_bf16p3_L1_SG8", &launch_experimental_backward_v5_dE_fp4p1_bf16p3<exp_bwd_v5_dE_fp4p1_bf16p3_L1_SG8>,
          "Developer-only NVFP4 CCE v5 dE hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_dE_fp4p1_bf16p3_L1_SG4", &launch_experimental_backward_v5_dE_fp4p1_bf16p3<exp_bwd_v5_dE_fp4p1_bf16p3_L1_SG4>,
          "Developer-only NVFP4 CCE v5 dE hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_dC_fp4_L4_SG8", &launch_experimental_backward_v5_dC<exp_bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC experimental sandbox");
    m.def("experimental_backward_v5_dC_fp4p1_bf16p3_L4_SG8", &launch_experimental_backward_v5_dC_fp4p1_bf16p3<exp_bwd_v5_dC_fp4p1_bf16p3_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_dC_fp4p1_bf16p3_L2_SG4", &launch_experimental_backward_v5_dC_fp4p1_bf16p3<exp_bwd_v5_dC_fp4p1_bf16p3_L2_SG4>,
          "Developer-only NVFP4 CCE v5 dC hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_dC_fp4p1_bf16p3_L1_SG8", &launch_experimental_backward_v5_dC_fp4p1_bf16p3<exp_bwd_v5_dC_fp4p1_bf16p3_L1_SG8>,
          "Developer-only NVFP4 CCE v5 dC hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_dC_fp4p1_bf16p3_L1_SG4", &launch_experimental_backward_v5_dC_fp4p1_bf16p3<exp_bwd_v5_dC_fp4p1_bf16p3_L1_SG4>,
          "Developer-only NVFP4 CCE v5 dC hybrid FP4 phase-1 + BF16 phase-3 path");
    m.def("experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8", &launch_experimental_backward_v5_combo_fp4p1_tritonstyle_exact<exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L4_SG8>,
          "Developer-only NVFP4 CCE v5 Triton-style combo path");
    m.def("experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E2", &launch_experimental_backward_v5_combo_fp4p1_tritonstyle_exact<exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E2>,
          "Developer-only NVFP4 CCE v5 Triton-style combo path with EPI depth 2");
    m.def("experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E1", &launch_experimental_backward_v5_combo_fp4p1_tritonstyle_exact<exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L4_SG8_E1>,
          "Developer-only NVFP4 CCE v5 Triton-style combo path with EPI depth 1");
    m.def("experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L1_SG8_E1", &launch_experimental_backward_v5_combo_fp4p1_tritonstyle_exact<exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L1_SG8_E1>,
          "Developer-only NVFP4 CCE v5 Triton-style combo path with load depth 1 and EPI depth 1");
    m.def("experimental_backward_v5_combo_fp4p1_tritonstyle_exact_L1_SG4_E1", &launch_experimental_backward_v5_combo_fp4p1_tritonstyle_exact<exp_bwd_v5_combo_fp4p1_tritonstyle_exact_L1_SG4_E1>,
          "Developer-only NVFP4 CCE v5 Triton-style combo path with load depth 1, supergroup 4, and EPI depth 1");
    m.def("experimental_backward_v5_combo_publicv3_fp4_L4_SG8",
          &launch_experimental_backward_v5_combo_publicv3_fp4<
              exp_bwd_v5_combo_publicv3_fp4_L4_SG8,
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v5_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_FULL>,
          "Developer-only NVFP4 combo path using the public-v3 front-half and in-kernel store-add dE/dC");
    m.def("experimental_backward_v5_combo_publicv3_fp4_gonly_L4_SG8",
          &launch_experimental_backward_v5_combo_publicv3_fp4<
              exp_bwd_v5_combo_publicv3_fp4_L4_SG8,
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v5_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_GONLY>,
          "Developer-only NVFP4 combo path using the public-v3 front-half only");
    m.def("experimental_backward_v5_combo_publicv3_fp4_dEonly_L4_SG8",
          &launch_experimental_backward_v5_combo_publicv3_fp4<
              exp_bwd_v5_combo_publicv3_fp4_L4_SG8,
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v5_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_DEONLY>,
          "Developer-only NVFP4 combo path using the public-v3 front-half plus in-kernel dE only");
    m.def("experimental_backward_v5_combo_publicv3_fp4_dConly_L4_SG8",
          &launch_experimental_backward_v5_combo_publicv3_fp4<
              exp_bwd_v5_combo_publicv3_fp4_L4_SG8,
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v5_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_DCONLY>,
          "Developer-only NVFP4 combo path using the public-v3 front-half plus in-kernel dC only");
    m.def("debug_experimental_v5_dC_trace_fp4_L4_SG8", &launch_debug_experimental_backward_v5_dC_trace<exp_bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 experimental dC bounded trace L4 SG8");
}
#endif
