// NVFP4 CCE Backward v6 — public-v3-front-half combo experiments
#include "nvfp4_cce_backward_v6.cuh"
#include "nvfp4_gemm.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef COMBO_DC_USE_STANDALONE_LAUNCH
#define COMBO_DC_USE_STANDALONE_LAUNCH 0
#endif

#ifndef V6_PUBLICV3_USE_CONSUMEROTF_FRONTEND
#define V6_PUBLICV3_USE_CONSUMEROTF_FRONTEND 0
#endif

#ifndef V6_PUBLICV3_USE_CONSUMEROTF_SAFUSED_FRONTEND
#define V6_PUBLICV3_USE_CONSUMEROTF_SAFUSED_FRONTEND 0
#endif

#ifndef V6_PUBLICV3_USE_ROWONLY_COMBO_DC_FRONTEND
#define V6_PUBLICV3_USE_ROWONLY_COMBO_DC_FRONTEND 0
#endif

using exp_bwd_v6_combo_publicv3_fp4_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd<4, 8, true, 4>;
using exp_bwd_v6_combo_publicv3_fp4_5wg_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_5wg<4, 8, true, 4>;
using exp_bwd_v6_combo_publicv3_fp4_rowonlygonly_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_dedicated_rowonly_gonly<4, 8, true, 4>;
using exp_bwd_v6_combo_publicv3_fp4_rowonlydc_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_dedicated_rowonly_combo_dc<4, 8, true, 4>;
#if V6_PUBLICV3_USE_CONSUMEROTF_SAFUSED_FRONTEND
using pub_bwd_v3_fp4_frontend_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_consumerotf_safused_warpbalanced<4, 8, true, 4>;
#elif V6_PUBLICV3_USE_CONSUMEROTF_FRONTEND
using pub_bwd_v3_fp4_frontend_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_consumerotf_warpbalanced<4, 8, true, 4>;
#else
using pub_bwd_v3_fp4_frontend_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<4, 8, true, 4>;
#endif
template <typename C>
static void launch_public_v3_fp4_frontend_for_bridge(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dE_out, at::Tensor &dC_out,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    using G = nvfp4_cce_backward_v3::globals_3wg<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    TORCH_CHECK(filter_eps == 0.0f,
                "v6 public-v3 bridge frontend does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "v6 public-v3 bridge frontend expects G_sg_row to be a CUDA tensor with one float element.");

    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update v6 public-v3 bridge analytic G_sg_row: ",
                cudaGetErrorString(err));
    const int combo_dummy_extent = (M > N) ? M : N;
    auto unused_combo_out = A.new_empty(
        {combo_dummy_extent, combo_dummy_extent},
        A.options().dtype(c10::kBFloat16));

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
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(B),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(A),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(A_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(unused_combo_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(unused_combo_out),
        .combo_mode = G::COMBO_MODE_GONLY,
        .diag_stage_cycles = nullptr,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg<C>>(g);
}

template <typename C, int StaticComboMode>
static void launch_public_v3_fp4_frontend_for_bridge_rowonly_combo_dc(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dE_out, at::Tensor &dC_out,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    using G = nvfp4_cce_backward_v3::globals_5wg<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    TORCH_CHECK(filter_eps == 0.0f,
                "v6 row-only combo-dC frontend does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "v6 row-only combo-dC frontend expects G_sg_row to be a CUDA tensor with one float element.");

    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update v6 row-only combo-dC analytic G_sg_row: ",
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
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
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
        .combo_mode = StaticComboMode,
    };
    if constexpr (StaticComboMode == G::COMBO_MODE_GONLY) {
        kittens::py::launch_kernel<
            C, G,
            nvfp4_cce_backward_v3::backward_kernel_v3_streaming_dedicated_rowonly_combo_gonly<C>>(g);
    } else {
        static_assert(StaticComboMode == G::COMBO_MODE_DCONLY,
                      "Dedicated row-only v6 bridge frontend only supports GONLY and DCONLY.");
        kittens::py::launch_kernel<
            C, G,
            nvfp4_cce_backward_v3::backward_kernel_v3_streaming_dedicated_rowonly_combo_dconly<C>>(g);
    }
}

template <typename C>
static void launch_public_v3_fp4_frontend_for_bridge_diag(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dE_out, at::Tensor &dC_out,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    at::Tensor &diag_stage_cycles,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    using G = nvfp4_cce_backward_v3::globals_3wg<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    TORCH_CHECK(filter_eps == 0.0f,
                "v6 public-v3 bridge frontend diag does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "v6 public-v3 bridge frontend diag expects G_sg_row to be a CUDA tensor with one float element.");
    TORCH_CHECK(diag_stage_cycles.is_cuda() &&
                    diag_stage_cycles.scalar_type() == at::kLong &&
                    diag_stage_cycles.numel() >= nvfp4_cce_backward_v3::kSafusedDiagTimingSlots,
                "diag_stage_cycles must be a CUDA int64 tensor with at least ",
                nvfp4_cce_backward_v3::kSafusedDiagTimingSlots,
                " elements.");

    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update v6 public-v3 bridge frontend diag analytic G_sg_row: ",
                cudaGetErrorString(err));
    err = cudaMemsetAsync(
        diag_stage_cycles.data_ptr<int64_t>(),
        0,
        sizeof(int64_t) * diag_stage_cycles.numel(),
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to clear v6 public-v3 bridge frontend diag buffer: ",
                cudaGetErrorString(err));
    const int combo_dummy_extent = (M > N) ? M : N;
    auto unused_combo_out = A.new_empty(
        {combo_dummy_extent, combo_dummy_extent},
        A.options().dtype(c10::kBFloat16));

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
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(B),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(A),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(A_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(unused_combo_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(unused_combo_out),
        .combo_mode = G::COMBO_MODE_GONLY,
        .diag_stage_cycles = reinterpret_cast<uint64_t*>(diag_stage_cycles.data_ptr<int64_t>())
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg<C>>(g);
}

template <typename C>
static void launch_nvfp4_gemm_bridge_with_config(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D)
{
    using G = nvfp4_gemm::globals<C>;
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
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .v_dim = 0,
        .use_split_D = false,
        .b_sg_per_tile = nullptr,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void launch_v6_dC_tail_from_global(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dE_out, at::Tensor &dC_out,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    using G = nvfp4_cce_backward_v6::globals_5wg<C>;

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
        .combo_mode = G::COMBO_MODE_DCONLY,
    };
    kittens::py::launch_kernel<
        C, G,
        nvfp4_cce_backward_v6::backward_kernel_v6_streaming_5wg_tail_dconly_from_global<C>>(g);
}

static void launch_nvfp4_gemm_bridge_dE_current(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D)
{
    const int M = D.size(0);
    const int N_out = D.size(1);
    const int K = B.size(1) * 2;
    if (K <= 2048) {
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else if (N_out >= 65536) {
        using C = nvfp4_gemm::config<256, 4, 16, 12, 2, false>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else if (K >= 8192 || M > K) {
        using C = nvfp4_gemm::config<256, 5, 8, 12, 2, true>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else {
        using C = nvfp4_gemm::config<256, 5, 8, 1, 2, false>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    }
}

static void launch_nvfp4_gemm_bridge_dC_current(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D)
{
    const int K = B.size(1) * 2;
    if (K <= 2048) {
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else {
        using C = nvfp4_gemm::config<256, 5, 8, 1, 2, false>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    }
}

static void launch_nvfp4_gemm_bridge(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D)
{
    const int K = B.size(1) * 2;
    if (K <= 2048) {
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else {
        using C = nvfp4_gemm::config<256, 4, 8, 12, 2, false>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    }
}

static void launch_nvfp4_gemm_bridge_nopdl(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D)
{
    const int K = B.size(1) * 2;
    if (K <= 2048) {
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else {
        using C = nvfp4_gemm::config<256, 4, 8, 12, 2, false, 256, false, 2>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    }
}

static void launch_nvfp4_gemm_bridge_dE_side_nopdl(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D)
{
    const int K = B.size(1) * 2;
    if (K <= 2048) {
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else {
        using C = nvfp4_gemm::config<256, 5, 8, 12, 2, true, 256, false, 2>;
        launch_nvfp4_gemm_bridge_with_config<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    }
}

static void launch_nvfp4_gemm_bridge_overlap_combo(
    const at::Tensor &G_fp4_row, const at::Tensor &G_sc_row, const at::Tensor &G_sg_row,
    const at::Tensor &G_fp4_col, const at::Tensor &G_sc_col,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dE_out, at::Tensor &dC_out)
{
    auto current_stream = at::cuda::getCurrentCUDAStream();
    const auto device_index = current_stream.device_index();
    const auto side_stream = c10::cuda::getStreamFromPool(false, device_index);

    thread_local cudaEvent_t frontend_done_event = nullptr;
    thread_local cudaEvent_t side_done_event = nullptr;
    if (frontend_done_event == nullptr) cudaEventCreateWithFlags(&frontend_done_event, cudaEventDisableTiming);
    if (side_done_event == nullptr) cudaEventCreateWithFlags(&side_done_event, cudaEventDisableTiming);

    cudaEventRecord(frontend_done_event, current_stream.stream());
    cudaStreamWaitEvent(side_stream.stream(), frontend_done_event, 0);

    const bool de_on_current = dE_out.size(0) >= dC_out.size(0);

    if (de_on_current) {
        {
            c10::cuda::CUDAStreamGuard guard(side_stream);
            auto G_fp4_col_fp4x2 = G_fp4_col.view(at::kFloat4_e2m1fn_x2);
            auto G_sc_col_fp8 = G_sc_col.view(at::kFloat8_e4m3fn);
            launch_nvfp4_gemm_bridge_nopdl(
                G_fp4_col_fp4x2, G_sc_col_fp8, G_sg_row,
                E_col, E_col_sc, E_col_sc_global,
                dC_out);
            cudaEventRecord(side_done_event, side_stream.stream());
        }
        auto G_sc_row_fp8 = G_sc_row.view(at::kFloat8_e4m3fn);
        launch_nvfp4_gemm_bridge_dE_current(
            G_fp4_row, G_sc_row_fp8, G_sg_row,
            C_col, C_col_sc, C_col_sc_global,
            dE_out);
    } else {
        {
            c10::cuda::CUDAStreamGuard guard(side_stream);
            auto G_sc_row_fp8 = G_sc_row.view(at::kFloat8_e4m3fn);
            launch_nvfp4_gemm_bridge_dE_side_nopdl(
                G_fp4_row, G_sc_row_fp8, G_sg_row,
                C_col, C_col_sc, C_col_sc_global,
                dE_out);
            cudaEventRecord(side_done_event, side_stream.stream());
        }
        auto G_fp4_col_fp4x2 = G_fp4_col.view(at::kFloat4_e2m1fn_x2);
        auto G_sc_col_fp8 = G_sc_col.view(at::kFloat8_e4m3fn);
        launch_nvfp4_gemm_bridge_dC_current(
            G_fp4_col_fp4x2, G_sc_col_fp8, G_sg_row,
            E_col, E_col_sc, E_col_sc_global,
            dC_out);
    }

    cudaStreamWaitEvent(current_stream.stream(), side_done_event, 0);
}

template <int ComboMode>
static void launch_experimental_backward_v6_combo_publicv3_fp4_bridge(
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
    (void)K;

    // Keep 3WG gonly/dEonly on the same dedicated row-only frontend that already
    // proved stable for 5WG. Combo/dConly continue to use the generic bridge.
    if constexpr (ComboMode == 1 || ComboMode == 2) {
        (void)G_fp4_col;
        (void)G_sc_col;
        constexpr int kRowonlyComboGonly =
            nvfp4_cce_backward_v3::globals_5wg<exp_bwd_v6_combo_publicv3_fp4_rowonlygonly_L4_SG8>::COMBO_MODE_GONLY;
        launch_public_v3_fp4_frontend_for_bridge_rowonly_combo_dc<
            exp_bwd_v6_combo_publicv3_fp4_rowonlygonly_L4_SG8,
            kRowonlyComboGonly>(
            A, A_sc, A_sc_global,
            B, B_sc, B_sc_global,
            C_col, C_col_sc, C_col_sc_global,
            E_col, E_col_sc, E_col_sc_global,
            dE_out, dC_out,
            G_fp4_row, G_sc_row, G_sg_row,
            lse, targets, grad_scale, M, N, filter_eps, encode_centric);
        if constexpr (ComboMode == 2) {
            auto G_sc_row_fp8 = G_sc_row.view(at::kFloat8_e4m3fn);
            launch_nvfp4_gemm_bridge_dE_current(
                G_fp4_row, G_sc_row_fp8, G_sg_row,
                C_col, C_col_sc, C_col_sc_global,
                dE_out);
        }
        return;
    }

#if V6_PUBLICV3_USE_ROWONLY_COMBO_DC_FRONTEND
    if constexpr (ComboMode == 0 || ComboMode == 3) {
        constexpr int kRowonlyComboDConly =
            nvfp4_cce_backward_v3::globals_5wg<exp_bwd_v6_combo_publicv3_fp4_rowonlydc_L4_SG8>::COMBO_MODE_DCONLY;
        launch_public_v3_fp4_frontend_for_bridge_rowonly_combo_dc<
            exp_bwd_v6_combo_publicv3_fp4_rowonlydc_L4_SG8,
            kRowonlyComboDConly>(
            A, A_sc, A_sc_global,
            B, B_sc, B_sc_global,
            C_col, C_col_sc, C_col_sc_global,
            E_col, E_col_sc, E_col_sc_global,
            dE_out, dC_out,
            G_fp4_row, G_sc_row, G_sg_row,
            lse, targets, grad_scale, M, N, filter_eps, encode_centric);
    }

    if constexpr (ComboMode == 0) {
        auto G_sc_row_fp8 = G_sc_row.view(at::kFloat8_e4m3fn);
        launch_nvfp4_gemm_bridge_dE_current(
            G_fp4_row, G_sc_row_fp8, G_sg_row,
            C_col, C_col_sc, C_col_sc_global,
            dE_out);
    }
    return;
#else
    launch_public_v3_fp4_frontend_for_bridge<pub_bwd_v3_fp4_frontend_L4_SG8>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        C_col, C_col_sc, C_col_sc_global,
        E_col, E_col_sc, E_col_sc_global,
        dE_out, dC_out,
        G_fp4_row, G_sc_row, G_sg_row,
        G_fp4_col, G_sc_col,
        lse, targets, grad_scale, M, N, filter_eps, encode_centric);

    if constexpr (ComboMode == 1) {
        return;
    }

    if constexpr (ComboMode == 0) {
        launch_nvfp4_gemm_bridge_overlap_combo(
            G_fp4_row, G_sc_row, G_sg_row,
            G_fp4_col, G_sc_col,
            C_col, C_col_sc, C_col_sc_global,
            E_col, E_col_sc, E_col_sc_global,
            dE_out, dC_out);
        return;
    }

    if constexpr (ComboMode == 2) {
        auto G_sc_row_fp8 = G_sc_row.view(at::kFloat8_e4m3fn);
        launch_nvfp4_gemm_bridge_dE_current(
            G_fp4_row, G_sc_row_fp8, G_sg_row,
            C_col, C_col_sc, C_col_sc_global,
            dE_out);
    }

    if constexpr (ComboMode == 3) {
        if constexpr (COMBO_DC_USE_STANDALONE_LAUNCH != 0) {
            launch_v6_dC_tail_from_global<exp_bwd_v6_combo_publicv3_fp4_L4_SG8>(
                A, A_sc, A_sc_global,
                B, B_sc, B_sc_global,
                C_col, C_col_sc, C_col_sc_global,
                E_col, E_col_sc, E_col_sc_global,
                dE_out, dC_out,
                G_fp4_row, G_sc_row, G_sg_row,
                G_fp4_col, G_sc_col,
                lse, targets, grad_scale, M, N, filter_eps, encode_centric);
        } else {
            auto G_fp4_col_fp4x2 = G_fp4_col.view(at::kFloat4_e2m1fn_x2);
            auto G_sc_col_fp8 = G_sc_col.view(at::kFloat8_e4m3fn);
            launch_nvfp4_gemm_bridge_dC_current(
                G_fp4_col_fp4x2, G_sc_col_fp8, G_sg_row,
                E_col, E_col_sc, E_col_sc_global,
                dC_out);
        }
    }
#endif
}

template <int ComboMode>
static void launch_experimental_backward_v6_combo_publicv3_fp4_bridge_5wg(
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
    if constexpr (ComboMode == 1 || ComboMode == 2) {
        (void)K;
        (void)G_fp4_col;
        (void)G_sc_col;
        constexpr int kRowonlyComboGonly =
            nvfp4_cce_backward_v3::globals_5wg<exp_bwd_v6_combo_publicv3_fp4_rowonlygonly_L4_SG8>::COMBO_MODE_GONLY;
        launch_public_v3_fp4_frontend_for_bridge_rowonly_combo_dc<
            exp_bwd_v6_combo_publicv3_fp4_rowonlygonly_L4_SG8,
            kRowonlyComboGonly>(
            A, A_sc, A_sc_global,
            B, B_sc, B_sc_global,
            C_col, C_col_sc, C_col_sc_global,
            E_col, E_col_sc, E_col_sc_global,
            dE_out, dC_out,
            G_fp4_row, G_sc_row, G_sg_row,
            lse, targets, grad_scale, M, N, filter_eps, encode_centric);
        if constexpr (ComboMode == 2) {
            auto G_sc_row_fp8 = G_sc_row.view(at::kFloat8_e4m3fn);
            launch_nvfp4_gemm_bridge_dE_current(
                G_fp4_row, G_sc_row_fp8, G_sg_row,
                C_col, C_col_sc, C_col_sc_global,
                dE_out);
        }
        return;
    }

    launch_experimental_backward_v6_combo_publicv3_fp4_bridge<ComboMode>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        C_col, C_col_sc, C_col_sc_global,
        E_col, E_col_sc, E_col_sc_global,
        dE_out, dC_out,
        G_fp4_row, G_sc_row, G_sg_row,
        G_fp4_col, G_sc_col,
        lse, targets, grad_scale, M, N, K, filter_eps, encode_centric);
}

static void launch_experimental_backward_v6_combo_publicv3_fp4_5wg_gonly_diag(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dE_out, at::Tensor &dC_out,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    at::Tensor &diag_stage_cycles,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    (void)K;
    launch_public_v3_fp4_frontend_for_bridge_diag<pub_bwd_v3_fp4_frontend_L4_SG8>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        C_col, C_col_sc, C_col_sc_global,
        E_col, E_col_sc, E_col_sc_global,
        dE_out, dC_out,
        G_fp4_row, G_sc_row, G_sg_row,
        G_fp4_col, G_sc_col,
        lse, targets, diag_stage_cycles, grad_scale, M, N, filter_eps, encode_centric);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("experimental_rowonly_combo_dc_frontend_enabled") =
        pybind11::bool_(V6_PUBLICV3_USE_ROWONLY_COMBO_DC_FRONTEND != 0);
    m.def("experimental_backward_v6_combo_publicv3_fp4_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge<
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v6_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_FULL>,
          "Developer-only NVFP4 v6 combo path using the public-v3 front-half plus bridged dE/dC GEMM tails");
    m.def("experimental_backward_v6_combo_publicv3_fp4_gonly_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge<
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v6_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_GONLY>,
          "Developer-only NVFP4 v6 combo path using the public-v3 front-half only");
    m.def("experimental_backward_v6_combo_publicv3_fp4_dEonly_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge<
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v6_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_DEONLY>,
          "Developer-only NVFP4 v6 combo path using the public-v3 front-half plus a bridged dE GEMM tail");
    m.def("experimental_backward_v6_combo_publicv3_fp4_dConly_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge<
              nvfp4_cce_backward_v3::globals_3wg<exp_bwd_v6_combo_publicv3_fp4_L4_SG8>::COMBO_MODE_DCONLY>,
          "Developer-only NVFP4 v6 combo path using the public-v3 front-half plus a bridged dC GEMM tail");
    m.def("experimental_backward_v6_combo_publicv3_fp4_5wg_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge_5wg<
              nvfp4_cce_backward_v3::globals_5wg<exp_bwd_v6_combo_publicv3_fp4_5wg_L4_SG8>::COMBO_MODE_FULL>,
          "Developer-only NVFP4 v6 combo path using the public-v3 front-half plus bridged dE/dC GEMM tails");
    m.def("experimental_backward_v6_combo_publicv3_fp4_5wg_gonly_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge_5wg<
              nvfp4_cce_backward_v3::globals_5wg<exp_bwd_v6_combo_publicv3_fp4_5wg_L4_SG8>::COMBO_MODE_GONLY>,
          "Developer-only NVFP4 v6 5wg combo path using the public-v3 front-half only");
    m.def("experimental_backward_v6_combo_publicv3_fp4_5wg_gonly_diag_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_5wg_gonly_diag,
          "Developer-only NVFP4 v6 5wg combo path using the public-v3 front-half only with diag timing output");
    m.def("experimental_backward_v6_combo_publicv3_fp4_5wg_dEonly_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge_5wg<
              nvfp4_cce_backward_v3::globals_5wg<exp_bwd_v6_combo_publicv3_fp4_5wg_L4_SG8>::COMBO_MODE_DEONLY>,
          "Developer-only NVFP4 v6 5wg combo path using the public-v3 front-half plus a bridged dE GEMM tail");
    m.def("experimental_backward_v6_combo_publicv3_fp4_5wg_dConly_L4_SG8",
          &launch_experimental_backward_v6_combo_publicv3_fp4_bridge_5wg<
              nvfp4_cce_backward_v3::globals_5wg<exp_bwd_v6_combo_publicv3_fp4_5wg_L4_SG8>::COMBO_MODE_DCONLY>,
          "Developer-only NVFP4 v6 5wg combo path using the public-v3 front-half plus a bridged dC GEMM tail");
}
#endif
