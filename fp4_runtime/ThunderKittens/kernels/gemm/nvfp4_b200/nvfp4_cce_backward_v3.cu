// NVFP4 CCE Backward v3 — Fused softmax gradient + optional FP4 quantization
#include "nvfp4_cce_backward_v3.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

static at::Tensor make_unused_bf16_placeholder(const at::Tensor &ref) {
    // FP4 paths never consume D_out, but the launch still constructs a TMA
    // descriptor for the BF16 tensor shape. Keep only the minimum tile-sized
    // placeholder instead of allocating a full MxN buffer every call.
    return ref.new_empty({128, 32}, ref.options().dtype(c10::kBFloat16));
}

static at::Tensor make_unused_combo_placeholder(const at::Tensor &ref) {
    // G-only combo frontends still need a legal BF16 tensor map for the unused
    // combo dE/dC outputs. A single 128x32 tile is sufficient and matches the
    // legal minimum already used by the col-only path below.
    return ref.new_empty({128, 32}, ref.options().dtype(c10::kBFloat16));
}

// BF16 mode: outputs BF16 grad_logits (identical interface to v1)
template <typename C>
static void launch_backward_v3_bf16(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    static_assert(C::USE_BF16_ACCUM, "Must use BF16 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    const auto padded_M = A.size(0);
    const auto padded_N = B.size(0);

    // Create dummy tensors for unused FP4 output fields
    auto dummy_fp4 = A.new_empty({padded_M, padded_N / 2}, A.options().dtype(c10::kFloat4_e2m1fn_x2));
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
        .G_sc_row_ptr = nullptr,
        .G_sc_row_kgroups = 1,
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(dummy_sg),
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 1,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N,
        .encode_centric = false
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3<C>>(g);
}

// FP4 mode: outputs quantized G in NVFP4 format for separate GEMM calls.
// G_sg_row is derived analytically from grad_scale so the per-16 FP8
// micro-scales stay within E4M3 range without a separate global-amax pass.
template <typename C>
static void launch_backward_v3_fp4(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    // Dummy BF16 output — D_out is unused in FP4 mode but TMA needs valid tensor
    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 v3 analytic G_sg_row: ",
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
        .G_sc_row_ptr = reinterpret_cast<uint8_t*>(G_sc_row.data_ptr()),
        .G_sc_row_kgroups = N / 64,  // V/64 k-groups for row-quant scale
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(G_sg_row),
        .G_fp4_col_ptr = reinterpret_cast<uint8_t*>(G_fp4_col.data_ptr()),
        .G_sc_col_ptr = reinterpret_cast<uint8_t*>(G_sc_col.data_ptr()),
        .G_sc_col_kgroups = M / 64,  // M/64 k-groups for col-quant scale
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming<C>>(g);
}

template <typename C, int StaticComboMode>
static void launch_experimental_backward_v3_fp4_rowonly_combo_dc_5wg(
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
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals_5wg<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    TORCH_CHECK(filter_eps == 0.0f,
                "NVFP4 experimental v3 row-only combo-dC frontend does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 row-only combo-dC analytic G_sg_row: ",
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
    kittens::py::launch_kernel<
        C, G,
        nvfp4_cce_backward_v3::backward_kernel_v3_streaming_5wg_mode<C, StaticComboMode>>(g);
}

template <typename C, int StaticComboMode>
static void launch_experimental_backward_v3_fp4_dedicated_rowonly_combo_dc_5wg(
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
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals_5wg<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    TORCH_CHECK(filter_eps == 0.0f,
                "NVFP4 experimental v3 dedicated row-only combo-dC frontend does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 dedicated row-only combo-dC analytic G_sg_row: ",
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
                      "Dedicated row-only combo-dC frontend only supports GONLY and DCONLY.");
        kittens::py::launch_kernel<
            C, G,
            nvfp4_cce_backward_v3::backward_kernel_v3_streaming_dedicated_rowonly_combo_dconly<C>>(g);
    }
}

template <typename C>
static void launch_experimental_backward_v3_fp4_replayonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 replay-only analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_replayonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_2ctaS(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 2CTA-S analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_2ctaS<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_2ctaS_replayonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 2CTA-S replay-only analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_2ctaS_replayonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_2ctaSdupB(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals_2ctaSdupB<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 2CTA-S duplicate-B analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_2ctaSdupB<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_2ctaSdupB_replayonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals_2ctaSdupB<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 2CTA-S duplicate-B replay-only analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_2ctaSdupB_replayonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_3wg(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
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
                "NVFP4 experimental v3 3WG path does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 3WG analytic G_sg_row: ",
                cudaGetErrorString(err));
    auto unused_combo_out = make_unused_combo_placeholder(A);
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
        .M = M, .N = N,
        .encode_centric = encode_centric,
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(B),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(A),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(A_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(unused_combo_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(unused_combo_out),
        .combo_mode = G::COMBO_MODE_GONLY,
        .diag_stage_cycles = nullptr
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_3wg_diag(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
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
                "NVFP4 experimental v3 3WG diag path does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
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
                "Failed to update NVFP4 experimental v3 3WG diag analytic G_sg_row: ",
                cudaGetErrorString(err));
    err = cudaMemsetAsync(
        diag_stage_cycles.data_ptr<int64_t>(),
        0,
        sizeof(int64_t) * diag_stage_cycles.numel(),
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to clear NVFP4 experimental v3 3WG diag buffer: ",
                cudaGetErrorString(err));
    auto unused_combo_out = make_unused_combo_placeholder(A);

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
        .M = M, .N = N,
        .encode_centric = encode_centric,
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(B),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(A),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(A_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(unused_combo_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(unused_combo_out),
        .combo_mode = G::COMBO_MODE_GONLY,
        .diag_stage_cycles = reinterpret_cast<uint64_t*>(diag_stage_cycles.data_ptr<int64_t>())
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_3wg_replayonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
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
                "NVFP4 experimental v3 3WG replay-only path does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 3WG replay-only analytic G_sg_row: ",
                cudaGetErrorString(err));
    auto unused_combo_out = make_unused_combo_placeholder(A);

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
        .M = M, .N = N,
        .encode_centric = encode_centric,
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(B),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(A),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(A_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(unused_combo_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(unused_combo_out),
        .combo_mode = G::COMBO_MODE_GONLY,
        .diag_stage_cycles = nullptr
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg_replayonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_3wg_rowonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
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
                "NVFP4 experimental v3 3WG row-only path does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 3WG row-only analytic G_sg_row: ",
                cudaGetErrorString(err));
    auto unused_combo_out = make_unused_combo_placeholder(A);

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
        .M = M, .N = N,
        .encode_centric = encode_centric,
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(B),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(A),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(A_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(unused_combo_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(unused_combo_out),
        .combo_mode = G::COMBO_MODE_GONLY,
        .diag_stage_cycles = nullptr
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg_rowonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_3wg_colonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
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
                "NVFP4 experimental v3 3WG col-only path does not support filter_eps yet.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 3WG col-only analytic G_sg_row: ",
                cudaGetErrorString(err));
    // Keep a legal combo TMA tile while forcing combo_num_k_blocks == 0 in G-only col-only launches.
    auto unused_combo_out = A.new_empty(
        {C::Nb, C::Nb / C::EPI_PIPE_DEPTH},
        A.options().dtype(c10::kBFloat16));

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
        .M = M, .N = N,
        .encode_centric = encode_centric,
        .C_col = kittens::py::tensor_to_gl<typename G::combo_p3_C_gl>(B),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_C_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::combo_p3_E_gl>(A),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::combo_p3_E_sc_global_gl>(A_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::combo_dE_gl>(unused_combo_out),
        .dC_out = kittens::py::tensor_to_gl<typename G::combo_dC_gl>(unused_combo_out),
        .combo_mode = G::COMBO_MODE_GONLY,
        .diag_stage_cycles = nullptr
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg_colonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_rowonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 row-only analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_rowonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_col2pass_stage1(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D_scratch,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    TORCH_CHECK(D_scratch.is_cuda() && D_scratch.scalar_type() == at::kBFloat16,
                "D_scratch must be a CUDA bfloat16 tensor.");
    TORCH_CHECK(D_scratch.dim() == 2,
                "D_scratch must be a rank-2 BF16 tensor.");

    using BF16C = nvfp4_cce_backward_v3::config<C::LOAD_PIPE_DEPTH, C::SUPERGROUP_SIZE, true, C::PINGPONG>;
    launch_backward_v3_bf16<BF16C>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        D_scratch, lse, targets,
        grad_scale, M, N, filter_eps);

    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 col2pass stage1 analytic G_sg_row: ",
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
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(D_scratch),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(G_fp4_row),
        .G_sc_row_ptr = reinterpret_cast<uint8_t*>(G_sc_row.data_ptr()),
        .G_sc_row_kgroups = N / 64,
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(G_sg_row),
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 1,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_rowonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_colonly(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    auto dummy_bf16 = make_unused_bf16_placeholder(A);
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "Phase 3 G_sg_row must be a CUDA tensor with one float element.");
    const float g_sg = fmaxf(grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto err = cudaMemcpyAsync(
        G_sg_row.data_ptr<float>(),
        &g_sg,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "Failed to update NVFP4 experimental v3 col-only analytic G_sg_row: ",
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
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3_streaming_colonly<C>>(g);
}

template <typename C>
static void launch_experimental_backward_v3_fp4_col2pass_stage2(
    const at::Tensor &D_scratch,
    const at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    int M, int N,
    bool encode_centric = false)
{
    TORCH_CHECK(D_scratch.is_cuda() && D_scratch.scalar_type() == at::kBFloat16,
                "D_scratch must be a CUDA bfloat16 tensor.");
    TORCH_CHECK(D_scratch.dim() == 2,
                "D_scratch must be rank-2.");
    TORCH_CHECK(G_sg_row.is_cuda() && G_sg_row.numel() == 1,
                "G_sg_row must be a CUDA tensor with one float element.");

    const int scratch_stride = D_scratch.size(1);
    const int col_fp4_stride = G_fp4_col.size(1);
    const int col_sc_kgroups = M / 64;
    const dim3 block(C::Nb);
    const dim3 grid((N + C::Nb - 1) / C::Nb, (M + (C::Mb / 2) - 1) / (C::Mb / 2));
    const auto stream = at::cuda::getCurrentCUDAStream();

    nvfp4_cce_backward_v3::backward_kernel_v3_col2pass_stage2<C><<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<const bf16*>(D_scratch.data_ptr<at::BFloat16>()),
        scratch_stride,
        G_sg_row.data_ptr<float>(),
        reinterpret_cast<uint8_t*>(G_fp4_col.data_ptr()),
        reinterpret_cast<uint8_t*>(G_sc_col.data_ptr()),
        col_fp4_stride,
        col_sc_kgroups,
        M,
        N,
        encode_centric);
    auto launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "Failed to launch NVFP4 experimental v3 col2pass stage2: ",
                cudaGetErrorString(launch_err));
}

template <typename C>
static void launch_experimental_backward_v3_fp4_col2pass(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    auto D_scratch = A.new_empty({M, N}, A.options().dtype(c10::kBFloat16));
    launch_experimental_backward_v3_fp4_col2pass_stage1<C>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        D_scratch,
        G_fp4_row, G_sc_row, G_sg_row,
        lse, targets,
        grad_scale, M, N, filter_eps, encode_centric);
    launch_experimental_backward_v3_fp4_col2pass_stage2<C>(
        D_scratch, G_sg_row, G_fp4_col, G_sc_col, M, N, encode_centric);
}

// Config instantiations
using bwd_v3_bf16_L4_SG8 = nvfp4_cce_backward_v3::config<4, 8, true, true>;
using bwd_v3_fp4_L4_SG8  = nvfp4_cce_backward_v3::config<4, 8, false, true>;
using bwd_v3_fp4_public_colwg_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_rowregs_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_rowregs<4, 8, true>;
using bwd_v3_fp4_public_colwg_rowregs_s3_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_rowregs_s3<4, 8, true>;
using bwd_v3_fp4_public_colwg_rowregs_s4_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_rowregs_s4<4, 8, true>;
using bwd_v3_fp4_public_colwg_aligned_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_aligned<4, 8, true>;
using bwd_v3_fp4_public_colwg_rowregs_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_rowregs_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_plainstage_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_plainstage<4, 8, true>;
using bwd_v3_fp4_public_colwg_bf16cache_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_bf16cache<4, 8, true>;
using bwd_v3_fp4_public_colwg_paircache_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_paircache<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowrecordregs_dualfloatcache_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowrecordregs_dualfloatcache_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_consumerotf_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_consumerotf<4, 8, true>;
using bwd_v3_fp4_public_colwg_consumerotf_warpbalanced_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_consumerotf_warpbalanced<4, 8, true>;
using bwd_v3_fp4_public_colwg_consumerotf_safused_warpbalanced_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_consumerotf_safused_warpbalanced<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colwaitskip_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colwaitskip<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colrecycleskip_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colrecycleskip<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colskipboth_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colskipboth<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut1_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<1, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut2_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<2, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut3_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<3, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut4_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<4, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut5_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<5, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut6_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<6, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut7_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<7, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut8_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut<8, 4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut1_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_trace<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut1_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut2_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_trace<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut2_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut5_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_trace<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut5_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut6_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_trace<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut6_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_collegacy_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_legacy_store<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_trace<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colbyte_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_byte_store<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colbytelegacy_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_legacy_store<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colbyte_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colnocache_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_debug_public_colonly_no_cache<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_debug_consumer_wg_sync_id<
        bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8, 3>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_quantizersync2_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_debug_quantizer_wg_sync_id<
        bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8, 2>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_L4_SG8 =
    nvfp4_cce_backward_v3::experimental_config_debug_quantizer_wg_sync_id<
        bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_L4_SG8, 2>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_rowonlydc_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_rowonlydc<4, 8, true, 4>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_dedicated_rowonly_combo_dc_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_dedicated_rowonly_combo_dc<4, 8, true, 4>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_E2_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag<4, 8, true, 2>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_lanepairrecord_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rcp_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowpair_rowrecord_rcp_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowfromcol_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowfromcol_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowregs_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowregs<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowregs_overlap_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowregs_overlap<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowleader_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowleader<4, 8, true>;
using bwd_v3_fp4_public_colwg_colpair_rowdual_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg_colpair_rowdual<4, 8, true>;
using bwd_v3_exp3wg_fp4_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_3wg<4, 8, true>;
using bwd_v3_exp3wg_fp4_L5_SG8 = nvfp4_cce_backward_v3::experimental_config_3wg<5, 8, true>;
using bwd_v3_exp3wg_fp4_L4_E2_SG8 = nvfp4_cce_backward_v3::experimental_config_3wg<4, 8, true, 2>;
using bwd_v3_expcolwg_fp4_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_colwg<4, 8, true>;
using bwd_v3_expcol2wg_fp4_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_col2wg<4, 8, true>;
using bwd_v3_exp4wg_fp4_L4_SG8 = nvfp4_cce_backward_v3::experimental_config_4wg<4, 8, true>;

static void launch_backward_v3_fp4_public_dispatch_L4_SG8(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    launch_experimental_backward_v3_fp4_3wg<
        bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_L4_SG8>(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        G_fp4_row, G_sc_row, G_sg_row,
        G_fp4_col, G_sc_col,
        lse, targets,
        grad_scale, M, N, filter_eps, encode_centric);
}

static void launch_backward_v3_fp4_public_enc_L4_SG8(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    // The public NV GEMM-facing FP4 ABI is fixed: explicit enc/dec modes in the
    // standalone quantizer produce identical row/col payloads, scale tensors,
    // and scalar globals. Keep the public v3 enc wrapper on that same contract
    // so backward_v3_fp4_enc_L4_SG8 matches the benchmarked nv_quant path.
    launch_backward_v3_fp4_public_dispatch_L4_SG8(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        G_fp4_row, G_sc_row, G_sg_row,
        G_fp4_col, G_sc_col,
        lse, targets,
        grad_scale, M, N, filter_eps, false);
}

static void launch_backward_v3_fp4_public_dec_L4_SG8(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    launch_backward_v3_fp4_public_dispatch_L4_SG8(
        A, A_sc, A_sc_global,
        B, B_sc, B_sc_global,
        G_fp4_row, G_sc_row, G_sg_row,
        G_fp4_col, G_sc_col,
        lse, targets,
        grad_scale, M, N, filter_eps, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v3_bf16_L4_SG8", &launch_backward_v3_bf16<bwd_v3_bf16_L4_SG8>,
          "NVFP4 CCE backward v3 (BF16 output) L4 SG8");
    // Public FP4 v3 stays on the shipped rowhwfp4 frontend until the newer
    // consumer-OTF variants recover both small-shape correctness and large-shape
    // stability from source.
    m.def("backward_v3_fp4_L4_SG8", &launch_backward_v3_fp4_public_dispatch_L4_SG8,
          "NVFP4 CCE backward v3 (FP4 output, consumer-row/col-WG) L4 SG8");
    m.def("backward_v3_fp4_enc_L4_SG8", &launch_backward_v3_fp4_public_enc_L4_SG8,
          "NVFP4 CCE backward v3 (FP4 encode-centric output, consumer-row/col-WG) L4 SG8");
    m.def("backward_v3_fp4_dec_L4_SG8", &launch_backward_v3_fp4_public_dec_L4_SG8,
          "NVFP4 CCE backward v3 (FP4 decode-centric output, consumer-row/col-WG) L4 SG8");
    m.def("experimental_backward_v3_fp4_L4_SG8", &launch_experimental_backward_v3_fp4<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_replayonly<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_rowonly<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_colonly<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_col2pass_stage1_L4_SG8", &launch_experimental_backward_v3_fp4_col2pass_stage1<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental col2pass stage1 (replay + row + BF16 scratch) L4 SG8");
    m.def("experimental_backward_v3_fp4_col2pass_stage2_L4_SG8", &launch_experimental_backward_v3_fp4_col2pass_stage2<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental col2pass stage2 (HBM BF16 to col FP4) L4 SG8");
    m.def("experimental_backward_v3_fp4_col2pass_L4_SG8", &launch_experimental_backward_v3_fp4_col2pass<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental col2pass fused (stage1 + stage2) L4 SG8");
    m.def("experimental_backward_v3_fp4_2ctaS_L4_SG8", &launch_experimental_backward_v3_fp4_2ctaS<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 2CTA-S (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_2ctaS_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_2ctaS_replayonly<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 2CTA-S replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_2ctaSdupB_L4_SG8", &launch_experimental_backward_v3_fp4_2ctaSdupB<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 2CTA-S duplicate-B (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_2ctaSdupB_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_2ctaSdupB_replayonly<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 2CTA-S duplicate-B replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_3wg_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_exp3wg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 3WG (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_3wg_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_exp3wg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 3WG replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_3wg_L5_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_exp3wg_fp4_L5_SG8>,
          "NVFP4 CCE backward v3 experimental 3WG (FP4 output) L5 SG8");
    m.def("experimental_backward_v3_fp4_3wg_replayonly_L5_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_exp3wg_fp4_L5_SG8>,
          "NVFP4 CCE backward v3 experimental 3WG replay-only (FP4 output) L5 SG8");
    m.def("experimental_backward_v3_fp4_3wg_L4_E2_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_exp3wg_fp4_L4_E2_SG8>,
          "NVFP4 CCE backward v3 experimental 3WG (FP4 output) L4 E2 SG8");
    m.def("experimental_backward_v3_fp4_3wg_replayonly_L4_E2_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_exp3wg_fp4_L4_E2_SG8>,
          "NVFP4 CCE backward v3 experimental 3WG replay-only (FP4 output) L4 E2 SG8");
    m.def("experimental_backward_v3_fp4_colwg_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_expcolwg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental consumer-row/col-WG (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_expcolwg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental consumer-row/col-WG replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_publiccolwg_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_L4_SG8>,
          "NVFP4 CCE backward v3 shipped public col-WG replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_publiccolwg_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_L4_SG8>,
          "NVFP4 CCE backward v3 shipped public col-WG row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_publiccolwg_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_L4_SG8>,
          "NVFP4 CCE backward v3 shipped public col-WG col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_rowregs_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_rowregs_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG row quant from registers (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_rowregs_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_rowregs_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG rowregs row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_rowregs_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_rowregs_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG rowregs col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_rowregs_s3_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_rowregs_s3_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG row quant from registers with 3 BF16 stages (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_rowregs_s4_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_rowregs_s4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG row quant from registers with 4 BF16 stages (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_rowregs_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_rowregs_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG rowregs + early-col-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_aligned_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_aligned_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG aligned quant fastpath (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG early-col-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_plainstage_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_plainstage_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG plain-stage col handoff (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_bf16cache_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_bf16cache_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG BF16-cache col quantization (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_paircache_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_paircache_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG BF16-pair-cache col quantization (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG transposed col-pair mailbox + packed col store (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + early-col-ready overlap on shared-row path (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + rowpair shared-row bounce removal (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord row-stage layout (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord warp-scope row-stage handoff (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord warp-scope row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord warp-scope col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair slot-major rowrecord warp-scope row-stage handoff (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair slot-major rowrecord warp-scope row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair slot-major rowrecord warp-scope col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord warp-scope producer-precomputed row rcp (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord warp-scope rowrcp row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord warp-scope rowrcp col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord warp-scope float-cache row consumer (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord float-cache row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord float-cache col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord dual float-cache consumer/quantizer (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord dual float-cache row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord dual float-cache col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowrecordregs_dualfloatcache_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowrecordregs_dualfloatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + producer-rowrecord register row quant (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowrecordregs_dualfloatcache_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowrecordregs_dualfloatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + producer-rowrecord register row quant row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowrecordregs_dualfloatcache_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowrecordregs_dualfloatcache_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + producer-rowrecord register row quant col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_consumerotf_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG true-OTF consumer-owned register epilogue (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_consumerotf_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG true-OTF consumer-owned register epilogue row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_consumerotf_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG true-OTF consumer-owned register epilogue col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_warpbalanced_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_consumerotf_warpbalanced_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG true-OTF warp-balanced consumer-owned register epilogue (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_warpbalanced_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_consumerotf_warpbalanced_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG true-OTF warp-balanced consumer-owned register epilogue row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_warpbalanced_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_consumerotf_warpbalanced_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG true-OTF warp-balanced consumer-owned register epilogue col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_safused_warpbalanced_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_consumerotf_safused_warpbalanced_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG SageAttention-style fused softmax+quant warp-balanced epilogue (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_safused_warpbalanced_diag_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_diag<bwd_v3_fp4_public_colwg_consumerotf_safused_warpbalanced_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG SageAttention-style fused softmax+quant warp-balanced epilogue with diag timing buffer (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_safused_warpbalanced_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_consumerotf_safused_warpbalanced_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG SageAttention-style fused softmax+quant warp-balanced row-only epilogue (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_consumerotf_safused_warpbalanced_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_consumerotf_safused_warpbalanced_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG SageAttention-style fused softmax+quant warp-balanced col-only epilogue (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord dual float-cache row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord dual float-cache row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord dual float-cache row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair lane-pair-record dual float-cache row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair lane-pair-record dual float-cache row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair lane-pair-record dual float-cache row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair padded-rowrecord dual float-cache row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair padded-rowrecord dual float-cache row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair padded-rowrecord dual float-cache row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair rowrecord dual float-cache row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair rowrecord dual float-cache row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair rowrecord dual float-cache row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row-helper row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row-helper row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row-helper row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 with consumer sync id forced to 3 (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with consumer sync id forced to 3 (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_quantizersync2_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_quantizersync2_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 with quantizer sync id forced to 2 (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_quantizersync2_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_quantizersync2_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with quantizer sync id forced to 2 (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 with consumer sync id 3 and quantizer sync id 2 (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with consumer sync id 3 and quantizer sync id 2 (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colwaitskip_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready overlap with col-ready wait skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colwaitskip_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready col-only with col-ready wait skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colrecycleskip_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready overlap with col-recycle wait skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colrecycleskip_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready col-only with col-recycle wait skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready overlap with both row waits skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready row-only with both row waits skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_rowskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready col-only with both row waits skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colskipboth_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready overlap with both col waits skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colskipboth_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record row hardware-fp4 row16-ready col-only with both col waits skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut1_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut1_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut after first row16-ready wait (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut2_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut2_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut after first row16 amax/scale contract (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut3_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut3_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut after first row16 FP4/sc writes (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut4_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut after first row16 completion (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut5_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut5_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut immediately after the shared/combo block (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut6_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut6_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut at the tail gate before the final quantizer sync (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut7_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut7_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut after the final quantizer tail sync (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut8_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colcut8_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only cut immediately after recycle publish (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut1_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut1_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only trace+cut after first row16-ready wait (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut2_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut2_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only trace+cut after first row16 amax/scale contract (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut5_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut5_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only trace+cut immediately after the shared/combo block (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut6_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltracecut6_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only trace+cut at the tail gate before the final quantizer sync (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_collegacy_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_collegacy_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with legacy direct col quant/store path restored (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with checkpoint trace prints (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colbyte_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colbyte_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with byte stores forced (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colbytelegacy_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colbytelegacy_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with byte stores plus legacy direct col quant/store path (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colnocache_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_colnocache_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public rowhwfp4 col-only with cached col values disabled (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_rowonlydc_L4_SG8", &launch_experimental_backward_v3_fp4_rowonly_combo_dc_5wg<
          bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_rowonlydc_L4_SG8,
          nvfp4_cce_backward_v3::globals_5wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_rowonlydc_L4_SG8>::COMBO_MODE_FULL>,
          "NVFP4 CCE backward v3 experimental row-only HBM materialization with one-pass combo dC back-half L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_dedicated_rowonly_combo_dc_L4_SG8", &launch_experimental_backward_v3_fp4_dedicated_rowonly_combo_dc_5wg<
          bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_dedicated_rowonly_combo_dc_L4_SG8,
          nvfp4_cce_backward_v3::globals_5wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_dedicated_rowonly_combo_dc_L4_SG8>::COMBO_MODE_DCONLY>,
          "NVFP4 CCE backward v3 experimental dedicated row-only frontend with on-chip combo dC back-half L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record producer row-rcp row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record producer row-rcp row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record producer row-rcp row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready epipipe (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready epipipe row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready epipipe col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready with dedicated row+col quantizer WGs (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready dedicated row+col quantizer WGs replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready dedicated row quantizer WG row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready dedicated col quantizer WG col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with dedicated row+col quantizer WGs (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache dedicated row+col quantizer WGs replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache dedicated row quantizer WG row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache dedicated col quantizer WG col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work and skipped row-ready wait (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work and skipped row-ready wait replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work and skipped row-ready wait row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row quantizer work and skipped row-ready wait col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work and skipped col-ready wait (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work and skipped col-ready wait replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work and skipped col-ready wait row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed col quantizer work and skipped col-ready wait col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped row-recycle wait (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped row-recycle wait replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped row-recycle wait row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped row-recycle wait col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped col-recycle wait (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped col-recycle wait replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped col-recycle wait row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with skipped col-recycle wait col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache dedicated row and col quantizer WGs without early col-ready (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache dedicated row and col quantizer WGs without early col-ready replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache dedicated row quantizer WG without early col-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache dedicated col quantizer WG without early col-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with row WG waiting on col-ready (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with row WG waiting on col-ready replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with row WG waiting on col-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with row WG waiting on col-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer col stage production (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer col stage production replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer col stage production row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer col stage production col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work, disabled consumer col stage production, and both row waits skipped (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work, disabled consumer col stage production, and both row waits skipped replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work, disabled consumer col stage production, and both row waits skipped row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_rowskipboth_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work, disabled consumer col stage production, and both row waits skipped col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer row stage production (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer row stage production replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer row stage production row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and disabled consumer row stage production col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and row WG waiting on col-ready (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and row WG waiting on col-ready replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and row WG waiting on col-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache with stubbed row and col quantizer work and row WG waiting on col-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready row-WG diagnostic replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready row-WG diagnostic fused (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready row-WG diagnostic row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair BF16-tile row-WG diagnostic replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair BF16-tile row-WG diagnostic row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public row-pair-only row-WG diagnostic replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public row-pair-only row-WG diagnostic fused (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public row-pair-only row-WG diagnostic row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready immediate-row overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready immediate-row row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready immediate-row col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready sequential col-then-row (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready sequential row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready sequential col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready col-done-row16 overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready col-done-row16 row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG padded-colpair+rowpair lane-pair-record dual float-cache row16-ready col-done-row16 col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair packed-rowstage dual float-cache row16-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair packed-rowstage dual float-cache row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair packed-rowstage dual float-cache row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord row16-granular col-ready overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord row16-ready row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord row16-ready col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_lanepairrecord_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair lane-pair-record row-stage layout (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_lanepairrecord_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair lane-pair-record row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_lanepairrecord_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_lanepairrecord_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair lane-pair-record col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rcp_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rcp_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord producer-precomputed row rcp (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rcp_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rcp_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord+rowrcp row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowpair_rowrecord_rcp_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowpair_rowrecord_rcp_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpair rowrecord+rowrcp col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowfromcol_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowfromcol_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair single-stage row-from-col overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowfromcol_overlap_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowfromcol_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair row-from-col row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowfromcol_overlap_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowfromcol_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair row-from-col col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowregs_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowregs_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + rowpairs row quant from registers (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowregs_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowregs_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpairs row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowregs_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowregs_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpairs col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowregs_overlap_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowregs_overlap_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowpairs early overlap (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowleader_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowleader_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + row-leader register row quant (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowleader_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowleader_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowleader row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowleader_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowleader_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowleader col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowdual_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_fp4_public_colwg_colpair_rowdual_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair + 2-lane row-owner register row quant (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowdual_rowonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_rowonly<bwd_v3_fp4_public_colwg_colpair_rowdual_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowdual row-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_colwg_colpair_rowdual_colonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_colonly<bwd_v3_fp4_public_colwg_colpair_rowdual_L4_SG8>,
          "NVFP4 CCE backward v3 experimental public col-WG colpair+rowdual col-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_col2wg_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_expcol2wg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental consumer-row/two-col-WG (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_col2wg_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_expcol2wg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental consumer-row/two-col-WG replay-only (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_4wg_L4_SG8", &launch_experimental_backward_v3_fp4_3wg<bwd_v3_exp4wg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 4WG (FP4 output) L4 SG8");
    m.def("experimental_backward_v3_fp4_4wg_replayonly_L4_SG8", &launch_experimental_backward_v3_fp4_3wg_replayonly<bwd_v3_exp4wg_fp4_L4_SG8>,
          "NVFP4 CCE backward v3 experimental 4WG replay-only (FP4 output) L4 SG8");
}
#endif
