#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

#include "pyutils/torchutils.cuh"
#include "nvfp4_localcta_kernel.cuh"
#include "nvfp4_localcta_batched_kernel.cuh"
#include "../nvfp4_accum_gemm.cuh"
#include "../nvfp4_gemm.cuh"
#include "../nvfp4_batched_gemm.cuh"
#include "../nvfp4_split2_accum_gemm.cuh"
#include "../nvfp4_split3_accum_gemm.cuh"

namespace {

constexpr int kScaleBytesPerTile = 512;
using localcta_regular_smalln_config = nvfp4_localcta_gemm::config<128, 5, 4, 12, 2, true, 256, true, 2, 128>;
using localcta_regular_smallk_config = nvfp4_localcta_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_regular_largek_config = nvfp4_localcta_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 128>;
using localcta_parity_config = nvfp4_localcta_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_fast_smallk_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
using localcta_fast_largek_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false>;
using localcta_fast_grouped_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
using localcta_fast_chunkgrid_smallk_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_fast_chunkgrid_largek_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 128>;
using localcta_fast_chunkgrid_grouped_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_fast_batched_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
using localcta_fast_split2_dgrad_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false>;
using localcta_fast_split3_dgrad_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false>;
using localcta_onepass_cfg0 = nvfp4_gemm::config<128, 5, 4, 12, 2, true, 256, false, 1>;
using localcta_onepass_cfg1 = nvfp4_gemm::config<128, 5, 4, 12, 2, true, 256, false, 2>;
using localcta_onepass_cfg2 = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 1>;
using localcta_onepass_cfg3 = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
using localcta_onepass_cfg4 = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, false, 1>;
using localcta_onepass_cfg5 = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, false, 2>;
using localcta_onepass_sg_cfg1 = nvfp4_gemm::config<128, 5, 4, 12, 2, true, 256, false, 2, 128>;
using localcta_onepass_sg_cfg3 = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2, 128>;
using localcta_onepass_sg_cfg5 = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, false, 2, 128>;

__global__ void sum3_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
);

__global__ void sum_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
);

void check_fp4_matrix(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat4_e2m1fn_x2, name, " must be fp4x2");
}

void check_scale_tensor(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == at::kFloat8_e4m3fn, name, " must be fp8 e4m3");
    TORCH_CHECK(t.size(0) == rows / 128, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols / 64, name, " second dim mismatch");
    TORCH_CHECK(t.size(2) == kScaleBytesPerTile, name, " third dim must be 512");
}

void check_scale_tensor_tma_compatible(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == at::kFloat8_e4m3fn, name, " must be fp8 e4m3");
    TORCH_CHECK(t.size(0) == rows / 128, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols / 64, name, " second dim mismatch");
    TORCH_CHECK(t.size(2) == kScaleBytesPerTile, name, " third dim must be 512");
    TORCH_CHECK(t.stride(2) == 1, name, " last dim must be contiguous");
    TORCH_CHECK(t.stride(1) == kScaleBytesPerTile,
                name, " second-dim stride must equal 512 bytes");
    TORCH_CHECK(t.stride(0) >= t.size(1) * kScaleBytesPerTile,
                name, " leading stride is too small for a prepared-scale view");
    const auto data_ptr = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK((data_ptr & 0xF) == 0, name, " data pointer must be 16-byte aligned");
    TORCH_CHECK((t.stride(1) % 16) == 0, name, " second-dim stride must be 16-byte aligned");
    TORCH_CHECK((t.stride(0) % 16) == 0, name, " leading stride must be 16-byte aligned");
}

template <typename ST>
void encode_prepared_scale_tensor_map(CUtensorMap* desc, const at::Tensor& t, const char* name) {
    static_assert(std::is_same_v<typename ST::dtype, kittens::half>,
                  "prepared scale TMA helper assumes half logical elements");
    static_assert(!ST::swizzle, "prepared scale TMA helper only supports non-swizzled tiles");
    check_scale_tensor_tma_compatible(t, name, t.size(0) * 128, t.size(1) * 64);

    constexpr uint64_t logical_cols = kScaleBytesPerTile / sizeof(typename ST::dtype);
    uint64_t gmem_shape[4] = {
        logical_cols,
        static_cast<uint64_t>(t.size(1)),
        static_cast<uint64_t>(t.size(0)),
        1,
    };
    uint64_t gmem_stride[3] = {
        static_cast<uint64_t>(t.stride(1)),
        static_cast<uint64_t>(t.stride(0)),
        static_cast<uint64_t>(t.size(0) * t.stride(0)),
    };
    uint32_t smem_shape[4] = {
        static_cast<uint32_t>(ST::cols),
        static_cast<uint32_t>(ST::rows),
        1, 1,
    };
    uint32_t smem_stride[4] = {1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        4,
        t.data_ptr(),
        gmem_shape,
        gmem_stride,
        smem_shape,
        smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, name, " TMA creation failed");
}

void check_chunk_grid(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == rows / 128, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols / 128, name, " second dim mismatch");
}

void check_output_matrix(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.size(0) == rows && t.size(1) == cols, name, " shape mismatch");
}

void check_fast_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(0) % 128 == 0 && B.size(0) % 128 == 0, "M and N must be multiples of 128");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share packed K");
    TORCH_CHECK((A.size(1) * 2) % 128 == 0, "K must be a multiple of 128");
    check_scale_tensor(A_sc_prepared, "A_sc_prepared", A.size(0), A.size(1) * 2);
    check_scale_tensor(B_sc_prepared, "B_sc_prepared", B.size(0), B.size(1) * 2);
    kittens::py::device_check(A, A_sc_prepared, B, B_sc_prepared);
}

void check_outer_sg_tensor(const at::Tensor& sg, const char* name, int64_t tiles) {
    TORCH_CHECK(sg.defined() && sg.is_cuda() && sg.scalar_type() == at::kFloat,
                name, " must be a CUDA float32 tensor");
    TORCH_CHECK(sg.dim() == 1 || sg.dim() == 2,
                name, " must be rank-1 or rank-2 outer SG");
    if (sg.dim() == 1) {
        TORCH_CHECK(sg.size(0) == tiles, name, " shape mismatch");
        return;
    }
    TORCH_CHECK((sg.size(0) == tiles && sg.size(1) == 1) ||
                (sg.size(0) == 1 && sg.size(1) == tiles),
                name, " must have shape [tiles], [tiles, 1], or [1, tiles]");
}

int outer_sg_stride(const at::Tensor& sg, int64_t tiles) {
    if (sg.dim() == 1) {
        TORCH_CHECK(sg.stride(0) == 1, "outer SG vector must be contiguous");
        return 1;
    }
    if (sg.size(0) == tiles) {
        return static_cast<int>(sg.stride(0));
    }
    return static_cast<int>(sg.stride(1));
}

void check_fast_gemm_outer_sg_inputs(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg
) {
    check_fast_gemm_inputs(A, A_sc, B, B_sc);
    TORCH_CHECK(A.size(0) % 256 == 0 && B.size(0) % 256 == 0,
                "outer SG fast GEMM requires M and N multiples of 256");
    check_outer_sg_tensor(A_sg, "A_sg", A.size(0) / 256);
    check_outer_sg_tensor(B_sg, "B_sg", B.size(0) / 256);
    kittens::py::device_check(A, A_sc, A_sg, B, B_sc, B_sg);
}

at::Tensor get_unit_scale_tensor(const at::Tensor& ref) {
    static thread_local std::vector<at::Tensor> cache;
    const int device_index = ref.get_device();
    if (device_index >= static_cast<int>(cache.size())) {
        cache.resize(device_index + 1);
    }
    if (!cache[device_index].defined()) {
        cache[device_index] = torch::ones({1}, torch::dtype(torch::kFloat32).device(ref.device()));
    }
    return cache[device_index];
}

void check_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(0) % 128 == 0 && B.size(0) % 128 == 0, "M and N must be multiples of 128");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share packed K");
    TORCH_CHECK((A.size(1) * 2) % 128 == 0, "K must be a multiple of 128");
    check_scale_tensor(A_sc, "A_sc", A.size(0), A.size(1) * 2);
    check_scale_tensor(B_sc, "B_sc", B.size(0), B.size(1) * 2);
    check_chunk_grid(A_sg_chunks, "A_sg_chunks", A.size(0), A.size(1) * 2);
    check_chunk_grid(B_sg_chunks, "B_sg_chunks", B.size(0), B.size(1) * 2);
    kittens::py::device_check(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
}

template <typename C>
void launch_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    using G = nvfp4_localcta_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sg_chunks = A_sg_chunks.data_ptr<float>(),
        .A_sg_stride = static_cast<int>(A_sg_chunks.size(1)),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sg_chunks = B_sg_chunks.data_ptr<float>(),
        .B_sg_stride = static_cast<int>(B_sg_chunks.size(1)),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .v_dim = 0,
        .use_split_D = false,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel<C>>(g);
}

template <typename C>
void launch_grouped_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    using G = nvfp4_localcta_gemm::globals<C>;
    const bool use_split_D = D_K_opt.has_value();
    const int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sg_chunks = A_sg_chunks.data_ptr<float>(),
        .A_sg_stride = static_cast<int>(A_sg_chunks.size(1)),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sg_chunks = B_sg_chunks.data_ptr<float>(),
        .B_sg_stride = static_cast<int>(B_sg_chunks.size(1)),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                           : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                   : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                  : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = use_split_D ? v_dim : 0,
        .use_split_D = use_split_D,
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel<C>>(g);
}

void launch_regular_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    const int64_t M = D.size(0);
    const int64_t K = A.size(1) * 2;
    const int64_t N = D.size(1);
    if (K <= 2048 && M <= 1024 && N <= 2048) {
        launch_gemm_with_config<localcta_regular_smalln_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    } else if (K <= 2048) {
        launch_gemm_with_config<localcta_regular_smallk_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    } else {
        launch_gemm_with_config<localcta_regular_largek_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    }
}

void launch_grouped_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    launch_grouped_gemm_with_config<localcta_parity_config>(
        A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks,
        D, D_K_opt, D_V_opt, silu_dim);
}

template <typename C>
void launch_fast_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    at::Tensor& D
) {
    using G = nvfp4_gemm::globals<C>;
    auto one = get_unit_scale_tensor(A);
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_prepared, 1, A_sc_prepared.size(0), A_sc_prepared.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(one),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared, 1, B_sc_prepared.size(0), B_sc_prepared.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(one),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .v_dim = 0,
        .use_split_D = false,
        .a_sg_per_tile = nullptr,
        .a_sg_stride = 0,
        .b_sg_per_tile = nullptr,
        .b_sg_stride = 0,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
void launch_fast_grouped_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    using G = nvfp4_gemm::globals<C>;
    const bool use_split_D = D_K_opt.has_value();
    const int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;
    auto one = get_unit_scale_tensor(A);

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_prepared, 1, A_sc_prepared.size(0), A_sc_prepared.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(one),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared, 1, B_sc_prepared.size(0), B_sc_prepared.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(one),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                           : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                   : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                  : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = use_split_D ? v_dim : 0,
        .use_split_D = use_split_D,
        .a_sg_per_tile = nullptr,
        .a_sg_stride = 0,
        .b_sg_per_tile = nullptr,
        .b_sg_stride = 0,
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
void launch_fast_gemm_outer_sg_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    at::Tensor& D
) {
    using G = nvfp4_gemm::globals<C>;
    auto one = get_unit_scale_tensor(A);
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(one),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(one),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .v_dim = 0,
        .use_split_D = false,
        .a_sg_per_tile = A_sg.data_ptr<float>(),
        .a_sg_stride = outer_sg_stride(A_sg, A.size(0) / 256),
        .b_sg_per_tile = B_sg.data_ptr<float>(),
        .b_sg_stride = outer_sg_stride(B_sg, B.size(0) / 256),
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
void launch_fast_grouped_gemm_outer_sg_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    using G = nvfp4_gemm::globals<C>;
    const bool use_split_D = D_K_opt.has_value();
    const int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;
    auto one = get_unit_scale_tensor(A);

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(one),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(one),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                           : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                   : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                  : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = use_split_D ? v_dim : 0,
        .use_split_D = use_split_D,
        .a_sg_per_tile = A_sg.data_ptr<float>(),
        .a_sg_stride = outer_sg_stride(A_sg, A.size(0) / 256),
        .b_sg_per_tile = B_sg.data_ptr<float>(),
        .b_sg_stride = outer_sg_stride(B_sg, B.size(0) / 256),
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
void launch_fast_gemm_sg_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    using G = nvfp4_gemm::globals<C>;
    auto one = get_unit_scale_tensor(A);
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(one),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(one),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .v_dim = 0,
        .use_split_D = false,
        .a_sg_per_tile = A_sg_chunks.data_ptr<float>(),
        .a_sg_stride = -static_cast<int>(A_sg_chunks.size(1)),
        .b_sg_per_tile = B_sg_chunks.data_ptr<float>(),
        .b_sg_stride = -static_cast<int>(B_sg_chunks.size(1)),
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel_chunk_grid<C>>(g);
}

template <typename C>
void launch_fast_grouped_gemm_sg_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    using G = nvfp4_gemm::globals<C>;
    const bool use_split_D = D_K_opt.has_value();
    const int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;
    auto one = get_unit_scale_tensor(A);

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(one),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(one),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                           : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                   : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                  : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = use_split_D ? v_dim : 0,
        .use_split_D = use_split_D,
        .a_sg_per_tile = A_sg_chunks.data_ptr<float>(),
        .a_sg_stride = -static_cast<int>(A_sg_chunks.size(1)),
        .b_sg_per_tile = B_sg_chunks.data_ptr<float>(),
        .b_sg_stride = -static_cast<int>(B_sg_chunks.size(1)),
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel_chunk_grid<C>>(g);
}

void launch_fast_regular_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    at::Tensor& D
) {
    const int64_t K = A.size(1) * 2;
    if (K <= 2048) {
        launch_fast_gemm_with_config<localcta_fast_smallk_config>(
            A, A_sc_prepared, B, B_sc_prepared, D);
    } else {
        launch_fast_gemm_with_config<localcta_fast_largek_config>(
            A, A_sc_prepared, B, B_sc_prepared, D);
    }
}

void launch_fast_grouped_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    const bool use_split_D = D_K_opt.has_value();
    const int64_t reduction_k = A.size(1) * 2;
    if (!use_split_D && reduction_k > 2048) {
        launch_fast_grouped_gemm_with_config<localcta_fast_largek_config>(
            A, A_sc_prepared, B, B_sc_prepared, D, D_K_opt, D_V_opt, silu_dim);
        return;
    }
    launch_fast_grouped_gemm_with_config<localcta_fast_grouped_config>(
        A, A_sc_prepared, B, B_sc_prepared, D, D_K_opt, D_V_opt, silu_dim);
}

void launch_fast_regular_gemm_outer_sg(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    at::Tensor& D
) {
    const int64_t K = A.size(1) * 2;
    if (K <= 2048) {
        launch_fast_gemm_outer_sg_with_config<localcta_fast_smallk_config>(
            A, A_sc, A_sg, B, B_sc, B_sg, D);
    } else {
        launch_fast_gemm_outer_sg_with_config<localcta_fast_largek_config>(
            A, A_sc, A_sg, B, B_sc, B_sg, D);
    }
}

void launch_fast_grouped_gemm_outer_sg(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    const bool use_split_D = D_K_opt.has_value();
    const int64_t reduction_k = A.size(1) * 2;
    if (!use_split_D && reduction_k > 2048) {
        launch_fast_grouped_gemm_outer_sg_with_config<localcta_fast_largek_config>(
            A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim);
        return;
    }
    launch_fast_grouped_gemm_outer_sg_with_config<localcta_fast_grouped_config>(
        A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim);
}

void launch_fast_regular_gemm_sg(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    const int64_t K = A.size(1) * 2;
    if (K <= 2048) {
        launch_fast_gemm_sg_with_config<localcta_fast_chunkgrid_smallk_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    } else {
        launch_fast_gemm_sg_with_config<localcta_fast_chunkgrid_largek_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    }
}

void launch_fast_grouped_gemm_sg(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    const bool use_split_D = D_K_opt.has_value();
    const int64_t reduction_k = A.size(1) * 2;
    if (!use_split_D && reduction_k > 2048) {
        launch_fast_grouped_gemm_sg_with_config<localcta_fast_chunkgrid_largek_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D, D_K_opt, D_V_opt, silu_dim);
        return;
    }
    launch_fast_grouped_gemm_sg_with_config<localcta_fast_chunkgrid_grouped_config>(
        A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D, D_K_opt, D_V_opt, silu_dim);
}

void check_batched_inputs(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list
) {
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n > 0, "batched GEMM requires at least one batch");
    TORCH_CHECK(n == static_cast<int>(A_sc_list.size()) &&
                n == static_cast<int>(A_sg_chunks_list.size()) &&
                n == static_cast<int>(B_list.size()) &&
                n == static_cast<int>(B_sc_list.size()) &&
                n == static_cast<int>(B_sg_chunks_list.size()),
                "all batched input lists must have the same length");
    for (int i = 0; i < n; ++i) {
        check_gemm_inputs(
            A_list[i], A_sc_list[i], A_sg_chunks_list[i],
            B_list[i], B_sc_list[i], B_sg_chunks_list[i]);
    }
}

template <typename AList, typename BList>
void check_batched_shape_compatibility(
    const AList& A_list,
    const BList& B_list,
    const std::vector<at::Tensor>& D_list
) {
    const int64_t M = A_list[0].size(0);
    const int64_t N = B_list[0].size(0);
    const int64_t K_packed = A_list[0].size(1);
    for (size_t i = 0; i < A_list.size(); ++i) {
        TORCH_CHECK(A_list[i].size(0) == M, "all batched A tensors must share M");
        TORCH_CHECK(B_list[i].size(0) == N, "all batched B tensors must share N");
        TORCH_CHECK(A_list[i].size(1) == K_packed && B_list[i].size(1) == K_packed,
                    "all batched tensors must share packed K");
        TORCH_CHECK(D_list[i].size(0) == M && D_list[i].size(1) == N,
                    "all batched outputs must share the same shape");
    }
}

void check_fast_batched_inputs(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list
) {
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n > 0, "batched GEMM requires at least one batch");
    TORCH_CHECK(n == static_cast<int>(A_sc_prepared_list.size()) &&
                n == static_cast<int>(B_list.size()) &&
                n == static_cast<int>(B_sc_prepared_list.size()),
                "all fast batched input lists must have the same length");
    for (int i = 0; i < n; ++i) {
        check_fast_gemm_inputs(A_list[i], A_sc_prepared_list[i], B_list[i], B_sc_prepared_list[i]);
    }
}

void check_fast_batched_outer_sg_inputs(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_list
) {
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n > 0, "batched outer-SG GEMM requires at least one batch");
    TORCH_CHECK(n == static_cast<int>(A_sc_list.size()) &&
                n == static_cast<int>(A_sg_list.size()) &&
                n == static_cast<int>(B_list.size()) &&
                n == static_cast<int>(B_sc_list.size()) &&
                n == static_cast<int>(B_sg_list.size()),
                "all batched outer-SG input lists must have the same length");
    for (int i = 0; i < n; ++i) {
        check_fast_gemm_outer_sg_inputs(
            A_list[i], A_sc_list[i], A_sg_list[i],
            B_list[i], B_sc_list[i], B_sg_list[i]);
    }
}

void check_fast_batched_strided_inputs(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list
) {
    check_fp4_matrix(A_full, "A_full");
    TORCH_CHECK(A_full.size(0) % 128 == 0, "A_full M must be a multiple of 128");
    TORCH_CHECK((A_full.size(1) * 2) % 128 == 0, "A_full K must be a multiple of 128");

    const int n = static_cast<int>(A_sc_prepared_list.size());
    TORCH_CHECK(n > 0, "strided batched accum requires at least one batch");
    TORCH_CHECK(n == static_cast<int>(A_col_offsets.size()) &&
                n == static_cast<int>(A_col_widths.size()) &&
                n == static_cast<int>(B_list.size()) &&
                n == static_cast<int>(B_sc_prepared_list.size()),
                "all strided batched accum inputs must have the same length");

    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets must be non-negative");
        TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths must be positive");
        TORCH_CHECK(A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
                    "A_full slice exceeds packed width");
        TORCH_CHECK((A_col_widths[i] * 2) % 128 == 0,
                    "split widths must be multiples of 128");
        check_scale_tensor(A_sc_prepared_list[i], "A_sc_prepared_list[i]",
                           A_full.size(0), A_col_widths[i] * 2);
        check_fp4_matrix(B_list[i], "B_list[i]");
        TORCH_CHECK(B_list[i].size(1) == A_col_widths[i],
                    "B_list packed K must match A_col_widths");
        TORCH_CHECK(B_list[i].size(0) % 128 == 0,
                    "B_list rows must be multiples of 128");
        check_scale_tensor(B_sc_prepared_list[i], "B_sc_prepared_list[i]",
                           B_list[i].size(0), B_list[i].size(1) * 2);
        kittens::py::device_check(
            A_full, A_sc_prepared_list[i], B_list[i], B_sc_prepared_list[i]);
    }
}

void check_fast_batched_strided_inputs_allow_a_sc_views(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list
) {
    check_fp4_matrix(A_full, "A_full");
    TORCH_CHECK(A_full.size(0) % 128 == 0, "A_full M must be a multiple of 128");
    TORCH_CHECK((A_full.size(1) * 2) % 128 == 0, "A_full K must be a multiple of 128");

    const int n = static_cast<int>(A_sc_prepared_list.size());
    TORCH_CHECK(n > 0, "strided batched accum requires at least one batch");
    TORCH_CHECK(n == static_cast<int>(A_col_offsets.size()) &&
                n == static_cast<int>(A_col_widths.size()) &&
                n == static_cast<int>(B_list.size()) &&
                n == static_cast<int>(B_sc_prepared_list.size()),
                "all strided batched accum inputs must have the same length");

    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets must be non-negative");
        TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths must be positive");
        TORCH_CHECK(A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
                    "A_full slice exceeds packed width");
        TORCH_CHECK((A_col_widths[i] * 2) % 128 == 0,
                    "split widths must be multiples of 128");
        check_scale_tensor_tma_compatible(A_sc_prepared_list[i], "A_sc_prepared_list[i]",
                                          A_full.size(0), A_col_widths[i] * 2);
        check_fp4_matrix(B_list[i], "B_list[i]");
        TORCH_CHECK(B_list[i].size(1) == A_col_widths[i],
                    "B_list packed K must match A_col_widths");
        TORCH_CHECK(B_list[i].size(0) % 128 == 0,
                    "B_list rows must be multiples of 128");
        check_scale_tensor(B_sc_prepared_list[i], "B_sc_prepared_list[i]",
                           B_list[i].size(0), B_list[i].size(1) * 2);
        kittens::py::device_check(
            A_full, A_sc_prepared_list[i], B_list[i], B_sc_prepared_list[i]);
    }
}

template <typename C>
void launch_batched_gemm_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    using G = nvfp4_localcta_batched_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_list.size());
    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    g_host.num_red_blocks = static_cast<int>((2 * A_list[0].size(1)) / C::Kb);

    for (int i = 0; i < n; ++i) {
        auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_list[i], 1, A_sc_list[i].size(0), A_sc_list[i].size(1), 256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_list[i], 1, B_sc_list[i].size(0), B_sc_list[i].size(1), 256);
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        g_host.A_sg_chunks[i] = A_sg_chunks_list[i].data_ptr<float>();
        g_host.A_sg_stride[i] = static_cast<int>(A_sg_chunks_list[i].size(1));
        g_host.B_sg_chunks[i] = B_sg_chunks_list[i].data_ptr<float>();
        g_host.B_sg_stride[i] = static_cast<int>(B_sg_chunks_list[i].size(1));
    }

    kittens::py::launch_kernel<C, G, nvfp4_localcta_batched_gemm::kernel<C>>(g_host);
}

void launch_batched_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    launch_batched_gemm_with_config<localcta_parity_config>(
        A_list, A_sc_list, A_sg_chunks_list, B_list, B_sc_list, B_sg_chunks_list, D_list);
}

template <typename C>
void launch_fast_batched_gemm_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    std::vector<at::Tensor>& D_list
) {
    using G = nvfp4_batched_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_list.size());
    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    auto one = get_unit_scale_tensor(A_list[0]);
    const float* one_ptr = one.data_ptr<float>();

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    g_host.num_red_blocks = static_cast<int>((2 * A_list[0].size(1)) / C::Kb);

    for (int i = 0; i < n; ++i) {
        auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_prepared_list[i], 1, A_sc_prepared_list[i].size(0), A_sc_prepared_list[i].size(1), 256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared_list[i], 1, B_sc_prepared_list[i].size(0), B_sc_prepared_list[i].size(1), 256);
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        g_host.A_sg[i] = one_ptr;
        g_host.B_sg[i] = one_ptr;
    }

    kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
}

void launch_fast_batched_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    std::vector<at::Tensor>& D_list
) {
    const int spatial_tiles =
        static_cast<int>((D_list[0].size(0) / localcta_fast_batched_config::Mb) *
                         (D_list[0].size(1) / localcta_fast_batched_config::Nb));

    // Keep the old per-batch path only for tiny launches that cannot form a full cluster.
    if (spatial_tiles < localcta_fast_batched_config::CLUSTER_SIZE) {
        for (size_t i = 0; i < A_list.size(); ++i) {
            launch_fast_regular_gemm(
                A_list[i], A_sc_prepared_list[i],
                B_list[i], B_sc_prepared_list[i],
                D_list[i]);
        }
        return;
    }

    launch_fast_batched_gemm_with_config<localcta_fast_batched_config>(
        A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_list);
}

template <typename C>
void launch_fast_batched_accum_gemm_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    using G = nvfp4_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_list.size());
    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    auto one = get_unit_scale_tensor(A_list[0]);
    const float* one_ptr = one.data_ptr<float>();

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    g_host.num_red_blocks = static_cast<int>((2 * A_list[0].size(1)) / C::Kb);
    const int num_tiles = g_host.num_row_blocks * 2 * g_host.num_col_blocks;

    static thread_local std::vector<at::Tensor> tile_done_cache;
    const int device_index = A_list[0].get_device();
    if (device_index >= static_cast<int>(tile_done_cache.size())) {
        tile_done_cache.resize(device_index + 1);
    }
    auto& tile_done_buf = tile_done_cache[device_index];
    if (!tile_done_buf.defined() || tile_done_buf.numel() < num_tiles) {
        tile_done_buf = torch::zeros({num_tiles}, torch::dtype(torch::kInt32).device(A_list[0].device()));
    } else {
        tile_done_buf.narrow(0, 0, num_tiles).zero_();
    }
    g_host.tile_done = tile_done_buf.data_ptr<int>();

    for (int i = 0; i < n; ++i) {
        auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_prepared_list[i], 1,
            A_sc_prepared_list[i].dim() == 2 ? A_sc_prepared_list[i].size(0) / 128 : A_sc_prepared_list[i].size(0),
            A_sc_prepared_list[i].dim() == 2 ? A_sc_prepared_list[i].size(1) / 4 : A_sc_prepared_list[i].size(1),
            256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared_list[i], 1,
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(0) / 128 : B_sc_prepared_list[i].size(0),
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(1) / 4 : B_sc_prepared_list[i].size(1),
            256);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        g_host.A_sg[i] = one_ptr;
        g_host.B_sg[i] = one_ptr;
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_batched_accum_gemm_outer_sg_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_list,
    at::Tensor& D_out
) {
    using G = nvfp4_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_list.size());
    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    g_host.num_red_blocks = static_cast<int>((2 * A_list[0].size(1)) / C::Kb);
    const int num_tiles = g_host.num_row_blocks * 2 * g_host.num_col_blocks;

    static thread_local std::vector<at::Tensor> tile_done_cache;
    const int device_index = A_list[0].get_device();
    if (device_index >= static_cast<int>(tile_done_cache.size())) {
        tile_done_cache.resize(device_index + 1);
    }
    auto& tile_done_buf = tile_done_cache[device_index];
    if (!tile_done_buf.defined() || tile_done_buf.numel() < num_tiles) {
        tile_done_buf = torch::zeros({num_tiles}, torch::dtype(torch::kInt32).device(A_list[0].device()));
    } else {
        tile_done_buf.narrow(0, 0, num_tiles).zero_();
    }
    g_host.tile_done = tile_done_buf.data_ptr<int>();

    for (int i = 0; i < n; ++i) {
        auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_list[i], 1,
            A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0) / 128 : A_sc_list[i].size(0),
            A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1) / 4 : A_sc_list[i].size(1),
            256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_list[i], 1,
            B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0) / 128 : B_sc_list[i].size(0),
            B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1) / 4 : B_sc_list[i].size(1),
            256);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
        g_host.A_sg_stride[i] = outer_sg_stride(A_sg_list[i], A_list[i].size(0) / 256);
        g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        g_host.B_sg_stride[i] = outer_sg_stride(B_sg_list[i], B_list[i].size(0) / 256);
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_batched_accum_gemm_strided_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    at::Tensor& D_out
) {
    using G = nvfp4_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_sc_prepared_list.size());
    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const int64_t max_fp4_cols =
        *std::max_element(A_col_widths.begin(), A_col_widths.end());
    auto one = get_unit_scale_tensor(A_full);
    const float* one_ptr = one.data_ptr<float>();

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    g_host.num_red_blocks = static_cast<int>((2 * max_fp4_cols) / C::Kb);

    const int num_tiles = g_host.num_row_blocks * 2 * g_host.num_col_blocks;
    static thread_local std::vector<at::Tensor> tile_done_cache;
    const int device_index = A_full.get_device();
    if (device_index >= static_cast<int>(tile_done_cache.size())) {
        tile_done_cache.resize(device_index + 1);
    }
    auto& tile_done_buf = tile_done_cache[device_index];
    if (!tile_done_buf.defined() || tile_done_buf.numel() < num_tiles) {
        tile_done_buf = torch::zeros({num_tiles}, torch::dtype(torch::kInt32).device(A_full.device()));
    } else {
        tile_done_buf.narrow(0, 0, num_tiles).zero_();
    }
    g_host.tile_done = tile_done_buf.data_ptr<int>();

    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    for (int i = 0; i < n; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        const int64_t fp4_offset = A_col_offsets[i];
        const void* data_ptr = a_base + fp4_offset;

        uint64_t gmem_shape[5] = {
            static_cast<uint64_t>(swizzle_elements),
            static_cast<uint64_t>(M),
            static_cast<uint64_t>((fp4_cols + swizzle_elements - 1) / swizzle_elements),
            1, 1
        };
        uint64_t gmem_stride[4] = {
            static_cast<uint64_t>(a_full_row_stride),
            128,
            static_cast<uint64_t>(M * a_full_row_stride),
            static_cast<uint64_t>(M * a_full_row_stride)
        };
        uint32_t smem_shape[5] = {
            static_cast<uint32_t>(swizzle_elements),
            static_cast<uint32_t>(C::Mb / 2),
            1, 1, 1
        };
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        CUresult result = cuTensorMapEncodeTiled(
            &g_host.A_tma[i],
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            5,
            const_cast<void*>(data_ptr),
            gmem_shape,
            gmem_stride,
            smem_shape,
            smem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(result == CUDA_SUCCESS,
                    "Strided localCTA A TMA creation failed for batch ", i);

        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared_list[i], 1,
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(0) / 128 : B_sc_prepared_list[i].size(0),
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(1) / 4 : B_sc_prepared_list[i].size(1),
            256);

        encode_prepared_scale_tensor_map<typename G::A_sc_tile>(
            &g_host.A_sc_tma[i], A_sc_prepared_list[i], "A_sc_prepared_list[i]");
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        g_host.A_sg[i] = one_ptr;
        g_host.B_sg[i] = one_ptr;
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_batched_gemm_strided_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    std::vector<at::Tensor>& D_list
) {
    using G = nvfp4_batched_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_sc_prepared_list.size());
    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    auto one = get_unit_scale_tensor(A_full);
    const float* one_ptr = one.data_ptr<float>();

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    g_host.num_red_blocks = static_cast<int>((2 * A_col_widths[0]) / C::Kb);

    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    for (int i = 0; i < n; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        const int64_t fp4_offset = A_col_offsets[i];
        const void* data_ptr = a_base + fp4_offset;

        uint64_t gmem_shape[5] = {
            static_cast<uint64_t>(swizzle_elements),
            static_cast<uint64_t>(M),
            static_cast<uint64_t>((fp4_cols + swizzle_elements - 1) / swizzle_elements),
            1, 1
        };
        uint64_t gmem_stride[4] = {
            static_cast<uint64_t>(a_full_row_stride),
            128,
            static_cast<uint64_t>(M * a_full_row_stride),
            static_cast<uint64_t>(M * a_full_row_stride)
        };
        uint32_t smem_shape[5] = {
            static_cast<uint32_t>(swizzle_elements),
            static_cast<uint32_t>(C::Mb / 2),
            1, 1, 1
        };
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        CUresult result = cuTensorMapEncodeTiled(
            &g_host.A_tma[i],
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            5,
            const_cast<void*>(data_ptr),
            gmem_shape,
            gmem_stride,
            smem_shape,
            smem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(result == CUDA_SUCCESS,
                    "Strided localCTA A TMA creation failed for batch ", i);

        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_prepared_list[i], 1,
            A_sc_prepared_list[i].dim() == 2 ? A_sc_prepared_list[i].size(0) / 128 : A_sc_prepared_list[i].size(0),
            A_sc_prepared_list[i].dim() == 2 ? A_sc_prepared_list[i].size(1) / 4 : A_sc_prepared_list[i].size(1),
            256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared_list[i], 1,
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(0) / 128 : B_sc_prepared_list[i].size(0),
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(1) / 4 : B_sc_prepared_list[i].size(1),
            256);
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);

        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        g_host.A_sg[i] = one_ptr;
        g_host.B_sg[i] = one_ptr;
    }

    kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_split3_dgrad_gemm_strided_onepass_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split3_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_sc_prepared_list.size() == 3, "one-pass split3 dgrad expects 3 A scale batches");
    TORCH_CHECK(B_list.size() == 3, "one-pass split3 dgrad expects 3 B batches");
    TORCH_CHECK(B_sc_prepared_list.size() == 3, "one-pass split3 dgrad expects 3 B scale batches");
    TORCH_CHECK(A_col_offsets.size() == 3, "one-pass split3 dgrad expects 3 A offsets");
    TORCH_CHECK(A_col_widths.size() == 3, "one-pass split3 dgrad expects 3 A widths");

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    for (int i = 0; i < 3; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        const int64_t fp4_offset = A_col_offsets[i];
        const void* data_ptr = a_base + fp4_offset;

        TORCH_CHECK(fp4_cols > 0, "A_col_widths must be positive");
        TORCH_CHECK((2 * fp4_cols) % C::Kb == 0,
                    "one-pass split3 dgrad expects reduction widths aligned to Kb=", C::Kb);
        g_host.num_red_blocks[i] = static_cast<int>((2 * fp4_cols) / C::Kb);

        uint64_t gmem_shape[5] = {
            static_cast<uint64_t>(swizzle_elements),
            static_cast<uint64_t>(M),
            static_cast<uint64_t>((fp4_cols + swizzle_elements - 1) / swizzle_elements),
            1, 1
        };
        uint64_t gmem_stride[4] = {
            static_cast<uint64_t>(a_full_row_stride),
            128,
            static_cast<uint64_t>(M * a_full_row_stride),
            static_cast<uint64_t>(M * a_full_row_stride)
        };
        uint32_t smem_shape[5] = {
            static_cast<uint32_t>(swizzle_elements),
            static_cast<uint32_t>(C::Mb / 2),
            1, 1, 1
        };
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        CUresult result = cuTensorMapEncodeTiled(
            &g_host.A_tma[i],
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            5,
            const_cast<void*>(data_ptr),
            gmem_shape,
            gmem_stride,
            smem_shape,
            smem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(result == CUDA_SUCCESS,
                    "One-pass split3 localCTA A TMA creation failed for batch ", i);

        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared_list[i], 1,
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(0) / 128 : B_sc_prepared_list[i].size(0),
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(1) / 4 : B_sc_prepared_list[i].size(1),
            256);

        encode_prepared_scale_tensor_map<typename G::A_sc_tile>(
            &g_host.A_sc_tma[i], A_sc_prepared_list[i], "A_sc_prepared_list[i]");
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split3_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_split2_dgrad_gemm_strided_onepass_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 A scale batches");
    TORCH_CHECK(B_list.size() == 2, "one-pass split2 dgrad expects 2 B batches");
    TORCH_CHECK(B_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 B scale batches");
    TORCH_CHECK(A_col_offsets.size() == 2, "one-pass split2 dgrad expects 2 A offsets");
    TORCH_CHECK(A_col_widths.size() == 2, "one-pass split2 dgrad expects 2 A widths");

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    for (int i = 0; i < 2; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        const int64_t fp4_offset = A_col_offsets[i];
        const void* data_ptr = a_base + fp4_offset;

        TORCH_CHECK(fp4_cols > 0, "A_col_widths must be positive");
        TORCH_CHECK((2 * fp4_cols) % C::Kb == 0,
                    "one-pass split2 dgrad expects reduction widths aligned to Kb=", C::Kb);
        g_host.num_red_blocks[i] = static_cast<int>((2 * fp4_cols) / C::Kb);

        uint64_t gmem_shape[5] = {
            static_cast<uint64_t>(swizzle_elements),
            static_cast<uint64_t>(M),
            static_cast<uint64_t>((fp4_cols + swizzle_elements - 1) / swizzle_elements),
            1, 1
        };
        uint64_t gmem_stride[4] = {
            static_cast<uint64_t>(a_full_row_stride),
            128,
            static_cast<uint64_t>(M * a_full_row_stride),
            static_cast<uint64_t>(M * a_full_row_stride)
        };
        uint32_t smem_shape[5] = {
            static_cast<uint32_t>(swizzle_elements),
            static_cast<uint32_t>(C::Mb / 2),
            1, 1, 1
        };
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        CUresult result = cuTensorMapEncodeTiled(
            &g_host.A_tma[i],
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            5,
            const_cast<void*>(data_ptr),
            gmem_shape,
            gmem_stride,
            smem_shape,
            smem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(result == CUDA_SUCCESS,
                    "One-pass split2 localCTA A TMA creation failed for batch ", i);

        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared_list[i], 1,
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(0) / 128 : B_sc_prepared_list[i].size(0),
            B_sc_prepared_list[i].dim() == 2 ? B_sc_prepared_list[i].size(1) / 4 : B_sc_prepared_list[i].size(1),
            256);

        encode_prepared_scale_tensor_map<typename G::A_sc_tile>(
            &g_host.A_sc_tma[i], A_sc_prepared_list[i], "A_sc_prepared_list[i]");
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split2_accum_gemm::kernel<C>>(g_host);
}

void launch_fast_split3_dgrad_gemm_strided_onepass(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "one-pass split3 config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_fast_split3_dgrad_gemm_strided_onepass_with_config<localcta_onepass_cfg1>(
                A_full, A_sc_prepared_list, A_col_offsets, A_col_widths, B_list, B_sc_prepared_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split3 config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split3_dgrad_gemm_strided_onepass_with_config<localcta_onepass_cfg3>(
                A_full, A_sc_prepared_list, A_col_offsets, A_col_widths, B_list, B_sc_prepared_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split3 config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split3_dgrad_gemm_strided_onepass_with_config<localcta_onepass_cfg5>(
                A_full, A_sc_prepared_list, A_col_offsets, A_col_widths, B_list, B_sc_prepared_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split3 config_idx=", resolved_idx);
    }
}

void launch_fast_split2_dgrad_gemm_strided_onepass(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "one-pass split2 config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_fast_split2_dgrad_gemm_strided_onepass_with_config<localcta_onepass_cfg1>(
                A_full, A_sc_prepared_list, A_col_offsets, A_col_widths, B_list, B_sc_prepared_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split2 config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split2_dgrad_gemm_strided_onepass_with_config<localcta_onepass_cfg3>(
                A_full, A_sc_prepared_list, A_col_offsets, A_col_widths, B_list, B_sc_prepared_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split2 config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split2_dgrad_gemm_strided_onepass_with_config<localcta_onepass_cfg5>(
                A_full, A_sc_prepared_list, A_col_offsets, A_col_widths, B_list, B_sc_prepared_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split2 config_idx=", resolved_idx);
    }
}

void launch_fast_batched_accum_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    if (A_list.size() == 1) {
        launch_fast_regular_gemm(
            A_list[0], A_sc_prepared_list[0],
            B_list[0], B_sc_prepared_list[0],
            D_out);
        return;
    }
    const int64_t reduction_k = A_list[0].size(1) * 2;
    if (A_list.size() == 2 && reduction_k >= 4096) {
        launch_fast_batched_accum_gemm_with_config<localcta_fast_largek_config>(
            A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
        return;
    }
    launch_fast_batched_accum_gemm_with_config<localcta_fast_batched_config>(
        A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
}

void launch_fast_batched_accum_gemm_outer_sg(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_list,
    at::Tensor& D_out
) {
    if (A_list.size() == 1) {
        launch_fast_regular_gemm_outer_sg(
            A_list[0], A_sc_list[0], A_sg_list[0],
            B_list[0], B_sc_list[0], B_sg_list[0],
            D_out);
        return;
    }
    const int64_t reduction_k = A_list[0].size(1) * 2;
    if (A_list.size() == 2 && reduction_k >= 4096) {
        launch_fast_batched_accum_gemm_outer_sg_with_config<localcta_fast_largek_config>(
            A_list, A_sc_list, A_sg_list,
            B_list, B_sc_list, B_sg_list,
            D_out);
        return;
    }
    launch_fast_batched_accum_gemm_outer_sg_with_config<localcta_fast_batched_config>(
        A_list, A_sc_list, A_sg_list,
        B_list, B_sc_list, B_sg_list,
        D_out);
}

void launch_fast_split3_dgrad_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    TORCH_CHECK(A_list.size() == 3, "split3 dgrad expects 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 dgrad expects 3 B batches");
    const int64_t q_reduction_k = A_list[0].size(1) * 2;
    if (q_reduction_k >= 2048) {
        launch_fast_batched_accum_gemm_with_config<localcta_fast_split3_dgrad_config>(
            A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
        return;
    }
    launch_fast_batched_accum_gemm_with_config<localcta_fast_batched_config>(
        A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
}

template <typename C>
void launch_fast_split2_dgrad_gemm_onepass_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_list.size() == 2, "one-pass split2 dgrad expects 2 A batches");
    TORCH_CHECK(A_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 A scale batches");
    TORCH_CHECK(B_list.size() == 2, "one-pass split2 dgrad expects 2 B batches");
    TORCH_CHECK(B_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 B scale batches");

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    for (int i = 0; i < 2; ++i) {
        g_host.num_red_blocks[i] = static_cast<int>((2 * A_list[i].size(1)) / C::Kb);

        auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_prepared_list[i], 1,
            A_sc_prepared_list[i].size(0), A_sc_prepared_list[i].size(1), 256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_prepared_list[i], 1,
            B_sc_prepared_list[i].size(0), B_sc_prepared_list[i].size(1), 256);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split2_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_split2_dgrad_gemm_onepass_sg_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_list.size() == 2, "one-pass split2 dgrad expects 2 A batches");
    TORCH_CHECK(A_sc_list.size() == 2, "one-pass split2 dgrad expects 2 A scale batches");
    TORCH_CHECK(A_sg_chunks_list.size() == 2, "one-pass split2 dgrad expects 2 A SG batches");
    TORCH_CHECK(B_list.size() == 2, "one-pass split2 dgrad expects 2 B batches");
    TORCH_CHECK(B_sc_list.size() == 2, "one-pass split2 dgrad expects 2 B scale batches");
    TORCH_CHECK(B_sg_chunks_list.size() == 2, "one-pass split2 dgrad expects 2 B SG batches");

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    for (int i = 0; i < 2; ++i) {
        g_host.num_red_blocks[i] = static_cast<int>((2 * A_list[i].size(1)) / C::Kb);

        auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_list[i], 1, A_sc_list[i].size(0), A_sc_list[i].size(1), 256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_list[i], 1, B_sc_list[i].size(0), B_sc_list[i].size(1), 256);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        g_host.A_sg[i] = A_sg_chunks_list[i].data_ptr<float>();
        g_host.B_sg[i] = B_sg_chunks_list[i].data_ptr<float>();
        // Negative stride marks strict v4 chunk-grid SG. The split2 kernel
        // keeps positive strides for the older outer-scale epilogue contract.
        g_host.A_sg_stride[i] = -static_cast<int>(A_sg_chunks_list[i].size(1));
        g_host.B_sg_stride[i] = -static_cast<int>(B_sg_chunks_list[i].size(1));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split2_accum_gemm::kernel_chunk_grid<C>>(g_host);
}

void launch_fast_split2_dgrad_gemm_onepass(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "one-pass split2 config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_fast_split2_dgrad_gemm_onepass_with_config<localcta_onepass_cfg1>(
                A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split2 config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split2_dgrad_gemm_onepass_with_config<localcta_onepass_cfg3>(
                A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split2 config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split2_dgrad_gemm_onepass_with_config<localcta_onepass_cfg5>(
                A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split2 config_idx=", resolved_idx);
    }
}

void launch_fast_split2_dgrad_gemm_onepass_sg(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "one-pass split2 config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_fast_split2_dgrad_gemm_onepass_sg_with_config<localcta_onepass_sg_cfg1>(
                A_list, A_sc_list, A_sg_chunks_list, B_list, B_sc_list, B_sg_chunks_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split2 config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split2_dgrad_gemm_onepass_sg_with_config<localcta_onepass_sg_cfg3>(
                A_list, A_sc_list, A_sg_chunks_list, B_list, B_sc_list, B_sg_chunks_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split2 config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split2_dgrad_gemm_onepass_sg_with_config<localcta_onepass_sg_cfg5>(
                A_list, A_sc_list, A_sg_chunks_list, B_list, B_sc_list, B_sg_chunks_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split2 config_idx=", resolved_idx);
    }
}

void launch_fast_split3_dgrad_gemm_strided(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects 3 A scale batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects 3 B batches");
    const int64_t max_reduction_k =
        2 * (*std::max_element(A_col_widths.begin(), A_col_widths.end()));
    if (max_reduction_k >= 2048) {
        launch_fast_batched_accum_gemm_strided_with_config<localcta_fast_split3_dgrad_config>(
            A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
            A_col_offsets, A_col_widths, D_out);
        return;
    }
    launch_fast_batched_accum_gemm_strided_with_config<localcta_fast_batched_config>(
        A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
        A_col_offsets, A_col_widths, D_out);
}

void launch_fast_split3_dgrad_gemm_strided_sum(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects 3 A scale batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects 3 B batches");

    std::vector<at::Tensor> D_list;
    D_list.reserve(3);
    for (int i = 0; i < 3; ++i) {
        D_list.push_back(at::empty_like(D_out));
    }

    const int64_t max_reduction_k =
        2 * (*std::max_element(A_col_widths.begin(), A_col_widths.end()));
    if (max_reduction_k >= 2048) {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_split3_dgrad_config>(
            A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
            A_col_offsets, A_col_widths, D_list);
    } else {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_batched_config>(
            A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
            A_col_offsets, A_col_widths, D_list);
    }

    const int64_t numel = D_out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
        numel);
    CUDACHECK(cudaGetLastError());
}

void launch_fast_split2_dgrad_gemm_strided_sum(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    TORCH_CHECK(A_sc_prepared_list.size() == 2, "split2 strided dgrad expects 2 A scale batches");
    TORCH_CHECK(B_list.size() == 2, "split2 strided dgrad expects 2 B batches");

    std::vector<at::Tensor> D_list;
    D_list.reserve(2);
    for (int i = 0; i < 2; ++i) {
        D_list.push_back(at::empty_like(D_out));
    }

    const int64_t max_reduction_k =
        2 * (*std::max_element(A_col_widths.begin(), A_col_widths.end()));
    if (max_reduction_k >= 2048) {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_split2_dgrad_config>(
            A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
            A_col_offsets, A_col_widths, D_list);
    } else {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_batched_config>(
            A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
            A_col_offsets, A_col_widths, D_list);
    }

    const int64_t numel = D_out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    sum_tensors_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
        numel);
    CUDACHECK(cudaGetLastError());
}

void launch_fast_batched_gemm_strided(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    std::vector<at::Tensor>& D_list
) {
    const int64_t max_reduction_k =
        2 * (*std::max_element(A_col_widths.begin(), A_col_widths.end()));
    if (max_reduction_k >= 2048) {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_split3_dgrad_config>(
            A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
            A_col_offsets, A_col_widths, D_list);
        return;
    }
    launch_fast_batched_gemm_strided_with_config<localcta_fast_batched_config>(
        A_full, A_sc_prepared_list, B_list, B_sc_prepared_list,
        A_col_offsets, A_col_widths, D_list);
}

__global__ void sum_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(A[idx], B[idx]);
    }
}

__global__ void sum3_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(__hadd(A[idx], B[idx]), C[idx]);
    }
}

__global__ void sum4_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ D,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(__hadd(A[idx], B[idx]), __hadd(C[idx], D[idx]));
    }
}

}  // namespace

void nvfp4_localcta_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    check_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_regular_gemm(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
}

void nvfp4_localcta_fast_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    at::Tensor& D
) {
    check_fast_gemm_inputs(A, A_sc_prepared, B, B_sc_prepared);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm(A, A_sc_prepared, B, B_sc_prepared, D);
}

void nvfp4_localcta_fast_gemm_sg_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    check_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm_sg(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
}

void nvfp4_localcta_grouped_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    check_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kBFloat16,
                "D must be a contiguous CUDA bf16 tensor");
    if (D_K_opt.has_value()) {
        TORCH_CHECK(D_K_opt.value().is_cuda() && D_K_opt.value().is_contiguous() &&
                    D_K_opt.value().scalar_type() == at::kBFloat16,
                    "D_K must be a contiguous CUDA bf16 tensor");
    }
    if (D_V_opt.has_value()) {
        TORCH_CHECK(D_V_opt.value().is_cuda() && D_V_opt.value().is_contiguous() &&
                    D_V_opt.value().scalar_type() == at::kBFloat16,
                    "D_V must be a contiguous CUDA bf16 tensor");
    }

    launch_grouped_gemm(
        A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks,
        D, D_K_opt, D_V_opt, silu_dim);
}

void nvfp4_localcta_fast_grouped_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    check_fast_gemm_inputs(A, A_sc_prepared, B, B_sc_prepared);
    TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kBFloat16,
                "D must be a contiguous CUDA bf16 tensor");
    if (D_K_opt.has_value()) {
        TORCH_CHECK(D_K_opt.value().is_cuda() && D_K_opt.value().is_contiguous() &&
                    D_K_opt.value().scalar_type() == at::kBFloat16,
                    "D_K must be a contiguous CUDA bf16 tensor");
    }
    if (D_V_opt.has_value()) {
        TORCH_CHECK(D_V_opt.value().is_cuda() && D_V_opt.value().is_contiguous() &&
                    D_V_opt.value().scalar_type() == at::kBFloat16,
                    "D_V must be a contiguous CUDA bf16 tensor");
    }

    launch_fast_grouped_gemm(
        A, A_sc_prepared, B, B_sc_prepared, D, D_K_opt, D_V_opt, silu_dim);
}

void nvfp4_localcta_fast_gemm_outer_sg_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    at::Tensor& D
) {
    check_fast_gemm_outer_sg_inputs(A, A_sc, A_sg, B, B_sc, B_sg);
    TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kBFloat16,
                "D must be a contiguous CUDA bf16 tensor");
    launch_fast_regular_gemm_outer_sg(A, A_sc, A_sg, B, B_sc, B_sg, D);
}

void nvfp4_localcta_fast_grouped_gemm_outer_sg_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    check_fast_gemm_outer_sg_inputs(A, A_sc, A_sg, B, B_sc, B_sg);
    TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kBFloat16,
                "D must be a contiguous CUDA bf16 tensor");
    if (D_K_opt.has_value()) {
        TORCH_CHECK(D_K_opt.value().is_cuda() && D_K_opt.value().is_contiguous() &&
                    D_K_opt.value().scalar_type() == at::kBFloat16,
                    "D_K must be a contiguous CUDA bf16 tensor");
    }
    if (D_V_opt.has_value()) {
        TORCH_CHECK(D_V_opt.value().is_cuda() && D_V_opt.value().is_contiguous() &&
                    D_V_opt.value().scalar_type() == at::kBFloat16,
                    "D_V must be a contiguous CUDA bf16 tensor");
    }

    launch_fast_grouped_gemm_outer_sg(
        A, A_sc, A_sg, B, B_sc, B_sg, D, D_K_opt, D_V_opt, silu_dim);
}

void nvfp4_localcta_fast_grouped_gemm_sg_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    check_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kBFloat16,
                "D must be a contiguous CUDA bf16 tensor");
    if (D_K_opt.has_value()) {
        TORCH_CHECK(D_K_opt.value().is_cuda() && D_K_opt.value().is_contiguous() &&
                    D_K_opt.value().scalar_type() == at::kBFloat16,
                    "D_K must be a contiguous CUDA bf16 tensor");
    }
    if (D_V_opt.has_value()) {
        TORCH_CHECK(D_V_opt.value().is_cuda() && D_V_opt.value().is_contiguous() &&
                    D_V_opt.value().scalar_type() == at::kBFloat16,
                    "D_V must be a contiguous CUDA bf16 tensor");
    }

    launch_fast_grouped_gemm_sg(
        A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D, D_K_opt, D_V_opt, silu_dim);
}

void nvfp4_localcta_batched_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    check_batched_inputs(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list);

    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= nvfp4_localcta_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_localcta_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == static_cast<int>(D_list.size()),
                "D_list length must match number of batches");

    for (int i = 0; i < n; ++i) {
        check_output_matrix(D_list[i], "D_list[i]", A_list[i].size(0), B_list[i].size(0));
    }
    check_batched_shape_compatibility(A_list, B_list, D_list);

    launch_batched_gemm(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list,
        D_list);
}

void nvfp4_localcta_fast_batched_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    std::vector<at::Tensor>& D_list
) {
    check_fast_batched_inputs(A_list, A_sc_prepared_list, B_list, B_sc_prepared_list);

    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= nvfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == static_cast<int>(D_list.size()),
                "D_list length must match number of batches");

    for (int i = 0; i < n; ++i) {
        check_output_matrix(D_list[i], "D_list[i]", A_list[i].size(0), B_list[i].size(0));
    }
    check_batched_shape_compatibility(A_list, B_list, D_list);

    launch_fast_batched_gemm(A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_list);
}

void nvfp4_localcta_fast_batched_gemm_strided_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    (void)A_sg_chunks_list;
    (void)B_sg_chunks_list;
    check_fast_batched_strided_inputs(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);

    const int n = static_cast<int>(A_sc_prepared_list.size());
    TORCH_CHECK(n <= nvfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == static_cast<int>(D_list.size()),
                "D_list length must match number of batches");
    for (int i = 0; i < n; ++i) {
        check_output_matrix(D_list[i], "D_list[i]", A_full.size(0), B_list[i].size(0));
    }
    launch_fast_batched_gemm_strided(
        A_full, A_sc_prepared_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, D_list);
}

void nvfp4_localcta_fast_batched_gemm_strided_nopdl_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    nvfp4_localcta_fast_batched_gemm_strided_entrypoint(
        A_full, A_sc_prepared_list, A_sg_chunks_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, B_sg_chunks_list, D_list);
}

void nvfp4_localcta_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out
) {
    check_batched_inputs(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list);
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= 4, "num_batches must be 1..4");
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));

    if (n == 1) {
        launch_regular_gemm(
            A_list[0], A_sc_list[0], A_sg_chunks_list[0],
            B_list[0], B_sc_list[0], B_sg_chunks_list[0], D_out);
        return;
    }

    std::vector<at::Tensor> D_list;
    D_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        D_list.push_back(at::empty_like(D_out));
    }

    launch_batched_gemm(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list,
        D_list);

    const int64_t numel = D_out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (n == 2) {
        sum_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    } else if (n == 3) {
        sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    } else {
        sum4_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[3].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    }

    CUDACHECK(cudaGetLastError());
}

void nvfp4_localcta_fast_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    check_fast_batched_inputs(A_list, A_sc_prepared_list, B_list, B_sc_prepared_list);
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= 4, "num_batches must be 1..4");
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));

    launch_fast_batched_accum_gemm(
        A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
}

void nvfp4_localcta_fast_batched_accum_gemm_outer_sg_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_list,
    at::Tensor& D_out
) {
    check_fast_batched_outer_sg_inputs(
        A_list, A_sc_list, A_sg_list,
        B_list, B_sc_list, B_sg_list);
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= 4, "num_batches must be 1..4");
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));

    launch_fast_batched_accum_gemm_outer_sg(
        A_list, A_sc_list, A_sg_list,
        B_list, B_sc_list, B_sg_list,
        D_out);
}

void nvfp4_localcta_fast_split3_dgrad_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    check_fast_batched_inputs(A_list, A_sc_prepared_list, B_list, B_sc_prepared_list);
    TORCH_CHECK(A_list.size() == 3, "split3 dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 dgrad expects exactly 3 B batches");
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm(
        A_list, A_sc_prepared_list, B_list, B_sc_prepared_list, D_out);
}

void nvfp4_localcta_fast_split2_dgrad_onepass_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_fast_batched_inputs(A_list, A_sc_prepared_list, B_list, B_sc_prepared_list);
    TORCH_CHECK(A_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B batches");
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));
    launch_fast_split2_dgrad_gemm_onepass(
        A_list, A_sc_prepared_list, B_list, B_sc_prepared_list,
        D_out, static_cast<int>(config_idx));
}

void nvfp4_localcta_fast_split2_dgrad_onepass_gemm_sg_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_batched_inputs(A_list, A_sc_list, A_sg_chunks_list, B_list, B_sc_list, B_sg_chunks_list);
    TORCH_CHECK(A_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B batches");
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));
    launch_fast_split2_dgrad_gemm_onepass_sg(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list,
        D_out, static_cast<int>(config_idx));
}

void nvfp4_localcta_fast_split3_dgrad_strided_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    check_fast_batched_strided_inputs(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects exactly 3 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm_strided(
        A_full, A_sc_prepared_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, D_out);
}

void nvfp4_localcta_fast_split3_dgrad_strided_sum_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    check_fast_batched_strided_inputs(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects exactly 3 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm_strided_sum(
        A_full, A_sc_prepared_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, D_out);
}

void nvfp4_localcta_fast_split3_dgrad_strided_onepass_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_fast_batched_strided_inputs(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 one-pass dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 one-pass dgrad expects exactly 3 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm_strided_onepass(
        A_full, A_sc_prepared_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, D_out, static_cast<int>(config_idx));
}

void nvfp4_localcta_fast_split2_dgrad_strided_sum_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out
) {
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 2, "split2 strided dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "split2 strided dgrad expects exactly 2 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split2_dgrad_gemm_strided_sum(
        A_full, A_sc_prepared_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, D_out);
}

void nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split2_dgrad_gemm_strided_onepass(
        A_full, A_sc_prepared_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, D_out, static_cast<int>(config_idx));
}

void nvfp4_localcta_split_dgrad_sum_entrypoint(
    const at::Tensor& A_fp4_cat,
    const at::Tensor& A_sc_cat,
    const std::vector<at::Tensor>& A_sg_list,
    const std::vector<at::Tensor>& B_fp4_list,
    const std::vector<at::Tensor>& B_sc_list,
    const at::Tensor& B_sg_cat,
    const std::vector<int64_t>& split_dims,
    at::Tensor& D_out
) {
    (void)A_sg_list;
    (void)B_sg_cat;
    check_fp4_matrix(A_fp4_cat, "A_fp4_cat");
    TORCH_CHECK(A_sc_cat.is_cuda() && A_sc_cat.is_contiguous(), "A_sc_cat must be contiguous CUDA tensor");
    TORCH_CHECK(A_sc_cat.scalar_type() == at::kFloat8_e4m3fn, "A_sc_cat must be fp8 e4m3");
    TORCH_CHECK(A_sc_cat.dim() == 3, "A_sc_cat must be 3D");
    const int n_splits = static_cast<int>(split_dims.size());
    TORCH_CHECK(n_splits >= 2 && n_splits <= 3, "split_dgrad_sum supports 2 or 3 splits");
    TORCH_CHECK(n_splits == static_cast<int>(B_fp4_list.size()), "B_fp4_list size mismatch");
    TORCH_CHECK(n_splits == static_cast<int>(B_sc_list.size()), "B_sc_list size mismatch");

    std::vector<at::Tensor> A_sc_list;
    std::vector<int64_t> A_col_offsets;
    std::vector<int64_t> A_col_widths;
    A_sc_list.reserve(n_splits);
    A_col_offsets.reserve(n_splits);
    A_col_widths.reserve(n_splits);

    int64_t sc_offset = 0;
    int64_t fp4_offset = 0;
    for (int i = 0; i < n_splits; ++i) {
        const int64_t n_i = split_dims[i];
        TORCH_CHECK(n_i % 128 == 0, "split dims must be multiples of 128");
        A_sc_list.push_back(A_sc_cat.narrow(1, sc_offset, n_i / 64));
        A_col_offsets.push_back(fp4_offset);
        A_col_widths.push_back(n_i / 2);
        sc_offset += n_i / 64;
        fp4_offset += n_i / 2;
    }

    std::vector<at::Tensor> D_list;
    D_list.reserve(n_splits);
    for (int i = 0; i < n_splits; ++i) {
        D_list.push_back(at::empty_like(D_out));
    }

    launch_fast_batched_gemm_strided(
        A_fp4_cat, A_sc_list, A_col_offsets, A_col_widths,
        B_fp4_list, B_sc_list, D_list);

    const int64_t numel = D_out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (n_splits == 2) {
        sum_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    } else {
        sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    }
    CUDACHECK(cudaGetLastError());
}

void sum2_bf16_entrypoint(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& D_out
) {
    check_output_matrix(D_out, "D_out", A.size(0), A.size(1));
    TORCH_CHECK(B.is_cuda() && B.is_contiguous() && B.scalar_type() == at::kBFloat16,
                "B must be contiguous CUDA bf16");
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.scalar_type() == at::kBFloat16,
                "A must be contiguous CUDA bf16");
    TORCH_CHECK(A.sizes() == B.sizes() && A.sizes() == D_out.sizes(),
                "sum2_bf16 inputs must have identical shapes");
    const int64_t numel = D_out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    sum_tensors_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
        numel);
    CUDACHECK(cudaGetLastError());
}

void sum3_bf16_entrypoint(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& D_out
) {
    check_output_matrix(D_out, "D_out", A.size(0), A.size(1));
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.scalar_type() == at::kBFloat16,
                "A must be contiguous CUDA bf16");
    TORCH_CHECK(B.is_cuda() && B.is_contiguous() && B.scalar_type() == at::kBFloat16,
                "B must be contiguous CUDA bf16");
    TORCH_CHECK(C.is_cuda() && C.is_contiguous() && C.scalar_type() == at::kBFloat16,
                "C must be contiguous CUDA bf16");
    TORCH_CHECK(A.sizes() == B.sizes() && A.sizes() == C.sizes() && A.sizes() == D_out.sizes(),
                "sum3_bf16 inputs must have identical shapes");
    const int64_t numel = D_out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(C.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
        numel);
    CUDACHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_localcta_gemm", &nvfp4_localcta_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_fast_gemm", &nvfp4_localcta_fast_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc_prepared"),
          pybind11::arg("B"), pybind11::arg("B_sc_prepared"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_fast_gemm_outer_sg", &nvfp4_localcta_fast_gemm_outer_sg_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_fast_gemm_sg", &nvfp4_localcta_fast_gemm_sg_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_grouped_gemm", &nvfp4_localcta_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_fast_grouped_gemm", &nvfp4_localcta_fast_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc_prepared"),
          pybind11::arg("B"), pybind11::arg("B_sc_prepared"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_fast_grouped_gemm_outer_sg", &nvfp4_localcta_fast_grouped_gemm_outer_sg_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_fast_grouped_gemm_sg", &nvfp4_localcta_fast_grouped_gemm_sg_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_batched_gemm", &nvfp4_localcta_batched_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_localcta_fast_batched_gemm", &nvfp4_localcta_fast_batched_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_prepared_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_localcta_fast_batched_gemm_strided",
          &nvfp4_localcta_fast_batched_gemm_strided_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_localcta_fast_batched_gemm_strided_nopdl",
          &nvfp4_localcta_fast_batched_gemm_strided_nopdl_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_localcta_batched_accum_gemm", &nvfp4_localcta_batched_accum_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_batched_accum_gemm", &nvfp4_localcta_fast_batched_accum_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_prepared_list"),
          pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_batched_accum_gemm_outer_sg",
          &nvfp4_localcta_fast_batched_accum_gemm_outer_sg_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_list"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_split3_dgrad_gemm", &nvfp4_localcta_fast_split3_dgrad_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_prepared_list"),
          pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_split2_dgrad_onepass_gemm",
          &nvfp4_localcta_fast_split2_dgrad_onepass_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_prepared_list"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_fast_split2_dgrad_onepass_gemm_sg",
          &nvfp4_localcta_fast_split2_dgrad_onepass_gemm_sg_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_fast_split3_dgrad_strided_gemm",
          &nvfp4_localcta_fast_split3_dgrad_strided_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_split3_dgrad_strided_sum_gemm",
          &nvfp4_localcta_fast_split3_dgrad_strided_sum_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_split3_dgrad_strided_onepass_gemm",
          &nvfp4_localcta_fast_split3_dgrad_strided_onepass_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("D_out"),
          pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_fast_split2_dgrad_strided_sum_gemm",
          &nvfp4_localcta_fast_split2_dgrad_strided_sum_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm",
          &nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("D_out"),
          pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_split_dgrad_sum",
          &nvfp4_localcta_split_dgrad_sum_entrypoint,
          pybind11::arg("A_fp4_cat"), pybind11::arg("A_sc_cat"),
          pybind11::arg("A_sg_list"), pybind11::arg("B_fp4_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_cat"),
          pybind11::arg("split_dims"), pybind11::arg("D_out"));
    m.def("sum2_bf16", &sum2_bf16_entrypoint,
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("D_out"));
    m.def("sum3_bf16", &sum3_bf16_entrypoint,
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"),
          pybind11::arg("D_out"));
}
