#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "pyutils/torchutils.cuh"
#include "nvfp4_localcta_kernel.cuh"
#include "nvfp4_localcta_batched_kernel.cuh"
#include "../nvfp4_accum_gemm.cuh"
#include "../nvfp4_batched_accum_gemm.cuh"
#include "../nvfp4_gemm.cuh"
#include "../nvfp4_batched_gemm.cuh"
#include "../nvfp4_split2_accum_gemm.cuh"
#include "../nvfp4_split3_accum_gemm.cuh"
#include "nvfp4_localcta_silu_dgrad_quant_gemm.cuh"  // fused W2-dgrad -> SiLU split2 quant producer

namespace {

constexpr int kScaleBytesPerTile = 512;
using localcta_regular_smalln_config = nvfp4_localcta_gemm::config<128, 5, 4, 12, 2, true, 256, true, 2, 128>;
using localcta_regular_smallk_config = nvfp4_localcta_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_regular_largek_config = nvfp4_localcta_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 128>;
using localcta_parity_config = nvfp4_localcta_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_tilegrid256_config = nvfp4_localcta_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 256>;
using localcta_fast_smallk_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
using localcta_fast_largek_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false>;
using localcta_fast_config6 = nvfp4_gemm::config<256, 4, 8, 12, 2, false>;
using localcta_fast_config12 = nvfp4_gemm::config<256, 5, 8, 1, 2, false>;
using localcta_fast_config13 = nvfp4_gemm::config<256, 5, 8, 2, 2, false>;
using localcta_fast_config25 = nvfp4_gemm::config<256, 4, 8, 1, 2, false>;
using localcta_fast_config27 = nvfp4_gemm::config<256, 4, 8, 2, 2, false>;
using localcta_fast_config28 = nvfp4_gemm::config<256, 4, 8, 8, 2, false>;
using localcta_fast_config28_residual_config = nvfp4_gemm::config<256, 4, 8, 8, 2, false, 256, true, 2, 256, false, true>;
using localcta_fast_smallk_residual_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 256, false, true>;
using localcta_fast_largek_residual_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 256, false, true>;
using localcta_fast_smallk_residual_rms_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 256, false, true, false, true, true>;
using localcta_fast_largek_residual_rms_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 256, false, true, false, true, true>;
using localcta_fast_smallk_residual_rms_pipeline_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 256, false, true, false, true, true, true>;
using localcta_fast_largek_residual_rms_pipeline_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 256, false, true, false, true, true, true>;
using localcta_fast_smallk_h_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 256, false, true, true>;
using localcta_fast_largek_h_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 256, false, true, true>;
using localcta_fast_grouped_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
using localcta_fast_largek_rope_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 256, true>;
using localcta_fast_grouped_rope_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 256, true>;
using localcta_fast_chunkgrid_smallk_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_fast_chunkgrid_largek_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 128>;
using localcta_fast_chunkgrid_grouped_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_fast_chunkgrid_smalln_config = nvfp4_gemm::config<128, 5, 4, 12, 2, true, 256, true, 2, 128>;
using localcta_fast_chunkgrid_smalln_epi2_config = nvfp4_gemm::config<128, 5, 2, 12, 2, true, 256, true, 2, 128>;
using localcta_fast_chunkgrid_smalln_epi1_config = nvfp4_gemm::config<128, 5, 1, 12, 2, true, 256, true, 2, 128>;
using localcta_fast_batched_config = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
using localcta_fast_split2_dgrad_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false>;
using localcta_fast_split3_dgrad_config = nvfp4_gemm::config<256, 5, 8, 12, 2, false>;
using localcta_fast_split3_dgrad_sg_smalln_config = nvfp4_gemm::config<128, 5, 4, 12, 2, true>;
using localcta_onepass_cfg0 = nvfp4_gemm::config<128, 5, 4, 12, 2, true, 256, false, 1>;
using localcta_onepass_cfg1 = nvfp4_gemm::config<128, 5, 4, 12, 2, true, 256, false, 2>;
using localcta_onepass_cfg2 = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 1>;
using localcta_onepass_cfg3 = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
using localcta_onepass_cfg4 = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, false, 1>;
using localcta_onepass_cfg5 = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, false, 2>;
using localcta_onepass_sg_cfg1 = nvfp4_gemm::config<128, 5, 4, 12, 2, true, 256, false, 2, 128>;
using localcta_onepass_sg_cfg3 = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2, 128>;
using localcta_onepass_sg_cfg5 = nvfp4_gemm::config<256, 5, 8, 12, 2, false, 256, false, 2, 128>;
using localcta_onepass_sg_cfg6 = nvfp4_gemm::config<128, 5, 2, 12, 2, true, 256, false, 2, 128>;
using localcta_onepass_sg_cfg7 = nvfp4_gemm::config<128, 5, 1, 12, 2, true, 256, false, 2, 128>;

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

__global__ void sum4_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ D,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
);

enum class OuterScaleReduceKind : int {
    FillScalar = 0,
    Pair1D = 1,
    ReduceRows = 2,
    PairReduceRows = 3,
    ReduceCols = 4,
    PairReduceCols = 5,
    PairMeanRows = 6,
    GlobalMeanFill = 7,
};

__global__ void reduce_outer_scale_tiles_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t tiles,
    int64_t dim0,
    int64_t dim1,
    int64_t stride0,
    int64_t stride1,
    int kind,
    float scale
);

void launch_batched_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
);

enum class V3ContractMode {
    OuterScale,
    TileGrid256,
};

V3ContractMode get_v3_contract_mode() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_CONTRACT");
    if (value == nullptr) {
        return V3ContractMode::OuterScale;
    }
    const std::string mode(value);
    if (mode == "tilegrid256" || mode == "tilegrid" || mode == "2d") {
        return V3ContractMode::TileGrid256;
    }
    return V3ContractMode::OuterScale;
}

bool use_v3_split3_batched_accum() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_SPLIT3_BATCHED_ACCUM");
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "0") != 0;
}

bool use_v3_split3_batched_accum_smalln() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_SPLIT3_BATCHED_ACCUM_SMALLN");
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "0") != 0;
}

bool use_v3_split2_onepass() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_SPLIT2_ONEPASS");
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "0") != 0;
}

bool use_v4_pack4_fold_outer_sg() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_PACK4_FOLD_OUTER_SG");
    if (value == nullptr) {
        return true;
    }
    return std::strcmp(value, "0") != 0;
}

bool use_v4_exact_gemm_selectors() {
    const char* value =
        std::getenv("USE_TK_LOCALCTA_V4_EXACT_GEMM_SELECTORS");
    if (value == nullptr) {
        return true;
    }
    return std::strcmp(value, "0") != 0;
}

bool use_v4_exact_residual_gemm_selectors() {
    const char* value =
        std::getenv("USE_TK_LOCALCTA_V4_EXACT_RESIDUAL_GEMM_SELECTORS");
    if (value == nullptr) {
        return true;
    }
    return std::strcmp(value, "0") != 0;
}

int get_chunkgrid_gemm_config_idx() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_CHUNKGRID_GEMM_CONFIG");
    if (value == nullptr) {
        return 7;
    }
    try {
        return std::stoi(std::string(value));
    } catch (...) {
        return 7;
    }
}

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

struct outer_scale_desc {
    const float* ptr;
    int stride;
};

enum class SGContractMode {
    ChunkGrid128,
    OuterScale,
    TileGrid256,
};

bool is_chunk_grid_tensor_shape(
    const at::Tensor& t,
    int64_t rows,
    int64_t cols
) {
    return t.defined() &&
           t.is_cuda() &&
           t.is_contiguous() &&
           t.dim() == 2 &&
           t.scalar_type() == at::kFloat &&
           t.size(0) == rows / 128 &&
           t.size(1) == cols / 128;
}

bool is_flat_chunk_grid_tensor_shape(
    const at::Tensor& t,
    int64_t rows,
    int64_t cols
) {
    return t.defined() &&
           t.is_cuda() &&
           t.is_contiguous() &&
           t.dim() == 1 &&
           t.scalar_type() == at::kFloat &&
           t.numel() == (rows / 128) * (cols / 128);
}

at::Tensor as_chunk_grid_tensor(
    const at::Tensor& t,
    int64_t rows,
    int64_t cols
) {
    if (is_chunk_grid_tensor_shape(t, rows, cols)) {
        return t;
    }
    if (is_flat_chunk_grid_tensor_shape(t, rows, cols)) {
        return t.view({rows / 128, cols / 128});
    }
    return t;
}

at::Tensor normalize_outer_scale_tiles_tensor(
    const at::Tensor& t,
    int64_t tiles,
    bool row_axis
) {
    TORCH_CHECK(t.defined(), "SG tensor must be defined");
    TORCH_CHECK(t.is_cuda(), "SG tensor must be CUDA");
    auto x = t.scalar_type() == at::kFloat ? t : t.to(torch::kFloat32);

    if (x.dim() == 0) {
        auto out = at::empty({tiles, 1}, x.options());
        const int threads = 256;
        const int blocks = static_cast<int>((tiles + threads - 1) / threads);
            auto stream = at::cuda::getCurrentCUDAStream();
            reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), tiles, 1, 1, 1, 1,
                static_cast<int>(OuterScaleReduceKind::FillScalar), 1.0f);
            CUDACHECK(cudaGetLastError());
        return out;
    }
    if (x.dim() == 1) {
        const int64_t stride0 = x.stride(0);
        if (x.numel() == tiles) {
            return x.view({tiles, 1});
        }
        if (x.numel() == tiles * 2) {
            auto out = at::empty({tiles, 1}, x.options());
            const int threads = 256;
            const int blocks = static_cast<int>((tiles + threads - 1) / threads);
            auto stream = at::cuda::getCurrentCUDAStream();
            reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), tiles, 1, 2, stride0, 1,
                static_cast<int>(OuterScaleReduceKind::Pair1D), 1.0f);
            CUDACHECK(cudaGetLastError());
            return out;
        }
        if (x.numel() == 1) {
            auto out = at::empty({tiles, 1}, x.options());
            const int threads = 256;
            const int blocks = static_cast<int>((tiles + threads - 1) / threads);
            auto stream = at::cuda::getCurrentCUDAStream();
            reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), tiles, 1, 1, stride0, 1,
                static_cast<int>(OuterScaleReduceKind::FillScalar), 1.0f);
            CUDACHECK(cudaGetLastError());
            return out;
        }
        TORCH_CHECK(false, "Unsupported 1D SG tensor length for outer-scale normalization");
    }

    TORCH_CHECK(x.dim() == 2, "SG tensor must be 0D, 1D, or 2D");
    if (row_axis) {
        if ((x.size(0) == tiles && x.size(1) == 1) ||
            (x.size(0) == 1 && x.size(1) == tiles)) {
            return x;
        }
    } else {
        if ((x.size(0) == 1 && x.size(1) == tiles) ||
            (x.size(0) == tiles && x.size(1) == 1)) {
            return x;
        }
    }
    auto out = at::empty({tiles, 1}, x.options());
    const int threads = 256;
    const int blocks = static_cast<int>((tiles + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t stride0 = x.stride(0);
    const int64_t stride1 = x.stride(1);
    if (x.size(0) == tiles || x.size(0) == tiles * 2) {
        const auto kind = x.size(0) == tiles * 2
            ? OuterScaleReduceKind::PairReduceRows
            : OuterScaleReduceKind::ReduceRows;
        reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), tiles, x.size(0), x.size(1), stride0, stride1,
            static_cast<int>(kind), 1.0f);
        CUDACHECK(cudaGetLastError());
        return out;
    }
    if (x.size(1) == tiles || x.size(1) == tiles * 2) {
        const auto kind = x.size(1) == tiles * 2
            ? OuterScaleReduceKind::PairReduceCols
            : OuterScaleReduceKind::ReduceCols;
        reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), tiles, x.size(0), x.size(1), stride0, stride1,
            static_cast<int>(kind), 1.0f);
        CUDACHECK(cudaGetLastError());
        return out;
    }
    TORCH_CHECK(false, "Unsupported 2D SG tensor shape for outer-scale normalization");
}

at::Tensor prepare_split_wgrad_a_sg_tensor(
    const at::Tensor& t,
    int64_t tiles,
    float scale
) {
    TORCH_CHECK(t.defined(), "SG tensor must be defined");
    TORCH_CHECK(t.is_cuda(), "SG tensor must be CUDA");
    auto x = t.scalar_type() == at::kFloat ? t : t.to(torch::kFloat32);
    if (x.dim() == 2 && x.size(0) == tiles * 2) {
        auto out = at::empty({tiles, 1}, x.options());
        const int threads = 256;
        const int blocks = static_cast<int>((tiles + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), tiles, x.size(0), x.size(1), x.stride(0), x.stride(1),
            static_cast<int>(OuterScaleReduceKind::GlobalMeanFill), scale);
        CUDACHECK(cudaGetLastError());
        return out;
    }
    return normalize_outer_scale_tiles_tensor(x, tiles, true);
}

at::Tensor prepare_split2_b_sg_tensor(
    const at::Tensor& t,
    int64_t tiles,
    float scale
) {
    TORCH_CHECK(t.defined(), "SG tensor must be defined");
    TORCH_CHECK(t.is_cuda(), "SG tensor must be CUDA");
    auto x = t.scalar_type() == at::kFloat ? t : t.to(torch::kFloat32);
    if (x.dim() == 2 && x.size(0) == tiles * 2) {
        auto out = at::empty({tiles, 1}, x.options());
        const int threads = 256;
        const int blocks = static_cast<int>((tiles + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), tiles, x.size(0), x.size(1), x.stride(0), x.stride(1),
            static_cast<int>(OuterScaleReduceKind::PairMeanRows), scale);
        CUDACHECK(cudaGetLastError());
        return out;
    }
    return normalize_outer_scale_tiles_tensor(x, tiles, true);
}

at::Tensor prepare_w2_dgrad_b_sg_tensor(
    const at::Tensor& t,
    int64_t tiles,
    float scale
) {
    TORCH_CHECK(t.defined(), "SG tensor must be defined");
    TORCH_CHECK(t.is_cuda(), "SG tensor must be CUDA");
    auto x = t.scalar_type() == at::kFloat ? t : t.to(torch::kFloat32);
    if (x.dim() == 2) {
        auto out = at::empty({tiles, 1}, x.options());
        const int threads = 256;
        const int blocks = static_cast<int>((tiles + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        reduce_outer_scale_tiles_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), tiles, x.size(0), x.size(1), x.stride(0), x.stride(1),
            static_cast<int>(OuterScaleReduceKind::GlobalMeanFill), scale);
        CUDACHECK(cudaGetLastError());
        return out;
    }
    return normalize_outer_scale_tiles_tensor(x, tiles, true);
}

at::Tensor fold_sg_into_prepared_sc_tensor(
    const at::Tensor& sc_raw,
    const at::Tensor& sg,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(sc_raw.defined(), "sc_raw must be defined");
    TORCH_CHECK(sc_raw.is_cuda(), "sc_raw must be CUDA");
    TORCH_CHECK(sc_raw.is_contiguous(), "sc_raw must be contiguous");
    TORCH_CHECK(sc_raw.scalar_type() == at::kFloat8_e4m3fn,
                "sc_raw must be fp8 e4m3fn prepared scales");
    TORCH_CHECK(sc_raw.dim() == 3, "sc_raw must be rank-3 prepared scales");
    TORCH_CHECK(sg.defined(), "SG tensor must be defined");
    TORCH_CHECK(sg.is_cuda(), "SG tensor must be CUDA");

    const int64_t row_tiles = rows / 128;
    const int64_t col_tiles = cols / 128;
    TORCH_CHECK(row_tiles > 0 && col_tiles > 0, "rows/cols must be multiples of 128");

    auto x = sg.scalar_type() == at::kFloat ? sg : sg.to(torch::kFloat32);
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    at::Tensor sg_grid;
    if (x.dim() == 0) {
        sg_grid = x.view({1, 1}).expand({row_tiles, col_tiles});
    } else if (x.dim() == 1) {
        if (x.numel() == row_tiles) {
            sg_grid = x.view({row_tiles, 1}).expand({row_tiles, col_tiles});
        } else if (x.numel() == col_tiles) {
            sg_grid = x.view({1, col_tiles}).expand({row_tiles, col_tiles});
        } else if (x.numel() == 1) {
            sg_grid = x.view({1, 1}).expand({row_tiles, col_tiles});
        } else {
            TORCH_CHECK(false, "Unsupported 1D SG tensor length for prepared-scale folding");
        }
    } else if (x.dim() == 2) {
        if (x.size(0) == row_tiles && x.size(1) == col_tiles) {
            sg_grid = x;
        } else if (x.size(0) == row_tiles && x.size(1) == 1) {
            sg_grid = x.expand({row_tiles, col_tiles});
        } else if (x.size(0) == 1 && x.size(1) == col_tiles) {
            sg_grid = x.expand({row_tiles, col_tiles});
        } else if (x.size(0) == 1 && x.size(1) == 1) {
            sg_grid = x.expand({row_tiles, col_tiles});
        } else {
            TORCH_CHECK(false, "Unsupported 2D SG tensor shape for prepared-scale folding");
        }
    } else {
        TORCH_CHECK(false, "SG tensor must be 0D, 1D, or 2D for prepared-scale folding");
    }

    auto sg_prepared = at::repeat_interleave(sg_grid, 2, 1).unsqueeze(-1);
    return (sc_raw.to(torch::kFloat32) * sg_prepared).contiguous().to(torch::kFloat8_e4m3fn);
}

__global__ void fold_outer_sg_into_prepared_sc_kernel(
    const __nv_fp8_e4m3* __restrict__ sc_raw,
    __nv_fp8_e4m3* __restrict__ sc_out,
    const float* __restrict__ sg_outer,
    int64_t stride0,
    int64_t stride1,
    int64_t scale_cols,
    int64_t elems
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int64_t tile_elems = scale_cols * kScaleBytesPerTile;
    const int64_t row_tile = idx / tile_elems;
    const int64_t rem = idx - row_tile * tile_elems;
    const int64_t col_tile = rem / kScaleBytesPerTile;
    const int64_t byte_idx = rem - col_tile * kScaleBytesPerTile;
    const float scale = sg_outer[row_tile >> 1];
    const int64_t src_idx = row_tile * stride0 + col_tile * stride1 + byte_idx;
    sc_out[idx] = __nv_fp8_e4m3(static_cast<float>(sc_raw[src_idx]) * scale);
}

__global__ void fold_outer_sg_into_prepared_sc_pack4_kernel(
    const __nv_fp8_e4m3* __restrict__ sc_raw,
    __nv_fp8_e4m3* __restrict__ sc_out,
    const float* __restrict__ sg_outer,
    int64_t stride0,
    int64_t stride1,
    int64_t scale_cols,
    int64_t packs
) {
    const int64_t pack_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pack_idx >= packs) return;

    const int64_t idx = pack_idx * 4;
    const int64_t tile_elems = scale_cols * kScaleBytesPerTile;
    const int64_t row_tile = idx / tile_elems;
    const int64_t rem = idx - row_tile * tile_elems;
    const int64_t col_tile = rem / kScaleBytesPerTile;
    const int64_t byte_idx = rem - col_tile * kScaleBytesPerTile;
    const float scale = sg_outer[row_tile >> 1];
    const int64_t src_idx = row_tile * stride0 + col_tile * stride1 + byte_idx;

    const fp8e4m3_4 packed_in = *reinterpret_cast<const fp8e4m3_4*>(sc_raw + src_idx);
    float4 unpacked = base_types::convertor<float4, fp8e4m3_4>::convert(packed_in);
    unpacked.x *= scale;
    unpacked.y *= scale;
    unpacked.z *= scale;
    unpacked.w *= scale;
    const fp8e4m3_4 packed_out = base_types::convertor<fp8e4m3_4, float4>::convert(unpacked);
    *reinterpret_cast<fp8e4m3_4*>(sc_out + idx) = packed_out;
}

at::Tensor fold_outer_sg_into_prepared_sc_tensor(
    const at::Tensor& sc_raw,
    const at::Tensor& sg,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(sc_raw.defined(), "sc_raw must be defined");
    TORCH_CHECK(sc_raw.is_cuda(), "sc_raw must be CUDA");
    TORCH_CHECK(sc_raw.scalar_type() == at::kFloat8_e4m3fn,
                "sc_raw must be fp8 e4m3fn prepared scales");
    TORCH_CHECK(sc_raw.dim() == 3, "sc_raw must be rank-3 prepared scales");
    TORCH_CHECK(rows % 256 == 0, "outer-SG prepared-scale folding requires rows multiple of 256");
    TORCH_CHECK(cols % 64 == 0, "prepared-scale folding requires cols multiple of 64");
    TORCH_CHECK(sc_raw.size(0) == rows / 128, "sc_raw first dim must be rows / 128");
    TORCH_CHECK(sc_raw.size(1) == cols / 64, "sc_raw second dim must be cols / 64");
    TORCH_CHECK(sc_raw.size(2) == kScaleBytesPerTile, "sc_raw scale-byte dim mismatch");
    TORCH_CHECK(sc_raw.stride(2) == 1, "sc_raw scale-byte dim must be contiguous");
    TORCH_CHECK(sc_raw.stride(1) == kScaleBytesPerTile,
                "sc_raw second-dim stride must equal 512 bytes");

    auto sg_outer = normalize_outer_scale_tiles_tensor(sg, rows / 256, true)
        .contiguous()
        .view({rows / 256});
    auto out = at::empty_like(sc_raw);

    const int threads = 256;
    const int64_t elems = sc_raw.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool pack4_safe =
        use_v4_pack4_fold_outer_sg() &&
        (elems % 4 == 0) &&
        (sc_raw.stride(0) % 4 == 0) &&
        (sc_raw.stride(1) % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(sc_raw.data_ptr()) & 0x3) == 0) &&
        ((reinterpret_cast<uintptr_t>(out.data_ptr()) & 0x3) == 0);
    if (pack4_safe) {
        const int64_t packs = elems / 4;
        const int blocks = static_cast<int>((packs + threads - 1) / threads);
        fold_outer_sg_into_prepared_sc_pack4_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(sc_raw.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr()),
            sg_outer.data_ptr<float>(),
            sc_raw.stride(0),
            sc_raw.stride(1),
            sc_raw.size(1),
            packs);
    } else {
        const int blocks = static_cast<int>((elems + threads - 1) / threads);
        fold_outer_sg_into_prepared_sc_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(sc_raw.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr()),
            sg_outer.data_ptr<float>(),
            sc_raw.stride(0),
            sc_raw.stride(1),
            sc_raw.size(1),
            elems);
    }
    CUDACHECK(cudaGetLastError());
    return out;
}

bool is_tilegrid256_tensor_shape(
    const at::Tensor& t,
    int64_t rows,
    int64_t cols
) {
    return t.defined() &&
           t.is_cuda() &&
           t.is_contiguous() &&
           t.dim() == 2 &&
           t.scalar_type() == at::kFloat &&
           t.size(0) == rows / 256 &&
           t.size(1) == cols / 256;
}

SGContractMode infer_regular_sg_contract(
    const at::Tensor& A,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sg
) {
    const int64_t K = A.size(1) * 2;
    const bool a_chunk =
        is_chunk_grid_tensor_shape(A_sg, A.size(0), K) ||
        is_flat_chunk_grid_tensor_shape(A_sg, A.size(0), K);
    const bool b_chunk =
        is_chunk_grid_tensor_shape(B_sg, B.size(0), K) ||
        is_flat_chunk_grid_tensor_shape(B_sg, B.size(0), K);
    if (a_chunk && b_chunk) {
        return SGContractMode::ChunkGrid128;
    }

    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        const bool a_tilegrid = is_tilegrid256_tensor_shape(A_sg, A.size(0), K);
        const bool b_tilegrid = is_tilegrid256_tensor_shape(B_sg, B.size(0), K);
        if (a_tilegrid && b_tilegrid) {
            return SGContractMode::TileGrid256;
        }
    }

    return SGContractMode::OuterScale;
}

outer_scale_desc check_outer_scale_tiles(
    const at::Tensor& t,
    const char* name,
    int64_t tiles,
    bool row_axis
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    if (t.dim() == 1) {
        TORCH_CHECK(t.size(0) == tiles, name, " length mismatch");
        return {t.data_ptr<float>(), 1};
    }
    TORCH_CHECK(t.dim() == 2, name, " must be 1D or 2D");
    if (row_axis) {
        if (t.size(0) == tiles && t.size(1) == 1) {
            return {t.data_ptr<float>(), static_cast<int>(t.stride(0))};
        }
        if (t.size(0) == 1 && t.size(1) == tiles) {
            return {t.data_ptr<float>(), static_cast<int>(t.stride(1))};
        }
    } else {
        if (t.size(0) == 1 && t.size(1) == tiles) {
            return {t.data_ptr<float>(), static_cast<int>(t.stride(1))};
        }
        if (t.size(0) == tiles && t.size(1) == 1) {
            return {t.data_ptr<float>(), static_cast<int>(t.stride(0))};
        }
    }
    TORCH_CHECK(false, name, " must have shape [tiles], [tiles,1], or [1,tiles]");
}

void check_output_matrix(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.size(0) == rows && t.size(1) == cols, name, " shape mismatch");
}

bool is_power_of_two_i64(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void check_rope_live64_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t seq_len
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == seq_len, name, " seq_len mismatch");
    TORCH_CHECK(t.size(1) == 32, name, " second dim must equal 32");
    TORCH_CHECK(t.size(2) == 2, name, " third dim must equal 2");
}

void check_rope_live64_qkv_args(
    const at::Tensor& D,
    const at::Tensor& D_K,
    const at::Tensor& D_V,
    const at::Tensor& rope_cs,
    int64_t rope_seq_len
) {
    TORCH_CHECK(rope_seq_len > 0, "rope_seq_len must be positive");
    TORCH_CHECK(is_power_of_two_i64(rope_seq_len), "rope_seq_len must be a power of two");
    TORCH_CHECK(D.size(0) % rope_seq_len == 0, "Q output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D_K.size(0) % rope_seq_len == 0, "K output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D.size(1) % 64 == 0, "Q output cols must be divisible by 64");
    TORCH_CHECK(D_K.size(1) % 64 == 0, "K output cols must be divisible by 64");
    TORCH_CHECK(D_V.size(1) % 128 == 0, "V output cols must be divisible by 128");
    check_rope_live64_tensor(rope_cs, "rope_cs", rope_seq_len);
    kittens::py::device_check(D, D_K, D_V, rope_cs);
}

void check_rope_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t seq_len,
    int64_t pair_dim
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == seq_len, name, " seq_len mismatch");
    TORCH_CHECK(t.size(1) == pair_dim, name, " pair dim mismatch");
}

void check_rope_qkv_args(
    const at::Tensor& D,
    const at::Tensor& D_K,
    const at::Tensor& D_V,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim
) {
    TORCH_CHECK(rope_seq_len > 0, "rope_seq_len must be positive");
    TORCH_CHECK(rope_head_dim > 0, "rope_head_dim must be positive");
    TORCH_CHECK(rope_rotary_dim > 0, "rope_rotary_dim must be positive");
    TORCH_CHECK((rope_head_dim % 2) == 0, "rope_head_dim must be even");
    TORCH_CHECK((rope_rotary_dim % 2) == 0, "rope_rotary_dim must be even");
    TORCH_CHECK(rope_rotary_dim <= rope_head_dim,
                "rope_rotary_dim must be <= rope_head_dim");
    TORCH_CHECK(D.size(0) % rope_seq_len == 0, "Q output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D_K.size(0) % rope_seq_len == 0, "K output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D.size(1) % rope_head_dim == 0, "Q output cols must be divisible by rope_head_dim");
    TORCH_CHECK(D_K.size(1) % rope_head_dim == 0, "K output cols must be divisible by rope_head_dim");
    TORCH_CHECK(D_V.size(1) % 128 == 0, "V output cols must be divisible by 128");
    const int64_t pair_dim = rope_rotary_dim / 2;
    check_rope_tensor(rope_cos, "rope_cos", rope_seq_len, pair_dim);
    check_rope_tensor(rope_sin, "rope_sin", rope_seq_len, pair_dim);
    kittens::py::device_check(D, D_K, D_V, rope_cos, rope_sin);
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

void check_v3_fast_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(0) % 128 == 0 && B.size(0) % 128 == 0, "M and N must be multiples of 128");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share packed K");
    TORCH_CHECK((A.size(1) * 2) % 128 == 0, "K must be a multiple of 128");
    check_scale_tensor(A_sc, "A_sc", A.size(0), A.size(1) * 2);
    check_scale_tensor(B_sc, "B_sc", B.size(0), B.size(1) * 2);
    (void)check_outer_scale_tiles(A_sg_tiles, "A_sg_tiles", A.size(0) / 256, true);
    (void)check_outer_scale_tiles(B_sg_tiles, "B_sg_tiles", B.size(0) / 256, false);
    kittens::py::device_check(A, A_sc, A_sg_tiles, B, B_sc, B_sg_tiles);
}

void check_tilegrid256_sg(
    const at::Tensor& t,
    const char* name,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == rows, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols, name, " second dim mismatch");
}

void check_v3_tilegrid256_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_grid,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_grid
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(0) % 256 == 0 && B.size(0) % 256 == 0,
                "tilegrid256 GEMM requires M and N to be multiples of 256");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share packed K");
    TORCH_CHECK((A.size(1) * 2) % 256 == 0, "tilegrid256 GEMM requires K to be a multiple of 256");
    check_scale_tensor(A_sc, "A_sc", A.size(0), A.size(1) * 2);
    check_scale_tensor(B_sc, "B_sc", B.size(0), B.size(1) * 2);
    check_tilegrid256_sg(A_sg_grid, "A_sg_grid", A.size(0) / 256, (A.size(1) * 2) / 256);
    check_tilegrid256_sg(B_sg_grid, "B_sg_grid", (B.size(1) * 2) / 256, B.size(0) / 256);
    kittens::py::device_check(A, A_sc, A_sg_grid, B, B_sc, B_sg_grid);
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
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel_reduction_scaled<C>>(g);
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
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel_reduction_scaled<C>>(g);
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

void launch_chunkgrid_batched_accum_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out
) {
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n >= 1 && n <= 4, "chunk-grid accum supports 1..4 batches");
    if (n == 1) {
        launch_regular_gemm(
            A_list[0], A_sc_list[0], A_sg_chunks_list[0],
            B_list[0], B_sc_list[0], B_sg_chunks_list[0],
            D_out);
        return;
    }
    std::vector<at::Tensor> D_list;
    D_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        D_list.push_back(at::empty_like(D_out));
    }
    for (int i = 0; i < n; ++i) {
        launch_regular_gemm(
            A_list[i], A_sc_list[i], A_sg_chunks_list[i],
            B_list[i], B_sc_list[i], B_sg_chunks_list[i],
            D_list[i]);
    }

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

template <typename C>
void launch_tilegrid256_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_grid,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_grid,
    at::Tensor& D
) {
    using G = nvfp4_localcta_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sg_chunks = A_sg_grid.data_ptr<float>(),
        .A_sg_stride = static_cast<int>(A_sg_grid.size(1)),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sg_chunks = B_sg_grid.data_ptr<float>(),
        .B_sg_stride = static_cast<int>(B_sg_grid.size(1)),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .v_dim = 0,
        .use_split_D = false,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel_reduction_scaled<C>>(g);
}

template <typename C>
void launch_tilegrid256_grouped_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_grid,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_grid,
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
        .A_sg_chunks = A_sg_grid.data_ptr<float>(),
        .A_sg_stride = static_cast<int>(A_sg_grid.size(1)),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sg_chunks = B_sg_grid.data_ptr<float>(),
        .B_sg_stride = static_cast<int>(B_sg_grid.size(1)),
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
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel_reduction_scaled<C>>(g);
}

void launch_tilegrid256_regular_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_grid,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_grid,
    at::Tensor& D
) {
    launch_tilegrid256_gemm_with_config<localcta_tilegrid256_config>(
        A, A_sc, A_sg_grid, B, B_sc, B_sg_grid, D);
}

void launch_tilegrid256_grouped_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_grid,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_grid,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    launch_tilegrid256_grouped_gemm_with_config<localcta_tilegrid256_config>(
        A, A_sc, A_sg_grid, B, B_sc, B_sg_grid, D, D_K_opt, D_V_opt, silu_dim);
}

template <typename C>
void launch_fast_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    at::Tensor& D
) {
    using G = nvfp4_gemm::globals<C>;
    auto one = get_unit_scale_tensor(A);
    const bool has_tile_scales = A_sg_tiles.defined() && B_sg_tiles.defined();
    outer_scale_desc a_sg_desc{nullptr, 1};
    outer_scale_desc b_sg_desc{nullptr, 1};
    if (has_tile_scales) {
        a_sg_desc = check_outer_scale_tiles(A_sg_tiles, "A_sg_tiles", A.size(0) / C::Mb, true);
        b_sg_desc = check_outer_scale_tiles(B_sg_tiles, "B_sg_tiles", B.size(0) / C::Nb, false);
    }
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
        .a_sg_per_tile = has_tile_scales ? a_sg_desc.ptr : nullptr,
        .a_sg_stride = has_tile_scales ? a_sg_desc.stride : 1,
        .b_sg_per_tile = has_tile_scales ? b_sg_desc.ptr : nullptr,
        .b_sg_stride = has_tile_scales ? b_sg_desc.stride : 1,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
void launch_fast_gemm_with_config_residual(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& R,
    at::Tensor& D,
    at::Tensor* row_rms_partial = nullptr
) {
    using G = nvfp4_gemm::globals<C>;
    auto one = get_unit_scale_tensor(A);
    const bool has_tile_scales = A_sg_tiles.defined() && B_sg_tiles.defined();
    outer_scale_desc a_sg_desc{nullptr, 1};
    outer_scale_desc b_sg_desc{nullptr, 1};
    if (has_tile_scales) {
        a_sg_desc = check_outer_scale_tiles(A_sg_tiles, "A_sg_tiles", A.size(0) / C::Mb, true);
        b_sg_desc = check_outer_scale_tiles(B_sg_tiles, "B_sg_tiles", B.size(0) / C::Nb, false);
    }
    check_output_matrix(R, "R", D.size(0), D.size(1));
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
        .a_sg_per_tile = has_tile_scales ? a_sg_desc.ptr : nullptr,
        .a_sg_stride = has_tile_scales ? a_sg_desc.stride : 1,
        .b_sg_per_tile = has_tile_scales ? b_sg_desc.ptr : nullptr,
        .b_sg_stride = has_tile_scales ? b_sg_desc.stride : 1,
        .silu_dim = 0,
        .row_rms_partial = row_rms_partial == nullptr
            ? nullptr : row_rms_partial->data_ptr<float>(),
        .row_rms_partial_stride = row_rms_partial == nullptr
            ? 0 : static_cast<int>(row_rms_partial->size(1))
    };
    auto r_gl = kittens::py::tensor_to_gl<typename G::D_gl>(R);
    std::memcpy(&g.R_tma, &r_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
void launch_fast_gemm_with_config_h(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& R,
    const at::Tensor& gamma,
    at::Tensor& D,
    at::Tensor& inverse_rms_tiles,
    at::Tensor& tile_amax,
    at::Tensor& row_outer_scales,
    at::Tensor& col_outer_scales,
    float eps
) {
    using G = nvfp4_gemm::globals<C>;
    auto one = get_unit_scale_tensor(A);
    const auto a_sg_desc = check_outer_scale_tiles(
        A_sg_tiles, "A_sg_tiles", A.size(0) / C::Mb, true);
    const auto b_sg_desc = check_outer_scale_tiles(
        B_sg_tiles, "B_sg_tiles", B.size(0) / C::Nb, false);
    check_output_matrix(R, "R", D.size(0), D.size(1));
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
        .a_sg_per_tile = a_sg_desc.ptr,
        .a_sg_stride = a_sg_desc.stride,
        .b_sg_per_tile = b_sg_desc.ptr,
        .b_sg_stride = b_sg_desc.stride,
        .silu_dim = 0,
        .h_r_tile = inverse_rms_tiles.data_ptr<float>(),
        .h_amax_tile = tile_amax.data_ptr<float>(),
        .h_gamma = reinterpret_cast<const kittens::bf16*>(gamma.data_ptr()),
        .h_cols = static_cast<int>(D.size(1)),
        .h_eps = eps,
    };
    auto residual_gl = kittens::py::tensor_to_gl<typename G::D_gl>(R);
    std::memcpy(
        &g.R_tma, &residual_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int tile_rows = static_cast<int>(D.size(0) / 128);
    const int tile_cols = static_cast<int>(D.size(1) / 128);
    h_nvfp4_tile_carrier::localcta_reduce_kernel<256><<<
        tile_rows / 2 + tile_cols / 2, 256, 0, stream>>>(
        tile_amax.data_ptr<float>(),
        row_outer_scales.data_ptr<float>(),
        col_outer_scales.data_ptr<float>(),
        tile_rows,
        tile_cols);
    CUDACHECK(cudaGetLastError());
}

template <typename C>
void launch_fast_grouped_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim,
    nvfp4_rope_epilogue::rope_live64_desc rope_live64 = {},
    nvfp4_rope_epilogue::rope_desc rope = {}
) {
    using G = nvfp4_gemm::globals<C>;
    const bool use_split_D = D_K_opt.has_value();
    const int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;
    auto one = get_unit_scale_tensor(A);
    const bool has_tile_scales = A_sg_tiles.defined() && B_sg_tiles.defined();
    outer_scale_desc a_sg_desc{nullptr, 1};
    outer_scale_desc b_sg_desc{nullptr, 1};
    if (has_tile_scales) {
        a_sg_desc = check_outer_scale_tiles(A_sg_tiles, "A_sg_tiles", A.size(0) / C::Mb, true);
        b_sg_desc = check_outer_scale_tiles(B_sg_tiles, "B_sg_tiles", B.size(0) / C::Nb, false);
    }

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
        .a_sg_per_tile = has_tile_scales ? a_sg_desc.ptr : nullptr,
        .a_sg_stride = has_tile_scales ? a_sg_desc.stride : 1,
        .b_sg_per_tile = has_tile_scales ? b_sg_desc.ptr : nullptr,
        .b_sg_stride = has_tile_scales ? b_sg_desc.stride : 1,
        .silu_dim = silu_dim,
        .rope = rope,
        .rope_live64 = rope_live64
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

bool has_virtual_rescale_chunk_grid(
    const at::Tensor& t,
    const char* name,
    int64_t rows,
    int64_t cols
) {
    if (!t.defined() || t.numel() == 0) {
        return false;
    }
    check_chunk_grid(t, name, rows, cols);
    return true;
}

template <typename C>
void launch_fast_gemm_virtual_rescale_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    using G = nvfp4_gemm::globals<C>;
    check_fast_gemm_inputs(A, A_sc, B, B_sc);
    outer_scale_desc a_sg_desc = check_outer_scale_tiles(A_sg_tiles, "A_sg_tiles", A.size(0) / C::Mb, true);
    outer_scale_desc b_sg_desc = check_outer_scale_tiles(B_sg_tiles, "B_sg_tiles", B.size(0) / C::Nb, false);
    const bool has_a_chunks = has_virtual_rescale_chunk_grid(
        A_sg_chunks, "A_sg_chunks", A.size(0), A.size(1) * 2);
    const bool has_b_chunks = has_virtual_rescale_chunk_grid(
        B_sg_chunks, "B_sg_chunks", B.size(0), B.size(1) * 2);
    TORCH_CHECK(has_a_chunks || has_b_chunks,
                "virtual-rescale GEMM requires at least one chunk SG grid");
    kittens::py::device_check(A, A_sc, A_sg_tiles, B, B_sc, B_sg_tiles);
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
        .a_sg_per_tile = a_sg_desc.ptr,
        .a_sg_stride = a_sg_desc.stride,
        .b_sg_per_tile = b_sg_desc.ptr,
        .b_sg_stride = b_sg_desc.stride,
        .a_sg_chunk_grid = has_a_chunks ? A_sg_chunks.data_ptr<float>() : nullptr,
        .a_sg_chunk_stride = has_a_chunks ? static_cast<int>(A_sg_chunks.size(1)) : 1,
        .b_sg_chunk_grid = has_b_chunks ? B_sg_chunks.data_ptr<float>() : nullptr,
        .b_sg_chunk_stride = has_b_chunks ? static_cast<int>(B_sg_chunks.size(1)) : 1,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel_virtual_rescale<C>>(g);
}

template <typename C>
void launch_fast_grouped_gemm_virtual_rescale_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    using G = nvfp4_gemm::globals<C>;
    const bool use_split_D = D_K_opt.has_value();
    const int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;
    check_fast_gemm_inputs(A, A_sc, B, B_sc);
    outer_scale_desc a_sg_desc = check_outer_scale_tiles(A_sg_tiles, "A_sg_tiles", A.size(0) / C::Mb, true);
    outer_scale_desc b_sg_desc = check_outer_scale_tiles(B_sg_tiles, "B_sg_tiles", B.size(0) / C::Nb, false);
    const bool has_a_chunks = has_virtual_rescale_chunk_grid(
        A_sg_chunks, "A_sg_chunks", A.size(0), A.size(1) * 2);
    const bool has_b_chunks = has_virtual_rescale_chunk_grid(
        B_sg_chunks, "B_sg_chunks", B.size(0), B.size(1) * 2);
    TORCH_CHECK(has_a_chunks || has_b_chunks,
                "virtual-rescale grouped GEMM requires at least one chunk SG grid");
    kittens::py::device_check(A, A_sc, A_sg_tiles, B, B_sc, B_sg_tiles);
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
        .a_sg_per_tile = a_sg_desc.ptr,
        .a_sg_stride = a_sg_desc.stride,
        .b_sg_per_tile = b_sg_desc.ptr,
        .b_sg_stride = b_sg_desc.stride,
        .a_sg_chunk_grid = has_a_chunks ? A_sg_chunks.data_ptr<float>() : nullptr,
        .a_sg_chunk_stride = has_a_chunks ? static_cast<int>(A_sg_chunks.size(1)) : 1,
        .b_sg_chunk_grid = has_b_chunks ? B_sg_chunks.data_ptr<float>() : nullptr,
        .b_sg_chunk_stride = has_b_chunks ? static_cast<int>(B_sg_chunks.size(1)) : 1,
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel_virtual_rescale<C>>(g);
}

void launch_fast_regular_gemm_virtual_rescale(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    const int64_t reduction_k = A.size(1) * 2;
    if (reduction_k > 2048) {
        launch_fast_gemm_virtual_rescale_with_config<localcta_fast_largek_config>(
            A, A_sc, A_sg_tiles, A_sg_chunks, B, B_sc, B_sg_tiles, B_sg_chunks, D);
        return;
    }
    launch_fast_gemm_virtual_rescale_with_config<localcta_fast_smallk_config>(
        A, A_sc, A_sg_tiles, A_sg_chunks, B, B_sc, B_sg_tiles, B_sg_chunks, D);
}

void launch_fast_grouped_gemm_virtual_rescale(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    const bool use_split_D = D_K_opt.has_value();
    const int64_t reduction_k = A.size(1) * 2;
    if (!use_split_D && reduction_k > 2048) {
        launch_fast_grouped_gemm_virtual_rescale_with_config<localcta_fast_largek_config>(
            A, A_sc, A_sg_tiles, A_sg_chunks, B, B_sc, B_sg_tiles, B_sg_chunks,
            D, D_K_opt, D_V_opt, silu_dim);
        return;
    }
    launch_fast_grouped_gemm_virtual_rescale_with_config<localcta_fast_grouped_config>(
        A, A_sc, A_sg_tiles, A_sg_chunks, B, B_sc, B_sg_tiles, B_sg_chunks,
        D, D_K_opt, D_V_opt, silu_dim);
}

void launch_fast_regular_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    at::Tensor& D
) {
    const int64_t M = D.size(0);
    const int64_t N = D.size(1);
    const int64_t K = A.size(1) * 2;
    const bool exact_selectors = (
        use_v4_exact_gemm_selectors()
        && A_sg_tiles.defined() && A_sg_tiles.numel() > 0
        && B_sg_tiles.defined() && B_sg_tiles.numel() > 0
    );
    if (exact_selectors) {
        if (
            (M == 32768 && N == 4096 && K == 4096)
            || (M == 24576 && N == 4096 && K == 4096)
            || (M == 24576 && N == 4096 && K == 8192)
        ) {
            launch_fast_gemm_with_config<localcta_fast_config28>(
                A, A_sc_prepared, A_sg_tiles,
                B, B_sc_prepared, B_sg_tiles, D);
            return;
        }
        if (M == 32768 && N == 14336 && K == 4096) {
            launch_fast_gemm_with_config<localcta_fast_config25>(
                A, A_sc_prepared, A_sg_tiles,
                B, B_sc_prepared, B_sg_tiles, D);
            return;
        }
        if (M == 4096 && N == 14336 && K == 32768) {
            launch_fast_gemm_with_config<localcta_fast_config6>(
                A, A_sc_prepared, A_sg_tiles,
                B, B_sc_prepared, B_sg_tiles, D);
            return;
        }
        if (
            (M == 32768 && N == 21504 && K == 4096)
            || (M == 24576 && N == 21504 && K == 4096)
            || (M == 24576 && N == 4096 && K == 21504)
            || (M == 21504 && N == 4096 && K == 24576)
            || (M == 24576 && N == 18688 && K == 4096)
        ) {
            launch_fast_gemm_with_config<localcta_fast_config12>(
                A, A_sc_prepared, A_sg_tiles,
                B, B_sc_prepared, B_sg_tiles, D);
            return;
        }
        if (M == 24576 && N == 4096 && K == 18688) {
            launch_fast_gemm_with_config<localcta_fast_config13>(
                A, A_sc_prepared, A_sg_tiles,
                B, B_sc_prepared, B_sg_tiles, D);
            return;
        }
        if (M == 18688 && N == 4096 && K == 24576) {
            launch_fast_gemm_with_config<localcta_fast_config12>(
                A, A_sc_prepared, A_sg_tiles,
                B, B_sc_prepared, B_sg_tiles, D);
            return;
        }
        if (M == 24576 && N == 8192 && K == 4096) {
            launch_fast_gemm_with_config<localcta_fast_config27>(
                A, A_sc_prepared, A_sg_tiles,
                B, B_sc_prepared, B_sg_tiles, D);
            return;
        }
    }
    if (K <= 2048) {
        launch_fast_gemm_with_config<localcta_fast_smallk_config>(
            A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles, D);
    } else {
        launch_fast_gemm_with_config<localcta_fast_largek_config>(
            A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles, D);
    }
}

void launch_fast_regular_gemm_config(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    at::Tensor& D,
    int config_id
) {
#define LOCALCTA_FAST_GEMM_CONFIG_CASES(X) \
    X(0,  256, 4, 16,  1, 2, false) \
    X(1,  256, 4, 16,  4, 2, false) \
    X(2,  256, 4, 16, 12, 2, false) \
    X(3,  256, 5,  8,  4, 2, true ) \
    X(4,  256, 5,  8, 12, 2, true ) \
    X(5,  256, 5,  8,  4, 2, false) \
    X(6,  256, 4,  8, 12, 2, false) \
    X(7,  128, 5,  4, 12, 2, true ) \
    X(8,  128, 4,  4, 12, 2, false) \
    X(9,  128, 5,  4,  4, 2, true ) \
    X(10, 256, 5, 16,  4, 2, true ) \
    X(11, 256, 5, 16, 12, 2, true ) \
    X(12, 256, 5,  8,  1, 2, false) \
    X(13, 256, 5,  8,  2, 2, false) \
    X(14, 256, 5,  8,  8, 2, false) \
    X(15, 256, 5,  8, 12, 2, false) \
    X(16, 256, 5,  8,  1, 2, true ) \
    X(17, 256, 5,  8,  2, 2, true ) \
    X(18, 256, 5,  8,  8, 2, true ) \
    X(19, 256, 5, 16,  1, 2, false) \
    X(20, 256, 5, 16,  2, 2, false) \
    X(21, 256, 5, 16,  8, 2, false) \
    X(22, 256, 5, 16,  1, 2, true ) \
    X(23, 256, 5, 16,  2, 2, true ) \
    X(24, 256, 5, 16,  8, 2, true ) \
    X(25, 256, 4,  8,  1, 2, false) \
    X(26, 256, 4,  8,  4, 2, false) \
    X(27, 256, 4,  8,  2, 2, false) \
    X(28, 256, 4,  8,  8, 2, false) \
    X(29, 256, 3,  8,  4, 2, false) \
    X(30, 256, 3, 16,  4, 2, false) \
    X(31, 256, 4, 16,  2, 2, false) \
    X(32, 256, 4, 16,  8, 2, false) \
    X(33, 256, 4, 16,  1, 2, true ) \
    X(34, 256, 4, 16,  4, 2, true ) \
    X(35, 128, 5,  8,  4, 2, false) \
    X(36, 128, 5,  8, 12, 2, false) \
    X(37, 128, 5,  8,  1, 2, false) \
    X(38, 128, 5,  8,  4, 2, true ) \
    X(39, 128, 5,  8, 12, 2, true ) \
    X(40, 128, 4,  8,  4, 2, false) \
    X(41, 128, 4,  8, 12, 2, false) \
    X(42, 128, 5,  4,  1, 2, false) \
    X(43, 128, 5,  4,  2, 2, false) \
    X(44, 128, 5,  4,  1, 2, true ) \
    X(45, 128, 4,  4,  4, 2, false) \
    X(46, 128, 4,  4,  1, 2, false)
#define LOCALCTA_FAST_GEMM_DISPATCH_CASE(ID, NB, LP, EP, SG, DT, OVERLAP) \
        case ID: \
            launch_fast_gemm_with_config< \
                nvfp4_gemm::config<NB, LP, EP, SG, DT, OVERLAP>>( \
                    A, A_sc_prepared, A_sg_tiles, \
                    B, B_sc_prepared, B_sg_tiles, D); \
            return;
    switch (config_id) {
        LOCALCTA_FAST_GEMM_CONFIG_CASES(
            LOCALCTA_FAST_GEMM_DISPATCH_CASE)
        default:
            TORCH_CHECK(
                false,
                "Unsupported localCTA fast GEMM config_id=", config_id,
                " (valid: 0-46)");
    }
#undef LOCALCTA_FAST_GEMM_DISPATCH_CASE
#undef LOCALCTA_FAST_GEMM_CONFIG_CASES
}

void launch_fast_regular_gemm_residual(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& R,
    at::Tensor& D
) {
    const int64_t M = D.size(0);
    const int64_t N = D.size(1);
    const int64_t K = A.size(1) * 2;
    const bool exact_selectors = (
        use_v4_exact_residual_gemm_selectors()
        && A_sg_tiles.defined() && A_sg_tiles.numel() > 0
        && B_sg_tiles.defined() && B_sg_tiles.numel() > 0
    );
    if (exact_selectors && M == 32768 && N == 4096 && K == 21504) {
        launch_fast_gemm_with_config_residual<localcta_fast_config28_residual_config>(
            A, A_sc_prepared, A_sg_tiles,
            B, B_sc_prepared, B_sg_tiles, R, D);
        return;
    }
    if (K <= 2048) {
        launch_fast_gemm_with_config_residual<localcta_fast_smallk_residual_config>(
            A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles, R, D);
    } else {
        launch_fast_gemm_with_config_residual<localcta_fast_largek_residual_config>(
            A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles, R, D);
    }
}

void launch_fast_regular_gemm_residual_rms(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& R,
    at::Tensor& D,
    at::Tensor& row_rms_partial
) {
    const int64_t K = A.size(1) * 2;
    const char* pipeline_env = std::getenv("USE_TK_LOCALCTA_V4_RESIDUAL_PIPELINE");
    const bool use_pipeline =
        pipeline_env == nullptr || std::string(pipeline_env) != "0";
    if (K <= 2048) {
        if (use_pipeline) {
            launch_fast_gemm_with_config_residual<localcta_fast_smallk_residual_rms_pipeline_config>(
                A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
                R, D, &row_rms_partial);
        } else {
            launch_fast_gemm_with_config_residual<localcta_fast_smallk_residual_rms_config>(
                A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
                R, D, &row_rms_partial);
        }
    } else {
        if (use_pipeline) {
            launch_fast_gemm_with_config_residual<localcta_fast_largek_residual_rms_pipeline_config>(
                A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
                R, D, &row_rms_partial);
        } else {
            launch_fast_gemm_with_config_residual<localcta_fast_largek_residual_rms_config>(
                A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
                R, D, &row_rms_partial);
        }
    }
}

void launch_fast_regular_gemm_h(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& R,
    const at::Tensor& gamma,
    at::Tensor& D,
    at::Tensor& inverse_rms_tiles,
    at::Tensor& tile_amax,
    at::Tensor& row_outer_scales,
    at::Tensor& col_outer_scales,
    float eps
) {
    const int64_t K = A.size(1) * 2;
    if (K <= 2048) {
        launch_fast_gemm_with_config_h<localcta_fast_smallk_h_config>(
            A, A_sc, A_sg_tiles, B, B_sc, B_sg_tiles, R, gamma, D,
            inverse_rms_tiles, tile_amax,
            row_outer_scales, col_outer_scales, eps);
    } else {
        launch_fast_gemm_with_config_h<localcta_fast_largek_h_config>(
            A, A_sc, A_sg_tiles, B, B_sc, B_sg_tiles, R, gamma, D,
            inverse_rms_tiles, tile_amax,
            row_outer_scales, col_outer_scales, eps);
    }
}

void launch_fast_grouped_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim,
    nvfp4_rope_epilogue::rope_live64_desc rope_live64 = {}
) {
    const bool use_split_D = D_K_opt.has_value();
    const int64_t reduction_k = A.size(1) * 2;
    if (!use_split_D && reduction_k > 2048) {
        launch_fast_grouped_gemm_with_config<localcta_fast_largek_config>(
            A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
            D, D_K_opt, D_V_opt, silu_dim, rope_live64);
        return;
    }
    launch_fast_grouped_gemm_with_config<localcta_fast_grouped_config>(
        A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
        D, D_K_opt, D_V_opt, silu_dim, rope_live64);
}

void launch_fast_grouped_gemm_rope_live64(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim,
    nvfp4_rope_epilogue::rope_live64_desc rope_live64
) {
    const bool use_split_D = D_K_opt.has_value();
    const int64_t reduction_k = A.size(1) * 2;
    if (!use_split_D && reduction_k > 2048) {
        launch_fast_grouped_gemm_with_config<localcta_fast_largek_rope_config>(
            A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
            D, D_K_opt, D_V_opt, silu_dim, rope_live64);
        return;
    }
    launch_fast_grouped_gemm_with_config<localcta_fast_grouped_rope_config>(
        A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
        D, D_K_opt, D_V_opt, silu_dim, rope_live64);
}

void launch_fast_grouped_gemm_rope(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& B_sg_tiles,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim,
    nvfp4_rope_epilogue::rope_desc rope
) {
    const bool use_split_D = D_K_opt.has_value();
    const int64_t reduction_k = A.size(1) * 2;
    if (!use_split_D && reduction_k > 2048) {
        launch_fast_grouped_gemm_with_config<localcta_fast_largek_rope_config>(
            A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
            D, D_K_opt, D_V_opt, silu_dim, {}, rope);
        return;
    }
    launch_fast_grouped_gemm_with_config<localcta_fast_grouped_rope_config>(
        A, A_sc_prepared, A_sg_tiles, B, B_sc_prepared, B_sg_tiles,
        D, D_K_opt, D_V_opt, silu_dim, {}, rope);
}

template <typename C>
void launch_fast_gemm_chunkgrid_sg_with_config(
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
        .a_sg_stride = static_cast<int>(A_sg_chunks.size(1)),
        .b_sg_per_tile = B_sg_chunks.data_ptr<float>(),
        .b_sg_stride = static_cast<int>(B_sg_chunks.size(1)),
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel_chunk_grid<C>>(g);
}

template <typename C>
void launch_fast_grouped_gemm_chunkgrid_sg_with_config(
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
        .a_sg_stride = static_cast<int>(A_sg_chunks.size(1)),
        .b_sg_per_tile = B_sg_chunks.data_ptr<float>(),
        .b_sg_stride = static_cast<int>(B_sg_chunks.size(1)),
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel_chunk_grid<C>>(g);
}

void launch_fast_regular_gemm_chunkgrid_sg(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    switch (get_chunkgrid_gemm_config_idx()) {
        case 6:
            launch_fast_gemm_chunkgrid_sg_with_config<localcta_fast_chunkgrid_smalln_epi2_config>(
                A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
            break;
        case 7:
            launch_fast_gemm_chunkgrid_sg_with_config<localcta_fast_chunkgrid_smalln_epi1_config>(
                A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
            break;
        case 1:
        default:
            launch_fast_gemm_chunkgrid_sg_with_config<localcta_fast_chunkgrid_smalln_config>(
                A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
            break;
    }
}

void launch_fast_grouped_gemm_chunkgrid_sg(
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
    switch (get_chunkgrid_gemm_config_idx()) {
        case 6:
            launch_fast_grouped_gemm_chunkgrid_sg_with_config<localcta_fast_chunkgrid_smalln_epi2_config>(
                A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D, D_K_opt, D_V_opt, silu_dim);
            break;
        case 7:
            launch_fast_grouped_gemm_chunkgrid_sg_with_config<localcta_fast_chunkgrid_smalln_epi1_config>(
                A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D, D_K_opt, D_V_opt, silu_dim);
            break;
        case 1:
        default:
            launch_fast_grouped_gemm_chunkgrid_sg_with_config<localcta_fast_chunkgrid_smalln_config>(
                A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D, D_K_opt, D_V_opt, silu_dim);
            break;
    }
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
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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

    for (int i = 0; i < n; ++i) {
        g_host.num_red_blocks[i] = static_cast<int>((2 * A_list[i].size(1)) / C::Kb);
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
        if (!A_sg_tiles_list.empty() && i < static_cast<int>(A_sg_tiles_list.size()) &&
            A_sg_tiles_list[i].defined() && A_sg_tiles_list[i].numel() > 0 &&
            !B_sg_tiles_list.empty() && i < static_cast<int>(B_sg_tiles_list.size()) &&
            B_sg_tiles_list[i].defined() && B_sg_tiles_list[i].numel() > 0) {
            auto a_sg_desc = check_outer_scale_tiles(A_sg_tiles_list[i], "A_sg_tiles_list[i]", M / C::Mb, true);
            auto b_sg_desc = check_outer_scale_tiles(B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / C::Nb, false);
            g_host.A_sg[i] = a_sg_desc.ptr;
            g_host.B_sg[i] = b_sg_desc.ptr;
            g_host.A_sg_stride[i] = a_sg_desc.stride;
            g_host.B_sg_stride[i] = b_sg_desc.stride;
        } else {
            g_host.A_sg[i] = one_ptr;
            g_host.B_sg[i] = one_ptr;
            g_host.A_sg_stride[i] = 0;
            g_host.B_sg_stride[i] = 0;
        }
    }

    kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
}

void launch_fast_batched_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    std::vector<at::Tensor>& D_list
) {
    const int spatial_tiles =
        static_cast<int>((D_list[0].size(0) / localcta_fast_batched_config::Mb) *
                         (D_list[0].size(1) / localcta_fast_batched_config::Nb));

    // Keep the old per-batch path only for tiny launches that cannot form a full cluster.
    if (spatial_tiles < localcta_fast_batched_config::CLUSTER_SIZE) {
        for (size_t i = 0; i < A_list.size(); ++i) {
            launch_fast_regular_gemm(
                A_list[i], A_sc_prepared_list[i], A_sg_tiles_list[i],
                B_list[i], B_sc_prepared_list[i], B_sg_tiles_list[i],
                D_list[i]);
        }
        return;
    }

    launch_fast_batched_gemm_with_config<localcta_fast_batched_config>(
        A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_list);
}

template <typename C>
void launch_fast_batched_accum_gemm_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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
        g_host.num_red_blocks[i] = static_cast<int>((2 * A_list[i].size(1)) / C::Kb);
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
        if (!A_sg_tiles_list.empty() && i < static_cast<int>(A_sg_tiles_list.size()) &&
            A_sg_tiles_list[i].defined() && A_sg_tiles_list[i].numel() > 0 &&
            !B_sg_tiles_list.empty() && i < static_cast<int>(B_sg_tiles_list.size()) &&
            B_sg_tiles_list[i].defined() && B_sg_tiles_list[i].numel() > 0) {
            auto a_sg_desc = check_outer_scale_tiles(A_sg_tiles_list[i], "A_sg_tiles_list[i]", M / C::Mb, true);
            auto b_sg_desc = check_outer_scale_tiles(B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / C::Nb, false);
            g_host.A_sg[i] = a_sg_desc.ptr;
            g_host.B_sg[i] = b_sg_desc.ptr;
            g_host.A_sg_stride[i] = a_sg_desc.stride;
            g_host.B_sg_stride[i] = b_sg_desc.stride;
        } else {
            g_host.A_sg[i] = one_ptr;
            g_host.B_sg[i] = one_ptr;
            g_host.A_sg_stride[i] = 0;
            g_host.B_sg_stride[i] = 0;
        }
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_batched_accum_gemm_strided_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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
    auto one = get_unit_scale_tensor(A_full);
    const float* one_ptr = one.data_ptr<float>();

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
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
        g_host.num_red_blocks[i] = static_cast<int>((2 * fp4_cols) / C::Kb);
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
        if (!A_sg_tiles_list.empty() && i < static_cast<int>(A_sg_tiles_list.size()) &&
            A_sg_tiles_list[i].defined() && A_sg_tiles_list[i].numel() > 0 &&
            !B_sg_tiles_list.empty() && i < static_cast<int>(B_sg_tiles_list.size()) &&
            B_sg_tiles_list[i].defined() && B_sg_tiles_list[i].numel() > 0) {
            auto a_sg_desc = check_outer_scale_tiles(A_sg_tiles_list[i], "A_sg_tiles_list[i]", M / C::Mb, true);
            auto b_sg_desc = check_outer_scale_tiles(B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / C::Nb, false);
            g_host.A_sg[i] = a_sg_desc.ptr;
            g_host.B_sg[i] = b_sg_desc.ptr;
            g_host.A_sg_stride[i] = a_sg_desc.stride;
            g_host.B_sg_stride[i] = b_sg_desc.stride;
        } else {
            g_host.A_sg[i] = one_ptr;
            g_host.B_sg[i] = one_ptr;
            g_host.A_sg_stride[i] = 0;
            g_host.B_sg_stride[i] = 0;
        }
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_batched_accum_gemm_strided_v3_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    at::Tensor& D_out
) {
    using G = nvfp4_batched_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_sc_prepared_list.size());
    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    auto one = get_unit_scale_tensor(A_full);
    const float* one_ptr = one.data_ptr<float>();

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    for (int i = 0; i < n; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        g_host.num_red_blocks[i] = static_cast<int>((2 * fp4_cols) / C::Kb);
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

        if (!A_sg_tiles_list.empty() && i < static_cast<int>(A_sg_tiles_list.size()) &&
            A_sg_tiles_list[i].defined() && A_sg_tiles_list[i].numel() > 0 &&
            !B_sg_tiles_list.empty() && i < static_cast<int>(B_sg_tiles_list.size()) &&
            B_sg_tiles_list[i].defined() && B_sg_tiles_list[i].numel() > 0) {
            // Exact v3 outerscale SG is always defined on 256x256 output tiles,
            // regardless of the internal GEMM tile shape used by this kernel.
            auto a_sg_desc = check_outer_scale_tiles(A_sg_tiles_list[i], "A_sg_tiles_list[i]", M / 256, true);
            auto b_sg_desc = check_outer_scale_tiles(B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / 256, false);
            g_host.A_sg[i] = a_sg_desc.ptr;
            g_host.B_sg[i] = b_sg_desc.ptr;
            g_host.A_sg_stride[i] = a_sg_desc.stride;
            g_host.B_sg_stride[i] = b_sg_desc.stride;
        } else {
            g_host.A_sg[i] = one_ptr;
            g_host.B_sg[i] = one_ptr;
            g_host.A_sg_stride[i] = 0;
            g_host.B_sg_stride[i] = 0;
        }
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    if (n == 3) {
        kittens::py::launch_kernel<C, G, nvfp4_batched_accum_gemm::kernel_fixed3<C>>(g_host);
        return;
    }
    kittens::py::launch_kernel<C, G, nvfp4_batched_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_batched_gemm_strided_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    for (int i = 0; i < n; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        g_host.num_red_blocks[i] = static_cast<int>((2 * fp4_cols) / C::Kb);
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

        if (!A_sg_tiles_list.empty() && i < static_cast<int>(A_sg_tiles_list.size()) &&
            A_sg_tiles_list[i].defined() && A_sg_tiles_list[i].numel() > 0 &&
            !B_sg_tiles_list.empty() && i < static_cast<int>(B_sg_tiles_list.size()) &&
            B_sg_tiles_list[i].defined() && B_sg_tiles_list[i].numel() > 0) {
            auto a_sg_desc = check_outer_scale_tiles(
                A_sg_tiles_list[i], "A_sg_tiles_list[i]", A_full.size(0) / C::Mb, true);
            auto b_sg_desc = check_outer_scale_tiles(
                B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / C::Nb, false);
            g_host.A_sg[i] = a_sg_desc.ptr;
            g_host.B_sg[i] = b_sg_desc.ptr;
            g_host.A_sg_stride[i] = a_sg_desc.stride;
            g_host.B_sg_stride[i] = b_sg_desc.stride;
        } else {
            g_host.A_sg[i] = one_ptr;
            g_host.B_sg[i] = one_ptr;
            g_host.A_sg_stride[i] = 0;
            g_host.B_sg_stride[i] = 0;
        }
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
void encode_fp4_col_slice_tensor_map(
    CUtensorMap* desc,
    const at::Tensor& full,
    int64_t col_offset,
    int64_t col_width,
    const char* name
) {
    check_fp4_matrix(full, name);
    TORCH_CHECK(col_offset >= 0, name, " col_offset must be non-negative");
    TORCH_CHECK(col_width > 0, name, " col_width must be positive");
    TORCH_CHECK(col_offset + col_width <= full.size(1), name, " slice exceeds packed width");
    TORCH_CHECK((2 * col_width) % C::Kb == 0,
                name, " reduction width must be aligned to Kb=", C::Kb);

    constexpr int64_t swizzle_elements = 128;
    const int64_t rows = full.size(0);
    const int64_t row_stride = full.size(1);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(full.data_ptr());
    const void* data_ptr = base + col_offset;

    uint64_t gmem_shape[5] = {
        static_cast<uint64_t>(swizzle_elements),
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>((col_width + swizzle_elements - 1) / swizzle_elements),
        1, 1
    };
    uint64_t gmem_stride[4] = {
        static_cast<uint64_t>(row_stride),
        128,
        static_cast<uint64_t>(rows * row_stride),
        static_cast<uint64_t>(rows * row_stride)
    };
    uint32_t smem_shape[5] = {
        static_cast<uint32_t>(swizzle_elements),
        static_cast<uint32_t>(C::Nb / 2),
        1, 1, 1
    };
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        desc,
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
    TORCH_CHECK(result == CUDA_SUCCESS, name, " FP4 slice TMA creation failed");
}

template <typename C>
void launch_fast_split3_dgrad_gemm_strided_onepass_full_b_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const at::Tensor& B_full,
    const at::Tensor& B_sc_prepared_full,
    const at::Tensor& B_sg_tiles_full,
    at::Tensor& D_out
) {
    using G = nvfp4_split3_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_sc_prepared_list.size() == 3, "one-pass split3 full-B dgrad expects 3 A scale batches");
    TORCH_CHECK(A_col_offsets.size() == 3, "one-pass split3 full-B dgrad expects 3 A offsets");
    TORCH_CHECK(A_col_widths.size() == 3, "one-pass split3 full-B dgrad expects 3 A widths");
    const bool has_outer_sg = !A_sg_tiles_list.empty() && B_sg_tiles_full.defined() && B_sg_tiles_full.numel() > 0;
    if (has_outer_sg) {
        TORCH_CHECK(A_sg_tiles_list.size() == 3, "one-pass split3 full-B direct SG expects 3 A SG batches");
    }
    check_fp4_matrix(A_full, "A_full");
    check_fp4_matrix(B_full, "B_full");
    TORCH_CHECK(A_full.size(1) == B_full.size(1), "A_full and B_full must share packed reduction width");
    TORCH_CHECK(D_out.size(0) == A_full.size(0), "D_out row mismatch");
    TORCH_CHECK(D_out.size(1) == B_full.size(0), "D_out col mismatch");
    check_scale_tensor_tma_compatible(
        B_sc_prepared_full, "B_sc_prepared_full", B_full.size(0), B_full.size(1) * 2);

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    std::vector<at::Tensor> A_sg_outer_list;
    at::Tensor B_sg_outer;
    if (has_outer_sg) {
        A_sg_outer_list.reserve(3);
        for (int i = 0; i < 3; ++i) {
            A_sg_outer_list.push_back(
                normalize_outer_scale_tiles_tensor(A_sg_tiles_list[i], M / 256, true)
                    .contiguous());
        }
        B_sg_outer = normalize_outer_scale_tiles_tensor(
            B_sg_tiles_full, B_full.size(0) / 256, true).contiguous();
    }

    for (int i = 0; i < 3; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        const int64_t fp4_offset = A_col_offsets[i];
        const void* data_ptr = a_base + fp4_offset;

        TORCH_CHECK(fp4_cols > 0, "A_col_widths must be positive");
        TORCH_CHECK((2 * fp4_cols) % C::Kb == 0,
                    "one-pass split3 full-B dgrad expects reduction widths aligned to Kb=", C::Kb);
        TORCH_CHECK(fp4_offset % 32 == 0,
                    "one-pass split3 full-B dgrad expects packed offsets aligned to scale tiles");
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
                    "One-pass split3 full-B localCTA A TMA creation failed for batch ", i);

        encode_fp4_col_slice_tensor_map<C>(
            &g_host.B_tma[i], B_full, fp4_offset, fp4_cols, "B_full");

        const int64_t sc_offset = fp4_offset / 32;
        const int64_t sc_width = fp4_cols / 32;
        auto B_sc_view = B_sc_prepared_full.narrow(1, sc_offset, sc_width);

        encode_prepared_scale_tensor_map<typename G::A_sc_tile>(
            &g_host.A_sc_tma[i], A_sc_prepared_list[i], "A_sc_prepared_list[i]");
        encode_prepared_scale_tensor_map<typename G::B_sc_tile>(
            &g_host.B_sc_tma[i], B_sc_view, "B_sc_prepared_full_slice");
        if (has_outer_sg) {
            auto a_sg_desc = check_outer_scale_tiles(
                A_sg_outer_list[i], "A_sg_tiles_list[i]", M / 256, true);
            auto b_sg_desc = check_outer_scale_tiles(
                B_sg_outer, "B_sg_tiles_full", B_full.size(0) / 256, true);
            g_host.A_sg[i] = a_sg_desc.ptr;
            g_host.B_sg[i] = b_sg_desc.ptr;
            g_host.A_sg_stride[i] = a_sg_desc.stride;
            g_host.B_sg_stride[i] = b_sg_desc.stride;
        }
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split3_accum_gemm::kernel<C>>(g_host);
}

void launch_fast_split3_dgrad_gemm_strided_onepass_full_b(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const at::Tensor& B_full,
    const at::Tensor& B_sc_prepared_full,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "one-pass split3 full-B config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_fast_split3_dgrad_gemm_strided_onepass_full_b_with_config<localcta_onepass_cfg1>(
                A_full, A_sc_prepared_list, std::vector<at::Tensor>(), A_col_offsets, A_col_widths,
                B_full, B_sc_prepared_full, at::Tensor(), D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split3 full-B config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split3_dgrad_gemm_strided_onepass_full_b_with_config<localcta_onepass_cfg3>(
                A_full, A_sc_prepared_list, std::vector<at::Tensor>(), A_col_offsets, A_col_widths,
                B_full, B_sc_prepared_full, at::Tensor(), D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split3 full-B config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split3_dgrad_gemm_strided_onepass_full_b_with_config<localcta_onepass_cfg5>(
                A_full, A_sc_prepared_list, std::vector<at::Tensor>(), A_col_offsets, A_col_widths,
                B_full, B_sc_prepared_full, at::Tensor(), D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split3 full-B config_idx=", resolved_idx);
    }
}

void launch_fast_split3_dgrad_gemm_strided_onepass_full_b_sg(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const at::Tensor& B_full,
    const at::Tensor& B_sc_prepared_full,
    const at::Tensor& B_sg_tiles_full,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "one-pass split3 full-B direct-SG config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_fast_split3_dgrad_gemm_strided_onepass_full_b_with_config<localcta_onepass_cfg1>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_full, B_sc_prepared_full, B_sg_tiles_full, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split3 full-B direct-SG config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split3_dgrad_gemm_strided_onepass_full_b_with_config<localcta_onepass_cfg3>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_full, B_sc_prepared_full, B_sg_tiles_full, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split3 full-B direct-SG config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split3_dgrad_gemm_strided_onepass_full_b_with_config<localcta_onepass_cfg5>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_full, B_sc_prepared_full, B_sg_tiles_full, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split3 full-B direct-SG config_idx=", resolved_idx);
    }
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
    auto one = get_unit_scale_tensor(A_full);
    const float* one_ptr = one.data_ptr<float>();

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
        g_host.A_sg[i] = one_ptr;
        g_host.B_sg[i] = one_ptr;
        g_host.A_sg_stride[i] = 0;
        g_host.B_sg_stride[i] = 0;
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

template <typename C>
void launch_fast_split2_dgrad_gemm_strided_onepass_outer_sg_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 A scale batches");
    TORCH_CHECK(A_sg_tiles_list.size() == 2, "one-pass split2 dgrad expects 2 A SG batches");
    TORCH_CHECK(B_list.size() == 2, "one-pass split2 dgrad expects 2 B batches");
    TORCH_CHECK(B_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 B scale batches");
    TORCH_CHECK(B_sg_tiles_list.size() == 2, "one-pass split2 dgrad expects 2 B SG batches");
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
        constexpr int64_t swizzle_elements = C::Kb / 2;
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
            static_cast<uint64_t>(swizzle_elements),
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
            swizzle_elements == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
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

        auto a_sg_desc = check_outer_scale_tiles(
            A_sg_tiles_list[i], "A_sg_tiles_list[i]", M / C::Mb, true);
        auto b_sg_desc = check_outer_scale_tiles(
            B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / C::Nb, false);
        g_host.A_sg[i] = a_sg_desc.ptr;
        g_host.B_sg[i] = b_sg_desc.ptr;
        g_host.A_sg_stride[i] = a_sg_desc.stride;
        g_host.B_sg_stride[i] = b_sg_desc.stride;
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split2_accum_gemm::kernel<C>>(g_host);
}

void launch_fast_split2_dgrad_gemm_strided_onepass_outer_sg(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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
            launch_fast_split2_dgrad_gemm_strided_onepass_outer_sg_with_config<localcta_onepass_cfg1>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split2 config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split2_dgrad_gemm_strided_onepass_outer_sg_with_config<localcta_onepass_cfg3>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split2 config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split2_dgrad_gemm_strided_onepass_outer_sg_with_config<localcta_onepass_cfg5>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split2 outer-SG config_idx=", resolved_idx);
    }
}

template <typename C>
void launch_fast_split2_dgrad_gemm_strided_onepass_sg_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 A scale batches");
    TORCH_CHECK(A_sg_tiles_list.size() == 2, "one-pass split2 dgrad expects 2 A SG batches");
    TORCH_CHECK(B_list.size() == 2, "one-pass split2 dgrad expects 2 B batches");
    TORCH_CHECK(B_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 B scale batches");
    TORCH_CHECK(B_sg_tiles_list.size() == 2, "one-pass split2 dgrad expects 2 B SG batches");
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
        static_assert(C::Kb == 128, "strided split2 SG path expects 128-wide reduction chunks");
        constexpr int64_t swizzle_elements = C::Kb / 2;
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
            static_cast<uint64_t>(swizzle_elements),
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
            CU_TENSOR_MAP_SWIZZLE_64B,
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
        check_chunk_grid(A_sg_tiles_list[i], "A_sg_tiles_list[i]", M, fp4_cols * 2);
        check_chunk_grid(B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0), B_list[i].size(1) * 2);
        g_host.A_sg[i] = A_sg_tiles_list[i].data_ptr<float>();
        g_host.B_sg[i] = B_sg_tiles_list[i].data_ptr<float>();
        g_host.A_sg_stride[i] = static_cast<int>(A_sg_tiles_list[i].size(1));
        g_host.B_sg_stride[i] = static_cast<int>(B_sg_tiles_list[i].size(1));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split2_accum_gemm::kernel_chunk_grid<C>>(g_host);
}

void launch_fast_split2_dgrad_gemm_strided_onepass_sg(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "one-pass split2 SG config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_fast_split2_dgrad_gemm_strided_onepass_sg_with_config<localcta_onepass_sg_cfg1>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split2 SG config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split2_dgrad_gemm_strided_onepass_sg_with_config<localcta_onepass_sg_cfg3>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split2 SG config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split2_dgrad_gemm_strided_onepass_sg_with_config<localcta_onepass_sg_cfg5>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 6:
            launch_fast_split2_dgrad_gemm_strided_onepass_sg_with_config<localcta_onepass_sg_cfg6>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 7:
            launch_fast_split2_dgrad_gemm_strided_onepass_sg_with_config<localcta_onepass_sg_cfg7>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
                B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split2 SG config_idx=", resolved_idx);
    }
}

void launch_fast_batched_accum_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out
) {
    if (A_list.size() == 1) {
        launch_fast_regular_gemm(
            A_list[0], A_sc_prepared_list[0], A_sg_tiles_list[0],
            B_list[0], B_sc_prepared_list[0], B_sg_tiles_list[0],
            D_out);
        return;
    }
    const int64_t reduction_k = A_list[0].size(1) * 2;
    if (A_list.size() == 2 && reduction_k >= 4096) {
        launch_fast_batched_accum_gemm_with_config<localcta_fast_largek_config>(
            A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
        return;
    }
    launch_fast_batched_accum_gemm_with_config<localcta_fast_batched_config>(
        A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
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
    std::vector<at::Tensor> empty_a_sg(A_list.size(), torch::Tensor());
    std::vector<at::Tensor> empty_b_sg(B_list.size(), torch::Tensor());
    const int64_t q_reduction_k = A_list[0].size(1) * 2;
    if (q_reduction_k >= 2048) {
        launch_fast_batched_accum_gemm_with_config<localcta_fast_split3_dgrad_config>(
            A_list, A_sc_prepared_list, empty_a_sg, B_list, B_sc_prepared_list, empty_b_sg, D_out);
        return;
    }
    launch_fast_batched_accum_gemm_with_config<localcta_fast_batched_config>(
        A_list, A_sc_prepared_list, empty_a_sg, B_list, B_sc_prepared_list, empty_b_sg, D_out);
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
    auto one = get_unit_scale_tensor(D_out);
    const float* one_ptr = one.data_ptr<float>();
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
        g_host.A_sg[i] = one_ptr;
        g_host.B_sg[i] = one_ptr;
        g_host.A_sg_stride[i] = 0;
        g_host.B_sg_stride[i] = 0;
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split2_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_fast_split2_dgrad_gemm_onepass_sg_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(A_list.size() == 2, "one-pass split2 dgrad expects 2 A batches");
    TORCH_CHECK(A_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 A scale batches");
    TORCH_CHECK(A_sg_tiles_list.size() == 2, "one-pass split2 dgrad expects 2 A SG batches");
    TORCH_CHECK(B_list.size() == 2, "one-pass split2 dgrad expects 2 B batches");
    TORCH_CHECK(B_sc_prepared_list.size() == 2, "one-pass split2 dgrad expects 2 B scale batches");
    TORCH_CHECK(B_sg_tiles_list.size() == 2, "one-pass split2 dgrad expects 2 B SG batches");

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
        check_chunk_grid(A_sg_tiles_list[i], "A_sg_tiles_list[i]", A_list[i].size(0), A_list[i].size(1) * 2);
        check_chunk_grid(B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0), B_list[i].size(1) * 2);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        g_host.A_sg[i] = A_sg_tiles_list[i].data_ptr<float>();
        g_host.B_sg[i] = B_sg_tiles_list[i].data_ptr<float>();
        g_host.A_sg_stride[i] = static_cast<int>(A_sg_tiles_list[i].size(1));
        g_host.B_sg_stride[i] = static_cast<int>(B_sg_tiles_list[i].size(1));
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
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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
                A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "one-pass split2 config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_fast_split2_dgrad_gemm_onepass_sg_with_config<localcta_onepass_sg_cfg3>(
                A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "one-pass split2 config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_fast_split2_dgrad_gemm_onepass_sg_with_config<localcta_onepass_sg_cfg5>(
                A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 6:
            launch_fast_split2_dgrad_gemm_onepass_sg_with_config<localcta_onepass_sg_cfg6>(
                A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        case 7:
            launch_fast_split2_dgrad_gemm_onepass_sg_with_config<localcta_onepass_sg_cfg7>(
                A_list, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown one-pass split2 config_idx=", resolved_idx);
    }
}

template <typename C>
void launch_v3_split2_dgrad_gemm_onepass_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out
) {
    using G = nvfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "v3 split2 one-pass exact path only supports outerscale contract");
    TORCH_CHECK(A_list.size() == 2, "v3 split2 one-pass dgrad expects 2 A batches");
    TORCH_CHECK(A_sc_list.size() == 2, "v3 split2 one-pass dgrad expects 2 A scale batches");
    TORCH_CHECK(A_sg_tiles_list.size() == 2, "v3 split2 one-pass dgrad expects 2 A SG batches");
    TORCH_CHECK(B_list.size() == 2, "v3 split2 one-pass dgrad expects 2 B batches");
    TORCH_CHECK(B_sc_list.size() == 2, "v3 split2 one-pass dgrad expects 2 B scale batches");
    TORCH_CHECK(B_sg_tiles_list.size() == 2, "v3 split2 one-pass dgrad expects 2 B SG batches");

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    auto one = get_unit_scale_tensor(A_list[0]);
    const float* one_ptr = one.data_ptr<float>();
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    for (int i = 0; i < 2; ++i) {
        g_host.num_red_blocks[i] = static_cast<int>((2 * A_list[i].size(1)) / C::Kb);

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

        if (A_sg_tiles_list[i].defined() && A_sg_tiles_list[i].numel() > 0 &&
            B_sg_tiles_list[i].defined() && B_sg_tiles_list[i].numel() > 0) {
            auto a_sg_desc = check_outer_scale_tiles(
                A_sg_tiles_list[i], "A_sg_tiles_list[i]", M / C::Mb, true);
            auto b_sg_desc = check_outer_scale_tiles(
                B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / C::Nb, false);
            g_host.A_sg[i] = a_sg_desc.ptr;
            g_host.B_sg[i] = b_sg_desc.ptr;
            g_host.A_sg_stride[i] = a_sg_desc.stride;
            g_host.B_sg_stride[i] = b_sg_desc.stride;
        } else {
            g_host.A_sg[i] = one_ptr;
            g_host.B_sg[i] = one_ptr;
            g_host.A_sg_stride[i] = 0;
            g_host.B_sg_stride[i] = 0;
        }
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, nvfp4_split2_accum_gemm::kernel<C>>(g_host);
}

void launch_v3_split2_dgrad_gemm_onepass(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 0:
            TORCH_CHECK(false, "v3 one-pass split2 config_idx=0 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 1:
            launch_v3_split2_dgrad_gemm_onepass_with_config<localcta_onepass_cfg1>(
                A_list, A_sc_list, A_sg_tiles_list, B_list, B_sc_list, B_sg_tiles_list, D_out);
            break;
        case 2:
            TORCH_CHECK(false, "v3 one-pass split2 config_idx=2 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 3:
            launch_v3_split2_dgrad_gemm_onepass_with_config<localcta_onepass_cfg3>(
                A_list, A_sc_list, A_sg_tiles_list, B_list, B_sc_list, B_sg_tiles_list, D_out);
            break;
        case 4:
            TORCH_CHECK(false, "v3 one-pass split2 config_idx=4 is not legal with CLUSTER_SIZE=1 on this kernel");
            break;
        case 5:
            launch_v3_split2_dgrad_gemm_onepass_with_config<localcta_onepass_cfg5>(
                A_list, A_sc_list, A_sg_tiles_list, B_list, B_sc_list, B_sg_tiles_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown v3 one-pass split2 config_idx=", resolved_idx);
    }
}

void launch_fast_split3_dgrad_gemm_strided(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out
) {
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects 3 A scale batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects 3 B batches");
    const int64_t max_reduction_k =
        2 * (*std::max_element(A_col_widths.begin(), A_col_widths.end()));
    if (use_v3_split3_batched_accum() &&
        !A_sg_tiles_list.empty() &&
        !B_sg_tiles_list.empty() &&
        A_sg_tiles_list.size() == 3 &&
        B_sg_tiles_list.size() == 3) {
        if (use_v3_split3_batched_accum_smalln()) {
            launch_fast_batched_accum_gemm_strided_v3_with_config<localcta_fast_split3_dgrad_sg_smalln_config>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
                A_col_offsets, A_col_widths, D_out);
            return;
        }
        if (max_reduction_k >= 2048) {
            launch_fast_batched_accum_gemm_strided_v3_with_config<localcta_fast_split3_dgrad_config>(
                A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
                A_col_offsets, A_col_widths, D_out);
            return;
        }
        launch_fast_batched_accum_gemm_strided_v3_with_config<localcta_fast_batched_config>(
            A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
            A_col_offsets, A_col_widths, D_out);
        return;
    }
    if (max_reduction_k >= 2048) {
        launch_fast_batched_accum_gemm_strided_with_config<localcta_fast_split3_dgrad_config>(
            A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
            A_col_offsets, A_col_widths, D_out);
        return;
    }
    launch_fast_batched_accum_gemm_strided_with_config<localcta_fast_batched_config>(
        A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
        A_col_offsets, A_col_widths, D_out);
}

void launch_fast_split3_dgrad_gemm_strided_sum(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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
            A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
            A_col_offsets, A_col_widths, D_list);
    } else {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_batched_config>(
            A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
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
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
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
            A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
            A_col_offsets, A_col_widths, D_list);
    } else {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_batched_config>(
            A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
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
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    std::vector<at::Tensor>& D_list
) {
    const int64_t max_reduction_k =
        2 * (*std::max_element(A_col_widths.begin(), A_col_widths.end()));
    if (max_reduction_k >= 2048) {
        launch_fast_batched_gemm_strided_with_config<localcta_fast_split3_dgrad_config>(
            A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
            A_col_offsets, A_col_widths, D_list);
        return;
    }
    launch_fast_batched_gemm_strided_with_config<localcta_fast_batched_config>(
        A_full, A_sc_prepared_list, A_sg_tiles_list, B_list, B_sc_prepared_list, B_sg_tiles_list,
        A_col_offsets, A_col_widths, D_list);
}

__global__ void reduce_outer_scale_tiles_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t tiles,
    int64_t dim0,
    int64_t dim1,
    int64_t stride0,
    int64_t stride1,
    int kind,
    float scale
) {
    const int64_t tile_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tile_idx >= tiles) {
        return;
    }

    float acc = 0.0f;
    switch (static_cast<OuterScaleReduceKind>(kind)) {
        case OuterScaleReduceKind::FillScalar:
            acc = in[0];
            break;
        case OuterScaleReduceKind::Pair1D:
            acc = fmaxf(in[(2 * tile_idx) * stride0], in[(2 * tile_idx + 1) * stride0]);
            break;
        case OuterScaleReduceKind::ReduceRows:
            acc = in[tile_idx * stride0];
            for (int64_t j = 1; j < dim1; ++j) {
                acc = fmaxf(acc, in[tile_idx * stride0 + j * stride1]);
            }
            break;
        case OuterScaleReduceKind::PairReduceRows:
            acc = fmaxf(in[(2 * tile_idx) * stride0], in[(2 * tile_idx + 1) * stride0]);
            for (int64_t j = 1; j < dim1; ++j) {
                const float v0 = in[(2 * tile_idx) * stride0 + j * stride1];
                const float v1 = in[(2 * tile_idx + 1) * stride0 + j * stride1];
                acc = fmaxf(acc, fmaxf(v0, v1));
            }
            break;
        case OuterScaleReduceKind::ReduceCols:
            acc = in[tile_idx * stride1];
            for (int64_t i = 1; i < dim0; ++i) {
                acc = fmaxf(acc, in[i * stride0 + tile_idx * stride1]);
            }
            break;
        case OuterScaleReduceKind::PairReduceCols:
            acc = fmaxf(in[(2 * tile_idx) * stride1], in[(2 * tile_idx + 1) * stride1]);
            for (int64_t i = 1; i < dim0; ++i) {
                const int64_t base = i * stride0 + (2 * tile_idx) * stride1;
                acc = fmaxf(acc, fmaxf(in[base], in[base + stride1]));
            }
            break;
        case OuterScaleReduceKind::PairMeanRows: {
            float sum = 0.0f;
            const int64_t row0 = 2 * tile_idx;
            const int64_t row1 = row0 + 1;
            for (int64_t j = 0; j < dim1; ++j) {
                sum += in[row0 * stride0 + j * stride1] + in[row1 * stride0 + j * stride1];
            }
            acc = (sum / static_cast<float>(2 * dim1)) * scale;
            out[tile_idx] = acc;
            return;
        }
        case OuterScaleReduceKind::GlobalMeanFill: {
            float sum = 0.0f;
            for (int64_t i = 0; i < dim0; ++i) {
                for (int64_t j = 0; j < dim1; ++j) {
                    sum += in[i * stride0 + j * stride1];
                }
            }
            acc = (sum / static_cast<float>(dim0 * dim1)) * scale;
            out[tile_idx] = acc;
            return;
        }
    }
    out[tile_idx] = acc;
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
    const auto sg_contract = infer_regular_sg_contract(A, A_sg_chunks, B, B_sg_chunks);
    if (sg_contract == SGContractMode::ChunkGrid128) {
        auto A_sg_chunkgrid = as_chunk_grid_tensor(A_sg_chunks, A.size(0), A.size(1) * 2);
        auto B_sg_chunkgrid = as_chunk_grid_tensor(B_sg_chunks, B.size(0), B.size(1) * 2);
        check_gemm_inputs(A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid);
        check_output_matrix(D, "D", A.size(0), B.size(0));
        launch_regular_gemm(A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid, D);
        return;
    }
    if (sg_contract == SGContractMode::TileGrid256) {
        check_v3_tilegrid256_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
        check_output_matrix(D, "D", A.size(0), B.size(0));
        launch_fast_regular_gemm(A, A_sc, torch::Tensor(), B, B_sc, torch::Tensor(), D);
        return;
    }
    auto A_sg_outer = normalize_outer_scale_tiles_tensor(A_sg_chunks, A.size(0) / 256, true);
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(B_sg_chunks, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm(A, A_sc, A_sg_outer, B, B_sc, B_sg_outer, D);
}

void nvfp4_localcta_fast_gemm_outer_sg_config_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    at::Tensor& D,
    int config_id
) {
    auto A_sg_outer =
        normalize_outer_scale_tiles_tensor(A_sg, A.size(0) / 256, true);
    auto B_sg_outer =
        normalize_outer_scale_tiles_tensor(B_sg, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm_config(
        A, A_sc, A_sg_outer,
        B, B_sc, B_sg_outer,
        D, config_id);
}

void nvfp4_localcta_gemm_residual_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    const at::Tensor& R,
    at::Tensor& D
) {
    const auto sg_contract = infer_regular_sg_contract(A, A_sg_chunks, B, B_sg_chunks);
    TORCH_CHECK(
        sg_contract != SGContractMode::ChunkGrid128,
        "nvfp4_localcta_gemm_residual only supports prepared TileGrid256 or outer-SG contracts");
    if (sg_contract == SGContractMode::TileGrid256) {
        check_v3_tilegrid256_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
        check_output_matrix(D, "D", A.size(0), B.size(0));
        launch_fast_regular_gemm_residual(A, A_sc, torch::Tensor(), B, B_sc, torch::Tensor(), R, D);
        return;
    }
    auto A_sg_outer = normalize_outer_scale_tiles_tensor(A_sg_chunks, A.size(0) / 256, true);
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(B_sg_chunks, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm_residual(A, A_sc, A_sg_outer, B, B_sc, B_sg_outer, R, D);
}

void nvfp4_localcta_gemm_residual_rms_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    const at::Tensor& R,
    at::Tensor& D,
    at::Tensor& row_rms_partial
) {
    const auto sg_contract = infer_regular_sg_contract(
        A, A_sg_chunks, B, B_sg_chunks);
    TORCH_CHECK(
        sg_contract == SGContractMode::OuterScale,
        "localCTA exact residual RMS requires the v4 outer-SG contract");
    auto A_sg_outer = normalize_outer_scale_tiles_tensor(
        A_sg_chunks, A.size(0) / 256, true);
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(
        B_sg_chunks, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
    const int64_t M = A.size(0);
    const int64_t N = B.size(0);
    check_output_matrix(R, "R", M, N);
    check_output_matrix(D, "D", M, N);
    TORCH_CHECK(
        N <= 4096 && N % 32 == 0,
        "localCTA exact residual RMS requires 32-aligned N <= 4096");
    TORCH_CHECK(
        row_rms_partial.is_cuda() && row_rms_partial.is_contiguous() &&
            row_rms_partial.scalar_type() == at::kFloat &&
            row_rms_partial.dim() == 2 &&
            row_rms_partial.size(0) == M &&
            row_rms_partial.size(1) == N / 256,
        "row_rms_partial must be contiguous CUDA float32 [M,N/256]");
    kittens::py::device_check(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer,
        R, D, row_rms_partial);
    at::assert_no_overlap(row_rms_partial, A);
    at::assert_no_overlap(row_rms_partial, A_sc);
    at::assert_no_overlap(row_rms_partial, A_sg_outer);
    at::assert_no_overlap(row_rms_partial, B);
    at::assert_no_overlap(row_rms_partial, B_sc);
    at::assert_no_overlap(row_rms_partial, B_sg_outer);
    at::assert_no_overlap(row_rms_partial, R);
    at::assert_no_overlap(row_rms_partial, D);
    const c10::cuda::CUDAGuard device_guard(A.device());
    launch_fast_regular_gemm_residual_rms(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer,
        R, D, row_rms_partial);
}

void nvfp4_localcta_h_residual_carrier_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    const at::Tensor& R,
    const at::Tensor& gamma,
    at::Tensor& D,
    at::Tensor& inverse_rms_tiles,
    at::Tensor& tile_amax,
    at::Tensor& row_outer_scales,
    at::Tensor& col_outer_scales,
    double eps
) {
    const auto sg_contract = infer_regular_sg_contract(A, A_sg, B, B_sg);
    TORCH_CHECK(
        sg_contract == SGContractMode::OuterScale,
        "localCTA H carrier requires the v4 outer-scale contract");
    auto A_sg_outer = normalize_outer_scale_tiles_tensor(
        A_sg, A.size(0) / 256, true);
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(
        B_sg, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
    const int64_t M = A.size(0);
    const int64_t N = B.size(0);
    TORCH_CHECK(
        M % 256 == 0 && N % 256 == 0,
        "localCTA H carrier requires M and N multiples of 256");
    check_output_matrix(R, "R", M, N);
    check_output_matrix(D, "D", M, N);
    TORCH_CHECK(
        gamma.is_cuda() && gamma.is_contiguous() && gamma.dim() == 1 &&
            gamma.scalar_type() == at::kBFloat16 && gamma.numel() == N,
        "localCTA H gamma must be contiguous CUDA bf16 [N]");
    TORCH_CHECK(
        inverse_rms_tiles.is_cuda() && inverse_rms_tiles.is_contiguous() &&
            inverse_rms_tiles.scalar_type() == at::kFloat &&
            inverse_rms_tiles.sizes() ==
                at::IntArrayRef({M / 128, N / 128}),
        "localCTA H inverse_rms_tiles must be float32 [M/128,N/128]");
    TORCH_CHECK(
        tile_amax.is_cuda() && tile_amax.is_contiguous() &&
            tile_amax.scalar_type() == at::kFloat &&
            tile_amax.sizes() == inverse_rms_tiles.sizes(),
        "localCTA H tile_amax must match inverse_rms_tiles");
    TORCH_CHECK(
        row_outer_scales.is_cuda() && row_outer_scales.is_contiguous() &&
            row_outer_scales.scalar_type() == at::kFloat &&
            row_outer_scales.numel() == M / 256,
        "localCTA H row_outer_scales must contain M/256 float32 values");
    TORCH_CHECK(
        col_outer_scales.is_cuda() && col_outer_scales.is_contiguous() &&
            col_outer_scales.scalar_type() == at::kFloat &&
            col_outer_scales.numel() == N / 256,
        "localCTA H col_outer_scales must contain N/256 float32 values");
    TORCH_CHECK(
        std::isfinite(eps) && eps >= 0.0,
        "localCTA H epsilon must be finite and non-negative");
    kittens::py::device_check(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer, R, gamma, D,
        inverse_rms_tiles, tile_amax,
        row_outer_scales, col_outer_scales);
    const c10::cuda::CUDAGuard device_guard(A.device());
    launch_fast_regular_gemm_h(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer, R, gamma, D,
        inverse_rms_tiles, tile_amax,
        row_outer_scales, col_outer_scales,
        static_cast<float>(eps));
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
    launch_fast_regular_gemm(A, A_sc_prepared, torch::Tensor(), B, B_sc_prepared, torch::Tensor(), D);
}

void nvfp4_localcta_fast_gemm_residual_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc_prepared,
    const at::Tensor& B,
    const at::Tensor& B_sc_prepared,
    const at::Tensor& R,
    at::Tensor& D
) {
    check_fast_gemm_inputs(A, A_sc_prepared, B, B_sc_prepared);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm_residual(A, A_sc_prepared, torch::Tensor(), B, B_sc_prepared, torch::Tensor(), R, D);
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
    auto A_sg_chunkgrid = as_chunk_grid_tensor(A_sg_chunks, A.size(0), A.size(1) * 2);
    auto B_sg_chunkgrid = as_chunk_grid_tensor(B_sg_chunks, B.size(0), B.size(1) * 2);
    check_gemm_inputs(A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm_chunkgrid_sg(A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid, D);
}

void nvfp4_localcta_fast_gemm_virtual_rescale_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_fast_regular_gemm_virtual_rescale(
        A, A_sc, A_sg_tiles, A_sg_chunks,
        B, B_sc, B_sg_tiles, B_sg_chunks,
        D);
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
    const auto sg_contract = infer_regular_sg_contract(A, A_sg_chunks, B, B_sg_chunks);
    if (sg_contract == SGContractMode::ChunkGrid128) {
        auto A_sg_chunkgrid = as_chunk_grid_tensor(A_sg_chunks, A.size(0), A.size(1) * 2);
        auto B_sg_chunkgrid = as_chunk_grid_tensor(B_sg_chunks, B.size(0), B.size(1) * 2);
        check_gemm_inputs(A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid);
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
            A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid,
            D, D_K_opt, D_V_opt, silu_dim);
        return;
    }
    if (sg_contract == SGContractMode::TileGrid256) {
        check_v3_tilegrid256_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
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
            A, A_sc, torch::Tensor(), B, B_sc, torch::Tensor(),
            D, D_K_opt, D_V_opt, silu_dim);
        return;
    }
    auto A_sg_outer = normalize_outer_scale_tiles_tensor(A_sg_chunks, A.size(0) / 256, true);
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(B_sg_chunks, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
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
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer,
        D, D_K_opt, D_V_opt, silu_dim);
}

void nvfp4_localcta_grouped_gemm_rope_live64_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    at::Tensor& D_K,
    at::Tensor& D_V,
    const at::Tensor& rope_cs,
    int64_t rope_seq_len,
    int silu_dim = 0
) {
    const auto sg_contract = infer_regular_sg_contract(A, A_sg_chunks, B, B_sg_chunks);
    TORCH_CHECK(sg_contract != SGContractMode::ChunkGrid128,
                "live64 RoPE epilogue is only implemented for fast outer-scale localCTA grouped GEMM");
    TORCH_CHECK(D.size(1) + D_K.size(1) + D_V.size(1) == B.size(0),
                "Q/K/V output columns must sum to B rows");

    check_output_matrix(D, "D", A.size(0), D.size(1));
    check_output_matrix(D_K, "D_K", A.size(0), D_K.size(1));
    check_output_matrix(D_V, "D_V", A.size(0), D_V.size(1));
    check_rope_live64_qkv_args(D, D_K, D_V, rope_cs, rope_seq_len);

    nvfp4_rope_epilogue::rope_live64_desc rope_live64 {
        .cs = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>()),
        .seq_len = static_cast<int>(rope_seq_len),
        .seq_mask = static_cast<int>(rope_seq_len - 1),
    };

    if (sg_contract == SGContractMode::TileGrid256) {
        check_v3_tilegrid256_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
        launch_fast_grouped_gemm_rope_live64(
            A, A_sc, torch::Tensor(), B, B_sc, torch::Tensor(),
            D, std::optional<at::Tensor>(D_K), std::optional<at::Tensor>(D_V), silu_dim, rope_live64);
        return;
    }

    auto A_sg_outer = normalize_outer_scale_tiles_tensor(A_sg_chunks, A.size(0) / 256, true);
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(B_sg_chunks, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
    launch_fast_grouped_gemm_rope_live64(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer,
        D, std::optional<at::Tensor>(D_K), std::optional<at::Tensor>(D_V), silu_dim, rope_live64);
}

void nvfp4_localcta_grouped_gemm_rope_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    at::Tensor& D_K,
    at::Tensor& D_V,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim,
    int silu_dim = 0
) {
    const auto sg_contract = infer_regular_sg_contract(A, A_sg_chunks, B, B_sg_chunks);
    TORCH_CHECK(sg_contract != SGContractMode::ChunkGrid128,
                "generic RoPE epilogue is only implemented for fast outer-scale localCTA grouped GEMM");
    TORCH_CHECK(D.size(1) + D_K.size(1) + D_V.size(1) == B.size(0),
                "Q/K/V output columns must sum to B rows");

    check_output_matrix(D, "D", A.size(0), D.size(1));
    check_output_matrix(D_K, "D_K", A.size(0), D_K.size(1));
    check_output_matrix(D_V, "D_V", A.size(0), D_V.size(1));
    check_rope_qkv_args(D, D_K, D_V, rope_cos, rope_sin,
                        rope_seq_len, rope_head_dim, rope_rotary_dim);

    nvfp4_rope_epilogue::rope_desc rope {
        .cos = rope_cos.data_ptr<float>(),
        .sin = rope_sin.data_ptr<float>(),
        .seq_len = static_cast<int>(rope_seq_len),
        .head_dim = static_cast<int>(rope_head_dim),
        .rotary_dim = static_cast<int>(rope_rotary_dim),
    };

    if (sg_contract == SGContractMode::TileGrid256) {
        check_v3_tilegrid256_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
        launch_fast_grouped_gemm_rope(
            A, A_sc, torch::Tensor(), B, B_sc, torch::Tensor(),
            D, std::optional<at::Tensor>(D_K), std::optional<at::Tensor>(D_V), silu_dim, rope);
        return;
    }

    auto A_sg_outer = normalize_outer_scale_tiles_tensor(A_sg_chunks, A.size(0) / 256, true);
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(B_sg_chunks, B.size(0) / 256, false);
    check_v3_fast_gemm_inputs(A, A_sc, A_sg_outer, B, B_sc, B_sg_outer);
    launch_fast_grouped_gemm_rope(
        A, A_sc, A_sg_outer, B, B_sc, B_sg_outer,
        D, std::optional<at::Tensor>(D_K), std::optional<at::Tensor>(D_V), silu_dim, rope);
}

void nvfp4_localcta_v3_regular_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    nvfp4_localcta_gemm_entrypoint(
        A, A_sc, A_sg_chunks,
        B, B_sc, B_sg_chunks,
        D
    );
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
        A, A_sc_prepared, torch::Tensor(), B, B_sc_prepared, torch::Tensor(), D, D_K_opt, D_V_opt, silu_dim);
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
    auto A_sg_chunkgrid = as_chunk_grid_tensor(A_sg_chunks, A.size(0), A.size(1) * 2);
    auto B_sg_chunkgrid = as_chunk_grid_tensor(B_sg_chunks, B.size(0), B.size(1) * 2);
    check_gemm_inputs(A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid);
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
    launch_fast_grouped_gemm_chunkgrid_sg(
        A, A_sc, A_sg_chunkgrid, B, B_sc, B_sg_chunkgrid, D, D_K_opt, D_V_opt, silu_dim);
}

void nvfp4_localcta_fast_grouped_gemm_virtual_rescale_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_tiles,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_tiles,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
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
    launch_fast_grouped_gemm_virtual_rescale(
        A, A_sc, A_sg_tiles, A_sg_chunks,
        B, B_sc, B_sg_tiles, B_sg_chunks,
        D, D_K_opt, D_V_opt, silu_dim);
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
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= nvfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == static_cast<int>(D_list.size()),
                "D_list length must match number of batches");

    const bool use_chunkgrid =
        n > 0 && infer_regular_sg_contract(A_list[0], A_sg_chunks_list[0], B_list[0], B_sg_chunks_list[0]) == SGContractMode::ChunkGrid128;
    if (use_chunkgrid) {
        std::vector<at::Tensor> A_sg_chunkgrid_list;
        std::vector<at::Tensor> B_sg_chunkgrid_list;
        A_sg_chunkgrid_list.reserve(n);
        B_sg_chunkgrid_list.reserve(n);
        for (int i = 0; i < n; ++i) {
            A_sg_chunkgrid_list.push_back(
                as_chunk_grid_tensor(A_sg_chunks_list[i], A_list[i].size(0), A_list[i].size(1) * 2));
            B_sg_chunkgrid_list.push_back(
                as_chunk_grid_tensor(B_sg_chunks_list[i], B_list[i].size(0), B_list[i].size(1) * 2));
        }
        check_batched_inputs(
            A_list, A_sc_list, A_sg_chunkgrid_list,
            B_list, B_sc_list, B_sg_chunkgrid_list);
        for (int i = 0; i < n; ++i) {
            check_output_matrix(D_list[i], "D_list[i]", A_list[i].size(0), B_list[i].size(0));
        }
        check_batched_shape_compatibility(A_list, B_list, D_list);
        for (int i = 0; i < n; ++i) {
            launch_regular_gemm(
                A_list[i], A_sc_list[i], A_sg_chunkgrid_list[i],
                B_list[i], B_sc_list[i], B_sg_chunkgrid_list[i],
                D_list[i]);
        }
        return;
    }

    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        for (int i = 0; i < n; ++i) {
            check_v3_tilegrid256_gemm_inputs(
                A_list[i], A_sc_list[i], A_sg_chunks_list[i],
                B_list[i], B_sc_list[i], B_sg_chunks_list[i]);
            check_output_matrix(D_list[i], "D_list[i]", A_list[i].size(0), B_list[i].size(0));
        }
        check_batched_shape_compatibility(A_list, B_list, D_list);
        launch_fast_batched_gemm(
            A_list, A_sc_list, std::vector<at::Tensor>(n, torch::Tensor()),
            B_list, B_sc_list, std::vector<at::Tensor>(n, torch::Tensor()),
            D_list);
        return;
    }
    std::vector<at::Tensor> A_sg_outer_list;
    std::vector<at::Tensor> B_sg_outer_list;
    A_sg_outer_list.reserve(n);
    B_sg_outer_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        A_sg_outer_list.push_back(
            normalize_outer_scale_tiles_tensor(A_sg_chunks_list[i], A_list[i].size(0) / 256, true));
        B_sg_outer_list.push_back(
            normalize_outer_scale_tiles_tensor(B_sg_chunks_list[i], B_list[i].size(0) / 256, false));
        check_v3_fast_gemm_inputs(
            A_list[i], A_sc_list[i], A_sg_outer_list[i],
            B_list[i], B_sc_list[i], B_sg_outer_list[i]);
        check_output_matrix(D_list[i], "D_list[i]", A_list[i].size(0), B_list[i].size(0));
    }
    check_batched_shape_compatibility(A_list, B_list, D_list);

    launch_fast_batched_gemm(
        A_list, A_sc_list, A_sg_outer_list,
        B_list, B_sc_list, B_sg_outer_list,
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

    launch_fast_batched_gemm(
        A_list, A_sc_prepared_list, std::vector<at::Tensor>(A_list.size(), torch::Tensor()),
        B_list, B_sc_prepared_list, std::vector<at::Tensor>(B_list.size(), torch::Tensor()),
        D_list);
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
    check_fast_batched_strided_inputs_allow_a_sc_views(
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
        A_full, A_sc_prepared_list, std::vector<at::Tensor>(),
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, std::vector<at::Tensor>(), D_list);
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

void nvfp4_localcta_v3_batched_gemm_strided_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    std::vector<at::Tensor>& D_list
) {
    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        const int n = static_cast<int>(A_sc_list.size());
        TORCH_CHECK(n >= 1, "num_batches must be positive");
        TORCH_CHECK(
            n == static_cast<int>(A_sg_tiles_list.size()) &&
            n == static_cast<int>(A_col_offsets.size()) &&
            n == static_cast<int>(A_col_widths.size()) &&
            n == static_cast<int>(B_list.size()) &&
            n == static_cast<int>(B_sc_list.size()) &&
            n == static_cast<int>(B_sg_tiles_list.size()) &&
            n == static_cast<int>(D_list.size()),
            "All batched strided v3 lists must have matching lengths");
        check_fp4_matrix(A_full, "A_full");
        TORCH_CHECK(A_full.size(0) % 256 == 0, "A_full rows must be multiple of 256");

        for (int i = 0; i < n; ++i) {
            TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets entries must be non-negative");
            TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths entries must be positive");
            TORCH_CHECK(A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
                        "A_full slice exceeds packed width");
            TORCH_CHECK((2 * A_col_widths[i]) % 256 == 0,
                        "Each tilegrid256 strided K width must be a multiple of 256");
            check_v3_tilegrid256_gemm_inputs(
                A_full.narrow(1, A_col_offsets[i], A_col_widths[i]).contiguous(),
                A_sc_list[i], A_sg_tiles_list[i],
                B_list[i], B_sc_list[i], B_sg_tiles_list[i]);
            check_output_matrix(D_list[i], "D_list[i]", A_full.size(0), B_list[i].size(0));
        }
        launch_fast_batched_gemm_strided(
            A_full, A_sc_list, std::vector<at::Tensor>(n, torch::Tensor()),
            A_col_offsets, A_col_widths,
            B_list, B_sc_list, std::vector<at::Tensor>(n, torch::Tensor()),
            D_list);
        return;
    }
    const int n = static_cast<int>(A_sc_list.size());
    TORCH_CHECK(n >= 1, "num_batches must be positive");
    TORCH_CHECK(
        n == static_cast<int>(A_sg_tiles_list.size()) &&
        n == static_cast<int>(A_col_offsets.size()) &&
        n == static_cast<int>(A_col_widths.size()) &&
        n == static_cast<int>(B_list.size()) &&
        n == static_cast<int>(B_sc_list.size()) &&
        n == static_cast<int>(B_sg_tiles_list.size()) &&
        n == static_cast<int>(D_list.size()),
        "All batched strided v3 lists must have matching lengths");
    TORCH_CHECK(A_full.is_cuda() && A_full.is_contiguous(), "A_full must be contiguous CUDA tensor");
    TORCH_CHECK(A_full.dim() == 2, "A_full must be 2D");
    TORCH_CHECK(A_full.scalar_type() == at::kFloat4_e2m1fn_x2, "A_full must be fp4x2");
    TORCH_CHECK(A_full.size(0) % 128 == 0, "A_full rows must be multiple of 128");
    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths entries must be positive");
        TORCH_CHECK((2 * A_col_widths[i]) % 128 == 0, "Each strided K width must be a multiple of 128");
        check_scale_tensor_tma_compatible(A_sc_list[i], "A_sc_list[i]", A_full.size(0), 2 * A_col_widths[i]);
        (void)check_outer_scale_tiles(A_sg_tiles_list[i], "A_sg_tiles_list[i]", A_full.size(0) / 256, true);
        check_fp4_matrix(B_list[i], "B_list[i]");
        TORCH_CHECK(B_list[i].size(1) == A_col_widths[i], "B_list[i] packed K must match A_col_widths[i]");
        check_scale_tensor(B_sc_list[i], "B_sc_list[i]", B_list[i].size(0), 2 * B_list[i].size(1));
        (void)check_outer_scale_tiles(B_sg_tiles_list[i], "B_sg_tiles_list[i]", B_list[i].size(0) / 256, false);
        check_output_matrix(D_list[i], "D_list[i]", A_full.size(0), B_list[i].size(0));
        kittens::py::device_check(
            A_full, A_sc_list[i], A_sg_tiles_list[i],
            B_list[i], B_sc_list[i], B_sg_tiles_list[i], D_list[i]);
    }

    launch_fast_batched_gemm_strided(
        A_full, A_sc_list, A_sg_tiles_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_list, B_sg_tiles_list, D_list);
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
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= 4, "num_batches must be 1..4");
    const bool use_chunkgrid =
        n > 0 && infer_regular_sg_contract(A_list[0], A_sg_chunks_list[0], B_list[0], B_sg_chunks_list[0]) == SGContractMode::ChunkGrid128;
    if (use_chunkgrid) {
        std::vector<at::Tensor> A_sg_chunkgrid_list;
        std::vector<at::Tensor> B_sg_chunkgrid_list;
        A_sg_chunkgrid_list.reserve(n);
        B_sg_chunkgrid_list.reserve(n);
        for (int i = 0; i < n; ++i) {
            A_sg_chunkgrid_list.push_back(
                as_chunk_grid_tensor(A_sg_chunks_list[i], A_list[i].size(0), A_list[i].size(1) * 2));
            B_sg_chunkgrid_list.push_back(
                as_chunk_grid_tensor(B_sg_chunks_list[i], B_list[i].size(0), B_list[i].size(1) * 2));
        }
        check_batched_inputs(
            A_list, A_sc_list, A_sg_chunkgrid_list,
            B_list, B_sc_list, B_sg_chunkgrid_list);
        check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));
        launch_chunkgrid_batched_accum_gemm(
            A_list, A_sc_list, A_sg_chunkgrid_list,
            B_list, B_sc_list, B_sg_chunkgrid_list,
            D_out);
        return;
    }
    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        for (int i = 0; i < n; ++i) {
            check_v3_tilegrid256_gemm_inputs(
                A_list[i], A_sc_list[i], A_sg_chunks_list[i],
                B_list[i], B_sc_list[i], B_sg_chunks_list[i]);
        }
        check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));
        if (n == 1) {
            launch_fast_regular_gemm(
                A_list[0], A_sc_list[0], torch::Tensor(),
                B_list[0], B_sc_list[0], torch::Tensor(), D_out);
            return;
        }
        launch_fast_batched_accum_gemm(
            A_list, A_sc_list, std::vector<at::Tensor>(n, torch::Tensor()),
            B_list, B_sc_list, std::vector<at::Tensor>(n, torch::Tensor()),
            D_out);
        return;
    }
    std::vector<at::Tensor> A_sg_outer_list;
    std::vector<at::Tensor> B_sg_outer_list;
    A_sg_outer_list.reserve(n);
    B_sg_outer_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        A_sg_outer_list.push_back(
            normalize_outer_scale_tiles_tensor(A_sg_chunks_list[i], A_list[i].size(0) / 256, true));
        B_sg_outer_list.push_back(
            normalize_outer_scale_tiles_tensor(B_sg_chunks_list[i], B_list[i].size(0) / 256, false));
        check_v3_fast_gemm_inputs(
            A_list[i], A_sc_list[i], A_sg_outer_list[i],
            B_list[i], B_sc_list[i], B_sg_outer_list[i]);
    }
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));

    if (n == 1) {
        launch_fast_regular_gemm(
            A_list[0], A_sc_list[0], A_sg_outer_list[0],
            B_list[0], B_sc_list[0], B_sg_outer_list[0], D_out);
        return;
    }
    launch_fast_batched_accum_gemm(
        A_list, A_sc_list, A_sg_outer_list,
        B_list, B_sc_list, B_sg_outer_list,
        D_out);
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
        A_list, A_sc_prepared_list, std::vector<at::Tensor>(A_list.size(), torch::Tensor()),
        B_list, B_sc_prepared_list, std::vector<at::Tensor>(B_list.size(), torch::Tensor()),
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
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_fast_batched_inputs(A_list, A_sc_prepared_list, B_list, B_sc_prepared_list);
    TORCH_CHECK(A_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B batches");
    TORCH_CHECK(A_sg_chunks_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A SG batches");
    TORCH_CHECK(B_sg_chunks_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B SG batches");
    std::vector<at::Tensor> A_sg_tiles_list;
    std::vector<at::Tensor> B_sg_tiles_list;
    A_sg_tiles_list.reserve(2);
    B_sg_tiles_list.reserve(2);
    for (int i = 0; i < 2; ++i) {
        A_sg_tiles_list.push_back(
            as_chunk_grid_tensor(A_sg_chunks_list[i], A_list[i].size(0), A_list[i].size(1) * 2));
        B_sg_tiles_list.push_back(
            as_chunk_grid_tensor(B_sg_chunks_list[i], B_list[i].size(0), B_list[i].size(1) * 2));
    }
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));
    launch_fast_split2_dgrad_gemm_onepass_sg(
        A_list, A_sc_prepared_list, A_sg_tiles_list,
        B_list, B_sc_prepared_list, B_sg_tiles_list,
        D_out, static_cast<int>(config_idx));
}

void nvfp4_localcta_v3_split2_dgrad_onepass_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    TORCH_CHECK(A_list.size() == 2, "v3 split2 one-pass dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "v3 split2 one-pass dgrad expects exactly 2 B batches");
    for (int i = 0; i < 2; ++i) {
        check_v3_fast_gemm_inputs(
            A_list[i], A_sc_list[i], A_sg_tiles_list[i],
            B_list[i], B_sc_list[i], B_sg_tiles_list[i]);
    }
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));
    launch_v3_split2_dgrad_gemm_onepass(
        A_list, A_sc_list, A_sg_tiles_list,
        B_list, B_sc_list, B_sg_tiles_list,
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
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects exactly 3 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm_strided(
        A_full, A_sc_prepared_list, std::vector<at::Tensor>(),
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, std::vector<at::Tensor>(), D_out);
}

void nvfp4_localcta_fast_split3_dgrad_strided_gemm_sg_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out
) {
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects exactly 3 B batches");
    TORCH_CHECK(A_sg_chunks_list.size() == 3, "split3 strided dgrad expects exactly 3 A SG batches");
    TORCH_CHECK(B_sg_chunks_list.size() == 3, "split3 strided dgrad expects exactly 3 B SG batches");
    std::vector<at::Tensor> A_sg_tiles_list;
    std::vector<at::Tensor> B_sg_tiles_list;
    A_sg_tiles_list.reserve(3);
    B_sg_tiles_list.reserve(3);
    for (int i = 0; i < 3; ++i) {
        A_sg_tiles_list.push_back(
            normalize_outer_scale_tiles_tensor(A_sg_chunks_list[i], A_full.size(0) / 256, true));
        B_sg_tiles_list.push_back(
            normalize_outer_scale_tiles_tensor(B_sg_chunks_list[i], B_list[i].size(0) / 256, false));
    }
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm_strided(
        A_full, A_sc_prepared_list, A_sg_tiles_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, B_sg_tiles_list, D_out);
}

void nvfp4_localcta_v3_split3_dgrad_strided_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out
) {
    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        TORCH_CHECK(A_sc_list.size() == 3, "split3 strided dgrad expects exactly 3 A batches");
        TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects exactly 3 B batches");
        TORCH_CHECK(
            A_sc_list.size() == A_sg_tiles_list.size() &&
            A_sc_list.size() == A_col_offsets.size() &&
            A_sc_list.size() == A_col_widths.size() &&
            A_sc_list.size() == B_sc_list.size() &&
            A_sc_list.size() == B_sg_tiles_list.size(),
            "tilegrid256 split3 strided dgrad lists must have matching lengths");
        check_fp4_matrix(A_full, "A_full");
        TORCH_CHECK(A_full.size(0) % 256 == 0, "A_full rows must be multiple of 256");
        check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
        for (int i = 0; i < 3; ++i) {
            TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets entries must be non-negative");
            TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths entries must be positive");
            TORCH_CHECK(A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
                        "A_full slice exceeds packed width");
            TORCH_CHECK((2 * A_col_widths[i]) % 256 == 0,
                        "Each tilegrid256 strided K width must be a multiple of 256");
            check_v3_tilegrid256_gemm_inputs(
                A_full.narrow(1, A_col_offsets[i], A_col_widths[i]).contiguous(),
                A_sc_list[i], A_sg_tiles_list[i],
                B_list[i], B_sc_list[i], B_sg_tiles_list[i]);
        }
        launch_fast_split3_dgrad_gemm_strided(
            A_full, A_sc_list, std::vector<at::Tensor>(3, torch::Tensor()),
            A_col_offsets, A_col_widths,
            B_list, B_sc_list, std::vector<at::Tensor>(3, torch::Tensor()), D_out);
        return;
    }
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_list);
    TORCH_CHECK(A_sc_list.size() == 3, "split3 strided dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects exactly 3 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm_strided(
        A_full, A_sc_list, A_sg_tiles_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_list, B_sg_tiles_list, D_out);
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
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 3, "split3 strided dgrad expects exactly 3 A batches");
    TORCH_CHECK(B_list.size() == 3, "split3 strided dgrad expects exactly 3 B batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split3_dgrad_gemm_strided_sum(
        A_full, A_sc_prepared_list, std::vector<at::Tensor>(),
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, std::vector<at::Tensor>(), D_out);
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
    check_fast_batched_strided_inputs_allow_a_sc_views(
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

void nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const at::Tensor& B_full,
    const at::Tensor& B_sc_prepared_full,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_output_matrix(D_out, "D_out", A_full.size(0), B_full.size(0));
    launch_fast_split3_dgrad_gemm_strided_onepass_full_b(
        A_full, A_sc_prepared_list, A_col_offsets, A_col_widths,
        B_full, B_sc_prepared_full, D_out, static_cast<int>(config_idx));
}

void nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_sg_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const at::Tensor& B_full,
    const at::Tensor& B_sc_prepared_full,
    const at::Tensor& B_sg_tiles_full,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_output_matrix(D_out, "D_out", A_full.size(0), B_full.size(0));
    launch_fast_split3_dgrad_gemm_strided_onepass_full_b_sg(
        A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
        B_full, B_sc_prepared_full, B_sg_tiles_full, D_out, static_cast<int>(config_idx));
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
        A_full, A_sc_prepared_list, std::vector<at::Tensor>(),
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, std::vector<at::Tensor>(), D_out);
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

void nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_tiles_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_tiles_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B batches");
    TORCH_CHECK(A_sg_tiles_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A SG batches");
    TORCH_CHECK(B_sg_tiles_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B SG batches");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split2_dgrad_gemm_strided_onepass_outer_sg(
        A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, B_sg_tiles_list,
        D_out, static_cast<int>(config_idx));
}

void nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_sg_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_prepared_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_prepared_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_fast_batched_strided_inputs_allow_a_sc_views(
        A_full, A_sc_prepared_list,
        A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list);
    TORCH_CHECK(A_sc_prepared_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A batches");
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B batches");
    TORCH_CHECK(A_sg_chunks_list.size() == 2, "split2 one-pass dgrad expects exactly 2 A SG batches");
    TORCH_CHECK(B_sg_chunks_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B SG batches");
    std::vector<at::Tensor> A_sg_tiles_list;
    std::vector<at::Tensor> B_sg_tiles_list;
    A_sg_tiles_list.reserve(2);
    B_sg_tiles_list.reserve(2);
    for (int i = 0; i < 2; ++i) {
        A_sg_tiles_list.push_back(
            as_chunk_grid_tensor(
                A_sg_chunks_list[i],
                A_full.size(0),
                A_col_widths[i] * 2));
        B_sg_tiles_list.push_back(
            as_chunk_grid_tensor(
                B_sg_chunks_list[i],
                B_list[i].size(0),
                B_list[i].size(1) * 2));
    }
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_fast_split2_dgrad_gemm_strided_onepass_sg(
        A_full, A_sc_prepared_list, A_sg_tiles_list, A_col_offsets, A_col_widths,
        B_list, B_sc_prepared_list, B_sg_tiles_list,
        D_out, static_cast<int>(config_idx));
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
        A_fp4_cat, A_sc_list, std::vector<at::Tensor>(),
        A_col_offsets, A_col_widths,
        B_fp4_list, B_sc_list, std::vector<at::Tensor>(), D_list);

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

at::Tensor nvfp4_localcta_prepare_outer_sg_entrypoint(
    const at::Tensor& sg,
    int64_t tiles,
    bool row_axis = true
) {
    return normalize_outer_scale_tiles_tensor(sg, tiles, row_axis);
}

at::Tensor nvfp4_localcta_prepare_split_wgrad_a_sg_entrypoint(
    const at::Tensor& sg,
    int64_t tiles,
    float scale
) {
    return prepare_split_wgrad_a_sg_tensor(sg, tiles, scale);
}

at::Tensor nvfp4_localcta_prepare_split2_b_sg_entrypoint(
    const at::Tensor& sg,
    int64_t tiles,
    float scale
) {
    return prepare_split2_b_sg_tensor(sg, tiles, scale);
}

at::Tensor nvfp4_localcta_prepare_w2_dgrad_b_sg_entrypoint(
    const at::Tensor& sg,
    int64_t tiles,
    float scale
) {
    return prepare_w2_dgrad_b_sg_tensor(sg, tiles, scale);
}

at::Tensor nvfp4_localcta_fold_sg_into_prepared_sc_entrypoint(
    const at::Tensor& sc_raw,
    const at::Tensor& sg,
    int64_t rows,
    int64_t cols
) {
    return fold_sg_into_prepared_sc_tensor(sc_raw, sg, rows, cols);
}

at::Tensor nvfp4_localcta_fold_outer_sg_into_prepared_sc_entrypoint(
    const at::Tensor& sc_raw,
    const at::Tensor& sg,
    int64_t rows,
    int64_t cols
) {
    return fold_outer_sg_into_prepared_sc_tensor(sc_raw, sg, rows, cols);
}

template <typename C>
void launch_localcta_silu_dgrad_quant_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    const at::Tensor& h3,
    const at::Tensor& h1_raw,
    at::Tensor& row_fp4_cat,
    at::Tensor& row_sc_prepared_cat,
    at::Tensor& row_sg_cat,
    at::Tensor& col_fp4_cat,
    at::Tensor& col_sc_prepared_cat,
    at::Tensor& col_sg_cat
) {
    using G = nvfp4_localcta_silu_dgrad_quant_gemm::globals<C>;
    const int64_t M = A.size(0);
    const int64_t H = B.size(0);
    auto A_sg_outer = normalize_outer_scale_tiles_tensor(A_sg, M / C::Mb, true).contiguous();
    auto B_sg_outer = normalize_outer_scale_tiles_tensor(B_sg, H / C::Nb, false).contiguous();

    outer_scale_desc a_sg_desc = check_outer_scale_tiles(A_sg_outer, "A_sg_outer", M / C::Mb, true);
    outer_scale_desc b_sg_desc = check_outer_scale_tiles(B_sg_outer, "B_sg_outer", H / C::Nb, false);

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .A_sg = a_sg_desc.ptr,
        .A_sg_stride = a_sg_desc.stride,
        .B_sg = b_sg_desc.ptr,
        .B_sg_stride = b_sg_desc.stride,
        .h3 = reinterpret_cast<const bf16*>(h3.data_ptr<at::BFloat16>()),
        .h1_raw = reinterpret_cast<const bf16*>(h1_raw.data_ptr<at::BFloat16>()),
        .row_fp4 = reinterpret_cast<uint8_t*>(row_fp4_cat.data_ptr()),
        .row_sc = reinterpret_cast<uint8_t*>(row_sc_prepared_cat.data_ptr()),
        .row_sg = row_sg_cat.data_ptr<float>(),
        .col_fp4 = reinterpret_cast<uint8_t*>(col_fp4_cat.data_ptr()),
        .col_sc = reinterpret_cast<uint8_t*>(col_sc_prepared_cat.data_ptr()),
        .col_sg = col_sg_cat.data_ptr<float>(),
        .M = static_cast<int>(M),
        .H = static_cast<int>(H),
    };
    kittens::py::launch_kernel<C, G, nvfp4_localcta_silu_dgrad_quant_gemm::kernel<C>>(g);
}

void nvfp4_localcta_w2_dgrad_silu_quant_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg,
    const at::Tensor& h3,
    const at::Tensor& h1_raw,
    at::Tensor& row_fp4_cat,
    at::Tensor& row_sc_prepared_cat,
    at::Tensor& row_sg_cat,
    at::Tensor& col_fp4_cat,
    at::Tensor& col_sc_prepared_cat,
    at::Tensor& col_sg_cat,
    int config_id = 4
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share packed K");
    const int64_t M = A.size(0);
    const int64_t H = B.size(0);
    const int64_t K = A.size(1) * 2;
    TORCH_CHECK(M % 256 == 0 && H % 256 == 0 && K % 256 == 0,
                "localCTA W2-dgrad SiLU producer requires M,H,K divisible by 256");
    check_scale_tensor(A_sc, "A_sc", M, K);
    check_scale_tensor(B_sc, "B_sc", H, K);
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous() && h3.scalar_type() == at::kBFloat16,
                "h3 must be contiguous CUDA bf16");
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous() && h1_raw.scalar_type() == at::kBFloat16,
                "h1_raw must be contiguous CUDA bf16");
    TORCH_CHECK(h3.sizes() == torch::IntArrayRef({M, H}) &&
                h1_raw.sizes() == torch::IntArrayRef({M, H}),
                "h3 and h1_raw must have shape [M,H]");

    const int64_t total_n = 2 * H;
    TORCH_CHECK(row_fp4_cat.is_cuda() && row_fp4_cat.is_contiguous() &&
                row_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2 &&
                row_fp4_cat.sizes() == torch::IntArrayRef({M, total_n / 2}),
                "row_fp4_cat must be contiguous CUDA fp4 [M,total_n/2]");
    TORCH_CHECK(row_sc_prepared_cat.is_cuda() && row_sc_prepared_cat.is_contiguous() &&
                row_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn &&
                row_sc_prepared_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 64, 512}),
                "row_sc_prepared_cat must be contiguous CUDA fp8 [M/128,total_n/64,512]");
    TORCH_CHECK(row_sg_cat.is_cuda() && row_sg_cat.is_contiguous() &&
                row_sg_cat.scalar_type() == torch::kFloat32 &&
                row_sg_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 128}),
                "row_sg_cat must be contiguous CUDA fp32 [M/128,total_n/128]");
    TORCH_CHECK(col_fp4_cat.is_cuda() && col_fp4_cat.is_contiguous() &&
                col_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2 &&
                col_fp4_cat.sizes() == torch::IntArrayRef({total_n, M / 2}),
                "col_fp4_cat must be contiguous CUDA fp4 [total_n,M/2]");
    TORCH_CHECK(col_sc_prepared_cat.is_cuda() && col_sc_prepared_cat.is_contiguous() &&
                col_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn &&
                col_sc_prepared_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 64, 512}),
                "col_sc_prepared_cat must be contiguous CUDA fp8 [total_n/128,M/64,512]");
    TORCH_CHECK(col_sg_cat.is_cuda() && col_sg_cat.is_contiguous() &&
                col_sg_cat.scalar_type() == torch::kFloat32 &&
                col_sg_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 128}),
                "col_sg_cat must be contiguous CUDA fp32 [total_n/128,M/128]");
    kittens::py::device_check(A, A_sc, A_sg, B, B_sc, B_sg, h3, h1_raw,
                              row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                              col_fp4_cat, col_sc_prepared_cat, col_sg_cat);

    if (config_id == 0) {
        launch_localcta_silu_dgrad_quant_gemm_with_config<
            nvfp4_localcta_silu_dgrad_quant_gemm::config<3, 4>>(
            A, A_sc, A_sg, B, B_sc, B_sg, h3, h1_raw,
            row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
            col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
    } else if (config_id == 1) {
        launch_localcta_silu_dgrad_quant_gemm_with_config<
            nvfp4_localcta_silu_dgrad_quant_gemm::config<2, 12, false>>(
            A, A_sc, A_sg, B, B_sc, B_sg, h3, h1_raw,
            row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
            col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
    } else if (config_id == 2) {
        launch_localcta_silu_dgrad_quant_gemm_with_config<
            nvfp4_localcta_silu_dgrad_quant_gemm::config<3, 4, false>>(
            A, A_sc, A_sg, B, B_sc, B_sg, h3, h1_raw,
            row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
            col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
    } else if (config_id == 3) {
        launch_localcta_silu_dgrad_quant_gemm_with_config<
            nvfp4_localcta_silu_dgrad_quant_gemm::config<3, 12, false>>(
            A, A_sc, A_sg, B, B_sc, B_sg, h3, h1_raw,
            row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
            col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
    } else {
        launch_localcta_silu_dgrad_quant_gemm_with_config<
            nvfp4_localcta_silu_dgrad_quant_gemm::config<3, 12>>(
            A, A_sc, A_sg, B, B_sc, B_sg, h3, h1_raw,
            row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
            col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_localcta_gemm", &nvfp4_localcta_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_gemm_residual", &nvfp4_localcta_gemm_residual_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("R"), pybind11::arg("D"));
    m.def("nvfp4_localcta_gemm_residual_rms",
          &nvfp4_localcta_gemm_residual_rms_entrypoint,
          "localCTA v4 residual GEMM with exact-row RMS partials",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("R"), pybind11::arg("D"),
          pybind11::arg("row_rms_partial"));
    m.def("nvfp4_localcta_h_residual_carrier",
          &nvfp4_localcta_h_residual_carrier_entrypoint,
          "localCTA residual GEMM plus native 128x128 tile-RMS carrier",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg"),
          pybind11::arg("R"), pybind11::arg("gamma"), pybind11::arg("D"),
          pybind11::arg("inverse_rms_tiles"), pybind11::arg("tile_amax"),
          pybind11::arg("row_outer_scales"),
          pybind11::arg("col_outer_scales"),
          pybind11::arg("eps") = 1.0e-5);
    m.def("nvfp4_localcta_fast_gemm", &nvfp4_localcta_fast_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc_prepared"),
          pybind11::arg("B"), pybind11::arg("B_sc_prepared"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_fast_gemm_outer_sg_config",
          &nvfp4_localcta_fast_gemm_outer_sg_config_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("A_sg"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("B_sg"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("nvfp4_localcta_fast_gemm_residual", &nvfp4_localcta_fast_gemm_residual_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc_prepared"),
          pybind11::arg("B"), pybind11::arg("B_sc_prepared"),
          pybind11::arg("R"), pybind11::arg("D"));
    m.def("nvfp4_localcta_fast_gemm_sg", &nvfp4_localcta_fast_gemm_sg_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc_prepared"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc_prepared"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_fast_gemm_virtual_rescale",
          &nvfp4_localcta_fast_gemm_virtual_rescale_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_tiles"),
          pybind11::arg("A_sg_chunks"), pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("B_sg_tiles"), pybind11::arg("B_sg_chunks"), pybind11::arg("D"));
    m.def("nvfp4_localcta_grouped_gemm", &nvfp4_localcta_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_grouped_gemm_rope_live64", &nvfp4_localcta_grouped_gemm_rope_live64_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"), pybind11::arg("D_K"), pybind11::arg("D_V"),
          pybind11::arg("rope_cs"), pybind11::arg("rope_seq_len"),
          pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_grouped_gemm_rope", &nvfp4_localcta_grouped_gemm_rope_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"), pybind11::arg("D_K"), pybind11::arg("D_V"),
          pybind11::arg("rope_cos"), pybind11::arg("rope_sin"),
          pybind11::arg("rope_seq_len"), pybind11::arg("rope_head_dim"),
          pybind11::arg("rope_rotary_dim"), pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_v3_regular_gemm", &nvfp4_localcta_v3_regular_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_fast_grouped_gemm", &nvfp4_localcta_fast_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc_prepared"),
          pybind11::arg("B"), pybind11::arg("B_sc_prepared"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_fast_grouped_gemm_sg", &nvfp4_localcta_fast_grouped_gemm_sg_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc_prepared"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc_prepared"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_fast_grouped_gemm_virtual_rescale",
          &nvfp4_localcta_fast_grouped_gemm_virtual_rescale_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_tiles"),
          pybind11::arg("A_sg_chunks"), pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("B_sg_tiles"), pybind11::arg("B_sg_chunks"),
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
    m.def("nvfp4_localcta_v3_batched_gemm_strided",
          &nvfp4_localcta_v3_batched_gemm_strided_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_tiles_list"), pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_tiles_list"),
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
          pybind11::arg("A_list"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_v3_split2_dgrad_onepass_gemm",
          &nvfp4_localcta_v3_split2_dgrad_onepass_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_tiles_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("B_sg_tiles_list"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_fast_split3_dgrad_strided_gemm",
          &nvfp4_localcta_fast_split3_dgrad_strided_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("D_out"));
    m.def("nvfp4_localcta_fast_split3_dgrad_strided_gemm_sg",
          &nvfp4_localcta_fast_split3_dgrad_strided_gemm_sg_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_out"));
    m.def("nvfp4_localcta_v3_split3_dgrad_strided_gemm",
          &nvfp4_localcta_v3_split3_dgrad_strided_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_tiles_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_tiles_list"),
          pybind11::arg("D_out"));
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
    m.def("nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm",
          &nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_full"), pybind11::arg("B_sc_prepared_full"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_sg",
          &nvfp4_localcta_fast_split3_dgrad_strided_onepass_full_b_gemm_sg_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_sg_tiles_list"), pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"), pybind11::arg("B_full"),
          pybind11::arg("B_sc_prepared_full"), pybind11::arg("B_sg_tiles_full"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
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
    m.def("nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg",
          &nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_outer_sg_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_sg_tiles_list"), pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("B_sg_tiles_list"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
    m.def("nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_sg",
          &nvfp4_localcta_fast_split2_dgrad_strided_onepass_gemm_sg_entrypoint,
          pybind11::arg("A_full"), pybind11::arg("A_sc_prepared_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_prepared_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_out"), pybind11::arg("config_idx") = -1);
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
    m.def("nvfp4_localcta_prepare_outer_sg", &nvfp4_localcta_prepare_outer_sg_entrypoint,
          pybind11::arg("sg"), pybind11::arg("tiles"), pybind11::arg("row_axis") = true);
    m.def("nvfp4_localcta_prepare_split_wgrad_a_sg", &nvfp4_localcta_prepare_split_wgrad_a_sg_entrypoint,
          pybind11::arg("sg"), pybind11::arg("tiles"), pybind11::arg("scale"));
    m.def("nvfp4_localcta_prepare_split2_b_sg", &nvfp4_localcta_prepare_split2_b_sg_entrypoint,
          pybind11::arg("sg"), pybind11::arg("tiles"), pybind11::arg("scale"));
    m.def("nvfp4_localcta_prepare_w2_dgrad_b_sg", &nvfp4_localcta_prepare_w2_dgrad_b_sg_entrypoint,
          pybind11::arg("sg"), pybind11::arg("tiles"), pybind11::arg("scale"));
    m.def("nvfp4_localcta_fold_sg_into_prepared_sc", &nvfp4_localcta_fold_sg_into_prepared_sc_entrypoint,
          pybind11::arg("sc_raw"), pybind11::arg("sg"), pybind11::arg("rows"), pybind11::arg("cols"));
    m.def("nvfp4_localcta_fold_outer_sg_into_prepared_sc", &nvfp4_localcta_fold_outer_sg_into_prepared_sc_entrypoint,
          pybind11::arg("sc_raw"), pybind11::arg("sg"), pybind11::arg("rows"), pybind11::arg("cols"));
    m.def("nvfp4_localcta_w2_dgrad_silu_quant_gemm",
          &nvfp4_localcta_w2_dgrad_silu_quant_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg"),
          pybind11::arg("h3"), pybind11::arg("h1_raw"),
          pybind11::arg("row_fp4_cat"), pybind11::arg("row_sc_prepared_cat"),
          pybind11::arg("row_sg_cat"), pybind11::arg("col_fp4_cat"),
          pybind11::arg("col_sc_prepared_cat"), pybind11::arg("col_sg_cat"),
          pybind11::arg("config_id") = 4);
}
