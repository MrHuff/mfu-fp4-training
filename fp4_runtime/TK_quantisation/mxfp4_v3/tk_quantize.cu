// MXFP4 v3 Quantize — PyBind11 dispatch
//
// Exports:
//   mxfp4_quantize_for_gemm(input) → (fp4, scales)
//   mxfp4_group_quantize_dim0(input, Ms) → list[(fp4, scales)]

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

#include "mxfp4_v3_quantize.cuh"

using namespace mxfp4_v3;

// ═══════════════════════════════════════════════════════════════════
// TMA helper
// ═══════════════════════════════════════════════════════════════════
static void create_tma_2d(
    CUtensorMap& tma,
    void* ptr,
    uint64_t dimY, uint64_t dimX,
    uint32_t boxY, uint32_t boxX,
    uint64_t strideX, size_t elemBits,
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE
) {
    typedef CUresult (*cuTensorMapEncodeTiled_t)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
        const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    static cuTensorMapEncodeTiled_t fn = nullptr;
    if (!fn) {
        void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
        TORCH_CHECK(handle, "Failed to open libcuda.so.1");
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(handle, "cuTensorMapEncodeTiled"));
        TORCH_CHECK(fn, "cuTensorMapEncodeTiled not found");
    }

    CUtensorMapDataType dtype;
    if (elemBits == 16) dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (elemBits == 8) dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if (elemBits == 4) dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported elem bits: ", elemBits);

    uint64_t size[2]   = {dimX, dimY};
    uint64_t stride[1] = {strideX * elemBits / 8};
    uint32_t box[2]    = {boxX, boxY};
    uint32_t elStride[2] = {1, 1};
    auto result = fn(&tma, dtype, 2, ptr,
        size, stride, box, elStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        l2promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}

// ═══════════════════════════════════════════════════════════════════
// Cached SM count + persistent threshold
// ═══════════════════════════════════════════════════════════════════
struct CachedInfo {
    int num_sms = 0;
    int max_bps = 0;   // max blocks per SM (fused kernel)
    int p_max_bps = 0; // max blocks per SM (persistent kernel)
    int dshmem = 0;
    bool initialized = false;
};

static CachedInfo& get_cached() {
    static CachedInfo ci;
    if (!ci.initialized) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ci.num_sms, cudaDevAttrMultiProcessorCount, dev);
        ci.dshmem = v3_shmem_size();

        // Fused kernel occupancy (default RTE template)
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ci.max_bps, mxfp4_v3_fused_kernel<QuantMode::RTE>, THREADS, ci.dshmem);

        // Set attributes for other mode instantiations too
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);

        // Persistent kernel occupancy
        cudaFuncSetAttribute(mxfp4_v3_persistent_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ci.p_max_bps, mxfp4_v3_persistent_kernel<QuantMode::RTE>, THREADS, ci.dshmem);

        ci.initialized = true;
    }
    return ci;
}

// Reserve the persistent path for very large tensors only. Smaller surfaces
// remain on the fused path that is already well-characterized.
static constexpr int PERSISTENT_THRESHOLD = 4096;

// ═══════════════════════════════════════════════════════════════════
// Global work counter for persistent kernel
// ═══════════════════════════════════════════════════════════════════
static unsigned int* g_work_counter = nullptr;
static void ensure_work_counter() {
    if (!g_work_counter) {
        cudaMalloc(&g_work_counter, sizeof(unsigned int));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Single tensor quantize
// ═══════════════════════════════════════════════════════════════════
// Templated quantize function
template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm_impl(torch::Tensor input, cudaStream_t stream) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();

    auto fp4_out = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    // TMA maps: use TILE_DIM (64) as box size for sub-tile pipelining
    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 16,
                  total_tiles >= PERSISTENT_THRESHOLD ?
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B :
                      CU_TENSOR_MAP_L2_PROMOTION_NONE);
    create_tma_2d(tma_out, fp4_out.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 4);

    if (total_tiles >= PERSISTENT_THRESHOLD) {
        // Persistent path
        ensure_work_counter();
        cudaMemsetAsync(g_work_counter, 0, sizeof(unsigned int), stream);

        int num_persistent = ci.p_max_bps * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        PersistentArgs args;
        args.work_counter = g_work_counter;
        args.tiles_X = tiles_X;
        args.tiles_Y = tiles_Y;
        args.total_tiles = total_tiles;

        mxfp4_v3_persistent_kernel<MODE><<<num_persistent, THREADS, dshmem, stream>>>(
            tma_in, tma_out,
            reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            M, K, args);
    } else {
        // Fused path — one CTA per chunk
        dim3 grid(tiles_X, tiles_Y);
        mxfp4_v3_fused_kernel<MODE><<<grid, THREADS, dshmem, stream>>>(
            tma_in, tma_out,
            reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            M, K);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 quantize: ", cudaGetErrorString(err));

    return std::make_tuple(fp4_out, sc_out);
}

// Python-facing dispatch by mode int
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm(torch::Tensor input, int mode) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    switch (mode) {
        case 1: return mxfp4_quantize_for_gemm_impl<QuantMode::ENCODE>(input, stream);
        case 2: return mxfp4_quantize_for_gemm_impl<QuantMode::DECODE>(input, stream);
        default: return mxfp4_quantize_for_gemm_impl<QuantMode::RTE>(input, stream);
    }
}


// ═══════════════════════════════════════════════════════════════════
// Group quantize dim0 — single kernel launch for all groups
//
// Uses mxfp4_v3_fused_group_kernel: one kernel launch over the
// entire contiguous input, with per-group scale pointers in GroupArgs.
// FP4 output is contiguous, scales are per-group.
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_group_quantize_dim0(torch::Tensor input, std::vector<int64_t> group_sizes) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % 128 == 0);

    int ng = (int)group_sizes.size();
    TORCH_CHECK(ng >= 1 && ng <= MAX_GROUPS);

    int64_t total = 0;
    for (auto s : group_sizes) {
        TORCH_CHECK(s % 128 == 0);
        total += s;
    }
    TORCH_CHECK(total == M);

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Allocate contiguous FP4 output
    auto fp4_all = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));

    // Allocate per-group scale tensors
    std::vector<torch::Tensor> sc_allocs(ng);
    GroupArgs args;
    memset(&args, 0, sizeof(args));
    args.num_groups = ng;
    args.boundaries[0] = 0;

    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        args.boundaries[i + 1] = args.boundaries[i] + (int)Mi;
        sc_allocs[i] = torch::empty({Mi / 128, K / 128, 32, 16},
            torch::dtype(torch::kUInt8).device(device));
        args.scale_ptrs[i] = reinterpret_cast<uint8_t*>(sc_allocs[i].data_ptr());
    }

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    // Set up grouped kernel occupancy (first time only)
    static bool grp_init = false;
    if (!grp_init) {
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        grp_init = true;
    }

    // TMA maps
    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_all.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 4);

    // Single kernel launch for all groups
    dim3 grid(tiles_X, tiles_Y);
    mxfp4_v3_fused_group_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
        tma_in, tma_out, M, K, args);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 group quantize: ", cudaGetErrorString(err));

    // Split FP4 output and pair with per-group scales
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    int64_t row_off = 0;
    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        auto fp4_slice = fp4_all.narrow(0, row_off, Mi).contiguous();
        results.push_back(std::make_tuple(fp4_slice, sc_allocs[i]));
        row_off += Mi;
    }
    return results;
}

// ═══════════════════════════════════════════════════════════════════
// Multi quantize — multiple non-contiguous tensors
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_multi_quantize(std::vector<torch::Tensor> inputs) {
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    for (auto& t : inputs) {
        results.push_back(mxfp4_quantize_for_gemm(t, 0));
    }
    return results;
}


// ═══════════════════════════════════════════════════════════════════
// Fast BF16 matrix transpose kernel (tiled, smem-based)
// ═══════════════════════════════════════════════════════════════════
static constexpr int TR_TILE = 32;
__global__ void bf16_transpose_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int M, int K
) {
    __shared__ __nv_bfloat16 tile[TR_TILE][TR_TILE + 1];  // +1 to avoid bank conflicts

    int bx = blockIdx.x * TR_TILE;
    int by = blockIdx.y * TR_TILE;

    // Load tile from src (row-major: M×K)
    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    for (int j = 0; j < TR_TILE; j += blockDim.y) {
        if ((y + j) < M && x < K) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * K + x];
        }
    }
    __syncthreads();

    // Write transposed tile to dst (row-major: K×M)
    x = by + threadIdx.x;  // swapped
    y = bx + threadIdx.y;
    for (int j = 0; j < TR_TILE; j += blockDim.y) {
        if ((y + j) < K && x < M) {
            dst[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Row + Col quantize: quantize input AND its transpose
//
// Returns: (row_fp4, row_sc, col_fp4, col_sc)
// where col is the MXFP4 quantization of input^T
// ═══════════════════════════════════════════════════════════════════
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col(torch::Tensor input, int mode) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v3_rowcol_shmem_size();
    static bool rowcol_init = false;
    if (!rowcol_init) {
        cudaFuncSetAttribute(mxfp4_v3_rowcol_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_rowcol_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_rowcol_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        rowcol_init = true;
    }

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v3_rowcol_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
        case 2:
            mxfp4_v3_rowcol_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
        default:
            mxfp4_v3_rowcol_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v3 row/col quantize: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11
// ═══════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_quantize_for_gemm", &mxfp4_quantize_for_gemm,
          "MXFP4 v3 quantize (pipelined) → (fp4, scales). mode: 0=RTE, 1=ENCODE, 2=DECODE",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_group_quantize_dim0", &mxfp4_group_quantize_dim0,
          "MXFP4 v3 group quantize contiguous input → list[(fp4, scales)]");
    m.def("mxfp4_multi_quantize", &mxfp4_multi_quantize,
          "MXFP4 v3 multi-tensor quantize → list[(fp4, scales)]");
    m.def("mxfp4_quantize_row_and_col", &mxfp4_quantize_row_and_col,
          "MXFP4 v3 quantize row + transpose + col quantize → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("input"), py::arg("mode") = 0);
}
