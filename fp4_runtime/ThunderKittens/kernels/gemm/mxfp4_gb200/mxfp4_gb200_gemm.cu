// ================================================================
// MXFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "mxfp4_gemm.cuh"
// mxfp4_quantize.cuh removed — use standalone mxfp4_v2 quantizer
#include "mxfp4_batched_gemm.cuh"
#include "mxfp4_atb_gemm.cuh"
#include "mxfp4_split2_accum_gemm.cuh"
#include "mxfp4_split3_accum_gemm.cuh"
#include "mxfp4_silu_dgrad_quant_gemm.cuh"
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <optional>

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_entrypoint(const __grid_constant__ mxfp4_gemm::globals<C> g) {
    mxfp4_gemm::kernel<C>(g);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = mxfp4_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << " NUM_D_TILES=" << C::NUM_D_TILES
              << " OVERLAP_EPI=" << C::OVERLAP_EPI << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = size_t(M) * K / 2 + size_t(N) * K / 2 + size_t(M) * N * 2;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp4x2_e2m1*> d_A(arg_group_count);
    std::vector<__nv_fp4x2_e2m1*> d_B(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_B_sc(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_D[i], M*N*sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M*N*sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i*100, 0.0f, 255.0f);
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i*100 + 1, 0.0f, 255.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_A_sc[i], M*K/32, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_B_sc[i], N*K/32, seed + i*100 + 3, 0.1f, 10.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device (MXFP4 with E8M0 scales, block size 32)
    reference_blockscaled_gemm<__nv_fp4x2_e2m1, __nv_fp8_e8m0, __nv_bfloat16, 32>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl Ag{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl Asg{d_A_sc[i], M/128, K/128, nullptr, nullptr};
        typename G::B_fp4x2_gl Bg{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl Bsg{d_B_sc[i], N/128, K/128, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Bg, Bsg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));

    // Prepare kernel launch configuration
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, C::CLUSTER_SIZE);

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check correctness
    check_correctness(d_D[0], d_D_ref, M * N);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_A_sc[i]);
        cudaFree(d_B_sc[i]);
        cudaFree(d_D[i]);
    }
    cudaFree(d_D_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main() {
    int N;
    bool ncu = false;

    // Template parameters: Nb, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH, SUPERGROUP_SIZE, NUM_D_TILES, OVERLAP_EPI
    N = 1024;
    run_benchmark<mxfp4_gemm::config<128, 5, 4, 12, 2, true>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<mxfp4_gemm::config<256, 5, 8, 12, 2, true>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<mxfp4_gemm::config<256, 5, 8, 8, 2, false>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<mxfp4_gemm::config<256, 4, 16, 16, 4, false>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<mxfp4_gemm::config<256, 4, 8, 8, 2, false>>(N, N, N, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"

namespace {

using mxfp4_onepass_cfg1 = mxfp4_split2_accum_gemm::config<128, 5, 4, 12, 2, true, 2, false>;
using mxfp4_onepass_cfg3 = mxfp4_split2_accum_gemm::config<256, 5, 8, 4, 2, false, 2, false>;
using mxfp4_onepass_cfg5 = mxfp4_split2_accum_gemm::config<256, 5, 8, 12, 2, false, 2, false>;
using mxfp4_onepass_cfg5_h = mxfp4_split2_accum_gemm::config<
    256, 5, 8, 12, 2, false, 2, false, true>;
using mxfp4_split3_onepass_cfg1 = mxfp4_split3_accum_gemm::config<128, 5, 4, 12, 2, true, 2, false>;
using mxfp4_split3_onepass_cfg3 = mxfp4_split3_accum_gemm::config<256, 5, 8, 4, 2, false, 2, false>;
using mxfp4_split3_onepass_cfg5 = mxfp4_split3_accum_gemm::config<256, 5, 8, 12, 2, false, 2, false>;

void check_fp4_matrix(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat4_e2m1fn_x2, name, " must be fp4x2");
}

void check_output_matrix(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.size(0) == rows && t.size(1) == cols, name, " shape mismatch");
}

template <typename GL>
GL tensor_to_gl_tma_view(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 2 || t.dim() == 4, name, " must be 2D or 4D");

    if constexpr (std::is_same_v<typename GL::dtype, kittens::fp4e2m1_2>) {
        TORCH_CHECK(t.scalar_type() == at::kFloat4_e2m1fn_x2, name, " must be fp4x2");
    } else if constexpr (std::is_same_v<typename GL::dtype, kittens::fp8e8m0>) {
        TORCH_CHECK(t.scalar_type() == at::kFloat8_e8m0fnu || t.scalar_type() == at::kByte,
                    name, " must be fp8e8m0/uint8");
    } else if constexpr (std::is_same_v<typename GL::dtype, kittens::bf16>) {
        TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    }

    int b = 1;
    int d = 1;
    int r = 1;
    int c = 1;

    if (t.dim() == 2) {
        TORCH_CHECK(t.stride(1) == 1, name, " 2D TMA view must have unit inner stride");
        TORCH_CHECK(t.stride(0) >= t.size(1), name, " 2D TMA leading stride is smaller than logical width");
        r = static_cast<int>(t.size(0));
        c = static_cast<int>(t.stride(0));
    } else {
        TORCH_CHECK(t.stride(3) == 1, name, " 4D TMA view must have unit innermost stride");
        TORCH_CHECK(t.stride(2) == t.size(3), name, " 4D TMA inner tile stride mismatch");
        TORCH_CHECK(t.stride(1) == t.size(2) * t.size(3), name, " 4D TMA depth stride mismatch");
        TORCH_CHECK(t.stride(0) % t.stride(1) == 0, name, " 4D TMA batch stride mismatch");
        b = static_cast<int>(t.size(0));
        d = static_cast<int>(t.stride(0) / t.stride(1));
        r = static_cast<int>(t.size(2));
        c = static_cast<int>(t.size(3));
        TORCH_CHECK(d >= t.size(1), name, " 4D TMA leading depth is smaller than logical depth");
    }

    return kittens::make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()), b, d, r, c);
}

void check_tilemask(
    const at::Tensor& t,
    const char* name,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kByte, name, " must be uint8");
    TORCH_CHECK(t.size(0) == rows, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols, name, " second dim mismatch");
}

bool rope_tensor_disabled(const at::Tensor& t) {
    return t.numel() == 0;
}

bool is_power_of_two(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void check_rope_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t seq_len,
    int64_t rotary_dim
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == seq_len, name, " seq_len mismatch");
    TORCH_CHECK(t.size(1) == rotary_dim / 2, name, " rotary_dim/2 mismatch");
}

void check_rope_epilogue_args(
    const at::Tensor& D,
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
    TORCH_CHECK(rope_rotary_dim <= rope_head_dim, "rope_rotary_dim must be <= rope_head_dim");
    TORCH_CHECK(D.size(0) % rope_seq_len == 0, "output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D.size(1) % rope_head_dim == 0, "output cols must be divisible by rope_head_dim");
    check_rope_tensor(rope_cos, "rope_cos", rope_seq_len, rope_rotary_dim);
    check_rope_tensor(rope_sin, "rope_sin", rope_seq_len, rope_rotary_dim);
    kittens::py::device_check(D, rope_cos, rope_sin);
}

int64_t check_rope_live_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t seq_len
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == seq_len, name, " seq_len mismatch");
    const int64_t pair_dim = t.size(1);
    TORCH_CHECK(
        pair_dim >= 16 && pair_dim <= 128 && is_power_of_two(pair_dim),
        name, " second dim must be a power of two in [16, 128]"
    );
    TORCH_CHECK(t.size(2) == 2, name, " third dim must equal 2");
    return pair_dim;
}

int64_t check_rope_live_args(
    const at::Tensor& D,
    const at::Tensor& rope_cs,
    int64_t rope_seq_len
) {
    TORCH_CHECK(rope_seq_len > 0, "rope_seq_len must be positive");
    TORCH_CHECK(is_power_of_two(rope_seq_len), "rope_seq_len must be a power of two");
    TORCH_CHECK(D.size(0) % rope_seq_len == 0, "output rows must be divisible by rope_seq_len");
    const int64_t pair_dim = check_rope_live_tensor(rope_cs, "rope_cs", rope_seq_len);
    TORCH_CHECK(
        D.size(1) % (2 * pair_dim) == 0,
        "output cols must be divisible by the RoPE head dimension"
    );
    kittens::py::device_check(D, rope_cs);
    return pair_dim;
}

static bool use_rope_live64_rht32() {
    return std::getenv("MXFP4_GEMM_ROPE_LIVE64_RHT32") != nullptr;
}

template <typename C>
static void check_rope_live64_rht32_config(bool enabled) {
    if (enabled) {
        TORCH_CHECK((C::Nb / C::EPI_PIPE_DEPTH) % 32 == 0,
                    "MXFP4_GEMM_ROPE_LIVE64_RHT32 requires epilogue fragments divisible by 32 columns");
    }
}

template <typename C>
using rope_live64_rht32_config = mxfp4_gemm::config<
    C::Nb,
    C::LOAD_PIPE_DEPTH,
    C::EPI_PIPE_DEPTH,
    C::SUPERGROUP_SIZE,
    C::NUM_D_TILES,
    C::OVERLAP_EPI,
    C::Kb,
    true>;

void check_mxfp4_scale_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t rows,
    int64_t cols,
    bool allow_views
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    if (!allow_views) {
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    }
    TORCH_CHECK(t.dim() == 4, name, " must be 4D");
    TORCH_CHECK(
        t.scalar_type() == at::kFloat8_e8m0fnu || t.scalar_type() == at::kByte,
        name, " must be fp8 e8m0 or byte view"
    );
    TORCH_CHECK(t.size(0) == rows / 128, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols / 128, name, " second dim mismatch");
    TORCH_CHECK(t.size(2) == 32, name, " third dim must equal 32");
    TORCH_CHECK(t.size(3) == 16, name, " fourth dim must equal 16");

    if (allow_views) {
        TORCH_CHECK(t.stride(3) == 1, name, " last stride must be contiguous");
        TORCH_CHECK(t.stride(2) == 16, name, " stride(2) must equal 16");
        TORCH_CHECK(t.stride(1) == 512, name, " stride(1) must equal 512");
        TORCH_CHECK(t.stride(0) >= t.size(1) * 512, name, " leading stride too small");
        const auto data_ptr = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK((data_ptr & 0xF) == 0, name, " data pointer must be 16-byte aligned");
        TORCH_CHECK((t.stride(1) % 16) == 0, name, " stride(1) must be 16-byte aligned");
        TORCH_CHECK((t.stride(0) % 16) == 0, name, " stride(0) must be 16-byte aligned");
    }
}

template <typename ST>
void encode_mxfp4_scale_tensor_map(CUtensorMap* desc, const at::Tensor& t, const char* name) {
    static_assert(std::is_same_v<typename ST::dtype, kittens::fp8e8m0>,
                  "MXFP4 scale TMA helper assumes fp8e8m0 logical elements");
    static_assert(!ST::swizzle, "MXFP4 scale TMA helper only supports non-swizzled tiles");

    check_mxfp4_scale_tensor(t, name, t.size(0) * 128, t.size(1) * 128, true);

    uint64_t gmem_shape[4] = {
        static_cast<uint64_t>(t.size(3)),
        static_cast<uint64_t>(t.size(2)),
        static_cast<uint64_t>(t.size(1)),
        static_cast<uint64_t>(t.size(0)),
    };
    uint64_t gmem_stride[3] = {
        static_cast<uint64_t>(t.stride(2)),
        static_cast<uint64_t>(t.stride(1)),
        static_cast<uint64_t>(t.stride(0)),
    };
    uint32_t smem_shape[4] = {
        static_cast<uint32_t>(ST::cols),
        static_cast<uint32_t>(ST::rows),
        1, 1,
    };
    uint32_t smem_stride[4] = {1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
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

void check_mxfp4_split2_dgrad_inputs(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list
) {
    check_fp4_matrix(A_full, "A_full");
    TORCH_CHECK(A_full.size(0) % 128 == 0, "A_full M must be a multiple of 128");
    TORCH_CHECK((A_full.size(1) * 2) % 128 == 0, "A_full K must be a multiple of 128");

    const int n = static_cast<int>(A_sc_list.size());
    TORCH_CHECK(n == 2, "split2 one-pass dgrad expects exactly 2 A scale tensors");
    TORCH_CHECK(
        n == static_cast<int>(A_col_offsets.size()) &&
        n == static_cast<int>(A_col_widths.size()) &&
        n == static_cast<int>(B_list.size()) &&
        n == static_cast<int>(B_sc_list.size()),
        "all split2 one-pass inputs must have length 2"
    );

    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets must be non-negative");
        TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths must be positive");
        TORCH_CHECK(
            A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
            "A_full slice exceeds packed width"
        );
        TORCH_CHECK((A_col_widths[i] * 2) % 128 == 0, "split widths must be multiples of 128");
        check_mxfp4_scale_tensor(A_sc_list[i], "A_sc_list[i]", A_full.size(0), A_col_widths[i] * 2, true);
        check_fp4_matrix(B_list[i], "B_list[i]");
        TORCH_CHECK(B_list[i].size(1) == A_col_widths[i], "B_list packed K must match A_col_widths");
        TORCH_CHECK(B_list[i].size(0) % 128 == 0, "B_list rows must be multiples of 128");
        check_mxfp4_scale_tensor(B_sc_list[i], "B_sc_list[i]", B_list[i].size(0), B_list[i].size(1) * 2, false);
        kittens::py::device_check(A_full, A_sc_list[i], B_list[i], B_sc_list[i]);
    }
}

void check_mxfp4_split3_dgrad_inputs(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list
) {
    check_fp4_matrix(A_full, "A_full");
    TORCH_CHECK(A_full.size(0) % 128 == 0, "A_full M must be a multiple of 128");
    TORCH_CHECK((A_full.size(1) * 2) % 128 == 0, "A_full K must be a multiple of 128");

    const int n = static_cast<int>(A_sc_list.size());
    TORCH_CHECK(n == 3, "split3 one-pass dgrad expects exactly 3 A scale tensors");
    TORCH_CHECK(
        n == static_cast<int>(A_col_offsets.size()) &&
        n == static_cast<int>(A_col_widths.size()) &&
        n == static_cast<int>(B_list.size()) &&
        n == static_cast<int>(B_sc_list.size()),
        "all split3 one-pass inputs must have length 3"
    );

    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets must be non-negative");
        TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths must be positive");
        TORCH_CHECK(
            A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
            "A_full slice exceeds packed width"
        );
        TORCH_CHECK((A_col_widths[i] * 2) % 128 == 0, "split widths must be multiples of 128");
        check_mxfp4_scale_tensor(A_sc_list[i], "A_sc_list[i]", A_full.size(0), A_col_widths[i] * 2, true);
        check_fp4_matrix(B_list[i], "B_list[i]");
        TORCH_CHECK(B_list[i].size(1) == A_col_widths[i], "B_list packed K must match A_col_widths");
        TORCH_CHECK(B_list[i].size(0) % 128 == 0, "B_list rows must be multiples of 128");
        check_mxfp4_scale_tensor(B_sc_list[i], "B_sc_list[i]", B_list[i].size(0), B_list[i].size(1) * 2, false);
        kittens::py::device_check(A_full, A_sc_list[i], B_list[i], B_sc_list[i]);
    }
}


template <typename C>
void launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    const at::Tensor* h_z = nullptr,
    const at::Tensor* h_gamma = nullptr,
    const at::Tensor* h_r_tile = nullptr,
    at::Tensor* h_dgamma_partial = nullptr
) {
    using G = mxfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    if constexpr (C::FUSE_H_BWD) {
        g_host.h_z = reinterpret_cast<const bf16*>(h_z->data_ptr());
        g_host.h_gamma =
            reinterpret_cast<const bf16*>(h_gamma->data_ptr());
        g_host.h_r_tile = h_r_tile->data_ptr<float>();
        g_host.h_dgamma_partial = h_dgamma_partial->data_ptr<float>();
        g_host.h_cols = static_cast<int>(N_out);
    }

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
        TORCH_CHECK(result == CUDA_SUCCESS, "One-pass split2 MXFP4 A TMA creation failed for batch ", i);

        if (A_sc_list[i].is_contiguous()) {
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        } else {
            encode_mxfp4_scale_tensor_map<typename G::A_sc_tile>(&g_host.A_sc_tma[i], A_sc_list[i], "A_sc_list[i]");
        }

        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, mxfp4_split2_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out
) {
    using G = mxfp4_split3_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

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
        TORCH_CHECK(result == CUDA_SUCCESS, "One-pass split3 MXFP4 A TMA creation failed for batch ", i);

        if (A_sc_list[i].is_contiguous()) {
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        } else {
            encode_mxfp4_scale_tensor_map<typename G::A_sc_tile>(&g_host.A_sc_tma[i], A_sc_list[i], "A_sc_list[i]");
        }

        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, mxfp4_split3_accum_gemm::kernel<C>>(g_host);
}


void launch_mxfp4_split2_dgrad_gemm_strided_onepass(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 1:
            launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config<mxfp4_onepass_cfg1>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 3:
            launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config<mxfp4_onepass_cfg3>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 5:
            launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config<mxfp4_onepass_cfg5>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown MXFP4 split2 one-pass config_idx=", resolved_idx);
    }
}

void launch_mxfp4_split3_dgrad_gemm_strided_onepass(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 1:
            launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config<mxfp4_split3_onepass_cfg1>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 3:
            launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config<mxfp4_split3_onepass_cfg3>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 5:
            launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config<mxfp4_split3_onepass_cfg5>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown MXFP4 split3 one-pass config_idx=", resolved_idx);
    }
}


} // namespace

template <typename C>
static void launch_mxfp4_gemm_dense(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor* output_scale = nullptr,
    bool output_causal = false
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .output_scale = output_scale == nullptr ? nullptr : output_scale->data_ptr<float>(),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false,
        .output_causal = output_causal
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void launch_mxfp4_gemm_dense_residual(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false
    };
    auto r_gl = kittens::py::tensor_to_gl<typename G::D_gl>(R);
    memcpy(&g.R_tma, &r_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void launch_mxfp4_gemm_dense_residual_rms(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D,
    at::Tensor &row_rms_partial
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .row_rms_partial = row_rms_partial.data_ptr<float>(),
        .row_rms_partial_stride = static_cast<int>(row_rms_partial.size(1)),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false
    };
    auto r_gl = kittens::py::tensor_to_gl<typename G::D_gl>(R);
    memcpy(&g.R_tma, &r_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void launch_mxfp4_h_residual_carrier(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    const at::Tensor &gamma,
    at::Tensor &z_out,
    at::Tensor &row_fp4,
    at::Tensor &row_sc,
    at::Tensor &col_fp4,
    at::Tensor &col_sc,
    at::Tensor &r_tile,
    double eps
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(z_out),
        .output_scale = nullptr,
        .h_row_fp4 = reinterpret_cast<uint8_t*>(row_fp4.data_ptr()),
        .h_row_sc = row_sc.data_ptr<uint8_t>(),
        .h_col_fp4 = reinterpret_cast<uint8_t*>(col_fp4.data_ptr()),
        .h_col_sc = col_sc.data_ptr<uint8_t>(),
        .h_r_tile = r_tile.data_ptr<float>(),
        .h_gamma = reinterpret_cast<const bf16*>(gamma.data_ptr()),
        .h_rows = static_cast<int>(R.size(0)),
        .h_cols = static_cast<int>(R.size(1)),
        .h_eps = static_cast<float>(eps),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false
    };
    auto r_gl = kittens::py::tensor_to_gl<typename G::D_gl>(R);
    memcpy(&g.R_tma, &r_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void launch_mxfp4_gemm_masked(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .tilemask_ptr = tilemask.data_ptr<uint8_t>(),
        .tilemask_rows = static_cast<int>(tilemask.size(0)),
        .tilemask_cols = static_cast<int>(tilemask.size(1)),
        .tilemask_transposed = tilemask_transposed
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

void mxfp4_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    // Single config that works for all shapes with Kb=256.
    // config<256,5,8,4,2,false> = Nb=256, LOAD_PIPE=5, EPI=8, SG=4, DT=2, no overlap
    launch_mxfp4_gemm_dense<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>(A, A_sc, B, B_sc, D);
}

void mxfp4_gemm_scaled_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &output_scale
) {
    TORCH_CHECK(output_scale.is_cuda(), "output_scale must be a CUDA scalar tensor");
    TORCH_CHECK(output_scale.scalar_type() == at::kFloat, "output_scale must be float32");
    TORCH_CHECK(output_scale.numel() == 1, "output_scale must contain one element");
    const int64_t M = A.size(0);
    const int64_t N = B.size(0);
    const int64_t K = A.size(1) * 2;
    if (M == 4096 && N == 4096 && K == 128256) {
        // Llama-8B CCE dE: overlap the epilogue with an eight-tile
        // supergroup. This is bit-identical and 2-3% faster on GB200/B200.
        launch_mxfp4_gemm_dense<mxfp4_gemm::config<
            256, 5, 8, 8, 2, true, 256, false, false, true>>(
            A, A_sc, B, B_sc, D, &output_scale);
    } else if (M == 128256 && N == 4096 && K == 4096) {
        // Llama-8B CCE dW: the short reduction benefits from a shallow
        // epilogue while preserving the same eight-tile overlap.
        launch_mxfp4_gemm_dense<mxfp4_gemm::config<
            256, 5, 4, 8, 2, true, 256, false, false, true>>(
            A, A_sc, B, B_sc, D, &output_scale);
    } else {
        launch_mxfp4_gemm_dense<mxfp4_gemm::config<
            256, 5, 8, 4, 2, false, 256, false, false, true>>(
            A, A_sc, B, B_sc, D, &output_scale);
    }
}

void mxfp4_gemm_scaled_config_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &output_scale,
    int config_id
) {
    TORCH_CHECK(output_scale.is_cuda(), "output_scale must be a CUDA scalar tensor");
    TORCH_CHECK(output_scale.scalar_type() == at::kFloat, "output_scale must be float32");
    TORCH_CHECK(output_scale.numel() == 1, "output_scale must contain one element");
    switch (config_id) {
    case 0:
        launch_mxfp4_gemm_dense<mxfp4_gemm::config<
            256, 5, 8, 4, 2, false, 256, false, false, true>>(
            A, A_sc, B, B_sc, D, &output_scale);
        break;
    case 2:
        launch_mxfp4_gemm_dense<mxfp4_gemm::config<
            256, 5, 8, 8, 2, true, 256, false, false, true>>(
            A, A_sc, B, B_sc, D, &output_scale);
        break;
    case 17:
        launch_mxfp4_gemm_dense<mxfp4_gemm::config<
            256, 5, 4, 8, 2, true, 256, false, false, true>>(
            A, A_sc, B, B_sc, D, &output_scale);
        break;
    default:
        TORCH_CHECK(false, "Invalid scaled config_id: ", config_id, " (valid: 0, 2, 17)");
    }
}

void mxfp4_gemm_residual_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D
) {
    check_output_matrix(R, "R", D.size(0), D.size(1));
    kittens::py::device_check(A, A_sc, B, B_sc, R, D);
    launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, true>>(
        A, A_sc, B, B_sc, R, D);
}

void mxfp4_gemm_residual_rms_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D,
    at::Tensor &row_rms_partial
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B logical K mismatch");
    const int64_t M = A.size(0);
    const int64_t N = B.size(0);
    const int64_t K = A.size(1) * 2;
    TORCH_CHECK(
        M > 0 && M % 256 == 0 && N == 4096 && K == 14336,
        "MXFP4 exact C/D/E producer requires [M,4096,14336] with M divisible by 256");
    check_output_matrix(R, "R", M, N);
    check_output_matrix(D, "D", M, N);
    TORCH_CHECK(
        row_rms_partial.is_cuda() && row_rms_partial.is_contiguous() &&
            row_rms_partial.scalar_type() == at::kFloat &&
            row_rms_partial.dim() == 2 &&
            row_rms_partial.size(0) == M &&
            row_rms_partial.size(1) == N / 256,
        "row_rms_partial must be contiguous CUDA fp32 [M,N/256]");
    kittens::py::device_check(
        A, A_sc, B, B_sc, R, D, row_rms_partial);
    at::assert_no_overlap(D, R);
    at::assert_no_overlap(D, row_rms_partial);
    at::assert_no_overlap(row_rms_partial, R);
    const c10::cuda::CUDAGuard device_guard(A.device());

    using cde_config = mxfp4_gemm::config<
        256, 5, 4, 12, 2, false, 256, false, true, false, false, true, false>;
    launch_mxfp4_gemm_dense_residual_rms<cde_config>(
        A, A_sc, B, B_sc, R, D, row_rms_partial);
}

void mxfp4_h_residual_carrier_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    const at::Tensor &gamma,
    at::Tensor &z_out,
    at::Tensor &row_fp4,
    at::Tensor &row_sc,
    at::Tensor &col_fp4,
    at::Tensor &col_sc,
    at::Tensor &r_tile,
    double eps = 1.0e-5
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(1) == B.size(1), "H A and B logical K mismatch");
    const int64_t M = A.size(0);
    const int64_t N = B.size(0);
    check_output_matrix(R, "R", M, N);
    check_output_matrix(z_out, "z_out", M, N);
    TORCH_CHECK(
        M > 0 && N > 0 && M % 128 == 0 && N % 256 == 0,
        "H output dimensions must be positive multiples of 128x256");
    TORCH_CHECK(
        gamma.is_cuda() && gamma.is_contiguous() &&
            gamma.scalar_type() == at::kBFloat16 && gamma.dim() == 1 &&
            gamma.numel() == N,
        "H gamma must be contiguous CUDA bf16 [N]");
    TORCH_CHECK(
        row_fp4.is_cuda() && row_fp4.is_contiguous() &&
            row_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            row_fp4.sizes() == at::IntArrayRef({M, N / 2}),
        "H row_fp4 must be contiguous CUDA float4_e2m1fn_x2 [M,N/2]");
    TORCH_CHECK(
        col_fp4.is_cuda() && col_fp4.is_contiguous() &&
            col_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            col_fp4.sizes() == at::IntArrayRef({N, M / 2}),
        "H col_fp4 must be contiguous CUDA float4_e2m1fn_x2 [N,M/2]");
    TORCH_CHECK(
        row_sc.is_cuda() && row_sc.is_contiguous() &&
            row_sc.scalar_type() == at::kByte &&
            row_sc.sizes() == at::IntArrayRef({M / 128, N / 128, 32, 16}),
        "H row_sc must have contiguous E8M0 layout [M/128,N/128,32,16]");
    TORCH_CHECK(
        col_sc.is_cuda() && col_sc.is_contiguous() &&
            col_sc.scalar_type() == at::kByte &&
            col_sc.sizes() == at::IntArrayRef({N / 128, M / 128, 32, 16}),
        "H col_sc must have contiguous E8M0 layout [N/128,M/128,32,16]");
    TORCH_CHECK(
        r_tile.is_cuda() && r_tile.is_contiguous() &&
            r_tile.scalar_type() == at::kFloat &&
            r_tile.sizes() == at::IntArrayRef({M / 128, N / 128}),
        "H r_tile must be contiguous CUDA fp32 [M/128,N/128]");
    TORCH_CHECK(
        std::isfinite(eps) && eps >= 0.0,
        "H eps must be finite and non-negative");
    kittens::py::device_check(
        A, A_sc, B, B_sc, R, gamma, z_out, row_fp4, row_sc,
        col_fp4, col_sc, r_tile);
    const c10::cuda::CUDAGuard device_guard(A.device());
    launch_mxfp4_h_residual_carrier<mxfp4_gemm::config<
        256, 5, 4, 12, 2, false, 256, false, true, false, true>>(
        A, A_sc, B, B_sc, R, gamma, z_out, row_fp4, row_sc,
        col_fp4, col_sc, r_tile, eps);
}

__global__ void mxfp4_h_tile_backward_kernel(
    const bf16* __restrict__ du,
    const bf16* __restrict__ z,
    const bf16* __restrict__ gamma,
    const float* __restrict__ r_tile,
    bf16* __restrict__ dx,
    float* __restrict__ dgamma_partial,
    int rows,
    int cols
) {
    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float dot_s[256];
    const int row0 = tile_row * 128;
    const int col0 = tile_col * 128;
    float dot = 0.0f;
    #pragma unroll
    for (int e = tid; e < 128 * 128; e += 256) {
        const int r = row0 + e / 128;
        const int c = col0 + e % 128;
        const float zv = __bfloat162float(z[r * cols + c]);
        const float gv =
            __bfloat162float(du[r * cols + c]) *
            __bfloat162float(gamma[c]);
        dot = fmaf(gv, zv, dot);
    }
    dot_s[tid] = dot;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) dot_s[tid] += dot_s[tid + stride];
        __syncthreads();
    }
    const float rt = r_tile[tile_row * (cols / 128) + tile_col];
    const float corr = (rt * rt * rt) * (dot_s[0] * 0x1p-14f);
    #pragma unroll
    for (int e = tid; e < 128 * 128; e += 256) {
        const int r = row0 + e / 128;
        const int c = col0 + e % 128;
        const float zv = __bfloat162float(z[r * cols + c]);
        const float gv =
            __bfloat162float(du[r * cols + c]) *
            __bfloat162float(gamma[c]);
        dx[r * cols + c] =
            __float2bfloat16_rn(fmaf(-zv, corr, rt * gv));
    }
    if (tid < 128) {
        const int c = col0 + tid;
        float dg = 0.0f;
        #pragma unroll
        for (int rr = 0; rr < 128; ++rr) {
            const int off = (row0 + rr) * cols + c;
            const float norm = __bfloat162float(z[off]) * rt;
            dg = fmaf(__bfloat162float(du[off]), norm, dg);
        }
        dgamma_partial[tile_row * cols + c] = dg;
    }
}

__global__ void mxfp4_h_dgamma_reduce_kernel(
    const float* __restrict__ partial,
    bf16* __restrict__ dgamma,
    int tile_rows,
    int cols
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float sum = 0.0f;
    for (int tr = 0; tr < tile_rows; ++tr) {
        sum += partial[tr * cols + c];
    }
    dgamma[c] = __float2bfloat16_rn(sum);
}

void mxfp4_h_tile_backward_entrypoint(
    const at::Tensor &du,
    const at::Tensor &z,
    const at::Tensor &gamma,
    const at::Tensor &r_tile,
    at::Tensor &dx,
    at::Tensor &dgamma_partial,
    at::Tensor &dgamma
) {
    TORCH_CHECK(
        du.is_cuda() && du.is_contiguous() &&
            du.scalar_type() == at::kBFloat16 && du.dim() == 2,
        "H du must be contiguous CUDA bf16 [M,N]");
    TORCH_CHECK(
        z.is_cuda() && z.is_contiguous() &&
            z.scalar_type() == at::kBFloat16 && z.sizes() == du.sizes(),
        "H z must match du");
    const int64_t M = du.size(0);
    const int64_t N = du.size(1);
    TORCH_CHECK(
        M > 0 && N > 0 && M % 128 == 0 && N % 128 == 0,
        "H backward shape must be divisible by 128x128");
    TORCH_CHECK(
        gamma.is_cuda() && gamma.is_contiguous() &&
            gamma.scalar_type() == at::kBFloat16 && gamma.dim() == 1 &&
            gamma.numel() == N,
        "H backward gamma must be contiguous CUDA bf16 [N]");
    TORCH_CHECK(
        r_tile.is_cuda() && r_tile.is_contiguous() &&
            r_tile.scalar_type() == at::kFloat &&
            r_tile.sizes() == at::IntArrayRef({M / 128, N / 128}),
        "H backward r_tile shape mismatch");
    TORCH_CHECK(
        dx.is_cuda() && dx.is_contiguous() &&
            dx.scalar_type() == at::kBFloat16 && dx.sizes() == du.sizes(),
        "H backward dx must be CUDA bf16 [M,N]");
    TORCH_CHECK(
        dgamma_partial.is_cuda() && dgamma_partial.is_contiguous() &&
            dgamma_partial.scalar_type() == at::kFloat &&
            dgamma_partial.sizes() == at::IntArrayRef({M / 128, N}),
        "H dgamma_partial shape mismatch");
    TORCH_CHECK(
        dgamma.is_cuda() && dgamma.is_contiguous() &&
            dgamma.scalar_type() == at::kBFloat16 && dgamma.dim() == 1 &&
            dgamma.numel() == N,
        "H dgamma must be CUDA bf16 [N]");
    kittens::py::device_check(
        du, z, gamma, r_tile, dx, dgamma_partial, dgamma);
    const c10::cuda::CUDAGuard device_guard(du.device());
    const auto stream = at::cuda::getCurrentCUDAStream(du.get_device());
    mxfp4_h_tile_backward_kernel<<<
        dim3(N / 128, M / 128), 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(du.data_ptr()),
        reinterpret_cast<const bf16*>(z.data_ptr()),
        reinterpret_cast<const bf16*>(gamma.data_ptr()),
        r_tile.data_ptr<float>(),
        reinterpret_cast<bf16*>(dx.data_ptr()),
        dgamma_partial.data_ptr<float>(),
        M,
        N);
    mxfp4_h_dgamma_reduce_kernel<<<
        (N + 255) / 256, 256, 0, stream>>>(
        dgamma_partial.data_ptr<float>(),
        reinterpret_cast<bf16*>(dgamma.data_ptr()),
        M / 128,
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mxfp4_gemm_residual_config_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D,
    int config_id
) {
    check_output_matrix(R, "R", D.size(0), D.size(1));
    kittens::py::device_check(A, A_sc, B, B_sc, R, D);

    //                     Nb   LOAD EPI  SG  DT  OVERLAP Kb   RHT   RESIDUAL
    switch (config_id) {
    case 0:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  4, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 1:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 4, 16,  4, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 2:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  8, 2, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 3:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8, 12, 4, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 4:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 5:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5, 16,  4, 2, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 6:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 4,  8, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 7:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  4, 4, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 8:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 4, 16, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 9:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  4, 2, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 10: launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  4, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    default: TORCH_CHECK(false, "Invalid residual config_id: ", config_id, " (valid: 0-10)");
    }
}

void mxfp4_gemm_k128_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    // Same launch shape as the default consumer, but with Kb=128 so the
    // reduction granularity matches a single 128-column tile.
    launch_mxfp4_gemm_dense<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>(A, A_sc, B, B_sc, D);
}

void mxfp4_gemm_k128_output_causal_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    launch_mxfp4_gemm_dense<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>(
        A, A_sc, B, B_sc, D, nullptr, true);
}

void mxfp4_gemm_n128_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    // Attention PV with Dvo=128 only has one output-column tile. Avoid the
    // default Nb=256 kernel shape that computes a second unused half tile.
    launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 5, 4, 12, 2, true, 256>>(A, A_sc, B, B_sc, D);
}

void mxfp4_gemm_n128_config_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    int config_id
) {
    switch (config_id) {
    case 0: launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 5, 4,  4, 2, false, 256>>(A, A_sc, B, B_sc, D); break;
    case 1: launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 5, 4,  8, 2, false, 256>>(A, A_sc, B, B_sc, D); break;
    case 2: launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 5, 4, 12, 2, false, 256>>(A, A_sc, B, B_sc, D); break;
    case 3: launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 5, 4, 12, 2, true,  256>>(A, A_sc, B, B_sc, D); break;
    case 4: launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 4, 4,  4, 2, false, 256>>(A, A_sc, B, B_sc, D); break;
    case 5: launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 5, 8,  4, 2, false, 256>>(A, A_sc, B, B_sc, D); break;
    case 6: launch_mxfp4_gemm_dense<mxfp4_gemm::config<128, 4, 8, 12, 2, false, 256>>(A, A_sc, B, B_sc, D); break;
    default: TORCH_CHECK(false, "Invalid n128 config_id: ", config_id, " (valid: 0-6)");
    }
}

void mxfp4_gemm_atb_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    TORCH_CHECK(A.is_cuda() && A_sc.is_cuda() && B.is_cuda() && B_sc.is_cuda() && D.is_cuda(),
                "mxfp4_gemm_atb expects CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && A_sc.is_contiguous() && B.is_contiguous() && B_sc.is_contiguous() && D.is_contiguous(),
                "mxfp4_gemm_atb expects contiguous tensors");
    TORCH_CHECK(A.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A must be float4_e2m1fn_x2");
    TORCH_CHECK(B.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(D.dtype() == at::ScalarType::BFloat16, "D must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2,
                "mxfp4_gemm_atb expects A/B/D rank-2 tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must share K rows");
    const int64_t K = A.size(0);
    const int64_t M = D.size(0);
    const int64_t N = D.size(1);
    TORCH_CHECK(2 * A.size(1) == M, "A packed columns must equal D rows");
    TORCH_CHECK(2 * B.size(1) == N, "B packed columns must equal D cols");
    TORCH_CHECK(M % 256 == 0 && N % 256 == 0 && K % 256 == 0,
                "mxfp4_gemm_atb prototype expects M/N/K multiples of 256");
    TORCH_CHECK(A_sc.dim() == 4 && A_sc.size(0) == M / 128 && A_sc.size(1) == K / 128 &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (M/128, K/128, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4 && B_sc.size(0) == N / 128 && B_sc.size(1) == K / 128 &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (N/128, K/128, 32, 16)");

    using C = mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>;
    using G = mxfp4_atb_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
    };
    kittens::py::launch_kernel<C, G, mxfp4_atb_gemm::kernel<C>>(g);
}

void mxfp4_gemm_atbt_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    TORCH_CHECK(A.is_cuda() && A_sc.is_cuda() && B.is_cuda() && B_sc.is_cuda() && D.is_cuda(),
                "mxfp4_gemm_atbt expects CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && A_sc.is_contiguous() && B.is_contiguous() && B_sc.is_contiguous() && D.is_contiguous(),
                "mxfp4_gemm_atbt expects contiguous tensors");
    TORCH_CHECK(A.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A must be float4_e2m1fn_x2");
    TORCH_CHECK(B.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(D.dtype() == at::ScalarType::BFloat16, "D must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2,
                "mxfp4_gemm_atbt expects A/B/D rank-2 tensors");
    const int64_t K = A.size(0);
    const int64_t M = D.size(0);
    const int64_t N = D.size(1);
    TORCH_CHECK(2 * A.size(1) == M, "A packed columns must equal D rows");
    TORCH_CHECK(B.size(0) == N, "B rows must equal D cols");
    TORCH_CHECK(2 * B.size(1) == K, "B packed columns must equal A rows");
    TORCH_CHECK(M % 256 == 0 && N % 256 == 0 && K % 256 == 0,
                "mxfp4_gemm_atbt prototype expects M/N/K multiples of 256");
    TORCH_CHECK(A_sc.dim() == 4 && A_sc.size(0) == K / 128 && A_sc.size(1) == M / 128 &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (K/128, M/128, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4 && B_sc.size(0) == N / 128 && B_sc.size(1) == K / 128 &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (N/128, K/128, 32, 16)");

    using C = mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>;
    using G = mxfp4_atb_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
    };
    kittens::py::launch_kernel<C, G, mxfp4_atb_gemm::kernel<C, true, true>>(g);
}

void mxfp4_gemm_masked_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D
) {
    int64_t mask_rows = tilemask_transposed ? A_sc.size(1) : A_sc.size(0);
    int64_t mask_cols = tilemask_transposed ? A_sc.size(0) : A_sc.size(1);
    check_tilemask(tilemask, "tilemask", mask_rows, mask_cols);

    launch_mxfp4_gemm_masked<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>(
        A, A_sc, B, B_sc, tilemask, tilemask_transposed, D);
}

void mxfp4_gemm_masked_k128_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D
) {
    int64_t mask_rows = tilemask_transposed ? A_sc.size(1) : A_sc.size(0);
    int64_t mask_cols = tilemask_transposed ? A_sc.size(0) : A_sc.size(1);
    check_tilemask(tilemask, "tilemask", mask_rows, mask_cols);

    launch_mxfp4_gemm_masked<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>(
        A, A_sc, B, B_sc, tilemask, tilemask_transposed, D);
}


// ================================================================
// Config-selectable GEMM for tile tuning sweeps.
// ================================================================
template <typename C>
static void run_gemm_with_config(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_gemm_with_config_rope(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cos,
    const at::Tensor &rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .rope = {
            .cos = rope_cos.data_ptr<float>(),
            .sin = rope_sin.data_ptr<float>(),
            .seq_len = static_cast<int>(rope_seq_len),
            .head_dim = static_cast<int>(rope_head_dim),
            .rotary_dim = static_cast<int>(rope_rotary_dim),
        },
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_gemm_with_config_rope_live64_impl(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len
) {
    using G = mxfp4_gemm::globals<C>;
    const int pair_stride = static_cast<int>(rope_cs.size(1));
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .rope_live64 = {
            .cs = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>()),
            .seq_len = static_cast<int>(rope_seq_len),
            .seq_mask = static_cast<int>(rope_seq_len - 1),
            .pair_stride = pair_stride,
            .head_mask = 2 * pair_stride - 1,
        },
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_gemm_with_config_rope_live64(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len
) {
    const bool apply_rht32 = use_rope_live64_rht32();
    check_rope_live64_rht32_config<C>(apply_rht32);
    if (apply_rht32) {
        run_gemm_with_config_rope_live64_impl<rope_live64_rht32_config<C>>(
            A, A_sc, B, B_sc, D, rope_cs, rope_seq_len);
    } else {
        run_gemm_with_config_rope_live64_impl<C>(
            A, A_sc, B, B_sc, D, rope_cs, rope_seq_len);
    }
}

void mxfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D, int config_id
) {
    //                     Nb   LOAD EPI  SG  DT  OVERLAP
    switch (config_id) {
    // Defaults (used by mxfp4_gemm_entrypoint)
    case 0:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 1:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    // Best from Kb=128 sweep
    case 2:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 3:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D); break;
    // Additional candidates
    case 4:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 5:  run_gemm_with_config<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 6:  run_gemm_with_config<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 7:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 8:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 9:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 10: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(A, A_sc, B, B_sc, D); break;

    // EPI=4 neighborhood for the Llama-8B CCE logits shape.
    case 11: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4,  1, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 12: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4,  2, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 13: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 14: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4,  8, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 15: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4, 16, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 16: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 17: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4,  8, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 18: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4, 12, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 19: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4, 16, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 20: run_gemm_with_config<mxfp4_gemm::config<256, 4,  4,  8, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 21: run_gemm_with_config<mxfp4_gemm::config<256, 4,  4, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 22: run_gemm_with_config<mxfp4_gemm::config<256, 4,  4, 16, 2, false>>(A, A_sc, B, B_sc, D); break;

    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-22)");
    }
}

template <typename C>
static void run_silu_dgrad_quant_gemm_with_config(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    const at::Tensor &h3,
    const at::Tensor &h1_raw,
    const at::Tensor *sig_h1,
    at::Tensor &row_fp4,
    at::Tensor &row_sc,
    at::Tensor &col0_fp4,
    at::Tensor &col0_sc,
    at::Tensor &col1_fp4,
    at::Tensor &col1_sc
) {
    using G = mxfp4_silu_dgrad_quant_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .h3 = reinterpret_cast<const kittens::bf16*>(h3.data_ptr<at::BFloat16>()),
        .h1_raw = reinterpret_cast<const kittens::bf16*>(h1_raw.data_ptr<at::BFloat16>()),
        .sig_h1 = sig_h1 == nullptr ? nullptr : reinterpret_cast<const kittens::bf16*>(sig_h1->data_ptr<at::BFloat16>()),
        .row_fp4 = tensor_to_gl_tma_view<typename G::row_fp4_gl>(row_fp4, "row_fp4"),
        .row_sc = reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        .col0_fp4 = reinterpret_cast<uint8_t*>(col0_fp4.data_ptr()),
        .col0_sc = reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
        .col1_fp4 = reinterpret_cast<uint8_t*>(col1_fp4.data_ptr()),
        .col1_sc = reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
        .M = static_cast<int>(h3.size(0)),
        .H = static_cast<int>(h3.size(1)),
    };
    kittens::py::launch_kernel<C, G, mxfp4_silu_dgrad_quant_gemm::kernel<C>>(g);
}

static void mxfp4_gemm_silu_dgrad_quant_entrypoint_impl(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    const at::Tensor &h3,
    const at::Tensor &h1_raw,
    const at::Tensor *sig_h1,
    at::Tensor &row_fp4,
    at::Tensor &row_sc,
    at::Tensor &col0_fp4,
    at::Tensor &col0_sc,
    at::Tensor &col1_fp4,
    at::Tensor &col1_sc,
    int config_id,
    int mode
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A_sc.is_cuda() && A_sc.is_contiguous(), "A_sc must be contiguous CUDA");
    TORCH_CHECK(B_sc.is_cuda() && B_sc.is_contiguous(), "B_sc must be contiguous CUDA");
    TORCH_CHECK(A_sc.scalar_type() == at::kFloat8_e8m0fnu || A_sc.scalar_type() == at::kByte,
                "A_sc must be fp8e8m0/uint8");
    TORCH_CHECK(B_sc.scalar_type() == at::kFloat8_e8m0fnu || B_sc.scalar_type() == at::kByte,
                "B_sc must be fp8e8m0/uint8");
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous() && h3.dim() == 2 && h3.scalar_type() == at::kBFloat16,
                "h3 must be contiguous CUDA bf16 matrix");
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous() && h1_raw.dim() == 2 && h1_raw.scalar_type() == at::kBFloat16,
                "h1_raw must be contiguous CUDA bf16 matrix");
    TORCH_CHECK(h3.sizes() == h1_raw.sizes(), "h3 and h1_raw shape mismatch");
    if (sig_h1 != nullptr) {
        TORCH_CHECK(sig_h1->is_cuda() && sig_h1->is_contiguous() && sig_h1->dim() == 2 && sig_h1->scalar_type() == at::kBFloat16,
                    "sig_h1 must be contiguous CUDA bf16 matrix");
        TORCH_CHECK(sig_h1->sizes() == h1_raw.sizes(), "sig_h1 and h1_raw shape mismatch");
    }
    const int64_t M = h3.size(0);
    const int64_t H = h3.size(1);
    const int64_t K = A.size(1) * 2;
    TORCH_CHECK(A.size(0) == M && B.size(0) == H && B.size(1) * 2 == K,
                "A/B shapes must produce an MxH W2 dgrad tile");
    TORCH_CHECK(M % 256 == 0 && H % 256 == 0 && K % 256 == 0,
                "mxfp4_gemm_silu_dgrad_quant requires M,H,K divisible by 256");
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous() && row_fp4.dim() == 2 &&
                row_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
                row_fp4.size(0) == M && row_fp4.size(1) == H,
                "row_fp4 must be contiguous fp4x2 with shape (M, H)");
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous() && row_sc.scalar_type() == at::kByte &&
                row_sc.sizes() == at::IntArrayRef({M / 128, (2 * H) / 128, 32, 16}),
                "row_sc must have shape (M/128, 2H/128, 32, 16)");
    TORCH_CHECK(col0_fp4.is_cuda() && col0_fp4.is_contiguous() &&
                col0_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
                col0_fp4.sizes() == at::IntArrayRef({H, M / 2}),
                "col0_fp4 must have shape (H, M/2)");
    TORCH_CHECK(col1_fp4.is_cuda() && col1_fp4.is_contiguous() &&
                col1_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
                col1_fp4.sizes() == at::IntArrayRef({H, M / 2}),
                "col1_fp4 must have shape (H, M/2)");
    TORCH_CHECK(col0_sc.is_cuda() && col0_sc.is_contiguous() && col0_sc.scalar_type() == at::kByte &&
                col0_sc.sizes() == at::IntArrayRef({H / 128, M / 128, 32, 16}),
                "col0_sc must have shape (H/128, M/128, 32, 16)");
    TORCH_CHECK(col1_sc.is_cuda() && col1_sc.is_contiguous() && col1_sc.scalar_type() == at::kByte &&
                col1_sc.sizes() == at::IntArrayRef({H / 128, M / 128, 32, 16}),
                "col1_sc must have shape (H/128, M/128, 32, 16)");

    auto launch_mode = [&]<int MODE>() {
        switch (config_id) {
        case 0:
            if (sig_h1 != nullptr) {
                run_silu_dgrad_quant_gemm_with_config<mxfp4_silu_dgrad_quant_gemm::config<5, 4, MODE, true>>(
                    A, A_sc, B, B_sc, h3, h1_raw, sig_h1, row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc);
            } else {
                run_silu_dgrad_quant_gemm_with_config<mxfp4_silu_dgrad_quant_gemm::config<5, 4, MODE>>(
                    A, A_sc, B, B_sc, h3, h1_raw, nullptr, row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc);
            }
            break;
        case 4:
            if (sig_h1 != nullptr) {
                run_silu_dgrad_quant_gemm_with_config<mxfp4_silu_dgrad_quant_gemm::config<5, 12, MODE, true>>(
                    A, A_sc, B, B_sc, h3, h1_raw, sig_h1, row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc);
            } else {
                run_silu_dgrad_quant_gemm_with_config<mxfp4_silu_dgrad_quant_gemm::config<5, 12, MODE>>(
                    A, A_sc, B, B_sc, h3, h1_raw, nullptr, row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc);
            }
            break;
        default:
            TORCH_CHECK(false, "Invalid silu dgrad quant config_id: ", config_id, " (valid: 0, 4)");
        }
    };
    switch (mode) {
    case 0: launch_mode.template operator()<0>(); break;
    case 1: launch_mode.template operator()<1>(); break;
    case 2: launch_mode.template operator()<2>(); break;
    default:
        TORCH_CHECK(false, "Invalid silu dgrad quant mode: ", mode, " (valid: 0, 1, 2)");
    }
}

void mxfp4_gemm_silu_dgrad_quant_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    const at::Tensor &h3,
    const at::Tensor &h1_raw,
    at::Tensor &row_fp4,
    at::Tensor &row_sc,
    at::Tensor &col0_fp4,
    at::Tensor &col0_sc,
    at::Tensor &col1_fp4,
    at::Tensor &col1_sc,
    int config_id,
    int mode
) {
    mxfp4_gemm_silu_dgrad_quant_entrypoint_impl(
        A, A_sc, B, B_sc, h3, h1_raw, nullptr,
        row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc,
        config_id, mode);
}

void mxfp4_gemm_silu_dgrad_from_sigmoid_quant_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    const at::Tensor &h3,
    const at::Tensor &h1_raw,
    const at::Tensor &sig_h1,
    at::Tensor &row_fp4,
    at::Tensor &row_sc,
    at::Tensor &col0_fp4,
    at::Tensor &col0_sc,
    at::Tensor &col1_fp4,
    at::Tensor &col1_sc,
    int config_id,
    int mode
) {
    mxfp4_gemm_silu_dgrad_quant_entrypoint_impl(
        A, A_sc, B, B_sc, h3, h1_raw, &sig_h1,
        row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc,
        config_id, mode);
}

void mxfp4_gemm_rope_live64_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len
) {
    check_rope_live_args(D, rope_cs, rope_seq_len);
    run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>(
        A, A_sc, B, B_sc, D, rope_cs, rope_seq_len);
}

void mxfp4_gemm_rope_live64_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int config_id
) {
    check_rope_live_args(D, rope_cs, rope_seq_len);
    switch (config_id) {
    case 0:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 1:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 2:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 3:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 4:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 5:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 6:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 7:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 8:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 9:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 10: run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-10)");
    }
}

void mxfp4_gemm_rope_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cos,
    const at::Tensor &rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim
) {
    check_rope_epilogue_args(D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim);
    run_gemm_with_config_rope<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>(
        A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim);
}

void mxfp4_gemm_rope_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cos,
    const at::Tensor &rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim,
    int config_id
) {
    check_rope_epilogue_args(D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim);
    switch (config_id) {
    case 0:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 1:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 2:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 3:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 4:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 5:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 6:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 7:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 8:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 9:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 10: run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-10)");
    }
}

// ================================================================
// True Batched GEMM entrypoint
// D_out_list[i] = A_list[i] × B_list[i]^T, independently per batch.
// ================================================================
void mxfp4_batched_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;

        for (int i = 0; i < n; ++i) {
            int row_blocks = (int)(D_out_list[i].size(0) / C::Mb);
            int col_blocks = (int)(D_out_list[i].size(1) / C::Nb);
            int red_blocks = (int)(2 * A_list[i].size(1) / C::Kb);
            TORCH_CHECK(row_blocks > 0 && col_blocks > 0 && red_blocks > 0,
                        "mxfp4_batched_gemm expects positive tile counts");
            TORCH_CHECK(D_out_list[i].size(0) % C::Mb == 0,
                        "mxfp4_batched_gemm D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(D_out_list[i].size(1) % C::Nb == 0,
                        "mxfp4_batched_gemm D cols must be a multiple of ", C::Nb);
            TORCH_CHECK((2 * A_list[i].size(1)) % C::Kb == 0,
                        "mxfp4_batched_gemm K must be a multiple of ", C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = tensor_to_gl_tma_view<typename G::A_fp4x2_gl>(A_list[i], "A_list");
            auto a_sc_gl = tensor_to_gl_tma_view<typename G::A_sc_gl>(A_sc_list[i], "A_sc_list");
            auto b_gl = tensor_to_gl_tma_view<typename G::B_fp4x2_gl>(B_list[i], "B_list");
            auto b_sc_gl = tensor_to_gl_tma_view<typename G::B_sc_gl>(B_sc_list[i], "B_sc_list");
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = tensor_to_gl_tma_view<typename G::D_gl>(D_out_list[i], "D_out_list");
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C, false, false>>(g_host);
    };

    // For batched GEMM, use MMA_PER_TILE-friendly configs to avoid resource overflow.
    // DeepSeek expert hidden dims such as 1408 require Kb=128; Kb=256 would skip
    // the final 128-wide reduction tile.
    const int64_t K0 = 2 * A_list[0].size(1);
    if (N_out % 256 != 0 && N_out % 128 == 0) {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 128>>();
        }
    } else if (N_out <= 4096) {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>();
        }
    } else {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 128>>();
        }
    }
}

template <bool ATBT = false>
void mxfp4_grouped_gemm_strided_impl(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    int64_t num_batches,
    int64_t m_per_batch,
    int64_t n_per_batch,
    int64_t k_per_batch,
    int64_t a_row_stride,
    int64_t a_k_stride,
    int64_t b_row_stride,
    int64_t b_k_stride,
    int64_t d_row_stride,
    int config_id = -1,
    int64_t a_k_offset = 0,
    int64_t b_k_offset = 0,
    const at::Tensor *tilemask = nullptr,
    bool tilemask_transposed = false,
    bool output_causal = false
) {
    TORCH_CHECK(num_batches > 0, "num_batches must be positive");
    TORCH_CHECK(A.is_cuda() && A_sc.is_cuda() && B.is_cuda() && B_sc.is_cuda() && D.is_cuda(),
                "mxfp4_grouped_gemm_strided expects CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && A_sc.is_contiguous() && B.is_contiguous() && B_sc.is_contiguous() && D.is_contiguous(),
                "mxfp4_grouped_gemm_strided expects contiguous tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2,
                "mxfp4_grouped_gemm_strided expects flat 2D A/B/D tensors");
    TORCH_CHECK(m_per_batch > 0 && n_per_batch > 0 && k_per_batch > 0,
                "m/n/k per batch must be positive");
    TORCH_CHECK(D.size(1) == n_per_batch, "D second dim must equal n_per_batch");

    const int64_t M = m_per_batch;
    const int64_t N_out = n_per_batch;
    const int64_t K0 = k_per_batch;
    TORCH_CHECK(a_k_offset >= 0 && b_k_offset >= 0, "K offsets must be non-negative");
    if constexpr (ATBT) {
        TORCH_CHECK(a_k_offset == 0, "mxfp4_grouped_gemm_atbt_strided does not support A column offsets");
        TORCH_CHECK(A.size(0) >= num_batches * K0, "A rows must cover num_batches * k_per_batch");
        TORCH_CHECK(2 * A.size(1) >= M, "A width must cover m_per_batch");
        TORCH_CHECK(b_k_offset + K0 <= 2 * B.size(1), "B K offset plus K exceeds B width");
        TORCH_CHECK(K0 % 256 == 0, "mxfp4_grouped_gemm_atbt_strided requires K to be a multiple of 256");
    } else {
        TORCH_CHECK(a_k_offset + K0 <= 2 * A.size(1), "A K offset plus K exceeds A width");
        TORCH_CHECK(b_k_offset + K0 <= 2 * B.size(1), "B K offset plus K exceeds B width");
    }
    if (tilemask != nullptr) {
        TORCH_CHECK(tilemask->is_cuda(), "mxfp4_grouped_gemm_strided_masked expects a CUDA tilemask");
        TORCH_CHECK(tilemask->is_contiguous(), "mxfp4_grouped_gemm_strided_masked expects a contiguous tilemask");
        TORCH_CHECK(tilemask->dtype() == at::ScalarType::Byte, "mxfp4_grouped_gemm_strided_masked expects a uint8 tilemask");
        const int64_t masked_output_tiles = ATBT ? N_out / 128 : M / 128;
        const int64_t mask_rows = tilemask_transposed ? K0 / 128 : masked_output_tiles;
        const int64_t mask_cols = tilemask_transposed ? masked_output_tiles : K0 / 128;
        check_tilemask(*tilemask, "tilemask", mask_rows, mask_cols);
    }

    auto build_and_launch = [&]<typename C>() {
        if constexpr (ATBT && C::Kb != 256) {
            TORCH_CHECK(false, "mxfp4_grouped_gemm_atbt_strided requires a Kb=256 config");
        } else {
            using G = mxfp4_batched_gemm::globals<C>;
            G g_host {};
            g_host.uniform_strided = true;
            g_host.num_batches = (int)num_batches;
            g_host.num_row_blocks = (int)(M / C::Mb);
            g_host.num_col_blocks = (int)(N_out / C::Nb);
            g_host.num_red_blocks = (int)(K0 / C::Kb);
            g_host.total_spatial_tiles = 0;
            TORCH_CHECK(g_host.num_row_blocks > 0 && g_host.num_col_blocks > 0 && g_host.num_red_blocks > 0,
                        "mxfp4_grouped_gemm_strided expects positive tile counts");
            TORCH_CHECK(M % C::Mb == 0, "mxfp4_grouped_gemm_strided M must be a multiple of ", C::Mb);
            TORCH_CHECK(N_out % C::Nb == 0, "mxfp4_grouped_gemm_strided N must be a multiple of ", C::Nb);
            TORCH_CHECK(K0 % C::Kb == 0, "mxfp4_grouped_gemm_strided K must be a multiple of ", C::Kb);
            TORCH_CHECK(a_k_offset % C::Kb == 0 && b_k_offset % C::Kb == 0,
                        "K offsets must be multiples of the selected K tile");
            TORCH_CHECK(a_row_stride % 128 == 0 && b_row_stride % 128 == 0 && d_row_stride % 128 == 0,
                        "row strides must be multiples of 128");
            if constexpr (ATBT) {
                TORCH_CHECK(a_k_stride % 128 == 0 && b_k_stride % C::Kb == 0,
                            "AtBt A column stride must be a multiple of 128 and B K stride must be a multiple of the selected K tile");
            } else {
                TORCH_CHECK(a_k_stride % C::Kb == 0 && b_k_stride % C::Kb == 0,
                            "K strides must be multiples of the selected K tile");
            }
            g_host.a_row_block_stride = (int)(a_row_stride / 128);
            g_host.a_k_block_stride = (int)(a_k_stride / (ATBT ? 128 : C::Kb));
            g_host.b_row_block_stride = (int)(b_row_stride / 128);
            g_host.b_k_block_stride = (int)(b_k_stride / C::Kb);
            g_host.d_row_block_stride = (int)(d_row_stride / 128);
            g_host.a_k_block_offset = (int)(a_k_offset / (ATBT ? 128 : C::Kb));
            g_host.b_k_block_offset = (int)(b_k_offset / C::Kb);
            g_host.output_causal = output_causal;
            if (tilemask != nullptr) {
                g_host.tilemask_ptr = tilemask->data_ptr<uint8_t>();
                g_host.tilemask_rows = static_cast<int>(tilemask->size(0));
                g_host.tilemask_cols = static_cast<int>(tilemask->size(1));
                g_host.tilemask_transposed = tilemask_transposed;
            }

            auto a_gl = tensor_to_gl_tma_view<typename G::A_fp4x2_gl>(A, "A");
            auto a_sc_gl = tensor_to_gl_tma_view<typename G::A_sc_gl>(A_sc, "A_sc");
            auto b_gl = tensor_to_gl_tma_view<typename G::B_fp4x2_gl>(B, "B");
            auto b_sc_gl = tensor_to_gl_tma_view<typename G::B_sc_gl>(B_sc, "B_sc");
            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D);
            memcpy(&g_host.A_tma[0], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[0], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[0], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[0], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.D_tma[0], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C, ATBT, false>>(g_host);
        }
    };

    auto build_nb128_kb128 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, false, 128>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, true,  128>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  8, 2, true,  128>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 4, true,  128>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 2, false, 128>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided Nb=128 config_id: ", cfg, " (valid: 0-4)");
        }
    };
    auto build_nb128_kb256 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, false, 256>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, true,  256>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  8, 2, true,  256>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 4, true,  256>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 2, false, 256>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided Nb=128 config_id: ", cfg, " (valid: 0-4)");
        }
    };
    auto build_nb256_kb128 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false, 128>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true,  128>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true,  128>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true,  128>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false, 128>>(); break;
        case 5: build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false, 128>>(); break;
        case 6: build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false, 128>>(); break;
        case 7: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  4, 12, 2, false, 128>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided config_id: ", cfg, " (valid: 0-7)");
        }
    };
    auto build_nb256_kb256 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false, 256>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true,  256>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true,  256>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true,  256>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false, 256>>(); break;
        case 5: build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false, 256>>(); break;
        case 6: build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false, 256>>(); break;
        case 7: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  4, 12, 2, false, 256>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided config_id: ", cfg, " (valid: 0-7)");
        }
    };

    if (N_out % 256 != 0 && N_out % 128 == 0) {
        if (K0 % 256 == 0) {
            if (config_id >= 0) build_nb128_kb256((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 256>>();
        } else {
            if (config_id >= 0) build_nb128_kb128((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 128>>();
        }
    } else if (N_out <= 4096) {
        if (K0 % 256 == 0) {
            if (config_id >= 0) build_nb256_kb256((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>();
        } else {
            if (config_id >= 0) build_nb256_kb128((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>();
        }
    } else {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 128>>();
        }
    }
}

void mxfp4_grouped_gemm_strided_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    int64_t num_batches,
    int64_t m_per_batch,
    int64_t n_per_batch,
    int64_t k_per_batch,
    int64_t a_row_stride,
    int64_t a_k_stride,
    int64_t b_row_stride,
    int64_t b_k_stride,
    int64_t d_row_stride,
    int config_id = -1,
    int64_t a_k_offset = 0,
    int64_t b_k_offset = 0
) {
    mxfp4_grouped_gemm_strided_impl<false>(
        A, A_sc, B, B_sc, D, num_batches, m_per_batch, n_per_batch, k_per_batch,
        a_row_stride, a_k_stride, b_row_stride, b_k_stride, d_row_stride,
        config_id, a_k_offset, b_k_offset, nullptr, false, false);
}

void mxfp4_grouped_gemm_strided_output_causal_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    int64_t num_batches,
    int64_t m_per_batch,
    int64_t n_per_batch,
    int64_t k_per_batch,
    int64_t a_row_stride,
    int64_t a_k_stride,
    int64_t b_row_stride,
    int64_t b_k_stride,
    int64_t d_row_stride,
    int config_id = -1,
    int64_t a_k_offset = 0,
    int64_t b_k_offset = 0
) {
    mxfp4_grouped_gemm_strided_impl<false>(
        A, A_sc, B, B_sc, D, num_batches, m_per_batch, n_per_batch, k_per_batch,
        a_row_stride, a_k_stride, b_row_stride, b_k_stride, d_row_stride,
        config_id, a_k_offset, b_k_offset, nullptr, false, true);
}

void mxfp4_grouped_gemm_atbt_strided_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    int64_t num_batches,
    int64_t m_per_batch,
    int64_t n_per_batch,
    int64_t k_per_batch,
    int64_t a_row_stride,
    int64_t a_k_stride,
    int64_t b_row_stride,
    int64_t b_k_stride,
    int64_t d_row_stride,
    int config_id = -1,
    int64_t a_k_offset = 0,
    int64_t b_k_offset = 0
) {
    mxfp4_grouped_gemm_strided_impl<true>(
        A, A_sc, B, B_sc, D, num_batches, m_per_batch, n_per_batch, k_per_batch,
        a_row_stride, a_k_stride, b_row_stride, b_k_stride, d_row_stride,
        config_id, a_k_offset, b_k_offset, nullptr, false, false);
}

void mxfp4_grouped_gemm_atbt_strided_masked_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D,
    int64_t num_batches,
    int64_t m_per_batch,
    int64_t n_per_batch,
    int64_t k_per_batch,
    int64_t a_row_stride,
    int64_t a_k_stride,
    int64_t b_row_stride,
    int64_t b_k_stride,
    int64_t d_row_stride,
    int config_id = -1,
    int64_t a_k_offset = 0,
    int64_t b_k_offset = 0
) {
    mxfp4_grouped_gemm_strided_impl<true>(
        A, A_sc, B, B_sc, D, num_batches, m_per_batch, n_per_batch, k_per_batch,
        a_row_stride, a_k_stride, b_row_stride, b_k_stride, d_row_stride,
        config_id, a_k_offset, b_k_offset, &tilemask, tilemask_transposed, false);
}

void mxfp4_grouped_gemm_strided_masked_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D,
    int64_t num_batches,
    int64_t m_per_batch,
    int64_t n_per_batch,
    int64_t k_per_batch,
    int64_t a_row_stride,
    int64_t a_k_stride,
    int64_t b_row_stride,
    int64_t b_k_stride,
    int64_t d_row_stride,
    int config_id = -1,
    int64_t a_k_offset = 0,
    int64_t b_k_offset = 0
) {
    mxfp4_grouped_gemm_strided_impl<false>(
        A, A_sc, B, B_sc, D, num_batches, m_per_batch, n_per_batch, k_per_batch,
        a_row_stride, a_k_stride, b_row_stride, b_k_stride, d_row_stride,
        config_id, a_k_offset, b_k_offset, &tilemask, tilemask_transposed, false);
}

void mxfp4_batched_gemm_config_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    int config_id
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;

        for (int i = 0; i < n; ++i) {
            int row_blocks = (int)(D_out_list[i].size(0) / C::Mb);
            int col_blocks = (int)(D_out_list[i].size(1) / C::Nb);
            int red_blocks = (int)(2 * A_list[i].size(1) / C::Kb);
            TORCH_CHECK(row_blocks > 0 && col_blocks > 0 && red_blocks > 0,
                        "mxfp4_batched_gemm_config expects positive tile counts");
            TORCH_CHECK(D_out_list[i].size(0) % C::Mb == 0,
                        "mxfp4_batched_gemm_config D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(D_out_list[i].size(1) % C::Nb == 0,
                        "mxfp4_batched_gemm_config D cols must be a multiple of ", C::Nb);
            TORCH_CHECK((2 * A_list[i].size(1)) % C::Kb == 0,
                        "mxfp4_batched_gemm_config K must be a multiple of ", C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = tensor_to_gl_tma_view<typename G::A_fp4x2_gl>(A_list[i], "A_list");
            auto a_sc_gl = tensor_to_gl_tma_view<typename G::A_sc_gl>(A_sc_list[i], "A_sc_list");
            auto b_gl = tensor_to_gl_tma_view<typename G::B_fp4x2_gl>(B_list[i], "B_list");
            auto b_sc_gl = tensor_to_gl_tma_view<typename G::B_sc_gl>(B_sc_list[i], "B_sc_list");
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = tensor_to_gl_tma_view<typename G::D_gl>(D_out_list[i], "D_out_list");
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C, false, false>>(g_host);
    };

    switch (config_id) {
    case 0:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(); break;
    case 1:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(); break;
    case 2:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(); break;
    case 3:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(); break;
    case 4:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(); break;
    case 5:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(); break;
    case 6:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(); break;
    case 7:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(); break;
    case 8:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(); break;
    case 9:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-9)");
    }
}

void mxfp4_batched_gemm_rope_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    const std::vector<at::Tensor> &rope_cos_list,
    const std::vector<at::Tensor> &rope_sin_list,
    const std::vector<int64_t> &rope_seq_len_list,
    const std::vector<int64_t> &rope_head_dim_list,
    const std::vector<int64_t> &rope_rotary_dim_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());
    TORCH_CHECK(n == (int)rope_cos_list.size());
    TORCH_CHECK(n == (int)rope_sin_list.size());
    TORCH_CHECK(n == (int)rope_seq_len_list.size());
    TORCH_CHECK(n == (int)rope_head_dim_list.size());
    TORCH_CHECK(n == (int)rope_rotary_dim_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);

        for (int i = 0; i < n; ++i) {
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            if (!rope_tensor_disabled(rope_cos_list[i]) && !rope_tensor_disabled(rope_sin_list[i])) {
                check_rope_epilogue_args(
                    D_out_list[i],
                    rope_cos_list[i],
                    rope_sin_list[i],
                    rope_seq_len_list[i],
                    rope_head_dim_list[i],
                    rope_rotary_dim_list[i]);
                g_host.rope[i].cos = rope_cos_list[i].data_ptr<float>();
                g_host.rope[i].sin = rope_sin_list[i].data_ptr<float>();
                g_host.rope[i].seq_len = static_cast<int>(rope_seq_len_list[i]);
                g_host.rope[i].head_dim = static_cast<int>(rope_head_dim_list[i]);
                g_host.rope[i].rotary_dim = static_cast<int>(rope_rotary_dim_list[i]);
            }
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    if (N_out <= 4096) {
        build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false>>();
    }
}

void mxfp4_batched_gemm_rope_live64_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    const std::vector<at::Tensor> &rope_cs_list,
    const std::vector<int64_t> &rope_seq_len_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());
    TORCH_CHECK(n == (int)rope_cs_list.size());
    TORCH_CHECK(n == (int)rope_seq_len_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);

        for (int i = 0; i < n; ++i) {
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            if (!rope_tensor_disabled(rope_cs_list[i])) {
                const int pair_stride = static_cast<int>(
                    check_rope_live_args(D_out_list[i], rope_cs_list[i], rope_seq_len_list[i])
                );
                g_host.rope_live64[i].cs = reinterpret_cast<const float2*>(rope_cs_list[i].data_ptr<float>());
                g_host.rope_live64[i].seq_len = static_cast<int>(rope_seq_len_list[i]);
                g_host.rope_live64[i].seq_mask = static_cast<int>(rope_seq_len_list[i] - 1);
                g_host.rope_live64[i].pair_stride = pair_stride;
                g_host.rope_live64[i].head_mask = 2 * pair_stride - 1;
            }
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    const bool apply_rht32 = use_rope_live64_rht32();
    auto build_selected = [&]<typename Base>() {
        check_rope_live64_rht32_config<Base>(apply_rht32);
        if (apply_rht32) {
            build_and_launch.template operator()<rope_live64_rht32_config<Base>>();
        } else {
            build_and_launch.template operator()<Base>();
        }
    };

    if (N_out <= 4096) {
        build_selected.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_selected.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false>>();
    }
}

void mxfp4_batched_gemm_rope_live64_config_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    const std::vector<at::Tensor> &rope_cs_list,
    const std::vector<int64_t> &rope_seq_len_list,
    int config_id
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());
    TORCH_CHECK(n == (int)rope_cs_list.size());
    TORCH_CHECK(n == (int)rope_seq_len_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);

        for (int i = 0; i < n; ++i) {
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            if (!rope_tensor_disabled(rope_cs_list[i])) {
                const int pair_stride = static_cast<int>(
                    check_rope_live_args(D_out_list[i], rope_cs_list[i], rope_seq_len_list[i])
                );
                g_host.rope_live64[i].cs = reinterpret_cast<const float2*>(rope_cs_list[i].data_ptr<float>());
                g_host.rope_live64[i].seq_len = static_cast<int>(rope_seq_len_list[i]);
                g_host.rope_live64[i].seq_mask = static_cast<int>(rope_seq_len_list[i] - 1);
                g_host.rope_live64[i].pair_stride = pair_stride;
                g_host.rope_live64[i].head_mask = 2 * pair_stride - 1;
            }
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    const bool apply_rht32 = use_rope_live64_rht32();
    auto build_selected = [&]<typename Base>() {
        check_rope_live64_rht32_config<Base>(apply_rht32);
        if (apply_rht32) {
            build_and_launch.template operator()<rope_live64_rht32_config<Base>>();
        } else {
            build_and_launch.template operator()<Base>();
        }
    };

    switch (config_id) {
    case 0:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(); break;
    case 1:  build_selected.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(); break;
    case 2:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(); break;
    case 3:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(); break;
    case 4:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(); break;
    case 5:  build_selected.template operator()<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(); break;
    case 6:  build_selected.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(); break;
    case 7:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(); break;
    case 8:  build_selected.template operator()<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(); break;
    case 9:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(); break;
    case 10: build_selected.template operator()<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-10)");
    }
}

void mxfp4_split2_dgrad_strided_onepass_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_mxfp4_split2_dgrad_inputs(
        A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list);
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B tensors");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_mxfp4_split2_dgrad_gemm_strided_onepass(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        D_out,
        static_cast<int>(config_idx));
}

void mxfp4_split2_dgrad_strided_onepass_h_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const at::Tensor& z,
    const at::Tensor& gamma,
    const at::Tensor& r_tile,
    at::Tensor& D_out,
    at::Tensor& dgamma_partial,
    at::Tensor& dgamma,
    int64_t config_idx
) {
    check_mxfp4_split2_dgrad_inputs(
        A_full, A_sc_list, A_col_offsets, A_col_widths,
        B_list, B_sc_list);
    TORCH_CHECK(
        B_list.size() == 2,
        "split2 H dgrad expects exactly 2 B tensors");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    const int64_t M = D_out.size(0);
    const int64_t N = D_out.size(1);
    TORCH_CHECK(
        config_idx == -1 || config_idx == 5,
        "split2 H dgrad supports only production config 5");
    TORCH_CHECK(
        M > 0 && N > 0 && M % 128 == 0 && N % 256 == 0,
        "split2 H dgrad output must be divisible by 128x256");
    TORCH_CHECK(
        z.is_cuda() && z.is_contiguous() &&
            z.scalar_type() == at::kBFloat16 && z.sizes() == D_out.sizes(),
        "split2 H z must match D_out bf16 [M,N]");
    TORCH_CHECK(
        gamma.is_cuda() && gamma.is_contiguous() &&
            gamma.scalar_type() == at::kBFloat16 && gamma.dim() == 1 &&
            gamma.numel() == N,
        "split2 H gamma must be contiguous CUDA bf16 [N]");
    TORCH_CHECK(
        r_tile.is_cuda() && r_tile.is_contiguous() &&
            r_tile.scalar_type() == at::kFloat &&
            r_tile.sizes() == at::IntArrayRef({M / 128, N / 128}),
        "split2 H r_tile must be contiguous CUDA fp32 [M/128,N/128]");
    TORCH_CHECK(
        dgamma_partial.is_cuda() && dgamma_partial.is_contiguous() &&
            dgamma_partial.scalar_type() == at::kFloat &&
            dgamma_partial.sizes() == at::IntArrayRef({M / 128, N}),
        "split2 H dgamma_partial must be contiguous CUDA fp32 [M/128,N]");
    TORCH_CHECK(
        dgamma.is_cuda() && dgamma.is_contiguous() &&
            dgamma.scalar_type() == at::kBFloat16 && dgamma.dim() == 1 &&
            dgamma.numel() == N,
        "split2 H dgamma must be contiguous CUDA bf16 [N]");
    kittens::py::device_check(
        A_full, z, gamma, r_tile, D_out, dgamma_partial, dgamma);
    const c10::cuda::CUDAGuard device_guard(A_full.device());
    launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config<
        mxfp4_onepass_cfg5_h>(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        D_out,
        &z,
        &gamma,
        &r_tile,
        &dgamma_partial);
    const auto stream = at::cuda::getCurrentCUDAStream(A_full.get_device());
    mxfp4_h_dgamma_reduce_kernel<<<
        (N + 255) / 256, 256, 0, stream>>>(
        dgamma_partial.data_ptr<float>(),
        reinterpret_cast<bf16*>(dgamma.data_ptr()),
        M / 128,
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mxfp4_split3_dgrad_strided_onepass_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_mxfp4_split3_dgrad_inputs(
        A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list);
    TORCH_CHECK(B_list.size() == 3, "split3 one-pass dgrad expects exactly 3 B tensors");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_mxfp4_split3_dgrad_gemm_strided_onepass(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        D_out,
        static_cast<int>(config_idx));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_gemm", &mxfp4_gemm_entrypoint);
    m.def("mxfp4_gemm_scaled", &mxfp4_gemm_scaled_entrypoint,
          "MXFP4 GEMM with an extra CUDA scalar epilogue multiplier",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("output_scale"));
    m.def("mxfp4_gemm_scaled_config", &mxfp4_gemm_scaled_config_entrypoint,
          "MXFP4 GEMM with a CUDA scalar epilogue multiplier and explicit config",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("output_scale"),
          pybind11::arg("config_id"));
    m.def("mxfp4_gemm_residual", &mxfp4_gemm_residual_entrypoint,
          "Dense GEMM with fused bf16 residual add in the epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("R"), pybind11::arg("D"));
    m.def("mxfp4_gemm_residual_rms", &mxfp4_gemm_residual_rms_entrypoint,
          "Production W2 residual GEMM with exact row-sumsq CTA partials",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("R"), pybind11::arg("D"),
          pybind11::arg("row_rms_partial"));
    m.def("mxfp4_h_residual_carrier", &mxfp4_h_residual_carrier_entrypoint,
          "MXFP4 residual GEMM with direct 128x128 tile-RMS row/column carrier",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("R"), pybind11::arg("gamma"),
          pybind11::arg("z_out"), pybind11::arg("row_fp4"),
          pybind11::arg("row_sc"), pybind11::arg("col_fp4"),
          pybind11::arg("col_sc"), pybind11::arg("r_tile"),
          pybind11::arg("eps") = 1.0e-5);
    m.def("mxfp4_h_tile_backward", &mxfp4_h_tile_backward_entrypoint,
          "Native MX H tile-RMS backward and deterministic dgamma reduction",
          pybind11::arg("du"), pybind11::arg("z"),
          pybind11::arg("gamma"), pybind11::arg("r_tile"),
          pybind11::arg("dx"), pybind11::arg("dgamma_partial"),
          pybind11::arg("dgamma"));
    m.def("mxfp4_gemm_residual_config", &mxfp4_gemm_residual_config_entrypoint,
          "Dense GEMM with fused bf16 residual add and explicit kernel config",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("R"), pybind11::arg("D"),
          pybind11::arg("config_id"));
    m.def("mxfp4_gemm_k128", &mxfp4_gemm_k128_entrypoint);
    m.def("mxfp4_gemm_k128_output_causal", &mxfp4_gemm_k128_output_causal_entrypoint,
          "MXFP4 K128 GEMM that skips output tiles strictly above the causal diagonal");
    m.def("mxfp4_gemm_atb", &mxfp4_gemm_atb_entrypoint,
          "Prototype MXFP4 GEMM computing D = A^T x B for row-major packed operands",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"));
    m.def("mxfp4_gemm_atbt", &mxfp4_gemm_atbt_entrypoint,
          "Prototype MXFP4 GEMM computing D = A^T x B^T for row-major/transposed packed operands",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"));
    m.def("mxfp4_gemm_n128", &mxfp4_gemm_n128_entrypoint);
    m.def("mxfp4_gemm_n128_config", &mxfp4_gemm_n128_config_entrypoint,
          "N=128 MXFP4 GEMM with selectable tile config",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("mxfp4_gemm_masked", &mxfp4_gemm_masked_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("tilemask"), pybind11::arg("tilemask_transposed"),
          pybind11::arg("D"));
    m.def("mxfp4_gemm_masked_k128", &mxfp4_gemm_masked_k128_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("tilemask"), pybind11::arg("tilemask_transposed"),
          pybind11::arg("D"));
    m.def("mxfp4_gemm_config", &mxfp4_gemm_config_entrypoint,
          "GEMM with selectable tile config (for sweeping)",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("mxfp4_gemm_silu_dgrad_quant", &mxfp4_gemm_silu_dgrad_quant_entrypoint,
          "W2 dgrad GEMM with fused SiLU derivative MXFP4 row/col quantization",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("h3"), pybind11::arg("h1_raw"),
          pybind11::arg("row_fp4"), pybind11::arg("row_sc"),
          pybind11::arg("col0_fp4"), pybind11::arg("col0_sc"),
          pybind11::arg("col1_fp4"), pybind11::arg("col1_sc"),
          pybind11::arg("config_id"), pybind11::arg("mode") = 1);
    m.def("mxfp4_gemm_silu_dgrad_from_sigmoid_quant", &mxfp4_gemm_silu_dgrad_from_sigmoid_quant_entrypoint,
          "W2 dgrad GEMM with fused saved-sigmoid SiLU derivative MXFP4 row/col quantization",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("h3"), pybind11::arg("h1_raw"), pybind11::arg("sig_h1"),
          pybind11::arg("row_fp4"), pybind11::arg("row_sc"),
          pybind11::arg("col0_fp4"), pybind11::arg("col0_sc"),
          pybind11::arg("col1_fp4"), pybind11::arg("col1_sc"),
          pybind11::arg("config_id"), pybind11::arg("mode") = 1);
    m.def("mxfp4_gemm_rope", &mxfp4_gemm_rope_entrypoint,
          "GEMM with a RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cos"), pybind11::arg("rope_sin"),
          pybind11::arg("rope_seq_len"),
          pybind11::arg("rope_head_dim"),
          pybind11::arg("rope_rotary_dim"));
    m.def("mxfp4_gemm_rope_live64", &mxfp4_gemm_rope_live64_entrypoint,
          "GEMM with an exact-shape live64 RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"));
    m.def("mxfp4_gemm_rope_live64_config", &mxfp4_gemm_rope_live64_config_entrypoint,
          "GEMM with selectable tile config and an exact-shape live64 RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"),
          pybind11::arg("config_id"));
    m.def("mxfp4_gemm_rope_live", &mxfp4_gemm_rope_live64_entrypoint,
          "GEMM with a packed power-of-two RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"));
    m.def("mxfp4_gemm_rope_live_config", &mxfp4_gemm_rope_live64_config_entrypoint,
          "GEMM with selectable tile config and a packed power-of-two RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"),
          pybind11::arg("config_id"));
    m.def("mxfp4_gemm_rope_config", &mxfp4_gemm_rope_config_entrypoint,
          "GEMM with selectable tile config and a RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cos"), pybind11::arg("rope_sin"),
          pybind11::arg("rope_seq_len"),
          pybind11::arg("rope_head_dim"),
          pybind11::arg("rope_rotary_dim"),
          pybind11::arg("config_id"));

    m.def("mxfp4_batched_gemm", &mxfp4_batched_gemm_entrypoint,
          "True Batched GEMM: D_i = A_i × B_i^T, independently per batch",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"));
    m.def("mxfp4_grouped_gemm_strided", &mxfp4_grouped_gemm_strided_entrypoint,
          "Uniform grouped GEMM over flat packed tensors using one TMA descriptor per operand",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("num_batches"),
          pybind11::arg("m_per_batch"),
          pybind11::arg("n_per_batch"),
          pybind11::arg("k_per_batch"),
          pybind11::arg("a_row_stride"),
          pybind11::arg("a_k_stride"),
          pybind11::arg("b_row_stride"),
          pybind11::arg("b_k_stride"),
          pybind11::arg("d_row_stride"),
          pybind11::arg("config_id") = -1,
          pybind11::arg("a_k_offset") = 0,
          pybind11::arg("b_k_offset") = 0);
    m.def("mxfp4_grouped_gemm_strided_output_causal", &mxfp4_grouped_gemm_strided_output_causal_entrypoint,
          "Uniform grouped GEMM that skips output blocks above the causal diagonal",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("num_batches"),
          pybind11::arg("m_per_batch"),
          pybind11::arg("n_per_batch"),
          pybind11::arg("k_per_batch"),
          pybind11::arg("a_row_stride"),
          pybind11::arg("a_k_stride"),
          pybind11::arg("b_row_stride"),
          pybind11::arg("b_k_stride"),
          pybind11::arg("d_row_stride"),
          pybind11::arg("config_id") = -1,
          pybind11::arg("a_k_offset") = 0,
          pybind11::arg("b_k_offset") = 0);
    m.def("mxfp4_grouped_gemm_atbt_strided", &mxfp4_grouped_gemm_atbt_strided_entrypoint,
          "Uniform grouped AtBt GEMM over flat packed tensors using one TMA descriptor per operand",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("num_batches"),
          pybind11::arg("m_per_batch"),
          pybind11::arg("n_per_batch"),
          pybind11::arg("k_per_batch"),
          pybind11::arg("a_row_stride"),
          pybind11::arg("a_k_stride"),
          pybind11::arg("b_row_stride"),
          pybind11::arg("b_k_stride"),
          pybind11::arg("d_row_stride"),
          pybind11::arg("config_id") = -1,
          pybind11::arg("a_k_offset") = 0,
          pybind11::arg("b_k_offset") = 0);
    m.def("mxfp4_grouped_gemm_atbt_strided_masked", &mxfp4_grouped_gemm_atbt_strided_masked_entrypoint,
          "Uniform grouped AtBt GEMM with a shared per-GEMM tilemask for inactive reduction tiles",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("tilemask"),
          pybind11::arg("tilemask_transposed"),
          pybind11::arg("D"),
          pybind11::arg("num_batches"),
          pybind11::arg("m_per_batch"),
          pybind11::arg("n_per_batch"),
          pybind11::arg("k_per_batch"),
          pybind11::arg("a_row_stride"),
          pybind11::arg("a_k_stride"),
          pybind11::arg("b_row_stride"),
          pybind11::arg("b_k_stride"),
          pybind11::arg("d_row_stride"),
          pybind11::arg("config_id") = -1,
          pybind11::arg("a_k_offset") = 0,
          pybind11::arg("b_k_offset") = 0);
    m.def("mxfp4_grouped_gemm_strided_masked", &mxfp4_grouped_gemm_strided_masked_entrypoint,
          "Uniform grouped GEMM with a shared per-GEMM tilemask for inactive reduction tiles",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("tilemask"),
          pybind11::arg("tilemask_transposed"),
          pybind11::arg("D"),
          pybind11::arg("num_batches"),
          pybind11::arg("m_per_batch"),
          pybind11::arg("n_per_batch"),
          pybind11::arg("k_per_batch"),
          pybind11::arg("a_row_stride"),
          pybind11::arg("a_k_stride"),
          pybind11::arg("b_row_stride"),
          pybind11::arg("b_k_stride"),
          pybind11::arg("d_row_stride"),
          pybind11::arg("config_id") = -1,
          pybind11::arg("a_k_offset") = 0,
          pybind11::arg("b_k_offset") = 0);
    m.def("mxfp4_batched_gemm_config", &mxfp4_batched_gemm_config_entrypoint,
          "True Batched GEMM with selectable tile config",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"), pybind11::arg("config_id"));
    m.def("mxfp4_batched_gemm_rope", &mxfp4_batched_gemm_rope_entrypoint,
          "True Batched GEMM with an optional per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cos_list"), pybind11::arg("rope_sin_list"),
          pybind11::arg("rope_seq_len_list"),
          pybind11::arg("rope_head_dim_list"),
          pybind11::arg("rope_rotary_dim_list"));
    m.def("mxfp4_batched_gemm_rope_live64", &mxfp4_batched_gemm_rope_live64_entrypoint,
          "True Batched GEMM with an exact-shape live64 per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cs_list"),
          pybind11::arg("rope_seq_len_list"));
    m.def("mxfp4_batched_gemm_rope_live64_config", &mxfp4_batched_gemm_rope_live64_config_entrypoint,
          "True Batched GEMM with selectable tile config and an exact-shape live64 per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cs_list"),
          pybind11::arg("rope_seq_len_list"),
          pybind11::arg("config_id"));
    m.def("mxfp4_batched_gemm_rope_live", &mxfp4_batched_gemm_rope_live64_entrypoint,
          "True Batched GEMM with a packed power-of-two per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cs_list"),
          pybind11::arg("rope_seq_len_list"));
    m.def("mxfp4_batched_gemm_rope_live_config", &mxfp4_batched_gemm_rope_live64_config_entrypoint,
          "True Batched GEMM with selectable tile config and a packed power-of-two per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cs_list"),
          pybind11::arg("rope_seq_len_list"),
          pybind11::arg("config_id"));
    m.def("mxfp4_split2_dgrad_strided_onepass_gemm",
          &mxfp4_split2_dgrad_strided_onepass_gemm_entrypoint,
          "MXFP4 split2 one-pass dgrad GEMM with strided row slices",
          pybind11::arg("A_full"),
          pybind11::arg("A_sc_list"),
          pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"),
          pybind11::arg("D_out"),
          pybind11::arg("config_idx") = -1);
    m.def("mxfp4_split2_dgrad_strided_onepass_h_gemm",
          &mxfp4_split2_dgrad_strided_onepass_h_gemm_entrypoint,
          "MXFP4 split2 one-pass dgrad with fused H tile backward",
          pybind11::arg("A_full"),
          pybind11::arg("A_sc_list"),
          pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"),
          pybind11::arg("z"),
          pybind11::arg("gamma"),
          pybind11::arg("r_tile"),
          pybind11::arg("D_out"),
          pybind11::arg("dgamma_partial"),
          pybind11::arg("dgamma"),
          pybind11::arg("config_idx") = -1);
    m.def("mxfp4_split3_dgrad_strided_onepass_gemm",
          &mxfp4_split3_dgrad_strided_onepass_gemm_entrypoint,
          "MXFP4 split3 one-pass dgrad GEMM with strided row slices",
          pybind11::arg("A_full"),
          pybind11::arg("A_sc_list"),
          pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"),
          pybind11::arg("D_out"),
          pybind11::arg("config_idx") = -1);
}

#endif
