// ================================================================
// NVFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "nvfp4_gemm.cuh"
#include "nvfp4_quantize.cuh"
#include "nvfp4_batched_accum_gemm.cuh"
#include "nvfp4_accum_gemm.cuh"
#include "nvfp4_fused_gemm.cuh"
#include "nvfp4_persistent_gemm.cuh"
#include <optional>

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ nvfp4_gemm::globals<C> g) {
    nvfp4_gemm::kernel<C>(g);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = nvfp4_gemm::globals<C>;

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
    std::vector<__nv_fp8_e4m3*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B_sc(arg_group_count);
    std::vector<float*> d_A_sc_global(arg_group_count);
    std::vector<float*> d_B_sc_global(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e4m3)/16);
        cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e4m3)/16);
        cudaMalloc(&d_A_sc_global[i], sizeof(float));
        cudaMalloc(&d_B_sc_global[i], sizeof(float));
        cudaMalloc(&d_D[i], M * N * sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M * N * sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i * 100, 0.0f, 255.0f);
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i * 100 + 1, 0.0f, 255.0f);    
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A_sc[i], M*K/16, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B_sc[i], N*K/16, seed + i*100 + 3, 0.1f, 10.0f);
        fill<float, FillMode::RANDOM>(d_A_sc_global[i], 1, seed + i * 100 + 4, 0.1f, 10.0f);
        fill<float, FillMode::RANDOM>(d_B_sc_global[i], 1, seed + i * 100 + 5, 0.1f, 10.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_nvfp4_gemm<__nv_bfloat16>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], d_A_sc_global[0], d_B_sc_global[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    // Note: The kernel expects scales as half, but we store fp8e4m3. Reinterpret the pointers.
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl Ag{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl Asg{reinterpret_cast<half*>(d_A_sc[i]), nullptr, M/128, K/64, nullptr};
        typename G::A_sc_global_gl Asgg{d_A_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::B_fp4x2_gl Bg{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl Bsg{reinterpret_cast<half*>(d_B_sc[i]), nullptr, N/128, K/64, nullptr};
        typename G::B_sc_global_gl Bsgg{d_B_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Asgg, Bg, Bsg, Bsgg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));
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
        cudaFree(d_A_sc[i]);
        cudaFree(d_A_sc_global[i]);
        cudaFree(d_B[i]);
        cudaFree(d_B_sc[i]);
        cudaFree(d_B_sc_global[i]);
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
    run_benchmark<nvfp4_gemm::config<128, 5, 4, 12, 2, true>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<nvfp4_gemm::config<256, 5, 8, 4, 2, true>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<nvfp4_gemm::config<256, 4, 16, 1, 2, false>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<nvfp4_gemm::config<256, 4, 16, 12, 2, false>>(N, N, N, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"
#include "ATen/Functions.h"
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_reduce.cuh>
#include "../common/c1_rms_reduce.cuh"

namespace c3_row_scale {
static void check_output_no_overlap(
    const at::Tensor& output, const at::Tensor& input, const char* input_name
) {
    const auto out_begin = reinterpret_cast<std::uintptr_t>(output.data_ptr());
    const auto in_begin = reinterpret_cast<std::uintptr_t>(input.data_ptr());
    const auto out_end = out_begin + output.numel() * output.element_size();
    const auto in_end = in_begin + input.numel() * input.element_size();
    TORCH_CHECK(out_end <= in_begin || in_end <= out_begin,
                "output must not overlap ", input_name);
}
}  // namespace c3_row_scale

template <int BLOCK_SIZE>
__device__ __forceinline__ float v5_sum3_block_reduce_sum(float value) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage storage;
    return BlockReduce(storage).Sum(value);
}

template <int BLOCK_SIZE>
__global__ void v5_sum3_rmsnorm_bwd_dx_kernel(
    const __nv_bfloat16* __restrict__ d0,
    const __nv_bfloat16* __restrict__ d1,
    const __nv_bfloat16* __restrict__ d2,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    __nv_bfloat16* __restrict__ d_sum,
    __nv_bfloat16* __restrict__ grad_input,
    int M,
    int K
) {
    int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    int tid = threadIdx.x;
    const __nv_bfloat16* row_d0 = d0 + row * K;
    const __nv_bfloat16* row_d1 = d1 + row * K;
    const __nv_bfloat16* row_d2 = d2 + row * K;
    const __nv_bfloat16* row_input = input + row * K;
    __nv_bfloat16* row_sum = d_sum + row * K;
    __nv_bfloat16* row_dx = grad_input + row * K;
    float inv = inv_rms[row];
    float local_sum = 0.0f;

    for (int col = tid; col < K; col += BLOCK_SIZE) {
        __nv_bfloat16 dy_bf16 = __hadd(
            __hadd(row_d0[col], row_d1[col]), row_d2[col]
        );
        row_sum[col] = dy_bf16;
        float x = __bfloat162float(row_input[col]);
        float gamma = __bfloat162float(norm_weight[col]);
        float d_y = __bfloat162float(dy_bf16);
        float normed = x * inv * gamma;
        local_sum += d_y * normed;
    }
    float reduced = v5_sum3_block_reduce_sum<BLOCK_SIZE>(local_sum);
    __shared__ float mean;
    if (tid == 0) {
        mean = reduced / static_cast<float>(K);
    }
    __syncthreads();

    float mean_value = mean;
    for (int col = tid; col < K; col += BLOCK_SIZE) {
        float x = __bfloat162float(row_input[col]);
        float gamma = __bfloat162float(norm_weight[col]);
        float d_y = __bfloat162float(row_sum[col]);
        float d_z = d_y * gamma;
        float dx = inv * (d_z - x * inv * mean_value);
        row_dx[col] = __float2bfloat16(dx);
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void v5_sum3_rmsnorm_bwd_dx_k4096_kernel(
    const __nv_bfloat16* __restrict__ d0,
    const __nv_bfloat16* __restrict__ d1,
    const __nv_bfloat16* __restrict__ d2,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    __nv_bfloat16* __restrict__ d_sum,
    __nv_bfloat16* __restrict__ grad_input,
    int M
) {
    static_assert(BLOCK_SIZE * ITEMS_PER_THREAD == 4096);
    int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    int tid = threadIdx.x;
    constexpr int K = BLOCK_SIZE * ITEMS_PER_THREAD;
    const __nv_bfloat16* row_d0 = d0 + row * K;
    const __nv_bfloat16* row_d1 = d1 + row * K;
    const __nv_bfloat16* row_d2 = d2 + row * K;
    const __nv_bfloat16* row_input = input + row * K;
    __nv_bfloat16* row_sum = d_sum + row * K;
    __nv_bfloat16* row_dx = grad_input + row * K;
    __nv_bfloat16 dy_cache[ITEMS_PER_THREAD];
    __nv_bfloat16 x_cache[ITEMS_PER_THREAD];
    __nv_bfloat16 gamma_cache[ITEMS_PER_THREAD];
    float inv = inv_rms[row];
    float local_sum = 0.0f;

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = tid + item * BLOCK_SIZE;
        __nv_bfloat16 dy_bf16 = __hadd(
            __hadd(row_d0[col], row_d1[col]), row_d2[col]
        );
        __nv_bfloat16 x_bf16 = row_input[col];
        __nv_bfloat16 gamma_bf16 = norm_weight[col];
        dy_cache[item] = dy_bf16;
        x_cache[item] = x_bf16;
        gamma_cache[item] = gamma_bf16;
        row_sum[col] = dy_bf16;
        float x = __bfloat162float(x_bf16);
        float gamma = __bfloat162float(gamma_bf16);
        float d_y = __bfloat162float(dy_bf16);
        float normed = x * inv * gamma;
        local_sum += d_y * normed;
    }
    float reduced = v5_sum3_block_reduce_sum<BLOCK_SIZE>(local_sum);
    __shared__ float mean;
    if (tid == 0) {
        mean = reduced / static_cast<float>(K);
    }
    __syncthreads();

    float mean_value = mean;
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = tid + item * BLOCK_SIZE;
        float x = __bfloat162float(x_cache[item]);
        float gamma = __bfloat162float(gamma_cache[item]);
        float d_y = __bfloat162float(dy_cache[item]);
        float d_z = d_y * gamma;
        float dx = inv * (d_z - x * inv * mean_value);
        row_dx[col] = __float2bfloat16(dx);
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void v5_rmsnorm_bwd_residual_cached_dx_k4096_kernel(
    const __nv_bfloat16* __restrict__ d_normed,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    const __nv_bfloat16* __restrict__ residual_grad,
    __nv_bfloat16* __restrict__ grad_input,
    float* __restrict__ dgamma,
    int M
) {
    static_assert(BLOCK_SIZE * ITEMS_PER_THREAD == 4096);
    int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    int tid = threadIdx.x;
    constexpr int K = 4096;
    const __nv_bfloat16* row_dy = d_normed + row * K;
    const __nv_bfloat16* row_input = input + row * K;
    const __nv_bfloat16* row_residual = residual_grad + row * K;
    __nv_bfloat16* row_dx = grad_input + row * K;
    __nv_bfloat16 dy_cache[ITEMS_PER_THREAD];
    __nv_bfloat16 x_cache[ITEMS_PER_THREAD];
    float inv = inv_rms[row];
    float local_sum = 0.0f;

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = tid + item * BLOCK_SIZE;
        __nv_bfloat16 dy_bf16 = row_dy[col];
        __nv_bfloat16 x_bf16 = row_input[col];
        __nv_bfloat16 gamma_bf16 = norm_weight[col];
        dy_cache[item] = dy_bf16;
        x_cache[item] = x_bf16;
        float d_y = __bfloat162float(dy_bf16);
        float x_hat = __bfloat162float(x_bf16) * inv;
        float gamma = __bfloat162float(gamma_bf16);
        local_sum += d_y * (x_hat * gamma);
        atomicAdd(&dgamma[col], d_y * x_hat);
    }
    float reduced = v5_sum3_block_reduce_sum<BLOCK_SIZE>(local_sum);
    __shared__ float mean;
    if (tid == 0) {
        mean = reduced / static_cast<float>(K);
    }
    __syncthreads();

    float mean_value = mean;
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = tid + item * BLOCK_SIZE;
        float d_y = __bfloat162float(dy_cache[item]);
        float x = __bfloat162float(x_cache[item]);
        float gamma = __bfloat162float(norm_weight[col]);
        float dx = inv * (d_y * gamma - x * inv * mean_value);
        row_dx[col] = __hadd(
            __float2bfloat16(dx), row_residual[col]
        );
    }
}

template <int ROW_TILE = 256>
__global__ void v5_sum3_rmsnorm_bwd_dgamma_partial_kernel(
    const __nv_bfloat16* __restrict__ d_sum,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ inv_rms,
    float* __restrict__ partials,
    int M,
    int K
) {
    int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= K) {
        return;
    }
    int row_start = static_cast<int>(blockIdx.y) * ROW_TILE;
    int row_end = min(row_start + ROW_TILE, M);
    float partial = 0.0f;
    for (int row = row_start; row < row_end; ++row) {
        int idx = row * K + col;
        partial += __bfloat162float(d_sum[idx])
            * __bfloat162float(input[idx])
            * inv_rms[row];
    }
    partials[static_cast<int>(blockIdx.y) * K + col] = partial;
}

template <typename output_t>
__global__ void v5_sum3_rmsnorm_bwd_dgamma_reduce_kernel(
    const float* __restrict__ partials,
    output_t* __restrict__ dgamma,
    int row_tiles,
    int K
) {
    int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= K) {
        return;
    }
    float sum = 0.0f;
    for (int tile = 0; tile < row_tiles; ++tile) {
        sum += partials[tile * K + col];
    }
    if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
        dgamma[col] = __float2bfloat16_rn(sum);
    } else {
        dgamma[col] = sum;
    }
}

void nvfp4_sum3_rmsnorm_bwd_out_entrypoint(
    const at::Tensor& d0,
    const at::Tensor& d1,
    const at::Tensor& d2,
    const at::Tensor& input,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_rms,
    at::Tensor& d_sum,
    at::Tensor& grad_input,
    at::Tensor& dgamma_partials,
    at::Tensor& dgamma
) {
    TORCH_CHECK(d0.is_cuda() && d0.is_contiguous()
                    && d0.scalar_type() == at::kBFloat16 && d0.dim() == 2,
                "d0 must be contiguous CUDA bf16 [M,K]");
    const int64_t M64 = d0.size(0);
    const int64_t K64 = d0.size(1);
    TORCH_CHECK(M64 > 0 && K64 > 0 && M64 <= INT_MAX && K64 <= INT_MAX
                    && M64 <= INT_MAX / K64,
                "sum3_rmsnorm_bwd_out requires positive int32-indexable M*K");
    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    auto check_bf16_matrix = [&](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous()
                        && tensor.scalar_type() == at::kBFloat16
                        && tensor.sizes() == d0.sizes(),
                    name, " must be contiguous CUDA bf16 [M,K]");
    };
    check_bf16_matrix(d1, "d1");
    check_bf16_matrix(d2, "d2");
    check_bf16_matrix(input, "input");
    check_bf16_matrix(d_sum, "d_sum");
    check_bf16_matrix(grad_input, "grad_input");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous()
                    && norm_weight.scalar_type() == at::kBFloat16
                    && norm_weight.dim() == 1 && norm_weight.numel() == K,
                "norm_weight must be contiguous CUDA bf16 [K]");
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.is_contiguous()
                    && inv_rms.scalar_type() == at::kFloat
                    && inv_rms.numel() == M,
                "inv_rms must be contiguous CUDA fp32 [M]");
    constexpr int row_tile = 256;
    const int row_tiles = (M + row_tile - 1) / row_tile;
    TORCH_CHECK(dgamma_partials.is_cuda() && dgamma_partials.is_contiguous()
                    && dgamma_partials.scalar_type() == at::kFloat
                    && dgamma_partials.dim() == 2
                    && dgamma_partials.size(0) == row_tiles
                    && dgamma_partials.size(1) == K,
                "dgamma_partials must be contiguous CUDA fp32 [ceil(M/256),K]");
    TORCH_CHECK(dgamma.is_cuda() && dgamma.is_contiguous()
                    && (dgamma.scalar_type() == at::kFloat
                        || dgamma.scalar_type() == at::kBFloat16)
                    && dgamma.dim() == 1 && dgamma.numel() == K,
                "dgamma must be contiguous CUDA fp32 or bf16 [K]");
    kittens::py::device_check(d0, d1, d2, input, norm_weight, inv_rms,
                              d_sum, grad_input, dgamma_partials, dgamma);

    const at::Tensor* outputs[] = {
        &d_sum, &grad_input, &dgamma_partials, &dgamma,
    };
    const at::Tensor* inputs[] = {
        &d0, &d1, &d2, &input, &norm_weight, &inv_rms,
    };
    for (const at::Tensor* output : outputs) {
        for (const at::Tensor* read : inputs) {
            at::assert_no_overlap(*output, *read);
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            at::assert_no_overlap(*outputs[i], *outputs[j]);
        }
    }

    const c10::cuda::CUDAGuard device_guard(d0.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int dx_threads = 512;
    if (K == 4096) {
        v5_sum3_rmsnorm_bwd_dx_k4096_kernel<dx_threads, 8>
            <<<M, dx_threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(d0.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(d1.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(d2.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
                inv_rms.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(d_sum.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
                M);
    } else {
        v5_sum3_rmsnorm_bwd_dx_kernel<dx_threads>
            <<<M, dx_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(d0.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(d1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(d2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            inv_rms.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(d_sum.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
            M,
            K);
    }
    constexpr int dgamma_threads = 256;
    const int col_tiles = (K + dgamma_threads - 1) / dgamma_threads;
    v5_sum3_rmsnorm_bwd_dgamma_partial_kernel<row_tile>
        <<<dim3(col_tiles, row_tiles), dgamma_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(d_sum.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            inv_rms.data_ptr<float>(),
            dgamma_partials.data_ptr<float>(),
            M,
            K);
    if (dgamma.scalar_type() == at::kBFloat16) {
        v5_sum3_rmsnorm_bwd_dgamma_reduce_kernel
            <<<col_tiles, dgamma_threads, 0, stream>>>(
                dgamma_partials.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    dgamma.data_ptr<at::BFloat16>()),
                row_tiles,
                K);
    } else {
        v5_sum3_rmsnorm_bwd_dgamma_reduce_kernel
            <<<col_tiles, dgamma_threads, 0, stream>>>(
                dgamma_partials.data_ptr<float>(),
                dgamma.data_ptr<float>(),
                row_tiles,
                K);
    }
    CUDACHECK(cudaGetLastError());
}

void nvfp4_rmsnorm_bwd_residual_out_entrypoint(
    const at::Tensor& d_normed,
    const at::Tensor& input,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_rms,
    const at::Tensor& residual_grad,
    at::Tensor& grad_input,
    at::Tensor& dgamma
) {
    TORCH_CHECK(d_normed.is_cuda() && d_normed.is_contiguous()
                    && d_normed.scalar_type() == at::kBFloat16
                    && d_normed.dim() == 2,
                "d_normed must be contiguous CUDA bf16 [M,4096]");
    const int64_t M64 = d_normed.size(0);
    const int64_t K64 = d_normed.size(1);
    TORCH_CHECK(M64 > 0 && K64 == 4096 && M64 <= INT_MAX / K64,
                "rmsnorm_bwd_residual_out requires [M,4096] with int32 M");
    const int M = static_cast<int>(M64);
    constexpr int K = 4096;
    auto check_bf16_matrix = [&](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous()
                        && tensor.scalar_type() == at::kBFloat16
                        && tensor.sizes() == d_normed.sizes(),
                    name, " must be contiguous CUDA bf16 [M,4096]");
    };
    check_bf16_matrix(input, "input");
    check_bf16_matrix(residual_grad, "residual_grad");
    check_bf16_matrix(grad_input, "grad_input");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous()
                    && norm_weight.scalar_type() == at::kBFloat16
                    && norm_weight.dim() == 1 && norm_weight.numel() == K,
                "norm_weight must be contiguous CUDA bf16 [4096]");
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.is_contiguous()
                    && inv_rms.scalar_type() == at::kFloat
                    && inv_rms.dim() == 1 && inv_rms.numel() == M,
                "inv_rms must be contiguous CUDA fp32 [M]");
    TORCH_CHECK(dgamma.is_cuda() && dgamma.is_contiguous()
                    && dgamma.scalar_type() == at::kFloat
                    && dgamma.dim() == 1 && dgamma.numel() == K,
                "dgamma must be contiguous CUDA fp32 [4096]");
    kittens::py::device_check(
        d_normed, input, norm_weight, inv_rms, residual_grad,
        grad_input, dgamma);

    const at::Tensor* outputs[] = {
        &grad_input, &dgamma,
    };
    const at::Tensor* inputs[] = {
        &d_normed, &input, &norm_weight, &inv_rms, &residual_grad,
    };
    for (const at::Tensor* output : outputs) {
        for (const at::Tensor* read : inputs) {
            at::assert_no_overlap(*output, *read);
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 2; ++j) {
            at::assert_no_overlap(*outputs[i], *outputs[j]);
        }
    }

    const c10::cuda::CUDAGuard device_guard(d_normed.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    CUDACHECK(cudaMemsetAsync(dgamma.data_ptr<float>(), 0,
                              K * sizeof(float), stream));
    auto d_normed_ptr = reinterpret_cast<const __nv_bfloat16*>(
        d_normed.data_ptr<at::BFloat16>());
    auto input_ptr = reinterpret_cast<const __nv_bfloat16*>(
        input.data_ptr<at::BFloat16>());
    auto norm_weight_ptr = reinterpret_cast<const __nv_bfloat16*>(
        norm_weight.data_ptr<at::BFloat16>());
    auto residual_ptr = reinterpret_cast<const __nv_bfloat16*>(
        residual_grad.data_ptr<at::BFloat16>());
    auto grad_input_ptr = reinterpret_cast<__nv_bfloat16*>(
        grad_input.data_ptr<at::BFloat16>());
    constexpr int dx_threads = 256;
    v5_rmsnorm_bwd_residual_cached_dx_k4096_kernel<dx_threads, 16>
        <<<M, dx_threads, 0, stream>>>(
            d_normed_ptr, input_ptr, norm_weight_ptr,
            inv_rms.data_ptr<float>(), residual_ptr, grad_input_ptr,
            dgamma.data_ptr<float>(), M);
    CUDACHECK(cudaGetLastError());
}

void nvfp4_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    int K = B.size(1) * 2;
    int N_out = D.size(1);
    if (K <= 2048 && N_out <= 4096) {
        // Dgrad + small-N shapes: sweep-optimized config
        // Nb=256, LOAD_PIPE=5, EPI_PIPE=8, SG=4, OVL=false
        // 1.33x faster than Nb=128 on Wo dgrad, 1.49x on small-M dgrad
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = 0,
            .k_dim = 0,
            .use_split_D = false,
            .b_sg_per_tile = nullptr,
            .silu_dim = 0
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else if (K <= 2048) {
        // Sweep-optimized: Nb=512 beats Nb=1024 by 22% for large-N (e.g. N=6144)
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = 0,
            .k_dim = 0,
            .use_split_D = false,
            .b_sg_per_tile = nullptr
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else {
        using C = nvfp4_gemm::config<256, 4, 8, 12, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = 0,
            .k_dim = 0,
            .use_split_D = false,
            .b_sg_per_tile = nullptr
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    }
}

// ================================================================
// Non-PDL standard GEMM: USE_PDL=false, CLUSTER_SIZE=1.
// Safe for CUDA graph capture and replay.
// Regular nvfp4_gemm uses PDL + CLUSTER_SIZE=2 which do not replay
// correctly inside CUDA graphs.
// ================================================================
void nvfp4_gemm_nopdl_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    // USE_PDL=false (8th arg), CLUSTER_SIZE=2 (9th arg, default — kernel requires cluster pairs)
    using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .use_split_D = false,
        .b_sg_per_tile = nullptr,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

// Grouped GEMM: concatenated weights with per-tile B_sc_global
// B_sg_per_tile: [num_col_tiles] float, pre-computed on GPU by Python.
//   Each entry has the B_sg value for that column tile's group.
void nvfp4_grouped_gemm_entrypoint(
    const at::Tensor &A,              // [M, K/2] fp4
    const at::Tensor &A_sc,           // [M/16, K/16] fp8
    const at::Tensor &A_sc_global,    // [1] float
    const at::Tensor &B,              // [N_total, K/2] fp4 (concatenated weights)
    const at::Tensor &B_sc,           // [N_total/16, K/16] fp8
    const at::Tensor &B_sg_per_tile,  // [num_col_tiles] float — pre-computed per-tile B_sg (on GPU)
    at::Tensor &D,                    // [M, N_total] or [M, Nq] bf16
    std::optional<at::Tensor> D_K_opt = std::nullopt, // Optional K output
    std::optional<at::Tensor> D_V_opt = std::nullopt, // Optional V output
    int silu_dim = 0                  // Apply SiLU to output columns [0, silu_dim). 0 = disabled.
) {
    static thread_local at::Tensor dummy_bsg;
    if (!dummy_bsg.defined()) {
        dummy_bsg = at::zeros({1}, at::dtype(at::kFloat).device(at::kCUDA));
    }
    bool use_split_D = D_K_opt.has_value();

    int K = B.size(1) * 2;
    if (K <= 2048) {
        // Sweep-optimized: Nb=512 beats Nb=1024 by 22% for QKV fwd (K=2048, N=6144)
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                       : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                      : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
            .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
            .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
            .v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0,
            .use_split_D = use_split_D,
            .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
            .b_sg_stride = 1,
            .silu_dim = silu_dim
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else {
        // Sweep-optimized: SG=4 OVL=false beats SG=12 OVL=true by 5-6% for wgrad at M=32K-65K
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                       : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                      : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
            .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
            .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
            .v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0,
            .use_split_D = use_split_D,
            .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
            .b_sg_stride = 1,
            .silu_dim = silu_dim
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    }
}


// ================================================================
// Non-PDL Grouped GEMM: same as above but with USE_PDL=false.
// Safe for multi-stream (side stream) and CUDA graph capture.
// PDL (Programmatic Dependent Launch) requires per-stream arrive/wait
// which deadlocks on side streams without prior PDL kernel launches.
// ================================================================
void nvfp4_grouped_gemm_nopdl_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    static thread_local at::Tensor dummy_bsg;
    if (!dummy_bsg.defined()) {
        dummy_bsg = at::zeros({1}, at::dtype(at::kFloat).device(at::kCUDA));
    }
    bool use_split_D = D_K_opt.has_value();

    // USE_PDL=false, CLUSTER_SIZE=2 (default) — safe for multi-stream
    // USE_PDL=false (8th arg), CLUSTER_SIZE=2 (9th arg — kernel requires cluster pairs)
    using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                   : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                  : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0,
        .use_split_D = use_split_D,
        .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
        .b_sg_stride = 1,
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

static bool nvfp4_is_power_of_two_i64(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static void nvfp4_check_rope_live64_tensor(
    const at::Tensor &t,
    const char *name,
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

static void nvfp4_check_rope_packed_tensor(
    const at::Tensor &t,
    const char *name,
    int64_t seq_len,
    int64_t pair_dim
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == seq_len, name, " seq_len mismatch");
    TORCH_CHECK(t.size(1) == pair_dim, name, " pair dim mismatch");
    TORCH_CHECK(t.size(2) == 2, name, " last dim must equal 2");
}

static void nvfp4_check_rope_live64_qkv_args(
    const at::Tensor &D,
    const at::Tensor &D_K,
    const at::Tensor &D_V,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len
) {
    TORCH_CHECK(rope_seq_len > 0, "rope_seq_len must be positive");
    TORCH_CHECK(nvfp4_is_power_of_two_i64(rope_seq_len), "rope_seq_len must be a power of two");
    TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kBFloat16, "D must be contiguous CUDA bf16");
    TORCH_CHECK(D_K.is_cuda() && D_K.is_contiguous() && D_K.scalar_type() == at::kBFloat16, "D_K must be contiguous CUDA bf16");
    TORCH_CHECK(D_V.is_cuda() && D_V.is_contiguous() && D_V.scalar_type() == at::kBFloat16, "D_V must be contiguous CUDA bf16");
    TORCH_CHECK(D.dim() == 2 && D_K.dim() == 2 && D_V.dim() == 2, "Q/K/V outputs must be 2D");
    TORCH_CHECK(D.size(0) == D_K.size(0) && D.size(0) == D_V.size(0), "Q/K/V output rows must match");
    TORCH_CHECK(D.size(0) % rope_seq_len == 0, "Q output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D_K.size(0) % rope_seq_len == 0, "K output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D.size(1) % 64 == 0, "Q output cols must be divisible by 64");
    TORCH_CHECK(D_K.size(1) % 64 == 0, "K output cols must be divisible by 64");
    TORCH_CHECK(D_V.size(1) % 128 == 0, "V output cols must be divisible by 128");
    nvfp4_check_rope_live64_tensor(rope_cs, "rope_cs", rope_seq_len);
    kittens::py::device_check(D, D_K, D_V, rope_cs);
}

static void nvfp4_check_rope_packed_args(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    const at::Tensor &D,
    const at::Tensor *D_K,
    const at::Tensor *D_V,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim,
    int64_t q_dim,
    int64_t k_dim
) {
    const auto check_fp4 = [](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.dim() == 2, name, " must be rank 2");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat4_e2m1fn_x2,
                    name, " must be fp4x2");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 0xF) == 0,
                    name, " must be 16-byte aligned");
    };
    check_fp4(A, "A");
    check_fp4(B, "B");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share packed K");

    const int64_t M = A.size(0);
    const int64_t N = B.size(0);
    const int64_t K = A.size(1) * 2;
    TORCH_CHECK(M > 0 && N > 0 && K > 0 &&
                M % 256 == 0 && N % 256 == 0 && K % 256 == 0,
                "packed RoPE M,N,K must be positive multiples of 256");

    const auto check_sc = [K](
        const at::Tensor &tensor, const char *name, int64_t rows
    ) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat8_e4m3fn,
                    name, " must be fp8 e4m3");
        const bool legacy_2d =
            tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == K / 16;
        const bool prepared_3d =
            tensor.dim() == 3 && tensor.size(0) == rows / 128 &&
            tensor.size(1) == K / 64 && tensor.size(2) == 512;
        TORCH_CHECK(legacy_2d || prepared_3d,
                    name, " must be legacy [rows,K/16] or prepared [rows/128,K/64,512]");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 0xF) == 0,
                    name, " must be 16-byte aligned");
    };
    check_sc(A_sc, "A_sc", M);
    check_sc(B_sc, "B_sc", N);
    TORCH_CHECK(A_sc_global.is_cuda() && A_sc_global.is_contiguous() &&
                A_sc_global.scalar_type() == at::kFloat && A_sc_global.numel() == 1,
                "A_sc_global must be contiguous CUDA float32 [1]");
    TORCH_CHECK(B_sg_per_tile.is_cuda() && B_sg_per_tile.is_contiguous() &&
                B_sg_per_tile.scalar_type() == at::kFloat &&
                B_sg_per_tile.dim() == 1 && B_sg_per_tile.numel() == N / 256,
                "B_sg_per_tile must be contiguous CUDA float32 [N/256]");
    const auto check_output = [M](
        const at::Tensor &tensor, const char *name, int64_t cols
    ) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.scalar_type() == at::kBFloat16 &&
                    tensor.size(0) == M && tensor.size(1) == cols,
                    name, " must be contiguous CUDA bfloat16 [M,", cols, "]");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 0xF) == 0,
                    name, " must be 16-byte aligned");
    };
    const bool split_outputs = D_K != nullptr || D_V != nullptr;
    TORCH_CHECK((D_K == nullptr) == (D_V == nullptr),
                "packed RoPE split outputs must provide both K and V");
    check_output(D, "D", split_outputs ? q_dim : N);
    if (split_outputs) {
        check_output(*D_K, "D_K", k_dim);
        check_output(*D_V, "D_V", N - q_dim - k_dim);
    }

    TORCH_CHECK(rope_seq_len > 0 && M % rope_seq_len == 0,
                "rope_seq_len must be positive and divide M");
    TORCH_CHECK(nvfp4_is_power_of_two_i64(rope_seq_len),
                "rope_seq_len must be a power of two");
    TORCH_CHECK(rope_head_dim > 0 && rope_head_dim % 2 == 0,
                "rope_head_dim must be positive and even");
    TORCH_CHECK(nvfp4_is_power_of_two_i64(rope_head_dim),
                "rope_head_dim must be a power of two");
    TORCH_CHECK(rope_rotary_dim == rope_head_dim,
                "packed RoPE requires rotary_dim == head_dim");
    TORCH_CHECK(q_dim > 0 && k_dim > 0 && q_dim + k_dim < N &&
                q_dim % rope_head_dim == 0 && k_dim % rope_head_dim == 0,
                "Q/K dimensions must be positive head-aligned prefixes with a V suffix");
    TORCH_CHECK((N - q_dim - k_dim) % 128 == 0,
                "V suffix must be divisible by 128");
    constexpr int kEpilogueCols = 32;
    TORCH_CHECK(q_dim % kEpilogueCols == 0 &&
                (q_dim + k_dim) % kEpilogueCols == 0,
                "Q/K epilogue boundaries must be aligned to 32 columns");
    TORCH_CHECK(q_dim % 256 == 0 && k_dim % 256 == 0 &&
                (N - q_dim - k_dim) % 256 == 0,
                "Q/K/V packed dimensions must each be divisible by 256");
    TORCH_CHECK(rope_cs.is_cuda() && rope_cs.is_contiguous() &&
                rope_cs.scalar_type() == at::kFloat && rope_cs.dim() == 3 &&
                rope_cs.size(0) == rope_seq_len &&
                rope_cs.size(1) == rope_rotary_dim / 2 && rope_cs.size(2) == 2,
                "rope_cs must be contiguous CUDA float32 [seq_len,rotary_dim/2,2]");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(rope_cs.data_ptr()) & 0x7) == 0,
                "rope_cs must be 8-byte aligned");

    if (split_outputs) {
        kittens::py::device_check(
            A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile,
            D, *D_K, *D_V, rope_cs);
    } else {
        kittens::py::device_check(
            A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile, D, rope_cs);
    }
    const auto check_reads = [&](const at::Tensor &output) {
        c3_row_scale::check_output_no_overlap(output, A, "A");
        c3_row_scale::check_output_no_overlap(output, A_sc, "A_sc");
        c3_row_scale::check_output_no_overlap(output, A_sc_global, "A_sc_global");
        c3_row_scale::check_output_no_overlap(output, B, "B");
        c3_row_scale::check_output_no_overlap(output, B_sc, "B_sc");
        c3_row_scale::check_output_no_overlap(output, B_sg_per_tile, "B_sg_per_tile");
        c3_row_scale::check_output_no_overlap(output, rope_cs, "rope_cs");
    };
    check_reads(D);
    if (split_outputs) {
        check_reads(*D_K);
        check_reads(*D_V);
        c3_row_scale::check_output_no_overlap(D, *D_K, "D_K");
        c3_row_scale::check_output_no_overlap(D, *D_V, "D_V");
        c3_row_scale::check_output_no_overlap(*D_K, *D_V, "D_V");
    }
}

template <bool INVERSE>
__global__ void nvfp4_rope_packed_qk_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    const float2* rope_cs,
    int64_t rows,
    int64_t q_dim,
    int64_t k_dim,
    int pair_dim,
    int seq_mask
) {
    const int64_t q_pairs_per_row = q_dim / 2;
    const int64_t k_pairs_per_row = k_dim / 2;
    const int pair_mask = pair_dim - 1;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const int64_t rope_row = (row & seq_mask) * pair_dim;
        const int64_t q_row = row * q_dim;
        for (int64_t pair_col = threadIdx.x; pair_col < q_pairs_per_row;
             pair_col += blockDim.x) {
            const int64_t elem = q_row + pair_col * 2;
            const float x = __bfloat162float(q[elem]);
            const float y = __bfloat162float(q[elem + 1]);
            const float2 cs = rope_cs[rope_row + (pair_col & pair_mask)];
            if constexpr (INVERSE) {
                q_out[elem] = __float2bfloat16_rn(__fmaf_rn(y, cs.y, x * cs.x));
                q_out[elem + 1] = __float2bfloat16_rn(__fmaf_rn(-x, cs.y, y * cs.x));
            } else {
                q_out[elem] = __float2bfloat16_rn(__fmaf_rn(-y, cs.y, x * cs.x));
                q_out[elem + 1] = __float2bfloat16_rn(__fmaf_rn(x, cs.y, y * cs.x));
            }
        }
        const int64_t k_row = row * k_dim;
        for (int64_t pair_col = threadIdx.x; pair_col < k_pairs_per_row;
             pair_col += blockDim.x) {
            const int64_t elem = k_row + pair_col * 2;
            const float x = __bfloat162float(k[elem]);
            const float y = __bfloat162float(k[elem + 1]);
            const float2 cs = rope_cs[rope_row + (pair_col & pair_mask)];
            if constexpr (INVERSE) {
                k_out[elem] = __float2bfloat16_rn(__fmaf_rn(y, cs.y, x * cs.x));
                k_out[elem + 1] = __float2bfloat16_rn(__fmaf_rn(-x, cs.y, y * cs.x));
            } else {
                k_out[elem] = __float2bfloat16_rn(__fmaf_rn(-y, cs.y, x * cs.x));
                k_out[elem + 1] = __float2bfloat16_rn(__fmaf_rn(x, cs.y, y * cs.x));
            }
        }
    }
}

template <bool INVERSE>
void nvfp4_rope_packed_qk_entrypoint(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    at::Tensor &q_out,
    at::Tensor &k_out
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && q_out.is_cuda() && k_out.is_cuda(),
                "Q/K inputs and outputs must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                q_out.scalar_type() == at::kBFloat16 && k_out.scalar_type() == at::kBFloat16,
                "Q/K inputs and outputs must be bf16");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && q_out.dim() == 2 && k_out.dim() == 2,
                "Q/K inputs and outputs must be 2D");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && q_out.is_contiguous() && k_out.is_contiguous(),
                "Q/K inputs and outputs must be contiguous");
    TORCH_CHECK(q.sizes() == q_out.sizes() && k.sizes() == k_out.sizes(),
                "Q/K output shapes must match their inputs");
    TORCH_CHECK(rope_seq_len > 0 && nvfp4_is_power_of_two_i64(rope_seq_len),
                "rope_seq_len must be a positive power of two");
    TORCH_CHECK(rope_head_dim > 0 && rope_head_dim % 2 == 0 &&
                nvfp4_is_power_of_two_i64(rope_head_dim),
                "rope_head_dim must be a positive even power of two");
    TORCH_CHECK(q.size(0) > 0 && q.size(0) == k.size(0) &&
                q.size(0) % rope_seq_len == 0,
                "Q/K rows must be nonzero, match, and be divisible by rope_seq_len");
    TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0 &&
                q.size(1) % rope_head_dim == 0 && k.size(1) % rope_head_dim == 0,
                "Q/K columns must be positive multiples of rope_head_dim");
    nvfp4_check_rope_packed_tensor(
        rope_cs, "rope_cs", rope_seq_len, rope_head_dim / 2);
    kittens::py::device_check(q, k, rope_cs, q_out, k_out);
    c3_row_scale::check_output_no_overlap(q_out, q, "q");
    c3_row_scale::check_output_no_overlap(q_out, k, "k");
    c3_row_scale::check_output_no_overlap(q_out, rope_cs, "rope_cs");
    c3_row_scale::check_output_no_overlap(q_out, k_out, "k_out");
    c3_row_scale::check_output_no_overlap(k_out, q, "q");
    c3_row_scale::check_output_no_overlap(k_out, k, "k");
    c3_row_scale::check_output_no_overlap(k_out, rope_cs, "rope_cs");

    const c10::cuda::CUDAGuard device_guard(q.device());
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(q.size(0), 65535));
    auto stream = at::cuda::getCurrentCUDAStream();
    nvfp4_rope_packed_qk_kernel<INVERSE><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const float2*>(rope_cs.data_ptr<float>()),
        q.size(0), q.size(1), k.size(1),
        static_cast<int>(rope_head_dim / 2), static_cast<int>(rope_seq_len - 1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_forward_rope_packed_qk_entrypoint(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    at::Tensor &q_out,
    at::Tensor &k_out
) {
    nvfp4_rope_packed_qk_entrypoint<false>(
        q, k, rope_cs, rope_seq_len, rope_head_dim, q_out, k_out);
}

void nvfp4_inverse_rope_packed_qk_entrypoint(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    at::Tensor &q_out,
    at::Tensor &k_out
) {
    nvfp4_rope_packed_qk_entrypoint<true>(
        q, k, rope_cs, rope_seq_len, rope_head_dim, q_out, k_out);
}

template <typename C>
static void run_grouped_gemm_rope_live64_with_config(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    at::Tensor &D_K,
    at::Tensor &D_V,
    const nvfp4_rope_epilogue::rope_live64_desc &rope_live64,
    int silu_dim = 0
) {
    static thread_local at::Tensor dummy_bsg;
    if (!dummy_bsg.defined()) {
        dummy_bsg = at::zeros({1}, at::dtype(at::kFloat).device(at::kCUDA));
    }

    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D_K),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D_V),
        .q_dim = static_cast<int>(D.size(1)),
        .k_dim = static_cast<int>(D_K.size(1)),
        .v_dim = static_cast<int>(D_V.size(1)),
        .use_split_D = true,
        .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
        .b_sg_stride = 1,
        .silu_dim = silu_dim,
        .rope_live64 = rope_live64
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_grouped_gemm_rope_packed_cat_with_config(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    int q_dim,
    int k_dim,
    const nvfp4_rope_epilogue::rope_live64_desc &rope_packed
) {
    static thread_local at::Tensor dummy_bsg;
    if (!dummy_bsg.defined()) {
        dummy_bsg = at::zeros({1}, at::dtype(at::kFloat).device(at::kCUDA));
    }

    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = q_dim,
        .k_dim = k_dim,
        .v_dim = static_cast<int>(D.size(1)) - q_dim - k_dim,
        .use_split_D = false,
        .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
        .b_sg_stride = 1,
        .silu_dim = 0,
        .rope_live64 = rope_packed
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_grouped_gemm_rope_packed_split_with_config(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    at::Tensor &D_K,
    at::Tensor &D_V,
    const nvfp4_rope_epilogue::rope_live64_desc &rope_packed
) {
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        // b_sg_per_tile is the active B-scale contract for this route.
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(A_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D_K),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D_V),
        .q_dim = static_cast<int>(D.size(1)),
        .k_dim = static_cast<int>(D_K.size(1)),
        .v_dim = static_cast<int>(D_V.size(1)),
        .use_split_D = true,
        .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
        .b_sg_stride = 1,
        .silu_dim = 0,
        .rope_live64 = rope_packed
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

void nvfp4_grouped_gemm_rope_live64_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    at::Tensor &D_K,
    at::Tensor &D_V,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int silu_dim = 0
) {
    TORCH_CHECK(D.size(0) == A.size(0), "Q output rows must match A rows");
    TORCH_CHECK(D_K.size(0) == A.size(0), "K output rows must match A rows");
    TORCH_CHECK(D_V.size(0) == A.size(0), "V output rows must match A rows");
    TORCH_CHECK(D.size(1) + D_K.size(1) + D_V.size(1) == B.size(0),
                "Q/K/V output columns must sum to B rows");
    nvfp4_check_rope_live64_qkv_args(D, D_K, D_V, rope_cs, rope_seq_len);

    nvfp4_rope_epilogue::rope_live64_desc rope_live64 {
        .cs = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>()),
        .seq_len = static_cast<int>(rope_seq_len),
        .seq_mask = static_cast<int>(rope_seq_len - 1),
    };

    using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 256, true>;
    run_grouped_gemm_rope_live64_with_config<C>(
        A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile,
        D, D_K, D_V, rope_live64, silu_dim);
}

void nvfp4_grouped_gemm_rope_packed_cat_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim,
    int64_t q_dim,
    int64_t k_dim
) {
    nvfp4_check_rope_packed_args(
        A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile, D, nullptr, nullptr, rope_cs,
        rope_seq_len, rope_head_dim, rope_rotary_dim, q_dim, k_dim);
    const c10::cuda::CUDAGuard device_guard(A.device());

    nvfp4_rope_epilogue::rope_live64_desc rope_packed {
        .cs = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>()),
        .seq_len = static_cast<int>(rope_seq_len),
        .seq_mask = static_cast<int>(rope_seq_len - 1),
        .pair_dim = static_cast<int>(rope_rotary_dim / 2),
        .head_mask = static_cast<int>(rope_head_dim - 1),
        .round_input_bf16 = true,
    };

    using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2, 256, true>;
    run_grouped_gemm_rope_packed_cat_with_config<C>(
        A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile,
        D, static_cast<int>(q_dim), static_cast<int>(k_dim), rope_packed);
}

void nvfp4_grouped_gemm_rope_packed_split_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    at::Tensor &D_K,
    at::Tensor &D_V,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim
) {
    nvfp4_check_rope_packed_args(
        A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile,
        D, &D_K, &D_V, rope_cs,
        rope_seq_len, rope_head_dim, rope_rotary_dim,
        D.size(1), D_K.size(1));
    const c10::cuda::CUDAGuard device_guard(A.device());

    nvfp4_rope_epilogue::rope_live64_desc rope_packed {
        .cs = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>()),
        .seq_len = static_cast<int>(rope_seq_len),
        .seq_mask = static_cast<int>(rope_seq_len - 1),
        .pair_dim = static_cast<int>(rope_rotary_dim / 2),
        .head_mask = static_cast<int>(rope_head_dim - 1),
        .round_input_bf16 = true,
    };

    using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2, 256, true>;
    run_grouped_gemm_rope_packed_split_with_config<C>(
        A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile,
        D, D_K, D_V, rope_packed);
}

void nvfp4_quantize_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp4x2,
    at::Tensor &A_sc,
    at::Tensor &A_sc_global,
    bool scale_2d
) {
    using C = nvfp4_quantize::quantize_config;
    using G = nvfp4_quantize::globals;

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(A_bf16),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<G::A_sc_global_gl>(A_sc_global)
    };

    // MUST use PyTorch's current stream — bare <<<>>> uses default stream 0
    // which races with PyTorch ops on the current stream, causing NaN Heisenbug.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvfp4_quantize::zero_kernel<<<1, 1, 0, stream>>>(g);
    nvfp4_quantize::absmax_kernel<<<nvfp4_quantize::absmax_config::NUM_BLOCKS, nvfp4_quantize::absmax_config::NUM_THREADS, 0, stream>>>(g);
    nvfp4_quantize::divide_kernel<<<1, 1, 0, stream>>>(g);
    if (scale_2d) kittens::py::launch_kernel<C, G, nvfp4_quantize::quantize_kernel<true>>(g);
    else          kittens::py::launch_kernel<C, G, nvfp4_quantize::quantize_kernel<false>>(g);

    // Fixup FP8 E4M3 NaN in scale tensor: overflow during quantization can produce
    // NaN bit patterns (0x7F) which poison downstream MMA operations.
    {
        int64_t sc_numel = A_sc.numel();
        int threads = 256;
        int blocks = ((sc_numel / 4) + threads - 1) / threads;
        fp8_nan_fixup_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<uint8_t*>(A_sc.data_ptr()), sc_numel);
    }
}

at::Tensor fp32_to_fp4x2_entrypoint(at::Tensor A_fp32) {
    using C = nvfp4_utils::config;
    using G = nvfp4_utils::globals;

    auto options = A_fp32.options().dtype(at::kFloat4_e2m1fn_x2).requires_grad(false);
    at::Tensor A_fp4x2 = at::empty({A_fp32.size(0), A_fp32.size(1) / 2}, options);

    G g {
        .A_fp32 = kittens::py::tensor_to_gl<G::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<C, G, nvfp4_utils::fp32_to_fp4x2_kernel>(g);

    return A_fp4x2;
}

at::Tensor fp4x2_to_fp32_entrypoint(at::Tensor A_fp4x2) {
    using C = nvfp4_utils::config;
    using G = nvfp4_utils::globals;

    auto options = A_fp4x2.options().dtype(at::kFloat).requires_grad(false);
    at::Tensor A_fp32 = at::empty({A_fp4x2.size(0), A_fp4x2.size(1) * 2}, options);

    G g {
        .A_fp32 = kittens::py::tensor_to_gl<G::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<C, G, nvfp4_utils::fp4x2_to_fp32_kernel>(g);

    return A_fp32;
}

// ================================================================
// Config-selectable GEMM for tile tuning sweeps.
// config_id selects from pre-compiled configs below.
// ================================================================
template <typename C>
static void run_gemm_with_config(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0, .k_dim = 0, .use_split_D = false, .b_sg_per_tile = nullptr, .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_gemm_residual_with_config(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &R,
    at::Tensor &D,
    at::Tensor *row_rms_partial = nullptr
) {
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
        .a_sg_per_tile = nullptr,
        .a_sg_stride = 1,
        .b_sg_per_tile = nullptr,
        .b_sg_stride = 1,
        .a_sg_chunk_grid = nullptr,
        .a_sg_chunk_stride = 1,
        .b_sg_chunk_grid = nullptr,
        .b_sg_chunk_stride = 1,
        .silu_dim = 0,
        .row_rms_partial = row_rms_partial == nullptr
            ? nullptr : row_rms_partial->data_ptr<float>(),
        .row_rms_partial_stride = row_rms_partial == nullptr
            ? 0 : static_cast<int>(row_rms_partial->size(1)),
    };
    auto r_gl = kittens::py::tensor_to_gl<typename G::D_gl>(R);
    memcpy(&g.R_tma, &r_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    // CUDA function attributes are device-specific, but launch_kernel's cache
    // is process-global. Set this specialization once on each active device.
    struct AttributeCache {
        std::atomic<int> max_dynamic_smem[64];
        std::mutex locks[64];
        AttributeCache() {
            for (auto &value : max_dynamic_smem) {
                value.store(-1, std::memory_order_relaxed);
            }
        }
    };
    static AttributeCache attribute_cache;
    const int device = D.get_device();
    TORCH_CHECK(device >= 0 && device < 64,
                "v5 residual device index is outside the attribute-cache range");
    const int dynamic_smem = g.dynamic_shared_memory();
    if (dynamic_smem >
        attribute_cache.max_dynamic_smem[device].load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(attribute_cache.locks[device]);
        if (dynamic_smem >
            attribute_cache.max_dynamic_smem[device].load(std::memory_order_relaxed)) {
            CUDACHECK(cudaFuncSetAttribute(
                kittens::py::global_kernel<C, G, nvfp4_gemm::kernel<C>>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dynamic_smem));
            attribute_cache.max_dynamic_smem[device].store(
                dynamic_smem, std::memory_order_release);
        }
    }
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

static void check_nvfp4_residual_gemm_inputs(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    const at::Tensor &R,
    const at::Tensor &D
) {
    const auto check_tma_alignment = [](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK((reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 0xF) == 0,
                    "v5 residual ", name,
                    " data pointer must be 16-byte aligned");
    };
    const auto check_fp4 = [&](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.is_cuda(), "v5 residual ", name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), "v5 residual ", name,
                    " must be contiguous");
        TORCH_CHECK(tensor.dim() == 2, "v5 residual ", name, " must be 2D");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat4_e2m1fn_x2,
                    "v5 residual ", name, " must be fp4x2");
        check_tma_alignment(tensor, name);
    };
    check_fp4(A, "A");
    check_fp4(B, "B");
    TORCH_CHECK(A.size(1) == B.size(1),
                "v5 residual A and B must share packed K");
    const int64_t M = A.size(0);
    const int64_t N = B.size(0);
    const int64_t K = A.size(1) * 2;
    TORCH_CHECK(M > 0 && N > 0 && K > 0 &&
                    M % 256 == 0 && N % 256 == 0 && K % 256 == 0,
                "v5 residual GEMM M, N, and K must be positive multiples of 256");

    const auto check_scale = [&](const at::Tensor &tensor, const char *name,
                                 int64_t rows) {
        TORCH_CHECK(tensor.is_cuda(), "v5 residual ", name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), "v5 residual ", name,
                    " must be contiguous");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat8_e4m3fn,
                    "v5 residual ", name, " must be fp8 e4m3");
        TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == rows / 128 &&
                        tensor.size(1) == K / 64 && tensor.size(2) == 512,
                    "v5 residual ", name,
                    " must be prepared [rows/128,K/64,512]");
        check_tma_alignment(tensor, name);
    };
    check_scale(A_sc, "A_sc", M);
    check_scale(B_sc, "B_sc", N);

    const auto check_global = [&](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.is_cuda(), "v5 residual ", name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), "v5 residual ", name,
                    " must be contiguous");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat &&
                        tensor.dim() == 1 && tensor.numel() == 1,
                    "v5 residual ", name,
                    " must be contiguous float32 [1]");
    };
    check_global(A_sc_global, "A_sg");
    check_global(B_sc_global, "B_sg");

    const auto check_bf16_output = [&](const at::Tensor &tensor,
                                       const char *name) {
        TORCH_CHECK(tensor.is_cuda(), "v5 residual ", name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), "v5 residual ", name,
                    " must be contiguous");
        TORCH_CHECK(tensor.scalar_type() == at::kBFloat16,
                    "v5 residual ", name, " must be bf16");
        TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == M &&
                        tensor.size(1) == N,
                    "v5 residual ", name, " must have shape [M,N]");
        check_tma_alignment(tensor, name);
    };
    check_bf16_output(R, "R");
    check_bf16_output(D, "D");

    const at::Tensor *inputs[] = {
        &A, &A_sc, &A_sc_global, &B, &B_sc, &B_sc_global, &R,
    };
    for (const at::Tensor *input : inputs) {
        at::assert_no_overlap(D, *input);
    }
}

void nvfp4_gemm_residual_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    const at::Tensor &R,
    at::Tensor &D
) {
    check_nvfp4_residual_gemm_inputs(
        A, A_sc, A_sc_global, B, B_sc, B_sc_global, R, D);
    kittens::py::device_check(
        A, A_sc, A_sc_global, B, B_sc, B_sc_global, R, D);
    const c10::cuda::CUDAGuard device_guard(A.device());
    const int K = B.size(1) * 2;
    const int N = D.size(1);
    if (K <= 2048 && N <= 4096) {
        using C = nvfp4_gemm::config<
            256, 5, 8, 4, 2, false, 256, true, 2, 256, false, true>;
        run_gemm_residual_with_config<C>(
            A, A_sc, A_sc_global, B, B_sc, B_sc_global, R, D);
    } else if (K <= 2048) {
        using C = nvfp4_gemm::config<
            256, 5, 8, 4, 2, false, 256, true, 2, 256, false, true>;
        run_gemm_residual_with_config<C>(
            A, A_sc, A_sc_global, B, B_sc, B_sc_global, R, D);
    } else {
        using C = nvfp4_gemm::config<
            256, 4, 8, 12, 2, false, 256, true, 2, 256, false, true>;
        run_gemm_residual_with_config<C>(
            A, A_sc, A_sc_global, B, B_sc, B_sc_global, R, D);
    }
}

void nvfp4_gemm_residual_rms_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    const at::Tensor &R,
    at::Tensor &D,
    at::Tensor &row_rms_partial
) {
    check_nvfp4_residual_gemm_inputs(
        A, A_sc, A_sc_global, B, B_sc, B_sc_global, R, D);
    const int64_t M = D.size(0);
    const int64_t N = D.size(1);
    TORCH_CHECK(N <= 4096,
                "v5 residual RMS currently supports hidden_size <= 4096");
    TORCH_CHECK(row_rms_partial.is_cuda() &&
                    row_rms_partial.is_contiguous() &&
                    row_rms_partial.scalar_type() == at::kFloat &&
                    row_rms_partial.dim() == 2 &&
                    row_rms_partial.size(0) == M &&
                    row_rms_partial.size(1) == N / 32,
                "row_rms_partial must be contiguous CUDA float32 [M,N/32]");
    kittens::py::device_check(
        A, A_sc, A_sc_global, B, B_sc, B_sc_global,
        R, D, row_rms_partial);
    at::assert_no_overlap(row_rms_partial, A);
    at::assert_no_overlap(row_rms_partial, A_sc);
    at::assert_no_overlap(row_rms_partial, A_sc_global);
    at::assert_no_overlap(row_rms_partial, B);
    at::assert_no_overlap(row_rms_partial, B_sc);
    at::assert_no_overlap(row_rms_partial, B_sc_global);
    at::assert_no_overlap(row_rms_partial, R);
    at::assert_no_overlap(row_rms_partial, D);

    const c10::cuda::CUDAGuard device_guard(A.device());
    const int K = B.size(1) * 2;
    if (K <= 2048) {
        using C = nvfp4_gemm::config<
            256, 5, 8, 4, 2, false, 256, true, 2, 256,
            false, true, false, true>;
        run_gemm_residual_with_config<C>(
            A, A_sc, A_sc_global, B, B_sc, B_sc_global,
            R, D, &row_rms_partial);
    } else {
        using C = nvfp4_gemm::config<
            256, 4, 8, 12, 2, false, 256, true, 2, 256,
            false, true, false, true>;
        run_gemm_residual_with_config<C>(
            A, A_sc, A_sc_global, B, B_sc, B_sc_global,
            R, D, &row_rms_partial);
    }
}

void nvfp4_row_rms_reduce_entrypoint(
    const at::Tensor &row_rms_partial,
    at::Tensor &inv_rms,
    int64_t hidden_size,
    double epsilon
) {
    const c10::cuda::CUDAGuard device_guard(row_rms_partial.device());
    c1_rms_reduce::row_rms_reduce_entrypoint(
        row_rms_partial, inv_rms, hidden_size, epsilon);
}

template <typename C>
static void run_grouped_gemm_with_config(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    static thread_local at::Tensor dummy_bsg;
    if (!dummy_bsg.defined()) {
        dummy_bsg = at::zeros({1}, at::dtype(at::kFloat).device(at::kCUDA));
    }
    const bool use_split_D = D_K_opt.has_value();
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                   : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                  : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0,
        .use_split_D = use_split_D,
        .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
        .b_sg_stride = 1,
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

#define NVFP4_GEMM_CONFIG_CASES(X) \
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

#define NVFP4_GEMM_DISPATCH_CASE(ID, NB, LP, EP, SG, DT, OVERLAP) \
    case ID: run_gemm_with_config<nvfp4_gemm::config<NB, LP, EP, SG, DT, OVERLAP>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

#define NVFP4_GEMM_DISPATCH_NOPDL_CASE(ID, NB, LP, EP, SG, DT, OVERLAP) \
    case ID: run_gemm_with_config<nvfp4_gemm::config<NB, LP, EP, SG, DT, OVERLAP, 256, false, 2>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

#define NVFP4_GROUPED_GEMM_DISPATCH_NOPDL_CASE(ID, NB, LP, EP, SG, DT, OVERLAP) \
    case ID: run_grouped_gemm_with_config<nvfp4_gemm::config<NB, LP, EP, SG, DT, OVERLAP, 256, false, 2>>(A, A_sc, A_sc_global, B, B_sc, B_sg_per_tile, D, D_K_opt, D_V_opt, silu_dim); break;

void nvfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D, int config_id
) {
    switch (config_id) {
        NVFP4_GEMM_CONFIG_CASES(NVFP4_GEMM_DISPATCH_CASE)
        // NOTE: Nb=128 + EP=16 is INVALID (D_tile cols = 128/16 = 8 < 16 minimum)
        default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-46)");
    }
}

void nvfp4_gemm_config_nopdl_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D, int config_id
) {
    switch (config_id) {
        NVFP4_GEMM_CONFIG_CASES(NVFP4_GEMM_DISPATCH_NOPDL_CASE)
        default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-46)");
    }
}

void nvfp4_grouped_gemm_config_nopdl_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim,
    int config_id
) {
    switch (config_id) {
        NVFP4_GEMM_CONFIG_CASES(NVFP4_GROUPED_GEMM_DISPATCH_NOPDL_CASE)
        default: TORCH_CHECK(false, "Invalid grouped config_id: ", config_id, " (valid: 0-46)");
    }
}

#undef NVFP4_GROUPED_GEMM_DISPATCH_NOPDL_CASE
#undef NVFP4_GEMM_DISPATCH_NOPDL_CASE
#undef NVFP4_GEMM_DISPATCH_CASE
#undef NVFP4_GEMM_CONFIG_CASES

// ================================================================
// Batched GEMM entrypoint (z-dim parallel): D_i = A_i × B_i^T
// Each batch writes to a separate output buffer.
// ================================================================
void nvfp4_batched_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    std::vector<at::Tensor> &D_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)D_list.size());

    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    const int K_first = (int)(A_list[0].size(1) * 2);

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);

        for (int i = 0; i < n; ++i) {
            TORCH_CHECK(2 * A_list[i].size(1) % C::Kb == 0,
                        "batched GEMM reduction width must be divisible by ", C::Kb);
            g_host.num_red_blocks[i] = (int)(2 * A_list[i].size(1) / C::Kb);
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }
        kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
    };

    if (K_first <= 2048 && N_out <= 4096) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else if (K_first <= 2048) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    }
}

// ================================================================
// Accumulating Batched GEMM: all batches accumulate into a single D_out.
// Batch 0 stores via TMA, batches 1+ load-add-store with per-tile
// semaphores for ordering. Eliminates the separate sum3 kernel.
// ================================================================
void nvfp4_accum_gemm_v2_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    at::Tensor &D_out,
    at::Tensor &tile_done_buf
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_accum_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_accum_gemm::MAX_BATCHES);

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int K_first = (int)(A_list[0].size(1) * 2);

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_accum_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);

        // Per-tile completion counters (zeroed by caller)
        int num_tiles = g_host.num_row_blocks * 2 * g_host.num_col_blocks;
        TORCH_CHECK(tile_done_buf.numel() >= num_tiles,
                    "tile_done_buf too small: need ", num_tiles, " got ", tile_done_buf.numel());
        g_host.tile_done = tile_done_buf.data_ptr<int>();

        for (int i = 0; i < n; ++i) {
            TORCH_CHECK(2 * A_list[i].size(1) % C::Kb == 0,
                        "accumulating GEMM reduction width must be divisible by ", C::Kb);
            g_host.num_red_blocks[i] = (int)(2 * A_list[i].size(1) / C::Kb);
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }

        // Single shared D output TMA descriptor
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
        memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        kittens::py::launch_kernel<C, G, nvfp4_accum_gemm::kernel<C>>(g_host);
    };

    build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
}

// ================================================================
// Strided batched GEMM: reads A from a full (M, K_total/2) buffer
// with per-batch column offsets, avoiding .contiguous() copies.
// ================================================================

// Create a TMA descriptor for FP4 data with custom row stride.
// This allows reading from a sub-region of a larger contiguous buffer
// without requiring the sub-region itself to be contiguous.
static void create_strided_fp4_tma(
    CUtensorMap *tma_map,
    const void  *global_addr,    // pointer to start of sub-region
    int64_t      rows,           // M (number of rows)
    int64_t      sub_cols,       // N_g/2 (columns in this batch's sub-region)
    int64_t      full_row_stride // N_total/2 * sizeof(fp4x2) = total bytes between rows
) {
    // FP4 TMA: 5D with 128B swizzle (matching st_fp4e2m1_2 tile layout)
    constexpr uint32_t  tma_dim = 5;
    constexpr int64_t   swizzle_elements = 128;  // 128B / sizeof(fp4x2=1) = 128 elements

    uint64_t gmem_shape [5];
    uint64_t gmem_stride[4];
    uint32_t smem_shape [5];
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    gmem_shape[0] = (uint64_t)swizzle_elements;
    gmem_shape[1] = (uint64_t)rows;
    gmem_shape[2] = (uint64_t)(sub_cols + swizzle_elements - 1) / swizzle_elements;
    gmem_shape[3] = 1;
    gmem_shape[4] = 1;

    // KEY: gmem_stride[0] uses full_row_stride, NOT sub_cols
    gmem_stride[0] = (uint64_t)full_row_stride;
    gmem_stride[1] = 128;  // swizzle_bytes
    gmem_stride[2] = (uint64_t)rows * full_row_stride;
    gmem_stride[3] = (uint64_t)rows * full_row_stride;

    // Shared memory tile shape — must match the FP4 tile used by the kernel
    // st_fp4e2m1_2<Mb/2, Kb/2>: e.g. <64, 128> → smem rows=64, cols=128
    // The TMA loads swizzle_elements per inner dim and tile_height per outer dim
    smem_shape[0] = swizzle_elements;
    smem_shape[1] = 64;  // will be overridden below based on actual tile
    smem_shape[2] = 1;   // sub_cols / swizzle_elements;
    smem_shape[3] = 1;
    smem_shape[4] = 1;

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,  // fp4x2 → uint8
        tma_dim,
        const_cast<void*>(global_addr),
        gmem_shape, gmem_stride,
        smem_shape, smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, "Failed to create strided FP4 TMA descriptor");
}

// Create a TMA descriptor for FP8 scale data with custom layout.
// Scales: [ntm, ntk, 512] where ntm = M/128, ntk = N_g/64.
// Scale tiles are stored separately per-group already in our case,
// but for strided A scales, we use the same approach.
static void create_strided_sc_tma(
    CUtensorMap *tma_map,
    const void  *global_addr,
    int64_t      depth,        // ntm = M/128
    int64_t      sub_rows,     // ntk = N_g/64
    int64_t      full_depth_stride  // bytes between depth slices in full buffer
) {
    // Scale TMA: 5D with 128B swizzle (matching st_hf<4, 256> tile layout)
    // The scale layout is (1, depth=ntm, rows=ntk, cols=256) with half type
    constexpr uint32_t  tma_dim = 5;
    constexpr int64_t   swizzle_elements = 64;  // 128B / sizeof(half=2) = 64 elements

    uint64_t gmem_shape [5];
    uint64_t gmem_stride[4];
    uint32_t smem_shape [5];
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    gmem_shape[0] = (uint64_t)swizzle_elements;
    gmem_shape[1] = (uint64_t)depth;  // ntm
    gmem_shape[2] = (uint64_t)(256 + swizzle_elements - 1) / swizzle_elements;  // 256/64 = 4
    gmem_shape[3] = (uint64_t)sub_rows;  // ntk
    gmem_shape[4] = 1;

    gmem_stride[0] = (uint64_t)256 * sizeof(uint16_t);  // stride between ntm slices = 256 * 2 = 512B
    gmem_stride[1] = 128;  // swizzle_bytes
    gmem_stride[2] = (uint64_t)depth * 256 * sizeof(uint16_t);  // stride between ntk slices
    gmem_stride[3] = full_depth_stride;  // stride between batches (if applicable)

    smem_shape[0] = swizzle_elements;
    smem_shape[1] = 4;   // tile height
    smem_shape[2] = 256 / swizzle_elements;  // 4
    smem_shape[3] = 1;
    smem_shape[4] = 1;

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        tma_dim,
        const_cast<void*>(global_addr),
        gmem_shape, gmem_stride,
        smem_shape, smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, "Failed to create strided scale TMA descriptor");
}

// Strided batched GEMM: A FP4 data is read from a single full buffer with per-batch offsets.
// Scales are still passed per-batch (contiguous) since their copies are negligible.
// This avoids the expensive .contiguous() copies for FP4 narrow(1,...) views (0.4ms at M=64K).
void nvfp4_batched_gemm_strided_entrypoint(
    // A FP4: single full row-quantized buffer
    const at::Tensor &A_full,           // [M, K_total/2] fp4x2 (full concatenated row FP4)
    // A scales: per-batch contiguous (unchanged from regular batched GEMM)
    const std::vector<at::Tensor> &A_sc_list,  // per-batch scale tensors
    const std::vector<at::Tensor> &A_sg_list,  // per-batch [1] float32
    const std::vector<int64_t> &A_col_offsets,  // per-batch FP4 column offsets (in fp4x2 elements)
    const std::vector<int64_t> &A_col_widths,   // per-batch FP4 column widths (= N_g/2)
    // B: per-batch (unchanged)
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    std::vector<at::Tensor> &D_list
) {
    const int n = (int)A_sg_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)D_list.size());
    TORCH_CHECK(n == (int)A_col_offsets.size());
    TORCH_CHECK(n == (int)A_col_widths.size());
    TORCH_CHECK(n == (int)A_sc_list.size());

    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    const int64_t K_total_fp4 = A_full.size(1);  // total fp4x2 columns
    const int K_first = (int)(A_col_widths[0] * 2);  // first batch's K in elements

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);

        const uint8_t *a_base = (const uint8_t*)A_full.data_ptr();
        const int64_t a_full_row_stride = K_total_fp4;  // bytes per row (sizeof(fp4x2) = 1)

        for (int i = 0; i < n; ++i) {
            const int64_t fp4_cols = A_col_widths[i];     // N_g/2 in fp4x2 elements
            const int64_t fp4_offset = A_col_offsets[i];  // column offset in fp4x2
            TORCH_CHECK(2 * fp4_cols % C::Kb == 0,
                        "strided batched GEMM reduction width must be divisible by ", C::Kb);
            g_host.num_red_blocks[i] = (int)(2 * fp4_cols / C::Kb);

            // --- A FP4 TMA: strided ---
            // Create TMA for (M, fp4_cols) sub-region from (M, K_total_fp4) buffer
            // by overriding gmem_stride[0] to use full row stride.
            {
                // First create reference TMA via tensor_to_gl on a dummy contiguous tensor
                // to get correct smem_shape and other params, then override.
                // Actually, build from scratch matching create_tensor_map<st_fp4e2m1_2<128,128>, 2>:
                constexpr int64_t swizzle_elements = 128;  // 128B / sizeof(fp4x2=1) = 128
                const void *data_ptr = a_base + fp4_offset;
                
                uint64_t gmem_shape [5] = {
                    (uint64_t)swizzle_elements,                             // dim0: swizzle block
                    (uint64_t)M,                                            // dim1: rows
                    (uint64_t)(fp4_cols + swizzle_elements - 1) / swizzle_elements, // dim2: col-blocks
                    1, 1                                                    // dim3,4: depth, batch
                };
                uint64_t gmem_stride[4] = {
                    (uint64_t)a_full_row_stride,       // stride between rows (KEY: full buffer stride)
                    128,                               // swizzle_bytes
                    (uint64_t)M * a_full_row_stride,   // stride between depth slices
                    (uint64_t)M * a_full_row_stride    // stride between batches
                };
                // smem tile: st_fp4e2m1_2<Mb/2=128, Kb/2=128> → rows=128, cols=128
                uint32_t smem_shape [5] = {(uint32_t)swizzle_elements, (uint32_t)(C::Mb/2), 1, 1, 1};
                uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

                CUresult result = cuTensorMapEncodeTiled(
                    &g_host.A_tma[i],
                    CU_TENSOR_MAP_DATA_TYPE_UINT8,
                    5,
                    const_cast<void*>(data_ptr),
                    gmem_shape, gmem_stride,
                    smem_shape, smem_stride,
                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                );
                TORCH_CHECK(result == CUDA_SUCCESS,
                    "Strided FP4 TMA creation failed for batch ", i);
            }

            // --- A Scale TMA: standard (using tensor_to_gl) ---
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            // --- B and D: standard (unchanged) ---
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }
        kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
    };

    if (K_first <= 2048 && N_out <= 4096) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else if (K_first <= 2048) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    }
}

// ================================================================
// Non-PDL batched GEMM strided: USE_PDL=false for CUDA graph safety.
// ================================================================
void nvfp4_batched_gemm_strided_nopdl_entrypoint(
    const at::Tensor &A_full,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<int64_t> &A_col_offsets,
    const std::vector<int64_t> &A_col_widths,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    std::vector<at::Tensor> &D_list
) {
    const int n = (int)A_sg_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)D_list.size());
    TORCH_CHECK(n == (int)A_col_offsets.size());
    TORCH_CHECK(n == (int)A_col_widths.size());
    TORCH_CHECK(n == (int)A_sc_list.size());

    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const int K_first = (int)(A_col_widths[0] * 2);

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);

        const uint8_t *a_base = (const uint8_t*)A_full.data_ptr();
        const int64_t a_full_row_stride = K_total_fp4;

        for (int i = 0; i < n; ++i) {
            const int64_t fp4_cols = A_col_widths[i];
            const int64_t fp4_offset = A_col_offsets[i];
            TORCH_CHECK(2 * fp4_cols % C::Kb == 0,
                        "strided batched GEMM reduction width must be divisible by ", C::Kb);
            g_host.num_red_blocks[i] = (int)(2 * fp4_cols / C::Kb);

            {
                constexpr int64_t swizzle_elements = 128;
                const void *data_ptr = a_base + fp4_offset;
                uint64_t gmem_shape [5] = {
                    (uint64_t)swizzle_elements, (uint64_t)M,
                    (uint64_t)(fp4_cols + swizzle_elements - 1) / swizzle_elements, 1, 1
                };
                uint64_t gmem_stride[4] = {
                    (uint64_t)a_full_row_stride, 128,
                    (uint64_t)M * a_full_row_stride, (uint64_t)M * a_full_row_stride
                };
                uint32_t smem_shape [5] = {(uint32_t)swizzle_elements, (uint32_t)(C::Mb/2), 1, 1, 1};
                uint32_t smem_stride[5] = {1, 1, 1, 1, 1};
                CUresult result = cuTensorMapEncodeTiled(
                    &g_host.A_tma[i], CU_TENSOR_MAP_DATA_TYPE_UINT8, 5,
                    const_cast<void*>(data_ptr), gmem_shape, gmem_stride,
                    smem_shape, smem_stride,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                );
                TORCH_CHECK(result == CUDA_SUCCESS, "Strided FP4 TMA (nopdl) failed for batch ", i);
            }

            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }
        kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
    };

    // USE_PDL=false (8th arg), CLUSTER_SIZE=2 (9th arg)
    build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>>();
}

// ================================================================
// Fused split dgrad+sum: slices concatenated dY, runs batched GEMM
// (z-dim parallel), then sums the per-split outputs.
// ================================================================

// Forward declaration — defined below
void nvfp4_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    at::Tensor &D_out
);
void nvfp4_split_dgrad_sum(
    // Concatenated row-quantized gradient: dy_cat_q
    const at::Tensor &A_fp4_cat,     // [M, N_total/2] fp4x2 (row-quantized dY)
    const at::Tensor &A_sc_cat,      // [ntm, ntk_total, 512] fp8 (row scales)
    const std::vector<at::Tensor> &A_sg_list,  // [n_splits] each [1] float32 (per-split global scale)
    // Per-split column-quantized weights
    const std::vector<at::Tensor> &B_fp4_list,  // [n_splits] each [K, N_i/2] fp4x2
    const std::vector<at::Tensor> &B_sc_list,   // [n_splits] each [ntm_c, ntk_c_i, 512] fp8
    const at::Tensor &B_sg_cat,                  // [n_splits] float32
    // Split dimensions
    const std::vector<int64_t> &split_dims,      // [q_dim, k_dim, v_dim]
    // Output
    at::Tensor &D_out                            // [M, K] bf16 — accumulated dgrad
) {
    const int n_splits = (int)split_dims.size();
    TORCH_CHECK(n_splits == (int)B_fp4_list.size());
    TORCH_CHECK(n_splits == (int)B_sc_list.size());
    TORCH_CHECK(n_splits == (int)A_sg_list.size());

    const int64_t M = D_out.size(0);
    const int64_t K = D_out.size(1);

    // Slice concatenated A into per-split tensors for batched GEMM
    auto a_fp4_bytes = A_fp4_cat.view(c10::ScalarType::Byte);
    auto a_sc_bytes = A_sc_cat.view(c10::ScalarType::Byte);

    std::vector<at::Tensor> A_list, A_sc_list_v, B_sg_list;
    std::vector<at::Tensor> D_list;

    int64_t fp4_col_offset = 0;
    int64_t sc_col_offset = 0;
    for (int i = 0; i < n_splits; ++i) {
        const int64_t N_i = split_dims[i];
        const int64_t fp4_cols_i = N_i / 2;
        const int64_t sc_tiles_i = N_i / 64;

        A_list.push_back(
            a_fp4_bytes.narrow(1, fp4_col_offset, fp4_cols_i)
                .contiguous().view(at::kFloat4_e2m1fn_x2)
        );
        A_sc_list_v.push_back(
            a_sc_bytes.narrow(1, sc_col_offset, sc_tiles_i)
                .contiguous().view(at::kFloat8_e4m3fn)
        );
        B_sg_list.push_back(B_sg_cat.narrow(0, i, 1));
        D_list.push_back(at::empty({M, K}, D_out.options()));

        fp4_col_offset += fp4_cols_i;
        sc_col_offset += sc_tiles_i;
    }

    // Z-dim parallel batched GEMM: one kernel launch, per-batch outputs
    nvfp4_batched_gemm_entrypoint(
        A_list, A_sc_list_v, A_sg_list,
        B_fp4_list, B_sc_list, B_sg_list,
        D_list
    );

    // Sum per-split outputs into D_out
    D_out.copy_(D_list[0]);
    for (int i = 1; i < n_splits; ++i) {
        D_out.add_(D_list[i]);
    }
}

// ================================================================
// ================================================================
// Simple CUDA kernel to sum N tensors element-wise into output
// ================================================================
__global__ void sum_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
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
    // Vectorized path: 8 bf16 = 128 bits = int4 per iteration
    const int64_t vec_numel = numel / 8;
    const int4* A4 = reinterpret_cast<const int4*>(A);
    const int4* B4 = reinterpret_cast<const int4*>(B);
    const int4* C4 = reinterpret_cast<const int4*>(C);
    int4* out4 = reinterpret_cast<int4*>(out);

    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_numel; i += stride) {
        int4 a = A4[i];
        int4 b = B4[i];
        int4 c = C4[i];

        // Reinterpret as bf162 pairs and add
        __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&a);
        __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(&b);
        __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(&c);

        int4 r;
        __nv_bfloat162* r2 = reinterpret_cast<__nv_bfloat162*>(&r);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            r2[j] = __hadd2(__hadd2(a2[j], b2[j]), c2[j]);
        }
        out4[i] = r;
    }

    // Scalar tail for non-8-aligned remainder
    int64_t tail_start = vec_numel * 8;
    for (int64_t i = tail_start + threadIdx.x; i < numel; i += blockDim.x) {
        if (blockIdx.x == 0)
            out[i] = __hadd(__hadd(A[i], B[i]), C[i]);
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
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(__hadd(A[idx], B[idx]), __hadd(C[idx], D[idx]));
    }
}

// ================================================================
// Batched GEMM with Fused Accumulation — TRUE in-kernel accumulation.
// D_out = sum_i(A_i × B_i^T), accumulated in TMEM registers.
// No intermediate D buffers, no sum3 kernel.
// ================================================================
void nvfp4_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,       // per-batch [M, K/2] fp4x2
    const std::vector<at::Tensor> &A_sc_list,    // per-batch [ntm, ntk, 512] fp8
    const std::vector<at::Tensor> &A_sg_list,    // per-batch [1] float32
    const std::vector<at::Tensor> &B_list,       // per-batch [N, K/2] fp4x2
    const std::vector<at::Tensor> &B_sc_list,    // per-batch [ntm_b, ntk, 512] fp8
    const std::vector<at::Tensor> &B_sg_list,    // per-batch [1] float32
    at::Tensor &D_out                            // accumulated [M, N] bf16
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= 4, "num_batches must be 1..4");
    TORCH_CHECK(D_out.dim() == 2);

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);

    if (n == 1) {
        // Single batch: just run regular GEMM directly into D_out
        std::vector<at::Tensor> D_list = {D_out};
        nvfp4_batched_gemm_entrypoint(A_list, A_sc_list, A_sg_list,
                                       B_list, B_sc_list, B_sg_list, D_list);
        return;
    }

    // Use the real in-kernel accumulation kernel
    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_accum_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);

        for (int i = 0; i < n; ++i) {
            TORCH_CHECK(2 * A_list[i].size(1) % C::Kb == 0,
                        "batched accumulation reduction width must be divisible by ", C::Kb);
            g_host.num_red_blocks[i] = (int)(2 * A_list[i].size(1) / C::Kb);
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }

        // Single output D TMA descriptor (accumulated result)
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
        memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        kittens::py::launch_kernel<C, G, nvfp4_batched_accum_gemm::kernel<C>>(g_host);
    };

    build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
}

// ================================================================
// Standalone fused 3-way bf16 sum: out = A + B + C
// Reads 3 inputs, writes 1 output in a single kernel launch.
// ================================================================
void sum3_bf16_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    at::Tensor &out
) {
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() && out.is_contiguous());
    const int64_t numel = A.numel();
    TORCH_CHECK(numel == B.numel() && numel == C.numel() && numel == out.numel());
    const int threads = 256;
    // vec_numel = numel/8 (each thread handles 8 bf16 via int4 loads)
    const int64_t vec_numel = numel / 8;
    int blocks = (int)((vec_numel + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;  // grid-striding handles the rest
    auto stream = at::cuda::getCurrentCUDAStream();
    sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(C.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        numel);
}
// ================================================================
// Fused Quantize+GEMM: A is bf16, B is pre-quantized NVFP4
// Mode 1: constant SCALE_MAX (USE_CTA_AMAX=false) — fastest
// Mode 2: CTA-level amax (USE_CTA_AMAX=true) — better accuracy
// ================================================================
template <typename C>
static inline void launch_fused_gemm_with_config(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    using G = nvfp4_fused_gemm::globals<C>;

    const int M = A_bf16.size(0);
    const int K = A_bf16.size(1);

    // Allocate scratch buffers for quantized A reuse across col tiles
    auto A_scratch_byte = at::zeros({M, K/2}, at::TensorOptions().dtype(at::kByte).device(A_bf16.device()));
    auto A_scratch = A_scratch_byte.view(at::kFloat4_e2m1fn_x2);
    auto A_sc_scratch = at::zeros({M/128, K/64, 512}, at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(A_bf16.device()));
    // Per-(cta_half, K_stage) completion flags, zeroed before each launch
    // num_cta_halves = M / 128 (each 256-row block has 2 CTA halves)
    // num_red_blocks = K / 256
    const int num_flags = (M / 128) * (K / 256);
    auto A_scratch_ready = at::zeros({num_flags}, at::TensorOptions().dtype(at::kInt).device(A_bf16.device()));

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<typename G::A_bf16_gl>(A_bf16),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1),
            256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .use_split_D = false,
        .b_sg_per_tile = nullptr,
        .silu_dim = 0,
        .A_scratch = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_scratch),
        .A_sc_scratch = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_scratch, 1, M/128, K/64, 256),
        .A_scratch_ready = (uint32_t*)A_scratch_ready.data_ptr<int>(),
    };

    kittens::py::launch_kernel<C, G, nvfp4_fused_gemm::kernel<C>>(g);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    const int K = A_bf16.size(1);
    const int N = D.size(1);

    if (N % 256 == 0 && N <= 2048) {
        if (K <= 256 && N <= 256) {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 4, 4, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>>(
                A_bf16, B, B_sc, B_sc_global, D);
        } else {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 4, 8, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>>(
                A_bf16, B, B_sc, B_sc_global, D);
        }
    } else {
        if (N % 256 != 0) {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 5, 4, 4, 2, false, USE_CTA_AMAX>>(
                A_bf16, B, B_sc, B_sc_global, D);
        } else {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<256, 4, 8, 4, 2, false, USE_CTA_AMAX>>(
                A_bf16, B, B_sc, B_sc_global, D);
        }
    }
}

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
    dispatch_fused_gemm<false>(A_bf16, B, B_sc, B_sc_global, D);
}

// CTA-level amax version — pre-scans A for per-CTA max|A|
void nvfp4_fused_gemm_cta_amax_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    int M = A_bf16.size(0);
    int K = A_bf16.size(1);
    int N = D.size(1);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::kFloat4_e2m1fn_x2, "B must be fp4x2");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");
    TORCH_CHECK(N % 128 == 0, "N must be multiple of 128");
    dispatch_fused_gemm<true>(A_bf16, B, B_sc, B_sc_global, D);
}

// ================================================================
// Persistent Quantize→GEMM: single kernel launch.
// Quantizes A (and optionally B) to FP4 in HBM, then runs GEMM.
// Uses constant SCALE_MAX (no amax scan).
// ================================================================
template <typename KC>
static __cluster_dims__(KC::CLUSTER_SIZE) __launch_bounds__(KC::NUM_THREADS)
__global__ void persistent_gemm_entry(const __grid_constant__ nvfp4_persistent_gemm::globals<KC> g) {
    nvfp4_persistent_gemm::kernel<KC>(g);
}

void nvfp4_persistent_gemm_entrypoint(
    const at::Tensor &A_bf16,       // [M, K] bf16
    const at::Tensor &B_bf16,       // [N, K] bf16
    at::Tensor &D                   // [M, N] bf16 output
) {
    int M = A_bf16.size(0);
    int K = A_bf16.size(1);
    int N = B_bf16.size(0);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B must be bf16");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");
    TORCH_CHECK(N % 256 == 0, "N must be multiple of 256");

    constexpr float SCALE_MAX_DEC = 65504.0f / (6.0f * 448.0f);

    // Allocate FP4 scratch buffers in HBM
    auto A_fp4 = at::empty({M, K/2}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2).device(A_bf16.device()));
    auto A_sc  = at::empty({M/128, K/64, 512}, at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(A_bf16.device()));
    auto A_sg  = at::full({1}, SCALE_MAX_DEC, at::TensorOptions().dtype(at::kFloat).device(A_bf16.device()));
    auto B_fp4 = at::empty({N, K/2}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2).device(A_bf16.device()));
    auto B_sc  = at::empty({N/128, K/64, 512}, at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(A_bf16.device()));
    auto B_sg  = at::full({1}, SCALE_MAX_DEC, at::TensorOptions().dtype(at::kFloat).device(A_bf16.device()));

    // Grid barrier counter
    auto barrier = at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(A_bf16.device()));

    using C = nvfp4_persistent_gemm::config<256, 5, 8, 4, 2, false>;
    using G = nvfp4_persistent_gemm::globals<C>;

    G g {
        // Phase 1: quantize
        .A_bf16  = kittens::py::tensor_to_gl<typename G::Q_bf16_gl>(A_bf16),
        .A_q_fp4 = kittens::py::tensor_to_gl<typename G::Q_fp4_gl>(A_fp4),
        .A_q_sc  = kittens::py::tensor_to_gl<typename G::Q_sc_gl, false>(A_sc, 1, M/128, K/64, 256),
        .B_bf16  = kittens::py::tensor_to_gl<typename G::Q_bf16_gl>(B_bf16),
        .B_q_fp4 = kittens::py::tensor_to_gl<typename G::Q_fp4_gl>(B_fp4),
        .B_q_sc  = kittens::py::tensor_to_gl<typename G::Q_sc_gl, false>(B_sc, 1, N/128, K/64, 256),
        .quantize_b = true,
        // Phase 2: GEMM (same HBM data)
        .A       = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_fp4),
        .A_sc    = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, M/128, K/64, 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sg),
        .B       = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_fp4),
        .B_sc    = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, N/128, K/64, 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sg),
        .D       = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .barrier = barrier.data_ptr<int>(),
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    auto smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(persistent_gemm_entry<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    LaunchConfig<true, false> lc(g.grid(), g.block(), smem, (cudaStream_t)stream, C::CLUSTER_SIZE);
    cudaLaunchKernelEx(lc, persistent_gemm_entry<C>, g);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm_entrypoint);
    m.def("nvfp4_gemm_residual", &nvfp4_gemm_residual_entrypoint,
          "NVFP4 GEMM with fused bf16 residual add",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("R"), pybind11::arg("D"));
    m.def("nvfp4_gemm_residual_rms", &nvfp4_gemm_residual_rms_entrypoint,
          "NVFP4 residual GEMM with exact-row RMS partials",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("R"), pybind11::arg("D"),
          pybind11::arg("row_rms_partial"));
    m.def("nvfp4_row_rms_reduce", &nvfp4_row_rms_reduce_entrypoint,
          "Reduce residual-GEMM row partials to inverse RMS",
          pybind11::arg("row_rms_partial"), pybind11::arg("inv_rms"),
          pybind11::arg("hidden_size"), pybind11::arg("epsilon"));
    m.def("nvfp4_gemm_nopdl", &nvfp4_gemm_nopdl_entrypoint,
          "Non-PDL GEMM for CUDA graph capture (CLUSTER_SIZE=1, USE_PDL=false)");
    m.def("nvfp4_gemm_config", &nvfp4_gemm_config_entrypoint,
          "GEMM with selectable tile config (for sweeping)",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("nvfp4_gemm_config_nopdl", &nvfp4_gemm_config_nopdl_entrypoint,
          "Non-PDL GEMM with selectable tile config (for sweeping side-stream tails)",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("nvfp4_grouped_gemm", &nvfp4_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K_opt") = std::nullopt, pybind11::arg("D_V_opt") = std::nullopt,
          pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_grouped_gemm_rope_live64", &nvfp4_grouped_gemm_rope_live64_entrypoint,
          "Grouped GEMM with split Q/K/V outputs and fused live64 RoPE on Q/K",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K"), pybind11::arg("D_V"),
          pybind11::arg("rope_cs"), pybind11::arg("rope_seq_len"),
          pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_grouped_gemm_rope_packed_cat", &nvfp4_grouped_gemm_rope_packed_cat_entrypoint,
          "Single-output grouped GEMM with packed RoPE on Q/K and an untouched V suffix",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("rope_cs"), pybind11::arg("rope_seq_len"),
          pybind11::arg("rope_head_dim"), pybind11::arg("rope_rotary_dim"),
          pybind11::arg("q_dim"), pybind11::arg("k_dim"));
    m.def("nvfp4_grouped_gemm_rope_packed_split", &nvfp4_grouped_gemm_rope_packed_split_entrypoint,
          "No-PDL grouped GEMM with distinct Q/K/V outputs and packed RoPE on Q/K",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K"), pybind11::arg("D_V"),
          pybind11::arg("rope_cs"), pybind11::arg("rope_seq_len"),
          pybind11::arg("rope_head_dim"), pybind11::arg("rope_rotary_dim"));
    m.def("nvfp4_inverse_rope_packed_qk", &nvfp4_inverse_rope_packed_qk_entrypoint,
          "Apply inverse packed RoPE to contiguous bf16 Q/K in one CUDA launch",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"), pybind11::arg("rope_head_dim"),
          pybind11::arg("q_out"), pybind11::arg("k_out"));
    m.def("nvfp4_forward_rope_packed_qk", &nvfp4_forward_rope_packed_qk_entrypoint,
          "Apply forward packed RoPE to contiguous bf16 Q/K in one CUDA launch",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"), pybind11::arg("rope_head_dim"),
          pybind11::arg("q_out"), pybind11::arg("k_out"));
    m.def("nvfp4_grouped_gemm_nopdl", &nvfp4_grouped_gemm_nopdl_entrypoint,
          "Non-PDL grouped GEMM for multi-stream and CUDA graph usage",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K_opt") = std::nullopt, pybind11::arg("D_V_opt") = std::nullopt,
          pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_grouped_gemm_config_nopdl", &nvfp4_grouped_gemm_config_nopdl_entrypoint,
          "Non-PDL grouped GEMM with selectable tile config",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K_opt") = std::nullopt, pybind11::arg("D_V_opt") = std::nullopt,
          pybind11::arg("silu_dim") = 0, pybind11::arg("config_id") = 5);
    m.def("nvfp4_split_dgrad_sum", &nvfp4_split_dgrad_sum,
          "Fused split dgrad: slice concatenated row-quantized gradient → batched GEMM + accumulation",
          pybind11::arg("A_fp4_cat"), pybind11::arg("A_sc_cat"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_fp4_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_cat"),
          pybind11::arg("split_dims"), pybind11::arg("D_out"));
    m.def("nvfp4_accum_gemm_v2", &nvfp4_accum_gemm_v2_entrypoint,
          "Accumulating Batched GEMM: D_out = sum_i(A_i × B_i^T), fused in epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_out"), pybind11::arg("tile_done_buf"));
    m.def("nvfp4_batched_accum_gemm", &nvfp4_batched_accum_gemm_entrypoint,
          "True Batched GEMM with Fused Accumulation: D_out = sum_i(A_i × B_i^T)",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_out"));
    m.def("nvfp4_batched_gemm", &nvfp4_batched_gemm_entrypoint,
          "True Batched GEMM (z-dim parallel): D_i = A_i × B_i^T",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_batched_gemm_strided", &nvfp4_batched_gemm_strided_entrypoint,
          "Strided Batched GEMM: reads A FP4 from full buffer with column offsets, avoiding .contiguous()",
          pybind11::arg("A_full"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_batched_gemm_strided_nopdl", &nvfp4_batched_gemm_strided_nopdl_entrypoint,
          "Non-PDL batched GEMM strided (USE_PDL=false) for CUDA graph safety",
          pybind11::arg("A_full"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_quantize", &nvfp4_quantize_entrypoint);
    m.def("sum3_bf16", &sum3_bf16_entrypoint,
          "Fused 3-way bf16 sum: out = A + B + C (single kernel)",
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("out"));
    m.def("sum3_rmsnorm_bwd_out", &nvfp4_sum3_rmsnorm_bwd_out_entrypoint,
          "Native BF16 sum3 plus RMSNorm backward into caller-owned outputs",
          pybind11::arg("d0"), pybind11::arg("d1"), pybind11::arg("d2"),
          pybind11::arg("input"), pybind11::arg("norm_weight"),
          pybind11::arg("inv_rms"), pybind11::arg("d_sum"),
          pybind11::arg("grad_input"), pybind11::arg("dgamma_partials"),
          pybind11::arg("dgamma"));
    m.def("rmsnorm_bwd_residual_out", &nvfp4_rmsnorm_bwd_residual_out_entrypoint,
          "Native BF16 RMSNorm backward plus residual into caller-owned outputs",
          pybind11::arg("d_normed"), pybind11::arg("input"),
          pybind11::arg("norm_weight"), pybind11::arg("inv_rms"),
          pybind11::arg("residual_grad"), pybind11::arg("grad_input"),
          pybind11::arg("dgamma"));
    m.def("fp32_to_fp4x2", &fp32_to_fp4x2_entrypoint);
    m.def("fp4x2_to_fp32", &fp4x2_to_fp32_entrypoint);
    m.def("nvfp4_fused_gemm", &nvfp4_fused_gemm_entrypoint,
          "Fused Quantize+GEMM: constant SCALE_MAX (fastest, no pre-scan)",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_cta_amax", &nvfp4_fused_gemm_cta_amax_entrypoint,
          "Fused Quantize+GEMM: CTA-level amax pre-scan (better accuracy)",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"));
    m.def("nvfp4_persistent_gemm", &nvfp4_persistent_gemm_entrypoint,
          "Persistent Quantize->GEMM: quantize A+B to HBM then GEMM, single kernel",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));

    // Fused TE→TK GEMM: takes raw TE NVFP4 tensors + dimensions.
    // ALL tensor manipulation (view, reshape, amax*recip) happens in C++.
    // Python side passes only raw data pointers + integer dimensions.
    //
    // Args:
    //   a_fp4_data:   raw fp4 packed data (any shape, viewed as fp4x2)
    //   a_scale_inv:  flat swizzled scales (any shape, reshaped to tiles)
    //   a_amax:       [1] float32
    //   a_M, a_K:     dimensions of A matrix
    //   b_fp4_data, b_scale_inv, b_amax, b_M, b_K: same for B
    //   out:          [a_M, b_M] bf16 pre-allocated output
    m.def("nvfp4_gemm_from_te", [](
        const at::Tensor &a_fp4_data,
        const at::Tensor &a_scale_inv,
        const at::Tensor &a_amax,
        int64_t a_M, int64_t a_K,
        const at::Tensor &b_fp4_data,
        const at::Tensor &b_scale_inv,
        const at::Tensor &b_amax,
        int64_t b_M, int64_t b_K,
        at::Tensor &out
    ) {
        const float NVFP4_SCALE_RECIP = 1.0f / (6.0f * 448.0f);

        // View fp4_data as fp4x2
        auto A = a_fp4_data.view(at::kFloat4_e2m1fn_x2);
        auto B = b_fp4_data.view(at::kFloat4_e2m1fn_x2);

        // Reshape flat scales to tile layout and view as fp8
        int64_t a_ntm = a_M / 128, a_ntk = a_K / 64;
        int64_t b_ntm = b_M / 128, b_ntk = b_K / 64;
        auto A_sc = a_scale_inv.reshape({a_ntm, a_ntk, 512}).view(at::kFloat8_e4m3fn);
        auto B_sc = b_scale_inv.reshape({b_ntm, b_ntk, 512}).view(at::kFloat8_e4m3fn);

        // Compute sc_global
        auto A_sg = a_amax.mul(NVFP4_SCALE_RECIP);
        auto B_sg = b_amax.mul(NVFP4_SCALE_RECIP);

        nvfp4_gemm_entrypoint(A, A_sc, A_sg, B, B_sc, B_sg, out);
    });
}

#endif
