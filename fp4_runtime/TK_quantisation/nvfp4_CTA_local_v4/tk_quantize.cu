#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

#define TK_STANDALONE
#include "fused_localcta_quantize.cuh"
#include "../mxfp4_v4/mxfp4_v3_quantize.cuh"
#include "mixed_mxfp4_localcta.cuh"
#include "direct_localcta_fused_quantize.cuh"
#include "persistent_localcta_silu_quantize.cuh"
#include "persistent_localcta_silu_deriv_quantize.cuh"
#include "localcta_reconstruct.cuh"
#include "../nvfp4_v5/silu_split_bf16.cuh"
#include "../../ThunderKittens/kernels/gemm/common/c1_rms_reduce.cuh"

using transformer_engine::dispatch::nvfp4::nvfp4_scale_t;
namespace py = pybind11;

namespace {

static constexpr unsigned long long kLocalCTASrInvocationStride = 1ull << 32;
__device__ unsigned long long localcta_sr_invocation_offset = 0;

__global__ void localcta_prepare_advancing_rng_state_kernel(
    unsigned long long* rng_state,
    unsigned long long rng_seed,
    unsigned long long rng_subsequence_base
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long offset = atomicAdd(
            &localcta_sr_invocation_offset, kLocalCTASrInvocationStride);
        rng_state[0] = rng_seed;
        rng_state[1] = rng_subsequence_base + offset;
    }
}

__global__ void localcta_prepare_explicit_rng_state_kernel(
    unsigned long long* rng_state,
    unsigned long long* persistent_rng_state
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // The persistent state is owned by one logical producer and all uses
        // of that producer are ordered on its CUDA stream.  atomicAdd keeps a
        // misuse on overlapping streams collision-free, but callers must not
        // share one state across logical producers: which stream wins such a
        // race would still make producer-to-subsequence assignment unstable.
        const unsigned long long subsequence = atomicAdd(
            &persistent_rng_state[1], kLocalCTASrInvocationStride);
        rng_state[0] = persistent_rng_state[0];
        rng_state[1] = subsequence;
    }
}

static void check_localcta_explicit_rng_state(
    const torch::Tensor& persistent_rng_state,
    const torch::Device& expected_device
) {
    TORCH_CHECK(
        persistent_rng_state.is_cuda() &&
            persistent_rng_state.device() == expected_device,
        "localCTA explicit RNG state must be a CUDA tensor on ",
        expected_device);
    TORCH_CHECK(
        persistent_rng_state.scalar_type() == torch::kInt64 &&
            persistent_rng_state.is_contiguous() &&
            persistent_rng_state.numel() == 2,
        "localCTA explicit RNG state must be a contiguous int64 tensor with "
        "exactly two elements: [seed, next_subsequence]");
}

static torch::Tensor make_localcta_advancing_rng_state(
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream,
    torch::Tensor persistent_rng_state = torch::Tensor()
) {
    auto rng_state = torch::empty(
        {2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    if (persistent_rng_state.defined()) {
        check_localcta_explicit_rng_state(
            persistent_rng_state, rng_state.device());
        localcta_prepare_explicit_rng_state_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<unsigned long long*>(rng_state.data_ptr<int64_t>()),
            reinterpret_cast<unsigned long long*>(
                persistent_rng_state.data_ptr<int64_t>()));
    } else {
        localcta_prepare_advancing_rng_state_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<unsigned long long*>(rng_state.data_ptr<int64_t>()),
            static_cast<unsigned long long>(rng_seed),
            static_cast<unsigned long long>(rng_subsequence_base));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return rng_state;
}

__device__ __forceinline__ float rmsnorm_contract_value(
    float x,
    float inv_rms,
    float gamma,
    bool with_silu
) {
    float out = __bfloat162float(__float2bfloat16_rn(x * inv_rms * gamma));
    if (with_silu) {
        out = __bfloat162float(__float2bfloat16_rn(out / (1.0f + expf(-out))));
    }
    return out;
}

__device__ __forceinline__ float localcta_outer_sg_ratio(
    float chunk_sg,
    float outer_sg
) {
    // The outer scale is a max-reduction over the participating chunk scales.
    // A zero outer scale therefore means every participating chunk is exactly
    // zero; preserve that zero without imposing an absolute magnitude floor.
    return outer_sg == 0.0f ? 0.0f : chunk_sg / outer_sg;
}

__device__ __forceinline__ __nv_fp8_e4m3
localcta_rescale_weight_2d_prepared_scale(float value, float ratio) {
    __nv_fp8_e4m3 out = static_cast<__nv_fp8_e4m3>(value * ratio);
    // The common FP32 outer scale carries the absolute magnitude. Preserve a
    // positive normalized block scale if E4M3 rounding would otherwise erase
    // the smallest block in a heterogeneous weight.
    if (value > 0.0f && ratio > 0.0f && static_cast<float>(out) == 0.0f) {
        constexpr float kE4m3MinNonzero = 0.001953125f;  // 2^-9
        out = static_cast<__nv_fp8_e4m3>(kE4m3MinNonzero);
    }
    return out;
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_weight_2d_common_outer_sg_kernel(
    float* __restrict__ row_sg_chunks,
    int64_t count
) {
    float thread_max = 0.0f;
    for (int64_t index = threadIdx.x; index < count; index += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_sg_chunks[index]);
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(
                smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sg_chunks[0] = smem[0];
    }
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_weight_2d_row_sc_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ transposed_sg_chunks,
    const float* __restrict__ common_outer_sg,
    int row_chunks,
    int sc_cols
) {
    const int row = static_cast<int>(blockIdx.x);
    const int sc_col = static_cast<int>(blockIdx.y);
    if (row >= row_chunks || sc_col >= sc_cols) return;

    // Read the preserved column grid because row_sg[0] is reused as the
    // common-scale reduction output. The grids are exact physical transposes.
    const float ratio = localcta_outer_sg_ratio(
        transposed_sg_chunks[(sc_col / 2) * row_chunks + row],
        common_outer_sg[0]);
    const int64_t base =
        (static_cast<int64_t>(row) * sc_cols + sc_col) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float value = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] =
            localcta_rescale_weight_2d_prepared_scale(value, ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_weight_2d_col_sc_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc,
    const float* __restrict__ col_sg_chunks,
    const float* __restrict__ common_outer_sg,
    int col_chunks,
    int sc_rows,
    int sg_rows
) {
    const int col = static_cast<int>(blockIdx.x);
    const int sc_row = static_cast<int>(blockIdx.y);
    if (col >= col_chunks || sc_row >= sc_rows) return;

    const float ratio = localcta_outer_sg_ratio(
        col_sg_chunks[col * sg_rows + sc_row / 2], common_outer_sg[0]);
    const int64_t base =
        (static_cast<int64_t>(col) * sc_rows + sc_row) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float value = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] =
            localcta_rescale_weight_2d_prepared_scale(value, ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void fill_weight_2d_common_outer_sg_kernel(
    float* __restrict__ row_sg,
    int64_t row_count,
    float* __restrict__ col_sg,
    int64_t col_count
) {
    const float value = row_sg[0];
    const int64_t total = row_count + col_count;
    for (int64_t index = threadIdx.x; index < total; index += BLOCK_SIZE) {
        if (index < row_count) {
            row_sg[index] = value;
        } else {
            col_sg[index - row_count] = value;
        }
    }
}

static bool is_power_of_two_int64(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static int64_t outer_sg_tiles_128(int64_t extent) {
    return (extent / 128 + 1) / 2;
}

static void check_rope_live64_tensor(
    const torch::Tensor& rope_cs,
    int64_t rope_seq_len
) {
    TORCH_CHECK(rope_cs.is_cuda() && rope_cs.is_contiguous(),
                "rope_cs must be a contiguous CUDA tensor");
    TORCH_CHECK(rope_cs.scalar_type() == torch::kFloat32,
                "rope_cs must be float32");
    TORCH_CHECK(rope_cs.dim() == 3,
                "rope_cs must have rank 3");
    TORCH_CHECK(is_power_of_two_int64(rope_seq_len),
                "rope_seq_len must be a positive power of two");
    TORCH_CHECK(
        rope_cs.sizes() == torch::IntArrayRef({rope_seq_len, 32, 2}),
        "rope_cs must have shape (rope_seq_len, 32, 2)");
}

template <int BLOCK_SIZE = 256>
__global__ void localcta_fused_norm_to_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ inv_rms_out,
    float epsilon,
    int rows,
    int cols,
    bool with_silu
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* row_x = x + (int64_t)row * cols;
    __nv_bfloat16* row_out = out + (int64_t)row * cols;
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float v = __bfloat162float(row_x[i]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    }

    __shared__ float warp_sums[BLOCK_SIZE / 32];
    __shared__ float row_inv_rms;
    int wid = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();

    if (wid == 0) {
        sum_sq = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int mask = (BLOCK_SIZE / 32) / 2; mask > 0; mask >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        }
        if (lane == 0) {
            row_inv_rms = rsqrtf(sum_sq / cols + epsilon);
            inv_rms_out[row] = row_inv_rms;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float v = __bfloat162float(row_x[i]);
        float g = __bfloat162float(gamma[i]);
        float transformed = rmsnorm_contract_value(v, row_inv_rms, g, with_silu);
        row_out[i] = __float2bfloat16_rn(transformed);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void localcta_compute_inv_rms_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ inv_rms_out,
    float epsilon,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* row_x = x + (int64_t)row * cols;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float v = __bfloat162float(row_x[i]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    }

    __shared__ float warp_sums[BLOCK_SIZE / 32];
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_sums[wid] = sum_sq;
    }
    __syncthreads();

    if (wid == 0) {
        sum_sq = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int mask = (BLOCK_SIZE / 32) / 2; mask > 0; mask >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        }
        if (lane == 0) {
            inv_rms_out[row] = rsqrtf(sum_sq / cols + epsilon);
        }
    }
}

template <int BLOCK_SIZE = 256>
__global__ void initialize_atomic_final_sg_kernel(
    float* __restrict__ row_sg,
    int row_sg_count,
    float* __restrict__ col_sg,
    int col_sg_count,
    unsigned int* __restrict__ work_counter
) {
    const int total = row_sg_count + col_sg_count + 1;
    for (int index = threadIdx.x; index < total; index += BLOCK_SIZE) {
        if (index < row_sg_count) {
            row_sg[index] = 0.0f;
        } else if (index < row_sg_count + col_sg_count) {
            col_sg[index - row_sg_count] = 0.0f;
        } else {
            *work_counter = 0;
        }
    }
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_row_sg_tiles_kernel(
    const float* __restrict__ row_sg_chunk,
    float* __restrict__ row_sg_tiles,
    int row_chunks,
    int sg_cols
) {
    const int tile = blockIdx.x;
    const int row0 = tile * 2;
    const int row1 = row0 + 1;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_cols * 2; idx += BLOCK_SIZE) {
        const int r = idx / sg_cols;
        const int c = idx % sg_cols;
        const int row = (r == 0) ? row0 : row1;
        if (row < row_chunks) {
            thread_max = fmaxf(thread_max, row_sg_chunk[row * sg_cols + c]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sg_tiles[tile] = smem[0];
    }
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_row_sc_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    const float* __restrict__ row_sg_tiles,
    int row_chunks,
    int sc_cols,
    int sg_cols
) {
    const int row = blockIdx.x;
    const int sc_col = blockIdx.y;
    if (row >= row_chunks || sc_col >= sc_cols) return;

    const float denom = row_sg_tiles[row / 2];
    const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
    const float ratio = localcta_outer_sg_ratio(numer, denom);
    const int64_t base = ((int64_t)row * sc_cols + sc_col) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_row_sc_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    float* __restrict__ row_sg_tiles,
    int row_chunks,
    int sc_cols,
    int sg_cols
) {
    const int tile = blockIdx.x;
    const int sc_col = blockIdx.y;
    const int row0 = tile * 2;
    if (row0 >= row_chunks || sc_col >= sc_cols) return;
    const int row1 = row0 + 1;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_cols * 2; idx += BLOCK_SIZE) {
        const int r = idx / sg_cols;
        const int c = idx % sg_cols;
        const int row = (r == 0) ? row0 : row1;
        if (row < row_chunks) {
            thread_max = fmaxf(thread_max, row_sg_chunk[row * sg_cols + c]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = smem[0];
    if (threadIdx.x == 0 && sc_col == 0) {
        row_sg_tiles[tile] = smem[0];
    }

    const int rows_in_tile = min(2, row_chunks - row0);
    const int total = rows_in_tile * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_row = idx / 512;
        const int i = idx % 512;
        const int row = row0 + local_row;
        const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = ((int64_t)row * sc_cols + sc_col) * 512;
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_row_sc_strided_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    const float* __restrict__ row_sg_tiles,
    int row_chunks,
    int sc_cols,
    int sg_cols,
    int64_t row_sc_stride0,
    int64_t row_sc_stride1
) {
    const int row = blockIdx.x;
    const int sc_col = blockIdx.y;
    if (row >= row_chunks || sc_col >= sc_cols) return;

    const float denom = fmaxf(row_sg_tiles[row / 2], 1e-12f);
    const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
    const float ratio = numer / denom;
    const int64_t base = static_cast<int64_t>(row) * row_sc_stride0 +
                         static_cast<int64_t>(sc_col) * row_sc_stride1;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_row_sc_strided_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    float* __restrict__ row_sg_tiles,
    int row_chunks,
    int sc_cols,
    int sg_cols,
    int64_t row_sc_stride0,
    int64_t row_sc_stride1,
    int64_t row_sg_chunk_stride0,
    int64_t row_sg_chunk_stride1
) {
    const int tile = blockIdx.x;
    const int sc_col = blockIdx.y;
    const int row0 = tile * 2;
    if (row0 >= row_chunks || sc_col >= sc_cols) return;
    const int row1 = row0 + 1;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_cols * 2; idx += BLOCK_SIZE) {
        const int r = idx / sg_cols;
        const int c = idx % sg_cols;
        const int row = (r == 0) ? row0 : row1;
        if (row < row_chunks) {
            const int64_t sg_idx = static_cast<int64_t>(row) * row_sg_chunk_stride0 +
                                   static_cast<int64_t>(c) * row_sg_chunk_stride1;
            thread_max = fmaxf(thread_max, row_sg_chunk[sg_idx]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = smem[0];
    if (threadIdx.x == 0 && sc_col == 0) {
        row_sg_tiles[tile] = smem[0];
    }

    const int rows_in_tile = min(2, row_chunks - row0);
    const int total = rows_in_tile * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_row = idx / 512;
        const int i = idx % 512;
        const int row = row0 + local_row;
        const int64_t sg_idx = static_cast<int64_t>(row) * row_sg_chunk_stride0 +
                               static_cast<int64_t>(sc_col / 2) * row_sg_chunk_stride1;
        const float numer = row_sg_chunk[sg_idx];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = static_cast<int64_t>(row) * row_sc_stride0 +
                             static_cast<int64_t>(sc_col) * row_sc_stride1;
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_row_sg_tiles_strided_split2_kernel(
    const float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_0,
    int sg_cols_0,
    int64_t row_sg_chunk_stride0_0,
    int64_t row_sg_chunk_stride1_0,
    const float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_1,
    int sg_cols_1,
    int64_t row_sg_chunk_stride0_1,
    int64_t row_sg_chunk_stride1_1,
    int row_chunks
) {
    const int split = blockIdx.y;
    const int tile = blockIdx.x;
    const int row0 = tile * 2;
    if (row0 >= row_chunks) return;

    const float* row_sg_chunk =
        split == 0 ? row_sg_chunk_0 : row_sg_chunk_1;
    float* row_sg = split == 0 ? row_sg_0 : row_sg_1;
    const int sg_cols = split == 0 ? sg_cols_0 : sg_cols_1;
    const int64_t row_sg_chunk_stride0 =
        split == 0 ? row_sg_chunk_stride0_0 : row_sg_chunk_stride0_1;
    const int64_t row_sg_chunk_stride1 =
        split == 0 ? row_sg_chunk_stride1_0 : row_sg_chunk_stride1_1;

    // Preserve finalize_row_sc_strided_kernel's exact two-row scan and
    // reduction order, but compute the denominator only once per arm/tile
    // instead of once for every scale column.
    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_cols * 2; idx += BLOCK_SIZE) {
        const int r = idx / sg_cols;
        const int c = idx % sg_cols;
        const int row = row0 + r;
        if (row < row_chunks) {
            const int64_t sg_idx =
                static_cast<int64_t>(row) * row_sg_chunk_stride0 +
                static_cast<int64_t>(c) * row_sg_chunk_stride1;
            thread_max = fmaxf(thread_max, row_sg_chunk[sg_idx]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(
                smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sg[tile] = smem[0];
    }
}

__device__ __forceinline__ void rescale_fp8x4_inplace(
    __nv_fp8_e4m3* __restrict__ ptr,
    float ratio
) {
    auto* packed_ptr = reinterpret_cast<__nv_fp8x4_e4m3*>(ptr);
    const float4 values = static_cast<float4>(*packed_ptr);
    const float4 scaled = {
        values.x * ratio,
        values.y * ratio,
        values.z * ratio,
        values.w * ratio,
    };
    *packed_ptr = __nv_fp8x4_e4m3(scaled);
}

// The production split2 carrier has two independently finalized localCTA
// arms stored as strided views into one allocation.  Keep one launch for both
// arms, but assign one scale vector to each warp and update four E4M3 values
// per memory operation.  Eight adjacent scale columns per CTA is the retained
// production grouping: it cuts the M32768/H14336 grid from 57,344 blocks to
// 7,168 without changing either arm's denominator or strided base address.
template <int COLS_PER_BLOCK = 8, int BLOCK_SIZE = 256>
__global__ void rescale_row_sc_strided_split2_fp8x4_warp_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc_0,
    const float* __restrict__ row_sg_chunk_0,
    const float* __restrict__ row_sg_0,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    int64_t row_sg_chunk_stride0_0,
    int64_t row_sg_chunk_stride1_0,
    __nv_fp8_e4m3* __restrict__ row_sc_1,
    const float* __restrict__ row_sg_chunk_1,
    const float* __restrict__ row_sg_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    int64_t row_sg_chunk_stride0_1,
    int64_t row_sg_chunk_stride1_1,
    int row_chunks
) {
    static_assert(BLOCK_SIZE % 32 == 0, "split2 rescaler requires whole warps");
    static_assert(COLS_PER_BLOCK > 0, "split2 rescaler requires a column group");
    constexpr int kWarpsPerBlock = BLOCK_SIZE / 32;
    constexpr unsigned kFullWarpMask = 0xffffffffu;
    const int split = blockIdx.z;
    const int tile = blockIdx.x;
    const int sc_col_start = blockIdx.y * COLS_PER_BLOCK;
    const int row0 = tile * 2;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    __nv_fp8_e4m3* row_sc = split == 0 ? row_sc_0 : row_sc_1;
    const float* row_sg_chunk =
        split == 0 ? row_sg_chunk_0 : row_sg_chunk_1;
    const float* row_sg = split == 0 ? row_sg_0 : row_sg_1;
    const int sc_cols = split == 0 ? sc_cols_0 : sc_cols_1;
    const int sg_cols = split == 0 ? sg_cols_0 : sg_cols_1;
    const int64_t row_sc_stride0 =
        split == 0 ? row_sc_stride0_0 : row_sc_stride0_1;
    const int64_t row_sc_stride1 =
        split == 0 ? row_sc_stride1_0 : row_sc_stride1_1;
    const int64_t row_sg_chunk_stride0 =
        split == 0 ? row_sg_chunk_stride0_0 : row_sg_chunk_stride0_1;
    const int64_t row_sg_chunk_stride1 =
        split == 0 ? row_sg_chunk_stride1_0 : row_sg_chunk_stride1_1;
    if (row0 >= row_chunks || sc_col_start >= sc_cols) return;

    const float denom = row_sg[tile];
    const int rows_in_tile = min(2, row_chunks - row0);
    const int cols_this_block = min(COLS_PER_BLOCK, sc_cols - sc_col_start);
    const int scale_tasks = rows_in_tile * cols_this_block;
    for (int scale_task = warp;
         scale_task < scale_tasks;
         scale_task += kWarpsPerBlock) {
        const int local_row = scale_task / cols_this_block;
        const int local_col = scale_task - local_row * cols_this_block;
        const int row = row0 + local_row;
        const int sc_col = sc_col_start + local_col;
        const int64_t sg_idx =
            static_cast<int64_t>(row) * row_sg_chunk_stride0 +
            static_cast<int64_t>(sc_col / 2) * row_sg_chunk_stride1;
        float ratio = 0.0f;
        if (lane == 0) {
            const float numer = row_sg_chunk[sg_idx];
            ratio = localcta_outer_sg_ratio(numer, denom);
        }
        ratio = __shfl_sync(kFullWarpMask, ratio, 0);
        const int64_t base =
            static_cast<int64_t>(row) * row_sc_stride0 +
            static_cast<int64_t>(sc_col) * row_sc_stride1;
        #pragma unroll
        for (int packed_group = 0; packed_group < 4; ++packed_group) {
            const int packed_i = lane + packed_group * 32;
            rescale_fp8x4_inplace(row_sc + base + packed_i * 4, ratio);
        }
    }
}

template <int BLOCK_SIZE = 256>
__global__ void fold_row_sc_chunk_sg_strided_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    int row_chunks,
    int sc_cols,
    int sg_cols,
    int64_t row_sc_stride0,
    int64_t row_sc_stride1
) {
    const int row = blockIdx.x;
    const int sc_col = blockIdx.y;
    if (row >= row_chunks || sc_col >= sc_cols) return;

    const float ratio = row_sg_chunk[row * sg_cols + (sc_col / 2)];
    const int64_t base = static_cast<int64_t>(row) * row_sc_stride0 +
                         static_cast<int64_t>(sc_col) * row_sc_stride1;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_col_sc_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc,
    const float* __restrict__ col_sg_chunk,
    float* __restrict__ col_sg_tiles,
    int k_chunks,
    int sc_rows,
    int sg_rows
) {
    const int tile = blockIdx.x;
    const int sc_row = blockIdx.y;
    const int k0 = tile * 2;
    if (k0 >= k_chunks || sc_row >= sc_rows) return;
    const int k1 = k0 + 1;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = smem[0];
    if (threadIdx.x == 0 && sc_row == 0) {
        col_sg_tiles[tile] = smem[0];
    }

    const int ks_in_tile = min(2, k_chunks - k0);
    const int total = ks_in_tile * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_k = idx / 512;
        const int i = idx % 512;
        const int k = k0 + local_k;
        const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_row_col_sc_from_final_sg_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    const float* __restrict__ row_sg_tiles,
    int row_chunks,
    int row_sc_cols,
    int row_sg_cols,
    __nv_fp8_e4m3* __restrict__ col_sc,
    const float* __restrict__ col_sg_chunk,
    const float* __restrict__ col_sg_tiles,
    int col_chunks,
    int col_sc_rows,
    int col_sg_rows,
    bool has_col
) {
    const int row_tiles = (row_chunks + 1) / 2;
    const int row_tasks = row_tiles * row_sc_cols;
    const int task = static_cast<int>(blockIdx.x);

    if (task < row_tasks) {
        const int tile = task / row_sc_cols;
        const int sc_col = task - tile * row_sc_cols;
        const int row0 = tile * 2;
        if (row0 >= row_chunks) return;
        const float denom = row_sg_tiles[tile];
        const int rows_in_tile = min(2, row_chunks - row0);
        const int total = rows_in_tile * 512;
        for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
            const int local_row = idx / 512;
            const int i = idx % 512;
            const int row = row0 + local_row;
            const float numer = row_sg_chunk[row * row_sg_cols + (sc_col / 2)];
            const float ratio = localcta_outer_sg_ratio(numer, denom);
            const int64_t base = ((int64_t)row * row_sc_cols + sc_col) * 512;
            const float v = static_cast<float>(row_sc[base + i]);
            row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
        }
        return;
    }

    if (!has_col) return;
    const int col_task = task - row_tasks;
    const int col_tiles = (col_chunks + 1) / 2;
    const int col_tasks = col_tiles * col_sc_rows;
    if (col_task >= col_tasks) return;
    const int tile = col_task / col_sc_rows;
    const int sc_row = col_task - tile * col_sc_rows;
    const int col0 = tile * 2;
    if (col0 >= col_chunks) return;
    const float denom = col_sg_tiles[tile];
    const int cols_in_tile = min(2, col_chunks - col0);
    const int total = cols_in_tile * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_col = idx / 512;
        const int i = idx % 512;
        const int col = col0 + local_col;
        const float numer = col_sg_chunk[col * col_sg_rows + (sc_row / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = ((int64_t)col * col_sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int COLS_PER_BLOCK, int BLOCK_SIZE = 256>
__global__ void rescale_row_col_sc_from_final_sg_vector_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    const float* __restrict__ row_sg_tiles,
    int row_chunks,
    int row_sc_cols,
    int row_sg_cols,
    __nv_fp8_e4m3* __restrict__ col_sc,
    const float* __restrict__ col_sg_chunk,
    const float* __restrict__ col_sg_tiles,
    int col_chunks,
    int col_sc_rows,
    int col_sg_rows,
    bool has_col
) {
    constexpr int kPackedValuesPerScaleTile = 512 / 4;
    const int row_tiles = (row_chunks + 1) / 2;
    const int row_col_blocks = (row_sc_cols + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    const int row_tasks = row_tiles * row_col_blocks;
    const int task = static_cast<int>(blockIdx.x);

    if (task < row_tasks) {
        const int tile = task / row_col_blocks;
        const int sc_col_start = (task - tile * row_col_blocks) * COLS_PER_BLOCK;
        const int row0 = tile * 2;
        if (row0 >= row_chunks || sc_col_start >= row_sc_cols) return;

        const int rows_in_tile = min(2, row_chunks - row0);
        const int cols_this_block = min(COLS_PER_BLOCK, row_sc_cols - sc_col_start);
        const float denom = row_sg_tiles[tile];
        const int total = rows_in_tile * cols_this_block * kPackedValuesPerScaleTile;
        for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
            const int local_row = idx / (cols_this_block * kPackedValuesPerScaleTile);
            const int rem = idx - local_row * cols_this_block * kPackedValuesPerScaleTile;
            const int local_col = rem / kPackedValuesPerScaleTile;
            const int packed_i = rem - local_col * kPackedValuesPerScaleTile;
            const int row = row0 + local_row;
            const int sc_col = sc_col_start + local_col;
            const float numer = row_sg_chunk[row * row_sg_cols + (sc_col / 2)];
            const float ratio = localcta_outer_sg_ratio(numer, denom);
            const int64_t base = ((int64_t)row * row_sc_cols + sc_col) * 512;
            rescale_fp8x4_inplace(row_sc + base + packed_i * 4, ratio);
        }
        return;
    }

    if (!has_col) return;
    const int col_task = task - row_tasks;
    const int col_tiles = (col_chunks + 1) / 2;
    const int col_row_blocks = (col_sc_rows + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    const int col_tasks = col_tiles * col_row_blocks;
    if (col_task >= col_tasks) return;

    const int tile = col_task / col_row_blocks;
    const int sc_row_start = (col_task - tile * col_row_blocks) * COLS_PER_BLOCK;
    const int col0 = tile * 2;
    if (col0 >= col_chunks || sc_row_start >= col_sc_rows) return;

    const int cols_in_tile = min(2, col_chunks - col0);
    const int rows_this_block = min(COLS_PER_BLOCK, col_sc_rows - sc_row_start);
    const float denom = col_sg_tiles[tile];
    const int total = cols_in_tile * rows_this_block * kPackedValuesPerScaleTile;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_col = idx / (rows_this_block * kPackedValuesPerScaleTile);
        const int rem = idx - local_col * rows_this_block * kPackedValuesPerScaleTile;
        const int local_row = rem / kPackedValuesPerScaleTile;
        const int packed_i = rem - local_row * kPackedValuesPerScaleTile;
        const int col = col0 + local_col;
        const int sc_row = sc_row_start + local_row;
        const float numer = col_sg_chunk[col * col_sg_rows + (sc_row / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = ((int64_t)col * col_sc_rows + sc_row) * 512;
        rescale_fp8x4_inplace(col_sc + base + packed_i * 4, ratio);
    }
}

template <int COLS_PER_BLOCK, int BLOCK_SIZE = 256>
__global__ void rescale_row_col_sc_from_final_sg_warp_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc,
    const float* __restrict__ row_sg_chunk,
    const float* __restrict__ row_sg_tiles,
    int row_chunks,
    int row_sc_cols,
    int row_sg_cols,
    __nv_fp8_e4m3* __restrict__ col_sc,
    const float* __restrict__ col_sg_chunk,
    const float* __restrict__ col_sg_tiles,
    int col_chunks,
    int col_sc_rows,
    int col_sg_rows,
    bool has_col
) {
    constexpr int kWarpsPerBlock = BLOCK_SIZE / 32;
    constexpr unsigned kFullWarpMask = 0xffffffffu;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_tiles = (row_chunks + 1) / 2;
    const int row_col_blocks = (row_sc_cols + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    const int row_tasks = row_tiles * row_col_blocks;
    const int task = static_cast<int>(blockIdx.x);

    if (task < row_tasks) {
        const int tile = task / row_col_blocks;
        const int sc_col_start = (task - tile * row_col_blocks) * COLS_PER_BLOCK;
        const int row0 = tile * 2;
        if (row0 >= row_chunks || sc_col_start >= row_sc_cols) return;

        const int rows_in_tile = min(2, row_chunks - row0);
        const int cols_this_block = min(COLS_PER_BLOCK, row_sc_cols - sc_col_start);
        const int scale_tasks = rows_in_tile * cols_this_block;
        for (int scale_task = warp; scale_task < scale_tasks; scale_task += kWarpsPerBlock) {
            const int local_row = scale_task / cols_this_block;
            const int local_col = scale_task - local_row * cols_this_block;
            const int row = row0 + local_row;
            const int sc_col = sc_col_start + local_col;
            float ratio = 0.0f;
            if (lane == 0) {
                const float denom = row_sg_tiles[tile];
                const float numer = row_sg_chunk[row * row_sg_cols + (sc_col / 2)];
                ratio = localcta_outer_sg_ratio(numer, denom);
            }
            ratio = __shfl_sync(kFullWarpMask, ratio, 0);
            const int64_t base = ((int64_t)row * row_sc_cols + sc_col) * 512;
            #pragma unroll
            for (int packed_group = 0; packed_group < 4; ++packed_group) {
                const int packed_i = lane + packed_group * 32;
                rescale_fp8x4_inplace(row_sc + base + packed_i * 4, ratio);
            }
        }
        return;
    }

    if (!has_col) return;
    const int col_task = task - row_tasks;
    const int col_tiles = (col_chunks + 1) / 2;
    const int col_row_blocks = (col_sc_rows + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    const int col_tasks = col_tiles * col_row_blocks;
    if (col_task >= col_tasks) return;

    const int tile = col_task / col_row_blocks;
    const int sc_row_start = (col_task - tile * col_row_blocks) * COLS_PER_BLOCK;
    const int col0 = tile * 2;
    if (col0 >= col_chunks || sc_row_start >= col_sc_rows) return;

    const int cols_in_tile = min(2, col_chunks - col0);
    const int rows_this_block = min(COLS_PER_BLOCK, col_sc_rows - sc_row_start);
    const int scale_tasks = cols_in_tile * rows_this_block;
    for (int scale_task = warp; scale_task < scale_tasks; scale_task += kWarpsPerBlock) {
        const int local_col = scale_task / rows_this_block;
        const int local_row = scale_task - local_col * rows_this_block;
        const int col = col0 + local_col;
        const int sc_row = sc_row_start + local_row;
        float ratio = 0.0f;
        if (lane == 0) {
            const float denom = col_sg_tiles[tile];
            const float numer = col_sg_chunk[col * col_sg_rows + (sc_row / 2)];
            ratio = localcta_outer_sg_ratio(numer, denom);
        }
        ratio = __shfl_sync(kFullWarpMask, ratio, 0);
        const int64_t base = ((int64_t)col * col_sc_rows + sc_row) * 512;
        #pragma unroll
        for (int packed_group = 0; packed_group < 4; ++packed_group) {
            const int packed_i = lane + packed_group * 32;
            rescale_fp8x4_inplace(col_sc + base + packed_i * 4, ratio);
        }
    }
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_col_sg_tiles_kernel(
    const float* __restrict__ col_sg_chunk,
    float* __restrict__ col_sg_tiles,
    int k_chunks,
    int sg_rows
) {
    const int tile = blockIdx.x;
    const int k0 = tile * 2;
    const int k1 = k0 + 1;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        col_sg_tiles[tile] = smem[0];
    }
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_row_col_sg_tiles_kernel(
    const float* __restrict__ row_sg_chunk,
    float* __restrict__ row_sg_tiles,
    int row_chunks,
    int row_sg_cols,
    const float* __restrict__ col_sg_chunk,
    float* __restrict__ col_sg_tiles,
    int col_chunks,
    int col_sg_rows
) {
    const int tile = static_cast<int>(blockIdx.x);
    __shared__ float smem[BLOCK_SIZE];

    float thread_max = 0.0f;
    const int row0 = tile * 2;
    for (int idx = threadIdx.x; idx < row_sg_cols * 2; idx += BLOCK_SIZE) {
        const int row = row0 + idx / row_sg_cols;
        const int col = idx % row_sg_cols;
        if (row < row_chunks) {
            thread_max = fmaxf(
                thread_max, row_sg_chunk[row * row_sg_cols + col]);
        }
    }
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(
                smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && row0 < row_chunks) {
        row_sg_tiles[tile] = smem[0];
    }
    __syncthreads();

    const int col0 = tile * 2;
    if (col0 >= col_chunks) {
        return;
    }
    thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < col_sg_rows * 2; idx += BLOCK_SIZE) {
        const int col = col0 + idx / col_sg_rows;
        const int row = idx % col_sg_rows;
        if (col < col_chunks) {
            thread_max = fmaxf(
                thread_max, col_sg_chunk[col * col_sg_rows + row]);
        }
    }
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(
                smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        col_sg_tiles[tile] = smem[0];
    }
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_col_sc_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc,
    const float* __restrict__ col_sg_chunk,
    const float* __restrict__ col_sg_tiles,
    int k_chunks,
    int sc_rows,
    int sg_rows
) {
    const int k = blockIdx.x;
    const int sc_row = blockIdx.y;
    if (k >= k_chunks || sc_row >= sc_rows) return;

    const float denom = col_sg_tiles[k / 2];
    const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
    const float ratio = localcta_outer_sg_ratio(numer, denom);
    const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void fold_col_sc_chunk_sg_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc,
    const float* __restrict__ col_sg_chunk,
    int k_chunks,
    int sc_rows,
    int sg_rows
) {
    const int k = blockIdx.x;
    const int sc_row = blockIdx.y;
    if (k >= k_chunks || sc_row >= sc_rows) return;

    const float ratio = col_sg_chunk[k * sg_rows + (sc_row / 2)];
    const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_row_sc_split2_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc_0,
    const float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_0,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* __restrict__ row_sc_1,
    const float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1
) {
    const int split = blockIdx.y;
    const int tile = blockIdx.x;
    const int row0 = tile * 2;
    const int row1 = row0 + 1;

    __nv_fp8_e4m3* row_sc = split == 0 ? row_sc_0 : row_sc_1;
    const float* row_sg_chunk = split == 0 ? row_sg_chunk_0 : row_sg_chunk_1;
    float* row_sg = split == 0 ? row_sg_0 : row_sg_1;
    const int sc_cols = split == 0 ? sc_cols_0 : sc_cols_1;
    const int sg_cols = split == 0 ? sg_cols_0 : sg_cols_1;
    const int64_t row_sc_stride0 = split == 0 ? row_sc_stride0_0 : row_sc_stride0_1;
    const int64_t row_sc_stride1 = split == 0 ? row_sc_stride1_0 : row_sc_stride1_1;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_cols * 2; idx += BLOCK_SIZE) {
        const int r = idx / sg_cols;
        const int c = idx % sg_cols;
        const int row = (r == 0) ? row0 : row1;
        thread_max = fmaxf(thread_max, row_sg_chunk[row * sg_cols + c]);
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(smem[0], 1e-12f);
    if (threadIdx.x == 0) {
        row_sg[tile] = smem[0];
    }
    __syncthreads();

    const int total = 2 * sc_cols * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_row = idx / (sc_cols * 512);
        const int rem = idx % (sc_cols * 512);
        const int sc_col = rem / 512;
        const int i = rem % 512;
        const int row = local_row == 0 ? row0 : row1;
        const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
        const float ratio = numer / denom;
        const int64_t base = (int64_t)row * row_sc_stride0 + (int64_t)sc_col * row_sc_stride1;
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_row_sc_split2_shared_outer_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc_0,
    const float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_0,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* __restrict__ row_sc_1,
    const float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    int row_chunks
) {
    const int tile = blockIdx.x;
    const int global_sc_col = blockIdx.y;
    const int row0 = tile * 2;
    if (row0 >= row_chunks || global_sc_col >= sc_cols_0 + sc_cols_1) return;
    const int row1 = row0 + 1;
    const int total_sg_cols = sg_cols_0 + sg_cols_1;

    // Match a single [M, K0 + K1] carrier's reduction order exactly: the
    // first split's chunk SGs precede the second split's chunk SGs for each
    // of the two rows participating in an outer-SG tile.
    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < total_sg_cols * 2; idx += BLOCK_SIZE) {
        const int local_row = idx / total_sg_cols;
        const int global_sg_col = idx % total_sg_cols;
        const int row = local_row == 0 ? row0 : row1;
        if (row < row_chunks) {
            const float value = global_sg_col < sg_cols_0
                ? row_sg_chunk_0[row * sg_cols_0 + global_sg_col]
                : row_sg_chunk_1[
                    row * sg_cols_1 + (global_sg_col - sg_cols_0)];
            thread_max = fmaxf(thread_max, value);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = smem[0];
    if (threadIdx.x == 0 && global_sc_col == 0) {
        row_sg_0[tile] = smem[0];
        row_sg_1[tile] = smem[0];
    }

    const bool second_split = global_sc_col >= sc_cols_0;
    __nv_fp8_e4m3* row_sc = second_split ? row_sc_1 : row_sc_0;
    const float* row_sg_chunk = second_split ? row_sg_chunk_1 : row_sg_chunk_0;
    const int sc_col = second_split ? global_sc_col - sc_cols_0 : global_sc_col;
    const int sc_cols = second_split ? sc_cols_1 : sc_cols_0;
    const int sg_cols = second_split ? sg_cols_1 : sg_cols_0;
    const int64_t row_sc_stride0 = second_split ? row_sc_stride0_1 : row_sc_stride0_0;
    const int64_t row_sc_stride1 = second_split ? row_sc_stride1_1 : row_sc_stride1_0;

    const int rows_in_tile = min(2, row_chunks - row0);
    const int total = rows_in_tile * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_row = idx / 512;
        const int packed_i = idx % 512;
        const int row = row0 + local_row;
        const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base =
            static_cast<int64_t>(row) * row_sc_stride0 +
            static_cast<int64_t>(sc_col) * row_sc_stride1;
        const float value = static_cast<float>(row_sc[base + packed_i]);
        row_sc[base + packed_i] = static_cast<__nv_fp8_e4m3>(value * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_row_sc_split3_shared_outer_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc_0,
    const float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_shared,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* __restrict__ row_sc_1,
    const float* __restrict__ row_sg_chunk_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    __nv_fp8_e4m3* __restrict__ row_sc_2,
    const float* __restrict__ row_sg_chunk_2,
    int sc_cols_2,
    int sg_cols_2,
    int64_t row_sc_stride0_2,
    int64_t row_sc_stride1_2,
    int row_chunks
) {
    const int tile = blockIdx.x;
    const int global_sc_col = blockIdx.y;
    const int row0 = tile * 2;
    const int total_sc_cols = sc_cols_0 + sc_cols_1 + sc_cols_2;
    if (row0 >= row_chunks || global_sc_col >= total_sc_cols) return;
    const int row1 = row0 + 1;
    const int total_sg_cols = sg_cols_0 + sg_cols_1 + sg_cols_2;

    // Match one logical [M, N0 + N1 + N2] carrier exactly: for each row,
    // Q chunks precede K chunks, which precede V chunks in the reduction.
    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < total_sg_cols * 2; idx += BLOCK_SIZE) {
        const int local_row = idx / total_sg_cols;
        const int global_sg_col = idx % total_sg_cols;
        const int row = local_row == 0 ? row0 : row1;
        if (row < row_chunks) {
            float value;
            if (global_sg_col < sg_cols_0) {
                value = row_sg_chunk_0[row * sg_cols_0 + global_sg_col];
            } else if (global_sg_col < sg_cols_0 + sg_cols_1) {
                value = row_sg_chunk_1[
                    row * sg_cols_1 + (global_sg_col - sg_cols_0)];
            } else {
                value = row_sg_chunk_2[
                    row * sg_cols_2 + (global_sg_col - sg_cols_0 - sg_cols_1)];
            }
            thread_max = fmaxf(thread_max, value);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = smem[0];
    if (threadIdx.x == 0 && global_sc_col == 0) {
        row_sg_shared[tile] = denom;
    }

    __nv_fp8_e4m3* row_sc;
    const float* row_sg_chunk;
    int sc_col;
    int sg_cols;
    int64_t row_sc_stride0;
    int64_t row_sc_stride1;
    if (global_sc_col < sc_cols_0) {
        row_sc = row_sc_0;
        row_sg_chunk = row_sg_chunk_0;
        sc_col = global_sc_col;
        sg_cols = sg_cols_0;
        row_sc_stride0 = row_sc_stride0_0;
        row_sc_stride1 = row_sc_stride1_0;
    } else if (global_sc_col < sc_cols_0 + sc_cols_1) {
        row_sc = row_sc_1;
        row_sg_chunk = row_sg_chunk_1;
        sc_col = global_sc_col - sc_cols_0;
        sg_cols = sg_cols_1;
        row_sc_stride0 = row_sc_stride0_1;
        row_sc_stride1 = row_sc_stride1_1;
    } else {
        row_sc = row_sc_2;
        row_sg_chunk = row_sg_chunk_2;
        sc_col = global_sc_col - sc_cols_0 - sc_cols_1;
        sg_cols = sg_cols_2;
        row_sc_stride0 = row_sc_stride0_2;
        row_sc_stride1 = row_sc_stride1_2;
    }

    const int rows_in_tile = min(2, row_chunks - row0);
    const int total = rows_in_tile * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_row = idx / 512;
        const int packed_i = idx % 512;
        const int row = row0 + local_row;
        const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base =
            static_cast<int64_t>(row) * row_sc_stride0 +
            static_cast<int64_t>(sc_col) * row_sc_stride1;
        const float value = static_cast<float>(row_sc[base + packed_i]);
        row_sc[base + packed_i] = static_cast<__nv_fp8_e4m3>(value * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_col_sc_split2_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_1,
    int k_chunks_1
) {
    const int split = blockIdx.y;
    const int tile = blockIdx.x;
    const int k0 = tile * 2;
    const int k1 = k0 + 1;

    __nv_fp8_e4m3* col_sc = split == 0 ? col_sc_0 : col_sc_1;
    const float* col_sg_chunk = split == 0 ? col_sg_chunk_0 : col_sg_chunk_1;
    float* col_sg = split == 0 ? col_sg_0 : col_sg_1;
    const int k_chunks = split == 0 ? k_chunks_0 : k_chunks_1;
    if (k0 >= k_chunks) return;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(smem[0], 1e-12f);
    if (threadIdx.x == 0) {
        col_sg[tile] = smem[0];
    }
    __syncthreads();

    const int total = min(2, k_chunks - k0) * sc_rows * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_k = idx / (sc_rows * 512);
        const int rem = idx % (sc_rows * 512);
        const int sc_row = rem / 512;
        const int i = rem % 512;
        const int k = k0 + local_k;
        const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
        const float ratio = numer / denom;
        const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_col_sc_split2_chunked_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    int sc_rows_per_block,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_1,
    int k_chunks_1
) {
    const int tile = blockIdx.x;
    const int sc_row_block = blockIdx.y;
    const int split = blockIdx.z;
    const int k0 = tile * 2;
    const int k1 = k0 + 1;

    __nv_fp8_e4m3* col_sc = split == 0 ? col_sc_0 : col_sc_1;
    const float* col_sg_chunk = split == 0 ? col_sg_chunk_0 : col_sg_chunk_1;
    float* col_sg = split == 0 ? col_sg_0 : col_sg_1;
    const int k_chunks = split == 0 ? k_chunks_0 : k_chunks_1;
    if (k0 >= k_chunks) return;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(smem[0], 1e-12f);
    if (threadIdx.x == 0 && sc_row_block == 0) {
        col_sg[tile] = smem[0];
    }
    __syncthreads();

    const int sc_row_start = sc_row_block * sc_rows_per_block;
    int sc_row_count = sc_rows - sc_row_start;
    if (sc_row_count <= 0) return;
    if (sc_row_count > sc_rows_per_block) sc_row_count = sc_rows_per_block;
    const int total = min(2, k_chunks - k0) * sc_row_count * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_k = idx / (sc_row_count * 512);
        const int rem = idx % (sc_row_count * 512);
        const int local_sc_row = rem / 512;
        const int i = rem % 512;
        const int sc_row = sc_row_start + local_sc_row;
        const int k = k0 + local_k;
        const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
        const float ratio = numer / denom;
        const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_col_sg_tiles_split2_kernel(
    const float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sg_rows,
    const float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_1,
    int k_chunks_1
) {
    const int split = blockIdx.y;
    const int tile = blockIdx.x;
    const int k0 = tile * 2;
    const int k1 = k0 + 1;

    const float* col_sg_chunk = split == 0 ? col_sg_chunk_0 : col_sg_chunk_1;
    float* col_sg = split == 0 ? col_sg_0 : col_sg_1;
    const int k_chunks = split == 0 ? k_chunks_0 : k_chunks_1;
    if (k0 >= k_chunks) return;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        col_sg[tile] = smem[0];
    }
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_col_sc_split2_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    const float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    const float* __restrict__ col_sg_1,
    int k_chunks_1
) {
    const int k = blockIdx.x;
    const int sc_row = blockIdx.y;
    const int split = blockIdx.z;

    __nv_fp8_e4m3* col_sc = split == 0 ? col_sc_0 : col_sc_1;
    const float* col_sg_chunk = split == 0 ? col_sg_chunk_0 : col_sg_chunk_1;
    const float* col_sg = split == 0 ? col_sg_0 : col_sg_1;
    const int k_chunks = split == 0 ? k_chunks_0 : k_chunks_1;
    if (k >= k_chunks || sc_row >= sc_rows) return;

    const float denom = col_sg[k / 2];
    const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
    const float ratio = localcta_outer_sg_ratio(numer, denom);
    const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_row_sc_split3_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc_0,
    const float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_0,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* __restrict__ row_sc_1,
    const float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    __nv_fp8_e4m3* __restrict__ row_sc_2,
    const float* __restrict__ row_sg_chunk_2,
    float* __restrict__ row_sg_2,
    int sc_cols_2,
    int sg_cols_2,
    int64_t row_sc_stride0_2,
    int64_t row_sc_stride1_2
) {
    const int split = blockIdx.y;
    const int tile = blockIdx.x;
    const int row0 = tile * 2;
    const int row1 = row0 + 1;

    __nv_fp8_e4m3* row_sc;
    const float* row_sg_chunk;
    float* row_sg;
    int sc_cols;
    int sg_cols;
    int64_t row_sc_stride0;
    int64_t row_sc_stride1;
    if (split == 0) {
        row_sc = row_sc_0; row_sg_chunk = row_sg_chunk_0; row_sg = row_sg_0; sc_cols = sc_cols_0; sg_cols = sg_cols_0;
        row_sc_stride0 = row_sc_stride0_0; row_sc_stride1 = row_sc_stride1_0;
    } else if (split == 1) {
        row_sc = row_sc_1; row_sg_chunk = row_sg_chunk_1; row_sg = row_sg_1; sc_cols = sc_cols_1; sg_cols = sg_cols_1;
        row_sc_stride0 = row_sc_stride0_1; row_sc_stride1 = row_sc_stride1_1;
    } else {
        row_sc = row_sc_2; row_sg_chunk = row_sg_chunk_2; row_sg = row_sg_2; sc_cols = sc_cols_2; sg_cols = sg_cols_2;
        row_sc_stride0 = row_sc_stride0_2; row_sc_stride1 = row_sc_stride1_2;
    }

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_cols * 2; idx += BLOCK_SIZE) {
        const int r = idx / sg_cols;
        const int c = idx % sg_cols;
        const int row = (r == 0) ? row0 : row1;
        thread_max = fmaxf(thread_max, row_sg_chunk[row * sg_cols + c]);
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(smem[0], 1e-12f);
    if (threadIdx.x == 0) {
        row_sg[tile] = smem[0];
    }
    __syncthreads();

    const int total = 2 * sc_cols * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_row = idx / (sc_cols * 512);
        const int rem = idx % (sc_cols * 512);
        const int sc_col = rem / 512;
        const int i = rem % 512;
        const int row = local_row == 0 ? row0 : row1;
        const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
        const float ratio = numer / denom;
        const int64_t base = (int64_t)row * row_sc_stride0 + (int64_t)sc_col * row_sc_stride1;
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE>
static void launch_finalize_row_sc_split3(
    dim3 grid,
    cudaStream_t stream,
    __nv_fp8_e4m3* row_sc_0,
    const float* row_sg_chunk_0,
    float* row_sg_0,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* row_sc_1,
    const float* row_sg_chunk_1,
    float* row_sg_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    __nv_fp8_e4m3* row_sc_2,
    const float* row_sg_chunk_2,
    float* row_sg_2,
    int sc_cols_2,
    int sg_cols_2,
    int64_t row_sc_stride0_2,
    int64_t row_sc_stride1_2
) {
    finalize_row_sc_split3_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
        row_sc_0,
        row_sg_chunk_0,
        row_sg_0,
        sc_cols_0,
        sg_cols_0,
        row_sc_stride0_0,
        row_sc_stride1_0,
        row_sc_1,
        row_sg_chunk_1,
        row_sg_1,
        sc_cols_1,
        sg_cols_1,
        row_sc_stride0_1,
        row_sc_stride1_1,
        row_sc_2,
        row_sg_chunk_2,
        row_sg_2,
        sc_cols_2,
        sg_cols_2,
        row_sc_stride0_2,
        row_sc_stride1_2);
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_row_sg_split3_kernel(
    const float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_0,
    int sg_cols_0,
    const float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_1,
    int sg_cols_1,
    const float* __restrict__ row_sg_chunk_2,
    float* __restrict__ row_sg_2,
    int sg_cols_2
) {
    const int tile = blockIdx.x;
    const int split = blockIdx.y;
    const int row0 = tile * 2;
    const int row1 = row0 + 1;

    const float* row_sg_chunk;
    float* row_sg;
    int sg_cols;
    if (split == 0) {
        row_sg_chunk = row_sg_chunk_0; row_sg = row_sg_0; sg_cols = sg_cols_0;
    } else if (split == 1) {
        row_sg_chunk = row_sg_chunk_1; row_sg = row_sg_1; sg_cols = sg_cols_1;
    } else {
        row_sg_chunk = row_sg_chunk_2; row_sg = row_sg_2; sg_cols = sg_cols_2;
    }

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_cols * 2; idx += BLOCK_SIZE) {
        const int r = idx / sg_cols;
        const int c = idx % sg_cols;
        const int row = (r == 0) ? row0 : row1;
        thread_max = fmaxf(thread_max, row_sg_chunk[row * sg_cols + c]);
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sg[tile] = smem[0];
    }
}

template <int COLS_PER_BLOCK, int BLOCK_SIZE = 256>
__global__ void rescale_row_sc_split3_cols_kernel(
    __nv_fp8_e4m3* __restrict__ row_sc_0,
    const float* __restrict__ row_sg_chunk_0,
    const float* __restrict__ row_sg_0,
    int row_chunks_0,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* __restrict__ row_sc_1,
    const float* __restrict__ row_sg_chunk_1,
    const float* __restrict__ row_sg_1,
    int row_chunks_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    __nv_fp8_e4m3* __restrict__ row_sc_2,
    const float* __restrict__ row_sg_chunk_2,
    const float* __restrict__ row_sg_2,
    int row_chunks_2,
    int sc_cols_2,
    int sg_cols_2,
    int64_t row_sc_stride0_2,
    int64_t row_sc_stride1_2
) {
    const int tile = blockIdx.x;
    const int sc_col_start = blockIdx.y * COLS_PER_BLOCK;
    const int split = blockIdx.z;
    const int row0 = tile * 2;

    __nv_fp8_e4m3* row_sc;
    const float* row_sg_chunk;
    const float* row_sg;
    int row_chunks;
    int sc_cols;
    int sg_cols;
    int64_t row_sc_stride0;
    int64_t row_sc_stride1;
    if (split == 0) {
        row_sc = row_sc_0; row_sg_chunk = row_sg_chunk_0; row_sg = row_sg_0;
        row_chunks = row_chunks_0; sc_cols = sc_cols_0; sg_cols = sg_cols_0;
        row_sc_stride0 = row_sc_stride0_0; row_sc_stride1 = row_sc_stride1_0;
    } else if (split == 1) {
        row_sc = row_sc_1; row_sg_chunk = row_sg_chunk_1; row_sg = row_sg_1;
        row_chunks = row_chunks_1; sc_cols = sc_cols_1; sg_cols = sg_cols_1;
        row_sc_stride0 = row_sc_stride0_1; row_sc_stride1 = row_sc_stride1_1;
    } else {
        row_sc = row_sc_2; row_sg_chunk = row_sg_chunk_2; row_sg = row_sg_2;
        row_chunks = row_chunks_2; sc_cols = sc_cols_2; sg_cols = sg_cols_2;
        row_sc_stride0 = row_sc_stride0_2; row_sc_stride1 = row_sc_stride1_2;
    }
    if (row0 >= row_chunks || sc_col_start >= sc_cols) return;

    const int rows_in_tile = min(2, row_chunks - row0);
    const int cols_this_block = min(COLS_PER_BLOCK, sc_cols - sc_col_start);
    const float denom = row_sg[tile];
    const int total = rows_in_tile * cols_this_block * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_row = idx / (cols_this_block * 512);
        const int rem = idx - local_row * cols_this_block * 512;
        const int local_col = rem / 512;
        const int i = rem - local_col * 512;
        const int row = row0 + local_row;
        const int sc_col = sc_col_start + local_col;
        const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = static_cast<int64_t>(row) * row_sc_stride0 +
                             static_cast<int64_t>(sc_col) * row_sc_stride1;
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int COLS_PER_BLOCK>
static void launch_rescale_row_sc_split3_cols(
    dim3 grid,
    cudaStream_t stream,
    __nv_fp8_e4m3* row_sc_0,
    const float* row_sg_chunk_0,
    const float* row_sg_0,
    int row_chunks_0,
    int sc_cols_0,
    int sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* row_sc_1,
    const float* row_sg_chunk_1,
    const float* row_sg_1,
    int row_chunks_1,
    int sc_cols_1,
    int sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    __nv_fp8_e4m3* row_sc_2,
    const float* row_sg_chunk_2,
    const float* row_sg_2,
    int row_chunks_2,
    int sc_cols_2,
    int sg_cols_2,
    int64_t row_sc_stride0_2,
    int64_t row_sc_stride1_2
) {
    rescale_row_sc_split3_cols_kernel<COLS_PER_BLOCK><<<grid, 256, 0, stream>>>(
        row_sc_0,
        row_sg_chunk_0,
        row_sg_0,
        row_chunks_0,
        sc_cols_0,
        sg_cols_0,
        row_sc_stride0_0,
        row_sc_stride1_0,
        row_sc_1,
        row_sg_chunk_1,
        row_sg_1,
        row_chunks_1,
        sc_cols_1,
        sg_cols_1,
        row_sc_stride0_1,
        row_sc_stride1_1,
        row_sc_2,
        row_sg_chunk_2,
        row_sg_2,
        row_chunks_2,
        sc_cols_2,
        sg_cols_2,
        row_sc_stride0_2,
        row_sc_stride1_2);
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_col_sc_split3_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_1,
    int k_chunks_1,
    __nv_fp8_e4m3* __restrict__ col_sc_2,
    const float* __restrict__ col_sg_chunk_2,
    float* __restrict__ col_sg_2,
    int k_chunks_2
) {
    const int split = blockIdx.y;
    const int tile = blockIdx.x;
    const int k0 = tile * 2;
    const int k1 = k0 + 1;

    __nv_fp8_e4m3* col_sc;
    const float* col_sg_chunk;
    float* col_sg;
    int k_chunks;
    if (split == 0) {
        col_sc = col_sc_0; col_sg_chunk = col_sg_chunk_0; col_sg = col_sg_0; k_chunks = k_chunks_0;
    } else if (split == 1) {
        col_sc = col_sc_1; col_sg_chunk = col_sg_chunk_1; col_sg = col_sg_1; k_chunks = k_chunks_1;
    } else {
        col_sc = col_sc_2; col_sg_chunk = col_sg_chunk_2; col_sg = col_sg_2; k_chunks = k_chunks_2;
    }
    if (k0 >= k_chunks) return;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(smem[0], 1e-12f);
    if (threadIdx.x == 0) {
        col_sg[tile] = smem[0];
    }
    __syncthreads();

    const int total = min(2, k_chunks - k0) * sc_rows * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_k = idx / (sc_rows * 512);
        const int rem = idx % (sc_rows * 512);
        const int sc_row = rem / 512;
        const int i = rem % 512;
        const int k = k0 + local_k;
        const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
        const float ratio = numer / denom;
        const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void finalize_col_sc_split3_chunked_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    int sc_rows_per_block,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_1,
    int k_chunks_1,
    __nv_fp8_e4m3* __restrict__ col_sc_2,
    const float* __restrict__ col_sg_chunk_2,
    float* __restrict__ col_sg_2,
    int k_chunks_2
) {
    const int tile = blockIdx.x;
    const int sc_row_block = blockIdx.y;
    const int split = blockIdx.z;
    const int k0 = tile * 2;
    const int k1 = k0 + 1;

    __nv_fp8_e4m3* col_sc;
    const float* col_sg_chunk;
    float* col_sg;
    int k_chunks;
    if (split == 0) {
        col_sc = col_sc_0; col_sg_chunk = col_sg_chunk_0; col_sg = col_sg_0; k_chunks = k_chunks_0;
    } else if (split == 1) {
        col_sc = col_sc_1; col_sg_chunk = col_sg_chunk_1; col_sg = col_sg_1; k_chunks = k_chunks_1;
    } else {
        col_sc = col_sc_2; col_sg_chunk = col_sg_chunk_2; col_sg = col_sg_2; k_chunks = k_chunks_2;
    }
    if (k0 >= k_chunks) return;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(smem[0], 1e-12f);
    if (threadIdx.x == 0 && sc_row_block == 0) {
        col_sg[tile] = smem[0];
    }
    __syncthreads();

    const int sc_row_start = sc_row_block * sc_rows_per_block;
    int sc_row_count = sc_rows - sc_row_start;
    if (sc_row_count <= 0) return;
    if (sc_row_count > sc_rows_per_block) sc_row_count = sc_rows_per_block;
    const int total = min(2, k_chunks - k0) * sc_row_count * 512;
    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        const int local_k = idx / (sc_row_count * 512);
        const int rem = idx % (sc_row_count * 512);
        const int local_sc_row = rem / 512;
        const int i = rem % 512;
        const int sc_row = sc_row_start + local_sc_row;
        const int k = k0 + local_k;
        const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
        const float ratio = numer / denom;
        const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_col_sg_tiles_split3_kernel(
    const float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sg_rows,
    const float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_1,
    int k_chunks_1,
    const float* __restrict__ col_sg_chunk_2,
    float* __restrict__ col_sg_2,
    int k_chunks_2
) {
    const int split = blockIdx.y;
    const int tile = blockIdx.x;
    const int k0 = tile * 2;
    const int k1 = k0 + 1;

    const float* col_sg_chunk;
    float* col_sg;
    int k_chunks;
    if (split == 0) {
        col_sg_chunk = col_sg_chunk_0; col_sg = col_sg_0; k_chunks = k_chunks_0;
    } else if (split == 1) {
        col_sg_chunk = col_sg_chunk_1; col_sg = col_sg_1; k_chunks = k_chunks_1;
    } else {
        col_sg_chunk = col_sg_chunk_2; col_sg = col_sg_2; k_chunks = k_chunks_2;
    }
    if (k0 >= k_chunks) return;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < sg_rows * 2; idx += BLOCK_SIZE) {
        const int k = idx / sg_rows;
        const int r = idx % sg_rows;
        const int kk = (k == 0) ? k0 : k1;
        if (kk < k_chunks) {
            thread_max = fmaxf(thread_max, col_sg_chunk[kk * sg_rows + r]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        col_sg[tile] = smem[0];
    }
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_row_col_sg_split3_kernel(
    const float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_0,
    int sg_cols_0,
    const float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_1,
    int sg_cols_1,
    const float* __restrict__ row_sg_chunk_2,
    float* __restrict__ row_sg_2,
    int sg_cols_2,
    const float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sg_rows,
    const float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_1,
    int k_chunks_1,
    const float* __restrict__ col_sg_chunk_2,
    float* __restrict__ col_sg_2,
    int k_chunks_2,
    int row_tiles
) {
    const int tile = blockIdx.x;
    const int split = blockIdx.y;
    const bool reduce_cols = blockIdx.z != 0;

    const float* chunk;
    float* sg;
    int limit;
    int secondary;
    if (reduce_cols) {
        if (split == 0) {
            chunk = col_sg_chunk_0; sg = col_sg_0; limit = k_chunks_0;
        } else if (split == 1) {
            chunk = col_sg_chunk_1; sg = col_sg_1; limit = k_chunks_1;
        } else {
            chunk = col_sg_chunk_2; sg = col_sg_2; limit = k_chunks_2;
        }
        secondary = sg_rows;
    } else {
        if (tile >= row_tiles) return;
        if (split == 0) {
            chunk = row_sg_chunk_0; sg = row_sg_0; limit = sg_cols_0;
        } else if (split == 1) {
            chunk = row_sg_chunk_1; sg = row_sg_1; limit = sg_cols_1;
        } else {
            chunk = row_sg_chunk_2; sg = row_sg_2; limit = sg_cols_2;
        }
        secondary = limit;
    }

    float thread_max = 0.0f;
    if (reduce_cols) {
        const int k0 = tile * 2;
        const int k1 = k0 + 1;
        if (k0 >= limit) return;
        for (int idx = threadIdx.x; idx < secondary * 2; idx += BLOCK_SIZE) {
            const int k = idx / secondary;
            const int r = idx % secondary;
            const int kk = (k == 0) ? k0 : k1;
            if (kk < limit) {
                thread_max = fmaxf(thread_max, chunk[kk * secondary + r]);
            }
        }
    } else {
        const int row0 = tile * 2;
        const int row1 = row0 + 1;
        for (int idx = threadIdx.x; idx < limit * 2; idx += BLOCK_SIZE) {
            const int r = idx / limit;
            const int c = idx % limit;
            const int row = (r == 0) ? row0 : row1;
            thread_max = fmaxf(thread_max, chunk[row * limit + c]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sg[tile] = smem[0];
    }
}

static void launch_reduce_row_col_sg_split3(
    const torch::Tensor& row_sg_chunk_0,
    torch::Tensor& row_sg_0,
    const torch::Tensor& row_sg_chunk_1,
    torch::Tensor& row_sg_1,
    const torch::Tensor& row_sg_chunk_2,
    torch::Tensor& row_sg_2,
    const torch::Tensor& col_sg_chunk_0,
    torch::Tensor& col_sg_0,
    const torch::Tensor& col_sg_chunk_1,
    torch::Tensor& col_sg_1,
    const torch::Tensor& col_sg_chunk_2,
    torch::Tensor& col_sg_2
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int row_tiles = static_cast<int>(row_sg_0.numel());
    const int k_chunks_0 = static_cast<int>(col_sg_chunk_0.size(0));
    const int k_chunks_1 = static_cast<int>(col_sg_chunk_1.size(0));
    const int k_chunks_2 = static_cast<int>(col_sg_chunk_2.size(0));
    const int max_col_tiles = std::max(
        std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2),
        (k_chunks_2 + 1) / 2);
    const int grid_tiles = std::max(row_tiles, max_col_tiles);
    dim3 grid(static_cast<unsigned int>(grid_tiles), 3u, 2u);
    reduce_row_col_sg_split3_kernel<<<grid, 256, 0, stream>>>(
        row_sg_chunk_0.data_ptr<float>(),
        row_sg_0.data_ptr<float>(),
        static_cast<int>(row_sg_chunk_0.size(1)),
        row_sg_chunk_1.data_ptr<float>(),
        row_sg_1.data_ptr<float>(),
        static_cast<int>(row_sg_chunk_1.size(1)),
        row_sg_chunk_2.data_ptr<float>(),
        row_sg_2.data_ptr<float>(),
        static_cast<int>(row_sg_chunk_2.size(1)),
        col_sg_chunk_0.data_ptr<float>(),
        col_sg_0.data_ptr<float>(),
        k_chunks_0,
        static_cast<int>(col_sg_chunk_0.size(1)),
        col_sg_chunk_1.data_ptr<float>(),
        col_sg_1.data_ptr<float>(),
        k_chunks_1,
        col_sg_chunk_2.data_ptr<float>(),
        col_sg_2.data_ptr<float>(),
        k_chunks_2,
        row_tiles);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reduce_row_col_sg_split3_kernel failed: ", cudaGetErrorString(err));
}

template <int BLOCK_SIZE = 256>
__global__ void rescale_col_sc_split3_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    const float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    const float* __restrict__ col_sg_1,
    int k_chunks_1,
    __nv_fp8_e4m3* __restrict__ col_sc_2,
    const float* __restrict__ col_sg_chunk_2,
    const float* __restrict__ col_sg_2,
    int k_chunks_2
) {
    const int k = blockIdx.x;
    const int sc_row = blockIdx.y;
    const int split = blockIdx.z;

    __nv_fp8_e4m3* col_sc;
    const float* col_sg_chunk;
    const float* col_sg;
    int k_chunks;
    if (split == 0) {
        col_sc = col_sc_0; col_sg_chunk = col_sg_chunk_0; col_sg = col_sg_0; k_chunks = k_chunks_0;
    } else if (split == 1) {
        col_sc = col_sc_1; col_sg_chunk = col_sg_chunk_1; col_sg = col_sg_1; k_chunks = k_chunks_1;
    } else {
        col_sc = col_sc_2; col_sg_chunk = col_sg_chunk_2; col_sg = col_sg_2; k_chunks = k_chunks_2;
    }
    if (k >= k_chunks || sc_row >= sc_rows) return;

    const float denom = col_sg[k / 2];
    const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
    const float ratio = localcta_outer_sg_ratio(numer, denom);
    const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int ROWS_PER_BLOCK, int BLOCK_SIZE = 256>
__global__ void rescale_col_sc_split3_rows_kernel(
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    const float* __restrict__ col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    const float* __restrict__ col_sg_1,
    int k_chunks_1,
    __nv_fp8_e4m3* __restrict__ col_sc_2,
    const float* __restrict__ col_sg_chunk_2,
    const float* __restrict__ col_sg_2,
    int k_chunks_2
) {
    const int k = blockIdx.x;
    const int sc_row_start = blockIdx.y * ROWS_PER_BLOCK;
    const int split = blockIdx.z;

    __nv_fp8_e4m3* col_sc;
    const float* col_sg_chunk;
    const float* col_sg;
    int k_chunks;
    if (split == 0) {
        col_sc = col_sc_0; col_sg_chunk = col_sg_chunk_0; col_sg = col_sg_0; k_chunks = k_chunks_0;
    } else if (split == 1) {
        col_sc = col_sc_1; col_sg_chunk = col_sg_chunk_1; col_sg = col_sg_1; k_chunks = k_chunks_1;
    } else {
        col_sc = col_sc_2; col_sg_chunk = col_sg_chunk_2; col_sg = col_sg_2; k_chunks = k_chunks_2;
    }
    if (k >= k_chunks || sc_row_start >= sc_rows) return;

    const int remaining_rows = sc_rows - sc_row_start;
    const int rows_this_block = remaining_rows < ROWS_PER_BLOCK ? remaining_rows : ROWS_PER_BLOCK;
    const float denom = col_sg[k / 2];
    for (int idx = threadIdx.x; idx < rows_this_block * 512; idx += BLOCK_SIZE) {
        const int local_row = idx / 512;
        const int i = idx - local_row * 512;
        const int sc_row = sc_row_start + local_row;
        const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

template <int ROWS_PER_BLOCK>
static void launch_rescale_col_sc_split3_rows(
    dim3 grid,
    cudaStream_t stream,
    __nv_fp8_e4m3* col_sc_0,
    const float* col_sg_chunk_0,
    const float* col_sg_0,
    int k_chunks_0,
    int sc_rows,
    int sg_rows,
    __nv_fp8_e4m3* col_sc_1,
    const float* col_sg_chunk_1,
    const float* col_sg_1,
    int k_chunks_1,
    __nv_fp8_e4m3* col_sc_2,
    const float* col_sg_chunk_2,
    const float* col_sg_2,
    int k_chunks_2
) {
    rescale_col_sc_split3_rows_kernel<ROWS_PER_BLOCK><<<grid, 256, 0, stream>>>(
        col_sc_0,
        col_sg_chunk_0,
        col_sg_0,
        k_chunks_0,
        sc_rows,
        sg_rows,
        col_sc_1,
        col_sg_chunk_1,
        col_sg_1,
        k_chunks_1,
        col_sc_2,
        col_sg_chunk_2,
        col_sg_2,
        k_chunks_2);
}

template <int ROW_COLS_PER_BLOCK, int COL_ROWS_PER_BLOCK, int BLOCK_SIZE = 256>
__global__ void rescale_row_col_sc_split3_kernel(
    int row_tasks,
    int row_tiles,
    int row_col_blocks,
    int col_row_blocks,
    __nv_fp8_e4m3* __restrict__ row_sc_0,
    const float* __restrict__ row_sg_chunk_0,
    const float* __restrict__ row_sg_0,
    int row_chunks_0,
    int row_sc_cols_0,
    int row_sg_cols_0,
    int64_t row_sc_stride0_0,
    int64_t row_sc_stride1_0,
    __nv_fp8_e4m3* __restrict__ row_sc_1,
    const float* __restrict__ row_sg_chunk_1,
    const float* __restrict__ row_sg_1,
    int row_chunks_1,
    int row_sc_cols_1,
    int row_sg_cols_1,
    int64_t row_sc_stride0_1,
    int64_t row_sc_stride1_1,
    __nv_fp8_e4m3* __restrict__ row_sc_2,
    const float* __restrict__ row_sg_chunk_2,
    const float* __restrict__ row_sg_2,
    int row_chunks_2,
    int row_sc_cols_2,
    int row_sg_cols_2,
    int64_t row_sc_stride0_2,
    int64_t row_sc_stride1_2,
    __nv_fp8_e4m3* __restrict__ col_sc_0,
    const float* __restrict__ col_sg_chunk_0,
    const float* __restrict__ col_sg_0,
    int k_chunks_0,
    int col_sc_rows,
    int col_sg_rows,
    __nv_fp8_e4m3* __restrict__ col_sc_1,
    const float* __restrict__ col_sg_chunk_1,
    const float* __restrict__ col_sg_1,
    int k_chunks_1,
    __nv_fp8_e4m3* __restrict__ col_sc_2,
    const float* __restrict__ col_sg_chunk_2,
    const float* __restrict__ col_sg_2,
    int k_chunks_2
) {
    const int task = blockIdx.x;
    if (task < row_tasks) {
        const int split = task % 3;
        const int rem0 = task / 3;
        const int sc_col_start = (rem0 % row_col_blocks) * ROW_COLS_PER_BLOCK;
        const int tile = rem0 / row_col_blocks;
        if (tile >= row_tiles) return;

        __nv_fp8_e4m3* row_sc;
        const float* row_sg_chunk;
        const float* row_sg;
        int row_chunks;
        int sc_cols;
        int sg_cols;
        int64_t row_sc_stride0;
        int64_t row_sc_stride1;
        if (split == 0) {
            row_sc = row_sc_0; row_sg_chunk = row_sg_chunk_0; row_sg = row_sg_0;
            row_chunks = row_chunks_0; sc_cols = row_sc_cols_0; sg_cols = row_sg_cols_0;
            row_sc_stride0 = row_sc_stride0_0; row_sc_stride1 = row_sc_stride1_0;
        } else if (split == 1) {
            row_sc = row_sc_1; row_sg_chunk = row_sg_chunk_1; row_sg = row_sg_1;
            row_chunks = row_chunks_1; sc_cols = row_sc_cols_1; sg_cols = row_sg_cols_1;
            row_sc_stride0 = row_sc_stride0_1; row_sc_stride1 = row_sc_stride1_1;
        } else {
            row_sc = row_sc_2; row_sg_chunk = row_sg_chunk_2; row_sg = row_sg_2;
            row_chunks = row_chunks_2; sc_cols = row_sc_cols_2; sg_cols = row_sg_cols_2;
            row_sc_stride0 = row_sc_stride0_2; row_sc_stride1 = row_sc_stride1_2;
        }
        const int row0 = tile * 2;
        if (row0 >= row_chunks || sc_col_start >= sc_cols) return;

        const int rows_in_tile = min(2, row_chunks - row0);
        const int cols_this_block = min(ROW_COLS_PER_BLOCK, sc_cols - sc_col_start);
        const float denom = row_sg[tile];
        const int total = rows_in_tile * cols_this_block * 512;
        for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
            const int local_row = idx / (cols_this_block * 512);
            const int rem = idx - local_row * cols_this_block * 512;
            const int local_col = rem / 512;
            const int i = rem - local_col * 512;
            const int row = row0 + local_row;
            const int sc_col = sc_col_start + local_col;
            const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
            const float ratio = localcta_outer_sg_ratio(numer, denom);
            const int64_t base = static_cast<int64_t>(row) * row_sc_stride0 +
                                 static_cast<int64_t>(sc_col) * row_sc_stride1;
            const float v = static_cast<float>(row_sc[base + i]);
            row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
        }
        return;
    }

    const int ctask = task - row_tasks;
    const int split = ctask % 3;
    const int rem0 = ctask / 3;
    const int sc_row_start = (rem0 % col_row_blocks) * COL_ROWS_PER_BLOCK;
    const int k = rem0 / col_row_blocks;

    __nv_fp8_e4m3* col_sc;
    const float* col_sg_chunk;
    const float* col_sg;
    int k_chunks;
    if (split == 0) {
        col_sc = col_sc_0; col_sg_chunk = col_sg_chunk_0; col_sg = col_sg_0; k_chunks = k_chunks_0;
    } else if (split == 1) {
        col_sc = col_sc_1; col_sg_chunk = col_sg_chunk_1; col_sg = col_sg_1; k_chunks = k_chunks_1;
    } else {
        col_sc = col_sc_2; col_sg_chunk = col_sg_chunk_2; col_sg = col_sg_2; k_chunks = k_chunks_2;
    }
    if (k >= k_chunks || sc_row_start >= col_sc_rows) return;

    const int remaining_rows = col_sc_rows - sc_row_start;
    const int rows_this_block = remaining_rows < COL_ROWS_PER_BLOCK ? remaining_rows : COL_ROWS_PER_BLOCK;
    const float denom = col_sg[k / 2];
    for (int idx = threadIdx.x; idx < rows_this_block * 512; idx += BLOCK_SIZE) {
        const int local_row = idx / 512;
        const int i = idx - local_row * 512;
        const int sc_row = sc_row_start + local_row;
        const float numer = col_sg_chunk[k * col_sg_rows + (sc_row / 2)];
        const float ratio = localcta_outer_sg_ratio(numer, denom);
        const int64_t base = ((int64_t)k * col_sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
    }
}

struct LocalCTACachedInfo {
    int num_sms = 0;
    int max_bps = 0;
    int max_bps_t = 0;
    bool initialized = false;
};

static LocalCTACachedInfo& get_localcta_cached_info() {
    static LocalCTACachedInfo info;
    if (!info.initialized) {
        using namespace tk_localcta;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&info.num_sms, cudaDevAttrMultiProcessorCount, dev);

        const int dshmem = shmem_size<false>();
        auto kernel = fused_localcta_quantize_kernel<false, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.max_bps, kernel, THREADS, dshmem);

        const int dshmem_t = shmem_size<true>();
        auto kernel_t = fused_localcta_quantize_kernel<true, true>;
        cudaFuncSetAttribute(kernel_t, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.max_bps_t, kernel_t, THREADS, dshmem_t);

        info.initialized = true;
    }
    return info;
}

struct LocalCTA2PreparedTuning {
    int threads = 192;
    int pipe_depth = 2;
    bool shared_amax = true;
};

struct LocalCTA2PreparedSplit2Tuning {
    int threads = 384;
    int pipe_depth = 2;
    bool shared_amax = false;
};

struct LocalCTA1PreparedTuning {
    int threads = 160;
    int pipe_depth = 1;
};

static LocalCTA2PreparedTuning& get_localcta2_prepared_tuning() {
    static LocalCTA2PreparedTuning tuning;
    return tuning;
}

static LocalCTA2PreparedSplit2Tuning& get_localcta2_prepared_split2_tuning() {
    static LocalCTA2PreparedSplit2Tuning tuning;
    return tuning;
}

static LocalCTA1PreparedTuning& get_localcta1_prepared_tuning() {
    static LocalCTA1PreparedTuning tuning;
    return tuning;
}

struct LocalCTAPersistentCounter {
    int device_index = -1;
    uintptr_t stream_key = 0;
    torch::Tensor counter;
};

static torch::Tensor& get_localcta_persistent_counter(torch::Device device) {
    // Persistent kernels reset and consume this counter asynchronously.  A
    // device-global counter lets a memset/kernel pair on one CUDA stream race
    // with a pair launched on another stream, which can skip or duplicate
    // tiles.  Keep one counter per host thread, device, and CUDA stream so
    // same-stream reuse remains allocation-free while independent streams
    // cannot reset each other's in-flight work.
    static thread_local std::vector<LocalCTAPersistentCounter> entries;
    const int device_index = device.index();
    TORCH_CHECK(device_index >= 0, "device must have a concrete CUDA index");
    const auto stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    const uintptr_t stream_key = reinterpret_cast<uintptr_t>(stream);
    for (auto& entry : entries) {
        if (entry.device_index == device_index &&
            entry.stream_key == stream_key) {
            return entry.counter;
        }
    }

    LocalCTAPersistentCounter entry;
    entry.device_index = device_index;
    entry.stream_key = stream_key;
    entry.counter = torch::empty(
        {1}, torch::dtype(torch::kInt32).device(device));
    entries.push_back(std::move(entry));
    return entries.back().counter;
}

struct LocalCTAAtomicScratch {
    int device_index = -1;
    uintptr_t stream_key = 0;
    int64_t M = 0;
    int64_t K = 0;
    bool return_transpose = false;
    torch::Tensor row_sg_chunk;
    torch::Tensor col_sg_chunk;
    torch::Tensor work_counter;
};

static LocalCTAAtomicScratch& get_localcta_atomic_scratch(
    torch::Device device,
    cudaStream_t stream,
    int64_t M,
    int64_t K,
    bool return_transpose
) {
    static thread_local std::vector<LocalCTAAtomicScratch> entries;
    const int device_index = device.index();
    const uintptr_t stream_key = reinterpret_cast<uintptr_t>(stream);
    for (auto& entry : entries) {
        if (entry.device_index == device_index &&
            entry.stream_key == stream_key &&
            entry.M == M && entry.K == K &&
            entry.return_transpose == return_transpose) {
            return entry;
        }
    }

    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);
    LocalCTAAtomicScratch entry;
    entry.device_index = device_index;
    entry.stream_key = stream_key;
    entry.M = M;
    entry.K = K;
    entry.return_transpose = return_transpose;
    entry.row_sg_chunk = torch::empty({M / 128, K / 128}, opts_f32);
    entry.col_sg_chunk = return_transpose
        ? torch::empty({K / 128, M / 128}, opts_f32)
        : torch::empty({0}, opts_f32);
    entry.work_counter = torch::empty({1}, opts_i32);
    entries.push_back(std::move(entry));
    return entries.back();
}

static float& get_localcta_global_scale_num_host() {
    static float value = tk_localcta::LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM;
    return value;
}

static bool use_localcta_fused_direct_experimental() {
    const char* value = std::getenv("USE_TK_LOCALCTA_FUSED_DIRECT");
    return value != nullptr && std::string(value) == "1";
}

static bool use_localcta_v4_tuned_fused_split2() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_TUNED_FUSED");
    return value != nullptr && std::string(value) == "1";
}

static bool use_localcta_v4_direct_cluster_split2() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_DIRECT_CLUSTER");
    return value != nullptr && std::string(value) == "1";
}

static bool use_localcta_v4_direct_strict_split2() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_DIRECT_STRICT_SPLIT2");
    return value != nullptr && std::string(value) == "1";
}

static bool use_localcta_v4_tuned_strict_split2() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_TUNED_STRICT_SPLIT2");
    return value != nullptr && std::string(value) == "1";
}

static bool use_localcta_v4_split2_precompute_amax() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT2_PRECOMPUTE_AMAX");
    return value == nullptr || std::string(value) != "0";
}

static bool use_localcta_v4_split2_prefinalize_outer_sg() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT2_PREFINALIZE_OUTER_SG");
    return value == nullptr || std::string(value) != "0";
}

static bool use_localcta_v4_delayed_collect_amax() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_DELAYED_COLLECT_AMAX");
    return value == nullptr || std::string(value) != "0";
}

static bool localcta_env_flag(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    const std::string mode(value);
    return !(mode == "0" || mode == "false" || mode == "False" ||
             mode == "off" || mode == "OFF" || mode == "no" || mode == "NO");
}

static int64_t localcta_env_int64(const char* name, int64_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    try {
        return std::stoll(value);
    } catch (...) {
        return default_value;
    }
}

static float localcta_env_float(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    try {
        return std::stof(value);
    } catch (...) {
        return default_value;
    }
}

static bool use_localcta_v4_final_sg_producer() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER", true);
}

static bool use_localcta_v4_atomic_final_sg_producer() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER", false);
}

static bool use_localcta_v4_fused_atomic_init() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_FUSED_ATOMIC_INIT", false);
}

static bool use_localcta_v4_reuse_atomic_scratch() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_REUSE_ATOMIC_SCRATCH", false);
}

static bool use_localcta_v4_final_sg_view_splits() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_FINAL_SG_VIEW_SPLITS", false);
}

static bool use_localcta_v4_split3_direct_final_sg_scan() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SPLIT3_DIRECT_FINAL_SG_SCAN", true);
}

static bool use_localcta_v4_final_sg_opt_direct_final_scan() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_FINAL_SG_OPT_DIRECT_FINAL_SCAN", true);
}

static bool use_localcta_v4_split3_fused_sg_reduce() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SPLIT3_FUSED_SG_REDUCE", true);
}

static bool use_localcta_v4_silu_final_sg_producer() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SILU_FINAL_SG_PRODUCER", false);
}

static bool use_localcta_v4_silu_atomic_final_sg_producer() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SILU_ATOMIC_FINAL_SG_PRODUCER", true);
}

static bool use_localcta_v4_gemm_virtual_rescale() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE", false);
}

static bool use_localcta_v4_gemm_virtual_rescale_force_raw() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE_FORCE_RAW", false);
}

static bool use_localcta_v4_fast_data_sr() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_FAST_DATA_SR", false);
}

static int localcta_current_device_sm_count() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 0;
    }
    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return 0;
    }
    return sm_count;
}

static int localcta_v4_direct_grid_override(int default_grid, int total_tiles) {
    int grid = default_grid;
    const int64_t exact = localcta_env_int64("USE_TK_LOCALCTA_V4_DIRECT_GRID", 0);
    if (exact > 0) {
        grid = static_cast<int>(exact);
    } else {
        const float mult = localcta_env_float("USE_TK_LOCALCTA_V4_DIRECT_GRID_MULT", 1.0f);
        if (std::isfinite(mult) && mult > 0.0f && mult != 1.0f) {
            grid = static_cast<int>(std::lround(static_cast<float>(grid) * mult));
        }
        const int64_t cap = localcta_env_int64("USE_TK_LOCALCTA_V4_DIRECT_GRID_CAP", 0);
        if (cap > 0 && grid > static_cast<int>(cap)) {
            grid = static_cast<int>(cap);
        }
        const float sm_mult = localcta_env_float("USE_TK_LOCALCTA_V4_DIRECT_GRID_SM_MULT", 0.0f);
        if (std::isfinite(sm_mult) && sm_mult > 0.0f) {
            const int sm_count = localcta_current_device_sm_count();
            const int sm_cap = std::max(1, static_cast<int>(std::lround(sm_mult * sm_count)));
            if (sm_count > 0 && grid > sm_cap) {
                grid = sm_cap;
            }
        }
    }
    if (grid > total_tiles) {
        grid = total_tiles;
    }
    if (grid <= 0) {
        grid = 1;
    }
    return grid;
}

static bool use_localcta_v4_col_rht_amax_from_raw() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW", false);
}

static float localcta_v4_col_rht_amax_raw_multiplier() {
    const float mult = localcta_env_float("USE_TK_LOCALCTA_V4_COL_RHT_AMAX_RAW_MULTIPLIER", 2.0f);
    return mult > 0.0f ? mult : 2.0f;
}

static int64_t localcta_v4_gemm_virtual_rescale_max_m() {
    return localcta_env_int64("USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE_MAX_M", 0);
}

static bool use_localcta_v4_gemm_virtual_rescale_for_m(int64_t M) {
    if (!use_localcta_v4_gemm_virtual_rescale()) {
        return false;
    }
    const int64_t max_m = localcta_v4_gemm_virtual_rescale_max_m();
    return max_m <= 0 || M <= max_m;
}

static bool use_localcta_v4_split_finalize_single() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT_FINALIZE_SINGLE");
    return value != nullptr && std::string(value) == "1";
}

static bool use_localcta_v4_nhsd_reduced_warp_finalize() {
    return localcta_env_flag(
        "USE_TK_LOCALCTA_V4_NHSD_REDUCED_WARP_FINALIZE", false);
}

static bool use_localcta_v4_nhsd_combined_sg_reduce() {
    return localcta_env_flag(
        "USE_TK_LOCALCTA_V4_NHSD_COMBINED_SG_REDUCE", true);
}

static bool use_localcta_v4_split3_fold_row_sg_in_producer() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_FOLD_ROW_SG_IN_PRODUCER");
    return value != nullptr && std::string(value) == "1";
}

static bool use_localcta_v4_fused_silu_raw() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_FUSED_SILU_RAW");
    return value != nullptr && std::string(value) == "1";
}

static int localcta_v4_silu_h3_ring_slots() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SILU_H3_RING");
    if (value == nullptr) {
        return 1;
    }
    const std::string mode(value);
    if (mode == "0") {
        return 0;
    }
    if (mode == "1") {
        return 1;
    }
    return 0;
}

static bool use_localcta_v4_silu_parallel_row_col() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SILU_PARALLEL_ROW_COL", true);
}

static bool use_localcta_v4_silu_fast_divide() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SILU_FAST_DIVIDE", true);
}

static bool use_localcta_v4_silu_deriv_fast_divide() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SILU_DERIV_FAST_DIVIDE", true);
}

enum class V3MultiInputQuantMode {
    OneCall,
    Loop,
    SplitFinalize,
    ColSplitFinalize,
    RowColSplitFinalize,
};

enum class V3ContractMode {
    OuterScale,
    TileGrid256,
};

static V3MultiInputQuantMode get_v3_multiinput_quant_mode() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_MULTIINPUT_QUANT");
    if (value == nullptr) {
        return V3MultiInputQuantMode::OneCall;
    }
    const std::string mode(value);
    if (mode == "0" || mode == "loop") {
        return V3MultiInputQuantMode::Loop;
    }
    if (mode == "splitfinal" || mode == "split_finalize") {
        return V3MultiInputQuantMode::SplitFinalize;
    }
    if (mode == "colsplitfinal" || mode == "col_splitfinal" || mode == "col_split_finalize") {
        return V3MultiInputQuantMode::ColSplitFinalize;
    }
    if (mode == "rowcolsplitfinal" || mode == "row_colsplitfinal" ||
        mode == "row_col_splitfinal" || mode == "row_col_split_finalize") {
        return V3MultiInputQuantMode::RowColSplitFinalize;
    }
    return V3MultiInputQuantMode::OneCall;
}

static V3MultiInputQuantMode get_v3_split2_multiinput_quant_mode() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_SPLIT2_MULTIINPUT_QUANT");
    if (value == nullptr || *value == '\0') {
        return V3MultiInputQuantMode::SplitFinalize;
    }
    const std::string mode(value);
    if (mode == "0" || mode == "loop") {
        return V3MultiInputQuantMode::Loop;
    }
    if (mode == "splitfinal" || mode == "split_finalize") {
        return V3MultiInputQuantMode::SplitFinalize;
    }
    if (mode == "colsplitfinal" || mode == "col_splitfinal" || mode == "col_split_finalize") {
        return V3MultiInputQuantMode::ColSplitFinalize;
    }
    if (mode == "rowcolsplitfinal" || mode == "row_colsplitfinal" ||
        mode == "row_col_splitfinal" || mode == "row_col_split_finalize") {
        return V3MultiInputQuantMode::RowColSplitFinalize;
    }
    return V3MultiInputQuantMode::OneCall;
}

static V3MultiInputQuantMode get_v3_split3_multiinput_quant_mode() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_SPLIT3_MULTIINPUT_QUANT");
    if (value == nullptr || *value == '\0') {
        return get_v3_multiinput_quant_mode();
    }
    const std::string mode(value);
    if (mode == "0" || mode == "loop") {
        return V3MultiInputQuantMode::Loop;
    }
    if (mode == "splitfinal" || mode == "split_finalize") {
        return V3MultiInputQuantMode::SplitFinalize;
    }
    if (mode == "colsplitfinal" || mode == "col_splitfinal" || mode == "col_split_finalize") {
        return V3MultiInputQuantMode::ColSplitFinalize;
    }
    if (mode == "rowcolsplitfinal" || mode == "row_colsplitfinal" ||
        mode == "row_col_splitfinal" || mode == "row_col_split_finalize") {
        return V3MultiInputQuantMode::RowColSplitFinalize;
    }
    return V3MultiInputQuantMode::OneCall;
}

static V3ContractMode get_v3_contract_mode() {
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

static int get_v3_split2_raw_persistent_divisor() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_SPLIT2_RAW_PERSISTENT_DIV");
    if (value == nullptr || *value == '\0') {
        return 1;
    }
    const int divisor = std::atoi(value);
    return divisor > 0 ? divisor : 1;
}

static int parse_v4_scan_threads(const char* value, int fallback) {
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    const int threads = std::atoi(value);
    if (threads == 96 || threads == 128 || threads == 160 || threads == 192 || threads == 256) {
        return threads;
    }
    return fallback;
}

static int get_v4_split3_scan_threads() {
    return parse_v4_scan_threads(std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_SCAN_THREADS"), 96);
}

static int get_v4_split3_rope_scan_threads() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_ROPE_SCAN_THREADS");
    if (value == nullptr || *value == '\0') {
        return 128;
    }
    return parse_v4_scan_threads(value, get_v4_split3_scan_threads());
}

static int get_v4_final_sg_opt_scan_threads(bool col_rht) {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_FINAL_SG_SCAN_THREADS");
    if (value != nullptr && *value != '\0') {
        return parse_v4_scan_threads(value, col_rht ? 256 : get_v4_split3_scan_threads());
    }
    if (std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_SCAN_THREADS") != nullptr) {
        return get_v4_split3_scan_threads();
    }
    return col_rht ? 256 : get_v4_split3_scan_threads();
}

static bool use_v4_split3_rope_materialize_rotated() {
    return localcta_env_flag("USE_TK_LOCALCTA_V4_SPLIT3_ROPE_MATERIALIZE_ROTATED", false);
}

static int get_v3_split2_finalize_block_size() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_SPLIT2_FINALIZE_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 64;
    }
    const int block = std::atoi(value);
    return block == 64 ? 64 : (block == 128 ? 128 : 256);
}

static int get_v3_col_finalize_rows_per_block() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V3_COL_FINALIZE_ROWS_PER_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 32;
    }
    const int rows = std::atoi(value);
    return rows > 0 ? rows : 32;
}

static int get_v4_split3_col_rescale_rows_per_block() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_COL_RESCALE_ROWS_PER_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 8;
    }
    const int rows = std::atoi(value);
    if (rows <= 1) return 1;
    if (rows <= 2) return 2;
    if (rows <= 4) return 4;
    if (rows <= 8) return 8;
    return 16;
}

static int get_v4_split3_row_finalize_block_size() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_ROW_FINALIZE_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 512;
    }
    const int block = std::atoi(value);
    if (block <= 128) return 128;
    if (block <= 256) return 256;
    return 512;
}

static bool use_v4_split3_row_split_rescale() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_ROW_SPLIT_RESCALE");
    if (value == nullptr || *value == '\0') {
        return true;
    }
    return value != nullptr && std::atoi(value) != 0;
}

static int get_v4_split3_row_rescale_cols_per_block() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_ROW_RESCALE_COLS_PER_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 2;
    }
    const int cols = std::atoi(value);
    if (cols <= 1) return 1;
    if (cols <= 2) return 2;
    if (cols <= 4) return 4;
    return 8;
}

static bool use_v4_split3_emit_row_sg_cat() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_EMIT_ROW_SG_CAT");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    return std::atoi(value) != 0;
}

static bool use_v4_split3_combined_rescale() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_COMBINED_RESCALE");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    return std::atoi(value) != 0;
}

static bool use_v4_split3_two_phase_quant() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_SPLIT3_TWO_PHASE");
    return value == nullptr || *value == '\0' || std::atoi(value) != 0;
}

static bool use_v4_direct_swizzled_scales() {
    const char* value = std::getenv("USE_TK_LOCALCTA_V4_DIRECT_SWIZZLED_SCALES");
    return value == nullptr || *value == '\0' || std::atoi(value) != 0;
}

void tk_localcta_set_global_scale_num(float value) {
    TORCH_CHECK(std::isfinite(value) && value > 0.0f, "global scale number must be finite and > 0");
    auto err = cudaMemcpyToSymbol(tk_localcta::kLocalCTAGlobalScaleNum, &value, sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for localCTA global scale num: ",
                cudaGetErrorString(err));
    get_localcta_global_scale_num_host() = value;
}

float tk_localcta_get_global_scale_num() {
    return get_localcta_global_scale_num_host();
}

void tk_localcta_reset_global_scale_num() {
    tk_localcta_set_global_scale_num(tk_localcta::LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM);
}

static bool should_use_localcta2_prepared_auto(int64_t M, int64_t K) {
    using namespace tk_localcta;
    const int blocks_Y = static_cast<int>((M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X = static_cast<int>((K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X);
    if (blocks_Y < 2) {
        return false;
    }
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    return total_macro_tiles > 0 && total_macro_tiles <= 1024;
}

static bool should_use_localcta1_prepared_auto(int64_t M, int64_t K, bool return_transpose) {
    const char* force_env = std::getenv("USE_TK_LOCALCTA_V4_FORCE_1CTA_PREPARED_TUNED");
    const bool force_return_transpose_tuned =
        force_env != nullptr && !(force_env[0] == '0' && force_env[1] == '\0');
    if (return_transpose && !force_return_transpose_tuned) {
        return false;
    }
    using namespace tk_localcta;
    const int blocks_Y = static_cast<int>((M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X = static_cast<int>((K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X);
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    return total_macro_tiles > 1024;
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX>
static void launch_localcta_quant_2cta_prepared_tuned(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_2cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_kernel_2cta_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, SHARED_AMAX, RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_clusters = max_bps * num_sms / 2;
    if (num_clusters > total_macro_tiles) {
        num_clusters = total_macro_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(TOTAL_THREADS, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, K, blocks_X, blocks_Y, total_macro_tiles);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchKernelEx failed for tuned localCTA 2-CTA prepared quant: ",
                cudaGetErrorString(err));
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant_2cta_prepared_tuned_dispatch(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    const auto cfg = get_localcta2_prepared_tuning();
    if (cfg.threads == 160 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else {
        TORCH_CHECK(false, "Unsupported localCTA2 prepared tuning config: threads=", cfg.threads,
                    " pipe_depth=", cfg.pipe_depth, " shared_amax=", cfg.shared_amax);
    }
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH,
          bool WRITE_ROW_RAW = false, bool WRITE_COL_RAW = false>
static void launch_localcta_quant_prepared_tuned(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_row_raw,
    const CUtensorMap &tmap_sc_col_prepared,
    const CUtensorMap &tmap_sc_col_raw,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_1cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_kernel_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, RETURN_TRANSPOSE, ENCODE_CENTRIC, WRITE_ROW_RAW, WRITE_COL_RAW>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_persistent = max_bps * num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    kernel<<<num_persistent, TOTAL_THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
        row_sg_ptr, col_sg_ptr,
        M, K, blocks_X, total_tiles);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, bool WRITE_ROW_RAW = false, bool WRITE_COL_RAW = false>
static void launch_localcta_quant_prepared_tuned_dispatch(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_row_raw,
    const CUtensorMap &tmap_sc_col_prepared,
    const CUtensorMap &tmap_sc_col_raw,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    const auto cfg = get_localcta1_prepared_tuning();
    if (cfg.threads == 160 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, WRITE_ROW_RAW, WRITE_COL_RAW>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2, WRITE_ROW_RAW, WRITE_COL_RAW>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1, WRITE_ROW_RAW, WRITE_COL_RAW>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2, WRITE_ROW_RAW, WRITE_COL_RAW>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1, WRITE_ROW_RAW, WRITE_COL_RAW>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2, WRITE_ROW_RAW, WRITE_COL_RAW>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else {
        TORCH_CHECK(false, "Unsupported localCTA1 prepared tuning config: threads=", cfg.threads,
                    " pipe_depth=", cfg.pipe_depth);
    }
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH,
          bool WRITE_COL_RAW = false,
          bool DATA_SR = false, bool FAST_DATA_SR = false, bool SCALE_SR = false,
          bool ROW_WITH_RHT = false, bool COL_WITH_RHT = false,
          bool COL_RHT_AMAX_FROM_RAW = false,
          bool WITH_RANDOM_SIGN_MASK = false>
static void launch_localcta_sqrelu_prepared_tuned(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    const CUtensorMap &tmap_sc_col_raw,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    float col_rht_raw_amax_multiplier,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_1cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_sqrelu_quantize_kernel_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, RETURN_TRANSPOSE, ENCODE_CENTRIC, WRITE_COL_RAW,
        DATA_SR, FAST_DATA_SR, SCALE_SR, ROW_WITH_RHT, COL_WITH_RHT,
        COL_RHT_AMAX_FROM_RAW, WITH_RANDOM_SIGN_MASK>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_persistent = max_bps * num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    kernel<<<num_persistent, TOTAL_THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared, tmap_sc_col_raw,
        row_sg_ptr, col_sg_ptr,
        M, K, blocks_X, total_tiles,
        rng_seed, rng_subsequence_base, col_rht_raw_amax_multiplier);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, bool WRITE_COL_RAW = false>
static void launch_localcta_sqrelu_prepared_tuned_dispatch(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    const CUtensorMap &tmap_sc_col_raw,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    bool data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream
) {
    TORCH_CHECK(!random_sign, "square-ReLU localCTA fused producer does not support random RHT signs yet");
#define LOCALCTA_SQRELU_LAUNCH(DATA, SCALE, ROW_RHT, COL_RHT, RAW_COL_AMAX) \
    do { \
        TORCH_CHECK(cfg.threads == 160 && cfg.pipe_depth == 1, \
                    "square-ReLU RHT/SR producer currently supports only localCTA1 160x1 tuning"); \
        launch_localcta_sqrelu_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, WRITE_COL_RAW, DATA, false, SCALE, ROW_RHT, COL_RHT, RAW_COL_AMAX, false>( \
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared, tmap_sc_col_raw, \
            row_sg_ptr, col_sg_ptr, M, K, rng_seed, rng_subsequence_base, raw_col_rht_amax_multiplier, stream); \
    } while (0)
    const auto cfg = get_localcta1_prepared_tuning();
    const bool raw_col_rht_amax = col_rht && use_localcta_v4_col_rht_amax_from_raw();
    const float raw_col_rht_amax_multiplier =
        raw_col_rht_amax ? localcta_v4_col_rht_amax_raw_multiplier() : 1.0f;

    const int mask =
        (data_sr ? 1 : 0) |
        (scale_sr ? 2 : 0) |
        (row_rht ? 4 : 0) |
        (col_rht ? 8 : 0);
    switch (mask) {
        case 0: LOCALCTA_SQRELU_LAUNCH(false, false, false, false, false); break;
        case 4: LOCALCTA_SQRELU_LAUNCH(false, false, true,  false, false); break;
        case 8:
            if (raw_col_rht_amax) {
                LOCALCTA_SQRELU_LAUNCH(false, false, false, true, true);
            } else {
                LOCALCTA_SQRELU_LAUNCH(false, false, false, true, false);
            }
            break;
        case 7: LOCALCTA_SQRELU_LAUNCH(true, true, true,  false, false); break;
        case 11:
            if (raw_col_rht_amax) {
                LOCALCTA_SQRELU_LAUNCH(true, true, false, true, true);
            } else {
                LOCALCTA_SQRELU_LAUNCH(true, true, false, true, false);
            }
            break;
        default: TORCH_CHECK(false, "Unsupported square-ReLU localCTA RHT/SR combination");
    }
#undef LOCALCTA_SQRELU_LAUNCH
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH,
          bool WRITE_ROW_RAW = false, bool WRITE_COL_RAW = false,
          bool DATA_SR = false, bool FAST_DATA_SR = false, bool SCALE_SR = false,
          bool ROW_WITH_RHT = false, bool COL_WITH_RHT = false,
          bool COL_RHT_AMAX_FROM_RAW = false,
          bool WITH_RANDOM_SIGN_MASK = false>
static void launch_localcta_sqrelu_deriv_prepared_tuned(
    const CUtensorMap &tmap_dh,
    const CUtensorMap &tmap_h1,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_row_raw,
    const CUtensorMap &tmap_sc_col_prepared,
    const CUtensorMap &tmap_sc_col_raw,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    float col_rht_raw_amax_multiplier,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_split2_dual_1cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_sqrelu_deriv_quantize_kernel_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, RETURN_TRANSPOSE, ENCODE_CENTRIC,
        WRITE_ROW_RAW, WRITE_COL_RAW,
        DATA_SR, FAST_DATA_SR, SCALE_SR, ROW_WITH_RHT, COL_WITH_RHT,
        COL_RHT_AMAX_FROM_RAW, WITH_RANDOM_SIGN_MASK>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_persistent = max_bps * num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    kernel<<<num_persistent, TOTAL_THREADS, dshmem, stream>>>(
        tmap_dh, tmap_h1, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw,
        row_sg_ptr, col_sg_ptr,
        M, K, blocks_X, total_tiles,
        rng_seed, rng_subsequence_base, col_rht_raw_amax_multiplier);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC,
          bool WRITE_ROW_RAW = false, bool WRITE_COL_RAW = false>
static void launch_localcta_sqrelu_deriv_prepared_tuned_dispatch(
    const CUtensorMap &tmap_dh,
    const CUtensorMap &tmap_h1,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_row_raw,
    const CUtensorMap &tmap_sc_col_prepared,
    const CUtensorMap &tmap_sc_col_raw,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    bool data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream
) {
    TORCH_CHECK(!random_sign, "square-ReLU derivative localCTA fused producer does not support random RHT signs yet");
#define LOCALCTA_SQRELU_DERIV_LAUNCH(DATA, FAST, SCALE, ROW_RHT, COL_RHT, RAW_COL_AMAX) \
    do { \
        TORCH_CHECK(cfg.threads == 160 && cfg.pipe_depth == 1, \
                    "square-ReLU derivative RHT/SR producer currently supports only localCTA1 160x1 tuning"); \
        launch_localcta_sqrelu_deriv_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, WRITE_ROW_RAW, WRITE_COL_RAW, DATA, FAST, SCALE, ROW_RHT, COL_RHT, RAW_COL_AMAX, false>( \
            tmap_dh, tmap_h1, tmap_out, tmap_out_t, \
            tmap_sc_row_prepared, tmap_sc_row_raw, tmap_sc_col_prepared, tmap_sc_col_raw, \
            row_sg_ptr, col_sg_ptr, M, K, rng_seed, rng_subsequence_base, raw_col_rht_amax_multiplier, stream); \
    } while (0)
    const auto cfg = get_localcta1_prepared_tuning();
    const bool fast_data_sr = data_sr && !scale_sr && use_localcta_v4_fast_data_sr();
    const bool raw_col_rht_amax = col_rht && use_localcta_v4_col_rht_amax_from_raw();
    const float raw_col_rht_amax_multiplier =
        raw_col_rht_amax ? localcta_v4_col_rht_amax_raw_multiplier() : 1.0f;

    const int mask =
        (data_sr ? 1 : 0) |
        (scale_sr ? 2 : 0) |
        (row_rht ? 4 : 0) |
        (col_rht ? 8 : 0);
    switch (mask) {
        case 0: LOCALCTA_SQRELU_DERIV_LAUNCH(false, false, false, false, false, false); break;
        case 4: LOCALCTA_SQRELU_DERIV_LAUNCH(false, false, false, true,  false, false); break;
        case 8:
            if (raw_col_rht_amax) {
                LOCALCTA_SQRELU_DERIV_LAUNCH(false, false, false, false, true, true);
            } else {
                LOCALCTA_SQRELU_DERIV_LAUNCH(false, false, false, false, true, false);
            }
            break;
        case 5:
            if (fast_data_sr) {
                LOCALCTA_SQRELU_DERIV_LAUNCH(true, true, false, true, false, false);
            } else {
                LOCALCTA_SQRELU_DERIV_LAUNCH(true, false, false, true, false, false);
            }
            break;
        case 9:
            if (fast_data_sr) {
                if (raw_col_rht_amax) {
                    LOCALCTA_SQRELU_DERIV_LAUNCH(true, true, false, false, true, true);
                } else {
                    LOCALCTA_SQRELU_DERIV_LAUNCH(true, true, false, false, true, false);
                }
            } else {
                if (raw_col_rht_amax) {
                    LOCALCTA_SQRELU_DERIV_LAUNCH(true, false, false, false, true, true);
                } else {
                    LOCALCTA_SQRELU_DERIV_LAUNCH(true, false, false, false, true, false);
                }
            }
            break;
        case 7: LOCALCTA_SQRELU_DERIV_LAUNCH(true, false, true, true,  false, false); break;
        case 11:
            if (raw_col_rht_amax) {
                LOCALCTA_SQRELU_DERIV_LAUNCH(true, false, true, false, true, true);
            } else {
                LOCALCTA_SQRELU_DERIV_LAUNCH(true, false, true, false, true, false);
            }
            break;
        default: TORCH_CHECK(false, "Unsupported square-ReLU derivative localCTA RHT/SR combination");
    }
#undef LOCALCTA_SQRELU_DERIV_LAUNCH
}

static void create_tma_2d(
    CUtensorMap &map, void *ptr,
    uint64_t globalY, uint64_t globalX,
    uint32_t shmemY, uint32_t shmemX,
    uint64_t strideX, size_t type_num_bits,
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
        TORCH_CHECK(handle != nullptr, "Failed to open libcuda.so.1");
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(
            dlsym(handle, "cuTensorMapEncodeTiled"));
        TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not found");
    }

    CUtensorMapDataType dataType;
    uint64_t globalDims[2] = {globalX, globalY};
    uint32_t boxDims[2] = {shmemX, shmemY};
    uint64_t globalStrides[1] = {(strideX * type_num_bits) / 8};
    uint32_t elementStrides[2] = {1, 1};

    if (type_num_bits == 16) dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (type_num_bits == 8) dataType = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if (type_num_bits == 4) dataType = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported type_num_bits: ", type_num_bits);

    auto result = fn(&map, dataType, 2, ptr,
                     globalDims, globalStrides, boxDims, elementStrides,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                     l2promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}

template <bool RETURN_TRANSPOSE>
static void create_prepared_output_tmaps_strided(
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    CUtensorMap& tmap_out,
    CUtensorMap& tmap_out_t,
    CUtensorMap& tmap_sc_row_prepared,
    CUtensorMap& tmap_sc_col_prepared
);

template <bool RETURN_TRANSPOSE>
static void create_raw_output_tmaps(
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    CUtensorMap& tmap_out,
    CUtensorMap& tmap_out_t,
    CUtensorMap& tmap_sc_row,
    CUtensorMap& tmap_sc_col
) {
    const int64_t M = row_fp4.size(0);
    const int64_t K = row_fp4.size(1) * 2;

    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 4);

    if constexpr (RETURN_TRANSPOSE) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = K / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
}

static void create_col_only_raw_output_tmaps(
    int64_t M,
    int64_t K,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    CUtensorMap& tmap_out_dummy,
    CUtensorMap& tmap_out_t,
    CUtensorMap& tmap_sc_row_dummy,
    CUtensorMap& tmap_sc_col
) {
    // The row descriptors are never dereferenced by the EMIT_ROW=false kernel.
    // Reusing the equally-sized column allocations avoids reserving full dummy
    // row tensors solely to satisfy the kernel's fixed argument list.
    create_tma_2d(tmap_out_dummy, col_fp4.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                  tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_dummy, col_sc.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_col, col_sc.data_ptr(),
                  ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
}

template <bool RETURN_TRANSPOSE>
static void create_raw_output_tmaps_strided(
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    CUtensorMap& tmap_out,
    CUtensorMap& tmap_out_t,
    CUtensorMap& tmap_sc_row,
    CUtensorMap& tmap_sc_col
) {
    const int64_t M = row_fp4.size(0);
    const int64_t K = row_fp4.size(1) * 2;
    const int64_t row_fp4_stride = row_fp4.stride(0) * 2;

    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, row_fp4_stride, 4);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t row_sc_stride_bf16 = row_sc.stride(0) / 2;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, row_sc_stride_bf16, 16);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t col_fp4_stride = col_fp4.stride(0) * 2;
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, col_fp4_stride, 4);

        const int64_t ntm_c = K / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        const int64_t col_sc_stride_bf16 = col_sc.stride(0) / 2;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, col_sc_stride_bf16, 16);
    }
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, bool SHARED_2D_WEIGHT = false>
static void launch_localcta_quant(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    bool write_raw_scales,
    bool write_prepared,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    const int dshmem = shmem_size<RETURN_TRANSPOSE>();
    auto &ci = get_localcta_cached_info();
    int num_persistent = (RETURN_TRANSPOSE ? ci.max_bps_t : ci.max_bps) * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args {
        .work_counter = work_counter,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles
    };

    auto kernel = fused_localcta_quantize_kernel<
        RETURN_TRANSPOSE, ENCODE_CENTRIC, false, false, SHARED_2D_WEIGHT>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        nullptr, nullptr,
        M, K, args, write_raw_scales, write_prepared);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, bool WITH_RMSNORM = false>
static void launch_localcta_quant_final_sg(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    float *row_sg_ptr,
    float *col_sg_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    cudaStream_t stream,
    const tk_localcta::IType *rms_gamma_ptr = nullptr,
    const float *rms_inv_rms_ptr = nullptr
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    const int dshmem = shmem_size<RETURN_TRANSPOSE>();
    auto &ci = get_localcta_cached_info();
    int num_persistent = (RETURN_TRANSPOSE ? ci.max_bps_t : ci.max_bps) * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args {
        .work_counter = work_counter,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles
    };

    alignas(64) CUtensorMap empty_sc_prepared{};
    auto kernel = fused_localcta_quantize_kernel<
        RETURN_TRANSPOSE, ENCODE_CENTRIC, true, WITH_RMSNORM>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        empty_sc_prepared, empty_sc_prepared,
        row_sg_ptr, col_sg_ptr,
        rms_gamma_ptr, rms_inv_rms_ptr,
        M, K, args, true, false);
}

template <
    bool RETURN_TRANSPOSE,
    bool ENCODE_CENTRIC,
    bool ROW_DATA_SR,
    bool COL_DATA_SR,
    bool SCALE_SR,
    bool ROW_WITH_RHT,
    bool COL_WITH_RHT,
    bool WITH_RANDOM_SIGN_MASK,
    bool WITH_RMSNORM = false,
    bool WITH_RMSNORM_SILU = false,
    bool PREFINALIZED_OUTER_SG = false,
    bool FAST_DATA_SR = false,
    bool ATOMIC_FINAL_OUTER_SG = false,
    bool FOUR_OVER_SIX_MAE = false,
    bool EMIT_ROW = true>
static void launch_localcta_quant_opt(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    const tk_localcta::IType *rms_gamma_ptr,
    const float *rms_inv_rms_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    bool write_raw_scales,
    bool write_prepared,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const unsigned long long *rng_state_ptr,
    cudaStream_t stream,
    float *row_sg_final_ptr = nullptr,
    float *col_sg_final_ptr = nullptr
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    const int dshmem = shmem_size<RETURN_TRANSPOSE>();
    auto &ci = get_localcta_cached_info();
    int num_persistent = (RETURN_TRANSPOSE ? ci.max_bps_t : ci.max_bps) * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args {
        .work_counter = work_counter,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles
    };

    auto kernel = fused_localcta_quantize_kernel_opt<
        RETURN_TRANSPOSE, ENCODE_CENTRIC, ROW_DATA_SR, COL_DATA_SR, SCALE_SR,
        ROW_WITH_RHT, COL_WITH_RHT, WITH_RANDOM_SIGN_MASK,
        WITH_RMSNORM, WITH_RMSNORM_SILU, PREFINALIZED_OUTER_SG, FAST_DATA_SR,
        ATOMIC_FINAL_OUTER_SG, FOUR_OVER_SIX_MAE, EMIT_ROW>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        row_sg_final_ptr, col_sg_final_ptr,
        rms_gamma_ptr, rms_inv_rms_ptr,
        M, K, args, write_raw_scales, write_prepared,
        rng_seed, rng_subsequence_base, rng_state_ptr);
}

template <
    bool RETURN_TRANSPOSE,
    bool ENCODE_CENTRIC,
    bool WITH_RMSNORM = false,
    bool EMIT_ROW = true,
    bool ROW_DATA_SR = false,
    bool COL_DATA_SR = false,
    bool FAST_DATA_SR = false>
static void launch_localcta_quant_four_over_six_final_sg(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    float *row_sg_ptr,
    float *col_sg_ptr,
    const tk_localcta::IType *rms_gamma_ptr,
    const float *rms_inv_rms_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    cudaStream_t stream,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    alignas(64) CUtensorMap empty_sc_prepared{};
    auto rng_state = (ROW_DATA_SR || COL_DATA_SR)
        ? make_localcta_advancing_rng_state(
              rng_seed, rng_subsequence_base, stream)
        : torch::Tensor();
    const auto *rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const unsigned long long*>(
              rng_state.data_ptr<int64_t>())
        : nullptr;
    launch_localcta_quant_opt<
        RETURN_TRANSPOSE, ENCODE_CENTRIC,
        ROW_DATA_SR, COL_DATA_SR, false, false, false, false,
        WITH_RMSNORM, false, true, FAST_DATA_SR, false, true, EMIT_ROW>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        empty_sc_prepared, empty_sc_prepared,
        row_sg_ptr, col_sg_ptr,
        rms_gamma_ptr, rms_inv_rms_ptr,
        work_counter, M, K, true, false,
        rng_seed, rng_subsequence_base, rng_state_ptr, stream);
}

template <
    bool RETURN_TRANSPOSE,
    bool ENCODE_CENTRIC,
    bool WITH_RMSNORM = false,
    bool WITH_RMSNORM_SILU = false,
    bool PREFINALIZED_OUTER_SG = false,
    bool EMIT_ROW = true>
static void launch_localcta_quant_opt_dispatch(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    const tk_localcta::IType *rms_gamma_ptr,
    const float *rms_inv_rms_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    bool write_raw_scales,
    bool write_prepared,
    bool data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream,
    torch::Tensor persistent_rng_state = torch::Tensor()
) {
    auto rng_state = (data_sr || scale_sr)
        ? make_localcta_advancing_rng_state(
              rng_seed, rng_subsequence_base, stream,
              persistent_rng_state)
        : torch::Tensor();
    const auto *rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const unsigned long long*>(
              rng_state.data_ptr<int64_t>())
        : nullptr;
#define LOCALCTA_OPT_LAUNCH(DATA, SCALE, ROW_RHT, COL_RHT, SIGN) \
    do { \
        if (fast_data_sr) { \
            launch_localcta_quant_opt< \
                RETURN_TRANSPOSE, ENCODE_CENTRIC, DATA, DATA, SCALE, ROW_RHT, COL_RHT, SIGN, \
                WITH_RMSNORM, WITH_RMSNORM_SILU, PREFINALIZED_OUTER_SG, true, false, false, EMIT_ROW>( \
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col, \
                tmap_sc_row_prepared, tmap_sc_col_prepared, row_sg_ptr, col_sg_ptr, \
                rms_gamma_ptr, rms_inv_rms_ptr, \
                work_counter, M, K, write_raw_scales, write_prepared, \
                rng_seed, rng_subsequence_base, rng_state_ptr, stream); \
        } else { \
            launch_localcta_quant_opt< \
                RETURN_TRANSPOSE, ENCODE_CENTRIC, DATA, DATA, SCALE, ROW_RHT, COL_RHT, SIGN, \
                WITH_RMSNORM, WITH_RMSNORM_SILU, PREFINALIZED_OUTER_SG, false, false, false, EMIT_ROW>( \
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col, \
                tmap_sc_row_prepared, tmap_sc_col_prepared, row_sg_ptr, col_sg_ptr, \
                rms_gamma_ptr, rms_inv_rms_ptr, \
                work_counter, M, K, write_raw_scales, write_prepared, \
                rng_seed, rng_subsequence_base, rng_state_ptr, stream); \
        } \
    } while (0)

    const bool fast_data_sr = data_sr && use_localcta_v4_fast_data_sr();
    const int mask =
        (data_sr ? 1 : 0) |
        (scale_sr ? 2 : 0) |
        (row_rht ? 4 : 0) |
        (col_rht ? 8 : 0) |
        (random_sign ? 16 : 0);
    switch (mask) {
        case 0: LOCALCTA_OPT_LAUNCH(false, false, false, false, false); break;
        case 1: LOCALCTA_OPT_LAUNCH(true, false, false, false, false); break;
        case 2: LOCALCTA_OPT_LAUNCH(false, true, false, false, false); break;
        case 3: LOCALCTA_OPT_LAUNCH(true, true, false, false, false); break;
        case 4: LOCALCTA_OPT_LAUNCH(false, false, true, false, false); break;
        case 5: LOCALCTA_OPT_LAUNCH(true, false, true, false, false); break;
        case 6: LOCALCTA_OPT_LAUNCH(false, true, true, false, false); break;
        case 7: LOCALCTA_OPT_LAUNCH(true, true, true, false, false); break;
        case 8: LOCALCTA_OPT_LAUNCH(false, false, false, true, false); break;
        case 9: LOCALCTA_OPT_LAUNCH(true, false, false, true, false); break;
        case 10: LOCALCTA_OPT_LAUNCH(false, true, false, true, false); break;
        case 11: LOCALCTA_OPT_LAUNCH(true, true, false, true, false); break;
        case 12: LOCALCTA_OPT_LAUNCH(false, false, true, true, false); break;
        case 13: LOCALCTA_OPT_LAUNCH(true, false, true, true, false); break;
        case 14: LOCALCTA_OPT_LAUNCH(false, true, true, true, false); break;
        case 15: LOCALCTA_OPT_LAUNCH(true, true, true, true, false); break;
        case 16: LOCALCTA_OPT_LAUNCH(false, false, false, false, true); break;
        case 17: LOCALCTA_OPT_LAUNCH(true, false, false, false, true); break;
        case 18: LOCALCTA_OPT_LAUNCH(false, true, false, false, true); break;
        case 19: LOCALCTA_OPT_LAUNCH(true, true, false, false, true); break;
        case 20: LOCALCTA_OPT_LAUNCH(false, false, true, false, true); break;
        case 21: LOCALCTA_OPT_LAUNCH(true, false, true, false, true); break;
        case 22: LOCALCTA_OPT_LAUNCH(false, true, true, false, true); break;
        case 23: LOCALCTA_OPT_LAUNCH(true, true, true, false, true); break;
        case 24: LOCALCTA_OPT_LAUNCH(false, false, false, true, true); break;
        case 25: LOCALCTA_OPT_LAUNCH(true, false, false, true, true); break;
        case 26: LOCALCTA_OPT_LAUNCH(false, true, false, true, true); break;
        case 27: LOCALCTA_OPT_LAUNCH(true, true, false, true, true); break;
        case 28: LOCALCTA_OPT_LAUNCH(false, false, true, true, true); break;
        case 29: LOCALCTA_OPT_LAUNCH(true, false, true, true, true); break;
        case 30: LOCALCTA_OPT_LAUNCH(false, true, true, true, true); break;
        default: LOCALCTA_OPT_LAUNCH(true, true, true, true, true); break;
    }
#undef LOCALCTA_OPT_LAUNCH
}

template <bool ENCODE_CENTRIC>
static void launch_mxfp8_row_nvfp4_col_final_sg(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out_dummy,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_dummy,
    const CUtensorMap &tmap_sc_col,
    float *row_sg_ptr,
    float *col_sg_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    bool four_over_six_mae,
    bool col_data_sr,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream
) {
    alignas(64) CUtensorMap empty_sc_prepared{};
    if (four_over_six_mae) {
        const bool fast_data_sr = col_data_sr && use_localcta_v4_fast_data_sr();
        if (col_data_sr && fast_data_sr) {
            launch_localcta_quant_four_over_six_final_sg<
                true, ENCODE_CENTRIC, false, false, false, true, true>(
                tmap_in, tmap_out_dummy, tmap_out_t,
                tmap_sc_row_dummy, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr,
                work_counter, M, K, stream,
                rng_seed, rng_subsequence_base);
        } else if (col_data_sr) {
            launch_localcta_quant_four_over_six_final_sg<
                true, ENCODE_CENTRIC, false, false, false, true, false>(
                tmap_in, tmap_out_dummy, tmap_out_t,
                tmap_sc_row_dummy, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr,
                work_counter, M, K, stream,
                rng_seed, rng_subsequence_base);
        } else {
            launch_localcta_quant_four_over_six_final_sg<
                true, ENCODE_CENTRIC, false, false>(
                tmap_in, tmap_out_dummy, tmap_out_t,
                tmap_sc_row_dummy, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr,
                work_counter, M, K, stream);
        }
        return;
    }

    launch_localcta_quant_opt_dispatch<
        true, ENCODE_CENTRIC, false, false, true, false>(
        tmap_in, tmap_out_dummy, tmap_out_t,
        tmap_sc_row_dummy, tmap_sc_col,
        empty_sc_prepared, empty_sc_prepared,
        row_sg_ptr, col_sg_ptr, nullptr, nullptr,
        work_counter, M, K, true, false,
        col_data_sr, false, false, false, false,
        rng_seed, rng_subsequence_base, stream);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant_axis_sr(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    bool row_data_sr,
    bool col_data_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream,
    torch::Tensor persistent_rng_state = torch::Tensor()
) {
    TORCH_CHECK(row_data_sr != col_data_sr,
                "axis-selective localCTA launcher requires exactly one SR orientation");
    TORCH_CHECK(RETURN_TRANSPOSE,
                "column-selective localCTA SR requires return_transpose=True");

    auto rng_state = make_localcta_advancing_rng_state(
        rng_seed, rng_subsequence_base, stream, persistent_rng_state);
    const auto *rng_state_ptr = reinterpret_cast<const unsigned long long*>(
        rng_state.data_ptr<int64_t>());
    const bool fast_data_sr = use_localcta_v4_fast_data_sr();

#define LOCALCTA_AXIS_SR_LAUNCH(ROW_SR, COL_SR, ROW_RHT, COL_RHT, RANDOM_SIGN, FAST_SR) \
    launch_localcta_quant_opt< \
        RETURN_TRANSPOSE, ENCODE_CENTRIC, ROW_SR, COL_SR, false, ROW_RHT, COL_RHT, RANDOM_SIGN, \
        false, false, false, FAST_SR, false, false, true>( \
        tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col, \
        tmap_sc_row_prepared, tmap_sc_col_prepared, row_sg_ptr, col_sg_ptr, \
        nullptr, nullptr, work_counter, M, K, true, false, \
        rng_seed, rng_subsequence_base, rng_state_ptr, stream)

    if (row_data_sr) {
        if (fast_data_sr) {
            if (col_rht) {
                LOCALCTA_AXIS_SR_LAUNCH(true, false, false, true, true, true);
            } else {
                LOCALCTA_AXIS_SR_LAUNCH(true, false, false, false, false, true);
            }
        } else if (col_rht) {
            LOCALCTA_AXIS_SR_LAUNCH(true, false, false, true, true, false);
        } else {
            LOCALCTA_AXIS_SR_LAUNCH(true, false, false, false, false, false);
        }
    } else if (fast_data_sr) {
        LOCALCTA_AXIS_SR_LAUNCH(false, true, false, false, false, true);
    } else {
        LOCALCTA_AXIS_SR_LAUNCH(false, true, false, false, false, false);
    }
#undef LOCALCTA_AXIS_SR_LAUNCH
}

static void launch_localcta_split3_quant_prepared(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_fp4_full,
    torch::Tensor row_sc_prepared_full,
    torch::Tensor col_fp4_full,
    torch::Tensor col_sc_prepared_full,
    torch::Tensor row_sg_full,
    torch::Tensor col_sg_full
) {
    using namespace tk_localcta;
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);
    const int64_t total_n = n0 + n1 + n2;

    alignas(64) CUtensorMap tmap_in0{}, tmap_in1{}, tmap_in2{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_in0, input0.data_ptr(), M, n0, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, n0, 16);
    create_tma_2d(tmap_in1, input1.data_ptr(), M, n1, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, n1, 16);
    create_tma_2d(tmap_in2, input2.data_ptr(), M, n2, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, n2, 16);
    create_tma_2d(tmap_out, row_fp4_full.data_ptr(), M, total_n,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, total_n, 4);
    create_tma_2d(tmap_out_t, col_fp4_full.data_ptr(), total_n, M,
                  tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = total_n / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared_full.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    const int64_t ntm_c = total_n / 128;
    const int64_t ntk_c = M / 64;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_col_prepared, col_sc_prepared_full.data_ptr(),
                  ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input0.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X0 = (n0 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X1 = (n1 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X2 = (n2 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X = blocks_X0 + blocks_X1 + blocks_X2;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = shmem_size<true>();
    auto& ci = get_localcta_cached_info();
    int num_persistent = ci.max_bps_t * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };

    auto kernel = fused_localcta_quantize_split3_prepared_kernel<true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in0, tmap_in1, tmap_in2,
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_full.data_ptr<float>(), col_sg_full.data_ptr<float>(),
        M, total_n, args, blocks_X0, blocks_X1);
}

__device__ __forceinline__ unsigned int split3_bf16x2_abs_max_bits(__nv_bfloat162 v) {
    const unsigned int bits = *reinterpret_cast<const unsigned int*>(&v);
    const unsigned int lo = bits & 0x7fffu;
    const unsigned int hi = (bits >> 16) & 0x7fffu;
    return lo > hi ? lo : hi;
}

__device__ __forceinline__ unsigned int split3_bf16x4_abs_max_bits(int2 packed) {
    const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&packed.x);
    const __nv_bfloat162 v1 = *reinterpret_cast<const __nv_bfloat162*>(&packed.y);
    const unsigned int max0 = split3_bf16x2_abs_max_bits(v0);
    const unsigned int max1 = split3_bf16x2_abs_max_bits(v1);
    return max0 > max1 ? max0 : max1;
}

__device__ __forceinline__ unsigned int split3_bf16x8_abs_max_bits(int4 packed) {
    const unsigned int max0 = split3_bf16x4_abs_max_bits(make_int2(packed.x, packed.y));
    const unsigned int max1 = split3_bf16x4_abs_max_bits(make_int2(packed.z, packed.w));
    return max0 > max1 ? max0 : max1;
}

__device__ __forceinline__ float split3_bf16_abs_bits_to_float(unsigned int bits) {
    union {
        unsigned int u;
        float f;
    } out;
    out.u = bits << 16;
    return out.f;
}

template <int BLOCK_SIZE = 256>
__global__ void scan_single_tile_sg_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ row_sg_chunk,
    float* __restrict__ col_sg_chunk,
    int rows,
    int cols,
    int blocks_x,
    int total_tiles,
    bool return_transpose
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= total_tiles) return;

    const int ctaid_x = chunk_id % blocks_x;
    const int ctaid_y = chunk_id / blocks_x;
    const int row_base = ctaid_y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
    const int col_base = ctaid_x * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;

    unsigned int thread_max_bits = 0;
    constexpr int total = tk_localcta::LocalCTAConfig::CHUNK_DIM_Y *
                          tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int vec = 8;
    const bool full_tile =
        (row_base + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y <= rows) &&
        (col_base + tk_localcta::LocalCTAConfig::CHUNK_DIM_X <= cols);

    if (full_tile) {
        for (int idx = threadIdx.x * vec; idx < total; idx += BLOCK_SIZE * vec) {
            const int r = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int c = idx - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int gr = row_base + r;
            const int gc = col_base + c;
            const int4 packed = *reinterpret_cast<const int4*>(input + (int64_t)gr * cols + gc);
            const unsigned int packed_max = split3_bf16x8_abs_max_bits(packed);
            thread_max_bits = packed_max > thread_max_bits ? packed_max : thread_max_bits;
        }
    } else {
        for (int idx = threadIdx.x * vec; idx < total; idx += BLOCK_SIZE * vec) {
            const int r = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int c = idx - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int gr = row_base + r;
            const int gc = col_base + c;
            if (gr < rows && gc + (vec - 1) < cols) {
                const int4 packed = *reinterpret_cast<const int4*>(input + (int64_t)gr * cols + gc);
                const unsigned int packed_max = split3_bf16x8_abs_max_bits(packed);
                thread_max_bits = packed_max > thread_max_bits ? packed_max : thread_max_bits;
            } else {
                #pragma unroll
                for (int j = 0; j < vec; ++j) {
                    const int cc = gc + j;
                    if (gr < rows && cc < cols) {
                        const unsigned int bits =
                            *reinterpret_cast<const unsigned short*>(input + (int64_t)gr * cols + cc) & 0x7fffu;
                        thread_max_bits = bits > thread_max_bits ? bits : thread_max_bits;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const unsigned int other = __shfl_xor_sync(0xffffffff, thread_max_bits, mask);
        thread_max_bits = other > thread_max_bits ? other : thread_max_bits;
    }

    __shared__ unsigned int warp_max[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[wid] = thread_max_bits;
    }
    __syncthreads();

    if (wid == 0) {
        unsigned int block_max_bits = (lane < (BLOCK_SIZE / 32)) ? warp_max[lane] : 0;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const unsigned int other = __shfl_xor_sync(0xffffffff, block_max_bits, mask);
            block_max_bits = other > block_max_bits ? other : block_max_bits;
        }
        if (lane == 0) {
            const float sg_val =
                split3_bf16_abs_bits_to_float(block_max_bits) / tk_localcta::localcta_global_scale_num();
            row_sg_chunk[ctaid_y * blocks_x + ctaid_x] = sg_val;
            if (return_transpose) {
                const int blocks_y = (rows + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y - 1) /
                                     tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
                col_sg_chunk[ctaid_x * blocks_y + ctaid_y] = sg_val;
            }
        }
    }
}

static void launch_scan_single_sg(
    torch::Tensor input,
    bool return_transpose,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk
) {
    using namespace tk_localcta;
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const int blocks_y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * blocks_x;
    if (total_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
    float* row_sg_ptr = row_sg_chunk.data_ptr<float>();
    float* col_sg_ptr = return_transpose ? col_sg_chunk.data_ptr<float>() : row_sg_ptr;
    const int scan_threads = get_v4_split3_scan_threads();
    if (scan_threads == 96) {
        scan_single_tile_sg_kernel<96><<<total_tiles, 96, 0, stream>>>(
            input_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else if (scan_threads == 160) {
        scan_single_tile_sg_kernel<160><<<total_tiles, 160, 0, stream>>>(
            input_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else if (scan_threads == 192) {
        scan_single_tile_sg_kernel<192><<<total_tiles, 192, 0, stream>>>(
            input_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else if (scan_threads == 256) {
        scan_single_tile_sg_kernel<256><<<total_tiles, 256, 0, stream>>>(
            input_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else {
        scan_single_tile_sg_kernel<128><<<total_tiles, 128, 0, stream>>>(
            input_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scan_single_tile_sg_kernel failed: ", cudaGetErrorString(err));
}

template <
    int BLOCK_SIZE = 256,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = false,
    bool DIRECT_FINAL_SG = false,
    bool WITH_RMSNORM = false>
__global__ void scan_single_tile_sg_row_opt_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ rms_gamma,
    const float* __restrict__ rms_inv_rms,
    float* __restrict__ row_sg_chunk,
    float* __restrict__ col_sg_chunk,
    int rows,
    int cols,
    int blocks_x,
    int total_tiles,
    bool return_transpose
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= total_tiles) return;

    const int ctaid_x = chunk_id % blocks_x;
    const int ctaid_y = chunk_id / blocks_x;
    const int row_base = ctaid_y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
    const int col_base = ctaid_x * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int kBlocksPerTile = (tk_localcta::LocalCTAConfig::CHUNK_DIM_Y *
                                    tk_localcta::LocalCTAConfig::CHUNK_DIM_X) /
                                   tk_localcta::SCALE_DIM;

    float row_thread_max = 0.0f;
    float col_thread_max = 0.0f;
    for (int block = threadIdx.x; block < kBlocksPerTile; block += BLOCK_SIZE) {
        const int elem = block * tk_localcta::SCALE_DIM;
        const int r = elem / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int c = elem - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int gr = row_base + r;
        const int gc = col_base + c;

        float vals[tk_localcta::ELTS_PER_THREAD];
        if (gr < rows && gc + tk_localcta::ELTS_PER_THREAD <= cols) {
            if constexpr (WITH_RMSNORM) {
                const float inv = rms_inv_rms[gr];
                const auto* block_ptr = input + static_cast<int64_t>(gr) * cols + gc;
                const auto* gamma_ptr = rms_gamma + gc;
                const uint4 lo = reinterpret_cast<const uint4*>(block_ptr)[0];
                const uint4 hi = reinterpret_cast<const uint4*>(block_ptr)[1];
                const uint4 glo = reinterpret_cast<const uint4*>(gamma_ptr)[0];
                const uint4 ghi = reinterpret_cast<const uint4*>(gamma_ptr)[1];
                const __nv_bfloat162* lo_pairs = reinterpret_cast<const __nv_bfloat162*>(&lo);
                const __nv_bfloat162* hi_pairs = reinterpret_cast<const __nv_bfloat162*>(&hi);
                const __nv_bfloat162* glo_pairs = reinterpret_cast<const __nv_bfloat162*>(&glo);
                const __nv_bfloat162* ghi_pairs = reinterpret_cast<const __nv_bfloat162*>(&ghi);
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 packed = __bfloat1622float2(lo_pairs[p]);
                    const float2 gamma = __bfloat1622float2(glo_pairs[p]);
                    vals[2 * p + 0] = rmsnorm_contract_value(packed.x, inv, gamma.x, false);
                    vals[2 * p + 1] = rmsnorm_contract_value(packed.y, inv, gamma.y, false);
                    if constexpr (!COL_WITH_RHT) {
                        col_thread_max = fmaxf(col_thread_max, fabsf(vals[2 * p + 0]));
                        col_thread_max = fmaxf(col_thread_max, fabsf(vals[2 * p + 1]));
                    }
                }
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 packed = __bfloat1622float2(hi_pairs[p]);
                    const float2 gamma = __bfloat1622float2(ghi_pairs[p]);
                    vals[8 + 2 * p + 0] = rmsnorm_contract_value(packed.x, inv, gamma.x, false);
                    vals[8 + 2 * p + 1] = rmsnorm_contract_value(packed.y, inv, gamma.y, false);
                    if constexpr (!COL_WITH_RHT) {
                        col_thread_max = fmaxf(col_thread_max, fabsf(vals[8 + 2 * p + 0]));
                        col_thread_max = fmaxf(col_thread_max, fabsf(vals[8 + 2 * p + 1]));
                    }
                }
            } else {
                const auto* block_ptr = input + static_cast<int64_t>(gr) * cols + gc;
                const uint4 lo = reinterpret_cast<const uint4*>(block_ptr)[0];
                const uint4 hi = reinterpret_cast<const uint4*>(block_ptr)[1];
                const __nv_bfloat162* lo_pairs = reinterpret_cast<const __nv_bfloat162*>(&lo);
                const __nv_bfloat162* hi_pairs = reinterpret_cast<const __nv_bfloat162*>(&hi);
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 packed = __bfloat1622float2(lo_pairs[p]);
                    vals[2 * p + 0] = packed.x;
                    vals[2 * p + 1] = packed.y;
                    if constexpr (!COL_WITH_RHT) {
                        col_thread_max = fmaxf(col_thread_max, fabsf(packed.x));
                        col_thread_max = fmaxf(col_thread_max, fabsf(packed.y));
                    }
                }
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 packed = __bfloat1622float2(hi_pairs[p]);
                    vals[8 + 2 * p + 0] = packed.x;
                    vals[8 + 2 * p + 1] = packed.y;
                    if constexpr (!COL_WITH_RHT) {
                        col_thread_max = fmaxf(col_thread_max, fabsf(packed.x));
                        col_thread_max = fmaxf(col_thread_max, fabsf(packed.y));
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < tk_localcta::ELTS_PER_THREAD; ++i) {
                vals[i] = 0.0f;
                const int cc = gc + i;
                if (gr < rows && cc < cols) {
                    vals[i] = __bfloat162float(input[(int64_t)gr * cols + cc]);
                    if constexpr (WITH_RMSNORM) {
                        vals[i] = rmsnorm_contract_value(
                            vals[i], rms_inv_rms[gr], __bfloat162float(rms_gamma[cc]), false);
                    }
                    if constexpr (!COL_WITH_RHT) {
                        col_thread_max = fmaxf(col_thread_max, fabsf(vals[i]));
                    }
                }
            }
        }

        if constexpr (ROW_WITH_RHT) {
            tk_localcta::localcta_apply_rht16_registers<false>(vals, 0u);
        }
        #pragma unroll
        for (int i = 0; i < tk_localcta::ELTS_PER_THREAD; ++i) {
            row_thread_max = fmaxf(row_thread_max, fabsf(vals[i]));
        }
    }

    if constexpr (COL_WITH_RHT) {
        constexpr int scale_rows_per_tile =
            tk_localcta::LocalCTAConfig::CHUNK_DIM_Y / tk_localcta::SCALE_DIM;
        constexpr int col_vectors =
            scale_rows_per_tile * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        for (int block = threadIdx.x; block < col_vectors; block += BLOCK_SIZE) {
            const int scale_row = block / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int c = block - scale_row * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int gr0 = row_base + scale_row * tk_localcta::SCALE_DIM;
            const int gc = col_base + c;

            float vals[tk_localcta::ELTS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < tk_localcta::ELTS_PER_THREAD; ++i) {
                const int gr = gr0 + i;
                if (gr < rows && gc < cols) {
                    vals[i] = __bfloat162float(input[(int64_t)gr * cols + gc]);
                    if constexpr (WITH_RMSNORM) {
                        vals[i] = rmsnorm_contract_value(
                            vals[i], rms_inv_rms[gr], __bfloat162float(rms_gamma[gc]), false);
                    }
                } else {
                    vals[i] = 0.0f;
                }
            }
            tk_localcta::localcta_apply_rht16_registers<false>(vals, 0u);
            #pragma unroll
            for (int i = 0; i < tk_localcta::ELTS_PER_THREAD; ++i) {
                col_thread_max = fmaxf(col_thread_max, fabsf(vals[i]));
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        row_thread_max = fmaxf(row_thread_max, __shfl_xor_sync(0xffffffff, row_thread_max, mask));
        col_thread_max = fmaxf(col_thread_max, __shfl_xor_sync(0xffffffff, col_thread_max, mask));
    }

    __shared__ float row_warp_max[BLOCK_SIZE / 32];
    __shared__ float col_warp_max[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        row_warp_max[wid] = row_thread_max;
        col_warp_max[wid] = col_thread_max;
    }
    __syncthreads();

    if (wid == 0) {
        float row_block_max = (lane < (BLOCK_SIZE / 32)) ? row_warp_max[lane] : 0.0f;
        float col_block_max = (lane < (BLOCK_SIZE / 32)) ? col_warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            row_block_max = fmaxf(row_block_max, __shfl_xor_sync(0xffffffff, row_block_max, mask));
            col_block_max = fmaxf(col_block_max, __shfl_xor_sync(0xffffffff, col_block_max, mask));
        }
        if (lane == 0) {
            const float row_sg_val = row_block_max / tk_localcta::localcta_global_scale_num();
            const float col_sg_val = col_block_max / tk_localcta::localcta_global_scale_num();
            if constexpr (DIRECT_FINAL_SG) {
                transformer_engine::atomicMaxFloat(row_sg_chunk + (ctaid_y >> 1), row_sg_val);
            } else {
                row_sg_chunk[ctaid_y * blocks_x + ctaid_x] = row_sg_val;
            }
            if (return_transpose) {
                if constexpr (DIRECT_FINAL_SG) {
                    transformer_engine::atomicMaxFloat(col_sg_chunk + (ctaid_x >> 1), col_sg_val);
                } else {
                    const int blocks_y = (rows + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y - 1) /
                                         tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
                    col_sg_chunk[ctaid_x * blocks_y + ctaid_y] = col_sg_val;
                }
            }
        }
    }
}

template <int BLOCK_SIZE, bool WITH_RMSNORM = false>
__global__ void scan_single_tile_sg_mxfp8_direct_final_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ row_mxfp8,
    uint8_t* __restrict__ row_mxsc,
    float* __restrict__ row_sg,
    float* __restrict__ col_sg,
    const __nv_bfloat16* __restrict__ rms_gamma,
    const float* __restrict__ rms_inv_rms,
    __nv_bfloat16* __restrict__ normed_output,
    int rows,
    int cols,
    int blocks_x,
    int total_tiles
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= total_tiles) {
        return;
    }

    const int ctaid_x = chunk_id % blocks_x;
    const int ctaid_y = chunk_id / blocks_x;
    const int row_base = ctaid_y * 128;
    const int col_base = ctaid_x * 128;
    constexpr int kMxBlocksPerTile = 128 * 4;
    float thread_max = 0.0f;
    __shared__ __nv_bfloat16 gamma_tile[128];

    if constexpr (WITH_RMSNORM) {
        for (int i = threadIdx.x; i < 128; i += BLOCK_SIZE) {
            gamma_tile[i] = rms_gamma[col_base + i];
        }
        __syncthreads();
    }

    for (int block = threadIdx.x; block < kMxBlocksPerTile; block += BLOCK_SIZE) {
        const int local_row = block / 4;
        const int kb = block & 3;
        const int gr = row_base + local_row;
        const int gc = col_base + kb * 32;
        if (gr >= rows || gc + 32 > cols) {
            continue;
        }

        const auto* block_ptr = input + static_cast<int64_t>(gr) * cols + gc;
        uint4 packed_input[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed_input[i] = reinterpret_cast<const uint4*>(block_ptr)[i];
        }
        auto* values = reinterpret_cast<__nv_bfloat16*>(packed_input);

        if constexpr (WITH_RMSNORM) {
            const float inv_rms = rms_inv_rms[gr];
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                values[i] = __float2bfloat16_rn(
                    __bfloat162float(values[i]) * inv_rms *
                    __bfloat162float(gamma_tile[kb * 32 + i]));
            }
            auto* normed_ptr = normed_output + static_cast<int64_t>(gr) * cols + gc;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                reinterpret_cast<uint4*>(normed_ptr)[i] = packed_input[i];
            }
        }

        float block_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            block_amax = fmaxf(
                block_amax, fabsf(__bfloat162float(values[i])));
        }
        thread_max = fmaxf(thread_max, block_amax);

        __nv_fp8_e8m0 scale;
        scale.__x = __nv_cvt_float_to_e8m0(
            fmaxf(block_amax * 0.002232142857f, 1.0e-12f),
            __NV_SATFINITE,
            cudaRoundPosInf);
        const float scale_inv = 1.0f / static_cast<float>(scale);

        uint4 packed_output[2]{};
        auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const __nv_fp8_e4m3 quantized(
                __bfloat162float(values[i]) * scale_inv);
            output_bytes[i] = quantized.__x;
        }
        auto* output_ptr = row_mxfp8 + static_cast<int64_t>(gr) * cols + gc;
        reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
        reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];

        const int scale_offset =
            chunk_id * 512 +
            (local_row % 32) * 16 +
            (local_row / 32) * 4 + kb;
        row_mxsc[scale_offset] = scale.__x;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_max = fmaxf(
            thread_max,
            __shfl_xor_sync(0xffffffff, thread_max, mask));
    }
    __shared__ float warp_max[(BLOCK_SIZE + 31) / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[wid] = thread_max;
    }
    __syncthreads();

    if (wid == 0) {
        float block_max = lane < ((BLOCK_SIZE + 31) / 32)
            ? warp_max[lane]
            : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            block_max = fmaxf(
                block_max,
                __shfl_xor_sync(0xffffffff, block_max, mask));
        }
        if (lane == 0) {
            const float sg = block_max / tk_localcta::localcta_global_scale_num();
            transformer_engine::atomicMaxFloat(row_sg + (ctaid_y >> 1), sg);
            transformer_engine::atomicMaxFloat(col_sg + (ctaid_x >> 1), sg);
        }
    }
}

__device__ __forceinline__ uint8_t mxfp4_encode_ceil_amax(float value) {
    if (value <= 1.0e-38f) {
        return 0;
    }
    const uint32_t bits = __float_as_uint(value);
    uint8_t exponent = static_cast<uint8_t>((bits >> 23) & 0xff);
    if ((bits & 0x7fffff) != 0 && exponent < 0xfe) {
        ++exponent;
    }
    return exponent;
}

__device__ __forceinline__ float mxfp4_reciprocal_e8m0(uint8_t exponent) {
    if (exponent == 0xff) {
        return __int_as_float(0x7fffffff);
    }
    if (exponent == 0xfe) {
        return __int_as_float(0x00400000);
    }
    return __int_as_float((254 - static_cast<int>(exponent)) << 23);
}

// Load each BF16 tile once and emit the two contracts needed by CCE: an
// MXFP8 or fixed-scale E4M3 row for logits and MXFP4(input.T) for backward.
template <bool DIRECT_FP8, bool WITH_RMSNORM = false>
__global__ void __launch_bounds__(256)
quantize_mxfp8_row_mxfp4_col_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ row_mxfp8,
    uint8_t* __restrict__ row_mxsc,
    uint8_t* __restrict__ col_mxfp4,
    uint8_t* __restrict__ col_mxsc,
    int rows,
    int cols,
    int blocks_x,
    int blocks_y,
    float direct_fp8_scale_inv,
    const __nv_bfloat16* __restrict__ rms_gamma = nullptr,
    const float* __restrict__ rms_inv_rms = nullptr,
    __nv_bfloat16* __restrict__ normed_output = nullptr
) {
    constexpr int kTile = 128;
    constexpr int kMxBlock = 32;
    constexpr int kMxBlocksPerTile = kTile * (kTile / kMxBlock);
    // Eight BF16 values are staged per instruction. The padded, 16-byte
    // aligned stride also avoids the worst transpose-read bank pattern.
    __shared__ __align__(16) __nv_bfloat16 tile[kTile][kTile + 8];
    __shared__ __align__(16) __nv_bfloat16 gamma_tile[kTile];

    const int tile_id = blockIdx.x;
    const int tile_x = tile_id % blocks_x;
    const int tile_y = tile_id / blocks_x;
    const int row_base = tile_y * kTile;
    const int col_base = tile_x * kTile;

    if constexpr (WITH_RMSNORM) {
        for (int col = threadIdx.x; col < kTile; col += blockDim.x) {
            gamma_tile[col] = rms_gamma[col_base + col];
        }
        __syncthreads();
    }

    constexpr int kValuesPerLoad = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int kLoadsPerRow = kTile / kValuesPerLoad;
    for (int load = threadIdx.x;
         load < kTile * kLoadsPerRow;
         load += blockDim.x) {
        const int local_row = load / kLoadsPerRow;
        const int local_col = (load % kLoadsPerRow) * kValuesPerLoad;
        const auto* input_ptr =
            input + static_cast<int64_t>(row_base + local_row) * cols +
            col_base + local_col;
        const uint4 packed_input = *reinterpret_cast<const uint4*>(input_ptr);
        if constexpr (WITH_RMSNORM) {
            uint4 packed_normed{};
            const auto* input_values =
                reinterpret_cast<const __nv_bfloat16*>(&packed_input);
            auto* normed_values =
                reinterpret_cast<__nv_bfloat16*>(&packed_normed);
            const float inv_rms = rms_inv_rms[row_base + local_row];
            #pragma unroll
            for (int i = 0; i < kValuesPerLoad; ++i) {
                normed_values[i] = __float2bfloat16_rn(
                    __bfloat162float(input_values[i]) * inv_rms *
                    __bfloat162float(gamma_tile[local_col + i]));
            }
            *reinterpret_cast<uint4*>(&tile[local_row][local_col]) =
                packed_normed;
            auto* normed_ptr =
                normed_output +
                static_cast<int64_t>(row_base + local_row) * cols +
                col_base + local_col;
            *reinterpret_cast<uint4*>(normed_ptr) = packed_normed;
        } else {
            *reinterpret_cast<uint4*>(&tile[local_row][local_col]) =
                packed_input;
        }
    }
    __syncthreads();

    for (int block = threadIdx.x; block < kMxBlocksPerTile; block += blockDim.x) {
        const int local_row = block / 4;
        const int block_k = block & 3;
        const int local_col = block_k * kMxBlock;

        float scale_inv = direct_fp8_scale_inv;
        uint8_t scale_byte = 0;
        if constexpr (!DIRECT_FP8) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < kMxBlock; ++i) {
                block_amax = fmaxf(
                    block_amax,
                    fabsf(__bfloat162float(
                        tile[local_row][local_col + i])));
            }

            __nv_fp8_e8m0 scale;
            scale.__x = __nv_cvt_float_to_e8m0(
                fmaxf(block_amax * (1.0f / 448.0f), 1.0e-12f),
                __NV_SATFINITE,
                cudaRoundPosInf);
            scale_byte = scale.__x;
            scale_inv = 1.0f / static_cast<float>(scale);
        }

        uint4 packed_output[2]{};
        auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
        #pragma unroll
        for (int i = 0; i < kMxBlock; ++i) {
            const __nv_fp8_e4m3 quantized(
                __bfloat162float(tile[local_row][local_col + i]) * scale_inv);
            output_bytes[i] = quantized.__x;
        }
        auto* output_ptr =
            row_mxfp8 + static_cast<int64_t>(row_base + local_row) * cols +
            col_base + local_col;
        reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
        reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];

        if constexpr (!DIRECT_FP8) {
            const int scale_offset =
                tile_id * 512 +
                (local_row % 32) * 16 +
                (local_row / 32) * 4 + block_k;
            row_mxsc[scale_offset] = scale_byte;
        }
    }

    for (int block = threadIdx.x; block < kMxBlocksPerTile; block += blockDim.x) {
        const int local_col = block / 4;
        const int block_k = block & 3;
        const int local_row = block_k * kMxBlock;

        float block_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < kMxBlock; ++i) {
            block_amax = fmaxf(
                block_amax,
                fabsf(__bfloat162float(tile[local_row + i][local_col])));
        }
        const uint8_t scale = mxfp4_encode_ceil_amax(block_amax);
        const float coefficient = 6.0f * mxfp4_reciprocal_e8m0(scale);

        uint4 packed_output{};
        auto* output_bytes = reinterpret_cast<uint8_t*>(&packed_output);
        #pragma unroll
        for (int i = 0; i < kMxBlock / 2; ++i) {
            const float2 values = make_float2(
                __bfloat162float(tile[local_row + 2 * i][local_col]) * coefficient,
                __bfloat162float(tile[local_row + 2 * i + 1][local_col]) * coefficient);
            output_bytes[i] = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(values, __NV_E2M1, cudaRoundNearest));
        }
        auto* output_ptr =
            col_mxfp4 + static_cast<int64_t>(col_base + local_col) * (rows / 2) +
            (row_base + local_row) / 2;
        *reinterpret_cast<uint4*>(output_ptr) = packed_output;

        const int col_tile_id = tile_x * blocks_y + tile_y;
        const int scale_offset =
            col_tile_id * 512 +
            (local_col % 32) * 16 +
            (local_col / 32) * 4 + block_k;
        col_mxsc[scale_offset] = scale;
    }
}

template <bool WITH_RMSNORM = false>
static void launch_scan_single_sg_mxfp8_direct_final_impl(
    torch::Tensor input,
    torch::Tensor row_mxfp8,
    torch::Tensor row_mxsc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    const __nv_bfloat16* rms_gamma_ptr,
    const float* rms_inv_rms_ptr,
    __nv_bfloat16* normed_output_ptr
) {
    using namespace tk_localcta;
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const int blocks_y = M / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x = K / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * blocks_x;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(row_sg.data_ptr<float>(), 0,
                    row_sg.numel() * sizeof(float), stream);
    cudaMemsetAsync(col_sg.data_ptr<float>(), 0,
                    col_sg.numel() * sizeof(float), stream);
    const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(
        input.data_ptr<at::BFloat16>());
    auto* output_ptr = reinterpret_cast<uint8_t*>(row_mxfp8.data_ptr());
    auto* scale_ptr = row_mxsc.data_ptr<uint8_t>();
    const int scan_threads = get_v4_final_sg_opt_scan_threads(false);
#define LAUNCH_MX_SCAN(THREADS_) \
    scan_single_tile_sg_mxfp8_direct_final_kernel<THREADS_, WITH_RMSNORM> \
        <<<total_tiles, THREADS_, 0, stream>>>( \
            input_ptr, output_ptr, scale_ptr, \
            row_sg.data_ptr<float>(), col_sg.data_ptr<float>(), \
            rms_gamma_ptr, rms_inv_rms_ptr, normed_output_ptr, \
            M, K, blocks_x, total_tiles)
    if (scan_threads == 96) {
        LAUNCH_MX_SCAN(96);
    } else if (scan_threads == 160) {
        LAUNCH_MX_SCAN(160);
    } else if (scan_threads == 192) {
        LAUNCH_MX_SCAN(192);
    } else if (scan_threads == 256) {
        LAUNCH_MX_SCAN(256);
    } else {
        LAUNCH_MX_SCAN(128);
    }
#undef LAUNCH_MX_SCAN
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "scan_single_tile_sg_mxfp8_direct_final_kernel failed: ",
                cudaGetErrorString(err));
}

static void launch_scan_single_sg_mxfp8_direct_final(
    torch::Tensor input,
    torch::Tensor row_mxfp8,
    torch::Tensor row_mxsc,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    launch_scan_single_sg_mxfp8_direct_final_impl<false>(
        input, row_mxfp8, row_mxsc, row_sg, col_sg,
        nullptr, nullptr, nullptr);
}

static void launch_scan_single_sg_mxfp8_rmsnorm_direct_final(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    torch::Tensor normed_output,
    torch::Tensor row_mxfp8,
    torch::Tensor row_mxsc,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    launch_scan_single_sg_mxfp8_direct_final_impl<true>(
        input, row_mxfp8, row_mxsc, row_sg, col_sg,
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()),
        inv_rms.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(normed_output.data_ptr<at::BFloat16>()));
}

template <bool DIRECT_FINAL_SG = false, bool WITH_RMSNORM = false>
static void launch_scan_single_sg_opt_impl(
    torch::Tensor input,
    bool return_transpose,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk,
    const __nv_bfloat16* rms_gamma_ptr,
    const float* rms_inv_rms_ptr,
    bool row_rht,
    bool col_rht,
    bool random_sign
) {
    TORCH_CHECK(!random_sign, "final-SG opt scan does not support random RHT signs yet");
    using namespace tk_localcta;
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const int blocks_y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * blocks_x;
    if (total_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
    float* row_sg_ptr = row_sg_chunk.data_ptr<float>();
    float* col_sg_ptr = return_transpose ? col_sg_chunk.data_ptr<float>() : row_sg_ptr;
    const int scan_threads = get_v4_final_sg_opt_scan_threads(col_rht);
#define LAUNCH_SCAN_OPT(THREADS_) \
    do { \
        if (row_rht && col_rht) { \
            scan_single_tile_sg_row_opt_kernel<THREADS_, true, true, DIRECT_FINAL_SG, WITH_RMSNORM><<<total_tiles, THREADS_, 0, stream>>>( \
                input_ptr, rms_gamma_ptr, rms_inv_rms_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose); \
        } else if (row_rht) { \
            scan_single_tile_sg_row_opt_kernel<THREADS_, true, false, DIRECT_FINAL_SG, WITH_RMSNORM><<<total_tiles, THREADS_, 0, stream>>>( \
                input_ptr, rms_gamma_ptr, rms_inv_rms_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose); \
        } else if (col_rht) { \
            scan_single_tile_sg_row_opt_kernel<THREADS_, false, true, DIRECT_FINAL_SG, WITH_RMSNORM><<<total_tiles, THREADS_, 0, stream>>>( \
                input_ptr, rms_gamma_ptr, rms_inv_rms_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose); \
        } else { \
            scan_single_tile_sg_row_opt_kernel<THREADS_, false, false, DIRECT_FINAL_SG, WITH_RMSNORM><<<total_tiles, THREADS_, 0, stream>>>( \
                input_ptr, rms_gamma_ptr, rms_inv_rms_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose); \
        } \
    } while (0)
    if (scan_threads == 96) {
        LAUNCH_SCAN_OPT(96);
    } else if (scan_threads == 160) {
        LAUNCH_SCAN_OPT(160);
    } else if (scan_threads == 192) {
        LAUNCH_SCAN_OPT(192);
    } else if (scan_threads == 256) {
        LAUNCH_SCAN_OPT(256);
    } else {
        LAUNCH_SCAN_OPT(128);
    }
#undef LAUNCH_SCAN_OPT
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scan_single_tile_sg_row_opt_kernel failed: ", cudaGetErrorString(err));
}

static void launch_scan_single_sg_opt(
    torch::Tensor input,
    bool return_transpose,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk,
    bool row_rht,
    bool col_rht,
    bool random_sign
) {
    launch_scan_single_sg_opt_impl<false, false>(
        input, return_transpose, row_sg_chunk, col_sg_chunk, nullptr, nullptr,
        row_rht, col_rht, random_sign);
}

static void launch_scan_single_sg_opt_rmsnorm(
    torch::Tensor input,
    bool return_transpose,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk,
    const __nv_bfloat16* rms_gamma_ptr,
    const float* rms_inv_rms_ptr,
    bool row_rht,
    bool col_rht,
    bool random_sign
) {
    launch_scan_single_sg_opt_impl<false, true>(
        input, return_transpose, row_sg_chunk, col_sg_chunk,
        rms_gamma_ptr, rms_inv_rms_ptr, row_rht, col_rht, random_sign);
}

static void launch_scan_single_sg_opt_direct_final(
    torch::Tensor input,
    bool return_transpose,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool row_rht,
    bool col_rht,
    bool random_sign
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(row_sg.data_ptr<float>(), 0, row_sg.numel() * sizeof(float), stream);
    if (return_transpose) {
        cudaMemsetAsync(col_sg.data_ptr<float>(), 0, col_sg.numel() * sizeof(float), stream);
    }
    launch_scan_single_sg_opt_impl<true, false>(
        input, return_transpose, row_sg, col_sg, nullptr, nullptr,
        row_rht, col_rht, random_sign);
}

static void launch_scan_single_sg_opt_direct_final_rmsnorm(
    torch::Tensor input,
    bool return_transpose,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    const __nv_bfloat16* rms_gamma_ptr,
    const float* rms_inv_rms_ptr,
    bool row_rht,
    bool col_rht,
    bool random_sign
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(row_sg.data_ptr<float>(), 0, row_sg.numel() * sizeof(float), stream);
    if (return_transpose) {
        cudaMemsetAsync(col_sg.data_ptr<float>(), 0, col_sg.numel() * sizeof(float), stream);
    }
    launch_scan_single_sg_opt_impl<true, true>(
        input, return_transpose, row_sg, col_sg, rms_gamma_ptr, rms_inv_rms_ptr,
        row_rht, col_rht, random_sign);
}

template <int BLOCK_SIZE = 256>
__global__ void scan_silu_tile_sg_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __nv_bfloat16* __restrict__ h3,
    float* __restrict__ row_sg_chunk,
    float* __restrict__ col_sg_chunk,
    int rows,
    int cols,
    int blocks_x,
    int total_tiles,
    bool return_transpose
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= total_tiles) return;

    const int ctaid_x = chunk_id % blocks_x;
    const int ctaid_y = chunk_id / blocks_x;
    const int row_base = ctaid_y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
    const int col_base = ctaid_x * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;

    float thread_max = 0.0f;
    constexpr int total = tk_localcta::LocalCTAConfig::CHUNK_DIM_Y *
                          tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int vec = 4;

    for (int idx = threadIdx.x * vec; idx < total; idx += BLOCK_SIZE * vec) {
        const int r = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int c = idx - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int gr = row_base + r;
        const int gc = col_base + c;
        if (gr >= rows) {
            continue;
        }
        if (gc + (vec - 1) < cols) {
            const int64_t base = static_cast<int64_t>(gr) * cols + gc;
            const int2 h1_pack = *reinterpret_cast<const int2*>(h1_raw + base);
            const int2 h3_pack = *reinterpret_cast<const int2*>(h3 + base);

            const __nv_bfloat162 h1_0 = *reinterpret_cast<const __nv_bfloat162*>(&h1_pack.x);
            const __nv_bfloat162 h1_1 = *reinterpret_cast<const __nv_bfloat162*>(&h1_pack.y);
            const __nv_bfloat162 h3_0 = *reinterpret_cast<const __nv_bfloat162*>(&h3_pack.x);
            const __nv_bfloat162 h3_1 = *reinterpret_cast<const __nv_bfloat162*>(&h3_pack.y);
            const float2 h1_0f = __bfloat1622float2(h1_0);
            const float2 h1_1f = __bfloat1622float2(h1_1);
            const float2 h3_0f = __bfloat1622float2(h3_0);
            const float2 h3_1f = __bfloat1622float2(h3_1);

            const float o0x = __bfloat162float(__float2bfloat16_rn(
                (h1_0f.x / (1.0f + __expf(-h1_0f.x))) * h3_0f.x));
            const float o0y = __bfloat162float(__float2bfloat16_rn(
                (h1_0f.y / (1.0f + __expf(-h1_0f.y))) * h3_0f.y));
            const float o1x = __bfloat162float(__float2bfloat16_rn(
                (h1_1f.x / (1.0f + __expf(-h1_1f.x))) * h3_1f.x));
            const float o1y = __bfloat162float(__float2bfloat16_rn(
                (h1_1f.y / (1.0f + __expf(-h1_1f.y))) * h3_1f.y));
            thread_max = fmaxf(thread_max, fabsf(o0x));
            thread_max = fmaxf(thread_max, fabsf(o0y));
            thread_max = fmaxf(thread_max, fabsf(o1x));
            thread_max = fmaxf(thread_max, fabsf(o1y));
        } else {
            #pragma unroll
            for (int j = 0; j < vec; ++j) {
                const int cc = gc + j;
                if (cc < cols) {
                    const int64_t base = static_cast<int64_t>(gr) * cols + cc;
                    const float h1v = __bfloat162float(h1_raw[base]);
                    const float h3v = __bfloat162float(h3[base]);
                    const float out = __bfloat162float(__float2bfloat16_rn(
                        (h1v / (1.0f + __expf(-h1v))) * h3v));
                    thread_max = fmaxf(thread_max, fabsf(out));
                }
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    __shared__ float warp_max[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[wid] = thread_max;
    }
    __syncthreads();

    if (wid == 0) {
        float block_max = (lane < (BLOCK_SIZE / 32)) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, mask));
        }
        if (lane == 0) {
            const float sg_val = block_max / tk_localcta::localcta_global_scale_num();
            row_sg_chunk[ctaid_y * blocks_x + ctaid_x] = sg_val;
            if (return_transpose) {
                const int blocks_y = (rows + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y - 1) /
                                     tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
                col_sg_chunk[ctaid_x * blocks_y + ctaid_y] = sg_val;
            }
        }
    }
}

static void launch_scan_silu_sg(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    bool return_transpose,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk
) {
    using namespace tk_localcta;
    const int M = static_cast<int>(h1_raw.size(0));
    const int K = static_cast<int>(h1_raw.size(1));
    const int blocks_y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * blocks_x;
    if (total_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* h1_ptr = reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr<at::BFloat16>());
    const auto* h3_ptr = reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr<at::BFloat16>());
    float* row_sg_ptr = row_sg_chunk.data_ptr<float>();
    float* col_sg_ptr = return_transpose ? col_sg_chunk.data_ptr<float>() : row_sg_ptr;
    const int scan_threads = get_v4_split3_scan_threads();
    if (scan_threads == 96) {
        scan_silu_tile_sg_kernel<96><<<total_tiles, 96, 0, stream>>>(
            h1_ptr, h3_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else if (scan_threads == 160) {
        scan_silu_tile_sg_kernel<160><<<total_tiles, 160, 0, stream>>>(
            h1_ptr, h3_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else if (scan_threads == 192) {
        scan_silu_tile_sg_kernel<192><<<total_tiles, 192, 0, stream>>>(
            h1_ptr, h3_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else if (scan_threads == 256) {
        scan_silu_tile_sg_kernel<256><<<total_tiles, 256, 0, stream>>>(
            h1_ptr, h3_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    } else {
        scan_silu_tile_sg_kernel<128><<<total_tiles, 128, 0, stream>>>(
            h1_ptr, h3_ptr, row_sg_ptr, col_sg_ptr, M, K, blocks_x, total_tiles, return_transpose);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scan_silu_tile_sg_kernel failed: ", cudaGetErrorString(err));
}

template <int BLOCK_SIZE = 256, bool DIRECT_FINAL_SG = false>
__global__ void scan_split3_tile_sg_kernel(
    const __nv_bfloat16* __restrict__ input0,
    const __nv_bfloat16* __restrict__ input1,
    const __nv_bfloat16* __restrict__ input2,
    float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_chunk_2,
    float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_chunk_2,
    float* __restrict__ row_sg_final_0,
    float* __restrict__ row_sg_final_1,
    float* __restrict__ row_sg_final_2,
    float* __restrict__ col_sg_final_0,
    float* __restrict__ col_sg_final_1,
    float* __restrict__ col_sg_final_2,
    int rows,
    int n0,
    int n1,
    int n2,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int blocks_x0,
    int blocks_x1,
    int blocks_x2,
    int total_tiles
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= total_tiles) return;

    const int blocks_x = blocks_x0 + blocks_x1 + blocks_x2;
    const int ctaid_x = chunk_id % blocks_x;
    const int ctaid_y = chunk_id / blocks_x;
    const int split01 = blocks_x0 + blocks_x1;

    const __nv_bfloat16* input;
    float* row_sg_chunk;
    float* col_sg_chunk;
    float* row_sg_final;
    float* col_sg_final;
    int cols;
    int64_t row_stride;
    int local_ctaid_x = ctaid_x;
    int split_tiles = blocks_x0;
    if (ctaid_x >= split01) {
        input = input2;
        row_sg_chunk = row_sg_chunk_2;
        col_sg_chunk = col_sg_chunk_2;
        row_sg_final = row_sg_final_2;
        col_sg_final = col_sg_final_2;
        cols = n2;
        row_stride = stride2;
        local_ctaid_x -= split01;
        split_tiles = blocks_x2;
    } else if (ctaid_x >= blocks_x0) {
        input = input1;
        row_sg_chunk = row_sg_chunk_1;
        col_sg_chunk = col_sg_chunk_1;
        row_sg_final = row_sg_final_1;
        col_sg_final = col_sg_final_1;
        cols = n1;
        row_stride = stride1;
        local_ctaid_x -= blocks_x0;
        split_tiles = blocks_x1;
    } else {
        input = input0;
        row_sg_chunk = row_sg_chunk_0;
        col_sg_chunk = col_sg_chunk_0;
        row_sg_final = row_sg_final_0;
        col_sg_final = col_sg_final_0;
        cols = n0;
        row_stride = stride0;
    }

    unsigned int thread_max_bits = 0;
    const int row_base = ctaid_y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
    const int col_base = local_ctaid_x * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int total = tk_localcta::LocalCTAConfig::CHUNK_DIM_Y *
                          tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int vec = 8;
    const bool full_tile =
        (row_base + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y <= rows) &&
        (col_base + tk_localcta::LocalCTAConfig::CHUNK_DIM_X <= cols);
    if (full_tile) {
        for (int idx = threadIdx.x * vec; idx < total; idx += BLOCK_SIZE * vec) {
            const int r = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int c = idx - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int gr = row_base + r;
            const int gc = col_base + c;
            const int4 packed = *reinterpret_cast<const int4*>(input + (int64_t)gr * row_stride + gc);
            const unsigned int packed_max = split3_bf16x8_abs_max_bits(packed);
            thread_max_bits = packed_max > thread_max_bits ? packed_max : thread_max_bits;
        }
    } else {
        for (int idx = threadIdx.x * vec; idx < total; idx += BLOCK_SIZE * vec) {
            const int r = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int c = idx - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int gr = row_base + r;
            const int gc = col_base + c;
            if (gr < rows && gc + (vec - 1) < cols) {
                const int4 packed = *reinterpret_cast<const int4*>(input + (int64_t)gr * row_stride + gc);
                const unsigned int packed_max = split3_bf16x8_abs_max_bits(packed);
                thread_max_bits = packed_max > thread_max_bits ? packed_max : thread_max_bits;
            } else {
                #pragma unroll
                for (int j = 0; j < vec; ++j) {
                    const int cc = gc + j;
                    if (gr < rows && cc < cols) {
                        const unsigned int bits =
                            *reinterpret_cast<const unsigned short*>(input + (int64_t)gr * row_stride + cc) & 0x7fffu;
                        thread_max_bits = bits > thread_max_bits ? bits : thread_max_bits;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const unsigned int other = __shfl_xor_sync(0xffffffff, thread_max_bits, mask);
        thread_max_bits = other > thread_max_bits ? other : thread_max_bits;
    }

    __shared__ unsigned int warp_max[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[wid] = thread_max_bits;
    }
    __syncthreads();

    if (wid == 0) {
        unsigned int block_max_bits = (lane < (BLOCK_SIZE / 32)) ? warp_max[lane] : 0;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const unsigned int other = __shfl_xor_sync(0xffffffff, block_max_bits, mask);
            block_max_bits = other > block_max_bits ? other : block_max_bits;
        }
        if (lane == 0) {
            const float sg_val =
                split3_bf16_abs_bits_to_float(block_max_bits) / tk_localcta::localcta_global_scale_num();
            if constexpr (DIRECT_FINAL_SG) {
                transformer_engine::atomicMaxFloat(row_sg_final + (ctaid_y >> 1), sg_val);
                transformer_engine::atomicMaxFloat(col_sg_final + (local_ctaid_x >> 1), sg_val);
            } else {
                row_sg_chunk[ctaid_y * split_tiles + local_ctaid_x] = sg_val;
                const int tiles_y = (rows + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y - 1) /
                                    tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
                col_sg_chunk[local_ctaid_x * tiles_y + ctaid_y] = sg_val;
            }
        }
    }
}

template <int BLOCK_SIZE = 256, bool DIRECT_FINAL_SG = false>
__global__ void scan_split3_tile_sg_rope_kernel(
    const __nv_bfloat16* __restrict__ input0,
    const __nv_bfloat16* __restrict__ input1,
    const __nv_bfloat16* __restrict__ input2,
    const float2* __restrict__ rope_cs,
    __nv_bfloat16* __restrict__ rotated0,
    __nv_bfloat16* __restrict__ rotated1,
    float* __restrict__ row_sg_chunk_0,
    float* __restrict__ row_sg_chunk_1,
    float* __restrict__ row_sg_chunk_2,
    float* __restrict__ col_sg_chunk_0,
    float* __restrict__ col_sg_chunk_1,
    float* __restrict__ col_sg_chunk_2,
    float* __restrict__ row_sg_final_0,
    float* __restrict__ row_sg_final_1,
    float* __restrict__ row_sg_final_2,
    float* __restrict__ col_sg_final_0,
    float* __restrict__ col_sg_final_1,
    float* __restrict__ col_sg_final_2,
    int rows,
    int n0,
    int n1,
    int n2,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t rotated_stride0,
    int64_t rotated_stride1,
    int blocks_x0,
    int blocks_x1,
    int blocks_x2,
    int total_tiles,
    int rope_seq_mask
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= total_tiles) return;

    const int blocks_x = blocks_x0 + blocks_x1 + blocks_x2;
    const int ctaid_x = chunk_id % blocks_x;
    const int ctaid_y = chunk_id / blocks_x;
    const int split01 = blocks_x0 + blocks_x1;

    const __nv_bfloat16* input;
    float* row_sg_chunk;
    float* col_sg_chunk;
    float* row_sg_final;
    float* col_sg_final;
    int cols;
    int64_t row_stride;
    __nv_bfloat16* rotated = nullptr;
    int64_t rotated_stride = 0;
    int local_ctaid_x = ctaid_x;
    int split_tiles = blocks_x0;
    if (ctaid_x >= split01) {
        input = input2;
        row_sg_chunk = row_sg_chunk_2;
        col_sg_chunk = col_sg_chunk_2;
        row_sg_final = row_sg_final_2;
        col_sg_final = col_sg_final_2;
        cols = n2;
        row_stride = stride2;
        local_ctaid_x -= split01;
        split_tiles = blocks_x2;
    } else if (ctaid_x >= blocks_x0) {
        input = input1;
        row_sg_chunk = row_sg_chunk_1;
        col_sg_chunk = col_sg_chunk_1;
        row_sg_final = row_sg_final_1;
        col_sg_final = col_sg_final_1;
        cols = n1;
        row_stride = stride1;
        rotated = rotated1;
        rotated_stride = rotated_stride1;
        local_ctaid_x -= blocks_x0;
        split_tiles = blocks_x1;
    } else {
        input = input0;
        row_sg_chunk = row_sg_chunk_0;
        col_sg_chunk = col_sg_chunk_0;
        row_sg_final = row_sg_final_0;
        col_sg_final = col_sg_final_0;
        cols = n0;
        row_stride = stride0;
        rotated = rotated0;
        rotated_stride = rotated_stride0;
    }

    float thread_max = 0.0f;
    const int row_base = ctaid_y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
    const int col_base = local_ctaid_x * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int total = tk_localcta::LocalCTAConfig::CHUNK_DIM_Y *
                          tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int vec = 8;
    const bool full_tile =
        (row_base + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y <= rows) &&
        (col_base + tk_localcta::LocalCTAConfig::CHUNK_DIM_X <= cols);

    if (ctaid_x < split01) {
        constexpr int pair_total = tk_localcta::LocalCTAConfig::CHUNK_DIM_Y *
                                   (tk_localcta::LocalCTAConfig::CHUNK_DIM_X / 2);
        if (full_tile) {
            for (int pair_idx = threadIdx.x; pair_idx < pair_total; pair_idx += BLOCK_SIZE) {
                const int r = pair_idx / (tk_localcta::LocalCTAConfig::CHUNK_DIM_X / 2);
                const int pair_col = pair_idx - r * (tk_localcta::LocalCTAConfig::CHUNK_DIM_X / 2);
                const int gr = row_base + r;
                const int gc = col_base + pair_col * 2;
                const auto packed =
                    *reinterpret_cast<const __nv_bfloat162*>(input + (int64_t)gr * row_stride + gc);
                const float2 v = __bfloat1622float2(packed);
                const float2 cs = rope_cs[((gr & rope_seq_mask) * 32) + (((gc >> 1) & 31))];
                const float x = v.x * cs.x + v.y * cs.y;
                const float y = v.y * cs.x - v.x * cs.y;
                const __nv_bfloat162 out = __float22bfloat162_rn(make_float2(x, y));
                if (rotated != nullptr) {
                    *reinterpret_cast<__nv_bfloat162*>(rotated + (int64_t)gr * rotated_stride + gc) = out;
                }
                const float2 out_f = __bfloat1622float2(out);
                thread_max = fmaxf(thread_max, fmaxf(fabsf(out_f.x), fabsf(out_f.y)));
            }
        } else {
            for (int pair_idx = threadIdx.x; pair_idx < pair_total; pair_idx += BLOCK_SIZE) {
                const int r = pair_idx / (tk_localcta::LocalCTAConfig::CHUNK_DIM_X / 2);
                const int pair_col = pair_idx - r * (tk_localcta::LocalCTAConfig::CHUNK_DIM_X / 2);
                const int gr = row_base + r;
                const int gc = col_base + pair_col * 2;
                if (gr < rows && gc + 1 < cols) {
                    const auto packed =
                        *reinterpret_cast<const __nv_bfloat162*>(input + (int64_t)gr * row_stride + gc);
                    const float2 v = __bfloat1622float2(packed);
                    const float2 cs = rope_cs[((gr & rope_seq_mask) * 32) + (((gc >> 1) & 31))];
                    const float x = v.x * cs.x + v.y * cs.y;
                    const float y = v.y * cs.x - v.x * cs.y;
                    const __nv_bfloat162 out = __float22bfloat162_rn(make_float2(x, y));
                    if (rotated != nullptr) {
                        *reinterpret_cast<__nv_bfloat162*>(rotated + (int64_t)gr * rotated_stride + gc) = out;
                    }
                    const float2 out_f = __bfloat1622float2(out);
                    thread_max = fmaxf(thread_max, fmaxf(fabsf(out_f.x), fabsf(out_f.y)));
                }
            }
        }
    } else if (full_tile) {
        unsigned int thread_max_bits = 0;
        for (int idx = threadIdx.x * vec; idx < total; idx += BLOCK_SIZE * vec) {
            const int r = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int c = idx - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int gr = row_base + r;
            const int gc = col_base + c;
            const int4 packed = *reinterpret_cast<const int4*>(input + (int64_t)gr * row_stride + gc);
            const unsigned int packed_max = split3_bf16x8_abs_max_bits(packed);
            thread_max_bits = packed_max > thread_max_bits ? packed_max : thread_max_bits;
        }
        thread_max = split3_bf16_abs_bits_to_float(thread_max_bits);
    } else {
        unsigned int thread_max_bits = 0;
        for (int idx = threadIdx.x * vec; idx < total; idx += BLOCK_SIZE * vec) {
            const int r = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int c = idx - r * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
            const int gr = row_base + r;
            const int gc = col_base + c;
            if (gr < rows && gc + (vec - 1) < cols) {
                const int4 packed = *reinterpret_cast<const int4*>(input + (int64_t)gr * row_stride + gc);
                const unsigned int packed_max = split3_bf16x8_abs_max_bits(packed);
                thread_max_bits = packed_max > thread_max_bits ? packed_max : thread_max_bits;
            } else {
                #pragma unroll
                for (int j = 0; j < vec; ++j) {
                    const int cc = gc + j;
                    if (gr < rows && cc < cols) {
                        const unsigned int bits =
                            *reinterpret_cast<const unsigned short*>(input + (int64_t)gr * row_stride + cc) & 0x7fffu;
                        thread_max_bits = bits > thread_max_bits ? bits : thread_max_bits;
                    }
                }
            }
        }
        thread_max = split3_bf16_abs_bits_to_float(thread_max_bits);
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    __shared__ float warp_max[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[wid] = thread_max;
    }
    __syncthreads();

    if (wid == 0) {
        float block_max = (lane < (BLOCK_SIZE / 32)) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, mask));
        }
        if (lane == 0) {
            const float sg_val = block_max / tk_localcta::localcta_global_scale_num();
            if constexpr (DIRECT_FINAL_SG) {
                transformer_engine::atomicMaxFloat(row_sg_final + (ctaid_y >> 1), sg_val);
                transformer_engine::atomicMaxFloat(col_sg_final + (local_ctaid_x >> 1), sg_val);
            } else {
                row_sg_chunk[ctaid_y * split_tiles + local_ctaid_x] = sg_val;
                const int tiles_y = (rows + tk_localcta::LocalCTAConfig::CHUNK_DIM_Y - 1) /
                                    tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
                col_sg_chunk[local_ctaid_x * tiles_y + ctaid_y] = sg_val;
            }
        }
    }
}

static void launch_scan_split3_sg(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor row_sg_chunk_2,
    torch::Tensor col_sg_chunk_2
) {
    using namespace tk_localcta;
    const int M = static_cast<int>(input0.size(0));
    const int n0 = static_cast<int>(input0.size(1));
    const int n1 = static_cast<int>(input1.size(1));
    const int n2 = static_cast<int>(input2.size(1));
    TORCH_CHECK(input0.stride(1) == 1 && input1.stride(1) == 1 && input2.stride(1) == 1,
                "split inputs must have contiguous last dimension [M, N_i]");
    const int64_t stride0 = input0.stride(0);
    const int64_t stride1 = input1.stride(0);
    const int64_t stride2 = input2.stride(0);
    const int blocks_y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x0 = (n0 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x1 = (n1 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x2 = (n2 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * (blocks_x0 + blocks_x1 + blocks_x2);
    if (total_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* input0_ptr = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr<at::BFloat16>());
    const auto* input1_ptr = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr<at::BFloat16>());
    const auto* input2_ptr = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr<at::BFloat16>());
    float* row_sg_0_ptr = row_sg_chunk_0.data_ptr<float>();
    float* row_sg_1_ptr = row_sg_chunk_1.data_ptr<float>();
    float* row_sg_2_ptr = row_sg_chunk_2.data_ptr<float>();
    float* col_sg_0_ptr = col_sg_chunk_0.data_ptr<float>();
    float* col_sg_1_ptr = col_sg_chunk_1.data_ptr<float>();
    float* col_sg_2_ptr = col_sg_chunk_2.data_ptr<float>();
    const int scan_threads = get_v4_split3_scan_threads();
    if (scan_threads == 96) {
        scan_split3_tile_sg_kernel<96><<<total_tiles, 96, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else if (scan_threads == 160) {
        scan_split3_tile_sg_kernel<160><<<total_tiles, 160, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else if (scan_threads == 192) {
        scan_split3_tile_sg_kernel<192><<<total_tiles, 192, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else if (scan_threads == 256) {
        scan_split3_tile_sg_kernel<256><<<total_tiles, 256, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else {
        scan_split3_tile_sg_kernel<128><<<total_tiles, 128, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scan_split3_tile_sg_kernel failed: ", cudaGetErrorString(err));
}

static void zero_split3_final_sg(
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sg_2,
    torch::Tensor col_sg_2,
    cudaStream_t stream
) {
    cudaMemsetAsync(row_sg_0.data_ptr<float>(), 0, row_sg_0.numel() * sizeof(float), stream);
    cudaMemsetAsync(row_sg_1.data_ptr<float>(), 0, row_sg_1.numel() * sizeof(float), stream);
    cudaMemsetAsync(row_sg_2.data_ptr<float>(), 0, row_sg_2.numel() * sizeof(float), stream);
    auto* col0 = col_sg_0.data_ptr<float>();
    auto* col1 = col_sg_1.data_ptr<float>();
    auto* col2 = col_sg_2.data_ptr<float>();
    const int64_t n0 = col_sg_0.numel();
    const int64_t n1 = col_sg_1.numel();
    const int64_t n2 = col_sg_2.numel();
    if (col1 == col0 + n0 && col2 == col1 + n1) {
        cudaMemsetAsync(col0, 0, (n0 + n1 + n2) * sizeof(float), stream);
    } else {
        cudaMemsetAsync(col0, 0, n0 * sizeof(float), stream);
        cudaMemsetAsync(col1, 0, n1 * sizeof(float), stream);
        cudaMemsetAsync(col2, 0, n2 * sizeof(float), stream);
    }
}

static void launch_scan_split3_sg_direct_final(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sg_2,
    torch::Tensor col_sg_2
) {
    using namespace tk_localcta;
    const int M = static_cast<int>(input0.size(0));
    const int n0 = static_cast<int>(input0.size(1));
    const int n1 = static_cast<int>(input1.size(1));
    const int n2 = static_cast<int>(input2.size(1));
    TORCH_CHECK(input0.stride(1) == 1 && input1.stride(1) == 1 && input2.stride(1) == 1,
                "split inputs must have contiguous last dimension [M, N_i]");
    const int64_t stride0 = input0.stride(0);
    const int64_t stride1 = input1.stride(0);
    const int64_t stride2 = input2.stride(0);
    const int blocks_y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x0 = (n0 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x1 = (n1 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x2 = (n2 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * (blocks_x0 + blocks_x1 + blocks_x2);
    if (total_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    zero_split3_final_sg(row_sg_0, col_sg_0, row_sg_1, col_sg_1, row_sg_2, col_sg_2, stream);
    const auto* input0_ptr = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr<at::BFloat16>());
    const auto* input1_ptr = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr<at::BFloat16>());
    const auto* input2_ptr = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr<at::BFloat16>());
    float* row_sg_0_ptr = row_sg_0.data_ptr<float>();
    float* row_sg_1_ptr = row_sg_1.data_ptr<float>();
    float* row_sg_2_ptr = row_sg_2.data_ptr<float>();
    float* col_sg_0_ptr = col_sg_0.data_ptr<float>();
    float* col_sg_1_ptr = col_sg_1.data_ptr<float>();
    float* col_sg_2_ptr = col_sg_2.data_ptr<float>();
    const int scan_threads = get_v4_split3_scan_threads();
    if (scan_threads == 96) {
        scan_split3_tile_sg_kernel<96, true><<<total_tiles, 96, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else if (scan_threads == 160) {
        scan_split3_tile_sg_kernel<160, true><<<total_tiles, 160, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else if (scan_threads == 192) {
        scan_split3_tile_sg_kernel<192, true><<<total_tiles, 192, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else if (scan_threads == 256) {
        scan_split3_tile_sg_kernel<256, true><<<total_tiles, 256, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    } else {
        scan_split3_tile_sg_kernel<128, true><<<total_tiles, 128, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            blocks_x0, blocks_x1, blocks_x2, total_tiles);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scan_split3_tile_sg_kernel direct-final failed: ", cudaGetErrorString(err));
}

static void launch_scan_split3_sg_rope(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor rope_cs,
    int64_t rope_seq_len,
    torch::Tensor rotated0,
    torch::Tensor rotated1,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor row_sg_chunk_2,
    torch::Tensor col_sg_chunk_2
) {
    using namespace tk_localcta;
    check_rope_live64_tensor(rope_cs, rope_seq_len);
    const int M = static_cast<int>(input0.size(0));
    const int n0 = static_cast<int>(input0.size(1));
    const int n1 = static_cast<int>(input1.size(1));
    const int n2 = static_cast<int>(input2.size(1));
    TORCH_CHECK(input0.stride(1) == 1 && input1.stride(1) == 1 && input2.stride(1) == 1,
                "split inputs must have contiguous last dimension [M, N_i]");
    const int64_t stride0 = input0.stride(0);
    const int64_t stride1 = input1.stride(0);
    const int64_t stride2 = input2.stride(0);
    const bool materialize_rotated = rotated0.defined() && rotated0.numel() > 0 &&
                                     rotated1.defined() && rotated1.numel() > 0;
    if (materialize_rotated) {
        TORCH_CHECK(rotated0.dim() == 2 && rotated1.dim() == 2 &&
                    rotated0.is_cuda() && rotated1.is_cuda() &&
                    rotated0.scalar_type() == torch::kBFloat16 &&
                    rotated1.scalar_type() == torch::kBFloat16 &&
                    rotated0.size(0) == M && rotated0.size(1) == n0 &&
                    rotated1.size(0) == M && rotated1.size(1) == n1 &&
                    rotated0.stride(1) == 1 && rotated1.stride(1) == 1,
                    "rotated split outputs must be bf16 CUDA tensors shaped [M, N_i] with contiguous last dim");
    }
    const int64_t rotated_stride0 = materialize_rotated ? rotated0.stride(0) : 0;
    const int64_t rotated_stride1 = materialize_rotated ? rotated1.stride(0) : 0;
    const int blocks_y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x0 = (n0 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x1 = (n1 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x2 = (n2 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * (blocks_x0 + blocks_x1 + blocks_x2);
    if (total_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* input0_ptr = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr<at::BFloat16>());
    const auto* input1_ptr = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr<at::BFloat16>());
    const auto* input2_ptr = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr<at::BFloat16>());
    const auto* rope_ptr = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>());
    auto* rotated0_ptr = materialize_rotated ? reinterpret_cast<__nv_bfloat16*>(rotated0.data_ptr<at::BFloat16>()) : nullptr;
    auto* rotated1_ptr = materialize_rotated ? reinterpret_cast<__nv_bfloat16*>(rotated1.data_ptr<at::BFloat16>()) : nullptr;
    float* row_sg_0_ptr = row_sg_chunk_0.data_ptr<float>();
    float* row_sg_1_ptr = row_sg_chunk_1.data_ptr<float>();
    float* row_sg_2_ptr = row_sg_chunk_2.data_ptr<float>();
    float* col_sg_0_ptr = col_sg_chunk_0.data_ptr<float>();
    float* col_sg_1_ptr = col_sg_chunk_1.data_ptr<float>();
    float* col_sg_2_ptr = col_sg_chunk_2.data_ptr<float>();
    const int scan_threads = get_v4_split3_rope_scan_threads();
    const int rope_seq_mask = static_cast<int>(rope_seq_len - 1);
    if (scan_threads == 96) {
        scan_split3_tile_sg_rope_kernel<96><<<total_tiles, 96, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else if (scan_threads == 160) {
        scan_split3_tile_sg_rope_kernel<160><<<total_tiles, 160, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else if (scan_threads == 192) {
        scan_split3_tile_sg_rope_kernel<192><<<total_tiles, 192, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else if (scan_threads == 256) {
        scan_split3_tile_sg_rope_kernel<256><<<total_tiles, 256, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else {
        scan_split3_tile_sg_rope_kernel<128><<<total_tiles, 128, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scan_split3_tile_sg_rope_kernel failed: ", cudaGetErrorString(err));
}

static void launch_scan_split3_sg_rope_direct_final(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor rope_cs,
    int64_t rope_seq_len,
    torch::Tensor rotated0,
    torch::Tensor rotated1,
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sg_2,
    torch::Tensor col_sg_2
) {
    using namespace tk_localcta;
    check_rope_live64_tensor(rope_cs, rope_seq_len);
    const int M = static_cast<int>(input0.size(0));
    const int n0 = static_cast<int>(input0.size(1));
    const int n1 = static_cast<int>(input1.size(1));
    const int n2 = static_cast<int>(input2.size(1));
    TORCH_CHECK(input0.stride(1) == 1 && input1.stride(1) == 1 && input2.stride(1) == 1,
                "split inputs must have contiguous last dimension [M, N_i]");
    const int64_t stride0 = input0.stride(0);
    const int64_t stride1 = input1.stride(0);
    const int64_t stride2 = input2.stride(0);
    const bool materialize_rotated = rotated0.defined() && rotated0.numel() > 0 &&
                                     rotated1.defined() && rotated1.numel() > 0;
    if (materialize_rotated) {
        TORCH_CHECK(rotated0.dim() == 2 && rotated1.dim() == 2 &&
                    rotated0.is_cuda() && rotated1.is_cuda() &&
                    rotated0.scalar_type() == torch::kBFloat16 &&
                    rotated1.scalar_type() == torch::kBFloat16 &&
                    rotated0.size(0) == M && rotated0.size(1) == n0 &&
                    rotated1.size(0) == M && rotated1.size(1) == n1 &&
                    rotated0.stride(1) == 1 && rotated1.stride(1) == 1,
                    "rotated split outputs must be bf16 CUDA tensors shaped [M, N_i] with contiguous last dim");
    }
    const int64_t rotated_stride0 = materialize_rotated ? rotated0.stride(0) : 0;
    const int64_t rotated_stride1 = materialize_rotated ? rotated1.stride(0) : 0;
    const int blocks_y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_x0 = (n0 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x1 = (n1 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_x2 = (n2 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_y * (blocks_x0 + blocks_x1 + blocks_x2);
    if (total_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    zero_split3_final_sg(row_sg_0, col_sg_0, row_sg_1, col_sg_1, row_sg_2, col_sg_2, stream);
    const auto* input0_ptr = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr<at::BFloat16>());
    const auto* input1_ptr = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr<at::BFloat16>());
    const auto* input2_ptr = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr<at::BFloat16>());
    const auto* rope_ptr = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>());
    auto* rotated0_ptr = materialize_rotated ? reinterpret_cast<__nv_bfloat16*>(rotated0.data_ptr<at::BFloat16>()) : nullptr;
    auto* rotated1_ptr = materialize_rotated ? reinterpret_cast<__nv_bfloat16*>(rotated1.data_ptr<at::BFloat16>()) : nullptr;
    float* row_sg_0_ptr = row_sg_0.data_ptr<float>();
    float* row_sg_1_ptr = row_sg_1.data_ptr<float>();
    float* row_sg_2_ptr = row_sg_2.data_ptr<float>();
    float* col_sg_0_ptr = col_sg_0.data_ptr<float>();
    float* col_sg_1_ptr = col_sg_1.data_ptr<float>();
    float* col_sg_2_ptr = col_sg_2.data_ptr<float>();
    const int scan_threads = get_v4_split3_rope_scan_threads();
    const int rope_seq_mask = static_cast<int>(rope_seq_len - 1);
    if (scan_threads == 96) {
        scan_split3_tile_sg_rope_kernel<96, true><<<total_tiles, 96, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else if (scan_threads == 160) {
        scan_split3_tile_sg_rope_kernel<160, true><<<total_tiles, 160, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else if (scan_threads == 192) {
        scan_split3_tile_sg_rope_kernel<192, true><<<total_tiles, 192, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else if (scan_threads == 256) {
        scan_split3_tile_sg_rope_kernel<256, true><<<total_tiles, 256, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    } else {
        scan_split3_tile_sg_rope_kernel<128, true><<<total_tiles, 128, 0, stream>>>(
            input0_ptr, input1_ptr, input2_ptr, rope_ptr,
            rotated0_ptr, rotated1_ptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            row_sg_0_ptr, row_sg_1_ptr, row_sg_2_ptr,
            col_sg_0_ptr, col_sg_1_ptr, col_sg_2_ptr,
            M, n0, n1, n2, stride0, stride1, stride2,
            rotated_stride0, rotated_stride1,
            blocks_x0, blocks_x1, blocks_x2, total_tiles, rope_seq_mask);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scan_split3_tile_sg_rope_kernel direct-final failed: ", cudaGetErrorString(err));
}

struct LocalCTADataSRAxes {
    bool row;
    bool col;
};

static LocalCTADataSRAxes resolve_localcta_data_sr_axes(
    bool enabled,
    std::string axes,
    const char* context);

static void launch_localcta_split3_quant_raw(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_2,
    torch::Tensor col_fp4_2,
    torch::Tensor col_sc_2,
    torch::Tensor row_sg_chunk_2,
    torch::Tensor col_sg_chunk_2,
    bool data_stochastic_rounding = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    std::string data_sr_axes = "both",
    torch::Tensor persistent_rng_state = torch::Tensor(),
    bool col_rht = false,
    bool with_random_sign_mask = false,
    bool encode_centric = true
) {
    using namespace tk_localcta;
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);
    const auto sr_axes = resolve_localcta_data_sr_axes(
        data_stochastic_rounding, data_sr_axes, "localCTA v4 split3 producer");
    TORCH_CHECK(!with_random_sign_mask || col_rht,
                "localCTA v4 split3 fixed-sign mask requires column RHT");
    if (col_rht) {
        TORCH_CHECK(sr_axes.row && !sr_axes.col,
                    "localCTA v4 split3 column RHT requires row-only data SR");
        TORCH_CHECK(M % 256 == 0 && n0 % 256 == 0 && n1 % 256 == 0 && n2 % 256 == 0,
                    "localCTA v4 split3 column RHT requires M and all split widths "
                    "to be multiples of 256");
        TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                    "localCTA v4 split3 column RHT requires the v4 outer-SG contract");
        TORCH_CHECK(persistent_rng_state.defined(),
                    "localCTA v4 split3 column RHT requires explicit persistent RNG state");
    }
    TORCH_CHECK(!sr_axes.col,
                "localCTA v4 split3 raw producer currently supports row-only data SR");

    alignas(64) CUtensorMap tmap_in0{}, tmap_in1{}, tmap_in2{};
    alignas(64) CUtensorMap tmap_out0{}, tmap_out1{}, tmap_out2{};
    alignas(64) CUtensorMap tmap_out_t0{}, tmap_out_t1{}, tmap_out_t2{};
    alignas(64) CUtensorMap tmap_sc_row0{}, tmap_sc_row1{}, tmap_sc_row2{};
    alignas(64) CUtensorMap tmap_sc_col0{}, tmap_sc_col1{}, tmap_sc_col2{};

    create_tma_2d(tmap_in0, input0.data_ptr(), M, n0, BUFF_DIM_Y, BUFF_DIM_X, input0.stride(0), 16);
    create_tma_2d(tmap_in1, input1.data_ptr(), M, n1, BUFF_DIM_Y, BUFF_DIM_X, input1.stride(0), 16);
    create_tma_2d(tmap_in2, input2.data_ptr(), M, n2, BUFF_DIM_Y, BUFF_DIM_X, input2.stride(0), 16);

    create_raw_output_tmaps_strided<true>(row_fp4_0, row_sc_0, col_fp4_0, col_sc_0,
                                  tmap_out0, tmap_out_t0, tmap_sc_row0, tmap_sc_col0);
    create_raw_output_tmaps_strided<true>(row_fp4_1, row_sc_1, col_fp4_1, col_sc_1,
                                  tmap_out1, tmap_out_t1, tmap_sc_row1, tmap_sc_col1);
    create_raw_output_tmaps_strided<true>(row_fp4_2, row_sc_2, col_fp4_2, col_sc_2,
                                  tmap_out2, tmap_out_t2, tmap_sc_row2, tmap_sc_col2);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input0.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto advancing_rng_state = sr_axes.row
        ? make_localcta_advancing_rng_state(
              rng_seed, rng_subsequence_base, stream,
              persistent_rng_state)
        : torch::Tensor();
    const auto* advancing_rng_state_ptr = advancing_rng_state.defined()
        ? reinterpret_cast<const unsigned long long*>(
              advancing_rng_state.data_ptr<int64_t>())
        : nullptr;

    const int blocks_Y = static_cast<int>(M / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X0 = static_cast<int>(n0 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X1 = static_cast<int>(n1 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X2 = static_cast<int>(n2 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X = blocks_X0 + blocks_X1 + blocks_X2;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = shmem_size<true>();
    auto& ci = get_localcta_cached_info();
    int num_persistent = ci.max_bps_t * ci.num_sms;
    const int persistent_divisor = get_v3_split2_raw_persistent_divisor();
    if (persistent_divisor > 1) {
        num_persistent = (num_persistent + persistent_divisor - 1) / persistent_divisor;
    }
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
#define LAUNCH_SPLIT3_RAW(ENCODE_MODE, ROW_SR, FAST_SR, COL_RHT, RANDOM_SIGN) \
    do { \
    auto kernel = fused_localcta_quantize_split3_raw_kernel< \
        ENCODE_MODE, ROW_SR, FAST_SR, COL_RHT, RANDOM_SIGN>; \
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem); \
    kernel<<<num_persistent, THREADS, dshmem, stream>>>( \
        tmap_in0, tmap_in1, tmap_in2, \
        tmap_out0, tmap_out1, tmap_out2, \
        tmap_out_t0, tmap_out_t1, tmap_out_t2, \
        tmap_sc_row0, tmap_sc_row1, tmap_sc_row2, \
        tmap_sc_col0, tmap_sc_col1, tmap_sc_col2, \
        row_sg_chunk_0.data_ptr<float>(), \
        row_sg_chunk_1.data_ptr<float>(), \
        row_sg_chunk_2.data_ptr<float>(), \
        col_sg_chunk_0.data_ptr<float>(), \
        col_sg_chunk_1.data_ptr<float>(), \
        col_sg_chunk_2.data_ptr<float>(), \
        M, args, blocks_X0, blocks_X1, blocks_X2, \
        rng_seed, rng_subsequence_base, advancing_rng_state_ptr); \
    } while (0)

    const bool fast_data_sr = sr_axes.row && use_localcta_v4_fast_data_sr();
    if (encode_centric) {
        if (sr_axes.row) {
            if (fast_data_sr) {
                if (col_rht) {
                    if (with_random_sign_mask) {
                        LAUNCH_SPLIT3_RAW(true, true, true, true, true);
                    } else {
                        LAUNCH_SPLIT3_RAW(true, true, true, true, false);
                    }
                } else {
                    LAUNCH_SPLIT3_RAW(true, true, true, false, false);
                }
            } else if (col_rht) {
                if (with_random_sign_mask) {
                    LAUNCH_SPLIT3_RAW(true, true, false, true, true);
                } else {
                    LAUNCH_SPLIT3_RAW(true, true, false, true, false);
                }
            } else {
                LAUNCH_SPLIT3_RAW(true, true, false, false, false);
            }
        } else {
            LAUNCH_SPLIT3_RAW(true, false, false, false, false);
        }
    } else if (sr_axes.row) {
        if (fast_data_sr) {
            if (col_rht) {
                if (with_random_sign_mask) {
                    LAUNCH_SPLIT3_RAW(false, true, true, true, true);
                } else {
                    LAUNCH_SPLIT3_RAW(false, true, true, true, false);
                }
            } else {
                LAUNCH_SPLIT3_RAW(false, true, true, false, false);
            }
        } else if (col_rht) {
            if (with_random_sign_mask) {
                LAUNCH_SPLIT3_RAW(false, true, false, true, true);
            } else {
                LAUNCH_SPLIT3_RAW(false, true, false, true, false);
            }
        } else {
            LAUNCH_SPLIT3_RAW(false, true, false, false, false);
        }
    } else {
        LAUNCH_SPLIT3_RAW(false, false, false, false, false);
    }
#undef LAUNCH_SPLIT3_RAW
}

static LocalCTADataSRAxes resolve_localcta_data_sr_axes(
    bool enabled,
    std::string axes,
    const char* context
) {
    std::transform(axes.begin(), axes.end(), axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(axes.begin(), axes.end(), '-', '_');
    if (axes == "all" || axes == "row_col" || axes == "rowcol") {
        axes = "both";
    } else if (axes == "column" || axes == "columns" || axes == "wgrad") {
        axes = "col";
    } else if (axes == "dgrad") {
        axes = "row";
    } else if (axes == "off" || axes == "0") {
        axes = "none";
    }
    TORCH_CHECK(axes == "none" || axes == "row" || axes == "col" || axes == "both",
                "Unsupported ", context, " data SR axes: ", axes);
    return {
        enabled && (axes == "row" || axes == "both"),
        enabled && (axes == "col" || axes == "both"),
    };
}

static void launch_localcta_split3_quant_final_sg(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_2,
    torch::Tensor col_fp4_2,
    torch::Tensor col_sc_2,
    torch::Tensor row_sg_2,
    torch::Tensor col_sg_2,
    torch::Tensor rope_cs = torch::Tensor(),
    int64_t rope_seq_len = 0,
    bool data_sr = false,
    bool scale_sr = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    std::string data_sr_axes = "both",
    torch::Tensor persistent_rng_state = torch::Tensor()
) {
    using namespace tk_localcta;
    const bool apply_inverse_rope = rope_cs.defined() && rope_cs.numel() > 0;
    if (apply_inverse_rope) {
        check_rope_live64_tensor(rope_cs, rope_seq_len);
    }
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);

    alignas(64) CUtensorMap tmap_in0{}, tmap_in1{}, tmap_in2{};
    alignas(64) CUtensorMap tmap_out0{}, tmap_out1{}, tmap_out2{};
    alignas(64) CUtensorMap tmap_out_t0{}, tmap_out_t1{}, tmap_out_t2{};
    alignas(64) CUtensorMap tmap_sc_row0{}, tmap_sc_row1{}, tmap_sc_row2{};
    alignas(64) CUtensorMap tmap_sc_col0{}, tmap_sc_col1{}, tmap_sc_col2{};

    create_tma_2d(tmap_in0, input0.data_ptr(), M, n0, BUFF_DIM_Y, BUFF_DIM_X, input0.stride(0), 16);
    create_tma_2d(tmap_in1, input1.data_ptr(), M, n1, BUFF_DIM_Y, BUFF_DIM_X, input1.stride(0), 16);
    create_tma_2d(tmap_in2, input2.data_ptr(), M, n2, BUFF_DIM_Y, BUFF_DIM_X, input2.stride(0), 16);

    create_raw_output_tmaps_strided<true>(row_fp4_0, row_sc_0, col_fp4_0, col_sc_0,
                                  tmap_out0, tmap_out_t0, tmap_sc_row0, tmap_sc_col0);
    create_raw_output_tmaps_strided<true>(row_fp4_1, row_sc_1, col_fp4_1, col_sc_1,
                                  tmap_out1, tmap_out_t1, tmap_sc_row1, tmap_sc_col1);
    create_raw_output_tmaps_strided<true>(row_fp4_2, row_sc_2, col_fp4_2, col_sc_2,
                                  tmap_out2, tmap_out_t2, tmap_sc_row2, tmap_sc_col2);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input0.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

    const int blocks_Y = static_cast<int>(M / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X0 = static_cast<int>(n0 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X1 = static_cast<int>(n1 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X2 = static_cast<int>(n2 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X = blocks_X0 + blocks_X1 + blocks_X2;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = shmem_size<true>();
    auto& ci = get_localcta_cached_info();
    int num_persistent = ci.max_bps_t * ci.num_sms;
    const int persistent_divisor = get_v3_split2_raw_persistent_divisor();
    if (persistent_divisor > 1) {
        num_persistent = (num_persistent + persistent_divisor - 1) / persistent_divisor;
    }
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    const auto* rope_ptr = apply_inverse_rope
        ? reinterpret_cast<const float2*>(rope_cs.data_ptr<float>())
        : nullptr;
    const int rope_seq_mask = apply_inverse_rope ? static_cast<int>(rope_seq_len - 1) : 0;
    const bool fold_row_sg_in_producer = use_localcta_v4_split3_fold_row_sg_in_producer();
    const auto sr_axes = resolve_localcta_data_sr_axes(
        data_sr, std::move(data_sr_axes), "localCTA v4 split3");
    const bool row_data_sr = sr_axes.row;
    const bool col_data_sr = sr_axes.col;
    TORCH_CHECK(!(fold_row_sg_in_producer && (row_data_sr || col_data_sr || scale_sr)),
                "split3 inverse-RoPE SR does not support folded row outer-SG scales yet");
    const bool fast_data_sr =
        (row_data_sr || col_data_sr) && use_localcta_v4_fast_data_sr();
    torch::Tensor advancing_rng_state;
    const unsigned long long* advancing_rng_state_ptr = nullptr;
    if (row_data_sr || col_data_sr || scale_sr) {
        advancing_rng_state = make_localcta_advancing_rng_state(
            rng_seed, rng_subsequence_base, stream,
            persistent_rng_state);
        advancing_rng_state_ptr = reinterpret_cast<const unsigned long long*>(
            advancing_rng_state.data_ptr<int64_t>());
    }

#define LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, ROW_DATA, COL_DATA, SCALE, FAST) \
    do {                                                                               \
        auto kernel = fused_localcta_quantize_split3_final_sg_kernel<                  \
            true, DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, ROW_DATA, FAST, SCALE, COL_DATA>; \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                             dshmem);                                                  \
        kernel<<<num_persistent, THREADS, dshmem, stream>>>(                           \
            tmap_in0, tmap_in1, tmap_in2,                                              \
            tmap_out0, tmap_out1, tmap_out2,                                           \
            tmap_out_t0, tmap_out_t1, tmap_out_t2,                                     \
            tmap_sc_row0, tmap_sc_row1, tmap_sc_row2,                                  \
            tmap_sc_col0, tmap_sc_col1, tmap_sc_col2,                                  \
            row_sg_0.data_ptr<float>(),                                                \
            row_sg_1.data_ptr<float>(),                                                \
            row_sg_2.data_ptr<float>(),                                                \
            col_sg_0.data_ptr<float>(),                                                \
            col_sg_1.data_ptr<float>(),                                                \
            col_sg_2.data_ptr<float>(),                                                \
            (APPLY_ROPE) ? rope_ptr : nullptr,                                         \
            (APPLY_ROPE) ? rope_seq_mask : 0,                                          \
            rng_seed,                                                                  \
            rng_subsequence_base,                                                      \
            advancing_rng_state_ptr,                                                   \
            M, args, blocks_X0, blocks_X1, blocks_X2);                                 \
    } while (0)

#define DISPATCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG)             \
    do {                                                                               \
        if (row_data_sr && col_data_sr && scale_sr) {                                 \
            if (fast_data_sr) {                                                        \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, true, true, true); \
            } else {                                                                   \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, true, true, false); \
            }                                                                          \
        } else if (row_data_sr && col_data_sr) {                                       \
            if (fast_data_sr) {                                                        \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, true, false, true); \
            } else {                                                                   \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, true, false, false); \
            }                                                                          \
        } else if (row_data_sr) {                                                       \
            if (scale_sr && fast_data_sr) {                                             \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, false, true, true); \
            } else if (scale_sr) {                                                      \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, false, true, false); \
            } else if (fast_data_sr) {                                                  \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, false, false, true); \
            } else {                                                                   \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, true, false, false, false); \
            }                                                                          \
        } else if (col_data_sr) {                                                       \
            if (scale_sr && fast_data_sr) {                                             \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, false, true, true, true); \
            } else if (scale_sr) {                                                      \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, false, true, true, false); \
            } else if (fast_data_sr) {                                                  \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, false, true, false, true); \
            } else {                                                                   \
                LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, false, true, false, false); \
            }                                                                          \
        } else if (scale_sr) {                                                         \
            LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, false, false, true, false); \
        } else {                                                                       \
            LAUNCH_SPLIT3_FINAL_SG(DIRECT_SWIZZLED, APPLY_ROPE, FOLD_ROW_SG, false, false, false, false); \
        }                                                                              \
    } while (0)

    if (use_v4_direct_swizzled_scales()) {
        if (apply_inverse_rope) {
            if (fold_row_sg_in_producer) {
                DISPATCH_SPLIT3_FINAL_SG(true, true, true);
            } else {
                DISPATCH_SPLIT3_FINAL_SG(true, true, false);
            }
        } else {
            if (fold_row_sg_in_producer) {
                DISPATCH_SPLIT3_FINAL_SG(true, false, true);
            } else {
                DISPATCH_SPLIT3_FINAL_SG(true, false, false);
            }
        }
    } else {
        if (apply_inverse_rope) {
            if (fold_row_sg_in_producer) {
                DISPATCH_SPLIT3_FINAL_SG(false, true, true);
            } else {
                DISPATCH_SPLIT3_FINAL_SG(false, true, false);
            }
        } else {
            if (fold_row_sg_in_producer) {
                DISPATCH_SPLIT3_FINAL_SG(false, false, true);
            } else {
                DISPATCH_SPLIT3_FINAL_SG(false, false, false);
            }
        }
    }
#undef DISPATCH_SPLIT3_FINAL_SG
#undef LAUNCH_SPLIT3_FINAL_SG
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_localcta_quantize_split3_final_sg_kernel failed: ", cudaGetErrorString(err));
}

static void launch_localcta_split2_quant_raw(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    bool use_precomputed_amax = false,
    bool use_prefinalized_outer_sg = false,
    torch::Tensor row_sg_0 = torch::Tensor(),
    torch::Tensor col_sg_0 = torch::Tensor(),
    torch::Tensor row_sg_1 = torch::Tensor(),
    torch::Tensor col_sg_1 = torch::Tensor(),
    torch::Tensor delayed_amax = torch::Tensor(),
    bool delayed_amax_precomputed = false,
    bool data_sr = false,
    bool scale_sr = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    torch::Tensor row_amax_input_0 = torch::Tensor(),
    torch::Tensor row_amax_input_1 = torch::Tensor(),
    torch::Tensor current_row_amax_out_0 = torch::Tensor(),
    torch::Tensor current_col_amax_out_0 = torch::Tensor(),
    torch::Tensor current_row_amax_out_1 = torch::Tensor(),
    torch::Tensor current_col_amax_out_1 = torch::Tensor(),
    torch::Tensor current_row_sg_outer_out_0 = torch::Tensor(),
    torch::Tensor current_col_sg_outer_out_0 = torch::Tensor(),
    torch::Tensor current_row_sg_outer_out_1 = torch::Tensor(),
    torch::Tensor current_col_sg_outer_out_1 = torch::Tensor(),
    std::string data_sr_axes = "both",
    torch::Tensor persistent_rng_state = torch::Tensor(),
    bool col_rht = false,
    bool with_random_sign_mask = false,
    bool encode_centric = true
) {
    using namespace tk_localcta;
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);

    alignas(64) CUtensorMap tmap_in0{}, tmap_in1{};
    alignas(64) CUtensorMap tmap_out0{}, tmap_out1{};
    alignas(64) CUtensorMap tmap_out_t0{}, tmap_out_t1{};
    alignas(64) CUtensorMap tmap_sc_row0{}, tmap_sc_row1{};
    alignas(64) CUtensorMap tmap_sc_col0{}, tmap_sc_col1{};

    create_tma_2d(tmap_in0, input0.data_ptr(), M, n0, BUFF_DIM_Y, BUFF_DIM_X, n0, 16);
    create_tma_2d(tmap_in1, input1.data_ptr(), M, n1, BUFF_DIM_Y, BUFF_DIM_X, n1, 16);

    create_raw_output_tmaps_strided<true>(row_fp4_0, row_sc_0, col_fp4_0, col_sc_0,
                                  tmap_out0, tmap_out_t0, tmap_sc_row0, tmap_sc_col0);
    create_raw_output_tmaps_strided<true>(row_fp4_1, row_sc_1, col_fp4_1, col_sc_1,
                                  tmap_out1, tmap_out_t1, tmap_sc_row1, tmap_sc_col1);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto& work_counter = get_localcta_persistent_counter(input0.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    const int blocks_Y = static_cast<int>(M / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X0 = static_cast<int>(n0 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X1 = static_cast<int>(n1 / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X = blocks_X0 + blocks_X1;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = shmem_size<true>();
    auto& ci = get_localcta_cached_info();
    int num_persistent = ci.max_bps_t * ci.num_sms;
    const int persistent_divisor = get_v3_split2_raw_persistent_divisor();
    if (persistent_divisor > 1) {
        num_persistent = (num_persistent + persistent_divisor - 1) / persistent_divisor;
    }
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };

    use_prefinalized_outer_sg = use_precomputed_amax && use_prefinalized_outer_sg;
    if (use_prefinalized_outer_sg) {
        TORCH_CHECK(row_sg_0.defined() && col_sg_0.defined() &&
                    row_sg_1.defined() && col_sg_1.defined(),
                    "prefinalized split2 outer SG requires row/col SG tensors");
    }

    const auto sr_axes = resolve_localcta_data_sr_axes(
        data_sr, std::move(data_sr_axes), "localCTA v4 split2");
    const bool row_data_sr = sr_axes.row;
    const bool col_data_sr = sr_axes.col;
    const bool fast_data_sr = (row_data_sr || col_data_sr) && use_localcta_v4_fast_data_sr();
    TORCH_CHECK(!with_random_sign_mask || col_rht,
                "split2 fixed-sign mask requires column RHT");
    if (col_rht) {
        TORCH_CHECK(row_data_sr && !col_data_sr && !scale_sr,
                    "split2 column RHT currently requires row-only data SR and scale SR off");
        TORCH_CHECK(!use_precomputed_amax && !use_prefinalized_outer_sg &&
                    !delayed_amax.defined() && !delayed_amax_precomputed,
                    "split2 column RHT does not support precomputed, prefinalized, or delayed amax");
        TORCH_CHECK(!row_amax_input_0.defined() && !row_amax_input_1.defined() &&
                    !current_row_amax_out_0.defined() && !current_row_amax_out_1.defined() &&
                    !current_col_amax_out_0.defined() && !current_col_amax_out_1.defined() &&
                    !current_row_sg_outer_out_0.defined() && !current_row_sg_outer_out_1.defined() &&
                    !current_col_sg_outer_out_0.defined() && !current_col_sg_outer_out_1.defined(),
                    "split2 column RHT does not support read-only or collected amax state");
        TORCH_CHECK(M % 256 == 0 && n0 % 256 == 0 && n1 % 256 == 0,
                    "split2 column RHT requires M and both split widths to be multiples of 256");
        TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                    "split2 column RHT requires the v4 outer-SG contract");
    }
    torch::Tensor advancing_rng_state;
    const unsigned long long* advancing_rng_state_ptr = nullptr;
    if (row_data_sr || col_data_sr || scale_sr || with_random_sign_mask) {
        advancing_rng_state = make_localcta_advancing_rng_state(
            rng_seed, rng_subsequence_base, stream,
            persistent_rng_state);
        advancing_rng_state_ptr = reinterpret_cast<const unsigned long long*>(
            advancing_rng_state.data_ptr<int64_t>());
    }
    const bool use_readonly_tile_amax = row_amax_input_0.defined() || row_amax_input_1.defined();
    if (use_readonly_tile_amax) {
        TORCH_CHECK(row_amax_input_0.defined() && row_amax_input_1.defined(),
                    "read-only localCTA split2 amax input requires both split tensors");
        TORCH_CHECK(row_amax_input_0.is_cuda() && row_amax_input_0.is_contiguous() &&
                    row_amax_input_1.is_cuda() && row_amax_input_1.is_contiguous(),
                    "read-only localCTA split2 amax inputs must be contiguous CUDA tensors");
        TORCH_CHECK(row_amax_input_0.scalar_type() == torch::kFloat32 &&
                    row_amax_input_1.scalar_type() == torch::kFloat32,
                    "read-only localCTA split2 amax inputs must be float32");
        TORCH_CHECK(row_amax_input_0.sizes() == row_sg_chunk_0.sizes() &&
                    row_amax_input_1.sizes() == row_sg_chunk_1.sizes(),
                    "read-only localCTA split2 amax input shapes must match row SG chunk shapes");
        TORCH_CHECK(use_precomputed_amax && !delayed_amax.defined(),
                    "read-only localCTA split2 amax input is only supported for precomputed amax output");
    }
    const bool collect_current_tile_amax = (
        current_row_amax_out_0.defined() || current_row_amax_out_1.defined());
    if (collect_current_tile_amax) {
        TORCH_CHECK(use_readonly_tile_amax,
                    "current localCTA split2 amax output requires read-only previous amax input");
        auto check_current_tile_amax_tensor = [](
            const torch::Tensor& tensor,
            const torch::Tensor& reference,
            const char* name
        ) {
            TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.is_contiguous(),
                        name, " must be a contiguous CUDA tensor");
            TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                        name, " must be float32");
            TORCH_CHECK(tensor.sizes() == reference.sizes(),
                        name, " has shape ", tensor.sizes(),
                        ", expected ", reference.sizes());
        };
        check_current_tile_amax_tensor(current_row_amax_out_0, row_sg_chunk_0, "current_row_amax_out_0");
        check_current_tile_amax_tensor(current_row_amax_out_1, row_sg_chunk_1, "current_row_amax_out_1");
        if (current_col_amax_out_0.defined()) {
            check_current_tile_amax_tensor(current_col_amax_out_0, col_sg_chunk_0, "current_col_amax_out_0");
        }
        if (current_col_amax_out_1.defined()) {
            check_current_tile_amax_tensor(current_col_amax_out_1, col_sg_chunk_1, "current_col_amax_out_1");
        }
    }
    const bool collect_current_outer_sg = (
        current_row_sg_outer_out_0.defined() || current_col_sg_outer_out_0.defined() ||
        current_row_sg_outer_out_1.defined() || current_col_sg_outer_out_1.defined());
    if (collect_current_outer_sg) {
        TORCH_CHECK(collect_current_tile_amax && use_prefinalized_outer_sg,
                    "current outer SG output requires current tile amax output and prefinalized outer SG input");
        auto check_current_outer_sg_tensor = [](
            const torch::Tensor& tensor,
            const torch::Tensor& reference,
            const char* name
        ) {
            TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.is_contiguous(),
                        name, " must be a contiguous CUDA tensor");
            TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                        name, " must be float32");
            TORCH_CHECK(tensor.sizes() == reference.sizes(),
                        name, " has shape ", tensor.sizes(),
                        ", expected ", reference.sizes());
        };
        check_current_outer_sg_tensor(current_row_sg_outer_out_0, row_sg_0, "current_row_sg_outer_out_0");
        check_current_outer_sg_tensor(current_col_sg_outer_out_0, col_sg_0, "current_col_sg_outer_out_0");
        check_current_outer_sg_tensor(current_row_sg_outer_out_1, row_sg_1, "current_row_sg_outer_out_1");
        check_current_outer_sg_tensor(current_col_sg_outer_out_1, col_sg_1, "current_col_sg_outer_out_1");
    }

#define LAUNCH_SPLIT2_RAW_IMPL(ENCODE_MODE, USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, ROW_DATA, COL_DATA, SCALE, FAST, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX, COL_RHT, RANDOM_SIGN) \
    do {                                                                               \
        auto kernel = fused_localcta_quantize_split2_raw_kernel<                       \
            ENCODE_MODE, USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, ROW_DATA, FAST, SCALE, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX, COL_DATA, COL_RHT, RANDOM_SIGN>; \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem); \
        float* amax_out0 = nullptr;                                                    \
        float* amax_out1 = nullptr;                                                    \
        if constexpr (DELAYED) {                                                       \
            TORCH_CHECK(delayed_amax.defined() && delayed_amax.is_cuda() &&            \
                        delayed_amax.scalar_type() == torch::kFloat32 &&               \
                        delayed_amax.numel() >= 2,                                     \
                        "delayed localCTA split2 raw quant requires a CUDA float32 amax buffer with at least two elements"); \
            amax_out0 = delayed_amax.data_ptr<float>();                                \
            amax_out1 = delayed_amax.data_ptr<float>() + 1;                            \
        }                                                                              \
        const float* row_amax_in0 = nullptr;                                           \
        const float* row_amax_in1 = nullptr;                                           \
        if constexpr (READONLY_AMAX) {                                                 \
            row_amax_in0 = row_amax_input_0.data_ptr<float>();                         \
            row_amax_in1 = row_amax_input_1.data_ptr<float>();                         \
        }                                                                              \
        float* current_row_amax0 = nullptr;                                            \
        float* current_col_amax0 = nullptr;                                            \
        float* current_row_amax1 = nullptr;                                            \
        float* current_col_amax1 = nullptr;                                            \
        float* current_row_sg_outer0 = nullptr;                                        \
        float* current_col_sg_outer0 = nullptr;                                        \
        float* current_row_sg_outer1 = nullptr;                                        \
        float* current_col_sg_outer1 = nullptr;                                        \
        if constexpr (COLLECT_AMAX) {                                                  \
            current_row_amax0 = current_row_amax_out_0.data_ptr<float>();              \
            current_row_amax1 = current_row_amax_out_1.data_ptr<float>();              \
            if (current_col_amax_out_0.defined()) {                                    \
                current_col_amax0 = current_col_amax_out_0.data_ptr<float>();          \
            }                                                                          \
            if (current_col_amax_out_1.defined()) {                                    \
                current_col_amax1 = current_col_amax_out_1.data_ptr<float>();          \
            }                                                                          \
            if (collect_current_outer_sg) {                                            \
                current_row_sg_outer0 = current_row_sg_outer_out_0.data_ptr<float>();   \
                current_col_sg_outer0 = current_col_sg_outer_out_0.data_ptr<float>();   \
                current_row_sg_outer1 = current_row_sg_outer_out_1.data_ptr<float>();   \
                current_col_sg_outer1 = current_col_sg_outer_out_1.data_ptr<float>();   \
            }                                                                          \
        }                                                                              \
        kernel<<<num_persistent, THREADS, dshmem, stream>>>(                           \
            tmap_in0, tmap_in1,                                                        \
            tmap_out0, tmap_out1,                                                      \
            tmap_out_t0, tmap_out_t1,                                                  \
            tmap_sc_row0, tmap_sc_row1,                                                \
            tmap_sc_col0, tmap_sc_col1,                                                \
            row_sg_chunk_0.data_ptr<float>(),                                          \
            row_sg_chunk_1.data_ptr<float>(),                                          \
            col_sg_chunk_0.data_ptr<float>(),                                          \
            col_sg_chunk_1.data_ptr<float>(),                                          \
            (USE_FINAL_SG) ? row_sg_0.data_ptr<float>() : nullptr,                     \
            (USE_FINAL_SG) ? row_sg_1.data_ptr<float>() : nullptr,                     \
            (USE_FINAL_SG) ? col_sg_0.data_ptr<float>() : nullptr,                     \
            (USE_FINAL_SG) ? col_sg_1.data_ptr<float>() : nullptr,                     \
            amax_out0,                                                                 \
            amax_out1,                                                                 \
            row_amax_in0,                                                              \
            row_amax_in1,                                                              \
            current_row_amax0,                                                         \
            current_col_amax0,                                                         \
            current_row_amax1,                                                         \
            current_col_amax1,                                                         \
            current_row_sg_outer0,                                                     \
            current_col_sg_outer0,                                                     \
            current_row_sg_outer1,                                                     \
            current_col_sg_outer1,                                                     \
            M, args, blocks_X0, blocks_X1, rng_seed, rng_subsequence_base,             \
            advancing_rng_state_ptr);                                                  \
    } while (0)

#define LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, ROW_DATA, COL_DATA, SCALE, FAST, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX) \
    LAUNCH_SPLIT2_RAW_IMPL(true, USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, ROW_DATA, COL_DATA, SCALE, FAST, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX, false, false)

#define LAUNCH_SPLIT2_RAW_RHT(ENCODE_MODE, FAST, RANDOM_SIGN) \
    LAUNCH_SPLIT2_RAW_IMPL(ENCODE_MODE, false, false, false, true, false, false, FAST, false, false, false, false, true, RANDOM_SIGN)

#define DISPATCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX) \
    do {                                                                               \
        if (row_data_sr && col_data_sr && scale_sr) {                                 \
            if (fast_data_sr) {                                                        \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, true, true, true, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else {                                                                   \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, true, true, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            }                                                                          \
        } else if (row_data_sr && col_data_sr) {                                       \
            if (fast_data_sr) {                                                        \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, true, false, true, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else {                                                                   \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, true, false, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            }                                                                          \
        } else if (row_data_sr) {                                                       \
            if (scale_sr && fast_data_sr) {                                             \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, false, true, true, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else if (scale_sr) {                                                      \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, false, true, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else if (fast_data_sr) {                                                  \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, false, false, true, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else {                                                                   \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, true, false, false, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            }                                                                          \
        } else if (col_data_sr) {                                                       \
            if (scale_sr && fast_data_sr) {                                             \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, false, true, true, true, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else if (scale_sr) {                                                      \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, false, true, true, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else if (fast_data_sr) {                                                  \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, false, true, false, true, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            } else {                                                                   \
                LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, false, true, false, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
            }                                                                          \
        } else if (scale_sr) {                                                         \
            LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, false, false, true, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
        } else {                                                                       \
            LAUNCH_SPLIT2_RAW(USE_AMAX, USE_FINAL_SG, DIRECT_SWIZZLED, false, false, false, false, DELAYED, DELAYED_READY, READONLY_AMAX, COLLECT_AMAX); \
        }                                                                              \
    } while (0)

    const bool use_delayed_scaling = delayed_amax.defined();
    if (col_rht) {
        if (encode_centric) {
            if (fast_data_sr) {
                if (with_random_sign_mask) {
                    LAUNCH_SPLIT2_RAW_RHT(true, true, true);
                } else {
                    LAUNCH_SPLIT2_RAW_RHT(true, true, false);
                }
            } else if (with_random_sign_mask) {
                LAUNCH_SPLIT2_RAW_RHT(true, false, true);
            } else {
                LAUNCH_SPLIT2_RAW_RHT(true, false, false);
            }
        } else {
            if (fast_data_sr) {
                if (with_random_sign_mask) {
                    LAUNCH_SPLIT2_RAW_RHT(false, true, true);
                } else {
                    LAUNCH_SPLIT2_RAW_RHT(false, true, false);
                }
            } else if (with_random_sign_mask) {
                LAUNCH_SPLIT2_RAW_RHT(false, false, true);
            } else {
                LAUNCH_SPLIT2_RAW_RHT(false, false, false);
            }
        }
    } else if (use_readonly_tile_amax) {
        if (use_prefinalized_outer_sg) {
            TORCH_CHECK(!row_data_sr && !col_data_sr && !scale_sr,
                        "read-only tile-amax with prefinalized outer SG currently supports deterministic scale/data only");
            const bool direct_swizzled = use_v4_direct_swizzled_scales();
            if (collect_current_tile_amax) {
                if (direct_swizzled) {
                    LAUNCH_SPLIT2_RAW(true, true, true, false, false, false, false, false, false, true, true);
                } else {
                    LAUNCH_SPLIT2_RAW(true, true, false, false, false, false, false, false, false, true, true);
                }
            } else {
                if (direct_swizzled) {
                    LAUNCH_SPLIT2_RAW(true, true, true, false, false, false, false, false, false, true, false);
                } else {
                    LAUNCH_SPLIT2_RAW(true, true, false, false, false, false, false, false, false, true, false);
                }
            }
        } else {
            if (collect_current_tile_amax) {
                DISPATCH_SPLIT2_RAW(true, false, false, false, false, true, true);
            } else {
                DISPATCH_SPLIT2_RAW(true, false, false, false, false, true, false);
            }
        }
    } else if (use_delayed_scaling && use_precomputed_amax) {
        if (delayed_amax_precomputed && use_prefinalized_outer_sg) {
            DISPATCH_SPLIT2_RAW(true, true, false, true, true, false, false);
        } else if (delayed_amax_precomputed) {
            DISPATCH_SPLIT2_RAW(true, false, false, true, true, false, false);
        } else {
            DISPATCH_SPLIT2_RAW(true, false, false, true, false, false, false);
        }
    } else if (use_precomputed_amax && use_prefinalized_outer_sg) {
        if (use_v4_direct_swizzled_scales()) {
            DISPATCH_SPLIT2_RAW(true, true, true, false, false, false, false);
        } else {
            DISPATCH_SPLIT2_RAW(true, true, false, false, false, false, false);
        }
    } else if (use_precomputed_amax) {
        DISPATCH_SPLIT2_RAW(true, false, false, false, false, false, false);
    } else {
        DISPATCH_SPLIT2_RAW(false, false, false, false, false, false, false);
    }

#undef DISPATCH_SPLIT2_RAW
#undef LAUNCH_SPLIT2_RAW_RHT
#undef LAUNCH_SPLIT2_RAW
#undef LAUNCH_SPLIT2_RAW_IMPL
}

constexpr int kSplit2AmaxChunksPerTile = 2;
constexpr int kSplit2AmaxThreads = 128;

__device__ __forceinline__ unsigned int bf16x2_abs_max_bits(__nv_bfloat162 v) {
    const unsigned int bits = *reinterpret_cast<const unsigned int*>(&v);
    const unsigned int lo = bits & 0x7fffu;
    const unsigned int hi = (bits >> 16) & 0x7fffu;
    return lo > hi ? lo : hi;
}

__device__ __forceinline__ float split2_bf16_abs_bits_to_float(unsigned int bits) {
    union {
        unsigned int u;
        float f;
    } out;
    out.u = bits << 16;
    return out.f;
}

template <bool WRITE_AMAX = true>
// H1/out1 and H3/out2 may alias when backward recomputes the preactivations.
__global__ void silu_deriv_dual_split_tile_amax_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* h3,
    const __nv_bfloat16* h1_raw,
    __nv_bfloat16* out1,
    __nv_bfloat16* out2,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks1,
    float* __restrict__ row_sg_chunks2,
    float* __restrict__ col_sg_chunks2,
    float* __restrict__ row_sg_outer1,
    float* __restrict__ col_sg_outer1,
    float* __restrict__ row_sg_outer2,
    float* __restrict__ col_sg_outer2,
    float* __restrict__ delayed_amax,
    int64_t M,
    int64_t H,
    int tiles_X,
    int tiles_Y,
    bool write_chunks_with_outer = false,
    bool fast_divide = false
) {
    constexpr int TILE = tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
    constexpr int VEC = 4;
    constexpr int VECS_PER_TILE = (TILE * TILE) / VEC;
    constexpr int VECS_PER_CHUNK =
        (VECS_PER_TILE + kSplit2AmaxChunksPerTile - 1) / kSplit2AmaxChunksPerTile;

    const int chunk_id = blockIdx.x;
    const int tile_id = chunk_id / kSplit2AmaxChunksPerTile;
    const int sub_chunk = chunk_id - tile_id * kSplit2AmaxChunksPerTile;
    const int tile_y = tile_id / tiles_X;
    const int tile_x = tile_id - tile_y * tiles_X;
    if (tile_y >= tiles_Y) {
        return;
    }

    const int vec_begin = sub_chunk * VECS_PER_CHUNK;
    const int vec_end = min(vec_begin + VECS_PER_CHUNK, VECS_PER_TILE);
    unsigned int local_max1_bits = 0;
    unsigned int local_max2_bits = 0;

    for (int v = vec_begin + threadIdx.x; v < vec_end; v += blockDim.x) {
        const int elem = v * VEC;
        const int local_row = elem / TILE;
        const int local_col = elem - local_row * TILE;
        const int global_row = tile_y * TILE + local_row;
        const int global_col = tile_x * TILE + local_col;
        const int64_t base = static_cast<int64_t>(global_row) * H + global_col;

        const int2 d = *reinterpret_cast<const int2*>(dh + base);
        const int2 a = *reinterpret_cast<const int2*>(h3 + base);
        const int2 b = *reinterpret_cast<const int2*>(h1_raw + base);

        const __nv_bfloat162 d0 = *reinterpret_cast<const __nv_bfloat162*>(&d.x);
        const __nv_bfloat162 d1 = *reinterpret_cast<const __nv_bfloat162*>(&d.y);
        const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
        const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
        const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
        const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);

        const float2 d0f = __bfloat1622float2(d0);
        const float2 d1f = __bfloat1622float2(d1);
        const float2 a0f = __bfloat1622float2(a0);
        const float2 a1f = __bfloat1622float2(a1);
        const float2 b0f = __bfloat1622float2(b0);
        const float2 b1f = __bfloat1622float2(b1);

        const float denom0x = 1.0f + __expf(-b0f.x);
        const float denom0y = 1.0f + __expf(-b0f.y);
        const float denom1x = 1.0f + __expf(-b1f.x);
        const float denom1y = 1.0f + __expf(-b1f.y);
        const float sig0x = fast_divide ? __fdividef(1.0f, denom0x) : 1.0f / denom0x;
        const float sig0y = fast_divide ? __fdividef(1.0f, denom0y) : 1.0f / denom0y;
        const float sig1x = fast_divide ? __fdividef(1.0f, denom1x) : 1.0f / denom1x;
        const float sig1y = fast_divide ? __fdividef(1.0f, denom1y) : 1.0f / denom1y;

        const float silu0x = b0f.x * sig0x;
        const float silu0y = b0f.y * sig0y;
        const float silu1x = b1f.x * sig1x;
        const float silu1y = b1f.y * sig1y;

        const float silup0x = sig0x * (1.0f + b0f.x - silu0x);
        const float silup0y = sig0y * (1.0f + b0f.y - silu0y);
        const float silup1x = sig1x * (1.0f + b1f.x - silu1x);
        const float silup1y = sig1y * (1.0f + b1f.y - silu1y);

        const __nv_bfloat162 o10 = __float22bfloat162_rn(
            make_float2(d0f.x * a0f.x * silup0x, d0f.y * a0f.y * silup0y));
        const __nv_bfloat162 o11 = __float22bfloat162_rn(
            make_float2(d1f.x * a1f.x * silup1x, d1f.y * a1f.y * silup1y));
        const __nv_bfloat162 o20 = __float22bfloat162_rn(
            make_float2(d0f.x * silu0x, d0f.y * silu0y));
        const __nv_bfloat162 o21 = __float22bfloat162_rn(
            make_float2(d1f.x * silu1x, d1f.y * silu1y));

        const unsigned int o10_bits = *reinterpret_cast<const unsigned int*>(&o10);
        const unsigned int o11_bits = *reinterpret_cast<const unsigned int*>(&o11);
        const unsigned int o20_bits = *reinterpret_cast<const unsigned int*>(&o20);
        const unsigned int o21_bits = *reinterpret_cast<const unsigned int*>(&o21);

        int2 r1, r2;
        r1.x = static_cast<int>(o10_bits);
        r1.y = static_cast<int>(o11_bits);
        r2.x = static_cast<int>(o20_bits);
        r2.y = static_cast<int>(o21_bits);
        *reinterpret_cast<int2*>(out1 + base) = r1;
        *reinterpret_cast<int2*>(out2 + base) = r2;

        if constexpr (WRITE_AMAX) {
            const unsigned int max10 = bf16x2_abs_max_bits(o10);
            const unsigned int max11 = bf16x2_abs_max_bits(o11);
            const unsigned int max20 = bf16x2_abs_max_bits(o20);
            const unsigned int max21 = bf16x2_abs_max_bits(o21);
            local_max1_bits = max10 > local_max1_bits ? max10 : local_max1_bits;
            local_max1_bits = max11 > local_max1_bits ? max11 : local_max1_bits;
            local_max2_bits = max20 > local_max2_bits ? max20 : local_max2_bits;
            local_max2_bits = max21 > local_max2_bits ? max21 : local_max2_bits;
        }
    }

    if constexpr (WRITE_AMAX) {
        for (int mask = 16; mask > 0; mask >>= 1) {
            const unsigned int other1 = __shfl_xor_sync(0xffffffff, local_max1_bits, mask);
            const unsigned int other2 = __shfl_xor_sync(0xffffffff, local_max2_bits, mask);
            local_max1_bits = other1 > local_max1_bits ? other1 : local_max1_bits;
            local_max2_bits = other2 > local_max2_bits ? other2 : local_max2_bits;
        }

        __shared__ unsigned int warp_max1[kSplit2AmaxThreads / 32];
        __shared__ unsigned int warp_max2[kSplit2AmaxThreads / 32];
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        if (lane == 0) {
            warp_max1[wid] = local_max1_bits;
            warp_max2[wid] = local_max2_bits;
        }
        __syncthreads();

        if (wid == 0) {
            unsigned int block_max1_bits = (lane < (kSplit2AmaxThreads / 32)) ? warp_max1[lane] : 0;
            unsigned int block_max2_bits = (lane < (kSplit2AmaxThreads / 32)) ? warp_max2[lane] : 0;
        for (int mask = 16; mask > 0; mask >>= 1) {
            const unsigned int other1 = __shfl_xor_sync(0xffffffff, block_max1_bits, mask);
            const unsigned int other2 = __shfl_xor_sync(0xffffffff, block_max2_bits, mask);
            block_max1_bits = other1 > block_max1_bits ? other1 : block_max1_bits;
            block_max2_bits = other2 > block_max2_bits ? other2 : block_max2_bits;
        }
        if (lane == 0) {
            const float block_max1 = split2_bf16_abs_bits_to_float(block_max1_bits);
            const float block_max2 = split2_bf16_abs_bits_to_float(block_max2_bits);
            if (delayed_amax != nullptr) {
                transformer_engine::atomicMaxFloat(delayed_amax, block_max1);
                transformer_engine::atomicMaxFloat(delayed_amax + 1, block_max2);
            } else if (row_sg_outer1 != nullptr) {
                const int row_tile = tile_y / 2;
                const int col_tile = tile_x / 2;
                const float sg1 = block_max1 / tk_localcta::localcta_global_scale_num();
                const float sg2 = block_max2 / tk_localcta::localcta_global_scale_num();
                transformer_engine::atomicMaxFloat(row_sg_outer1 + row_tile, sg1);
                transformer_engine::atomicMaxFloat(col_sg_outer1 + col_tile, sg1);
                transformer_engine::atomicMaxFloat(row_sg_outer2 + row_tile, sg2);
                transformer_engine::atomicMaxFloat(col_sg_outer2 + col_tile, sg2);
                if (write_chunks_with_outer) {
                    const int row_idx = tile_y * tiles_X + tile_x;
                    if constexpr (kSplit2AmaxChunksPerTile == 1) {
                        row_sg_chunks1[row_idx] = block_max1;
                        row_sg_chunks2[row_idx] = block_max2;
                    } else {
                        transformer_engine::atomicMaxFloat(row_sg_chunks1 + row_idx, block_max1);
                        transformer_engine::atomicMaxFloat(row_sg_chunks2 + row_idx, block_max2);
                    }
                    const int col_idx = tile_x * tiles_Y + tile_y;
                    transformer_engine::atomicMaxFloat(col_sg_chunks1 + col_idx, block_max1);
                    transformer_engine::atomicMaxFloat(col_sg_chunks2 + col_idx, block_max2);
                }
            } else {
                const int row_idx = tile_y * tiles_X + tile_x;
                if constexpr (kSplit2AmaxChunksPerTile == 1) {
                    row_sg_chunks1[row_idx] = block_max1;
                    row_sg_chunks2[row_idx] = block_max2;
                } else {
                    transformer_engine::atomicMaxFloat(row_sg_chunks1 + row_idx, block_max1);
                    transformer_engine::atomicMaxFloat(row_sg_chunks2 + row_idx, block_max2);
                }
                const int col_idx = tile_x * tiles_Y + tile_y;
                transformer_engine::atomicMaxFloat(col_sg_chunks1 + col_idx, block_max1);
                transformer_engine::atomicMaxFloat(col_sg_chunks2 + col_idx, block_max2);
            }
        }
    }
    }
}

static bool launch_silu_deriv_split_with_tile_amax(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1,
    torch::Tensor dh3_out,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    bool fill_outer_sg = false,
    torch::Tensor row_sg_0 = torch::Tensor(),
    torch::Tensor col_sg_0 = torch::Tensor(),
    torch::Tensor row_sg_1 = torch::Tensor(),
    torch::Tensor col_sg_1 = torch::Tensor(),
    torch::Tensor delayed_amax = torch::Tensor(),
    bool skip_amax_outputs = false,
    bool write_chunks_with_outer = false
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    if (M % tk_localcta::LocalCTAConfig::CHUNK_DIM_Y != 0 ||
        H % tk_localcta::LocalCTAConfig::CHUNK_DIM_X != 0) {
        return false;
    }
    const int tiles_Y = static_cast<int>(M / tk_localcta::LocalCTAConfig::CHUNK_DIM_Y);
    const int tiles_X = static_cast<int>(H / tk_localcta::LocalCTAConfig::CHUNK_DIM_X);
    const int total_chunks = tiles_X * tiles_Y * kSplit2AmaxChunksPerTile;
    if (total_chunks <= 0) {
        return false;
    }
    const bool reduce_delayed_amax = delayed_amax.defined();
    if (skip_amax_outputs) {
        TORCH_CHECK(!fill_outer_sg && !reduce_delayed_amax,
                    "skip_amax_outputs cannot be combined with outer SG or delayed amax collection");
    }
    if (write_chunks_with_outer) {
        TORCH_CHECK(fill_outer_sg && !reduce_delayed_amax && !skip_amax_outputs,
                    "write_chunks_with_outer requires outer SG fill without delayed/global amax reduction");
    }
    if (reduce_delayed_amax) {
        TORCH_CHECK(delayed_amax.is_cuda() &&
                    delayed_amax.scalar_type() == torch::kFloat32 &&
                    delayed_amax.numel() >= 2,
                    "delayed localCTA split2 precompute requires a CUDA float32 amax buffer with at least two elements");
        fill_outer_sg = false;
    }
    fill_outer_sg = fill_outer_sg && row_sg_0.defined() && col_sg_0.defined() &&
                    row_sg_1.defined() && col_sg_1.defined();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if ((!fill_outer_sg || write_chunks_with_outer) && !reduce_delayed_amax && !skip_amax_outputs) {
        cudaMemsetAsync(row_sg_chunk_0.data_ptr<float>(), 0, row_sg_chunk_0.numel() * sizeof(float), stream);
        cudaMemsetAsync(row_sg_chunk_1.data_ptr<float>(), 0, row_sg_chunk_1.numel() * sizeof(float), stream);
        cudaMemsetAsync(col_sg_chunk_0.data_ptr<float>(), 0, col_sg_chunk_0.numel() * sizeof(float), stream);
        cudaMemsetAsync(col_sg_chunk_1.data_ptr<float>(), 0, col_sg_chunk_1.numel() * sizeof(float), stream);
    }
    if (fill_outer_sg) {
        cudaMemsetAsync(row_sg_0.data_ptr<float>(), 0, row_sg_0.numel() * sizeof(float), stream);
        cudaMemsetAsync(col_sg_0.data_ptr<float>(), 0, col_sg_0.numel() * sizeof(float), stream);
        cudaMemsetAsync(row_sg_1.data_ptr<float>(), 0, row_sg_1.numel() * sizeof(float), stream);
        cudaMemsetAsync(col_sg_1.data_ptr<float>(), 0, col_sg_1.numel() * sizeof(float), stream);
    }
    if (skip_amax_outputs) {
        silu_deriv_dual_split_tile_amax_kernel<false><<<
            total_chunks, kSplit2AmaxThreads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
                row_sg_chunk_0.data_ptr<float>(),
                col_sg_chunk_0.data_ptr<float>(),
                row_sg_chunk_1.data_ptr<float>(),
                col_sg_chunk_1.data_ptr<float>(),
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                M, H, tiles_X, tiles_Y, false,
                use_localcta_v4_silu_deriv_fast_divide());
    } else {
        silu_deriv_dual_split_tile_amax_kernel<true><<<
            total_chunks, kSplit2AmaxThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
            row_sg_chunk_0.data_ptr<float>(),
            col_sg_chunk_0.data_ptr<float>(),
            row_sg_chunk_1.data_ptr<float>(),
            col_sg_chunk_1.data_ptr<float>(),
            fill_outer_sg ? row_sg_0.data_ptr<float>() : nullptr,
            fill_outer_sg ? col_sg_0.data_ptr<float>() : nullptr,
            fill_outer_sg ? row_sg_1.data_ptr<float>() : nullptr,
            fill_outer_sg ? col_sg_1.data_ptr<float>() : nullptr,
            reduce_delayed_amax ? delayed_amax.data_ptr<float>() : nullptr,
            M, H, tiles_X, tiles_Y, write_chunks_with_outer,
            use_localcta_v4_silu_deriv_fast_divide());
    }
    return true;
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH>
static void launch_localcta_split2_quant_prepared_tuned(
    const CUtensorMap &tmap_in0,
    const CUtensorMap &tmap_in1,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t total_n,
    int split0_tiles,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (total_n + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_1cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_split2_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_persistent = max_bps * num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    kernel<<<num_persistent, TOTAL_THREADS, dshmem, stream>>>(
        tmap_in0, tmap_in1,
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, total_n, blocks_X, total_tiles, split0_tiles);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_split2_quant_prepared_tuned_dispatch(
    const CUtensorMap &tmap_in0,
    const CUtensorMap &tmap_in1,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t total_n,
    int split0_tiles,
    cudaStream_t stream
) {
    const auto cfg = get_localcta1_prepared_tuning();
    if (cfg.threads == 160 && cfg.pipe_depth == 1) {
        launch_localcta_split2_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2) {
        launch_localcta_split2_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1) {
        launch_localcta_split2_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2) {
        launch_localcta_split2_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1) {
        launch_localcta_split2_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2) {
        launch_localcta_split2_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else {
        TORCH_CHECK(false, "Unsupported split2 localCTA1 prepared tuning config: threads=", cfg.threads,
                    " pipe_depth=", cfg.pipe_depth);
    }
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX>
static void launch_localcta_split2_quant_2cta_prepared_tuned(
    const CUtensorMap &tmap_in0,
    const CUtensorMap &tmap_in1,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t total_n,
    int split0_tiles,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (total_n + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_2cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_split2_kernel_2cta_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, SHARED_AMAX, RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_clusters = max_bps * num_sms / 2;
    if (num_clusters > total_macro_tiles) {
        num_clusters = total_macro_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(TOTAL_THREADS, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        tmap_in0, tmap_in1, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, total_n, blocks_X, blocks_Y, total_macro_tiles, split0_tiles);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchKernelEx failed for tuned localCTA split2 2-CTA prepared quant: ",
                cudaGetErrorString(err));
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_split2_quant_2cta_prepared_tuned_dispatch(
    const CUtensorMap &tmap_in0,
    const CUtensorMap &tmap_in1,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t total_n,
    int split0_tiles,
    cudaStream_t stream
) {
    const auto cfg = get_localcta2_prepared_tuning();
    if (cfg.threads == 160 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 3, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 4, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 3, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 4, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 3, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 4, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 1, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 2, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 3, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 4, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 1, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 2, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 3, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 4, false>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 3, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 4, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 3, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 4, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 3, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 4, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 1, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 2, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 3, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 4, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 1, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 2, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 3, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_split2_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 4, true>(
            tmap_in0, tmap_in1, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, total_n, split0_tiles, stream);
    } else {
        TORCH_CHECK(false, "Unsupported split2 localCTA2 prepared tuning config: threads=", cfg.threads,
                    " pipe_depth=", cfg.pipe_depth, " shared_amax=", cfg.shared_amax);
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_split2_quant_prepared_impl(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_full,
    torch::Tensor row_sc_prepared_full,
    torch::Tensor col_fp4_full,
    torch::Tensor col_sc_prepared_full,
    torch::Tensor row_sg_full,
    torch::Tensor col_sg_full
) {
    using namespace tk_localcta;
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t total_n = n0 + n1;
    const int64_t ld0 = input0.stride(0);
    const int64_t ld1 = input1.stride(0);

    alignas(64) CUtensorMap tmap_in0{}, tmap_in1{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_in0, input0.data_ptr(), M, n0, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, ld0, 16);
    create_tma_2d(tmap_in1, input1.data_ptr(), M, n1, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, ld1, 16);
    create_tma_2d(tmap_out, row_fp4_full.data_ptr(), M, total_n,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, total_n, 4);
    if constexpr (RETURN_TRANSPOSE) {
        create_tma_2d(tmap_out_t, col_fp4_full.data_ptr(), total_n, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = total_n / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared_full.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = total_n / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col_prepared, col_sc_prepared_full.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input0.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X0 = (n0 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X1 = (n1 + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X = blocks_X0 + blocks_X1;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = shmem_size<RETURN_TRANSPOSE>();
    auto& ci = get_localcta_cached_info();
    auto kernel = fused_localcta_quantize_split2_prepared_kernel<RETURN_TRANSPOSE>;
    static int max_bps = -1;
    if (max_bps < 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_bps, kernel, THREADS, dshmem);
    }
    int num_persistent = max_bps * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };

    float* row_sg_ptr = row_sg_full.data_ptr<float>();
    float* col_sg_ptr = nullptr;
    if constexpr (RETURN_TRANSPOSE) {
        col_sg_ptr = col_sg_full.data_ptr<float>();
    }

    // Keep the tuned split2 backend as bring-up-only for now. The current
    // implementation is close, but it has not yet cleared the exactness/perf
    // gates against the live copy-based FFN route.
    const bool use_tuned = false && (input0.stride(1) == 1 && input1.stride(1) == 1);
    if (use_tuned) {
        const bool use_2cta_tuned = should_use_localcta2_prepared_auto(M, total_n);
        if (use_2cta_tuned) {
            if constexpr (RETURN_TRANSPOSE) {
                launch_localcta_split2_quant_2cta_prepared_tuned_dispatch<true, true>(
                    tmap_in0, tmap_in1, tmap_out, tmap_out_t,
                    tmap_sc_row_prepared, tmap_sc_col_prepared,
                    row_sg_ptr, col_sg_ptr,
                    M, total_n, blocks_X0, stream);
            } else {
                launch_localcta_split2_quant_2cta_prepared_tuned_dispatch<false, true>(
                    tmap_in0, tmap_in1, tmap_out, tmap_out_t,
                    tmap_sc_row_prepared, tmap_sc_col_prepared,
                    row_sg_ptr, col_sg_ptr,
                    M, total_n, blocks_X0, stream);
            }
        } else {
            if constexpr (RETURN_TRANSPOSE) {
                launch_localcta_split2_quant_prepared_tuned_dispatch<true, true>(
                    tmap_in0, tmap_in1, tmap_out, tmap_out_t,
                    tmap_sc_row_prepared, tmap_sc_col_prepared,
                    row_sg_ptr, col_sg_ptr,
                    M, total_n, blocks_X0, stream);
            } else {
                launch_localcta_split2_quant_prepared_tuned_dispatch<false, true>(
                    tmap_in0, tmap_in1, tmap_out, tmap_out_t,
                    tmap_sc_row_prepared, tmap_sc_col_prepared,
                    row_sg_ptr, col_sg_ptr,
                    M, total_n, blocks_X0, stream);
            }
        }
        return;
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in0, tmap_in1,
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, total_n, args, blocks_X0);
}

static void launch_localcta_split2_quant_prepared(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_full,
    torch::Tensor row_sc_prepared_full,
    torch::Tensor col_fp4_full,
    torch::Tensor col_sc_prepared_full,
    torch::Tensor row_sg_full,
    torch::Tensor col_sg_full
) {
    launch_localcta_split2_quant_prepared_impl<true>(
        input0, input1,
        row_fp4_full, row_sc_prepared_full,
        col_fp4_full, col_sc_prepared_full,
        row_sg_full, col_sg_full);
}

static void launch_localcta_split2_row_quant_prepared(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_full,
    torch::Tensor row_sc_prepared_full,
    torch::Tensor row_sg_full
) {
    launch_localcta_split2_quant_prepared_impl<false>(
        input0, input1,
        row_fp4_full, row_sc_prepared_full,
        torch::Tensor(), torch::Tensor(),
        row_sg_full, torch::Tensor());
}

template <bool OUTPUT_DH1>
__device__ __forceinline__ float load_silu_deriv_split2_chunk_explicit(
    tk_localcta::IType* sIn_ptr,
    const __nv_bfloat16* dh,
    const __nv_bfloat16* h3,
    const __nv_bfloat16* h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    float local_max = 0.0f;
    constexpr int VEC = 4;
    constexpr int TOTAL = tk_localcta::LocalCTAConfig::CHUNK_DIM_Y * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += tk_localcta::THREADS * VEC) {
        const int row = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * cols + global_col;
            const int2 d = *reinterpret_cast<const int2*>(dh + base);
            const int2 a = *reinterpret_cast<const int2*>(h3 + base);
            const int2 b = *reinterpret_cast<const int2*>(h1_raw + base);

            const __nv_bfloat162 d0 = *reinterpret_cast<const __nv_bfloat162*>(&d.x);
            const __nv_bfloat162 d1 = *reinterpret_cast<const __nv_bfloat162*>(&d.y);
            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);

            const float2 d0f = __bfloat1622float2(d0);
            const float2 d1f = __bfloat1622float2(d1);
            const float2 a0f = __bfloat1622float2(a0);
            const float2 a1f = __bfloat1622float2(a1);
            const float2 b0f = __bfloat1622float2(b0);
            const float2 b1f = __bfloat1622float2(b1);

            const float sig0x = 1.0f / (1.0f + expf(-b0f.x));
            const float sig0y = 1.0f / (1.0f + expf(-b0f.y));
            const float sig1x = 1.0f / (1.0f + expf(-b1f.x));
            const float sig1y = 1.0f / (1.0f + expf(-b1f.y));

            const float silu0x = b0f.x * sig0x;
            const float silu0y = b0f.y * sig0y;
            const float silu1x = b1f.x * sig1x;
            const float silu1y = b1f.y * sig1y;

            __nv_bfloat162 out0;
            __nv_bfloat162 out1;
            if constexpr (OUTPUT_DH1) {
                const float silup0x = sig0x * (1.0f + b0f.x - silu0x);
                const float silup0y = sig0y * (1.0f + b0f.y - silu0y);
                const float silup1x = sig1x * (1.0f + b1f.x - silu1x);
                const float silup1y = sig1y * (1.0f + b1f.y - silu1y);
                out0 = __float22bfloat162_rn(
                    make_float2(d0f.x * a0f.x * silup0x, d0f.y * a0f.y * silup0y));
                out1 = __float22bfloat162_rn(
                    make_float2(d1f.x * a1f.x * silup1x, d1f.y * a1f.y * silup1y));
            } else {
                out0 = __float22bfloat162_rn(make_float2(d0f.x * silu0x, d0f.y * silu0y));
                out1 = __float22bfloat162_rn(make_float2(d1f.x * silu1x, d1f.y * silu1y));
            }

            const float2 out0f = __bfloat1622float2(out0);
            const float2 out1f = __bfloat1622float2(out1);
            local_max = fmaxf(local_max, fabsf(out0f.x));
            local_max = fmaxf(local_max, fabsf(out0f.y));
            local_max = fmaxf(local_max, fabsf(out1f.x));
            local_max = fmaxf(local_max, fabsf(out1f.y));

            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 0, out0.x);
            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 1, out0.y);
            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 2, out1.x);
            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 3, out1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    const float sig = 1.0f / (1.0f + expf(-v1));
                    const float silu_v1 = v1 * sig;
                    float transformed;
                    if constexpr (OUTPUT_DH1) {
                        const float silup_v1 = sig * (1.0f + v1 - silu_v1);
                        transformed = vd * v3 * silup_v1;
                    } else {
                        transformed = vd * silu_v1;
                    }
                    out = __float2bfloat16_rn(transformed);
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(out)));
                }
                tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
    return local_max;
}

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(tk_localcta::THREADS)
localcta_silu_deriv_split2_prepared_direct_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t split_cols,
    const size_t total_cols,
    tk_localcta::LocalCTAPersistentArgs args,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    __shared__ float warp_max[tk_localcta::THREADS / 32];
    __shared__ float cta_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = transformer_engine::common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFFS_NUM_IN * tk_localcta::BUFF_IN_ELEMS * (int)sizeof(tk_localcta::IType),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFFS_NUM_OUT * tk_localcta::BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(tk_localcta::BUFFS_NUM_OUT_TR * tk_localcta::BUFF_OUT_TR_SIZE,
                          TMA_SHMEM_ALIGNMENT)
        : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::LocalCTAConfig::CHUNK_DIM_Y * tk_localcta::SCALES_PER_CHUNK_X *
            (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    tk_localcta::IType* sIn_ptr = reinterpret_cast<tk_localcta::IType*>(dshmem);
    transformer_engine::fp4e2m1x2* sOut_ptr =
        reinterpret_cast<transformer_engine::fp4e2m1x2*>(dshmem + in_bytes);
    transformer_engine::fp4e2m1x2* sOut_tr_ptr =
        reinterpret_cast<transformer_engine::fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = static_cast<int>(s_chunk_id % args.tiles_X);
        const int ctaid_Y = static_cast<int>(s_chunk_id / args.tiles_X);
        const int block_offset_Y = ctaid_Y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const bool output_dh1 = ctaid_X < split0_tiles;
        const int local_ctaid_X = output_dh1 ? ctaid_X : (ctaid_X - split0_tiles);
        const int input_block_offset_X = local_ctaid_X * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;

        float local_max = output_dh1
            ? load_silu_deriv_split2_chunk_explicit<true>(
                sIn_ptr, dh, h3, h1_raw, (int)rows, (int)split_cols, block_offset_Y, input_block_offset_X)
            : load_silu_deriv_split2_chunk_explicit<false>(
                sIn_ptr, dh, h3, h1_raw, (int)rows, (int)split_cols, block_offset_Y, input_block_offset_X);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
        }
        if (lane == 0) {
            warp_max[wid] = local_max;
        }
        __syncthreads();
        if (wid == 0) {
            float warp_val = (lane < tk_localcta::THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                warp_val = fmaxf(warp_val, __shfl_xor_sync(0xffffffff, warp_val, mask));
            }
            if (lane == 0) {
                cta_amax_shared = warp_val;
            }
        }
        __syncthreads();
        const float amax_val = cta_amax_shared;
        const float S_enc = tk_localcta::compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / tk_localcta::localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / tk_localcta::LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }
        __syncthreads();

        tk_localcta_fused_direct::quantize_store_prepared_chunk<RETURN_TRANSPOSE>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            S_enc, sg_val,
            block_offset_Y, block_offset_X,
            (int)rows, (int)total_cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_silu_deriv_split2_quant_prepared_impl(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_full,
    torch::Tensor row_sc_prepared_full,
    torch::Tensor col_fp4_full,
    torch::Tensor col_sc_prepared_full,
    torch::Tensor row_sg_full,
    torch::Tensor col_sg_full
) {
    using namespace tk_localcta;
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1_raw{}, tmap_h3{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h1_raw, h1_raw.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_out, row_fp4_full.data_ptr(), M, total_n,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, total_n, 4);
    if constexpr (RETURN_TRANSPOSE) {
        create_tma_2d(tmap_out_t, col_fp4_full.data_ptr(), total_n, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = total_n / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared_full.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = total_n / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col_prepared, col_sc_prepared_full.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(dh.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X0 = (H + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X = blocks_X0 * 2;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    constexpr int tile_bytes_aligned = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFF_DIM_Y * tk_localcta::BUFF_DIM_X * (int)sizeof(tk_localcta::IType),
        TMA_SHMEM_ALIGNMENT);
    const int dshmem =
        3 * tile_bytes_aligned + shmem_size<RETURN_TRANSPOSE>() + TMA_SHMEM_ALIGNMENT;
    auto& ci = get_localcta_cached_info();
    auto kernel = fused_localcta_silu_deriv_split2_prepared_kernel<RETURN_TRANSPOSE>;
    static int max_bps = -1;
    if (max_bps < 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_bps, kernel, THREADS, dshmem);
    }
    int num_persistent = max_bps * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };

    float* row_sg_ptr = row_sg_full.data_ptr<float>();
    float* col_sg_ptr = nullptr;
    if constexpr (RETURN_TRANSPOSE) {
        col_sg_ptr = col_sg_full.data_ptr<float>();
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_dh, tmap_h1_raw, tmap_h3,
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, total_n, args, blocks_X0);
}

static void launch_localcta_silu_deriv_split2_quant_prepared(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_full,
    torch::Tensor row_sc_prepared_full,
    torch::Tensor col_fp4_full,
    torch::Tensor col_sc_prepared_full,
    torch::Tensor row_sg_full,
    torch::Tensor col_sg_full
) {
    launch_localcta_silu_deriv_split2_quant_prepared_impl<true>(
        dh, h3, h1_raw,
        row_fp4_full, row_sc_prepared_full,
        col_fp4_full, col_sc_prepared_full,
        row_sg_full, col_sg_full);
}

static void launch_localcta_silu_deriv_split2_row_quant_prepared(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_full,
    torch::Tensor row_sc_prepared_full,
    torch::Tensor row_sg_full
) {
    launch_localcta_silu_deriv_split2_quant_prepared_impl<false>(
        dh, h3, h1_raw,
        row_fp4_full, row_sc_prepared_full,
        torch::Tensor(), torch::Tensor(),
        row_sg_full, torch::Tensor());
}

static void launch_localcta_silu_deriv_split2_row_bf16_quant_prepared_tuned(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1_out,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor row_sg_cat
) {
    using namespace tk_localcta;

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    alignas(64) CUtensorMap tmap_out{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{};
    create_tma_2d(tmap_out, row_fp4_cat.data_ptr(), M, total_n,
                  BUFF_DIM_Y, BUFF_DIM_X, total_n, 4);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = total_n / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared_cat.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto& work_counter = get_localcta_persistent_counter(dh.device());
    work_counter.zero_();
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X0 = (H + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X = blocks_X0 * 2;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    constexpr int KERNEL_THREADS = 128;
    constexpr int PIPE_DEPTH = 1;
    const int dshmem = prepared_1cta_tuned_shmem_size<PIPE_DEPTH, false>();
    auto kernel = fused_localcta_silu_deriv_split2_row_bf16_prepared_tuned<KERNEL_THREADS, true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    static int max_bps = -1;
    if (max_bps < 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, kernel, KERNEL_THREADS, dshmem);
    }

    auto& ci = get_localcta_cached_info();
    int num_persistent = max_bps * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };

    kernel<<<num_persistent, KERNEL_THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dh1_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr<at::BFloat16>()),
        tmap_out,
        tmap_sc_row_prepared,
        row_sg_cat.data_ptr<float>(),
        M,
        H,
        args,
        blocks_X0);
}

static void launch_localcta_silu_deriv_split2_quant_prepared_tuned_1cta(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    using namespace tk_localcta;

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_out, row_fp4_cat.data_ptr(), M, total_n,
                  BUFF_DIM_Y, BUFF_DIM_X, total_n, 4);
    create_tma_2d(tmap_out_t, col_fp4_cat.data_ptr(), total_n, M,
                  BUFF_DIM_X, BUFF_DIM_Y, M, 4);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = total_n / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared_cat.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    const int64_t ntm_c = total_n / 128;
    const int64_t ntk_c = M / 64;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_col_prepared, col_sc_prepared_cat.data_ptr(),
                  ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto& work_counter = get_localcta_persistent_counter(dh.device());
    work_counter.zero_();
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X0 = (H + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X = blocks_X0;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    constexpr int KERNEL_THREADS = 128;
    const int dshmem = prepared_split2_dual_1cta_tuned_shmem_size<1, true>();
    auto kernel = fused_localcta_silu_deriv_split2_prepared_tuned<KERNEL_THREADS, true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    static int max_bps = -1;
    if (max_bps < 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, kernel, KERNEL_THREADS, dshmem);
    }

    auto& ci = get_localcta_cached_info();
    int num_persistent = max_bps * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };

    kernel<<<num_persistent, KERNEL_THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr<at::BFloat16>()),
        tmap_out,
        tmap_out_t,
        tmap_sc_row_prepared,
        tmap_sc_col_prepared,
        row_sg_cat.data_ptr<float>(),
        col_sg_cat.data_ptr<float>(),
        M,
        H,
        args,
        blocks_X0);
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX,
          bool RETURN_TRANSPOSE = true, bool ENCODE_CENTRIC = true>
static void launch_localcta_silu_deriv_split2_quant_prepared_tuned_2cta_impl(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    using namespace tk_localcta;

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_out, row_fp4_cat.data_ptr(), M, total_n,
                  BUFF_DIM_Y, BUFF_DIM_X, total_n, 4);
    create_tma_2d(tmap_out_t, col_fp4_cat.data_ptr(), total_n, M,
                  BUFF_DIM_X, BUFF_DIM_Y, M, 4);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = total_n / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared_cat.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    const int64_t ntm_c = total_n / 128;
    const int64_t ntk_c = M / 64;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_col_prepared, col_sc_prepared_cat.data_ptr(),
                  ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X0 = (H + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int blocks_X = blocks_X0 * 2;
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_2cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_silu_deriv_split2_kernel_2cta_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, SHARED_AMAX, RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int dev = 0;
    cudaGetDevice(&dev);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_clusters = max_bps * num_sms / 2;
    if (num_clusters > total_macro_tiles) {
        num_clusters = total_macro_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(TOTAL_THREADS, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr<at::BFloat16>()),
        tmap_out,
        tmap_out_t,
        tmap_sc_row_prepared,
        tmap_sc_col_prepared,
        row_sg_cat.data_ptr<float>(),
        col_sg_cat.data_ptr<float>(),
        M,
        H,
        blocks_X,
        blocks_Y,
        total_macro_tiles,
        blocks_X0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchKernelEx failed for v4 fused split2 2CTA prepared quant: ",
                cudaGetErrorString(err));
}

template <bool RETURN_TRANSPOSE = true, bool ENCODE_CENTRIC = true>
static void launch_localcta_silu_deriv_split2_quant_prepared_tuned_2cta_dispatch(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    const auto cfg = get_localcta2_prepared_split2_tuning();
#define V4_2CTA_CASE(T, P, S) \
    if (cfg.threads == (T) && cfg.pipe_depth == (P) && cfg.shared_amax == (S)) { \
        launch_localcta_silu_deriv_split2_quant_prepared_tuned_2cta_impl< \
            (T), (P), (S), RETURN_TRANSPOSE, ENCODE_CENTRIC>( \
            dh, h3, h1_raw, \
            row_fp4_cat, row_sc_prepared_cat, \
            col_fp4_cat, col_sc_prepared_cat, \
            row_sg_cat, col_sg_cat); \
        return; \
    }
    V4_2CTA_CASE(160, 1, false);
    V4_2CTA_CASE(160, 2, false);
    V4_2CTA_CASE(160, 3, false);
    V4_2CTA_CASE(160, 4, false);
    V4_2CTA_CASE(192, 1, false);
    V4_2CTA_CASE(192, 2, false);
    V4_2CTA_CASE(192, 3, false);
    V4_2CTA_CASE(192, 4, false);
    V4_2CTA_CASE(256, 1, false);
    V4_2CTA_CASE(256, 2, false);
    V4_2CTA_CASE(256, 3, false);
    V4_2CTA_CASE(256, 4, false);
    V4_2CTA_CASE(384, 1, false);
    V4_2CTA_CASE(384, 2, false);
    V4_2CTA_CASE(384, 3, false);
    V4_2CTA_CASE(384, 4, false);
    V4_2CTA_CASE(512, 1, false);
    V4_2CTA_CASE(512, 2, false);
    V4_2CTA_CASE(512, 3, false);
    V4_2CTA_CASE(512, 4, false);
    V4_2CTA_CASE(160, 1, true);
    V4_2CTA_CASE(160, 2, true);
    V4_2CTA_CASE(160, 3, true);
    V4_2CTA_CASE(160, 4, true);
    V4_2CTA_CASE(192, 1, true);
    V4_2CTA_CASE(192, 2, true);
    V4_2CTA_CASE(192, 3, true);
    V4_2CTA_CASE(192, 4, true);
    V4_2CTA_CASE(256, 1, true);
    V4_2CTA_CASE(256, 2, true);
    V4_2CTA_CASE(256, 3, true);
    V4_2CTA_CASE(256, 4, true);
    V4_2CTA_CASE(384, 1, true);
    V4_2CTA_CASE(384, 2, true);
    V4_2CTA_CASE(384, 3, true);
    V4_2CTA_CASE(384, 4, true);
    V4_2CTA_CASE(512, 1, true);
    V4_2CTA_CASE(512, 2, true);
    V4_2CTA_CASE(512, 3, true);
    V4_2CTA_CASE(512, 4, true);
#undef V4_2CTA_CASE
    TORCH_CHECK(false, "Unsupported v4 fused split2 2CTA tuning config: threads=", cfg.threads,
                " pipe_depth=", cfg.pipe_depth, " shared_amax=", cfg.shared_amax);
}

static void launch_localcta_silu_deriv_split2_quant_prepared_tuned(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    launch_localcta_silu_deriv_split2_quant_prepared_tuned_2cta_dispatch(
        dh, h3, h1_raw,
        row_fp4_cat, row_sc_prepared_cat,
        col_fp4_cat, col_sc_prepared_cat,
        row_sg_cat, col_sg_cat);
}

template <bool DELAYED_SCALING = false>
static void launch_localcta_silu_deriv_split2_quant_raw_tuned_2cta(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor delayed_amax = torch::Tensor()
) {
    using namespace tk_localcta;

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);

    alignas(64) CUtensorMap tmap_out0{}, tmap_out0_t{}, tmap_sc_row0{}, tmap_sc_col0{};
    alignas(64) CUtensorMap tmap_out1{}, tmap_out1_t{}, tmap_sc_row1{}, tmap_sc_col1{};
    create_raw_output_tmaps_strided<true>(
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0,
        tmap_out0, tmap_out0_t, tmap_sc_row0, tmap_sc_col0);
    create_raw_output_tmaps_strided<true>(
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1,
        tmap_out1, tmap_out1_t, tmap_sc_row1, tmap_sc_col1);

    const int blocks_Y = static_cast<int>(M / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X0 = static_cast<int>(H / LocalCTAConfig::CHUNK_DIM_X);
    const int blocks_X = blocks_X0 * 2;
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return;
    }

    constexpr int TOTAL_THREADS = 160;
    constexpr int PIPE_DEPTH = 1;
    constexpr bool SHARED_AMAX = false;
    constexpr bool RETURN_TRANSPOSE = true;
    constexpr bool ENCODE_CENTRIC = true;
    const int dshmem = prepared_2cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_silu_deriv_split2_kernel_2cta_raw_tuned<
        TOTAL_THREADS, PIPE_DEPTH, SHARED_AMAX, RETURN_TRANSPOSE, ENCODE_CENTRIC, DELAYED_SCALING>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int dev = 0;
    cudaGetDevice(&dev);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_clusters = max_bps * num_sms / 2;
    if (num_clusters > total_macro_tiles) {
        num_clusters = total_macro_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(TOTAL_THREADS, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = at::cuda::getCurrentCUDAStream().stream();
    config.attrs = attrs;
    config.numAttrs = 2;

    float* amax_out0 = nullptr;
    float* amax_out1 = nullptr;
    if constexpr (DELAYED_SCALING) {
        TORCH_CHECK(delayed_amax.defined() && delayed_amax.is_cuda() &&
                    delayed_amax.scalar_type() == torch::kFloat32 &&
                    delayed_amax.numel() >= 2,
                    "delayed localCTA tuned SiLU-deriv requires a CUDA float32 amax buffer with at least two elements");
        amax_out0 = delayed_amax.data_ptr<float>();
        amax_out1 = delayed_amax.data_ptr<float>() + 1;
    }

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr<at::BFloat16>()),
        tmap_out0, tmap_out0_t, tmap_sc_row0, tmap_sc_col0,
        row_sg_chunk_0.data_ptr<float>(), col_sg_chunk_0.data_ptr<float>(),
        tmap_out1, tmap_out1_t, tmap_sc_row1, tmap_sc_col1,
        row_sg_chunk_1.data_ptr<float>(), col_sg_chunk_1.data_ptr<float>(),
        amax_out0,
        amax_out1,
        M,
        H,
        blocks_X,
        blocks_Y,
        total_macro_tiles,
        blocks_X0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchKernelEx failed for v4 fused split2 2CTA raw quant: ",
                cudaGetErrorString(err));
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant_2cta(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    unsigned int *work_counter,
    float *cluster_amax_scratch,
    int64_t M,
    int64_t K,
    bool write_raw_scales,
    bool write_prepared,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return;
    }
    auto &ci = get_localcta_cached_info();
    int num_clusters = ci.num_sms;
    if (num_clusters > total_macro_tiles) {
        num_clusters = total_macro_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    const int dshmem = shmem_size<RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_kernel_2cta<RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    LocalCTA2ClusterArgs args {
        .work_counter = work_counter,
        .tiles_X = blocks_X,
        .tiles_Y = blocks_Y,
        .total_macro_tiles = total_macro_tiles
    };

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(THREADS, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        cluster_amax_scratch,
        M, K, args, write_raw_scales, write_prepared);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchKernelEx failed for localCTA 2-CTA quant: ",
                cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
allocate_quant_outputs(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4)
                                    : torch::empty({0}, opts_fp4);
    auto col_sc = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8)
                                   : torch::empty({0}, opts_fp8);
    auto row_sg = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg = return_transpose ? torch::empty({K / 128, M / 128}, opts_f32) : row_sg;

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
allocate_quant_outputs_v3(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4)
                                    : torch::empty({0}, opts_fp4);
    auto col_sc = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8)
                                   : torch::empty({0}, opts_fp8);

    torch::Tensor row_sg;
    torch::Tensor col_sg;
    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        TORCH_CHECK(M % 256 == 0 && K % 256 == 0,
                    "localCTA v3 tilegrid256 contract requires M and K to be multiples of 256");
        row_sg = torch::empty({M / 256, K / 256}, opts_f32);
        col_sg = return_transpose ? torch::empty({K / 256, M / 256}, opts_f32)
                                  : torch::empty({0}, opts_f32);
    } else {
        row_sg = torch::empty({outer_sg_tiles_128(M), 1}, opts_f32);
        col_sg = return_transpose ? torch::empty({1, outer_sg_tiles_128(K)}, opts_f32)
                                  : torch::empty({0}, opts_f32);
    }
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

void quantize_into_outputs(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_sc_prepared,
    bool use_2cta
);

static std::tuple<torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_chunkgrid_internal(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    bool use_2cta = false
) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        torch::Tensor(), torch::Tensor(), use_2cta);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

void quantize_into_outputs(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_sc_prepared,
    bool use_2cta
);

static void finalize_quant_contract_v3(
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor row_sg,
    torch::Tensor col_sc,
    torch::Tensor col_sg_chunk,
    torch::Tensor col_sg
) {
    const int64_t M = row_sc.size(0) * 128;
    const int64_t K = row_sc.size(1) * 64;
    const int64_t row_chunks = M / 128;
    const int64_t sg_cols = K / 128;
    const int64_t row_sc_cols = K / 64;
    auto stream = at::cuda::getCurrentCUDAStream();

    const bool has_col = col_sc.defined() && col_sc.numel() > 0;

    if (use_localcta_v4_split_finalize_single()) {
        reduce_row_sg_tiles_kernel<256><<<static_cast<unsigned int>((row_chunks + 1) / 2), 256, 0, stream>>>(
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(row_chunks),
            static_cast<int>(sg_cols));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_tiles_kernel failed: ", cudaGetErrorString(err));
        }
        dim3 row_rescale_grid(static_cast<unsigned int>(row_chunks),
                              static_cast<unsigned int>(row_sc_cols));
        rescale_row_sc_kernel<256><<<row_rescale_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(row_chunks),
            static_cast<int>(row_sc_cols),
            static_cast<int>(sg_cols));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "rescale_row_sc_kernel failed: ", cudaGetErrorString(err));
        }

        if (!has_col) {
            return;
        }

        const int64_t col_chunks = K / 128;
        const int64_t col_sg_rows = M / 128;
        const int64_t col_sc_rows = M / 64;
        reduce_col_sg_tiles_kernel<256><<<static_cast<unsigned int>((col_chunks + 1) / 2), 256, 0, stream>>>(
            col_sg_chunk.data_ptr<float>(),
            col_sg.data_ptr<float>(),
            static_cast<int>(col_chunks),
            static_cast<int>(col_sg_rows));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_kernel failed: ", cudaGetErrorString(err));
        }
        dim3 col_rescale_grid(static_cast<unsigned int>(col_chunks),
                              static_cast<unsigned int>(col_sc_rows));
        rescale_col_sc_kernel<256><<<col_rescale_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()),
            col_sg_chunk.data_ptr<float>(),
            col_sg.data_ptr<float>(),
            static_cast<int>(col_chunks),
            static_cast<int>(col_sc_rows),
            static_cast<int>(col_sg_rows));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "rescale_col_sc_kernel failed: ", cudaGetErrorString(err));
        }
        return;
    }

    dim3 row_grid(static_cast<unsigned int>((row_chunks + 1) / 2),
                  static_cast<unsigned int>(row_sc_cols));
    finalize_row_sc_kernel<<<row_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
        row_sg_chunk.data_ptr<float>(),
        row_sg.data_ptr<float>(),
        static_cast<int>(row_chunks),
        static_cast<int>(row_sc_cols),
        static_cast<int>(sg_cols));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_row_sc_kernel failed: ", cudaGetErrorString(err));
    }

    if (!has_col) {
        return;
    }

    const int64_t col_chunks = K / 128;
    const int64_t col_sg_rows = M / 128;
    const int64_t col_sc_rows = M / 64;
    dim3 col_grid(static_cast<unsigned int>((col_chunks + 1) / 2),
                  static_cast<unsigned int>(col_sc_rows));
    finalize_col_sc_kernel<<<col_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()),
        col_sg_chunk.data_ptr<float>(),
        col_sg.data_ptr<float>(),
        static_cast<int>(col_chunks),
        static_cast<int>(col_sc_rows),
        static_cast<int>(col_sg_rows));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "finalize_col_sc_kernel failed: ", cudaGetErrorString(err));
}

static void rescale_quant_contract_v3_from_final_sg(
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor row_sg,
    torch::Tensor col_sc,
    torch::Tensor col_sg_chunk,
    torch::Tensor col_sg
) {
    const int64_t M = row_sc.size(0) * 128;
    const int64_t K = row_sc.size(1) * 64;
    const int64_t row_chunks = M / 128;
    const int64_t row_sc_cols = K / 64;
    const int64_t row_sg_cols = K / 128;
    const bool has_col = col_sc.defined() && col_sc.numel() > 0;

    int64_t col_chunks = 0;
    int64_t col_sc_rows = 0;
    int64_t col_sg_rows = 0;
    if (has_col) {
        col_chunks = K / 128;
        col_sc_rows = M / 64;
        col_sg_rows = M / 128;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch_vector = [&](auto cols_tag) {
        constexpr int cols_per_block = decltype(cols_tag)::value;
        const int64_t row_col_blocks =
            (row_sc_cols + cols_per_block - 1) / cols_per_block;
        const int64_t col_row_blocks = has_col
            ? (col_sc_rows + cols_per_block - 1) / cols_per_block
            : 0;
        const int64_t row_tasks = ((row_chunks + 1) / 2) * row_col_blocks;
        const int64_t col_tasks = has_col ? ((col_chunks + 1) / 2) * col_row_blocks : 0;
        rescale_row_col_sc_from_final_sg_vector_kernel<cols_per_block><<<
            static_cast<unsigned int>(row_tasks + col_tasks), 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(row_chunks),
            static_cast<int>(row_sc_cols),
            static_cast<int>(row_sg_cols),
            has_col ? reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()) : nullptr,
            has_col ? col_sg_chunk.data_ptr<float>() : nullptr,
            has_col ? col_sg.data_ptr<float>() : nullptr,
            static_cast<int>(col_chunks),
            static_cast<int>(col_sc_rows),
            static_cast<int>(col_sg_rows),
            has_col);
    };

    auto launch_warp = [&](auto cols_tag) {
        constexpr int cols_per_block = decltype(cols_tag)::value;
        const int64_t row_col_blocks =
            (row_sc_cols + cols_per_block - 1) / cols_per_block;
        const int64_t col_row_blocks = has_col
            ? (col_sc_rows + cols_per_block - 1) / cols_per_block
            : 0;
        const int64_t row_tasks = ((row_chunks + 1) / 2) * row_col_blocks;
        const int64_t col_tasks = has_col ? ((col_chunks + 1) / 2) * col_row_blocks : 0;
        rescale_row_col_sc_from_final_sg_warp_kernel<cols_per_block><<<
            static_cast<unsigned int>(row_tasks + col_tasks), 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(row_chunks),
            static_cast<int>(row_sc_cols),
            static_cast<int>(row_sg_cols),
            has_col ? reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()) : nullptr,
            has_col ? col_sg_chunk.data_ptr<float>() : nullptr,
            has_col ? col_sg.data_ptr<float>() : nullptr,
            static_cast<int>(col_chunks),
            static_cast<int>(col_sc_rows),
            static_cast<int>(col_sg_rows),
            has_col);
    };

    const int64_t vector_cols = localcta_env_int64(
        "USE_TK_LOCALCTA_V4_FINAL_SG_RESCALE_COLS_PER_BLOCK", 16);
    if (vector_cols > 0) {
        if (localcta_env_flag("USE_TK_LOCALCTA_V4_FINAL_SG_RESCALE_WARP", true)) {
            if (vector_cols <= 8) {
                launch_warp(std::integral_constant<int, 8>{});
            } else if (vector_cols <= 16) {
                launch_warp(std::integral_constant<int, 16>{});
            } else {
                launch_warp(std::integral_constant<int, 32>{});
            }
        } else if (vector_cols <= 1) {
            launch_vector(std::integral_constant<int, 1>{});
        } else if (vector_cols <= 2) {
            launch_vector(std::integral_constant<int, 2>{});
        } else if (vector_cols <= 4) {
            launch_vector(std::integral_constant<int, 4>{});
        } else if (vector_cols <= 8) {
            launch_vector(std::integral_constant<int, 8>{});
        } else {
            launch_vector(std::integral_constant<int, 16>{});
        }
    } else {
        const int64_t row_tasks = ((row_chunks + 1) / 2) * row_sc_cols;
        const int64_t col_tasks = has_col ? ((col_chunks + 1) / 2) * col_sc_rows : 0;
        rescale_row_col_sc_from_final_sg_kernel<256><<<
            static_cast<unsigned int>(row_tasks + col_tasks), 256, 0, stream>>>(
                reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
                row_sg_chunk.data_ptr<float>(),
                row_sg.data_ptr<float>(),
                static_cast<int>(row_chunks),
                static_cast<int>(row_sc_cols),
                static_cast<int>(row_sg_cols),
                has_col ? reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()) : nullptr,
                has_col ? col_sg_chunk.data_ptr<float>() : nullptr,
                has_col ? col_sg.data_ptr<float>() : nullptr,
                static_cast<int>(col_chunks),
                static_cast<int>(col_sc_rows),
                static_cast<int>(col_sg_rows),
                has_col);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rescale_row_col_sc_from_final_sg_kernel failed: ",
                cudaGetErrorString(err));
}

static void finalize_quant_contract_v3_reduced_warp(
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor row_sg,
    torch::Tensor col_sc,
    torch::Tensor col_sg_chunk,
    torch::Tensor col_sg
) {
    const int64_t M = row_sc.size(0) * 128;
    const int64_t K = row_sc.size(1) * 64;
    const int64_t row_chunks = M / 128;
    const int64_t row_sg_cols = K / 128;
    const bool has_col = col_sc.defined() && col_sc.numel() > 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    cudaError_t err;
    if (has_col && use_localcta_v4_nhsd_combined_sg_reduce()) {
        const int64_t col_chunks = K / 128;
        const int64_t col_sg_rows = M / 128;
        reduce_row_col_sg_tiles_kernel<256><<<
            static_cast<unsigned int>(
                (std::max(row_chunks, col_chunks) + 1) / 2),
            256, 0, stream>>>(
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(row_chunks),
            static_cast<int>(row_sg_cols),
            col_sg_chunk.data_ptr<float>(),
            col_sg.data_ptr<float>(),
            static_cast<int>(col_chunks),
            static_cast<int>(col_sg_rows));
        err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "NHSD combined row/col SG reduction failed: ",
                    cudaGetErrorString(err));
    } else {
        reduce_row_sg_tiles_kernel<256><<<
            static_cast<unsigned int>((row_chunks + 1) / 2), 256, 0, stream>>>(
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(row_chunks),
            static_cast<int>(row_sg_cols));
        err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "NHSD reduced-warp row SG reduction failed: ",
                    cudaGetErrorString(err));

        if (has_col) {
            const int64_t col_chunks = K / 128;
            const int64_t col_sg_rows = M / 128;
            reduce_col_sg_tiles_kernel<256><<<
                static_cast<unsigned int>((col_chunks + 1) / 2), 256, 0, stream>>>(
                col_sg_chunk.data_ptr<float>(),
                col_sg.data_ptr<float>(),
                static_cast<int>(col_chunks),
                static_cast<int>(col_sg_rows));
            err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess,
                        "NHSD reduced-warp col SG reduction failed: ",
                        cudaGetErrorString(err));
        }
    }

    rescale_quant_contract_v3_from_final_sg(
        row_sc, row_sg_chunk, row_sg,
        col_sc, col_sg_chunk, col_sg);
}

static void finalize_col_quant_contract_v3(
    torch::Tensor col_sc,
    torch::Tensor col_sg_chunk,
    torch::Tensor col_sg
) {
    if (!col_sc.defined() || col_sc.numel() == 0) {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t K = col_sc.size(0) * 128;
    const int64_t M = col_sc.size(1) * 64;
    const int64_t col_chunks = K / 128;
    const int64_t col_sg_rows = M / 128;
    const int64_t col_sc_rows = M / 64;

    dim3 col_grid(static_cast<unsigned int>((col_chunks + 1) / 2),
                  static_cast<unsigned int>(col_sc_rows));
    finalize_col_sc_kernel<<<col_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()),
        col_sg_chunk.data_ptr<float>(),
        col_sg.data_ptr<float>(),
        static_cast<int>(col_chunks),
        static_cast<int>(col_sc_rows),
        static_cast<int>(col_sg_rows));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_col_sc_kernel failed: ", cudaGetErrorString(err));
    }
}

static void finalize_row_quant_contract_v3(
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor row_sg
) {
    const int64_t M = row_sc.size(0) * 128;
    const int64_t K = row_sc.size(1) * 64;
    const int64_t row_chunks = M / 128;
    const int64_t sg_cols = K / 128;
    const int64_t row_sc_cols = K / 64;
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 row_grid(static_cast<unsigned int>((row_chunks + 1) / 2),
                  static_cast<unsigned int>(row_sc_cols));
    finalize_row_sc_kernel<<<row_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
        row_sg_chunk.data_ptr<float>(),
        row_sg.data_ptr<float>(),
        static_cast<int>(row_chunks),
        static_cast<int>(row_sc_cols),
        static_cast<int>(sg_cols));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_row_sc_kernel failed: ", cudaGetErrorString(err));
    }
}

static void finalize_row_quant_contract_v3_strided(
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor row_sg
) {
    const int64_t M = row_sc.size(0) * 128;
    const int64_t K = row_sc.size(1) * 64;
    const int64_t row_chunks = M / 128;
    const int64_t sg_cols = K / 128;
    const int64_t row_sc_cols = K / 64;
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 row_grid(static_cast<unsigned int>((row_chunks + 1) / 2),
                  static_cast<unsigned int>(row_sc_cols));
    finalize_row_sc_strided_kernel<<<row_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
        row_sg_chunk.data_ptr<float>(),
        row_sg.data_ptr<float>(),
        static_cast<int>(row_chunks),
        static_cast<int>(row_sc_cols),
        static_cast<int>(sg_cols),
        row_sc.stride(0),
        row_sc.stride(1),
        row_sg_chunk.stride(0),
        row_sg_chunk.stride(1));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_row_sc_strided_kernel failed: ", cudaGetErrorString(err));
    }
}

static void finalize_row_quant_contract_v3_strided_split2(
    torch::Tensor row_sc_0,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor row_sg_0,
    torch::Tensor row_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor row_sg_1
) {
    TORCH_CHECK(row_sc_0.size(0) == row_sc_1.size(0) &&
                    row_sg_chunk_0.size(0) == row_sg_chunk_1.size(0),
                "strided split2 row finalizer requires equal row dimensions");
    TORCH_CHECK(row_sg_0.numel() == row_sg_1.numel(),
                "strided split2 row finalizer requires equal outer-SG shapes");
    TORCH_CHECK(
        row_sc_0.scalar_type() == torch::kFloat8_e4m3fn &&
            row_sc_1.scalar_type() == torch::kFloat8_e4m3fn &&
            row_sc_0.dim() == 3 && row_sc_1.dim() == 3 &&
            row_sc_0.size(2) == 512 && row_sc_1.size(2) == 512 &&
            row_sc_0.stride(2) == 1 && row_sc_1.stride(2) == 1,
        "strided split2 FP8x4 rescaler requires E4M3 [rows, cols, 512] "
        "views with a contiguous innermost scale tile");
    TORCH_CHECK(
        row_sg_chunk_0.scalar_type() == torch::kFloat32 &&
            row_sg_chunk_1.scalar_type() == torch::kFloat32 &&
            row_sg_chunk_0.dim() == 2 && row_sg_chunk_1.dim() == 2,
        "strided split2 FP8x4 rescaler requires rank-2 FP32 chunk scales");
    TORCH_CHECK(
        row_sg_0.scalar_type() == torch::kFloat32 &&
            row_sg_1.scalar_type() == torch::kFloat32 &&
            row_sg_0.is_contiguous() && row_sg_1.is_contiguous(),
        "strided split2 FP8x4 rescaler requires contiguous FP32 outer scales");
    TORCH_CHECK(
        row_sc_0.size(1) == 2 * row_sg_chunk_0.size(1) &&
            row_sc_1.size(1) == 2 * row_sg_chunk_1.size(1),
        "strided split2 FP8x4 rescaler requires two scale columns per "
        "chunk-scale column in each arm");
    TORCH_CHECK(
        row_sc_0.stride(0) > 0 && row_sc_0.stride(1) > 0 &&
            row_sc_1.stride(0) > 0 && row_sc_1.stride(1) > 0 &&
            row_sc_0.stride(1) % 4 == 0 &&
            row_sc_1.stride(1) % 4 == 0 &&
            (reinterpret_cast<uintptr_t>(row_sc_0.data_ptr()) & 0x3u) == 0 &&
            (reinterpret_cast<uintptr_t>(row_sc_1.data_ptr()) & 0x3u) == 0,
        "strided split2 FP8x4 rescaler requires four-byte-aligned scale "
        "vectors and column strides");

    const int row_chunks = static_cast<int>(row_sc_0.size(0));
    const int sc_cols_0 = static_cast<int>(row_sc_0.size(1));
    const int sc_cols_1 = static_cast<int>(row_sc_1.size(1));
    const int sg_cols_0 = static_cast<int>(row_sg_chunk_0.size(1));
    const int sg_cols_1 = static_cast<int>(row_sg_chunk_1.size(1));
    const int outer_tiles = (row_chunks + 1) / 2;
    const int max_sc_cols = std::max(sc_cols_0, sc_cols_1);
    TORCH_CHECK(
        row_sg_0.numel() == outer_tiles && row_sg_1.numel() == outer_tiles,
        "strided split2 FP8x4 rescaler outer-scale shape does not match rows");
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 reduce_grid(static_cast<unsigned int>(outer_tiles), 2u);
    reduce_row_sg_tiles_strided_split2_kernel<<<
        reduce_grid, 256, 0, stream>>>(
        row_sg_chunk_0.data_ptr<float>(),
        row_sg_0.data_ptr<float>(),
        sg_cols_0,
        row_sg_chunk_0.stride(0),
        row_sg_chunk_0.stride(1),
        row_sg_chunk_1.data_ptr<float>(),
        row_sg_1.data_ptr<float>(),
        sg_cols_1,
        row_sg_chunk_1.stride(0),
        row_sg_chunk_1.stride(1),
        row_chunks);
    {
        const cudaError_t err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "reduce_row_sg_tiles_strided_split2_kernel failed: ",
            cudaGetErrorString(err));
    }

    constexpr int kSplit2RescaleColsPerBlock = 8;
    const int max_sc_col_blocks =
        (max_sc_cols + kSplit2RescaleColsPerBlock - 1) /
        kSplit2RescaleColsPerBlock;
    dim3 rescale_grid(
        static_cast<unsigned int>(outer_tiles),
        static_cast<unsigned int>(max_sc_col_blocks),
        2u);
    rescale_row_sc_strided_split2_fp8x4_warp_kernel<
        kSplit2RescaleColsPerBlock><<<
        rescale_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(row_sc_0.data_ptr()),
        row_sg_chunk_0.data_ptr<float>(),
        row_sg_0.data_ptr<float>(),
        sc_cols_0,
        sg_cols_0,
        row_sc_0.stride(0),
        row_sc_0.stride(1),
        row_sg_chunk_0.stride(0),
        row_sg_chunk_0.stride(1),
        reinterpret_cast<__nv_fp8_e4m3*>(row_sc_1.data_ptr()),
        row_sg_chunk_1.data_ptr<float>(),
        row_sg_1.data_ptr<float>(),
        sc_cols_1,
        sg_cols_1,
        row_sc_1.stride(0),
        row_sc_1.stride(1),
        row_sg_chunk_1.stride(0),
        row_sg_chunk_1.stride(1),
        row_chunks);
    {
        const cudaError_t err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "rescale_row_sc_strided_split2_fp8x4_warp_kernel failed: ",
            cudaGetErrorString(err));
    }
}

static void finalize_quant_contract_v3_split2(
    torch::Tensor row_sc_0,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sc_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sc_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor col_sg_1,
    bool shared_row_outer_sg = false
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int block = get_v3_split2_finalize_block_size();
    if (shared_row_outer_sg) {
        TORCH_CHECK(row_sc_0.size(0) == row_sc_1.size(0) &&
                    row_sg_chunk_0.size(0) == row_sg_chunk_1.size(0),
                    "shared split2 row outer SG requires equal row dimensions");
        TORCH_CHECK(row_sg_0.numel() == row_sg_1.numel(),
                    "shared split2 row outer SG requires equal output shapes");
        const int row_chunks = static_cast<int>(row_sc_0.size(0));
        const int sc_cols_0 = static_cast<int>(row_sc_0.size(1));
        const int sc_cols_1 = static_cast<int>(row_sc_1.size(1));
        const int sg_cols_0 = static_cast<int>(row_sg_chunk_0.size(1));
        const int sg_cols_1 = static_cast<int>(row_sg_chunk_1.size(1));
        dim3 row_grid(
            static_cast<unsigned int>((row_chunks + 1) / 2),
            static_cast<unsigned int>(sc_cols_0 + sc_cols_1));
        finalize_row_sc_split2_shared_outer_kernel<<<row_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_0.data_ptr()),
            row_sg_chunk_0.data_ptr<float>(),
            row_sg_0.data_ptr<float>(),
            sc_cols_0,
            sg_cols_0,
            row_sc_0.stride(0),
            row_sc_0.stride(1),
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_1.data_ptr()),
            row_sg_chunk_1.data_ptr<float>(),
            row_sg_1.data_ptr<float>(),
            sc_cols_1,
            sg_cols_1,
            row_sc_1.stride(0),
            row_sc_1.stride(1),
            row_chunks);
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess,
                        "finalize_row_sc_split2_shared_outer_kernel failed: ",
                        cudaGetErrorString(err));
        }
    } else {
        finalize_row_quant_contract_v3(row_sc_0, row_sg_chunk_0, row_sg_0);
        finalize_row_quant_contract_v3(row_sc_1, row_sg_chunk_1, row_sg_1);
    }

    const int sc_rows = static_cast<int>(col_sc_0.size(1));
    const int sg_rows = static_cast<int>(col_sg_chunk_0.size(1));
    const int k_chunks_0 = static_cast<int>(col_sc_0.size(0));
    const int k_chunks_1 = static_cast<int>(col_sc_1.size(0));
    const int max_col_tiles = std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2);
    const int max_k_chunks = std::max(k_chunks_0, k_chunks_1);

    dim3 col_reduce_grid(static_cast<unsigned int>(max_col_tiles), 2u);
    if (block == 64) {
        reduce_col_sg_tiles_split2_kernel<64><<<col_reduce_grid, 64, 0, stream>>>(
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sg_rows,
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    } else if (block == 128) {
        reduce_col_sg_tiles_split2_kernel<128><<<col_reduce_grid, 128, 0, stream>>>(
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sg_rows,
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    } else {
        reduce_col_sg_tiles_split2_kernel<<<col_reduce_grid, 256, 0, stream>>>(
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sg_rows,
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    }
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_split2_kernel failed: ", cudaGetErrorString(err));
    }

    dim3 col_rescale_grid(static_cast<unsigned int>(max_k_chunks),
                          static_cast<unsigned int>(sc_rows),
                          2u);
    if (block == 64) {
        rescale_col_sc_split2_kernel<64><<<col_rescale_grid, 64, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    } else if (block == 128) {
        rescale_col_sc_split2_kernel<128><<<col_rescale_grid, 128, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    } else {
        rescale_col_sc_split2_kernel<<<col_rescale_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    }
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "rescale_col_sc_split2_kernel failed: ", cudaGetErrorString(err));
    }
}

__global__ void fill_split2_final_sg_unit_kernel(
    float* __restrict__ row_sg_0,
    int64_t row_sg_0_numel,
    float* __restrict__ col_sg_0,
    int64_t col_sg_0_numel,
    float* __restrict__ row_sg_1,
    int64_t row_sg_1_numel,
    float* __restrict__ col_sg_1,
    int64_t col_sg_1_numel
) {
    const int64_t total = row_sg_0_numel + col_sg_0_numel + row_sg_1_numel + col_sg_1_numel;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t local = idx;
        if (local < row_sg_0_numel) {
            row_sg_0[local] = 1.0f;
            continue;
        }
        local -= row_sg_0_numel;
        if (local < col_sg_0_numel) {
            col_sg_0[local] = 1.0f;
            continue;
        }
        local -= col_sg_0_numel;
        if (local < row_sg_1_numel) {
            row_sg_1[local] = 1.0f;
            continue;
        }
        local -= row_sg_1_numel;
        col_sg_1[local] = 1.0f;
    }
}

static void fill_split2_final_sg_unit(
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1
) {
    const int64_t total = row_sg_0.numel() + col_sg_0.numel() + row_sg_1.numel() + col_sg_1.numel();
    if (total <= 0) {
        return;
    }
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    if (grid > 1024) {
        grid = 1024;
    }
    fill_split2_final_sg_unit_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        row_sg_0.data_ptr<float>(), row_sg_0.numel(),
        col_sg_0.data_ptr<float>(), col_sg_0.numel(),
        row_sg_1.data_ptr<float>(), row_sg_1.numel(),
        col_sg_1.data_ptr<float>(), col_sg_1.numel());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fill_split2_final_sg_unit_kernel failed: ", cudaGetErrorString(err));
}

static void finalize_col_quant_contract_v3_split2(
    torch::Tensor col_sc_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor col_sg_0,
    torch::Tensor col_sc_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor col_sg_1
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int block = get_v3_split2_finalize_block_size();
    const int sc_rows_per_block = get_v3_col_finalize_rows_per_block();
    const int sg_rows = static_cast<int>(col_sg_chunk_0.size(1));
    const int sc_rows = static_cast<int>(col_sc_0.size(1));
    const int k_chunks_0 = static_cast<int>(col_sc_0.size(0));
    const int k_chunks_1 = static_cast<int>(col_sc_1.size(0));
    const int max_col_tiles = std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2);
    const int sc_row_blocks = (sc_rows + sc_rows_per_block - 1) / sc_rows_per_block;

    dim3 col_grid(static_cast<unsigned int>(max_col_tiles),
                  static_cast<unsigned int>(sc_row_blocks),
                  2u);
    if (block == 64) {
        finalize_col_sc_split2_chunked_kernel<64><<<col_grid, 64, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            sc_rows_per_block,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    } else if (block == 128) {
        finalize_col_sc_split2_chunked_kernel<128><<<col_grid, 128, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            sc_rows_per_block,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    } else {
        finalize_col_sc_split2_chunked_kernel<<<col_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            sc_rows_per_block,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1);
    }
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_col_sc_split2_chunked_kernel failed: ", cudaGetErrorString(err));
    }
}

static void finalize_quant_contract_v3_split3(
    torch::Tensor row_sc_0,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sc_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sc_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sc_2,
    torch::Tensor row_sg_chunk_2,
    torch::Tensor row_sg_2,
    torch::Tensor col_sc_2,
    torch::Tensor col_sg_chunk_2,
    torch::Tensor col_sg_2,
    bool shared_row_outer_sg = false
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t M = row_sc_0.size(0) * 128;
    const int64_t row_tiles = outer_sg_tiles_128(M);
    const int sc_rows = static_cast<int>(col_sc_0.size(1));
    const int sg_rows = static_cast<int>(col_sg_chunk_0.size(1));
    const int k_chunks_0 = static_cast<int>(col_sc_0.size(0));
    const int k_chunks_1 = static_cast<int>(col_sc_1.size(0));
    const int k_chunks_2 = static_cast<int>(col_sc_2.size(0));
    const int max_col_tiles = std::max(std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2), (k_chunks_2 + 1) / 2);
    const int max_k_chunks = std::max(std::max(k_chunks_0, k_chunks_1), k_chunks_2);

    dim3 row_grid(static_cast<unsigned int>(row_tiles), 3u);
    auto* row_sc_0_ptr = reinterpret_cast<__nv_fp8_e4m3*>(row_sc_0.data_ptr());
    auto* row_sc_1_ptr = reinterpret_cast<__nv_fp8_e4m3*>(row_sc_1.data_ptr());
    auto* row_sc_2_ptr = reinterpret_cast<__nv_fp8_e4m3*>(row_sc_2.data_ptr());
    auto* col_sc_0_ptr = reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr());
    auto* col_sc_1_ptr = reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr());
    auto* col_sc_2_ptr = reinterpret_cast<__nv_fp8_e4m3*>(col_sc_2.data_ptr());
    bool combined_rescale_launched = false;
    if (shared_row_outer_sg) {
        TORCH_CHECK(row_sc_0.size(0) == row_sc_1.size(0) &&
                    row_sc_0.size(0) == row_sc_2.size(0) &&
                    row_sg_chunk_0.size(0) == row_sg_chunk_1.size(0) &&
                    row_sg_chunk_0.size(0) == row_sg_chunk_2.size(0),
                    "shared split3 row outer SG requires equal row dimensions");
        TORCH_CHECK(row_sg_0.numel() == row_sg_1.numel() &&
                    row_sg_0.numel() == row_sg_2.numel(),
                    "shared split3 row outer SG requires equal output shapes");
        const int row_chunks = static_cast<int>(row_sc_0.size(0));
        const int sc_cols_0 = static_cast<int>(row_sc_0.size(1));
        const int sc_cols_1 = static_cast<int>(row_sc_1.size(1));
        const int sc_cols_2 = static_cast<int>(row_sc_2.size(1));
        const int sg_cols_0 = static_cast<int>(row_sg_chunk_0.size(1));
        const int sg_cols_1 = static_cast<int>(row_sg_chunk_1.size(1));
        const int sg_cols_2 = static_cast<int>(row_sg_chunk_2.size(1));
        dim3 shared_row_grid(
            static_cast<unsigned int>((row_chunks + 1) / 2),
            static_cast<unsigned int>(sc_cols_0 + sc_cols_1 + sc_cols_2));
        finalize_row_sc_split3_shared_outer_kernel<<<shared_row_grid, 256, 0, stream>>>(
            row_sc_0_ptr,
            row_sg_chunk_0.data_ptr<float>(),
            row_sg_0.data_ptr<float>(),
            sc_cols_0,
            sg_cols_0,
            row_sc_0.stride(0),
            row_sc_0.stride(1),
            row_sc_1_ptr,
            row_sg_chunk_1.data_ptr<float>(),
            sc_cols_1,
            sg_cols_1,
            row_sc_1.stride(0),
            row_sc_1.stride(1),
            row_sc_2_ptr,
            row_sg_chunk_2.data_ptr<float>(),
            sc_cols_2,
            sg_cols_2,
            row_sc_2.stride(0),
            row_sc_2.stride(1),
            row_chunks);
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess,
                        "finalize_row_sc_split3_shared_outer_kernel failed: ",
                        cudaGetErrorString(err));
        }
    } else if (use_v4_split3_row_split_rescale()) {
        reduce_row_sg_split3_kernel<<<row_grid, 256, 0, stream>>>(
            row_sg_chunk_0.data_ptr<float>(),
            row_sg_0.data_ptr<float>(),
            static_cast<int>(row_sg_chunk_0.size(1)),
            row_sg_chunk_1.data_ptr<float>(),
            row_sg_1.data_ptr<float>(),
            static_cast<int>(row_sg_chunk_1.size(1)),
            row_sg_chunk_2.data_ptr<float>(),
            row_sg_2.data_ptr<float>(),
            static_cast<int>(row_sg_chunk_2.size(1)));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_split3_kernel failed: ", cudaGetErrorString(err));
        }

        const int row_rescale_cols_per_block = get_v4_split3_row_rescale_cols_per_block();
        const int max_row_sc_cols = std::max(std::max(static_cast<int>(row_sc_0.size(1)),
                                                      static_cast<int>(row_sc_1.size(1))),
                                             static_cast<int>(row_sc_2.size(1)));
        const int row_col_blocks = (max_row_sc_cols + row_rescale_cols_per_block - 1) / row_rescale_cols_per_block;
        dim3 row_rescale_grid(static_cast<unsigned int>(row_tiles),
                              static_cast<unsigned int>(row_col_blocks),
                              3u);
        const int col_rescale_rows_per_block_for_combined = get_v4_split3_col_rescale_rows_per_block();
        const int col_row_blocks_for_combined =
            (sc_rows + col_rescale_rows_per_block_for_combined - 1) / col_rescale_rows_per_block_for_combined;
        if (
            use_v4_split3_combined_rescale()
            && row_rescale_cols_per_block == 2
            && col_rescale_rows_per_block_for_combined == 8
        ) {
            dim3 col_reduce_grid(static_cast<unsigned int>(max_col_tiles), 3u);
            reduce_col_sg_tiles_split3_kernel<<<col_reduce_grid, 256, 0, stream>>>(
                col_sg_chunk_0.data_ptr<float>(),
                col_sg_0.data_ptr<float>(),
                k_chunks_0,
                sg_rows,
                col_sg_chunk_1.data_ptr<float>(),
                col_sg_1.data_ptr<float>(),
                k_chunks_1,
                col_sg_chunk_2.data_ptr<float>(),
                col_sg_2.data_ptr<float>(),
                k_chunks_2);
            {
                cudaError_t err = cudaGetLastError();
                TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_split3_kernel failed: ", cudaGetErrorString(err));
            }

            const int row_tasks = static_cast<int>(row_tiles) * row_col_blocks * 3;
            const int col_tasks = max_k_chunks * col_row_blocks_for_combined * 3;
            rescale_row_col_sc_split3_kernel<2, 8><<<row_tasks + col_tasks, 256, 0, stream>>>(
                row_tasks,
                static_cast<int>(row_tiles),
                row_col_blocks,
                col_row_blocks_for_combined,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(0)), static_cast<int>(row_sc_0.size(1)),
                static_cast<int>(row_sg_chunk_0.size(1)), row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(0)), static_cast<int>(row_sc_1.size(1)),
                static_cast<int>(row_sg_chunk_1.size(1)), row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(0)), static_cast<int>(row_sc_2.size(1)),
                static_cast<int>(row_sg_chunk_2.size(1)), row_sc_2.stride(0), row_sc_2.stride(1),
                col_sc_0_ptr, col_sg_chunk_0.data_ptr<float>(), col_sg_0.data_ptr<float>(),
                k_chunks_0, sc_rows, sg_rows,
                col_sc_1_ptr, col_sg_chunk_1.data_ptr<float>(), col_sg_1.data_ptr<float>(),
                k_chunks_1,
                col_sc_2_ptr, col_sg_chunk_2.data_ptr<float>(), col_sg_2.data_ptr<float>(),
                k_chunks_2);
            {
                cudaError_t err = cudaGetLastError();
                TORCH_CHECK(err == cudaSuccess, "rescale_row_col_sc_split3_kernel failed: ", cudaGetErrorString(err));
            }
            combined_rescale_launched = true;
        }
        if (!combined_rescale_launched) {
        if (row_rescale_cols_per_block <= 1) {
            launch_rescale_row_sc_split3_cols<1>(
                row_rescale_grid, stream,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(0)), static_cast<int>(row_sc_0.size(1)),
                static_cast<int>(row_sg_chunk_0.size(1)), row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(0)), static_cast<int>(row_sc_1.size(1)),
                static_cast<int>(row_sg_chunk_1.size(1)), row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(0)), static_cast<int>(row_sc_2.size(1)),
                static_cast<int>(row_sg_chunk_2.size(1)), row_sc_2.stride(0), row_sc_2.stride(1));
        } else if (row_rescale_cols_per_block <= 2) {
            launch_rescale_row_sc_split3_cols<2>(
                row_rescale_grid, stream,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(0)), static_cast<int>(row_sc_0.size(1)),
                static_cast<int>(row_sg_chunk_0.size(1)), row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(0)), static_cast<int>(row_sc_1.size(1)),
                static_cast<int>(row_sg_chunk_1.size(1)), row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(0)), static_cast<int>(row_sc_2.size(1)),
                static_cast<int>(row_sg_chunk_2.size(1)), row_sc_2.stride(0), row_sc_2.stride(1));
        } else if (row_rescale_cols_per_block <= 4) {
            launch_rescale_row_sc_split3_cols<4>(
                row_rescale_grid, stream,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(0)), static_cast<int>(row_sc_0.size(1)),
                static_cast<int>(row_sg_chunk_0.size(1)), row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(0)), static_cast<int>(row_sc_1.size(1)),
                static_cast<int>(row_sg_chunk_1.size(1)), row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(0)), static_cast<int>(row_sc_2.size(1)),
                static_cast<int>(row_sg_chunk_2.size(1)), row_sc_2.stride(0), row_sc_2.stride(1));
        } else {
            launch_rescale_row_sc_split3_cols<8>(
                row_rescale_grid, stream,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(0)), static_cast<int>(row_sc_0.size(1)),
                static_cast<int>(row_sg_chunk_0.size(1)), row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(0)), static_cast<int>(row_sc_1.size(1)),
                static_cast<int>(row_sg_chunk_1.size(1)), row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(0)), static_cast<int>(row_sc_2.size(1)),
                static_cast<int>(row_sg_chunk_2.size(1)), row_sc_2.stride(0), row_sc_2.stride(1));
        }
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "rescale_row_sc_split3_cols_kernel failed: ", cudaGetErrorString(err));
        }
        }
    } else {
        const int row_finalize_block = get_v4_split3_row_finalize_block_size();
        if (row_finalize_block <= 128) {
            launch_finalize_row_sc_split3<128>(
                row_grid, stream,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(1)), static_cast<int>(row_sg_chunk_0.size(1)),
                row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(1)), static_cast<int>(row_sg_chunk_1.size(1)),
                row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(1)), static_cast<int>(row_sg_chunk_2.size(1)),
                row_sc_2.stride(0), row_sc_2.stride(1));
        } else if (row_finalize_block <= 256) {
            launch_finalize_row_sc_split3<256>(
                row_grid, stream,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(1)), static_cast<int>(row_sg_chunk_0.size(1)),
                row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(1)), static_cast<int>(row_sg_chunk_1.size(1)),
                row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(1)), static_cast<int>(row_sg_chunk_2.size(1)),
                row_sc_2.stride(0), row_sc_2.stride(1));
        } else {
            launch_finalize_row_sc_split3<512>(
                row_grid, stream,
                row_sc_0_ptr, row_sg_chunk_0.data_ptr<float>(), row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sc_0.size(1)), static_cast<int>(row_sg_chunk_0.size(1)),
                row_sc_0.stride(0), row_sc_0.stride(1),
                row_sc_1_ptr, row_sg_chunk_1.data_ptr<float>(), row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sc_1.size(1)), static_cast<int>(row_sg_chunk_1.size(1)),
                row_sc_1.stride(0), row_sc_1.stride(1),
                row_sc_2_ptr, row_sg_chunk_2.data_ptr<float>(), row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sc_2.size(1)), static_cast<int>(row_sg_chunk_2.size(1)),
                row_sc_2.stride(0), row_sc_2.stride(1));
        }
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "finalize_row_sc_split3_kernel failed: ", cudaGetErrorString(err));
        }
    }

    if (!combined_rescale_launched) {
    dim3 col_reduce_grid(static_cast<unsigned int>(max_col_tiles), 3u);
    reduce_col_sg_tiles_split3_kernel<<<col_reduce_grid, 256, 0, stream>>>(
        col_sg_chunk_0.data_ptr<float>(),
        col_sg_0.data_ptr<float>(),
        k_chunks_0,
        sg_rows,
        col_sg_chunk_1.data_ptr<float>(),
        col_sg_1.data_ptr<float>(),
        k_chunks_1,
        col_sg_chunk_2.data_ptr<float>(),
        col_sg_2.data_ptr<float>(),
        k_chunks_2);
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_split3_kernel failed: ", cudaGetErrorString(err));
    }

    const int col_rescale_rows_per_block = get_v4_split3_col_rescale_rows_per_block();
    if (col_rescale_rows_per_block <= 1) {
        dim3 col_rescale_grid(static_cast<unsigned int>(max_k_chunks),
                              static_cast<unsigned int>(sc_rows),
                              3u);
        rescale_col_sc_split3_kernel<<<col_rescale_grid, 256, 0, stream>>>(
            col_sc_0_ptr,
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            col_sc_1_ptr,
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1,
            col_sc_2_ptr,
            col_sg_chunk_2.data_ptr<float>(),
            col_sg_2.data_ptr<float>(),
            k_chunks_2);
    } else {
        const int row_blocks = (sc_rows + col_rescale_rows_per_block - 1) / col_rescale_rows_per_block;
        dim3 col_rescale_grid(static_cast<unsigned int>(max_k_chunks),
                              static_cast<unsigned int>(row_blocks),
                              3u);
        if (col_rescale_rows_per_block <= 2) {
            launch_rescale_col_sc_split3_rows<2>(
                col_rescale_grid, stream, col_sc_0_ptr, col_sg_chunk_0.data_ptr<float>(),
                col_sg_0.data_ptr<float>(), k_chunks_0, sc_rows, sg_rows,
                col_sc_1_ptr, col_sg_chunk_1.data_ptr<float>(), col_sg_1.data_ptr<float>(),
                k_chunks_1, col_sc_2_ptr, col_sg_chunk_2.data_ptr<float>(),
                col_sg_2.data_ptr<float>(), k_chunks_2);
        } else if (col_rescale_rows_per_block <= 4) {
            launch_rescale_col_sc_split3_rows<4>(
                col_rescale_grid, stream, col_sc_0_ptr, col_sg_chunk_0.data_ptr<float>(),
                col_sg_0.data_ptr<float>(), k_chunks_0, sc_rows, sg_rows,
                col_sc_1_ptr, col_sg_chunk_1.data_ptr<float>(), col_sg_1.data_ptr<float>(),
                k_chunks_1, col_sc_2_ptr, col_sg_chunk_2.data_ptr<float>(),
                col_sg_2.data_ptr<float>(), k_chunks_2);
        } else if (col_rescale_rows_per_block <= 8) {
            launch_rescale_col_sc_split3_rows<8>(
                col_rescale_grid, stream, col_sc_0_ptr, col_sg_chunk_0.data_ptr<float>(),
                col_sg_0.data_ptr<float>(), k_chunks_0, sc_rows, sg_rows,
                col_sc_1_ptr, col_sg_chunk_1.data_ptr<float>(), col_sg_1.data_ptr<float>(),
                k_chunks_1, col_sc_2_ptr, col_sg_chunk_2.data_ptr<float>(),
                col_sg_2.data_ptr<float>(), k_chunks_2);
        } else {
            launch_rescale_col_sc_split3_rows<16>(
                col_rescale_grid, stream, col_sc_0_ptr, col_sg_chunk_0.data_ptr<float>(),
                col_sg_0.data_ptr<float>(), k_chunks_0, sc_rows, sg_rows,
                col_sc_1_ptr, col_sg_chunk_1.data_ptr<float>(), col_sg_1.data_ptr<float>(),
                k_chunks_1, col_sc_2_ptr, col_sg_chunk_2.data_ptr<float>(),
                col_sg_2.data_ptr<float>(), k_chunks_2);
        }
    }
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "rescale_col_sc_split3_kernel failed: ", cudaGetErrorString(err));
    }
    }
}

static void finalize_col_quant_contract_v3_split3(
    torch::Tensor col_sc_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor col_sg_0,
    torch::Tensor col_sc_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor col_sg_1,
    torch::Tensor col_sc_2,
    torch::Tensor col_sg_chunk_2,
    torch::Tensor col_sg_2
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int block = get_v3_split2_finalize_block_size();
    const int sc_rows = static_cast<int>(col_sc_0.size(1));
    const int sg_rows = static_cast<int>(col_sg_chunk_0.size(1));
    const int sc_rows_per_block = get_v3_col_finalize_rows_per_block();
    const int k_chunks_0 = static_cast<int>(col_sc_0.size(0));
    const int k_chunks_1 = static_cast<int>(col_sc_1.size(0));
    const int k_chunks_2 = static_cast<int>(col_sc_2.size(0));
    const int max_col_tiles = std::max(std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2), (k_chunks_2 + 1) / 2);
    const int sc_row_blocks = (sc_rows + sc_rows_per_block - 1) / sc_rows_per_block;

    dim3 col_grid(static_cast<unsigned int>(max_col_tiles),
                  static_cast<unsigned int>(sc_row_blocks),
                  3u);
    if (block == 64) {
        finalize_col_sc_split3_chunked_kernel<64><<<col_grid, 64, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            sc_rows_per_block,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_2.data_ptr()),
            col_sg_chunk_2.data_ptr<float>(),
            col_sg_2.data_ptr<float>(),
            k_chunks_2);
    } else if (block == 128) {
        finalize_col_sc_split3_chunked_kernel<128><<<col_grid, 128, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            sc_rows_per_block,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_2.data_ptr()),
            col_sg_chunk_2.data_ptr<float>(),
            col_sg_2.data_ptr<float>(),
            k_chunks_2);
    } else {
        finalize_col_sc_split3_chunked_kernel<<<col_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
            col_sg_chunk_0.data_ptr<float>(),
            col_sg_0.data_ptr<float>(),
            k_chunks_0,
            sc_rows,
            sg_rows,
            sc_rows_per_block,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
            col_sg_chunk_1.data_ptr<float>(),
            col_sg_1.data_ptr<float>(),
            k_chunks_1,
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc_2.data_ptr()),
            col_sg_chunk_2.data_ptr<float>(),
            col_sg_2.data_ptr<float>(),
            k_chunks_2);
    }
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_col_sc_split3_chunked_kernel failed: ", cudaGetErrorString(err));
    }
}

static void finalize_col_quant_contract_v3_split3_fused(
    torch::Tensor col_sc_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor col_sg_0,
    torch::Tensor col_sc_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor col_sg_1,
    torch::Tensor col_sc_2,
    torch::Tensor col_sg_chunk_2,
    torch::Tensor col_sg_2
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int sc_rows = static_cast<int>(col_sc_0.size(1));
    const int sg_rows = static_cast<int>(col_sg_chunk_0.size(1));
    const int k_chunks_0 = static_cast<int>(col_sc_0.size(0));
    const int k_chunks_1 = static_cast<int>(col_sc_1.size(0));
    const int k_chunks_2 = static_cast<int>(col_sc_2.size(0));
    const int max_col_tiles = std::max(std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2), (k_chunks_2 + 1) / 2);

    dim3 col_grid(static_cast<unsigned int>(max_col_tiles), 3u);
    finalize_col_sc_split3_kernel<<<col_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(col_sc_0.data_ptr()),
        col_sg_chunk_0.data_ptr<float>(),
        col_sg_0.data_ptr<float>(),
        k_chunks_0,
        sc_rows,
        sg_rows,
        reinterpret_cast<__nv_fp8_e4m3*>(col_sc_1.data_ptr()),
        col_sg_chunk_1.data_ptr<float>(),
        col_sg_1.data_ptr<float>(),
        k_chunks_1,
        reinterpret_cast<__nv_fp8_e4m3*>(col_sc_2.data_ptr()),
        col_sg_chunk_2.data_ptr<float>(),
        col_sg_2.data_ptr<float>(),
        k_chunks_2);
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_col_sc_split3_kernel failed: ", cudaGetErrorString(err));
    }
}

__global__ void reduce_sg_chunks_to_tilegrid256_kernel(
    const float* __restrict__ row_sg_chunk,
    float* __restrict__ row_sg_tilegrid,
    float* __restrict__ col_sg_tilegrid,
    int macro_tiles_y,
    int macro_tiles_x,
    int chunk_stride
) {
    const int tile_x = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int tile_y = static_cast<int>(blockIdx.y);
    if (tile_y >= macro_tiles_y || tile_x >= macro_tiles_x) {
        return;
    }

    const int chunk_y = tile_y * 2;
    const int chunk_x = tile_x * 2;
    const float s00 = row_sg_chunk[chunk_y * chunk_stride + chunk_x];
    const float s01 = row_sg_chunk[chunk_y * chunk_stride + chunk_x + 1];
    const float s10 = row_sg_chunk[(chunk_y + 1) * chunk_stride + chunk_x];
    const float s11 = row_sg_chunk[(chunk_y + 1) * chunk_stride + chunk_x + 1];
    const float sg = fmaxf(fmaxf(s00, s01), fmaxf(s10, s11));

    row_sg_tilegrid[tile_y * macro_tiles_x + tile_x] = sg;
    if (col_sg_tilegrid != nullptr) {
        col_sg_tilegrid[tile_x * macro_tiles_y + tile_y] = sg;
    }
}

static void finalize_quant_contract_v3_tilegrid256(
    torch::Tensor row_sg_chunk,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    TORCH_CHECK(row_sg_chunk.dim() == 2 && row_sg.dim() == 2, "tilegrid256 SG tensors must be 2D");
    TORCH_CHECK(row_sg_chunk.size(0) % 2 == 0 && row_sg_chunk.size(1) % 2 == 0,
                "tilegrid256 requires even chunk-grid dimensions");
    const int macro_tiles_y = static_cast<int>(row_sg_chunk.size(0) / 2);
    const int macro_tiles_x = static_cast<int>(row_sg_chunk.size(1) / 2);
    TORCH_CHECK(row_sg.size(0) == macro_tiles_y && row_sg.size(1) == macro_tiles_x,
                "row_sg tilegrid256 shape mismatch");
    if (col_sg.defined() && col_sg.numel() > 0) {
        TORCH_CHECK(col_sg.size(0) == macro_tiles_x && col_sg.size(1) == macro_tiles_y,
                    "col_sg tilegrid256 shape mismatch");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(128);
    dim3 grid(static_cast<unsigned int>((macro_tiles_x + block.x - 1) / block.x),
              static_cast<unsigned int>(macro_tiles_y));
    reduce_sg_chunks_to_tilegrid256_kernel<<<grid, block, 0, stream>>>(
        row_sg_chunk.data_ptr<float>(),
        row_sg.data_ptr<float>(),
        (col_sg.defined() && col_sg.numel() > 0) ? col_sg.data_ptr<float>() : nullptr,
        macro_tiles_y,
        macro_tiles_x,
        static_cast<int>(row_sg_chunk.size(1)));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "reduce_sg_chunks_to_tilegrid256_kernel failed: ",
                    cudaGetErrorString(err));
    }
}

static void fold_quant_scales_v3_tilegrid256(
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sc,
    torch::Tensor col_sg_chunk
) {
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t M = row_sc.size(0) * 128;
    const int64_t K = row_sc.size(1) * 64;
    const int64_t row_chunks = M / 128;
    const int64_t sg_cols = K / 128;
    const int64_t row_sc_cols = K / 64;
    dim3 row_grid(static_cast<unsigned int>(row_chunks),
                  static_cast<unsigned int>(row_sc_cols));
    fold_row_sc_chunk_sg_strided_kernel<<<row_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
        row_sg_chunk.data_ptr<float>(),
        static_cast<int>(row_chunks),
        static_cast<int>(row_sc_cols),
        static_cast<int>(sg_cols),
        row_sc.stride(0),
        row_sc.stride(1));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "fold_row_sc_chunk_sg_strided_kernel failed: ",
                    cudaGetErrorString(err));
    }

    if (!col_sc.defined() || col_sc.numel() == 0) {
        return;
    }

    const int64_t col_K = col_sc.size(0) * 128;
    const int64_t col_M = col_sc.size(1) * 64;
    const int64_t k_chunks = col_K / 128;
    const int64_t sg_rows = col_M / 128;
    const int64_t sc_rows = col_M / 64;
    dim3 col_grid(static_cast<unsigned int>(k_chunks),
                  static_cast<unsigned int>(sc_rows));
    fold_col_sc_chunk_sg_kernel<<<col_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()),
        col_sg_chunk.data_ptr<float>(),
        static_cast<int>(k_chunks),
        static_cast<int>(sc_rows),
        static_cast<int>(sg_rows));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "fold_col_sc_chunk_sg_kernel failed: ",
                    cudaGetErrorString(err));
    }
}

static void copy_cat_dim0_contiguous(
    const std::vector<torch::Tensor>& srcs,
    torch::Tensor dst
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    size_t byte_offset = 0;
    char* dst_ptr = static_cast<char*>(dst.data_ptr());
    for (const auto& src : srcs) {
        TORCH_CHECK(src.is_contiguous(), "copy_cat_dim0_contiguous expects contiguous inputs");
        cudaError_t err = cudaMemcpyAsync(
            dst_ptr + byte_offset,
            src.data_ptr(),
            src.nbytes(),
            cudaMemcpyDeviceToDevice,
            stream);
        TORCH_CHECK(err == cudaSuccess, "copy_cat_dim0_contiguous failed: ", cudaGetErrorString(err));
        byte_offset += src.nbytes();
    }
}

static void copy_cat_dim1_contiguous(
    const std::vector<torch::Tensor>& srcs,
    torch::Tensor dst
) {
    TORCH_CHECK(!srcs.empty(), "copy_cat_dim1_contiguous expects at least one source tensor");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t rows = dst.size(0);
    const size_t element_size = dst.element_size();
    int64_t inner = 1;
    for (int64_t d = 2; d < dst.dim(); ++d) inner *= dst.size(d);
    const size_t dst_pitch = static_cast<size_t>(dst.size(1) * inner) * element_size;
    char* dst_ptr = static_cast<char*>(dst.data_ptr());
    int64_t col_offset = 0;
    for (const auto& src : srcs) {
        TORCH_CHECK(src.is_contiguous(), "copy_cat_dim1_contiguous expects contiguous inputs");
        TORCH_CHECK(src.dim() == dst.dim(), "copy_cat_dim1_contiguous rank mismatch");
        const size_t src_pitch = static_cast<size_t>(src.size(1) * inner) * element_size;
        const size_t width = src_pitch;
        cudaError_t err = cudaMemcpy2DAsync(
            dst_ptr + static_cast<size_t>(col_offset * inner) * element_size,
            dst_pitch,
            src.data_ptr(),
            src_pitch,
            width,
            static_cast<size_t>(rows),
            cudaMemcpyDeviceToDevice,
            stream);
        TORCH_CHECK(err == cudaSuccess, "copy_cat_dim1_contiguous failed: ", cudaGetErrorString(err));
        col_offset += src.size(1);
    }
}

void quantize_into_outputs_v3(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    torch::Tensor row_sc_prepared = torch::Tensor(),
    torch::Tensor col_sc_prepared = torch::Tensor(),
    bool use_2cta = false
) {
    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        TORCH_CHECK(input.size(0) % 256 == 0 && input.size(1) % 256 == 0,
                    "localCTA v3 tilegrid256 contract requires M and K to be multiples of 256");
        auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
        auto row_sg_chunk = torch::empty({input.size(0) / 128, input.size(1) / 128}, opts_f32);
        auto col_sg_chunk = return_transpose
            ? torch::empty({input.size(1) / 128, input.size(0) / 128}, opts_f32)
            : torch::empty({0}, opts_f32);

        quantize_into_outputs(
            input, return_transpose, encode_centric,
            row_fp4, row_sc, col_fp4, col_sc,
            row_sg_chunk, col_sg_chunk,
            row_sc_prepared, col_sc_prepared, use_2cta);
        fold_quant_scales_v3_tilegrid256(row_sc, row_sg_chunk, col_sc, col_sg_chunk);
        finalize_quant_contract_v3_tilegrid256(row_sg_chunk, row_sg, col_sg);
        return;
    }

    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_sg_chunk = torch::empty({input.size(0) / 128, input.size(1) / 128}, opts_f32);
    auto col_sg_chunk = return_transpose
        ? torch::empty({input.size(1) / 128, input.size(0) / 128}, opts_f32)
        : torch::empty({0}, opts_f32);

    quantize_into_outputs(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc,
        row_sg_chunk, col_sg_chunk,
        row_sc_prepared, col_sc_prepared, use_2cta);
    finalize_quant_contract_v3(row_sc, row_sg_chunk, row_sg, col_sc, col_sg_chunk, col_sg);
}

namespace tk_localcta_wo_nhsd {

using namespace tk_localcta;
using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;

__device__ __forceinline__ float load_nhsd_as_wo_chunk(
    IType* __restrict__ sIn_ptr,
    const __nv_bfloat16* __restrict__ input,
    int B,
    int H,
    int S,
    int D,
    int block_offset_Y,
    int block_offset_X
) {
    float local_max = 0.0f;
    constexpr int CHUNK_ELEMS = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;
    const int M = B * S;
    const int K = H * D;

    for (int idx = threadIdx.x; idx < CHUNK_ELEMS; idx += THREADS) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx - row * LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        __nv_bfloat16 value = __float2bfloat16_rn(0.0f);
        if (global_row < M && global_col < K) {
            const int b = global_row / S;
            const int s = global_row - b * S;
            const int h = global_col / D;
            const int d = global_col - h * D;
            const int64_t offset =
                (((int64_t)b * H + h) * S + s) * D + d;
            value = input[offset];
            local_max = fmaxf(local_max, fabsf(__bfloat162float(value)));
        }
        tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col, value);
    }
    __syncthreads();
    return local_max;
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
__device__ __forceinline__ void quantize_store_raw_chunk(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr,
    nvfp4_scale_t* sSFcolwise_ptr,
    const CUtensorMap& tensor_map_output,
    const CUtensorMap& tensor_map_output_t,
    const CUtensorMap& tmap_scale_row,
    const CUtensorMap& tmap_scale_col,
    float S_enc,
    int block_offset_Y,
    int block_offset_X,
    int rows,
    int cols,
    int ctaid_X,
    int ctaid_Y
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_rows = rows - block_offset_Y;
    const int chunk_cols = cols - block_offset_X;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;

    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    int buff_out = 0;
    int buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM_Y;
        const int stage_offset_X = stage_X * TILE_DIM_X;

        if (t >= BUFFS_NUM_OUT) {
            if (leading) {
                ptx::cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        rowwise_scaling<ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sSFrowwise_ptr,
            S_enc, stage_Y, stage_X, t, buff_out);
        if constexpr (RETURN_TRANSPOSE) {
            colwise_scaling<ENCODE_CENTRIC>(
                sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                S_enc, stage_Y, stage_X, t, buff_out_tr);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));

            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
        buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
    }

    if (leading) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    {
        const int cnt = min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM);
        swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tma_store_scales_2x512(tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
        }
    }

    if constexpr (RETURN_TRANSPOSE) {
        const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
        swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tma_store_scales_2x512(tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
        }
    }

    if (leading) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
}

template <bool ENCODE_CENTRIC>
__global__ void __launch_bounds__(THREADS)
localcta_quantize_nhsd_wo_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    int B,
    int H,
    int S,
    int D,
    LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes =
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    const int rows = B * S;
    const int cols = H * D;
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);
    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    int mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = static_cast<int>(s_chunk_id % args.tiles_X);
        const int ctaid_Y = static_cast<int>(s_chunk_id / args.tiles_X);
        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

        float cta_max = 0.0f;
        const int b = block_offset_Y / S;
        const int s_base = block_offset_Y - b * S;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int global_k = block_offset_X + stage_X * TILE_DIM_X;
            const int h = global_k / D;
            const int d = global_k - h * D;
            const int input_row = (b * H + h) * S + s_base + stage_Y * TILE_DIM_Y;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[t]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    d,
                    input_row,
                    &in_mbar[t]);
            }
        }

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
        }

        const float amax_val = tk_localcta_fused_direct::block_reduce_max(cta_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            const int tiles_Y = rows / LocalCTAConfig::CHUNK_DIM_Y;
            col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
        }
        __syncthreads();

        quantize_store_raw_chunk<true, ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row, tmap_scale_col,
            S_enc,
            block_offset_Y, block_offset_X,
            rows, cols,
            ctaid_X, ctaid_Y);
        mbar_phase ^= 1;
    }
#else
    NVTE_DEVICE_ERROR("localCTA WO NHSD quantization requires SM 10.0+.");
#endif
}

}  // namespace tk_localcta_wo_nhsd

template <typename KernelFn>
static int persistent_grid_for_kernel(KernelFn kernel, int threads, int dshmem, int total_tiles);

static void launch_localcta_quantize_nhsd_wo_raw(
    torch::Tensor input,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk
) {
    const int64_t B = input.size(0);
    const int64_t H = input.size(1);
    const int64_t S = input.size(2);
    const int64_t D = input.size(3);
    const int64_t M = B * S;
    const int64_t K = H * D;
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(K / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_in, input.data_ptr(), B * H * S, D,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, D, 16);
    create_raw_output_tmaps_strided<true>(
        row_fp4, row_sc, col_fp4, col_sc,
        tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);

    const int dshmem =
        tk_localcta_fused_direct::direct_fused_single_shmem_size<true>();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto& work_counter = get_localcta_persistent_counter(input.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };

    if (encode_centric) {
        auto kernel = tk_localcta_wo_nhsd::localcta_quantize_nhsd_wo_kernel<true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);
        kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
            tmap_in,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
            row_sg_chunk.data_ptr<float>(),
            col_sg_chunk.data_ptr<float>(),
            (int)B, (int)H, (int)S, (int)D, args);
    } else {
        auto kernel = tk_localcta_wo_nhsd::localcta_quantize_nhsd_wo_kernel<false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);
        kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
            tmap_in,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
            row_sg_chunk.data_ptr<float>(),
            col_sg_chunk.data_ptr<float>(),
            (int)B, (int)H, (int)S, (int)D, args);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_nhsd_wo_for_gemm failed: ",
                cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
allocate_quant_outputs_fast(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs(M, K, return_transpose, device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto row_sc_prepared = torch::empty_like(row_sc, opts_fp8);
    auto col_sc_prepared = return_transpose ? torch::empty_like(col_sc, opts_fp8)
                                            : torch::empty({0}, opts_fp8);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
allocate_quant_outputs_prepared(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc_prepared = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4)
                                    : torch::empty({0}, opts_fp4);
    auto col_sc_prepared = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8)
                                            : torch::empty({0}, opts_fp8);
    auto row_sg = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg = return_transpose ? torch::empty({K / 128, M / 128}, opts_f32) : row_sg;

    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

template <bool RETURN_TRANSPOSE>
static void create_prepared_output_tmaps(
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    CUtensorMap& tmap_out,
    CUtensorMap& tmap_out_t,
    CUtensorMap& tmap_sc_row_prepared,
    CUtensorMap& tmap_sc_col_prepared
) {
    const int64_t M = row_fp4.size(0);
    const int64_t K = row_fp4.size(1) * 2;

    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 4);

    if constexpr (RETURN_TRANSPOSE) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = K / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col_prepared, col_sc_prepared.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
}

template <bool RETURN_TRANSPOSE>
static void create_prepared_output_tmaps_strided(
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    CUtensorMap& tmap_out,
    CUtensorMap& tmap_out_t,
    CUtensorMap& tmap_sc_row_prepared,
    CUtensorMap& tmap_sc_col_prepared
) {
    const int64_t M = row_fp4.size(0);
    const int64_t K = row_fp4.size(1) * 2;
    const int64_t row_fp4_stride = row_fp4.stride(0) * 2;

    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, row_fp4_stride, 4);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t row_sc_stride_bf16 = row_sc_prepared.stride(0) / 2;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, row_sc_stride_bf16, 16);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t col_fp4_stride = col_fp4.stride(0) * 2;
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, col_fp4_stride, 4);

        const int64_t ntm_c = K / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        const int64_t col_sc_stride_bf16 = col_sc_prepared.stride(0) / 2;
        create_tma_2d(tmap_sc_col_prepared, col_sc_prepared.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, col_sc_stride_bf16, 16);
    }
}

template <typename KernelFn>
static int persistent_grid_for_kernel(KernelFn kernel, int threads, int dshmem, int total_tiles) {
    int dev = 0;
    cudaGetDevice(&dev);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, kernel, threads, dshmem);
    int num_persistent = max_bps * num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }
    return num_persistent;
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_persistent_silu_split_prepared(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_h1_raw{}, tmap_h3{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_h1_raw, h1_raw.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    constexpr int threads = tk_localcta_persistent_silu::PRODUCER_CONSUMER_THREADS;
    const int dshmem = tk_localcta_persistent_silu::persistent_localcta_silu_quant_smem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_persistent_silu::localcta_tma_silu_quantize_kernel<
        threads, RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(h1_raw.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    const int grid = persistent_grid_for_kernel(kernel, threads, dshmem, total_tiles);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernel<<<grid, threads, dshmem, stream>>>(
        tmap_h1_raw, tmap_h3, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
        M, H, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_persistent_silu_split_raw(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_h1_raw{}, tmap_h3{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_h1_raw, h1_raw.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_raw_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4, row_sc, col_fp4, col_sc,
        tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);

    constexpr int threads = tk_localcta_persistent_silu::PRODUCER_CONSUMER_THREADS;
    const int dshmem = tk_localcta_persistent_silu::persistent_localcta_silu_quant_smem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_persistent_silu::localcta_tma_silu_quantize_kernel<
        threads, RETURN_TRANSPOSE, true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto& work_counter = get_localcta_persistent_counter(h1_raw.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    const int grid = persistent_grid_for_kernel(kernel, threads, dshmem, total_tiles);
    kernel<<<grid, threads, dshmem, stream>>>(
        tmap_h1_raw, tmap_h3, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        row_sg_chunk.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_chunk.data_ptr<float>() : row_sg_chunk.data_ptr<float>(),
        M, H, args);
}

template <
    bool RETURN_TRANSPOSE,
    bool PREFINALIZED_OUTER_SG = false,
    bool ATOMIC_FINAL_OUTER_SG = false,
    bool PAIRED_FIXED_SIGN_COL_RHT = false>
static void launch_localcta_tma_silu_split_raw(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk,
    torch::Tensor row_sg_final = torch::Tensor(),
    torch::Tensor col_sg_final = torch::Tensor()
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_h1_raw{}, tmap_h3{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_h1_raw, h1_raw.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_raw_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4, row_sc, col_fp4, col_sc,
        tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);

    constexpr int threads = tk_localcta::SILU_RAW_THREADS;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto& work_counter = get_localcta_persistent_counter(h1_raw.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);
    if constexpr (ATOMIC_FINAL_OUTER_SG) {
        TORCH_CHECK(row_sg_final.defined() && row_sg_final.numel() > 0,
                    "SiLU final-SG producer requires row_sg output");
        TORCH_CHECK(!RETURN_TRANSPOSE || (col_sg_final.defined() && col_sg_final.numel() > 0),
                    "SiLU final-SG producer requires col_sg output when return_transpose=True");
        TORCH_CHECK(!PREFINALIZED_OUTER_SG,
                    "SiLU final-SG producer is not valid with prefinalized outer SG input");
        cudaMemsetAsync(row_sg_final.data_ptr<float>(), 0,
                        row_sg_final.numel() * sizeof(float), stream);
        if constexpr (RETURN_TRANSPOSE) {
            cudaMemsetAsync(col_sg_final.data_ptr<float>(), 0,
                            col_sg_final.numel() * sizeof(float), stream);
        }
    }

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    auto launch_kernel = [&](auto h3_ring_tag, auto parallel_row_col_tag) {
        constexpr int h3_ring_slots = decltype(h3_ring_tag)::value;
        constexpr bool parallel_row_col = decltype(parallel_row_col_tag)::value;
        const int dshmem = h3_ring_slots == 1
            ? tk_localcta::silu_raw_h3_ring_shmem_size<RETURN_TRANSPOSE, 1>()
            : tk_localcta::silu_raw_shmem_size<RETURN_TRANSPOSE>();
        auto kernel = tk_localcta::fused_localcta_silu_quantize_raw_kernel<
            RETURN_TRANSPOSE, true, h3_ring_slots, PREFINALIZED_OUTER_SG,
            ATOMIC_FINAL_OUTER_SG, parallel_row_col,
            PAIRED_FIXED_SIGN_COL_RHT, PAIRED_FIXED_SIGN_COL_RHT,
            !PAIRED_FIXED_SIGN_COL_RHT>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        const int grid = persistent_grid_for_kernel(kernel, threads, dshmem, total_tiles);
        kernel<<<grid, threads, dshmem, stream>>>(
            tmap_h1_raw, tmap_h3, tmap_out, tmap_out_t,
            tmap_sc_row, tmap_sc_col,
            row_sg_chunk.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunk.data_ptr<float>() : row_sg_chunk.data_ptr<float>(),
            ATOMIC_FINAL_OUTER_SG ? row_sg_final.data_ptr<float>() : nullptr,
            (ATOMIC_FINAL_OUTER_SG && RETURN_TRANSPOSE) ? col_sg_final.data_ptr<float>() : nullptr,
            M, H, use_localcta_v4_silu_fast_divide(), args);
    };

    const bool parallel_row_col =
        RETURN_TRANSPOSE && use_localcta_v4_silu_parallel_row_col();
    if (localcta_v4_silu_h3_ring_slots() == 1) {
        if (parallel_row_col) {
            launch_kernel(std::integral_constant<int, 1>{}, std::true_type{});
        } else {
            launch_kernel(std::integral_constant<int, 1>{}, std::false_type{});
        }
    } else if (parallel_row_col) {
        launch_kernel(std::integral_constant<int, 0>{}, std::true_type{});
    } else {
        launch_kernel(std::integral_constant<int, 0>{}, std::false_type{});
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_direct_silu_split_prepared(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    const int dshmem = tk_localcta_fused_direct::direct_fused_single_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_fused_direct::localcta_silu_quantize_split_direct_kernel<RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);
    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(h1_raw.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
        M, H, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_direct_silu_split_raw(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_chunk
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    create_raw_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4, row_sc, col_fp4, col_sc,
        tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);

    const int dshmem = tk_localcta_fused_direct::direct_fused_single_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_fused_direct::localcta_silu_quantize_split_direct_raw_kernel<RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto& work_counter = get_localcta_persistent_counter(h1_raw.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        row_sg_chunk.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_chunk.data_ptr<float>() : row_sg_chunk.data_ptr<float>(),
        M, H, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_direct_sqrelu_prepared(
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool encode_centric,
    bool data_sr = false,
    bool scale_sr = false,
    bool row_rht = false,
    bool col_rht = false,
    bool random_sign = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    const int dshmem = tk_localcta_fused_direct::direct_fused_single_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = encode_centric
        ? tk_localcta_fused_direct::localcta_sqrelu_quantize_direct_kernel<RETURN_TRANSPOSE, true>
        : tk_localcta_fused_direct::localcta_sqrelu_quantize_direct_kernel<RETURN_TRANSPOSE, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);

    auto& work_counter = get_localcta_persistent_counter(h1_raw.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
        M, H, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_direct_sqrelu_deriv_prepared(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool encode_centric,
    bool data_sr = false,
    bool scale_sr = false,
    bool row_rht = false,
    bool col_rht = false,
    bool random_sign = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    const int dshmem = tk_localcta_fused_direct::direct_fused_single_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = encode_centric
        ? tk_localcta_fused_direct::localcta_sqrelu_deriv_quantize_direct_kernel<RETURN_TRANSPOSE, true>
        : tk_localcta_fused_direct::localcta_sqrelu_deriv_quantize_direct_kernel<RETURN_TRANSPOSE, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);

    auto& work_counter = get_localcta_persistent_counter(dh.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
        M, H, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_tma_quant_row_prepared_col_outer(
    torch::Tensor input,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_raw,
    torch::Tensor row_sg_chunks,
    torch::Tensor col_sg_chunks,
    bool encode_centric
) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(K / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_input{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{};
    alignas(64) CUtensorMap empty_sc_row_raw{}, empty_sc_col_prepared{}, tmap_sc_col_raw{};
    create_tma_2d(tmap_input, input.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 4);

    if constexpr (RETURN_TRANSPOSE) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_prepared, row_sc_prepared.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = K / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col_raw, col_sc_raw.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (encode_centric) {
        launch_localcta_quant_prepared_tuned_dispatch<RETURN_TRANSPOSE, true, false, true>(
            tmap_input, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, empty_sc_row_raw, empty_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, K, stream);
    } else {
        launch_localcta_quant_prepared_tuned_dispatch<RETURN_TRANSPOSE, false, false, true>(
            tmap_input, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, empty_sc_row_raw, empty_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, K, stream);
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_tma_quant_raw_outer(
    torch::Tensor input,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_raw,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_raw,
    torch::Tensor row_sg_chunks,
    torch::Tensor col_sg_chunks,
    bool encode_centric
) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(K / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_input{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_raw{};
    alignas(64) CUtensorMap empty_sc_row_prepared{}, empty_sc_col_prepared{}, tmap_sc_col_raw{};
    create_tma_2d(tmap_input, input.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 4);
    if constexpr (RETURN_TRANSPOSE) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_raw, row_sc_raw.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = K / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col_raw, col_sc_raw.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (encode_centric) {
        launch_localcta_quant_prepared_tuned_dispatch<RETURN_TRANSPOSE, true, true, true>(
            tmap_input, tmap_out, tmap_out_t,
            empty_sc_row_prepared, tmap_sc_row_raw, empty_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, K, stream);
    } else {
        launch_localcta_quant_prepared_tuned_dispatch<RETURN_TRANSPOSE, false, true, true>(
            tmap_input, tmap_out, tmap_out_t,
            empty_sc_row_prepared, tmap_sc_row_raw, empty_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, K, stream);
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_tma_sqrelu_prepared(
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool encode_centric,
    bool data_sr = false,
    bool scale_sr = false,
    bool row_rht = false,
    bool col_rht = false,
    bool random_sign = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_input{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    alignas(64) CUtensorMap empty_sc_col_raw{};
    create_tma_2d(tmap_input, h1_raw.data_ptr(), M, H,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, H, 16);
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (encode_centric) {
        launch_localcta_sqrelu_prepared_tuned_dispatch<RETURN_TRANSPOSE, true>(
            tmap_input, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, tmap_sc_col_prepared, empty_sc_col_raw,
            row_sg.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    } else {
        launch_localcta_sqrelu_prepared_tuned_dispatch<RETURN_TRANSPOSE, false>(
            tmap_input, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, tmap_sc_col_prepared, empty_sc_col_raw,
            row_sg.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_tma_sqrelu_row_prepared_col_outer(
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor col_sc_raw,
    torch::Tensor row_sg_chunks,
    torch::Tensor col_sg_chunks,
    bool encode_centric,
    bool data_sr = false,
    bool scale_sr = false,
    bool row_rht = false,
    bool col_rht = false,
    bool random_sign = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_input{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    alignas(64) CUtensorMap tmap_sc_col_raw{};
    create_tma_2d(tmap_input, h1_raw.data_ptr(), M, H,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, H, 16);
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = H / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col_raw, col_sc_raw.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (encode_centric) {
        launch_localcta_sqrelu_prepared_tuned_dispatch<RETURN_TRANSPOSE, true, true>(
            tmap_input, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    } else {
        launch_localcta_sqrelu_prepared_tuned_dispatch<RETURN_TRANSPOSE, false, true>(
            tmap_input, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, tmap_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_tma_sqrelu_deriv_prepared(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool encode_centric,
    bool data_sr = false,
    bool scale_sr = false,
    bool row_rht = false,
    bool col_rht = false,
    bool random_sign = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    alignas(64) CUtensorMap empty_sc_row_raw{}, empty_sc_col_raw{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h1, h1_raw.data_ptr(), M, H,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, H, 16);
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (encode_centric) {
        launch_localcta_sqrelu_deriv_prepared_tuned_dispatch<RETURN_TRANSPOSE, true>(
            tmap_dh, tmap_h1, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, empty_sc_row_raw, tmap_sc_col_prepared, empty_sc_col_raw,
            row_sg.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    } else {
        launch_localcta_sqrelu_deriv_prepared_tuned_dispatch<RETURN_TRANSPOSE, false>(
            tmap_dh, tmap_h1, tmap_out, tmap_out_t,
            tmap_sc_row_prepared, empty_sc_row_raw, tmap_sc_col_prepared, empty_sc_col_raw,
            row_sg.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_tma_sqrelu_deriv_raw_outer(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_raw,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_raw,
    torch::Tensor row_sg_chunks,
    torch::Tensor col_sg_chunks,
    bool encode_centric,
    bool data_sr = false,
    bool scale_sr = false,
    bool row_rht = false,
    bool col_rht = false,
    bool random_sign = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1{};
    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_raw{}, tmap_sc_col_raw{};
    alignas(64) CUtensorMap empty_sc_row_prepared{}, empty_sc_col_prepared{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h1, h1_raw.data_ptr(), M, H,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, H,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, H, 4);
    if constexpr (RETURN_TRANSPOSE) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), H, M,
                      tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = H / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row_raw, row_sc_raw.data_ptr(),
                  ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    if constexpr (RETURN_TRANSPOSE) {
        const int64_t ntm_c = H / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col_raw, col_sc_raw.data_ptr(),
                      ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (encode_centric) {
        launch_localcta_sqrelu_deriv_prepared_tuned_dispatch<RETURN_TRANSPOSE, true, true, true>(
            tmap_dh, tmap_h1, tmap_out, tmap_out_t,
            empty_sc_row_prepared, tmap_sc_row_raw, empty_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    } else {
        launch_localcta_sqrelu_deriv_prepared_tuned_dispatch<RETURN_TRANSPOSE, false, true, true>(
            tmap_dh, tmap_h1, tmap_out, tmap_out_t,
            empty_sc_row_prepared, tmap_sc_row_raw, empty_sc_col_prepared, tmap_sc_col_raw,
            row_sg_chunks.data_ptr<float>(),
            RETURN_TRANSPOSE ? col_sg_chunks.data_ptr<float>() : row_sg_chunks.data_ptr<float>(),
            M, H,
            data_sr, scale_sr, row_rht, col_rht && RETURN_TRANSPOSE, random_sign,
            rng_seed, rng_subsequence_base, stream);
    }
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_direct_norm_prepared(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    bool with_silu,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(K / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    const int dshmem = tk_localcta_fused_direct::direct_fused_single_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_fused_direct::localcta_norm_quantize_direct_kernel<RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        inv_rms.data_ptr<float>(),
        with_silu,
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg.data_ptr<float>() : row_sg.data_ptr<float>(),
        M, K, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_persistent_silu_deriv_split_prepared(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_prepared_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_prepared_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_prepared_2,
    torch::Tensor col_fp4_2,
    torch::Tensor col_sc_prepared_2,
    torch::Tensor row_sg_2,
    torch::Tensor col_sg_2
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1_raw{}, tmap_h3{};
    alignas(64) CUtensorMap tmap_out1{}, tmap_out1_t{}, tmap_sc_row_prepared1{}, tmap_sc_col_prepared1{};
    alignas(64) CUtensorMap tmap_out2{}, tmap_out2_t{}, tmap_sc_row_prepared2{}, tmap_sc_col_prepared2{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h1_raw, h1_raw.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_prepared_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_1, row_sc_prepared_1, col_fp4_1, col_sc_prepared_1,
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1);
    create_prepared_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_2, row_sc_prepared_2, col_fp4_2, col_sc_prepared_2,
        tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2);

    constexpr int threads = tk_localcta_persistent_silu_deriv::PRODUCER_CONSUMER_THREADS;
    const int dshmem =
        tk_localcta_persistent_silu_deriv::persistent_localcta_silu_deriv_quant_smem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_persistent_silu_deriv::localcta_tma_silu_deriv_quantize_kernel<
        threads, RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    auto& work_counter = get_localcta_persistent_counter(dh.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);
    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    const int grid = persistent_grid_for_kernel(kernel, threads, dshmem, total_tiles);
    kernel<<<grid, threads, dshmem, stream>>>(
        tmap_dh, tmap_h1_raw, tmap_h3,
        tmap_out1, tmap_out2,
        tmap_out1_t, tmap_out2_t,
        tmap_sc_row_prepared1, tmap_sc_row_prepared2,
        tmap_sc_col_prepared1, tmap_sc_col_prepared2,
        row_sg_1.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_1.data_ptr<float>() : row_sg_1.data_ptr<float>(),
        row_sg_2.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_2.data_ptr<float>() : row_sg_2.data_ptr<float>(),
        M, H, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_direct_silu_deriv_split_prepared(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_prepared_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_prepared_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_prepared_2,
    torch::Tensor col_fp4_2,
    torch::Tensor col_sc_prepared_2,
    torch::Tensor row_sg_2,
    torch::Tensor col_sg_2,
    bool data_sr = false,
    bool scale_sr = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out1{}, tmap_out1_t{}, tmap_sc_row_prepared1{}, tmap_sc_col_prepared1{};
    alignas(64) CUtensorMap tmap_out2{}, tmap_out2_t{}, tmap_sc_row_prepared2{}, tmap_sc_col_prepared2{};
    create_prepared_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_1, row_sc_prepared_1, col_fp4_1, col_sc_prepared_1,
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1);
    create_prepared_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_2, row_sc_prepared_2, col_fp4_2, col_sc_prepared_2,
        tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto& work_counter = get_localcta_persistent_counter(dh.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    const bool fast_data_sr = data_sr && use_localcta_v4_fast_data_sr();

    const int dshmem = tk_localcta_fused_direct::direct_fused_dual_shmem_size<RETURN_TRANSPOSE>();

#define LAUNCH_DIRECT_SILU_DERIV_SPLIT(DATA, SCALE, FAST)                              \
    do {                                                                               \
        auto kernel = tk_localcta_fused_direct::localcta_silu_deriv_quantize_split_direct_kernel< \
            RETURN_TRANSPOSE, DATA, FAST, SCALE>;                                      \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem); \
        const int grid = localcta_v4_direct_grid_override(                             \
            persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles), \
            total_tiles);                                                              \
        kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(                        \
            reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),                     \
            reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),                     \
            reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),                 \
            tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1,      \
            row_sg_1.data_ptr<float>(),                                                \
            RETURN_TRANSPOSE ? col_sg_1.data_ptr<float>() : row_sg_1.data_ptr<float>(), \
            tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2,      \
            row_sg_2.data_ptr<float>(),                                                \
            RETURN_TRANSPOSE ? col_sg_2.data_ptr<float>() : row_sg_2.data_ptr<float>(), \
            M, H, args, rng_seed, rng_subsequence_base);                               \
    } while (0)

    if (data_sr && scale_sr) {
        if (fast_data_sr) {
            LAUNCH_DIRECT_SILU_DERIV_SPLIT(true, true, true);
        } else {
            LAUNCH_DIRECT_SILU_DERIV_SPLIT(true, true, false);
        }
    } else if (data_sr) {
        if (fast_data_sr) {
            LAUNCH_DIRECT_SILU_DERIV_SPLIT(true, false, true);
        } else {
            LAUNCH_DIRECT_SILU_DERIV_SPLIT(true, false, false);
        }
    } else if (scale_sr) {
        LAUNCH_DIRECT_SILU_DERIV_SPLIT(false, true, false);
    } else {
        LAUNCH_DIRECT_SILU_DERIV_SPLIT(false, false, false);
    }

#undef LAUNCH_DIRECT_SILU_DERIV_SPLIT
}

template <bool RETURN_TRANSPOSE, bool DELAYED_SCALING = false>
static void launch_localcta_direct_silu_deriv_split_raw(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_2,
    torch::Tensor col_fp4_2,
    torch::Tensor col_sc_2,
    torch::Tensor row_sg_chunk_2,
    torch::Tensor col_sg_chunk_2,
    torch::Tensor delayed_amax = torch::Tensor()
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out1{}, tmap_out1_t{}, tmap_sc_row1{}, tmap_sc_col1{};
    alignas(64) CUtensorMap tmap_out2{}, tmap_out2_t{}, tmap_sc_row2{}, tmap_sc_col2{};
    create_raw_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1,
        tmap_out1, tmap_out1_t, tmap_sc_row1, tmap_sc_col1);
    create_raw_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_2, row_sc_2, col_fp4_2, col_sc_2,
        tmap_out2, tmap_out2_t, tmap_sc_row2, tmap_sc_col2);

    const int dshmem = tk_localcta_fused_direct::direct_fused_dual_shmem_size<RETURN_TRANSPOSE>();
    auto kernel =
        tk_localcta_fused_direct::localcta_silu_deriv_quantize_split_direct_raw_kernel<
            RETURN_TRANSPOSE, DELAYED_SCALING>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto& work_counter = get_localcta_persistent_counter(dh.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    float* amax_out1 = nullptr;
    float* amax_out2 = nullptr;
    if constexpr (DELAYED_SCALING) {
        TORCH_CHECK(delayed_amax.defined() && delayed_amax.is_cuda() &&
                    delayed_amax.scalar_type() == torch::kFloat32 &&
                    delayed_amax.numel() >= 2,
                    "delayed localCTA SiLU-deriv requires a CUDA float32 amax buffer with at least two elements");
        amax_out1 = delayed_amax.data_ptr<float>();
        amax_out2 = delayed_amax.data_ptr<float>() + 1;
    }
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        tmap_out1, tmap_out1_t, tmap_sc_row1, tmap_sc_col1,
        row_sg_chunk_1.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_chunk_1.data_ptr<float>() : row_sg_chunk_1.data_ptr<float>(),
        tmap_out2, tmap_out2_t, tmap_sc_row2, tmap_sc_col2,
        row_sg_chunk_2.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_chunk_2.data_ptr<float>() : row_sg_chunk_2.data_ptr<float>(),
        amax_out1,
        amax_out2,
        M, H, args);
}

template <bool RETURN_TRANSPOSE>
static void launch_localcta_cluster_silu_deriv_split_prepared(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_prepared_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_prepared_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_prepared_2,
    torch::Tensor col_fp4_2,
    torch::Tensor col_sc_prepared_2,
    torch::Tensor row_sg_2,
    torch::Tensor col_sg_2
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1_raw{}, tmap_h3{};
    alignas(64) CUtensorMap tmap_out1{}, tmap_out1_t{}, tmap_sc_row_prepared1{}, tmap_sc_col_prepared1{};
    alignas(64) CUtensorMap tmap_out2{}, tmap_out2_t{}, tmap_sc_row_prepared2{}, tmap_sc_col_prepared2{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h1_raw, h1_raw.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, H, 16);
    create_prepared_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_1, row_sc_prepared_1, col_fp4_1, col_sc_prepared_1,
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1);
    create_prepared_output_tmaps_strided<RETURN_TRANSPOSE>(
        row_fp4_2, row_sc_prepared_2, col_fp4_2, col_sc_prepared_2,
        tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2);

    constexpr int threads = 160;
    const int dshmem = tk_localcta_fused_direct::direct_fused_cluster_split_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_fused_direct::localcta_silu_deriv_quantize_split_cluster_multicast_kernel<
        threads, RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int dev = 0;
    cudaGetDevice(&dev);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, kernel, threads, dshmem);
    int num_clusters = max_bps * num_sms / 2;
    if (num_clusters > total_tiles) {
        num_clusters = total_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(threads, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = at::cuda::getCurrentCUDAStream().stream();
    config.attrs = attrs;
    config.numAttrs = 2;

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        tmap_dh, tmap_h1_raw, tmap_h3,
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1,
        row_sg_1.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_1.data_ptr<float>() : row_sg_1.data_ptr<float>(),
        tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2,
        row_sg_2.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_2.data_ptr<float>() : row_sg_2.data_ptr<float>(),
        M, H, blocks_X, total_tiles);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchKernelEx failed for localCTA split-cluster fused silu-deriv quant: ",
                cudaGetErrorString(err));
}

template <bool RETURN_TRANSPOSE = true>
inline int localcta_col_only_shmem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFFS_NUM_IN * tk_localcta::BUFF_IN_ELEMS * (int)sizeof(tk_localcta::IType),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(tk_localcta::BUFFS_NUM_OUT_TR * tk_localcta::BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(tk_localcta::LocalCTAConfig::CHUNK_DIM_X * tk_localcta::SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;
    return in_bytes + out_tr_bytes + sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

__device__ __forceinline__ void localcta_load_raw_chunk(
    tk_localcta::IType* sIn_ptr,
    const __nv_bfloat16* input,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = tk_localcta::LocalCTAConfig::CHUNK_DIM_Y * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += tk_localcta::THREADS * VEC) {
        const int row = idx / tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = (int64_t)global_row * cols + global_col;
            const int2 in = *reinterpret_cast<const int2*>(input + base);
            const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&in.x);
            const __nv_bfloat162 v1 = *reinterpret_cast<const __nv_bfloat162*>(&in.y);
            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 0, v0.x);
            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 1, v0.y);
            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 2, v1.x);
            tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, col + 3, v1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    out = input[(int64_t)global_row * cols + block_offset_X + c];
                }
                tk_localcta_fused_direct::store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
}

template <bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(tk_localcta::THREADS)
localcta_quantize_col_only_prepared_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ row_sg_chunks,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
    tk_localcta::LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = transformer_engine::common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFFS_NUM_IN * tk_localcta::BUFF_IN_ELEMS * (int)sizeof(tk_localcta::IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFFS_NUM_OUT_TR * tk_localcta::BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::LocalCTAConfig::CHUNK_DIM_X * tk_localcta::SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    auto* sIn_ptr = reinterpret_cast<tk_localcta::IType*>(dshmem);
    auto* sOut_tr_ptr = reinterpret_cast<transformer_engine::fp4e2m1x2*>(dshmem + in_bytes);
    auto* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_tr_bytes);
    auto& sOut_tr = *reinterpret_cast<tk_localcta::OType2xt3D*>(sOut_tr_ptr);

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = static_cast<int>(s_chunk_id % args.tiles_X);
        const int ctaid_Y = static_cast<int>(s_chunk_id / args.tiles_X);
        const int block_offset_Y = ctaid_Y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_Y_tr = ctaid_X * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;

        localcta_load_raw_chunk(sIn_ptr, input, (int)rows, (int)cols, block_offset_Y, block_offset_X);

        const float sg_val = row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X];
        const float amax_val = sg_val * tk_localcta::localcta_global_scale_num();
        const float S_enc = tk_localcta::compute_localcta_encode_scaling_factor_FP4(amax_val);
        if (leading) {
            const int tiles_Y = static_cast<int>(rows / tk_localcta::LocalCTAConfig::CHUNK_DIM_Y);
            col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
        }
        __syncthreads();

        int buff_out_tr = 0;
        #pragma unroll
        for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
            const int stage_Y = t / tk_localcta::TILES_X;
            const int stage_X = t % tk_localcta::TILES_X;
            const int stage_offset_Y = stage_Y * tk_localcta::TILE_DIM_Y;
            const int stage_offset_X = stage_X * tk_localcta::TILE_DIM_X;

            if (t > 0 && leading) {
                transformer_engine::ptx::cp_async_bulk_wait_group_read<1>();
            }

            tk_localcta::colwise_scaling<ENCODE_CENTRIC>(
                sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc,
                stage_Y, stage_X, t, buff_out_tr);

            transformer_engine::ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                transformer_engine::ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                transformer_engine::ptx::cp_async_bulk_commit_group();
            }

            buff_out_tr = (buff_out_tr + 1) % tk_localcta::BUFFS_NUM_OUT_TR;
        }

        if (leading) {
            transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        const int cnt = min((int)tk_localcta::SCALES_PER_CHUNK_Y, chunk_rows / tk_localcta::SCALE_DIM);
        tk_localcta::swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
        transformer_engine::ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        tk_localcta::scale_swizzled_scales_inplace(
            sSFcolwise_ptr,
            tk_localcta::LocalCTAConfig::CHUNK_DIM_X * tk_localcta::SCALES_PER_CHUNK_Y,
            sg_val);
        transformer_engine::ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tk_localcta::tma_store_scales_2x512(
                tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

static void launch_localcta_quantize_col_only_prepared(
    torch::Tensor input,
    torch::Tensor row_sg,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor col_sg
) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(K / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out_t{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                  tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_col_prepared, col_sc_prepared.data_ptr(),
                  ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    const int dshmem = localcta_col_only_shmem_size<true>();
    auto kernel = localcta_quantize_col_only_prepared_kernel<true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        row_sg.data_ptr<float>(),
        tmap_out_t,
        tmap_sc_col_prepared,
        col_sg.data_ptr<float>(),
        M, K, args);
}

void quantize_into_outputs(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    torch::Tensor row_sc_prepared = torch::Tensor(),
    torch::Tensor col_sc_prepared = torch::Tensor(),
    bool use_2cta = false
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(K / 128);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto *row_sg_ptr = row_sg.data_ptr<float>();
    auto *col_sg_ptr = return_transpose ? col_sg.data_ptr<float>() : row_sg_ptr;
    const bool write_raw_scales = row_sc.defined() && row_sc.numel() > 0;
    const bool write_prepared = row_sc_prepared.defined() && row_sc_prepared.numel() > 0;
    const bool prepared_only = write_prepared && !write_raw_scales;
    const bool use_1cta_tuned = prepared_only && !use_2cta &&
        should_use_localcta1_prepared_auto(M, K, return_transpose);
    TORCH_CHECK(write_raw_scales || write_prepared,
                "quantize_into_outputs requires raw scales, prepared scales, or both");

    torch::Tensor cluster_amax_scratch;
    float *cluster_amax_scratch_ptr = nullptr;
    torch::Tensor work_counter;
    unsigned int *work_counter_ptr = nullptr;
    if (use_2cta && !prepared_only) {
        const int total_macro_tiles = blocks_X * ((blocks_Y + 1) / 2);
        auto &ci = get_localcta_cached_info();
        int num_clusters = ci.num_sms;
        if (num_clusters > total_macro_tiles) {
            num_clusters = total_macro_tiles;
        }
        if (num_clusters <= 0) {
            num_clusters = 1;
        }
        cluster_amax_scratch = torch::zeros(
            {num_clusters * 2},
            torch::dtype(torch::kFloat32).device(input.device()));
        cluster_amax_scratch_ptr = cluster_amax_scratch.data_ptr<float>();
        work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
        work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, empty_sc_row_raw{}, tmap_sc_col_prepared{}, empty_sc_col_raw{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 16);
    const bool raw_outputs_contiguous =
        row_fp4.is_contiguous() &&
        (!write_raw_scales || row_sc.is_contiguous()) &&
        (!return_transpose || (col_fp4.is_contiguous() && (!write_raw_scales || col_sc.is_contiguous())));
    const bool prepared_outputs_contiguous =
        row_fp4.is_contiguous() &&
        (!write_prepared || row_sc_prepared.is_contiguous()) &&
        (!return_transpose || (col_fp4.is_contiguous() && (!write_prepared || col_sc_prepared.is_contiguous())));

    if (write_raw_scales) {
        TORCH_CHECK(row_sc.is_cuda(), "row_sc must be a CUDA tensor");
        if (return_transpose) {
            TORCH_CHECK(col_sc.is_cuda(), "col_sc must be a CUDA tensor");
        }
        if (raw_outputs_contiguous) {
            if (return_transpose) {
                create_raw_output_tmaps<true>(
                    row_fp4, row_sc, col_fp4, col_sc,
                    tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
            } else {
                create_raw_output_tmaps<false>(
                    row_fp4, row_sc, col_fp4, col_sc,
                    tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
            }
        } else {
            if (return_transpose) {
                create_raw_output_tmaps_strided<true>(
                    row_fp4, row_sc, col_fp4, col_sc,
                    tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
            } else {
                create_raw_output_tmaps_strided<false>(
                    row_fp4, row_sc, col_fp4, col_sc,
                    tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
            }
        }
    } else {
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                      tk_localcta::BUFF_DIM_X, K, 4);
        if (return_transpose) {
            create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, tk_localcta::BUFF_DIM_X,
                          tk_localcta::BUFF_DIM_Y, M, 4);
        }
    }

    if (write_prepared) {
        TORCH_CHECK(row_sc_prepared.is_cuda(), "row_sc_prepared must be a CUDA tensor");
        if (return_transpose) {
            TORCH_CHECK(col_sc_prepared.is_cuda(), "col_sc_prepared must be a CUDA tensor");
        }
        if (write_raw_scales) {
            TORCH_CHECK(row_sc_prepared.sizes() == row_sc.sizes(),
                        "row_sc_prepared must match row_sc shape");
            if (return_transpose) {
                TORCH_CHECK(col_sc_prepared.sizes() == col_sc.sizes(),
                            "col_sc_prepared must match col_sc shape");
            }
        }
        if (prepared_outputs_contiguous) {
            if (return_transpose) {
                create_prepared_output_tmaps<true>(
                    row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
                    tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);
            } else {
                create_prepared_output_tmaps<false>(
                    row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
                    tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);
            }
        } else {
            if (return_transpose) {
                create_prepared_output_tmaps_strided<true>(
                    row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
                    tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);
            } else {
                create_prepared_output_tmaps_strided<false>(
                    row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
                    tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared);
            }
        }
    }

    if (encode_centric) {
        if (return_transpose) {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<true, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<true, true>(tmap_in, tmap_out, tmap_out_t,
                                                           tmap_sc_row, tmap_sc_col,
                                                           tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                           row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                           M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<true, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, empty_sc_row_raw, tmap_sc_col_prepared, empty_sc_col_raw,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<true, true>(tmap_in, tmap_out, tmap_out_t,
                                                      tmap_sc_row, tmap_sc_col,
                                                      tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                      row_sg_ptr, col_sg_ptr,
                                                      work_counter_ptr,
                                                      M, K, write_raw_scales, write_prepared, stream);
                }
            }
        } else {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<false, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<false, true>(tmap_in, tmap_out, tmap_out_t,
                                                            tmap_sc_row, tmap_sc_col,
                                                            tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                            row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                            M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<false, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, empty_sc_row_raw, tmap_sc_col_prepared, empty_sc_col_raw,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<false, true>(tmap_in, tmap_out, tmap_out_t,
                                                       tmap_sc_row, tmap_sc_col,
                                                       tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                       row_sg_ptr, col_sg_ptr,
                                                       work_counter_ptr,
                                                       M, K, write_raw_scales, write_prepared, stream);
                }
            }
        }
    } else {
        if (return_transpose) {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<true, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<true, false>(tmap_in, tmap_out, tmap_out_t,
                                                            tmap_sc_row, tmap_sc_col,
                                                            tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                            row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                            M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<true, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, empty_sc_row_raw, tmap_sc_col_prepared, empty_sc_col_raw,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<true, false>(tmap_in, tmap_out, tmap_out_t,
                                                       tmap_sc_row, tmap_sc_col,
                                                       tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                       row_sg_ptr, col_sg_ptr,
                                                       work_counter_ptr,
                                                       M, K, write_raw_scales, write_prepared, stream);
                }
            }
        } else {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<false, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<false, false>(tmap_in, tmap_out, tmap_out_t,
                                                             tmap_sc_row, tmap_sc_col,
                                                             tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                             row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                             M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<false, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, empty_sc_row_raw, tmap_sc_col_prepared, empty_sc_col_raw,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<false, false>(tmap_in, tmap_out, tmap_out_t,
                                                        tmap_sc_row, tmap_sc_col,
                                                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                        row_sg_ptr, col_sg_ptr,
                                                        work_counter_ptr,
                                                        M, K, write_raw_scales, write_prepared, stream);
                }
            }
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_for_gemm failed: ",
                cudaGetErrorString(err));
}

void quantize_into_outputs_v3_final_sg(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "final-SG v4 producer is only valid for the outer-scale contract");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(row_fp4.is_contiguous() && row_sc.is_contiguous(),
                "final-SG producer requires contiguous row outputs");
    TORCH_CHECK(!return_transpose || (col_fp4.is_contiguous() && col_sc.is_contiguous()),
                "final-SG producer requires contiguous col outputs when return_transpose=True");

    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_sg_chunk = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg_chunk = return_transpose
        ? torch::empty({K / 128, M / 128}, opts_f32)
        : torch::empty({0}, opts_f32);

    launch_scan_single_sg(input, return_transpose, row_sg_chunk, col_sg_chunk);

    auto stream = at::cuda::getCurrentCUDAStream();
    reduce_row_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(M)), 256, 0, stream>>>(
        row_sg_chunk.data_ptr<float>(),
        row_sg.data_ptr<float>(),
        static_cast<int>(M / 128),
        static_cast<int>(K / 128));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_tiles_kernel failed: ", cudaGetErrorString(err));
    }
    if (return_transpose) {
        reduce_col_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(K)), 256, 0, stream>>>(
            col_sg_chunk.data_ptr<float>(),
            col_sg.data_ptr<float>(),
            static_cast<int>(K / 128),
            static_cast<int>(M / 128));
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_kernel failed: ", cudaGetErrorString(err));
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 16);
    if (return_transpose) {
        create_raw_output_tmaps<true>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    } else {
        create_raw_output_tmaps<false>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    }

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto *row_sg_ptr = row_sg.data_ptr<float>();
    auto *col_sg_ptr = return_transpose ? col_sg.data_ptr<float>() : row_sg_ptr;
    if (encode_centric) {
        if (return_transpose) {
            launch_localcta_quant_final_sg<true, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream);
        } else {
            launch_localcta_quant_final_sg<false, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream);
        }
    } else {
        if (return_transpose) {
            launch_localcta_quant_final_sg<true, false>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream);
        } else {
            launch_localcta_quant_final_sg<false, false>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream);
        }
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_for_gemm_final_sg failed: ",
                cudaGetErrorString(err));
}

void quantize_into_outputs_opt(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool row_data_sr,
    bool col_data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    torch::Tensor persistent_rng_state = torch::Tensor()
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(row_sc.defined() && row_sc.numel() > 0,
                "v4 opt quantize currently requires raw row scales");
    if (return_transpose) {
        TORCH_CHECK(col_sc.defined() && col_sc.numel() > 0,
                    "v4 opt quantize currently requires raw col scales when return_transpose=True");
    } else {
        col_rht = false;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    auto *row_sg_ptr = row_sg.data_ptr<float>();
    auto *col_sg_ptr = return_transpose ? col_sg.data_ptr<float>() : row_sg_ptr;

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 16);

    const bool raw_outputs_contiguous =
        row_fp4.is_contiguous() &&
        row_sc.is_contiguous() &&
        (!return_transpose || (col_fp4.is_contiguous() && col_sc.is_contiguous()));
    if (raw_outputs_contiguous) {
        if (return_transpose) {
            create_raw_output_tmaps<true>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        } else {
            create_raw_output_tmaps<false>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        }
    } else {
        if (return_transpose) {
            create_raw_output_tmaps_strided<true>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        } else {
            create_raw_output_tmaps_strided<false>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        }
    }

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    const bool axis_selective_sr = row_data_sr != col_data_sr;
    if (axis_selective_sr) {
        TORCH_CHECK(return_transpose,
                    "axis-selective localCTA data SR requires return_transpose=True");
        const bool paired_row_sr_col_rht =
            row_data_sr && !col_data_sr && !row_rht && col_rht && random_sign;
        TORCH_CHECK(!scale_sr,
                    "axis-selective localCTA data SR cannot be combined with scale SR");
        TORCH_CHECK(
            (!row_rht && !col_rht && !random_sign) || paired_row_sr_col_rht,
            "axis-selective localCTA data SR supports only the paired "
            "row-SR/column-RHT/fixed-sign experiment");
        if (encode_centric) {
            launch_localcta_quant_axis_sr<true, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, work_counter_ptr,
                M, K, row_data_sr, col_data_sr,
                row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream,
                persistent_rng_state);
        } else {
            launch_localcta_quant_axis_sr<true, false>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, work_counter_ptr,
                M, K, row_data_sr, col_data_sr,
                row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream,
                persistent_rng_state);
        }
    } else if (encode_centric) {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                row_data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream,
                persistent_rng_state);
        } else {
            launch_localcta_quant_opt_dispatch<false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                row_data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream,
                persistent_rng_state);
        }
    } else {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, false>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                row_data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream,
                persistent_rng_state);
        } else {
            launch_localcta_quant_opt_dispatch<false, false>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                row_data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream,
                persistent_rng_state);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_for_gemm_opt failed: ",
                cudaGetErrorString(err));
}

void quantize_into_outputs_v3_opt(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool row_data_sr,
    bool col_data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    torch::Tensor persistent_rng_state = torch::Tensor()
) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_sg_chunk = torch::empty({input.size(0) / 128, input.size(1) / 128}, opts_f32);
    auto col_sg_chunk = return_transpose
        ? torch::empty({input.size(1) / 128, input.size(0) / 128}, opts_f32)
        : torch::empty({0}, opts_f32);

    quantize_into_outputs_opt(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc,
        row_sg_chunk, col_sg_chunk,
        row_data_sr, col_data_sr, scale_sr, row_rht, col_rht, random_sign,
        rng_seed, rng_subsequence_base, persistent_rng_state);

    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        TORCH_CHECK(input.size(0) % 256 == 0 && input.size(1) % 256 == 0,
                    "localCTA v3 tilegrid256 contract requires M and K to be multiples of 256");
        fold_quant_scales_v3_tilegrid256(row_sc, row_sg_chunk, col_sc, col_sg_chunk);
        finalize_quant_contract_v3_tilegrid256(row_sg_chunk, row_sg, col_sg);
    } else {
        finalize_quant_contract_v3(row_sc, row_sg_chunk, row_sg, col_sc, col_sg_chunk, col_sg);
    }
}

void quantize_into_outputs_v3_opt_final_sg(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    bool four_over_six_mae = false
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "opt final-SG producer is only valid for the outer-scale contract");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(row_fp4.is_contiguous() && row_sc.is_contiguous(),
                "opt final-SG producer requires contiguous row outputs");
    TORCH_CHECK(!return_transpose || (col_fp4.is_contiguous() && col_sc.is_contiguous()),
                "opt final-SG producer requires contiguous col outputs when return_transpose=True");
    TORCH_CHECK(
        !four_over_six_mae ||
            (!data_sr && !scale_sr && !row_rht && !col_rht && !random_sign),
        "Four-over-six MAE requires deterministic final-SG quantization without RHT");

    auto stream = at::cuda::getCurrentCUDAStream();
    if (use_localcta_v4_final_sg_opt_direct_final_scan()) {
        launch_scan_single_sg_opt_direct_final(
            input, return_transpose, row_sg, col_sg,
            row_rht, col_rht, random_sign);
    } else {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
        auto row_sg_chunk = torch::empty({M / 128, K / 128}, opts_f32);
        auto col_sg_chunk = return_transpose
            ? torch::empty({K / 128, M / 128}, opts_f32)
            : torch::empty({0}, opts_f32);
        launch_scan_single_sg_opt(
            input, return_transpose, row_sg_chunk, col_sg_chunk,
            row_rht, col_rht, random_sign);

        reduce_row_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(M)), 256, 0, stream>>>(
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(M / 128),
            static_cast<int>(K / 128));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_tiles_kernel failed: ", cudaGetErrorString(err));
        }
        if (return_transpose) {
            reduce_col_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(K)), 256, 0, stream>>>(
                col_sg_chunk.data_ptr<float>(),
                col_sg.data_ptr<float>(),
                static_cast<int>(K / 128),
                static_cast<int>(M / 128));
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_kernel failed: ", cudaGetErrorString(err));
        }
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap empty_sc_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 16);
    if (return_transpose) {
        create_raw_output_tmaps<true>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    } else {
        create_raw_output_tmaps<false>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    }

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto *row_sg_ptr = row_sg.data_ptr<float>();
    auto *col_sg_ptr = return_transpose ? col_sg.data_ptr<float>() : row_sg_ptr;
    if (four_over_six_mae) {
        if (encode_centric) {
            if (return_transpose) {
                launch_localcta_quant_four_over_six_final_sg<true, true>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                    M, K, stream);
            } else {
                launch_localcta_quant_four_over_six_final_sg<false, true>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                    M, K, stream);
            }
        } else if (return_transpose) {
            launch_localcta_quant_four_over_six_final_sg<true, false>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, stream);
        } else {
            launch_localcta_quant_four_over_six_final_sg<false, false>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, stream);
        }
    } else if (encode_centric) {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, true, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream);
        } else {
            launch_localcta_quant_opt_dispatch<false, true, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream);
        }
    } else {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, false, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream);
        } else {
            launch_localcta_quant_opt_dispatch<false, false, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream);
        }
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_for_gemm_opt_final_sg failed: ",
                cudaGetErrorString(err));
}

void quantize_into_outputs_v3_atomic_final_sg(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    int64_t quant_rows = -1,
    int64_t quant_cols = -1,
    bool paired_fixed_sign_col_rht = false
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "atomic final-SG producer is only valid for the outer-scale contract");

    const int64_t input_rows = input.size(0);
    const int64_t input_cols = input.size(1);
    const int64_t M = quant_rows < 0 ? input_rows : quant_rows;
    const int64_t K = quant_cols < 0 ? input_cols : quant_cols;
    if (paired_fixed_sign_col_rht) {
        TORCH_CHECK(return_transpose,
                    "paired column-RHT atomic producer requires return_transpose=True");
        TORCH_CHECK(M == input_rows && K == input_cols,
                    "paired column-RHT atomic producer does not support padded quantized extents");
    }
    TORCH_CHECK(input_rows % 128 == 0 && input_cols % 128 == 0,
                "logical input dimensions must be multiples of 128");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "atomic final-SG producer requires M and K to be multiples of 128");
    TORCH_CHECK(M >= input_rows && K >= input_cols,
                "padded quantized extent must cover the logical input");
    TORCH_CHECK(row_fp4.size(0) == M && row_fp4.size(1) * 2 == K,
                "row FP4 output does not match the requested quantized extent");
    TORCH_CHECK(!return_transpose ||
                    (col_fp4.size(0) == K && col_fp4.size(1) * 2 == M),
                "column FP4 output does not match the requested quantized extent");
    TORCH_CHECK(row_fp4.is_contiguous() && row_sc.is_contiguous(),
                "atomic final-SG producer requires contiguous row outputs");
    TORCH_CHECK(!return_transpose || (col_fp4.is_contiguous() && col_sc.is_contiguous()),
                "atomic final-SG producer requires contiguous col outputs when return_transpose=True");

    const bool reuse_atomic_scratch = use_localcta_v4_reuse_atomic_scratch();
    torch::Tensor row_sg_chunk;
    torch::Tensor col_sg_chunk;
    if (!reuse_atomic_scratch) {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
        row_sg_chunk = torch::empty({M / 128, K / 128}, opts_f32);
        col_sg_chunk = return_transpose
            ? torch::empty({K / 128, M / 128}, opts_f32)
            : torch::empty({0}, opts_f32);
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap empty_sc_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), input_rows, input_cols,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X,
                  input_cols, 16);
    if (return_transpose) {
        create_raw_output_tmaps<true>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    } else {
        create_raw_output_tmaps<false>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool fused_atomic_init = use_localcta_v4_fused_atomic_init();
    torch::Tensor work_counter;
    if (reuse_atomic_scratch) {
        auto& scratch = get_localcta_atomic_scratch(
            input.device(), stream, M, K, return_transpose);
        row_sg_chunk = scratch.row_sg_chunk;
        col_sg_chunk = scratch.col_sg_chunk;
        work_counter = scratch.work_counter;
    }
    if (!fused_atomic_init) {
        cudaMemsetAsync(row_sg.data_ptr<float>(), 0,
                        row_sg.numel() * sizeof(float), stream);
        if (return_transpose) {
            cudaMemsetAsync(col_sg.data_ptr<float>(), 0,
                            col_sg.numel() * sizeof(float), stream);
        }
    }
    if (!reuse_atomic_scratch) {
        auto counter_opts = torch::dtype(torch::kInt32).device(input.device());
        work_counter = fused_atomic_init
            ? torch::empty({1}, counter_opts)
            : torch::zeros({1}, counter_opts);
    } else if (!fused_atomic_init) {
        cudaMemsetAsync(work_counter.data_ptr<int>(), 0, sizeof(int), stream);
    }
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto *row_chunk_ptr = row_sg_chunk.data_ptr<float>();
    auto *col_chunk_ptr = return_transpose ? col_sg_chunk.data_ptr<float>() : row_chunk_ptr;
    auto *row_final_ptr = row_sg.data_ptr<float>();
    auto *col_final_ptr = return_transpose ? col_sg.data_ptr<float>() : row_final_ptr;

    if (fused_atomic_init) {
        const int row_sg_count = static_cast<int>(row_sg.numel());
        const int col_sg_count = return_transpose ? static_cast<int>(col_sg.numel()) : 0;
        const int init_threads = static_cast<int>(localcta_env_int64(
            "USE_TK_LOCALCTA_V4_FUSED_ATOMIC_INIT_THREADS", 128));
#define LOCALCTA_LAUNCH_ATOMIC_INIT(THREADS) \
        initialize_atomic_final_sg_kernel<THREADS><<<1, THREADS, 0, stream>>>( \
            row_final_ptr, row_sg_count, \
            return_transpose ? col_final_ptr : nullptr, col_sg_count, \
            work_counter_ptr)
        if (init_threads <= 32) {
            LOCALCTA_LAUNCH_ATOMIC_INIT(32);
        } else if (init_threads <= 64) {
            LOCALCTA_LAUNCH_ATOMIC_INIT(64);
        } else if (init_threads <= 128) {
            LOCALCTA_LAUNCH_ATOMIC_INIT(128);
        } else {
            LOCALCTA_LAUNCH_ATOMIC_INIT(256);
        }
#undef LOCALCTA_LAUNCH_ATOMIC_INIT
    }

    if (paired_fixed_sign_col_rht) {
        if (encode_centric) {
            launch_localcta_quant_opt<
                true, true, false, false, false, false, true, true,
                false, false, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_chunk_ptr, col_chunk_ptr, nullptr, nullptr,
                work_counter_ptr, M, K, true, false, 0, 0, nullptr, stream,
                row_final_ptr, col_final_ptr);
        } else {
            launch_localcta_quant_opt<
                true, false, false, false, false, false, true, true,
                false, false, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_chunk_ptr, col_chunk_ptr, nullptr, nullptr,
                work_counter_ptr, M, K, true, false, 0, 0, nullptr, stream,
                row_final_ptr, col_final_ptr);
        }
    } else if (encode_centric) {
        if (return_transpose) {
            launch_localcta_quant_opt<
                true, true, false, false, false, false, false, false,
                false, false, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_chunk_ptr, col_chunk_ptr, nullptr, nullptr,
                work_counter_ptr, M, K, true, false, 0, 0, nullptr, stream,
                row_final_ptr, col_final_ptr);
        } else {
            launch_localcta_quant_opt<
                false, true, false, false, false, false, false, false,
                false, false, false, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_chunk_ptr, col_chunk_ptr, nullptr, nullptr,
                work_counter_ptr, M, K, true, false, 0, 0, nullptr, stream,
                row_final_ptr, col_final_ptr);
        }
    } else if (return_transpose) {
        launch_localcta_quant_opt<
            true, false, false, false, false, false, false, false,
            false, false, false, false, true>(
            tmap_in, tmap_out, tmap_out_t,
            tmap_sc_row, tmap_sc_col,
            empty_sc_prepared, empty_sc_prepared,
            row_chunk_ptr, col_chunk_ptr, nullptr, nullptr,
            work_counter_ptr, M, K, true, false, 0, 0, nullptr, stream,
            row_final_ptr, col_final_ptr);
    } else {
        launch_localcta_quant_opt<
            false, false, false, false, false, false, false, false,
            false, false, false, false, true>(
            tmap_in, tmap_out, tmap_out_t,
            tmap_sc_row, tmap_sc_col,
            empty_sc_prepared, empty_sc_prepared,
            row_chunk_ptr, col_chunk_ptr, nullptr, nullptr,
            work_counter_ptr, M, K, true, false, 0, 0, nullptr, stream,
            row_final_ptr, col_final_ptr);
    }

    rescale_quant_contract_v3_from_final_sg(
        row_sc, row_sg_chunk, row_sg,
        col_sc, col_sg_chunk, col_sg);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_quantize_for_gemm_atomic_final_sg failed: ",
                cudaGetErrorString(err));
}

void quantize_rmsnorm_into_outputs_opt(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(), "gamma must be contiguous CUDA tensor");
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.is_contiguous(), "inv_rms must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");
    TORCH_CHECK(inv_rms.scalar_type() == torch::kFloat32 && inv_rms.dim() == 1,
                "inv_rms must be fp32 [M]");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(gamma.size(0) == K, "gamma must match input K");
    TORCH_CHECK(inv_rms.size(0) == M, "inv_rms must match input M");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(row_sc.defined() && row_sc.numel() > 0,
                "v4 opt quantize currently requires raw row scales");
    if (return_transpose) {
        TORCH_CHECK(col_sc.defined() && col_sc.numel() > 0,
                    "v4 opt quantize currently requires raw col scales when return_transpose=True");
    } else {
        col_rht = false;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    auto *row_sg_ptr = row_sg.data_ptr<float>();
    auto *col_sg_ptr = return_transpose ? col_sg.data_ptr<float>() : row_sg_ptr;
    auto *gamma_ptr = reinterpret_cast<const tk_localcta::IType*>(gamma.data_ptr());
    auto *inv_rms_ptr = inv_rms.data_ptr<float>();

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 16);

    const bool raw_outputs_contiguous =
        row_fp4.is_contiguous() &&
        row_sc.is_contiguous() &&
        (!return_transpose || (col_fp4.is_contiguous() && col_sc.is_contiguous()));
    if (raw_outputs_contiguous) {
        if (return_transpose) {
            create_raw_output_tmaps<true>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        } else {
            create_raw_output_tmaps<false>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        }
    } else {
        if (return_transpose) {
            create_raw_output_tmaps_strided<true>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        } else {
            create_raw_output_tmaps_strided<false>(
                row_fp4, row_sc, col_fp4, col_sc,
                tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
        }
    }

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    if (encode_centric) {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, true, true, false>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream);
        } else {
            launch_localcta_quant_opt_dispatch<false, true, true, false>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream);
        }
    } else {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, false, true, false>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream);
        } else {
            launch_localcta_quant_opt_dispatch<false, false, true, false>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                tmap_sc_row_prepared, tmap_sc_col_prepared,
                row_sg_ptr, col_sg_ptr, gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_rmsnorm_quantize_for_gemm_opt failed: ",
                cudaGetErrorString(err));
}

void quantize_rmsnorm_into_outputs_v3_opt(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_sg_chunk = torch::empty({input.size(0) / 128, input.size(1) / 128}, opts_f32);
    auto col_sg_chunk = return_transpose
        ? torch::empty({input.size(1) / 128, input.size(0) / 128}, opts_f32)
        : torch::empty({0}, opts_f32);

    quantize_rmsnorm_into_outputs_opt(
        input, gamma, inv_rms, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc,
        row_sg_chunk, col_sg_chunk,
        data_sr, scale_sr, row_rht, col_rht, random_sign,
        rng_seed, rng_subsequence_base);

    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        TORCH_CHECK(input.size(0) % 256 == 0 && input.size(1) % 256 == 0,
                    "localCTA v3 tilegrid256 contract requires M and K to be multiples of 256");
        fold_quant_scales_v3_tilegrid256(row_sc, row_sg_chunk, col_sc, col_sg_chunk);
        finalize_quant_contract_v3_tilegrid256(row_sg_chunk, row_sg, col_sg);
    } else {
        finalize_quant_contract_v3(row_sc, row_sg_chunk, row_sg, col_sc, col_sg_chunk, col_sg);
    }
}

void quantize_rmsnorm_into_outputs_v3_opt_final_sg(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    bool data_sr,
    bool scale_sr,
    bool row_rht,
    bool col_rht,
    bool random_sign,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    bool lean_rms_kernel,
    bool four_over_six_mae = false
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(), "gamma must be contiguous CUDA tensor");
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.is_contiguous(), "inv_rms must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");
    TORCH_CHECK(inv_rms.scalar_type() == torch::kFloat32 && inv_rms.dim() == 1,
                "inv_rms must be fp32 [M]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "RMSNorm final-SG producer is only valid for the outer-scale contract");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(gamma.size(0) == K, "gamma must match input K");
    TORCH_CHECK(inv_rms.size(0) == M, "inv_rms must match input M");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(row_fp4.is_contiguous() && row_sc.is_contiguous(),
                "RMSNorm final-SG producer requires contiguous row outputs");
    TORCH_CHECK(!return_transpose || (col_fp4.is_contiguous() && col_sc.is_contiguous()),
                "RMSNorm final-SG producer requires contiguous col outputs when return_transpose=True");
    TORCH_CHECK(
        !four_over_six_mae ||
            (!data_sr && !scale_sr && !row_rht && !col_rht && !random_sign),
        "Four-over-six MAE requires deterministic final-SG RMSNorm quantization without RHT");

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* gamma_ptr = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>());
    const float* inv_rms_ptr = inv_rms.data_ptr<float>();
    if (use_localcta_v4_final_sg_opt_direct_final_scan()) {
        launch_scan_single_sg_opt_direct_final_rmsnorm(
            input, return_transpose, row_sg, col_sg,
            gamma_ptr, inv_rms_ptr, row_rht, col_rht, random_sign);
    } else {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
        auto row_sg_chunk = torch::empty({M / 128, K / 128}, opts_f32);
        auto col_sg_chunk = return_transpose
            ? torch::empty({K / 128, M / 128}, opts_f32)
            : torch::empty({0}, opts_f32);
        launch_scan_single_sg_opt_rmsnorm(
            input, return_transpose, row_sg_chunk, col_sg_chunk,
            gamma_ptr, inv_rms_ptr, row_rht, col_rht, random_sign);

        reduce_row_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(M)), 256, 0, stream>>>(
            row_sg_chunk.data_ptr<float>(),
            row_sg.data_ptr<float>(),
            static_cast<int>(M / 128),
            static_cast<int>(K / 128));
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_tiles_kernel failed: ", cudaGetErrorString(err));
        if (return_transpose) {
            reduce_col_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(K)), 256, 0, stream>>>(
                col_sg_chunk.data_ptr<float>(),
                col_sg.data_ptr<float>(),
                static_cast<int>(K / 128),
                static_cast<int>(M / 128));
            err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_kernel failed: ", cudaGetErrorString(err));
        }
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap empty_sc_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 16);
    if (return_transpose) {
        create_raw_output_tmaps<true>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    } else {
        create_raw_output_tmaps<false>(
            row_fp4, row_sc, col_fp4, col_sc,
            tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    }

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto *row_sg_ptr = row_sg.data_ptr<float>();
    auto *col_sg_ptr = return_transpose ? col_sg.data_ptr<float>() : row_sg_ptr;
    const auto* quant_gamma_ptr = reinterpret_cast<const tk_localcta::IType*>(gamma.data_ptr());
    const bool use_lean_rms_kernel =
        lean_rms_kernel &&
        !data_sr && !scale_sr && !row_rht && !col_rht && !random_sign;
    if (four_over_six_mae) {
        if (encode_centric) {
            if (return_transpose) {
                launch_localcta_quant_four_over_six_final_sg<true, true, true>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr,
                    work_counter_ptr, M, K, stream);
            } else {
                launch_localcta_quant_four_over_six_final_sg<false, true, true>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr,
                    work_counter_ptr, M, K, stream);
            }
        } else if (return_transpose) {
            launch_localcta_quant_four_over_six_final_sg<true, false, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr,
                work_counter_ptr, M, K, stream);
        } else {
            launch_localcta_quant_four_over_six_final_sg<false, false, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr,
                work_counter_ptr, M, K, stream);
        }
    } else if (use_lean_rms_kernel && encode_centric) {
        if (return_transpose) {
            launch_localcta_quant_final_sg<true, true, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream,
                quant_gamma_ptr, inv_rms_ptr);
        } else {
            launch_localcta_quant_final_sg<false, true, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream,
                quant_gamma_ptr, inv_rms_ptr);
        }
    } else if (use_lean_rms_kernel) {
        if (return_transpose) {
            launch_localcta_quant_final_sg<true, false, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream,
                quant_gamma_ptr, inv_rms_ptr);
        } else {
            launch_localcta_quant_final_sg<false, false, true>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K, stream,
                quant_gamma_ptr, inv_rms_ptr);
        }
    } else if (encode_centric) {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, true, true, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream);
        } else {
            launch_localcta_quant_opt_dispatch<false, true, true, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream);
        }
    } else {
        if (return_transpose) {
            launch_localcta_quant_opt_dispatch<true, false, true, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, col_rht, random_sign,
                rng_seed, rng_subsequence_base, stream);
        } else {
            launch_localcta_quant_opt_dispatch<false, false, true, false, true>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                empty_sc_prepared, empty_sc_prepared,
                row_sg_ptr, col_sg_ptr, quant_gamma_ptr, inv_rms_ptr, work_counter_ptr,
                M, K, true, false,
                data_sr, scale_sr, row_rht, false, random_sign,
                rng_seed, rng_subsequence_base, stream);
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_rmsnorm_quantize_for_gemm_opt_final_sg failed: ",
                cudaGetErrorString(err));
}

template <bool CALL_FREE_TE_MATH>
__global__ void __launch_bounds__(tk_localcta::SILU_RAW_THREADS)
test_w2_transform_bf16_exact_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __nv_bfloat16* __restrict__ h3,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ tile_amax,
    int64_t cols,
    int tiles_x,
    bool fast_divide
) {
    static_assert(tk_localcta::SILU_RAW_THREADS == 256);
    static_assert(tk_localcta::BUFF_DIM_Y == 64 && tk_localcta::BUFF_DIM_X == 64);
    static_assert(tk_localcta::BUFF_IN_ELEMS % 4 == 0);
    constexpr int kVecElems = 4;
    constexpr int kVecsPerRow = tk_localcta::BUFF_DIM_X / kVecElems;
    constexpr int kTileVecs = tk_localcta::BUFF_IN_ELEMS / kVecElems;
    __shared__ __align__(16) int2 shared_h1[kTileVecs];
    __shared__ __align__(16) int2 shared_h3[kTileVecs];
    __shared__ float warp_max[tk_localcta::SILU_RAW_THREADS / 32];

    const int tile_x = static_cast<int>(blockIdx.x);
    const int tile_y = static_cast<int>(blockIdx.y);
    for (int vec = static_cast<int>(threadIdx.x);
         vec < kTileVecs;
         vec += tk_localcta::SILU_RAW_THREADS) {
        const int row = vec / kVecsPerRow;
        const int vec_col = vec % kVecsPerRow;
        const int64_t offset =
            static_cast<int64_t>(tile_y * tk_localcta::BUFF_DIM_Y + row) * cols +
            tile_x * tk_localcta::BUFF_DIM_X + vec_col * kVecElems;
        shared_h1[vec] = *reinterpret_cast<const int2*>(h1_raw + offset);
        shared_h3[vec] = *reinterpret_cast<const int2*>(h3 + offset);
    }
    __syncthreads();

    float local_max = tk_localcta::transform_silu_vectors_inplace_amax_linear<
        tk_localcta::SILU_RAW_THREADS, CALL_FREE_TE_MATH>(
            shared_h1, shared_h3, fast_divide);

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(
            local_max,
            __shfl_xor_sync(0xffffffff, local_max, mask));
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        warp_max[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
        local_max = lane < tk_localcta::SILU_RAW_THREADS / 32
            ? warp_max[lane]
            : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_max = fmaxf(
                local_max,
                __shfl_xor_sync(0xffffffff, local_max, mask));
        }
        if (lane == 0) {
            tile_amax[static_cast<int64_t>(tile_y) * tiles_x + tile_x] = local_max;
        }
    }
    __syncthreads();

    for (int vec = static_cast<int>(threadIdx.x);
         vec < kTileVecs;
         vec += tk_localcta::SILU_RAW_THREADS) {
        const int row = vec / kVecsPerRow;
        const int vec_col = vec % kVecsPerRow;
        const int64_t offset =
            static_cast<int64_t>(tile_y * tk_localcta::BUFF_DIM_Y + row) * cols +
            tile_x * tk_localcta::BUFF_DIM_X + vec_col * kVecElems;
        *reinterpret_cast<int2*>(output + offset) = shared_h1[vec];
    }
}

template <bool CALL_FREE_SCALE_MATH>
__global__ void test_scale_divide_kernel(
    const float* __restrict__ numerator,
    const float* __restrict__ denominator,
    float* __restrict__ output,
    int64_t numel
) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    output[index] = tk_localcta::localcta_scale_divide<
        CALL_FREE_SCALE_MATH>(numerator[index], denominator[index]);
}

torch::Tensor reconstruct_rowwise_impl(
    torch::Tensor fp4,
    torch::Tensor sc,
    torch::Tensor sg,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(fp4.is_cuda() && sc.is_cuda() && sg.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(fp4.device() == sc.device() && fp4.device() == sg.device(),
                "all tensors must be on the same CUDA device");
    TORCH_CHECK(fp4.scalar_type() == torch::kFloat4_e2m1fn_x2, "fp4 tensor dtype mismatch");
    TORCH_CHECK(sc.scalar_type() == torch::kFloat8_e4m3fn, "scale tensor dtype mismatch");
    TORCH_CHECK(sg.scalar_type() == torch::kFloat32, "sg tensor dtype mismatch");
    TORCH_CHECK(fp4.dim() == 2, "fp4 tensor must have rank 2");
    TORCH_CHECK(sg.dim() == 2, "sg tensor must have rank 2");
    TORCH_CHECK(fp4.is_contiguous() && sc.is_contiguous() && sg.is_contiguous(),
                "fp4, scale, and sg tensors must be contiguous");
    TORCH_CHECK(rows > 0 && cols > 0 && rows % 128 == 0 && cols % 128 == 0,
                "reconstruction rows and cols must be positive multiples of 128");
    TORCH_CHECK(fp4.size(0) == rows && fp4.size(1) * 2 == cols,
                "fp4 tensor shape does not match reconstruction extent");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    cols <= std::numeric_limits<int>::max(),
                "reconstruction extent exceeds the CUDA kernel index range");

    const int64_t expected_sc_numel =
        (rows / 128) * (cols / 64) * 512;
    TORCH_CHECK(sc.numel() == expected_sc_numel,
                "scale tensor has ", sc.numel(), " elements; expected ",
                expected_sc_numel, " for reconstruction extent [", rows, ", ", cols, "]");

    const int64_t outer_rows = (rows + 255) / 256;
    const bool is_outer_scale =
        (sg.size(0) == outer_rows && sg.size(1) == 1) ||
        (sg.size(0) == 1 && sg.size(1) == outer_rows);
    const bool is_tile_grid_256 =
        rows % 256 == 0 && cols % 256 == 0 &&
        sg.size(0) == rows / 256 && sg.size(1) == cols / 256;
    const bool is_chunk_grid_128 =
        sg.size(0) == rows / 128 && sg.size(1) == cols / 128;

    int sg_geometry = tk_localcta_reconstruct::SG_CHUNK_GRID_128;
    if (is_outer_scale) {
        sg_geometry = tk_localcta_reconstruct::SG_OUTER_SCALE;
    } else if (is_tile_grid_256) {
        sg_geometry = tk_localcta_reconstruct::SG_TILE_GRID_256;
    } else {
        TORCH_CHECK(is_chunk_grid_128,
                    "unsupported SG shape [", sg.size(0), ", ", sg.size(1),
                    "] for reconstruction extent [", rows, ", ", cols,
                    "]; expected outer scale [", outer_rows, ", 1] or [1, ",
                    outer_rows, "], tilegrid256 [", rows / 256, ", ",
                    cols / 256, "], or chunk grid [", rows / 128, ", ",
                    cols / 128, "]");
    }

    const c10::cuda::CUDAGuard device_guard(fp4.device());

    auto out = torch::empty({rows, cols}, torch::dtype(torch::kBFloat16).device(fp4.device()));
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t numel = rows * cols;
    const int threads = 256;
    const int blocks = (int)((numel + threads - 1) / threads);
    tk_localcta_reconstruct::reconstruct_rowwise_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_fp4x2_e2m1*>(fp4.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(sc.data_ptr()),
        sg.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)rows, (int)cols,
        sc.numel(), sg.numel(), (int)sg.size(1), sg_geometry);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_reconstruct failed: ",
                cudaGetErrorString(err));
    return out;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor>
tk_localcta_test_scale_divide_callfree(
    torch::Tensor numerator,
    torch::Tensor denominator
) {
    TORCH_CHECK(numerator.is_cuda() && denominator.is_cuda(),
                "scale divide oracle requires CUDA tensors");
    TORCH_CHECK(numerator.is_contiguous() && denominator.is_contiguous(),
                "scale divide oracle requires contiguous tensors");
    TORCH_CHECK(numerator.scalar_type() == torch::kFloat32 &&
                    denominator.scalar_type() == torch::kFloat32,
                "scale divide oracle requires float32 tensors");
    TORCH_CHECK(numerator.sizes() == denominator.sizes(),
                "scale divide oracle inputs must have identical shapes");
    TORCH_CHECK(numerator.device() == denominator.device(),
                "scale divide oracle inputs must share a CUDA device");
    const c10::cuda::CUDAGuard device_guard(numerator.device());
    const auto set_device_err = cudaSetDevice(numerator.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before scale divide oracle: ",
                cudaGetErrorString(set_device_err));

    auto precise = torch::empty_like(numerator);
    auto callfree = torch::empty_like(numerator);
    const int64_t numel = numerator.numel();
    if (numel == 0) {
        return std::make_tuple(precise, callfree);
    }
    constexpr int threads = 256;
    const int64_t blocks64 = (numel + threads - 1) / threads;
    TORCH_CHECK(blocks64 <= std::numeric_limits<int>::max(),
                "scale divide oracle input is too large");
    const int blocks = static_cast<int>(blocks64);
    auto stream = at::cuda::getCurrentCUDAStream();
    test_scale_divide_kernel<false><<<blocks, threads, 0, stream>>>(
        numerator.data_ptr<float>(), denominator.data_ptr<float>(),
        precise.data_ptr<float>(), numel);
    test_scale_divide_kernel<true><<<blocks, threads, 0, stream>>>(
        numerator.data_ptr<float>(), denominator.data_ptr<float>(),
        callfree.data_ptr<float>(), numel);
    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "scale divide oracle kernel failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(precise, callfree);
}

std::tuple<torch::Tensor, torch::Tensor>
tk_localcta_test_w2_transform_bf16_exact(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    bool fast_divide,
    bool call_free_te_math
) {
    TORCH_CHECK(h1_raw.is_cuda() && h3.is_cuda(),
                "W2 transform oracle requires CUDA tensors");
    TORCH_CHECK(h1_raw.is_contiguous() && h3.is_contiguous(),
                "W2 transform oracle requires contiguous tensors");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 &&
                    h3.scalar_type() == torch::kBFloat16,
                "W2 transform oracle requires BF16 tensors");
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(),
                "W2 transform oracle inputs must have identical shapes");
    TORCH_CHECK(h1_raw.device() == h3.device(),
                "W2 transform oracle inputs must share a CUDA device");
    TORCH_CHECK(h1_raw.dim() == 2,
                "W2 transform oracle requires rank-2 inputs");
    const int64_t rows = h1_raw.size(0);
    const int64_t cols = h1_raw.size(1);
    TORCH_CHECK(
        rows > 0 && cols > 0 &&
            rows % tk_localcta::BUFF_DIM_Y == 0 &&
            cols % tk_localcta::BUFF_DIM_X == 0,
        "W2 transform oracle dimensions must be positive multiples of the 64x64 tile");
    const c10::cuda::CUDAGuard device_guard(h1_raw.device());
    const auto set_device_err = cudaSetDevice(h1_raw.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before W2 transform oracle: ",
                cudaGetErrorString(set_device_err));

    auto output = torch::empty_like(h1_raw);
    const int tiles_y = static_cast<int>(rows / tk_localcta::BUFF_DIM_Y);
    const int tiles_x = static_cast<int>(cols / tk_localcta::BUFF_DIM_X);
    auto tile_amax = torch::empty(
        {tiles_y, tiles_x}, torch::dtype(torch::kFloat32).device(h1_raw.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch = [&](auto call_free_tag) {
        constexpr bool call_free = decltype(call_free_tag)::value;
        test_w2_transform_bf16_exact_kernel<call_free><<<
            dim3(static_cast<unsigned int>(tiles_x), static_cast<unsigned int>(tiles_y)),
            tk_localcta::SILU_RAW_THREADS,
            0,
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                tile_amax.data_ptr<float>(),
                cols,
                tiles_x,
                fast_divide);
    };
    if (call_free_te_math) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "W2 transform oracle kernel failed: ", cudaGetErrorString(err));
    return std::make_tuple(output, tile_amax);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm(torch::Tensor input,
                              bool return_transpose,
                              bool encode_centric) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before localCTA quantization: ",
                cudaGetErrorString(set_device_err));
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), return_transpose, input.device());
    if (use_localcta_v4_atomic_final_sg_producer() &&
        get_v3_contract_mode() != V3ContractMode::TileGrid256) {
        quantize_into_outputs_v3_atomic_final_sg(
            input, return_transpose, encode_centric,
            row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
        return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
    }
    if (use_localcta_v4_final_sg_producer() &&
        get_v3_contract_mode() != V3ContractMode::TileGrid256) {
        quantize_into_outputs_v3_opt_final_sg(
            input, return_transpose, encode_centric,
            row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
            false, false, false, false, false, 0, 0);
        return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
    }
    const bool use_2cta = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    quantize_into_outputs_v3(input, return_transpose, encode_centric,
                             row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                             torch::Tensor(), torch::Tensor(), use_2cta);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_padded(
    torch::Tensor input,
    int64_t output_rows,
    int64_t output_cols,
    bool return_transpose,
    bool encode_centric
) {
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "padded localCTA quantization requires the v4 outer-scale contract");
    TORCH_CHECK(output_rows % 256 == 0 && output_cols % 256 == 0,
                "localCTA GEMM padded quantization requires 256-aligned dimensions");
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before padded localCTA quantization: ",
                cudaGetErrorString(set_device_err));
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(
            output_rows, output_cols, return_transpose, input.device());
    quantize_into_outputs_v3_atomic_final_sg(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        output_rows, output_cols);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_final_sg(torch::Tensor input,
                                       bool return_transpose,
                                       bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs_v3_final_sg(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_atomic_final_sg(torch::Tensor input,
                                              bool return_transpose,
                                              bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs_v3_atomic_final_sg(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_atomic_paired_col_rht(torch::Tensor input,
                                                    bool return_transpose,
                                                    bool encode_centric) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(return_transpose,
                "paired column-RHT atomic producer requires return_transpose=True");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before paired atomic quantization: ",
                cudaGetErrorString(set_device_err));
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs_v3_atomic_final_sg(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        -1, -1, true);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

static void quantize_col_into_outputs_v3_opt_fixed_sign_rht(
    torch::Tensor input,
    bool encode_centric,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor col_sg
) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "paired column-RHT producer requires M and K multiples of 128");
    TORCH_CHECK(col_fp4.is_contiguous() && col_sc.is_contiguous() &&
                    col_sg.is_contiguous(),
                "paired column-RHT producer requires contiguous outputs");

    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    // Match the established v4-opt fallback contract exactly: the producer
    // writes per-128x128 chunk scales, then the ordinary v3 finalizer folds
    // them into the outer column scale.  EMIT_ROW=false removes the otherwise
    // redundant RHT row payload without changing any column arithmetic.
    auto row_sg_chunk_scratch = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg_chunk = torch::empty({K / 128, M / 128}, opts_f32);

    alignas(64) CUtensorMap tmap_in{}, tmap_out_dummy{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_dummy{}, tmap_sc_col{};
    alignas(64) CUtensorMap empty_sc_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_col_only_raw_output_tmaps(
        M, K, col_fp4, col_sc,
        tmap_out_dummy, tmap_out_t, tmap_sc_row_dummy, tmap_sc_col);

    auto work_counter = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(
        work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (encode_centric) {
        launch_localcta_quant_opt<
            true, true,
            false, false, false, false, true, true,
            false, false, false, false, false, false, false>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            empty_sc_prepared, empty_sc_prepared,
            row_sg_chunk_scratch.data_ptr<float>(),
            col_sg_chunk.data_ptr<float>(),
            nullptr, nullptr, work_counter_ptr,
            M, K, true, false,
            0, 0, nullptr, stream);
    } else {
        launch_localcta_quant_opt<
            true, false,
            false, false, false, false, true, true,
            false, false, false, false, false, false, false>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            empty_sc_prepared, empty_sc_prepared,
            row_sg_chunk_scratch.data_ptr<float>(),
            col_sg_chunk.data_ptr<float>(),
            nullptr, nullptr, work_counter_ptr,
            M, K, true, false,
            0, 0, nullptr, stream);
    }
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "paired column-RHT v4-opt producer failed: ",
                    cudaGetErrorString(err));
    }

    // Mirror finalize_quant_contract_v3's selected column route so this stays
    // byte-identical even when the split-finalize tuning knob is enabled.
    if (use_localcta_v4_split_finalize_single()) {
        const int64_t col_chunks = K / 128;
        const int64_t col_sg_rows = M / 128;
        const int64_t col_sc_rows = M / 64;
        reduce_col_sg_tiles_kernel<256><<<
            static_cast<unsigned int>((col_chunks + 1) / 2), 256, 0, stream>>>(
            col_sg_chunk.data_ptr<float>(),
            col_sg.data_ptr<float>(),
            static_cast<int>(col_chunks),
            static_cast<int>(col_sg_rows));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess,
                        "paired column-RHT SG reduction failed: ",
                        cudaGetErrorString(err));
        }
        dim3 col_rescale_grid(static_cast<unsigned int>(col_chunks),
                              static_cast<unsigned int>(col_sc_rows));
        rescale_col_sc_kernel<256><<<col_rescale_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()),
            col_sg_chunk.data_ptr<float>(),
            col_sg.data_ptr<float>(),
            static_cast<int>(col_chunks),
            static_cast<int>(col_sc_rows),
            static_cast<int>(col_sg_rows));
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "paired column-RHT scale rescale failed: ",
                    cudaGetErrorString(err));
    } else {
        finalize_col_quant_contract_v3(col_sc, col_sg_chunk, col_sg);
    }
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_final_sg_paired_col_rht(torch::Tensor input,
                                                     bool return_transpose,
                                                     bool encode_centric) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(return_transpose,
                "paired column-RHT final-SG producer requires return_transpose=True");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before paired final-SG quantization: ",
                cudaGetErrorString(set_device_err));
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), true, input.device());
    // Preserve the production legacy-final-SG row payload, but do not spend
    // bandwidth producing its unused plain column payload.
    quantize_into_outputs_v3_final_sg(
        input, false, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
    // Preserve the proven two-pass fallback column payload while skipping its
    // unused row half.  Both launches stay on the caller's current stream.
    quantize_col_into_outputs_v3_opt_fixed_sign_rht(
        input, encode_centric, col_fp4, col_sc, col_sg);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_opt(torch::Tensor input,
                                  bool return_transpose,
                                  bool encode_centric,
                                  bool data_stochastic_rounding,
                                  bool scale_stochastic_rounding,
                                  std::string rht_axes,
                                  bool with_random_sign_mask,
                                  uint64_t rng_seed,
                                  uint64_t rng_subsequence_base,
                                  std::string data_sr_axes,
                                  std::optional<torch::Tensor> persistent_rng_state = std::nullopt) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before localCTA v4 opt quantization: ",
                cudaGetErrorString(set_device_err));

    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    TORCH_CHECK(rht_axes == "none" || rht_axes == "off" || rht_axes == "0" ||
                rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
                rht_axes == "row_col" || rht_axes == "rowcol" || rht_axes == "all",
                "Unsupported localCTA v4 RHT axes: ", rht_axes);

    const auto sr_axes = resolve_localcta_data_sr_axes(
        data_stochastic_rounding, std::move(data_sr_axes), "localCTA v4");
    const bool row_data_sr = sr_axes.row;
    const bool col_data_sr = sr_axes.col;

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs_v3_opt(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        row_data_sr,
        col_data_sr,
        scale_stochastic_rounding,
        row_rht,
        col_rht && return_transpose,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base,
        persistent_rng_state.value_or(torch::Tensor()));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_final_sg_opt(torch::Tensor input,
                                           bool return_transpose,
                                           bool encode_centric,
                                           bool data_stochastic_rounding,
                                           bool scale_stochastic_rounding,
                                           std::string rht_axes,
                                           bool with_random_sign_mask,
                                           uint64_t rng_seed,
                                           uint64_t rng_subsequence_base,
                                           bool four_over_six_mae) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(!with_random_sign_mask,
                "final-SG opt producer does not support random RHT signs yet");
    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    TORCH_CHECK(rht_axes == "none" || rht_axes == "off" || rht_axes == "0" ||
                rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
                rht_axes == "row_col" || rht_axes == "rowcol" || rht_axes == "all",
                "Unsupported localCTA v4 RHT axes: ", rht_axes);
    TORCH_CHECK(
        !four_over_six_mae ||
            (!data_stochastic_rounding && !scale_stochastic_rounding &&
             !row_rht && !col_rht && !with_random_sign_mask),
        "Four-over-six MAE cannot be combined with stochastic rounding or RHT");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "final-SG opt producer requires M and K to be multiples of 128");

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(M, K, return_transpose, input.device());
    quantize_into_outputs_v3_opt_final_sg(
        input, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        col_rht && return_transpose,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base,
        four_over_six_mae);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_quantize_col_for_gemm_final_sg_opt(
    torch::Tensor input,
    bool encode_centric,
    bool four_over_six_mae
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "column-only producer requires the outer-SG contract");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "column-only producer requires M and K to be multiples of 128");

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(input.device());
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto col_fp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_sc = torch::empty({K / 128, M / 64, 512}, opts_fp8);
    auto row_sg_scratch = torch::empty({outer_sg_tiles_128(M), 1}, opts_f32);
    auto col_sg = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);

    launch_scan_single_sg_opt_direct_final(
        input,
        true,
        row_sg_scratch,
        col_sg,
        false,
        false,
        false);

    alignas(64) CUtensorMap tmap_in{}, tmap_out_dummy{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_dummy{}, tmap_sc_col{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_col_only_raw_output_tmaps(
        M, K, col_fp4, col_sc,
        tmap_out_dummy, tmap_out_t, tmap_sc_row_dummy, tmap_sc_col);

    auto work_counter = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(
        work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto *row_sg_ptr = row_sg_scratch.data_ptr<float>();
    auto *col_sg_ptr = col_sg.data_ptr<float>();
    alignas(64) CUtensorMap empty_sc_prepared{};

    if (four_over_six_mae) {
        if (encode_centric) {
            launch_localcta_quant_four_over_six_final_sg<true, true, false, false>(
                tmap_in, tmap_out_dummy, tmap_out_t,
                tmap_sc_row_dummy, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, stream);
        } else {
            launch_localcta_quant_four_over_six_final_sg<true, false, false, false>(
                tmap_in, tmap_out_dummy, tmap_out_t,
                tmap_sc_row_dummy, tmap_sc_col,
                row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
                M, K, stream);
        }
    } else if (encode_centric) {
        launch_localcta_quant_opt_dispatch<true, true, false, false, true, false>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            empty_sc_prepared, empty_sc_prepared,
            row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
            M, K, true, false,
            false, false, false, false, false,
            0, 0, stream);
    } else {
        launch_localcta_quant_opt_dispatch<true, false, false, false, true, false>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            empty_sc_prepared, empty_sc_prepared,
            row_sg_ptr, col_sg_ptr, nullptr, nullptr, work_counter_ptr,
            M, K, true, false,
            false, false, false, false, false,
            0, 0, stream);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_quantize_col_for_gemm_final_sg_opt failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(col_fp4, col_sc, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_mxfp8_row_mxfp4_col(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "fused MXFP8/MXFP4 producer requires multiples of 128");
    TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
                K <= std::numeric_limits<int>::max(),
                "fused MXFP8/MXFP4 dimensions exceed int32 range");

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(input.device());
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_u8 = torch::dtype(torch::kUInt8).device(input.device());
    auto row_mxfp8 = torch::empty({M, K}, opts_fp8);
    auto row_mxsc = torch::empty({M / 128, K / 128, 32, 16}, opts_u8);
    auto col_mxfp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_mxsc = torch::empty({K / 128, M / 128, 32, 16}, opts_u8);

    const int blocks_y = static_cast<int>(M / 128);
    const int blocks_x = static_cast<int>(K / 128);
    const int total_tiles = blocks_y * blocks_x;
    const int threads = static_cast<int>(localcta_env_int64(
        "FP4_CCE_V4_MXFP8_MXFP4_PRODUCER_THREADS", 128));
    TORCH_CHECK(threads == 128 || threads == 256,
                "MXFP8/MXFP4 producer threads must be 128 or 256");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_mxfp8_row_mxfp4_col_kernel<false, false>
        <<<total_tiles, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint8_t*>(row_mxfp8.data_ptr()),
        row_mxsc.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(col_mxfp4.data_ptr()),
        col_mxsc.data_ptr<uint8_t>(),
        static_cast<int>(M),
        static_cast<int>(K),
        blocks_x,
        blocks_y,
        1.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(row_mxfp8, row_mxsc, col_mxfp4, col_mxsc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_quantize_direct_fp8_row_mxfp4_col(
    torch::Tensor input,
    double fp8_scale) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(std::isfinite(fp8_scale) && fp8_scale > 0.0,
                "direct FP8 scale must be finite and positive");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "fused direct-FP8/MXFP4 producer requires multiples of 128");
    TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
                K <= std::numeric_limits<int>::max(),
                "fused direct-FP8/MXFP4 dimensions exceed int32 range");

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(input.device());
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_u8 = torch::dtype(torch::kUInt8).device(input.device());
    auto row_fp8 = torch::empty({M, K}, opts_fp8);
    auto col_mxfp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_mxsc = torch::empty({K / 128, M / 128, 32, 16}, opts_u8);

    const int blocks_y = static_cast<int>(M / 128);
    const int blocks_x = static_cast<int>(K / 128);
    const int total_tiles = blocks_y * blocks_x;
    const int threads = static_cast<int>(localcta_env_int64(
        "FP4_CCE_V4_DIRECT_FP8_PRODUCER_THREADS", 256));
    TORCH_CHECK(threads == 128 || threads == 256,
                "direct-FP8/MXFP4 producer threads must be 128 or 256");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_mxfp8_row_mxfp4_col_kernel<true, false>
        <<<total_tiles, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<uint8_t*>(row_fp8.data_ptr()),
            nullptr,
            reinterpret_cast<uint8_t*>(col_mxfp4.data_ptr()),
            col_mxsc.data_ptr<uint8_t>(),
            static_cast<int>(M),
            static_cast<int>(K),
            blocks_x,
            blocks_y,
            1.0f / static_cast<float>(fp8_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(row_fp8, col_mxfp4, col_mxsc);
}

template <bool DIRECT_FP8>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_rmsnorm_quantize_fp8_row_mxfp4_col_with_output_impl(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon,
    double fp8_scale) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");
    TORCH_CHECK(input.device() == gamma.device(),
                "input and gamma must be on the same CUDA device");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0,
                "epsilon must be finite and non-negative");
    if constexpr (DIRECT_FP8) {
        TORCH_CHECK(std::isfinite(fp8_scale) && fp8_scale > 0.0,
                    "direct FP8 scale must be finite and positive");
    }

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(gamma.size(0) == K, "gamma must match input K");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "fused RMSNorm FP8/MXFP4 producer requires multiples of 128");
    TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
                K <= std::numeric_limits<int>::max(),
                "fused RMSNorm FP8/MXFP4 dimensions exceed int32 range");

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(input.device());
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_u8 = torch::dtype(torch::kUInt8).device(input.device());
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto normed = torch::empty_like(input);
    auto row_fp8 = torch::empty({M, K}, opts_fp8);
    auto row_sc = DIRECT_FP8
        ? torch::empty({0}, opts_u8)
        : torch::empty({M / 128, K / 128, 32, 16}, opts_u8);
    auto col_mxfp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_mxsc = torch::empty({K / 128, M / 128, 32, 16}, opts_u8);
    auto inv_rms = torch::empty({M}, opts_f32);

    const int blocks_y = static_cast<int>(M / 128);
    const int blocks_x = static_cast<int>(K / 128);
    const int total_tiles = blocks_y * blocks_x;
    const char* threads_env = DIRECT_FP8
        ? "FP4_CCE_V4_DIRECT_FP8_PRODUCER_THREADS"
        : "FP4_CCE_V4_MXFP8_MXFP4_PRODUCER_THREADS";
    const int threads = static_cast<int>(localcta_env_int64(threads_env, 256));
    TORCH_CHECK(threads == 128 || threads == 256,
                "fused RMSNorm FP8/MXFP4 producer threads must be 128 or 256");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    localcta_compute_inv_rms_kernel<256><<<static_cast<int>(M), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            input.data_ptr<at::BFloat16>()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K));
    quantize_mxfp8_row_mxfp4_col_kernel<DIRECT_FP8, true>
        <<<total_tiles, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<uint8_t*>(row_fp8.data_ptr()),
            DIRECT_FP8 ? nullptr : row_sc.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(col_mxfp4.data_ptr()),
            col_mxsc.data_ptr<uint8_t>(),
            static_cast<int>(M),
            static_cast<int>(K),
            blocks_x,
            blocks_y,
            DIRECT_FP8 ? 1.0f / static_cast<float>(fp8_scale) : 1.0f,
            reinterpret_cast<const __nv_bfloat16*>(
                gamma.data_ptr<at::BFloat16>()),
            inv_rms.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                normed.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(
        normed, row_fp8, row_sc, col_mxfp4, col_mxsc, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_rmsnorm_quantize_mxfp8_row_mxfp4_col_with_output(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon) {
    return tk_localcta_rmsnorm_quantize_fp8_row_mxfp4_col_with_output_impl<false>(
        input, gamma, epsilon, 1.0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_rmsnorm_quantize_direct_fp8_row_mxfp4_col_with_output(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon,
    double fp8_scale) {
    return tk_localcta_rmsnorm_quantize_fp8_row_mxfp4_col_with_output_impl<true>(
        input, gamma, epsilon, fp8_scale);
}

std::tuple<torch::Tensor, torch::Tensor>
tk_localcta_quantize_mxfp8_row_only(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "MXFP8 row producer requires the outer-SG contract");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "MXFP8 row producer requires multiples of 128");

    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_u8 = torch::dtype(torch::kUInt8).device(input.device());
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_mxfp8 = torch::empty({M, K}, opts_fp8);
    auto row_mxsc = torch::empty({M / 128, K / 128, 32, 16}, opts_u8);
    // Keep the exact deployed first-stage launch, including its SG scratch
    // shapes, while deliberately omitting the second-stage NVFP4 column pass.
    auto row_sg_scratch = torch::empty({outer_sg_tiles_128(M), 1}, opts_f32);
    auto col_sg_scratch = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);

    launch_scan_single_sg_mxfp8_direct_final(
        input, row_mxfp8, row_mxsc, row_sg_scratch, col_sg_scratch);
    return std::make_tuple(row_mxfp8, row_mxsc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_rmsnorm_quantize_mxfp8_row_with_output(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");
    TORCH_CHECK(input.device() == gamma.device(),
                "input and gamma must be on the same CUDA device");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "RMSNorm MXFP8 row producer requires the outer-SG contract");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0,
                "epsilon must be finite and non-negative");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(gamma.size(0) == K, "gamma must match input K");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "RMSNorm MXFP8 row producer requires multiples of 128");

    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_u8 = torch::dtype(torch::kUInt8).device(input.device());
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto normed = torch::empty_like(input);
    auto row_mxfp8 = torch::empty({M, K}, opts_fp8);
    auto row_mxsc = torch::empty({M / 128, K / 128, 32, 16}, opts_u8);
    auto row_sg_scratch = torch::empty({outer_sg_tiles_128(M), 1}, opts_f32);
    auto col_sg_scratch = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);
    auto inv_rms = torch::empty({M}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    localcta_compute_inv_rms_kernel<256><<<static_cast<int>(M), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K));
    launch_scan_single_sg_mxfp8_rmsnorm_direct_final(
        input, gamma, inv_rms, normed,
        row_mxfp8, row_mxsc, row_sg_scratch, col_sg_scratch);
    return std::make_tuple(normed, row_mxfp8, row_mxsc, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_quantize_mxfp8_row_nvfp4_col_final_sg_opt(
    torch::Tensor input,
    bool encode_centric,
    bool four_over_six_mae,
    bool col_data_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "fused MXFP8/NVFP4 producer requires the outer-SG contract");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "fused MXFP8/NVFP4 producer requires multiples of 128");

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(input.device());
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_u8 = torch::dtype(torch::kUInt8).device(input.device());
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_mxfp8 = torch::empty({M, K}, opts_fp8);
    auto row_mxsc = torch::empty({M / 128, K / 128, 32, 16}, opts_u8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_sc = torch::empty({K / 128, M / 64, 512}, opts_fp8);
    auto row_sg_scratch = torch::empty({outer_sg_tiles_128(M), 1}, opts_f32);
    auto col_sg = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);

    launch_scan_single_sg_mxfp8_direct_final(
        input, row_mxfp8, row_mxsc, row_sg_scratch, col_sg);

    alignas(64) CUtensorMap tmap_in{}, tmap_out_dummy{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_dummy{}, tmap_sc_col{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_col_only_raw_output_tmaps(
        M, K, col_fp4, col_sc,
        tmap_out_dummy, tmap_out_t, tmap_sc_row_dummy, tmap_sc_col);

    auto work_counter = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(
        work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto *row_sg_ptr = row_sg_scratch.data_ptr<float>();
    auto *col_sg_ptr = col_sg.data_ptr<float>();
    if (encode_centric) {
        launch_mxfp8_row_nvfp4_col_final_sg<true>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K,
            four_over_six_mae, col_data_stochastic_rounding,
            rng_seed, rng_subsequence_base, stream);
    } else {
        launch_mxfp8_row_nvfp4_col_final_sg<false>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K,
            four_over_six_mae, col_data_stochastic_rounding,
            rng_seed, rng_subsequence_base, stream);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_quantize_mxfp8_row_nvfp4_col_final_sg_opt failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_mxfp8, row_mxsc, col_fp4, col_sc, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_rmsnorm_quantize_mxfp8_row_nvfp4_col_with_output_final_sg_opt(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon,
    bool encode_centric,
    bool four_over_six_mae,
    bool col_data_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "fused RMSNorm MXFP8/NVFP4 producer requires the outer-SG contract");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0,
                "epsilon must be finite and non-negative");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(gamma.size(0) == K, "gamma must match input K");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "fused RMSNorm MXFP8/NVFP4 producer requires multiples of 128");

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(input.device());
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input.device());
    auto opts_u8 = torch::dtype(torch::kUInt8).device(input.device());
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto normed = torch::empty_like(input);
    auto row_mxfp8 = torch::empty({M, K}, opts_fp8);
    auto row_mxsc = torch::empty({M / 128, K / 128, 32, 16}, opts_u8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_sc = torch::empty({K / 128, M / 64, 512}, opts_fp8);
    auto row_sg_scratch = torch::empty({outer_sg_tiles_128(M), 1}, opts_f32);
    auto col_sg = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);
    auto inv_rms = torch::empty({M}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    localcta_compute_inv_rms_kernel<256><<<static_cast<int>(M), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K));

    launch_scan_single_sg_mxfp8_rmsnorm_direct_final(
        input, gamma, inv_rms, normed,
        row_mxfp8, row_mxsc, row_sg_scratch, col_sg);

    alignas(64) CUtensorMap tmap_in{}, tmap_out_dummy{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_dummy{}, tmap_sc_col{};
    create_tma_2d(tmap_in, normed.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_col_only_raw_output_tmaps(
        M, K, col_fp4, col_sc,
        tmap_out_dummy, tmap_out_t, tmap_sc_row_dummy, tmap_sc_col);

    auto work_counter = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(
        work_counter.data_ptr<int>());
    auto *row_sg_ptr = row_sg_scratch.data_ptr<float>();
    auto *col_sg_ptr = col_sg.data_ptr<float>();
    if (encode_centric) {
        launch_mxfp8_row_nvfp4_col_final_sg<true>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K,
            four_over_six_mae, col_data_stochastic_rounding,
            rng_seed, rng_subsequence_base, stream);
    } else {
        launch_mxfp8_row_nvfp4_col_final_sg<false>(
            tmap_in, tmap_out_dummy, tmap_out_t,
            tmap_sc_row_dummy, tmap_sc_col,
            row_sg_ptr, col_sg_ptr, work_counter_ptr, M, K,
            four_over_six_mae, col_data_stochastic_rounding,
            rng_seed, rng_subsequence_base, stream);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "tk_localcta_rmsnorm_quantize_mxfp8_row_nvfp4_col_with_output_final_sg_opt failed: ",
        cudaGetErrorString(err));
    return std::make_tuple(
        normed, row_mxfp8, row_mxsc, col_fp4, col_sc, col_sg, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_localcta_rmsnorm_quantize_for_gemm_opt(torch::Tensor input,
                                          torch::Tensor gamma,
                                          double epsilon,
                                          bool return_transpose,
                                          bool encode_centric,
                                          bool data_stochastic_rounding,
                                          bool scale_stochastic_rounding,
                                          std::string rht_axes,
                                          bool with_random_sign_mask,
                                          uint64_t rng_seed,
                                          uint64_t rng_subsequence_base) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma must match input K");

    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    TORCH_CHECK(rht_axes == "none" || rht_axes == "off" || rht_axes == "0" ||
                rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
                rht_axes == "row_col" || rht_axes == "rowcol" || rht_axes == "all",
                "Unsupported localCTA v4 RHT axes: ", rht_axes);

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto inv_rms = torch::empty({M}, opts_f32);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    localcta_compute_inv_rms_kernel<256><<<static_cast<int>(M), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K));

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(M, K, return_transpose, input.device());
    quantize_rmsnorm_into_outputs_v3_opt(
        input, gamma, inv_rms, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        col_rht && return_transpose,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_rmsnorm_quantize_for_gemm_opt failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt(torch::Tensor input,
                                                   torch::Tensor gamma,
                                                   double epsilon,
                                                   bool return_transpose,
                                                   bool encode_centric,
                                                   bool data_stochastic_rounding,
                                                   bool scale_stochastic_rounding,
                                                   std::string rht_axes,
                                                   bool with_random_sign_mask,
                                                   uint64_t rng_seed,
                                                   uint64_t rng_subsequence_base,
                                                   bool four_over_six_mae) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma must match input K");
    TORCH_CHECK(gamma.device() == input.device(),
                "gamma must be on the same device as input");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before localCTA RMSNorm final-SG quantization: ",
                cudaGetErrorString(set_device_err));
    TORCH_CHECK(!with_random_sign_mask,
                "final-SG RMSNorm producer does not support random RHT signs yet");

    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    TORCH_CHECK(rht_axes == "none" || rht_axes == "off" || rht_axes == "0" ||
                rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
                rht_axes == "row_col" || rht_axes == "rowcol" || rht_axes == "all",
                "Unsupported localCTA v4 RHT axes: ", rht_axes);
    TORCH_CHECK(
        !four_over_six_mae ||
            (!data_stochastic_rounding && !scale_stochastic_rounding &&
             !row_rht && !col_rht && !with_random_sign_mask),
        "Four-over-six MAE cannot be combined with stochastic rounding or RHT");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(input.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    localcta_compute_inv_rms_kernel<256><<<static_cast<int>(M), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K));

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(M, K, return_transpose, input.device());
    quantize_rmsnorm_into_outputs_v3_opt_final_sg(
        input, gamma, inv_rms, return_transpose, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        col_rht && return_transpose,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base,
        false,
        four_over_six_mae);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_localcta_rmsnorm_quantize_from_row_rms_partial_final_sg(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor row_rms_partial,
    double epsilon,
    bool return_transpose,
    bool encode_centric) {
    TORCH_CHECK(
        get_v3_contract_mode() != V3ContractMode::TileGrid256,
        "localCTA exact C/D/E requires the v4 outer-SG contract");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                    input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be contiguous CUDA bf16 [M,K]");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous() &&
                    gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be contiguous CUDA bf16 [K]");
    TORCH_CHECK(row_rms_partial.is_cuda() && row_rms_partial.is_contiguous() &&
                    row_rms_partial.scalar_type() == torch::kFloat32 &&
                    row_rms_partial.dim() == 2,
                "row_rms_partial must be contiguous CUDA float32 [M,K/256]");
    TORCH_CHECK(input.device() == gamma.device() &&
                    input.device() == row_rms_partial.device(),
                "input, gamma, and row_rms_partial must share one CUDA device");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(return_transpose,
                "localCTA exact C/D/E production quantization requires transpose output");
    TORCH_CHECK(M % 256 == 0 && K % 256 == 0,
                "localCTA exact C/D/E requires M and K multiples of 256");
    TORCH_CHECK(K <= 4096,
                "localCTA exact C/D/E row reducer supports K <= 4096");
    TORCH_CHECK(gamma.size(0) == K, "gamma must match input K");
    TORCH_CHECK(row_rms_partial.size(0) == M &&
                    row_rms_partial.size(1) == K / 256,
                "row_rms_partial shape must be [M,K/256]");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0,
                "epsilon must be finite and non-negative");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto inv_rms = torch::empty(
        {M}, torch::dtype(torch::kFloat32).device(input.device()));
    c1_rms_reduce::row_rms_reduce_entrypoint(
        row_rms_partial, inv_rms, K, epsilon);

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(M, K, true, input.device());
    quantize_rmsnorm_into_outputs_v3_opt_final_sg(
        input, gamma, inv_rms, true, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
        false, false, false, false, false, 0, 0, true);

    auto err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "tk_localcta_rmsnorm_quantize_from_row_rms_partial_final_sg failed: ",
        cudaGetErrorString(err));
    return std::make_tuple(
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer(torch::Tensor input,
                                                             torch::Tensor gamma,
                                                             double epsilon,
                                                             bool return_transpose,
                                                             bool encode_centric,
                                                             bool data_stochastic_rounding,
                                                             bool scale_stochastic_rounding,
                                                             std::string rht_axes,
                                                             bool with_random_sign_mask,
                                                             uint64_t rng_seed,
                                                             uint64_t rng_subsequence_base) {
    TORCH_CHECK(return_transpose,
                "RMSNorm row-prepared/col-outer producer requires return_transpose=true");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "RMSNorm row-prepared/col-outer producer requires the v4 outer-SG contract");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be bf16 [K]");

    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    TORCH_CHECK(rht_axes == "none" || rht_axes == "off" || rht_axes == "0" ||
                rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
                rht_axes == "row_col" || rht_axes == "rowcol" || rht_axes == "all",
                "Unsupported localCTA v4 RHT axes: ", rht_axes);

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(gamma.size(0) == K, "gamma must match input K");
    TORCH_CHECK(M % 256 == 0 && K % 256 == 0,
                "RMSNorm row-prepared/col-outer producer requires M and K multiples of 256");

    auto device = input.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto inv_rms = torch::empty({M}, opts_f32);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    localcta_compute_inv_rms_kernel<256><<<static_cast<int>(M), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K));

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc_prepared = torch::empty({M / 128, K / 64, 512}, opts_fp8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_sc_raw = torch::empty({K / 128, M / 64, 512}, opts_fp8);
    auto row_sg_chunks = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg_chunks = torch::empty({K / 128, M / 128}, opts_f32);
    auto col_sg_outer = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);

    auto row_sc_raw_scratch = torch::empty({M / 128, K / 64, 512}, opts_fp8);
    auto col_sc_prepared_scratch = torch::empty({K / 128, M / 64, 512}, opts_fp8);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row_raw{}, tmap_sc_col_raw{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    alignas(64) CUtensorMap tmap_out_prepared{}, tmap_out_t_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_raw_output_tmaps<true>(
        row_fp4, row_sc_raw_scratch, col_fp4, col_sc_raw,
        tmap_out, tmap_out_t, tmap_sc_row_raw, tmap_sc_col_raw);
    create_prepared_output_tmaps<true>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared_scratch,
        tmap_out_prepared, tmap_out_t_prepared, tmap_sc_row_prepared, tmap_sc_col_prepared);

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    const auto* gamma_ptr = reinterpret_cast<const tk_localcta::IType*>(gamma.data_ptr());
    auto *inv_rms_ptr = inv_rms.data_ptr<float>();
    if (encode_centric) {
        launch_localcta_quant_opt_dispatch<true, true, true, false>(
            tmap_in, tmap_out, tmap_out_t,
            tmap_sc_row_raw, tmap_sc_col_raw,
            tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_chunks.data_ptr<float>(), col_sg_chunks.data_ptr<float>(),
            gamma_ptr, inv_rms_ptr, work_counter_ptr,
            M, K, true, true,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            row_rht,
            col_rht,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
            stream);
    } else {
        launch_localcta_quant_opt_dispatch<true, false, true, false>(
            tmap_in, tmap_out, tmap_out_t,
            tmap_sc_row_raw, tmap_sc_col_raw,
            tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_chunks.data_ptr<float>(), col_sg_chunks.data_ptr<float>(),
            gamma_ptr, inv_rms_ptr, work_counter_ptr,
            M, K, true, true,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            row_rht,
            col_rht,
            with_random_sign_mask,
            rng_seed,
            rng_subsequence_base,
            stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer failed: ",
                cudaGetErrorString(err));
    finalize_col_quant_contract_v3(col_sc_raw, col_sg_chunks, col_sg_outer);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer finalize failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc_prepared, row_sg_chunks,
                           col_fp4, col_sc_raw, col_sg_outer, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_nhsd_wo_for_gemm(torch::Tensor input,
                                      bool encode_centric) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4,
                "input must be bf16 [B, H, S, D]");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before localCTA NHSD WO quantization: ",
                cudaGetErrorString(set_device_err));

    const int64_t B = input.size(0);
    const int64_t H = input.size(1);
    const int64_t S = input.size(2);
    const int64_t D = input.size(3);
    const int64_t M = B * S;
    const int64_t K = H * D;
    TORCH_CHECK(D == 64,
                "NHSD WO direct quantization currently requires head_dim == 64");
    TORCH_CHECK(S % 128 == 0,
                "NHSD WO direct quantization requires sequence length to be a multiple of 128");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "logical [B*S, H*D] dimensions must be multiples of 128");

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(M, K, true, input.device());

    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_sg_chunk = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg_chunk = torch::empty({K / 128, M / 128}, opts_f32);

    launch_localcta_quantize_nhsd_wo_raw(
        input, encode_centric,
        row_fp4, row_sc, col_fp4, col_sc,
        row_sg_chunk, col_sg_chunk);

    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        TORCH_CHECK(M % 256 == 0 && K % 256 == 0,
                    "localCTA v3 tilegrid256 contract requires M and K to be multiples of 256");
        fold_quant_scales_v3_tilegrid256(row_sc, row_sg_chunk, col_sc, col_sg_chunk);
        finalize_quant_contract_v3_tilegrid256(row_sg_chunk, row_sg, col_sg);
    } else if (use_localcta_v4_nhsd_reduced_warp_finalize()) {
        finalize_quant_contract_v3_reduced_warp(
            row_sc, row_sg_chunk, row_sg,
            col_sc, col_sg_chunk, col_sg);
    } else {
        finalize_quant_contract_v3(row_sc, row_sg_chunk, row_sg, col_sc, col_sg_chunk, col_sg);
    }

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm(torch::Tensor input,
                               bool return_transpose,
                               bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs_v3(input, return_transpose, encode_centric,
                             row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                             torch::Tensor(), torch::Tensor(), true);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_raw(torch::Tensor input,
                                  bool return_transpose,
                                  bool encode_centric) {
    const bool use_2cta = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    return tk_localcta_quantize_for_gemm_chunkgrid_internal(
        input, return_transpose, encode_centric, use_2cta
    );
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_fast(torch::Tensor input,
                                   bool return_transpose,
                                   bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, row_sc_prepared, col_sc_prepared] =
        allocate_quant_outputs_fast(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_row_prepared_col_outer(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric
) {
    TORCH_CHECK(return_transpose,
                "row-prepared/col-outer localCTA quantizer requires return_transpose=true");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "row-prepared/col-outer localCTA quantizer requires the v4 outer-SG contract");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(input.size(0) % 256 == 0 && input.size(1) % 256 == 0,
                "row-prepared/col-outer localCTA quantizer requires M and K multiples of 256");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    auto device = input.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc_prepared = torch::empty({M / 128, K / 64, 512}, opts_fp8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_sc_raw = torch::empty({K / 128, M / 64, 512}, opts_fp8);
    auto row_sg_chunks = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg_chunks = torch::empty({K / 128, M / 128}, opts_f32);
    auto col_sg_outer = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);

    launch_localcta_tma_quant_row_prepared_col_outer<true>(
        input,
        row_fp4, row_sc_prepared,
        col_fp4, col_sc_raw,
        row_sg_chunks, col_sg_chunks,
        encode_centric);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_quantize_for_gemm_row_prepared_col_outer failed: ",
                cudaGetErrorString(err));
    finalize_col_quant_contract_v3(col_sc_raw, col_sg_chunks, col_sg_outer);

    return std::make_tuple(row_fp4, row_sc_prepared, row_sg_chunks,
                           col_fp4, col_sc_raw, col_sg_outer);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_raw_outer_tma(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric
) {
    TORCH_CHECK(return_transpose,
                "raw-outer TMA localCTA quantizer requires return_transpose=true");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "raw-outer TMA localCTA quantizer requires the v4 outer-SG contract");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(input.size(0) % 256 == 0 && input.size(1) % 256 == 0,
                "raw-outer TMA localCTA quantizer requires M and K multiples of 256");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(M, K, true, input.device());

    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto row_sg_chunks = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg_chunks = torch::empty({K / 128, M / 128}, opts_f32);

    launch_localcta_tma_quant_raw_outer<true>(
        input,
        row_fp4, row_sc,
        col_fp4, col_sc,
        row_sg_chunks, col_sg_chunks,
        encode_centric);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_quantize_for_gemm_raw_outer_tma failed: ",
                cudaGetErrorString(err));

    finalize_row_quant_contract_v3(row_sc, row_sg_chunks, row_sg);
    finalize_col_quant_contract_v3(col_sc, col_sg_chunks, col_sg);

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm_fast(torch::Tensor input,
                                    bool return_transpose,
                                    bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, row_sc_prepared, col_sc_prepared] =
        allocate_quant_outputs_fast(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, true);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

static void finalize_weight_2d_common_outer_scale(
    torch::Tensor row_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sc,
    torch::Tensor col_sg
) {
    TORCH_CHECK(row_sc.is_contiguous() && row_sg.is_contiguous() &&
                    col_sc.is_contiguous() && col_sg.is_contiguous(),
                "localCTA 2D weight scale outputs must be contiguous");
    TORCH_CHECK(row_sg.dim() == 2 && col_sg.dim() == 2 &&
                    row_sg.size(0) == col_sg.size(1) &&
                    row_sg.size(1) == col_sg.size(0),
                "localCTA 2D weight row/column SG grids must be transposes");
    TORCH_CHECK(row_sc.dim() == 3 && col_sc.dim() == 3 &&
                    row_sc.size(0) == row_sg.size(0) &&
                    row_sc.size(1) == 2 * row_sg.size(1) &&
                    row_sc.size(2) == 512 &&
                    col_sc.size(0) == col_sg.size(0) &&
                    col_sc.size(1) == 2 * col_sg.size(1) &&
                    col_sc.size(2) == 512,
                "localCTA 2D weight prepared-scale/SG geometry mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();

    // All source loads complete before thread zero publishes the reduction,
    // so row_sg[0] can safely double as the one-element FP32 scratch. Row
    // normalization reads the still-intact transposed grid from col_sg.
    reduce_weight_2d_common_outer_sg_kernel<256><<<1, 256, 0, stream>>>(
        row_sg.data_ptr<float>(),
        row_sg.numel());
    {
        const cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "localCTA 2D weight common outer-SG reduction failed: ",
                    cudaGetErrorString(err));
    }

    const int row_chunks = static_cast<int>(row_sg.size(0));
    const int row_sc_cols = static_cast<int>(row_sc.size(1));
    dim3 row_grid(static_cast<unsigned int>(row_chunks),
                  static_cast<unsigned int>(row_sc_cols));
    rescale_weight_2d_row_sc_kernel<256><<<row_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(row_sc.data_ptr()),
        col_sg.data_ptr<float>(),
        row_sg.data_ptr<float>(),
        row_chunks,
        row_sc_cols);
    {
        const cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "localCTA 2D weight row-scale normalization failed: ",
                    cudaGetErrorString(err));
    }

    const int col_chunks = static_cast<int>(col_sg.size(0));
    const int col_sg_rows = static_cast<int>(col_sg.size(1));
    const int col_sc_rows = static_cast<int>(col_sc.size(1));
    dim3 col_grid(static_cast<unsigned int>(col_chunks),
                  static_cast<unsigned int>(col_sc_rows));
    rescale_weight_2d_col_sc_kernel<256><<<col_grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(col_sc.data_ptr()),
        col_sg.data_ptr<float>(),
        row_sg.data_ptr<float>(),
        col_chunks,
        col_sc_rows,
        col_sg_rows);
    {
        const cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "localCTA 2D weight column-scale normalization failed: ",
                    cudaGetErrorString(err));
    }

    fill_weight_2d_common_outer_sg_kernel<256><<<1, 256, 0, stream>>>(
        row_sg.data_ptr<float>(), row_sg.numel(),
        col_sg.data_ptr<float>(), col_sg.numel());
    {
        const cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "localCTA 2D weight common outer-SG broadcast failed: ",
                    cudaGetErrorString(err));
    }
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_weight_2d(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before localCTA 2D weight quantization: ",
                cudaGetErrorString(set_device_err));
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "2D localCTA weights require M and K to be multiples of 128");

    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
          row_sg, col_sg] =
        allocate_quant_outputs_prepared(M, K, true, input.device());

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K,
                  tk_localcta::BUFF_DIM_Y, tk_localcta::BUFF_DIM_X, K, 16);
    create_raw_output_tmaps<true>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col);
    create_prepared_output_tmaps<true>(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared);

    auto work_counter = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(input.device()));
    auto *work_counter_ptr =
        reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_localcta_quant<true, true, true>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg.data_ptr<float>(), col_sg.data_ptr<float>(),
        work_counter_ptr, M, K, false, true, stream);

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_quantize_weight_2d failed: ",
                cudaGetErrorString(err));

    // Keep the absolute weight magnitude in FP32. The producer emits one
    // shared FP4 payload with chunk-normalized E4M3 scales for both physical
    // orientations; normalize every chunk to a single representable common
    // outer scale without changing that shared-payload/orientation contract.
    finalize_weight_2d_common_outer_scale(
        row_sc_prepared, row_sg, col_sc_prepared, col_sg);
    return std::make_tuple(
        row_fp4, row_sc_prepared, col_fp4, col_sc_prepared,
        row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_prepared(torch::Tensor input,
                                       bool return_transpose,
                                       bool encode_centric) {
    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(input.size(0), input.size(1), return_transpose, input.device());
    const bool use_2cta_prepared = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, use_2cta_prepared);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm_prepared(torch::Tensor input,
                                        bool return_transpose,
                                        bool encode_centric) {
    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, true);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

static bool localcta_prepared_can_try_borrow(const torch::Tensor& input) {
    if (!input.is_cuda() || input.scalar_type() != torch::kBFloat16 || input.dim() != 2) {
        return false;
    }
    if (input.size(0) % 128 != 0 || input.size(1) % 128 != 0) {
        return false;
    }
    if (input.stride(1) != 1 || input.stride(0) != input.size(1)) {
        return false;
    }
    const auto addr = reinterpret_cast<uintptr_t>(input.data_ptr());
    return (addr % 16) == 0;
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_prepared_maybe_borrow(
    torch::Tensor input,
    torch::Tensor staging_input,
    bool return_transpose,
    bool encode_centric
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    TORCH_CHECK(staging_input.is_cuda() && staging_input.is_contiguous(),
                "staging_input must be contiguous CUDA tensor");
    TORCH_CHECK(staging_input.scalar_type() == torch::kBFloat16 && staging_input.dim() == 2,
                "staging_input must be bf16 [M, K]");
    TORCH_CHECK(staging_input.sizes() == input.sizes(),
                "staging_input must match input shape");
    TORCH_CHECK(staging_input.device() == input.device(),
                "staging_input must be on the same device as input");

    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(input.size(0), input.size(1), return_transpose, input.device());
    const bool use_2cta_prepared = should_use_localcta2_prepared_auto(input.size(0), input.size(1));

    if (localcta_prepared_can_try_borrow(input)) {
        try {
            alignas(64) CUtensorMap tmap_probe{};
            create_tma_2d(
                tmap_probe,
                input.data_ptr(),
                input.size(0),
                input.size(1),
                tk_localcta::BUFF_DIM_Y,
                tk_localcta::BUFF_DIM_X,
                input.size(1),
                16
            );
            quantize_into_outputs(input, return_transpose, encode_centric,
                                  row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                                  row_sc_prepared, col_sc_prepared, use_2cta_prepared);
            return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
        } catch (const c10::Error&) {
        }
    }

    staging_input.copy_(input);
    quantize_into_outputs(staging_input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, use_2cta_prepared);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    return allocate_quant_outputs_v3(M, K, return_transpose, device);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_fast_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    return allocate_quant_outputs_fast(M, K, return_transpose, device);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_prepared_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    return allocate_quant_outputs_prepared(M, K, return_transpose, device);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_launch(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    const bool use_2cta = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    quantize_into_outputs_v3(input, return_transpose, encode_centric,
                             row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                             torch::Tensor(), torch::Tensor(), use_2cta);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_fast_launch(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_sc_prepared
) {
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_prepared_launch(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    const bool use_2cta_prepared = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, use_2cta_prepared);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    int64_t total_rows = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 256 == 0, "v3 split rows must be multiples of 256");
        total_rows += rows_i;
    }
    TORCH_CHECK(total_rows == input.size(0), "split rows must sum to input.size(0)");

    auto [row_fp4_cat, row_sc_cat, col_fp4_full, col_sc_full, row_sg_cat, col_sg_cat] =
        tk_localcta_quantize_for_gemm(input, true, true);

    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_sg_list;
    col_fp4_list.reserve(split_sections.size());
    col_sc_list.reserve(split_sections.size());
    row_sg_parts.reserve(split_sections.size());
    col_sg_list.reserve(split_sections.size());

    int64_t row_offset = 0;
    int64_t row_tile_offset = 0;
    for (int64_t rows_i : split_sections) {
        col_fp4_list.push_back(
            col_fp4_full.narrow(1, row_offset / 2, rows_i / 2)
            .view(torch::kUInt8)
            .contiguous()
            .view(torch::kFloat4_e2m1fn_x2)
        );
        col_sc_list.push_back(col_sc_full.narrow(1, row_offset / 64, rows_i / 64).contiguous());
        row_sg_parts.push_back(row_sg_cat.narrow(0, row_tile_offset, rows_i / 256).contiguous());
        if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
            col_sg_list.push_back(
                col_sg_cat.narrow(1, row_tile_offset, rows_i / 256)
                .transpose(0, 1)
                .contiguous()
            );
        } else {
            col_sg_list.push_back(col_sg_cat);
        }
        row_offset += rows_i;
        row_tile_offset += rows_i / 256;
    }

    return std::make_tuple(row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_list, col_sc_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           col_fp4_full, col_sc_full);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_inputs_for_gemm_final_sg(
    const std::vector<torch::Tensor>& inputs
) {
    TORCH_CHECK(!inputs.empty(), "grouped final-SG quantize requires at least one input");
    const auto& first = inputs[0];
    TORCH_CHECK(first.dim() == 2 && first.is_cuda() && first.is_contiguous(),
                "inputs must be contiguous CUDA [N_i, K]");
    TORCH_CHECK(first.scalar_type() == torch::kBFloat16, "inputs must be bf16");
    const int64_t K = first.size(1);
    auto device = first.device();

    int64_t total_rows = 0;
    std::vector<int64_t> split_sections;
    split_sections.reserve(inputs.size());
    for (const auto& input_i : inputs) {
        TORCH_CHECK(input_i.dim() == 2 && input_i.is_cuda() && input_i.is_contiguous(),
                    "inputs must be contiguous CUDA [N_i, K]");
        TORCH_CHECK(input_i.scalar_type() == torch::kBFloat16, "inputs must be bf16");
        TORCH_CHECK(input_i.device() == device, "all inputs must be on the same device");
        TORCH_CHECK(input_i.size(1) == K, "all inputs must have the same K dimension");
        const int64_t rows_i = input_i.size(0);
        TORCH_CHECK(rows_i % 256 == 0, "v4 split rows must be multiples of 256");
        total_rows += rows_i;
        split_sections.push_back(rows_i);
    }

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({total_rows, K / 2}, opts_fp4);
    auto row_sc_cat = torch::empty({total_rows / 128, K / 64, 512}, opts_fp8);
    auto row_sg_cat = torch::empty({total_rows / 256, 1}, opts_f32);

    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_sg_list;
    col_fp4_list.reserve(inputs.size());
    col_sc_list.reserve(inputs.size());
    row_sg_parts.reserve(inputs.size());
    col_sg_list.reserve(inputs.size());

    int64_t row_offset = 0;
    int64_t row_tile_offset = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input_i = inputs[i];
        const int64_t rows_i = split_sections[i];
        auto row_fp4_view = row_fp4_cat.narrow(0, row_offset, rows_i);
        auto row_sc_view = row_sc_cat.narrow(0, row_offset / 128, rows_i / 128);
        auto row_sg_view = row_sg_cat.narrow(0, row_tile_offset, rows_i / 256);
        auto col_fp4 = torch::empty({K, rows_i / 2}, opts_fp4);
        auto col_sc = torch::empty({K / 128, rows_i / 64, 512}, opts_fp8);
        auto col_sg = torch::empty({1, outer_sg_tiles_128(K)}, opts_f32);

        quantize_into_outputs_v3_final_sg(
            input_i, true, true,
            row_fp4_view, row_sc_view, col_fp4, col_sc, row_sg_view, col_sg);

        col_fp4_list.push_back(col_fp4);
        col_sc_list.push_back(col_sc);
        row_sg_parts.push_back(row_sg_view);
        col_sg_list.push_back(col_sg);
        row_offset += rows_i;
        row_tile_offset += rows_i / 256;
    }

    auto col_sg_cat = torch::cat(col_sg_list, 0);
    auto col_fp4_full = torch::empty({0}, opts_fp4);
    auto col_sc_full = torch::empty({0}, opts_fp8);

    return std::make_tuple(row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_list, col_sc_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           col_fp4_full, col_sc_full);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_for_gemm_final_sg(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    int64_t total_rows = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 256 == 0, "v4 split rows must be multiples of 256");
        total_rows += rows_i;
    }
    TORCH_CHECK(total_rows == input.size(0), "split rows must sum to input.size(0)");

    auto [row_fp4_cat, row_sc_cat, col_fp4_full, col_sc_full, row_sg_cat, col_sg_cat] =
        tk_localcta_quantize_for_gemm_final_sg(input, true, true);

    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_sg_list;
    col_fp4_list.reserve(split_sections.size());
    col_sc_list.reserve(split_sections.size());
    row_sg_parts.reserve(split_sections.size());
    col_sg_list.reserve(split_sections.size());

    int64_t row_offset = 0;
    int64_t row_tile_offset = 0;
    const bool use_view_splits =
        use_localcta_v4_final_sg_view_splits() &&
        localcta_env_flag("USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD", false);
    for (int64_t rows_i : split_sections) {
        auto col_fp4_view = col_fp4_full.narrow(1, row_offset / 2, rows_i / 2);
        auto col_sc_view = col_sc_full.narrow(1, row_offset / 64, rows_i / 64);
        auto row_sg_view = row_sg_cat.narrow(0, row_tile_offset, rows_i / 256);
        if (use_view_splits) {
            col_fp4_list.push_back(col_fp4_view);
            col_sc_list.push_back(col_sc_view);
            row_sg_parts.push_back(row_sg_view);
        } else {
            col_fp4_list.push_back(
                col_fp4_view
                .view(torch::kUInt8)
                .contiguous()
                .view(torch::kFloat4_e2m1fn_x2)
            );
            col_sc_list.push_back(col_sc_view.contiguous());
            row_sg_parts.push_back(row_sg_view.contiguous());
        }
        col_sg_list.push_back(col_sg_cat);
        row_offset += rows_i;
        row_tile_offset += rows_i / 256;
    }

    return std::make_tuple(row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_list, col_sc_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           col_fp4_full, col_sc_full);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_split2_for_gemm_final_sg(
    torch::Tensor input0,
    torch::Tensor input1
) {
    return tk_localcta_group_quantize_inputs_for_gemm_final_sg({input0, input1});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_for_gemm_raw(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    int64_t total_rows = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 256 == 0, "strict v4 split rows must be multiples of 256");
        total_rows += rows_i;
    }
    TORCH_CHECK(total_rows == input.size(0), "split rows must sum to input.size(0)");

    auto [row_fp4_full, row_sc_full, col_fp4_full, col_sc_full, row_sg_full, col_sg_full] =
        tk_localcta_quantize_for_gemm_raw(input, true, true);

    auto row_sg_tiles = torch::empty(
        {outer_sg_tiles_128(input.size(0)), 1},
        torch::dtype(torch::kFloat32).device(input.device())
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    reduce_row_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(input.size(0))), 256, 0, stream>>>(
        row_sg_full.data_ptr<float>(),
        row_sg_tiles.data_ptr<float>(),
        static_cast<int>(input.size(0) / 128),
        static_cast<int>(input.size(1) / 128)
    );
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_tiles_kernel failed: ", cudaGetErrorString(err));
    }

    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_sg_list;
    col_fp4_list.reserve(split_sections.size());
    col_sc_list.reserve(split_sections.size());
    row_sg_parts.reserve(split_sections.size());
    col_sg_list.reserve(split_sections.size());

    int64_t row_offset = 0;
    int64_t row_tile128_offset = 0;
    int64_t row_tile256_offset = 0;
    for (int64_t rows_i : split_sections) {
        col_fp4_list.push_back(
            col_fp4_full.narrow(1, row_offset / 2, rows_i / 2)
            .view(torch::kUInt8)
            .contiguous()
            .view(torch::kFloat4_e2m1fn_x2)
        );
        col_sc_list.push_back(
            col_sc_full.narrow(1, row_offset / 64, rows_i / 64).contiguous()
        );
        row_sg_parts.push_back(
            row_sg_tiles.narrow(0, row_tile256_offset, rows_i / 256).contiguous()
        );
        col_sg_list.push_back(
            col_sg_full.narrow(1, row_tile128_offset, rows_i / 128).contiguous()
        );
        row_offset += rows_i;
        row_tile128_offset += rows_i / 128;
        row_tile256_offset += rows_i / 256;
    }

    return std::make_tuple(row_fp4_full, row_sc_full, row_sg_tiles,
                           col_fp4_list, col_sc_list, col_sg_full,
                           row_sg_parts, col_sg_list,
                           col_fp4_full, col_sc_full);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>>
tk_localcta_group_quantize_for_gemm_fast(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_parts;
    std::vector<torch::Tensor> row_sc_parts;
    std::vector<torch::Tensor> row_sc_prepared_parts;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t row_offset = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 128 == 0, "split rows must be multiples of 128");
        auto chunk = input.narrow(0, row_offset, rows_i);
        auto [rf, rs, cf, cs, rsg, csg, rsp, csp] =
            tk_localcta_quantize_for_gemm_fast(chunk, true, true);
        row_fp4_parts.push_back(rf);
        row_sc_parts.push_back(rs);
        row_sc_prepared_parts.push_back(rsp);
        row_sg_parts.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_list.push_back(cs);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        row_offset += rows_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_parts, 0);
    auto row_sc_cat = torch::cat(row_sc_parts, 0);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_parts, 0);
    auto row_sg_cat = torch::cat(row_sg_parts, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_list, col_sc_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           row_sc_prepared_cat, col_sc_prepared_list);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    const int64_t total_rows = input.size(0);
    const int64_t K = input.size(1);
    auto device = input.device();

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({total_rows, K / 2}, opts_fp4);
    auto row_sc_prepared_cat = torch::empty({total_rows / 128, K / 64, 512}, opts_fp8);
    auto row_sg_cat = torch::empty({total_rows / 128, K / 128}, opts_f32);

    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t row_offset = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 128 == 0, "split rows must be multiples of 128");
        auto chunk = input.narrow(0, row_offset, rows_i);
        auto row_fp4_view = row_fp4_cat.narrow(0, row_offset, rows_i);
        auto row_sc_prepared_view = row_sc_prepared_cat.narrow(0, row_offset / 128, rows_i / 128);
        auto row_sg_view = row_sg_cat.narrow(0, row_offset / 128, rows_i / 128);
        auto col_fp4 = torch::empty({K, rows_i / 2}, opts_fp4);
        auto col_sc_prepared = torch::empty({K / 128, rows_i / 64, 512}, opts_fp8);
        auto col_sg = torch::empty({K / 128, rows_i / 128}, opts_f32);

        tk_localcta_quantize_for_gemm_prepared_launch(
            chunk, true, true,
            row_fp4_view, row_sc_prepared_view,
            col_fp4, col_sc_prepared,
            row_sg_view, col_sg);

        row_sg_parts.push_back(row_sg_view);
        col_fp4_list.push_back(col_fp4);
        col_sc_prepared_list.push_back(col_sc_prepared);
        col_sg_list.push_back(col_sg);
        row_offset += rows_i;
    }
    auto col_fp4_cat = torch::cat(col_fp4_list, 1);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 1);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_list, col_sc_prepared_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           col_fp4_cat, col_sc_prepared_cat);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta2_group_quantize_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    const int64_t total_rows = input.size(0);
    const int64_t K = input.size(1);
    auto device = input.device();

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({total_rows, K / 2}, opts_fp4);
    auto row_sc_prepared_cat = torch::empty({total_rows / 128, K / 64, 512}, opts_fp8);
    auto row_sg_cat = torch::empty({total_rows / 128, K / 128}, opts_f32);

    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t row_offset = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 128 == 0, "split rows must be multiples of 128");
        auto chunk = input.narrow(0, row_offset, rows_i);
        auto row_fp4_view = row_fp4_cat.narrow(0, row_offset, rows_i);
        auto row_sc_prepared_view = row_sc_prepared_cat.narrow(0, row_offset / 128, rows_i / 128);
        auto row_sg_view = row_sg_cat.narrow(0, row_offset / 128, rows_i / 128);
        auto col_fp4 = torch::empty({K, rows_i / 2}, opts_fp4);
        auto col_sc_prepared = torch::empty({K / 128, rows_i / 64, 512}, opts_fp8);
        auto col_sg = torch::empty({K / 128, rows_i / 128}, opts_f32);

        tk_localcta_quantize_for_gemm_prepared_launch(
            chunk, true, true,
            row_fp4_view, row_sc_prepared_view,
            col_fp4, col_sc_prepared,
            row_sg_view, col_sg);

        row_sg_parts.push_back(row_sg_view);
        col_fp4_list.push_back(col_fp4);
        col_sc_prepared_list.push_back(col_sc_prepared);
        col_sg_list.push_back(col_sg);
        row_offset += rows_i;
    }
    auto col_fp4_cat = torch::cat(col_fp4_list, 1);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 1);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_list, col_sc_prepared_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           col_fp4_cat, col_sc_prepared_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    int64_t total_cols = 0;
    bool can_use_tileglobal_concat = (input.size(0) % 128) == 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        total_cols += cols_i;
        can_use_tileglobal_concat = can_use_tileglobal_concat && ((cols_i % 256) == 0);
    }
    TORCH_CHECK(total_cols == input.size(1),
                "column splits must sum to input.size(1)");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> col_sg_list;

    torch::Tensor row_fp4_cat;
    torch::Tensor row_sc_cat;
    torch::Tensor row_sg_cat;
    torch::Tensor col_fp4_cat;
    torch::Tensor col_sc_cat;
    torch::Tensor col_sg_cat;

    if (can_use_tileglobal_concat) {
        auto [row_fp4_full, row_sc_full, col_fp4_full, col_sc_full, row_sg_full, col_sg_full] =
            tk_localcta_quantize_for_gemm(input, true, true);

        row_fp4_cat = row_fp4_full;
        row_sc_cat = row_sc_full;
        col_fp4_cat = col_fp4_full;
        col_sc_cat = col_sc_full;
        col_sg_cat = col_sg_full;

        row_fp4_list.reserve(col_split_sections.size());
        row_sc_list.reserve(col_split_sections.size());
        row_sg_list.reserve(col_split_sections.size());
        col_fp4_list.reserve(col_split_sections.size());
        col_sc_list.reserve(col_split_sections.size());
        col_sg_list.reserve(col_split_sections.size());

        int64_t fp4_offset = 0;
        int64_t sc_offset = 0;
        int64_t col_offset = 0;
        int64_t sg_offset = 0;
        for (int64_t cols_i : col_split_sections) {
            const int64_t fp4_cols = cols_i / 2;
            const int64_t sc_tiles = cols_i / 64;
            const int64_t sg_tiles = outer_sg_tiles_128(cols_i);
            row_fp4_list.push_back(
                row_fp4_full.narrow(1, fp4_offset, fp4_cols)
                .view(torch::kUInt8)
                .contiguous()
                .view(torch::kFloat4_e2m1fn_x2)
            );
            row_sc_list.push_back(row_sc_full.narrow(1, sc_offset, sc_tiles).contiguous());
            row_sg_list.push_back(row_sg_full);
            col_fp4_list.push_back(
                col_fp4_full.narrow(0, col_offset, cols_i)
                .view(torch::kUInt8)
                .contiguous()
                .view(torch::kFloat4_e2m1fn_x2)
            );
            col_sc_list.push_back(col_sc_full.narrow(0, col_offset / 128, sc_tiles / 2).contiguous());
            col_sg_list.push_back(col_sg_full.narrow(1, sg_offset, sg_tiles).contiguous());
            fp4_offset += fp4_cols;
            sc_offset += sc_tiles;
            col_offset += cols_i;
            sg_offset += sg_tiles;
        }
        row_sg_cat = row_sg_full;
    } else {
        int64_t col_offset = 0;
        for (int64_t cols_i : col_split_sections) {
            auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
            auto [rf, rs, cf, cs, rsg, csg] = tk_localcta_quantize_for_gemm(chunk, true, true);
            row_fp4_list.push_back(rf);
            row_sc_list.push_back(rs);
            row_sg_list.push_back(rsg);
            col_fp4_list.push_back(cf);
            col_sc_list.push_back(cs);
            col_sg_list.push_back(csg);
            col_offset += cols_i;
        }

        row_fp4_cat = torch::cat(row_fp4_list, 1);
        row_sc_cat = torch::cat(row_sc_list, 1);
        row_sg_cat = torch::stack(row_sg_list, 0).amax(0);
        col_fp4_cat = torch::cat(col_fp4_list, 0);
        col_sc_cat = torch::cat(col_sc_list, 0);
        col_sg_cat = torch::cat(col_sg_list, 1);
    }

    return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                           col_fp4_list, col_sc_list, col_sg_list,
                           row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_cat, col_sc_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_for_gemm_fast(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_list;
    std::vector<torch::Tensor> row_sc_prepared_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
        auto [rf, rs, cf, cs, rsg, csg, rsp, csp] =
            tk_localcta_quantize_for_gemm_fast(chunk, true, true);
        row_fp4_list.push_back(rf);
        row_sc_list.push_back(rs);
        row_sc_prepared_list.push_back(rsp);
        row_sg_list.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_list.push_back(cs);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        col_offset += cols_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_list, 1);
    auto row_sc_cat = torch::cat(row_sc_list, 1);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_list, 1);
    auto row_sg_cat = torch::stack(row_sg_list, 1);
    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_cat = torch::cat(col_sc_list, 0);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                           col_fp4_list, col_sc_list, col_sg_list,
                           row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_cat, col_sc_cat, col_sg_cat,
                           row_sc_prepared_list, col_sc_prepared_list,
                           row_sc_prepared_cat, col_sc_prepared_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_prepared_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta_quantize_for_gemm_prepared(chunk, true, true);
        row_fp4_list.push_back(rf);
        row_sc_prepared_list.push_back(rsp);
        row_sg_list.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        col_offset += cols_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_list, 1);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_list, 1);
    auto row_sg_cat = torch::stack(row_sg_list, 1);
    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_prepared_list, row_sg_list,
                           col_fp4_list, col_sc_prepared_list, col_sg_list,
                           row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_concat_group_quantize_dim1_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                "input dims must be multiples of 128");

    int64_t total_cols = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        total_cols += cols_i;
    }
    TORCH_CHECK(total_cols == input.size(1),
                "column splits must sum to input.size(1)");

    auto [row_fp4_full, row_sc_prepared_full,
          col_fp4_full, col_sc_prepared_full,
          row_sg_full, col_sg_full] =
        tk_localcta_quantize_for_gemm_prepared(input, true, true);

    std::vector<torch::Tensor> row_sc_prepared_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;
    row_sc_prepared_list.reserve(col_split_sections.size());
    row_sg_list.reserve(col_split_sections.size());
    col_fp4_list.reserve(col_split_sections.size());
    col_sc_prepared_list.reserve(col_split_sections.size());
    col_sg_list.reserve(col_split_sections.size());

    int64_t sc_offset = 0;
    int64_t sg_offset = 0;
    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        const int64_t sc_tiles = cols_i / 64;
        const int64_t sg_tiles = cols_i / 128;
        row_sc_prepared_list.push_back(
            row_sc_prepared_full.narrow(1, sc_offset, sc_tiles).contiguous()
        );
        row_sg_list.push_back(
            row_sg_full.narrow(1, sg_offset, sg_tiles).contiguous()
        );
        col_fp4_list.push_back(
            col_fp4_full.narrow(0, col_offset, cols_i)
        );
        col_sc_prepared_list.push_back(
            col_sc_prepared_full.narrow(0, sg_offset, sg_tiles)
        );
        col_sg_list.push_back(
            col_sg_full.narrow(0, sg_offset, sg_tiles)
        );
        sc_offset += sc_tiles;
        sg_offset += sg_tiles;
        col_offset += cols_i;
    }

    return std::make_tuple(
        row_fp4_full,
        row_sc_prepared_list,
        row_sg_list,
        col_fp4_list,
        col_sc_prepared_list,
        col_sg_list,
        col_fp4_full,
        col_sc_prepared_full,
        col_sg_full);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
make_localcta_split2_quant_views(
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor col_sg_cat,
    int64_t n0,
    int64_t n1
) {
    std::vector<torch::Tensor> row_fp4_list{
        row_fp4_cat.narrow(1, 0, n0 / 2),
        row_fp4_cat.narrow(1, n0 / 2, n1 / 2),
    };
    std::vector<torch::Tensor> row_sc_prepared_views{
        row_sc_prepared_cat.narrow(1, 0, n0 / 64),
        row_sc_prepared_cat.narrow(1, n0 / 64, n1 / 64),
    };
    std::vector<torch::Tensor> row_sg_views{
        row_sg_cat.narrow(1, 0, n0 / 128),
        row_sg_cat.narrow(1, n0 / 128, n1 / 128),
    };
    std::vector<torch::Tensor> col_fp4_list{
        col_fp4_cat.narrow(0, 0, n0),
        col_fp4_cat.narrow(0, n0, n1),
    };
    std::vector<torch::Tensor> col_sc_prepared_list{
        col_sc_prepared_cat.narrow(0, 0, n0 / 128),
        col_sc_prepared_cat.narrow(0, n0 / 128, n1 / 128),
    };
    std::vector<torch::Tensor> col_sg_list{
        col_sg_cat.narrow(0, 0, n0 / 128),
        col_sg_cat.narrow(0, n0 / 128, n1 / 128),
    };

    return std::make_tuple(row_fp4_list, row_sc_prepared_views, row_sg_views,
                           col_fp4_list, col_sc_prepared_list, col_sg_list,
                           row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
    int64_t M,
    int64_t n0,
    int64_t n1,
    torch::Device device
) {
    TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(n0 % 128 == 0 && n1 % 128 == 0, "split widths must be multiples of 128");
    const int64_t total_n = n0 + n1;
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({M, total_n / 2}, opts_fp4);
    auto row_sc_prepared_cat = torch::empty({M / 128, total_n / 64, 512}, opts_fp8);
    auto row_sg_cat = torch::empty({M / 128, total_n / 128}, opts_f32);
    auto col_fp4_cat = torch::empty({total_n, M / 2}, opts_fp4);
    auto col_sc_prepared_cat = torch::empty({total_n / 128, M / 64, 512}, opts_fp8);
    auto col_sg_cat = torch::empty({total_n / 128, M / 128}, opts_f32);

    return make_localcta_split2_quant_views(
        row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
        col_fp4_cat, col_sc_prepared_cat, col_sg_cat,
        n0, n1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
    int64_t M,
    int64_t H,
    torch::Device device
) {
    TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(H % 128 == 0, "H must be a multiple of 128");
    auto [row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0] =
        allocate_quant_outputs_v3(M, H, true, device);
    auto [row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1] =
        allocate_quant_outputs_v3(M, H, true, device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto row_sg_chunk_0 = torch::empty({M / 128, H / 128}, opts_f32);
    auto col_sg_chunk_0 = torch::empty({H / 128, M / 128}, opts_f32);
    auto row_sg_chunk_1 = torch::empty({M / 128, H / 128}, opts_f32);
    auto col_sg_chunk_1 = torch::empty({H / 128, M / 128}, opts_f32);
    return std::make_tuple(
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1,
        row_sg_chunk_0, col_sg_chunk_0, row_sg_chunk_1, col_sg_chunk_1);
}

std::vector<torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc(
    int64_t M,
    int64_t H,
    torch::Device device
) {
    TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(H % 128 == 0, "H must be a multiple of 128");

    auto out0 = allocate_quant_outputs_v3(M, H, true, device);
    auto out1 = allocate_quant_outputs_v3(M, H, true, device);

    auto row_fp4_0 = std::get<0>(out0);
    auto row_sc_0 = std::get<1>(out0);
    auto row_sg_0 = std::get<4>(out0);
    auto row_fp4_1 = std::get<0>(out1);
    auto row_sc_1 = std::get<1>(out1);
    auto row_sg_1 = std::get<4>(out1);

    auto col_fp4_template = std::get<2>(out0);
    auto col_sc_template = std::get<3>(out0);
    auto col_sg_template = std::get<5>(out0);

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto col_fp4_cat = torch::empty(
        {2 * col_fp4_template.size(0), col_fp4_template.size(1)},
        opts_fp4);
    auto col_sc_cat = torch::empty(
        {2 * col_sc_template.size(0), col_sc_template.size(1), col_sc_template.size(2)},
        opts_fp8);
    auto col_fp4_0 = col_fp4_cat.narrow(0, 0, col_fp4_template.size(0));
    auto col_fp4_1 = col_fp4_cat.narrow(0, col_fp4_template.size(0), col_fp4_template.size(0));
    auto col_sc_0 = col_sc_cat.narrow(0, 0, col_sc_template.size(0));
    auto col_sc_1 = col_sc_cat.narrow(0, col_sc_template.size(0), col_sc_template.size(0));

    torch::Tensor col_sg_cat;
    torch::Tensor col_sg_0;
    torch::Tensor col_sg_1;
    if (col_sg_template.dim() == 2 && col_sg_template.size(0) == 1) {
        col_sg_cat = torch::empty({1, 2 * col_sg_template.size(1)}, opts_f32);
        col_sg_0 = col_sg_cat.narrow(1, 0, col_sg_template.size(1));
        col_sg_1 = col_sg_cat.narrow(1, col_sg_template.size(1), col_sg_template.size(1));
    } else {
        col_sg_cat = torch::empty(
            {2 * col_sg_template.size(0), col_sg_template.size(1)},
            opts_f32);
        col_sg_0 = col_sg_cat.narrow(0, 0, col_sg_template.size(0));
        col_sg_1 = col_sg_cat.narrow(0, col_sg_template.size(0), col_sg_template.size(0));
    }

    auto row_sg_chunk_0 = torch::empty({M / 128, H / 128}, opts_f32);
    auto col_sg_chunk_0 = torch::empty({H / 128, M / 128}, opts_f32);
    auto row_sg_chunk_1 = torch::empty({M / 128, H / 128}, opts_f32);
    auto col_sg_chunk_1 = torch::empty({H / 128, M / 128}, opts_f32);

    return {
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1,
        row_sg_chunk_0, col_sg_chunk_0, row_sg_chunk_1, col_sg_chunk_1,
        col_fp4_cat, col_sc_cat, col_sg_cat,
    };
}

void tk_localcta_silu_deriv_split_bf16_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1,
    torch::Tensor dh3_out
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have identical shape");
    TORCH_CHECK(dh1.sizes() == dh.sizes() && dh3_out.sizes() == dh.sizes(),
                "dh1/dh3_out must match dh shape");
    TORCH_CHECK(dh1.scalar_type() == torch::kBFloat16 && dh3_out.scalar_type() == torch::kBFloat16,
                "dh1/dh3_out must be bf16");
    TORCH_CHECK(dh1.is_cuda() && dh1.is_contiguous() && dh3_out.is_cuda() && dh3_out.is_contiguous(),
                "dh1/dh3_out must be contiguous CUDA tensors");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
        dh.size(0), dh.size(1), stream);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_silu_deriv_split_bf16_launch_inplace failed: ",
                cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_alloc(
    int64_t M,
    int64_t H,
    torch::Device device
) {
    TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(H % 128 == 0, "H must be a multiple of 128");
    auto [row_fp4_1, row_sc_prepared_1, col_fp4_1, col_sc_prepared_1, row_sg_1, col_sg_1] =
        allocate_quant_outputs_prepared(M, H, false, device);
    auto [row_fp4_2, row_sc_prepared_2, col_fp4_2, col_sc_prepared_2, row_sg_2, col_sg_2] =
        allocate_quant_outputs_prepared(M, H, false, device);
    (void)col_fp4_1;
    (void)col_sc_prepared_1;
    (void)col_sg_1;
    (void)col_fp4_2;
    (void)col_sc_prepared_2;
    (void)col_sg_2;
    return std::make_tuple(
        row_fp4_1, row_sc_prepared_1, row_sg_1,
        row_fp4_2, row_sc_prepared_2, row_sg_2);
}

void tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
);

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
        input0, input1,
        row_fp4_cat, row_sc_prepared_cat,
        col_fp4_cat, col_sc_prepared_cat,
        row_sg_cat, col_sg_cat);

    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    return make_localcta_split2_quant_views(
        row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
        col_fp4_cat, col_sc_prepared_cat, col_sg_cat,
        n0, n1);
}

void tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    for (const auto &input : {input0, input1}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda(),
                    "split inputs must be CUDA [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
        TORCH_CHECK(input.stride(1) == 1, "split inputs must be row-major with stride(1)==1");
        TORCH_CHECK(input.stride(0) >= input.size(1), "split inputs must have valid row stride");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t total_n = n0 + n1;

    TORCH_CHECK(row_fp4_cat.is_cuda() && row_fp4_cat.is_contiguous(),
                "row_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_cat.sizes() == torch::IntArrayRef({M, total_n / 2}),
                "row_fp4_cat shape mismatch");

    TORCH_CHECK(row_sc_prepared_cat.is_cuda() && row_sc_prepared_cat.is_contiguous(),
                "row_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 64, 512}),
                "row_sc_prepared_cat shape mismatch");

    TORCH_CHECK(row_sg_cat.is_cuda() && row_sg_cat.is_contiguous(),
                "row_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_cat.scalar_type() == torch::kFloat32,
                "row_sg_cat must be float32");
    TORCH_CHECK(row_sg_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 128}),
                "row_sg_cat shape mismatch");

    TORCH_CHECK(col_fp4_cat.is_cuda() && col_fp4_cat.is_contiguous(),
                "col_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "col_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(col_fp4_cat.sizes() == torch::IntArrayRef({total_n, M / 2}),
                "col_fp4_cat shape mismatch");

    TORCH_CHECK(col_sc_prepared_cat.is_cuda() && col_sc_prepared_cat.is_contiguous(),
                "col_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "col_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(col_sc_prepared_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 64, 512}),
                "col_sc_prepared_cat shape mismatch");

    TORCH_CHECK(col_sg_cat.is_cuda() && col_sg_cat.is_contiguous(),
                "col_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_sg_cat.scalar_type() == torch::kFloat32,
                "col_sg_cat must be float32");
    TORCH_CHECK(col_sg_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 128}),
                "col_sg_cat shape mismatch");

    launch_localcta_split2_quant_prepared(
        input0, input1,
        row_fp4_cat, row_sc_prepared_cat,
        col_fp4_cat, col_sc_prepared_cat,
        row_sg_cat, col_sg_cat);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
}

void tk_localcta_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor row_sg_cat
) {
    for (const auto &input : {input0, input1}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda(),
                    "split inputs must be CUDA [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
        TORCH_CHECK(input.stride(1) == 1, "split inputs must be row-major with stride(1)==1");
        TORCH_CHECK(input.stride(0) >= input.size(1), "split inputs must have valid row stride");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t total_n = n0 + n1;

    TORCH_CHECK(row_fp4_cat.is_cuda() && row_fp4_cat.is_contiguous(),
                "row_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_cat.sizes() == torch::IntArrayRef({M, total_n / 2}),
                "row_fp4_cat shape mismatch");

    TORCH_CHECK(row_sc_prepared_cat.is_cuda() && row_sc_prepared_cat.is_contiguous(),
                "row_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 64, 512}),
                "row_sc_prepared_cat shape mismatch");

    TORCH_CHECK(row_sg_cat.is_cuda() && row_sg_cat.is_contiguous(),
                "row_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_cat.scalar_type() == torch::kFloat32,
                "row_sg_cat must be float32");
    TORCH_CHECK(row_sg_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 128}),
                "row_sg_cat shape mismatch");

    launch_localcta_split2_row_quant_prepared(
        input0, input1,
        row_fp4_cat, row_sc_prepared_cat,
        row_sg_cat);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
}

void tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_prepared_1,
    torch::Tensor row_sg_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_prepared_2,
    torch::Tensor row_sg_2
) {
    for (const auto &input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split-deriv inputs must be contiguous CUDA [M, H]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split-deriv inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have identical shapes");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);

    TORCH_CHECK(row_fp4_1.is_cuda() && row_fp4_1.is_contiguous(),
                "row_fp4_1 must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_1.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_1 must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_1.sizes() == torch::IntArrayRef({M, H / 2}),
                "row_fp4_1 shape mismatch");

    TORCH_CHECK(row_sc_prepared_1.is_cuda() && row_sc_prepared_1.is_contiguous(),
                "row_sc_prepared_1 must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_1.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_1 must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_1.sizes() == torch::IntArrayRef({M / 128, H / 64, 512}),
                "row_sc_prepared_1 shape mismatch");

    TORCH_CHECK(row_sg_1.is_cuda() && row_sg_1.is_contiguous(),
                "row_sg_1 must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_1.scalar_type() == torch::kFloat32,
                "row_sg_1 must be float32");
    TORCH_CHECK(row_sg_1.sizes() == torch::IntArrayRef({M / 128, H / 128}),
                "row_sg_1 shape mismatch");

    TORCH_CHECK(row_fp4_2.is_cuda() && row_fp4_2.is_contiguous(),
                "row_fp4_2 must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_2.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_2 must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_2.sizes() == torch::IntArrayRef({M, H / 2}),
                "row_fp4_2 shape mismatch");

    TORCH_CHECK(row_sc_prepared_2.is_cuda() && row_sc_prepared_2.is_contiguous(),
                "row_sc_prepared_2 must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_2.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_2 must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_2.sizes() == torch::IntArrayRef({M / 128, H / 64, 512}),
                "row_sc_prepared_2 shape mismatch");

    TORCH_CHECK(row_sg_2.is_cuda() && row_sg_2.is_contiguous(),
                "row_sg_2 must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_2.scalar_type() == torch::kFloat32,
                "row_sg_2 must be float32");
    TORCH_CHECK(row_sg_2.sizes() == torch::IntArrayRef({M / 128, H / 128}),
                "row_sg_2 shape mismatch");

    launch_localcta_direct_silu_deriv_split_prepared<false>(
        dh, h3, h1_raw,
        row_fp4_1, row_sc_prepared_1,
        row_fp4_1, row_sc_prepared_1,
        row_sg_1, row_sg_1,
        row_fp4_2, row_sc_prepared_2,
        row_fp4_2, row_sc_prepared_2,
        row_sg_2, row_sg_2);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
}

void tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat,
    bool data_stochastic_rounding = false,
    bool scale_stochastic_rounding = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
    for (const auto &input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split-deriv inputs must be contiguous CUDA [M, H]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split-deriv inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have identical shapes");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    TORCH_CHECK(row_fp4_cat.is_cuda() && row_fp4_cat.is_contiguous(),
                "row_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_cat.sizes() == torch::IntArrayRef({M, total_n / 2}),
                "row_fp4_cat shape mismatch");

    TORCH_CHECK(row_sc_prepared_cat.is_cuda() && row_sc_prepared_cat.is_contiguous(),
                "row_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 64, 512}),
                "row_sc_prepared_cat shape mismatch");

    TORCH_CHECK(row_sg_cat.is_cuda() && row_sg_cat.is_contiguous(),
                "row_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_cat.scalar_type() == torch::kFloat32,
                "row_sg_cat must be float32");
    TORCH_CHECK(row_sg_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 128}),
                "row_sg_cat shape mismatch");

    TORCH_CHECK(col_fp4_cat.is_cuda() && col_fp4_cat.is_contiguous(),
                "col_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "col_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(col_fp4_cat.sizes() == torch::IntArrayRef({total_n, M / 2}),
                "col_fp4_cat shape mismatch");

    TORCH_CHECK(col_sc_prepared_cat.is_cuda() && col_sc_prepared_cat.is_contiguous(),
                "col_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "col_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(col_sc_prepared_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 64, 512}),
                "col_sc_prepared_cat shape mismatch");

    TORCH_CHECK(col_sg_cat.is_cuda() && col_sg_cat.is_contiguous(),
                "col_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_sg_cat.scalar_type() == torch::kFloat32,
                "col_sg_cat must be float32");
    TORCH_CHECK(col_sg_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 128}),
                "col_sg_cat shape mismatch");

    auto row_fp4_0 = row_fp4_cat.narrow(1, 0, H / 2);
    auto row_fp4_1 = row_fp4_cat.narrow(1, H / 2, H / 2);
    auto row_sc_0 = row_sc_prepared_cat.narrow(1, 0, H / 64);
    auto row_sc_1 = row_sc_prepared_cat.narrow(1, H / 64, H / 64);
    auto row_sg_0 = row_sg_cat.narrow(1, 0, H / 128);
    auto row_sg_1 = row_sg_cat.narrow(1, H / 128, H / 128);

    auto col_fp4_0 = col_fp4_cat.narrow(0, 0, H);
    auto col_fp4_1 = col_fp4_cat.narrow(0, H, H);
    auto col_sc_0 = col_sc_prepared_cat.narrow(0, 0, H / 128);
    auto col_sc_1 = col_sc_prepared_cat.narrow(0, H / 128, H / 128);
    auto col_sg_0 = col_sg_cat.narrow(0, 0, H / 128);
    auto col_sg_1 = col_sg_cat.narrow(0, H / 128, H / 128);

    launch_localcta_direct_silu_deriv_split_prepared<true>(
        dh, h3, h1_raw,
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rng_seed,
        rng_subsequence_base);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
}

void tk_localcta_finalize_split2_for_gemm_prepared_inplace(
    torch::Tensor row_sc_0,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sc_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_sc_1,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sc_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor col_sg_1
) {
    for (const auto& t : {row_sc_0, row_sg_chunk_0, row_sg_0,
                          col_sc_0, col_sg_chunk_0, col_sg_0,
                          row_sc_1, row_sg_chunk_1, row_sg_1,
                          col_sc_1, col_sg_chunk_1, col_sg_1}) {
        TORCH_CHECK(t.is_cuda(), "split2 prepared finalizer tensors must be CUDA");
    }
    TORCH_CHECK(row_sg_0.is_contiguous() && row_sg_1.is_contiguous() &&
                col_sc_0.is_contiguous() && col_sc_1.is_contiguous() &&
                col_sg_chunk_0.is_contiguous() && col_sg_chunk_1.is_contiguous() &&
                col_sg_0.is_contiguous() && col_sg_1.is_contiguous(),
                "split2 prepared finalizer SG/col tensors must be contiguous");
    TORCH_CHECK(row_sc_0.scalar_type() == torch::kFloat8_e4m3fn &&
                row_sc_1.scalar_type() == torch::kFloat8_e4m3fn &&
                col_sc_0.scalar_type() == torch::kFloat8_e4m3fn &&
                col_sc_1.scalar_type() == torch::kFloat8_e4m3fn,
                "split2 prepared finalizer scale tensors must be fp8 e4m3");
    TORCH_CHECK(row_sg_chunk_0.scalar_type() == torch::kFloat32 &&
                row_sg_chunk_1.scalar_type() == torch::kFloat32 &&
                col_sg_chunk_0.scalar_type() == torch::kFloat32 &&
                col_sg_chunk_1.scalar_type() == torch::kFloat32 &&
                row_sg_0.scalar_type() == torch::kFloat32 &&
                row_sg_1.scalar_type() == torch::kFloat32 &&
                col_sg_0.scalar_type() == torch::kFloat32 &&
                col_sg_1.scalar_type() == torch::kFloat32,
                "split2 prepared finalizer SG tensors must be float32");
    finalize_row_quant_contract_v3_strided(row_sc_0, row_sg_chunk_0, row_sg_0);
    finalize_row_quant_contract_v3_strided(row_sc_1, row_sg_chunk_1, row_sg_1);
    finalize_col_quant_contract_v3_split2(
        col_sc_0, col_sg_chunk_0, col_sg_0,
        col_sc_1, col_sg_chunk_1, col_sg_1);
}

static void launch_reduce_row_sg_to_col_tile_sg(
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_tile
);

static void launch_build_col_sg_ratio_from_row_sg(
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_tile,
    torch::Tensor col_sg_ratio
);

static void launch_broadcast_col_tile_sg_to_chunk_grid(
    torch::Tensor col_sg_tile,
    torch::Tensor col_sg_chunk
);

void tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    for (const auto &input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split-deriv inputs must be contiguous CUDA [M, H]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split-deriv inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have identical shapes");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    TORCH_CHECK(row_fp4_cat.is_cuda() && row_fp4_cat.is_contiguous(),
                "row_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_cat.sizes() == torch::IntArrayRef({M, total_n / 2}),
                "row_fp4_cat shape mismatch");

    TORCH_CHECK(row_sc_prepared_cat.is_cuda() && row_sc_prepared_cat.is_contiguous(),
                "row_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 64, 512}),
                "row_sc_prepared_cat shape mismatch");

    TORCH_CHECK(row_sg_cat.is_cuda() && row_sg_cat.is_contiguous(),
                "row_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_cat.scalar_type() == torch::kFloat32,
                "row_sg_cat must be float32");
    TORCH_CHECK(row_sg_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 128}),
                "row_sg_cat shape mismatch");

    TORCH_CHECK(col_fp4_cat.is_cuda() && col_fp4_cat.is_contiguous(),
                "col_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "col_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(col_fp4_cat.sizes() == torch::IntArrayRef({total_n, M / 2}),
                "col_fp4_cat shape mismatch");

    TORCH_CHECK(col_sc_prepared_cat.is_cuda() && col_sc_prepared_cat.is_contiguous(),
                "col_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "col_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(col_sc_prepared_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 64, 512}),
                "col_sc_prepared_cat shape mismatch");

    TORCH_CHECK(col_sg_cat.is_cuda() && col_sg_cat.is_contiguous(),
                "col_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(col_sg_cat.scalar_type() == torch::kFloat32,
                "col_sg_cat must be float32");
    TORCH_CHECK(col_sg_cat.sizes() == torch::IntArrayRef({total_n / 128, M / 128}),
                "col_sg_cat shape mismatch");

    if (use_localcta_v4_tuned_fused_split2()) {
        launch_localcta_silu_deriv_split2_quant_prepared_tuned(
            dh, h3, h1_raw,
            row_fp4_cat, row_sc_prepared_cat,
            col_fp4_cat, col_sc_prepared_cat,
            row_sg_cat, col_sg_cat);
    } else if (use_localcta_v4_direct_cluster_split2()) {
        auto row_fp4s = row_fp4_cat.split(H / 2, 1);
        auto row_scs = row_sc_prepared_cat.split(H / 64, 1);
        auto row_sgs = row_sg_cat.split(H / 128, 1);
        auto col_fp4s = col_fp4_cat.split(H, 0);
        auto col_scs = col_sc_prepared_cat.split(H / 128, 0);
        auto col_sgs = col_sg_cat.split(H / 128, 0);
        launch_localcta_cluster_silu_deriv_split_prepared<true>(
            dh, h3, h1_raw,
            row_fp4s[0], row_scs[0],
            col_fp4s[0], col_scs[0],
            row_sgs[0], col_sgs[0],
            row_fp4s[1], row_scs[1],
            col_fp4s[1], col_scs[1],
            row_sgs[1], col_sgs[1]);
    } else {
        // Stable baseline fallback.
        tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
            dh, h3, h1_raw,
            row_fp4_cat, row_sc_prepared_cat,
            col_fp4_cat, col_sc_prepared_cat,
            row_sg_cat, col_sg_cat);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace failed: ",
                cudaGetErrorString(err));
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace(
        dh, h3, h1_raw,
        row_fp4_cat, row_sc_prepared_cat,
        col_fp4_cat, col_sc_prepared_cat,
        row_sg_cat, col_sg_cat);
    const int64_t H = dh.size(1);
    return make_localcta_split2_quant_views(
        row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
        col_fp4_cat, col_sc_prepared_cat, col_sg_cat,
        H, H);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto bufs = tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
        M, H, H, dh.device());
    return tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch(
        dh, h3, h1_raw,
        std::get<6>(bufs), std::get<7>(bufs), std::get<9>(bufs),
        std::get<10>(bufs), std::get<8>(bufs), std::get<11>(bufs));
}

void tk_localcta_silu_deriv_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor row_sg_cat
) {
    for (const auto &input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split-deriv inputs must be contiguous CUDA [M, H]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split-deriv inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have identical shapes");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    TORCH_CHECK(row_fp4_cat.is_cuda() && row_fp4_cat.is_contiguous(),
                "row_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_cat.sizes() == torch::IntArrayRef({M, total_n / 2}),
                "row_fp4_cat shape mismatch");

    TORCH_CHECK(row_sc_prepared_cat.is_cuda() && row_sc_prepared_cat.is_contiguous(),
                "row_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 64, 512}),
                "row_sc_prepared_cat shape mismatch");

    TORCH_CHECK(row_sg_cat.is_cuda() && row_sg_cat.is_contiguous(),
                "row_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_cat.scalar_type() == torch::kFloat32,
                "row_sg_cat must be float32");
    TORCH_CHECK(row_sg_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 128}),
                "row_sg_cat shape mismatch");

    launch_localcta_silu_deriv_split2_row_quant_prepared(
        dh, h3, h1_raw,
        row_fp4_cat, row_sc_prepared_cat, row_sg_cat);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
}

void tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1_out,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor row_sg_cat
) {
    for (const auto &input : {dh, h3, h1_raw, dh1_out, dh3_out}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split-deriv bf16 row-fused inputs/outputs must be contiguous CUDA [M, H]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                    "split-deriv bf16 row-fused inputs/outputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have identical shapes");
    TORCH_CHECK(dh.sizes() == dh1_out.sizes() && dh.sizes() == dh3_out.sizes(),
                "dh1_out and dh3_out must match split-deriv input shapes");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    const int64_t total_n = 2 * H;

    TORCH_CHECK(row_fp4_cat.is_cuda() && row_fp4_cat.is_contiguous(),
                "row_fp4_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4_cat.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4_cat must be fp4 e2m1 x2");
    TORCH_CHECK(row_fp4_cat.sizes() == torch::IntArrayRef({M, total_n / 2}),
                "row_fp4_cat shape mismatch");

    TORCH_CHECK(row_sc_prepared_cat.is_cuda() && row_sc_prepared_cat.is_contiguous(),
                "row_sc_prepared_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc_prepared_cat.scalar_type() == torch::kFloat8_e4m3fn,
                "row_sc_prepared_cat must be fp8 e4m3");
    TORCH_CHECK(row_sc_prepared_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 64, 512}),
                "row_sc_prepared_cat shape mismatch");

    TORCH_CHECK(row_sg_cat.is_cuda() && row_sg_cat.is_contiguous(),
                "row_sg_cat must be contiguous CUDA tensor");
    TORCH_CHECK(row_sg_cat.scalar_type() == torch::kFloat32,
                "row_sg_cat must be float32");
    TORCH_CHECK(row_sg_cat.sizes() == torch::IntArrayRef({M / 128, total_n / 128}),
                "row_sg_cat shape mismatch");

    launch_localcta_silu_deriv_split2_row_bf16_quant_prepared_tuned(
        dh, h3, h1_raw,
        dh1_out, dh3_out,
        row_fp4_cat, row_sc_prepared_cat, row_sg_cat);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_cat,
    torch::Tensor row_sc_prepared_cat,
    torch::Tensor col_fp4_cat,
    torch::Tensor col_sc_prepared_cat,
    torch::Tensor row_sg_cat,
    torch::Tensor col_sg_cat
) {
    tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace(
        dh, h3, h1_raw,
        row_fp4_cat, row_sc_prepared_cat,
        col_fp4_cat, col_sc_prepared_cat,
        row_sg_cat, col_sg_cat);
    const int64_t H = dh.size(1);
    return make_localcta_split2_quant_views(
        row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
        col_fp4_cat, col_sc_prepared_cat, col_sg_cat,
        H, H);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_prepared_1,
    torch::Tensor row_sg_1,
    torch::Tensor row_fp4_2,
    torch::Tensor row_sc_prepared_2,
    torch::Tensor row_sg_2
) {
    tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch_inplace(
        dh, h3, h1_raw,
        row_fp4_1, row_sc_prepared_1, row_sg_1,
        row_fp4_2, row_sc_prepared_2, row_sg_2);
    return std::make_tuple(
        row_fp4_1, row_sc_prepared_1, row_sg_1,
        row_fp4_2, row_sc_prepared_2, row_sg_2);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto bufs = tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_alloc(
        M, H, dh.device());
    return tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch(
        dh, h3, h1_raw,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs),
        std::get<3>(bufs), std::get<4>(bufs), std::get<5>(bufs));
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto bufs = tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
        M, H, H, dh.device());
    return tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch(
        dh, h3, h1_raw,
        std::get<6>(bufs), std::get<7>(bufs), std::get<9>(bufs),
        std::get<10>(bufs), std::get<8>(bufs), std::get<11>(bufs));
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split2_for_gemm_prepared(
    torch::Tensor input0,
    torch::Tensor input1
) {
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    auto bufs = tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc(
        M, n0, n1, input0.device());
    return tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch(
        input0, input1,
        std::get<6>(bufs), std::get<7>(bufs), std::get<9>(bufs),
        std::get<10>(bufs), std::get<8>(bufs), std::get<11>(bufs));
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta2_group_quantize_dim1_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_prepared_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta2_quantize_for_gemm_prepared(chunk, true, true);
        row_fp4_list.push_back(rf);
        row_sc_prepared_list.push_back(rsp);
        row_sg_list.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        col_offset += cols_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_list, 1);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_list, 1);
    auto row_sg_cat = torch::stack(row_sg_list, 1);
    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_prepared_list, row_sg_list,
                           col_fp4_list, col_sc_prepared_list, col_sg_list,
                           row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_batched_quantize_for_gemm(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        std::vector<torch::Tensor> row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs;
        row_fp4s.reserve(inputs.size());
        row_scs.reserve(inputs.size());
        col_fp4s.reserve(inputs.size());
        col_scs.reserve(inputs.size());
        row_sgs.reserve(inputs.size());
        col_sgs.reserve(inputs.size());

        for (const auto &input : inputs) {
            auto [rf, rs, cf, cs, rsg, csg] =
                tk_localcta_quantize_for_gemm(input, return_transpose, encode_centric);
            row_fp4s.push_back(rf);
            row_scs.push_back(rs);
            col_fp4s.push_back(cf);
            col_scs.push_back(cs);
            row_sgs.push_back(rsg);
            col_sgs.push_back(csg);
        }
        return std::make_tuple(
            row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs,
            torch::Tensor(), torch::Tensor(), torch::Tensor());
    }

    const auto multiinput_mode = get_v3_split2_multiinput_quant_mode();

    if (return_transpose && inputs.size() == 2 && multiinput_mode != V3MultiInputQuantMode::Loop) {
        const auto& input0 = inputs[0];
        const auto& input1 = inputs[1];
        const bool compatible =
            encode_centric &&
            input0.dim() == 2 && input1.dim() == 2 &&
            input0.is_cuda() && input1.is_cuda() &&
            input0.is_contiguous() && input1.is_contiguous() &&
            input0.scalar_type() == torch::kBFloat16 &&
            input1.scalar_type() == torch::kBFloat16 &&
            input0.size(0) == input1.size(0) &&
            input0.size(1) % 128 == 0 && input1.size(1) % 128 == 0;
        if (compatible) {
            auto [row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0] =
                allocate_quant_outputs_v3(input0.size(0), input0.size(1), true, input0.device());
            auto [row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1] =
                allocate_quant_outputs_v3(input1.size(0), input1.size(1), true, input1.device());

            auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(input0.device());
            auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(input0.device());
            auto opts_f32 = torch::dtype(torch::kFloat32).device(input0.device());
            auto col_fp4_cat = torch::empty({input0.size(1) + input1.size(1), input0.size(0) / 2}, opts_fp4);
            auto col_sc_cat = torch::empty({(input0.size(1) + input1.size(1)) / 128, input0.size(0) / 64, 512}, opts_fp8);
            const int64_t col_sg_tiles_0 = outer_sg_tiles_128(input0.size(1));
            const int64_t col_sg_tiles_1 = outer_sg_tiles_128(input1.size(1));
            torch::Tensor col_sg_cat;
            if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
                col_sg_cat = torch::empty({(input0.size(1) + input1.size(1)) / 256, input0.size(0) / 256}, opts_f32);
            } else {
                col_sg_cat = torch::empty({1, col_sg_tiles_0 + col_sg_tiles_1}, opts_f32);
            }

            col_fp4_0 = col_fp4_cat.narrow(0, 0, input0.size(1));
            col_fp4_1 = col_fp4_cat.narrow(0, input0.size(1), input1.size(1));
            col_sc_0 = col_sc_cat.narrow(0, 0, input0.size(1) / 128);
            col_sc_1 = col_sc_cat.narrow(0, input0.size(1) / 128, input1.size(1) / 128);
            if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
                col_sg_0 = col_sg_cat.narrow(0, 0, input0.size(1) / 256);
                col_sg_1 = col_sg_cat.narrow(0, input0.size(1) / 256, input1.size(1) / 256);
            } else {
                col_sg_0 = col_sg_cat.narrow(1, 0, col_sg_tiles_0);
                col_sg_1 = col_sg_cat.narrow(1, col_sg_tiles_0, col_sg_tiles_1);
            }

            auto row_sg_chunk_0 = torch::empty({input0.size(0) / 128, input0.size(1) / 128}, opts_f32);
            auto col_sg_chunk_0 = torch::empty({input0.size(1) / 128, input0.size(0) / 128}, opts_f32);
            auto row_sg_chunk_1 = torch::empty({input1.size(0) / 128, input1.size(1) / 128}, opts_f32);
            auto col_sg_chunk_1 = torch::empty({input1.size(1) / 128, input1.size(0) / 128}, opts_f32);

            launch_localcta_split2_quant_raw(
                input0, input1,
                row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
                row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1);
            if (multiinput_mode == V3MultiInputQuantMode::SplitFinalize) {
                finalize_quant_contract_v3_split2(
                    row_sc_0, row_sg_chunk_0, row_sg_0, col_sc_0, col_sg_chunk_0, col_sg_0,
                    row_sc_1, row_sg_chunk_1, row_sg_1, col_sc_1, col_sg_chunk_1, col_sg_1);
            } else if (multiinput_mode == V3MultiInputQuantMode::ColSplitFinalize ||
                       multiinput_mode == V3MultiInputQuantMode::RowColSplitFinalize) {
                finalize_row_quant_contract_v3_strided(row_sc_0, row_sg_chunk_0, row_sg_0);
                finalize_row_quant_contract_v3_strided(row_sc_1, row_sg_chunk_1, row_sg_1);
                finalize_col_quant_contract_v3_split2(
                    col_sc_0, col_sg_chunk_0, col_sg_0,
                    col_sc_1, col_sg_chunk_1, col_sg_1);
            } else {
                finalize_quant_contract_v3(row_sc_0, row_sg_chunk_0, row_sg_0, col_sc_0, col_sg_chunk_0, col_sg_0);
                finalize_quant_contract_v3(row_sc_1, row_sg_chunk_1, row_sg_1, col_sc_1, col_sg_chunk_1, col_sg_1);
            }

            std::vector<torch::Tensor> row_fp4s{row_fp4_0, row_fp4_1};
            std::vector<torch::Tensor> row_scs{row_sc_0, row_sc_1};
            std::vector<torch::Tensor> col_fp4s{col_fp4_0, col_fp4_1};
            std::vector<torch::Tensor> col_scs{col_sc_0, col_sc_1};
            std::vector<torch::Tensor> row_sgs{row_sg_0, row_sg_1};
            std::vector<torch::Tensor> col_sgs{col_sg_0, col_sg_1};
            return std::make_tuple(
                row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs,
                col_fp4_cat, col_sc_cat, col_sg_cat);
        }
    }

    std::vector<torch::Tensor> row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs;
    row_fp4s.reserve(inputs.size());
    row_scs.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_scs.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rs, cf, cs, rsg, csg] =
            tk_localcta_quantize_for_gemm(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_scs.push_back(rs);
        col_fp4s.push_back(cf);
        col_scs.push_back(cs);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
    }
    return std::make_tuple(
        row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs,
        torch::Tensor(), torch::Tensor(), torch::Tensor());
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_batched_quantize_for_gemm_fast(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    std::vector<torch::Tensor> row_fp4s, row_scs, col_fp4s, col_scs;
    std::vector<torch::Tensor> row_sgs, col_sgs, row_sc_prepareds, col_sc_prepareds;
    row_fp4s.reserve(inputs.size());
    row_scs.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_scs.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());
    row_sc_prepareds.reserve(inputs.size());
    col_sc_prepareds.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rs, cf, cs, rsg, csg, rsp, csp] =
            tk_localcta_quantize_for_gemm_fast(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_scs.push_back(rs);
        col_fp4s.push_back(cf);
        col_scs.push_back(cs);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
        row_sc_prepareds.push_back(rsp);
        col_sc_prepareds.push_back(csp);
    }
    return std::make_tuple(row_fp4s, row_scs, col_fp4s, col_scs,
                           row_sgs, col_sgs, row_sc_prepareds, col_sc_prepareds);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_batched_quantize_for_gemm_prepared(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    std::vector<torch::Tensor> row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs;
    row_fp4s.reserve(inputs.size());
    row_sc_prepareds.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_sc_prepareds.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta_quantize_for_gemm_prepared(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_sc_prepareds.push_back(rsp);
        col_fp4s.push_back(cf);
        col_sc_prepareds.push_back(csp);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
    }
    return std::make_tuple(row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta2_batched_quantize_for_gemm_prepared(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    std::vector<torch::Tensor> row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs;
    row_fp4s.reserve(inputs.size());
    row_sc_prepareds.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_sc_prepareds.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta2_quantize_for_gemm_prepared(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_sc_prepareds.push_back(rsp);
        col_fp4s.push_back(cf);
        col_sc_prepareds.push_back(csp);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
    }
    return std::make_tuple(row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_quantize_col_only_prepared(
    torch::Tensor input,
    torch::Tensor sg_tensor
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous CUDA [M, N]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(sg_tensor.is_cuda() && sg_tensor.is_contiguous(),
                "sg_tensor must be contiguous CUDA tensor");
    TORCH_CHECK(sg_tensor.scalar_type() == torch::kFloat32, "sg_tensor must be float32");
    TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                "input M and N must be multiples of 128");
    TORCH_CHECK(sg_tensor.sizes() == torch::IntArrayRef({input.size(0) / 128, input.size(1) / 128}),
                "sg_tensor must have shape [M/128, N/128]");

    const int64_t M = input.size(0);
    const int64_t N = input.size(1);
    const int64_t ntm_c = N / 128;
    const int64_t ntk_c = M / 64;
    auto device = input.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto col_fp4 = torch::empty({N, M / 2}, opts_fp4);
    auto col_sc_prepared = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
    auto col_sg = torch::empty({N / 128, M / 128}, opts_f32);

    launch_localcta_quantize_col_only_prepared(
        input, sg_tensor, col_fp4, col_sc_prepared, col_sg
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_col_only_prepared failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(col_fp4, col_sc_prepared, col_sg);
}

void tk_localcta_quantize_col_only_prepared_launch_inplace(
    torch::Tensor input,
    torch::Tensor sg_tensor,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor col_sg
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous CUDA [M, N]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(sg_tensor.is_cuda() && sg_tensor.is_contiguous(),
                "sg_tensor must be contiguous CUDA tensor");
    TORCH_CHECK(sg_tensor.scalar_type() == torch::kFloat32, "sg_tensor must be float32");
    TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                "input M and N must be multiples of 128");
    TORCH_CHECK(sg_tensor.sizes() == torch::IntArrayRef({input.size(0) / 128, input.size(1) / 128}),
                "sg_tensor must have shape [M/128, N/128]");

    const int64_t M = input.size(0);
    const int64_t N = input.size(1);
    const int64_t ntm_c = N / 128;
    const int64_t ntk_c = M / 64;

    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous(),
                "col_fp4 must be contiguous CUDA tensor");
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "col_fp4 must be fp4 e2m1 x2");
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({N, M / 2}),
                "col_fp4 shape mismatch");

    TORCH_CHECK(col_sc_prepared.is_cuda() && col_sc_prepared.is_contiguous(),
                "col_sc_prepared must be contiguous CUDA tensor");
    TORCH_CHECK(col_sc_prepared.scalar_type() == torch::kFloat8_e4m3fn,
                "col_sc_prepared must be fp8 e4m3");
    TORCH_CHECK(col_sc_prepared.sizes() == torch::IntArrayRef({ntm_c, ntk_c, 512}),
                "col_sc_prepared shape mismatch");

    TORCH_CHECK(col_sg.is_cuda() && col_sg.is_contiguous(),
                "col_sg must be contiguous CUDA tensor");
    TORCH_CHECK(col_sg.scalar_type() == torch::kFloat32,
                "col_sg must be float32");
    TORCH_CHECK(col_sg.sizes() == torch::IntArrayRef({N / 128, M / 128}),
                "col_sg shape mismatch");

    launch_localcta_quantize_col_only_prepared(
        input, sg_tensor, col_fp4, col_sc_prepared, col_sg
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_col_only_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
}

template <bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(tk_localcta::THREADS)
localcta_sqrelu_quantize_col_only_raw_outer_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_col_raw,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
    tk_localcta::LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = transformer_engine::common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFFS_NUM_IN * tk_localcta::BUFF_IN_ELEMS * (int)sizeof(tk_localcta::IType),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::BUFFS_NUM_OUT_TR * tk_localcta::BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = DIVUP_TO_MULTIPLE(
        tk_localcta::LocalCTAConfig::CHUNK_DIM_X * tk_localcta::SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    auto* sIn_ptr = reinterpret_cast<tk_localcta::IType*>(dshmem);
    auto* sOut_tr_ptr = reinterpret_cast<transformer_engine::fp4e2m1x2*>(dshmem + in_bytes);
    auto* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_tr_bytes);
    auto& sOut_tr = *reinterpret_cast<tk_localcta::OType2xt3D*>(sOut_tr_ptr);
    __shared__ float warp_max[tk_localcta::THREADS / 32];
    __shared__ float slot_amax;

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = static_cast<int>(s_chunk_id % args.tiles_X);
        const int ctaid_Y = static_cast<int>(s_chunk_id / args.tiles_X);
        const int block_offset_Y = ctaid_Y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_Y_tr = ctaid_X * tk_localcta::LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * tk_localcta::LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;

        localcta_load_raw_chunk(sIn_ptr, h1_raw, (int)rows, (int)cols, block_offset_Y, block_offset_X);

        float cta_max = 0.0f;
        #pragma unroll
        for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
            cta_max = fmaxf(
                cta_max,
                tk_localcta::transform_sqrelu_tile_inplace_amax_group<tk_localcta::THREADS>(
                    sIn_ptr, t, threadIdx.x));
        }

        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }
        if (lane == 0) {
            warp_max[wid] = cta_max;
        }
        __syncthreads();
        if (wid == 0) {
            cta_max = (lane < tk_localcta::THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (tk_localcta::THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                slot_amax = cta_max;
                const int tiles_Y = static_cast<int>(rows / tk_localcta::LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] =
                    cta_max / tk_localcta::localcta_global_scale_num();
            }
        }
        __syncthreads();

        const float amax_val = slot_amax;
        const float sg_val = amax_val / tk_localcta::localcta_global_scale_num();
        const float S_enc = tk_localcta::compute_localcta_encode_scaling_factor_FP4(amax_val);

        int buff_out_tr = 0;
        #pragma unroll
        for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
            const int stage_Y = t / tk_localcta::TILES_X;
            const int stage_X = t % tk_localcta::TILES_X;
            const int stage_offset_Y = stage_Y * tk_localcta::TILE_DIM_Y;
            const int stage_offset_X = stage_X * tk_localcta::TILE_DIM_X;

            if (t > 0 && leading) {
                transformer_engine::ptx::cp_async_bulk_wait_group_read<1>();
            }

            tk_localcta::colwise_scaling<ENCODE_CENTRIC>(
                sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc,
                stage_Y, stage_X, t, buff_out_tr);

            transformer_engine::ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                transformer_engine::ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                transformer_engine::ptx::cp_async_bulk_commit_group();
            }

            buff_out_tr = (buff_out_tr + 1) % tk_localcta::BUFFS_NUM_OUT_TR;
        }

        if (leading) {
            transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        const int cnt = min((int)tk_localcta::SCALES_PER_CHUNK_Y, chunk_rows / tk_localcta::SCALE_DIM);
        tk_localcta::swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
        transformer_engine::ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tk_localcta::tma_store_scales_2x512(
                tmap_scale_col_raw, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

static void launch_localcta_sqrelu_quantize_col_only_raw_outer(
    torch::Tensor h1_raw,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_raw,
    torch::Tensor col_sg_chunks,
    bool encode_centric
) {
    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(H / 128);
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    alignas(64) CUtensorMap tmap_out_t{}, tmap_sc_col_raw{};
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), H, M,
                  tk_localcta::BUFF_DIM_X, tk_localcta::BUFF_DIM_Y, M, 4);
    const int64_t ntm_c = H / 128;
    const int64_t ntk_c = M / 64;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_col_raw, col_sc_raw.data_ptr(),
                  ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto& work_counter = get_localcta_persistent_counter(h1_raw.device());
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(work_counter_ptr, 0, sizeof(unsigned int), stream);

    const int dshmem = localcta_col_only_shmem_size<true>();
    auto kernel = encode_centric
        ? localcta_sqrelu_quantize_col_only_raw_outer_kernel<true>
        : localcta_sqrelu_quantize_col_only_raw_outer_kernel<false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);

    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        tmap_out_t,
        tmap_sc_col_raw,
        col_sg_chunks.data_ptr<float>(),
        M, H, args);
}

template <int BLOCK_SIZE = 256>
__global__ void reduce_row_sg_to_col_tile_sg_kernel(
    const float* __restrict__ row_sg_chunk,
    float* __restrict__ col_sg_tile,
    int row_chunks,
    int sg_cols
) {
    const int tile = blockIdx.x;
    const int c0 = tile * 2;
    const int c1 = c0 + 1;
    if (c0 >= sg_cols) return;

    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < row_chunks * 2; idx += BLOCK_SIZE) {
        const int row = idx / 2;
        const int local_col = idx % 2;
        const int col = local_col == 0 ? c0 : c1;
        if (col < sg_cols) {
            thread_max = fmaxf(thread_max, row_sg_chunk[row * sg_cols + col]);
        }
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        col_sg_tile[tile] = smem[0];
    }
}

template <int BLOCK_SIZE = 256>
__global__ void build_col_sg_ratio_from_row_sg_kernel(
    const float* __restrict__ row_sg_chunk,
    const float* __restrict__ col_sg_tile,
    float* __restrict__ col_sg_ratio,
    int row_chunks,
    int sg_cols
) {
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = row_chunks * sg_cols;
    if (idx >= total) return;

    const int row = idx / sg_cols;
    const int col = idx % sg_cols;
    const float denom = fmaxf(col_sg_tile[col / 2], 1e-12f);
    const float numer = row_sg_chunk[row * sg_cols + col];
    col_sg_ratio[idx] = numer / denom;
}

template <int BLOCK_SIZE = 256>
__global__ void broadcast_col_tile_sg_to_chunk_grid_kernel(
    const float* __restrict__ col_sg_tile,
    float* __restrict__ col_sg_chunk,
    int sg_cols,
    int row_chunks
) {
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = sg_cols * row_chunks;
    if (idx >= total) return;

    const int col = idx / row_chunks;
    const int row = idx % row_chunks;
    (void)row;
    col_sg_chunk[idx] = col_sg_tile[col / 2];
}

static void launch_reduce_row_sg_to_col_tile_sg(
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_tile
) {
    const int row_chunks = static_cast<int>(row_sg_chunk.size(0));
    const int sg_cols = static_cast<int>(row_sg_chunk.size(1));
    const int col_tiles = static_cast<int>(col_sg_tile.numel());
    if (col_tiles <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_row_sg_to_col_tile_sg_kernel<256><<<col_tiles, 256, 0, stream>>>(
        row_sg_chunk.data_ptr<float>(),
        col_sg_tile.data_ptr<float>(),
        row_chunks,
        sg_cols);
}

static void launch_build_col_sg_ratio_from_row_sg(
    torch::Tensor row_sg_chunk,
    torch::Tensor col_sg_tile,
    torch::Tensor col_sg_ratio
) {
    const int row_chunks = static_cast<int>(row_sg_chunk.size(0));
    const int sg_cols = static_cast<int>(row_sg_chunk.size(1));
    const int total = row_chunks * sg_cols;
    if (total <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int blocks = (total + 255) / 256;
    build_col_sg_ratio_from_row_sg_kernel<256><<<blocks, 256, 0, stream>>>(
        row_sg_chunk.data_ptr<float>(),
        col_sg_tile.data_ptr<float>(),
        col_sg_ratio.data_ptr<float>(),
        row_chunks,
        sg_cols);
}

static void launch_broadcast_col_tile_sg_to_chunk_grid(
    torch::Tensor col_sg_tile,
    torch::Tensor col_sg_chunk
) {
    const int sg_cols = static_cast<int>(col_sg_chunk.size(0));
    const int row_chunks = static_cast<int>(col_sg_chunk.size(1));
    const int total = sg_cols * row_chunks;
    if (total <= 0) {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int blocks = (total + 255) / 256;
    broadcast_col_tile_sg_to_chunk_grid_kernel<256><<<blocks, 256, 0, stream>>>(
        col_sg_tile.data_ptr<float>(),
        col_sg_chunk.data_ptr<float>(),
        sg_cols,
        row_chunks);
}

std::tuple<torch::Tensor, torch::Tensor>
tk_localcta_rmsnorm_to_bf16(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous CUDA [M, K]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be contiguous CUDA [K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be bf16");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma must match K dimension");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = input.device();
    auto normed = torch::empty(
        {M, K}, torch::dtype(torch::kBFloat16).device(device));
    auto inv_rms = torch::empty(
        {M}, torch::dtype(torch::kFloat32).device(device));
    constexpr int BS = 256;
    localcta_fused_norm_to_bf16_kernel<BS><<<M, BS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(normed.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K),
        false);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_rmsnorm_to_bf16 failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(normed, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_localcta_fused_norm_quantize(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon,
    bool with_silu = false,
    bool return_transpose = true
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous CUDA [M, K]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.is_cuda() && gamma.is_contiguous(),
                "gamma must be contiguous CUDA [K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be bf16");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma must match K dimension");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = input.device();
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto inv_rms = torch::empty({M}, opts_f32);

    auto normed = torch::empty({M, K}, opts_bf16);
    constexpr int BS = 256;
    localcta_fused_norm_to_bf16_kernel<BS><<<M, BS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(normed.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<float>(epsilon),
        static_cast<int>(M),
        static_cast<int>(K),
        with_silu);

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        tk_localcta_quantize_for_gemm_prepared(normed, return_transpose, true);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_fused_norm_quantize failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, inv_rms
    );
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor>
tk_localcta_silu_quantize_split_for_gemm_impl(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    bool paired_fixed_sign_col_rht
) {
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(), "h1_raw and h3 must have identical shape");
    TORCH_CHECK(h1_raw.device() == h3.device(),
                "h1_raw and h3 must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(h1_raw.device());
    const auto set_device_err = cudaSetDevice(h1_raw.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before fused SiLU quantization: ",
                cudaGetErrorString(set_device_err));

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);

    if (paired_fixed_sign_col_rht) {
        TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                    "paired SiLU column RHT requires the v4 outer-scale contract");
        TORCH_CHECK(M % 256 == 0 && H % 256 == 0,
                    "paired SiLU column RHT requires M and H to be multiples of 256");
        TORCH_CHECK(use_localcta_v4_silu_atomic_final_sg_producer(),
                    "paired SiLU column RHT requires the atomic final-SG producer");
        TORCH_CHECK(!use_localcta_v4_gemm_virtual_rescale_for_m(M),
                    "paired SiLU column RHT does not yet support virtual raw rescale");
    }

    const char* fused_silu_env = std::getenv("USE_TK_LOCALCTA_V4_FUSED_SILU_RAW");
    const bool fused_silu_disabled =
        fused_silu_env != nullptr && std::string(fused_silu_env) == "0";
    TORCH_CHECK(!paired_fixed_sign_col_rht || !fused_silu_disabled,
                "paired SiLU column RHT requires USE_TK_LOCALCTA_V4_FUSED_SILU_RAW!=0");
    const bool use_virtual_rescale = use_localcta_v4_gemm_virtual_rescale_for_m(M);
    const bool use_fused_silu_raw =
        paired_fixed_sign_col_rht || use_localcta_v4_fused_silu_raw() ||
        (use_virtual_rescale && use_localcta_v4_gemm_virtual_rescale_force_raw()) ||
        (!fused_silu_disabled && !should_use_localcta2_prepared_auto(M, H));

    if (use_fused_silu_raw) {
        auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
            allocate_quant_outputs_v3(M, H, true, h1_raw.device());
        auto opts_f32 = torch::dtype(torch::kFloat32).device(h1_raw.device());
        auto row_sg_chunk = torch::empty({M / 128, H / 128}, opts_f32);
        auto col_sg_chunk = torch::empty({H / 128, M / 128}, opts_f32);
        const bool use_atomic_final_sg = use_localcta_v4_silu_atomic_final_sg_producer();
        const bool use_virtual_raw_consumer = use_atomic_final_sg && use_virtual_rescale;

        if (use_atomic_final_sg) {
            if (paired_fixed_sign_col_rht) {
                launch_localcta_tma_silu_split_raw<true, false, true, true>(
                    h1_raw, h3,
                    row_fp4, row_sc, col_fp4, col_sc,
                    row_sg_chunk, col_sg_chunk,
                    row_sg, col_sg);
            } else {
                launch_localcta_tma_silu_split_raw<true, false, true>(
                    h1_raw, h3,
                    row_fp4, row_sc, col_fp4, col_sc,
                    row_sg_chunk, col_sg_chunk,
                    row_sg, col_sg);
            }
            if (!use_virtual_raw_consumer) {
                rescale_quant_contract_v3_from_final_sg(
                    row_sc, row_sg_chunk, row_sg, col_sc, col_sg_chunk, col_sg);
            }
        } else if (use_localcta_v4_silu_final_sg_producer()) {
            launch_scan_silu_sg(h1_raw, h3, true, row_sg_chunk, col_sg_chunk);

            auto stream = at::cuda::getCurrentCUDAStream();
            reduce_row_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(M)), 256, 0, stream>>>(
                row_sg_chunk.data_ptr<float>(),
                row_sg.data_ptr<float>(),
                static_cast<int>(M / 128),
                static_cast<int>(H / 128));
            {
                cudaError_t err = cudaGetLastError();
                TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_tiles_kernel failed for silu: ",
                            cudaGetErrorString(err));
            }
            reduce_col_sg_tiles_kernel<256><<<static_cast<unsigned int>(outer_sg_tiles_128(H)), 256, 0, stream>>>(
                col_sg_chunk.data_ptr<float>(),
                col_sg.data_ptr<float>(),
                static_cast<int>(H / 128),
                static_cast<int>(M / 128));
            {
                cudaError_t err = cudaGetLastError();
                TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_kernel failed for silu: ",
                            cudaGetErrorString(err));
            }

            launch_localcta_tma_silu_split_raw<true, true>(
                h1_raw, h3,
                row_fp4, row_sc, col_fp4, col_sc,
                row_sg, col_sg);
        } else {
            launch_localcta_tma_silu_split_raw<true>(
                h1_raw, h3,
                row_fp4, row_sc, col_fp4, col_sc,
                row_sg_chunk, col_sg_chunk);
            finalize_quant_contract_v3(row_sc, row_sg_chunk, row_sg, col_sc, col_sg_chunk, col_sg);
        }

        auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "tk_localcta_silu_quantize_split_for_gemm fused raw path failed: ",
                    cudaGetErrorString(err));
        return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                               use_virtual_raw_consumer ? row_sg_chunk : torch::Tensor(),
                               use_virtual_raw_consumer ? col_sg_chunk : torch::Tensor());
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(h1_raw.device());

    auto out = torch::empty({M, H}, opts_bf16);
    tk_silu_split::launch_forward(
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, H, stream);

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        tk_localcta_quantize_for_gemm(out, true, true);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_silu_quantize_split_for_gemm failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                           torch::Tensor(), torch::Tensor());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_silu_quantize_split_for_gemm(
    torch::Tensor h1_raw,
    torch::Tensor h3
) {
    return tk_localcta_silu_quantize_split_for_gemm_impl(
        h1_raw, h3, false);
}

bool tk_localcta_silu_supports_paired_col_rht() {
    return true;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_silu_quantize_split_for_gemm_paired_col_rht(
    torch::Tensor h1_raw,
    torch::Tensor h3
) {
    return tk_localcta_silu_quantize_split_for_gemm_impl(
        h1_raw, h3, true);
}

static void parse_localcta_sqrelu_rht_axes(
    std::string rht_axes,
    bool& row_rht,
    bool& col_rht
) {
    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "row_col" ||
               rht_axes == "rowcol" || rht_axes == "all");
    col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "row_col" ||
               rht_axes == "rowcol" || rht_axes == "all");
    TORCH_CHECK(rht_axes == "none" || rht_axes == "off" || rht_axes == "0" ||
                rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
                rht_axes == "row_col" || rht_axes == "rowcol" || rht_axes == "all",
                "Unsupported square-ReLU localCTA RHT axes: ", rht_axes);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_sqrelu_quantize_for_gemm_prepared(
    torch::Tensor h1_raw,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(h1_raw.size(0) % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(h1_raw.size(1) % 128 == 0, "H must be a multiple of 128");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    bool row_rht = false;
    bool col_rht = false;
    parse_localcta_sqrelu_rht_axes(rht_axes, row_rht, col_rht);
    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(M, H, true, h1_raw.device());

    launch_localcta_tma_sqrelu_prepared<true>(
        h1_raw,
        row_fp4, row_sc_prepared,
        col_fp4, col_sc_prepared,
        row_sg, col_sg,
        encode_centric,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        col_rht,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_sqrelu_quantize_for_gemm_prepared failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer(
    torch::Tensor h1_raw,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "square-ReLU row-prepared/col-outer quantizer requires the v4 outer-SG contract");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(h1_raw.size(0) % 256 == 0, "M must be a multiple of 256");
    TORCH_CHECK(h1_raw.size(1) % 256 == 0, "H must be a multiple of 256");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    bool row_rht = false;
    bool col_rht = false;
    parse_localcta_sqrelu_rht_axes(rht_axes, row_rht, col_rht);
    auto device = h1_raw.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, H / 2}, opts_fp4);
    auto row_sc_prepared = torch::empty({M / 128, H / 64, 512}, opts_fp8);
    auto col_fp4 = torch::empty({H, M / 2}, opts_fp4);
    auto col_sc_raw = torch::empty({H / 128, M / 64, 512}, opts_fp8);
    auto row_sg_chunks = torch::empty({M / 128, H / 128}, opts_f32);
    auto col_sg_chunks = torch::empty({H / 128, M / 128}, opts_f32);
    auto col_sg_outer = torch::empty({1, H / 256}, opts_f32);

    launch_localcta_tma_sqrelu_row_prepared_col_outer<true>(
        h1_raw,
        row_fp4, row_sc_prepared,
        col_fp4, col_sc_raw, col_sc_raw,
        row_sg_chunks, col_sg_chunks,
        encode_centric,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        col_rht,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer failed: ",
                cudaGetErrorString(err));
    finalize_col_quant_contract_v3(col_sc_raw, col_sg_chunks, col_sg_outer);
    return std::make_tuple(row_fp4, row_sc_prepared, row_sg_chunks,
                           col_fp4, col_sc_raw, col_sg_outer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_sqrelu_quantize_row_only_prepared(
    torch::Tensor h1_raw,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(h1_raw.size(0) % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(h1_raw.size(1) % 128 == 0, "H must be a multiple of 128");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    bool row_rht = false;
    bool col_rht = false;
    parse_localcta_sqrelu_rht_axes(rht_axes, row_rht, col_rht);
    TORCH_CHECK(!col_rht, "square-ReLU row-only producer does not emit a col-RHT view");

    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(M, H, false, h1_raw.device());
    (void)col_fp4;
    (void)col_sc_prepared;
    (void)col_sg;

    launch_localcta_tma_sqrelu_row_prepared_col_outer<false>(
        h1_raw,
        row_fp4, row_sc_prepared,
        col_fp4, col_sc_prepared, col_sc_prepared,
        row_sg, row_sg,
        encode_centric,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        false,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_sqrelu_quantize_row_only_prepared failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc_prepared, row_sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_sqrelu_quantize_col_only_raw_outer(
    torch::Tensor h1_raw,
    bool encode_centric
) {
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "square-ReLU col-only quantizer requires the v4 outer-SG contract");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(h1_raw.size(0) % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(h1_raw.size(1) % 128 == 0, "H must be a multiple of 128");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    auto device = h1_raw.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto col_fp4 = torch::empty({H, M / 2}, opts_fp4);
    auto col_sc_raw = torch::empty({H / 128, M / 64, 512}, opts_fp8);
    auto col_sg_chunks = torch::empty({H / 128, M / 128}, opts_f32);
    auto col_sg_outer = torch::empty({1, H / 256}, opts_f32);

    launch_localcta_sqrelu_quantize_col_only_raw_outer(
        h1_raw, col_fp4, col_sc_raw, col_sg_chunks, encode_centric);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_sqrelu_quantize_col_only_raw_outer failed: ",
                cudaGetErrorString(err));
    finalize_col_quant_contract_v3(col_sc_raw, col_sg_chunks, col_sg_outer);
    return std::make_tuple(col_fp4, col_sc_raw, col_sg_outer);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_sqrelu_deriv_quantize_for_gemm_prepared(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");
    TORCH_CHECK(dh.size(0) % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(dh.size(1) % 128 == 0, "H must be a multiple of 128");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    bool row_rht = false;
    bool col_rht = false;
    parse_localcta_sqrelu_rht_axes(rht_axes, row_rht, col_rht);
    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(M, H, true, dh.device());

    launch_localcta_tma_sqrelu_deriv_prepared<true>(
        dh, h1_raw,
        row_fp4, row_sc_prepared,
        col_fp4, col_sc_prepared,
        row_sg, col_sg,
        encode_centric,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        col_rht,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_sqrelu_deriv_quantize_for_gemm_prepared failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "square-ReLU derivative raw-outer quantizer requires the v4 outer-SG contract");
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");
    TORCH_CHECK(dh.size(0) % 256 == 0 && dh.size(1) % 256 == 0,
                "square-ReLU derivative raw-outer quantizer requires M and H multiples of 256");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    bool row_rht = false;
    bool col_rht = false;
    parse_localcta_sqrelu_rht_axes(rht_axes, row_rht, col_rht);

    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(M, H, true, dh.device());

    auto opts_f32 = torch::dtype(torch::kFloat32).device(dh.device());
    auto row_sg_chunks = torch::empty({M / 128, H / 128}, opts_f32);
    auto col_sg_chunks = torch::empty({H / 128, M / 128}, opts_f32);

    launch_localcta_tma_sqrelu_deriv_raw_outer<true>(
        dh, h1_raw,
        row_fp4, row_sc,
        col_fp4, col_sc,
        row_sg_chunks, col_sg_chunks,
        encode_centric,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        row_rht,
        col_rht,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence_base);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer failed: ",
                cudaGetErrorString(err));
    finalize_row_quant_contract_v3(row_sc, row_sg_chunks, row_sg);
    finalize_col_quant_contract_v3(col_sc, col_sg_chunks, col_sg);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

bool tk_localcta_silu_deriv_split2_supports_rht() {
    return true;
}

bool tk_localcta_split3_supports_paired_rht() {
    return true;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    bool finalize_contract = true,
    bool data_stochastic_rounding = false,
    bool scale_stochastic_rounding = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    std::string data_sr_axes = "both",
    std::string rht_axes = "none",
    bool with_random_sign_mask = false,
    bool derivatives_precomputed = false,
    bool encode_centric = true,
    std::optional<torch::Tensor> persistent_rng_state = std::nullopt
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");
    TORCH_CHECK(dh1.sizes() == dh.sizes() && dh3_out.sizes() == dh.sizes(),
                "dh1/dh3_out must match dh shape");
    TORCH_CHECK(dh1.scalar_type() == torch::kBFloat16 && dh3_out.scalar_type() == torch::kBFloat16,
                "dh1/dh3_out must be bf16");
    TORCH_CHECK(dh1.is_cuda() && dh1.is_contiguous() && dh3_out.is_cuda() && dh3_out.is_contiguous(),
                "dh1/dh3_out must be contiguous CUDA tensors");
    const auto device = dh.device();
    for (const auto& tensor : {
             h3, h1_raw, dh1, dh3_out,
             row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
             row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1,
             row_sg_chunk_0, col_sg_chunk_0, row_sg_chunk_1, col_sg_chunk_1}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == device,
                    "all split2 derivative inputs and outputs must be on ", device);
    }
    const c10::cuda::CUDAGuard device_guard(device);
    const auto set_device_err = cudaSetDevice(dh.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before localCTA v4 split2 quantization: ",
                cudaGetErrorString(set_device_err));

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool contract_already_finalized = false;
    const auto sr_axes = resolve_localcta_data_sr_axes(
        data_stochastic_rounding, data_sr_axes, "localCTA v4 split2 producer");
    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(rht_axes.begin(), rht_axes.end(), '-', '_');
    if (rht_axes == "column" || rht_axes == "columns" || rht_axes == "wgrad") {
        rht_axes = "col";
    } else if (rht_axes == "off" || rht_axes == "0") {
        rht_axes = "none";
    }
    TORCH_CHECK(rht_axes == "none" || rht_axes == "col",
                "localCTA v4 split2 producer supports only column RHT, got ",
                rht_axes);
    const bool col_rht = rht_axes == "col";
    TORCH_CHECK(!with_random_sign_mask || col_rht,
                "localCTA v4 split2 fixed-sign mask requires column RHT");
    TORCH_CHECK(!derivatives_precomputed || col_rht,
                "precomputed split2 derivatives are only supported by the column-RHT route");
    if (col_rht) {
        TORCH_CHECK(finalize_contract,
                    "localCTA v4 split2 column RHT requires a finalized quantization contract");
        TORCH_CHECK(sr_axes.row && !sr_axes.col && !scale_stochastic_rounding,
                    "localCTA v4 split2 column RHT requires row-only data SR and scale SR off");
        TORCH_CHECK(M % 256 == 0 && H % 256 == 0,
                    "localCTA v4 split2 column RHT requires M and H multiples of 256");
        TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                    "localCTA v4 split2 column RHT requires the v4 outer-SG contract");
        TORCH_CHECK(persistent_rng_state.has_value() && persistent_rng_state->defined(),
                    "localCTA v4 split2 column RHT requires explicit persistent RNG state");
        TORCH_CHECK(derivatives_precomputed,
                    "localCTA v4 split2 column RHT requires derivatives_precomputed=true; "
                    "the caller must fill dh1/dh3_out with the production derivative producer");
    }
    const bool has_quantizer_extras =
        sr_axes.row || sr_axes.col || scale_stochastic_rounding || col_rht;

    if (!has_quantizer_extras && use_localcta_v4_tuned_strict_split2()) {
        launch_localcta_silu_deriv_split2_quant_raw_tuned_2cta(
            dh, h3, h1_raw,
            row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
            row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1);
    } else if (!has_quantizer_extras && use_localcta_v4_direct_strict_split2()) {
        launch_localcta_direct_silu_deriv_split_raw<true>(
            dh, h3, h1_raw,
            row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
            row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1);
    } else {
        const bool want_prefinalized_outer_sg =
            finalize_contract && use_localcta_v4_split2_prefinalize_outer_sg();
        const bool used_precomputed_amax = (
            !col_rht &&
            use_localcta_v4_split2_precompute_amax() &&
            launch_silu_deriv_split_with_tile_amax(
                dh, h3, h1_raw, dh1, dh3_out,
                row_sg_chunk_0, col_sg_chunk_0,
                row_sg_chunk_1, col_sg_chunk_1,
                want_prefinalized_outer_sg,
                row_sg_0, col_sg_0, row_sg_1, col_sg_1)
        );
        const bool used_prefinalized_outer_sg =
            used_precomputed_amax && want_prefinalized_outer_sg;
        if (!used_precomputed_amax && !derivatives_precomputed) {
            tk_silu_split::launch_backward(
                reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
                M, H, stream);
        }

        launch_localcta_split2_quant_raw(
            dh1, dh3_out,
            row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
            row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1,
            used_precomputed_amax,
            used_prefinalized_outer_sg,
            row_sg_0, col_sg_0, row_sg_1, col_sg_1,
            torch::Tensor(),
            false,
            data_stochastic_rounding,
            scale_stochastic_rounding,
            rng_seed,
            rng_subsequence_base,
            torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            data_sr_axes,
            persistent_rng_state.value_or(torch::Tensor()),
            col_rht,
            with_random_sign_mask,
            encode_centric);
        if (used_prefinalized_outer_sg) {
            contract_already_finalized = true;
        }
    }
    if (finalize_contract && !contract_already_finalized) {
        finalize_quant_contract_v3_split2(
            row_sc_0, row_sg_chunk_0, row_sg_0, col_sc_0, col_sg_chunk_0, col_sg_0,
            row_sc_1, row_sg_chunk_1, row_sg_1, col_sc_1, col_sg_chunk_1, col_sg_1,
            col_rht);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace failed: ",
                cudaGetErrorString(err));

    if (finalize_contract) {
        return std::make_tuple(
            row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
            row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1
        );
    }
    return std::make_tuple(
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1
    );
}

// Preserve the pre-RHT positional ABI where the argument immediately after
// data_sr_axes was the persistent RNG state.  The pybind overload below only
// participates when that state is explicitly supplied; calls using the new
// RHT keywords resolve to the full entry point above.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace_legacy_state(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    bool finalize_contract,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    std::string data_sr_axes,
    std::optional<torch::Tensor> persistent_rng_state
) {
    return tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
        dh, h3, h1_raw, dh1, dh3_out,
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1,
        row_sg_chunk_0, col_sg_chunk_0, row_sg_chunk_1, col_sg_chunk_1,
        finalize_contract,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rng_seed,
        rng_subsequence_base,
        std::move(data_sr_axes),
        "none",
        false,
        false,
        true,
        std::move(persistent_rng_state));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
        M, H, dh.device());
    return tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
        dh, h3, h1_raw, dh1, dh3_out,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<4>(bufs), std::get<5>(bufs), std::get<6>(bufs), std::get<7>(bufs),
        std::get<8>(bufs), std::get<9>(bufs), std::get<10>(bufs), std::get<11>(bufs),
        std::get<12>(bufs), std::get<13>(bufs), std::get<14>(bufs), std::get<15>(bufs));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_delayed(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(dh.device());
    auto opts_i32 = torch::dtype(torch::kInt32).device(dh.device());
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(dh.device());
    auto delayed_amax = torch::empty({2}, opts_f32);
    auto delayed_sync = torch::empty({0}, opts_i32);
    auto dh13_keepalive = torch::empty({0}, opts_bf16);
    auto fp4_keepalive = torch::empty({0}, opts_fp4);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool collect_delayed_amax = use_localcta_v4_delayed_collect_amax();
    if (collect_delayed_amax) {
        cudaMemsetAsync(delayed_amax.data_ptr<float>(), 0, 2 * sizeof(float), stream);
    }
    bool final_sg_filled = false;

    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
        M, H, dh.device());

    if (use_localcta_v4_tuned_strict_split2()) {
        launch_localcta_silu_deriv_split2_quant_raw_tuned_2cta<true>(
            dh, h3, h1_raw,
            std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
            std::get<12>(bufs), std::get<13>(bufs),
            std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
            std::get<14>(bufs), std::get<15>(bufs),
            delayed_amax);
    } else {
        auto dh1 = torch::empty({M, H}, opts_bf16);
        auto dh3_out = torch::empty({M, H}, opts_bf16);
        const bool used_precomputed_amax = (
            use_localcta_v4_split2_precompute_amax() &&
            launch_silu_deriv_split_with_tile_amax(
                dh, h3, h1_raw, dh1, dh3_out,
                std::get<12>(bufs), std::get<13>(bufs),
                std::get<14>(bufs), std::get<15>(bufs),
                false,
                torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                collect_delayed_amax ? delayed_amax : torch::Tensor(),
                !collect_delayed_amax)
        );
        if (used_precomputed_amax) {
            launch_localcta_split2_quant_raw(
                dh1, dh3_out,
                std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
                std::get<12>(bufs), std::get<13>(bufs),
                std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
                std::get<14>(bufs), std::get<15>(bufs),
                true,
                true,
                std::get<4>(bufs), std::get<5>(bufs), std::get<10>(bufs), std::get<11>(bufs),
                delayed_amax,
                true);
            final_sg_filled = true;
            dh13_keepalive = dh1;
            fp4_keepalive = dh3_out;
        } else {
            launch_localcta_direct_silu_deriv_split_raw<true, true>(
                dh, h3, h1_raw,
                std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
                std::get<12>(bufs), std::get<13>(bufs),
                std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
                std::get<14>(bufs), std::get<15>(bufs),
                delayed_amax);
        }
    }

    if (!final_sg_filled) {
        fill_split2_final_sg_unit(
            std::get<4>(bufs), std::get<5>(bufs),
            std::get<10>(bufs), std::get<11>(bufs));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_silu_deriv_quantize_split_for_gemm_delayed failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<4>(bufs), std::get<5>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<10>(bufs), std::get<11>(bufs),
        delayed_amax, delayed_sync, dh13_keepalive, fp4_keepalive);
}

static void check_split2_tile_amax_tensor(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name
) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.is_contiguous(),
                name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                name, " must be float32");
    TORCH_CHECK(tensor.sizes() == reference.sizes(),
                name, " has shape ", tensor.sizes(),
                ", expected ", reference.sizes());
}

static void check_split2_outer_sg_tensor(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name
) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.is_contiguous(),
                name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                name, " must be float32");
    TORCH_CHECK(tensor.sizes() == reference.sizes(),
                name, " has shape ", tensor.sizes(),
                ", expected ", reference.sizes());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(M, H, dh.device());
    auto cur_row_amax_0 = torch::empty_like(std::get<12>(bufs));
    auto cur_col_amax_0 = torch::empty_like(std::get<13>(bufs));
    auto cur_row_amax_1 = torch::empty_like(std::get<14>(bufs));
    auto cur_col_amax_1 = torch::empty_like(std::get<15>(bufs));

    const bool collected_tile_amax = launch_silu_deriv_split_with_tile_amax(
        dh, h3, h1_raw, dh1, dh3_out,
        cur_row_amax_0, cur_col_amax_0,
        cur_row_amax_1, cur_col_amax_1);
    TORCH_CHECK(collected_tile_amax,
                "localCTA v4 tile-amax collection requires M/H compatible with the split2 tile grid");

    launch_localcta_split2_quant_raw(
        dh1, dh3_out,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<12>(bufs), std::get<13>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<14>(bufs), std::get<15>(bufs),
        true,
        false,
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), false,
        false, false, 0, 0,
        cur_row_amax_0, cur_row_amax_1);

    finalize_quant_contract_v3_split2(
        std::get<1>(bufs), std::get<12>(bufs), std::get<4>(bufs),
        std::get<3>(bufs), std::get<13>(bufs), std::get<5>(bufs),
        std::get<7>(bufs), std::get<14>(bufs), std::get<10>(bufs),
        std::get<9>(bufs), std::get<15>(bufs), std::get<11>(bufs));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<4>(bufs), std::get<5>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<10>(bufs), std::get<11>(bufs),
        cur_row_amax_0, cur_col_amax_0, cur_row_amax_1, cur_col_amax_1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax_outer(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(M, H, dh.device());
    auto cur_row_amax_0 = torch::empty_like(std::get<12>(bufs));
    auto cur_col_amax_0 = torch::empty_like(std::get<13>(bufs));
    auto cur_row_amax_1 = torch::empty_like(std::get<14>(bufs));
    auto cur_col_amax_1 = torch::empty_like(std::get<15>(bufs));

    const bool collected_tile_amax = launch_silu_deriv_split_with_tile_amax(
        dh, h3, h1_raw, dh1, dh3_out,
        cur_row_amax_0, cur_col_amax_0,
        cur_row_amax_1, cur_col_amax_1);
    TORCH_CHECK(collected_tile_amax,
                "localCTA v4 tile-amax collection requires M/H compatible with the split2 tile grid");

    launch_localcta_split2_quant_raw(
        dh1, dh3_out,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<12>(bufs), std::get<13>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<14>(bufs), std::get<15>(bufs),
        true,
        false,
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), false,
        false, false, 0, 0,
        cur_row_amax_0, cur_row_amax_1);

    finalize_quant_contract_v3_split2(
        std::get<1>(bufs), std::get<12>(bufs), std::get<4>(bufs),
        std::get<3>(bufs), std::get<13>(bufs), std::get<5>(bufs),
        std::get<7>(bufs), std::get<14>(bufs), std::get<10>(bufs),
        std::get<9>(bufs), std::get<15>(bufs), std::get<11>(bufs));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax_outer failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<4>(bufs), std::get<5>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<10>(bufs), std::get<11>(bufs),
        cur_row_amax_0, cur_col_amax_0, cur_row_amax_1, cur_col_amax_1,
        std::get<4>(bufs), std::get<5>(bufs), std::get<10>(bufs), std::get<11>(bufs));
}

using Split2DelayedOuterResult = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor prev_row_amax_0,
    torch::Tensor prev_col_amax_0,
    torch::Tensor prev_row_amax_1,
    torch::Tensor prev_col_amax_1
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(M, H, dh.device());
    auto cur_row_amax_0 = torch::empty_like(std::get<12>(bufs));
    auto cur_col_amax_0 = torch::empty_like(std::get<13>(bufs));
    auto cur_row_amax_1 = torch::empty_like(std::get<14>(bufs));
    auto cur_col_amax_1 = torch::empty_like(std::get<15>(bufs));

    check_split2_tile_amax_tensor(prev_row_amax_0, std::get<12>(bufs), "prev_row_amax_0");
    check_split2_tile_amax_tensor(prev_col_amax_0, std::get<13>(bufs), "prev_col_amax_0");
    check_split2_tile_amax_tensor(prev_row_amax_1, std::get<14>(bufs), "prev_row_amax_1");
    check_split2_tile_amax_tensor(prev_col_amax_1, std::get<15>(bufs), "prev_col_amax_1");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
        M, H, stream);

    launch_localcta_split2_quant_raw(
        dh1, dh3_out,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<12>(bufs), std::get<13>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<14>(bufs), std::get<15>(bufs),
        true,
        false,
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), false,
        false, false, 0, 0,
        prev_row_amax_0, prev_row_amax_1,
        cur_row_amax_0, cur_col_amax_0,
        cur_row_amax_1, cur_col_amax_1);

    finalize_quant_contract_v3_split2(
        std::get<1>(bufs), std::get<12>(bufs), std::get<4>(bufs),
        std::get<3>(bufs), std::get<13>(bufs), std::get<5>(bufs),
        std::get<7>(bufs), std::get<14>(bufs), std::get<10>(bufs),
        std::get<9>(bufs), std::get<15>(bufs), std::get<11>(bufs));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<4>(bufs), std::get<5>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<10>(bufs), std::get<11>(bufs),
        cur_row_amax_0, cur_col_amax_0, cur_row_amax_1, cur_col_amax_1);
}

Split2DelayedOuterResult
tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor prev_row_amax_0,
    torch::Tensor prev_col_amax_0,
    torch::Tensor prev_row_amax_1,
    torch::Tensor prev_col_amax_1,
    torch::Tensor prev_row_sg_0,
    torch::Tensor prev_col_sg_0,
    torch::Tensor prev_row_sg_1,
    torch::Tensor prev_col_sg_1,
    torch::Tensor cur_row_amax_0,
    torch::Tensor cur_row_amax_1,
    torch::Tensor cur_row_sg_0,
    torch::Tensor cur_col_sg_0,
    torch::Tensor cur_row_sg_1,
    torch::Tensor cur_col_sg_1
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");
    TORCH_CHECK(dh1.sizes() == dh.sizes() && dh3_out.sizes() == dh.sizes(),
                "dh1/dh3_out must match dh shape");
    TORCH_CHECK(dh1.scalar_type() == torch::kBFloat16 && dh3_out.scalar_type() == torch::kBFloat16,
                "dh1/dh3_out must be bf16");
    TORCH_CHECK(dh1.is_cuda() && dh1.is_contiguous() && dh3_out.is_cuda() && dh3_out.is_contiguous(),
                "dh1/dh3_out must be contiguous CUDA tensors");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);

    check_split2_tile_amax_tensor(prev_row_amax_0, row_sg_chunk_0, "prev_row_amax_0");
    check_split2_tile_amax_tensor(prev_row_amax_1, row_sg_chunk_1, "prev_row_amax_1");
    TORCH_CHECK(prev_col_amax_0.defined() && prev_col_amax_1.defined(),
                "prev_col_amax tensors must be supplied for delayed-outer state compatibility");
    check_split2_outer_sg_tensor(prev_row_sg_0, row_sg_0, "prev_row_sg_0");
    check_split2_outer_sg_tensor(prev_col_sg_0, col_sg_0, "prev_col_sg_0");
    check_split2_outer_sg_tensor(prev_row_sg_1, row_sg_1, "prev_row_sg_1");
    check_split2_outer_sg_tensor(prev_col_sg_1, col_sg_1, "prev_col_sg_1");
    check_split2_tile_amax_tensor(cur_row_amax_0, row_sg_chunk_0, "cur_row_amax_0");
    check_split2_tile_amax_tensor(cur_row_amax_1, row_sg_chunk_1, "cur_row_amax_1");
    check_split2_outer_sg_tensor(cur_row_sg_0, row_sg_0, "cur_row_sg_0");
    check_split2_outer_sg_tensor(cur_col_sg_0, col_sg_0, "cur_col_sg_0");
    check_split2_outer_sg_tensor(cur_row_sg_1, row_sg_1, "cur_row_sg_1");
    check_split2_outer_sg_tensor(cur_col_sg_1, col_sg_1, "cur_col_sg_1");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(cur_row_sg_0.data_ptr<float>(), 0, cur_row_sg_0.numel() * sizeof(float), stream);
    cudaMemsetAsync(cur_col_sg_0.data_ptr<float>(), 0, cur_col_sg_0.numel() * sizeof(float), stream);
    cudaMemsetAsync(cur_row_sg_1.data_ptr<float>(), 0, cur_row_sg_1.numel() * sizeof(float), stream);
    cudaMemsetAsync(cur_col_sg_1.data_ptr<float>(), 0, cur_col_sg_1.numel() * sizeof(float), stream);

    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
        M, H, stream);

    launch_localcta_split2_quant_raw(
        dh1, dh3_out,
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1,
        true,
        true,
        prev_row_sg_0, prev_col_sg_0, prev_row_sg_1, prev_col_sg_1,
        torch::Tensor(), false,
        false, false, 0, 0,
        prev_row_amax_0, prev_row_amax_1,
        cur_row_amax_0, torch::Tensor(),
        cur_row_amax_1, torch::Tensor(),
        cur_row_sg_0, cur_col_sg_0,
        cur_row_sg_1, cur_col_sg_1);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_launch_inplace failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0,
        prev_row_sg_0, prev_col_sg_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1,
        prev_row_sg_1, prev_col_sg_1,
        cur_row_amax_0, cur_row_amax_0, cur_row_amax_1, cur_row_amax_1,
        cur_row_sg_0, cur_col_sg_0, cur_row_sg_1, cur_col_sg_1);
}

Split2DelayedOuterResult
tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4_0,
    torch::Tensor row_sc_0,
    torch::Tensor col_fp4_0,
    torch::Tensor col_sc_0,
    torch::Tensor row_sg_0,
    torch::Tensor col_sg_0,
    torch::Tensor row_fp4_1,
    torch::Tensor row_sc_1,
    torch::Tensor col_fp4_1,
    torch::Tensor col_sc_1,
    torch::Tensor row_sg_1,
    torch::Tensor col_sg_1,
    torch::Tensor row_sg_chunk_0,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor row_sg_chunk_1,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor prev_row_amax_0,
    torch::Tensor prev_col_amax_0,
    torch::Tensor prev_row_amax_1,
    torch::Tensor prev_col_amax_1,
    torch::Tensor prev_row_sg_0,
    torch::Tensor prev_col_sg_0,
    torch::Tensor prev_row_sg_1,
    torch::Tensor prev_col_sg_1
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");
    TORCH_CHECK(dh1.sizes() == dh.sizes() && dh3_out.sizes() == dh.sizes(),
                "dh1/dh3_out must match dh shape");
    TORCH_CHECK(dh1.scalar_type() == torch::kBFloat16 && dh3_out.scalar_type() == torch::kBFloat16,
                "dh1/dh3_out must be bf16");
    TORCH_CHECK(dh1.is_cuda() && dh1.is_contiguous() && dh3_out.is_cuda() && dh3_out.is_contiguous(),
                "dh1/dh3_out must be contiguous CUDA tensors");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);

    check_split2_tile_amax_tensor(prev_row_amax_0, row_sg_chunk_0, "prev_row_amax_0");
    check_split2_tile_amax_tensor(prev_row_amax_1, row_sg_chunk_1, "prev_row_amax_1");
    TORCH_CHECK(prev_col_amax_0.defined() && prev_col_amax_1.defined(),
                "prev_col_amax tensors must be supplied for delayed-outer state compatibility");
    check_split2_outer_sg_tensor(prev_row_sg_0, row_sg_0, "prev_row_sg_0");
    check_split2_outer_sg_tensor(prev_col_sg_0, col_sg_0, "prev_col_sg_0");
    check_split2_outer_sg_tensor(prev_row_sg_1, row_sg_1, "prev_row_sg_1");
    check_split2_outer_sg_tensor(prev_col_sg_1, col_sg_1, "prev_col_sg_1");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
        M, H, stream);

    launch_localcta_split2_quant_raw(
        dh1, dh3_out,
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1,
        true,
        true,
        prev_row_sg_0, prev_col_sg_0, prev_row_sg_1, prev_col_sg_1,
        torch::Tensor(), false,
        false, false, 0, 0,
        prev_row_amax_0, prev_row_amax_1);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect_launch_inplace failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0,
        prev_row_sg_0, prev_col_sg_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1,
        prev_row_sg_1, prev_col_sg_1,
        prev_row_amax_0, prev_col_amax_0, prev_row_amax_1, prev_col_amax_1,
        prev_row_sg_0, prev_col_sg_0, prev_row_sg_1, prev_col_sg_1);
}

Split2DelayedOuterResult
tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor prev_row_amax_0,
    torch::Tensor prev_col_amax_0,
    torch::Tensor prev_row_amax_1,
    torch::Tensor prev_col_amax_1,
    torch::Tensor prev_row_sg_0,
    torch::Tensor prev_col_sg_0,
    torch::Tensor prev_row_sg_1,
    torch::Tensor prev_col_sg_1
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(M, H, dh.device());
    auto cur_row_amax_0 = torch::empty_like(std::get<12>(bufs));
    auto cur_row_amax_1 = torch::empty_like(std::get<14>(bufs));
    auto cur_row_sg_0 = torch::empty_like(prev_row_sg_0);
    auto cur_col_sg_0 = torch::empty_like(prev_col_sg_0);
    auto cur_row_sg_1 = torch::empty_like(prev_row_sg_1);
    auto cur_col_sg_1 = torch::empty_like(prev_col_sg_1);
    return tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_launch_inplace(
        dh, h3, h1_raw, dh1, dh3_out,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<4>(bufs), std::get<5>(bufs), std::get<6>(bufs), std::get<7>(bufs),
        std::get<8>(bufs), std::get<9>(bufs), std::get<10>(bufs), std::get<11>(bufs),
        std::get<12>(bufs), std::get<13>(bufs), std::get<14>(bufs), std::get<15>(bufs),
        prev_row_amax_0, prev_col_amax_0, prev_row_amax_1, prev_col_amax_1,
        prev_row_sg_0, prev_col_sg_0, prev_row_sg_1, prev_col_sg_1,
        cur_row_amax_0, cur_row_amax_1,
        cur_row_sg_0, cur_col_sg_0, cur_row_sg_1, cur_col_sg_1);
}

Split2DelayedOuterResult
tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor prev_row_amax_0,
    torch::Tensor prev_col_amax_0,
    torch::Tensor prev_row_amax_1,
    torch::Tensor prev_col_amax_1,
    torch::Tensor prev_row_sg_0,
    torch::Tensor prev_col_sg_0,
    torch::Tensor prev_row_sg_1,
    torch::Tensor prev_col_sg_1
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(M, H, dh.device());
    return tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect_launch_inplace(
        dh, h3, h1_raw, dh1, dh3_out,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<4>(bufs), std::get<5>(bufs), std::get<6>(bufs), std::get<7>(bufs),
        std::get<8>(bufs), std::get<9>(bufs), std::get<10>(bufs), std::get<11>(bufs),
        std::get<12>(bufs), std::get<13>(bufs), std::get<14>(bufs), std::get<15>(bufs),
        prev_row_amax_0, prev_col_amax_0, prev_row_amax_1, prev_col_amax_1,
        prev_row_sg_0, prev_col_sg_0, prev_row_sg_1, prev_col_sg_1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_split_collect(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor prev_row_amax_0,
    torch::Tensor prev_col_amax_0,
    torch::Tensor prev_row_amax_1,
    torch::Tensor prev_col_amax_1,
    torch::Tensor prev_row_sg_0,
    torch::Tensor prev_col_sg_0,
    torch::Tensor prev_row_sg_1,
    torch::Tensor prev_col_sg_1
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto bufs = tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(M, H, dh.device());
    auto cur_row_amax_0 = torch::empty_like(std::get<12>(bufs));
    auto cur_col_amax_0 = torch::empty_like(std::get<13>(bufs));
    auto cur_row_amax_1 = torch::empty_like(std::get<14>(bufs));
    auto cur_col_amax_1 = torch::empty_like(std::get<15>(bufs));
    auto cur_row_sg_0 = torch::empty_like(prev_row_sg_0);
    auto cur_col_sg_0 = torch::empty_like(prev_col_sg_0);
    auto cur_row_sg_1 = torch::empty_like(prev_row_sg_1);
    auto cur_col_sg_1 = torch::empty_like(prev_col_sg_1);

    check_split2_tile_amax_tensor(prev_row_amax_0, std::get<12>(bufs), "prev_row_amax_0");
    check_split2_tile_amax_tensor(prev_col_amax_0, std::get<13>(bufs), "prev_col_amax_0");
    check_split2_tile_amax_tensor(prev_row_amax_1, std::get<14>(bufs), "prev_row_amax_1");
    check_split2_tile_amax_tensor(prev_col_amax_1, std::get<15>(bufs), "prev_col_amax_1");
    check_split2_outer_sg_tensor(prev_row_sg_0, std::get<4>(bufs), "prev_row_sg_0");
    check_split2_outer_sg_tensor(prev_col_sg_0, std::get<5>(bufs), "prev_col_sg_0");
    check_split2_outer_sg_tensor(prev_row_sg_1, std::get<10>(bufs), "prev_row_sg_1");
    check_split2_outer_sg_tensor(prev_col_sg_1, std::get<11>(bufs), "prev_col_sg_1");

    const bool collected_current_state = launch_silu_deriv_split_with_tile_amax(
        dh, h3, h1_raw, dh1, dh3_out,
        cur_row_amax_0, cur_col_amax_0,
        cur_row_amax_1, cur_col_amax_1,
        true,
        cur_row_sg_0, cur_col_sg_0, cur_row_sg_1, cur_col_sg_1,
        torch::Tensor(),
        false,
        true);
    TORCH_CHECK(collected_current_state,
                "localCTA v4 delayed-outer split-collect requires M/H compatible with the split2 tile grid");

    launch_localcta_split2_quant_raw(
        dh1, dh3_out,
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        std::get<12>(bufs), std::get<13>(bufs),
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        std::get<14>(bufs), std::get<15>(bufs),
        true,
        true,
        prev_row_sg_0, prev_col_sg_0, prev_row_sg_1, prev_col_sg_1,
        torch::Tensor(), false,
        false, false, 0, 0,
        prev_row_amax_0, prev_row_amax_1);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_split_collect failed: ",
                cudaGetErrorString(err));
    return std::make_tuple(
        std::get<0>(bufs), std::get<1>(bufs), std::get<2>(bufs), std::get<3>(bufs),
        prev_row_sg_0, prev_col_sg_0,
        std::get<6>(bufs), std::get<7>(bufs), std::get<8>(bufs), std::get<9>(bufs),
        prev_row_sg_1, prev_col_sg_1,
        cur_row_amax_0, cur_col_amax_0, cur_row_amax_1, cur_col_amax_1,
        cur_row_sg_0, cur_col_sg_0, cur_row_sg_1, cur_col_sg_1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_split_for_gemm_prepared(
    torch::Tensor input0,
    torch::Tensor input1
) {
    for (const auto &input : {input0, input1}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split inputs must be contiguous [N_i, K]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(0) % 128 == 0, "split rows must be multiples of 128");
    }
    TORCH_CHECK(input0.size(1) == input1.size(1),
                "split inputs must have the same K dimension");

    const int64_t rows0 = input0.size(0);
    const int64_t rows1 = input1.size(0);
    const int64_t total_rows = rows0 + rows1;
    const int64_t K = input0.size(1);
    auto device = input0.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({total_rows, K / 2}, opts_fp4);
    auto row_sc_prepared_cat = torch::empty({total_rows / 128, K / 64, 512}, opts_fp8);
    auto row_sg_cat = torch::empty({total_rows / 128, K / 128}, opts_f32);

    auto row_fp4_0 = row_fp4_cat.narrow(0, 0, rows0);
    auto row_fp4_1 = row_fp4_cat.narrow(0, rows0, rows1);
    auto row_sc_0 = row_sc_prepared_cat.narrow(0, 0, rows0 / 128);
    auto row_sc_1 = row_sc_prepared_cat.narrow(0, rows0 / 128, rows1 / 128);
    auto row_sg_0 = row_sg_cat.narrow(0, 0, rows0 / 128);
    auto row_sg_1 = row_sg_cat.narrow(0, rows0 / 128, rows1 / 128);

    auto col_fp4_0 = torch::empty({K, rows0 / 2}, opts_fp4);
    auto col_fp4_1 = torch::empty({K, rows1 / 2}, opts_fp4);
    auto col_sc_0 = torch::empty({K / 128, rows0 / 64, 512}, opts_fp8);
    auto col_sc_1 = torch::empty({K / 128, rows1 / 64, 512}, opts_fp8);
    auto col_sg_0 = torch::empty({K / 128, rows0 / 128}, opts_f32);
    auto col_sg_1 = torch::empty({K / 128, rows1 / 128}, opts_f32);

    tk_localcta_quantize_for_gemm_prepared_launch(
        input0, true, true, row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0);
    tk_localcta_quantize_for_gemm_prepared_launch(
        input1, true, true, row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1);

    std::vector<torch::Tensor> col_fp4_list{col_fp4_0, col_fp4_1};
    std::vector<torch::Tensor> col_sc_prepared_list{col_sc_0, col_sc_1};
    std::vector<torch::Tensor> row_sg_parts{row_sg_0, row_sg_1};
    std::vector<torch::Tensor> col_sg_list{col_sg_0, col_sg_1};
    auto col_fp4_cat = torch::cat(col_fp4_list, 1);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 1);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_list, col_sc_prepared_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           col_fp4_cat, col_sc_prepared_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split3_for_gemm(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    bool data_stochastic_rounding = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    std::string data_sr_axes = "both",
    std::optional<torch::Tensor> persistent_rng_state = std::nullopt,
    std::string rht_axes = "none",
    bool with_random_sign_mask = false,
    bool encode_centric = true
) {
    for (const auto &input : {input0, input1, input2}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.stride(1) == 1,
                    "split inputs must have contiguous last dimension [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");
    const auto device = input0.device();
    TORCH_CHECK(input1.device() == device && input2.device() == device,
                "all split3 inputs must be on ", device);
    const c10::cuda::CUDAGuard device_guard(device);
    const auto set_device_err = cudaSetDevice(input0.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before localCTA split3 quantization: ",
                cudaGetErrorString(set_device_err));
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);
    const int64_t total_n = n0 + n1 + n2;
    const int64_t m_sg_tiles = outer_sg_tiles_128(M);
    const int64_t n0_sg_tiles = outer_sg_tiles_128(n0);
    const int64_t n1_sg_tiles = outer_sg_tiles_128(n1);
    const int64_t n2_sg_tiles = outer_sg_tiles_128(n2);
    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(rht_axes.begin(), rht_axes.end(), '-', '_');
    if (rht_axes == "column" || rht_axes == "columns" || rht_axes == "wgrad") {
        rht_axes = "col";
    } else if (rht_axes == "off" || rht_axes == "0") {
        rht_axes = "none";
    }
    TORCH_CHECK(rht_axes == "none" || rht_axes == "col",
                "localCTA v4 split3 producer supports only column RHT, got ",
                rht_axes);
    const bool col_rht = rht_axes == "col";
    TORCH_CHECK(!with_random_sign_mask || col_rht,
                "localCTA v4 split3 fixed-sign mask requires column RHT");
    if (col_rht) {
        const auto sr_axes = resolve_localcta_data_sr_axes(
            data_stochastic_rounding, data_sr_axes,
            "localCTA v4 split3 column-RHT producer");
        TORCH_CHECK(sr_axes.row && !sr_axes.col,
                    "localCTA v4 split3 column RHT requires row-only data SR");
        TORCH_CHECK(persistent_rng_state.has_value() && persistent_rng_state->defined(),
                    "localCTA v4 split3 column RHT requires explicit persistent RNG state");
        TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                    "localCTA v4 split3 column RHT requires the v4 outer-SG contract");
        TORCH_CHECK(M % 256 == 0 && n0 % 256 == 0 && n1 % 256 == 0 && n2 % 256 == 0,
                    "localCTA v4 split3 column RHT requires M and all split widths "
                    "to be multiples of 256");
    }
    TORCH_CHECK(
        !data_stochastic_rounding ||
            get_v3_contract_mode() != V3ContractMode::TileGrid256,
        "split3 data SR requires the localCTA v4 outer-scale contract");
    if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
        auto [row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0] =
            tk_localcta_quantize_for_gemm(input0, true, true);
        auto [row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1] =
            tk_localcta_quantize_for_gemm(input1, true, true);
        auto [row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_2, col_sg_2] =
            tk_localcta_quantize_for_gemm(input2, true, true);

        auto row_fp4_cat = torch::cat({row_fp4_0, row_fp4_1, row_fp4_2}, 1);
        auto row_sc_cat = torch::cat({row_sc_0, row_sc_1, row_sc_2}, 1);
        auto row_sg_cat = torch::cat({row_sg_0, row_sg_1, row_sg_2}, 1);
        auto col_fp4_cat = torch::cat({col_fp4_0, col_fp4_1, col_fp4_2}, 0);
        auto col_sc_cat = torch::cat({col_sc_0, col_sc_1, col_sc_2}, 0);
        auto col_sg_cat = torch::cat({col_sg_0, col_sg_1, col_sg_2}, 0);

        std::vector<torch::Tensor> row_fp4_list{row_fp4_0, row_fp4_1, row_fp4_2};
        std::vector<torch::Tensor> row_sc_list{row_sc_0, row_sc_1, row_sc_2};
        std::vector<torch::Tensor> row_sg_list{row_sg_0, row_sg_1, row_sg_2};
        std::vector<torch::Tensor> col_fp4_list{col_fp4_0, col_fp4_1, col_fp4_2};
        std::vector<torch::Tensor> col_sc_list{col_sc_0, col_sc_1, col_sc_2};
        std::vector<torch::Tensor> col_sg_list{col_sg_0, col_sg_1, col_sg_2};

        return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                               col_fp4_list, col_sc_list, col_sg_list,
                               row_fp4_cat, row_sc_cat, row_sg_cat,
                               col_fp4_cat, col_sc_cat, col_sg_cat);
    }

    const auto multiinput_mode = get_v3_split3_multiinput_quant_mode();

    if (multiinput_mode == V3MultiInputQuantMode::Loop &&
        !data_stochastic_rounding) {
        auto [row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0] =
            tk_localcta_quantize_for_gemm(input0, true, true);
        auto [row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1] =
            tk_localcta_quantize_for_gemm(input1, true, true);
        auto [row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_2, col_sg_2] =
            tk_localcta_quantize_for_gemm(input2, true, true);

        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        auto row_fp4_cat = torch::cat({row_fp4_0, row_fp4_1, row_fp4_2}, 1);
        auto row_sc_cat = torch::cat({row_sc_0, row_sc_1, row_sc_2}, 1);
        auto row_sg_cat = torch::empty({0}, opts_f32);
        if (use_v4_split3_emit_row_sg_cat()) {
            row_sg_cat = torch::empty({m_sg_tiles, 3}, opts_f32);
            copy_cat_dim1_contiguous({row_sg_0, row_sg_1, row_sg_2}, row_sg_cat);
        }
        auto col_fp4_cat = torch::cat({col_fp4_0, col_fp4_1, col_fp4_2}, 0);
        auto col_sc_cat = torch::cat({col_sc_0, col_sc_1, col_sc_2}, 0);
        auto col_sg_cat = torch::cat({col_sg_0, col_sg_1, col_sg_2}, 1);

        std::vector<torch::Tensor> row_fp4_list{row_fp4_0, row_fp4_1, row_fp4_2};
        std::vector<torch::Tensor> row_sc_list{row_sc_0, row_sc_1, row_sc_2};
        std::vector<torch::Tensor> row_sg_list{row_sg_0, row_sg_1, row_sg_2};
        std::vector<torch::Tensor> col_fp4_list{col_fp4_0, col_fp4_1, col_fp4_2};
        std::vector<torch::Tensor> col_sc_list{col_sc_0, col_sc_1, col_sc_2};
        std::vector<torch::Tensor> col_sg_list{col_sg_0, col_sg_1, col_sg_2};

        return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                               col_fp4_list, col_sc_list, col_sg_list,
                               row_fp4_cat, row_sc_cat, row_sg_cat,
                               col_fp4_cat, col_sc_cat, col_sg_cat);
    }

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({M, total_n / 2}, opts_fp4);
    auto row_sc_cat = torch::empty({M / 128, total_n / 64, 512}, opts_fp8);
    auto col_fp4_cat = torch::empty({total_n, M / 2}, opts_fp4);
    auto col_sc_cat = torch::empty({total_n / 128, M / 64, 512}, opts_fp8);
    auto col_sg_cat = torch::empty({1, n0_sg_tiles + n1_sg_tiles + n2_sg_tiles}, opts_f32);

    auto row_fp4_0 = row_fp4_cat.narrow(1, 0, n0 / 2);
    auto row_fp4_1 = row_fp4_cat.narrow(1, n0 / 2, n1 / 2);
    auto row_fp4_2 = row_fp4_cat.narrow(1, (n0 + n1) / 2, n2 / 2);
    auto row_sc_0 = row_sc_cat.narrow(1, 0, n0 / 64);
    auto row_sc_1 = row_sc_cat.narrow(1, n0 / 64, n1 / 64);
    auto row_sc_2 = row_sc_cat.narrow(1, (n0 + n1) / 64, n2 / 64);
    auto col_fp4_0 = col_fp4_cat.narrow(0, 0, n0);
    auto col_fp4_1 = col_fp4_cat.narrow(0, n0, n1);
    auto col_fp4_2 = col_fp4_cat.narrow(0, n0 + n1, n2);
    auto col_sc_0 = col_sc_cat.narrow(0, 0, n0 / 128);
    auto col_sc_1 = col_sc_cat.narrow(0, n0 / 128, n1 / 128);
    auto col_sc_2 = col_sc_cat.narrow(0, (n0 + n1) / 128, n2 / 128);
    auto row_sg_0 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto row_sg_1 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto row_sg_2 = torch::empty({m_sg_tiles, 1}, opts_f32);
    if (col_rht) {
        // A concatenated carrier has one row outer scale.  Preserve that alias
        // contract directly instead of copying the same values three times.
        row_sg_1 = row_sg_0;
        row_sg_2 = row_sg_0;
    }
    auto col_sg_0 = col_sg_cat.narrow(1, 0, n0_sg_tiles);
    auto col_sg_1 = col_sg_cat.narrow(1, n0_sg_tiles, n1_sg_tiles);
    auto col_sg_2 = col_sg_cat.narrow(1, n0_sg_tiles + n1_sg_tiles, n2_sg_tiles);

    auto row_sg_chunk_0 = torch::empty({M / 128, n0 / 128}, opts_f32);
    auto row_sg_chunk_1 = torch::empty({M / 128, n1 / 128}, opts_f32);
    auto row_sg_chunk_2 = torch::empty({M / 128, n2 / 128}, opts_f32);
    auto col_sg_chunk_0 = torch::empty({n0 / 128, M / 128}, opts_f32);
    auto col_sg_chunk_1 = torch::empty({n1 / 128, M / 128}, opts_f32);
    auto col_sg_chunk_2 = torch::empty({n2 / 128, M / 128}, opts_f32);

    if (col_rht) {
        launch_localcta_split3_quant_raw(
            input0, input1, input2,
            row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
            row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1,
            row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_chunk_2, col_sg_chunk_2,
            data_stochastic_rounding,
            rng_seed,
            rng_subsequence_base,
            data_sr_axes,
            persistent_rng_state.value_or(torch::Tensor()),
            true,
            with_random_sign_mask,
            encode_centric);
        finalize_quant_contract_v3_split3(
            row_sc_0, row_sg_chunk_0, row_sg_0, col_sc_0, col_sg_chunk_0, col_sg_0,
            row_sc_1, row_sg_chunk_1, row_sg_1, col_sc_1, col_sg_chunk_1, col_sg_1,
            row_sc_2, row_sg_chunk_2, row_sg_2, col_sc_2, col_sg_chunk_2, col_sg_2,
            true);
    } else if (use_v4_split3_two_phase_quant() || data_stochastic_rounding) {
        if (use_localcta_v4_split3_direct_final_sg_scan()) {
            launch_scan_split3_sg_direct_final(
                input0, input1, input2,
                row_sg_0, col_sg_0,
                row_sg_1, col_sg_1,
                row_sg_2, col_sg_2);
        } else {
            launch_scan_split3_sg(
                input0, input1, input2,
                row_sg_chunk_0, col_sg_chunk_0,
                row_sg_chunk_1, col_sg_chunk_1,
                row_sg_chunk_2, col_sg_chunk_2);

            if (use_localcta_v4_split3_fused_sg_reduce()) {
                launch_reduce_row_col_sg_split3(
                    row_sg_chunk_0, row_sg_0,
                    row_sg_chunk_1, row_sg_1,
                    row_sg_chunk_2, row_sg_2,
                    col_sg_chunk_0, col_sg_0,
                    col_sg_chunk_1, col_sg_1,
                    col_sg_chunk_2, col_sg_2);
            } else {
                auto stream = at::cuda::getCurrentCUDAStream();
                const int64_t row_tiles = m_sg_tiles;
                dim3 row_grid(static_cast<unsigned int>(row_tiles), 3u);
                reduce_row_sg_split3_kernel<<<row_grid, 256, 0, stream>>>(
                    row_sg_chunk_0.data_ptr<float>(),
                    row_sg_0.data_ptr<float>(),
                    static_cast<int>(row_sg_chunk_0.size(1)),
                    row_sg_chunk_1.data_ptr<float>(),
                    row_sg_1.data_ptr<float>(),
                    static_cast<int>(row_sg_chunk_1.size(1)),
                    row_sg_chunk_2.data_ptr<float>(),
                    row_sg_2.data_ptr<float>(),
                    static_cast<int>(row_sg_chunk_2.size(1)));
                {
                    cudaError_t err = cudaGetLastError();
                    TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_split3_kernel failed: ", cudaGetErrorString(err));
                }

                const int k_chunks_0 = static_cast<int>(col_sg_chunk_0.size(0));
                const int k_chunks_1 = static_cast<int>(col_sg_chunk_1.size(0));
                const int k_chunks_2 = static_cast<int>(col_sg_chunk_2.size(0));
                const int max_col_tiles = std::max(
                    std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2),
                    (k_chunks_2 + 1) / 2);
                dim3 col_reduce_grid(static_cast<unsigned int>(max_col_tiles), 3u);
                reduce_col_sg_tiles_split3_kernel<<<col_reduce_grid, 256, 0, stream>>>(
                    col_sg_chunk_0.data_ptr<float>(),
                    col_sg_0.data_ptr<float>(),
                    k_chunks_0,
                    static_cast<int>(col_sg_chunk_0.size(1)),
                    col_sg_chunk_1.data_ptr<float>(),
                    col_sg_1.data_ptr<float>(),
                    k_chunks_1,
                    col_sg_chunk_2.data_ptr<float>(),
                    col_sg_2.data_ptr<float>(),
                    k_chunks_2);
                {
                    cudaError_t err = cudaGetLastError();
                    TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_split3_kernel failed: ", cudaGetErrorString(err));
                }
            }
        }

        launch_localcta_split3_quant_final_sg(
            input0, input1, input2,
            row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
            row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1,
            row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_2, col_sg_2,
            torch::Tensor(), 0,
            data_stochastic_rounding, false,
            rng_seed, rng_subsequence_base, data_sr_axes,
            persistent_rng_state.value_or(torch::Tensor()));
    } else {
        launch_localcta_split3_quant_raw(
            input0, input1, input2,
            row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
            row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1,
            row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_chunk_2, col_sg_chunk_2);
        if (multiinput_mode == V3MultiInputQuantMode::SplitFinalize) {
        finalize_quant_contract_v3_split3(
            row_sc_0, row_sg_chunk_0, row_sg_0, col_sc_0, col_sg_chunk_0, col_sg_0,
            row_sc_1, row_sg_chunk_1, row_sg_1, col_sc_1, col_sg_chunk_1, col_sg_1,
            row_sc_2, row_sg_chunk_2, row_sg_2, col_sc_2, col_sg_chunk_2, col_sg_2);
        } else if (multiinput_mode == V3MultiInputQuantMode::ColSplitFinalize) {
        finalize_row_quant_contract_v3_strided(row_sc_0, row_sg_chunk_0, row_sg_0);
        finalize_row_quant_contract_v3_strided(row_sc_1, row_sg_chunk_1, row_sg_1);
        finalize_row_quant_contract_v3_strided(row_sc_2, row_sg_chunk_2, row_sg_2);
        finalize_col_quant_contract_v3_split3(
            col_sc_0, col_sg_chunk_0, col_sg_0,
            col_sc_1, col_sg_chunk_1, col_sg_1,
            col_sc_2, col_sg_chunk_2, col_sg_2);
        } else if (multiinput_mode == V3MultiInputQuantMode::RowColSplitFinalize) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int64_t row_tiles = m_sg_tiles;
        dim3 row_grid(static_cast<unsigned int>(row_tiles), 3u);
        finalize_row_sc_split3_kernel<<<row_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_0.data_ptr()),
            row_sg_chunk_0.data_ptr<float>(),
            row_sg_0.data_ptr<float>(),
            static_cast<int>(row_sc_0.size(1)),
            static_cast<int>(row_sg_chunk_0.size(1)),
            row_sc_0.stride(0),
            row_sc_0.stride(1),
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_1.data_ptr()),
            row_sg_chunk_1.data_ptr<float>(),
            row_sg_1.data_ptr<float>(),
            static_cast<int>(row_sc_1.size(1)),
            static_cast<int>(row_sg_chunk_1.size(1)),
            row_sc_1.stride(0),
            row_sc_1.stride(1),
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_2.data_ptr()),
            row_sg_chunk_2.data_ptr<float>(),
            row_sg_2.data_ptr<float>(),
            static_cast<int>(row_sc_2.size(1)),
            static_cast<int>(row_sg_chunk_2.size(1)),
            row_sc_2.stride(0),
            row_sc_2.stride(1));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "finalize_row_sc_split3_kernel failed: ", cudaGetErrorString(err));
        }
        finalize_col_quant_contract_v3_split3_fused(
            col_sc_0, col_sg_chunk_0, col_sg_0,
            col_sc_1, col_sg_chunk_1, col_sg_1,
            col_sc_2, col_sg_chunk_2, col_sg_2);
        } else {
        finalize_row_quant_contract_v3_strided(row_sc_0, row_sg_chunk_0, row_sg_0);
        finalize_row_quant_contract_v3_strided(row_sc_1, row_sg_chunk_1, row_sg_1);
        finalize_row_quant_contract_v3_strided(row_sc_2, row_sg_chunk_2, row_sg_2);
        finalize_col_quant_contract_v3(col_sc_0, col_sg_chunk_0, col_sg_0);
        finalize_col_quant_contract_v3(col_sc_1, col_sg_chunk_1, col_sg_1);
        finalize_col_quant_contract_v3(col_sc_2, col_sg_chunk_2, col_sg_2);
        }
    }

    std::vector<torch::Tensor> row_fp4_list{row_fp4_0, row_fp4_1, row_fp4_2};
    std::vector<torch::Tensor> row_sc_list{row_sc_0, row_sc_1, row_sc_2};
    std::vector<torch::Tensor> row_sg_list{row_sg_0, row_sg_1, row_sg_2};
    std::vector<torch::Tensor> col_fp4_list{col_fp4_0, col_fp4_1, col_fp4_2};
    std::vector<torch::Tensor> col_sc_list{col_sc_0, col_sc_1, col_sc_2};
    std::vector<torch::Tensor> col_sg_list{col_sg_0, col_sg_1, col_sg_2};
    auto row_sg_cat = torch::empty({0}, opts_f32);
    if (use_v4_split3_emit_row_sg_cat()) {
        row_sg_cat = torch::empty({m_sg_tiles, 3}, opts_f32);
        copy_cat_dim1_contiguous({row_sg_0, row_sg_1, row_sg_2}, row_sg_cat);
    }

    return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                           col_fp4_list, col_sc_list, col_sg_list,
                           row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_cat, col_sc_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor rope_cs,
    int64_t rope_seq_len,
    bool data_stochastic_rounding = false,
    bool scale_stochastic_rounding = false,
    std::string rht_axes = "none",
    bool with_random_sign_mask = false,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    std::string data_sr_axes = "both",
    std::optional<torch::Tensor> persistent_rng_state = std::nullopt
) {
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "split3 inverse RoPE requires the outerscale v4 contract");
    std::transform(rht_axes.begin(), rht_axes.end(), rht_axes.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "row_col" ||
                          rht_axes == "rowcol" || rht_axes == "all");
    TORCH_CHECK(rht_axes == "none" || rht_axes == "off" || rht_axes == "0" ||
                rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
                rht_axes == "row_col" || rht_axes == "rowcol" || rht_axes == "all",
                "Unsupported split3 inverse-RoPE RHT axes: ", rht_axes);
    TORCH_CHECK(!row_rht && !col_rht && !with_random_sign_mask,
                "split3 inverse-RoPE grad RHT/random-sign is not fused yet; "
                "use grad SR without grad RHT");
    check_rope_live64_tensor(rope_cs, rope_seq_len);
    for (const auto &input : {input0, input1, input2}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.stride(1) == 1,
                    "split inputs must have contiguous last dimension [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 256 == 0,
                    "split widths must be multiples of 256 for v4 split3 inverse RoPE");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");
    TORCH_CHECK(input0.size(0) % 256 == 0,
                "M must be a multiple of 256 for v4 split3 inverse RoPE");

    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);
    const int64_t total_n = n0 + n1 + n2;
    const int64_t m_sg_tiles = outer_sg_tiles_128(M);
    const int64_t n0_sg_tiles = outer_sg_tiles_128(n0);
    const int64_t n1_sg_tiles = outer_sg_tiles_128(n1);
    const int64_t n2_sg_tiles = outer_sg_tiles_128(n2);
    auto device = input0.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({M, total_n / 2}, opts_fp4);
    auto row_sc_cat = torch::empty({M / 128, total_n / 64, 512}, opts_fp8);
    auto col_fp4_cat = torch::empty({total_n, M / 2}, opts_fp4);
    auto col_sc_cat = torch::empty({total_n / 128, M / 64, 512}, opts_fp8);
    auto col_sg_cat = torch::empty({1, n0_sg_tiles + n1_sg_tiles + n2_sg_tiles}, opts_f32);

    auto row_fp4_0 = row_fp4_cat.narrow(1, 0, n0 / 2);
    auto row_fp4_1 = row_fp4_cat.narrow(1, n0 / 2, n1 / 2);
    auto row_fp4_2 = row_fp4_cat.narrow(1, (n0 + n1) / 2, n2 / 2);
    auto row_sc_0 = row_sc_cat.narrow(1, 0, n0 / 64);
    auto row_sc_1 = row_sc_cat.narrow(1, n0 / 64, n1 / 64);
    auto row_sc_2 = row_sc_cat.narrow(1, (n0 + n1) / 64, n2 / 64);
    auto col_fp4_0 = col_fp4_cat.narrow(0, 0, n0);
    auto col_fp4_1 = col_fp4_cat.narrow(0, n0, n1);
    auto col_fp4_2 = col_fp4_cat.narrow(0, n0 + n1, n2);
    auto col_sc_0 = col_sc_cat.narrow(0, 0, n0 / 128);
    auto col_sc_1 = col_sc_cat.narrow(0, n0 / 128, n1 / 128);
    auto col_sc_2 = col_sc_cat.narrow(0, (n0 + n1) / 128, n2 / 128);
    auto row_sg_0 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto row_sg_1 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto row_sg_2 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto col_sg_0 = col_sg_cat.narrow(1, 0, n0_sg_tiles);
    auto col_sg_1 = col_sg_cat.narrow(1, n0_sg_tiles, n1_sg_tiles);
    auto col_sg_2 = col_sg_cat.narrow(1, n0_sg_tiles + n1_sg_tiles, n2_sg_tiles);

    auto row_sg_chunk_0 = torch::empty({M / 128, n0 / 128}, opts_f32);
    auto row_sg_chunk_1 = torch::empty({M / 128, n1 / 128}, opts_f32);
    auto row_sg_chunk_2 = torch::empty({M / 128, n2 / 128}, opts_f32);
    auto col_sg_chunk_0 = torch::empty({n0 / 128, M / 128}, opts_f32);
    auto col_sg_chunk_1 = torch::empty({n1 / 128, M / 128}, opts_f32);
    auto col_sg_chunk_2 = torch::empty({n2 / 128, M / 128}, opts_f32);
    torch::Tensor rotated0;
    torch::Tensor rotated1;
    const bool materialize_rotated = use_v4_split3_rope_materialize_rotated();
    if (materialize_rotated) {
        rotated0 = torch::empty({M, n0}, input0.options());
        rotated1 = torch::empty({M, n1}, input1.options());
    }

    if (use_localcta_v4_split3_direct_final_sg_scan()) {
        launch_scan_split3_sg_rope_direct_final(
            input0, input1, input2, rope_cs, rope_seq_len,
            rotated0, rotated1,
            row_sg_0, col_sg_0,
            row_sg_1, col_sg_1,
            row_sg_2, col_sg_2);
    } else {
        launch_scan_split3_sg_rope(
            input0, input1, input2, rope_cs, rope_seq_len,
            rotated0, rotated1,
            row_sg_chunk_0, col_sg_chunk_0,
            row_sg_chunk_1, col_sg_chunk_1,
            row_sg_chunk_2, col_sg_chunk_2);

        if (use_localcta_v4_split3_fused_sg_reduce()) {
            launch_reduce_row_col_sg_split3(
                row_sg_chunk_0, row_sg_0,
                row_sg_chunk_1, row_sg_1,
                row_sg_chunk_2, row_sg_2,
                col_sg_chunk_0, col_sg_0,
                col_sg_chunk_1, col_sg_1,
                col_sg_chunk_2, col_sg_2);
        } else {
            auto stream = at::cuda::getCurrentCUDAStream();
            const int64_t row_tiles = m_sg_tiles;
            dim3 row_grid(static_cast<unsigned int>(row_tiles), 3u);
            reduce_row_sg_split3_kernel<<<row_grid, 256, 0, stream>>>(
                row_sg_chunk_0.data_ptr<float>(),
                row_sg_0.data_ptr<float>(),
                static_cast<int>(row_sg_chunk_0.size(1)),
                row_sg_chunk_1.data_ptr<float>(),
                row_sg_1.data_ptr<float>(),
                static_cast<int>(row_sg_chunk_1.size(1)),
                row_sg_chunk_2.data_ptr<float>(),
                row_sg_2.data_ptr<float>(),
                static_cast<int>(row_sg_chunk_2.size(1)));
            {
                cudaError_t err = cudaGetLastError();
                TORCH_CHECK(err == cudaSuccess, "reduce_row_sg_split3_kernel failed: ", cudaGetErrorString(err));
            }

            const int k_chunks_0 = static_cast<int>(col_sg_chunk_0.size(0));
            const int k_chunks_1 = static_cast<int>(col_sg_chunk_1.size(0));
            const int k_chunks_2 = static_cast<int>(col_sg_chunk_2.size(0));
            const int max_col_tiles = std::max(std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2), (k_chunks_2 + 1) / 2);
            dim3 col_reduce_grid(static_cast<unsigned int>(max_col_tiles), 3u);
            reduce_col_sg_tiles_split3_kernel<<<col_reduce_grid, 256, 0, stream>>>(
                col_sg_chunk_0.data_ptr<float>(),
                col_sg_0.data_ptr<float>(),
                k_chunks_0,
                static_cast<int>(col_sg_chunk_0.size(1)),
                col_sg_chunk_1.data_ptr<float>(),
                col_sg_1.data_ptr<float>(),
                k_chunks_1,
                col_sg_chunk_2.data_ptr<float>(),
                col_sg_2.data_ptr<float>(),
                k_chunks_2);
            {
                cudaError_t err = cudaGetLastError();
                TORCH_CHECK(err == cudaSuccess, "reduce_col_sg_tiles_split3_kernel failed: ", cudaGetErrorString(err));
            }
        }
    }

    launch_localcta_split3_quant_final_sg(
        materialize_rotated ? rotated0 : input0,
        materialize_rotated ? rotated1 : input1,
        input2,
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1,
        row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_2, col_sg_2,
        materialize_rotated ? torch::Tensor() : rope_cs,
        materialize_rotated ? 0 : rope_seq_len,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rng_seed,
        rng_subsequence_base,
        data_sr_axes,
        persistent_rng_state.value_or(torch::Tensor()));

    std::vector<torch::Tensor> row_fp4_list{row_fp4_0, row_fp4_1, row_fp4_2};
    std::vector<torch::Tensor> row_sc_list{row_sc_0, row_sc_1, row_sc_2};
    std::vector<torch::Tensor> row_sg_list{row_sg_0, row_sg_1, row_sg_2};
    std::vector<torch::Tensor> col_fp4_list{col_fp4_0, col_fp4_1, col_fp4_2};
    std::vector<torch::Tensor> col_sc_list{col_sc_0, col_sc_1, col_sc_2};
    std::vector<torch::Tensor> col_sg_list{col_sg_0, col_sg_1, col_sg_2};
    auto row_sg_cat = torch::empty({0}, opts_f32);
    if (use_v4_split3_emit_row_sg_cat()) {
        row_sg_cat = torch::empty({m_sg_tiles, 3}, opts_f32);
        copy_cat_dim1_contiguous({row_sg_0, row_sg_1, row_sg_2}, row_sg_cat);
    }

    return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                           col_fp4_list, col_sc_list, col_sg_list,
                           row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_cat, col_sc_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split3_rowphase_for_gemm(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2
) {
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "split3 rowphase is only supported for the outerscale contract");
    for (const auto &input : {input0, input1, input2}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split inputs must be contiguous [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);
    const int64_t total_n = n0 + n1 + n2;
    const int64_t m_sg_tiles = outer_sg_tiles_128(M);
    const int64_t n0_sg_tiles = outer_sg_tiles_128(n0);
    const int64_t n1_sg_tiles = outer_sg_tiles_128(n1);
    const int64_t n2_sg_tiles = outer_sg_tiles_128(n2);
    auto device = input0.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({M, total_n / 2}, opts_fp4);
    auto row_sc_cat = torch::empty({M / 128, total_n / 64, 512}, opts_fp8);
    auto col_fp4_cat = torch::empty({total_n, M / 2}, opts_fp4);
    auto col_sc_cat = torch::empty({total_n / 128, M / 64, 512}, opts_fp8);
    auto col_sg_cat = torch::empty({1, n0_sg_tiles + n1_sg_tiles + n2_sg_tiles}, opts_f32);

    auto row_fp4_0 = row_fp4_cat.narrow(1, 0, n0 / 2);
    auto row_fp4_1 = row_fp4_cat.narrow(1, n0 / 2, n1 / 2);
    auto row_fp4_2 = row_fp4_cat.narrow(1, (n0 + n1) / 2, n2 / 2);
    auto row_sc_0 = row_sc_cat.narrow(1, 0, n0 / 64);
    auto row_sc_1 = row_sc_cat.narrow(1, n0 / 64, n1 / 64);
    auto row_sc_2 = row_sc_cat.narrow(1, (n0 + n1) / 64, n2 / 64);
    auto row_sg_0 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto row_sg_1 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto row_sg_2 = torch::empty({m_sg_tiles, 1}, opts_f32);
    auto row_sg_cat = torch::empty({0}, opts_f32);

    auto col_sg_0 = col_sg_cat.narrow(1, 0, n0_sg_tiles);
    auto col_sg_1 = col_sg_cat.narrow(1, n0_sg_tiles, n1_sg_tiles);
    auto col_sg_2 = col_sg_cat.narrow(1, n0_sg_tiles + n1_sg_tiles, n2_sg_tiles);
    auto col_fp4_0 = col_fp4_cat.narrow(0, 0, n0);
    auto col_fp4_1 = col_fp4_cat.narrow(0, n0, n1);
    auto col_fp4_2 = col_fp4_cat.narrow(0, n0 + n1, n2);
    auto col_sc_0 = col_sc_cat.narrow(0, 0, n0 / 128);
    auto col_sc_1 = col_sc_cat.narrow(0, n0 / 128, n1 / 128);
    auto col_sc_2 = col_sc_cat.narrow(0, (n0 + n1) / 128, n2 / 128);

    auto row_sg_chunk_0 = torch::empty({M / 128, n0 / 128}, opts_f32);
    auto row_sg_chunk_1 = torch::empty({M / 128, n1 / 128}, opts_f32);
    auto row_sg_chunk_2 = torch::empty({M / 128, n2 / 128}, opts_f32);
    auto col_sg_chunk_0 = torch::empty({n0 / 128, M / 128}, opts_f32);
    auto col_sg_chunk_1 = torch::empty({n1 / 128, M / 128}, opts_f32);
    auto col_sg_chunk_2 = torch::empty({n2 / 128, M / 128}, opts_f32);

    launch_localcta_split3_quant_raw(
        input0, input1, input2,
        row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_chunk_0, col_sg_chunk_0,
        row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_chunk_1, col_sg_chunk_1,
        row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_chunk_2, col_sg_chunk_2);

    const auto multiinput_mode = get_v3_split3_multiinput_quant_mode();
    if (multiinput_mode == V3MultiInputQuantMode::RowColSplitFinalize) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int64_t row_tiles = m_sg_tiles;
        dim3 row_grid(static_cast<unsigned int>(row_tiles), 3u);
        finalize_row_sc_split3_kernel<<<row_grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_0.data_ptr()),
            row_sg_chunk_0.data_ptr<float>(),
            row_sg_0.data_ptr<float>(),
            static_cast<int>(row_sc_0.size(1)),
            static_cast<int>(row_sg_chunk_0.size(1)),
            row_sc_0.stride(0),
            row_sc_0.stride(1),
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_1.data_ptr()),
            row_sg_chunk_1.data_ptr<float>(),
            row_sg_1.data_ptr<float>(),
            static_cast<int>(row_sc_1.size(1)),
            static_cast<int>(row_sg_chunk_1.size(1)),
            row_sc_1.stride(0),
            row_sc_1.stride(1),
            reinterpret_cast<__nv_fp8_e4m3*>(row_sc_2.data_ptr()),
            row_sg_chunk_2.data_ptr<float>(),
            row_sg_2.data_ptr<float>(),
            static_cast<int>(row_sc_2.size(1)),
            static_cast<int>(row_sg_chunk_2.size(1)),
            row_sc_2.stride(0),
            row_sc_2.stride(1));
        {
            cudaError_t err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "finalize_row_sc_split3_kernel failed: ", cudaGetErrorString(err));
        }
    } else {
        finalize_row_quant_contract_v3_strided(row_sc_0, row_sg_chunk_0, row_sg_0);
        finalize_row_quant_contract_v3_strided(row_sc_1, row_sg_chunk_1, row_sg_1);
        finalize_row_quant_contract_v3_strided(row_sc_2, row_sg_chunk_2, row_sg_2);
    }

    if (use_v4_split3_emit_row_sg_cat()) {
        row_sg_cat = torch::empty({m_sg_tiles, 3}, opts_f32);
        copy_cat_dim1_contiguous({row_sg_0, row_sg_1, row_sg_2}, row_sg_cat);
    }

    std::vector<torch::Tensor> row_fp4_list{row_fp4_0, row_fp4_1, row_fp4_2};
    std::vector<torch::Tensor> row_sc_list{row_sc_0, row_sc_1, row_sc_2};
    std::vector<torch::Tensor> row_sg_list{row_sg_0, row_sg_1, row_sg_2};
    return std::make_tuple(
        row_fp4_list, row_sc_list, row_sg_list,
        row_fp4_cat, row_sc_cat, row_sg_cat,
        col_fp4_cat, col_sc_cat, col_sg_cat,
        col_sg_chunk_0, col_sg_chunk_1, col_sg_chunk_2);
}

void tk_localcta_group_quantize_dim1_split3_finalize_col_inplace(
    torch::Tensor col_sc_cat,
    torch::Tensor col_sg_cat,
    torch::Tensor col_sg_chunk_0,
    torch::Tensor col_sg_chunk_1,
    torch::Tensor col_sg_chunk_2
) {
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "split3 deferred col finalize is only supported for the outerscale contract");
    TORCH_CHECK(col_sc_cat.is_cuda() && col_sg_cat.is_cuda(),
                "split3 deferred col finalize expects CUDA tensors");
    TORCH_CHECK(col_sg_chunk_0.is_cuda() && col_sg_chunk_1.is_cuda() && col_sg_chunk_2.is_cuda(),
                "split3 deferred col finalize expects CUDA SG chunk tensors");

    const int64_t n0 = col_sg_chunk_0.size(0) * 128;
    const int64_t n1 = col_sg_chunk_1.size(0) * 128;
    const int64_t n2 = col_sg_chunk_2.size(0) * 128;

    auto col_sc_0 = col_sc_cat.narrow(0, 0, n0 / 128);
    auto col_sc_1 = col_sc_cat.narrow(0, n0 / 128, n1 / 128);
    auto col_sc_2 = col_sc_cat.narrow(0, (n0 + n1) / 128, n2 / 128);
    const int64_t n0_sg_tiles = outer_sg_tiles_128(n0);
    const int64_t n1_sg_tiles = outer_sg_tiles_128(n1);
    const int64_t n2_sg_tiles = outer_sg_tiles_128(n2);
    auto col_sg_0 = col_sg_cat.narrow(1, 0, n0_sg_tiles);
    auto col_sg_1 = col_sg_cat.narrow(1, n0_sg_tiles, n1_sg_tiles);
    auto col_sg_2 = col_sg_cat.narrow(1, n0_sg_tiles + n1_sg_tiles, n2_sg_tiles);

    finalize_col_quant_contract_v3_split3(
        col_sc_0, col_sg_chunk_0, col_sg_0,
        col_sc_1, col_sg_chunk_1, col_sg_1,
        col_sc_2, col_sg_chunk_2, col_sg_2);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split3_for_gemm_prepared(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2
) {
    for (const auto &input : {input0, input1, input2}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split inputs must be contiguous [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);
    const int64_t total_n = n0 + n1 + n2;
    auto device = input0.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({M, total_n / 2}, opts_fp4);
    auto row_sc_prepared_cat = torch::empty({M / 128, total_n / 64, 512}, opts_fp8);
    auto row_sg_cat = torch::empty({M / 128, total_n / 128}, opts_f32);
    auto col_fp4_cat = torch::empty({total_n, M / 2}, opts_fp4);
    auto col_sc_prepared_cat = torch::empty({total_n / 128, M / 64, 512}, opts_fp8);
    auto col_sg_cat = torch::empty({total_n / 128, M / 128}, opts_f32);

    std::vector<torch::Tensor> row_fp4_list{
        row_fp4_cat.narrow(1, 0, n0 / 2),
        row_fp4_cat.narrow(1, n0 / 2, n1 / 2),
        row_fp4_cat.narrow(1, (n0 + n1) / 2, n2 / 2),
    };
    std::vector<torch::Tensor> row_sc_prepared_list{
        row_sc_prepared_cat.narrow(1, 0, n0 / 64),
        row_sc_prepared_cat.narrow(1, n0 / 64, n1 / 64),
        row_sc_prepared_cat.narrow(1, (n0 + n1) / 64, n2 / 64),
    };
    std::vector<torch::Tensor> row_sg_list{
        row_sg_cat.narrow(1, 0, n0 / 128),
        row_sg_cat.narrow(1, n0 / 128, n1 / 128),
        row_sg_cat.narrow(1, (n0 + n1) / 128, n2 / 128),
    };
    std::vector<torch::Tensor> col_fp4_list{
        col_fp4_cat.narrow(0, 0, n0),
        col_fp4_cat.narrow(0, n0, n1),
        col_fp4_cat.narrow(0, n0 + n1, n2),
    };
    std::vector<torch::Tensor> col_sc_prepared_list{
        col_sc_prepared_cat.narrow(0, 0, n0 / 128),
        col_sc_prepared_cat.narrow(0, n0 / 128, n1 / 128),
        col_sc_prepared_cat.narrow(0, (n0 + n1) / 128, n2 / 128),
    };
    std::vector<torch::Tensor> col_sg_list{
        col_sg_cat.narrow(0, 0, n0 / 128),
        col_sg_cat.narrow(0, n0 / 128, n1 / 128),
        col_sg_cat.narrow(0, (n0 + n1) / 128, n2 / 128),
    };

    launch_localcta_split3_quant_prepared(
        input0, input1, input2,
        row_fp4_cat, row_sc_prepared_cat,
        col_fp4_cat, col_sc_prepared_cat,
        row_sg_cat, col_sg_cat);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_group_quantize_dim1_split3_for_gemm_prepared failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(row_fp4_list, row_sc_prepared_list, row_sg_list,
                           col_fp4_list, col_sc_prepared_list, col_sg_list,
                           row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
}

torch::Tensor tk_localcta_reconstruct_row(
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunks
) {
    return reconstruct_rowwise_impl(row_fp4, row_sc, row_sg_chunks,
                                    row_fp4.size(0), row_fp4.size(1) * 2);
}

torch::Tensor tk_localcta_reconstruct_col(
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor col_sg_chunks
) {
    return reconstruct_rowwise_impl(col_fp4, col_sc, col_sg_chunks,
                                    col_fp4.size(0), col_fp4.size(1) * 2);
}

void tk_localcta_set_2cta_prepared_tuning(
    int threads,
    int pipe_depth,
    bool shared_amax
) {
    TORCH_CHECK(threads == 160 || threads == 192 || threads == 256 || threads == 384 || threads == 512,
                "threads must be one of {160, 192, 256, 384, 512}");
    TORCH_CHECK(pipe_depth >= 1 && pipe_depth <= 4,
                "pipe_depth must be in [1, 4]");
    auto &cfg = get_localcta2_prepared_tuning();
    cfg.threads = threads;
    cfg.pipe_depth = pipe_depth;
    cfg.shared_amax = shared_amax;
}

std::tuple<int, int, bool> tk_localcta_get_2cta_prepared_tuning() {
    const auto &cfg = get_localcta2_prepared_tuning();
    return std::make_tuple(cfg.threads, cfg.pipe_depth, cfg.shared_amax);
}

void tk_localcta_set_2cta_prepared_split2_tuning(
    int threads,
    int pipe_depth,
    bool shared_amax
) {
    TORCH_CHECK(threads == 160 || threads == 192 || threads == 256 || threads == 384 || threads == 512,
                "threads must be one of {160, 192, 256, 384, 512}");
    TORCH_CHECK(pipe_depth >= 1 && pipe_depth <= 4,
                "pipe_depth must be in [1, 4]");
    auto &cfg = get_localcta2_prepared_split2_tuning();
    cfg.threads = threads;
    cfg.pipe_depth = pipe_depth;
    cfg.shared_amax = shared_amax;
}

std::tuple<int, int, bool> tk_localcta_get_2cta_prepared_split2_tuning() {
    const auto &cfg = get_localcta2_prepared_split2_tuning();
    return std::make_tuple(cfg.threads, cfg.pipe_depth, cfg.shared_amax);
}

void tk_localcta_set_1cta_prepared_tuning(
    int threads,
    int pipe_depth
) {
    TORCH_CHECK(threads == 160 || threads == 192 || threads == 256,
                "threads must be one of {160, 192, 256}");
    TORCH_CHECK(pipe_depth == 1 || pipe_depth == 2,
                "pipe_depth must be one of {1, 2}");
    auto &cfg = get_localcta1_prepared_tuning();
    cfg.threads = threads;
    cfg.pipe_depth = pipe_depth;
}

std::tuple<int, int> tk_localcta_get_1cta_prepared_tuning() {
    const auto &cfg = get_localcta1_prepared_tuning();
    return std::make_tuple(cfg.threads, cfg.pipe_depth);
}

static void check_mixed_rank2_input(
    const torch::Tensor& input,
    const char* producer
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                producer, " requires a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                producer, " requires BF16 [M, K]");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 &&
                    input.size(0) % 256 == 0 && input.size(1) % 256 == 0,
                producer, " currently requires positive 256-aligned dimensions; got ",
                input.sizes());
}

static void check_mixed_tensor(
    const torch::Tensor& tensor,
    const torch::Tensor& input,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    const char* name
) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == input.device(),
                name, " must be a CUDA tensor on ", input.device());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype,
                name, " has wrong dtype: expected ", dtype, ", got ",
                tensor.scalar_type());
    TORCH_CHECK(tensor.sizes() == sizes,
                name, " has wrong shape: expected ", sizes, ", got ",
                tensor.sizes());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_mixed_grad_localcta_row_mx_col_alloc(
    int64_t M,
    int64_t N,
    torch::Device device
) {
    TORCH_CHECK(device.is_cuda(), "mixed grad carrier requires a CUDA device");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "mixed grad carrier requires the localCTA outer-SG contract; "
                "USE_TK_LOCALCTA_V3_CONTRACT=tilegrid256 is unsupported");
    TORCH_CHECK(M > 0 && N > 0 && M % 256 == 0 && N % 256 == 0,
                "mixed grad carrier requires positive 256-aligned M/N");
    auto fp4_opts = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto fp8_opts = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto u8_opts = torch::dtype(torch::kUInt8).device(device);
    auto f32_opts = torch::dtype(torch::kFloat32).device(device);
    return std::make_tuple(
        torch::empty({M, N / 2}, fp4_opts),
        torch::empty({M / 128, N / 64, 512}, fp8_opts),
        torch::empty({outer_sg_tiles_128(M), 1}, f32_opts),
        torch::empty({N, M / 2}, fp4_opts),
        torch::empty({N / 128, M / 128, 32, 16}, u8_opts),
        torch::empty({M / 128, N / 128}, f32_opts));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_mixed_grad_localcta_row_mx_col_launch_inplace(
    torch::Tensor grad,
    torch::Tensor local_row_fp4,
    torch::Tensor local_row_sc,
    torch::Tensor local_row_sg,
    torch::Tensor mx_col_fp4,
    torch::Tensor mx_col_sc,
    torch::Tensor local_row_sg_chunk,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    check_mixed_rank2_input(grad, "mixed grad carrier");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "mixed grad carrier requires the localCTA outer-SG contract; "
                "USE_TK_LOCALCTA_V3_CONTRACT=tilegrid256 is unsupported");
    const c10::cuda::CUDAGuard device_guard(grad.device());
    const auto set_device_err = cudaSetDevice(grad.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before mixed grad carrier: ",
                cudaGetErrorString(set_device_err));
    const int64_t M = grad.size(0);
    const int64_t N = grad.size(1);
    check_mixed_tensor(local_row_fp4, grad, torch::kFloat4_e2m1fn_x2,
                       {M, N / 2}, "local_row_fp4");
    check_mixed_tensor(local_row_sc, grad, torch::kFloat8_e4m3fn,
                       {M / 128, N / 64, 512}, "local_row_sc");
    check_mixed_tensor(local_row_sg, grad, torch::kFloat32,
                       {outer_sg_tiles_128(M), 1}, "local_row_sg");
    check_mixed_tensor(mx_col_fp4, grad, torch::kFloat4_e2m1fn_x2,
                       {N, M / 2}, "mx_col_fp4");
    check_mixed_tensor(mx_col_sc, grad, torch::kUInt8,
                       {N / 128, M / 128, 32, 16}, "mx_col_sc");
    check_mixed_tensor(local_row_sg_chunk, grad, torch::kFloat32,
                       {M / 128, N / 128}, "local_row_sg_chunk");

    alignas(64) CUtensorMap tmap_in{}, tmap_local_row{};
    alignas(64) CUtensorMap tmap_local_row_sc{}, tmap_mx_col{};
    create_tma_2d(tmap_in, grad.data_ptr(), M, N, 64, 64, N, 16);
    create_tma_2d(tmap_local_row, local_row_fp4.data_ptr(),
                  M, N, 64, 64, N, 4);
    create_tma_2d(tmap_mx_col, mx_col_fp4.data_ptr(),
                  N, M, 64, 64, M, 4);
    const int64_t local_sc_x_bf16 = (N / 64) * 256;
    create_tma_2d(tmap_local_row_sc, local_row_sc.data_ptr(),
                  M / 128, local_sc_x_bf16,
                  1, 256, local_sc_x_bf16, 16);

    auto kernel =
        tk_mixed_mx_localcta::mixed_grad_localcta_row_mx_col_kernel;
    const int dshmem = tk_mixed_mx_localcta::mixed_grad_shmem_size();
    const auto attr_err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    TORCH_CHECK(attr_err == cudaSuccess,
                "mixed grad carrier dynamic-shared-memory attribute failed: ",
                cudaGetErrorString(attr_err));
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel<<<dim3(N / 128, M / 128), 128, dshmem, stream>>>(
        tmap_in,
        tmap_in,
        tmap_local_row,
        tmap_local_row_sc,
        tmap_mx_col,
        reinterpret_cast<uint8_t*>(mx_col_sc.data_ptr()),
        local_row_sg_chunk.data_ptr<float>(),
        M,
        N,
        -1,
        rng_seed,
        rng_subsequence_base);
    {
        const auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "mixed grad carrier launch failed: ",
                    cudaGetErrorString(err));
    }

    // The fused producer emits chunk-local localCTA scales.  Reuse the
    // production row finalizer to obtain the native outer-SG contract; these
    // scale-only kernels do not reread or retain the BF16 gradient.
    finalize_quant_contract_v3(
        local_row_sc,
        local_row_sg_chunk,
        local_row_sg,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor());
    return std::make_tuple(
        local_row_fp4, local_row_sc, local_row_sg,
        mx_col_fp4, mx_col_sc, local_row_sg_chunk);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_mixed_split2_grad_localcta_row_mx_col_alloc(
    int64_t M,
    int64_t H,
    torch::Device device
) {
    TORCH_CHECK(device.is_cuda(),
                "mixed split2 grad carrier requires a CUDA device");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "mixed split2 grad carrier requires the localCTA outer-SG "
                "contract; tilegrid256 is unsupported");
    TORCH_CHECK(M > 0 && H > 0 && M % 256 == 0 && H % 256 == 0,
                "mixed split2 grad carrier requires positive 256-aligned M/H");
    TORCH_CHECK(H <= std::numeric_limits<int64_t>::max() / 2,
                "mixed split2 grad width overflows int64");
    auto fp4_opts = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto fp8_opts = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto u8_opts = torch::dtype(torch::kUInt8).device(device);
    auto f32_opts = torch::dtype(torch::kFloat32).device(device);
    return std::make_tuple(
        torch::empty({M, H}, fp4_opts),
        torch::empty({M / 128, H / 32, 512}, fp8_opts),
        torch::empty({outer_sg_tiles_128(M), 1}, f32_opts),
        torch::empty({outer_sg_tiles_128(M), 1}, f32_opts),
        torch::empty({2 * H, M / 2}, fp4_opts),
        torch::empty({H / 64, M / 128, 32, 16}, u8_opts),
        torch::empty({M / 128, H / 64}, f32_opts));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace(
    torch::Tensor grad0,
    torch::Tensor grad1,
    torch::Tensor local_row_fp4,
    torch::Tensor local_row_sc,
    torch::Tensor local_row_sg0,
    torch::Tensor local_row_sg1,
    torch::Tensor mx_col_fp4,
    torch::Tensor mx_col_sc,
    torch::Tensor local_row_sg_chunk,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    check_mixed_rank2_input(grad0, "mixed split2 grad carrier input0");
    check_mixed_rank2_input(grad1, "mixed split2 grad carrier input1");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "mixed split2 grad carrier requires the localCTA outer-SG "
                "contract; tilegrid256 is unsupported");
    TORCH_CHECK(grad0.device() == grad1.device() &&
                    grad0.sizes() == grad1.sizes(),
                "mixed split2 grad inputs must have identical shape/device");
    const c10::cuda::CUDAGuard device_guard(grad0.device());
    const auto set_device_err = cudaSetDevice(grad0.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before mixed split2 grad carrier: ",
                cudaGetErrorString(set_device_err));
    const int64_t M = grad0.size(0);
    const int64_t H = grad0.size(1);
    const int64_t N = 2 * H;
    check_mixed_tensor(local_row_fp4, grad0, torch::kFloat4_e2m1fn_x2,
                       {M, N / 2}, "local_row_fp4");
    check_mixed_tensor(local_row_sc, grad0, torch::kFloat8_e4m3fn,
                       {M / 128, N / 64, 512}, "local_row_sc");
    check_mixed_tensor(local_row_sg0, grad0, torch::kFloat32,
                       {outer_sg_tiles_128(M), 1}, "local_row_sg0");
    check_mixed_tensor(local_row_sg1, grad0, torch::kFloat32,
                       {outer_sg_tiles_128(M), 1}, "local_row_sg1");
    check_mixed_tensor(mx_col_fp4, grad0, torch::kFloat4_e2m1fn_x2,
                       {N, M / 2}, "mx_col_fp4");
    check_mixed_tensor(mx_col_sc, grad0, torch::kUInt8,
                       {N / 128, M / 128, 32, 16}, "mx_col_sc");
    check_mixed_tensor(local_row_sg_chunk, grad0, torch::kFloat32,
                       {M / 128, N / 128}, "local_row_sg_chunk");

    alignas(64) CUtensorMap tmap_in0{}, tmap_in1{}, tmap_local_row{};
    alignas(64) CUtensorMap tmap_local_row_sc{}, tmap_mx_col{};
    create_tma_2d(tmap_in0, grad0.data_ptr(), M, H, 64, 64, H, 16);
    create_tma_2d(tmap_in1, grad1.data_ptr(), M, H, 64, 64, H, 16);
    create_tma_2d(tmap_local_row, local_row_fp4.data_ptr(),
                  M, N, 64, 64, N, 4);
    create_tma_2d(tmap_mx_col, mx_col_fp4.data_ptr(),
                  N, M, 64, 64, M, 4);
    const int64_t local_sc_x_bf16 = (N / 64) * 256;
    create_tma_2d(tmap_local_row_sc, local_row_sc.data_ptr(),
                  M / 128, local_sc_x_bf16,
                  1, 256, local_sc_x_bf16, 16);

    auto kernel =
        tk_mixed_mx_localcta::mixed_grad_localcta_row_mx_col_kernel;
    const int dshmem = tk_mixed_mx_localcta::mixed_grad_shmem_size();
    const auto attr_err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    TORCH_CHECK(attr_err == cudaSuccess,
                "mixed split2 grad dynamic-shared-memory attribute failed: ",
                cudaGetErrorString(attr_err));
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel<<<dim3(N / 128, M / 128), 128, dshmem, stream>>>(
        tmap_in0,
        tmap_in1,
        tmap_local_row,
        tmap_local_row_sc,
        tmap_mx_col,
        reinterpret_cast<uint8_t*>(mx_col_sc.data_ptr()),
        local_row_sg_chunk.data_ptr<float>(),
        M,
        N,
        static_cast<int>(H / 128),
        rng_seed,
        rng_subsequence_base);
    {
        const auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "mixed split2 grad carrier launch failed: ",
                    cudaGetErrorString(err));
    }
    // Production split2 localCTA uses one logical SR coordinate but two
    // independently finalized row outer scales.  Preserve combined physical
    // storage and finalize each strided half in place.
    auto local_row_sc0 = local_row_sc.narrow(1, 0, H / 64);
    auto local_row_sc1 = local_row_sc.narrow(1, H / 64, H / 64);
    auto local_row_sg_chunk0 = local_row_sg_chunk.narrow(1, 0, H / 128);
    auto local_row_sg_chunk1 =
        local_row_sg_chunk.narrow(1, H / 128, H / 128);
    finalize_row_quant_contract_v3_strided_split2(
        local_row_sc0,
        local_row_sg_chunk0,
        local_row_sg0,
        local_row_sc1,
        local_row_sg_chunk1,
        local_row_sg1);
    return std::make_tuple(
        local_row_fp4, local_row_sc, local_row_sg0, local_row_sg1,
        mx_col_fp4, mx_col_sc, local_row_sg_chunk);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_mixed_weight_mx_row_localcta_col_alloc(
    int64_t N,
    int64_t K,
    torch::Device device
) {
    TORCH_CHECK(device.is_cuda(), "mixed weight carrier requires a CUDA device");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "mixed weight carrier requires the localCTA outer-SG contract; "
                "USE_TK_LOCALCTA_V3_CONTRACT=tilegrid256 is unsupported");
    TORCH_CHECK(N > 0 && K > 0 && N % 256 == 0 && K % 256 == 0,
                "mixed weight carrier requires positive 256-aligned N/K");
    auto fp4_opts = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto fp8_opts = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto u8_opts = torch::dtype(torch::kUInt8).device(device);
    auto f32_opts = torch::dtype(torch::kFloat32).device(device);
    return std::make_tuple(
        torch::empty({N, K / 2}, fp4_opts),
        torch::empty({N / 128, K / 128, 32, 16}, u8_opts),
        torch::empty({K, N / 2}, fp4_opts),
        torch::empty({K / 128, N / 64, 512}, fp8_opts),
        torch::empty({K / 128, N / 128}, f32_opts),
        torch::empty({K / 128, N / 128}, f32_opts));
}

static void finalize_mixed_weight_localcta_col(
    torch::Tensor local_col_sc,
    torch::Tensor local_col_sg,
    torch::Tensor local_col_sg_chunk
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto copy_err = cudaMemcpyAsync(
        local_col_sg.data_ptr<float>(),
        local_col_sg_chunk.data_ptr<float>(),
        local_col_sg.numel() * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream);
    TORCH_CHECK(copy_err == cudaSuccess,
                "mixed weight localCTA SG scratch copy failed: ",
                cudaGetErrorString(copy_err));
    reduce_weight_2d_common_outer_sg_kernel<256><<<1, 256, 0, stream>>>(
        local_col_sg.data_ptr<float>(), local_col_sg.numel());
    {
        const auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "mixed weight localCTA common outer-SG reduction failed: ",
                    cudaGetErrorString(err));
    }

    const int col_chunks = static_cast<int>(local_col_sg_chunk.size(0));
    const int col_sg_rows = static_cast<int>(local_col_sg_chunk.size(1));
    const int col_sc_rows = static_cast<int>(local_col_sc.size(1));
    rescale_weight_2d_col_sc_kernel<256><<<
        dim3(col_chunks, col_sc_rows), 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(local_col_sc.data_ptr()),
            local_col_sg_chunk.data_ptr<float>(),
            local_col_sg.data_ptr<float>(),
            col_chunks,
            col_sc_rows,
            col_sg_rows);
    {
        const auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "mixed weight localCTA column-scale normalization failed: ",
                    cudaGetErrorString(err));
    }
    fill_weight_2d_common_outer_sg_kernel<256><<<1, 256, 0, stream>>>(
        local_col_sg.data_ptr<float>(), 0,
        local_col_sg.data_ptr<float>(), local_col_sg.numel());
    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "mixed weight localCTA outer-SG broadcast failed: ",
                cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_mixed_weight_mx_row_localcta_col_launch_inplace(
    torch::Tensor weight,
    torch::Tensor mx_row_fp4,
    torch::Tensor mx_row_sc,
    torch::Tensor local_col_fp4,
    torch::Tensor local_col_sc,
    torch::Tensor local_col_sg,
    torch::Tensor local_col_sg_chunk
) {
    check_mixed_rank2_input(weight, "mixed weight carrier");
    TORCH_CHECK(get_v3_contract_mode() != V3ContractMode::TileGrid256,
                "mixed weight carrier requires the localCTA outer-SG contract; "
                "USE_TK_LOCALCTA_V3_CONTRACT=tilegrid256 is unsupported");
    const c10::cuda::CUDAGuard device_guard(weight.device());
    const auto set_device_err = cudaSetDevice(weight.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before mixed weight carrier: ",
                cudaGetErrorString(set_device_err));
    const int64_t N = weight.size(0);
    const int64_t K = weight.size(1);
    check_mixed_tensor(mx_row_fp4, weight, torch::kFloat4_e2m1fn_x2,
                       {N, K / 2}, "mx_row_fp4");
    check_mixed_tensor(mx_row_sc, weight, torch::kUInt8,
                       {N / 128, K / 128, 32, 16}, "mx_row_sc");
    check_mixed_tensor(local_col_fp4, weight, torch::kFloat4_e2m1fn_x2,
                       {K, N / 2}, "local_col_fp4");
    check_mixed_tensor(local_col_sc, weight, torch::kFloat8_e4m3fn,
                       {K / 128, N / 64, 512}, "local_col_sc");
    check_mixed_tensor(local_col_sg, weight, torch::kFloat32,
                       {K / 128, N / 128}, "local_col_sg");
    check_mixed_tensor(local_col_sg_chunk, weight, torch::kFloat32,
                       {K / 128, N / 128}, "local_col_sg_chunk");

    alignas(64) CUtensorMap tmap_in{}, tmap_mx_row{};
    alignas(64) CUtensorMap tmap_local_col{}, tmap_local_col_sc{};
    create_tma_2d(tmap_in, weight.data_ptr(), N, K, 64, 64, K, 16);
    create_tma_2d(tmap_mx_row, mx_row_fp4.data_ptr(),
                  N, K, 64, 64, K, 4);
    create_tma_2d(tmap_local_col, local_col_fp4.data_ptr(),
                  K, N, 64, 64, N, 4);
    const int64_t local_sc_x_bf16 = (N / 64) * 256;
    create_tma_2d(tmap_local_col_sc, local_col_sc.data_ptr(),
                  K / 128, local_sc_x_bf16,
                  1, 256, local_sc_x_bf16, 16);

    auto kernel =
        tk_mixed_mx_localcta::mixed_weight_mx_row_localcta_col_kernel;
    const int dshmem = tk_mixed_mx_localcta::mixed_weight_shmem_size();
    const auto attr_err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    TORCH_CHECK(attr_err == cudaSuccess,
                "mixed weight carrier dynamic-shared-memory attribute failed: ",
                cudaGetErrorString(attr_err));
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel<<<dim3(K / 128, N / 128), 128, dshmem, stream>>>(
        tmap_in,
        tmap_mx_row,
        tmap_local_col,
        tmap_local_col_sc,
        reinterpret_cast<uint8_t*>(mx_row_sc.data_ptr()),
        local_col_sg_chunk.data_ptr<float>(),
        N,
        K);
    {
        const auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "mixed weight carrier launch failed: ",
                    cudaGetErrorString(err));
    }
    finalize_mixed_weight_localcta_col(
        local_col_sc, local_col_sg, local_col_sg_chunk);
    return std::make_tuple(
        mx_row_fp4, mx_row_sc,
        local_col_fp4, local_col_sc, local_col_sg,
        local_col_sg_chunk);
}

py::dict tk_mixed_mx_localcta_capabilities() {
    py::dict result;
    result["abi_version"] = 1;
    result["grad_coordinate_mode"] = "explicit_seed_subsequence";
    result["grad_mx_col_rht"] = "block32_fixed_0x2817";
    result["mxfp4_rht_block_size"] = 32;
    result["mxfp4_rht_sign_contract"] = "fixed_0x2817_per_h16_half";
    result["grad_localcta_row_sr"] = true;
    result["localcta_encode_mode"] = "encode_centric";
    result["localcta_sg_contract"] = "outer";
    result["grad_scale_sr"] = false;
    result["weight_mx_2d"] = true;
    result["weight_localcta_2d"] = true;
    result["prepared_outer_sg"] = true;
    result["min_alignment"] = 256;
    result["single_bf16_tile_load"] = true;
    result["runtime_advances_rng"] = false;
    result["split2_grad_one_coordinate"] = true;
    result["split2_dgrad_onepass_outer_sg"] = true;
    result["split2_row_outer_sg"] = "per_arm";
    result["split2_row_rescale"] = "grouped_fp8x4_warp";
    result["split2_row_rescale_cols_per_block"] = 8;
    result["split2_row_rescale_launches"] = 1;
    result["split2_layout"] =
        "logical_dim1_concat_per_arm_outer_no_bf16_materialization";
    result["grad_dynamic_smem_bytes"] =
        tk_mixed_mx_localcta::mixed_grad_shmem_size();
    result["weight_dynamic_smem_bytes"] =
        tk_mixed_mx_localcta::mixed_weight_shmem_size();
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tk_mixed_mx_localcta_capabilities",
          &tk_mixed_mx_localcta_capabilities);
    m.def("tk_mixed_grad_localcta_row_mx_col_alloc",
          &tk_mixed_grad_localcta_row_mx_col_alloc,
          py::arg("M"), py::arg("N"), py::arg("device"));
    m.def("tk_mixed_grad_localcta_row_mx_col_launch_inplace",
          &tk_mixed_grad_localcta_row_mx_col_launch_inplace,
          py::arg("grad"),
          py::arg("local_row_fp4"), py::arg("local_row_sc"),
          py::arg("local_row_sg"),
          py::arg("mx_col_fp4"), py::arg("mx_col_sc"),
          py::arg("local_row_sg_chunk"),
          py::arg("rng_seed"), py::arg("rng_subsequence_base"));
    m.def("tk_mixed_split2_grad_localcta_row_mx_col_alloc",
          &tk_mixed_split2_grad_localcta_row_mx_col_alloc,
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace",
          &tk_mixed_split2_grad_localcta_row_mx_col_launch_inplace,
          py::arg("grad0"), py::arg("grad1"),
          py::arg("local_row_fp4"), py::arg("local_row_sc"),
          py::arg("local_row_sg0"), py::arg("local_row_sg1"),
          py::arg("mx_col_fp4"), py::arg("mx_col_sc"),
          py::arg("local_row_sg_chunk"),
          py::arg("rng_seed"), py::arg("rng_subsequence_base"));
    m.def("tk_mixed_weight_mx_row_localcta_col_alloc",
          &tk_mixed_weight_mx_row_localcta_col_alloc,
          py::arg("N"), py::arg("K"), py::arg("device"));
    m.def("tk_mixed_weight_mx_row_localcta_col_launch_inplace",
          &tk_mixed_weight_mx_row_localcta_col_launch_inplace,
          py::arg("weight"),
          py::arg("mx_row_fp4"), py::arg("mx_row_sc"),
          py::arg("local_col_fp4"), py::arg("local_col_sc"),
          py::arg("local_col_sg"), py::arg("local_col_sg_chunk"));
    m.def("tk_localcta_quantize_for_gemm", &tk_localcta_quantize_for_gemm,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_padded",
          &tk_localcta_quantize_for_gemm_padded,
          py::arg("input"), py::arg("output_rows"), py::arg("output_cols"),
          py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_final_sg", &tk_localcta_quantize_for_gemm_final_sg,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_atomic_final_sg",
          &tk_localcta_quantize_for_gemm_atomic_final_sg,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_atomic_paired_col_rht",
          &tk_localcta_quantize_for_gemm_atomic_paired_col_rht,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_final_sg_paired_col_rht",
          &tk_localcta_quantize_for_gemm_final_sg_paired_col_rht,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_opt", &tk_localcta_quantize_for_gemm_opt,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0,
          py::arg("data_sr_axes") = "both",
          py::arg("persistent_rng_state") = std::nullopt);
    m.def("tk_localcta_quantize_for_gemm_final_sg_opt",
          &tk_localcta_quantize_for_gemm_final_sg_opt,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0,
          py::arg("four_over_six_mae") = false);
    m.def("tk_localcta_quantize_col_for_gemm_final_sg_opt",
          &tk_localcta_quantize_col_for_gemm_final_sg_opt,
          py::arg("input"),
          py::arg("encode_centric") = true,
          py::arg("four_over_six_mae") = false);
    m.def("tk_localcta_quantize_mxfp8_row_nvfp4_col_final_sg_opt",
          &tk_localcta_quantize_mxfp8_row_nvfp4_col_final_sg_opt,
          py::arg("input"),
          py::arg("encode_centric") = true,
          py::arg("four_over_six_mae") = false,
          py::arg("col_data_stochastic_rounding") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_quantize_mxfp8_row_mxfp4_col",
          &tk_localcta_quantize_mxfp8_row_mxfp4_col,
          py::arg("input"));
    m.def("tk_localcta_quantize_direct_fp8_row_mxfp4_col",
          &tk_localcta_quantize_direct_fp8_row_mxfp4_col,
          py::arg("input"), py::arg("fp8_scale"));
    m.def("tk_localcta_rmsnorm_quantize_mxfp8_row_mxfp4_col_with_output",
          &tk_localcta_rmsnorm_quantize_mxfp8_row_mxfp4_col_with_output,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"));
    m.def("tk_localcta_rmsnorm_quantize_direct_fp8_row_mxfp4_col_with_output",
          &tk_localcta_rmsnorm_quantize_direct_fp8_row_mxfp4_col_with_output,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("fp8_scale"));
    m.def("tk_localcta_quantize_mxfp8_row_only",
          &tk_localcta_quantize_mxfp8_row_only,
          py::arg("input"));
    m.def("tk_localcta_rmsnorm_quantize_mxfp8_row_with_output",
          &tk_localcta_rmsnorm_quantize_mxfp8_row_with_output,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"));
    m.def("tk_localcta_rmsnorm_quantize_mxfp8_row_nvfp4_col_with_output_final_sg_opt",
          &tk_localcta_rmsnorm_quantize_mxfp8_row_nvfp4_col_with_output_final_sg_opt,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("encode_centric") = true,
          py::arg("four_over_six_mae") = false,
          py::arg("col_data_stochastic_rounding") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_rmsnorm_quantize_for_gemm_opt", &tk_localcta_rmsnorm_quantize_for_gemm_opt,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("return_transpose"),
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt",
          &tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("return_transpose"),
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0,
          py::arg("four_over_six_mae") = false);
    m.def("tk_localcta_rmsnorm_quantize_from_row_rms_partial_final_sg",
          &tk_localcta_rmsnorm_quantize_from_row_rms_partial_final_sg,
          py::arg("input"), py::arg("gamma"), py::arg("row_rms_partial"),
          py::arg("epsilon"), py::arg("return_transpose") = true,
          py::arg("encode_centric") = true);
    m.def("tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer",
          &tk_localcta_rmsnorm_quantize_for_gemm_row_prepared_col_outer,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("return_transpose"),
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_quantize_nhsd_wo_for_gemm",
          &tk_localcta_quantize_nhsd_wo_for_gemm,
          py::arg("input"), py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_raw", &tk_localcta_quantize_for_gemm_raw,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_fast", &tk_localcta_quantize_for_gemm_fast,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_weight_2d", &tk_localcta_quantize_weight_2d,
          py::arg("input"),
          "Quantize each 16x16 weight block once and transpose its exact payload");
    m.def("tk_localcta_quantize_for_gemm_row_prepared_col_outer",
          &tk_localcta_quantize_for_gemm_row_prepared_col_outer,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_raw_outer_tma",
          &tk_localcta_quantize_for_gemm_raw_outer_tma,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_prepared", &tk_localcta_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_prepared_maybe_borrow",
          &tk_localcta_quantize_for_gemm_prepared_maybe_borrow,
          py::arg("input"), py::arg("staging_input"),
          py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta2_quantize_for_gemm", &tk_localcta2_quantize_for_gemm,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta2_quantize_for_gemm_fast", &tk_localcta2_quantize_for_gemm_fast,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta2_quantize_for_gemm_prepared", &tk_localcta2_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_set_2cta_prepared_tuning", &tk_localcta_set_2cta_prepared_tuning,
          py::arg("threads"), py::arg("pipe_depth"), py::arg("shared_amax"));
    m.def("tk_localcta_get_2cta_prepared_tuning", &tk_localcta_get_2cta_prepared_tuning);
    m.def("tk_localcta_set_2cta_prepared_split2_tuning", &tk_localcta_set_2cta_prepared_split2_tuning,
          py::arg("threads"), py::arg("pipe_depth"), py::arg("shared_amax"));
    m.def("tk_localcta_get_2cta_prepared_split2_tuning", &tk_localcta_get_2cta_prepared_split2_tuning);
    m.def("tk_localcta_set_1cta_prepared_tuning", &tk_localcta_set_1cta_prepared_tuning,
          py::arg("threads"), py::arg("pipe_depth"));
    m.def("tk_localcta_get_1cta_prepared_tuning", &tk_localcta_get_1cta_prepared_tuning);
    m.def("tk_localcta_set_global_scale_num", &tk_localcta_set_global_scale_num,
          py::arg("value"));
    m.def("tk_localcta_get_global_scale_num", &tk_localcta_get_global_scale_num);
    m.def("tk_localcta_reset_global_scale_num", &tk_localcta_reset_global_scale_num);
    m.def("tk_localcta_quantize_for_gemm_alloc", &tk_localcta_quantize_for_gemm_alloc,
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_localcta_quantize_for_gemm_fast_alloc", &tk_localcta_quantize_for_gemm_fast_alloc,
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_localcta_quantize_for_gemm_prepared_alloc", &tk_localcta_quantize_for_gemm_prepared_alloc,
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_localcta_quantize_for_gemm_launch", &tk_localcta_quantize_for_gemm_launch,
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("row_sg_chunks"), py::arg("col_sg_chunks"));
    m.def("tk_localcta_quantize_for_gemm_fast_launch", &tk_localcta_quantize_for_gemm_fast_launch,
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("row_sg_chunks"), py::arg("col_sg_chunks"),
          py::arg("row_sc_prepared"), py::arg("col_sc_prepared"));
    m.def("tk_localcta_quantize_for_gemm_prepared_launch", &tk_localcta_quantize_for_gemm_prepared_launch,
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric"),
          py::arg("row_fp4"), py::arg("row_sc_prepared"), py::arg("col_fp4"), py::arg("col_sc_prepared"),
          py::arg("row_sg_chunks"), py::arg("col_sg_chunks"));
    m.def("tk_localcta_batched_quantize_for_gemm", &tk_localcta_batched_quantize_for_gemm,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta_batched_quantize_for_gemm_fast", &tk_localcta_batched_quantize_for_gemm_fast,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta_batched_quantize_for_gemm_prepared", &tk_localcta_batched_quantize_for_gemm_prepared,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta2_batched_quantize_for_gemm_prepared", &tk_localcta2_batched_quantize_for_gemm_prepared,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_col_only_prepared", &tk_localcta_quantize_col_only_prepared,
          py::arg("input"), py::arg("sg_tensor"));
    m.def("tk_localcta_quantize_col_only_prepared_launch_inplace", &tk_localcta_quantize_col_only_prepared_launch_inplace,
          py::arg("input"), py::arg("sg_tensor"),
          py::arg("col_fp4"), py::arg("col_sc_prepared"), py::arg("col_sg"));
    m.def("tk_localcta_fused_norm_quantize", &tk_localcta_fused_norm_quantize,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("with_silu") = false, py::arg("return_transpose") = true);
    m.def("tk_localcta_rmsnorm_to_bf16", &tk_localcta_rmsnorm_to_bf16,
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"));
    m.def("tk_localcta_silu_quantize_split_for_gemm", &tk_localcta_silu_quantize_split_for_gemm,
          py::arg("h1_raw"), py::arg("h3"));
    m.def("tk_localcta_silu_supports_paired_col_rht",
          &tk_localcta_silu_supports_paired_col_rht);
    m.def("tk_localcta_silu_quantize_split_for_gemm_paired_col_rht",
          &tk_localcta_silu_quantize_split_for_gemm_paired_col_rht,
          py::arg("h1_raw"), py::arg("h3"));
    m.def("tk_localcta_test_w2_transform_bf16_exact",
          &tk_localcta_test_w2_transform_bf16_exact,
          py::arg("h1_raw"), py::arg("h3"), py::arg("fast_divide"),
          py::arg("call_free_te_math") = false);
    m.def("tk_localcta_test_scale_divide_callfree",
          &tk_localcta_test_scale_divide_callfree,
          py::arg("numerator"), py::arg("denominator"));
    m.def("tk_localcta_sqrelu_quantize_for_gemm_prepared", &tk_localcta_sqrelu_quantize_for_gemm_prepared,
          py::arg("h1_raw"), py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer",
          &tk_localcta_sqrelu_quantize_for_gemm_row_prepared_col_outer,
          py::arg("h1_raw"), py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_sqrelu_quantize_row_only_prepared",
          &tk_localcta_sqrelu_quantize_row_only_prepared,
          py::arg("h1_raw"), py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_sqrelu_quantize_col_only_raw_outer",
          &tk_localcta_sqrelu_quantize_col_only_raw_outer,
          py::arg("h1_raw"), py::arg("encode_centric") = true);
    m.def("tk_localcta_sqrelu_deriv_quantize_for_gemm_prepared",
          &tk_localcta_sqrelu_deriv_quantize_for_gemm_prepared,
          py::arg("dh"), py::arg("h1_raw"), py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer",
          &tk_localcta_sqrelu_deriv_quantize_for_gemm_raw_outer,
          py::arg("dh"), py::arg("h1_raw"), py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_silu_deriv_split2_supports_rht",
          &tk_localcta_silu_deriv_split2_supports_rht);
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_alloc",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_alloc,
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_cat_alloc,
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_localcta_silu_deriv_split_bf16_launch_inplace",
          &tk_localcta_silu_deriv_split_bf16_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1"), py::arg("dh3_out"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace_legacy_state,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1"), py::arg("dh3_out"),
          py::arg("row_fp4_0"), py::arg("row_sc_0"),
          py::arg("col_fp4_0"), py::arg("col_sc_0"),
          py::arg("row_sg_0"), py::arg("col_sg_0"),
          py::arg("row_fp4_1"), py::arg("row_sc_1"),
          py::arg("col_fp4_1"), py::arg("col_sc_1"),
          py::arg("row_sg_1"), py::arg("col_sg_1"),
          py::arg("row_sg_chunk_0"), py::arg("col_sg_chunk_0"),
          py::arg("row_sg_chunk_1"), py::arg("col_sg_chunk_1"),
          py::arg("finalize_contract"),
          py::arg("data_stochastic_rounding"),
          py::arg("scale_stochastic_rounding"),
          py::arg("rng_seed"),
          py::arg("rng_subsequence_base"),
          py::arg("data_sr_axes"),
          py::arg("persistent_rng_state"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1"), py::arg("dh3_out"),
          py::arg("row_fp4_0"), py::arg("row_sc_0"),
          py::arg("col_fp4_0"), py::arg("col_sc_0"),
          py::arg("row_sg_0"), py::arg("col_sg_0"),
          py::arg("row_fp4_1"), py::arg("row_sc_1"),
          py::arg("col_fp4_1"), py::arg("col_sc_1"),
          py::arg("row_sg_1"), py::arg("col_sg_1"),
          py::arg("row_sg_chunk_0"), py::arg("col_sg_chunk_0"),
          py::arg("row_sg_chunk_1"), py::arg("col_sg_chunk_1"),
          py::arg("finalize_contract") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0,
          py::arg("data_sr_axes") = "both",
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("derivatives_precomputed") = false,
          py::arg("encode_centric") = true,
          py::arg("persistent_rng_state") = std::nullopt);
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm", &tk_localcta_silu_deriv_quantize_split_for_gemm,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_delayed",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_delayed,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax_outer",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_collect_tile_amax_outer,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("prev_row_amax_0"), py::arg("prev_col_amax_0"),
          py::arg("prev_row_amax_1"), py::arg("prev_col_amax_1"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("prev_row_amax_0"), py::arg("prev_col_amax_0"),
          py::arg("prev_row_amax_1"), py::arg("prev_col_amax_1"),
          py::arg("prev_row_sg_0"), py::arg("prev_col_sg_0"),
          py::arg("prev_row_sg_1"), py::arg("prev_col_sg_1"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_launch_inplace",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1"), py::arg("dh3_out"),
          py::arg("row_fp4_0"), py::arg("row_sc_0"),
          py::arg("col_fp4_0"), py::arg("col_sc_0"),
          py::arg("row_sg_0"), py::arg("col_sg_0"),
          py::arg("row_fp4_1"), py::arg("row_sc_1"),
          py::arg("col_fp4_1"), py::arg("col_sc_1"),
          py::arg("row_sg_1"), py::arg("col_sg_1"),
          py::arg("row_sg_chunk_0"), py::arg("col_sg_chunk_0"),
          py::arg("row_sg_chunk_1"), py::arg("col_sg_chunk_1"),
          py::arg("prev_row_amax_0"), py::arg("prev_col_amax_0"),
          py::arg("prev_row_amax_1"), py::arg("prev_col_amax_1"),
          py::arg("prev_row_sg_0"), py::arg("prev_col_sg_0"),
          py::arg("prev_row_sg_1"), py::arg("prev_col_sg_1"),
          py::arg("cur_row_amax_0"), py::arg("cur_row_amax_1"),
          py::arg("cur_row_sg_0"), py::arg("cur_col_sg_0"),
          py::arg("cur_row_sg_1"), py::arg("cur_col_sg_1"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("prev_row_amax_0"), py::arg("prev_col_amax_0"),
          py::arg("prev_row_amax_1"), py::arg("prev_col_amax_1"),
          py::arg("prev_row_sg_0"), py::arg("prev_col_sg_0"),
          py::arg("prev_row_sg_1"), py::arg("prev_col_sg_1"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect_launch_inplace",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_no_collect_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1"), py::arg("dh3_out"),
          py::arg("row_fp4_0"), py::arg("row_sc_0"),
          py::arg("col_fp4_0"), py::arg("col_sc_0"),
          py::arg("row_sg_0"), py::arg("col_sg_0"),
          py::arg("row_fp4_1"), py::arg("row_sc_1"),
          py::arg("col_fp4_1"), py::arg("col_sc_1"),
          py::arg("row_sg_1"), py::arg("col_sg_1"),
          py::arg("row_sg_chunk_0"), py::arg("col_sg_chunk_0"),
          py::arg("row_sg_chunk_1"), py::arg("col_sg_chunk_1"),
          py::arg("prev_row_amax_0"), py::arg("prev_col_amax_0"),
          py::arg("prev_row_amax_1"), py::arg("prev_col_amax_1"),
          py::arg("prev_row_sg_0"), py::arg("prev_col_sg_0"),
          py::arg("prev_row_sg_1"), py::arg("prev_col_sg_1"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_split_collect",
          &tk_localcta_silu_deriv_quantize_split_for_gemm_tile_delayed_outer_split_collect,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("prev_row_amax_0"), py::arg("prev_col_amax_0"),
          py::arg("prev_row_amax_1"), py::arg("prev_col_amax_1"),
          py::arg("prev_row_sg_0"), py::arg("prev_col_sg_0"),
          py::arg("prev_row_sg_1"), py::arg("prev_col_sg_1"));
    m.def("tk_localcta_group_quantize_for_gemm", &tk_localcta_group_quantize_for_gemm,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_for_gemm_final_sg", &tk_localcta_group_quantize_for_gemm_final_sg,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_split2_for_gemm_final_sg",
          &tk_localcta_group_quantize_split2_for_gemm_final_sg,
          py::arg("input0"), py::arg("input1"));
    m.def("tk_localcta_group_quantize_for_gemm_raw", &tk_localcta_group_quantize_for_gemm_raw,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_for_gemm_fast", &tk_localcta_group_quantize_for_gemm_fast,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_for_gemm_prepared", &tk_localcta_group_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_split_for_gemm_prepared", &tk_localcta_group_quantize_split_for_gemm_prepared,
          py::arg("input0"), py::arg("input1"));
    m.def("tk_localcta2_group_quantize_for_gemm_prepared", &tk_localcta2_group_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_dim1_for_gemm", &tk_localcta_group_quantize_dim1_for_gemm,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_group_quantize_dim1_for_gemm_fast", &tk_localcta_group_quantize_dim1_for_gemm_fast,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_group_quantize_dim1_for_gemm_prepared", &tk_localcta_group_quantize_dim1_for_gemm_prepared,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_concat_group_quantize_dim1_for_gemm_prepared",
          &tk_localcta_concat_group_quantize_dim1_for_gemm_prepared,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_group_quantize_dim1_split3_for_gemm", &tk_localcta_group_quantize_dim1_split3_for_gemm,
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("data_stochastic_rounding") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0,
          py::arg("data_sr_axes") = "both",
          py::arg("persistent_rng_state") = std::nullopt,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("encode_centric") = true);
    m.def("tk_localcta_split3_supports_paired_rht",
          &tk_localcta_split3_supports_paired_rht);
    m.def("tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64",
          &tk_localcta_group_quantize_dim1_split3_for_gemm_inverse_rope_live64,
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("rope_cs"), py::arg("rope_seq_len"),
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0,
          py::arg("data_sr_axes") = "both",
          py::arg("persistent_rng_state") = std::nullopt);
    m.def("tk_localcta_group_quantize_dim1_split3_rowphase_for_gemm",
          &tk_localcta_group_quantize_dim1_split3_rowphase_for_gemm,
          py::arg("input0"), py::arg("input1"), py::arg("input2"));
    m.def("tk_localcta_group_quantize_dim1_split3_finalize_col_inplace",
          &tk_localcta_group_quantize_dim1_split3_finalize_col_inplace,
          py::arg("col_sc_cat"), py::arg("col_sg_cat"),
          py::arg("col_sg_chunk_0"), py::arg("col_sg_chunk_1"), py::arg("col_sg_chunk_2"));
    m.def("tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc",
          &tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_alloc,
          py::arg("M"), py::arg("n0"), py::arg("n1"), py::arg("device"));
    m.def("tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch",
          &tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch,
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("col_fp4_cat"), py::arg("col_sc_prepared_cat"),
          py::arg("row_sg_cat"), py::arg("col_sg_cat"));
    m.def("tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace",
          &tk_localcta_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace,
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("col_fp4_cat"), py::arg("col_sc_prepared_cat"),
          py::arg("row_sg_cat"), py::arg("col_sg_cat"));
    m.def("tk_localcta_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace",
          &tk_localcta_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace,
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("row_sg_cat"));
    m.def("tk_localcta_group_quantize_dim1_split2_for_gemm_prepared", &tk_localcta_group_quantize_dim1_split2_for_gemm_prepared,
          py::arg("input0"), py::arg("input1"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("col_fp4_cat"), py::arg("col_sc_prepared_cat"),
          py::arg("row_sg_cat"), py::arg("col_sg_cat"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("col_fp4_cat"), py::arg("col_sc_prepared_cat"),
          py::arg("row_sg_cat"), py::arg("col_sg_cat"),
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence_base") = 0);
    m.def("tk_localcta_finalize_split2_for_gemm_prepared_inplace",
          &tk_localcta_finalize_split2_for_gemm_prepared_inplace,
          py::arg("row_sc_0"), py::arg("row_sg_chunk_0"), py::arg("row_sg_0"),
          py::arg("col_sc_0"), py::arg("col_sg_chunk_0"), py::arg("col_sg_0"),
          py::arg("row_sc_1"), py::arg("row_sg_chunk_1"), py::arg("row_sg_1"),
          py::arg("col_sc_1"), py::arg("col_sg_chunk_1"), py::arg("col_sg_1"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("col_fp4_cat"), py::arg("col_sc_prepared_cat"),
          py::arg("row_sg_cat"), py::arg("col_sg_cat"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("col_fp4_cat"), py::arg("col_sc_prepared_cat"),
          py::arg("row_sg_cat"), py::arg("col_sg_cat"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_row_for_gemm_prepared_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("row_sg_cat"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_row_bf16_for_gemm_prepared_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1_out"), py::arg("dh3_out"),
          py::arg("row_fp4_cat"), py::arg("row_sc_prepared_cat"),
          py::arg("row_sg_cat"));
    m.def("tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_alloc",
          &tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_alloc,
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch",
          &tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4_1"), py::arg("row_sc_prepared_1"), py::arg("row_sg_1"),
          py::arg("row_fp4_2"), py::arg("row_sc_prepared_2"), py::arg("row_sg_2"));
    m.def("tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch_inplace",
          &tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared_launch_inplace,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4_1"), py::arg("row_sc_prepared_1"), py::arg("row_sg_1"),
          py::arg("row_fp4_2"), py::arg("row_sc_prepared_2"), py::arg("row_sg_2"));
    m.def("tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared",
          &tk_localcta_silu_deriv_quantize_split_row_for_gemm_prepared,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage",
          &tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_v4_twostage,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_group_quantize_dim1_split3_for_gemm_prepared", &tk_localcta_group_quantize_dim1_split3_for_gemm_prepared,
          py::arg("input0"), py::arg("input1"), py::arg("input2"));
    m.def("tk_localcta2_group_quantize_dim1_for_gemm_prepared", &tk_localcta2_group_quantize_dim1_for_gemm_prepared,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_reconstruct_row", &tk_localcta_reconstruct_row,
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("row_sg_chunks"));
    m.def("tk_localcta_reconstruct_col", &tk_localcta_reconstruct_col,
          py::arg("col_fp4"), py::arg("col_sc"), py::arg("col_sg_chunks"));
}
