#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <tuple>
#include <vector>

#define TK_STANDALONE
#include "fused_localcta_quantize.cuh"
#include "direct_localcta_fused_quantize.cuh"
#include "persistent_localcta_silu_quantize.cuh"
#include "persistent_localcta_silu_deriv_quantize.cuh"
#include "localcta_reconstruct.cuh"
#include "../nvfp4_v5/silu_split_bf16.cuh"

using transformer_engine::dispatch::nvfp4::nvfp4_scale_t;
namespace py = pybind11;

namespace {

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
        float transformed = v * row_inv_rms * g;
        if (with_silu) {
            transformed = transformed / (1.0f + expf(-transformed));
        }
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

    const float denom = fmaxf(row_sg_tiles[row / 2], 1e-12f);
    const float numer = row_sg_chunk[row * sg_cols + (sc_col / 2)];
    const float ratio = numer / denom;
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

    const float denom = fmaxf(smem[0], 1e-12f);
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
        const float ratio = numer / denom;
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
    int64_t row_sc_stride1
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

    const float denom = fmaxf(smem[0], 1e-12f);
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
        const float ratio = numer / denom;
        const int64_t base = static_cast<int64_t>(row) * row_sc_stride0 +
                             static_cast<int64_t>(sc_col) * row_sc_stride1;
        const float v = static_cast<float>(row_sc[base + i]);
        row_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
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

    const float denom = fmaxf(smem[0], 1e-12f);
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
        const float ratio = numer / denom;
        const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
        const float v = static_cast<float>(col_sc[base + i]);
        col_sc[base + i] = static_cast<__nv_fp8_e4m3>(v * ratio);
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

    const float denom = fmaxf(col_sg_tiles[k / 2], 1e-12f);
    const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
    const float ratio = numer / denom;
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

    const float denom = fmaxf(col_sg[k / 2], 1e-12f);
    const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
    const float ratio = numer / denom;
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

    const float denom = fmaxf(col_sg[k / 2], 1e-12f);
    const float numer = col_sg_chunk[k * sg_rows + (sc_row / 2)];
    const float ratio = numer / denom;
    const int64_t base = ((int64_t)k * sc_rows + sc_row) * 512;
    for (int i = threadIdx.x; i < 512; i += BLOCK_SIZE) {
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
    int threads = 160;
    int pipe_depth = 1;
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

static LocalCTA1PreparedTuning& get_localcta1_prepared_tuning() {
    static LocalCTA1PreparedTuning tuning;
    return tuning;
}

static torch::Tensor& get_localcta_persistent_counter(torch::Device device) {
    static std::vector<torch::Tensor> counters;
    const int index = device.index();
    TORCH_CHECK(index >= 0, "device must have a concrete CUDA index");
    if (index >= static_cast<int>(counters.size())) {
        counters.resize(index + 1);
    }
    auto& counter = counters[index];
    if (!counter.defined()) {
        counter = torch::empty({1}, torch::dtype(torch::kInt32).device(device));
    }
    return counter;
}

static float& get_localcta_global_scale_num_host() {
    static float value = tk_localcta::LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM;
    return value;
}

static bool use_localcta_fused_direct_experimental() {
    const char* value = std::getenv("USE_TK_LOCALCTA_FUSED_DIRECT");
    return value != nullptr && std::string(value) == "1";
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
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    return total_macro_tiles > 0 && total_macro_tiles <= 1024;
}

static bool should_use_localcta1_prepared_auto(int64_t M, int64_t K, bool return_transpose) {
    if (return_transpose) {
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

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH>
static void launch_localcta_quant_prepared_tuned(
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
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_1cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_kernel_prepared_tuned<
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
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, K, blocks_X, total_tiles);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant_prepared_tuned_dispatch(
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
    const auto cfg = get_localcta1_prepared_tuning();
    if (cfg.threads == 160 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else {
        TORCH_CHECK(false, "Unsupported localCTA1 prepared tuning config: threads=", cfg.threads,
                    " pipe_depth=", cfg.pipe_depth);
    }
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

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
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

    auto kernel = fused_localcta_quantize_kernel<RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, K, args, write_raw_scales, write_prepared);
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
    torch::Tensor col_sg_chunk_2
) {
    using namespace tk_localcta;
    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1);
    const int64_t n1 = input1.size(1);
    const int64_t n2 = input2.size(1);

    alignas(64) CUtensorMap tmap_in0{}, tmap_in1{}, tmap_in2{};
    alignas(64) CUtensorMap tmap_out0{}, tmap_out1{}, tmap_out2{};
    alignas(64) CUtensorMap tmap_out_t0{}, tmap_out_t1{}, tmap_out_t2{};
    alignas(64) CUtensorMap tmap_sc_row0{}, tmap_sc_row1{}, tmap_sc_row2{};
    alignas(64) CUtensorMap tmap_sc_col0{}, tmap_sc_col1{}, tmap_sc_col2{};

    create_tma_2d(tmap_in0, input0.data_ptr(), M, n0, BUFF_DIM_Y, BUFF_DIM_X, n0, 16);
    create_tma_2d(tmap_in1, input1.data_ptr(), M, n1, BUFF_DIM_Y, BUFF_DIM_X, n1, 16);
    create_tma_2d(tmap_in2, input2.data_ptr(), M, n2, BUFF_DIM_Y, BUFF_DIM_X, n2, 16);

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

    auto kernel = fused_localcta_quantize_split3_raw_kernel<true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in0, tmap_in1, tmap_in2,
        tmap_out0, tmap_out1, tmap_out2,
        tmap_out_t0, tmap_out_t1, tmap_out_t2,
        tmap_sc_row0, tmap_sc_row1, tmap_sc_row2,
        tmap_sc_col0, tmap_sc_col1, tmap_sc_col2,
        row_sg_chunk_0.data_ptr<float>(),
        row_sg_chunk_1.data_ptr<float>(),
        row_sg_chunk_2.data_ptr<float>(),
        col_sg_chunk_0.data_ptr<float>(),
        col_sg_chunk_1.data_ptr<float>(),
        col_sg_chunk_2.data_ptr<float>(),
        M, args, blocks_X0, blocks_X1, blocks_X2);
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
    torch::Tensor col_sg_chunk_1
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
    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input0.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());

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

    auto kernel = fused_localcta_quantize_split2_raw_kernel<true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in0, tmap_in1,
        tmap_out0, tmap_out1,
        tmap_out_t0, tmap_out_t1,
        tmap_sc_row0, tmap_sc_row1,
        tmap_sc_col0, tmap_sc_col1,
        row_sg_chunk_0.data_ptr<float>(),
        row_sg_chunk_1.data_ptr<float>(),
        col_sg_chunk_0.data_ptr<float>(),
        col_sg_chunk_1.data_ptr<float>(),
        M, args, blocks_X0, blocks_X1);
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
        row_sg = torch::empty({M / 256, 1}, opts_f32);
        col_sg = return_transpose ? torch::empty({1, K / 256}, opts_f32)
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

    if (!col_sc.defined() || col_sc.numel() == 0) {
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
        row_sc.stride(1));
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "finalize_row_sc_strided_kernel failed: ", cudaGetErrorString(err));
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
    torch::Tensor col_sg_1
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int block = get_v3_split2_finalize_block_size();
    finalize_row_quant_contract_v3(row_sc_0, row_sg_chunk_0, row_sg_0);
    finalize_row_quant_contract_v3(row_sc_1, row_sg_chunk_1, row_sg_1);

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
    torch::Tensor col_sg_2
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t M = row_sc_0.size(0) * 128;
    const int64_t row_tiles = M / 256;
    const int sc_rows = static_cast<int>(col_sc_0.size(1));
    const int sg_rows = static_cast<int>(col_sg_chunk_0.size(1));
    const int k_chunks_0 = static_cast<int>(col_sc_0.size(0));
    const int k_chunks_1 = static_cast<int>(col_sc_1.size(0));
    const int k_chunks_2 = static_cast<int>(col_sc_2.size(0));
    const int max_col_tiles = std::max(std::max((k_chunks_0 + 1) / 2, (k_chunks_1 + 1) / 2), (k_chunks_2 + 1) / 2);
    const int max_k_chunks = std::max(std::max(k_chunks_0, k_chunks_1), k_chunks_2);

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

    dim3 col_rescale_grid(static_cast<unsigned int>(max_k_chunks),
                          static_cast<unsigned int>(sc_rows),
                          3u);
    rescale_col_sc_split3_kernel<<<col_rescale_grid, 256, 0, stream>>>(
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
        TORCH_CHECK(err == cudaSuccess, "rescale_col_sc_split3_kernel failed: ", cudaGetErrorString(err));
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

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(h1_raw.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    const int dshmem = tk_localcta_fused_direct::direct_fused_single_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_fused_direct::localcta_silu_quantize_split_direct_kernel<RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    const int grid = persistent_grid_for_kernel(kernel, tk_localcta::THREADS, dshmem, total_tiles);

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
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4_1, row_sc_prepared_1, col_fp4_1, col_sc_prepared_1,
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1);
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4_2, row_sc_prepared_2, col_fp4_2, col_sc_prepared_2,
        tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2);

    constexpr int threads = tk_localcta_persistent_silu_deriv::PRODUCER_CONSUMER_THREADS;
    const int dshmem =
        tk_localcta_persistent_silu_deriv::persistent_localcta_silu_deriv_quant_smem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_persistent_silu_deriv::localcta_tma_silu_deriv_quantize_kernel<
        threads, RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(dh.device()));
    auto* work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    tk_localcta::LocalCTAPersistentArgs args{
        .work_counter = work_counter_ptr,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles,
    };
    const int grid = persistent_grid_for_kernel(kernel, threads, dshmem, total_tiles);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
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

    alignas(64) CUtensorMap tmap_out1{}, tmap_out1_t{}, tmap_sc_row_prepared1{}, tmap_sc_col_prepared1{};
    alignas(64) CUtensorMap tmap_out2{}, tmap_out2_t{}, tmap_sc_row_prepared2{}, tmap_sc_col_prepared2{};
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4_1, row_sc_prepared_1, col_fp4_1, col_sc_prepared_1,
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1);
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4_2, row_sc_prepared_2, col_fp4_2, col_sc_prepared_2,
        tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2);

    const int dshmem = tk_localcta_fused_direct::direct_fused_dual_shmem_size<RETURN_TRANSPOSE>();
    auto kernel = tk_localcta_fused_direct::localcta_silu_deriv_quantize_split_direct_kernel<RETURN_TRANSPOSE>;
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
    kernel<<<grid, tk_localcta::THREADS, dshmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1,
        row_sg_1.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_1.data_ptr<float>() : row_sg_1.data_ptr<float>(),
        tmap_out2, tmap_out2_t, tmap_sc_row_prepared2, tmap_sc_col_prepared2,
        row_sg_2.data_ptr<float>(),
        RETURN_TRANSPOSE ? col_sg_2.data_ptr<float>() : row_sg_2.data_ptr<float>(),
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
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
        row_fp4_1, row_sc_prepared_1, col_fp4_1, col_sc_prepared_1,
        tmap_out1, tmap_out1_t, tmap_sc_row_prepared1, tmap_sc_col_prepared1);
    create_prepared_output_tmaps<RETURN_TRANSPOSE>(
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
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
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
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
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
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
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
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
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
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
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

torch::Tensor reconstruct_rowwise_impl(
    torch::Tensor fp4,
    torch::Tensor sc,
    torch::Tensor sg,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(fp4.is_cuda() && sc.is_cuda() && sg.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(fp4.scalar_type() == torch::kFloat4_e2m1fn_x2, "fp4 tensor dtype mismatch");
    TORCH_CHECK(sc.scalar_type() == torch::kFloat8_e4m3fn, "scale tensor dtype mismatch");
    TORCH_CHECK(sg.scalar_type() == torch::kFloat32, "sg tensor dtype mismatch");

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
        (int)rows, (int)cols, (int)sg.size(0), (int)sg.size(1));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_reconstruct failed: ",
                cudaGetErrorString(err));
    return out;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm(torch::Tensor input,
                              bool return_transpose,
                              bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs_v3(input.size(0), input.size(1), return_transpose, input.device());
    const bool use_2cta = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    quantize_into_outputs_v3(input, return_transpose, encode_centric,
                             row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                             torch::Tensor(), torch::Tensor(), use_2cta);
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
    auto col_sg_cat = torch::cat(col_sg_list, 0);

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
        can_use_tileglobal_concat = can_use_tileglobal_concat && ((cols_i % 128) == 0);
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
            const int64_t sg_tiles = cols_i / 256;
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

    launch_localcta_silu_deriv_split2_quant_prepared(
        dh, h3, h1_raw,
        row_fp4_cat, row_sc_prepared_cat,
        col_fp4_cat, col_sc_prepared_cat,
        row_sg_cat, col_sg_cat);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "tk_localcta_silu_deriv_group_quantize_dim1_split2_for_gemm_prepared_launch_inplace failed: ",
                cudaGetErrorString(err));
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
            torch::Tensor col_sg_cat;
            if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
                col_sg_cat = torch::empty({(input0.size(1) + input1.size(1)) / 256, input0.size(0) / 256}, opts_f32);
            } else {
                col_sg_cat = torch::empty({1, (input0.size(1) + input1.size(1)) / 256}, opts_f32);
            }

            col_fp4_0 = col_fp4_cat.narrow(0, 0, input0.size(1));
            col_fp4_1 = col_fp4_cat.narrow(0, input0.size(1), input1.size(1));
            col_sc_0 = col_sc_cat.narrow(0, 0, input0.size(1) / 128);
            col_sc_1 = col_sc_cat.narrow(0, input0.size(1) / 128, input1.size(1) / 128);
            if (get_v3_contract_mode() == V3ContractMode::TileGrid256) {
                col_sg_0 = col_sg_cat.narrow(0, 0, input0.size(1) / 256);
                col_sg_1 = col_sg_cat.narrow(0, input0.size(1) / 256, input1.size(1) / 256);
            } else {
                col_sg_0 = col_sg_cat.narrow(1, 0, input0.size(1) / 256);
                col_sg_1 = col_sg_cat.narrow(1, input0.size(1) / 256, input1.size(1) / 256);
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
        tk_localcta_quantize_for_gemm(normed, return_transpose, true);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_fused_norm_quantize failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, inv_rms
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_quantize_split_for_gemm(
    torch::Tensor h1_raw,
    torch::Tensor h3
) {
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(), "h1_raw and h3 must have identical shape");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
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
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
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
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());

    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);

    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
        M, H, stream);

    auto split2_quant =
        tk_localcta_batched_quantize_for_gemm({dh1, dh3_out}, true, true);
    auto row_fp4s = std::get<0>(split2_quant);
    auto row_scs = std::get<1>(split2_quant);
    auto col_fp4s = std::get<2>(split2_quant);
    auto col_scs = std::get<3>(split2_quant);
    auto row_sgs = std::get<4>(split2_quant);
    auto col_sgs = std::get<5>(split2_quant);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_silu_deriv_quantize_split_for_gemm failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4s[0], row_scs[0], col_fp4s[0], col_scs[0],
        row_sgs[0], col_sgs[0],
        row_fp4s[1], row_scs[1], col_fp4s[1], col_scs[1],
        row_sgs[1], col_sgs[1]
    );
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

    if (multiinput_mode == V3MultiInputQuantMode::Loop) {
        auto [row_fp4_0, row_sc_0, col_fp4_0, col_sc_0, row_sg_0, col_sg_0] =
            tk_localcta_quantize_for_gemm(input0, true, true);
        auto [row_fp4_1, row_sc_1, col_fp4_1, col_sc_1, row_sg_1, col_sg_1] =
            tk_localcta_quantize_for_gemm(input1, true, true);
        auto [row_fp4_2, row_sc_2, col_fp4_2, col_sc_2, row_sg_2, col_sg_2] =
            tk_localcta_quantize_for_gemm(input2, true, true);

        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        auto row_fp4_cat = torch::cat({row_fp4_0, row_fp4_1, row_fp4_2}, 1);
        auto row_sc_cat = torch::cat({row_sc_0, row_sc_1, row_sc_2}, 1);
        auto row_sg_cat = torch::empty({M / 256, 3}, opts_f32);
        copy_cat_dim1_contiguous({row_sg_0, row_sg_1, row_sg_2}, row_sg_cat);
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
    auto col_sg_cat = torch::empty({1, total_n / 256}, opts_f32);

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
    auto row_sg_0 = torch::empty({M / 256, 1}, opts_f32);
    auto row_sg_1 = torch::empty({M / 256, 1}, opts_f32);
    auto row_sg_2 = torch::empty({M / 256, 1}, opts_f32);
    auto col_sg_0 = col_sg_cat.narrow(1, 0, n0 / 256);
    auto col_sg_1 = col_sg_cat.narrow(1, n0 / 256, n1 / 256);
    auto col_sg_2 = col_sg_cat.narrow(1, (n0 + n1) / 256, n2 / 256);

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
        const int64_t row_tiles = M / 256;
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

    std::vector<torch::Tensor> row_fp4_list{row_fp4_0, row_fp4_1, row_fp4_2};
    std::vector<torch::Tensor> row_sc_list{row_sc_0, row_sc_1, row_sc_2};
    std::vector<torch::Tensor> row_sg_list{row_sg_0, row_sg_1, row_sg_2};
    std::vector<torch::Tensor> col_fp4_list{col_fp4_0, col_fp4_1, col_fp4_2};
    std::vector<torch::Tensor> col_sc_list{col_sc_0, col_sc_1, col_sc_2};
    std::vector<torch::Tensor> col_sg_list{col_sg_0, col_sg_1, col_sg_2};
    auto row_sg_cat = torch::empty({M / 256, 3}, opts_f32);
    copy_cat_dim1_contiguous({row_sg_0, row_sg_1, row_sg_2}, row_sg_cat);

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
    auto device = input0.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4_cat = torch::empty({M, total_n / 2}, opts_fp4);
    auto row_sc_cat = torch::empty({M / 128, total_n / 64, 512}, opts_fp8);
    auto col_fp4_cat = torch::empty({total_n, M / 2}, opts_fp4);
    auto col_sc_cat = torch::empty({total_n / 128, M / 64, 512}, opts_fp8);
    auto col_sg_cat = torch::empty({1, total_n / 256}, opts_f32);

    auto row_fp4_0 = row_fp4_cat.narrow(1, 0, n0 / 2);
    auto row_fp4_1 = row_fp4_cat.narrow(1, n0 / 2, n1 / 2);
    auto row_fp4_2 = row_fp4_cat.narrow(1, (n0 + n1) / 2, n2 / 2);
    auto row_sc_0 = row_sc_cat.narrow(1, 0, n0 / 64);
    auto row_sc_1 = row_sc_cat.narrow(1, n0 / 64, n1 / 64);
    auto row_sc_2 = row_sc_cat.narrow(1, (n0 + n1) / 64, n2 / 64);
    auto row_sg_0 = torch::empty({M / 256, 1}, opts_f32);
    auto row_sg_1 = torch::empty({M / 256, 1}, opts_f32);
    auto row_sg_2 = torch::empty({M / 256, 1}, opts_f32);
    auto row_sg_cat = torch::empty({M / 256, 3}, opts_f32);

    auto col_sg_0 = col_sg_cat.narrow(1, 0, n0 / 256);
    auto col_sg_1 = col_sg_cat.narrow(1, n0 / 256, n1 / 256);
    auto col_sg_2 = col_sg_cat.narrow(1, (n0 + n1) / 256, n2 / 256);
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
        const int64_t row_tiles = M / 256;
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

    copy_cat_dim1_contiguous({row_sg_0, row_sg_1, row_sg_2}, row_sg_cat);

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
    auto col_sg_0 = col_sg_cat.narrow(1, 0, n0 / 256);
    auto col_sg_1 = col_sg_cat.narrow(1, n0 / 256, n1 / 256);
    auto col_sg_2 = col_sg_cat.narrow(1, (n0 + n1) / 256, n2 / 256);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tk_localcta_quantize_for_gemm", &tk_localcta_quantize_for_gemm,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_fast", &tk_localcta_quantize_for_gemm_fast,
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
    m.def("tk_localcta_silu_quantize_split_for_gemm", &tk_localcta_silu_quantize_split_for_gemm,
          py::arg("h1_raw"), py::arg("h3"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm", &tk_localcta_silu_deriv_quantize_split_for_gemm,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_group_quantize_for_gemm", &tk_localcta_group_quantize_for_gemm,
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
          py::arg("input0"), py::arg("input1"), py::arg("input2"));
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
    m.def("tk_localcta_group_quantize_dim1_split3_for_gemm_prepared", &tk_localcta_group_quantize_dim1_split3_for_gemm_prepared,
          py::arg("input0"), py::arg("input1"), py::arg("input2"));
    m.def("tk_localcta2_group_quantize_dim1_for_gemm_prepared", &tk_localcta2_group_quantize_dim1_for_gemm_prepared,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_reconstruct_row", &tk_localcta_reconstruct_row,
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("row_sg_chunks"));
    m.def("tk_localcta_reconstruct_col", &tk_localcta_reconstruct_col,
          py::arg("col_fp4"), py::arg("col_sc"), py::arg("col_sg_chunks"));
}
