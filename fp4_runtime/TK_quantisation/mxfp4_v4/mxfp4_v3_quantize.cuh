/*************************************************************************
 * MXFP4 v3 Quantize — Single-phase pipelined kernel
 *
 * Based on NVFP4 v5 architecture but simplified:
 *   - NO global amax phase (MXFP4 uses per-32-element E8M0 scales)
 *   - Single-phase: TMA load → quantize → TMA store, pipelined
 *   - 4 sub-tiles of 64×64 within each 128×128 chunk
 *   - Double-buffered FP4 output (write one while TMA-storing the other)
 *   - Prefetch tile t+2 while processing tile t
 *   - Persistent work-stealing for large tensors
 *
 * SMEM budget (128×128 chunk):
 *   Input:  4 × 64×64 × 2B = 32 KB  (all sub-tiles)
 *   FP4:    2 × 64×32 × 1B =  4 KB  (double-buffered)
 *   Scales: 128×4 uint8     = 512 B  (E8M0, 4 blocks of 32 per 128)
 *   Total:                  ≈ 37 KB  → ~6 CTAs/SM on GB200 (228KB)
 *************************************************************************/

#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cstdio>

#define TK_STANDALONE
#include "../nvfp4_v5/util/cast_common.h"
#include "../nvfp4_v5/util/curanddx.hpp"
#include "../nvfp4_v5/util/ptx.cuh"
#include "../nvfp4_v5/util/utils.cuh"
#include "../nvfp4_v5/util/math.h"

namespace mxfp4_v3 {

using namespace transformer_engine;
using namespace transformer_engine::ptx;

// ═══════════════════════════════════════════════════════════════════
// Configuration — mirrors NV v5 sub-tile layout
// ═══════════════════════════════════════════════════════════════════

static constexpr int CHUNK_DIM   = 128;
static constexpr int TILE_DIM    = 64;       // 64×64 sub-tiles
static constexpr int TILES_Y     = 2;        // 128/64
static constexpr int TILES_X     = 2;
static constexpr int NUM_TILES   = 4;
static constexpr int THREADS     = 128;
static constexpr int MX_BLOCK    = 32;       // MXFP4 scale block (32 elements)
static constexpr int MX_BLOCKS_PER_CHUNK = CHUNK_DIM / MX_BLOCK;  // 4

// Thread layout for rowwise quantization
static constexpr int ELTS_PER_THREAD = 16;                        // 16 bf16 per thread
static constexpr int THREADS_X   = TILE_DIM / ELTS_PER_THREAD;   // 4
static constexpr int THREADS_Y   = THREADS / THREADS_X;          // 32
static constexpr int ITERATIONS  = TILE_DIM / THREADS_Y;         // 2

static constexpr int PACK_SIZE   = 8;        // 8 elements per vectorised load
static constexpr int WAVES       = ELTS_PER_THREAD / PACK_SIZE;  // 2
static constexpr int RMS_BLOCK_THREADS = 256;

// Double-buffered output
static constexpr int BUFFS_OUT   = 2;
static constexpr int OUT_DIM_Y   = TILE_DIM;
static constexpr int OUT_DIM_X   = (TILE_DIM * 4) / 8;  // FP4 packed: 64 elts -> 32 bytes
static constexpr int OUT_SIZE    = OUT_DIM_Y * OUT_DIM_X;

// Scales: 128 rows × 4 blocks (per chunk)
static constexpr int SCALES_PER_CHUNK = CHUNK_DIM / MX_BLOCK;  // 4

// SMEM bank conflict avoidance
static constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;
static constexpr int THREADS_PER_BANK  = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════
using IType  = bf16;
using IType2 = typename ptx::FPx2<IType>;
using InputBuf3D   = IType[NUM_TILES][TILE_DIM][TILE_DIM];
using InputBuf2x3D = IType2[NUM_TILES][TILE_DIM][TILE_DIM / 2];
using OutputBuf3D  = fp4e2m1x2[BUFFS_OUT][OUT_DIM_Y][OUT_DIM_X];
using TileInputBuf2D = IType[TILE_DIM][TILE_DIM];
using TileInputBuf2x2D = IType2[TILE_DIM][TILE_DIM / 2];
using Split2OutputBuf4D = fp4e2m1x2[4][OUT_DIM_Y][OUT_DIM_X];

// Persistent kernel args
struct PersistentArgs {
    unsigned int* work_counter;
    int tiles_X, tiles_Y, total_tiles;
};

// Group quantize args (max 16 groups)
static constexpr int MAX_GROUPS = 16;
struct GroupArgs {
    int num_groups;
    int boundaries[MAX_GROUPS + 1];
    uint8_t* scale_ptrs[MAX_GROUPS];
};

struct RopeLive64Desc {
    const float2* cs;
    int seq_mask;
};

// ═══════════════════════════════════════════════════════════════════
// RMSNorm helpers
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum_fast(float val) {
    __shared__ float warp_vals[BLOCK_SIZE / 32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_vals[warp] = val;
    }
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? warp_vals[threadIdx.x] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

template<int BLOCK_SIZE = RMS_BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_SIZE)
compute_inv_rms_kernel(
    const IType* __restrict__ x,
    float* __restrict__ inv_rms_out,
    float epsilon,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const IType* row_x = x + static_cast<int64_t>(row) * cols;
    float sum_sq = 0.0f;
    constexpr int elems_per_thread = 8;
    const int vec_count = cols / elems_per_thread;

    for (int i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
        const int off = i * elems_per_thread;
        #pragma unroll
        for (int k = 0; k < elems_per_thread; ++k) {
            const float v = __bfloat162float(row_x[off + k]);
            sum_sq += v * v;
        }
    }

    const float total_sq = block_reduce_sum_fast<BLOCK_SIZE>(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(total_sq / cols + epsilon);
        inv_rms_out[row] = s_inv_rms;
    }
}

__device__ __forceinline__ void apply_rmsnorm_tile_inplace(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ gamma_chunk,
    const float* __restrict__ inv_rms_chunk,
    const int buff_in,
    const int stage_Y,
    const int stage_X
) {
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;
    const int tile_row_offset = stage_Y * TILE_DIM;
    const int tile_col_offset = stage_X * TILE_DIM;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const float row_inv_rms = inv_rms_chunk[tile_row_offset + row];

        #pragma unroll
        for (int e = 0; e < ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            const float x_val = __bfloat162float(sIn[buff_in][row][col]);
            const float g_val = __bfloat162float(gamma_chunk[tile_col_offset + col]);
            const float normed_val = x_val * row_inv_rms * g_val;
            sIn[buff_in][row][col] = __float2bfloat16_rn(normed_val);
        }
    }
}

__device__ __forceinline__ void apply_rmsnorm_tile_inplace_transposed_load(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ gamma_chunk,
    const float* __restrict__ inv_rms_chunk,
    const int buff_in,
    const int stage_Y,
    const int stage_X
) {
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;
    const int tile_row_offset = stage_X * TILE_DIM;
    const int tile_col_offset = stage_Y * TILE_DIM;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const float row_inv_rms = inv_rms_chunk[tile_row_offset + row];

        #pragma unroll
        for (int e = 0; e < ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            const float x_val = __bfloat162float(sIn[buff_in][row][col]);
            const float g_val = __bfloat162float(gamma_chunk[tile_col_offset + col]);
            const float normed_val = x_val * row_inv_rms * g_val;
            sIn[buff_in][row][col] = __float2bfloat16_rn(normed_val);
        }
    }
}

__device__ __forceinline__ void store_chunk_value(
    IType* __restrict__ sIn_ptr,
    int row,
    int col,
    IType value
) {
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    const int tile_y = row / TILE_DIM;
    const int tile_x = col / TILE_DIM;
    const int tile = tile_y * TILES_X + tile_x;
    sIn[tile][row % TILE_DIM][col % TILE_DIM] = value;
}

__device__ __forceinline__ void apply_inverse_rope_tile_inplace_live64(
    IType* __restrict__ sIn_ptr,
    const RopeLive64Desc& rope,
    const int buff_in,
    const int stage_Y,
    const int stage_X,
    const int input_block_offset_Y,
    const int input_block_offset_X
) {
    auto& sIn2x = *reinterpret_cast<InputBuf2x3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int pair_thread_offset_X = tid_X * (ELTS_PER_THREAD / 2);
    const int tile_row_offset = input_block_offset_Y + stage_Y * TILE_DIM;
    const int tile_pair_offset = (input_block_offset_X + stage_X * TILE_DIM) >> 1;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const int seq_idx = (tile_row_offset + row) & rope.seq_mask;
        const int rope_row_offset = seq_idx * 32;

        #pragma unroll
        for (int p = 0; p < ELTS_PER_THREAD / 2; ++p) {
            const int pair_col = pair_thread_offset_X + p;
            const float2 cs = rope.cs[rope_row_offset + ((tile_pair_offset + pair_col) & 31)];
            auto* packed_ptr = reinterpret_cast<__nv_bfloat162*>(&sIn2x[buff_in][row][pair_col]);
            const float2 packed = __bfloat1622float2(*packed_ptr);
            *packed_ptr = __float22bfloat162_rn(make_float2(
                packed.x * cs.x + packed.y * cs.y,
                packed.y * cs.x - packed.x * cs.y));
        }
    }
    __syncthreads();
}

struct DualLocalMax {
    float a;
    float b;
};

__device__ __forceinline__ void load_silu_mul_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ h3,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * cols + global_col;
            const int2 a = *reinterpret_cast<const int2*>(h1_raw + base);
            const int2 b = *reinterpret_cast<const int2*>(h3 + base);

            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);

            const float2 a0f = __bfloat1622float2(a0);
            const float2 a1f = __bfloat1622float2(a1);
            const float2 b0f = __bfloat1622float2(b0);
            const float2 b1f = __bfloat1622float2(b1);

            const float sig0x = __fdividef(1.0f, 1.0f + __expf(-a0f.x));
            const float sig0y = __fdividef(1.0f, 1.0f + __expf(-a0f.y));
            const float sig1x = __fdividef(1.0f, 1.0f + __expf(-a1f.x));
            const float sig1y = __fdividef(1.0f, 1.0f + __expf(-a1f.y));

            const __nv_bfloat162 out0 = __float22bfloat162_rn(make_float2(
                (a0f.x * sig0x) * b0f.x,
                (a0f.y * sig0y) * b0f.y));
            const __nv_bfloat162 out1 = __float22bfloat162_rn(make_float2(
                (a1f.x * sig1x) * b1f.x,
                (a1f.y * sig1y) * b1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, reinterpret_cast<const IType&>(out0.x));
            store_chunk_value(sIn_ptr, row, col + 1, reinterpret_cast<const IType&>(out0.y));
            store_chunk_value(sIn_ptr, row, col + 2, reinterpret_cast<const IType&>(out1.x));
            store_chunk_value(sIn_ptr, row, col + 3, reinterpret_cast<const IType&>(out1.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    const float sig = 1.0f / (1.0f + __expf(-v1));
                    out = __float2bfloat16_rn((v1 * sig) * v3);
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_silu_mul_chunk_direct_save_sigmoid(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ h3,
    IType* __restrict__ sig_h1,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * cols + global_col;
            const int2 a = *reinterpret_cast<const int2*>(h1_raw + base);
            const int2 b = *reinterpret_cast<const int2*>(h3 + base);

            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);

            const float2 a0f = __bfloat1622float2(a0);
            const float2 a1f = __bfloat1622float2(a1);
            const float2 b0f = __bfloat1622float2(b0);
            const float2 b1f = __bfloat1622float2(b1);

            const float sig0x = 1.0f / (1.0f + __expf(-a0f.x));
            const float sig0y = 1.0f / (1.0f + __expf(-a0f.y));
            const float sig1x = 1.0f / (1.0f + __expf(-a1f.x));
            const float sig1y = 1.0f / (1.0f + __expf(-a1f.y));

            const __nv_bfloat162 sig0 = __float22bfloat162_rn(make_float2(sig0x, sig0y));
            const __nv_bfloat162 sig1 = __float22bfloat162_rn(make_float2(sig1x, sig1y));
            int2 sig_store;
            sig_store.x = *reinterpret_cast<const int*>(&sig0);
            sig_store.y = *reinterpret_cast<const int*>(&sig1);
            *reinterpret_cast<int2*>(sig_h1 + base) = sig_store;

            const __nv_bfloat162 out0 = __float22bfloat162_rn(make_float2(
                (a0f.x * sig0x) * b0f.x,
                (a0f.y * sig0y) * b0f.y));
            const __nv_bfloat162 out1 = __float22bfloat162_rn(make_float2(
                (a1f.x * sig1x) * b1f.x,
                (a1f.y * sig1y) * b1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, reinterpret_cast<const IType&>(out0.x));
            store_chunk_value(sIn_ptr, row, col + 1, reinterpret_cast<const IType&>(out0.y));
            store_chunk_value(sIn_ptr, row, col + 2, reinterpret_cast<const IType&>(out1.x));
            store_chunk_value(sIn_ptr, row, col + 3, reinterpret_cast<const IType&>(out1.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out = __float2bfloat16_rn(0.0f);
                IType sig_out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    const float sig = 1.0f / (1.0f + __expf(-v1));
                    sig_out = __float2bfloat16_rn(sig);
                    out = __float2bfloat16_rn((v1 * sig) * v3);
                    sig_h1[offset] = sig_out;
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_sqrelu_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * cols + global_col;
            const int2 a = *reinterpret_cast<const int2*>(h1_raw + base);

            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);

            const float2 a0f = __bfloat1622float2(a0);
            const float2 a1f = __bfloat1622float2(a1);

            const __nv_bfloat162 out0 = __float22bfloat162_rn(make_float2(
                a0f.x > 0.0f ? a0f.x * a0f.x : 0.0f,
                a0f.y > 0.0f ? a0f.y * a0f.y : 0.0f));
            const __nv_bfloat162 out1 = __float22bfloat162_rn(make_float2(
                a1f.x > 0.0f ? a1f.x * a1f.x : 0.0f,
                a1f.y > 0.0f ? a1f.y * a1f.y : 0.0f));

            store_chunk_value(sIn_ptr, row, col + 0, reinterpret_cast<const IType&>(out0.x));
            store_chunk_value(sIn_ptr, row, col + 1, reinterpret_cast<const IType&>(out0.y));
            store_chunk_value(sIn_ptr, row, col + 2, reinterpret_cast<const IType&>(out1.x));
            store_chunk_value(sIn_ptr, row, col + 3, reinterpret_cast<const IType&>(out1.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float v = __bfloat162float(h1_raw[offset]);
                    out = __float2bfloat16_rn(v > 0.0f ? v * v : 0.0f);
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_sqrelu_deriv_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ dh,
    const IType* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * cols + global_col;
            const int2 d = *reinterpret_cast<const int2*>(dh + base);
            const int2 a = *reinterpret_cast<const int2*>(h1_raw + base);

            const __nv_bfloat162 d0 = *reinterpret_cast<const __nv_bfloat162*>(&d.x);
            const __nv_bfloat162 d1 = *reinterpret_cast<const __nv_bfloat162*>(&d.y);
            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);

            const float2 d0f = __bfloat1622float2(d0);
            const float2 d1f = __bfloat1622float2(d1);
            const float2 a0f = __bfloat1622float2(a0);
            const float2 a1f = __bfloat1622float2(a1);

            const __nv_bfloat162 out0 = __float22bfloat162_rn(make_float2(
                a0f.x > 0.0f ? (2.0f * d0f.x) * a0f.x : 0.0f,
                a0f.y > 0.0f ? (2.0f * d0f.y) * a0f.y : 0.0f));
            const __nv_bfloat162 out1 = __float22bfloat162_rn(make_float2(
                a1f.x > 0.0f ? (2.0f * d1f.x) * a1f.x : 0.0f,
                a1f.y > 0.0f ? (2.0f * d1f.y) * a1f.y : 0.0f));

            store_chunk_value(sIn_ptr, row, col + 0, reinterpret_cast<const IType&>(out0.x));
            store_chunk_value(sIn_ptr, row, col + 1, reinterpret_cast<const IType&>(out0.y));
            store_chunk_value(sIn_ptr, row, col + 2, reinterpret_cast<const IType&>(out1.x));
            store_chunk_value(sIn_ptr, row, col + 3, reinterpret_cast<const IType&>(out1.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v = __bfloat162float(h1_raw[offset]);
                    out = __float2bfloat16_rn(v > 0.0f ? (2.0f * vd) * v : 0.0f);
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void transform_sqrelu_tile_inplace(
    IType* __restrict__ sIn_ptr,
    int tile
) {
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int VEC_ELEMS = 4;
    constexpr int TILE_VECS = (TILE_DIM * TILE_DIM) / VEC_ELEMS;

    for (int vec = threadIdx.x; vec < TILE_VECS; vec += THREADS) {
        const int elem = vec * VEC_ELEMS;
        const int row = elem / TILE_DIM;
        const int col = elem % TILE_DIM;
        const int2 packed = *reinterpret_cast<const int2*>(&sIn[tile][row][col]);

        const __nv_bfloat162 in0 = *reinterpret_cast<const __nv_bfloat162*>(&packed.x);
        const __nv_bfloat162 in1 = *reinterpret_cast<const __nv_bfloat162*>(&packed.y);
        const float2 f0 = __bfloat1622float2(in0);
        const float2 f1 = __bfloat1622float2(in1);

        const __nv_bfloat162 out0 = __float22bfloat162_rn(
            make_float2(f0.x > 0.0f ? f0.x * f0.x : 0.0f,
                        f0.y > 0.0f ? f0.y * f0.y : 0.0f));
        const __nv_bfloat162 out1 = __float22bfloat162_rn(
            make_float2(f1.x > 0.0f ? f1.x * f1.x : 0.0f,
                        f1.y > 0.0f ? f1.y * f1.y : 0.0f));

        int2 out;
        out.x = *reinterpret_cast<const int*>(&out0);
        out.y = *reinterpret_cast<const int*>(&out1);
        *reinterpret_cast<int2*>(&sIn[tile][row][col]) = out;
    }
    __syncthreads();
}

__device__ __forceinline__ void transform_sqrelu_deriv_tile_inplace(
    IType* __restrict__ sDh_ptr,
    const IType* __restrict__ sH1_ptr,
    int tile
) {
    auto& sDh = *reinterpret_cast<InputBuf3D*>(sDh_ptr);
    const auto& sH1 = *reinterpret_cast<const InputBuf3D*>(sH1_ptr);
    constexpr int VEC_ELEMS = 4;
    constexpr int TILE_VECS = (TILE_DIM * TILE_DIM) / VEC_ELEMS;

    for (int vec = threadIdx.x; vec < TILE_VECS; vec += THREADS) {
        const int elem = vec * VEC_ELEMS;
        const int row = elem / TILE_DIM;
        const int col = elem % TILE_DIM;
        const int2 dh = *reinterpret_cast<const int2*>(&sDh[tile][row][col]);
        const int2 h1 = *reinterpret_cast<const int2*>(&sH1[tile][row][col]);

        const __nv_bfloat162 dh0 = *reinterpret_cast<const __nv_bfloat162*>(&dh.x);
        const __nv_bfloat162 dh1 = *reinterpret_cast<const __nv_bfloat162*>(&dh.y);
        const __nv_bfloat162 h10 = *reinterpret_cast<const __nv_bfloat162*>(&h1.x);
        const __nv_bfloat162 h11 = *reinterpret_cast<const __nv_bfloat162*>(&h1.y);
        const float2 d0 = __bfloat1622float2(dh0);
        const float2 d1 = __bfloat1622float2(dh1);
        const float2 x0 = __bfloat1622float2(h10);
        const float2 x1 = __bfloat1622float2(h11);

        const __nv_bfloat162 out0 = __float22bfloat162_rn(
            make_float2(x0.x > 0.0f ? (2.0f * d0.x) * x0.x : 0.0f,
                        x0.y > 0.0f ? (2.0f * d0.y) * x0.y : 0.0f));
        const __nv_bfloat162 out1 = __float22bfloat162_rn(
            make_float2(x1.x > 0.0f ? (2.0f * d1.x) * x1.x : 0.0f,
                        x1.y > 0.0f ? (2.0f * d1.y) * x1.y : 0.0f));

        int2 out;
        out.x = *reinterpret_cast<const int*>(&out0);
        out.y = *reinterpret_cast<const int*>(&out1);
        *reinterpret_cast<int2*>(&sDh[tile][row][col]) = out;
    }
    __syncthreads();
}

__device__ __forceinline__ void load_silu_mul_chunk_strided(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ h13,
    int rows,
    int cols,
    int input_stride,
    int h3_offset,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * input_stride + global_col;
            const int2 a = *reinterpret_cast<const int2*>(h13 + base);
            const int2 b = *reinterpret_cast<const int2*>(h13 + base + h3_offset);

            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);

            const float2 a0f = __bfloat1622float2(a0);
            const float2 a1f = __bfloat1622float2(a1);
            const float2 b0f = __bfloat1622float2(b0);
            const float2 b1f = __bfloat1622float2(b1);

            const float sig0x = 1.0f / (1.0f + __expf(-a0f.x));
            const float sig0y = 1.0f / (1.0f + __expf(-a0f.y));
            const float sig1x = 1.0f / (1.0f + __expf(-a1f.x));
            const float sig1y = 1.0f / (1.0f + __expf(-a1f.y));

            const __nv_bfloat162 out0 = __float22bfloat162_rn(make_float2(
                (a0f.x * sig0x) * b0f.x,
                (a0f.y * sig0y) * b0f.y));
            const __nv_bfloat162 out1 = __float22bfloat162_rn(make_float2(
                (a1f.x * sig1x) * b1f.x,
                (a1f.y * sig1y) * b1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, reinterpret_cast<const IType&>(out0.x));
            store_chunk_value(sIn_ptr, row, col + 1, reinterpret_cast<const IType&>(out0.y));
            store_chunk_value(sIn_ptr, row, col + 2, reinterpret_cast<const IType&>(out1.x));
            store_chunk_value(sIn_ptr, row, col + 3, reinterpret_cast<const IType&>(out1.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * input_stride + block_offset_X + c;
                    const float v1 = __bfloat162float(h13[offset]);
                    const float v3 = __bfloat162float(h13[offset + h3_offset]);
                    const float sig = 1.0f / (1.0f + __expf(-v1));
                    out = __float2bfloat16_rn((v1 * sig) * v3);
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_grouped_rows_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ input,
    int num_batches,
    int live_rows_per_batch,
    int padded_rows_per_batch,
    int input_cols,
    int output_cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;
        const int batch = global_row / padded_rows_per_batch;
        const int row_in_batch = global_row - batch * padded_rows_per_batch;
        const bool live_row = batch < num_batches && row_in_batch < live_rows_per_batch;

        if (live_row && global_col + (VEC - 1) < input_cols && global_col + (VEC - 1) < output_cols) {
            const int64_t src_row = static_cast<int64_t>(batch) * live_rows_per_batch + row_in_batch;
            const int64_t base = src_row * input_cols + global_col;
            const int2 v = *reinterpret_cast<const int2*>(input + base);
            const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
            const __nv_bfloat162 v1 = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
            store_chunk_value(sIn_ptr, row, col + 0, reinterpret_cast<const IType&>(v0.x));
            store_chunk_value(sIn_ptr, row, col + 1, reinterpret_cast<const IType&>(v0.y));
            store_chunk_value(sIn_ptr, row, col + 2, reinterpret_cast<const IType&>(v1.x));
            store_chunk_value(sIn_ptr, row, col + 3, reinterpret_cast<const IType&>(v1.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out = __float2bfloat16_rn(0.0f);
                if (live_row && global_col + j < input_cols && global_col + j < output_cols) {
                    const int64_t src_row = static_cast<int64_t>(batch) * live_rows_per_batch + row_in_batch;
                    out = input[src_row * input_cols + global_col + j];
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_indexed_scaled_grouped_rows_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ input,
    const int64_t* __restrict__ token_indices,
    const float* __restrict__ scores,
    int num_batches,
    int live_rows_per_batch,
    int padded_rows_per_batch,
    int input_cols,
    int output_cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;
    const int batch_for_tile = block_offset_Y / padded_rows_per_batch;
    const int row_offset_in_batch = block_offset_Y - batch_for_tile * padded_rows_per_batch;
    const bool batch_live = batch_for_tile < num_batches;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_col = block_offset_X + col;
        const int row_in_batch = row_offset_in_batch + row;
        const bool live_row = batch_live && row_in_batch < live_rows_per_batch;
        const int64_t routed_row = static_cast<int64_t>(batch_for_tile) * live_rows_per_batch + row_in_batch;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int c = col + j;
            IType out = __float2bfloat16_rn(0.0f);
            if (live_row && global_col + j < input_cols && global_col + j < output_cols) {
                const int64_t src_row = token_indices[routed_row];
                const float score = scores[routed_row];
                const float v = __bfloat162float(input[src_row * input_cols + global_col + j]);
                out = __float2bfloat16_rn(v * score);
            }
            store_chunk_value(sIn_ptr, row, c, out);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_indexed_rmsnorm_grouped_rows_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ input,
    const IType* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    const int64_t* __restrict__ token_indices,
    int num_batches,
    int live_rows_per_batch,
    int padded_rows_per_batch,
    int input_cols,
    int output_cols,
    int block_offset_Y,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;
    const int batch_for_tile = block_offset_Y / padded_rows_per_batch;
    const int row_offset_in_batch = block_offset_Y - batch_for_tile * padded_rows_per_batch;
    const bool batch_live = batch_for_tile < num_batches;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_col = block_offset_X + col;
        const int row_in_batch = row_offset_in_batch + row;
        const bool live_row = batch_live && row_in_batch < live_rows_per_batch;
        const int64_t routed_row = static_cast<int64_t>(batch_for_tile) * live_rows_per_batch + row_in_batch;
        const int64_t src_row = live_row ? token_indices[routed_row] : 0;
        const float row_inv = live_row ? inv_rms[src_row] : 0.0f;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int c = col + j;
            IType out = __float2bfloat16_rn(0.0f);
            if (live_row && global_col + j < input_cols && global_col + j < output_cols) {
                const int64_t input_offset = src_row * input_cols + global_col + j;
                const float v = __bfloat162float(input[input_offset]);
                const float w = __bfloat162float(norm_weight[global_col + j]);
                out = __float2bfloat16_rn(v * row_inv * w);
            }
            store_chunk_value(sIn_ptr, row, c, out);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_indexed_scaled_variable_grouped_rows_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ input,
    const int64_t* __restrict__ token_indices,
    const float* __restrict__ scores,
    const int64_t* __restrict__ route_starts,
    const int64_t* __restrict__ rows,
    int expert,
    int tile_row_in_expert,
    int input_cols,
    int output_cols,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;
    const int live_rows = static_cast<int>(rows[expert]);
    const int64_t route_start = route_starts[expert];

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_col = block_offset_X + col;
        const int row_in_expert = tile_row_in_expert + row;
        const bool live_row = row_in_expert < live_rows;
        const int64_t routed_row = route_start + row_in_expert;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int c = col + j;
            IType out = __float2bfloat16_rn(0.0f);
            if (live_row && global_col + j < input_cols && global_col + j < output_cols) {
                const int64_t src_row = token_indices[routed_row];
                const float score = scores[routed_row];
                const float v = __bfloat162float(input[src_row * input_cols + global_col + j]);
                out = __float2bfloat16_rn(v * score);
            }
            store_chunk_value(sIn_ptr, row, c, out);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void load_indexed_rmsnorm_variable_grouped_rows_chunk_direct(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ input,
    const IType* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    const int64_t* __restrict__ token_indices,
    const int64_t* __restrict__ route_starts,
    const int64_t* __restrict__ rows,
    int expert,
    int tile_row_in_expert,
    int input_cols,
    int output_cols,
    int block_offset_X
) {
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;
    const int live_rows = static_cast<int>(rows[expert]);
    const int64_t route_start = route_starts[expert];

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_col = block_offset_X + col;
        const int row_in_expert = tile_row_in_expert + row;
        const bool live_row = row_in_expert < live_rows;
        const int64_t routed_row = route_start + row_in_expert;
        const int64_t src_row = live_row ? token_indices[routed_row] : 0;
        const float row_inv = live_row ? inv_rms[src_row] : 0.0f;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int c = col + j;
            IType out = __float2bfloat16_rn(0.0f);
            if (live_row && global_col + j < input_cols && global_col + j < output_cols) {
                const int64_t input_offset = src_row * input_cols + global_col + j;
                const float v = __bfloat162float(input[input_offset]);
                const float w = __bfloat162float(norm_weight[global_col + j]);
                out = __float2bfloat16_rn(v * row_inv * w);
            }
            store_chunk_value(sIn_ptr, row, c, out);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ DualLocalMax load_silu_deriv_chunk_direct(
    IType* __restrict__ sIn1_ptr,
    IType* __restrict__ sIn2_ptr,
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    DualLocalMax local{0.0f, 0.0f};
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
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

            const float sig0x = 1.0f / (1.0f + __expf(-b0f.x));
            const float sig0y = 1.0f / (1.0f + __expf(-b0f.y));
            const float sig1x = 1.0f / (1.0f + __expf(-b1f.x));
            const float sig1y = 1.0f / (1.0f + __expf(-b1f.y));

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

            const float2 o10f = __bfloat1622float2(o10);
            const float2 o11f = __bfloat1622float2(o11);
            const float2 o20f = __bfloat1622float2(o20);
            const float2 o21f = __bfloat1622float2(o21);
            local.a = fmaxf(local.a, fabsf(o10f.x));
            local.a = fmaxf(local.a, fabsf(o10f.y));
            local.a = fmaxf(local.a, fabsf(o11f.x));
            local.a = fmaxf(local.a, fabsf(o11f.y));
            local.b = fmaxf(local.b, fabsf(o20f.x));
            local.b = fmaxf(local.b, fabsf(o20f.y));
            local.b = fmaxf(local.b, fabsf(o21f.x));
            local.b = fmaxf(local.b, fabsf(o21f.y));

            store_chunk_value(sIn1_ptr, row, col + 0, reinterpret_cast<const IType&>(o10.x));
            store_chunk_value(sIn1_ptr, row, col + 1, reinterpret_cast<const IType&>(o10.y));
            store_chunk_value(sIn1_ptr, row, col + 2, reinterpret_cast<const IType&>(o11.x));
            store_chunk_value(sIn1_ptr, row, col + 3, reinterpret_cast<const IType&>(o11.y));
            store_chunk_value(sIn2_ptr, row, col + 0, reinterpret_cast<const IType&>(o20.x));
            store_chunk_value(sIn2_ptr, row, col + 1, reinterpret_cast<const IType&>(o20.y));
            store_chunk_value(sIn2_ptr, row, col + 2, reinterpret_cast<const IType&>(o21.x));
            store_chunk_value(sIn2_ptr, row, col + 3, reinterpret_cast<const IType&>(o21.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out1 = __float2bfloat16_rn(0.0f);
                IType out2 = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    const float sig = 1.0f / (1.0f + __expf(-v1));
                    const float silu_v1 = v1 * sig;
                    const float silup_v1 = sig * (1.0f + v1 - silu_v1);
                    out1 = __float2bfloat16_rn(vd * v3 * silup_v1);
                    out2 = __float2bfloat16_rn(vd * silu_v1);
                    local.a = fmaxf(local.a, fabsf(__bfloat162float(out1)));
                    local.b = fmaxf(local.b, fabsf(__bfloat162float(out2)));
                }
                store_chunk_value(sIn1_ptr, row, c, out1);
                store_chunk_value(sIn2_ptr, row, c, out2);
            }
        }
    }
    __syncthreads();
    return local;
}

template<bool USE_SAVED_SIGMOID = false>
__device__ __forceinline__ DualLocalMax load_silu_deriv_chunk_direct_row_bf16(
    IType* __restrict__ sIn1_ptr,
    IType* __restrict__ sIn2_ptr,
    IType* __restrict__ out1_bf16,
    IType* __restrict__ out2_bf16,
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ sig_h1,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    DualLocalMax local{0.0f, 0.0f};
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * cols + global_col;
            const int2 d = *reinterpret_cast<const int2*>(dh + base);
            const int2 a = *reinterpret_cast<const int2*>(h3 + base);
            const int2 b = *reinterpret_cast<const int2*>(h1_raw + base);
            int2 s;
            if constexpr (USE_SAVED_SIGMOID) {
                s = *reinterpret_cast<const int2*>(sig_h1 + base);
            }

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

            float sig0x, sig0y, sig1x, sig1y;
            if constexpr (USE_SAVED_SIGMOID) {
                const __nv_bfloat162 s0 = *reinterpret_cast<const __nv_bfloat162*>(&s.x);
                const __nv_bfloat162 s1 = *reinterpret_cast<const __nv_bfloat162*>(&s.y);
                const float2 s0f = __bfloat1622float2(s0);
                const float2 s1f = __bfloat1622float2(s1);
                sig0x = s0f.x;
                sig0y = s0f.y;
                sig1x = s1f.x;
                sig1y = s1f.y;
            } else {
                sig0x = 1.0f / (1.0f + __expf(-b0f.x));
                sig0y = 1.0f / (1.0f + __expf(-b0f.y));
                sig1x = 1.0f / (1.0f + __expf(-b1f.x));
                sig1y = 1.0f / (1.0f + __expf(-b1f.y));
            }

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

            const float2 o10f = __bfloat1622float2(o10);
            const float2 o11f = __bfloat1622float2(o11);
            const float2 o20f = __bfloat1622float2(o20);
            const float2 o21f = __bfloat1622float2(o21);
            local.a = fmaxf(local.a, fabsf(o10f.x));
            local.a = fmaxf(local.a, fabsf(o10f.y));
            local.a = fmaxf(local.a, fabsf(o11f.x));
            local.a = fmaxf(local.a, fabsf(o11f.y));
            local.b = fmaxf(local.b, fabsf(o20f.x));
            local.b = fmaxf(local.b, fabsf(o20f.y));
            local.b = fmaxf(local.b, fabsf(o21f.x));
            local.b = fmaxf(local.b, fabsf(o21f.y));

            store_chunk_value(sIn1_ptr, row, col + 0, reinterpret_cast<const IType&>(o10.x));
            store_chunk_value(sIn1_ptr, row, col + 1, reinterpret_cast<const IType&>(o10.y));
            store_chunk_value(sIn1_ptr, row, col + 2, reinterpret_cast<const IType&>(o11.x));
            store_chunk_value(sIn1_ptr, row, col + 3, reinterpret_cast<const IType&>(o11.y));
            store_chunk_value(sIn2_ptr, row, col + 0, reinterpret_cast<const IType&>(o20.x));
            store_chunk_value(sIn2_ptr, row, col + 1, reinterpret_cast<const IType&>(o20.y));
            store_chunk_value(sIn2_ptr, row, col + 2, reinterpret_cast<const IType&>(o21.x));
            store_chunk_value(sIn2_ptr, row, col + 3, reinterpret_cast<const IType&>(o21.y));

            *reinterpret_cast<__nv_bfloat162*>(out1_bf16 + base + 0) = o10;
            *reinterpret_cast<__nv_bfloat162*>(out1_bf16 + base + 2) = o11;
            *reinterpret_cast<__nv_bfloat162*>(out2_bf16 + base + 0) = o20;
            *reinterpret_cast<__nv_bfloat162*>(out2_bf16 + base + 2) = o21;
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out1 = __float2bfloat16_rn(0.0f);
                IType out2 = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    float sig;
                    if constexpr (USE_SAVED_SIGMOID) {
                        sig = __bfloat162float(sig_h1[offset]);
                    } else {
                        sig = 1.0f / (1.0f + __expf(-v1));
                    }
                    const float silu_v1 = v1 * sig;
                    const float silup_v1 = sig * (1.0f + v1 - silu_v1);
                    out1 = __float2bfloat16_rn(vd * v3 * silup_v1);
                    out2 = __float2bfloat16_rn(vd * silu_v1);
                    local.a = fmaxf(local.a, fabsf(__bfloat162float(out1)));
                    local.b = fmaxf(local.b, fabsf(__bfloat162float(out2)));
                    out1_bf16[offset] = out1;
                    out2_bf16[offset] = out2;
                }
                store_chunk_value(sIn1_ptr, row, c, out1);
                store_chunk_value(sIn2_ptr, row, c, out2);
            }
        }
    }
    __syncthreads();
    return local;
}

template<bool FIRST_OUTPUT>
__device__ __forceinline__ float load_silu_deriv_chunk_direct_single(
    IType* __restrict__ sIn_ptr,
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    float local = 0.0f;
    constexpr int VEC = 4;
    constexpr int TOTAL = CHUNK_DIM * CHUNK_DIM;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / CHUNK_DIM;
        const int col = idx % CHUNK_DIM;
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

            const float sig0x = 1.0f / (1.0f + __expf(-b0f.x));
            const float sig0y = 1.0f / (1.0f + __expf(-b0f.y));
            const float sig1x = 1.0f / (1.0f + __expf(-b1f.x));
            const float sig1y = 1.0f / (1.0f + __expf(-b1f.y));

            const float silu0x = b0f.x * sig0x;
            const float silu0y = b0f.y * sig0y;
            const float silu1x = b1f.x * sig1x;
            const float silu1y = b1f.y * sig1y;

            __nv_bfloat162 out0, out1;
            if constexpr (FIRST_OUTPUT) {
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
            local = fmaxf(local, fabsf(out0f.x));
            local = fmaxf(local, fabsf(out0f.y));
            local = fmaxf(local, fabsf(out1f.x));
            local = fmaxf(local, fabsf(out1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, reinterpret_cast<const IType&>(out0.x));
            store_chunk_value(sIn_ptr, row, col + 1, reinterpret_cast<const IType&>(out0.y));
            store_chunk_value(sIn_ptr, row, col + 2, reinterpret_cast<const IType&>(out1.x));
            store_chunk_value(sIn_ptr, row, col + 3, reinterpret_cast<const IType&>(out1.y));
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                IType out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = static_cast<int64_t>(global_row) * cols + block_offset_X + c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    const float sig = 1.0f / (1.0f + __expf(-v1));
                    const float silu_v1 = v1 * sig;
                    if constexpr (FIRST_OUTPUT) {
                        const float silup_v1 = sig * (1.0f + v1 - silu_v1);
                        out = __float2bfloat16_rn(vd * v3 * silup_v1);
                    } else {
                        out = __float2bfloat16_rn(vd * silu_v1);
                    }
                    local = fmaxf(local, fabsf(__bfloat162float(out)));
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
    return local;
}


// ═══════════════════════════════════════════════════════════════════
// Quantization mode: how to round E8M0 scale exponent
// ═══════════════════════════════════════════════════════════════════
enum class QuantMode : int {
    RTE = 0,     // Round-ties-to-even (default)
    ENCODE = 1,  // Encode-centric: ceil exponent → scale >= amax → no clipping
    DECODE = 2   // Decode-centric: floor exponent → scale < amax → fills range
};

// ═══════════════════════════════════════════════════════════════════
// E8M0 scale computation
// ═══════════════════════════════════════════════════════════════════
// Round-ties-to-even (original)
__device__ __forceinline__ uint8_t float_to_e8m0_rte(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    constexpr uint32_t half = 1u << 22;
    bool round_up = (mant > half) || (mant == half && (exp & 1));
    if (round_up && exp < 0xFE) ++exp;
    return exp;
}

// Encode-centric: ceil exponent → 2^exp >= val always
__device__ __forceinline__ uint8_t float_to_e8m0_ceil(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    // If there's any mantissa, round UP so scale >= val
    if (mant > 0 && exp < 0xFE) ++exp;
    return exp;
}

// Decode-centric: floor exponent → 2^exp <= val always
__device__ __forceinline__ uint8_t float_to_e8m0_floor(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    // Just truncate mantissa → always rounds down
    return exp;
}

// Dispatch helper
template<QuantMode MODE>
__device__ __forceinline__ uint8_t float_to_e8m0(float val) {
    if constexpr (MODE == QuantMode::ENCODE) return float_to_e8m0_ceil(val);
    else if constexpr (MODE == QuantMode::DECODE) return float_to_e8m0_floor(val);
    else return float_to_e8m0_rte(val);
}

using RNGState = transformer_engine::curanddx::detail::philox4x32_native_state<10>;

__device__ __forceinline__ uint32_t next_rbits(
    RNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if (rnd_idx == 4) {
        rnd_idx = 0;
        random_uint4 = rng.generate4();
    }
    const uint32_t* rbits_arr = reinterpret_cast<uint32_t*>(&random_uint4);
    return rbits_arr[rnd_idx++];
}

__device__ __forceinline__ uint32_t mxfp4_gf16_xtime(uint32_t value) {
    const uint32_t reduction_mask = 0u - (value >> 15);
    return ((value << 1) & 0xffffu) ^ (reduction_mask & 0x100bu);
}

__device__ __forceinline__ uint32_t mxfp4_pack_correlated_sr_pair(
    uint32_t ef,
    uint32_t ab
) {
    return ((ab & 0xff00u) << 16) |
           ((ef & 0xff00u) << 8) |
           ((ab & 0x00ffu) << 8) |
           (ef & 0x00ffu);
}

template <bool ROW_VECTOR>
__device__ __forceinline__ uint4 mxfp4_correlated_sr_rbits16(
    uint64_t base,
    int row,
    int col
) {
    const uint32_t tile_row = static_cast<uint32_t>(row) >> 4;
    const uint32_t tile_col = static_cast<uint32_t>(col) >> 4;
    uint32_t tile_rbits =
        static_cast<uint32_t>(base) ^ static_cast<uint32_t>(base >> 32);
    tile_rbits ^= tile_row * 0x9e3779b9u;
    tile_rbits ^= tile_col * 0x85ebca6bu;
    tile_rbits ^= tile_rbits >> 16;
    tile_rbits *= 0x7feb352du;
    tile_rbits ^= tile_rbits >> 15;
    tile_rbits *= 0x846ca68bu;
    tile_rbits ^= tile_rbits >> 16;

    // Sixty-four pairwise-independent GF(2^16) affine fields from one
    // uniformly mixed (a, b) pair. A row or column vector consumes eight
    // fields from the same logical 16x16 tile.
    const uint32_t field_a = tile_rbits & 0xffffu;
    const uint32_t field_b = tile_rbits >> 16;
    const uint32_t field_a2 = mxfp4_gf16_xtime(field_a);
    const uint32_t field_a4 = mxfp4_gf16_xtime(field_a2);
    const uint32_t field_a8 = mxfp4_gf16_xtime(field_a4);
    const uint32_t field_a16 = mxfp4_gf16_xtime(field_a8);
    const uint32_t field_a32 = mxfp4_gf16_xtime(field_a16);
    uint32_t field_product = 0;
    if constexpr (ROW_VECTOR) {
        const uint32_t row_group =
            (static_cast<uint32_t>(row) & 14u) >> 1;
        field_product ^= field_a8 & (0u - (row_group & 1u));
        field_product ^= field_a16 & (0u - ((row_group >> 1) & 1u));
        field_product ^= field_a32 & (0u - ((row_group >> 2) & 1u));
    } else {
        const uint32_t col_group =
            (static_cast<uint32_t>(col) & 14u) >> 1;
        field_product ^= field_a & (0u - (col_group & 1u));
        field_product ^= field_a2 & (0u - ((col_group >> 1) & 1u));
        field_product ^= field_a4 & (0u - ((col_group >> 2) & 1u));
    }

    const uint32_t field0 = field_b ^ field_product;
    const uint32_t field_step = ROW_VECTOR ? field_a : field_a8;
    const uint32_t field_step2 = ROW_VECTOR ? field_a2 : field_a16;
    const uint32_t field_step4 = ROW_VECTOR ? field_a4 : field_a32;
    const uint32_t field1 = field0 ^ field_step;
    const uint32_t field2 = field0 ^ field_step2;
    const uint32_t field3 = field0 ^ field_step ^ field_step2;
    const uint32_t field4 = field0 ^ field_step4;
    const uint32_t field5 = field4 ^ field_step;
    const uint32_t field6 = field4 ^ field_step2;
    const uint32_t field7 = field4 ^ field_step ^ field_step2;
    return make_uint4(
        mxfp4_pack_correlated_sr_pair(field0, field1),
        mxfp4_pack_correlated_sr_pair(field2, field3),
        mxfp4_pack_correlated_sr_pair(field4, field5),
        mxfp4_pack_correlated_sr_pair(field6, field7));
}

__device__ __forceinline__ uint64_t mxfp4_swap_bf16_pair_lanes(
    uint64_t packed
) {
    return ((packed & 0xffff0000ffff0000ull) >> 16) |
           ((packed & 0x0000ffff0000ffffull) << 16);
}

__device__ __forceinline__ uint32_t mxfp4_swap_fp4_pair_lanes(
    uint32_t packed
) {
    return ((packed & 0xf0f0f0f0u) >> 4) |
           ((packed & 0x0f0f0f0fu) << 4);
}

__device__ __forceinline__ uint8_t float_to_e8m0_stochastic(float val, uint32_t rbits) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    if (mant == 0 || exp >= 0xFE) {
        return exp;
    }
    const uint64_t threshold = static_cast<uint64_t>(mant) << 9;
    return static_cast<uint64_t>(rbits) < threshold ? static_cast<uint8_t>(exp + 1) : exp;
}

template<bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ uint32_t make_rht_sign_bits(
    RNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if constexpr (!WITH_RANDOM_SIGN_MASK) {
        return 0xffffffffu;
    }
    // This production branch uses the same deterministic sign diagonal as
    // the completed MXFP4+RHT trajectory.  The historical template name is
    // retained for ABI compatibility: `true` means "apply the signed RHT",
    // while the literal below makes the transform replayable and consumes no
    // stochastic-rounding coordinate.
    return 0x00002817u;
}

__device__ __forceinline__ void fwht16_unnormalized(float (&vals)[ELTS_PER_THREAD]) {
    #pragma unroll
    for (int step = 1; step < ELTS_PER_THREAD; step <<= 1) {
        #pragma unroll
        for (int base = 0; base < ELTS_PER_THREAD; base += 2 * step) {
            #pragma unroll
            for (int j = 0; j < step; ++j) {
                const float a = vals[base + j];
                const float b = vals[base + j + step];
                vals[base + j] = a + b;
                vals[base + j + step] = a - b;
            }
        }
    }
}

template<int RHT_BLOCK_SIZE, bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ void apply_block_rht_registers(
    float (&vals)[ELTS_PER_THREAD],
    const int block_thread_rank,
    const uint32_t sign_bits
) {
    static_assert(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32,
                  "RHT block size must be 16 or 32");

    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; ++i) {
        if constexpr (WITH_RANDOM_SIGN_MASK) {
            vals[i] *= ((sign_bits >> i) & 1u) ? 1.0f : -1.0f;
        }
    }

    fwht16_unnormalized(vals);

    if constexpr (RHT_BLOCK_SIZE == 32) {
        #pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; ++i) {
            const float peer = __shfl_xor_sync(0xffffffff, vals[i], 1);
            vals[i] = (block_thread_rank == 0) ? (vals[i] + peer) : (peer - vals[i]);
        }
        constexpr float kNorm32 = 0.1767766952966369f;  // 1 / sqrt(32)
        #pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; ++i) {
            vals[i] *= kNorm32;
        }
    } else {
        constexpr float kNorm16 = 0.25f;  // 1 / sqrt(16)
        #pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; ++i) {
            vals[i] *= kNorm16;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Per-row quantize for one sub-tile: 16 bf16 → 8 fp4x2 + scale update
//
// Each thread handles 16 elements in a row (= half of MX_BLOCK=32).
// Scales are keyed by MX block index, so threads in the same block
// cooperate via __shfl to share the block amax.
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void mx_rowwise_quantize(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,     // [CHUNK_DIM × SCALES_PER_CHUNK]
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out
) {
    const auto& sIn = *reinterpret_cast<const InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    const int thread_lane = threadIdx.x % 32;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    // Scale indices: each thread covers 16 elements, MX block = 32
    // So 2 threads per MX block in X direction
    const int mx_block_in_tile = thread_offset_X / MX_BLOCK;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + mx_block_in_tile;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const int global_row = stage_Y * TILE_DIM + row;

        // Load 16 bf16 from SMEM using vectorized loads (2 waves × 8)
        __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e)
                abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
        }

        // Compute per-16-element amax, then reduce within MX block (32 elements)
        float my_amax = fmaxf(
            __bfloat162float(__habs(amax_2x.x)),
            __bfloat162float(__habs(amax_2x.y))
        );

        // Reduce with the other half of the MX block (the other thread covers the other 16 elements)
        // tid_X % 2 tells us which half we are; pair with XOR mask
        float pair_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
        float block_amax = fmaxf(my_amax, pair_amax);

        // Compute E8M0 scale using selected rounding mode
        uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);

        // Store scale (only one thread per MX block)
        if ((tid_X % 2) == 0) {
            scale_buf[global_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_val;
        }

        // Compute quantization coefficient: 6.0 / 2^(e8m0 - 127)
        float scale_rcp = exp2f_rcp(e8m0_val);
        float coeff = 6.0f * scale_rcp;

        // Quantize and pack to FP4
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true, bool CORRELATED_DATA_SR = false>
__device__ __forceinline__ void mx_rowwise_quantize_opt(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base,
    const int logical_row_offset = 0,
    const int logical_col_offset = 0,
    const uint64_t correlated_sr_base = 0
) {
    const auto& sIn = *reinterpret_cast<const InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    const int thread_lane = threadIdx.x % 32;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int mx_block_in_tile = thread_offset_X / MX_BLOCK;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + mx_block_in_tile;

    RNGState rng;
    const uint64_t tile_linear_idx = static_cast<uint64_t>(stage_Y * TILES_X + stage_X);
    if constexpr (
        (DATA_SR && !CORRELATED_DATA_SR) || SCALE_SR ||
        (WITH_RHT && WITH_RANDOM_SIGN_MASK)) {
        rng.init(
            rng_seed,
            rng_subsequence_base + tile_linear_idx * THREADS + threadIdx.x,
            0);
    }
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const int global_row = stage_Y * TILE_DIM + row;

        __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        uint4 correlated_rbits = make_uint4(0, 0, 0, 0);
        if constexpr (DATA_SR && CORRELATED_DATA_SR) {
            const int logical_row =
                logical_row_offset + stage_Y * TILE_DIM + row;
            const int logical_col =
                logical_col_offset + stage_X * TILE_DIM + thread_offset_X;
            correlated_rbits = mxfp4_correlated_sr_rbits16<true>(
                correlated_sr_base, logical_row, logical_col);
        }

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            if constexpr (!WITH_RHT) {
                #pragma unroll
                for (int e = 0; e < PACK_SIZE / 2; ++e) {
                    abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
                }
            }
        }

        float my_amax;
        if constexpr (WITH_RHT) {
            float vals[ELTS_PER_THREAD];
            #pragma unroll
            for (int w = 0; w < WAVES; ++w) {
                const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
                #pragma unroll
                for (int e = 0; e < PACK_SIZE / 2; ++e) {
                    const float2 packed = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&rIn[w][e]));
                    vals[sw + 2 * e + 0] = packed.x;
                    vals[sw + 2 * e + 1] = packed.y;
                }
            }

            const int block_thread_rank = (thread_offset_X % RHT_BLOCK_SIZE) / ELTS_PER_THREAD;
            const uint32_t sign_bits =
                make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
            apply_block_rht_registers<RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
                vals,
                block_thread_rank,
                sign_bits
            );

            my_amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELTS_PER_THREAD; ++i) {
                my_amax = fmaxf(my_amax, fabsf(vals[i]));
            }

            #pragma unroll
            for (int w = 0; w < WAVES; ++w) {
                const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
                #pragma unroll
                for (int e = 0; e < PACK_SIZE / 2; ++e) {
                    rIn[w][e] = IType2{
                        __float2bfloat16_rn(vals[sw + 2 * e + 0]),
                        __float2bfloat16_rn(vals[sw + 2 * e + 1]),
                    };
                }
            }
        } else {
            my_amax = fmaxf(
                __bfloat162float(__habs(amax_2x.x)),
                __bfloat162float(__habs(amax_2x.y))
            );
        }

        float pair_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
        float block_amax = fmaxf(my_amax, pair_amax);

        uint8_t e8m0_val;
        if constexpr (SCALE_SR) {
            e8m0_val = float_to_e8m0_stochastic(block_amax, next_rbits(rng, random_uint4, rnd_idx));
        } else {
            e8m0_val = float_to_e8m0<MODE>(block_amax);
        }

        if ((tid_X % 2) == 0) {
            scale_buf[global_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_val;
        }

        float scale_rcp = exp2f_rcp(e8m0_val);
        float coeff = 6.0f * scale_rcp;

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out;
            if constexpr (DATA_SR) {
                uint32_t rbits03;
                uint32_t rbits47;
                if constexpr (CORRELATED_DATA_SR) {
                    const int logical_row =
                        logical_row_offset + stage_Y * TILE_DIM + row;
                    if ((logical_row & 1) != 0) {
                        e03 = mxfp4_swap_bf16_pair_lanes(e03);
                        e47 = mxfp4_swap_bf16_pair_lanes(e47);
                    }
                    const bool high_half = (sw & 8) != 0;
                    rbits03 = high_half ? correlated_rbits.z : correlated_rbits.x;
                    rbits47 = high_half ? correlated_rbits.w : correlated_rbits.y;
                } else {
                    rbits03 = next_rbits(rng, random_uint4, rnd_idx);
                    rbits47 = next_rbits(rng, random_uint4, rnd_idx);
                }
                const bf16 coeff_bf16 = __float2bfloat16(coeff);
                out = mul_cvt_bf16_to_fp4_8x_stochastic_rounding<bf16>(
                    e03, e47, coeff_bf16, rbits03, rbits47);
                if constexpr (CORRELATED_DATA_SR) {
                    const int logical_row =
                        logical_row_offset + stage_Y * TILE_DIM + row;
                    if ((logical_row & 1) != 0) {
                        out = mxfp4_swap_fp4_pair_lanes(out);
                    }
                }
            } else {
                const bf16 coeff_bf16 = __float2bfloat16(coeff);
                out = mul_cvt_bf16_to_fp4_8x_round_to_nearest<bf16>(e03, e47, coeff_bf16);
            }
            st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

// One warp owns one 32x32 weight tile. A single E8M0 scale and one set of
// E2M1 codes are emitted in both orientations, so forward and dgrad consume
// exactly the same effective weight matrix.
template <bool EMIT_COL = true>
__device__ __forceinline__ void mx_weight_2d_quantize(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ row_scale_buf,
    uint8_t* __restrict__ col_scale_buf,
    const int stage_Y,
    const int stage_X,
    const int buff_in,
    const int row_buff_out,
    const int col_buff_out
) {
    const auto& sIn = *reinterpret_cast<const InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    __shared__ float warp_tile_amax[4][THREADS / 32];
    __shared__ uint8_t tile_scales[4];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int bank_group = lane / THREADS_PER_BANK;
    const int tid_y = threadIdx.x / THREADS_X;
    const int tid_x = threadIdx.x % THREADS_X;
    const int thread_offset_x = tid_x * ELTS_PER_THREAD;

    // Compute each 32x32 tile amax with the same bank-swizzled row mapping as
    // the tuned row quantizer. The previous one-warp-per-tile scan accessed
    // rows at a 128-byte stride and serialized on shared-memory banks.
    __align__(16) IType2 values[ITERATIONS][WAVES][PACK_SIZE / 2];
    float local_amax[ITERATIONS];
    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_y + it * THREADS_Y;
        IType2 amax_2x = {
            __float2bfloat16(0.0f),
            __float2bfloat16(0.0f),
        };
        #pragma unroll
        for (int wave = 0; wave < WAVES; ++wave) {
            const int sw =
                ((wave + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            const __uint128_t packed = ptx::ld_shared_b128(
                &sIn[buff_in][row][thread_offset_x + sw]);
            *reinterpret_cast<__uint128_t*>(&values[it][wave]) = packed;
            #pragma unroll
            for (int i = 0; i < PACK_SIZE / 2; ++i) {
                abs_max_2x(amax_2x, amax_2x, values[it][wave][i]);
            }
        }
        local_amax[it] = fmaxf(
            __bfloat162float(__habs(amax_2x.x)),
            __bfloat162float(__habs(amax_2x.y)));
    }

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        float value = fmaxf(
            local_amax[it],
            __shfl_xor_sync(0xffffffff, local_amax[it], 1));
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 4));
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 8));
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 16));
        if (lane == 0 || lane == 2) {
            const int tile_x = lane / 2;
            warp_tile_amax[it * 2 + tile_x][warp] = value;
        }
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        const int tile = threadIdx.x;
        float tile_amax = 0.0f;
        #pragma unroll
        for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp) {
            tile_amax = fmaxf(tile_amax, warp_tile_amax[tile][source_warp]);
        }
        const uint8_t safe_e8m0 = float_to_e8m0_ceil(tile_amax);
        // Whether 4-over-6 wins is almost entirely determined by the amax
        // phase inside its E8M0 bin. This scalar rule is within 0.16 point of
        // exhaustive selection on all three trained FFN weights.
        const float amax_phase = tile_amax * exp2f_rcp(safe_e8m0);
        const bool use_dense = safe_e8m0 > 0 && amax_phase < 0.87f;
        tile_scales[tile] = use_dense ? safe_e8m0 - 1 : safe_e8m0;
    }
    __syncthreads();

    // Quantize the row payload using the tuned 128-thread mapping. All rows
    // in a 32x32 tile read the same precomputed scale.
    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_y + it * THREADS_Y;
        const int row_tile_y = row / MX_BLOCK;
        const int row_tile_x = thread_offset_x / MX_BLOCK;
        const uint8_t row_e8m0 = tile_scales[row_tile_y * 2 + row_tile_x];
        const float coeff = 6.0f * exp2f_rcp(row_e8m0);

        if ((tid_x & 1) == 0) {
            const int row_scale_row = stage_Y * TILE_DIM + row;
            const int row_scale_col =
                stage_X * (TILE_DIM / MX_BLOCK) + row_tile_x;
            row_scale_buf[row_scale_row * SCALES_PER_CHUNK + row_scale_col] =
                row_e8m0;
        }

        #pragma unroll
        for (int wave = 0; wave < WAVES; ++wave) {
            const int sw =
                ((wave + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            const uint64_t e03 =
                *reinterpret_cast<const uint64_t*>(&values[it][wave][0]);
            const uint64_t e47 =
                *reinterpret_cast<const uint64_t*>(&values[it][wave][2]);
            const uint32_t out =
                mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            ptx::st_shared_b32(
                &sOut[row_buff_out][row][(thread_offset_x + sw) / 2], out);
        }
    }

    if constexpr (EMIT_COL) {
        __syncthreads();

        // Transpose the packed payload with the tuned columnwise thread
        // mapping.  Mixed MX/localCTA weights consume only the MX row, so
        // their specialization compiles this entire transpose and the
        // otherwise-dead column-scale writes away.
        const int tid_x_colwise = (lane % 16) + (warp / 2) * 16;
        const int tid_y_colwise = (warp % 2) * 2 + (lane / 16);
        const int source_row = tid_y_colwise * ELTS_PER_THREAD;
        const int source_byte_col = tid_x_colwise;
        uint64_t low_nibbles = 0;
        uint64_t high_nibbles = 0;
        #pragma unroll
        for (int pair = 0; pair < ELTS_PER_THREAD / 2; ++pair) {
            const uint8_t first = reinterpret_cast<const uint8_t*>(
                &sOut[row_buff_out][source_row + pair * 2][source_byte_col])[0];
            const uint8_t second = reinterpret_cast<const uint8_t*>(
                &sOut[row_buff_out][source_row + pair * 2 + 1][source_byte_col])[0];
            const uint8_t low = (first & 0x0f) | ((second & 0x0f) << 4);
            const uint8_t high = (first >> 4) | (second & 0xf0);
            low_nibbles |= static_cast<uint64_t>(low) << (pair * 8);
            high_nibbles |= static_cast<uint64_t>(high) << (pair * 8);
        }
        const int output_row = tid_x_colwise * 2;
        const int output_byte_col = tid_y_colwise * (ELTS_PER_THREAD / 2);
        ptx::st_shared_b64(
            &sOut[col_buff_out][output_row][output_byte_col], low_nibbles);
        ptx::st_shared_b64(
            &sOut[col_buff_out][output_row + 1][output_byte_col], high_nibbles);

        if ((tid_y_colwise & 1) == 0) {
            const int col_tile_y = tid_y_colwise / 2;
            const int col_tile_x = tid_x_colwise / 16;
            const uint8_t col_e8m0 = tile_scales[col_tile_y * 2 + col_tile_x];
            const int col_scale_row = stage_X * TILE_DIM + output_row;
            const int col_scale_col =
                stage_Y * (TILE_DIM / MX_BLOCK) + col_tile_y;
            col_scale_buf[
                (col_scale_row + 0) * SCALES_PER_CHUNK + col_scale_col] =
                col_e8m0;
            col_scale_buf[
                (col_scale_row + 1) * SCALES_PER_CHUNK + col_scale_col] =
                col_e8m0;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Pipelined quantize-and-store for one chunk (4 sub-tiles)
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_pipelined(
    IType* sIn_ptr, fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y, int block_offset_X,
    uint64_t* in_mbar, int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        // Prefetch tile t+2 if it exists
        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X, ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        // Wait for current tile
        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        // Quantize
        mx_rowwise_quantize<MODE>(sIn_ptr, sOut_ptr, scale_buf,
                           stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store quantized tile
        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    // Wait for all FP4 TMA stores
    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void quantize_chunk_pipelined_opt(
    IType* sIn_ptr, fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y, int block_offset_X,
    uint64_t* in_mbar, int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X, ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr,
            sOut_ptr,
            scale_buf,
            stage_Y,
            stage_X,
            t,
            buff_out,
            rng_seed,
            rng_subsequence_base
        );

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE>
__device__ __forceinline__ void mx_colwise_quantize_direct(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,
    const int stage_Y,
    const int stage_X,
    const int buff_in,
    const int buff_out
);

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true, bool CORRELATED_DATA_SR = false>
__device__ __forceinline__ void mx_colwise_quantize_direct_opt(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,
    const int stage_Y,
    const int stage_X,
    const int buff_in,
    const int buff_out,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base,
    const int logical_row_offset = 0,
    const int logical_col_offset = 0,
    const uint64_t correlated_sr_base = 0
);

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_pipelined_split_input(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int output_block_offset_Y,
    int output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + ntx * TILE_DIM,
                    input_block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_rowwise_quantize<MODE>(sIn_ptr, sOut_ptr, scale_buf,
                           stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                output_block_offset_X + stage_offset_X,
                output_block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void quantize_chunk_pipelined_split_input_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int output_block_offset_Y,
    int output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    const uint64_t cta_y = static_cast<uint64_t>(output_block_offset_Y / CHUNK_DIM);
    const uint64_t cta_x = static_cast<uint64_t>(output_block_offset_X / CHUNK_DIM);
    const uint64_t cta_rng_subsequence_base =
        rng_subsequence_base + (cta_y * static_cast<uint64_t>(gridDim.x) + cta_x) * NUM_TILES * THREADS;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + ntx * TILE_DIM,
                    input_block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, scale_buf, stage_Y, stage_X, t, buff_out,
            rng_seed, cta_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                output_block_offset_X + stage_offset_X,
                output_block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void mx_colwise_quantize_direct_tile(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,
    const int stage_Y,
    const int stage_X,
    const int buff_out
) {
    const auto& sIn2x = *reinterpret_cast<const TileInputBuf2x2D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int tid_X_colwise = (lane % 16) + (warp / 2) * 16;
    const int tid_Y_colwise = (warp % 2) * 2 + (lane / 16);

    const int thread_offset_Y = tid_Y_colwise * ELTS_PER_THREAD;
    const int in_thread_offset_X = tid_X_colwise;

    const int out_thread_offset_Y = tid_X_colwise * 2;
    const int out_thread_offset_X = tid_Y_colwise * (ELTS_PER_THREAD / 2);

    const int scale_block_in_tile = tid_Y_colwise / 2;
    const int global_scale_y = stage_Y * TILE_DIM + out_thread_offset_Y;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + scale_block_in_tile;
    const bool scale_storing_thread = ((tid_Y_colwise & 1) == 0);

    __align__(8) IType rIn[2][ELTS_PER_THREAD];
    IType2 thread_amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; ++i) {
        const IType2 elt_pair =
            ptx::ld_shared_b32(&sIn2x[thread_offset_Y + i][in_thread_offset_X]);
        rIn[0][i] = elt_pair.x;
        rIn[1][i] = elt_pair.y;
        ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, elt_pair);
    }

    const float thread_amax0 = __bfloat162float(__habs(thread_amax_2x.x));
    const float thread_amax1 = __bfloat162float(__habs(thread_amax_2x.y));
    const float pair_amax0 = __shfl_xor_sync(0xffffffff, thread_amax0, 16);
    const float pair_amax1 = __shfl_xor_sync(0xffffffff, thread_amax1, 16);
    const float block_amax0 = fmaxf(thread_amax0, pair_amax0);
    const float block_amax1 = fmaxf(thread_amax1, pair_amax1);

    const uint8_t e8m0_0 = float_to_e8m0<MODE>(block_amax0);
    const uint8_t e8m0_1 = float_to_e8m0<MODE>(block_amax1);

    if (scale_storing_thread) {
        scale_buf[(global_scale_y + 0) * SCALES_PER_CHUNK + global_scale_x] = e8m0_0;
        scale_buf[(global_scale_y + 1) * SCALES_PER_CHUNK + global_scale_x] = e8m0_1;
    }

    const float coeff0 = 6.0f * exp2f_rcp(e8m0_0);
    const float coeff1 = 6.0f * exp2f_rcp(e8m0_1);

    #pragma unroll
    for (int row_pair = 0; row_pair < 2; ++row_pair) {
        const float coeff = (row_pair == 0) ? coeff0 : coeff1;
        const uint64_t elts03_lo = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][0]);
        const uint64_t elts47_lo = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][4]);
        const uint64_t elts03_hi = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][8]);
        const uint64_t elts47_hi = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][12]);

        const uint32_t out_lo =
            mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(elts03_lo, elts47_lo, coeff);
        const uint32_t out_hi =
            mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(elts03_hi, elts47_hi, coeff);

        const uint64_t packed = static_cast<uint64_t>(out_lo) |
                                (static_cast<uint64_t>(out_hi) << 32);
        ptx::st_shared_b64(
            &sOut[buff_out][out_thread_offset_Y + row_pair][out_thread_offset_X], packed);
    }
}

template<QuantMode MODE = QuantMode::RTE, bool STORE_BF16 = false>
__device__ __forceinline__ void load_silu_deriv_tile_direct_and_rowwise_quant(
    IType* __restrict__ sIn0_ptr,
    IType* __restrict__ sIn1_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ row_scale_buf0,
    uint8_t* __restrict__ row_scale_buf1,
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    int cols,
    int block_offset_Y,
    int block_offset_X,
    int stage_Y,
    int stage_X,
    IType* __restrict__ out0_bf16 = nullptr,
    IType* __restrict__ out1_bf16 = nullptr
) {
    auto& sIn0 = *reinterpret_cast<TileInputBuf2D*>(sIn0_ptr);
    auto& sIn1 = *reinterpret_cast<TileInputBuf2D*>(sIn1_ptr);
    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);

    const int thread_lane = threadIdx.x % 32;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int mx_block_in_tile = thread_offset_X / MX_BLOCK;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + mx_block_in_tile;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const int global_row = block_offset_Y + row;
        const int scale_row = stage_Y * TILE_DIM + row;

        __align__(16) IType2 rOut0[WAVES][PACK_SIZE / 2];
        __align__(16) IType2 rOut1[WAVES][PACK_SIZE / 2];
        IType2 amax0_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        IType2 amax1_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            const int64_t base =
                static_cast<int64_t>(global_row) * cols + block_offset_X + thread_offset_X + sw;

            const int4 d_vec = *reinterpret_cast<const int4*>(dh + base);
            const int4 a_vec = *reinterpret_cast<const int4*>(h3 + base);
            const int4 b_vec = *reinterpret_cast<const int4*>(h1_raw + base);

            const auto* d_pairs = reinterpret_cast<const __nv_bfloat162*>(&d_vec);
            const auto* a_pairs = reinterpret_cast<const __nv_bfloat162*>(&a_vec);
            const auto* b_pairs = reinterpret_cast<const __nv_bfloat162*>(&b_vec);

            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                const float2 d = __bfloat1622float2(d_pairs[e]);
                const float2 a = __bfloat1622float2(a_pairs[e]);
                const float2 b = __bfloat1622float2(b_pairs[e]);

                const float sig_x = 1.0f / (1.0f + __expf(-b.x));
                const float sig_y = 1.0f / (1.0f + __expf(-b.y));
                const float silu_x = b.x * sig_x;
                const float silu_y = b.y * sig_y;
                const float silup_x = sig_x * (1.0f + b.x - silu_x);
                const float silup_y = sig_y * (1.0f + b.y - silu_y);

                rOut0[w][e] = {
                    __float2bfloat16_rn(d.x * a.x * silup_x),
                    __float2bfloat16_rn(d.y * a.y * silup_y),
                };
                rOut1[w][e] = {
                    __float2bfloat16_rn(d.x * silu_x),
                    __float2bfloat16_rn(d.y * silu_y),
                };
                abs_max_2x(amax0_2x, amax0_2x, rOut0[w][e]);
                abs_max_2x(amax1_2x, amax1_2x, rOut1[w][e]);
            }

            *reinterpret_cast<int4*>(&sIn0[row][thread_offset_X + sw]) =
                *reinterpret_cast<const int4*>(&rOut0[w][0]);
            *reinterpret_cast<int4*>(&sIn1[row][thread_offset_X + sw]) =
                *reinterpret_cast<const int4*>(&rOut1[w][0]);
            if constexpr (STORE_BF16) {
                *reinterpret_cast<int4*>(out0_bf16 + base) =
                    *reinterpret_cast<const int4*>(&rOut0[w][0]);
                *reinterpret_cast<int4*>(out1_bf16 + base) =
                    *reinterpret_cast<const int4*>(&rOut1[w][0]);
            }
        }

        const float my_amax0 = fmaxf(
            __bfloat162float(__habs(amax0_2x.x)),
            __bfloat162float(__habs(amax0_2x.y)));
        const float my_amax1 = fmaxf(
            __bfloat162float(__habs(amax1_2x.x)),
            __bfloat162float(__habs(amax1_2x.y)));

        const float pair_amax0 = __shfl_xor_sync(0xffffffff, my_amax0, 1);
        const float pair_amax1 = __shfl_xor_sync(0xffffffff, my_amax1, 1);
        const float block_amax0 = fmaxf(my_amax0, pair_amax0);
        const float block_amax1 = fmaxf(my_amax1, pair_amax1);

        const uint8_t e8m0_0 = float_to_e8m0<MODE>(block_amax0);
        const uint8_t e8m0_1 = float_to_e8m0<MODE>(block_amax1);

        if ((tid_X % 2) == 0) {
            row_scale_buf0[scale_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_0;
            row_scale_buf1[scale_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_1;
        }

        const float coeff0 = 6.0f * exp2f_rcp(e8m0_0);
        const float coeff1 = 6.0f * exp2f_rcp(e8m0_1);

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03_0 = *reinterpret_cast<const uint64_t*>(&rOut0[w][0]);
            const uint64_t e47_0 = *reinterpret_cast<const uint64_t*>(&rOut0[w][2]);
            const uint64_t e03_1 = *reinterpret_cast<const uint64_t*>(&rOut1[w][0]);
            const uint64_t e47_1 = *reinterpret_cast<const uint64_t*>(&rOut1[w][2]);
            const uint32_t out0 =
                mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff0);
            const uint32_t out1 =
                mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff1);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[0][row][(sw + thread_offset_X) / 2], out0);
            ptx::st_shared_b32(&sOut[1][row][(sw + thread_offset_X) / 2], out1);
        }
    }

    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE, bool STORE_BF16 = false>
__device__ __forceinline__ void load_silu_deriv_from_sigmoid_tile_direct_and_rowwise_quant(
    IType* __restrict__ sIn0_ptr,
    IType* __restrict__ sIn1_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ row_scale_buf0,
    uint8_t* __restrict__ row_scale_buf1,
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ sig_h1,
    int cols,
    int block_offset_Y,
    int block_offset_X,
    int stage_Y,
    int stage_X,
    IType* __restrict__ out0_bf16 = nullptr,
    IType* __restrict__ out1_bf16 = nullptr
) {
    auto& sIn0 = *reinterpret_cast<TileInputBuf2D*>(sIn0_ptr);
    auto& sIn1 = *reinterpret_cast<TileInputBuf2D*>(sIn1_ptr);
    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);

    const int thread_lane = threadIdx.x % 32;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int mx_block_in_tile = thread_offset_X / MX_BLOCK;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + mx_block_in_tile;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const int global_row = block_offset_Y + row;
        const int scale_row = stage_Y * TILE_DIM + row;

        __align__(16) IType2 rOut0[WAVES][PACK_SIZE / 2];
        __align__(16) IType2 rOut1[WAVES][PACK_SIZE / 2];
        IType2 amax0_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        IType2 amax1_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            const int64_t base =
                static_cast<int64_t>(global_row) * cols + block_offset_X + thread_offset_X + sw;

            const int4 d_vec = *reinterpret_cast<const int4*>(dh + base);
            const int4 a_vec = *reinterpret_cast<const int4*>(h3 + base);
            const int4 b_vec = *reinterpret_cast<const int4*>(h1_raw + base);
            const int4 s_vec = *reinterpret_cast<const int4*>(sig_h1 + base);

            const auto* d_pairs = reinterpret_cast<const __nv_bfloat162*>(&d_vec);
            const auto* a_pairs = reinterpret_cast<const __nv_bfloat162*>(&a_vec);
            const auto* b_pairs = reinterpret_cast<const __nv_bfloat162*>(&b_vec);
            const auto* s_pairs = reinterpret_cast<const __nv_bfloat162*>(&s_vec);

            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                const float2 d = __bfloat1622float2(d_pairs[e]);
                const float2 a = __bfloat1622float2(a_pairs[e]);
                const float2 b = __bfloat1622float2(b_pairs[e]);
                const float2 sig = __bfloat1622float2(s_pairs[e]);

                const float silu_x = b.x * sig.x;
                const float silu_y = b.y * sig.y;
                const float silup_x = sig.x * (1.0f + b.x - silu_x);
                const float silup_y = sig.y * (1.0f + b.y - silu_y);

                rOut0[w][e] = {
                    __float2bfloat16_rn(d.x * a.x * silup_x),
                    __float2bfloat16_rn(d.y * a.y * silup_y),
                };
                rOut1[w][e] = {
                    __float2bfloat16_rn(d.x * silu_x),
                    __float2bfloat16_rn(d.y * silu_y),
                };
                abs_max_2x(amax0_2x, amax0_2x, rOut0[w][e]);
                abs_max_2x(amax1_2x, amax1_2x, rOut1[w][e]);
            }

            *reinterpret_cast<int4*>(&sIn0[row][thread_offset_X + sw]) =
                *reinterpret_cast<const int4*>(&rOut0[w][0]);
            *reinterpret_cast<int4*>(&sIn1[row][thread_offset_X + sw]) =
                *reinterpret_cast<const int4*>(&rOut1[w][0]);
            if constexpr (STORE_BF16) {
                *reinterpret_cast<int4*>(out0_bf16 + base) =
                    *reinterpret_cast<const int4*>(&rOut0[w][0]);
                *reinterpret_cast<int4*>(out1_bf16 + base) =
                    *reinterpret_cast<const int4*>(&rOut1[w][0]);
            }
        }

        const float my_amax0 = fmaxf(
            __bfloat162float(__habs(amax0_2x.x)),
            __bfloat162float(__habs(amax0_2x.y)));
        const float my_amax1 = fmaxf(
            __bfloat162float(__habs(amax1_2x.x)),
            __bfloat162float(__habs(amax1_2x.y)));

        const float pair_amax0 = __shfl_xor_sync(0xffffffff, my_amax0, 1);
        const float pair_amax1 = __shfl_xor_sync(0xffffffff, my_amax1, 1);
        const float block_amax0 = fmaxf(my_amax0, pair_amax0);
        const float block_amax1 = fmaxf(my_amax1, pair_amax1);

        const uint8_t e8m0_0 = float_to_e8m0<MODE>(block_amax0);
        const uint8_t e8m0_1 = float_to_e8m0<MODE>(block_amax1);

        if ((tid_X % 2) == 0) {
            row_scale_buf0[scale_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_0;
            row_scale_buf1[scale_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_1;
        }

        const float coeff0 = 6.0f * exp2f_rcp(e8m0_0);
        const float coeff1 = 6.0f * exp2f_rcp(e8m0_1);

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03_0 = *reinterpret_cast<const uint64_t*>(&rOut0[w][0]);
            const uint64_t e47_0 = *reinterpret_cast<const uint64_t*>(&rOut0[w][2]);
            const uint64_t e03_1 = *reinterpret_cast<const uint64_t*>(&rOut1[w][0]);
            const uint64_t e47_1 = *reinterpret_cast<const uint64_t*>(&rOut1[w][2]);
            const uint32_t out0 =
                mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff0);
            const uint32_t out1 =
                mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff1);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[0][row][(sw + thread_offset_X) / 2], out0);
            ptx::st_shared_b32(&sOut[1][row][(sw + thread_offset_X) / 2], out1);
        }
    }

    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void load_silu_deriv_tile_strided_and_rowwise_quant(
    IType* __restrict__ sIn0_ptr,
    IType* __restrict__ sIn1_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ row_scale_buf0,
    uint8_t* __restrict__ row_scale_buf1,
    const IType* __restrict__ dh,
    const IType* __restrict__ h13,
    int dh_cols,
    int h13_stride,
    int h3_offset,
    int block_offset_Y,
    int block_offset_X,
    int stage_Y,
    int stage_X
) {
    auto& sIn0 = *reinterpret_cast<TileInputBuf2D*>(sIn0_ptr);
    auto& sIn1 = *reinterpret_cast<TileInputBuf2D*>(sIn1_ptr);
    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);

    const int thread_lane = threadIdx.x % 32;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int mx_block_in_tile = thread_offset_X / MX_BLOCK;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + mx_block_in_tile;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const int global_row = block_offset_Y + row;
        const int scale_row = stage_Y * TILE_DIM + row;

        __align__(16) IType2 rOut0[WAVES][PACK_SIZE / 2];
        __align__(16) IType2 rOut1[WAVES][PACK_SIZE / 2];
        IType2 amax0_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        IType2 amax1_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            const int64_t dh_base =
                static_cast<int64_t>(global_row) * dh_cols + block_offset_X + thread_offset_X + sw;
            const int64_t h_base =
                static_cast<int64_t>(global_row) * h13_stride + block_offset_X + thread_offset_X + sw;

            const int4 d_vec = *reinterpret_cast<const int4*>(dh + dh_base);
            const int4 a_vec = *reinterpret_cast<const int4*>(h13 + h_base + h3_offset);
            const int4 b_vec = *reinterpret_cast<const int4*>(h13 + h_base);

            const auto* d_pairs = reinterpret_cast<const __nv_bfloat162*>(&d_vec);
            const auto* a_pairs = reinterpret_cast<const __nv_bfloat162*>(&a_vec);
            const auto* b_pairs = reinterpret_cast<const __nv_bfloat162*>(&b_vec);

            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                const float2 d = __bfloat1622float2(d_pairs[e]);
                const float2 a = __bfloat1622float2(a_pairs[e]);
                const float2 b = __bfloat1622float2(b_pairs[e]);

                const float sig_x = 1.0f / (1.0f + __expf(-b.x));
                const float sig_y = 1.0f / (1.0f + __expf(-b.y));
                const float silu_x = b.x * sig_x;
                const float silu_y = b.y * sig_y;
                const float silup_x = sig_x * (1.0f + b.x - silu_x);
                const float silup_y = sig_y * (1.0f + b.y - silu_y);

                rOut0[w][e] = {
                    __float2bfloat16_rn(d.x * a.x * silup_x),
                    __float2bfloat16_rn(d.y * a.y * silup_y),
                };
                rOut1[w][e] = {
                    __float2bfloat16_rn(d.x * silu_x),
                    __float2bfloat16_rn(d.y * silu_y),
                };
                abs_max_2x(amax0_2x, amax0_2x, rOut0[w][e]);
                abs_max_2x(amax1_2x, amax1_2x, rOut1[w][e]);
            }

            *reinterpret_cast<int4*>(&sIn0[row][thread_offset_X + sw]) =
                *reinterpret_cast<const int4*>(&rOut0[w][0]);
            *reinterpret_cast<int4*>(&sIn1[row][thread_offset_X + sw]) =
                *reinterpret_cast<const int4*>(&rOut1[w][0]);
        }

        const float my_amax0 = fmaxf(
            __bfloat162float(__habs(amax0_2x.x)),
            __bfloat162float(__habs(amax0_2x.y)));
        const float my_amax1 = fmaxf(
            __bfloat162float(__habs(amax1_2x.x)),
            __bfloat162float(__habs(amax1_2x.y)));

        const float pair_amax0 = __shfl_xor_sync(0xffffffff, my_amax0, 1);
        const float pair_amax1 = __shfl_xor_sync(0xffffffff, my_amax1, 1);
        const float block_amax0 = fmaxf(my_amax0, pair_amax0);
        const float block_amax1 = fmaxf(my_amax1, pair_amax1);

        const uint8_t e8m0_0 = float_to_e8m0<MODE>(block_amax0);
        const uint8_t e8m0_1 = float_to_e8m0<MODE>(block_amax1);

        if ((tid_X % 2) == 0) {
            row_scale_buf0[scale_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_0;
            row_scale_buf1[scale_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_1;
        }

        const float coeff0 = 6.0f * exp2f_rcp(e8m0_0);
        const float coeff1 = 6.0f * exp2f_rcp(e8m0_1);

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03_0 = *reinterpret_cast<const uint64_t*>(&rOut0[w][0]);
            const uint64_t e47_0 = *reinterpret_cast<const uint64_t*>(&rOut0[w][2]);
            const uint64_t e03_1 = *reinterpret_cast<const uint64_t*>(&rOut1[w][0]);
            const uint64_t e47_1 = *reinterpret_cast<const uint64_t*>(&rOut1[w][2]);
            const uint32_t out0 =
                mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff0);
            const uint32_t out1 =
                mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff1);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[0][row][(sw + thread_offset_X) / 2], out0);
            ptx::st_shared_b32(&sOut[1][row][(sw + thread_offset_X) / 2], out1);
        }
    }

    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE, bool SHARED_2D_WEIGHT = false>
__device__ __forceinline__ void quantize_chunk_rowcol_pipelined_impl(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + ntx * TILE_DIM,
                    input_block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        if constexpr (SHARED_2D_WEIGHT) {
            mx_weight_2d_quantize(
                sIn_ptr,
                sOut_ptr,
                row_scale_buf,
                col_scale_buf,
                stage_Y,
                stage_X,
                t,
                row_buff_out,
                col_buff_out);
        } else {
            mx_rowwise_quantize<MODE>(
                sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out);
        }

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        if constexpr (!SHARED_2D_WEIGHT) {
            mx_colwise_quantize_direct<MODE>(
                sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);
        }

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE = QuantMode::RTE, bool SHARED_2D_WEIGHT = false>
__device__ __forceinline__ void quantize_chunk_rowcol_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    quantize_chunk_rowcol_pipelined_impl<MODE, SHARED_2D_WEIGHT>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y,
        in_mbar,
        mbar_phase,
        tensor_map_input_ptr);
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_sqrelu_chunk_rowcol_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
        transform_sqrelu_tile_inplace(sIn_ptr, t);

        mx_rowwise_quantize<MODE>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + col_stage_offset_X,
                block_offset_X + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_sqrelu_deriv_chunk_rowcol_pipelined(
    IType* sDh_ptr,
    IType* sH1_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* dh_mbar,
    uint64_t* h1_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_dh_ptr,
    const CUtensorMap* tensor_map_h1_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sDh = *reinterpret_cast<InputBuf3D*>(sDh_ptr);
    auto& sH1 = *reinterpret_cast<InputBuf3D*>(sH1_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&dh_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_dh_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &dh_mbar[next]);
                mbarrier_arrive_expect_tx(&h1_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_h1_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &h1_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], mbar_phase);
        mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);
        transform_sqrelu_deriv_tile_inplace(sDh_ptr, sH1_ptr, t);

        mx_rowwise_quantize<MODE>(
            sDh_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct<MODE>(
            sDh_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + col_stage_offset_X,
                block_offset_X + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RHT = false,
    int RHT_BLOCK_SIZE = 16,
    bool WITH_RANDOM_SIGN_MASK = true,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = WITH_RHT>
__device__ __forceinline__ void quantize_sqrelu_chunk_rowcol_pipelined_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
        transform_sqrelu_tile_inplace(sIn_ptr, t);

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, ROW_WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        const uint64_t row_cta_y = static_cast<uint64_t>(block_offset_Y / CHUNK_DIM);
        const uint64_t row_cta_x = static_cast<uint64_t>(block_offset_X / CHUNK_DIM);
        const uint64_t col_rng_subsequence_base =
            rng_subsequence_base
            + ((row_cta_x * static_cast<uint64_t>(gridDim.y) + row_cta_y)
               - (row_cta_y * static_cast<uint64_t>(gridDim.x) + row_cta_x))
                * NUM_TILES * THREADS;

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, COL_WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
            rng_seed, col_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + col_stage_offset_X,
                block_offset_X + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RHT = false,
    int RHT_BLOCK_SIZE = 16,
    bool WITH_RANDOM_SIGN_MASK = true,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = WITH_RHT>
__device__ __forceinline__ void quantize_sqrelu_deriv_chunk_rowcol_pipelined_opt(
    IType* sDh_ptr,
    IType* sH1_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* dh_mbar,
    uint64_t* h1_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_dh_ptr,
    const CUtensorMap* tensor_map_h1_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sDh = *reinterpret_cast<InputBuf3D*>(sDh_ptr);
    auto& sH1 = *reinterpret_cast<InputBuf3D*>(sH1_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&dh_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_dh_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &dh_mbar[next]);
                mbarrier_arrive_expect_tx(&h1_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_h1_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &h1_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], mbar_phase);
        mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);
        transform_sqrelu_deriv_tile_inplace(sDh_ptr, sH1_ptr, t);

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, ROW_WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sDh_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        const uint64_t row_cta_y = static_cast<uint64_t>(block_offset_Y / CHUNK_DIM);
        const uint64_t row_cta_x = static_cast<uint64_t>(block_offset_X / CHUNK_DIM);
        const uint64_t col_rng_subsequence_base =
            rng_subsequence_base
            + ((row_cta_x * static_cast<uint64_t>(gridDim.y) + row_cta_y)
               - (row_cta_y * static_cast<uint64_t>(gridDim.x) + row_cta_x))
                * NUM_TILES * THREADS;

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, COL_WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sDh_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
            rng_seed, col_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + col_stage_offset_X,
                block_offset_X + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

// ROW_WITH_RHT applies H across K for the row-output contract (A H).
// COL_WITH_RHT applies H across M for the transposed/col-output contract (H B).
// The legacy WITH_RHT parameter maps to COL_WITH_RHT by default.
template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RHT = false,
    int RHT_BLOCK_SIZE = 16,
    bool WITH_RANDOM_SIGN_MASK = true,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = WITH_RHT>
__device__ __forceinline__ void quantize_chunk_rowcol_pipelined_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;
    constexpr bool CORRELATED_DATA_SR =
        DATA_SR && !ROW_WITH_RHT && !COL_WITH_RHT;
    const uint64_t correlated_sr_base =
        rng_seed ^ rng_subsequence_base ^ 0xa0761d6478bd642full;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + ntx * TILE_DIM,
                    input_block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_rowwise_quantize_opt<
            MODE, DATA_SR, SCALE_SR, ROW_WITH_RHT, RHT_BLOCK_SIZE,
            WITH_RANDOM_SIGN_MASK, CORRELATED_DATA_SR>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, rng_subsequence_base,
            input_block_offset_Y, input_block_offset_X, correlated_sr_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        const uint64_t row_cta_y = static_cast<uint64_t>(input_block_offset_Y / CHUNK_DIM);
        const uint64_t row_cta_x = static_cast<uint64_t>(input_block_offset_X / CHUNK_DIM);
        const uint64_t col_rng_subsequence_base =
            rng_subsequence_base
            + ((row_cta_x * static_cast<uint64_t>(gridDim.y) + row_cta_y)
               - (row_cta_y * static_cast<uint64_t>(gridDim.x) + row_cta_x))
                * NUM_TILES * THREADS;

        mx_colwise_quantize_direct_opt<
            MODE, DATA_SR, SCALE_SR, COL_WITH_RHT, RHT_BLOCK_SIZE,
            WITH_RANDOM_SIGN_MASK, CORRELATED_DATA_SR>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
            rng_seed, col_rng_subsequence_base,
            input_block_offset_Y, input_block_offset_X, correlated_sr_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_rowcol_pipelined_split_input(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    quantize_chunk_rowcol_pipelined_impl<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        input_block_offset_Y,
        input_block_offset_X,
        row_output_block_offset_Y,
        row_output_block_offset_X,
        col_output_block_offset_Y,
        col_output_block_offset_X,
        in_mbar,
        mbar_phase,
        tensor_map_input_ptr);
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void quantize_chunk_rowcol_pipelined_split_input_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + ntx * TILE_DIM,
                    input_block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        const uint64_t row_cta_y = static_cast<uint64_t>(row_output_block_offset_Y / CHUNK_DIM);
        const uint64_t row_cta_x = static_cast<uint64_t>(row_output_block_offset_X / CHUNK_DIM);
        const uint64_t row_rng_subsequence_base =
            rng_subsequence_base
            + (row_cta_y * static_cast<uint64_t>(gridDim.x) + row_cta_x)
                * NUM_TILES * THREADS;

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, false, 16, true>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, row_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        const uint64_t col_cta_y = static_cast<uint64_t>(col_output_block_offset_Y / CHUNK_DIM);
        const uint64_t col_cta_x = static_cast<uint64_t>(col_output_block_offset_X / CHUNK_DIM);
        const uint64_t col_rng_subsequence_base =
            rng_subsequence_base
            + (col_cta_y * static_cast<uint64_t>(gridDim.y) + col_cta_x)
                * NUM_TILES * THREADS;

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
            rng_seed, col_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void quantize_chunk_rowcol_resident_split_input_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    int row_grid_x,
    int col_grid_x,
    uint64_t* in_mbar,
    int mbar_phase,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        const uint64_t row_cta_y = static_cast<uint64_t>(row_output_block_offset_Y / CHUNK_DIM);
        const uint64_t row_cta_x = static_cast<uint64_t>(row_output_block_offset_X / CHUNK_DIM);
        const uint64_t row_rng_subsequence_base =
            rng_subsequence_base
            + (row_cta_y * static_cast<uint64_t>(row_grid_x) + row_cta_x)
                * NUM_TILES * THREADS;

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, false, 16, true>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, row_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        const uint64_t col_cta_y = static_cast<uint64_t>(col_output_block_offset_Y / CHUNK_DIM);
        const uint64_t col_cta_x = static_cast<uint64_t>(col_output_block_offset_X / CHUNK_DIM);
        const uint64_t col_rng_subsequence_base =
            rng_subsequence_base
            + (col_cta_y * static_cast<uint64_t>(col_grid_x) + col_cta_x)
                * NUM_TILES * THREADS;

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
            rng_seed, col_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_rowcol_pipelined_split_input_inverse_rope_live64(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const RopeLive64Desc& rope
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + ntx * TILE_DIM,
                    input_block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        apply_inverse_rope_tile_inplace_live64(
            sIn_ptr,
            rope,
            t,
            stage_Y,
            stage_X,
            input_block_offset_Y,
            input_block_offset_X);

        mx_rowwise_quantize<MODE>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void quantize_chunk_rowcol_pipelined_split_input_inverse_rope_live64_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const RopeLive64Desc& rope,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + ntx * TILE_DIM,
                    input_block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        apply_inverse_rope_tile_inplace_live64(
            sIn_ptr,
            rope,
            t,
            stage_Y,
            stage_X,
            input_block_offset_Y,
            input_block_offset_X);

        const uint64_t row_cta_y = static_cast<uint64_t>(row_output_block_offset_Y / CHUNK_DIM);
        const uint64_t row_cta_x = static_cast<uint64_t>(row_output_block_offset_X / CHUNK_DIM);
        const uint64_t row_rng_subsequence_base =
            rng_subsequence_base
            + (row_cta_y * static_cast<uint64_t>(gridDim.x) + row_cta_x)
                * NUM_TILES * THREADS;

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, false, 16, true>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, row_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        const uint64_t col_cta_y = static_cast<uint64_t>(col_output_block_offset_Y / CHUNK_DIM);
        const uint64_t col_cta_x = static_cast<uint64_t>(col_output_block_offset_X / CHUNK_DIM);
        const uint64_t col_rng_subsequence_base =
            rng_subsequence_base
            + (col_cta_y * static_cast<uint64_t>(gridDim.y) + col_cta_x)
                * NUM_TILES * THREADS;

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
            rng_seed, col_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void fused_rmsnorm_quantize_chunk_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const float* inv_rms_chunk,
    const IType* gamma_chunk
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next], TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        apply_rmsnorm_tile_inplace(
            sIn_ptr, gamma_chunk, inv_rms_chunk, t, stage_Y, stage_X);
        __syncthreads();

        mx_rowwise_quantize<MODE>(sIn_ptr, sOut_ptr, scale_buf, stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) {
        cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// Direct transposed quantization for one loaded 64x64 tile.
//
// This mirrors the tuned NVFP4 transpose kernel shape: read bf16x2 pairs
// directly from the original shared tile and emit the logical transpose
// contract without building a second BF16 tile ring.
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void mx_colwise_quantize_direct(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,
    const int stage_Y,
    const int stage_X,
    const int buff_in,
    const int buff_out
) {
    const auto& sIn2x = *reinterpret_cast<const InputBuf2x3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Four warps cover the four 16-row slices of the original tile.
    // Each lane owns one adjacent column pair, so together the CTA emits
    // 64 logical transpose rows (32 pairs) without a full shared transpose.
    const int tid_X_colwise = (lane % 16) + (warp / 2) * 16;   // 0..31 col pairs
    const int tid_Y_colwise = (warp % 2) * 2 + (lane / 16);    // 0..3 row-slice id

    const int thread_offset_Y = tid_Y_colwise * ELTS_PER_THREAD;
    const int in_thread_offset_X = tid_X_colwise;

    const int out_thread_offset_Y = tid_X_colwise * 2;
    const int out_thread_offset_X = tid_Y_colwise * (ELTS_PER_THREAD / 2);

    const int scale_block_in_tile = tid_Y_colwise / 2;
    const int global_scale_y = stage_Y * TILE_DIM + out_thread_offset_Y;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + scale_block_in_tile;
    const bool scale_storing_thread = ((tid_Y_colwise & 1) == 0);

    __align__(8) IType rIn[2][ELTS_PER_THREAD];
    IType2 thread_amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; ++i) {
        const IType2 elt_pair =
            ptx::ld_shared_b32(&sIn2x[buff_in][thread_offset_Y + i][in_thread_offset_X]);
        rIn[0][i] = elt_pair.x;
        rIn[1][i] = elt_pair.y;
        ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, elt_pair);
    }

    // Pair 16-row slices inside the same 32-row MX block.
    const float thread_amax0 = __bfloat162float(__habs(thread_amax_2x.x));
    const float thread_amax1 = __bfloat162float(__habs(thread_amax_2x.y));
    const float pair_amax0 = __shfl_xor_sync(0xffffffff, thread_amax0, 16);
    const float pair_amax1 = __shfl_xor_sync(0xffffffff, thread_amax1, 16);
    const float block_amax0 = fmaxf(thread_amax0, pair_amax0);
    const float block_amax1 = fmaxf(thread_amax1, pair_amax1);

    const uint8_t e8m0_0 = float_to_e8m0<MODE>(block_amax0);
    const uint8_t e8m0_1 = float_to_e8m0<MODE>(block_amax1);

    if (scale_storing_thread) {
        scale_buf[(global_scale_y + 0) * SCALES_PER_CHUNK + global_scale_x] = e8m0_0;
        scale_buf[(global_scale_y + 1) * SCALES_PER_CHUNK + global_scale_x] = e8m0_1;
    }

    const float coeff0 = 6.0f * exp2f_rcp(e8m0_0);
    const float coeff1 = 6.0f * exp2f_rcp(e8m0_1);

    #pragma unroll
    for (int row_pair = 0; row_pair < 2; ++row_pair) {
        const float coeff = (row_pair == 0) ? coeff0 : coeff1;
        const uint64_t elts03_lo = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][0]);
        const uint64_t elts47_lo = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][4]);
        const uint64_t elts03_hi = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][8]);
        const uint64_t elts47_hi = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][12]);

        const uint32_t out_lo =
            mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(elts03_lo, elts47_lo, coeff);
        const uint32_t out_hi =
            mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(elts03_hi, elts47_hi, coeff);

        uint64_t packed = static_cast<uint64_t>(out_lo) |
                          (static_cast<uint64_t>(out_hi) << 32);
        ptx::st_shared_b64(
            &sOut[buff_out][out_thread_offset_Y + row_pair][out_thread_offset_X], packed);
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT, int RHT_BLOCK_SIZE, bool WITH_RANDOM_SIGN_MASK, bool CORRELATED_DATA_SR>
__device__ __forceinline__ void mx_colwise_quantize_direct_opt(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,
    const int stage_Y,
    const int stage_X,
    const int buff_in,
    const int buff_out,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base,
    const int logical_row_offset,
    const int logical_col_offset,
    const uint64_t correlated_sr_base
) {
    static_assert(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32,
                  "RHT block size must be 16 or 32");

    const auto& sIn2x = *reinterpret_cast<const InputBuf2x3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int tid_X_colwise = (lane % 16) + (warp / 2) * 16;
    const int tid_Y_colwise = (warp % 2) * 2 + (lane / 16);

    const int thread_offset_Y = tid_Y_colwise * ELTS_PER_THREAD;
    const int in_thread_offset_X = tid_X_colwise;

    const int out_thread_offset_Y = tid_X_colwise * 2;
    const int out_thread_offset_X = tid_Y_colwise * (ELTS_PER_THREAD / 2);

    const int scale_block_in_tile = tid_Y_colwise / 2;
    const int global_scale_y = stage_Y * TILE_DIM + out_thread_offset_Y;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + scale_block_in_tile;
    const bool scale_storing_thread = ((tid_Y_colwise & 1) == 0);

    RNGState rng;
    const uint64_t tile_linear_idx = static_cast<uint64_t>(stage_Y * TILES_X + stage_X);
    if constexpr (
        (DATA_SR && !CORRELATED_DATA_SR) || SCALE_SR ||
        (WITH_RHT && WITH_RANDOM_SIGN_MASK)) {
        rng.init(
            rng_seed,
            rng_subsequence_base + tile_linear_idx * THREADS + threadIdx.x,
            0);
    }
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;

    __align__(8) IType rIn[2][ELTS_PER_THREAD];
    float vals[2][ELTS_PER_THREAD];
    float my_amax[2] = {0.0f, 0.0f};

    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; ++i) {
        const IType2 elt_pair =
            ptx::ld_shared_b32(&sIn2x[buff_in][thread_offset_Y + i][in_thread_offset_X]);
        rIn[0][i] = elt_pair.x;
        rIn[1][i] = elt_pair.y;
        if constexpr (WITH_RHT) {
            vals[0][i] = __bfloat162float(elt_pair.x);
            vals[1][i] = __bfloat162float(elt_pair.y);
        } else {
            my_amax[0] = fmaxf(my_amax[0], fabsf(__bfloat162float(elt_pair.x)));
            my_amax[1] = fmaxf(my_amax[1], fabsf(__bfloat162float(elt_pair.y)));
        }
    }

    if constexpr (WITH_RHT) {
        const int block_thread_rank =
            (thread_offset_Y % RHT_BLOCK_SIZE) / ELTS_PER_THREAD;
        #pragma unroll
        for (int row_pair = 0; row_pair < 2; ++row_pair) {
            if constexpr (RHT_BLOCK_SIZE == 32) {
                const uint32_t sign_bits =
                    make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
                #pragma unroll
                for (int i = 0; i < ELTS_PER_THREAD; ++i) {
                    if constexpr (WITH_RANDOM_SIGN_MASK) {
                        vals[row_pair][i] *= ((sign_bits >> i) & 1u) ? 1.0f : -1.0f;
                    }
                }
                fwht16_unnormalized(vals[row_pair]);
                #pragma unroll
                for (int i = 0; i < ELTS_PER_THREAD; ++i) {
                    const float peer = __shfl_xor_sync(0xffffffff, vals[row_pair][i], 16);
                    vals[row_pair][i] =
                        ((tid_Y_colwise & 1) == 0) ? (vals[row_pair][i] + peer) : (peer - vals[row_pair][i]);
                }
                constexpr float kNorm32 = 0.1767766952966369f;
                #pragma unroll
                for (int i = 0; i < ELTS_PER_THREAD; ++i) {
                    vals[row_pair][i] *= kNorm32;
                }
            } else {
                const uint32_t sign_bits =
                    make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
                apply_block_rht_registers<16, WITH_RANDOM_SIGN_MASK>(
                    vals[row_pair],
                    0,
                    sign_bits);
            }

            #pragma unroll
            for (int i = 0; i < ELTS_PER_THREAD; ++i) {
                my_amax[row_pair] = fmaxf(my_amax[row_pair], fabsf(vals[row_pair][i]));
                rIn[row_pair][i] = __float2bfloat16_rn(vals[row_pair][i]);
            }
        }
    }

    const float pair_amax0 = __shfl_xor_sync(0xffffffff, my_amax[0], 16);
    const float pair_amax1 = __shfl_xor_sync(0xffffffff, my_amax[1], 16);
    const float block_amax0 = fmaxf(my_amax[0], pair_amax0);
    const float block_amax1 = fmaxf(my_amax[1], pair_amax1);

    const uint8_t e8m0_0 = SCALE_SR
        ? float_to_e8m0_stochastic(block_amax0, next_rbits(rng, random_uint4, rnd_idx))
        : float_to_e8m0<MODE>(block_amax0);
    const uint8_t e8m0_1 = SCALE_SR
        ? float_to_e8m0_stochastic(block_amax1, next_rbits(rng, random_uint4, rnd_idx))
        : float_to_e8m0<MODE>(block_amax1);

    if (scale_storing_thread) {
        scale_buf[(global_scale_y + 0) * SCALES_PER_CHUNK + global_scale_x] = e8m0_0;
        scale_buf[(global_scale_y + 1) * SCALES_PER_CHUNK + global_scale_x] = e8m0_1;
    }

    const float coeff0 = 6.0f * exp2f_rcp(e8m0_0);
    const float coeff1 = 6.0f * exp2f_rcp(e8m0_1);

    #pragma unroll
    for (int row_pair = 0; row_pair < 2; ++row_pair) {
        const float coeff = (row_pair == 0) ? coeff0 : coeff1;
        uint64_t elts03_lo = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][0]);
        uint64_t elts47_lo = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][4]);
        uint64_t elts03_hi = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][8]);
        uint64_t elts47_hi = *reinterpret_cast<const uint64_t*>(&rIn[row_pair][12]);

        const int logical_row =
            logical_row_offset + stage_X * TILE_DIM + thread_offset_Y;
        const int logical_col =
            logical_col_offset + stage_Y * TILE_DIM + out_thread_offset_Y + row_pair;
        if constexpr (DATA_SR && CORRELATED_DATA_SR) {
            if ((logical_col & 1) != 0) {
                elts03_lo = mxfp4_swap_bf16_pair_lanes(elts03_lo);
                elts47_lo = mxfp4_swap_bf16_pair_lanes(elts47_lo);
                elts03_hi = mxfp4_swap_bf16_pair_lanes(elts03_hi);
                elts47_hi = mxfp4_swap_bf16_pair_lanes(elts47_hi);
            }
        }

        uint32_t out_lo;
        uint32_t out_hi;
        if constexpr (DATA_SR) {
            uint32_t rbits03_lo;
            uint32_t rbits47_lo;
            uint32_t rbits03_hi;
            uint32_t rbits47_hi;
            if constexpr (CORRELATED_DATA_SR) {
                const uint4 rbits =
                    mxfp4_correlated_sr_rbits16<false>(
                        correlated_sr_base, logical_row, logical_col);
                rbits03_lo = rbits.x;
                rbits47_lo = rbits.y;
                rbits03_hi = rbits.z;
                rbits47_hi = rbits.w;
            } else {
                rbits03_lo = next_rbits(rng, random_uint4, rnd_idx);
                rbits47_lo = next_rbits(rng, random_uint4, rnd_idx);
                rbits03_hi = next_rbits(rng, random_uint4, rnd_idx);
                rbits47_hi = next_rbits(rng, random_uint4, rnd_idx);
            }
            const bf16 coeff_bf16 = __float2bfloat16(coeff);
            out_lo = mul_cvt_bf16_to_fp4_8x_stochastic_rounding<bf16>(
                elts03_lo, elts47_lo, coeff_bf16, rbits03_lo, rbits47_lo);
            out_hi = mul_cvt_bf16_to_fp4_8x_stochastic_rounding<bf16>(
                elts03_hi, elts47_hi, coeff_bf16, rbits03_hi, rbits47_hi);
            if constexpr (CORRELATED_DATA_SR) {
                if ((logical_col & 1) != 0) {
                    out_lo = mxfp4_swap_fp4_pair_lanes(out_lo);
                    out_hi = mxfp4_swap_fp4_pair_lanes(out_hi);
                }
            }
        } else {
            const bf16 coeff_bf16 = __float2bfloat16(coeff);
            out_lo = mul_cvt_bf16_to_fp4_8x_round_to_nearest<bf16>(
                elts03_lo, elts47_lo, coeff_bf16);
            out_hi = mul_cvt_bf16_to_fp4_8x_round_to_nearest<bf16>(
                elts03_hi, elts47_hi, coeff_bf16);
        }

        const uint64_t packed =
            static_cast<uint64_t>(out_lo) | (static_cast<uint64_t>(out_hi) << 32);
        ptx::st_shared_b64(
            &sOut[buff_out][out_thread_offset_Y + row_pair][out_thread_offset_X], packed);
    }
}


// ═══════════════════════════════════════════════════════════════════
// Pipelined col-only quantize for one chunk.
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_transposed_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_Y + nty * TILE_DIM,
                    block_offset_X + ntx * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, scale_buf, stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void quantize_chunk_transposed_pipelined_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_Y + nty * TILE_DIM,
                    block_offset_X + ntx * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr,
            sOut_ptr,
            scale_buf,
            stage_Y,
            stage_X,
            t,
            buff_out,
            rng_seed,
            rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_transposed_pipelined_split_input(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int output_block_offset_Y,
    int output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_Y + nty * TILE_DIM,
                    input_block_offset_X + ntx * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, scale_buf, stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                output_block_offset_X + stage_offset_X,
                output_block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void quantize_chunk_transposed_pipelined_split_input_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int input_block_offset_Y,
    int input_block_offset_X,
    int output_block_offset_Y,
    int output_block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    const uint64_t cta_y = static_cast<uint64_t>(output_block_offset_Y / CHUNK_DIM);
    const uint64_t cta_x = static_cast<uint64_t>(output_block_offset_X / CHUNK_DIM);
    const uint64_t cta_rng_subsequence_base =
        rng_subsequence_base + (cta_y * static_cast<uint64_t>(gridDim.x) + cta_x) * NUM_TILES * THREADS;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_Y + nty * TILE_DIM,
                    input_block_offset_X + ntx * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, scale_buf, stage_Y, stage_X, t, buff_out,
            rng_seed, cta_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                output_block_offset_X + stage_offset_X,
                output_block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_preloaded_row(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y,
    int block_offset_X
) {
    const bool leading = (threadIdx.x == 0);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        mx_rowwise_quantize<MODE>(sIn_ptr, sOut_ptr, scale_buf, stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_preloaded_col(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int output_block_offset_Y,
    int output_block_offset_X
) {
    const bool leading = (threadIdx.x == 0);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        // The direct col emitter consumes the same original-layout shared tile
        // but expects transpose-oriented stage coordinates.
        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, scale_buf, stage_X, stage_Y, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                output_block_offset_X + col_stage_offset_X,
                output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_preloaded_rowcol(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X
) {
    const bool leading = (threadIdx.x == 0);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        mx_rowwise_quantize<MODE>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool ROW_WITH_RHT,
    bool COL_WITH_RHT,
    int RHT_BLOCK_SIZE = 32,
    bool WITH_RANDOM_SIGN_MASK = false>
__device__ __forceinline__ void quantize_chunk_preloaded_rowcol_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int row_output_block_offset_Y,
    int row_output_block_offset_X,
    int col_output_block_offset_Y,
    int col_output_block_offset_X,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, ROW_WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                row_output_block_offset_X + row_stage_offset_X,
                row_output_block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        if constexpr (COL_WITH_RHT || DATA_SR || SCALE_SR) {
            mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, COL_WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
                sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
                rng_seed, rng_subsequence_base);
        } else {
            mx_colwise_quantize_direct<MODE>(
                sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);
        }

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                col_output_block_offset_X + col_stage_offset_X,
                col_output_block_offset_Y + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void fused_rmsnorm_quantize_chunk_transposed_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const float* inv_rms_chunk,
    const IType* gamma_chunk
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        if (t >= BUFFS_OUT) {
            if (leading) {
                cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next], TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_Y + nty * TILE_DIM,
                    block_offset_X + ntx * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        apply_rmsnorm_tile_inplace_transposed_load(
            sIn_ptr, gamma_chunk, inv_rms_chunk, t, stage_Y, stage_X);
        __syncthreads();

        mx_colwise_quantize_direct<MODE>(sIn_ptr, sOut_ptr, scale_buf, stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    if (leading) {
        cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void fused_rmsnorm_quantize_chunk_rowcol_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const float* inv_rms_chunk,
    const IType* gamma_chunk,
    const CUtensorMap* tensor_map_norm_output_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next], TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        // Normalize once in the original tile layout, then emit both row and col
        // contracts directly from the same normalized shared tile.
        apply_rmsnorm_tile_inplace(
            sIn_ptr, gamma_chunk, inv_rms_chunk, t, stage_Y, stage_X);
        __syncthreads();

        if (tensor_map_norm_output_ptr != nullptr && leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(tensor_map_norm_output_ptr),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sIn[t]));
            cp_async_bulk_commit_group();
        }

        mx_rowwise_quantize<MODE>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        // The direct col emitter expects transpose-oriented stage coordinates.
        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + col_stage_offset_X,
                block_offset_X + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__device__ __forceinline__ void fused_rmsnorm_quantize_chunk_rowcol_pipelined_opt(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr,
    const float* inv_rms_chunk,
    const IType* gamma_chunk,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X;
            const int ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next], TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        apply_rmsnorm_tile_inplace(
            sIn_ptr, gamma_chunk, inv_rms_chunk, t, stage_Y, stage_X);
        __syncthreads();

        const uint64_t row_cta_y = static_cast<uint64_t>(block_offset_Y / CHUNK_DIM);
        const uint64_t row_cta_x = static_cast<uint64_t>(block_offset_X / CHUNK_DIM);
        const uint64_t row_rng_subsequence_base =
            rng_subsequence_base
            + (row_cta_y * static_cast<uint64_t>(gridDim.x) + row_cta_x)
                * NUM_TILES * THREADS;

        mx_rowwise_quantize_opt<MODE, DATA_SR, SCALE_SR, false, 16, true>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out,
            rng_seed, row_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        const uint64_t col_cta_y = static_cast<uint64_t>(block_offset_X / CHUNK_DIM);
        const uint64_t col_cta_x = static_cast<uint64_t>(block_offset_Y / CHUNK_DIM);
        const uint64_t col_rng_subsequence_base =
            rng_subsequence_base
            + (col_cta_y * static_cast<uint64_t>(gridDim.y) + col_cta_x)
                * NUM_TILES * THREADS;

        mx_colwise_quantize_direct_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out,
            rng_seed, col_rng_subsequence_base);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + col_stage_offset_X,
                block_offset_X + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}


// ═══════════════════════════════════════════════════════════════════
// Write scales to global memory (swizzled layout for TK GEMM)
// Layout: [M/128, K/128, 32, 16] packed as uint8
//   Scale for row r, block b goes to:
//     base = (r/128 * ntk + chunk_x) * 512 + (r%128)%32 * 16 + (r%128)/32 * 4 + b
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ void write_scales_swizzled(
    const uint8_t* scale_buf,   // [CHUNK_DIM × SCALES_PER_CHUNK]
    uint8_t* __restrict__ global_scales,
    int ctaid_X, int ctaid_Y,
    int ntk
) {
    // Each thread processes one row of scales (4 values)
    for (int row = threadIdx.x; row < CHUNK_DIM; row += THREADS) {
        const int j = row % 32;
        const int grp = row / 32;
        const int base = (ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
        uint32_t pk;
        uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
        p[0] = scale_buf[row * SCALES_PER_CHUNK + 0];
        p[1] = scale_buf[row * SCALES_PER_CHUNK + 1];
        p[2] = scale_buf[row * SCALES_PER_CHUNK + 2];
        p[3] = scale_buf[row * SCALES_PER_CHUNK + 3];
        *reinterpret_cast<uint32_t*>(global_scales + base) = pk;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Persistent single-phase quantize kernel
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_persistent_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M, const int64_t K,
    PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;

    // ─── SMEM layout ───
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();


    // ═══════════════════════════════════════════════════════════════
    // Persistent work-stealing loop — SINGLE PHASE
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * CHUNK_DIM;
        const int block_offset_X = ctaid_X * CHUNK_DIM;

        // Reinitialize mbarriers for this chunk (clean state)
        if (leading) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                mbarrier_invalid(&in_mbar[t]);
                mbarrier_init(&in_mbar[t], 1);
            }
            fence_proxy_async_shared_cta();
        }
        __syncthreads();

        // Prefetch first 2 tiles
        #pragma unroll
        for (int pre = 0; pre < 2; ++pre) {
            const int ty = pre / TILES_X, tx = pre % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[pre],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * TILE_DIM,
                    block_offset_Y + ty * TILE_DIM,
                    &in_mbar[pre]);
            }
        }

        // Pipelined quantize of all 4 sub-tiles (always phase 0)
        quantize_chunk_pipelined<MODE>(
            sIn_ptr, sOut_ptr, scale_buf, sOut,
            tensor_map_output,
            block_offset_Y, block_offset_X,
            in_mbar, 0,
            &tensor_map_input);

        // Write scales to global
        write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

        // Sync before next chunk to ensure all threads finished reading scale_buf
        __syncthreads();
    }

    // Cleanup
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Non-persistent (fused) kernel — one CTA per chunk
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M, const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Prefetch first 2 tiles
    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    // Pipelined quantize
    quantize_chunk_pipelined<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        block_offset_Y, block_offset_X,
        in_mbar, 0,
        &tensor_map_input);

    // Write scales
    write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_fused_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M, const int64_t K,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_pipelined_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        scale_buf,
        sOut,
        tensor_map_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input,
        rng_seed,
        rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS
    );

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE, bool SHARED_2D_WEIGHT = false>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_rowcol_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = K / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_rowcol_pipelined<MODE, SHARED_2D_WEIGHT>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RHT = false,
    int RHT_BLOCK_SIZE = 16,
    bool WITH_RANDOM_SIGN_MASK = true,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = WITH_RHT>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_rowcol_fused_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base,
    const uint64_t* __restrict__ rng_state
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint64_t active_rng_seed = rng_seed;
    uint64_t active_rng_subsequence_base = rng_subsequence_base;
    if constexpr (DATA_SR || SCALE_SR || (WITH_RHT && WITH_RANDOM_SIGN_MASK)) {
        if (rng_state != nullptr) {
            active_rng_seed = rng_state[0];
            active_rng_subsequence_base = rng_state[1];
        }
    }
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = K / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_rowcol_pipelined_opt<
        MODE,
        DATA_SR,
        SCALE_SR,
        WITH_RHT,
        RHT_BLOCK_SIZE,
        WITH_RANDOM_SIGN_MASK,
        ROW_WITH_RHT,
        COL_WITH_RHT>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y,
        in_mbar,
        0,
        &tensor_map_input,
        active_rng_seed,
        active_rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split2_rowcol_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk0 = K0 / CHUNK_DIM;
    const int row_ntk = row_ntk0 + (K1 / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < row_ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
    } else {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - row_ntk0;
    }

    const int input_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;
    const int row_output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int row_output_block_offset_X = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_Y = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_X = ctaid_Y * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_rowcol_pipelined_split_input<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        input_block_offset_Y,
        input_block_offset_X,
        row_output_block_offset_Y,
        row_output_block_offset_X,
        col_output_block_offset_Y,
        col_output_block_offset_X,
        in_mbar,
        0,
        tensor_map_input_ptr);

    write_scales_swizzled(row_scale_buf, row_scales_out, logical_ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, logical_ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split2_rowcol_fused_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk0 = K0 / CHUNK_DIM;
    const int row_ntk = row_ntk0 + (K1 / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < row_ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
    } else {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - row_ntk0;
    }

    const int input_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;
    const int row_output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int row_output_block_offset_X = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_Y = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_X = ctaid_Y * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_rowcol_pipelined_split_input_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        input_block_offset_Y,
        input_block_offset_X,
        row_output_block_offset_Y,
        row_output_block_offset_X,
        col_output_block_offset_Y,
        col_output_block_offset_X,
        in_mbar,
        0,
        tensor_map_input_ptr,
        rng_seed,
        rng_subsequence_base);

    write_scales_swizzled(row_scale_buf, row_scales_out, logical_ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, logical_ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split2_rowcol_persistent_resident_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base,
    PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk0 = K0 / CHUNK_DIM;
    const int row_ntk = row_ntk0 + (K1 / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int logical_ctaid_X = static_cast<int>(s_chunk_id % args.tiles_X);
        const int ctaid_Y = static_cast<int>(s_chunk_id / args.tiles_X);

        int local_ctaid_X = logical_ctaid_X;
        const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
        if (logical_ctaid_X < row_ntk0) {
            tensor_map_input_ptr = &tensor_map_input0;
        } else {
            tensor_map_input_ptr = &tensor_map_input1;
            local_ctaid_X = logical_ctaid_X - row_ntk0;
        }

        const int input_block_offset_Y = ctaid_Y * CHUNK_DIM;
        const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;
        const int row_output_block_offset_Y = ctaid_Y * CHUNK_DIM;
        const int row_output_block_offset_X = logical_ctaid_X * CHUNK_DIM;
        const int col_output_block_offset_Y = logical_ctaid_X * CHUNK_DIM;
        const int col_output_block_offset_X = ctaid_Y * CHUNK_DIM;

        if (leading) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                mbarrier_invalid(&in_mbar[t]);
                mbarrier_init(&in_mbar[t], 1);
            }
            fence_proxy_async_shared_cta();
        }
        __syncthreads();

        if (leading) {
            #pragma unroll
            for (int pre = 0; pre < NUM_TILES; ++pre) {
                const int ty = pre / TILES_X;
                const int tx = pre % TILES_X;
                mbarrier_arrive_expect_tx(&in_mbar[pre],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    input_block_offset_X + tx * TILE_DIM,
                    input_block_offset_Y + ty * TILE_DIM,
                    &in_mbar[pre]);
            }
        }

        quantize_chunk_rowcol_resident_split_input_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr,
            sOut_ptr,
            row_scale_buf,
            col_scale_buf,
            sOut,
            tensor_map_row_output,
            tensor_map_col_output,
            row_output_block_offset_Y,
            row_output_block_offset_X,
            col_output_block_offset_Y,
            col_output_block_offset_X,
            row_ntk,
            col_ntk,
            in_mbar,
            0,
            rng_seed,
            rng_subsequence_base);

        write_scales_swizzled(row_scale_buf, row_scales_out, logical_ctaid_X, ctaid_Y, row_ntk);
        write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, logical_ctaid_X, col_ntk);
        __syncthreads();
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split3_rowcol_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const int64_t K2
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk0 = K0 / CHUNK_DIM;
    const int row_ntk1 = K1 / CHUNK_DIM;
    const int row_ntk = row_ntk0 + row_ntk1 + (K2 / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < row_ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
    } else if (logical_ctaid_X < row_ntk0 + row_ntk1) {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - row_ntk0;
    } else {
        tensor_map_input_ptr = &tensor_map_input2;
        local_ctaid_X = logical_ctaid_X - row_ntk0 - row_ntk1;
    }

    const int input_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;
    const int row_output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int row_output_block_offset_X = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_Y = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_X = ctaid_Y * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_rowcol_pipelined_split_input<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        input_block_offset_Y,
        input_block_offset_X,
        row_output_block_offset_Y,
        row_output_block_offset_X,
        col_output_block_offset_Y,
        col_output_block_offset_X,
        in_mbar,
        0,
        tensor_map_input_ptr);

    write_scales_swizzled(row_scale_buf, row_scales_out, logical_ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, logical_ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split3_rowcol_fused_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const int64_t K2,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk0 = K0 / CHUNK_DIM;
    const int row_ntk1 = K1 / CHUNK_DIM;
    const int row_ntk = row_ntk0 + row_ntk1 + (K2 / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < row_ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
    } else if (logical_ctaid_X < row_ntk0 + row_ntk1) {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - row_ntk0;
    } else {
        tensor_map_input_ptr = &tensor_map_input2;
        local_ctaid_X = logical_ctaid_X - row_ntk0 - row_ntk1;
    }

    const int input_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;
    const int row_output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int row_output_block_offset_X = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_Y = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_X = ctaid_Y * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_rowcol_pipelined_split_input_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        input_block_offset_Y,
        input_block_offset_X,
        row_output_block_offset_Y,
        row_output_block_offset_X,
        col_output_block_offset_Y,
        col_output_block_offset_X,
        in_mbar,
        0,
        tensor_map_input_ptr,
        rng_seed,
        rng_subsequence_base);

    write_scales_swizzled(row_scale_buf, row_scales_out, logical_ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, logical_ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const float2* __restrict__ rope_cs,
    const int seq_mask,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const int64_t K2
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk0 = K0 / CHUNK_DIM;
    const int row_ntk1 = K1 / CHUNK_DIM;
    const int row_ntk = row_ntk0 + row_ntk1 + (K2 / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const bool apply_inverse_rope = logical_ctaid_X < (row_ntk0 + row_ntk1);

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < row_ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
    } else if (logical_ctaid_X < row_ntk0 + row_ntk1) {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - row_ntk0;
    } else {
        tensor_map_input_ptr = &tensor_map_input2;
        local_ctaid_X = logical_ctaid_X - row_ntk0 - row_ntk1;
    }

    const int input_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;
    const int row_output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int row_output_block_offset_X = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_Y = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_X = ctaid_Y * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    if (apply_inverse_rope) {
        quantize_chunk_rowcol_pipelined_split_input_inverse_rope_live64<MODE>(
            sIn_ptr,
            sOut_ptr,
            row_scale_buf,
            col_scale_buf,
            sOut,
            tensor_map_row_output,
            tensor_map_col_output,
            input_block_offset_Y,
            input_block_offset_X,
            row_output_block_offset_Y,
            row_output_block_offset_X,
            col_output_block_offset_Y,
            col_output_block_offset_X,
            in_mbar,
            0,
            tensor_map_input_ptr,
            RopeLive64Desc{rope_cs, seq_mask});
    } else {
        quantize_chunk_rowcol_pipelined_split_input<MODE>(
            sIn_ptr,
            sOut_ptr,
            row_scale_buf,
            col_scale_buf,
            sOut,
            tensor_map_row_output,
            tensor_map_col_output,
            input_block_offset_Y,
            input_block_offset_X,
            row_output_block_offset_Y,
            row_output_block_offset_X,
            col_output_block_offset_Y,
            col_output_block_offset_X,
            in_mbar,
            0,
            tensor_map_input_ptr);
    }

    write_scales_swizzled(row_scale_buf, row_scales_out, logical_ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, logical_ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const float2* __restrict__ rope_cs,
    const int seq_mask,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const int64_t K2,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk0 = K0 / CHUNK_DIM;
    const int row_ntk1 = K1 / CHUNK_DIM;
    const int row_ntk = row_ntk0 + row_ntk1 + (K2 / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const bool apply_inverse_rope = logical_ctaid_X < (row_ntk0 + row_ntk1);

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < row_ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
    } else if (logical_ctaid_X < row_ntk0 + row_ntk1) {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - row_ntk0;
    } else {
        tensor_map_input_ptr = &tensor_map_input2;
        local_ctaid_X = logical_ctaid_X - row_ntk0 - row_ntk1;
    }

    const int input_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;
    const int row_output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int row_output_block_offset_X = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_Y = logical_ctaid_X * CHUNK_DIM;
    const int col_output_block_offset_X = ctaid_Y * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    if (apply_inverse_rope) {
        quantize_chunk_rowcol_pipelined_split_input_inverse_rope_live64_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr,
            sOut_ptr,
            row_scale_buf,
            col_scale_buf,
            sOut,
            tensor_map_row_output,
            tensor_map_col_output,
            input_block_offset_Y,
            input_block_offset_X,
            row_output_block_offset_Y,
            row_output_block_offset_X,
            col_output_block_offset_Y,
            col_output_block_offset_X,
            in_mbar,
            0,
            tensor_map_input_ptr,
            RopeLive64Desc{rope_cs, seq_mask},
            rng_seed,
            rng_subsequence_base);
    } else {
        quantize_chunk_rowcol_pipelined_split_input_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
            sIn_ptr,
            sOut_ptr,
            row_scale_buf,
            col_scale_buf,
            sOut,
            tensor_map_row_output,
            tensor_map_col_output,
            input_block_offset_Y,
            input_block_offset_X,
            row_output_block_offset_Y,
            row_output_block_offset_X,
            col_output_block_offset_Y,
            col_output_block_offset_X,
            in_mbar,
            0,
            tensor_map_input_ptr,
            rng_seed,
            rng_subsequence_base);
    }

    write_scales_swizzled(row_scale_buf, row_scales_out, logical_ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, logical_ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split2_row_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk0 = K0 / CHUNK_DIM;
    const int ntk1 = K1 / CHUNK_DIM;
    const int ntk = ntk0 + ntk1;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int output_block_offset_X = logical_ctaid_X * CHUNK_DIM;

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
        local_ctaid_X = logical_ctaid_X;
    } else {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - ntk0;
    }
    const int input_block_offset_Y = output_block_offset_Y;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_pipelined_split_input<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        input_block_offset_Y, input_block_offset_X,
        output_block_offset_Y, output_block_offset_X,
        in_mbar, 0,
        tensor_map_input_ptr);

    write_scales_swizzled(scale_buf, scales_out, logical_ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split2_row_fused_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk0 = K0 / CHUNK_DIM;
    const int ntk1 = K1 / CHUNK_DIM;
    const int ntk = ntk0 + ntk1;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int output_block_offset_X = logical_ctaid_X * CHUNK_DIM;

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
        local_ctaid_X = logical_ctaid_X;
    } else {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - ntk0;
    }
    const int input_block_offset_Y = output_block_offset_Y;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_pipelined_split_input_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        input_block_offset_Y, input_block_offset_X,
        output_block_offset_Y, output_block_offset_X,
        in_mbar, 0,
        tensor_map_input_ptr,
        rng_seed, rng_subsequence_base);

    write_scales_swizzled(scale_buf, scales_out, logical_ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}


template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split2_col_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk0 = K0 / CHUNK_DIM;
    const int ntk1 = K1 / CHUNK_DIM;
    const int ntk = ntk0 + ntk1;
    const int ctaid_X = blockIdx.x;
    const int logical_ctaid_Y = blockIdx.y;
    const int output_block_offset_Y = logical_ctaid_Y * CHUNK_DIM;
    const int output_block_offset_X = ctaid_X * CHUNK_DIM;

    int local_ctaid_Y = logical_ctaid_Y;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_Y < ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
        local_ctaid_Y = logical_ctaid_Y;
    } else {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_Y = logical_ctaid_Y - ntk0;
    }
    const int input_block_offset_Y = local_ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = output_block_offset_X;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_Y + ty * TILE_DIM,
                input_block_offset_X + tx * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_transposed_pipelined_split_input<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        input_block_offset_Y, input_block_offset_X,
        output_block_offset_Y, output_block_offset_X,
        in_mbar, 0,
        tensor_map_input_ptr);

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, logical_ctaid_Y, M / CHUNK_DIM);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split2_col_fused_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk0 = K0 / CHUNK_DIM;
    const int ntk1 = K1 / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int logical_ctaid_Y = blockIdx.y;
    const int output_block_offset_Y = logical_ctaid_Y * CHUNK_DIM;
    const int output_block_offset_X = ctaid_X * CHUNK_DIM;

    int local_ctaid_Y = logical_ctaid_Y;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_Y < ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
        local_ctaid_Y = logical_ctaid_Y;
    } else {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_Y = logical_ctaid_Y - ntk0;
    }
    const int input_block_offset_Y = local_ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = output_block_offset_X;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_Y + ty * TILE_DIM,
                input_block_offset_X + tx * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_transposed_pipelined_split_input_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        input_block_offset_Y, input_block_offset_X,
        output_block_offset_Y, output_block_offset_X,
        in_mbar, 0,
        tensor_map_input_ptr,
        rng_seed, rng_subsequence_base);

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, logical_ctaid_Y, M / CHUNK_DIM);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_grouped_rows_pack_rowcol_kernel(
    const IType* __restrict__ input,
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t num_batches,
    const int64_t live_rows_per_batch,
    const int64_t padded_rows_per_batch,
    const int64_t input_cols,
    const int64_t output_cols
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int M = static_cast<int>(num_batches * padded_rows_per_batch);
    const int row_ntk = static_cast<int>(output_cols / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int batch = block_offset_Y / static_cast<int>(padded_rows_per_batch);
    const int row_in_batch = block_offset_Y - batch * static_cast<int>(padded_rows_per_batch);
    const bool full_live_tma =
        batch < static_cast<int>(num_batches) &&
        row_in_batch + CHUNK_DIM <= static_cast<int>(live_rows_per_batch) &&
        block_offset_X + CHUNK_DIM <= static_cast<int>(input_cols);

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    if (full_live_tma) {
        const bool leading = (threadIdx.x == 0);
        const int input_block_offset_Y =
            batch * static_cast<int>(live_rows_per_batch) + row_in_batch;
        __shared__ uint64_t in_mbar[NUM_TILES];
        if (leading) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                mbarrier_init(&in_mbar[t], 1);
            }
            fence_proxy_async_shared_cta();
        }
        __syncthreads();

        #pragma unroll
        for (int pre = 0; pre < 2; ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[pre],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * TILE_DIM,
                    input_block_offset_Y + ty * TILE_DIM,
                    &in_mbar[pre]);
            }
        }

        quantize_chunk_rowcol_pipelined_impl<MODE>(
            sIn_ptr,
            sOut_ptr,
            row_scale_buf,
            col_scale_buf,
            sOut,
            tensor_map_row_output,
            tensor_map_col_output,
            input_block_offset_Y,
            block_offset_X,
            block_offset_Y,
            block_offset_X,
            block_offset_X,
            block_offset_Y,
            in_mbar,
            0,
            &tensor_map_input);

        if (leading) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                mbarrier_invalid(&in_mbar[t]);
            }
        }
    } else {
        load_grouped_rows_chunk_direct(
            sIn_ptr,
            input,
            static_cast<int>(num_batches),
            static_cast<int>(live_rows_per_batch),
            static_cast<int>(padded_rows_per_batch),
            static_cast<int>(input_cols),
            static_cast<int>(output_cols),
            block_offset_Y,
            block_offset_X);

        quantize_chunk_preloaded_rowcol<MODE>(
            sIn_ptr,
            sOut_ptr,
            row_scale_buf,
            col_scale_buf,
            sOut,
            tensor_map_row_output,
            tensor_map_col_output,
            block_offset_Y,
            block_offset_X,
            block_offset_X,
            block_offset_Y);
    }

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_indexed_scaled_rows_pack_rowcol_kernel(
    const IType* __restrict__ input,
    const int64_t* __restrict__ token_indices,
    const float* __restrict__ scores,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t num_batches,
    const int64_t live_rows_per_batch,
    const int64_t padded_rows_per_batch,
    const int64_t input_cols,
    const int64_t output_cols
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int M = static_cast<int>(num_batches * padded_rows_per_batch);
    const int row_ntk = static_cast<int>(output_cols / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_indexed_scaled_grouped_rows_chunk_direct(
        sIn_ptr,
        input,
        token_indices,
        scores,
        static_cast<int>(num_batches),
        static_cast<int>(live_rows_per_batch),
        static_cast<int>(padded_rows_per_batch),
        static_cast<int>(input_cols),
        static_cast<int>(output_cols),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_kernel(
    const IType* __restrict__ input,
    const IType* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    const int64_t* __restrict__ token_indices,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t num_batches,
    const int64_t live_rows_per_batch,
    const int64_t padded_rows_per_batch,
    const int64_t input_cols,
    const int64_t output_cols
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int M = static_cast<int>(num_batches * padded_rows_per_batch);
    const int row_ntk = static_cast<int>(output_cols / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_indexed_rmsnorm_grouped_rows_chunk_direct(
        sIn_ptr,
        input,
        norm_weight,
        inv_rms,
        token_indices,
        static_cast<int>(num_batches),
        static_cast<int>(live_rows_per_batch),
        static_cast<int>(padded_rows_per_batch),
        static_cast<int>(input_cols),
        static_cast<int>(output_cols),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_indexed_scaled_rows_pack_rowcol_variable_kernel(
    const IType* __restrict__ input,
    const int64_t* __restrict__ token_indices,
    const float* __restrict__ scores,
    const int64_t* __restrict__ route_starts,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ padded_starts,
    const int64_t* __restrict__ padded_rows,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t total_padded,
    const int64_t input_cols,
    const int64_t output_cols
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int M = static_cast<int>(total_padded);
    const int row_ntk = static_cast<int>(output_cols / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int expert = blockIdx.z;
    const int tile_row_in_expert = blockIdx.y * CHUNK_DIM;
    if (tile_row_in_expert >= static_cast<int>(padded_rows[expert])) {
        return;
    }
    const int block_offset_Y = static_cast<int>(padded_starts[expert]) + tile_row_in_expert;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int logical_ctaid_Y = block_offset_Y / CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_indexed_scaled_variable_grouped_rows_chunk_direct(
        sIn_ptr,
        input,
        token_indices,
        scores,
        route_starts,
        rows,
        expert,
        tile_row_in_expert,
        static_cast<int>(input_cols),
        static_cast<int>(output_cols),
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, logical_ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, logical_ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_variable_kernel(
    const IType* __restrict__ input,
    const IType* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    const int64_t* __restrict__ token_indices,
    const int64_t* __restrict__ route_starts,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ padded_starts,
    const int64_t* __restrict__ padded_rows,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t total_padded,
    const int64_t input_cols,
    const int64_t output_cols
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int M = static_cast<int>(total_padded);
    const int row_ntk = static_cast<int>(output_cols / CHUNK_DIM);
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int expert = blockIdx.z;
    const int tile_row_in_expert = blockIdx.y * CHUNK_DIM;
    if (tile_row_in_expert >= static_cast<int>(padded_rows[expert])) {
        return;
    }
    const int block_offset_Y = static_cast<int>(padded_starts[expert]) + tile_row_in_expert;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int logical_ctaid_Y = block_offset_Y / CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_indexed_rmsnorm_variable_grouped_rows_chunk_direct(
        sIn_ptr,
        input,
        norm_weight,
        inv_rms,
        token_indices,
        route_starts,
        rows,
        expert,
        tile_row_in_expert,
        static_cast<int>(input_cols),
        static_cast<int>(output_cols),
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, logical_ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, logical_ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_mul_rowcol_kernel(
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ h3,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_silu_mul_chunk_direct(
        sIn_ptr,
        h1_raw,
        h3,
        static_cast<int>(M),
        static_cast<int>(H),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_mul_sigmoid_rowcol_kernel(
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ h3,
    IType* __restrict__ sig_h1,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_silu_mul_chunk_direct_save_sigmoid(
        sIn_ptr,
        h1_raw,
        h3,
        sig_h1,
        static_cast<int>(M),
        static_cast<int>(H),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_rowcol_kernel(
    const IType* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_sqrelu_chunk_direct(
        sIn_ptr,
        h1_raw,
        static_cast<int>(M),
        static_cast<int>(H),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel(
    const IType* __restrict__ dh,
    const IType* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_sqrelu_deriv_chunk_direct(
        sIn_ptr,
        dh,
        h1_raw,
        static_cast<int>(M),
        static_cast<int>(H),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_rowcol_tma_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_sqrelu_chunk_rowcol_pipelined<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel(
    const __grid_constant__ CUtensorMap tensor_map_dh,
    const __grid_constant__ CUtensorMap tensor_map_h1,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sDh_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sH1_ptr = reinterpret_cast<IType*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * in_bytes);
    uint8_t* row_scale_buf = dshmem + 2 * in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sDh = *reinterpret_cast<InputBuf3D*>(sDh_ptr);
    auto& sH1 = *reinterpret_cast<InputBuf3D*>(sH1_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sDh;
    (void)sH1;
    (void)sOut;

    __shared__ uint64_t dh_mbar[NUM_TILES];
    __shared__ uint64_t h1_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&dh_mbar[t], 1);
            mbarrier_init(&h1_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&dh_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sDh[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_dh),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &dh_mbar[pre]);
            mbarrier_arrive_expect_tx(&h1_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sH1[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &h1_mbar[pre]);
        }
    }

    quantize_sqrelu_deriv_chunk_rowcol_pipelined<MODE>(
        sDh_ptr,
        sH1_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        dh_mbar,
        h1_mbar,
        0,
        &tensor_map_dh,
        &tensor_map_h1);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&dh_mbar[t]);
            mbarrier_invalid(&h1_mbar[t]);
        }
    }
#endif
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RHT = false,
    int RHT_BLOCK_SIZE = 16,
    bool WITH_RANDOM_SIGN_MASK = true,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = WITH_RHT>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_rowcol_tma_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_sqrelu_chunk_rowcol_pipelined_opt<
        MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE,
        WITH_RANDOM_SIGN_MASK, ROW_WITH_RHT, COL_WITH_RHT>(
        sIn_ptr, sOut_ptr, row_scale_buf, col_scale_buf, sOut,
        tensor_map_row_output, tensor_map_col_output,
        block_offset_Y, block_offset_X, in_mbar, 0, &tensor_map_input,
        rng_seed,
        rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RHT = false,
    int RHT_BLOCK_SIZE = 16,
    bool WITH_RANDOM_SIGN_MASK = true,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = WITH_RHT>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_dh,
    const __grid_constant__ CUtensorMap tensor_map_h1,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sDh_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sH1_ptr = reinterpret_cast<IType*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * in_bytes);
    uint8_t* row_scale_buf = dshmem + 2 * in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sDh = *reinterpret_cast<InputBuf3D*>(sDh_ptr);
    auto& sH1 = *reinterpret_cast<InputBuf3D*>(sH1_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sDh;
    (void)sH1;
    (void)sOut;

    __shared__ uint64_t dh_mbar[NUM_TILES];
    __shared__ uint64_t h1_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&dh_mbar[t], 1);
            mbarrier_init(&h1_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&dh_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sDh[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_dh),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &dh_mbar[pre]);
            mbarrier_arrive_expect_tx(&h1_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sH1[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &h1_mbar[pre]);
        }
    }

    quantize_sqrelu_deriv_chunk_rowcol_pipelined_opt<
        MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE,
        WITH_RANDOM_SIGN_MASK, ROW_WITH_RHT, COL_WITH_RHT>(
        sDh_ptr, sH1_ptr, sOut_ptr, row_scale_buf, col_scale_buf, sOut,
        tensor_map_row_output, tensor_map_col_output,
        block_offset_Y, block_offset_X, dh_mbar, h1_mbar, 0,
        &tensor_map_dh, &tensor_map_h1,
        rng_seed,
        rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&dh_mbar[t]);
            mbarrier_invalid(&h1_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_mul_rowcol_strided_kernel(
    const IType* __restrict__ h13,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H,
    const int64_t input_stride,
    const int64_t h3_offset
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_silu_mul_chunk_strided(
        sIn_ptr,
        h13,
        static_cast<int>(M),
        static_cast<int>(H),
        static_cast<int>(input_stride),
        static_cast<int>(h3_offset),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool ROW_WITH_RHT,
    bool COL_WITH_RHT,
    int RHT_BLOCK_SIZE = 32,
    bool WITH_RANDOM_SIGN_MASK = false>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_mul_rowcol_kernel_opt(
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ h3,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_silu_mul_chunk_direct(
        sIn_ptr,
        h1_raw,
        h3,
        static_cast<int>(M),
        static_cast<int>(H),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol_opt<
        MODE,
        DATA_SR,
        SCALE_SR,
        ROW_WITH_RHT,
        COL_WITH_RHT,
        RHT_BLOCK_SIZE,
        WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y,
        rng_seed,
        rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool ROW_WITH_RHT,
    bool COL_WITH_RHT,
    int RHT_BLOCK_SIZE = 32,
    bool WITH_RANDOM_SIGN_MASK = false>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_rowcol_kernel_opt(
    const IType* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_sqrelu_chunk_direct(
        sIn_ptr,
        h1_raw,
        static_cast<int>(M),
        static_cast<int>(H),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol_opt<
        MODE,
        DATA_SR,
        SCALE_SR,
        ROW_WITH_RHT,
        COL_WITH_RHT,
        RHT_BLOCK_SIZE,
        WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y,
        rng_seed,
        rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool ROW_WITH_RHT,
    bool COL_WITH_RHT,
    int RHT_BLOCK_SIZE = 32,
    bool WITH_RANDOM_SIGN_MASK = false>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel_opt(
    const IType* __restrict__ dh,
    const IType* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_sqrelu_deriv_chunk_direct(
        sIn_ptr,
        dh,
        h1_raw,
        static_cast<int>(M),
        static_cast<int>(H),
        block_offset_Y,
        block_offset_X);

    quantize_chunk_preloaded_rowcol_opt<
        MODE,
        DATA_SR,
        SCALE_SR,
        ROW_WITH_RHT,
        COL_WITH_RHT,
        RHT_BLOCK_SIZE,
        WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        block_offset_X,
        block_offset_Y,
        rng_seed,
        rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel(
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_row_output0,
    uint8_t* __restrict__ row_scales_out0,
    const __grid_constant__ CUtensorMap tensor_map_col_output0,
    uint8_t* __restrict__ col_scales_out0,
    const __grid_constant__ CUtensorMap tensor_map_row_output1,
    uint8_t* __restrict__ row_scales_out1,
    const __grid_constant__ CUtensorMap tensor_map_col_output1,
    uint8_t* __restrict__ col_scales_out1,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int row_ntk = H / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;

    constexpr int tile_in_bytes =
        DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn0_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem + tile_in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + (2 * tile_in_bytes));
    uint8_t* row_scale_buf0 = dshmem + (2 * tile_in_bytes) + out_bytes;
    uint8_t* row_scale_buf1 = row_scale_buf0 + sc_bytes;
    uint8_t* col_scale_buf0 = row_scale_buf1 + sc_bytes;
    uint8_t* col_scale_buf1 = col_scale_buf0 + sc_bytes;

    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);
    (void)sOut;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int tile_offset_Y = stage_Y * TILE_DIM;
        const int tile_offset_X = stage_X * TILE_DIM;

        load_silu_deriv_tile_direct_and_rowwise_quant<MODE>(
            sIn0_ptr,
            sIn1_ptr,
            sOut_ptr,
            row_scale_buf0,
            row_scale_buf1,
            dh,
            h3,
            h1_raw,
            static_cast<int>(H),
            block_offset_Y + tile_offset_Y,
            block_offset_X + tile_offset_X,
            stage_Y,
            stage_X);

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output0),
                block_offset_X + tile_offset_X,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[0]));
            cp_async_bulk_commit_group();
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output1),
                block_offset_X + tile_offset_X,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[1]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn0_ptr, sOut_ptr, col_scale_buf0, stage_X, stage_Y, 2);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output0),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X,
                reinterpret_cast<uint64_t*>(&sOut[2]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn1_ptr, sOut_ptr, col_scale_buf1, stage_X, stage_Y, 3);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output1),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X,
                reinterpret_cast<uint64_t*>(&sOut[3]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }

    write_scales_swizzled(row_scale_buf0, row_scales_out0, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(row_scale_buf1, row_scales_out1, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf0, col_scales_out0, ctaid_Y, ctaid_X, col_ntk);
    write_scales_swizzled(col_scale_buf1, col_scales_out1, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel(
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output0,
    uint8_t* __restrict__ col_scales_out0,
    const __grid_constant__ CUtensorMap tensor_map_col_output1,
    uint8_t* __restrict__ col_scales_out1,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int row_ntk_single = H / CHUNK_DIM;
    const int row_ntk_total = (2 * H) / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;

    constexpr int tile_in_bytes =
        DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn0_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem + tile_in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + (2 * tile_in_bytes));
    uint8_t* row_scale_buf0 = dshmem + (2 * tile_in_bytes) + out_bytes;
    uint8_t* row_scale_buf1 = row_scale_buf0 + sc_bytes;
    uint8_t* col_scale_buf0 = row_scale_buf1 + sc_bytes;
    uint8_t* col_scale_buf1 = col_scale_buf0 + sc_bytes;

    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);
    (void)sOut;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int tile_offset_Y = stage_Y * TILE_DIM;
        const int tile_offset_X = stage_X * TILE_DIM;

        load_silu_deriv_tile_direct_and_rowwise_quant<MODE>(
            sIn0_ptr,
            sIn1_ptr,
            sOut_ptr,
            row_scale_buf0,
            row_scale_buf1,
            dh,
            h3,
            h1_raw,
            static_cast<int>(H),
            block_offset_Y + tile_offset_Y,
            block_offset_X + tile_offset_X,
            stage_Y,
            stage_X);

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[0]));
            cp_async_bulk_commit_group();
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X + H,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[1]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn0_ptr, sOut_ptr, col_scale_buf0, stage_X, stage_Y, 2);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output0),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X,
                reinterpret_cast<uint64_t*>(&sOut[2]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn1_ptr, sOut_ptr, col_scale_buf1, stage_X, stage_Y, 3);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output1),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X,
                reinterpret_cast<uint64_t*>(&sOut[3]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }

    write_scales_swizzled(row_scale_buf0, row_scales_out, ctaid_X, ctaid_Y, row_ntk_total);
    write_scales_swizzled(row_scale_buf1, row_scales_out, ctaid_X + row_ntk_single, ctaid_Y, row_ntk_total);
    write_scales_swizzled(col_scale_buf0, col_scales_out0, ctaid_Y, ctaid_X, col_ntk);
    write_scales_swizzled(col_scale_buf1, col_scales_out1, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_deriv_from_sigmoid_split2_rowcol_splitcols_kernel(
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ sig_h1,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output0,
    uint8_t* __restrict__ col_scales_out0,
    const __grid_constant__ CUtensorMap tensor_map_col_output1,
    uint8_t* __restrict__ col_scales_out1,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int row_ntk_single = H / CHUNK_DIM;
    const int row_ntk_total = (2 * H) / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;

    constexpr int tile_in_bytes =
        DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn0_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem + tile_in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + (2 * tile_in_bytes));
    uint8_t* row_scale_buf0 = dshmem + (2 * tile_in_bytes) + out_bytes;
    uint8_t* row_scale_buf1 = row_scale_buf0 + sc_bytes;
    uint8_t* col_scale_buf0 = row_scale_buf1 + sc_bytes;
    uint8_t* col_scale_buf1 = col_scale_buf0 + sc_bytes;

    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);
    (void)sOut;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int tile_offset_Y = stage_Y * TILE_DIM;
        const int tile_offset_X = stage_X * TILE_DIM;

        load_silu_deriv_from_sigmoid_tile_direct_and_rowwise_quant<MODE>(
            sIn0_ptr,
            sIn1_ptr,
            sOut_ptr,
            row_scale_buf0,
            row_scale_buf1,
            dh,
            h3,
            h1_raw,
            sig_h1,
            static_cast<int>(H),
            block_offset_Y + tile_offset_Y,
            block_offset_X + tile_offset_X,
            stage_Y,
            stage_X);

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[0]));
            cp_async_bulk_commit_group();
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X + H,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[1]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn0_ptr, sOut_ptr, col_scale_buf0, stage_X, stage_Y, 2);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output0),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X,
                reinterpret_cast<uint64_t*>(&sOut[2]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn1_ptr, sOut_ptr, col_scale_buf1, stage_X, stage_Y, 3);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output1),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X,
                reinterpret_cast<uint64_t*>(&sOut[3]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }

    write_scales_swizzled(row_scale_buf0, row_scales_out, ctaid_X, ctaid_Y, row_ntk_total);
    write_scales_swizzled(row_scale_buf1, row_scales_out, ctaid_X + row_ntk_single, ctaid_Y, row_ntk_total);
    write_scales_swizzled(col_scale_buf0, col_scales_out0, ctaid_Y, ctaid_X, col_ntk);
    write_scales_swizzled(col_scale_buf1, col_scales_out1, ctaid_Y, ctaid_X, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_deriv_split2_rowcol_strided_combined_kernel(
    const IType* __restrict__ dh,
    const IType* __restrict__ h13,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t H,
    const int64_t dh_stride,
    const int64_t h13_stride,
    const int64_t h3_offset
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int row_ntk_single = H / CHUNK_DIM;
    const int row_ntk_total = (2 * H) / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;

    constexpr int tile_in_bytes =
        DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn0_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem + tile_in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + (2 * tile_in_bytes));
    uint8_t* row_scale_buf0 = dshmem + (2 * tile_in_bytes) + out_bytes;
    uint8_t* row_scale_buf1 = row_scale_buf0 + sc_bytes;
    uint8_t* col_scale_buf0 = row_scale_buf1 + sc_bytes;
    uint8_t* col_scale_buf1 = col_scale_buf0 + sc_bytes;

    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);
    (void)sOut;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int tile_offset_Y = stage_Y * TILE_DIM;
        const int tile_offset_X = stage_X * TILE_DIM;

        load_silu_deriv_tile_strided_and_rowwise_quant<MODE>(
            sIn0_ptr,
            sIn1_ptr,
            sOut_ptr,
            row_scale_buf0,
            row_scale_buf1,
            dh,
            h13,
            static_cast<int>(dh_stride),
            static_cast<int>(h13_stride),
            static_cast<int>(h3_offset),
            block_offset_Y + tile_offset_Y,
            block_offset_X + tile_offset_X,
            stage_Y,
            stage_X);

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[0]));
            cp_async_bulk_commit_group();
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X + H,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[1]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn0_ptr, sOut_ptr, col_scale_buf0, stage_X, stage_Y, 2);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X,
                reinterpret_cast<uint64_t*>(&sOut[2]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct_tile<MODE>(
            sIn1_ptr, sOut_ptr, col_scale_buf1, stage_X, stage_Y, 3);
        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + tile_offset_Y,
                block_offset_X + tile_offset_X + H,
                reinterpret_cast<uint64_t*>(&sOut[3]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }

    write_scales_swizzled(row_scale_buf0, row_scales_out, ctaid_X, ctaid_Y, row_ntk_total);
    write_scales_swizzled(row_scale_buf1, row_scales_out, ctaid_X + row_ntk_single, ctaid_Y, row_ntk_total);
    write_scales_swizzled(col_scale_buf0, col_scales_out, ctaid_Y, ctaid_X, col_ntk);
    write_scales_swizzled(col_scale_buf1, col_scales_out, ctaid_Y, ctaid_X + row_ntk_single, col_ntk);
#endif
}

template<QuantMode MODE = QuantMode::RTE, bool USE_SAVED_SIGMOID = false>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel(
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    const IType* __restrict__ sig_h1,
    IType* __restrict__ dh1_out,
    IType* __restrict__ dh3_out,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int row_ntk_single = H / CHUNK_DIM;
    const int row_ntk_total = (2 * H) / CHUNK_DIM;

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn0_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + (2 * in_bytes));
    uint8_t* row_scale_buf = dshmem + (2 * in_bytes) + out_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sOut;

    load_silu_deriv_chunk_direct_row_bf16<USE_SAVED_SIGMOID>(
        sIn0_ptr, sIn1_ptr, dh1_out, dh3_out,
        dh, h3, h1_raw, sig_h1,
        static_cast<int>(M), static_cast<int>(H),
        block_offset_Y, block_offset_X);

    quantize_chunk_preloaded_row<MODE>(
        sIn0_ptr, sOut_ptr, row_scale_buf, sOut,
        tensor_map_row_output,
        block_offset_Y, block_offset_X);
    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk_total);

    quantize_chunk_preloaded_row<MODE>(
        sIn1_ptr, sOut_ptr, row_scale_buf, sOut,
        tensor_map_row_output,
        block_offset_Y, block_offset_X + H);
    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X + row_ntk_single, ctaid_Y, row_ntk_total);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_silu_deriv_split2_row_bf16_tile_kernel(
    const IType* __restrict__ dh,
    const IType* __restrict__ h3,
    const IType* __restrict__ h1_raw,
    IType* __restrict__ dh1_out,
    IType* __restrict__ dh3_out,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    uint8_t* __restrict__ row_scales_out,
    const int64_t M,
    const int64_t H
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;
    const int row_ntk_single = H / CHUNK_DIM;
    const int row_ntk_total = (2 * H) / CHUNK_DIM;

    constexpr int tile_in_bytes =
        DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn0_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem + tile_in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + (2 * tile_in_bytes));
    uint8_t* row_scale_buf0 = dshmem + (2 * tile_in_bytes) + out_bytes;
    uint8_t* row_scale_buf1 = row_scale_buf0 + sc_bytes;

    auto& sOut = *reinterpret_cast<Split2OutputBuf4D*>(sOut_ptr);
    (void)sOut;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int tile_offset_Y = stage_Y * TILE_DIM;
        const int tile_offset_X = stage_X * TILE_DIM;

        load_silu_deriv_tile_direct_and_rowwise_quant<MODE, true>(
            sIn0_ptr,
            sIn1_ptr,
            sOut_ptr,
            row_scale_buf0,
            row_scale_buf1,
            dh,
            h3,
            h1_raw,
            static_cast<int>(H),
            block_offset_Y + tile_offset_Y,
            block_offset_X + tile_offset_X,
            stage_Y,
            stage_X,
            dh1_out,
            dh3_out);

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[0]));
            cp_async_bulk_commit_group();
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + tile_offset_X + H,
                block_offset_Y + tile_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[1]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }

    write_scales_swizzled(row_scale_buf0, row_scales_out, ctaid_X, ctaid_Y, row_ntk_total);
    write_scales_swizzled(row_scale_buf1, row_scales_out, ctaid_X + row_ntk_single, ctaid_Y, row_ntk_total);
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split3_row_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const int64_t K2
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk0 = K0 / CHUNK_DIM;
    const int ntk1 = K1 / CHUNK_DIM;
    const int ntk2 = K2 / CHUNK_DIM;
    const int ntk = ntk0 + ntk1 + ntk2;
    const int logical_ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int output_block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int output_block_offset_X = logical_ctaid_X * CHUNK_DIM;

    int local_ctaid_X = logical_ctaid_X;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_X < ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
        local_ctaid_X = logical_ctaid_X;
    } else if (logical_ctaid_X < ntk0 + ntk1) {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_X = logical_ctaid_X - ntk0;
    } else {
        tensor_map_input_ptr = &tensor_map_input2;
        local_ctaid_X = logical_ctaid_X - ntk0 - ntk1;
    }
    const int input_block_offset_Y = output_block_offset_Y;
    const int input_block_offset_X = local_ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_X + tx * TILE_DIM,
                input_block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_pipelined_split_input<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        input_block_offset_Y, input_block_offset_X,
        output_block_offset_Y, output_block_offset_X,
        in_mbar, 0,
        tensor_map_input_ptr);

    write_scales_swizzled(scale_buf, scales_out, logical_ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Write scales to per-group buffers (grouped quantize)
//
// Determines which group this chunk belongs to from split_range[],
// computes local row offset within that group, then writes scales
// to the group's scale buffer in swizzled layout.
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ void write_scales_grouped(
    const uint8_t* scale_buf,   // [CHUNK_DIM × SCALES_PER_CHUNK]
    const GroupArgs& args,
    int block_offset_Y,         // global row offset of this chunk
    int ctaid_X,
    int ntk                     // K / CHUNK_DIM
) {
    // Determine group ID from row position
    int group_id = 0;
    for (int g = 1; g < args.num_groups; ++g) {
        if (block_offset_Y >= args.boundaries[g]) group_id = g;
    }
    const int group_start = args.boundaries[group_id];
    uint8_t* __restrict__ group_scales = args.scale_ptrs[group_id];

    // Local ctaid_Y within this group
    const int local_ctaid_Y = (block_offset_Y - group_start) / CHUNK_DIM;

    for (int row = threadIdx.x; row < CHUNK_DIM; row += THREADS) {
        const int j = row % 32;
        const int grp = row / 32;
        const int base = (local_ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
        uint32_t pk;
        uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
        p[0] = scale_buf[row * SCALES_PER_CHUNK + 0];
        p[1] = scale_buf[row * SCALES_PER_CHUNK + 1];
        p[2] = scale_buf[row * SCALES_PER_CHUNK + 2];
        p[3] = scale_buf[row * SCALES_PER_CHUNK + 3];
        *reinterpret_cast<uint32_t*>(group_scales + base) = pk;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Grouped fused kernel — single kernel launch for all groups
//
// Identical to mxfp4_v3_fused_kernel but uses GroupArgs to write
// scales to per-group buffers. FP4 output is written to a single
// contiguous buffer (no per-group FP4 needed for MXFP4).
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_fused_group_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const int64_t M, const int64_t K,
    GroupArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Prefetch first 2 tiles
    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    // Pipelined quantize
    quantize_chunk_pipelined<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        block_offset_Y, block_offset_X,
        in_mbar, 0,
        &tensor_map_input);

    // Write scales to per-group buffers
    write_scales_grouped(scale_buf, args, block_offset_Y, ctaid_X, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE, bool DATA_SR = false, bool SCALE_SR = false,
         bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_fused_group_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const int64_t M, const int64_t K,
    GroupArgs args,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_pipelined_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        block_offset_Y, block_offset_X,
        in_mbar, 0,
        &tensor_map_input,
        rng_seed,
        rng_subsequence_base);

    write_scales_grouped(scale_buf, args, block_offset_Y, ctaid_X, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Col-only fused kernel for mxfp4_v4.
//
// Emits the MXFP4 quantization of input^T directly without materializing a
// global BF16 transpose tensor.
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_col_only_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = M / CHUNK_DIM;   // logical K-dimension of input^T is original M
    const int ctaid_X = blockIdx.x;  // original row chunk
    const int ctaid_Y = blockIdx.y;  // original col chunk
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_Y + ty * TILE_DIM,
                block_offset_X + tx * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_transposed_pipelined<MODE>(
        sIn_ptr,
        sOut_ptr,
        scale_buf,
        sOut,
        tensor_map_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input);

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_col_only_fused_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_Y + ty * TILE_DIM,
                block_offset_X + tx * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_transposed_pipelined_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        scale_buf,
        sOut,
        tensor_map_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input,
        rng_seed,
        rng_subsequence_base + static_cast<uint64_t>(ctaid_Y * gridDim.x + ctaid_X) * NUM_TILES * THREADS);

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_split3_col_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M,
    const int64_t K0,
    const int64_t K1,
    const int64_t K2
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk0 = K0 / CHUNK_DIM;
    const int ntk1 = K1 / CHUNK_DIM;
    const int ntk2 = K2 / CHUNK_DIM;
    const int ntk = ntk0 + ntk1 + ntk2;
    const int ctaid_X = blockIdx.x;
    const int logical_ctaid_Y = blockIdx.y;
    const int output_block_offset_Y = logical_ctaid_Y * CHUNK_DIM;
    const int output_block_offset_X = ctaid_X * CHUNK_DIM;

    int local_ctaid_Y = logical_ctaid_Y;
    const CUtensorMap* tensor_map_input_ptr = &tensor_map_input0;
    if (logical_ctaid_Y < ntk0) {
        tensor_map_input_ptr = &tensor_map_input0;
        local_ctaid_Y = logical_ctaid_Y;
    } else if (logical_ctaid_Y < ntk0 + ntk1) {
        tensor_map_input_ptr = &tensor_map_input1;
        local_ctaid_Y = logical_ctaid_Y - ntk0;
    } else {
        tensor_map_input_ptr = &tensor_map_input2;
        local_ctaid_Y = logical_ctaid_Y - ntk0 - ntk1;
    }
    const int input_block_offset_Y = local_ctaid_Y * CHUNK_DIM;
    const int input_block_offset_X = output_block_offset_X;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                input_block_offset_Y + ty * TILE_DIM,
                input_block_offset_X + tx * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    quantize_chunk_transposed_pipelined_split_input<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        input_block_offset_Y, input_block_offset_X,
        output_block_offset_Y, output_block_offset_X,
        in_mbar, 0,
        tensor_map_input_ptr);

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, logical_ctaid_Y, M / CHUNK_DIM);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_rmsnorm_rowcol_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    const __grid_constant__ CUtensorMap tensor_map_norm_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const float* __restrict__ inv_rms,
    const IType* __restrict__ gamma,
    const int64_t M,
    const int64_t K,
    const bool write_norm_output
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = K / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;  // original K chunk
    const int ctaid_Y = blockIdx.y;  // original M chunk
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;
    IType* gamma_cache = reinterpret_cast<IType*>(col_scale_buf + sc_bytes);

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    for (int i = threadIdx.x; i < CHUNK_DIM; i += THREADS) {
        gamma_cache[i] = gamma[block_offset_X + i];
    }
    __syncthreads();

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre], TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    fused_rmsnorm_quantize_chunk_rowcol_pipelined<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input,
        inv_rms + block_offset_Y,
        gamma_cache,
        write_norm_output ? &tensor_map_norm_output : nullptr);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_rmsnorm_rowcol_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const float* __restrict__ inv_rms,
    const IType* __restrict__ gamma,
    const int64_t M,
    const int64_t K,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = K / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;
    IType* gamma_cache = reinterpret_cast<IType*>(col_scale_buf + sc_bytes);

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;
    (void)sOut;

    for (int i = threadIdx.x; i < CHUNK_DIM; i += THREADS) {
        gamma_cache[i] = gamma[block_offset_X + i];
    }
    __syncthreads();

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre], TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    fused_rmsnorm_quantize_chunk_rowcol_pipelined_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input,
        inv_rms + block_offset_Y,
        gamma_cache,
        rng_seed,
        rng_subsequence_base);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_rmsnorm_row_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const float* __restrict__ inv_rms,
    const IType* __restrict__ gamma,
    const int64_t M,
    const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* scale_buf = dshmem + in_bytes + out_bytes;
    IType* gamma_cache = reinterpret_cast<IType*>(dshmem + in_bytes + out_bytes + sc_bytes);

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    for (int i = threadIdx.x; i < CHUNK_DIM; i += THREADS) {
        gamma_cache[i] = gamma[block_offset_X + i];
    }
    __syncthreads();

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre], TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    fused_rmsnorm_quantize_chunk_pipelined<MODE>(
        sIn_ptr,
        sOut_ptr,
        scale_buf,
        sOut,
        tensor_map_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input,
        inv_rms + block_offset_Y,
        gamma_cache);

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_fused_rmsnorm_col_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const float* __restrict__ inv_rms,
    const IType* __restrict__ gamma,
    const int64_t M,
    const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* scale_buf = dshmem + in_bytes + out_bytes;
    IType* gamma_cache = reinterpret_cast<IType*>(dshmem + in_bytes + out_bytes + sc_bytes);

    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);
    (void)sIn;

    for (int i = threadIdx.x; i < CHUNK_DIM; i += THREADS) {
        gamma_cache[i] = gamma[block_offset_Y + i];
    }
    __syncthreads();

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X;
        const int tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre], TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_Y + ty * TILE_DIM,
                block_offset_X + tx * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    fused_rmsnorm_quantize_chunk_transposed_pipelined<MODE>(
        sIn_ptr,
        sOut_ptr,
        scale_buf,
        sOut,
        tensor_map_output,
        block_offset_Y,
        block_offset_X,
        in_mbar,
        0,
        &tensor_map_input,
        inv_rms + block_offset_X,
        gamma_cache);

    write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// SMEM size helper
// ═══════════════════════════════════════════════════════════════════
inline int v3_shmem_size() {
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    return in_bytes + out_bytes + sc_bytes + TMA_SHMEM_ALIGNMENT;
}

inline int v4_col_only_shmem_size() {
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    return in_bytes + out_bytes + sc_bytes + TMA_SHMEM_ALIGNMENT;
}

inline int v4_rowcol_shmem_size() {
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    return in_bytes + out_bytes + (2 * sc_bytes) + TMA_SHMEM_ALIGNMENT;
}

inline int v4_dual_input_rowcol_shmem_size() {
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    return (2 * in_bytes) + out_bytes + (2 * sc_bytes) + TMA_SHMEM_ALIGNMENT;
}

inline int v4_fused_rmsnorm_shmem_size() {
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(CHUNK_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    return in_bytes + out_bytes + (2 * sc_bytes) + gamma_bytes + TMA_SHMEM_ALIGNMENT;
}

} // namespace mxfp4_v3
