#include "../../ThunderKittens/include/kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mxfp4_v3_quantize.cuh"

namespace tcbench {
using namespace mxfp4_v3;

static constexpr int TILE_M = 64;
static constexpr int TILE_N32 = 32;
static constexpr int TILE_K32 = 32;
static constexpr int TILE_N16 = 16;
static constexpr int TILE_K16 = 16;
static constexpr int CHUNK_M = 128;
static constexpr int CHUNK_K = 128;
static constexpr int NUM_WARPS = 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
static constexpr int NUM_WARPS_2WG = 8;
static constexpr int NUM_THREADS_2WG = NUM_WARPS_2WG * WARP_THREADS;

using a32_tile = st_bf<TILE_M, TILE_K32>;
using b32_tile = st_bf<TILE_K32, TILE_N32>;
using d32_tile = st_bf<TILE_M, TILE_N32>;
using d32_tt   = half_tt_fl<TILE_N32>;
using d32_rt   = rt_bf<TILE_M / 4, TILE_N32>;
using a128_tile = st_bf<CHUNK_M, CHUNK_K>;

using a16_tile = st_bf<TILE_M, TILE_K16>;
using b16_tile = st_bf<TILE_K16, TILE_N16>;
using d16_tile = st_bf<TILE_M, TILE_N16>;
using d16_tt   = half_tt_fl<TILE_N16>;
using d16_rt   = rt_bf<TILE_M / 4, TILE_N16>;

using a_gl = gl<bf16, 1, 1, -1, -1, a32_tile>;
using a128_gl = gl<bf16, 1, 1, -1, -1, a128_tile>;
using h32_gl = gl<bf16, 1, 1, 32, 32, b32_tile>;
using h16_gl = gl<bf16, 1, 1, 16, 16, b16_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d32_tile>;

enum class ImplKind : int {
    Direct32 = 0,
    Recursive16 = 1,
};

template<QuantMode MODE>
__device__ __forceinline__ uint8_t float_to_e8m0_dispatch(float val) {
    return float_to_e8m0<MODE>(val);
}

template<bool SCALE_SR>
__device__ __forceinline__ uint8_t quantize_scale(
    float block_amax,
    RNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if constexpr (SCALE_SR) {
        return float_to_e8m0_stochastic(block_amax, next_rbits(rng, random_uint4, rnd_idx));
    } else {
        return float_to_e8m0_dispatch<QuantMode::RTE>(block_amax);
    }
}

template<bool DATA_SR, bool SCALE_SR>
__device__ __forceinline__ void quantize_tile_64x32(
    const d32_tile& tile,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int cols,
    int tile_row,
    int tile_col_block,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    const int tid = threadIdx.x & 127;
    const int row = tid >> 1;
    const int half = tid & 1;
    if (row >= TILE_M) {
        return;
    }

    float vals[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __bfloat162float(tile[{row, half * 16 + i}]);
    }

    float my_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        my_amax = fmaxf(my_amax, fabsf(vals[i]));
    }
    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);

    RNGState rng;
    const uint64_t block_linear = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    rng.init(rng_seed, rng_subsequence_base + block_linear * TILE_M + row, 0);
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;

    const uint8_t e8m0_val = quantize_scale<SCALE_SR>(block_amax, rng, random_uint4, rnd_idx);
    if (half == 0) {
        scales_out[(tile_row + row) * (cols / 32) + tile_col_block] = e8m0_val;
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);

    uint16_t packed16[4];
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const float2 in01 = make_float2(vals[pack * 4 + 0], vals[pack * 4 + 1]);
        const float2 in23 = make_float2(vals[pack * 4 + 2], vals[pack * 4 + 3]);
        const auto packed = mul_cvt_fp32_to_fp4_4x<DATA_SR>(
            in01,
            in23,
            scale,
            next_rbits(rng, random_uint4, rnd_idx)
        );
        packed16[pack] = *reinterpret_cast<const uint16_t*>(&packed);
    }

    uint8_t* row_ptr = fp4_out + static_cast<int64_t>(tile_row + row) * (cols / 2) + tile_col_block * 16 + half * 8;
    *reinterpret_cast<uint16_t*>(row_ptr + 0) = packed16[0];
    *reinterpret_cast<uint16_t*>(row_ptr + 2) = packed16[1];
    *reinterpret_cast<uint16_t*>(row_ptr + 4) = packed16[2];
    *reinterpret_cast<uint16_t*>(row_ptr + 6) = packed16[3];
}

template<bool WITH_RANDOM_SIGN_MASK, typename Tile>
__device__ __forceinline__ void apply_sign_mask_64x32(
    Tile& tile,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    const int tid = threadIdx.x & 127;
    const int row = tid >> 1;
    const int half = tid & 1;
    if (row >= TILE_M) {
        return;
    }

    RNGState rng;
    const uint64_t block_linear = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    rng.init(rng_seed, rng_subsequence_base + block_linear * TILE_M + row, 0);
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;
    const uint32_t sign_bits = make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = half * 16 + i;
        float v = __bfloat162float(tile[{row, col}]);
        v *= ((sign_bits >> col) & 1u) ? 1.0f : -1.0f;
        tile[{row, col}] = __float2bfloat16_rn(v);
    }
}

__device__ __forceinline__ void merge_recursive_outputs(
    const d16_tile& left,
    const d16_tile& right,
    d32_tile& out
);

__device__ __forceinline__ void store_tile_64x32_bf16(
    const d32_tile& tile,
    __nv_bfloat16* __restrict__ out,
    int cols,
    int tile_row,
    int tile_col
) {
    const int tid = threadIdx.x & 127;
    const int row = tid >> 1;
    const int half = tid & 1;
    if (row >= TILE_M) {
        return;
    }
    __nv_bfloat16* row_ptr = out + static_cast<int64_t>(tile_row + row) * cols + tile_col + half * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        row_ptr[i] = tile[{row, half * 16 + i}];
    }
}

template<typename SrcTile>
__device__ __forceinline__ void combine_recursive_inputs(
    const SrcTile& src,
    a16_tile& left,
    a16_tile& right
) {
    const int local_tid = threadIdx.x & 127;
    if (local_tid < TILE_M) {
        const int row = local_tid;
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            const float a = __bfloat162float(src[{row, col}]);
            const float b = __bfloat162float(src[{row, 16 + col}]);
            left[{row, col}] = __float2bfloat16_rn(a + b);
            right[{row, col}] = __float2bfloat16_rn(a - b);
        }
    }
}

template<bool WITH_RANDOM_SIGN_MASK, typename ATile>
__device__ __forceinline__ void do_direct32_block_hadonly(
    ATile& a_tile,
    const b32_tile& h_smem,
    d32_tile& d_smem,
    __nv_bfloat16* __restrict__ out_bf16,
    tensor_allocator<1, 1>& tm_alloc,
    semaphore& compute_done,
    int group_barrier_id,
    int accum_superlane,
    int accum_col_offset,
    int cols,
    int tile_row,
    int tile_col,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_tile, rng_seed, rng_subsequence_base);
    __syncthreads();
    if ((threadIdx.x & 127) == 0) {
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    d32_tt accum;
    if (warpgroup::laneid() == 0) {
        accum = tm_alloc.template allocate<d32_tt>(accum_superlane, accum_col_offset);
        mm_AB(accum, a_tile, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d32_rt d_reg;
    warpgroup::load_async(d_reg, accum);
    tensor_load_wait();
    warpgroup::sync(group_barrier_id);
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(group_barrier_id);
    __syncthreads();

    store_tile_64x32_bf16(d_smem, out_bf16, cols, tile_row, tile_col);
    __syncthreads();
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, typename ATile>
__device__ __forceinline__ void do_direct32_block(
    ATile& a_tile,
    const b32_tile& h_smem,
    d32_tile& d_smem,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    tensor_allocator<1, 1>& tm_alloc,
    semaphore& compute_done,
    int group_barrier_id,
    int accum_superlane,
    int accum_col_offset,
    int cols,
    int tile_row,
    int tile_col_block,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_tile, rng_seed, rng_subsequence_base);
    __syncthreads();
    if ((threadIdx.x & 127) == 0) {
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    d32_tt accum;
    if (warpgroup::laneid() == 0) {
        accum = tm_alloc.template allocate<d32_tt>(accum_superlane, accum_col_offset);
        mm_AB(accum, a_tile, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d32_rt d_reg;
    warpgroup::load_async(d_reg, accum);
    tensor_load_wait();
    warpgroup::sync(group_barrier_id);
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(group_barrier_id);
    __syncthreads();

    quantize_tile_64x32<DATA_SR, SCALE_SR>(
        d_smem,
        fp4_out,
        scales_out,
        cols,
        tile_row,
        tile_col_block,
        rng_seed,
        rng_subsequence_base
    );
    __syncthreads();
}

template<bool WITH_RANDOM_SIGN_MASK, typename ATile>
__device__ __forceinline__ void do_recursive16_block_hadonly(
    ATile& a_tile,
    const b16_tile& h_smem,
    a16_tile& tmp0_smem,
    a16_tile& tmp1_smem,
    d16_tile& d0_smem,
    d16_tile& d1_smem,
    d32_tile& d_smem,
    __nv_bfloat16* __restrict__ out_bf16,
    tensor_allocator<1, 1>& tm_alloc,
    semaphore& compute_done,
    int group_barrier_id,
    int cols,
    int tile_row,
    int tile_col,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_tile, rng_seed, rng_subsequence_base);
    __syncthreads();
    combine_recursive_inputs(a_tile, tmp0_smem, tmp1_smem);
    __syncthreads();
    if (threadIdx.x == 0) {
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    d16_tt accum0, accum1;
    if (warpgroup::laneid() == 0) {
        accum0 = tm_alloc.template allocate<d16_tt>(0, 0);
        accum1 = tm_alloc.template allocate<d16_tt>(0, TILE_N16);
        mm_AB(accum0, tmp0_smem, h_smem);
        mm_AB(accum1, tmp1_smem, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d16_rt d0_reg, d1_reg;
    warpgroup::load_async(d0_reg, accum0);
    warpgroup::load_async(d1_reg, accum1);
    tensor_load_wait();
    warpgroup::sync(group_barrier_id);
    warpgroup::store(d0_smem, d0_reg);
    warpgroup::store(d1_smem, d1_reg);
    warpgroup::sync(group_barrier_id);
    __syncthreads();

    merge_recursive_outputs(d0_smem, d1_smem, d_smem);
    __syncthreads();

    store_tile_64x32_bf16(d_smem, out_bf16, cols, tile_row, tile_col);
    __syncthreads();
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, typename ATile>
__device__ __forceinline__ void do_recursive16_block(
    ATile& a_tile,
    const b16_tile& h_smem,
    a16_tile& tmp0_smem,
    a16_tile& tmp1_smem,
    d16_tile& d0_smem,
    d16_tile& d1_smem,
    d32_tile& d_smem,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    tensor_allocator<1, 1>& tm_alloc,
    semaphore& compute_done,
    int group_barrier_id,
    int cols,
    int tile_row,
    int tile_col_block,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_tile, rng_seed, rng_subsequence_base);
    __syncthreads();
    combine_recursive_inputs(a_tile, tmp0_smem, tmp1_smem);
    __syncthreads();
    if (threadIdx.x == 0) {
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    d16_tt accum0, accum1;
    if (warpgroup::laneid() == 0) {
        accum0 = tm_alloc.template allocate<d16_tt>(0, 0);
        accum1 = tm_alloc.template allocate<d16_tt>(0, TILE_N16);
        mm_AB(accum0, tmp0_smem, h_smem);
        mm_AB(accum1, tmp1_smem, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d16_rt d0_reg, d1_reg;
    warpgroup::load_async(d0_reg, accum0);
    warpgroup::load_async(d1_reg, accum1);
    tensor_load_wait();
    warpgroup::sync(group_barrier_id);
    warpgroup::store(d0_smem, d0_reg);
    warpgroup::store(d1_smem, d1_reg);
    warpgroup::sync(group_barrier_id);
    __syncthreads();

    merge_recursive_outputs(d0_smem, d1_smem, d_smem);
    __syncthreads();

    quantize_tile_64x32<DATA_SR, SCALE_SR>(
        d_smem,
        fp4_out,
        scales_out,
        cols,
        tile_row,
        tile_col_block,
        rng_seed,
        rng_subsequence_base
    );
    __syncthreads();
}

template<bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard128_direct_bf16_kernel(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<b32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a128_tile (&a_smem) = al.allocate<a128_tile>();
    b32_tile (&h_smem) = al.allocate<b32_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a128_tile) + sizeof(b32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    const int tile_row_base = bid_m * CHUNK_M;
    const int tile_col_base = bid_k * CHUNK_K;
    int block_idx = 0;
    #pragma unroll
    for (int sy = 0; sy < 2; ++sy) {
        #pragma unroll
        for (int sx = 0; sx < 4; ++sx, ++block_idx) {
            auto a_sub = a_smem.template subtile<64, 32>({sy, sx});
            do_direct32_block_hadonly<WITH_RANDOM_SIGN_MASK>(
                a_sub,
                h_smem,
                d_smem,
                out_bf16,
                tm_alloc,
                compute_done,
                1,
                0,
                0,
                cols,
                tile_row_base + sy * 64,
                tile_col_base + sx * 32,
                rng_seed,
                rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * 8 + block_idx) * TILE_M
            );
        }
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard128_direct_quant_kernel(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<b32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a128_tile (&a_smem) = al.allocate<a128_tile>();
    b32_tile (&h_smem) = al.allocate<b32_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a128_tile) + sizeof(b32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    const int tile_row_base = bid_m * CHUNK_M;
    const int tile_col_base = bid_k * 4;
    int block_idx = 0;
    #pragma unroll
    for (int sy = 0; sy < 2; ++sy) {
        #pragma unroll
        for (int sx = 0; sx < 4; ++sx, ++block_idx) {
            auto a_sub = a_smem.template subtile<64, 32>({sy, sx});
            do_direct32_block<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK>(
                a_sub,
                h_smem,
                d_smem,
                fp4_out,
                scales_out,
                tm_alloc,
                compute_done,
                1,
                0,
                0,
                cols,
                tile_row_base + sy * 64,
                tile_col_base + sx,
                rng_seed,
                rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * 8 + block_idx) * TILE_M
            );
        }
    }
}

template<bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS_2WG, 1)
void hadamard128_direct_bf16_2wg_kernel(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<b32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    const int gid = warpgroupid();
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a128_tile (&a_smem) = al.allocate<a128_tile>();
    b32_tile (&h_smem) = al.allocate<b32_tile>();
    d32_tile (&d_smem0) = al.allocate<d32_tile>();
    d32_tile (&d_smem1) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[2];
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a128_tile) + sizeof(b32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    if (gid >= 2) {
        return;
    }

    tensor_allocator<1, 1> tm_alloc{};
    const int tile_row_base = bid_m * CHUNK_M + gid * 64;
    const int tile_col_base = bid_k * CHUNK_K;
    d32_tile& d_smem = gid == 0 ? d_smem0 : d_smem1;
    semaphore& done = compute_done[gid];
    for (int sx = 0; sx < 4; ++sx) {
        auto a_sub = a_smem.template subtile<64, 32>({gid, sx});
        do_direct32_block_hadonly<WITH_RANDOM_SIGN_MASK>(
            a_sub,
            h_smem,
            d_smem,
            out_bf16,
            tm_alloc,
            done,
            gid + 1,
            gid,
            0,
            cols,
            tile_row_base,
            tile_col_base + sx * 32,
            rng_seed,
            rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * 8 + gid * 4 + sx) * TILE_M
        );
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS_2WG, 1)
void hadamard128_direct_quant_2wg_kernel(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<b32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    const int gid = warpgroupid();
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a128_tile (&a_smem) = al.allocate<a128_tile>();
    b32_tile (&h_smem) = al.allocate<b32_tile>();
    d32_tile (&d_smem0) = al.allocate<d32_tile>();
    d32_tile (&d_smem1) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[2];
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a128_tile) + sizeof(b32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    if (gid >= 2) {
        return;
    }

    tensor_allocator<1, 1> tm_alloc{};
    const int tile_row_base = bid_m * CHUNK_M + gid * 64;
    const int tile_col_base = bid_k * CHUNK_K;
    d32_tile& d_smem = gid == 0 ? d_smem0 : d_smem1;
    semaphore& done = compute_done[gid];
    for (int sx = 0; sx < 4; ++sx) {
        auto a_sub = a_smem.template subtile<64, 32>({gid, sx});
        do_direct32_block<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK>(
            a_sub,
            h_smem,
            d_smem,
            fp4_out,
            scales_out,
            tm_alloc,
            done,
            gid + 1,
            gid,
            0,
            cols,
            tile_row_base,
            bid_k * 4 + gid * 0 + sx,
            rng_seed,
            rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * 8 + gid * 4 + sx) * TILE_M
        );
    }
}

template<bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard128_recursive_bf16_kernel(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<b16_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a128_tile (&a_smem) = al.allocate<a128_tile>();
    b16_tile (&h_smem) = al.allocate<b16_tile>();
    a16_tile (&tmp0_smem) = al.allocate<a16_tile>();
    a16_tile (&tmp1_smem) = al.allocate<a16_tile>();
    d16_tile (&d0_smem) = al.allocate<d16_tile>();
    d16_tile (&d1_smem) = al.allocate<d16_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a128_tile) + sizeof(b16_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    const int tile_row_base = bid_m * CHUNK_M;
    const int tile_col_base = bid_k * CHUNK_K;
    int block_idx = 0;
    #pragma unroll
    for (int sy = 0; sy < 2; ++sy) {
        #pragma unroll
        for (int sx = 0; sx < 4; ++sx, ++block_idx) {
            auto a_sub = a_smem.template subtile<64, 32>({sy, sx});
            do_recursive16_block_hadonly<WITH_RANDOM_SIGN_MASK>(
                a_sub,
                h_smem,
                tmp0_smem,
                tmp1_smem,
                d0_smem,
                d1_smem,
                d_smem,
                out_bf16,
                tm_alloc,
                compute_done,
                1,
                cols,
                tile_row_base + sy * 64,
                tile_col_base + sx * 32,
                rng_seed,
                rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * 8 + block_idx) * TILE_M
            );
        }
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard128_recursive_quant_kernel(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<b16_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a128_tile (&a_smem) = al.allocate<a128_tile>();
    b16_tile (&h_smem) = al.allocate<b16_tile>();
    a16_tile (&tmp0_smem) = al.allocate<a16_tile>();
    a16_tile (&tmp1_smem) = al.allocate<a16_tile>();
    d16_tile (&d0_smem) = al.allocate<d16_tile>();
    d16_tile (&d1_smem) = al.allocate<d16_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a128_tile) + sizeof(b16_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    const int tile_row_base = bid_m * CHUNK_M;
    const int tile_col_base = bid_k * 4;
    int block_idx = 0;
    #pragma unroll
    for (int sy = 0; sy < 2; ++sy) {
        #pragma unroll
        for (int sx = 0; sx < 4; ++sx, ++block_idx) {
            auto a_sub = a_smem.template subtile<64, 32>({sy, sx});
            do_recursive16_block<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK>(
                a_sub,
                h_smem,
                tmp0_smem,
                tmp1_smem,
                d0_smem,
                d1_smem,
                d_smem,
                fp4_out,
                scales_out,
                tm_alloc,
                compute_done,
                1,
                cols,
                tile_row_base + sy * 64,
                tile_col_base + sx,
                rng_seed,
                rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * 8 + block_idx) * TILE_M
            );
        }
    }
}

__device__ __forceinline__ void merge_recursive_outputs(
    const d16_tile& left,
    const d16_tile& right,
    d32_tile& out
) {
    constexpr float kCombine = 0.7071067811865475f;  // 1 / sqrt(2)
    const int local_tid = threadIdx.x & 127;
    if (local_tid < TILE_M) {
        const int row = local_tid;
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            out[{row, col}] = __float2bfloat16_rn(__bfloat162float(left[{row, col}]) * kCombine);
            out[{row, 16 + col}] = __float2bfloat16_rn(__bfloat162float(right[{row, col}]) * kCombine);
        }
    }
}

template<bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard32_direct_bf16_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    const __grid_constant__ d_gl D_layout,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a32_tile>();
        H_layout.template prefetch_tma<b32_tile>();
        D_layout.template prefetch_tma<d32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_n = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a32_tile (&a_smem) = al.allocate<a32_tile>();
    b32_tile (&h_smem) = al.allocate<b32_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a32_tile) + sizeof(b32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_n}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_smem, rng_seed, rng_subsequence_base);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d32_tt accum;
    if (warpgroup::laneid() == 0) {
        accum = tm_alloc.template allocate<d32_tt>(0, 0);
    }
    warpgroup::sync(1);

    if (warpgroup::laneid() == 0) {
        mm_AB(accum, a_smem, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d32_rt d_reg;
    warpgroup::load_async(d_reg, accum);
    tensor_load_wait();
    warpgroup::sync(1);
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(1);

    if (warpgroup::laneid() == 0) {
        tma::store_async(D_layout, d_smem, {bid_m, bid_n});
        tma::store_async_read_wait();
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard32_direct_quant_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a32_tile>();
        H_layout.template prefetch_tma<b32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_n = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a32_tile (&a_smem) = al.allocate<a32_tile>();
    b32_tile (&h_smem) = al.allocate<b32_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a32_tile) + sizeof(b32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_n}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_smem, rng_seed, rng_subsequence_base);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d32_tt accum;
    if (warpgroup::laneid() == 0) {
        accum = tm_alloc.template allocate<d32_tt>(0, 0);
    }
    warpgroup::sync(1);

    if (warpgroup::laneid() == 0) {
        mm_AB(accum, a_smem, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d32_rt d_reg;
    warpgroup::load_async(d_reg, accum);
    tensor_load_wait();
    warpgroup::sync(1);
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(1);
    __syncthreads();

    quantize_tile_64x32<DATA_SR, SCALE_SR>(
        d_smem,
        fp4_out,
        scales_out,
        cols,
        bid_m * TILE_M,
        bid_n,
        rng_seed,
        rng_subsequence_base
    );
}

template<bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard32_recursive_bf16_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    const __grid_constant__ d_gl D_layout,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a32_tile>();
        H_layout.template prefetch_tma<b16_tile>();
        D_layout.template prefetch_tma<d32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_n = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a32_tile (&a_smem) = al.allocate<a32_tile>();
    a16_tile (&tmp0_smem) = al.allocate<a16_tile>();
    a16_tile (&tmp1_smem) = al.allocate<a16_tile>();
    b16_tile (&h_smem) = al.allocate<b16_tile>();
    d16_tile (&d0_smem) = al.allocate<d16_tile>();
    d16_tile (&d1_smem) = al.allocate<d16_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a32_tile) + sizeof(b16_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_n}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_smem, rng_seed, rng_subsequence_base);
    __syncthreads();
    combine_recursive_inputs(a_smem, tmp0_smem, tmp1_smem);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d16_tt accum0, accum1;
    if (warpgroup::laneid() == 0) {
        accum0 = tm_alloc.template allocate<d16_tt>(0, 0);
        accum1 = tm_alloc.template allocate<d16_tt>(0, TILE_N16);
    }
    warpgroup::sync(1);

    if (warpgroup::laneid() == 0) {
        mm_AB(accum0, tmp0_smem, h_smem);
        mm_AB(accum1, tmp1_smem, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d16_rt d0_reg, d1_reg;
    warpgroup::load_async(d0_reg, accum0);
    warpgroup::load_async(d1_reg, accum1);
    tensor_load_wait();
    warpgroup::sync(1);
    warpgroup::store(d0_smem, d0_reg);
    warpgroup::store(d1_smem, d1_reg);
    warpgroup::sync(1);
    __syncthreads();

    merge_recursive_outputs(d0_smem, d1_smem, d_smem);
    __syncthreads();

    if (warpgroup::laneid() == 0) {
        tma::store_async(D_layout, d_smem, {bid_m, bid_n});
        tma::store_async_read_wait();
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void hadamard32_recursive_quant_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a32_tile>();
        H_layout.template prefetch_tma<b16_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_n = blockIdx.x;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a32_tile (&a_smem) = al.allocate<a32_tile>();
    a16_tile (&tmp0_smem) = al.allocate<a16_tile>();
    a16_tile (&tmp1_smem) = al.allocate<a16_tile>();
    b16_tile (&h_smem) = al.allocate<b16_tile>();
    d16_tile (&d0_smem) = al.allocate<d16_tile>();
    d16_tile (&d1_smem) = al.allocate<d16_tile>();
    d32_tile (&d_smem) = al.allocate<d32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a32_tile) + sizeof(b16_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_n}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }
    wait(inputs_arrived, 0);
    __syncthreads();

    apply_sign_mask_64x32<WITH_RANDOM_SIGN_MASK>(a_smem, rng_seed, rng_subsequence_base);
    __syncthreads();
    combine_recursive_inputs(a_smem, tmp0_smem, tmp1_smem);
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d16_tt accum0, accum1;
    if (warpgroup::laneid() == 0) {
        accum0 = tm_alloc.template allocate<d16_tt>(0, 0);
        accum1 = tm_alloc.template allocate<d16_tt>(0, TILE_N16);
    }
    warpgroup::sync(1);

    if (warpgroup::laneid() == 0) {
        mm_AB(accum0, tmp0_smem, h_smem);
        mm_AB(accum1, tmp1_smem, h_smem);
        kittens::detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    d16_rt d0_reg, d1_reg;
    warpgroup::load_async(d0_reg, accum0);
    warpgroup::load_async(d1_reg, accum1);
    tensor_load_wait();
    warpgroup::sync(1);
    warpgroup::store(d0_smem, d0_reg);
    warpgroup::store(d1_smem, d1_reg);
    warpgroup::sync(1);
    __syncthreads();

    merge_recursive_outputs(d0_smem, d1_smem, d_smem);
    __syncthreads();

    quantize_tile_64x32<DATA_SR, SCALE_SR>(
        d_smem,
        fp4_out,
        scales_out,
        cols,
        bid_m * TILE_M,
        bid_n,
        rng_seed,
        rng_subsequence_base
    );
}

struct DeviceBuffers {
    __nv_bfloat16* input = nullptr;
    __nv_bfloat16* out_bf16 = nullptr;
    uint8_t* out_fp4 = nullptr;
    uint8_t* out_sc = nullptr;
    __nv_bfloat16* h16 = nullptr;
    __nv_bfloat16* h32 = nullptr;
};

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

std::vector<__nv_bfloat16> make_hadamard_host(int n) {
    std::vector<float> h(n * n, 1.0f);
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                for (int r = 0; r < n; ++r) {
                    h[r * n + i + len + j] = -h[r * n + i + len + j];
                }
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    std::vector<__nv_bfloat16> out(n * n);
    for (int i = 0; i < n * n; ++i) {
        out[i] = __float2bfloat16_rn(h[i] * scale);
    }
    return out;
}

__global__ void init_bf16_pattern_kernel(__nv_bfloat16* input, size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        uint32_t x = static_cast<uint32_t>(i);
        x ^= x >> 17;
        x *= 0xed5ad4bbU;
        x ^= x >> 11;
        const float v = (static_cast<int>(x & 511U) - 255) * (1.0f / 255.0f);
        input[i] = __float2bfloat16_rn(v);
    }
}

void init_device_buffers(DeviceBuffers& bufs, int rows, int cols) {
    const bool fast_init = std::getenv("TCBENCH_FAST_INIT") != nullptr;
    const size_t input_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const auto h16 = make_hadamard_host(16);
    const auto h32 = make_hadamard_host(32);

    check(cudaMalloc(&bufs.input, input_elems * sizeof(__nv_bfloat16)), "cudaMalloc input");
    check(cudaMalloc(&bufs.out_bf16, rows * cols * sizeof(__nv_bfloat16)), "cudaMalloc out_bf16");
    check(cudaMalloc(&bufs.out_fp4, rows * (cols / 2) * sizeof(uint8_t)), "cudaMalloc out_fp4");
    check(cudaMalloc(&bufs.out_sc, rows * (cols / 32) * sizeof(uint8_t)), "cudaMalloc out_sc");
    check(cudaMalloc(&bufs.h16, 16 * 16 * sizeof(__nv_bfloat16)), "cudaMalloc h16");
    check(cudaMalloc(&bufs.h32, 32 * 32 * sizeof(__nv_bfloat16)), "cudaMalloc h32");

    if (fast_init) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((input_elems + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        init_bf16_pattern_kernel<<<blocks, threads>>>(reinterpret_cast<__nv_bfloat16*>(bufs.input), input_elems);
        check(cudaGetLastError(), "launch input pattern");
        check(cudaDeviceSynchronize(), "sync input pattern");
    } else {
        std::vector<__nv_bfloat16> host(input_elems);
        std::mt19937 gen(1234);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto& v : host) {
            v = __float2bfloat16_rn(dis(gen));
        }
        check(cudaMemcpy(bufs.input, host.data(), input_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "memcpy input");
    }
    check(cudaMemcpy(bufs.h16, h16.data(), 16 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "memcpy h16");
    check(cudaMemcpy(bufs.h32, h32.data(), 32 * 32 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "memcpy h32");
}

void free_device_buffers(DeviceBuffers& bufs) {
    cudaFree(bufs.input);
    cudaFree(bufs.out_bf16);
    cudaFree(bufs.out_fp4);
    cudaFree(bufs.out_sc);
    cudaFree(bufs.h16);
    cudaFree(bufs.h32);
}

template<bool WITH_QUANT>
float run_direct32(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr
) {
    a128_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h32_gl H_layout{reinterpret_cast<bf16*>(bufs.h32), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    if constexpr (WITH_QUANT) {
        if (data_sr && scale_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_kernel<true, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct quant sr");
        } else if (data_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_kernel<true, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct quant data sr");
        } else if (scale_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_kernel<false, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct quant scale sr");
        } else {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_kernel<false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct quant");
        }
    } else {
        check(cudaFuncSetAttribute(
            hadamard128_direct_bf16_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr direct bf16");
    }
    auto launch = [&]() {
        if constexpr (WITH_QUANT) {
            if (data_sr && scale_sr) {
                hadamard128_direct_quant_kernel<true, true, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else if (data_sr) {
                hadamard128_direct_quant_kernel<true, false, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else if (scale_sr) {
                hadamard128_direct_quant_kernel<false, true, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else {
                hadamard128_direct_quant_kernel<false, false, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            }
        } else {
            hadamard128_direct_bf16_kernel<true><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_bf16, rows, cols, 1234, 0);
        }
    };

    if constexpr (WITH_QUANT) {
        check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 direct");
        check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc direct");
    } else {
        check(cudaMemset(bufs.out_bf16, 0, rows * cols * sizeof(__nv_bfloat16)), "memset bf16 direct");
    }

    for (int i = 0; i < 5; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "direct32 warmup sync");
    check(cudaGetLastError(), "direct32 warmup launch");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "direct32 timed sync");
    check(cudaGetLastError(), "direct32 timed launch");
    auto stop = std::chrono::high_resolution_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(stop - start).count() / 10.0f;

    std::array<uint8_t, 32> sample{};
    if constexpr (WITH_QUANT) {
        check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 direct sample");
    } else {
        check(cudaMemcpy(sample.data(), bufs.out_bf16, sample.size(), cudaMemcpyDeviceToHost), "copy bf16 direct sample");
    }
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum=" << checksum << "\n";
    return ms;
}

template<bool WITH_QUANT>
float run_direct32_2wg(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr
) {
    a128_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h32_gl H_layout{reinterpret_cast<bf16*>(bufs.h32), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    if constexpr (WITH_QUANT) {
        if (data_sr && scale_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_2wg_kernel<true, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct2wg quant sr");
        } else if (data_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_2wg_kernel<true, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct2wg quant data sr");
        } else if (scale_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_2wg_kernel<false, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct2wg quant scale sr");
        } else {
            check(cudaFuncSetAttribute(
                hadamard128_direct_quant_2wg_kernel<false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr direct2wg quant");
        }
    } else {
        check(cudaFuncSetAttribute(
            hadamard128_direct_bf16_2wg_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr direct2wg bf16");
    }
    auto launch = [&]() {
        if constexpr (WITH_QUANT) {
            if (data_sr && scale_sr) {
                hadamard128_direct_quant_2wg_kernel<true, true, true><<<grid, NUM_THREADS_2WG, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else if (data_sr) {
                hadamard128_direct_quant_2wg_kernel<true, false, true><<<grid, NUM_THREADS_2WG, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else if (scale_sr) {
                hadamard128_direct_quant_2wg_kernel<false, true, true><<<grid, NUM_THREADS_2WG, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else {
                hadamard128_direct_quant_2wg_kernel<false, false, true><<<grid, NUM_THREADS_2WG, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            }
        } else {
            hadamard128_direct_bf16_2wg_kernel<true><<<grid, NUM_THREADS_2WG, smem_size>>>(
                A_layout, H_layout, bufs.out_bf16, rows, cols, 1234, 0);
        }
    };

    if constexpr (WITH_QUANT) {
        check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 direct2wg");
        check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc direct2wg");
    } else {
        check(cudaMemset(bufs.out_bf16, 0, rows * cols * sizeof(__nv_bfloat16)), "memset bf16 direct2wg");
    }

    for (int i = 0; i < 5; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "direct2wg warmup sync");
    check(cudaGetLastError(), "direct2wg warmup launch");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "direct2wg timed sync");
    check(cudaGetLastError(), "direct2wg timed launch");
    auto stop = std::chrono::high_resolution_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(stop - start).count() / 10.0f;

    std::array<uint8_t, 32> sample{};
    if constexpr (WITH_QUANT) {
        check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 direct2wg sample");
    } else {
        check(cudaMemcpy(sample.data(), bufs.out_bf16, sample.size(), cudaMemcpyDeviceToHost), "copy bf16 direct2wg sample");
    }
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum=" << checksum << "\n";
    return ms;
}

template<bool WITH_QUANT>
float run_recursive16(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr
) {
    a128_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h16_gl H_layout{reinterpret_cast<bf16*>(bufs.h16), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    if constexpr (WITH_QUANT) {
        if (data_sr && scale_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_recursive_quant_kernel<true, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr recur quant sr");
        } else if (data_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_recursive_quant_kernel<true, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr recur quant data sr");
        } else if (scale_sr) {
            check(cudaFuncSetAttribute(
                hadamard128_recursive_quant_kernel<false, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr recur quant scale sr");
        } else {
            check(cudaFuncSetAttribute(
                hadamard128_recursive_quant_kernel<false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr recur quant");
        }
    } else {
        check(cudaFuncSetAttribute(
            hadamard128_recursive_bf16_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr recur bf16");
    }
    auto launch = [&]() {
        if constexpr (WITH_QUANT) {
            if (data_sr && scale_sr) {
                hadamard128_recursive_quant_kernel<true, true, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else if (data_sr) {
                hadamard128_recursive_quant_kernel<true, false, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else if (scale_sr) {
                hadamard128_recursive_quant_kernel<false, true, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            } else {
                hadamard128_recursive_quant_kernel<false, false, true><<<grid, NUM_THREADS, smem_size>>>(
                    A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
            }
        } else {
            hadamard128_recursive_bf16_kernel<true><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_bf16, rows, cols, 1234, 0);
        }
    };

    if constexpr (WITH_QUANT) {
        check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 recursive");
        check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc recursive");
    } else {
        check(cudaMemset(bufs.out_bf16, 0, rows * cols * sizeof(__nv_bfloat16)), "memset bf16 recursive");
    }

    for (int i = 0; i < 5; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "recursive16 warmup sync");
    check(cudaGetLastError(), "recursive16 warmup launch");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "recursive16 timed sync");
    check(cudaGetLastError(), "recursive16 timed launch");
    auto stop = std::chrono::high_resolution_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(stop - start).count() / 10.0f;

    std::array<uint8_t, 32> sample{};
    if constexpr (WITH_QUANT) {
        check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 recursive sample");
    } else {
        check(cudaMemcpy(sample.data(), bufs.out_bf16, sample.size(), cudaMemcpyDeviceToHost), "copy bf16 recursive sample");
    }
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum=" << checksum << "\n";
    return ms;
}

void run_case(int rows, int cols) {
    std::cout << "==== shape " << rows << " x " << cols << " ====\n";
    DeviceBuffers bufs;
    init_device_buffers(bufs, rows, cols);

    const bool only_2wg = std::getenv("TCBENCH_ONLY_2WG") != nullptr;
    const bool only_direct = std::getenv("TCBENCH_ONLY_DIRECT") != nullptr;
    const bool only_recursive = std::getenv("TCBENCH_ONLY_RECURSIVE") != nullptr;
    const bool skip_sr = std::getenv("TCBENCH_SKIP_SR") != nullptr;
    if (only_direct) {
        const float direct_had_quant = run_direct32<true>(bufs, rows, cols, false, false);
        const float direct_had_sr = skip_sr ? -1.0f : run_direct32<true>(bufs, rows, cols, true, true);
        std::cout << "direct32 had+quant:     " << direct_had_quant << " ms\n";
        if (direct_had_sr >= 0.0f) {
            std::cout << "direct32 had+quant+sr:  " << direct_had_sr << " ms\n";
        }
        free_device_buffers(bufs);
        return;
    }
    if (only_recursive) {
        const float recur_had_quant = run_recursive16<true>(bufs, rows, cols, false, false);
        const float recur_had_sr = skip_sr ? -1.0f : run_recursive16<true>(bufs, rows, cols, true, true);
        std::cout << "recursive16 had+quant:  " << recur_had_quant << " ms\n";
        if (recur_had_sr >= 0.0f) {
            std::cout << "recursive16 had+quant+sr:" << recur_had_sr << " ms\n";
        }
        free_device_buffers(bufs);
        return;
    }
    if (only_2wg) {
        const float direct2wg_had = run_direct32_2wg<false>(bufs, rows, cols, false, false);
        const float direct2wg_had_quant = run_direct32_2wg<true>(bufs, rows, cols, false, false);
        const float direct2wg_had_sr = skip_sr ? -1.0f : run_direct32_2wg<true>(bufs, rows, cols, true, true);
        std::cout << "direct32 2wg had-only:  " << direct2wg_had << " ms\n";
        std::cout << "direct32 2wg had+quant: " << direct2wg_had_quant << " ms\n";
        if (direct2wg_had_sr >= 0.0f) {
            std::cout << "direct32 2wg had+quant+sr:" << direct2wg_had_sr << " ms\n";
        }
        free_device_buffers(bufs);
        return;
    }

    const float direct_had = run_direct32<false>(bufs, rows, cols, false, false);
    const float direct_had_quant = run_direct32<true>(bufs, rows, cols, false, false);
    const float direct_had_sr = skip_sr ? -1.0f : run_direct32<true>(bufs, rows, cols, true, true);
    const float direct2wg_had = run_direct32_2wg<false>(bufs, rows, cols, false, false);
    const float direct2wg_had_quant = run_direct32_2wg<true>(bufs, rows, cols, false, false);
    const float direct2wg_had_sr = skip_sr ? -1.0f : run_direct32_2wg<true>(bufs, rows, cols, true, true);

    const float recur_had = run_recursive16<false>(bufs, rows, cols, false, false);
    const float recur_had_quant = run_recursive16<true>(bufs, rows, cols, false, false);
    const float recur_had_sr = skip_sr ? -1.0f : run_recursive16<true>(bufs, rows, cols, true, true);

    std::cout << "direct32 had-only:      " << direct_had << " ms\n";
    std::cout << "direct32 had+quant:     " << direct_had_quant << " ms\n";
    if (direct_had_sr >= 0.0f) {
        std::cout << "direct32 had+quant+sr:  " << direct_had_sr << " ms\n";
    }
    std::cout << "direct32 2wg had-only:  " << direct2wg_had << " ms\n";
    std::cout << "direct32 2wg had+quant: " << direct2wg_had_quant << " ms\n";
    if (direct2wg_had_sr >= 0.0f) {
        std::cout << "direct32 2wg had+quant+sr:" << direct2wg_had_sr << " ms\n";
    }
    std::cout << "recursive16 had-only:   " << recur_had << " ms\n";
    std::cout << "recursive16 had+quant:  " << recur_had_quant << " ms\n";
    if (recur_had_sr >= 0.0f) {
        std::cout << "recursive16 had+quant+sr:" << recur_had_sr << " ms\n";
    }

    free_device_buffers(bufs);
}

}  // namespace tcbench

int main() {
    if (const char* rows_env = std::getenv("TCBENCH_ROWS")) {
        const char* cols_env = std::getenv("TCBENCH_COLS");
        if (cols_env == nullptr) {
            std::cerr << "TCBENCH_COLS must be set with TCBENCH_ROWS\n";
            return 1;
        }
        tcbench::run_case(std::atoi(rows_env), std::atoi(cols_env));
        return 0;
    }
    if (std::getenv("TCBENCH_SMALL") != nullptr) {
        tcbench::run_case(128, 128);
        return 0;
    }
    tcbench::run_case(2048, 65536);
    tcbench::run_case(3072, 65536);
    tcbench::run_case(8192, 2048);
    return 0;
}
