#pragma once

#include <cuda_fp4.h>

#include "fused_amax_quantize.cuh"
#include "persistent_quantize.cuh"
#include "quantize_transpose_tuned.cuh"

namespace persistent_sqrelu_deriv_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

__device__ __forceinline__ float sqrelu_deriv_transform_tile(
    IType* sIn_dh_ptr,
    const IType* sIn_h1_ptr,
    int tile_idx) {
    auto& sDh = *reinterpret_cast<V3_IType3D*>(sIn_dh_ptr);
    const auto& sH1 = *reinterpret_cast<const V3_IType3D*>(sIn_h1_ptr);

    const int tid_Y = threadIdx.x / (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int tid_X = threadIdx.x % (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    float tile_max = 0.0f;

    constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int THREADS_Y = V3_THREADS / THREADS_X;
    constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            const float dh = __bfloat162float(sDh[tile_idx][row][col]);
            const float h1 = __bfloat162float(sH1[tile_idx][row][col]);
            const float relu = fmaxf(h1, 0.0f);
            const __nv_bfloat16 out_bf = __float2bfloat16_rn(2.0f * dh * relu);
            sDh[tile_idx][row][col] = out_bf;
            tile_max = fmaxf(tile_max, fabsf(__bfloat162float(out_bf)));
        }
    }

    return tile_max;
}

__device__ __forceinline__ float sqrelu_transform_tile(
    IType* sIn_ptr,
    int tile_idx) {
    auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int tid_X = threadIdx.x % (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    float tile_max = 0.0f;

    constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int THREADS_Y = V3_THREADS / THREADS_X;
    constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            const float x = __bfloat162float(sIn[tile_idx][row][col]);
            const float relu = fmaxf(x, 0.0f);
            const __nv_bfloat16 out_bf = __float2bfloat16_rn(relu * relu);
            sIn[tile_idx][row][col] = out_bf;
            tile_max = fmaxf(tile_max, fabsf(__bfloat162float(out_bf)));
        }
    }

    return tile_max;
}

__device__ __forceinline__ float scan_row_rht_amax_tile(
    const IType* sIn_ptr,
    int tile_idx) {
    const auto& sIn = *reinterpret_cast<const V3_IType3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / V3_THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % V3_THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    float tile_max = 0.0f;

#pragma unroll
    for (int it = 0; it < V3_ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * V3_THREADS_Y_ROWWISE;
        float vals[16];
#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            vals[e] = __bfloat162float(sIn[tile_idx][row][thread_offset_X + e]);
        }

        fwht16(vals);

#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            tile_max = fmaxf(tile_max, fabsf(vals[e]));
        }
    }

    return tile_max;
}

__device__ __forceinline__ float2 sqrelu_deriv_transform_scan_orig_row_tile(
    IType* sIn_dh_ptr,
    const IType* sIn_h1_ptr,
    int tile_idx) {
    auto& sDh = *reinterpret_cast<V3_IType3D*>(sIn_dh_ptr);
    const auto& sH1 = *reinterpret_cast<const V3_IType3D*>(sIn_h1_ptr);

    const int tid_Y = threadIdx.x / V3_THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % V3_THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    float orig_max = 0.0f;
    float row_max = 0.0f;

#pragma unroll
    for (int it = 0; it < V3_ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * V3_THREADS_Y_ROWWISE;
        float vals[16];

#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            const float dh = __bfloat162float(sDh[tile_idx][row][col]);
            const float h1 = __bfloat162float(sH1[tile_idx][row][col]);
            const float relu = fmaxf(h1, 0.0f);
            const __nv_bfloat16 out_bf = __float2bfloat16_rn(2.0f * dh * relu);
            const float out = __bfloat162float(out_bf);
            sDh[tile_idx][row][col] = out_bf;
            vals[e] = out;
            orig_max = fmaxf(orig_max, fabsf(out));
        }

        fwht16(vals);

#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            row_max = fmaxf(row_max, fabsf(vals[e]));
        }
    }

    return make_float2(orig_max, row_max);
}

__device__ __forceinline__ float2 sqrelu_transform_scan_orig_row_tile(
    IType* sIn_ptr,
    int tile_idx) {
    auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / V3_THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % V3_THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    float orig_max = 0.0f;
    float row_max = 0.0f;

#pragma unroll
    for (int it = 0; it < V3_ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * V3_THREADS_Y_ROWWISE;
        float vals[16];

#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            const float x = __bfloat162float(sIn[tile_idx][row][col]);
            const float relu = fmaxf(x, 0.0f);
            const __nv_bfloat16 out_bf = __float2bfloat16_rn(relu * relu);
            const float out = __bfloat162float(out_bf);
            sIn[tile_idx][row][col] = out_bf;
            vals[e] = out;
            orig_max = fmaxf(orig_max, fabsf(out));
        }

        fwht16(vals);

#pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            row_max = fmaxf(row_max, fabsf(vals[e]));
        }
    }

    return make_float2(orig_max, row_max);
}

__device__ __forceinline__ void reduce_orig_row_pair(float& orig, float& row) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        orig = fmaxf(orig, __shfl_xor_sync(0xffffffff, orig, mask));
        row = fmaxf(row, __shfl_xor_sync(0xffffffff, row, mask));
    }

    __shared__ float warp_orig[V3_THREADS / 32];
    __shared__ float warp_row[V3_THREADS / 32];
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_orig[wid] = orig;
        warp_row[wid] = row;
    }
    __syncthreads();

    if (wid == 0) {
        orig = (lane < V3_THREADS / 32) ? warp_orig[lane] : 0.0f;
        row = (lane < V3_THREADS / 32) ? warp_row[lane] : 0.0f;
#pragma unroll
        for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1) {
            orig = fmaxf(orig, __shfl_xor_sync(0xffffffff, orig, mask));
            row = fmaxf(row, __shfl_xor_sync(0xffffffff, row, mask));
        }
    }
}

template <bool RETURN_TRANSPOSE, bool ROW_RHT = true, bool USE_STOCHASTIC_ROUNDING = false,
          bool ENCODE_CENTRIC = true, bool USE_SCALE_STOCHASTIC_ROUNDING = false>
__device__ __forceinline__ void sqrelu_quantize_and_store_chunk_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr,
    nvfp4_scale_t* sSFcolwise_ptr,
    V3_OType2x3D& sOut,
    V3_OType2xt3D& sOut_tr,
    const CUtensorMap& tmap_out,
    const CUtensorMap& tmap_out_t,
    const CUtensorMap& tmap_scale_row,
    const CUtensorMap& tmap_scale_col,
    float S_enc_row,
    float S_enc_col,
    int block_offset_Y,
    int block_offset_X,
    int rows,
    int cols,
    int ctaid_X,
    int ctaid_Y,
    uint64_t rng_seed,
    uint64_t rng_offset,
    uint64_t* in_mbar,
    int mbar_phase,
    const CUtensorMap* tmap_in) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_cols = cols - block_offset_X;
    const int chunk_rows = rows - block_offset_Y;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    int buff_out = 0;
    int buff_out_tr = 0;

    auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);

    using QRNG = transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel::RNG_t;
    QRNG rng;
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 0;
    if constexpr (USE_STOCHASTIC_ROUNDING || USE_SCALE_STOCHASTIC_ROUNDING) {
        const uint64_t chunk_sequence =
            (static_cast<uint64_t>(ctaid_Y) *
                 static_cast<uint64_t>((cols + V3Config::CHUNK_DIM_X - 1) / V3Config::CHUNK_DIM_X) +
             static_cast<uint64_t>(ctaid_X)) *
                static_cast<uint64_t>(V3_THREADS) +
            static_cast<uint64_t>(threadIdx.x);
        rng.init(rng_seed, chunk_sequence, rng_offset);
        random_uint4 = rng.generate4();
    }

#pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        if (t + 2 < V3_NUM_TILES) {
            const int next = t + 2;
            const int nty = next / V3_TILES_X;
            const int ntx = next % V3_TILES_X;
            if (leading) {
                constexpr int bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);
                ptx::mbarrier_arrive_expect_tx(&in_mbar[next], bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tmap_in),
                    block_offset_X + ntx * V3_TILE_DIM_X,
                    block_offset_Y + nty * V3_TILE_DIM_Y,
                    &in_mbar[next]);
            }
        }

        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        sqrelu_transform_tile(sIn_ptr, t);
        __syncthreads();

        transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel::rowwise_scaling<
            USE_STOCHASTIC_ROUNDING, false, ENCODE_CENTRIC, ROW_RHT,
            USE_SCALE_STOCHASTIC_ROUNDING>(
            sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc_row, stage_Y, stage_X, t, buff_out,
            rng, random_uint4, rnd_idx);

        if constexpr (RETURN_TRANSPOSE) {
            transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel::colwise_scaling<
                USE_STOCHASTIC_ROUNDING, false, ENCODE_CENTRIC, false,
                USE_SCALE_STOCHASTIC_ROUNDING>(
                sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc_col, stage_Y, stage_X, t,
                buff_out_tr, rng, random_uint4, rnd_idx);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_out),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));

            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
        if constexpr (RETURN_TRANSPOSE) {
            buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
        }
    }

    if (leading) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    {
        const int cnt = min(static_cast<int>(V3_SCALES_PER_CHUNK_X), chunk_cols / static_cast<int>(V3_SCALE_DIM));
        if (cnt == static_cast<int>(V3_SCALES_PER_CHUNK_X)) {
            tk_v5::swizzle_scales_row_full_inplace(sSFrowwise_ptr);
        } else {
            tk_v5::swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm = block_offset_Y / 128;
            const int tma_x_base = ctaid_X * 2 * 256;
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row),
                tma_x_base,
                tm,
                reinterpret_cast<uint64_t*>(sSFrowwise_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row),
                tma_x_base + 256,
                tm,
                reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFrowwise_ptr) + 512));
            ptx::cp_async_bulk_commit_group();
        }
    }

    if constexpr (RETURN_TRANSPOSE) {
        const int cnt = min(static_cast<int>(V3_SCALES_PER_CHUNK_Y), chunk_rows / static_cast<int>(V3_SCALE_DIM));
        if (cnt == static_cast<int>(V3_SCALES_PER_CHUNK_Y)) {
            tk_v5::swizzle_scales_col_full_inplace(sSFcolwise_ptr);
        } else {
            tk_v5::swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm_col = block_offset_X / 128;
            const int tma_x_base = ctaid_Y * 2 * 256;
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                tma_x_base,
                tm_col,
                reinterpret_cast<uint64_t*>(sSFcolwise_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                tma_x_base + 256,
                tm_col,
                reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise_ptr) + 512));
            ptx::cp_async_bulk_commit_group();
        }
    }

    if (leading) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
}

template <bool RETURN_TRANSPOSE, bool ROW_RHT = true, bool USE_STOCHASTIC_ROUNDING = false,
          bool ENCODE_CENTRIC = true, bool USE_SCALE_STOCHASTIC_ROUNDING = false>
__device__ __forceinline__ void sqrelu_deriv_quantize_and_store_chunk_pipelined(
    IType* sIn_dh_ptr,
    IType* sIn_h1_ptr,
    fp4e2m1x2* sOut_ptr,
    fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr,
    nvfp4_scale_t* sSFcolwise_ptr,
    V3_OType2x3D& sOut,
    V3_OType2xt3D& sOut_tr,
    const CUtensorMap& tmap_out,
    const CUtensorMap& tmap_out_t,
    const CUtensorMap& tmap_scale_row,
    const CUtensorMap& tmap_scale_col,
    float S_enc_row,
    float S_enc_col,
    int block_offset_Y,
    int block_offset_X,
    int rows,
    int cols,
    int ctaid_X,
    int ctaid_Y,
    uint64_t rng_seed,
    uint64_t rng_offset,
    uint64_t* dh_mbar,
    uint64_t* h1_mbar,
    int mbar_phase,
    const CUtensorMap* tmap_dh,
    const CUtensorMap* tmap_h1) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_cols = cols - block_offset_X;
    const int chunk_rows = rows - block_offset_Y;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    int buff_out = 0;
    int buff_out_tr = 0;

    auto& sDh = *reinterpret_cast<V3_IType3D*>(sIn_dh_ptr);

    using QRNG = transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel::RNG_t;
    QRNG rng;
    const uint64_t chunk_sequence =
        (static_cast<uint64_t>(ctaid_Y) *
             static_cast<uint64_t>((cols + V3Config::CHUNK_DIM_X - 1) / V3Config::CHUNK_DIM_X) +
         static_cast<uint64_t>(ctaid_X)) *
            static_cast<uint64_t>(V3_THREADS) +
        static_cast<uint64_t>(threadIdx.x);
    rng.init(rng_seed, chunk_sequence, rng_offset);
    uint4 random_uint4 = rng.generate4();
    int rnd_idx = 0;

#pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        if (t + 2 < V3_NUM_TILES) {
            const int next = t + 2;
            const int nty = next / V3_TILES_X;
            const int ntx = next % V3_TILES_X;
            if (leading) {
                constexpr int bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[next], bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[next]),
                    reinterpret_cast<const uint64_t*>(tmap_dh),
                    block_offset_X + ntx * V3_TILE_DIM_X,
                    block_offset_Y + nty * V3_TILE_DIM_Y,
                    &dh_mbar[next]);

                ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&(*reinterpret_cast<V3_IType3D*>(sIn_h1_ptr))[next]),
                    reinterpret_cast<const uint64_t*>(tmap_h1),
                    block_offset_X + ntx * V3_TILE_DIM_X,
                    block_offset_Y + nty * V3_TILE_DIM_Y,
                    &h1_mbar[next]);
            }
        }

        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], mbar_phase);
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);

        sqrelu_deriv_transform_tile(sIn_dh_ptr, sIn_h1_ptr, t);
        __syncthreads();

        transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel::rowwise_scaling<
            USE_STOCHASTIC_ROUNDING, false, ENCODE_CENTRIC, ROW_RHT,
            USE_SCALE_STOCHASTIC_ROUNDING>(
            sIn_dh_ptr, sOut_ptr, sSFrowwise_ptr, S_enc_row, stage_Y, stage_X, t, buff_out,
            rng, random_uint4, rnd_idx);

        if constexpr (RETURN_TRANSPOSE) {
            transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel::colwise_scaling<
                USE_STOCHASTIC_ROUNDING, false, ENCODE_CENTRIC, false,
                USE_SCALE_STOCHASTIC_ROUNDING>(
                sIn_dh_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc_col, stage_Y, stage_X, t,
                buff_out_tr, rng, random_uint4, rnd_idx);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_out),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));

            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
        if constexpr (RETURN_TRANSPOSE) {
            buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
        }
    }

    if (leading) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    {
        const int cnt = min(static_cast<int>(V3_SCALES_PER_CHUNK_X), chunk_cols / static_cast<int>(V3_SCALE_DIM));
        if (cnt == static_cast<int>(V3_SCALES_PER_CHUNK_X)) {
            tk_v5::swizzle_scales_row_full_inplace(sSFrowwise_ptr);
        } else {
            tk_v5::swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm = block_offset_Y / 128;
            const int tma_x_base = ctaid_X * 2 * 256;
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row),
                tma_x_base,
                tm,
                reinterpret_cast<uint64_t*>(sSFrowwise_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row),
                tma_x_base + 256,
                tm,
                reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFrowwise_ptr) + 512));
            ptx::cp_async_bulk_commit_group();
        }
    }

    if constexpr (RETURN_TRANSPOSE) {
        const int cnt = min(static_cast<int>(V3_SCALES_PER_CHUNK_Y), chunk_rows / static_cast<int>(V3_SCALE_DIM));
        if (cnt == static_cast<int>(V3_SCALES_PER_CHUNK_Y)) {
            tk_v5::swizzle_scales_col_full_inplace(sSFcolwise_ptr);
        } else {
            tk_v5::swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm_col = block_offset_X / 128;
            const int tma_x_base = ctaid_Y * 2 * 256;
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                tma_x_base,
                tm_col,
                reinterpret_cast<uint64_t*>(sSFcolwise_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                tma_x_base + 256,
                tm_col,
                reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise_ptr) + 512));
            ptx::cp_async_bulk_commit_group();
        }
    }

    if (leading) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
}

template <bool RETURN_TRANSPOSE>
inline int persistent_sqrelu_quant_smem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * static_cast<int>(sizeof(IType)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE,
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE
        ? DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT)
        : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * static_cast<int>(sizeof(nvfp4_scale_t)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE
        ? DIVUP_TO_MULTIPLE(
              V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y *
                  static_cast<int>(sizeof(nvfp4_scale_t)),
              TMA_SHMEM_ALIGNMENT)
        : 0;
    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <bool RETURN_TRANSPOSE, bool ROW_RHT = true, bool USE_STOCHASTIC_ROUNDING = false,
          bool ENCODE_CENTRIC = true, bool USE_SCALE_STOCHASTIC_ROUNDING = false>
__global__ void __launch_bounds__(V3_THREADS)
persistent_sqrelu_quantize_kernel(
    const __grid_constant__ CUtensorMap tmap_in,
    const __grid_constant__ CUtensorMap tmap_out,
    const __grid_constant__ CUtensorMap tmap_out_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const size_t rows,
    const size_t cols,
    const size_t scale_stride,
    tk_v5::PersistentArgs args,
    float* global_row_amax,
    uint64_t rng_seed = 0,
    uint64_t rng_offset = 0,
    const uint64_t* rng_state = nullptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    if constexpr (USE_STOCHASTIC_ROUNDING || USE_SCALE_STOCHASTIC_ROUNDING) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_offset = rng_state[1];
        }
    }
    (void)scale_stride;
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * static_cast<int>(sizeof(IType)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE,
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE
        ? DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT)
        : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * static_cast<int>(sizeof(nvfp4_scale_t)),
        TMA_SHMEM_ALIGNMENT);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    int off = in_bytes;
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off);
    off += out_bytes;
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off);
    off += out_tr_bytes;
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);
    off += sc_row_bytes;
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);

    auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float block_orig_max = 0.0f;
    float block_row_max = 0.0f;
    int mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

#pragma unroll
        for (int pre = 0; pre < min(2, static_cast<int>(V3_NUM_TILES)); ++pre) {
            const int ty = pre / V3_TILES_X;
            const int tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_in),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            if (t + 2 < V3_NUM_TILES) {
                const int next = t + 2;
                const int nty = next / V3_TILES_X;
                const int ntx = next % V3_TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[next]),
                        reinterpret_cast<const uint64_t*>(&tmap_in),
                        block_offset_X + ntx * V3_TILE_DIM_X,
                        block_offset_Y + nty * V3_TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

            float orig;
            float row;
            if constexpr (ROW_RHT) {
                const float2 tile_maxes = sqrelu_transform_scan_orig_row_tile(sIn_ptr, t);
                orig = tile_maxes.x;
                row = tile_maxes.y;
            } else {
                orig = sqrelu_transform_tile(sIn_ptr, t);
                row = orig;
            }
            block_orig_max = fmaxf(block_orig_max, orig);
            block_row_max = fmaxf(block_row_max, row);
        }
        mbar_phase ^= 1;
    }

    reduce_orig_row_pair(block_orig_max, block_row_max);

    __shared__ bool is_last_block;
    if (leading) {
        if (block_orig_max > 0.0f) {
            atomicMax(reinterpret_cast<unsigned int*>(args.global_amax), __float_as_uint(block_orig_max));
        }
        if constexpr (ROW_RHT) {
            if (block_row_max > 0.0f) {
                atomicMax(reinterpret_cast<unsigned int*>(global_row_amax), __float_as_uint(block_row_max));
            }
        }

        __threadfence();
        const unsigned int cnt = atomicAdd(args.done_counter, 1);
        is_last_block = (cnt == static_cast<unsigned int>(args.num_persistent - 1));
    }
    __syncthreads();

    if (is_last_block) {
        if (leading) {
            atomicExch(args.ready_flag, 1);
            *args.done_counter = 0;
        }
    } else {
        while (*(volatile unsigned int*)args.ready_flag != 1) {}
    }
    __syncthreads();

    const float orig_amax_val =
        __uint_as_float(*(volatile unsigned int*)args.global_amax);
    const float row_amax_val = ROW_RHT
        ? __uint_as_float(*(volatile unsigned int*)global_row_amax)
        : orig_amax_val;
    const float S_enc_row = compute_global_encode_scaling_factor_FP4(row_amax_val);
    const float S_enc_col = compute_global_encode_scaling_factor_FP4(orig_amax_val);

    if (leading && blockIdx.x == 0 && args.sg_output) {
        args.sg_output[0] = row_amax_val / 2688.0f;
        args.sg_output[1] = orig_amax_val / 2688.0f;
    }

    if (leading) {
#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id2;
        if (leading) {
            s_chunk_id2 = atomicAdd(args.work_counter_phase2, 1);
        }
        __syncthreads();
        if (s_chunk_id2 >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = s_chunk_id2 % args.tiles_X;
        const int ctaid_Y = s_chunk_id2 / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

#pragma unroll
        for (int pre = 0; pre < min(2, static_cast<int>(V3_NUM_TILES)); ++pre) {
            const int ty = pre / V3_TILES_X;
            const int tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_in),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

        sqrelu_quantize_and_store_chunk_pipelined<
            RETURN_TRANSPOSE, ROW_RHT, USE_STOCHASTIC_ROUNDING, ENCODE_CENTRIC,
            USE_SCALE_STOCHASTIC_ROUNDING>(
            sIn_ptr,
            sOut_ptr,
            sOut_tr_ptr,
            sSFrowwise_ptr,
            sSFcolwise_ptr,
            sOut,
            sOut_tr,
            tmap_out,
            tmap_out_t,
            tmap_scale_row,
            tmap_scale_col,
            S_enc_row,
            S_enc_col,
            static_cast<int>(block_offset_Y),
            static_cast<int>(block_offset_X),
            static_cast<int>(rows),
            static_cast<int>(cols),
            ctaid_X,
            ctaid_Y,
            rng_seed,
            rng_offset,
            in_mbar,
            mbar_phase,
            &tmap_in);
        mbar_phase ^= 1;
    }

    if (leading) {
#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
inline int persistent_sqrelu_deriv_quant_smem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * static_cast<int>(sizeof(IType)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE,
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE
        ? DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT)
        : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * static_cast<int>(sizeof(nvfp4_scale_t)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE
        ? DIVUP_TO_MULTIPLE(
              V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y *
                  static_cast<int>(sizeof(nvfp4_scale_t)),
              TMA_SHMEM_ALIGNMENT)
        : 0;
    return 2 * in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <bool RETURN_TRANSPOSE, bool ROW_RHT = true, bool USE_STOCHASTIC_ROUNDING = false,
          bool ENCODE_CENTRIC = true, bool USE_SCALE_STOCHASTIC_ROUNDING = false>
__global__ void __launch_bounds__(V3_THREADS)
persistent_sqrelu_deriv_quantize_kernel(
    const __grid_constant__ CUtensorMap tmap_dh,
    const __grid_constant__ CUtensorMap tmap_h1,
    const __grid_constant__ CUtensorMap tmap_out,
    const __grid_constant__ CUtensorMap tmap_out_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const size_t rows,
    const size_t cols,
    const size_t scale_stride,
    tk_v5::PersistentArgs args,
    float* global_row_amax,
    uint64_t rng_seed = 0,
    uint64_t rng_offset = 0,
    const uint64_t* rng_state = nullptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    if constexpr (USE_STOCHASTIC_ROUNDING || USE_SCALE_STOCHASTIC_ROUNDING) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_offset = rng_state[1];
        }
    }
    (void)scale_stride;
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * static_cast<int>(sizeof(IType)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE,
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE
        ? DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT)
        : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * static_cast<int>(sizeof(nvfp4_scale_t)),
        TMA_SHMEM_ALIGNMENT);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_dh_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn_h1_ptr = reinterpret_cast<IType*>(dshmem + in_bytes);
    int off = 2 * in_bytes;
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off);
    off += out_bytes;
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off);
    off += out_tr_bytes;
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);
    off += sc_row_bytes;
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);

    auto& sDh = *reinterpret_cast<V3_IType3D*>(sIn_dh_ptr);
    auto& sH1 = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    auto& sOut = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t dh_mbar[V3_NUM_TILES];
    __shared__ uint64_t h1_mbar[V3_NUM_TILES];
    if (leading) {
#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_init(&dh_mbar[t], 1);
            ptx::mbarrier_init(&h1_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float block_orig_max = 0.0f;
    float block_row_max = 0.0f;
    int mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

#pragma unroll
        for (int pre = 0; pre < min(2, static_cast<int>(V3_NUM_TILES)); ++pre) {
            const int ty = pre / V3_TILES_X;
            const int tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_dh),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &dh_mbar[pre]);

                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_h1),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &h1_mbar[pre]);
            }
        }

#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            if (t + 2 < V3_NUM_TILES) {
                const int next = t + 2;
                const int nty = next / V3_TILES_X;
                const int ntx = next % V3_TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&dh_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sDh[next]),
                        reinterpret_cast<const uint64_t*>(&tmap_dh),
                        block_offset_X + ntx * V3_TILE_DIM_X,
                        block_offset_Y + nty * V3_TILE_DIM_Y,
                        &dh_mbar[next]);

                    ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH1[next]),
                        reinterpret_cast<const uint64_t*>(&tmap_h1),
                        block_offset_X + ntx * V3_TILE_DIM_X,
                        block_offset_Y + nty * V3_TILE_DIM_Y,
                        &h1_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);

            float orig;
            float row;
            if constexpr (ROW_RHT) {
                const float2 tile_maxes =
                    sqrelu_deriv_transform_scan_orig_row_tile(sIn_dh_ptr, sIn_h1_ptr, t);
                orig = tile_maxes.x;
                row = tile_maxes.y;
            } else {
                orig = sqrelu_deriv_transform_tile(sIn_dh_ptr, sIn_h1_ptr, t);
                row = orig;
            }
            block_orig_max = fmaxf(block_orig_max, orig);
            block_row_max = fmaxf(block_row_max, row);
        }
        mbar_phase ^= 1;
    }

    reduce_orig_row_pair(block_orig_max, block_row_max);

    __shared__ bool is_last_block;
    if (leading) {
        if (block_orig_max > 0.0f) {
            atomicMax(reinterpret_cast<unsigned int*>(args.global_amax), __float_as_uint(block_orig_max));
        }
        if constexpr (ROW_RHT) {
            if (block_row_max > 0.0f) {
                atomicMax(reinterpret_cast<unsigned int*>(global_row_amax), __float_as_uint(block_row_max));
            }
        }

        __threadfence();
        const unsigned int cnt = atomicAdd(args.done_counter, 1);
        is_last_block = (cnt == static_cast<unsigned int>(args.num_persistent - 1));
    }
    __syncthreads();

    if (is_last_block) {
        if (leading) {
            atomicExch(args.ready_flag, 1);
            *args.done_counter = 0;
        }
    } else {
        while (*(volatile unsigned int*)args.ready_flag != 1) {}
    }
    __syncthreads();

    const float orig_amax_val =
        __uint_as_float(*(volatile unsigned int*)args.global_amax);
    const float row_amax_val = ROW_RHT
        ? __uint_as_float(*(volatile unsigned int*)global_row_amax)
        : orig_amax_val;
    const float S_enc_row = compute_global_encode_scaling_factor_FP4(row_amax_val);
    const float S_enc_col = compute_global_encode_scaling_factor_FP4(orig_amax_val);

    if (leading && blockIdx.x == 0 && args.sg_output) {
        args.sg_output[0] = row_amax_val / 2688.0f;
        args.sg_output[1] = orig_amax_val / 2688.0f;
    }

    if (leading) {
#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&dh_mbar[t]);
            ptx::mbarrier_init(&dh_mbar[t], 1);
            ptx::mbarrier_invalid(&h1_mbar[t]);
            ptx::mbarrier_init(&h1_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id2;
        if (leading) {
            s_chunk_id2 = atomicAdd(args.work_counter_phase2, 1);
        }
        __syncthreads();
        if (s_chunk_id2 >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = s_chunk_id2 % args.tiles_X;
        const int ctaid_Y = s_chunk_id2 / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

#pragma unroll
        for (int pre = 0; pre < min(2, static_cast<int>(V3_NUM_TILES)); ++pre) {
            const int ty = pre / V3_TILES_X;
            const int tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_dh),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &dh_mbar[pre]);

                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_h1),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &h1_mbar[pre]);
            }
        }

        sqrelu_deriv_quantize_and_store_chunk_pipelined<
            RETURN_TRANSPOSE, ROW_RHT, USE_STOCHASTIC_ROUNDING, ENCODE_CENTRIC,
            USE_SCALE_STOCHASTIC_ROUNDING>(
            sIn_dh_ptr,
            sIn_h1_ptr,
            sOut_ptr,
            sOut_tr_ptr,
            sSFrowwise_ptr,
            sSFcolwise_ptr,
            sOut,
            sOut_tr,
            tmap_out,
            tmap_out_t,
            tmap_scale_row,
            tmap_scale_col,
            S_enc_row,
            S_enc_col,
            static_cast<int>(block_offset_Y),
            static_cast<int>(block_offset_X),
            static_cast<int>(rows),
            static_cast<int>(cols),
            ctaid_X,
            ctaid_Y,
            rng_seed,
            rng_offset,
            dh_mbar,
            h1_mbar,
            mbar_phase,
            &tmap_dh,
            &tmap_h1);
        mbar_phase ^= 1;
    }

    if (leading) {
#pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&dh_mbar[t]);
            ptx::mbarrier_invalid(&h1_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

#endif

}  // namespace persistent_sqrelu_deriv_quant
