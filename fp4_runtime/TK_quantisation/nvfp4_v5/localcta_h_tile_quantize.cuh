#pragma once

#include "persistent_norm_quantize.cuh"

namespace tk_v5_h_tile {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

__host__ __forceinline__ int dynamic_smem_size() {
    return tk_v5::persistent_norm_quant_smem_size(true);
}

__device__ __forceinline__ void normalize_tile(
    IType* input,
    const IType* gamma,
    float inverse_rms,
    int tile) {
    auto& tiles = *reinterpret_cast<V3_IType3D*>(input);
    constexpr int threads_x = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int threads_y = V3_THREADS / threads_x;
    constexpr int iterations = V3_TILE_DIM_Y / threads_y;
    const int thread_y = threadIdx.x / threads_x;
    const int thread_x = threadIdx.x % threads_x;
    const int col0 = thread_x * V3_ELTS_PER_THREAD;

    #pragma unroll
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int row = thread_y + iteration * threads_y;
        #pragma unroll
        for (int element = 0; element < V3_ELTS_PER_THREAD; ++element) {
            const int col = col0 + element;
            const float value =
                __bfloat162float(tiles[tile][row][col]) * inverse_rms *
                __bfloat162float(gamma[col]);
            tiles[tile][row][col] = __float2bfloat16_rn(value);
        }
    }
}

}  // namespace tk_v5_h_tile

namespace tk_v5_localcta_h_tile {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

constexpr float LOCALCTA_SCALE_NUM = 1493.0f;

__device__ __forceinline__ float compute_local_encode(float local_amax) {
    float encode = LOCALCTA_SCALE_NUM / local_amax;
    encode = fminf(encode, transformer_engine::detail::TypeExtrema<float>::max);
    if (local_amax == 0.0f || encode == 0.0f) {
        return 1.0f;
    }
    return encode;
}

__host__ __forceinline__ int dynamic_smem_size() {
    return tk_v5_h_tile::dynamic_smem_size();
}

__global__ void __launch_bounds__(V3_THREADS)
persistent_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const float* __restrict__ r_tile,
    const IType* __restrict__ gamma,
    const float* __restrict__ row_outer_scales,
    const float* __restrict__ col_outer_scales,
    size_t rows,
    size_t cols,
    unsigned int* __restrict__ work_counter,
    int total_tiles) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = threadIdx.x == 0;
    constexpr int tile_bytes =
        V3_BUFF_DIM_Y * V3_BUFF_DIM_X * static_cast<int>(sizeof(IType));
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * static_cast<int>(sizeof(IType)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X *
            static_cast<int>(sizeof(nvfp4_scale_t)),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y *
            static_cast<int>(sizeof(nvfp4_scale_t)),
        TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* smem = common::align_smem_ptr_per_TMA_requirements(
        dynamic_shmem);
    IType* s_in_ptr = reinterpret_cast<IType*>(smem);
    auto* s_out_ptr = reinterpret_cast<fp4e2m1x2*>(smem + in_bytes);
    auto* s_out_tr_ptr = reinterpret_cast<fp4e2m1x2*>(
        smem + in_bytes + out_bytes);
    auto* s_row_sc_ptr = reinterpret_cast<nvfp4_scale_t*>(
        smem + in_bytes + out_bytes + out_tr_bytes);
    auto* s_col_sc_ptr = reinterpret_cast<nvfp4_scale_t*>(
        smem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);
    IType* gamma_cache = reinterpret_cast<IType*>(
        smem + in_bytes + out_bytes + out_tr_bytes +
        sc_row_bytes + sc_col_bytes);

    auto& s_in = *reinterpret_cast<V3_IType3D*>(s_in_ptr);
    auto& s_out = *reinterpret_cast<V3_OType2x3D*>(s_out_ptr);
    auto& s_out_tr = *reinterpret_cast<V3_OType2xt3D*>(s_out_tr_ptr);

    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    __shared__ float row_encode_shared;
    __shared__ float col_encode_shared;
    if (leading) {
        #pragma unroll
        for (int tile = 0; tile < V3_NUM_TILES; ++tile) {
            ptx::mbarrier_init(&in_mbar[tile], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int mbar_phase = 0;
    while (true) {
        __shared__ unsigned int chunk;
        if (leading) {
            chunk = atomicAdd(work_counter, 1);
        }
        __syncthreads();
        if (chunk >= static_cast<unsigned int>(total_tiles)) {
            break;
        }

        const int tiles_x = static_cast<int>(cols) / V3Config::CHUNK_DIM_X;
        const int tile_x = static_cast<int>(chunk) % tiles_x;
        const int tile_y = static_cast<int>(chunk) / tiles_x;
        const int row0 = tile_y * V3Config::CHUNK_DIM_Y;
        const int col0 = tile_x * V3Config::CHUNK_DIM_X;
        const float r = r_tile[chunk];

        if (leading) {
            row_encode_shared = compute_local_encode(
                row_outer_scales[tile_y / 2] * LOCALCTA_SCALE_NUM);
            col_encode_shared = compute_local_encode(
                col_outer_scales[tile_x / 2] * LOCALCTA_SCALE_NUM);
        }
        for (int col = threadIdx.x;
             col < V3Config::CHUNK_DIM_X;
             col += V3_THREADS) {
            gamma_cache[col] = gamma[col0 + col];
        }
        __syncthreads();
        const float row_encode = row_encode_shared;
        const float col_encode = col_encode_shared;

        #pragma unroll
        for (int preload = 0; preload < 2; ++preload) {
            const int local_y = preload / V3_TILES_X;
            const int local_x = preload % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[preload], tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&s_in[preload]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    col0 + local_x * V3_TILE_DIM_X,
                    row0 + local_y * V3_TILE_DIM_Y,
                    &in_mbar[preload]);
            }
        }

        int out_stage = 0;
        int out_tr_stage = 0;
        #pragma unroll
        for (int tile = 0; tile < V3_NUM_TILES; ++tile) {
            const int local_y = tile / V3_TILES_X;
            const int local_x = tile % V3_TILES_X;
            if (tile + 2 < V3_NUM_TILES) {
                const int next = tile + 2;
                const int next_y = next / V3_TILES_X;
                const int next_x = next % V3_TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&s_in[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        col0 + next_x * V3_TILE_DIM_X,
                        row0 + next_y * V3_TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(
                &in_mbar[tile], mbar_phase);
            tk_v5_h_tile::normalize_tile(
                s_in_ptr,
                gamma_cache + local_x * V3_TILE_DIM_X,
                r,
                tile);
            __syncthreads();

            v3_rowwise_scaling<false>(
                s_in_ptr, s_out_ptr, s_row_sc_ptr,
                row_encode, local_y, local_x, tile, out_stage);
            v3_colwise_scaling<false>(
                s_in_ptr, s_out_tr_ptr, s_col_sc_ptr,
                col_encode, local_y, local_x, tile, out_tr_stage);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    col0 + local_x * V3_TILE_DIM_X,
                    row0 + local_y * V3_TILE_DIM_Y,
                    reinterpret_cast<uint64_t*>(&s_out[out_stage]));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    row0 + local_y * V3_TILE_DIM_Y,
                    col0 + local_x * V3_TILE_DIM_X,
                    reinterpret_cast<uint64_t*>(&s_out_tr[out_tr_stage]));
                ptx::cp_async_bulk_commit_group();
            }
            out_stage = (out_stage + 1) % V3_BUFFS_NUM_OUT;
            out_tr_stage = (out_tr_stage + 1) % V3_BUFFS_NUM_OUT_TR;
        }

        if (leading) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        tk_v5::swizzle_scales_row_full_inplace(s_row_sc_ptr);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tk_v5::tma_store_scales_2x512(
                tmap_scale_row, s_row_sc_ptr, tile_y, tile_x * 512);
        }

        tk_v5::swizzle_scales_col_full_inplace(s_col_sc_ptr);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tk_v5::tma_store_scales_2x512(
                tmap_scale_col, s_col_sc_ptr, tile_x, tile_y * 512);
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int tile = 0; tile < V3_NUM_TILES; ++tile) {
            ptx::mbarrier_invalid(&in_mbar[tile]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

}  // namespace tk_v5_localcta_h_tile
