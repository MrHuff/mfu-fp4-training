#pragma once

#include "fused_localcta_quantize.cuh"

namespace tk_localcta_fused_direct {

using namespace tk_localcta;
using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;

#if FP4_TYPE_SUPPORTED

template <bool RETURN_TRANSPOSE>
inline int direct_fused_single_shmem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;
    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

template <bool RETURN_TRANSPOSE>
inline int direct_fused_dual_shmem_size() {
    constexpr int in1_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int in2_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;
    return in1_bytes + in2_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <bool RETURN_TRANSPOSE>
inline int direct_fused_cluster_split_shmem_size() {
    constexpr int tile_bytes = DIVUP_TO_MULTIPLE(
        BUFF_DIM_Y * BUFF_DIM_X * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int chunk_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;
    return 3 * tile_bytes + chunk_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <bool RETURN_TRANSPOSE>
inline int direct_fused_cluster_dsmem_split_shmem_size() {
    constexpr int chunk_one = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;
    return 2 * chunk_one + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float warp_max[THREADS / 32];
    __shared__ float block_max;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) {
        warp_max[wid] = val;
    }
    __syncthreads();

    float block_val = (wid == 0 && lane < THREADS / 32) ? warp_max[lane] : 0.0f;
    if (wid == 0) {
        block_val = warp_reduce_max(block_val);
        if (lane == 0) {
            block_max = block_val;
        }
    }
    __syncthreads();
    return block_max;
}

__device__ __forceinline__ void block_reduce_max_pair(float& v0, float& v1) {
    __shared__ float warp_max0[THREADS / 32];
    __shared__ float warp_max1[THREADS / 32];
    __shared__ float block_max0;
    __shared__ float block_max1;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    v0 = warp_reduce_max(v0);
    v1 = warp_reduce_max(v1);
    if (lane == 0) {
        warp_max0[wid] = v0;
        warp_max1[wid] = v1;
    }
    __syncthreads();

    float block_v0 = (wid == 0 && lane < THREADS / 32) ? warp_max0[lane] : 0.0f;
    float block_v1 = (wid == 0 && lane < THREADS / 32) ? warp_max1[lane] : 0.0f;
    if (wid == 0) {
        block_v0 = warp_reduce_max(block_v0);
        block_v1 = warp_reduce_max(block_v1);
        if (lane == 0) {
            block_max0 = block_v0;
            block_max1 = block_v1;
        }
    }
    __syncthreads();
    v0 = block_max0;
    v1 = block_max1;
}

template <int ACTIVE_THREADS>
__device__ __forceinline__ float block_reduce_max_prefix(float val) {
    constexpr int NUM_WARPS = ACTIVE_THREADS / 32;
    __shared__ float warp_max[NUM_WARPS];
    __shared__ float block_max;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    if (threadIdx.x >= ACTIVE_THREADS) {
        val = 0.0f;
    }
    val = warp_reduce_max(val);
    if (threadIdx.x < ACTIVE_THREADS && lane == 0) {
        warp_max[wid] = val;
    }
    __syncthreads();

    float block_val = (wid == 0 && lane < NUM_WARPS) ? warp_max[lane] : 0.0f;
    if (wid == 0) {
        block_val = warp_reduce_max(block_val);
        if (lane == 0) {
            block_max = block_val;
        }
    }
    __syncthreads();
    return block_max;
}

__device__ __forceinline__ void cluster_expect_bytes(uint64_t *mbar, uint32_t bytes) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
                 :
                 : "r"(mbar_addr), "r"(bytes));
#else
    NVTE_DEVICE_ERROR("cluster_expect_bytes is only supported on SM 10.0+.");
#endif
}

__device__ __forceinline__ void cluster_sync_aligned() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile ("barrier.cluster.arrive.release.aligned;\n");
    asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
#else
    NVTE_DEVICE_ERROR("cluster_sync_aligned is only supported on SM 10.0+.");
#endif
}

template <typename T>
__device__ __forceinline__ T* map_peer_shared_ptr(T* local_ptr, int dst_cta) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_ptr));
    uint32_t remote_addr;
    asm volatile(
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(remote_addr)
        : "r"(local_addr), "r"(dst_cta));
    return reinterpret_cast<T*>(__cvta_shared_to_generic(static_cast<size_t>(remote_addr)));
#else
    NVTE_DEVICE_ERROR("map_peer_shared_ptr is only supported on SM 10.0+.");
    return local_ptr;
#endif
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared_multicast(
    uint64_t *dst_shmem, const uint64_t *tensor_map_ptr, const uint32_t offset_x,
    const uint32_t offset_y, uint64_t *mbar, uint16_t cta_mask) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(dst_shmem_ptr), "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(mbar_ptr), "h"(cta_mask)
        : "memory");
#else
    NVTE_DEVICE_ERROR("cp_async_bulk_tensor_2d_global_to_shared_multicast is only supported on SM 10.0+.");
#endif
}

__device__ __forceinline__ void store_chunk_value(
    IType* sIn_ptr,
    int row,
    int col,
    __nv_bfloat16 value
) {
    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    const int tile_y = row / TILE_DIM_Y;
    const int tile_x = col / TILE_DIM_X;
    const int tile = tile_y * TILES_X + tile_x;
    sIn[tile][row % TILE_DIM_Y][col % TILE_DIM_X] = value;
}

template <
    bool RETURN_TRANSPOSE,
    bool ENCODE_CENTRIC = true,
    bool DATA_SR = false,
    bool FAST_DATA_SR = false,
    bool SCALE_SR = false>
__device__ __forceinline__ void quantize_store_prepared_chunk(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr,
    nvfp4_scale_t* sSFcolwise_ptr,
    const CUtensorMap& tensor_map_output,
    const CUtensorMap& tensor_map_output_t,
    const CUtensorMap& tmap_scale_row_prepared,
    const CUtensorMap& tmap_scale_col_prepared,
    float S_enc,
    float sg_val,
    int block_offset_Y,
    int block_offset_X,
    int rows,
    int cols,
    int ctaid_X,
    int ctaid_Y,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
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

        if (t > 0 && leading) {
            ptx::cp_async_bulk_wait_group_read<1>();
        }

        if constexpr (DATA_SR || SCALE_SR) {
            const uint64_t row_rng_subsequence =
                rng_subsequence_base + (static_cast<uint64_t>(t) * 2ull + 0ull) * THREADS + threadIdx.x;
            LocalCTARNGState row_rng;
            if constexpr (!FAST_DATA_SR || SCALE_SR) {
                row_rng.init(rng_seed, row_rng_subsequence, 0);
            }
            uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
            int row_rnd_idx = 4;
            const uint64_t row_fast_sr_base =
                rng_seed ^ row_rng_subsequence ^ 0xd1342543de82ef95ull;
            rowwise_scaling_opt<
                ENCODE_CENTRIC, false, DATA_SR, FAST_DATA_SR, SCALE_SR, false, false>(
                sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                S_enc, stage_Y, stage_X, t, buff_out,
                row_rng, row_random_uint4, row_rnd_idx, row_fast_sr_base);
            if constexpr (RETURN_TRANSPOSE) {
                const uint64_t col_rng_subsequence =
                    rng_subsequence_base + (static_cast<uint64_t>(t) * 2ull + 1ull) * THREADS + threadIdx.x;
                LocalCTARNGState col_rng;
                if constexpr (!FAST_DATA_SR || SCALE_SR) {
                    col_rng.init(rng_seed, col_rng_subsequence, 0);
                }
                uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                int col_rnd_idx = 4;
                const uint64_t col_fast_sr_base =
                    rng_seed ^ col_rng_subsequence ^ 0x94d049bb133111ebull;
                colwise_scaling_opt<
                    ENCODE_CENTRIC, false, DATA_SR, FAST_DATA_SR, SCALE_SR, false, false>(
                    sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                    S_enc, stage_Y, stage_X, t, buff_out_tr,
                    col_rng, col_random_uint4, col_rnd_idx, col_fast_sr_base);
            }
        } else {
            rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc, stage_Y, stage_X, t, buff_out);
            if constexpr (RETURN_TRANSPOSE) {
                colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                                S_enc, stage_Y, stage_X, t, buff_out_tr);
            }
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
        scale_swizzled_scales_inplace(
            sSFrowwise_ptr,
            LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
            sg_val);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tma_store_scales_2x512(
                tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
        }
    }

    if constexpr (RETURN_TRANSPOSE) {
        const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
        swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        scale_swizzled_scales_inplace(
            sSFcolwise_ptr,
            LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
            sg_val);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tma_store_scales_2x512(
                tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
        }
    }

    if (leading) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
}

template <bool RETURN_TRANSPOSE>
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

        if (t > 0 && leading) {
            ptx::cp_async_bulk_wait_group_read<1>();
        }

        rowwise_scaling<true>(sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc, stage_Y, stage_X, t, buff_out);
        if constexpr (RETURN_TRANSPOSE) {
            colwise_scaling<true>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
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

__device__ __forceinline__ float load_silu_split_chunk(
    IType* sIn_ptr,
    const __nv_bfloat16* h1_raw,
    const __nv_bfloat16* h3,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    float local_max = 0.0f;
    constexpr int VEC = 4;
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = (int64_t)global_row * cols + global_col;
            const int2 a = *reinterpret_cast<const int2*>(h1_raw + base);
            const int2 b = *reinterpret_cast<const int2*>(h3 + base);

            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);

            float2 a0f = __bfloat1622float2(a0);
            float2 a1f = __bfloat1622float2(a1);
            const float2 b0f = __bfloat1622float2(b0);
            const float2 b1f = __bfloat1622float2(b1);

            a0f.x = a0f.x / (1.0f + __expf(-a0f.x));
            a0f.y = a0f.y / (1.0f + __expf(-a0f.y));
            a1f.x = a1f.x / (1.0f + __expf(-a1f.x));
            a1f.y = a1f.y / (1.0f + __expf(-a1f.y));

            const __nv_bfloat162 o0 = __float22bfloat162_rn(
                make_float2(a0f.x * b0f.x, a0f.y * b0f.y));
            const __nv_bfloat162 o1 = __float22bfloat162_rn(
                make_float2(a1f.x * b1f.x, a1f.y * b1f.y));

            const float2 o0f = __bfloat1622float2(o0);
            const float2 o1f = __bfloat1622float2(o1);
            local_max = fmaxf(local_max, fabsf(o0f.x));
            local_max = fmaxf(local_max, fabsf(o0f.y));
            local_max = fmaxf(local_max, fabsf(o1f.x));
            local_max = fmaxf(local_max, fabsf(o1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, o0.x);
            store_chunk_value(sIn_ptr, row, col + 1, o0.y);
            store_chunk_value(sIn_ptr, row, col + 2, o1.x);
            store_chunk_value(sIn_ptr, row, col + 3, o1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = (int64_t)global_row * cols + block_offset_X + c;
                    const float a = __bfloat162float(h1_raw[offset]);
                    const float b = __bfloat162float(h3[offset]);
                    out = __float2bfloat16_rn((a / (1.0f + __expf(-a))) * b);
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(out)));
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
    return local_max;
}

__device__ __forceinline__ float load_sqrelu_chunk(
    IType* sIn_ptr,
    const __nv_bfloat16* h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    float local_max = 0.0f;
    constexpr int VEC = 4;
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = (int64_t)global_row * cols + global_col;
            const int2 a = *reinterpret_cast<const int2*>(h1_raw + base);

            const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
            const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);

            const float2 a0f = __bfloat1622float2(a0);
            const float2 a1f = __bfloat1622float2(a1);

            const __nv_bfloat162 o0 = __float22bfloat162_rn(
                make_float2(
                    a0f.x > 0.0f ? a0f.x * a0f.x : 0.0f,
                    a0f.y > 0.0f ? a0f.y * a0f.y : 0.0f));
            const __nv_bfloat162 o1 = __float22bfloat162_rn(
                make_float2(
                    a1f.x > 0.0f ? a1f.x * a1f.x : 0.0f,
                    a1f.y > 0.0f ? a1f.y * a1f.y : 0.0f));

            const float2 o0f = __bfloat1622float2(o0);
            const float2 o1f = __bfloat1622float2(o1);
            local_max = fmaxf(local_max, fabsf(o0f.x));
            local_max = fmaxf(local_max, fabsf(o0f.y));
            local_max = fmaxf(local_max, fabsf(o1f.x));
            local_max = fmaxf(local_max, fabsf(o1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, o0.x);
            store_chunk_value(sIn_ptr, row, col + 1, o0.y);
            store_chunk_value(sIn_ptr, row, col + 2, o1.x);
            store_chunk_value(sIn_ptr, row, col + 3, o1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = (int64_t)global_row * cols + block_offset_X + c;
                    const float v = __bfloat162float(h1_raw[offset]);
                    out = __float2bfloat16_rn(v > 0.0f ? v * v : 0.0f);
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(out)));
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
    return local_max;
}

template <bool OUTPUT_DH1>
__device__ __forceinline__ float transform_silu_deriv_tile_to_chunk(
    IType* sOutChunk_ptr,
    const IType* sDhTile_ptr,
    const IType* sH1Tile_ptr,
    const IType* sH3Tile_ptr,
    int tile_idx,
    int consumer_tid
) {
    using Tile2D = IType[BUFF_DIM_Y][BUFF_DIM_X];
    auto& sOutChunk = *reinterpret_cast<IType3D*>(sOutChunk_ptr);
    const auto& sDhTile = *reinterpret_cast<const Tile2D*>(sDhTile_ptr);
    const auto& sH1Tile = *reinterpret_cast<const Tile2D*>(sH1Tile_ptr);
    const auto& sH3Tile = *reinterpret_cast<const Tile2D*>(sH3Tile_ptr);

    float local_max = 0.0f;
    constexpr int VEC = 4;
    constexpr int TILE_TOTAL = BUFF_DIM_Y * BUFF_DIM_X;

    for (int idx = consumer_tid * VEC; idx < TILE_TOTAL; idx += 128 * VEC) {
        const int row = idx / BUFF_DIM_X;
        const int col = idx % BUFF_DIM_X;

        if (col + (VEC - 1) < BUFF_DIM_X) {
            const int2 d = *reinterpret_cast<const int2*>(&sDhTile[row][col]);
            const int2 a = *reinterpret_cast<const int2*>(&sH3Tile[row][col]);
            const int2 b = *reinterpret_cast<const int2*>(&sH1Tile[row][col]);

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

            sOutChunk[tile_idx][row][col + 0] = out0.x;
            sOutChunk[tile_idx][row][col + 1] = out0.y;
            sOutChunk[tile_idx][row][col + 2] = out1.x;
            sOutChunk[tile_idx][row][col + 3] = out1.y;
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                if (c < BUFF_DIM_X) {
                    const float vd = __bfloat162float(sDhTile[row][c]);
                    const float v1 = __bfloat162float(sH1Tile[row][c]);
                    const float v3 = __bfloat162float(sH3Tile[row][c]);
                    const float sig = 1.0f / (1.0f + __expf(-v1));
                    const float silu_v1 = v1 * sig;
                    float transformed;
                    if constexpr (OUTPUT_DH1) {
                        const float silup_v1 = sig * (1.0f + v1 - silu_v1);
                        transformed = vd * v3 * silup_v1;
                    } else {
                        transformed = vd * silu_v1;
                    }
                    const __nv_bfloat16 out = __float2bfloat16_rn(transformed);
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(out)));
                    sOutChunk[tile_idx][row][c] = out;
                }
            }
        }
    }
    return local_max;
}

__device__ __forceinline__ float load_norm_chunk(
    IType* sIn_ptr,
    const __nv_bfloat16* input,
    const __nv_bfloat16* gamma,
    const float* inv_rms,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X,
    bool with_silu
) {
    float local_max = 0.0f;
    constexpr int VEC = 4;
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = (int64_t)global_row * cols + global_col;
            const int2 x = *reinterpret_cast<const int2*>(input + base);
            const int2 g = *reinterpret_cast<const int2*>(gamma + global_col);
            const float row_inv = inv_rms[global_row];

            const __nv_bfloat162 x0 = *reinterpret_cast<const __nv_bfloat162*>(&x.x);
            const __nv_bfloat162 x1 = *reinterpret_cast<const __nv_bfloat162*>(&x.y);
            const __nv_bfloat162 g0 = *reinterpret_cast<const __nv_bfloat162*>(&g.x);
            const __nv_bfloat162 g1 = *reinterpret_cast<const __nv_bfloat162*>(&g.y);

            float2 x0f = __bfloat1622float2(x0);
            float2 x1f = __bfloat1622float2(x1);
            const float2 g0f = __bfloat1622float2(g0);
            const float2 g1f = __bfloat1622float2(g1);

            x0f.x *= row_inv * g0f.x;
            x0f.y *= row_inv * g0f.y;
            x1f.x *= row_inv * g1f.x;
            x1f.y *= row_inv * g1f.y;
            if (with_silu) {
                x0f.x = x0f.x / (1.0f + __expf(-x0f.x));
                x0f.y = x0f.y / (1.0f + __expf(-x0f.y));
                x1f.x = x1f.x / (1.0f + __expf(-x1f.x));
                x1f.y = x1f.y / (1.0f + __expf(-x1f.y));
            }

            const __nv_bfloat162 o0 = __float22bfloat162_rn(x0f);
            const __nv_bfloat162 o1 = __float22bfloat162_rn(x1f);
            const float2 o0f = __bfloat1622float2(o0);
            const float2 o1f = __bfloat1622float2(o1);
            local_max = fmaxf(local_max, fabsf(o0f.x));
            local_max = fmaxf(local_max, fabsf(o0f.y));
            local_max = fmaxf(local_max, fabsf(o1f.x));
            local_max = fmaxf(local_max, fabsf(o1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, o0.x);
            store_chunk_value(sIn_ptr, row, col + 1, o0.y);
            store_chunk_value(sIn_ptr, row, col + 2, o1.x);
            store_chunk_value(sIn_ptr, row, col + 3, o1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const float x = __bfloat162float(input[(int64_t)global_row * cols + block_offset_X + c]);
                    const float g = __bfloat162float(gamma[block_offset_X + c]);
                    float transformed = x * inv_rms[global_row] * g;
                    if (with_silu) {
                        transformed = transformed / (1.0f + __expf(-transformed));
                    }
                    out = __float2bfloat16_rn(transformed);
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(out)));
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
    return local_max;
}

struct DualLocalMax {
    float a;
    float b;
};

__device__ __forceinline__ DualLocalMax load_silu_deriv_chunk(
    IType* sIn1_ptr,
    IType* sIn2_ptr,
    const __nv_bfloat16* dh,
    const __nv_bfloat16* h3,
    const __nv_bfloat16* h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    DualLocalMax local{0.0f, 0.0f};
    constexpr int VEC = 4;
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = (int64_t)global_row * cols + global_col;
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

            store_chunk_value(sIn1_ptr, row, col + 0, o10.x);
            store_chunk_value(sIn1_ptr, row, col + 1, o10.y);
            store_chunk_value(sIn1_ptr, row, col + 2, o11.x);
            store_chunk_value(sIn1_ptr, row, col + 3, o11.y);
            store_chunk_value(sIn2_ptr, row, col + 0, o20.x);
            store_chunk_value(sIn2_ptr, row, col + 1, o20.y);
            store_chunk_value(sIn2_ptr, row, col + 2, o21.x);
            store_chunk_value(sIn2_ptr, row, col + 3, o21.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out1 = __float2bfloat16_rn(0.0f);
                __nv_bfloat16 out2 = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = (int64_t)global_row * cols + block_offset_X + c;
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

__device__ __forceinline__ float load_sqrelu_deriv_chunk(
    IType* sIn_ptr,
    const __nv_bfloat16* dh,
    const __nv_bfloat16* h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X
) {
    float local_max = 0.0f;
    constexpr int VEC = 4;
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = (int64_t)global_row * cols + global_col;
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

            const __nv_bfloat162 o0 = __float22bfloat162_rn(
                make_float2(
                    a0f.x > 0.0f ? (2.0f * d0f.x) * a0f.x : 0.0f,
                    a0f.y > 0.0f ? (2.0f * d0f.y) * a0f.y : 0.0f));
            const __nv_bfloat162 o1 = __float22bfloat162_rn(
                make_float2(
                    a1f.x > 0.0f ? (2.0f * d1f.x) * a1f.x : 0.0f,
                    a1f.y > 0.0f ? (2.0f * d1f.y) * a1f.y : 0.0f));

            const float2 o0f = __bfloat1622float2(o0);
            const float2 o1f = __bfloat1622float2(o1);
            local_max = fmaxf(local_max, fabsf(o0f.x));
            local_max = fmaxf(local_max, fabsf(o0f.y));
            local_max = fmaxf(local_max, fabsf(o1f.x));
            local_max = fmaxf(local_max, fabsf(o1f.y));

            store_chunk_value(sIn_ptr, row, col + 0, o0.x);
            store_chunk_value(sIn_ptr, row, col + 1, o0.y);
            store_chunk_value(sIn_ptr, row, col + 2, o1.x);
            store_chunk_value(sIn_ptr, row, col + 3, o1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = (int64_t)global_row * cols + block_offset_X + c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v = __bfloat162float(h1_raw[offset]);
                    out = __float2bfloat16_rn(v > 0.0f ? (2.0f * vd) * v : 0.0f);
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(out)));
                }
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
    return local_max;
}

template <bool OUTPUT_DH1>
__device__ __forceinline__ float load_silu_deriv_single_chunk(
    IType* sIn_ptr,
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
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TOTAL; idx += THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = (int64_t)global_row * cols + global_col;
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

            store_chunk_value(sIn_ptr, row, col + 0, out0.x);
            store_chunk_value(sIn_ptr, row, col + 1, out0.y);
            store_chunk_value(sIn_ptr, row, col + 2, out1.x);
            store_chunk_value(sIn_ptr, row, col + 3, out1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset = (int64_t)global_row * cols + block_offset_X + c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    const float sig = 1.0f / (1.0f + __expf(-v1));
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
                store_chunk_value(sIn_ptr, row, c, out);
            }
        }
    }
    __syncthreads();
    return local_max;
}

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS)
localcta_silu_quantize_split_direct_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __nv_bfloat16* __restrict__ h3,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
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
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
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

        float local_max = load_silu_split_chunk(
            sIn_ptr, h1_raw, h3, (int)rows, (int)cols, block_offset_Y, block_offset_X);
        const float amax_val = block_reduce_max(local_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }
        __syncthreads();

        quantize_store_prepared_chunk<RETURN_TRANSPOSE>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            S_enc, sg_val,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS)
localcta_silu_quantize_split_direct_raw_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __nv_bfloat16* __restrict__ h3,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
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
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
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

        float local_max = load_silu_split_chunk(
            sIn_ptr, h1_raw, h3, (int)rows, (int)cols, block_offset_Y, block_offset_X);
        const float amax_val = block_reduce_max(local_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }
        __syncthreads();

        quantize_store_raw_chunk<RETURN_TRANSPOSE>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row, tmap_scale_col,
            S_enc,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
localcta_sqrelu_quantize_direct_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
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
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
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

        float local_max = load_sqrelu_chunk(
            sIn_ptr, h1_raw, (int)rows, (int)cols, block_offset_Y, block_offset_X);
        const float amax_val = block_reduce_max(local_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }
        __syncthreads();

        quantize_store_prepared_chunk<RETURN_TRANSPOSE, ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            S_enc, sg_val,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS)
localcta_norm_quantize_direct_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ inv_rms,
    bool with_silu,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
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
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
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

        float local_max = load_norm_chunk(
            sIn_ptr, input, gamma, inv_rms,
            (int)rows, (int)cols, block_offset_Y, block_offset_X, with_silu);
        const float amax_val = block_reduce_max(local_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }
        __syncthreads();

        quantize_store_prepared_chunk<RETURN_TRANSPOSE>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            S_enc, sg_val,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
localcta_sqrelu_deriv_quantize_direct_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
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
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
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

        float local_max = load_sqrelu_deriv_chunk(
            sIn_ptr, dh, h1_raw, (int)rows, (int)cols, block_offset_Y, block_offset_X);
        const float amax_val = block_reduce_max(local_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }
        __syncthreads();

        quantize_store_prepared_chunk<RETURN_TRANSPOSE, ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            S_enc, sg_val,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <
    bool RETURN_TRANSPOSE,
    bool DATA_SR = false,
    bool FAST_DATA_SR = false,
    bool SCALE_SR = false>
__global__ void __launch_bounds__(THREADS)
localcta_silu_deriv_quantize_split_direct_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output1_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared1,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared1,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks1,
    const __grid_constant__ CUtensorMap tensor_map_output2,
    const __grid_constant__ CUtensorMap tensor_map_output2_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared2,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared2,
    float* __restrict__ row_sg_chunks2,
    float* __restrict__ col_sg_chunks2,
    const size_t rows,
    const size_t cols,
    LocalCTAPersistentArgs args,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in1_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int in2_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;

    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn2_ptr = reinterpret_cast<IType*>(dshmem + in1_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in1_bytes + in2_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in1_bytes + in2_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in1_bytes + in2_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in1_bytes + in2_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

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

        const DualLocalMax local = load_silu_deriv_chunk(
            sIn1_ptr, sIn2_ptr, dh, h3, h1_raw,
            (int)rows, (int)cols, block_offset_Y, block_offset_X);
        float amax1 = local.a;
        float amax2 = local.b;
        block_reduce_max_pair(amax1, amax2);
        const float S_enc1 = compute_localcta_encode_scaling_factor_FP4(amax1);
        const float S_enc2 = compute_localcta_encode_scaling_factor_FP4(amax2);
        const float sg1 = amax1 / localcta_global_scale_num();
        const float sg2 = amax2 / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks1[ctaid_Y * args.tiles_X + ctaid_X] = sg1;
            row_sg_chunks2[ctaid_Y * args.tiles_X + ctaid_X] = sg2;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks1[ctaid_X * tiles_Y + ctaid_Y] = sg1;
                col_sg_chunks2[ctaid_X * tiles_Y + ctaid_Y] = sg2;
            }
        }
        __syncthreads();

        const uint64_t chunk_rng_base =
            rng_subsequence_base +
            static_cast<uint64_t>(s_chunk_id) * 4ull * NUM_TILES * THREADS;

        quantize_store_prepared_chunk<RETURN_TRANSPOSE, true, DATA_SR, FAST_DATA_SR, SCALE_SR>(
            sIn1_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output1, tensor_map_output1_t,
            tmap_scale_row_prepared1, tmap_scale_col_prepared1,
            S_enc1, sg1,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y,
            rng_seed, chunk_rng_base);

        quantize_store_prepared_chunk<RETURN_TRANSPOSE, true, DATA_SR, FAST_DATA_SR, SCALE_SR>(
            sIn2_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output2, tensor_map_output2_t,
            tmap_scale_row_prepared2, tmap_scale_col_prepared2,
            S_enc2, sg2,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y,
            rng_seed, chunk_rng_base + 2ull * NUM_TILES * THREADS);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE, bool DELAYED_SCALING = false>
__global__ void __launch_bounds__(THREADS)
localcta_silu_deriv_quantize_split_direct_raw_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output1_t,
    const __grid_constant__ CUtensorMap tmap_scale_row1,
    const __grid_constant__ CUtensorMap tmap_scale_col1,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks1,
    const __grid_constant__ CUtensorMap tensor_map_output2,
    const __grid_constant__ CUtensorMap tensor_map_output2_t,
    const __grid_constant__ CUtensorMap tmap_scale_row2,
    const __grid_constant__ CUtensorMap tmap_scale_col2,
    float* __restrict__ row_sg_chunks2,
    float* __restrict__ col_sg_chunks2,
    float* __restrict__ amax_out1,
    float* __restrict__ amax_out2,
    const size_t rows,
    const size_t cols,
    LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in1_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int in2_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn2_ptr = reinterpret_cast<IType*>(dshmem + in1_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in1_bytes + in2_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in1_bytes + in2_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in1_bytes + in2_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in1_bytes + in2_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

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

        const DualLocalMax local = load_silu_deriv_chunk(
            sIn1_ptr, sIn2_ptr, dh, h3, h1_raw,
            (int)rows, (int)cols, block_offset_Y, block_offset_X);
        float amax1 = local.a;
        float amax2 = local.b;
        block_reduce_max_pair(amax1, amax2);
        if constexpr (DELAYED_SCALING) {
            if (leading) {
                if (amax_out1 != nullptr) {
                    transformer_engine::atomicMaxFloat(amax_out1, amax1);
                }
                if (amax_out2 != nullptr) {
                    transformer_engine::atomicMaxFloat(amax_out2, amax2);
                }
            }
        }
        const float S_enc1 = DELAYED_SCALING
            ? 1.0f
            : compute_localcta_encode_scaling_factor_FP4(amax1);
        const float S_enc2 = DELAYED_SCALING
            ? 1.0f
            : compute_localcta_encode_scaling_factor_FP4(amax2);
        const float sg1 = DELAYED_SCALING ? 1.0f : amax1 / localcta_global_scale_num();
        const float sg2 = DELAYED_SCALING ? 1.0f : amax2 / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks1[ctaid_Y * args.tiles_X + ctaid_X] = sg1;
            row_sg_chunks2[ctaid_Y * args.tiles_X + ctaid_X] = sg2;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks1[ctaid_X * tiles_Y + ctaid_Y] = sg1;
                col_sg_chunks2[ctaid_X * tiles_Y + ctaid_Y] = sg2;
            }
        }
        __syncthreads();

        quantize_store_raw_chunk<RETURN_TRANSPOSE>(
            sIn1_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output1, tensor_map_output1_t,
            tmap_scale_row1, tmap_scale_col1,
            S_enc1,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);

        quantize_store_raw_chunk<RETURN_TRANSPOSE>(
            sIn2_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output2, tensor_map_output2_t,
            tmap_scale_row2, tmap_scale_col2,
            S_enc2,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE, bool OUTPUT_DH1>
__global__ void __launch_bounds__(THREADS)
localcta_silu_deriv_quantize_single_direct_kernel(
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
    const size_t cols,
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
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
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

        float local_max = load_silu_deriv_single_chunk<OUTPUT_DH1>(
            sIn_ptr, dh, h3, h1_raw, (int)rows, (int)cols, block_offset_Y, block_offset_X);
        const float amax_val = block_reduce_max(local_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }
        __syncthreads();

        quantize_store_prepared_chunk<RETURN_TRANSPOSE>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            S_enc, sg_val,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int TOTAL_THREADS, bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS) __cluster_dims__(2, 1, 1)
localcta_silu_deriv_quantize_split_cluster_multicast_kernel(
    const __grid_constant__ CUtensorMap tensor_map_dh,
    const __grid_constant__ CUtensorMap tensor_map_h1,
    const __grid_constant__ CUtensorMap tensor_map_h3,
    const __grid_constant__ CUtensorMap tensor_map_output0,
    const __grid_constant__ CUtensorMap tensor_map_output_t0,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared0,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared0,
    float* __restrict__ row_sg_chunks0,
    float* __restrict__ col_sg_chunks0,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output_t1,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared1,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared1,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks1,
    const size_t rows, const size_t cols,
    int tiles_X,
    int total_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int CONSUMER_THREADS = 128;
    constexpr uint16_t CLUSTER_MASK = 0x3;
    constexpr int tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);
    constexpr int tile_bytes_aligned = DIVUP_TO_MULTIPLE(tile_bytes, TMA_SHMEM_ALIGNMENT);
    constexpr int chunk_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sDh_tile_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sH1_tile_ptr = reinterpret_cast<IType*>(dshmem + tile_bytes_aligned);
    IType* sH3_tile_ptr = reinterpret_cast<IType*>(dshmem + 2 * tile_bytes_aligned);
    IType* sChunk_ptr = reinterpret_cast<IType*>(dshmem + 3 * tile_bytes_aligned);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 3 * tile_bytes_aligned + chunk_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 3 * tile_bytes_aligned + chunk_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 3 * tile_bytes_aligned + chunk_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 3 * tile_bytes_aligned + chunk_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sChunk = *reinterpret_cast<IType3D*>(sChunk_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t dh_mbar;
    __shared__ uint64_t h1_mbar;
    __shared__ uint64_t h3_mbar;
    if (threadIdx.x == 0) {
        ptx::mbarrier_init(&dh_mbar, 1);
        ptx::mbarrier_init(&h1_mbar, 1);
        ptx::mbarrier_init(&h3_mbar, 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int cta_rank = blockIdx.x & 1;
    const int pair_id = blockIdx.x >> 1;
    const int num_pairs = max(1, (int)(gridDim.x >> 1));
    const int pair_iters = (pair_id < total_tiles)
        ? ((total_tiles - 1 - pair_id) / num_pairs + 1)
        : 0;

    for (int iter = 0; iter < pair_iters; ++iter) {
        const int tile_id = pair_id + iter * num_pairs;
        const int ctaid_Y = tile_id / tiles_X;
        const int ctaid_X = tile_id % tiles_X;
        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
        const int chunk_cols = static_cast<int>(cols) - block_offset_X;

        float local_max = 0.0f;
        int phase = 0;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int tile_offset_Y = block_offset_Y + stage_Y * TILE_DIM_Y;
            const int tile_offset_X = block_offset_X + stage_X * TILE_DIM_X;

            if (producer_leader) {
                cluster_expect_bytes(&dh_mbar, tile_bytes);
                cluster_expect_bytes(&h1_mbar, tile_bytes);
                cluster_expect_bytes(&h3_mbar, tile_bytes);
            }
            cluster_sync_aligned();

            if (cta_rank == 0 && producer_leader) {
                cp_async_bulk_tensor_2d_global_to_shared_multicast(
                    reinterpret_cast<uint64_t*>(sDh_tile_ptr),
                    reinterpret_cast<const uint64_t*>(&tensor_map_dh),
                    tile_offset_X, tile_offset_Y, &dh_mbar, CLUSTER_MASK);
                cp_async_bulk_tensor_2d_global_to_shared_multicast(
                    reinterpret_cast<uint64_t*>(sH1_tile_ptr),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                    tile_offset_X, tile_offset_Y, &h1_mbar, CLUSTER_MASK);
                cp_async_bulk_tensor_2d_global_to_shared_multicast(
                    reinterpret_cast<uint64_t*>(sH3_tile_ptr),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                    tile_offset_X, tile_offset_Y, &h3_mbar, CLUSTER_MASK);
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar, phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar, phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar, phase);

            if (is_consumer) {
                if (cta_rank == 0) {
                    local_max = fmaxf(local_max, transform_silu_deriv_tile_to_chunk<true>(
                        sChunk_ptr, sDh_tile_ptr, sH1_tile_ptr, sH3_tile_ptr, t, consumer_tid));
                } else {
                    local_max = fmaxf(local_max, transform_silu_deriv_tile_to_chunk<false>(
                        sChunk_ptr, sDh_tile_ptr, sH1_tile_ptr, sH3_tile_ptr, t, consumer_tid));
                }
            }
            __syncthreads();
            cluster_sync_aligned();
            phase ^= 1;
        }

        const float amax_val = block_reduce_max_prefix<CONSUMER_THREADS>(local_max);
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();
        if (consumer_leader) {
            if (cta_rank == 0) {
                row_sg_chunks0[ctaid_Y * tiles_X + ctaid_X] = sg_val;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks0[ctaid_X * tiles_Y_full + ctaid_Y] = sg_val;
                }
            } else {
                row_sg_chunks1[ctaid_Y * tiles_X + ctaid_X] = sg_val;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks1[ctaid_X * tiles_Y_full + ctaid_Y] = sg_val;
                }
            }
        }

        int buff_out = 0;
        int buff_out_tr = 0;
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int stage_offset_Y = stage_Y * TILE_DIM_Y;
            const int stage_offset_X = stage_X * TILE_DIM_X;

            if (t > 0 && consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<1>();
            }

            rowwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                sChunk_ptr, sOut_ptr, sSFrowwise_ptr,
                S_enc, stage_Y, stage_X, t, buff_out, consumer_tid);
            if constexpr (RETURN_TRANSPOSE) {
                colwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                    sChunk_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                    S_enc, stage_Y, stage_X, t, buff_out_tr, consumer_tid);
            }

            subgroup_barrier_sync<CONSUMER_THREADS>();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if (consumer_leader) {
                if (cta_rank == 0) {
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output0),
                        block_offset_X + stage_offset_X,
                        block_offset_Y + stage_offset_Y,
                        reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                    if constexpr (RETURN_TRANSPOSE) {
                        ptx::cp_async_bulk_tensor_2d_shared_to_global(
                            reinterpret_cast<const uint64_t*>(&tensor_map_output_t0),
                            block_offset_Y_tr + stage_offset_X,
                            block_offset_X_tr + stage_offset_Y,
                            reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                    }
                } else {
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output1),
                        block_offset_X + stage_offset_X,
                        block_offset_Y + stage_offset_Y,
                        reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                    if constexpr (RETURN_TRANSPOSE) {
                        ptx::cp_async_bulk_tensor_2d_shared_to_global(
                            reinterpret_cast<const uint64_t*>(&tensor_map_output_t1),
                            block_offset_Y_tr + stage_offset_X,
                            block_offset_X_tr + stage_offset_Y,
                            reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                    }
                }
                ptx::cp_async_bulk_commit_group();
            }

            buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
            buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
        }

        if (consumer_leader) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<CONSUMER_THREADS>();

        swizzle_scales_row_inplace_group<CONSUMER_THREADS>(
            sSFrowwise_ptr,
            min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM),
            consumer_tid);
        scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
            sSFrowwise_ptr,
            LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
            sg_val,
            consumer_tid);
        subgroup_barrier_sync<CONSUMER_THREADS>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        subgroup_barrier_sync<CONSUMER_THREADS>();
        if (consumer_leader) {
            if (cta_rank == 0) {
                tma_store_scales_2x512(
                    tmap_scale_row_prepared0, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            } else {
                tma_store_scales_2x512(
                    tmap_scale_row_prepared1, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            if (consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
            swizzle_scales_col_inplace_group<CONSUMER_THREADS>(
                sSFcolwise_ptr,
                min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM),
                consumer_tid);
            scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                sSFcolwise_ptr,
                LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                sg_val,
                consumer_tid);
            subgroup_barrier_sync<CONSUMER_THREADS>();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            subgroup_barrier_sync<CONSUMER_THREADS>();
            if (consumer_leader) {
                if (cta_rank == 0) {
                    tma_store_scales_2x512(
                        tmap_scale_col_prepared0, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                } else {
                    tma_store_scales_2x512(
                        tmap_scale_col_prepared1, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
            }
        }

        if (consumer_leader) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<CONSUMER_THREADS>();
        cluster_sync_aligned();
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS) __cluster_dims__(2, 1, 1)
localcta_silu_deriv_quantize_split_cluster_dsmem_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output0,
    const __grid_constant__ CUtensorMap tensor_map_output_t0,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared0,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared0,
    float* __restrict__ row_sg_chunks0,
    float* __restrict__ col_sg_chunks0,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output_t1,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared1,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared1,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks1,
    const size_t rows,
    const size_t cols,
    int tiles_X,
    int total_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int chunk_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    IType* sIn0_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem + chunk_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * chunk_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * chunk_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * chunk_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * chunk_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    const int cta_rank = blockIdx.x & 1;
    const int pair_id = blockIdx.x >> 1;
    const int num_pairs = max(1, (int)(gridDim.x >> 1));
    const int pair_iters = (pair_id < total_tiles)
        ? ((total_tiles - 1 - pair_id) / num_pairs + 1)
        : 0;

    for (int iter = 0; iter < pair_iters; ++iter) {
        const int tile_id = pair_id + iter * num_pairs;
        const int ctaid_Y = tile_id / tiles_X;
        const int ctaid_X = tile_id % tiles_X;
        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

        if (cta_rank == 0) {
            load_silu_deriv_chunk(
                sIn0_ptr, sIn1_ptr, dh, h3, h1_raw,
                (int)rows, (int)cols, block_offset_Y, block_offset_X);
        }
        __syncthreads();
        cluster_sync_aligned();

        if (cta_rank == 1) {
            copy_peer_shared_buffer_to_local<IType, THREADS>(
                sIn1_ptr, sIn1_ptr, 0, BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType));
        }

        const IType* sRead_ptr = (cta_rank == 0) ? sIn0_ptr : sIn1_ptr;
        float cta_max = 0.0f;
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            cta_max = fmaxf(cta_max, scan_tile_amax(sRead_ptr, t));
        }
        cta_max = block_reduce_max(cta_max);
        const float amax_val = cta_max;
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            if (cta_rank == 0) {
                row_sg_chunks0[ctaid_Y * tiles_X + ctaid_X] = sg_val;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks0[ctaid_X * tiles_Y_full + ctaid_Y] = sg_val;
                }
            } else {
                row_sg_chunks1[ctaid_Y * tiles_X + ctaid_X] = sg_val;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks1[ctaid_X * tiles_Y_full + ctaid_Y] = sg_val;
                }
            }
        }
        __syncthreads();

        if (cta_rank == 0) {
            quantize_store_prepared_chunk<RETURN_TRANSPOSE>(
                sIn0_ptr, sOut_ptr, sOut_tr_ptr,
                sSFrowwise_ptr, sSFcolwise_ptr,
                tensor_map_output0, tensor_map_output_t0,
                tmap_scale_row_prepared0, tmap_scale_col_prepared0,
                S_enc, sg_val,
                block_offset_Y, block_offset_X,
                (int)rows, (int)cols,
                ctaid_X, ctaid_Y);
        } else {
            quantize_store_prepared_chunk<RETURN_TRANSPOSE>(
                sIn1_ptr, sOut_ptr, sOut_tr_ptr,
                sSFrowwise_ptr, sSFcolwise_ptr,
                tensor_map_output1, tensor_map_output_t1,
                tmap_scale_row_prepared1, tmap_scale_col_prepared1,
                S_enc, sg_val,
                block_offset_Y, block_offset_X,
                (int)rows, (int)cols,
                ctaid_X, ctaid_Y);
        }
        cluster_sync_aligned();
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace tk_localcta_fused_direct
