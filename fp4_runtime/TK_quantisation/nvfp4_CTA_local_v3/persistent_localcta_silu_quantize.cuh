#pragma once

#include "fused_localcta_quantize.cuh"

namespace tk_localcta_persistent_silu {

using namespace tk_localcta;
using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;

#if FP4_TYPE_SUPPORTED

static constexpr int PRODUCER_CONSUMER_THREADS = 160;
static constexpr int CONSUMER_THREADS = 128;
static constexpr int PIPE_DEPTH = 2;

template <int DEPTH>
using TileRing = IType[DEPTH][BUFF_DIM_Y][BUFF_DIM_X];
using Tile2D = IType[BUFF_DIM_Y][BUFF_DIM_X];

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void rowwise_scaling_tile_group(
    const IType* __restrict__ sTile_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_out,
    int tid
) {
    rowwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
        sTile_ptr, sOut_ptr, sSFrowwise_ptr, S_enc, stage_Y, stage_X, 0, buff_out, tid);
}

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void colwise_scaling_tile_group(
    const IType* __restrict__ sTile_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_out_tr,
    int tid
) {
    colwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
        sTile_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc, stage_Y, stage_X, 0, buff_out_tr, tid);
}

template <int GROUP_THREADS>
__device__ __forceinline__ float reduce_group_max(float val, int tid) {
    __shared__ float warp_max[GROUP_THREADS / 32];
    __shared__ float group_max;

    const int lane = tid & 31;
    const int wid = tid >> 5;

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    if (lane == 0) {
        warp_max[wid] = val;
    }
    subgroup_barrier_sync<GROUP_THREADS>();

    if (wid == 0) {
        float block_val = (lane < GROUP_THREADS / 32) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = (GROUP_THREADS / 32) / 2; mask > 0; mask >>= 1) {
            block_val = fmaxf(block_val, __shfl_xor_sync(0xffffffff, block_val, mask));
        }
        if (lane == 0) {
            group_max = block_val;
        }
    }
    subgroup_barrier_sync<GROUP_THREADS>();
    return group_max;
}

template <int GROUP_THREADS>
__device__ __forceinline__ float transform_silu_tile_inplace_group(
    IType* sTile_h1_ptr,
    const IType* sTile_h3_ptr,
    int consumer_tid
) {
    auto& sH1 = *reinterpret_cast<Tile2D*>(sTile_h1_ptr);
    const auto& sH3 = *reinterpret_cast<const Tile2D*>(sTile_h3_ptr);

    constexpr int THREADS_X = TILE_DIM_X / ELTS_PER_THREAD;
    constexpr int THREADS_Y = GROUP_THREADS / THREADS_X;
    constexpr int ITERS = TILE_DIM_Y / THREADS_Y;

    const int tid_y = consumer_tid / THREADS_X;
    const int tid_x = consumer_tid % THREADS_X;
    const int thread_offset_x = tid_x * ELTS_PER_THREAD;

    float tile_max = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_y + it * THREADS_Y;

        #pragma unroll
        for (int e = 0; e < ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_x + e;
            const float h1_val = __bfloat162float(sH1[local_row][col]);
            const float h3_val = __bfloat162float(sH3[local_row][col]);
            const __nv_bfloat16 out_bf16 =
                __float2bfloat16_rn(h1_val / (1.0f + __expf(-h1_val)) * h3_val);
            const float out = __bfloat162float(out_bf16);
            sH1[local_row][col] = out_bf16;
            tile_max = fmaxf(tile_max, fabsf(out));
        }
    }

    return tile_max;
}

template <bool RETURN_TRANSPOSE>
__device__ __forceinline__ void emit_prepared_tile_group(
    IType* sTile_ptr,
    fp4e2m1x2* sOut_ptr,
    fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr,
    nvfp4_scale_t* sSFcolwise_ptr,
    const CUtensorMap& tensor_map_output,
    const CUtensorMap& tensor_map_output_t,
    float S_enc,
    int tile_idx,
    int block_offset_Y,
    int block_offset_X,
    int consumer_tid
) {
    const bool consumer_leader = (consumer_tid == 0);
    const int stage_Y = tile_idx / TILES_X;
    const int stage_X = tile_idx % TILES_X;
    const int stage_offset_Y = stage_Y * TILE_DIM_Y;
    const int stage_offset_X = stage_X * TILE_DIM_X;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    const int slot = tile_idx % BUFFS_NUM_OUT;
    const int slot_tr = tile_idx % BUFFS_NUM_OUT_TR;

    if (tile_idx > 0 && consumer_leader) {
        ptx::cp_async_bulk_wait_group_read<1>();
    }
    subgroup_barrier_sync<CONSUMER_THREADS>();

    rowwise_scaling_tile_group<CONSUMER_THREADS, true>(
        sTile_ptr, sOut_ptr, sSFrowwise_ptr, S_enc, stage_Y, stage_X, slot, consumer_tid);

    if constexpr (RETURN_TRANSPOSE) {
        colwise_scaling_tile_group<CONSUMER_THREADS, true>(
            sTile_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc, stage_Y, stage_X, slot_tr, consumer_tid);
    }

    subgroup_barrier_sync<CONSUMER_THREADS>();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    subgroup_barrier_sync<CONSUMER_THREADS>();

    if (consumer_leader) {
        auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t*>(&tensor_map_output),
            block_offset_X + stage_offset_X,
            block_offset_Y + stage_offset_Y,
            reinterpret_cast<uint64_t*>(&sOut[slot]));

        if constexpr (RETURN_TRANSPOSE) {
            auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                block_offset_X_tr + stage_offset_Y,
                block_offset_Y_tr + stage_offset_X,
                reinterpret_cast<uint64_t*>(&sOut_tr[slot_tr]));
        }
        ptx::cp_async_bulk_commit_group();
    }
}

template <int TOTAL_THREADS, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(TOTAL_THREADS)
localcta_tma_silu_quantize_kernel(
    const __grid_constant__ CUtensorMap tmap_h1,
    const __grid_constant__ CUtensorMap tmap_h3,
    const __grid_constant__ CUtensorMap tmap_out,
    const __grid_constant__ CUtensorMap tmap_out_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows,
    const size_t cols,
    LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS >= PRODUCER_CONSUMER_THREADS && TOTAL_THREADS % 32 == 0,
                  "localCTA SiLU kernel expects 128 consumer threads plus a producer warp");

    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);
    constexpr int tile_bytes = DIVUP_TO_MULTIPLE(
        BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
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

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_h1_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn_h3_ptr = reinterpret_cast<IType*>(dshmem + PIPE_DEPTH * tile_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * PIPE_DEPTH * tile_bytes);
    fp4e2m1x2* sOut_tr_ptr =
        reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * PIPE_DEPTH * tile_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * PIPE_DEPTH * tile_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * PIPE_DEPTH * tile_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sH1 = *reinterpret_cast<TileRing<PIPE_DEPTH>*>(sIn_h1_ptr);
    auto& sH3 = *reinterpret_cast<TileRing<PIPE_DEPTH>*>(sIn_h3_ptr);

    __shared__ uint64_t h1_mbar[NUM_TILES];
    __shared__ uint64_t h3_mbar[NUM_TILES];

    const int preload_tiles = (NUM_TILES < PIPE_DEPTH) ? NUM_TILES : PIPE_DEPTH;
    const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (threadIdx.x == 0) {
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

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                ptx::mbarrier_init(&h1_mbar[t], 1);
                ptx::mbarrier_init(&h3_mbar[t], 1);
            }
            ptx::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        #pragma unroll
        for (int pre = 0; pre < preload_tiles; ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            const int slot = pre % PIPE_DEPTH;
            if (producer_leader) {
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[slot]),
                    reinterpret_cast<const uint64_t*>(&tmap_h1),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &h1_mbar[pre]);

                ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH3[slot]),
                    reinterpret_cast<const uint64_t*>(&tmap_h3),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &h3_mbar[pre]);
            }
        }
        __syncthreads();

        float cta_max = 0.0f;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int slot = t % PIPE_DEPTH;
            if (is_consumer) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], 0);
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], 0);
                cta_max = fmaxf(
                    cta_max,
                    transform_silu_tile_inplace_group<CONSUMER_THREADS>(
                        &sH1[slot][0][0], &sH3[slot][0][0], consumer_tid));
            }
            __syncthreads();

            if (t + PIPE_DEPTH < NUM_TILES) {
                const int next = t + PIPE_DEPTH;
                const int ty = next / TILES_X;
                const int tx = next % TILES_X;
                if (producer_leader) {
                    ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH1[slot]),
                        reinterpret_cast<const uint64_t*>(&tmap_h1),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &h1_mbar[next]);

                    ptx::mbarrier_arrive_expect_tx(&h3_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH3[slot]),
                        reinterpret_cast<const uint64_t*>(&tmap_h3),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &h3_mbar[next]);
                }
            }
            __syncthreads();
        }

        if (is_consumer) {
            cta_max = reduce_group_max<CONSUMER_THREADS>(cta_max, consumer_tid);
        }
        __syncthreads();

        const float S_enc = compute_localcta_encode_scaling_factor_FP4(cta_max);
        const float sg = cta_max / localcta_global_scale_num();

        if (consumer_leader) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg;
            if constexpr (RETURN_TRANSPOSE) {
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                ptx::mbarrier_init(&h1_mbar[t], 1);
                ptx::mbarrier_init(&h3_mbar[t], 1);
            }
            ptx::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        #pragma unroll
        for (int pre = 0; pre < preload_tiles; ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            const int slot = pre % PIPE_DEPTH;
            if (producer_leader) {
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[slot]),
                    reinterpret_cast<const uint64_t*>(&tmap_h1),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &h1_mbar[pre]);

                ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH3[slot]),
                    reinterpret_cast<const uint64_t*>(&tmap_h3),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &h3_mbar[pre]);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int slot = t % PIPE_DEPTH;
            if (is_consumer) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], 0);
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], 0);
                transform_silu_tile_inplace_group<CONSUMER_THREADS>(
                    &sH1[slot][0][0], &sH3[slot][0][0], consumer_tid);
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
            if (is_consumer) {
                emit_prepared_tile_group<RETURN_TRANSPOSE>(
                    &sH1[slot][0][0],
                    sOut_ptr,
                    sOut_tr_ptr,
                    sSFrowwise_ptr,
                    sSFcolwise_ptr,
                    tmap_out,
                    tmap_out_t,
                    S_enc,
                    t,
                    block_offset_Y,
                    block_offset_X,
                    consumer_tid);
            }
            __syncthreads();

            if (t + PIPE_DEPTH < NUM_TILES) {
                const int next = t + PIPE_DEPTH;
                const int ty = next / TILES_X;
                const int tx = next % TILES_X;
                if (producer_leader) {
                    ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH1[slot]),
                        reinterpret_cast<const uint64_t*>(&tmap_h1),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &h1_mbar[next]);

                    ptx::mbarrier_arrive_expect_tx(&h3_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH3[slot]),
                        reinterpret_cast<const uint64_t*>(&tmap_h3),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &h3_mbar[next]);
                }
            }
            __syncthreads();
        }

        if (is_consumer) {
            if (consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            swizzle_scales_row_inplace_group<CONSUMER_THREADS>(
                sSFrowwise_ptr,
                min((int)SCALES_PER_CHUNK_X, static_cast<int>(cols - block_offset_X) / SCALE_DIM),
                consumer_tid);
            scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                sSFrowwise_ptr,
                LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                sg,
                consumer_tid);
            subgroup_barrier_sync<CONSUMER_THREADS>();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            subgroup_barrier_sync<CONSUMER_THREADS>();
            if (consumer_leader) {
                tma_store_scales_2x512(
                    tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            }

            if constexpr (RETURN_TRANSPOSE) {
                if (consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
                swizzle_scales_col_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise_ptr,
                    min((int)SCALES_PER_CHUNK_Y, static_cast<int>(rows - block_offset_Y) / SCALE_DIM),
                    consumer_tid);
                scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise_ptr,
                    LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                    sg,
                    consumer_tid);
                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (consumer_leader) {
                    tma_store_scales_2x512(
                        tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
            }

            if (consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                ptx::mbarrier_invalid(&h1_mbar[t]);
                ptx::mbarrier_invalid(&h3_mbar[t]);
            }
        }
        __syncthreads();
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
inline int persistent_localcta_silu_quant_smem_size() {
    constexpr int tile_bytes = DIVUP_TO_MULTIPLE(
        BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
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

    return 2 * PIPE_DEPTH * tile_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

#endif

}  // namespace tk_localcta_persistent_silu
