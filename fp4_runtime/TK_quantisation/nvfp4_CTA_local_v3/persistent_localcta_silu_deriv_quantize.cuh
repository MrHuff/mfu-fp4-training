#pragma once

#include "fused_localcta_quantize.cuh"

namespace tk_localcta_persistent_silu_deriv {

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
__device__ __forceinline__ void dual_rowwise_scaling_tile_group(
    const IType* __restrict__ sTile1_ptr,
    const IType* __restrict__ sTile2_ptr,
    fp4e2m1x2* __restrict__ sOut1_ptr,
    fp4e2m1x2* __restrict__ sOut2_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise1_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise2_ptr,
    const float S_enc1,
    const float S_enc2,
    const int stage_Y, const int stage_X,
    const int buff_out,
    int tid
) {
    rowwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
        sTile1_ptr, sOut1_ptr, sSFrowwise1_ptr, S_enc1, stage_Y, stage_X, 0, buff_out, tid);
    rowwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
        sTile2_ptr, sOut2_ptr, sSFrowwise2_ptr, S_enc2, stage_Y, stage_X, 0, buff_out, tid);
}

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void dual_colwise_scaling_tile_group(
    const IType* __restrict__ sTile1_ptr,
    const IType* __restrict__ sTile2_ptr,
    fp4e2m1x2* __restrict__ sOut1_tr_ptr,
    fp4e2m1x2* __restrict__ sOut2_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise1_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise2_ptr,
    const float S_enc1,
    const float S_enc2,
    const int stage_Y, const int stage_X,
    const int buff_out_tr,
    int tid
) {
    colwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
        sTile1_ptr, sOut1_tr_ptr, sSFcolwise1_ptr, S_enc1, stage_Y, stage_X, 0, buff_out_tr, tid);
    colwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
        sTile2_ptr, sOut2_tr_ptr, sSFcolwise2_ptr, S_enc2, stage_Y, stage_X, 0, buff_out_tr, tid);
}

template <int GROUP_THREADS>
__device__ __forceinline__ void reduce_group_max_pair(float& max1, float& max2, int tid) {
    __shared__ float warp_max1[GROUP_THREADS / 32];
    __shared__ float warp_max2[GROUP_THREADS / 32];
    __shared__ float group_max1;
    __shared__ float group_max2;

    const int lane = tid & 31;
    const int wid = tid >> 5;

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, mask));
        max2 = fmaxf(max2, __shfl_xor_sync(0xffffffff, max2, mask));
    }
    if (lane == 0) {
        warp_max1[wid] = max1;
        warp_max2[wid] = max2;
    }
    subgroup_barrier_sync<GROUP_THREADS>();

    if (wid == 0) {
        float block_max1 = (lane < GROUP_THREADS / 32) ? warp_max1[lane] : 0.0f;
        float block_max2 = (lane < GROUP_THREADS / 32) ? warp_max2[lane] : 0.0f;
        #pragma unroll
        for (int mask = (GROUP_THREADS / 32) / 2; mask > 0; mask >>= 1) {
            block_max1 = fmaxf(block_max1, __shfl_xor_sync(0xffffffff, block_max1, mask));
            block_max2 = fmaxf(block_max2, __shfl_xor_sync(0xffffffff, block_max2, mask));
        }
        if (lane == 0) {
            group_max1 = block_max1;
            group_max2 = block_max2;
        }
    }
    subgroup_barrier_sync<GROUP_THREADS>();
    max1 = group_max1;
    max2 = group_max2;
}

__device__ __forceinline__ void compute_silu_and_deriv_local(
    float x, float& silu_out, float& silup_out
) {
    const float sig = 1.0f / (1.0f + __expf(-x));
    silu_out = x * sig;
    silup_out = sig * (1.0f + x - silu_out);
}

template <int GROUP_THREADS>
__device__ __forceinline__ float2 transform_silu_deriv_tile_inplace_group(
    const IType* sTile_dh_ptr,
    IType* sTile_h1_ptr,
    IType* sTile_h3_ptr,
    int consumer_tid
) {
    const auto& sDh = *reinterpret_cast<const Tile2D*>(sTile_dh_ptr);
    auto& sH1 = *reinterpret_cast<Tile2D*>(sTile_h1_ptr);
    auto& sH3 = *reinterpret_cast<Tile2D*>(sTile_h3_ptr);

    constexpr int THREADS_X = TILE_DIM_X / ELTS_PER_THREAD;
    constexpr int THREADS_Y = GROUP_THREADS / THREADS_X;
    constexpr int ITERS = TILE_DIM_Y / THREADS_Y;

    const int tid_y = consumer_tid / THREADS_X;
    const int tid_x = consumer_tid % THREADS_X;
    const int thread_offset_x = tid_x * ELTS_PER_THREAD;

    float tile_max1 = 0.0f;
    float tile_max2 = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_y + it * THREADS_Y;

        #pragma unroll
        for (int e = 0; e < ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_x + e;
            const float dh_val = __bfloat162float(sDh[local_row][col]);
            const float h1_val = __bfloat162float(sH1[local_row][col]);
            const float h3_val = __bfloat162float(sH3[local_row][col]);

            float silu_v = 0.0f;
            float silup_v = 0.0f;
            compute_silu_and_deriv_local(h1_val, silu_v, silup_v);

            const __nv_bfloat16 dh1_bf16 = __float2bfloat16_rn(dh_val * h3_val * silup_v);
            const __nv_bfloat16 dh3_bf16 = __float2bfloat16_rn(dh_val * silu_v);
            const float dh1_val = __bfloat162float(dh1_bf16);
            const float dh3_val = __bfloat162float(dh3_bf16);

            sH1[local_row][col] = dh1_bf16;
            sH3[local_row][col] = dh3_bf16;
            tile_max1 = fmaxf(tile_max1, fabsf(dh1_val));
            tile_max2 = fmaxf(tile_max2, fabsf(dh3_val));
        }
    }

    return make_float2(tile_max1, tile_max2);
}

template <bool RETURN_TRANSPOSE>
__device__ __forceinline__ void emit_dual_prepared_tile_group(
    IType* sTile1_ptr,
    IType* sTile2_ptr,
    fp4e2m1x2* sOut1_ptr,
    fp4e2m1x2* sOut2_ptr,
    fp4e2m1x2* sOut1_tr_ptr,
    fp4e2m1x2* sOut2_tr_ptr,
    nvfp4_scale_t* sSFrowwise1_ptr,
    nvfp4_scale_t* sSFrowwise2_ptr,
    nvfp4_scale_t* sSFcolwise1_ptr,
    nvfp4_scale_t* sSFcolwise2_ptr,
    const CUtensorMap& tensor_map_output1,
    const CUtensorMap& tensor_map_output2,
    const CUtensorMap& tensor_map_output1_t,
    const CUtensorMap& tensor_map_output2_t,
    float S_enc1,
    float S_enc2,
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
        // Keep one prior dual-output transfer group in flight, matching the
        // double-buffered row/col staging slots instead of fully draining.
        ptx::cp_async_bulk_wait_group_read<1>();
    }
    subgroup_barrier_sync<CONSUMER_THREADS>();

    dual_rowwise_scaling_tile_group<CONSUMER_THREADS, true>(
        sTile1_ptr, sTile2_ptr, sOut1_ptr, sOut2_ptr,
        sSFrowwise1_ptr, sSFrowwise2_ptr, S_enc1, S_enc2,
        stage_Y, stage_X, slot, consumer_tid);

    if constexpr (RETURN_TRANSPOSE) {
        dual_colwise_scaling_tile_group<CONSUMER_THREADS, true>(
            sTile1_ptr, sTile2_ptr, sOut1_tr_ptr, sOut2_tr_ptr,
            sSFcolwise1_ptr, sSFcolwise2_ptr, S_enc1, S_enc2,
            stage_Y, stage_X, slot_tr, consumer_tid);
    }

    subgroup_barrier_sync<CONSUMER_THREADS>();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    subgroup_barrier_sync<CONSUMER_THREADS>();

    if (consumer_leader) {
        auto& sOut1 = *reinterpret_cast<OType2x3D*>(sOut1_ptr);
        auto& sOut2 = *reinterpret_cast<OType2x3D*>(sOut2_ptr);
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t*>(&tensor_map_output1),
            block_offset_X + stage_offset_X,
            block_offset_Y + stage_offset_Y,
            reinterpret_cast<uint64_t*>(&sOut1[slot]));
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t*>(&tensor_map_output2),
            block_offset_X + stage_offset_X,
            block_offset_Y + stage_offset_Y,
            reinterpret_cast<uint64_t*>(&sOut2[slot]));

        if constexpr (RETURN_TRANSPOSE) {
            auto& sOut1_tr = *reinterpret_cast<OType2xt3D*>(sOut1_tr_ptr);
            auto& sOut2_tr = *reinterpret_cast<OType2xt3D*>(sOut2_tr_ptr);
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output1_t),
                block_offset_X_tr + stage_offset_Y,
                block_offset_Y_tr + stage_offset_X,
                reinterpret_cast<uint64_t*>(&sOut1_tr[slot_tr]));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output2_t),
                block_offset_X_tr + stage_offset_Y,
                block_offset_Y_tr + stage_offset_X,
                reinterpret_cast<uint64_t*>(&sOut2_tr[slot_tr]));
        }
        ptx::cp_async_bulk_commit_group();
    }
}

template <int TOTAL_THREADS, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(TOTAL_THREADS)
localcta_tma_silu_deriv_quantize_kernel(
    const __grid_constant__ CUtensorMap tmap_dh,
    const __grid_constant__ CUtensorMap tmap_h1,
    const __grid_constant__ CUtensorMap tmap_h3,
    const __grid_constant__ CUtensorMap tmap_out1,
    const __grid_constant__ CUtensorMap tmap_out2,
    const __grid_constant__ CUtensorMap tmap_out1_t,
    const __grid_constant__ CUtensorMap tmap_out2_t,
    const __grid_constant__ CUtensorMap tmap_scale_row1,
    const __grid_constant__ CUtensorMap tmap_scale_row2,
    const __grid_constant__ CUtensorMap tmap_scale_col1,
    const __grid_constant__ CUtensorMap tmap_scale_col2,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks1,
    float* __restrict__ row_sg_chunks2,
    float* __restrict__ col_sg_chunks2,
    const size_t rows,
    const size_t cols,
    LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS >= PRODUCER_CONSUMER_THREADS && TOTAL_THREADS % 32 == 0,
                  "localCTA SiLU-deriv kernel expects 128 consumer threads plus a producer warp");

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

    IType* sIn_dh_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn_h1_ptr = reinterpret_cast<IType*>(dshmem + PIPE_DEPTH * tile_bytes);
    IType* sIn_h3_ptr = reinterpret_cast<IType*>(dshmem + 2 * PIPE_DEPTH * tile_bytes);
    int off = 3 * PIPE_DEPTH * tile_bytes;
    fp4e2m1x2* sOut1_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_bytes;
    fp4e2m1x2* sOut2_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_bytes;
    fp4e2m1x2* sOut1_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_tr_bytes;
    fp4e2m1x2* sOut2_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_tr_bytes;
    nvfp4_scale_t* sSFrowwise1_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSFrowwise2_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSFcolwise1_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_col_bytes;
    nvfp4_scale_t* sSFcolwise2_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);

    auto& sDh = *reinterpret_cast<TileRing<PIPE_DEPTH>*>(sIn_dh_ptr);
    auto& sH1 = *reinterpret_cast<TileRing<PIPE_DEPTH>*>(sIn_h1_ptr);
    auto& sH3 = *reinterpret_cast<TileRing<PIPE_DEPTH>*>(sIn_h3_ptr);

    __shared__ uint64_t dh_mbar[NUM_TILES];
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
                ptx::mbarrier_init(&dh_mbar[t], 1);
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
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[slot]),
                    reinterpret_cast<const uint64_t*>(&tmap_dh),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &dh_mbar[pre]);

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

        float cta_max1 = 0.0f;
        float cta_max2 = 0.0f;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int slot = t % PIPE_DEPTH;
            if (is_consumer) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], 0);
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], 0);
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], 0);
                const float2 tile_maxes = transform_silu_deriv_tile_inplace_group<CONSUMER_THREADS>(
                    &sDh[slot][0][0], &sH1[slot][0][0], &sH3[slot][0][0], consumer_tid);
                cta_max1 = fmaxf(cta_max1, tile_maxes.x);
                cta_max2 = fmaxf(cta_max2, tile_maxes.y);
            }
            __syncthreads();

            if (t + PIPE_DEPTH < NUM_TILES) {
                const int next = t + PIPE_DEPTH;
                const int ty = next / TILES_X;
                const int tx = next % TILES_X;
                if (producer_leader) {
                    ptx::mbarrier_arrive_expect_tx(&dh_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sDh[slot]),
                        reinterpret_cast<const uint64_t*>(&tmap_dh),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &dh_mbar[next]);

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
            reduce_group_max_pair<CONSUMER_THREADS>(cta_max1, cta_max2, consumer_tid);
        }
        __syncthreads();

        const float S_enc1 = compute_localcta_encode_scaling_factor_FP4(cta_max1);
        const float S_enc2 = compute_localcta_encode_scaling_factor_FP4(cta_max2);
        const float sg1 = cta_max1 / localcta_global_scale_num();
        const float sg2 = cta_max2 / localcta_global_scale_num();

        if (consumer_leader) {
            row_sg_chunks1[ctaid_Y * args.tiles_X + ctaid_X] = sg1;
            row_sg_chunks2[ctaid_Y * args.tiles_X + ctaid_X] = sg2;
            if constexpr (RETURN_TRANSPOSE) {
                col_sg_chunks1[ctaid_X * tiles_Y + ctaid_Y] = sg1;
                col_sg_chunks2[ctaid_X * tiles_Y + ctaid_Y] = sg2;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                ptx::mbarrier_init(&dh_mbar[t], 1);
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
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[slot]),
                    reinterpret_cast<const uint64_t*>(&tmap_dh),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &dh_mbar[pre]);

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
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], 0);
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], 0);
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], 0);
                transform_silu_deriv_tile_inplace_group<CONSUMER_THREADS>(
                    &sDh[slot][0][0], &sH1[slot][0][0], &sH3[slot][0][0], consumer_tid);
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
            if (is_consumer) {
                emit_dual_prepared_tile_group<RETURN_TRANSPOSE>(
                    &sH1[slot][0][0],
                    &sH3[slot][0][0],
                    sOut1_ptr,
                    sOut2_ptr,
                    sOut1_tr_ptr,
                    sOut2_tr_ptr,
                    sSFrowwise1_ptr,
                    sSFrowwise2_ptr,
                    sSFcolwise1_ptr,
                    sSFcolwise2_ptr,
                    tmap_out1,
                    tmap_out2,
                    tmap_out1_t,
                    tmap_out2_t,
                    S_enc1,
                    S_enc2,
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
                    ptx::mbarrier_arrive_expect_tx(&dh_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sDh[slot]),
                        reinterpret_cast<const uint64_t*>(&tmap_dh),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &dh_mbar[next]);

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
                sSFrowwise1_ptr,
                min((int)SCALES_PER_CHUNK_X, static_cast<int>(cols - block_offset_X) / SCALE_DIM),
                consumer_tid);
            swizzle_scales_row_inplace_group<CONSUMER_THREADS>(
                sSFrowwise2_ptr,
                min((int)SCALES_PER_CHUNK_X, static_cast<int>(cols - block_offset_X) / SCALE_DIM),
                consumer_tid);
            scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                sSFrowwise1_ptr,
                LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                sg1,
                consumer_tid);
            scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                sSFrowwise2_ptr,
                LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                sg2,
                consumer_tid);
            subgroup_barrier_sync<CONSUMER_THREADS>();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            subgroup_barrier_sync<CONSUMER_THREADS>();
            if (consumer_leader) {
                tma_store_scales_2x512(
                    tmap_scale_row1, sSFrowwise1_ptr, ctaid_Y, ctaid_X * 2 * 256);
                tma_store_scales_2x512(
                    tmap_scale_row2, sSFrowwise2_ptr, ctaid_Y, ctaid_X * 2 * 256);
            }

            if constexpr (RETURN_TRANSPOSE) {
                if (consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
                swizzle_scales_col_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise1_ptr,
                    min((int)SCALES_PER_CHUNK_Y, static_cast<int>(rows - block_offset_Y) / SCALE_DIM),
                    consumer_tid);
                swizzle_scales_col_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise2_ptr,
                    min((int)SCALES_PER_CHUNK_Y, static_cast<int>(rows - block_offset_Y) / SCALE_DIM),
                    consumer_tid);
                scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise1_ptr,
                    LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                    sg1,
                    consumer_tid);
                scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise2_ptr,
                    LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                    sg2,
                    consumer_tid);
                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (consumer_leader) {
                    tma_store_scales_2x512(
                        tmap_scale_col1, sSFcolwise1_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    tma_store_scales_2x512(
                        tmap_scale_col2, sSFcolwise2_ptr, ctaid_X, ctaid_Y * 2 * 256);
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
                ptx::mbarrier_invalid(&dh_mbar[t]);
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
inline int persistent_localcta_silu_deriv_quant_smem_size() {
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

    return 3 * PIPE_DEPTH * tile_bytes + 2 * out_bytes + 2 * out_tr_bytes +
           2 * sc_row_bytes + 2 * sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

#endif

}  // namespace tk_localcta_persistent_silu_deriv
