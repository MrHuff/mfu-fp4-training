// persistent_quantize_col_only.cuh — Col-only Phase2 persistent kernel
//
// Produces ONLY col FP4 + col scales (no row output).
// Takes pre-computed sg from device memory (from a prior row-only quant call).
//
// IMPORTANT: Uses the SAME SMEM layout as the full RETURN_TRANSPOSE=true kernel
// (allocates row output buffers but skips row computation and TMA stores).
// This ensures identical occupancy and SMEM addresses, guaranteeing
// bit-identical col output vs the full kernel.
//
#pragma once
#include "fused_amax_quantize.cuh"
#include "persistent_quantize.cuh"

namespace tk_v5_col_only {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;
using namespace tk_v5;

struct Phase2ColOnlyArgs {
    unsigned int* work_counter;
    float*        sg_ptr;       // pre-computed sg on device (sg = amax / 2688)
    int tiles_X, tiles_Y, total_tiles;
};


// ─── Pipelined col-only quantize helper ───
// Uses full SMEM layout but skips v3_rowwise_scaling and row TMA stores.
template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void quantize_and_store_chunk_col_only_pipelined(
    IType* sIn_ptr, fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFcolwise_ptr,
    V3_OType2xt3D& sOut_tr,
    V3_ScalesTypeTr2D& sSFcolwise,
    const CUtensorMap& tensor_map_output_t,
    const CUtensorMap& tmap_scale_col,
    float S_enc,
    int block_offset_Y, int block_offset_X,
    int rows, int cols,
    int ctaid_X, int ctaid_Y,
    uint64_t* in_mbar, int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_rows = (int)rows - block_offset_Y;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);

    int buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        // Prefetch tile t+2
        if (t + 2 < V3_NUM_TILES) {
            const int next = t + 2;
            const int nty = next / V3_TILES_X, ntx = next % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * V3_TILE_DIM_X,
                    block_offset_Y + nty * V3_TILE_DIM_Y,
                    &in_mbar[next]);
            }
        }

        // Wait for current tile
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        // Col-only: skip v3_rowwise_scaling, only do v3_colwise_scaling
        v3_colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out_tr);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store col FP4 only (skip row FP4 store)
        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                block_offset_X_tr + stage_offset_Y,
                block_offset_Y_tr + stage_offset_X,
                reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            ptx::cp_async_bulk_commit_group();
        }

        buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
    }

    // Wait for all FP4 TMA stores
    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    // ─── TMA col scale stores (skip row scales) ───
    {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
        swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm_col = block_offset_X / 128;
            const int tma_x_base = ctaid_Y * 2 * 256;
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                tma_x_base, tm_col,
                reinterpret_cast<uint64_t*>(sSFcolwise_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                tma_x_base + 256, tm_col,
                reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise_ptr) + 512));
            ptx::cp_async_bulk_commit_group();
        }
    }

    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// Phase-2-only col-only persistent kernel
//
// Uses IDENTICAL SMEM layout as the full RETURN_TRANSPOSE=true kernel
// to guarantee same occupancy and bit-identical col output.
// Skips row compute + row TMA stores for speedup.
// ═══════════════════════════════════════════════════════════════════

template <bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
persistent_quantize_phase2_col_only_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const size_t rows, const size_t cols,
    Phase2ColOnlyArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);

    // *** CRITICAL: Use IDENTICAL SMEM layout as full RETURN_TRANSPOSE=true kernel ***
    // This ensures same occupancy and prevents any subtle address-dependent behavior.
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    // Place buffers at SAME offsets as full kernel (skip sOut_ptr, sSFrowwise_ptr)
    IType*         sIn_ptr        = reinterpret_cast<IType*>(dshmem);
    // sOut_ptr would be at (dshmem + in_bytes) — allocated but unused
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    // sSFrowwise_ptr would be at (dshmem + in_bytes + out_bytes + out_tr_bytes) — unused
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn        = *reinterpret_cast<V3_IType3D*>(sIn_ptr);
    auto& sSFcolwise = *reinterpret_cast<V3_ScalesTypeTr2D*>(sSFcolwise_ptr);
    auto& sOut_tr    = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_init(&in_mbar[t], 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Read pre-computed sg from device memory and compute S_enc
    const float sg_val = args.sg_ptr[0];
    const float amax_val = sg_val * 2688.0f;
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);

    int mbar_phase = 0;

    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // Work-stealing quantize loop (col-only, Phase 2 only)
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        // Prefetch first 2 tiles
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

        quantize_and_store_chunk_col_only_pipelined<ENCODE_CENTRIC>(
            sIn_ptr, sOut_tr_ptr,
            sSFcolwise_ptr,
            sOut_tr, sSFcolwise,
            tensor_map_output_t,
            tmap_scale_col,
            S_enc,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y,
            in_mbar, mbar_phase,
            &tensor_map_input);
        mbar_phase ^= 1;
    }

    // Cleanup
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_invalid(&in_mbar[t]);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}


// SMEM size helper — uses SAME size as full RETURN_TRANSPOSE=true kernel
inline int col_only_shmem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

    // Same total as full kernel with RETURN_TRANSPOSE=true
    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

}  // namespace tk_v5_col_only
