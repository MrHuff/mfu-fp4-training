/*************************************************************************
 * Slim Persistent Fused SiLU Derivative + Dual Quantize Kernel (v2)
 *
 * Processes 128×128 chunks (4 tiles) using the proven persistent_quantize
 * infrastructure. For each sub-tile:
 *   1. TMA load dh, h1, h3 into a temp buffer (sequentially, reused)
 *   2. Compute silu_deriv in SMEM (dh1 → sIn1[t], dh3 → sIn2[t])
 *   3. After all 4 tiles: quantize_and_store_chunk for dh1 and dh3
 *
 * Key design: reuses v3_rowwise_scaling, swizzle_scales_inplace, and
 * TMA scale stores from persistent_quantize — 100% proven code path.
 *
 * SMEM budget:
 *   sIn1: 4 × 64×64 × 2B = 32 KB  (dh1 results, quantize reads)
 *   sIn2: 4 × 64×64 × 2B = 32 KB  (dh3 results, quantize reads)
 *   tmp:  1 × 64×64 × 2B =  8 KB  (temp for sequential dh/h1/h3 loads)
 *   FP4 out (2×2KB) + col (2×1KB) + scales (2×1KB) = ~8 KB
 *   Total: ~80 KB → ~2 CTAs/SM (228KB on GB200)
 *
 * Target: SM100 (GB200), compiled with -arch sm_100a
 *************************************************************************/

#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"
#include "persistent_quantize.cuh"   // for quantize_and_store_chunk_v5, swizzle_*

namespace slim_silu_deriv_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

// ─── Args for slim phase2 kernel ───
struct SlimPhase2Args {
    unsigned int* work_counter;
    float*        global_amax1;
    float*        global_amax2;
    int tiles_X, tiles_Y, total_tiles;
    float* sg_output;         // writes sg[0] = amax1/2688, sg[1] = amax2/2688
    // Scale output pointers (unused in v2 — scales via TMA maps)
    nvfp4_scale_t* sc_row1_ptr;
    nvfp4_scale_t* sc_row2_ptr;
    nvfp4_scale_t* sc_col1_ptr;
    nvfp4_scale_t* sc_col2_ptr;
    int64_t sc_row_stride;
    int64_t sc_col_stride;
};

// ─── SMEM size calculation ───
template <bool RETURN_TRANSPOSE>
inline int slim_silu_deriv_quant_smem_size() {
    // 2 × 4-tile input buffers (dh1 and dh3 results)
    constexpr int in_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * (int)sizeof(IType);
    constexpr int in_one_set = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * in_tile_bytes, TMA_SHMEM_ALIGNMENT);
    constexpr int in_total = 2 * in_one_set;  // dh1 + dh3

    // 1 temp buffer for sequential loads (dh, h1, h3)
    constexpr int tmp_bytes = DIVUP_TO_MULTIPLE(in_tile_bytes, TMA_SHMEM_ALIGNMENT);

    // FP4 row output (double-buffered)
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    // FP4 col output (double-buffered, optional)
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE,
                          TMA_SHMEM_ALIGNMENT) : 0;

    // Scales
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ? DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT) : 0;

    return in_total + tmp_bytes + out_bytes + out_tr_bytes +
           sc_row_bytes + sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

// ─── Inline silu-deriv: transforms tile in-place ───
// Reads from sTmp (dh), sH1 (h1), sH3 (h3)
// Writes dh1 to sDst1, dh3 to sDst2
__device__ __forceinline__ void compute_silu_deriv_tile(
    const IType* __restrict__ sTmp_dh,
    const IType* __restrict__ sTmp_h1,
    const IType* __restrict__ sTmp_h3,
    IType* __restrict__ sDst1,   // output: dh1
    IType* __restrict__ sDst2    // output: dh3
) {
    constexpr int TILE = V3_TILE_DIM_Y * V3_TILE_DIM_X;  // 4096
    constexpr int ELTS_PER_THREAD = TILE / V3_THREADS;     // 32

    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; ++i) {
        const int idx = threadIdx.x + i * V3_THREADS;

        float dh_val = __bfloat162float(sTmp_dh[idx]);
        float h1_val = __bfloat162float(sTmp_h1[idx]);
        float h3_val = __bfloat162float(sTmp_h3[idx]);

        float sig = 1.0f / (1.0f + expf(-h1_val));
        float silu_v = h1_val * sig;
        float silup_v = sig * (1.0f + h1_val - silu_v);

        sDst1[idx] = __float2bfloat16_rn(dh_val * h3_val * silup_v);  // dh1
        sDst2[idx] = __float2bfloat16_rn(dh_val * silu_v);             // dh3
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main kernel: slim persistent silu-deriv + dual quantize
// Uses standard quantize_and_store_chunk_v5 for proven correctness
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
persistent_silu_deriv_quantize_slim_kernel(
    // Input TMA maps
    const __grid_constant__ CUtensorMap tmap_dh,
    const __grid_constant__ CUtensorMap tmap_h1,
    const __grid_constant__ CUtensorMap tmap_h3,
    // Output TMA maps for dh1
    const __grid_constant__ CUtensorMap tmap_out1,
    const __grid_constant__ CUtensorMap tmap_out1_t,
    const __grid_constant__ CUtensorMap tmap_sc_row1,
    const __grid_constant__ CUtensorMap tmap_sc_col1,
    // Output TMA maps for dh3
    const __grid_constant__ CUtensorMap tmap_out2,
    const __grid_constant__ CUtensorMap tmap_out2_t,
    const __grid_constant__ CUtensorMap tmap_sc_row2,
    const __grid_constant__ CUtensorMap tmap_sc_col2,
    // Dimensions
    const size_t rows, const size_t cols,
    SlimPhase2Args args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // ─── SMEM layout ───
    constexpr int in_one_set = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int tmp_aligned = DIVUP_TO_MULTIPLE(tile_bytes, TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE,
                          TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ? DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT) : 0;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    // sIn1[4 tiles] — holds dh1 results for quantize
    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem);
    // sIn2[4 tiles] — holds dh3 results for quantize
    IType* sIn2_ptr = reinterpret_cast<IType*>(dshmem + in_one_set);
    // sTmp[1 tile] — temp for sequential dh/h1/h3 loads
    IType* sTmp = reinterpret_cast<IType*>(dshmem + 2 * in_one_set);

    int off = 2 * in_one_set + tmp_aligned;
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_bytes;
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_tr_bytes;
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);

    auto& sIn1        = *reinterpret_cast<V3_IType3D*>(sIn1_ptr);
    auto& sIn2        = *reinterpret_cast<V3_IType3D*>(sIn2_ptr);
    auto& sSFrowwise  = *reinterpret_cast<V3_ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise  = *reinterpret_cast<V3_ScalesTypeTr2D*>(sSFcolwise_ptr);
    auto& sOut        = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr     = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    // TMA barriers: 1 for temp loads (reused)
    __shared__ uint64_t mbar_tmp;
    if (leading) {
        ptx::mbarrier_init(&mbar_tmp, 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // ─── Read pre-computed amaxes ───
    const float amax1 = args.global_amax1[0];
    const float amax2 = args.global_amax2[0];
    const float S_enc1 = compute_global_encode_scaling_factor_FP4(amax1);
    const float S_enc2 = compute_global_encode_scaling_factor_FP4(amax2);

    if (leading && blockIdx.x == 0 && args.sg_output) {
        args.sg_output[0] = amax1 / 2688.0f;
        args.sg_output[1] = amax2 / 2688.0f;
    }

    int mbar_phase = 0;

    // ─── Work-stealing loop over 128×128 chunks ───
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        // ─── Load + silu_deriv for each of the 4 sub-tiles ───
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int ty = t / V3_TILES_X;
            const int tx = t % V3_TILES_X;
            const int tile_off_X = block_offset_X + tx * V3_TILE_DIM_X;
            const int tile_off_Y = block_offset_Y + ty * V3_TILE_DIM_Y;

            IType* dst1 = &sIn1[t][0][0];  // dh1 destination for this tile
            IType* dst2 = &sIn2[t][0][0];  // dh3 destination for this tile

            // Load dh into sTmp
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&mbar_tmp, tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(sTmp),
                    reinterpret_cast<const uint64_t*>(&tmap_dh),
                    tile_off_X, tile_off_Y, &mbar_tmp);
            }
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&mbar_tmp, mbar_phase);
            mbar_phase ^= 1;

            // Copy dh from sTmp to dst1 (will be overwritten by silu_deriv later)
            // Actually store dh in registers — but we can't do 3 loads to registers
            // with 128 threads and 4096 elements. Instead, copy dh to dst1 temporarily.
            for (int i = threadIdx.x; i < V3_BUFF_IN_ELEMS; i += V3_THREADS)
                dst1[i] = sTmp[i];
            __syncthreads();

            // Load h1 into sTmp
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&mbar_tmp, tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(sTmp),
                    reinterpret_cast<const uint64_t*>(&tmap_h1),
                    tile_off_X, tile_off_Y, &mbar_tmp);
            }
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&mbar_tmp, mbar_phase);
            mbar_phase ^= 1;

            // Copy h1 from sTmp to dst2 temporarily
            for (int i = threadIdx.x; i < V3_BUFF_IN_ELEMS; i += V3_THREADS)
                dst2[i] = sTmp[i];
            __syncthreads();

            // Load h3 into sTmp
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&mbar_tmp, tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(sTmp),
                    reinterpret_cast<const uint64_t*>(&tmap_h3),
                    tile_off_X, tile_off_Y, &mbar_tmp);
            }
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&mbar_tmp, mbar_phase);
            mbar_phase ^= 1;

            // Now: dst1 = dh, dst2 = h1, sTmp = h3
            // Compute silu_deriv: dst1 ← dh1, dst2 ← dh3
            compute_silu_deriv_tile(dst1, dst2, sTmp, dst1, dst2);
            __syncthreads();
        }

        // ─── sIn1 now holds dh1 (4 tiles), sIn2 holds dh3 (4 tiles) ───
        // Quantize dh1 using standard quantize_and_store_chunk_v5
        tk_v5::quantize_and_store_chunk_v5<RETURN_TRANSPOSE>(
            sIn1_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            sOut, sOut_tr, sSFrowwise, sSFcolwise,
            tmap_out1, tmap_out1_t,
            tmap_sc_row1, tmap_sc_col1,
            S_enc1,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);

        // Quantize dh3 using same output SMEM (reused)
        tk_v5::quantize_and_store_chunk_v5<RETURN_TRANSPOSE>(
            sIn2_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            sOut, sOut_tr, sSFrowwise, sSFcolwise,
            tmap_out2, tmap_out2_t,
            tmap_sc_row2, tmap_sc_col2,
            S_enc2,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
    }

    // Clean up
    if (leading) {
        ptx::mbarrier_invalid(&mbar_tmp);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

#endif  // FP4_TYPE_SUPPORTED
}  // namespace slim_silu_deriv_quant
