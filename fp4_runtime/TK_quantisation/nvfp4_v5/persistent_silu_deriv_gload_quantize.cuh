/*************************************************************************
 * Persistent Fused SiLU Derivative + Quantize via Global Loads
 *
 * Key difference from persistent_silu_deriv_quantize.cuh:
 *   - Inputs (dh, h1, h3) are read via coalesced GLOBAL LOADS, not TMA
 *   - SiLU derivative is computed in registers
 *   - Results written to a single SMEM tile buffer for quantization
 *   - TMA is used only for OUTPUT stores (FP4 data + scales)
 *
 * This avoids the TMA input saturation problem that occurs when 3 TMA
 * input streams compete at large grid sizes.
 *
 * SMEM budget (per tile):
 *   Input tile (reused): 1 x 64x64 x 2B = 8 KB
 *   Out dh1 row:  V3_BUFFS_NUM_OUT x 64x32 x 1B ≈ 4 KB
 *   Out dh3 row:  V3_BUFFS_NUM_OUT x 64x32 x 1B ≈ 4 KB
 *   Out dh1 col:  (if RETURN_TRANSPOSE)           ≈ 4 KB
 *   Out dh3 col:  (if RETURN_TRANSPOSE)           ≈ 4 KB
 *   Scales row x2:                                ≈ 4 KB
 *   Scales col x2:                                ≈ 4 KB
 *   Total: ~32-40 KB → higher occupancy than TMA approach
 *************************************************************************/

#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"
#include "fused_silu_deriv_quantize.cuh"

namespace gload_silu_deriv_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

// ─── Helper: cooperative global load of dh+h13, compute silu_deriv,
//     write ONE output (dh1 or dh3) to SMEM tile, return amax for both ───
//
// output_id=0: writes dh1 to SMEM, output_id=1: writes dh3 to SMEM
// Always returns float2(max_dh1, max_dh3) for amax accumulation.
__device__ __forceinline__ float2 gload_silu_deriv_to_smem(
    const __nv_bfloat16* __restrict__ dh_ptr,   // (M, H) global
    const __nv_bfloat16* __restrict__ h13_ptr,  // (M, 2H) global
    IType* __restrict__ sOut_ptr,                // SMEM tile buffer [64][64]
    int M, int H,
    int global_row_base,   // block_offset_Y + stage_Y * V3_TILE_DIM_Y
    int global_col_base,   // block_offset_X + stage_X * V3_TILE_DIM_X
    int output_id          // 0=dh1, 1=dh3
) {
    auto& sBuf = *reinterpret_cast<IType(*)[V3_BUFF_DIM_Y][V3_BUFF_DIM_X]>(sOut_ptr);

    constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int THREADS_Y = V3_THREADS / THREADS_X;
    constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int col_base = tid_X * V3_ELTS_PER_THREAD;

    float max1 = 0.0f, max2 = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_Y + it * THREADS_Y;
        const int grow = global_row_base + local_row;

        #pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            const int col = col_base + e;
            const int gcol = global_col_base + col;

            // Bounds check (tiles may exceed M or H partially — but
            // we require M%128==0 and H%128==0, so always in-bounds)
            float dh_f  = __bfloat162float(dh_ptr[grow * H + gcol]);
            float h1_f  = __bfloat162float(h13_ptr[grow * (2*H) + gcol]);
            float h3_f  = __bfloat162float(h13_ptr[grow * (2*H) + H + gcol]);

            float silu_v, silup_v;
            fused_silu_deriv_quant::compute_silu_and_deriv(h1_f, silu_v, silup_v);

            float dh1_v = dh_f * h3_f * silup_v;
            float dh3_v = dh_f * silu_v;

            max1 = fmaxf(max1, fabsf(dh1_v));
            max2 = fmaxf(max2, fabsf(dh3_v));

            // Write the selected output to SMEM
            sBuf[local_row][col] = __float2bfloat16_rn(output_id == 0 ? dh1_v : dh3_v);
        }
    }

    return make_float2(max1, max2);
}

// ─── Phase 2 helper: load, compute, quantize, and store a full chunk ───
template <bool RETURN_TRANSPOSE>
__device__ __forceinline__ void gload_quantize_and_store_chunk(
    const __nv_bfloat16* __restrict__ dh_ptr,
    const __nv_bfloat16* __restrict__ h13_ptr,
    IType* sIn_ptr,
    fp4e2m1x2* sOut1_ptr, fp4e2m1x2* sOut2_ptr,
    fp4e2m1x2* sOut1_tr_ptr, fp4e2m1x2* sOut2_tr_ptr,
    nvfp4_scale_t* sSFrowwise1_ptr, nvfp4_scale_t* sSFrowwise2_ptr,
    nvfp4_scale_t* sSFcolwise1_ptr, nvfp4_scale_t* sSFcolwise2_ptr,
    V3_OType2x3D& sOut1, V3_OType2x3D& sOut2,
    V3_OType2xt3D& sOut1_tr, V3_OType2xt3D& sOut2_tr,
    const CUtensorMap& tmap_out1, const CUtensorMap& tmap_out2,
    const CUtensorMap& tmap_out1_t, const CUtensorMap& tmap_out2_t,
    const CUtensorMap& tmap_scale_row1, const CUtensorMap& tmap_scale_row2,
    const CUtensorMap& tmap_scale_col1, const CUtensorMap& tmap_scale_col2,
    float S_enc1, float S_enc2,
    int block_offset_Y, int block_offset_X,
    int rows, int cols, int M, int H,
    int ctaid_X, int ctaid_Y,
    const int64_t ntk_r, const int64_t ntk_c
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_cols = (int)cols - block_offset_X;
    const int chunk_rows = (int)rows - block_offset_Y;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    int buff_out1 = 0, buff_out2 = 0;
    int buff_out1_tr = 0, buff_out2_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;
        const int grow_base = block_offset_Y + stage_offset_Y;
        const int gcol_base = block_offset_X + stage_offset_X;

        // ── Process dh1 ──
        // Global-load dh+h13, compute silu_deriv, write dh1 to SMEM
        gload_silu_deriv_to_smem(dh_ptr, h13_ptr, sIn_ptr, M, H,
                                  grow_base, gcol_base, 0);
        __syncthreads();

        // Quantize dh1 from SMEM
        v3_rowwise_scaling(sIn_ptr, sOut1_ptr, sSFrowwise1_ptr,
                          S_enc1, stage_Y, stage_X, 0, buff_out1);
        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling(sIn_ptr, sOut1_tr_ptr, sSFcolwise1_ptr,
                              S_enc1, stage_Y, stage_X, 0, buff_out1_tr);
        }
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store dh1 outputs
        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_out1),
                block_offset_X + stage_offset_X, block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut1[buff_out1]));
            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out1_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut1_tr[buff_out1_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        // ── Process dh3 ──
        // Global-load (re-reads dh+h13), compute silu_deriv, write dh3 to SMEM
        gload_silu_deriv_to_smem(dh_ptr, h13_ptr, sIn_ptr, M, H,
                                  grow_base, gcol_base, 1);
        __syncthreads();

        // Quantize dh3 from SMEM
        v3_rowwise_scaling(sIn_ptr, sOut2_ptr, sSFrowwise2_ptr,
                          S_enc2, stage_Y, stage_X, 0, buff_out2);
        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling(sIn_ptr, sOut2_tr_ptr, sSFcolwise2_ptr,
                              S_enc2, stage_Y, stage_X, 0, buff_out2_tr);
        }
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store dh3 outputs
        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_out2),
                block_offset_X + stage_offset_X, block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut2[buff_out2]));
            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out2_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut2_tr[buff_out2_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out1 = (buff_out1 + 1) % V3_BUFFS_NUM_OUT;
        buff_out2 = (buff_out2 + 1) % V3_BUFFS_NUM_OUT;
        if constexpr (RETURN_TRANSPOSE) {
            buff_out1_tr = (buff_out1_tr + 1) % V3_BUFFS_NUM_OUT_TR;
            buff_out2_tr = (buff_out2_tr + 1) % V3_BUFFS_NUM_OUT_TR;
        }
    }

    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    // ─── TMA Scale Stores (row) ───
    {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
        tk_v5::swizzle_scales_row_inplace(sSFrowwise1_ptr, cnt);
        tk_v5::swizzle_scales_row_inplace(sSFrowwise2_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm = block_offset_Y / 128;
            const int tma_x_base = ctaid_X * 2 * 256;

            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row1),
                tma_x_base, tm, reinterpret_cast<uint64_t*>(sSFrowwise1_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row1),
                tma_x_base + 256, tm, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFrowwise1_ptr) + 512));

            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row2),
                tma_x_base, tm, reinterpret_cast<uint64_t*>(sSFrowwise2_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row2),
                tma_x_base + 256, tm, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFrowwise2_ptr) + 512));

            ptx::cp_async_bulk_commit_group();
        }
    }

    // ─── TMA Scale Stores (col) ───
    if constexpr (RETURN_TRANSPOSE) {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
        tk_v5::swizzle_scales_col_inplace(sSFcolwise1_ptr, cnt);
        tk_v5::swizzle_scales_col_inplace(sSFcolwise2_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm_col = block_offset_X / 128;
            const int tma_x_base = ctaid_Y * 2 * 256;

            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col1),
                tma_x_base, tm_col, reinterpret_cast<uint64_t*>(sSFcolwise1_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col1),
                tma_x_base + 256, tm_col, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise1_ptr) + 512));

            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col2),
                tma_x_base, tm_col, reinterpret_cast<uint64_t*>(sSFcolwise2_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col2),
                tma_x_base + 256, tm_col, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise2_ptr) + 512));

            ptx::cp_async_bulk_commit_group();
        }
    }

    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// Main Kernel: 2-phase persistent, global-load inputs
// ═══════════════════════════════════════════════════════════════════

struct GloadSiluDerivArgs {
    // Phase 1 sync
    unsigned int* work_counter_phase1;
    float*        global_amax1;
    float*        global_amax2;
    unsigned int* done_counter;
    unsigned int* ready_flag;
    int tiles_X, tiles_Y, total_tiles;
    int num_persistent;
    float* sg_output;          // writes sg[0]=amax1/2688, sg[1]=amax2/2688
    // Phase 2 sync
    unsigned int* work_counter_phase2;
};

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
persistent_silu_deriv_gload_quantize_kernel(
    const __nv_bfloat16* __restrict__ dh_ptr,   // (M, H)
    const __nv_bfloat16* __restrict__ h13_ptr,  // (M, 2H)
    const __grid_constant__ CUtensorMap tmap_out1,
    const __grid_constant__ CUtensorMap tmap_out2,
    const __grid_constant__ CUtensorMap tmap_out1_t,
    const __grid_constant__ CUtensorMap tmap_out2_t,
    const __grid_constant__ CUtensorMap tmap_scale_row1,
    const __grid_constant__ CUtensorMap tmap_scale_row2,
    const __grid_constant__ CUtensorMap tmap_scale_col1,
    const __grid_constant__ CUtensorMap tmap_scale_col2,
    const int M, const int H,
    const size_t scale_stride,
    GloadSiluDerivArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);

    const int64_t ntk_r = H / 64;
    const int64_t ntk_c = M / 64;

    // SMEM layout: 1 input tile buffer + 2 row outputs + 2 col outputs + 4 scale buffers
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        1 * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    int off = in_bytes;
    fp4e2m1x2* sOut1_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_bytes;
    fp4e2m1x2* sOut2_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_bytes;
    fp4e2m1x2* sOut1_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_tr_bytes;
    fp4e2m1x2* sOut2_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_tr_bytes;

    nvfp4_scale_t* sSFrowwise1_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSFrowwise2_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSFcolwise1_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_col_bytes;
    nvfp4_scale_t* sSFcolwise2_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);

    auto& sOut1 = *reinterpret_cast<V3_OType2x3D*>(sOut1_ptr);
    auto& sOut2 = *reinterpret_cast<V3_OType2x3D*>(sOut2_ptr);
    auto& sOut1_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut1_tr_ptr);
    auto& sOut2_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut2_tr_ptr);

    float block_max1 = 0.0f;
    float block_max2 = 0.0f;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Scan dual amaxes via global loads (NO writes, NO SMEM)
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int stage_Y = t / V3_TILES_X;
            const int stage_X = t % V3_TILES_X;
            const int grow_base = block_offset_Y + stage_Y * V3_TILE_DIM_Y;
            const int gcol_base = block_offset_X + stage_X * V3_TILE_DIM_X;

            constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
            constexpr int THREADS_Y = V3_THREADS / THREADS_X;
            constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

            const int tid_Y = threadIdx.x / THREADS_X;
            const int tid_X = threadIdx.x % THREADS_X;
            const int col_base = tid_X * V3_ELTS_PER_THREAD;

            #pragma unroll
            for (int it = 0; it < ITERS; ++it) {
                const int local_row = tid_Y + it * THREADS_Y;
                const int grow = grow_base + local_row;

                #pragma unroll
                for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
                    const int gcol = gcol_base + col_base + e;

                    float dh_f  = __bfloat162float(dh_ptr[grow * H + gcol]);
                    float h1_f  = __bfloat162float(h13_ptr[grow * (2*H) + gcol]);
                    float h3_f  = __bfloat162float(h13_ptr[grow * (2*H) + H + gcol]);

                    float silu_v, silup_v;
                    fused_silu_deriv_quant::compute_silu_and_deriv(h1_f, silu_v, silup_v);

                    block_max1 = fmaxf(block_max1, fabsf(dh_f * h3_f * silup_v));
                    block_max2 = fmaxf(block_max2, fabsf(dh_f * silu_v));
                }
            }
        }
    }

    // Block reduction for both amaxes
    {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            block_max1 = fmaxf(block_max1, __shfl_xor_sync(0xffffffff, block_max1, mask));
            block_max2 = fmaxf(block_max2, __shfl_xor_sync(0xffffffff, block_max2, mask));
        }

        __shared__ float warp_max1[V3_THREADS / 32];
        __shared__ float warp_max2[V3_THREADS / 32];
        int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) {
            warp_max1[wid] = block_max1;
            warp_max2[wid] = block_max2;
        }
        __syncthreads();
        if (wid == 0) {
            block_max1 = (lane < V3_THREADS / 32) ? warp_max1[lane] : 0.0f;
            block_max2 = (lane < V3_THREADS / 32) ? warp_max2[lane] : 0.0f;
            #pragma unroll
            for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                block_max1 = fmaxf(block_max1, __shfl_xor_sync(0xffffffff, block_max1, mask));
                block_max2 = fmaxf(block_max2, __shfl_xor_sync(0xffffffff, block_max2, mask));
            }
        }
    }

    // Grid barrier
    __shared__ bool is_last_block;
    if (leading) {
        if (block_max1 > 0.0f) atomicMax((unsigned int*)args.global_amax1, __float_as_uint(block_max1));
        if (block_max2 > 0.0f) atomicMax((unsigned int*)args.global_amax2, __float_as_uint(block_max2));
        __threadfence();
        unsigned int cnt = atomicAdd(args.done_counter, 1);
        is_last_block = (cnt == args.num_persistent - 1u);
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

    const float amax_val1 = __uint_as_float(*(volatile unsigned int*)args.global_amax1);
    const float amax_val2 = __uint_as_float(*(volatile unsigned int*)args.global_amax2);
    const float S_enc1 = compute_global_encode_scaling_factor_FP4(amax_val1);
    const float S_enc2 = compute_global_encode_scaling_factor_FP4(amax_val2);

    if (leading && blockIdx.x == 0) {
        if (args.sg_output) {
            args.sg_output[0] = amax_val1 / 2688.0f;
            args.sg_output[1] = amax_val2 / 2688.0f;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Re-load from GMEM, recompute silu_deriv, quantize
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id2;
        if (leading) s_chunk_id2 = atomicAdd(args.work_counter_phase2, 1);
        __syncthreads();
        if (s_chunk_id2 >= (unsigned int)args.total_tiles) break;

        const int ctaid_X2 = s_chunk_id2 % args.tiles_X;
        const int ctaid_Y2 = s_chunk_id2 / args.tiles_X;
        const int block_offset_Y = ctaid_Y2 * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X2 * V3Config::CHUNK_DIM_X;

        gload_quantize_and_store_chunk<RETURN_TRANSPOSE>(
            dh_ptr, h13_ptr, sIn_ptr,
            sOut1_ptr, sOut2_ptr,
            sOut1_tr_ptr, sOut2_tr_ptr,
            sSFrowwise1_ptr, sSFrowwise2_ptr,
            sSFcolwise1_ptr, sSFcolwise2_ptr,
            sOut1, sOut2, sOut1_tr, sOut2_tr,
            tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
            tmap_scale_row1, tmap_scale_row2,
            tmap_scale_col1, tmap_scale_col2,
            S_enc1, S_enc2,
            block_offset_Y, block_offset_X,
            (int)M, (int)H, M, H,
            ctaid_X2, ctaid_Y2, ntk_r, ntk_c);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

// SMEM size
template <bool RETURN_TRANSPOSE>
inline int gload_silu_deriv_quant_smem_size() {
    // 1 input tile (NOT 3x or 4x pipeline stages)
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        1 * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;

    return in_bytes + 2 * out_bytes + 2 * out_tr_bytes
         + 2 * sc_row_bytes + 2 * sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace gload_silu_deriv_quant
