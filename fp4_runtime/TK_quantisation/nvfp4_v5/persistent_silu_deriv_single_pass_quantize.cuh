/*************************************************************************
 * Single-Pass SiLU Derivative + FP4 Quantize via Global Loads
 *
 * TE-style delayed scaling: uses pre-computed S_enc from previous
 * iteration, accumulates dynamic amax for next iteration.
 *
 * Single GMEM pass: reads dh + h13, computes silu_deriv in registers,
 * writes to SMEM tile buffer, quantizes both dh1/dh3, TMA stores.
 *
 * Traffic: 3MH read + ~1MH write = 4MH total (vs 7-8MH for 2-phase).
 *
 * SMEM budget (~32 KB → high occupancy):
 *   Input tile (reused): 1 x 64x64 x 2B = 8 KB
 *   Out dh1 row + col:   ~8 KB
 *   Out dh3 row + col:   ~8 KB
 *   Scales:              ~8 KB
 *************************************************************************/

#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"
#include "fused_silu_deriv_quantize.cuh"

namespace single_pass_silu_deriv_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

struct SinglePassArgs {
    unsigned int* work_counter;
    float*        amax_out1;     // output: dynamic amax for dh1 (for next iteration)
    float*        amax_out2;     // output: dynamic amax for dh3 (for next iteration)
    const float*  prev_amax;     // optional input: previous iteration amax for delayed scaling
    int tiles_X, tiles_Y, total_tiles;
    float* sg_output;            // writes sg[0]=S_enc1_inv, sg[1]=S_enc2_inv
    bool collect_current_amax;    // whether to accumulate amax_out for the next iteration
};

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
persistent_silu_deriv_single_pass_kernel(
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
    const float S_enc1,           // pre-computed: 2688 / prev_amax1
    const float S_enc2,           // pre-computed: 2688 / prev_amax2
    SinglePassArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);



    // SMEM layout: 1 input tile + 2 row outputs + 2 col outputs + 4 scale buffers
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

    float S_enc1_eff = S_enc1;
    float S_enc2_eff = S_enc2;
    if (args.prev_amax != nullptr) {
        S_enc1_eff = compute_global_encode_scaling_factor_FP4(args.prev_amax[0]);
        S_enc2_eff = compute_global_encode_scaling_factor_FP4(args.prev_amax[1]);
    }

    // Thread-local amax accumulators
    const bool collect_current_amax =
        args.collect_current_amax && args.amax_out1 != nullptr && args.amax_out2 != nullptr;
    float block_max1 = 0.0f;
    float block_max2 = 0.0f;

    // Write sg_output (scale_inv = 1/S_enc = amax/2688)
    if (leading && blockIdx.x == 0 && args.sg_output) {
        args.sg_output[0] = (S_enc1_eff > 0.0f) ? 1.0f / S_enc1_eff : 0.0f;
        args.sg_output[1] = (S_enc2_eff > 0.0f) ? 1.0f / S_enc2_eff : 0.0f;
    }

    // ═══════════════════════════════════════════════════════════════
    // SINGLE PASS: load → silu_deriv → quantize → store
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;
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

            // ── Global load + silu_deriv + write dh1 to SMEM + accumulate amax ──
            {
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

                        float dh1_v = dh_f * h3_f * silup_v;
                        float dh3_v = dh_f * silu_v;

                        if (collect_current_amax) {
                            block_max1 = fmaxf(block_max1, fabsf(dh1_v));
                            block_max2 = fmaxf(block_max2, fabsf(dh3_v));
                        }

                        // Write dh1 to SMEM for quantization (flat indexing)
                        sIn_ptr[local_row * V3_BUFF_DIM_X + col_base + e] = __float2bfloat16_rn(dh1_v);
                    }
                }
            }
            __syncthreads();

            // Quantize dh1 from SMEM
            v3_rowwise_scaling(sIn_ptr, sOut1_ptr, sSFrowwise1_ptr,
                              S_enc1_eff, stage_Y, stage_X, 0, buff_out1);
            if constexpr (RETURN_TRANSPOSE) {
                v3_colwise_scaling(sIn_ptr, sOut1_tr_ptr, sSFcolwise1_ptr,
                                  S_enc1_eff, stage_Y, stage_X, 0, buff_out1_tr);
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            // TMA store dh1
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

            // ── Now recompute dh3 and write to SMEM ──
            // Re-read from GMEM (L2 cached from dh1 pass above)
            {
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

                        float sig = 1.0f / (1.0f + expf(-h1_f));
                        float silu_v = h1_f * sig;
                        float dh3_v = dh_f * silu_v;

                        sIn_ptr[local_row * V3_BUFF_DIM_X + col_base + e] = __float2bfloat16_rn(dh3_v);
                    }
                }
            }
            __syncthreads();

            // Quantize dh3 from SMEM
            v3_rowwise_scaling(sIn_ptr, sOut2_ptr, sSFrowwise2_ptr,
                              S_enc2_eff, stage_Y, stage_X, 0, buff_out2);
            if constexpr (RETURN_TRANSPOSE) {
                v3_colwise_scaling(sIn_ptr, sOut2_tr_ptr, sSFcolwise2_ptr,
                                  S_enc2_eff, stage_Y, stage_X, 0, buff_out2_tr);
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            // TMA store dh3
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

        // Wait for TMA stores to complete before reusing output buffers
        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        // ─── Scale stores (row) ───
        {
            const int chunk_cols = (int)H - block_offset_X;
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

        // ─── Scale stores (col) ───
        if constexpr (RETURN_TRANSPOSE) {
            const int chunk_rows = (int)M - block_offset_Y;
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

    // ─── Block reduction for amaxes → atomicMax to global ───
    if (collect_current_amax) {
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
        if (leading) {
            if (block_max1 > 0.0f) atomicMax((unsigned int*)args.amax_out1, __float_as_uint(block_max1));
            if (block_max2 > 0.0f) atomicMax((unsigned int*)args.amax_out2, __float_as_uint(block_max2));
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

// SMEM size
template <bool RETURN_TRANSPOSE>
inline int single_pass_silu_deriv_quant_smem_size() {
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

}  // namespace single_pass_silu_deriv_quant
