// v6: Register-level fused silu_deriv + dual quantize (PHASE 2 ONLY)
// Pre-req: silu_deriv_amax_only_kernel computes dual amaxes first.
// This kernel loads dh, h1, h3 from GMEM via TMA, computes silu_deriv
// in registers, and quantizes to FP4 — NO intermediate bf16 tensors.
#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"
#include "persistent_quantize.cuh"

namespace v6_silu_deriv {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

// Compute silu_deriv for 8 elements (4 pairs) in registers
__device__ __forceinline__ void silu_deriv_8x(
    const IType2* dh, const IType2* h1, const IType2* h3,
    IType2* out_dh1, IType2* out_dh3, int count
) {
    #pragma unroll
    for (int e = 0; e < count; ++e) {
        float dx = __bfloat162float(dh[e].x);
        float dy = __bfloat162float(dh[e].y);
        float h1x = __bfloat162float(h1[e].x);
        float h1y = __bfloat162float(h1[e].y);
        float h3x = __bfloat162float(h3[e].x);
        float h3y = __bfloat162float(h3[e].y);

        float sx = 1.0f / (1.0f + expf(-h1x));
        float sy = 1.0f / (1.0f + expf(-h1y));
        float silux = h1x * sx;
        float siluy = h1y * sy;
        float spx = sx * (1.0f + h1x - silux);
        float spy = sy * (1.0f + h1y - siluy);

        out_dh1[e].x = __float2bfloat16_rn(dx * h3x * spx);
        out_dh1[e].y = __float2bfloat16_rn(dy * h3y * spy);
        out_dh3[e].x = __float2bfloat16_rn(dx * silux);
        out_dh3[e].y = __float2bfloat16_rn(dy * siluy);
    }
}

// Phase2: quantize with silu_deriv in registers
// Loads 3 inputs from SMEM, computes silu_deriv, quantizes BOTH outputs
__device__ __forceinline__ void v3_rowwise_scaling_silu_deriv(
    const IType* __restrict__ sIn_dh,
    const IType* __restrict__ sIn_h1,
    const IType* __restrict__ sIn_h3,
    fp4e2m1x2* __restrict__ sOut1_ptr,
    fp4e2m1x2* __restrict__ sOut2_ptr,
    nvfp4_scale_t* __restrict__ sSF1_ptr,
    nvfp4_scale_t* __restrict__ sSF2_ptr,
    const float S_enc1, const float S_enc2,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out1, const int buff_out2
) {
    using namespace quantization_and_transposition_SF;

    const auto& sDh = *reinterpret_cast<const V3_IType3D*>(sIn_dh);
    const auto& sH1 = *reinterpret_cast<const V3_IType3D*>(sIn_h1);
    const auto& sH3 = *reinterpret_cast<const V3_IType3D*>(sIn_h3);
    auto& sOut1 = *reinterpret_cast<V3_OType2x3D*>(sOut1_ptr);
    auto& sOut2 = *reinterpret_cast<V3_OType2x3D*>(sOut2_ptr);
    auto& sSF1 = *reinterpret_cast<V3_ScalesType2D*>(sSF1_ptr);
    auto& sSF2 = *reinterpret_cast<V3_ScalesType2D*>(sSF2_ptr);

    const int lane = threadIdx.x % THREADS_PER_WARP;
    const int bank_group = lane / V3_THREADS_PER_BANK;
    const int tid_Y = threadIdx.x / V3_THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % V3_THREADS_X_ROWWISE;
    const int off_X = tid_X * V3_ELTS_PER_THREAD;

    const int SF_tid_Y = tid_Y;
    const int SF_tid_X = tid_X / V3_THREADS_PER_SCALE_ROWWISE;
    const bool SF_storing = (tid_X % V3_THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_Y = SF_tid_Y + stage_Y * V3_TILE_DIM_Y;
    const int stage_sc_X = SF_tid_X + stage_X * V3_SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < V3_ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * V3_THREADS_Y_ROWWISE;

        IType2 r_dh1[V3_WAVES][V3_PACK_SIZE/2];
        IType2 r_dh3[V3_WAVES][V3_PACK_SIZE/2];
        IType2 amax1_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        IType2 amax3_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < V3_WAVES; ++w) {
            const int sw = ((w + bank_group) * V3_PACK_SIZE) % V3_ELTS_PER_THREAD;
            const int col = off_X + sw;

            __uint128_t e_dh = ptx::ld_shared_b128(&sDh[buff_in][row][col]);
            __uint128_t e_h1 = ptx::ld_shared_b128(&sH1[buff_in][row][col]);
            __uint128_t e_h3 = ptx::ld_shared_b128(&sH3[buff_in][row][col]);

            silu_deriv_8x(
                reinterpret_cast<const IType2*>(&e_dh),
                reinterpret_cast<const IType2*>(&e_h1),
                reinterpret_cast<const IType2*>(&e_h3),
                r_dh1[w], r_dh3[w], V3_PACK_SIZE/2);

            #pragma unroll
            for (int e = 0; e < V3_PACK_SIZE/2; ++e) {
                ptx::abs_max_2x(amax1_2x, amax1_2x, r_dh1[w][e]);
                ptx::abs_max_2x(amax3_2x, amax3_2x, r_dh3[w][e]);
            }
        }

        const float blk_amax1 = get_amax_of_pair(amax1_2x);
        const nvfp4_scale_t S_mult1 = compute_encoding_scaling_factor_nv(blk_amax1, S_enc1);
        const float coeff1 = static_cast<float>(S_mult1) * S_enc1;
        const nvfp4_scale_t S_b1 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult1));

        const float blk_amax3 = get_amax_of_pair(amax3_2x);
        const nvfp4_scale_t S_mult3 = compute_encoding_scaling_factor_nv(blk_amax3, S_enc2);
        const float coeff3 = static_cast<float>(S_mult3) * S_enc2;
        const nvfp4_scale_t S_b3 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult3));

        if (SF_storing) {
            sSF1[stage_sc_Y + it * V3_THREADS_Y_ROWWISE][stage_sc_X] = S_b1;
            sSF2[stage_sc_Y + it * V3_THREADS_Y_ROWWISE][stage_sc_X] = S_b3;
        }

        #pragma unroll
        for (int w = 0; w < V3_WAVES; ++w) {
            const int sw = ((w + bank_group) * V3_PACK_SIZE) % V3_ELTS_PER_THREAD;
            const int ocol = (sw + off_X) / 2;

            uint64_t e03_1 = *reinterpret_cast<uint64_t*>(&r_dh1[w][0]);
            uint64_t e47_1 = *reinterpret_cast<uint64_t*>(&r_dh1[w][2]);
            uint32_t o1 = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff1);
            ptx::st_shared_b32(&sOut1[buff_out1][row][ocol], o1);

            uint64_t e03_3 = *reinterpret_cast<uint64_t*>(&r_dh3[w][0]);
            uint64_t e47_3 = *reinterpret_cast<uint64_t*>(&r_dh3[w][2]);
            uint32_t o3 = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_3, e47_3, coeff3);
            ptx::st_shared_b32(&sOut2[buff_out2][row][ocol], o3);
        }
    }
}

// SMEM size
template <bool RETURN_TRANSPOSE>
inline int v6_silu_deriv_quant_smem_size() {
    constexpr int in_one = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int in_total = 3 * in_one;
    constexpr int out_one = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_total = 2 * out_one;
    constexpr int sc_row_one = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_total = 2 * sc_row_one;
    return in_total + out_total + sc_total + TMA_SHMEM_ALIGNMENT;
}

struct V6Args {
    unsigned int* work_counter;
    float*        global_amax1;   // pre-computed by silu_deriv_amax_only_kernel
    float*        global_amax2;
    int tiles_X, tiles_Y, total_tiles;
    float* sg_output;
};

// V6 kernel: PHASE 2 ONLY — amaxes pre-computed externally
template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
persistent_silu_deriv_quantize_v6_kernel(
    const __grid_constant__ CUtensorMap tmap_dh,
    const __grid_constant__ CUtensorMap tmap_h1,
    const __grid_constant__ CUtensorMap tmap_h3,
    const __grid_constant__ CUtensorMap tmap_out1,
    const __grid_constant__ CUtensorMap tmap_out2,
    const __grid_constant__ CUtensorMap tmap_sc_row1,
    const __grid_constant__ CUtensorMap tmap_sc_row2,
    const size_t rows, const size_t cols,
    V6Args args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    constexpr int in_one = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_one = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);
    int off = 0;
    IType* sIn_dh = reinterpret_cast<IType*>(dshmem + off); off += in_one;
    IType* sIn_h1 = reinterpret_cast<IType*>(dshmem + off); off += in_one;
    IType* sIn_h3 = reinterpret_cast<IType*>(dshmem + off); off += in_one;
    fp4e2m1x2* sOut1 = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_one;
    fp4e2m1x2* sOut2 = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_one;
    nvfp4_scale_t* sSF_row1 = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSF_row2 = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);

    auto& sIn_dh_3d = *reinterpret_cast<V3_IType3D*>(sIn_dh);
    auto& sIn_h1_3d = *reinterpret_cast<V3_IType3D*>(sIn_h1);
    auto& sIn_h3_3d = *reinterpret_cast<V3_IType3D*>(sIn_h3);

    __shared__ uint64_t mbar_dh[V3_NUM_TILES];
    __shared__ uint64_t mbar_h1[V3_NUM_TILES];
    __shared__ uint64_t mbar_h3[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_init(&mbar_dh[t], 1);
            ptx::mbarrier_init(&mbar_h1[t], 1);
            ptx::mbarrier_init(&mbar_h3[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Read pre-computed amaxes and compute S_enc
    const float amax1 = args.global_amax1[0];
    const float amax2 = args.global_amax2[0];
    const float S_enc1 = compute_global_encode_scaling_factor_FP4(amax1);
    const float S_enc2 = compute_global_encode_scaling_factor_FP4(amax2);

    if (leading && blockIdx.x == 0 && args.sg_output) {
        args.sg_output[0] = amax1 / 2688.0f;
        args.sg_output[1] = amax2 / 2688.0f;
    }

    int mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int cx = s_chunk_id % args.tiles_X;
        const int cy = s_chunk_id / args.tiles_X;
        const int bY = cy * V3Config::CHUNK_DIM_Y;
        const int bX = cx * V3Config::CHUNK_DIM_X;

        // Prefetch first 2 tiles for all 3 inputs
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&mbar_dh[pre], tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_dh_3d[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_dh),
                    bX + tx*V3_TILE_DIM_X, bY + ty*V3_TILE_DIM_Y, &mbar_dh[pre]);

                ptx::mbarrier_arrive_expect_tx(&mbar_h1[pre], tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h1_3d[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_h1),
                    bX + tx*V3_TILE_DIM_X, bY + ty*V3_TILE_DIM_Y, &mbar_h1[pre]);

                ptx::mbarrier_arrive_expect_tx(&mbar_h3[pre], tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h3_3d[pre]),
                    reinterpret_cast<const uint64_t*>(&tmap_h3),
                    bX + tx*V3_TILE_DIM_X, bY + ty*V3_TILE_DIM_Y, &mbar_h3[pre]);
            }
        }

        int bout1 = 0, bout2 = 0;

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int sY = t / V3_TILES_X, sX = t % V3_TILES_X;

            if (t + 2 < V3_NUM_TILES) {
                const int n = t + 2;
                const int ty = n / V3_TILES_X, tx = n % V3_TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&mbar_dh[n], tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_dh_3d[n]),
                        reinterpret_cast<const uint64_t*>(&tmap_dh),
                        bX + tx*V3_TILE_DIM_X, bY + ty*V3_TILE_DIM_Y, &mbar_dh[n]);
                    ptx::mbarrier_arrive_expect_tx(&mbar_h1[n], tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_h1_3d[n]),
                        reinterpret_cast<const uint64_t*>(&tmap_h1),
                        bX + tx*V3_TILE_DIM_X, bY + ty*V3_TILE_DIM_Y, &mbar_h1[n]);
                    ptx::mbarrier_arrive_expect_tx(&mbar_h3[n], tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_h3_3d[n]),
                        reinterpret_cast<const uint64_t*>(&tmap_h3),
                        bX + tx*V3_TILE_DIM_X, bY + ty*V3_TILE_DIM_Y, &mbar_h3[n]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&mbar_dh[t], mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&mbar_h1[t], mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&mbar_h3[t], mbar_phase);

            v3_rowwise_scaling_silu_deriv(
                sIn_dh, sIn_h1, sIn_h3,
                sOut1, sOut2, sSF_row1, sSF_row2,
                S_enc1, S_enc2, sY, sX, t, bout1, bout2);

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            // TMA store dh1 + dh3 FP4
            if (leading) {
                auto& sO1 = *reinterpret_cast<V3_OType2x3D*>(sOut1);
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out1),
                    bX + sX*V3_TILE_DIM_X, bY + sY*V3_TILE_DIM_Y,
                    reinterpret_cast<uint64_t*>(&sO1[bout1]));

                auto& sO2 = *reinterpret_cast<V3_OType2x3D*>(sOut2);
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out2),
                    bX + sX*V3_TILE_DIM_X, bY + sY*V3_TILE_DIM_Y,
                    reinterpret_cast<uint64_t*>(&sO2[bout2]));

                ptx::cp_async_bulk_commit_group();
            }

            bout1 = (bout1 + 1) % V3_BUFFS_NUM_OUT;
            bout2 = (bout2 + 1) % V3_BUFFS_NUM_OUT;
        }

        // Wait for TMA stores
        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        // Scale stores for dh1
        {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_X,
                               ((int)cols - bX) / (int)V3_SCALE_DIM);
            tk_v5::swizzle_scales_row_inplace(sSF_row1, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                const int tm = bY / 128;
                const int tma_x = cx * 2 * 256;
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_sc_row1),
                    tma_x, tm, reinterpret_cast<uint64_t*>(sSF_row1));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_sc_row1),
                    tma_x+256, tm,
                    reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSF_row1)+512));
                ptx::cp_async_bulk_commit_group();
            }
        }
        // Scale stores for dh3
        {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_X,
                               ((int)cols - bX) / (int)V3_SCALE_DIM);
            tk_v5::swizzle_scales_row_inplace(sSF_row2, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                const int tm = bY / 128;
                const int tma_x = cx * 2 * 256;
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_sc_row2),
                    tma_x, tm, reinterpret_cast<uint64_t*>(sSF_row2));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_sc_row2),
                    tma_x+256, tm,
                    reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSF_row2)+512));
                ptx::cp_async_bulk_commit_group();
            }
        }

        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        mbar_phase ^= 1;
    }

    // Cleanup
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&mbar_dh[t]);
            ptx::mbarrier_invalid(&mbar_h1[t]);
            ptx::mbarrier_invalid(&mbar_h3[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

#endif // FP4_TYPE_SUPPORTED
} // namespace v6_silu_deriv
