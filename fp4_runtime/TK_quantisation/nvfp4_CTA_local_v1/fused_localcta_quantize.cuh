/*************************************************************************
 * CTA-local single-pass NVFP4 quantization.
 *
 * This is a fork of the v5 fused amax+quantize kernel with the tensor-wide
 * reduction removed. Each CTA quantizes one 128x128 chunk using its own
 * chunk-local amax and writes one decode scale per chunk.
 *************************************************************************/

#ifndef TK_LOCALCTA_FUSED_QUANTIZE_CUH_
#define TK_LOCALCTA_FUSED_QUANTIZE_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include "../nvfp4_v5/util/cast_common.h"
#include "../nvfp4_v5/util/math.h"
#include "../nvfp4_v5/util/ptx.cuh"
#include "../nvfp4_v5/util/utils.cuh"
#include "../nvfp4_v5/core.cuh"

namespace tk_localcta {

using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::dispatch::nvfp4::core;
using namespace transformer_engine;
using namespace transformer_engine::ptx;

#if FP4_TYPE_SUPPORTED

struct LocalCTAConfig {
    static constexpr int CHUNK_DIM_Y = 128;
    static constexpr int CHUNK_DIM_X = 128;
};

static constexpr float LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM = 1493.0f;
static __device__ __constant__ float kLocalCTAGlobalScaleNum = LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM;

static constexpr int SCALE_DIM = 16;
static constexpr int THREADS = 128;
static constexpr int ELTS_PER_THREAD = 16;
static constexpr int TILE_DIM_Y = 64;
static constexpr int TILE_DIM_X = 64;

static constexpr int TILES_Y = LocalCTAConfig::CHUNK_DIM_Y / TILE_DIM_Y;
static constexpr int TILES_X = LocalCTAConfig::CHUNK_DIM_X / TILE_DIM_X;
static constexpr int NUM_TILES = TILES_Y * TILES_X;

static constexpr int BUFFS_NUM_IN = NUM_TILES;
static constexpr int BUFFS_NUM_OUT = 2;
static constexpr int BUFFS_NUM_OUT_TR = 2;

static constexpr int BUFF_DIM_Y = TILE_DIM_Y;
static constexpr int BUFF_DIM_X = TILE_DIM_X;
static constexpr int BUFF_IN_ELEMS = BUFF_DIM_Y * BUFF_DIM_X;

static constexpr int BUFF_OUT_DIM_Y = BUFF_DIM_Y;
static constexpr int BUFF_OUT_DIM_X = (BUFF_DIM_X * 4) / 8;
static constexpr int BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

static constexpr int BUFF_OUT_TR_DIM_Y = BUFF_DIM_X;
static constexpr int BUFF_OUT_TR_DIM_X = (BUFF_DIM_Y * 4) / 8;
static constexpr int BUFF_OUT_TR_SIZE = BUFF_OUT_TR_DIM_Y * BUFF_OUT_TR_DIM_X;

static constexpr int SCALES_PER_CHUNK_Y = LocalCTAConfig::CHUNK_DIM_Y / SCALE_DIM;
static constexpr int SCALES_PER_CHUNK_X = LocalCTAConfig::CHUNK_DIM_X / SCALE_DIM;
static constexpr int SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;
static constexpr int SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;

static constexpr int PACK_SIZE = 8;
static constexpr int WAVES = ELTS_PER_THREAD / PACK_SIZE;
static constexpr float LOCALCTA_PREPARED_MIN_NONZERO_SCALE = 0.001953125f;  // 2^-9

static constexpr int THREADS_X_ROWWISE = TILE_DIM_X / ELTS_PER_THREAD;
static constexpr int THREADS_Y_ROWWISE = THREADS / THREADS_X_ROWWISE;
static constexpr int ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;

static constexpr int THREADS_PER_SCALE_ROWWISE = SCALE_DIM / ELTS_PER_THREAD;

static constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;
static constexpr int THREADS_PER_BANK = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;

using IType = bf16;
using IType2 = typename ptx::FPx2<IType>;
using IType3D = IType[BUFFS_NUM_IN][BUFF_DIM_Y][BUFF_DIM_X];
template <int PIPE_DEPTH>
using ITypeRing4D = IType[PIPE_DEPTH][BUFFS_NUM_IN][BUFF_DIM_Y][BUFF_DIM_X];
using IType2x3D = IType2[BUFFS_NUM_IN][BUFF_DIM_Y][BUFF_DIM_X / 2];
using OType2x3D = fp4e2m1x2[BUFFS_NUM_OUT][BUFF_OUT_DIM_Y][BUFF_OUT_DIM_X];
using OType2xt3D = fp4e2m1x2[BUFFS_NUM_OUT_TR][BUFF_OUT_TR_DIM_Y][BUFF_OUT_TR_DIM_X];
using ScalesType2D = nvfp4_scale_t[LocalCTAConfig::CHUNK_DIM_Y][SCALES_PER_CHUNK_X];
using ScalesTypeTr2D = nvfp4_scale_t[LocalCTAConfig::CHUNK_DIM_X][SCALES_PER_CHUNK_Y];

struct LocalCTAPersistentArgs {
    unsigned int* work_counter;
    int tiles_X;
    int total_tiles;
};

struct LocalCTA2ClusterArgs {
    unsigned int* work_counter;
    int tiles_X;
    int tiles_Y;
    int total_macro_tiles;
};

__device__ __forceinline__ void cluster_sync_aligned() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");
}

template <typename T>
__device__ __forceinline__ uint32_t map_shared_cluster_addr(const T* ptr, int dst_cta) {
    uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(remote_addr)
                 : "r"(local_addr), "r"(dst_cta));
    return remote_addr;
}

__device__ __forceinline__ float cluster_load_shared_f32(const float* ptr, int src_cta) {
    const uint32_t remote_addr = map_shared_cluster_addr(ptr, src_cta);
    uint32_t bits;
    asm volatile("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(bits) : "r"(remote_addr));
    return *reinterpret_cast<const float*>(&bits);
}

__device__ __forceinline__ float localcta_global_scale_num() {
    return kLocalCTAGlobalScaleNum;
}

__device__ __forceinline__ float compute_localcta_encode_scaling_factor_FP4(const float local_amax) {
    float local_encode_scale = localcta_global_scale_num() / local_amax;
    local_encode_scale = fminf(local_encode_scale, transformer_engine::detail::TypeExtrema<float>::max);
    if (local_amax == 0.0f || local_encode_scale == 0.0f) {
        return 1.0f;
    }
    return local_encode_scale;
}

__device__ __forceinline__ uint32_t cluster_load_shared_u32(const uint32_t* ptr, int src_cta) {
    const uint32_t remote_addr = map_shared_cluster_addr(ptr, src_cta);
    uint32_t value;
    asm volatile("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(value) : "r"(remote_addr));
    return value;
}

template <int GROUP_THREADS, uint32_t BARRIER_ID = 1u>
__device__ __forceinline__ void subgroup_barrier_sync() {
    if (threadIdx.x < GROUP_THREADS) {
        ptx::numbered_barrier_sync(GROUP_THREADS, BARRIER_ID);
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ void swizzle_scales_row_inplace_group(
    nvfp4_scale_t* sSFrowwise_ptr, int num_scales_x, int tid
) {
    uint8_t my_scales[SCALES_PER_CHUNK_X];

    for (int row = tid; row < (int)LocalCTAConfig::CHUNK_DIM_Y; row += GROUP_THREADS) {
        const int j = row % 32;
        const int grp = row / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFrowwise_ptr[row * SCALES_PER_CHUNK_X + k]);
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFrowwise_ptr)[dest] = my_scales[k];
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ void swizzle_scales_col_inplace_group(
    nvfp4_scale_t* sSFcolwise_ptr, int num_scales_y, int tid
) {
    uint8_t my_scales[SCALES_PER_CHUNK_Y];

    for (int col = tid; col < (int)LocalCTAConfig::CHUNK_DIM_X; col += GROUP_THREADS) {
        const int j = col % 32;
        const int grp = col / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFcolwise_ptr[col * SCALES_PER_CHUNK_Y + k]);
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFcolwise_ptr)[dest] = my_scales[k];
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ void scale_swizzled_scales_inplace_group(
    nvfp4_scale_t* scales_ptr,
    int num_elements,
    float global_scale,
    int tid
) {
    for (int idx = tid; idx < num_elements; idx += GROUP_THREADS) {
        const float raw_v = static_cast<float>(scales_ptr[idx]);
        const float scaled_v = raw_v * global_scale;
        nvfp4_scale_t stored_fp8 = static_cast<nvfp4_scale_t>(scaled_v);
        if (raw_v > 0.0f && global_scale > 0.0f && static_cast<float>(stored_fp8) == 0.0f) {
            stored_fp8 = static_cast<nvfp4_scale_t>(LOCALCTA_PREPARED_MIN_NONZERO_SCALE);
        }
        scales_ptr[idx] = stored_fp8;
    }
}

__device__ __forceinline__ void swizzle_scales_row_inplace(
    nvfp4_scale_t* sSFrowwise_ptr, int num_scales_x
) {
    uint8_t my_scales[SCALES_PER_CHUNK_X];

    for (int row = threadIdx.x; row < (int)LocalCTAConfig::CHUNK_DIM_Y; row += THREADS) {
        const int j = row % 32;
        const int grp = row / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFrowwise_ptr[row * SCALES_PER_CHUNK_X + k]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFrowwise_ptr)[dest] = my_scales[k];
            }
        }

        __syncthreads();
    }
}

__device__ __forceinline__ void swizzle_scales_col_inplace(
    nvfp4_scale_t* sSFcolwise_ptr, int num_scales_y
) {
    uint8_t my_scales[SCALES_PER_CHUNK_Y];

    for (int col = threadIdx.x; col < (int)LocalCTAConfig::CHUNK_DIM_X; col += THREADS) {
        const int j = col % 32;
        const int grp = col / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFcolwise_ptr[col * SCALES_PER_CHUNK_Y + k]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFcolwise_ptr)[dest] = my_scales[k];
            }
        }

        __syncthreads();
    }
}

__device__ __forceinline__ void tma_store_scales_2x512(
    const CUtensorMap& tmap, nvfp4_scale_t* smem_ptr,
    int tm_row, int tma_x_base
) {
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        reinterpret_cast<const uint64_t*>(&tmap),
        tma_x_base, tm_row,
        reinterpret_cast<uint64_t*>(smem_ptr));
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        reinterpret_cast<const uint64_t*>(&tmap),
        tma_x_base + 256, tm_row,
        reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_ptr) + 512));
    ptx::cp_async_bulk_commit_group();
}

__device__ __forceinline__ void scale_swizzled_scales_inplace(
    nvfp4_scale_t* scales_ptr,
    int num_elements,
    float global_scale
) {
    for (int idx = threadIdx.x; idx < num_elements; idx += THREADS) {
        const float raw_v = static_cast<float>(scales_ptr[idx]);
        const float scaled_v = raw_v * global_scale;
        nvfp4_scale_t stored_fp8 = static_cast<nvfp4_scale_t>(scaled_v);
        if (raw_v > 0.0f && global_scale > 0.0f && static_cast<float>(stored_fp8) == 0.0f) {
            stored_fp8 = static_cast<nvfp4_scale_t>(LOCALCTA_PREPARED_MIN_NONZERO_SCALE);
        }
        scales_ptr[idx] = stored_fp8;
    }
}

__device__ __forceinline__ float get_amax_of_pair(const IType2 pair) {
    const float ax = __bfloat162float(__habs(pair.x));
    const float ay = __bfloat162float(__habs(pair.y));
    return fmaxf(ax, ay);
}

template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void rowwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);

    const int thread_lane = threadIdx.x % THREADS_PER_WARP;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int SF_tid_Y = tid_Y;
    const int SF_tid_X = tid_X / THREADS_PER_SCALE_ROWWISE;
    const bool SF_storing = (tid_X % THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_Y = SF_tid_Y + stage_Y * TILE_DIM_Y;
    const int stage_sc_X = SF_tid_X + stage_X * SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;

        __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ptx::ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                ptx::abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
            }
        }

        const float block_amax = get_amax_of_pair(amax_2x);

        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(block_amax, S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }

        if (SF_storing) {
            sSFrowwise[stage_sc_Y + it * THREADS_Y_ROWWISE][stage_sc_X] = S_b_fp8;
        }

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void colwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out_tr
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

    const int warp = threadIdx.x / THREADS_PER_WARP;
    const int lane = threadIdx.x % THREADS_PER_WARP;

    const int tid_Y = (lane % 4 + warp) % 4;
    const int tid_X = lane;
    const int off_Y = tid_Y * SCALE_DIM;
    const int off_X = tid_X * 2;
    const int in_Y = off_Y, in_X = off_X / 2;
    const int out_tr_Y = off_X, out_tr_X = off_Y / 2;
    const int sc_tr_Y = (stage_X * TILE_DIM_X) + 2 * tid_X;
    const int sc_tr_X = (stage_Y * SCALES_PER_TILE_Y) + tid_Y;

    __align__(8) IType rIn[2][SCALE_DIM];
    IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int i = 0; i < SCALE_DIM; ++i) {
        const IType2 pair = ptx::ld_shared_b32(&sIn2x[buff_in][in_Y + i][in_X]);
        rIn[0][i] = pair.x;
        rIn[1][i] = pair.y;
        ptx::abs_max_2x(amax_2x, amax_2x, pair);
    }

    const float bmax[2] = {
        __bfloat162float(__habs(amax_2x.x)),
        __bfloat162float(__habs(amax_2x.y))
    };

    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(bmax[w], S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(bmax[w], S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }
        sSFcolwise[sc_tr_Y + w][sc_tr_X] = S_b_fp8;

        __align__(8) uint32_t rOut[SCALE_DIM / 8];
        #pragma unroll
        for (int e = 0; e < SCALE_DIM / 8; ++e) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e + 4]);
            rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
        }
        ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_Y + w][out_tr_X],
                           *reinterpret_cast<uint64_t*>(rOut));
    }
}

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void rowwise_scaling_group(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out,
    int tid
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    static_assert(GROUP_THREADS == 128, "rowwise_scaling_group currently expects 128 consumer threads");

    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);

    const int thread_lane = tid % THREADS_PER_WARP;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = tid / THREADS_X_ROWWISE;
    const int tid_X = tid % THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int SF_tid_Y = tid_Y;
    const int SF_tid_X = tid_X / THREADS_PER_SCALE_ROWWISE;
    const bool SF_storing = (tid_X % THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_Y = SF_tid_Y + stage_Y * TILE_DIM_Y;
    const int stage_sc_X = SF_tid_X + stage_X * SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;

        __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ptx::ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                ptx::abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
            }
        }

        const float block_amax = get_amax_of_pair(amax_2x);

        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(block_amax, S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }

        if (SF_storing) {
            sSFrowwise[stage_sc_Y + it * THREADS_Y_ROWWISE][stage_sc_X] = S_b_fp8;
        }

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void colwise_scaling_group(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out_tr,
    int tid
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    static_assert(GROUP_THREADS == 128, "colwise_scaling_group currently expects 128 consumer threads");

    const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

    const int warp = tid / THREADS_PER_WARP;
    const int lane = tid % THREADS_PER_WARP;

    const int tid_Y = (lane % 4 + warp) % 4;
    const int tid_X = lane;
    const int off_Y = tid_Y * SCALE_DIM;
    const int off_X = tid_X * 2;
    const int in_Y = off_Y, in_X = off_X / 2;
    const int out_tr_Y = off_X, out_tr_X = off_Y / 2;
    const int sc_tr_Y = (stage_X * TILE_DIM_X) + 2 * tid_X;
    const int sc_tr_X = (stage_Y * SCALES_PER_TILE_Y) + tid_Y;

    __align__(8) IType rIn[2][SCALE_DIM];
    IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int i = 0; i < SCALE_DIM; ++i) {
        const IType2 pair = ptx::ld_shared_b32(&sIn2x[buff_in][in_Y + i][in_X]);
        rIn[0][i] = pair.x;
        rIn[1][i] = pair.y;
        ptx::abs_max_2x(amax_2x, amax_2x, pair);
    }

    const float bmax[2] = {
        __bfloat162float(__habs(amax_2x.x)),
        __bfloat162float(__habs(amax_2x.y))
    };

    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(bmax[w], S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(bmax[w], S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }
        sSFcolwise[sc_tr_Y + w][sc_tr_X] = S_b_fp8;

        __align__(8) uint32_t rOut[SCALE_DIM / 8];
        #pragma unroll
        for (int e = 0; e < SCALE_DIM / 8; ++e) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e + 4]);
            rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
        }
        ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_Y + w][out_tr_X],
                           *reinterpret_cast<uint64_t*>(rOut));
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ float scan_tile_amax_group(
    const IType* __restrict__ sIn_ptr,
    int buff_in,
    int tid
) {
    static_assert(GROUP_THREADS == 128, "scan_tile_amax_group currently expects 128 consumer threads");
    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    const int lane = tid % THREADS_PER_WARP;
    const int bank_group = lane / THREADS_PER_BANK;
    const int tid_Y = tid / THREADS_X_ROWWISE;
    const int tid_X = tid % THREADS_X_ROWWISE;
    const int off_X = tid_X * ELTS_PER_THREAD;
    float tile_max = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t elts = ptx::ld_shared_b128(&sIn[buff_in][row][off_X + sw]);
            const IType2* pairs = reinterpret_cast<const IType2*>(&elts);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                float a = __bfloat162float(__habs(pairs[e].x));
                float b = __bfloat162float(__habs(pairs[e].y));
                tile_max = fmaxf(tile_max, fmaxf(a, b));
            }
        }
    }
    return tile_max;
}

__device__ __forceinline__ float scan_tile_amax(const IType* __restrict__ sIn_ptr, int buff_in) {
    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    const int lane = threadIdx.x % THREADS_PER_WARP;
    const int bank_group = lane / THREADS_PER_BANK;
    const int tid_Y = threadIdx.x / THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % THREADS_X_ROWWISE;
    const int off_X = tid_X * ELTS_PER_THREAD;
    float tile_max = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t elts = ptx::ld_shared_b128(&sIn[buff_in][row][off_X + sw]);
            const IType2* pairs = reinterpret_cast<const IType2*>(&elts);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                float a = __bfloat162float(__habs(pairs[e].x));
                float b = __bfloat162float(__habs(pairs[e].y));
                tile_max = fmaxf(tile_max, fmaxf(a, b));
            }
        }
    }
    return tile_max;
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    LocalCTAPersistentArgs args,
    bool write_raw_scales,
    bool write_prepared
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

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
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    __shared__ float warp_max[THREADS / 32];
    __shared__ float cta_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;

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
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
        const int chunk_cols = static_cast<int>(cols) - block_offset_X;

        float cta_max = 0.0f;

        #pragma unroll
        for (int pre = 0; pre < min(2, (int)NUM_TILES); ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            if (t + 2 < NUM_TILES) {
                const int next = t + 2;
                const int ty = next / TILES_X;
                const int tx = next % TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }

        if (lane == 0) {
            warp_max[wid] = cta_max;
        }
        __syncthreads();

        if (wid == 0) {
            cta_max = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                cta_amax_shared = cta_max;
            }
        }
        __syncthreads();

        const float amax_val = cta_amax_shared;
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
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

            if (t > 0) {
                ptx::cp_async_bulk_wait_group_read<1>();
            }

            rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                                            S_enc, stage_Y, stage_X, t, buff_out);

            if constexpr (RETURN_TRANSPOSE) {
                colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
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
            if (write_raw_scales && leading) {
                tma_store_scales_2x512(tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            }
            if (write_prepared) {
                if (write_raw_scales && leading) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
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
        }

        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
            swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (write_raw_scales && leading) {
                tma_store_scales_2x512(tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            }
            if (write_prepared) {
                if (write_raw_scales && leading) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
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
        }

        if (leading && (write_raw_scales || write_prepared)) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE = true, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_split2_prepared_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    LocalCTAPersistentArgs args,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

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
        DIVUP_TO_MULTIPLE(
            LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
            TMA_SHMEM_ALIGNMENT) : 0;
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    __shared__ float warp_max[THREADS / 32];
    __shared__ float cta_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;

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
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
        const int chunk_cols = static_cast<int>(cols) - block_offset_X;

        int local_ctaid_X = ctaid_X;
        bool second_split = false;
        if (ctaid_X >= split0_tiles) {
            second_split = true;
            local_ctaid_X -= split0_tiles;
        }
        const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const uint64_t* selected_tensor_map = second_split ?
            reinterpret_cast<const uint64_t*>(&tensor_map_input1) :
            reinterpret_cast<const uint64_t*>(&tensor_map_input0);

        float cta_max = 0.0f;

        #pragma unroll
        for (int pre = 0; pre < min(2, (int)NUM_TILES); ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    selected_tensor_map,
                    input_block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            if (t + 2 < NUM_TILES) {
                const int next = t + 2;
                const int ty = next / TILES_X;
                const int tx = next % TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[next]),
                        selected_tensor_map,
                        input_block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }
        if (lane == 0) {
            warp_max[wid] = cta_max;
        }
        __syncthreads();

        if (wid == 0) {
            float warp_val = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                warp_val = fmaxf(warp_val, __shfl_xor_sync(0xffffffff, warp_val, mask));
            }
            if (lane == 0) {
                cta_amax_shared = warp_val;
            }
        }
        __syncthreads();
        cta_max = cta_amax_shared;

        const float S_enc = compute_localcta_encode_scaling_factor_FP4(cta_max);
        const float sg = cta_max / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg;
            if constexpr (RETURN_TRANSPOSE) {
                col_sg_chunks[ctaid_X * (rows / LocalCTAConfig::CHUNK_DIM_Y) + ctaid_Y] = sg;
            }
        }
        __syncthreads();

        const int chunk_scales_x = min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM);
        const int chunk_scales_y = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int ty = t / TILES_X;
            const int tx = t % TILES_X;

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            rowwise_scaling<ENCODE_CENTRIC>(
                sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc,
                ty, tx, t, t % BUFFS_NUM_OUT);
            if constexpr (RETURN_TRANSPOSE) {
                colwise_scaling<ENCODE_CENTRIC>(
                    sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc,
                    ty, tx, t, t % BUFFS_NUM_OUT_TR);
            }

            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();

            if (leading) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    reinterpret_cast<uint64_t*>(&sOut[t % BUFFS_NUM_OUT]));
                if constexpr (RETURN_TRANSPOSE) {
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                        block_offset_X_tr + ty * TILE_DIM_Y,
                        block_offset_Y_tr + tx * TILE_DIM_X,
                        reinterpret_cast<uint64_t*>(&sOut_tr[t % BUFFS_NUM_OUT_TR]));
                }
                ptx::cp_async_bulk_commit_group();
            }
            if (leading && t > 0) {
                ptx::cp_async_bulk_wait_group_read<1>();
            }
            __syncthreads();
        }

        if (leading) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        swizzle_scales_row_inplace(sSFrowwise_ptr, chunk_scales_x);
        scale_swizzled_scales_inplace(sSFrowwise_ptr, LocalCTAConfig::CHUNK_DIM_Y * chunk_scales_x, sg);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if (leading) {
            tma_store_scales_2x512(
                tmap_scale_row_prepared,
                sSFrowwise_ptr,
                ctaid_Y,
                ctaid_X * 2 * 256);
        }

        if constexpr (RETURN_TRANSPOSE) {
            swizzle_scales_col_inplace(sSFcolwise_ptr, chunk_scales_y);
            scale_swizzled_scales_inplace(sSFcolwise_ptr, LocalCTAConfig::CHUNK_DIM_X * chunk_scales_y, sg);
            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(
                    tmap_scale_col_prepared,
                    sSFcolwise_ptr,
                    ctaid_X,
                    ctaid_Y * 2 * 256);
            }
        }

        if (leading) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                ptx::mbarrier_invalid(&in_mbar[t]);
            }
        }
        __syncthreads();
        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool OUTPUT_DH1>
__device__ __forceinline__ float transform_silu_deriv_tile_to_chunk_local(
    IType* sChunk_ptr,
    const IType* sDhTile_ptr,
    const IType* sH1Tile_ptr,
    const IType* sH3Tile_ptr,
    int tile_idx
) {
    using Tile2D = IType[BUFF_DIM_Y][BUFF_DIM_X];
    auto& sChunk = *reinterpret_cast<IType3D*>(sChunk_ptr);
    const auto& sDhTile = *reinterpret_cast<const Tile2D*>(sDhTile_ptr);
    const auto& sH1Tile = *reinterpret_cast<const Tile2D*>(sH1Tile_ptr);
    const auto& sH3Tile = *reinterpret_cast<const Tile2D*>(sH3Tile_ptr);

    float local_max = 0.0f;
    constexpr int VEC = 4;
    constexpr int TILE_TOTAL = BUFF_DIM_Y * BUFF_DIM_X;

    for (int idx = threadIdx.x * VEC; idx < TILE_TOTAL; idx += THREADS * VEC) {
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

            const float sig0x = 1.0f / (1.0f + expf(-b0f.x));
            const float sig0y = 1.0f / (1.0f + expf(-b0f.y));
            const float sig1x = 1.0f / (1.0f + expf(-b1f.x));
            const float sig1y = 1.0f / (1.0f + expf(-b1f.y));
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

            sChunk[tile_idx][row][col + 0] = out0.x;
            sChunk[tile_idx][row][col + 1] = out0.y;
            sChunk[tile_idx][row][col + 2] = out1.x;
            sChunk[tile_idx][row][col + 3] = out1.y;
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                if (c < BUFF_DIM_X) {
                    const float vd = __bfloat162float(sDhTile[row][c]);
                    const float v1 = __bfloat162float(sH1Tile[row][c]);
                    const float v3 = __bfloat162float(sH3Tile[row][c]);
                    const float sig = 1.0f / (1.0f + expf(-v1));
                    const float silu_v1 = v1 * sig;
                    const float transformed = OUTPUT_DH1
                        ? vd * v3 * (sig * (1.0f + v1 - silu_v1))
                        : vd * silu_v1;
                    const __nv_bfloat16 out = __float2bfloat16_rn(transformed);
                    local_max = fmaxf(local_max, fabsf(__bfloat162float(out)));
                    sChunk[tile_idx][row][c] = out;
                }
            }
        }
    }
    return local_max;
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
fused_localcta_silu_deriv_split2_prepared_kernel(
    const __grid_constant__ CUtensorMap tensor_map_dh,
    const __grid_constant__ CUtensorMap tensor_map_h1,
    const __grid_constant__ CUtensorMap tensor_map_h3,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    LocalCTAPersistentArgs args,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);
    constexpr int tile_bytes_aligned = DIVUP_TO_MULTIPLE(shmem_tile_bytes, TMA_SHMEM_ALIGNMENT);
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sDh_tile_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sH1_tile_ptr = reinterpret_cast<IType*>(dshmem + tile_bytes_aligned);
    IType* sH3_tile_ptr = reinterpret_cast<IType*>(dshmem + 2 * tile_bytes_aligned);
    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem + 3 * tile_bytes_aligned);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 3 * tile_bytes_aligned + in_bytes);
    fp4e2m1x2* sOut_tr_ptr =
        reinterpret_cast<fp4e2m1x2*>(dshmem + 3 * tile_bytes_aligned + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 3 * tile_bytes_aligned + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 3 * tile_bytes_aligned + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t dh_mbar;
    __shared__ uint64_t h1_mbar;
    __shared__ uint64_t h3_mbar;
    __shared__ float warp_max[THREADS / 32];
    __shared__ float cta_amax_shared;
    if (leading) {
        ptx::mbarrier_init(&dh_mbar, 1);
        ptx::mbarrier_init(&h1_mbar, 1);
        ptx::mbarrier_init(&h3_mbar, 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;

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
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;

        const bool output_dh1 = ctaid_X < split0_tiles;
        const int local_ctaid_X = output_dh1 ? ctaid_X : (ctaid_X - split0_tiles);
        const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
        const int chunk_cols = static_cast<int>(cols) - block_offset_X;
        const int chunk_scales_x = max(0, min((int)SCALES_PER_CHUNK_X,
                                              (chunk_cols + SCALE_DIM - 1) / SCALE_DIM));
        const int chunk_scales_y = max(0, min((int)SCALES_PER_CHUNK_Y,
                                              (chunk_rows + SCALE_DIM - 1) / SCALE_DIM));

        float cta_max = 0.0f;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int tile_offset_Y = block_offset_Y + stage_Y * TILE_DIM_Y;
            const int tile_offset_X = input_block_offset_X + stage_X * TILE_DIM_X;

            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&dh_mbar, shmem_tile_bytes);
                ptx::mbarrier_arrive_expect_tx(&h1_mbar, shmem_tile_bytes);
                ptx::mbarrier_arrive_expect_tx(&h3_mbar, shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(sDh_tile_ptr),
                    reinterpret_cast<const uint64_t*>(&tensor_map_dh),
                    tile_offset_X, tile_offset_Y, &dh_mbar);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(sH1_tile_ptr),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                    tile_offset_X, tile_offset_Y, &h1_mbar);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(sH3_tile_ptr),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                    tile_offset_X, tile_offset_Y, &h3_mbar);
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar, mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar, mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar, mbar_phase);

            cta_max = fmaxf(
                cta_max,
                output_dh1
                    ? transform_silu_deriv_tile_to_chunk_local<true>(
                          sIn_ptr, sDh_tile_ptr, sH1_tile_ptr, sH3_tile_ptr, t)
                    : transform_silu_deriv_tile_to_chunk_local<false>(
                          sIn_ptr, sDh_tile_ptr, sH1_tile_ptr, sH3_tile_ptr, t));
            __syncthreads();
            mbar_phase ^= 1;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }
        if (lane == 0) {
            warp_max[wid] = cta_max;
        }
        __syncthreads();
        if (wid == 0) {
            float warp_val = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                warp_val = fmaxf(warp_val, __shfl_xor_sync(0xffffffff, warp_val, mask));
            }
            if (lane == 0) {
                cta_amax_shared = warp_val;
                row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] =
                    warp_val / localcta_global_scale_num();
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] =
                        warp_val / localcta_global_scale_num();
                }
            }
        }
        __syncthreads();

        const float amax_val = cta_amax_shared;
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

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
                colwise_scaling<true>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc, stage_Y, stage_X, t, buff_out_tr);
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
                        block_offset_Y_tr + stage_offset_X,
                        block_offset_X_tr + stage_offset_Y,
                        reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                }
                ptx::cp_async_bulk_commit_group();
            }

            buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
            if constexpr (RETURN_TRANSPOSE) {
                buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
            }
        }

        if (leading) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
        swizzle_scales_row_inplace(sSFrowwise_ptr, chunk_scales_x);
        scale_swizzled_scales_inplace(
            sSFrowwise_ptr,
            LocalCTAConfig::CHUNK_DIM_Y * chunk_scales_x,
            sg_val);
        ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            tma_store_scales_2x512(tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
        }

        if constexpr (RETURN_TRANSPOSE) {
            __syncthreads();
            swizzle_scales_col_inplace(sSFcolwise_ptr, chunk_scales_y);
            scale_swizzled_scales_inplace(
                sSFcolwise_ptr,
                LocalCTAConfig::CHUNK_DIM_X * chunk_scales_y,
                sg_val);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();
        }
    }

    if (leading) {
        ptx::mbarrier_invalid(&dh_mbar);
        ptx::mbarrier_invalid(&h1_mbar);
        ptx::mbarrier_invalid(&h3_mbar);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_split3_prepared_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    LocalCTAPersistentArgs args,
    int split0_tiles,
    int split1_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes =
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    __shared__ float warp_max[THREADS / 32];
    __shared__ float cta_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;
    const int split01_tiles = split0_tiles + split1_tiles;

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
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
        const int chunk_cols = static_cast<int>(cols) - block_offset_X;

        int local_ctaid_X = ctaid_X;
        int split_id = 0;
        if (ctaid_X >= split01_tiles) {
            split_id = 2;
            local_ctaid_X -= split01_tiles;
        } else if (ctaid_X >= split0_tiles) {
            split_id = 1;
            local_ctaid_X -= split0_tiles;
        }
        const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

        float cta_max = 0.0f;

        #pragma unroll
        for (int pre = 0; pre < min(2, (int)NUM_TILES); ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                if (split_id == 0) {
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[pre]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input0),
                        input_block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[pre]);
                } else if (split_id == 1) {
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[pre]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input1),
                        input_block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[pre]);
                } else {
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[pre]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input2),
                        input_block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[pre]);
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            if (t + 2 < NUM_TILES) {
                const int next = t + 2;
                const int ty = next / TILES_X;
                const int tx = next % TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                    if (split_id == 0) {
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn[next]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input0),
                            input_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[next]);
                    } else if (split_id == 1) {
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn[next]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input1),
                            input_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[next]);
                    } else {
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn[next]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input2),
                            input_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[next]);
                    }
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }

        if (lane == 0) {
            warp_max[wid] = cta_max;
        }
        __syncthreads();

        if (wid == 0) {
            cta_max = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                cta_amax_shared = cta_max;
            }
        }
        __syncthreads();

        const float amax_val = cta_amax_shared;
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
            col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
        }

        int buff_out = 0;
        int buff_out_tr = 0;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int stage_offset_Y = stage_Y * TILE_DIM_Y;
            const int stage_offset_X = stage_X * TILE_DIM_X;

            if (t > 0) {
                ptx::cp_async_bulk_wait_group_read<1>();
            }

            rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                                            S_enc, stage_Y, stage_X, t, buff_out);
            colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                            S_enc, stage_Y, stage_X, t, buff_out_tr);

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    block_offset_X + stage_offset_X,
                    block_offset_Y + stage_offset_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));

                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
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

        {
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
        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS) __cluster_dims__(2, 1, 1)
fused_localcta_quantize_kernel_2cta(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    float* __restrict__ cluster_amax_scratch,
    const size_t rows, const size_t cols,
    LocalCTA2ClusterArgs args,
    bool write_raw_scales,
    bool write_prepared
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

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
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int cta_rank = blockIdx.x & 1;
    const int owner_block = blockIdx.x & ~1;
    __shared__ float warp_max[THREADS / 32];
    __shared__ float cluster_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;

    while (true) {
        if (leading) {
            if (cta_rank == 0) {
                cluster_amax_scratch[owner_block] = static_cast<float>(atomicAdd(args.work_counter, 1));
            } else {
                cluster_amax_scratch[blockIdx.x] = 0.0f;
            }
            __threadfence();
        }
        cluster_sync_aligned();

        const int macro_tile_id = static_cast<int>(cluster_amax_scratch[owner_block]);
        if (macro_tile_id >= args.total_macro_tiles) {
            break;
        }

        const int macro_ctaid_Y = macro_tile_id / args.tiles_X;
        const int ctaid_X = macro_tile_id % args.tiles_X;
        const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
        const bool cta_active = ctaid_Y < args.tiles_Y;

        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = cta_active ? static_cast<int>(rows) - block_offset_Y : 0;
        const int chunk_cols = cta_active ? static_cast<int>(cols) - block_offset_X : 0;

        float cta_max = 0.0f;
        if (cta_active) {
            #pragma unroll
            for (int pre = 0; pre < min(2, (int)NUM_TILES); ++pre) {
                const int ty = pre / TILES_X;
                const int tx = pre % TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[pre]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[pre]);
                }
            }

            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                if (t + 2 < NUM_TILES) {
                    const int next = t + 2;
                    const int ty = next / TILES_X;
                    const int tx = next % TILES_X;
                    if (leading) {
                        ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn[next]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input),
                            block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[next]);
                    }
                }

                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
                cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
            }

            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
        }

        if (lane == 0) {
            warp_max[wid] = cta_max;
        }
        __syncthreads();

        if (wid == 0) {
            cta_max = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                cluster_amax_scratch[blockIdx.x] = cta_max;
            }
        }
        __syncthreads();

        if (leading) {
            __threadfence();
        }
        cluster_sync_aligned();

        if (leading) {
            cluster_amax_shared = fmaxf(cluster_amax_scratch[owner_block],
                                        cluster_amax_scratch[owner_block + 1]);
        }
        __syncthreads();

        if (cta_active) {
            const float amax_val = cluster_amax_shared;
            const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
            const float sg_val = amax_val / localcta_global_scale_num();

            if (leading) {
                row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
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

                if (t > 0) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }

                rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                                                S_enc, stage_Y, stage_X, t, buff_out);

                if constexpr (RETURN_TRANSPOSE) {
                    colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
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
                if (write_raw_scales && leading) {
                    tma_store_scales_2x512(tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                }
                if (write_prepared) {
                    if (write_raw_scales && leading) {
                        ptx::cp_async_bulk_wait_group_read<0>();
                    }
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
            }

            if constexpr (RETURN_TRANSPOSE) {
                const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
                swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
                ptx::fence_proxy_async_shared_cta();
                __syncthreads();
                if (write_raw_scales && leading) {
                    tma_store_scales_2x512(tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
                if (write_prepared) {
                    if (write_raw_scales && leading) {
                        ptx::cp_async_bulk_wait_group_read<0>();
                    }
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
            }

            if (leading && (write_raw_scales || write_prepared)) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();
        }

        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS)
fused_localcta_quantize_split2_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int total_tiles,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS > 128 && TOTAL_THREADS <= 256 && TOTAL_THREADS % 32 == 0,
                  "tuned split2 1CTA prepared kernel expects 128 consumer threads plus at least one producer warp");

    constexpr int CONSUMER_THREADS = 128;
    constexpr int NUM_TILES_PER_CHUNK = BUFFS_NUM_IN;
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
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
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ring_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sIn_ring_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ int slot_ready[PIPE_DEPTH];
    __shared__ int slot_phase[PIPE_DEPTH];
    __shared__ int slot_tile_id[PIPE_DEPTH];
    __shared__ float slot_amax[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_tile_id[s] = -1;
            slot_amax[s] = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_init(&in_mbar[s][t], 1);
            }
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int block_iters = (blockIdx.x < total_tiles)
        ? ((total_tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1)
        : 0;

    if (producer_leader) {
        for (int fill_iter = 0; fill_iter < block_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 0) {
                __nanosleep(64);
            }

            const int tile_id = (int)blockIdx.x + fill_iter * (int)gridDim.x;
            slot_tile_id[slot] = tile_id;

            const int ctaid_Y = tile_id / tiles_X;
            const int ctaid_X = tile_id % tiles_X;
            int local_ctaid_X = ctaid_X;
            int split_id = 0;
            if (ctaid_X >= split0_tiles) {
                split_id = 1;
                local_ctaid_X -= split0_tiles;
            }
            const int input_block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                const int ty = t / TILES_X;
                const int tx = t % TILES_X;
                ptx::mbarrier_arrive_expect_tx(&in_mbar[slot][t], shmem_tile_bytes);
                if (split_id == 0) {
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input0),
                        input_block_offset_X + tx * TILE_DIM_X,
                        input_block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[slot][t]);
                } else {
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input1),
                        input_block_offset_X + tx * TILE_DIM_X,
                        input_block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[slot][t]);
                }
            }

            __threadfence_block();
            slot_ready[slot] = 1;
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < block_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] == 0) {
                __nanosleep(64);
            }

            const int tile_id = slot_tile_id[slot];
            const int ctaid_Y = tile_id / tiles_X;
            const int ctaid_X = tile_id % tiles_X;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
            const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
            const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
            const int chunk_cols = static_cast<int>(cols) - block_offset_X;

            float cta_max = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[slot][t], slot_phase[slot]);
                cta_max = fmaxf(cta_max, scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
            }

            const int lane = consumer_tid % 32;
            const int wid = consumer_tid / 32;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                warp_max[wid] = cta_max;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if (wid == 0) {
                cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                #pragma unroll
                for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                }
                if (lane == 0) {
                    slot_amax[slot] = cta_max;
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] = cta_max / localcta_global_scale_num();
                    if constexpr (RETURN_TRANSPOSE) {
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] = cta_max / localcta_global_scale_num();
                    }
                }
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            const float amax_val = slot_amax[slot];
            const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
            const float sg_val = amax_val / localcta_global_scale_num();

            int buff_out = 0;
            int buff_out_tr = 0;

            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                const int stage_Y = t / TILES_X;
                const int stage_X = t % TILES_X;
                const int stage_offset_Y = stage_Y * TILE_DIM_Y;
                const int stage_offset_X = stage_X * TILE_DIM_X;

                if (t > 0 && consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }
                if (t > 0) {
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                }

                rowwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                    &sIn_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                    S_enc, stage_Y, stage_X, t, buff_out, consumer_tid);

                if constexpr (RETURN_TRANSPOSE) {
                    colwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                        S_enc, stage_Y, stage_X, t, buff_out_tr, consumer_tid);
                }

                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();

                if (consumer_leader) {
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
                tma_store_scales_2x512(
                    tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
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
                    tma_store_scales_2x512(
                        tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
            }

            if (consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<0>();
                slot_phase[slot] ^= 1;
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_invalid(&in_mbar[s][t]);
            }
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX,
          bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS) __cluster_dims__(2, 1, 1)
fused_localcta_quantize_split2_kernel_2cta_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int tiles_Y,
    int total_macro_tiles,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS >= 160 && TOTAL_THREADS <= 512 && TOTAL_THREADS % 32 == 0,
                  "tuned split2 2CTA prepared kernel expects at least 128 consumer threads plus producer warp(s)");

    constexpr int CONSUMER_THREADS = 128;
    constexpr int NUM_TILES_PER_CHUNK = BUFFS_NUM_IN;
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
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
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ring_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sIn_ring_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ int slot_ready[PIPE_DEPTH];
    __shared__ int slot_phase[PIPE_DEPTH];
    __shared__ int slot_macro_id[PIPE_DEPTH];
    __shared__ float slot_local_amax[PIPE_DEPTH];
    __shared__ uint32_t slot_local_ready[PIPE_DEPTH];
    __shared__ float slot_combined_amax[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_macro_id[s] = -1;
            slot_local_amax[s] = 0.0f;
            slot_local_ready[s] = 0;
            slot_combined_amax[s] = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_init(&in_mbar[s][t], 1);
            }
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int cta_rank = blockIdx.x & 1;
    const int peer_cta = cta_rank ^ 1;
    const int cluster_id = blockIdx.x >> 1;
    const int num_clusters = max(1, (int)(gridDim.x >> 1));
    const int cluster_iters = (cluster_id < total_macro_tiles)
        ? ((total_macro_tiles - 1 - cluster_id) / num_clusters + 1)
        : 0;

    if (producer_leader) {
        for (int fill_iter = 0; fill_iter < cluster_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 0) {
                __nanosleep(64);
            }

            const int macro_tile_id = cluster_id + fill_iter * num_clusters;
            slot_macro_id[slot] = macro_tile_id;
            slot_local_ready[slot] = 0;

            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            int local_ctaid_X = ctaid_X;
            int split_id = 0;
            if (ctaid_X >= split0_tiles) {
                split_id = 1;
                local_ctaid_X -= split0_tiles;
            }
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            if (cta_active) {
                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    const int ty = t / TILES_X;
                    const int tx = t % TILES_X;
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[slot][t], shmem_tile_bytes);
                    if (split_id == 0) {
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input0),
                            input_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[slot][t]);
                    } else {
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input1),
                            input_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[slot][t]);
                    }
                }
            }

            __threadfence_block();
            slot_ready[slot] = 1;
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < cluster_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] == 0) {
                __nanosleep(64);
            }

            const int macro_tile_id = slot_macro_id[slot];
            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;

            if (cta_active) {
                const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
                const int chunk_cols = static_cast<int>(cols) - block_offset_X;

                float cta_max = 0.0f;
                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[slot][t], slot_phase[slot]);
                    cta_max = fmaxf(cta_max, scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
                }

                const int lane = consumer_tid % 32;
                const int wid = consumer_tid / 32;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                }
                if (lane == 0) {
                    warp_max[wid] = cta_max;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();

                if (wid == 0) {
                    cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                    #pragma unroll
                    for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                        cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                    }
                    if (lane == 0) {
                        slot_local_amax[slot] = cta_max;
                        __threadfence_block();
                        slot_local_ready[slot] = 1;
                    }
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();

                float amax_val = slot_local_amax[slot];
                if constexpr (SHARED_AMAX) {
                    if (consumer_leader) {
                        if (ctaid_Y + 1 >= tiles_Y) {
                            slot_combined_amax[slot] = slot_local_amax[slot];
                        } else {
                            while (cluster_load_shared_u32(&slot_local_ready[slot], peer_cta) == 0u) {
                                __nanosleep(64);
                            }
                            const float peer_amax = cluster_load_shared_f32(&slot_local_amax[slot], peer_cta);
                            slot_combined_amax[slot] = fmaxf(slot_local_amax[slot], peer_amax);
                        }
                    }
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    amax_val = slot_combined_amax[slot];
                }

                const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
                const float sg_val = amax_val / localcta_global_scale_num();

                if (consumer_leader) {
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] = sg_val;
                    if constexpr (RETURN_TRANSPOSE) {
                        col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
                    }
                }

                int buff_out = 0;
                int buff_out_tr = 0;

                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    const int stage_Y = t / TILES_X;
                    const int stage_X = t % TILES_X;
                    const int stage_offset_Y = stage_Y * TILE_DIM_Y;
                    const int stage_offset_X = stage_X * TILE_DIM_X;

                    if (t > 0 && consumer_leader) {
                        ptx::cp_async_bulk_wait_group_read<1>();
                    }
                    if (t > 0) {
                        subgroup_barrier_sync<CONSUMER_THREADS>();
                    }

                    rowwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                        S_enc, stage_Y, stage_X, t, buff_out, consumer_tid);

                    if constexpr (RETURN_TRANSPOSE) {
                        colwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                            &sIn_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                            S_enc, stage_Y, stage_X, t, buff_out_tr, consumer_tid);
                    }

                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();

                    if (consumer_leader) {
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
                    tma_store_scales_2x512(
                        tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
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
                        tma_store_scales_2x512(
                            tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }

                if (consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
            }

            if (consumer_leader) {
                slot_phase[slot] ^= 1;
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_invalid(&in_mbar[s][t]);
            }
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

__device__ __forceinline__ void store_chunk_value_group(
    IType* sIn_ptr,
    int tile,
    int row,
    int col,
    __nv_bfloat16 value
) {
    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    sIn[tile][row][col] = value;
}

template <bool OUTPUT_DH1, int GROUP_THREADS>
__device__ __forceinline__ float transform_store_silu_deriv_split2_row_tile_group(
    IType* sIn_ptr,
    __nv_bfloat16* __restrict__ out_bf16,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int input_block_offset_X,
    int tile_idx,
    int tid
) {
    constexpr int VEC = 4;
    constexpr int TILE_ELTS = TILE_DIM_Y * TILE_DIM_X;

    const int stage_Y = tile_idx / TILES_X;
    const int stage_X = tile_idx % TILES_X;
    const int tile_offset_Y = stage_Y * TILE_DIM_Y;
    const int tile_offset_X = stage_X * TILE_DIM_X;

    float local_max = 0.0f;

    for (int idx = tid * VEC; idx < TILE_ELTS; idx += GROUP_THREADS * VEC) {
        const int local_row = idx / TILE_DIM_X;
        const int local_col = idx % TILE_DIM_X;
        const int row = tile_offset_Y + local_row;
        const int col = tile_offset_X + local_col;
        const int global_row = block_offset_Y + row;
        const int global_col = input_block_offset_X + col;

        if (global_row < rows && global_col + (VEC - 1) < cols) {
            const int64_t base = static_cast<int64_t>(global_row) * cols + global_col;

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

            const float sig0x = 1.0f / (1.0f + expf(-b0f.x));
            const float sig0y = 1.0f / (1.0f + expf(-b0f.y));
            const float sig1x = 1.0f / (1.0f + expf(-b1f.x));
            const float sig1y = 1.0f / (1.0f + expf(-b1f.y));

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

            *reinterpret_cast<__nv_bfloat162*>(out_bf16 + base) = out0;
            *reinterpret_cast<__nv_bfloat162*>(out_bf16 + base + 2) = out1;

            const float2 out0f = __bfloat1622float2(out0);
            const float2 out1f = __bfloat1622float2(out1);
            local_max = fmaxf(local_max, fabsf(out0f.x));
            local_max = fmaxf(local_max, fabsf(out0f.y));
            local_max = fmaxf(local_max, fabsf(out1f.x));
            local_max = fmaxf(local_max, fabsf(out1f.y));

            store_chunk_value_group(sIn_ptr, tile_idx, local_row, local_col + 0, out0.x);
            store_chunk_value_group(sIn_ptr, tile_idx, local_row, local_col + 1, out0.y);
            store_chunk_value_group(sIn_ptr, tile_idx, local_row, local_col + 2, out1.x);
            store_chunk_value_group(sIn_ptr, tile_idx, local_row, local_col + 3, out1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                const int local_c = local_col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && input_block_offset_X + tile_offset_X + local_c < cols) {
                    const int64_t offset =
                        static_cast<int64_t>(global_row) * cols + input_block_offset_X + tile_offset_X + local_c;
                    const float vd = __bfloat162float(dh[offset]);
                    const float v1 = __bfloat162float(h1_raw[offset]);
                    const float v3 = __bfloat162float(h3[offset]);
                    const float sig = 1.0f / (1.0f + expf(-v1));
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
                    out_bf16[offset] = out;
                }
                store_chunk_value_group(sIn_ptr, tile_idx, local_row, local_c, out);
            }
        }
    }

    return local_max;
}

template <int GROUP_THREADS = 128, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(GROUP_THREADS)
fused_localcta_silu_deriv_split2_row_bf16_prepared_tuned(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    __nv_bfloat16* __restrict__ dh1_out,
    __nv_bfloat16* __restrict__ dh3_out,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    float* __restrict__ row_sg_chunks,
    const size_t rows,
    const size_t split_cols,
    LocalCTAPersistentArgs args,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(GROUP_THREADS == 128, "split2 row+bfloat tuned kernel expects 128 consumer threads");

    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT));
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(
        dshmem + DIVUP_TO_MULTIPLE(BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)
        + out_bytes);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);

    __shared__ unsigned int s_chunk_id;
    __shared__ float warp_max[GROUP_THREADS / 32];
    __shared__ float cta_amax_shared;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wid = tid / 32;
    const bool consumer_leader = (tid == 0);

    while (true) {
        if (consumer_leader) {
            s_chunk_id = atomicAdd(args.work_counter, 1u);
        }
        subgroup_barrier_sync<GROUP_THREADS>();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int tile_id = static_cast<int>(s_chunk_id);
        const int ctaid_Y = tile_id / args.tiles_X;
        const int ctaid_X = tile_id % args.tiles_X;
        const bool output_dh1 = (ctaid_X < split0_tiles);
        const int local_ctaid_X = output_dh1 ? ctaid_X : (ctaid_X - split0_tiles);
        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int chunk_cols = static_cast<int>(split_cols) - input_block_offset_X;

        float local_max = 0.0f;
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            local_max = fmaxf(
                local_max,
                output_dh1
                    ? transform_store_silu_deriv_split2_row_tile_group<true, GROUP_THREADS>(
                        sIn_ptr, dh1_out, dh, h3, h1_raw,
                        static_cast<int>(rows), static_cast<int>(split_cols),
                        block_offset_Y, input_block_offset_X, t, tid)
                    : transform_store_silu_deriv_split2_row_tile_group<false, GROUP_THREADS>(
                        sIn_ptr, dh3_out, dh, h3, h1_raw,
                        static_cast<int>(rows), static_cast<int>(split_cols),
                        block_offset_Y, input_block_offset_X, t, tid));
        }
        subgroup_barrier_sync<GROUP_THREADS>();

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
        }
        if (lane == 0) {
            warp_max[wid] = local_max;
        }
        subgroup_barrier_sync<GROUP_THREADS>();

        if (wid == 0) {
            float cta_max = (lane < GROUP_THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (GROUP_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                cta_amax_shared = cta_max;
                row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = cta_max / localcta_global_scale_num();
            }
        }
        subgroup_barrier_sync<GROUP_THREADS>();

        const float amax_val = cta_amax_shared;
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        int buff_out = 0;
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int stage_offset_Y = stage_Y * TILE_DIM_Y;
            const int stage_offset_X = stage_X * TILE_DIM_X;

            if (t > 0 && consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<1>();
            }
            if (t > 0) {
                subgroup_barrier_sync<GROUP_THREADS>();
            }

            rowwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
                sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                S_enc, stage_Y, stage_X, t, buff_out, tid);

            subgroup_barrier_sync<GROUP_THREADS>();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            subgroup_barrier_sync<GROUP_THREADS>();

            if (consumer_leader) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    block_offset_X + stage_offset_X,
                    block_offset_Y + stage_offset_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                ptx::cp_async_bulk_commit_group();
            }

            buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
        }

        if (consumer_leader) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<GROUP_THREADS>();

        swizzle_scales_row_inplace_group<GROUP_THREADS>(
            sSFrowwise_ptr,
            min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM),
            tid);
        scale_swizzled_scales_inplace_group<GROUP_THREADS>(
            sSFrowwise_ptr,
            LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
            sg_val,
            tid);
        subgroup_barrier_sync<GROUP_THREADS>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        subgroup_barrier_sync<GROUP_THREADS>();
        if (consumer_leader) {
            tma_store_scales_2x512(
                tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<GROUP_THREADS>();
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int PIPE_DEPTH, bool RETURN_TRANSPOSE>
inline int prepared_2cta_tuned_shmem_size() {
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int input_ring_bytes = PIPE_DEPTH * slot_in_bytes;
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
    return input_ring_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <int PIPE_DEPTH, bool RETURN_TRANSPOSE>
inline int prepared_1cta_tuned_shmem_size() {
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int input_ring_bytes = PIPE_DEPTH * slot_in_bytes;
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
    return input_ring_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS)
fused_localcta_quantize_kernel_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int total_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS > 128 && TOTAL_THREADS <= 256 && TOTAL_THREADS % 32 == 0,
                  "tuned 1CTA prepared kernel expects 128 consumer threads plus at least one producer warp");

    constexpr int CONSUMER_THREADS = 128;
    constexpr int NUM_TILES_PER_CHUNK = BUFFS_NUM_IN;
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
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
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ring_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sIn_ring_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ int slot_ready[PIPE_DEPTH];
    __shared__ int slot_phase[PIPE_DEPTH];
    __shared__ int slot_tile_id[PIPE_DEPTH];
    __shared__ float slot_amax[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_tile_id[s] = -1;
            slot_amax[s] = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_init(&in_mbar[s][t], 1);
            }
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int block_iters = (blockIdx.x < total_tiles)
        ? ((total_tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1)
        : 0;

    if (producer_leader) {
        for (int fill_iter = 0; fill_iter < block_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 0) {
                __nanosleep(64);
            }

            const int tile_id = (int)blockIdx.x + fill_iter * (int)gridDim.x;
            slot_tile_id[slot] = tile_id;

            const int ctaid_Y = tile_id / tiles_X;
            const int ctaid_X = tile_id % tiles_X;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                const int ty = t / TILES_X;
                const int tx = t % TILES_X;
                ptx::mbarrier_arrive_expect_tx(&in_mbar[slot][t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &in_mbar[slot][t]);
            }

            __threadfence_block();
            slot_ready[slot] = 1;
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < block_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] == 0) {
                __nanosleep(64);
            }

            const int tile_id = slot_tile_id[slot];
            const int ctaid_Y = tile_id / tiles_X;
            const int ctaid_X = tile_id % tiles_X;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
            const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
            const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
            const int chunk_cols = static_cast<int>(cols) - block_offset_X;

            float cta_max = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[slot][t], slot_phase[slot]);
                cta_max = fmaxf(cta_max, scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
            }

            const int lane = consumer_tid % 32;
            const int wid = consumer_tid / 32;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                warp_max[wid] = cta_max;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if (wid == 0) {
                cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                #pragma unroll
                for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                }
                if (lane == 0) {
                    slot_amax[slot] = cta_max;
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] = cta_max / localcta_global_scale_num();
                    if constexpr (RETURN_TRANSPOSE) {
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] = cta_max / localcta_global_scale_num();
                    }
                }
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            const float amax_val = slot_amax[slot];
            const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
            const float sg_val = amax_val / localcta_global_scale_num();

            int buff_out = 0;
            int buff_out_tr = 0;

            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                const int stage_Y = t / TILES_X;
                const int stage_X = t % TILES_X;
                const int stage_offset_Y = stage_Y * TILE_DIM_Y;
                const int stage_offset_X = stage_X * TILE_DIM_X;

                if (t > 0 && consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }

                rowwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                    &sIn_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                    S_enc, stage_Y, stage_X, t, buff_out, consumer_tid);

                if constexpr (RETURN_TRANSPOSE) {
                    colwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                        S_enc, stage_Y, stage_X, t, buff_out_tr, consumer_tid);
                }

                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();

                if (consumer_leader) {
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
                tma_store_scales_2x512(
                    tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
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
                    tma_store_scales_2x512(
                        tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
            }

            if (consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<0>();
                slot_phase[slot] ^= 1;
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_invalid(&in_mbar[s][t]);
            }
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX,
          bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS) __cluster_dims__(2, 1, 1)
fused_localcta_quantize_kernel_2cta_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int tiles_Y,
    int total_macro_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS >= 160 && TOTAL_THREADS <= 512 && TOTAL_THREADS % 32 == 0,
                  "tuned 2CTA prepared kernel expects at least 128 consumer threads plus producer warp(s)");

    constexpr int CONSUMER_THREADS = 128;
    constexpr int NUM_TILES_PER_CHUNK = BUFFS_NUM_IN;
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
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
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ring_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sIn_ring_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ int slot_ready[PIPE_DEPTH];
    __shared__ int slot_phase[PIPE_DEPTH];
    __shared__ int slot_macro_id[PIPE_DEPTH];
    __shared__ float slot_local_amax[PIPE_DEPTH];
    __shared__ uint32_t slot_local_ready[PIPE_DEPTH];
    __shared__ float slot_combined_amax[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_macro_id[s] = -1;
            slot_local_amax[s] = 0.0f;
            slot_local_ready[s] = 0;
            slot_combined_amax[s] = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_init(&in_mbar[s][t], 1);
            }
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int cta_rank = blockIdx.x & 1;
    const int peer_cta = cta_rank ^ 1;
    const int cluster_id = blockIdx.x >> 1;
    const int num_clusters = max(1, (int)(gridDim.x >> 1));
    const int cluster_iters = (cluster_id < total_macro_tiles)
        ? ((total_macro_tiles - 1 - cluster_id) / num_clusters + 1)
        : 0;

    if (producer_leader) {
        for (int fill_iter = 0; fill_iter < cluster_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 0) {
                __nanosleep(64);
            }

            const int macro_tile_id = cluster_id + fill_iter * num_clusters;
            slot_macro_id[slot] = macro_tile_id;
            slot_local_ready[slot] = 0;

            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            if (cta_active) {
                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    const int ty = t / TILES_X;
                    const int tx = t % TILES_X;
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[slot][t], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[slot][t]);
                }
            }

            __threadfence_block();
            slot_ready[slot] = 1;
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < cluster_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] == 0) {
                __nanosleep(64);
            }

            const int macro_tile_id = slot_macro_id[slot];
            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;

            if (cta_active) {
                const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
                const int chunk_cols = static_cast<int>(cols) - block_offset_X;

                float cta_max = 0.0f;
                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[slot][t], slot_phase[slot]);
                    cta_max = fmaxf(cta_max, scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
                }

                const int lane = consumer_tid % 32;
                const int wid = consumer_tid / 32;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                }
                if (lane == 0) {
                    warp_max[wid] = cta_max;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();

                if (wid == 0) {
                    cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                    #pragma unroll
                    for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                        cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                    }
                    if (lane == 0) {
                        slot_local_amax[slot] = cta_max;
                        __threadfence_block();
                        slot_local_ready[slot] = 1;
                    }
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();

                float amax_val = slot_local_amax[slot];
                if constexpr (SHARED_AMAX) {
                    if (consumer_leader) {
                        while (cluster_load_shared_u32(&slot_local_ready[slot], peer_cta) == 0u) {
                            __nanosleep(64);
                        }
                        const float peer_amax = cluster_load_shared_f32(&slot_local_amax[slot], peer_cta);
                        slot_combined_amax[slot] = fmaxf(slot_local_amax[slot], peer_amax);
                    }
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    amax_val = slot_combined_amax[slot];
                }

                const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
                const float sg_val = amax_val / localcta_global_scale_num();
                if (consumer_leader) {
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] = sg_val;
                    if constexpr (RETURN_TRANSPOSE) {
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] = sg_val;
                    }
                }

                int buff_out = 0;
                int buff_out_tr = 0;

                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    const int stage_Y = t / TILES_X;
                    const int stage_X = t % TILES_X;
                    const int stage_offset_Y = stage_Y * TILE_DIM_Y;
                    const int stage_offset_X = stage_X * TILE_DIM_X;

                    if (t > 0 && consumer_leader) {
                        ptx::cp_async_bulk_wait_group_read<1>();
                    }

                    rowwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                        S_enc, stage_Y, stage_X, t, buff_out, consumer_tid);

                    if constexpr (RETURN_TRANSPOSE) {
                        colwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                            &sIn_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                            S_enc, stage_Y, stage_X, t, buff_out_tr, consumer_tid);
                    }

                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();

                    if (consumer_leader) {
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
                    tma_store_scales_2x512(
                        tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
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
                        tma_store_scales_2x512(
                            tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }

                if (consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                    slot_local_ready[slot] = 0;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
            }

            if (consumer_leader) {
                slot_phase[slot] ^= 1;
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_invalid(&in_mbar[s][t]);
            }
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
inline int shmem_size() {
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

    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace tk_localcta

#endif  // TK_LOCALCTA_FUSED_QUANTIZE_CUH_
