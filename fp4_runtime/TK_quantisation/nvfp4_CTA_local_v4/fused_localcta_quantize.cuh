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
static constexpr int SILU_RAW_THREADS = 256;
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

template <bool WITH_SILU>
__device__ __forceinline__ float rmsnorm_contract_value(float x, float inv_rms, float gamma) {
    float out = __bfloat162float(__float2bfloat16_rn(x * inv_rms * gamma));
    if constexpr (WITH_SILU) {
        out = __bfloat162float(__float2bfloat16_rn(out / (1.0f + expf(-out))));
    }
    return out;
}

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

struct RopeLive64Desc {
    const float2* cs;
    int seq_mask;
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

// CUDA's correctly-rounded f32 division lowers to an out-of-line exceptional
// path in the CUDA 13.2 x86 SM100a compiler.  The paired-RHT W2 kernel is large
// enough that those calls also reserve a per-thread caller frame.  Keep this
// leaf implementation private to that specialization: ordinary localCTA
// routes continue to use the original CUDA division expressions below.
__device__ __forceinline__ float localcta_scale_quiet_nan_callfree(
    uint32_t numerator_bits,
    uint32_t denominator_bits
) {
    constexpr uint32_t kExponentMask = 0x7f800000u;
    constexpr uint32_t kMantissaMask = 0x007fffffu;
    constexpr uint32_t kQuietBit = 0x00400000u;
    const uint32_t numerator_abs = numerator_bits & 0x7fffffffu;
    uint32_t bits = numerator_abs > kExponentMask
        ? numerator_bits
        : denominator_bits;
    bits |= kQuietBit;
    if ((bits & kMantissaMask) == 0u) {
        bits |= kQuietBit;
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ float localcta_scale_divide_callfree(
    float numerator,
    float denominator
) {
    constexpr uint32_t kSignMask = 0x80000000u;
    constexpr uint32_t kInf = 0x7f800000u;
    constexpr uint32_t kQuietNaN = 0x7fc00000u;
    const uint32_t numerator_bits = __float_as_uint(numerator);
    const uint32_t denominator_bits = __float_as_uint(denominator);
    const uint32_t numerator_abs_bits = numerator_bits & ~kSignMask;
    const uint32_t denominator_abs_bits = denominator_bits & ~kSignMask;
    const uint32_t quotient_sign =
        (numerator_bits ^ denominator_bits) & kSignMask;

    if (numerator_abs_bits > kInf || denominator_abs_bits > kInf) {
        return localcta_scale_quiet_nan_callfree(
            numerator_bits, denominator_bits);
    }
    if (denominator_abs_bits == 0u) {
        return __uint_as_float(
            numerator_abs_bits == 0u ? kQuietNaN : (quotient_sign | kInf));
    }
    if (numerator_abs_bits == kInf) {
        return __uint_as_float(
            denominator_abs_bits == kInf ? kQuietNaN : (quotient_sign | kInf));
    }
    if (denominator_abs_bits == kInf || numerator_abs_bits == 0u) {
        return __uint_as_float(quotient_sign);
    }

    // A float quotient's complete exponent range is normal in binary64.  Do
    // the reciprocal and residual correction there so finite underflow and
    // overflow never need CUDA's out-of-line f32 exceptional path.  Two
    // Newton steps leave ample guard bits before the final correctly-rounded
    // f64-to-f32 conversion.
    const double numerator_abs = static_cast<double>(
        __uint_as_float(numerator_abs_bits));
    const double denominator_abs = static_cast<double>(
        __uint_as_float(denominator_abs_bits));
    double reciprocal;
    asm("rcp.approx.ftz.f64 %0, %1;"
        : "=d"(reciprocal)
        : "d"(denominator_abs));
    double reciprocal_error = fma(-denominator_abs, reciprocal, 1.0);
    reciprocal = fma(reciprocal_error, reciprocal, reciprocal);
    reciprocal_error = fma(-denominator_abs, reciprocal, 1.0);
    reciprocal = fma(reciprocal_error, reciprocal, reciprocal);

    double quotient = numerator_abs * reciprocal;
    const double quotient_error = fma(
        -denominator_abs, quotient, numerator_abs);
    quotient = fma(quotient_error, reciprocal, quotient);
    const float quotient_f32 = static_cast<float>(quotient);
    return __uint_as_float(__float_as_uint(quotient_f32) | quotient_sign);
}

__device__ __forceinline__ float localcta_scale_reciprocal_callfree(float value) {
    return localcta_scale_divide_callfree(1.0f, value);
}

template <bool CALL_FREE_SCALE_MATH>
__device__ __forceinline__ float localcta_scale_divide(
    float numerator,
    float denominator
) {
    if constexpr (CALL_FREE_SCALE_MATH) {
        return localcta_scale_divide_callfree(numerator, denominator);
    } else {
        return numerator / denominator;
    }
}

template <bool CALL_FREE_SCALE_MATH>
__device__ __forceinline__ float localcta_scale_reciprocal(float value) {
    if constexpr (CALL_FREE_SCALE_MATH) {
        return localcta_scale_reciprocal_callfree(value);
    } else {
        return 1.0f / value;
    }
}

template <bool CALL_FREE_SCALE_MATH = false>
__device__ __forceinline__ float compute_localcta_encode_scaling_factor_FP4(const float local_amax) {
    float local_encode_scale = localcta_scale_divide<CALL_FREE_SCALE_MATH>(
        localcta_global_scale_num(), local_amax);
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

template <bool SYNC_AFTER = true>
__device__ __forceinline__ void apply_inverse_rope_tile_inplace_live64(
    IType* __restrict__ sIn_ptr,
    const RopeLive64Desc& rope,
    const int buff_in,
    const int stage_Y,
    const int stage_X,
    const int input_block_offset_Y,
    const int input_block_offset_X
) {
    auto& sIn2x = *reinterpret_cast<IType2x3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % THREADS_X_ROWWISE;
    const int pair_thread_offset_X = tid_X * (ELTS_PER_THREAD / 2);
    const int tile_row_offset = input_block_offset_Y + stage_Y * TILE_DIM_Y;
    const int tile_pair_offset = (input_block_offset_X + stage_X * TILE_DIM_X) >> 1;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;
        const int seq_idx = (tile_row_offset + row) & rope.seq_mask;
        const int rope_row_offset = seq_idx * 32;

        #pragma unroll
        for (int p = 0; p < ELTS_PER_THREAD / 2; ++p) {
            const int pair_col = pair_thread_offset_X + p;
            const float2 cs = rope.cs[rope_row_offset + ((tile_pair_offset + pair_col) & 31)];
            auto* packed_ptr = reinterpret_cast<__nv_bfloat162*>(&sIn2x[buff_in][row][pair_col]);
            const float2 packed = __bfloat1622float2(*packed_ptr);
            *packed_ptr = __float22bfloat162_rn(make_float2(
                packed.x * cs.x + packed.y * cs.y,
                packed.y * cs.x - packed.x * cs.y));
        }
    }
    if constexpr (SYNC_AFTER) {
        __syncthreads();
    }
}

template <int GROUP_THREADS, uint32_t BARRIER_ID = 1u, int THREAD_OFFSET = 0>
__device__ __forceinline__ void subgroup_barrier_sync() {
    if (threadIdx.x >= THREAD_OFFSET && threadIdx.x < THREAD_OFFSET + GROUP_THREADS) {
        ptx::numbered_barrier_sync(GROUP_THREADS, BARRIER_ID);
    }
}

template <int GROUP_THREADS, uint32_t BARRIER_ID = 1u, int THREAD_OFFSET = 0>
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

        subgroup_barrier_sync<GROUP_THREADS, BARRIER_ID, THREAD_OFFSET>();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFrowwise_ptr)[dest] = my_scales[k];
            }
        }

        subgroup_barrier_sync<GROUP_THREADS, BARRIER_ID, THREAD_OFFSET>();
    }
}

template <int GROUP_THREADS, uint32_t BARRIER_ID = 1u, int THREAD_OFFSET = 0>
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

        subgroup_barrier_sync<GROUP_THREADS, BARRIER_ID, THREAD_OFFSET>();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFcolwise_ptr)[dest] = my_scales[k];
            }
        }

        subgroup_barrier_sync<GROUP_THREADS, BARRIER_ID, THREAD_OFFSET>();
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

__device__ __forceinline__ void store_scale_row_direct_swizzled(
    nvfp4_scale_t* sSFrowwise_ptr,
    int row,
    int scale_x,
    nvfp4_scale_t value
) {
    const int j = row % 32;
    const int grp = row / 32;
    const int koffset = scale_x / 4;
    const int k_byte = scale_x % 4;
    const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
    reinterpret_cast<uint8_t*>(sSFrowwise_ptr)[dest] = reinterpret_cast<const uint8_t&>(value);
}

__device__ __forceinline__ void store_scale_col_direct_swizzled(
    nvfp4_scale_t* sSFcolwise_ptr,
    int col,
    int scale_y,
    nvfp4_scale_t value
) {
    const int j = col % 32;
    const int grp = col / 32;
    const int koffset = scale_y / 4;
    const int k_byte = scale_y % 4;
    const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
    reinterpret_cast<uint8_t*>(sSFcolwise_ptr)[dest] = reinterpret_cast<const uint8_t&>(value);
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

__device__ __forceinline__ void scale_local_scales_inplace(
    nvfp4_scale_t* scales_ptr,
    int num_elements,
    float scale
) {
    for (int idx = threadIdx.x; idx < num_elements; idx += THREADS) {
        const float raw_v = static_cast<float>(scales_ptr[idx]);
        scales_ptr[idx] = static_cast<nvfp4_scale_t>(raw_v * scale);
    }
}

__device__ __forceinline__ float get_amax_of_pair(const IType2 pair) {
    const float ax = __bfloat162float(__habs(pair.x));
    const float ay = __bfloat162float(__habs(pair.y));
    return fmaxf(ax, ay);
}

using LocalCTARNGState = transformer_engine::curanddx::detail::philox4x32_native_state<10>;

__device__ __forceinline__ uint32_t localcta_next_rbits(
    LocalCTARNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if (rnd_idx == 4) {
        rnd_idx = 0;
        random_uint4 = rng.generate4();
    }
    const uint32_t* rbits_arr = reinterpret_cast<uint32_t*>(&random_uint4);
    return rbits_arr[rnd_idx++];
}

__device__ __forceinline__ uint32_t localcta_fast_sr_rbits(
    uint64_t base,
    uint64_t counter
) {
    uint32_t x = static_cast<uint32_t>(base) ^ static_cast<uint32_t>(base >> 32);
    x ^= static_cast<uint32_t>(counter) * 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ uint32_t localcta_gf16_xtime(uint32_t value) {
    const uint32_t reduction_mask = 0u - (value >> 15);
    return ((value << 1) & 0xffffu) ^ (reduction_mask & 0x100bu);
}

__device__ __forceinline__ uint32_t localcta_pack_correlated_sr_pair(
    uint32_t ef,
    uint32_t ab
) {
    return ((ab & 0xff00u) << 16) |
           ((ef & 0xff00u) << 8) |
           ((ab & 0x00ffu) << 8) |
           (ef & 0x00ffu);
}

template <bool ROW_VECTOR>
__device__ __forceinline__ uint4 localcta_correlated_sr_rbits16(
    uint64_t base,
    int row,
    int col
) {
    const uint32_t tile_row = static_cast<uint32_t>(row) >> 4;
    const uint32_t tile_col = static_cast<uint32_t>(col) >> 4;
    uint32_t tile_rbits =
        static_cast<uint32_t>(base) ^ static_cast<uint32_t>(base >> 32);
    tile_rbits ^= tile_row * 0x9e3779b9u;
    tile_rbits ^= tile_col * 0x85ebca6bu;
    tile_rbits ^= tile_rbits >> 16;
    tile_rbits *= 0x7feb352du;
    tile_rbits ^= tile_rbits >> 15;
    tile_rbits *= 0x846ca68bu;
    tile_rbits ^= tile_rbits >> 16;

    // Sixty-four pairwise-independent GF(2^16) affine fields from one
    // uniformly mixed (a, b) pair. A row or column vector consumes eight
    // fields from the same logical 16x16 tile.
    const uint32_t field_a = tile_rbits & 0xffffu;
    const uint32_t field_b = tile_rbits >> 16;
    const uint32_t field_a2 = localcta_gf16_xtime(field_a);
    const uint32_t field_a4 = localcta_gf16_xtime(field_a2);
    const uint32_t field_a8 = localcta_gf16_xtime(field_a4);
    const uint32_t field_a16 = localcta_gf16_xtime(field_a8);
    const uint32_t field_a32 = localcta_gf16_xtime(field_a16);
    uint32_t field_product = 0;
    if constexpr (ROW_VECTOR) {
        const uint32_t row_group =
            (static_cast<uint32_t>(row) & 14u) >> 1;
        field_product ^= field_a8 & (0u - (row_group & 1u));
        field_product ^= field_a16 & (0u - ((row_group >> 1) & 1u));
        field_product ^= field_a32 & (0u - ((row_group >> 2) & 1u));
    } else {
        const uint32_t col_group =
            (static_cast<uint32_t>(col) & 14u) >> 1;
        field_product ^= field_a & (0u - (col_group & 1u));
        field_product ^= field_a2 & (0u - ((col_group >> 1) & 1u));
        field_product ^= field_a4 & (0u - ((col_group >> 2) & 1u));
    }

    const uint32_t field0 = field_b ^ field_product;
    const uint32_t field_step = ROW_VECTOR ? field_a : field_a8;
    const uint32_t field_step2 = ROW_VECTOR ? field_a2 : field_a16;
    const uint32_t field_step4 = ROW_VECTOR ? field_a4 : field_a32;
    const uint32_t field1 = field0 ^ field_step;
    const uint32_t field2 = field0 ^ field_step2;
    const uint32_t field3 = field0 ^ field_step ^ field_step2;
    const uint32_t field4 = field0 ^ field_step4;
    const uint32_t field5 = field4 ^ field_step;
    const uint32_t field6 = field4 ^ field_step2;
    const uint32_t field7 = field4 ^ field_step ^ field_step2;
    return make_uint4(
        localcta_pack_correlated_sr_pair(field0, field1),
        localcta_pack_correlated_sr_pair(field2, field3),
        localcta_pack_correlated_sr_pair(field4, field5),
        localcta_pack_correlated_sr_pair(field6, field7));
}

__device__ __forceinline__ uint64_t localcta_swap_bf16_pair_lanes(
    uint64_t packed
) {
    return ((packed & 0xffff0000ffff0000ull) >> 16) |
           ((packed & 0x0000ffff0000ffffull) << 16);
}

__device__ __forceinline__ uint32_t localcta_swap_fp4_pair_lanes(
    uint32_t packed
) {
    return ((packed & 0xf0f0f0f0u) >> 4) |
           ((packed & 0x0f0f0f0fu) << 4);
}

template <bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ uint32_t localcta_make_rht_sign_bits(
    LocalCTARNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if constexpr (!WITH_RANDOM_SIGN_MASK) {
        return 0xffffffffu;
    }
    return 0x00002817u;
}

__device__ __forceinline__ void localcta_fwht16_unnormalized(float (&vals)[ELTS_PER_THREAD]) {
    #pragma unroll
    for (int step = 1; step < ELTS_PER_THREAD; step <<= 1) {
        #pragma unroll
        for (int base = 0; base < ELTS_PER_THREAD; base += 2 * step) {
            #pragma unroll
            for (int j = 0; j < step; ++j) {
                const float a = vals[base + j];
                const float b = vals[base + j + step];
                vals[base + j] = a + b;
                vals[base + j + step] = a - b;
            }
        }
    }
}

template <bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ void localcta_apply_rht16_registers(
    float (&vals)[ELTS_PER_THREAD],
    const uint32_t sign_bits
) {
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; ++i) {
        if constexpr (WITH_RANDOM_SIGN_MASK) {
            vals[i] *= ((sign_bits >> i) & 1u) ? 1.0f : -1.0f;
        }
    }
    localcta_fwht16_unnormalized(vals);
    constexpr float kNorm16 = 0.25f;
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; ++i) {
        vals[i] *= kNorm16;
    }
}

__device__ __forceinline__ nvfp4_scale_t localcta_float_to_e4m3_sr(float val, uint32_t rbits) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t packed = 0;
    asm volatile(
        "{\n\t"
        "cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %1, %1, %1}, %2;\n\t"
        "}\n\t"
        : "=r"(packed)
        : "f"(val), "r"(rbits));
    nvfp4_scale_t out;
    *reinterpret_cast<unsigned char*>(&out) = static_cast<unsigned char>(packed & 0xffu);
    return out;
#else
    NVTE_DEVICE_ERROR("E4M3 stochastic conversion requires SM 10.0+.");
    return static_cast<nvfp4_scale_t>(val);
#endif
}

template <bool CALL_FREE_SCALE_MATH = false>
__device__ __forceinline__ nvfp4_scale_t localcta_compute_encoding_scale(
    const float block_amax,
    const float S_enc
) {
    constexpr float fp4_max = transformer_engine::detail::TypeExtrema<fp4e2m1>::max;
    constexpr float fp8_max = transformer_engine::detail::TypeExtrema<fp8e4m3>::max;
    constexpr float float_max = transformer_engine::detail::TypeExtrema<float>::max;
    // The CTA outer scale normalizes tiny absolute gradients. A fixed epsilon
    // here therefore destroys valid relative signal; only an exact-zero block
    // should use the all-zero fallback multiplier.
    if (block_amax == 0.0f) {
        return static_cast<nvfp4_scale_t>(fp8_max);
    }
    const float denominator = block_amax * S_enc;
    const float multiplier = localcta_scale_divide<CALL_FREE_SCALE_MATH>(
        fp4_max, denominator);
    return static_cast<nvfp4_scale_t>(fminf(multiplier, float_max));
}

template <
    bool ENCODE_CENTRIC,
    bool SCALE_SR,
    bool CALL_FREE_SCALE_MATH = false>
__device__ __forceinline__ void localcta_compute_scale_and_coeff(
    const float block_amax,
    const float S_enc,
    LocalCTARNGState& rng,
    uint4& random_uint4,
    int& rnd_idx,
    float& coeff,
    nvfp4_scale_t& S_b_fp8
) {
    using namespace quantization_and_transposition_SF;
    if constexpr (ENCODE_CENTRIC) {
        nvfp4_scale_t S_mult_fp8;
        if constexpr (SCALE_SR) {
            constexpr float fp4_max = transformer_engine::detail::TypeExtrema<fp4e2m1>::max;
            const float M = block_amax == 0.0f
                ? transformer_engine::detail::TypeExtrema<fp8e4m3>::max
                : fminf(localcta_scale_divide<CALL_FREE_SCALE_MATH>(
                            fp4_max, block_amax * S_enc),
                        transformer_engine::detail::TypeExtrema<float>::max);
            S_mult_fp8 = localcta_float_to_e4m3_sr(
                M, localcta_next_rbits(rng, random_uint4, rnd_idx));
        } else {
            S_mult_fp8 = localcta_compute_encoding_scale<
                CALL_FREE_SCALE_MATH>(block_amax, S_enc);
        }
        coeff = static_cast<float>(S_mult_fp8) * S_enc;
        const float S_mult = static_cast<float>(S_mult_fp8);
        const float S_b = localcta_scale_reciprocal<CALL_FREE_SCALE_MATH>(
            S_mult);
        S_b_fp8 = SCALE_SR
            ? localcta_float_to_e4m3_sr(S_b, localcta_next_rbits(rng, random_uint4, rnd_idx))
            : static_cast<nvfp4_scale_t>(S_b);
    } else {
        if constexpr (SCALE_SR) {
            constexpr float fp4_max = transformer_engine::detail::TypeExtrema<fp4e2m1>::max;
            const float S_dec_b = localcta_scale_divide<
                CALL_FREE_SCALE_MATH>(block_amax, fp4_max) * S_enc;
            S_b_fp8 = localcta_float_to_e4m3_sr(
                fminf(S_dec_b, transformer_engine::detail::TypeExtrema<float>::max),
                localcta_next_rbits(rng, random_uint4, rnd_idx));
        } else {
            if constexpr (CALL_FREE_SCALE_MATH) {
                constexpr float fp4_max =
                    transformer_engine::detail::TypeExtrema<fp4e2m1>::max;
                const float S_dec_b = localcta_scale_divide<true>(
                    block_amax, fp4_max) * S_enc;
                S_b_fp8 = static_cast<nvfp4_scale_t>(fminf(
                    S_dec_b,
                    transformer_engine::detail::TypeExtrema<float>::max));
            } else {
                S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            }
        }
        constexpr float float_max = 3.4028235e+38f;
        const float S_dec = localcta_scale_reciprocal<
            CALL_FREE_SCALE_MATH>(S_enc);
        const float coeff_denominator = static_cast<float>(S_b_fp8) * S_dec;
        coeff = fminf(
            localcta_scale_reciprocal<CALL_FREE_SCALE_MATH>(
                coeff_denominator),
            float_max);
    }
}

__device__ __forceinline__ void localcta_compute_four_over_six_candidates(
    const float block_amax,
    const float S_enc,
    float& coeff_6,
    float& inv_coeff_6,
    nvfp4_scale_t& scale_6,
    float& coeff_4,
    float& inv_coeff_4,
    nvfp4_scale_t& scale_4
) {
    constexpr float rcp_6 = 1.0f / 6.0f;
    constexpr float float_max = 3.4028235e+38f;
    const float scale_6_hp = block_amax * rcp_6 * S_enc;
    scale_6 = static_cast<nvfp4_scale_t>(fminf(scale_6_hp, float_max));
    scale_4 = static_cast<nvfp4_scale_t>(fminf(1.5f * scale_6_hp, float_max));

    const float scale_6_f = static_cast<float>(scale_6);
    const float scale_4_f = static_cast<float>(scale_4);
    if (scale_6_f > 0.0f) {
        coeff_6 = fminf(S_enc / scale_6_f, float_max);
        inv_coeff_6 = scale_6_f / S_enc;
    } else {
        coeff_6 = 1.0f;
        inv_coeff_6 = 0.0f;
    }
    if (scale_4_f > 0.0f) {
        coeff_4 = fminf(S_enc / scale_4_f, float_max);
        inv_coeff_4 = scale_4_f / S_enc;
    } else {
        coeff_4 = 1.0f;
        inv_coeff_4 = 0.0f;
    }
}

__device__ __forceinline__ uint32_t localcta_quantize_e2m1_8x_rn_mae(
    const IType* __restrict__ values,
    const float coeff,
    const float inv_coeff,
    float& mae
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const float x0 = __bfloat162float(values[0]);
    const float x1 = __bfloat162float(values[1]);
    const float x2 = __bfloat162float(values[2]);
    const float x3 = __bfloat162float(values[3]);
    const float x4 = __bfloat162float(values[4]);
    const float x5 = __bfloat162float(values[5]);
    const float x6 = __bfloat162float(values[6]);
    const float x7 = __bfloat162float(values[7]);
    const float s0 = x0 * coeff;
    const float s1 = x1 * coeff;
    const float s2 = x2 * coeff;
    const float s3 = x3 * coeff;
    const float s4 = x4 * coeff;
    const float s5 = x5 * coeff;
    const float s6 = x6 * coeff;
    const float s7 = x7 * coeff;

    uint32_t packed;
    uint32_t dequant_01;
    uint32_t dequant_23;
    uint32_t dequant_45;
    uint32_t dequant_67;
    asm volatile(
        "{\n\t"
        ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %6, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %8, %7;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %10, %9;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %12, %11;\n\t"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, byte2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %4, byte3;\n\t"
        "}\n\t"
        : "=r"(packed), "=r"(dequant_01), "=r"(dequant_23),
          "=r"(dequant_45), "=r"(dequant_67)
        : "f"(s0), "f"(s1), "f"(s2), "f"(s3),
          "f"(s4), "f"(s5), "f"(s6), "f"(s7));

    const float2 q01 = __half22float2(*reinterpret_cast<const __half2*>(&dequant_01));
    const float2 q23 = __half22float2(*reinterpret_cast<const __half2*>(&dequant_23));
    const float2 q45 = __half22float2(*reinterpret_cast<const __half2*>(&dequant_45));
    const float2 q67 = __half22float2(*reinterpret_cast<const __half2*>(&dequant_67));
    mae += fabsf(q01.x * inv_coeff - x0) + fabsf(q01.y * inv_coeff - x1);
    mae += fabsf(q23.x * inv_coeff - x2) + fabsf(q23.y * inv_coeff - x3);
    mae += fabsf(q45.x * inv_coeff - x4) + fabsf(q45.y * inv_coeff - x5);
    mae += fabsf(q67.x * inv_coeff - x6) + fabsf(q67.y * inv_coeff - x7);
    return packed;
#else
    NVTE_DEVICE_ERROR("Four-over-six quantization requires SM 10.0+.");
    return 0;
#endif
}

template <
    bool ENCODE_CENTRIC = true,
    bool DIRECT_SWIZZLED_SCALES = false,
    bool FOLD_STORED_SCALE = false>
__device__ __forceinline__ void rowwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out,
    const float stored_scale_multiplier = 1.0f
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
                localcta_compute_encoding_scale(block_amax, S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }

        if (SF_storing) {
            nvfp4_scale_t stored_scale = S_b_fp8;
            if constexpr (FOLD_STORED_SCALE) {
                stored_scale = static_cast<nvfp4_scale_t>(
                    static_cast<float>(S_b_fp8) * stored_scale_multiplier);
            }
            const int scale_row = stage_sc_Y + it * THREADS_Y_ROWWISE;
            if constexpr (DIRECT_SWIZZLED_SCALES) {
                store_scale_row_direct_swizzled(sSFrowwise_ptr, scale_row, stage_sc_X, stored_scale);
            } else {
                sSFrowwise[scale_row][stage_sc_X] = stored_scale;
            }
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

template <
    bool ENCODE_CENTRIC = true,
    bool DIRECT_SWIZZLED_SCALES = false,
    bool FOLD_STORED_SCALE = false>
__device__ __forceinline__ void colwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out_tr,
    const float stored_scale_multiplier = 1.0f
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
                localcta_compute_encoding_scale(bmax[w], S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(bmax[w], S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }
        nvfp4_scale_t stored_scale = S_b_fp8;
        if constexpr (FOLD_STORED_SCALE) {
            stored_scale = static_cast<nvfp4_scale_t>(
                static_cast<float>(S_b_fp8) * stored_scale_multiplier);
        }
        if constexpr (DIRECT_SWIZZLED_SCALES) {
            store_scale_col_direct_swizzled(sSFcolwise_ptr, sc_tr_Y + w, sc_tr_X, stored_scale);
        } else {
            sSFcolwise[sc_tr_Y + w][sc_tr_X] = stored_scale;
        }

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

template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void weight_2d_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out, const int buff_out_tr
) {
    using namespace quantization_and_transposition_SF;

    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

    __shared__ float warp_block_amax[THREADS / THREADS_PER_WARP]
                                          [SCALES_PER_TILE_X];
    __shared__ float block_coeff[SCALES_PER_TILE_Y * SCALES_PER_TILE_X];
    __shared__ nvfp4_scale_t
        block_scale[SCALES_PER_TILE_Y * SCALES_PER_TILE_X];

    const int lane = threadIdx.x % THREADS_PER_WARP;
    const int warp = threadIdx.x / THREADS_PER_WARP;
    const int bank_group = lane / THREADS_PER_BANK;
    const int tid_y = threadIdx.x / THREADS_X_ROWWISE;
    const int tid_x = threadIdx.x % THREADS_X_ROWWISE;
    const int thread_offset_x = tid_x * ELTS_PER_THREAD;

    // Each pass covers 32 rows. Four threads per row retain the tuned
    // bank-friendly load pattern while the four warps reduce one 16x16 amax
    // for every pair of warps and x segment.
    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_y + it * THREADS_Y_ROWWISE;
        __align__(16) IType2 values[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {
            __float2bfloat16(0.0f),
            __float2bfloat16(0.0f),
        };
        #pragma unroll
        for (int wave = 0; wave < WAVES; ++wave) {
            const int sw =
                ((wave + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            const __uint128_t packed = ptx::ld_shared_b128(
                &sIn[buff_in][row][thread_offset_x + sw]);
            *reinterpret_cast<__uint128_t*>(&values[wave]) = packed;
            #pragma unroll
            for (int i = 0; i < PACK_SIZE / 2; ++i) {
                ptx::abs_max_2x(amax_2x, amax_2x, values[wave][i]);
            }
        }

        float local_amax = get_amax_of_pair(amax_2x);
        local_amax = fmaxf(
            local_amax, __shfl_xor_sync(0xffffffff, local_amax, 4));
        local_amax = fmaxf(
            local_amax, __shfl_xor_sync(0xffffffff, local_amax, 8));
        local_amax = fmaxf(
            local_amax, __shfl_xor_sync(0xffffffff, local_amax, 16));
        if (lane < SCALES_PER_TILE_X) {
            warp_block_amax[warp][lane] = local_amax;
        }
        __syncthreads();

        if (threadIdx.x < 2 * SCALES_PER_TILE_X) {
            const int local_block_y = threadIdx.x / SCALES_PER_TILE_X;
            const int block_x = threadIdx.x % SCALES_PER_TILE_X;
            const int source_warp = local_block_y * 2;
            const float block_amax = fmaxf(
                warp_block_amax[source_warp][block_x],
                warp_block_amax[source_warp + 1][block_x]);
            const int block_y = it * 2 + local_block_y;
            const int block = block_y * SCALES_PER_TILE_X + block_x;

            float coeff;
            nvfp4_scale_t stored_scale;
            if constexpr (ENCODE_CENTRIC) {
                const nvfp4_scale_t scale_mult =
                    localcta_compute_encoding_scale(block_amax, S_enc);
                coeff = static_cast<float>(scale_mult) * S_enc;
                stored_scale = static_cast<nvfp4_scale_t>(
                    1.0f / static_cast<float>(scale_mult));
            } else {
                stored_scale =
                    compute_decoding_scaling_factor(block_amax, S_enc);
                constexpr float float_max = 3.4028235e+38f;
                const float S_dec = 1.0f / S_enc;
                coeff = fminf(
                    1.0f / (static_cast<float>(stored_scale) * S_dec),
                    float_max);
            }
            block_coeff[block] = coeff;
            block_scale[block] = stored_scale;
        }
        __syncthreads();

        const int block_y = row / SCALE_DIM;
        const int block = block_y * SCALES_PER_TILE_X + tid_x;
        const float coeff = block_coeff[block];
        #pragma unroll
        for (int wave = 0; wave < WAVES; ++wave) {
            const int sw =
                ((wave + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            const uint64_t e03 =
                *reinterpret_cast<const uint64_t*>(&values[wave][0]);
            const uint64_t e47 =
                *reinterpret_cast<const uint64_t*>(&values[wave][2]);
            const uint32_t packed =
                ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(
                    e03, e47, coeff);
            ptx::st_shared_b32(
                &sOut[buff_out][row][(thread_offset_x + sw) / 2], packed);
        }

        const int row_scale_row = stage_Y * TILE_DIM_Y + row;
        const int row_scale_col = stage_X * SCALES_PER_TILE_X + tid_x;
        sSFrowwise[row_scale_row][row_scale_col] = block_scale[block];
        __syncthreads();
    }

    // One thread owns a source column pair in each 16x16 block. It gathers
    // the two transposed output rows and emits each as a single 64-bit store.
    const int block = threadIdx.x / (SCALE_DIM / 2);
    const int pair_col = threadIdx.x % (SCALE_DIM / 2);
    const int block_y = block / SCALES_PER_TILE_X;
    const int block_x = block % SCALES_PER_TILE_X;
    const int source_byte_col = block_x * (SCALE_DIM / 2) + pair_col;
    uint64_t low_nibbles = 0;
    uint64_t high_nibbles = 0;
    #pragma unroll
    for (int row_pair = 0; row_pair < SCALE_DIM / 2; ++row_pair) {
        const int source_row = block_y * SCALE_DIM + row_pair * 2;
        const uint8_t first = reinterpret_cast<const uint8_t*>(
            &sOut[buff_out][source_row][source_byte_col])[0];
        const uint8_t second = reinterpret_cast<const uint8_t*>(
            &sOut[buff_out][source_row + 1][source_byte_col])[0];
        const uint8_t low = (first & 0x0f) | ((second & 0x0f) << 4);
        const uint8_t high = (first >> 4) | (second & 0xf0);
        low_nibbles |= static_cast<uint64_t>(low) << (row_pair * 8);
        high_nibbles |= static_cast<uint64_t>(high) << (row_pair * 8);
    }
    const int output_row = block_x * SCALE_DIM + pair_col * 2;
    const int output_byte_col = block_y * (SCALE_DIM / 2);
    ptx::st_shared_b64(
        &sOut_tr[buff_out_tr][output_row][output_byte_col], low_nibbles);
    ptx::st_shared_b64(
        &sOut_tr[buff_out_tr][output_row + 1][output_byte_col], high_nibbles);

    const nvfp4_scale_t stored_scale = block_scale[block];
    const int col_scale_row = stage_X * TILE_DIM_X + output_row;
    const int col_scale_col = stage_Y * SCALES_PER_TILE_Y + block_y;
    sSFcolwise[col_scale_row][col_scale_col] = stored_scale;
    sSFcolwise[col_scale_row + 1][col_scale_col] = stored_scale;
}

template <
    bool ENCODE_CENTRIC = true,
    bool DIRECT_SWIZZLED_SCALES = false,
    bool DATA_SR = false,
    bool FAST_DATA_SR = false,
    bool SCALE_SR = false,
    bool WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false,
    bool FOUR_OVER_SIX_MAE = false,
    bool CORRELATED_DATA_SR = false>
__device__ __forceinline__ void rowwise_scaling_opt(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out,
    LocalCTARNGState& rng,
    uint4& random_uint4,
    int& rnd_idx,
    uint64_t fast_sr_base,
    int logical_row_offset = 0,
    int logical_col_offset = 0
) {
    static_assert(
        !FOUR_OVER_SIX_MAE ||
            (!DATA_SR && !SCALE_SR && !WITH_RHT && !WITH_RANDOM_SIGN_MASK),
        "Four-over-six MAE selection currently requires deterministic non-RHT quantization");
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

        uint4 correlated_rbits = make_uint4(0, 0, 0, 0);
        if constexpr (DATA_SR && FAST_DATA_SR && CORRELATED_DATA_SR) {
            const int logical_row =
                logical_row_offset + stage_Y * TILE_DIM_Y + row;
            const int logical_col =
                logical_col_offset + stage_X * TILE_DIM_X + thread_offset_X;
            correlated_rbits = localcta_correlated_sr_rbits16<true>(
                fast_sr_base, logical_row, logical_col);
        }

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ptx::ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            if constexpr (!WITH_RHT) {
                #pragma unroll
                for (int e = 0; e < PACK_SIZE / 2; ++e) {
                    ptx::abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
                }
            }
        }

        float block_amax;
        if constexpr (WITH_RHT) {
            float vals[ELTS_PER_THREAD];
            #pragma unroll
            for (int w = 0; w < WAVES; ++w) {
                const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
                #pragma unroll
                for (int e = 0; e < PACK_SIZE / 2; ++e) {
                    const float2 packed =
                        __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&rIn[w][e]));
                    vals[sw + 2 * e + 0] = packed.x;
                    vals[sw + 2 * e + 1] = packed.y;
                }
            }
            const uint32_t sign_bits =
                localcta_make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
            localcta_apply_rht16_registers<WITH_RANDOM_SIGN_MASK>(vals, sign_bits);

            block_amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELTS_PER_THREAD; ++i) {
                block_amax = fmaxf(block_amax, fabsf(vals[i]));
            }
            #pragma unroll
            for (int w = 0; w < WAVES; ++w) {
                const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
                #pragma unroll
                for (int e = 0; e < PACK_SIZE / 2; ++e) {
                    rIn[w][e] = IType2{
                        __float2bfloat16_rn(vals[sw + 2 * e + 0]),
                        __float2bfloat16_rn(vals[sw + 2 * e + 1]),
                    };
                }
            }
        } else {
            block_amax = get_amax_of_pair(amax_2x);
        }

        float coeff;
        nvfp4_scale_t S_b_fp8;
        uint32_t four_over_six_out[WAVES];
        if constexpr (FOUR_OVER_SIX_MAE) {
            float coeff_6;
            float inv_coeff_6;
            nvfp4_scale_t scale_6;
            float coeff_4;
            float inv_coeff_4;
            nvfp4_scale_t scale_4;
            localcta_compute_four_over_six_candidates(
                block_amax, S_enc,
                coeff_6, inv_coeff_6, scale_6,
                coeff_4, inv_coeff_4, scale_4);

            float mae_6 = 0.0f;
            float mae_4 = 0.0f;
            uint32_t out_6[WAVES];
            uint32_t out_4[WAVES];
            #pragma unroll
            for (int w = 0; w < WAVES; ++w) {
                const IType* values = reinterpret_cast<const IType*>(&rIn[w][0]);
                out_6[w] = localcta_quantize_e2m1_8x_rn_mae(
                    values, coeff_6, inv_coeff_6, mae_6);
                out_4[w] = localcta_quantize_e2m1_8x_rn_mae(
                    values, coeff_4, inv_coeff_4, mae_4);
            }
            const bool select_4 = mae_4 < mae_6;
            coeff = select_4 ? coeff_4 : coeff_6;
            S_b_fp8 = select_4 ? scale_4 : scale_6;
            #pragma unroll
            for (int w = 0; w < WAVES; ++w) {
                four_over_six_out[w] = select_4 ? out_4[w] : out_6[w];
            }
        } else {
            localcta_compute_scale_and_coeff<ENCODE_CENTRIC, SCALE_SR>(
                block_amax, S_enc, rng, random_uint4, rnd_idx, coeff, S_b_fp8);
        }

        if (SF_storing) {
            const int scale_row = stage_sc_Y + it * THREADS_Y_ROWWISE;
            if constexpr (DIRECT_SWIZZLED_SCALES) {
                store_scale_row_direct_swizzled(sSFrowwise_ptr, scale_row, stage_sc_X, S_b_fp8);
            } else {
                sSFrowwise[scale_row][stage_sc_X] = S_b_fp8;
            }
        }

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out;
            if constexpr (FOUR_OVER_SIX_MAE) {
                out = four_over_six_out[w];
            } else if constexpr (DATA_SR) {
                uint32_t rbits03;
                uint32_t rbits47;
                if constexpr (FAST_DATA_SR) {
                    if constexpr (CORRELATED_DATA_SR) {
                        const int logical_row =
                            logical_row_offset + stage_Y * TILE_DIM_Y + row;
                        // Canonicalize each 2x2 tile by checkerboard parity so
                        // row and column layouts assign an element to the same
                        // member of the hardware-shared operand pair.
                        if ((logical_row & 1ll) != 0) {
                            e03 = localcta_swap_bf16_pair_lanes(e03);
                            e47 = localcta_swap_bf16_pair_lanes(e47);
                        }
                        const bool high_half = (sw & 8) != 0;
                        rbits03 = high_half ? correlated_rbits.z : correlated_rbits.x;
                        rbits47 = high_half ? correlated_rbits.w : correlated_rbits.y;
                    } else {
                        const uint64_t ctr =
                            (static_cast<uint64_t>(it) * WAVES + static_cast<uint64_t>(w)) * 2ull;
                        rbits03 = localcta_fast_sr_rbits(fast_sr_base, ctr);
                        rbits47 = localcta_fast_sr_rbits(fast_sr_base, ctr + 1ull);
                    }
                } else {
                    rbits03 = localcta_next_rbits(rng, random_uint4, rnd_idx);
                    rbits47 = localcta_next_rbits(rng, random_uint4, rnd_idx);
                }
                out = ptx::mul_cvt_bf16_to_fp4_8x_stochastic_rounding<float>(
                    e03, e47, coeff, rbits03, rbits47);
                if constexpr (FAST_DATA_SR && CORRELATED_DATA_SR) {
                    const int logical_row =
                        logical_row_offset + stage_Y * TILE_DIM_Y + row;
                    if ((logical_row & 1) != 0) {
                        out = localcta_swap_fp4_pair_lanes(out);
                    }
                }
            } else {
                out = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            }
            ptx::st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

template <
    bool ENCODE_CENTRIC = true,
    bool DIRECT_SWIZZLED_SCALES = false,
    bool DATA_SR = false,
    bool FAST_DATA_SR = false,
    bool SCALE_SR = false,
    bool WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false,
    bool FOUR_OVER_SIX_MAE = false,
    bool CORRELATED_DATA_SR = false,
    bool CALL_FREE_SCALE_MATH = false>
__device__ __forceinline__ void colwise_scaling_opt(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out_tr,
    LocalCTARNGState& rng,
    uint4& random_uint4,
    int& rnd_idx,
    uint64_t fast_sr_base,
    int logical_row_offset = 0,
    int logical_col_offset = 0,
    int logical_tid = -1
) {
    static_assert(
        !FOUR_OVER_SIX_MAE ||
            (!SCALE_SR && !WITH_RHT && !WITH_RANDOM_SIGN_MASK),
        "Four-over-six MAE selection requires deterministic scales and non-RHT quantization");
    const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

    const int quant_tid = logical_tid >= 0 ? logical_tid : static_cast<int>(threadIdx.x);
    const int warp = quant_tid / THREADS_PER_WARP;
    const int lane = quant_tid % THREADS_PER_WARP;

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
        if constexpr (!WITH_RHT) {
            ptx::abs_max_2x(amax_2x, amax_2x, pair);
        }
    }

    float bmax[2];
    if constexpr (WITH_RHT) {
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            float vals[SCALE_DIM];
            #pragma unroll
            for (int i = 0; i < SCALE_DIM; ++i) {
                vals[i] = __bfloat162float(rIn[w][i]);
            }
            const uint32_t sign_bits =
                localcta_make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
            localcta_apply_rht16_registers<WITH_RANDOM_SIGN_MASK>(vals, sign_bits);
            bmax[w] = 0.0f;
            #pragma unroll
            for (int i = 0; i < SCALE_DIM; ++i) {
                bmax[w] = fmaxf(bmax[w], fabsf(vals[i]));
                rIn[w][i] = __float2bfloat16_rn(vals[i]);
            }
        }
    } else {
        bmax[0] = __bfloat162float(__habs(amax_2x.x));
        bmax[1] = __bfloat162float(__habs(amax_2x.y));
    }

    // Both values in each column pair use the same affine field. Expand it
    // once rather than repeating the GF(2^16) arithmetic for w=0 and w=1.
    uint4 correlated_rbits = make_uint4(0, 0, 0, 0);
    if constexpr (DATA_SR && FAST_DATA_SR && CORRELATED_DATA_SR) {
        const int logical_row =
            logical_row_offset + stage_Y * TILE_DIM_Y + off_Y;
        const int logical_col =
            logical_col_offset + stage_X * TILE_DIM_X + off_X;
        correlated_rbits = localcta_correlated_sr_rbits16<false>(
            fast_sr_base, logical_row, logical_col);
    }

    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        float coeff;
        nvfp4_scale_t S_b_fp8;
        uint32_t four_over_six_out[SCALE_DIM / 8];
        if constexpr (FOUR_OVER_SIX_MAE) {
            float coeff_6;
            float inv_coeff_6;
            nvfp4_scale_t scale_6;
            float coeff_4;
            float inv_coeff_4;
            nvfp4_scale_t scale_4;
            localcta_compute_four_over_six_candidates(
                bmax[w], S_enc,
                coeff_6, inv_coeff_6, scale_6,
                coeff_4, inv_coeff_4, scale_4);

            float mae_6 = 0.0f;
            float mae_4 = 0.0f;
            uint32_t out_6[SCALE_DIM / 8];
            uint32_t out_4[SCALE_DIM / 8];
            #pragma unroll
            for (int e = 0; e < SCALE_DIM / 8; ++e) {
                const IType* values = &rIn[w][8 * e];
                out_6[e] = localcta_quantize_e2m1_8x_rn_mae(
                    values, coeff_6, inv_coeff_6, mae_6);
                out_4[e] = localcta_quantize_e2m1_8x_rn_mae(
                    values, coeff_4, inv_coeff_4, mae_4);
            }
            const bool select_4 = mae_4 < mae_6;
            coeff = select_4 ? coeff_4 : coeff_6;
            S_b_fp8 = select_4 ? scale_4 : scale_6;
            #pragma unroll
            for (int e = 0; e < SCALE_DIM / 8; ++e) {
                four_over_six_out[e] = select_4 ? out_4[e] : out_6[e];
            }
        } else {
            localcta_compute_scale_and_coeff<
                ENCODE_CENTRIC, SCALE_SR, CALL_FREE_SCALE_MATH>(
                bmax[w], S_enc, rng, random_uint4, rnd_idx, coeff, S_b_fp8);
        }

        if constexpr (DIRECT_SWIZZLED_SCALES) {
            store_scale_col_direct_swizzled(sSFcolwise_ptr, sc_tr_Y + w, sc_tr_X, S_b_fp8);
        } else {
            sSFcolwise[sc_tr_Y + w][sc_tr_X] = S_b_fp8;
        }

        __align__(8) uint32_t rOut[SCALE_DIM / 8];
        #pragma unroll
        for (int e = 0; e < SCALE_DIM / 8; ++e) {
            uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e]);
            uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e + 4]);
            if constexpr (FOUR_OVER_SIX_MAE && !DATA_SR) {
                rOut[e] = four_over_six_out[e];
            } else if constexpr (DATA_SR) {
                uint32_t rbits03;
                uint32_t rbits47;
                if constexpr (FAST_DATA_SR) {
                    if constexpr (CORRELATED_DATA_SR) {
                        const int logical_col =
                            logical_col_offset + stage_X * TILE_DIM_X + off_X + w;
                        // The column view uses the same checkerboard ordering
                        // as the row view after swapping odd logical columns.
                        if ((logical_col & 1ll) != 0) {
                            e03 = localcta_swap_bf16_pair_lanes(e03);
                            e47 = localcta_swap_bf16_pair_lanes(e47);
                        }
                        const bool high_half = e != 0;
                        rbits03 = high_half ? correlated_rbits.z : correlated_rbits.x;
                        rbits47 = high_half ? correlated_rbits.w : correlated_rbits.y;
                    } else {
                        const uint64_t ctr =
                            (static_cast<uint64_t>(w) * (SCALE_DIM / 8) + static_cast<uint64_t>(e)) * 2ull;
                        rbits03 = localcta_fast_sr_rbits(fast_sr_base, ctr);
                        rbits47 = localcta_fast_sr_rbits(fast_sr_base, ctr + 1ull);
                    }
                } else {
                    rbits03 = localcta_next_rbits(rng, random_uint4, rnd_idx);
                    rbits47 = localcta_next_rbits(rng, random_uint4, rnd_idx);
                }
                rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_stochastic_rounding<float>(
                    e03, e47, coeff, rbits03, rbits47);
                if constexpr (FAST_DATA_SR && CORRELATED_DATA_SR) {
                    const int logical_col =
                        logical_col_offset + stage_X * TILE_DIM_X + off_X + w;
                    if ((logical_col & 1) != 0) {
                        rOut[e] = localcta_swap_fp4_pair_lanes(rOut[e]);
                    }
                }
            } else {
                rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            }
        }
        ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_Y + w][out_tr_X],
                           *reinterpret_cast<uint64_t*>(rOut));
    }
}

template <
    int GROUP_THREADS,
    bool ENCODE_CENTRIC = true,
    bool CALL_FREE_SCALE_MATH = false>
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
                localcta_compute_encoding_scale<CALL_FREE_SCALE_MATH>(
                    block_amax, S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            const float S_mult = static_cast<float>(S_mult_fp8);
            const float S_b = localcta_scale_reciprocal<
                CALL_FREE_SCALE_MATH>(S_mult);
            S_b_fp8 = static_cast<nvfp4_scale_t>(S_b);
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = localcta_scale_reciprocal<
                CALL_FREE_SCALE_MATH>(S_enc);
            const float coeff_denominator =
                static_cast<float>(S_b_fp8) * S_dec;
            coeff = fminf(
                localcta_scale_reciprocal<CALL_FREE_SCALE_MATH>(
                    coeff_denominator),
                float_max);
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
                localcta_compute_encoding_scale(bmax[w], S_enc);
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

template <int GROUP_THREADS>
__device__ __forceinline__ float transform_sqrelu_tile_inplace_amax_group(
    IType* __restrict__ sIn_ptr,
    int buff_in,
    int tid
) {
    static_assert(GROUP_THREADS == 128, "transform_sqrelu_tile_inplace_amax_group expects 128 consumer threads");
    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    constexpr int VEC_ELEMS = 4;
    constexpr int TILE_VECS = BUFF_IN_ELEMS / VEC_ELEMS;
    float tile_max = 0.0f;

    for (int vec = tid; vec < TILE_VECS; vec += GROUP_THREADS) {
        const int elem = vec * VEC_ELEMS;
        const int row = elem / BUFF_DIM_X;
        const int col = elem % BUFF_DIM_X;
        const int2 packed = *reinterpret_cast<const int2*>(&sIn[buff_in][row][col]);

        const __nv_bfloat162 in0 = *reinterpret_cast<const __nv_bfloat162*>(&packed.x);
        const __nv_bfloat162 in1 = *reinterpret_cast<const __nv_bfloat162*>(&packed.y);
        const float2 f0 = __bfloat1622float2(in0);
        const float2 f1 = __bfloat1622float2(in1);

        const __nv_bfloat162 out0 = __float22bfloat162_rn(
            make_float2(f0.x > 0.0f ? f0.x * f0.x : 0.0f,
                        f0.y > 0.0f ? f0.y * f0.y : 0.0f));
        const __nv_bfloat162 out1 = __float22bfloat162_rn(
            make_float2(f1.x > 0.0f ? f1.x * f1.x : 0.0f,
                        f1.y > 0.0f ? f1.y * f1.y : 0.0f));

        const float2 of0 = __bfloat1622float2(out0);
        const float2 of1 = __bfloat1622float2(out1);
        tile_max = fmaxf(tile_max, fabsf(of0.x));
        tile_max = fmaxf(tile_max, fabsf(of0.y));
        tile_max = fmaxf(tile_max, fabsf(of1.x));
        tile_max = fmaxf(tile_max, fabsf(of1.y));

        int2 out;
        out.x = *reinterpret_cast<const int*>(&out0);
        out.y = *reinterpret_cast<const int*>(&out1);
        *reinterpret_cast<int2*>(&sIn[buff_in][row][col]) = out;
    }
    return tile_max;
}

template <int GROUP_THREADS>
__device__ __forceinline__ float transform_sqrelu_deriv_tile_inplace_amax_group(
    IType* __restrict__ sDh_ptr,
    const IType* __restrict__ sH1_ptr,
    int buff_in,
    int tid
) {
    static_assert(GROUP_THREADS == 128, "transform_sqrelu_deriv_tile_inplace_amax_group expects 128 consumer threads");
    auto& sDh = *reinterpret_cast<IType3D*>(sDh_ptr);
    const auto& sH1 = *reinterpret_cast<const IType3D*>(sH1_ptr);
    constexpr int VEC_ELEMS = 4;
    constexpr int TILE_VECS = BUFF_IN_ELEMS / VEC_ELEMS;
    float tile_max = 0.0f;

    for (int vec = tid; vec < TILE_VECS; vec += GROUP_THREADS) {
        const int elem = vec * VEC_ELEMS;
        const int row = elem / BUFF_DIM_X;
        const int col = elem % BUFF_DIM_X;
        const int2 dh = *reinterpret_cast<const int2*>(&sDh[buff_in][row][col]);
        const int2 h1 = *reinterpret_cast<const int2*>(&sH1[buff_in][row][col]);

        const __nv_bfloat162 dh0 = *reinterpret_cast<const __nv_bfloat162*>(&dh.x);
        const __nv_bfloat162 dh1 = *reinterpret_cast<const __nv_bfloat162*>(&dh.y);
        const __nv_bfloat162 h10 = *reinterpret_cast<const __nv_bfloat162*>(&h1.x);
        const __nv_bfloat162 h11 = *reinterpret_cast<const __nv_bfloat162*>(&h1.y);
        const float2 d0 = __bfloat1622float2(dh0);
        const float2 d1 = __bfloat1622float2(dh1);
        const float2 x0 = __bfloat1622float2(h10);
        const float2 x1 = __bfloat1622float2(h11);

        const __nv_bfloat162 out0 = __float22bfloat162_rn(
            make_float2(x0.x > 0.0f ? (2.0f * d0.x) * x0.x : 0.0f,
                        x0.y > 0.0f ? (2.0f * d0.y) * x0.y : 0.0f));
        const __nv_bfloat162 out1 = __float22bfloat162_rn(
            make_float2(x1.x > 0.0f ? (2.0f * d1.x) * x1.x : 0.0f,
                        x1.y > 0.0f ? (2.0f * d1.y) * x1.y : 0.0f));

        const float2 of0 = __bfloat1622float2(out0);
        const float2 of1 = __bfloat1622float2(out1);
        tile_max = fmaxf(tile_max, fabsf(of0.x));
        tile_max = fmaxf(tile_max, fabsf(of0.y));
        tile_max = fmaxf(tile_max, fabsf(of1.x));
        tile_max = fmaxf(tile_max, fabsf(of1.y));

        int2 out;
        out.x = *reinterpret_cast<const int*>(&out0);
        out.y = *reinterpret_cast<const int*>(&out1);
        *reinterpret_cast<int2*>(&sDh[buff_in][row][col]) = out;
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

template <bool WITH_SILU = false>
__device__ __forceinline__ void apply_rmsnorm_to_shared_tile_opt(
    IType* __restrict__ sIn_ptr,
    int buff_in,
    int block_offset_Y,
    int block_offset_X,
    int stage_Y,
    int stage_X,
    const IType* __restrict__ gamma,
    const float* __restrict__ inv_rms
) {
    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    const int row_base = block_offset_Y + stage_Y * TILE_DIM_Y;
    const int col_base = block_offset_X + stage_X * TILE_DIM_X;

    for (int idx = threadIdx.x; idx < BUFF_IN_ELEMS; idx += THREADS) {
        const int row = idx / BUFF_DIM_X;
        const int col = idx % BUFF_DIM_X;
        const float x = __bfloat162float(sIn[buff_in][row][col]);
        const float g = __bfloat162float(gamma[col_base + col]);
        const float out = rmsnorm_contract_value<WITH_SILU>(x, inv_rms[row_base + row], g);
        sIn[buff_in][row][col] = __float2bfloat16_rn(out);
    }
}

__device__ __forceinline__ void apply_rmsnorm_to_shared_tile_direct(
    IType* __restrict__ sIn_ptr,
    int buff_in,
    int block_offset_Y,
    int block_offset_X,
    int stage_Y,
    int stage_X,
    const IType* __restrict__ gamma,
    const float* __restrict__ inv_rms
) {
    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    const int row_base = block_offset_Y + stage_Y * TILE_DIM_Y;
    const int col_base = block_offset_X + stage_X * TILE_DIM_X;
    constexpr int PACK_ELEMS = 8;
    constexpr int PACKS_PER_TILE = BUFF_IN_ELEMS / PACK_ELEMS;
    static_assert(BUFF_DIM_X % PACK_ELEMS == 0);
    auto* tile = &sIn[buff_in][0][0];
    for (int pack = threadIdx.x; pack < PACKS_PER_TILE; pack += THREADS) {
        const int idx = pack * PACK_ELEMS;
        const int row = idx / BUFF_DIM_X;
        const int col = idx % BUFF_DIM_X;
        const float inv = inv_rms[row_base + row];
        const uint4 packed_x = *reinterpret_cast<const uint4*>(tile + idx);
        const uint4 packed_gamma =
            *reinterpret_cast<const uint4*>(gamma + col_base + col);
        uint4 packed_out;
        const auto* x_pairs =
            reinterpret_cast<const __nv_bfloat162*>(&packed_x);
        const auto* gamma_pairs =
            reinterpret_cast<const __nv_bfloat162*>(&packed_gamma);
        auto* out_pairs = reinterpret_cast<__nv_bfloat162*>(&packed_out);
        #pragma unroll
        for (int p = 0; p < PACK_ELEMS / 2; ++p) {
            const float2 x2 = __bfloat1622float2(x_pairs[p]);
            const float2 g2 = __bfloat1622float2(gamma_pairs[p]);
            out_pairs[p] = __float22bfloat162_rn(make_float2(
                rmsnorm_contract_value<false>(x2.x, inv, g2.x),
                rmsnorm_contract_value<false>(x2.y, inv, g2.y)));
        }
        *reinterpret_cast<uint4*>(tile + idx) = packed_out;
    }
}

template <bool WITH_RHT = false, bool WITH_RANDOM_SIGN_MASK = false>
__device__ __forceinline__ float scan_tile_amax_row_opt(
    const IType* __restrict__ sIn_ptr,
    int buff_in,
    LocalCTARNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if constexpr (!WITH_RHT) {
        return scan_tile_amax(sIn_ptr, buff_in);
    }

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
        float vals[ELTS_PER_THREAD];
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t elts = ptx::ld_shared_b128(&sIn[buff_in][row][off_X + sw]);
            const IType2* pairs = reinterpret_cast<const IType2*>(&elts);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                const float2 packed =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pairs[e]));
                vals[sw + 2 * e + 0] = packed.x;
                vals[sw + 2 * e + 1] = packed.y;
            }
        }
        const uint32_t sign_bits =
            localcta_make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
        localcta_apply_rht16_registers<WITH_RANDOM_SIGN_MASK>(vals, sign_bits);
        #pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; ++i) {
            tile_max = fmaxf(tile_max, fabsf(vals[i]));
        }
    }
    return tile_max;
}

template <bool WITH_RHT = false, bool WITH_RANDOM_SIGN_MASK = false>
__device__ __forceinline__ float scan_tile_amax_col_opt(
    const IType* __restrict__ sIn_ptr,
    int buff_in,
    LocalCTARNGState& rng,
    uint4& random_uint4,
    int& rnd_idx,
    int logical_tid = -1
) {
    if constexpr (!WITH_RHT) {
        return scan_tile_amax(sIn_ptr, buff_in);
    }

    const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
    const int quant_tid = logical_tid >= 0 ? logical_tid : static_cast<int>(threadIdx.x);
    const int warp = quant_tid / THREADS_PER_WARP;
    const int lane = quant_tid % THREADS_PER_WARP;
    const int tid_Y = (lane % 4 + warp) % 4;
    const int tid_X = lane;
    const int in_Y = tid_Y * SCALE_DIM;
    const int in_X = tid_X;
    float tile_max = 0.0f;

    __align__(8) IType rIn[2][SCALE_DIM];
    #pragma unroll
    for (int i = 0; i < SCALE_DIM; ++i) {
        const IType2 pair = ptx::ld_shared_b32(&sIn2x[buff_in][in_Y + i][in_X]);
        rIn[0][i] = pair.x;
        rIn[1][i] = pair.y;
    }

    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        float vals[SCALE_DIM];
        #pragma unroll
        for (int i = 0; i < SCALE_DIM; ++i) {
            vals[i] = __bfloat162float(rIn[w][i]);
        }
        const uint32_t sign_bits =
            localcta_make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
        localcta_apply_rht16_registers<WITH_RANDOM_SIGN_MASK>(vals, sign_bits);
        #pragma unroll
        for (int i = 0; i < SCALE_DIM; ++i) {
            tile_max = fmaxf(tile_max, fabsf(vals[i]));
        }
    }
    return tile_max;
}

__device__ __forceinline__ float localcta_ex2_approx_ftz(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

// The x86 SM100a ABI reserves a large caller frame when the fused W2 producer
// contains the CUDA math-library calls emitted by __expf and precise division.
// This implementation is leaf-only while preserving the production TE BF16
// carrier.  Three BF16 inputs sit at the exp overflow/subnormal boundary; use
// the exact FP32 carriers produced by the precise CUDA expression for them.
__device__ __forceinline__ float localcta_silu_callfree_te(
    float x,
    bool fast_divide
) {
    const uint32_t x_bits = __float_as_uint(x);
    if ((x_bits & 0x7fffffffu) == 0u || x_bits == 0x7f800000u) {
        return x;
    }
    if (x_bits == 0xc2af0000u) {  // BF16 -87.5
        return __uint_as_float(0x83949c56u);
    }
    if (x_bits == 0xc2b00000u) {  // BF16 -88.0
        return __uint_as_float(0x83354df1u);
    }
    if (x_bits == 0xc2b10000u) {  // BF16 -88.5
        return __uint_as_float(0x82dd2eb1u);
    }

    constexpr float LOG2_E = 1.4426950408889634074f;
    const float exp_neg_x = localcta_ex2_approx_ftz(-x * LOG2_E);
    const float denom = 1.0f + exp_neg_x;
    const uint32_t denom_abs_bits = __float_as_uint(denom) & 0x7fffffffu;
    if (denom_abs_bits == 0x7f800000u) {
        return x * 0.0f;
    }
    if (denom_abs_bits > 0x7f800000u) {
        return x * denom;
    }

    float reciprocal = __fdividef(1.0f, denom);
    const float reciprocal_error = fmaf(-denom, reciprocal, 1.0f);
    reciprocal = fmaf(reciprocal_error, reciprocal, reciprocal);
    if (fast_divide) {
        return x * reciprocal;
    }

    float quotient = __fdividef(x, denom);
    const float quotient_error = fmaf(-denom, quotient, x);
    return fmaf(quotient_error, reciprocal, quotient);
}

template <bool CALL_FREE_TE_MATH>
__device__ __forceinline__ float localcta_silu_value(
    float x,
    bool fast_divide
) {
    if constexpr (CALL_FREE_TE_MATH) {
        return localcta_silu_callfree_te(x, fast_divide);
    }
    const float denom = 1.0f + __expf(-x);
    return fast_divide ? x * (1.0f / denom) : x / denom;
}

template <int TOTAL_THREADS, bool CALL_FREE_TE_MATH = false>
__device__ __forceinline__ float transform_silu_vectors_inplace_amax_linear(
    int2* __restrict__ h1_vec,
    const int2* __restrict__ h3_vec,
    bool fast_divide
) {
    constexpr int VEC_ELEMS = 4;
    static_assert(BUFF_IN_ELEMS % VEC_ELEMS == 0);
    static_assert(sizeof(__nv_bfloat162) == sizeof(int));
    static_assert(sizeof(int2) == 2 * sizeof(int));
    constexpr int TILE_VECS = BUFF_IN_ELEMS / VEC_ELEMS;
    float tile_max = 0.0f;

    for (int idx = threadIdx.x; idx < TILE_VECS; idx += TOTAL_THREADS) {
        const int2 h1 = h1_vec[idx];
        const int2 h3 = h3_vec[idx];
        const __nv_bfloat162 h1_0 =
            *reinterpret_cast<const __nv_bfloat162*>(&h1.x);
        const __nv_bfloat162 h1_1 =
            *reinterpret_cast<const __nv_bfloat162*>(&h1.y);
        const __nv_bfloat162 h3_0 =
            *reinterpret_cast<const __nv_bfloat162*>(&h3.x);
        const __nv_bfloat162 h3_1 =
            *reinterpret_cast<const __nv_bfloat162*>(&h3.y);
        const float2 h1_0f = __bfloat1622float2(h1_0);
        const float2 h1_1f = __bfloat1622float2(h1_1);
        const float2 h3_0f = __bfloat1622float2(h3_0);
        const float2 h3_1f = __bfloat1622float2(h3_1);
        const float silu0x = localcta_silu_value<CALL_FREE_TE_MATH>(
            h1_0f.x, fast_divide);
        const float silu0y = localcta_silu_value<CALL_FREE_TE_MATH>(
            h1_0f.y, fast_divide);
        const float silu1x = localcta_silu_value<CALL_FREE_TE_MATH>(
            h1_1f.x, fast_divide);
        const float silu1y = localcta_silu_value<CALL_FREE_TE_MATH>(
            h1_1f.y, fast_divide);
        const __nv_bfloat162 out0 = __float22bfloat162_rn(
            make_float2(silu0x * h3_0f.x, silu0y * h3_0f.y));
        const __nv_bfloat162 out1 = __float22bfloat162_rn(
            make_float2(silu1x * h3_1f.x, silu1y * h3_1f.y));
        const float2 out0f = __bfloat1622float2(out0);
        const float2 out1f = __bfloat1622float2(out1);
        tile_max = fmaxf(tile_max, fabsf(out0f.x));
        tile_max = fmaxf(tile_max, fabsf(out0f.y));
        tile_max = fmaxf(tile_max, fabsf(out1f.x));
        tile_max = fmaxf(tile_max, fabsf(out1f.y));

        int2 packed;
        packed.x = *reinterpret_cast<const int*>(&out0);
        packed.y = *reinterpret_cast<const int*>(&out1);

        h1_vec[idx] = packed;
    }
    return tile_max;
}

template <int TOTAL_THREADS, bool CALL_FREE_TE_MATH = false>
__device__ __forceinline__ float transform_silu_tile_inplace_amax_linear(
    IType* __restrict__ sH1_ptr,
    const IType* __restrict__ sH3_ptr,
    int buff_in,
    bool fast_divide
) {
    auto& sH1 = *reinterpret_cast<IType3D*>(sH1_ptr);
    const auto& sH3 = *reinterpret_cast<const IType3D*>(sH3_ptr);
    int2* h1_vec = reinterpret_cast<int2*>(&sH1[buff_in][0][0]);
    const int2* h3_vec = reinterpret_cast<const int2*>(&sH3[buff_in][0][0]);
    return transform_silu_vectors_inplace_amax_linear<TOTAL_THREADS, CALL_FREE_TE_MATH>(
        h1_vec, h3_vec, fast_divide);
}

template <int TOTAL_THREADS, bool CALL_FREE_TE_MATH = false>
__device__ __forceinline__ float transform_silu_tile_inplace_amax_linear_h3_slot(
    IType* __restrict__ sH1_ptr,
    const IType* __restrict__ sH3_tile_ptr,
    int buff_in,
    bool fast_divide
) {
    using Tile2D = IType[BUFF_DIM_Y][BUFF_DIM_X];
    auto& sH1 = *reinterpret_cast<IType3D*>(sH1_ptr);
    const auto& sH3 = *reinterpret_cast<const Tile2D*>(sH3_tile_ptr);
    int2* h1_vec = reinterpret_cast<int2*>(&sH1[buff_in][0][0]);
    const int2* h3_vec = reinterpret_cast<const int2*>(&sH3[0][0]);
    return transform_silu_vectors_inplace_amax_linear<TOTAL_THREADS, CALL_FREE_TE_MATH>(
        h1_vec, h3_vec, fast_divide);
}

__device__ __forceinline__ void atomic_max_positive_float(
    float* __restrict__ addr,
    float value
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // SG values are non-negative, so IEEE float ordering matches unsigned int ordering.
    atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(value));
#else
    NVTE_DEVICE_ERROR("atomic_max_positive_float requires SM 10.0+.");
#endif
}

template <
    bool RETURN_TRANSPOSE,
    bool ENCODE_CENTRIC = true,
    bool PREFINALIZED_OUTER_SG = false,
    bool WITH_RMSNORM = false,
    bool SHARED_2D_WEIGHT = false>
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
    const IType* __restrict__ rms_gamma,
    const float* __restrict__ rms_inv_rms,
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
            if constexpr (WITH_RMSNORM) {
                const int stage_Y = t / TILES_X;
                const int stage_X = t % TILES_X;
                apply_rmsnorm_to_shared_tile_direct(
                    sIn_ptr, t, block_offset_Y, block_offset_X,
                    stage_Y, stage_X, rms_gamma, rms_inv_rms);
                __syncthreads();
            }
            if constexpr (!PREFINALIZED_OUTER_SG) {
                cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
            }
        }

        if constexpr (!PREFINALIZED_OUTER_SG) {
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
        }

        float S_enc_row;
        float S_enc_col;
        float sg_val;
        if constexpr (PREFINALIZED_OUTER_SG) {
            const float row_sg_val = row_sg_chunks[ctaid_Y / 2];
            float col_sg_val = row_sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                col_sg_val = col_sg_chunks[ctaid_X / 2];
            }
            S_enc_row = compute_localcta_encode_scaling_factor_FP4(
                row_sg_val * localcta_global_scale_num());
            S_enc_col = compute_localcta_encode_scaling_factor_FP4(
                col_sg_val * localcta_global_scale_num());
            sg_val = row_sg_val;
        } else {
            const float amax_val = cta_amax_shared;
            const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
            sg_val = amax_val / localcta_global_scale_num();
            S_enc_row = S_enc;
            S_enc_col = S_enc;
        }

        if constexpr (!PREFINALIZED_OUTER_SG) {
        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
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

            if (t >= BUFFS_NUM_OUT) {
                if (leading) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }
                __syncthreads();
            }

            if constexpr (SHARED_2D_WEIGHT) {
                static_assert(RETURN_TRANSPOSE,
                              "2D weight quantization requires a transpose payload");
                static_assert(!PREFINALIZED_OUTER_SG,
                              "2D weight quantization requires one chunk-local outer scale");
                weight_2d_scaling<ENCODE_CENTRIC>(
                    sIn_ptr, sOut_ptr, sOut_tr_ptr,
                    sSFrowwise_ptr, sSFcolwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out, buff_out_tr);
            } else {
                rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                                                S_enc_row, stage_Y, stage_X, t, buff_out);

                if constexpr (RETURN_TRANSPOSE) {
                    colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                                    S_enc_col, stage_Y, stage_X, t, buff_out_tr);
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
                    SHARED_2D_WEIGHT ? 1.0f : sg_val);
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
                    SHARED_2D_WEIGHT ? 1.0f : sg_val);
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

template <
    bool RETURN_TRANSPOSE,
    bool ENCODE_CENTRIC = true,
    bool ROW_DATA_SR = false,
    bool COL_DATA_SR = ROW_DATA_SR,
    bool SCALE_SR = false,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false,
    bool WITH_RMSNORM = false,
    bool WITH_RMSNORM_SILU = false,
    bool PREFINALIZED_OUTER_SG = false,
    bool FAST_DATA_SR = false,
    bool ATOMIC_FINAL_OUTER_SG = false,
    bool FOUR_OVER_SIX_MAE = false,
    bool EMIT_ROW = true>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_kernel_opt(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    float* __restrict__ row_sg_final,
    float* __restrict__ col_sg_final,
    const IType* __restrict__ rms_gamma,
    const float* __restrict__ rms_inv_rms,
    const size_t rows, const size_t cols,
    LocalCTAPersistentArgs args,
    bool write_raw_scales,
    bool write_prepared,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const unsigned long long* __restrict__ rng_state
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    if constexpr (ROW_DATA_SR || COL_DATA_SR || SCALE_SR) {
        if (rng_state != nullptr) {
            rng_seed = static_cast<uint64_t>(rng_state[0]);
            rng_subsequence_base = static_cast<uint64_t>(rng_state[1]);
        }
    }
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
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(
        dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(
            dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(
            dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

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
    __shared__ float warp_max_row[THREADS / 32];
    __shared__ float warp_max_col[THREADS / 32];
    __shared__ float cta_amax_row_shared;
    __shared__ float cta_amax_col_shared;
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

        float row_cta_max = 0.0f;
        float col_cta_max = 0.0f;

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
            if constexpr (WITH_RMSNORM) {
                const int stage_Y = t / TILES_X;
                const int stage_X = t % TILES_X;
                apply_rmsnorm_to_shared_tile_opt<WITH_RMSNORM_SILU>(
                    sIn_ptr, t, block_offset_Y, block_offset_X,
                    stage_Y, stage_X, rms_gamma, rms_inv_rms);
                __syncthreads();
            }
            if constexpr (PREFINALIZED_OUTER_SG) {
                // Input tiles are still staged for the quantization pass below;
                // the outer SGs were produced by an upstream transform/scan.
            } else if constexpr (ROW_WITH_RHT) {
                LocalCTARNGState row_rng;
                if constexpr (WITH_RANDOM_SIGN_MASK) {
                    row_rng.init(
                        rng_seed,
                        rng_subsequence_base +
                            ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 0ull) * THREADS +
                            threadIdx.x,
                        0);
                }
                uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                int row_rnd_idx = 4;
                row_cta_max = fmaxf(
                    row_cta_max,
                    scan_tile_amax_row_opt<true, WITH_RANDOM_SIGN_MASK>(
                        sIn_ptr, t, row_rng, row_random_uint4, row_rnd_idx));
            } else {
                row_cta_max = fmaxf(row_cta_max, scan_tile_amax(sIn_ptr, t));
            }
            if constexpr (!PREFINALIZED_OUTER_SG && RETURN_TRANSPOSE) {
                if constexpr (COL_WITH_RHT) {
                    LocalCTARNGState col_rng;
                    if constexpr (WITH_RANDOM_SIGN_MASK) {
                        col_rng.init(
                            rng_seed,
                            rng_subsequence_base +
                                ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 1ull) * THREADS +
                                threadIdx.x,
                            0);
                    }
                    uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                    int col_rnd_idx = 4;
                    col_cta_max = fmaxf(
                        col_cta_max,
                        scan_tile_amax_col_opt<true, WITH_RANDOM_SIGN_MASK>(
                            sIn_ptr, t, col_rng, col_random_uint4, col_rnd_idx));
                } else {
                    col_cta_max = fmaxf(col_cta_max, scan_tile_amax(sIn_ptr, t));
                }
            }
        }

        if constexpr (!PREFINALIZED_OUTER_SG) {
            if constexpr (!RETURN_TRANSPOSE) {
                col_cta_max = row_cta_max;
            }

            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                row_cta_max = fmaxf(row_cta_max, __shfl_xor_sync(0xffffffff, row_cta_max, mask));
                col_cta_max = fmaxf(col_cta_max, __shfl_xor_sync(0xffffffff, col_cta_max, mask));
            }

            if (lane == 0) {
                warp_max_row[wid] = row_cta_max;
                warp_max_col[wid] = col_cta_max;
            }
            __syncthreads();

            if (wid == 0) {
                row_cta_max = (lane < THREADS / 32) ? warp_max_row[lane] : 0.0f;
                col_cta_max = (lane < THREADS / 32) ? warp_max_col[lane] : 0.0f;
                #pragma unroll
                for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    row_cta_max = fmaxf(row_cta_max, __shfl_xor_sync(0xffffffff, row_cta_max, mask));
                    col_cta_max = fmaxf(col_cta_max, __shfl_xor_sync(0xffffffff, col_cta_max, mask));
                }
                if (lane == 0) {
                    cta_amax_row_shared = row_cta_max;
                    cta_amax_col_shared = col_cta_max;
                }
            }
            __syncthreads();
        }

        float row_sg_val;
        float col_sg_val;
        float S_enc_row;
        float S_enc_col;
        if constexpr (PREFINALIZED_OUTER_SG) {
            if constexpr (EMIT_ROW) {
                row_sg_val = row_sg_chunks[ctaid_Y / 2];
            } else {
                row_sg_val = 1.0f;
            }
            if constexpr (RETURN_TRANSPOSE) {
                col_sg_val = col_sg_chunks[ctaid_X / 2];
            } else {
                col_sg_val = row_sg_val;
            }
            if constexpr (EMIT_ROW) {
                S_enc_row = compute_localcta_encode_scaling_factor_FP4(
                    row_sg_val * localcta_global_scale_num());
            } else {
                S_enc_row = 1.0f;
            }
            S_enc_col = compute_localcta_encode_scaling_factor_FP4(
                col_sg_val * localcta_global_scale_num());
        } else {
            const float row_amax_val = cta_amax_row_shared;
            const float col_amax_val = cta_amax_col_shared;
            S_enc_row = compute_localcta_encode_scaling_factor_FP4(row_amax_val);
            S_enc_col = compute_localcta_encode_scaling_factor_FP4(col_amax_val);
            row_sg_val = row_amax_val / localcta_global_scale_num();
            col_sg_val = col_amax_val / localcta_global_scale_num();
        }

        if constexpr (!PREFINALIZED_OUTER_SG) {
            if (leading) {
                row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = row_sg_val;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = col_sg_val;
                }
                if constexpr (ATOMIC_FINAL_OUTER_SG) {
                    atomic_max_positive_float(row_sg_final + (ctaid_Y / 2), row_sg_val);
                    if constexpr (RETURN_TRANSPOSE) {
                        atomic_max_positive_float(col_sg_final + (ctaid_X / 2), col_sg_val);
                    }
                }
            }
        }

        int buff_out = 0;
        int buff_out_tr = 0;
        const uint64_t correlated_fast_sr_base =
            rng_seed ^ rng_subsequence_base ^ 0xa0761d6478bd642full;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int stage_offset_Y = stage_Y * TILE_DIM_Y;
            const int stage_offset_X = stage_X * TILE_DIM_X;

            if (t >= BUFFS_NUM_OUT) {
                if (leading) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }
                __syncthreads();
            }

            if constexpr (EMIT_ROW) {
                const uint64_t row_rng_subsequence =
                    rng_subsequence_base +
                        ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 0ull) * THREADS +
                        threadIdx.x;
                LocalCTARNGState row_rng;
                if constexpr ((ROW_DATA_SR && !FAST_DATA_SR) || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
                    row_rng.init(rng_seed, row_rng_subsequence, 0);
                }
                uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                int row_rnd_idx = 4;
                constexpr bool correlated_data_sr =
                    ROW_DATA_SR && COL_DATA_SR && FAST_DATA_SR && EMIT_ROW && RETURN_TRANSPOSE &&
                    !ROW_WITH_RHT && !COL_WITH_RHT;
                const uint64_t row_fast_sr_base =
                    correlated_data_sr
                        ? correlated_fast_sr_base
                        : (rng_seed ^ row_rng_subsequence ^ 0xd1342543de82ef95ull);
                rowwise_scaling_opt<
                    ENCODE_CENTRIC, false, ROW_DATA_SR, FAST_DATA_SR, SCALE_SR,
                    ROW_WITH_RHT, WITH_RANDOM_SIGN_MASK, FOUR_OVER_SIX_MAE,
                    correlated_data_sr>(
                    sIn_ptr,
                    sOut_ptr, sSFrowwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out,
                    row_rng, row_random_uint4, row_rnd_idx, row_fast_sr_base,
                    block_offset_Y, block_offset_X);
            }

            if constexpr (RETURN_TRANSPOSE) {
                const uint64_t col_rng_subsequence =
                    rng_subsequence_base +
                        ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 1ull) * THREADS +
                        threadIdx.x;
                LocalCTARNGState col_rng;
                if constexpr ((COL_DATA_SR && !FAST_DATA_SR) || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
                    col_rng.init(rng_seed, col_rng_subsequence, 0);
                }
                uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                int col_rnd_idx = 4;
                constexpr bool correlated_data_sr =
                    ROW_DATA_SR && COL_DATA_SR && FAST_DATA_SR && EMIT_ROW && RETURN_TRANSPOSE &&
                    !ROW_WITH_RHT && !COL_WITH_RHT;
                const uint64_t col_fast_sr_base =
                    correlated_data_sr
                        ? correlated_fast_sr_base
                        : (rng_seed ^ col_rng_subsequence ^ 0x94d049bb133111ebull);
                colwise_scaling_opt<
                    ENCODE_CENTRIC, false, COL_DATA_SR, FAST_DATA_SR,
                    SCALE_SR, COL_WITH_RHT, WITH_RANDOM_SIGN_MASK,
                    FOUR_OVER_SIX_MAE,
                    correlated_data_sr>(
                    sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                    S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                    col_rng, col_random_uint4, col_rnd_idx, col_fast_sr_base,
                    block_offset_Y, block_offset_X);
            }

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                if constexpr (EMIT_ROW) {
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output),
                        block_offset_X + stage_offset_X,
                        block_offset_Y + stage_offset_Y,
                        reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                }

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

        if constexpr (EMIT_ROW) {
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
                    row_sg_val);
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
                    col_sg_val);
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

template <
    bool RETURN_TRANSPOSE,
    bool ENCODE_CENTRIC = true,
    int H3_RING_TILES = 0,
    bool PREFINALIZED_OUTER_SG = false,
    bool ATOMIC_FINAL_OUTER_SG = false,
    bool PARALLEL_ROW_COL = false,
    bool COL_WITH_RHT = false,
    bool COL_WITH_RANDOM_SIGN_MASK = false,
    bool COL_ENCODE_CENTRIC = ENCODE_CENTRIC>
__global__ void __launch_bounds__(SILU_RAW_THREADS)
fused_localcta_silu_quantize_raw_kernel(
    const __grid_constant__ CUtensorMap tensor_map_h1,
    const __grid_constant__ CUtensorMap tensor_map_h3,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    float* __restrict__ row_sg_final,
    float* __restrict__ col_sg_final,
    const size_t rows, const size_t cols,
    bool fast_silu_divide,
    LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(SILU_RAW_THREADS == 2 * THREADS);
    static_assert(THREADS % 32 == 0);
    static_assert(
        !COL_WITH_RHT ||
            (RETURN_TRANSPOSE && COL_WITH_RANDOM_SIGN_MASK &&
             ATOMIC_FINAL_OUTER_SG && !PREFINALIZED_OUTER_SG &&
             ENCODE_CENTRIC && !COL_ENCODE_CENTRIC),
        "paired SiLU column RHT is sealed to the fixed-sign atomic final-SG contract");
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);
    constexpr int h3_mbar_count = H3_RING_TILES > 0 ? H3_RING_TILES : NUM_TILES;

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int h3_bytes = H3_RING_TILES > 0 ?
        DIVUP_TO_MULTIPLE(H3_RING_TILES * BUFF_IN_ELEMS * (int)sizeof(IType),
                          TMA_SHMEM_ALIGNMENT) :
        in_bytes;
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

    IType* sH1_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sH3_ptr = reinterpret_cast<IType*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + h3_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(
        dshmem + in_bytes + h3_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(
        dshmem + in_bytes + h3_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(
        dshmem + in_bytes + h3_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sH1 = *reinterpret_cast<IType3D*>(sH1_ptr);
    auto& sH3 = *reinterpret_cast<IType3D*>(sH3_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t h1_mbar[NUM_TILES];
    __shared__ uint64_t h3_mbar[h3_mbar_count];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_init(&h1_mbar[t], 1);
        }
        #pragma unroll
        for (int t = 0; t < h3_mbar_count; ++t) {
            ptx::mbarrier_init(&h3_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    __shared__ float warp_max[SILU_RAW_THREADS / 32];
    __shared__ float cta_amax_shared;
    __shared__ float cta_col_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;
    int h3_mbar_phase[h3_mbar_count];
    #pragma unroll
    for (int t = 0; t < h3_mbar_count; ++t) {
        h3_mbar_phase[t] = H3_RING_TILES > 0 ? 1 : 0;
    }

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
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &h1_mbar[pre]);
                if constexpr (H3_RING_TILES == 0) {
                    ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH3[pre]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &h3_mbar[pre]);
                }
            }
        }

        if constexpr (H3_RING_TILES > 0) {
            #pragma unroll
            for (int pre = 0; pre < H3_RING_TILES; ++pre) {
                const int ty = pre / TILES_X;
                const int tx = pre % TILES_X;
                h3_mbar_phase[pre] ^= 1;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(
                            sH3_ptr + pre * BUFF_IN_ELEMS),
                        reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &h3_mbar[pre]);
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
                    ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH1[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &h1_mbar[next]);
                    if constexpr (H3_RING_TILES == 0) {
                        ptx::mbarrier_arrive_expect_tx(&h3_mbar[next], shmem_tile_bytes);
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sH3[next]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                            block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &h3_mbar[next]);
                    }
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);
            if constexpr (H3_RING_TILES > 0) {
                const int slot = t % H3_RING_TILES;
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(
                    &h3_mbar[slot], h3_mbar_phase[slot]);
                const float tile_max =
                    transform_silu_tile_inplace_amax_linear_h3_slot<
                        SILU_RAW_THREADS, COL_WITH_RHT>(
                        sH1_ptr, sH3_ptr + slot * BUFF_IN_ELEMS, t,
                        fast_silu_divide);
                if constexpr (!PREFINALIZED_OUTER_SG) {
                    cta_max = fmaxf(cta_max, tile_max);
                }
                __syncthreads();

                const int next_h3 = t + H3_RING_TILES;
                if (next_h3 < NUM_TILES) {
                    const int ty = next_h3 / TILES_X;
                    const int tx = next_h3 % TILES_X;
                    h3_mbar_phase[slot] ^= 1;
                    if (leading) {
                        ptx::mbarrier_arrive_expect_tx(&h3_mbar[slot], shmem_tile_bytes);
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(
                                sH3_ptr + slot * BUFF_IN_ELEMS),
                            reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                            block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &h3_mbar[slot]);
                    }
                }
            } else {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], mbar_phase);
                const float tile_max =
                    transform_silu_tile_inplace_amax_linear<
                        SILU_RAW_THREADS, COL_WITH_RHT>(
                        sH1_ptr, sH3_ptr, t, fast_silu_divide);
                if constexpr (!PREFINALIZED_OUTER_SG) {
                    cta_max = fmaxf(cta_max, tile_max);
                }
            }
        }

        if constexpr (!PREFINALIZED_OUTER_SG) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }

            if (lane == 0) {
                warp_max[wid] = cta_max;
            }
            __syncthreads();

            if (wid == 0) {
                cta_max = (lane < SILU_RAW_THREADS / 32) ? warp_max[lane] : 0.0f;
                #pragma unroll
                for (int mask = (SILU_RAW_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                }
                if (lane == 0) {
                    cta_amax_shared = cta_max;
                }
            }
            __syncthreads();
        }

        if constexpr (COL_WITH_RHT) {
            float cta_col_max = 0.0f;
            if (threadIdx.x < THREADS) {
                LocalCTARNGState col_rng;
                uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                int col_rnd_idx = 4;
                #pragma unroll
                for (int t = 0; t < NUM_TILES; ++t) {
                    cta_col_max = fmaxf(
                        cta_col_max,
                        scan_tile_amax_col_opt<true, COL_WITH_RANDOM_SIGN_MASK>(
                            sH1_ptr, t, col_rng, col_random_uint4, col_rnd_idx,
                            static_cast<int>(threadIdx.x)));
                }

                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    cta_col_max = fmaxf(
                        cta_col_max,
                        __shfl_xor_sync(0xffffffff, cta_col_max, mask));
                }
                if (lane == 0) {
                    warp_max[wid] = cta_col_max;
                }
            }
            __syncthreads();

            if (wid == 0) {
                cta_col_max = lane < THREADS / 32 ? warp_max[lane] : 0.0f;
                #pragma unroll
                for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    cta_col_max = fmaxf(
                        cta_col_max,
                        __shfl_xor_sync(0xffffffff, cta_col_max, mask));
                }
                if (lane == 0) {
                    cta_col_amax_shared = cta_col_max;
                }
            }
            __syncthreads();
        }

        float S_enc_row;
        float S_enc_col;
        float sg_val;
        float col_sg_val;
        if constexpr (PREFINALIZED_OUTER_SG) {
            const float row_sg_val = row_sg_chunks[ctaid_Y / 2];
            col_sg_val = row_sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                col_sg_val = col_sg_chunks[ctaid_X / 2];
            }
            S_enc_row = compute_localcta_encode_scaling_factor_FP4<
                COL_WITH_RHT>(
                row_sg_val * localcta_global_scale_num());
            S_enc_col = compute_localcta_encode_scaling_factor_FP4<
                COL_WITH_RHT>(
                col_sg_val * localcta_global_scale_num());
            sg_val = row_sg_val;
        } else {
            const float amax_val = cta_amax_shared;
            const float col_amax_val = COL_WITH_RHT ? cta_col_amax_shared : amax_val;
            S_enc_row = compute_localcta_encode_scaling_factor_FP4<
                COL_WITH_RHT>(amax_val);
            S_enc_col = compute_localcta_encode_scaling_factor_FP4<
                COL_WITH_RHT>(col_amax_val);
            sg_val = localcta_scale_divide<COL_WITH_RHT>(
                amax_val, localcta_global_scale_num());
            col_sg_val = localcta_scale_divide<COL_WITH_RHT>(
                col_amax_val, localcta_global_scale_num());
        }

        if constexpr (!PREFINALIZED_OUTER_SG) {
            if (leading) {
                row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = col_sg_val;
                }
                if constexpr (ATOMIC_FINAL_OUTER_SG) {
                    atomic_max_positive_float(row_sg_final + (ctaid_Y / 2), sg_val);
                    if constexpr (RETURN_TRANSPOSE) {
                        atomic_max_positive_float(col_sg_final + (ctaid_X / 2), col_sg_val);
                    }
                }
            }
        }

        int buff_out = 0;
        int buff_out_tr = 0;

        // Fully unrolling the parallel row/column path makes CUDA 13.2 hoist
        // every tile's scale addresses across the whole persistent work body.
        // Under the two-CTA launch bound those long-lived addresses spill.
        // Keep the established unrolling for all other specializations, but
        // make the hot parallel producer carry only one tile's addresses.
        constexpr int quant_loop_unroll =
            (PARALLEL_ROW_COL && RETURN_TRANSPOSE) ? 1 : NUM_TILES;
        #pragma unroll quant_loop_unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int stage_offset_Y = stage_Y * TILE_DIM_Y;
            const int stage_offset_X = stage_X * TILE_DIM_X;

            if constexpr (PARALLEL_ROW_COL && RETURN_TRANSPOSE) {
                if (t > 0 && leading) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }
                if (t > 0) {
                    __syncthreads();
                }

                if (threadIdx.x < THREADS) {
                    rowwise_scaling_group<
                        THREADS, ENCODE_CENTRIC, COL_WITH_RHT>(
                        sH1_ptr, sOut_ptr, sSFrowwise_ptr,
                        S_enc_row, stage_Y, stage_X, t, buff_out, threadIdx.x);
                } else {
                    if constexpr (COL_WITH_RHT) {
                        LocalCTARNGState col_rng;
                        uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                        int col_rnd_idx = 4;
                        colwise_scaling_opt<
                            COL_ENCODE_CENTRIC, false, false, false, false,
                            true, COL_WITH_RANDOM_SIGN_MASK, false, false,
                            COL_WITH_RHT>(
                            sH1_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                            S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                            col_rng, col_random_uint4, col_rnd_idx, 0,
                            block_offset_Y, block_offset_X,
                            static_cast<int>(threadIdx.x) - THREADS);
                    } else {
                        colwise_scaling_group<THREADS, ENCODE_CENTRIC>(
                            sH1_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                            S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                            threadIdx.x - THREADS);
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

                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                        block_offset_X_tr + stage_offset_Y,
                        block_offset_Y_tr + stage_offset_X,
                        reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                    ptx::cp_async_bulk_commit_group();
                }
            } else {
                if (threadIdx.x < THREADS) {
                    if (t > 0 && leading) {
                        ptx::cp_async_bulk_wait_group_read<1>();
                    }
                    if (t > 0) {
                        subgroup_barrier_sync<THREADS>();
                    }

                    rowwise_scaling_group<
                        THREADS, ENCODE_CENTRIC, COL_WITH_RHT>(
                        sH1_ptr, sOut_ptr, sSFrowwise_ptr,
                        S_enc_row, stage_Y, stage_X, t, buff_out, threadIdx.x);

                    if constexpr (RETURN_TRANSPOSE) {
                        if constexpr (COL_WITH_RHT) {
                            LocalCTARNGState col_rng;
                            uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                            int col_rnd_idx = 4;
                            colwise_scaling_opt<
                                COL_ENCODE_CENTRIC, false, false, false, false,
                                true, COL_WITH_RANDOM_SIGN_MASK, false, false,
                                COL_WITH_RHT>(
                                sH1_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                                col_rng, col_random_uint4, col_rnd_idx, 0,
                                block_offset_Y, block_offset_X,
                                static_cast<int>(threadIdx.x));
                        } else {
                            colwise_scaling_group<THREADS, ENCODE_CENTRIC>(
                                sH1_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                                threadIdx.x);
                        }
                    }

                    subgroup_barrier_sync<THREADS>();
                    ptx::fence_proxy_async_shared_cta();
                    subgroup_barrier_sync<THREADS>();

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
                }
            }

            __syncthreads();

            buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
            buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
        }

        if constexpr (PARALLEL_ROW_COL && RETURN_TRANSPOSE) {
            const int cnt = min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM);
            const int cnt_tr = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
            if (leading) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();

            if (threadIdx.x < THREADS) {
                swizzle_scales_row_inplace_group<THREADS, 1u, 0>(
                    sSFrowwise_ptr, cnt, threadIdx.x);
            } else {
                swizzle_scales_col_inplace_group<THREADS, 2u, THREADS>(
                    sSFcolwise_ptr, cnt_tr, threadIdx.x - THREADS);
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(
                    tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                tma_store_scales_2x512(
                    tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            }
        } else {
            if (leading) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();

            {
                const int cnt = min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM);
                if (threadIdx.x < THREADS) {
                    swizzle_scales_row_inplace_group<THREADS>(sSFrowwise_ptr, cnt, threadIdx.x);
                    subgroup_barrier_sync<THREADS>();
                    ptx::fence_proxy_async_shared_cta();
                    subgroup_barrier_sync<THREADS>();
                    if (leading) {
                        tma_store_scales_2x512(
                            tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                    }
                }
            }

            if constexpr (RETURN_TRANSPOSE) {
                const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
                if (threadIdx.x < THREADS) {
                    if (leading) {
                        ptx::cp_async_bulk_wait_group_read<0>();
                    }
                    subgroup_barrier_sync<THREADS>();
                    swizzle_scales_col_inplace_group<THREADS>(sSFcolwise_ptr, cnt, threadIdx.x);
                    subgroup_barrier_sync<THREADS>();
                    ptx::fence_proxy_async_shared_cta();
                    subgroup_barrier_sync<THREADS>();
                    if (leading) {
                        tma_store_scales_2x512(
                            tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }
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
            ptx::mbarrier_invalid(&h1_mbar[t]);
        }
        #pragma unroll
        for (int t = 0; t < h3_mbar_count; ++t) {
            ptx::mbarrier_invalid(&h3_mbar[t]);
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

template <
    bool ENCODE_CENTRIC = true,
    bool ROW_DATA_SR = false,
    bool FAST_DATA_SR = false,
    bool COL_WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_split3_raw_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_output0,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output2,
    const __grid_constant__ CUtensorMap tensor_map_output_t0,
    const __grid_constant__ CUtensorMap tensor_map_output_t1,
    const __grid_constant__ CUtensorMap tensor_map_output_t2,
    const __grid_constant__ CUtensorMap tmap_scale_row0,
    const __grid_constant__ CUtensorMap tmap_scale_row1,
    const __grid_constant__ CUtensorMap tmap_scale_row2,
    const __grid_constant__ CUtensorMap tmap_scale_col0,
    const __grid_constant__ CUtensorMap tmap_scale_col1,
    const __grid_constant__ CUtensorMap tmap_scale_col2,
    float* __restrict__ row_sg_chunks0,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ row_sg_chunks2,
    float* __restrict__ col_sg_chunks0,
    float* __restrict__ col_sg_chunks1,
    float* __restrict__ col_sg_chunks2,
    const size_t rows,
    LocalCTAPersistentArgs args,
    int split0_tiles,
    int split1_tiles,
    int split2_tiles,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const unsigned long long* __restrict__ rng_state
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    if constexpr (ROW_DATA_SR || WITH_RANDOM_SIGN_MASK) {
        if (rng_state != nullptr) {
            rng_seed = static_cast<uint64_t>(rng_state[0]);
            rng_subsequence_base = static_cast<uint64_t>(rng_state[1]);
        }
    }
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
    __shared__ float warp_max_col[THREADS / 32];
    __shared__ float cta_amax_shared;
    __shared__ float cta_amax_col_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;
    const int split01_tiles = split0_tiles + split1_tiles;
    const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);

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
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;

        int local_ctaid_X = ctaid_X;
        int split_id = 0;
        int split_tiles = split0_tiles;
        if (ctaid_X >= split01_tiles) {
            split_id = 2;
            split_tiles = split2_tiles;
            local_ctaid_X -= split01_tiles;
        } else if (ctaid_X >= split0_tiles) {
            split_id = 1;
            split_tiles = split1_tiles;
            local_ctaid_X -= split0_tiles;
        }
        const int local_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = block_offset_Y;
        const int block_offset_Y_tr = local_block_offset_X;
        const int chunk_cols = split_tiles * LocalCTAConfig::CHUNK_DIM_X - local_block_offset_X;

        float cta_max = 0.0f;
        float cta_col_max = 0.0f;

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
                        local_block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[pre]);
                } else if (split_id == 1) {
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[pre]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input1),
                        local_block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[pre]);
                } else {
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[pre]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input2),
                        local_block_offset_X + tx * TILE_DIM_X,
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
                            local_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[next]);
                    } else if (split_id == 1) {
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn[next]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input1),
                            local_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[next]);
                    } else {
                        ptx::cp_async_bulk_tensor_2d_global_to_shared(
                            reinterpret_cast<uint64_t*>(&sIn[next]),
                            reinterpret_cast<const uint64_t*>(&tensor_map_input2),
                            local_block_offset_X + tx * TILE_DIM_X,
                            block_offset_Y + ty * TILE_DIM_Y,
                            &in_mbar[next]);
                    }
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
            if constexpr (COL_WITH_RHT) {
                LocalCTARNGState col_scan_rng;
                if constexpr (WITH_RANDOM_SIGN_MASK) {
                    col_scan_rng.init(
                        rng_seed,
                        rng_subsequence_base +
                            ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 1ull) *
                                THREADS +
                            threadIdx.x,
                        0);
                }
                uint4 col_scan_random_uint4 = make_uint4(0, 0, 0, 0);
                int col_scan_rnd_idx = 4;
                cta_col_max = fmaxf(
                    cta_col_max,
                    scan_tile_amax_col_opt<true, WITH_RANDOM_SIGN_MASK>(
                        sIn_ptr,
                        t,
                        col_scan_rng,
                        col_scan_random_uint4,
                        col_scan_rnd_idx));
            }
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            if constexpr (COL_WITH_RHT) {
                cta_col_max = fmaxf(
                    cta_col_max,
                    __shfl_xor_sync(0xffffffff, cta_col_max, mask));
            }
        }
        if (lane == 0) {
            warp_max[wid] = cta_max;
            if constexpr (COL_WITH_RHT) {
                warp_max_col[wid] = cta_col_max;
            }
        }
        __syncthreads();

        if (wid == 0) {
            cta_max = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
            if constexpr (COL_WITH_RHT) {
                cta_col_max = (lane < THREADS / 32) ? warp_max_col[lane] : 0.0f;
            }
            #pragma unroll
            for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                if constexpr (COL_WITH_RHT) {
                    cta_col_max = fmaxf(
                        cta_col_max,
                        __shfl_xor_sync(0xffffffff, cta_col_max, mask));
                }
            }
            if (lane == 0) {
                cta_amax_shared = cta_max;
                if constexpr (COL_WITH_RHT) {
                    cta_amax_col_shared = cta_col_max;
                }
            }
        }
        __syncthreads();

        const float row_amax_val = cta_amax_shared;
        const float col_amax_val = COL_WITH_RHT
            ? cta_amax_col_shared
            : row_amax_val;
        const float S_enc_row = compute_localcta_encode_scaling_factor_FP4(row_amax_val);
        const float S_enc_col = compute_localcta_encode_scaling_factor_FP4(col_amax_val);
        const float row_sg_val = row_amax_val / localcta_global_scale_num();
        const float col_sg_val = col_amax_val / localcta_global_scale_num();

        if (leading) {
            if (split_id == 0) {
                row_sg_chunks0[ctaid_Y * split0_tiles + local_ctaid_X] = row_sg_val;
                col_sg_chunks0[local_ctaid_X * tiles_Y + ctaid_Y] = col_sg_val;
            } else if (split_id == 1) {
                row_sg_chunks1[ctaid_Y * split1_tiles + local_ctaid_X] = row_sg_val;
                col_sg_chunks1[local_ctaid_X * tiles_Y + ctaid_Y] = col_sg_val;
            } else {
                row_sg_chunks2[ctaid_Y * split2_tiles + local_ctaid_X] = row_sg_val;
                col_sg_chunks2[local_ctaid_X * tiles_Y + ctaid_Y] = col_sg_val;
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

            if constexpr (ROW_DATA_SR || COL_WITH_RHT) {
                const uint64_t tile_rng_base =
                    rng_subsequence_base +
                    static_cast<uint64_t>(s_chunk_id) * 2ull * NUM_TILES * THREADS;
                const uint64_t row_rng_subsequence =
                    tile_rng_base + (static_cast<uint64_t>(t) * 2ull + 0ull) * THREADS + threadIdx.x;
                LocalCTARNGState row_rng;
                if constexpr (ROW_DATA_SR && !FAST_DATA_SR) {
                    row_rng.init(rng_seed, row_rng_subsequence, 0);
                }
                uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                int row_rnd_idx = 4;
                const uint64_t row_fast_sr_base =
                    rng_seed ^ row_rng_subsequence ^ 0xd1342543de82ef95ull;
                rowwise_scaling_opt<
                    ENCODE_CENTRIC, false, ROW_DATA_SR, FAST_DATA_SR,
                    false, false, false>(
                    sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out,
                    row_rng, row_random_uint4, row_rnd_idx, row_fast_sr_base);

                const uint64_t col_rng_subsequence =
                    tile_rng_base + (static_cast<uint64_t>(t) * 2ull + 1ull) * THREADS + threadIdx.x;
                LocalCTARNGState col_rng;
                if constexpr (WITH_RANDOM_SIGN_MASK) {
                    col_rng.init(rng_seed, col_rng_subsequence, 0);
                }
                uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                int col_rnd_idx = 4;
                const uint64_t col_fast_sr_base =
                    rng_seed ^ col_rng_subsequence ^ 0x94d049bb133111ebull;
                colwise_scaling_opt<
                    ENCODE_CENTRIC, false, false, FAST_DATA_SR,
                    false, COL_WITH_RHT, WITH_RANDOM_SIGN_MASK>(
                    sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                    S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                    col_rng, col_random_uint4, col_rnd_idx, col_fast_sr_base,
                    block_offset_Y,
                    ctaid_X * LocalCTAConfig::CHUNK_DIM_X);
            } else {
                rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                                                S_enc_row, stage_Y, stage_X, t, buff_out);
                colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                                S_enc_col, stage_Y, stage_X, t, buff_out_tr);
            }

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                const uint64_t* out_map = reinterpret_cast<const uint64_t*>(
                    split_id == 0 ? &tensor_map_output0 : (split_id == 1 ? &tensor_map_output1 : &tensor_map_output2));
                const uint64_t* out_t_map = reinterpret_cast<const uint64_t*>(
                    split_id == 0 ? &tensor_map_output_t0 : (split_id == 1 ? &tensor_map_output_t1 : &tensor_map_output_t2));

                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    out_map,
                    local_block_offset_X + stage_offset_X,
                    block_offset_Y + stage_offset_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    out_t_map,
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
            if (leading) {
                tma_store_scales_2x512(
                    split_id == 0 ? tmap_scale_row0 : (split_id == 1 ? tmap_scale_row1 : tmap_scale_row2),
                    sSFrowwise_ptr,
                    ctaid_Y,
                    local_ctaid_X * 2 * 256);
            }
        }

        {
            const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
            swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(
                    split_id == 0 ? tmap_scale_col0 : (split_id == 1 ? tmap_scale_col1 : tmap_scale_col2),
                    sSFcolwise_ptr,
                    local_ctaid_X,
                    ctaid_Y * 2 * 256);
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

template <
    bool ENCODE_CENTRIC = true,
    bool DIRECT_SWIZZLED_SCALES = false,
    bool APPLY_INVERSE_ROPE = false,
    bool FOLD_ROW_OUTER_SG = false,
    bool ROW_DATA_SR = false,
    bool FAST_DATA_SR = false,
    bool SCALE_SR = false,
    bool COL_DATA_SR = ROW_DATA_SR>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_split3_final_sg_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_input2,
    const __grid_constant__ CUtensorMap tensor_map_output0,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output2,
    const __grid_constant__ CUtensorMap tensor_map_output_t0,
    const __grid_constant__ CUtensorMap tensor_map_output_t1,
    const __grid_constant__ CUtensorMap tensor_map_output_t2,
    const __grid_constant__ CUtensorMap tmap_scale_row0,
    const __grid_constant__ CUtensorMap tmap_scale_row1,
    const __grid_constant__ CUtensorMap tmap_scale_row2,
    const __grid_constant__ CUtensorMap tmap_scale_col0,
    const __grid_constant__ CUtensorMap tmap_scale_col1,
    const __grid_constant__ CUtensorMap tmap_scale_col2,
    const float* __restrict__ row_sg_0,
    const float* __restrict__ row_sg_1,
    const float* __restrict__ row_sg_2,
    const float* __restrict__ col_sg_0,
    const float* __restrict__ col_sg_1,
    const float* __restrict__ col_sg_2,
    const float2* __restrict__ rope_cs,
    int rope_seq_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const unsigned long long* __restrict__ rng_state,
    const size_t rows,
    LocalCTAPersistentArgs args,
    int split0_tiles,
    int split1_tiles,
    int split2_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    if constexpr (ROW_DATA_SR || COL_DATA_SR || SCALE_SR) {
        if (rng_state != nullptr) {
            rng_seed = static_cast<uint64_t>(rng_state[0]);
            rng_subsequence_base = static_cast<uint64_t>(rng_state[1]);
        }
    }
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
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;

        int local_ctaid_X = ctaid_X;
        int split_id = 0;
        int split_tiles = split0_tiles;
        if (ctaid_X >= split01_tiles) {
            split_id = 2;
            split_tiles = split2_tiles;
            local_ctaid_X -= split01_tiles;
        } else if (ctaid_X >= split0_tiles) {
            split_id = 1;
            split_tiles = split1_tiles;
            local_ctaid_X -= split0_tiles;
        }
        const int local_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = block_offset_Y;
        const int block_offset_Y_tr = local_block_offset_X;
        const int chunk_cols = split_tiles * LocalCTAConfig::CHUNK_DIM_X - local_block_offset_X;

        const uint64_t* in_map = reinterpret_cast<const uint64_t*>(
            split_id == 0 ? &tensor_map_input0 : (split_id == 1 ? &tensor_map_input1 : &tensor_map_input2));
        const uint64_t* out_map = reinterpret_cast<const uint64_t*>(
            split_id == 0 ? &tensor_map_output0 : (split_id == 1 ? &tensor_map_output1 : &tensor_map_output2));
        const uint64_t* out_t_map = reinterpret_cast<const uint64_t*>(
            split_id == 0 ? &tensor_map_output_t0 : (split_id == 1 ? &tensor_map_output_t1 : &tensor_map_output_t2));

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int ty = t / TILES_X;
            const int tx = t % TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[t]),
                    in_map,
                    local_block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &in_mbar[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
        }

        if constexpr (APPLY_INVERSE_ROPE) {
            if (split_id < 2) {
                const RopeLive64Desc rope{rope_cs, rope_seq_mask};
                #pragma unroll
                for (int t = 0; t < NUM_TILES; ++t) {
                    const int stage_Y = t / TILES_X;
                    const int stage_X = t % TILES_X;
                    apply_inverse_rope_tile_inplace_live64<false>(
                        sIn_ptr,
                        rope,
                        t,
                        stage_Y,
                        stage_X,
                        block_offset_Y,
                        local_block_offset_X);
                }
                __syncthreads();
            }
        }

        const float* row_sg = split_id == 0 ? row_sg_0 : (split_id == 1 ? row_sg_1 : row_sg_2);
        const float* col_sg = split_id == 0 ? col_sg_0 : (split_id == 1 ? col_sg_1 : col_sg_2);
        const float row_sg_val = row_sg[ctaid_Y / 2];
        const float col_sg_val = col_sg[local_ctaid_X / 2];
        const float S_enc_row = compute_localcta_encode_scaling_factor_FP4(
            row_sg_val * localcta_global_scale_num());
        const float S_enc_col = compute_localcta_encode_scaling_factor_FP4(
            col_sg_val * localcta_global_scale_num());

        int buff_out = 0;
        int buff_out_tr = 0;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int stage_offset_Y = stage_Y * TILE_DIM_Y;
            const int stage_offset_X = stage_X * TILE_DIM_X;

            if (t >= BUFFS_NUM_OUT) {
                if (leading) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }
                __syncthreads();
            }

            if constexpr (ROW_DATA_SR || COL_DATA_SR || SCALE_SR) {
                const uint64_t row_rng_subsequence =
                    rng_subsequence_base +
                    ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 0ull) * THREADS +
                    threadIdx.x;
                LocalCTARNGState row_rng;
                if constexpr ((ROW_DATA_SR && !FAST_DATA_SR) || SCALE_SR) {
                    row_rng.init(rng_seed, row_rng_subsequence, 0);
                }
                uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                int row_rnd_idx = 4;
                const uint64_t row_fast_sr_base =
                    rng_seed ^ row_rng_subsequence ^ 0xd1342543de82ef95ull;
                rowwise_scaling_opt<
                    ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES, ROW_DATA_SR, FAST_DATA_SR,
                    SCALE_SR, false, false>(
                    sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out,
                    row_rng, row_random_uint4, row_rnd_idx, row_fast_sr_base);

                const uint64_t col_rng_subsequence =
                    rng_subsequence_base +
                    ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 1ull) * THREADS +
                    threadIdx.x;
                LocalCTARNGState col_rng;
                if constexpr ((COL_DATA_SR && !FAST_DATA_SR) || SCALE_SR) {
                    col_rng.init(rng_seed, col_rng_subsequence, 0);
                }
                uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                int col_rnd_idx = 4;
                const uint64_t col_fast_sr_base =
                    rng_seed ^ col_rng_subsequence ^ 0x94d049bb133111ebull;
                colwise_scaling_opt<
                    ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES, COL_DATA_SR, FAST_DATA_SR,
                    SCALE_SR, false, false>(
                    sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                    S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                    col_rng, col_random_uint4, col_rnd_idx, col_fast_sr_base);
            } else {
                rowwise_scaling<ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES, FOLD_ROW_OUTER_SG>(
                    sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out, row_sg_val);
                colwise_scaling<ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES>(
                    sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                    S_enc_col, stage_Y, stage_X, t, buff_out_tr);
            }

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    out_map,
                    local_block_offset_X + stage_offset_X,
                    block_offset_Y + stage_offset_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    out_t_map,
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
            if constexpr (!DIRECT_SWIZZLED_SCALES) {
                swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(
                    split_id == 0 ? tmap_scale_row0 : (split_id == 1 ? tmap_scale_row1 : tmap_scale_row2),
                    sSFrowwise_ptr,
                    ctaid_Y,
                    local_ctaid_X * 2 * 256);
            }
        }

        {
            const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
            if constexpr (!DIRECT_SWIZZLED_SCALES) {
                swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(
                    split_id == 0 ? tmap_scale_col0 : (split_id == 1 ? tmap_scale_col1 : tmap_scale_col2),
                    sSFcolwise_ptr,
                    local_ctaid_X,
                    ctaid_Y * 2 * 256);
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

template <
    bool ENCODE_CENTRIC = true,
    bool USE_PRECOMPUTED_AMAX = false,
    bool USE_PREFINALIZED_OUTER_SG = false,
    bool DIRECT_SWIZZLED_SCALES = false,
    bool ROW_DATA_SR = false,
    bool FAST_DATA_SR = false,
    bool SCALE_SR = false,
    bool DELAYED_SCALING = false,
    bool DELAYED_AMAX_READY = false,
    bool READ_ONLY_AMAX = false,
    bool COLLECT_CURRENT_AMAX = false,
    bool COL_DATA_SR = ROW_DATA_SR,
    bool COL_WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_split2_raw_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_output0,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output_t0,
    const __grid_constant__ CUtensorMap tensor_map_output_t1,
    const __grid_constant__ CUtensorMap tmap_scale_row0,
    const __grid_constant__ CUtensorMap tmap_scale_row1,
    const __grid_constant__ CUtensorMap tmap_scale_col0,
    const __grid_constant__ CUtensorMap tmap_scale_col1,
    float* __restrict__ row_sg_chunks0,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks0,
    float* __restrict__ col_sg_chunks1,
    float* __restrict__ row_sg_outer0,
    float* __restrict__ row_sg_outer1,
    float* __restrict__ col_sg_outer0,
    float* __restrict__ col_sg_outer1,
    float* __restrict__ amax_out0,
    float* __restrict__ amax_out1,
    const float* __restrict__ row_amax_in0,
    const float* __restrict__ row_amax_in1,
    float* __restrict__ current_row_amax_out0,
    float* __restrict__ current_col_amax_out0,
    float* __restrict__ current_row_amax_out1,
    float* __restrict__ current_col_amax_out1,
    float* __restrict__ current_row_sg_outer_out0,
    float* __restrict__ current_col_sg_outer_out0,
    float* __restrict__ current_row_sg_outer_out1,
    float* __restrict__ current_col_sg_outer_out1,
    const size_t rows,
    LocalCTAPersistentArgs args,
    int split0_tiles,
    int split1_tiles,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const unsigned long long* __restrict__ rng_state
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    if constexpr (ROW_DATA_SR || COL_DATA_SR || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
        if (rng_state != nullptr) {
            rng_seed = static_cast<uint64_t>(rng_state[0]);
            rng_subsequence_base = static_cast<uint64_t>(rng_state[1]);
        }
    }
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
    __shared__ float warp_max_col[THREADS / 32];
    __shared__ float cta_amax_shared;
    __shared__ float cta_amax_col_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;
    const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);

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
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;

        int local_ctaid_X = ctaid_X;
        bool second_split = false;
        if (ctaid_X >= split0_tiles) {
            second_split = true;
            local_ctaid_X -= split0_tiles;
        }
        const int split_tiles = second_split ? split1_tiles : split0_tiles;
        const int local_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = block_offset_Y;
        const int block_offset_Y_tr = local_block_offset_X;
        const int chunk_cols = split_tiles * LocalCTAConfig::CHUNK_DIM_X - local_block_offset_X;

        float cta_max = 0.0f;
        float cta_col_max = 0.0f;

        #pragma unroll
        for (int pre = 0; pre < min(2, (int)NUM_TILES); ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    second_split
                        ? reinterpret_cast<const uint64_t*>(&tensor_map_input1)
                        : reinterpret_cast<const uint64_t*>(&tensor_map_input0),
                    local_block_offset_X + tx * TILE_DIM_X,
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
                        second_split
                            ? reinterpret_cast<const uint64_t*>(&tensor_map_input1)
                            : reinterpret_cast<const uint64_t*>(&tensor_map_input0),
                        local_block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            if constexpr (!USE_PRECOMPUTED_AMAX || COLLECT_CURRENT_AMAX) {
                cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
                if constexpr (COL_WITH_RHT) {
                    LocalCTARNGState col_scan_rng;
                    if constexpr (WITH_RANDOM_SIGN_MASK) {
                        col_scan_rng.init(
                            rng_seed,
                            rng_subsequence_base +
                                ((static_cast<uint64_t>(s_chunk_id) * NUM_TILES + t) * 2ull + 1ull) *
                                    THREADS +
                                threadIdx.x,
                            0);
                    }
                    uint4 col_scan_random_uint4 = make_uint4(0, 0, 0, 0);
                    int col_scan_rnd_idx = 4;
                    cta_col_max = fmaxf(
                        cta_col_max,
                        scan_tile_amax_col_opt<true, WITH_RANDOM_SIGN_MASK>(
                            sIn_ptr,
                            t,
                            col_scan_rng,
                            col_scan_random_uint4,
                            col_scan_rnd_idx));
                }
            }
        }

        if constexpr (!USE_PRECOMPUTED_AMAX || COLLECT_CURRENT_AMAX) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                if constexpr (COL_WITH_RHT) {
                    cta_col_max = fmaxf(
                        cta_col_max,
                        __shfl_xor_sync(0xffffffff, cta_col_max, mask));
                }
            }
            if (lane == 0) {
                warp_max[wid] = cta_max;
                if constexpr (COL_WITH_RHT) {
                    warp_max_col[wid] = cta_col_max;
                }
            }
            __syncthreads();

            if (wid == 0) {
                cta_max = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
                if constexpr (COL_WITH_RHT) {
                    cta_col_max = (lane < THREADS / 32) ? warp_max_col[lane] : 0.0f;
                }
                #pragma unroll
                for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                    if constexpr (COL_WITH_RHT) {
                        cta_col_max = fmaxf(
                            cta_col_max,
                            __shfl_xor_sync(0xffffffff, cta_col_max, mask));
                    }
                }
                if (lane == 0) {
                    cta_amax_shared = cta_max;
                    if constexpr (COL_WITH_RHT) {
                        cta_amax_col_shared = cta_col_max;
                    }
                }
            }
            __syncthreads();
        }

        float S_enc_row = 1.0f;
        float S_enc_col = 1.0f;
        float row_stored_scale_multiplier = 1.0f;
        float col_stored_scale_multiplier = 1.0f;
        if constexpr (USE_PREFINALIZED_OUTER_SG && READ_ONLY_AMAX) {
            const int row_tile = ctaid_Y / 2;
            const int col_tile = local_ctaid_X / 2;
            const float prev_amax_val = second_split
                ? row_amax_in1[ctaid_Y * split1_tiles + local_ctaid_X]
                : row_amax_in0[ctaid_Y * split0_tiles + local_ctaid_X];
            const float prev_tile_sg = prev_amax_val / localcta_global_scale_num();
            const float row_outer_sg = second_split ? row_sg_outer1[row_tile] : row_sg_outer0[row_tile];
            const float col_outer_sg = second_split ? col_sg_outer1[col_tile] : col_sg_outer0[col_tile];
            const float S_enc = compute_localcta_encode_scaling_factor_FP4(prev_amax_val);
            S_enc_row = S_enc;
            S_enc_col = S_enc;
            row_stored_scale_multiplier = prev_tile_sg / fmaxf(row_outer_sg, 1.0e-12f);
            col_stored_scale_multiplier = prev_tile_sg / fmaxf(col_outer_sg, 1.0e-12f);
            if (leading) {
                if constexpr (COLLECT_CURRENT_AMAX) {
                    const float current_amax = cta_amax_shared;
                    const float current_sg = current_amax / localcta_global_scale_num();
                    if (second_split) {
                        current_row_amax_out1[ctaid_Y * split1_tiles + local_ctaid_X] = current_amax;
                        if (current_col_amax_out1 != nullptr) {
                            current_col_amax_out1[local_ctaid_X * tiles_Y + ctaid_Y] = current_amax;
                        }
                        if (current_row_sg_outer_out1 != nullptr) {
                            transformer_engine::atomicMaxFloat(current_row_sg_outer_out1 + row_tile, current_sg);
                        }
                        if (current_col_sg_outer_out1 != nullptr) {
                            transformer_engine::atomicMaxFloat(current_col_sg_outer_out1 + col_tile, current_sg);
                        }
                    } else {
                        current_row_amax_out0[ctaid_Y * split0_tiles + local_ctaid_X] = current_amax;
                        if (current_col_amax_out0 != nullptr) {
                            current_col_amax_out0[local_ctaid_X * tiles_Y + ctaid_Y] = current_amax;
                        }
                        if (current_row_sg_outer_out0 != nullptr) {
                            transformer_engine::atomicMaxFloat(current_row_sg_outer_out0 + row_tile, current_sg);
                        }
                        if (current_col_sg_outer_out0 != nullptr) {
                            transformer_engine::atomicMaxFloat(current_col_sg_outer_out0 + col_tile, current_sg);
                        }
                    }
                }
            }
        } else if constexpr (DELAYED_SCALING) {
            S_enc_row = 1.0f;
            S_enc_col = 1.0f;
            if constexpr (USE_PREFINALIZED_OUTER_SG) {
                if (leading) {
                    const int row_tile = ctaid_Y / 2;
                    const int col_tile = local_ctaid_X / 2;
                    if ((ctaid_Y & 1) == 0 && local_ctaid_X == 0) {
                        float* row_outer_sg = second_split ? row_sg_outer1 : row_sg_outer0;
                        if (row_outer_sg != nullptr) {
                            row_outer_sg[row_tile] = 1.0f;
                        }
                    }
                    if (ctaid_Y == 0 && (local_ctaid_X & 1) == 0) {
                        float* col_outer_sg = second_split ? col_sg_outer1 : col_sg_outer0;
                        if (col_outer_sg != nullptr) {
                            col_outer_sg[col_tile] = 1.0f;
                        }
                    }
                }
            }
            if constexpr (!DELAYED_AMAX_READY) {
                const float amax_val = USE_PRECOMPUTED_AMAX
                    ? (second_split
                        ? row_sg_chunks1[ctaid_Y * split1_tiles + local_ctaid_X]
                        : row_sg_chunks0[ctaid_Y * split0_tiles + local_ctaid_X])
                    : cta_amax_shared;
                if (leading) {
                    float* amax_out = second_split ? amax_out1 : amax_out0;
                    if (amax_out != nullptr) {
                        transformer_engine::atomicMaxFloat(amax_out, amax_val);
                    }
                }
            }
        } else if constexpr (USE_PREFINALIZED_OUTER_SG) {
            const int row_tile = ctaid_Y / 2;
            const int col_tile = local_ctaid_X / 2;
            const float row_outer_sg = second_split ? row_sg_outer1[row_tile] : row_sg_outer0[row_tile];
            const float col_outer_sg = second_split ? col_sg_outer1[col_tile] : col_sg_outer0[col_tile];
            S_enc_row = compute_localcta_encode_scaling_factor_FP4(
                row_outer_sg * localcta_global_scale_num());
            S_enc_col = compute_localcta_encode_scaling_factor_FP4(
                col_outer_sg * localcta_global_scale_num());
        } else {
            const float row_amax_val = USE_PRECOMPUTED_AMAX
                ? (READ_ONLY_AMAX
                    ? (second_split
                        ? row_amax_in1[ctaid_Y * split1_tiles + local_ctaid_X]
                        : row_amax_in0[ctaid_Y * split0_tiles + local_ctaid_X])
                    : (second_split
                        ? row_sg_chunks1[ctaid_Y * split1_tiles + local_ctaid_X]
                        : row_sg_chunks0[ctaid_Y * split0_tiles + local_ctaid_X]))
                : cta_amax_shared;
            const float col_amax_val = COL_WITH_RHT
                ? cta_amax_col_shared
                : row_amax_val;
            S_enc_row = compute_localcta_encode_scaling_factor_FP4(row_amax_val);
            S_enc_col = compute_localcta_encode_scaling_factor_FP4(col_amax_val);
            if (leading) {
                const float row_sg_val = row_amax_val / localcta_global_scale_num();
                const float col_sg_val = col_amax_val / localcta_global_scale_num();
                if (second_split) {
                    row_sg_chunks1[ctaid_Y * split1_tiles + local_ctaid_X] = row_sg_val;
                    col_sg_chunks1[local_ctaid_X * tiles_Y + ctaid_Y] = col_sg_val;
                } else {
                    row_sg_chunks0[ctaid_Y * split0_tiles + local_ctaid_X] = row_sg_val;
                    col_sg_chunks0[local_ctaid_X * tiles_Y + ctaid_Y] = col_sg_val;
                }
                if constexpr (COLLECT_CURRENT_AMAX) {
                    const float current_amax = cta_amax_shared;
                    if (second_split) {
                        current_row_amax_out1[ctaid_Y * split1_tiles + local_ctaid_X] = current_amax;
                        if (current_col_amax_out1 != nullptr) {
                            current_col_amax_out1[local_ctaid_X * tiles_Y + ctaid_Y] = current_amax;
                        }
                    } else {
                        current_row_amax_out0[ctaid_Y * split0_tiles + local_ctaid_X] = current_amax;
                        if (current_col_amax_out0 != nullptr) {
                            current_col_amax_out0[local_ctaid_X * tiles_Y + ctaid_Y] = current_amax;
                        }
                    }
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

            if (t >= BUFFS_NUM_OUT) {
                if (leading) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }
                __syncthreads();
            }

            if constexpr (ROW_DATA_SR || COL_DATA_SR || SCALE_SR || COL_WITH_RHT) {
                const uint64_t tile_rng_base =
                    rng_subsequence_base +
                    static_cast<uint64_t>(s_chunk_id) * 2ull * NUM_TILES * THREADS;
                const uint64_t row_rng_subsequence =
                    tile_rng_base + (static_cast<uint64_t>(t) * 2ull + 0ull) * THREADS + threadIdx.x;
                LocalCTARNGState row_rng;
                if constexpr ((ROW_DATA_SR && !FAST_DATA_SR) || SCALE_SR) {
                    row_rng.init(rng_seed, row_rng_subsequence, 0);
                }
                uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                int row_rnd_idx = 4;
                const uint64_t row_fast_sr_base =
                    rng_seed ^ row_rng_subsequence ^ 0xd1342543de82ef95ull;
                rowwise_scaling_opt<
                    ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES, ROW_DATA_SR, FAST_DATA_SR,
                    SCALE_SR, false, false>(
                    sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out,
                    row_rng, row_random_uint4, row_rnd_idx, row_fast_sr_base);

                const uint64_t col_rng_subsequence =
                    tile_rng_base + (static_cast<uint64_t>(t) * 2ull + 1ull) * THREADS + threadIdx.x;
                LocalCTARNGState col_rng;
                if constexpr ((COL_DATA_SR && !FAST_DATA_SR) || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
                    col_rng.init(rng_seed, col_rng_subsequence, 0);
                }
                uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                int col_rnd_idx = 4;
                const uint64_t col_fast_sr_base =
                    rng_seed ^ col_rng_subsequence ^ 0x94d049bb133111ebull;
                colwise_scaling_opt<
                    ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES, COL_DATA_SR, FAST_DATA_SR,
                    SCALE_SR, COL_WITH_RHT, WITH_RANDOM_SIGN_MASK>(
                    sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                    S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                    col_rng, col_random_uint4, col_rnd_idx, col_fast_sr_base,
                    block_offset_Y,
                    ctaid_X * LocalCTAConfig::CHUNK_DIM_X);
            } else {
                if constexpr (USE_PREFINALIZED_OUTER_SG && READ_ONLY_AMAX) {
                    rowwise_scaling<ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES, true>(
                        sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                        S_enc_row, stage_Y, stage_X, t, buff_out,
                        row_stored_scale_multiplier);
                    colwise_scaling<ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES, true>(
                        sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                        S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                        col_stored_scale_multiplier);
                } else {
                    rowwise_scaling<ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES>(
                        sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                        S_enc_row, stage_Y, stage_X, t, buff_out);
                    colwise_scaling<ENCODE_CENTRIC, DIRECT_SWIZZLED_SCALES>(
                        sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                        S_enc_col, stage_Y, stage_X, t, buff_out_tr);
                }
            }

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    second_split
                        ? reinterpret_cast<const uint64_t*>(&tensor_map_output1)
                        : reinterpret_cast<const uint64_t*>(&tensor_map_output0),
                    local_block_offset_X + stage_offset_X,
                    block_offset_Y + stage_offset_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    second_split
                        ? reinterpret_cast<const uint64_t*>(&tensor_map_output_t1)
                        : reinterpret_cast<const uint64_t*>(&tensor_map_output_t0),
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
            if constexpr (!DIRECT_SWIZZLED_SCALES) {
                swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(
                    second_split ? tmap_scale_row1 : tmap_scale_row0,
                    sSFrowwise_ptr,
                    ctaid_Y,
                    local_ctaid_X * 2 * 256);
            }
        }

        {
            const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
            if constexpr (!DIRECT_SWIZZLED_SCALES) {
                swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            }
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(
                    second_split ? tmap_scale_col1 : tmap_scale_col0,
                    sSFcolwise_ptr,
                    local_ctaid_X,
                    ctaid_Y * 2 * 256);
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
    __shared__ volatile int slot_ready[PIPE_DEPTH];
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

struct DualLocalMaxGroup {
    float a;
    float b;
};

__device__ __forceinline__ void store_chunk_value_linear(
    IType* sIn_ptr,
    int row,
    int col,
    __nv_bfloat16 value
) {
    const int tile_y = row / TILE_DIM_Y;
    const int tile_x = col / TILE_DIM_X;
    const int tile = tile_y * TILES_X + tile_x;
    store_chunk_value_group(sIn_ptr, tile, row % TILE_DIM_Y, col % TILE_DIM_X, value);
}

template <int GROUP_THREADS>
__device__ __forceinline__ DualLocalMaxGroup load_silu_deriv_chunk_group(
    IType* sIn1_ptr,
    IType* sIn2_ptr,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X,
    int tid
) {
    DualLocalMaxGroup local{0.0f, 0.0f};
    constexpr int VEC = 4;
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = tid * VEC; idx < TOTAL; idx += GROUP_THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

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

            store_chunk_value_linear(sIn1_ptr, row, col + 0, o10.x);
            store_chunk_value_linear(sIn1_ptr, row, col + 1, o10.y);
            store_chunk_value_linear(sIn1_ptr, row, col + 2, o11.x);
            store_chunk_value_linear(sIn1_ptr, row, col + 3, o11.y);
            store_chunk_value_linear(sIn2_ptr, row, col + 0, o20.x);
            store_chunk_value_linear(sIn2_ptr, row, col + 1, o20.y);
            store_chunk_value_linear(sIn2_ptr, row, col + 2, o21.x);
            store_chunk_value_linear(sIn2_ptr, row, col + 3, o21.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out1 = __float2bfloat16_rn(0.0f);
                __nv_bfloat16 out2 = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset =
                        static_cast<int64_t>(global_row) * cols + block_offset_X + c;
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
                store_chunk_value_linear(sIn1_ptr, row, c, out1);
                store_chunk_value_linear(sIn2_ptr, row, c, out2);
            }
        }
    }
    subgroup_barrier_sync<GROUP_THREADS>();
    return local;
}

template <bool OUTPUT_DH1, int GROUP_THREADS>
__device__ __forceinline__ float load_silu_deriv_chunk_single_group(
    IType* sIn_ptr,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int block_offset_X,
    int tid
) {
    float local = 0.0f;
    constexpr int VEC = 4;
    constexpr int TOTAL = LocalCTAConfig::CHUNK_DIM_Y * LocalCTAConfig::CHUNK_DIM_X;

    for (int idx = tid * VEC; idx < TOTAL; idx += GROUP_THREADS * VEC) {
        const int row = idx / LocalCTAConfig::CHUNK_DIM_X;
        const int col = idx % LocalCTAConfig::CHUNK_DIM_X;
        const int global_row = block_offset_Y + row;
        const int global_col = block_offset_X + col;

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
                out0 = __float22bfloat162_rn(
                    make_float2(d0f.x * silu0x, d0f.y * silu0y));
                out1 = __float22bfloat162_rn(
                    make_float2(d1f.x * silu1x, d1f.y * silu1y));
            }

            const float2 out0f = __bfloat1622float2(out0);
            const float2 out1f = __bfloat1622float2(out1);
            local = fmaxf(local, fabsf(out0f.x));
            local = fmaxf(local, fabsf(out0f.y));
            local = fmaxf(local, fabsf(out1f.x));
            local = fmaxf(local, fabsf(out1f.y));

            store_chunk_value_linear(sIn_ptr, row, col + 0, out0.x);
            store_chunk_value_linear(sIn_ptr, row, col + 1, out0.y);
            store_chunk_value_linear(sIn_ptr, row, col + 2, out1.x);
            store_chunk_value_linear(sIn_ptr, row, col + 3, out1.y);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int c = col + j;
                __nv_bfloat16 out = __float2bfloat16_rn(0.0f);
                if (global_row < rows && block_offset_X + c < cols) {
                    const int64_t offset =
                        static_cast<int64_t>(global_row) * cols + block_offset_X + c;
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
                    local = fmaxf(local, fabsf(__bfloat162float(out)));
                }
                store_chunk_value_linear(sIn_ptr, row, c, out);
            }
        }
    }
    subgroup_barrier_sync<GROUP_THREADS>();
    return local;
}

template <int GROUP_THREADS, bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
__device__ __forceinline__ void quantize_store_prepared_chunk_group(
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
    int block_offset_Y_tr,
    int block_offset_X_tr,
    int chunk_rows,
    int chunk_cols,
    int ctaid_X,
    int ctaid_Y,
    int tid
) {
    const bool consumer_leader = (tid == 0);
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

        if (t > 0 && consumer_leader) {
            ptx::cp_async_bulk_wait_group_read<1>();
        }
        if (t > 0) {
            subgroup_barrier_sync<GROUP_THREADS>();
        }

        rowwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc, stage_Y, stage_X, t, buff_out, tid);

        if constexpr (RETURN_TRANSPOSE) {
            colwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
                sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc, stage_Y, stage_X, t, buff_out_tr, tid);
        }

        subgroup_barrier_sync<GROUP_THREADS>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        subgroup_barrier_sync<GROUP_THREADS>();

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
        if constexpr (RETURN_TRANSPOSE) {
            buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
        }
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
    }

    if constexpr (RETURN_TRANSPOSE) {
        if (consumer_leader) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<GROUP_THREADS>();
        swizzle_scales_col_inplace_group<GROUP_THREADS>(
            sSFcolwise_ptr,
            min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM),
            tid);
        scale_swizzled_scales_inplace_group<GROUP_THREADS>(
            sSFcolwise_ptr,
            LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
            sg_val,
            tid);
        subgroup_barrier_sync<GROUP_THREADS>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        subgroup_barrier_sync<GROUP_THREADS>();
        if (consumer_leader) {
            tma_store_scales_2x512(
                tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<GROUP_THREADS>();
    }
}

template <int GROUP_THREADS, bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
__device__ __forceinline__ void quantize_store_raw_chunk_group(
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
    int block_offset_Y_tr,
    int block_offset_X_tr,
    int chunk_rows,
    int chunk_cols,
    int ctaid_X,
    int ctaid_Y,
    int tid
) {
    const bool consumer_leader = (tid == 0);
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

        if (t > 0 && consumer_leader) {
            ptx::cp_async_bulk_wait_group_read<1>();
        }
        if (t > 0) {
            subgroup_barrier_sync<GROUP_THREADS>();
        }

        rowwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc, stage_Y, stage_X, t, buff_out, tid);

        if constexpr (RETURN_TRANSPOSE) {
            colwise_scaling_group<GROUP_THREADS, ENCODE_CENTRIC>(
                sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc, stage_Y, stage_X, t, buff_out_tr, tid);
        }

        subgroup_barrier_sync<GROUP_THREADS>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        subgroup_barrier_sync<GROUP_THREADS>();

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
        if constexpr (RETURN_TRANSPOSE) {
            buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
        }
    }

    if (consumer_leader) {
        ptx::cp_async_bulk_wait_group_read<0>();
    }
    subgroup_barrier_sync<GROUP_THREADS>();

    swizzle_scales_row_inplace_group<GROUP_THREADS>(
        sSFrowwise_ptr,
        min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM),
        tid);
    subgroup_barrier_sync<GROUP_THREADS>();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    subgroup_barrier_sync<GROUP_THREADS>();
    if (consumer_leader) {
        tma_store_scales_2x512(tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
    }

    if constexpr (RETURN_TRANSPOSE) {
        if (consumer_leader) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<GROUP_THREADS>();
        swizzle_scales_col_inplace_group<GROUP_THREADS>(
            sSFcolwise_ptr,
            min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM),
            tid);
        subgroup_barrier_sync<GROUP_THREADS>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        subgroup_barrier_sync<GROUP_THREADS>();
        if (consumer_leader) {
            tma_store_scales_2x512(tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        subgroup_barrier_sync<GROUP_THREADS>();
    }
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

            if (out_bf16 != nullptr) {
                *reinterpret_cast<__nv_bfloat162*>(out_bf16 + base) = out0;
                *reinterpret_cast<__nv_bfloat162*>(out_bf16 + base + 2) = out1;
            }

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
                    if (out_bf16 != nullptr) {
                        out_bf16[offset] = out;
                    }
                }
                store_chunk_value_group(sIn_ptr, tile_idx, local_row, local_c, out);
            }
        }
    }

    return local_max;
}

template <bool OUTPUT_DH1>
__device__ __forceinline__ float transform_store_silu_deriv_split2_tile_warp(
    IType* sIn_ptr,
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    int rows,
    int cols,
    int block_offset_Y,
    int input_block_offset_X,
    int tile_idx,
    int producer_tid
) {
    constexpr int PRODUCER_THREADS = 32;
    constexpr int VEC = 4;
    constexpr int TILE_ELTS = TILE_DIM_Y * TILE_DIM_X;

    const int stage_Y = tile_idx / TILES_X;
    const int stage_X = tile_idx % TILES_X;
    const int tile_offset_Y = stage_Y * TILE_DIM_Y;
    const int tile_offset_X = stage_X * TILE_DIM_X;
    float local_max = 0.0f;

    for (int idx = producer_tid * VEC; idx < TILE_ELTS; idx += PRODUCER_THREADS * VEC) {
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

            const float sig0x = 1.0f / (1.0f + __expf(-b0f.x));
            const float sig0y = 1.0f / (1.0f + __expf(-b0f.y));
            const float sig1x = 1.0f / (1.0f + __expf(-b1f.x));
            const float sig1y = 1.0f / (1.0f + __expf(-b1f.y));

            const float silu0x = b0f.x * sig0x;
            const float silu0y = b0f.y * sig0y;
            const float silu1x = b1f.x * sig1x;
            const float silu1y = b1f.y * sig1y;

            float out0x;
            float out0y;
            float out1x;
            float out1y;
            __nv_bfloat162 out0;
            __nv_bfloat162 out1;
            if constexpr (OUTPUT_DH1) {
                const float silup0x = sig0x * (1.0f + b0f.x - silu0x);
                const float silup0y = sig0y * (1.0f + b0f.y - silu0y);
                const float silup1x = sig1x * (1.0f + b1f.x - silu1x);
                const float silup1y = sig1y * (1.0f + b1f.y - silu1y);
                out0x = d0f.x * a0f.x * silup0x;
                out0y = d0f.y * a0f.y * silup0y;
                out1x = d1f.x * a1f.x * silup1x;
                out1y = d1f.y * a1f.y * silup1y;
            } else {
                out0x = d0f.x * silu0x;
                out0y = d0f.y * silu0y;
                out1x = d1f.x * silu1x;
                out1y = d1f.y * silu1y;
            }
            out0 = __float22bfloat162_rn(make_float2(out0x, out0y));
            out1 = __float22bfloat162_rn(make_float2(out1x, out1y));
            local_max = fmaxf(local_max, fabsf(out0x));
            local_max = fmaxf(local_max, fabsf(out0y));
            local_max = fmaxf(local_max, fabsf(out1x));
            local_max = fmaxf(local_max, fabsf(out1y));

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
                    local_max = fmaxf(local_max, fabsf(transformed));
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

template <int GROUP_THREADS = 128, bool RETURN_TRANSPOSE = true, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(GROUP_THREADS)
fused_localcta_silu_deriv_split2_prepared_tuned(
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
    const size_t split_cols,
    LocalCTAPersistentArgs args,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(GROUP_THREADS == 128, "split2 fused tuned kernel expects 128 consumer threads");

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

    IType* sIn1_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn2_ptr = reinterpret_cast<IType*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    __shared__ unsigned int s_chunk_id;
    __shared__ float warp_max0[GROUP_THREADS / 32];
    __shared__ float warp_max1[GROUP_THREADS / 32];
    __shared__ float cta_amax0_shared;
    __shared__ float cta_amax1_shared;

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
        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_other = (ctaid_X + split0_tiles) * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_Y_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_Y_tr_other = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr_other = (ctaid_X + split0_tiles) * LocalCTAConfig::CHUNK_DIM_X;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
        const int chunk_cols = static_cast<int>(split_cols) - block_offset_X;

        DualLocalMaxGroup local = load_silu_deriv_chunk_group<GROUP_THREADS>(
            sIn1_ptr, sIn2_ptr, dh, h3, h1_raw,
            static_cast<int>(rows), static_cast<int>(split_cols),
            block_offset_Y, block_offset_X, tid);
        subgroup_barrier_sync<GROUP_THREADS>();

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local.a = fmaxf(local.a, __shfl_xor_sync(0xffffffff, local.a, mask));
            local.b = fmaxf(local.b, __shfl_xor_sync(0xffffffff, local.b, mask));
        }
        if (lane == 0) {
            warp_max0[wid] = local.a;
            warp_max1[wid] = local.b;
        }
        subgroup_barrier_sync<GROUP_THREADS>();

        if (wid == 0) {
            float cta_max0 = (lane < GROUP_THREADS / 32) ? warp_max0[lane] : 0.0f;
            float cta_max1 = (lane < GROUP_THREADS / 32) ? warp_max1[lane] : 0.0f;
            #pragma unroll
            for (int mask = (GROUP_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max0 = fmaxf(cta_max0, __shfl_xor_sync(0xffffffff, cta_max0, mask));
                cta_max1 = fmaxf(cta_max1, __shfl_xor_sync(0xffffffff, cta_max1, mask));
            }
            if (lane == 0) {
                cta_amax0_shared = cta_max0;
                cta_amax1_shared = cta_max1;
                const int row_sg_cols = split0_tiles * 4;
                const int row_sg_col0 = ctaid_X * 2;
                const int row_sg_col1 = split0_tiles * 2 + ctaid_X * 2;
                const float row_sg0 = cta_max0 / localcta_global_scale_num();
                const float row_sg1 = cta_max1 / localcta_global_scale_num();
                row_sg_chunks[ctaid_Y * row_sg_cols + row_sg_col0] = row_sg0;
                row_sg_chunks[ctaid_Y * row_sg_cols + row_sg_col0 + 1] = row_sg0;
                row_sg_chunks[ctaid_Y * row_sg_cols + row_sg_col1] = row_sg1;
                row_sg_chunks[ctaid_Y * row_sg_cols + row_sg_col1 + 1] = row_sg1;
                if constexpr (RETURN_TRANSPOSE) {
                    const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                    col_sg_chunks[row_sg_col0 * tiles_Y_full + ctaid_Y] = row_sg0;
                    col_sg_chunks[(row_sg_col0 + 1) * tiles_Y_full + ctaid_Y] = row_sg0;
                    col_sg_chunks[row_sg_col1 * tiles_Y_full + ctaid_Y] = row_sg1;
                    col_sg_chunks[(row_sg_col1 + 1) * tiles_Y_full + ctaid_Y] = row_sg1;
                }
            }
        }
        subgroup_barrier_sync<GROUP_THREADS>();

        quantize_store_prepared_chunk_group<GROUP_THREADS, RETURN_TRANSPOSE, ENCODE_CENTRIC>(
            sIn1_ptr, sOut_ptr, sOut_tr_ptr, sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            compute_localcta_encode_scaling_factor_FP4(cta_amax0_shared),
            cta_amax0_shared / localcta_global_scale_num(),
            block_offset_Y, block_offset_X,
            block_offset_X_tr, block_offset_Y_tr,
            chunk_rows, chunk_cols, ctaid_X, ctaid_Y, tid);

        quantize_store_prepared_chunk_group<GROUP_THREADS, RETURN_TRANSPOSE, ENCODE_CENTRIC>(
            sIn2_ptr, sOut_ptr, sOut_tr_ptr, sSFrowwise_ptr, sSFcolwise_ptr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row_prepared, tmap_scale_col_prepared,
            compute_localcta_encode_scaling_factor_FP4(cta_amax1_shared),
            cta_amax1_shared / localcta_global_scale_num(),
            block_offset_Y, block_offset_X_other,
            block_offset_X_tr_other, block_offset_Y_tr_other,
            chunk_rows, chunk_cols, ctaid_X + split0_tiles, ctaid_Y, tid);
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

template <int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX,
          bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS) __cluster_dims__(2, 1, 1)
fused_localcta_silu_deriv_split2_kernel_2cta_prepared_tuned(
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
    const size_t split_cols,
    int tiles_X,
    int tiles_Y,
    int total_macro_tiles,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS >= 160 && TOTAL_THREADS <= 512 && TOTAL_THREADS % 32 == 0,
                  "v4 fused split2 2CTA tuned kernel expects 128 consumer threads plus producer warp(s)");
    constexpr int CONSUMER_THREADS = 128;
    constexpr int PRODUCER_THREADS = TOTAL_THREADS - CONSUMER_THREADS;
    constexpr int PRODUCER_WARPS = PRODUCER_THREADS / 32;
    constexpr int ACTIVE_PRODUCER_WARPS =
        (PRODUCER_WARPS < BUFFS_NUM_IN) ? PRODUCER_WARPS : BUFFS_NUM_IN;
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
    static_assert(PRODUCER_THREADS >= 32, "v4 fused split2 2CTA tuned kernel requires at least one producer warp");

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

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const bool is_producer = !is_consumer;
    const int consumer_tid = threadIdx.x;
    const int producer_tid = threadIdx.x - CONSUMER_THREADS;
    const int producer_lane = producer_tid & 31;
    const int producer_warp = producer_tid >> 5;
    const bool consumer_leader = (threadIdx.x == 0);
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);

    const int cta_rank = blockIdx.x & 1;
    const int peer_cta = cta_rank ^ 1;
    const int cluster_id = blockIdx.x >> 1;
    const int num_clusters = max(1, (int)(gridDim.x >> 1));
    const int cluster_iters = (cluster_id < total_macro_tiles)
        ? ((total_macro_tiles - 1 - cluster_id) / num_clusters + 1)
        : 0;

    __shared__ volatile int slot_ready[PIPE_DEPTH];
    __shared__ int slot_macro_id[PIPE_DEPTH];
    __shared__ volatile int slot_prod_done[PIPE_DEPTH];
    __shared__ float slot_local_amax[PIPE_DEPTH];
    __shared__ volatile uint32_t slot_local_ready[PIPE_DEPTH];
    __shared__ float slot_combined_amax[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_macro_id[s] = -1;
            slot_prod_done[s] = 0;
            slot_local_amax[s] = 0.0f;
            slot_local_ready[s] = 0;
            slot_combined_amax[s] = 0.0f;
        }
    }
    __syncthreads();

    if (is_producer) {
        for (int fill_iter = 0; fill_iter < cluster_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            if (producer_leader) {
                while (slot_ready[slot] != 0) {
                    __nanosleep(64);
                }
                slot_macro_id[slot] = cluster_id + fill_iter * num_clusters;
                slot_local_ready[slot] = 0;
                slot_prod_done[slot] = 0;
                __threadfence_block();
                slot_ready[slot] = -1;
            }
            while (slot_ready[slot] != -1) {
                __nanosleep(16);
            }

            const int macro_tile_id = cluster_id + fill_iter * num_clusters;
            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            int local_ctaid_X = ctaid_X;
            bool output_dh1 = true;
            if (ctaid_X >= split0_tiles) {
                output_dh1 = false;
                local_ctaid_X -= split0_tiles;
            }
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            if (producer_warp < ACTIVE_PRODUCER_WARPS && cta_active) {
                for (int t = producer_warp; t < NUM_TILES_PER_CHUNK; t += ACTIVE_PRODUCER_WARPS) {
                    if (output_dh1) {
                        transform_store_silu_deriv_split2_tile_warp<true>(
                            &sIn_ring[slot][0][0][0],
                            dh, h3, h1_raw,
                            static_cast<int>(rows), static_cast<int>(split_cols),
                            block_offset_Y, input_block_offset_X,
                            t, producer_lane);
                    } else {
                        transform_store_silu_deriv_split2_tile_warp<false>(
                            &sIn_ring[slot][0][0][0],
                            dh, h3, h1_raw,
                            static_cast<int>(rows), static_cast<int>(split_cols),
                            block_offset_Y, input_block_offset_X,
                            t, producer_lane);
                    }
                }
            }

            if (producer_warp < ACTIVE_PRODUCER_WARPS) {
                __syncwarp();
                __threadfence_block();
            }
            if (producer_warp < ACTIVE_PRODUCER_WARPS && producer_lane == 0) {
                atomicAdd((int*)&slot_prod_done[slot], 1);
            }

            if (producer_leader) {
                while (slot_prod_done[slot] < ACTIVE_PRODUCER_WARPS) {
                    __nanosleep(16);
                }
                __threadfence_block();
                slot_ready[slot] = 1;
            }
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < cluster_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 1) {
                __nanosleep(64);
            }

            const int macro_tile_id = slot_macro_id[slot];
            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;

            if (cta_active) {
                const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int output_block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                int local_ctaid_X = ctaid_X;
                if (ctaid_X >= split0_tiles) {
                    local_ctaid_X -= split0_tiles;
                }
                const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int block_offset_Y_tr = output_block_offset_X;
                const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
                const int chunk_cols = static_cast<int>(split_cols) - input_block_offset_X;

                float cta_max = 0.0f;
                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    cta_max = fmaxf(
                        cta_max,
                        scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
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
                            while (cluster_load_shared_u32(
                                       (const uint32_t*)&slot_local_ready[slot],
                                       peer_cta) == 0u) {
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
                    const int sg_cols = tiles_X * 2;
                    const int sg_col = ctaid_X * 2;
                    row_sg_chunks[ctaid_Y * sg_cols + sg_col] = sg_val;
                    row_sg_chunks[ctaid_Y * sg_cols + sg_col + 1] = sg_val;
                    if constexpr (RETURN_TRANSPOSE) {
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[sg_col * tiles_Y_full + ctaid_Y] = sg_val;
                        col_sg_chunks[(sg_col + 1) * tiles_Y_full + ctaid_Y] = sg_val;
                    }
                }
                quantize_store_prepared_chunk_group<CONSUMER_THREADS, RETURN_TRANSPOSE, ENCODE_CENTRIC>(
                    &sIn_ring[slot][0][0][0],
                    sOut_ptr, sOut_tr_ptr, sSFrowwise_ptr, sSFcolwise_ptr,
                    tensor_map_output, tensor_map_output_t,
                    tmap_scale_row_prepared, tmap_scale_col_prepared,
                    S_enc, sg_val,
                    block_offset_Y, output_block_offset_X,
                    block_offset_Y_tr, block_offset_X_tr,
                    chunk_rows, chunk_cols, ctaid_X, ctaid_Y, consumer_tid);

                if (consumer_leader) {
                    slot_local_ready[slot] = 0;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
            }

            if (consumer_leader) {
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX,
          bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true,
          bool DELAYED_SCALING = false>
__global__ void __launch_bounds__(TOTAL_THREADS) __cluster_dims__(2, 1, 1)
fused_localcta_silu_deriv_split2_kernel_2cta_raw_tuned(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    const __grid_constant__ CUtensorMap tensor_map_output0,
    const __grid_constant__ CUtensorMap tensor_map_output_t0,
    const __grid_constant__ CUtensorMap tmap_scale_row0,
    const __grid_constant__ CUtensorMap tmap_scale_col0,
    float* __restrict__ row_sg_chunks0,
    float* __restrict__ col_sg_chunks0,
    const __grid_constant__ CUtensorMap tensor_map_output1,
    const __grid_constant__ CUtensorMap tensor_map_output_t1,
    const __grid_constant__ CUtensorMap tmap_scale_row1,
    const __grid_constant__ CUtensorMap tmap_scale_col1,
    float* __restrict__ row_sg_chunks1,
    float* __restrict__ col_sg_chunks1,
    float* __restrict__ amax_out0,
    float* __restrict__ amax_out1,
    const size_t rows,
    const size_t split_cols,
    int tiles_X,
    int tiles_Y,
    int total_macro_tiles,
    int split0_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS >= 160 && TOTAL_THREADS <= 512 && TOTAL_THREADS % 32 == 0,
                  "v4 fused split2 raw 2CTA kernel expects 128 consumer threads plus producer warp(s)");
    constexpr int CONSUMER_THREADS = 128;
    constexpr int PRODUCER_THREADS = TOTAL_THREADS - CONSUMER_THREADS;
    constexpr int PRODUCER_WARPS = PRODUCER_THREADS / 32;
    constexpr int ACTIVE_PRODUCER_WARPS =
        (PRODUCER_WARPS < BUFFS_NUM_IN) ? PRODUCER_WARPS : BUFFS_NUM_IN;
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
    static_assert(PRODUCER_THREADS >= 32, "v4 fused split2 raw 2CTA kernel requires a producer warp");

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

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const bool is_producer = !is_consumer;
    const int consumer_tid = threadIdx.x;
    const int producer_tid = threadIdx.x - CONSUMER_THREADS;
    const int producer_lane = producer_tid & 31;
    const int producer_warp = producer_tid >> 5;
    const bool consumer_leader = (threadIdx.x == 0);
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);

    const int cta_rank = blockIdx.x & 1;
    const int peer_cta = cta_rank ^ 1;
    const int cluster_id = blockIdx.x >> 1;
    const int num_clusters = max(1, (int)(gridDim.x >> 1));
    const int cluster_iters = (cluster_id < total_macro_tiles)
        ? ((total_macro_tiles - 1 - cluster_id) / num_clusters + 1)
        : 0;

    __shared__ volatile int slot_ready[PIPE_DEPTH];
    __shared__ int slot_macro_id[PIPE_DEPTH];
    __shared__ volatile int slot_prod_done[PIPE_DEPTH];
    __shared__ float slot_local_amax[PIPE_DEPTH];
    __shared__ volatile uint32_t slot_local_ready[PIPE_DEPTH];
    __shared__ float slot_combined_amax[PIPE_DEPTH];
    __shared__ float slot_producer_amax[PIPE_DEPTH][BUFFS_NUM_IN];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_macro_id[s] = -1;
            slot_prod_done[s] = 0;
            slot_local_amax[s] = 0.0f;
            slot_local_ready[s] = 0;
            slot_combined_amax[s] = 0.0f;
            #pragma unroll
            for (int w = 0; w < BUFFS_NUM_IN; ++w) {
                slot_producer_amax[s][w] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (is_producer) {
        for (int fill_iter = 0; fill_iter < cluster_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            if (producer_leader) {
                while (slot_ready[slot] != 0) {
                    __nanosleep(64);
                }
                slot_macro_id[slot] = cluster_id + fill_iter * num_clusters;
                slot_local_ready[slot] = 0;
                slot_prod_done[slot] = 0;
                #pragma unroll
                for (int w = 0; w < BUFFS_NUM_IN; ++w) {
                    slot_producer_amax[slot][w] = 0.0f;
                }
                __threadfence_block();
                slot_ready[slot] = -1;
            }
            while (slot_ready[slot] != -1) {
                __nanosleep(16);
            }

            const int macro_tile_id = cluster_id + fill_iter * num_clusters;
            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            int local_ctaid_X = ctaid_X;
            bool output_dh1 = true;
            if (ctaid_X >= split0_tiles) {
                output_dh1 = false;
                local_ctaid_X -= split0_tiles;
            }
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int input_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            float producer_local_amax = 0.0f;
            if (producer_warp < ACTIVE_PRODUCER_WARPS && cta_active) {
                for (int t = producer_warp; t < NUM_TILES_PER_CHUNK; t += ACTIVE_PRODUCER_WARPS) {
                    float tile_amax;
                    if (output_dh1) {
                        tile_amax = transform_store_silu_deriv_split2_tile_warp<true>(
                            &sIn_ring[slot][0][0][0],
                            dh, h3, h1_raw,
                            static_cast<int>(rows), static_cast<int>(split_cols),
                            block_offset_Y, input_block_offset_X,
                            t, producer_lane);
                    } else {
                        tile_amax = transform_store_silu_deriv_split2_tile_warp<false>(
                            &sIn_ring[slot][0][0][0],
                            dh, h3, h1_raw,
                            static_cast<int>(rows), static_cast<int>(split_cols),
                            block_offset_Y, input_block_offset_X,
                            t, producer_lane);
                    }
                    producer_local_amax = fmaxf(producer_local_amax, tile_amax);
                }
            }
            if constexpr (DELAYED_SCALING) {
                if (producer_warp < ACTIVE_PRODUCER_WARPS) {
                    #pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        producer_local_amax = fmaxf(
                            producer_local_amax,
                            __shfl_xor_sync(0xffffffff, producer_local_amax, mask));
                    }
                    if (producer_lane == 0) {
                        slot_producer_amax[slot][producer_warp] = producer_local_amax;
                    }
                }
            }

            if (producer_warp < ACTIVE_PRODUCER_WARPS) {
                __syncwarp();
                __threadfence_block();
            }
            if (producer_warp < ACTIVE_PRODUCER_WARPS && producer_lane == 0) {
                atomicAdd((int*)&slot_prod_done[slot], 1);
            }

            if (producer_leader) {
                while (slot_prod_done[slot] < ACTIVE_PRODUCER_WARPS) {
                    __nanosleep(16);
                }
                if constexpr (DELAYED_SCALING) {
                    float slot_amax = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < ACTIVE_PRODUCER_WARPS; ++w) {
                        slot_amax = fmaxf(slot_amax, slot_producer_amax[slot][w]);
                    }
                    slot_local_amax[slot] = slot_amax;
                    slot_local_ready[slot] = 1;
                }
                __threadfence_block();
                slot_ready[slot] = 1;
            }
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < cluster_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 1) {
                __nanosleep(64);
            }

            const int macro_tile_id = slot_macro_id[slot];
            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;

            if (cta_active) {
                const bool second_split = ctaid_X >= split0_tiles;
                const int local_ctaid_X = second_split ? (ctaid_X - split0_tiles) : ctaid_X;
                const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int output_block_offset_X = local_ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int input_block_offset_X = output_block_offset_X;
                const int block_offset_Y_tr = output_block_offset_X;
                const int block_offset_X_tr = block_offset_Y;
                const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
                const int chunk_cols = static_cast<int>(split_cols) - input_block_offset_X;

                float amax_val = slot_local_amax[slot];
                if constexpr (!DELAYED_SCALING) {
                    float cta_max = 0.0f;
                    #pragma unroll
                    for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                        cta_max = fmaxf(
                            cta_max,
                            scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
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

                    amax_val = slot_local_amax[slot];
                    if constexpr (SHARED_AMAX) {
                        if (consumer_leader) {
                            if (ctaid_Y + 1 >= tiles_Y) {
                                slot_combined_amax[slot] = slot_local_amax[slot];
                            } else {
                                while (cluster_load_shared_u32(
                                           (const uint32_t*)&slot_local_ready[slot],
                                           peer_cta) == 0u) {
                                    __nanosleep(64);
                                }
                                const float peer_amax = cluster_load_shared_f32(&slot_local_amax[slot], peer_cta);
                                slot_combined_amax[slot] = fmaxf(slot_local_amax[slot], peer_amax);
                            }
                        }
                        subgroup_barrier_sync<CONSUMER_THREADS>();
                        amax_val = slot_combined_amax[slot];
                    }
                } else if (consumer_leader) {
                    float* amax_out = second_split ? amax_out1 : amax_out0;
                    if (amax_out != nullptr) {
                        transformer_engine::atomicMaxFloat(amax_out, amax_val);
                    }
                }

                const float S_enc = DELAYED_SCALING ? 1.0f : compute_localcta_encode_scaling_factor_FP4(amax_val);
                const float sg_val = DELAYED_SCALING ? 1.0f : amax_val / localcta_global_scale_num();
                if (consumer_leader) {
                    if (second_split) {
                        row_sg_chunks1[ctaid_Y * split0_tiles + local_ctaid_X] = sg_val;
                        if constexpr (RETURN_TRANSPOSE) {
                            col_sg_chunks1[local_ctaid_X * tiles_Y + ctaid_Y] = sg_val;
                        }
                    } else {
                        row_sg_chunks0[ctaid_Y * split0_tiles + local_ctaid_X] = sg_val;
                        if constexpr (RETURN_TRANSPOSE) {
                            col_sg_chunks0[local_ctaid_X * tiles_Y + ctaid_Y] = sg_val;
                        }
                    }
                }

                if (second_split) {
                    quantize_store_raw_chunk_group<CONSUMER_THREADS, RETURN_TRANSPOSE, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0],
                        sOut_ptr, sOut_tr_ptr, sSFrowwise_ptr, sSFcolwise_ptr,
                        tensor_map_output1, tensor_map_output_t1,
                        tmap_scale_row1, tmap_scale_col1,
                        S_enc,
                        block_offset_Y, output_block_offset_X,
                        block_offset_Y_tr, block_offset_X_tr,
                        chunk_rows, chunk_cols, local_ctaid_X, ctaid_Y, consumer_tid);
                } else {
                    quantize_store_raw_chunk_group<CONSUMER_THREADS, RETURN_TRANSPOSE, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0],
                        sOut_ptr, sOut_tr_ptr, sSFrowwise_ptr, sSFcolwise_ptr,
                        tensor_map_output0, tensor_map_output_t0,
                        tmap_scale_row0, tmap_scale_col0,
                        S_enc,
                        block_offset_Y, output_block_offset_X,
                        block_offset_Y_tr, block_offset_X_tr,
                        chunk_rows, chunk_cols, local_ctaid_X, ctaid_Y, consumer_tid);
                }

                if (consumer_leader) {
                    slot_local_ready[slot] = 0;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
            }

            if (consumer_leader) {
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
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

template <int PIPE_DEPTH, bool RETURN_TRANSPOSE>
inline int prepared_split2_dual_1cta_tuned_shmem_size() {
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int input_ring_bytes = 2 * PIPE_DEPTH * slot_in_bytes;
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

template <int TOTAL_THREADS, int PIPE_DEPTH, bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true,
          bool WRITE_ROW_RAW = false, bool WRITE_COL_RAW = false>
__global__ void __launch_bounds__(TOTAL_THREADS)
fused_localcta_quantize_kernel_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_row_raw,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_raw,
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
            if constexpr (WRITE_ROW_RAW) {
                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (consumer_leader) {
                    tma_store_scales_2x512(
                        tmap_scale_row_raw, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                }
            }
            if constexpr (!WRITE_ROW_RAW) {
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
                if constexpr (WRITE_COL_RAW) {
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    if (consumer_leader) {
                        tma_store_scales_2x512(
                            tmap_scale_col_raw, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }
                if constexpr (!WRITE_COL_RAW) {
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

template <int TOTAL_THREADS, int PIPE_DEPTH, bool RETURN_TRANSPOSE,
          bool ENCODE_CENTRIC = true, bool WRITE_COL_RAW = false,
          bool DATA_SR = false, bool FAST_DATA_SR = false, bool SCALE_SR = false,
          bool ROW_WITH_RHT = false, bool COL_WITH_RHT = false,
          bool COL_RHT_AMAX_FROM_RAW = false,
          bool WITH_RANDOM_SIGN_MASK = false>
__global__ void __launch_bounds__(TOTAL_THREADS)
fused_localcta_sqrelu_quantize_kernel_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_raw,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int total_tiles,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    float col_rht_raw_amax_multiplier
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS > 128 && TOTAL_THREADS <= 256 && TOTAL_THREADS % 32 == 0,
                  "tuned square-ReLU 1CTA prepared kernel expects 128 consumer threads plus producer warp(s)");

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
    __shared__ float slot_amax_row[PIPE_DEPTH];
    __shared__ float slot_amax_col[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_tile_id[s] = -1;
            slot_amax_row[s] = 0.0f;
            slot_amax_col[s] = 0.0f;
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

            float row_cta_max = 0.0f;
            float col_cta_max = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[slot][t], slot_phase[slot]);
                const float raw_tile_max =
                    transform_sqrelu_tile_inplace_amax_group<CONSUMER_THREADS>(
                        &sIn_ring[slot][0][0][0], t, consumer_tid);
                if constexpr (ROW_WITH_RHT) {
                    LocalCTARNGState row_rng;
                    if constexpr (WITH_RANDOM_SIGN_MASK) {
                        row_rng.init(
                            rng_seed,
                            rng_subsequence_base +
                                ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 0ull) *
                                    CONSUMER_THREADS +
                                static_cast<uint64_t>(consumer_tid),
                            0);
                    }
                    uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                    int row_rnd_idx = 4;
                    row_cta_max = fmaxf(
                        row_cta_max,
                        scan_tile_amax_row_opt<true, WITH_RANDOM_SIGN_MASK>(
                            &sIn_ring[slot][0][0][0], t, row_rng, row_random_uint4, row_rnd_idx));
                } else {
                    row_cta_max = fmaxf(row_cta_max, raw_tile_max);
                }
                if constexpr (RETURN_TRANSPOSE) {
                    if constexpr (COL_WITH_RHT && !COL_RHT_AMAX_FROM_RAW) {
                        LocalCTARNGState col_rng;
                        if constexpr (WITH_RANDOM_SIGN_MASK) {
                            col_rng.init(
                                rng_seed,
                                rng_subsequence_base +
                                    ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 1ull) *
                                        CONSUMER_THREADS +
                                    static_cast<uint64_t>(consumer_tid),
                                0);
                        }
                        uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                        int col_rnd_idx = 4;
                        col_cta_max = fmaxf(
                            col_cta_max,
                            scan_tile_amax_col_opt<true, WITH_RANDOM_SIGN_MASK>(
                                &sIn_ring[slot][0][0][0], t, col_rng, col_random_uint4, col_rnd_idx));
                    } else {
                        const float col_tile_max =
                            (COL_WITH_RHT && COL_RHT_AMAX_FROM_RAW)
                                ? raw_tile_max * col_rht_raw_amax_multiplier
                                : raw_tile_max;
                        col_cta_max = fmaxf(col_cta_max, col_tile_max);
                    }
                }
            }

            const int lane = consumer_tid % 32;
            const int wid = consumer_tid / 32;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                row_cta_max = fmaxf(row_cta_max, __shfl_xor_sync(0xffffffff, row_cta_max, mask));
                col_cta_max = fmaxf(col_cta_max, __shfl_xor_sync(0xffffffff, col_cta_max, mask));
            }
            if (lane == 0) {
                warp_max[wid] = row_cta_max;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if (wid == 0) {
                row_cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                #pragma unroll
                for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    row_cta_max = fmaxf(row_cta_max, __shfl_xor_sync(0xffffffff, row_cta_max, mask));
                }
                if (lane == 0) {
                    slot_amax_row[slot] = row_cta_max;
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] =
                        row_cta_max / localcta_global_scale_num();
                }
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if constexpr (RETURN_TRANSPOSE) {
                if (lane == 0) {
                    warp_max[wid] = col_cta_max;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (wid == 0) {
                    col_cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                    #pragma unroll
                    for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                        col_cta_max = fmaxf(col_cta_max, __shfl_xor_sync(0xffffffff, col_cta_max, mask));
                    }
                    if (lane == 0) {
                        slot_amax_col[slot] = col_cta_max;
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] =
                            col_cta_max / localcta_global_scale_num();
                    }
                }
            } else if (consumer_leader) {
                slot_amax_col[slot] = slot_amax_row[slot];
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            const float row_amax_val = slot_amax_row[slot];
            const float col_amax_val = slot_amax_col[slot];
            const float S_enc_row = compute_localcta_encode_scaling_factor_FP4(row_amax_val);
            const float S_enc_col = compute_localcta_encode_scaling_factor_FP4(col_amax_val);
            const float row_sg_val = row_amax_val / localcta_global_scale_num();
            const float col_sg_val = col_amax_val / localcta_global_scale_num();

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

                const uint64_t row_rng_subsequence =
                    rng_subsequence_base +
                        ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 0ull) *
                            CONSUMER_THREADS +
                        static_cast<uint64_t>(consumer_tid);
                LocalCTARNGState row_rng;
                if constexpr ((DATA_SR && !FAST_DATA_SR) || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
                    row_rng.init(rng_seed, row_rng_subsequence, 0);
                }
                uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                int row_rnd_idx = 4;
                const uint64_t row_fast_sr_base =
                    rng_seed ^ row_rng_subsequence ^ 0xd1342543de82ef95ull;
                rowwise_scaling_opt<
                    ENCODE_CENTRIC, false, DATA_SR, FAST_DATA_SR, SCALE_SR,
                    ROW_WITH_RHT, WITH_RANDOM_SIGN_MASK>(
                    &sIn_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out,
                    row_rng, row_random_uint4, row_rnd_idx, row_fast_sr_base);

                if constexpr (RETURN_TRANSPOSE) {
                    const uint64_t col_rng_subsequence =
                        rng_subsequence_base +
                            ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 1ull) *
                                CONSUMER_THREADS +
                            static_cast<uint64_t>(consumer_tid);
                    LocalCTARNGState col_rng;
                    if constexpr ((DATA_SR && !FAST_DATA_SR) || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
                        col_rng.init(rng_seed, col_rng_subsequence, 0);
                    }
                    uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                    int col_rnd_idx = 4;
                    const uint64_t col_fast_sr_base =
                        rng_seed ^ col_rng_subsequence ^ 0x94d049bb133111ebull;
                    colwise_scaling_opt<
                        ENCODE_CENTRIC, false, DATA_SR, FAST_DATA_SR,
                        SCALE_SR, COL_WITH_RHT, WITH_RANDOM_SIGN_MASK>(
                        &sIn_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                        S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                        col_rng, col_random_uint4, col_rnd_idx, col_fast_sr_base);
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
                row_sg_val,
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
                if constexpr (WRITE_COL_RAW) {
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    if (consumer_leader) {
                        tma_store_scales_2x512(
                            tmap_scale_col_raw, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }
                if constexpr (!WRITE_COL_RAW) {
                    scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                        sSFcolwise_ptr,
                        LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                        col_sg_val,
                        consumer_tid);
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    if (consumer_leader) {
                        tma_store_scales_2x512(
                            tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
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

template <int TOTAL_THREADS, int PIPE_DEPTH, bool RETURN_TRANSPOSE,
          bool ENCODE_CENTRIC = true,
          bool WRITE_ROW_RAW = false, bool WRITE_COL_RAW = false,
          bool DATA_SR = false, bool FAST_DATA_SR = false, bool SCALE_SR = false,
          bool ROW_WITH_RHT = false, bool COL_WITH_RHT = false,
          bool COL_RHT_AMAX_FROM_RAW = false,
          bool WITH_RANDOM_SIGN_MASK = false>
__global__ void __launch_bounds__(TOTAL_THREADS)
fused_localcta_sqrelu_deriv_quantize_kernel_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_dh,
    const __grid_constant__ CUtensorMap tensor_map_h1,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_row_raw,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_raw,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int total_tiles,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    float col_rht_raw_amax_multiplier
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS > 128 && TOTAL_THREADS <= 256 && TOTAL_THREADS % 32 == 0,
                  "tuned square-ReLU deriv 1CTA prepared kernel expects 128 consumer threads plus producer warp(s)");

    constexpr int CONSUMER_THREADS = 128;
    constexpr int NUM_TILES_PER_CHUNK = BUFFS_NUM_IN;
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
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sDh_ring_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sH1_ring_ptr = reinterpret_cast<IType*>(dshmem + input_ring_bytes);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * input_ring_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + 2 * input_ring_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * input_ring_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + 2 * input_ring_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sDh_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sDh_ring_ptr);
    auto& sH1_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sH1_ring_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t dh_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ uint64_t h1_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ int slot_ready[PIPE_DEPTH];
    __shared__ int slot_phase[PIPE_DEPTH];
    __shared__ int slot_tile_id[PIPE_DEPTH];
    __shared__ float slot_amax_row[PIPE_DEPTH];
    __shared__ float slot_amax_col[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_tile_id[s] = -1;
            slot_amax_row[s] = 0.0f;
            slot_amax_col[s] = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_init(&dh_mbar[s][t], 1);
                ptx::mbarrier_init(&h1_mbar[s][t], 1);
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
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[slot][t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh_ring[slot][t]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_dh),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &dh_mbar[slot][t]);
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[slot][t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1_ring[slot][t]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &h1_mbar[slot][t]);
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

            float row_cta_max = 0.0f;
            float col_cta_max = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[slot][t], slot_phase[slot]);
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[slot][t], slot_phase[slot]);
                const float raw_tile_max =
                    transform_sqrelu_deriv_tile_inplace_amax_group<CONSUMER_THREADS>(
                        &sDh_ring[slot][0][0][0],
                        &sH1_ring[slot][0][0][0],
                        t,
                        consumer_tid);
                if constexpr (ROW_WITH_RHT) {
                    LocalCTARNGState row_rng;
                    if constexpr (WITH_RANDOM_SIGN_MASK) {
                        row_rng.init(
                            rng_seed,
                            rng_subsequence_base +
                                ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 0ull) *
                                    CONSUMER_THREADS +
                                static_cast<uint64_t>(consumer_tid),
                            0);
                    }
                    uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                    int row_rnd_idx = 4;
                    row_cta_max = fmaxf(
                        row_cta_max,
                        scan_tile_amax_row_opt<true, WITH_RANDOM_SIGN_MASK>(
                            &sDh_ring[slot][0][0][0], t, row_rng, row_random_uint4, row_rnd_idx));
                } else {
                    row_cta_max = fmaxf(row_cta_max, raw_tile_max);
                }
                if constexpr (RETURN_TRANSPOSE) {
                    if constexpr (COL_WITH_RHT && !COL_RHT_AMAX_FROM_RAW) {
                        LocalCTARNGState col_rng;
                        if constexpr (WITH_RANDOM_SIGN_MASK) {
                            col_rng.init(
                                rng_seed,
                                rng_subsequence_base +
                                    ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 1ull) *
                                        CONSUMER_THREADS +
                                    static_cast<uint64_t>(consumer_tid),
                                0);
                        }
                        uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                        int col_rnd_idx = 4;
                        col_cta_max = fmaxf(
                            col_cta_max,
                            scan_tile_amax_col_opt<true, WITH_RANDOM_SIGN_MASK>(
                                &sDh_ring[slot][0][0][0], t, col_rng, col_random_uint4, col_rnd_idx));
                    } else {
                        const float col_tile_max =
                            (COL_WITH_RHT && COL_RHT_AMAX_FROM_RAW)
                                ? raw_tile_max * col_rht_raw_amax_multiplier
                                : raw_tile_max;
                        col_cta_max = fmaxf(col_cta_max, col_tile_max);
                    }
                }
            }

            const int lane = consumer_tid % 32;
            const int wid = consumer_tid / 32;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                row_cta_max = fmaxf(row_cta_max, __shfl_xor_sync(0xffffffff, row_cta_max, mask));
                col_cta_max = fmaxf(col_cta_max, __shfl_xor_sync(0xffffffff, col_cta_max, mask));
            }
            if (lane == 0) {
                warp_max[wid] = row_cta_max;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if (wid == 0) {
                row_cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                #pragma unroll
                for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    row_cta_max = fmaxf(row_cta_max, __shfl_xor_sync(0xffffffff, row_cta_max, mask));
                }
                if (lane == 0) {
                    slot_amax_row[slot] = row_cta_max;
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] =
                        row_cta_max / localcta_global_scale_num();
                }
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if constexpr (RETURN_TRANSPOSE) {
                if (lane == 0) {
                    warp_max[wid] = col_cta_max;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (wid == 0) {
                    col_cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                    #pragma unroll
                    for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                        col_cta_max = fmaxf(col_cta_max, __shfl_xor_sync(0xffffffff, col_cta_max, mask));
                    }
                    if (lane == 0) {
                        slot_amax_col[slot] = col_cta_max;
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] =
                            col_cta_max / localcta_global_scale_num();
                    }
                }
            } else if (consumer_leader) {
                slot_amax_col[slot] = slot_amax_row[slot];
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            const float row_amax_val = slot_amax_row[slot];
            const float col_amax_val = slot_amax_col[slot];
            const float S_enc_row = compute_localcta_encode_scaling_factor_FP4(row_amax_val);
            const float S_enc_col = compute_localcta_encode_scaling_factor_FP4(col_amax_val);
            const float row_sg_val = row_amax_val / localcta_global_scale_num();
            const float col_sg_val = col_amax_val / localcta_global_scale_num();

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

                const uint64_t row_rng_subsequence =
                    rng_subsequence_base +
                        ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 0ull) *
                            CONSUMER_THREADS +
                        static_cast<uint64_t>(consumer_tid);
                LocalCTARNGState row_rng;
                if constexpr ((DATA_SR && !FAST_DATA_SR) || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
                    row_rng.init(rng_seed, row_rng_subsequence, 0);
                }
                uint4 row_random_uint4 = make_uint4(0, 0, 0, 0);
                int row_rnd_idx = 4;
                const uint64_t row_fast_sr_base =
                    rng_seed ^ row_rng_subsequence ^ 0xd1342543de82ef95ull;
                rowwise_scaling_opt<
                    ENCODE_CENTRIC, false, DATA_SR, FAST_DATA_SR, SCALE_SR,
                    ROW_WITH_RHT, WITH_RANDOM_SIGN_MASK>(
                    &sDh_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                    S_enc_row, stage_Y, stage_X, t, buff_out,
                    row_rng, row_random_uint4, row_rnd_idx, row_fast_sr_base);

                if constexpr (RETURN_TRANSPOSE) {
                    const uint64_t col_rng_subsequence =
                        rng_subsequence_base +
                            ((static_cast<uint64_t>(tile_id) * NUM_TILES_PER_CHUNK + t) * 2ull + 1ull) *
                                CONSUMER_THREADS +
                            static_cast<uint64_t>(consumer_tid);
                    LocalCTARNGState col_rng;
                    if constexpr ((DATA_SR && !FAST_DATA_SR) || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
                        col_rng.init(rng_seed, col_rng_subsequence, 0);
                    }
                    uint4 col_random_uint4 = make_uint4(0, 0, 0, 0);
                    int col_rnd_idx = 4;
                    const uint64_t col_fast_sr_base =
                        rng_seed ^ col_rng_subsequence ^ 0x94d049bb133111ebull;
                    colwise_scaling_opt<
                        ENCODE_CENTRIC, false, DATA_SR, FAST_DATA_SR,
                        SCALE_SR, COL_WITH_RHT, WITH_RANDOM_SIGN_MASK>(
                        &sDh_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                        S_enc_col, stage_Y, stage_X, t, buff_out_tr,
                        col_rng, col_random_uint4, col_rnd_idx, col_fast_sr_base);
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
            if constexpr (WRITE_ROW_RAW) {
                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (consumer_leader) {
                    tma_store_scales_2x512(
                        tmap_scale_row_raw, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                }
            }
            if constexpr (!WRITE_ROW_RAW) {
                scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                    sSFrowwise_ptr,
                    LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                    row_sg_val,
                    consumer_tid);
                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (consumer_leader) {
                    tma_store_scales_2x512(
                        tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
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
                if constexpr (WRITE_COL_RAW) {
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    if (consumer_leader) {
                        tma_store_scales_2x512(
                            tmap_scale_col_raw, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }
                if constexpr (!WRITE_COL_RAW) {
                    scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                        sSFcolwise_ptr,
                        LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                        col_sg_val,
                        consumer_tid);
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    if (consumer_leader) {
                        tma_store_scales_2x512(
                            tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
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
                ptx::mbarrier_invalid(&dh_mbar[s][t]);
                ptx::mbarrier_invalid(&h1_mbar[s][t]);
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

template <bool RETURN_TRANSPOSE>
inline int silu_raw_shmem_size() {
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

    return 2 * in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <bool RETURN_TRANSPOSE, int H3_RING_TILES>
inline int silu_raw_h3_ring_shmem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int h3_bytes = DIVUP_TO_MULTIPLE(
        H3_RING_TILES * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
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

    return in_bytes + h3_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace tk_localcta

#endif  // TK_LOCALCTA_FUSED_QUANTIZE_CUH_
