#include "../../ThunderKittens/include/kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <random>
#include <vector>

#include "mxfp4_v3_quantize.cuh"

namespace te_style_tcbench {
using namespace mxfp4_v3;

int bench_iters() {
    const char* env = std::getenv("TCBENCH_ITERS");
    return env ? std::max(1, std::atoi(env)) : 10;
}

template<typename LaunchFn>
float time_launch_ms(LaunchFn&& launch, const char* start_name, const char* stop_name, const char* sync_name, const char* launch_name) {
    auto cuda_check_local = [](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            std::cerr << what << ": " << cudaGetErrorString(err) << std::endl;
            std::exit(1);
        }
    };
    cudaEvent_t start_ev, stop_ev;
    cuda_check_local(cudaEventCreate(&start_ev), start_name);
    cuda_check_local(cudaEventCreate(&stop_ev), stop_name);
    cuda_check_local(cudaEventRecord(start_ev), "event record start");
    const int iters = bench_iters();
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    cuda_check_local(cudaEventRecord(stop_ev), "event record stop");
    cuda_check_local(cudaEventSynchronize(stop_ev), sync_name);
    cuda_check_local(cudaGetLastError(), launch_name);
    float ms = 0.0f;
    cuda_check_local(cudaEventElapsedTime(&ms, start_ev, stop_ev), "event elapsed");
    cuda_check_local(cudaEventDestroy(start_ev), "destroy start event");
    cuda_check_local(cudaEventDestroy(stop_ev), "destroy stop event");
    return ms / iters;
}

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 32;
static constexpr int TILE_N_TE = 16;
static constexpr int CHUNK_M = 128;
static constexpr int CHUNK_K_STAGE = 64;
static constexpr int CHUNK_K = 128;
static constexpr int CHUNK_K_WIDE = 512;
static constexpr int CHUNK_K_PERSIST = 512;
static constexpr int PIPE64_STAGES = CHUNK_K_WIDE / CHUNK_K_STAGE;
static constexpr int TE_COPY16_PIPE_STAGES = CHUNK_K_WIDE / CHUNK_K;
static constexpr int TE_COPY16_ACCUM_PIPE = TE_COPY16_PIPE_STAGES;
static constexpr int LOAD_PIPE_STAGES = 2;
static constexpr int NUM_WARPS = 8;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
static constexpr int TE_COPY16_NUM_WARPS = 16;
static constexpr int TE_COPY16_NUM_THREADS = TE_COPY16_NUM_WARPS * WARP_THREADS;
static constexpr int TE_COPY16_EPILOGUE_WARPS = 8;
static constexpr int TE_COPY16_STORE_GROUP = 4;
static constexpr int TE_COPY16_COL_PRODUCER_WARPS = 4;
static constexpr int TE_COPY16_COL_PRODUCER_PIPE = 4;

#if !defined(TCBENCH_ROWCOL_DISABLE_BLOCK_PAIR64_STORE)
#define TCBENCH_ROWCOL_BLOCK_PAIR64_STORE 1
#endif

#ifndef TCBENCH_ROWCOL_ROW_REGS
#define TCBENCH_ROWCOL_ROW_REGS 144
#endif
#ifndef TCBENCH_ROWCOL_COL_REGS
#define TCBENCH_ROWCOL_COL_REGS 112
#endif

using a128_tile = st_bf<CHUNK_M, CHUNK_K>;
using a64_tile = st_bf<CHUNK_M, CHUNK_K_STAGE>;
using a512_tile = st_bf<CHUNK_M, CHUNK_K_WIDE>;
using h32_tile = st_bf<TILE_N, TILE_N>;
using h16_tile = st_bf<TILE_N_TE, TILE_N_TE>;
using d32_tile = st_bf<TILE_M, TILE_N>;
using d16_tile = st_bf<16, TILE_N>;
using d16x16_tile = st_bf<16, TILE_N_TE>;

using a128_gl = gl<bf16, 1, 1, -1, -1, a128_tile>;
using a64_gl = gl<bf16, 1, 1, -1, -1, a64_tile>;
using a512_gl = gl<bf16, 1, 1, -1, -1, a512_tile>;
using h32_gl = gl<bf16, 1, 1, 32, 32, h32_tile>;
using h16_gl = gl<bf16, 1, 1, 16, 16, h16_tile>;
using full_accum_tt = full_tt_fl<128>;
using full_accum_tt_stage = full_tt_fl<CHUNK_K_STAGE>;
using full_accum_tt_wide = full_tt_fl<512>;
using full_accum_tt_persist = full_tt_fl<CHUNK_K_PERSIST>;
using block_accum_tt = half_tt_fl<32>;
using block_accum16_tt = half_tt_fl<16>;
using full_accum16_tt = full_tt_fl<16>;
using lane_accum_tt = tt_fl<16, 32>;
using lane_accum_rt = rt_fl<16, 32>;
using lane_accum16_tt = tt_fl<16, 16>;
using lane_accum16_rt = rt_fl<16, 16>;

template<QuantMode MODE>
__device__ __forceinline__ uint8_t float_to_e8m0_dispatch(float val) {
    return float_to_e8m0<MODE>(val);
}

__device__ __forceinline__ uint8_t float_to_e8m0_rp_ptx(float val) {
    uint16_t out;
    asm volatile(
        "{\n\t"
        "cvt.rp.satfinite.ue8m0x2.f32 %0, 0.0, %1;\n\t"
        "}\n"
        : "=h"(out)
        : "f"(val));
    return static_cast<uint8_t>(out & 0xffu);
}

template<bool SCALE_SR>
__device__ __forceinline__ uint8_t quantize_scale(
    float block_amax,
    RNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if constexpr (SCALE_SR) {
        return float_to_e8m0_stochastic(block_amax, next_rbits(rng, random_uint4, rnd_idx));
    } else {
        return float_to_e8m0_dispatch<QuantMode::RTE>(block_amax);
    }
}

template<bool DATA_SR>
__device__ __forceinline__ uint32_t maybe_next_data_rbits(
    RNGState& rng,
    uint4& random_uint4,
    int& rnd_idx
) {
    if constexpr (DATA_SR) {
        return next_rbits(rng, random_uint4, rnd_idx);
    } else {
        return 0u;
    }
}

__device__ __forceinline__ uint32_t mix_rht_sign_bits(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= x >> 31;
    return static_cast<uint32_t>(x) ^ static_cast<uint32_t>(x >> 32);
}

template<bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ uint32_t make_rht_sign_bits_fast(
    uint64_t rng_seed,
    uint64_t row_subsequence
) {
    if constexpr (!WITH_RANDOM_SIGN_MASK) {
        return 0xffffffffu;
    } else {
        return mix_rht_sign_bits(rng_seed ^ (row_subsequence * 0xd1342543de82ef95ull));
    }
}

template<bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ float apply_rht_sign_fast(float val, uint32_t sign_bits, int col) {
    if constexpr (!WITH_RANDOM_SIGN_MASK) {
        return val;
    } else {
        const uint32_t flip = (((sign_bits >> col) & 1u) ^ 1u) << 31;
        return __uint_as_float(__float_as_uint(val) ^ flip);
    }
}

template<bool DATA_SR, bool SCALE_SR>
__device__ __forceinline__ void quantize_tile_64x32(
    const d32_tile& tile,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int cols,
    int tile_row,
    int tile_col_block,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    const int tid = threadIdx.x & 127;
    const int row = tid >> 1;
    const int half = tid & 1;
    if (row >= TILE_M) {
        return;
    }

    float vals[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __bfloat162float(tile[{row, half * 16 + i}]);
    }

    float my_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        my_amax = fmaxf(my_amax, fabsf(vals[i]));
    }
    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);

    RNGState rng;
    const uint64_t block_linear = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    rng.init(rng_seed, rng_subsequence_base + block_linear * TILE_M + row, 0);
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;

    const uint8_t e8m0_val = quantize_scale<SCALE_SR>(block_amax, rng, random_uint4, rnd_idx);
    if (half == 0) {
        scales_out[(tile_row + row) * (cols / 32) + tile_col_block] = e8m0_val;
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);

    uint16_t packed16[4];
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const float2 in01 = make_float2(vals[pack * 4 + 0], vals[pack * 4 + 1]);
        const float2 in23 = make_float2(vals[pack * 4 + 2], vals[pack * 4 + 3]);
        const auto packed = mul_cvt_fp32_to_fp4_4x<DATA_SR>(
            in01,
            in23,
            scale,
            maybe_next_data_rbits<DATA_SR>(rng, random_uint4, rnd_idx)
        );
        packed16[pack] = *reinterpret_cast<const uint16_t*>(&packed);
    }

    const uint32_t lo_word = static_cast<uint32_t>(packed16[0]) |
                             (static_cast<uint32_t>(packed16[1]) << 16);
    const uint32_t hi_word = static_cast<uint32_t>(packed16[2]) |
                             (static_cast<uint32_t>(packed16[3]) << 16);
    const uint32_t peer_lo_word = __shfl_xor_sync(0xffffffff, lo_word, 1);
    const uint32_t peer_hi_word = __shfl_xor_sync(0xffffffff, hi_word, 1);
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out + static_cast<int64_t>(tile_row + row) * (cols / 2) + tile_col_block * 16;
        *reinterpret_cast<uint4*>(row_ptr) = make_uint4(lo_word, hi_word, peer_lo_word, peer_hi_word);
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
__device__ __forceinline__ void quantize_tile_16x32(
    const d16_tile& tile,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int cols,
    int tile_row,
    int tile_col_block,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;

    RNGState rng;
    const uint64_t block_linear = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const uint64_t row_subsequence = rng_subsequence_base + block_linear * 16 + row;
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;
    uint32_t sign_bits = 0xffffffffu;
    if constexpr (WITH_RANDOM_SIGN_MASK && !DATA_SR && !SCALE_SR) {
        sign_bits = make_rht_sign_bits_fast<true>(rng_seed, row_subsequence);
    } else {
        if constexpr (WITH_RANDOM_SIGN_MASK || DATA_SR || SCALE_SR) {
            rng.init(rng_seed, row_subsequence, 0);
        }
        sign_bits = make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
    }

    float vals[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = half * 16 + i;
        vals[i] = __bfloat162float(tile[{row, col}]);
        if constexpr (WITH_RANDOM_SIGN_MASK && !DATA_SR && !SCALE_SR) {
            vals[i] = apply_rht_sign_fast<true>(vals[i], sign_bits, col);
        } else {
            vals[i] *= ((sign_bits >> col) & 1u) ? 1.0f : -1.0f;
        }
    }

    float my_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        my_amax = fmaxf(my_amax, fabsf(vals[i]));
    }
    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);

    const uint8_t e8m0_val = quantize_scale<SCALE_SR>(block_amax, rng, random_uint4, rnd_idx);
    if (half == 0 && scales_out != nullptr) {
        scales_out[(tile_row + row) * (cols / 32) + tile_col_block] = e8m0_val;
        if constexpr (WITH_AMAX) {
            amax_out[(tile_row + row) * (cols / 32) + tile_col_block] = block_amax;
        }
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);

    uint16_t packed16[4];
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const float2 in01 = make_float2(vals[pack * 4 + 0], vals[pack * 4 + 1]);
        const float2 in23 = make_float2(vals[pack * 4 + 2], vals[pack * 4 + 3]);
        const auto packed = mul_cvt_fp32_to_fp4_4x<DATA_SR>(
            in01,
            in23,
            scale,
            maybe_next_data_rbits<DATA_SR>(rng, random_uint4, rnd_idx)
        );
        packed16[pack] = *reinterpret_cast<const uint16_t*>(&packed);
    }

    const uint32_t lo_word = static_cast<uint32_t>(packed16[0]) |
                             (static_cast<uint32_t>(packed16[1]) << 16);
    const uint32_t hi_word = static_cast<uint32_t>(packed16[2]) |
                             (static_cast<uint32_t>(packed16[3]) << 16);
    const uint32_t peer_lo_word = __shfl_xor_sync(0xffffffff, lo_word, 1);
    const uint32_t peer_hi_word = __shfl_xor_sync(0xffffffff, hi_word, 1);
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out + static_cast<int64_t>(tile_row + row) * (cols / 2) + tile_col_block * 16;
        *reinterpret_cast<uint4*>(row_ptr) = make_uint4(lo_word, hi_word, peer_lo_word, peer_hi_word);
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ void quantize_tile_16x16_tecopy(
    const d16x16_tile& tile,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int cols,
    int rows,
    int tile_row,
    int tile_col_block16,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;

    RNGState rng;
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;
    uint32_t sign_bits = 0xffffffffu;
    if constexpr (WITH_RANDOM_SIGN_MASK || DATA_SR || SCALE_SR) {
        const uint64_t block_linear = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        const uint64_t row_subsequence = rng_subsequence_base + block_linear * 16 + row;
        if constexpr (WITH_RANDOM_SIGN_MASK && !DATA_SR && !SCALE_SR) {
            uint32_t generated_sign_bits = 0u;
            if (half == 0) {
                rng.init(rng_seed, row_subsequence, 0);
                generated_sign_bits = make_rht_sign_bits<true>(rng, random_uint4, rnd_idx);
            }
            const uint32_t peer_sign_bits = __shfl_xor_sync(0xffffffffu, generated_sign_bits, 1);
            sign_bits = half == 0 ? generated_sign_bits : peer_sign_bits;
        } else {
            rng.init(rng_seed, row_subsequence, 0);
            sign_bits = make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
        }
    }

    float vals[8];
    float my_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = half * 8 + i;
        vals[i] = __bfloat162float(tile[{row, col}]);
        if constexpr (WITH_RANDOM_SIGN_MASK) {
            vals[i] *= ((sign_bits >> col) & 1u) ? 1.0f : -1.0f;
        }
        my_amax = fmaxf(my_amax, fabsf(vals[i]));
    }
    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);

    const uint8_t e8m0_val = quantize_scale<SCALE_SR>(block_amax, rng, random_uint4, rnd_idx);
    if (half == 0 && scales_out != nullptr) {
        // TE-style scale layout: make adjacent rows contiguous for coalesced scale stores.
        scales_out[tile_col_block16 * rows + tile_row + row] = e8m0_val;
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);

    if constexpr (DATA_SR) {
        if (half == 1) {
            next_rbits(rng, random_uint4, rnd_idx);
            next_rbits(rng, random_uint4, rnd_idx);
        }
    }

    uint16_t packed16[2];
    #pragma unroll
    for (int pack = 0; pack < 2; ++pack) {
        const float2 in01 = make_float2(vals[pack * 4 + 0], vals[pack * 4 + 1]);
        const float2 in23 = make_float2(vals[pack * 4 + 2], vals[pack * 4 + 3]);
        const auto packed = mul_cvt_fp32_to_fp4_4x<DATA_SR>(
            in01,
            in23,
            scale,
            maybe_next_data_rbits<DATA_SR>(rng, random_uint4, rnd_idx)
        );
        packed16[pack] = *reinterpret_cast<const uint16_t*>(&packed);
    }

    const uint32_t word = static_cast<uint32_t>(packed16[0]) |
                          (static_cast<uint32_t>(packed16[1]) << 16);
    const uint32_t peer_word = __shfl_xor_sync(0xffffffff, word, 1);
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out + static_cast<int64_t>(tile_row + row) * (cols / 2) + tile_col_block16 * 8;
        *reinterpret_cast<uint2*>(row_ptr) = make_uint2(word, peer_word);
    }
}

template<bool SCALE_RP, bool WITH_AMAX, typename Tile>
__device__ __forceinline__ uint2 quantize_tile_16x16_tecopy_pack_scale(
    const Tile& tile,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int tile_row,
    int tile_col_block16
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;

    const uint32_t shared_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<bf16*>(&tile.data[0])));

    float vals[8];
    float my_amax = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        bf16_2 packed_bf;
        move<bf16_2>::lds(packed_bf, tile.idx(shared_addr, {row, half * 8 + pair * 2}));
        const float v0 = __bfloat162float(__low2bfloat16(packed_bf));
        const float v1 = __bfloat162float(__high2bfloat16(packed_bf));
        vals[pair * 2 + 0] = v0;
        vals[pair * 2 + 1] = v1;
        my_amax = fmaxf(my_amax, fabsf(v0));
        my_amax = fmaxf(my_amax, fabsf(v1));
    }

    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);
    uint8_t e8m0_val;
    if constexpr (SCALE_RP) {
        e8m0_val = float_to_e8m0_rp_ptx(block_amax);
    } else {
        e8m0_val = float_to_e8m0_dispatch<QuantMode::RTE>(block_amax);
    }
    if (half == 0 && scales_out != nullptr) {
        scales_out[tile_col_block16 * rows + tile_row + row] = e8m0_val;
    }
    if constexpr (WITH_AMAX) {
        if (half == 0 && amax_out != nullptr) {
            amax_out[tile_col_block16 * rows + tile_row + row] = block_amax;
        }
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);
    uint16_t packed16[2];
    #pragma unroll
    for (int pack = 0; pack < 2; ++pack) {
        const float2 in01 = make_float2(vals[pack * 4 + 0], vals[pack * 4 + 1]);
        const float2 in23 = make_float2(vals[pack * 4 + 2], vals[pack * 4 + 3]);
        const auto packed = mul_cvt_fp32_to_fp4_4x<false>(in01, in23, scale, 0u);
        packed16[pack] = *reinterpret_cast<const uint16_t*>(&packed);
    }

    const uint32_t word = static_cast<uint32_t>(packed16[0]) |
                          (static_cast<uint32_t>(packed16[1]) << 16);
    return make_uint2(word, __shfl_xor_sync(0xffffffff, word, 1));
}

template<bool SCALE_RP, bool WITH_AMAX, typename Tile0, typename Tile1>
__device__ __forceinline__ uint8_t quantize_tiles_16x32_tecopy_pack_scale(
    const Tile0& tile0,
    const Tile1& tile1,
    uint2& packed0,
    uint2& packed1,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int tile_row,
    int tile_col_block32,
    int ntk = 0,
    bool swizzled_scale = false
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    const uint32_t shared_addr0 = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<bf16*>(&tile0.data[0])));
    const uint32_t shared_addr1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<bf16*>(&tile1.data[0])));

    float vals0[8];
    float vals1[8];
    float my_amax = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        bf16_2 packed_bf0;
        bf16_2 packed_bf1;
        move<bf16_2>::lds(packed_bf0, tile0.idx(shared_addr0, {row, half * 8 + pair * 2}));
        move<bf16_2>::lds(packed_bf1, tile1.idx(shared_addr1, {row, half * 8 + pair * 2}));
        const float v00 = __bfloat162float(__low2bfloat16(packed_bf0));
        const float v01 = __bfloat162float(__high2bfloat16(packed_bf0));
        const float v10 = __bfloat162float(__low2bfloat16(packed_bf1));
        const float v11 = __bfloat162float(__high2bfloat16(packed_bf1));
        vals0[pair * 2 + 0] = v00;
        vals0[pair * 2 + 1] = v01;
        vals1[pair * 2 + 0] = v10;
        vals1[pair * 2 + 1] = v11;
        my_amax = fmaxf(my_amax, fabsf(v00));
        my_amax = fmaxf(my_amax, fabsf(v01));
        my_amax = fmaxf(my_amax, fabsf(v10));
        my_amax = fmaxf(my_amax, fabsf(v11));
    }

    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);
    uint8_t e8m0_val;
    if constexpr (SCALE_RP) {
        e8m0_val = float_to_e8m0_rp_ptx(block_amax);
    } else {
        e8m0_val = float_to_e8m0_dispatch<QuantMode::RTE>(block_amax);
    }
    if (half == 0 && scales_out != nullptr) {
        const int global_row = tile_row + row;
        if (swizzled_scale) {
            const int ctaid_X = tile_col_block32 / 4;
            const int scale_block = tile_col_block32 & 3;
            const int row_in_chunk = global_row & 127;
            const int j = row_in_chunk & 31;
            const int grp = row_in_chunk >> 5;
            const int ctaid_Y = global_row >> 7;
            scales_out[(ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4 + scale_block] = e8m0_val;
        } else {
            scales_out[tile_col_block32 * rows + global_row] = e8m0_val;
        }
    }
    if constexpr (WITH_AMAX) {
        if (half == 0 && amax_out != nullptr) {
            amax_out[tile_col_block32 * rows + tile_row + row] = block_amax;
        }
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);
    uint16_t packed16_0[2];
    uint16_t packed16_1[2];
    #pragma unroll
    for (int pack = 0; pack < 2; ++pack) {
        const auto out0 = mul_cvt_fp32_to_fp4_4x<false>(
            make_float2(vals0[pack * 4 + 0], vals0[pack * 4 + 1]),
            make_float2(vals0[pack * 4 + 2], vals0[pack * 4 + 3]),
            scale,
            0u);
        const auto out1 = mul_cvt_fp32_to_fp4_4x<false>(
            make_float2(vals1[pack * 4 + 0], vals1[pack * 4 + 1]),
            make_float2(vals1[pack * 4 + 2], vals1[pack * 4 + 3]),
            scale,
            0u);
        packed16_0[pack] = *reinterpret_cast<const uint16_t*>(&out0);
        packed16_1[pack] = *reinterpret_cast<const uint16_t*>(&out1);
    }

    const uint32_t word0 = static_cast<uint32_t>(packed16_0[0]) |
                           (static_cast<uint32_t>(packed16_0[1]) << 16);
    const uint32_t word1 = static_cast<uint32_t>(packed16_1[0]) |
                           (static_cast<uint32_t>(packed16_1[1]) << 16);
    packed0 = make_uint2(word0, __shfl_xor_sync(0xffffffff, word0, 1));
    packed1 = make_uint2(word1, __shfl_xor_sync(0xffffffff, word1, 1));
    return e8m0_val;
}

template<bool SCALE_RP, bool WITH_AMAX>
__device__ __forceinline__ void quantize_a512_transposed_16x32_tecopy_pack_scale(
    const a512_tile& a_smem,
    uint2& packed0,
    uint2& packed1,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int tile_row,
    int tile_col_block32,
    int input_m_base,
    int input_k_base,
    int ntk = 0,
    bool swizzled_scale = false
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    const uint32_t shared_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<bf16*>(&a_smem.data[0])));

    float vals0[8];
    float vals1[8];
    float my_amax = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const int m0 = input_m_base + half * 8 + pair * 2;
        const int m1 = m0 + 1;
        const int k = input_k_base + row;
        bf16 b00;
        bf16 b01;
        bf16 b10;
        bf16 b11;
        move<bf16>::lds(b00, a_smem.idx(shared_addr, {m0, k}));
        move<bf16>::lds(b01, a_smem.idx(shared_addr, {m1, k}));
        move<bf16>::lds(b10, a_smem.idx(shared_addr, {m0 + 16, k}));
        move<bf16>::lds(b11, a_smem.idx(shared_addr, {m1 + 16, k}));
        const float v00 = __bfloat162float(b00);
        const float v01 = __bfloat162float(b01);
        const float v10 = __bfloat162float(b10);
        const float v11 = __bfloat162float(b11);
        vals0[pair * 2 + 0] = v00;
        vals0[pair * 2 + 1] = v01;
        vals1[pair * 2 + 0] = v10;
        vals1[pair * 2 + 1] = v11;
        my_amax = fmaxf(my_amax, fabsf(v00));
        my_amax = fmaxf(my_amax, fabsf(v01));
        my_amax = fmaxf(my_amax, fabsf(v10));
        my_amax = fmaxf(my_amax, fabsf(v11));
    }

    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);
    uint8_t e8m0_val;
    if constexpr (SCALE_RP) {
        e8m0_val = float_to_e8m0_rp_ptx(block_amax);
    } else {
        e8m0_val = float_to_e8m0_dispatch<QuantMode::RTE>(block_amax);
    }
    if (half == 0 && scales_out != nullptr) {
        const int global_row = tile_row + row;
        if (swizzled_scale) {
            const int ctaid_X = tile_col_block32 / 4;
            const int scale_block = tile_col_block32 & 3;
            const int row_in_chunk = global_row & 127;
            const int j = row_in_chunk & 31;
            const int grp = row_in_chunk >> 5;
            const int ctaid_Y = global_row >> 7;
            scales_out[(ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4 + scale_block] = e8m0_val;
        } else {
            scales_out[tile_col_block32 * rows + global_row] = e8m0_val;
        }
    }
    if constexpr (WITH_AMAX) {
        if (half == 0 && amax_out != nullptr) {
            amax_out[tile_col_block32 * rows + tile_row + row] = block_amax;
        }
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);
    uint16_t packed16_0[2];
    uint16_t packed16_1[2];
    #pragma unroll
    for (int pack = 0; pack < 2; ++pack) {
        const auto out0 = mul_cvt_fp32_to_fp4_4x<false>(
            make_float2(vals0[pack * 4 + 0], vals0[pack * 4 + 1]),
            make_float2(vals0[pack * 4 + 2], vals0[pack * 4 + 3]),
            scale,
            0u);
        const auto out1 = mul_cvt_fp32_to_fp4_4x<false>(
            make_float2(vals1[pack * 4 + 0], vals1[pack * 4 + 1]),
            make_float2(vals1[pack * 4 + 2], vals1[pack * 4 + 3]),
            scale,
            0u);
        packed16_0[pack] = *reinterpret_cast<const uint16_t*>(&out0);
        packed16_1[pack] = *reinterpret_cast<const uint16_t*>(&out1);
    }

    const uint32_t word0 = static_cast<uint32_t>(packed16_0[0]) |
                           (static_cast<uint32_t>(packed16_0[1]) << 16);
    const uint32_t word1 = static_cast<uint32_t>(packed16_1[0]) |
                           (static_cast<uint32_t>(packed16_1[1]) << 16);
    packed0 = make_uint2(word0, __shfl_xor_sync(0xffffffff, word0, 1));
    packed1 = make_uint2(word1, __shfl_xor_sync(0xffffffff, word1, 1));
}

__device__ __forceinline__ uint32_t bf16x2_bits_update_amax(
    bf16_2 vals,
    IType2& thread_amax_2x
) {
    const IType2 pair = {__low2bfloat16(vals), __high2bfloat16(vals)};
    ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, pair);
    return *reinterpret_cast<const uint32_t*>(&vals);
}

__device__ __forceinline__ uint32_t bf16x2_bits_update_amax(
    uint32_t vals,
    IType2& thread_amax_2x
) {
    const bf16_2 packed = *reinterpret_cast<const bf16_2*>(&vals);
    return bf16x2_bits_update_amax(packed, thread_amax_2x);
}

__device__ __forceinline__ ulonglong2 lds_v2_u64_shared(uint32_t addr) {
    ulonglong2 out;
    asm volatile(
        "ld.shared.v2.u64 {%0, %1}, [%2];\n"
        : "=l"(out.x), "=l"(out.y)
        : "r"(addr));
    return out;
}

__device__ __forceinline__ float exp2f_rcp_common_e8m0(uint8_t biased_exp) {
    const uint32_t e = static_cast<uint32_t>(biased_exp);
    const uint32_t normal_bits = (254u - e) << 23;
    const uint32_t special_bits = (e == 254u) ? 0x00400000u : 0x7fffffffu;
    return __int_as_float((e >= 254u) ? special_bits : normal_bits);
}

template<bool SCALE_RP, bool WITH_AMAX, typename Tile0, typename Tile1>
__device__ __forceinline__ void quantize_tiles_16x32_tecopy_pack_scale_bf16x8(
    const Tile0& tile0,
    const Tile1& tile1,
    uint2& packed0,
    uint2& packed1,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int tile_row,
    int tile_col_block32
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    const uint32_t shared_addr0 = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<bf16*>(&tile0.data[0])));
    const uint32_t shared_addr1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<bf16*>(&tile1.data[0])));

    IType2 thread_amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    const ulonglong2 elts07_0 = lds_v2_u64_shared(tile0.idx(shared_addr0, {row, half * 8 + 0}));
    const ulonglong2 elts07_1 = lds_v2_u64_shared(tile1.idx(shared_addr1, {row, half * 8 + 0}));
    const uint64_t elts03_0 = elts07_0.x;
    const uint64_t elts47_0 = elts07_0.y;
    const uint64_t elts03_1 = elts07_1.x;
    const uint64_t elts47_1 = elts07_1.y;
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts03_0), thread_amax_2x);
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts03_0 >> 32), thread_amax_2x);
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts47_0), thread_amax_2x);
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts47_0 >> 32), thread_amax_2x);
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts03_1), thread_amax_2x);
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts03_1 >> 32), thread_amax_2x);
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts47_1), thread_amax_2x);
    bf16x2_bits_update_amax(static_cast<uint32_t>(elts47_1 >> 32), thread_amax_2x);

    const float thread_amax = fmaxf(
        __bfloat162float(__habs(thread_amax_2x.x)),
        __bfloat162float(__habs(thread_amax_2x.y)));
    const float peer_amax = __shfl_xor_sync(0xffffffff, thread_amax, 1);
    const float block_amax = fmaxf(thread_amax, peer_amax);
    uint8_t e8m0_val;
    if constexpr (SCALE_RP) {
        e8m0_val = float_to_e8m0_rp_ptx(block_amax);
    } else {
        e8m0_val = float_to_e8m0_dispatch<QuantMode::RTE>(block_amax);
    }
    if (half == 0 && scales_out != nullptr) {
        scales_out[tile_col_block32 * rows + tile_row + row] = e8m0_val;
    }
    if constexpr (WITH_AMAX) {
        if (half == 0 && amax_out != nullptr) {
            amax_out[tile_col_block32 * rows + tile_row + row] = block_amax;
        }
    }

    const bf16 coeff = __float2bfloat16(6.0f * exp2f_rcp_common_e8m0(e8m0_val));
    const uint32_t word0 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<bf16>(elts03_0, elts47_0, coeff);
    const uint32_t word1 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<bf16>(elts03_1, elts47_1, coeff);
    packed0 = make_uint2(word0, __shfl_xor_sync(0xffffffff, word0, 1));
    packed1 = make_uint2(word1, __shfl_xor_sync(0xffffffff, word1, 1));
}

template<bool SCALE_RP, bool WITH_AMAX, bool BF16_ROUND = false>
__device__ __forceinline__ void quantize_reg_tiles_16x32_tecopy_pack_scale(
    const lane_accum16_rt& frag0,
    const lane_accum16_rt& frag1,
    uint2& packed0,
    uint2& packed1,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int tile_row,
    int tile_col_block32
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;

    float vals0[8];
    float vals1[8];
    float my_amax = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const int src_lane = (row & 7) * 4 + pair;
        const int data_idx = half * 2 + (row >> 3);
        const float2 src0 = frag0.tiles[0][0].data[data_idx];
        const float2 src1 = frag1.tiles[0][0].data[data_idx];
        float2 v0 = make_float2(
            __shfl_sync(0xffffffff, src0.x, src_lane),
            __shfl_sync(0xffffffff, src0.y, src_lane));
        float2 v1 = make_float2(
            __shfl_sync(0xffffffff, src1.x, src_lane),
            __shfl_sync(0xffffffff, src1.y, src_lane));
        if constexpr (BF16_ROUND) {
            const bf16_2 b0 = __floats2bfloat162_rn(v0.x, v0.y);
            const bf16_2 b1 = __floats2bfloat162_rn(v1.x, v1.y);
            v0 = make_float2(__bfloat162float(__low2bfloat16(b0)),
                             __bfloat162float(__high2bfloat16(b0)));
            v1 = make_float2(__bfloat162float(__low2bfloat16(b1)),
                             __bfloat162float(__high2bfloat16(b1)));
        }
        vals0[pair * 2 + 0] = v0.x;
        vals0[pair * 2 + 1] = v0.y;
        vals1[pair * 2 + 0] = v1.x;
        vals1[pair * 2 + 1] = v1.y;
        my_amax = fmaxf(my_amax, fabsf(v0.x));
        my_amax = fmaxf(my_amax, fabsf(v0.y));
        my_amax = fmaxf(my_amax, fabsf(v1.x));
        my_amax = fmaxf(my_amax, fabsf(v1.y));
    }

    const float peer_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
    const float block_amax = fmaxf(my_amax, peer_amax);
    uint8_t e8m0_val;
    if constexpr (SCALE_RP) {
        e8m0_val = float_to_e8m0_rp_ptx(block_amax);
    } else {
        e8m0_val = float_to_e8m0_dispatch<QuantMode::RTE>(block_amax);
    }
    if (half == 0 && scales_out != nullptr) {
        scales_out[tile_col_block32 * rows + tile_row + row] = e8m0_val;
    }
    if constexpr (WITH_AMAX) {
        if (half == 0 && amax_out != nullptr) {
            amax_out[tile_col_block32 * rows + tile_row + row] = block_amax;
        }
    }

    const float coeff = 6.0f * exp2f_rcp(e8m0_val);
    const float2 scale = make_float2(coeff, coeff);
    uint16_t packed16_0[2];
    uint16_t packed16_1[2];
    #pragma unroll
    for (int pack = 0; pack < 2; ++pack) {
        const auto out0 = mul_cvt_fp32_to_fp4_4x<false>(
            make_float2(vals0[pack * 4 + 0], vals0[pack * 4 + 1]),
            make_float2(vals0[pack * 4 + 2], vals0[pack * 4 + 3]),
            scale,
            0u);
        const auto out1 = mul_cvt_fp32_to_fp4_4x<false>(
            make_float2(vals1[pack * 4 + 0], vals1[pack * 4 + 1]),
            make_float2(vals1[pack * 4 + 2], vals1[pack * 4 + 3]),
            scale,
            0u);
        packed16_0[pack] = *reinterpret_cast<const uint16_t*>(&out0);
        packed16_1[pack] = *reinterpret_cast<const uint16_t*>(&out1);
    }

    const uint32_t word0 = static_cast<uint32_t>(packed16_0[0]) |
                           (static_cast<uint32_t>(packed16_0[1]) << 16);
    const uint32_t word1 = static_cast<uint32_t>(packed16_1[0]) |
                           (static_cast<uint32_t>(packed16_1[1]) << 16);
    packed0 = make_uint2(word0, __shfl_xor_sync(0xffffffff, word0, 1));
    packed1 = make_uint2(word1, __shfl_xor_sync(0xffffffff, word1, 1));
}

__device__ __forceinline__ void store_packed_16x64_tecopy_rte(
    const uint2 (&packed_words)[TE_COPY16_STORE_GROUP],
    uint8_t* __restrict__ fp4_out,
    int cols,
    int tile_row,
    int tile_col_block16
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out
            + static_cast<int64_t>(tile_row + row) * (cols / 2)
            + tile_col_block16 * 8;
        const ulonglong4_32a out_vec{
            (static_cast<unsigned long long>(packed_words[0].y) << 32) | packed_words[0].x,
            (static_cast<unsigned long long>(packed_words[1].y) << 32) | packed_words[1].x,
            (static_cast<unsigned long long>(packed_words[2].y) << 32) | packed_words[2].x,
            (static_cast<unsigned long long>(packed_words[3].y) << 32) | packed_words[3].x};
        *reinterpret_cast<ulonglong4_32a*>(row_ptr) = out_vec;
    }
}

__device__ __forceinline__ void store_packed_16x64_tecopy_rte_noalloc(
    const uint2 (&packed_words)[TE_COPY16_STORE_GROUP],
    uint8_t* __restrict__ fp4_out,
    int cols,
    int tile_row,
    int tile_col_block16
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out
            + static_cast<int64_t>(tile_row + row) * (cols / 2)
            + tile_col_block16 * 8;
        const unsigned long long v0 =
            (static_cast<unsigned long long>(packed_words[0].y) << 32) | packed_words[0].x;
        const unsigned long long v1 =
            (static_cast<unsigned long long>(packed_words[1].y) << 32) | packed_words[1].x;
        const unsigned long long v2 =
            (static_cast<unsigned long long>(packed_words[2].y) << 32) | packed_words[2].x;
        const unsigned long long v3 =
            (static_cast<unsigned long long>(packed_words[3].y) << 32) | packed_words[3].x;
        asm volatile(
            "st.global.L1::no_allocate.v4.u64 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(row_ptr), "l"(v0), "l"(v1), "l"(v2), "l"(v3)
            : "memory");
    }
}

__device__ __forceinline__ void store_packed_16x32_tecopy_reg_rte(
    const uint2& packed0,
    const uint2& packed1,
    uint8_t* __restrict__ fp4_out,
    int output_cols,
    int tile_row,
    int tile_col_block32
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out
            + static_cast<int64_t>(tile_row + row) * (output_cols / 2)
            + tile_col_block32 * 16;
        asm volatile(
            "st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(row_ptr), "r"(packed0.x), "r"(packed0.y), "r"(packed1.x), "r"(packed1.y)
            : "memory");
    }
}

__device__ __forceinline__ void store_packed_16x32_tecopy_rte(
    const uint2& packed0,
    const uint2& packed1,
    uint8_t* __restrict__ fp4_out,
    int output_cols,
    int tile_row,
    int tile_col_block32
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out
            + static_cast<int64_t>(tile_row + row) * (output_cols / 2)
            + tile_col_block32 * 16;
        const uint4 out_vec{packed0.x, packed0.y, packed1.x, packed1.y};
        *reinterpret_cast<uint4*>(row_ptr) = out_vec;
    }
}

__device__ __forceinline__ void store_packed_16x32_tecopy_rte_noalloc(
    const uint2& packed0,
    const uint2& packed1,
    uint8_t* __restrict__ fp4_out,
    int output_cols,
    int tile_row,
    int tile_col_block32
) {
    const int lane = laneid();
    const int row = lane >> 1;
    const int half = lane & 1;
    if (half == 0 && fp4_out != nullptr) {
        uint8_t* row_ptr = fp4_out
            + static_cast<int64_t>(tile_row + row) * (output_cols / 2)
            + tile_col_block32 * 16;
        asm volatile(
            "st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(row_ptr), "r"(packed0.x), "r"(packed0.y), "r"(packed1.x), "r"(packed1.y)
            : "memory");
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
__device__ __forceinline__ void quantize_reg_tile_16x32(
    const lane_accum_rt& reg_frag,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int cols,
    int tile_row,
    int tile_col_block,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    const int lane = laneid();
    const int row_group = lane >> 2;
    const int lane_in_group = lane & 3;
    constexpr uint32_t FULL_MASK = 0xffffffffu;

    float local_amax_low = 0.0f;
    float local_amax_high = 0.0f;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const float2 low0 = reg_frag.tiles[0][j].data[kk * 2 + 0];
            const float2 high0 = reg_frag.tiles[0][j].data[kk * 2 + 1];
            local_amax_low = fmaxf(local_amax_low, fabsf(low0.x));
            local_amax_low = fmaxf(local_amax_low, fabsf(low0.y));
            local_amax_high = fmaxf(local_amax_high, fabsf(high0.x));
            local_amax_high = fmaxf(local_amax_high, fabsf(high0.y));
        }
    }

    float block_amax_low = fmaxf(local_amax_low, __shfl_xor_sync(FULL_MASK, local_amax_low, 1));
    block_amax_low = fmaxf(block_amax_low, __shfl_xor_sync(FULL_MASK, block_amax_low, 2));
    float block_amax_high = fmaxf(local_amax_high, __shfl_xor_sync(FULL_MASK, local_amax_high, 1));
    block_amax_high = fmaxf(block_amax_high, __shfl_xor_sync(FULL_MASK, block_amax_high, 2));

    const uint64_t block_linear = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    RNGState scale_rng_low;
    scale_rng_low.init(rng_seed, rng_subsequence_base + block_linear * 16 + row_group, 0);
    uint4 scale_random_low = make_uint4(0, 0, 0, 0);
    int scale_rnd_low = 4;
    make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(scale_rng_low, scale_random_low, scale_rnd_low);
    const uint8_t scale_low = quantize_scale<SCALE_SR>(block_amax_low, scale_rng_low, scale_random_low, scale_rnd_low);

    RNGState scale_rng_high;
    scale_rng_high.init(rng_seed, rng_subsequence_base + block_linear * 16 + row_group + 8, 0);
    uint4 scale_random_high = make_uint4(0, 0, 0, 0);
    int scale_rnd_high = 4;
    make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(scale_rng_high, scale_random_high, scale_rnd_high);
    const uint8_t scale_high = quantize_scale<SCALE_SR>(block_amax_high, scale_rng_high, scale_random_high, scale_rnd_high);

    if (lane_in_group == 0) {
        scales_out[(tile_row + row_group) * (cols / 32) + tile_col_block] = scale_low;
        if constexpr (WITH_AMAX) {
            amax_out[(tile_row + row_group) * (cols / 32) + tile_col_block] = block_amax_low;
        }
    }
    if (lane_in_group == 1) {
        scales_out[(tile_row + row_group + 8) * (cols / 32) + tile_col_block] = scale_high;
        if constexpr (WITH_AMAX) {
            amax_out[(tile_row + row_group + 8) * (cols / 32) + tile_col_block] = block_amax_high;
        }
    }

    #pragma unroll
    for (int row_sel = 0; row_sel < 2; ++row_sel) {
        const int row = row_group + row_sel * 8;
        const int k_base = row_sel;
        const bool lane_writes_row = ((lane_in_group & 1) == row_sel);
        const int pack_parity = lane_in_group >> 1;
        const float block_amax = row_sel == 0 ? block_amax_low : block_amax_high;

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            RNGState rng;
            rng.init(rng_seed, rng_subsequence_base + block_linear * 16 + row, 0);
            uint4 random_uint4 = make_uint4(0, 0, 0, 0);
            int rnd_idx = 4;
            const uint32_t sign_bits = make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
            const uint8_t e8m0_val = quantize_scale<SCALE_SR>(block_amax, rng, random_uint4, rnd_idx);
            const float coeff = 6.0f * exp2f_rcp(e8m0_val);
            const float2 scale = make_float2(coeff, coeff);

            #pragma unroll
            for (int pack = 0; pack < 4; ++pack) {
                const int j = half;
                const int k = k_base + (pack >= 2 ? 2 : 0);
                const int pair_base = (pack & 1) * 2;
                const int src0 = row_group * 4 + pair_base;
                const int src1 = src0 + 1;
                const int col0 = half * 16 + pack * 4;

                float2 in01 = make_float2(
                    __shfl_sync(FULL_MASK, reg_frag.tiles[0][j].data[k].x, src0),
                    __shfl_sync(FULL_MASK, reg_frag.tiles[0][j].data[k].y, src0));
                float2 in23 = make_float2(
                    __shfl_sync(FULL_MASK, reg_frag.tiles[0][j].data[k].x, src1),
                    __shfl_sync(FULL_MASK, reg_frag.tiles[0][j].data[k].y, src1));
                in01.x *= ((sign_bits >> (col0 + 0)) & 1u) ? 1.0f : -1.0f;
                in01.y *= ((sign_bits >> (col0 + 1)) & 1u) ? 1.0f : -1.0f;
                in23.x *= ((sign_bits >> (col0 + 2)) & 1u) ? 1.0f : -1.0f;
                in23.y *= ((sign_bits >> (col0 + 3)) & 1u) ? 1.0f : -1.0f;

                const auto packed = mul_cvt_fp32_to_fp4_4x<DATA_SR>(
                    in01,
                    in23,
                    scale,
                    maybe_next_data_rbits<DATA_SR>(rng, random_uint4, rnd_idx)
                );
                if (lane_writes_row && ((pack & 1) == pack_parity)) {
                    uint8_t* row_ptr = fp4_out + static_cast<int64_t>(tile_row + row) * (cols / 2) + tile_col_block * 16 + half * 8;
                    *reinterpret_cast<uint16_t*>(row_ptr + pack * 2) = *reinterpret_cast<const uint16_t*>(&packed);
                }
            }
        }
    }
}

template<bool WITH_RANDOM_SIGN_MASK, typename Tile>
__device__ __forceinline__ void apply_sign_mask_64x32(
    Tile& tile,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    const int tid = threadIdx.x & 127;
    const int row = tid >> 1;
    const int half = tid & 1;
    if (row >= TILE_M) {
        return;
    }

    RNGState rng;
    const uint64_t block_linear = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    rng.init(rng_seed, rng_subsequence_base + block_linear * TILE_M + row, 0);
    uint4 random_uint4 = make_uint4(0, 0, 0, 0);
    int rnd_idx = 4;
    const uint32_t sign_bits = make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = half * 16 + i;
        float v = __bfloat162float(tile[{row, col}]);
        v *= ((sign_bits >> col) & 1u) ? 1.0f : -1.0f;
        tile[{row, col}] = __float2bfloat16_rn(v);
    }
}

struct DeviceBuffers {
    __nv_bfloat16* input = nullptr;
    uint8_t* out_fp4 = nullptr;
    uint8_t* out_col_fp4 = nullptr;
    uint8_t* out_sc = nullptr;
    uint8_t* out_col_sc = nullptr;
    float* out_amax = nullptr;
    float* out_col_amax = nullptr;
    __nv_bfloat16* h16 = nullptr;
    __nv_bfloat16* h32 = nullptr;
};

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

void check_cu(CUresult err, const char* what) {
    if (err != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(err, &name);
        std::cerr << what << ": " << (name ? name : "unknown CUresult") << " (" << err << ")" << std::endl;
        std::exit(1);
    }
}

void* libcuda_symbol(const char* symbol) {
    static void* handle = []() {
        void* h = dlopen("libcuda.so.1", RTLD_LAZY);
        if (h == nullptr) {
            std::cerr << "dlopen libcuda.so.1 failed: " << dlerror() << std::endl;
            std::exit(1);
        }
        return h;
    }();
    void* fn = dlsym(handle, symbol);
    if (fn == nullptr) {
        std::cerr << "dlsym " << symbol << " failed: " << dlerror() << std::endl;
        std::exit(1);
    }
    return fn;
}

void create_tma_2d_bench(
    CUtensorMap& tma,
    void* ptr,
    uint64_t dim_y,
    uint64_t dim_x,
    uint32_t box_y,
    uint32_t box_x,
    uint64_t stride_x,
    size_t elem_bits
) {
    using cuTensorMapEncodeTiled_t = CUresult (*)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
        const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static auto encode = reinterpret_cast<cuTensorMapEncodeTiled_t>(
        libcuda_symbol("cuTensorMapEncodeTiled"));

    CUtensorMapDataType dtype;
    if (elem_bits == 16) {
        dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if (elem_bits == 8) {
        dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if (elem_bits == 4) {
        dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    } else {
        std::cerr << "unsupported TMA elem bits: " << elem_bits << std::endl;
        std::exit(1);
    }

    const uint64_t size[2] = {dim_x, dim_y};
    const uint64_t stride[1] = {stride_x * elem_bits / 8};
    const uint32_t box[2] = {box_x, box_y};
    const uint32_t elem_stride[2] = {1, 1};
    check_cu(
        encode(
            &tma,
            dtype,
            2,
            ptr,
            size,
            stride,
            box,
            elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");
}

std::vector<__nv_bfloat16> make_hadamard_host(int n, bool with_static_random_sign = false) {
    std::vector<float> h(n * n, 1.0f);
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                for (int r = 0; r < n; ++r) {
                    h[r * n + i + len + j] = -h[r * n + i + len + j];
                }
            }
        }
    }
    if (with_static_random_sign && (n == 16 || n == 32)) {
        constexpr int signs[32] = {
            1, 1, 1, -1, 1, -1, -1, -1,
            -1, -1, -1, 1, -1, 1, -1, -1,
            1, -1, 1, 1, -1, 1, -1, 1,
            -1, 1, 1, -1, -1, 1, -1, 1
        };
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                h[r * n + c] *= static_cast<float>(signs[c]);
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    std::vector<__nv_bfloat16> out(n * n);
    for (int i = 0; i < n * n; ++i) {
        out[i] = __float2bfloat16_rn(h[i] * scale);
    }
    return out;
}

__global__ void init_bf16_pattern_kernel(__nv_bfloat16* input, size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        uint32_t x = static_cast<uint32_t>(i);
        x ^= x >> 17;
        x *= 0xed5ad4bbU;
        x ^= x >> 11;
        const float v = (static_cast<int>(x & 511U) - 255) * (1.0f / 255.0f);
        input[i] = __float2bfloat16_rn(v);
    }
}

void init_device_buffers(DeviceBuffers& bufs, int rows, int cols) {
    const bool fast_init = std::getenv("TCBENCH_FAST_INIT") != nullptr;
    const bool static_rsm = std::getenv("TCBENCH_STATIC_RSM") != nullptr;
    const auto h16 = make_hadamard_host(16, static_rsm);
    const auto h32 = make_hadamard_host(32, static_rsm);
    const size_t input_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    check(cudaMalloc(&bufs.input, input_elems * sizeof(__nv_bfloat16)), "cudaMalloc input");
    check(cudaMalloc(&bufs.out_fp4, rows * (cols / 2) * sizeof(uint8_t)), "cudaMalloc out_fp4");
    check(cudaMalloc(&bufs.out_col_fp4, rows * (cols / 2) * sizeof(uint8_t)), "cudaMalloc out_col_fp4");
    check(cudaMalloc(&bufs.out_sc, rows * (cols / 16) * sizeof(uint8_t)), "cudaMalloc out_sc");
    check(cudaMalloc(&bufs.out_col_sc, rows * (cols / 16) * sizeof(uint8_t)), "cudaMalloc out_col_sc");
    check(cudaMalloc(&bufs.out_amax, rows * (cols / 16) * sizeof(float)), "cudaMalloc out_amax");
    check(cudaMalloc(&bufs.out_col_amax, rows * (cols / 16) * sizeof(float)), "cudaMalloc out_col_amax");
    check(cudaMalloc(&bufs.h16, 16 * 16 * sizeof(__nv_bfloat16)), "cudaMalloc h16");
    check(cudaMalloc(&bufs.h32, 32 * 32 * sizeof(__nv_bfloat16)), "cudaMalloc h32");

    if (fast_init) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((input_elems + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        init_bf16_pattern_kernel<<<blocks, threads>>>(reinterpret_cast<__nv_bfloat16*>(bufs.input), input_elems);
        check(cudaGetLastError(), "launch input pattern");
        check(cudaDeviceSynchronize(), "sync input pattern");
    } else {
        std::vector<__nv_bfloat16> host(input_elems);
        std::mt19937 gen(1234);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto& v : host) {
            v = __float2bfloat16_rn(dis(gen));
        }
        check(cudaMemcpy(bufs.input, host.data(), input_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "memcpy input");
    }
    check(cudaMemcpy(bufs.h16, h16.data(), 16 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "memcpy h16");
    check(cudaMemcpy(bufs.h32, h32.data(), 32 * 32 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "memcpy h32");
}

void free_device_buffers(DeviceBuffers& bufs) {
    cudaFree(bufs.input);
    cudaFree(bufs.out_fp4);
    cudaFree(bufs.out_col_fp4);
    cudaFree(bufs.out_sc);
    cudaFree(bufs.out_col_sc);
    cudaFree(bufs.out_amax);
    cudaFree(bufs.out_col_amax);
    cudaFree(bufs.h16);
    cudaFree(bufs.h32);
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
__global__ __launch_bounds__(NUM_THREADS, 1)
void te_style_direct32_quant_kernel(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<h32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a128_tile (&a_smem) = al.allocate<a128_tile>();
    h32_tile (&h_smem) = al.allocate<h32_tile>();
    d16_tile (&epi_smem)[4] = al.allocate<d16_tile, 4>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[2];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr;
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done[0], 0, 1);
        init_semaphore(compute_done[1], 0, 1);
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a128_tile) + sizeof(h32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(inputs_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    full_accum_tt accum;
    if (warp_id == 1 && lane == 0) {
        accum = tm_alloc.template allocate<full_accum_tt>(0);
        accum_addr = accum.addr;
        warp::arrive(accum_ready);
    }
    if (warp_id == 1 || warp_id == 2) {
        wait(accum_ready, 0);
        accum = full_accum_tt(accum_addr);
        const int sy = warp_id - 1;
        #pragma unroll
        for (int sx = 0; sx < 4; ++sx) {
            auto a_sub = a_smem.template subtile<64, 32>({sy, sx});
            if (lane == 0) {
                auto acc_sub = accum.template subtile<block_accum_tt>(sy * 64, sx * 32);
                mm_AB(acc_sub, a_sub, h_smem);
            }
        }
        if (lane == 0) {
            kittens::detail::tcgen05::commit<1>(compute_done[sy]);
        }
    }

    wait(compute_done[0], 0);
    wait(compute_done[1], 0);

    if (warpgroup::groupid() == 1) {
        const int local_warp = warpgroup::warpid();
        lane_accum_rt reg_frag;
        #pragma unroll
        for (int sy = 0; sy < 2; ++sy) {
            #pragma unroll
            for (int sx = 0; sx < 4; ++sx) {
                auto tm_sub = accum.template subtile<lane_accum_tt>(sy * 64 + local_warp * 16, sx * 32);
                warp::load_async(reg_frag, tm_sub);
                tensor_load_wait();
                group<1>::store(epi_smem[local_warp], reg_frag);
                __syncwarp();
                quantize_tile_16x32<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>(
                    epi_smem[local_warp],
                    fp4_out,
                    scales_out,
                    amax_out,
                    cols,
                    bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                    bid_k * 4 + sx,
                    rng_seed,
                    rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * 8 + sy * 4 + sx) * TILE_M + local_warp * 16
                );
                __syncwarp();
            }
        }
    }

    __syncthreads();
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
__global__ __launch_bounds__(NUM_THREADS, 1)
void te_style_direct32_quant_kernel_wide(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h32_tile (&h_smem) = al.allocate<h32_tile>();
    d16_tile (&epi_smem)[4] = al.allocate<d16_tile, 4>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[2];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr;
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done[0], 0, 1);
        init_semaphore(compute_done[1], 0, 1);
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a512_tile) + sizeof(h32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(inputs_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    full_accum_tt_wide accum;
    if (warp_id == 1 && lane == 0) {
        accum = tm_alloc.template allocate<full_accum_tt_wide>(0);
        accum_addr = accum.addr;
        warp::arrive(accum_ready);
    }
    if (warp_id == 1 || warp_id == 2) {
        wait(accum_ready, 0);
        accum = full_accum_tt_wide(accum_addr);
        const int sy = warp_id - 1;
        #pragma unroll
        for (int sx = 0; sx < CHUNK_K_WIDE / TILE_N; ++sx) {
            auto a_sub = a_smem.template subtile<64, 32>({sy, sx});
            if (lane == 0) {
                auto acc_sub = accum.template subtile<block_accum_tt>(sy * 64, sx * 32);
                mm_AB(acc_sub, a_sub, h_smem);
            }
        }
        if (lane == 0) {
            kittens::detail::tcgen05::commit<1>(compute_done[sy]);
        }
    }

    wait(compute_done[0], 0);
    wait(compute_done[1], 0);

    if (warpgroup::groupid() == 1) {
        const int local_warp = warpgroup::warpid();
        lane_accum_rt reg_frag;
        #pragma unroll
        for (int sy = 0; sy < 2; ++sy) {
            #pragma unroll
            for (int sx = 0; sx < CHUNK_K_WIDE / TILE_N; ++sx) {
                auto tm_sub = accum.template subtile<lane_accum_tt>(sy * 64 + local_warp * 16, sx * 32);
                warp::load_async(reg_frag, tm_sub);
                tensor_load_wait();
                group<1>::store(epi_smem[local_warp], reg_frag);
                __syncwarp();
                quantize_tile_16x32<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>(
                    epi_smem[local_warp],
                    fp4_out,
                    scales_out,
                    amax_out,
                    cols,
                    bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                    bid_k * (CHUNK_K_WIDE / TILE_N) + sx,
                    rng_seed,
                    rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * (2 * (CHUNK_K_WIDE / TILE_N)) + sy * (CHUNK_K_WIDE / TILE_N) + sx) * TILE_M + local_warp * 16
                );
                __syncwarp();
            }
        }
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
__global__ __launch_bounds__(NUM_THREADS, 1)
void te_style_direct32_quant_kernel_pipe64(
    const __grid_constant__ a64_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    // TE-style split pipeline: one TMA producer streams 64-column chunks,
    // two MMA warps publish staged accumulators, and the epilogue warpgroup
    // quantizes each stage as soon as both row halves are complete.
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a64_tile>();
        H_layout.template prefetch_tma<h32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    const int warp_id = warpid();
    const int lane = laneid();
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a64_tile (&a_smem)[LOAD_PIPE_STAGES] = al.allocate<a64_tile, LOAD_PIPE_STAGES>();
    h32_tile (&h_smem) = al.allocate<h32_tile>();
    d16_tile (&epi_smem)[4] = al.allocate<d16_tile, 4>();

    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_arrived[LOAD_PIPE_STAGES];
    __shared__ semaphore inputs_finished[LOAD_PIPE_STAGES];
    __shared__ semaphore compute_done[PIPE64_STAGES][2];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr[PIPE64_STAGES];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < LOAD_PIPE_STAGES; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 2);
        }
        #pragma unroll
        for (int stage = 0; stage < PIPE64_STAGES; ++stage) {
            init_semaphore(compute_done[stage][0], 0, 1);
            init_semaphore(compute_done[stage][1], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h32_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int stage = 0; stage < PIPE64_STAGES; ++stage) {
            auto accum = tm_alloc.template allocate<full_accum_tt_stage>(stage * CHUNK_K_STAGE);
            accum_addr[stage] = accum.addr;
        }
        warp::arrive(accum_ready);
    }

    if (warp_id == 0 && lane == 0) {
        uint32_t load_stage = 0;
        #pragma unroll
        for (int stage = 0; stage < PIPE64_STAGES; ++stage) {
            wait(inputs_finished[load_stage], get_phasebit<1>(phasebits, load_stage));
            tma::expect_bytes(inputs_arrived[load_stage], sizeof(a64_tile));
            tma::load_async(a_smem[load_stage], A_layout, {bid_m, bid_k * PIPE64_STAGES + stage}, inputs_arrived[load_stage]);
            update_phasebit<1>(phasebits, load_stage);
            load_stage = (load_stage + 1) % LOAD_PIPE_STAGES;
        }
    }

    if (warp_id == 1 || warp_id == 2) {
        wait(accum_ready, 0);
        const int sy = warp_id - 1;
        uint32_t load_stage = 0;
        #pragma unroll
        for (int stage = 0; stage < PIPE64_STAGES; ++stage) {
            wait(inputs_arrived[load_stage], get_phasebit<0>(phasebits, load_stage));
            full_accum_tt_stage accum(accum_addr[stage]);
            #pragma unroll
            for (int sx = 0; sx < CHUNK_K_STAGE / TILE_N; ++sx) {
                auto a_sub = a_smem[load_stage].template subtile<64, 32>({sy, sx});
                if (lane == 0) {
                    auto acc_sub = accum.template subtile<block_accum_tt>(sy * 64, sx * 32);
                    if (sx == (CHUNK_K_STAGE / TILE_N) - 1) {
                        mm_AB(acc_sub, a_sub, h_smem, inputs_finished[load_stage]);
                    } else {
                        mm_AB(acc_sub, a_sub, h_smem);
                    }
                }
            }
            if (lane == 0) {
                kittens::detail::tcgen05::commit<1>(compute_done[stage][sy]);
            }
            update_phasebit<0>(phasebits, load_stage);
            load_stage = (load_stage + 1) % LOAD_PIPE_STAGES;
        }
    }

    if (warpgroup::groupid() == 1) {
        wait(accum_ready, 0);
        const int local_warp = warpgroup::warpid();
        lane_accum_rt reg_frag;
        #pragma unroll
        for (int stage = 0; stage < PIPE64_STAGES; ++stage) {
            wait(compute_done[stage][0], 0);
            wait(compute_done[stage][1], 0);
            full_accum_tt_stage accum(accum_addr[stage]);
            #pragma unroll
            for (int sy = 0; sy < 2; ++sy) {
                #pragma unroll
                for (int sx = 0; sx < CHUNK_K_STAGE / TILE_N; ++sx) {
                    auto tm_sub = accum.template subtile<lane_accum_tt>(sy * 64 + local_warp * 16, sx * 32);
                    warp::load_async(reg_frag, tm_sub);
                    tensor_load_wait();
                    group<1>::store(epi_smem[local_warp], reg_frag);
                    const int sx_global = stage * (CHUNK_K_STAGE / TILE_N) + sx;
                    quantize_tile_16x32<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>(
                        epi_smem[local_warp],
                        fp4_out,
                        scales_out,
                        amax_out,
                        cols,
                        bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                        bid_k * (CHUNK_K_WIDE / TILE_N) + sx_global,
                        rng_seed,
                        rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * (2 * (CHUNK_K_WIDE / TILE_N)) + sy * (CHUNK_K_WIDE / TILE_N) + sx_global) * TILE_M + local_warp * 16
                    );
                }
            }
        }
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RANDOM_SIGN_MASK,
    bool SCALE_RP = true,
    bool SKIP_QUANT_PACK = false,
    bool MX32_SCALE = false,
    bool WITH_AMAX = false>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_quant_kernel_pipe64(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int cols,
    bool swizzled_scale,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    // Mirrors TE's core RHT geometry: 128x128 A stages, H16, eight 16-column
    // accumulator fragments per stage, and one scale per 16 output values.
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }
    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_m * tiles_k;
    const int warp_id = warpid();
    const int lane = laneid();
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a128_tile (&a_smem)[LOAD_PIPE_STAGES] = al.allocate<a128_tile, LOAD_PIPE_STAGES>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    d16x16_tile (&epi_smem)[TE_COPY16_EPILOGUE_WARPS * (MX32_SCALE ? 2 : 1)] =
        al.allocate<d16x16_tile, TE_COPY16_EPILOGUE_WARPS * (MX32_SCALE ? 2 : 1)>();

    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_arrived[LOAD_PIPE_STAGES];
    __shared__ semaphore inputs_finished[LOAD_PIPE_STAGES];
    __shared__ semaphore compute_done[TE_COPY16_ACCUM_PIPE][2];
    __shared__ semaphore outputs_finished[TE_COPY16_ACCUM_PIPE];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr[TE_COPY16_ACCUM_PIPE];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < LOAD_PIPE_STAGES; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 1);
        }
        #pragma unroll
        for (int stage = 0; stage < TE_COPY16_ACCUM_PIPE; ++stage) {
            init_semaphore(compute_done[stage][0], 0, 1);
            init_semaphore(compute_done[stage][1], 0, 1);
            init_semaphore(outputs_finished[stage], 0, TE_COPY16_EPILOGUE_WARPS);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h16_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int stage = 0; stage < TE_COPY16_ACCUM_PIPE; ++stage) {
            auto accum = tm_alloc.template allocate<full_accum_tt>(stage * CHUNK_K);
            accum_addr[stage] = accum.addr;
        }
        warp::arrive(accum_ready);
    }

    if (warp_id == 0 && lane == 0) {
        uint32_t load_stage = 0;
        for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x) {
            const int bid_m = linear_tile % tiles_m;
            const int bid_k = linear_tile / tiles_m;
            #pragma unroll
            for (int stage = 0; stage < TE_COPY16_PIPE_STAGES; ++stage) {
                wait(inputs_finished[load_stage], get_phasebit<1>(phasebits, load_stage));
                tma::expect_bytes(inputs_arrived[load_stage], sizeof(a128_tile));
                tma::load_async(a_smem[load_stage], A_layout, {bid_m, bid_k * TE_COPY16_PIPE_STAGES + stage}, inputs_arrived[load_stage]);
                update_phasebit<1>(phasebits, load_stage);
                load_stage = (load_stage + 1) % LOAD_PIPE_STAGES;
            }
        }
    }

    if (warp_id == 1) {
        wait(accum_ready, 0);
        uint32_t load_stage = 0;
        int tile_iter = 0;
        for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x, ++tile_iter) {
            #pragma unroll
            for (int stage = 0; stage < TE_COPY16_PIPE_STAGES; ++stage) {
                const int global_stage = tile_iter * TE_COPY16_PIPE_STAGES + stage;
                const int accum_slot = stage % TE_COPY16_ACCUM_PIPE;
                if (global_stage >= TE_COPY16_ACCUM_PIPE && lane == 0) {
                    wait(outputs_finished[accum_slot], ((global_stage / TE_COPY16_ACCUM_PIPE) - 1) & 1);
                }
                wait(inputs_arrived[load_stage], get_phasebit<0>(phasebits, load_stage));
                full_accum_tt accum(accum_addr[accum_slot]);
                #pragma unroll
                for (int sx = 0; sx < CHUNK_K / TILE_N_TE; ++sx) {
                    auto a_sub = a_smem[load_stage].template subtile<128, 16>({0, sx});
                    if (lane == 0) {
                        auto acc_sub = accum.template subtile<full_accum16_tt>(0, sx * 16);
                        if (sx == (CHUNK_K / TILE_N_TE) - 1) {
                            mm_AB(acc_sub, a_sub, h_smem, inputs_finished[load_stage]);
                        } else {
                            mm_AB(acc_sub, a_sub, h_smem);
                        }
                    }
                }
                if (lane == 0) {
                    kittens::detail::tcgen05::commit<1>(compute_done[accum_slot][0]);
                    kittens::detail::tcgen05::commit<1>(compute_done[accum_slot][1]);
                }
                update_phasebit<0>(phasebits, load_stage);
                load_stage = (load_stage + 1) % LOAD_PIPE_STAGES;
            }
        }
    }

    if (warpgroup::groupid() == 1 || warpgroup::groupid() == 2) {
        wait(accum_ready, 0);
        const int local_warp = warpgroup::warpid();
        const int local_epi_warp = (warpgroup::groupid() - 1) * 4 + local_warp;
        const int sy = warpgroup::groupid() - 1;
        lane_accum16_rt reg_frag;
        int tile_iter = 0;
        for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x, ++tile_iter) {
            const int bid_m = linear_tile % tiles_m;
            const int bid_k = linear_tile / tiles_m;
            #pragma unroll
            for (int stage = 0; stage < TE_COPY16_PIPE_STAGES; ++stage) {
                const int global_stage = tile_iter * TE_COPY16_PIPE_STAGES + stage;
                const int accum_slot = stage % TE_COPY16_ACCUM_PIPE;
                const int compute_phase = (global_stage / TE_COPY16_ACCUM_PIPE) & 1;
                wait(compute_done[accum_slot][sy], compute_phase);
                full_accum_tt accum(accum_addr[accum_slot]);
                if constexpr (!DATA_SR && !SCALE_SR && !WITH_RANDOM_SIGN_MASK) {
                    if constexpr (SKIP_QUANT_PACK) {
                        #pragma unroll
                        for (int sx = 0; sx < CHUNK_K / TILE_N_TE; ++sx) {
                            auto tm_sub = accum.template subtile<lane_accum16_tt>(sy * 64 + local_warp * 16, sx * 16);
                            warp::load_async(reg_frag, tm_sub);
                            tensor_load_wait();
                            group<1>::store(epi_smem[local_epi_warp], reg_frag);
                        }
                    } else if constexpr (MX32_SCALE) {
                        #pragma unroll
                        for (int sx_base = 0; sx_base < CHUNK_K / TILE_N_TE; sx_base += TE_COPY16_STORE_GROUP) {
                            uint2 packed_words[TE_COPY16_STORE_GROUP];
                            #pragma unroll
                            for (int g = 0; g < TE_COPY16_STORE_GROUP; g += 2) {
                                constexpr int smem_stride = MX32_SCALE ? 2 : 1;
                                const int smem_base = local_epi_warp * smem_stride;
                                auto tm_sub0 = accum.template subtile<lane_accum16_tt>(
                                    sy * 64 + local_warp * 16,
                                    (sx_base + g) * 16);
                                warp::load_async(reg_frag, tm_sub0);
                                tensor_load_wait();
                                group<1>::store(epi_smem[smem_base], reg_frag);

                                auto tm_sub1 = accum.template subtile<lane_accum16_tt>(
                                    sy * 64 + local_warp * 16,
                                    (sx_base + g + 1) * 16);
                                warp::load_async(reg_frag, tm_sub1);
                                tensor_load_wait();
                                group<1>::store(epi_smem[smem_base + 1], reg_frag);

                                const int sx_global = stage * (CHUNK_K / TILE_N_TE) + sx_base + g;
                                quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                                    epi_smem[smem_base],
                                    epi_smem[smem_base + 1],
                                    packed_words[g],
                                    packed_words[g + 1],
                                    scales_out,
                                    amax_out,
                                    rows,
                                    bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                                    (bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_global) / 2,
                                    cols / CHUNK_DIM,
                                    swizzled_scale);
                            }
                            const int sx_global = stage * (CHUNK_K / TILE_N_TE) + sx_base;
                            store_packed_16x64_tecopy_rte(
                                packed_words,
                                fp4_out,
                                cols,
                                bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                                bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_global);
                        }
                    } else {
                        #pragma unroll
                        for (int sx_base = 0; sx_base < CHUNK_K / TILE_N_TE; sx_base += TE_COPY16_STORE_GROUP) {
                            uint2 packed_words[TE_COPY16_STORE_GROUP];
                            #pragma unroll
                            for (int g = 0; g < TE_COPY16_STORE_GROUP; ++g) {
                                auto tm_sub = accum.template subtile<lane_accum16_tt>(
                                    sy * 64 + local_warp * 16,
                                    (sx_base + g) * 16);
                                warp::load_async(reg_frag, tm_sub);
                                tensor_load_wait();
                                group<1>::store(epi_smem[local_epi_warp], reg_frag);
                                const int sx_global = stage * (CHUNK_K / TILE_N_TE) + sx_base + g;
                                packed_words[g] = quantize_tile_16x16_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                                    epi_smem[local_epi_warp],
                                    scales_out,
                                    amax_out,
                                    rows,
                                    bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                                    bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_global);
                            }
                            const int sx_global = stage * (CHUNK_K / TILE_N_TE) + sx_base;
                            store_packed_16x64_tecopy_rte(
                                packed_words,
                                fp4_out,
                                cols,
                                bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                                bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_global);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int sx = 0; sx < CHUNK_K / TILE_N_TE; ++sx) {
                        auto tm_sub = accum.template subtile<lane_accum16_tt>(sy * 64 + local_warp * 16, sx * 16);
                        warp::load_async(reg_frag, tm_sub);
                        tensor_load_wait();
                        const int sx_global = stage * (CHUNK_K / TILE_N_TE) + sx;
                        group<1>::store(epi_smem[local_epi_warp], reg_frag);
                        quantize_tile_16x16_tecopy<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK>(
                            epi_smem[local_epi_warp],
                            fp4_out,
                            scales_out,
                            cols,
                            rows,
                            bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                            bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_global,
                            rng_seed,
                            rng_subsequence_base + static_cast<uint64_t>((linear_tile * (2 * (CHUNK_K_WIDE / TILE_N_TE)) + sy * (CHUNK_K_WIDE / TILE_N_TE) + sx_global)) * TILE_M + local_warp * 16
                        );
                    }
                }
                __syncwarp();
                if (lane == 0) {
                    warp::arrive(outputs_finished[accum_slot]);
                }
            }
        }
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool SCALE_RP = true, bool WITH_AMAX = false, int COL_WARPS = TE_COPY16_EPILOGUE_WARPS>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_col_quant_kernel(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ col_fp4_out,
    uint8_t* __restrict__ col_scales_out,
    float* __restrict__ col_amax_out,
    int rows,
    int cols,
    bool swizzled_scale
) {
    // Prototype column-contract tensor-core RHT:
    // for each 16x16 input tile A_mk, compute A_mk^T * H16 so output rows are K.
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }

    const int bid_k = blockIdx.x;
    const int bid_m = blockIdx.y;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    d16x16_tile (&epi_smem)[TE_COPY16_EPILOGUE_WARPS * 16] =
        al.allocate<d16x16_tile, TE_COPY16_EPILOGUE_WARPS * 16>();
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[TE_COPY16_EPILOGUE_WARPS];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr[TE_COPY16_EPILOGUE_WARPS];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            init_semaphore(compute_done[i], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a512_tile) + sizeof(h16_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(inputs_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            auto acc = tm_alloc.template allocate<full_accum16_tt>(i * 16);
            accum_addr[i] = acc.addr;
        }
        warp::arrive(accum_ready);
    }

    if ((COL_WARPS == TE_COPY16_EPILOGUE_WARPS &&
         (warpgroup::groupid() == 1 || warpgroup::groupid() == 2)) ||
        (COL_WARPS == 4 && warpgroup::groupid() == 1)) {
        wait(accum_ready, 0);
        const int local_warp = warpgroup::warpid();
        const int local_epi_warp = (COL_WARPS == TE_COPY16_EPILOGUE_WARPS)
            ? (warpgroup::groupid() - 1) * 4 + local_warp
            : local_warp;
        const int smem_base = local_epi_warp * 16;
        full_accum16_tt accum_block(accum_addr[local_epi_warp]);
        lane_accum16_rt reg_frag;
        int compute_phase = 0;
        #pragma unroll
        for (int task = local_epi_warp; task < 16; task += COL_WARPS) {
            const int m_pair = task & 3;
            const int sx128 = (task >> 2) * 8;
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int m_block = m_pair * 2 + part;
                auto a_sub = a_smem.template subtile<16, 128>({m_block, sx128 / 8});
                if (lane == 0) {
                    mm_AtB(accum_block, a_sub, h_smem);
                    kittens::detail::tcgen05::commit<1>(compute_done[local_epi_warp]);
                }
                wait(compute_done[local_epi_warp], compute_phase);
                compute_phase ^= 1;
                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    auto tm_sub = accum_block.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                    warp::load_async(reg_frag, tm_sub);
                    tensor_load_wait();
                    group<1>::store(epi_smem[smem_base + part * 8 + k_slice], reg_frag);
                    __syncwarp();
                }
            }

            #pragma unroll
            for (int k_slice = 0; k_slice < 8; ++k_slice) {
                uint2 packed0;
                uint2 packed1;
                const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                    epi_smem[smem_base + k_slice],
                    epi_smem[smem_base + 8 + k_slice],
                    packed0,
                    packed1,
                    col_scales_out,
                    col_amax_out,
                    cols,
                    global_k,
                    global_m_block32,
                    rows / CHUNK_DIM,
                    swizzled_scale);
                store_packed_16x32_tecopy_rte(
                    packed0,
                    packed1,
                    col_fp4_out,
                    rows,
                    global_k,
                    global_m_block32);
                __syncwarp();
            }
        }
    }

    __syncthreads();
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool SCALE_RP = true, bool WITH_AMAX = false>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_col_quant_kernel_persistent(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ col_fp4_out,
    uint8_t* __restrict__ col_scales_out,
    float* __restrict__ col_amax_out,
    int rows,
    int cols
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }

    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_m * tiles_k;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    d16x16_tile (&epi_smem)[TE_COPY16_EPILOGUE_WARPS * 16] =
        al.allocate<d16x16_tile, TE_COPY16_EPILOGUE_WARPS * 16>();
    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[TE_COPY16_EPILOGUE_WARPS];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr[TE_COPY16_EPILOGUE_WARPS];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        init_semaphore(inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            init_semaphore(compute_done[i], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h16_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            auto acc = tm_alloc.template allocate<full_accum16_tt>(i * 16);
            accum_addr[i] = acc.addr;
        }
        warp::arrive(accum_ready);
    }
    wait(accum_ready, 0);

    int input_phase = 0;
    for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x) {
        const int bid_m = linear_tile % tiles_m;
        const int bid_k = linear_tile / tiles_m;
        if (threadIdx.x == 0) {
            tma::expect_bytes(inputs_arrived, sizeof(a512_tile));
            tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        }
        wait(inputs_arrived, input_phase);
        input_phase ^= 1;
        __syncthreads();

        if (warpgroup::groupid() == 1 || warpgroup::groupid() == 2) {
            const int local_warp = warpgroup::warpid();
            const int local_epi_warp = (warpgroup::groupid() - 1) * 4 + local_warp;
            const int smem_base = local_epi_warp * 16;
            full_accum16_tt accum_block(accum_addr[local_epi_warp]);
            lane_accum16_rt reg_frag;
            int compute_phase = 0;
            #pragma unroll
            for (int task = local_epi_warp; task < 16; task += TE_COPY16_EPILOGUE_WARPS) {
                const int m_pair = task & 3;
                const int sx128 = (task >> 2) * 8;
                #pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const int m_block = m_pair * 2 + part;
                    auto a_sub = a_smem.template subtile<16, 128>({m_block, sx128 / 8});
                    if (lane == 0) {
                        mm_AtB(accum_block, a_sub, h_smem);
                        kittens::detail::tcgen05::commit<1>(compute_done[local_epi_warp]);
                    }
                    wait(compute_done[local_epi_warp], compute_phase);
                    compute_phase ^= 1;
                    #pragma unroll
                    for (int k_slice = 0; k_slice < 8; ++k_slice) {
                        auto tm_sub = accum_block.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                        warp::load_async(reg_frag, tm_sub);
                        tensor_load_wait();
                        group<1>::store(epi_smem[smem_base + part * 8 + k_slice], reg_frag);
                        __syncwarp();
                    }
                }

                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    uint2 packed0;
                    uint2 packed1;
                    const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                    const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                    quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                        epi_smem[smem_base + k_slice],
                        epi_smem[smem_base + 8 + k_slice],
                        packed0,
                        packed1,
                        col_scales_out,
                        col_amax_out,
                        cols,
                        global_k,
                        global_m_block32);
                    store_packed_16x32_tecopy_rte(
                        packed0,
                        packed1,
                        col_fp4_out,
                        rows,
                        global_k,
                        global_m_block32);
                    __syncwarp();
                }
            }
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool SCALE_RP = true, bool WITH_AMAX = false, bool BF16_ROUND = false>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_col_quant_kernel_persistent_reg2(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ col_fp4_out,
    uint8_t* __restrict__ col_scales_out,
    float* __restrict__ col_amax_out,
    int rows,
    int cols
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }

    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_m * tiles_k;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[TE_COPY16_EPILOGUE_WARPS][2];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr[TE_COPY16_EPILOGUE_WARPS * 2];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        init_semaphore(inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            init_semaphore(compute_done[i][0], 0, 1);
            init_semaphore(compute_done[i][1], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h16_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS * 2; ++i) {
            auto acc = tm_alloc.template allocate<full_accum16_tt>(i * 16);
            accum_addr[i] = acc.addr;
        }
        warp::arrive(accum_ready);
    }
    wait(accum_ready, 0);

    int input_phase = 0;
    for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x) {
        const int bid_m = linear_tile % tiles_m;
        const int bid_k = linear_tile / tiles_m;
        if (threadIdx.x == 0) {
            tma::expect_bytes(inputs_arrived, sizeof(a512_tile));
            tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        }
        wait(inputs_arrived, input_phase);
        input_phase ^= 1;
        __syncthreads();

        if (warpgroup::groupid() == 1 || warpgroup::groupid() == 2) {
            const int local_warp = warpgroup::warpid();
            const int local_epi_warp = (warpgroup::groupid() - 1) * 4 + local_warp;
            full_accum16_tt accum0(accum_addr[local_epi_warp * 2 + 0]);
            full_accum16_tt accum1(accum_addr[local_epi_warp * 2 + 1]);
            lane_accum16_rt reg0;
            lane_accum16_rt reg1;
            int compute_phase = 0;
            #pragma unroll
            for (int task = local_epi_warp; task < 16; task += TE_COPY16_EPILOGUE_WARPS) {
                const int m_pair = task & 3;
                const int sx128 = (task >> 2) * 8;
                auto a_sub0 = a_smem.template subtile<16, 128>({m_pair * 2 + 0, sx128 / 8});
                auto a_sub1 = a_smem.template subtile<16, 128>({m_pair * 2 + 1, sx128 / 8});
                if (lane == 0) {
                    mm_AtB(accum0, a_sub0, h_smem);
                    kittens::detail::tcgen05::commit<1>(compute_done[local_epi_warp][0]);
                    mm_AtB(accum1, a_sub1, h_smem);
                    kittens::detail::tcgen05::commit<1>(compute_done[local_epi_warp][1]);
                }
                wait(compute_done[local_epi_warp][0], compute_phase);
                wait(compute_done[local_epi_warp][1], compute_phase);
                compute_phase ^= 1;

                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    auto tm_sub0 = accum0.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                    auto tm_sub1 = accum1.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                    warp::load_async(reg0, tm_sub0);
                    warp::load_async(reg1, tm_sub1);
                    tensor_load_wait();
                    uint2 packed0;
                    uint2 packed1;
                    const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                    const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                    quantize_reg_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX, BF16_ROUND>(
                        reg0,
                        reg1,
                        packed0,
                        packed1,
                        col_scales_out,
                        col_amax_out,
                        cols,
                        global_k,
                        global_m_block32);
                    store_packed_16x32_tecopy_reg_rte(
                        packed0,
                        packed1,
                        col_fp4_out,
                        rows,
                        global_k,
                        global_m_block32);
                    __syncwarp();
                }
            }
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<
    bool SCALE_RP = true,
    bool WITH_AMAX = false,
    bool SKIP_QUANT_PACK = false,
    bool BF16X8_QUANT = false,
    bool COMPACT_WARPS = false>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_col_quant_kernel_persistent_pair_smem(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ col_fp4_out,
    uint8_t* __restrict__ col_scales_out,
    float* __restrict__ col_amax_out,
    int rows,
    int cols
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }

    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_m * tiles_k;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    d16x16_tile (&epi_smem)[TE_COPY16_EPILOGUE_WARPS * 2] =
        al.allocate<d16x16_tile, TE_COPY16_EPILOGUE_WARPS * 2>();
    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[TE_COPY16_EPILOGUE_WARPS][2];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr[TE_COPY16_EPILOGUE_WARPS * 2];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        init_semaphore(inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            init_semaphore(compute_done[i][0], 0, 1);
            init_semaphore(compute_done[i][1], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h16_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == (COMPACT_WARPS ? 0 : 1) && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == (COMPACT_WARPS ? 0 : 1) && lane == 0) {
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS * 2; ++i) {
            auto acc = tm_alloc.template allocate<full_accum16_tt>(i * 16);
            accum_addr[i] = acc.addr;
        }
        warp::arrive(accum_ready);
    }
    wait(accum_ready, 0);

    int input_phase = 0;
    for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x) {
        const int bid_m = linear_tile % tiles_m;
        const int bid_k = linear_tile / tiles_m;
        if (threadIdx.x == 0) {
            tma::expect_bytes(inputs_arrived, sizeof(a512_tile));
            tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        }
        wait(inputs_arrived, input_phase);
        input_phase ^= 1;
        __syncthreads();

        const bool active_col_group = COMPACT_WARPS
            ? (warpgroup::groupid() == 0 || warpgroup::groupid() == 1)
            : (warpgroup::groupid() == 1 || warpgroup::groupid() == 2);
        if (active_col_group) {
            const int local_warp = warpgroup::warpid();
            const int local_epi_warp =
                ((COMPACT_WARPS ? warpgroup::groupid() : (warpgroup::groupid() - 1)) * 4) + local_warp;
            const int smem_base = local_epi_warp * 2;
            full_accum16_tt accum0(accum_addr[local_epi_warp * 2 + 0]);
            full_accum16_tt accum1(accum_addr[local_epi_warp * 2 + 1]);
            lane_accum16_rt reg0;
            lane_accum16_rt reg1;
            int compute_phase = 0;
            #pragma unroll
            for (int task = local_epi_warp; task < 16; task += TE_COPY16_EPILOGUE_WARPS) {
                const int m_pair = task & 3;
                const int sx128 = (task >> 2) * 8;
                auto a_sub0 = a_smem.template subtile<16, 128>({m_pair * 2 + 0, sx128 / 8});
                auto a_sub1 = a_smem.template subtile<16, 128>({m_pair * 2 + 1, sx128 / 8});
                if (lane == 0) {
                    mm_AtB(accum0, a_sub0, h_smem);
                    kittens::detail::tcgen05::commit<1>(compute_done[local_epi_warp][0]);
                    mm_AtB(accum1, a_sub1, h_smem);
                    kittens::detail::tcgen05::commit<1>(compute_done[local_epi_warp][1]);
                }
                wait(compute_done[local_epi_warp][0], compute_phase);
                wait(compute_done[local_epi_warp][1], compute_phase);
                compute_phase ^= 1;

                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    auto tm_sub0 = accum0.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                    auto tm_sub1 = accum1.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                    warp::load_async(reg0, tm_sub0);
                    warp::load_async(reg1, tm_sub1);
                    tensor_load_wait();
                    group<1>::store(epi_smem[smem_base + 0], reg0);
                    group<1>::store(epi_smem[smem_base + 1], reg1);
                    __syncwarp();

                    if constexpr (!SKIP_QUANT_PACK) {
                        uint2 packed0;
                        uint2 packed1;
                        const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                        const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                        if constexpr (BF16X8_QUANT) {
                            quantize_tiles_16x32_tecopy_pack_scale_bf16x8<SCALE_RP, WITH_AMAX>(
                                epi_smem[smem_base + 0],
                                epi_smem[smem_base + 1],
                                packed0,
                                packed1,
                                col_scales_out,
                                col_amax_out,
                                cols,
                                global_k,
                                global_m_block32);
                        } else {
                            quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                                epi_smem[smem_base + 0],
                                epi_smem[smem_base + 1],
                                packed0,
                                packed1,
                                col_scales_out,
                                col_amax_out,
                                cols,
                                global_k,
                                global_m_block32);
                        }
                        store_packed_16x32_tecopy_rte_noalloc(
                            packed0,
                            packed1,
                            col_fp4_out,
                            rows,
                            global_k,
                            global_m_block32);
                    }
                    __syncwarp();
                }
            }
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == (COMPACT_WARPS ? 0 : 1) && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool SCALE_RP = true, bool WITH_AMAX = false, bool DO_ROW_QUANT = false>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_col_quant_kernel_persistent_producer(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ row_fp4_out,
    uint8_t* __restrict__ row_scales_out,
    float* __restrict__ row_amax_out,
    uint8_t* __restrict__ col_fp4_out,
    uint8_t* __restrict__ col_scales_out,
    float* __restrict__ col_amax_out,
    int rows,
    int cols
) {
    // TE-like scheduling: warp 0 performs TMA, warp 1 issues all col-RHT MMA,
    // warpgroup 1 quantizes column output, and optionally warpgroups 2/3 emit
    // TE-equivalent raw row quantization from the same 128x512 A tile.
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }

    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_m * tiles_k;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    d16x16_tile (&epi_smem)[TE_COPY16_COL_PRODUCER_WARPS * 2] =
        al.allocate<d16x16_tile, TE_COPY16_COL_PRODUCER_WARPS * 2>();

    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[TE_COPY16_COL_PRODUCER_PIPE][2];
    __shared__ semaphore outputs_finished[TE_COPY16_COL_PRODUCER_PIPE];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr[TE_COPY16_COL_PRODUCER_PIPE * 2];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        init_semaphore(inputs_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < TE_COPY16_COL_PRODUCER_PIPE; ++stage) {
            init_semaphore(compute_done[stage][0], 0, 1);
            init_semaphore(compute_done[stage][1], 0, 1);
            init_semaphore(outputs_finished[stage], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h16_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warp_id == 1) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int i = 0; i < TE_COPY16_COL_PRODUCER_PIPE * 2; ++i) {
            auto acc = tm_alloc.template allocate<full_accum16_tt>(i * 16);
            accum_addr[i] = acc.addr;
        }
        warp::arrive(accum_ready);
    }

    int input_phase = 0;
    for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x) {
        const int bid_m = linear_tile % tiles_m;
        const int bid_k = linear_tile / tiles_m;
        if (threadIdx.x == 0) {
            tma::expect_bytes(inputs_arrived, sizeof(a512_tile));
            tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        }
        wait(inputs_arrived, input_phase);
        input_phase ^= 1;
        __syncthreads();

        if (warp_id == 1) {
            wait(accum_ready, 0);
            #pragma unroll
            for (int task = 0; task < 16; ++task) {
                const int global_task = linear_tile * 16 + task;
                const int slot = task % TE_COPY16_COL_PRODUCER_PIPE;
                if (global_task >= TE_COPY16_COL_PRODUCER_PIPE && lane == 0) {
                    wait(outputs_finished[slot],
                         ((global_task / TE_COPY16_COL_PRODUCER_PIPE) - 1) & 1);
                }
                const int m_pair = task & 3;
                const int sx128 = (task >> 2) * 8;
                full_accum16_tt accum0(accum_addr[slot * 2 + 0]);
                full_accum16_tt accum1(accum_addr[slot * 2 + 1]);
                auto a_sub0 = a_smem.template subtile<16, 128>({m_pair * 2 + 0, sx128 / 8});
                auto a_sub1 = a_smem.template subtile<16, 128>({m_pair * 2 + 1, sx128 / 8});
                if (lane == 0) {
                    mm_AtB(accum0, a_sub0, h_smem);
                    kittens::detail::tcgen05::commit<1>(compute_done[slot][0]);
                    mm_AtB(accum1, a_sub1, h_smem);
                    kittens::detail::tcgen05::commit<1>(compute_done[slot][1]);
                }
            }
        }

        if constexpr (DO_ROW_QUANT) {
            if (warpgroup::groupid() == 2 || warpgroup::groupid() == 3) {
                const int local_warp = warpgroup::warpid();
                const int row_warp = (warpgroup::groupid() - 2) * 4 + local_warp;
                #pragma unroll
                for (int sx_base = 0; sx_base < CHUNK_K_WIDE / TILE_N_TE; sx_base += TE_COPY16_STORE_GROUP) {
                    uint2 packed_words[TE_COPY16_STORE_GROUP];
                    #pragma unroll
                    for (int g = 0; g < TE_COPY16_STORE_GROUP; g += 2) {
                        auto tile0 = a_smem.template subtile<16, 16>({row_warp, sx_base + g});
                        auto tile1 = a_smem.template subtile<16, 16>({row_warp, sx_base + g + 1});
                        quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                            tile0,
                            tile1,
                            packed_words[g],
                            packed_words[g + 1],
                            row_scales_out,
                            row_amax_out,
                            rows,
                            bid_m * CHUNK_M + row_warp * 16,
                            (bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base + g) / 2);
                    }
                    store_packed_16x64_tecopy_rte(
                        packed_words,
                        row_fp4_out,
                        cols,
                        bid_m * CHUNK_M + row_warp * 16,
                        bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base);
                }
            }
        }

        if (warpgroup::groupid() == 1) {
            wait(accum_ready, 0);
            const int local_col_warp = warpgroup::warpid();
            const int smem_base = local_col_warp * 2;
            lane_accum16_rt reg0;
            lane_accum16_rt reg1;
            #pragma unroll
            for (int task = local_col_warp; task < 16; task += TE_COPY16_COL_PRODUCER_WARPS) {
                const int global_task = linear_tile * 16 + task;
                const int slot = task % TE_COPY16_COL_PRODUCER_PIPE;
                const int compute_phase = (global_task / TE_COPY16_COL_PRODUCER_PIPE) & 1;
                wait(compute_done[slot][0], compute_phase);
                wait(compute_done[slot][1], compute_phase);
                const int m_pair = task & 3;
                const int sx128 = (task >> 2) * 8;
                full_accum16_tt accum0(accum_addr[slot * 2 + 0]);
                full_accum16_tt accum1(accum_addr[slot * 2 + 1]);
                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    auto tm_sub0 = accum0.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                    auto tm_sub1 = accum1.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                    warp::load_async(reg0, tm_sub0);
                    warp::load_async(reg1, tm_sub1);
                    tensor_load_wait();
                    group<1>::store(epi_smem[smem_base + 0], reg0);
                    group<1>::store(epi_smem[smem_base + 1], reg1);
                    __syncwarp();

                    uint2 packed0;
                    uint2 packed1;
                    const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                    const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                    quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                        epi_smem[smem_base + 0],
                        epi_smem[smem_base + 1],
                        packed0,
                        packed1,
                        col_scales_out,
                        col_amax_out,
                        cols,
                        global_k,
                        global_m_block32);
                    store_packed_16x32_tecopy_rte(
                        packed0,
                        packed1,
                        col_fp4_out,
                        rows,
                        global_k,
                        global_m_block32);
                    __syncwarp();
                }
                if (lane == 0) {
                    warp::arrive(outputs_finished[slot]);
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 1) {
        tm_alloc.deprovision();
    }
}

template<bool SCALE_RP = true, bool WITH_AMAX = false, bool ROW_RHT = false, bool COL_RHT = true>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_rowcol_quant_kernel(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ row_fp4_out,
    uint8_t* __restrict__ row_scales_out,
    float* __restrict__ row_amax_out,
    uint8_t* __restrict__ col_fp4_out,
    uint8_t* __restrict__ col_scales_out,
    float* __restrict__ col_amax_out,
    int rows,
    int cols,
    bool swizzled_scale
) {
    // Benchmark-only fused row+col path. The CTA owns one 128x512 input tile.
    // ROW_RHT=false,COL_RHT=true matches TE's row+col fusion: raw row quant plus col RHT.
    // ROW_RHT=true,COL_RHT=false matches the MXFP4 policy: row-contract A*H plus raw col quant.
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }

    const int bid_k = blockIdx.x;
    const int bid_m = blockIdx.y;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    d16x16_tile (&epi_smem)[TE_COPY16_EPILOGUE_WARPS * 16] =
        al.allocate<d16x16_tile, TE_COPY16_EPILOGUE_WARPS * 16>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore row_compute_done[TE_COPY16_PIPE_STAGES][2];
    __shared__ semaphore col_compute_done[TE_COPY16_EPILOGUE_WARPS];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t row_accum_addr[TE_COPY16_PIPE_STAGES];
    __shared__ uint32_t col_accum_addr[TE_COPY16_EPILOGUE_WARPS];
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < TE_COPY16_PIPE_STAGES; ++stage) {
            init_semaphore(row_compute_done[stage][0], 0, 1);
            init_semaphore(row_compute_done[stage][1], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            init_semaphore(col_compute_done[i], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a512_tile) + sizeof(h16_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(inputs_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int stage = 0; stage < TE_COPY16_PIPE_STAGES; ++stage) {
            auto acc = tm_alloc.template allocate<full_accum_tt>(stage * CHUNK_K);
            row_accum_addr[stage] = acc.addr;
        }
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            auto acc = tm_alloc.template allocate<full_accum16_tt>(i * 16);
            col_accum_addr[i] = acc.addr;
        }
        warp::arrive(accum_ready);
    }
    wait(accum_ready, 0);

    if constexpr (ROW_RHT) {
        if (warp_id == 1) {
            #pragma unroll
            for (int stage = 0; stage < TE_COPY16_PIPE_STAGES; ++stage) {
                full_accum_tt accum(row_accum_addr[stage]);
                #pragma unroll
                for (int sx = 0; sx < CHUNK_K / TILE_N_TE; ++sx) {
                    auto a_sub = a_smem.template subtile<128, 16>({0, stage * (CHUNK_K / TILE_N_TE) + sx});
                    if (lane == 0) {
                        auto acc_sub = accum.template subtile<full_accum16_tt>(0, sx * 16);
                        mm_AB(acc_sub, a_sub, h_smem);
                    }
                }
                if (lane == 0) {
                    kittens::detail::tcgen05::commit<1>(row_compute_done[stage][0]);
                    kittens::detail::tcgen05::commit<1>(row_compute_done[stage][1]);
                }
            }
        }
    }

    if (warpgroup::groupid() == 1 || warpgroup::groupid() == 2) {
        const int local_warp = warpgroup::warpid();
        const int local_epi_warp = (warpgroup::groupid() - 1) * 4 + local_warp;
        const int sy = warpgroup::groupid() - 1;
        lane_accum16_rt reg_frag;
        if constexpr (ROW_RHT) {
            #pragma unroll
            for (int stage = 0; stage < TE_COPY16_PIPE_STAGES; ++stage) {
                wait(row_compute_done[stage][sy], 0);
                full_accum_tt accum(row_accum_addr[stage]);
                const bool pack_swizzled_scales = swizzled_scale && row_scales_out != nullptr;
                uint32_t swizzled_scale_word = 0;
                #pragma unroll
                for (int sx_base = 0; sx_base < CHUNK_K / TILE_N_TE; sx_base += TE_COPY16_STORE_GROUP) {
                    uint2 packed_words[TE_COPY16_STORE_GROUP];
                    #pragma unroll
                    for (int g = 0; g < TE_COPY16_STORE_GROUP; g += 2) {
                        constexpr int smem_stride = 2;
                        const int smem_base = local_epi_warp * smem_stride;
                        auto tm_sub0 = accum.template subtile<lane_accum16_tt>(
                            sy * 64 + local_warp * 16,
                            (sx_base + g) * 16);
                        warp::load_async(reg_frag, tm_sub0);
                        tensor_load_wait();
                        group<1>::store(epi_smem[smem_base], reg_frag);

                        auto tm_sub1 = accum.template subtile<lane_accum16_tt>(
                            sy * 64 + local_warp * 16,
                            (sx_base + g + 1) * 16);
                        warp::load_async(reg_frag, tm_sub1);
                        tensor_load_wait();
                        group<1>::store(epi_smem[smem_base + 1], reg_frag);

                        const int sx_global = stage * (CHUNK_K / TILE_N_TE) + sx_base + g;
                        const int tile_col_block32 = (bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_global) / 2;
                        const uint8_t e8m0_val = quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                            epi_smem[smem_base],
                            epi_smem[smem_base + 1],
                            packed_words[g],
                            packed_words[g + 1],
                            pack_swizzled_scales ? nullptr : row_scales_out,
                            row_amax_out,
                            rows,
                            bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                            tile_col_block32,
                            cols / CHUNK_DIM,
                            swizzled_scale);
                        if (pack_swizzled_scales && ((lane & 1) == 0)) {
                            swizzled_scale_word |= static_cast<uint32_t>(e8m0_val) << ((tile_col_block32 & 3) * 8);
                        }
                    }
                    const int sx_global = stage * (CHUNK_K / TILE_N_TE) + sx_base;
                    store_packed_16x64_tecopy_rte(
                        packed_words,
                        row_fp4_out,
                        cols,
                        bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                        bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_global);
                }
                if (pack_swizzled_scales && ((lane & 1) == 0)) {
                    const int global_row = bid_m * CHUNK_M + sy * 64 + local_warp * 16 + (lane >> 1);
                    const int base_tile_col_block32 =
                        (bid_k * (CHUNK_K_WIDE / TILE_N_TE) + stage * (CHUNK_K / TILE_N_TE)) / 2;
                    const int ctaid_X = base_tile_col_block32 / 4;
                    const int row_in_chunk = global_row & 127;
                    const int j = row_in_chunk & 31;
                    const int grp = row_in_chunk >> 5;
                    const int ctaid_Y = global_row >> 7;
                    uint8_t* scale_ptr = row_scales_out
                        + (ctaid_Y * (cols / CHUNK_DIM) + ctaid_X) * 512
                        + j * 16 + grp * 4;
                    *reinterpret_cast<uint32_t*>(scale_ptr) = swizzled_scale_word;
                }
            }
        } else {
            #pragma unroll
            for (int sx_base = 0; sx_base < CHUNK_K_WIDE / TILE_N_TE; sx_base += TE_COPY16_STORE_GROUP) {
                uint2 packed_words[TE_COPY16_STORE_GROUP];
                #pragma unroll
                for (int g = 0; g < TE_COPY16_STORE_GROUP; g += 2) {
                    const int tile_row_block16 = sy * 4 + local_warp;
                    auto tile0 = a_smem.template subtile<16, 16>({tile_row_block16, sx_base + g});
                    auto tile1 = a_smem.template subtile<16, 16>({tile_row_block16, sx_base + g + 1});
                    quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                        tile0,
                        tile1,
                        packed_words[g],
                        packed_words[g + 1],
                        row_scales_out,
                        row_amax_out,
                        rows,
                        bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                        (bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base + g) / 2,
                        cols / CHUNK_DIM,
                        swizzled_scale);
                }
                store_packed_16x64_tecopy_rte(
                    packed_words,
                    row_fp4_out,
                    cols,
                    bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                    bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base);
            }
        }
    }

    __syncthreads();

    if (warpgroup::groupid() == 1 || warpgroup::groupid() == 2) {
        const int local_warp = warpgroup::warpid();
        const int local_epi_warp = (warpgroup::groupid() - 1) * 4 + local_warp;
        const int smem_base = local_epi_warp * 16;
        full_accum16_tt accum_block(col_accum_addr[local_epi_warp]);
        lane_accum16_rt reg_frag;
        int compute_phase = 0;
        #pragma unroll
        for (int task = local_epi_warp; task < 16; task += TE_COPY16_EPILOGUE_WARPS) {
            const int m_pair = task & 3;
            const int sx128 = (task >> 2) * 8;
            if constexpr (COL_RHT) {
                #pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const int m_block = m_pair * 2 + part;
                    auto a_sub = a_smem.template subtile<16, 128>({m_block, sx128 / 8});
                    if (lane == 0) {
                        mm_AtB(accum_block, a_sub, h_smem);
                        kittens::detail::tcgen05::commit<1>(col_compute_done[local_epi_warp]);
                    }
                    wait(col_compute_done[local_epi_warp], compute_phase);
                    compute_phase ^= 1;
                    #pragma unroll
                    for (int k_slice = 0; k_slice < 8; ++k_slice) {
                        auto tm_sub = accum_block.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                        warp::load_async(reg_frag, tm_sub);
                        tensor_load_wait();
                        group<1>::store(epi_smem[smem_base + part * 8 + k_slice], reg_frag);
                        __syncwarp();
                    }
                }

                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    uint2 packed0;
                    uint2 packed1;
                    const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                    const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                    quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                        epi_smem[smem_base + k_slice],
                        epi_smem[smem_base + 8 + k_slice],
                        packed0,
                        packed1,
                        col_scales_out,
                        col_amax_out,
                        cols,
                        global_k,
                        global_m_block32,
                        rows / CHUNK_DIM,
                        swizzled_scale);
                    store_packed_16x32_tecopy_rte(
                        packed0,
                        packed1,
                        col_fp4_out,
                        rows,
                        global_k,
                        global_m_block32);
                    __syncwarp();
                }
            } else {
                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    uint2 packed0;
                    uint2 packed1;
                    const int local_k = (sx128 + k_slice) * TILE_N_TE;
                    const int global_k = bid_k * CHUNK_K_WIDE + local_k;
                    const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                    quantize_a512_transposed_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                        a_smem,
                        packed0,
                        packed1,
                        col_scales_out,
                        col_amax_out,
                        cols,
                        global_k,
                        global_m_block32,
                        m_pair * 32,
                        local_k,
                        rows / CHUNK_DIM,
                        swizzled_scale);
                    store_packed_16x32_tecopy_rte(
                        packed0,
                        packed1,
                        col_fp4_out,
                        rows,
                        global_k,
                        global_m_block32);
                    __syncwarp();
                }
            }
        }
    }

    __syncthreads();
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<
    bool SCALE_RP = true,
    bool WITH_AMAX = false,
    bool PAIR_SMEM = false,
    bool ROW_BF16X8_QUANT = false,
    bool COL_BF16X8_QUANT = false,
    bool REG_SPLIT = true>
__global__ __launch_bounds__(TE_COPY16_NUM_THREADS, 1)
void te_copy16_rowcol_quant_kernel_persistent(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h16_gl H_layout,
    uint8_t* __restrict__ row_fp4_out,
    uint8_t* __restrict__ row_scales_out,
    float* __restrict__ row_amax_out,
    uint8_t* __restrict__ col_fp4_out,
    uint8_t* __restrict__ col_scales_out,
    float* __restrict__ col_amax_out,
    int rows,
    int cols
) {
    // TE-style benchmark path: persistent CTAs, H loaded once, raw row quant
    // overlaps with tensor-core col-RHT quant from the same 128x512 A tile.
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h16_tile>();
    }

    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_m * tiles_k;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h16_tile (&h_smem) = al.allocate<h16_tile>();
    d16x16_tile (&epi_smem)[TE_COPY16_EPILOGUE_WARPS * (PAIR_SMEM ? 2 : 16)] =
        al.allocate<d16x16_tile, TE_COPY16_EPILOGUE_WARPS * (PAIR_SMEM ? 2 : 16)>();

    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore col_compute_done[TE_COPY16_EPILOGUE_WARPS][2];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t col_accum_addr[TE_COPY16_EPILOGUE_WARPS * 2];
    __shared__ semaphore accum_ready;
    #if defined(TCBENCH_ROWCOL_BLOCK_PAIR64_STORE)
    __shared__ uint4 col_pair64_block
        [PAIR_SMEM ? 2 : 1][PAIR_SMEM ? 4 : 1][PAIR_SMEM ? 4 : 1][PAIR_SMEM ? 8 : 1][PAIR_SMEM ? 16 : 1];
    #endif

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        init_semaphore(inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS; ++i) {
            init_semaphore(col_compute_done[i][0], 0, 1);
            init_semaphore(col_compute_done[i][1], 0, 1);
        }
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h16_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    if (warp_id == 1 && lane == 0) {
        #pragma unroll
        for (int i = 0; i < TE_COPY16_EPILOGUE_WARPS * (PAIR_SMEM ? 2 : 1); ++i) {
            auto acc = tm_alloc.template allocate<full_accum16_tt>(i * 16);
            col_accum_addr[i] = acc.addr;
        }
        warp::arrive(accum_ready);
    }
    wait(accum_ready, 0);

    #if !defined(TCBENCH_ROWCOL_NO_REG_SPLIT)
    if constexpr (REG_SPLIT) {
        if (warpgroup::groupid() == 0 || warpgroup::groupid() == 3) {
            warpgroup::increase_registers<TCBENCH_ROWCOL_ROW_REGS>();
        } else {
            warpgroup::decrease_registers<TCBENCH_ROWCOL_COL_REGS>();
        }
        __syncthreads();
    }
    #endif

    int input_phase = 0;
    for (int linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x) {
        const int bid_m = linear_tile % tiles_m;
        const int bid_k = linear_tile / tiles_m;
        if (threadIdx.x == 0) {
            tma::expect_bytes(inputs_arrived, sizeof(a512_tile));
            tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        }
        wait(inputs_arrived, input_phase);
        input_phase ^= 1;
        const int pair_buffer = input_phase;
        #if defined(TCBENCH_ROWCOL_FORCE_INPUT_SYNC)
        __syncthreads();
        #endif

        if (warpgroup::groupid() == 0 || warpgroup::groupid() == 3) {
            const int local_warp = warpgroup::warpid();
            const int row_warp = (warpgroup::groupid() == 0) ? local_warp : 4 + local_warp;
            #pragma unroll
            for (int sx_base = 0; sx_base < CHUNK_K_WIDE / TILE_N_TE; sx_base += TE_COPY16_STORE_GROUP) {
                uint2 packed_words[TE_COPY16_STORE_GROUP];
                #pragma unroll
                for (int g = 0; g < TE_COPY16_STORE_GROUP; g += 2) {
                    auto tile0 = a_smem.template subtile<16, 16>({row_warp, sx_base + g});
                    auto tile1 = a_smem.template subtile<16, 16>({row_warp, sx_base + g + 1});
                    if constexpr (ROW_BF16X8_QUANT) {
                        quantize_tiles_16x32_tecopy_pack_scale_bf16x8<SCALE_RP, WITH_AMAX>(
                            tile0,
                            tile1,
                            packed_words[g],
                            packed_words[g + 1],
                            row_scales_out,
                            row_amax_out,
                            rows,
                            bid_m * CHUNK_M + row_warp * 16,
                            (bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base + g) / 2);
                    } else {
                        quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                            tile0,
                            tile1,
                            packed_words[g],
                            packed_words[g + 1],
                            row_scales_out,
                            row_amax_out,
                            rows,
                            bid_m * CHUNK_M + row_warp * 16,
                            (bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base + g) / 2);
                    }
                }
                if constexpr (PAIR_SMEM) {
                    store_packed_16x64_tecopy_rte_noalloc(
                        packed_words,
                        row_fp4_out,
                        cols,
                        bid_m * CHUNK_M + row_warp * 16,
                        bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base);
                } else {
                    store_packed_16x64_tecopy_rte(
                            packed_words,
                            row_fp4_out,
                            cols,
                            bid_m * CHUNK_M + row_warp * 16,
                            bid_k * (CHUNK_K_WIDE / TILE_N_TE) + sx_base);
                }
            }
        }

        if (warpgroup::groupid() == 1 || warpgroup::groupid() == 2) {
            const int local_warp = warpgroup::warpid();
            const int local_epi_warp = (warpgroup::groupid() - 1) * 4 + local_warp;
            int compute_phase = 0;
            if constexpr (PAIR_SMEM) {
                const int smem_base = local_epi_warp * 2;
                full_accum16_tt accum0(col_accum_addr[local_epi_warp * 2 + 0]);
                full_accum16_tt accum1(col_accum_addr[local_epi_warp * 2 + 1]);
                lane_accum16_rt reg0;
                lane_accum16_rt reg1;
                #pragma unroll
                for (int task = local_epi_warp; task < 16; task += TE_COPY16_EPILOGUE_WARPS) {
                    const int m_pair = task & 3;
                    const int sx128 = (task >> 2) * 8;
                    auto a_sub0 = a_smem.template subtile<16, 128>({m_pair * 2 + 0, sx128 / 8});
                    auto a_sub1 = a_smem.template subtile<16, 128>({m_pair * 2 + 1, sx128 / 8});
                    if (lane == 0) {
                        mm_AtB(accum0, a_sub0, h_smem);
                        kittens::detail::tcgen05::commit<1>(col_compute_done[local_epi_warp][0]);
                        mm_AtB(accum1, a_sub1, h_smem);
                        kittens::detail::tcgen05::commit<1>(col_compute_done[local_epi_warp][1]);
                    }
                    wait(col_compute_done[local_epi_warp][0], compute_phase);
                    wait(col_compute_done[local_epi_warp][1], compute_phase);
                    compute_phase ^= 1;

                    #pragma unroll
                    for (int k_slice = 0; k_slice < 8; ++k_slice) {
                        auto tm_sub0 = accum0.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                        auto tm_sub1 = accum1.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                        warp::load_async(reg0, tm_sub0);
                        warp::load_async(reg1, tm_sub1);
                        tensor_load_wait();
                        group<1>::store(epi_smem[smem_base + 0], reg0);
                        group<1>::store(epi_smem[smem_base + 1], reg1);
                        __syncwarp();

                        uint2 packed0;
                        uint2 packed1;
                        const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                        const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                        if constexpr (COL_BF16X8_QUANT) {
                            quantize_tiles_16x32_tecopy_pack_scale_bf16x8<SCALE_RP, WITH_AMAX>(
                                epi_smem[smem_base + 0],
                                epi_smem[smem_base + 1],
                                packed0,
                                packed1,
                                col_scales_out,
                                col_amax_out,
                                cols,
                                global_k,
                                global_m_block32);
                        } else {
                            quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                                epi_smem[smem_base + 0],
                                epi_smem[smem_base + 1],
                                packed0,
                                packed1,
                                col_scales_out,
                                col_amax_out,
                                cols,
                                global_k,
                                global_m_block32);
                        }
                        #if defined(TCBENCH_ROWCOL_BLOCK_PAIR64_STORE)
                        if ((lane & 1) == 0 && col_fp4_out != nullptr) {
                            const int sx_group = sx128 >> 3;
                            col_pair64_block[pair_buffer][sx_group][m_pair][k_slice][lane >> 1] =
                                uint4{packed0.x, packed0.y, packed1.x, packed1.y};
                        }
                        #else
                        store_packed_16x32_tecopy_rte_noalloc(
                            packed0,
                            packed1,
                            col_fp4_out,
                            rows,
                            global_k,
                            global_m_block32);
                        __syncwarp();
                        #endif
                    }
                }
            } else {
                const int smem_base = local_epi_warp * 16;
                full_accum16_tt accum_block(col_accum_addr[local_epi_warp]);
                lane_accum16_rt reg_frag;
                #pragma unroll
                for (int task = local_epi_warp; task < 16; task += TE_COPY16_EPILOGUE_WARPS) {
                    const int m_pair = task & 3;
                    const int sx128 = (task >> 2) * 8;
                    #pragma unroll
                    for (int part = 0; part < 2; ++part) {
                        const int m_block = m_pair * 2 + part;
                        auto a_sub = a_smem.template subtile<16, 128>({m_block, sx128 / 8});
                        if (lane == 0) {
                            mm_AtB(accum_block, a_sub, h_smem);
                            kittens::detail::tcgen05::commit<1>(col_compute_done[local_epi_warp][0]);
                        }
                        wait(col_compute_done[local_epi_warp][0], compute_phase);
                        compute_phase ^= 1;
                        #pragma unroll
                        for (int k_slice = 0; k_slice < 8; ++k_slice) {
                            auto tm_sub = accum_block.template subtile<lane_accum16_tt>(k_slice * 16, 0);
                            warp::load_async(reg_frag, tm_sub);
                            tensor_load_wait();
                            group<1>::store(epi_smem[smem_base + part * 8 + k_slice], reg_frag);
                            __syncwarp();
                        }
                    }

                    #pragma unroll
                    for (int k_slice = 0; k_slice < 8; ++k_slice) {
                        uint2 packed0;
                        uint2 packed1;
                        const int global_k = bid_k * CHUNK_K_WIDE + (sx128 + k_slice) * TILE_N_TE;
                        const int global_m_block32 = (bid_m * CHUNK_M + m_pair * 32) / 32;
                        quantize_tiles_16x32_tecopy_pack_scale<SCALE_RP, WITH_AMAX>(
                            epi_smem[smem_base + k_slice],
                            epi_smem[smem_base + 8 + k_slice],
                            packed0,
                            packed1,
                            col_scales_out,
                            col_amax_out,
                            cols,
                            global_k,
                            global_m_block32);
                        store_packed_16x32_tecopy_rte(
                            packed0,
                            packed1,
                            col_fp4_out,
                            rows,
                            global_k,
                            global_m_block32);
                        __syncwarp();
                    }
                }
            }
        }
        #if defined(TCBENCH_ROWCOL_BLOCK_PAIR64_STORE)
        if constexpr (PAIR_SMEM) {
            __syncthreads();
            if ((warpgroup::groupid() == 1 || warpgroup::groupid() == 2) && col_fp4_out != nullptr) {
                const int local_warp = warpgroup::warpid();
                const int local_epi_warp = (warpgroup::groupid() - 1) * 4 + local_warp;
                const int sx_group = local_epi_warp >> 1;
                const int m_pair_base = (local_epi_warp & 1) * 2;
                #pragma unroll
                for (int k_slice = 0; k_slice < 8; ++k_slice) {
                    if ((lane & 1) == 0) {
                        const int store_row = lane >> 1;
                        const uint4 first = col_pair64_block[pair_buffer][sx_group][m_pair_base][k_slice][store_row];
                        const uint4 second = col_pair64_block[pair_buffer][sx_group][m_pair_base + 1][k_slice][store_row];
                        const int global_k = bid_k * CHUNK_K_WIDE + (sx_group * 8 + k_slice) * TILE_N_TE;
                        const int global_m_block32 = (bid_m * CHUNK_M + m_pair_base * 32) / 32;
                        uint8_t* row_ptr = col_fp4_out
                            + static_cast<int64_t>(global_k + store_row) * (rows / 2)
                            + global_m_block32 * 16;
                        const unsigned long long v0 =
                            (static_cast<unsigned long long>(first.y) << 32) | first.x;
                        const unsigned long long v1 =
                            (static_cast<unsigned long long>(first.w) << 32) | first.z;
                        const unsigned long long v2 =
                            (static_cast<unsigned long long>(second.y) << 32) | second.x;
                        const unsigned long long v3 =
                            (static_cast<unsigned long long>(second.w) << 32) | second.z;
                        asm volatile(
                            "st.global.L1::no_allocate.v4.u64 [%0], {%1, %2, %3, %4};\n"
                            :
                            : "l"(row_ptr), "l"(v0), "l"(v1), "l"(v2), "l"(v3)
                            : "memory");
                    }
                }
            }
        }
        #endif
        #if !defined(TCBENCH_ROWCOL_BLOCK_PAIR64_STORE)
        __syncthreads();
        #else
        if constexpr (!PAIR_SMEM) {
            __syncthreads();
        }
        #endif
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
__global__ __launch_bounds__(NUM_THREADS, 1)
void te_style_direct32_quant_kernel_wide_reg_epilogue(
    const __grid_constant__ a512_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ amax_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a512_tile>();
        H_layout.template prefetch_tma<h32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.x;
    const int warp_id = warpid();
    const int lane = laneid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a512_tile (&a_smem) = al.allocate<a512_tile>();
    h32_tile (&h_smem) = al.allocate<h32_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore compute_done[2];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr;
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(compute_done[0], 0, 1);
        init_semaphore(compute_done[1], 0, 1);
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(a512_tile) + sizeof(h32_tile));
        tma::load_async(a_smem, A_layout, {bid_m, bid_k}, inputs_arrived);
        tma::load_async(h_smem, H_layout, {0, 0}, inputs_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(inputs_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    full_accum_tt_wide accum;
    if (warp_id == 1 && lane == 0) {
        accum = tm_alloc.template allocate<full_accum_tt_wide>(0);
        accum_addr = accum.addr;
        warp::arrive(accum_ready);
    }
    if (warp_id == 1 || warp_id == 2) {
        wait(accum_ready, 0);
        accum = full_accum_tt_wide(accum_addr);
        const int sy = warp_id - 1;
        #pragma unroll
        for (int sx = 0; sx < CHUNK_K_WIDE / TILE_N; ++sx) {
            auto a_sub = a_smem.template subtile<64, 32>({sy, sx});
            if (lane == 0) {
                auto acc_sub = accum.template subtile<block_accum_tt>(sy * 64, sx * 32);
                mm_AB(acc_sub, a_sub, h_smem);
            }
        }
        if (lane == 0) {
            kittens::detail::tcgen05::commit<1>(compute_done[sy]);
        }
    }

    wait(compute_done[0], 0);
    wait(compute_done[1], 0);

    if (warpgroup::groupid() == 1) {
        const int local_warp = warpgroup::warpid();
        lane_accum_rt reg_frag;
        #pragma unroll
        for (int sy = 0; sy < 2; ++sy) {
            #pragma unroll
            for (int sx = 0; sx < CHUNK_K_WIDE / TILE_N; ++sx) {
                auto tm_sub = accum.template subtile<lane_accum_tt>(sy * 64 + local_warp * 16, sx * 32);
                warp::load_async(reg_frag, tm_sub);
                tensor_load_wait();
                quantize_reg_tile_16x32<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>(
                    reg_frag,
                    fp4_out,
                    scales_out,
                    amax_out,
                    cols,
                    bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                    bid_k * (CHUNK_K_WIDE / TILE_N) + sx,
                    rng_seed,
                    rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k) * (2 * (CHUNK_K_WIDE / TILE_N)) + sy * (CHUNK_K_WIDE / TILE_N) + sx) * TILE_M + local_warp * 16
                );
            }
        }
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
float run_te_style_direct32_quant_impl(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    a128_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h32_gl H_layout{reinterpret_cast<bf16*>(bufs.h32), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    check(cudaFuncSetAttribute(
        te_style_direct32_quant_kernel<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te style quant");

    auto launch = [&]() {
        te_style_direct32_quant_kernel<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX><<<grid, NUM_THREADS, smem_size>>>(
            A_layout, H_layout, bufs.out_fp4, bufs.out_sc, WITH_AMAX ? bufs.out_amax : nullptr, rows, cols, 1234, 0);
    };

    check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4");
    check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc");
    if constexpr (WITH_AMAX) {
        check(cudaMemset(bufs.out_amax, 0, rows * (cols / 32) * sizeof(float)), "memset amax");
    }

    const int iters = bench_iters();
    for (int i = 0; i < 3; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "warmup sync");
    check(cudaGetLastError(), "warmup launch");
    const float ms = time_launch_ms(launch, "create start event", "create stop event", "timed sync", "timed launch");

    std::array<uint8_t, 32> sample{};
    check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample");
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum=" << checksum << "\n";
    return ms;
}

float run_te_style_direct32_quant(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr,
    bool with_random_sign_mask,
    bool with_amax
) {
    if (with_amax) {
        if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_impl<true, true, true, true>(bufs, rows, cols);
        if (data_sr && scale_sr) return run_te_style_direct32_quant_impl<true, true, false, true>(bufs, rows, cols);
        if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_impl<true, false, true, true>(bufs, rows, cols);
        if (data_sr) return run_te_style_direct32_quant_impl<true, false, false, true>(bufs, rows, cols);
        if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_impl<false, true, true, true>(bufs, rows, cols);
        if (scale_sr) return run_te_style_direct32_quant_impl<false, true, false, true>(bufs, rows, cols);
        if (with_random_sign_mask) return run_te_style_direct32_quant_impl<false, false, true, true>(bufs, rows, cols);
        return run_te_style_direct32_quant_impl<false, false, false, true>(bufs, rows, cols);
    }
    if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_impl<true, true, true, false>(bufs, rows, cols);
    if (data_sr && scale_sr) return run_te_style_direct32_quant_impl<true, true, false, false>(bufs, rows, cols);
    if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_impl<true, false, true, false>(bufs, rows, cols);
    if (data_sr) return run_te_style_direct32_quant_impl<true, false, false, false>(bufs, rows, cols);
    if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_impl<false, true, true, false>(bufs, rows, cols);
    if (scale_sr) return run_te_style_direct32_quant_impl<false, true, false, false>(bufs, rows, cols);
    if (with_random_sign_mask) return run_te_style_direct32_quant_impl<false, false, true, false>(bufs, rows, cols);
    return run_te_style_direct32_quant_impl<false, false, false, false>(bufs, rows, cols);
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
float run_te_style_direct32_quant_wide_impl(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (cols < CHUNK_K_WIDE || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a512_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h32_gl H_layout{reinterpret_cast<bf16*>(bufs.h32), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K_WIDE, rows / CHUNK_M);
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    check(cudaFuncSetAttribute(
        te_style_direct32_quant_kernel_wide<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te style quant wide");

    auto launch = [&]() {
        te_style_direct32_quant_kernel_wide<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX><<<grid, NUM_THREADS, smem_size>>>(
            A_layout, H_layout, bufs.out_fp4, bufs.out_sc, WITH_AMAX ? bufs.out_amax : nullptr, rows, cols, 1234, 0);
    };

    check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 wide");
    check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc wide");
    if constexpr (WITH_AMAX) {
        check(cudaMemset(bufs.out_amax, 0, rows * (cols / 32) * sizeof(float)), "memset amax wide");
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync wide");
    check(cudaGetLastError(), "warmup launch wide");
    const float ms = time_launch_ms(launch, "create start event wide", "create stop event wide", "timed sync wide", "timed launch wide");

    std::array<uint8_t, 32> sample{};
    check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample wide");
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum_wide=" << checksum << "\n";
    return ms;
}

float run_te_style_direct32_quant_wide(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr,
    bool with_random_sign_mask,
    bool with_amax
) {
    if (with_amax) {
        if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<true, true, true, true>(bufs, rows, cols);
        if (data_sr && scale_sr) return run_te_style_direct32_quant_wide_impl<true, true, false, true>(bufs, rows, cols);
        if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<true, false, true, true>(bufs, rows, cols);
        if (data_sr) return run_te_style_direct32_quant_wide_impl<true, false, false, true>(bufs, rows, cols);
        if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<false, true, true, true>(bufs, rows, cols);
        if (scale_sr) return run_te_style_direct32_quant_wide_impl<false, true, false, true>(bufs, rows, cols);
        if (with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<false, false, true, true>(bufs, rows, cols);
        return run_te_style_direct32_quant_wide_impl<false, false, false, true>(bufs, rows, cols);
    }
    if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<true, true, true, false>(bufs, rows, cols);
    if (data_sr && scale_sr) return run_te_style_direct32_quant_wide_impl<true, true, false, false>(bufs, rows, cols);
    if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<true, false, true, false>(bufs, rows, cols);
    if (data_sr) return run_te_style_direct32_quant_wide_impl<true, false, false, false>(bufs, rows, cols);
    if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<false, true, true, false>(bufs, rows, cols);
    if (scale_sr) return run_te_style_direct32_quant_wide_impl<false, true, false, false>(bufs, rows, cols);
    if (with_random_sign_mask) return run_te_style_direct32_quant_wide_impl<false, false, true, false>(bufs, rows, cols);
    return run_te_style_direct32_quant_wide_impl<false, false, false, false>(bufs, rows, cols);
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
float run_te_style_direct32_quant_pipe64_impl(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (cols < CHUNK_K_WIDE || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a64_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h32_gl H_layout{reinterpret_cast<bf16*>(bufs.h32), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K_WIDE, rows / CHUNK_M);
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    check(cudaFuncSetAttribute(
        te_style_direct32_quant_kernel_pipe64<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te style quant pipe64");

    auto launch = [&]() {
        te_style_direct32_quant_kernel_pipe64<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX><<<grid, NUM_THREADS, smem_size>>>(
            A_layout,
            H_layout,
            skip_fp4_store ? nullptr : bufs.out_fp4,
            skip_scale_store ? nullptr : bufs.out_sc,
            WITH_AMAX ? bufs.out_amax : nullptr,
            rows,
            cols,
            1234,
            0);
    };

    if (!skip_fp4_store) {
        check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 pipe64");
    }
    if (!skip_scale_store) {
        check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc pipe64");
    }
    if constexpr (WITH_AMAX) {
        check(cudaMemset(bufs.out_amax, 0, rows * (cols / 32) * sizeof(float)), "memset amax pipe64");
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync pipe64");
    check(cudaGetLastError(), "warmup launch pipe64");
    const float ms = time_launch_ms(launch, "create start event pipe64", "create stop event pipe64", "timed sync pipe64", "timed launch pipe64");

    std::array<uint8_t, 32> sample{};
    check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample pipe64");
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum_pipe64=" << checksum << "\n";
    return ms;
}

float run_te_style_direct32_quant_pipe64(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr,
    bool with_random_sign_mask,
    bool with_amax
) {
    if (with_amax) {
        if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<true, true, true, true>(bufs, rows, cols);
        if (data_sr && scale_sr) return run_te_style_direct32_quant_pipe64_impl<true, true, false, true>(bufs, rows, cols);
        if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<true, false, true, true>(bufs, rows, cols);
        if (data_sr) return run_te_style_direct32_quant_pipe64_impl<true, false, false, true>(bufs, rows, cols);
        if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<false, true, true, true>(bufs, rows, cols);
        if (scale_sr) return run_te_style_direct32_quant_pipe64_impl<false, true, false, true>(bufs, rows, cols);
        if (with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<false, false, true, true>(bufs, rows, cols);
        return run_te_style_direct32_quant_pipe64_impl<false, false, false, true>(bufs, rows, cols);
    }
    if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<true, true, true, false>(bufs, rows, cols);
    if (data_sr && scale_sr) return run_te_style_direct32_quant_pipe64_impl<true, true, false, false>(bufs, rows, cols);
    if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<true, false, true, false>(bufs, rows, cols);
    if (data_sr) return run_te_style_direct32_quant_pipe64_impl<true, false, false, false>(bufs, rows, cols);
    if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<false, true, true, false>(bufs, rows, cols);
    if (scale_sr) return run_te_style_direct32_quant_pipe64_impl<false, true, false, false>(bufs, rows, cols);
    if (with_random_sign_mask) return run_te_style_direct32_quant_pipe64_impl<false, false, true, false>(bufs, rows, cols);
    return run_te_style_direct32_quant_pipe64_impl<false, false, false, false>(bufs, rows, cols);
}

template<
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RANDOM_SIGN_MASK,
    bool SCALE_RP = true,
    bool SKIP_QUANT_PACK = false,
    bool MX32_SCALE = false,
    bool WITH_AMAX = false>
float run_te_copy16_quant_pipe64_impl(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (cols < CHUNK_K_WIDE || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a128_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h16_gl H_layout{reinterpret_cast<bf16*>(bufs.h16), nullptr, nullptr, nullptr, nullptr};
    int sm_count = 0;
    check(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0), "get sm count te copy16");
    const int total_tiles = (cols / CHUNK_K_WIDE) * (rows / CHUNK_M);
    const bool use_rte_scale = std::getenv("TCBENCH_TE_COPY16_RTE_SCALE") != nullptr;
    int grid_mult = (use_rte_scale && rows >= 3072 && cols >= 65536) ? 2 : 1;
    if (const char* env = std::getenv("TCBENCH_TE_COPY16_GRID_MULT")) {
        grid_mult = std::max(1, std::atoi(env));
    }
    const int target_grid = sm_count * grid_mult;
    dim3 grid(total_tiles < target_grid ? total_tiles : target_grid);
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    check(cudaFuncSetAttribute(
        te_copy16_quant_kernel_pipe64<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, SCALE_RP, SKIP_QUANT_PACK, MX32_SCALE, WITH_AMAX>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te copy16 quant pipe64");

    auto launch = [&]() {
        te_copy16_quant_kernel_pipe64<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, SCALE_RP, SKIP_QUANT_PACK, MX32_SCALE, WITH_AMAX><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
            A_layout,
            H_layout,
            skip_fp4_store ? nullptr : bufs.out_fp4,
            skip_scale_store ? nullptr : bufs.out_sc,
            WITH_AMAX ? bufs.out_amax : nullptr,
            rows,
            cols,
            std::getenv("TCBENCH_TE_COPY16_SWIZZLED_SCALE") != nullptr,
            1234,
            0);
    };

    if (!skip_fp4_store) {
        check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 te copy16");
    }
    if (!skip_scale_store) {
        check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset sc te copy16");
    }
    if constexpr (WITH_AMAX) {
        check(cudaMemset(bufs.out_amax, 0, rows * (cols / 16) * sizeof(float)), "memset amax te copy16");
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync te copy16");
    check(cudaGetLastError(), "warmup launch te copy16");
    const float ms = time_launch_ms(launch, "create start event te copy16", "create stop event te copy16", "timed sync te copy16", "timed launch te copy16");

    std::array<uint8_t, 32> sample{};
    check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample te copy16");
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum_te_copy16=" << checksum << "\n";
    if (std::getenv("TCBENCH_PRINT_TE_COPY16_SAMPLE") != nullptr) {
        std::cout << "  sample_te_copy16=";
        for (uint8_t v : sample) {
            std::cout << " " << static_cast<int>(v);
        }
        std::cout << "\n";
    }
    return ms;
}

float run_te_copy16_quant_pipe64(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr,
    bool with_random_sign_mask
) {
    const bool with_amax = std::getenv("TCBENCH_USE_AMAX") != nullptr;
#define TE_COPY16_RUN(DATA_SR_FLAG, SCALE_SR_FLAG, SIGN_FLAG, SCALE_RP_FLAG, SKIP_PACK_FLAG, MX32_FLAG) \
    (with_amax \
        ? run_te_copy16_quant_pipe64_impl<DATA_SR_FLAG, SCALE_SR_FLAG, SIGN_FLAG, SCALE_RP_FLAG, SKIP_PACK_FLAG, MX32_FLAG, true>(bufs, rows, cols) \
        : run_te_copy16_quant_pipe64_impl<DATA_SR_FLAG, SCALE_SR_FLAG, SIGN_FLAG, SCALE_RP_FLAG, SKIP_PACK_FLAG, MX32_FLAG, false>(bufs, rows, cols))
    if (data_sr && scale_sr && with_random_sign_mask) return run_te_copy16_quant_pipe64_impl<true, true, true>(bufs, rows, cols);
    if (data_sr && scale_sr) return run_te_copy16_quant_pipe64_impl<true, true, false>(bufs, rows, cols);
    if (data_sr && with_random_sign_mask) return run_te_copy16_quant_pipe64_impl<true, false, true>(bufs, rows, cols);
    if (data_sr) return run_te_copy16_quant_pipe64_impl<true, false, false>(bufs, rows, cols);
    if (scale_sr && with_random_sign_mask) return run_te_copy16_quant_pipe64_impl<false, true, true>(bufs, rows, cols);
    if (scale_sr) return run_te_copy16_quant_pipe64_impl<false, true, false>(bufs, rows, cols);
    if (with_random_sign_mask) return run_te_copy16_quant_pipe64_impl<false, false, true>(bufs, rows, cols);
    if (std::getenv("TCBENCH_SKIP_QUANT_PACK") != nullptr) {
        return TE_COPY16_RUN(false, false, false, true, true, false);
    }
    if (std::getenv("TCBENCH_TE_COPY16_MX32_SCALE") != nullptr) {
        if (std::getenv("TCBENCH_TE_COPY16_RTE_SCALE") != nullptr) {
            return TE_COPY16_RUN(false, false, false, false, false, true);
        }
        return TE_COPY16_RUN(false, false, false, true, false, true);
    }
    if (std::getenv("TCBENCH_TE_COPY16_RTE_SCALE") != nullptr) {
        return TE_COPY16_RUN(false, false, false, false, false, false);
    }
    return TE_COPY16_RUN(false, false, false, true, false, false);
#undef TE_COPY16_RUN
}

template<bool SCALE_RP = true, bool WITH_AMAX = false, int COL_WARPS = TE_COPY16_EPILOGUE_WARPS>
float run_te_copy16_col_quant(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (rows < CHUNK_M || cols < CHUNK_K_WIDE || (rows % CHUNK_M) != 0 || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a512_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h16_gl H_layout{reinterpret_cast<bf16*>(bufs.h16), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K_WIDE, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    const bool skip_amax_store = std::getenv("TCBENCH_SKIP_AMAX_STORE") != nullptr;
    const bool swizzled_scale = std::getenv("TCBENCH_TC_ROWCOL_SWIZZLED_SCALE") != nullptr;
    check(cudaFuncSetAttribute(
        te_copy16_col_quant_kernel<SCALE_RP, WITH_AMAX, COL_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te copy16 col quant");

    auto launch = [&]() {
        te_copy16_col_quant_kernel<SCALE_RP, WITH_AMAX, COL_WARPS><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
            A_layout,
            H_layout,
            skip_fp4_store ? nullptr : bufs.out_col_fp4,
            skip_scale_store ? nullptr : bufs.out_col_sc,
            (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
            rows,
            cols,
            swizzled_scale);
    };

    if (!skip_fp4_store) {
        check(cudaMemset(bufs.out_col_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset col fp4 te copy16 col");
    }
    if (!skip_scale_store) {
        check(cudaMemset(bufs.out_col_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset col sc te copy16 col");
    }
    if constexpr (WITH_AMAX) {
        if (!skip_amax_store) {
            check(cudaMemset(bufs.out_col_amax, 0, rows * (cols / 16) * sizeof(float)), "memset col amax te copy16 col");
        }
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync te copy16 col");
    check(cudaGetLastError(), "warmup launch te copy16 col");
    const float ms = time_launch_ms(launch, "create start event te copy16 col", "create stop event te copy16 col", "timed sync te copy16 col", "timed launch te copy16 col");

    std::array<uint8_t, 32> sample{};
    if (!skip_fp4_store) {
        check(cudaMemcpy(sample.data(), bufs.out_col_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample te copy16 col");
    }
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum_te_copy16_col=" << checksum << "\n";
    return ms;
}

template<bool SCALE_RP = true, bool WITH_AMAX = false>
float run_te_copy16_col_quant_persistent(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (rows < CHUNK_M || cols < CHUNK_K_WIDE || (rows % CHUNK_M) != 0 || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a512_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h16_gl H_layout{reinterpret_cast<bf16*>(bufs.h16), nullptr, nullptr, nullptr, nullptr};
    int sm_count = 0;
    check(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0), "get sm count te copy16 col persistent");
    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_k * tiles_m;
    int grid_mult = (tiles_k >= 128) ? ((tiles_m >= 24) ? 3 : 2) : 1;
    if (const char* env = std::getenv("TCBENCH_TC_COL_PERSIST_GRID_MULT")) {
        grid_mult = std::max(1, std::atoi(env));
    }
    const int target_grid = sm_count * grid_mult;
    dim3 grid(total_tiles < target_grid ? total_tiles : target_grid);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    const bool skip_amax_store = std::getenv("TCBENCH_SKIP_AMAX_STORE") != nullptr;
    const bool reg2 = std::getenv("TCBENCH_TC_COL_REG2") != nullptr;
    const bool reg2_bf16 = std::getenv("TCBENCH_TC_COL_REG2_BF16") != nullptr;
    const bool pair_smem = std::getenv("TCBENCH_TC_COL_PAIR_SMEM") != nullptr;
    const bool skip_quant_pack = std::getenv("TCBENCH_SKIP_QUANT_PACK") != nullptr;
    const bool bf16x8_quant =
        pair_smem && std::getenv("TCBENCH_TC_COL_NO_BF16X8_QUANT") == nullptr;
    const bool compact_warps = pair_smem && std::getenv("TCBENCH_TC_COL_NO_COMPACT_WARPS") == nullptr;
    if (pair_smem) {
        if (skip_quant_pack) {
            check(cudaFuncSetAttribute(
                te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, true, false, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr te copy16 col quant persistent pair smem skip pack");
            if (compact_warps) {
                check(cudaFuncSetAttribute(
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, true, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size), "setattr te copy16 col quant persistent pair smem skip pack compact");
            }
        } else if (bf16x8_quant) {
            check(cudaFuncSetAttribute(
                te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, true, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr te copy16 col quant persistent pair smem bf16x8");
            if (compact_warps) {
                check(cudaFuncSetAttribute(
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, true, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size), "setattr te copy16 col quant persistent pair smem bf16x8 compact");
            }
        } else {
            check(cudaFuncSetAttribute(
                te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, false, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr te copy16 col quant persistent pair smem");
            if (compact_warps) {
                check(cudaFuncSetAttribute(
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size), "setattr te copy16 col quant persistent pair smem compact");
            }
        }
    } else if (reg2) {
        if (reg2_bf16) {
            check(cudaFuncSetAttribute(
                te_copy16_col_quant_kernel_persistent_reg2<SCALE_RP, WITH_AMAX, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr te copy16 col quant persistent reg2 bf16");
        } else {
            check(cudaFuncSetAttribute(
                te_copy16_col_quant_kernel_persistent_reg2<SCALE_RP, WITH_AMAX, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size), "setattr te copy16 col quant persistent reg2");
        }
    } else {
        check(cudaFuncSetAttribute(
            te_copy16_col_quant_kernel_persistent<SCALE_RP, WITH_AMAX>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te copy16 col quant persistent");
    }

    auto launch = [&]() {
        if (pair_smem) {
            const int pair_threads = compact_warps ? (TE_COPY16_NUM_THREADS / 2) : TE_COPY16_NUM_THREADS;
            if (skip_quant_pack) {
                if (compact_warps) {
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, true, false, true><<<grid, pair_threads, smem_size>>>(
                        A_layout,
                        H_layout,
                        skip_fp4_store ? nullptr : bufs.out_col_fp4,
                        skip_scale_store ? nullptr : bufs.out_col_sc,
                        (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                        rows,
                        cols);
                } else {
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, true, false, false><<<grid, pair_threads, smem_size>>>(
                        A_layout,
                        H_layout,
                        skip_fp4_store ? nullptr : bufs.out_col_fp4,
                        skip_scale_store ? nullptr : bufs.out_col_sc,
                        (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                        rows,
                        cols);
                }
            } else if (bf16x8_quant) {
                if (compact_warps) {
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, true, true><<<grid, pair_threads, smem_size>>>(
                        A_layout,
                        H_layout,
                        skip_fp4_store ? nullptr : bufs.out_col_fp4,
                        skip_scale_store ? nullptr : bufs.out_col_sc,
                        (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                        rows,
                        cols);
                } else {
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, true, false><<<grid, pair_threads, smem_size>>>(
                        A_layout,
                        H_layout,
                        skip_fp4_store ? nullptr : bufs.out_col_fp4,
                        skip_scale_store ? nullptr : bufs.out_col_sc,
                        (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                        rows,
                        cols);
                }
            } else {
                if (compact_warps) {
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, false, true><<<grid, pair_threads, smem_size>>>(
                        A_layout,
                        H_layout,
                        skip_fp4_store ? nullptr : bufs.out_col_fp4,
                        skip_scale_store ? nullptr : bufs.out_col_sc,
                        (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                        rows,
                        cols);
                } else {
                    te_copy16_col_quant_kernel_persistent_pair_smem<SCALE_RP, WITH_AMAX, false, false, false><<<grid, pair_threads, smem_size>>>(
                        A_layout,
                        H_layout,
                        skip_fp4_store ? nullptr : bufs.out_col_fp4,
                        skip_scale_store ? nullptr : bufs.out_col_sc,
                        (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                        rows,
                        cols);
                }
            }
        } else if (reg2) {
            if (reg2_bf16) {
                te_copy16_col_quant_kernel_persistent_reg2<SCALE_RP, WITH_AMAX, true><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
                    A_layout,
                    H_layout,
                    skip_fp4_store ? nullptr : bufs.out_col_fp4,
                    skip_scale_store ? nullptr : bufs.out_col_sc,
                    (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                    rows,
                    cols);
            } else {
                te_copy16_col_quant_kernel_persistent_reg2<SCALE_RP, WITH_AMAX, false><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
                    A_layout,
                    H_layout,
                    skip_fp4_store ? nullptr : bufs.out_col_fp4,
                    skip_scale_store ? nullptr : bufs.out_col_sc,
                    (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                    rows,
                    cols);
            }
        } else {
            te_copy16_col_quant_kernel_persistent<SCALE_RP, WITH_AMAX><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
                A_layout,
                H_layout,
                skip_fp4_store ? nullptr : bufs.out_col_fp4,
                skip_scale_store ? nullptr : bufs.out_col_sc,
                (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                rows,
                cols);
        }
    };

    if (!skip_fp4_store) {
        check(cudaMemset(bufs.out_col_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset col fp4 te copy16 col persistent");
    }
    if (!skip_scale_store) {
        check(cudaMemset(bufs.out_col_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset col sc te copy16 col persistent");
    }
    if constexpr (WITH_AMAX) {
        if (!skip_amax_store) {
            check(cudaMemset(bufs.out_col_amax, 0, rows * (cols / 16) * sizeof(float)), "memset col amax te copy16 col persistent");
        }
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync te copy16 col persistent");
    check(cudaGetLastError(), "warmup launch te copy16 col persistent");
    const float ms = time_launch_ms(launch, "create start event te copy16 col persistent", "create stop event te copy16 col persistent", "timed sync te copy16 col persistent", "timed launch te copy16 col persistent");

    std::array<uint8_t, 32> sample{};
    if (!skip_fp4_store) {
        check(cudaMemcpy(sample.data(), bufs.out_col_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample te copy16 col persistent");
    }
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum_te_copy16_col_persist=" << checksum << "\n";
    if (std::getenv("TCBENCH_PRINT_COL_SAMPLE") != nullptr) {
        std::cout << "  sample_te_copy16_col_persist=";
        for (uint8_t v : sample) {
            std::cout << " " << static_cast<int>(v);
        }
        std::cout << "\n";
    }
    return ms;
}

template<bool SCALE_RP = true, bool WITH_AMAX = false, bool DO_ROW_QUANT = false>
float run_te_copy16_col_quant_persistent_producer(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (rows < CHUNK_M || cols < CHUNK_K_WIDE || (rows % CHUNK_M) != 0 || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a512_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h16_gl H_layout{reinterpret_cast<bf16*>(bufs.h16), nullptr, nullptr, nullptr, nullptr};
    int sm_count = 0;
    check(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0), "get sm count te copy16 col producer");
    const int total_tiles = (cols / CHUNK_K_WIDE) * (rows / CHUNK_M);
    int grid_mult = 1;
    if (const char* env = std::getenv("TCBENCH_TC_COL_PRODUCER_GRID_MULT")) {
        grid_mult = std::max(1, std::atoi(env));
    }
    const int target_grid = sm_count * grid_mult;
    dim3 grid(total_tiles < target_grid ? total_tiles : target_grid);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    const bool skip_amax_store = std::getenv("TCBENCH_SKIP_AMAX_STORE") != nullptr;
    check(cudaFuncSetAttribute(
        te_copy16_col_quant_kernel_persistent_producer<SCALE_RP, WITH_AMAX, DO_ROW_QUANT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te copy16 col quant producer");

    auto launch = [&]() {
        te_copy16_col_quant_kernel_persistent_producer<SCALE_RP, WITH_AMAX, DO_ROW_QUANT><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
            A_layout,
            H_layout,
            (DO_ROW_QUANT && !skip_fp4_store) ? bufs.out_fp4 : nullptr,
            (DO_ROW_QUANT && !skip_scale_store) ? bufs.out_sc : nullptr,
            (DO_ROW_QUANT && WITH_AMAX && !skip_amax_store) ? bufs.out_amax : nullptr,
            skip_fp4_store ? nullptr : bufs.out_col_fp4,
            skip_scale_store ? nullptr : bufs.out_col_sc,
            (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
            rows,
            cols);
    };

    if (!skip_fp4_store) {
        check(cudaMemset(bufs.out_col_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset col fp4 te copy16 col producer");
        if constexpr (DO_ROW_QUANT) {
            check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset row fp4 te copy16 col producer");
        }
    }
    if (!skip_scale_store) {
        check(cudaMemset(bufs.out_col_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset col sc te copy16 col producer");
        if constexpr (DO_ROW_QUANT) {
            check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset row sc te copy16 col producer");
        }
    }
    if constexpr (WITH_AMAX) {
        if (!skip_amax_store) {
            check(cudaMemset(bufs.out_col_amax, 0, rows * (cols / 16) * sizeof(float)), "memset col amax te copy16 col producer");
            if constexpr (DO_ROW_QUANT) {
                check(cudaMemset(bufs.out_amax, 0, rows * (cols / 16) * sizeof(float)), "memset row amax te copy16 col producer");
            }
        }
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync te copy16 col producer");
    check(cudaGetLastError(), "warmup launch te copy16 col producer");
    const float ms = time_launch_ms(launch, "create start event te copy16 col producer", "create stop event te copy16 col producer", "timed sync te copy16 col producer", "timed launch te copy16 col producer");

    std::array<uint8_t, 32> row_sample{};
    std::array<uint8_t, 32> col_sample{};
    if (!skip_fp4_store) {
        check(cudaMemcpy(col_sample.data(), bufs.out_col_fp4, col_sample.size(), cudaMemcpyDeviceToHost), "copy col fp4 sample te copy16 col producer");
        if constexpr (DO_ROW_QUANT) {
            check(cudaMemcpy(row_sample.data(), bufs.out_fp4, row_sample.size(), cudaMemcpyDeviceToHost), "copy row fp4 sample te copy16 col producer");
        }
    }
    uint64_t row_checksum = 0;
    uint64_t col_checksum = 0;
    for (uint8_t v : row_sample) row_checksum += v;
    for (uint8_t v : col_sample) col_checksum += v;
    if constexpr (DO_ROW_QUANT) {
        std::cout << "  checksum_te_copy16_producer_row=" << row_checksum << "\n";
    }
    std::cout << "  checksum_te_copy16_producer_col=" << col_checksum << "\n";
    return ms;
}

template<bool SCALE_RP = true, bool WITH_AMAX = false, bool ROW_RHT = false, bool COL_RHT = true>
float run_te_copy16_rowcol_quant(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (rows < CHUNK_M || cols < CHUNK_K_WIDE || (rows % CHUNK_M) != 0 || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a512_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h16_gl H_layout{reinterpret_cast<bf16*>(bufs.h16), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K_WIDE, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    const bool skip_amax_store = std::getenv("TCBENCH_SKIP_AMAX_STORE") != nullptr;
    const bool swizzled_scale = std::getenv("TCBENCH_TC_ROWCOL_SWIZZLED_SCALE") != nullptr;
    check(cudaFuncSetAttribute(
        te_copy16_rowcol_quant_kernel<SCALE_RP, WITH_AMAX, ROW_RHT, COL_RHT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te copy16 rowcol quant");

    auto launch = [&]() {
        te_copy16_rowcol_quant_kernel<SCALE_RP, WITH_AMAX, ROW_RHT, COL_RHT><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
            A_layout,
            H_layout,
            skip_fp4_store ? nullptr : bufs.out_fp4,
            skip_scale_store ? nullptr : bufs.out_sc,
            (WITH_AMAX && !skip_amax_store) ? bufs.out_amax : nullptr,
            skip_fp4_store ? nullptr : bufs.out_col_fp4,
            skip_scale_store ? nullptr : bufs.out_col_sc,
            (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
            rows,
            cols,
            swizzled_scale);
    };

    if (!skip_fp4_store) {
        check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset row fp4 te copy16 rowcol");
        check(cudaMemset(bufs.out_col_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset col fp4 te copy16 rowcol");
    }
    if (!skip_scale_store) {
        check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset row sc te copy16 rowcol");
        check(cudaMemset(bufs.out_col_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset col sc te copy16 rowcol");
    }
    if constexpr (WITH_AMAX) {
        if (!skip_amax_store) {
            check(cudaMemset(bufs.out_amax, 0, rows * (cols / 16) * sizeof(float)), "memset row amax te copy16 rowcol");
            check(cudaMemset(bufs.out_col_amax, 0, rows * (cols / 16) * sizeof(float)), "memset col amax te copy16 rowcol");
        }
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync te copy16 rowcol");
    check(cudaGetLastError(), "warmup launch te copy16 rowcol");
    const float ms = time_launch_ms(launch, "create start event te copy16 rowcol", "create stop event te copy16 rowcol", "timed sync te copy16 rowcol", "timed launch te copy16 rowcol");

    std::array<uint8_t, 32> row_sample{};
    std::array<uint8_t, 32> col_sample{};
    if (!skip_fp4_store) {
        check(cudaMemcpy(row_sample.data(), bufs.out_fp4, row_sample.size(), cudaMemcpyDeviceToHost), "copy row fp4 sample te copy16 rowcol");
        check(cudaMemcpy(col_sample.data(), bufs.out_col_fp4, col_sample.size(), cudaMemcpyDeviceToHost), "copy col fp4 sample te copy16 rowcol");
    }
    uint64_t row_checksum = 0;
    uint64_t col_checksum = 0;
    for (uint8_t v : row_sample) row_checksum += v;
    for (uint8_t v : col_sample) col_checksum += v;
    std::cout << "  checksum_te_copy16_rowcol_row=" << row_checksum << "\n";
    std::cout << "  checksum_te_copy16_rowcol_col=" << col_checksum << "\n";
    return ms;
}

template<
    bool SCALE_RP = true,
    bool WITH_AMAX = false,
    bool PAIR_SMEM = false,
    bool ROW_BF16X8_QUANT = false,
    bool COL_BF16X8_QUANT = false>
float run_te_copy16_rowcol_quant_persistent(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (rows < CHUNK_M || cols < CHUNK_K_WIDE || (rows % CHUNK_M) != 0 || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a512_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h16_gl H_layout{reinterpret_cast<bf16*>(bufs.h16), nullptr, nullptr, nullptr, nullptr};
    int sm_count = 0;
    check(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0), "get sm count te copy16 rowcol persistent");
    const int tiles_m = rows / CHUNK_M;
    const int tiles_k = cols / CHUNK_K_WIDE;
    const int total_tiles = tiles_k * tiles_m;
    int grid_mult = (total_tiles == 2048) ? 2 : 1;
    if (const char* env = std::getenv("TCBENCH_TC_ROWCOL_PERSIST_GRID_MULT")) {
        grid_mult = std::max(1, std::atoi(env));
    }
    int target_grid = sm_count * grid_mult;
    if (const char* env = std::getenv("TCBENCH_TC_ROWCOL_PERSIST_GRID")) {
        target_grid = std::max(1, std::atoi(env));
    }
    dim3 grid(total_tiles < target_grid ? total_tiles : target_grid);
    #if defined(TCBENCH_ROWCOL_BLOCK_PAIR64_STORE)
    int smem_size = PAIR_SMEM ? (160 * 1024) : (MAX_SHARED_MEMORY - 1024);
    if constexpr (PAIR_SMEM) {
        if (const char* env = std::getenv("TCBENCH_TC_ROWCOL_BLOCK_PAIR64_SMEM_KB")) {
            smem_size = std::max(0, std::atoi(env)) * 1024;
        }
    }
    #else
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    #endif
    const bool skip_fp4_store = std::getenv("TCBENCH_SKIP_FP4_STORE") != nullptr;
    const bool skip_row_fp4_store =
        skip_fp4_store || std::getenv("TCBENCH_SKIP_ROW_FP4_STORE") != nullptr;
    const bool skip_col_fp4_store =
        skip_fp4_store || std::getenv("TCBENCH_SKIP_COL_FP4_STORE") != nullptr;
    const bool skip_scale_store = std::getenv("TCBENCH_SKIP_SCALE_STORE") != nullptr;
    const bool skip_amax_store = std::getenv("TCBENCH_SKIP_AMAX_STORE") != nullptr;
    const bool use_reg_split = !skip_row_fp4_store && !skip_col_fp4_store;

    auto run_variant = [&]<bool REG_SPLIT>() -> float {
        check(cudaFuncSetAttribute(
            te_copy16_rowcol_quant_kernel_persistent<SCALE_RP, WITH_AMAX, PAIR_SMEM, ROW_BF16X8_QUANT, COL_BF16X8_QUANT, REG_SPLIT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te copy16 rowcol quant persistent");

        auto launch = [&]() {
            te_copy16_rowcol_quant_kernel_persistent<SCALE_RP, WITH_AMAX, PAIR_SMEM, ROW_BF16X8_QUANT, COL_BF16X8_QUANT, REG_SPLIT><<<grid, TE_COPY16_NUM_THREADS, smem_size>>>(
                A_layout,
                H_layout,
                skip_row_fp4_store ? nullptr : bufs.out_fp4,
                skip_scale_store ? nullptr : bufs.out_sc,
                (WITH_AMAX && !skip_amax_store) ? bufs.out_amax : nullptr,
                skip_col_fp4_store ? nullptr : bufs.out_col_fp4,
                skip_scale_store ? nullptr : bufs.out_col_sc,
                (WITH_AMAX && !skip_amax_store) ? bufs.out_col_amax : nullptr,
                rows,
                cols);
        };

        if (!skip_row_fp4_store) {
            check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset row fp4 te copy16 rowcol persistent");
        }
        if (!skip_col_fp4_store) {
            check(cudaMemset(bufs.out_col_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset col fp4 te copy16 rowcol persistent");
        }
        if (!skip_scale_store) {
            check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset row sc te copy16 rowcol persistent");
            check(cudaMemset(bufs.out_col_sc, 0xa5, rows * (cols / 16) * sizeof(uint8_t)), "memset col sc te copy16 rowcol persistent");
        }
        if constexpr (WITH_AMAX) {
            if (!skip_amax_store) {
                check(cudaMemset(bufs.out_amax, 0, rows * (cols / 16) * sizeof(float)), "memset row amax te copy16 rowcol persistent");
                check(cudaMemset(bufs.out_col_amax, 0, rows * (cols / 16) * sizeof(float)), "memset col amax te copy16 rowcol persistent");
            }
        }
        for (int i = 0; i < 3; ++i) launch();
        check(cudaDeviceSynchronize(), "warmup sync te copy16 rowcol persistent");
        check(cudaGetLastError(), "warmup launch te copy16 rowcol persistent");
        const float ms = time_launch_ms(launch, "create start event te copy16 rowcol persistent", "create stop event te copy16 rowcol persistent", "timed sync te copy16 rowcol persistent", "timed launch te copy16 rowcol persistent");

        std::array<uint8_t, 32> row_sample{};
        std::array<uint8_t, 32> col_sample{};
        if (!skip_fp4_store) {
            check(cudaMemcpy(row_sample.data(), bufs.out_fp4, row_sample.size(), cudaMemcpyDeviceToHost), "copy row fp4 sample te copy16 rowcol persistent");
            check(cudaMemcpy(col_sample.data(), bufs.out_col_fp4, col_sample.size(), cudaMemcpyDeviceToHost), "copy col fp4 sample te copy16 rowcol persistent");
        }
        uint64_t row_checksum = 0;
        uint64_t col_checksum = 0;
        for (uint8_t v : row_sample) row_checksum += v;
        for (uint8_t v : col_sample) col_checksum += v;
        std::cout << "  checksum_te_copy16_rowcol_persist_row=" << row_checksum << "\n";
        std::cout << "  checksum_te_copy16_rowcol_persist_col=" << col_checksum << "\n";
        return ms;
    };

    return use_reg_split ? run_variant.template operator()<true>() : run_variant.template operator()<false>();
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
float run_mxfp4_v4_pure_rowcol_quant_impl(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if ((rows % CHUNK_DIM) != 0 || (cols % CHUNK_DIM) != 0) {
        return -1.0f;
    }

    alignas(64) CUtensorMap tma_in{};
    alignas(64) CUtensorMap tma_row_out{};
    alignas(64) CUtensorMap tma_col_out{};
    create_tma_2d_bench(tma_in, bufs.input, rows, cols, TILE_DIM, TILE_DIM, cols, 16);
    create_tma_2d_bench(tma_row_out, bufs.out_fp4, rows, cols, TILE_DIM, TILE_DIM, cols, 4);
    create_tma_2d_bench(tma_col_out, bufs.out_col_fp4, cols, rows, TILE_DIM, TILE_DIM, rows, 4);

    const int smem_size = v4_rowcol_shmem_size();
    check(cudaFuncSetAttribute(
        mxfp4_v4_rowcol_fused_kernel_opt<QuantMode::RTE, DATA_SR, SCALE_SR, false, 16, WITH_RANDOM_SIGN_MASK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr mxfp4 v4 pure rowcol");
    dim3 grid(cols / CHUNK_DIM, rows / CHUNK_DIM);

    auto launch = [&]() {
        mxfp4_v4_rowcol_fused_kernel_opt<QuantMode::RTE, DATA_SR, SCALE_SR, false, 16, WITH_RANDOM_SIGN_MASK>
            <<<grid, THREADS, smem_size>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                bufs.out_sc,
                bufs.out_col_sc,
                rows,
                cols,
                1234,
                0,
                nullptr);
    };

    const size_t fp4_bytes = static_cast<size_t>(rows) * (cols / 2);
    const size_t scale_bytes = static_cast<size_t>(rows) * (cols / 32);
    check(cudaMemset(bufs.out_fp4, 0x5a, fp4_bytes), "memset row fp4 pure rowcol");
    check(cudaMemset(bufs.out_col_fp4, 0xa6, fp4_bytes), "memset col fp4 pure rowcol");
    check(cudaMemset(bufs.out_sc, 0xa5, scale_bytes), "memset row sc pure rowcol");
    check(cudaMemset(bufs.out_col_sc, 0x5b, scale_bytes), "memset col sc pure rowcol");
    for (int i = 0; i < 3; ++i) {
        launch();
    }
    check(cudaDeviceSynchronize(), "warmup sync pure rowcol");
    check(cudaGetLastError(), "warmup launch pure rowcol");
    const float ms = time_launch_ms(
        launch,
        "create start event pure rowcol",
        "create stop event pure rowcol",
        "timed sync pure rowcol",
        "timed launch pure rowcol");

    std::array<uint8_t, 32> row_sample{};
    std::array<uint8_t, 32> col_sample{};
    check(cudaMemcpy(row_sample.data(), bufs.out_fp4, row_sample.size(), cudaMemcpyDeviceToHost), "copy row fp4 sample pure rowcol");
    check(cudaMemcpy(col_sample.data(), bufs.out_col_fp4, col_sample.size(), cudaMemcpyDeviceToHost), "copy col fp4 sample pure rowcol");
    uint64_t row_checksum = 0;
    uint64_t col_checksum = 0;
    for (uint8_t v : row_sample) row_checksum += v;
    for (uint8_t v : col_sample) col_checksum += v;
    std::cout << "  checksum_pure_rowcol_row=" << row_checksum
              << " col=" << col_checksum << "\n";
    return ms;
}

float run_mxfp4_v4_pure_rowcol_quant(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr,
    bool with_random_sign_mask
) {
    if (data_sr && scale_sr && with_random_sign_mask) return run_mxfp4_v4_pure_rowcol_quant_impl<true, true, true>(bufs, rows, cols);
    if (data_sr && scale_sr) return run_mxfp4_v4_pure_rowcol_quant_impl<true, true, false>(bufs, rows, cols);
    if (data_sr && with_random_sign_mask) return run_mxfp4_v4_pure_rowcol_quant_impl<true, false, true>(bufs, rows, cols);
    if (data_sr) return run_mxfp4_v4_pure_rowcol_quant_impl<true, false, false>(bufs, rows, cols);
    if (scale_sr && with_random_sign_mask) return run_mxfp4_v4_pure_rowcol_quant_impl<false, true, true>(bufs, rows, cols);
    if (scale_sr) return run_mxfp4_v4_pure_rowcol_quant_impl<false, true, false>(bufs, rows, cols);
    if (with_random_sign_mask) return run_mxfp4_v4_pure_rowcol_quant_impl<false, false, true>(bufs, rows, cols);
    return run_mxfp4_v4_pure_rowcol_quant_impl<false, false, false>(bufs, rows, cols);
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK, bool WITH_AMAX>
float run_te_style_direct32_quant_wide_reg_impl(
    const DeviceBuffers& bufs,
    int rows,
    int cols
) {
    if (cols < CHUNK_K_WIDE || (cols % CHUNK_K_WIDE) != 0) {
        return -1.0f;
    }
    a512_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h32_gl H_layout{reinterpret_cast<bf16*>(bufs.h32), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K_WIDE, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;
    check(cudaFuncSetAttribute(
        te_style_direct32_quant_kernel_wide_reg_epilogue<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size), "setattr te style quant wide reg");

    auto launch = [&]() {
        te_style_direct32_quant_kernel_wide_reg_epilogue<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, WITH_AMAX><<<grid, NUM_THREADS, smem_size>>>(
            A_layout, H_layout, bufs.out_fp4, bufs.out_sc, WITH_AMAX ? bufs.out_amax : nullptr, rows, cols, 1234, 0);
    };

    check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 wide reg");
    check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc wide reg");
    if constexpr (WITH_AMAX) {
        check(cudaMemset(bufs.out_amax, 0, rows * (cols / 32) * sizeof(float)), "memset amax wide reg");
    }
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync wide reg");
    check(cudaGetLastError(), "warmup launch wide reg");
    const float ms = time_launch_ms(launch, "create start event wide reg", "create stop event wide reg", "timed sync wide reg", "timed launch wide reg");

    std::array<uint8_t, 32> sample{};
    check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample wide reg");
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum_wide_reg=" << checksum << "\n";
    return ms;
}

float run_te_style_direct32_quant_wide_reg(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr,
    bool with_random_sign_mask,
    bool with_amax
) {
    if (with_amax) {
        if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<true, true, true, true>(bufs, rows, cols);
        if (data_sr && scale_sr) return run_te_style_direct32_quant_wide_reg_impl<true, true, false, true>(bufs, rows, cols);
        if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<true, false, true, true>(bufs, rows, cols);
        if (data_sr) return run_te_style_direct32_quant_wide_reg_impl<true, false, false, true>(bufs, rows, cols);
        if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<false, true, true, true>(bufs, rows, cols);
        if (scale_sr) return run_te_style_direct32_quant_wide_reg_impl<false, true, false, true>(bufs, rows, cols);
        if (with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<false, false, true, true>(bufs, rows, cols);
        return run_te_style_direct32_quant_wide_reg_impl<false, false, false, true>(bufs, rows, cols);
    }
    if (data_sr && scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<true, true, true, false>(bufs, rows, cols);
    if (data_sr && scale_sr) return run_te_style_direct32_quant_wide_reg_impl<true, true, false, false>(bufs, rows, cols);
    if (data_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<true, false, true, false>(bufs, rows, cols);
    if (data_sr) return run_te_style_direct32_quant_wide_reg_impl<true, false, false, false>(bufs, rows, cols);
    if (scale_sr && with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<false, true, true, false>(bufs, rows, cols);
    if (scale_sr) return run_te_style_direct32_quant_wide_reg_impl<false, true, false, false>(bufs, rows, cols);
    if (with_random_sign_mask) return run_te_style_direct32_quant_wide_reg_impl<false, false, true, false>(bufs, rows, cols);
    return run_te_style_direct32_quant_wide_reg_impl<false, false, false, false>(bufs, rows, cols);
}

void compare_wide_reg_epilogue(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool with_amax
) {
    if (cols < CHUNK_K_WIDE || (cols % CHUNK_K_WIDE) != 0) {
        std::cout << "reg-epi compare skipped: shape is not x512 compatible\n";
        return;
    }

    run_te_style_direct32_quant_wide(bufs, rows, cols, false, false, true, with_amax);
    const size_t fp4_bytes = static_cast<size_t>(rows) * (cols / 2);
    const size_t scale_bytes = static_cast<size_t>(rows) * (cols / 32);
    std::vector<uint8_t> ref_fp4(fp4_bytes);
    std::vector<uint8_t> ref_sc(scale_bytes);
    std::vector<float> ref_amax(with_amax ? scale_bytes : 0);
    check(cudaMemcpy(ref_fp4.data(), bufs.out_fp4, fp4_bytes, cudaMemcpyDeviceToHost), "copy ref fp4 reg compare");
    check(cudaMemcpy(ref_sc.data(), bufs.out_sc, scale_bytes, cudaMemcpyDeviceToHost), "copy ref scales reg compare");
    if (with_amax) {
        check(cudaMemcpy(ref_amax.data(), bufs.out_amax, scale_bytes * sizeof(float), cudaMemcpyDeviceToHost), "copy ref amax reg compare");
    }

    run_te_style_direct32_quant_wide_reg(bufs, rows, cols, false, false, true, with_amax);
    std::vector<uint8_t> got_fp4(fp4_bytes);
    std::vector<uint8_t> got_sc(scale_bytes);
    std::vector<float> got_amax(with_amax ? scale_bytes : 0);
    check(cudaMemcpy(got_fp4.data(), bufs.out_fp4, fp4_bytes, cudaMemcpyDeviceToHost), "copy got fp4 reg compare");
    check(cudaMemcpy(got_sc.data(), bufs.out_sc, scale_bytes, cudaMemcpyDeviceToHost), "copy got scales reg compare");
    if (with_amax) {
        check(cudaMemcpy(got_amax.data(), bufs.out_amax, scale_bytes * sizeof(float), cudaMemcpyDeviceToHost), "copy got amax reg compare");
    }

    size_t fp4_mismatch = 0;
    size_t first_fp4 = fp4_bytes;
    for (size_t i = 0; i < fp4_bytes; ++i) {
        if (ref_fp4[i] != got_fp4[i]) {
            if (first_fp4 == fp4_bytes) first_fp4 = i;
            ++fp4_mismatch;
        }
    }

    size_t scale_mismatch = 0;
    size_t first_scale = scale_bytes;
    for (size_t i = 0; i < scale_bytes; ++i) {
        if (ref_sc[i] != got_sc[i]) {
            if (first_scale == scale_bytes) first_scale = i;
            ++scale_mismatch;
        }
    }

    size_t amax_mismatch = 0;
    size_t first_amax = scale_bytes;
    if (with_amax) {
        for (size_t i = 0; i < scale_bytes; ++i) {
            if (ref_amax[i] != got_amax[i]) {
                if (first_amax == scale_bytes) first_amax = i;
                ++amax_mismatch;
            }
        }
    }

    std::cout << "reg-epi compare fp4_mismatch=" << fp4_mismatch;
    if (first_fp4 != fp4_bytes) {
        std::cout << " first_fp4=" << first_fp4 << " ref=" << static_cast<int>(ref_fp4[first_fp4])
                  << " got=" << static_cast<int>(got_fp4[first_fp4]);
    }
    std::cout << " scale_mismatch=" << scale_mismatch;
    if (first_scale != scale_bytes) {
        std::cout << " first_scale=" << first_scale << " ref=" << static_cast<int>(ref_sc[first_scale])
                  << " got=" << static_cast<int>(got_sc[first_scale]);
    }
    if (with_amax) {
        std::cout << " amax_mismatch=" << amax_mismatch;
        if (first_amax != scale_bytes) {
            std::cout << " first_amax=" << first_amax << " ref=" << ref_amax[first_amax]
                      << " got=" << got_amax[first_amax];
        }
    }
    std::cout << "\n";
}

template<bool DATA_SR, bool SCALE_SR, bool WITH_RANDOM_SIGN_MASK>
__global__ __launch_bounds__(NUM_THREADS, 1)
void te_style_direct32_quant_kernel_persist4(
    const __grid_constant__ a128_gl A_layout,
    const __grid_constant__ h32_gl H_layout,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a128_tile>();
        H_layout.template prefetch_tma<h32_tile>();
    }
    const int bid_m = blockIdx.y;
    const int bid_k_group = blockIdx.x;
    const int warp_id = warpid();
    const int lane = laneid();
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a128_tile (&a_smem)[LOAD_PIPE_STAGES] = al.allocate<a128_tile, LOAD_PIPE_STAGES>();
    h32_tile (&h_smem) = al.allocate<h32_tile>();
    d16_tile (&epi_smem)[4] = al.allocate<d16_tile, 4>();

    __shared__ semaphore inputs_arrived[LOAD_PIPE_STAGES];
    __shared__ semaphore h_arrived;
    __shared__ semaphore inputs_finished[LOAD_PIPE_STAGES];
    __shared__ semaphore compute_done;
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ uint32_t accum_addr;
    __shared__ semaphore accum_ready;

    if (threadIdx.x == 0) {
        init_semaphore(h_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < LOAD_PIPE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 2);
        }
        init_semaphore(compute_done, 0, 2);
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(accum_ready, 0, 1);
        tma::expect_bytes(h_arrived, sizeof(h32_tile));
        tma::load_async(h_smem, H_layout, {0, 0}, h_arrived);
    }

    tensor_allocator<1, 1, false> tm_alloc{};
    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    wait(h_arrived, 0);
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);

    full_accum_tt_persist accum;
    if (warp_id == 1 && lane == 0) {
        accum = tm_alloc.template allocate<full_accum_tt_persist>(0);
        accum_addr = accum.addr;
        warp::arrive(accum_ready);
    }

    if (warp_id == 0 && lane == 0) {
        uint32_t stage = 0;
        for (int sub = 0; sub < CHUNK_K_PERSIST / CHUNK_K; ++sub) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            tma::expect_bytes(inputs_arrived[stage], sizeof(a128_tile));
            tma::load_async(a_smem[stage], A_layout, {bid_m, bid_k_group * (CHUNK_K_PERSIST / CHUNK_K) + sub}, inputs_arrived[stage]);
            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % LOAD_PIPE_STAGES;
        }
    }

    if (warp_id == 1 || warp_id == 2) {
        wait(accum_ready, 0);
        accum = full_accum_tt_persist(accum_addr);
        const int sy = warp_id - 1;
        uint32_t stage = 0;
        for (int sub = 0; sub < CHUNK_K_PERSIST / CHUNK_K; ++sub) {
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            const int col_base = sub * CHUNK_K;
            #pragma unroll
            for (int sx = 0; sx < CHUNK_K / TILE_N; ++sx) {
                auto a_sub = a_smem[stage].template subtile<64, 32>({sy, sx});
                auto acc_sub = accum.template subtile<block_accum_tt>(sy * 64, col_base + sx * 32);
                if (lane == 0) {
                    if (sx == (CHUNK_K / TILE_N) - 1) {
                        mm_AB(acc_sub, a_sub, h_smem, inputs_finished[stage]);
                    } else {
                        mm_AB(acc_sub, a_sub, h_smem);
                    }
                }
            }
            update_phasebit<0>(phasebits, stage);
            stage = (stage + 1) % LOAD_PIPE_STAGES;
        }
        if (lane == 0) {
            warp::arrive(compute_done);
        }
    }

    wait(compute_done, 0);

    if (warpgroup::groupid() == 1) {
        wait(accum_ready, 0);
        accum = full_accum_tt_persist(accum_addr);
        const int local_warp = warpgroup::warpid();
        lane_accum_rt reg_frag;
        #pragma unroll
        for (int sy = 0; sy < 2; ++sy) {
            #pragma unroll
            for (int sx = 0; sx < CHUNK_K_PERSIST / TILE_N; ++sx) {
                auto tm_sub = accum.template subtile<lane_accum_tt>(sy * 64 + local_warp * 16, sx * 32);
                warp::load_async(reg_frag, tm_sub);
                tensor_load_wait();
                group<1>::store(epi_smem[local_warp], reg_frag);
                __syncwarp();
                quantize_tile_16x32<DATA_SR, SCALE_SR, WITH_RANDOM_SIGN_MASK, false>(
                    epi_smem[local_warp],
                    fp4_out,
                    scales_out,
                    nullptr,
                    cols,
                    bid_m * CHUNK_M + sy * 64 + local_warp * 16,
                    bid_k_group * (CHUNK_K_PERSIST / TILE_N) + sx,
                    rng_seed,
                    rng_subsequence_base + static_cast<uint64_t>((bid_m * gridDim.x + bid_k_group) * (2 * (CHUNK_K_PERSIST / TILE_N)) + sy * (CHUNK_K_PERSIST / TILE_N) + sx) * TILE_M + local_warp * 16
                );
                __syncwarp();
            }
        }
    }

    if (warpgroup::groupid() == 1 && warpgroup::warpid() == 0) {
        tm_alloc.deprovision();
    }
}

float run_te_style_direct32_quant_persist4(
    const DeviceBuffers& bufs,
    int rows,
    int cols,
    bool data_sr,
    bool scale_sr,
    bool with_random_sign_mask
) {
    if (cols < CHUNK_K_PERSIST || (cols % CHUNK_K_PERSIST) != 0) {
        return -1.0f;
    }
    a128_gl A_layout{reinterpret_cast<bf16*>(bufs.input), nullptr, nullptr, (unsigned long)rows, (unsigned long)cols};
    h32_gl H_layout{reinterpret_cast<bf16*>(bufs.h32), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K_PERSIST, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;

    if (data_sr && scale_sr && with_random_sign_mask) {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<true, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist sr");
    } else if (data_sr && scale_sr) {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<true, true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist sr nosign");
    } else if (data_sr && with_random_sign_mask) {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<true, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist data sr");
    } else if (data_sr) {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<true, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist data sr nosign");
    } else if (scale_sr && with_random_sign_mask) {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist scale sr");
    } else if (scale_sr) {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<false, true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist scale sr nosign");
    } else if (with_random_sign_mask) {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist");
    } else {
        check(cudaFuncSetAttribute(
            te_style_direct32_quant_kernel_persist4<false, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size), "setattr te style quant persist nosign");
    }

    auto launch = [&]() {
        if (data_sr && scale_sr && with_random_sign_mask) {
            te_style_direct32_quant_kernel_persist4<true, true, true><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        } else if (data_sr && scale_sr) {
            te_style_direct32_quant_kernel_persist4<true, true, false><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        } else if (data_sr && with_random_sign_mask) {
            te_style_direct32_quant_kernel_persist4<true, false, true><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        } else if (data_sr) {
            te_style_direct32_quant_kernel_persist4<true, false, false><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        } else if (scale_sr && with_random_sign_mask) {
            te_style_direct32_quant_kernel_persist4<false, true, true><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        } else if (scale_sr) {
            te_style_direct32_quant_kernel_persist4<false, true, false><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        } else if (with_random_sign_mask) {
            te_style_direct32_quant_kernel_persist4<false, false, true><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        } else {
            te_style_direct32_quant_kernel_persist4<false, false, false><<<grid, NUM_THREADS, smem_size>>>(
                A_layout, H_layout, bufs.out_fp4, bufs.out_sc, rows, cols, 1234, 0);
        }
    };

    check(cudaMemset(bufs.out_fp4, 0x5a, rows * (cols / 2) * sizeof(uint8_t)), "memset fp4 persist");
    check(cudaMemset(bufs.out_sc, 0xa5, rows * (cols / 32) * sizeof(uint8_t)), "memset sc persist");
    for (int i = 0; i < 3; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync persist");
    check(cudaGetLastError(), "warmup launch persist");
    const float ms = time_launch_ms(launch, "create start event persist", "create stop event persist", "timed sync persist", "timed launch persist");

    std::array<uint8_t, 32> sample{};
    check(cudaMemcpy(sample.data(), bufs.out_fp4, sample.size(), cudaMemcpyDeviceToHost), "copy fp4 sample persist");
    uint64_t checksum = 0;
    for (uint8_t v : sample) checksum += v;
    std::cout << "  checksum_persist=" << checksum << "\n";
    return ms;
}

void run_case(int rows, int cols) {
    std::cout << "==== shape " << rows << " x " << cols << " ====\n";
    DeviceBuffers bufs;
    init_device_buffers(bufs, rows, cols);
    const bool with_amax = std::getenv("TCBENCH_USE_AMAX") != nullptr;
    const bool skip_sr = std::getenv("TCBENCH_SKIP_SR") != nullptr;
    const bool with_random_sign_mask =
        std::getenv("TCBENCH_NO_RSM") == nullptr && std::getenv("TCBENCH_STATIC_RSM") == nullptr;
    if (std::getenv("TCBENCH_ONLY_PURE_ROWCOL") != nullptr) {
        const float pure_rowcol = run_mxfp4_v4_pure_rowcol_quant(bufs, rows, cols, false, false, false);
        const float pure_rowcol_sr = skip_sr ? -1.0f : run_mxfp4_v4_pure_rowcol_quant(bufs, rows, cols, true, true, false);
        std::cout << "mode: pure_rowcol\n";
        std::cout << "mxfp4-v4 pure row+col quant: " << pure_rowcol << " ms\n";
        if (pure_rowcol_sr >= 0.0f) {
            std::cout << "mxfp4-v4 pure row+col quant+sr: " << pure_rowcol_sr << " ms\n";
        }
        free_device_buffers(bufs);
        return;
    }
    if (std::getenv("TCBENCH_ONLY_TE_COPY16") != nullptr) {
        const float te_copy16 = run_te_copy16_quant_pipe64(bufs, rows, cols, false, false, with_random_sign_mask);
        const float te_copy16_sr = skip_sr ? -1.0f : run_te_copy16_quant_pipe64(bufs, rows, cols, true, true, with_random_sign_mask);
        const bool skip_pure_rowcol = std::getenv("TCBENCH_SKIP_PURE_ROWCOL") != nullptr;
        const float pure_rowcol = skip_pure_rowcol ? -1.0f : run_mxfp4_v4_pure_rowcol_quant(bufs, rows, cols, false, false, false);
        const bool rte_scale = std::getenv("TCBENCH_TE_COPY16_RTE_SCALE") != nullptr;
        const bool mx32_scale = std::getenv("TCBENCH_TE_COPY16_MX32_SCALE") != nullptr;
        std::cout << "mode: row_rht_only_te_copy16\n";
        std::cout << "te-copy16 row-oriented RHT+quant only pipe128 " << (mx32_scale ? "mx32-" : "")
                  << (rte_scale ? "rte-scale" : "te-rp-scale")
                  << ": " << te_copy16 << " ms\n";
        if (te_copy16_sr >= 0.0f) {
            std::cout << "te-copy16 row-oriented RHT+quant only pipe128 " << (mx32_scale ? "mx32-" : "")
                      << (rte_scale ? "rte-scale" : "te-rp-scale")
                      << "+sr: " << te_copy16_sr << " ms\n";
        }
        if (pure_rowcol >= 0.0f) {
            std::cout << "mxfp4-v4 pure row+col quant: " << pure_rowcol << " ms\n";
        }
        free_device_buffers(bufs);
        return;
    }
    if (std::getenv("TCBENCH_ONLY_TC_COL16") != nullptr) {
        const bool with_amax_col = std::getenv("TCBENCH_USE_AMAX") != nullptr;
        const bool rte_scale = std::getenv("TCBENCH_TE_COPY16_RTE_SCALE") != nullptr;
        const bool col4 = std::getenv("TCBENCH_TC_COL4") != nullptr;
        const bool persist = std::getenv("TCBENCH_TC_COL_PERSIST") != nullptr;
        const bool producer = std::getenv("TCBENCH_TC_COL_PRODUCER") != nullptr;
        float tc_col = -1.0f;
        if (producer && with_amax_col) {
            tc_col = rte_scale
                ? run_te_copy16_col_quant_persistent_producer<false, true, false>(bufs, rows, cols)
                : run_te_copy16_col_quant_persistent_producer<true, true, false>(bufs, rows, cols);
        } else if (producer) {
            tc_col = rte_scale
                ? run_te_copy16_col_quant_persistent_producer<false, false, false>(bufs, rows, cols)
                : run_te_copy16_col_quant_persistent_producer<true, false, false>(bufs, rows, cols);
        } else if (persist && with_amax_col) {
            tc_col = rte_scale
                ? run_te_copy16_col_quant_persistent<false, true>(bufs, rows, cols)
                : run_te_copy16_col_quant_persistent<true, true>(bufs, rows, cols);
        } else if (persist) {
            tc_col = rte_scale
                ? run_te_copy16_col_quant_persistent<false, false>(bufs, rows, cols)
                : run_te_copy16_col_quant_persistent<true, false>(bufs, rows, cols);
        } else if (with_amax_col) {
            if (col4) {
                tc_col = rte_scale
                    ? run_te_copy16_col_quant<false, true, 4>(bufs, rows, cols)
                    : run_te_copy16_col_quant<true, true, 4>(bufs, rows, cols);
            } else {
                tc_col = rte_scale
                    ? run_te_copy16_col_quant<false, true>(bufs, rows, cols)
                    : run_te_copy16_col_quant<true, true>(bufs, rows, cols);
            }
        } else {
            if (col4) {
                tc_col = rte_scale
                    ? run_te_copy16_col_quant<false, false, 4>(bufs, rows, cols)
                    : run_te_copy16_col_quant<true, false, 4>(bufs, rows, cols);
            } else {
                tc_col = rte_scale
                    ? run_te_copy16_col_quant<false, false>(bufs, rows, cols)
                    : run_te_copy16_col_quant<true, false>(bufs, rows, cols);
            }
        }
        std::cout << "mode: col_rht_only_tc16" << (with_amax_col ? "_with_amax" : "_no_amax") << "\n";
        std::cout << "te-copy16 col-RHT+quant only tensorcore "
                  << (producer ? "producer " : "")
                  << (persist ? "persist " : "")
                  << (std::getenv("TCBENCH_TC_COL_PAIR_SMEM") != nullptr ? "pair-smem " : "")
                  << ((std::getenv("TCBENCH_TC_COL_PAIR_SMEM") != nullptr &&
                       std::getenv("TCBENCH_TC_COL_NO_COMPACT_WARPS") == nullptr) ? "compact-warps " : "")
                  << ((std::getenv("TCBENCH_TC_COL_PAIR_SMEM") != nullptr &&
                       std::getenv("TCBENCH_TC_COL_NO_BF16X8_QUANT") == nullptr) ? "bf16x8 " : "")
                  << (std::getenv("TCBENCH_SKIP_QUANT_PACK") != nullptr ? "skip-pack " : "")
                  << (std::getenv("TCBENCH_TC_COL_REG2") != nullptr ? "reg2 " : "")
                  << (std::getenv("TCBENCH_TC_COL_REG2_BF16") != nullptr ? "bf16 " : "")
                  << ((producer || col4) ? "col4 " : "col8 ")
                  << (rte_scale ? "rte-scale" : "te-rp-scale")
                  << " had+quant: " << tc_col << " ms\n";
        free_device_buffers(bufs);
        return;
    }
    if (std::getenv("TCBENCH_ONLY_TC_ROWCOL16") != nullptr) {
        const bool with_amax_rowcol = std::getenv("TCBENCH_USE_AMAX") != nullptr;
        const bool rte_scale = std::getenv("TCBENCH_TE_COPY16_RTE_SCALE") != nullptr;
        const bool row_rht = std::getenv("TCBENCH_TC_ROW_RHT") != nullptr;
        const bool raw_col = std::getenv("TCBENCH_TC_RAW_COL") != nullptr;
        const bool persist_requested = std::getenv("TCBENCH_TC_ROWCOL_PERSIST") != nullptr;
        const bool persist = persist_requested && !row_rht && !raw_col;
        const bool producer = std::getenv("TCBENCH_TC_ROWCOL_PRODUCER") != nullptr;
        const bool pair_smem = std::getenv("TCBENCH_TC_ROWCOL_PAIR_SMEM") != nullptr;
        const bool bf16x8_default = persist && pair_smem && std::getenv("TCBENCH_TC_ROWCOL_NO_BF16X8_QUANT") == nullptr;
        const bool bf16x8_all = bf16x8_default || std::getenv("TCBENCH_TC_ROWCOL_BF16X8_QUANT") != nullptr;
        const bool row_bf16x8_quant = bf16x8_all || std::getenv("TCBENCH_TC_ROWCOL_ROW_BF16X8_QUANT") != nullptr;
        const bool col_bf16x8_quant = bf16x8_all || std::getenv("TCBENCH_TC_ROWCOL_COL_BF16X8_QUANT") != nullptr;
        if (persist_requested && row_rht) {
            std::cout << "note: persistent row-RHT+col-RHT is not implemented; using non-persistent experimental path\n";
        }
        if (persist_requested && raw_col) {
            std::cout << "note: persistent rowcol raw-col path is not implemented; using non-persistent experimental path\n";
        }
        float tc_rowcol = -1.0f;
        if (producer && row_rht) {
            std::cout << "note: producer rowcol path matches TE raw-row-quant+col-RHT only; ignoring TCBENCH_TC_ROW_RHT\n";
        }
        if (producer && with_amax_rowcol) {
            tc_rowcol = rte_scale
                ? run_te_copy16_col_quant_persistent_producer<false, true, true>(bufs, rows, cols)
                : run_te_copy16_col_quant_persistent_producer<true, true, true>(bufs, rows, cols);
        } else if (producer) {
            tc_rowcol = rte_scale
                ? run_te_copy16_col_quant_persistent_producer<false, false, true>(bufs, rows, cols)
                : run_te_copy16_col_quant_persistent_producer<true, false, true>(bufs, rows, cols);
        } else if (persist && pair_smem && with_amax_rowcol) {
            if (row_bf16x8_quant && col_bf16x8_quant) {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, true, true, true, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, true, true, true, true>(bufs, rows, cols);
            } else if (row_bf16x8_quant) {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, true, true, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, true, true, true>(bufs, rows, cols);
            } else if (col_bf16x8_quant) {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, true, true, false, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, true, true, false, true>(bufs, rows, cols);
            } else {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, true, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, true, true>(bufs, rows, cols);
            }
        } else if (persist && pair_smem) {
            if (row_bf16x8_quant && col_bf16x8_quant) {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, false, true, true, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, false, true, true, true>(bufs, rows, cols);
            } else if (row_bf16x8_quant) {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, false, true, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, false, true, true>(bufs, rows, cols);
            } else if (col_bf16x8_quant) {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, false, true, false, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, false, true, false, true>(bufs, rows, cols);
            } else {
                tc_rowcol = rte_scale
                    ? run_te_copy16_rowcol_quant_persistent<false, false, true>(bufs, rows, cols)
                    : run_te_copy16_rowcol_quant_persistent<true, false, true>(bufs, rows, cols);
            }
        } else if (persist && with_amax_rowcol) {
            tc_rowcol = rte_scale
                ? run_te_copy16_rowcol_quant_persistent<false, true>(bufs, rows, cols)
                : run_te_copy16_rowcol_quant_persistent<true, true>(bufs, rows, cols);
        } else if (persist) {
            tc_rowcol = rte_scale
                ? run_te_copy16_rowcol_quant_persistent<false, false>(bufs, rows, cols)
                : run_te_copy16_rowcol_quant_persistent<true, false>(bufs, rows, cols);
        } else if (with_amax_rowcol) {
            if (row_rht) {
                tc_rowcol = raw_col
                    ? (rte_scale
                        ? run_te_copy16_rowcol_quant<false, true, true, false>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, true, true, false>(bufs, rows, cols))
                    : (rte_scale
                        ? run_te_copy16_rowcol_quant<false, true, true, true>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, true, true, true>(bufs, rows, cols));
            } else {
                tc_rowcol = raw_col
                    ? (rte_scale
                        ? run_te_copy16_rowcol_quant<false, true, false, false>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, true, false, false>(bufs, rows, cols))
                    : (rte_scale
                        ? run_te_copy16_rowcol_quant<false, true, false, true>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, true, false, true>(bufs, rows, cols));
            }
        } else {
            if (row_rht) {
                tc_rowcol = raw_col
                    ? (rte_scale
                        ? run_te_copy16_rowcol_quant<false, false, true, false>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, false, true, false>(bufs, rows, cols))
                    : (rte_scale
                        ? run_te_copy16_rowcol_quant<false, false, true, true>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, false, true, true>(bufs, rows, cols));
            } else {
                tc_rowcol = raw_col
                    ? (rte_scale
                        ? run_te_copy16_rowcol_quant<false, false, false, false>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, false, false, false>(bufs, rows, cols))
                    : (rte_scale
                        ? run_te_copy16_rowcol_quant<false, false, false, true>(bufs, rows, cols)
                        : run_te_copy16_rowcol_quant<true, false, false, true>(bufs, rows, cols));
            }
        }
        std::cout << "mode: fused_rowcol16" << (with_amax_rowcol ? "_with_amax" : "_no_amax") << "\n";
        std::cout << "te-copy16 "
                  << (producer ? "producer " : "")
                  << (persist ? "persist " : "")
                  << (pair_smem ? "pair-smem " : "")
                  << ((row_rht && raw_col && !producer) ? "experimental row-RHT+raw-col-quant" :
                      ((row_rht && !producer) ? "experimental row-RHT+col-RHT" :
                       (raw_col ? "raw-row-quant+raw-col-quant" : "TE-equivalent raw-row-quant+col-RHT-quant")))
                  << " tensorcore "
                  << (rte_scale ? "rte-scale" : "te-rp-scale")
                  << ": " << tc_rowcol << " ms\n";
        free_device_buffers(bufs);
        return;
    }
    if (std::getenv("TCBENCH_ONLY_PIPE64") != nullptr) {
        if (const char* variant = std::getenv("TCBENCH_PIPE64_VARIANT")) {
            bool data_sr = false;
            bool scale_sr = false;
            if (std::strcmp(variant, "data_sr") == 0) {
                data_sr = true;
            } else if (std::strcmp(variant, "scale_sr") == 0) {
                scale_sr = true;
            } else if (std::strcmp(variant, "sr") == 0) {
                data_sr = true;
                scale_sr = true;
            } else if (std::strcmp(variant, "plain") != 0) {
                std::cerr << "unknown TCBENCH_PIPE64_VARIANT=" << variant << "\n";
                free_device_buffers(bufs);
                return;
            }
            const float ms = run_te_style_direct32_quant_pipe64(bufs, rows, cols, data_sr, scale_sr, with_random_sign_mask, with_amax);
            std::cout << "mode: " << (with_amax ? "with_amax" : "no_amax") << "\n";
            std::cout << "te-style direct32 pipe64 " << variant << ": " << ms << " ms\n";
            free_device_buffers(bufs);
            return;
        }
        const float direct_quant_pipe64 = run_te_style_direct32_quant_pipe64(bufs, rows, cols, false, false, with_random_sign_mask, with_amax);
        const float direct_quant_sr_pipe64 = skip_sr ? -1.0f : run_te_style_direct32_quant_pipe64(bufs, rows, cols, true, true, with_random_sign_mask, with_amax);
        std::cout << "mode: " << (with_amax ? "with_amax" : "no_amax") << "\n";
        std::cout << "te-style direct32 pipe64 had+quant:    " << direct_quant_pipe64 << " ms\n";
        if (direct_quant_sr_pipe64 >= 0.0f) {
            std::cout << "te-style direct32 pipe64 had+quant+sr: " << direct_quant_sr_pipe64 << " ms\n";
        }
        free_device_buffers(bufs);
        return;
    }
    if (std::getenv("TCBENCH_ONLY_WIDE") != nullptr) {
        const bool use_reg_epi = std::getenv("TCBENCH_USE_REG_EPI") != nullptr;
        const bool use_pipe64 = std::getenv("TCBENCH_USE_PIPE64") != nullptr;
        const float direct_quant_wide = run_te_style_direct32_quant_wide(bufs, rows, cols, false, false, with_random_sign_mask, with_amax);
        const float direct_quant_sr_wide = skip_sr ? -1.0f : run_te_style_direct32_quant_wide(bufs, rows, cols, true, true, with_random_sign_mask, with_amax);
        const float direct_quant_pipe64 = use_pipe64 ? run_te_style_direct32_quant_pipe64(bufs, rows, cols, false, false, with_random_sign_mask, with_amax) : -1.0f;
        const float direct_quant_sr_pipe64 = (use_pipe64 && !skip_sr) ? run_te_style_direct32_quant_pipe64(bufs, rows, cols, true, true, with_random_sign_mask, with_amax) : -1.0f;
        const float direct_quant_wide_reg = use_reg_epi ? run_te_style_direct32_quant_wide_reg(bufs, rows, cols, false, false, with_random_sign_mask, with_amax) : -1.0f;
        std::cout << "mode: " << (with_amax ? "with_amax" : "no_amax") << "\n";
        std::cout << "te-style direct32x512 had+quant:       " << direct_quant_wide << " ms\n";
        if (direct_quant_sr_wide >= 0.0f) {
            std::cout << "te-style direct32x512 had+quant+sr:    " << direct_quant_sr_wide << " ms\n";
        }
        if (direct_quant_pipe64 >= 0.0f) {
            std::cout << "te-style direct32 pipe64 had+quant:    " << direct_quant_pipe64 << " ms\n";
            if (direct_quant_sr_pipe64 >= 0.0f) {
                std::cout << "te-style direct32 pipe64 had+quant+sr: " << direct_quant_sr_pipe64 << " ms\n";
            }
        }
        if (direct_quant_wide_reg >= 0.0f) {
            std::cout << "te-style direct32x512 reg-epi had+quant:" << direct_quant_wide_reg << " ms\n";
        }
        free_device_buffers(bufs);
        return;
    }
    const float direct_quant_nosign = run_te_style_direct32_quant(bufs, rows, cols, false, false, false, with_amax);
    const float direct_quant = run_te_style_direct32_quant(bufs, rows, cols, false, false, with_random_sign_mask, with_amax);
    const float direct_quant_sr = skip_sr ? -1.0f : run_te_style_direct32_quant(bufs, rows, cols, true, true, with_random_sign_mask, with_amax);
    const float direct_quant_wide = run_te_style_direct32_quant_wide(bufs, rows, cols, false, false, with_random_sign_mask, with_amax);
    const float direct_quant_sr_wide = skip_sr ? -1.0f : run_te_style_direct32_quant_wide(bufs, rows, cols, true, true, with_random_sign_mask, with_amax);
    const bool use_reg_epi = std::getenv("TCBENCH_USE_REG_EPI") != nullptr;
    const float direct_quant_wide_reg = use_reg_epi ? run_te_style_direct32_quant_wide_reg(bufs, rows, cols, false, false, with_random_sign_mask, with_amax) : -1.0f;
    const float direct_quant_sr_wide_reg = (use_reg_epi && !skip_sr) ? run_te_style_direct32_quant_wide_reg(bufs, rows, cols, true, true, with_random_sign_mask, with_amax) : -1.0f;
    const bool use_pipe64 = std::getenv("TCBENCH_USE_PIPE64") != nullptr;
    const float direct_quant_pipe64 = use_pipe64 ? run_te_style_direct32_quant_pipe64(bufs, rows, cols, false, false, with_random_sign_mask, with_amax) : -1.0f;
    const float direct_quant_sr_pipe64 = (use_pipe64 && !skip_sr) ? run_te_style_direct32_quant_pipe64(bufs, rows, cols, true, true, with_random_sign_mask, with_amax) : -1.0f;
    const bool use_persist = std::getenv("TCBENCH_USE_PERSIST") != nullptr;
    const float direct_quant_persist = use_persist ? run_te_style_direct32_quant_persist4(bufs, rows, cols, false, false, with_random_sign_mask) : -1.0f;
    const float direct_quant_sr_persist = (use_persist && !skip_sr) ? run_te_style_direct32_quant_persist4(bufs, rows, cols, true, true, with_random_sign_mask) : -1.0f;
    std::cout << "mode: " << (with_amax ? "with_amax" : "no_amax") << "\n";
    std::cout << "te-style direct32 had+quant (no sign): " << direct_quant_nosign << " ms\n";
    std::cout << "te-style direct32 had+quant:           " << direct_quant << " ms\n";
    if (direct_quant_sr >= 0.0f) {
        std::cout << "te-style direct32 had+quant+sr:        " << direct_quant_sr << " ms\n";
    }
    if (direct_quant_wide >= 0.0f) {
        std::cout << "te-style direct32x512 had+quant:       " << direct_quant_wide << " ms\n";
        if (direct_quant_sr_wide >= 0.0f) {
            std::cout << "te-style direct32x512 had+quant+sr:    " << direct_quant_sr_wide << " ms\n";
        }
    }
    if (direct_quant_wide_reg >= 0.0f) {
        std::cout << "te-style direct32x512 reg-epi had+quant:" << direct_quant_wide_reg << " ms\n";
        if (direct_quant_sr_wide_reg >= 0.0f) {
            std::cout << "te-style direct32x512 reg-epi had+quant+sr:" << direct_quant_sr_wide_reg << " ms\n";
        }
    }
    if (direct_quant_pipe64 >= 0.0f) {
        std::cout << "te-style direct32 pipe64 had+quant:    " << direct_quant_pipe64 << " ms\n";
        if (direct_quant_sr_pipe64 >= 0.0f) {
            std::cout << "te-style direct32 pipe64 had+quant+sr: " << direct_quant_sr_pipe64 << " ms\n";
        }
    }
    if (direct_quant_persist >= 0.0f) {
        std::cout << "te-style direct32 persist4 had+quant:  " << direct_quant_persist << " ms\n";
        if (direct_quant_sr_persist >= 0.0f) {
            std::cout << "te-style direct32 persist4 had+quant+sr:" << direct_quant_sr_persist << " ms\n";
        }
    }
    if (std::getenv("TCBENCH_COMPARE_REG_EPI") != nullptr) {
        compare_wide_reg_epilogue(bufs, rows, cols, with_amax);
    }
    free_device_buffers(bufs);
}

}  // namespace te_style_tcbench

#ifndef TE_COPY16_NO_MAIN
int main() {
    if (const char* rows_env = std::getenv("TCBENCH_ROWS")) {
        const char* cols_env = std::getenv("TCBENCH_COLS");
        if (cols_env == nullptr) {
            std::cerr << "TCBENCH_COLS must be set with TCBENCH_ROWS\n";
            return 1;
        }
        te_style_tcbench::run_case(std::atoi(rows_env), std::atoi(cols_env));
        return 0;
    }
    if (std::getenv("TCBENCH_SMALL") != nullptr) {
        te_style_tcbench::run_case(128, 128);
        return 0;
    }
    te_style_tcbench::run_case(2048, 65536);
    te_style_tcbench::run_case(3072, 65536);
    te_style_tcbench::run_case(8192, 2048);
    return 0;
}
#endif
